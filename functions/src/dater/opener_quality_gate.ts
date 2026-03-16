/**
 * Embedding-based quality filter for Hinge openers.
 *
 * Uses sentence-transformers (all-MiniLM-L6-v2) via a persistent Python
 * HTTP server to detect templated/formulaic openers and enforce diversity.
 *
 * The Python server (embed_checker.py) keeps the model loaded in memory,
 * so after the initial ~3s cold start, each request takes ~20-50ms.
 *
 * Designed to integrate into runner.ts after opener generation and before
 * the bestPl ranker call. Server lifecycle is managed via
 * ensureEmbedServer() / stopEmbedServer(), matching the siglip pattern.
 */

import axios from 'axios';
import { execFile, spawn } from 'child_process';
import path from 'path';

// ── Types ──────────────────────────────────────────────────────────────

export interface QualityResult {
  /** Max cosine similarity to known bad template embeddings */
  templateSimilarity: number;
  /** Whether the opener contains first-person pronouns (I, my, I'm, etc.) */
  hasFirstPerson: boolean;
  /** Whether the opener uses "just" as a connector (formulaic pattern) */
  hasToxicWord: boolean;
  /** Cosine distance of last 4 words from generic closer embeddings (higher = more specific = better) */
  closerSpecificity: number;
  /** Overall quality gate pass/fail */
  pass: boolean;
  /** Why it failed, if it did */
  reason: string;
}

export interface BatchQualityResult extends QualityResult {
  /** Max pairwise similarity to already-accepted openers (for diversity) */
  pairwiseSimilarity: number;
}

interface EmbedCheckerInput {
  openers: string[];
  templates: string[];
  generic_closers: string[];
  accepted: string[];
}

interface EmbedCheckerOutput {
  results: {
    opener: string;
    template_similarity: number;
    closer_specificity: number;
    pairwise_max_similarity: number;
  }[];
}

// ── Bad template library ───────────────────────────────────────────────
// These are embedded once per call and compared against candidate openers.
// Include concrete formulations, not just abstract patterns — embedding
// models need semantic content, not variable placeholders.

const BAD_TEMPLATES = [
  // "X is just Y with Z" formula and variants
  'hiking is just walking with pretension',
  'running is just walking with anxiety',
  'baking is just science with snacks',
  'cooking is just chemistry with deadlines',
  'yoga is just stretching with branding',
  'camping is just homelessness with extra steps',
  'this is just competitive shopping',
  'dating is just interviews with wine',
  'brunch is just breakfast with commitment issues',

  // "that's not X that's Y" formula
  "that's not a hobby that's a lifestyle",
  "that's not cooking that's an art form",
  "that's not a pet that's a roommate",

  // "who hurt you" / "what did you do"
  'who hurt you',
  'who hurt you and your taste in music',
  'what did you do to deserve this',
  'what did you do to end up here',

  // "coffee or boba" era
  'coffee or boba',
  'coffee or boba which one defines your personality',

  // Generic conversion/metric formulas
  "what's the conversion rate",
  'how many reps is that',
  'bribery with extra steps',
  "what's your fastest checkout time",

  // Generic closer formulas
  "that's a bold claim we should test it over coffee",
  'we should settle this over drinks',
  "I feel like there's a story here",
];

// Generic closer phrases — openers whose last 4 words are too similar
// to these are using weak, formulaic closers.
const GENERIC_CLOSERS = [
  'what do you think',
  'right or wrong',
  'yes or no',
  'agree or disagree',
  'coffee or drinks sometime',
  'we should test this',
  'over coffee sometime',
  'settle this over drinks',
  'there is a story here',
  'tell me everything',
  'I need to know',
];

// ── Thresholds ─────────────────────────────────────────────────────────

const TEMPLATE_SIMILARITY_THRESHOLD = 0.4;
const PAIRWISE_DIVERSITY_THRESHOLD = 0.7;

// ── Regex checks (fast, no Python needed) ──────────────────────────────

const FIRST_PERSON_RE = /\b(i|my|i'm|i'd|i'll|i've|i was|i have|i can)\b/i;

/** Detects "just" used as a connector/minimizer (not "just now", "just a sec") */
function hasJustAsConnector(text: string): boolean {
  // Match "is just", "are just", "'s just" — the formulaic connector pattern
  return /\b(is|are|was|were|'s)\s+just\b/i.test(text);
}

// ── Embed server lifecycle ─────────────────────────────────────────────

const EMBED_SCRIPT = path.join(__dirname, 'embed_checker.py');
const PYTHON_BIN = 'python3';
const EMBED_SERVER_PORT = 5201;
const EMBED_SERVER_URL = `http://127.0.0.1:${EMBED_SERVER_PORT}`;

let embedServerPid: number | null = null;

/**
 * Ensure the embed checker server is running. Starts it if needed.
 * Call this once at the start of a run (e.g. alongside ensureSiglipServer).
 */
export async function ensureEmbedServer(): Promise<void> {
  try {
    await axios.get(`${EMBED_SERVER_URL}/healthcheck`, { timeout: 3000 });
    return; // Already running
  } catch {
    // Not running — start it
  }

  console.log('Embed checker server not running, starting it...');
  const child = spawn(PYTHON_BIN, [EMBED_SCRIPT], {
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let startupError = '';
  child.stderr?.on('data', (data: Buffer) => { startupError += data.toString(); });
  child.unref();
  embedServerPid = child.pid ?? null;

  // Poll until ready (model load takes ~2-3s)
  for (let i = 0; i < 30; i++) {
    await new Promise(r => setTimeout(r, 500));
    try {
      await axios.get(`${EMBED_SERVER_URL}/healthcheck`, { timeout: 2000 });
      console.log('Embed checker server started successfully');
      return;
    } catch {
      // still starting
    }
  }
  throw new Error(`Embed checker server failed to start within 15s${startupError ? '\n' + startupError : ''}`);
}

/** Stop the embed checker server. Call in a finally block after a run. */
export function stopEmbedServer(): void {
  if (embedServerPid) {
    try {
      process.kill(embedServerPid, 'SIGTERM');
      console.log(`Embed checker server (pid ${embedServerPid}) stopped`);
    } catch {
      // already dead
    }
    embedServerPid = null;
  }
}

// ── HTTP client to embed server ────────────────────────────────────────

async function callEmbedChecker(input: EmbedCheckerInput): Promise<EmbedCheckerOutput> {
  try {
    const resp = await axios.post(`${EMBED_SERVER_URL}/check`, input, { timeout: 10_000 });
    return resp.data;
  } catch (httpError) {
    // Fallback: try one-shot subprocess if server is down
    console.warn('Embed server unavailable, falling back to subprocess');
    return callEmbedCheckerSubprocess(input);
  }
}

function callEmbedCheckerSubprocess(input: EmbedCheckerInput): Promise<EmbedCheckerOutput> {
  return new Promise((resolve, reject) => {
    const child = execFile(
      PYTHON_BIN,
      [EMBED_SCRIPT, '--stdin'],
      { maxBuffer: 10 * 1024 * 1024, timeout: 30_000 },
      (error, stdout, stderr) => {
        if (error) {
          reject(new Error(`embed_checker failed: ${error.message}${stderr ? '\n' + stderr : ''}`));
          return;
        }
        try {
          resolve(JSON.parse(stdout));
        } catch (e) {
          reject(new Error(`embed_checker returned invalid JSON: ${stdout.slice(0, 200)}`));
        }
      }
    );
    child.stdin?.write(JSON.stringify(input));
    child.stdin?.end();
  });
}

// ── Public API ─────────────────────────────────────────────────────────

/**
 * Check a single opener against the quality gate.
 *
 * Returns pass/fail with detailed scores and rejection reason.
 * With server running: ~20-50ms. Subprocess fallback: ~3s.
 */
export async function checkOpenerQuality(opener: string): Promise<QualityResult> {
  const results = await checkOpenerQualityBatch([opener]);
  return results[0];
}

/**
 * Check a batch of openers against the quality gate.
 *
 * More efficient than calling checkOpenerQuality in a loop because
 * it makes a single call for the whole batch.
 *
 * Does NOT check pairwise diversity within the batch — use
 * filterOpenersBatch() for that.
 */
export async function checkOpenerQualityBatch(openers: string[]): Promise<QualityResult[]> {
  const output = await callEmbedChecker({
    openers,
    templates: BAD_TEMPLATES,
    generic_closers: GENERIC_CLOSERS,
    accepted: [],
  });

  return openers.map((opener, i) => {
    const r = output.results[i];
    const hasFirstPerson = FIRST_PERSON_RE.test(opener);
    const hasToxicWord = hasJustAsConnector(opener);

    const reasons: string[] = [];
    if (r.template_similarity > TEMPLATE_SIMILARITY_THRESHOLD) {
      reasons.push(`template_sim=${r.template_similarity.toFixed(2)} > ${TEMPLATE_SIMILARITY_THRESHOLD}`);
    }
    if (hasToxicWord) {
      reasons.push(`contains "just" as connector`);
    }

    return {
      templateSimilarity: r.template_similarity,
      hasFirstPerson,
      hasToxicWord,
      closerSpecificity: r.closer_specificity,
      pass: reasons.length === 0,
      reason: reasons.join('; ') || 'ok',
    };
  });
}

/**
 * Filter a batch of openers, enforcing both individual quality AND
 * pairwise diversity (no two accepted openers > 0.7 similar).
 *
 * Returns openers that pass all checks, in original order.
 * This is the main entry point for the production pipeline.
 */
export async function filterOpenersBatch(
  openers: string[],
  alreadyAccepted: string[] = [],
): Promise<{ accepted: string[]; rejected: { opener: string; reason: string }[] }> {
  const output = await callEmbedChecker({
    openers,
    templates: BAD_TEMPLATES,
    generic_closers: GENERIC_CLOSERS,
    accepted: alreadyAccepted,
  });

  const acceptedResults: string[] = [];
  const rejected: { opener: string; reason: string }[] = [];

  for (let i = 0; i < openers.length; i++) {
    const opener = openers[i];
    const r = output.results[i];
    const hasToxicWord = hasJustAsConnector(opener);

    const reasons: string[] = [];

    if (r.template_similarity > TEMPLATE_SIMILARITY_THRESHOLD) {
      reasons.push(`template_sim=${r.template_similarity.toFixed(2)} > ${TEMPLATE_SIMILARITY_THRESHOLD}`);
    }
    if (hasToxicWord) {
      reasons.push(`contains "just" as connector`);
    }
    if (r.pairwise_max_similarity > PAIRWISE_DIVERSITY_THRESHOLD) {
      reasons.push(`too similar to accepted opener (sim=${r.pairwise_max_similarity.toFixed(2)} > ${PAIRWISE_DIVERSITY_THRESHOLD})`);
    }

    if (reasons.length > 0) {
      rejected.push({ opener, reason: reasons.join('; ') });
    } else {
      acceptedResults.push(opener);
    }
  }

  return { accepted: acceptedResults, rejected };
}
