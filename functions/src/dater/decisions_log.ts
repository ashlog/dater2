import fs from 'fs';
import path from 'path';
import { getQuestionById } from './questions';

export type DecisionEntry = {
  timestamp: string;
  location: { latitude: number; longitude: number };
  userId: string;
  ratingToken: string;
  decision: 'like' | 'skip';
  decisionSource?: 'image' | 'text';
  comment?: string;
  photoUsed?: string | null;
  images: string[];
  prompts: { question: string; answer: string }[];
  profile?: { firstName: string; age?: number };
  imageScores?: { url: string; score: number }[];
  medianImageScore?: number;
  openerPromptHash?: string;
  model?: string;
  deliveryStatus?: 'success' | 'error';
  deliveryError?: { code?: string; message: string };
};

function resolveDecisionsPath(): string {
  const candidates = [
    path.join(process.cwd(), 'src', 'profiles_decisions.jsonl'),
    path.join(process.cwd(), 'profiles_decisions.jsonl'),
    path.join(__dirname, '..', 'profiles_decisions.jsonl'),
  ];
  for (const p of candidates) {
    try {
      const dir = path.dirname(p);
      if (fs.existsSync(dir)) return p;
    } catch {}
  }
  return path.join(__dirname, 'profiles_decisions.jsonl');
}

const decisionsPath = resolveDecisionsPath();
let decisionsWriteQueue: Promise<void> = Promise.resolve();

function resolveOpenerPromptMapPath(): string {
  const dir = path.dirname(decisionsPath);
  return path.join(dir, 'opener_prompts_map.json');
}

const openerPromptMapPath = resolveOpenerPromptMapPath();
let openerPromptMapCache: Record<string, string> | null = null;

async function loadOpenerPromptMap(): Promise<Record<string, string>> {
  if (openerPromptMapCache) return openerPromptMapCache;
  try {
    const txt = await fs.promises.readFile(openerPromptMapPath, 'utf8');
    const parsed = JSON.parse(txt);
    openerPromptMapCache = parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    openerPromptMapCache = {};
  }
  return openerPromptMapCache!;
}

async function persistOpenerPromptMap(map: Record<string, string>): Promise<void> {
  try {
    await fs.promises.writeFile(openerPromptMapPath, JSON.stringify(map, null, 2));
  } catch (e) {
    console.error('Failed to persist opener_prompt map', e);
  }
}

async function ensureOpenerPromptStored(hash: string, text: string): Promise<void> {
  const map = await loadOpenerPromptMap();
  if (map[hash]) return;
  map[hash] = text;
  await persistOpenerPromptMap(map);
}

export async function appendDecision(
  entry: DecisionEntry,
  options?: { openerPrompt?: { hash: string; text: string } }
): Promise<void> {
  decisionsWriteQueue = decisionsWriteQueue.then(async () => {
    try {
      const dir = path.dirname(decisionsPath);
      await fs.promises.mkdir(dir, { recursive: true });
      await fs.promises.appendFile(decisionsPath, JSON.stringify(entry) + '\n', 'utf8');
      if (options?.openerPrompt) {
        await ensureOpenerPromptStored(options.openerPrompt.hash, options.openerPrompt.text);
      }
    } catch (e) {
      console.error('Failed to append decision', e);
    }
  });
  return decisionsWriteQueue;
}

export function buildPromptsList(filteredAnswers: { questionId: string; response: string }[]): { question: string; answer: string }[] {
  return filteredAnswers.map(a => ({
    question: getQuestionById(a.questionId),
    answer: a.response,
  }));
}

function resolveDryRunPath(): string {
  const candidates = [
    path.join(process.cwd(), 'src', 'dry_run_profiles.jsonl'),
    path.join(process.cwd(), 'dry_run_profiles.jsonl'),
    path.join(__dirname, '..', 'dry_run_profiles.jsonl'),
  ];
  for (const p of candidates) {
    try {
      const dir = path.dirname(p);
      if (fs.existsSync(dir)) return p;
    } catch {}
  }
  return path.join(__dirname, 'dry_run_profiles.jsonl');
}

const dryRunPath = resolveDryRunPath();
let dryRunWriteQueue: Promise<void> = Promise.resolve();

export async function appendDryRunDecision(
  entry: DecisionEntry
): Promise<void> {
  dryRunWriteQueue = dryRunWriteQueue.then(async () => {
    try {
      const dir = path.dirname(dryRunPath);
      await fs.promises.mkdir(dir, { recursive: true });
      await fs.promises.appendFile(dryRunPath, JSON.stringify(entry) + '\n', 'utf8');
    } catch (e) {
      console.error('Failed to append dry run decision', e);
    }
  });
  return dryRunWriteQueue;
}
