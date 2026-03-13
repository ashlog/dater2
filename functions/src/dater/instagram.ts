/**
 * Instagram mode: Score follower photos with SigLIP, generate openers for top profiles.
 * Uses cached classification data from browser-based scraping pipeline.
 */
import axios from 'axios';
import fs from 'fs';
import path from 'path';
import os from 'os';
import pLimit from 'p-limit';
import {
  generateImageOpenerFromYaml,
  setLLMProvider,
  getCostSummary,
  resetCostTracker,
  type LLMProviderName,
} from './llm';
import { cleanPickupLineResponse, isBadResponse } from './helpers';

const llmProvider: LLMProviderName = 'claude-direct';
const model = 'claude-opus-4-6';

setLLMProvider(llmProvider);

export interface IgPost {
  code: string;
  username: string;
  score: number;
  link: string;
  full_name?: string;
}

export interface IgCandidate {
  username: string;
  displayName: string;
  bio: string;
  post: IgPost;
  imageUrl: string;
  opener: string;
}

export function loadClassificationCache(cachePath: string): IgPost[] {
  const raw = fs.readFileSync(cachePath, 'utf8');
  return JSON.parse(raw);
}

export function getTopPostsPerUser(posts: IgPost[], topN: number): IgPost[] {
  const byUser = new Map<string, IgPost>();
  for (const post of posts) {
    const existing = byUser.get(post.username);
    if (!existing || post.score > existing.score) {
      byUser.set(post.username, post);
    }
  }
  return [...byUser.values()]
    .sort((a, b) => b.score - a.score)
    .slice(0, topN);
}

/** Follow the Instagram embed redirect and download the image as a data URI.
 *  Anthropic blocks Instagram CDN URLs via robots.txt, so we must send base64. */
export async function resolveImageAsDataUri(code: string): Promise<{ dataUri: string; cdnUrl: string } | null> {
  try {
    // Step 1: get CDN URL from redirect
    const r = await axios.get(`https://www.instagram.com/p/${code}/media/?size=l`, {
      maxRedirects: 0,
      validateStatus: () => true,
      timeout: 10000,
    });
    let cdnUrl: string | undefined;
    if (r.status === 301 || r.status === 302) {
      cdnUrl = r.headers.location;
    }
    if (!cdnUrl) return null;

    // Step 2: download image bytes
    const imgResp = await axios.get(cdnUrl, {
      responseType: 'arraybuffer',
      timeout: 15000,
    });
    const buf = Buffer.from(imgResp.data);
    const contentType = imgResp.headers['content-type'] || 'image/jpeg';
    const dataUri = `data:${contentType};base64,${buf.toString('base64')}`;
    return { dataUri, cdnUrl };
  } catch {
    return null;
  }
}

/** Fetch post caption from Instagram's public embed page. */
export async function fetchPostCaption(code: string): Promise<string> {
  try {
    const r = await axios.get(`https://www.instagram.com/p/${code}/embed/`, {
      headers: { 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36' },
      timeout: 10000,
    });
    const html = r.data as string;
    // Caption is in escaped JSON: \"caption\":\"...\",\"next_key\"
    const captionMatch = html.match(/\\"caption\\":\\"((?:[^\\]|\\[^"])*)\\",\\"/);
    if (captionMatch?.[1]) {
      return captionMatch[1]
        .replace(/\\n/g, ' ')
        .replace(/\\u[\da-fA-F]{4}/g, m => {
          try { return String.fromCodePoint(parseInt(m.slice(2), 16)); } catch { return m; }
        })
        .replace(/\\(.)/g, '$1')
        .replace(/\s{2,}/g, ' ')
        .trim()
        .substring(0, 500);
    }
    return '';
  } catch {
    return '';
  }
}

export async function fetchIgProfile(username: string): Promise<{ name: string; bio: string }> {
  try {
    const r = await axios.get(`https://www.instagram.com/${username}/`, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
      },
      timeout: 10000,
    });
    const html = r.data as string;
    const titleMatch = html.match(/<meta property="og:title" content="([^"]+)"/);
    const descMatch = html.match(/<meta property="og:description" content="([^"]+)"/);

    let name = username;
    if (titleMatch?.[1]) {
      const raw = titleMatch[1];
      const parenIdx = raw.indexOf('(');
      name = parenIdx > 0 ? raw.substring(0, parenIdx).trim() : raw.split('•')[0].trim();
    }

    let bio = '';
    if (descMatch?.[1]) {
      bio = descMatch[1].replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>');
    }

    return { name, bio };
  } catch {
    return { name: username, bio: '' };
  }
}

export async function runInstagram(options: {
  cachePath?: string;
  topN?: number;
  outputPath?: string;
} = {}): Promise<IgCandidate[]> {
  const cachePath = options.cachePath || path.join(os.homedir(), 'projects/siglip-test/ig_classify_cache_full.json');
  const topN = options.topN || 10;
  const outputPath = options.outputPath || path.join(os.homedir(), 'projects/siglip-test/ig_openers.html');

  resetCostTracker();

  // 1. Load scored posts
  console.log('Loading classification cache...');
  const allPosts = loadClassificationCache(cachePath);
  console.log(`Loaded ${allPosts.length} scored posts`);

  // 2. Get top N unique users
  const topPosts = getTopPostsPerUser(allPosts, topN);
  console.log(`\nTop ${topPosts.length} users by score:`);
  for (const p of topPosts) {
    console.log(`  ${p.score.toFixed(6)} @${p.username} ${p.link}`);
  }

  // 3. Resolve image URLs, fetch profiles, and generate openers
  const candidates: IgCandidate[] = [];
  const limit = pLimit(2);

  await Promise.all(topPosts.map(post => limit(async () => {
    console.log(`\nProcessing @${post.username}...`);

    const image = await resolveImageAsDataUri(post.code);
    if (!image) {
      console.log(`  Failed to resolve image for ${post.code}`);
      return;
    }
    console.log(`  Image downloaded (${(image.dataUri.length / 1024).toFixed(0)}KB base64)`);

    const profile = await fetchIgProfile(post.username);
    console.log(`  Profile: ${profile.name}`);
    if (profile.bio) console.log(`  Bio: ${profile.bio.substring(0, 80)}...`);

    try {
      const rawOpener = await generateImageOpenerFromYaml(
        image.dataUri,
        '',
        profile.bio,
        profile.name,
        model
      );
      const opener = cleanPickupLineResponse(rawOpener);
      if (isBadResponse(opener)) {
        console.log(`  Bad response, skipping`);
        return;
      }
      console.log(`  Opener: "${opener}"`);

      candidates.push({
        username: post.username,
        displayName: profile.name,
        bio: profile.bio,
        post,
        imageUrl: image.cdnUrl,
        opener,
      });
    } catch (e) {
      console.error(`  Error generating opener for @${post.username}:`, e);
    }
  })));

  candidates.sort((a, b) => b.post.score - a.post.score);

  // 4. Generate HTML
  const html = generateHtml(candidates);
  fs.writeFileSync(outputPath, html);
  console.log(`\nResults saved to ${outputPath}`);
  console.log(`Cost: ${JSON.stringify(getCostSummary())}`);

  return candidates;
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function generateHtml(candidates: IgCandidate[]): string {
  const rows = candidates.map((c, i) => `
    <div class="card">
      <div class="rank">#${i + 1}</div>
      <div class="photo">
        <a href="${c.post.link}" target="_blank">
          <img src="${c.imageUrl}" loading="lazy" onerror="this.src='';this.alt='expired'">
        </a>
      </div>
      <div class="info">
        <div class="username">
          <a href="https://www.instagram.com/${c.username}/" target="_blank">@${c.username}</a>
          <span class="name">${c.displayName !== c.username ? escapeHtml(c.displayName) : ''}</span>
        </div>
        <div class="score">Score: ${c.post.score.toFixed(4)}</div>
        <div class="bio">${escapeHtml(c.bio)}</div>
        <div class="opener">
          <div class="opener-label">Generated DM:</div>
          <div class="opener-text">"${escapeHtml(c.opener)}"</div>
          <button onclick="navigator.clipboard.writeText(this.dataset.text).then(()=>{this.textContent='Copied!';setTimeout(()=>this.textContent='Copy',1500)})" data-text="${escapeHtml(c.opener)}" class="copy-btn">Copy</button>
        </div>
      </div>
    </div>`).join('\n');

  return `<!DOCTYPE html>
<html><head><title>Instagram Openers - Dater2</title>
<style>
* { box-sizing: border-box; }
body { font-family: -apple-system, sans-serif; background: #0a0a0a; color: #eee; padding: 20px; max-width: 900px; margin: 0 auto; }
h1 { color: #fff; margin-bottom: 8px; }
.subtitle { color: #888; margin-bottom: 24px; }
.card { display: flex; gap: 16px; padding: 16px; margin-bottom: 16px; background: #1a1a2e; border-radius: 12px; border: 1px solid #333; }
.rank { font-size: 24px; font-weight: bold; color: #58a6ff; min-width: 40px; padding-top: 4px; }
.photo img { width: 200px; height: 200px; object-fit: cover; border-radius: 8px; }
.info { flex: 1; }
.username a { color: #58a6ff; font-size: 18px; font-weight: bold; text-decoration: none; }
.username a:hover { text-decoration: underline; }
.name { color: #aaa; margin-left: 8px; }
.score { color: #4ade80; margin: 4px 0; }
.bio { color: #999; font-size: 13px; margin: 8px 0; max-height: 40px; overflow: hidden; }
.opener { margin-top: 12px; padding: 12px; background: #111; border-radius: 8px; border: 1px solid #444; }
.opener-label { color: #888; font-size: 12px; margin-bottom: 4px; }
.opener-text { color: #f0f0f0; font-size: 16px; font-style: italic; }
.copy-btn { margin-top: 8px; padding: 4px 12px; background: #333; color: #fff; border: 1px solid #555; border-radius: 4px; cursor: pointer; font-size: 12px; }
.copy-btn:hover { background: #444; }
</style></head>
<body>
<h1>Instagram Openers</h1>
<p class="subtitle">Top ${candidates.length} profiles ranked by image classifier, with generated DM openers</p>
${rows}
</body></html>`;
}
