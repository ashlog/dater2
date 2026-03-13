import express from 'express';
import cors from 'cors';
import path from 'path';
import os from 'os';
import pLimit from 'p-limit';
import {
  generateImageOpenerFromYaml,
  setLLMProvider,
  getCostSummary,
  getOpenerPromptMetadata,
  setOpenerPromptOverride,
} from './dater/llm';
import {
  loadClassificationCache,
  getTopPostsPerUser,
  resolveImageAsDataUri,
  fetchIgProfile,
  fetchPostCaption,
  type IgPost,
} from './dater/instagram';
import { cleanPickupLineResponse, isBadResponse } from './dater/helpers';

setLLMProvider('claude-direct');

const app = express();
app.use(cors());
app.use(express.json());

const PORT = 3001;
const CACHE_PATH = path.join(os.homedir(), 'projects/siglip-test/ig_classify_cache_full.json');

// In-memory caches
const imageCache = new Map<string, { dataUri: string; cdnUrl: string }>();
const profileCache = new Map<string, { name: string; bio: string }>();
const captionCache = new Map<string, string>();

// Mutex for opener generation (setOpenerPromptOverride is global state)
const openerLimit = pLimit(1);

// Swap prompt context from Hinge to Instagram DM
async function withInstagramContext<T>(fn: () => Promise<T>): Promise<T> {
  const { text } = await getOpenerPromptMetadata();
  const igText = text
    .replace(/for Hinge/g, 'as an Instagram DM')
    .replace(/on hinge/gi, 'on instagram')
    .replace(/on Hinge/g, 'on Instagram');
  setOpenerPromptOverride(igText);
  try {
    return await fn();
  } finally {
    setOpenerPromptOverride(null);
  }
}

// GET /api/profiles — all unique users sorted by top score
app.get('/api/profiles', (_req, res) => {
  try {
    const posts = loadClassificationCache(CACHE_PATH);
    const top = getTopPostsPerUser(posts, 999);
    res.json(top);
  } catch (e: any) {
    res.status(500).json({ error: e.message });
  }
});

// GET /api/image/:code — resolve Instagram image to base64
app.get('/api/image/:code', async (req, res): Promise<void> => {
  const { code } = req.params;
  const cached = imageCache.get(code);
  if (cached) { res.json(cached); return; }

  const result = await resolveImageAsDataUri(code);
  if (!result) { res.status(404).json({ error: 'Could not resolve image' }); return; }

  imageCache.set(code, result);
  res.json(result);
});

// GET /api/profile/:username — fetch IG profile info
app.get('/api/profile/:username', async (req, res): Promise<void> => {
  const { username } = req.params;
  const cached = profileCache.get(username);
  if (cached) { res.json(cached); return; }

  const result = await fetchIgProfile(username);
  profileCache.set(username, result);
  res.json(result);
});

// GET /api/caption/:code — fetch post caption
app.get('/api/caption/:code', async (req, res): Promise<void> => {
  const { code } = req.params;
  const cached = captionCache.get(code);
  if (cached !== undefined) { res.json({ caption: cached }); return; }

  const caption = await fetchPostCaption(code);
  captionCache.set(code, caption);
  res.json({ caption });
});

// POST /api/opener — generate an opener for a profile
app.post('/api/opener', async (req, res): Promise<void> => {
  const { code, username, name, bio, caption } = req.body;
  if (!code || !username) { res.status(400).json({ error: 'code and username required' }); return; }

  try {
    const opener = await openerLimit(async () => {
      // Resolve image
      let image = imageCache.get(code);
      if (!image) {
        const resolved = await resolveImageAsDataUri(code);
        if (!resolved) throw new Error('Could not resolve image');
        imageCache.set(code, resolved);
        image = resolved;
      }

      // Resolve caption if not provided
      let postCaption = caption || '';
      if (!postCaption) {
        const cached = captionCache.get(code);
        if (cached !== undefined) {
          postCaption = cached;
        } else {
          postCaption = await fetchPostCaption(code);
          captionCache.set(code, postCaption);
        }
      }

      // Generate with Instagram context
      const rawOpener = await withInstagramContext(() =>
        generateImageOpenerFromYaml(
          image!.dataUri,
          postCaption,
          bio || '',
          name || username,
          'claude-opus-4-6'
        )
      );

      const cleaned = cleanPickupLineResponse(rawOpener);
      if (isBadResponse(cleaned)) throw new Error('Bad response from LLM');
      return cleaned;
    });

    res.json({ opener, cost: getCostSummary() });
  } catch (e: any) {
    console.error('Opener generation error:', e.message);
    res.status(500).json({ error: e.message });
  }
});

app.listen(PORT, () => {
  console.log(`Dater2 Instagram API running on http://localhost:${PORT}`);
});
