import fs from 'fs';
import path from 'path';
import pLimit from 'p-limit';
import { Profile } from './types';

export type CachedEntry = {
  ratingToken: string;
  subjectId: string;
  visited?: boolean;
  visitedAt?: string;
  profile: Profile;
};

export type LocationCache = {
  createdAt: string;
  batch: CachedEntry[];
};

export type ProfilesCache = {
  [locationKey: string]: LocationCache;
};

function resolveProfilesCachePath(): string {
  const candidates = [
    path.join(process.cwd(), 'src', 'profiles_cache.json'),
    path.join(process.cwd(), 'profiles_cache.json'),
    path.join(__dirname, '..', 'profiles_cache.json'),
  ];
  for (const p of candidates) {
    try {
      const dir = path.dirname(p);
      if (fs.existsSync(dir)) return p;
    } catch {}
  }
  return path.join(__dirname, 'profiles_cache.json');
}

const profilesCachePath = resolveProfilesCachePath();

// Serialize file access within this process to avoid lost updates
const ioLock = pLimit(1);

export async function readProfilesCache(): Promise<ProfilesCache> {
  return ioLock(async () => {
    try {
      const txt = await fs.promises.readFile(profilesCachePath, 'utf8');
      return JSON.parse(txt);
    } catch {
      return {} as ProfilesCache;
    }
  });
}

export async function writeProfilesCache(cache: ProfilesCache): Promise<void> {
  return ioLock(async () => {
    await fs.promises.writeFile(profilesCachePath, JSON.stringify(cache, null, 2));
  });
}

export type Settings = {
  longitude: number;
  latitude: number;
};

export function locationKeyOf(settings: Settings): string {
  return `${settings.latitude},${settings.longitude}`;
}

export function getUnvisited(cache: ProfilesCache, key: string): CachedEntry[] {
  const loc = cache[key];
  if (!loc || !Array.isArray(loc.batch)) return [];
  return loc.batch.filter(e => !e.visited);
}

// Atomically mark an entry as visited by doing a read-modify-write
// under a single-process lock to prevent lost updates.
export async function markVisited(_cache: ProfilesCache | undefined, key: string, ratingToken: string): Promise<void> {
  await ioLock(async () => {
    // Always read the latest state to merge concurrent updates
    let cache: ProfilesCache;
    try {
      const txt = await fs.promises.readFile(profilesCachePath, 'utf8');
      cache = JSON.parse(txt || '{}');
    } catch {
      cache = {} as ProfilesCache;
    }
    const loc = cache[key];
    if (!loc || !Array.isArray(loc.batch)) return;
    const entry = loc.batch.find(e => e.ratingToken === ratingToken);
    if (entry && !entry.visited) {
      entry.visited = true;
      entry.visitedAt = new Date().toISOString();
      await fs.promises.writeFile(profilesCachePath, JSON.stringify(cache, null, 2));
    }
  });
}

// Atomically replace the location batch for a key under the lock.
// This avoids full-object writes by other callers and prevents lost updates.
export async function upsertLocationBatch(key: string, batch: CachedEntry[]): Promise<void> {
  await ioLock(async () => {
    let cache: ProfilesCache;
    try {
      const txt = await fs.promises.readFile(profilesCachePath, 'utf8');
      cache = JSON.parse(txt || '{}');
    } catch {
      cache = {} as ProfilesCache;
    }
    cache[key] = { createdAt: new Date().toISOString(), batch };
    await fs.promises.writeFile(profilesCachePath, JSON.stringify(cache, null, 2));
  });
}
