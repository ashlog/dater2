import fs from 'fs';
import path from 'path';
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

export async function readProfilesCache(): Promise<ProfilesCache> {
  try {
    const txt = await fs.promises.readFile(profilesCachePath, 'utf8');
    return JSON.parse(txt);
  } catch {
    return {} as ProfilesCache;
  }
}

export async function writeProfilesCache(cache: ProfilesCache): Promise<void> {
  await fs.promises.writeFile(profilesCachePath, JSON.stringify(cache, null, 2));
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

export async function markVisited(cache: ProfilesCache, key: string, ratingToken: string): Promise<void> {
  const loc = cache[key];
  if (!loc) return;
  const entry = loc.batch.find(e => e.ratingToken === ratingToken);
  if (entry && !entry.visited) {
    entry.visited = true;
    entry.visitedAt = new Date().toISOString();
    await writeProfilesCache(cache);
  }
}

