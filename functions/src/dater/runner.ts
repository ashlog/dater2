import { Profile } from './types';
import { bestPl, generateOpenerFromYaml, generateImageOpenerFromYaml, judgeOpener, getImageScore, saveImage } from './openai';
import { hinge } from './token';
import { runningLocally } from '../globals';
import pLimit from 'p-limit';
import { distance } from 'fastest-levenshtein';
import { timeout } from './utils';
import fs from 'fs';
import { readProfilesCache, writeProfilesCache, locationKeyOf, getUnvisited, markVisited, CachedEntry, Settings } from './profile_cache';
import { buildPromptsList, appendDecision } from './decisions_log';
import { getQuestionById } from './questions';
import { cleanPickupLineResponse, isBadResponse } from './helpers';
import { createLikeLimiter, createCancelToken, createHingeRequestLimiter } from './limits';
import { all } from 'axios';

const concurrency = 10;

function chunkArray<T>(array: T[], chunkSize: number): T[][] {
  const chunks: T[][] = [];
  for (let i = 0; i < array.length; i += chunkSize) {
    chunks.push(array.slice(i, i + chunkSize));
  }
  return chunks;
}

type LikeContext = {
  likeLimiter: ReturnType<typeof createLikeLimiter>;
  cancelToken: ReturnType<typeof createCancelToken>;
  hingeRequestLimiter: ReturnType<typeof createHingeRequestLimiter>;
  getLikeRate: () => number;
  setting: Settings;
  locKey: string;
};

async function fetchProfilesForSetting(setting: Settings, locKey: string, hingeRequestLimiter: ReturnType<typeof createHingeRequestLimiter>, cancelToken: ReturnType<typeof createCancelToken>) {
  let cache = await readProfilesCache();
  let unvisited = getUnvisited(cache, locKey);
  if (unvisited.length === 0) {
    const recs = await (async () => {
      try {
        hingeRequestLimiter.tryConsume();
        return await hinge.getRecommendations(setting.longitude, setting.latitude);
      } catch (e) {
        cancelToken.abort(e);
        throw e;
      }
    })();
    const allSubjects = recs.feeds.flatMap(feed => feed.subjects);
    const previewSubjects = recs.feeds.flatMap(feed => feed.subjects.map(it => ({ ...it, preview: feed.preview.subjects })));
    allSubjects.push(...previewSubjects);
    const profiles = await (async () => {
      try {
        const subjectIds = allSubjects.map(it => it.subjectId);
        const batches = chunkArray(subjectIds, 20);
        const allProfiles: Profile[] = [];
        
        for (const batch of batches) {
          hingeRequestLimiter.tryConsume();
          const batchProfiles = await hinge.getProfiles(batch);
          allProfiles.push(...batchProfiles);
        }
        
        return allProfiles;
      } catch (e) {
        cancelToken.abort(e);
        throw e;
      }
    })();
    const batch: CachedEntry[] = [];
    for (const subject of allSubjects) {
      const profile = profiles.find(p => p.profile.userId === subject.subjectId);
      if (profile) batch.push({ ratingToken: subject.ratingToken, subjectId: subject.subjectId, visited: false, profile });
    }
    cache[locKey] = { createdAt: new Date().toISOString(), batch };
    await writeProfilesCache(cache);
    unvisited = getUnvisited(cache, locKey);
  }
  const profileMap = new Map<string, Profile>();
  for (const entry of unvisited) profileMap.set(entry.ratingToken, entry.profile);
  return profileMap;
}

async function buildImageOpener(
  subject: Profile,
  filteredAnswers: { questionId: string; response: string }[],
  pre: { scored: { url: string; score: number }[]; medianScore: number }
) {
  const prompts = filteredAnswers.map(a => `${getQuestionById(a.questionId)}: ${a.response}`).join('\n');
  let medianScore: number | undefined = pre?.medianScore;
  const plMap = await (async () => {
    let scores: { photo: any; score: number }[] = [];
    const scoreMap = new Map(pre.scored.map(s => [s.url, s.score] as const));
    scores = subject.profile.photos.map(photo => ({ photo, score: scoreMap.get(photo.url) ?? 0 }));
    if (medianScore === undefined) {
      const sortedScores = scores.map(({ score }) => score).sort((a, b) => a - b);
      const mid = Math.floor(sortedScores.length / 2);
      medianScore = sortedScores.length % 2 !== 0 ? sortedScores[mid] : (sortedScores[mid - 1] + sortedScores[mid]) / 2;
    }
    const processLimit = pLimit(concurrency);
    const processPromises = scores
      .sort((a, b) => b.score - a.score)
      .map(({ photo, score }) =>
        processLimit(async () => {
          if (score > 0.5) {
            try {
              if ((medianScore ?? 0) < 0.2 && score < 0.7) return null;
              const comment = await generateImageOpenerFromYaml(photo.url, photo.caption, prompts, subject.profile.firstName, 'gemini-2.5-pro');
              const cleanedComment = cleanPickupLineResponse(comment);
              if (isBadResponse(cleanedComment)) return null;
              return { photo, comment: cleanedComment };
            } catch { return null; }
          }
          return null;
        })
      );
    const results = await Promise.all(processPromises);
    return results.filter((r): r is { photo: any; comment: string } => r !== null);
  })();
  const plList = plMap.map(it => it.comment).filter(Boolean);
  if (plList.length === 0) return { bestAnswer: undefined as any, medianScore, contextPrompts: prompts } as const;
  if (plList.length > 1) {
    const pl = await bestPl(plList);
    plMap.sort((a, b) => distance(pl!, a.comment) - distance(pl!, b.comment));
  }
  const bestAnswer = plMap[0];
  return { bestAnswer, medianScore, contextPrompts: prompts } as const;
}

async function generateBestTextAnswer(subject: Profile, filteredAnswers: { questionId: string; response: string }[]) {
  let bestAnswer: { answer: { questionId: string; response: string }; prompt: string; line: string } | undefined;
  let tries = 0;
  while (!bestAnswer && ++tries < 2) {
    try {
      if (tries > 1) await timeout(3000);
      const plMap = await Promise.all(
        filteredAnswers.map(async answer => {
          try {
            const question = getQuestionById(answer.questionId);
            const prompt = `${question}: ${answer.response}`;
            let line = await generateOpenerFromYaml(prompt, subject.profile.firstName);
            line = cleanPickupLineResponse(line);
            return { answer, prompt, line };
          } catch { return null; }
        })
      ).then(results => results.filter(r => r !== null) as NonNullable<typeof results[number]>[]);
      try { console.log('  Generated lines: ', plMap.map(it => `${it.prompt} -- ${it.line}`)); } catch { }
      const pl = (await bestPl(plMap.map(it => it.line))).replaceAll('"', '');
      plMap.sort((a, b) => distance(pl!, a.line) - distance(pl!, b.line));
      bestAnswer = plMap[0];
    } catch { }
  }
  return bestAnswer;
}

async function recordSkipAndVisit(
  setting: Settings,
  subject: Profile,
  ratingToken: string,
  filteredAnswers: { questionId: string; response: string }[],
  locKey: string,
  validation?: { scored: { url: string; score: number }[]; medianScore: number }
) {
  try {
    await appendDecision({
      timestamp: new Date().toISOString(),
      location: { latitude: setting.latitude, longitude: setting.longitude },
      userId: subject.profile.userId,
      ratingToken,
      decision: 'skip',
      images: subject.profile.photos.map(p => p.url),
      prompts: buildPromptsList(filteredAnswers.map(a => ({ questionId: a.questionId, response: a.response }))),
      profile: { firstName: subject.profile.firstName, age: subject.profile.age },
      imageScores: validation?.scored,
      medianImageScore: validation?.medianScore,
    });
  } catch { }
  try {
    const c = await readProfilesCache();
    await markVisited(c, locKey, ratingToken);
  } catch { }
}

async function likeWithImage(
  ctx: LikeContext,
  subject: Profile,
  ratingToken: string,
  imageCandidate: { bestAnswer: any; medianScore: number | undefined; contextPrompts: string },
  filteredAnswers: { questionId: string; response: string }[],
  validation?: { scored: { url: string; score: number }[]; medianScore: number }
) {
  const { hingeRequestLimiter, setting, locKey, likeLimiter, getLikeRate, cancelToken } = ctx;
  if (cancelToken.isAborted()) return;
  if (!likeLimiter.tryTake()) return;
  try {
    hingeRequestLimiter.tryConsume();
    const best = imageCandidate.bestAnswer;
    console.log('Liking ' + best.photo.url + ' image with "' + best.comment + '". Average score: (' + imageCandidate.medianScore + '). Context:', imageCandidate.contextPrompts);
    await hinge.sendLike(subject.profile.userId, ratingToken, { photoData: { url: best.photo.url, cdnId: best.photo.cdnId }, comment: best.comment });
    try {
      await appendDecision({
        timestamp: new Date().toISOString(),
        location: { latitude: setting.latitude, longitude: setting.longitude },
        userId: subject.profile.userId,
        ratingToken,
        decision: 'like',
        decisionSource: 'image',
        comment: best.comment,
        photoUsed: best.photo.url,
        images: subject.profile.photos.map(p => p.url),
        prompts: buildPromptsList(filteredAnswers.map(a => ({ questionId: a.questionId, response: a.response }))),
        profile: { firstName: subject.profile.firstName, age: subject.profile.age },
        medianImageScore: imageCandidate.medianScore as any,
        imageScores: validation?.scored,
      });
    } catch { }
    const c = await readProfilesCache();
    await markVisited(c, locKey, ratingToken);
    console.log('Likes', likeLimiter.getUsed(), 'Like Rate:', getLikeRate().toFixed(2) + '%');
  } catch (e) {
    likeLimiter.release();
    cancelToken.abort(e);
    throw e;
  }
}

async function likeWithText(
  ctx: LikeContext,
  subject: Profile,
  ratingToken: string,
  bestAnswer: { answer: { questionId: string; response: string }; prompt: string; line: string },
  filteredAnswers: { questionId: string; response: string }[],
  validation?: { scored: { url: string; score: number }[]; medianScore: number }
) {
  const { hingeRequestLimiter, setting, locKey, likeLimiter, getLikeRate, cancelToken } = ctx;
  if (cancelToken.isAborted()) return;
  if (!likeLimiter.tryTake()) return;
  try {
    hingeRequestLimiter.tryConsume();
    const question = getQuestionById(bestAnswer.answer.questionId);
    console.log('Prompt: "' + bestAnswer.prompt + '"\n  - [PICKUP LINE] "' + bestAnswer.line + '"');
    await hinge.sendLike(subject.profile.userId, ratingToken, { content: { prompt: { question: question, answer: bestAnswer.answer.response } }, comment: bestAnswer.line });
    try {
      await appendDecision({
        timestamp: new Date().toISOString(),
        location: { latitude: setting.latitude, longitude: setting.longitude },
        userId: subject.profile.userId,
        ratingToken,
        decision: 'like',
        decisionSource: 'text',
        comment: bestAnswer.line,
        photoUsed: null,
        images: subject.profile.photos.map(p => p.url),
        prompts: buildPromptsList(filteredAnswers.map(a => ({ questionId: a.questionId, response: a.response }))),
        profile: { firstName: subject.profile.firstName, age: subject.profile.age },
        imageScores: validation?.scored,
        medianImageScore: validation?.medianScore,
      });
    } catch { }
    const c = await readProfilesCache();
    await markVisited(c, locKey, ratingToken);
    console.log('Likes', likeLimiter.getUsed(), 'Like Rate:', getLikeRate().toFixed(2) + '%');
  } catch (e) {
    likeLimiter.release();
    cancelToken.abort(e);
    throw e;
  }
}

async function processProfileEntry(entry: [string, Profile], ctx: LikeContext, counters: { profileCount: number }) {
  const ratingToken = entry[0];
  const subject = entry[1];
  counters.profileCount++;
  const filteredAnswers = subject.profile.answers.filter(it => it.type === 'text');
  // Pre-validation: only proceed if image scores pass basic threshold
  const validation = await validateProfileImages(subject);
  if (!validation.pass) {
    console.log(
      'Image validation failed for',
      subject.profile.userId,
      'topScore:',
      validation.topScore,
      'medianScore:',
      validation.medianScore
    );
    await recordSkipAndVisit(ctx.setting, subject, ratingToken, filteredAnswers, ctx.locKey, validation as any);
    return;
  }
  const imageCandidatePromise = buildImageOpener(subject, filteredAnswers, { scored: validation.scored, medianScore: validation.medianScore });
  if (filteredAnswers.length > 0) {
    const bestAnswer = await generateBestTextAnswer(subject, filteredAnswers);
    if (runningLocally && bestAnswer) {
      fs.writeFileSync('bestAnswer.txt', `Prompt: "${bestAnswer.prompt}"\n  - [PICKUP LINE] "${bestAnswer.line}"` + '\n\n', { flag: 'a' });
    }
    const imageCandidate = await imageCandidatePromise;
    let useImage = false;
    if (imageCandidate?.bestAnswer && bestAnswer) {
      const choice = await judgeOpener(imageCandidate.bestAnswer.comment, bestAnswer.line);
      useImage = choice === '1';
    } else if (imageCandidate?.bestAnswer && !bestAnswer) {
      useImage = true;
    } else if (!imageCandidate?.bestAnswer && bestAnswer) {
      useImage = false;
    } else {
      await recordSkipAndVisit(ctx.setting, subject, ratingToken, filteredAnswers, ctx.locKey, validation as any);
      return;
    }
    if (useImage && imageCandidate?.bestAnswer) await likeWithImage(ctx, subject, ratingToken, imageCandidate, filteredAnswers, validation as any);
    else if (bestAnswer) await likeWithText(ctx, subject, ratingToken, bestAnswer as any, filteredAnswers, validation as any);
  } else {
    const imageCandidate = await imageCandidatePromise;
    if (!imageCandidate?.bestAnswer) {
      await recordSkipAndVisit(ctx.setting, subject, ratingToken, [], ctx.locKey, validation as any);
      return;
    }
    await likeWithImage(ctx, subject, ratingToken, imageCandidate, [], validation as any);
  }
}

async function validateProfileImages(subject: Profile): Promise<{ pass: boolean; topScore: number; medianScore: number; scored: { url: string; score: number }[]; }> {
  // Mirror buildImageOpener logic for scoring and gating, and save images here
  const imageLimit = pLimit(concurrency);
  const scored = await Promise.all(
    subject.profile.photos.map(photo =>
      imageLimit(async () => {
        const sc = await getImageScore(photo.url);
        console.log('Score:', sc, photo.url);
        return { url: photo.url, score: sc.score };
      })
    )
  );
  const scores = scored.map(s => s.score);
  if (scores.length === 0) return { pass: false, topScore: 0, medianScore: 0, scored };
  const sorted = [...scores].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  const top = Math.max(...scores);
  // Save images that clearly pass (same condition previously in buildImageOpener)
  try {
    for (const s of scored) {
      if (s.score > 0.5 && median > 0.5) {
        await saveImage(s.url, 'like');
      }
    }
  } catch { }
  // Pass if any candidate would be processed in buildImageOpener
  const anyCandidate = scored.some(s => s.score > 0.5 && !(median < 0.2 && s.score < 0.7));
  return { pass: anyCandidate, topScore: top, medianScore: median, scored };
}

export async function run(settings: Settings[], maxLikes = 100000) {
  let profileCount = 0;
  const likeLimiter = createLikeLimiter(maxLikes);
  const cancelToken = createCancelToken();
  const hingeRequestLimiter = createHingeRequestLimiter(cancelToken, 2000);
  const getLikeRate = () => (profileCount > 0 ? (likeLimiter.getUsed() / profileCount) * 100 : 0);

  for (const setting of settings) {
    try {
      for (let i = 0; i < 10 && !likeLimiter.isExhausted() && !cancelToken.isAborted(); i++) {
        const locKey = locationKeyOf(setting);
        const profileMap = await fetchProfilesForSetting(setting, locKey, hingeRequestLimiter, cancelToken);
        if (profileMap.size === 0) continue;
        const ctx: LikeContext = { likeLimiter, cancelToken, hingeRequestLimiter, getLikeRate, setting, locKey };
        const limit = pLimit(concurrency);
        const counters = { profileCount };
        const tasks = [...profileMap.entries()].map(e => limit(() => processProfileEntry(e, ctx, counters)));
        try { await Promise.all(tasks); } catch { break; }
        profileCount = counters.profileCount;
        if (likeLimiter.isExhausted() || cancelToken.isAborted()) break;
      }
    } catch (e) { console.error(e); }
    if (likeLimiter.isExhausted() || cancelToken.isAborted()) break;
  }
  console.log(`Total Likes: ${likeLimiter.getUsed()}, Total Profiles Scanned: ${profileCount}, Like Rate: ${getLikeRate().toFixed(2)}%`);
}
