import { Profile } from './types';
import { bestPl, generateOpenersFromYamlBatch, generateImageOpenerFromYaml, getImageScore, saveImage } from './openai';
import { hinge } from './token';
import { runningLocally } from '../globals';
import pLimit from 'p-limit';
import { timeout } from './utils';
import fs from 'fs';
import { readProfilesCache, locationKeyOf, getUnvisited, markVisited, CachedEntry, Settings, upsertLocationBatch } from './profile_cache';
import { buildPromptsList, appendDecision } from './decisions_log';
import { getQuestionById } from './questions';
import { cleanPickupLineResponse, isBadResponse } from './helpers';
import { createLikeLimiter, createCancelToken, createHingeRequestLimiter } from './limits';
import { all } from 'axios';

const concurrency = 10;
const model = 'anthropic/claude-sonnet-4';

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
    const previewSubjects = recs.feeds.flatMap(feed => feed.preview?.subjects ?? []);
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
    await upsertLocationBatch(locKey, batch);
    // New batch is all unvisited entries
    unvisited = batch;
  }
  const profileMap = new Map<string, Profile>();
  for (const entry of unvisited) profileMap.set(entry.ratingToken, entry.profile);
  return profileMap;
}

// Generate an opener only for the highest-scoring image; return single candidate
async function buildImageOpeners(
  subject: Profile,
  filteredAnswers: { questionId: string; response: string }[],
  pre: { scored: { url: string; score: number }[]; medianScore: number }
) {
  const prompts = filteredAnswers.map(a => `${getQuestionById(a.questionId)}: ${a.response}`).join('\n');
  let medianScore: number | undefined = pre?.medianScore;
  const scoreMap = new Map(pre.scored.map(s => [s.url, s.score] as const));
  const scoredPhotos: { photo: any; score: number }[] = subject.profile.photos.map(photo => ({ photo, score: scoreMap.get(photo.url) ?? 0 }));

  // Compute median if not provided
  if (medianScore === undefined) {
    const sortedScores = scoredPhotos.map(({ score }) => score).sort((a, b) => a - b);
    const mid = Math.floor(sortedScores.length / 2);
    medianScore = sortedScores.length % 2 !== 0 ? sortedScores[mid] : (sortedScores[mid - 1] + sortedScores[mid]) / 2;
  }

  // Pick the single highest scoring photo
  let top: { photo: any; score: number } | null = null;
  for (const s of scoredPhotos) {
    if (!top || s.score > top.score) top = s;
  }

  if (!top) return { candidates: [], medianScore, contextPrompts: prompts } as const;

  // Generate opener only for the top photo
  try {
    const comment = await generateImageOpenerFromYaml(
      top.photo.url,
      top.photo.caption,
      prompts,
      subject.profile.firstName,
      model
    );
    const cleanedComment = cleanPickupLineResponse(comment);
    if (isBadResponse(cleanedComment)) return { candidates: [], medianScore, contextPrompts: prompts } as const;
    return { candidates: [{ photo: top.photo, comment: cleanedComment }], medianScore, contextPrompts: prompts } as const;
  } catch {
    return { candidates: [], medianScore, contextPrompts: prompts } as const;
  }
}

// Generate a text opener for each text answer using a single model call; return all candidates
async function generateTextOpeners(subject: Profile, filteredAnswers: { questionId: string; response: string }[]) {
  let tries = 0;
  while (++tries <= 2) {
    try {
      if (tries > 1) await timeout(3000);

      // Build batched inputs with stable IDs
      const items = filteredAnswers.map((answer, idx) => {
        const question = getQuestionById(answer.questionId);
        const prompt = `${question}: ${answer.response}`;
        // Use simple sequential IDs: "1", "2", "3", ...
        const id = String(idx + 1);
        return { id, prompt, answer } as const;
      });

      const idToText = await generateOpenersFromYamlBatch(
        items.map(it => ({ id: it.id, profilePrompt: it.prompt })),
        subject.profile.firstName,
        model
      );

      const results = items
        .map(it => {
          const raw = idToText.get(it.id);
          if (!raw) return null;
          const line = cleanPickupLineResponse(raw);
          if (isBadResponse(line)) return null;
          return { answer: it.answer, prompt: it.prompt, line };
        })
        .filter((r): r is { answer: { questionId: string; response: string }; prompt: string; line: string } => r !== null);

      try { console.log('  Generated text lines: ', results.map(it => `${it.prompt} -- ${it.line}`)); } catch { }
      return results;
    } catch (e) { console.error('generateTextOpeners batch error', e); }
  }
  return [] as { answer: { questionId: string; response: string }; prompt: string; line: string }[];
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
    await markVisited(undefined, locKey, ratingToken);
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
    await markVisited(undefined, locKey, ratingToken);
    console.log('Likes', likeLimiter.getUsed(), 'Like Rate:', getLikeRate().toFixed(2) + '%');
  } catch (e) {
    likeLimiter.release();
    cancelToken.abort(e);
    console.error(e);
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
    await markVisited(undefined, locKey, ratingToken);
    console.log('Likes', likeLimiter.getUsed(), 'Like Rate:', getLikeRate().toFixed(2) + '%');
  } catch (e) {
    likeLimiter.release();
    cancelToken.abort(e);
    console.error(e);
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
  const imageCandidatesPromise = buildImageOpeners(subject, filteredAnswers, { scored: validation.scored, medianScore: validation.medianScore });
  const textCandidates = filteredAnswers.length > 0 ? await generateTextOpeners(subject, filteredAnswers) : [];
  
  const imageCandidates = await imageCandidatesPromise;
  const allCandidates: { type: 'image' | 'text'; text: string; photo?: any; prompt?: string; answer?: { questionId: string; response: string } }[] = [];
  for (const ic of imageCandidates.candidates) allCandidates.push({ type: 'image', text: ic.comment, photo: ic.photo });
  for (const tc of textCandidates) allCandidates.push({ type: 'text', text: tc.line, prompt: tc.prompt, answer: tc.answer });

  if (allCandidates.length === 0) {
    await recordSkipAndVisit(ctx.setting, subject, ratingToken, filteredAnswers, ctx.locKey, validation as any);
    return;
  }

  // Final ranker across ALL generations via single JSON index selection
  const bestIndex = await bestPl(allCandidates.map(c => c.text), model);
  const best = allCandidates[Math.max(0, Math.min(bestIndex, allCandidates.length - 1))];

  // Log the final winner locally for inspection as a single readable line with context
  if (runningLocally) {
    try {
      if (best.type === 'image' && best.photo) {
        const ctx = (imageCandidates.contextPrompts || '').replace(/\n+/g, ' || ').trim();
        const header = `FINAL [IMAGE] | user=${subject.profile.firstName} (${subject.profile.age}) | context=[${ctx}]`;
        const bodyLine2 = `Image: ${best.photo.url}`;
        const bodyLine3 = `  - [PICKUP LINE] \"${best.text}\"`;
        const log = header + '\n' + bodyLine2 + '\n' + bodyLine3 + '\n';
        fs.writeFileSync('bestAnswer.txt', log, { flag: 'a' });
      } else if (best.type === 'text' && best.answer && best.prompt) {
        const answered = best.answer; // narrowed and captured
        const respondedPrompt = best.prompt;
        const others = filteredAnswers
          .filter(a => a.questionId !== answered.questionId)
          .map(a => `${getQuestionById(a.questionId)}: ${a.response}`)
          .join(' || ');
        const header = `FINAL [TEXT] | user=${subject.profile.firstName} (${subject.profile.age}) | context=[${others}]`;
        const bodyLine2 = `Prompt: \"${respondedPrompt}\"`;
        const bodyLine3 = `  - [PICKUP LINE] \"${best.text}\"`;
        const log = header + '\n' + bodyLine2 + '\n' + bodyLine3 + '\n';
        fs.writeFileSync('bestAnswer.txt', log, { flag: 'a' });
      }
    } catch {}
  }

  if (best.type === 'image' && best.photo) {
    const imageCandidate = { bestAnswer: { photo: best.photo, comment: best.text }, medianScore: imageCandidates.medianScore, contextPrompts: imageCandidates.contextPrompts } as const;
    await likeWithImage(ctx, subject, ratingToken, imageCandidate, filteredAnswers, validation as any);
  } else if (best.type === 'text' && best.answer && best.prompt) {
    const bestAnswer = { answer: best.answer, prompt: best.prompt, line: best.text } as const;
    await likeWithText(ctx, subject, ratingToken, bestAnswer as any, filteredAnswers, validation as any);
  } else {
    await recordSkipAndVisit(ctx.setting, subject, ratingToken, filteredAnswers, ctx.locKey, validation as any);
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
        try {
          await Promise.all(tasks);
        }
        catch (e) {
          console.error(e);
          break;
        }
        profileCount = counters.profileCount;
        if (likeLimiter.isExhausted() || cancelToken.isAborted()) break;
      }
    } catch (e) {
      console.error(e);
    }
    if (likeLimiter.isExhausted() || cancelToken.isAborted()) break;
  }
  console.log(`Total Likes: ${likeLimiter.getUsed()}, Total Profiles Scanned: ${profileCount}, Like Rate: ${getLikeRate().toFixed(2)}%`);
}
