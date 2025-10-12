import { Profile } from './types';
import {
  bestPl,
  generateOpenersFromYamlBatch,
  generateImageOpenerFromYaml,
  getImageScore,
  saveImage,
  getOpenerPromptMetadata,
  setLLMProvider,
  type LLMProviderName,
  setLLMThinkingConfig,
  LLMRefusalError,
} from './llm';
import { hinge, sessionId } from './token';
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

const concurrency = 2;
const llmProvider: LLMProviderName = 'anthropic';
const model = 'anthropic/claude-sonnet-4.5';
const thinkingBudgetTokens = 1024;

setLLMProvider(llmProvider);
// setLLMThinkingConfig({ type: 'enabled', budget_tokens: thinkingBudgetTokens });

function formatDeliveryError(error: unknown): { code?: string; message: string } {
  const err = error as any;
  let code: string | undefined;
  if (err && typeof err === 'object') {
    if (err.code !== undefined) {
      code = String(err.code);
    } else if (err.response && typeof err.response.status === 'number') {
      code = String(err.response.status);
    }
  }
  let message: string;
  if (error instanceof Error && error.message) {
    message = error.message;
  } else if (err && typeof err.message === 'string') {
    message = err.message;
  } else {
    try {
      message = JSON.stringify(err);
    } catch {
      message = String(error);
    }
  }
  return { code, message };
}

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
  failureCounter: { consecutive: number; max: number };
};

async function fetchProfilesForSetting(setting: Settings, locKey: string, hingeRequestLimiter: ReturnType<typeof createHingeRequestLimiter>, cancelToken: ReturnType<typeof createCancelToken>) {
  let cache = await readProfilesCache();
  let unvisited = getUnvisited(cache, locKey);
  console.log(`[FETCH] Location ${locKey}: ${unvisited.length} unvisited profiles in cache`);
  if (unvisited.length === 0) {
    console.log(`[FETCH] Fetching fresh recommendations from Hinge for location ${locKey}`);
    let recs: Awaited<ReturnType<typeof hinge.getRecommendations>>;
    try {
      hingeRequestLimiter.tryConsume();
      recs = await hinge.getRecommendations(setting.longitude, setting.latitude);
    } catch (e) {
      // Only abort on rate limiter exhaustion, not transient API errors
      const errMsg = e instanceof Error ? e.message : String(e);
      if (errMsg.includes('request limit') || errMsg.includes('exhausted')) {
        cancelToken.abort(e);
        throw e;
      }
      // For other errors, log and return empty (will move to next location)
      console.error(`[FETCH] Error fetching recommendations for ${locKey}:`, e);
      console.log(`[FETCH] Skipping location ${locKey} due to fetch error`);
      return new Map();
    }

    const allSubjects = recs.feeds.flatMap((feed: any) => feed.subjects);
    const previewSubjects = recs.feeds.flatMap((feed: any) => feed.preview?.subjects ?? []);
    allSubjects.push(...previewSubjects);

    let profiles: Profile[];
    try {
      const subjectIds = allSubjects.map((it: any) => it.subjectId);
      const batches = chunkArray(subjectIds, 20);
      const allProfiles: Profile[] = [];

      for (const batch of batches) {
        hingeRequestLimiter.tryConsume();
        const batchProfiles = await hinge.getProfiles(batch as string[]);
        allProfiles.push(...batchProfiles);
      }

      profiles = allProfiles;
    } catch (e) {
      // Only abort on rate limiter exhaustion
      const errMsg = e instanceof Error ? e.message : String(e);
      if (errMsg.includes('request limit') || errMsg.includes('exhausted')) {
        cancelToken.abort(e);
        throw e;
      }
      // For other errors, log and return empty
      console.error(`[FETCH] Error fetching profiles for ${locKey}:`, e);
      console.log(`[FETCH] Skipping location ${locKey} due to fetch error`);
      return new Map();
    }
    const batch: CachedEntry[] = [];
    for (const subject of allSubjects) {
      const profile = profiles.find((p: any) => p.profile.userId === subject.subjectId);
      if (profile) batch.push({ ratingToken: subject.ratingToken, subjectId: subject.subjectId, visited: false, profile });
    }
    await upsertLocationBatch(locKey, batch);
    // New batch is all unvisited entries
    unvisited = batch;
    console.log(`[FETCH] Fetched ${unvisited.length} fresh profiles from Hinge`);
  }
  const profileMap = new Map<string, Profile>();
  for (const entry of unvisited) profileMap.set(entry.ratingToken, entry.profile);
  console.log(`[FETCH] Returning ${profileMap.size} profiles for processing`);
  return profileMap;
}

// Generate an opener only for the highest-scoring image; return single candidate
async function buildImageOpeners(
  subject: Profile,
  filteredAnswers: { questionId: string; response: string }[],
  pre: { scored: { url: string; score: number }[]; medianScore: number }
): Promise<{ candidates: { photo: any; comment: string }[]; medianScore: number | undefined; contextPrompts: string }> {
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

  if (!top) return { candidates: [], medianScore, contextPrompts: prompts };

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
    if (isBadResponse(cleanedComment)) return { candidates: [], medianScore, contextPrompts: prompts };
    return { candidates: [{ photo: top.photo, comment: cleanedComment }], medianScore, contextPrompts: prompts };
  } catch (error) {
    if (error instanceof LLMRefusalError) {
      // Log the refusal and re-throw so processProfileEntry can handle it
      console.warn('[buildImageOpeners] LLM refused to generate image opener, re-throwing for fallback handling');
      throw error;
    }
    // For other errors, log and return empty candidates
    console.error('[buildImageOpeners] Unexpected error generating image opener:', error);
    return { candidates: [], medianScore, contextPrompts: prompts };
  }
}

// Generate a text opener for each text answer using a single model call; return all candidates
async function generateTextOpeners(subject: Profile, filteredAnswers: { questionId: string; response: string }[]) {
  let tries = 0;
  while (++tries <= 2) {
    try {
      if (tries > 1) {
        console.log('generateTextOpeners: retrying (attempt', tries, 'of 2)');
        await timeout(3000);
      }

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
          if (!raw) {
            console.error(`generateTextOpeners: missing opener for id="${it.id}". Available ids:`, Array.from(idToText.keys()));
            return null;
          }
          const line = cleanPickupLineResponse(raw);
          if (isBadResponse(line)) {
            console.error(`generateTextOpeners: bad response for id="${it.id}": "${line}"`);
            return null;
          }
          return { answer: it.answer, prompt: it.prompt, line };
        })
        .filter((r): r is { answer: { questionId: string; response: string }; prompt: string; line: string } => r !== null);

      if (results.length === 0) {
        console.error('generateTextOpeners: API returned no openers for', items.length, 'items. Attempt', tries, 'of 2');
        if (tries >= 2) return results;
        continue;
      }

      // If we got at least one valid result, return it
      if (results.length > 0) {
        try { console.log('  Generated text lines: ', results.map(it => `${it.prompt} -- ${it.line}`)); } catch { }
        return results;
      }
    } catch (e) {
      if (e instanceof LLMRefusalError) {
        throw e;
      }
      console.error('generateTextOpeners batch error on attempt', tries, ':', e);
      if (tries >= 2) break; // Don't retry after last attempt
    }
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
    const openerPrompt = await getOpenerPromptMetadata();
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
      openerPromptHash: openerPrompt.hash,
      model,
      deliveryStatus: 'success',
    }, { openerPrompt });
  } catch { }
  try {
    await markVisited(undefined, locKey, ratingToken);
  } catch { }
}

async function recordRefusalAndVisit(
  ctx: LikeContext,
  subject: Profile,
  ratingToken: string,
  filteredAnswers: { questionId: string; response: string }[],
  validation: { scored: { url: string; score: number }[]; medianScore: number } | undefined,
  error: LLMRefusalError,
  options?: { skipLikeRelease?: boolean }
) {
  if (!options?.skipLikeRelease) {
    ctx.likeLimiter.release();
  }
  ctx.failureCounter.consecutive = 0;
  console.warn(
    '[LLM_REFUSAL] Model refused to generate opener for profile',
    subject.profile.userId,
    'stopReason=',
    error.stopReason
  );
  const promptsList = buildPromptsList(filteredAnswers.map(a => ({ questionId: a.questionId, response: a.response })));
  try {
    const openerPrompt = await getOpenerPromptMetadata();
    await appendDecision({
      timestamp: new Date().toISOString(),
      location: { latitude: ctx.setting.latitude, longitude: ctx.setting.longitude },
      userId: subject.profile.userId,
      ratingToken,
      decision: 'skip',
      images: subject.profile.photos.map(p => p.url),
      prompts: promptsList,
      profile: { firstName: subject.profile.firstName, age: subject.profile.age },
      imageScores: validation?.scored,
      medianImageScore: validation?.medianScore,
      openerPromptHash: openerPrompt.hash,
      model,
      deliveryStatus: 'error',
      deliveryError: {
        code: 'LLM_REFUSAL',
        message: `${error.message || 'Model refused to comply'} (stop_reason: ${error.stopReason})`,
      },
    }, { openerPrompt });
  } catch (logError) {
    console.error('Failed to record refusal decision entry', logError);
  }
  try {
    await markVisited(undefined, ctx.locKey, ratingToken);
  } catch (visitError) {
    console.error('Failed to mark rating token as visited after refusal', visitError);
  }
  console.log(
    '[LLM_REFUSAL] Marked profile as visited after refusal. Likes:',
    ctx.likeLimiter.getUsed(),
    'Like Rate:',
    ctx.getLikeRate().toFixed(2) + '%'
  );
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
  if (cancelToken.isAborted()) {
    // Release like reservation if cancelled
    likeLimiter.release();
    return;
  }
  // Note: like slot already reserved in processProfileEntry
  const best = imageCandidate.bestAnswer;
  const promptsList = buildPromptsList(filteredAnswers.map(a => ({ questionId: a.questionId, response: a.response })));
  let error: unknown = null;
  try {
    hingeRequestLimiter.tryConsume();
    console.log('Liking ' + best.photo.url + ' image with "' + best.comment + '". Average score: (' + imageCandidate.medianScore + '). Context:', imageCandidate.contextPrompts);
    await hinge.sendLike(subject.profile.userId, ratingToken, sessionId, { photoData: { url: best.photo.url, cdnId: best.photo.cdnId }, comment: best.comment });
  } catch (e) {
    error = e;
    likeLimiter.release();
    // Don't abort on individual like errors - they might be transient
    console.error('Error sending like:', e);
  } finally {
    try {
      let openerPrompt: { text: string; hash: string } | null = null;
      try {
        openerPrompt = await getOpenerPromptMetadata();
      } catch (metaError) {
        console.error('Failed to load opener prompt metadata', metaError);
      }
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
        prompts: promptsList,
        profile: { firstName: subject.profile.firstName, age: subject.profile.age },
        medianImageScore: imageCandidate.medianScore as any,
        imageScores: validation?.scored,
        openerPromptHash: openerPrompt?.hash,
        model,
        deliveryStatus: error ? 'error' : 'success',
        deliveryError: error ? formatDeliveryError(error) : undefined,
      }, openerPrompt ? { openerPrompt } : undefined);
    } catch (logError) {
      console.error('Failed to record decision entry', logError);
    }
  }
  // Mark as visited if like was successful OR if we got a 400 (invalid request - no point retrying)
  if (!error) {
    await markVisited(undefined, locKey, ratingToken);
    console.log('Likes', likeLimiter.getUsed(), 'Like Rate:', getLikeRate().toFixed(2) + '%');
  } else {
    // Check if error is a 400 (bad request) - if so, mark as visited to avoid retrying
    const errorInfo = formatDeliveryError(error);
    const is400 = errorInfo.code === '400' || errorInfo.message.includes('400');

    if (is400) {
      await markVisited(undefined, locKey, ratingToken);
      console.log('Like failed with 400 (invalid request), marking as visited. Likes:', likeLimiter.getUsed(), 'Like Rate:', getLikeRate().toFixed(2) + '%');
    } else {
      // Don't mark as visited for transient errors - profile will remain unvisited for retry
      console.log('Like failed with transient error, profile left unvisited for retry. Likes:', likeLimiter.getUsed(), 'Like Rate:', getLikeRate().toFixed(2) + '%');
    }
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
  if (cancelToken.isAborted()) {
    // Release like reservation if cancelled
    likeLimiter.release();
    return;
  }
  // Note: like slot already reserved in processProfileEntry
  const question = getQuestionById(bestAnswer.answer.questionId);
  const promptsList = buildPromptsList(filteredAnswers.map(a => ({ questionId: a.questionId, response: a.response })));
  let error: unknown = null;
  try {
    hingeRequestLimiter.tryConsume();
    console.log('Prompt: "' + bestAnswer.prompt + '"\n  - [PICKUP LINE] "' + bestAnswer.line + '"');
    // Find the original answer object to get contentId
    const originalAnswer = subject.profile.answers.find(a => a.questionId === bestAnswer.answer.questionId);
    await hinge.sendLike(subject.profile.userId, ratingToken, sessionId, {
      content: {
        prompt: {
          contentId: originalAnswer?.contentId,
          question: question,
          answer: bestAnswer.answer.response
        }
      },
      comment: bestAnswer.line
    });
  } catch (e) {
    error = e;
    likeLimiter.release();
    // Don't abort on individual like errors - they might be transient
    console.error('Error sending like:', e);
  } finally {
    try {
      let openerPrompt: { text: string; hash: string } | null = null;
      try {
        openerPrompt = await getOpenerPromptMetadata();
      } catch (metaError) {
        console.error('Failed to load opener prompt metadata', metaError);
      }
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
        prompts: promptsList,
        profile: { firstName: subject.profile.firstName, age: subject.profile.age },
        imageScores: validation?.scored,
        medianImageScore: validation?.medianScore,
        openerPromptHash: openerPrompt?.hash,
        model,
        deliveryStatus: error ? 'error' : 'success',
        deliveryError: error ? formatDeliveryError(error) : undefined,
      }, openerPrompt ? { openerPrompt } : undefined);
    } catch (logError) {
      console.error('Failed to record decision entry', logError);
    }
  }
  // Mark as visited if like was successful OR if we got a 400 (invalid request - no point retrying)
  if (!error) {
    await markVisited(undefined, locKey, ratingToken);
    console.log('Likes', likeLimiter.getUsed(), 'Like Rate:', getLikeRate().toFixed(2) + '%');
  } else {
    // Check if error is a 400 (bad request) - if so, mark as visited to avoid retrying
    const errorInfo = formatDeliveryError(error);
    const is400 = errorInfo.code === '400' || errorInfo.message.includes('400');

    if (is400) {
      await markVisited(undefined, locKey, ratingToken);
      console.log('Like failed with 400 (invalid request), marking as visited. Likes:', likeLimiter.getUsed(), 'Like Rate:', getLikeRate().toFixed(2) + '%');
    } else {
      // Don't mark as visited for transient errors - profile will remain unvisited for retry
      console.log('Like failed with transient error, profile left unvisited for retry. Likes:', likeLimiter.getUsed(), 'Like Rate:', getLikeRate().toFixed(2) + '%');
    }
  }
}

async function processProfileEntry(entry: [string, Profile], ctx: LikeContext, counters: { profileCount: number }) {
  const startTime = Date.now();
  const ratingToken = entry[0];
  const subject = entry[1];

  console.log(`\n[TIMING] Processing profile: ${subject.profile.firstName} (${subject.profile.userId})`);

  // Reserve a like slot BEFORE expensive processing to avoid wasted work
  // If we decide not to like, we'll release it later
  if (!ctx.likeLimiter.tryTake()) {
    console.log('Like limit reached, skipping profile (no reservation):', subject.profile.firstName, subject.profile.userId);
    // Profile remains unvisited for next run
    return;
  }

  const afterReservation = Date.now();
  console.log(`[TIMING] Like reservation: ${afterReservation - startTime}ms`);

  counters.profileCount++;
  const filteredAnswers = subject.profile.answers.filter(it => it.type === 'text');

  // Pre-validation: only proceed if image scores pass basic threshold
  const validationStart = Date.now();
  const validation = await validateProfileImages(subject);
  const validationEnd = Date.now();
  console.log(`[TIMING] Image validation: ${validationEnd - validationStart}ms`);

  if (!validation.pass) {
    console.log(
      'Image validation failed for',
      subject.profile.userId,
      'topScore:',
      validation.topScore,
      'medianScore:',
      validation.medianScore
    );
    // Release the like reservation - we're not using it
    ctx.likeLimiter.release();
    await recordSkipAndVisit(ctx.setting, subject, ratingToken, filteredAnswers, ctx.locKey, validation as any);
    return;
  }

  // Start both image and text generation in parallel, with proper error handling
  const imageCandidatesPromise = buildImageOpeners(subject, filteredAnswers, { scored: validation.scored, medianScore: validation.medianScore })
    .catch((error) => {
      // Catch errors immediately to prevent unhandled rejections
      if (error instanceof LLMRefusalError) {
        console.warn('[LLM_REFUSAL] Image generation refused (caught early):', error.stopReason);
        return { error, candidates: [], medianScore: validation?.medianScore, contextPrompts: '' };
      }
      console.error('[buildImageOpeners] Unexpected error (caught early):', error);
      throw error;
    });

  const textCandidatesStart = Date.now();
  let textCandidates: { answer: { questionId: string; response: string }; prompt: string; line: string }[] = [];
  const refusalErrors: LLMRefusalError[] = [];
  try {
    textCandidates = filteredAnswers.length > 0 ? await generateTextOpeners(subject, filteredAnswers) : [];
  } catch (error) {
    if (error instanceof LLMRefusalError) {
      refusalErrors.push(error);
      console.warn(
        '[LLM_REFUSAL] Text generation refused for profile',
        subject.profile.userId,
        'stopReason=',
        error.stopReason
      );
      textCandidates = [];
    } else {
      console.error('Unexpected error during text opener generation', error);
      textCandidates = [];
    }
  }
  const textCandidatesEnd = Date.now();
  console.log(`[TIMING] Text openers generation: ${textCandidatesEnd - textCandidatesStart}ms`);

  const imageCandidatesStart = Date.now();
  let imageCandidates: { candidates: { photo: any; comment: string }[]; medianScore: number | undefined; contextPrompts: string };
  const imageResult = await imageCandidatesPromise;
  if ('error' in imageResult && imageResult.error) {
    // Image generation was refused
    refusalErrors.push(imageResult.error as LLMRefusalError);
    console.warn(
      '[LLM_REFUSAL] Image generation refused for profile',
      subject.profile.userId,
      'stopReason=',
      (imageResult.error as LLMRefusalError).stopReason
    );
    imageCandidates = { candidates: [], medianScore: validation?.medianScore, contextPrompts: '' };
  } else {
    imageCandidates = imageResult as { candidates: { photo: any; comment: string }[]; medianScore: number | undefined; contextPrompts: string };
  }
  const imageCandidatesEnd = Date.now();
  console.log(`[TIMING] Image openers generation: ${imageCandidatesEnd - imageCandidatesStart}ms`);

  const allCandidates: { type: 'image' | 'text'; text: string; photo?: any; prompt?: string; answer?: { questionId: string; response: string } }[] = [];
  for (const ic of imageCandidates.candidates) allCandidates.push({ type: 'image', text: ic.comment, photo: ic.photo });
  for (const tc of textCandidates) allCandidates.push({ type: 'text', text: tc.line, prompt: tc.prompt, answer: tc.answer });

  console.log(`[TIMING] Total candidates generated: ${allCandidates.length} (${imageCandidates.candidates.length} image, ${textCandidates.length} text)`);

  if (allCandidates.length === 0) {
    if (refusalErrors.length > 0) {
      // Every generation path refused. Instead of retrying forever, fire off a best-effort
      // like with a high-quality photo and no comment so we can move on.
      const fallbackPhoto = (() => {
        const scored = validation?.scored ?? [];
        if (scored.length > 0) {
          const top = scored.reduce<{ url: string; score: number } | null>((acc, curr) => {
            if (!acc || curr.score > acc.score) return curr;
            return acc;
          }, null);
          if (top) {
            const bestMatch = subject.profile.photos.find(photo => photo.url === top.url);
            if (bestMatch) return bestMatch;
          }
        }
        return subject.profile.photos[0];
      })();
      if (fallbackPhoto) {
        console.warn('[LLM_REFUSAL] Falling back to silent like with top-rated photo for', subject.profile.userId);
        ctx.failureCounter.consecutive = 0;
        const fallbackCandidate = {
          bestAnswer: { photo: fallbackPhoto, comment: '' },
          medianScore: validation?.medianScore,
          contextPrompts: '',
        } as const;
        await likeWithImage(ctx, subject, ratingToken, fallbackCandidate, filteredAnswers, validation as any);
        return;
      }
      // No photo to fall back on – mark as visited via refusal handler.
      const refusalError = refusalErrors[0];
      await recordRefusalAndVisit(ctx, subject, ratingToken, filteredAnswers, validation as any, refusalError);
      return;
    }

    // The model produced nothing we can rank (non-refusal). Release the like slot and try again later.
    ctx.likeLimiter.release();

    // Increment consecutive failure counter
    ctx.failureCounter.consecutive++;
    console.log(
      'Generation failed for profile that passed validation:',
      subject.profile.userId,
      '- releasing like reservation and leaving UNVISITED for retry',
      `(consecutive failures: ${ctx.failureCounter.consecutive}/${ctx.failureCounter.max})`
    );

    // Check if we've hit the failure limit
    if (ctx.failureCounter.consecutive >= ctx.failureCounter.max) {
      console.error(
        `FATAL: ${ctx.failureCounter.consecutive} consecutive generation failures detected.`,
        'OpenRouter may be experiencing issues. Killing the app to prevent further failures.'
      );
      process.exit(1);
    }

    // Still log the decision for analytics, but DON'T mark as visited
    try {
      const openerPrompt = await getOpenerPromptMetadata();
      await appendDecision({
        timestamp: new Date().toISOString(),
        location: { latitude: ctx.setting.latitude, longitude: ctx.setting.longitude },
        userId: subject.profile.userId,
        ratingToken,
        decision: 'skip',
        images: subject.profile.photos.map(p => p.url),
        prompts: buildPromptsList(filteredAnswers.map(a => ({ questionId: a.questionId, response: a.response }))),
        profile: { firstName: subject.profile.firstName, age: subject.profile.age },
        imageScores: validation?.scored,
        medianImageScore: validation?.medianScore,
        openerPromptHash: openerPrompt.hash,
        model,
        deliveryStatus: 'error', // Mark as error since generation failed
        deliveryError: { code: 'GENERATION_FAILED', message: 'No openers generated despite passing validation' },
      }, { openerPrompt });
    } catch { }
    return;
  }

  // Reset consecutive failures on successful generation
  ctx.failureCounter.consecutive = 0;

  // Final ranker across ALL generations via single JSON index selection
  const bestPickerStart = Date.now();
  let bestIndex: number;
  try {
    bestIndex = await bestPl(allCandidates.map(c => c.text), model);
  } catch (error) {
    if (error instanceof LLMRefusalError) {
      refusalErrors.push(error);
      await recordRefusalAndVisit(ctx, subject, ratingToken, filteredAnswers, validation as any, error);
      return;
    }
    throw error;
  }
  const bestPickerEnd = Date.now();
  console.log(`[TIMING] Best picker: ${bestPickerEnd - bestPickerStart}ms (selected index ${bestIndex})`);

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

  const likeStart = Date.now();
  if (best.type === 'image' && best.photo) {
    const imageCandidate = { bestAnswer: { photo: best.photo, comment: best.text }, medianScore: imageCandidates.medianScore, contextPrompts: imageCandidates.contextPrompts } as const;
    await likeWithImage(ctx, subject, ratingToken, imageCandidate, filteredAnswers, validation as any);
  } else if (best.type === 'text' && best.answer && best.prompt) {
    const bestAnswer = { answer: best.answer, prompt: best.prompt, line: best.text } as const;
    await likeWithText(ctx, subject, ratingToken, bestAnswer as any, filteredAnswers, validation as any);
  } else {
    await recordSkipAndVisit(ctx.setting, subject, ratingToken, filteredAnswers, ctx.locKey, validation as any);
  }
  const likeEnd = Date.now();
  console.log(`[TIMING] Like/skip operation: ${likeEnd - likeStart}ms`);

  const totalTime = Date.now() - startTime;
  console.log(`[TIMING] TOTAL profile processing time: ${totalTime}ms (${(totalTime / 1000).toFixed(1)}s)\n`);
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
  let consecutiveGenerationFailures = 0;
  const MAX_CONSECUTIVE_FAILURES = 3;
  const likeLimiter = createLikeLimiter(maxLikes);
  const cancelToken = createCancelToken();
  const hingeRequestLimiter = createHingeRequestLimiter(cancelToken, 2000);
  const getLikeRate = () => (profileCount > 0 ? (likeLimiter.getUsed() / profileCount) * 100 : 0);

  for (const setting of settings) {
    const locKey = locationKeyOf(setting);
    console.log(`\n========== Processing location: ${locKey} ==========`);
    console.log(`Current likes: ${likeLimiter.getUsed()}/${maxLikes}`);
    try {
      while (!likeLimiter.isExhausted() && !cancelToken.isAborted()) {
        const profileMap = await fetchProfilesForSetting(setting, locKey, hingeRequestLimiter, cancelToken);
        // If no profiles available for this location, try next location
        if (profileMap.size === 0) {
          console.log(`[LOCATION] No more profiles available for location: ${locKey}, moving to next location`);
          break;
        }
        const ctx: LikeContext = {
          likeLimiter,
          cancelToken,
          hingeRequestLimiter,
          getLikeRate,
          setting,
          locKey,
          failureCounter: { consecutive: consecutiveGenerationFailures, max: MAX_CONSECUTIVE_FAILURES }
        };
        const limit = pLimit(concurrency);
        const counters = { profileCount };
        const tasks = [...profileMap.entries()].map(e => limit(() => processProfileEntry(e, ctx, counters)));
        try {
          await Promise.all(tasks);
        }
        catch (e) {
          // Log error but continue - individual profile errors shouldn't stop the entire run
          console.error('[ERROR] Error processing profiles batch:', e);
          // Don't break - continue to next batch
        }
        profileCount = counters.profileCount;
        consecutiveGenerationFailures = ctx.failureCounter.consecutive;
        if (likeLimiter.isExhausted() || cancelToken.isAborted()) break;
      }
    } catch (e) {
      console.error(e);
    }
    if (likeLimiter.isExhausted() || cancelToken.isAborted()) break;
  }
  console.log(`Total Likes: ${likeLimiter.getUsed()}, Total Profiles Scanned: ${profileCount}, Like Rate: ${getLikeRate().toFixed(2)}%`);
}
