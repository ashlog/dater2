// Set up environment variables before any imports
process.env.OPENROUTER_API_KEY = 'test-key';
process.env.OPENAI_API_KEY = 'test-key';

import { Profile } from './types';
import { Settings } from './profile_cache';

// Mock all external dependencies
jest.mock('./llm');
jest.mock('./token');
jest.mock('./profile_cache');
jest.mock('./decisions_log');
jest.mock('./questions');
jest.mock('./helpers');
jest.mock('./limits');
jest.mock('./utils');
jest.mock('fs');
jest.mock('axios', () => ({
  __esModule: true,
  default: { get: jest.fn().mockResolvedValue({ data: 'OK' }) },
}));

import * as llm from './llm';
import * as token from './token';
import * as profileCache from './profile_cache';
import * as decisionsLog from './decisions_log';
import * as questions from './questions';
import * as helpers from './helpers';
import * as limits from './limits';
import fs from 'fs';

// Import the run function after mocks are set up
import { run } from './runner';

const fsMock = fs as unknown as {
  appendFileSync: jest.Mock;
  existsSync: jest.Mock;
  mkdirSync: jest.Mock;
};

describe('Runner - maxLikes and visited property behavior', () => {
  let mockHinge: any;
  let mockLikeLimiter: any;
  let mockCancelToken: any;
  let mockHingeRequestLimiter: any;
  let mockProfilesCache: any;

  const createMockProfile = (userId: string, firstName: string): Profile => ({
    identityId: userId,
    profile: {
      userId,
      firstName,
      age: 25,
      answers: [{
        position: 0,
        questionId: 'q1',
        type: 'text',
        response: 'Test answer'
      }],
      genderId: 1,
      height: 170,
      instafeedVisible: false,
      justJoinedDisplayUntil: '',
      location: { name: 'SF' },
      lastActiveStatus: 'active',
      lastActiveStatusId: 1,
      selfieVerified: true,
      drinking: 0,
      drugs: 0,
      marijuana: 0,
      smoking: 0,
      photos: [{
        cdnId: 'cdn1',
        source: 'upload',
        sourceId: 'src1',
        boundingBox: {
          bottomRight: { x: 1, y: 1 },
          topLeft: { x: 0, y: 0 }
        },
        caption: '',
        location: '',
        promptId: '',
        videos: [{
          url: `https://example.com/${userId}.mp4`,
          height: 100,
          width: 100,
          quality: 'high'
        }],
        url: `https://example.com/${userId}.jpg`,
        width: 100,
        height: 100
      }]
    }
  });

  const createMockSettings = (): Settings => ({
    latitude: 37.75,
    longitude: -122.44
  });

  beforeEach(() => {
    jest.clearAllMocks();
    fsMock.appendFileSync = jest.fn();
    fsMock.existsSync = jest.fn().mockReturnValue(true);
    fsMock.mkdirSync = jest.fn();

    // Setup mock Hinge API
    mockHinge = {
      getRecommendations: jest.fn().mockResolvedValue({ feeds: [], preview: { subjects: [] } }),
      getProfiles: jest.fn().mockResolvedValue([]),
      textReview: jest.fn().mockResolvedValue({ hcmRunId: 'test-hcm-id', isHarmful: false }),
      sendLike: jest.fn()
    };
    (token as any).hinge = mockHinge;
    (token as any).sessionId = 'test-session-id';

    // Setup mock like limiter
    let remaining = 0;
    let used = 0;
    mockLikeLimiter = {
      tryTake: jest.fn(() => {
        if (remaining <= 0) return false;
        remaining--;
        used++;
        return true;
      }),
      release: jest.fn(() => {
        if (used > 0) {
          used--;
          remaining++;
        }
      }),
      isExhausted: jest.fn(() => remaining <= 0),
      getUsed: jest.fn(() => used)
    };

    // Setup mock cancel token
    let aborted = false;
    mockCancelToken = {
      abort: jest.fn((err: unknown) => { aborted = true; }),
      isAborted: jest.fn(() => aborted),
      reason: jest.fn(() => undefined)
    };

    // Setup mock hinge request limiter
    mockHingeRequestLimiter = {
      tryConsume: jest.fn(),
      remaining: jest.fn(() => 2000)
    };

    // Setup mock profiles cache
    mockProfilesCache = {};
    (profileCache.readProfilesCache as jest.Mock).mockResolvedValue(mockProfilesCache);
    (profileCache.getUnvisited as jest.Mock).mockReturnValue([]);
    (profileCache.markVisited as jest.Mock).mockResolvedValue(undefined);
    (profileCache.upsertLocationBatch as jest.Mock).mockResolvedValue(undefined);
    (profileCache.locationKeyOf as jest.Mock).mockImplementation((s: Settings) =>
      `${s.latitude},${s.longitude}`
    );

    // Setup mock limits
    (limits.createLikeLimiter as jest.Mock).mockImplementation((max: number) => {
      remaining = max;
      used = 0;
      return mockLikeLimiter;
    });
    (limits.createCancelToken as jest.Mock).mockReturnValue(mockCancelToken);
    (limits.createHingeRequestLimiter as jest.Mock).mockReturnValue(mockHingeRequestLimiter);

    // Setup mock OpenAI functions
    (llm.getImageScore as jest.Mock).mockResolvedValue({ score: 0.8 });
    (llm.generateOpenersFromYamlBatch as jest.Mock).mockResolvedValue(
      new Map([['1', 'Great opener!']])
    );
    (llm.generateImageOpenerFromYaml as jest.Mock).mockResolvedValue('Nice photo!');
    (llm.bestPl as jest.Mock).mockResolvedValue(0);
    (llm.saveImage as jest.Mock).mockResolvedValue(undefined);
    (llm.getOpenerPromptMetadata as jest.Mock).mockResolvedValue({
      text: 'prompt',
      hash: 'hash123'
    });

    // Setup mock decisions log
    (decisionsLog.appendDecision as jest.Mock).mockResolvedValue(undefined);
    (decisionsLog.buildPromptsList as jest.Mock).mockReturnValue([]);

    // Setup mock questions
    (questions.getQuestionById as jest.Mock).mockReturnValue('Test question?');

    // Setup mock helpers
    (helpers.cleanPickupLineResponse as jest.Mock).mockImplementation((s: string) => s);
    (helpers.isBadResponse as jest.Mock).mockImplementation((s: string) => !s); // Returns true if empty

    // Provide cost tracker mocks
    (llm.getCostSummary as jest.Mock).mockReturnValue({
      inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0, totalCost: 0, calls: 0,
    });
    (llm.resetCostTracker as jest.Mock).mockImplementation(() => {});

    // Provide LLMRefusalError class for refusal handling branches
    (llm as any).LLMRefusalError = class extends Error {
      stopReason: string;
      provider: string;
      raw?: unknown;
      constructor(message = 'refusal', provider = 'anthropic', stopReason = 'refusal', raw?: unknown) {
        super(message);
        this.name = 'LLMRefusalError';
        this.provider = provider;
        this.stopReason = stopReason;
        this.raw = raw;
      }
    };
  });

  describe('profile cache and logging integration', () => {
    test('fetches fresh profiles, updates cache, generates openers, logs decision, and sends like', async () => {
      const settings: Settings = { latitude: 37.75, longitude: -122.44, label: 'Test City' };
      const profile = createMockProfile('user-new', 'Dana');

      // Initial cache hit returns no unvisited profiles so we fetch fresh ones
      (profileCache.readProfilesCache as jest.Mock).mockResolvedValueOnce({});
      (profileCache.getUnvisited as jest.Mock)
        .mockReturnValueOnce([])
        .mockReturnValue([]); // Subsequent calls return empty to avoid extra batches

      mockHinge.getRecommendations.mockResolvedValueOnce({
        feeds: [
          {
            subjects: [{ subjectId: 'user-new', ratingToken: 'token-new' }],
            preview: { subjects: [] }
          }
        ]
      });
      mockHinge.getProfiles.mockResolvedValueOnce([profile]);

      (llm.generateImageOpenerFromYaml as jest.Mock).mockResolvedValue('Great photo opener!');
      (llm.generateOpenersFromYamlBatch as jest.Mock).mockResolvedValue(
        new Map([['1', 'Witty text opener!']])
      );
      (llm.bestPl as jest.Mock).mockResolvedValue(1); // Prefer text opener over image
      mockHinge.sendLike.mockResolvedValue({});

      await run([settings], 5);

      expect(mockHinge.getRecommendations).toHaveBeenCalledWith(
        settings.longitude,
        settings.latitude
      );
      expect(mockHinge.getProfiles).toHaveBeenCalledTimes(1);
      expect(profileCache.upsertLocationBatch).toHaveBeenCalledWith(
        `${settings.latitude},${settings.longitude}`,
        expect.arrayContaining([
          expect.objectContaining({
            ratingToken: 'token-new',
            subjectId: 'user-new',
            visited: false,
            profile
          })
        ])
      );
      expect(llm.generateImageOpenerFromYaml).toHaveBeenCalled();
      expect(llm.generateOpenersFromYamlBatch).toHaveBeenCalled();
      expect(mockHinge.sendLike).toHaveBeenCalledTimes(1);
      expect(mockHinge.sendLike.mock.calls[0][3]).toEqual(
        expect.objectContaining({ comment: 'Witty text opener!' })
      );
      expect(decisionsLog.appendDecision).toHaveBeenCalledTimes(1);
      expect(decisionsLog.appendDecision).toHaveBeenCalledWith(
        expect.objectContaining({
          decision: 'like',
          decisionSource: 'text',
          comment: 'Witty text opener!',
          deliveryStatus: 'success'
        }),
        expect.anything()
      );
      expect(profileCache.markVisited).toHaveBeenCalledWith(
        undefined,
        `${settings.latitude},${settings.longitude}`,
        'token-new'
      );
      expect(fsMock.appendFileSync).toHaveBeenCalledTimes(1);
      const benchmarkPayload = fsMock.appendFileSync.mock.calls[0][1] as string;
      const benchmarkEntry = JSON.parse(benchmarkPayload.trim());
      expect(benchmarkEntry.date).toMatch(/^\d{4}-\d{2}-\d{2}$/);
      expect(benchmarkEntry.locations[0]).toEqual(
        expect.objectContaining({
          label: 'Test City',
          scanned: 1,
          likes: 1,
          averageAge: 25,
        })
      );
    });
  });

  describe('maxLikes behavior', () => {
    test('should respect maxLikes=1 and not process more profiles after limit reached', async () => {
      const profile1 = createMockProfile('user1', 'Alice');
      const profile2 = createMockProfile('user2', 'Bob');
      const profile3 = createMockProfile('user3', 'Charlie');

      // Mock getUnvisited to return 3 profiles on first call, 0 on subsequent
      let callCount = 0;
      (profileCache.getUnvisited as jest.Mock).mockImplementation(() => {
        callCount++;
        if (callCount === 1) {
          return [
            { ratingToken: 'token1', subjectId: 'user1', visited: false, profile: profile1 },
            { ratingToken: 'token2', subjectId: 'user2', visited: false, profile: profile2 },
            { ratingToken: 'token3', subjectId: 'user3', visited: false, profile: profile3 }
          ];
        }
        return [];
      });

      mockHinge.sendLike.mockResolvedValue({});

      await run([createMockSettings()], 1); // maxLikes = 1

      // Should only send 1 like, not 3
      expect(mockHinge.sendLike).toHaveBeenCalledTimes(1);
      expect(mockLikeLimiter.getUsed()).toBe(1);

      // Should only mark 1 profile as visited (the one that was liked)
      expect(profileCache.markVisited).toHaveBeenCalledTimes(1);
    });

    test('should respect maxLikes=5 with concurrent processing', async () => {
      // Create 10 profiles but only allow 5 likes
      const profiles = Array.from({ length: 10 }, (_, i) => ({
        ratingToken: `token${i}`,
        subjectId: `user${i}`,
        visited: false,
        profile: createMockProfile(`user${i}`, `User${i}`)
      }));

      let callCount = 0;
      (profileCache.getUnvisited as jest.Mock).mockImplementation(() => {
        callCount++;
        if (callCount === 1) return profiles;
        return [];
      });

      mockHinge.sendLike.mockResolvedValue({});

      await run([createMockSettings()], 5); // maxLikes = 5

      // Should send at most 5 likes (may be less due to early exhaustion check)
      expect(mockHinge.sendLike).toHaveBeenCalledTimes(5);
      expect(mockLikeLimiter.getUsed()).toBe(5);
    });

    test('should skip profiles when limit exhausted without marking as visited', async () => {
      const profile1 = createMockProfile('user1', 'Alice');
      const profile2 = createMockProfile('user2', 'Bob');

      let callCount = 0;
      (profileCache.getUnvisited as jest.Mock).mockImplementation(() => {
        callCount++;
        if (callCount === 1) {
          return [
            { ratingToken: 'token1', subjectId: 'user1', visited: false, profile: profile1 },
            { ratingToken: 'token2', subjectId: 'user2', visited: false, profile: profile2 }
          ];
        }
        return [];
      });

      mockHinge.sendLike.mockResolvedValue({});

      await run([createMockSettings()], 1); // maxLikes = 1

      // Only 1 like should be sent
      expect(mockHinge.sendLike).toHaveBeenCalledTimes(1);

      // Only the liked profile should be marked as visited
      // The second profile should NOT be marked as visited so it can be retried
      expect(profileCache.markVisited).toHaveBeenCalledTimes(1);
    });
  });

  describe('visited property behavior', () => {
    test('should mark profile as visited after successful like', async () => {
      const profile = createMockProfile('user1', 'Alice');

      (profileCache.getUnvisited as jest.Mock).mockReturnValueOnce([
        { ratingToken: 'token1', subjectId: 'user1', visited: false, profile }
      ]).mockReturnValue([]);

      mockHinge.sendLike.mockResolvedValue({});

      await run([createMockSettings()], 10);

      expect(mockHinge.sendLike).toHaveBeenCalledTimes(1);
      expect(profileCache.markVisited).toHaveBeenCalledWith(
        undefined,
        '37.75,-122.44',
        'token1'
      );
    });

    test('should NOT mark profile as visited after failed like (for retry)', async () => {
      const profile = createMockProfile('user1', 'Alice');

      (profileCache.getUnvisited as jest.Mock).mockReturnValueOnce([
        { ratingToken: 'token1', subjectId: 'user1', visited: false, profile }
      ]).mockReturnValue([]);

      // Simulate API error
      mockHinge.sendLike.mockRejectedValue(new Error('API Error'));

      await run([createMockSettings()], 10);

      // Like was attempted
      expect(mockHinge.sendLike).toHaveBeenCalledTimes(1);

      // Profile should NOT be marked as visited (so it can be retried)
      expect(profileCache.markVisited).not.toHaveBeenCalled();

      // Decision should be logged with error status
      expect(decisionsLog.appendDecision).toHaveBeenCalledWith(
        expect.objectContaining({
          deliveryStatus: 'error'
        }),
        expect.anything()
      );
    });

    test('should mark profile as visited when deliberately skipped (validation failed)', async () => {
      const profile = createMockProfile('user1', 'Alice');

      (profileCache.getUnvisited as jest.Mock).mockReturnValueOnce([
        { ratingToken: 'token1', subjectId: 'user1', visited: false, profile }
      ]).mockReturnValue([]);

      // Mock low image scores to fail validation
      (llm.getImageScore as jest.Mock).mockResolvedValue({ score: 0.1 });

      await run([createMockSettings()], 10);

      // No like should be sent (validation failed)
      expect(mockHinge.sendLike).not.toHaveBeenCalled();

      // Profile should be marked as visited (deliberate skip)
      expect(profileCache.markVisited).toHaveBeenCalledWith(
        undefined,
        '37.75,-122.44',
        'token1'
      );

      // Decision should be logged as skip
      expect(decisionsLog.appendDecision).toHaveBeenCalledWith(
        expect.objectContaining({
          decision: 'skip'
        }),
        expect.anything()
      );
    });

    test('should NOT mark profile as visited when skipped due to limit exhaustion', async () => {
      const profile1 = createMockProfile('user1', 'Alice');
      const profile2 = createMockProfile('user2', 'Bob');

      (profileCache.getUnvisited as jest.Mock).mockReturnValueOnce([
        { ratingToken: 'token1', subjectId: 'user1', visited: false, profile: profile1 },
        { ratingToken: 'token2', subjectId: 'user2', visited: false, profile: profile2 }
      ]).mockReturnValue([]);

      mockHinge.sendLike.mockResolvedValue({});

      await run([createMockSettings()], 1); // Only 1 like allowed

      // First profile gets liked
      expect(mockHinge.sendLike).toHaveBeenCalledTimes(1);

      // Only the first profile should be marked as visited
      expect(profileCache.markVisited).toHaveBeenCalledTimes(1);
      expect(profileCache.markVisited).toHaveBeenCalledWith(
        undefined,
        '37.75,-122.44',
        'token1'
      );

      // Second profile should NOT be marked as visited
      expect(profileCache.markVisited).not.toHaveBeenCalledWith(
        undefined,
        '37.75,-122.44',
        'token2'
      );
    });

    test('should handle concurrent profile processing without exceeding limit', async () => {
      // Create exactly 10 profiles (matching concurrency limit)
      const profiles = Array.from({ length: 10 }, (_, i) => ({
        ratingToken: `token${i}`,
        subjectId: `user${i}`,
        visited: false,
        profile: createMockProfile(`user${i}`, `User${i}`)
      }));

      (profileCache.getUnvisited as jest.Mock).mockReturnValueOnce(profiles).mockReturnValue([]);

      mockHinge.sendLike.mockResolvedValue({});

      await run([createMockSettings()], 3); // Only 3 likes allowed, but 10 profiles

      // Should respect the limit
      expect(mockHinge.sendLike).toHaveBeenCalledTimes(3);
      expect(mockLikeLimiter.getUsed()).toBe(3);

      // Only 3 profiles should be marked as visited
      expect(profileCache.markVisited).toHaveBeenCalledTimes(3);
    });

    test('should release like limiter when API call fails', async () => {
      const profile = createMockProfile('user1', 'Alice');

      (profileCache.getUnvisited as jest.Mock).mockReturnValueOnce([
        { ratingToken: 'token1', subjectId: 'user1', visited: false, profile }
      ]).mockReturnValue([]);

      mockHinge.sendLike.mockRejectedValue(new Error('API Error'));

      await run([createMockSettings()], 10);

      // Like was attempted and failed
      expect(mockHinge.sendLike).toHaveBeenCalledTimes(1);

      // Limiter should have been released on error
      expect(mockLikeLimiter.release).toHaveBeenCalled();

      // Profile should not be marked as visited (for retry)
      expect(profileCache.markVisited).not.toHaveBeenCalled();
    });
  });

  describe('empty text lines generation', () => {
    test('should NOT mark profile as visited when generation fails but profile passed validation', async () => {
      const profile = createMockProfile('user1', 'Alice');

      (profileCache.getUnvisited as jest.Mock).mockReturnValueOnce([
        { ratingToken: 'token1', subjectId: 'user1', visited: false, profile }
      ]).mockReturnValue([]);

      // Mock empty Map (no text openers generated)
      (llm.generateOpenersFromYamlBatch as jest.Mock).mockResolvedValue(new Map());
      // Mock image opener generation fails
      (llm.generateImageOpenerFromYaml as jest.Mock).mockResolvedValue('');

      await run([createMockSettings()], 10);

      // No like should be sent (no candidates generated)
      expect(mockHinge.sendLike).not.toHaveBeenCalled();

      // Profile should NOT be marked as visited (left for retry since it passed validation)
      expect(profileCache.markVisited).not.toHaveBeenCalled();

      // Decision should be logged as skip with error status
      expect(decisionsLog.appendDecision).toHaveBeenCalledWith(
        expect.objectContaining({
          decision: 'skip',
          deliveryStatus: 'error',
          deliveryError: expect.objectContaining({
            code: 'GENERATION_FAILED'
          })
        }),
        expect.anything()
      );
    });

    test('should NOT mark profile as visited when Map has wrong IDs', async () => {
      const profile = createMockProfile('user1', 'Alice');

      (profileCache.getUnvisited as jest.Mock).mockReturnValueOnce([
        { ratingToken: 'token1', subjectId: 'user1', visited: false, profile }
      ]).mockReturnValue([]);

      // Mock Map with wrong IDs (not matching "1", "2", etc.)
      (llm.generateOpenersFromYamlBatch as jest.Mock).mockResolvedValue(
        new Map([['wrong-id', 'Great opener!']])
      );
      // Mock image opener generation fails
      (llm.generateImageOpenerFromYaml as jest.Mock).mockResolvedValue('');

      await run([createMockSettings()], 10);

      // No like should be sent (no valid candidates)
      expect(mockHinge.sendLike).not.toHaveBeenCalled();

      // Profile should NOT be marked as visited (left for retry)
      expect(profileCache.markVisited).not.toHaveBeenCalled();

      // Decision should be logged with error status
      expect(decisionsLog.appendDecision).toHaveBeenCalledWith(
        expect.objectContaining({
          decision: 'skip',
          deliveryStatus: 'error'
        }),
        expect.anything()
      );
    });

    test('should NOT mark as visited when API returns unexpected JSON structure', async () => {
      const profile = createMockProfile('user1', 'Alice');

      (profileCache.getUnvisited as jest.Mock).mockReturnValueOnce([
        { ratingToken: 'token1', subjectId: 'user1', visited: false, profile }
      ]).mockReturnValue([]);

      // Simulate the actual bug: API returns empty array or wrong structure
      // This would cause generateOpenersFromYamlBatch to return empty Map
      (llm.generateOpenersFromYamlBatch as jest.Mock).mockResolvedValue(new Map());
      (llm.generateImageOpenerFromYaml as jest.Mock).mockResolvedValue('');

      await run([createMockSettings()], 10);

      expect(mockHinge.sendLike).not.toHaveBeenCalled();
      // Profile should NOT be marked as visited (generation failed temporarily)
      expect(profileCache.markVisited).not.toHaveBeenCalled();
    });

    test('should succeed with at least one valid text opener even if some fail', async () => {
      const profile = createMockProfile('user1', 'Alice');
      (profile.profile.answers as any) = [
        { position: 0, questionId: 'q1', type: 'text', response: 'First answer' },
        { position: 1, questionId: 'q2', type: 'text', response: 'Second answer' },
        { position: 2, questionId: 'q3', type: 'text', response: 'Third answer' }
      ];

      (profileCache.getUnvisited as jest.Mock).mockReturnValueOnce([
        { ratingToken: 'token1', subjectId: 'user1', visited: false, profile }
      ]).mockReturnValue([]);

      // Only return opener for ID "2", missing "1" and "3"
      (llm.generateOpenersFromYamlBatch as jest.Mock).mockResolvedValue(
        new Map([['2', 'Great opener for second question!']])
      );
      (llm.generateImageOpenerFromYaml as jest.Mock).mockResolvedValue('');
      mockHinge.sendLike.mockResolvedValue({});

      await run([createMockSettings()], 10);

      // Should succeed with the one valid opener
      expect(mockHinge.sendLike).toHaveBeenCalledTimes(1);
    });

    test('should still mark as visited when profile fails validation (not generation)', async () => {
      const profile = createMockProfile('user1', 'Alice');

      (profileCache.getUnvisited as jest.Mock).mockReturnValueOnce([
        { ratingToken: 'token1', subjectId: 'user1', visited: false, profile }
      ]).mockReturnValue([]);

      // Mock low image scores to fail validation
      (llm.getImageScore as jest.Mock).mockResolvedValue({ score: 0.1 });

      await run([createMockSettings()], 10);

      // No like should be sent (validation failed)
      expect(mockHinge.sendLike).not.toHaveBeenCalled();

      // Profile SHOULD be marked as visited (deliberate skip due to validation failure)
      expect(profileCache.markVisited).toHaveBeenCalledWith(
        undefined,
        '37.75,-122.44',
        'token1'
      );

      // Decision should be logged as skip with success status (intentional skip)
      expect(decisionsLog.appendDecision).toHaveBeenCalledWith(
        expect.objectContaining({
          decision: 'skip',
          deliveryStatus: 'success'
        }),
        expect.anything()
      );
    });
  });

  describe('LLM refusal handling', () => {
    test('falls back to image like when text generation refuses', async () => {
      const profile = createMockProfile('user1', 'Alice');

      (profileCache.getUnvisited as jest.Mock)
        .mockReturnValueOnce([{ ratingToken: 'token1', subjectId: 'user1', visited: false, profile }])
        .mockReturnValue([]);

      const refusalError = new (llm as any).LLMRefusalError('Model refused');
      refusalError.stopReason = 'refusal';

      (llm.generateOpenersFromYamlBatch as jest.Mock).mockImplementation(() => {
        throw refusalError;
      });
      (llm.generateImageOpenerFromYaml as jest.Mock).mockResolvedValue('Image opener!');
      (llm.bestPl as jest.Mock).mockResolvedValue(0);
      mockHinge.sendLike.mockResolvedValue({});

      await run([createMockSettings()], 5);

      expect(mockHinge.sendLike).toHaveBeenCalledTimes(1);
      expect(profileCache.markVisited).toHaveBeenCalledWith(
        undefined,
        '37.75,-122.44',
        'token1'
      );
      expect(decisionsLog.appendDecision).toHaveBeenCalledWith(
        expect.objectContaining({
          decision: 'like',
          decisionSource: 'image',
          deliveryStatus: 'success'
        }),
        expect.anything()
      );
    });

    test('marks profile as visited when all generations refuse', async () => {
      const profile = createMockProfile('user1', 'Alice');
      const topPhoto = {
        ...profile.profile.photos[0],
        url: 'https://example.com/user1-top.jpg',
        cdnId: 'cdn-top'
      };
      profile.profile.photos.push(topPhoto);

      (profileCache.getUnvisited as jest.Mock)
        .mockReturnValueOnce([{ ratingToken: 'token1', subjectId: 'user1', visited: false, profile }])
        .mockReturnValue([]);

      const refusalError = new (llm as any).LLMRefusalError('Model refused');
      refusalError.stopReason = 'refusal';

      (llm.generateOpenersFromYamlBatch as jest.Mock).mockImplementation(() => {
        throw refusalError;
      });
      (llm.generateImageOpenerFromYaml as jest.Mock).mockImplementation(() => {
        throw refusalError;
      });
      (llm.bestPl as jest.Mock).mockResolvedValue(0);
      (llm.getImageScore as jest.Mock).mockImplementation((url: string) => {
        if (url.includes('top')) {
          return Promise.resolve({ score: 0.95 });
        }
        return Promise.resolve({ score: 0.1 });
      });
      mockHinge.sendLike.mockResolvedValue({});

      await run([createMockSettings()], 5);

      expect(mockHinge.sendLike).toHaveBeenCalledTimes(1);
      const payload = mockHinge.sendLike.mock.calls[0][3];
      expect(payload.comment).toBe('');
      expect(payload.photoData.url).toBe(topPhoto.url);
      expect(mockLikeLimiter.release).not.toHaveBeenCalled();
      expect(profileCache.markVisited).toHaveBeenCalledWith(
        undefined,
        '37.75,-122.44',
        'token1'
      );
      expect(decisionsLog.appendDecision).toHaveBeenCalledWith(
        expect.objectContaining({
          decision: 'like',
          decisionSource: 'image',
          photoUsed: topPhoto.url,
          comment: '',
          deliveryStatus: 'success',
        }),
        expect.anything()
      );
    });
  });

  describe('like reservation and expensive processing', () => {
    test('should NOT do expensive processing when like limit is exhausted', async () => {
      const profile1 = createMockProfile('user1', 'Alice');
      const profile2 = createMockProfile('user2', 'Bob');

      let callCount = 0;
      (profileCache.getUnvisited as jest.Mock).mockImplementation(() => {
        callCount++;
        if (callCount === 1) {
          return [
            { ratingToken: 'token1', subjectId: 'user1', visited: false, profile: profile1 },
            { ratingToken: 'token2', subjectId: 'user2', visited: false, profile: profile2 }
          ];
        }
        return [];
      });

      mockHinge.sendLike.mockResolvedValue({});

      await run([createMockSettings()], 1); // Only 1 like allowed

      // First profile should be processed
      expect(mockHinge.sendLike).toHaveBeenCalledTimes(1);

      // Image scoring should only be called for the first profile (not the second)
      // Since we reserve the like before validation, second profile never gets validated
      expect(llm.getImageScore).toHaveBeenCalled();
      const imageScoreCalls = (llm.getImageScore as jest.Mock).mock.calls.length;

      // Should only score images for the first profile
      expect(imageScoreCalls).toBe(1); // 1 photo for profile1

      // Text generation should only be called once (for first profile)
      expect(llm.generateOpenersFromYamlBatch).toHaveBeenCalledTimes(1);
    });

    test('should release like reservation when validation fails and allow next profile to use it', async () => {
      const profile1 = createMockProfile('user1', 'Alice');
      const profile2 = createMockProfile('user2', 'Bob');

      // Process profiles in TWO batches to avoid concurrency race
      let callCount = 0;
      (profileCache.getUnvisited as jest.Mock).mockImplementation(() => {
        callCount++;
        if (callCount === 1) {
          return [{ ratingToken: 'token1', subjectId: 'user1', visited: false, profile: profile1 }];
        }
        if (callCount === 2) {
          return [{ ratingToken: 'token2', subjectId: 'user2', visited: false, profile: profile2 }];
        }
        return [];
      });

      // First profile fails validation, second profile passes
      let scoreCallCount = 0;
      (llm.getImageScore as jest.Mock).mockImplementation(() => {
        scoreCallCount++;
        const score = scoreCallCount <= 1 ? 0.1 : 0.8; // First call low, rest high
        return Promise.resolve({ score });
      });

      mockHinge.sendLike.mockResolvedValue({});

      await run([createMockSettings()], 1); // Only 1 like allowed

      // First profile fails validation (batch 1), releases reservation
      // Second profile (batch 2) uses the released slot and succeeds
      expect(mockHinge.sendLike).toHaveBeenCalledTimes(1);
      expect(mockHinge.sendLike).toHaveBeenCalledWith(
        'user2',
        'token2',
        expect.anything(),
        expect.anything()
      );

      // Both profiles should have been validated (in separate batches)
      expect(llm.getImageScore).toHaveBeenCalledTimes(2);
    });

    test('should release like reservation when generation fails and allow next profile to use it', async () => {
      const profile1 = createMockProfile('user1', 'Alice');
      const profile2 = createMockProfile('user2', 'Bob');

      // Process profiles in TWO batches to avoid concurrency race
      let callCount = 0;
      (profileCache.getUnvisited as jest.Mock).mockReset().mockImplementation(() => {
        callCount++;
        if (callCount === 1) {
          return [{ ratingToken: 'token1', subjectId: 'user1', visited: false, profile: profile1 }];
        }
        if (callCount === 2) {
          return [{ ratingToken: 'token2', subjectId: 'user2', visited: false, profile: profile2 }];
        }
        return [];
      });

      // First profile: generation returns empty on both retry attempts (fails completely)
      // Second profile: generation succeeds
      let genCallCount = 0;
      (llm.generateOpenersFromYamlBatch as jest.Mock).mockReset().mockImplementation(() => {
        genCallCount++;
        if (genCallCount <=2) {
          return Promise.resolve(new Map()); // Empty for first profile (both retries)
        }
        return Promise.resolve(new Map([['1', 'Great opener!']])); // Success for second profile
      });
      (llm.generateImageOpenerFromYaml as jest.Mock).mockReset().mockResolvedValue(''); // Image fails for both
      mockHinge.sendLike.mockReset().mockResolvedValue({});

      await run([createMockSettings()], 1); // Only 1 like allowed

      // First profile fails generation (batch 1), releases reservation
      // Second profile (batch 2) uses the released slot and succeeds
      expect(mockHinge.sendLike).toHaveBeenCalledTimes(1);
      expect(mockHinge.sendLike).toHaveBeenCalledWith(
        'user2',
        'token2',
        expect.anything(),
        expect.anything()
      );
    });
  });

  describe('edge cases', () => {
    test('should handle empty profile list gracefully', async () => {
      (profileCache.getUnvisited as jest.Mock).mockReturnValue([]);

      await run([createMockSettings()], 10);

      expect(mockHinge.sendLike).not.toHaveBeenCalled();
      expect(profileCache.markVisited).not.toHaveBeenCalled();
    });

    test('should handle maxLikes=0 correctly', async () => {
      const profile = createMockProfile('user1', 'Alice');

      (profileCache.getUnvisited as jest.Mock).mockReturnValueOnce([
        { ratingToken: 'token1', subjectId: 'user1', visited: false, profile }
      ]);

      await run([createMockSettings()], 0);

      expect(mockHinge.sendLike).not.toHaveBeenCalled();
      expect(profileCache.markVisited).not.toHaveBeenCalled();
    });

    test('should continue processing remaining profiles after one fails validation', async () => {
      const profile1 = createMockProfile('user1', 'Alice');
      const profile2 = createMockProfile('user2', 'Bob');

      (profileCache.getUnvisited as jest.Mock)
        .mockReset()
        .mockReturnValueOnce([
          { ratingToken: 'token1', subjectId: 'user1', visited: false, profile: profile1 },
          { ratingToken: 'token2', subjectId: 'user2', visited: false, profile: profile2 }
        ])
        .mockReturnValue([]);

      // Mock image scores based on URL
      // user1's photos get low scores (validation fails)
      // user2's photos get high scores (validation passes)
      (llm.getImageScore as jest.Mock).mockReset().mockImplementation((url: string) => {
        const score = url.includes('user1') ? 0.1 : 0.8;
        return Promise.resolve({ score });
      });

      mockHinge.sendLike.mockReset().mockResolvedValue({});
      (profileCache.markVisited as jest.Mock).mockReset().mockResolvedValue(undefined);

      await run([createMockSettings()], 10);

      // First profile skipped (validation failed), second profile liked
      expect(mockHinge.sendLike).toHaveBeenCalledTimes(1);
      expect(mockHinge.sendLike).toHaveBeenCalledWith(
        'user2',
        'token2',
        expect.anything(),
        expect.anything()
      );

      // Both profiles should be marked as visited (one skipped, one liked)
      expect(profileCache.markVisited).toHaveBeenCalledTimes(2);
    });
  });
});
