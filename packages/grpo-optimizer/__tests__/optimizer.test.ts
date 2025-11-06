import {
  GRPOOptimizer,
  swipesToRewards,
  hasRewardVariance,
} from '../src/optimizer';
import {
  LLMAdapter,
  Rollout,
  RewardSignal,
  TrajectorySummary,
  Experience,
  SemanticAdvantage,
  ExperienceOperation,
  OptimizerConfig,
} from '../src/types';

// Mock LLM adapter for testing
class MockLLMAdapter implements LLMAdapter<string> {
  summarizeCalls: Array<{
    query: string;
    rollout: Rollout<string>;
    wasSuccessful: boolean;
  }> = [];

  extractCalls: Array<{
    query: string;
    summaries: TrajectorySummary[];
    experiences: Experience[];
  }> = [];

  operationCalls: Array<{
    advantages: SemanticAdvantage[];
    experiences: Experience[];
    maxOps: number;
  }> = [];

  // Configurable responses
  summaryResponse = 'Mocked trajectory summary';
  advantagesResponse: SemanticAdvantage[] = [];
  operationsResponse: ExperienceOperation[] = [];

  async summarizeTrajectory(
    query: string,
    rollout: Rollout<string>,
    wasSuccessful: boolean
  ): Promise<string> {
    this.summarizeCalls.push({ query, rollout, wasSuccessful });
    return `${this.summaryResponse} for ${rollout.id} (success: ${wasSuccessful})`;
  }

  async extractSemanticAdvantage(
    query: string,
    summaries: TrajectorySummary[],
    currentExperiences: Experience[]
  ): Promise<SemanticAdvantage[]> {
    this.extractCalls.push({
      query,
      summaries,
      experiences: currentExperiences,
    });
    return this.advantagesResponse;
  }

  async determineExperienceOperations(
    semanticAdvantages: SemanticAdvantage[],
    currentExperiences: Experience[],
    maxOperations: number
  ): Promise<ExperienceOperation[]> {
    this.operationCalls.push({
      advantages: semanticAdvantages,
      experiences: currentExperiences,
      maxOps: maxOperations,
    });
    return this.operationsResponse;
  }
}

describe('GRPOOptimizer', () => {
  let mockAdapter: MockLLMAdapter;
  let config: OptimizerConfig<string>;
  let optimizer: GRPOOptimizer<string>;

  beforeEach(() => {
    mockAdapter = new MockLLMAdapter();
    config = {
      groupSize: 4,
      maxExperiences: 50,
      maxOperationsPerStep: 3,
      llmAdapter: mockAdapter,
    };
    optimizer = new GRPOOptimizer<string>(config);
  });

  describe('buildPromptWithExperiences', () => {
    it('should return base prompt when no experiences', () => {
      const prompt = optimizer.buildPromptWithExperiences('Generate an ad');
      expect(prompt).toBe('Generate an ad');
    });

    it('should inject experiences into prompt', () => {
      optimizer.importExperiences([
        { lesson: 'Use bright colors', context: 'ads' },
        { lesson: 'Include call to action', context: 'ads' },
      ]);

      const prompt = optimizer.buildPromptWithExperiences('Generate an ad');

      expect(prompt).toContain('Generate an ad');
      expect(prompt).toContain('<generation_parameters>');
      expect(prompt).toContain('[G1] Use bright colors');
      expect(prompt).toContain('[G2] Include call to action');
      expect(prompt).toContain('</generation_parameters>');
    });
  });

  describe('processGroup', () => {
    it('should skip optimization when no rollouts provided', async () => {
      const result = await optimizer.processGroup('query', [], []);

      expect(result.optimized).toBe(false);
      expect(result.skipReason).toContain('No rollouts');
    });

    it('should throw when rollout/reward count mismatch', async () => {
      const rollouts: Rollout<string>[] = [
        { id: 'r1', output: 'output1', prompt: 'prompt' },
      ];
      const rewards: RewardSignal[] = [
        { rolloutId: 'r1', reward: 1 },
        { rolloutId: 'r2', reward: -1 },
      ];

      await expect(
        optimizer.processGroup('query', rollouts, rewards)
      ).rejects.toThrow('Mismatch');
    });

    it('should skip when all rewards are identical (std=0)', async () => {
      const rollouts: Rollout<string>[] = [
        { id: 'r1', output: 'output1', prompt: 'prompt' },
        { id: 'r2', output: 'output2', prompt: 'prompt' },
        { id: 'r3', output: 'output3', prompt: 'prompt' },
      ];
      const rewards: RewardSignal[] = [
        { rolloutId: 'r1', reward: 1 },
        { rolloutId: 'r2', reward: 1 },
        { rolloutId: 'r3', reward: 1 },
      ];

      const result = await optimizer.processGroup('query', rollouts, rewards);

      expect(result.optimized).toBe(false);
      expect(result.skipReason).toContain('identical rewards');
      expect(mockAdapter.summarizeCalls.length).toBe(0);
    });

    it('should process group with variance in rewards', async () => {
      const rollouts: Rollout<string>[] = [
        { id: 'r1', output: 'good output', prompt: 'prompt' },
        { id: 'r2', output: 'bad output', prompt: 'prompt' },
      ];
      const rewards: RewardSignal[] = [
        { rolloutId: 'r1', reward: 1 },
        { rolloutId: 'r2', reward: -1 },
      ];

      mockAdapter.advantagesResponse = [
        {
          lesson: 'Good outputs have quality X',
          context: 'generation',
          sourceRolloutIds: ['r1', 'r2'],
        },
      ];
      mockAdapter.operationsResponse = [
        {
          type: 'add',
          experience: {
            lesson: 'Good outputs have quality X',
            context: 'generation',
          },
        },
      ];

      const result = await optimizer.processGroup('query', rollouts, rewards);

      expect(result.optimized).toBe(true);
      expect(result.operations.length).toBe(1);
      expect(result.semanticAdvantages.length).toBe(1);
    });

    it('should call LLM adapter for summarization', async () => {
      const rollouts: Rollout<string>[] = [
        { id: 'r1', output: 'output1', prompt: 'prompt' },
        { id: 'r2', output: 'output2', prompt: 'prompt' },
      ];
      const rewards: RewardSignal[] = [
        { rolloutId: 'r1', reward: 1 },
        { rolloutId: 'r2', reward: -1 },
      ];

      await optimizer.processGroup('test query', rollouts, rewards);

      expect(mockAdapter.summarizeCalls.length).toBe(2);
      expect(mockAdapter.summarizeCalls[0].query).toBe('test query');
      expect(mockAdapter.summarizeCalls[0].rollout.id).toBe('r1');
      expect(mockAdapter.summarizeCalls[0].wasSuccessful).toBe(true);
      expect(mockAdapter.summarizeCalls[1].wasSuccessful).toBe(false);
    });

    it('should update experiences based on operations', async () => {
      const rollouts: Rollout<string>[] = [
        { id: 'r1', output: 'good', prompt: 'p' },
        { id: 'r2', output: 'bad', prompt: 'p' },
      ];
      const rewards: RewardSignal[] = [
        { rolloutId: 'r1', reward: 1 },
        { rolloutId: 'r2', reward: -1 },
      ];

      mockAdapter.operationsResponse = [
        {
          type: 'add',
          experience: { lesson: 'New lesson', context: 'test' },
        },
      ];

      expect(optimizer.getExperienceCount()).toBe(0);

      await optimizer.processGroup('query', rollouts, rewards);

      expect(optimizer.getExperienceCount()).toBe(1);
      expect(optimizer.getExperiences()[0].lesson).toBe('New lesson');
    });
  });

  describe('experience management', () => {
    it('should import experiences', () => {
      optimizer.importExperiences([
        { lesson: 'Lesson 1', context: 'ctx1' },
        { lesson: 'Lesson 2', context: 'ctx2' },
      ]);

      expect(optimizer.getExperienceCount()).toBe(2);
    });

    it('should clear experiences', () => {
      optimizer.importExperiences([{ lesson: 'L', context: 'c' }]);
      expect(optimizer.getExperienceCount()).toBe(1);

      optimizer.clearExperiences();
      expect(optimizer.getExperienceCount()).toBe(0);
    });

    it('should get formatted experiences', () => {
      optimizer.importExperiences([{ lesson: 'Test lesson', context: 'test' }]);

      const formatted = optimizer.getFormattedExperiences();
      expect(formatted).toContain('[G1] Test lesson');
    });
  });

  describe('serialization', () => {
    it('should serialize and deserialize', () => {
      optimizer.importExperiences([
        { lesson: 'Lesson 1', context: 'ctx1' },
        { lesson: 'Lesson 2', context: 'ctx2' },
      ]);

      const serialized = optimizer.serialize();
      const restored = GRPOOptimizer.deserialize<string>(serialized, config);

      expect(restored.getExperienceCount()).toBe(2);
      expect(restored.getExperiences()[0].lesson).toBe('Lesson 1');
    });
  });

  describe('with initial experiences', () => {
    it('should initialize with provided experiences', () => {
      const initialExp: Experience[] = [
        {
          id: 'G1',
          lesson: 'Existing lesson',
          context: 'existing',
          createdAt: new Date(),
          updatedAt: new Date(),
        },
      ];

      const optWithExp = new GRPOOptimizer<string>(config, initialExp);

      expect(optWithExp.getExperienceCount()).toBe(1);
      expect(optWithExp.getExperiences()[0].lesson).toBe('Existing lesson');
    });
  });
});

describe('swipesToRewards', () => {
  it('should convert swipes to rewards', () => {
    const swipes = [
      { rolloutId: 'r1', action: 'like' as const },
      { rolloutId: 'r2', action: 'skip' as const },
      { rolloutId: 'r3', action: 'dislike' as const },
    ];

    const rewards = swipesToRewards(swipes);

    expect(rewards).toEqual([
      { rolloutId: 'r1', reward: 1 },
      { rolloutId: 'r2', reward: 0 },
      { rolloutId: 'r3', reward: -1 },
    ]);
  });
});

describe('hasRewardVariance', () => {
  it('should return false for identical rewards', () => {
    const rewards: RewardSignal[] = [
      { rolloutId: 'r1', reward: 1 },
      { rolloutId: 'r2', reward: 1 },
      { rolloutId: 'r3', reward: 1 },
    ];

    expect(hasRewardVariance(rewards)).toBe(false);
  });

  it('should return true for varied rewards', () => {
    const rewards: RewardSignal[] = [
      { rolloutId: 'r1', reward: 1 },
      { rolloutId: 'r2', reward: 0 },
      { rolloutId: 'r3', reward: -1 },
    ];

    expect(hasRewardVariance(rewards)).toBe(true);
  });

  it('should return true even with small variance', () => {
    const rewards: RewardSignal[] = [
      { rolloutId: 'r1', reward: 1 },
      { rolloutId: 'r2', reward: 1 },
      { rolloutId: 'r3', reward: 0 },
    ];

    expect(hasRewardVariance(rewards)).toBe(true);
  });
});
