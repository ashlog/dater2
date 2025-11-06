/**
 * Training-Free GRPO Optimizer
 *
 * Implements the Training-Free Group Relative Policy Optimization algorithm
 * from the paper. Instead of updating model parameters, this optimizer
 * maintains an experience library that serves as a "learned token prior"
 * to guide future generations.
 *
 * Core algorithm:
 * 1. Generate G rollouts for a query using policy conditioned on experiences
 * 2. Score each rollout with a reward signal
 * 3. If std(rewards) > 0, compute semantic advantages:
 *    a. Summarize each trajectory: s_i = M(p_summary, q, o_i)
 *    b. Extract semantic advantage: A_text = M(p_extract, q, s_i, E)
 * 4. Controller updates experience library E based on A_text
 */

import { ExperienceStore } from './experience-store';
import {
  Rollout,
  RewardSignal,
  TrajectorySummary,
  SemanticAdvantage,
  Experience,
  ExperienceOperation,
  OptimizerConfig,
  ProcessGroupResult,
  RewardStats,
  LLMAdapter,
} from './types';

/**
 * Calculate statistics for reward distribution
 */
function calculateRewardStats(rewards: number[]): RewardStats {
  if (rewards.length === 0) {
    return { mean: 0, std: 0, min: 0, max: 0, hasVariance: false };
  }

  const mean = rewards.reduce((a, b) => a + b, 0) / rewards.length;
  const squaredDiffs = rewards.map((r) => Math.pow(r - mean, 2));
  const variance = squaredDiffs.reduce((a, b) => a + b, 0) / rewards.length;
  const std = Math.sqrt(variance);
  const min = Math.min(...rewards);
  const max = Math.max(...rewards);

  // Paper: "Since Â_i = 0 when all G outputs receive identical rewards
  // (i.e., std(r) = 0), we generate semantic advantages only for groups
  // with both clear winners and losers."
  const hasVariance = std > 0;

  return { mean, std, min, max, hasVariance };
}

/**
 * Training-Free GRPO Optimizer
 *
 * @template T The type of output from the policy model
 */
export class GRPOOptimizer<T = unknown> {
  private config: OptimizerConfig<T>;
  private experienceStore: ExperienceStore;

  constructor(config: OptimizerConfig<T>, initialExperiences?: Experience[]) {
    this.config = config;
    this.experienceStore = initialExperiences
      ? ExperienceStore.fromExperiences(initialExperiences, config.maxExperiences)
      : new ExperienceStore(config.maxExperiences);
  }

  /**
   * Build a prompt with current experiences injected
   *
   * Paper: π_θ(o_i|q, E) - the experiences E are injected into the prompt
   * to influence the policy's output distribution.
   *
   * @param basePrompt The original user prompt/query
   * @returns Enhanced prompt with experiences
   */
  buildPromptWithExperiences(basePrompt: string): string {
    if (this.experienceStore.size() === 0) {
      return basePrompt;
    }

    // Format experiences, handling JSON insights specially
    const experiences = this.experienceStore.getAll();
    const formattedExperiences = experiences
      .map((exp) => {
        // Try to parse as JSON for better formatting
        try {
          const parsed = JSON.parse(exp.insight);
          return `[${exp.id}] Generation Parameters:\n${JSON.stringify(parsed, null, 2)}`;
        } catch {
          // Not JSON, use as-is
          return `[${exp.id}] ${exp.insight}`;
        }
      })
      .join('\n\n');

    return `${basePrompt}

When generating, you MUST apply these learned parameters from previous feedback:

<generation_parameters>
${formattedExperiences}
</generation_parameters>

Use these parameters to guide your output. Follow the composition, style, subjects, and avoid patterns specified.`;
  }

  /**
   * Process a group of rollouts with their rewards
   *
   * This is the core optimization step from the paper:
   * 1. Check if std(rewards) > 0 (otherwise skip)
   * 2. Summarize each trajectory: s_i = M(p_summary, q, o_i)
   * 3. Extract semantic advantage: A_text = M(p_extract, q, s_i, E)
   * 4. Controller determines operations and updates E
   *
   * @param query The original query/task
   * @param rollouts The generated rollouts
   * @param rewards Reward signals for each rollout
   * @param groundTruth Optional ground truth for comparison
   * @returns Result including updated experiences and operations applied
   */
  async processGroup(
    query: string,
    rollouts: Rollout<T>[],
    rewards: RewardSignal[],
    groundTruth?: string
  ): Promise<ProcessGroupResult> {
    // Validate inputs
    if (rollouts.length === 0) {
      return {
        optimized: false,
        experiences: this.getExperiences(),
        operations: [],
        semanticAdvantages: [],
        trajectorySummaries: [],
        skipReason: 'No rollouts provided',
      };
    }

    if (rollouts.length !== rewards.length) {
      throw new Error(
        `Mismatch: ${rollouts.length} rollouts but ${rewards.length} rewards`
      );
    }

    // Create reward map for easy lookup
    const rewardMap = new Map(rewards.map((r) => [r.rolloutId, r]));

    // Calculate reward statistics
    const rewardValues = rewards.map((r) => r.reward);
    const stats = calculateRewardStats(rewardValues);

    // Paper: Skip if std(rewards) = 0
    if (!stats.hasVariance) {
      return {
        optimized: false,
        experiences: this.getExperiences(),
        operations: [],
        semanticAdvantages: [],
        trajectorySummaries: [],
        skipReason: `All rollouts received identical rewards (mean=${stats.mean}, std=0)`,
      };
    }

    // Step 1: Summarize each trajectory
    // Paper: s_i = M(p_summary, q, o_i)
    const summaries: TrajectorySummary[] = await Promise.all(
      rollouts.map(async (rollout) => {
        const reward = rewardMap.get(rollout.id);
        if (!reward) {
          throw new Error(`No reward found for rollout ${rollout.id}`);
        }

        // Determine success based on reward relative to mean
        const wasSuccessful = reward.reward > stats.mean;

        const summary = await this.config.llmAdapter.summarizeTrajectory(
          query,
          rollout,
          wasSuccessful,
          groundTruth
        );

        return {
          rolloutId: rollout.id,
          summary,
          wasSuccessful,
          reward: reward.reward,
        };
      })
    );

    // Step 2: Extract semantic advantages
    // Paper: A_text = M(p_extract, q, s_i, E)
    const currentExperiences = this.getExperiences();
    const semanticAdvantages = await this.config.llmAdapter.extractSemanticAdvantage(
      query,
      summaries,
      currentExperiences
    );

    // Step 3: Determine experience operations
    // Paper: Controller determines ADD/DELETE/MODIFY/KEEP
    const operations = await this.config.llmAdapter.determineExperienceOperations(
      semanticAdvantages,
      currentExperiences,
      this.config.maxOperationsPerStep
    );

    // Step 4: Apply operations to experience store
    console.log(`[GRPO-OPTIMIZER] About to call applyOperations with ${operations.length} operations`);
    this.experienceStore.applyOperations(operations);
    console.log(`[GRPO-OPTIMIZER] applyOperations completed`);

    return {
      optimized: true,
      experiences: this.getExperiences(),
      operations,
      semanticAdvantages,
      trajectorySummaries: summaries,
    };
  }

  /**
   * Get all current experiences
   */
  getExperiences(): Experience[] {
    return this.experienceStore.getAll();
  }

  /**
   * Get the number of experiences
   */
  getExperienceCount(): number {
    return this.experienceStore.size();
  }

  /**
   * Get experiences formatted for prompt
   */
  getFormattedExperiences(): string {
    return this.experienceStore.formatForPrompt();
  }

  /**
   * Clear all experiences
   */
  clearExperiences(): void {
    this.experienceStore.clear();
  }

  /**
   * Import experiences directly
   */
  importExperiences(experiences: Array<{ insight: string; context: string }>): void {
    this.experienceStore.importExperiences(experiences);
  }

  /**
   * Serialize optimizer state for persistence
   */
  serialize(): string {
    return this.experienceStore.serialize();
  }

  /**
   * Create optimizer from serialized state
   */
  static deserialize<T>(
    json: string,
    config: OptimizerConfig<T>
  ): GRPOOptimizer<T> {
    const store = ExperienceStore.deserialize(json, config.maxExperiences);
    const optimizer = new GRPOOptimizer<T>(config);
    // Replace the store
    (optimizer as unknown as { experienceStore: ExperienceStore }).experienceStore = store;
    return optimizer;
  }

  /**
   * Get configuration
   */
  getConfig(): OptimizerConfig<T> {
    return this.config;
  }
}

/**
 * Utility function to create reward signals from swipe actions
 */
export function swipesToRewards(
  swipes: Array<{ rolloutId: string; action: 'like' | 'dislike' | 'skip' }>
): RewardSignal[] {
  const rewardMap: Record<string, number> = {
    like: 1,
    skip: 0,
    dislike: -1,
  };

  return swipes.map((swipe) => ({
    rolloutId: swipe.rolloutId,
    reward: rewardMap[swipe.action],
  }));
}

/**
 * Check if a group has variance in rewards
 */
export function hasRewardVariance(rewards: RewardSignal[]): boolean {
  const values = rewards.map((r) => r.reward);
  return calculateRewardStats(values).hasVariance;
}
