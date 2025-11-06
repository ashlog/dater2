/**
 * Training-Free GRPO (Group Relative Policy Optimization)
 *
 * A cost-effective method to enhance LLM agent performance without parameter updates.
 * Instead of gradient-based learning, this optimizer maintains an experience library
 * that serves as a "learned token prior" to guide future generations.
 *
 * Based on the paper: "Training-Free Group Relative Policy Optimization"
 *
 * @example
 * ```typescript
 * import { GRPOOptimizer, LLMAdapter, swipesToRewards } from '@webworks/grpo-optimizer';
 *
 * // Implement your LLM adapter
 * const adapter: LLMAdapter<MyOutputType> = {
 *   summarizeTrajectory: async (query, rollout, wasSuccessful) => { ... },
 *   extractSemanticAdvantage: async (query, summaries, experiences) => { ... },
 *   determineExperienceOperations: async (advantages, experiences, maxOps) => { ... },
 * };
 *
 * // Create optimizer
 * const optimizer = new GRPOOptimizer({
 *   groupSize: 4,
 *   maxExperiences: 50,
 *   maxOperationsPerStep: 3,
 *   llmAdapter: adapter,
 * });
 *
 * // Build prompt with experiences
 * const enhancedPrompt = optimizer.buildPromptWithExperiences(basePrompt);
 *
 * // Generate rollouts using your policy model...
 *
 * // Process feedback
 * const rewards = swipesToRewards(userSwipes);
 * const result = await optimizer.processGroup(query, rollouts, rewards);
 *
 * // Get updated experiences for next round
 * const experiences = optimizer.getExperiences();
 * ```
 */

// Core optimizer
export { GRPOOptimizer, swipesToRewards, hasRewardVariance } from './optimizer';

// Experience store (for advanced use cases)
export { ExperienceStore } from './experience-store';

// Types
export type {
  // Core types
  Rollout,
  RewardSignal,
  TrajectorySummary,
  SemanticAdvantage,
  Experience,
  SerializedExperience,

  // Operation types
  ExperienceOperation,
  AddOperation,
  DeleteOperation,
  ModifyOperation,
  MergeOperation,
  KeepOperation,

  // Config and adapter
  LLMAdapter,
  OptimizerConfig,

  // Results
  ProcessGroupResult,
  RewardStats,
  SerializedOptimizerState,
} from './types';
