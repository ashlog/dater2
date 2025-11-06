/**
 * Training-Free GRPO Types
 *
 * These types follow the terminology from the paper:
 * "Training-Free Group Relative Policy Optimization"
 *
 * Key concepts:
 * - Rollout: A single output from the policy model (o_i)
 * - Reward Signal: Scalar reward from evaluation (r_i = R(q, o_i))
 * - Trajectory Summary: Natural language summary of a rollout (s_i)
 * - Semantic Advantage: Natural language experience extracted from group comparison (A_text)
 * - Experience: Entry in the experience library (E)
 */

/**
 * A single rollout output from the policy model
 * In the paper: o_i from π_θ(o_i|q, E)
 *
 * @template T The type of the output (e.g., GeneratedImage, string, etc.)
 */
export interface Rollout<T = unknown> {
  /** Unique identifier for this rollout */
  id: string;

  /** The generated output (image, text, etc.) */
  output: T;

  /** The prompt used to generate this output (includes experiences) */
  prompt: string;

  /** Optional metadata about the generation */
  metadata?: Record<string, unknown>;
}

/**
 * Reward signal from external evaluation
 * In the paper: r_i = R(q, o_i)
 *
 * For swipe-based evaluation:
 * - Like = +1
 * - Skip = 0
 * - Dislike = -1
 */
export interface RewardSignal {
  /** ID of the rollout this reward is for */
  rolloutId: string;

  /** Scalar reward value */
  reward: number;

  /** Optional text feedback from the evaluator */
  feedback?: string;
}

/**
 * Trajectory summary for group advantage computation
 * In the paper: s_i = M(p_summary, q, o_i)
 *
 * This is the intermediate representation created by summarizing each rollout
 * before extracting semantic advantages.
 */
export interface TrajectorySummary {
  /** ID of the rollout this summary is for */
  rolloutId: string;

  /** Natural language summary of what happened in this rollout */
  summary: string;

  /** Whether this rollout was successful (based on reward) */
  wasSuccessful: boolean;

  /** The reward this rollout received */
  reward: number;
}

/**
 * Semantic advantage - the natural language experience extracted from group comparison
 * In the paper: A_text = M(p_extract, q, s_i, E)
 *
 * This replaces the numerical advantage Â_i in vanilla GRPO.
 * Instead of a gradient signal, we get a natural language insight.
 */
export interface SemanticAdvantage {
  /** The extracted insight */
  insight: string;

  /** Context describing when this experience applies */
  context: string;

  /** IDs of rollouts that contributed to this advantage */
  sourceRolloutIds: string[];
}

/**
 * Experience entry in the library
 * In the paper: entries in E (experiential knowledge)
 *
 * Experiences are the "learned token prior" that guides future generations.
 * They are injected into the prompt to influence the policy's output distribution.
 */
export interface Experience {
  /** Unique identifier (e.g., "G1", "G2", etc.) */
  id: string;

  /** Natural language insight extracted from feedback */
  insight: string;

  /** Context describing when this experience applies */
  context: string;

  /** When this experience was created */
  createdAt: Date;

  /** When this experience was last modified */
  updatedAt: Date;
}

/**
 * Serializable version of Experience (for JSON storage)
 */
export interface SerializedExperience {
  id: string;
  insight: string;
  context: string;
  createdAt: string;
  updatedAt: string;
}

/**
 * Operations for updating the experience library
 * In the paper: Controller operations (ADD, DELETE, MODIFY, KEEP)
 *
 * The controller analyzes semantic advantages and determines how to update
 * the experience library to improve future performance.
 */
export type ExperienceOperation =
  | AddOperation
  | DeleteOperation
  | ModifyOperation
  | MergeOperation
  | KeepOperation;

/** Add a new experience to the library */
export interface AddOperation {
  type: 'add';
  experience: {
    insight: string;
    context: string;
  };
}

/** Delete an experience from the library */
export interface DeleteOperation {
  type: 'delete';
  experienceId: string;
}

/** Modify an existing experience */
export interface ModifyOperation {
  type: 'modify';
  experienceId: string;
  newInsight: string;
}

/** Merge multiple experiences into one */
export interface MergeOperation {
  type: 'merge';
  experienceIds: string[];
  mergedInsight: string;
  mergedContext: string;
}

/** Keep the experience library unchanged */
export interface KeepOperation {
  type: 'keep';
}

/**
 * LLM adapter interface
 *
 * Consumers of the GRPO optimizer must implement this interface to provide
 * the LLM-based operations needed for the algorithm:
 * 1. Summarize trajectories (s_i = M(p_summary, q, o_i))
 * 2. Extract semantic advantages (A_text = M(p_extract, q, s_i, E))
 * 3. Determine experience operations (Controller logic)
 */
export interface LLMAdapter<T = unknown> {
  /**
   * Summarize a single rollout trajectory
   * Paper: s_i = M(p_summary, q, o_i)
   *
   * @param query The original query/task
   * @param rollout The rollout to summarize
   * @param wasSuccessful Whether the rollout was successful
   * @param groundTruth Optional ground truth for comparison
   * @returns Natural language summary of the trajectory
   */
  summarizeTrajectory(
    query: string,
    rollout: Rollout<T>,
    wasSuccessful: boolean,
    groundTruth?: string
  ): Promise<string>;

  /**
   * Extract semantic advantages from trajectory summaries
   * Paper: A_text = M(p_extract, q, s_i, E)
   *
   * This is where the "group relative" comparison happens - comparing
   * successful vs failed trajectories to extract generalizable lessons.
   *
   * @param query The original query/task
   * @param summaries Summaries of all rollouts in the group
   * @param currentExperiences Current experience library
   * @returns Array of semantic advantages (lessons learned)
   */
  extractSemanticAdvantage(
    query: string,
    summaries: TrajectorySummary[],
    currentExperiences: Experience[]
  ): Promise<SemanticAdvantage[]>;

  /**
   * Determine how to update the experience library
   * Paper: Controller determines ADD/DELETE/MODIFY/KEEP
   *
   * Given the semantic advantages from this batch, decide how to
   * update the experience library for future generations.
   *
   * @param semanticAdvantages Advantages extracted from this batch
   * @param currentExperiences Current experience library
   * @param maxOperations Maximum number of operations to perform
   * @returns Array of operations to apply to the experience library
   */
  determineExperienceOperations(
    semanticAdvantages: SemanticAdvantage[],
    currentExperiences: Experience[],
    maxOperations: number
  ): Promise<ExperienceOperation[]>;
}

/**
 * Configuration for the GRPO optimizer
 */
export interface OptimizerConfig<T = unknown> {
  /** Number of rollouts per query (G in the paper) */
  groupSize: number;

  /** Maximum number of experiences to keep in the library */
  maxExperiences: number;

  /** Maximum operations per optimization step */
  maxOperationsPerStep: number;

  /** LLM adapter for summarization and extraction */
  llmAdapter: LLMAdapter<T>;
}

/**
 * Result of processing a group of rollouts
 */
export interface ProcessGroupResult {
  /** Whether optimization was performed (false if std(rewards) = 0) */
  optimized: boolean;

  /** Updated experience library */
  experiences: Experience[];

  /** Operations that were applied */
  operations: ExperienceOperation[];

  /** Semantic advantages extracted (empty if not optimized) */
  semanticAdvantages: SemanticAdvantage[];

  /** Trajectory summaries from the summarization step */
  trajectorySummaries: TrajectorySummary[];

  /** Reason if optimization was skipped */
  skipReason?: string;
}

/**
 * Serialized state of the optimizer (for persistence)
 */
export interface SerializedOptimizerState {
  experiences: SerializedExperience[];
  nextId: number;
  version: string;
}

/**
 * Statistics about the reward distribution in a group
 */
export interface RewardStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  hasVariance: boolean;
}
