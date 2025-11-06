import { NextResponse } from 'next/server';
import {
  GRPOOptimizer,
  swipesToRewards,
  hasRewardVariance,
  Rollout,
} from '@webworks/grpo-optimizer';
import {
  readOptimizerState,
  writeOptimizerState,
  getRecentTrainingBatch,
} from '@/lib/data';
import { pickupLineLLMAdapter, PickupLineOutput } from '@/lib/grpo-adapter';
import { TrainingDataEntry, Experience } from '@/lib/types';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const batchSize = body.batchSize || 20;

    // Get recent training data
    const trainingBatch = getRecentTrainingBatch(batchSize);

    if (trainingBatch.length < 2) {
      return NextResponse.json({
        success: false,
        message: 'Need at least 2 rated entries to train',
        trainingBatchSize: trainingBatch.length,
      });
    }

    // Get current optimizer state
    const currentState = readOptimizerState();

    // Create optimizer with current experiences
    const optimizer = new GRPOOptimizer<PickupLineOutput>({
      groupSize: trainingBatch.length,
      maxExperiences: 20,
      maxOperationsPerStep: 3,
      llmAdapter: pickupLineLLMAdapter,
    });

    // Load existing experiences
    for (const exp of currentState.experiences) {
      optimizer.getExperiences().push({
        ...exp,
        createdAt: new Date(exp.createdAt),
        updatedAt: new Date(exp.updatedAt),
      });
    }

    // Convert training data to rollouts and rewards
    const rollouts: Rollout<PickupLineOutput>[] = trainingBatch.map((entry, i) => ({
      id: `rollout_${i}`,
      output: {
        text: entry.originalDecision.comment || '',
        source: entry.originalDecision.decisionSource || 'text',
        promptUsed: entry.originalDecision.prompts
          .map(p => `${p.question}: ${p.answer}`)
          .join('\n'),
      },
      prompt: entry.originalDecision.prompts
        .map(p => `${p.question}: ${p.answer}`)
        .join('\n'),
      metadata: {
        userId: entry.originalDecision.userId,
        herName: entry.originalDecision.profile?.firstName || 'Unknown',
      },
    }));

    // Convert human ratings to reward signals
    const swipes = rollouts.map((r, i) => ({
      rolloutId: r.id,
      action: trainingBatch[i].humanRating === 'like' ? 'like' as const : 'dislike' as const,
    }));
    const rewardSignals = swipesToRewards(swipes);

    // Check if there's variance in rewards (needed for optimization)
    if (!hasRewardVariance(rewardSignals)) {
      return NextResponse.json({
        success: false,
        message: 'All ratings are the same - need variance to learn',
        likes: swipes.filter(s => s.action === 'like').length,
        dislikes: swipes.filter(s => s.action === 'dislike').length,
      });
    }

    // Build query from first entry (representative)
    const query = trainingBatch[0].originalDecision.prompts
      .map(p => `${p.question}: ${p.answer}`)
      .join('\n');

    // Process the group with GRPO
    const result = await optimizer.processGroup(query, rollouts, rewardSignals);

    // Serialize experiences for storage
    const serializedExperiences: Experience[] = result.experiences.map(exp => ({
      id: exp.id,
      insight: exp.insight,
      context: exp.context,
      createdAt: exp.createdAt.toISOString(),
      updatedAt: exp.updatedAt.toISOString(),
    }));

    // Update optimizer state
    const newState = {
      experiences: serializedExperiences,
      lastTrainedAt: new Date().toISOString(),
      totalTrainingBatches: currentState.totalTrainingBatches + 1,
    };
    writeOptimizerState(newState);

    return NextResponse.json({
      success: true,
      optimized: result.optimized,
      operationsApplied: result.operations.length,
      operations: result.operations,
      newExperienceCount: serializedExperiences.length,
      experiences: serializedExperiences,
      trajectorySummaries: result.trajectorySummaries,
      semanticAdvantages: result.semanticAdvantages,
      trainingBatchSize: trainingBatch.length,
      skipReason: result.skipReason,
    });
  } catch (error) {
    console.error('Error training GRPO:', error);
    return NextResponse.json(
      { error: 'Failed to train GRPO optimizer', details: String(error) },
      { status: 500 }
    );
  }
}
