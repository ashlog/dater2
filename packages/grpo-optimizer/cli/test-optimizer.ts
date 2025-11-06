#!/usr/bin/env tsx
/**
 * CLI for testing the GRPO Optimizer
 *
 * Usage:
 *   npx tsx cli/test-optimizer.ts test-store     # Test experience store
 *   npx tsx cli/test-optimizer.ts test-optimizer # Test optimizer with mock LLM
 *   npx tsx cli/test-optimizer.ts simulate       # Simulate swipe session
 */

import {
  GRPOOptimizer,
  ExperienceStore,
  swipesToRewards,
  hasRewardVariance,
  LLMAdapter,
  Rollout,
  TrajectorySummary,
  Experience,
  SemanticAdvantage,
  ExperienceOperation,
} from '../src';

// Simple mock LLM adapter for testing
class MockLLMAdapter implements LLMAdapter<string> {
  async summarizeTrajectory(
    query: string,
    rollout: Rollout<string>,
    wasSuccessful: boolean
  ): Promise<string> {
    return `Summary of ${rollout.id}: ${wasSuccessful ? 'Success' : 'Failure'} - Output was "${rollout.output.substring(0, 50)}..."`;
  }

  async extractSemanticAdvantage(
    query: string,
    summaries: TrajectorySummary[],
    currentExperiences: Experience[]
  ): Promise<SemanticAdvantage[]> {
    const successful = summaries.filter((s) => s.wasSuccessful);
    const failed = summaries.filter((s) => !s.wasSuccessful);

    if (successful.length === 0 || failed.length === 0) {
      return [];
    }

    return [
      {
        lesson: `Successful outputs tend to be preferred. Consider patterns from successful examples.`,
        context: 'general',
        sourceRolloutIds: summaries.map((s) => s.rolloutId),
      },
    ];
  }

  async determineExperienceOperations(
    semanticAdvantages: SemanticAdvantage[],
    currentExperiences: Experience[],
    maxOperations: number
  ): Promise<ExperienceOperation[]> {
    if (semanticAdvantages.length === 0) {
      return [{ type: 'keep' }];
    }

    // Add each semantic advantage as a new experience
    return semanticAdvantages.slice(0, maxOperations).map((adv) => ({
      type: 'add' as const,
      experience: {
        lesson: adv.lesson,
        context: adv.context,
      },
    }));
  }
}

function testExperienceStore() {
  console.log('\n=== Testing Experience Store ===\n');

  const store = new ExperienceStore(5);
  console.log('Created store with max 5 experiences');

  // Add experiences
  store.applyOperations([
    {
      type: 'add',
      experience: { lesson: 'Use bright colors for attention', context: 'ads' },
    },
    {
      type: 'add',
      experience: { lesson: 'Include a clear call to action', context: 'ads' },
    },
    {
      type: 'add',
      experience: { lesson: 'Keep text minimal', context: 'ads' },
    },
  ]);

  console.log(`Added 3 experiences. Current count: ${store.size()}`);
  console.log('\nFormatted for prompt:');
  console.log(store.formatForPrompt());

  // Test serialization
  const serialized = store.serialize();
  console.log('\nSerialized state:', serialized.substring(0, 200) + '...');

  const restored = ExperienceStore.deserialize(serialized);
  console.log(`Restored store with ${restored.size()} experiences`);

  // Test modify
  store.applyOperations([
    {
      type: 'modify',
      experienceId: 'G1',
      newLesson: 'Use bright, contrasting colors for maximum attention',
    },
  ]);
  console.log('\nModified G1:');
  console.log(store.get('G1')?.lesson);

  // Test limit
  console.log('\n--- Testing limit ---');
  for (let i = 0; i < 5; i++) {
    store.applyOperations([
      {
        type: 'add',
        experience: { lesson: `Extra lesson ${i + 1}`, context: 'test' },
      },
    ]);
  }
  console.log(`After adding 5 more, count: ${store.size()} (max is 5)`);
  console.log('Remaining experiences:');
  store.getAll().forEach((exp) => console.log(`  ${exp.id}: ${exp.lesson}`));

  console.log('\n=== Experience Store Tests Complete ===\n');
}

function testOptimizer() {
  console.log('\n=== Testing Optimizer ===\n');

  const adapter = new MockLLMAdapter();
  const optimizer = new GRPOOptimizer<string>({
    groupSize: 4,
    maxExperiences: 50,
    maxOperationsPerStep: 3,
    llmAdapter: adapter,
  });

  console.log('Created optimizer with mock LLM adapter');

  // Test prompt building
  console.log('\n--- Testing prompt building ---');
  let prompt = optimizer.buildPromptWithExperiences('Generate a luxury watch ad');
  console.log('Prompt (no experiences):');
  console.log(prompt.substring(0, 100));

  optimizer.importExperiences([
    { lesson: 'Use elegant fonts', context: 'luxury' },
    { lesson: 'Emphasize craftsmanship', context: 'watches' },
  ]);

  prompt = optimizer.buildPromptWithExperiences('Generate a luxury watch ad');
  console.log('\nPrompt (with experiences):');
  console.log(prompt);

  console.log('\n=== Optimizer Tests Complete ===\n');
}

async function simulateSession() {
  console.log('\n=== Simulating Swipe Session ===\n');

  const adapter = new MockLLMAdapter();
  const optimizer = new GRPOOptimizer<string>({
    groupSize: 4,
    maxExperiences: 50,
    maxOperationsPerStep: 3,
    llmAdapter: adapter,
  });

  const basePrompt = 'Generate a luxury watch advertisement';

  for (let round = 1; round <= 3; round++) {
    console.log(`\n--- Round ${round} ---`);

    // Build prompt with experiences
    const prompt = optimizer.buildPromptWithExperiences(basePrompt);
    console.log(`Prompt includes ${optimizer.getExperienceCount()} experiences`);

    // Simulate rollouts (in real usage, these would be generated by Gemini)
    const rollouts: Rollout<string>[] = [
      { id: `r${round}_1`, output: 'Elegant watch on marble surface', prompt },
      { id: `r${round}_2`, output: 'Watch with busy background', prompt },
      { id: `r${round}_3`, output: 'Minimalist watch closeup', prompt },
      { id: `r${round}_4`, output: 'Watch in outdoor setting', prompt },
    ];

    // Simulate user swipes
    const swipes = [
      { rolloutId: `r${round}_1`, action: 'like' as const },
      { rolloutId: `r${round}_2`, action: 'dislike' as const },
      { rolloutId: `r${round}_3`, action: 'like' as const },
      { rolloutId: `r${round}_4`, action: 'skip' as const },
    ];

    console.log('Simulated swipes:', swipes.map((s) => `${s.rolloutId}: ${s.action}`).join(', '));

    const rewards = swipesToRewards(swipes);
    console.log('Has variance:', hasRewardVariance(rewards));

    // Process the group
    const result = await optimizer.processGroup(basePrompt, rollouts, rewards);

    console.log('Optimized:', result.optimized);
    if (result.skipReason) {
      console.log('Skip reason:', result.skipReason);
    } else {
      console.log('Operations applied:', result.operations.length);
      console.log('Semantic advantages:', result.semanticAdvantages.length);
    }
    console.log('Total experiences:', optimizer.getExperienceCount());
  }

  console.log('\n--- Final Experiences ---');
  console.log(optimizer.getFormattedExperiences());

  console.log('\n=== Simulation Complete ===\n');
}

// Main
async function main() {
  const command = process.argv[2] || 'all';

  switch (command) {
    case 'test-store':
      testExperienceStore();
      break;
    case 'test-optimizer':
      testOptimizer();
      break;
    case 'simulate':
      await simulateSession();
      break;
    case 'all':
    default:
      testExperienceStore();
      testOptimizer();
      await simulateSession();
      break;
  }
}

main().catch(console.error);
