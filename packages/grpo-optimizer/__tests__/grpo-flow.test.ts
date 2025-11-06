/**
 * Unit tests to verify the GRPO flow and operation handling
 *
 * The key insight: Having ONE consolidated experience that gets refined
 * over time is the CORRECT behavior for comprehensive JSON parameters.
 *
 * What matters is that operations work correctly:
 * - ADD: Creates new experiences
 * - MODIFY: Updates existing experiences
 * - DELETE: Removes experiences
 * - MERGE: Combines multiple experiences
 * - KEEP: No changes
 */

import {
  GRPOOptimizer,
  ExperienceStore,
  swipesToRewards,
  LLMAdapter,
  Rollout,
  TrajectorySummary,
  Experience,
  SemanticAdvantage,
  ExperienceOperation,
} from '../src';

/**
 * MERMAID DIAGRAM - Correct Flow (Consolidated & Refined)
 *
 * ```mermaid
 * flowchart TD
 *     subgraph "Round 1 - No experiences"
 *         A1[Generate Images] --> B1[Feedback: L,D,L,D]
 *         B1 --> C1[Extract Insights]
 *         C1 --> D1{determineOperations}
 *         D1 --> |"ADD"| E1["Library: [G1]"]
 *     end
 *
 *     subgraph "Round 2 - Has G1"
 *         E1 --> A2[Generate with G1]
 *         A2 --> B2[Feedback: L,D,L,D]
 *         B2 --> C2[Extract New Insights]
 *         C2 --> D2{determineOperations}
 *         D2 --> |"MODIFY G1"| E2["Library: [G1' refined]"]
 *     end
 *
 *     subgraph "Round 3 - Has G1'"
 *         E2 --> A3[Generate with G1']
 *         A3 --> B3[Feedback: L,D,L,D]
 *         B3 --> C3[Extract New Insights]
 *         C3 --> D3{determineOperations}
 *         D3 --> |"ADD new aspect"| E3["Library: [G1', G2]"]
 *     end
 *
 *     style E1 fill:#69db7c
 *     style E2 fill:#69db7c
 *     style E3 fill:#69db7c
 * ```
 *
 * This is CORRECT behavior:
 * - Round 1: ADD initial experience
 * - Round 2: MODIFY to refine based on new feedback
 * - Round 3: ADD if genuinely new aspect discovered
 */

// Mock adapter that simulates realistic LLM behavior
class RealisticLLMAdapter implements LLMAdapter<string> {
  private callCount = 0;
  public operationHistory: ExperienceOperation[][] = [];

  async summarizeTrajectory(
    query: string,
    rollout: Rollout<string>,
    wasSuccessful: boolean
  ): Promise<string> {
    return JSON.stringify({
      rolloutId: rollout.id,
      wasSuccessful,
      observations: wasSuccessful ? 'Clean layout, good colors' : 'Cluttered, poor contrast',
    });
  }

  async extractSemanticAdvantage(
    query: string,
    summaries: TrajectorySummary[],
    currentExperiences: Experience[]
  ): Promise<SemanticAdvantage[]> {
    this.callCount++;

    // Generate a comprehensive insight based on comparing liked vs disliked
    const insight = {
      composition: { layout: `Refined layout insight v${this.callCount}` },
      style: { aesthetic: `Refined style v${this.callCount}` },
      avoid: [`bad_pattern_${this.callCount}`],
      reasoning: `Based on batch ${this.callCount} comparison`,
    };

    return [
      {
        lesson: JSON.stringify(insight, null, 2),
        context: 'image_generation_params',
        sourceRolloutIds: summaries.map((s) => s.rolloutId),
      },
    ];
  }

  async determineExperienceOperations(
    semanticAdvantages: SemanticAdvantage[],
    currentExperiences: Experience[],
    maxOperations: number
  ): Promise<ExperienceOperation[]> {
    let operations: ExperienceOperation[];

    if (currentExperiences.length === 0) {
      // First time: ADD the insight
      operations = [
        {
          type: 'add' as const,
          experience: {
            lesson: semanticAdvantages[0]?.lesson || '{}',
            context: 'image_generation_params',
          },
        },
      ];
    } else if (this.callCount === 3) {
      // Third call: ADD a new separate experience (simulating discovery of new aspect)
      operations = [
        {
          type: 'add' as const,
          experience: {
            lesson: JSON.stringify({ new_aspect: 'discovered in round 3' }),
            context: 'new_discovery',
          },
        },
      ];
    } else {
      // Subsequent calls: MODIFY existing experience
      const existingLesson = JSON.parse(currentExperiences[0].lesson);
      const newInsight = JSON.parse(semanticAdvantages[0]?.lesson || '{}');

      // Merge the insights
      const merged = {
        ...existingLesson,
        ...newInsight,
        avoid: [...(existingLesson.avoid || []), ...(newInsight.avoid || [])],
      };

      operations = [
        {
          type: 'modify' as const,
          experienceId: currentExperiences[0].id,
          newLesson: JSON.stringify(merged, null, 2),
        },
      ];
    }

    this.operationHistory.push(operations);
    return operations;
  }
}

// Adapter that tests DELETE and MERGE operations
class OperationsTestAdapter implements LLMAdapter<string> {
  private scenario: 'add' | 'modify' | 'delete' | 'merge' | 'keep';

  constructor(scenario: 'add' | 'modify' | 'delete' | 'merge' | 'keep') {
    this.scenario = scenario;
  }

  async summarizeTrajectory(): Promise<string> {
    return 'Summary';
  }

  async extractSemanticAdvantage(): Promise<SemanticAdvantage[]> {
    return [{ lesson: '{"test": true}', context: 'test', sourceRolloutIds: [] }];
  }

  async determineExperienceOperations(
    semanticAdvantages: SemanticAdvantage[],
    currentExperiences: Experience[],
    maxOperations: number
  ): Promise<ExperienceOperation[]> {
    switch (this.scenario) {
      case 'add':
        return [{ type: 'add', experience: { lesson: '{"added": true}', context: 'added' } }];

      case 'modify':
        if (currentExperiences.length === 0) {
          return [{ type: 'add', experience: { lesson: '{"initial": true}', context: 'initial' } }];
        }
        return [{ type: 'modify', experienceId: currentExperiences[0].id, newLesson: '{"modified": true}' }];

      case 'delete':
        if (currentExperiences.length === 0) {
          return [{ type: 'add', experience: { lesson: '{"to_delete": true}', context: 'temp' } }];
        }
        return [{ type: 'delete', experienceId: currentExperiences[0].id }];

      case 'merge':
        if (currentExperiences.length < 2) {
          return [{ type: 'add', experience: { lesson: `{"merge_${currentExperiences.length}": true}`, context: 'merge' } }];
        }
        return [{
          type: 'merge',
          experienceIds: currentExperiences.map((e) => e.id),
          mergedLesson: '{"merged": true}',
          mergedContext: 'merged',
        }];

      case 'keep':
      default:
        return [{ type: 'keep' }];
    }
  }
}

describe('GRPO Operations', () => {
  // Helper: Create rollouts with variance to pass the std > 0 check
  // The optimizer requires variance in rewards to trigger optimization
  const makeRollouts = (prefix: string) => [
    { id: `${prefix}_liked`, output: 'good', prompt: 'test' },
    { id: `${prefix}_disliked`, output: 'bad', prompt: 'test' },
  ];

  const makeRewards = (prefix: string) => [
    { rolloutId: `${prefix}_liked`, reward: 1 },
    { rolloutId: `${prefix}_disliked`, reward: -1 },
  ];

  describe('ADD operation', () => {
    it('should add new experiences to the library', async () => {
      const adapter = new OperationsTestAdapter('add');
      const optimizer = new GRPOOptimizer<string>({
        groupSize: 4,
        maxExperiences: 50,
        maxOperationsPerStep: 3,
        llmAdapter: adapter,
      });

      // First ADD
      await optimizer.processGroup('test', makeRollouts('r1'), makeRewards('r1'));
      expect(optimizer.getExperienceCount()).toBe(1);

      // Second ADD
      await optimizer.processGroup('test', makeRollouts('r2'), makeRewards('r2'));
      expect(optimizer.getExperienceCount()).toBe(2);

      console.log('✓ ADD: Experiences accumulate correctly');
    });
  });

  describe('MODIFY operation', () => {
    it('should modify existing experiences', async () => {
      const adapter = new OperationsTestAdapter('modify');
      const optimizer = new GRPOOptimizer<string>({
        groupSize: 4,
        maxExperiences: 50,
        maxOperationsPerStep: 3,
        llmAdapter: adapter,
      });

      // First call: ADD initial
      await optimizer.processGroup('test', makeRollouts('r1'), makeRewards('r1'));
      expect(optimizer.getExperienceCount()).toBe(1);
      expect(optimizer.getExperiences()[0].lesson).toContain('initial');

      // Second call: MODIFY
      await optimizer.processGroup('test', makeRollouts('r2'), makeRewards('r2'));
      expect(optimizer.getExperienceCount()).toBe(1); // Still 1
      expect(optimizer.getExperiences()[0].lesson).toContain('modified');

      console.log('✓ MODIFY: Experience updated in place');
    });
  });

  describe('DELETE operation', () => {
    it('should delete experiences from the library', async () => {
      const adapter = new OperationsTestAdapter('delete');
      const optimizer = new GRPOOptimizer<string>({
        groupSize: 4,
        maxExperiences: 50,
        maxOperationsPerStep: 3,
        llmAdapter: adapter,
      });

      // First call: ADD
      await optimizer.processGroup('test', makeRollouts('r1'), makeRewards('r1'));
      expect(optimizer.getExperienceCount()).toBe(1);

      // Second call: DELETE
      await optimizer.processGroup('test', makeRollouts('r2'), makeRewards('r2'));
      expect(optimizer.getExperienceCount()).toBe(0);

      console.log('✓ DELETE: Experience removed correctly');
    });
  });

  describe('MERGE operation', () => {
    it('should merge multiple experiences into one', async () => {
      const adapter = new OperationsTestAdapter('merge');
      const optimizer = new GRPOOptimizer<string>({
        groupSize: 4,
        maxExperiences: 50,
        maxOperationsPerStep: 3,
        llmAdapter: adapter,
      });

      // First two calls: ADD experiences
      await optimizer.processGroup('test', makeRollouts('r1'), makeRewards('r1'));
      await optimizer.processGroup('test', makeRollouts('r2'), makeRewards('r2'));
      expect(optimizer.getExperienceCount()).toBe(2);

      // Third call: MERGE
      await optimizer.processGroup('test', makeRollouts('r3'), makeRewards('r3'));
      expect(optimizer.getExperienceCount()).toBe(1);
      expect(optimizer.getExperiences()[0].lesson).toContain('merged');

      console.log('✓ MERGE: Multiple experiences combined into one');
    });
  });

  describe('KEEP operation', () => {
    it('should not modify the library', async () => {
      const store = new ExperienceStore(10);
      store.applyOperations([
        { type: 'add', experience: { lesson: 'original', context: 'test' } },
      ]);

      const before = store.getAll()[0].lesson;
      store.applyOperations([{ type: 'keep' }]);
      const after = store.getAll()[0].lesson;

      expect(before).toBe(after);
      expect(store.size()).toBe(1);

      console.log('✓ KEEP: Library unchanged');
    });
  });
});

describe('Realistic Multi-Round Flow', () => {
  it('should correctly ADD then MODIFY then ADD new aspects', async () => {
    const adapter = new RealisticLLMAdapter();
    const optimizer = new GRPOOptimizer<string>({
      groupSize: 4,
      maxExperiences: 50,
      maxOperationsPerStep: 3,
      llmAdapter: adapter,
    });

    const makeRollouts = (round: number) => [
      { id: `r${round}_1`, output: 'img1', prompt: 'test' },
      { id: `r${round}_2`, output: 'img2', prompt: 'test' },
    ];

    const rewards = (round: number) => [
      { rolloutId: `r${round}_1`, reward: 1 },
      { rolloutId: `r${round}_2`, reward: -1 },
    ];

    console.log('\n=== Realistic Multi-Round Flow ===\n');

    // Round 1: Should ADD
    const result1 = await optimizer.processGroup('test', makeRollouts(1), rewards(1));
    console.log('Round 1:', adapter.operationHistory[0].map((o) => o.type).join(', '));
    expect(result1.operations[0].type).toBe('add');
    expect(optimizer.getExperienceCount()).toBe(1);

    // Round 2: Should MODIFY
    const result2 = await optimizer.processGroup('test', makeRollouts(2), rewards(2));
    console.log('Round 2:', adapter.operationHistory[1].map((o) => o.type).join(', '));
    expect(result2.operations[0].type).toBe('modify');
    expect(optimizer.getExperienceCount()).toBe(1); // Still 1, but refined

    // Verify the experience was actually modified
    const exp = optimizer.getExperiences()[0];
    expect(exp.lesson).toContain('v2'); // Should have v2 refinement

    // Round 3: Should ADD new aspect
    const result3 = await optimizer.processGroup('test', makeRollouts(3), rewards(3));
    console.log('Round 3:', adapter.operationHistory[2].map((o) => o.type).join(', '));
    expect(result3.operations[0].type).toBe('add');
    expect(optimizer.getExperienceCount()).toBe(2); // Now 2!

    console.log('\nFinal experiences:');
    optimizer.getExperiences().forEach((e) => {
      console.log(`  [${e.id}] context: ${e.context}`);
    });

    console.log('\n✓ Flow: ADD → MODIFY → ADD works correctly\n');
  });
});

describe('Experience Store Direct Operations', () => {
  it('should apply all operation types correctly', () => {
    const store = new ExperienceStore(10);

    console.log('\n=== Direct Store Operations ===\n');

    // ADD
    store.applyOperations([
      { type: 'add', experience: { lesson: '{"a": 1}', context: 'a' } },
      { type: 'add', experience: { lesson: '{"b": 2}', context: 'b' } },
      { type: 'add', experience: { lesson: '{"c": 3}', context: 'c' } },
    ]);
    console.log('After ADD x3:', store.getAll().map((e) => e.id).join(', '));
    expect(store.size()).toBe(3);

    // MODIFY G2
    store.applyOperations([
      { type: 'modify', experienceId: 'G2', newLesson: '{"b": "modified"}' },
    ]);
    console.log('After MODIFY G2:', store.get('G2')?.lesson);
    expect(store.get('G2')?.lesson).toContain('modified');

    // DELETE G1
    store.applyOperations([{ type: 'delete', experienceId: 'G1' }]);
    console.log('After DELETE G1:', store.getAll().map((e) => e.id).join(', '));
    expect(store.size()).toBe(2);
    expect(store.get('G1')).toBeUndefined();

    // MERGE G2 and G3
    store.applyOperations([
      {
        type: 'merge',
        experienceIds: ['G2', 'G3'],
        mergedLesson: '{"merged": "b+c"}',
        mergedContext: 'merged',
      },
    ]);
    console.log('After MERGE G2+G3:', store.getAll().map((e) => e.id).join(', '));
    expect(store.size()).toBe(1);
    expect(store.get('G4')?.lesson).toContain('merged');

    // KEEP
    const beforeKeep = store.serialize();
    store.applyOperations([{ type: 'keep' }]);
    const afterKeep = store.serialize();
    expect(beforeKeep).toBe(afterKeep);
    console.log('After KEEP: No change');

    console.log('\n✓ All operations work correctly\n');
  });
});

describe('Prompt Building with Experiences', () => {
  it('should include all experiences in the prompt', () => {
    const adapter = new OperationsTestAdapter('add');
    const optimizer = new GRPOOptimizer<string>({
      groupSize: 4,
      maxExperiences: 50,
      maxOperationsPerStep: 3,
      llmAdapter: adapter,
    });

    // Add some experiences directly
    const store = (optimizer as any).experienceStore as ExperienceStore;
    store.applyOperations([
      { type: 'add', experience: { lesson: '{"composition": "grid layout"}', context: 'layout' } },
      { type: 'add', experience: { lesson: '{"style": "minimal"}', context: 'style' } },
      { type: 'add', experience: { lesson: '{"avoid": ["clutter"]}', context: 'avoid' } },
    ]);

    const prompt = optimizer.buildPromptWithExperiences('Generate an ad');

    console.log('\n=== Prompt with Multiple Experiences ===\n');
    console.log(prompt);

    expect(prompt).toContain('[G1]');
    expect(prompt).toContain('[G2]');
    expect(prompt).toContain('[G3]');
    expect(prompt).toContain('grid layout');
    expect(prompt).toContain('minimal');
    expect(prompt).toContain('clutter');

    console.log('\n✓ All experiences included in prompt\n');
  });
});
