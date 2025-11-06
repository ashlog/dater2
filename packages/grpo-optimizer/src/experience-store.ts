/**
 * Experience Store
 *
 * Manages the experience library E from the Training-Free GRPO paper.
 * The experience library is the "learned token prior" that guides future generations.
 *
 * Key operations (from the paper):
 * - ADD: Append new experience to E
 * - DELETE: Remove low-quality experience from E
 * - MODIFY: Refine existing experience based on A_text
 * - MERGE: Combine similar experiences (mentioned in optimization prompt)
 * - KEEP: No change
 */

import {
  Experience,
  SerializedExperience,
  ExperienceOperation,
  AddOperation,
  DeleteOperation,
  ModifyOperation,
  MergeOperation,
} from './types';

/**
 * Manages the experience library for Training-Free GRPO
 */
export class ExperienceStore {
  private experiences: Map<string, Experience>;
  private maxExperiences: number;
  private nextId: number;

  constructor(maxExperiences: number = 50) {
    this.experiences = new Map();
    this.maxExperiences = maxExperiences;
    this.nextId = 1;
  }

  /**
   * Get all experiences in the library
   */
  getAll(): Experience[] {
    return Array.from(this.experiences.values()).sort((a, b) => {
      // Sort by ID for consistent ordering
      const aNum = parseInt(a.id.replace('G', ''), 10);
      const bNum = parseInt(b.id.replace('G', ''), 10);
      return aNum - bNum;
    });
  }

  /**
   * Get a specific experience by ID
   */
  get(id: string): Experience | undefined {
    return this.experiences.get(id);
  }

  /**
   * Get the number of experiences in the library
   */
  size(): number {
    return this.experiences.size;
  }

  /**
   * Format experiences for injection into a prompt
   *
   * Following the paper's format (Appendix):
   * [1] When solving geometry problems...
   * [2] When multiple locations are described...
   */
  formatForPrompt(): string {
    const experiences = this.getAll();
    if (experiences.length === 0) {
      return 'No experiences yet.';
    }

    return experiences
      .map((exp) => `[${exp.id}] ${exp.insight}`)
      .join('\n');
  }

  /**
   * Apply a list of operations to the experience library
   * This is the "controller" step from the paper
   */
  applyOperations(operations: ExperienceOperation[]): void {
    console.log(`[EXP-STORE] Applying ${operations.length} operations. Current size: ${this.experiences.size}`);

    for (const op of operations) {
      console.log(`[EXP-STORE] Processing operation: ${op.type}`);
      switch (op.type) {
        case 'add':
          console.log(`[EXP-STORE]   -> ADD new experience`);
          this.add(op);
          break;
        case 'delete':
          console.log(`[EXP-STORE]   -> DELETE experience ${op.experienceId}`);
          this.delete(op);
          break;
        case 'modify':
          console.log(`[EXP-STORE]   -> MODIFY experience ${op.experienceId}`);
          this.modify(op);
          break;
        case 'merge':
          console.log(`[EXP-STORE]   -> MERGE experiences ${op.experienceIds?.join(', ')}`);
          this.merge(op);
          break;
        case 'keep':
          console.log(`[EXP-STORE]   -> KEEP (no changes)`);
          break;
      }
    }

    console.log(`[EXP-STORE] After operations. Size: ${this.experiences.size}. IDs: ${Array.from(this.experiences.keys()).join(', ')}`);

    // Enforce max experiences limit
    this.enforceLimit();
  }

  /**
   * Add a new experience to the library
   */
  private add(op: AddOperation): void {
    const id = `G${this.nextId++}`;
    const now = new Date();

    const experience: Experience = {
      id,
      insight: op.experience.insight,
      context: op.experience.context,
      createdAt: now,
      updatedAt: now,
    };

    this.experiences.set(id, experience);
  }

  /**
   * Delete an experience from the library
   */
  private delete(op: DeleteOperation): void {
    this.experiences.delete(op.experienceId);
  }

  /**
   * Modify an existing experience
   */
  private modify(op: ModifyOperation): void {
    const existing = this.experiences.get(op.experienceId);
    if (existing) {
      this.experiences.set(op.experienceId, {
        ...existing,
        insight: op.newInsight,
        updatedAt: new Date(),
      });
    }
  }

  /**
   * Merge multiple experiences into one
   */
  private merge(op: MergeOperation): void {
    // Delete all the experiences being merged
    for (const id of op.experienceIds) {
      this.experiences.delete(id);
    }

    // Add the merged experience
    const id = `G${this.nextId++}`;
    const now = new Date();

    const experience: Experience = {
      id,
      insight: op.mergedInsight,
      context: op.mergedContext,
      createdAt: now,
      updatedAt: now,
    };

    this.experiences.set(id, experience);
  }

  /**
   * Enforce the maximum experiences limit
   * Removes oldest experiences (by creation date) when limit is exceeded
   */
  private enforceLimit(): void {
    if (this.experiences.size <= this.maxExperiences) {
      return;
    }

    const sorted = Array.from(this.experiences.values()).sort(
      (a, b) => a.createdAt.getTime() - b.createdAt.getTime()
    );

    const toRemove = sorted.slice(0, sorted.length - this.maxExperiences);
    for (const exp of toRemove) {
      this.experiences.delete(exp.id);
    }
  }

  /**
   * Clear all experiences
   */
  clear(): void {
    this.experiences.clear();
    this.nextId = 1;
  }

  /**
   * Import experiences from an array (useful for initialization)
   */
  importExperiences(experiences: Array<{ insight: string; context: string }>): void {
    for (const exp of experiences) {
      this.applyOperations([
        {
          type: 'add',
          experience: exp,
        },
      ]);
    }
  }

  /**
   * Serialize the store to JSON for persistence
   */
  serialize(): string {
    const serialized: {
      experiences: SerializedExperience[];
      nextId: number;
      version: string;
    } = {
      experiences: Array.from(this.experiences.values()).map((exp) => ({
        id: exp.id,
        insight: exp.insight,
        context: exp.context,
        createdAt: exp.createdAt.toISOString(),
        updatedAt: exp.updatedAt.toISOString(),
      })),
      nextId: this.nextId,
      version: '1.0.0',
    };

    return JSON.stringify(serialized, null, 2);
  }

  /**
   * Deserialize from JSON
   */
  static deserialize(json: string, maxExperiences: number = 50): ExperienceStore {
    const data = JSON.parse(json) as {
      experiences: SerializedExperience[];
      nextId: number;
      version: string;
    };

    const store = new ExperienceStore(maxExperiences);
    store.nextId = data.nextId;

    for (const exp of data.experiences) {
      store.experiences.set(exp.id, {
        id: exp.id,
        insight: exp.insight,
        context: exp.context,
        createdAt: new Date(exp.createdAt),
        updatedAt: new Date(exp.updatedAt),
      });
    }

    return store;
  }

  /**
   * Create a new store initialized with experiences from serialized data
   */
  static fromExperiences(
    experiences: Experience[],
    maxExperiences: number = 50
  ): ExperienceStore {
    const store = new ExperienceStore(maxExperiences);

    for (const exp of experiences) {
      store.experiences.set(exp.id, exp);
      // Update nextId to be higher than any existing ID
      const idNum = parseInt(exp.id.replace('G', ''), 10);
      if (!isNaN(idNum) && idNum >= store.nextId) {
        store.nextId = idNum + 1;
      }
    }

    return store;
  }
}
