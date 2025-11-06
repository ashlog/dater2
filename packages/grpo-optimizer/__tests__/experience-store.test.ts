import { ExperienceStore } from '../src/experience-store';
import { Experience, ExperienceOperation } from '../src/types';

describe('ExperienceStore', () => {
  let store: ExperienceStore;

  beforeEach(() => {
    store = new ExperienceStore(10);
  });

  describe('basic operations', () => {
    it('should start empty', () => {
      expect(store.size()).toBe(0);
      expect(store.getAll()).toEqual([]);
    });

    it('should add new experiences', () => {
      store.applyOperations([
        {
          type: 'add',
          experience: {
            lesson: 'Always validate inputs',
            context: 'form validation',
          },
        },
      ]);

      expect(store.size()).toBe(1);
      const exp = store.get('G1');
      expect(exp).toBeDefined();
      expect(exp?.lesson).toBe('Always validate inputs');
      expect(exp?.context).toBe('form validation');
    });

    it('should delete experiences by id', () => {
      store.applyOperations([
        {
          type: 'add',
          experience: { lesson: 'Lesson 1', context: 'context 1' },
        },
        {
          type: 'add',
          experience: { lesson: 'Lesson 2', context: 'context 2' },
        },
      ]);

      expect(store.size()).toBe(2);

      store.applyOperations([{ type: 'delete', experienceId: 'G1' }]);

      expect(store.size()).toBe(1);
      expect(store.get('G1')).toBeUndefined();
      expect(store.get('G2')).toBeDefined();
    });

    it('should modify existing experiences', () => {
      store.applyOperations([
        {
          type: 'add',
          experience: { lesson: 'Original lesson', context: 'context' },
        },
      ]);

      const originalUpdatedAt = store.get('G1')?.updatedAt;

      // Wait a bit to ensure timestamp changes
      jest.useFakeTimers();
      jest.advanceTimersByTime(1000);

      store.applyOperations([
        {
          type: 'modify',
          experienceId: 'G1',
          newLesson: 'Updated lesson',
        },
      ]);

      jest.useRealTimers();

      const exp = store.get('G1');
      expect(exp?.lesson).toBe('Updated lesson');
      expect(exp?.context).toBe('context'); // Context unchanged
    });

    it('should handle keep operation (no changes)', () => {
      store.applyOperations([
        {
          type: 'add',
          experience: { lesson: 'Lesson', context: 'context' },
        },
      ]);

      const before = store.serialize();

      store.applyOperations([{ type: 'keep' }]);

      // Experiences should be unchanged
      expect(store.size()).toBe(1);
    });
  });

  describe('merge operation', () => {
    it('should merge multiple experiences into one', () => {
      store.applyOperations([
        {
          type: 'add',
          experience: { lesson: 'Lesson 1', context: 'context 1' },
        },
        {
          type: 'add',
          experience: { lesson: 'Lesson 2', context: 'context 2' },
        },
        {
          type: 'add',
          experience: { lesson: 'Lesson 3', context: 'context 3' },
        },
      ]);

      expect(store.size()).toBe(3);

      store.applyOperations([
        {
          type: 'merge',
          experienceIds: ['G1', 'G2'],
          mergedLesson: 'Merged lesson combining 1 and 2',
          mergedContext: 'merged context',
        },
      ]);

      expect(store.size()).toBe(2);
      expect(store.get('G1')).toBeUndefined();
      expect(store.get('G2')).toBeUndefined();
      expect(store.get('G3')).toBeDefined();
      expect(store.get('G4')).toBeDefined();
      expect(store.get('G4')?.lesson).toBe('Merged lesson combining 1 and 2');
    });
  });

  describe('max experiences limit', () => {
    it('should respect max experience limit', () => {
      const smallStore = new ExperienceStore(3);

      // Add 5 experiences
      for (let i = 0; i < 5; i++) {
        smallStore.applyOperations([
          {
            type: 'add',
            experience: { lesson: `Lesson ${i + 1}`, context: `context ${i + 1}` },
          },
        ]);
      }

      // Should only keep 3 (the most recent ones)
      expect(smallStore.size()).toBe(3);

      // The oldest ones (G1, G2) should be removed
      expect(smallStore.get('G1')).toBeUndefined();
      expect(smallStore.get('G2')).toBeUndefined();
      expect(smallStore.get('G3')).toBeDefined();
      expect(smallStore.get('G4')).toBeDefined();
      expect(smallStore.get('G5')).toBeDefined();
    });
  });

  describe('serialization', () => {
    it('should serialize and deserialize correctly', () => {
      store.applyOperations([
        {
          type: 'add',
          experience: { lesson: 'Lesson 1', context: 'context 1' },
        },
        {
          type: 'add',
          experience: { lesson: 'Lesson 2', context: 'context 2' },
        },
      ]);

      const serialized = store.serialize();
      const restored = ExperienceStore.deserialize(serialized);

      expect(restored.size()).toBe(2);
      expect(restored.get('G1')?.lesson).toBe('Lesson 1');
      expect(restored.get('G2')?.lesson).toBe('Lesson 2');
    });

    it('should preserve nextId across serialization', () => {
      store.applyOperations([
        {
          type: 'add',
          experience: { lesson: 'Lesson 1', context: 'context 1' },
        },
      ]);

      const serialized = store.serialize();
      const restored = ExperienceStore.deserialize(serialized);

      // Add another experience - should use G2, not G1
      restored.applyOperations([
        {
          type: 'add',
          experience: { lesson: 'Lesson 2', context: 'context 2' },
        },
      ]);

      expect(restored.get('G2')).toBeDefined();
      expect(restored.get('G2')?.lesson).toBe('Lesson 2');
    });
  });

  describe('formatForPrompt', () => {
    it('should format experiences for prompt injection', () => {
      store.applyOperations([
        {
          type: 'add',
          experience: {
            lesson: 'When solving geometry problems, validate solutions',
            context: 'geometry',
          },
        },
        {
          type: 'add',
          experience: {
            lesson: 'Use systematic exploration for web searching',
            context: 'web',
          },
        },
      ]);

      const formatted = store.formatForPrompt();

      expect(formatted).toContain('[G1] When solving geometry problems, validate solutions');
      expect(formatted).toContain('[G2] Use systematic exploration for web searching');
    });

    it('should return placeholder when empty', () => {
      expect(store.formatForPrompt()).toBe('No experiences yet.');
    });

    it('should sort experiences by ID', () => {
      // Add in reverse order
      store.applyOperations([
        { type: 'add', experience: { lesson: 'First', context: 'c' } },
        { type: 'add', experience: { lesson: 'Second', context: 'c' } },
        { type: 'add', experience: { lesson: 'Third', context: 'c' } },
      ]);

      // Delete middle one
      store.applyOperations([{ type: 'delete', experienceId: 'G2' }]);

      const formatted = store.formatForPrompt();
      const lines = formatted.split('\n');

      expect(lines[0]).toContain('[G1]');
      expect(lines[1]).toContain('[G3]');
    });
  });

  describe('importExperiences', () => {
    it('should import experiences from array', () => {
      store.importExperiences([
        { lesson: 'Lesson A', context: 'context A' },
        { lesson: 'Lesson B', context: 'context B' },
      ]);

      expect(store.size()).toBe(2);
      expect(store.get('G1')?.lesson).toBe('Lesson A');
      expect(store.get('G2')?.lesson).toBe('Lesson B');
    });
  });

  describe('fromExperiences', () => {
    it('should create store from existing experiences', () => {
      const experiences: Experience[] = [
        {
          id: 'G5',
          lesson: 'Lesson 5',
          context: 'context',
          createdAt: new Date(),
          updatedAt: new Date(),
        },
        {
          id: 'G10',
          lesson: 'Lesson 10',
          context: 'context',
          createdAt: new Date(),
          updatedAt: new Date(),
        },
      ];

      const newStore = ExperienceStore.fromExperiences(experiences);

      expect(newStore.size()).toBe(2);
      expect(newStore.get('G5')?.lesson).toBe('Lesson 5');
      expect(newStore.get('G10')?.lesson).toBe('Lesson 10');

      // Next ID should be 11 (one higher than max existing)
      newStore.applyOperations([
        { type: 'add', experience: { lesson: 'New', context: 'new' } },
      ]);
      expect(newStore.get('G11')).toBeDefined();
    });
  });

  describe('clear', () => {
    it('should clear all experiences and reset nextId', () => {
      store.applyOperations([
        { type: 'add', experience: { lesson: 'L1', context: 'c' } },
        { type: 'add', experience: { lesson: 'L2', context: 'c' } },
      ]);

      expect(store.size()).toBe(2);

      store.clear();

      expect(store.size()).toBe(0);

      // NextId should be reset
      store.applyOperations([
        { type: 'add', experience: { lesson: 'New', context: 'c' } },
      ]);
      expect(store.get('G1')).toBeDefined();
    });
  });
});
