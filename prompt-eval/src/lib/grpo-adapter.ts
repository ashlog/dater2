import {
  LLMAdapter,
  Rollout,
  TrajectorySummary,
  Experience,
  SemanticAdvantage,
  ExperienceOperation,
} from '@webworks/grpo-optimizer';
import { chat, DEFAULT_MODEL } from './llm';

// Output type for pickup line generation
export interface PickupLineOutput {
  text: string;
  source: 'text' | 'image';
  promptUsed?: string;
}

export const pickupLineLLMAdapter: LLMAdapter<PickupLineOutput> = {
  async summarizeTrajectory(
    query: string,
    rollout: Rollout<PickupLineOutput>,
    wasSuccessful: boolean
  ): Promise<string> {
    const output = rollout.output;
    const prompt = `Summarize this pickup line generation attempt in 2-3 sentences.

Profile/Prompt: ${query}

Generated pickup line: "${output.text}"
Source: ${output.source}
Was successful (human liked it): ${wasSuccessful}

Focus on:
- What approach was used (pun, question, observation, etc.)
- Tone and style characteristics
- What made it work or not work

Summary:`;

    return await chat(prompt, undefined, DEFAULT_MODEL);
  },

  async extractSemanticAdvantage(
    query: string,
    summaries: TrajectorySummary[],
    currentExperiences: Experience[]
  ): Promise<SemanticAdvantage[]> {
    const successfulSummaries = summaries.filter(s => s.wasSuccessful);
    const failedSummaries = summaries.filter(s => !s.wasSuccessful);

    if (successfulSummaries.length === 0 || failedSummaries.length === 0) {
      return [];
    }

    const existingInsights = currentExperiences.map(e => `- ${e.insight}`).join('\n');

    const prompt = `Analyze these pickup line attempts to extract actionable insights.

SUCCESSFUL (human liked):
${successfulSummaries.map(s => `- ${s.summary}`).join('\n')}

FAILED (human disliked):
${failedSummaries.map(s => `- ${s.summary}`).join('\n')}

EXISTING INSIGHTS (avoid duplicates):
${existingInsights || '(none yet)'}

Extract 1-2 NEW insights that explain what made successful lines work and failed lines fail.
Format each as JSON: {"insight": "...", "context": "when to apply this"}

Insights:`;

    const text = await chat(prompt, undefined, DEFAULT_MODEL);

    // Parse JSON insights from response
    const advantages: SemanticAdvantage[] = [];
    const jsonMatches = text.match(/\{[^}]+\}/g) || [];

    for (const match of jsonMatches) {
      try {
        const obj = JSON.parse(match);
        if (obj.insight && obj.context) {
          advantages.push({
            insight: obj.insight,
            context: obj.context,
            sourceRolloutIds: summaries.map(s => s.rolloutId),
          });
        }
      } catch {
        // Skip invalid JSON
      }
    }

    return advantages;
  },

  async determineExperienceOperations(
    semanticAdvantages: SemanticAdvantage[],
    currentExperiences: Experience[],
    maxOperations: number
  ): Promise<ExperienceOperation[]> {
    if (semanticAdvantages.length === 0) {
      return [{ type: 'keep' }];
    }

    const operations: ExperienceOperation[] = [];

    // Add new insights as experiences (up to maxOperations)
    for (const advantage of semanticAdvantages.slice(0, maxOperations)) {
      operations.push({
        type: 'add',
        experience: {
          insight: advantage.insight,
          context: advantage.context,
        },
      });
    }

    return operations;
  },
};

// Simple experience type for building prompts (compatible with both local and GRPO types)
interface SimpleExperience {
  insight: string;
  context: string;
}

// Build prompt with experiences appended
export function buildPromptWithExperiences(
  basePrompt: string,
  experiences: SimpleExperience[]
): string {
  if (experiences.length === 0) {
    return basePrompt;
  }

  const experienceSection = `

## LEARNED EXPERIENCES (apply these insights):
${experiences.map((e, i) => `${i + 1}. [${e.context}] ${e.insight}`).join('\n')}

Apply the above experiences when generating your response.`;

  return basePrompt + experienceSection;
}
