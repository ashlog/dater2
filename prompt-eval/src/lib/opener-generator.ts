import { readOptimizerState, readOpenerPrompt } from './data';
import { buildPromptWithExperiences } from './grpo-adapter';
import { chat, chatWithImage, DEFAULT_MODEL } from './llm';

const MY_NAME = 'Ash';

function indentBlock(text: string, spaces = 2): string {
  const pad = ' '.repeat(spaces);
  return text
    .split('\n')
    .map(line => pad + line)
    .join('\n');
}

function buildOutputContractYaml(mode: 'single' | 'batch'): string {
  if (mode === 'batch') {
    return [
      'format: "JSON only"',
      'schema: "Exactly one object with a single key \\"items\\" which is an array of objects with keys \\"id\\" and \\"text\\"."',
      'constraints:',
      '  - "Return only the object with an items array; no extra keys or commentary."',
      '  - "One and only one item per input id, in the same order as inputs."',
    ].join('\n');
  }
  return [
    'format: "JSON only"',
    'schema: "Exactly one object with a single key \\"text\\" whose value is the opener string."',
    'constraints:',
    '  - "Return one and only one opener."',
    '  - "No labels, extra keys, placeholders, ellipses, or commentary."',
  ].join('\n');
}

function withOutputContract(spec: string, mode: 'single' | 'batch'): string {
  const injection = buildOutputContractYaml(mode);
  if (spec.includes('[[OUTPUT_CONTRACT]]')) {
    return spec.replace('[[OUTPUT_CONTRACT]]', injection);
  }
  // If no placeholder, append at the end
  return spec + '\n\noutput_contract:\n' + injection;
}

function extractOpenerText(output: string): string {
  try {
    const trimmed = output.trim();
    // Try parsing as JSON
    const firstBrace = trimmed.indexOf('{');
    const lastBrace = trimmed.lastIndexOf('}');
    if (firstBrace !== -1 && lastBrace > firstBrace) {
      const jsonStr = trimmed.substring(firstBrace, lastBrace + 1);
      const obj = JSON.parse(jsonStr);
      if (obj && typeof obj.text === 'string') return obj.text;
    }
  } catch {
    // Fall through
  }
  // Regex fallback
  const match = output.trim().match(/"text"\s*:\s*"([\s\S]*?)"/);
  if (match) return match[1];
  return output.trim();
}

export interface RegenerateParams {
  profilePrompt: string;
  herName: string;
  imageUrl?: string;
  caption?: string;
  useExperiences?: boolean;
  model?: string;
}

export async function regenerateOpener(params: RegenerateParams): Promise<string> {
  const {
    profilePrompt,
    herName,
    imageUrl,
    caption,
    useExperiences = true,
    model = DEFAULT_MODEL,
  } = params;

  let spec = readOpenerPrompt();
  if (!spec) {
    throw new Error('Could not read opener_prompt.yaml');
  }

  spec = withOutputContract(spec, 'single');

  // Add experiences if enabled
  if (useExperiences) {
    const state = readOptimizerState();
    if (state.experiences.length > 0) {
      spec = buildPromptWithExperiences(spec, state.experiences);
    }
  }

  let rawOutput: string;

  if (imageUrl) {
    // Image-based opener
    const inputBlock = `---
input_type: image_from_profile
her_profile:
  name: ${herName}
  prompt: |
${indentBlock(profilePrompt || '', 4)}
my_name: ${MY_NAME}
caption: |
${indentBlock(caption || '', 2)}
`;

    rawOutput = await chatWithImage(
      spec + '\n\n' + inputBlock,
      imageUrl,
      undefined,
      model
    );
  } else {
    // Text-based opener
    const inputBlock = `---
input_type: text_prompt_from_profile
her_profile:
  name: ${herName}
  prompt: |
${indentBlock(profilePrompt, 4)}
my_name: ${MY_NAME}
`;

    rawOutput = await chat(inputBlock, spec, model);
  }

  return extractOpenerText(rawOutput);
}
