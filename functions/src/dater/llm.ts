import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import type {
  TextBlockParam,
  ImageBlockParam,
  MessageParam,
  CacheControlEphemeral,
  ContentBlockParam,
  ThinkingConfigParam,
} from '@anthropic-ai/sdk/resources/messages/index.js';
import { myName } from './token';
import { languageStyle, dos2, guidelines, imageGuidelines } from './dodonts';
import axios from 'axios';
import fs from 'fs';
import path from 'path';
import { createHash } from 'crypto';
import {
  ChatCompletionContentPart,
  ChatCompletionMessageParam,
} from 'openai/resources';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;

export type LLMProviderName = 'openrouter' | 'anthropic';

interface ChatRequest {
  model: string;
  messages: ChatCompletionMessageParam[];
  stop?: string[];
  responseFormat?: 'json_object';
  temperature?: number;
  top_p?: number;
  reasoning?: { effort: 'low' | 'medium' | 'high' };
  maxTokens?: number;
  thinking?: ThinkingConfigParam;
}

interface CompletionRequest {
  model: string;
  prompt: string;
  maxTokens?: number;
  temperature?: number;
  top_p?: number;
  stop?: string[];
}

interface LLMResult {
  content: string;
  usage?: unknown;
  raw?: unknown;
  stopReason?: string;
}

interface LLMProvider {
  readonly name: LLMProviderName;
  chat(request: ChatRequest): Promise<LLMResult>;
  complete?(request: CompletionRequest): Promise<LLMResult>;
}

class OpenRouterProvider implements LLMProvider {
  readonly name: LLMProviderName = 'openrouter';
  private readonly client: OpenAI;

  constructor(apiKey: string | undefined) {
    this.client = new OpenAI({
      baseURL: 'https://openrouter.ai/api/v1',
      apiKey,
    });
  }

  async chat(request: ChatRequest): Promise<LLMResult> {
    const { thinking: _thinking, ...rest } = request;
    const response = await this.client.chat.completions.create({
      model: rest.model,
      messages: rest.messages,
      stop: rest.stop,
      temperature: rest.temperature,
      top_p: rest.top_p,
      reasoning: rest.reasoning,
      ...(rest.responseFormat
        ? { response_format: { type: rest.responseFormat } as any }
        : {}),
    } as any);

    const content = response.choices?.[0]?.message?.content ?? '';
    return {
      content,
      usage: response.usage,
      raw: response,
      stopReason: response.choices?.[0]?.finish_reason,
    };
  }

  async complete(request: CompletionRequest): Promise<LLMResult> {
    const response = await this.client.completions.create({
      model: request.model,
      prompt: request.prompt,
      temperature: request.temperature ?? 0.7,
      max_tokens: request.maxTokens ?? 256,
      top_p: request.top_p ?? 1,
      frequency_penalty: 0,
      presence_penalty: 0,
      stop: request.stop,
    });
    const content = response.choices?.[0]?.text ?? '';
    return {
      content,
      usage: response.usage,
      raw: response,
      stopReason: response.choices?.[0]?.finish_reason,
    };
  }
}

class AnthropicProvider implements LLMProvider {
  readonly name: LLMProviderName = 'anthropic';
  private readonly client: Anthropic | null;
  private readonly defaultMaxTokens = 5024;

  constructor(apiKey: string | undefined) {
    this.client = apiKey ? new Anthropic({ apiKey }) : null;
  }

  async chat(request: ChatRequest): Promise<LLMResult> {
    if (!this.client) {
      throw new Error('ANTHROPIC_API_KEY is not set');
    }

    const { system, messages } = await this.convertMessages(request.messages);

    const response = await this.client.messages.create({
      model: this.normalizeModel(request.model),
      system: system ?? undefined,
      messages,
      max_tokens: request.maxTokens ?? this.defaultMaxTokens,
      temperature: request.temperature,
      top_p: request.top_p,
      stop_sequences: request.stop,
      thinking: this.resolveThinkingConfig(request),
    });

    const content = (response.content || [])
      .filter((block: any) => block.type === 'text' && typeof block.text === 'string')
      .map((block: any) => block.text)
      .join('\n');

    return {
      content,
      usage: (response as any).usage,
      raw: response,
      stopReason: (response as any).stop_reason,
    };
  }

  async complete(request: CompletionRequest): Promise<LLMResult> {
    const result = await this.chat({
      model: request.model,
      messages: [
        {
          role: 'user',
          content: request.prompt,
        },
      ],
      maxTokens: request.maxTokens ?? this.defaultMaxTokens,
      temperature: request.temperature,
      top_p: request.top_p,
      stop: request.stop,
    });
    return result;
  }

  private normalizeModel(model: string): string {
    if (!model) return 'claude-3-5-sonnet-20241022';
    const trimmed = model.trim();
    const aliasMap: Record<string, string> = {
      'anthropic/claude-sonnet-4.5': 'claude-sonnet-4-5-20250929',
      'anthropic/claude-3-opus:beta': 'claude-3-opus-20240229',
      'anthropic/claude-3-haiku:beta': 'claude-3-haiku-20240307',
    };
    if (aliasMap[trimmed]) return aliasMap[trimmed];
    if (trimmed.includes('/')) {
      return trimmed.split('/').pop() || trimmed;
    }
    return trimmed;
  }

  private async convertMessages(messages: ChatCompletionMessageParam[]): Promise<{
    system: TextBlockParam[] | null;
    messages: MessageParam[];
  }> {
    const systemParts: TextBlockParam[] = [];
    const converted: MessageParam[] = [];

    for (const msg of messages) {
      const contentBlocks = await this.convertContent(msg.content);
      if (msg.role === 'system') {
        const textBlocks = contentBlocks.filter((block): block is TextBlockParam => block.type === 'text');
        if (textBlocks.length) {
          systemParts.push(...textBlocks);
        }
        continue;
      }

      const role: 'user' | 'assistant' = msg.role === 'assistant' ? 'assistant' : 'user';
      if (contentBlocks.length === 0) continue;
      converted.push({ role, content: contentBlocks });
    }

    return {
      system: systemParts.length ? systemParts : null,
      messages: converted,
    };
  }

  private async convertContent(
    content: ChatCompletionMessageParam['content']
  ): Promise<ContentBlockParam[]> {
    if (typeof content === 'string') {
      if (!content.trim()) return [];
      return [{ type: 'text', text: content }];
    }

    if (!Array.isArray(content)) {
      return [];
    }

    const blocks: ContentBlockParam[] = [];
    for (const part of content as any[]) {
      if (!part) continue;
      if (part.type === 'text') {
        const text = typeof part.text === 'string' ? part.text : '';
        if (text.trim()) {
          const textBlock: TextBlockParam = { type: 'text', text };
          const cacheControl = this.normalizeCacheControl(part.cache_control);
          if (cacheControl) {
            textBlock.cache_control = cacheControl;
          }
          blocks.push(textBlock);
        }
        continue;
      }
      if (part.type === 'image_url') {
        const block = this.convertImageUrlPart(part);
        if (block) {
          blocks.push(block);
        }
        continue;
      }
      if (typeof part.text === 'string' && part.text.trim()) {
        const textBlock: TextBlockParam = { type: 'text', text: part.text };
        const cacheControl = this.normalizeCacheControl(part.cache_control);
        if (cacheControl) {
          textBlock.cache_control = cacheControl;
        }
        blocks.push(textBlock);
      }
    }
    return blocks;
  }

  private convertImageUrlPart(part: any): ImageBlockParam | null {
    const rawUrl = typeof part?.image_url === 'string' ? part.image_url : part?.image_url?.url;
    if (!rawUrl || typeof rawUrl !== 'string') {
      return null;
    }

    const cacheControl = this.normalizeCacheControl(part?.cache_control);
    const block: ImageBlockParam = {
      type: 'image',
      source: {
        type: 'url',
        url: rawUrl,
      },
    };
    if (cacheControl) {
      block.cache_control = cacheControl;
    }
    return block;
  }

  private normalizeCacheControl(value: any): CacheControlEphemeral | undefined {
    if (!value || typeof value !== 'object') {
      return undefined;
    }
    if (value.type !== 'ephemeral') {
      return undefined;
    }
    const ttl = value.ttl;
    if (ttl === '5m' || ttl === '1h') {
      return { type: 'ephemeral', ttl };
    }
    return { type: 'ephemeral' };
  }

  private resolveThinkingConfig(request: ChatRequest): ThinkingConfigParam | undefined {
    const config = request.thinking;
    if (!config) return undefined;
    if (config.type === 'enabled') {
      const maxTokens = request.maxTokens ?? this.defaultMaxTokens;
      if (typeof config.budget_tokens === 'number' && config.budget_tokens > maxTokens) {
        return undefined;
      }
    }
    return config;
  }
}

const providers: Record<LLMProviderName, LLMProvider> = {
  openrouter: new OpenRouterProvider(OPENROUTER_API_KEY),
  anthropic: new AnthropicProvider(ANTHROPIC_API_KEY),
};

let activeProvider: LLMProviderName = (process.env.LLM_PROVIDER as LLMProviderName) || 'openrouter';
let defaultThinkingConfig: ThinkingConfigParam | undefined = undefined;

function getProvider(): LLMProvider {
  return providers[activeProvider] || providers.openrouter;
}

export function setLLMProvider(provider: LLMProviderName): void {
  if (!providers[provider]) {
    throw new Error(`Unknown LLM provider: ${provider}`);
  }
  activeProvider = provider;
}

export function getLLMProviderName(): LLMProviderName {
  return activeProvider;
}

export function setLLMThinkingConfig(config: ThinkingConfigParam | undefined): void {
  defaultThinkingConfig = config;
}

export class LLMRefusalError extends Error {
  constructor(
    message: string,
    public readonly provider: LLMProviderName,
    public readonly stopReason: string,
    public readonly raw?: unknown
  ) {
    super(message);
    this.name = 'LLMRefusalError';
    if (typeof Error.captureStackTrace === 'function') {
      Error.captureStackTrace(this, LLMRefusalError);
    }
  }
}

function isLLMRefusalError(error: unknown): error is LLMRefusalError {
  if (error instanceof LLMRefusalError) return true;
  if (!error || typeof error !== 'object') return false;
  return (error as any).name === 'LLMRefusalError' && (error as any).stopReason === 'refusal';
}

function throwIfRefusal(result: LLMResult, providerName: LLMProviderName, context: string): void {
  if (result.stopReason === 'refusal') {
    throw new LLMRefusalError(`LLM refusal during ${context}`, providerName, result.stopReason, result.raw);
  }
}
export async function generatePickupLineFromGreentext(
  image: string,
  greentext: string,
  herName: string,
  model: string
): Promise<string> {
  return chatImage(
    // `[QUESTION] Write a short, very creative, and not awkward pickup line to respond with to the following [TOPIC].\nGuidelines: Only generate concise and well written responses that are well thought out. Do not use cliche and overused pickup lines, instead be more creative. Do not use overly complex and verbose words. Limit the word count per sentence to a maximum of 20 words. Do not digress from [TOPIC]. Respond as if you are talking to a friend, and be kind. Assume that you have not met ${herName} yet, nor gone out on a date with her before. Do not generate fabricated experiences which did not happen. Only generate the pickup line and nothing else. Do not add hashtags. My name is ${myName}, her name is ${herName}. You are responding to ${herName}. Only use her name to make a pun out of ${herName} while still remaining on topic. Do not mention the name if a pun is not possible. Do not come off desperate. Do not just repeat the prompt and add words to it. Be as funny as possible.\nGenerate 3 pickup lines. At the end start with [FINAL ANSWER] and use quotes ("") to wrap the final answer in.\n\n[TOPIC]\n${herName} said: ${prompt}\n\n`,
    `[TASK] Using your immense creative wit with writing highly successful pickup lines, write the funniest possible pickup line from the following 4chan greentext. The woman's name is ${herName}. Do not return anything else, just the pickup line without quotes.
${greentext}`,
    image,
    [],
    model
  );
}

export async function generatePickupLine(
  prompt: string,
  herName: string
): Promise<string> {
  return chat(
    // `[QUESTION] Write a short, very creative, and not awkward pickup line to respond with to the following [TOPIC].\nGuidelines: Only generate concise and well written responses that are well thought out. Do not use cliche and overused pickup lines, instead be more creative. Do not use overly complex and verbose words. Limit the word count per sentence to a maximum of 20 words. Do not digress from [TOPIC]. Respond as if you are talking to a friend, and be kind. Assume that you have not met ${herName} yet, nor gone out on a date with her before. Do not generate fabricated experiences which did not happen. Only generate the pickup line and nothing else. Do not add hashtags. My name is ${myName}, her name is ${herName}. You are responding to ${herName}. Only use her name to make a pun out of ${herName} while still remaining on topic. Do not mention the name if a pun is not possible. Do not come off desperate. Do not just repeat the prompt and add words to it. Be as funny as possible.\nGenerate 3 pickup lines. At the end start with [FINAL ANSWER] and use quotes ("") to wrap the final answer in.\n\n[TOPIC]\n${herName} said: ${prompt}\n\n`,
    `[TASK] Write a question for ${myName} to respond with to the following [TOPIC].
Guidelines:
${guidelines(herName)}

Language Style Examples:
${languageStyle}

Generate 3 questions closely adhering these guidelines. At the end start with [FINAL QUESTION] and use quotes ("") to wrap the best pickup line in. Choose only one.
[TOPIC]
The girl's name is ${herName} and she said:
${prompt}`,
    PL_PREPROMPT
  );
}

export async function getDemographics(image: string) {
  return chatImage(
    `Generate a demographic profile for the person in the image. Respond with only the following JSON response type format:
type PersonDemographics = {
  fitzpatrickSkinTone: number; // 1-6 scale 6 being the darkest
  bodyThickness: string; // one-of: ['skinny', 'underweight', 'normal', 'overweight', 'obese']. Don't be nice. Be honest. Look at their body.
  attractiveness: number; // 1-10 scale on how attractive the person's face is.
  groupPhoto: boolean; // true if there is more than one obviously prominent person in the image
};

// Example usage
const examplePerson: PersonDemographics = {
  fitzpatrickSkinTone: 6,
  bodyThickness: 'overweight',
  attractiveness: 4,
  groupPhoto: false
};`,
    image,
    [],
    'anthropic/claude-3-opus:beta'
  );
}

export async function generatePickupLineImage(
  image: string,
  caption: string,
  background: string,
  herName: string,
  model: string
): Promise<string> {
  return chatImage(
    `[TASK] Write a question for Chad, a very good looking guy, to the following [IMAGE] and [HER BIO].
    Guidelines:
    1. Generate a witty pickup line based on the following background context and image. Women love confidence, food, and humor.
    2. You are writing a pickup line as a man to a woman.
    3. Using the girl's Hinge prompts and image, write one breezy, non-corny, 90-140-character first message optimized for women in San Francisco ages 20-29 that asks an open-ended, playful either/or or mini-game question grounded in a real detail from their profile (food, nostalgia, or light SF color), with no placeholders, no generic greetings, no negs, no direct invites, no innuendo, at most one emoji, and avoid fast-aging name-drops.
    4. The setting is "online dating on Hinge."
    5. Do not be creepy, be flirty.
    6. Write your thoughts inside a <thinking></thinking> block.
    7. Turn it into a witty joke or question that she would probably answer to, with a maximum of 20 words in the sentence.
    8. Subtle references help.
    9. Finally start with [FINAL THOUGHT] and use quotes ("") to wrap the question in. Do not use more than 2 quotes. Do not say anything else besides the FINAL THOUGHT.

    Language Style Examples:
    ${languageStyle}
    
    [HER BIO]
    ${background}

    [CAPTION]
    ${caption}

    [IMAGE]`,
    // `[QUESTION] Write a short, very creative, and not awkward pickup line to respond with to the following [TOPIC].\nGuidelines: Only generate concise and well written responses that are well thought out. Do not use cliche and overused pickup lines, instead be more creative. Do not use overly complex and verbose words. Limit the word count per sentence to a maximum of 20 words. Do not digress from [TOPIC]. Respond as if you are talking to a friend, and be kind. Assume that you have not met ${herName} yet, nor gone out on a date with her before. Do not generate fabricated experiences which did not happen. Only generate the pickup line and nothing else. Do not add hashtags. My name is ${myName}, her name is ${herName}. You are responding to ${herName}. Only use her name to make a pun out of ${herName} while still remaining on topic. Do not mention the name if a pun is not possible. Do not come off desperate. Do not just repeat the prompt and add words to it. Be as funny as possible.\nGenerate 3 pickup lines. At the end start with [FINAL ANSWER] and use quotes ("") to wrap the final answer in.\n\n[TOPIC]\n${herName} said: ${prompt}\n\n`,
    //     `[TASK] Write a question for ${myName} to the following [IMAGE].
    // Guidelines:
    // ${imageGuidelines(herName)}
    //
    // Language Style Examples:
    // ${languageStyle}
    //
    // Important:
    // Before generating questions, summarize the image environment and extract specific items and small details from the image to use as context to come up with a good question or joke.
    // If relevant, use details from [HER BIO] and [CAPTION] for additional context to the image: When using information from [HER BIO], lightly repeat and mix the information within your response.
    // 1. Think about the entire profile and list your thoughts in a <THOUGHT>></THOUGHT> block. Transform visual details into personal, playful, romantic connections. Charismatically exploit details to explore different angles for a witty pickup line in parentheses using highly condensed and short phrases. Exhaustively make connections within the image that might aid in crafting a question. Use the bio as background information that shows her interests.
    // 2. Generate 3 of the most charismatic questions based on your thoughts and guidelines. Match the language style examples as closely as possible.
    // 3. Critique the reference and if it might be too hard to remember from her bio in a <REACT></REACT> block. Pick the least cringe and the best question to ask or a combination.
    // 4. Think about the language style examples and check if it can be matched closer in a <REVISE></REVISE> block.
    // 5. Minimally edit the question to become a pickup line based on your reflection during <REACT> and <REVISE>. Remove the question part of the prompt if it is less cringe without a question. Add context if references to bio are subtle. Reduce the length of the response to less than 4 words if possible without losing context using language tricks like emojis or symbols. Make sure that the line generated makes sense in context of the image.
    // 6. Check if the revision is any better for a reaction.
    //
    // 7. Start with [FINAL THOUGHT] and use quotes ("") to wrap the question in. Do not use more than 2 quotes. Do not say anything else besides the FINAL THOUGHT.
    // [HER BIO]
    // ${background}
    // [CAPTION]
    // ${caption}
    // [IMAGE]`,
    image,
    [],
    // PL_PREPROMPT
    model
  );
}

export async function extractFinalThought(
  text: string,
  model: string
): Promise<string> {
  return (
    await chat(
      // `[QUESTION] Write a short, very creative, and not awkward pickup line to respond with to the following [TOPIC].\nGuidelines: Only generate concise and well written responses that are well thought out. Do not use cliche and overused pickup lines, instead be more creative. Do not use overly complex and verbose words. Limit the word count per sentence to a maximum of 20 words. Do not digress from [TOPIC]. Respond as if you are talking to a friend, and be kind. Assume that you have not met ${herName} yet, nor gone out on a date with her before. Do not generate fabricated experiences which did not happen. Only generate the pickup line and nothing else. Do not add hashtags. My name is ${myName}, her name is ${herName}. You are responding to ${herName}. Only use her name to make a pun out of ${herName} while still remaining on topic. Do not mention the name if a pun is not possible. Do not come off desperate. Do not just repeat the prompt and add words to it. Be as funny as possible.\nGenerate 3 pickup lines. At the end start with [FINAL ANSWER] and use quotes ("") to wrap the final answer in.\n\n[TOPIC]\n${herName} said: ${prompt}\n\n`,
      `Extract the text from [FINAL THOUGHT]. Do not return anything else, just the text without quotes. If you cannot find it, return "<EMPTY>" without quotes.
${text}`,
      [],
      model
    )
  ).trim();
}

export async function describeImage(image: string): Promise<string> {
  return chatImage(
    // `[QUESTION] Write a short, very creative, and not awkward pickup line to respond with to the following [TOPIC].\nGuidelines: Only generate concise and well written responses that are well thought out. Do not use cliche and overused pickup lines, instead be more creative. Do not use overly complex and verbose words. Limit the word count per sentence to a maximum of 20 words. Do not digress from [TOPIC]. Respond as if you are talking to a friend, and be kind. Assume that you have not met ${herName} yet, nor gone out on a date with her before. Do not generate fabricated experiences which did not happen. Only generate the pickup line and nothing else. Do not add hashtags. My name is ${myName}, her name is ${herName}. You are responding to ${herName}. Only use her name to make a pun out of ${herName} while still remaining on topic. Do not mention the name if a pun is not possible. Do not come off desperate. Do not just repeat the prompt and add words to it. Be as funny as possible.\nGenerate 3 pickup lines. At the end start with [FINAL ANSWER] and use quotes ("") to wrap the final answer in.\n\n[TOPIC]\n${herName} said: ${prompt}\n\n`,
    'Describe this image in detail. 100 words.',
    image,
    PL_PREPROMPT
  );
}

export async function complete(prompt: string): Promise<string> {
  const provider = getProvider();
  try {
    if (provider.complete) {
      const result = await provider.complete({
        model: 'openai/gpt-4o',
        prompt,
        temperature: 0.7,
        maxTokens: 256,
        top_p: 1,
        stop: ['===='],
      });
      throwIfRefusal(result, provider.name, 'complete');
      return result.content;
    }

    const fallback = await provider.chat({
      model: 'openai/gpt-4o',
      messages: [
        {
          role: 'user',
          content: prompt,
        } as ChatCompletionMessageParam,
      ],
      temperature: 0.7,
      top_p: 1,
      maxTokens: 256,
      stop: ['===='],
      thinking: defaultThinkingConfig,
    });
    throwIfRefusal(fallback, provider.name, 'complete');
    return fallback.content;
  } catch (e) {
    if (isLLMRefusalError(e)) {
      throw e;
    }
    console.error('complete() error', e);
    return '';
  }
}

export const PL_PREPROMPT: ChatCompletionMessageParam[] = [
  {
    role: 'user',
    content: [
      {
        type: 'text',
        text: `You are ${myName}, a man navigating online dating with a flair for being flirty.
You live in the Bay Area.
Your interests are:
Traveling in the Bay Area, Mexico, or Hawaii, eating, cooking, hiking, skiing, watching sci-fi, horror, mysteries, thrillers, video games, driving, beaches, coding, machine learning, girls with skirts, and having many random hobbies.`,
        cache_control: {
          type: 'ephemeral',
        },
      } as unknown as ChatCompletionContentPart,
    ],
  },
];

export async function chat(
  prompt: string,
  preprompt: any[],
  // model = 'gpt-4-1106-preview'
  // model = 'meta-llama/llama-3-70b-instruct'
  // model = 'openai/gpt-4o'
  model = 'gemini-2.5-pro',
  responseFormat?: 'json_object'
): Promise<string> {
  try {
    const provider = getProvider();
    const baseMessages = Array.isArray(preprompt)
      ? (preprompt as ChatCompletionMessageParam[])
      : [];
    const messages: ChatCompletionMessageParam[] = [
      ...baseMessages,
      { role: 'user', content: prompt },
    ];

    const result = await provider.chat({
      model,
      messages,
      reasoning: { effort: 'medium' },
      stop: ['===='],
      responseFormat,
      thinking: defaultThinkingConfig,
    });

    throwIfRefusal(result, provider.name, 'chat');

    return result.content;
  } catch (e) {
    if (isLLMRefusalError(e)) {
      throw e;
    }
    console.error('chat() error', e);
    return '';
  }
}

// Read a single canonical YAML spec from repo root: functions/src/opener_prompt.yaml
let openerPromptCache: string | null = null;
let openerPromptHashCache: string | null = null;
async function getOpenerPromptText(): Promise<string> {
  if (openerPromptCache) return openerPromptCache;
  // __dirname is functions/src/dater or functions/lib/dater -> go up 3 levels to repo root
  const repoRoot = path.resolve(__dirname, '../../..');
  const yamlPath = path.join(repoRoot, 'functions', 'src', 'opener_prompt.yaml');
  try {
    const text = await fs.promises.readFile(yamlPath, 'utf8');
    openerPromptCache = text;
    openerPromptHashCache = createHash('sha256').update(text).digest('hex');
    return text;
  } catch (e) {
    console.error('Failed to read opener_prompt.yaml at', yamlPath, e);
    throw e;
  }
}

export async function getOpenerPromptMetadata(): Promise<{ text: string; hash: string }> {
  const text = await getOpenerPromptText();
  if (!openerPromptHashCache) {
    openerPromptHashCache = createHash('sha256').update(text).digest('hex');
  }
  return { text, hash: openerPromptHashCache };
}

function buildOutputContractYaml(mode: 'single' | 'batch'): string {
  if (mode === 'batch') {
    return [
      'format: "JSON only"',
      'schema: "Exactly one object with a single key \"items\" which is an array of objects with keys \"id\" and \"text\"."',
      'constraints:',
      '  - "Return only the object with an items array; no extra keys or commentary."',
      '  - "One and only one item per input id, in the same order as inputs."',
    ].join('\n');
  }
  return [
    'format: "JSON only"',
    'schema: "Exactly one object with a single key \"text\" whose value is the opener string."',
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
  throw new Error('[[OUTPUT_CONTRACT]] not found in spec');
}

function indentBlock(text: string, spaces = 2): string {
  const pad = ' '.repeat(spaces);
  return text
    .split('\n')
    .map(line => pad + line)
    .join('\n');
}

/**
 * Extracts and parses JSON from text that may contain markdown code blocks or extra wrapper text.
 * Finds the first '{' and last '}' and attempts to parse the content between them.
 * @param text - Raw text that may contain JSON wrapped in markdown or other text
 * @returns Parsed JSON object or throws if parsing fails
 */
export function extractAndParseJSON(text: string): any {
  const trimmed = text.trim();

  if (!trimmed) {
    throw new Error('Empty text provided to extractAndParseJSON');
  }

  // Try parsing as-is first (fast path for clean JSON)
  try {
    return JSON.parse(trimmed);
  } catch {
    // Continue to extraction logic
  }

  // Extract JSON from first { to last }, ignoring markdown code blocks and other wrapper text
  const firstBrace = trimmed.indexOf('{');
  const lastBrace = trimmed.lastIndexOf('}');

  if (firstBrace === -1 || lastBrace === -1 || lastBrace <= firstBrace) {
    throw new Error('No valid JSON object found in text');
  }

  const extracted = trimmed.substring(firstBrace, lastBrace + 1);
  return JSON.parse(extracted);
}

// Try to extract {"text":"..."} from model output; otherwise return raw
function extractOpenerText(output: string): string {
  try {
    const obj = extractAndParseJSON(output);
    if (obj && typeof obj.text === 'string') return obj.text;
  } catch { }
  const match = output.trim().match(/"text"\s*:\s*"([\s\S]*?)"/);
  if (match) return match[1];
  return output.trim();
}

// Generate an opener using the YAML spec as the direct prompt text
export async function generateOpenerFromYaml(
  profilePrompt: string,
  herName: string,
  model = 'gemini-2.5-flash'
): Promise<string> {
  const spec = withOutputContract(await getOpenerPromptText(), 'single');
  // Send the YAML spec as a system message, and the input block as the user message
  const inputBlock = `---\ninput_type: text_prompt_from_profile\nher_profile:\n  name: ${herName}\n  prompt: |\n${indentBlock(profilePrompt, 4)}\nmy_name: ${myName}\n`;
  const out = await chat(
    inputBlock,
    [
      {
        role: 'system',
        content: [
          {
            type: "text",
            text: spec,
            cache_control: {
              type: "ephemeral"
            }
          } as unknown as ChatCompletionContentPart,
        ],
      },
    ],
    model
  );
  return extractOpenerText(out);
}

// Batch variant: generate openers for many prompts in a single call.
// Returns a Map of id -> opener text.
export async function generateOpenersFromYamlBatch(
  items: { id: string; profilePrompt: string }[],
  herName: string,
  model = 'gemini-2.5-flash'
): Promise<Map<string, string>> {
  const spec = withOutputContract(await getOpenerPromptText(), 'batch');
  // Build a single user message that contains an instruction and a list of inputs with ids.
  // We intentionally treat the YAML spec as a style guide, while overriding the one-output contract
  // by explicitly asking for a JSON array mapping each input id to a single opener.
  const inputBlocks = items
    .map(
      (it) =>
        `- id: ${it.id}\n  input_type: text_prompt_from_profile\n  her_profile:\n    name: ${herName}\n    prompt: |\n${indentBlock(it.profilePrompt, 6)}\n  my_name: ${myName}`
    )
    .join('\n');

  const instruction = `You act strictly under the following style guide (YAML below).\nFor EACH input item, generate exactly one opener that fully adheres to the style_tone, cta_rules, ai_scent_filters, generation_algorithm, and formatting_checks.\nFollow the output_contract defined in the style guide.\nInputs:\n${inputBlocks}`;

  const out = await chat(
    instruction,
    [
      {
        role: 'system',
        content: [
          {
            type: "text",
            text: spec,
            cache_control: {
              type: "ephemeral"
            }
          } as unknown as ChatCompletionContentPart,
        ],
      },
    ],
    model
  );

  // Try strict JSON first (array), fall back to lenient parsing.
  const map = new Map<string, string>();
  const text = (out || '').trim();

  if (!text) {
    console.error('generateOpenersFromYamlBatch: empty response from API');
    return map;
  }

  try {
    const parsed = extractAndParseJSON(text);
    const arr = Array.isArray(parsed) ? parsed : parsed?.results || parsed?.items || [];
    if (Array.isArray(arr)) {
      for (const item of arr) {
        const id = String(item?.id ?? '');
        const value = typeof item?.text === 'string' ? item.text : '';
        if (id) map.set(id, value);
      }
      if (map.size > 0) return map;
      console.error('generateOpenersFromYamlBatch: parsed array but no valid items. Array:', arr);
    } else {
      console.error('generateOpenersFromYamlBatch: unexpected JSON structure (not array or missing results/items). Parsed:', parsed);
    }
  } catch (parseError) {
    console.error('generateOpenersFromYamlBatch: JSON parse failed, trying regex fallback. Error:', parseError, 'Text:', text.substring(0, 200));
  }

  // Last resort: regex to find objects with id/text pairs
  try {
    const matches = text.match(/\{[^}]*\}/g) || [];
    for (const m of matches) {
      try {
        const obj = JSON.parse(m);
        if (obj && obj.id && typeof obj.text === 'string') {
          map.set(String(obj.id), obj.text);
        }
      } catch { }
    }
    if (map.size === 0) {
      console.error('generateOpenersFromYamlBatch: regex fallback found no valid items. Matches count:', matches.length);
    }
  } catch (regexError) {
    console.error('generateOpenersFromYamlBatch: regex fallback failed. Error:', regexError);
  }
  return map;
}

// Generate an opener for an image using the YAML spec; includes bio and caption
export async function generateImageOpenerFromYaml(
  image: string,
  caption: string,
  background: string,
  herName: string,
  model = 'x-ai/grok-4'
): Promise<string> {
  const spec = withOutputContract(await getOpenerPromptText(), 'single');
  // Send the YAML spec as a system message, and the input block as the user message
  const inputBlock = `---\ninput_type: image_from_profile\nher_profile:\n  name: ${herName}\n  prompt: |\n${indentBlock(background || '', 4)}\nmy_name: ${myName}\ncaption: |\n${indentBlock(caption || '', 2)}\n`;
  const out = await chatImage(
    inputBlock,
    image,
    [
      {
        role: 'system',
        content: [
          {
            type: 'text',
            text: spec,
            cache_control: {
              type: "ephemeral"
            },
          } as unknown as ChatCompletionContentPart,
        ],
      },
    ],
    model
  );
  return extractOpenerText(out);
}

export async function chatImage(
  prompt: string,
  image: string,
  preprompt: any[],
  // model = 'gpt-4-1106-preview'
  // model = 'meta-llama/llama-3-70b-instruct'
  // model = 'openai/gpt-4o'
  // model = 'google/gemini-exp-1114'
  // model = 'anthropic/claude-3.5-sonnet:beta'
  // model = 'google/gemini-2.0-flash-001'
  model = 'gemini-2.5-pro',
  responseFormat?: 'json_object'
  // model = 'anthropic/claude-3.5-sonnet:beta'
  // model = 'openai/gpt-4.5-preview'
  // model = 'openai/o1-preview'
  // model = 'google/gemini-2.0-flash-exp:free'
  // model = 'gemini-2.0-flash-exp'
  // model = 'meta-llama/llama-3.1-405b-instruct'
  // model = 'perplexity/llama-3-sonar-large-32k-chat'
): Promise<string> {
  const maxRetries = 3;
  const baseDelay = 500; //milliseconds

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const provider = getProvider();
      const baseMessages = Array.isArray(preprompt)
        ? (preprompt as ChatCompletionMessageParam[])
        : [];
      const messages: ChatCompletionMessageParam[] = [
        ...baseMessages,
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: prompt,
            },
            {
              type: 'image_url',
              image_url: {
                url: image,
              },
            },
          ],
        } as ChatCompletionMessageParam,
      ];

      const result = await provider.chat({
        model,
        messages,
        temperature: 1,
        reasoning: {
          effort: 'low',
        },
        responseFormat,
        thinking: defaultThinkingConfig,
      });

      throwIfRefusal(result, provider.name, 'chatImage');

      if ((result.raw as any)?.error) {
        console.error('chatImage error', (result.raw as any).error);
        throw (result.raw as any).error;
      }
      if (result.usage) {
        console.log('usage', result.usage);
      }
      return result.content;
    } catch (e) {
      if (isLLMRefusalError(e)) {
        throw e;
      }
      console.error(`Attempt ${attempt} failed:`, e);

      if (attempt === maxRetries) {
        console.error('All retry attempts failed');
        return '';
      }

      // Exponential backoff: wait longer between each retry
      const delay = baseDelay * Math.pow(2, attempt - 1);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
  return ''; // This should never be reached due to the return in the catch block
}

// Ask the model to choose the best pickup line by index and return JSON only.
// Returns the zero-based index of the best line, or 0 on failure.
export async function bestPl(pl: string[], model = 'gemini-2.5-pro'): Promise<number> {
  const provider = getProvider();
  const result = await provider.chat({
    model,
    messages: [
      {
        role: 'user',
        content:
          'From the numbered choices below, select the single line that is most likely to get a reply (witty, funny, charismatic, authentic).\n' +
          'Respond with strict JSON only: {"index": <number>} where index is zero-based. No prose.\n' +
          'Choices:\n' +
          pl.map((it, i) => `${i}. "${it}"`).join('\n'),
      },
    ],
    stop: ['===='],
    thinking: defaultThinkingConfig,
  });
  throwIfRefusal(result, provider.name, 'bestPl');
  const content = (result.content || '').trim();
  try {
    const obj = extractAndParseJSON(content);
    const idx = Number(obj?.index);
    return Number.isFinite(idx) && idx >= 0 && idx < pl.length ? idx : 0;
  } catch {
    // Fallback: try to pull the last integer in the content
    const m = content.match(/(\d+)/g);
    if (m && m.length) {
      const idx = Number(m[m.length - 1]);
      if (Number.isFinite(idx) && idx >= 0 && idx < pl.length) return idx;
    }
    return 0;
  }
}

export async function compare(pl: string[]): Promise<string> {
  try {
    const provider = getProvider();
    const result = await provider.chat({
      model: 'gemini-2.5-pro',
      messages: [
        {
          role: 'user',
          content:
            'Pick the better response from the two choices. Return the response that is most authentic and genuine in quotes. Include the entire response. Prefix with "[BEST]".\n' +
            'Reply format:\n' +
            '```\n' +
            '[BEST] "This is a response."\n' +
            '```\n' +
            'Choices:\n' +
            pl.map((it, _) => `"${it}"`).join('\n'),
        },
      ],
      stop: ['===='],
      thinking: defaultThinkingConfig,
    });

    const content = result?.content;
    const afterBest = content?.split('[BEST]')[1];
    const extractedMessage = afterBest?.split('"')[1];

    return extractedMessage || '';
  } catch (e) {
    console.error(e);
    return '';
  }
}

// Judge between two opener candidates; return '1' if first is better, '2' if second
export async function judgeOpener(
  candidate1: string,
  candidate2: string
): Promise<'1' | '2'> {
  try {
    const spec = await getOpenerPromptText();
    const composed = `${spec}\n\n---\ninput_type: evaluate_openers\ncriteria: "Choose the opener most likely to get a response while adhering to the above style_tone, cta_rules, ai_scent_filters, generation_algorithm, and formatting_checks."\ncandidates:\n  - id: 1\n    source: image\n    text: |\n${indentBlock(candidate1 || '<EMPTY>', 6)}\n  - id: 2\n    source: text\n    text: |\n${indentBlock(candidate2 || '<EMPTY>', 6)}\n\nReturn only a JSON object: {\"choice\": <1|2>} with no extra text.`;
    const out = await chat(composed, [], undefined as any);
    try {
      const obj = extractAndParseJSON(out || '');
      const c = Number(obj?.choice);
      return c === 2 ? '2' : '1';
    } catch {
      const t = (out || '').trim();
      if (t.includes('2')) return '2';
      return '1';
    }
  } catch (e) {
    console.error('judgeOpener error', e);
    return '1';
  }
}

export interface ImageScore {
  score: number;
  class: string;
}

export async function getImageScore(url: string): Promise<ImageScore> {
  const endpoint = 'http://localhost:5200/predict';
  const params = { image_url: url };
  try {
    return (await axios.post<ImageScore>(endpoint, params)).data;
  } catch (e) {
    console.error('getImageScore error: Make sure to run siglip server!', e);
    throw e;
  }
}

export async function saveImage(
  url: string,
  decision: 'like' | null
): Promise<void> {
  const endpoint = 'http://localhost:5200/save';
  const params = { image_url: url, decision: decision };
  try {
    await axios.post(endpoint, params);
  } catch (e) {
    console.error('saveImage error', e);
  }
}

export async function downloadImageToBase64(url: string): Promise<string> {
  const response = await axios.get(url, { responseType: 'arraybuffer' });
  return Buffer.from(response.data).toString('base64');
}

export { ChatCompletionMessageParam };
