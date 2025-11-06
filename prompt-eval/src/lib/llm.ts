import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import type {
  TextBlockParam,
  ImageBlockParam,
  MessageParam,
  ContentBlockParam,
} from '@anthropic-ai/sdk/resources/messages/index.js';

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;

export type LLMProviderName = 'openrouter' | 'anthropic';

interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string | Array<{ type: string; text?: string; image_url?: { url: string } }>;
}

interface ChatRequest {
  model: string;
  messages: ChatMessage[];
  temperature?: number;
  maxTokens?: number;
}

interface LLMResult {
  content: string;
  stopReason?: string;
}

interface LLMProvider {
  readonly name: LLMProviderName;
  chat(request: ChatRequest): Promise<LLMResult>;
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
    const response = await this.client.chat.completions.create({
      model: request.model,
      messages: request.messages as any,
      temperature: request.temperature,
      max_tokens: request.maxTokens,
    });

    const content = response.choices?.[0]?.message?.content ?? '';
    return {
      content,
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

    const { system, messages } = this.convertMessages(request.messages);

    const response = await this.client.messages.create({
      model: this.normalizeModel(request.model),
      system: system ?? undefined,
      messages,
      max_tokens: request.maxTokens ?? this.defaultMaxTokens,
      temperature: request.temperature,
    });

    const content = (response.content || [])
      .filter((block): block is Anthropic.TextBlock => block.type === 'text')
      .map((block) => block.text)
      .join('\n');

    return {
      content,
      stopReason: response.stop_reason ?? undefined,
    };
  }

  private normalizeModel(model: string): string {
    if (!model) return 'claude-3-5-sonnet-20241022';
    const trimmed = model.trim();
    const aliasMap: Record<string, string> = {
      'anthropic/claude-haiku-4.5': 'claude-haiku-4-5-20251001',
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

  private convertMessages(messages: ChatMessage[]): {
    system: TextBlockParam[] | null;
    messages: MessageParam[];
  } {
    const systemParts: TextBlockParam[] = [];
    const converted: MessageParam[] = [];

    for (const msg of messages) {
      const contentBlocks = this.convertContent(msg.content);
      if (msg.role === 'system') {
        const textBlocks = contentBlocks.filter(
          (block): block is TextBlockParam => block.type === 'text'
        );
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

  private convertContent(content: ChatMessage['content']): ContentBlockParam[] {
    if (typeof content === 'string') {
      if (!content.trim()) return [];
      return [{ type: 'text', text: content }];
    }

    if (!Array.isArray(content)) {
      return [];
    }

    const blocks: ContentBlockParam[] = [];
    for (const part of content) {
      if (!part) continue;
      if (part.type === 'text' && part.text) {
        blocks.push({ type: 'text', text: part.text });
      } else if (part.type === 'image_url' && part.image_url?.url) {
        blocks.push({
          type: 'image',
          source: { type: 'url', url: part.image_url.url },
        } as ImageBlockParam);
      }
    }
    return blocks;
  }
}

const providers: Record<LLMProviderName, LLMProvider> = {
  openrouter: new OpenRouterProvider(OPENROUTER_API_KEY),
  anthropic: new AnthropicProvider(ANTHROPIC_API_KEY),
};

let activeProvider: LLMProviderName =
  (process.env.LLM_PROVIDER as LLMProviderName) || 'anthropic';

function getProvider(): LLMProvider {
  return providers[activeProvider] || providers.anthropic;
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

// Match the functions project default model
const DEFAULT_MODEL = 'anthropic/claude-haiku-4.5';

export async function chat(
  prompt: string,
  systemPrompt?: string,
  model = DEFAULT_MODEL
): Promise<string> {
  const provider = getProvider();
  const messages: ChatMessage[] = [];

  if (systemPrompt) {
    messages.push({ role: 'system', content: systemPrompt });
  }
  messages.push({ role: 'user', content: prompt });

  const result = await provider.chat({
    model,
    messages,
    temperature: 1,
  });

  return result.content;
}

export async function chatWithImage(
  prompt: string,
  imageUrl: string,
  systemPrompt?: string,
  model = DEFAULT_MODEL
): Promise<string> {
  const provider = getProvider();
  const messages: ChatMessage[] = [];

  if (systemPrompt) {
    messages.push({ role: 'system', content: systemPrompt });
  }

  messages.push({
    role: 'user',
    content: [
      { type: 'text', text: prompt },
      { type: 'image_url', image_url: { url: imageUrl } },
    ],
  });

  const result = await provider.chat({
    model,
    messages,
    temperature: 1,
  });

  return result.content;
}

// Export for direct use
export { DEFAULT_MODEL };
