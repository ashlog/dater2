import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import { query as agentQuery, unstable_v2_createSession } from '@anthropic-ai/claude-agent-sdk';
import type { SDKSession, SDKMessage } from '@anthropic-ai/claude-agent-sdk';
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

export type LLMProviderName = 'openrouter' | 'anthropic' | 'claude-agent' | 'claude-direct';

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

// --- Cost tracker (uses cost from OpenRouter response) ---
const costTracker = {
  inputTokens: 0,
  outputTokens: 0,
  cacheReadTokens: 0,
  cacheWriteTokens: 0,
  totalCost: 0,
  calls: 0,
};

function trackUsage(usage: any, cost?: number) {
  if (!usage) return;
  // OpenRouter/OpenAI format: prompt_tokens, completion_tokens
  // Anthropic format: input_tokens, output_tokens
  costTracker.inputTokens += usage.prompt_tokens ?? usage.input_tokens ?? 0;
  costTracker.outputTokens += usage.completion_tokens ?? usage.output_tokens ?? 0;
  costTracker.cacheReadTokens +=
    usage.prompt_tokens_details?.cached_tokens ?? usage.cache_read_input_tokens ?? 0;
  costTracker.cacheWriteTokens +=
    usage.prompt_tokens_details?.cache_write_tokens ?? usage.cache_creation_input_tokens ?? 0;
  costTracker.totalCost += cost ?? usage.cost ?? 0;
  costTracker.calls++;
}

export function getCostSummary() {
  return { ...costTracker };
}

export function resetCostTracker() {
  costTracker.inputTokens = 0;
  costTracker.outputTokens = 0;
  costTracker.cacheReadTokens = 0;
  costTracker.cacheWriteTokens = 0;
  costTracker.totalCost = 0;
  costTracker.calls = 0;
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
      // Route to Anthropic directly so cache_control works
      provider: { order: ['anthropic'], allow_fallbacks: true },
      ...(rest.responseFormat
        ? { response_format: { type: rest.responseFormat } as any }
        : {}),
    } as any);

    const content = response.choices?.[0]?.message?.content ?? '';
    trackUsage(response.usage);
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
    trackUsage(response.usage);
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

  constructor(apiKey: string | undefined, opts?: { defaultHeaders?: Record<string, string>; baseURL?: string; authToken?: string }) {
    const key = apiKey || opts?.authToken;
    this.client = key
      ? new Anthropic({
          // When using authToken (OAuth), explicitly null out apiKey to prevent
          // the SDK from picking up ANTHROPIC_API_KEY from env
          ...(opts?.authToken
            ? { authToken: opts.authToken, apiKey: null as any }
            : { apiKey: key }),
          ...(opts?.defaultHeaders ? { defaultHeaders: opts.defaultHeaders } : {}),
          ...(opts?.baseURL ? { baseURL: opts.baseURL } : {}),
        })
      : null;
  }

  async chat(request: ChatRequest): Promise<LLMResult> {
    if (!this.client) {
      throw new Error('ANTHROPIC_API_KEY is not set');
    }

    const { system, messages } = await this.convertMessages(request.messages);

    const thinkingConfig = this.resolveThinkingConfig(request);
    const isThinking = thinkingConfig?.type === 'enabled';
    const response = await this.client.messages.create({
      model: this.normalizeModel(request.model),
      system: system ?? undefined,
      messages,
      max_tokens: request.maxTokens ?? this.defaultMaxTokens,
      // Anthropic disallows temperature/top_p when thinking is enabled
      ...(isThinking ? {} : { temperature: request.temperature, top_p: request.top_p }),
      stop_sequences: request.stop,
      thinking: thinkingConfig,
    });

    const content = (response.content || [])
      .filter((block: any) => block.type === 'text' && typeof block.text === 'string')
      .map((block: any) => block.text)
      .join('\n');

    trackUsage((response as any).usage);

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

    // Handle data URIs — extract base64 directly instead of sending as URL
    const dataMatch = rawUrl.match(/^data:(image\/[^;]+);base64,(.+)$/);
    if (dataMatch) {
      const block: ImageBlockParam = {
        type: 'image',
        source: {
          type: 'base64',
          media_type: dataMatch[1] as 'image/jpeg' | 'image/png' | 'image/gif' | 'image/webp',
          data: dataMatch[2],
        },
      };
      if (cacheControl) {
        block.cache_control = cacheControl;
      }
      return block;
    }

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
    if (config.type === 'enabled' && typeof config.budget_tokens === 'number') {
      // Ensure max_tokens > budget_tokens so there's room for actual output
      const maxTokens = request.maxTokens ?? this.defaultMaxTokens;
      if (config.budget_tokens >= maxTokens) {
        request.maxTokens = config.budget_tokens + 4096;
      }
    }
    return config;
  }
}

class ClaudeAgentProvider implements LLMProvider {
  readonly name: LLMProviderName = 'claude-agent';

  // Pool of persistent sessions per model — enables concurrent calls.
  private static POOL_SIZE = 4;
  private pools = new Map<string, { sessions: SDKSession[]; queue: Array<(session: SDKSession) => void> }>();
  // Track cumulative cost per session to compute per-call deltas
  private sessionCostAccum = new WeakMap<SDKSession, number>();

  private normalizeModel(model: string): string {
    if (!model) return 'claude-sonnet-4-6';
    const trimmed = model.trim();
    if (trimmed.includes('/')) {
      return trimmed.split('/').pop() || trimmed;
    }
    return trimmed;
  }

  private createSession(model: string): SDKSession {
    return unstable_v2_createSession({
      model,
      permissionMode: 'dontAsk',
      allowedTools: [],
      disallowedTools: ['Read', 'Write', 'Edit', 'Bash', 'Glob', 'Grep', 'Agent', 'NotebookEdit'],
      env: Object.fromEntries(
        Object.entries(process.env).filter(([k]) => k !== 'CLAUDECODE' && k !== 'ANTHROPIC_API_KEY')
      ),
    });
  }

  private async acquireSession(model: string): Promise<SDKSession> {
    let pool = this.pools.get(model);
    if (!pool) {
      pool = { sessions: [], queue: [] };
      // Pre-create pool sessions
      for (let i = 0; i < ClaudeAgentProvider.POOL_SIZE; i++) {
        pool.sessions.push(this.createSession(model));
      }
      this.pools.set(model, pool);
    }

    // If a session is available, take it
    if (pool.sessions.length > 0) {
      return pool.sessions.pop()!;
    }

    // Otherwise wait for one to be released
    return new Promise<SDKSession>((resolve) => {
      pool!.queue.push(resolve);
    });
  }

  private releaseSession(model: string, session: SDKSession): void {
    const pool = this.pools.get(model);
    if (!pool) return;

    // If someone is waiting, give it directly
    const waiter = pool.queue.shift();
    if (waiter) {
      waiter(session);
    } else {
      pool.sessions.push(session);
    }
  }

  private removeSession(model: string, session: SDKSession): void {
    // Replace a dead session with a fresh one
    const pool = this.pools.get(model);
    if (!pool) return;
    try { session.close(); } catch {}
    const fresh = this.createSession(model);
    const waiter = pool.queue.shift();
    if (waiter) {
      waiter(fresh);
    } else {
      pool.sessions.push(fresh);
    }
  }

  cleanup(): void {
    for (const [, pool] of this.pools) {
      for (const session of pool.sessions) {
        try { session.close(); } catch {}
      }
      pool.sessions.length = 0;
      pool.queue.length = 0;
    }
    this.pools.clear();
  }

  private async collectSessionResponse(session: SDKSession): Promise<{ text: string; cost: number; usage: any }> {
    let text = '';
    let cost = 0;
    let usage: any = undefined;
    for await (const msg of session.stream()) {
      if (msg.type === 'assistant') {
        const parts = ((msg as any).message?.content || [])
          .filter((b: any) => b.type === 'text')
          .map((b: any) => b.text);
        text += parts.join('');
      }
      if (msg.type === 'result') {
        cost = (msg as any).subtype === 'success' ? (msg as any).total_cost_usd : 0;
        usage = (msg as any).subtype === 'success' ? (msg as any).usage : undefined;
        break;
      }
    }
    return { text, cost, usage };
  }

  private async clearSession(session: SDKSession): Promise<void> {
    await session.send('/clear');
    // Drain the stream until we get the result (init + result:success)
    for await (const msg of session.stream()) {
      if (msg.type === 'result') break;
    }
  }

  async chat(request: ChatRequest): Promise<LLMResult> {
    // Extract system prompt from system-role messages
    let systemPrompt = '';
    const nonSystemMessages: ChatCompletionMessageParam[] = [];

    for (const msg of request.messages) {
      if (msg.role === 'system') {
        systemPrompt += this.extractText(msg.content) + '\n';
      } else {
        nonSystemMessages.push(msg);
      }
    }

    // Convert non-system messages into Anthropic MessageParam content blocks,
    // downloading any image URLs to base64 so the SDK subprocess never fetches Hinge URLs.
    const anthropicMessages = await this.convertToAnthropicMessages(nonSystemMessages);
    const model = this.normalizeModel(request.model);

    let resultText = '';
    let totalCost = 0;
    let usage: any = undefined;

    const session = await this.acquireSession(model);

    try {
      // Build the user message with system prompt prepended
      const userContent: any[] = [];
      if (systemPrompt.trim()) {
        userContent.push({ type: 'text', text: `[SYSTEM INSTRUCTIONS]\n${systemPrompt.trim()}\n[END SYSTEM INSTRUCTIONS]` });
      }
      for (const msg of anthropicMessages) {
        if (msg.role === 'user') {
          const content = Array.isArray(msg.content) ? msg.content : [{ type: 'text', text: msg.content }];
          userContent.push(...content);
        }
      }

      await session.send({
        type: 'user',
        session_id: '',
        message: { role: 'user', content: userContent },
        parent_tool_use_id: null,
      } as any);

      const response = await this.collectSessionResponse(session);
      resultText = response.text;
      // total_cost_usd is cumulative per session — compute the delta
      const prevCost = this.sessionCostAccum.get(session) || 0;
      totalCost = Math.max(0, response.cost - prevCost);
      this.sessionCostAccum.set(session, response.cost);
      usage = response.usage;

      // Clear context for next call, then return session to pool
      await this.clearSession(session);
      this.releaseSession(model, session);
    } catch (e: any) {
      // Session may have died — replace it in the pool
      this.removeSession(model, session);
      if (!resultText) throw e;
    }

    trackUsage(usage, totalCost);

    return {
      content: resultText,
      usage,
      stopReason: undefined,
    };
  }

  async complete(request: CompletionRequest): Promise<LLMResult> {
    return this.chat({
      model: request.model,
      messages: [{ role: 'user', content: request.prompt }],
      maxTokens: request.maxTokens,
      temperature: request.temperature,
      top_p: request.top_p,
      stop: request.stop,
    });
  }

  private extractText(content: ChatCompletionMessageParam['content']): string {
    if (typeof content === 'string') return content;
    if (!Array.isArray(content)) return '';
    return (content as any[])
      .filter((p) => p.type === 'text')
      .map((p) => p.text || '')
      .filter(Boolean)
      .join('\n');
  }

  /** Convert OpenAI-format messages to Anthropic MessageParam[], downloading images to base64. */
  private async convertToAnthropicMessages(
    messages: ChatCompletionMessageParam[]
  ): Promise<MessageParam[]> {
    const result: MessageParam[] = [];
    for (const msg of messages) {
      const role: 'user' | 'assistant' = msg.role === 'assistant' ? 'assistant' : 'user';
      const content = await this.convertContent(msg.content);
      if (content.length > 0) {
        result.push({ role, content });
      }
    }
    return result;
  }

  /** Convert OpenAI content parts to Anthropic ContentBlockParam[], fetching images as base64. */
  private async convertContent(
    content: ChatCompletionMessageParam['content']
  ): Promise<ContentBlockParam[]> {
    if (typeof content === 'string') {
      return content.trim() ? [{ type: 'text', text: content }] : [];
    }
    if (!Array.isArray(content)) return [];

    const blocks: ContentBlockParam[] = [];
    for (const part of content as any[]) {
      if (!part) continue;
      if (part.type === 'text' && part.text?.trim()) {
        blocks.push({ type: 'text', text: part.text });
      } else if (part.type === 'image_url') {
        const block = await this.convertImagePart(part);
        if (block) blocks.push(block);
      }
    }
    return blocks;
  }

  /** Convert an OpenAI image_url part to an Anthropic ImageBlockParam with base64 data. */
  private async convertImagePart(part: any): Promise<ImageBlockParam | null> {
    const rawUrl: string | undefined =
      typeof part?.image_url === 'string' ? part.image_url : part?.image_url?.url;
    if (!rawUrl) return null;

    // Already a data URI — extract base64 directly
    const dataMatch = rawUrl.match(/^data:(image\/[^;]+);base64,(.+)$/);
    if (dataMatch) {
      return {
        type: 'image',
        source: {
          type: 'base64',
          media_type: dataMatch[1] as 'image/jpeg' | 'image/png' | 'image/gif' | 'image/webp',
          data: dataMatch[2],
        },
      };
    }

    // Regular URL — download and convert to base64
    try {
      const resp = await axios.get(rawUrl, { responseType: 'arraybuffer' });
      const contentType = resp.headers['content-type'] || 'image/jpeg';
      const mediaType = contentType.split(';')[0].trim() as
        'image/jpeg' | 'image/png' | 'image/gif' | 'image/webp';
      const data = Buffer.from(resp.data).toString('base64');
      return {
        type: 'image',
        source: { type: 'base64', media_type: mediaType, data },
      };
    } catch (e) {
      console.error('ClaudeAgentProvider: failed to download image for base64 conversion', e);
      return null;
    }
  }
}

/**
 * Extracts the OAuth token from a claude-agent session via mitmproxy,
 * then uses the standard Anthropic SDK for direct API calls.
 * Same auth as claude-agent, but no subprocess or Claude Code system prompt overhead.
 */
class ClaudeDirectProvider implements LLMProvider {
  readonly name: LLMProviderName = 'claude-direct';
  private inner: AnthropicProvider | null = null;
  private initPromise: Promise<void> | null = null;

  // Headers to skip when copying (managed by the SDK or transport layer)
  private static readonly HEADERS_TO_SKIP = new Set([
    'host',
    'connection',
    'content-type',
    'content-length',
    'accept',
    'accept-encoding',
    'accept-language',
    'sec-fetch-mode',
    'authorization',
  ]);

  private createInnerProvider(token: string, headers: Record<string, string>): AnthropicProvider {
    const provider = new AnthropicProvider(undefined, { authToken: token, defaultHeaders: headers });
    (provider as any).name = 'claude-direct';
    return provider;
  }

  private static parseHeadersFromLog(log: string): Record<string, string> {
    // Find the POST /v1/messages request block and extract ALL headers
    const requestMatch = log.match(/POST https:\/\/api\.anthropic\.com\/v1\/messages[^\n]*\n([\s\S]*?)(?:\n\n|\n *<<|\n\d)/);
    if (!requestMatch) return {};

    const headerBlock = requestMatch[1];
    const headers: Record<string, string> = {};
    for (const line of headerBlock.split('\n')) {
      const m = line.match(/^\s+([^:]+):\s*(.+)$/);
      if (m) {
        const key = m[1].trim();
        const val = m[2].trim();
        if (!ClaudeDirectProvider.HEADERS_TO_SKIP.has(key.toLowerCase())) {
          headers[key] = val;
        }
      }
    }
    return headers;
  }

  private async ensureInit(): Promise<AnthropicProvider> {
    if (this.inner) return this.inner;
    if (!this.initPromise) {
      this.initPromise = this.extractTokenAndInit();
    }
    await this.initPromise;
    return this.inner!;
  }

  private async extractTokenAndInit(): Promise<void> {
    const http = require('http');
    const net = require('net');
    const { createServer } = require('https');

    // Start mitmproxy-style CONNECT proxy to capture the OAuth token
    // We use a simple CONNECT proxy + TLS MITM with the mitmproxy CA
    // Simpler approach: spawn a one-shot claude session through our local proxy
    // and read the token from the mitmproxy capture that's already been done.

    // First try: check if we already have a cached token
    const tokenCachePath = require('path').join(require('os').homedir(), '.claude', '.oauth-token-cache');
    const fsSync = require('fs');
    if (fsSync.existsSync(tokenCachePath)) {
      try {
        const cached = JSON.parse(fsSync.readFileSync(tokenCachePath, 'utf-8'));
        if (cached.token && cached.headers && cached.expiresAt && Date.now() < cached.expiresAt) {
          console.log('[claude-direct] Using cached OAuth token + headers');
          this.inner = this.createInnerProvider(cached.token, cached.headers);
          return;
        }
      } catch {}
    }

    // Extract token by running mitmproxy capture
    console.log('[claude-direct] Extracting OAuth token via mitmproxy...');
    const { execSync, spawn } = require('child_process');

    // Find a free port for mitmdump
    const getPort = (): Promise<number> => new Promise((resolve) => {
      const srv = net.createServer();
      srv.listen(0, '127.0.0.1', () => {
        const port = srv.address().port;
        srv.close(() => resolve(port));
      });
    });

    const port = await getPort();
    const logFile = `/tmp/claude_direct_capture_${Date.now()}.log`;

    // Start mitmdump
    const mitm = spawn('mitmdump', ['--listen-port', String(port), '--set', 'flow_detail=3'], {
      stdio: ['ignore', fsSync.openSync(logFile, 'w'), fsSync.openSync(logFile, 'a')],
    });

    await new Promise((r) => setTimeout(r, 2000)); // let mitmdump start

    // Spawn a one-shot claude session through the proxy
    const session = unstable_v2_createSession({
      model: 'claude-haiku-4-5',
      permissionMode: 'dontAsk',
      allowedTools: [],
      disallowedTools: ['Read', 'Write', 'Edit', 'Bash', 'Glob', 'Grep', 'Agent', 'NotebookEdit'],
      env: {
        ...Object.fromEntries(
          Object.entries(process.env).filter(([k]) => k !== 'CLAUDECODE' && k !== 'ANTHROPIC_API_KEY')
        ),
        HTTPS_PROXY: `http://127.0.0.1:${port}`,
        HTTP_PROXY: `http://127.0.0.1:${port}`,
        NODE_TLS_REJECT_UNAUTHORIZED: '0',
        SSL_CERT_FILE: require('path').join(require('os').homedir(), '.mitmproxy', 'mitmproxy-ca-cert.pem'),
      },
    });

    await session.send('Say "ok"');
    for await (const msg of session.stream()) {
      if (msg.type === 'result') break;
    }
    session.close();

    // Kill mitmdump and read the log
    mitm.kill();
    await new Promise((r) => setTimeout(r, 500));

    const log = fsSync.readFileSync(logFile, 'utf-8');
    const tokenMatch = log.match(/[Aa]uthorization: Bearer (sk-ant-oat01-[^\s]+)/);
    const headers = ClaudeDirectProvider.parseHeadersFromLog(log);
    fsSync.unlinkSync(logFile);

    if (!tokenMatch) {
      throw new Error('[claude-direct] Failed to extract OAuth token from capture');
    }

    const token = tokenMatch[1];
    console.log(`[claude-direct] Got OAuth token: ${token.slice(0, 20)}...`);
    console.log(`[claude-direct] Extracted headers:`, Object.keys(headers));

    // Cache token + headers for 24 hours
    fsSync.writeFileSync(tokenCachePath, JSON.stringify({
      token,
      headers,
      expiresAt: Date.now() + 24 * 60 * 60 * 1000,
    }));

    this.inner = this.createInnerProvider(token, headers);
  }

  private invalidateAndRetry(): void {
    // Clear cached token so next ensureInit re-extracts
    const tokenCachePath = require('path').join(require('os').homedir(), '.claude', '.oauth-token-cache');
    try { require('fs').unlinkSync(tokenCachePath); } catch {}
    this.inner = null;
    this.initPromise = null;
    console.log('[claude-direct] Token expired, will re-extract on next call');
  }

  private isAuthError(e: any): boolean {
    return e?.status === 401 || e?.error?.type === 'authentication_error';
  }

  async chat(request: ChatRequest): Promise<LLMResult> {
    const provider = await this.ensureInit();
    try {
      return await provider.chat(request);
    } catch (e: any) {
      if (this.isAuthError(e)) {
        this.invalidateAndRetry();
        const fresh = await this.ensureInit();
        return fresh.chat(request);
      }
      throw e;
    }
  }

  async complete(request: CompletionRequest): Promise<LLMResult> {
    const provider = await this.ensureInit();
    try {
      return await provider.complete!(request);
    } catch (e: any) {
      if (this.isAuthError(e)) {
        this.invalidateAndRetry();
        const fresh = await this.ensureInit();
        return fresh.complete!(request);
      }
      throw e;
    }
  }
}

const providers: Record<LLMProviderName, LLMProvider> = {
  openrouter: new OpenRouterProvider(OPENROUTER_API_KEY),
  anthropic: new AnthropicProvider(ANTHROPIC_API_KEY),
  'claude-agent': new ClaudeAgentProvider(),
  'claude-direct': new ClaudeDirectProvider(),
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

export function cleanupLLMProvider(): void {
  const provider = getProvider();
  if ('cleanup' in provider && typeof (provider as any).cleanup === 'function') {
    (provider as any).cleanup();
  }
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

export function setOpenerPromptOverride(text: string | null): void {
  openerPromptCache = text;
  openerPromptHashCache = text ? createHash('sha256').update(text).digest('hex') : null;
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

  // Extract JSON from first {/[ to last }/], ignoring markdown code blocks and other wrapper text
  const firstBrace = trimmed.indexOf('{');
  const firstBracket = trimmed.indexOf('[');
  const isArray = firstBracket !== -1 && (firstBrace === -1 || firstBracket < firstBrace);

  if (isArray) {
    const lastBracket = trimmed.lastIndexOf(']');
    if (lastBracket > firstBracket) {
      return JSON.parse(trimmed.substring(firstBracket, lastBracket + 1));
    }
  }

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

// Best-of-N generation: generate N batches in parallel, collect candidates, pick best via LLM.
export async function generateOpenersFromYamlBatchBestOfN(
  items: { id: string; profilePrompt: string }[],
  herName: string,
  model = 'gemini-2.5-flash',
  n = 3
): Promise<Map<string, string>> {
  // Generate N batches in parallel
  const batches = await Promise.all(
    Array.from({ length: n }, () => generateOpenersFromYamlBatch(items, herName, model))
  );

  // Collect candidates per id
  const candidatesById = new Map<string, string[]>();
  for (const batch of batches) {
    for (const [id, text] of batch) {
      if (!text) continue;
      const list = candidatesById.get(id) || [];
      list.push(text);
      candidatesById.set(id, list);
    }
  }

  // For items with only 1 candidate, use it directly
  const result = new Map<string, string>();
  const toSelect: { id: string; candidates: string[]; prompt: string }[] = [];

  for (const item of items) {
    const candidates = candidatesById.get(item.id) || [];
    if (candidates.length <= 1) {
      if (candidates[0]) result.set(item.id, candidates[0]);
    } else {
      // Deduplicate
      const unique = [...new Set(candidates)];
      if (unique.length === 1) {
        result.set(item.id, unique[0]);
      } else {
        toSelect.push({ id: item.id, candidates: unique, prompt: item.profilePrompt });
      }
    }
  }

  // Use Anthropic (Haiku) to pick the best candidate — direct call, no opener system prompt
  if (toSelect.length > 0) {
    const pickerModel = 'claude-haiku-4-5';
    const selectionItems = toSelect.map((s) => {
      const opts = s.candidates.map((c, j) => `${String.fromCharCode(65 + j)}) "${c}"`).join('\n');
      return `ID: ${s.id}\nHer prompt: ${s.prompt}\n${opts}`;
    }).join('\n---\n');

    const pickerPrompt = `Pick the best Hinge opener for each prompt. Choose the one that:
- Is wittiest and has a clear comedic twist
- Is shortest and punchiest
- Ends with something she'd reply to
- Sounds most natural/human

Return JSON only: [{"id": "...", "pick": "A"}]

${selectionItems}`;

    try {
      const pickResult = await getProvider().chat({
        model: pickerModel,
        messages: [{ role: 'user', content: pickerPrompt }],
        maxTokens: 1024,
      });
      const picks = extractAndParseJSON(pickResult.content);
      const pickArr = Array.isArray(picks) ? picks : picks?.items || [];
      for (const pick of pickArr) {
        const match = toSelect.find((s) => s.id === pick.id);
        if (match) {
          const idx = (pick.pick || 'A').charCodeAt(0) - 65;
          result.set(match.id, match.candidates[Math.min(idx, match.candidates.length - 1)]);
        }
      }
      trackUsage(pickResult.usage);
    } catch (e) {
      console.error('Best-of-N selection failed, falling back to first candidate:', e);
    }

    // Fill any remaining with first candidate
    for (const s of toSelect) {
      if (!result.has(s.id)) {
        result.set(s.id, s.candidates[0]);
      }
    }
  }

  return result;
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

export async function getImageScore(url: string, retries = 2): Promise<ImageScore> {
  const endpoint = 'http://localhost:5200/predict';
  const params = { image_url: url };
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return (await axios.post<ImageScore>(endpoint, params)).data;
    } catch (e: any) {
      const status = e?.response?.status;
      if (attempt < retries && status === 400) {
        console.warn(`getImageScore 400 for ${url}, retrying (${attempt + 1}/${retries})...`);
        await new Promise(r => setTimeout(r, 1000 * (attempt + 1)));
        continue;
      }
      console.error('getImageScore error: Make sure to run siglip server!', e);
      throw e;
    }
  }
  throw new Error('getImageScore: unreachable');
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
