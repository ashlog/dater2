import fs from 'fs';

const ORIGINAL_ENV = { ...process.env };

function mockOpenAI(chatImpl = jest.fn(), completionsImpl = jest.fn()) {
  return {
    __esModule: true,
    default: jest.fn().mockImplementation(() => ({
      chat: { completions: { create: chatImpl } },
      completions: { create: completionsImpl },
    })),
  };
}

function mockAnthropic(messagesImpl = jest.fn()) {
  return {
    __esModule: true,
    default: jest.fn().mockImplementation(() => ({
      messages: { create: messagesImpl },
    })),
  };
}

describe('LLM provider switching', () => {
  beforeEach(() => {
    jest.resetModules();
    jest.clearAllMocks();
    process.env = {
      ...ORIGINAL_ENV,
      OPENROUTER_API_KEY: 'openrouter-test',
      ANTHROPIC_API_KEY: 'anthropic-test',
    };
  });

  afterEach(() => {
    process.env = { ...ORIGINAL_ENV };
  });

  it('uses OpenRouter provider by default', async () => {
    const chatCreateMock = jest.fn().mockResolvedValue({
      choices: [{ message: { content: 'hello world' } }],
      usage: { total_tokens: 42 },
    });

    jest.doMock('openai', () => mockOpenAI(chatCreateMock));
    jest.doMock('@anthropic-ai/sdk', () => mockAnthropic());

    const { chat } = await import('./llm');
    const preprompt = [
      {
        role: 'system',
        content: [
          {
            type: 'text',
            text: 'cached system prompt',
            cache_control: { type: 'ephemeral' },
          },
        ],
      },
    ];
    const result = await chat('test prompt', preprompt, 'openai/gpt-test');

    expect(result).toBe('hello world');
    expect(chatCreateMock).toHaveBeenCalledTimes(1);
    expect(chatCreateMock).toHaveBeenCalledWith(
      expect.objectContaining({ model: 'openai/gpt-test' })
    );
    const messages = chatCreateMock.mock.calls[0][0].messages as any[];
    expect(messages[0].content[0].cache_control).toEqual({ type: 'ephemeral' });
  });

  it('switches to Anthropic provider when configured', async () => {
    const anthropicCreateMock = jest.fn().mockResolvedValue({
      content: [{ type: 'text', text: 'anthropic reply' }],
    });
    const openaiCreateMock = jest.fn().mockResolvedValue({
      choices: [{ message: { content: 'openrouter reply' } }],
    });

    jest.doMock('openai', () => mockOpenAI(openaiCreateMock));
    jest.doMock('@anthropic-ai/sdk', () => mockAnthropic(anthropicCreateMock));

    const { chat, setLLMProvider, setLLMThinkingConfig } = await import('./llm');
    setLLMProvider('anthropic');
    setLLMThinkingConfig({ type: 'enabled', budget_tokens: 1024 });

    const preprompt = [
      {
        role: 'system',
        content: [
          {
            type: 'text',
            text: 'anthropic instructions',
            cache_control: { type: 'ephemeral' },
          },
        ],
      },
    ];

    const result = await chat('anthropic prompt', preprompt, 'anthropic/claude-sonnet-4.5');

    expect(result).toBe('anthropic reply');
    expect(anthropicCreateMock).toHaveBeenCalledTimes(1);
    const callArgs = anthropicCreateMock.mock.calls[0]?.[0] as any;
    expect(callArgs.model).toBeDefined();
    expect(callArgs.model).not.toContain('anthropic/');
    expect(callArgs.model).toContain('claude');
    expect(Array.isArray(callArgs.system)).toBe(true);
    expect(callArgs.system).toEqual([
      {
        type: 'text',
        text: 'anthropic instructions',
        cache_control: { type: 'ephemeral' },
      },
    ]);
    expect(callArgs.thinking).toEqual({ type: 'enabled', budget_tokens: 1024 });
    expect(openaiCreateMock).not.toHaveBeenCalled();
  });
});

describe('getOpenerPromptMetadata caching', () => {
  beforeEach(() => {
    jest.resetModules();
    jest.clearAllMocks();
    process.env = {
      ...ORIGINAL_ENV,
      OPENROUTER_API_KEY: 'openrouter-test',
      ANTHROPIC_API_KEY: 'anthropic-test',
    };
  });

  afterEach(() => {
    process.env = { ...ORIGINAL_ENV };
  });

  it('reads opener prompt file once and caches subsequent calls', async () => {
    const chatCreateMock = jest.fn().mockResolvedValue({
      choices: [{ message: { content: 'ok' } }],
    });
    jest.doMock('openai', () => mockOpenAI(chatCreateMock));
    jest.doMock('@anthropic-ai/sdk', () => mockAnthropic());

    const readFileSpy = jest.spyOn(fs.promises, 'readFile');

    const module = await import('./llm');
    await module.getOpenerPromptMetadata();
    await module.getOpenerPromptMetadata();

    expect(readFileSpy).toHaveBeenCalledTimes(1);

    readFileSpy.mockRestore();
  });
});
