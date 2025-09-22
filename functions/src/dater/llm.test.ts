import { extractAndParseJSON } from './llm';

describe('extractAndParseJSON', () => {
  it('should parse clean JSON', () => {
    const input = '{"text": "hello"}';
    const result = extractAndParseJSON(input);
    expect(result).toEqual({ text: 'hello' });
  });

  it('should extract JSON from markdown code blocks', () => {
    const input = '```json\n{"text": "hello"}\n```';
    const result = extractAndParseJSON(input);
    expect(result).toEqual({ text: 'hello' });
  });

  it('should handle markdown with backticks prefix', () => {
    const input = '```json\n{\n  "items": [\n    {"id": 1, "text": "first"},\n    {"id": 2, "text": "second"}\n  ]\n}\n```';
    const result = extractAndParseJSON(input);
    expect(result).toEqual({
      items: [
        { id: 1, text: 'first' },
        { id: 2, text: 'second' }
      ]
    });
  });

  it('should extract JSON from text with prefix', () => {
    const input = 'Here is the result:\n{"text": "hello"}';
    const result = extractAndParseJSON(input);
    expect(result).toEqual({ text: 'hello' });
  });

  it('should extract JSON from text with suffix', () => {
    const input = '{"text": "hello"}\nThat was the result.';
    const result = extractAndParseJSON(input);
    expect(result).toEqual({ text: 'hello' });
  });

  it('should extract JSON from text with both prefix and suffix', () => {
    const input = 'Here is the JSON:\n{"text": "hello"}\nEnd of JSON.';
    const result = extractAndParseJSON(input);
    expect(result).toEqual({ text: 'hello' });
  });

  it('should handle nested objects', () => {
    const input = '```json\n{"outer": {"inner": {"deep": "value"}}}\n```';
    const result = extractAndParseJSON(input);
    expect(result).toEqual({ outer: { inner: { deep: 'value' } } });
  });

  it('should handle arrays at root level with wrapper object', () => {
    const input = 'Some text before\n{"items": [{"id": 1}, {"id": 2}]}\nSome text after';
    const result = extractAndParseJSON(input);
    expect(result).toEqual({ items: [{ id: 1 }, { id: 2 }] });
  });

  it('should throw error for empty string', () => {
    expect(() => extractAndParseJSON('')).toThrow('Empty text provided');
  });

  it('should throw error for text without JSON', () => {
    expect(() => extractAndParseJSON('Just some text')).toThrow('No valid JSON object found');
  });

  it('should throw error for invalid JSON', () => {
    expect(() => extractAndParseJSON('{invalid json}')).toThrow();
  });

  it('should handle whitespace around JSON', () => {
    const input = '   \n  {"text": "hello"}  \n  ';
    const result = extractAndParseJSON(input);
    expect(result).toEqual({ text: 'hello' });
  });

  it('should handle real-world example from error log', () => {
    const input = '```json\n{\n  "items": [\n    {\n      "id": 1,\n      "text": "lettuce grab food and test who\'s funnier?"\n    },\n    {\n      "id": 2,\n      "text": "Stranger Things, convince me otherwise over coffee?"\n    }\n  ]\n}\n```';
    const result = extractAndParseJSON(input);
    expect(result).toEqual({
      items: [
        { id: 1, text: "lettuce grab food and test who's funnier?" },
        { id: 2, text: "Stranger Things, convince me otherwise over coffee?" }
      ]
    });
  });

  it('should handle incomplete JSON in markdown (truncated case)', () => {
    const input = '```json\n{\n  "items": [\n    {\n      "id": 1,\n      "text": "first"\n    },\n    {\n      "id": 2,\n      "text": "secon';
    // This should throw because the JSON is incomplete
    expect(() => extractAndParseJSON(input)).toThrow();
  });
});
