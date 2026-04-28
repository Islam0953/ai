import { LangSmithDeploymentTransport } from './transport';
import { describe, it, expect } from 'vitest';

describe('LangSmithDeploymentTransport', () => {
  it('should create transport with options', () => {
    const transport = new LangSmithDeploymentTransport({
      url: 'https://test.langsmith.app',
      apiKey: 'test-key',
    });

    expect(typeof transport.sendMessages).toBe('function');
    expect(typeof transport.reconnectToStream).toBe('function');
  });

  it('should create transport with only url', () => {
    const transport = new LangSmithDeploymentTransport({
      url: 'https://test.langsmith.app',
    });

    expect(typeof transport.sendMessages).toBe('function');
  });

  it('should create transport with custom graphId', () => {
    const transport = new LangSmithDeploymentTransport({
      url: 'https://test.langsmith.app',
      graphId: 'custom-agent',
    });

    expect(typeof transport.sendMessages).toBe('function');
  });

  it('should throw error for reconnectToStream', async () => {
    const transport = new LangSmithDeploymentTransport({
      url: 'https://test.langsmith.app',
    });

    await expect(
      transport.reconnectToStream({ chatId: 'chat-1' }),
    ).rejects.toThrow('Method not implemented.');
  });
});
