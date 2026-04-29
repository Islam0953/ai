import type { OpenAILanguageModelResponsesOptions } from '@ai-sdk/openai';
import { openai } from '@ai-sdk/openai';
import type { InferAgentUIMessage } from 'ai';
import { ToolLoopAgent } from 'ai';

export const openaiWebSearchAgent = new ToolLoopAgent({
  model: openai('gpt-5-mini'),
  tools: {
    webSearch: openai.tools.webSearch({
      searchContextSize: 'low',
      userLocation: {
        type: 'approximate',
        city: 'San Francisco',
        region: 'California',
        country: 'US',
      },
    }),
  },
  reasoning: 'medium',
  providerOptions: {
    openai: {
      reasoningSummary: 'detailed',
      // store: false,
    } satisfies OpenAILanguageModelResponsesOptions,
  },
  onStepFinish: ({ request }) => {
    console.dir(request.body, { depth: Infinity });
  },
});

export type OpenAIWebSearchMessage = InferAgentUIMessage<
  typeof openaiWebSearchAgent
>;
