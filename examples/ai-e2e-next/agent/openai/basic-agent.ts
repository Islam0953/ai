import type { OpenAILanguageModelResponsesOptions } from '@ai-sdk/openai';
import { openai } from '@ai-sdk/openai';
import type { InferAgentUIMessage } from 'ai';
import { ToolLoopAgent } from 'ai';

export const openaiBasicAgent = new ToolLoopAgent({
  model: openai('gpt-5-mini'),
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

export type OpenAIBasicMessage = InferAgentUIMessage<typeof openaiBasicAgent>;
