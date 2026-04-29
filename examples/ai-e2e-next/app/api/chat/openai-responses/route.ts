import { openai } from '@ai-sdk/openai';
import type { OpenAILanguageModelResponsesOptions } from '@ai-sdk/openai';
import type { UIMessage } from 'ai';
import { convertToModelMessages, streamText } from 'ai';

export const maxDuration = 30;

export async function POST(req: Request) {
  const { messages }: { messages: UIMessage[] } = await req.json();

  const result = streamText({
    model: openai.responses('o3-mini'),
    messages: await convertToModelMessages(messages),
    reasoning: 'low',
    providerOptions: {
      openai: {
        reasoningSummary: 'auto',
      } satisfies OpenAILanguageModelResponsesOptions,
    },
  });

  return result.toUIMessageStreamResponse();
}
