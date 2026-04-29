import { google } from '@ai-sdk/google';
import type { UIMessage } from 'ai';
import { convertToModelMessages, streamText } from 'ai';

export async function POST(req: Request) {
  const { messages }: { messages: UIMessage[] } = await req.json();

  const result = streamText({
    model: google('gemini-3-flash-preview'),
    tools: {
      google_search: google.tools.googleSearch({}),
    },
    messages: await convertToModelMessages(messages),
  });

  return result.toUIMessageStreamResponse({
    sendSources: true,
  });
}
