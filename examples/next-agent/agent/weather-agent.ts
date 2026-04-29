import { weatherTool } from '@/tool/weather-tool';
import { openai } from '@ai-sdk/openai';
import type { InferAgentUIMessage } from 'ai';
import { ToolLoopAgent } from 'ai';

export const weatherAgent = new ToolLoopAgent({
  model: openai('gpt-4o'),
  instructions: 'You are a helpful assistant.',
  tools: {
    weather: weatherTool,
  },
});

export type WeatherAgentUIMessage = InferAgentUIMessage<typeof weatherAgent>;
