import { weatherTool } from '@/tool/weather-tool';
import { deepseek } from '@ai-sdk/deepseek';
import type { InferAgentUIMessage } from 'ai';
import { ToolLoopAgent } from 'ai';

export const deepseekToolsAgent = new ToolLoopAgent({
  model: deepseek('deepseek-reasoner'),
  tools: { weather: weatherTool },
});

export type DeepSeekToolsAgentMessage = InferAgentUIMessage<
  typeof deepseekToolsAgent
>;
