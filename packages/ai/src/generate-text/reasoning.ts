import { ReasoningPart, ReasoningFilePart } from '@ai-sdk/provider-utils';

export function asReasoningText(
  reasoningParts: Array<ReasoningPart | ReasoningFilePart>,
): string | undefined {
  const reasoningText = reasoningParts
    .map(part => (Object.hasOwn(part, 'text') ? part.text : ''))
    .join('');
  return reasoningText.length > 0 ? reasoningText : undefined;
}
