---
'@ai-sdk/openai': patch
---

feat(openai): preserve `namespace` on function_call output items

When the OpenAI Responses API dispatches a tool call from inside a `namespace` tool wrapper, the response includes a `namespace` field on the resulting `function_call` output item identifying which namespace the dispatch came from. The provider's response schema previously didn't include `namespace`, so the field was stripped during validation and downstream consumers couldn't access it.

Adds `namespace` to the function_call schemas (response output, output items, and `response.output_item.done` stream events) and surfaces it via `providerMetadata.openai.namespace` on the resulting tool-call content/stream parts. Optional / nullish — no impact on requests that don't use namespace tools.
