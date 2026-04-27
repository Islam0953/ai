---
"@ai-sdk/openai": patch
"@ai-sdk/provider": patch
---

feat(provider, openai): preserve `start_index` / `end_index` on URL citation source parts

OpenAI's chat and responses APIs return `start_index` and `end_index` on each `url_citation` annotation, identifying the span of `output_text` the citation applies to. The OpenAI provider was reading these from the response, validating them, and then dropping them when constructing `LanguageModelV3Source` / `LanguageModelV4Source` parts — so downstream consumers received URLs and titles but no positional info.

Adds optional `startIndex` and `endIndex` to the URL variant of both `LanguageModelV3Source` and `LanguageModelV4Source`, and populates them in the OpenAI provider's chat (non-streaming + streaming) and responses (non-streaming + streaming) paths. Optional because not every provider supplies positional info; existing call sites that don't populate them keep working.
