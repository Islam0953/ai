import {
  Attributes,
  AttributeValue,
  context,
  Context as OpenTelemetryContext,
  Span,
  SpanKind,
  SpanStatusCode,
  trace,
  Tracer,
} from '@opentelemetry/api';
import type { LanguageModelV4Prompt } from '@ai-sdk/provider';
import type {
  EmbeddingModelCallEndEvent,
  EmbedEndEvent,
  EmbedStartEvent,
  LanguageModelCallEndEvent,
  LanguageModelCallStartEvent,
  EmbeddingModelCallStartEvent,
  GenerateObjectEndEvent,
  GenerateObjectStartEvent,
  GenerateObjectStepEndEvent,
  GenerateObjectStepStartEvent,
  StreamTextChunkEvent,
  GenerateTextEndEvent,
  GenerateTextStartEvent,
  GenerateTextStepEndEvent,
  GenerateTextStepStartEvent,
  ToolExecutionEndEvent,
  ToolExecutionStartEvent,
  RerankingModelCallEndEvent,
  RerankEndEvent,
  RerankStartEvent,
  RerankingModelCallStartEvent,
  InferTelemetryEvent,
  Telemetry,
  TelemetryOptions,
  ToolSet,
} from 'ai';
import {
  formatInputMessages,
  formatModelMessages,
  formatObjectOutputMessages,
  formatOutputMessages,
  formatSystemInstructions,
  mapOperationName,
  mapProviderName,
} from './gen-ai-format-messages';
import { assembleOperationName } from './assemble-operation-name';
import { getBaseTelemetryAttributes } from './get-base-telemetry-attributes';
import { stringifyForTelemetry } from './stringify-for-telemetry';

type LegacyAttributeGroup =
  | 'operation'
  | 'settings'
  | 'runtimeContext'
  | 'requestHeaders'
  | 'prompts'
  | 'schemas'
  | 'responses'
  | 'usage'
  | 'toolCalls'
  | 'streaming'
  | 'embeddings'
  | 'reranking';

export type OpenTelemetryLegacyAttributes =
  | true
  | Partial<Record<LegacyAttributeGroup, boolean>>;

type NormalizedLegacyAttributes = Record<LegacyAttributeGroup, boolean>;

const disabledLegacyAttributes: NormalizedLegacyAttributes = {
  operation: false,
  settings: false,
  runtimeContext: false,
  requestHeaders: false,
  prompts: false,
  schemas: false,
  responses: false,
  usage: false,
  toolCalls: false,
  streaming: false,
  embeddings: false,
  reranking: false,
};

function normalizeLegacyAttributes(
  legacyAttributes: OpenTelemetryLegacyAttributes | undefined,
): NormalizedLegacyAttributes | undefined {
  if (legacyAttributes == null) {
    return undefined;
  }

  if (legacyAttributes === true) {
    return Object.fromEntries(
      Object.keys(disabledLegacyAttributes).map(key => [key, true]),
    ) as NormalizedLegacyAttributes;
  }

  return {
    ...disabledLegacyAttributes,
    ...legacyAttributes,
  };
}

function hasEnabledLegacyAttributes(
  legacyAttributes: NormalizedLegacyAttributes,
): boolean {
  return Object.values(legacyAttributes).some(Boolean);
}

function isLegacyAttributeEnabled(
  key: string,
  legacyAttributes: NormalizedLegacyAttributes,
): boolean {
  if (
    key === 'operation.name' ||
    key === 'resource.name' ||
    key === 'ai.operationId' ||
    key === 'ai.telemetry.functionId' ||
    key === 'ai.model.provider' ||
    key === 'ai.model.id' ||
    key === 'gen_ai.system' ||
    key === 'gen_ai.request.model'
  ) {
    return legacyAttributes.operation;
  }

  if (key.startsWith('ai.settings.context.')) {
    return legacyAttributes.runtimeContext;
  }

  if (
    key === 'ai.schema' ||
    key.startsWith('ai.schema.') ||
    key === 'ai.settings.output'
  ) {
    return legacyAttributes.schemas;
  }

  if (key.startsWith('ai.settings.') || key.startsWith('gen_ai.request.')) {
    return legacyAttributes.settings;
  }

  if (key.startsWith('ai.request.headers.')) {
    return legacyAttributes.requestHeaders;
  }

  if (
    key === 'ai.prompt' ||
    key.startsWith('ai.prompt.') ||
    key === 'ai.value' ||
    key === 'ai.values'
  ) {
    return legacyAttributes.prompts || legacyAttributes.embeddings;
  }

  if (key === 'ai.documents') {
    return legacyAttributes.prompts || legacyAttributes.reranking;
  }

  if (key.startsWith('ai.response.') || key.startsWith('gen_ai.response.')) {
    return legacyAttributes.responses;
  }

  if (key.startsWith('ai.usage.') || key.startsWith('gen_ai.usage.')) {
    return legacyAttributes.usage;
  }

  if (key.startsWith('ai.toolCall.')) {
    return legacyAttributes.toolCalls;
  }

  if (key.startsWith('ai.stream.')) {
    return legacyAttributes.streaming;
  }

  if (key === 'ai.embedding' || key === 'ai.embeddings') {
    return legacyAttributes.embeddings;
  }

  if (key.startsWith('ai.ranking')) {
    return legacyAttributes.reranking;
  }

  return false;
}

function selectLegacyAttributes(
  telemetry: TelemetryOptions | undefined,
  legacyAttributes: NormalizedLegacyAttributes | undefined,
  attributes: Record<
    string,
    | AttributeValue
    | { input: () => AttributeValue | undefined }
    | { output: () => AttributeValue | undefined }
    | undefined
  >,
): Attributes {
  if (legacyAttributes == null) {
    return {};
  }

  return (
    filterLegacyAttributes(
      selectAttributes(telemetry, attributes),
      legacyAttributes,
    ) ?? {}
  );
}

function filterLegacyAttributes(
  attributes: Attributes | undefined,
  legacyAttributes: NormalizedLegacyAttributes,
): Attributes | undefined {
  if (attributes == null) {
    return attributes;
  }

  return Object.fromEntries(
    Object.entries(attributes).filter(([key]) =>
      isLegacyAttributeEnabled(key, legacyAttributes),
    ),
  ) as Attributes;
}

function recordSpanError(span: Span, error: unknown): void {
  if (error instanceof Error) {
    span.recordException({
      name: error.name,
      message: error.message,
      stack: error.stack,
    });
    span.setStatus({
      code: SpanStatusCode.ERROR,
      message: error.message,
    });
  } else {
    span.setStatus({ code: SpanStatusCode.ERROR });
  }
}

function shouldRecord(
  telemetry: TelemetryOptions | undefined,
): telemetry is TelemetryOptions {
  return telemetry?.isEnabled !== false;
}

function selectAttributes(
  telemetry: TelemetryOptions | undefined,
  attributes: Record<
    string,
    | AttributeValue
    | { input: () => AttributeValue | undefined }
    | { output: () => AttributeValue | undefined }
    | undefined
  >,
): Attributes {
  if (!shouldRecord(telemetry)) {
    return {};
  }

  const result: Attributes = {};

  for (const [key, value] of Object.entries(attributes)) {
    if (value == null) continue;

    if (
      typeof value === 'object' &&
      'input' in value &&
      typeof value.input === 'function'
    ) {
      if (telemetry?.recordInputs === false) continue;
      const resolved = value.input();
      if (resolved != null) result[key] = resolved;
      continue;
    }

    if (
      typeof value === 'object' &&
      'output' in value &&
      typeof value.output === 'function'
    ) {
      if (telemetry?.recordOutputs === false) continue;
      const resolved = value.output();
      if (resolved != null) result[key] = resolved;
      continue;
    }

    result[key] = value as AttributeValue;
  }

  return result;
}

interface OtelStepStartEvent extends GenerateTextStepStartEvent<ToolSet> {
  readonly promptMessages?: LanguageModelV4Prompt;
  readonly stepTools?: ReadonlyArray<Record<string, unknown>>;
  readonly stepToolChoice?: unknown;
}

interface CallState {
  operationId: string;
  telemetry: TelemetryOptions | undefined;
  rootSpan: Span | undefined;
  rootContext: OpenTelemetryContext | undefined;
  stepSpan: Span | undefined;
  stepContext: OpenTelemetryContext | undefined;
  inferenceSpan: Span | undefined;
  inferenceContext: OpenTelemetryContext | undefined;
  embedSpans: Map<string, { span: Span; context: OpenTelemetryContext }>;
  rerankSpan: { span: Span; context: OpenTelemetryContext } | undefined;
  toolSpans: Map<string, { span: Span; context: OpenTelemetryContext }>;
  settings: Record<string, unknown>;
  provider: string;
  modelId: string;
  baseTelemetryAttributes: Attributes;
}

export class OpenTelemetry implements Telemetry {
  private readonly callStates = new Map<string, CallState>();

  private readonly tracer: Tracer;
  private readonly legacyAttributes: NormalizedLegacyAttributes | undefined;

  constructor(
    options: {
      tracer?: Tracer;
      legacyAttributes?: OpenTelemetryLegacyAttributes;
    } = {},
  ) {
    this.tracer = options.tracer ?? trace.getTracer('gen_ai');
    const legacyAttributes = normalizeLegacyAttributes(
      options.legacyAttributes,
    );
    this.legacyAttributes =
      legacyAttributes != null && hasEnabledLegacyAttributes(legacyAttributes)
        ? legacyAttributes
        : undefined;
  }

  private getCallState(callId: string): CallState | undefined {
    return this.callStates.get(callId);
  }

  private cleanupCallState(callId: string): void {
    this.callStates.delete(callId);
  }

  executeTool<T>({
    callId,
    toolCallId,
    execute,
  }: {
    callId: string;
    toolCallId: string;
    execute: () => PromiseLike<T>;
  }): PromiseLike<T> {
    const toolSpanEntry = this.getCallState(callId)?.toolSpans.get(toolCallId);

    if (toolSpanEntry == null) {
      return execute();
    }

    return context.with(toolSpanEntry.context, execute);
  }

  onStart(
    event:
      | InferTelemetryEvent<GenerateTextStartEvent>
      | InferTelemetryEvent<GenerateObjectStartEvent>
      | InferTelemetryEvent<EmbedStartEvent>
      | InferTelemetryEvent<RerankStartEvent>,
  ): void {
    if (
      event.operationId === 'ai.embed' ||
      event.operationId === 'ai.embedMany'
    ) {
      this.onEmbedOperationStart(event as InferTelemetryEvent<EmbedStartEvent>);
      return;
    }

    if (event.operationId === 'ai.rerank') {
      this.onRerankOperationStart(
        event as InferTelemetryEvent<RerankStartEvent>,
      );
      return;
    }

    if (
      event.operationId === 'ai.generateObject' ||
      event.operationId === 'ai.streamObject'
    ) {
      this.onObjectOperationStart(
        event as InferTelemetryEvent<GenerateObjectStartEvent>,
      );
      return;
    }

    this.onGenerateStart(event as InferTelemetryEvent<GenerateTextStartEvent>);
  }

  private onGenerateStart(
    event: InferTelemetryEvent<GenerateTextStartEvent>,
  ): void {
    const telemetry: TelemetryOptions = {
      recordInputs: event.recordInputs,
      recordOutputs: event.recordOutputs,
      functionId: event.functionId,
    };

    const settings: Record<string, unknown> = {
      maxOutputTokens: event.maxOutputTokens,
      temperature: event.temperature,
      topP: event.topP,
      topK: event.topK,
      presencePenalty: event.presencePenalty,
      frequencyPenalty: event.frequencyPenalty,
      stopSequences: event.stopSequences,
      seed: event.seed,
      maxRetries: event.maxRetries,
    };

    const providerName = mapProviderName(event.provider);
    const operationName = mapOperationName(event.operationId);
    const baseTelemetryAttributes = getBaseTelemetryAttributes({
      model: { provider: event.provider, modelId: event.modelId },
      headers: event.headers,
      settings,
      context: event.runtimeContext as Record<string, unknown> | undefined,
    });

    const attributes = selectAttributes(telemetry, {
      'gen_ai.operation.name': operationName,
      'gen_ai.provider.name': providerName,
      'gen_ai.request.model': event.modelId,
      'gen_ai.agent.name': telemetry.functionId,
      'gen_ai.request.frequency_penalty': event.frequencyPenalty,
      'gen_ai.request.max_tokens': event.maxOutputTokens,
      'gen_ai.request.presence_penalty': event.presencePenalty,
      'gen_ai.request.temperature': (event.temperature ?? undefined) as
        | number
        | undefined,
      'gen_ai.request.top_k': event.topK,
      'gen_ai.request.top_p': event.topP,
      'gen_ai.request.stop_sequences': event.stopSequences,
      'gen_ai.request.seed': event.seed,
      'gen_ai.system_instructions': event.system
        ? {
            input: () =>
              JSON.stringify(formatSystemInstructions(event.system!)),
          }
        : undefined,
      'gen_ai.input.messages': {
        input: () =>
          JSON.stringify(
            formatModelMessages({
              prompt: undefined,
              messages: event.messages,
            }),
          ),
      },
      ...selectLegacyAttributes(telemetry, this.legacyAttributes, {
        ...assembleOperationName({
          operationId: event.operationId,
          telemetry,
        }),
        ...baseTelemetryAttributes,
        'ai.prompt': {
          input: () =>
            JSON.stringify({
              system: event.system,
              messages: event.messages,
            }),
        },
      }),
    });

    const spanName = `${operationName} ${event.modelId}`;
    const rootSpan = this.tracer.startSpan(spanName, {
      attributes,
      kind: SpanKind.INTERNAL,
    });
    const rootContext = trace.setSpan(context.active(), rootSpan);

    this.callStates.set(event.callId, {
      operationId: event.operationId,
      telemetry,
      rootSpan,
      rootContext,
      stepSpan: undefined,
      stepContext: undefined,
      inferenceSpan: undefined,
      inferenceContext: undefined,
      embedSpans: new Map(),
      rerankSpan: undefined,
      toolSpans: new Map(),
      settings,
      provider: event.provider,
      modelId: event.modelId,
      baseTelemetryAttributes,
    });
  }

  private onObjectOperationStart(
    event: InferTelemetryEvent<GenerateObjectStartEvent>,
  ): void {
    const telemetry: TelemetryOptions = {
      recordInputs: event.recordInputs,
      recordOutputs: event.recordOutputs,
      functionId: event.functionId,
    };

    const settings: Record<string, unknown> = {
      maxOutputTokens: event.maxOutputTokens,
      temperature: event.temperature,
      topP: event.topP,
      topK: event.topK,
      presencePenalty: event.presencePenalty,
      frequencyPenalty: event.frequencyPenalty,
      seed: event.seed,
      maxRetries: event.maxRetries,
    };

    const providerName = mapProviderName(event.provider);
    const operationName = mapOperationName(event.operationId);
    const baseTelemetryAttributes = getBaseTelemetryAttributes({
      model: { provider: event.provider, modelId: event.modelId },
      headers: event.headers,
      settings,
      context: undefined,
    });

    const attributes = selectAttributes(telemetry, {
      'gen_ai.operation.name': operationName,
      'gen_ai.provider.name': providerName,
      'gen_ai.request.model': event.modelId,
      'gen_ai.agent.name': telemetry.functionId,
      'gen_ai.output.type': 'json',
      'gen_ai.request.frequency_penalty': event.frequencyPenalty,
      'gen_ai.request.max_tokens': event.maxOutputTokens,
      'gen_ai.request.presence_penalty': event.presencePenalty,
      'gen_ai.request.temperature': (event.temperature ?? undefined) as
        | number
        | undefined,
      'gen_ai.request.top_k': event.topK,
      'gen_ai.request.top_p': event.topP,
      'gen_ai.request.seed': event.seed,
      'gen_ai.system_instructions': event.system
        ? {
            input: () =>
              JSON.stringify(formatSystemInstructions(event.system!)),
          }
        : undefined,
      'gen_ai.input.messages': {
        input: () =>
          JSON.stringify(
            formatModelMessages({
              prompt: event.prompt,
              messages: event.messages,
            }),
          ),
      },
      ...selectLegacyAttributes(telemetry, this.legacyAttributes, {
        ...assembleOperationName({
          operationId: event.operationId,
          telemetry,
        }),
        ...baseTelemetryAttributes,
        'ai.prompt': {
          input: () =>
            JSON.stringify({
              system: event.system,
              prompt: event.prompt,
              messages: event.messages,
            }),
        },
        'ai.schema': event.schema
          ? { input: () => JSON.stringify(event.schema) }
          : undefined,
        'ai.schema.name': event.schemaName,
        'ai.schema.description': event.schemaDescription,
        'ai.settings.output': event.output,
      }),
    });

    const spanName = `${operationName} ${event.modelId}`;
    const rootSpan = this.tracer.startSpan(spanName, {
      attributes,
      kind: SpanKind.INTERNAL,
    });
    const rootContext = trace.setSpan(context.active(), rootSpan);

    this.callStates.set(event.callId, {
      operationId: event.operationId,
      telemetry,
      rootSpan,
      rootContext,
      stepSpan: undefined,
      stepContext: undefined,
      inferenceSpan: undefined,
      inferenceContext: undefined,
      embedSpans: new Map(),
      rerankSpan: undefined,
      toolSpans: new Map(),
      settings,
      provider: event.provider,
      modelId: event.modelId,
      baseTelemetryAttributes,
    });
  }

  /** @deprecated */
  onObjectStepStart(event: GenerateObjectStepStartEvent): void {
    const state = this.getCallState(event.callId);
    if (!state?.rootSpan || !state.rootContext) return;

    const { telemetry } = state;
    const providerName = mapProviderName(event.provider);

    const attributes = selectAttributes(telemetry, {
      'gen_ai.operation.name': 'chat',
      'gen_ai.provider.name': providerName,
      'gen_ai.request.model': event.modelId,
      'gen_ai.output.type': 'json',
      'gen_ai.request.frequency_penalty': state.settings.frequencyPenalty as
        | number
        | undefined,
      'gen_ai.request.max_tokens': state.settings.maxOutputTokens as
        | number
        | undefined,
      'gen_ai.request.presence_penalty': state.settings.presencePenalty as
        | number
        | undefined,
      'gen_ai.request.temperature': (state.settings.temperature ?? undefined) as
        | number
        | undefined,
      'gen_ai.request.top_k': state.settings.topK as number | undefined,
      'gen_ai.request.top_p': state.settings.topP as number | undefined,
      'gen_ai.input.messages': {
        input: () =>
          event.promptMessages
            ? JSON.stringify(formatInputMessages(event.promptMessages))
            : undefined,
      },
      ...selectLegacyAttributes(telemetry, this.legacyAttributes, {
        ...assembleOperationName({
          operationId:
            state.operationId === 'ai.streamObject'
              ? 'ai.streamObject.doStream'
              : 'ai.generateObject.doGenerate',
          telemetry,
        }),
        ...state.baseTelemetryAttributes,
        'ai.prompt.messages': {
          input: () =>
            event.promptMessages
              ? stringifyForTelemetry(event.promptMessages)
              : undefined,
        },
      }),
    });

    const spanName = `chat ${event.modelId}`;
    state.inferenceSpan = this.tracer.startSpan(
      spanName,
      { attributes, kind: SpanKind.CLIENT },
      state.rootContext,
    );
    state.inferenceContext = trace.setSpan(
      state.rootContext,
      state.inferenceSpan,
    );
  }

  /** @deprecated */
  onObjectStepFinish(event: GenerateObjectStepEndEvent): void {
    const state = this.getCallState(event.callId);
    if (!state?.inferenceSpan) return;

    const { telemetry } = state;

    state.inferenceSpan.setAttributes(
      selectAttributes(telemetry, {
        'gen_ai.response.finish_reasons': [event.finishReason],
        'gen_ai.response.id': event.response.id,
        'gen_ai.response.model': event.response.modelId,
        'gen_ai.usage.input_tokens': event.usage.inputTokens,
        'gen_ai.usage.output_tokens': event.usage.outputTokens,
        'gen_ai.usage.cache_read.input_tokens': event.usage.cachedInputTokens,
        'gen_ai.output.messages': {
          output: () => {
            try {
              return JSON.stringify(
                formatObjectOutputMessages({
                  objectText: event.objectText,
                  finishReason: event.finishReason,
                }),
              );
            } catch {
              return event.objectText;
            }
          },
        },
        ...selectLegacyAttributes(telemetry, this.legacyAttributes, {
          'ai.response.finishReason': event.finishReason,
          'ai.response.object': {
            output: () => {
              try {
                return JSON.stringify(JSON.parse(event.objectText));
              } catch {
                return event.objectText;
              }
            },
          },
          'ai.response.id': event.response.id,
          'ai.response.model': event.response.modelId,
          'ai.response.timestamp': event.response.timestamp.toISOString(),
          'ai.response.providerMetadata': event.providerMetadata
            ? JSON.stringify(event.providerMetadata)
            : undefined,
          'ai.usage.inputTokens': event.usage.inputTokens,
          'ai.usage.outputTokens': event.usage.outputTokens,
          'ai.usage.totalTokens': event.usage.totalTokens,
          'ai.usage.reasoningTokens': event.usage.reasoningTokens,
          'ai.usage.cachedInputTokens': event.usage.cachedInputTokens,
        }),
      }),
    );

    state.inferenceSpan.end();
    state.inferenceSpan = undefined;
    state.inferenceContext = undefined;
  }

  private onEmbedOperationStart(
    event: InferTelemetryEvent<EmbedStartEvent>,
  ): void {
    const telemetry: TelemetryOptions = {
      recordInputs: event.recordInputs,
      recordOutputs: event.recordOutputs,
      functionId: event.functionId,
    };

    const settings: Record<string, unknown> = {
      maxRetries: event.maxRetries,
    };

    const providerName = mapProviderName(event.provider);
    const baseTelemetryAttributes = getBaseTelemetryAttributes({
      model: { provider: event.provider, modelId: event.modelId },
      headers: event.headers,
      settings,
      context: undefined,
    });
    const value = event.value;
    const isMany = event.operationId === 'ai.embedMany';

    const attributes = selectAttributes(telemetry, {
      'gen_ai.operation.name': 'embeddings',
      'gen_ai.provider.name': providerName,
      'gen_ai.request.model': event.modelId,
      ...selectLegacyAttributes(telemetry, this.legacyAttributes, {
        ...assembleOperationName({
          operationId: event.operationId,
          telemetry,
        }),
        ...baseTelemetryAttributes,
        ...(isMany
          ? {
              'ai.values': {
                input: () => (value as string[]).map(v => JSON.stringify(v)),
              },
            }
          : {
              'ai.value': {
                input: () => JSON.stringify(value),
              },
            }),
      }),
    });

    const spanName = `embeddings ${event.modelId}`;
    const rootSpan = this.tracer.startSpan(spanName, {
      attributes,
      kind: SpanKind.CLIENT,
    });
    const rootContext = trace.setSpan(context.active(), rootSpan);

    this.callStates.set(event.callId, {
      operationId: event.operationId,
      telemetry,
      rootSpan,
      rootContext,
      stepSpan: undefined,
      stepContext: undefined,
      inferenceSpan: undefined,
      inferenceContext: undefined,
      embedSpans: new Map(),
      rerankSpan: undefined,
      toolSpans: new Map(),
      settings,
      provider: event.provider,
      modelId: event.modelId,
      baseTelemetryAttributes,
    });
  }

  onStepStart(event: OtelStepStartEvent): void {
    const state = this.getCallState(event.callId);
    if (!state?.rootSpan || !state.rootContext) return;

    const { telemetry } = state;
    const stepAttributes = selectAttributes(telemetry, {
      'gen_ai.operation.name': 'agent_step',
      ...selectLegacyAttributes(telemetry, this.legacyAttributes, {
        ...assembleOperationName({
          operationId:
            state.operationId === 'ai.streamText'
              ? 'ai.streamText.doStream'
              : 'ai.generateText.doGenerate',
          telemetry,
        }),
        ...state.baseTelemetryAttributes,
        'ai.model.provider': event.provider,
        'ai.model.id': event.modelId,
        'ai.prompt.messages': {
          input: () =>
            event.promptMessages
              ? stringifyForTelemetry(event.promptMessages)
              : undefined,
        },
        'ai.prompt.tools': {
          input: () => event.stepTools?.map(tool => JSON.stringify(tool)),
        },
        'ai.prompt.toolChoice': {
          input: () =>
            event.stepToolChoice != null
              ? JSON.stringify(event.stepToolChoice)
              : undefined,
        },
        'gen_ai.system': event.provider,
        'gen_ai.request.model': event.modelId,
        'gen_ai.request.frequency_penalty': state.settings.frequencyPenalty as
          | number
          | undefined,
        'gen_ai.request.max_tokens': state.settings.maxOutputTokens as
          | number
          | undefined,
        'gen_ai.request.presence_penalty': state.settings.presencePenalty as
          | number
          | undefined,
        'gen_ai.request.stop_sequences': state.settings.stopSequences as
          | string[]
          | undefined,
        'gen_ai.request.temperature': (state.settings.temperature ??
          undefined) as number | undefined,
        'gen_ai.request.top_k': state.settings.topK as number | undefined,
        'gen_ai.request.top_p': state.settings.topP as number | undefined,
      }),
    });

    state.stepSpan = this.tracer.startSpan(
      `step ${event.steps.length + 1}`,
      { attributes: stepAttributes, kind: SpanKind.INTERNAL },
      state.rootContext,
    );
    state.stepContext = trace.setSpan(state.rootContext, state.stepSpan);
  }

  onLanguageModelCallStart(event: LanguageModelCallStartEvent): void {
    const state = this.getCallState(event.callId);
    if (!state?.stepContext) return;

    const { telemetry } = state;
    const providerName = mapProviderName(event.provider);

    const inferenceAttributes = selectAttributes(telemetry, {
      'gen_ai.operation.name': 'chat',
      'gen_ai.provider.name': providerName,
      'gen_ai.request.model': event.modelId,
      'gen_ai.request.frequency_penalty': state.settings.frequencyPenalty as
        | number
        | undefined,
      'gen_ai.request.max_tokens': state.settings.maxOutputTokens as
        | number
        | undefined,
      'gen_ai.request.presence_penalty': state.settings.presencePenalty as
        | number
        | undefined,
      'gen_ai.request.stop_sequences': state.settings.stopSequences as
        | string[]
        | undefined,
      'gen_ai.request.temperature': (state.settings.temperature ?? undefined) as
        | number
        | undefined,
      'gen_ai.request.top_k': state.settings.topK as number | undefined,
      'gen_ai.request.top_p': state.settings.topP as number | undefined,
      'gen_ai.input.messages': {
        input: () => {
          const formattedMessages = formatModelMessages({
            prompt: undefined,
            messages: event.messages,
          });

          return formattedMessages.length > 0
            ? JSON.stringify(formattedMessages)
            : undefined;
        },
      },
      'gen_ai.tool.definitions': {
        input: () => (event.tools ? JSON.stringify(event.tools) : undefined),
      },
      ...selectLegacyAttributes(telemetry, this.legacyAttributes, {
        'ai.prompt.messages': {
          input: () => JSON.stringify(event.messages),
        },
        'ai.prompt.tools': {
          input: () => event.tools?.map(tool => JSON.stringify(tool)),
        },
      }),
    });

    state.inferenceSpan = this.tracer.startSpan(
      `chat ${event.modelId}`,
      { attributes: inferenceAttributes, kind: SpanKind.CLIENT },
      state.stepContext,
    );
    state.inferenceContext = trace.setSpan(
      state.stepContext,
      state.inferenceSpan,
    );
  }

  onLanguageModelCallEnd(event: LanguageModelCallEndEvent<ToolSet>): void {
    const state = this.getCallState(event.callId);
    if (!state?.inferenceSpan) return;

    const { telemetry } = state;

    state.inferenceSpan.setAttributes(
      selectAttributes(telemetry, {
        'gen_ai.response.finish_reasons': [event.finishReason],
        'gen_ai.response.id': event.responseId,
        'gen_ai.usage.input_tokens': event.usage.inputTokens,
        'gen_ai.usage.output_tokens': event.usage.outputTokens,
        'gen_ai.usage.cache_read.input_tokens':
          event.usage.inputTokenDetails?.cacheReadTokens ??
          event.usage.cachedInputTokens,
        'gen_ai.usage.cache_creation.input_tokens':
          event.usage.inputTokenDetails?.cacheWriteTokens,
        'gen_ai.output.messages': {
          output: () =>
            JSON.stringify(
              formatOutputMessages({
                text:
                  event.content
                    .filter(p => p.type === 'text')
                    .map(p => p.text)
                    .join('') || undefined,
                reasoning: event.content.filter(p => p.type === 'reasoning'),
                toolCalls: event.content.filter(p => p.type === 'tool-call'),
                files: event.content
                  .filter(p => p.type === 'file')
                  .map(p => p.file),
                finishReason: event.finishReason,
              }),
            ),
        },
        ...selectLegacyAttributes(telemetry, this.legacyAttributes, {
          'ai.response.finishReason': event.finishReason,
          'ai.response.text': {
            output: () =>
              event.content
                .filter(part => part.type === 'text')
                .map(part => part.text)
                .join('') || undefined,
          },
          'ai.response.reasoning': {
            output: () => {
              const reasoning = event.content.filter(
                part => part.type === 'reasoning',
              );
              return reasoning.length > 0
                ? reasoning
                    .filter(part => 'text' in part)
                    .map(part => part.text)
                    .join('\n')
                : undefined;
            },
          },
          'ai.response.toolCalls': {
            output: () => {
              const toolCalls = event.content.filter(
                part => part.type === 'tool-call',
              );
              return toolCalls.length > 0
                ? JSON.stringify(
                    toolCalls.map(toolCall => ({
                      toolCallId: toolCall.toolCallId,
                      toolName: toolCall.toolName,
                      input: toolCall.input,
                    })),
                  )
                : undefined;
            },
          },
          'ai.response.files': {
            output: () => {
              const files = event.content.filter(part => part.type === 'file');
              return files.length > 0
                ? JSON.stringify(
                    files.map(file => ({
                      type: 'file',
                      mediaType: file.file.mediaType,
                      data: file.file.base64,
                    })),
                  )
                : undefined;
            },
          },
          'ai.response.id': event.responseId,
          'ai.usage.inputTokens': event.usage.inputTokens,
          'ai.usage.outputTokens': event.usage.outputTokens,
          'ai.usage.totalTokens': event.usage.totalTokens,
          'ai.usage.reasoningTokens': event.usage.reasoningTokens,
          'ai.usage.cachedInputTokens': event.usage.cachedInputTokens,
          'ai.usage.inputTokenDetails.noCacheTokens':
            event.usage.inputTokenDetails?.noCacheTokens,
          'ai.usage.inputTokenDetails.cacheReadTokens':
            event.usage.inputTokenDetails?.cacheReadTokens,
          'ai.usage.inputTokenDetails.cacheWriteTokens':
            event.usage.inputTokenDetails?.cacheWriteTokens,
          'ai.usage.outputTokenDetails.textTokens':
            event.usage.outputTokenDetails?.textTokens,
          'ai.usage.outputTokenDetails.reasoningTokens':
            event.usage.outputTokenDetails?.reasoningTokens,
        }),
      }),
    );

    state.inferenceSpan.end();
    state.inferenceSpan = undefined;
    state.inferenceContext = undefined;
  }

  onToolExecutionStart(event: ToolExecutionStartEvent<ToolSet>): void {
    const state = this.getCallState(event.callId);
    if (!state?.stepContext) return;

    const { telemetry } = state;
    const { toolCall } = event;

    const attributes = selectAttributes(telemetry, {
      'gen_ai.operation.name': 'execute_tool',
      'gen_ai.tool.name': toolCall.toolName,
      'gen_ai.tool.call.id': toolCall.toolCallId,
      'gen_ai.tool.type': 'function',
      'gen_ai.tool.call.arguments': {
        input: () => JSON.stringify(toolCall.input),
      },
      ...selectLegacyAttributes(telemetry, this.legacyAttributes, {
        ...assembleOperationName({
          operationId: 'ai.toolCall',
          telemetry,
        }),
        'ai.toolCall.name': toolCall.toolName,
        'ai.toolCall.id': toolCall.toolCallId,
        'ai.toolCall.args': {
          output: () => JSON.stringify(toolCall.input),
        },
      }),
    });

    const spanName = `execute_tool ${toolCall.toolName}`;
    const toolSpan = this.tracer.startSpan(
      spanName,
      { attributes, kind: SpanKind.INTERNAL },
      state.stepContext,
    );
    const toolContext = trace.setSpan(state.stepContext, toolSpan);

    state.toolSpans.set(toolCall.toolCallId, {
      span: toolSpan,
      context: toolContext,
    });
  }

  onToolExecutionEnd(event: ToolExecutionEndEvent<ToolSet>): void {
    const state = this.getCallState(event.callId);
    if (!state) return;

    const toolSpanEntry = state.toolSpans.get(event.toolCall.toolCallId);
    if (!toolSpanEntry) return;

    const { span } = toolSpanEntry;
    const { telemetry } = state;

    const { toolOutput } = event;
    if (toolOutput.type === 'tool-result') {
      try {
        span.setAttributes(
          selectAttributes(telemetry, {
            'gen_ai.tool.call.result': {
              output: () => JSON.stringify(toolOutput.output),
            },
            ...selectLegacyAttributes(telemetry, this.legacyAttributes, {
              'ai.toolCall.result': {
                output: () => JSON.stringify(toolOutput.output),
              },
            }),
          }),
        );
      } catch {
        // JSON.stringify might fail for non-serializable results
      }
    } else {
      recordSpanError(span, toolOutput.error);
    }

    span.end();
    state.toolSpans.delete(event.toolCall.toolCallId);
  }

  onStepFinish(event: GenerateTextStepEndEvent<ToolSet>): void {
    const state = this.getCallState(event.callId);
    if (!state?.stepSpan) return;

    const { telemetry } = state;

    state.stepSpan.setAttributes(
      selectLegacyAttributes(telemetry, this.legacyAttributes, {
        'ai.response.finishReason': event.finishReason,
        'ai.response.text': {
          output: () => event.text ?? undefined,
        },
        'ai.response.reasoning': {
          output: () =>
            event.reasoning.length > 0
              ? event.reasoning
                  .filter(part => 'text' in part)
                  .map(part => part.text)
                  .join('\n')
              : undefined,
        },
        'ai.response.toolCalls': {
          output: () =>
            event.toolCalls.length > 0
              ? JSON.stringify(
                  event.toolCalls.map(toolCall => ({
                    toolCallId: toolCall.toolCallId,
                    toolName: toolCall.toolName,
                    input: toolCall.input,
                  })),
                )
              : undefined,
        },
        'ai.response.files': {
          output: () =>
            event.files.length > 0
              ? JSON.stringify(
                  event.files.map(file => ({
                    type: 'file',
                    mediaType: file.mediaType,
                    data: file.base64,
                  })),
                )
              : undefined,
        },
        'ai.response.id': event.response.id,
        'ai.response.model': event.response.modelId,
        'ai.response.timestamp': event.response.timestamp.toISOString(),
        'ai.response.providerMetadata': event.providerMetadata
          ? JSON.stringify(event.providerMetadata)
          : undefined,
        'ai.usage.inputTokens': event.usage.inputTokens,
        'ai.usage.outputTokens': event.usage.outputTokens,
        'ai.usage.totalTokens': event.usage.totalTokens,
        'ai.usage.reasoningTokens': event.usage.reasoningTokens,
        'ai.usage.cachedInputTokens': event.usage.cachedInputTokens,
        'ai.usage.inputTokenDetails.noCacheTokens':
          event.usage.inputTokenDetails?.noCacheTokens,
        'ai.usage.inputTokenDetails.cacheReadTokens':
          event.usage.inputTokenDetails?.cacheReadTokens,
        'ai.usage.inputTokenDetails.cacheWriteTokens':
          event.usage.inputTokenDetails?.cacheWriteTokens,
        'ai.usage.outputTokenDetails.textTokens':
          event.usage.outputTokenDetails?.textTokens,
        'ai.usage.outputTokenDetails.reasoningTokens':
          event.usage.outputTokenDetails?.reasoningTokens,
      }),
    );

    state.stepSpan.end();
    state.stepSpan = undefined;
    state.stepContext = undefined;
  }

  onFinish(
    event:
      | GenerateTextEndEvent<ToolSet>
      | GenerateObjectEndEvent<unknown>
      | EmbedEndEvent
      | RerankEndEvent,
  ): void {
    const state = this.getCallState(event.callId);
    if (!state?.rootSpan) return;

    if (
      state.operationId === 'ai.embed' ||
      state.operationId === 'ai.embedMany'
    ) {
      this.onEmbedOperationFinish(event as EmbedEndEvent);
      return;
    }

    if (state.operationId === 'ai.rerank') {
      this.onRerankOperationFinish(event as RerankEndEvent);
      return;
    }

    if (
      state.operationId === 'ai.generateObject' ||
      state.operationId === 'ai.streamObject'
    ) {
      this.onObjectOperationFinish(event as GenerateObjectEndEvent<unknown>);
      return;
    }

    this.onGenerateFinish(event as GenerateTextEndEvent<ToolSet>);
  }

  private onGenerateFinish(event: GenerateTextEndEvent<ToolSet>): void {
    const state = this.getCallState(event.callId);
    if (!state?.rootSpan) return;

    const { telemetry } = state;

    state.rootSpan.setAttributes(
      selectAttributes(telemetry, {
        'gen_ai.response.finish_reasons': [event.finishReason],
        'gen_ai.usage.input_tokens': event.totalUsage.inputTokens,
        'gen_ai.usage.output_tokens': event.totalUsage.outputTokens,
        'gen_ai.usage.cache_read.input_tokens':
          event.totalUsage.inputTokenDetails?.cacheReadTokens ??
          event.totalUsage.cachedInputTokens,
        'gen_ai.usage.cache_creation.input_tokens':
          event.totalUsage.inputTokenDetails?.cacheWriteTokens,
        'gen_ai.output.messages': {
          output: () =>
            JSON.stringify(
              formatOutputMessages({
                text: event.text ?? undefined,
                reasoning: event.reasoning as ReadonlyArray<{ text?: string }>,
                toolCalls: event.toolCalls,
                files: event.files,
                finishReason: event.finishReason,
              }),
            ),
        },
        ...selectLegacyAttributes(telemetry, this.legacyAttributes, {
          'ai.response.finishReason': event.finishReason,
          'ai.response.text': {
            output: () => event.text ?? undefined,
          },
          'ai.response.reasoning': {
            output: () =>
              event.reasoning.length > 0
                ? event.reasoning
                    .filter(part => 'text' in part)
                    .map(part => part.text)
                    .join('\n')
                : undefined,
          },
          'ai.response.toolCalls': {
            output: () =>
              event.toolCalls.length > 0
                ? JSON.stringify(
                    event.toolCalls.map(toolCall => ({
                      toolCallId: toolCall.toolCallId,
                      toolName: toolCall.toolName,
                      input: toolCall.input,
                    })),
                  )
                : undefined,
          },
          'ai.response.files': {
            output: () =>
              event.files.length > 0
                ? JSON.stringify(
                    event.files.map(file => ({
                      type: 'file',
                      mediaType: file.mediaType,
                      data: file.base64,
                    })),
                  )
                : undefined,
          },
          'ai.response.providerMetadata': event.providerMetadata
            ? JSON.stringify(event.providerMetadata)
            : undefined,
          'ai.usage.inputTokens': event.totalUsage.inputTokens,
          'ai.usage.outputTokens': event.totalUsage.outputTokens,
          'ai.usage.totalTokens': event.totalUsage.totalTokens,
          'ai.usage.reasoningTokens': event.totalUsage.reasoningTokens,
          'ai.usage.cachedInputTokens': event.totalUsage.cachedInputTokens,
          'ai.usage.inputTokenDetails.noCacheTokens':
            event.totalUsage.inputTokenDetails?.noCacheTokens,
          'ai.usage.inputTokenDetails.cacheReadTokens':
            event.totalUsage.inputTokenDetails?.cacheReadTokens,
          'ai.usage.inputTokenDetails.cacheWriteTokens':
            event.totalUsage.inputTokenDetails?.cacheWriteTokens,
          'ai.usage.outputTokenDetails.textTokens':
            event.totalUsage.outputTokenDetails?.textTokens,
          'ai.usage.outputTokenDetails.reasoningTokens':
            event.totalUsage.outputTokenDetails?.reasoningTokens,
        }),
      }),
    );

    state.rootSpan.end();
    this.cleanupCallState(event.callId);
  }

  private onObjectOperationFinish(
    event: GenerateObjectEndEvent<unknown>,
  ): void {
    const state = this.getCallState(event.callId);
    if (!state?.rootSpan) return;

    const { telemetry } = state;

    state.rootSpan.setAttributes(
      selectAttributes(telemetry, {
        'gen_ai.response.finish_reasons': [event.finishReason],
        'gen_ai.usage.input_tokens': event.usage.inputTokens,
        'gen_ai.usage.output_tokens': event.usage.outputTokens,
        'gen_ai.usage.cache_read.input_tokens': event.usage.cachedInputTokens,
        'gen_ai.output.messages': {
          output: () =>
            event.object != null
              ? JSON.stringify(
                  formatObjectOutputMessages({
                    objectText: JSON.stringify(event.object),
                    finishReason: event.finishReason,
                  }),
                )
              : undefined,
        },
        ...selectLegacyAttributes(telemetry, this.legacyAttributes, {
          'ai.response.finishReason': event.finishReason,
          'ai.response.object': {
            output: () =>
              event.object != null ? JSON.stringify(event.object) : undefined,
          },
          'ai.response.providerMetadata': event.providerMetadata
            ? JSON.stringify(event.providerMetadata)
            : undefined,
          'ai.usage.inputTokens': event.usage.inputTokens,
          'ai.usage.outputTokens': event.usage.outputTokens,
          'ai.usage.totalTokens': event.usage.totalTokens,
          'ai.usage.reasoningTokens': event.usage.reasoningTokens,
          'ai.usage.cachedInputTokens': event.usage.cachedInputTokens,
        }),
      }),
    );

    state.rootSpan.end();
    this.cleanupCallState(event.callId);
  }

  private onEmbedOperationFinish(event: EmbedEndEvent): void {
    const state = this.getCallState(event.callId);
    if (!state?.rootSpan) return;

    const { telemetry } = state;
    const isMany = state.operationId === 'ai.embedMany';

    state.rootSpan.setAttributes(
      selectAttributes(telemetry, {
        'gen_ai.usage.input_tokens': event.usage.tokens,
        ...selectLegacyAttributes(telemetry, this.legacyAttributes, {
          ...(isMany
            ? {
                'ai.embeddings': {
                  output: () =>
                    (event.embedding as number[][]).map(e => JSON.stringify(e)),
                },
              }
            : {
                'ai.embedding': {
                  output: () => JSON.stringify(event.embedding),
                },
              }),
          'ai.usage.tokens': event.usage.tokens,
        }),
      }),
    );

    state.rootSpan.end();
    this.cleanupCallState(event.callId);
  }

  onEmbedStart(event: EmbeddingModelCallStartEvent): void {
    const state = this.getCallState(event.callId);
    if (!state?.rootSpan || !state.rootContext) return;

    const { telemetry } = state;
    const providerName = mapProviderName(state.provider);

    const attributes = selectAttributes(telemetry, {
      'gen_ai.operation.name': 'embeddings',
      'gen_ai.provider.name': providerName,
      'gen_ai.request.model': state.modelId,
      ...selectLegacyAttributes(telemetry, this.legacyAttributes, {
        ...assembleOperationName({
          operationId: event.operationId,
          telemetry,
        }),
        ...state.baseTelemetryAttributes,
        'ai.values': {
          input: () => event.values.map(v => JSON.stringify(v)),
        },
      }),
    });

    const spanName = `embeddings ${state.modelId}`;
    const embedSpan = this.tracer.startSpan(
      spanName,
      { attributes, kind: SpanKind.CLIENT },
      state.rootContext,
    );
    const embedContext = trace.setSpan(state.rootContext, embedSpan);

    state.embedSpans.set(event.embedCallId, {
      span: embedSpan,
      context: embedContext,
    });
  }

  onEmbedFinish(event: EmbeddingModelCallEndEvent): void {
    const state = this.getCallState(event.callId);
    if (!state) return;

    const embedSpanEntry = state.embedSpans.get(event.embedCallId);
    if (!embedSpanEntry) return;

    const { span } = embedSpanEntry;
    const { telemetry } = state;

    span.setAttributes(
      selectAttributes(telemetry, {
        'gen_ai.usage.input_tokens': event.usage.tokens,
        ...selectLegacyAttributes(telemetry, this.legacyAttributes, {
          'ai.embeddings': {
            output: () =>
              event.embeddings.map(embedding => JSON.stringify(embedding)),
          },
          'ai.usage.tokens': event.usage.tokens,
        }),
      }),
    );

    span.end();
    state.embedSpans.delete(event.embedCallId);
  }

  private onRerankOperationStart(
    event: InferTelemetryEvent<RerankStartEvent>,
  ): void {
    const telemetry: TelemetryOptions = {
      recordInputs: event.recordInputs,
      recordOutputs: event.recordOutputs,
      functionId: event.functionId,
    };

    const settings: Record<string, unknown> = {
      maxRetries: event.maxRetries,
    };

    const providerName = mapProviderName(event.provider);
    const baseTelemetryAttributes = getBaseTelemetryAttributes({
      model: { provider: event.provider, modelId: event.modelId },
      headers: event.headers,
      settings,
      context: undefined,
    });

    const attributes = selectAttributes(telemetry, {
      'gen_ai.operation.name': 'rerank',
      'gen_ai.provider.name': providerName,
      'gen_ai.request.model': event.modelId,
      ...selectLegacyAttributes(telemetry, this.legacyAttributes, {
        ...assembleOperationName({
          operationId: event.operationId,
          telemetry,
        }),
        ...baseTelemetryAttributes,
        'ai.documents': {
          input: () => event.documents.map(d => JSON.stringify(d)),
        },
      }),
    });

    const spanName = `rerank ${event.modelId}`;
    const rootSpan = this.tracer.startSpan(spanName, {
      attributes,
      kind: SpanKind.CLIENT,
    });
    const rootContext = trace.setSpan(context.active(), rootSpan);

    this.callStates.set(event.callId, {
      operationId: event.operationId,
      telemetry,
      rootSpan,
      rootContext,
      stepSpan: undefined,
      stepContext: undefined,
      inferenceSpan: undefined,
      inferenceContext: undefined,
      embedSpans: new Map(),
      rerankSpan: undefined,
      toolSpans: new Map(),
      settings,
      provider: event.provider,
      modelId: event.modelId,
      baseTelemetryAttributes,
    });
  }

  private onRerankOperationFinish(event: RerankEndEvent): void {
    const state = this.getCallState(event.callId);
    if (!state?.rootSpan) return;

    state.rootSpan.end();
    this.cleanupCallState(event.callId);
  }

  onRerankStart(event: RerankingModelCallStartEvent): void {
    const state = this.getCallState(event.callId);
    if (!state?.rootSpan || !state.rootContext) return;

    const { telemetry } = state;
    const providerName = mapProviderName(state.provider);

    const attributes = selectAttributes(telemetry, {
      'gen_ai.operation.name': 'rerank',
      'gen_ai.provider.name': providerName,
      'gen_ai.request.model': state.modelId,
      ...selectLegacyAttributes(telemetry, this.legacyAttributes, {
        ...assembleOperationName({
          operationId: event.operationId,
          telemetry,
        }),
        ...state.baseTelemetryAttributes,
        'ai.documents': {
          input: () => event.documents.map(d => JSON.stringify(d)),
        },
      }),
    });

    const spanName = `rerank ${state.modelId}`;
    const rerankSpan = this.tracer.startSpan(
      spanName,
      { attributes, kind: SpanKind.CLIENT },
      state.rootContext,
    );
    const rerankContext = trace.setSpan(state.rootContext, rerankSpan);

    state.rerankSpan = { span: rerankSpan, context: rerankContext };
  }

  onRerankFinish(event: RerankingModelCallEndEvent): void {
    const state = this.getCallState(event.callId);
    if (!state?.rerankSpan) return;

    const { span } = state.rerankSpan;
    const { telemetry } = state;

    span.setAttributes(
      selectLegacyAttributes(telemetry, this.legacyAttributes, {
        'ai.ranking.type': event.documentsType,
        'ai.ranking': {
          output: () => event.ranking.map(r => JSON.stringify(r)),
        },
      }),
    );

    span.end();
    state.rerankSpan = undefined;
  }

  onChunk(event: StreamTextChunkEvent<ToolSet>): void {
    // No-op: streaming chunk events are not part of the GenAI SemConv.
    if (!this.legacyAttributes?.streaming) return;

    const chunk = event.chunk as {
      type: string;
      callId?: unknown;
      attributes?: unknown;
    };

    if (
      typeof chunk.callId !== 'string' ||
      (chunk.type !== 'ai.stream.firstChunk' &&
        chunk.type !== 'ai.stream.finish')
    ) {
      return;
    }

    const state = this.getCallState(chunk.callId);
    if (!state?.stepSpan) return;

    const attributes = filterLegacyAttributes(
      Object.fromEntries(
        Object.entries(
          (chunk.attributes as Record<string, unknown>) ?? {},
        ).filter(([, value]) => value != null),
      ) as Attributes,
      this.legacyAttributes,
    );

    if (attributes != null && Object.keys(attributes).length > 0) {
      state.stepSpan.setAttributes(attributes);
    }
  }

  onError(error: unknown): void {
    const event = error as { callId?: string; error?: unknown };
    if (!event?.callId) return;

    const state = this.getCallState(event.callId);
    if (!state?.rootSpan) return;

    const actualError = event.error ?? error;

    for (const { span: toolSpan } of state.toolSpans.values()) {
      recordSpanError(toolSpan, actualError);
      toolSpan.end();
    }
    state.toolSpans.clear();

    if (state.inferenceSpan) {
      recordSpanError(state.inferenceSpan, actualError);
      state.inferenceSpan.end();
      state.inferenceSpan = undefined;
      state.inferenceContext = undefined;
    }

    if (state.stepSpan) {
      recordSpanError(state.stepSpan, actualError);
      state.stepSpan.end();
      state.stepSpan = undefined;
      state.stepContext = undefined;
    }

    for (const { span: embedSpan } of state.embedSpans.values()) {
      recordSpanError(embedSpan, actualError);
      embedSpan.end();
    }
    state.embedSpans.clear();

    if (state.rerankSpan) {
      recordSpanError(state.rerankSpan.span, actualError);
      state.rerankSpan.span.end();
      state.rerankSpan = undefined;
    }

    recordSpanError(state.rootSpan, actualError);

    state.rootSpan.end();
    this.cleanupCallState(event.callId);
  }
}
