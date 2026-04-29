import type {
  EmbeddingModelV4,
  ImageModelV4,
  LanguageModelV4,
  ProviderV4,
  RerankingModelV4,
  SpeechModelV4,
  TranscriptionModelV4,
} from '@ai-sdk/provider';
import { describe, expectTypeOf, it } from 'vitest';
import type { ExtractLiteralUnion } from '../util/extract-literal-union';
import { customProvider } from './custom-provider';
import {
  createProviderRegistry,
  type ProviderRegistryProvider,
} from './provider-registry';
import { MockEmbeddingModelV4 } from '../test/mock-embedding-model-v4';
import { MockImageModelV4 } from '../test/mock-image-model-v4';
import { MockLanguageModelV4 } from '../test/mock-language-model-v4';
import { MockProviderV4 } from '../test/mock-provider-v4';
import { MockRerankingModelV4 } from '../test/mock-reranking-model-v4';
import { MockSpeechModelV4 } from '../test/mock-speech-model-v4';
import { MockTranscriptionModelV4 } from '../test/mock-transcription-model-v4';

/** Same construction as `ProviderRegistryProvider` for assertion-only tests. */
type RegistryLanguageModelIdentifier<
  registeredProviders extends Record<string, ProviderV4>,
  separator extends string = ':',
> = {
  [providerKey in keyof registeredProviders]: providerKey extends string
    ? `${providerKey & string}${separator}${ExtractLiteralUnion<
        Parameters<
          NonNullable<registeredProviders[providerKey]['languageModel']>
        >[0]
      >}`
    : never;
}[keyof registeredProviders];

type RegistryEmbeddingModelIdentifier<
  registeredProviders extends Record<string, ProviderV4>,
  separator extends string = ':',
> = {
  [providerKey in keyof registeredProviders]: providerKey extends string
    ? `${providerKey & string}${separator}${ExtractLiteralUnion<
        Parameters<
          NonNullable<registeredProviders[providerKey]['embeddingModel']>
        >[0]
      >}`
    : never;
}[keyof registeredProviders];

type RegistryImageModelIdentifier<
  registeredProviders extends Record<string, ProviderV4>,
  separator extends string = ':',
> = {
  [providerKey in keyof registeredProviders]: providerKey extends string
    ? `${providerKey & string}${separator}${ExtractLiteralUnion<
        Parameters<
          NonNullable<registeredProviders[providerKey]['imageModel']>
        >[0]
      >}`
    : never;
}[keyof registeredProviders];

type RegistryTranscriptionModelIdentifier<
  registeredProviders extends Record<string, ProviderV4>,
  separator extends string = ':',
> = {
  [providerKey in keyof registeredProviders]: providerKey extends string
    ? `${providerKey & string}${separator}${ExtractLiteralUnion<
        Parameters<
          NonNullable<registeredProviders[providerKey]['transcriptionModel']>
        >[0]
      >}`
    : never;
}[keyof registeredProviders];

type RegistrySpeechModelIdentifier<
  registeredProviders extends Record<string, ProviderV4>,
  separator extends string = ':',
> = {
  [providerKey in keyof registeredProviders]: providerKey extends string
    ? `${providerKey & string}${separator}${ExtractLiteralUnion<
        Parameters<
          NonNullable<registeredProviders[providerKey]['speechModel']>
        >[0]
      >}`
    : never;
}[keyof registeredProviders];

type RegistryRerankingModelIdentifier<
  registeredProviders extends Record<string, ProviderV4>,
  separator extends string = ':',
> = {
  [providerKey in keyof registeredProviders]: providerKey extends string
    ? `${providerKey & string}${separator}${ExtractLiteralUnion<
        Parameters<
          NonNullable<registeredProviders[providerKey]['rerankingModel']>
        >[0]
      >}`
    : never;
}[keyof registeredProviders];

describe('createProviderRegistry autocomplete / literal identifiers', () => {
  const languageModel = new MockLanguageModelV4();
  const embeddingModel = new MockEmbeddingModelV4();
  const imageModel = new MockImageModelV4();
  const transcriptionModel = new MockTranscriptionModelV4();
  const speechModel = new MockSpeechModelV4();
  const rerankingModel = new MockRerankingModelV4();

  const anthropicCustomProvider = customProvider({
    languageModels: {
      opus: languageModel,
      sonnet: languageModel,
      haiku: languageModel,
    },
    embeddingModels: {
      small: embeddingModel,
      large: embeddingModel,
    },
    imageModels: {
      photon: imageModel,
    },
    transcriptionModels: {
      'whisper-1': transcriptionModel,
    },
    speechModels: {
      tts1: speechModel,
    },
    rerankingModels: {
      rerank: rerankingModel,
    },
  });

  const openaiCustomProvider = customProvider({
    languageModels: {
      'gpt-5': languageModel,
      'gpt-4o-high-reasoning': languageModel,
    },
    embeddingModels: {
      'text-embedding-3-small': embeddingModel,
    },
    imageModels: {
      dalle: imageModel,
    },
  });

  const registeredProviders = {
    openai: openaiCustomProvider,
    anthropic: anthropicCustomProvider,
  };

  const registry = createProviderRegistry(registeredProviders);

  it('infers a finite union of language model identifiers (autocomplete-friendly)', () => {
    type ExpectedLanguageModelIdentifiers =
      | 'openai:gpt-5'
      | 'openai:gpt-4o-high-reasoning'
      | 'anthropic:opus'
      | 'anthropic:sonnet'
      | 'anthropic:haiku';

    expectTypeOf<
      RegistryLanguageModelIdentifier<typeof registeredProviders>
    >().toEqualTypeOf<ExpectedLanguageModelIdentifiers>();

    expectTypeOf(
      registry.languageModel('openai:gpt-5'),
    ).toEqualTypeOf<LanguageModelV4>();
    expectTypeOf(
      registry.languageModel('anthropic:haiku'),
    ).toEqualTypeOf<LanguageModelV4>();

    expectTypeOf<'anthropic:opus'>().toMatchTypeOf<
      RegistryLanguageModelIdentifier<typeof registeredProviders>
    >();
    expectTypeOf<'openai:gpt-5'>().toMatchTypeOf<
      RegistryLanguageModelIdentifier<typeof registeredProviders>
    >();

    expectTypeOf<
      RegistryLanguageModelIdentifier<typeof registeredProviders>
    >().not.toMatchTypeOf<'openai:not-a-configured-alias'>();
    expectTypeOf<
      RegistryLanguageModelIdentifier<typeof registeredProviders>
    >().not.toMatchTypeOf<'anthropic:typo-model'>();
  });

  it('infers literal unions per model kind', () => {
    type ExpectedEmbeddingModelIdentifiers =
      | 'openai:text-embedding-3-small'
      | 'anthropic:small'
      | 'anthropic:large';

    type ExpectedImageModelIdentifiers = 'openai:dalle' | 'anthropic:photon';

    type ExpectedTranscriptionModelIdentifiers = 'anthropic:whisper-1';

    type ExpectedSpeechModelIdentifiers = 'anthropic:tts1';

    type ExpectedRerankingModelIdentifiers = 'anthropic:rerank';

    expectTypeOf<
      RegistryEmbeddingModelIdentifier<typeof registeredProviders>
    >().toEqualTypeOf<ExpectedEmbeddingModelIdentifiers>();
    expectTypeOf<
      RegistryImageModelIdentifier<typeof registeredProviders>
    >().toEqualTypeOf<ExpectedImageModelIdentifiers>();
    expectTypeOf<
      RegistryTranscriptionModelIdentifier<typeof registeredProviders>
    >().toEqualTypeOf<ExpectedTranscriptionModelIdentifiers>();
    expectTypeOf<
      RegistrySpeechModelIdentifier<typeof registeredProviders>
    >().toEqualTypeOf<ExpectedSpeechModelIdentifiers>();
    expectTypeOf<
      RegistryRerankingModelIdentifier<typeof registeredProviders>
    >().toEqualTypeOf<ExpectedRerankingModelIdentifiers>();

    expectTypeOf(
      registry.embeddingModel('anthropic:small'),
    ).toEqualTypeOf<EmbeddingModelV4>();
    expectTypeOf(
      registry.imageModel('openai:dalle'),
    ).toEqualTypeOf<ImageModelV4>();
    expectTypeOf(
      registry.transcriptionModel('anthropic:whisper-1'),
    ).toEqualTypeOf<TranscriptionModelV4>();
    expectTypeOf(
      registry.speechModel('anthropic:tts1'),
    ).toEqualTypeOf<SpeechModelV4>();
    expectTypeOf(
      registry.rerankingModel('anthropic:rerank'),
    ).toEqualTypeOf<RerankingModelV4>();
  });

  it('uses the custom separator in template-literal identifiers', () => {
    const registryWithCustomSeparator = createProviderRegistry(
      registeredProviders,
      { separator: ' > ' },
    );

    type ExpectedLanguageModelIdentifiersWithSeparator =
      RegistryLanguageModelIdentifier<typeof registeredProviders, ' > '>;

    expectTypeOf<
      RegistryLanguageModelIdentifier<typeof registeredProviders, ' > '>
    >().toEqualTypeOf<
      | 'openai > gpt-5'
      | 'openai > gpt-4o-high-reasoning'
      | 'anthropic > opus'
      | 'anthropic > sonnet'
      | 'anthropic > haiku'
    >();

    expectTypeOf(
      registryWithCustomSeparator.languageModel('anthropic > haiku'),
    ).toEqualTypeOf<LanguageModelV4>();

    expectTypeOf(registryWithCustomSeparator).toEqualTypeOf<
      ProviderRegistryProvider<typeof registeredProviders, ' > '>
    >();

    expectTypeOf<'anthropic > haiku'>().toMatchTypeOf<ExpectedLanguageModelIdentifiersWithSeparator>();
  });

  it('falls back to providerKey:any-model when the provider only uses string model ids', () => {
    const registryWithPlainProvider = createProviderRegistry({
      plain: new MockProviderV4({
        languageModels: { a: languageModel, b: languageModel },
      }),
    });

    expectTypeOf<
      RegistryLanguageModelIdentifier<{ plain: MockProviderV4 }>
    >().toEqualTypeOf<never>();

    expectTypeOf(
      registryWithPlainProvider.languageModel('plain:a'),
    ).toEqualTypeOf<LanguageModelV4>();
    expectTypeOf(
      registryWithPlainProvider.languageModel('plain:anything-goes'),
    ).toEqualTypeOf<LanguageModelV4>();

    type looseLanguageModelArgument = Parameters<
      (typeof registryWithPlainProvider)['languageModel']
    >[0];
    expectTypeOf<looseLanguageModelArgument>().toMatchTypeOf<`plain:${string}`>();
  });
});

describe('createProviderRegistry negative typing', () => {
  const languageModel = new MockLanguageModelV4();
  const embeddingModel = new MockEmbeddingModelV4();
  const imageModel = new MockImageModelV4();
  const transcriptionModel = new MockTranscriptionModelV4();
  const speechModel = new MockSpeechModelV4();
  const rerankingModel = new MockRerankingModelV4();

  const anthropicOnlyRegisteredProviders = {
    anthropic: customProvider({
      languageModels: { haiku: languageModel },
      embeddingModels: { small: embeddingModel },
      imageModels: { photon: imageModel },
      transcriptionModels: { 'whisper-1': transcriptionModel },
      speechModels: { tts1: speechModel },
      rerankingModels: { rerank: rerankingModel },
    }),
  };

  const registryAnthropicOnly = createProviderRegistry(
    anthropicOnlyRegisteredProviders,
  );

  it('rejects registry identifiers whose provider key is not registered (language model)', () => {
    registryAnthropicOnly.languageModel('anthropic:haiku');

    // @ts-expect-error provider key must exist in the registry
    registryAnthropicOnly.languageModel('unknown:haiku');
  });

  it('rejects registry identifiers whose provider key is not registered (other model kinds)', () => {
    registryAnthropicOnly.embeddingModel('anthropic:small');
    registryAnthropicOnly.imageModel('anthropic:photon');
    registryAnthropicOnly.transcriptionModel('anthropic:whisper-1');
    registryAnthropicOnly.speechModel('anthropic:tts1');
    registryAnthropicOnly.rerankingModel('anthropic:rerank');

    // @ts-expect-error provider key must exist in the registry
    registryAnthropicOnly.embeddingModel('unknown:small');
    // @ts-expect-error provider key must exist in the registry
    registryAnthropicOnly.imageModel('unknown:photon');
    // @ts-expect-error provider key must exist in the registry
    registryAnthropicOnly.transcriptionModel('unknown:whisper-1');
    // @ts-expect-error provider key must exist in the registry
    registryAnthropicOnly.speechModel('unknown:tts1');
    // @ts-expect-error provider key must exist in the registry
    registryAnthropicOnly.rerankingModel('unknown:rerank');
  });

  it('rejects the default colon separator when the registry uses a custom separator', () => {
    const registryWithCustomSeparator = createProviderRegistry(
      anthropicOnlyRegisteredProviders,
      { separator: ' > ' },
    );

    registryWithCustomSeparator.languageModel('anthropic > haiku');

    // @ts-expect-error separator must match the configured registry separator
    registryWithCustomSeparator.languageModel('anthropic:haiku');
  });

  it('rejects identifiers when the provider key is not registered (narrow registry)', () => {
    const registeredProvidersOpenaiOnly = {
      openai: customProvider({
        languageModels: { 'gpt-5': languageModel },
        embeddingModels: { 'text-embedding-3-small': embeddingModel },
      }),
    };
    const registryOpenaiOnly = createProviderRegistry(
      registeredProvidersOpenaiOnly,
    );

    registryOpenaiOnly.languageModel('openai:gpt-5');
    registryOpenaiOnly.embeddingModel('openai:text-embedding-3-small');

    // @ts-expect-error anthropic is not a registered provider key
    registryOpenaiOnly.languageModel('anthropic:haiku');
    // @ts-expect-error anthropic is not a registered provider key
    registryOpenaiOnly.embeddingModel('anthropic:small');
  });

  it('rejects identifiers for providers that are absent from a multi-provider registry', () => {
    const multiRegisteredProviders = {
      openai: customProvider({
        languageModels: { 'gpt-5': languageModel },
      }),
      anthropic: customProvider({
        languageModels: { haiku: languageModel },
      }),
    };
    const multiProviderRegistry = createProviderRegistry(
      multiRegisteredProviders,
    );

    multiProviderRegistry.languageModel('openai:gpt-5');
    multiProviderRegistry.languageModel('anthropic:haiku');

    // @ts-expect-error mistral is not a registered provider key
    multiProviderRegistry.languageModel('mistral:whatever');
    // @ts-expect-error mistral is not a registered provider key
    multiProviderRegistry.imageModel('mistral:any');
  });

  it('still accepts arbitrary model suffix for a registered provider (second overload)', () => {
    // Not every typo is a type error; identifier validation happens at runtime.
    registryAnthropicOnly.languageModel('anthropic:dynamic-or-typo');
  });
});
