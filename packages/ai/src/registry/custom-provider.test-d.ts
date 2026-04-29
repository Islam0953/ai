import type {
  EmbeddingModelV4,
  Experimental_VideoModelV4,
  ImageModelV4,
  LanguageModelV4,
  ProviderV4,
  RerankingModelV4,
  SpeechModelV4,
  TranscriptionModelV4,
} from '@ai-sdk/provider';
import { describe, expectTypeOf, it } from 'vitest';
import { customProvider } from './custom-provider';
import { MockEmbeddingModelV4 } from '../test/mock-embedding-model-v4';
import { MockImageModelV4 } from '../test/mock-image-model-v4';
import { MockLanguageModelV4 } from '../test/mock-language-model-v4';
import { MockProviderV4 } from '../test/mock-provider-v4';
import { MockRerankingModelV4 } from '../test/mock-reranking-model-v4';
import { MockSpeechModelV4 } from '../test/mock-speech-model-v4';
import { MockTranscriptionModelV4 } from '../test/mock-transcription-model-v4';
import { MockVideoModelV4 } from '../test/mock-video-model-v4';

/**
 * Type-level tests for `customProvider`. Literal model identifiers follow the private
 * `ExtractModelId` helper in `custom-provider.ts` (string keys of each model record).
 */

describe('customProvider autocomplete / literal model identifiers', () => {
  const languageModel = new MockLanguageModelV4();
  const embeddingModel = new MockEmbeddingModelV4();
  const imageModel = new MockImageModelV4();
  const transcriptionModel = new MockTranscriptionModelV4();
  const speechModel = new MockSpeechModelV4();
  const rerankingModel = new MockRerankingModelV4();
  const videoModel = new MockVideoModelV4();

  const provider = customProvider({
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
    videoModels: {
      'preview-video': videoModel,
    },
  });

  it('narrows languageModel identifiers to the configured languageModels keys', () => {
    type InferredLanguageModelIdentifier = Parameters<
      (typeof provider)['languageModel']
    >[0];

    expectTypeOf<InferredLanguageModelIdentifier>().toEqualTypeOf<
      'opus' | 'sonnet' | 'haiku'
    >();

    expectTypeOf(
      provider.languageModel('haiku'),
    ).toEqualTypeOf<LanguageModelV4>();
    expectTypeOf(
      provider.languageModel('opus'),
    ).toEqualTypeOf<LanguageModelV4>();

    expectTypeOf<'sonnet'>().toMatchTypeOf<InferredLanguageModelIdentifier>();
    expectTypeOf<InferredLanguageModelIdentifier>().not.toMatchTypeOf<'gpt-5'>();
    expectTypeOf<InferredLanguageModelIdentifier>().not.toMatchTypeOf<'typo'>();
  });

  it('narrows embeddingModel, imageModel, and other accessors to their record keys', () => {
    type InferredEmbeddingIdentifier = Parameters<
      (typeof provider)['embeddingModel']
    >[0];
    type InferredImageIdentifier = Parameters<
      (typeof provider)['imageModel']
    >[0];
    type InferredTranscriptionIdentifier = Parameters<
      (typeof provider)['transcriptionModel']
    >[0];
    type InferredSpeechIdentifier = Parameters<
      (typeof provider)['speechModel']
    >[0];
    type InferredRerankingIdentifier = Parameters<
      (typeof provider)['rerankingModel']
    >[0];
    type InferredVideoIdentifier = Parameters<
      (typeof provider)['videoModel']
    >[0];

    expectTypeOf<InferredEmbeddingIdentifier>().toEqualTypeOf<
      'small' | 'large'
    >();
    expectTypeOf<InferredImageIdentifier>().toEqualTypeOf<'photon'>();
    expectTypeOf<InferredTranscriptionIdentifier>().toEqualTypeOf<'whisper-1'>();
    expectTypeOf<InferredSpeechIdentifier>().toEqualTypeOf<'tts1'>();
    expectTypeOf<InferredRerankingIdentifier>().toEqualTypeOf<'rerank'>();
    expectTypeOf<InferredVideoIdentifier>().toEqualTypeOf<'preview-video'>();

    expectTypeOf(
      provider.embeddingModel('small'),
    ).toEqualTypeOf<EmbeddingModelV4>();
    expectTypeOf(provider.imageModel('photon')).toEqualTypeOf<ImageModelV4>();
    expectTypeOf(
      provider.transcriptionModel('whisper-1'),
    ).toEqualTypeOf<TranscriptionModelV4>();
    expectTypeOf(provider.speechModel('tts1')).toEqualTypeOf<SpeechModelV4>();
    expectTypeOf(
      provider.rerankingModel('rerank'),
    ).toEqualTypeOf<RerankingModelV4>();
    expectTypeOf(
      provider.videoModel('preview-video'),
    ).toEqualTypeOf<Experimental_VideoModelV4>();

    expectTypeOf<InferredEmbeddingIdentifier>().not.toMatchTypeOf<'wrong-key'>();
  });

  it('satisfies ProviderV4 so the instance is assignable where a provider is required', () => {
    expectTypeOf(provider).toMatchTypeOf<ProviderV4>();
  });

  it('uses hyphenated and dotted keys in the literal unions', () => {
    const hyphenProvider = customProvider({
      languageModels: {
        'gpt-4o-high-reasoning': languageModel,
        'claude-3-5-sonnet': languageModel,
      },
    });

    type HyphenLanguageIdentifiers = Parameters<
      (typeof hyphenProvider)['languageModel']
    >[0];

    expectTypeOf<HyphenLanguageIdentifiers>().toEqualTypeOf<
      'gpt-4o-high-reasoning' | 'claude-3-5-sonnet'
    >();
  });
});

describe('customProvider negative typing', () => {
  const languageModel = new MockLanguageModelV4();

  it('rejects identifiers that are not keys on the corresponding record (assignability)', () => {
    const narrowProvider = customProvider({
      languageModels: { haiku: languageModel },
    });

    type ConfiguredLanguageModelIdentifier = Parameters<
      (typeof narrowProvider)['languageModel']
    >[0];

    expectTypeOf<ConfiguredLanguageModelIdentifier>().toEqualTypeOf<'haiku'>();

    // `customProvider` returns `ProviderV4 & { ... narrow methods ... }`. Method parameters
    // in intersections are checked bivariantly for direct calls, so invalid literals may
    // still type-check on `narrowProvider.languageModel('opus')`. Assigning to the
    // parameter type catches the intended rejection.
    // @ts-expect-error opus is not a key in languageModels
    const _wrongLanguageIdentifier: ConfiguredLanguageModelIdentifier = 'opus';
  });

  it('uses string for embeddingModel identifiers when embeddingModels is not configured', () => {
    const languageOnlyProvider = customProvider({
      languageModels: { haiku: languageModel },
    });

    type EmbeddingIdentifierWhenUnconfigured = Parameters<
      (typeof languageOnlyProvider)['embeddingModel']
    >[0];

    expectTypeOf<EmbeddingIdentifierWhenUnconfigured>().toEqualTypeOf<string>();
  });

  it('rejects wrong embedding and image identifiers for a fully configured provider', () => {
    const configuredProvider = customProvider({
      languageModels: { haiku: languageModel },
      embeddingModels: { small: new MockEmbeddingModelV4() },
      imageModels: { photon: new MockImageModelV4() },
    });

    type EmbeddingIdentifier = Parameters<
      (typeof configuredProvider)['embeddingModel']
    >[0];
    type ImageIdentifier = Parameters<
      (typeof configuredProvider)['imageModel']
    >[0];

    // @ts-expect-error typo is not a configured embedding model id
    const _wrongEmbeddingIdentifier: EmbeddingIdentifier = 'typo';
    // @ts-expect-error typo is not a configured image model id
    const _wrongImageIdentifier: ImageIdentifier = 'typo';
  });
});

describe('customProvider with fallback provider typing', () => {
  const languageModel = new MockLanguageModelV4();

  const fallbackProvider = new MockProviderV4({
    languageModels: { 'fallback-language': languageModel },
    embeddingModels: { 'fallback-embedding': new MockEmbeddingModelV4() },
  });

  it('narrows languageModel identifiers to configured keys; fallback-only ids are not in the union', () => {
    const provider = customProvider({
      languageModels: { alias: languageModel },
      fallbackProvider,
    });

    type LanguageIdentifiers = Parameters<
      (typeof provider)['languageModel']
    >[0];

    expectTypeOf<LanguageIdentifiers>().toEqualTypeOf<'alias'>();

    provider.languageModel('alias');

    // @ts-expect-error identifiers resolved only via fallback are not part of ExtractModelId
    const _fallbackOnlyIdentifier: LanguageIdentifiers = 'fallback-language';
  });
});
