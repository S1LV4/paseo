import { existsSync } from "node:fs";
import type pino from "pino";

import { loadSherpaOnnxNode } from "./sherpa-onnx-node-loader.js";

function assertFileExists(filePath: string, label: string): void {
  if (!existsSync(filePath)) {
    throw new Error(`Missing ${label}: ${filePath}`);
  }
}

export type SherpaOfflineRecognizerModel =
  | {
      kind: "nemo_transducer";
      encoder: string;
      decoder: string;
      joiner: string;
      tokens: string;
    }
  | {
      kind: "whisper";
      encoder: string;
      decoder: string;
      tokens: string;
      language: string;
      task?: "transcribe" | "translate";
      tailPaddings?: number;
    };

export interface SherpaOfflineRecognizerConfig {
  model: SherpaOfflineRecognizerModel;
  numThreads?: number;
  provider?: "cpu";
  debug?: 0 | 1;
  sampleRate?: number;
  featureDim?: number;
  decodingMethod?: "greedy_search";
  maxActivePaths?: number;
}

interface SherpaOfflineRecognizerNative {
  config?: { featConfig?: { sampleRate?: number } };
  createStream: () => unknown;
  decode: (stream: unknown) => void;
  getResult: (stream: unknown) => { text?: string } | string | undefined;
  free?: () => void;
}

interface SherpaOfflineStreamNative {
  acceptWaveform: ((arg: { samples: Float32Array; sampleRate: number }) => void) &
    ((sampleRate: number, samples: Float32Array) => void);
  free?: () => void;
}

function buildNemoTransducerConfig(config: SherpaOfflineRecognizerConfig & { model: { kind: "nemo_transducer"; encoder: string; decoder: string; joiner: string; tokens: string } }): unknown {
  return {
    featConfig: {
      sampleRate: config.sampleRate ?? 16000,
      featureDim: config.featureDim ?? 80,
    },
    modelConfig: {
      transducer: {
        encoder: config.model.encoder,
        decoder: config.model.decoder,
        joiner: config.model.joiner,
      },
      tokens: config.model.tokens,
      modelType: "nemo_transducer",
      numThreads: config.numThreads ?? 1,
      provider: config.provider ?? "cpu",
      debug: config.debug ?? 0,
    },
    decodingMethod: config.decodingMethod ?? "greedy_search",
    maxActivePaths: config.maxActivePaths ?? 4,
  };
}

function buildWhisperConfig(config: SherpaOfflineRecognizerConfig & { model: { kind: "whisper"; encoder: string; decoder: string; tokens: string; language: string; task?: "transcribe" | "translate"; tailPaddings?: number } }): unknown {
  return {
    featConfig: {
      sampleRate: config.sampleRate ?? 16000,
      featureDim: config.featureDim ?? 80,
    },
    modelConfig: {
      whisper: {
        encoder: config.model.encoder,
        decoder: config.model.decoder,
        language: config.model.language,
        task: config.model.task ?? "transcribe",
        tailPaddings: config.model.tailPaddings ?? -1,
      },
      tokens: config.model.tokens,
      modelType: "whisper",
      numThreads: config.numThreads ?? 2,
      provider: config.provider ?? "cpu",
      debug: config.debug ?? 0,
    },
    decodingMethod: config.decodingMethod ?? "greedy_search",
    maxActivePaths: config.maxActivePaths ?? 4,
  };
}

function buildRecognizerConfig(config: SherpaOfflineRecognizerConfig): unknown {
  if (config.model.kind === "whisper") {
    return buildWhisperConfig(config as SherpaOfflineRecognizerConfig & { model: { kind: "whisper"; encoder: string; decoder: string; tokens: string; language: string; task?: "transcribe" | "translate"; tailPaddings?: number } });
  }
  return buildNemoTransducerConfig(config as SherpaOfflineRecognizerConfig & { model: { kind: "nemo_transducer"; encoder: string; decoder: string; joiner: string; tokens: string } });
}

function assertModelFiles(model: SherpaOfflineRecognizerModel): void {
  assertFileExists(model.encoder, "offline encoder");
  assertFileExists(model.decoder, "offline decoder");
  if (model.kind === "nemo_transducer") {
    assertFileExists(model.joiner, "offline joiner");
  }
  assertFileExists(model.tokens, "tokens");
}

export class SherpaOfflineRecognizerEngine {
  public readonly recognizer: SherpaOfflineRecognizerNative;
  public readonly sampleRate: number;
  private readonly logger: pino.Logger;

  constructor(config: SherpaOfflineRecognizerConfig, logger: pino.Logger) {
    this.logger = logger.child({
      module: "speech",
      provider: "local",
      component: "offline-recognizer",
    });

    assertModelFiles(config.model);

    const sherpa = loadSherpaOnnxNode();
    const recognizerConfig = buildRecognizerConfig(config);

    this.recognizer = new (
      sherpa as unknown as {
        OfflineRecognizer: new (config: unknown) => SherpaOfflineRecognizerNative;
      }
    ).OfflineRecognizer(recognizerConfig);

    const featConfig = (recognizerConfig as { featConfig: { sampleRate: number } }).featConfig;
    const sr = this.recognizer?.config?.featConfig?.sampleRate;
    this.sampleRate =
      typeof sr === "number" && Number.isFinite(sr) && sr > 0 ? sr : featConfig.sampleRate;

    this.logger.info(
      {
        sampleRate: this.sampleRate,
        numThreads: config.numThreads ?? (config.model.kind === "whisper" ? 2 : 1),
        modelType: config.model.kind,
        ...(config.model.kind === "whisper" ? { language: config.model.language } : {}),
      },
      "Sherpa offline recognizer initialized",
    );
  }

  createStream(): SherpaOfflineStreamNative {
    return this.recognizer.createStream() as SherpaOfflineStreamNative;
  }

  acceptWaveform(
    stream: SherpaOfflineStreamNative,
    sampleRate: number,
    samples: Float32Array,
  ): void {
    if (!stream || typeof stream.acceptWaveform !== "function") {
      throw new Error("Unexpected sherpa offline stream: missing acceptWaveform()");
    }

    // sherpa-onnx-node expects: acceptWaveform({ samples, sampleRate })
    // sherpa-onnx (WASM) expects: acceptWaveform(sampleRate, samples)
    if (stream.acceptWaveform.length <= 1) {
      stream.acceptWaveform({ samples, sampleRate });
    } else {
      stream.acceptWaveform(sampleRate, samples);
    }
  }

  free(): void {
    try {
      this.recognizer?.free?.();
    } catch (err) {
      this.logger.warn({ err }, "Failed to free sherpa offline recognizer");
    }
  }
}
