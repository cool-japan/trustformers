/**
 * TypeScript type definitions for TrustformeRS C API
 */

import { Pointer } from 'ref-napi';

// Error types
export enum TrustformersError {
  Success = 0,
  NullPointer = 1,
  InvalidParameter = 2,
  OutOfMemory = 3,
  FileNotFound = 4,
  SerializationError = 5,
  DeserializationError = 6,
  RuntimeError = 7,
  UnsupportedOperation = 8,
  NetworkError = 9,
  TimeoutError = 10
}

// Memory usage structure
export interface TrustformersMemoryUsage {
  totalMemoryBytes: number;
  peakMemoryBytes: number;
  allocatedModels: number;
  allocatedTokenizers: number;
  allocatedPipelines: number;
  allocatedTensors: number;
}

// Advanced memory usage structure
export interface TrustformersAdvancedMemoryUsage {
  basic: TrustformersMemoryUsage;
  fragmentationRatio: number;
  avgAllocationSize: number;
  typeUsageJson: string;
  pressureLevel: number;
  allocationRate: number;
  deallocationRate: number;
}

// Build information structure
export interface TrustformersBuildInfo {
  version: string;
  features: string;
  buildDate: string;
  target: string;
}

// Performance metrics structure
export interface TrustformersPerformanceMetrics {
  totalOperations: number;
  avgOperationTimeMs: number;
  minOperationTimeMs: number;
  maxOperationTimeMs: number;
  cacheHitRate: number;
  performanceScore: number;
  numOptimizationHints: number;
  optimizationHintsJson: string;
}

// Optimization configuration structure
export interface TrustformersOptimizationConfig {
  enableTracking: boolean;
  enableCaching: boolean;
  cacheSizeMb: number;
  numThreads: number;
  enableSimd: boolean;
  optimizeBatchSize: boolean;
  memoryOptimizationLevel: number;
}

// Model configuration
export interface ModelConfig {
  modelPath: string;
  tokenizerPath?: string;
  device?: 'cpu' | 'cuda' | 'rocm' | 'auto';
  quantization?: 'fp16' | 'int8' | 'int4' | 'none';
  batchSize?: number;
  maxLength?: number;
  numThreads?: number;
}

// Pipeline configuration
export interface PipelineConfig {
  task: 'text-generation' | 'text-classification' | 'question-answering' | 'conversational';
  model: string | ModelConfig;
  tokenizer?: string;
  device?: 'cpu' | 'cuda' | 'rocm' | 'auto';
  maxLength?: number;
  batchSize?: number;
  streamingMode?: boolean;
}

// Tokenizer configuration
export interface TokenizerConfig {
  tokenizerPath: string;
  addSpecialTokens?: boolean;
  padding?: boolean | 'max_length';
  truncation?: boolean;
  maxLength?: number;
  returnTensors?: 'raw' | 'array';
}

// Generation parameters
export interface GenerationConfig {
  maxLength?: number;
  minLength?: number;
  temperature?: number;
  topK?: number;
  topP?: number;
  repetitionPenalty?: number;
  doSample?: boolean;
  numBeams?: number;
  earlyStopping?: boolean;
  lengthPenalty?: number;
  noRepeatNgramSize?: number;
  stopTokens?: string[];
}

// Text generation result
export interface GenerationResult {
  text: string;
  score?: number;
  finishReason?: 'length' | 'stop_token' | 'eos_token';
  tokens?: string[];
  tokenIds?: number[];
  attentionWeights?: number[][];
}

// Classification result
export interface ClassificationResult {
  label: string;
  score: number;
  logits?: number[];
}

// Question answering result
export interface QuestionAnsweringResult {
  answer: string;
  score: number;
  startPosition: number;
  endPosition: number;
  context?: string;
}

// Tokenization result
export interface TokenizationResult {
  inputIds: number[];
  attentionMask?: number[];
  tokenTypeIds?: number[];
  specialTokensMask?: number[];
  tokens?: string[];
  overflowing?: TokenizationResult[];
}

// Encoding options
export interface EncodeOptions {
  addSpecialTokens?: boolean;
  padding?: boolean | 'max_length' | 'longest';
  truncation?: boolean;
  maxLength?: number;
  stride?: number;
  returnOverflowing?: boolean;
  returnSpecialTokensMask?: boolean;
  returnOffsets?: boolean;
  returnAttentionMask?: boolean;
  returnTensors?: 'list' | 'array';
}

// Decode options
export interface DecodeOptions {
  skipSpecialTokens?: boolean;
  cleanUpTokenizationSpaces?: boolean;
}

// Conversation state
export interface ConversationState {
  conversationId: string;
  turns: ConversationTurn[];
  maxHistory?: number;
  systemPrompt?: string;
}

// Conversation turn
export interface ConversationTurn {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: Date;
  metadata?: Record<string, any>;
}

// Streaming callback types
export type StreamingCallback = (chunk: string, isComplete: boolean) => void;
export type ProgressCallback = (progress: number, message?: string) => void;
export type ErrorCallback = (error: Error) => void;

// Event types for async operations
export interface TrustformersEvents {
  'model-loaded': (modelId: number) => void;
  'tokenizer-loaded': (tokenizerId: number) => void;
  'pipeline-created': (pipelineId: number) => void;
  'generation-start': (requestId: string) => void;
  'generation-chunk': (requestId: string, chunk: string) => void;
  'generation-complete': (requestId: string, result: GenerationResult) => void;
  'error': (error: Error) => void;
  'memory-warning': (usage: TrustformersMemoryUsage) => void;
  'performance-hint': (hint: string) => void;
}

// Resource handles (opaque to JavaScript)
export type ModelHandle = number;
export type TokenizerHandle = number;
export type PipelineHandle = number;
export type TensorHandle = number;

// Utility types
export type BufferLike = Buffer | Uint8Array | ArrayBuffer;
export type StringOrBuffer = string | BufferLike;

// Platform information
export interface PlatformInfo {
  arch: string;
  platform: string;
  features: string[];
  hasGpu: boolean;
  hasCuda: boolean;
  hasRocm: boolean;
  numCores: number;
  totalMemory: number;
}

// Library capabilities
export interface LibraryCapabilities {
  version: string;
  buildInfo: TrustformersBuildInfo;
  supportedFeatures: string[];
  supportedDevices: string[];
  supportedFormats: string[];
  maxBatchSize: number;
  maxSequenceLength: number;
}

// Batch processing types
export interface BatchRequest {
  id: string;
  input: string | string[];
  config?: Partial<GenerationConfig>;
  priority?: number;
}

export interface BatchResult {
  id: string;
  result: GenerationResult | GenerationResult[];
  processingTime: number;
  queueTime: number;
}

// Error context for detailed error reporting
export interface ErrorContext {
  operation: string;
  modelId?: number;
  tokenizerId?: number;
  pipelineId?: number;
  inputLength?: number;
  parameters?: Record<string, any>;
  stackTrace?: string;
  timestamp: Date;
}

// Custom error class
export class TrustformersNativeError extends Error {
  public readonly code: TrustformersError;
  public readonly context?: ErrorContext;

  constructor(message: string, code: TrustformersError, context?: ErrorContext) {
    super(message);
    this.name = 'TrustformersNativeError';
    this.code = code;
    this.context = context;
    
    // Maintain proper stack trace for debugging
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, TrustformersNativeError);
    }
  }

  public toJSON() {
    return {
      name: this.name,
      message: this.message,
      code: this.code,
      context: this.context,
      stack: this.stack
    };
  }
}

// Promise-based async variants
export type AsyncGenerationResult = Promise<GenerationResult>;
export type AsyncClassificationResult = Promise<ClassificationResult>;
export type AsyncQuestionAnsweringResult = Promise<QuestionAnsweringResult>;
export type AsyncTokenizationResult = Promise<TokenizationResult>;
export type AsyncBatchResult = Promise<BatchResult[]>;

// Streaming variants
export interface StreamingOptions {
  onChunk?: StreamingCallback;
  onProgress?: ProgressCallback;
  onError?: ErrorCallback;
  onComplete?: () => void;
}