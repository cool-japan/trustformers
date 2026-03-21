/**
 * TrustformeRS JavaScript API TypeScript Definitions
 */

export interface InitializeOptions {
  wasmPath?: string;
  initPanicHook?: boolean;
}

export interface Tensor {
  data: Float32Array;
  shape: Uint32Array;
  add(other: Tensor): Tensor;
  sub(other: Tensor): Tensor;
  mul(other: Tensor): Tensor;
  matmul(other: Tensor): Tensor;
  transpose(): Tensor;
  reshape(newShape: number[]): Tensor;
  sum(): number;
  mean(): number;
  exp(): Tensor;
  log(): Tensor;
  softmax(axis: number): Tensor;
  gelu(): Tensor;
  relu(): Tensor;
  toString(): string;
  free(): void;
}

export interface ModelConfig {
  architecture: ModelArchitecture;
  vocab_size: number;
  hidden_size: number;
  num_layers: number;
  num_heads: number;
  max_position_embeddings: number;
  intermediate_size: number;
  hidden_dropout_prob: number;
  attention_dropout_prob: number;
  free(): void;
}

export interface Model {
  config: ModelConfig;
  initialized: boolean;
  load_weights(weightsData: Uint8Array): Promise<void>;
  load_from_url(url: string): Promise<void>;
  forward(inputIds: Tensor): Tensor;
  memory_usage_mb(): number;
  free(): void;
}

export interface Tokenizer {
  vocab_size: number;
  load_vocab(vocab: any): void;
  encode(text: string, addSpecialTokens: boolean): Uint32Array;
  decode(tokenIds: Uint32Array, skipSpecialTokens: boolean): string;
  batch_encode(texts: string[], addSpecialTokens: boolean): BatchEncodingOutput;
  set_max_length(maxLength: number): void;
  get_special_token_ids(): Uint32Array;
  free(): void;
}

export interface BatchEncodingOutput {
  len(): number;
  get_sequence(index: number): Uint32Array | undefined;
  free(): void;
}

export interface GenerationConfig {
  max_length: number;
  min_length: number;
  temperature: number;
  top_k: number;
  top_p: number;
  num_beams: number;
  do_sample: boolean;
  early_stopping: boolean;
  repetition_penalty: number;
}

export interface TextGenerationPipeline {
  generate(prompt: string): Promise<string>;
  generate_batch(prompts: string[]): Promise<string[]>;
  set_config(config: GenerationConfig): void;
  free(): void;
}

export interface ClassificationResult {
  label: string;
  score: number;
  all_scores: Float32Array;
  free(): void;
}

export interface TextClassificationPipeline {
  set_labels(labels: string[]): void;
  classify(text: string): Promise<ClassificationResult>;
  classify_batch(texts: string[]): Promise<ClassificationResult[]>;
  free(): void;
}

export interface AnswerResult {
  answer: string;
  start: number;
  end: number;
  score: number;
  free(): void;
}

export interface QuestionAnsweringPipeline {
  answer(question: string, context: string): Promise<AnswerResult>;
  free(): void;
}

export interface MemoryStats {
  used: number;
  limit: number;
  used_mb: number;
  limit_mb: number;
  free(): void;
}

export interface Timer {
  elapsed(): number;
  log_elapsed(): void;
  free(): void;
}

export enum ModelArchitecture {
  BERT = 0,
  GPT2 = 1,
  T5 = 2,
  LLAMA = 3,
  MISTRAL = 4
}

export enum PipelineType {
  TEXT_GENERATION = 0,
  TEXT_CLASSIFICATION = 1,
  TOKEN_CLASSIFICATION = 2,
  QUESTION_ANSWERING = 3,
  SUMMARIZATION = 4,
  TRANSLATION = 5
}

export enum TokenizerType {
  WORDPIECE = 0,
  BPE = 1,
  SENTENCEPIECE = 2
}

// Main API functions
export function initialize(options?: InitializeOptions): Promise<void>;
export function initializeEnhanced(options?: InitializeOptions): Promise<void>;
export function tensor(data: number[], shape: number[]): Tensor;
export function zeros(shape: number[]): Tensor;
export function ones(shape: number[]): Tensor;
export function eye(size: number): Tensor;
export function randn(shape: number[], options?: { seed?: number; distribution?: string }): Promise<Tensor>;
export function create_advanced_tensor(shape: number[], distribution_config: any, seed?: number): Promise<Tensor>;
export function create_bayesian_tensor(shape: number[], prior_config: any, seed?: number): Promise<Tensor>;
export function analyze_tensor(tensor: Tensor, seed?: number, options?: any): Promise<any>;
export function createModelConfig(modelType: 'bert_base' | 'gpt2_base' | 't5_small' | 'llama_7b' | 'mistral_7b'): ModelConfig;
export function createModel(configOrType: ModelConfig | string): Model;
export function createTokenizer(type: 'wordpiece' | 'bpe' | 'sentencepiece', vocab?: any): Tokenizer;
export function getRawModule(): any;

// Pipeline factory
export class Pipeline {
  static textGeneration(model: Model, tokenizer: Tokenizer, config?: Partial<GenerationConfig>): TextGenerationPipeline;
  static textClassification(model: Model, tokenizer: Tokenizer, labels?: string[]): TextClassificationPipeline;
  static questionAnswering(model: Model, tokenizer: Tokenizer): QuestionAnsweringPipeline;
  static fromPretrained(task: string, modelName: string): Promise<any>;
}

// Memory utilities
export const memory: {
  getStats(): MemoryStats;
  getUsage(): number;
};

// Utility functions
export const utils: {
  log(message: string): void;
  logError(message: string): void;
  logWarning(message: string): void;
  timer(name: string): Timer;
  version(): string;
  features(): any[];
};

// Comprehensive tensor operations
export const tensor_ops: {
  // Basic arithmetic operations
  add(a: Tensor, b: Tensor): Tensor;
  sub(a: Tensor, b: Tensor): Tensor;
  mul(a: Tensor, b: Tensor): Tensor;
  div(a: Tensor, b: Tensor): Tensor;
  matmul(a: Tensor, b: Tensor): Tensor;
  
  // Scalar operations
  addScalar(tensor: Tensor, scalar: number): Tensor;
  mulScalar(tensor: Tensor, scalar: number): Tensor;
  
  // Shape operations
  reshape(tensor: Tensor, newShape: number[]): Tensor;
  transpose(tensor: Tensor, dims?: number[]): Tensor;
  squeeze(tensor: Tensor, dim?: number): Tensor;
  unsqueeze(tensor: Tensor, dim: number): Tensor;
  
  // Slicing and indexing
  slice(tensor: Tensor, start: number[], end: number[], step?: number): Tensor;
  indexSelect(tensor: Tensor, dim: number, indices: number[]): Tensor;
  
  // Reduction operations
  sum(tensor: Tensor, dim?: number, keepDim?: boolean): Tensor;
  mean(tensor: Tensor, dim?: number, keepDim?: boolean): Tensor;
  max(tensor: Tensor, dim?: number, keepDim?: boolean): Tensor;
  min(tensor: Tensor, dim?: number, keepDim?: boolean): Tensor;
  
  // Comparison operations
  eq(a: Tensor, b: Tensor): Tensor;
  gt(a: Tensor, b: Tensor): Tensor;
  lt(a: Tensor, b: Tensor): Tensor;
  
  // Concatenation and stacking
  cat(tensors: Tensor[], dim?: number): Tensor;
  stack(tensors: Tensor[], dim?: number): Tensor;
  
  // Mathematical functions
  exp(tensor: Tensor): Tensor;
  log(tensor: Tensor): Tensor;
  sqrt(tensor: Tensor): Tensor;
  pow(tensor: Tensor, exponent: number): Tensor;
  abs(tensor: Tensor): Tensor;
  
  // Normalization
  layerNorm(tensor: Tensor, normalized_shape: number[], eps?: number): Tensor;
  batchNorm(tensor: Tensor, running_mean: Tensor, running_var: Tensor, weight?: Tensor, bias?: Tensor, eps?: number): Tensor;
};

// Activation functions
export const activations: {
  relu(tensor: Tensor): Tensor;
  leakyRelu(tensor: Tensor, negative_slope?: number): Tensor;
  gelu(tensor: Tensor): Tensor;
  swish(tensor: Tensor): Tensor;
  sigmoid(tensor: Tensor): Tensor;
  tanh(tensor: Tensor): Tensor;
  softmax(tensor: Tensor, dim?: number): Tensor;
  logSoftmax(tensor: Tensor, dim?: number): Tensor;
};

// Streaming utilities
export const streaming: {
  textGeneration(model: Model, tokenizer: Tokenizer, config?: Partial<GenerationConfig>): AsyncGenerator<string, void, unknown>;
  tokenize(tokenizer: Tokenizer, text: string): AsyncGenerator<number, void, unknown>;
};

// Performance monitoring utilities
export const performance: {
  getReport(): {
    timestamp: string;
    capabilities: any;
    memory: any;
    profiling: any;
    webgl: any;
  };
  startSession(name: string, metadata?: any): any;
  endSession(): any;
  profile<T>(name: string, fn: () => Promise<T>, metadata?: any): Promise<T>;
  getMemoryUsage(): any;
  cleanup(): void;
};

// Async utilities
export const async_utils: {
  processBatch<T, R>(tensors: T[], processFunc: (item: T) => Promise<R>, batchSize?: number): Promise<R[]>;
  runInference(model: Model, inputs: Tensor | Tensor[], options?: { autoCleanup?: boolean; timeout?: number }): Promise<Tensor>;
};

// Tensor creation utilities
export const tensor_utils: {
  fromTypedArray(data: TypedArray, shape: number[]): Tensor;
  fromNestedArray(array: any[]): Tensor;
  random(shape: number[], distribution?: 'normal' | 'uniform' | 'binomial', params?: { mean?: number; std?: number; low?: number; high?: number; n?: number; p?: number }): Tensor;
};

// WebGPU support
export const webgpu: {
  isAvailable(): boolean;
  getStatus(): string;
  createOps(): any;
  getDeviceInfo(): Promise<any>;
};

// Additional tensor interface extensions
export interface Tensor {
  data: Float32Array;
  shape: Uint32Array;
  dtype: string;
  device: string;
  
  // Operations
  add(other: Tensor): Tensor;
  sub(other: Tensor): Tensor;
  mul(other: Tensor): Tensor;
  div(other: Tensor): Tensor;
  matmul(other: Tensor): Tensor;
  
  // Shape operations
  transpose(dims?: number[]): Tensor;
  reshape(newShape: number[]): Tensor;
  squeeze(dim?: number): Tensor;
  unsqueeze(dim: number): Tensor;
  
  // Reductions
  sum(dim?: number, keepDim?: boolean): Tensor;
  mean(dim?: number, keepDim?: boolean): Tensor;
  max(dim?: number, keepDim?: boolean): Tensor;
  min(dim?: number, keepDim?: boolean): Tensor;
  
  // Math functions
  exp(): Tensor;
  log(): Tensor;
  sqrt(): Tensor;
  pow(exponent: number): Tensor;
  abs(): Tensor;
  
  // Activations
  relu(): Tensor;
  gelu(): Tensor;
  sigmoid(): Tensor;
  tanh(): Tensor;
  softmax(dim?: number): Tensor;
  
  // Utilities
  clone(): Tensor;
  detach(): Tensor;
  requires_grad_(requires_grad: boolean): Tensor;
  backward(gradient?: Tensor): void;
  
  // Conversion
  toArray(): number[];
  toNestedArray(): any[];
  toString(): string;
  
  // Memory management
  free(): void;
}

// Extended Model interface
export interface Model {
  config: ModelConfig;
  initialized: boolean;
  device: string;
  
  // Loading methods
  load_weights(weightsData: Uint8Array): Promise<void>;
  load_from_url(url: string): Promise<void>;
  load_from_hub(model_id: string): Promise<void>;
  
  // Inference methods
  forward(inputIds: Tensor): Tensor;
  forward_async(inputIds: Tensor): Promise<Tensor>;
  forward_batch(inputIds: Tensor[]): Tensor[];
  
  // Generation methods
  generate(inputs: Tensor, config?: GenerationConfig): Promise<Tensor>;
  generate_stream(inputs: Tensor, config?: GenerationConfig): AsyncGenerator<Tensor, void, unknown>;
  
  // Utilities
  memory_usage_mb(): number;
  get_parameters(): { [key: string]: Tensor };
  set_device(device: string): void;
  eval(): void;
  train(): void;
  
  // Memory management
  free(): void;
}

// Extended Tokenizer interface
export interface Tokenizer {
  vocab_size: number;
  model_max_length: number;
  
  // Vocabulary operations
  load_vocab(vocab: any): void;
  get_vocab(): { [key: string]: number };
  
  // Encoding operations
  encode(text: string, addSpecialTokens?: boolean): Uint32Array;
  encode_batch(texts: string[], addSpecialTokens?: boolean): BatchEncodingOutput;
  decode(tokenIds: Uint32Array, skipSpecialTokens?: boolean): string;
  decode_batch(tokenIds: Uint32Array[], skipSpecialTokens?: boolean): string[];
  
  // Special tokens
  get_special_token_ids(): Uint32Array;
  get_special_tokens_dict(): { [key: string]: number };
  
  // Configuration
  set_max_length(maxLength: number): void;
  set_padding(padding: boolean): void;
  set_truncation(truncation: boolean): void;
  
  // Utilities
  token_to_id(token: string): number | null;
  id_to_token(id: number): string | null;
  
  // Memory management
  free(): void;
}

// Enhanced modules and utilities
export const enhanced_tensor_ops: any;
export const enhanced_tensor_utils: any;
export const enhanced_inference: any;
export const devTools: any;
export const nodejs: any;

// Core classes
export class WebGLBackend {
  initialize(): Promise<void>;
  getInfo(): any;
  dispose(): void;
}

export class TensorMemoryPool {
  allocate(size: number): any;
  deallocate(ptr: any): void;
  getStats(): any;
}

export class MemoryManager {
  getStats(): any;
  cleanup(): void;
}

export class PerformanceProfiler {
  startSession(name: string, metadata?: any): any;
  endSession(): any;
  profileOperation<T>(name: string, fn: () => Promise<T>, metadata?: any): Promise<T>;
  generateReport(): any;
}

export class ZeroCopyTensorView {
  constructor(buffer: ArrayBuffer, shape: number[], dtype: string);
  getData(): TypedArray;
  getShape(): number[];
}

// Factory functions
export function createWebGLBackend(): WebGLBackend;
export function getMemoryManager(): MemoryManager;
export function getProfiler(): PerformanceProfiler;
export function createZeroCopyTensor(data: TypedArray, shape: number[]): ZeroCopyTensorView;

// Debug and visualization utilities
export class DebugUtilities {
  static log(message: string): void;
  static inspect(tensor: Tensor): any;
}

export class TensorInspector {
  static inspect(tensor: Tensor): any;
  static visualize(tensor: Tensor, options?: any): any;
}

export class ModelVisualizer {
  static visualize(model: Model, options?: any): any;
  static exportGraph(model: Model): any;
}

export class ErrorDiagnostics {
  static diagnose(error: Error): any;
  static getErrorCode(error: Error): string;
}

// Image processing utilities
export class ImageClassifier {
  constructor(model: Model, preprocessor: ImagePreprocessor);
  classify(image: ImageData | HTMLImageElement): Promise<any>;
}

export class ImagePreprocessor {
  constructor(config?: any);
  preprocess(image: ImageData | HTMLImageElement): Tensor;
}

// Re-export enums for convenience
export { ModelArchitecture, PipelineType, TokenizerType };