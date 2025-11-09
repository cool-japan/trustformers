/**
 * TrustformeRS TypeScript Client Models
 * 
 * Type definitions for requests, responses, and server data structures.
 * Provides comprehensive TypeScript interfaces for API interactions.
 */

/**
 * Supported inference task types
 */
export enum TaskType {
  TEXT_GENERATION = 'text-generation',
  TEXT_CLASSIFICATION = 'text-classification',
  TOKEN_CLASSIFICATION = 'token-classification',
  QUESTION_ANSWERING = 'question-answering',
  SUMMARIZATION = 'summarization',
  TRANSLATION = 'translation',
  FILL_MASK = 'fill-mask',
  FEATURE_EXTRACTION = 'feature-extraction',
  SENTENCE_SIMILARITY = 'sentence-similarity',
  ZERO_SHOT_CLASSIFICATION = 'zero-shot-classification',
  CONVERSATIONAL = 'conversational',
  IMAGE_CLASSIFICATION = 'image-classification',
  OBJECT_DETECTION = 'object-detection',
  IMAGE_TO_TEXT = 'image-to-text',
  TEXT_TO_IMAGE = 'text-to-image',
  AUDIO_CLASSIFICATION = 'audio-classification',
  AUTOMATIC_SPEECH_RECOGNITION = 'automatic-speech-recognition',
  TEXT_TO_SPEECH = 'text-to-speech',
  CUSTOM = 'custom',
}

/**
 * Supported model architectures
 */
export enum ModelType {
  BERT = 'bert',
  GPT2 = 'gpt2',
  GPT3 = 'gpt3',
  T5 = 't5',
  ROBERTA = 'roberta',
  DISTILBERT = 'distilbert',
  ELECTRA = 'electra',
  ALBERT = 'albert',
  DEBERTA = 'deberta',
  LONGFORMER = 'longformer',
  BART = 'bart',
  PEGASUS = 'pegasus',
  MARIAN = 'marian',
  WAV2VEC2 = 'wav2vec2',
  VIT = 'vit',
  DEIT = 'deit',
  CLIP = 'clip',
  BLIP = 'blip',
  CUSTOM = 'custom',
}

/**
 * Device types for inference
 */
export enum DeviceType {
  CPU = 'cpu',
  GPU = 'gpu',
  TPU = 'tpu',
  AUTO = 'auto',
}

/**
 * Request priority levels
 */
export enum Priority {
  LOW = 'low',
  NORMAL = 'normal',
  HIGH = 'high',
  CRITICAL = 'critical',
}

/**
 * General status values
 */
export enum Status {
  HEALTHY = 'healthy',
  DEGRADED = 'degraded',
  UNHEALTHY = 'unhealthy',
  UNKNOWN = 'unknown',
}

/**
 * Model loading/status states
 */
export enum ModelStatusEnum {
  UNLOADED = 'unloaded',
  LOADING = 'loading',
  LOADED = 'loaded',
  FAILED = 'failed',
  UNLOADING = 'unloading',
}

/**
 * Request for single inference
 */
export interface InferenceRequest {
  /** Input text for inference */
  inputText: string;
  /** Model identifier */
  modelId: string;
  /** Type of task to perform */
  taskType?: TaskType;
  
  // Generation parameters
  /** Maximum output length (1-4096) */
  maxLength?: number;
  /** Minimum output length */
  minLength?: number;
  /** Sampling temperature (0.0-2.0) */
  temperature?: number;
  /** Top-p nucleus sampling (0.0-1.0) */
  topP?: number;
  /** Top-k sampling */
  topK?: number;
  /** Number of beams for beam search (1-20) */
  numBeams?: number;
  /** Whether to use sampling */
  doSample?: boolean;
  /** Repetition penalty (0.0-2.0) */
  repetitionPenalty?: number;
  
  // Processing parameters
  /** Whether to return raw tensors */
  returnTensors?: boolean;
  /** Whether to return attention weights */
  returnAttention?: boolean;
  /** Whether to return hidden states */
  returnHiddenStates?: boolean;
  
  // Request metadata
  /** Optional request identifier */
  requestId?: string;
  /** Request priority */
  priority?: Priority;
  /** Request timeout in seconds (1-300) */
  timeout?: number;
  
  // Advanced parameters
  /** Device to use for inference */
  device?: DeviceType;
  /** Whether to enable result caching */
  enableCaching?: boolean;
  /** Custom model-specific parameters */
  customParams?: Record<string, any>;
}

/**
 * Response from single inference
 */
export interface InferenceResponse {
  /** Request identifier */
  requestId?: string;
  /** Generated output text */
  outputText: string;
  /** Generated tokens */
  outputTokens?: string[];
  
  // Confidence and scores
  /** Overall confidence score (0.0-1.0) */
  confidence?: number;
  /** Per-token confidence scores */
  tokenScores?: number[];
  
  // Model outputs
  /** Raw model logits */
  logits?: number[][];
  /** Attention weights */
  attentionWeights?: number[][][];
  /** Hidden states */
  hiddenStates?: number[][][];
  
  // Metadata
  /** Model used for inference */
  modelId: string;
  /** Task type performed */
  taskType?: TaskType;
  /** Device used for inference */
  deviceUsed?: DeviceType;
  
  // Performance metrics
  /** Inference time in seconds */
  inferenceTime?: number;
  /** Generation speed */
  tokensPerSecond?: number;
  /** Memory used in bytes */
  memoryUsed?: number;
  
  // Generation details
  /** Why generation finished */
  finishReason?: string;
  /** Number of tokens generated */
  numGeneratedTokens?: number;
  
  // Timestamps
  /** Response creation time */
  createdAt?: string;
  
  // Caching info
  /** Whether response came from cache */
  cacheHit?: boolean;
}

/**
 * Request for batch inference
 */
export interface BatchInferenceRequest {
  /** List of input texts (1-100 items) */
  inputs: string[];
  /** Model identifier */
  modelId: string;
  /** Type of task to perform */
  taskType?: TaskType;
  
  // Shared generation parameters
  /** Maximum output length (1-4096) */
  maxLength?: number;
  /** Sampling temperature (0.0-2.0) */
  temperature?: number;
  /** Top-p nucleus sampling (0.0-1.0) */
  topP?: number;
  /** Whether to use sampling */
  doSample?: boolean;
  
  // Batch parameters
  /** Internal batch size for processing (1-32) */
  batchSize?: number;
  /** Whether to return all outputs */
  returnAllOutputs?: boolean;
  
  // Request metadata
  /** Optional request identifier */
  requestId?: string;
  /** Request priority */
  priority?: Priority;
  /** Request timeout in seconds (1-600) */
  timeout?: number;
}

/**
 * Response from batch inference
 */
export interface BatchInferenceResponse {
  /** Request identifier */
  requestId?: string;
  /** Individual inference results */
  results: InferenceResponse[];
  
  // Batch metadata
  /** Total number of inputs processed */
  totalInputs: number;
  /** Number of successful outputs */
  successfulOutputs: number;
  /** Number of failed outputs */
  failedOutputs: number;
  
  // Performance metrics
  /** Total batch processing time */
  totalInferenceTime?: number;
  /** Average time per item */
  averageInferenceTime?: number;
  /** Total tokens generated */
  totalTokensGenerated?: number;
  
  // Timestamps
  /** Response creation time */
  createdAt?: string;
}

/**
 * Individual token in streaming response
 */
export interface StreamingToken {
  /** Generated token */
  token: string;
  /** Decoded text so far */
  text: string;
  /** Whether generation is complete */
  isFinished: boolean;
  
  // Token metadata
  /** Token ID in vocabulary */
  tokenId?: number;
  /** Log probability of token */
  logprob?: number;
  /** Token confidence (0.0-1.0) */
  confidence?: number;
  
  // Position information
  /** Token position in sequence */
  position?: number;
  
  // Timing
  /** When token was generated */
  generatedAt?: string;
  
  // Special tokens
  /** Whether token is special (EOS, etc.) */
  isSpecial?: boolean;
}

/**
 * Information about a model
 */
export interface ModelInfo {
  /** Model identifier */
  modelId: string;
  /** Human-readable model name */
  name: string;
  /** Model description */
  description?: string;
  
  // Model metadata
  /** Model architecture type */
  modelType: ModelType;
  /** Supported task types */
  taskTypes: TaskType[];
  /** Model version */
  version: string;
  
  // Model specifications
  /** Maximum input sequence length */
  maxSequenceLength: number;
  /** Vocabulary size */
  vocabularySize?: number;
  /** Number of parameters */
  numParameters?: number;
  /** Model size in MB */
  modelSizeMb?: number;
  
  // Supported features
  /** Whether model supports streaming */
  supportsStreaming: boolean;
  /** Whether model supports batching */
  supportsBatching: boolean;
  /** Whether model can return attention */
  supportsAttentionWeights: boolean;
  
  // Performance characteristics
  /** Average inference latency */
  averageLatencyMs?: number;
  /** Maximum batch size */
  maxBatchSize?: number;
  
  // Requirements
  /** Required memory in MB */
  requiredMemoryMb?: number;
  /** Recommended device type */
  recommendedDevice?: DeviceType;
  
  // Timestamps
  /** Model creation time */
  createdAt?: string;
  /** Last update time */
  updatedAt?: string;
}

/**
 * Current status of a model
 */
export interface ModelStatus {
  /** Model identifier */
  modelId: string;
  /** Current model status */
  status: ModelStatusEnum;
  
  // Load information
  /** When model was loaded */
  loadedAt?: string;
  /** Time taken to load */
  loadTimeSeconds?: number;
  
  // Resource usage
  /** Memory used by model */
  memoryUsedMb?: number;
  /** Device model is loaded on */
  deviceUsed?: DeviceType;
  
  // Performance metrics
  /** Total requests served */
  totalRequests?: number;
  /** Successful requests */
  successfulRequests?: number;
  /** Failed requests */
  failedRequests?: number;
  /** Average response time */
  averageLatencyMs?: number;
  
  // Health
  /** Last time model was used */
  lastUsedAt?: string;
  /** Error message if status is FAILED */
  errorMessage?: string;
}

/**
 * Server health status
 */
export interface HealthStatus {
  /** Overall health status */
  status: Status;
  /** Health check timestamp */
  timestamp: string;
  
  // Service information
  /** Service name */
  serviceName: string;
  /** Service version */
  version: string;
  /** Service uptime */
  uptimeSeconds: number;
  
  // Resource health
  /** Memory usage (0.0-100.0) */
  memoryUsagePercent?: number;
  /** CPU usage (0.0-100.0) */
  cpuUsagePercent?: number;
  /** GPU usage (0.0-100.0) */
  gpuUsagePercent?: number;
  
  // Request metrics
  /** Total requests served */
  totalRequests?: number;
  /** Current RPS */
  requestsPerSecond?: number;
  /** Average response time */
  averageResponseTimeMs?: number;
  
  // Loaded models
  /** List of loaded model IDs */
  loadedModels?: string[];
  
  // Additional details
  /** Additional health details */
  details?: Record<string, any>;
}

/**
 * Performance metrics for the service
 */
export interface PerformanceMetrics {
  // Request metrics
  /** Total requests processed */
  totalRequests: number;
  /** Successful requests */
  successfulRequests: number;
  /** Failed requests */
  failedRequests: number;
  
  // Timing metrics
  /** Average response time */
  averageResponseTimeMs: number;
  /** 50th percentile response time */
  p50ResponseTimeMs: number;
  /** 95th percentile response time */
  p95ResponseTimeMs: number;
  /** 99th percentile response time */
  p99ResponseTimeMs: number;
  
  // Throughput metrics
  /** Current requests per second */
  requestsPerSecond: number;
  /** Tokens generated per second */
  tokensPerSecond?: number;
  
  // Resource metrics
  /** Current memory usage */
  memoryUsageMb: number;
  /** CPU utilization (0.0-100.0) */
  cpuUsagePercent: number;
  /** GPU utilization (0.0-100.0) */
  gpuUsagePercent?: number;
  
  // Model metrics
  /** Number of loaded models */
  totalModelsLoaded: number;
  /** Memory used by models */
  modelMemoryUsageMb?: number;
  
  // Cache metrics
  /** Cache hit rate (0.0-1.0) */
  cacheHitRate?: number;
  /** Cache size */
  cacheSizeMb?: number;
  
  // Timestamps
  /** When metrics were collected */
  collectedAt: string;
  /** Metrics window start */
  windowStart?: string;
  /** Metrics window end */
  windowEnd?: string;
}

/**
 * Service-level metrics
 */
export interface ServiceMetrics {
  // Service information
  /** Service name */
  serviceName: string;
  /** Service version */
  version: string;
  /** Service instance identifier */
  instanceId: string;
  
  // Performance metrics
  /** Performance metrics */
  performance: PerformanceMetrics;
  
  // Health metrics
  /** Current health status */
  healthStatus: Status;
  /** Last health check time */
  lastHealthCheck: string;
  
  // Model metrics
  /** Status of all models */
  models: ModelStatus[];
  
  // System metrics
  /** Total system memory */
  systemMemoryTotalMb: number;
  /** Available system memory */
  systemMemoryAvailableMb: number;
  /** Number of CPU cores */
  systemCpuCount: number;
  /** Number of GPUs */
  systemGpuCount?: number;
  
  // Network metrics
  /** Active network connections */
  activeConnections: number;
  /** Total bytes sent */
  totalBytesSent: number;
  /** Total bytes received */
  totalBytesReceived: number;
  
  // Collection metadata
  /** When metrics were collected */
  collectedAt: string;
}

// Specialized request types

/**
 * Specialized request for text generation tasks
 */
export interface TextGenerationRequest extends Omit<InferenceRequest, 'taskType' | 'inputText'> {
  /** Fixed to text generation */
  taskType: TaskType.TEXT_GENERATION;
  /** Generation prompt */
  prompt: string;
  
  // Generation-specific parameters
  /** Sequences that stop generation */
  stopSequences?: string[];
  /** Whether to include prompt in output */
  includePrompt?: boolean;
  /** Random seed for reproducible generation */
  seed?: number;
}

/**
 * Specialized request for classification tasks
 */
export interface ClassificationRequest extends Omit<InferenceRequest, 'taskType' | 'inputText'> {
  /** Fixed to classification */
  taskType: TaskType.TEXT_CLASSIFICATION;
  /** Text to classify */
  text: string;
  
  // Classification-specific parameters
  /** Return scores for all classes */
  returnAllScores?: boolean;
  /** Classification threshold (0.0-1.0) */
  threshold?: number;
}

/**
 * Specialized request for question answering tasks
 */
export interface QuestionAnsweringRequest extends Omit<InferenceRequest, 'taskType'> {
  /** Fixed to QA */
  taskType: TaskType.QUESTION_ANSWERING;
  /** Question to answer */
  question: string;
  /** Context for answering */
  context: string;
  
  // QA-specific parameters
  /** Maximum answer length */
  maxAnswerLength?: number;
  /** Minimum answer length */
  minAnswerLength?: number;
  /** Number of answers to return (1-10) */
  topKAnswers?: number;
}

// Error response models

/**
 * Standard error response format
 */
export interface ErrorResponse {
  /** Error message */
  error: string;
  /** Machine-readable error code */
  errorCode?: string;
  /** Error type/category */
  errorType?: string;
  
  // Request context
  /** Request identifier */
  requestId?: string;
  /** Error timestamp */
  timestamp: string;
  
  // Additional details
  /** Additional error details */
  details?: Record<string, any>;
  /** Suggested resolution */
  suggestion?: string;
  
  // Retry information
  /** Whether the request can be retried */
  retryable?: boolean;
  /** Suggested retry delay */
  retryAfterSeconds?: number;
}

/**
 * Validation error details
 */
export interface ValidationError {
  /** Field that failed validation */
  field: string;
  /** Validation error message */
  message: string;
  /** The invalid value */
  invalidValue?: any;
  /** Expected value type */
  expectedType?: string;
}

/**
 * Validation error response with field-specific details
 */
export interface ValidationErrorResponse extends ErrorResponse {
  /** Fixed error type */
  errorType: 'validation_error';
  /** Specific validation errors */
  validationErrors: ValidationError[];
}