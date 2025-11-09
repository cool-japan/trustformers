/**
 * TrustformeRS TypeScript Client Constants
 * 
 * Defines constants used throughout the TrustformeRS client library.
 */

import { TaskType, ModelType } from './models';

/**
 * API version
 */
export const API_VERSION = 'v1';

/**
 * Default timeout for requests in milliseconds
 */
export const DEFAULT_TIMEOUT = 30000;

/**
 * Default maximum number of retries
 */
export const DEFAULT_MAX_RETRIES = 3;

/**
 * Default retry delay in milliseconds
 */
export const DEFAULT_RETRY_DELAY = 1000;

/**
 * Maximum retry delay in milliseconds
 */
export const MAX_RETRY_DELAY = 60000;

/**
 * Default base URL for local development
 */
export const DEFAULT_BASE_URL = 'http://localhost:8080';

/**
 * Supported model types
 */
export const SUPPORTED_MODEL_TYPES = Object.values(ModelType);

/**
 * Supported task types
 */
export const SUPPORTED_TASKS = Object.values(TaskType);

/**
 * HTTP status codes
 */
export const HTTP_STATUS = {
  OK: 200,
  CREATED: 201,
  ACCEPTED: 202,
  NO_CONTENT: 204,
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  METHOD_NOT_ALLOWED: 405,
  CONFLICT: 409,
  UNPROCESSABLE_ENTITY: 422,
  TOO_MANY_REQUESTS: 429,
  INTERNAL_SERVER_ERROR: 500,
  BAD_GATEWAY: 502,
  SERVICE_UNAVAILABLE: 503,
  GATEWAY_TIMEOUT: 504,
} as const;

/**
 * Content types
 */
export const CONTENT_TYPES = {
  JSON: 'application/json',
  FORM_DATA: 'multipart/form-data',
  FORM_URLENCODED: 'application/x-www-form-urlencoded',
  TEXT_PLAIN: 'text/plain',
  TEXT_HTML: 'text/html',
  OCTET_STREAM: 'application/octet-stream',
} as const;

/**
 * Common HTTP headers
 */
export const HEADERS = {
  AUTHORIZATION: 'Authorization',
  CONTENT_TYPE: 'Content-Type',
  ACCEPT: 'Accept',
  USER_AGENT: 'User-Agent',
  X_API_KEY: 'X-API-Key',
  X_REQUEST_ID: 'X-Request-ID',
  X_CLIENT_VERSION: 'X-Client-Version',
  CACHE_CONTROL: 'Cache-Control',
  IF_NONE_MATCH: 'If-None-Match',
  ETAG: 'ETag',
} as const;

/**
 * Authentication types
 */
export const AUTH_TYPES = {
  API_KEY: 'api_key',
  JWT: 'jwt',
  OAUTH2: 'oauth2',
  BASIC: 'basic',
  CUSTOM: 'custom',
  NONE: 'none',
} as const;

/**
 * API endpoints
 */
export const ENDPOINTS = {
  HEALTH: '/health',
  HEALTH_DETAILED: '/health/detailed',
  HEALTH_READINESS: '/health/readiness',
  HEALTH_LIVENESS: '/health/liveness',
  INFER: '/v1/infer',
  BATCH_INFER: '/v1/batch_infer',
  STREAM_INFER: '/v1/stream_infer',
  MODELS: '/v1/models',
  MODEL_INFO: (modelId: string) => `/v1/models/${modelId}`,
  MODEL_STATUS: (modelId: string) => `/v1/models/${modelId}/status`,
  MODEL_LOAD: (modelId: string) => `/v1/models/${modelId}/load`,
  MODEL_UNLOAD: (modelId: string) => `/v1/models/${modelId}/unload`,
  METRICS: '/metrics',
  PERFORMANCE_METRICS: '/v1/metrics/performance',
  STATS: '/stats',
} as const;

/**
 * Error codes
 */
export const ERROR_CODES = {
  // Client errors
  VALIDATION_ERROR: 'VALIDATION_ERROR',
  AUTHENTICATION_ERROR: 'AUTHENTICATION_ERROR',
  AUTHORIZATION_ERROR: 'AUTHORIZATION_ERROR',
  NOT_FOUND: 'NOT_FOUND',
  RATE_LIMITED: 'RATE_LIMITED',
  
  // Server errors
  INTERNAL_ERROR: 'INTERNAL_ERROR',
  SERVICE_UNAVAILABLE: 'SERVICE_UNAVAILABLE',
  TIMEOUT: 'TIMEOUT',
  CONNECTION_ERROR: 'CONNECTION_ERROR',
  
  // Model errors
  MODEL_NOT_FOUND: 'MODEL_NOT_FOUND',
  MODEL_LOAD_ERROR: 'MODEL_LOAD_ERROR',
  MODEL_UNAVAILABLE: 'MODEL_UNAVAILABLE',
  
  // Resource errors
  INSUFFICIENT_MEMORY: 'INSUFFICIENT_MEMORY',
  INSUFFICIENT_GPU: 'INSUFFICIENT_GPU',
  RESOURCE_EXHAUSTED: 'RESOURCE_EXHAUSTED',
  
  // Inference errors
  INFERENCE_ERROR: 'INFERENCE_ERROR',
  INVALID_INPUT: 'INVALID_INPUT',
  OUTPUT_TOO_LONG: 'OUTPUT_TOO_LONG',
  
  // Streaming errors
  STREAM_ERROR: 'STREAM_ERROR',
  STREAM_INTERRUPTED: 'STREAM_INTERRUPTED',
  
  // Batch errors
  BATCH_ERROR: 'BATCH_ERROR',
  PARTIAL_BATCH_FAILURE: 'PARTIAL_BATCH_FAILURE',
} as const;

/**
 * Event types for event listeners
 */
export const EVENT_TYPES = {
  REQUEST_START: 'request_start',
  REQUEST_END: 'request_end',
  REQUEST_ERROR: 'request_error',
  STREAM_START: 'stream_start',
  STREAM_TOKEN: 'stream_token',
  STREAM_END: 'stream_end',
  STREAM_ERROR: 'stream_error',
  MODEL_LOADED: 'model_loaded',
  MODEL_UNLOADED: 'model_unloaded',
  AUTHENTICATION_REFRESH: 'auth_refresh',
  RATE_LIMITED: 'rate_limited',
  CONNECTION_ERROR: 'connection_error',
} as const;

/**
 * Default configuration values
 */
export const DEFAULTS = {
  MAX_SEQUENCE_LENGTH: 512,
  MAX_OUTPUT_LENGTH: 100,
  TEMPERATURE: 1.0,
  TOP_P: 0.9,
  TOP_K: 50,
  NUM_BEAMS: 1,
  REPETITION_PENALTY: 1.0,
  BATCH_SIZE: 8,
  MAX_BATCH_SIZE: 32,
  CONNECTION_POOL_SIZE: 100,
  KEEPALIVE_CONNECTIONS: 20,
  REQUEST_COMPRESSION_THRESHOLD: 1024, // bytes
} as const;

/**
 * Model parameter limits
 */
export const LIMITS = {
  MIN_TEMPERATURE: 0.0,
  MAX_TEMPERATURE: 2.0,
  MIN_TOP_P: 0.0,
  MAX_TOP_P: 1.0,
  MIN_TOP_K: 1,
  MAX_TOP_K: 1000,
  MIN_NUM_BEAMS: 1,
  MAX_NUM_BEAMS: 20,
  MIN_REPETITION_PENALTY: 0.0,
  MAX_REPETITION_PENALTY: 2.0,
  MAX_INPUT_LENGTH: 8192,
  MAX_OUTPUT_LENGTH: 4096,
  MIN_OUTPUT_LENGTH: 1,
  MAX_BATCH_SIZE: 100,
  MAX_SEQUENCE_LENGTH: 16384,
} as const;

/**
 * Cache control values
 */
export const CACHE = {
  NO_CACHE: 'no-cache',
  NO_STORE: 'no-store',
  MUST_REVALIDATE: 'must-revalidate',
  PUBLIC: 'public',
  PRIVATE: 'private',
  MAX_AGE: (seconds: number) => `max-age=${seconds}`,
} as const;

/**
 * User agent patterns
 */
export const USER_AGENTS = {
  DEFAULT: 'trustformers-client-typescript/0.1.0',
  BROWSER: (version: string) => `trustformers-client-browser/${version}`,
  NODE: (version: string, nodeVersion: string) => `trustformers-client-node/${version} (Node.js ${nodeVersion})`,
  MOBILE: (version: string, platform: string) => `trustformers-client-mobile/${version} (${platform})`,
} as const;

/**
 * Streaming constants
 */
export const STREAMING = {
  SSE_PREFIX: 'data: ',
  SSE_DONE: '[DONE]',
  CHUNK_SEPARATOR: '\n',
  RECONNECT_INTERVAL: 3000, // milliseconds
  MAX_RECONNECT_ATTEMPTS: 5,
  HEARTBEAT_INTERVAL: 30000, // milliseconds
} as const;

/**
 * Performance monitoring constants
 */
export const PERFORMANCE = {
  SLOW_REQUEST_THRESHOLD: 5000, // milliseconds
  MEMORY_WARNING_THRESHOLD: 0.8, // 80% of available memory
  CPU_WARNING_THRESHOLD: 0.9, // 90% CPU usage
  ERROR_RATE_WARNING_THRESHOLD: 0.1, // 10% error rate
  METRICS_COLLECTION_INTERVAL: 60000, // 1 minute
  METRICS_RETENTION_PERIOD: 3600000, // 1 hour
} as const;

/**
 * Validation patterns
 */
export const PATTERNS = {
  MODEL_ID: /^[a-zA-Z0-9_-]+$/,
  REQUEST_ID: /^[a-zA-Z0-9_-]{8,64}$/,
  API_KEY: /^[a-zA-Z0-9_-]{16,128}$/,
  JWT_TOKEN: /^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$/,
  URL: /^https?:\/\/.+/,
  SEMANTIC_VERSION: /^\d+\.\d+\.\d+$/,
} as const;

/**
 * Environment detection
 */
export const ENVIRONMENT = {
  IS_BROWSER: typeof window !== 'undefined',
  IS_NODE: typeof process !== 'undefined' && process.versions?.node,
  IS_WORKER: typeof importScripts === 'function',
  IS_SERVICE_WORKER: typeof ServiceWorkerGlobalScope !== 'undefined',
  SUPPORTS_FETCH: typeof fetch !== 'undefined',
  SUPPORTS_WEBSOCKET: typeof WebSocket !== 'undefined',
  SUPPORTS_EVENT_SOURCE: typeof EventSource !== 'undefined',
  SUPPORTS_STREAMS: typeof ReadableStream !== 'undefined',
  SUPPORTS_ABORT_CONTROLLER: typeof AbortController !== 'undefined',
} as const;

/**
 * File size constants
 */
export const FILE_SIZES = {
  KB: 1024,
  MB: 1024 * 1024,
  GB: 1024 * 1024 * 1024,
  MAX_REQUEST_SIZE: 10 * 1024 * 1024, // 10MB
  MAX_RESPONSE_SIZE: 100 * 1024 * 1024, // 100MB
} as const;

/**
 * Time constants in milliseconds
 */
export const TIME = {
  SECOND: 1000,
  MINUTE: 60 * 1000,
  HOUR: 60 * 60 * 1000,
  DAY: 24 * 60 * 60 * 1000,
  WEEK: 7 * 24 * 60 * 60 * 1000,
} as const;

/**
 * Priority levels for requests
 */
export const PRIORITY_LEVELS = {
  LOW: 0,
  NORMAL: 1,
  HIGH: 2,
  CRITICAL: 3,
} as const;

/**
 * Feature flags
 */
export const FEATURES = {
  CACHING: 'caching',
  METRICS: 'metrics',
  LOGGING: 'logging',
  COMPRESSION: 'compression',
  STREAMING: 'streaming',
  BATCHING: 'batching',
  RETRY: 'retry',
  CIRCUIT_BREAKER: 'circuit_breaker',
  RATE_LIMITING: 'rate_limiting',
  AUTHENTICATION: 'authentication',
} as const;

/**
 * Library metadata
 */
export const LIBRARY = {
  NAME: 'trustformers-client-typescript',
  VERSION: '0.1.0',
  DESCRIPTION: 'TypeScript client library for TrustformeRS serving infrastructure',
  HOMEPAGE: 'https://github.com/trustformers/trustformers-serve',
  LICENSE: 'MIT',
  AUTHOR: 'TrustformeRS Team',
} as const;