/**
 * TrustformeRS TypeScript Client Types
 * 
 * Additional type definitions and utility types for the TrustformeRS client library.
 */

import type { 
  InferenceRequest, 
  InferenceResponse, 
  BatchInferenceRequest, 
  BatchInferenceResponse,
  StreamingToken,
  ModelInfo,
  ModelStatus,
  HealthStatus,
  PerformanceMetrics,
  ServiceMetrics,
} from './models';
import type { AuthConfig } from './auth';
import type { RetryConfig } from './retry';

/**
 * Generic API response wrapper
 */
export interface ApiResponse<T> {
  data: T;
  status: number;
  statusText: string;
  headers: Record<string, string>;
  requestId?: string;
  timestamp: string;
}

/**
 * Paginated response interface
 */
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasNext: boolean;
  hasPrevious: boolean;
  nextPage?: number;
  previousPage?: number;
}

/**
 * Request options for client methods
 */
export interface RequestOptions {
  /** Request timeout in milliseconds */
  timeout?: number;
  /** Number of retries for this specific request */
  retries?: number;
  /** Additional headers for this request */
  headers?: Record<string, string>;
  /** AbortSignal for request cancellation */
  signal?: AbortSignal;
  /** Whether to enable caching for this request */
  cache?: boolean;
  /** Cache TTL in milliseconds */
  cacheTtl?: number;
  /** Request priority */
  priority?: 'low' | 'normal' | 'high' | 'critical';
  /** Request metadata */
  metadata?: Record<string, any>;
}

/**
 * Streaming options
 */
export interface StreamingOptions extends RequestOptions {
  /** Whether to reconnect automatically on connection loss */
  autoReconnect?: boolean;
  /** Maximum number of reconnection attempts */
  maxReconnectAttempts?: number;
  /** Delay between reconnection attempts in milliseconds */
  reconnectDelay?: number;
  /** Whether to enable heartbeat */
  enableHeartbeat?: boolean;
  /** Heartbeat interval in milliseconds */
  heartbeatInterval?: number;
}

/**
 * Batch processing options
 */
export interface BatchOptions extends RequestOptions {
  /** Batch size for processing */
  batchSize?: number;
  /** Whether to fail fast on first error */
  failFast?: boolean;
  /** Maximum parallel batches */
  maxParallelBatches?: number;
  /** Whether to return partial results on failure */
  returnPartialResults?: boolean;
}

/**
 * Model loading options
 */
export interface ModelLoadOptions {
  /** Device to load model on */
  device?: 'cpu' | 'gpu' | 'auto';
  /** Model configuration overrides */
  config?: Record<string, any>;
  /** Whether to load model in background */
  background?: boolean;
  /** Timeout for model loading */
  timeout?: number;
  /** Memory limit for model in MB */
  memoryLimit?: number;
}

/**
 * Model unloading options
 */
export interface ModelUnloadOptions {
  /** Whether to force unload even if model is in use */
  force?: boolean;
  /** Timeout for model unloading */
  timeout?: number;
  /** Whether to cleanup resources immediately */
  cleanup?: boolean;
}

/**
 * Health check options
 */
export interface HealthCheckOptions {
  /** Whether to include detailed health information */
  detailed?: boolean;
  /** Components to check */
  components?: string[];
  /** Timeout for health check */
  timeout?: number;
}

/**
 * Metrics collection options
 */
export interface MetricsOptions {
  /** Time window for metrics */
  window?: 'last_minute' | 'last_hour' | 'last_day' | 'all_time';
  /** Specific metrics to include */
  include?: string[];
  /** Specific metrics to exclude */
  exclude?: string[];
  /** Whether to include system metrics */
  includeSystem?: boolean;
  /** Whether to include model metrics */
  includeModels?: boolean;
}

/**
 * Event listener types
 */
export type EventListener<T = any> = (event: T) => void;

/**
 * Event types
 */
export interface Events {
  'request:start': { method: string; url: string; requestId: string };
  'request:end': { method: string; url: string; requestId: string; duration: number; status: number };
  'request:error': { method: string; url: string; requestId: string; error: Error };
  'stream:start': { requestId: string };
  'stream:token': { requestId: string; token: StreamingToken };
  'stream:end': { requestId: string };
  'stream:error': { requestId: string; error: Error };
  'model:loaded': { modelId: string; status: ModelStatus };
  'model:unloaded': { modelId: string };
  'auth:refresh': { type: string; success: boolean };
  'connection:error': { error: Error; attempt: number };
  'rate:limited': { retryAfter: number };
}

/**
 * Event emitter interface
 */
export interface EventEmitter {
  on<K extends keyof Events>(event: K, listener: EventListener<Events[K]>): void;
  off<K extends keyof Events>(event: K, listener: EventListener<Events[K]>): void;
  emit<K extends keyof Events>(event: K, data: Events[K]): void;
  once<K extends keyof Events>(event: K, listener: EventListener<Events[K]>): void;
  removeAllListeners(event?: keyof Events): void;
}

/**
 * Middleware interface
 */
export interface Middleware {
  name: string;
  request?: (config: RequestConfig) => Promise<RequestConfig> | RequestConfig;
  response?: (response: ApiResponse<any>) => Promise<ApiResponse<any>> | ApiResponse<any>;
  error?: (error: Error, config: RequestConfig) => Promise<Error> | Error;
}

/**
 * Request configuration
 */
export interface RequestConfig {
  method: string;
  url: string;
  headers: Record<string, string>;
  data?: any;
  params?: Record<string, any>;
  timeout: number;
  retries: number;
  auth?: AuthConfig;
  metadata?: Record<string, any>;
}

/**
 * Client statistics
 */
export interface ClientStats {
  requests: {
    total: number;
    successful: number;
    failed: number;
    pending: number;
  };
  models: {
    loaded: string[];
    loading: string[];
    failed: string[];
  };
  performance: {
    averageResponseTime: number;
    slowestRequest: number;
    fastestRequest: number;
    requestsPerSecond: number;
  };
  errors: {
    total: number;
    byType: Record<string, number>;
    recent: Array<{ error: string; timestamp: number }>;
  };
  uptime: number;
  memoryUsage?: {
    used: number;
    total: number;
    percentage: number;
  };
}

/**
 * Cache entry
 */
export interface CacheEntry<T> {
  data: T;
  timestamp: number;
  ttl: number;
  key: string;
  size: number;
  hits: number;
}

/**
 * Cache interface
 */
export interface Cache {
  get<T>(key: string): Promise<T | null>;
  set<T>(key: string, value: T, ttl?: number): Promise<void>;
  delete(key: string): Promise<boolean>;
  clear(): Promise<void>;
  has(key: string): Promise<boolean>;
  size(): Promise<number>;
  keys(): Promise<string[]>;
  stats(): Promise<{
    size: number;
    hits: number;
    misses: number;
    hitRate: number;
  }>;
}

/**
 * Rate limiter interface
 */
export interface RateLimiter {
  allowRequest(key: string): Promise<boolean>;
  getRemainingRequests(key: string): Promise<number>;
  getResetTime(key: string): Promise<number>;
  reset(key: string): Promise<void>;
}

/**
 * Connection pool interface
 */
export interface ConnectionPool {
  acquire(): Promise<Connection>;
  release(connection: Connection): void;
  destroy(connection: Connection): void;
  size(): number;
  available(): number;
  pending(): number;
  stats(): ConnectionPoolStats;
}

/**
 * Connection interface
 */
export interface Connection {
  id: string;
  isActive: boolean;
  createdAt: number;
  lastUsed: number;
  requestCount: number;
}

/**
 * Connection pool statistics
 */
export interface ConnectionPoolStats {
  total: number;
  active: number;
  idle: number;
  pending: number;
  created: number;
  destroyed: number;
  errors: number;
}

/**
 * Circuit breaker state
 */
export type CircuitBreakerState = 'closed' | 'open' | 'half-open';

/**
 * Circuit breaker interface
 */
export interface CircuitBreaker {
  execute<T>(fn: () => Promise<T>): Promise<T>;
  getState(): CircuitBreakerState;
  getFailureCount(): number;
  reset(): void;
  forceOpen(): void;
  forceClose(): void;
}

/**
 * Webhook configuration
 */
export interface WebhookConfig {
  url: string;
  events: (keyof Events)[];
  secret?: string;
  headers?: Record<string, string>;
  retries?: number;
  timeout?: number;
}

/**
 * Plugin interface
 */
export interface Plugin {
  name: string;
  version: string;
  initialize?(client: any): Promise<void> | void;
  destroy?(): Promise<void> | void;
}

/**
 * Configuration validation result
 */
export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
}

/**
 * Validation error
 */
export interface ValidationError {
  field: string;
  message: string;
  code: string;
  value?: any;
}

/**
 * Validation warning
 */
export interface ValidationWarning {
  field: string;
  message: string;
  code: string;
  value?: any;
}

/**
 * Utility types
 */

/**
 * Make specific properties optional
 */
export type PartialBy<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

/**
 * Make specific properties required
 */
export type RequiredBy<T, K extends keyof T> = T & Required<Pick<T, K>>;

/**
 * Extract function parameters
 */
export type Parameters<T extends (...args: any) => any> = T extends (...args: infer P) => any ? P : never;

/**
 * Extract function return type
 */
export type ReturnType<T extends (...args: any) => any> = T extends (...args: any) => infer R ? R : any;

/**
 * Async function type
 */
export type AsyncFunction<T extends any[], R> = (...args: T) => Promise<R>;

/**
 * Event handler type
 */
export type EventHandler<T> = (event: T) => void | Promise<void>;

/**
 * Callback function type
 */
export type Callback<T> = (error: Error | null, result?: T) => void;

/**
 * Constructor type
 */
export type Constructor<T = {}> = new (...args: any[]) => T;

/**
 * JSON serializable types
 */
export type JSONValue = string | number | boolean | null | JSONObject | JSONArray;
export interface JSONObject { [key: string]: JSONValue; }
export interface JSONArray extends Array<JSONValue> { }

/**
 * Deep partial type
 */
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

/**
 * Deep readonly type
 */
export type DeepReadonly<T> = {
  readonly [P in keyof T]: T[P] extends object ? DeepReadonly<T[P]> : T[P];
};

/**
 * Mutable type (opposite of readonly)
 */
export type Mutable<T> = {
  -readonly [P in keyof T]: T[P];
};

/**
 * Non-nullable type
 */
export type NonNullable<T> = T extends null | undefined ? never : T;

/**
 * Extract keys of type T that have values of type U
 */
export type KeysOfType<T, U> = {
  [K in keyof T]: T[K] extends U ? K : never;
}[keyof T];

/**
 * Omit keys of type T that have values of type U
 */
export type OmitByType<T, U> = Omit<T, KeysOfType<T, U>>;

/**
 * Pick keys of type T that have values of type U
 */
export type PickByType<T, U> = Pick<T, KeysOfType<T, U>>;

/**
 * Environment-specific types
 */
export interface BrowserEnvironment {
  userAgent: string;
  cookiesEnabled: boolean;
  localStorage: boolean;
  sessionStorage: boolean;
  webSockets: boolean;
  webWorkers: boolean;
  serviceWorkers: boolean;
}

export interface NodeEnvironment {
  version: string;
  platform: string;
  arch: string;
  memoryUsage: NodeJS.MemoryUsage;
  cpus: number;
}

/**
 * Platform detection
 */
export type Platform = 'browser' | 'node' | 'worker' | 'react-native' | 'electron' | 'unknown';

/**
 * Feature detection result
 */
export interface FeatureSupport {
  fetch: boolean;
  websockets: boolean;
  eventSource: boolean;
  streams: boolean;
  abortController: boolean;
  proxy: boolean;
  weakMap: boolean;
  performance: boolean;
}