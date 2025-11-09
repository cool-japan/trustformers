/**
 * TrustformeRS TypeScript Client
 * 
 * Main client implementation providing synchronous and promise-based APIs
 * for interacting with TrustformeRS serving infrastructure.
 */

import {
  InferenceRequest,
  InferenceResponse,
  BatchInferenceRequest,
  BatchInferenceResponse,
  ModelInfo,
  ModelStatus,
  HealthStatus,
  PerformanceMetrics,
  ServiceMetrics,
  StreamingToken,
} from './models';
import {
  TrustformersError,
  TrustformersAPIError,
  TrustformersTimeoutError,
  TrustformersConnectionError,
  TrustformersAuthenticationError,
} from './errors';
import { AuthConfig } from './auth';
import { RetryManager, RetryConfig } from './retry';
import { RequestTracker } from './monitoring';
import { ClientConfig, createDefaultConfig } from './config';
import { Logger, createLogger } from './logger';

/**
 * HTTP response interface
 */
interface HttpResponse<T = any> {
  data: T;
  status: number;
  statusText: string;
  headers: Record<string, string>;
}

/**
 * Request options interface
 */
interface RequestOptions {
  timeout?: number;
  retries?: number;
  headers?: Record<string, string>;
  signal?: AbortSignal;
}

/**
 * Main TrustformeRS client class
 */
export class TrustformersClient {
  private readonly config: ClientConfig;
  private readonly auth?: AuthConfig;
  private readonly retryManager: RetryManager;
  private readonly requestTracker?: RequestTracker;
  private readonly logger: Logger;
  private readonly baseUrl: string;
  private readonly defaultHeaders: Record<string, string>;

  /**
   * Initialize the TrustformeRS client
   */
  constructor(config: Partial<ClientConfig> = {}) {
    this.config = { ...createDefaultConfig(), ...config };
    this.auth = config.auth;
    this.baseUrl = this.config.baseUrl.replace(/\/$/, '');
    
    // Initialize default headers
    this.defaultHeaders = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'User-Agent': this.config.userAgent,
      ...this.config.headers,
    };

    // Initialize retry manager
    this.retryManager = new RetryManager({
      maxRetries: this.config.maxRetries,
      baseDelay: this.config.retryDelay,
      maxDelay: 60000,
      exponentialBase: 2,
    });

    // Initialize request tracking if enabled
    if (this.config.enableMetrics) {
      this.requestTracker = new RequestTracker();
    }

    // Initialize logger
    this.logger = createLogger({
      enabled: this.config.enableLogging,
      level: this.config.logLevel,
      prefix: 'TrustformersClient',
    });

    this.logger.info(`Initialized TrustformeRS client for ${this.baseUrl}`);
  }

  /**
   * Get authentication headers
   */
  private getAuthHeaders(): Record<string, string> {
    if (!this.auth) {
      return {};
    }
    return this.auth.getHeaders();
  }

  /**
   * Make an HTTP request with retry logic
   */
  private async makeRequest<T>(
    method: string,
    endpoint: string,
    data?: any,
    options: RequestOptions = {}
  ): Promise<HttpResponse<T>> {
    const url = `${this.baseUrl}${endpoint}`;
    const headers = {
      ...this.defaultHeaders,
      ...this.getAuthHeaders(),
      ...options.headers,
    };

    const requestOptions: RequestInit = {
      method,
      headers,
      signal: options.signal,
    };

    if (data && method !== 'GET') {
      requestOptions.body = JSON.stringify(data);
    }

    const startTime = performance.now();
    let lastError: Error | null = null;

    const maxRetries = options.retries ?? this.config.maxRetries;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        if (this.requestTracker) {
          this.requestTracker.recordRequestStart(method, endpoint);
        }

        // Create timeout controller
        const timeoutController = new AbortController();
        const timeout = options.timeout ?? this.config.timeout;
        const timeoutId = setTimeout(() => timeoutController.abort(), timeout);

        // Combine signals if provided
        let signal = timeoutController.signal;
        if (options.signal) {
          // Create a combined signal
          signal = this.combineAbortSignals([options.signal, timeoutController.signal]);
        }

        const response = await fetch(url, {
          ...requestOptions,
          signal,
        });

        clearTimeout(timeoutId);

        const duration = performance.now() - startTime;

        if (this.requestTracker) {
          this.requestTracker.recordRequestEnd(method, endpoint, response.status, duration);
        }

        // Handle authentication errors
        if (response.status === 401) {
          throw new TrustformersAuthenticationError(
            'Authentication failed. Check your credentials.'
          );
        }

        // Handle client errors (4xx)
        if (response.status >= 400 && response.status < 500) {
          let errorData: any = {};
          try {
            errorData = await response.json();
          } catch {
            // Ignore JSON parsing errors
          }

          const errorMessage = errorData.error || `HTTP ${response.status}: ${response.statusText}`;
          
          throw new TrustformersAPIError(
            errorMessage,
            response.status,
            errorData
          );
        }

        // Handle server errors (5xx) - these should be retried
        if (response.status >= 500) {
          if (attempt < maxRetries) {
            const delay = this.retryManager.calculateDelay(attempt);
            this.logger.warn(
              `Server error (attempt ${attempt + 1}/${maxRetries + 1}): ` +
              `HTTP ${response.status}. Retrying in ${delay}ms...`
            );
            await this.sleep(delay);
            continue;
          } else {
            let errorData: any = {};
            try {
              errorData = await response.json();
            } catch {
              // Ignore JSON parsing errors
            }

            const errorMessage = errorData.error || `HTTP ${response.status}: ${response.statusText}`;
            
            throw new TrustformersAPIError(
              errorMessage,
              response.status,
              errorData
            );
          }
        }

        // Success case
        const responseData = response.headers.get('content-type')?.includes('application/json')
          ? await response.json()
          : await response.text();

        const responseHeaders: Record<string, string> = {};
        response.headers.forEach((value, key) => {
          responseHeaders[key] = value;
        });

        return {
          data: responseData,
          status: response.status,
          statusText: response.statusText,
          headers: responseHeaders,
        };

      } catch (error) {
        if (error instanceof TrustformersError) {
          throw error;
        }

        if (error instanceof DOMException && error.name === 'AbortError') {
          if (options.signal?.aborted) {
            throw new TrustformersError('Request was cancelled');
          } else {
            lastError = new TrustformersTimeoutError(
              `Request timed out after ${timeout}ms`
            );
          }
        } else if (error instanceof TypeError && error.message.includes('fetch')) {
          lastError = new TrustformersConnectionError(
            `Connection failed: ${error.message}`
          );
        } else {
          lastError = new TrustformersError(`Unexpected error: ${error}`);
        }

        if (attempt < maxRetries && this.isRetryableError(lastError)) {
          const delay = this.retryManager.calculateDelay(attempt);
          this.logger.warn(
            `Request failed (attempt ${attempt + 1}/${maxRetries + 1}): ` +
            `${lastError.message}. Retrying in ${delay}ms...`
          );
          await this.sleep(delay);
          continue;
        }

        break;
      }
    }

    if (lastError) {
      throw lastError;
    }

    throw new TrustformersError('All retry attempts exhausted');
  }

  /**
   * Check if an error is retryable
   */
  private isRetryableError(error: Error): boolean {
    return (
      error instanceof TrustformersConnectionError ||
      error instanceof TrustformersTimeoutError ||
      (error instanceof TrustformersAPIError && error.status >= 500)
    );
  }

  /**
   * Sleep for a specified duration
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Combine multiple AbortSignals
   */
  private combineAbortSignals(signals: AbortSignal[]): AbortSignal {
    const controller = new AbortController();
    
    for (const signal of signals) {
      if (signal.aborted) {
        controller.abort();
        break;
      }
      
      signal.addEventListener('abort', () => controller.abort(), { once: true });
    }
    
    return controller.signal;
  }

  /**
   * Get server health status
   */
  async getHealth(options?: RequestOptions): Promise<HealthStatus> {
    const response = await this.makeRequest<HealthStatus>('GET', '/health', undefined, options);
    return response.data;
  }

  /**
   * Get detailed server health information
   */
  async getDetailedHealth(options?: RequestOptions): Promise<any> {
    const response = await this.makeRequest('GET', '/health/detailed', undefined, options);
    return response.data;
  }

  /**
   * Get server readiness status
   */
  async getReadiness(options?: RequestOptions): Promise<any> {
    const response = await this.makeRequest('GET', '/health/readiness', undefined, options);
    return response.data;
  }

  /**
   * Get server liveness status
   */
  async getLiveness(options?: RequestOptions): Promise<any> {
    const response = await this.makeRequest('GET', '/health/liveness', undefined, options);
    return response.data;
  }

  /**
   * Run inference on a single input
   */
  async infer(request: InferenceRequest, options?: RequestOptions): Promise<InferenceResponse> {
    const response = await this.makeRequest<InferenceResponse>(
      'POST',
      '/v1/infer',
      request,
      options
    );
    return response.data;
  }

  /**
   * Run inference on a batch of inputs
   */
  async batchInfer(
    request: BatchInferenceRequest,
    options?: RequestOptions
  ): Promise<BatchInferenceResponse> {
    const response = await this.makeRequest<BatchInferenceResponse>(
      'POST',
      '/v1/batch_infer',
      request,
      options
    );
    return response.data;
  }

  /**
   * Run streaming inference with token-by-token generation
   */
  async *streamInfer(
    request: InferenceRequest,
    options?: RequestOptions
  ): AsyncGenerator<StreamingToken, void, unknown> {
    const url = `${this.baseUrl}/v1/stream_infer`;
    const headers = {
      ...this.defaultHeaders,
      ...this.getAuthHeaders(),
      ...options?.headers,
    };

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(request),
        signal: options?.signal,
      });

      if (!response.ok) {
        throw new TrustformersAPIError(
          `Streaming failed with status ${response.status}`,
          response.status
        );
      }

      if (!response.body) {
        throw new TrustformersError('Response body is null');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      try {
        while (true) {
          const { done, value } = await reader.read();

          if (done) {
            break;
          }

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const dataStr = line.substring(6);
              if (dataStr.trim() === '[DONE]') {
                return;
              }

              try {
                const tokenData = JSON.parse(dataStr);
                yield tokenData as StreamingToken;
              } catch (error) {
                this.logger.warn(`Failed to parse streaming token: ${error}`);
                continue;
              }
            }
          }
        }
      } finally {
        reader.releaseLock();
      }
    } catch (error) {
      if (error instanceof TrustformersError) {
        throw error;
      }

      if (error instanceof DOMException && error.name === 'AbortError') {
        throw new TrustformersError('Streaming request was cancelled');
      }

      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new TrustformersConnectionError(`Streaming connection failed: ${error.message}`);
      }

      throw new TrustformersError(`Streaming error: ${error}`);
    }
  }

  /**
   * List all available models
   */
  async listModels(options?: RequestOptions): Promise<ModelInfo[]> {
    const response = await this.makeRequest<ModelInfo[]>('GET', '/v1/models', undefined, options);
    return response.data;
  }

  /**
   * Get information about a specific model
   */
  async getModelInfo(modelId: string, options?: RequestOptions): Promise<ModelInfo> {
    const response = await this.makeRequest<ModelInfo>(
      'GET',
      `/v1/models/${modelId}`,
      undefined,
      options
    );
    return response.data;
  }

  /**
   * Get the status of a specific model
   */
  async getModelStatus(modelId: string, options?: RequestOptions): Promise<ModelStatus> {
    const response = await this.makeRequest<ModelStatus>(
      'GET',
      `/v1/models/${modelId}/status`,
      undefined,
      options
    );
    return response.data;
  }

  /**
   * Load a model on the server
   */
  async loadModel(
    modelId: string,
    modelConfig?: Record<string, any>,
    options?: RequestOptions
  ): Promise<any> {
    const data = { model_id: modelId };
    if (modelConfig) {
      (data as any).config = modelConfig;
    }

    const response = await this.makeRequest(
      'POST',
      `/v1/models/${modelId}/load`,
      data,
      options
    );
    return response.data;
  }

  /**
   * Unload a model from the server
   */
  async unloadModel(modelId: string, options?: RequestOptions): Promise<any> {
    const response = await this.makeRequest(
      'POST',
      `/v1/models/${modelId}/unload`,
      undefined,
      options
    );
    return response.data;
  }

  /**
   * Get server metrics
   */
  async getMetrics(options?: RequestOptions): Promise<ServiceMetrics> {
    const response = await this.makeRequest<ServiceMetrics>('GET', '/metrics', undefined, options);
    return response.data;
  }

  /**
   * Get performance metrics
   */
  async getPerformanceMetrics(options?: RequestOptions): Promise<PerformanceMetrics> {
    const response = await this.makeRequest<PerformanceMetrics>(
      'GET',
      '/v1/metrics/performance',
      undefined,
      options
    );
    return response.data;
  }

  /**
   * Get server statistics
   */
  async getStats(options?: RequestOptions): Promise<any> {
    const response = await this.makeRequest('GET', '/stats', undefined, options);
    return response.data;
  }

  /**
   * Close the client and cleanup resources
   */
  close(): void {
    this.logger.debug('TrustformeRS client closed');
  }

  /**
   * Get client configuration
   */
  getConfig(): ClientConfig {
    return { ...this.config };
  }

  /**
   * Get request tracking statistics if enabled
   */
  getRequestStats(): any {
    return this.requestTracker?.getStats() || null;
  }

  /**
   * Clear request tracking statistics
   */
  clearRequestStats(): void {
    this.requestTracker?.reset();
  }
}