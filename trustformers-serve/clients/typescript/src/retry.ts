/**
 * TrustformeRS TypeScript Client Retry Logic
 * 
 * Provides retry mechanisms for handling transient failures in API requests.
 * Supports exponential backoff, jitter, and configurable retry strategies.
 */

import { isRetryableError, getRetryDelay } from './errors';

/**
 * Retry configuration options
 */
export interface RetryConfig {
  /** Maximum number of retry attempts */
  maxRetries: number;
  /** Base delay between retries in milliseconds */
  baseDelay: number;
  /** Maximum delay between retries in milliseconds */
  maxDelay: number;
  /** Exponential backoff multiplier */
  exponentialBase: number;
  /** Whether to add jitter to delays */
  enableJitter: boolean;
  /** Maximum jitter amount as percentage of delay (0.0-1.0) */
  jitterFactor: number;
  /** Custom retry condition function */
  retryCondition?: (error: Error, attempt: number) => boolean;
  /** Custom delay calculation function */
  delayCalculation?: (attempt: number, baseDelay: number, config: RetryConfig) => number;
}

/**
 * Default retry configuration
 */
export const DEFAULT_RETRY_CONFIG: RetryConfig = {
  maxRetries: 3,
  baseDelay: 1000,
  maxDelay: 60000,
  exponentialBase: 2,
  enableJitter: true,
  jitterFactor: 0.1,
};

/**
 * Retry manager for handling request retries
 */
export class RetryManager {
  private readonly config: RetryConfig;

  constructor(config: Partial<RetryConfig> = {}) {
    this.config = { ...DEFAULT_RETRY_CONFIG, ...config };
  }

  /**
   * Calculate delay for a retry attempt
   */
  calculateDelay(attempt: number): number {
    if (this.config.delayCalculation) {
      return this.config.delayCalculation(attempt, this.config.baseDelay, this.config);
    }

    // Exponential backoff calculation
    let delay = this.config.baseDelay * Math.pow(this.config.exponentialBase, attempt);
    
    // Apply maximum delay cap
    delay = Math.min(delay, this.config.maxDelay);
    
    // Add jitter if enabled
    if (this.config.enableJitter) {
      const jitter = delay * this.config.jitterFactor * Math.random();
      delay += jitter;
    }
    
    return Math.floor(delay);
  }

  /**
   * Check if an error should be retried
   */
  shouldRetry(error: Error, attempt: number): boolean {
    // Check attempt limit
    if (attempt >= this.config.maxRetries) {
      return false;
    }
    
    // Use custom retry condition if provided
    if (this.config.retryCondition) {
      return this.config.retryCondition(error, attempt);
    }
    
    // Use default retry logic from error module
    return isRetryableError(error);
  }

  /**
   * Execute a function with retry logic
   */
  async retry<T>(
    fn: () => Promise<T>,
    context?: string
  ): Promise<T> {
    let lastError: Error | null = null;
    
    for (let attempt = 0; attempt <= this.config.maxRetries; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        
        if (!this.shouldRetry(lastError, attempt)) {
          break;
        }
        
        const delay = this.calculateDelay(attempt);
        
        if (context) {
          console.warn(
            `${context} failed (attempt ${attempt + 1}/${this.config.maxRetries + 1}): ` +
            `${lastError.message}. Retrying in ${delay}ms...`
          );
        }
        
        await this.sleep(delay);
      }
    }
    
    // All retries exhausted
    if (lastError) {
      throw lastError;
    }
    
    throw new Error('All retry attempts exhausted');
  }

  /**
   * Sleep for specified milliseconds
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get current retry configuration
   */
  getConfig(): RetryConfig {
    return { ...this.config };
  }

  /**
   * Update retry configuration
   */
  updateConfig(updates: Partial<RetryConfig>): void {
    Object.assign(this.config, updates);
  }
}

/**
 * Retry strategies
 */
export class RetryStrategies {
  /**
   * Linear backoff strategy
   */
  static linear(baseDelay: number): (attempt: number) => number {
    return (attempt: number) => baseDelay * (attempt + 1);
  }

  /**
   * Exponential backoff strategy
   */
  static exponential(baseDelay: number, base: number = 2): (attempt: number) => number {
    return (attempt: number) => baseDelay * Math.pow(base, attempt);
  }

  /**
   * Fixed delay strategy
   */
  static fixed(delay: number): (attempt: number) => number {
    return () => delay;
  }

  /**
   * Custom fibonacci-like backoff
   */
  static fibonacci(baseDelay: number): (attempt: number) => number {
    return (attempt: number) => {
      if (attempt <= 1) return baseDelay;
      
      let a = 1, b = 1;
      for (let i = 2; i <= attempt; i++) {
        [a, b] = [b, a + b];
      }
      
      return baseDelay * b;
    };
  }

  /**
   * Decorrelated jitter strategy (recommended for distributed systems)
   */
  static decorrelatedJitter(baseDelay: number, maxDelay: number): (attempt: number, previousDelay?: number) => number {
    let lastDelay = baseDelay;
    
    return (attempt: number) => {
      const delay = Math.random() * (lastDelay * 3 - baseDelay) + baseDelay;
      lastDelay = Math.min(delay, maxDelay);
      return Math.floor(lastDelay);
    };
  }
}

/**
 * Retry conditions
 */
export class RetryConditions {
  /**
   * Retry on specific HTTP status codes
   */
  static httpStatusCodes(statusCodes: number[]): (error: Error) => boolean {
    return (error: Error) => {
      if ('status' in error && typeof error.status === 'number') {
        return statusCodes.includes(error.status);
      }
      return false;
    };
  }

  /**
   * Retry on specific error types
   */
  static errorTypes(errorTypes: (new (...args: any[]) => Error)[]): (error: Error) => boolean {
    return (error: Error) => {
      return errorTypes.some(ErrorType => error instanceof ErrorType);
    };
  }

  /**
   * Retry on errors matching regex pattern
   */
  static messagePattern(pattern: RegExp): (error: Error) => boolean {
    return (error: Error) => pattern.test(error.message);
  }

  /**
   * Combine multiple retry conditions with AND logic
   */
  static and(...conditions: ((error: Error) => boolean)[]): (error: Error) => boolean {
    return (error: Error) => conditions.every(condition => condition(error));
  }

  /**
   * Combine multiple retry conditions with OR logic
   */
  static or(...conditions: ((error: Error) => boolean)[]): (error: Error) => boolean {
    return (error: Error) => conditions.some(condition => condition(error));
  }

  /**
   * Negate a retry condition
   */
  static not(condition: (error: Error) => boolean): (error: Error) => boolean {
    return (error: Error) => !condition(error);
  }
}

/**
 * Circuit breaker pattern for handling cascading failures
 */
export class CircuitBreaker {
  private failures = 0;
  private lastFailureTime = 0;
  private state: 'closed' | 'open' | 'half-open' = 'closed';

  constructor(
    private readonly failureThreshold: number = 5,
    private readonly recoveryTimeout: number = 60000, // 1 minute
    private readonly successThreshold: number = 2
  ) {}

  /**
   * Execute function with circuit breaker protection
   */
  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.state === 'open') {
      if (Date.now() - this.lastFailureTime > this.recoveryTimeout) {
        this.state = 'half-open';
      } else {
        throw new Error('Circuit breaker is open');
      }
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  private onSuccess(): void {
    this.failures = 0;
    if (this.state === 'half-open') {
      this.state = 'closed';
    }
  }

  private onFailure(): void {
    this.failures++;
    this.lastFailureTime = Date.now();
    
    if (this.failures >= this.failureThreshold) {
      this.state = 'open';
    }
  }

  /**
   * Get current circuit breaker state
   */
  getState(): 'closed' | 'open' | 'half-open' {
    return this.state;
  }

  /**
   * Reset circuit breaker to closed state
   */
  reset(): void {
    this.failures = 0;
    this.lastFailureTime = 0;
    this.state = 'closed';
  }

  /**
   * Get failure count
   */
  getFailureCount(): number {
    return this.failures;
  }
}

// Utility functions

/**
 * Add jitter to a delay value
 */
export function addJitter(delay: number, factor: number = 0.1): number {
  const jitter = delay * factor * Math.random();
  return Math.floor(delay + jitter);
}

/**
 * Calculate exponential backoff delay
 */
export function exponentialBackoff(
  attempt: number,
  baseDelay: number = 1000,
  maxDelay: number = 60000,
  base: number = 2
): number {
  const delay = Math.min(baseDelay * Math.pow(base, attempt), maxDelay);
  return addJitter(delay);
}

/**
 * Create a retry manager with common configurations
 */
export function createRetryManager(type: 'aggressive' | 'moderate' | 'conservative'): RetryManager {
  const configs = {
    aggressive: {
      maxRetries: 5,
      baseDelay: 500,
      maxDelay: 30000,
      exponentialBase: 1.5,
    },
    moderate: {
      maxRetries: 3,
      baseDelay: 1000,
      maxDelay: 60000,
      exponentialBase: 2,
    },
    conservative: {
      maxRetries: 2,
      baseDelay: 2000,
      maxDelay: 120000,
      exponentialBase: 3,
    },
  };

  return new RetryManager(configs[type]);
}

/**
 * Retry decorator for methods
 */
export function withRetry(config: Partial<RetryConfig> = {}) {
  const retryManager = new RetryManager(config);

  return function <T extends (...args: any[]) => Promise<any>>(
    target: any,
    propertyKey: string,
    descriptor: TypedPropertyDescriptor<T>
  ) {
    const originalMethod = descriptor.value!;

    descriptor.value = async function (...args: any[]) {
      return retryManager.retry(
        () => originalMethod.apply(this, args),
        `${target.constructor.name}.${propertyKey}`
      );
    } as T;

    return descriptor;
  };
}