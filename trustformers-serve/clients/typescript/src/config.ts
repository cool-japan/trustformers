/**
 * TrustformeRS TypeScript Client Configuration
 * 
 * Provides configuration management for TrustformeRS client instances.
 * Supports environment variables, file-based config, and runtime configuration.
 */

import type { AuthConfig } from './auth';
import type { RetryConfig } from './retry';

/**
 * Log levels for client logging
 */
export enum LogLevel {
  DEBUG = 'debug',
  INFO = 'info',
  WARN = 'warn',
  ERROR = 'error',
  NONE = 'none',
}

/**
 * Client configuration interface
 */
export interface ClientConfig {
  /** Base URL of the TrustformeRS server */
  baseUrl: string;
  /** Authentication configuration */
  auth?: AuthConfig;
  /** Request timeout in milliseconds */
  timeout: number;
  /** Maximum number of retry attempts */
  maxRetries: number;
  /** Base delay between retries in milliseconds */
  retryDelay: number;
  /** Whether to verify SSL certificates */
  verifySsl: boolean;
  /** Additional headers to include in all requests */
  headers: Record<string, string>;
  /** User agent string */
  userAgent: string;
  /** Whether to enable request/response caching */
  enableCaching: boolean;
  /** Whether to enable request metrics collection */
  enableMetrics: boolean;
  /** Whether to enable logging */
  enableLogging: boolean;
  /** Log level */
  logLevel: LogLevel;
  /** Maximum number of connections in the pool */
  maxConnections: number;
  /** Maximum keepalive connections */
  maxKeepaliveConnections: number;
  /** Enable HTTP/2 if available */
  enableHttp2: boolean;
  /** Custom retry configuration */
  retryConfig?: Partial<RetryConfig>;
  /** Environment-specific settings */
  environment: 'development' | 'staging' | 'production';
  /** Rate limiting configuration */
  rateLimit?: {
    maxRequestsPerSecond: number;
    maxConcurrentRequests: number;
  };
  /** Compression settings */
  compression: {
    enableRequestCompression: boolean;
    enableResponseCompression: boolean;
    compressionThreshold: number; // bytes
  };
  /** Circuit breaker settings */
  circuitBreaker?: {
    enabled: boolean;
    failureThreshold: number;
    recoveryTimeout: number;
  };
}

/**
 * Default client configuration
 */
export const DEFAULT_CLIENT_CONFIG: ClientConfig = {
  baseUrl: 'http://localhost:8080',
  timeout: 30000,
  maxRetries: 3,
  retryDelay: 1000,
  verifySsl: true,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
  userAgent: 'trustformers-client-typescript/0.1.0',
  enableCaching: true,
  enableMetrics: true,
  enableLogging: true,
  logLevel: LogLevel.INFO,
  maxConnections: 100,
  maxKeepaliveConnections: 20,
  enableHttp2: true,
  environment: 'development',
  compression: {
    enableRequestCompression: true,
    enableResponseCompression: true,
    compressionThreshold: 1024,
  },
};

/**
 * Environment-specific configuration presets
 */
export const ENVIRONMENT_CONFIGS: Record<string, Partial<ClientConfig>> = {
  development: {
    baseUrl: 'http://localhost:8080',
    enableLogging: true,
    logLevel: LogLevel.DEBUG,
    verifySsl: false,
    environment: 'development',
  },
  staging: {
    baseUrl: 'https://staging-api.trustformers.ai',
    enableLogging: true,
    logLevel: LogLevel.INFO,
    verifySsl: true,
    environment: 'staging',
  },
  production: {
    baseUrl: 'https://api.trustformers.ai',
    enableLogging: false,
    logLevel: LogLevel.ERROR,
    verifySsl: true,
    maxRetries: 5,
    timeout: 60000,
    environment: 'production',
    circuitBreaker: {
      enabled: true,
      failureThreshold: 10,
      recoveryTimeout: 300000, // 5 minutes
    },
  },
};

/**
 * Configuration validation and type checking
 */
export class ConfigValidator {
  static validate(config: Partial<ClientConfig>): string[] {
    const errors: string[] = [];

    if (config.baseUrl && !this.isValidUrl(config.baseUrl)) {
      errors.push('baseUrl must be a valid URL');
    }

    if (config.timeout && (config.timeout <= 0 || config.timeout > 300000)) {
      errors.push('timeout must be between 1 and 300000 milliseconds');
    }

    if (config.maxRetries && (config.maxRetries < 0 || config.maxRetries > 10)) {
      errors.push('maxRetries must be between 0 and 10');
    }

    if (config.retryDelay && config.retryDelay < 0) {
      errors.push('retryDelay must be non-negative');
    }

    if (config.maxConnections && config.maxConnections <= 0) {
      errors.push('maxConnections must be positive');
    }

    if (config.maxKeepaliveConnections && config.maxKeepaliveConnections < 0) {
      errors.push('maxKeepaliveConnections must be non-negative');
    }

    if (config.rateLimit) {
      if (config.rateLimit.maxRequestsPerSecond <= 0) {
        errors.push('rateLimit.maxRequestsPerSecond must be positive');
      }
      if (config.rateLimit.maxConcurrentRequests <= 0) {
        errors.push('rateLimit.maxConcurrentRequests must be positive');
      }
    }

    return errors;
  }

  private static isValidUrl(url: string): boolean {
    try {
      new URL(url);
      return true;
    } catch {
      return false;
    }
  }
}

/**
 * Configuration manager for loading and merging configurations
 */
export class ConfigManager {
  private config: ClientConfig;

  constructor(initialConfig: Partial<ClientConfig> = {}) {
    this.config = this.createDefaultConfig();
    this.mergeConfig(initialConfig);
  }

  /**
   * Load configuration from environment variables
   */
  loadFromEnvironment(): void {
    const envConfig: Partial<ClientConfig> = {};

    // Load from environment variables
    if (process?.env) {
      const env = process.env;
      
      if (env.TRUSTFORMERS_BASE_URL) {
        envConfig.baseUrl = env.TRUSTFORMERS_BASE_URL;
      }
      
      if (env.TRUSTFORMERS_TIMEOUT) {
        envConfig.timeout = parseInt(env.TRUSTFORMERS_TIMEOUT, 10);
      }
      
      if (env.TRUSTFORMERS_MAX_RETRIES) {
        envConfig.maxRetries = parseInt(env.TRUSTFORMERS_MAX_RETRIES, 10);
      }
      
      if (env.TRUSTFORMERS_VERIFY_SSL) {
        envConfig.verifySsl = env.TRUSTFORMERS_VERIFY_SSL.toLowerCase() === 'true';
      }
      
      if (env.TRUSTFORMERS_LOG_LEVEL) {
        envConfig.logLevel = env.TRUSTFORMERS_LOG_LEVEL as LogLevel;
      }
      
      if (env.TRUSTFORMERS_ENABLE_LOGGING) {
        envConfig.enableLogging = env.TRUSTFORMERS_ENABLE_LOGGING.toLowerCase() === 'true';
      }
      
      if (env.NODE_ENV) {
        envConfig.environment = env.NODE_ENV as 'development' | 'staging' | 'production';
      }
    }

    this.mergeConfig(envConfig);
  }

  /**
   * Load configuration from a JSON file (Node.js only)
   */
  async loadFromFile(filePath: string): Promise<void> {
    if (typeof require === 'undefined') {
      throw new Error('File loading is only supported in Node.js environments');
    }

    try {
      const fs = require('fs').promises;
      const configText = await fs.readFile(filePath, 'utf8');
      const fileConfig = JSON.parse(configText);
      this.mergeConfig(fileConfig);
    } catch (error) {
      throw new Error(`Failed to load config from file: ${error}`);
    }
  }

  /**
   * Merge configuration with existing config
   */
  mergeConfig(config: Partial<ClientConfig>): void {
    // Validate configuration
    const errors = ConfigValidator.validate(config);
    if (errors.length > 0) {
      throw new Error(`Configuration validation failed: ${errors.join(', ')}`);
    }

    // Deep merge configuration
    this.config = this.deepMerge(this.config, config);

    // Apply environment-specific configuration
    if (config.environment) {
      const envConfig = ENVIRONMENT_CONFIGS[config.environment];
      if (envConfig) {
        this.config = this.deepMerge(this.config, envConfig);
      }
    }
  }

  /**
   * Get current configuration
   */
  getConfig(): ClientConfig {
    return { ...this.config };
  }

  /**
   * Update specific configuration values
   */
  updateConfig(updates: Partial<ClientConfig>): void {
    this.mergeConfig(updates);
  }

  /**
   * Reset to default configuration
   */
  reset(): void {
    this.config = this.createDefaultConfig();
  }

  /**
   * Export configuration as JSON
   */
  exportConfig(): string {
    return JSON.stringify(this.config, null, 2);
  }

  private createDefaultConfig(): ClientConfig {
    return { ...DEFAULT_CLIENT_CONFIG };
  }

  private deepMerge<T>(target: T, source: Partial<T>): T {
    const result = { ...target };

    for (const key in source) {
      if (source.hasOwnProperty(key)) {
        const sourceValue = source[key];
        const targetValue = result[key];

        if (
          sourceValue && 
          typeof sourceValue === 'object' && 
          !Array.isArray(sourceValue) &&
          targetValue &&
          typeof targetValue === 'object' &&
          !Array.isArray(targetValue)
        ) {
          result[key] = this.deepMerge(targetValue, sourceValue);
        } else if (sourceValue !== undefined) {
          result[key] = sourceValue as any;
        }
      }
    }

    return result;
  }
}

/**
 * Create default configuration
 */
export function createDefaultConfig(): ClientConfig {
  return { ...DEFAULT_CLIENT_CONFIG };
}

/**
 * Create configuration for specific environment
 */
export function createEnvironmentConfig(environment: string): ClientConfig {
  const baseConfig = createDefaultConfig();
  const envConfig = ENVIRONMENT_CONFIGS[environment];
  
  if (envConfig) {
    return { ...baseConfig, ...envConfig };
  }
  
  return baseConfig;
}

/**
 * Validate configuration object
 */
export function validateConfig(config: Partial<ClientConfig>): { valid: boolean; errors: string[] } {
  const errors = ConfigValidator.validate(config);
  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Load configuration from environment variables
 */
export function loadConfigFromEnvironment(): Partial<ClientConfig> {
  const manager = new ConfigManager();
  manager.loadFromEnvironment();
  return manager.getConfig();
}

/**
 * Create configuration builder for fluent API
 */
export class ConfigBuilder {
  private config: Partial<ClientConfig> = {};

  baseUrl(url: string): ConfigBuilder {
    this.config.baseUrl = url;
    return this;
  }

  timeout(ms: number): ConfigBuilder {
    this.config.timeout = ms;
    return this;
  }

  retries(max: number, delay?: number): ConfigBuilder {
    this.config.maxRetries = max;
    if (delay !== undefined) {
      this.config.retryDelay = delay;
    }
    return this;
  }

  auth(authConfig: AuthConfig): ConfigBuilder {
    this.config.auth = authConfig;
    return this;
  }

  headers(headers: Record<string, string>): ConfigBuilder {
    this.config.headers = { ...this.config.headers, ...headers };
    return this;
  }

  logging(enabled: boolean, level?: LogLevel): ConfigBuilder {
    this.config.enableLogging = enabled;
    if (level !== undefined) {
      this.config.logLevel = level;
    }
    return this;
  }

  metrics(enabled: boolean): ConfigBuilder {
    this.config.enableMetrics = enabled;
    return this;
  }

  environment(env: 'development' | 'staging' | 'production'): ConfigBuilder {
    this.config.environment = env;
    return this;
  }

  ssl(verify: boolean): ConfigBuilder {
    this.config.verifySsl = verify;
    return this;
  }

  connections(max: number, keepalive?: number): ConfigBuilder {
    this.config.maxConnections = max;
    if (keepalive !== undefined) {
      this.config.maxKeepaliveConnections = keepalive;
    }
    return this;
  }

  compression(enabled: boolean, threshold?: number): ConfigBuilder {
    this.config.compression = {
      enableRequestCompression: enabled,
      enableResponseCompression: enabled,
      compressionThreshold: threshold || 1024,
    };
    return this;
  }

  circuitBreaker(enabled: boolean, failureThreshold?: number, recoveryTimeout?: number): ConfigBuilder {
    this.config.circuitBreaker = {
      enabled,
      failureThreshold: failureThreshold || 5,
      recoveryTimeout: recoveryTimeout || 60000,
    };
    return this;
  }

  build(): ClientConfig {
    const manager = new ConfigManager(this.config);
    return manager.getConfig();
  }
}