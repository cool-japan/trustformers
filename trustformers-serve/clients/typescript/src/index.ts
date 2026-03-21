/**
 * TrustformeRS TypeScript Client Library
 * 
 * A comprehensive TypeScript client for interacting with TrustformeRS serving infrastructure.
 * Provides type-safe APIs for inference, model management, streaming, and monitoring.
 */

// Core client exports
export { TrustformersClient } from './client';
export { AsyncTrustformersClient } from './async-client';
export { StreamingClient } from './streaming';
export { BatchManager } from './batch';

// Type definitions
export * from './types';
export * from './models';
export * from './errors';

// Authentication
export * from './auth';

// Monitoring and utilities
export { MonitoringClient } from './monitoring';
export { PerformanceTracker } from './performance';
export { RetryManager } from './retry';

// Configuration
export { ClientConfig, createDefaultConfig } from './config';

// Utilities
export {
  createClient,
  createAsyncClient,
  createStreamingClient,
  isRetryableError,
  formatError,
} from './utils';

// Constants
export {
  SUPPORTED_MODEL_TYPES,
  SUPPORTED_TASKS,
  DEFAULT_TIMEOUT,
  DEFAULT_MAX_RETRIES,
  API_VERSION,
} from './constants';

// Version information
export const VERSION = '0.1.0';
export const CLIENT_NAME = 'trustformers-client-typescript';
export const USER_AGENT = `${CLIENT_NAME}/${VERSION}`;

/**
 * Library information and capabilities
 */
export const LIBRARY_INFO = {
  name: CLIENT_NAME,
  version: VERSION,
  description: 'TypeScript client library for TrustformeRS serving infrastructure',
  homepage: 'https://github.com/trustformers/trustformers-serve',
  author: 'TrustformeRS Team',
  license: 'MIT',
  
  features: [
    'Type-safe API interactions',
    'Synchronous and asynchronous clients',
    'Streaming inference support',
    'Batch processing',
    'Model management',
    'Performance monitoring',
    'Authentication (API Key, JWT, OAuth2)',
    'Automatic retries and error handling',
    'WebSocket and SSE streaming',
    'Comprehensive logging',
    'Request/response caching',
    'Middleware support',
  ],
  
  supportedApis: [
    'REST API v1',
    'WebSocket Streaming',
    'Server-Sent Events',
    'GraphQL API',
    'gRPC (via grpc-web)',
  ],
  
  requirements: {
    typescript: '>=4.7.0',
    node: '>=16.0.0',
    browsers: [
      'Chrome >= 88',
      'Firefox >= 85',
      'Safari >= 14',
      'Edge >= 88',
    ],
  },
};

/**
 * Quick start example for getting started with the library
 */
export const QUICK_START_EXAMPLE = `
// Quick Start Example

import { TrustformersClient, InferenceRequest, TaskType } from 'trustformers-client';

// Initialize client
const client = new TrustformersClient({
  baseUrl: 'http://localhost:8080',
  apiKey: 'your-api-key', // Optional
  timeout: 30000,
});

// Check server health
const health = await client.getHealth();
console.log('Server status:', health.status);

// Run inference
const request: InferenceRequest = {
  inputText: 'Hello, how are you?',
  modelId: 'gpt2-small',
  taskType: TaskType.TEXT_GENERATION,
  maxLength: 50,
};

const response = await client.infer(request);
console.log('Generated text:', response.outputText);

// Streaming inference
for await (const token of client.streamInfer(request)) {
  process.stdout.write(token.text);
}

// Batch inference
const batchRequest = {
  inputs: ['Text 1', 'Text 2', 'Text 3'],
  modelId: 'bert-base-uncased',
  taskType: TaskType.TEXT_CLASSIFICATION,
};

const batchResponse = await client.batchInfer(batchRequest);
batchResponse.results.forEach((result, index) => {
  console.log(\`Result \${index + 1}:\`, result.outputText);
});
`;

/**
 * Default configuration for the client library
 */
export const DEFAULT_CONFIG = {
  baseUrl: 'http://localhost:8080',
  timeout: 30000,
  maxRetries: 3,
  retryDelay: 1000,
  enableCaching: true,
  enableMetrics: true,
  enableLogging: true,
  logLevel: 'info' as const,
  userAgent: USER_AGENT,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
} as const;

/**
 * Configure global defaults for all client instances
 */
let globalConfig = { ...DEFAULT_CONFIG };

export function setGlobalConfig(config: Partial<typeof DEFAULT_CONFIG>): void {
  globalConfig = { ...globalConfig, ...config };
}

export function getGlobalConfig(): typeof DEFAULT_CONFIG {
  return { ...globalConfig };
}

/**
 * Initialize the library with global configuration
 */
export function initialize(config?: Partial<typeof DEFAULT_CONFIG>): void {
  if (config) {
    setGlobalConfig(config);
  }
  
  // Initialize logging if in Node.js environment
  if (typeof window === 'undefined' && globalConfig.enableLogging) {
    console.log(`${CLIENT_NAME} v${VERSION} initialized`);
  }
}

/**
 * Get library version and build information
 */
export function getVersion(): { version: string; buildInfo: Record<string, any> } {
  return {
    version: VERSION,
    buildInfo: {
      name: CLIENT_NAME,
      userAgent: USER_AGENT,
      timestamp: new Date().toISOString(),
      nodeVersion: typeof process !== 'undefined' ? process.version : undefined,
      platform: typeof process !== 'undefined' ? process.platform : 'browser',
    },
  };
}

/**
 * Check if the current environment supports all client features
 */
export function checkCompatibility(): {
  compatible: boolean;
  missingFeatures: string[];
  warnings: string[];
} {
  const missingFeatures: string[] = [];
  const warnings: string[] = [];
  
  // Check for fetch API
  if (typeof fetch === 'undefined') {
    missingFeatures.push('fetch API');
  }
  
  // Check for WebSocket support
  if (typeof WebSocket === 'undefined') {
    warnings.push('WebSocket not available - streaming features limited');
  }
  
  // Check for EventSource support
  if (typeof EventSource === 'undefined') {
    warnings.push('EventSource not available - SSE streaming not supported');
  }
  
  // Check for AbortController support
  if (typeof AbortController === 'undefined') {
    warnings.push('AbortController not available - request cancellation limited');
  }
  
  // Check for Proxy support (for reactive features)
  try {
    new Proxy({}, {});
  } catch {
    warnings.push('Proxy not supported - some advanced features may not work');
  }
  
  return {
    compatible: missingFeatures.length === 0,
    missingFeatures,
    warnings,
  };
}

/**
 * Browser detection utilities
 */
export const browserInfo = (() => {
  if (typeof window === 'undefined') {
    return { name: 'node', version: '', supported: true };
  }
  
  const userAgent = navigator.userAgent;
  let name = 'unknown';
  let version = '';
  let supported = false;
  
  if (userAgent.includes('Chrome/')) {
    name = 'chrome';
    version = userAgent.match(/Chrome\\/([0-9.]+)/)?.[1] || '';
    supported = parseInt(version) >= 88;
  } else if (userAgent.includes('Firefox/')) {
    name = 'firefox';
    version = userAgent.match(/Firefox\\/([0-9.]+)/)?.[1] || '';
    supported = parseInt(version) >= 85;
  } else if (userAgent.includes('Safari/') && !userAgent.includes('Chrome/')) {
    name = 'safari';
    version = userAgent.match(/Version\\/([0-9.]+)/)?.[1] || '';
    supported = parseInt(version) >= 14;
  } else if (userAgent.includes('Edge/')) {
    name = 'edge';
    version = userAgent.match(/Edge\\/([0-9.]+)/)?.[1] || '';
    supported = parseInt(version) >= 88;
  }
  
  return { name, version, supported };
})();

/**
 * Environment detection
 */
export const environment = {
  isNode: typeof window === 'undefined',
  isBrowser: typeof window !== 'undefined',
  isWebWorker: typeof importScripts === 'function',
  isServiceWorker: typeof ServiceWorkerGlobalScope !== 'undefined',
  supportsFetch: typeof fetch !== 'undefined',
  supportsWebSocket: typeof WebSocket !== 'undefined',
  supportsEventSource: typeof EventSource !== 'undefined',
  supportsStreams: typeof ReadableStream !== 'undefined',
};

/**
 * Performance utilities
 */
export const performance = {
  now: (() => {
    if (typeof window !== 'undefined' && window.performance) {
      return () => window.performance.now();
    } else if (typeof process !== 'undefined' && process.hrtime) {
      return () => {
        const [seconds, nanoseconds] = process.hrtime();
        return seconds * 1000 + nanoseconds / 1000000;
      };
    } else {
      return () => Date.now();
    }
  })(),
  
  mark: (name: string) => {
    if (typeof window !== 'undefined' && window.performance?.mark) {
      window.performance.mark(name);
    }
  },
  
  measure: (name: string, startMark: string, endMark: string) => {
    if (typeof window !== 'undefined' && window.performance?.measure) {
      return window.performance.measure(name, startMark, endMark);
    }
    return null;
  },
};

// Auto-initialization in browser environments
if (environment.isBrowser) {
  // Check compatibility and warn about issues
  const compat = checkCompatibility();
  if (!compat.compatible) {
    console.warn(
      `TrustformeRS client compatibility issues detected:`,
      compat.missingFeatures
    );
  }
  if (compat.warnings.length > 0) {
    console.info(
      `TrustformeRS client warnings:`,
      compat.warnings
    );
  }
  
  // Browser-specific initialization
  if (!browserInfo.supported) {
    console.warn(
      `Browser ${browserInfo.name} ${browserInfo.version} may not be fully supported. ` +
      `Please update to a supported version.`
    );
  }
}

// Default export for convenience
export default {
  TrustformersClient,
  AsyncTrustformersClient,
  StreamingClient,
  BatchManager,
  MonitoringClient,
  createClient,
  createAsyncClient,
  initialize,
  getVersion,
  checkCompatibility,
  VERSION,
  LIBRARY_INFO,
  environment,
  browserInfo,
  performance,
};