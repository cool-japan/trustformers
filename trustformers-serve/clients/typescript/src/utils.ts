/**
 * TrustformeRS TypeScript Client Utilities
 * 
 * Utility functions and helpers for the TrustformeRS client library.
 */

import type { AuthConfig } from './auth';
import type { ClientConfig } from './config';
import type { Platform, FeatureSupport } from './types';
import { TrustformersClient } from './client';
import { 
  TrustformersError, 
  TrustformersAPIError, 
  TrustformersConnectionError, 
  TrustformersTimeoutError,
  isRetryableError as baseIsRetryableError,
  getRetryDelay,
  extractErrorDetails
} from './errors';

/**
 * Create a simple client with API key authentication
 */
export function createClient(
  baseUrl: string = 'http://localhost:8080',
  apiKey?: string,
  options: Partial<ClientConfig> = {}
): TrustformersClient {
  const config: Partial<ClientConfig> = {
    baseUrl,
    ...options,
  };

  if (apiKey) {
    const { APIKeyAuth } = require('./auth');
    config.auth = new APIKeyAuth(apiKey);
  }

  return new TrustformersClient(config);
}

/**
 * Create an async client with API key authentication
 */
export async function createAsyncClient(
  baseUrl: string = 'http://localhost:8080',
  apiKey?: string,
  options: Partial<ClientConfig> = {}
): Promise<any> {
  // This would return AsyncTrustformersClient when implemented
  throw new Error('AsyncTrustformersClient not yet implemented');
}

/**
 * Create a streaming client
 */
export function createStreamingClient(
  baseUrl: string = 'http://localhost:8080',
  apiKey?: string,
  options: any = {}
): any {
  // This would return StreamingClient when implemented
  throw new Error('StreamingClient not yet implemented');
}

/**
 * Check if an error is retryable
 */
export function isRetryableError(error: Error): boolean {
  return baseIsRetryableError(error);
}

/**
 * Format error for display
 */
export function formatError(error: Error): string {
  if (error instanceof TrustformersAPIError) {
    const parts = [error.message];
    if (error.status) {
      parts.push(`(HTTP ${error.status})`);
    }
    if (error.requestId) {
      parts.push(`[${error.requestId}]`);
    }
    return parts.join(' ');
  }

  if (error instanceof TrustformersError) {
    return error.toString();
  }

  return error.message || 'Unknown error';
}

/**
 * Generate a unique request ID
 */
export function generateRequestId(): string {
  const timestamp = Date.now().toString(36);
  const randomPart = Math.random().toString(36).substring(2, 15);
  return `req_${timestamp}_${randomPart}`;
}

/**
 * Generate a UUID v4
 */
export function generateUUID(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }

  // Fallback implementation
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

/**
 * Validate URL format
 */
export function isValidUrl(url: string): boolean {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
}

/**
 * Validate API key format
 */
export function isValidApiKey(apiKey: string): boolean {
  if (!apiKey || typeof apiKey !== 'string') {
    return false;
  }
  
  // Basic validation: should be at least 16 characters and contain only alphanumeric, underscore, hyphen
  return /^[a-zA-Z0-9_-]{16,}$/.test(apiKey);
}

/**
 * Validate model ID format
 */
export function isValidModelId(modelId: string): boolean {
  if (!modelId || typeof modelId !== 'string') {
    return false;
  }
  
  // Model IDs should contain only letters, numbers, underscores, and hyphens
  return /^[a-zA-Z0-9_-]+$/.test(modelId);
}

/**
 * Sanitize headers by removing or masking sensitive values
 */
export function sanitizeHeaders(headers: Record<string, string>): Record<string, string> {
  const sensitiveHeaders = ['authorization', 'cookie', 'x-api-key', 'x-auth-token'];
  const sanitized: Record<string, string> = {};

  for (const [key, value] of Object.entries(headers)) {
    if (sensitiveHeaders.includes(key.toLowerCase())) {
      sanitized[key] = '[REDACTED]';
    } else {
      sanitized[key] = value;
    }
  }

  return sanitized;
}

/**
 * Deep clone an object
 */
export function deepClone<T>(obj: T): T {
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }

  if (obj instanceof Date) {
    return new Date(obj.getTime()) as any;
  }

  if (obj instanceof Array) {
    return obj.map(item => deepClone(item)) as any;
  }

  if (typeof obj === 'object') {
    const cloned = {} as any;
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        cloned[key] = deepClone(obj[key]);
      }
    }
    return cloned;
  }

  return obj;
}

/**
 * Deep merge two objects
 */
export function deepMerge<T extends Record<string, any>>(target: T, source: Partial<T>): T {
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
        result[key] = deepMerge(targetValue, sourceValue);
      } else if (sourceValue !== undefined) {
        result[key] = sourceValue;
      }
    }
  }

  return result;
}

/**
 * Debounce function calls
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number,
  immediate?: boolean
): T {
  let timeout: NodeJS.Timeout | null = null;
  
  return ((...args: any[]) => {
    const later = () => {
      timeout = null;
      if (!immediate) func.apply(null, args);
    };
    
    const callNow = immediate && !timeout;
    
    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(later, wait);
    
    if (callNow) func.apply(null, args);
  }) as T;
}

/**
 * Throttle function calls
 */
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): T {
  let inThrottle: boolean;
  
  return ((...args: any[]) => {
    if (!inThrottle) {
      func.apply(null, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  }) as T;
}

/**
 * Sleep for specified milliseconds
 */
export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Timeout a promise
 */
export function timeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return Promise.race([
    promise,
    new Promise<never>((_, reject) => 
      setTimeout(() => reject(new TrustformersTimeoutError(`Operation timed out after ${ms}ms`)), ms)
    )
  ]);
}

/**
 * Retry a function with exponential backoff
 */
export async function retry<T>(
  fn: () => Promise<T>,
  options: {
    maxRetries?: number;
    baseDelay?: number;
    maxDelay?: number;
    exponentialBase?: number;
    jitter?: boolean;
    retryCondition?: (error: Error) => boolean;
  } = {}
): Promise<T> {
  const {
    maxRetries = 3,
    baseDelay = 1000,
    maxDelay = 30000,
    exponentialBase = 2,
    jitter = true,
    retryCondition = isRetryableError
  } = options;

  let lastError: Error;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      if (attempt === maxRetries || !retryCondition(lastError)) {
        throw lastError;
      }

      let delay = Math.min(baseDelay * Math.pow(exponentialBase, attempt), maxDelay);
      
      if (jitter) {
        delay = delay * (0.5 + Math.random() * 0.5);
      }

      await sleep(delay);
    }
  }

  throw lastError!;
}

/**
 * Format bytes to human readable string
 */
export function formatBytes(bytes: number, decimals = 2): string {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

/**
 * Format duration to human readable string
 */
export function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${ms}ms`;
  }

  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) {
    return `${seconds}s`;
  }

  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    const remainingSeconds = seconds % 60;
    return remainingSeconds > 0 ? `${minutes}m ${remainingSeconds}s` : `${minutes}m`;
  }

  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return remainingMinutes > 0 ? `${hours}h ${remainingMinutes}m` : `${hours}h`;
}

/**
 * Parse duration string to milliseconds
 */
export function parseDuration(duration: string): number {
  const units: Record<string, number> = {
    ms: 1,
    s: 1000,
    m: 60 * 1000,
    h: 60 * 60 * 1000,
    d: 24 * 60 * 60 * 1000
  };

  const match = duration.match(/^(\d+(?:\.\d+)?)(ms|s|m|h|d)$/);
  if (!match) {
    throw new Error(`Invalid duration format: ${duration}`);
  }

  const [, value, unit] = match;
  return parseFloat(value) * units[unit];
}

/**
 * Detect current platform
 */
export function detectPlatform(): Platform {
  if (typeof window !== 'undefined') {
    if (typeof document !== 'undefined') {
      return 'browser';
    }
    return 'unknown';
  }

  if (typeof global !== 'undefined' && typeof process !== 'undefined') {
    if (process.versions?.electron) {
      return 'electron';
    }
    if (process.versions?.node) {
      return 'node';
    }
  }

  if (typeof importScripts === 'function') {
    return 'worker';
  }

  if (typeof navigator !== 'undefined' && navigator.product === 'ReactNative') {
    return 'react-native';
  }

  return 'unknown';
}

/**
 * Detect feature support
 */
export function detectFeatureSupport(): FeatureSupport {
  return {
    fetch: typeof fetch !== 'undefined',
    websockets: typeof WebSocket !== 'undefined',
    eventSource: typeof EventSource !== 'undefined',
    streams: typeof ReadableStream !== 'undefined',
    abortController: typeof AbortController !== 'undefined',
    proxy: (() => {
      try {
        new Proxy({}, {});
        return true;
      } catch {
        return false;
      }
    })(),
    weakMap: typeof WeakMap !== 'undefined',
    performance: typeof performance !== 'undefined',
  };
}

/**
 * Get user agent string
 */
export function getUserAgent(): string {
  if (typeof navigator !== 'undefined' && navigator.userAgent) {
    return navigator.userAgent;
  }
  
  if (typeof process !== 'undefined' && process.versions?.node) {
    return `Node.js/${process.versions.node}`;
  }
  
  return 'Unknown';
}

/**
 * Create cache key from object
 */
export function createCacheKey(obj: any): string {
  if (typeof obj === 'string') {
    return obj;
  }
  
  if (typeof obj === 'object' && obj !== null) {
    // Sort keys for consistent hashing
    const sortedKeys = Object.keys(obj).sort();
    const sortedObj = sortedKeys.reduce((result, key) => {
      result[key] = obj[key];
      return result;
    }, {} as any);
    
    return JSON.stringify(sortedObj);
  }
  
  return String(obj);
}

/**
 * Simple hash function for strings
 */
export function simpleHash(str: string): number {
  let hash = 0;
  if (str.length === 0) return hash;
  
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  
  return Math.abs(hash);
}

/**
 * Check if value is a plain object
 */
export function isPlainObject(value: any): value is Record<string, any> {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  
  if (Object.getPrototypeOf(value) === null) {
    return true;
  }
  
  let proto = value;
  while (Object.getPrototypeOf(proto) !== null) {
    proto = Object.getPrototypeOf(proto);
  }
  
  return Object.getPrototypeOf(value) === proto;
}

/**
 * Check if value is empty (null, undefined, empty string, empty array, empty object)
 */
export function isEmpty(value: any): boolean {
  if (value == null) {
    return true;
  }
  
  if (typeof value === 'string' || Array.isArray(value)) {
    return value.length === 0;
  }
  
  if (typeof value === 'object') {
    return Object.keys(value).length === 0;
  }
  
  return false;
}

/**
 * Capitalize first letter of string
 */
export function capitalize(str: string): string {
  if (!str) return str;
  return str.charAt(0).toUpperCase() + str.slice(1);
}

/**
 * Convert camelCase to snake_case
 */
export function camelToSnake(str: string): string {
  return str.replace(/[A-Z]/g, letter => `_${letter.toLowerCase()}`);
}

/**
 * Convert snake_case to camelCase
 */
export function snakeToCamel(str: string): string {
  return str.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
}

/**
 * Pick specific properties from object
 */
export function pick<T, K extends keyof T>(obj: T, keys: K[]): Pick<T, K> {
  const result = {} as Pick<T, K>;
  
  for (const key of keys) {
    if (key in obj) {
      result[key] = obj[key];
    }
  }
  
  return result;
}

/**
 * Omit specific properties from object
 */
export function omit<T, K extends keyof T>(obj: T, keys: K[]): Omit<T, K> {
  const result = { ...obj };
  
  for (const key of keys) {
    delete result[key];
  }
  
  return result;
}

/**
 * Group array items by key
 */
export function groupBy<T, K extends keyof T>(array: T[], key: K): Record<string, T[]> {
  return array.reduce((groups, item) => {
    const groupKey = String(item[key]);
    if (!groups[groupKey]) {
      groups[groupKey] = [];
    }
    groups[groupKey].push(item);
    return groups;
  }, {} as Record<string, T[]>);
}

/**
 * Remove duplicates from array
 */
export function unique<T>(array: T[], keyFn?: (item: T) => any): T[] {
  if (!keyFn) {
    return [...new Set(array)];
  }
  
  const seen = new Set();
  return array.filter(item => {
    const key = keyFn(item);
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

/**
 * Chunk array into smaller arrays
 */
export function chunk<T>(array: T[], size: number): T[][] {
  const chunks: T[][] = [];
  
  for (let i = 0; i < array.length; i += size) {
    chunks.push(array.slice(i, i + size));
  }
  
  return chunks;
}

/**
 * Flatten nested arrays
 */
export function flatten<T>(array: (T | T[])[]): T[] {
  return array.reduce<T[]>((acc, item) => {
    if (Array.isArray(item)) {
      acc.push(...flatten(item));
    } else {
      acc.push(item);
    }
    return acc;
  }, []);
}

// Re-export commonly used error utilities
export { 
  isRetryableError,
  getRetryDelay,
  extractErrorDetails,
  formatError
};