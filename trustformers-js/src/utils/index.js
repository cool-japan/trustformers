/**
 * TrustformeRS Utils Module
 * Tree-shakable utility functions and helpers
 */

// No imports from main module to avoid circular dependencies

// Utility functions
export class TrustformersError extends Error {
  constructor(message, code = 'TRUSTFORMERS_ERROR') {
    super(message);
    this.name = 'TrustformersError';
    this.code = code;
  }
}

export function validateConfig(config, schema) {
  if (!config || typeof config !== 'object') {
    throw new TrustformersError('Config must be an object');
  }
  return true;
}

export function createDevice(deviceType = 'cpu') {
  return {
    type: deviceType,
    available: true,
    memory: deviceType === 'gpu' ? '8GB' : '16GB',
  };
}

export function createTensor(data, shape, options = {}) {
  // Simple tensor creation stub - actual implementation would need to avoid circular dependency
  // This is a placeholder that returns a basic tensor-like object
  return {
    data: Array.isArray(data) ? data : [data],
    shape: shape || [data.length || 1],
    dtype: options.dtype || 'f32',
    ...options
  };
}

// Browser compatibility utilities
export class BrowserCompat {
  /**
   * Check WebGL support
   */
  static hasWebGL() {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      return !!gl;
    } catch {
      return false;
    }
  }

  /**
   * Check WebGL2 support
   */
  static hasWebGL2() {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2');
      return !!gl;
    } catch {
      return false;
    }
  }

  /**
   * Check WebGPU support
   */
  static async hasWebGPU() {
    return 'gpu' in navigator && (await navigator.gpu?.requestAdapter());
  }

  /**
   * Check WebAssembly support
   */
  static hasWebAssembly() {
    return 'WebAssembly' in window;
  }

  /**
   * Check SharedArrayBuffer support
   */
  static hasSharedArrayBuffer() {
    return 'SharedArrayBuffer' in window;
  }

  /**
   * Check OffscreenCanvas support
   */
  static hasOffscreenCanvas() {
    return 'OffscreenCanvas' in window;
  }

  /**
   * Get browser capabilities summary
   */
  static async getCapabilities() {
    return {
      webgl: this.hasWebGL(),
      webgl2: this.hasWebGL2(),
      webgpu: await this.hasWebGPU(),
      webassembly: this.hasWebAssembly(),
      sharedArrayBuffer: this.hasSharedArrayBuffer(),
      offscreenCanvas: this.hasOffscreenCanvas(),
      userAgent: navigator.userAgent,
      platform: navigator.platform,
      cookieEnabled: navigator.cookieEnabled,
      onLine: navigator.onLine,
    };
  }
}

// Performance monitoring utilities
export class PerformanceMonitor {
  constructor() {
    this.metrics = new Map();
    this.observers = new Map();
  }

  /**
   * Start timing an operation
   */
  startTiming(name) {
    const startTime = performance.now();
    this.metrics.set(name, { startTime, endTime: null, duration: null });
    return startTime;
  }

  /**
   * End timing an operation
   */
  endTiming(name) {
    const metric = this.metrics.get(name);
    if (!metric) {
      throw new Error(`No timing started for '${name}'`);
    }

    const endTime = performance.now();
    metric.endTime = endTime;
    metric.duration = endTime - metric.startTime;

    return metric.duration;
  }

  /**
   * Get timing for an operation
   */
  getTiming(name) {
    return this.metrics.get(name);
  }

  /**
   * Get all timings
   */
  getAllTimings() {
    return Object.fromEntries(this.metrics);
  }

  /**
   * Clear all timings
   */
  clearTimings() {
    this.metrics.clear();
  }

  /**
   * Monitor memory usage
   */
  getMemoryInfo() {
    if (performance.memory) {
      return {
        usedJSHeapSize: performance.memory.usedJSHeapSize,
        totalJSHeapSize: performance.memory.totalJSHeapSize,
        jsHeapSizeLimit: performance.memory.jsHeapSizeLimit,
      };
    }
    return null;
  }

  /**
   * Create a performance observer
   */
  observe(entryTypes, callback) {
    if ('PerformanceObserver' in window) {
      const observer = new PerformanceObserver(callback);
      observer.observe({ entryTypes });
      this.observers.set(entryTypes.join(','), observer);
      return observer;
    }
    return null;
  }

  /**
   * Disconnect all observers
   */
  disconnectObservers() {
    for (const observer of this.observers.values()) {
      observer.disconnect();
    }
    this.observers.clear();
  }
}

// Data validation utilities
export class DataValidator {
  /**
   * Validate tensor-like data
   */
  static validateTensorData(data, expectedShape = null) {
    if (
      !Array.isArray(data) &&
      !(data instanceof Float32Array) &&
      !(data instanceof Float64Array) &&
      !(data instanceof Int32Array)
    ) {
      throw new Error('Data must be an array or typed array');
    }

    if (expectedShape) {
      const expectedSize = expectedShape.reduce((acc, dim) => acc * dim, 1);
      if (data.length !== expectedSize) {
        throw new Error(`Data length ${data.length} doesn't match expected size ${expectedSize}`);
      }
    }

    // Check for NaN or Infinity values
    const hasInvalidValues = Array.from(data).some(
      val => !Number.isFinite(val) && !Number.isNaN(val)
    );

    if (hasInvalidValues) {
      console.warn('Data contains NaN or Infinity values');
    }

    return true;
  }

  /**
   * Validate model configuration
   */
  static validateModelConfig(config) {
    const requiredFields = ['modelType'];
    const missingFields = requiredFields.filter(field => !(field in config));

    if (missingFields.length > 0) {
      throw new Error(`Missing required fields: ${missingFields.join(', ')}`);
    }

    return true;
  }

  /**
   * Validate URL
   */
  static validateURL(url) {
    try {
      new URL(url);
      return true;
    } catch {
      throw new Error(`Invalid URL: ${url}`);
    }
  }

  /**
   * Validate file path
   */
  static validateFilePath(path) {
    if (typeof path !== 'string' || path.trim() === '') {
      throw new Error('File path must be a non-empty string');
    }

    // Basic path validation
    const invalidChars = /[<>:"|?*]/;
    if (invalidChars.test(path)) {
      throw new Error('File path contains invalid characters');
    }

    return true;
  }
}

// File utilities
export class FileUtils {
  /**
   * Download file from URL
   */
  static async downloadFile(url, options = {}) {
    const response = await fetch(url, options);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response;
  }

  /**
   * Download and parse JSON
   */
  static async downloadJSON(url, options = {}) {
    const response = await this.downloadFile(url, options);
    return await response.json();
  }

  /**
   * Download and get ArrayBuffer
   */
  static async downloadArrayBuffer(url, options = {}) {
    const response = await this.downloadFile(url, options);
    return await response.arrayBuffer();
  }

  /**
   * Download with progress tracking
   */
  static async downloadWithProgress(url, progressCallback, options = {}) {
    const response = await fetch(url, options);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const contentLength = parseInt(response.headers.get('content-length') || '0', 10);
    const reader = response.body?.getReader();

    if (!reader) {
      throw new Error('ReadableStream not supported');
    }

    const chunks = [];
    let receivedLength = 0;

    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      chunks.push(value);
      receivedLength += value.length;

      if (progressCallback) {
        progressCallback({
          loaded: receivedLength,
          total: contentLength,
          progress: contentLength > 0 ? receivedLength / contentLength : 0,
        });
      }
    }

    // Combine chunks
    const result = new Uint8Array(receivedLength);
    let position = 0;
    for (const chunk of chunks) {
      result.set(chunk, position);
      position += chunk.length;
    }

    return result;
  }

  /**
   * Read file as text (browser)
   */
  static readFileAsText(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(reader.error);
      reader.readAsText(file);
    });
  }

  /**
   * Read file as ArrayBuffer (browser)
   */
  static readFileAsArrayBuffer(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(reader.error);
      reader.readAsArrayBuffer(file);
    });
  }
}

// Math utilities
export class MathUtils {
  /**
   * Clamp value between min and max
   */
  static clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  /**
   * Linear interpolation
   */
  static lerp(a, b, t) {
    return a + (b - a) * t;
  }

  /**
   * Map value from one range to another
   */
  static map(value, inMin, inMax, outMin, outMax) {
    return ((value - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin;
  }

  /**
   * Generate random number with normal distribution
   */
  static randomNormal(mean = 0, std = 1) {
    // Box-Muller transform using class static properties
    if (!MathUtils._hasSpare) {
      MathUtils._hasSpare = false;
      MathUtils._spare = 0;
    }

    if (MathUtils._hasSpare) {
      MathUtils._hasSpare = false;
      return (MathUtils._spare * std) + mean;
    }

    MathUtils._hasSpare = true;
    const u = Math.random();
    const v = Math.random();
    const mag = std * Math.sqrt(-2.0 * Math.log(u));
    MathUtils._spare = mag * Math.cos(2.0 * Math.PI * v);
    return (mag * Math.sin(2.0 * Math.PI * v)) + mean;
  }

  /**
   * Calculate softmax
   */
  static softmax(values) {
    const max = Math.max(...values);
    const exp = values.map(x => Math.exp(x - max));
    const sum = exp.reduce((acc, x) => acc + x, 0);
    return exp.map(x => x / sum);
  }

  /**
   * Calculate log softmax
   */
  static logSoftmax(values) {
    const max = Math.max(...values);
    const logSum = Math.log(values.reduce((acc, x) => acc + Math.exp(x - max), 0));
    return values.map(x => x - max - logSum);
  }
}

// Async utilities
export class AsyncUtils {
  /**
   * Sleep for specified milliseconds
   */
  static sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Retry an async operation
   */
  static async retry(fn, maxAttempts = 3, delay = 1000) {
    let lastError;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error;
        if (attempt < maxAttempts) {
          await this.sleep(delay * attempt);
        }
      }
    }

    throw lastError;
  }

  /**
   * Execute functions with concurrency limit
   */
  static async mapWithConcurrency(items, fn, concurrency = 5) {
    const results = [];
    const executing = [];

    for (const item of items) {
      const promise = fn(item).then(result => {
        executing.splice(executing.indexOf(promise), 1);
        return result;
      });

      results.push(promise);
      executing.push(promise);

      if (executing.length >= concurrency) {
        await Promise.race(executing);
      }
    }

    return Promise.all(results);
  }

  /**
   * Timeout wrapper for promises
   */
  static withTimeout(promise, timeoutMs, timeoutMessage = 'Operation timed out') {
    return Promise.race([
      promise,
      new Promise((_, reject) => setTimeout(() => reject(new Error(timeoutMessage)), timeoutMs)),
    ]);
  }
}

export default {
  BrowserCompat,
  PerformanceMonitor,
  DataValidator,
  FileUtils,
  MathUtils,
  AsyncUtils,
};
