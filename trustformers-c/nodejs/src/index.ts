/**
 * TrustformeRS Node.js Bindings
 * 
 * High-performance transformer library with C API bindings for Node.js
 * Provides TypeScript-first interface with full type safety and memory management
 */

// Re-export all types
export * from './types';

// Re-export the main API classes and functions
export {
  TrustformeRS,
  Model,
  Tokenizer,
  Pipeline,
  trustformers
} from './api';

// Re-export FFI utilities for advanced users
export {
  getLibrary,
  ensureInitialized,
  TrustformersMemoryUsageStruct,
  TrustformersAdvancedMemoryUsageStruct,
  TrustformersBuildInfoStruct,
  TrustformersPerformanceMetricsStruct,
  TrustformersOptimizationConfigStruct
} from './ffi';

// Default export is the main API instance
export { trustformers as default } from './api';

// Version information
export const VERSION = require('../package.json').version;

// Library information
export const LIBRARY_INFO = {
  name: 'TrustformeRS Node.js Bindings',
  version: VERSION,
  description: 'High-performance transformer library with C API bindings for Node.js',
  repository: 'https://github.com/cool-japan/trustformers',
  license: 'MIT',
  author: 'TrustformeRS Team'
};

// Convenience functions for common use cases

/**
 * Quick setup function for common configurations
 */
export function quickSetup(options: {
  logLevel?: number;
  memoryLimitMb?: number;
  enableProfiling?: boolean;
  optimizationLevel?: number;
} = {}) {
  const api = require('./api').trustformers;
  
  // Set log level
  if (options.logLevel !== undefined) {
    api.setLogLevel(options.logLevel);
  }
  
  // Set memory limits
  if (options.memoryLimitMb) {
    api.setMemoryLimits(options.memoryLimitMb, Math.floor(options.memoryLimitMb * 0.8));
  }
  
  // Enable profiling
  if (options.enableProfiling) {
    api.startProfiling();
  }
  
  // Apply optimization configuration
  if (options.optimizationLevel !== undefined) {
    api.applyOptimizations({
      enableTracking: true,
      enableCaching: true,
      cacheSizeMb: 256,
      numThreads: 0, // Auto-detect
      enableSimd: true,
      optimizeBatchSize: true,
      memoryOptimizationLevel: options.optimizationLevel
    });
  }
  
  return api;
}

/**
 * Create a simple text generation pipeline
 */
export async function createTextGenerationPipeline(modelPath: string, options: {
  tokenizerPath?: string;
  device?: 'cpu' | 'cuda' | 'rocm' | 'auto';
  maxLength?: number;
  batchSize?: number;
} = {}) {
  const { Pipeline } = require('./api');
  
  return new Pipeline({
    task: 'text-generation',
    model: {
      modelPath,
      tokenizerPath: options.tokenizerPath,
      device: options.device || 'auto',
      maxLength: options.maxLength || 2048,
      batchSize: options.batchSize || 1
    }
  });
}

/**
 * Create a simple text classification pipeline
 */
export async function createClassificationPipeline(modelPath: string, options: {
  tokenizerPath?: string;
  device?: 'cpu' | 'cuda' | 'rocm' | 'auto';
  batchSize?: number;
} = {}) {
  const { Pipeline } = require('./api');
  
  return new Pipeline({
    task: 'text-classification',
    model: {
      modelPath,
      tokenizerPath: options.tokenizerPath,
      device: options.device || 'auto',
      batchSize: options.batchSize || 1
    }
  });
}

/**
 * Create a question answering pipeline
 */
export async function createQuestionAnsweringPipeline(modelPath: string, options: {
  tokenizerPath?: string;
  device?: 'cpu' | 'cuda' | 'rocm' | 'auto';
  batchSize?: number;
} = {}) {
  const { Pipeline } = require('./api');
  
  return new Pipeline({
    task: 'question-answering',
    model: {
      modelPath,
      tokenizerPath: options.tokenizerPath,
      device: options.device || 'auto',
      batchSize: options.batchSize || 1
    }
  });
}

/**
 * Load a tokenizer with common configurations
 */
export function loadTokenizer(tokenizerPath: string, options: {
  addSpecialTokens?: boolean;
  padding?: boolean | 'max_length';
  truncation?: boolean;
  maxLength?: number;
} = {}) {
  const { Tokenizer } = require('./api');
  
  return new Tokenizer({
    tokenizerPath,
    addSpecialTokens: options.addSpecialTokens ?? true,
    padding: options.padding ?? false,
    truncation: options.truncation ?? true,
    maxLength: options.maxLength ?? 512
  });
}

/**
 * Load a model with common configurations
 */
export function loadModel(modelPath: string, options: {
  tokenizerPath?: string;
  device?: 'cpu' | 'cuda' | 'rocm' | 'auto';
  quantization?: 'fp16' | 'int8' | 'int4' | 'none';
  numThreads?: number;
} = {}) {
  const { Model } = require('./api');
  
  return new Model({
    modelPath,
    tokenizerPath: options.tokenizerPath,
    device: options.device || 'auto',
    quantization: options.quantization || 'none',
    numThreads: options.numThreads || 0
  });
}

/**
 * Utility function to get library status
 */
export function getLibraryStatus() {
  const api = require('./api').trustformers;
  
  return {
    version: api.getVersion(),
    buildInfo: api.getBuildInfo(),
    capabilities: api.getCapabilities(),
    platformInfo: api.getPlatformInfo(),
    memoryUsage: api.getMemoryUsage(),
    performanceMetrics: api.getPerformanceMetrics()
  };
}

/**
 * Utility function to benchmark the library
 */
export async function benchmark(options: {
  modelPath?: string;
  inputText?: string;
  iterations?: number;
  warmupIterations?: number;
} = {}) {
  const api = require('./api').trustformers;
  
  const modelPath = options.modelPath || 'gpt2';
  const inputText = options.inputText || 'Hello, world!';
  const iterations = options.iterations || 10;
  const warmupIterations = options.warmupIterations || 3;
  
  // Start profiling
  api.startProfiling();
  
  try {
    // Create pipeline
    const pipeline = await createTextGenerationPipeline(modelPath);
    
    // Warmup
    for (let i = 0; i < warmupIterations; i++) {
      await pipeline.run(inputText, { maxLength: 50 });
    }
    
    // Benchmark
    const startTime = Date.now();
    const results = [];
    
    for (let i = 0; i < iterations; i++) {
      const iterStart = Date.now();
      const result = await pipeline.run(inputText, { maxLength: 50 });
      const iterEnd = Date.now();
      
      results.push({
        iteration: i + 1,
        time: iterEnd - iterStart,
        result
      });
    }
    
    const endTime = Date.now();
    const totalTime = endTime - startTime;
    const avgTime = totalTime / iterations;
    
    // Get profiling report
    const profilingReport = api.stopProfiling();
    
    // Cleanup
    pipeline.dispose();
    
    return {
      totalTime,
      averageTime: avgTime,
      iterations,
      results,
      profilingReport,
      memoryUsage: api.getMemoryUsage(),
      performanceMetrics: api.getPerformanceMetrics()
    };
  } catch (error) {
    // Stop profiling on error
    try {
      api.stopProfiling();
    } catch {
      // Ignore profiling stop errors
    }
    throw error;
  }
}

// Export error types for error handling
export { TrustformersNativeError } from './types';