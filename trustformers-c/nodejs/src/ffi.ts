/**
 * FFI bindings for TrustformeRS C API
 */

import ffi from 'ffi-napi';
import ref from 'ref-napi';
import Struct from 'ref-struct-di';
import ArrayType from 'ref-array-di';
import { Library } from 'ffi-napi';
import path from 'path';
import os from 'os';
import { TrustformersError } from './types';

// Initialize ref-struct with ref-napi
const StructType = Struct(ref);
const ArrayTypeRef = ArrayType(ref);

// Define basic types
const int = ref.types.int;
const uint = ref.types.uint;
const float = ref.types.float;
const double = ref.types.double;
const uint64 = ref.types.uint64;
const charPtr = ref.refType(ref.types.char);
const voidPtr = ref.refType(ref.types.void);
const intPtr = ref.refType(ref.types.int);

// Define struct types for C API
export const TrustformersMemoryUsageStruct = StructType({
  total_memory_bytes: uint64,
  peak_memory_bytes: uint64,
  allocated_models: uint64,
  allocated_tokenizers: uint64,
  allocated_pipelines: uint64,
  allocated_tensors: uint64
});

export const TrustformersAdvancedMemoryUsageStruct = StructType({
  basic: TrustformersMemoryUsageStruct,
  fragmentation_ratio: double,
  avg_allocation_size: uint64,
  type_usage_json: charPtr,
  pressure_level: int,
  allocation_rate: double,
  deallocation_rate: double
});

export const TrustformersBuildInfoStruct = StructType({
  version: charPtr,
  features: charPtr,
  build_date: charPtr,
  target: charPtr
});

export const TrustformersPerformanceMetricsStruct = StructType({
  total_operations: uint64,
  avg_operation_time_ms: double,
  min_operation_time_ms: double,
  max_operation_time_ms: double,
  cache_hit_rate: double,
  performance_score: double,
  num_optimization_hints: int,
  optimization_hints_json: charPtr
});

export const TrustformersOptimizationConfigStruct = StructType({
  enable_tracking: int,
  enable_caching: int,
  cache_size_mb: int,
  num_threads: int,
  enable_simd: int,
  optimize_batch_size: int,
  memory_optimization_level: int
});

// Define pointer types
const TrustformersMemoryUsagePtr = ref.refType(TrustformersMemoryUsageStruct);
const TrustformersAdvancedMemoryUsagePtr = ref.refType(TrustformersAdvancedMemoryUsageStruct);
const TrustformersBuildInfoPtr = ref.refType(TrustformersBuildInfoStruct);
const TrustformersPerformanceMetricsPtr = ref.refType(TrustformersPerformanceMetricsStruct);
const TrustformersOptimizationConfigPtr = ref.refType(TrustformersOptimizationConfigStruct);
const charPtrPtr = ref.refType(charPtr);

// Function to get the library path based on platform
function getLibraryPath(): string {
  const platform = os.platform();
  const arch = os.arch();
  
  let libName: string;
  let libExt: string;
  
  switch (platform) {
    case 'win32':
      libName = 'trustformers_c';
      libExt = '.dll';
      break;
    case 'darwin':
      libName = 'libttrustformers_c';
      libExt = '.dylib';
      break;
    case 'linux':
    default:
      libName = 'libttrustformers_c';
      libExt = '.so';
      break;
  }
  
  // Try multiple possible paths
  const possiblePaths = [
    // Development paths (when building from source)
    path.join(__dirname, '..', '..', 'target', 'release', libName + libExt),
    path.join(__dirname, '..', '..', 'target', 'debug', libName + libExt),
    // Package paths (when installed via npm)
    path.join(__dirname, '..', 'native', platform, arch, libName + libExt),
    path.join(__dirname, '..', 'native', libName + libExt),
    // System paths
    libName + libExt,
  ];
  
  // Return the first path that exists or the last one as fallback
  for (const libPath of possiblePaths) {
    try {
      require.resolve(libPath);
      return libPath;
    } catch {
      continue;
    }
  }
  
  return possiblePaths[possiblePaths.length - 1];
}

// Define the FFI library interface
export interface TrustformersFFI extends Library {
  // Core API functions
  trustformers_init: () => TrustformersError;
  trustformers_cleanup: () => TrustformersError;
  trustformers_version: () => charPtr;
  trustformers_build_info: (info: TrustformersBuildInfoPtr) => TrustformersError;
  trustformers_free_string: (ptr: charPtr) => void;
  trustformers_has_feature: (feature: charPtr) => int;
  trustformers_set_log_level: (level: int) => TrustformersError;
  trustformers_gc: () => TrustformersError;

  // Memory management functions
  trustformers_get_memory_usage: (usage: TrustformersMemoryUsagePtr) => TrustformersError;
  trustformers_get_advanced_memory_usage: (usage: TrustformersAdvancedMemoryUsagePtr) => TrustformersError;
  trustformers_advanced_memory_usage_free: (usage: TrustformersAdvancedMemoryUsagePtr) => void;
  trustformers_memory_cleanup: () => TrustformersError;
  trustformers_set_memory_limits: (maxMemoryMb: uint64, warningThresholdMb: uint64) => TrustformersError;
  trustformers_check_memory_leaks: (leakReport: charPtrPtr) => TrustformersError;

  // Performance monitoring functions
  trustformers_get_performance_metrics: (metrics: TrustformersPerformanceMetricsPtr) => TrustformersError;
  trustformers_apply_optimizations: (config: TrustformersOptimizationConfigPtr) => TrustformersError;
  trustformers_start_profiling: () => TrustformersError;
  trustformers_stop_profiling: (report: charPtrPtr) => TrustformersError;
  trustformers_performance_metrics_free: (metrics: TrustformersPerformanceMetricsPtr) => void;

  // Model functions
  trustformers_model_load: (modelPath: charPtr, modelId: intPtr) => TrustformersError;
  trustformers_model_load_with_config: (modelPath: charPtr, configJson: charPtr, modelId: intPtr) => TrustformersError;
  trustformers_model_get_metadata: (modelId: int, metadata: charPtrPtr) => TrustformersError;
  trustformers_model_get_config: (modelId: int, config: charPtrPtr) => TrustformersError;
  trustformers_model_quantize: (modelId: int, quantizationType: charPtr, outputPath: charPtr) => TrustformersError;
  trustformers_model_validate: (modelId: int, validationReport: charPtrPtr) => TrustformersError;
  trustformers_model_free: (modelId: int) => TrustformersError;

  // Tokenizer functions
  trustformers_tokenizer_load: (tokenizerPath: charPtr, tokenizerId: intPtr) => TrustformersError;
  trustformers_tokenizer_load_with_config: (tokenizerPath: charPtr, configJson: charPtr, tokenizerId: intPtr) => TrustformersError;
  trustformers_tokenizer_encode: (tokenizerId: int, text: charPtr, options: charPtr, result: charPtrPtr) => TrustformersError;
  trustformers_tokenizer_encode_batch: (tokenizerId: int, texts: charPtrPtr, numTexts: int, options: charPtr, result: charPtrPtr) => TrustformersError;
  trustformers_tokenizer_decode: (tokenizerId: int, tokenIds: intPtr, numTokens: int, options: charPtr, result: charPtrPtr) => TrustformersError;
  trustformers_tokenizer_decode_batch: (tokenizerId: int, tokenIdsArray: intPtr, batchSize: int, options: charPtr, result: charPtrPtr) => TrustformersError;
  trustformers_tokenizer_get_vocab_size: (tokenizerId: int, vocabSize: intPtr) => TrustformersError;
  trustformers_tokenizer_get_vocab: (tokenizerId: int, vocab: charPtrPtr) => TrustformersError;
  trustformers_tokenizer_add_special_token: (tokenizerId: int, token: charPtr, tokenId: int) => TrustformersError;
  trustformers_tokenizer_train: (texts: charPtrPtr, numTexts: int, configJson: charPtr, outputPath: charPtr) => TrustformersError;
  trustformers_tokenizer_free: (tokenizerId: int) => TrustformersError;

  // Pipeline functions
  trustformers_pipeline_create: (task: charPtr, modelPath: charPtr, tokenizerPath: charPtr, configJson: charPtr, pipelineId: intPtr) => TrustformersError;
  trustformers_pipeline_create_with_model: (task: charPtr, modelId: int, tokenizerId: int, configJson: charPtr, pipelineId: intPtr) => TrustformersError;
  trustformers_pipeline_run: (pipelineId: int, input: charPtr, configJson: charPtr, result: charPtrPtr) => TrustformersError;
  trustformers_pipeline_run_batch: (pipelineId: int, inputs: charPtrPtr, numInputs: int, configJson: charPtr, result: charPtrPtr) => TrustformersError;
  trustformers_pipeline_stream: (pipelineId: int, input: charPtr, configJson: charPtr, callbackFn: voidPtr, callbackData: voidPtr) => TrustformersError;
  trustformers_pipeline_conversation_start: (pipelineId: int, systemPrompt: charPtr, conversationId: charPtrPtr) => TrustformersError;
  trustformers_pipeline_conversation_add_turn: (pipelineId: int, conversationId: charPtr, role: charPtr, content: charPtr) => TrustformersError;
  trustformers_pipeline_conversation_generate: (pipelineId: int, conversationId: charPtr, configJson: charPtr, result: charPtrPtr) => TrustformersError;
  trustformers_pipeline_get_stats: (pipelineId: int, stats: charPtrPtr) => TrustformersError;
  trustformers_pipeline_free: (pipelineId: int) => TrustformersError;
}

// Create the FFI library instance
let _library: TrustformersFFI | null = null;

export function getLibrary(): TrustformersFFI {
  if (!_library) {
    const libPath = getLibraryPath();
    
    try {
      _library = ffi.Library(libPath, {
        // Core API functions
        trustformers_init: [int, []],
        trustformers_cleanup: [int, []],
        trustformers_version: [charPtr, []],
        trustformers_build_info: [int, [TrustformersBuildInfoPtr]],
        trustformers_free_string: ['void', [charPtr]],
        trustformers_has_feature: [int, [charPtr]],
        trustformers_set_log_level: [int, [int]],
        trustformers_gc: [int, []],

        // Memory management functions
        trustformers_get_memory_usage: [int, [TrustformersMemoryUsagePtr]],
        trustformers_get_advanced_memory_usage: [int, [TrustformersAdvancedMemoryUsagePtr]],
        trustformers_advanced_memory_usage_free: ['void', [TrustformersAdvancedMemoryUsagePtr]],
        trustformers_memory_cleanup: [int, []],
        trustformers_set_memory_limits: [int, [uint64, uint64]],
        trustformers_check_memory_leaks: [int, [charPtrPtr]],

        // Performance monitoring functions
        trustformers_get_performance_metrics: [int, [TrustformersPerformanceMetricsPtr]],
        trustformers_apply_optimizations: [int, [TrustformersOptimizationConfigPtr]],
        trustformers_start_profiling: [int, []],
        trustformers_stop_profiling: [int, [charPtrPtr]],
        trustformers_performance_metrics_free: ['void', [TrustformersPerformanceMetricsPtr]],

        // Model functions
        trustformers_model_load: [int, [charPtr, intPtr]],
        trustformers_model_load_with_config: [int, [charPtr, charPtr, intPtr]],
        trustformers_model_get_metadata: [int, [int, charPtrPtr]],
        trustformers_model_get_config: [int, [int, charPtrPtr]],
        trustformers_model_quantize: [int, [int, charPtr, charPtr]],
        trustformers_model_validate: [int, [int, charPtrPtr]],
        trustformers_model_free: [int, [int]],

        // Tokenizer functions
        trustformers_tokenizer_load: [int, [charPtr, intPtr]],
        trustformers_tokenizer_load_with_config: [int, [charPtr, charPtr, intPtr]],
        trustformers_tokenizer_encode: [int, [int, charPtr, charPtr, charPtrPtr]],
        trustformers_tokenizer_encode_batch: [int, [int, charPtrPtr, int, charPtr, charPtrPtr]],
        trustformers_tokenizer_decode: [int, [int, intPtr, int, charPtr, charPtrPtr]],
        trustformers_tokenizer_decode_batch: [int, [int, intPtr, int, charPtr, charPtrPtr]],
        trustformers_tokenizer_get_vocab_size: [int, [int, intPtr]],
        trustformers_tokenizer_get_vocab: [int, [int, charPtrPtr]],
        trustformers_tokenizer_add_special_token: [int, [int, charPtr, int]],
        trustformers_tokenizer_train: [int, [charPtrPtr, int, charPtr, charPtr]],
        trustformers_tokenizer_free: [int, [int]],

        // Pipeline functions
        trustformers_pipeline_create: [int, [charPtr, charPtr, charPtr, charPtr, intPtr]],
        trustformers_pipeline_create_with_model: [int, [charPtr, int, int, charPtr, intPtr]],
        trustformers_pipeline_run: [int, [int, charPtr, charPtr, charPtrPtr]],
        trustformers_pipeline_run_batch: [int, [int, charPtrPtr, int, charPtr, charPtrPtr]],
        trustformers_pipeline_stream: [int, [int, charPtr, charPtr, voidPtr, voidPtr]],
        trustformers_pipeline_conversation_start: [int, [int, charPtr, charPtrPtr]],
        trustformers_pipeline_conversation_add_turn: [int, [int, charPtr, charPtr, charPtr]],
        trustformers_pipeline_conversation_generate: [int, [int, charPtr, charPtr, charPtrPtr]],
        trustformers_pipeline_get_stats: [int, [int, charPtrPtr]],
        trustformers_pipeline_free: [int, [int]]
      }) as TrustformersFFI;
    } catch (error) {
      throw new Error(
        `Failed to load TrustformeRS native library from ${libPath}. ` +
        `Make sure the library is built and available. Error: ${error.message}`
      );
    }
  }
  
  return _library;
}

// Utility functions for memory management
export function allocCString(str: string): Buffer {
  return Buffer.from(str + '\0', 'utf8');
}

export function readCString(ptr: any): string {
  if (ref.isNull(ptr)) {
    return '';
  }
  return ref.readCString(ptr, 0);
}

export function freeCString(ptr: any): void {
  if (!ref.isNull(ptr)) {
    getLibrary().trustformers_free_string(ptr);
  }
}

// Error handling utility
export function checkError(errorCode: TrustformersError, operation: string): void {
  if (errorCode !== TrustformersError.Success) {
    const errorName = TrustformersError[errorCode] || 'Unknown';
    throw new Error(`TrustformeRS ${operation} failed with error: ${errorName} (${errorCode})`);
  }
}

// Initialize the library when this module is imported
let _initialized = false;

export function ensureInitialized(): void {
  if (!_initialized) {
    const lib = getLibrary();
    const result = lib.trustformers_init();
    checkError(result, 'initialization');
    _initialized = true;
    
    // Setup cleanup on process exit
    process.on('exit', () => {
      if (_initialized) {
        lib.trustformers_cleanup();
        _initialized = false;
      }
    });
    
    process.on('SIGINT', () => {
      if (_initialized) {
        lib.trustformers_cleanup();
        _initialized = false;
      }
      process.exit(0);
    });
    
    process.on('SIGTERM', () => {
      if (_initialized) {
        lib.trustformers_cleanup();
        _initialized = false;
      }
      process.exit(0);
    });
  }
}

// Export struct types for external use
export {
  TrustformersMemoryUsageStruct,
  TrustformersAdvancedMemoryUsageStruct,
  TrustformersBuildInfoStruct,
  TrustformersPerformanceMetricsStruct,
  TrustformersOptimizationConfigStruct
};