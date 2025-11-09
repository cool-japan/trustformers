/**
 * TrustformeRS JavaScript API (Refactored)
 * High-level JavaScript interface for the TrustformeRS WebAssembly library
 * Modularized for better maintainability and compliance with 2000-line policy
 */

// Import core modules
import { enhanced_tensor_ops, enhanced_tensor_utils } from './enhanced-tensor-ops.js';
import { enhanced_inference } from './enhanced-inference.js';
import { devTools } from './dev-tools.js';
import { tensor_utils, async_utils } from './tensor-utils.js';
import { nodejs } from './nodejs-support.js';
import { webgpu, setWasmModule } from './webgpu-support.js';

// Import existing modules
import { WebGLBackend, createWebGLBackend, initWebGLBackend } from './webgl-backend.js';
import {
  TensorMemoryPool,
  WebGLMemoryPool,
  MemoryManager,
  getMemoryManager,
  withMemoryManagement,
  initMemoryPool
} from './memory-pool.js';
import {
  PerformanceProfiler,
  getProfiler,
  profile,
  profileMethod,
  initProfiler
} from './performance-profiler.js';
import {
  ZeroCopyTensorView,
  ZeroCopyBufferManager,
  getBufferManager,
  createZeroCopyTensor,
  wrapTensor,
  transferTensor,
  batchTransferTensors,
  initZeroCopy
} from './zero-copy.js';
import {
  DebugUtilities,
  debugUtils,
  debug,
  debugOperation
} from './debug-utilities.js';
import {
  TensorInspector,
  tensorInspector,
  inspect
} from './tensor-inspector.js';
import {
  ModelVisualizer,
  modelVisualizer,
  visualize
} from './model-visualization.js';
import {
  ErrorDiagnostics,
  errorDiagnostics,
  diagnose,
  handleErrors,
  ErrorTypes,
  ErrorSeverity
} from './error-diagnostics.js';
import {
  ImageClassifier,
  ImagePreprocessor,
  StreamProcessor,
  MultiModalProcessor,
  ImageUtils
} from './image-processing.js';

// Import new advanced modules
import {
  WebNNBackend,
  WebNNBackendManager,
  WebNNCapabilities,
  WebNNGraph,
  WebNNOperations,
  getWebNNManager,
  initWebNN,
  isWebNNAvailable
} from './webnn-backend.js';

import {
  QuantizationType,
  QuantizationScheme,
  QuantizationGranularity,
  Float16Utils,
  Int8Quantizer,
  Int4Quantizer,
  GGMLQuantizer,
  QuantizationCalibrator,
  MixedPrecisionQuantizer,
  QuantizationAwareTraining
} from './advanced-quantization.js';

import {
  BenchmarkConfig,
  BenchmarkSuite,
  TensorBenchmarks,
  ModelBenchmarks,
  BackendComparison,
  MemoryBenchmark,
  PerformanceStats
} from './benchmark-suite.js';

import {
  LRUCache,
  TTLCache,
  LFUCache,
  MultiLevelCache,
  PersistentCache,
  CacheManager
} from './advanced-caching.js';

// Import NEW advanced AI modules
import {
  GradientCheckpointingManager,
  MixedPrecisionManager,
  GradientAccumulationManager,
  LARSOptimizer,
  LookaheadOptimizer,
  SAMOptimizer,
  OptimizationStrategySelector,
  AdvancedOptimizer,
  createAdvancedOptimizer
} from './advanced-optimization.js';

import {
  FederatedClient,
  SecureAggregationProtocol,
  DifferentialPrivacyMechanism,
  ClientSelectionStrategy,
  ByzantineRobustAggregation,
  FederatedServer,
  createFederatedLearning
} from './federated-learning.js';

import {
  SearchSpace,
  PerformanceEstimator,
  RandomSearch,
  EvolutionarySearch,
  MultiObjectiveNAS,
  NASController,
  createNAS
} from './neural-architecture-search.js';

import {
  DistillationLoss,
  TeacherModel,
  StudentModel,
  DistillationTrainer,
  ProgressiveDistillation,
  SelfDistillation,
  DistillationController,
  createDistillation
} from './knowledge-distillation.js';

import {
  BaseStreamHandler,
  TextStreamHandler,
  ImageStreamHandler,
  AudioStreamHandler,
  MultiModalStreamCoordinator,
  createMultiModalStreaming
} from './multimodal-streaming.js';

import {
  ONNXRuntimeWrapper,
  ONNXModelConverter,
  ONNXModelAnalyzer,
  ONNXController,
  createONNXIntegration
} from './onnx-integration.js';

import {
  AttentionVisualizer,
  GradientExplainer,
  FeatureImportanceAnalyzer,
  InterpretabilityController,
  createInterpretability
} from './model-interpretability.js';

import {
  PerformanceProfiler as AutoPerformanceProfiler,
  BottleneckDetector,
  MLBasedOptimizer,
  AutoPerformanceOptimizer,
  createAutoOptimizer
} from './auto-performance-optimizer.js';

// Global state
let wasmModule = null;
let initialized = false;
let webglBackend = null;
let memoryManager = null;
let performanceProfiler = null;

// Capabilities tracking
const capabilities = {
  wasm: false,
  webgl: false,
  webgpu: false,
  memoryPool: false,
  profiling: false,
  zeroCopy: false
};

/**
 * Initialize the TrustformeRS WASM module
 * @param {Object} options - Initialization options
 * @param {string} options.wasmPath - Path to the WASM file
 * @param {boolean} options.initPanicHook - Whether to initialize panic hook for better error messages
 * @returns {Promise<void>}
 */
export async function initialize(options = {}) {
  if (initialized) {
    console.warn('TrustformeRS already initialized');
    return;
  }

  const { wasmPath = './trustformers_wasm_bg.wasm', initPanicHook = true } = options;

  try {
    // Dynamically import the WASM module
    const { default: init, ...exports } = await import('../pkg/trustformers_wasm.js');

    // Initialize WASM
    await init(wasmPath);

    // Store exports
    wasmModule = exports;

    // Set WASM module reference for WebGPU support
    setWasmModule(wasmModule);

    // Initialize panic hook for better error messages
    if (initPanicHook && wasmModule.init_panic_hook) {
      wasmModule.init_panic_hook();
    }

    // Check SIMD support
    const simdEnabled = wasmModule.enable_simd();
    console.warn(`TrustformeRS initialized. SIMD support: ${simdEnabled ? 'enabled' : 'disabled'}`);

    // Log version and features
    console.warn(`Version: ${wasmModule.version()}`);
    console.warn(`Features:`, wasmModule.features());

    initialized = true;
    capabilities.wasm = true;
  } catch (error) {
    // Graceful fallback for development environments without WASM
    console.warn(`WASM module not available - running in mock mode: ${error.message}`);

    // Initialize without WASM for development
    wasmModule = null;
    initialized = true;
    capabilities.wasm = false;

    // Still enable other capabilities
    capabilities.tensor = true;
    capabilities.models = false; // Models require WASM
    capabilities.pipeline = false; // Pipelines require WASM
    capabilities.webgl = true;
    capabilities.webgpu = true;
    capabilities.nodejs = typeof process !== 'undefined';
    capabilities.browser = typeof window !== 'undefined';
  }
}

/**
 * Enhanced initialization with performance optimizations
 * @param {Object} options - Enhanced initialization options
 * @returns {Promise<Object>} Initialization result with capabilities
 */
export async function initializeEnhanced(options = {}) {
  const {
    wasmPath = './trustformers_wasm_bg.wasm',
    initPanicHook = true,
    enableWebGL = true,
    enableMemoryPool = true,
    enableProfiling = true,
    enableZeroCopy = true,
    webglOptions = {},
    memoryOptions = {},
    profilingOptions = {},
    zeroCopyOptions = {}
  } = options;

  // Basic WASM initialization
  await initialize({ wasmPath, initPanicHook });

  // Initialize performance optimizations
  if (enableMemoryPool) {
    try {
      initMemoryPool(wasmModule);
      memoryManager = getMemoryManager(memoryOptions);
      capabilities.memoryPool = true;
      console.warn('✓ Memory pooling enabled');
    } catch (error) {
      console.warn('Memory pooling initialization failed:', error);
    }
  }

  if (enableProfiling) {
    try {
      initProfiler(wasmModule);
      performanceProfiler = getProfiler(profilingOptions);
      capabilities.profiling = true;
      console.warn('✓ Performance profiling enabled');
    } catch (error) {
      console.warn('Performance profiling initialization failed:', error);
    }
  }

  if (enableZeroCopy) {
    try {
      initZeroCopy(wasmModule);
      capabilities.zeroCopy = true;
      console.warn('✓ Zero-copy transfers enabled');
    } catch (error) {
      console.warn('Zero-copy initialization failed:', error);
    }
  }

  if (enableWebGL) {
    try {
      initWebGLBackend(wasmModule);
      webglBackend = await createWebGLBackend(webglOptions.canvas);

      if (memoryManager) {
        memoryManager.initWebGL(webglBackend.gl, webglOptions.memory);
      }

      capabilities.webgl = true;
      console.warn('✓ WebGL backend enabled');
      console.warn('  WebGL Info:', webglBackend.getInfo());
    } catch (error) {
      console.warn('WebGL backend initialization failed:', error);
    }
  }

  // Check WebGPU availability
  if (webgpu.isAvailable()) {
    capabilities.webgpu = true;
    console.warn('✓ WebGPU available');
  }

  console.warn('\nTrustformeRS Enhanced Initialization Complete');
  console.warn('Capabilities:', capabilities);

  // Make global references available
  global.webglBackend = webglBackend;
  global.performanceProfiler = performanceProfiler;
  global.memoryManager = memoryManager;

  return capabilities;
}

/**
 * Ensure the module is initialized
 * @private
 */
function ensureInitialized() {
  if (!initialized || !wasmModule) {
    throw new Error('TrustformeRS not initialized. Call initialize() first.');
  }
}

// Core tensor creation functions
export function tensor(data, shape) {
  ensureInitialized();
  const flatData = new Float32Array(data);
  const shapeArray = new Uint32Array(shape);
  return new wasmModule.WasmTensor(flatData, shapeArray);
}

export function zeros(shape) {
  ensureInitialized();
  return wasmModule.WasmTensor.zeros(new Uint32Array(shape));
}

export function ones(shape) {
  ensureInitialized();
  return wasmModule.WasmTensor.ones(new Uint32Array(shape));
}

export function eye(size) {
  ensureInitialized();
  const shape = [size, size];
  const totalElements = size * size;
  const data = new Float32Array(totalElements);

  // Fill diagonal with ones
  for (let i = 0; i < size; i++) {
    data[i * size + i] = 1.0;
  }

  return wasmModule.WasmTensor.from_array(data, new Uint32Array(shape));
}

// Enhanced tensor creation with probabilistic features
export async function randn(shape, options = {}) {
  ensureInitialized();

  try {
    const { create_advanced_random_tensor } = await import('./probabilistic-tensors.js');
    const { mean = 0, std = 1, seed } = options;

    const distribution_config = {
      type: 'normal',
      mean,
      std
    };

    const enhanced_tensor = create_advanced_random_tensor(shape, distribution_config, seed);
    const wasm_tensor = wasmModule.WasmTensor.from_array(enhanced_tensor.data, new Uint32Array(shape));

    wasm_tensor._enhanced_metadata = {
      distribution_config: enhanced_tensor.distribution_config,
      entropy: enhanced_tensor.entropy,
      creation_timestamp: enhanced_tensor.creation_timestamp,
      statistical_quality: 'high'
    };

    return wasm_tensor;
  } catch (error) {
    console.warn('Failed to create enhanced random tensor, falling back to basic:', error.message);
    return wasmModule.WasmTensor.randn(new Uint32Array(shape));
  }
}

export async function create_advanced_tensor(shape, distribution_config, seed = null) {
  ensureInitialized();

  try {
    // Try SciRS2-enhanced probabilistic tensors first
    const { create_advanced_random_tensor } = await import('./enhanced-probabilistic-tensors.js');
    const enhanced_tensor = create_advanced_random_tensor(shape, distribution_config, seed, {
      enableSciRS2: true,
      qualityLevel: 'research',
      statisticalValidation: true
    });

    // If it's already a SciRS2Tensor, return it directly
    if (enhanced_tensor._probabilistic_metadata?.creation_method === 'scirs2_enhanced') {
      return enhanced_tensor;
    }

    // Otherwise, wrap in WASM tensor
    const wasm_tensor = wasmModule.WasmTensor.from_array(enhanced_tensor.data, new Uint32Array(shape));
    wasm_tensor._enhanced_metadata = enhanced_tensor;
    wasm_tensor._probabilistic_properties = true;
    wasm_tensor._scirs2_enhanced = true;

    return wasm_tensor;
  } catch (error) {
    console.warn('Failed to create SciRS2-enhanced tensor, trying fallback:', error.message);

    try {
      // Fallback to basic probabilistic tensors
      const { create_advanced_random_tensor } = await import('./probabilistic-tensors.js');
      const enhanced_tensor = create_advanced_random_tensor(shape, distribution_config, seed);
      const wasm_tensor = wasmModule.WasmTensor.from_array(enhanced_tensor.data, new Uint32Array(shape));

      wasm_tensor._enhanced_metadata = enhanced_tensor;
      wasm_tensor._probabilistic_properties = true;

      return wasm_tensor;
    } catch (fallbackError) {
      console.warn('All probabilistic tensor creation methods failed, using basic:', fallbackError.message);
      return wasmModule.WasmTensor.zeros(new Uint32Array(shape));
    }
  }
}

export async function create_bayesian_tensor(shape, prior_config, seed = null) {
  ensureInitialized();

  try {
    const { ProbabilisticTensorOperations } = await import('./probabilistic-tensors.js');
    const prob_system = new ProbabilisticTensorOperations(seed);

    let distribution_config;
    switch (prior_config.type) {
      case 'normal_normal':
        distribution_config = {
          type: 'normal',
          mean: prior_config.prior_mean || 0,
          std: Math.sqrt(prior_config.prior_variance || 1)
        };
        break;
      case 'beta_binomial':
        distribution_config = {
          type: 'beta_bernoulli',
          alpha: prior_config.alpha || 1,
          beta: prior_config.beta || 1
        };
        break;
      default:
        distribution_config = { type: 'normal', mean: 0, std: 1 };
    }

    const tensor = prob_system.create_probabilistic_tensor(shape, distribution_config);
    const wasm_tensor = wasmModule.WasmTensor.from_array(tensor.data, new Uint32Array(shape));

    wasm_tensor._bayesian_config = prior_config;
    wasm_tensor._probabilistic_system = prob_system;
    wasm_tensor._is_bayesian = true;

    wasm_tensor.update_with_observations = async function(observed_data) {
      const inference_result = prob_system.perform_bayesian_inference(observed_data, prior_config);
      this._posterior_samples = inference_result.posterior_samples;
      this._credible_intervals = inference_result.credible_intervals;
      this._evidence = inference_result.evidence_estimate;
      return inference_result;
    };

    return wasm_tensor;
  } catch (error) {
    console.warn('Failed to create Bayesian tensor, falling back to normal tensor:', error);
    return wasmModule.WasmTensor.randn(new Uint32Array(shape));
  }
}

export async function analyze_tensor(tensor, seed = null, options = {}) {
  try {
    // Try SciRS2-enhanced analysis first
    const { analyze_tensor_statistics } = await import('./enhanced-probabilistic-tensors.js');

    let tensor_data = tensor.data || tensor;
    if (tensor_data instanceof Float32Array || tensor_data instanceof Array) {
      tensor_data = { data: tensor_data };
    }

    const analysis = analyze_tensor_statistics(tensor_data, {
      includeAdvancedStats: true,
      includeBayesianAnalysis: true,
      includeInformationTheory: true,
      includeQualityMetrics: true,
      seed,
      ...options
    });

    if (tensor && typeof tensor === 'object' && !Array.isArray(tensor)) {
      tensor._statistical_analysis = analysis;
      tensor._analysis_timestamp = Date.now();
      tensor._analysis_method = 'scirs2_enhanced';
    }

    return analysis;
  } catch (error) {
    console.warn('Failed to use SciRS2-enhanced analysis, trying fallback:', error.message);

    try {
      // Fallback to basic probabilistic analysis
      const { analyze_tensor_statistics } = await import('./probabilistic-tensors.js');

      let tensor_data = tensor.data || tensor;
      if (tensor_data instanceof Float32Array || tensor_data instanceof Array) {
        tensor_data = { data: tensor_data };
      }

      const analysis = analyze_tensor_statistics(tensor_data, seed);

      if (tensor && typeof tensor === 'object' && !Array.isArray(tensor)) {
        tensor._statistical_analysis = analysis;
        tensor._analysis_timestamp = Date.now();
        tensor._analysis_method = 'basic_fallback';
      }

      return analysis;
    } catch (fallbackError) {
      console.warn('All tensor analysis methods failed:', fallbackError.message);
      return null;
    }
  }
}

// Model and Pipeline enums
export const ModelArchitecture = {
  BERT: 0,
  GPT2: 1,
  T5: 2,
  LLAMA: 3,
  MISTRAL: 4
};

export const PipelineType = {
  TEXT_GENERATION: 0,
  TEXT_CLASSIFICATION: 1,
  TOKEN_CLASSIFICATION: 2,
  QUESTION_ANSWERING: 3,
  SUMMARIZATION: 4,
  TRANSLATION: 5
};

export const TokenizerType = {
  WORDPIECE: 0,
  BPE: 1,
  SENTENCEPIECE: 2
};

// Model and tokenizer creation functions
export function createModelConfig(modelType) {
  ensureInitialized();

  const configFactories = {
    'bert_base': wasmModule.ModelConfig.bert_base,
    'gpt2_base': wasmModule.ModelConfig.gpt2_base,
    't5_small': wasmModule.ModelConfig.t5_small,
    'llama_7b': wasmModule.ModelConfig.llama_7b,
    'mistral_7b': wasmModule.ModelConfig.mistral_7b
  };

  const factory = configFactories[modelType];
  if (!factory) {
    throw new Error(`Unknown model type: ${modelType}. Available: ${Object.keys(configFactories).join(', ')}`);
  }

  return factory();
}

export function createModel(configOrType) {
  ensureInitialized();

  const config = typeof configOrType === 'string'
    ? createModelConfig(configOrType)
    : configOrType;

  return new wasmModule.WasmModel(config);
}

export function createTokenizer(type, vocab = null) {
  ensureInitialized();

  const typeMap = {
    'wordpiece': TokenizerType.WORDPIECE,
    'bpe': TokenizerType.BPE,
    'sentencepiece': TokenizerType.SENTENCEPIECE
  };

  const tokenizerType = typeMap[type.toLowerCase()];
  if (tokenizerType === undefined) {
    throw new Error(`Unknown tokenizer type: ${type}`);
  }

  const tokenizer = new wasmModule.WasmTokenizer(tokenizerType);

  if (vocab) {
    tokenizer.load_vocab(vocab);
  }

  return tokenizer;
}

// Pipeline factory
export class Pipeline {
  static textGeneration(model, tokenizer, config = {}) {
    ensureInitialized();

    const pipeline = new wasmModule.TextGenerationPipeline(model, tokenizer);

    if (Object.keys(config).length > 0) {
      const genConfig = new wasmModule.GenerationConfig();
      Object.assign(genConfig, config);
      pipeline.set_config(genConfig);
    }

    return pipeline;
  }

  static textClassification(model, tokenizer, labels = []) {
    ensureInitialized();

    const pipeline = new wasmModule.TextClassificationPipeline(model, tokenizer);

    if (labels.length > 0) {
      pipeline.set_labels(labels);
    }

    return pipeline;
  }

  static questionAnswering(model, tokenizer) {
    ensureInitialized();
    return new wasmModule.QuestionAnsweringPipeline(model, tokenizer);
  }

  static async fromPretrained(task, modelName) {
    ensureInitialized();

    const pipelineTypeMap = {
      'text-generation': PipelineType.TEXT_GENERATION,
      'text-classification': PipelineType.TEXT_CLASSIFICATION,
      'token-classification': PipelineType.TOKEN_CLASSIFICATION,
      'question-answering': PipelineType.QUESTION_ANSWERING,
      'summarization': PipelineType.SUMMARIZATION,
      'translation': PipelineType.TRANSLATION
    };

    const pipelineType = pipelineTypeMap[task];
    if (pipelineType === undefined) {
      throw new Error(`Unknown task: ${task}`);
    }

    return wasmModule.PipelineFactory.from_pretrained(pipelineType, modelName);
  }
}

// Memory utilities
export const memory = {
  getStats() {
    ensureInitialized();
    return wasmModule.get_memory_stats();
  },

  getUsage() {
    ensureInitialized();
    return wasmModule.get_wasm_memory_usage();
  }
};

// Utility functions
export const utils = {
  log(message) {
    ensureInitialized();
    wasmModule.log(message);
  },

  logError(message) {
    ensureInitialized();
    wasmModule.log_error(message);
  },

  logWarning(message) {
    ensureInitialized();
    wasmModule.log_warn(message);
  },

  timer(name) {
    ensureInitialized();
    return new wasmModule.Timer(name);
  },

  version() {
    ensureInitialized();
    return wasmModule.version();
  },

  features() {
    ensureInitialized();
    return wasmModule.features();
  }
};

// Comprehensive tensor operations
export const tensor_ops = {
  // Basic arithmetic operations
  add(a, b) { ensureInitialized(); return wasmModule.tensor_add(a, b); },
  sub(a, b) { ensureInitialized(); return wasmModule.tensor_sub(a, b); },
  mul(a, b) { ensureInitialized(); return wasmModule.tensor_mul(a, b); },
  div(a, b) { ensureInitialized(); return wasmModule.tensor_div(a, b); },
  matmul(a, b) { ensureInitialized(); return wasmModule.tensor_matmul(a, b); },

  // Scalar operations
  addScalar(tensor, scalar) { ensureInitialized(); return wasmModule.tensor_add_scalar(tensor, scalar); },
  mulScalar(tensor, scalar) { ensureInitialized(); return wasmModule.tensor_mul_scalar(tensor, scalar); },

  // Shape operations
  reshape(tensor, newShape) { ensureInitialized(); return wasmModule.tensor_reshape(tensor, new Uint32Array(newShape)); },
  transpose(tensor, dims = null) {
    ensureInitialized();
    return dims ? wasmModule.tensor_transpose_dims(tensor, new Uint32Array(dims)) : wasmModule.tensor_transpose(tensor);
  },
  squeeze(tensor, dim = null) {
    ensureInitialized();
    return dim !== null ? wasmModule.tensor_squeeze_dim(tensor, dim) : wasmModule.tensor_squeeze(tensor);
  },
  unsqueeze(tensor, dim) { ensureInitialized(); return wasmModule.tensor_unsqueeze(tensor, dim); },

  // Reduction operations
  sum(tensor, dim = null, keepDim = false) {
    ensureInitialized();
    return dim !== null ? wasmModule.tensor_sum_dim(tensor, dim, keepDim) : wasmModule.tensor_sum(tensor);
  },
  mean(tensor, dim = null, keepDim = false) {
    ensureInitialized();
    return dim !== null ? wasmModule.tensor_mean_dim(tensor, dim, keepDim) : wasmModule.tensor_mean(tensor);
  },

  // Mathematical functions
  exp(tensor) { ensureInitialized(); return wasmModule.tensor_exp(tensor); },
  log(tensor) { ensureInitialized(); return wasmModule.tensor_log(tensor); },
  sqrt(tensor) { ensureInitialized(); return wasmModule.tensor_sqrt(tensor); },
  pow(tensor, exponent) { ensureInitialized(); return wasmModule.tensor_pow(tensor, exponent); },
  abs(tensor) { ensureInitialized(); return wasmModule.tensor_abs(tensor); },

  // Normalization
  layerNorm(tensor, normalized_shape, eps = 1e-5) {
    ensureInitialized();
    return wasmModule.tensor_layer_norm(tensor, new Uint32Array(normalized_shape), eps);
  },
  batchNorm(tensor, running_mean, running_var, weight = null, bias = null, eps = 1e-5) {
    ensureInitialized();
    return wasmModule.tensor_batch_norm(tensor, running_mean, running_var, weight, bias, eps);
  }
};

// Activation functions
export const activations = {
  relu(tensor) { ensureInitialized(); return wasmModule.relu(tensor); },
  leakyRelu(tensor, negative_slope = 0.01) { ensureInitialized(); return wasmModule.leaky_relu(tensor, negative_slope); },
  gelu(tensor) { ensureInitialized(); return wasmModule.gelu(tensor); },
  swish(tensor) { ensureInitialized(); return wasmModule.swish(tensor); },
  sigmoid(tensor) { ensureInitialized(); return wasmModule.sigmoid(tensor); },
  tanh(tensor) { ensureInitialized(); return wasmModule.tanh(tensor); },
  softmax(tensor, dim = -1) { ensureInitialized(); return wasmModule.softmax(tensor, dim); },
  logSoftmax(tensor, dim = -1) { ensureInitialized(); return wasmModule.log_softmax(tensor, dim); }
};

// Streaming utilities for real-time processing
export const streaming = {
  async* textGeneration(model, tokenizer, config = {}) {
    ensureInitialized();

    const generator = new wasmModule.StreamingTextGenerator(model, tokenizer, config);

    try {
      while (true) {
        const chunk = await generator.next();
        if (chunk.done) break;
        yield chunk.value;
      }
    } finally {
      generator.free();
    }
  },

  async* tokenize(tokenizer, text) {
    ensureInitialized();

    const stream = new wasmModule.StreamingTokenizer(tokenizer);

    try {
      for (const char of text) {
        const token = await stream.process_char(char);
        if (token) yield token;
      }

      const remaining = await stream.flush();
      if (remaining) yield remaining;
    } finally {
      stream.free();
    }
  }
};

// Performance monitoring utilities
export const performance = {
  getReport() {
    const report = {
      timestamp: new Date().toISOString(),
      capabilities,
      memory: memoryManager ? memoryManager.getStats() : null,
      profiling: performanceProfiler ? performanceProfiler.generateReport() : null,
      webgl: webglBackend ? webglBackend.getInfo() : null
    };

    return report;
  },

  startSession(name, metadata = {}) {
    if (performanceProfiler) {
      return performanceProfiler.startSession(name, metadata);
    }
    return null;
  },

  endSession() {
    if (performanceProfiler) {
      return performanceProfiler.endSession();
    }
    return null;
  },

  async profile(name, fn, metadata = {}) {
    if (performanceProfiler) {
      return await performanceProfiler.profileOperation(name, fn, metadata);
    } 
      return await fn();
    
  },

  getMemoryUsage() {
    if (memoryManager) {
      return memoryManager.getStats();
    } else if (performanceProfiler) {
      return performanceProfiler.getMemoryUsage();
    } 
      return memory.getStats();
    
  },

  cleanup() {
    if (memoryManager) {
      memoryManager.cleanup();
    }

    if (webglBackend) {
      // WebGL cleanup is automatic
    }

    // Trigger garbage collection hint
    if (typeof window !== 'undefined' && window.gc) {
      window.gc();
    } else if (typeof global !== 'undefined' && global.gc) {
      global.gc();
    }
  }
};

// Export raw WASM module for advanced users
export function getRawModule() {
  ensureInitialized();
  return wasmModule;
}

// Re-export imported modules and enhanced functionality
export {
  // Enhanced operations (imported from modules)
  enhanced_tensor_ops,
  enhanced_tensor_utils,
  enhanced_inference,
  tensor_utils,
  async_utils,
  devTools,
  nodejs,
  webgpu,

  // Core classes
  WebGLBackend,
  TensorMemoryPool,
  WebGLMemoryPool,
  MemoryManager,
  PerformanceProfiler,
  ZeroCopyTensorView,
  ZeroCopyBufferManager,

  // Factory functions
  createWebGLBackend,
  getMemoryManager,
  getProfiler,
  getBufferManager,

  // Utility functions
  createZeroCopyTensor,
  wrapTensor,
  transferTensor,
  batchTransferTensors,
  withMemoryManagement,
  profile,
  profileMethod,

  // Development tools
  DebugUtilities,
  debugUtils,
  debug,
  debugOperation,
  TensorInspector,
  tensorInspector,
  inspect,
  ModelVisualizer,
  modelVisualizer,
  visualize,
  ErrorDiagnostics,
  errorDiagnostics,
  diagnose,
  handleErrors,
  ErrorTypes,
  ErrorSeverity,

  // Image processing
  ImageClassifier,
  ImagePreprocessor,
  StreamProcessor,
  MultiModalProcessor,
  ImageUtils,

  // WebNN Backend
  WebNNBackend,
  WebNNBackendManager,
  WebNNCapabilities,
  WebNNGraph,
  WebNNOperations,
  getWebNNManager,
  initWebNN,
  isWebNNAvailable,

  // Advanced Quantization
  QuantizationType,
  QuantizationScheme,
  QuantizationGranularity,
  Float16Utils,
  Int8Quantizer,
  Int4Quantizer,
  GGMLQuantizer,
  QuantizationCalibrator,
  MixedPrecisionQuantizer,
  QuantizationAwareTraining,

  // Benchmarking Suite
  BenchmarkConfig,
  BenchmarkSuite,
  TensorBenchmarks,
  ModelBenchmarks,
  BackendComparison,
  MemoryBenchmark,
  PerformanceStats,

  // Advanced Caching
  LRUCache,
  TTLCache,
  LFUCache,
  MultiLevelCache,
  PersistentCache,
  CacheManager,

  // NEW: Advanced Optimization (2025-10-27)
  GradientCheckpointingManager,
  MixedPrecisionManager,
  GradientAccumulationManager,
  LARSOptimizer,
  LookaheadOptimizer,
  SAMOptimizer,
  OptimizationStrategySelector,
  AdvancedOptimizer,
  createAdvancedOptimizer,

  // NEW: Federated Learning (2025-10-27)
  FederatedClient,
  SecureAggregationProtocol,
  DifferentialPrivacyMechanism,
  ClientSelectionStrategy,
  ByzantineRobustAggregation,
  FederatedServer,
  createFederatedLearning,

  // NEW: Neural Architecture Search (2025-10-27)
  SearchSpace,
  PerformanceEstimator,
  RandomSearch,
  EvolutionarySearch,
  MultiObjectiveNAS,
  NASController,
  createNAS,

  // NEW: Knowledge Distillation (2025-10-27)
  DistillationLoss,
  TeacherModel,
  StudentModel,
  DistillationTrainer,
  ProgressiveDistillation,
  SelfDistillation,
  DistillationController,
  createDistillation,

  // NEW: Multi-Modal Streaming (2025-10-27)
  BaseStreamHandler,
  TextStreamHandler,
  ImageStreamHandler,
  AudioStreamHandler,
  MultiModalStreamCoordinator,
  createMultiModalStreaming,

  // NEW: ONNX Integration (2025-10-27)
  ONNXRuntimeWrapper,
  ONNXModelConverter,
  ONNXModelAnalyzer,
  ONNXController,
  createONNXIntegration,

  // NEW: Model Interpretability (2025-10-27)
  AttentionVisualizer,
  GradientExplainer,
  FeatureImportanceAnalyzer,
  InterpretabilityController,
  createInterpretability,

  // NEW: Auto Performance Optimization (2025-10-27)
  AutoPerformanceProfiler,
  BottleneckDetector,
  MLBasedOptimizer,
  AutoPerformanceOptimizer,
  createAutoOptimizer
};