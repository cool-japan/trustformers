/**
 * TrustformeRS Models Module
 * Tree-shakable model loading and inference utilities
 */

// Re-export model-related functions from main module
export {
  Pipeline,
  ModelArchitecture,
  PipelineType,
  TokenizerType,
  createModel,
  createModelConfig,
  createTokenizer,
} from '../index.js';

// Model management utilities
export class ModelManager {
  constructor() {
    this.models = new Map();
    this.cache = new Map();
  }

  /**
   * Load and cache a model
   */
  async load(modelId, modelConfig, config = {}) {
    if (this.models.has(modelId)) {
      return this.models.get(modelId);
    }

    const model = await createModel(modelConfig);
    this.models.set(modelId, model);
    return model;
  }

  /**
   * Get a cached model
   */
  get(modelId) {
    return this.models.get(modelId);
  }

  /**
   * Check if model is loaded
   */
  has(modelId) {
    return this.models.has(modelId);
  }

  /**
   * Unload a model
   */
  async unload(modelId) {
    const model = this.models.get(modelId);
    if (model && model.dispose) {
      await model.dispose();
    }
    this.models.delete(modelId);
  }

  /**
   * Unload all models
   */
  async unloadAll() {
    for (const [id, model] of this.models) {
      if (model.dispose) {
        await model.dispose();
      }
    }
    this.models.clear();
  }

  /**
   * Get memory usage of all loaded models
   */
  getMemoryUsage() {
    let totalMemory = 0;
    for (const [id, model] of this.models) {
      if (model.memoryUsage) {
        totalMemory += model.memoryUsage();
      }
    }
    return totalMemory;
  }

  /**
   * List all loaded models
   */
  list() {
    return Array.from(this.models.keys());
  }
}

// Model configuration presets
export const ModelPresets = {
  /**
   * BERT-style models configuration
   */
  bert: {
    maxLength: 512,
    padding: true,
    truncation: true,
    returnTensors: 'tf',
  },

  /**
   * GPT-style models configuration
   */
  gpt: {
    maxLength: 1024,
    temperature: 0.7,
    topK: 50,
    topP: 0.9,
    repetitionPenalty: 1.1,
  },

  /**
   * T5-style models configuration
   */
  t5: {
    maxLength: 512,
    maxNewTokens: 100,
    temperature: 0.8,
    doSample: true,
  },

  /**
   * Vision models configuration
   */
  vision: {
    imageSize: 224,
    normalize: true,
    centerCrop: true,
  },
};

// Model validation utilities
export class ModelValidator {
  /**
   * Validate model configuration
   */
  static validateConfig(config) {
    const required = ['modelType', 'architecture'];
    const missing = required.filter(key => !(key in config));
    if (missing.length > 0) {
      throw new Error(`Missing required config keys: ${missing.join(', ')}`);
    }
    return true;
  }

  /**
   * Validate model inputs
   */
  static validateInputs(inputs, expectedShape) {
    if (!inputs) {
      throw new Error('Model inputs cannot be null or undefined');
    }

    if (expectedShape && inputs.shape) {
      const compatible = this.isShapeCompatible(inputs.shape, expectedShape);
      if (!compatible) {
        throw new Error(
          `Input shape ${inputs.shape} incompatible with expected shape ${expectedShape}`
        );
      }
    }

    return true;
  }

  /**
   * Check if shapes are compatible
   */
  static isShapeCompatible(actual, expected) {
    if (actual.length !== expected.length) return false;

    return actual.every((dim, i) => {
      const expectedDim = expected[i];
      return expectedDim === -1 || expectedDim === dim;
    });
  }

  /**
   * Validate model outputs
   */
  static validateOutputs(outputs, expectedKeys = []) {
    if (!outputs || typeof outputs !== 'object') {
      throw new Error('Model outputs must be an object');
    }

    const missing = expectedKeys.filter(key => !(key in outputs));
    if (missing.length > 0) {
      console.warn(`Missing expected output keys: ${missing.join(', ')}`);
    }

    return true;
  }
}

// Model performance utilities
export class ModelProfiler {
  constructor(model) {
    this.model = model;
    this.metrics = {
      inferenceCount: 0,
      totalTime: 0,
      averageTime: 0,
      minTime: Infinity,
      maxTime: 0,
    };
  }

  /**
   * Profile a single inference
   */
  async profileInference(inputs, options = {}) {
    const startTime = performance.now();
    const outputs = await this.model.forward(inputs, options);
    const endTime = performance.now();

    const inferenceTime = endTime - startTime;
    this.updateMetrics(inferenceTime);

    return { outputs, inferenceTime };
  }

  /**
   * Update performance metrics
   */
  updateMetrics(inferenceTime) {
    this.metrics.inferenceCount++;
    this.metrics.totalTime += inferenceTime;
    this.metrics.averageTime = this.metrics.totalTime / this.metrics.inferenceCount;
    this.metrics.minTime = Math.min(this.metrics.minTime, inferenceTime);
    this.metrics.maxTime = Math.max(this.metrics.maxTime, inferenceTime);
  }

  /**
   * Get performance summary
   */
  getSummary() {
    return { ...this.metrics };
  }

  /**
   * Reset metrics
   */
  reset() {
    this.metrics = {
      inferenceCount: 0,
      totalTime: 0,
      averageTime: 0,
      minTime: Infinity,
      maxTime: 0,
    };
  }
}

export default {
  ModelManager,
  ModelPresets,
  ModelValidator,
  ModelProfiler,
};
