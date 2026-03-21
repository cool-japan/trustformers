/**
 * Angular services and directives for TrustformeRS
 * Provides Angular-specific integrations for the TrustformeRS ecosystem
 *
 * Note: This file provides Angular integration patterns and can be used as TypeScript
 * by renaming to .ts and adding proper type annotations
 */

import {
  initialize,
  initializeEnhanced,
  createModel,
  createTokenizer,
  Pipeline,
  tensor,
  enhanced_inference,
  performance,
  devTools,
  memory,
} from '../index.js';

/**
 * TrustformeRS Core Service
 * Main service for initializing and managing TrustformeRS
 */
export class TrustformersService {
  constructor() {
    this.isInitialized = false;
    this.isLoading = false;
    this.error = null;
    this.capabilities = null;
    this.initPromise = null;
  }

  /**
   * Initialize TrustformeRS with enhanced features
   * @param {Object} options - Initialization options
   * @returns {Promise<Object>} Initialization capabilities
   */
  async initialize(options = {}) {
    if (this.initPromise) return this.initPromise;

    this.isLoading = true;
    this.error = null;

    this.initPromise = this._performInitialization(options);

    try {
      this.capabilities = await this.initPromise;
      this.isInitialized = true;
      return this.capabilities;
    } catch (err) {
      this.error = err;
      this.initPromise = null;
      throw err;
    } finally {
      this.isLoading = false;
    }
  }

  async _performInitialization(options) {
    return await initializeEnhanced({
      enableWebGL: true,
      enableMemoryPool: true,
      enableProfiling: true,
      enableZeroCopy: true,
      ...options,
    });
  }

  /**
   * Reinitialize with new options
   * @param {Object} options - New initialization options
   * @returns {Promise<Object>} New capabilities
   */
  async reinitialize(options = {}) {
    this.initPromise = null;
    this.isInitialized = false;
    this.capabilities = null;
    return await this.initialize(options);
  }

  /**
   * Get current initialization state
   * @returns {Object} Current state
   */
  getState() {
    return {
      isInitialized: this.isInitialized,
      isLoading: this.isLoading,
      error: this.error,
      capabilities: this.capabilities,
    };
  }
}

/**
 * Model Management Service
 * Service for creating and managing ML models
 */
export class TrustformersModelService {
  constructor(trustformersService) {
    this.trustformersService = trustformersService;
    this.models = new Map();
    this.loadingStates = new Map();
  }

  /**
   * Create and cache a model
   * @param {string} modelId - Unique model identifier
   * @param {string|Object} config - Model configuration
   * @param {Object} options - Model options
   * @returns {Promise<Object>} Model instance
   */
  async createModel(modelId, config, options = {}) {
    if (this.models.has(modelId)) {
      return this.models.get(modelId);
    }

    if (this.loadingStates.has(modelId)) {
      return this.loadingStates.get(modelId);
    }

    const loadingPromise = this._loadModel(config, options);
    this.loadingStates.set(modelId, loadingPromise);

    try {
      const model = await loadingPromise;
      this.models.set(modelId, model);
      this.loadingStates.delete(modelId);
      return model;
    } catch (error) {
      this.loadingStates.delete(modelId);
      throw error;
    }
  }

  async _loadModel(config, options) {
    await this.trustformersService.initialize();
    return createModel(config);
  }

  /**
   * Get cached model
   * @param {string} modelId - Model identifier
   * @returns {Object|null} Model instance or null
   */
  getModel(modelId) {
    return this.models.get(modelId) || null;
  }

  /**
   * Remove model from cache
   * @param {string} modelId - Model identifier
   */
  removeModel(modelId) {
    const model = this.models.get(modelId);
    if (model && model.free) {
      model.free();
    }
    this.models.delete(modelId);
  }

  /**
   * Run inference on a model
   * @param {string} modelId - Model identifier
   * @param {Object} inputs - Input tensors
   * @param {Object} options - Inference options
   * @returns {Promise<Object>} Inference results
   */
  async runInference(modelId, inputs, options = {}) {
    const model = this.getModel(modelId);
    if (!model) {
      throw new Error(`Model ${modelId} not found`);
    }

    return await enhanced_inference.runInference(model, inputs, {
      profile: true,
      useMemoryPool: true,
      autoCleanup: true,
      ...options,
    });
  }

  /**
   * Run batch inference
   * @param {string} modelId - Model identifier
   * @param {Array} batchInputs - Array of input batches
   * @param {Object} options - Batch options
   * @returns {Promise<Array>} Array of results
   */
  async batchInference(modelId, batchInputs, options = {}) {
    const model = this.getModel(modelId);
    if (!model) {
      throw new Error(`Model ${modelId} not found`);
    }

    return await enhanced_inference.batchInference(model, batchInputs, {
      batchSize: 32,
      profile: true,
      ...options,
    });
  }

  /**
   * Cleanup all models
   */
  cleanup() {
    for (const [modelId, model] of this.models) {
      if (model && model.free) {
        model.free();
      }
    }
    this.models.clear();
    this.loadingStates.clear();
  }
}

/**
 * Tokenizer Service
 * Service for managing tokenizers
 */
export class TrustformersTokenizerService {
  constructor(trustformersService) {
    this.trustformersService = trustformersService;
    this.tokenizers = new Map();
  }

  /**
   * Create and cache a tokenizer
   * @param {string} tokenizerId - Unique tokenizer identifier
   * @param {string} type - Tokenizer type
   * @param {Object} vocab - Vocabulary object
   * @returns {Promise<Object>} Tokenizer instance
   */
  async createTokenizer(tokenizerId, type, vocab = null) {
    if (this.tokenizers.has(tokenizerId)) {
      return this.tokenizers.get(tokenizerId);
    }

    await this.trustformersService.initialize();
    const tokenizer = createTokenizer(type, vocab);
    this.tokenizers.set(tokenizerId, tokenizer);
    return tokenizer;
  }

  /**
   * Get cached tokenizer
   * @param {string} tokenizerId - Tokenizer identifier
   * @returns {Object|null} Tokenizer instance or null
   */
  getTokenizer(tokenizerId) {
    return this.tokenizers.get(tokenizerId) || null;
  }

  /**
   * Encode text using tokenizer
   * @param {string} tokenizerId - Tokenizer identifier
   * @param {string} text - Text to encode
   * @returns {Array} Encoded tokens
   */
  encode(tokenizerId, text) {
    const tokenizer = this.getTokenizer(tokenizerId);
    if (!tokenizer) {
      throw new Error(`Tokenizer ${tokenizerId} not found`);
    }
    return tokenizer.encode(text);
  }

  /**
   * Decode tokens using tokenizer
   * @param {string} tokenizerId - Tokenizer identifier
   * @param {Array} tokens - Tokens to decode
   * @returns {string} Decoded text
   */
  decode(tokenizerId, tokens) {
    const tokenizer = this.getTokenizer(tokenizerId);
    if (!tokenizer) {
      throw new Error(`Tokenizer ${tokenizerId} not found`);
    }
    return tokenizer.decode(tokens);
  }

  /**
   * Batch encode texts
   * @param {string} tokenizerId - Tokenizer identifier
   * @param {Array<string>} texts - Texts to encode
   * @returns {Array<Array>} Array of encoded token arrays
   */
  batchEncode(tokenizerId, texts) {
    const tokenizer = this.getTokenizer(tokenizerId);
    if (!tokenizer) {
      throw new Error(`Tokenizer ${tokenizerId} not found`);
    }
    return texts.map(text => tokenizer.encode(text));
  }

  /**
   * Remove tokenizer from cache
   * @param {string} tokenizerId - Tokenizer identifier
   */
  removeTokenizer(tokenizerId) {
    const tokenizer = this.tokenizers.get(tokenizerId);
    if (tokenizer && tokenizer.free) {
      tokenizer.free();
    }
    this.tokenizers.delete(tokenizerId);
  }

  /**
   * Cleanup all tokenizers
   */
  cleanup() {
    for (const [tokenizerId, tokenizer] of this.tokenizers) {
      if (tokenizer && tokenizer.free) {
        tokenizer.free();
      }
    }
    this.tokenizers.clear();
  }
}

/**
 * Pipeline Service
 * Service for managing ML pipelines
 */
export class TrustformersPipelineService {
  constructor(trustformersService, modelService, tokenizerService) {
    this.trustformersService = trustformersService;
    this.modelService = modelService;
    this.tokenizerService = tokenizerService;
    this.pipelines = new Map();
  }

  /**
   * Create and cache a pipeline
   * @param {string} pipelineId - Unique pipeline identifier
   * @param {string} task - Pipeline task type
   * @param {string} modelId - Model identifier
   * @param {string} tokenizerId - Tokenizer identifier
   * @param {Object} config - Pipeline configuration
   * @returns {Promise<Object>} Pipeline instance
   */
  async createPipeline(pipelineId, task, modelId, tokenizerId, config = {}) {
    if (this.pipelines.has(pipelineId)) {
      return this.pipelines.get(pipelineId);
    }

    const model = this.modelService.getModel(modelId);
    const tokenizer = this.tokenizerService.getTokenizer(tokenizerId);

    if (!model) {
      throw new Error(`Model ${modelId} not found`);
    }
    if (!tokenizer) {
      throw new Error(`Tokenizer ${tokenizerId} not found`);
    }

    let pipeline;
    switch (task) {
      case 'text-generation':
        pipeline = Pipeline.textGeneration(model, tokenizer, config);
        break;
      case 'text-classification':
        pipeline = Pipeline.textClassification(model, tokenizer, config.labels || []);
        break;
      case 'question-answering':
        pipeline = Pipeline.questionAnswering(model, tokenizer);
        break;
      default:
        pipeline = await Pipeline.fromPretrained(task, config.modelName || 'default');
    }

    this.pipelines.set(pipelineId, pipeline);
    return pipeline;
  }

  /**
   * Get cached pipeline
   * @param {string} pipelineId - Pipeline identifier
   * @returns {Object|null} Pipeline instance or null
   */
  getPipeline(pipelineId) {
    return this.pipelines.get(pipelineId) || null;
  }

  /**
   * Run pipeline
   * @param {string} pipelineId - Pipeline identifier
   * @param {any} input - Pipeline input
   * @param {Object} options - Pipeline options
   * @returns {Promise<any>} Pipeline output
   */
  async run(pipelineId, input, options = {}) {
    const pipeline = this.getPipeline(pipelineId);
    if (!pipeline) {
      throw new Error(`Pipeline ${pipelineId} not found`);
    }

    return await pipeline.run(input, {
      maxLength: 50,
      temperature: 0.7,
      ...options,
    });
  }

  /**
   * Run batch processing
   * @param {string} pipelineId - Pipeline identifier
   * @param {Array} inputs - Array of inputs
   * @param {Object} options - Batch options
   * @returns {Promise<Array>} Array of outputs
   */
  async batch(pipelineId, inputs, options = {}) {
    const pipeline = this.getPipeline(pipelineId);
    if (!pipeline) {
      throw new Error(`Pipeline ${pipelineId} not found`);
    }

    return Promise.all(inputs.map(input => pipeline.run(input, options)));
  }

  /**
   * Remove pipeline from cache
   * @param {string} pipelineId - Pipeline identifier
   */
  removePipeline(pipelineId) {
    const pipeline = this.pipelines.get(pipelineId);
    if (pipeline && pipeline.free) {
      pipeline.free();
    }
    this.pipelines.delete(pipelineId);
  }

  /**
   * Cleanup all pipelines
   */
  cleanup() {
    for (const [pipelineId, pipeline] of this.pipelines) {
      if (pipeline && pipeline.free) {
        pipeline.free();
      }
    }
    this.pipelines.clear();
  }
}

/**
 * Performance Monitoring Service
 * Service for tracking performance metrics
 */
export class TrustformersPerformanceService {
  constructor() {
    this.isEnabled = true;
    this.currentSession = null;
    this.metrics = null;
    this.intervalId = null;
  }

  /**
   * Enable performance monitoring
   * @param {Object} options - Monitoring options
   */
  enable(options = {}) {
    this.isEnabled = true;
    this.startMetricsCollection(options.updateInterval || 5000);
  }

  /**
   * Disable performance monitoring
   */
  disable() {
    this.isEnabled = false;
    this.stopMetricsCollection();
  }

  /**
   * Start metrics collection
   * @param {number} interval - Collection interval in ms
   */
  startMetricsCollection(interval = 5000) {
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }

    this.updateMetrics();
    this.intervalId = setInterval(() => this.updateMetrics(), interval);
  }

  /**
   * Stop metrics collection
   */
  stopMetricsCollection() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }

  /**
   * Update current metrics
   */
  updateMetrics() {
    if (!this.isEnabled) return;

    this.metrics = {
      memory: performance.getMemoryUsage(),
      report: performance.getReport(),
      timestamp: Date.now(),
    };
  }

  /**
   * Start performance session
   * @param {string} name - Session name
   * @param {Object} metadata - Session metadata
   * @returns {string} Session ID
   */
  startSession(name, metadata = {}) {
    if (!this.isEnabled) return null;

    this.currentSession = performance.startSession(name, {
      framework: 'angular',
      timestamp: new Date().toISOString(),
      ...metadata,
    });
    return this.currentSession;
  }

  /**
   * End current performance session
   * @returns {Object} Session report
   */
  endSession() {
    if (!this.isEnabled || !this.currentSession) return null;

    const report = performance.endSession();
    this.currentSession = null;
    return report;
  }

  /**
   * Profile an operation
   * @param {string} name - Operation name
   * @param {Function} operation - Operation to profile
   * @param {Object} metadata - Additional metadata
   * @returns {Promise<any>} Operation result
   */
  async profileOperation(name, operation, metadata = {}) {
    if (!this.isEnabled) return await operation();

    return await performance.profile(name, operation, {
      framework: 'angular',
      ...metadata,
    });
  }

  /**
   * Get current metrics
   * @returns {Object} Current metrics
   */
  getMetrics() {
    return this.metrics;
  }

  /**
   * Cleanup performance monitoring
   */
  cleanup() {
    this.stopMetricsCollection();
    performance.cleanup();
    this.updateMetrics();
  }
}

/**
 * Development Tools Service
 * Service for debugging and development utilities
 */
export class TrustformersDevToolsService {
  constructor() {
    this.isEnabled = process.env.NODE_ENV === 'development';
    this.debugState = null;
  }

  /**
   * Enable development tools
   * @param {Object} options - Debug options
   */
  enable(options = {}) {
    devTools.debug.enable({
      trackTensors: true,
      trackOperations: true,
      validateOperations: true,
      ...options,
    });
    this.isEnabled = true;
  }

  /**
   * Disable development tools
   */
  disable() {
    devTools.debug.disable();
    this.isEnabled = false;
  }

  /**
   * Start debug session
   * @param {string} name - Session name
   * @param {Object} metadata - Session metadata
   * @returns {string} Session ID
   */
  startDebugSession(name, metadata = {}) {
    if (!this.isEnabled) return null;

    return devTools.debug.startSession(name, {
      framework: 'angular',
      timestamp: new Date().toISOString(),
      ...metadata,
    });
  }

  /**
   * Track tensor operation
   * @param {Object} tensor - Tensor to track
   * @param {string} operation - Operation name
   * @param {Object} metadata - Additional metadata
   */
  trackTensor(tensor, operation, metadata = {}) {
    if (!this.isEnabled) return;

    devTools.debug.trackTensor(tensor, operation, {
      framework: 'angular',
      ...metadata,
    });
  }

  /**
   * Validate operation
   * @param {string} operation - Operation name
   * @param {Array} tensors - Tensors involved
   * @param {Object} options - Validation options
   * @returns {boolean} Validation result
   */
  validateOperation(operation, tensors, options = {}) {
    if (!this.isEnabled) return true;

    return devTools.debug.validateOperation(operation, tensors, options);
  }

  /**
   * Generate debug report
   * @returns {Object} Debug report
   */
  generateReport() {
    if (!this.isEnabled) return null;

    const report = devTools.debug.generateReport();
    this.debugState = report;
    return report;
  }

  /**
   * Comprehensive analysis
   * @param {Object} options - Analysis options
   * @returns {Object} Comprehensive analysis
   */
  analyzeComprehensive(options = {}) {
    if (!this.isEnabled) return null;

    return devTools.analyze.comprehensive({
      includeDebug: true,
      ...options,
    });
  }

  /**
   * Get current debug state
   * @returns {Object} Current debug state
   */
  getDebugState() {
    return this.debugState;
  }
}

/**
 * Angular Module Configuration
 * Provides Angular-style module configuration for dependency injection
 */
export const TrustformersAngularConfig = {
  providers: [
    { provide: 'TrustformersService', useClass: TrustformersService },
    {
      provide: 'TrustformersModelService',
      useFactory: trustformersService => new TrustformersModelService(trustformersService),
      deps: ['TrustformersService'],
    },
    {
      provide: 'TrustformersTokenizerService',
      useFactory: trustformersService => new TrustformersTokenizerService(trustformersService),
      deps: ['TrustformersService'],
    },
    {
      provide: 'TrustformersPipelineService',
      useFactory: (trustformersService, modelService, tokenizerService) =>
        new TrustformersPipelineService(trustformersService, modelService, tokenizerService),
      deps: ['TrustformersService', 'TrustformersModelService', 'TrustformersTokenizerService'],
    },
    { provide: 'TrustformersPerformanceService', useClass: TrustformersPerformanceService },
    { provide: 'TrustformersDevToolsService', useClass: TrustformersDevToolsService },
  ],
};

/**
 * Angular Directive Helper Functions
 * Helper functions for creating Angular directives
 */

/**
 * Performance Monitor Directive Factory
 * @param {TrustformersPerformanceService} performanceService - Performance service
 * @returns {Object} Directive definition
 */
export function createPerformanceMonitorDirective(performanceService) {
  return {
    restrict: 'E',
    template: `
      <div class="performance-monitor" ng-if="$ctrl.enabled && $ctrl.metrics" 
           ng-click="$ctrl.toggleExpanded()"
           style="position: fixed; top: 10px; right: 10px; background: rgba(0,0,0,0.8); 
                  color: white; padding: 10px; border-radius: 5px; font-size: 12px; 
                  font-family: monospace; z-index: 9999; cursor: pointer;">
        <div>🚀 TrustformeRS Performance</div>
        <div ng-if="$ctrl.metrics.memory">
          Memory: {{($ctrl.metrics.memory.used / 1024 / 1024).toFixed(1)}}MB
        </div>
        <div ng-if="$ctrl.expanded && $ctrl.metrics.report" 
             style="margin-top: 10px; font-size: 10px;">
          <div>
            Capabilities: {{$ctrl.getCapabilities()}}
          </div>
          <div ng-if="$ctrl.metrics.report.profiling">
            Operations: {{$ctrl.metrics.report.profiling.totalOperations || 0}}
          </div>
        </div>
      </div>
    `,
    controller() {
      this.enabled = true;
      this.expanded = false;
      this.metrics = null;

      this.toggleExpanded = () => {
        this.expanded = !this.expanded;
      };

      this.getCapabilities = () => {
        if (!this.metrics.report || !this.metrics.report.capabilities) return '';
        return Object.entries(this.metrics.report.capabilities)
          .filter(([, v]) => v)
          .map(([k]) => k)
          .join(', ');
      };

      // Start metrics collection
      performanceService.enable();

      // Update metrics periodically
      const updateMetrics = () => {
        this.metrics = performanceService.getMetrics();
      };

      const interval = setInterval(updateMetrics, 5000);
      updateMetrics();

      this.$onDestroy = () => {
        clearInterval(interval);
      };
    },
    controllerAs: '$ctrl',
  };
}

/**
 * Memory Monitor Directive Factory
 * @returns {Object} Directive definition
 */
export function createMemoryMonitorDirective() {
  return {
    restrict: 'E',
    template: `
      <div class="memory-monitor" ng-if="$ctrl.enabled && $ctrl.memoryStats" 
           ng-click="$ctrl.toggleExpanded()"
           style="position: fixed; top: 80px; right: 10px; background: rgba(0,0,0,0.8); 
                  color: white; padding: 10px; border-radius: 5px; font-size: 12px; 
                  font-family: monospace; z-index: 9999; cursor: pointer;">
        <div>💾 Memory</div>
        <div>{{$ctrl.getUsedMB()}}MB / {{$ctrl.getLimitMB()}}MB</div>
        <div ng-if="$ctrl.expanded" style="margin-top: 10px; font-size: 10px;">
          <div>Usage: {{$ctrl.getUsagePercent()}}%</div>
          <div>Updated: {{$ctrl.getUpdateTime()}}</div>
        </div>
      </div>
    `,
    controller() {
      this.enabled = true;
      this.expanded = false;
      this.memoryStats = null;

      this.toggleExpanded = () => {
        this.expanded = !this.expanded;
      };

      this.getUsedMB = () =>
        this.memoryStats?.used ? (this.memoryStats.used / 1024 / 1024).toFixed(1) : 'N/A';

      this.getLimitMB = () =>
        this.memoryStats?.limit ? (this.memoryStats.limit / 1024 / 1024).toFixed(1) : 'N/A';

      this.getUsagePercent = () =>
        this.memoryStats?.used && this.memoryStats?.limit
          ? ((this.memoryStats.used / this.memoryStats.limit) * 100).toFixed(1)
          : 'N/A';

      this.getUpdateTime = () =>
        this.memoryStats?.timestamp
          ? new Date(this.memoryStats.timestamp).toLocaleTimeString()
          : 'N/A';

      // Update memory stats periodically
      const updateMemoryStats = () => {
        const stats = memory.getStats();
        this.memoryStats = {
          ...stats,
          timestamp: Date.now(),
        };
      };

      const interval = setInterval(updateMemoryStats, 3000);
      updateMemoryStats();

      this.$onDestroy = () => {
        clearInterval(interval);
      };
    },
    controllerAs: '$ctrl',
  };
}

export default {
  // Services
  TrustformersService,
  TrustformersModelService,
  TrustformersTokenizerService,
  TrustformersPipelineService,
  TrustformersPerformanceService,
  TrustformersDevToolsService,

  // Configuration
  TrustformersAngularConfig,

  // Directive factories
  createPerformanceMonitorDirective,
  createMemoryMonitorDirective,
};
