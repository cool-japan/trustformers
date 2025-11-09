/**
 * Performance Profiler for TrustformeRS JavaScript API
 * Provides comprehensive performance monitoring and optimization recommendations
 */

let wasmModule = null;

/**
 * Initialize profiler with WASM module
 * @param {Object} module - WASM module reference
 */
export function initProfiler(module) {
  wasmModule = module;
}

/**
 * Performance metrics collector
 */
export class PerformanceProfiler {
  constructor(options = {}) {
    this.enabled = options.enabled !== false;
    this.detailed = options.detailed || false;
    this.autoReport = options.autoReport || false;
    this.reportInterval = options.reportInterval || 30000; // 30 seconds
    
    this.metrics = {
      operations: new Map(),
      memory: [],
      gpu: [],
      inference: [],
      timeline: []
    };
    
    this.currentSession = null;
    this.sessionStack = [];
    this.warningThresholds = {
      operationTime: 1000, // 1 second
      memoryUsage: 512 * 1024 * 1024, // 512MB
      memoryLeakRate: 10 * 1024 * 1024, // 10MB/minute
      cpuUsage: 80 // 80%
    };

    if (this.autoReport) {
      this.startAutoReporting();
    }
  }

  /**
   * Start a performance session
   * @param {string} name - Session name
   * @param {Object} metadata - Additional metadata
   * @returns {string} Session ID
   */
  startSession(name, metadata = {}) {
    if (!this.enabled) return null;

    const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const session = {
      id: sessionId,
      name,
      startTime: performance.now(),
      endTime: null,
      duration: null,
      operations: [],
      memorySnapshots: [],
      metadata,
      warnings: [],
      parent: this.currentSession
    };

    this.sessionStack.push(this.currentSession);
    this.currentSession = session;

    this.addTimelineEvent('session_start', { sessionId, name });
    return sessionId;
  }

  /**
   * End the current performance session
   * @returns {Object} Session results
   */
  endSession() {
    if (!this.enabled || !this.currentSession) return null;

    const session = this.currentSession;
    session.endTime = performance.now();
    session.duration = session.endTime - session.startTime;

    this.addTimelineEvent('session_end', { 
      sessionId: session.id, 
      duration: session.duration 
    });

    // Generate session report
    const report = this.generateSessionReport(session);

    // Store session results
    if (!this.metrics.sessions) {
      this.metrics.sessions = [];
    }
    this.metrics.sessions.push(session);

    // Restore parent session
    this.currentSession = this.sessionStack.pop() || null;

    return report;
  }

  /**
   * Profile a tensor operation
   * @param {string} operation - Operation name
   * @param {Function} fn - Operation function
   * @param {Object} metadata - Additional metadata
   * @returns {Promise<*>} Operation result
   */
  async profileOperation(operation, fn, metadata = {}) {
    if (!this.enabled) {
      return await fn();
    }

    const operationId = `op_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const startTime = performance.now();
    const startMemory = this.getMemoryUsage();

    let result; let error = null;
    let gpuTime = null;

    try {
      // Start GPU timing if available
      if (this.detailed && wasmModule && wasmModule.start_gpu_timer) {
        wasmModule.start_gpu_timer(operationId);
      }

      this.addTimelineEvent('operation_start', { operationId, operation });
      
      result = await fn();
      
      // End GPU timing
      if (this.detailed && wasmModule && wasmModule.end_gpu_timer) {
        gpuTime = wasmModule.end_gpu_timer(operationId);
      }

    } catch (e) {
      error = e;
      throw e;
    } finally {
      const endTime = performance.now();
      const endMemory = this.getMemoryUsage();
      const duration = endTime - startTime;

      const operationMetrics = {
        id: operationId,
        operation,
        startTime,
        endTime,
        duration,
        memoryBefore: startMemory,
        memoryAfter: endMemory,
        memoryDelta: endMemory.used - startMemory.used,
        gpuTime,
        error: error ? error.message : null,
        metadata
      };

      this.recordOperation(operationMetrics);
      this.addTimelineEvent('operation_end', { 
        operationId, 
        operation, 
        duration,
        success: !error 
      });

      // Check for warnings
      this.checkOperationWarnings(operationMetrics);
    }

    return result;
  }

  /**
   * Profile model inference
   * @param {Object} model - Model object
   * @param {Object} inputs - Input tensors
   * @param {Object} options - Profiling options
   * @returns {Promise<Object>} Inference results with metrics
   */
  async profileInference(model, inputs, options = {}) {
    if (!this.enabled) {
      return await model.forward(inputs);
    }

    const sessionId = this.startSession('inference', {
      modelType: model.config ? model.config.model_type : 'unknown',
      inputShapes: this.extractInputShapes(inputs),
      ...options.metadata
    });

    let result;
    try {
      result = await this.profileOperation('inference', async () => await model.forward(inputs), { sessionId });

      // Collect detailed inference metrics
      if (this.detailed) {
        await this.collectInferenceMetrics(model, inputs, result);
      }

    } finally {
      this.endSession();
    }

    return result;
  }

  /**
   * Profile memory usage over time
   * @param {number} duration - Monitoring duration in ms
   * @param {number} interval - Sampling interval in ms
   * @returns {Promise<Array>} Memory usage timeline
   */
  async profileMemory(duration = 10000, interval = 100) {
    if (!this.enabled) return [];

    const samples = [];
    const startTime = Date.now();
    
    const sampleMemory = () => {
      const memory = this.getMemoryUsage();
      samples.push({
        timestamp: Date.now() - startTime,
        ...memory
      });
    };

    const intervalId = setInterval(sampleMemory, interval);
    
    return new Promise((resolve) => {
      setTimeout(() => {
        clearInterval(intervalId);
        this.metrics.memory.push(...samples);
        resolve(samples);
      }, duration);
    });
  }

  /**
   * Record operation metrics
   * @param {Object} metrics - Operation metrics
   */
  recordOperation(metrics) {
    const operationStats = this.metrics.operations.get(metrics.operation) || {
      count: 0,
      totalTime: 0,
      totalGpuTime: 0,
      totalMemoryDelta: 0,
      errors: 0,
      averageTime: 0,
      averageGpuTime: 0,
      averageMemoryDelta: 0
    };

    operationStats.count++;
    operationStats.totalTime += metrics.duration;
    operationStats.totalMemoryDelta += metrics.memoryDelta;
    
    if (metrics.gpuTime) {
      operationStats.totalGpuTime += metrics.gpuTime;
      operationStats.averageGpuTime = operationStats.totalGpuTime / operationStats.count;
    }
    
    if (metrics.error) {
      operationStats.errors++;
    }

    operationStats.averageTime = operationStats.totalTime / operationStats.count;
    operationStats.averageMemoryDelta = operationStats.totalMemoryDelta / operationStats.count;

    this.metrics.operations.set(metrics.operation, operationStats);

    // Add to current session if active
    if (this.currentSession) {
      this.currentSession.operations.push(metrics);
    }
  }

  /**
   * Get current memory usage
   * @returns {Object} Memory usage information
   */
  getMemoryUsage() {
    const memory = {
      used: 0,
      total: 0,
      limit: 0,
      wasm: 0,
      gpu: 0,
      js: 0
    };

    // JavaScript heap memory
    if (performance.memory) {
      memory.used = performance.memory.usedJSHeapSize;
      memory.total = performance.memory.totalJSHeapSize;
      memory.limit = performance.memory.jsHeapSizeLimit;
      memory.js = memory.used;
    }

    // WASM memory
    if (wasmModule && wasmModule.get_memory_stats) {
      const wasmStats = wasmModule.get_memory_stats();
      memory.wasm = wasmStats.used || 0;
      memory.used += memory.wasm;
    }

    // GPU memory (if available)
    if (wasmModule && wasmModule.get_gpu_memory_usage) {
      memory.gpu = wasmModule.get_gpu_memory_usage();
      memory.used += memory.gpu;
    }

    return memory;
  }

  /**
   * Extract input shapes from various input formats
   * @param {*} inputs - Model inputs
   * @returns {Object} Input shapes
   */
  extractInputShapes(inputs) {
    const shapes = {};
    
    if (Array.isArray(inputs)) {
      inputs.forEach((input, index) => {
        if (input && input.shape) {
          shapes[`input_${index}`] = Array.from(input.shape());
        }
      });
    } else if (inputs && typeof inputs === 'object') {
      for (const [key, value] of Object.entries(inputs)) {
        if (value && value.shape) {
          shapes[key] = Array.from(value.shape());
        }
      }
    } else if (inputs && inputs.shape) {
      shapes.input = Array.from(inputs.shape());
    }

    return shapes;
  }

  /**
   * Collect detailed inference metrics
   * @param {Object} model - Model object
   * @param {Object} inputs - Input tensors
   * @param {Object} outputs - Output tensors
   */
  async collectInferenceMetrics(model, inputs, outputs) {
    const metrics = {
      timestamp: Date.now(),
      inputShapes: this.extractInputShapes(inputs),
      outputShapes: this.extractInputShapes(outputs),
      memory: this.getMemoryUsage()
    };

    // Layer-wise timing if available
    if (wasmModule && wasmModule.get_layer_timings) {
      metrics.layerTimings = wasmModule.get_layer_timings();
    }

    // Model parameters if available
    if (model.num_parameters) {
      metrics.parameters = model.num_parameters();
    }

    this.metrics.inference.push(metrics);
  }

  /**
   * Check for performance warnings
   * @param {Object} operationMetrics - Operation metrics to check
   */
  checkOperationWarnings(operationMetrics) {
    const warnings = [];

    if (operationMetrics.duration > this.warningThresholds.operationTime) {
      warnings.push({
        type: 'slow_operation',
        message: `Operation '${operationMetrics.operation}' took ${operationMetrics.duration.toFixed(2)}ms`,
        threshold: this.warningThresholds.operationTime,
        value: operationMetrics.duration
      });
    }

    if (operationMetrics.memoryDelta > this.warningThresholds.memoryUsage) {
      warnings.push({
        type: 'memory_spike',
        message: `Operation '${operationMetrics.operation}' allocated ${(operationMetrics.memoryDelta / 1024 / 1024).toFixed(2)}MB`,
        threshold: this.warningThresholds.memoryUsage,
        value: operationMetrics.memoryDelta
      });
    }

    if (warnings.length > 0 && this.currentSession) {
      this.currentSession.warnings.push(...warnings);
    }

    return warnings;
  }

  /**
   * Add timeline event
   * @param {string} type - Event type
   * @param {Object} data - Event data
   */
  addTimelineEvent(type, data) {
    this.metrics.timeline.push({
      timestamp: performance.now(),
      type,
      data
    });
  }

  /**
   * Generate comprehensive performance report
   * @returns {Object} Performance report
   */
  generateReport() {
    const report = {
      timestamp: new Date().toISOString(),
      summary: this.generateSummary(),
      operations: this.analyzeOperations(),
      memory: this.analyzeMemory(),
      recommendations: this.generateRecommendations(),
      sessions: this.metrics.sessions || []
    };

    return report;
  }

  /**
   * Generate session report
   * @param {Object} session - Session data
   * @returns {Object} Session report
   */
  generateSessionReport(session) {
    const operationStats = this.aggregateSessionOperations(session.operations);
    const memoryAnalysis = this.analyzeSessionMemory(session);

    return {
      session: {
        id: session.id,
        name: session.name,
        duration: session.duration,
        metadata: session.metadata
      },
      operations: operationStats,
      memory: memoryAnalysis,
      warnings: session.warnings,
      recommendations: this.generateSessionRecommendations(session)
    };
  }

  /**
   * Generate performance summary
   * @returns {Object} Performance summary
   */
  generateSummary() {
    const totalOperations = Array.from(this.metrics.operations.values())
      .reduce((sum, stats) => sum + stats.count, 0);
    
    const averageOperationTime = Array.from(this.metrics.operations.values())
      .reduce((sum, stats) => sum + stats.averageTime, 0) / this.metrics.operations.size;

    const currentMemory = this.getMemoryUsage();

    return {
      totalOperations,
      averageOperationTime: averageOperationTime || 0,
      operationTypes: this.metrics.operations.size,
      currentMemoryUsage: currentMemory.used,
      memoryEfficiency: currentMemory.limit > 0 ? currentMemory.used / currentMemory.limit : 0,
      timelineEvents: this.metrics.timeline.length
    };
  }

  /**
   * Analyze operation performance
   * @returns {Object} Operation analysis
   */
  analyzeOperations() {
    const operations = {};
    
    for (const [name, stats] of this.metrics.operations.entries()) {
      operations[name] = {
        ...stats,
        efficiency: stats.averageTime > 0 ? (stats.count - stats.errors) / stats.count : 0,
        throughput: stats.totalTime > 0 ? stats.count / (stats.totalTime / 1000) : 0 // ops/second
      };
    }

    return operations;
  }

  /**
   * Analyze memory usage patterns
   * @returns {Object} Memory analysis
   */
  analyzeMemory() {
    if (this.metrics.memory.length === 0) {
      return { analysis: 'No memory data available' };
    }

    const usage = this.metrics.memory.map(m => m.used);
    const peak = Math.max(...usage);
    const average = usage.reduce((a, b) => a + b, 0) / usage.length;
    const growth = usage.length > 1 ? usage[usage.length - 1] - usage[0] : 0;

    return {
      peak,
      average,
      growth,
      samples: this.metrics.memory.length,
      leakSuspicion: growth > this.warningThresholds.memoryLeakRate
    };
  }

  /**
   * Generate optimization recommendations
   * @returns {Array} List of recommendations
   */
  generateRecommendations() {
    const recommendations = [];

    // Analyze operations for recommendations
    for (const [name, stats] of this.metrics.operations.entries()) {
      if (stats.averageTime > this.warningThresholds.operationTime) {
        recommendations.push({
          type: 'performance',
          priority: 'high',
          operation: name,
          message: `Consider optimizing '${name}' operation (avg: ${stats.averageTime.toFixed(2)}ms)`,
          suggestions: [
            'Use batch processing for multiple inputs',
            'Enable WebGPU acceleration if available',
            'Consider model quantization',
            'Use memory pooling'
          ]
        });
      }

      if (stats.averageMemoryDelta > this.warningThresholds.memoryUsage) {
        recommendations.push({
          type: 'memory',
          priority: 'medium',
          operation: name,
          message: `'${name}' operation uses significant memory (avg: ${(stats.averageMemoryDelta / 1024 / 1024).toFixed(2)}MB)`,
          suggestions: [
            'Use tensor memory pooling',
            'Enable automatic cleanup',
            'Process data in smaller batches',
            'Consider model compression'
          ]
        });
      }
    }

    // Memory analysis recommendations
    const memoryAnalysis = this.analyzeMemory();
    if (memoryAnalysis.leakSuspicion) {
      recommendations.push({
        type: 'memory_leak',
        priority: 'high',
        message: 'Potential memory leak detected',
        suggestions: [
          'Ensure proper tensor cleanup',
          'Use memory management utilities',
          'Check for circular references',
          'Monitor long-running operations'
        ]
      });
    }

    return recommendations;
  }

  /**
   * Aggregate session operations
   * @param {Array} operations - Session operations
   * @returns {Object} Aggregated stats
   */
  aggregateSessionOperations(operations) {
    const stats = {};
    
    for (const op of operations) {
      if (!stats[op.operation]) {
        stats[op.operation] = {
          count: 0,
          totalTime: 0,
          totalMemoryDelta: 0,
          errors: 0
        };
      }
      
      stats[op.operation].count++;
      stats[op.operation].totalTime += op.duration;
      stats[op.operation].totalMemoryDelta += op.memoryDelta;
      
      if (op.error) {
        stats[op.operation].errors++;
      }
    }

    // Calculate averages
    for (const stat of Object.values(stats)) {
      stat.averageTime = stat.totalTime / stat.count;
      stat.averageMemoryDelta = stat.totalMemoryDelta / stat.count;
    }

    return stats;
  }

  /**
   * Analyze session memory usage
   * @param {Object} session - Session data
   * @returns {Object} Memory analysis
   */
  analyzeSessionMemory(session) {
    if (session.operations.length === 0) {
      return { analysis: 'No operations in session' };
    }

    const memoryDeltas = session.operations.map(op => op.memoryDelta);
    const totalDelta = memoryDeltas.reduce((a, b) => a + b, 0);
    const maxDelta = Math.max(...memoryDeltas);
    const minDelta = Math.min(...memoryDeltas);

    return {
      totalMemoryChange: totalDelta,
      maxMemoryDelta: maxDelta,
      minMemoryDelta: minDelta,
      averageMemoryDelta: totalDelta / memoryDeltas.length
    };
  }

  /**
   * Generate session-specific recommendations
   * @param {Object} session - Session data
   * @returns {Array} Recommendations
   */
  generateSessionRecommendations(session) {
    const recommendations = [];
    
    if (session.duration > 5000) { // 5 seconds
      recommendations.push({
        type: 'session_performance',
        message: `Session '${session.name}' took ${(session.duration / 1000).toFixed(2)} seconds`,
        suggestions: ['Consider breaking into smaller operations', 'Use async processing']
      });
    }

    if (session.warnings.length > 0) {
      recommendations.push({
        type: 'session_warnings',
        message: `Session had ${session.warnings.length} performance warnings`,
        suggestions: ['Review operation timings', 'Check memory usage patterns']
      });
    }

    return recommendations;
  }

  /**
   * Start automatic reporting
   */
  startAutoReporting() {
    this.reportInterval = setInterval(() => {
      const report = this.generateReport();
      console.warn('TrustformeRS Performance Report:', report);
    }, this.reportInterval);
  }

  /**
   * Export metrics data
   * @param {string} format - Export format ('json', 'csv')
   * @returns {string} Exported data
   */
  exportMetrics(format = 'json') {
    const data = {
      timestamp: new Date().toISOString(),
      metrics: this.metrics,
      summary: this.generateSummary()
    };

    if (format === 'json') {
      return JSON.stringify(data, null, 2);
    } else if (format === 'csv') {
      // Convert operations to CSV
      const operations = Array.from(this.metrics.operations.entries())
        .map(([name, stats]) => `${name},${stats.count},${stats.averageTime},${stats.errors}`)
        .join('\n');
      
      return `Operation,Count,Average Time (ms),Errors\n${operations}`;
    }

    throw new Error(`Unsupported export format: ${format}`);
  }

  /**
   * Clear all metrics
   */
  clear() {
    this.metrics = {
      operations: new Map(),
      memory: [],
      gpu: [],
      inference: [],
      timeline: [],
      sessions: []
    };
    this.currentSession = null;
    this.sessionStack = [];
  }

  /**
   * Dispose the profiler
   */
  dispose() {
    if (this.reportInterval) {
      clearInterval(this.reportInterval);
    }
    this.clear();
  }
}

// Global profiler instance
let globalProfiler = null;

/**
 * Get or create global profiler
 * @param {Object} options - Profiler options
 * @returns {PerformanceProfiler} Global profiler
 */
export function getProfiler(options = {}) {
  if (!globalProfiler) {
    globalProfiler = new PerformanceProfiler(options);
  }
  return globalProfiler;
}

/**
 * Utility function to profile any function
 * @param {string} name - Operation name
 * @param {Function} fn - Function to profile
 * @param {Object} options - Profiling options
 * @returns {Promise<*>} Function result
 */
export async function profile(name, fn, options = {}) {
  const profiler = getProfiler();
  return await profiler.profileOperation(name, fn, options.metadata);
}

/**
 * Decorator for profiling class methods
 * @param {string} operationName - Optional operation name
 * @returns {Function} Method decorator
 */
export function profileMethod(operationName = null) {
  return function(target, propertyKey, descriptor) {
    const originalMethod = descriptor.value;
    const name = operationName || `${target.constructor.name}.${propertyKey}`;

    descriptor.value = async function(...args) {
      const profiler = getProfiler();
      return await profiler.profileOperation(name, () => originalMethod.apply(this, args));
    };

    return descriptor;
  };
}