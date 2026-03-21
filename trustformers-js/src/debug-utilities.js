/**
 * TrustformeRS Debug Utilities
 * Comprehensive debugging tools for development and production environments
 */

import { performance } from 'perf_hooks';

/**
 * Debug configuration
 */
let debugConfig = {
  enabled: false,
  logLevel: 'info', // 'error', 'warn', 'info', 'debug', 'trace'
  maxHistorySize: 1000,
  enableStackTraces: true,
  enablePerformanceTracking: true,
  enableMemoryTracking: true,
  enableTensorLifecycle: true,
  enableOperationValidation: true
};

/**
 * Debug data storage
 */
const debugData = {
  operations: [],
  tensors: new Map(),
  errors: [],
  warnings: [],
  performance: [],
  memory: [],
  sessions: new Map()
};

/**
 * Debug session manager
 */
class DebugSession {
  constructor(name, metadata = {}) {
    this.id = `debug_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    this.name = name;
    this.metadata = metadata;
    this.startTime = performance.now();
    this.operations = [];
    this.tensors = new Set();
    this.warnings = [];
    this.errors = [];
    this.active = true;
  }

  addOperation(operation) {
    if (this.active) {
      this.operations.push({
        ...operation,
        sessionId: this.id,
        timestamp: performance.now()
      });
    }
  }

  addTensor(tensorId, info) {
    if (this.active) {
      this.tensors.add(tensorId);
    }
  }

  addError(error) {
    if (this.active) {
      this.errors.push({
        ...error,
        sessionId: this.id,
        timestamp: performance.now()
      });
    }
  }

  addWarning(warning) {
    if (this.active) {
      this.warnings.push({
        ...warning,
        sessionId: this.id,
        timestamp: performance.now()
      });
    }
  }

  end() {
    this.active = false;
    this.endTime = performance.now();
    this.duration = this.endTime - this.startTime;
    return this.generateReport();
  }

  generateReport() {
    return {
      session: {
        id: this.id,
        name: this.name,
        metadata: this.metadata,
        duration: this.duration,
        startTime: this.startTime,
        endTime: this.endTime
      },
      summary: {
        operationCount: this.operations.length,
        tensorCount: this.tensors.size,
        errorCount: this.errors.length,
        warningCount: this.warnings.length
      },
      operations: this.operations,
      tensors: Array.from(this.tensors),
      errors: this.errors,
      warnings: this.warnings
    };
  }
}

/**
 * Main debug utilities class
 */
export class DebugUtilities {
  constructor() {
    this.currentSession = null;
    this.initializeConsoleOverrides();
  }

  /**
   * Configure debug settings
   * @param {Object} config - Debug configuration
   */
  configure(config) {
    debugConfig = { ...debugConfig, ...config };
    this.log('Debug configuration updated', config);
  }

  /**
   * Enable debugging
   * @param {Object} options - Debug options
   */
  enable(options = {}) {
    debugConfig.enabled = true;
    debugConfig = { ...debugConfig, ...options };
    this.log('Debug mode enabled', debugConfig);
  }

  /**
   * Disable debugging
   */
  disable() {
    debugConfig.enabled = false;
    this.log('Debug mode disabled');
  }

  /**
   * Check if debug is enabled
   * @returns {boolean} Debug status
   */
  isEnabled() {
    return debugConfig.enabled;
  }

  /**
   * Start a debug session
   * @param {string} name - Session name
   * @param {Object} metadata - Session metadata
   * @returns {string} Session ID
   */
  startSession(name, metadata = {}) {
    if (!debugConfig.enabled) return null;

    this.endSession(); // End current session if any
    this.currentSession = new DebugSession(name, metadata);
    debugData.sessions.set(this.currentSession.id, this.currentSession);
    this.log(`Debug session started: ${name}`, { id: this.currentSession.id });
    return this.currentSession.id;
  }

  /**
   * End current debug session
   * @returns {Object|null} Session report
   */
  endSession() {
    if (this.currentSession && this.currentSession.active) {
      const report = this.currentSession.end();
      this.log(`Debug session ended: ${this.currentSession.name}`, {
        duration: report.session.duration,
        operations: report.summary.operationCount
      });
      this.currentSession = null;
      return report;
    }
    return null;
  }

  /**
   * Track tensor lifecycle
   * @param {Object} tensor - Tensor object
   * @param {string} operation - Operation that created the tensor
   * @param {Object} metadata - Additional metadata
   */
  trackTensor(tensor, operation, metadata = {}) {
    if (!debugConfig.enabled || !debugConfig.enableTensorLifecycle) return;

    const tensorId = this.getTensorId(tensor);
    const info = {
      id: tensorId,
      operation,
      metadata,
      shape: this.getTensorShape(tensor),
      dtype: this.getTensorDType(tensor),
      creationTime: performance.now(),
      creationStack: debugConfig.enableStackTraces ? this.getStackTrace() : null
    };

    debugData.tensors.set(tensorId, info);
    
    if (this.currentSession) {
      this.currentSession.addTensor(tensorId, info);
    }

    this.log(`Tensor created: ${tensorId}`, info);
  }

  /**
   * Track operation execution
   * @param {string} name - Operation name
   * @param {Function} fn - Function to track
   * @param {Object} metadata - Operation metadata
   * @returns {Promise<*>} Function result
   */
  async trackOperation(name, fn, metadata = {}) {
    if (!debugConfig.enabled) {
      return await fn();
    }

    const operation = {
      name,
      metadata,
      startTime: performance.now(),
      stack: debugConfig.enableStackTraces ? this.getStackTrace() : null
    };

    try {
      const result = await fn();
      operation.endTime = performance.now();
      operation.duration = operation.endTime - operation.startTime;
      operation.success = true;
      operation.resultType = this.getResultType(result);

      this.addOperation(operation);
      this.log(`Operation completed: ${name}`, {
        duration: operation.duration,
        success: true
      });

      return result;
    } catch (error) {
      operation.endTime = performance.now();
      operation.duration = operation.endTime - operation.startTime;
      operation.success = false;
      operation.error = {
        message: error.message,
        stack: error.stack,
        name: error.name
      };

      this.addOperation(operation);
      this.error(`Operation failed: ${name}`, error);
      throw error;
    }
  }

  /**
   * Validate tensor operation
   * @param {string} operation - Operation name
   * @param {Array} tensors - Input tensors
   * @param {Object} options - Validation options
   * @returns {boolean} Validation result
   */
  validateOperation(operation, tensors, options = {}) {
    if (!debugConfig.enabled || !debugConfig.enableOperationValidation) {
      return true;
    }

    const validation = {
      operation,
      tensors: tensors.map(t => ({
        shape: this.getTensorShape(t),
        dtype: this.getTensorDType(t),
        id: this.getTensorId(t)
      })),
      timestamp: performance.now(),
      warnings: [],
      errors: []
    };

    // Shape compatibility checks
    if (operation === 'matmul' && tensors.length >= 2) {
      const [a, b] = tensors;
      const shapeA = this.getTensorShape(a);
      const shapeB = this.getTensorShape(b);
      
      if (shapeA.length < 2 || shapeB.length < 2) {
        validation.errors.push('Matrix multiplication requires tensors with at least 2 dimensions');
      } else if (shapeA[shapeA.length - 1] !== shapeB[shapeB.length - 2]) {
        validation.errors.push(`Incompatible shapes for matrix multiplication: ${shapeA} @ ${shapeB}`);
      }
    }

    // Element-wise operation checks
    if (['add', 'sub', 'mul', 'div'].includes(operation) && tensors.length >= 2) {
      const shapes = tensors.map(t => this.getTensorShape(t));
      if (!this.areShapesBroadcastable(shapes[0], shapes[1])) {
        validation.warnings.push(`Shapes may not be broadcastable: ${shapes[0]} and ${shapes[1]}`);
      }
    }

    // Memory usage warnings
    const totalMemory = tensors.reduce((sum, tensor) => sum + this.estimateTensorMemory(tensor), 0);

    if (totalMemory > 100 * 1024 * 1024) { // 100MB
      validation.warnings.push(`Large memory usage detected: ${(totalMemory / 1024 / 1024).toFixed(2)}MB`);
    }

    // Log validation results
    if (validation.errors.length > 0) {
      validation.errors.forEach(error => this.error(`Validation error in ${operation}:`, error));
    }
    if (validation.warnings.length > 0) {
      validation.warnings.forEach(warning => this.warn(`Validation warning in ${operation}:`, warning));
    }

    return validation.errors.length === 0;
  }

  /**
   * Create a debugger decorator for functions
   * @param {string} name - Operation name
   * @param {Object} options - Debug options
   * @returns {Function} Decorator function
   */
  createDebugDecorator(name, options = {}) {
    return (target, propertyKey, descriptor) => {
      const originalMethod = descriptor.value;
      
      descriptor.value = async function(...args) {
        if (!debugConfig.enabled) {
          return await originalMethod.apply(this, args);
        }

        return await this.trackOperation(
          `${name}.${propertyKey}`,
          () => originalMethod.apply(this, args),
          { args: options.logArgs ? args : undefined }
        );
      };

      return descriptor;
    };
  }

  /**
   * Get memory usage information
   * @returns {Object} Memory usage stats
   */
  getMemoryUsage() {
    const memoryInfo = {
      timestamp: performance.now(),
      jsHeap: null,
      wasmMemory: null,
      tensorCount: debugData.tensors.size,
      activeTensors: []
    };

    // JavaScript heap info
    if (performance.memory) {
      memoryInfo.jsHeap = {
        used: performance.memory.usedJSHeapSize,
        total: performance.memory.totalJSHeapSize,
        limit: performance.memory.jsHeapSizeLimit
      };
    }

    // Active tensors info
    debugData.tensors.forEach((info, id) => {
      memoryInfo.activeTensors.push({
        id,
        shape: info.shape,
        dtype: info.dtype,
        estimatedSize: this.estimateTensorSize(info.shape, info.dtype),
        age: performance.now() - info.creationTime
      });
    });

    if (debugConfig.enableMemoryTracking) {
      debugData.memory.push(memoryInfo);
      this.trimHistory(debugData.memory);
    }

    return memoryInfo;
  }

  /**
   * Get performance metrics
   * @returns {Object} Performance metrics
   */
  getPerformanceMetrics() {
    const {operations} = debugData;
    const metrics = {
      totalOperations: operations.length,
      averageDuration: 0,
      operationsByType: {},
      slowestOperations: [],
      recentOperations: operations.slice(-10)
    };

    if (operations.length > 0) {
      const totalDuration = operations.reduce((sum, op) => sum + op.duration, 0);
      metrics.averageDuration = totalDuration / operations.length;

      // Group by operation type
      operations.forEach(op => {
        if (!metrics.operationsByType[op.name]) {
          metrics.operationsByType[op.name] = {
            count: 0,
            totalDuration: 0,
            averageDuration: 0
          };
        }
        metrics.operationsByType[op.name].count++;
        metrics.operationsByType[op.name].totalDuration += op.duration;
      });

      // Calculate averages
      Object.keys(metrics.operationsByType).forEach(name => {
        const type = metrics.operationsByType[name];
        type.averageDuration = type.totalDuration / type.count;
      });

      // Find slowest operations
      metrics.slowestOperations = [...operations]
        .sort((a, b) => b.duration - a.duration)
        .slice(0, 5);
    }

    return metrics;
  }

  /**
   * Generate comprehensive debug report
   * @returns {Object} Debug report
   */
  generateReport() {
    return {
      timestamp: new Date().toISOString(),
      config: debugConfig,
      summary: {
        operationCount: debugData.operations.length,
        tensorCount: debugData.tensors.size,
        errorCount: debugData.errors.length,
        warningCount: debugData.warnings.length,
        sessionCount: debugData.sessions.size
      },
      memory: this.getMemoryUsage(),
      performance: this.getPerformanceMetrics(),
      recentErrors: debugData.errors.slice(-5),
      recentWarnings: debugData.warnings.slice(-5),
      activeSessions: Array.from(debugData.sessions.values())
        .filter(session => session.active)
        .map(session => ({
          id: session.id,
          name: session.name,
          duration: performance.now() - session.startTime,
          operationCount: session.operations.length
        }))
    };
  }

  /**
   * Export debug data
   * @param {string} format - Export format ('json', 'csv')
   * @returns {string} Exported data
   */
  exportData(format = 'json') {
    const report = this.generateReport();
    
    switch (format.toLowerCase()) {
      case 'json':
        return JSON.stringify(report, null, 2);
      case 'csv':
        return this.generateCSVReport(report);
      default:
        throw new Error(`Unsupported export format: ${format}`);
    }
  }

  /**
   * Clear debug data
   */
  clear() {
    debugData.operations.length = 0;
    debugData.tensors.clear();
    debugData.errors.length = 0;
    debugData.warnings.length = 0;
    debugData.performance.length = 0;
    debugData.memory.length = 0;
    debugData.sessions.clear();
    this.currentSession = null;
    this.log('Debug data cleared');
  }

  // Private helper methods
  getTensorId(tensor) {
    if (tensor._debugId) return tensor._debugId;
    tensor._debugId = `tensor_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    return tensor._debugId;
  }

  getTensorShape(tensor) {
    if (tensor.shape && typeof tensor.shape === 'function') {
      return Array.from(tensor.shape());
    }
    if (tensor.shape && Array.isArray(tensor.shape)) {
      return tensor.shape;
    }
    return [];
  }

  getTensorDType(tensor) {
    if (tensor.dtype && typeof tensor.dtype === 'function') {
      return tensor.dtype();
    }
    if (tensor.dtype) {
      return tensor.dtype;
    }
    return 'unknown';
  }

  estimateTensorMemory(tensor) {
    const shape = this.getTensorShape(tensor);
    const dtype = this.getTensorDType(tensor);
    return this.estimateTensorSize(shape, dtype);
  }

  estimateTensorSize(shape, dtype) {
    const elements = shape.reduce((product, dim) => product * dim, 1);
    const bytesPerElement = this.getBytesPerElement(dtype);
    return elements * bytesPerElement;
  }

  getBytesPerElement(dtype) {
    const typeMap = {
      'f32': 4, 'float32': 4,
      'f64': 8, 'float64': 8,
      'i32': 4, 'int32': 4,
      'i64': 8, 'int64': 8,
      'u32': 4, 'uint32': 4,
      'i8': 1, 'int8': 1,
      'u8': 1, 'uint8': 1,
      'bool': 1
    };
    return typeMap[dtype] || 4;
  }

  areShapesBroadcastable(shape1, shape2) {
    const len1 = shape1.length;
    const len2 = shape2.length;
    const maxLen = Math.max(len1, len2);
    
    for (let i = 0; i < maxLen; i++) {
      const dim1 = i < len1 ? shape1[len1 - 1 - i] : 1;
      const dim2 = i < len2 ? shape2[len2 - 1 - i] : 1;
      
      if (dim1 !== dim2 && dim1 !== 1 && dim2 !== 1) {
        return false;
      }
    }
    return true;
  }

  getStackTrace() {
    try {
      throw new Error();
    } catch (e) {
      return e.stack.split('\n').slice(3, 8).join('\n');
    }
  }

  getResultType(result) {
    if (result === null) return 'null';
    if (result === undefined) return 'undefined';
    if (Array.isArray(result)) return 'array';
    return typeof result;
  }

  addOperation(operation) {
    debugData.operations.push(operation);
    this.trimHistory(debugData.operations);
    
    if (this.currentSession) {
      this.currentSession.addOperation(operation);
    }
  }

  trimHistory(array) {
    if (array.length > debugConfig.maxHistorySize) {
      array.splice(0, array.length - debugConfig.maxHistorySize);
    }
  }

  initializeConsoleOverrides() {
    // Store original console methods
    this.originalConsole = {
      log: console.log.bind(console),
      warn: console.warn.bind(console),
      error: console.error.bind(console),
      debug: console.debug.bind(console)
    };
  }

  shouldLog(level) {
    const levels = ['error', 'warn', 'info', 'debug', 'trace'];
    const currentLevelIndex = levels.indexOf(debugConfig.logLevel);
    const messageLevelIndex = levels.indexOf(level);
    return messageLevelIndex <= currentLevelIndex;
  }

  log(message, data = {}) {
    if (!debugConfig.enabled || !this.shouldLog('info')) return;
    this.originalConsole.log(`[TrustformeRS Debug]`, message, data);
  }

  warn(message, data = {}) {
    if (!debugConfig.enabled || !this.shouldLog('warn')) return;
    this.originalConsole.warn(`[TrustformeRS Warning]`, message, data);
    
    const warning = {
      message,
      data,
      timestamp: performance.now(),
      stack: debugConfig.enableStackTraces ? this.getStackTrace() : null
    };
    
    debugData.warnings.push(warning);
    this.trimHistory(debugData.warnings);
    
    if (this.currentSession) {
      this.currentSession.addWarning(warning);
    }
  }

  error(message, error = {}) {
    if (!debugConfig.enabled || !this.shouldLog('error')) return;
    this.originalConsole.error(`[TrustformeRS Error]`, message, error);
    
    const errorInfo = {
      message,
      error: {
        message: error.message || error,
        stack: error.stack,
        name: error.name
      },
      timestamp: performance.now(),
      stack: debugConfig.enableStackTraces ? this.getStackTrace() : null
    };
    
    debugData.errors.push(errorInfo);
    this.trimHistory(debugData.errors);
    
    if (this.currentSession) {
      this.currentSession.addError(errorInfo);
    }
  }

  generateCSVReport(report) {
    const lines = ['Operation,Duration,Success,Timestamp'];
    
    debugData.operations.forEach(op => {
      lines.push(`"${op.name}",${op.duration},${op.success},${op.startTime}`);
    });
    
    return lines.join('\n');
  }
}

// Global debug instance
export const debugUtils = new DebugUtilities();

// Export convenience functions
export const debug = {
  enable: (options) => debugUtils.enable(options),
  disable: () => debugUtils.disable(),
  configure: (config) => debugUtils.configure(config),
  isEnabled: () => debugUtils.isEnabled(),
  
  startSession: (name, metadata) => debugUtils.startSession(name, metadata),
  endSession: () => debugUtils.endSession(),
  
  trackTensor: (tensor, operation, metadata) => debugUtils.trackTensor(tensor, operation, metadata),
  trackOperation: (name, fn, metadata) => debugUtils.trackOperation(name, fn, metadata),
  validateOperation: (operation, tensors, options) => debugUtils.validateOperation(operation, tensors, options),
  
  getMemoryUsage: () => debugUtils.getMemoryUsage(),
  getPerformanceMetrics: () => debugUtils.getPerformanceMetrics(),
  generateReport: () => debugUtils.generateReport(),
  exportData: (format) => debugUtils.exportData(format),
  clear: () => debugUtils.clear()
};

// Debug decorator
export function debugOperation(name, options = {}) {
  return debugUtils.createDebugDecorator(name, options);
}