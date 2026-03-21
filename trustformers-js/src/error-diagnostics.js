/**
 * TrustformeRS Error Diagnostic Tools
 * Comprehensive error analysis, diagnosis, and resolution assistance
 */

import { debugUtils } from './debug-utilities.js';
import { tensorInspector } from './tensor-inspector.js';
import { modelVisualizer } from './model-visualization.js';

/**
 * Error classification types
 */
export const ErrorTypes = {
  INITIALIZATION: 'initialization',
  TENSOR_OPERATION: 'tensor_operation',
  MODEL_LOADING: 'model_loading',
  INFERENCE: 'inference',
  MEMORY: 'memory',
  WEBGL: 'webgl',
  WASM: 'wasm',
  NETWORK: 'network',
  VALIDATION: 'validation',
  RUNTIME: 'runtime',
  UNKNOWN: 'unknown'
};

/**
 * Error severity levels
 */
export const ErrorSeverity = {
  CRITICAL: 'critical',
  HIGH: 'high',
  MEDIUM: 'medium',
  LOW: 'low',
  INFO: 'info'
};

/**
 * Error diagnostic system
 */
export class ErrorDiagnostics {
  constructor() {
    this.errorHistory = [];
    this.errorPatterns = new Map();
    this.solutions = new Map();
    this.diagnosticCache = new Map();
    this.errorHandlers = new Map();
    this.initializeErrorHandlers();
    this.initializeSolutions();
  }

  /**
   * Diagnose an error and provide detailed analysis
   * @param {Error} error - Error object
   * @param {Object} context - Error context
   * @param {Object} options - Diagnostic options
   * @returns {Object} Diagnostic result
   */
  diagnose(error, context = {}, options = {}) {
    const {
      includeStackTrace = true,
      includeContext = true,
      includeSolutions = true,
      analyzePatterns = true,
      cacheResults = true
    } = options;

    const errorInfo = this.extractErrorInfo(error, context);
    const cacheKey = this.generateCacheKey(errorInfo);

    if (cacheResults && this.diagnosticCache.has(cacheKey)) {
      return this.diagnosticCache.get(cacheKey);
    }

    const diagnostic = {
      id: `diag_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toISOString(),
      error: errorInfo,
      classification: this.classifyError(error, context),
      severity: this.assessSeverity(error, context),
      context: includeContext ? this.analyzeContext(context) : null,
      stackTrace: includeStackTrace ? this.analyzeStackTrace(error) : null,
      patterns: analyzePatterns ? this.analyzePatterns(error) : null,
      solutions: includeSolutions ? this.generateSolutions(error, context) : null,
      relatedErrors: this.findRelatedErrors(error),
      metadata: {
        userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'Node.js',
        platform: typeof process !== 'undefined' ? process.platform : 'browser',
        memory: this.getMemoryInfo(),
        performance: this.getPerformanceInfo()
      }
    };

    // Store in history
    this.errorHistory.push(diagnostic);
    this.trimHistory();

    // Update patterns
    this.updatePatterns(diagnostic);

    if (cacheResults) {
      this.diagnosticCache.set(cacheKey, diagnostic);
    }

    return diagnostic;
  }

  /**
   * Extract comprehensive error information
   * @param {Error} error - Error object
   * @param {Object} context - Error context
   * @returns {Object} Error information
   */
  extractErrorInfo(error, context) {
    return {
      name: error.name,
      message: error.message,
      stack: error.stack,
      code: error.code,
      cause: error.cause,
      constructor: error.constructor.name,
      isCustom: error.constructor !== Error,
      contextData: context.data || null,
      operation: context.operation || null,
      inputs: context.inputs || null,
      expectedOutput: context.expectedOutput || null,
      actualOutput: context.actualOutput || null
    };
  }

  /**
   * Classify error type
   * @param {Error} error - Error object
   * @param {Object} context - Error context
   * @returns {string} Error type
   */
  classifyError(error, context) {
    const message = error.message.toLowerCase();
    const stack = error.stack ? error.stack.toLowerCase() : '';
    const operation = context.operation ? context.operation.toLowerCase() : '';

    // Check for specific patterns
    if (message.includes('wasm') || message.includes('webassembly')) {
      return ErrorTypes.WASM;
    }
    if (message.includes('webgl') || message.includes('gl')) {
      return ErrorTypes.WEBGL;
    }
    if (message.includes('memory') || message.includes('allocation')) {
      return ErrorTypes.MEMORY;
    }
    if (message.includes('network') || message.includes('fetch') || message.includes('request')) {
      return ErrorTypes.NETWORK;
    }
    if (message.includes('initialization') || message.includes('not initialized')) {
      return ErrorTypes.INITIALIZATION;
    }
    if (message.includes('tensor') || operation.includes('tensor')) {
      return ErrorTypes.TENSOR_OPERATION;
    }
    if (message.includes('model') || operation.includes('model')) {
      return ErrorTypes.MODEL_LOADING;
    }
    if (message.includes('inference') || operation.includes('inference')) {
      return ErrorTypes.INFERENCE;
    }
    if (message.includes('validation') || message.includes('invalid')) {
      return ErrorTypes.VALIDATION;
    }

    // Check stack trace
    if (stack.includes('tensor') || stack.includes('model')) {
      return ErrorTypes.TENSOR_OPERATION;
    }

    return ErrorTypes.UNKNOWN;
  }

  /**
   * Assess error severity
   * @param {Error} error - Error object
   * @param {Object} context - Error context
   * @returns {string} Severity level
   */
  assessSeverity(error, context) {
    const message = error.message.toLowerCase();
    const type = this.classifyError(error, context);

    // Critical errors
    if (message.includes('segfault') || message.includes('panic') || message.includes('abort')) {
      return ErrorSeverity.CRITICAL;
    }
    if (type === ErrorTypes.INITIALIZATION && message.includes('failed to initialize')) {
      return ErrorSeverity.CRITICAL;
    }

    // High severity
    if (type === ErrorTypes.MEMORY && message.includes('out of memory')) {
      return ErrorSeverity.HIGH;
    }
    if (type === ErrorTypes.WASM && message.includes('compilation failed')) {
      return ErrorSeverity.HIGH;
    }
    if (message.includes('nan') || message.includes('infinite') || message.includes('overflow')) {
      return ErrorSeverity.HIGH;
    }

    // Medium severity
    if (type === ErrorTypes.TENSOR_OPERATION || type === ErrorTypes.INFERENCE) {
      return ErrorSeverity.MEDIUM;
    }
    if (type === ErrorTypes.VALIDATION) {
      return ErrorSeverity.MEDIUM;
    }

    // Low severity
    if (type === ErrorTypes.NETWORK) {
      return ErrorSeverity.LOW;
    }

    return ErrorSeverity.MEDIUM;
  }

  /**
   * Analyze error context
   * @param {Object} context - Error context
   * @returns {Object} Context analysis
   */
  analyzeContext(context) {
    const analysis = {
      operation: context.operation || null,
      operationType: null,
      inputs: null,
      environment: this.getEnvironmentInfo(),
      memoryState: this.getMemoryState(),
      systemState: this.getSystemState()
    };

    // Analyze operation
    if (context.operation) {
      analysis.operationType = this.classifyOperation(context.operation);
    }

    // Analyze inputs
    if (context.inputs) {
      analysis.inputs = this.analyzeInputs(context.inputs);
    }

    // Analyze tensors if present
    if (context.tensors) {
      analysis.tensors = context.tensors.map(tensor => {
        try {
          return tensorInspector.summarize(tensor);
        } catch (e) {
          return { error: e.message, id: tensor.id || 'unknown' };
        }
      });
    }

    // Analyze model if present
    if (context.model) {
      try {
        analysis.model = modelVisualizer.summarize(context.model);
      } catch (e) {
        analysis.model = { error: e.message };
      }
    }

    return analysis;
  }

  /**
   * Analyze stack trace
   * @param {Error} error - Error object
   * @returns {Object} Stack trace analysis
   */
  analyzeStackTrace(error) {
    if (!error.stack) return null;

    const lines = error.stack.split('\n');
    const analysis = {
      totalLines: lines.length,
      errorLine: lines[0],
      frames: [],
      libraries: new Set(),
      userCode: [],
      systemCode: []
    };

    lines.slice(1).forEach((line, index) => {
      const frame = this.parseStackFrame(line, index);
      if (frame) {
        analysis.frames.push(frame);
        
        if (frame.library) {
          analysis.libraries.add(frame.library);
        }
        
        if (frame.isUserCode) {
          analysis.userCode.push(frame);
        } else {
          analysis.systemCode.push(frame);
        }
      }
    });

    // Identify likely error source
    analysis.likelySource = this.identifyErrorSource(analysis.frames);

    return analysis;
  }

  /**
   * Analyze error patterns
   * @param {Error} error - Error object
   * @returns {Object} Pattern analysis
   */
  analyzePatterns(error) {
    const patterns = {
      frequency: this.getErrorFrequency(error),
      recentOccurrences: this.getRecentOccurrences(error),
      timePattern: this.getTimePattern(error),
      correlations: this.getCorrelations(error),
      trend: this.getTrend(error)
    };

    return patterns;
  }

  /**
   * Generate solutions and recommendations
   * @param {Error} error - Error object
   * @param {Object} context - Error context
   * @returns {Object} Solutions and recommendations
   */
  generateSolutions(error, context) {
    const type = this.classifyError(error, context);
    const message = error.message.toLowerCase();
    
    const solutions = {
      immediate: [],
      preventive: [],
      diagnostic: [],
      references: []
    };

    // Get type-specific solutions
    if (this.solutions.has(type)) {
      const typeSolutions = this.solutions.get(type);
      solutions.immediate.push(...typeSolutions.immediate);
      solutions.preventive.push(...typeSolutions.preventive);
      solutions.diagnostic.push(...typeSolutions.diagnostic);
      solutions.references.push(...typeSolutions.references);
    }

    // Get pattern-specific solutions
    const patternSolutions = this.getPatternSolutions(error, context);
    solutions.immediate.push(...patternSolutions.immediate);
    solutions.preventive.push(...patternSolutions.preventive);

    // Get context-specific solutions
    const contextSolutions = this.getContextSolutions(error, context);
    solutions.immediate.push(...contextSolutions.immediate);
    solutions.preventive.push(...contextSolutions.preventive);

    // Generate automatic fixes if possible
    solutions.autoFixes = this.generateAutoFixes(error, context);

    return solutions;
  }

  /**
   * Find related errors
   * @param {Error} error - Error object
   * @returns {Array} Related errors
   */
  findRelatedErrors(error) {
    const related = [];
    const currentMessage = error.message.toLowerCase();
    
    this.errorHistory.forEach(diagnostic => {
      if (diagnostic.error.message.toLowerCase().includes(currentMessage.substring(0, 20))) {
        related.push({
          id: diagnostic.id,
          timestamp: diagnostic.timestamp,
          message: diagnostic.error.message,
          type: diagnostic.classification,
          similarity: this.calculateSimilarity(error.message, diagnostic.error.message)
        });
      }
    });

    return related.sort((a, b) => b.similarity - a.similarity).slice(0, 5);
  }

  /**
   * Generate comprehensive error report
   * @param {Error} error - Error object
   * @param {Object} context - Error context
   * @returns {Object} Error report
   */
  generateReport(error, context = {}) {
    const diagnostic = this.diagnose(error, context);
    
    const report = {
      summary: {
        type: diagnostic.classification,
        severity: diagnostic.severity,
        message: diagnostic.error.message,
        timestamp: diagnostic.timestamp,
        id: diagnostic.id
      },
      details: {
        errorInfo: diagnostic.error,
        context: diagnostic.context,
        stackTrace: diagnostic.stackTrace,
        patterns: diagnostic.patterns
      },
      solutions: diagnostic.solutions,
      recommendations: this.generateRecommendations(diagnostic),
      nextSteps: this.generateNextSteps(diagnostic),
      metadata: diagnostic.metadata
    };

    return report;
  }

  /**
   * Generate text report
   * @param {Error} error - Error object
   * @param {Object} context - Error context
   * @returns {string} Text report
   */
  generateTextReport(error, context = {}) {
    const report = this.generateReport(error, context);
    
    let text = `TrustformeRS Error Report\n`;
    text += `========================\n\n`;
    text += `Error ID: ${report.summary.id}\n`;
    text += `Timestamp: ${report.summary.timestamp}\n`;
    text += `Type: ${report.summary.type}\n`;
    text += `Severity: ${report.summary.severity}\n`;
    text += `Message: ${report.summary.message}\n\n`;

    if (report.details.context) {
      text += `Context:\n`;
      text += `--------\n`;
      text += `Operation: ${report.details.context.operation || 'unknown'}\n`;
      text += `Environment: ${report.details.context.environment.type}\n`;
      text += `Memory Usage: ${report.details.context.memoryState.usage}MB\n\n`;
    }

    if (report.solutions) {
      text += `Immediate Solutions:\n`;
      text += `-------------------\n`;
      report.solutions.immediate.forEach((solution, index) => {
        text += `${index + 1}. ${solution.description}\n`;
        if (solution.code) {
          text += `   Code: ${solution.code}\n`;
        }
      });
      text += `\n`;

      text += `Preventive Measures:\n`;
      text += `-------------------\n`;
      report.solutions.preventive.forEach((solution, index) => {
        text += `${index + 1}. ${solution.description}\n`;
      });
      text += `\n`;
    }

    if (report.recommendations.length > 0) {
      text += `Recommendations:\n`;
      text += `---------------\n`;
      report.recommendations.forEach((rec, index) => {
        text += `${index + 1}. ${rec}\n`;
      });
      text += `\n`;
    }

    if (report.nextSteps.length > 0) {
      text += `Next Steps:\n`;
      text += `----------\n`;
      report.nextSteps.forEach((step, index) => {
        text += `${index + 1}. ${step}\n`;
      });
    }

    return text;
  }

  /**
   * Generate HTML report
   * @param {Error} error - Error object
   * @param {Object} context - Error context
   * @returns {string} HTML report
   */
  generateHTMLReport(error, context = {}) {
    const report = this.generateReport(error, context);
    
    let html = `
    <div class="error-report" style="font-family: 'Courier New', monospace; border: 1px solid #dc3545; padding: 20px; margin: 10px; background: #f8f9fa;">
      <h2 style="color: #dc3545; margin-top: 0;">TrustformeRS Error Report</h2>
      
      <div class="error-summary" style="margin-bottom: 20px; padding: 15px; background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px;">
        <h3 style="margin-top: 0; color: #856404;">Summary</h3>
        <table style="border-collapse: collapse; width: 100%;">
          <tr><td style="padding: 4px 8px; border: 1px solid #ddd; font-weight: bold;">Type:</td><td style="padding: 4px 8px; border: 1px solid #ddd;">${report.summary.type}</td></tr>
          <tr><td style="padding: 4px 8px; border: 1px solid #ddd; font-weight: bold;">Severity:</td><td style="padding: 4px 8px; border: 1px solid #ddd;"><span style="color: ${this.getSeverityColor(report.summary.severity)};">${report.summary.severity}</span></td></tr>
          <tr><td style="padding: 4px 8px; border: 1px solid #ddd; font-weight: bold;">Message:</td><td style="padding: 4px 8px; border: 1px solid #ddd;">${report.summary.message}</td></tr>
          <tr><td style="padding: 4px 8px; border: 1px solid #ddd; font-weight: bold;">Timestamp:</td><td style="padding: 4px 8px; border: 1px solid #ddd;">${report.summary.timestamp}</td></tr>
        </table>
      </div>
    `;

    if (report.solutions) {
      html += `
      <div class="solutions" style="margin-bottom: 20px;">
        <h3 style="color: #28a745;">Immediate Solutions</h3>
        <div style="display: grid; gap: 10px;">
      `;
      
      report.solutions.immediate.forEach((solution, index) => {
        html += `
        <div style="padding: 10px; border: 1px solid #28a745; border-radius: 4px; background: #d4edda;">
          <strong>${index + 1}. ${solution.title}</strong><br>
          <span style="color: #155724;">${solution.description}</span>
          ${solution.code ? `<br><code style="background: #f8f9fa; padding: 2px 4px; border-radius: 2px;">${solution.code}</code>` : ''}
        </div>
        `;
      });
      
      html += `</div></div>`;
    }

    if (report.recommendations.length > 0) {
      html += `
      <div class="recommendations" style="margin-bottom: 20px;">
        <h3 style="color: #007bff;">Recommendations</h3>
        <ul style="padding-left: 20px;">
      `;
      
      report.recommendations.forEach(rec => {
        html += `<li style="margin-bottom: 8px; color: #495057;">${rec}</li>`;
      });
      
      html += `</ul></div>`;
    }

    if (report.details.stackTrace) {
      html += `
      <div class="stack-trace" style="margin-bottom: 20px;">
        <h3 style="color: #6c757d;">Stack Trace Analysis</h3>
        <div style="background: #f8f9fa; padding: 10px; border-radius: 4px; font-family: monospace; font-size: 12px;">
          <strong>Likely Source:</strong> ${report.details.stackTrace.likelySource || 'Unknown'}<br>
          <strong>Total Frames:</strong> ${report.details.stackTrace.totalLines}<br>
          <strong>User Code Frames:</strong> ${report.details.stackTrace.userCode.length}<br>
          <strong>System Code Frames:</strong> ${report.details.stackTrace.systemCode.length}
        </div>
      </div>
      `;
    }

    html += `</div>`;
    return html;
  }

  /**
   * Register custom error handler
   * @param {string} type - Error type
   * @param {Function} handler - Error handler function
   */
  registerErrorHandler(type, handler) {
    this.errorHandlers.set(type, handler);
  }

  /**
   * Register custom solution
   * @param {string} type - Error type
   * @param {Object} solution - Solution object
   */
  registerSolution(type, solution) {
    if (!this.solutions.has(type)) {
      this.solutions.set(type, {
        immediate: [],
        preventive: [],
        diagnostic: [],
        references: []
      });
    }
    
    const typeSolutions = this.solutions.get(type);
    if (solution.immediate) typeSolutions.immediate.push(solution.immediate);
    if (solution.preventive) typeSolutions.preventive.push(solution.preventive);
    if (solution.diagnostic) typeSolutions.diagnostic.push(solution.diagnostic);
    if (solution.references) typeSolutions.references.push(solution.references);
  }

  /**
   * Clear diagnostic cache
   */
  clearCache() {
    this.diagnosticCache.clear();
  }

  /**
   * Get error statistics
   * @returns {Object} Error statistics
   */
  getStatistics() {
    const stats = {
      totalErrors: this.errorHistory.length,
      byType: {},
      bySeverity: {},
      byTime: {},
      mostFrequent: null,
      recentTrends: []
    };

    this.errorHistory.forEach(diagnostic => {
      // By type
      if (!stats.byType[diagnostic.classification]) {
        stats.byType[diagnostic.classification] = 0;
      }
      stats.byType[diagnostic.classification]++;

      // By severity
      if (!stats.bySeverity[diagnostic.severity]) {
        stats.bySeverity[diagnostic.severity] = 0;
      }
      stats.bySeverity[diagnostic.severity]++;

      // By time (last 24 hours)
      const hour = new Date(diagnostic.timestamp).getHours();
      if (!stats.byTime[hour]) {
        stats.byTime[hour] = 0;
      }
      stats.byTime[hour]++;
    });

    // Find most frequent
    let maxCount = 0;
    Object.entries(stats.byType).forEach(([type, count]) => {
      if (count > maxCount) {
        maxCount = count;
        stats.mostFrequent = { type, count };
      }
    });

    return stats;
  }

  // Private helper methods
  initializeErrorHandlers() {
    this.errorHandlers.set(ErrorTypes.WASM, this.handleWASMError.bind(this));
    this.errorHandlers.set(ErrorTypes.WEBGL, this.handleWebGLError.bind(this));
    this.errorHandlers.set(ErrorTypes.MEMORY, this.handleMemoryError.bind(this));
    this.errorHandlers.set(ErrorTypes.INITIALIZATION, this.handleInitializationError.bind(this));
    this.errorHandlers.set(ErrorTypes.TENSOR_OPERATION, this.handleTensorError.bind(this));
  }

  initializeSolutions() {
    // WASM error solutions
    this.solutions.set(ErrorTypes.WASM, {
      immediate: [
        {
          title: 'Check WASM module loading',
          description: 'Verify that the WASM module is properly loaded and initialized',
          code: 'await initialize({ wasmPath: "./path/to/wasm/file.wasm" })'
        },
        {
          title: 'Check browser compatibility',
          description: 'Ensure your browser supports WebAssembly',
          code: 'if (!window.WebAssembly) { console.error("WebAssembly not supported"); }'
        }
      ],
      preventive: [
        {
          title: 'Use feature detection',
          description: 'Always check for WebAssembly support before initialization'
        },
        {
          title: 'Implement fallbacks',
          description: 'Provide JavaScript fallbacks for unsupported environments'
        }
      ],
      diagnostic: [
        {
          title: 'Check WASM compilation',
          description: 'Verify WASM module compilation and instantiation'
        }
      ],
      references: [
        {
          title: 'WebAssembly Documentation',
          url: 'https://developer.mozilla.org/en-US/docs/WebAssembly'
        }
      ]
    });

    // WebGL error solutions
    this.solutions.set(ErrorTypes.WEBGL, {
      immediate: [
        {
          title: 'Check WebGL context',
          description: 'Verify that WebGL context is available and not lost',
          code: 'const gl = canvas.getContext("webgl"); if (!gl) { /* Handle error */ }'
        },
        {
          title: 'Check WebGL extensions',
          description: 'Verify required WebGL extensions are available'
        }
      ],
      preventive: [
        {
          title: 'Implement context loss handling',
          description: 'Handle WebGL context loss and restoration'
        }
      ]
    });

    // Memory error solutions
    this.solutions.set(ErrorTypes.MEMORY, {
      immediate: [
        {
          title: 'Free unused tensors',
          description: 'Dispose of tensors that are no longer needed',
          code: 'tensor.dispose(); // or tensor.free()'
        },
        {
          title: 'Reduce batch size',
          description: 'Use smaller batch sizes to reduce memory usage'
        }
      ],
      preventive: [
        {
          title: 'Use memory pooling',
          description: 'Implement memory pooling to reuse tensor allocations'
        }
      ]
    });

    // Initialization error solutions
    this.solutions.set(ErrorTypes.INITIALIZATION, {
      immediate: [
        {
          title: 'Call initialize() first',
          description: 'Ensure the TrustformeRS library is initialized before use',
          code: 'await initialize(); // Call before any other operations'
        },
        {
          title: 'Check initialization options',
          description: 'Verify initialization options are correct'
        }
      ]
    });

    // Tensor operation error solutions
    this.solutions.set(ErrorTypes.TENSOR_OPERATION, {
      immediate: [
        {
          title: 'Check tensor shapes',
          description: 'Verify tensor shapes are compatible for the operation',
          code: 'console.warn(tensor.shape()); // Check tensor shape'
        },
        {
          title: 'Validate tensor data',
          description: 'Check for NaN or infinite values in tensor data'
        }
      ]
    });
  }

  handleWASMError(error, context) {
    return {
      type: ErrorTypes.WASM,
      severity: ErrorSeverity.HIGH,
      customSolutions: []
    };
  }

  handleWebGLError(error, context) {
    return {
      type: ErrorTypes.WEBGL,
      severity: ErrorSeverity.MEDIUM,
      customSolutions: []
    };
  }

  handleMemoryError(error, context) {
    return {
      type: ErrorTypes.MEMORY,
      severity: ErrorSeverity.HIGH,
      customSolutions: []
    };
  }

  handleInitializationError(error, context) {
    return {
      type: ErrorTypes.INITIALIZATION,
      severity: ErrorSeverity.CRITICAL,
      customSolutions: []
    };
  }

  handleTensorError(error, context) {
    return {
      type: ErrorTypes.TENSOR_OPERATION,
      severity: ErrorSeverity.MEDIUM,
      customSolutions: []
    };
  }

  generateCacheKey(errorInfo) {
    return `${errorInfo.name}_${errorInfo.message}_${errorInfo.operation || 'unknown'}`;
  }

  classifyOperation(operation) {
    if (operation.includes('tensor')) return 'tensor_operation';
    if (operation.includes('model')) return 'model_operation';
    if (operation.includes('inference')) return 'inference_operation';
    return 'unknown_operation';
  }

  analyzeInputs(inputs) {
    const analysis = {
      count: Array.isArray(inputs) ? inputs.length : 1,
      types: [],
      hasNaN: false,
      hasInfinite: false,
      shapes: []
    };

    const inputArray = Array.isArray(inputs) ? inputs : [inputs];
    
    inputArray.forEach(input => {
      analysis.types.push(typeof input);
      
      if (input && typeof input === 'object' && input.shape) {
        analysis.shapes.push(input.shape);
        
        // Check for tensor quality issues
        try {
          const inspection = tensorInspector.analyze(input, {
            includeNaN: true,
            includeInfinite: true
          });
          if (inspection.quality.hasNaN) analysis.hasNaN = true;
          if (inspection.quality.hasInfinite) analysis.hasInfinite = true;
        } catch (e) {
          // Ignore inspection errors
        }
      }
    });

    return analysis;
  }

  getEnvironmentInfo() {
    return {
      type: typeof window !== 'undefined' ? 'browser' : 'nodejs',
      userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'Node.js',
      platform: typeof process !== 'undefined' ? process.platform : 'browser',
      webgl: typeof WebGLRenderingContext !== 'undefined',
      webgpu: typeof window !== 'undefined' && 'gpu' in navigator,
      wasm: typeof WebAssembly !== 'undefined'
    };
  }

  getMemoryState() {
    const state = {
      usage: 0,
      limit: 0,
      percentage: 0
    };

    if (typeof performance !== 'undefined' && performance.memory) {
      state.usage = Math.round(performance.memory.usedJSHeapSize / 1024 / 1024);
      state.limit = Math.round(performance.memory.jsHeapSizeLimit / 1024 / 1024);
      state.percentage = (state.usage / state.limit) * 100;
    }

    return state;
  }

  getSystemState() {
    return {
      timestamp: Date.now(),
      debugEnabled: debugUtils.isEnabled(),
      activeSessionCount: debugUtils.debugData?.sessions?.size || 0,
      errorCount: this.errorHistory.length
    };
  }

  parseStackFrame(line, index) {
    const frame = {
      index,
      raw: line,
      function: null,
      file: null,
      line: null,
      column: null,
      library: null,
      isUserCode: false
    };

    // Parse Chrome/V8 format
    const chromeMatch = line.match(/^\s*at\s+(.+?)\s+\((.+?):(\d+):(\d+)\)$/);
    if (chromeMatch) {
      frame.function = chromeMatch[1];
      frame.file = chromeMatch[2];
      frame.line = parseInt(chromeMatch[3], 10);
      frame.column = parseInt(chromeMatch[4], 10);
    }

    // Parse Firefox format
    const firefoxMatch = line.match(/^(.+?)@(.+?):(\d+):(\d+)$/);
    if (firefoxMatch) {
      frame.function = firefoxMatch[1];
      frame.file = firefoxMatch[2];
      frame.line = parseInt(firefoxMatch[3], 10);
      frame.column = parseInt(firefoxMatch[4], 10);
    }

    // Determine library and user code
    if (frame.file) {
      if (frame.file.includes('trustformers')) {
        frame.library = 'trustformers';
        frame.isUserCode = false;
      } else if (frame.file.includes('node_modules') || frame.file.includes('webpack')) {
        frame.library = 'system';
        frame.isUserCode = false;
      } else {
        frame.isUserCode = true;
      }
    }

    return frame;
  }

  identifyErrorSource(frames) {
    const userFrames = frames.filter(f => f.isUserCode);
    if (userFrames.length > 0) {
      return `User code: ${userFrames[0].function || 'unknown'} (${userFrames[0].file}:${userFrames[0].line})`;
    }

    const trustformersFrames = frames.filter(f => f.library === 'trustformers');
    if (trustformersFrames.length > 0) {
      return `TrustformeRS: ${trustformersFrames[0].function || 'unknown'}`;
    }

    return 'Unknown';
  }

  getErrorFrequency(error) {
    const {message} = error;
    return this.errorHistory.filter(d => d.error.message === message).length;
  }

  getRecentOccurrences(error) {
    const {message} = error;
    const recent = this.errorHistory
      .filter(d => d.error.message === message)
      .filter(d => Date.now() - new Date(d.timestamp).getTime() < 24 * 60 * 60 * 1000)
      .length;
    return recent;
  }

  getTimePattern(error) {
    const {message} = error;
    const occurrences = this.errorHistory
      .filter(d => d.error.message === message)
      .map(d => new Date(d.timestamp).getHours());
    
    const pattern = {};
    occurrences.forEach(hour => {
      pattern[hour] = (pattern[hour] || 0) + 1;
    });
    
    return pattern;
  }

  getCorrelations(error) {
    // Find errors that tend to occur together
    const correlations = [];
    const errorTime = Date.now();
    
    this.errorHistory.forEach(diagnostic => {
      const timeDiff = Math.abs(errorTime - new Date(diagnostic.timestamp).getTime());
      if (timeDiff < 60000 && diagnostic.error.message !== error.message) { // Within 1 minute
        correlations.push({
          error: diagnostic.error.message,
          type: diagnostic.classification,
          timeDiff
        });
      }
    });
    
    return correlations.slice(0, 3);
  }

  getTrend(error) {
    const {message} = error;
    const recent = this.errorHistory
      .filter(d => d.error.message === message)
      .filter(d => Date.now() - new Date(d.timestamp).getTime() < 24 * 60 * 60 * 1000)
      .length;
    
    const older = this.errorHistory
      .filter(d => d.error.message === message)
      .filter(d => {
        const time = new Date(d.timestamp).getTime();
        const now = Date.now();
        return now - time >= 24 * 60 * 60 * 1000 && now - time < 48 * 60 * 60 * 1000;
      })
      .length;
    
    if (recent > older) return 'increasing';
    if (recent < older) return 'decreasing';
    return 'stable';
  }

  getPatternSolutions(error, context) {
    const solutions = { immediate: [], preventive: [] };
    
    // Check for common patterns
    const message = error.message.toLowerCase();
    
    if (message.includes('nan')) {
      solutions.immediate.push({
        title: 'Check for NaN values',
        description: 'Inspect your input data for NaN values',
        code: 'if (tensor.hasNaN()) { /* Handle NaN values */ }'
      });
    }
    
    if (message.includes('shape')) {
      solutions.immediate.push({
        title: 'Verify tensor shapes',
        description: 'Check that tensor shapes are compatible',
        code: 'console.warn("Shape A:", a.shape(), "Shape B:", b.shape());'
      });
    }
    
    return solutions;
  }

  getContextSolutions(error, context) {
    const solutions = { immediate: [], preventive: [] };
    
    if (context.operation && context.operation.includes('matmul')) {
      solutions.immediate.push({
        title: 'Check matrix multiplication compatibility',
        description: 'Ensure the inner dimensions match for matrix multiplication',
        code: 'if (a.shape()[1] !== b.shape()[0]) { /* Shapes incompatible */ }'
      });
    }
    
    return solutions;
  }

  generateAutoFixes(error, context) {
    const fixes = [];
    
    // Auto-fix for common shape issues
    if (error.message.includes('shape') && context.tensors) {
      fixes.push({
        type: 'shape_fix',
        description: 'Automatically reshape tensors to compatible shapes',
        canApply: true,
        apply: () => 
          // This would contain actual fix logic
           'Shape fix applied'
        
      });
    }
    
    return fixes;
  }

  calculateSimilarity(str1, str2) {
    const len1 = str1.length;
    const len2 = str2.length;
    const matrix = [];
    
    for (let i = 0; i <= len2; i++) {
      matrix[i] = [i];
    }
    
    for (let j = 0; j <= len1; j++) {
      matrix[0][j] = j;
    }
    
    for (let i = 1; i <= len2; i++) {
      for (let j = 1; j <= len1; j++) {
        if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
          matrix[i][j] = matrix[i - 1][j - 1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i - 1][j - 1] + 1,
            matrix[i][j - 1] + 1,
            matrix[i - 1][j] + 1
          );
        }
      }
    }
    
    return 1 - matrix[len2][len1] / Math.max(len1, len2);
  }

  generateRecommendations(diagnostic) {
    const recommendations = [];
    
    if (diagnostic.severity === ErrorSeverity.CRITICAL) {
      recommendations.push('Address this error immediately as it may cause system instability');
    }
    
    if (diagnostic.classification === ErrorTypes.MEMORY) {
      recommendations.push('Consider implementing memory pooling or reducing batch sizes');
    }
    
    if (diagnostic.patterns && diagnostic.patterns.frequency > 5) {
      recommendations.push('This error occurs frequently - consider implementing a permanent fix');
    }
    
    return recommendations;
  }

  generateNextSteps(diagnostic) {
    const steps = [];
    
    steps.push('Review the immediate solutions provided');
    
    if (diagnostic.solutions && diagnostic.solutions.diagnostic.length > 0) {
      steps.push('Run diagnostic tests to gather more information');
    }
    
    if (diagnostic.relatedErrors.length > 0) {
      steps.push('Check for patterns in related errors');
    }
    
    steps.push('Implement preventive measures to avoid future occurrences');
    
    return steps;
  }

  getSeverityColor(severity) {
    const colors = {
      [ErrorSeverity.CRITICAL]: '#dc3545',
      [ErrorSeverity.HIGH]: '#fd7e14',
      [ErrorSeverity.MEDIUM]: '#ffc107',
      [ErrorSeverity.LOW]: '#20c997',
      [ErrorSeverity.INFO]: '#6c757d'
    };
    return colors[severity] || '#6c757d';
  }

  getMemoryInfo() {
    if (typeof performance !== 'undefined' && performance.memory) {
      return {
        used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024),
        total: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024),
        limit: Math.round(performance.memory.jsHeapSizeLimit / 1024 / 1024)
      };
    }
    return null;
  }

  getPerformanceInfo() {
    return {
      now: performance.now(),
      timing: typeof performance.timing !== 'undefined' ? {
        navigationStart: performance.timing.navigationStart,
        loadEventEnd: performance.timing.loadEventEnd
      } : null
    };
  }

  trimHistory() {
    const maxHistory = 1000;
    if (this.errorHistory.length > maxHistory) {
      this.errorHistory = this.errorHistory.slice(-maxHistory);
    }
  }

  updatePatterns(diagnostic) {
    const key = `${diagnostic.classification}_${diagnostic.error.name}`;
    const pattern = this.errorPatterns.get(key) || {
      count: 0,
      firstSeen: diagnostic.timestamp,
      lastSeen: diagnostic.timestamp,
      messages: new Set()
    };
    
    pattern.count++;
    pattern.lastSeen = diagnostic.timestamp;
    pattern.messages.add(diagnostic.error.message);
    
    this.errorPatterns.set(key, pattern);
  }
}

// Global error diagnostics instance
export const errorDiagnostics = new ErrorDiagnostics();

// Convenience functions
export const diagnose = {
  error: (error, context, options) => errorDiagnostics.diagnose(error, context, options),
  generate: (error, context) => errorDiagnostics.generateReport(error, context),
  text: (error, context) => errorDiagnostics.generateTextReport(error, context),
  html: (error, context) => errorDiagnostics.generateHTMLReport(error, context),
  statistics: () => errorDiagnostics.getStatistics(),
  clearCache: () => errorDiagnostics.clearCache()
};

// Error handler decorator
export function handleErrors(options = {}) {
  return function(target, propertyKey, descriptor) {
    const originalMethod = descriptor.value;
    
    descriptor.value = async function(...args) {
      try {
        return await originalMethod.apply(this, args);
      } catch (error) {
        const context = {
          operation: `${target.constructor.name}.${propertyKey}`,
          inputs: options.logArgs ? args : undefined,
          ...options.context
        };
        
        const diagnostic = errorDiagnostics.diagnose(error, context);
        
        if (options.autoReport) {
          console.error(errorDiagnostics.generateTextReport(error, context));
        }
        
        if (options.rethrow !== false) {
          throw error;
        }
        
        return diagnostic;
      }
    };
    
    return descriptor;
  };
}

// Export for integration with other modules
export default ErrorDiagnostics;