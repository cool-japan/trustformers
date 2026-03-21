/**
 * Development Tools for Debugging and Analysis
 * Comprehensive development utilities for TrustformeRS debugging, analysis, and visualization
 */

/**
 * Development tools for debugging and analysis
 */
export const devTools = {
  // Debug utilities
  debug: {
    enable: (options) => import('./debug-utilities.js').then(({ debugUtils }) => debugUtils.enable(options)).catch(() => {
        console.warn('Debug utilities not available');
        return false;
      }),

    disable: () => import('./debug-utilities.js').then(({ debugUtils }) => debugUtils.disable()).catch(() => {
        console.warn('Debug utilities not available');
        return false;
      }),

    configure: (config) => import('./debug-utilities.js').then(({ debugUtils }) => debugUtils.configure(config)).catch(() => {
        console.warn('Debug utilities not available');
        return false;
      }),

    isEnabled: () => import('./debug-utilities.js').then(({ debugUtils }) => debugUtils.isEnabled()).catch(() => false),

    startSession: (name, metadata) => import('./debug-utilities.js').then(({ debugUtils }) => debugUtils.startSession(name, metadata)).catch(() => {
        console.warn('Debug utilities not available');
        return null;
      }),

    endSession: () => import('./debug-utilities.js').then(({ debugUtils }) => debugUtils.endSession()).catch(() => {
        console.warn('Debug utilities not available');
        return null;
      }),

    trackTensor: (tensor, operation, metadata) => import('./debug-utilities.js').then(({ debugUtils }) => debugUtils.trackTensor(tensor, operation, metadata)).catch(() => {
        console.warn('Debug utilities not available');
        return null;
      }),

    trackOperation: (name, fn, metadata) => import('./debug-utilities.js').then(({ debugUtils }) => debugUtils.trackOperation(name, fn, metadata)).catch(() => {
        console.warn('Debug utilities not available');
        return fn();
      }),

    validateOperation: (operation, tensors, options) => import('./debug-utilities.js').then(({ debugUtils }) => debugUtils.validateOperation(operation, tensors, options)).catch(() => {
        console.warn('Debug utilities not available');
        return true;
      }),

    getMemoryUsage: () => import('./debug-utilities.js').then(({ debugUtils }) => debugUtils.getMemoryUsage()).catch(() => this._getFallbackMemoryUsage()),

    getPerformanceMetrics: () => import('./debug-utilities.js').then(({ debugUtils }) => debugUtils.getPerformanceMetrics()).catch(() => this._getFallbackPerformanceMetrics()),

    generateReport: () => import('./debug-utilities.js').then(({ debugUtils }) => debugUtils.generateReport()).catch(() => this._generateFallbackReport()),

    exportData: (format) => import('./debug-utilities.js').then(({ debugUtils }) => debugUtils.exportData(format)).catch(() => {
        console.warn('Debug utilities not available');
        return null;
      }),

    clear: () => import('./debug-utilities.js').then(({ debugUtils }) => debugUtils.clear()).catch(() => {
        console.warn('Debug utilities not available');
        return false;
      }),

    // Fallback methods
    _getFallbackMemoryUsage() {
      if (typeof performance !== 'undefined' && performance.memory) {
        return {
          used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024),
          total: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024),
          limit: Math.round(performance.memory.jsHeapSizeLimit / 1024 / 1024)
        };
      }
      return null;
    },

    _getFallbackPerformanceMetrics() {
      return {
        timestamp: new Date().toISOString(),
        now: typeof performance !== 'undefined' ? performance.now() : Date.now(),
        memory: this._getFallbackMemoryUsage()
      };
    },

    _generateFallbackReport() {
      return {
        timestamp: new Date().toISOString(),
        type: 'fallback_report',
        memory: this._getFallbackMemoryUsage(),
        performance: this._getFallbackPerformanceMetrics(),
        message: 'Debug utilities not available - using fallback report'
      };
    }
  },

  // Tensor inspection
  tensor: {
    analyze: (tensor, options) => import('./tensor-inspector.js').then(({ tensorInspector }) => tensorInspector.analyze(tensor, options)).catch(() => this._fallbackTensorAnalysis(tensor)),

    compare: (tensor1, tensor2, options) => import('./tensor-inspector.js').then(({ tensorInspector }) => tensorInspector.compare(tensor1, tensor2, options)).catch(() => this._fallbackTensorComparison(tensor1, tensor2)),

    visualize: (tensor, options) => import('./tensor-inspector.js').then(({ tensorInspector }) => tensorInspector.visualizeText(tensor, options)).catch(() => this._fallbackTensorVisualization(tensor)),

    visualizeHTML: (tensor, options) => import('./tensor-inspector.js').then(({ tensorInspector }) => tensorInspector.visualizeHTML(tensor, options)).catch(() => this._fallbackTensorHTMLVisualization(tensor)),

    summarize: (tensor) => import('./tensor-inspector.js').then(({ tensorInspector }) => tensorInspector.summarize(tensor)).catch(() => this._fallbackTensorSummary(tensor)),

    clearCaches: () => import('./tensor-inspector.js').then(({ tensorInspector }) => tensorInspector.clearCaches()).catch(() => {
        console.warn('Tensor inspector not available');
        return false;
      }),

    // Fallback tensor analysis methods
    _fallbackTensorAnalysis(tensor) {
      if (!tensor || !tensor.data) {
        return { error: 'Invalid tensor for analysis' };
      }

      const {data} = tensor;
      const shape = tensor.shape || [data.length];

      return {
        shape: { shape: Array.from(shape), size: data.length },
        dtype: { dtype: tensor.dtype || 'f32', bytes: data.byteLength },
        statistics: this._calculateBasicStatistics(data),
        memory: { totalBytes: data.byteLength },
        quality: {
          hasNaN: this._hasNaN(data),
          hasInfinite: this._hasInfinite(data)
        }
      };
    },

    _fallbackTensorComparison(tensor1, tensor2) {
      const analysis1 = this._fallbackTensorAnalysis(tensor1);
      const analysis2 = this._fallbackTensorAnalysis(tensor2);

      return {
        tensor1: analysis1,
        tensor2: analysis2,
        comparison: {
          shapesMatch: JSON.stringify(analysis1.shape.shape) === JSON.stringify(analysis2.shape.shape),
          dtypesMatch: analysis1.dtype.dtype === analysis2.dtype.dtype,
          sizesMatch: analysis1.shape.size === analysis2.shape.size
        }
      };
    },

    _fallbackTensorVisualization(tensor) {
      const analysis = this._fallbackTensorAnalysis(tensor);
      return `
Tensor Analysis:
  Shape: ${JSON.stringify(analysis.shape.shape)}
  Size: ${analysis.shape.size}
  Type: ${analysis.dtype.dtype}
  Memory: ${(analysis.memory.totalBytes / 1024).toFixed(2)} KB
  Statistics: ${JSON.stringify(analysis.statistics, null, 2)}
      `.trim();
    },

    _fallbackTensorHTMLVisualization(tensor) {
      const analysis = this._fallbackTensorAnalysis(tensor);
      return `
        <div style="font-family: monospace; border: 1px solid #ccc; padding: 10px; margin: 5px;">
          <h4>Tensor Analysis</h4>
          <p><strong>Shape:</strong> ${JSON.stringify(analysis.shape.shape)}</p>
          <p><strong>Size:</strong> ${analysis.shape.size.toLocaleString()}</p>
          <p><strong>Type:</strong> ${analysis.dtype.dtype}</p>
          <p><strong>Memory:</strong> ${(analysis.memory.totalBytes / 1024).toFixed(2)} KB</p>
          <div style="margin-top: 10px;">
            <strong>Statistics:</strong>
            <pre style="background: #f5f5f5; padding: 5px; margin: 5px 0;">${JSON.stringify(analysis.statistics, null, 2)}</pre>
          </div>
        </div>
      `;
    },

    _fallbackTensorSummary(tensor) {
      const analysis = this._fallbackTensorAnalysis(tensor);
      return {
        shape: analysis.shape.shape,
        size: analysis.shape.size,
        dtype: analysis.dtype.dtype,
        memory: `${(analysis.memory.totalBytes / 1024).toFixed(2)} KB`
      };
    },

    _calculateBasicStatistics(data) {
      if (!data || data.length === 0) return null;

      let sum = 0;
      const [firstValue] = data;
      let min = firstValue;
      let max = firstValue;

      for (let i = 0; i < data.length; i++) {
        const value = data[i];
        sum += value;
        if (value < min) min = value;
        if (value > max) max = value;
      }

      const mean = sum / data.length;

      // Calculate variance
      let variance = 0;
      for (let i = 0; i < data.length; i++) {
        const diff = data[i] - mean;
        variance += diff * diff;
      }
      variance /= data.length;

      return {
        count: data.length,
        sum,
        mean,
        min,
        max,
        variance,
        std: Math.sqrt(variance)
      };
    },

    _hasNaN(data) {
      for (let i = 0; i < data.length; i++) {
        if (Number.isNaN(data[i])) return true;
      }
      return false;
    },

    _hasInfinite(data) {
      for (let i = 0; i < data.length; i++) {
        if (!Number.isFinite(data[i])) return true;
      }
      return false;
    }
  },

  // Model visualization
  model: {
    analyze: (model, options) => import('./model-visualization.js').then(({ modelVisualizer }) => modelVisualizer.analyzeModel(model, options)).catch(() => this._fallbackModelAnalysis(model)),

    visualize: (model, options) => import('./model-visualization.js').then(({ modelVisualizer }) => modelVisualizer.visualizeArchitecture(model, options)).catch(() => this._fallbackModelVisualization(model)),

    visualizeHTML: (model, options) => import('./model-visualization.js').then(({ modelVisualizer }) => modelVisualizer.visualizeHTML(model, options)).catch(() => this._fallbackModelHTMLVisualization(model)),

    summarize: (model) => import('./model-visualization.js').then(({ modelVisualizer }) => modelVisualizer.summarize(model)).catch(() => this._fallbackModelSummary(model)),

    clearCaches: () => import('./model-visualization.js').then(({ modelVisualizer }) => modelVisualizer.clearCaches()).catch(() => {
        console.warn('Model visualizer not available');
        return false;
      }),

    // Fallback model analysis methods
    _fallbackModelAnalysis(model) {
      return {
        type: typeof model,
        hasForward: typeof model.forward === 'function',
        hasConfig: !!model.config,
        timestamp: new Date().toISOString()
      };
    },

    _fallbackModelVisualization(model) {
      const analysis = this._fallbackModelAnalysis(model);
      return `
Model Analysis:
  Type: ${analysis.type}
  Has Forward: ${analysis.hasForward}
  Has Config: ${analysis.hasConfig}
  Analyzed: ${analysis.timestamp}
      `.trim();
    },

    _fallbackModelHTMLVisualization(model) {
      const analysis = this._fallbackModelAnalysis(model);
      return `
        <div style="font-family: monospace; border: 1px solid #ccc; padding: 10px; margin: 5px;">
          <h4>Model Analysis</h4>
          <p><strong>Type:</strong> ${analysis.type}</p>
          <p><strong>Has Forward:</strong> ${analysis.hasForward}</p>
          <p><strong>Has Config:</strong> ${analysis.hasConfig}</p>
          <p><strong>Analyzed:</strong> ${analysis.timestamp}</p>
        </div>
      `;
    },

    _fallbackModelSummary(model) {
      const analysis = this._fallbackModelAnalysis(model);
      return {
        type: analysis.type,
        capabilities: {
          forward: analysis.hasForward,
          config: analysis.hasConfig
        }
      };
    }
  },

  // Error diagnostics
  error: {
    diagnose: (error, context, options) => import('./error-diagnostics.js').then(({ errorDiagnostics }) => errorDiagnostics.diagnose(error, context, options)).catch(() => this._fallbackErrorDiagnosis(error, context)),

    generateReport: (error, context) => import('./error-diagnostics.js').then(({ errorDiagnostics }) => errorDiagnostics.generateReport(error, context)).catch(() => this._fallbackErrorReport(error, context)),

    generateTextReport: (error, context) => import('./error-diagnostics.js').then(({ errorDiagnostics }) => errorDiagnostics.generateTextReport(error, context)).catch(() => this._fallbackTextErrorReport(error, context)),

    generateHTMLReport: (error, context) => import('./error-diagnostics.js').then(({ errorDiagnostics }) => errorDiagnostics.generateHTMLReport(error, context)).catch(() => this._fallbackHTMLErrorReport(error, context)),

    getStatistics: () => import('./error-diagnostics.js').then(({ errorDiagnostics }) => errorDiagnostics.getStatistics()).catch(() => ({ errors: 0, warnings: 0 })),

    clearCache: () => import('./error-diagnostics.js').then(({ errorDiagnostics }) => errorDiagnostics.clearCache()).catch(() => {
        console.warn('Error diagnostics not available');
        return false;
      }),

    // Fallback error analysis methods
    _fallbackErrorDiagnosis(error, context) {
      return {
        error: {
          name: error.name,
          message: error.message,
          stack: error.stack
        },
        context: context || {},
        timestamp: new Date().toISOString(),
        type: 'fallback_diagnosis'
      };
    },

    _fallbackErrorReport(error, context) {
      const diagnosis = this._fallbackErrorDiagnosis(error, context);
      return {
        ...diagnosis,
        suggestions: [
          'Check input parameters',
          'Verify tensor shapes and types',
          'Ensure model is properly initialized'
        ]
      };
    },

    _fallbackTextErrorReport(error, context) {
      const report = this._fallbackErrorReport(error, context);
      return `
Error Report:
  Name: ${report.error.name}
  Message: ${report.error.message}
  Timestamp: ${report.timestamp}

Suggestions:
${report.suggestions.map(s => `  - ${s}`).join('\n')}
      `.trim();
    },

    _fallbackHTMLErrorReport(error, context) {
      const report = this._fallbackErrorReport(error, context);
      return `
        <div style="font-family: monospace; border: 2px solid #d32f2f; padding: 15px; margin: 5px; background: #ffebee;">
          <h4 style="color: #d32f2f; margin-top: 0;">Error Report</h4>
          <p><strong>Name:</strong> ${report.error.name}</p>
          <p><strong>Message:</strong> ${report.error.message}</p>
          <p><strong>Timestamp:</strong> ${report.timestamp}</p>
          <div style="margin-top: 10px;">
            <strong>Suggestions:</strong>
            <ul style="margin: 5px 0;">
              ${report.suggestions.map(s => `<li>${s}</li>`).join('')}
            </ul>
          </div>
        </div>
      `;
    }
  },

  // Combined analysis
  analyze: {
    /**
     * Comprehensive analysis of tensors, model, and context
     * @param {Object} options - Analysis options
     * @returns {Promise<Object>} Comprehensive analysis
     */
    async comprehensive(options = {}) {
      const { tensors = [], model = null, includeDebug = true } = options;

      const analysis = {
        timestamp: new Date().toISOString(),
        tensors: [],
        model: null,
        debug: null,
        performance: null,
        memory: null
      };

      // Analyze tensors
      if (tensors.length > 0) {
        analysis.tensors = await Promise.all(
          tensors.map(async tensor => {
            try {
              return await devTools.tensor.analyze(tensor, {
                includeStatistics: true,
                includeDistribution: true,
                includeNaN: true,
                includeInfinite: true
              });
            } catch (error) {
              return { error: error.message };
            }
          })
        );
      }

      // Analyze model
      if (model) {
        try {
          analysis.model = await devTools.model.analyze(model, {
            includeWeights: true,
            includeActivations: false,
            includeGradients: false
          });
        } catch (error) {
          analysis.model = { error: error.message };
        }
      }

      // Include debug information
      if (includeDebug) {
        try {
          const isEnabled = await devTools.debug.isEnabled();
          if (isEnabled) {
            analysis.debug = await devTools.debug.generateReport();
          }
        } catch (error) {
          analysis.debug = { error: error.message };
        }
      }

      // Include performance information
      if (typeof performance !== 'undefined') {
        analysis.performance = {
          now: performance.now(),
          memory: performance.memory ? {
            used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024),
            total: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024),
            limit: Math.round(performance.memory.jsHeapSizeLimit / 1024 / 1024)
          } : null
        };
      }

      return analysis;
    },

    /**
     * Generate comprehensive HTML report
     * @param {Object} options - Report options
     * @returns {Promise<string>} HTML report
     */
    async generateHTMLReport(options = {}) {
      const analysis = await this.comprehensive(options);

      let html = `
      <div class="comprehensive-analysis" style="font-family: 'Courier New', monospace; border: 1px solid #ccc; padding: 20px; margin: 10px; background: #f9f9f9;">
        <h2 style="margin-top: 0; color: #333;">TrustformeRS Comprehensive Analysis</h2>
        <p style="color: #666; margin-bottom: 20px;">Generated: ${analysis.timestamp}</p>
      `;

      if (analysis.model) {
        html += `
        <div class="model-section" style="margin-bottom: 20px;">
          <h3 style="color: #007bff;">Model Analysis</h3>
          ${await devTools.model.visualizeHTML(options.model, { includeMemory: true })}
        </div>
        `;
      }

      if (analysis.tensors.length > 0) {
        html += `
        <div class="tensor-section" style="margin-bottom: 20px;">
          <h3 style="color: #28a745;">Tensor Analysis</h3>
        `;

        for (let i = 0; i < analysis.tensors.length; i++) {
          const tensorAnalysis = analysis.tensors[i];
          if (!tensorAnalysis.error) {
            html += `
            <div style="margin-bottom: 15px; padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: #fff;">
              <h4>Tensor ${i + 1}</h4>
              <p><strong>Shape:</strong> ${JSON.stringify(tensorAnalysis.shape.shape)}</p>
              <p><strong>Data Type:</strong> ${tensorAnalysis.dtype.dtype}</p>
              <p><strong>Size:</strong> ${tensorAnalysis.shape.size.toLocaleString()} elements</p>
              <p><strong>Memory:</strong> ${(tensorAnalysis.memory.totalBytes / 1024).toFixed(2)} KB</p>
              ${tensorAnalysis.quality && tensorAnalysis.quality.hasNaN ? '<p style="color: #d32f2f;">⚠ Contains NaN values</p>' : ''}
              ${tensorAnalysis.quality && tensorAnalysis.quality.hasInfinite ? '<p style="color: #d32f2f;">⚠ Contains infinite values</p>' : ''}
            </div>
            `;
          }
        }

        html += '</div>';
      }

      if (analysis.debug) {
        html += `
        <div class="debug-section" style="margin-bottom: 20px;">
          <h3 style="color: #ffc107;">Debug Information</h3>
          <div style="background: #fff; padding: 15px; border-radius: 4px;">
            <pre style="margin: 0; font-size: 12px;">${JSON.stringify(analysis.debug, null, 2)}</pre>
          </div>
        </div>
        `;
      }

      if (analysis.performance) {
        html += `
        <div class="performance-section" style="margin-bottom: 20px;">
          <h3 style="color: #dc3545;">Performance Information</h3>
          <div style="background: #fff; padding: 15px; border-radius: 4px;">
            <p><strong>Current Time:</strong> ${analysis.performance.now.toFixed(2)}ms</p>
            ${analysis.performance.memory ? `
            <p><strong>Memory Usage:</strong> ${analysis.performance.memory.used}MB / ${analysis.performance.memory.limit}MB</p>
            <p><strong>Memory Utilization:</strong> ${((analysis.performance.memory.used / analysis.performance.memory.limit) * 100).toFixed(2)}%</p>
            ` : ''}
          </div>
        </div>
        `;
      }

      html += '</div>';
      return html;
    }
  }
};

export default devTools;