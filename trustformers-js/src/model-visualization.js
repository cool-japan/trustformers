/**
 * TrustformeRS Model Visualization Helpers
 * Comprehensive model analysis and visualization utilities for development
 */

import { tensorInspector } from './tensor-inspector.js';
import { debugUtils } from './debug-utilities.js';

/**
 * Model visualization and analysis class
 */
export class ModelVisualizer {
  constructor() {
    this.modelCache = new Map();
    this.layerCache = new Map();
    this.activationCache = new Map();
    this.visualizationCache = new Map();
  }

  /**
   * Analyze model architecture and structure
   * @param {Object} model - Model object
   * @param {Object} options - Analysis options
   * @returns {Object} Model analysis results
   */
  analyzeModel(model, options = {}) {
    const {
      includeWeights = true,
      includeActivations = false,
      includeGradients = false,
      cacheResults = true,
      maxDepth = 10
    } = options;

    const modelId = this.getModelId(model);
    const cacheKey = `${modelId}_${JSON.stringify(options)}`;
    
    if (cacheResults && this.modelCache.has(cacheKey)) {
      return this.modelCache.get(cacheKey);
    }

    const analysis = {
      basic: this.getBasicModelInfo(model),
      architecture: this.getArchitectureInfo(model, maxDepth),
      parameters: this.getParameterInfo(model),
      layers: this.getLayerInfo(model),
      computation: this.getComputationInfo(model),
      weights: includeWeights ? this.getWeightInfo(model) : null,
      activations: includeActivations ? this.getActivationInfo(model) : null,
      gradients: includeGradients ? this.getGradientInfo(model) : null,
      memory: this.getModelMemoryInfo(model),
      timestamp: performance.now()
    };

    if (cacheResults) {
      this.modelCache.set(cacheKey, analysis);
    }

    return analysis;
  }

  /**
   * Get basic model information
   * @param {Object} model - Model object
   * @returns {Object} Basic model info
   */
  getBasicModelInfo(model) {
    return {
      id: this.getModelId(model),
      name: model.name || model.constructor.name,
      type: this.getModelType(model),
      hasConfig: this.hasModelConfig(model),
      isTraining: this.isTrainingMode(model),
      isQuantized: this.isQuantized(model),
      device: this.getModelDevice(model),
      constructor: model.constructor.name,
      version: model.version || 'unknown'
    };
  }

  /**
   * Get model architecture information
   * @param {Object} model - Model object
   * @param {number} maxDepth - Maximum depth to analyze
   * @returns {Object} Architecture info
   */
  getArchitectureInfo(model, maxDepth = 10) {
    const architecture = {
      layers: [],
      connections: [],
      totalLayers: 0,
      totalParameters: 0,
      architectureType: this.getArchitectureType(model),
      hasAttention: false,
      hasEmbedding: false,
      hasNormalization: false,
      hasDropout: false
    };

    try {
      // Extract layer information
      const layers = this.extractLayers(model, maxDepth);
      architecture.layers = layers;
      architecture.totalLayers = layers.length;

      // Analyze layer types
      layers.forEach(layer => {
        if (layer.type.includes('attention')) {
          architecture.hasAttention = true;
        }
        if (layer.type.includes('embedding')) {
          architecture.hasEmbedding = true;
        }
        if (layer.type.includes('norm')) {
          architecture.hasNormalization = true;
        }
        if (layer.type.includes('dropout')) {
          architecture.hasDropout = true;
        }
        architecture.totalParameters += layer.parameters || 0;
      });

      // Extract connections
      architecture.connections = this.extractConnections(model);

    } catch (error) {
      architecture.error = error.message;
    }

    return architecture;
  }

  /**
   * Get parameter information
   * @param {Object} model - Model object
   * @returns {Object} Parameter info
   */
  getParameterInfo(model) {
    const info = {
      total: 0,
      trainable: 0,
      frozen: 0,
      byLayer: {},
      byType: {
        weights: 0,
        biases: 0,
        embeddings: 0,
        normalization: 0,
        other: 0
      },
      distribution: {
        min: Infinity,
        max: -Infinity,
        mean: 0,
        std: 0
      }
    };

    try {
      const parameters = this.extractParameters(model);
      
      parameters.forEach(param => {
        const paramCount = param.size || 0;
        info.total += paramCount;
        
        if (param.trainable) {
          info.trainable += paramCount;
        } else {
          info.frozen += paramCount;
        }

        // Categorize by layer
        if (!info.byLayer[param.layer]) {
          info.byLayer[param.layer] = 0;
        }
        info.byLayer[param.layer] += paramCount;

        // Categorize by type
        if (param.name.includes('weight')) {
          info.byType.weights += paramCount;
        } else if (param.name.includes('bias')) {
          info.byType.biases += paramCount;
        } else if (param.name.includes('embedding')) {
          info.byType.embeddings += paramCount;
        } else if (param.name.includes('norm')) {
          info.byType.normalization += paramCount;
        } else {
          info.byType.other += paramCount;
        }
      });

    } catch (error) {
      info.error = error.message;
    }

    return info;
  }

  /**
   * Get layer information
   * @param {Object} model - Model object
   * @returns {Object} Layer info
   */
  getLayerInfo(model) {
    const info = {
      count: 0,
      types: {},
      shapes: {},
      operations: {},
      activationFunctions: {},
      details: []
    };

    try {
      const layers = this.extractLayers(model);
      info.count = layers.length;
      
      layers.forEach(layer => {
        // Count layer types
        if (!info.types[layer.type]) {
          info.types[layer.type] = 0;
        }
        info.types[layer.type]++;

        // Count shapes
        const shapeKey = JSON.stringify(layer.shape);
        if (!info.shapes[shapeKey]) {
          info.shapes[shapeKey] = 0;
        }
        info.shapes[shapeKey]++;

        // Count operations
        if (layer.operation) {
          if (!info.operations[layer.operation]) {
            info.operations[layer.operation] = 0;
          }
          info.operations[layer.operation]++;
        }

        // Count activation functions
        if (layer.activation) {
          if (!info.activationFunctions[layer.activation]) {
            info.activationFunctions[layer.activation] = 0;
          }
          info.activationFunctions[layer.activation]++;
        }

        info.details.push({
          id: layer.id,
          name: layer.name,
          type: layer.type,
          shape: layer.shape,
          parameters: layer.parameters,
          activation: layer.activation,
          trainable: layer.trainable
        });
      });

    } catch (error) {
      info.error = error.message;
    }

    return info;
  }

  /**
   * Get computation information
   * @param {Object} model - Model object
   * @returns {Object} Computation info
   */
  getComputationInfo(model) {
    const info = {
      totalFLOPs: 0,
      totalMACs: 0,
      operationCounts: {},
      computationGraph: [],
      complexity: 'unknown',
      bottlenecks: []
    };

    try {
      const operations = this.extractOperations(model);
      
      operations.forEach(op => {
        info.totalFLOPs += op.flops || 0;
        info.totalMACs += op.macs || 0;
        
        if (!info.operationCounts[op.type]) {
          info.operationCounts[op.type] = 0;
        }
        info.operationCounts[op.type]++;
        
        info.computationGraph.push({
          id: op.id,
          type: op.type,
          flops: op.flops,
          macs: op.macs,
          inputShape: op.inputShape,
          outputShape: op.outputShape
        });
      });

      // Estimate complexity
      if (info.totalFLOPs > 1e12) {
        info.complexity = 'very_high';
      } else if (info.totalFLOPs > 1e9) {
        info.complexity = 'high';
      } else if (info.totalFLOPs > 1e6) {
        info.complexity = 'medium';
      } else {
        info.complexity = 'low';
      }

      // Identify bottlenecks
      info.bottlenecks = this.identifyBottlenecks(operations);

    } catch (error) {
      info.error = error.message;
    }

    return info;
  }

  /**
   * Get weight information
   * @param {Object} model - Model object
   * @returns {Object} Weight info
   */
  getWeightInfo(model) {
    const info = {
      weights: [],
      statistics: {},
      distributions: {},
      healthCheck: {
        hasNaN: false,
        hasInfinite: false,
        hasLargeValues: false,
        hasSmallGradients: false
      }
    };

    try {
      const weights = this.extractWeights(model);
      
      weights.forEach(weight => {
        const weightInfo = {
          name: weight.name,
          shape: weight.shape,
          dtype: weight.dtype,
          analysis: tensorInspector.analyze(weight.tensor, {
            includeStatistics: true,
            includeDistribution: true,
            includeNaN: true,
            includeInfinite: true
          })
        };

        info.weights.push(weightInfo);

        // Update health check
        if (weightInfo.analysis.quality.hasNaN) {
          info.healthCheck.hasNaN = true;
        }
        if (weightInfo.analysis.quality.hasInfinite) {
          info.healthCheck.hasInfinite = true;
        }
        if (weightInfo.analysis.statistics && Math.abs(weightInfo.analysis.statistics.max) > 100) {
          info.healthCheck.hasLargeValues = true;
        }
      });

    } catch (error) {
      info.error = error.message;
    }

    return info;
  }

  /**
   * Get activation information
   * @param {Object} model - Model object
   * @returns {Object} Activation info
   */
  getActivationInfo(model) {
    const info = {
      activations: [],
      patterns: {},
      sparsity: {},
      healthCheck: {
        hasDeadNeurons: false,
        hasExplodingActivations: false,
        hasVanishingActivations: false
      }
    };

    try {
      const activations = this.extractActivations(model);
      
      activations.forEach(activation => {
        const activationInfo = {
          name: activation.name,
          layer: activation.layer,
          shape: activation.shape,
          analysis: tensorInspector.analyze(activation.tensor, {
            includeStatistics: true,
            includeDistribution: true
          })
        };

        info.activations.push(activationInfo);

        // Check for dead neurons (all zeros)
        if (activationInfo.analysis.statistics && activationInfo.analysis.statistics.max === 0) {
          info.healthCheck.hasDeadNeurons = true;
        }

        // Check for exploding activations
        if (activationInfo.analysis.statistics && Math.abs(activationInfo.analysis.statistics.max) > 1000) {
          info.healthCheck.hasExplodingActivations = true;
        }

        // Check for vanishing activations
        if (activationInfo.analysis.statistics && Math.abs(activationInfo.analysis.statistics.max) < 1e-6) {
          info.healthCheck.hasVanishingActivations = true;
        }
      });

    } catch (error) {
      info.error = error.message;
    }

    return info;
  }

  /**
   * Get gradient information
   * @param {Object} model - Model object
   * @returns {Object} Gradient info
   */
  getGradientInfo(model) {
    const info = {
      gradients: [],
      norms: {},
      healthCheck: {
        hasVanishingGradients: false,
        hasExplodingGradients: false,
        hasNaNGradients: false
      }
    };

    try {
      const gradients = this.extractGradients(model);
      
      gradients.forEach(gradient => {
        const gradientInfo = {
          name: gradient.name,
          layer: gradient.layer,
          shape: gradient.shape,
          analysis: tensorInspector.analyze(gradient.tensor, {
            includeStatistics: true,
            includeNaN: true,
            includeInfinite: true
          })
        };

        info.gradients.push(gradientInfo);

        // Check gradient health
        if (gradientInfo.analysis.quality.hasNaN) {
          info.healthCheck.hasNaNGradients = true;
        }

        if (gradientInfo.analysis.statistics) {
          const gradNorm = Math.sqrt(gradientInfo.analysis.statistics.variance);
          info.norms[gradient.name] = gradNorm;

          if (gradNorm < 1e-6) {
            info.healthCheck.hasVanishingGradients = true;
          }
          if (gradNorm > 100) {
            info.healthCheck.hasExplodingGradients = true;
          }
        }
      });

    } catch (error) {
      info.error = error.message;
    }

    return info;
  }

  /**
   * Get model memory information
   * @param {Object} model - Model object
   * @returns {Object} Memory info
   */
  getModelMemoryInfo(model) {
    const info = {
      totalMemory: 0,
      parameterMemory: 0,
      activationMemory: 0,
      gradientMemory: 0,
      bufferMemory: 0,
      breakdown: {},
      efficiency: 1.0
    };

    try {
      // Calculate parameter memory
      const parameters = this.extractParameters(model);
      parameters.forEach(param => {
        const memory = param.size * this.getBytesPerElement(param.dtype);
        info.parameterMemory += memory;
      });

      // Calculate activation memory (estimated)
      const layers = this.extractLayers(model);
      layers.forEach(layer => {
        if (layer.outputSize) {
          const memory = layer.outputSize * 4; // Assume F32
          info.activationMemory += memory;
        }
      });

      // Calculate gradient memory (same as parameters if training)
      if (this.isTrainingMode(model)) {
        info.gradientMemory = info.parameterMemory;
      }

      info.totalMemory = info.parameterMemory + info.activationMemory + info.gradientMemory;

      // Memory breakdown
      info.breakdown = {
        parameters: (info.parameterMemory / info.totalMemory) * 100,
        activations: (info.activationMemory / info.totalMemory) * 100,
        gradients: (info.gradientMemory / info.totalMemory) * 100
      };

    } catch (error) {
      info.error = error.message;
    }

    return info;
  }

  /**
   * Visualize model architecture as text
   * @param {Object} model - Model object
   * @param {Object} options - Visualization options
   * @returns {string} Text visualization
   */
  visualizeArchitecture(model, options = {}) {
    const {
      maxDepth = 10,
      showParameters = true,
      showShapes = true,
      showTypes = true,
      compact = false
    } = options;

    try {
      const analysis = this.analyzeModel(model, { maxDepth });
      let output = '';

      // Header
      output += `Model: ${analysis.basic.name}\n`;
      output += `Type: ${analysis.basic.type}\n`;
      output += `Total Parameters: ${analysis.parameters.total.toLocaleString()}\n`;
      output += `Total Layers: ${analysis.architecture.totalLayers}\n`;
      output += `Memory Usage: ${(analysis.memory.totalMemory / 1024 / 1024).toFixed(2)} MB\n`;
      output += '\n';

      // Architecture
      output += 'Architecture:\n';
      output += '============\n';
      
      analysis.architecture.layers.forEach((layer, index) => {
        const indent = '  '.repeat(layer.depth || 0);
        let line = `${indent}${index + 1}. ${layer.name || layer.type}`;
        
        if (showTypes) {
          line += ` (${layer.type})`;
        }
        
        if (showShapes && layer.shape) {
          line += ` -> ${JSON.stringify(layer.shape)}`;
        }
        
        if (showParameters && layer.parameters) {
          line += ` [${layer.parameters.toLocaleString()} params]`;
        }
        
        output += `${line}\n`;
      });

      // Layer type summary
      output += '\nLayer Types:\n';
      output += '============\n';
      Object.entries(analysis.layers.types).forEach(([type, count]) => {
        output += `${type}: ${count}\n`;
      });

      // Parameter summary
      output += '\nParameter Summary:\n';
      output += '==================\n';
      output += `Total: ${analysis.parameters.total.toLocaleString()}\n`;
      output += `Trainable: ${analysis.parameters.trainable.toLocaleString()}\n`;
      output += `Frozen: ${analysis.parameters.frozen.toLocaleString()}\n`;

      return output;
    } catch (error) {
      return `Error visualizing model: ${error.message}`;
    }
  }

  /**
   * Generate HTML visualization
   * @param {Object} model - Model object
   * @param {Object} options - Visualization options
   * @returns {string} HTML visualization
   */
  visualizeHTML(model, options = {}) {
    const {
      includeWeights = false,
      includeActivations = false,
      includeMemory = true,
      colorScheme = 'default'
    } = options;

    try {
      const analysis = this.analyzeModel(model, {
        includeWeights,
        includeActivations
      });

      let html = `
      <div class="model-visualization" style="font-family: 'Courier New', monospace; border: 1px solid #ccc; padding: 20px; margin: 10px; background: #f9f9f9;">
        <h2 style="margin-top: 0; color: #333;">Model Analysis: ${analysis.basic.name}</h2>
        
        <div class="model-overview" style="margin-bottom: 20px;">
          <h3>Overview</h3>
          <table style="border-collapse: collapse; width: 100%; margin-bottom: 15px;">
            <tr><td style="padding: 4px 8px; border: 1px solid #ddd; font-weight: bold;">Type:</td><td style="padding: 4px 8px; border: 1px solid #ddd;">${analysis.basic.type}</td></tr>
            <tr><td style="padding: 4px 8px; border: 1px solid #ddd; font-weight: bold;">Total Parameters:</td><td style="padding: 4px 8px; border: 1px solid #ddd;">${analysis.parameters.total.toLocaleString()}</td></tr>
            <tr><td style="padding: 4px 8px; border: 1px solid #ddd; font-weight: bold;">Total Layers:</td><td style="padding: 4px 8px; border: 1px solid #ddd;">${analysis.architecture.totalLayers}</td></tr>
            <tr><td style="padding: 4px 8px; border: 1px solid #ddd; font-weight: bold;">Training Mode:</td><td style="padding: 4px 8px; border: 1px solid #ddd;">${analysis.basic.isTraining ? 'Yes' : 'No'}</td></tr>
            <tr><td style="padding: 4px 8px; border: 1px solid #ddd; font-weight: bold;">Device:</td><td style="padding: 4px 8px; border: 1px solid #ddd;">${analysis.basic.device}</td></tr>
          </table>
        </div>

        <div class="architecture-features" style="margin-bottom: 20px;">
          <h3>Architecture Features</h3>
          <div style="display: flex; gap: 10px; flex-wrap: wrap;">
            <span style="padding: 4px 8px; background: ${analysis.architecture.hasAttention ? '#d4edda' : '#f8d7da'}; border-radius: 4px; font-size: 12px;">
              ${analysis.architecture.hasAttention ? '✓' : '✗'} Attention
            </span>
            <span style="padding: 4px 8px; background: ${analysis.architecture.hasEmbedding ? '#d4edda' : '#f8d7da'}; border-radius: 4px; font-size: 12px;">
              ${analysis.architecture.hasEmbedding ? '✓' : '✗'} Embedding
            </span>
            <span style="padding: 4px 8px; background: ${analysis.architecture.hasNormalization ? '#d4edda' : '#f8d7da'}; border-radius: 4px; font-size: 12px;">
              ${analysis.architecture.hasNormalization ? '✓' : '✗'} Normalization
            </span>
            <span style="padding: 4px 8px; background: ${analysis.architecture.hasDropout ? '#d4edda' : '#f8d7da'}; border-radius: 4px; font-size: 12px;">
              ${analysis.architecture.hasDropout ? '✓' : '✗'} Dropout
            </span>
          </div>
        </div>

        <div class="layer-types" style="margin-bottom: 20px;">
          <h3>Layer Types</h3>
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
      `;

      Object.entries(analysis.layers.types).forEach(([type, count]) => {
        html += `
          <div style="padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: #fff;">
            <strong>${type}</strong><br>
            <span style="color: #666; font-size: 14px;">${count} layer${count > 1 ? 's' : ''}</span>
          </div>
        `;
      });

      html += `
          </div>
        </div>

        <div class="parameter-breakdown" style="margin-bottom: 20px;">
          <h3>Parameter Breakdown</h3>
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
            <div style="padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: #e8f5e8;">
              <strong>Trainable</strong><br>
              <span style="font-size: 18px; color: #2d5016;">${analysis.parameters.trainable.toLocaleString()}</span>
            </div>
            <div style="padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: #f8f9fa;">
              <strong>Frozen</strong><br>
              <span style="font-size: 18px; color: #6c757d;">${analysis.parameters.frozen.toLocaleString()}</span>
            </div>
            <div style="padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: #fff3cd;">
              <strong>Total</strong><br>
              <span style="font-size: 18px; color: #856404;">${analysis.parameters.total.toLocaleString()}</span>
            </div>
          </div>
        </div>
      `;

      if (includeMemory) {
        html += `
        <div class="memory-usage" style="margin-bottom: 20px;">
          <h3>Memory Usage</h3>
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
            <div style="padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: #e3f2fd;">
              <strong>Parameters</strong><br>
              <span style="font-size: 16px; color: #1565c0;">${(analysis.memory.parameterMemory / 1024 / 1024).toFixed(2)} MB</span>
            </div>
            <div style="padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: #fff3e0;">
              <strong>Activations</strong><br>
              <span style="font-size: 16px; color: #ef6c00;">${(analysis.memory.activationMemory / 1024 / 1024).toFixed(2)} MB</span>
            </div>
            <div style="padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: #fce4ec;">
              <strong>Total</strong><br>
              <span style="font-size: 16px; color: #ad1457;">${(analysis.memory.totalMemory / 1024 / 1024).toFixed(2)} MB</span>
            </div>
          </div>
        </div>
        `;
      }

      // Health checks
      if (analysis.weights && analysis.weights.healthCheck) {
        const healthIssues = Object.entries(analysis.weights.healthCheck).filter(([_, value]) => value);
        if (healthIssues.length > 0) {
          html += `
          <div class="health-warnings" style="margin-top: 20px; padding: 15px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px;">
            <h3 style="margin-top: 0; color: #856404;">Health Warnings</h3>
          `;
          
          healthIssues.forEach(([issue, _]) => {
            html += `<div style="color: #d32f2f; margin-bottom: 5px;">⚠ ${issue.replace(/([A-Z])/g, ' $1').toLowerCase()}</div>`;
          });
          
          html += '</div>';
        }
      }

      html += '</div>';
      return html;
    } catch (error) {
      return `<div style="color: red; padding: 20px;">Error generating model visualization: ${error.message}</div>`;
    }
  }

  /**
   * Generate model summary report
   * @param {Object} model - Model object
   * @returns {Object} Summary report
   */
  summarize(model) {
    const analysis = this.analyzeModel(model);
    
    const summary = {
      name: analysis.basic.name,
      type: analysis.basic.type,
      parameters: analysis.parameters.total,
      layers: analysis.architecture.totalLayers,
      memoryMB: analysis.memory.totalMemory / 1024 / 1024,
      complexity: analysis.computation.complexity,
      hasIssues: false,
      issues: [],
      recommendations: []
    };

    // Check for issues
    if (analysis.weights && analysis.weights.healthCheck) {
      Object.entries(analysis.weights.healthCheck).forEach(([issue, hasIssue]) => {
        if (hasIssue) {
          summary.hasIssues = true;
          summary.issues.push(issue);
        }
      });
    }

    // Generate recommendations
    if (analysis.memory.totalMemory > 1024 * 1024 * 1024) { // > 1GB
      summary.recommendations.push('Consider model quantization to reduce memory usage');
    }
    
    if (analysis.computation.complexity === 'very_high') {
      summary.recommendations.push('Model has high computational complexity, consider pruning or distillation');
    }
    
    if (analysis.parameters.frozen > analysis.parameters.trainable) {
      summary.recommendations.push('Many parameters are frozen, consider fine-tuning strategy');
    }

    return summary;
  }

  /**
   * Clear visualization caches
   */
  clearCaches() {
    this.modelCache.clear();
    this.layerCache.clear();
    this.activationCache.clear();
    this.visualizationCache.clear();
  }

  // Private helper methods
  getModelId(model) {
    if (model._visualizerId) return model._visualizerId;
    model._visualizerId = `model_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    return model._visualizerId;
  }

  getModelType(model) {
    if (model.config && model.config.model_type) {
      return model.config.model_type;
    }
    if (model.model_type) {
      return model.model_type;
    }
    return model.constructor.name.toLowerCase();
  }

  hasModelConfig(model) {
    return !!(model.config || model.configuration);
  }

  isTrainingMode(model) {
    return model.training || model.train_mode || false;
  }

  isQuantized(model) {
    return model.quantized || model.is_quantized || false;
  }

  getModelDevice(model) {
    if (model.device) return model.device;
    if (model.config && model.config.device) return model.config.device;
    return 'cpu';
  }

  getArchitectureType(model) {
    const type = this.getModelType(model);
    if (type.includes('bert')) return 'encoder';
    if (type.includes('gpt')) return 'decoder';
    if (type.includes('t5')) return 'encoder-decoder';
    if (type.includes('llama')) return 'decoder';
    if (type.includes('mistral')) return 'decoder';
    return 'unknown';
  }

  extractLayers(model, maxDepth = 10) {
    const layers = [];
    // This is a simplified extraction - in practice, this would need to be
    // adapted for specific model formats and structures
    try {
      if (model.layers) {
        model.layers.forEach((layer, index) => {
          layers.push({
            id: index,
            name: layer.name || `layer_${index}`,
            type: layer.constructor.name.toLowerCase(),
            shape: layer.output_shape || layer.shape,
            parameters: this.countLayerParameters(layer),
            activation: layer.activation || null,
            trainable: layer.trainable !== false,
            depth: 0
          });
        });
      } else if (model.modules) {
        // Handle PyTorch-style modules
        Object.entries(model.modules).forEach(([name, module], index) => {
          layers.push({
            id: index,
            name,
            type: module.constructor.name.toLowerCase(),
            shape: module.weight ? module.weight.shape : null,
            parameters: this.countModuleParameters(module),
            trainable: module.requires_grad !== false,
            depth: 0
          });
        });
      }
    } catch (error) {
      debugUtils.warn('Error extracting layers', error);
    }
    return layers;
  }

  extractConnections(model) {
    // Simplified connection extraction
    return [];
  }

  extractParameters(model) {
    const parameters = [];
    try {
      if (model.parameters) {
        model.parameters().forEach((param, index) => {
          parameters.push({
            name: param.name || `param_${index}`,
            layer: param.layer || 'unknown',
            size: param.size || 0,
            shape: param.shape || [],
            dtype: param.dtype || 'f32',
            trainable: param.requires_grad !== false,
            tensor: param
          });
        });
      }
    } catch (error) {
      debugUtils.warn('Error extracting parameters', error);
    }
    return parameters;
  }

  extractOperations(model) {
    // This would need to be implemented based on model format
    return [];
  }

  extractWeights(model) {
    const weights = [];
    try {
      if (model.state_dict) {
        const stateDict = model.state_dict();
        Object.entries(stateDict).forEach(([name, tensor]) => {
          weights.push({
            name,
            shape: tensor.shape,
            dtype: tensor.dtype,
            tensor
          });
        });
      }
    } catch (error) {
      debugUtils.warn('Error extracting weights', error);
    }
    return weights;
  }

  extractActivations(model) {
    // This would need activation hooks to be implemented
    return [];
  }

  extractGradients(model) {
    // This would need gradient hooks to be implemented
    return [];
  }

  countLayerParameters(layer) {
    if (layer.parameters) {
      return layer.parameters().reduce((sum, param) => sum + param.size, 0);
    }
    return 0;
  }

  countModuleParameters(module) {
    if (module.parameters) {
      return module.parameters().reduce((sum, param) => sum + param.size, 0);
    }
    return 0;
  }

  identifyBottlenecks(operations) {
    return operations
      .filter(op => op.flops > 1e9) // > 1B FLOPs
      .sort((a, b) => b.flops - a.flops)
      .slice(0, 5)
      .map(op => ({
        operation: op.id,
        type: op.type,
        flops: op.flops,
        percentage: (op.flops / operations.reduce((sum, o) => sum + o.flops, 0)) * 100
      }));
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
}

// Global model visualizer instance
export const modelVisualizer = new ModelVisualizer();

// Convenience functions
export const visualize = {
  model: (model, options) => modelVisualizer.analyzeModel(model, options),
  architecture: (model, options) => modelVisualizer.visualizeArchitecture(model, options),
  html: (model, options) => modelVisualizer.visualizeHTML(model, options),
  summarize: (model) => modelVisualizer.summarize(model),
  clearCaches: () => modelVisualizer.clearCaches()
};

// Export for integration with other modules
export default ModelVisualizer;