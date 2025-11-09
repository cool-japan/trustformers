/**
 * Advanced Model Optimization Framework
 *
 * Provides cutting-edge optimization techniques including:
 * - Gradient checkpointing for memory efficiency
 * - Mixed precision training (FP16/BF16)
 * - Dynamic loss scaling
 * - Gradient accumulation
 * - Layer-wise adaptive rate scaling (LARS)
 * - Look-ahead optimizer
 * - Sharpness-Aware Minimization (SAM)
 * - Automated optimization strategy selection
 */

/**
 * Gradient Checkpointing Manager
 * Reduces memory usage by selectively storing intermediate activations
 */
export class GradientCheckpointingManager {
  constructor(config = {}) {
    this.enabled = config.enabled !== false;
    this.checkpointEveryN = config.checkpointEveryN || 2;
    this.checkpointedLayers = new Set();
    this.activationCache = new Map();
    this.memoryThreshold = config.memoryThreshold || 0.8; // 80% of available memory
    this.statistics = {
      activationsSaved: 0,
      activationsRecomputed: 0,
      memorySaved: 0,
      computeOverhead: 0,
    };
  }

  /**
   * Determine which layers should use checkpointing
   */
  selectCheckpointLayers(model, availableMemory) {
    const layers = model.layers || [];
    const layerMemory = layers.map((layer, idx) => ({
      idx,
      layer,
      memorySize: this.estimateLayerMemory(layer),
      computeCost: this.estimateComputeCost(layer),
    }));

    // Sort by memory/compute ratio to prioritize layers with high memory, low compute
    layerMemory.sort((a, b) => b.memorySize / b.computeCost - a.memorySize / a.computeCost);

    const checkpointedLayers = [];
    let savedMemory = 0;
    const targetSaving = availableMemory * (1 - this.memoryThreshold);

    for (const layerInfo of layerMemory) {
      if (savedMemory >= targetSaving) break;

      checkpointedLayers.push(layerInfo.idx);
      savedMemory += layerInfo.memorySize;
      this.checkpointedLayers.add(layerInfo.idx);
    }

    return {
      checkpointedLayers,
      memorySaved: savedMemory,
      totalLayers: layers.length,
    };
  }

  estimateLayerMemory(layer) {
    // Estimate memory usage based on layer type and parameters
    const paramCount = layer.parameters?.length || 0;
    const activationSize = layer.outputShape?.reduce((a, b) => a * b, 1) || 1000;
    return (paramCount + activationSize) * 4; // 4 bytes per float32
  }

  estimateComputeCost(layer) {
    // Estimate compute cost (FLOPs) for recomputation
    const type = layer.type || 'unknown';
    const paramCount = layer.parameters?.length || 0;

    const costs = {
      linear: paramCount * 2,
      conv2d: paramCount * 10,
      attention: paramCount * 5,
      layernorm: paramCount * 2,
      activation: paramCount * 0.5,
      unknown: paramCount,
    };

    return costs[type] || costs['unknown'];
  }

  /**
   * Cache activation for checkpointing
   */
  cacheActivation(layerId, activation) {
    if (!this.enabled || !this.checkpointedLayers.has(layerId)) {
      return false;
    }

    this.activationCache.set(layerId, activation);
    this.statistics.activationsSaved++;
    return true;
  }

  /**
   * Retrieve or recompute activation
   */
  getActivation(layerId, recomputeFn) {
    if (this.activationCache.has(layerId)) {
      return this.activationCache.get(layerId);
    }

    // Recompute activation
    const startTime = performance.now();
    const activation = recomputeFn();
    this.statistics.activationsRecomputed++;
    this.statistics.computeOverhead += performance.now() - startTime;

    return activation;
  }

  getStatistics() {
    return { ...this.statistics };
  }

  clear() {
    this.activationCache.clear();
  }
}

/**
 * Mixed Precision Training Manager
 * Implements FP16/BF16 training with dynamic loss scaling
 */
export class MixedPrecisionManager {
  constructor(config = {}) {
    this.enabled = config.enabled !== false;
    this.dtype = config.dtype || 'float16'; // 'float16' or 'bfloat16'
    this.lossScale = config.initialLossScale || 65536.0;
    this.lossScaleWindow = config.lossScaleWindow || 2000;
    this.minLossScale = config.minLossScale || 1.0;
    this.maxLossScale = config.maxLossScale || 2 ** 24;
    this.dynamicScaling = config.dynamicScaling !== false;

    this.stepsSinceLastScale = 0;
    this.consecutiveNonFiniteCount = 0;
    this.scalingHistory = [];

    this.statistics = {
      overflows: 0,
      underflows: 0,
      scaleAdjustments: 0,
      averageScale: this.lossScale,
    };
  }

  /**
   * Convert tensor to lower precision
   */
  castToLowPrecision(tensor) {
    if (!this.enabled) return tensor;

    // In real implementation, this would convert to FP16/BF16
    // For now, we simulate the behavior
    return {
      data: tensor.data,
      dtype: this.dtype,
      shape: tensor.shape,
      originalDtype: tensor.dtype || 'float32',
      _isMixedPrecision: true,
    };
  }

  /**
   * Convert tensor back to FP32 for accumulation
   */
  castToHighPrecision(tensor) {
    if (!tensor._isMixedPrecision) return tensor;

    return {
      data: tensor.data,
      dtype: 'float32',
      shape: tensor.shape,
      _isMixedPrecision: false,
    };
  }

  /**
   * Convert Float32Array to Float16 (Uint16Array representation)
   */
  toFloat16(tensor) {
    if (!this.enabled) return tensor;

    const float32 = tensor instanceof Float32Array ? tensor : new Float32Array(tensor);
    const float16 = new Uint16Array(float32.length);

    for (let i = 0; i < float32.length; i++) {
      // Simple FP16 conversion (simplified IEEE 754 half-precision)
      const f32 = float32[i];
      const sign = f32 < 0 ? 1 : 0;
      const abs = Math.abs(f32);

      if (abs === 0) {
        float16[i] = sign << 15;
      } else if (!isFinite(abs)) {
        float16[i] = (sign << 15) | 0x7c00; // Infinity
      } else {
        const exp = Math.floor(Math.log2(abs));
        const mantissa = (abs / Math.pow(2, exp) - 1) * 1024;
        const expBits = Math.max(-14, Math.min(15, exp)) + 15;
        float16[i] = (sign << 15) | (expBits << 10) | (mantissa & 0x3ff);
      }
    }

    return float16;
  }

  /**
   * Convert Float16 (Uint16Array) back to Float32Array
   */
  toFloat32(float16) {
    const uint16 = float16 instanceof Uint16Array ? float16 : new Uint16Array(float16);
    const float32 = new Float32Array(uint16.length);

    for (let i = 0; i < uint16.length; i++) {
      const bits = uint16[i];
      const sign = (bits >> 15) & 1 ? -1 : 1;
      const exp = (bits >> 10) & 0x1f;
      const mantissa = bits & 0x3ff;

      if (exp === 0) {
        float32[i] = sign * Math.pow(2, -14) * (mantissa / 1024);
      } else if (exp === 31) {
        float32[i] = mantissa === 0 ? sign * Infinity : NaN;
      } else {
        float32[i] = sign * Math.pow(2, exp - 15) * (1 + mantissa / 1024);
      }
    }

    return float32;
  }

  /**
   * Unscale loss after forward pass
   */
  unscaleLoss(scaledLoss) {
    if (!this.enabled || !this.dynamicScaling) return scaledLoss;
    return scaledLoss / this.lossScale;
  }

  /**
   * Scale loss for backward pass
   */
  scaleLoss(loss) {
    if (!this.enabled || !this.dynamicScaling) return loss;
    return loss * this.lossScale;
  }

  /**
   * Unscale gradients after backward pass
   */
  unscaleGradients(gradients) {
    if (!this.enabled || !this.dynamicScaling) return gradients;

    const unscaledGradients = gradients.map(grad => ({
      ...grad,
      data:
        grad.data instanceof Float32Array
          ? grad.data.map(v => v / this.lossScale)
          : grad.data / this.lossScale,
    }));

    return unscaledGradients;
  }

  /**
   * Check for gradient overflow/underflow and adjust scale
   */
  checkAndUpdateScale(gradients) {
    if (!this.dynamicScaling) return true;

    const hasNonFinite = gradients.some(grad => {
      const values = grad.data instanceof Float32Array ? grad.data : [grad.data];
      return values.some(v => !isFinite(v));
    });

    if (hasNonFinite) {
      this.consecutiveNonFiniteCount++;
      this.statistics.overflows++;

      // Reduce loss scale
      this.lossScale = Math.max(this.minLossScale, this.lossScale / 2);
      this.stepsSinceLastScale = 0;
      this.statistics.scaleAdjustments++;

      console.warn(`Mixed precision overflow detected. Reducing loss scale to ${this.lossScale}`);
      return false; // Skip this update
    }

    this.consecutiveNonFiniteCount = 0;
    this.stepsSinceLastScale++;

    // Increase loss scale if stable
    if (this.stepsSinceLastScale >= this.lossScaleWindow) {
      this.lossScale = Math.min(this.maxLossScale, this.lossScale * 2);
      this.stepsSinceLastScale = 0;
      this.statistics.scaleAdjustments++;
    }

    this.scalingHistory.push(this.lossScale);
    this.statistics.averageScale =
      this.scalingHistory.reduce((a, b) => a + b, 0) / this.scalingHistory.length;

    return true;
  }

  getStatistics() {
    return {
      ...this.statistics,
      currentScale: this.lossScale,
      stepsSinceLastScale: this.stepsSinceLastScale,
    };
  }
}

/**
 * Gradient Accumulation Manager
 * Enables training with larger effective batch sizes
 */
export class GradientAccumulationManager {
  constructor(config = {}) {
    this.accumulationSteps = config.accumulationSteps || 1;
    this.currentStep = 0;
    this.accumulatedGradients = null;
    this.enabled = this.accumulationSteps > 1;
  }

  /**
   * Accumulate gradients
   */
  accumulate(gradients) {
    if (!this.enabled) return { shouldUpdate: true, gradients };

    if (this.accumulatedGradients === null) {
      this.accumulatedGradients = gradients.map(g => ({
        ...g,
        data: g.data instanceof Float32Array ? new Float32Array(g.data) : g.data,
      }));
    } else {
      // Add gradients
      for (let i = 0; i < gradients.length; i++) {
        if (this.accumulatedGradients[i].data instanceof Float32Array) {
          for (let j = 0; j < this.accumulatedGradients[i].data.length; j++) {
            this.accumulatedGradients[i].data[j] += gradients[i].data[j];
          }
        } else {
          this.accumulatedGradients[i].data += gradients[i].data;
        }
      }
    }

    this.currentStep++;

    if (this.currentStep >= this.accumulationSteps) {
      // Average gradients
      const avgGradients = this.accumulatedGradients.map(g => ({
        ...g,
        data:
          g.data instanceof Float32Array
            ? g.data.map(v => v / this.accumulationSteps)
            : g.data / this.accumulationSteps,
      }));

      this.reset();
      return { shouldUpdate: true, gradients: avgGradients };
    }

    return { shouldUpdate: false, gradients: null };
  }

  reset() {
    this.currentStep = 0;
    this.accumulatedGradients = null;
  }
}

/**
 * LARS (Layer-wise Adaptive Rate Scaling) Optimizer
 * Enables stable training with large batch sizes
 */
export class LARSOptimizer {
  constructor(baseOptimizer, config = {}) {
    this.baseOptimizer = baseOptimizer;
    this.trustCoefficient = config.trustCoefficient || 0.001;
    this.epsilon = config.epsilon || 1e-8;
    this.excludeFromAdaptation = new Set(config.excludeFromAdaptation || ['bias', 'bn']);
  }

  /**
   * Compute LARS learning rate for each layer
   */
  computeLayerLR(params, grads, baseLR) {
    const layerLRs = [];

    for (let i = 0; i < params.length; i++) {
      const param = params[i];
      const grad = grads[i];

      // Skip LARS for excluded parameters
      if (this.shouldExclude(param.name)) {
        layerLRs.push(baseLR);
        continue;
      }

      // Compute L2 norms
      const paramNorm = this.computeL2Norm(param.data);
      const gradNorm = this.computeL2Norm(grad.data);

      // LARS formula: local_lr = trust_coefficient * ||params|| / (||gradients|| + epsilon)
      const localLR = (this.trustCoefficient * paramNorm) / (gradNorm + this.epsilon);
      layerLRs.push(localLR * baseLR);
    }

    return layerLRs;
  }

  shouldExclude(paramName) {
    return Array.from(this.excludeFromAdaptation).some(
      pattern => paramName && paramName.includes(pattern)
    );
  }

  computeL2Norm(data) {
    if (data instanceof Float32Array) {
      return Math.sqrt(data.reduce((sum, val) => sum + val * val, 0));
    }
    return Math.abs(data);
  }

  /**
   * Compute LARS update for given weights and gradients
   */
  computeUpdate(weights, gradients, layerId) {
    const weightData = weights instanceof Float32Array ? weights : new Float32Array(weights);
    const gradData = gradients instanceof Float32Array ? gradients : new Float32Array(gradients);

    // Compute L2 norms
    const weightNorm = this.computeL2Norm(weightData);
    const gradNorm = this.computeL2Norm(gradData);

    // LARS formula: local_lr = trust_coefficient * ||weights|| / (||gradients|| + epsilon)
    const localLR = (this.trustCoefficient * weightNorm) / (gradNorm + this.epsilon);

    // Apply update: update = -localLR * gradients
    const update = new Float32Array(gradData.length);
    for (let i = 0; i < gradData.length; i++) {
      update[i] = -localLR * gradData[i];
    }

    return update;
  }

  step(params, grads, baseLR) {
    const layerLRs = this.computeLayerLR(params, grads, baseLR);

    // Apply base optimizer with layer-specific learning rates
    return this.baseOptimizer.step(params, grads, layerLRs);
  }
}

/**
 * Lookahead Optimizer
 * Improves convergence and generalization
 */
export class LookaheadOptimizer {
  constructor(baseOptimizer, config = {}) {
    this.baseOptimizer = baseOptimizer;
    this.alpha = config.alpha || 0.5; // Slow weights step size
    this.k = config.k || 5; // Number of fast weight updates
    this.currentStep = 0;
    this.slowWeights = null;
  }

  /**
   * Initialize slow weights
   */
  initSlowWeights(params) {
    this.slowWeights = params.map(p => ({
      ...p,
      data: p.data instanceof Float32Array ? new Float32Array(p.data) : p.data,
    }));
  }

  /**
   * Perform lookahead optimization step
   */
  step(params, grads, lr) {
    // Initialize slow weights on first step
    if (this.slowWeights === null) {
      this.initSlowWeights(params);
    }

    // Update fast weights using base optimizer
    const updatedParams = this.baseOptimizer.step(params, grads, lr);

    this.currentStep++;

    // Update slow weights every k steps
    if (this.currentStep % this.k === 0) {
      for (let i = 0; i < updatedParams.length; i++) {
        if (updatedParams[i].data instanceof Float32Array) {
          for (let j = 0; j < updatedParams[i].data.length; j++) {
            this.slowWeights[i].data[j] +=
              this.alpha * (updatedParams[i].data[j] - this.slowWeights[i].data[j]);
          }
          // Copy slow weights back to fast weights
          updatedParams[i].data.set(this.slowWeights[i].data);
        } else {
          this.slowWeights[i].data +=
            this.alpha * (updatedParams[i].data - this.slowWeights[i].data);
          updatedParams[i].data = this.slowWeights[i].data;
        }
      }
    }

    return updatedParams;
  }
}

/**
 * Sharpness-Aware Minimization (SAM) Optimizer
 * Seeks parameters in flat minima for better generalization
 */
export class SAMOptimizer {
  constructor(baseOptimizer, config = {}) {
    this.baseOptimizer = baseOptimizer;
    this.rho = config.rho || 0.05; // Neighborhood size
    this.adaptive = config.adaptive !== false;
  }

  /**
   * Perform SAM optimization step
   */
  async step(params, gradients, lr, lossFunction) {
    // First ascent step: find adversarial perturbation
    const epsilon = this.computeAdversarialPerturbation(params, gradients);

    // Perturb parameters
    const perturbedParams = this.perturbParameters(params, epsilon);

    // Compute gradients at perturbed point
    const perturbedGrads = await lossFunction(perturbedParams);

    // Descent step: update with perturbed gradients
    const updatedParams = this.baseOptimizer.step(params, perturbedGrads, lr);

    return updatedParams;
  }

  computeAdversarialPerturbation(params, gradients) {
    return gradients.map((grad, i) => {
      const param = params[i];
      const gradNorm = this.computeL2Norm(grad.data);

      if (gradNorm < 1e-12) {
        return { data: new Float32Array(grad.data.length) };
      }

      // Adaptive SAM
      const scale = this.adaptive
        ? this.rho / (gradNorm * (1 + this.computeL2Norm(param.data)))
        : this.rho / gradNorm;

      return {
        data: grad.data instanceof Float32Array ? grad.data.map(v => v * scale) : grad.data * scale,
      };
    });
  }

  perturbParameters(params, epsilon) {
    return params.map((param, i) => ({
      ...param,
      data:
        param.data instanceof Float32Array
          ? param.data.map((v, j) => v + epsilon[i].data[j])
          : param.data + epsilon[i].data,
    }));
  }

  computeL2Norm(data) {
    if (data instanceof Float32Array) {
      return Math.sqrt(data.reduce((sum, val) => sum + val * val, 0));
    }
    return Math.abs(data);
  }

  /**
   * Compute adversarial perturbation for SAM
   */
  computePerturbation(gradients) {
    const gradData = gradients instanceof Float32Array ? gradients : new Float32Array(gradients);
    const gradNorm = this.computeL2Norm(gradData);

    if (gradNorm < 1e-12) {
      return new Float32Array(gradData.length);
    }

    // Compute perturbation: epsilon = rho * gradients / ||gradients||
    const scale = this.rho / gradNorm;
    const perturbation = new Float32Array(gradData.length);

    for (let i = 0; i < gradData.length; i++) {
      perturbation[i] = scale * gradData[i];
    }

    return perturbation;
  }
}

/**
 * Automated Optimization Strategy Selector
 * Selects optimal optimization techniques based on model and data characteristics
 */
export class OptimizationStrategySelector {
  constructor() {
    this.strategies = new Map();
    this.initializeStrategies();
  }

  initializeStrategies() {
    this.strategies.set('memory_constrained', {
      name: 'Memory Constrained',
      techniques: ['gradient_checkpointing', 'mixed_precision'],
      priority: ['gradient_checkpointing', 'mixed_precision', 'gradient_accumulation'],
      description: 'Optimizes for low memory environments',
    });

    this.strategies.set('large_batch', {
      name: 'Large Batch Training',
      techniques: ['lars', 'mixed_precision', 'gradient_accumulation'],
      priority: ['lars', 'gradient_accumulation', 'mixed_precision'],
      description: 'Enables stable training with large batch sizes',
    });

    this.strategies.set('fast_convergence', {
      name: 'Fast Convergence',
      techniques: ['lookahead', 'sam', 'mixed_precision'],
      priority: ['lookahead', 'sam'],
      description: 'Optimizes for faster convergence',
    });

    this.strategies.set('best_generalization', {
      name: 'Best Generalization',
      techniques: ['sam', 'lookahead'],
      priority: ['sam', 'lookahead', 'gradient_checkpointing'],
      description: 'Seeks flat minima for better generalization',
    });

    this.strategies.set('balanced', {
      name: 'Balanced',
      techniques: ['mixed_precision', 'gradient_checkpointing', 'lookahead'],
      priority: ['mixed_precision', 'lookahead'],
      description: 'Balanced approach for most use cases',
    });
  }

  /**
   * Automatically select optimization strategy
   */
  selectStrategy(config) {
    const {
      modelSize = 'medium',
      availableMemory = 'medium',
      batchSize = 32,
      prioritizeSpeed = false,
      prioritizeGeneralization = false,
    } = config;

    // Memory-constrained scenario
    if (availableMemory === 'low' || modelSize === 'large') {
      return this.strategies.get('memory_constrained');
    }

    // Large batch scenario
    if (batchSize >= 512) {
      return this.strategies.get('large_batch');
    }

    // Generalization priority
    if (prioritizeGeneralization) {
      return this.strategies.get('best_generalization');
    }

    // Speed priority
    if (prioritizeSpeed) {
      return this.strategies.get('fast_convergence');
    }

    // Default balanced approach
    return this.strategies.get('balanced');
  }

  /**
   * Get recommended configuration
   */
  getRecommendedConfig(strategy, modelConfig) {
    const baseConfig = {};

    if (strategy.techniques.includes('gradient_checkpointing')) {
      baseConfig.gradientCheckpointing = {
        enabled: true,
        checkpointEveryN: 2,
        memoryThreshold: 0.8,
      };
    }

    if (strategy.techniques.includes('mixed_precision')) {
      baseConfig.mixedPrecision = {
        enabled: true,
        dtype: 'float16',
        dynamicScaling: true,
      };
    }

    if (strategy.techniques.includes('gradient_accumulation')) {
      baseConfig.gradientAccumulation = {
        accumulationSteps: Math.max(2, Math.floor(512 / modelConfig.batchSize)),
      };
    }

    if (strategy.techniques.includes('lars')) {
      baseConfig.lars = {
        enabled: true,
        trustCoefficient: 0.001,
      };
    }

    if (strategy.techniques.includes('lookahead')) {
      baseConfig.lookahead = {
        enabled: true,
        alpha: 0.5,
        k: 5,
      };
    }

    if (strategy.techniques.includes('sam')) {
      baseConfig.sam = {
        enabled: true,
        rho: 0.05,
        adaptive: true,
      };
    }

    return baseConfig;
  }

  listStrategies() {
    return Array.from(this.strategies.values()).map(s => ({
      name: s.name,
      techniques: s.techniques,
      description: s.description,
    }));
  }
}

/**
 * Unified Advanced Optimizer
 * Combines all optimization techniques with automatic configuration
 */
export class AdvancedOptimizer {
  constructor(baseOptimizer, config = {}) {
    this.baseOptimizer = baseOptimizer;
    this.config = config;

    // Initialize managers
    this.checkpointManager = config.gradientCheckpointing?.enabled
      ? new GradientCheckpointingManager(config.gradientCheckpointing)
      : null;

    this.mixedPrecisionManager = config.mixedPrecision?.enabled
      ? new MixedPrecisionManager(config.mixedPrecision)
      : null;

    this.gradAccumManager =
      config.gradientAccumulation?.accumulationSteps > 1
        ? new GradientAccumulationManager(config.gradientAccumulation)
        : null;

    // Wrap optimizer with advanced techniques
    let optimizer = baseOptimizer;

    if (config.lars?.enabled) {
      optimizer = new LARSOptimizer(optimizer, config.lars);
    }

    if (config.lookahead?.enabled) {
      optimizer = new LookaheadOptimizer(optimizer, config.lookahead);
    }

    if (config.sam?.enabled) {
      optimizer = new SAMOptimizer(optimizer, config.sam);
    }

    this.optimizer = optimizer;

    this.statistics = {
      steps: 0,
      skippedUpdates: 0,
      totalComputeTime: 0,
      totalMemorySaved: 0,
    };
  }

  /**
   * Perform optimization step with all enabled techniques
   */
  async step(params, gradients, lr, lossFunction = null) {
    const startTime = performance.now();
    this.statistics.steps++;

    // Mixed precision: unscale gradients
    if (this.mixedPrecisionManager) {
      gradients = this.mixedPrecisionManager.unscaleGradients(gradients);

      // Check for overflow
      if (!this.mixedPrecisionManager.checkAndUpdateScale(gradients)) {
        this.statistics.skippedUpdates++;
        return params; // Skip update on overflow
      }
    }

    // Gradient accumulation
    if (this.gradAccumManager) {
      const result = this.gradAccumManager.accumulate(gradients);
      if (!result.shouldUpdate) {
        return params; // Don't update yet
      }
      gradients = result.gradients;
    }

    // Apply optimizer (potentially SAM, LARS, Lookahead)
    let updatedParams;
    if (this.config.sam?.enabled && lossFunction) {
      updatedParams = await this.optimizer.step(params, gradients, lr, lossFunction);
    } else {
      updatedParams = this.optimizer.step(params, gradients, lr);
    }

    this.statistics.totalComputeTime += performance.now() - startTime;

    return updatedParams;
  }

  /**
   * Get comprehensive statistics
   */
  getStatistics() {
    const stats = { ...this.statistics };

    if (this.checkpointManager) {
      stats.checkpointing = this.checkpointManager.getStatistics();
    }

    if (this.mixedPrecisionManager) {
      stats.mixedPrecision = this.mixedPrecisionManager.getStatistics();
    }

    return stats;
  }

  /**
   * Clear caches and reset state
   */
  reset() {
    if (this.checkpointManager) {
      this.checkpointManager.clear();
    }

    if (this.gradAccumManager) {
      this.gradAccumManager.reset();
    }
  }
}

/**
 * Create advanced optimizer with automatic strategy selection
 */
export function createAdvancedOptimizer(baseOptimizer, modelConfig = {}) {
  const selector = new OptimizationStrategySelector();
  const strategy = selector.selectStrategy(modelConfig);
  const config = selector.getRecommendedConfig(strategy, modelConfig);

  console.log(`Selected optimization strategy: ${strategy.name}`);
  console.log(`Techniques: ${strategy.techniques.join(', ')}`);

  return new AdvancedOptimizer(baseOptimizer, config);
}

// All components already exported via 'export class' and 'export function' declarations above
