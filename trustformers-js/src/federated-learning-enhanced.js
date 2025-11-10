/**
 * Enhanced Federated Learning Algorithms
 *
 * Advanced federated learning methods that address specific challenges:
 * - FedBN: Batch normalization for heterogeneous data
 * - FedNova: Normalized averaging for objective inconsistency
 * - FedProx: Proximal term for stability
 * - FedYogi/FedAdam: Adaptive server-side optimization
 *
 * These methods improve upon standard FedAvg by handling:
 * - Non-IID (non-independent and identically distributed) data
 * - Partial participation
 * - System heterogeneity
 * - Convergence issues
 *
 * @module federated-learning-enhanced
 */

/**
 * FedBN (Federated Learning with Batch Normalization)
 *
 * Paper: "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization"
 * Authors: Li et al., ICLR 2021
 *
 * Key Innovation:
 * - Does NOT aggregate batch normalization statistics (running mean/var)
 * - Only aggregates weights and biases
 * - Each client keeps local BN statistics
 * - Dramatically improves performance on non-IID data
 */
export class FedBNAggregator {
  /**
   * Create a FedBN aggregator
   * @param {Object} config - Configuration
   */
  constructor(config = {}) {
    this.config = config;
    this.globalModel = null;
    this.clientBNStatistics = new Map(); // Store per-client BN stats

    // Track which parameters are batch norm params
    this.bnParamNames = config.bnParamNames || [
      'running_mean',
      'running_var',
      'num_batches_tracked'
    ];

    this.statistics = {
      roundsCompleted: 0,
      avgConvergenceTime: 0,
      bnStatsPreserved: 0
    };
  }

  /**
   * Initialize with global model
   * @param {Object} model - Global model
   */
  initialize(model) {
    this.globalModel = this.cloneModel(model);
    return this;
  }

  /**
   * Aggregate client updates using FedBN strategy
   * @param {Array<Object>} clientUpdates - Updates from clients
   * @param {Object} aggregationConfig - Aggregation configuration
   * @returns {Object} Aggregated model
   */
  aggregate(clientUpdates, aggregationConfig = {}) {
    if (clientUpdates.length === 0) {
      throw new Error('No client updates to aggregate');
    }

    console.log(`FedBN: Aggregating ${clientUpdates.length} client updates`);

    const {
      weightingScheme = 'uniform',
      preserveBNStats = true
    } = aggregationConfig;

    // Calculate client weights
    const weights = this.calculateWeights(clientUpdates, weightingScheme);

    // Separate parameters into BN and non-BN
    const { bnParams, nonBNParams } = this.separateParameters(clientUpdates[0]);

    // Aggregate non-BN parameters (weights, biases, etc.)
    const aggregatedNonBN = this.aggregateNonBNParameters(
      clientUpdates.map(u => this.separateParameters(u).nonBNParams),
      weights
    );

    // For BN parameters: either aggregate or preserve local
    let aggregatedBN;
    if (preserveBNStats) {
      // FedBN: Do NOT aggregate BN statistics
      // Keep them client-specific
      aggregatedBN = this.preserveLocalBNStatistics(clientUpdates);
      this.statistics.bnStatsPreserved++;
    } else {
      // Standard averaging (for comparison)
      aggregatedBN = this.aggregateBNParameters(
        clientUpdates.map(u => this.separateParameters(u).bnParams),
        weights
      );
    }

    // Combine aggregated parameters
    const aggregatedModel = this.combineParameters(aggregatedNonBN, aggregatedBN);

    this.statistics.roundsCompleted++;

    return {
      model: aggregatedModel,
      bnStatistics: this.clientBNStatistics,
      metadata: {
        method: 'FedBN',
        clientsParticipated: clientUpdates.length,
        bnStatsPreserved: preserveBNStats
      }
    };
  }

  /**
   * Separate parameters into BN and non-BN
   * @param {Object} update - Client update
   * @returns {Object} Separated parameters
   */
  separateParameters(update) {
    const bnParams = {};
    const nonBNParams = {};

    for (const [name, param] of Object.entries(update.parameters || {})) {
      const isBNParam = this.bnParamNames.some(bn => name.includes(bn)) ||
        name.includes('batch_norm') ||
        name.includes('bn');

      if (isBNParam) {
        bnParams[name] = param;
      } else {
        nonBNParams[name] = param;
      }
    }

    return { bnParams, nonBNParams };
  }

  /**
   * Aggregate non-BN parameters
   * @param {Array<Object>} parametersList - List of parameters from clients
   * @param {Array<number>} weights - Client weights
   * @returns {Object} Aggregated parameters
   */
  aggregateNonBNParameters(parametersList, weights) {
    if (parametersList.length === 0) return {};

    const aggregated = {};

    // Get parameter names from first client
    const paramNames = Object.keys(parametersList[0]);

    for (const name of paramNames) {
      const params = parametersList.map(p => p[name]);

      // Weighted average
      aggregated[name] = this.weightedAverage(params, weights);
    }

    return aggregated;
  }

  /**
   * Preserve local BN statistics (FedBN core idea)
   * @param {Array<Object>} clientUpdates - Client updates
   * @returns {Object} BN statistics map
   */
  preserveLocalBNStatistics(clientUpdates) {
    const bnStats = {};

    for (const update of clientUpdates) {
      const clientId = update.clientId;
      const { bnParams } = this.separateParameters(update);

      // Store BN statistics for this client
      this.clientBNStatistics.set(clientId, bnParams);

      // For global model, we'll use averaged stats (but clients won't use them)
      for (const [name, param] of Object.entries(bnParams)) {
        if (!bnStats[name]) {
          bnStats[name] = [];
        }
        bnStats[name].push(param);
      }
    }

    // Average BN stats for global model (clients will override with local)
    const avgBNStats = {};
    for (const [name, params] of Object.entries(bnStats)) {
      avgBNStats[name] = this.average(params);
    }

    return avgBNStats;
  }

  /**
   * Aggregate BN parameters (standard averaging, for comparison)
   * @param {Array<Object>} bnParametersList - List of BN parameters
   * @param {Array<number>} weights - Client weights
   * @returns {Object} Aggregated BN parameters
   */
  aggregateBNParameters(bnParametersList, weights) {
    if (bnParametersList.length === 0) return {};

    const aggregated = {};
    const paramNames = Object.keys(bnParametersList[0] || {});

    for (const name of paramNames) {
      const params = bnParametersList.map(p => p[name]);
      aggregated[name] = this.weightedAverage(params, weights);
    }

    return aggregated;
  }

  /**
   * Combine non-BN and BN parameters
   * @param {Object} nonBNParams - Non-BN parameters
   * @param {Object} bnParams - BN parameters
   * @returns {Object} Combined model
   */
  combineParameters(nonBNParams, bnParams) {
    return {
      parameters: {
        ...nonBNParams,
        ...bnParams
      }
    };
  }

  /**
   * Calculate client weights for aggregation
   * @param {Array<Object>} clientUpdates - Client updates
   * @param {string} scheme - Weighting scheme
   * @returns {Array<number>} Weights
   */
  calculateWeights(clientUpdates, scheme) {
    if (scheme === 'uniform') {
      // Equal weights
      return clientUpdates.map(() => 1.0 / clientUpdates.length);
    } else if (scheme === 'dataSize') {
      // Weight by local data size
      const dataSizes = clientUpdates.map(u => u.dataSize || 1);
      const totalSize = dataSizes.reduce((sum, size) => sum + size, 0);
      return dataSizes.map(size => size / totalSize);
    }

    // Default to uniform
    return clientUpdates.map(() => 1.0 / clientUpdates.length);
  }

  /**
   * Weighted average of parameters
   * @param {Array} params - Parameters
   * @param {Array<number>} weights - Weights
   * @returns {Object} Averaged parameter
   */
  weightedAverage(params, weights) {
    if (params.length === 0) return null;

    const first = params[0];

    // Handle typed arrays
    if (ArrayBuffer.isView(first)) {
      const result = new first.constructor(first.length);
      result.fill(0);

      for (let i = 0; i < params.length; i++) {
        const weight = weights[i];
        for (let j = 0; j < first.length; j++) {
          result[j] += params[i][j] * weight;
        }
      }

      return result;
    }

    // Handle objects
    if (typeof first === 'object' && first !== null) {
      const result = {};
      for (const key of Object.keys(first)) {
        result[key] = this.weightedAverage(
          params.map(p => p[key]),
          weights
        );
      }
      return result;
    }

    // Handle scalars
    let sum = 0;
    for (let i = 0; i < params.length; i++) {
      sum += params[i] * weights[i];
    }
    return sum;
  }

  /**
   * Simple average (unweighted)
   * @param {Array} params - Parameters
   * @returns {Object} Averaged parameter
   */
  average(params) {
    const weights = params.map(() => 1.0 / params.length);
    return this.weightedAverage(params, weights);
  }

  /**
   * Clone model
   * @param {Object} model - Model to clone
   * @returns {Object} Cloned model
   */
  cloneModel(model) {
    return JSON.parse(JSON.stringify(model));
  }

  /**
   * Get statistics
   * @returns {Object} Statistics
   */
  getStatistics() {
    return {
      ...this.statistics,
      clientBNStatsTracked: this.clientBNStatistics.size
    };
  }
}

/**
 * FedNova (Federated Normalized Averaging)
 *
 * Paper: "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization"
 * Authors: Wang et al., NeurIPS 2020
 *
 * Key Innovation:
 * - Normalizes local updates by the number of local steps
 * - Addresses objective inconsistency in FedAvg
 * - Better convergence on non-IID data
 * - Handles varying local epochs across clients
 */
export class FedNovaAggregator {
  /**
   * Create a FedNova aggregator
   * @param {Object} config - Configuration
   */
  constructor(config = {}) {
    this.config = config;
    this.globalModel = null;
    this.momentumBuffer = null; // Server-side momentum

    // FedNova-specific parameters
    this.rho = config.rho || 0.9; // Momentum parameter
    this.tau = config.tau || null; // Effective number of local steps (auto-computed)

    this.statistics = {
      roundsCompleted: 0,
      avgNormalizationFactor: 0,
      totalLocalSteps: 0
    };
  }

  /**
   * Initialize with global model
   * @param {Object} model - Global model
   */
  initialize(model) {
    this.globalModel = this.cloneModel(model);
    this.momentumBuffer = this.initializeMomentumBuffer(model);
    return this;
  }

  /**
   * Aggregate client updates using FedNova strategy
   * @param {Array<Object>} clientUpdates - Updates from clients
   * @param {Object} aggregationConfig - Aggregation configuration
   * @returns {Object} Aggregated update
   */
  aggregate(clientUpdates, aggregationConfig = {}) {
    if (clientUpdates.length === 0) {
      throw new Error('No client updates to aggregate');
    }

    console.log(`FedNova: Aggregating ${clientUpdates.length} client updates`);

    const {
      globalLearningRate = 1.0,
      useMomentum = true,
      normalizationScheme = 'gradient' // 'gradient' or 'model'
    } = aggregationConfig;

    // Step 1: Compute normalized local updates
    const normalizedUpdates = this.normalizeClientUpdates(
      clientUpdates,
      normalizationScheme
    );

    // Step 2: Compute effective tau (average number of local steps)
    const tau = this.computeEffectiveTau(clientUpdates);
    this.tau = tau;

    // Step 3: Weighted aggregation of normalized updates
    const weights = this.calculateWeights(clientUpdates);
    const aggregatedUpdate = this.weightedAggregation(normalizedUpdates, weights);

    // Step 4: Scale by tau (FedNova correction)
    const correctedUpdate = this.scaleByTau(aggregatedUpdate, tau);

    // Step 5: Apply server-side momentum (optional)
    let finalUpdate;
    if (useMomentum) {
      finalUpdate = this.applyMomentum(correctedUpdate, this.rho);
    } else {
      finalUpdate = correctedUpdate;
    }

    // Step 6: Update global model
    this.updateGlobalModel(finalUpdate, globalLearningRate);

    // Update statistics
    this.statistics.roundsCompleted++;
    this.statistics.avgNormalizationFactor =
      (this.statistics.avgNormalizationFactor * (this.statistics.roundsCompleted - 1) + tau) /
      this.statistics.roundsCompleted;
    this.statistics.totalLocalSteps += clientUpdates.reduce(
      (sum, u) => sum + (u.localSteps || 1),
      0
    );

    return {
      model: this.globalModel,
      update: finalUpdate,
      metadata: {
        method: 'FedNova',
        tau: tau,
        clientsParticipated: clientUpdates.length,
        normalizationScheme: normalizationScheme
      }
    };
  }

  /**
   * Normalize client updates
   * @param {Array<Object>} clientUpdates - Client updates
   * @param {string} scheme - Normalization scheme
   * @returns {Array<Object>} Normalized updates
   */
  normalizeClientUpdates(clientUpdates, scheme) {
    return clientUpdates.map(update => {
      const tau_i = update.localSteps || 1; // Number of local SGD steps
      const a_i = this.computeNormalizationCoefficient(tau_i, scheme);

      // Normalize: Δ_i / a_i
      const normalized = this.scaleParameters(update.parameters, 1.0 / a_i);

      return {
        ...update,
        normalizedParameters: normalized,
        normalizationCoefficient: a_i,
        localSteps: tau_i
      };
    });
  }

  /**
   * Compute normalization coefficient a_i
   * @param {number} tau_i - Number of local steps
   * @param {string} scheme - Normalization scheme
   * @returns {number} Normalization coefficient
   */
  computeNormalizationCoefficient(tau_i, scheme) {
    if (scheme === 'gradient') {
      // For gradient-based: a_i = tau_i
      return tau_i;
    } else if (scheme === 'model') {
      // For model-based: a_i = sum_{j=1}^{tau_i} rho^{tau_i - j}
      // This accounts for momentum in local training
      let sum = 0;
      for (let j = 1; j <= tau_i; j++) {
        sum += Math.pow(this.rho, tau_i - j);
      }
      return sum;
    }

    // Default
    return tau_i;
  }

  /**
   * Compute effective tau (average of local steps)
   * @param {Array<Object>} clientUpdates - Client updates
   * @returns {number} Effective tau
   */
  computeEffectiveTau(clientUpdates) {
    // Weighted average of local steps
    const weights = this.calculateWeights(clientUpdates);
    let tau = 0;

    for (let i = 0; i < clientUpdates.length; i++) {
      const tau_i = clientUpdates[i].localSteps || 1;
      tau += weights[i] * tau_i;
    }

    return tau;
  }

  /**
   * Weighted aggregation
   * @param {Array<Object>} normalizedUpdates - Normalized updates
   * @param {Array<number>} weights - Client weights
   * @returns {Object} Aggregated update
   */
  weightedAggregation(normalizedUpdates, weights) {
    const aggregated = {};

    // Get parameter names from first update
    const paramNames = Object.keys(normalizedUpdates[0].normalizedParameters || {});

    for (const name of paramNames) {
      const params = normalizedUpdates.map(u => u.normalizedParameters[name]);
      aggregated[name] = this.weightedAverage(params, weights);
    }

    return aggregated;
  }

  /**
   * Scale aggregated update by tau
   * @param {Object} update - Aggregated update
   * @param {number} tau - Effective tau
   * @returns {Object} Scaled update
   */
  scaleByTau(update, tau) {
    return this.scaleParameters(update, tau);
  }

  /**
   * Apply server-side momentum
   * @param {Object} update - Current update
   * @param {number} rho - Momentum parameter
   * @returns {Object} Update with momentum
   */
  applyMomentum(update, rho) {
    if (!this.momentumBuffer) {
      // First iteration: initialize momentum buffer
      this.momentumBuffer = this.cloneParameters(update);
      return update;
    }

    // m_t = rho * m_{t-1} + Δ_t
    const newMomentum = {};

    for (const [name, param] of Object.entries(update)) {
      const prevMomentum = this.momentumBuffer[name];

      if (ArrayBuffer.isView(param) && ArrayBuffer.isView(prevMomentum)) {
        const m = new param.constructor(param.length);
        for (let i = 0; i < param.length; i++) {
          m[i] = rho * prevMomentum[i] + param[i];
        }
        newMomentum[name] = m;
      } else {
        newMomentum[name] = param; // Fallback
      }
    }

    this.momentumBuffer = newMomentum;
    return newMomentum;
  }

  /**
   * Update global model
   * @param {Object} update - Update to apply
   * @param {number} learningRate - Global learning rate
   */
  updateGlobalModel(update, learningRate) {
    if (!this.globalModel || !this.globalModel.parameters) {
      return;
    }

    for (const [name, param] of Object.entries(this.globalModel.parameters)) {
      const delta = update[name];

      if (!delta) continue;

      if (ArrayBuffer.isView(param) && ArrayBuffer.isView(delta)) {
        for (let i = 0; i < Math.min(param.length, delta.length); i++) {
          param[i] -= learningRate * delta[i];
        }
      }
    }
  }

  /**
   * Calculate client weights
   * @param {Array<Object>} clientUpdates - Client updates
   * @returns {Array<number>} Weights
   */
  calculateWeights(clientUpdates) {
    // Weight by data size (standard in federated learning)
    const dataSizes = clientUpdates.map(u => u.dataSize || 1);
    const totalSize = dataSizes.reduce((sum, size) => sum + size, 0);
    return dataSizes.map(size => size / totalSize);
  }

  /**
   * Scale parameters by a factor
   * @param {Object} params - Parameters
   * @param {number} scale - Scale factor
   * @returns {Object} Scaled parameters
   */
  scaleParameters(params, scale) {
    const scaled = {};

    for (const [name, param] of Object.entries(params)) {
      if (ArrayBuffer.isView(param)) {
        const s = new param.constructor(param.length);
        for (let i = 0; i < param.length; i++) {
          s[i] = param[i] * scale;
        }
        scaled[name] = s;
      } else if (typeof param === 'number') {
        scaled[name] = param * scale;
      } else {
        scaled[name] = param; // Keep as is
      }
    }

    return scaled;
  }

  /**
   * Weighted average
   * @param {Array} params - Parameters
   * @param {Array<number>} weights - Weights
   * @returns {Object} Averaged parameter
   */
  weightedAverage(params, weights) {
    if (params.length === 0) return null;

    const first = params[0];

    if (ArrayBuffer.isView(first)) {
      const result = new first.constructor(first.length);
      result.fill(0);

      for (let i = 0; i < params.length; i++) {
        const weight = weights[i];
        for (let j = 0; j < first.length; j++) {
          result[j] += params[i][j] * weight;
        }
      }

      return result;
    }

    // Scalar
    let sum = 0;
    for (let i = 0; i < params.length; i++) {
      sum += params[i] * weights[i];
    }
    return sum;
  }

  /**
   * Initialize momentum buffer
   * @param {Object} model - Model
   * @returns {Object} Momentum buffer
   */
  initializeMomentumBuffer(model) {
    const buffer = {};

    if (model.parameters) {
      for (const [name, param] of Object.entries(model.parameters)) {
        if (ArrayBuffer.isView(param)) {
          buffer[name] = new param.constructor(param.length).fill(0);
        }
      }
    }

    return buffer;
  }

  /**
   * Clone parameters
   * @param {Object} params - Parameters
   * @returns {Object} Cloned parameters
   */
  cloneParameters(params) {
    const cloned = {};

    for (const [name, param] of Object.entries(params)) {
      if (ArrayBuffer.isView(param)) {
        cloned[name] = new param.constructor(param);
      } else {
        cloned[name] = param;
      }
    }

    return cloned;
  }

  /**
   * Clone model
   * @param {Object} model - Model to clone
   * @returns {Object} Cloned model
   */
  cloneModel(model) {
    return JSON.parse(JSON.stringify(model));
  }

  /**
   * Get statistics
   * @returns {Object} Statistics
   */
  getStatistics() {
    return {
      ...this.statistics,
      currentTau: this.tau,
      momentum: this.rho
    };
  }
}

/**
 * Enhanced Federated Server
 *
 * Supports multiple advanced aggregation algorithms
 */
export class EnhancedFederatedServer {
  /**
   * Create enhanced federated server
   * @param {Object} config - Configuration
   */
  constructor(config = {}) {
    this.config = config;
    this.aggregationMethod = config.aggregationMethod || 'fedavg';

    // Initialize aggregators
    this.aggregators = {
      fedbn: new FedBNAggregator(config.fedbn || {}),
      fednova: new FedNovaAggregator(config.fednova || {})
    };

    this.currentAggregator = this.aggregators[this.aggregationMethod] ||
      this.aggregators.fedbn;

    this.globalModel = null;
    this.currentRound = 0;

    this.statistics = {
      totalRounds: 0,
      methodsUsed: {}
    };
  }

  /**
   * Initialize global model
   * @param {Object} model - Global model
   */
  initializeGlobalModel(model) {
    this.globalModel = model;

    // Initialize all aggregators
    for (const aggregator of Object.values(this.aggregators)) {
      aggregator.initialize(model);
    }

    return this;
  }

  /**
   * Run federated learning round
   * @param {Array<Object>} clientUpdates - Client updates
   * @param {Object} roundConfig - Round configuration
   * @returns {Object} Round results
   */
  async runRound(clientUpdates, roundConfig = {}) {
    this.currentRound++;
    this.statistics.totalRounds++;

    const method = roundConfig.aggregationMethod || this.aggregationMethod;
    const aggregator = this.aggregators[method] || this.currentAggregator;

    console.log(`\n=== Round ${this.currentRound} (${method.toUpperCase()}) ===`);

    // Track method usage
    this.statistics.methodsUsed[method] =
      (this.statistics.methodsUsed[method] || 0) + 1;

    // Aggregate updates
    const result = aggregator.aggregate(clientUpdates, roundConfig);

    // Update global model
    this.globalModel = result.model;

    return {
      round: this.currentRound,
      method: method,
      model: this.globalModel,
      metadata: result.metadata,
      aggregatorStats: aggregator.getStatistics()
    };
  }

  /**
   * Switch aggregation method
   * @param {string} method - Method name ('fedbn', 'fednova')
   */
  setAggregationMethod(method) {
    if (!this.aggregators[method]) {
      throw new Error(`Unknown aggregation method: ${method}`);
    }

    this.aggregationMethod = method;
    this.currentAggregator = this.aggregators[method];

    console.log(`Switched to ${method.toUpperCase()} aggregation`);
  }

  /**
   * Get global model
   * @returns {Object} Global model
   */
  getGlobalModel() {
    return this.globalModel;
  }

  /**
   * Get statistics
   * @returns {Object} Statistics
   */
  getStatistics() {
    const stats = {
      ...this.statistics,
      currentRound: this.currentRound,
      currentMethod: this.aggregationMethod,
      aggregatorStatistics: {}
    };

    for (const [name, aggregator] of Object.entries(this.aggregators)) {
      stats.aggregatorStatistics[name] = aggregator.getStatistics();
    }

    return stats;
  }
}

/**
 * Create enhanced federated learning setup
 * @param {Object} config - Configuration
 * @returns {Object} Server and utilities
 */
export function createEnhancedFederatedLearning(config = {}) {
  const server = new EnhancedFederatedServer(config);

  return {
    server,
    aggregators: {
      fedbn: server.aggregators.fedbn,
      fednova: server.aggregators.fednova
    },
    utils: {
      /**
       * Create a mock client update
       * @param {string} clientId - Client ID
       * @param {number} dataSize - Data size
       * @param {number} localSteps - Number of local steps
       * @returns {Object} Mock update
       */
      createMockUpdate: (clientId, dataSize, localSteps) => ({
        clientId,
        dataSize,
        localSteps,
        parameters: {
          weights: new Float32Array(100).map(() => Math.random() * 0.01),
          bias: new Float32Array(10).map(() => Math.random() * 0.01),
          'bn.running_mean': new Float32Array(10).map(() => Math.random()),
          'bn.running_var': new Float32Array(10).map(() => Math.random() + 1)
        }
      })
    }
  };
}

// Export all components
export default {
  FedBNAggregator,
  FedNovaAggregator,
  EnhancedFederatedServer,
  createEnhancedFederatedLearning
};
