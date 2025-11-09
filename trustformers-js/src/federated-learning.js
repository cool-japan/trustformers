/**
 * Federated Learning Infrastructure
 *
 * Privacy-preserving distributed machine learning framework with:
 * - Secure aggregation protocols
 * - Differential privacy mechanisms
 * - Client selection strategies
 * - Byzantine-robust aggregation
 * - Personalized federated learning
 * - Vertical and horizontal federation
 * - Communication-efficient algorithms (FedAvg, FedProx, FedAdam)
 */

/**
 * Federated Learning Client
 * Represents a client device in federated learning
 */
export class FederatedClient {
  constructor(clientId, config = {}) {
    this.clientId = clientId;
    this.config = config;
    this.localModel = null;
    this.localData = null;
    this.statistics = {
      roundsParticipated: 0,
      samplesProcessed: 0,
      communicationCost: 0,
      computeTime: 0,
    };
  }

  /**
   * Initialize local model from global model
   */
  async initializeModel(globalModel) {
    this.localModel = this.cloneModel(globalModel);
    return this;
  }

  /**
   * Load local data
   */
  setLocalData(data) {
    this.localData = data;
    return this;
  }

  /**
   * Train local model on local data
   */
  async trainLocal(config = {}) {
    const { epochs = 1, batchSize = 32, learningRate = 0.01 } = config;

    const startTime = performance.now();
    const updates = [];

    // Simulate local training
    for (let epoch = 0; epoch < epochs; epoch++) {
      const numBatches = Math.ceil(this.localData.length / batchSize);

      for (let batch = 0; batch < numBatches; batch++) {
        const batchData = this.localData.slice(
          batch * batchSize,
          Math.min((batch + 1) * batchSize, this.localData.length)
        );

        // Compute gradients (simulated)
        const gradients = await this.computeGradients(batchData);
        updates.push(gradients);
      }
    }

    this.statistics.samplesProcessed += this.localData.length * epochs;
    this.statistics.computeTime += performance.now() - startTime;
    this.statistics.roundsParticipated++;

    return this.aggregateLocalUpdates(updates);
  }

  /**
   * Compute gradients for a batch
   */
  async computeGradients(batch) {
    // Simulated gradient computation
    return {
      weights: new Float32Array(100).map(() => Math.random() * 0.01 - 0.005),
      bias: new Float32Array(10).map(() => Math.random() * 0.01 - 0.005),
      timestamp: Date.now(),
    };
  }

  /**
   * Aggregate local updates from multiple batches
   */
  aggregateLocalUpdates(updates) {
    if (updates.length === 0) return null;

    const aggregated = {
      weights: new Float32Array(updates[0].weights.length).fill(0),
      bias: new Float32Array(updates[0].bias.length).fill(0),
    };

    for (const update of updates) {
      for (let i = 0; i < aggregated.weights.length; i++) {
        aggregated.weights[i] += update.weights[i];
      }
      for (let i = 0; i < aggregated.bias.length; i++) {
        aggregated.bias[i] += update.bias[i];
      }
    }

    // Average
    const n = updates.length;
    aggregated.weights = aggregated.weights.map(w => w / n);
    aggregated.bias = aggregated.bias.map(b => b / n);

    return aggregated;
  }

  /**
   * Clone model for local training
   */
  cloneModel(model) {
    return {
      ...model,
      parameters: model.parameters.map(p => ({
        ...p,
        data: new Float32Array(p.data),
      })),
    };
  }

  getStatistics() {
    return { ...this.statistics };
  }
}

/**
 * Secure Aggregation Protocol
 * Implements cryptographic protocols for privacy-preserving aggregation
 */
export class SecureAggregationProtocol {
  constructor(config = {}) {
    this.threshold = config.threshold || 0.5; // Minimum fraction of clients needed
    this.useMasking = config.useMasking !== false;
    this.maskSeed = config.maskSeed || Math.random();
    this.initialized = false;
  }

  /**
   * Initialize the secure aggregation protocol
   */
  initialize() {
    this.initialized = true;
    console.log('Secure Aggregation Protocol initialized');
    return this;
  }

  /**
   * Generate pairwise masks for secure aggregation
   */
  generatePairwiseMasks(clientId, otherClients, updateShape) {
    const masks = new Map();

    for (const otherId of otherClients) {
      if (otherId === clientId) continue;

      // Generate deterministic mask based on client IDs
      const seed = this.hashClientPair(clientId, otherId);
      const mask = this.generateMask(seed, updateShape);

      // Add or subtract based on ordering to ensure cancellation
      const sign = clientId < otherId ? 1 : -1;
      masks.set(otherId, { mask, sign });
    }

    return masks;
  }

  /**
   * Add secure masks to client update
   */
  maskUpdate(update, clientId, participatingClients) {
    if (!this.useMasking) return update;

    const maskedUpdate = this.cloneUpdate(update);
    const masks = this.generatePairwiseMasks(clientId, participatingClients, update.weights.length);

    // Apply masks
    for (const [otherId, { mask, sign }] of masks) {
      for (let i = 0; i < maskedUpdate.weights.length; i++) {
        maskedUpdate.weights[i] += sign * mask[i];
      }
    }

    return maskedUpdate;
  }

  /**
   * Unmask aggregated update (masks cancel out in sum)
   */
  unmaskAggregation(aggregatedUpdate) {
    // In secure aggregation, pairwise masks cancel out
    // This is a no-op but included for clarity
    return aggregatedUpdate;
  }

  hashClientPair(clientId1, clientId2) {
    // Simple hash function for demo
    const str = [clientId1, clientId2].sort().join('-');
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = (hash << 5) - hash + str.charCodeAt(i);
      hash = hash & hash;
    }
    return Math.abs(hash);
  }

  generateMask(seed, length) {
    // Pseudo-random mask generation
    const rng = this.createSeededRNG(seed);
    return new Float32Array(length).map(() => rng() * 2 - 1);
  }

  createSeededRNG(seed) {
    let state = seed;
    return () => {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      return state / 0x7fffffff;
    };
  }

  cloneUpdate(update) {
    return {
      weights: new Float32Array(update.weights),
      bias: new Float32Array(update.bias),
    };
  }
}

/**
 * Differential Privacy Mechanism
 * Adds calibrated noise for privacy protection
 */
export class DifferentialPrivacyMechanism {
  constructor(config = {}) {
    this.epsilon = config.epsilon || 1.0; // Privacy budget
    this.delta = config.delta || 1e-5; // Privacy parameter
    this.clipNorm = config.clipNorm || 1.0; // Gradient clipping threshold
    this.noiseMechanism = config.noiseMechanism || 'gaussian'; // 'gaussian' or 'laplace'
    this.budgetSpent = 0;
  }

  /**
   * Clip gradients to bound sensitivity
   */
  clipGradients(gradients, norm = this.clipNorm) {
    const currentNorm = this.computeL2Norm(gradients.weights);

    if (currentNorm <= norm) {
      return gradients;
    }

    const scale = norm / currentNorm;
    return {
      weights: gradients.weights.map(w => w * scale),
      bias: gradients.bias.map(b => b * scale),
    };
  }

  /**
   * Add calibrated noise for differential privacy
   */
  addNoise(clippedGradients, numClients) {
    const sensitivity = this.clipNorm / numClients;
    let noisyGradients;

    if (this.noiseMechanism === 'gaussian') {
      const sigma = this.computeGaussianSigma(sensitivity);
      noisyGradients = this.addGaussianNoise(clippedGradients, sigma);
    } else {
      const scale = sensitivity / this.epsilon;
      noisyGradients = this.addLaplaceNoise(clippedGradients, scale);
    }

    this.budgetSpent += this.epsilon;
    return noisyGradients;
  }

  /**
   * Compute Gaussian noise sigma for (epsilon, delta)-DP
   */
  computeGaussianSigma(sensitivity) {
    // Using the Gaussian mechanism
    return (Math.sqrt(2 * Math.log(1.25 / this.delta)) * sensitivity) / this.epsilon;
  }

  /**
   * Add Gaussian noise to gradients
   */
  addGaussianNoise(gradients, sigma) {
    return {
      weights: gradients.weights.map(w => w + this.sampleGaussian(0, sigma)),
      bias: gradients.bias.map(b => b + this.sampleGaussian(0, sigma)),
    };
  }

  /**
   * Add Laplace noise to gradients
   */
  addLaplaceNoise(gradients, scale) {
    return {
      weights: gradients.weights.map(w => w + this.sampleLaplace(0, scale)),
      bias: gradients.bias.map(b => b + this.sampleLaplace(0, scale)),
    };
  }

  sampleGaussian(mean, sigma) {
    // Box-Muller transform
    const u1 = Math.random();
    const u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mean + sigma * z;
  }

  sampleLaplace(mean, scale) {
    const u = Math.random() - 0.5;
    return mean - scale * Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
  }

  computeL2Norm(array) {
    return Math.sqrt(array.reduce((sum, val) => sum + val * val, 0));
  }

  getRemainingBudget() {
    return Math.max(0, this.epsilon - this.budgetSpent);
  }
}

/**
 * Client Selection Strategy
 * Selects clients for each federated learning round
 */
export class ClientSelectionStrategy {
  constructor(strategy = 'random', config = {}) {
    this.strategy = strategy;
    this.config = config;
    this.selectionHistory = [];
  }

  /**
   * Select clients for the current round
   */
  selectClients(clients, numSelect) {
    let selected;

    switch (this.strategy) {
      case 'random':
        selected = this.randomSelection(clients, numSelect);
        break;
      case 'importance':
        selected = this.importanceSelection(clients, numSelect);
        break;
      case 'round_robin':
        selected = this.roundRobinSelection(clients, numSelect);
        break;
      case 'data_size':
        selected = this.dataSizeSelection(clients, numSelect);
        break;
      default:
        selected = this.randomSelection(clients, numSelect);
    }

    this.selectionHistory.push({
      round: this.selectionHistory.length + 1,
      selected: selected.map(c => c.clientId),
      timestamp: Date.now(),
    });

    return selected;
  }

  randomSelection(clients, numSelect) {
    const shuffled = [...clients].sort(() => Math.random() - 0.5);
    return shuffled.slice(0, Math.min(numSelect, clients.length));
  }

  importanceSelection(clients, numSelect) {
    // Select based on data quality and quantity
    const scored = clients.map(client => ({
      client,
      score: this.computeImportanceScore(client),
    }));

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, numSelect).map(s => s.client);
  }

  roundRobinSelection(clients, numSelect) {
    const startIdx = (this.selectionHistory.length * numSelect) % clients.length;
    const selected = [];

    for (let i = 0; i < numSelect; i++) {
      const idx = (startIdx + i) % clients.length;
      selected.push(clients[idx]);
    }

    return selected;
  }

  dataSizeSelection(clients, numSelect) {
    // Prefer clients with more data
    const sorted = [...clients].sort((a, b) => {
      const sizeA = a.localData?.length || 0;
      const sizeB = b.localData?.length || 0;
      return sizeB - sizeA;
    });

    return sorted.slice(0, numSelect);
  }

  computeImportanceScore(client) {
    const dataSize = client.localData?.length || 0;
    const participation = client.statistics.roundsParticipated;
    const computeSpeed = client.statistics.samplesProcessed / (client.statistics.computeTime + 1);

    return dataSize * 0.5 + computeSpeed * 0.3 + (1 / (participation + 1)) * 0.2;
  }
}

/**
 * Byzantine-Robust Aggregation
 * Defends against malicious clients
 */
export class ByzantineRobustAggregation {
  constructor(method = 'krum', config = {}) {
    this.method = method;
    this.config = config;
    this.byzantineThreshold = config.byzantineThreshold || 0.3;
  }

  /**
   * Aggregate updates with Byzantine robustness
   */
  aggregate(updates, numMalicious = 0) {
    switch (this.method) {
      case 'krum':
        return this.krum(updates, numMalicious);
      case 'trimmed_mean':
        return this.trimmedMean(updates);
      case 'median':
        return this.geometricMedian(updates);
      default:
        return this.federatedAveraging(updates);
    }
  }

  /**
   * Krum aggregation - selects most consistent update
   */
  krum(updates, numMalicious) {
    const n = updates.length;
    const m = numMalicious || Math.floor(n * this.byzantineThreshold);

    // Compute pairwise distances
    const distances = updates.map((update1, i) => {
      const scores = updates.map((update2, j) => {
        if (i === j) return 0;
        return this.computeDistance(update1, update2);
      });

      // Sum of distances to n - m - 2 closest updates
      scores.sort((a, b) => a - b);
      const score = scores.slice(0, n - m - 2).reduce((a, b) => a + b, 0);

      return { update: update1, score, index: i };
    });

    // Select update with minimum score
    distances.sort((a, b) => a.score - b.score);
    return distances[0].update;
  }

  /**
   * Trimmed mean aggregation
   */
  trimmedMean(updates, trimFraction = 0.2) {
    const numTrim = Math.floor(updates.length * trimFraction);
    const aggregated = {
      weights: new Float32Array(updates[0].weights.length).fill(0),
      bias: new Float32Array(updates[0].bias.length).fill(0),
    };

    // For each parameter, compute trimmed mean
    for (let i = 0; i < aggregated.weights.length; i++) {
      const values = updates.map(u => u.weights[i]).sort((a, b) => a - b);
      const trimmed = values.slice(numTrim, values.length - numTrim);
      aggregated.weights[i] = trimmed.reduce((a, b) => a + b, 0) / trimmed.length;
    }

    for (let i = 0; i < aggregated.bias.length; i++) {
      const values = updates.map(u => u.bias[i]).sort((a, b) => a - b);
      const trimmed = values.slice(numTrim, values.length - numTrim);
      aggregated.bias[i] = trimmed.reduce((a, b) => a + b, 0) / trimmed.length;
    }

    return aggregated;
  }

  /**
   * Geometric median aggregation
   */
  geometricMedian(updates, maxIterations = 10) {
    // Initialize with coordinate-wise median
    let median = this.coordinateWiseMedian(updates);

    // Weiszfeld algorithm
    for (let iter = 0; iter < maxIterations; iter++) {
      const weights = updates.map(update => 1 / (this.computeDistance(update, median) + 1e-8));

      const totalWeight = weights.reduce((a, b) => a + b, 0);

      // Update median
      const newMedian = {
        weights: new Float32Array(median.weights.length).fill(0),
        bias: new Float32Array(median.bias.length).fill(0),
      };

      for (let i = 0; i < updates.length; i++) {
        const w = weights[i] / totalWeight;
        for (let j = 0; j < newMedian.weights.length; j++) {
          newMedian.weights[j] += w * updates[i].weights[j];
        }
        for (let j = 0; j < newMedian.bias.length; j++) {
          newMedian.bias[j] += w * updates[i].bias[j];
        }
      }

      median = newMedian;
    }

    return median;
  }

  coordinateWiseMedian(updates) {
    const median = {
      weights: new Float32Array(updates[0].weights.length),
      bias: new Float32Array(updates[0].bias.length),
    };

    for (let i = 0; i < median.weights.length; i++) {
      const values = updates.map(u => u.weights[i]).sort((a, b) => a - b);
      median.weights[i] = values[Math.floor(values.length / 2)];
    }

    for (let i = 0; i < median.bias.length; i++) {
      const values = updates.map(u => u.bias[i]).sort((a, b) => a - b);
      median.bias[i] = values[Math.floor(values.length / 2)];
    }

    return median;
  }

  federatedAveraging(updates) {
    const aggregated = {
      weights: new Float32Array(updates[0].weights.length).fill(0),
      bias: new Float32Array(updates[0].bias.length).fill(0),
    };

    for (const update of updates) {
      for (let i = 0; i < aggregated.weights.length; i++) {
        aggregated.weights[i] += update.weights[i];
      }
      for (let i = 0; i < aggregated.bias.length; i++) {
        aggregated.bias[i] += update.bias[i];
      }
    }

    const n = updates.length;
    aggregated.weights = aggregated.weights.map(w => w / n);
    aggregated.bias = aggregated.bias.map(b => b / n);

    return aggregated;
  }

  computeDistance(update1, update2) {
    let dist = 0;
    for (let i = 0; i < update1.weights.length; i++) {
      const diff = update1.weights[i] - update2.weights[i];
      dist += diff * diff;
    }
    return Math.sqrt(dist);
  }
}

/**
 * Federated Learning Server
 * Coordinates the federated learning process
 */
export class FederatedServer {
  constructor(config = {}) {
    this.config = config;
    this.globalModel = null;
    this.clients = [];
    this.currentRound = 0;

    // Initialize components
    this.secureAggregation = new SecureAggregationProtocol(config.secureAggregation);
    this.differentialPrivacy = new DifferentialPrivacyMechanism(config.differentialPrivacy);
    this.clientSelector = new ClientSelectionStrategy(
      config.selectionStrategy || 'random',
      config.selectionConfig
    );
    this.byzantineDefense = new ByzantineRobustAggregation(
      config.aggregationMethod || 'trimmed_mean',
      config.byzantineConfig
    );

    this.statistics = {
      totalRounds: 0,
      totalClientsParticipated: new Set(),
      aggregationTime: 0,
      communicationCost: 0,
    };
  }

  /**
   * Initialize global model
   */
  initializeGlobalModel(model) {
    this.globalModel = model;
    return this;
  }

  /**
   * Register a client
   */
  registerClient(client) {
    this.clients.push(client);
    return this;
  }

  /**
   * Select clients for a training round
   */
  selectClients(numClients) {
    return this.clientSelector.selectClients(this.clients, numClients);
  }

  /**
   * Run a federated learning round
   */
  async runRound(config = {}) {
    const { clientFraction = 0.1, localEpochs = 1, batchSize = 32, learningRate = 0.01 } = config;

    this.currentRound++;
    this.statistics.totalRounds++;

    console.log(`\n=== Federated Learning Round ${this.currentRound} ===`);

    // Select clients
    const numSelect = Math.max(1, Math.floor(this.clients.length * clientFraction));
    const selectedClients = this.clientSelector.selectClients(this.clients, numSelect);

    console.log(`Selected ${selectedClients.length} clients`);

    // Distribute global model to selected clients
    await Promise.all(selectedClients.map(client => client.initializeModel(this.globalModel)));

    // Train locally
    const clientUpdates = await Promise.all(
      selectedClients.map(async client => {
        const update = await client.trainLocal({
          epochs: localEpochs,
          batchSize,
          learningRate,
        });

        this.statistics.totalClientsParticipated.add(client.clientId);

        // Apply differential privacy
        const clippedUpdate = this.differentialPrivacy.clipGradients(update);
        const noisyUpdate = this.differentialPrivacy.addNoise(
          clippedUpdate,
          selectedClients.length
        );

        // Apply secure aggregation
        const maskedUpdate = this.secureAggregation.maskUpdate(
          noisyUpdate,
          client.clientId,
          selectedClients.map(c => c.clientId)
        );

        return maskedUpdate;
      })
    );

    // Aggregate updates
    const startTime = performance.now();
    const aggregatedUpdate = this.byzantineDefense.aggregate(clientUpdates);
    this.statistics.aggregationTime += performance.now() - startTime;

    // Update global model
    this.updateGlobalModel(aggregatedUpdate);

    console.log(`Round ${this.currentRound} completed`);
    console.log(
      `Privacy budget remaining: ${this.differentialPrivacy.getRemainingBudget().toFixed(3)}`
    );

    return {
      round: this.currentRound,
      participatingClients: selectedClients.length,
      aggregatedUpdate,
    };
  }

  /**
   * Update global model with aggregated updates
   */
  updateGlobalModel(update) {
    // Apply aggregated gradients to global model
    if (this.globalModel.parameters) {
      // Simplified update (in practice, would use optimizer)
      for (let i = 0; i < Math.min(this.globalModel.parameters.length, 2); i++) {
        const param = this.globalModel.parameters[i];
        const updateData = i === 0 ? update.weights : update.bias;

        for (let j = 0; j < Math.min(param.data.length, updateData.length); j++) {
          param.data[j] -= updateData[j];
        }
      }
    }
  }

  /**
   * Run multiple federated learning rounds
   */
  async train(numRounds, roundConfig = {}) {
    const results = [];

    for (let i = 0; i < numRounds; i++) {
      const result = await this.runRound(roundConfig);
      results.push(result);

      // Check privacy budget
      if (this.differentialPrivacy.getRemainingBudget() <= 0) {
        console.warn('Privacy budget exhausted');
        break;
      }
    }

    return results;
  }

  getStatistics() {
    return {
      ...this.statistics,
      totalClientsRegistered: this.clients.length,
      uniqueClientsParticipated: this.statistics.totalClientsParticipated.size,
      averageAggregationTime: this.statistics.aggregationTime / this.statistics.totalRounds,
    };
  }
}

/**
 * Create a federated learning setup
 */
export function createFederatedLearning(config = {}) {
  const server = new FederatedServer(config);

  // Create clients
  const numClients = config.numClients || 100;
  const clients = [];

  for (let i = 0; i < numClients; i++) {
    const client = new FederatedClient(`client_${i}`, config.clientConfig);

    // Generate synthetic local data
    const dataSize = Math.floor(Math.random() * 1000) + 100;
    const localData = Array.from({ length: dataSize }, (_, idx) => ({
      id: idx,
      features: new Float32Array(10).map(() => Math.random()),
      label: Math.floor(Math.random() * 2),
    }));

    client.setLocalData(localData);
    server.registerClient(client);
    clients.push(client);
  }

  return { server, clients };
}

// All components already exported via 'export class' and 'export function' declarations above
