/**
 * Real-Time Collaboration Features
 *
 * Enables multiple users to collaborate on ML model development in real-time.
 * Features include:
 * - Collaborative model training (multiple users training the same model)
 * - Shared inference sessions (collaborative predictions)
 * - Real-time model synchronization (WebSocket/WebRTC based)
 * - Operational transformation for conflict resolution
 * - Presence awareness (see who's working on what)
 * - Collaborative experimentation (A/B testing, hyperparameter tuning)
 * - Shared workspace (datasets, models, results)
 * - Real-time metrics dashboard
 *
 * @module realtime-collaboration
 */

/**
 * Collaborative Session
 *
 * Manages a real-time collaboration session for ML development
 */
export class CollaborativeSession {
  /**
   * Create a collaborative session
   * @param {Object} config - Session configuration
   */
  constructor(config = {}) {
    this.sessionId = config.sessionId || this.generateSessionId();
    this.userId = config.userId || this.generateUserId();
    this.userName = config.userName || 'Anonymous';

    // Connection settings
    this.serverUrl = config.serverUrl || 'ws://localhost:8080';
    this.useWebRTC = config.useWebRTC || false;

    // Session state
    this.connection = null;
    this.peers = new Map(); // Map of userId -> peerInfo
    this.sharedModel = null;
    this.sharedDatasets = new Map();
    this.sharedExperiments = new Map();

    // Operational transformation state
    this.localVersion = 0;
    this.remoteVersion = 0;
    this.pendingOperations = [];

    // Callbacks
    this.eventHandlers = {
      connected: [],
      disconnected: [],
      peerJoined: [],
      peerLeft: [],
      modelUpdated: [],
      experimentUpdated: [],
      messageReceived: []
    };

    // Statistics
    this.statistics = {
      messagesSent: 0,
      messagesReceived: 0,
      modelSyncs: 0,
      totalCollaborationTime: 0,
      peersEncountered: new Set()
    };

    this.startTime = Date.now();
  }

  /**
   * Connect to collaboration server
   * @returns {Promise<void>}
   */
  async connect() {
    console.log(`Connecting to collaboration server: ${this.serverUrl}`);

    if (this.useWebRTC) {
      await this.connectWebRTC();
    } else {
      await this.connectWebSocket();
    }

    // Send join message
    await this.sendMessage({
      type: 'join',
      sessionId: this.sessionId,
      userId: this.userId,
      userName: this.userName,
      timestamp: Date.now()
    });

    this.emit('connected', { sessionId: this.sessionId });
  }

  /**
   * Connect via WebSocket
   * @returns {Promise<void>}
   */
  async connectWebSocket() {
    return new Promise((resolve, reject) => {
      try {
        this.connection = new WebSocket(this.serverUrl);

        this.connection.onopen = () => {
          console.log('WebSocket connected');
          resolve();
        };

        this.connection.onmessage = (event) => {
          this.handleMessage(JSON.parse(event.data));
        };

        this.connection.onerror = (error) => {
          console.error('WebSocket error:', error);
          reject(error);
        };

        this.connection.onclose = () => {
          console.log('WebSocket disconnected');
          this.emit('disconnected', {});
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Connect via WebRTC (peer-to-peer)
   * @returns {Promise<void>}
   */
  async connectWebRTC() {
    // Simplified WebRTC implementation
    // In production, would use a signaling server

    console.log('WebRTC not fully implemented, falling back to WebSocket');
    await this.connectWebSocket();
  }

  /**
   * Disconnect from session
   */
  async disconnect() {
    // Send leave message
    await this.sendMessage({
      type: 'leave',
      sessionId: this.sessionId,
      userId: this.userId,
      timestamp: Date.now()
    });

    if (this.connection) {
      this.connection.close();
      this.connection = null;
    }

    this.statistics.totalCollaborationTime = Date.now() - this.startTime;
  }

  /**
   * Send message to all peers
   * @param {Object} message - Message to send
   * @returns {Promise<void>}
   */
  async sendMessage(message) {
    if (!this.connection || this.connection.readyState !== WebSocket.OPEN) {
      throw new Error('Not connected to collaboration server');
    }

    const envelope = {
      ...message,
      senderId: this.userId,
      senderName: this.userName,
      version: this.localVersion++,
      timestamp: Date.now()
    };

    this.connection.send(JSON.stringify(envelope));
    this.statistics.messagesSent++;
  }

  /**
   * Handle incoming message
   * @param {Object} message - Received message
   */
  handleMessage(message) {
    this.statistics.messagesReceived++;

    // Update remote version for OT
    if (message.version !== undefined) {
      this.remoteVersion = Math.max(this.remoteVersion, message.version);
    }

    switch (message.type) {
      case 'join':
        this.handlePeerJoined(message);
        break;

      case 'leave':
        this.handlePeerLeft(message);
        break;

      case 'modelUpdate':
        this.handleModelUpdate(message);
        break;

      case 'experimentUpdate':
        this.handleExperimentUpdate(message);
        break;

      case 'datasetShare':
        this.handleDatasetShare(message);
        break;

      case 'operation':
        this.handleOperation(message);
        break;

      case 'presence':
        this.handlePresenceUpdate(message);
        break;

      default:
        this.emit('messageReceived', message);
    }
  }

  /**
   * Handle peer joined
   * @param {Object} message - Join message
   */
  handlePeerJoined(message) {
    if (message.senderId === this.userId) return;

    this.peers.set(message.senderId, {
      userId: message.senderId,
      userName: message.senderName,
      joinedAt: message.timestamp,
      presence: 'active'
    });

    this.statistics.peersEncountered.add(message.senderId);

    console.log(`Peer joined: ${message.senderName} (${message.senderId})`);
    this.emit('peerJoined', this.peers.get(message.senderId));
  }

  /**
   * Handle peer left
   * @param {Object} message - Leave message
   */
  handlePeerLeft(message) {
    if (this.peers.has(message.senderId)) {
      const peer = this.peers.get(message.senderId);
      this.peers.delete(message.senderId);

      console.log(`Peer left: ${peer.userName} (${peer.userId})`);
      this.emit('peerLeft', peer);
    }
  }

  /**
   * Handle model update from peer
   * @param {Object} message - Model update message
   */
  handleModelUpdate(message) {
    const { modelDelta, operation, metadata } = message;

    // Apply operational transformation
    const transformedDelta = this.applyOT(modelDelta, operation);

    // Merge with local model
    this.sharedModel = this.mergeModelUpdate(this.sharedModel, transformedDelta);

    this.statistics.modelSyncs++;

    console.log(`Model updated by ${message.senderName}`);
    this.emit('modelUpdated', {
      model: this.sharedModel,
      updatedBy: message.senderId,
      metadata
    });
  }

  /**
   * Handle experiment update
   * @param {Object} message - Experiment update message
   */
  handleExperimentUpdate(message) {
    const { experimentId, update } = message;

    if (!this.sharedExperiments.has(experimentId)) {
      this.sharedExperiments.set(experimentId, {
        id: experimentId,
        createdBy: message.senderId,
        createdAt: message.timestamp,
        results: []
      });
    }

    const experiment = this.sharedExperiments.get(experimentId);

    if (update.type === 'result') {
      experiment.results.push({
        ...update.result,
        submittedBy: message.senderId,
        submittedAt: message.timestamp
      });
    } else if (update.type === 'config') {
      experiment.config = update.config;
    }

    console.log(`Experiment updated: ${experimentId}`);
    this.emit('experimentUpdated', experiment);
  }

  /**
   * Handle dataset share
   * @param {Object} message - Dataset share message
   */
  handleDatasetShare(message) {
    const { datasetId, dataset, metadata } = message;

    this.sharedDatasets.set(datasetId, {
      id: datasetId,
      data: dataset,
      metadata: metadata || {},
      sharedBy: message.senderId,
      sharedAt: message.timestamp
    });

    console.log(`Dataset shared: ${datasetId} by ${message.senderName}`);
  }

  /**
   * Handle generic operation
   * @param {Object} message - Operation message
   */
  handleOperation(message) {
    const { operation } = message;

    // Queue for operational transformation
    this.pendingOperations.push({
      ...operation,
      receivedAt: Date.now(),
      senderId: message.senderId
    });

    // Process pending operations
    this.processPendingOperations();
  }

  /**
   * Handle presence update
   * @param {Object} message - Presence update message
   */
  handlePresenceUpdate(message) {
    const { presence } = message;

    if (this.peers.has(message.senderId)) {
      const peer = this.peers.get(message.senderId);
      peer.presence = presence;
      peer.lastActivity = message.timestamp;
    }
  }

  /**
   * Share model update
   * @param {Object} modelDelta - Model changes
   * @param {Object} metadata - Update metadata
   * @returns {Promise<void>}
   */
  async shareModelUpdate(modelDelta, metadata = {}) {
    await this.sendMessage({
      type: 'modelUpdate',
      modelDelta,
      operation: {
        type: 'modelUpdate',
        version: this.localVersion,
        basedOn: this.remoteVersion
      },
      metadata
    });
  }

  /**
   * Share experiment result
   * @param {string} experimentId - Experiment ID
   * @param {Object} result - Experiment result
   * @returns {Promise<void>}
   */
  async shareExperimentResult(experimentId, result) {
    await this.sendMessage({
      type: 'experimentUpdate',
      experimentId,
      update: {
        type: 'result',
        result
      }
    });
  }

  /**
   * Share dataset
   * @param {string} datasetId - Dataset ID
   * @param {Object} dataset - Dataset
   * @param {Object} metadata - Dataset metadata
   * @returns {Promise<void>}
   */
  async shareDataset(datasetId, dataset, metadata = {}) {
    await this.sendMessage({
      type: 'datasetShare',
      datasetId,
      dataset,
      metadata
    });
  }

  /**
   * Update presence status
   * @param {string} presence - Presence status ('active', 'idle', 'away')
   * @returns {Promise<void>}
   */
  async updatePresence(presence) {
    await this.sendMessage({
      type: 'presence',
      presence
    });
  }

  /**
   * Apply operational transformation
   * @param {Object} delta - Model delta
   * @param {Object} operation - Operation metadata
   * @returns {Object} Transformed delta
   */
  applyOT(delta, operation) {
    // Simplified OT - in production would use full OT algorithm

    if (operation.basedOn < this.localVersion) {
      // Need to transform against local operations
      const localOps = this.pendingOperations.filter(
        op => op.version > operation.basedOn && op.version <= this.localVersion
      );

      let transformed = delta;
      for (const localOp of localOps) {
        transformed = this.transformOperation(transformed, localOp);
      }

      return transformed;
    }

    return delta;
  }

  /**
   * Transform operation
   * @param {Object} delta - Delta to transform
   * @param {Object} operation - Local operation
   * @returns {Object} Transformed delta
   */
  transformOperation(delta, _operation) {
    // Simplified transformation
    // In production, would implement full OT algorithms (e.g., Google Wave OT)

    return delta;
  }

  /**
   * Merge model update
   * @param {Object} currentModel - Current model
   * @param {Object} delta - Model delta
   * @returns {Object} Merged model
   */
  mergeModelUpdate(currentModel, delta) {
    if (!currentModel) return delta;

    // Deep merge
    const merged = JSON.parse(JSON.stringify(currentModel));

    for (const [key, value] of Object.entries(delta)) {
      if (typeof value === 'object' && !Array.isArray(value)) {
        merged[key] = this.mergeModelUpdate(merged[key] || {}, value);
      } else {
        merged[key] = value;
      }
    }

    return merged;
  }

  /**
   * Process pending operations
   */
  processPendingOperations() {
    // Sort by version
    this.pendingOperations.sort((a, b) => a.version - b.version);

    // Process operations that can be applied
    while (this.pendingOperations.length > 0) {
      const [op] = this.pendingOperations;

      if (op.version <= this.remoteVersion) {
        // Already processed
        this.pendingOperations.shift();
      } else if (op.version === this.remoteVersion + 1) {
        // Can apply
        this.applyOperation(op);
        this.remoteVersion = op.version;
        this.pendingOperations.shift();
      } else {
        // Need to wait for earlier operations
        break;
      }
    }
  }

  /**
   * Apply operation
   * @param {Object} operation - Operation to apply
   */
  applyOperation(operation) {
    // Apply operation to local state
    console.log(`Applying operation from ${operation.senderId}`);
  }

  /**
   * Register event handler
   * @param {string} event - Event name
   * @param {Function} handler - Event handler
   */
  on(event, handler) {
    if (this.eventHandlers[event]) {
      this.eventHandlers[event].push(handler);
    }
  }

  /**
   * Emit event
   * @param {string} event - Event name
   * @param {*} data - Event data
   */
  emit(event, data) {
    if (this.eventHandlers[event]) {
      for (const handler of this.eventHandlers[event]) {
        try {
          handler(data);
        } catch (error) {
          console.error(`Error in event handler for ${event}:`, error);
        }
      }
    }
  }

  /**
   * Get active peers
   * @returns {Array<Object>} List of active peers
   */
  getActivePeers() {
    return Array.from(this.peers.values()).filter(
      peer => peer.presence === 'active'
    );
  }

  /**
   * Get shared model
   * @returns {Object} Shared model
   */
  getSharedModel() {
    return this.sharedModel;
  }

  /**
   * Get shared experiments
   * @returns {Array<Object>} List of shared experiments
   */
  getSharedExperiments() {
    return Array.from(this.sharedExperiments.values());
  }

  /**
   * Get shared datasets
   * @returns {Array<Object>} List of shared datasets
   */
  getSharedDatasets() {
    return Array.from(this.sharedDatasets.values());
  }

  /**
   * Get statistics
   * @returns {Object} Session statistics
   */
  getStatistics() {
    return {
      ...this.statistics,
      activePeers: this.getActivePeers().length,
      totalPeers: this.peers.size,
      uniquePeersEncountered: this.statistics.peersEncountered.size,
      sessionDuration: Date.now() - this.startTime,
      operationsPending: this.pendingOperations.length
    };
  }

  /**
   * Generate session ID
   * @returns {string} Session ID
   */
  generateSessionId() {
    return `session_${Date.now()}_${Math.random().toString(36).substring(7)}`;
  }

  /**
   * Generate user ID
   * @returns {string} User ID
   */
  generateUserId() {
    return `user_${Date.now()}_${Math.random().toString(36).substring(7)}`;
  }
}

/**
 * Collaborative Experiment
 *
 * Manages collaborative hyperparameter tuning and A/B testing
 */
export class CollaborativeExperiment {
  /**
   * Create collaborative experiment
   * @param {Object} config - Experiment configuration
   */
  constructor(config = {}) {
    this.experimentId = config.experimentId || this.generateExperimentId();
    this.name = config.name || 'Untitled Experiment';
    this.description = config.description || '';
    this.createdBy = config.createdBy;
    this.createdAt = Date.now();

    // Experiment configuration
    this.searchSpace = config.searchSpace || {};
    this.metric = config.metric || 'accuracy';
    this.goal = config.goal || 'maximize'; // 'maximize' or 'minimize'

    // Results from all collaborators
    this.results = [];
    this.bestResult = null;

    // Collaboration session
    this.session = config.session;

    // Statistics
    this.statistics = {
      totalTrials: 0,
      contributors: new Set(),
      startTime: Date.now()
    };
  }

  /**
   * Submit experiment result
   * @param {Object} config - Configuration used
   * @param {Object} metrics - Metrics achieved
   * @param {string} contributorId - Contributor ID
   * @returns {Promise<void>}
   */
  async submitResult(config, metrics, contributorId) {
    const result = {
      id: this.generateResultId(),
      config,
      metrics,
      contributorId,
      submittedAt: Date.now()
    };

    this.results.push(result);
    this.statistics.totalTrials++;
    this.statistics.contributors.add(contributorId);

    // Update best result
    const metricValue = metrics[this.metric];
    if (!this.bestResult ||
        (this.goal === 'maximize' && metricValue > this.bestResult.metrics[this.metric]) ||
        (this.goal === 'minimize' && metricValue < this.bestResult.metrics[this.metric])) {
      this.bestResult = result;
    }

    // Share with collaborators
    if (this.session) {
      await this.session.shareExperimentResult(this.experimentId, result);
    }

    return result;
  }

  /**
   * Get next configuration to try (Bayesian optimization)
   * @returns {Object} Suggested configuration
   */
  suggestConfiguration() {
    // Simplified suggestion - in production would use Bayesian optimization

    if (this.results.length === 0) {
      // Random sampling for first trial
      return this.randomSample(this.searchSpace);
    }

    // Exploit best result with some exploration
    const exploitRatio = 0.8;

    if (Math.random() < exploitRatio && this.bestResult) {
      // Explore around best result
      return this.perturbConfig(this.bestResult.config);
    } 
      // Random exploration
      return this.randomSample(this.searchSpace);
    
  }

  /**
   * Random sample from search space
   * @param {Object} searchSpace - Search space
   * @returns {Object} Sampled configuration
   */
  randomSample(searchSpace) {
    const config = {};

    for (const [param, space] of Object.entries(searchSpace)) {
      if (space.type === 'continuous') {
        config[param] = (Math.random() * (space.max - space.min)) + space.min;
      } else if (space.type === 'discrete') {
        config[param] = space.values[Math.floor(Math.random() * space.values.length)];
      } else if (space.type === 'integer') {
        config[param] = Math.floor(Math.random() * (space.max - space.min + 1)) + space.min;
      }
    }

    return config;
  }

  /**
   * Perturb configuration
   * @param {Object} config - Base configuration
   * @returns {Object} Perturbed configuration
   */
  perturbConfig(config) {
    const perturbed = { ...config };

    for (const [param, space] of Object.entries(this.searchSpace)) {
      if (Math.random() < 0.3) { // Perturb 30% of parameters
        if (space.type === 'continuous') {
          const range = space.max - space.min;
          const noise = (Math.random() - 0.5) * range * 0.2;
          perturbed[param] = Math.max(space.min, Math.min(space.max, config[param] + noise));
        } else if (space.type === 'discrete') {
          perturbed[param] = space.values[Math.floor(Math.random() * space.values.length)];
        } else if (space.type === 'integer') {
          const noise = Math.floor((Math.random() - 0.5) * 5);
          perturbed[param] = Math.max(space.min, Math.min(space.max, config[param] + noise));
        }
      }
    }

    return perturbed;
  }

  /**
   * Get experiment summary
   * @returns {Object} Experiment summary
   */
  getSummary() {
    return {
      experimentId: this.experimentId,
      name: this.name,
      description: this.description,
      bestResult: this.bestResult,
      totalTrials: this.statistics.totalTrials,
      contributors: Array.from(this.statistics.contributors),
      duration: Date.now() - this.statistics.startTime,
      resultsDistribution: this.getResultsDistribution()
    };
  }

  /**
   * Get results distribution
   * @returns {Object} Distribution statistics
   */
  getResultsDistribution() {
    if (this.results.length === 0) return null;

    const metricValues = this.results.map(r => r.metrics[this.metric]);

    return {
      mean: metricValues.reduce((a, b) => a + b, 0) / metricValues.length,
      min: Math.min(...metricValues),
      max: Math.max(...metricValues),
      std: this.calculateStd(metricValues)
    };
  }

  /**
   * Calculate standard deviation
   * @param {Array<number>} values - Values
   * @returns {number} Standard deviation
   */
  calculateStd(values) {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const squaredDiffs = values.map(v => Math.pow(v - mean, 2));
    const variance = squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
    return Math.sqrt(variance);
  }

  /**
   * Generate experiment ID
   * @returns {string} Experiment ID
   */
  generateExperimentId() {
    return `exp_${Date.now()}_${Math.random().toString(36).substring(7)}`;
  }

  /**
   * Generate result ID
   * @returns {string} Result ID
   */
  generateResultId() {
    return `result_${Date.now()}_${Math.random().toString(36).substring(7)}`;
  }
}

/**
 * Collaborative Metrics Dashboard
 *
 * Real-time metrics visualization and sharing
 */
export class CollaborativeMetricsDashboard {
  /**
   * Create metrics dashboard
   * @param {Object} config - Dashboard configuration
   */
  constructor(config = {}) {
    this.session = config.session;
    this.metrics = new Map();
    this.updateInterval = config.updateInterval || 1000; // 1 second
    this.updateTimer = null;
  }

  /**
   * Start dashboard
   */
  start() {
    this.updateTimer = setInterval(() => {
      this.broadcastMetrics();
    }, this.updateInterval);
  }

  /**
   * Stop dashboard
   */
  stop() {
    if (this.updateTimer) {
      clearInterval(this.updateTimer);
      this.updateTimer = null;
    }
  }

  /**
   * Update metric
   * @param {string} name - Metric name
   * @param {number} value - Metric value
   * @param {Object} metadata - Metric metadata
   */
  updateMetric(name, value, metadata = {}) {
    this.metrics.set(name, {
      name,
      value,
      metadata,
      updatedAt: Date.now()
    });
  }

  /**
   * Broadcast metrics to collaborators
   * @returns {Promise<void>}
   */
  async broadcastMetrics() {
    if (!this.session) return;

    const metricsSnapshot = Object.fromEntries(this.metrics);

    await this.session.sendMessage({
      type: 'metricsUpdate',
      metrics: metricsSnapshot
    });
  }

  /**
   * Get all metrics
   * @returns {Object} All metrics
   */
  getMetrics() {
    return Object.fromEntries(this.metrics);
  }
}

/**
 * Create collaborative session
 * @param {Object} config - Session configuration
 * @returns {CollaborativeSession} Collaborative session
 */
export function createCollaborativeSession(config = {}) {
  return new CollaborativeSession(config);
}

/**
 * Create collaborative experiment
 * @param {Object} config - Experiment configuration
 * @returns {CollaborativeExperiment} Collaborative experiment
 */
export function createCollaborativeExperiment(config = {}) {
  return new CollaborativeExperiment(config);
}

/**
 * Create metrics dashboard
 * @param {Object} config - Dashboard configuration
 * @returns {CollaborativeMetricsDashboard} Metrics dashboard
 */
export function createMetricsDashboard(config = {}) {
  return new CollaborativeMetricsDashboard(config);
}

// Export all components
export default {
  CollaborativeSession,
  CollaborativeExperiment,
  CollaborativeMetricsDashboard,
  createCollaborativeSession,
  createCollaborativeExperiment,
  createMetricsDashboard
};
