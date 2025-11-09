/**
 * Node.js Optimizations and Support
 * Enhanced Node.js-specific functionality and optimizations
 * Only available in Node.js environment
 */

/**
 * Node.js optimizations
 * Only available in Node.js environment
 */
export const nodejs = (() => {
  // Check if we're in Node.js environment
  if (typeof process === 'undefined' || !process.versions || !process.versions.node) {
    return {
      isAvailable: () => false,
      error: 'Node.js optimizations only available in Node.js environment'
    };
  }

  // Node.js modules (lazy loaded)
  let os = null;
  let cluster = null;
  let fs = null;
  let path = null;
  let worker_threads = null;

  const loadNodeModules = () => {
    if (!os) {
      try {
        os = require('os');
        cluster = require('cluster');
        fs = require('fs');
        path = require('path');
        worker_threads = require('worker_threads');
      } catch (error) {
        console.warn('Failed to load Node.js modules:', error.message);
      }
    }
  };

  return {
    /**
     * Check if Node.js optimizations are available
     * @returns {boolean} True if in Node.js environment
     */
    isAvailable: () => true,

    /**
     * Get Node.js specific system information
     * @returns {Object} System information
     */
    getSystemInfo() {
      loadNodeModules();

      try {
        return {
          nodeVersion: process.version,
          platform: process.platform,
          arch: process.arch,
          cpus: os ? os.cpus().length : 'unknown',
          memory: os ? {
            total: os.totalmem(),
            free: os.freemem(),
            process: process.memoryUsage()
          } : null,
          pid: process.pid,
          uptime: process.uptime(),
          argv: process.argv,
          env: {
            NODE_ENV: process.env.NODE_ENV,
            UV_THREADPOOL_SIZE: process.env.UV_THREADPOOL_SIZE
          }
        };
      } catch (error) {
        return {
          nodeVersion: process.version,
          error: error.message
        };
      }
    },

    /**
     * Create native module fallback system
     * @param {Object} options - Configuration options
     * @returns {NativeModuleFallback} Fallback system instance
     */
    createNativeModuleFallback(options = {}) {
      return new NativeModuleFallback(options);
    },

    /**
     * Create cluster manager for multi-process inference
     * @param {Object} options - Cluster configuration
     * @returns {ClusterManager} Cluster manager instance
     */
    createClusterManager(options = {}) {
      return new ClusterManager(options);
    },

    /**
     * Create streaming processor for large datasets
     * @param {Object} options - Streaming configuration
     * @returns {StreamingProcessor} Streaming processor instance
     */
    createStreamingProcessor(options = {}) {
      return new StreamingProcessor(options);
    },

    /**
     * Create file system optimizer
     * @param {Object} options - File system configuration
     * @returns {FileSystemOptimizer} File system optimizer instance
     */
    createFileSystemOptimizer(options = {}) {
      return new FileSystemOptimizer(options);
    },

    /**
     * Create memory manager for process optimization
     * @param {Object} options - Memory management configuration
     * @returns {MemoryManager} Memory manager instance
     */
    createMemoryManager(options = {}) {
      return new NodeJSMemoryManager(options);
    },

    /**
     * Optimize Node.js environment for ML workloads
     * @param {Object} options - Optimization options
     */
    optimizeForML(options = {}) {
      const {
        maxMemory = null,
        threadPoolSize = null,
        gcSettings = true
      } = options;

      try {
        // Set UV_THREADPOOL_SIZE if specified
        if (threadPoolSize && !process.env.UV_THREADPOOL_SIZE) {
          process.env.UV_THREADPOOL_SIZE = threadPoolSize.toString();
          console.warn(`Set UV_THREADPOOL_SIZE to ${threadPoolSize}`);
        }

        // Configure V8 flags for better ML performance
        if (gcSettings && process.argv0) {
          const v8Flags = [
            `--max-old-space-size=${maxMemory || 4096}`,
            '--optimize-for-size',
            '--gc-interval=100'
          ];

          console.warn('Recommended V8 flags:', v8Flags.join(' '));
          console.warn(`Add these to your node command: node ${v8Flags.join(' ')} your-script.js`);
        }

        // Set process title for easier identification
        if (process.title) {
          process.title = 'trustformers-ml-worker';
        }

        return {
          optimized: true,
          settings: {
            maxMemory,
            threadPoolSize,
            gcSettings
          }
        };
      } catch (error) {
        console.warn('Failed to optimize Node.js environment:', error.message);
        return { optimized: false, error: error.message };
      }
    }
  };
})();

/**
 * Native Module Fallback System
 * Provides fallback functionality when native modules are not available
 */
class NativeModuleFallback {
  constructor(options = {}) {
    this.options = {
      enableLogging: true,
      enableMetrics: true,
      fallbackTimeout: 5000,
      ...options
    };

    this.availableModules = new Map();
    this.fallbackCounts = new Map();
  }

  /**
   * Check if a native module is available
   * @param {string} moduleName - Name of the module
   * @returns {boolean} True if available
   */
  isModuleAvailable(moduleName) {
    if (this.availableModules.has(moduleName)) {
      return this.availableModules.get(moduleName);
    }

    try {
      require.resolve(moduleName);
      this.availableModules.set(moduleName, true);
      return true;
    } catch (error) {
      this.availableModules.set(moduleName, false);
      if (this.options.enableLogging) {
        console.warn(`Native module ${moduleName} not available:`, error.message);
      }
      return false;
    }
  }

  /**
   * Use module with fallback
   * @param {string} moduleName - Module name
   * @param {Function} nativeFunc - Function using native module
   * @param {Function} fallbackFunc - Fallback function
   * @returns {*} Function result
   */
  async useWithFallback(moduleName, nativeFunc, fallbackFunc) {
    if (this.isModuleAvailable(moduleName)) {
      try {
        return await nativeFunc();
      } catch (error) {
        if (this.options.enableLogging) {
          console.warn(`Native module ${moduleName} failed, using fallback:`, error.message);
        }
        this.recordFallback(moduleName);
        return await fallbackFunc();
      }
    } else {
      this.recordFallback(moduleName);
      return await fallbackFunc();
    }
  }

  /**
   * Record fallback usage
   * @private
   */
  recordFallback(moduleName) {
    if (this.options.enableMetrics) {
      const count = this.fallbackCounts.get(moduleName) || 0;
      this.fallbackCounts.set(moduleName, count + 1);
    }
  }

  /**
   * Get fallback statistics
   * @returns {Object} Statistics
   */
  getStatistics() {
    return {
      availableModules: Object.fromEntries(this.availableModules),
      fallbackCounts: Object.fromEntries(this.fallbackCounts)
    };
  }
}

/**
 * Cluster Manager for Multi-Process Inference
 */
class ClusterManager {
  constructor(options = {}) {
    this.options = {
      numWorkers: require('os').cpus().length,
      enableLogging: true,
      gracefulShutdown: true,
      workerRestartDelay: 1000,
      ...options
    };

    this.cluster = require('cluster');
    this.workers = new Map();
    this.stats = {
      totalRequests: 0,
      activeRequests: 0,
      completedRequests: 0,
      errors: 0
    };
  }

  /**
   * Initialize cluster
   * @returns {Promise<void>}
   */
  async initialize() {
    if (this.cluster.isMaster || this.cluster.isPrimary) {
      await this.startMaster();
    } else {
      await this.startWorker();
    }
  }

  /**
   * Start master process
   * @private
   */
  async startMaster() {
    if (this.options.enableLogging) {
      console.warn(`Master process ${process.pid} starting with ${this.options.numWorkers} workers`);
    }

    // Fork workers
    for (let i = 0; i < this.options.numWorkers; i++) {
      this.forkWorker();
    }

    // Handle worker events
    this.cluster.on('exit', (worker, code, signal) => {
      if (this.options.enableLogging) {
        console.warn(`Worker ${worker.process.pid} died with code ${code} and signal ${signal}`);
      }

      this.workers.delete(worker.id);

      // Restart worker after delay
      setTimeout(() => {
        this.forkWorker();
      }, this.options.workerRestartDelay);
    });

    // Graceful shutdown
    if (this.options.gracefulShutdown) {
      process.on('SIGTERM', () => this.shutdown());
      process.on('SIGINT', () => this.shutdown());
    }
  }

  /**
   * Fork a new worker
   * @private
   */
  forkWorker() {
    const worker = this.cluster.fork();
    this.workers.set(worker.id, {
      worker,
      pid: worker.process.pid,
      requests: 0,
      startTime: Date.now()
    });

    worker.on('message', (message) => {
      this.handleWorkerMessage(worker.id, message);
    });
  }

  /**
   * Start worker process
   * @private
   */
  async startWorker() {
    if (this.options.enableLogging) {
      console.warn(`Worker process ${process.pid} started`);
    }

    // Handle messages from master
    process.on('message', async (message) => {
      try {
        const result = await this.handleInferenceRequest(message);
        process.send({ type: 'result', requestId: message.requestId, result });
      } catch (error) {
        process.send({
          type: 'error',
          requestId: message.requestId,
          error: {
            name: error.name,
            message: error.message,
            stack: error.stack
          }
        });
      }
    });
  }

  /**
   * Handle worker message
   * @private
   */
  handleWorkerMessage(workerId, message) {
    const workerInfo = this.workers.get(workerId);
    if (!workerInfo) return;

    switch (message.type) {
      case 'result':
        this.stats.completedRequests++;
        this.stats.activeRequests--;
        workerInfo.requests++;
        break;

      case 'error':
        this.stats.errors++;
        this.stats.activeRequests--;
        break;
    }
  }

  /**
   * Handle inference request in worker
   * @private
   */
  async handleInferenceRequest(message) {
    // This would be implemented to handle actual inference
    // For now, return a placeholder
    return {
      processed: true,
      timestamp: Date.now(),
      workerId: process.pid
    };
  }

  /**
   * Distribute inference task
   * @param {Object} task - Inference task
   * @returns {Promise<*>} Task result
   */
  async processTask(task) {
    return new Promise((resolve, reject) => {
      const requestId = Math.random().toString(36).substr(2, 9);
      this.stats.totalRequests++;
      this.stats.activeRequests++;

      // Find least loaded worker
      const worker = this.getLeastLoadedWorker();
      if (!worker) {
        reject(new Error('No available workers'));
        return;
      }

      const timeout = setTimeout(() => {
        reject(new Error('Task timeout'));
      }, 30000);

      const messageHandler = (message) => {
        if (message.requestId === requestId) {
          clearTimeout(timeout);
          worker.worker.removeListener('message', messageHandler);

          if (message.type === 'result') {
            resolve(message.result);
          } else {
            reject(new Error(message.error.message));
          }
        }
      };

      worker.worker.on('message', messageHandler);
      worker.worker.send({ ...task, requestId });
    });
  }

  /**
   * Get least loaded worker
   * @private
   */
  getLeastLoadedWorker() {
    let leastLoaded = null;
    let minRequests = Infinity;

    for (const workerInfo of this.workers.values()) {
      if (workerInfo.requests < minRequests) {
        minRequests = workerInfo.requests;
        leastLoaded = workerInfo;
      }
    }

    return leastLoaded;
  }

  /**
   * Get cluster statistics
   * @returns {Object} Statistics
   */
  getStatistics() {
    return {
      ...this.stats,
      workers: this.workers.size,
      uptime: Date.now() - (this.startTime || Date.now())
    };
  }

  /**
   * Shutdown cluster gracefully
   */
  async shutdown() {
    if (this.options.enableLogging) {
      console.warn('Shutting down cluster gracefully...');
    }

    for (const workerInfo of this.workers.values()) {
      workerInfo.worker.kill('SIGTERM');
    }

    // Wait for workers to exit
    await new Promise(resolve => {
      const checkWorkers = () => {
        if (this.workers.size === 0) {
          resolve();
        } else {
          setTimeout(checkWorkers, 100);
        }
      };
      checkWorkers();
    });

    if (this.options.enableLogging) {
      console.warn('Cluster shutdown complete');
    }

    process.exit(0);
  }
}

/**
 * Streaming Processor for Large Datasets
 */
class StreamingProcessor {
  constructor(options = {}) {
    this.options = {
      bufferSize: 1000,
      enableLogging: true,
      enableBackpressure: true,
      maxMemoryUsage: 500 * 1024 * 1024, // 500MB
      ...options
    };

    this.Readable = require('stream').Readable;
    this.Transform = require('stream').Transform;
    this.pipeline = require('stream').pipeline;
  }

  /**
   * Create readable stream from array
   * @param {Array} data - Data array
   * @returns {Readable} Readable stream
   */
  createArrayStream(data) {
    let index = 0;

    return new this.Readable({
      objectMode: true,
      read() {
        if (index < data.length) {
          this.push(data[index++]);
        } else {
          this.push(null);
        }
      }
    });
  }

  /**
   * Create transform stream for processing
   * @param {Function} processFunc - Processing function
   * @returns {Transform} Transform stream
   */
  createProcessingStream(processFunc) {
    const {options} = this;

    return new this.Transform({
      objectMode: true,
      transform(chunk, encoding, callback) {
        try {
          const result = processFunc(chunk);

          if (result instanceof Promise) {
            result
              .then(res => callback(null, res))
              .catch(err => callback(err));
          } else {
            callback(null, result);
          }
        } catch (error) {
          callback(error);
        }
      }
    });
  }

  /**
   * Process data stream
   * @param {Readable} input - Input stream
   * @param {Function} processFunc - Processing function
   * @returns {Promise<Array>} Processed results
   */
  async processStream(input, processFunc) {
    const results = [];
    const processor = this.createProcessingStream(processFunc);

    return new Promise((resolve, reject) => {
      this.pipeline(
        input,
        processor,
        new this.Transform({
          objectMode: true,
          transform(chunk, encoding, callback) {
            results.push(chunk);
            callback();
          }
        }),
        (error) => {
          if (error) {
            reject(error);
          } else {
            resolve(results);
          }
        }
      );
    });
  }

  /**
   * Process large dataset in chunks
   * @param {Array} dataset - Large dataset
   * @param {Function} processFunc - Processing function
   * @param {Object} options - Processing options
   * @returns {AsyncGenerator} Results generator
   */
  async* processLargeDataset(dataset, processFunc, options = {}) {
    const { chunkSize = this.options.bufferSize } = options;

    for (let i = 0; i < dataset.length; i += chunkSize) {
      const chunk = dataset.slice(i, i + chunkSize);
      const stream = this.createArrayStream(chunk);
      const results = await this.processStream(stream, processFunc);

      for (const result of results) {
        yield result;
      }

      // Memory usage check
      if (this.options.enableBackpressure) {
        const memUsage = process.memoryUsage();
        if (memUsage.heapUsed > this.options.maxMemoryUsage) {
          // Force garbage collection if available
          if (global.gc) {
            global.gc();
          }
          // Small delay to allow memory cleanup
          await new Promise(resolve => setTimeout(resolve, 10));
        }
      }
    }
  }
}

/**
 * File System Optimizer
 */
class FileSystemOptimizer {
  constructor(options = {}) {
    this.options = {
      cacheSize: 100,
      enableAtomicWrites: true,
      enableLogging: false,
      tempDir: require('os').tmpdir(),
      ...options
    };

    this.fs = require('fs').promises;
    this.path = require('path');
    this.cache = new Map();
  }

  /**
   * Read file with caching
   * @param {string} filePath - File path
   * @param {Object} options - Read options
   * @returns {Promise<Buffer|string>} File content
   */
  async readFile(filePath, options = {}) {
    const { encoding = null, useCache = true } = options;
    const cacheKey = `${filePath}:${encoding}`;

    if (useCache && this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey);
    }

    try {
      const content = await this.fs.readFile(filePath, encoding);

      if (useCache && this.cache.size < this.options.cacheSize) {
        this.cache.set(cacheKey, content);
      }

      return content;
    } catch (error) {
      if (this.options.enableLogging) {
        console.error(`Failed to read file ${filePath}:`, error.message);
      }
      throw error;
    }
  }

  /**
   * Write file with atomic operation
   * @param {string} filePath - File path
   * @param {Buffer|string} data - Data to write
   * @param {Object} options - Write options
   * @returns {Promise<void>}
   */
  async writeFile(filePath, data, options = {}) {
    const { atomic = this.options.enableAtomicWrites } = options;

    if (atomic) {
      const tempPath = this.path.join(
        this.options.tempDir,
        `tmp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      );

      try {
        await this.fs.writeFile(tempPath, data, options);
        await this.fs.rename(tempPath, filePath);
      } catch (error) {
        // Clean up temp file if it exists
        try {
          await this.fs.unlink(tempPath);
        } catch (_) {
          // Ignore cleanup errors
        }
        throw error;
      }
    } else {
      await this.fs.writeFile(filePath, data, options);
    }

    // Invalidate cache
    this.cache.forEach((_, key) => {
      if (key.startsWith(`${filePath}:`)) {
        this.cache.delete(key);
      }
    });
  }

  /**
   * Batch file operations
   * @param {Array} operations - Array of file operations
   * @returns {Promise<Array>} Results
   */
  async batchOperations(operations) {
    const results = [];

    for (const operation of operations) {
      try {
        let result;

        switch (operation.type) {
          case 'read':
            result = await this.readFile(operation.path, operation.options);
            break;

          case 'write':
            result = await this.writeFile(operation.path, operation.data, operation.options);
            break;

          default:
            throw new Error(`Unknown operation type: ${operation.type}`);
        }

        results.push({ success: true, result });
      } catch (error) {
        results.push({ success: false, error: error.message });
      }
    }

    return results;
  }

  /**
   * Clear file cache
   */
  clearCache() {
    this.cache.clear();
  }

  /**
   * Get cache statistics
   * @returns {Object} Cache statistics
   */
  getCacheStats() {
    return {
      size: this.cache.size,
      maxSize: this.options.cacheSize,
      hitRatio: this.hitCount / (this.hitCount + this.missCount) || 0
    };
  }
}

/**
 * Node.js Memory Manager
 */
class NodeJSMemoryManager {
  constructor(options = {}) {
    this.options = {
      maxHeapSize: 1024 * 1024 * 1024, // 1GB
      gcThreshold: 0.8,
      enableMonitoring: true,
      monitoringInterval: 5000,
      ...options
    };

    this.monitoringTimer = null;
    this.stats = {
      gcCount: 0,
      lastGc: null,
      peakMemory: 0
    };

    if (this.options.enableMonitoring) {
      this.startMonitoring();
    }
  }

  /**
   * Start memory monitoring
   */
  startMonitoring() {
    if (this.monitoringTimer) return;

    this.monitoringTimer = setInterval(() => {
      const memUsage = process.memoryUsage();
      const heapUsedMB = memUsage.heapUsed / 1024 / 1024;
      const heapTotalMB = memUsage.heapTotal / 1024 / 1024;

      this.stats.peakMemory = Math.max(this.stats.peakMemory, memUsage.heapUsed);

      if (heapUsedMB > this.options.maxHeapSize * this.options.gcThreshold) {
        this.triggerGC();
      }
    }, this.options.monitoringInterval);
  }

  /**
   * Stop memory monitoring
   */
  stopMonitoring() {
    if (this.monitoringTimer) {
      clearInterval(this.monitoringTimer);
      this.monitoringTimer = null;
    }
  }

  /**
   * Trigger garbage collection
   */
  triggerGC() {
    if (global.gc) {
      try {
        global.gc();
        this.stats.gcCount++;
        this.stats.lastGc = new Date();
      } catch (error) {
        console.warn('Failed to trigger GC:', error.message);
      }
    }
  }

  /**
   * Get memory usage statistics
   * @returns {Object} Memory statistics
   */
  getMemoryStats() {
    const memUsage = process.memoryUsage();

    return {
      heap: {
        used: Math.round(memUsage.heapUsed / 1024 / 1024),
        total: Math.round(memUsage.heapTotal / 1024 / 1024),
        utilization: (memUsage.heapUsed / memUsage.heapTotal * 100).toFixed(2)
      },
      external: Math.round(memUsage.external / 1024 / 1024),
      arrayBuffers: Math.round(memUsage.arrayBuffers / 1024 / 1024),
      rss: Math.round(memUsage.rss / 1024 / 1024),
      gc: this.stats,
      monitoring: {
        enabled: this.options.enableMonitoring,
        interval: this.options.monitoringInterval
      }
    };
  }

  /**
   * Optimize memory settings
   * @returns {Object} Optimization result
   */
  optimizeMemory() {
    const recommendations = [];

    const memUsage = process.memoryUsage();
    const heapUtilization = memUsage.heapUsed / memUsage.heapTotal;

    if (heapUtilization > 0.9) {
      recommendations.push('Consider increasing max heap size with --max-old-space-size');
    }

    if (memUsage.external > memUsage.heapUsed) {
      recommendations.push('High external memory usage detected - check for memory leaks');
    }

    if (this.stats.gcCount > 100) {
      recommendations.push('High GC frequency - consider optimizing object creation');
    }

    return {
      optimized: true,
      recommendations,
      currentStats: this.getMemoryStats()
    };
  }

  /**
   * Cleanup memory manager
   */
  cleanup() {
    this.stopMonitoring();
    this.triggerGC();
  }
}

export default nodejs;