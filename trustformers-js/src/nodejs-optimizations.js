/**
 * Node.js Optimizations for TrustformeRS
 * 
 * This module provides Node.js-specific optimizations including:
 * - Native module fallbacks
 * - Cluster support
 * - Streaming data processing
 * - File system optimizations
 * - Process memory management
 */

const os = require('os');
const fs = require('fs');
const path = require('path');
const cluster = require('cluster');
const { Worker } = require('worker_threads');
const { performance } = require('perf_hooks');

/**
 * Native Module Fallback System
 * Provides fallback mechanisms when native modules aren't available
 */
class NativeModuleFallback {
    constructor() {
        this.nativeModules = new Map();
        this.fallbackHandlers = new Map();
        this.availability = new Map();
    }

    /**
     * Register a native module with fallback
     * @param {string} moduleName - Name of the native module
     * @param {Function} nativeHandler - Native implementation
     * @param {Function} fallbackHandler - Fallback implementation
     */
    registerModule(moduleName, nativeHandler, fallbackHandler) {
        this.nativeModules.set(moduleName, nativeHandler);
        this.fallbackHandlers.set(moduleName, fallbackHandler);
        
        // Test module availability
        try {
            nativeHandler();
            this.availability.set(moduleName, true);
        } catch (error) {
            console.warn(`Native module ${moduleName} not available, using fallback:`, error.message);
            this.availability.set(moduleName, false);
        }
    }

    /**
     * Execute module with fallback
     * @param {string} moduleName - Name of the module
     * @param {...any} args - Arguments to pass
     */
    execute(moduleName, ...args) {
        const isNativeAvailable = this.availability.get(moduleName);
        
        if (isNativeAvailable) {
            try {
                return this.nativeModules.get(moduleName)(...args);
            } catch (error) {
                console.warn(`Native module ${moduleName} failed, falling back:`, error.message);
                this.availability.set(moduleName, false);
            }
        }
        
        const fallbackHandler = this.fallbackHandlers.get(moduleName);
        if (fallbackHandler) {
            return fallbackHandler(...args);
        }
        
        throw new Error(`No handler available for module: ${moduleName}`);
    }

    /**
     * Get module availability status
     */
    getAvailability() {
        return Object.fromEntries(this.availability);
    }
}

/**
 * Cluster Support for Multi-Process Model Inference
 */
class ClusterManager {
    constructor(options = {}) {
        this.numWorkers = options.numWorkers || os.cpus().length;
        this.workerOptions = options.workerOptions || {};
        this.workers = new Map();
        this.requestQueue = [];
        this.roundRobinIndex = 0;
        this.stats = {
            totalRequests: 0,
            completedRequests: 0,
            failedRequests: 0,
            avgResponseTime: 0
        };
    }

    /**
     * Initialize cluster with worker processes
     */
    async initialize() {
        if (cluster.isMaster) {
            console.warn(`Master process ${process.pid} starting ${this.numWorkers} workers`);
            
            for (let i = 0; i < this.numWorkers; i++) {
                await this.createWorker(i);
            }
            
            // Handle worker messages
            cluster.on('message', (worker, message) => {
                this.handleWorkerMessage(worker.id, message);
            });
            
            // Handle worker exit
            cluster.on('exit', (worker, code, signal) => {
                console.warn(`Worker ${worker.process.pid} died. Restarting...`);
                this.createWorker(worker.id);
            });
        }
    }

    /**
     * Create a worker process
     * @param {number} workerId - Worker ID
     */
    async createWorker(workerId) {
        const worker = cluster.fork({
            WORKER_ID: workerId,
            ...this.workerOptions
        });
        
        this.workers.set(workerId, {
            process: worker,
            busy: false,
            requestCount: 0,
            avgResponseTime: 0
        });
        
        return worker;
    }

    /**
     * Distribute inference request to available worker
     * @param {Object} request - Inference request
     */
    async distributeRequest(request) {
        const startTime = performance.now();
        this.stats.totalRequests++;

        return new Promise((resolve, reject) => {
            const worker = this.selectWorker();
            
            if (!worker) {
                this.requestQueue.push({ request, resolve, reject, startTime });
                return;
            }

            const workerInfo = this.workers.get(worker.id);
            workerInfo.busy = true;
            workerInfo.requestCount++;

            const timeout = setTimeout(() => {
                reject(new Error('Request timeout'));
                workerInfo.busy = false;
            }, 30000);

            const messageHandler = (message) => {
                if (message.type === 'inference_result' && message.requestId === request.id) {
                    clearTimeout(timeout);
                    worker.off('message', messageHandler);
                    
                    const responseTime = performance.now() - startTime;
                    this.updateStats(responseTime, true);
                    
                    workerInfo.busy = false;
                    workerInfo.avgResponseTime = (workerInfo.avgResponseTime + responseTime) / 2;
                    
                    resolve(message.result);
                    this.processQueue();
                }
            };

            worker.on('message', messageHandler);
            worker.send({
                type: 'inference_request',
                requestId: request.id,
                data: request
            });
        });
    }

    /**
     * Select best available worker using round-robin with load balancing
     */
    selectWorker() {
        const availableWorkers = Array.from(this.workers.values())
            .filter(w => !w.busy)
            .sort((a, b) => a.requestCount - b.requestCount);

        if (availableWorkers.length === 0) {
            return null;
        }

        // Use round-robin among available workers
        const worker = availableWorkers[this.roundRobinIndex % availableWorkers.length];
        this.roundRobinIndex++;
        
        return worker.process;
    }

    /**
     * Process queued requests
     */
    processQueue() {
        if (this.requestQueue.length === 0) return;

        const worker = this.selectWorker();
        if (!worker) return;

        const { request, resolve, reject, startTime } = this.requestQueue.shift();
        
        const workerInfo = this.workers.get(worker.id);
        workerInfo.busy = true;
        workerInfo.requestCount++;

        const timeout = setTimeout(() => {
            reject(new Error('Request timeout'));
            workerInfo.busy = false;
        }, 30000);

        const messageHandler = (message) => {
            if (message.type === 'inference_result' && message.requestId === request.id) {
                clearTimeout(timeout);
                worker.off('message', messageHandler);
                
                const responseTime = performance.now() - startTime;
                this.updateStats(responseTime, true);
                
                workerInfo.busy = false;
                workerInfo.avgResponseTime = (workerInfo.avgResponseTime + responseTime) / 2;
                
                resolve(message.result);
                this.processQueue();
            }
        };

        worker.on('message', messageHandler);
        worker.send({
            type: 'inference_request',
            requestId: request.id,
            data: request
        });
    }

    /**
     * Update performance statistics
     */
    updateStats(responseTime, success) {
        if (success) {
            this.stats.completedRequests++;
        } else {
            this.stats.failedRequests++;
        }
        
        this.stats.avgResponseTime = (
            (this.stats.avgResponseTime * (this.stats.completedRequests - 1) + responseTime) /
            this.stats.completedRequests
        );
    }

    /**
     * Get cluster statistics
     */
    getStats() {
        return {
            ...this.stats,
            workers: Array.from(this.workers.entries()).map(([id, info]) => ({
                id,
                busy: info.busy,
                requestCount: info.requestCount,
                avgResponseTime: info.avgResponseTime
            })),
            queueLength: this.requestQueue.length
        };
    }
}

/**
 * Streaming Data Processing for Large Datasets
 */
class StreamingProcessor {
    constructor(options = {}) {
        this.batchSize = options.batchSize || 1000;
        this.maxMemory = options.maxMemory || 1024 * 1024 * 1024; // 1GB
        this.concurrency = options.concurrency || 4;
        this.tempDir = options.tempDir || os.tmpdir();
    }

    /**
     * Process streaming data in batches
     * @param {ReadableStream} inputStream - Input data stream
     * @param {Function} processor - Processing function
     * @param {WritableStream} outputStream - Output stream
     */
    async processStream(inputStream, processor, outputStream) {
        const { Transform } = require('stream');
        const { pipeline } = require('stream/promises');
        
        let batch = [];
        let batchCount = 0;
        let processedCount = 0;
        
        const batchProcessor = new Transform({
            objectMode: true,
            transform: async (chunk, encoding, callback) => {
                batch.push(chunk);
                
                if (batch.length >= this.batchSize) {
                    try {
                        const results = await processor(batch);
                        batch = [];
                        batchCount++;
                        processedCount += results.length;
                        
                        // Monitor memory usage
                        if (process.memoryUsage().heapUsed > this.maxMemory) {
                            await this.gc();
                        }
                        
                        callback(null, results);
                    } catch (error) {
                        callback(error);
                    }
                } else {
                    callback();
                }
            },
            flush: async (callback) => {
                if (batch.length > 0) {
                    try {
                        const results = await processor(batch);
                        processedCount += results.length;
                        callback(null, results);
                    } catch (error) {
                        callback(error);
                    }
                } else {
                    callback();
                }
            }
        });

        await pipeline(inputStream, batchProcessor, outputStream);
        
        return {
            batchCount,
            processedCount,
            avgBatchSize: processedCount / batchCount
        };
    }

    /**
     * Force garbage collection if available
     */
    async gc() {
        if (global.gc) {
            global.gc();
        }
        
        // Brief pause to allow memory cleanup
        await new Promise(resolve => setTimeout(resolve, 100));
    }

    /**
     * Create temporary file for spilling data
     */
    async createTempFile(data) {
        const tempFile = path.join(this.tempDir, `tf-stream-${Date.now()}-${Math.random()}.tmp`);
        await fs.promises.writeFile(tempFile, JSON.stringify(data));
        return tempFile;
    }

    /**
     * Read and delete temporary file
     */
    async readTempFile(tempFile) {
        const data = await fs.promises.readFile(tempFile, 'utf8');
        await fs.promises.unlink(tempFile);
        return JSON.parse(data);
    }
}

/**
 * File System Optimizations
 */
class FileSystemOptimizer {
    constructor(options = {}) {
        this.cacheSize = options.cacheSize || 100;
        this.cache = new Map();
        this.stats = {
            hits: 0,
            misses: 0,
            reads: 0,
            writes: 0
        };
    }

    /**
     * Optimized file reading with caching
     * @param {string} filePath - File path
     * @param {Object} options - Read options
     */
    async readFile(filePath, options = {}) {
        const cacheKey = `${filePath}:${JSON.stringify(options)}`;
        
        if (this.cache.has(cacheKey)) {
            this.stats.hits++;
            return this.cache.get(cacheKey);
        }

        this.stats.misses++;
        this.stats.reads++;

        const data = await fs.promises.readFile(filePath, options);
        
        // Cache management
        if (this.cache.size >= this.cacheSize) {
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }
        
        this.cache.set(cacheKey, data);
        return data;
    }

    /**
     * Optimized file writing with atomic operations
     * @param {string} filePath - File path
     * @param {any} data - Data to write
     * @param {Object} options - Write options
     */
    async writeFile(filePath, data, options = {}) {
        this.stats.writes++;
        
        const tempFile = `${filePath}.tmp`;
        
        try {
            await fs.promises.writeFile(tempFile, data, options);
            await fs.promises.rename(tempFile, filePath);
        } catch (error) {
            try {
                await fs.promises.unlink(tempFile);
            } catch (cleanupError) {
                // Ignore cleanup errors
            }
            throw error;
        }
        
        // Invalidate cache
        for (const key of this.cache.keys()) {
            if (key.startsWith(filePath)) {
                this.cache.delete(key);
            }
        }
    }

    /**
     * Batch file operations for efficiency
     * @param {Array} operations - Array of {type, path, data, options}
     */
    async batchOperations(operations) {
        const results = [];
        
        // Group operations by type
        const reads = operations.filter(op => op.type === 'read');
        const writes = operations.filter(op => op.type === 'write');
        
        // Process reads concurrently
        const readPromises = reads.map(op => 
            this.readFile(op.path, op.options)
                .then(data => ({ success: true, data }))
                .catch(error => ({ success: false, error }))
        );
        
        // Process writes sequentially for consistency
        const writeResults = [];
        for (const op of writes) {
            try {
                await this.writeFile(op.path, op.data, op.options);
                writeResults.push({ success: true });
            } catch (error) {
                writeResults.push({ success: false, error });
            }
        }
        
        const readResults = await Promise.all(readPromises);
        
        // Combine results in original order
        let readIndex = 0;
        let writeIndex = 0;
        
        for (const op of operations) {
            if (op.type === 'read') {
                results.push(readResults[readIndex++]);
            } else {
                results.push(writeResults[writeIndex++]);
            }
        }
        
        return results;
    }

    /**
     * Get file system cache statistics
     */
    getStats() {
        return {
            ...this.stats,
            cacheSize: this.cache.size,
            hitRate: this.stats.hits / (this.stats.hits + this.stats.misses)
        };
    }
}

/**
 * Process Memory Management
 */
class MemoryManager {
    constructor(options = {}) {
        this.maxHeapSize = options.maxHeapSize || 1024 * 1024 * 1024; // 1GB
        this.gcThreshold = options.gcThreshold || 0.8;
        this.monitorInterval = options.monitorInterval || 10000; // 10 seconds
        this.stats = {
            collections: 0,
            maxHeapUsed: 0,
            avgHeapUsed: 0,
            measurements: 0
        };
        this.monitoring = false;
    }

    /**
     * Start memory monitoring
     */
    startMonitoring() {
        if (this.monitoring) return;
        
        this.monitoring = true;
        this.monitorTimer = setInterval(() => {
            this.checkMemory();
        }, this.monitorInterval);
    }

    /**
     * Stop memory monitoring
     */
    stopMonitoring() {
        if (!this.monitoring) return;
        
        this.monitoring = false;
        clearInterval(this.monitorTimer);
    }

    /**
     * Check memory usage and trigger GC if needed
     */
    checkMemory() {
        const memUsage = process.memoryUsage();
        const {heapUsed} = memUsage;
        
        // Update statistics
        this.stats.measurements++;
        this.stats.maxHeapUsed = Math.max(this.stats.maxHeapUsed, heapUsed);
        this.stats.avgHeapUsed = (this.stats.avgHeapUsed + heapUsed) / 2;
        
        // Check if GC is needed
        if (heapUsed > this.maxHeapSize * this.gcThreshold) {
            this.forceGC();
        }
    }

    /**
     * Force garbage collection
     */
    forceGC() {
        if (global.gc) {
            global.gc();
            this.stats.collections++;
        }
    }

    /**
     * Get memory statistics
     */
    getStats() {
        return {
            ...this.stats,
            currentMemory: process.memoryUsage(),
            maxHeapSize: this.maxHeapSize,
            gcThreshold: this.gcThreshold
        };
    }

    /**
     * Optimize memory for large operations
     * @param {Function} operation - Operation to perform
     */
    async optimizeOperation(operation) {
        // Pre-operation cleanup
        this.forceGC();
        
        const startMemory = process.memoryUsage().heapUsed;
        
        try {
            const result = await operation();
            return result;
        } finally {
            // Post-operation cleanup
            this.forceGC();
            
            const endMemory = process.memoryUsage().heapUsed;
            const memoryDelta = endMemory - startMemory;
            
            if (memoryDelta > 100 * 1024 * 1024) { // 100MB
                console.warn(`Large memory increase detected: ${memoryDelta / (1024 * 1024)}MB`);
            }
        }
    }
}

module.exports = {
    NativeModuleFallback,
    ClusterManager,
    StreamingProcessor,
    FileSystemOptimizer,
    MemoryManager
};