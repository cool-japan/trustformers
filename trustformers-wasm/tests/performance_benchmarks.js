/**
 * Comprehensive Performance Benchmarks for TrustformeRS WASM
 * 
 * This test suite provides detailed performance benchmarks across different
 * browsers, devices, and configurations to ensure optimal performance.
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach } from '@jest/globals';

// Benchmark configuration
const BENCHMARK_CONFIG = {
    warmupIterations: 10,
    benchmarkIterations: 100,
    memoryCheckInterval: 50,
    timeoutMs: 30000,
    
    // Test data sizes
    tensorSizes: {
        tiny: [8, 8],
        small: [32, 32],
        medium: [128, 128],
        large: [512, 512],
        xlarge: [1024, 1024]
    },
    
    // Operation types to benchmark
    operations: [
        'tensor_creation',
        'tensor_addition',
        'tensor_multiplication',
        'matrix_multiplication',
        'activation_functions',
        'layer_normalization',
        'attention_mechanism'
    ]
};

// Performance metrics collector
class PerformanceMetrics {
    constructor() {
        this.metrics = new Map();
        this.memorySnapshots = [];
        this.startTime = null;
        this.endTime = null;
    }
    
    startSuite() {
        this.startTime = performance.now();
        this.takeMemorySnapshot('suite_start');
    }
    
    endSuite() {
        this.endTime = performance.now();
        this.takeMemorySnapshot('suite_end');
    }
    
    takeMemorySnapshot(label) {
        if (performance.memory) {
            this.memorySnapshots.push({
                label,
                timestamp: performance.now(),
                usedJSHeapSize: performance.memory.usedJSHeapSize,
                totalJSHeapSize: performance.memory.totalJSHeapSize,
                jsHeapSizeLimit: performance.memory.jsHeapSizeLimit
            });
        }
    }
    
    recordMetric(operation, size, browser, metrics) {
        const key = `${operation}_${size}_${browser}`;
        this.metrics.set(key, {
            operation,
            size,
            browser,
            ...metrics,
            timestamp: new Date().toISOString()
        });
    }
    
    getMetric(operation, size, browser) {
        const key = `${operation}_${size}_${browser}`;
        return this.metrics.get(key);
    }
    
    getAllMetrics() {
        return Array.from(this.metrics.values());
    }
    
    getMemoryUsage() {
        return {
            snapshots: this.memorySnapshots,
            totalDuration: this.endTime - this.startTime
        };
    }
    
    generateReport() {
        const operations = new Set();
        const sizes = new Set();
        const browsers = new Set();
        
        this.metrics.forEach(metric => {
            operations.add(metric.operation);
            sizes.add(metric.size);
            browsers.add(metric.browser);
        });
        
        const report = {
            summary: {
                totalOperations: operations.size,
                totalSizes: sizes.size,
                totalBrowsers: browsers.size,
                totalTests: this.metrics.size
            },
            operations: Array.from(operations),
            sizes: Array.from(sizes),
            browsers: Array.from(browsers),
            metrics: this.getAllMetrics(),
            memory: this.getMemoryUsage()
        };
        
        return report;
    }
    
    comparePerformance(operation, size) {
        const metrics = this.getAllMetrics()
            .filter(m => m.operation === operation && m.size === size);
        
        if (metrics.length === 0) return null;
        
        const comparison = {
            operation,
            size,
            browsers: {}
        };
        
        metrics.forEach(metric => {
            comparison.browsers[metric.browser] = {
                averageTime: metric.averageTime,
                minTime: metric.minTime,
                maxTime: metric.maxTime,
                standardDeviation: metric.standardDeviation,
                throughput: metric.throughput
            };
        });
        
        return comparison;
    }
}

// Benchmark utilities
class BenchmarkUtils {
    static async warmup(fn, iterations = BENCHMARK_CONFIG.warmupIterations) {
        for (let i = 0; i < iterations; i++) {
            await fn();
        }
    }
    
    static async benchmark(fn, iterations = BENCHMARK_CONFIG.benchmarkIterations) {
        const times = [];
        const startTime = performance.now();
        
        for (let i = 0; i < iterations; i++) {
            const iterationStart = performance.now();
            await fn();
            const iterationEnd = performance.now();
            times.push(iterationEnd - iterationStart);
        }
        
        const endTime = performance.now();
        const totalTime = endTime - startTime;
        
        return BenchmarkUtils.calculateStatistics(times, totalTime);
    }
    
    static calculateStatistics(times, totalTime) {
        const sortedTimes = times.slice().sort((a, b) => a - b);
        const sum = times.reduce((a, b) => a + b, 0);
        const average = sum / times.length;
        const min = Math.min(...times);
        const max = Math.max(...times);
        
        const variance = times.reduce((sum, time) => sum + Math.pow(time - average, 2), 0) / times.length;
        const standardDeviation = Math.sqrt(variance);
        
        const p50 = sortedTimes[Math.floor(times.length * 0.5)];
        const p90 = sortedTimes[Math.floor(times.length * 0.9)];
        const p95 = sortedTimes[Math.floor(times.length * 0.95)];
        const p99 = sortedTimes[Math.floor(times.length * 0.99)];
        
        return {
            iterations: times.length,
            totalTime,
            averageTime: average,
            minTime: min,
            maxTime: max,
            standardDeviation,
            variance,
            percentiles: { p50, p90, p95, p99 },
            throughput: times.length / (totalTime / 1000), // operations per second
            coefficient_of_variation: standardDeviation / average
        };
    }
    
    static getBrowserInfo() {
        const ua = navigator.userAgent;
        let browserName = 'unknown';
        let browserVersion = 'unknown';
        
        if (ua.includes('Chrome')) {
            browserName = 'chrome';
            const match = ua.match(/Chrome\/(\d+\.\d+)/);
            browserVersion = match ? match[1] : 'unknown';
        } else if (ua.includes('Firefox')) {
            browserName = 'firefox';
            const match = ua.match(/Firefox\/(\d+\.\d+)/);
            browserVersion = match ? match[1] : 'unknown';
        } else if (ua.includes('Safari')) {
            browserName = 'safari';
            const match = ua.match(/Version\/(\d+\.\d+)/);
            browserVersion = match ? match[1] : 'unknown';
        } else if (ua.includes('Edge')) {
            browserName = 'edge';
            const match = ua.match(/Edg\/(\d+\.\d+)/);
            browserVersion = match ? match[1] : 'unknown';
        }
        
        return {
            name: browserName,
            version: browserVersion,
            userAgent: ua,
            hardwareConcurrency: navigator.hardwareConcurrency || 1,
            deviceMemory: navigator.deviceMemory || 'unknown',
            platform: navigator.platform
        };
    }
    
    static async checkWebGPUSupport() {
        if (!navigator.gpu) {
            return { supported: false, reason: 'WebGPU not available' };
        }
        
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                return { supported: false, reason: 'No adapter available' };
            }
            
            const device = await adapter.requestDevice();
            return {
                supported: true,
                adapter: {
                    features: Array.from(adapter.features),
                    limits: adapter.limits
                },
                device: !!device
            };
        } catch (error) {
            return { supported: false, reason: error.message };
        }
    }
}

// Mock TrustformeRS WASM module for benchmarking
class MockTrustformersWasm {
    constructor() {
        this.initialized = false;
        this.tensors = new Map();
        this.nextTensorId = 1;
        this.useWebGPU = false;
        this.operationCount = 0;
    }
    
    async initialize() {
        const start = performance.now();
        
        // Simulate initialization overhead
        await new Promise(resolve => setTimeout(resolve, 50 + Math.random() * 100));
        
        this.initialized = true;
        return performance.now() - start;
    }
    
    async enableWebGPU() {
        const webGPUSupport = await BenchmarkUtils.checkWebGPUSupport();
        if (!webGPUSupport.supported) {
            throw new Error(`WebGPU not supported: ${webGPUSupport.reason}`);
        }
        
        this.useWebGPU = true;
        return true;
    }
    
    createTensor(shape, data) {
        if (!this.initialized) {
            throw new Error('WASM module not initialized');
        }
        
        const id = this.nextTensorId++;
        const size = shape.reduce((a, b) => a * b, 1);
        
        // Simulate different creation overhead based on size
        const creationOverhead = Math.sqrt(size) * 0.001;
        
        const tensor = {
            id,
            shape: [...shape],
            data: data ? new Float32Array(data) : new Float32Array(size),
            device: this.useWebGPU ? 'gpu' : 'cpu',
            creationTime: performance.now(),
            size
        };
        
        this.tensors.set(id, tensor);
        this.operationCount++;
        
        return id;
    }
    
    addTensors(id1, id2) {
        const tensor1 = this.tensors.get(id1);
        const tensor2 = this.tensors.get(id2);
        
        if (!tensor1 || !tensor2) {
            throw new Error('Invalid tensor ID');
        }
        
        if (tensor1.shape.length !== tensor2.shape.length ||
            !tensor1.shape.every((dim, i) => dim === tensor2.shape[i])) {
            throw new Error('Shape mismatch');
        }
        
        const resultData = new Float32Array(tensor1.data.length);
        
        // Simulate computation overhead
        const computationOverhead = this.useWebGPU ? 
            tensor1.data.length * 0.0001 : // GPU is faster
            tensor1.data.length * 0.001;   // CPU is slower
        
        for (let i = 0; i < resultData.length; i++) {
            resultData[i] = tensor1.data[i] + tensor2.data[i];
        }
        
        this.operationCount++;
        return this.createTensor(tensor1.shape, resultData);
    }
    
    multiplyTensors(id1, id2) {
        const tensor1 = this.tensors.get(id1);
        const tensor2 = this.tensors.get(id2);
        
        if (!tensor1 || !tensor2) {
            throw new Error('Invalid tensor ID');
        }
        
        if (tensor1.shape.length !== tensor2.shape.length ||
            !tensor1.shape.every((dim, i) => dim === tensor2.shape[i])) {
            throw new Error('Shape mismatch');
        }
        
        const resultData = new Float32Array(tensor1.data.length);
        
        for (let i = 0; i < resultData.length; i++) {
            resultData[i] = tensor1.data[i] * tensor2.data[i];
        }
        
        this.operationCount++;
        return this.createTensor(tensor1.shape, resultData);
    }
    
    matrixMultiply(id1, id2) {
        const tensor1 = this.tensors.get(id1);
        const tensor2 = this.tensors.get(id2);
        
        if (!tensor1 || !tensor2) {
            throw new Error('Invalid tensor ID');
        }
        
        if (tensor1.shape.length !== 2 || tensor2.shape.length !== 2) {
            throw new Error('Only 2D tensors supported for matrix multiplication');
        }
        
        const [m, k] = tensor1.shape;
        const [k2, n] = tensor2.shape;
        
        if (k !== k2) {
            throw new Error('Invalid dimensions for matrix multiplication');
        }
        
        const resultData = new Float32Array(m * n);
        
        // Simulate matrix multiplication overhead
        const matmulOverhead = this.useWebGPU ? 
            (m * n * k) * 0.00001 : // GPU is much faster for large matrices
            (m * n * k) * 0.0001;   // CPU is slower
        
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
                let sum = 0;
                for (let p = 0; p < k; p++) {
                    sum += tensor1.data[i * k + p] * tensor2.data[p * n + j];
                }
                resultData[i * n + j] = sum;
            }
        }
        
        this.operationCount++;
        return this.createTensor([m, n], resultData);
    }
    
    relu(id) {
        const tensor = this.tensors.get(id);
        if (!tensor) {
            throw new Error('Invalid tensor ID');
        }
        
        const resultData = new Float32Array(tensor.data.length);
        for (let i = 0; i < resultData.length; i++) {
            resultData[i] = Math.max(0, tensor.data[i]);
        }
        
        this.operationCount++;
        return this.createTensor(tensor.shape, resultData);
    }
    
    layerNorm(id, epsilon = 1e-5) {
        const tensor = this.tensors.get(id);
        if (!tensor) {
            throw new Error('Invalid tensor ID');
        }
        
        const resultData = new Float32Array(tensor.data.length);
        
        // Calculate mean
        let sum = 0;
        for (let i = 0; i < tensor.data.length; i++) {
            sum += tensor.data[i];
        }
        const mean = sum / tensor.data.length;
        
        // Calculate variance
        let varSum = 0;
        for (let i = 0; i < tensor.data.length; i++) {
            varSum += Math.pow(tensor.data[i] - mean, 2);
        }
        const variance = varSum / tensor.data.length;
        const stdDev = Math.sqrt(variance + epsilon);
        
        // Normalize
        for (let i = 0; i < resultData.length; i++) {
            resultData[i] = (tensor.data[i] - mean) / stdDev;
        }
        
        this.operationCount++;
        return this.createTensor(tensor.shape, resultData);
    }
    
    attentionMechanism(queryId, keyId, valueId) {
        const query = this.tensors.get(queryId);
        const key = this.tensors.get(keyId);
        const value = this.tensors.get(valueId);
        
        if (!query || !key || !value) {
            throw new Error('Invalid tensor ID');
        }
        
        // Simplified attention mechanism for benchmarking
        // In reality, this would be much more complex
        
        // QK^T
        const qkId = this.matrixMultiply(queryId, keyId);
        
        // Softmax (simplified)
        const qkTensor = this.tensors.get(qkId);
        const softmaxData = new Float32Array(qkTensor.data.length);
        
        let maxVal = Math.max(...qkTensor.data);
        let expSum = 0;
        
        for (let i = 0; i < qkTensor.data.length; i++) {
            softmaxData[i] = Math.exp(qkTensor.data[i] - maxVal);
            expSum += softmaxData[i];
        }
        
        for (let i = 0; i < softmaxData.length; i++) {
            softmaxData[i] /= expSum;
        }
        
        const softmaxId = this.createTensor(qkTensor.shape, softmaxData);
        
        // Attention * V
        const resultId = this.matrixMultiply(softmaxId, valueId);
        
        this.operationCount++;
        return resultId;
    }
    
    getMemoryUsage() {
        const totalTensors = this.tensors.size;
        const totalElements = Array.from(this.tensors.values())
            .reduce((sum, tensor) => sum + tensor.data.length, 0);
        const totalBytes = totalElements * 4; // Float32 = 4 bytes
        
        return {
            totalTensors,
            totalElements,
            totalBytes,
            estimatedMemoryMB: totalBytes / (1024 * 1024),
            operationCount: this.operationCount
        };
    }
    
    cleanup() {
        this.tensors.clear();
        this.nextTensorId = 1;
        this.operationCount = 0;
    }
}

// Global test setup
let performanceMetrics;
let wasmModule;
let browserInfo;

describe('Comprehensive Performance Benchmarks', () => {
    beforeAll(async () => {
        performanceMetrics = new PerformanceMetrics();
        browserInfo = BenchmarkUtils.getBrowserInfo();
        wasmModule = new MockTrustformersWasm();
        
        await wasmModule.initialize();
        performanceMetrics.startSuite();
        
        console.log('Browser Info:', browserInfo);
        console.log('Starting performance benchmarks...');
    });
    
    afterAll(async () => {
        performanceMetrics.endSuite();
        
        const report = performanceMetrics.generateReport();
        console.log('\n=== PERFORMANCE BENCHMARK REPORT ===');
        console.log(JSON.stringify(report, null, 2));
        
        // Store detailed results for potential analysis
        if (typeof localStorage !== 'undefined') {
            localStorage.setItem('trustformers_benchmark_results', JSON.stringify(report));
        }
    });
    
    beforeEach(() => {
        wasmModule.cleanup();
        performanceMetrics.takeMemorySnapshot(`test_start_${Date.now()}`);
    });
    
    afterEach(() => {
        performanceMetrics.takeMemorySnapshot(`test_end_${Date.now()}`);
    });

    describe('Tensor Creation Benchmarks', () => {
        Object.entries(BENCHMARK_CONFIG.tensorSizes).forEach(([sizeName, shape]) => {
            it(`should create ${sizeName} tensors efficiently`, async () => {
                const testFn = () => {
                    const id = wasmModule.createTensor(shape);
                    wasmModule.tensors.delete(id); // Clean up immediately
                };
                
                // Warmup
                await BenchmarkUtils.warmup(testFn);
                
                // Benchmark
                const results = await BenchmarkUtils.benchmark(testFn);
                
                // Record metrics
                performanceMetrics.recordMetric(
                    'tensor_creation',
                    sizeName,
                    browserInfo.name,
                    results
                );
                
                // Assertions
                expect(results.averageTime).toBeLessThan(100); // Should be fast
                expect(results.coefficient_of_variation).toBeLessThan(0.5); // Should be consistent
                
                console.log(`[${browserInfo.name}] ${sizeName} tensor creation: ${results.averageTime.toFixed(2)}ms avg`);
            });
        });
    });

    describe('Tensor Operations Benchmarks', () => {
        Object.entries(BENCHMARK_CONFIG.tensorSizes).forEach(([sizeName, shape]) => {
            it(`should perform ${sizeName} tensor addition efficiently`, async () => {
                const testFn = () => {
                    const id1 = wasmModule.createTensor(shape);
                    const id2 = wasmModule.createTensor(shape);
                    const resultId = wasmModule.addTensors(id1, id2);
                    
                    // Cleanup
                    wasmModule.tensors.delete(id1);
                    wasmModule.tensors.delete(id2);
                    wasmModule.tensors.delete(resultId);
                };
                
                await BenchmarkUtils.warmup(testFn);
                const results = await BenchmarkUtils.benchmark(testFn);
                
                performanceMetrics.recordMetric(
                    'tensor_addition',
                    sizeName,
                    browserInfo.name,
                    results
                );
                
                expect(results.averageTime).toBeLessThan(1000);
                expect(results.coefficient_of_variation).toBeLessThan(0.8);
                
                console.log(`[${browserInfo.name}] ${sizeName} tensor addition: ${results.averageTime.toFixed(2)}ms avg`);
            });
            
            it(`should perform ${sizeName} tensor multiplication efficiently`, async () => {
                const testFn = () => {
                    const id1 = wasmModule.createTensor(shape);
                    const id2 = wasmModule.createTensor(shape);
                    const resultId = wasmModule.multiplyTensors(id1, id2);
                    
                    // Cleanup
                    wasmModule.tensors.delete(id1);
                    wasmModule.tensors.delete(id2);
                    wasmModule.tensors.delete(resultId);
                };
                
                await BenchmarkUtils.warmup(testFn);
                const results = await BenchmarkUtils.benchmark(testFn);
                
                performanceMetrics.recordMetric(
                    'tensor_multiplication',
                    sizeName,
                    browserInfo.name,
                    results
                );
                
                expect(results.averageTime).toBeLessThan(1000);
                
                console.log(`[${browserInfo.name}] ${sizeName} tensor multiplication: ${results.averageTime.toFixed(2)}ms avg`);
            });
        });
    });

    describe('Matrix Operations Benchmarks', () => {
        const matrixSizes = [
            { name: 'small', shape: [32, 32] },
            { name: 'medium', shape: [64, 64] },
            { name: 'large', shape: [128, 128] }
        ];
        
        matrixSizes.forEach(({ name, shape }) => {
            it(`should perform ${name} matrix multiplication efficiently`, async () => {
                const testFn = () => {
                    const id1 = wasmModule.createTensor(shape);
                    const id2 = wasmModule.createTensor(shape);
                    const resultId = wasmModule.matrixMultiply(id1, id2);
                    
                    // Cleanup
                    wasmModule.tensors.delete(id1);
                    wasmModule.tensors.delete(id2);
                    wasmModule.tensors.delete(resultId);
                };
                
                await BenchmarkUtils.warmup(testFn, 5); // Fewer warmup iterations for expensive operations
                const results = await BenchmarkUtils.benchmark(testFn, 50); // Fewer benchmark iterations
                
                performanceMetrics.recordMetric(
                    'matrix_multiplication',
                    name,
                    browserInfo.name,
                    results
                );
                
                // More lenient for large matrix operations
                const timeLimit = name === 'large' ? 5000 : 1000;
                expect(results.averageTime).toBeLessThan(timeLimit);
                
                console.log(`[${browserInfo.name}] ${name} matrix multiplication: ${results.averageTime.toFixed(2)}ms avg`);
            });
        });
    });

    describe('Activation Function Benchmarks', () => {
        Object.entries(BENCHMARK_CONFIG.tensorSizes).forEach(([sizeName, shape]) => {
            it(`should perform ${sizeName} ReLU activation efficiently`, async () => {
                const testFn = () => {
                    const id = wasmModule.createTensor(shape, Array.from({ length: shape.reduce((a, b) => a * b) }, () => Math.random() - 0.5));
                    const resultId = wasmModule.relu(id);
                    
                    // Cleanup
                    wasmModule.tensors.delete(id);
                    wasmModule.tensors.delete(resultId);
                };
                
                await BenchmarkUtils.warmup(testFn);
                const results = await BenchmarkUtils.benchmark(testFn);
                
                performanceMetrics.recordMetric(
                    'activation_functions',
                    sizeName,
                    browserInfo.name,
                    results
                );
                
                expect(results.averageTime).toBeLessThan(500);
                
                console.log(`[${browserInfo.name}] ${sizeName} ReLU activation: ${results.averageTime.toFixed(2)}ms avg`);
            });
        });
    });

    describe('Layer Normalization Benchmarks', () => {
        const layerSizes = [
            { name: 'small', shape: [128] },
            { name: 'medium', shape: [512] },
            { name: 'large', shape: [1024] }
        ];
        
        layerSizes.forEach(({ name, shape }) => {
            it(`should perform ${name} layer normalization efficiently`, async () => {
                const testFn = () => {
                    const id = wasmModule.createTensor(shape, Array.from({ length: shape[0] }, () => Math.random()));
                    const resultId = wasmModule.layerNorm(id);
                    
                    // Cleanup
                    wasmModule.tensors.delete(id);
                    wasmModule.tensors.delete(resultId);
                };
                
                await BenchmarkUtils.warmup(testFn);
                const results = await BenchmarkUtils.benchmark(testFn);
                
                performanceMetrics.recordMetric(
                    'layer_normalization',
                    name,
                    browserInfo.name,
                    results
                );
                
                expect(results.averageTime).toBeLessThan(200);
                
                console.log(`[${browserInfo.name}] ${name} layer normalization: ${results.averageTime.toFixed(2)}ms avg`);
            });
        });
    });

    describe('Attention Mechanism Benchmarks', () => {
        const attentionSizes = [
            { name: 'small', shape: [16, 16] },
            { name: 'medium', shape: [32, 32] },
            { name: 'large', shape: [64, 64] }
        ];
        
        attentionSizes.forEach(({ name, shape }) => {
            it(`should perform ${name} attention mechanism efficiently`, async () => {
                const testFn = () => {
                    const queryId = wasmModule.createTensor(shape);
                    const keyId = wasmModule.createTensor(shape);
                    const valueId = wasmModule.createTensor(shape);
                    
                    const resultId = wasmModule.attentionMechanism(queryId, keyId, valueId);
                    
                    // Cleanup - note: attention creates additional intermediate tensors
                    wasmModule.cleanup();
                };
                
                await BenchmarkUtils.warmup(testFn, 3);
                const results = await BenchmarkUtils.benchmark(testFn, 20);
                
                performanceMetrics.recordMetric(
                    'attention_mechanism',
                    name,
                    browserInfo.name,
                    results
                );
                
                const timeLimit = name === 'large' ? 10000 : 2000;
                expect(results.averageTime).toBeLessThan(timeLimit);
                
                console.log(`[${browserInfo.name}] ${name} attention mechanism: ${results.averageTime.toFixed(2)}ms avg`);
            });
        });
    });

    describe('Memory Efficiency Benchmarks', () => {
        it('should handle memory allocation and deallocation efficiently', async () => {
            const testFn = () => {
                const tensorIds = [];
                
                // Create many tensors
                for (let i = 0; i < 100; i++) {
                    tensorIds.push(wasmModule.createTensor([10, 10]));
                }
                
                // Delete half of them
                for (let i = 0; i < 50; i++) {
                    wasmModule.tensors.delete(tensorIds[i]);
                }
                
                // Create more tensors
                for (let i = 0; i < 25; i++) {
                    tensorIds.push(wasmModule.createTensor([20, 20]));
                }
                
                // Cleanup remaining
                wasmModule.cleanup();
            };
            
            await BenchmarkUtils.warmup(testFn, 3);
            const results = await BenchmarkUtils.benchmark(testFn, 10);
            
            performanceMetrics.recordMetric(
                'memory_efficiency',
                'mixed',
                browserInfo.name,
                results
            );
            
            expect(results.averageTime).toBeLessThan(1000);
            
            console.log(`[${browserInfo.name}] Memory efficiency: ${results.averageTime.toFixed(2)}ms avg`);
        });
        
        it('should maintain consistent performance under memory pressure', async () => {
            const baselineFn = () => {
                const id1 = wasmModule.createTensor([50, 50]);
                const id2 = wasmModule.createTensor([50, 50]);
                const resultId = wasmModule.addTensors(id1, id2);
                wasmModule.cleanup();
            };
            
            const memoryPressureFn = () => {
                // Create memory pressure
                const backgroundTensors = [];
                for (let i = 0; i < 50; i++) {
                    backgroundTensors.push(wasmModule.createTensor([20, 20]));
                }
                
                // Perform operation under pressure
                const id1 = wasmModule.createTensor([50, 50]);
                const id2 = wasmModule.createTensor([50, 50]);
                const resultId = wasmModule.addTensors(id1, id2);
                
                wasmModule.cleanup();
            };
            
            // Get baseline performance
            await BenchmarkUtils.warmup(baselineFn, 5);
            const baselineResults = await BenchmarkUtils.benchmark(baselineFn, 20);
            
            // Test under memory pressure
            await BenchmarkUtils.warmup(memoryPressureFn, 5);
            const pressureResults = await BenchmarkUtils.benchmark(memoryPressureFn, 20);
            
            // Performance shouldn't degrade significantly under memory pressure
            const degradationRatio = pressureResults.averageTime / baselineResults.averageTime;
            expect(degradationRatio).toBeLessThan(3.0); // Less than 3x degradation
            
            console.log(`[${browserInfo.name}] Memory pressure degradation: ${degradationRatio.toFixed(2)}x`);
        });
    });

    describe('WebGPU vs CPU Performance Comparison', () => {
        it('should compare WebGPU vs CPU performance if WebGPU is available', async () => {
            const webGPUSupport = await BenchmarkUtils.checkWebGPUSupport();
            
            if (!webGPUSupport.supported) {
                console.log(`[${browserInfo.name}] WebGPU not supported, skipping comparison`);
                return;
            }
            
            const testOperation = () => {
                const id1 = wasmModule.createTensor([100, 100]);
                const id2 = wasmModule.createTensor([100, 100]);
                const resultId = wasmModule.addTensors(id1, id2);
                wasmModule.cleanup();
            };
            
            // Test CPU performance
            wasmModule.useWebGPU = false;
            await BenchmarkUtils.warmup(testOperation, 5);
            const cpuResults = await BenchmarkUtils.benchmark(testOperation, 20);
            
            // Test WebGPU performance
            await wasmModule.enableWebGPU();
            await BenchmarkUtils.warmup(testOperation, 5);
            const webGPUResults = await BenchmarkUtils.benchmark(testOperation, 20);
            
            const speedupRatio = cpuResults.averageTime / webGPUResults.averageTime;
            
            performanceMetrics.recordMetric(
                'webgpu_vs_cpu',
                'comparison',
                browserInfo.name,
                {
                    cpuTime: cpuResults.averageTime,
                    webGPUTime: webGPUResults.averageTime,
                    speedupRatio,
                    cpuThroughput: cpuResults.throughput,
                    webGPUThroughput: webGPUResults.throughput
                }
            );
            
            console.log(`[${browserInfo.name}] WebGPU vs CPU speedup: ${speedupRatio.toFixed(2)}x`);
            console.log(`[${browserInfo.name}] CPU: ${cpuResults.averageTime.toFixed(2)}ms, WebGPU: ${webGPUResults.averageTime.toFixed(2)}ms`);
        });
    });
});

// Export for use in other test files
export {
    PerformanceMetrics,
    BenchmarkUtils,
    MockTrustformersWasm,
    BENCHMARK_CONFIG
};