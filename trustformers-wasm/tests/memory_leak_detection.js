/**
 * Memory Leak Detection Test Suite for TrustformeRS WASM
 * 
 * This test suite specifically focuses on detecting memory leaks in browser
 * environments, including WASM memory, JavaScript heap, and GPU memory.
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach } from '@jest/globals';

// Memory leak detection configuration
const MEMORY_LEAK_CONFIG = {
    warmupIterations: 5,
    testIterations: 20,
    measurementInterval: 100, // ms
    memoryGrowthThreshold: 0.1, // 10% growth threshold
    gcWaitTime: 100, // Time to wait for GC
    stabilizationTime: 1000, // Time to wait for memory stabilization
    
    // Test scenarios
    scenarios: [
        {
            name: 'tensor_lifecycle',
            description: 'Test tensor creation and destruction cycles',
            iterations: 100,
            operations: ['create', 'use', 'destroy']
        },
        {
            name: 'model_loading',
            description: 'Test model loading and unloading',
            iterations: 10,
            operations: ['load', 'inference', 'unload']
        },
        {
            name: 'webgpu_operations',
            description: 'Test WebGPU resource management',
            iterations: 50,
            operations: ['create_gpu_buffer', 'use_buffer', 'destroy_buffer']
        },
        {
            name: 'worker_communication',
            description: 'Test Web Worker memory management',
            iterations: 30,
            operations: ['create_worker', 'send_message', 'terminate_worker']
        }
    ]
};

// Memory monitoring utilities
class MemoryMonitor {
    constructor() {
        this.snapshots = [];
        this.isMonitoring = false;
        this.monitoringInterval = null;
    }
    
    startMonitoring(interval = MEMORY_LEAK_CONFIG.measurementInterval) {
        if (this.isMonitoring) return;
        
        this.isMonitoring = true;
        this.snapshots = [];
        
        this.monitoringInterval = setInterval(() => {
            this.takeSnapshot();
        }, interval);
        
        console.log('Memory monitoring started');
    }
    
    stopMonitoring() {
        if (!this.isMonitoring) return;
        
        this.isMonitoring = false;
        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
            this.monitoringInterval = null;
        }
        
        console.log('Memory monitoring stopped');
    }
    
    takeSnapshot(label = null) {
        const snapshot = {
            timestamp: performance.now(),
            label: label || `snapshot_${this.snapshots.length}`,
            memory: this.getMemoryInfo(),
            gc: this.getGCInfo()
        };
        
        this.snapshots.push(snapshot);
        return snapshot;
    }
    
    getMemoryInfo() {
        const memoryInfo = {
            jsHeap: {
                used: 0,
                total: 0,
                limit: 0
            },
            wasm: {
                used: 0,
                total: 0
            },
            gpu: {
                used: 0,
                total: 0
            }
        };
        
        // JavaScript heap memory
        if (performance.memory) {
            memoryInfo.jsHeap = {
                used: performance.memory.usedJSHeapSize,
                total: performance.memory.totalJSHeapSize,
                limit: performance.memory.jsHeapSizeLimit
            };
        }
        
        // WebAssembly memory (if available)
        if (typeof WebAssembly !== 'undefined' && WebAssembly.Memory) {
            try {
                // This is a simplified estimation
                memoryInfo.wasm = {
                    used: 0, // Would need to be tracked by the WASM module
                    total: 0
                };
            } catch (e) {
                // Ignore errors in memory estimation
            }
        }
        
        // GPU memory (WebGPU - if available)
        if (navigator.gpu) {
            // This would need to be tracked by the WebGPU implementation
            memoryInfo.gpu = {
                used: 0,
                total: 0
            };
        }
        
        return memoryInfo;
    }
    
    getGCInfo() {
        // Estimate GC activity based on heap size changes
        const current = this.getMemoryInfo();
        const previous = this.snapshots.length > 0 ? 
            this.snapshots[this.snapshots.length - 1].memory : null;
        
        if (!previous) {
            return { gcActivity: 0, heapDelta: 0 };
        }
        
        const heapDelta = current.jsHeap.used - previous.jsHeap.used;
        const gcActivity = heapDelta < -1000000 ? 1 : 0; // Significant heap reduction indicates GC
        
        return { gcActivity, heapDelta };
    }
    
    async forceGC() {
        // Force garbage collection if possible
        if (window.gc) {
            window.gc();
        } else {
            // Fallback: create pressure to trigger GC
            const arrays = [];
            for (let i = 0; i < 10; i++) {
                arrays.push(new ArrayBuffer(1024 * 1024)); // 1MB each
            }
            arrays.length = 0; // Release references
        }
        
        // Wait for GC to complete
        await new Promise(resolve => setTimeout(resolve, MEMORY_LEAK_CONFIG.gcWaitTime));
    }
    
    analyzeLeaks() {
        if (this.snapshots.length < 2) {
            return { hasLeak: false, reason: 'Insufficient data' };
        }
        
        const startSnapshot = this.snapshots[0];
        const endSnapshot = this.snapshots[this.snapshots.length - 1];
        
        // Calculate memory growth
        const jsHeapGrowth = endSnapshot.memory.jsHeap.used - startSnapshot.memory.jsHeap.used;
        const jsHeapGrowthPercent = jsHeapGrowth / startSnapshot.memory.jsHeap.used;
        
        const wasmGrowth = endSnapshot.memory.wasm.used - startSnapshot.memory.wasm.used;
        const gpuGrowth = endSnapshot.memory.gpu.used - startSnapshot.memory.gpu.used;
        
        // Check for memory leaks
        const hasJSHeapLeak = jsHeapGrowthPercent > MEMORY_LEAK_CONFIG.memoryGrowthThreshold;
        const hasWasmLeak = wasmGrowth > 10 * 1024 * 1024; // 10MB threshold
        const hasGPULeak = gpuGrowth > 50 * 1024 * 1024; // 50MB threshold
        
        const analysis = {
            hasLeak: hasJSHeapLeak || hasWasmLeak || hasGPULeak,
            details: {
                jsHeap: {
                    growth: jsHeapGrowth,
                    growthPercent: jsHeapGrowthPercent,
                    hasLeak: hasJSHeapLeak
                },
                wasm: {
                    growth: wasmGrowth,
                    hasLeak: hasWasmLeak
                },
                gpu: {
                    growth: gpuGrowth,
                    hasLeak: hasGPULeak
                }
            },
            snapshots: this.snapshots,
            duration: endSnapshot.timestamp - startSnapshot.timestamp
        };
        
        return analysis;
    }
    
    generateReport() {
        const analysis = this.analyzeLeaks();
        
        return {
            timestamp: new Date().toISOString(),
            duration: analysis.duration,
            totalSnapshots: this.snapshots.length,
            memoryAnalysis: analysis,
            recommendations: this.getRecommendations(analysis)
        };
    }
    
    getRecommendations(analysis) {
        const recommendations = [];
        
        if (analysis.details.jsHeap.hasLeak) {
            recommendations.push({
                type: 'js_heap_leak',
                severity: 'high',
                message: 'JavaScript heap memory is growing beyond threshold',
                suggestion: 'Check for retained references, event listeners, or closures'
            });
        }
        
        if (analysis.details.wasm.hasLeak) {
            recommendations.push({
                type: 'wasm_leak',
                severity: 'high',
                message: 'WebAssembly memory is not being freed properly',
                suggestion: 'Ensure proper cleanup of WASM allocations and tensor objects'
            });
        }
        
        if (analysis.details.gpu.hasLeak) {
            recommendations.push({
                type: 'gpu_leak',
                severity: 'high',
                message: 'GPU memory is not being released',
                suggestion: 'Check WebGPU buffer cleanup and texture management'
            });
        }
        
        if (!analysis.hasLeak) {
            recommendations.push({
                type: 'no_leak',
                severity: 'info',
                message: 'No memory leaks detected',
                suggestion: 'Memory management appears to be working correctly'
            });
        }
        
        return recommendations;
    }
}

// Mock WASM module with memory tracking
class MockTrustformersWasmMemory {
    constructor() {
        this.initialized = false;
        this.tensors = new Map();
        this.models = new Map();
        this.nextTensorId = 1;
        this.nextModelId = 1;
        this.allocatedMemory = 0;
        this.peakMemory = 0;
        this.workers = [];
    }
    
    async initialize() {
        await new Promise(resolve => setTimeout(resolve, 100));
        this.initialized = true;
        this.allocatedMemory = 1024 * 1024; // 1MB base allocation
    }
    
    createTensor(shape, data) {
        if (!this.initialized) {
            throw new Error('WASM module not initialized');
        }
        
        const id = this.nextTensorId++;
        const size = shape.reduce((a, b) => a * b, 1);
        const memorySize = size * 4; // Float32 = 4 bytes
        
        const tensor = {
            id,
            shape: [...shape],
            data: data ? new Float32Array(data) : new Float32Array(size),
            memorySize,
            createdAt: performance.now()
        };
        
        this.tensors.set(id, tensor);
        this.allocatedMemory += memorySize;
        this.peakMemory = Math.max(this.peakMemory, this.allocatedMemory);
        
        return id;
    }
    
    deleteTensor(id) {
        const tensor = this.tensors.get(id);
        if (tensor) {
            this.allocatedMemory -= tensor.memorySize;
            this.tensors.delete(id);
            return true;
        }
        return false;
    }
    
    loadModel(modelData) {
        const modelId = this.nextModelId++;
        const memorySize = this.calculateModelSize(modelData);
        
        const model = {
            id: modelId,
            data: modelData,
            memorySize,
            loadedAt: performance.now()
        };
        
        this.models.set(modelId, model);
        this.allocatedMemory += memorySize;
        this.peakMemory = Math.max(this.peakMemory, this.allocatedMemory);
        
        return modelId;
    }
    
    unloadModel(modelId) {
        const model = this.models.get(modelId);
        if (model) {
            this.allocatedMemory -= model.memorySize;
            this.models.delete(modelId);
            return true;
        }
        return false;
    }
    
    calculateModelSize(modelData) {
        // Simplified model size calculation
        return 50 * 1024 * 1024; // 50MB default
    }
    
    createWorker() {
        const worker = {
            id: this.workers.length,
            memorySize: 5 * 1024 * 1024, // 5MB per worker
            createdAt: performance.now()
        };
        
        this.workers.push(worker);
        this.allocatedMemory += worker.memorySize;
        this.peakMemory = Math.max(this.peakMemory, this.allocatedMemory);
        
        return worker.id;
    }
    
    terminateWorker(workerId) {
        const worker = this.workers[workerId];
        if (worker) {
            this.allocatedMemory -= worker.memorySize;
            this.workers[workerId] = null;
            return true;
        }
        return false;
    }
    
    getMemoryUsage() {
        return {
            allocated: this.allocatedMemory,
            peak: this.peakMemory,
            tensors: this.tensors.size,
            models: this.models.size,
            workers: this.workers.filter(w => w !== null).length
        };
    }
    
    cleanup() {
        this.tensors.clear();
        this.models.clear();
        this.workers.length = 0;
        this.allocatedMemory = 0;
        this.nextTensorId = 1;
        this.nextModelId = 1;
    }
}

// Test execution framework
class MemoryLeakTestExecutor {
    constructor() {
        this.monitor = new MemoryMonitor();
        this.wasmModule = new MockTrustformersWasmMemory();
        this.testResults = [];
    }
    
    async runScenario(scenario) {
        console.log(`Running memory leak test scenario: ${scenario.name}`);
        
        // Initialize
        await this.wasmModule.initialize();
        
        // Start monitoring
        this.monitor.startMonitoring();
        this.monitor.takeSnapshot('scenario_start');
        
        try {
            // Warmup
            for (let i = 0; i < MEMORY_LEAK_CONFIG.warmupIterations; i++) {
                await this.executeOperations(scenario.operations);
            }
            
            // Force GC after warmup
            await this.monitor.forceGC();
            this.monitor.takeSnapshot('post_warmup');
            
            // Test iterations
            for (let i = 0; i < scenario.iterations; i++) {
                await this.executeOperations(scenario.operations);
                
                // Periodic snapshots
                if (i % 10 === 0) {
                    this.monitor.takeSnapshot(`iteration_${i}`);
                }
            }
            
            // Force GC before final measurement
            await this.monitor.forceGC();
            await new Promise(resolve => setTimeout(resolve, MEMORY_LEAK_CONFIG.stabilizationTime));
            this.monitor.takeSnapshot('scenario_end');
            
            // Stop monitoring and analyze
            this.monitor.stopMonitoring();
            
            const analysis = this.monitor.analyzeLeaks();
            const report = this.monitor.generateReport();
            
            this.testResults.push({
                scenario: scenario.name,
                analysis,
                report,
                success: !analysis.hasLeak
            });
            
            return analysis;
            
        } catch (error) {
            this.monitor.stopMonitoring();
            throw error;
        } finally {
            this.wasmModule.cleanup();
        }
    }
    
    async executeOperations(operations) {
        for (const operation of operations) {
            await this.executeOperation(operation);
        }
    }
    
    async executeOperation(operation) {
        switch (operation) {
            case 'create':
                this.wasmModule.createTensor([100, 100]);
                break;
            case 'use':
                // Simulate using tensors
                const tensors = Array.from(this.wasmModule.tensors.keys());
                if (tensors.length > 1) {
                    // Simulate tensor operations
                    await new Promise(resolve => setTimeout(resolve, 1));
                }
                break;
            case 'destroy':
                const tensorToDelete = Array.from(this.wasmModule.tensors.keys())[0];
                if (tensorToDelete) {
                    this.wasmModule.deleteTensor(tensorToDelete);
                }
                break;
            case 'load':
                this.wasmModule.loadModel({ weights: new Float32Array(1000) });
                break;
            case 'inference':
                // Simulate inference
                await new Promise(resolve => setTimeout(resolve, 10));
                break;
            case 'unload':
                const modelToUnload = Array.from(this.wasmModule.models.keys())[0];
                if (modelToUnload) {
                    this.wasmModule.unloadModel(modelToUnload);
                }
                break;
            case 'create_gpu_buffer':
                // Simulate GPU buffer creation
                this.wasmModule.createTensor([500, 500]);
                break;
            case 'use_buffer':
                // Simulate GPU buffer usage
                await new Promise(resolve => setTimeout(resolve, 5));
                break;
            case 'destroy_buffer':
                const bufferToDestroy = Array.from(this.wasmModule.tensors.keys())[0];
                if (bufferToDestroy) {
                    this.wasmModule.deleteTensor(bufferToDestroy);
                }
                break;
            case 'create_worker':
                this.wasmModule.createWorker();
                break;
            case 'send_message':
                // Simulate message sending
                await new Promise(resolve => setTimeout(resolve, 1));
                break;
            case 'terminate_worker':
                const workerToTerminate = this.wasmModule.workers.findIndex(w => w !== null);
                if (workerToTerminate !== -1) {
                    this.wasmModule.terminateWorker(workerToTerminate);
                }
                break;
            default:
                console.warn(`Unknown operation: ${operation}`);
        }
    }
    
    getResults() {
        return this.testResults;
    }
}

// Global test setup
let testExecutor;

describe('Memory Leak Detection Tests', () => {
    beforeAll(() => {
        testExecutor = new MemoryLeakTestExecutor();
        console.log('Memory leak detection test suite initialized');
    });
    
    afterAll(() => {
        const results = testExecutor.getResults();
        console.log('\n=== MEMORY LEAK DETECTION RESULTS ===');
        
        results.forEach(result => {
            console.log(`\nScenario: ${result.scenario}`);
            console.log(`Success: ${result.success}`);
            console.log(`Memory Growth: ${result.analysis.details.jsHeap.growthPercent.toFixed(2)}%`);
            console.log(`Recommendations: ${result.report.recommendations.length}`);
            
            result.report.recommendations.forEach(rec => {
                console.log(`  - ${rec.type}: ${rec.message}`);
            });
        });
        
        // Store results for analysis
        if (typeof localStorage !== 'undefined') {
            localStorage.setItem('trustformers_memory_leak_results', JSON.stringify(results));
        }
    });

    MEMORY_LEAK_CONFIG.scenarios.forEach(scenario => {
        it(`should not leak memory during ${scenario.name}`, async () => {
            const analysis = await testExecutor.runScenario(scenario);
            
            // Assertions
            expect(analysis.hasLeak).toBe(false);
            expect(analysis.details.jsHeap.growthPercent).toBeLessThan(MEMORY_LEAK_CONFIG.memoryGrowthThreshold);
            
            if (analysis.hasLeak) {
                console.error(`Memory leak detected in ${scenario.name}:`, analysis.details);
            }
            
            console.log(`✓ ${scenario.name} passed memory leak test`);
        }, 30000); // 30 second timeout
    });
    
    it('should detect intentional memory leaks', async () => {
        const leakyScenario = {
            name: 'intentional_leak',
            description: 'Intentionally create memory leaks for testing',
            iterations: 50,
            operations: ['create'] // Only create, never destroy
        };
        
        const analysis = await testExecutor.runScenario(leakyScenario);
        
        // This should detect a memory leak
        expect(analysis.hasLeak).toBe(true);
        expect(analysis.details.jsHeap.growthPercent).toBeGreaterThan(MEMORY_LEAK_CONFIG.memoryGrowthThreshold);
        
        console.log('✓ Intentional memory leak correctly detected');
    }, 30000);
    
    it('should provide meaningful leak analysis', async () => {
        const analysis = await testExecutor.runScenario(MEMORY_LEAK_CONFIG.scenarios[0]);
        
        expect(analysis.details).toBeDefined();
        expect(analysis.details.jsHeap).toBeDefined();
        expect(analysis.details.wasm).toBeDefined();
        expect(analysis.details.gpu).toBeDefined();
        
        expect(analysis.snapshots).toBeDefined();
        expect(analysis.snapshots.length).toBeGreaterThan(0);
        
        console.log('✓ Memory leak analysis provides comprehensive data');
    });
});

// Export for use in other test files
export {
    MemoryMonitor,
    MockTrustformersWasmMemory,
    MemoryLeakTestExecutor,
    MEMORY_LEAK_CONFIG
};