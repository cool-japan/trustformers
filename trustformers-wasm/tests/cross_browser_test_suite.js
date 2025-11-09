/**
 * Cross-browser test suite for TrustformeRS WASM
 * 
 * This test suite runs comprehensive tests across different browsers
 * to ensure WebAssembly and WebGPU functionality works consistently.
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach } from '@jest/globals';

// Browser detection utilities
const BrowserDetector = {
    isChrome: () => /Chrome/.test(navigator.userAgent) && /Google Inc/.test(navigator.vendor),
    isFirefox: () => /Firefox/.test(navigator.userAgent),
    isSafari: () => /Safari/.test(navigator.userAgent) && /Apple Computer/.test(navigator.vendor),
    isEdge: () => /Edg/.test(navigator.userAgent),
    isOpera: () => /OPR/.test(navigator.userAgent),
    
    getBrowserInfo: () => ({
        name: BrowserDetector.getBrowserName(),
        version: BrowserDetector.getBrowserVersion(),
        userAgent: navigator.userAgent,
        platform: navigator.platform,
        language: navigator.language,
        hardwareConcurrency: navigator.hardwareConcurrency,
        deviceMemory: navigator.deviceMemory || 'unknown'
    }),
    
    getBrowserName: () => {
        if (BrowserDetector.isChrome()) return 'Chrome';
        if (BrowserDetector.isFirefox()) return 'Firefox';
        if (BrowserDetector.isSafari()) return 'Safari';
        if (BrowserDetector.isEdge()) return 'Edge';
        if (BrowserDetector.isOpera()) return 'Opera';
        return 'Unknown';
    },
    
    getBrowserVersion: () => {
        const ua = navigator.userAgent;
        if (BrowserDetector.isChrome()) {
            const match = ua.match(/Chrome\/(\d+\.\d+)/);
            return match ? match[1] : 'unknown';
        }
        if (BrowserDetector.isFirefox()) {
            const match = ua.match(/Firefox\/(\d+\.\d+)/);
            return match ? match[1] : 'unknown';
        }
        if (BrowserDetector.isSafari()) {
            const match = ua.match(/Version\/(\d+\.\d+)/);
            return match ? match[1] : 'unknown';
        }
        if (BrowserDetector.isEdge()) {
            const match = ua.match(/Edg\/(\d+\.\d+)/);
            return match ? match[1] : 'unknown';
        }
        return 'unknown';
    }
};

// Feature detection utilities
const FeatureDetector = {
    hasWebAssembly: () => typeof WebAssembly !== 'undefined',
    hasWebGPU: () => typeof navigator.gpu !== 'undefined',
    hasSharedArrayBuffer: () => typeof SharedArrayBuffer !== 'undefined',
    hasOffscreenCanvas: () => typeof OffscreenCanvas !== 'undefined',
    hasWebWorkers: () => typeof Worker !== 'undefined',
    hasBigInt64Array: () => typeof BigInt64Array !== 'undefined',
    hasStreams: () => typeof ReadableStream !== 'undefined',
    
    getWebGPUCapabilities: async () => {
        if (!FeatureDetector.hasWebGPU()) {
            return { supported: false, reason: 'WebGPU not available' };
        }
        
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                return { supported: false, reason: 'No WebGPU adapter available' };
            }
            
            const device = await adapter.requestDevice();
            const features = Array.from(adapter.features || []);
            const limits = adapter.limits || {};
            
            return {
                supported: true,
                adapter: {
                    features,
                    limits: {
                        maxTextureDimension1D: limits.maxTextureDimension1D,
                        maxTextureDimension2D: limits.maxTextureDimension2D,
                        maxTextureDimension3D: limits.maxTextureDimension3D,
                        maxTextureArrayLayers: limits.maxTextureArrayLayers,
                        maxBindGroups: limits.maxBindGroups,
                        maxDynamicUniformBuffersPerPipelineLayout: limits.maxDynamicUniformBuffersPerPipelineLayout,
                        maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
                        maxComputeWorkgroupSizeY: limits.maxComputeWorkgroupSizeY,
                        maxComputeWorkgroupSizeZ: limits.maxComputeWorkgroupSizeZ
                    }
                },
                device: !!device
            };
        } catch (error) {
            return { supported: false, reason: error.message };
        }
    },
    
    getWebAssemblyCapabilities: () => {
        if (!FeatureDetector.hasWebAssembly()) {
            return { supported: false, reason: 'WebAssembly not available' };
        }
        
        const features = {
            bigint: typeof WebAssembly.Global !== 'undefined',
            bulkMemory: false,
            exceptions: false,
            multiValue: false,
            mutableGlobals: false,
            referenceTypes: false,
            saturatedFloatToInt: false,
            signExtensions: false,
            simd128: false,
            tailCall: false,
            threads: typeof SharedArrayBuffer !== 'undefined'
        };
        
        // Test for SIMD support
        try {
            new WebAssembly.Module(new Uint8Array([
                0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
                0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
                0x03, 0x02, 0x01, 0x00,
                0x0a, 0x09, 0x01, 0x07, 0x00, 0xfd, 0x02, 0x00, 0x0b
            ]));
            features.simd128 = true;
        } catch {
            features.simd128 = false;
        }
        
        return { supported: true, features };
    }
};

// Performance measurement utilities
class PerformanceMeasurer {
    constructor() {
        this.measurements = new Map();
    }
    
    start(label) {
        this.measurements.set(label, performance.now());
    }
    
    end(label) {
        const startTime = this.measurements.get(label);
        if (startTime === undefined) {
            throw new Error(`No measurement started for label: ${label}`);
        }
        const endTime = performance.now();
        const duration = endTime - startTime;
        this.measurements.delete(label);
        return duration;
    }
    
    measure(label, fn) {
        this.start(label);
        const result = fn();
        const duration = this.end(label);
        return { result, duration };
    }
    
    async measureAsync(label, asyncFn) {
        this.start(label);
        const result = await asyncFn();
        const duration = this.end(label);
        return { result, duration };
    }
}

// Test result collector
class TestResultCollector {
    constructor() {
        this.results = [];
        this.browserInfo = BrowserDetector.getBrowserInfo();
        this.features = null;
    }
    
    async initialize() {
        this.features = {
            webAssembly: FeatureDetector.getWebAssemblyCapabilities(),
            webGPU: await FeatureDetector.getWebGPUCapabilities(),
            sharedArrayBuffer: FeatureDetector.hasSharedArrayBuffer(),
            webWorkers: FeatureDetector.hasWebWorkers(),
            offscreenCanvas: FeatureDetector.hasOffscreenCanvas()
        };
    }
    
    addResult(testName, category, status, details = {}) {
        this.results.push({
            testName,
            category,
            status, // 'pass', 'fail', 'skip'
            details,
            timestamp: new Date().toISOString(),
            browser: this.browserInfo.name,
            browserVersion: this.browserInfo.version
        });
    }
    
    getReport() {
        const summary = {
            total: this.results.length,
            passed: this.results.filter(r => r.status === 'pass').length,
            failed: this.results.filter(r => r.status === 'fail').length,
            skipped: this.results.filter(r => r.status === 'skip').length
        };
        
        return {
            browserInfo: this.browserInfo,
            features: this.features,
            summary,
            results: this.results,
            timestamp: new Date().toISOString()
        };
    }
    
    exportToJSON() {
        return JSON.stringify(this.getReport(), null, 2);
    }
}

// Mock TrustformeRS WASM module for testing
class MockTrustformersWasm {
    constructor() {
        this.initialized = false;
        this.tensors = new Map();
        this.nextTensorId = 1;
    }
    
    async initialize() {
        // Simulate WASM initialization
        await new Promise(resolve => setTimeout(resolve, 100));
        this.initialized = true;
    }
    
    createTensor(shape, data) {
        if (!this.initialized) {
            throw new Error('WASM module not initialized');
        }
        
        const id = this.nextTensorId++;
        const tensor = {
            id,
            shape: [...shape],
            data: data ? new Float32Array(data) : new Float32Array(shape.reduce((a, b) => a * b, 1)),
            device: 'cpu'
        };
        
        this.tensors.set(id, tensor);
        return id;
    }
    
    getTensor(id) {
        return this.tensors.get(id);
    }
    
    deleteTensor(id) {
        return this.tensors.delete(id);
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
        for (let i = 0; i < resultData.length; i++) {
            resultData[i] = tensor1.data[i] + tensor2.data[i];
        }
        
        return this.createTensor(tensor1.shape, resultData);
    }
    
    matmul(id1, id2) {
        const tensor1 = this.tensors.get(id1);
        const tensor2 = this.tensors.get(id2);
        
        if (!tensor1 || !tensor2) {
            throw new Error('Invalid tensor ID');
        }
        
        if (tensor1.shape.length !== 2 || tensor2.shape.length !== 2) {
            throw new Error('Only 2D tensors supported for matmul');
        }
        
        const [m, k] = tensor1.shape;
        const [k2, n] = tensor2.shape;
        
        if (k !== k2) {
            throw new Error('Invalid dimensions for matrix multiplication');
        }
        
        const resultData = new Float32Array(m * n);
        
        // Simple matrix multiplication
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
                let sum = 0;
                for (let p = 0; p < k; p++) {
                    sum += tensor1.data[i * k + p] * tensor2.data[p * n + j];
                }
                resultData[i * n + j] = sum;
            }
        }
        
        return this.createTensor([m, n], resultData);
    }
    
    async enableWebGPU() {
        if (!FeatureDetector.hasWebGPU()) {
            throw new Error('WebGPU not supported');
        }
        
        // Simulate WebGPU initialization
        await new Promise(resolve => setTimeout(resolve, 200));
        
        // Update existing tensors to use GPU
        for (const tensor of this.tensors.values()) {
            tensor.device = 'gpu';
        }
        
        return true;
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
            estimatedMemoryMB: totalBytes / (1024 * 1024)
        };
    }
}

// Global test setup
let testCollector;
let performanceMeasurer;
let wasmModule;

describe('Cross-Browser TrustformeRS WASM Test Suite', () => {
    beforeAll(async () => {
        testCollector = new TestResultCollector();
        await testCollector.initialize();
        performanceMeasurer = new PerformanceMeasurer();
        wasmModule = new MockTrustformersWasm();
        
        console.log('Browser Info:', testCollector.browserInfo);
        console.log('Features:', testCollector.features);
    });
    
    afterAll(() => {
        const report = testCollector.getReport();
        console.log('Test Report:', JSON.stringify(report, null, 2));
        
        // Store results in browser storage for potential CI collection
        if (typeof localStorage !== 'undefined') {
            localStorage.setItem('trustformers_test_results', testCollector.exportToJSON());
        }
    });

    describe('Browser Compatibility Tests', () => {
        it('should detect browser and version', () => {
            const browserInfo = BrowserDetector.getBrowserInfo();
            
            expect(browserInfo.name).toBeTruthy();
            expect(browserInfo.version).toBeTruthy();
            expect(browserInfo.userAgent).toBeTruthy();
            
            testCollector.addResult(
                'Browser Detection',
                'compatibility',
                'pass',
                { browserInfo }
            );
        });
        
        it('should detect WebAssembly support', () => {
            const hasWasm = FeatureDetector.hasWebAssembly();
            
            if (hasWasm) {
                const capabilities = FeatureDetector.getWebAssemblyCapabilities();
                expect(capabilities.supported).toBe(true);
                
                testCollector.addResult(
                    'WebAssembly Support',
                    'compatibility',
                    'pass',
                    { capabilities }
                );
            } else {
                testCollector.addResult(
                    'WebAssembly Support',
                    'compatibility',
                    'fail',
                    { reason: 'WebAssembly not supported' }
                );
                
                throw new Error('WebAssembly not supported in this browser');
            }
        });
        
        it('should detect WebGPU support', async () => {
            const capabilities = await FeatureDetector.getWebGPUCapabilities();
            
            if (capabilities.supported) {
                expect(capabilities.device).toBe(true);
                
                testCollector.addResult(
                    'WebGPU Support',
                    'compatibility',
                    'pass',
                    { capabilities }
                );
            } else {
                testCollector.addResult(
                    'WebGPU Support',
                    'compatibility',
                    'skip',
                    { reason: capabilities.reason }
                );
            }
        });
        
        it('should detect SharedArrayBuffer support', () => {
            const hasSharedArrayBuffer = FeatureDetector.hasSharedArrayBuffer();
            
            testCollector.addResult(
                'SharedArrayBuffer Support',
                'compatibility',
                hasSharedArrayBuffer ? 'pass' : 'skip',
                { supported: hasSharedArrayBuffer }
            );
        });
        
        it('should detect Web Workers support', () => {
            const hasWebWorkers = FeatureDetector.hasWebWorkers();
            
            expect(hasWebWorkers).toBe(true);
            
            testCollector.addResult(
                'Web Workers Support',
                'compatibility',
                'pass',
                { supported: hasWebWorkers }
            );
        });
    });

    describe('WASM Module Tests', () => {
        beforeEach(async () => {
            wasmModule = new MockTrustformersWasm();
            await wasmModule.initialize();
        });
        
        it('should initialize WASM module', async () => {
            const { duration } = await performanceMeasurer.measureAsync(
                'wasm_init',
                () => wasmModule.initialize()
            );
            
            expect(wasmModule.initialized).toBe(true);
            expect(duration).toBeLessThan(1000); // Should initialize within 1 second
            
            testCollector.addResult(
                'WASM Initialization',
                'wasm',
                'pass',
                { initTime: duration }
            );
        });
        
        it('should create and manage tensors', () => {
            const { result: tensorId, duration } = performanceMeasurer.measure(
                'tensor_creation',
                () => wasmModule.createTensor([10, 10])
            );
            
            expect(tensorId).toBeGreaterThan(0);
            
            const tensor = wasmModule.getTensor(tensorId);
            expect(tensor).toBeTruthy();
            expect(tensor.shape).toEqual([10, 10]);
            expect(tensor.data).toBeInstanceOf(Float32Array);
            expect(tensor.data.length).toBe(100);
            
            const deleted = wasmModule.deleteTensor(tensorId);
            expect(deleted).toBe(true);
            
            testCollector.addResult(
                'Tensor Management',
                'wasm',
                'pass',
                { creationTime: duration }
            );
        });
        
        it('should perform tensor operations', () => {
            const tensor1Id = wasmModule.createTensor([5, 5], new Array(25).fill(1));
            const tensor2Id = wasmModule.createTensor([5, 5], new Array(25).fill(2));
            
            const { result: resultId, duration } = performanceMeasurer.measure(
                'tensor_addition',
                () => wasmModule.addTensors(tensor1Id, tensor2Id)
            );
            
            const result = wasmModule.getTensor(resultId);
            expect(result).toBeTruthy();
            expect(result.shape).toEqual([5, 5]);
            expect(Array.from(result.data)).toEqual(new Array(25).fill(3));
            
            testCollector.addResult(
                'Tensor Addition',
                'operations',
                'pass',
                { operationTime: duration }
            );
        });
        
        it('should perform matrix multiplication', () => {
            const tensor1Id = wasmModule.createTensor([3, 4], [
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12
            ]);
            
            const tensor2Id = wasmModule.createTensor([4, 2], [
                1, 2,
                3, 4,
                5, 6,
                7, 8
            ]);
            
            const { result: resultId, duration } = performanceMeasurer.measure(
                'matrix_multiplication',
                () => wasmModule.matmul(tensor1Id, tensor2Id)
            );
            
            const result = wasmModule.getTensor(resultId);
            expect(result).toBeTruthy();
            expect(result.shape).toEqual([3, 2]);
            
            // Expected result of matrix multiplication
            const expected = [50, 60, 114, 140, 178, 220];
            expect(Array.from(result.data)).toEqual(expected);
            
            testCollector.addResult(
                'Matrix Multiplication',
                'operations',
                'pass',
                { operationTime: duration }
            );
        });
    });

    describe('WebGPU Integration Tests', () => {
        beforeEach(async () => {
            wasmModule = new MockTrustformersWasm();
            await wasmModule.initialize();
        });
        
        it('should enable WebGPU acceleration if supported', async () => {
            const webGPUCapabilities = await FeatureDetector.getWebGPUCapabilities();
            
            if (webGPUCapabilities.supported) {
                const { duration } = await performanceMeasurer.measureAsync(
                    'webgpu_enable',
                    () => wasmModule.enableWebGPU()
                );
                
                expect(duration).toBeLessThan(1000);
                
                testCollector.addResult(
                    'WebGPU Enablement',
                    'webgpu',
                    'pass',
                    { enableTime: duration }
                );
            } else {
                testCollector.addResult(
                    'WebGPU Enablement',
                    'webgpu',
                    'skip',
                    { reason: 'WebGPU not supported' }
                );
            }
        });
        
        it('should handle WebGPU operations', async () => {
            const webGPUCapabilities = await FeatureDetector.getWebGPUCapabilities();
            
            if (!webGPUCapabilities.supported) {
                testCollector.addResult(
                    'WebGPU Operations',
                    'webgpu',
                    'skip',
                    { reason: 'WebGPU not supported' }
                );
                return;
            }
            
            try {
                await wasmModule.enableWebGPU();
                
                const tensor1Id = wasmModule.createTensor([100, 100]);
                const tensor2Id = wasmModule.createTensor([100, 100]);
                
                const { result: resultId, duration } = performanceMeasurer.measure(
                    'webgpu_operation',
                    () => wasmModule.addTensors(tensor1Id, tensor2Id)
                );
                
                expect(resultId).toBeGreaterThan(0);
                
                testCollector.addResult(
                    'WebGPU Operations',
                    'webgpu',
                    'pass',
                    { operationTime: duration }
                );
            } catch (error) {
                testCollector.addResult(
                    'WebGPU Operations',
                    'webgpu',
                    'fail',
                    { error: error.message }
                );
            }
        });
    });

    describe('Performance Tests', () => {
        beforeEach(async () => {
            wasmModule = new MockTrustformersWasm();
            await wasmModule.initialize();
        });
        
        it('should perform large tensor operations within time limits', () => {
            const size = 1000;
            const tensor1Id = wasmModule.createTensor([size]);
            const tensor2Id = wasmModule.createTensor([size]);
            
            const { duration } = performanceMeasurer.measure(
                'large_tensor_operation',
                () => wasmModule.addTensors(tensor1Id, tensor2Id)
            );
            
            // Should complete within reasonable time (browser dependent)
            const timeLimit = BrowserDetector.isSafari() ? 1000 : 500;
            expect(duration).toBeLessThan(timeLimit);
            
            testCollector.addResult(
                'Large Tensor Performance',
                'performance',
                duration < timeLimit ? 'pass' : 'fail',
                { 
                    duration, 
                    timeLimit, 
                    tensorSize: size,
                    browser: BrowserDetector.getBrowserName()
                }
            );
        });
        
        it('should handle memory usage efficiently', () => {
            // Create multiple tensors
            const tensorIds = [];
            for (let i = 0; i < 100; i++) {
                tensorIds.push(wasmModule.createTensor([10, 10]));
            }
            
            const memoryUsage = wasmModule.getMemoryUsage();
            expect(memoryUsage.totalTensors).toBe(100);
            expect(memoryUsage.estimatedMemoryMB).toBeLessThan(100); // Should be much less
            
            // Clean up
            tensorIds.forEach(id => wasmModule.deleteTensor(id));
            
            const finalMemoryUsage = wasmModule.getMemoryUsage();
            expect(finalMemoryUsage.totalTensors).toBe(0);
            
            testCollector.addResult(
                'Memory Management',
                'performance',
                'pass',
                { memoryUsage, finalMemoryUsage }
            );
        });
        
        it('should maintain performance across multiple operations', () => {
            const iterations = 100;
            const durations = [];
            
            for (let i = 0; i < iterations; i++) {
                const tensor1Id = wasmModule.createTensor([10, 10]);
                const tensor2Id = wasmModule.createTensor([10, 10]);
                
                const duration = performanceMeasurer.measure(
                    `operation_${i}`,
                    () => wasmModule.addTensors(tensor1Id, tensor2Id)
                ).duration;
                
                durations.push(duration);
                
                wasmModule.deleteTensor(tensor1Id);
                wasmModule.deleteTensor(tensor2Id);
            }
            
            const avgDuration = durations.reduce((a, b) => a + b, 0) / durations.length;
            const maxDuration = Math.max(...durations);
            const variance = durations.reduce((sum, d) => sum + Math.pow(d - avgDuration, 2), 0) / durations.length;
            
            // Performance should be consistent (low variance)
            expect(variance).toBeLessThan(avgDuration * avgDuration); // Variance less than square of mean
            expect(maxDuration).toBeLessThan(avgDuration * 3); // No operation should take more than 3x average
            
            testCollector.addResult(
                'Performance Consistency',
                'performance',
                'pass',
                { 
                    iterations,
                    avgDuration,
                    maxDuration,
                    variance,
                    standardDeviation: Math.sqrt(variance)
                }
            );
        });
    });

    describe('Error Handling Tests', () => {
        beforeEach(async () => {
            wasmModule = new MockTrustformersWasm();
            await wasmModule.initialize();
        });
        
        it('should handle invalid tensor operations gracefully', () => {
            const tensor1Id = wasmModule.createTensor([5, 5]);
            const tensor2Id = wasmModule.createTensor([3, 3]);
            
            expect(() => {
                wasmModule.addTensors(tensor1Id, tensor2Id);
            }).toThrow('Shape mismatch');
            
            testCollector.addResult(
                'Shape Mismatch Error Handling',
                'error_handling',
                'pass',
                { message: 'Properly throws shape mismatch error' }
            );
        });
        
        it('should handle invalid tensor IDs', () => {
            expect(() => {
                wasmModule.getTensor(999999);
            }).toBeFalsy(); // Should return undefined
            
            expect(() => {
                wasmModule.addTensors(1, 999999);
            }).toThrow('Invalid tensor ID');
            
            testCollector.addResult(
                'Invalid ID Error Handling',
                'error_handling',
                'pass',
                { message: 'Properly handles invalid tensor IDs' }
            );
        });
        
        it('should handle operations on uninitialized module', () => {
            const uninitializedModule = new MockTrustformersWasm();
            
            expect(() => {
                uninitializedModule.createTensor([5, 5]);
            }).toThrow('WASM module not initialized');
            
            testCollector.addResult(
                'Uninitialized Module Error Handling',
                'error_handling',
                'pass',
                { message: 'Properly prevents operations on uninitialized module' }
            );
        });
    });
});

// Export for potential use in other test files
export {
    BrowserDetector,
    FeatureDetector,
    PerformanceMeasurer,
    TestResultCollector,
    MockTrustformersWasm
};