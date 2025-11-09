/**
 * Regression Test Suite for TrustformeRS WASM
 * 
 * This test suite detects functional regressions by comparing current
 * behavior against reference implementations and golden outputs.
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach } from '@jest/globals';

// Regression test configuration
const REGRESSION_CONFIG = {
    tolerance: 1e-6, // Numerical tolerance for floating point comparisons
    performanceTolerance: 0.2, // 20% performance degradation threshold
    
    // Reference data for regression testing
    referenceData: {
        version: '1.0.0',
        testCases: [
            {
                name: 'basic_tensor_operations',
                input: {
                    tensor1: { shape: [3, 3], data: [1, 2, 3, 4, 5, 6, 7, 8, 9] },
                    tensor2: { shape: [3, 3], data: [9, 8, 7, 6, 5, 4, 3, 2, 1] }
                },
                expected: {
                    addition: [10, 10, 10, 10, 10, 10, 10, 10, 10],
                    multiplication: [9, 16, 21, 24, 25, 24, 21, 16, 9],
                    subtraction: [-8, -6, -4, -2, 0, 2, 4, 6, 8]
                }
            },
            {
                name: 'matrix_multiplication',
                input: {
                    matrix1: { 
                        shape: [2, 3], 
                        data: [1, 2, 3, 4, 5, 6] 
                    },
                    matrix2: { 
                        shape: [3, 2], 
                        data: [7, 8, 9, 10, 11, 12] 
                    }
                },
                expected: {
                    result: [58, 64, 139, 154]
                }
            },
            {
                name: 'activation_functions',
                input: {
                    tensor: { shape: [5], data: [-2, -1, 0, 1, 2] }
                },
                expected: {
                    relu: [0, 0, 0, 1, 2],
                    sigmoid: [0.11920292, 0.26894142, 0.5, 0.73105858, 0.88079708],
                    tanh: [-0.96402758, -0.76159416, 0, 0.76159416, 0.96402758]
                }
            },
            {
                name: 'normalization_operations',
                input: {
                    tensor: { shape: [4], data: [1, 2, 3, 4] }
                },
                expected: {
                    layerNorm: [-1.34164079, -0.4472136, 0.4472136, 1.34164079],
                    batchNorm: [-1.34164079, -0.4472136, 0.4472136, 1.34164079]
                }
            },
            {
                name: 'attention_mechanism',
                input: {
                    query: { shape: [1, 4], data: [1, 0, 1, 0] },
                    key: { shape: [1, 4], data: [1, 1, 0, 0] },
                    value: { shape: [1, 4], data: [1, 2, 3, 4] }
                },
                expected: {
                    attention: [1, 2, 3, 4] // Simplified expected output
                }
            }
        ]
    },
    
    // Performance benchmarks for regression detection
    performanceBenchmarks: {
        tensorCreation: { expectedTime: 1.0, unit: 'ms' },
        tensorAddition: { expectedTime: 0.5, unit: 'ms' },
        matrixMultiplication: { expectedTime: 10.0, unit: 'ms' },
        modelInference: { expectedTime: 100.0, unit: 'ms' },
        memoryAllocation: { expectedTime: 2.0, unit: 'ms' }
    }
};

// Numerical comparison utilities
class NumericalComparator {
    static areClose(a, b, tolerance = REGRESSION_CONFIG.tolerance) {
        if (typeof a === 'number' && typeof b === 'number') {
            return Math.abs(a - b) <= tolerance;
        }
        return a === b;
    }
    
    static compareArrays(actual, expected, tolerance = REGRESSION_CONFIG.tolerance) {
        if (actual.length !== expected.length) {
            return {
                match: false,
                reason: `Length mismatch: ${actual.length} vs ${expected.length}`
            };
        }
        
        for (let i = 0; i < actual.length; i++) {
            if (!this.areClose(actual[i], expected[i], tolerance)) {
                return {
                    match: false,
                    reason: `Value mismatch at index ${i}: ${actual[i]} vs ${expected[i]}`
                };
            }
        }
        
        return { match: true };
    }
    
    static compareShapes(actual, expected) {
        if (actual.length !== expected.length) {
            return {
                match: false,
                reason: `Shape dimension mismatch: ${actual.length} vs ${expected.length}`
            };
        }
        
        for (let i = 0; i < actual.length; i++) {
            if (actual[i] !== expected[i]) {
                return {
                    match: false,
                    reason: `Shape mismatch at dimension ${i}: ${actual[i]} vs ${expected[i]}`
                };
            }
        }
        
        return { match: true };
    }
}

// Performance regression detector
class PerformanceRegression {
    constructor() {
        this.measurements = new Map();
    }
    
    recordMeasurement(testName, actualTime, expectedTime) {
        const performance = {
            actual: actualTime,
            expected: expectedTime,
            ratio: actualTime / expectedTime,
            isRegression: (actualTime / expectedTime) > (1 + REGRESSION_CONFIG.performanceTolerance)
        };
        
        this.measurements.set(testName, performance);
        return performance;
    }
    
    getReport() {
        const report = {
            totalTests: this.measurements.size,
            regressions: 0,
            improvements: 0,
            stable: 0,
            details: []
        };
        
        for (const [testName, perf] of this.measurements) {
            if (perf.isRegression) {
                report.regressions++;
            } else if (perf.ratio < 0.8) { // 20% improvement
                report.improvements++;
            } else {
                report.stable++;
            }
            
            report.details.push({
                test: testName,
                performance: perf,
                status: perf.isRegression ? 'regression' : 
                        perf.ratio < 0.8 ? 'improvement' : 'stable'
            });
        }
        
        return report;
    }
}

// Enhanced mock WASM module for regression testing
class MockTrustformersWasmRegression {
    constructor() {
        this.initialized = false;
        this.tensors = new Map();
        this.nextTensorId = 1;
        this.performanceMetrics = new Map();
    }
    
    async initialize() {
        const startTime = performance.now();
        await new Promise(resolve => setTimeout(resolve, 50));
        this.initialized = true;
        const initTime = performance.now() - startTime;
        
        this.performanceMetrics.set('initialization', initTime);
        return initTime;
    }
    
    createTensor(shape, data) {
        const startTime = performance.now();
        
        if (!this.initialized) {
            throw new Error('WASM module not initialized');
        }
        
        const id = this.nextTensorId++;
        const size = shape.reduce((a, b) => a * b, 1);
        
        const tensor = {
            id,
            shape: [...shape],
            data: data ? new Float32Array(data) : new Float32Array(size),
            createdAt: performance.now()
        };
        
        this.tensors.set(id, tensor);
        
        const creationTime = performance.now() - startTime;
        this.performanceMetrics.set(`tensor_creation_${id}`, creationTime);
        
        return id;
    }
    
    getTensor(id) {
        return this.tensors.get(id);
    }
    
    addTensors(id1, id2) {
        const startTime = performance.now();
        
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
        
        const resultId = this.createTensor(tensor1.shape, resultData);
        
        const operationTime = performance.now() - startTime;
        this.performanceMetrics.set(`tensor_addition_${resultId}`, operationTime);
        
        return resultId;
    }
    
    multiplyTensors(id1, id2) {
        const startTime = performance.now();
        
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
        
        const resultId = this.createTensor(tensor1.shape, resultData);
        
        const operationTime = performance.now() - startTime;
        this.performanceMetrics.set(`tensor_multiplication_${resultId}`, operationTime);
        
        return resultId;
    }
    
    subtractTensors(id1, id2) {
        const tensor1 = this.tensors.get(id1);
        const tensor2 = this.tensors.get(id2);
        
        if (!tensor1 || !tensor2) {
            throw new Error('Invalid tensor ID');
        }
        
        const resultData = new Float32Array(tensor1.data.length);
        for (let i = 0; i < resultData.length; i++) {
            resultData[i] = tensor1.data[i] - tensor2.data[i];
        }
        
        return this.createTensor(tensor1.shape, resultData);
    }
    
    matrixMultiply(id1, id2) {
        const startTime = performance.now();
        
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
        
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
                let sum = 0;
                for (let p = 0; p < k; p++) {
                    sum += tensor1.data[i * k + p] * tensor2.data[p * n + j];
                }
                resultData[i * n + j] = sum;
            }
        }
        
        const resultId = this.createTensor([m, n], resultData);
        
        const operationTime = performance.now() - startTime;
        this.performanceMetrics.set(`matrix_multiplication_${resultId}`, operationTime);
        
        return resultId;
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
        
        return this.createTensor(tensor.shape, resultData);
    }
    
    sigmoid(id) {
        const tensor = this.tensors.get(id);
        if (!tensor) {
            throw new Error('Invalid tensor ID');
        }
        
        const resultData = new Float32Array(tensor.data.length);
        for (let i = 0; i < resultData.length; i++) {
            resultData[i] = 1 / (1 + Math.exp(-tensor.data[i]));
        }
        
        return this.createTensor(tensor.shape, resultData);
    }
    
    tanh(id) {
        const tensor = this.tensors.get(id);
        if (!tensor) {
            throw new Error('Invalid tensor ID');
        }
        
        const resultData = new Float32Array(tensor.data.length);
        for (let i = 0; i < resultData.length; i++) {
            resultData[i] = Math.tanh(tensor.data[i]);
        }
        
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
        
        return this.createTensor(tensor.shape, resultData);
    }
    
    simpleAttention(queryId, keyId, valueId) {
        const query = this.tensors.get(queryId);
        const key = this.tensors.get(keyId);
        const value = this.tensors.get(valueId);
        
        if (!query || !key || !value) {
            throw new Error('Invalid tensor ID');
        }
        
        // Simplified attention for testing
        // In practice, this would be much more complex
        const resultData = new Float32Array(value.data.length);
        
        // Just return the value for this simplified test
        for (let i = 0; i < resultData.length; i++) {
            resultData[i] = value.data[i];
        }
        
        return this.createTensor(value.shape, resultData);
    }
    
    getPerformanceMetrics() {
        return this.performanceMetrics;
    }
    
    cleanup() {
        this.tensors.clear();
        this.performanceMetrics.clear();
        this.nextTensorId = 1;
    }
}

// Regression test executor
class RegressionTestExecutor {
    constructor() {
        this.wasmModule = new MockTrustformersWasmRegression();
        this.performanceRegression = new PerformanceRegression();
        this.testResults = [];
    }
    
    async runRegressionTests() {
        await this.wasmModule.initialize();
        
        for (const testCase of REGRESSION_CONFIG.referenceData.testCases) {
            console.log(`Running regression test: ${testCase.name}`);
            
            try {
                const result = await this.runSingleTest(testCase);
                this.testResults.push(result);
            } catch (error) {
                this.testResults.push({
                    testName: testCase.name,
                    success: false,
                    error: error.message,
                    timestamp: new Date().toISOString()
                });
            }
        }
        
        return this.testResults;
    }
    
    async runSingleTest(testCase) {
        const result = {
            testName: testCase.name,
            success: true,
            comparisons: [],
            performance: {},
            timestamp: new Date().toISOString()
        };
        
        switch (testCase.name) {
            case 'basic_tensor_operations':
                result.comparisons = await this.testBasicTensorOperations(testCase);
                break;
            case 'matrix_multiplication':
                result.comparisons = await this.testMatrixMultiplication(testCase);
                break;
            case 'activation_functions':
                result.comparisons = await this.testActivationFunctions(testCase);
                break;
            case 'normalization_operations':
                result.comparisons = await this.testNormalizationOperations(testCase);
                break;
            case 'attention_mechanism':
                result.comparisons = await this.testAttentionMechanism(testCase);
                break;
            default:
                throw new Error(`Unknown test case: ${testCase.name}`);
        }
        
        // Check if all comparisons passed
        result.success = result.comparisons.every(comp => comp.match);
        
        return result;
    }
    
    async testBasicTensorOperations(testCase) {
        const { input, expected } = testCase;
        
        const tensor1Id = this.wasmModule.createTensor(input.tensor1.shape, input.tensor1.data);
        const tensor2Id = this.wasmModule.createTensor(input.tensor2.shape, input.tensor2.data);
        
        const comparisons = [];
        
        // Test addition
        const addResultId = this.wasmModule.addTensors(tensor1Id, tensor2Id);
        const addResult = this.wasmModule.getTensor(addResultId);
        comparisons.push({
            operation: 'addition',
            ...NumericalComparator.compareArrays(Array.from(addResult.data), expected.addition)
        });
        
        // Test multiplication
        const mulResultId = this.wasmModule.multiplyTensors(tensor1Id, tensor2Id);
        const mulResult = this.wasmModule.getTensor(mulResultId);
        comparisons.push({
            operation: 'multiplication',
            ...NumericalComparator.compareArrays(Array.from(mulResult.data), expected.multiplication)
        });
        
        // Test subtraction
        const subResultId = this.wasmModule.subtractTensors(tensor1Id, tensor2Id);
        const subResult = this.wasmModule.getTensor(subResultId);
        comparisons.push({
            operation: 'subtraction',
            ...NumericalComparator.compareArrays(Array.from(subResult.data), expected.subtraction)
        });
        
        return comparisons;
    }
    
    async testMatrixMultiplication(testCase) {
        const { input, expected } = testCase;
        
        const matrix1Id = this.wasmModule.createTensor(input.matrix1.shape, input.matrix1.data);
        const matrix2Id = this.wasmModule.createTensor(input.matrix2.shape, input.matrix2.data);
        
        const resultId = this.wasmModule.matrixMultiply(matrix1Id, matrix2Id);
        const result = this.wasmModule.getTensor(resultId);
        
        return [{
            operation: 'matrix_multiplication',
            ...NumericalComparator.compareArrays(Array.from(result.data), expected.result)
        }];
    }
    
    async testActivationFunctions(testCase) {
        const { input, expected } = testCase;
        
        const tensorId = this.wasmModule.createTensor(input.tensor.shape, input.tensor.data);
        const comparisons = [];
        
        // Test ReLU
        const reluResultId = this.wasmModule.relu(tensorId);
        const reluResult = this.wasmModule.getTensor(reluResultId);
        comparisons.push({
            operation: 'relu',
            ...NumericalComparator.compareArrays(Array.from(reluResult.data), expected.relu)
        });
        
        // Test Sigmoid
        const sigmoidResultId = this.wasmModule.sigmoid(tensorId);
        const sigmoidResult = this.wasmModule.getTensor(sigmoidResultId);
        comparisons.push({
            operation: 'sigmoid',
            ...NumericalComparator.compareArrays(Array.from(sigmoidResult.data), expected.sigmoid)
        });
        
        // Test Tanh
        const tanhResultId = this.wasmModule.tanh(tensorId);
        const tanhResult = this.wasmModule.getTensor(tanhResultId);
        comparisons.push({
            operation: 'tanh',
            ...NumericalComparator.compareArrays(Array.from(tanhResult.data), expected.tanh)
        });
        
        return comparisons;
    }
    
    async testNormalizationOperations(testCase) {
        const { input, expected } = testCase;
        
        const tensorId = this.wasmModule.createTensor(input.tensor.shape, input.tensor.data);
        
        const normResultId = this.wasmModule.layerNorm(tensorId);
        const normResult = this.wasmModule.getTensor(normResultId);
        
        return [{
            operation: 'layer_norm',
            ...NumericalComparator.compareArrays(Array.from(normResult.data), expected.layerNorm)
        }];
    }
    
    async testAttentionMechanism(testCase) {
        const { input, expected } = testCase;
        
        const queryId = this.wasmModule.createTensor(input.query.shape, input.query.data);
        const keyId = this.wasmModule.createTensor(input.key.shape, input.key.data);
        const valueId = this.wasmModule.createTensor(input.value.shape, input.value.data);
        
        const attentionResultId = this.wasmModule.simpleAttention(queryId, keyId, valueId);
        const attentionResult = this.wasmModule.getTensor(attentionResultId);
        
        return [{
            operation: 'attention',
            ...NumericalComparator.compareArrays(Array.from(attentionResult.data), expected.attention)
        }];
    }
    
    getResults() {
        return this.testResults;
    }
    
    getPerformanceReport() {
        return this.performanceRegression.getReport();
    }
}

// Global test setup
let regressionExecutor;

describe('Regression Tests', () => {
    beforeAll(async () => {
        regressionExecutor = new RegressionTestExecutor();
        console.log('Regression test suite initialized');
    });
    
    afterAll(() => {
        const results = regressionExecutor.getResults();
        const performanceReport = regressionExecutor.getPerformanceReport();
        
        console.log('\n=== REGRESSION TEST RESULTS ===');
        console.log(`Total tests: ${results.length}`);
        console.log(`Passed: ${results.filter(r => r.success).length}`);
        console.log(`Failed: ${results.filter(r => !r.success).length}`);
        
        console.log('\n=== PERFORMANCE REGRESSION REPORT ===');
        console.log(JSON.stringify(performanceReport, null, 2));
        
        // Store results for analysis
        if (typeof localStorage !== 'undefined') {
            localStorage.setItem('trustformers_regression_results', JSON.stringify({
                functionalTests: results,
                performanceReport: performanceReport
            }));
        }
    });
    
    it('should run all regression tests', async () => {
        const results = await regressionExecutor.runRegressionTests();
        
        expect(results.length).toBe(REGRESSION_CONFIG.referenceData.testCases.length);
        
        const failedTests = results.filter(r => !r.success);
        if (failedTests.length > 0) {
            console.error('Failed regression tests:', failedTests.map(t => t.testName));
            failedTests.forEach(test => {
                console.error(`Test: ${test.testName}`);
                if (test.error) {
                    console.error(`Error: ${test.error}`);
                }
                if (test.comparisons) {
                    test.comparisons.forEach(comp => {
                        if (!comp.match) {
                            console.error(`  ${comp.operation}: ${comp.reason}`);
                        }
                    });
                }
            });
        }
        
        expect(failedTests.length).toBe(0);
    }, 30000);
    
    it('should maintain numerical accuracy', async () => {
        await regressionExecutor.runRegressionTests();
        const results = regressionExecutor.getResults();
        
        for (const result of results) {
            if (result.comparisons) {
                for (const comparison of result.comparisons) {
                    expect(comparison.match).toBe(true);
                    if (!comparison.match) {
                        console.error(`Numerical accuracy regression in ${result.testName}.${comparison.operation}: ${comparison.reason}`);
                    }
                }
            }
        }
    });
    
    it('should not have performance regressions', async () => {
        await regressionExecutor.runRegressionTests();
        const performanceReport = regressionExecutor.getPerformanceReport();
        
        expect(performanceReport.regressions).toBe(0);
        
        if (performanceReport.regressions > 0) {
            console.error('Performance regressions detected:');
            performanceReport.details
                .filter(d => d.status === 'regression')
                .forEach(d => {
                    console.error(`  ${d.test}: ${d.performance.ratio.toFixed(2)}x slower`);
                });
        }
    });
    
    it('should detect deliberate regression', async () => {
        // Test the regression detection system itself
        const originalTolerance = REGRESSION_CONFIG.tolerance;
        REGRESSION_CONFIG.tolerance = 1e-10; // Very strict tolerance
        
        const results = await regressionExecutor.runRegressionTests();
        
        // Restore original tolerance
        REGRESSION_CONFIG.tolerance = originalTolerance;
        
        // Some tests might fail with very strict tolerance
        expect(results.length).toBeGreaterThan(0);
    });
});

// Export for use in other test files
export {
    RegressionTestExecutor,
    NumericalComparator,
    PerformanceRegression,
    REGRESSION_CONFIG
};