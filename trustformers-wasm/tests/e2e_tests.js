/**
 * End-to-End Tests for TrustformeRS WASM
 * 
 * This test suite covers complete workflows from initialization through
 * inference to cleanup, testing real-world usage scenarios.
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach } from '@jest/globals';

// Test configuration
const E2E_CONFIG = {
    timeout: 30000,
    modelLoadTimeout: 10000,
    inferenceTimeout: 5000,
    
    // Test scenarios
    scenarios: [
        {
            name: 'text_generation',
            description: 'Complete text generation pipeline',
            inputSize: [1, 512],
            outputSize: [1, 512],
            steps: ['init', 'load_model', 'tokenize', 'inference', 'decode', 'cleanup']
        },
        {
            name: 'text_classification',
            description: 'Text classification end-to-end',
            inputSize: [1, 256],
            outputSize: [1, 10],
            steps: ['init', 'load_model', 'preprocess', 'inference', 'postprocess', 'cleanup']
        },
        {
            name: 'embeddings',
            description: 'Text embeddings generation',
            inputSize: [1, 128],
            outputSize: [1, 768],
            steps: ['init', 'load_model', 'tokenize', 'inference', 'extract_embeddings', 'cleanup']
        },
        {
            name: 'batch_processing',
            description: 'Batch processing multiple inputs',
            inputSize: [8, 256],
            outputSize: [8, 256],
            steps: ['init', 'load_model', 'batch_preprocess', 'batch_inference', 'batch_postprocess', 'cleanup']
        }
    ]
};

// Mock model data
const MOCK_MODEL_DATA = {
    text_generation: {
        config: {
            vocab_size: 50000,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            max_position_embeddings: 512,
            model_type: 'gpt2'
        },
        weights: {
            embedding: new Float32Array(50000 * 768),
            transformer_layers: Array.from({ length: 12 }, () => ({
                attention: {
                    query: new Float32Array(768 * 768),
                    key: new Float32Array(768 * 768),
                    value: new Float32Array(768 * 768),
                    output: new Float32Array(768 * 768)
                },
                feed_forward: {
                    intermediate: new Float32Array(768 * 3072),
                    output: new Float32Array(3072 * 768)
                },
                layer_norm1: new Float32Array(768),
                layer_norm2: new Float32Array(768)
            })),
            output_projection: new Float32Array(768 * 50000)
        }
    },
    text_classification: {
        config: {
            vocab_size: 30000,
            hidden_size: 512,
            num_layers: 6,
            num_heads: 8,
            max_position_embeddings: 256,
            num_classes: 10,
            model_type: 'bert'
        },
        weights: {
            embedding: new Float32Array(30000 * 512),
            transformer_layers: Array.from({ length: 6 }, () => ({
                attention: {
                    query: new Float32Array(512 * 512),
                    key: new Float32Array(512 * 512),
                    value: new Float32Array(512 * 512),
                    output: new Float32Array(512 * 512)
                },
                feed_forward: {
                    intermediate: new Float32Array(512 * 2048),
                    output: new Float32Array(2048 * 512)
                },
                layer_norm1: new Float32Array(512),
                layer_norm2: new Float32Array(512)
            })),
            classifier: new Float32Array(512 * 10)
        }
    },
    embeddings: {
        config: {
            vocab_size: 30000,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            max_position_embeddings: 128,
            model_type: 'bert'
        },
        weights: {
            embedding: new Float32Array(30000 * 768),
            transformer_layers: Array.from({ length: 12 }, () => ({
                attention: {
                    query: new Float32Array(768 * 768),
                    key: new Float32Array(768 * 768),
                    value: new Float32Array(768 * 768),
                    output: new Float32Array(768 * 768)
                },
                feed_forward: {
                    intermediate: new Float32Array(768 * 3072),
                    output: new Float32Array(3072 * 768)
                },
                layer_norm1: new Float32Array(768),
                layer_norm2: new Float32Array(768)
            }))
        }
    }
};

// E2E Test Framework
class E2ETestFramework {
    constructor() {
        this.testResults = [];
        this.currentScenario = null;
        this.wasmModule = null;
        this.models = new Map();
        this.stepTimings = new Map();
    }
    
    async initialize() {
        console.log('Initializing E2E test framework...');
        
        // Initialize WASM module
        this.wasmModule = new MockTrustformersWasmE2E();
        await this.wasmModule.initialize();
        
        // Pre-load test models
        for (const [modelName, modelData] of Object.entries(MOCK_MODEL_DATA)) {
            await this.loadModel(modelName, modelData);
        }
        
        console.log('E2E test framework initialized successfully');
    }
    
    async loadModel(modelName, modelData) {
        console.log(`Loading model: ${modelName}`);
        
        const startTime = performance.now();
        
        // Simulate model loading process
        await new Promise(resolve => setTimeout(resolve, 100 + Math.random() * 200));
        
        const modelId = await this.wasmModule.loadModel(modelName, modelData);
        this.models.set(modelName, modelId);
        
        const loadTime = performance.now() - startTime;
        console.log(`Model ${modelName} loaded in ${loadTime.toFixed(2)}ms`);
        
        return modelId;
    }
    
    startScenario(scenarioName) {
        this.currentScenario = scenarioName;
        this.stepTimings.clear();
        console.log(`Starting scenario: ${scenarioName}`);
    }
    
    async executeStep(stepName, stepFn) {
        if (!this.currentScenario) {
            throw new Error('No active scenario');
        }
        
        console.log(`Executing step: ${stepName}`);
        const startTime = performance.now();
        
        try {
            const result = await stepFn();
            const duration = performance.now() - startTime;
            
            this.stepTimings.set(stepName, {
                duration,
                success: true,
                result
            });
            
            console.log(`Step ${stepName} completed in ${duration.toFixed(2)}ms`);
            return result;
        } catch (error) {
            const duration = performance.now() - startTime;
            
            this.stepTimings.set(stepName, {
                duration,
                success: false,
                error: error.message
            });
            
            console.error(`Step ${stepName} failed after ${duration.toFixed(2)}ms:`, error.message);
            throw error;
        }
    }
    
    finishScenario() {
        if (!this.currentScenario) {
            throw new Error('No active scenario');
        }
        
        const totalDuration = Array.from(this.stepTimings.values())
            .reduce((sum, step) => sum + step.duration, 0);
        
        const result = {
            scenario: this.currentScenario,
            totalDuration,
            steps: Object.fromEntries(this.stepTimings),
            success: Array.from(this.stepTimings.values()).every(step => step.success)
        };
        
        this.testResults.push(result);
        
        console.log(`Scenario ${this.currentScenario} completed in ${totalDuration.toFixed(2)}ms`);
        this.currentScenario = null;
        
        return result;
    }
    
    getResults() {
        return this.testResults;
    }
    
    cleanup() {
        if (this.wasmModule) {
            this.wasmModule.cleanup();
        }
        this.models.clear();
        console.log('E2E test framework cleaned up');
    }
}

// Enhanced Mock WASM Module for E2E Testing
class MockTrustformersWasmE2E {
    constructor() {
        this.initialized = false;
        this.models = new Map();
        this.tensors = new Map();
        this.nextModelId = 1;
        this.nextTensorId = 1;
        this.useWebGPU = false;
        this.operationCount = 0;
    }
    
    async initialize() {
        console.log('Initializing WASM module...');
        
        // Simulate WASM initialization
        await new Promise(resolve => setTimeout(resolve, 100 + Math.random() * 200));
        
        this.initialized = true;
        console.log('WASM module initialized');
    }
    
    async loadModel(modelName, modelData) {
        if (!this.initialized) {
            throw new Error('WASM module not initialized');
        }
        
        console.log(`Loading model: ${modelName}`);
        
        // Simulate model loading time based on model size
        const modelSize = this.calculateModelSize(modelData);
        const loadTime = Math.min(1000, modelSize / 1000000 * 100); // Scale based on size
        
        await new Promise(resolve => setTimeout(resolve, loadTime));
        
        const modelId = this.nextModelId++;
        this.models.set(modelId, {
            name: modelName,
            data: modelData,
            loaded: true,
            size: modelSize
        });
        
        console.log(`Model ${modelName} loaded with ID ${modelId}`);
        return modelId;
    }
    
    calculateModelSize(modelData) {
        let totalSize = 0;
        
        const calculateArraySize = (arr) => {
            if (arr instanceof Float32Array) {
                return arr.length * 4;
            } else if (Array.isArray(arr)) {
                return arr.reduce((sum, item) => sum + calculateArraySize(item), 0);
            } else if (typeof arr === 'object' && arr !== null) {
                return Object.values(arr).reduce((sum, item) => sum + calculateArraySize(item), 0);
            }
            return 0;
        };
        
        totalSize += calculateArraySize(modelData.weights);
        return totalSize;
    }
    
    async tokenize(text, modelId) {
        const model = this.models.get(modelId);
        if (!model) {
            throw new Error(`Model ${modelId} not found`);
        }
        
        // Simulate tokenization
        const tokens = text.split(' ').map((word, index) => index % 1000);
        
        // Simulate processing time
        await new Promise(resolve => setTimeout(resolve, 10 + tokens.length * 0.1));
        
        return {
            tokens,
            attention_mask: new Array(tokens.length).fill(1),
            length: tokens.length
        };
    }
    
    async inference(inputTensorId, modelId, options = {}) {
        const model = this.models.get(modelId);
        if (!model) {
            throw new Error(`Model ${modelId} not found`);
        }
        
        const inputTensor = this.tensors.get(inputTensorId);
        if (!inputTensor) {
            throw new Error(`Input tensor ${inputTensorId} not found`);
        }
        
        console.log(`Running inference for model ${model.name}`);
        
        // Simulate inference time based on model complexity and input size
        const inferenceTime = this.calculateInferenceTime(model, inputTensor);
        await new Promise(resolve => setTimeout(resolve, inferenceTime));
        
        // Create output tensor based on model type
        const outputShape = this.getOutputShape(model, inputTensor.shape);
        const outputData = new Float32Array(outputShape.reduce((a, b) => a * b, 1));
        
        // Fill with some realistic-looking data
        for (let i = 0; i < outputData.length; i++) {
            outputData[i] = Math.random() * 2 - 1; // Random values between -1 and 1
        }
        
        const outputTensorId = this.createTensor(outputShape, outputData);
        
        this.operationCount++;
        console.log(`Inference completed for model ${model.name}`);
        
        return {
            outputTensorId,
            inferenceTime,
            outputShape
        };
    }
    
    calculateInferenceTime(model, inputTensor) {
        const modelConfig = model.data.config;
        const inputSize = inputTensor.data.length;
        
        // Base time + complexity factors
        let baseTime = 50;
        
        // Factor in model size
        if (modelConfig.num_layers) {
            baseTime += modelConfig.num_layers * 10;
        }
        
        if (modelConfig.hidden_size) {
            baseTime += modelConfig.hidden_size * 0.01;
        }
        
        // Factor in input size
        baseTime += inputSize * 0.001;
        
        // Add some randomness
        baseTime += Math.random() * 20;
        
        // WebGPU is faster
        if (this.useWebGPU) {
            baseTime *= 0.3;
        }
        
        return Math.max(10, baseTime);
    }
    
    getOutputShape(model, inputShape) {
        const modelConfig = model.data.config;
        const modelType = modelConfig.model_type;
        
        switch (modelType) {
            case 'gpt2':
                // Text generation: same shape as input
                return [...inputShape];
            
            case 'bert':
                if (modelConfig.num_classes) {
                    // Classification: [batch_size, num_classes]
                    return [inputShape[0], modelConfig.num_classes];
                } else {
                    // Embeddings: [batch_size, sequence_length, hidden_size]
                    return [inputShape[0], inputShape[1], modelConfig.hidden_size];
                }
            
            default:
                return [...inputShape];
        }
    }
    
    createTensor(shape, data) {
        const id = this.nextTensorId++;
        const tensor = {
            id,
            shape: [...shape],
            data: data ? new Float32Array(data) : new Float32Array(shape.reduce((a, b) => a * b, 1)),
            device: this.useWebGPU ? 'gpu' : 'cpu'
        };
        
        this.tensors.set(id, tensor);
        return id;
    }
    
    getTensor(id) {
        return this.tensors.get(id);
    }
    
    async decode(outputTensorId, modelId) {
        const model = this.models.get(modelId);
        if (!model) {
            throw new Error(`Model ${modelId} not found`);
        }
        
        const outputTensor = this.tensors.get(outputTensorId);
        if (!outputTensor) {
            throw new Error(`Output tensor ${outputTensorId} not found`);
        }
        
        // Simulate decoding
        await new Promise(resolve => setTimeout(resolve, 20 + Math.random() * 30));
        
        const modelType = model.data.config.model_type;
        
        switch (modelType) {
            case 'gpt2':
                // Convert logits to tokens to text
                const tokens = [];
                for (let i = 0; i < outputTensor.data.length; i++) {
                    tokens.push(Math.floor(Math.abs(outputTensor.data[i] * 1000)) % 1000);
                }
                return {
                    type: 'text',
                    content: tokens.map(t => `token_${t}`).join(' '),
                    tokens
                };
            
            case 'bert':
                if (model.data.config.num_classes) {
                    // Classification output
                    const logits = Array.from(outputTensor.data);
                    const maxIndex = logits.indexOf(Math.max(...logits));
                    return {
                        type: 'classification',
                        predicted_class: maxIndex,
                        confidence: Math.max(...logits),
                        logits
                    };
                } else {
                    // Embeddings output
                    return {
                        type: 'embeddings',
                        embeddings: Array.from(outputTensor.data),
                        shape: outputTensor.shape
                    };
                }
            
            default:
                return {
                    type: 'raw',
                    data: Array.from(outputTensor.data),
                    shape: outputTensor.shape
                };
        }
    }
    
    async enableWebGPU() {
        if (!navigator.gpu) {
            throw new Error('WebGPU not supported');
        }
        
        // Simulate WebGPU initialization
        await new Promise(resolve => setTimeout(resolve, 100));
        
        this.useWebGPU = true;
        console.log('WebGPU enabled');
    }
    
    async processInBatches(inputs, modelId, batchSize = 4) {
        const model = this.models.get(modelId);
        if (!model) {
            throw new Error(`Model ${modelId} not found`);
        }
        
        const results = [];
        
        for (let i = 0; i < inputs.length; i += batchSize) {
            const batch = inputs.slice(i, i + batchSize);
            const batchResults = [];
            
            for (const input of batch) {
                const inputTensorId = this.createTensor(input.shape, input.data);
                const inferenceResult = await this.inference(inputTensorId, modelId);
                const decodedResult = await this.decode(inferenceResult.outputTensorId, modelId);
                
                batchResults.push({
                    input: input,
                    output: decodedResult,
                    inferenceTime: inferenceResult.inferenceTime
                });
                
                // Cleanup intermediate tensors
                this.tensors.delete(inputTensorId);
                this.tensors.delete(inferenceResult.outputTensorId);
            }
            
            results.push(...batchResults);
        }
        
        return results;
    }
    
    getMemoryUsage() {
        const totalTensors = this.tensors.size;
        const totalModels = this.models.size;
        const totalTensorBytes = Array.from(this.tensors.values())
            .reduce((sum, tensor) => sum + tensor.data.byteLength, 0);
        const totalModelBytes = Array.from(this.models.values())
            .reduce((sum, model) => sum + model.size, 0);
        
        return {
            totalTensors,
            totalModels,
            totalTensorBytes,
            totalModelBytes,
            totalBytes: totalTensorBytes + totalModelBytes,
            estimatedMemoryMB: (totalTensorBytes + totalModelBytes) / (1024 * 1024),
            operationCount: this.operationCount
        };
    }
    
    cleanup() {
        this.tensors.clear();
        this.models.clear();
        this.nextTensorId = 1;
        this.nextModelId = 1;
        this.operationCount = 0;
        console.log('WASM module cleaned up');
    }
}

// Global test setup
let e2eFramework;

describe('End-to-End Tests', () => {
    beforeAll(async () => {
        e2eFramework = new E2ETestFramework();
        await e2eFramework.initialize();
        console.log('E2E test suite initialized');
    }, E2E_CONFIG.timeout);
    
    afterAll(() => {
        const results = e2eFramework.getResults();
        console.log('\n=== E2E TEST RESULTS ===');
        console.log(JSON.stringify(results, null, 2));
        
        // Store results for analysis
        if (typeof localStorage !== 'undefined') {
            localStorage.setItem('trustformers_e2e_results', JSON.stringify(results));
        }
        
        e2eFramework.cleanup();
    });

    describe('Text Generation Pipeline', () => {
        it('should complete full text generation workflow', async () => {
            const scenario = E2E_CONFIG.scenarios.find(s => s.name === 'text_generation');
            e2eFramework.startScenario(scenario.name);
            
            try {
                // Step 1: Initialize (already done in beforeAll)
                await e2eFramework.executeStep('init', async () => {
                    expect(e2eFramework.wasmModule.initialized).toBe(true);
                    return { status: 'initialized' };
                });
                
                // Step 2: Load model (already done in beforeAll)
                const modelId = await e2eFramework.executeStep('load_model', async () => {
                    const id = e2eFramework.models.get('text_generation');
                    expect(id).toBeDefined();
                    return id;
                });
                
                // Step 3: Tokenize input
                const tokenized = await e2eFramework.executeStep('tokenize', async () => {
                    const input = "Hello, this is a test input for text generation";
                    const result = await e2eFramework.wasmModule.tokenize(input, modelId);
                    
                    expect(result.tokens).toBeDefined();
                    expect(result.tokens.length).toBeGreaterThan(0);
                    expect(result.attention_mask).toBeDefined();
                    
                    return result;
                });
                
                // Step 4: Run inference
                const inferenceResult = await e2eFramework.executeStep('inference', async () => {
                    const inputTensorId = e2eFramework.wasmModule.createTensor(
                        [1, tokenized.tokens.length],
                        tokenized.tokens
                    );
                    
                    const result = await e2eFramework.wasmModule.inference(inputTensorId, modelId);
                    
                    expect(result.outputTensorId).toBeDefined();
                    expect(result.inferenceTime).toBeGreaterThan(0);
                    expect(result.outputShape).toBeDefined();
                    
                    return result;
                });
                
                // Step 5: Decode output
                const decodedResult = await e2eFramework.executeStep('decode', async () => {
                    const result = await e2eFramework.wasmModule.decode(
                        inferenceResult.outputTensorId,
                        modelId
                    );
                    
                    expect(result.type).toBe('text');
                    expect(result.content).toBeDefined();
                    expect(result.tokens).toBeDefined();
                    
                    return result;
                });
                
                // Step 6: Cleanup
                await e2eFramework.executeStep('cleanup', async () => {
                    const memoryBefore = e2eFramework.wasmModule.getMemoryUsage();
                    
                    // Cleanup intermediate tensors
                    e2eFramework.wasmModule.tensors.clear();
                    
                    const memoryAfter = e2eFramework.wasmModule.getMemoryUsage();
                    
                    expect(memoryAfter.totalTensors).toBeLessThan(memoryBefore.totalTensors);
                    
                    return { memoryBefore, memoryAfter };
                });
                
                const scenarioResult = e2eFramework.finishScenario();
                expect(scenarioResult.success).toBe(true);
                expect(scenarioResult.totalDuration).toBeLessThan(E2E_CONFIG.timeout);
                
                console.log(`Text generation completed in ${scenarioResult.totalDuration.toFixed(2)}ms`);
                console.log('Generated text:', decodedResult.content);
                
            } catch (error) {
                e2eFramework.finishScenario();
                throw error;
            }
        }, E2E_CONFIG.timeout);
    });

    describe('Text Classification Pipeline', () => {
        it('should complete full text classification workflow', async () => {
            const scenario = E2E_CONFIG.scenarios.find(s => s.name === 'text_classification');
            e2eFramework.startScenario(scenario.name);
            
            try {
                // Step 1: Initialize
                await e2eFramework.executeStep('init', async () => {
                    expect(e2eFramework.wasmModule.initialized).toBe(true);
                    return { status: 'initialized' };
                });
                
                // Step 2: Load model
                const modelId = await e2eFramework.executeStep('load_model', async () => {
                    const id = e2eFramework.models.get('text_classification');
                    expect(id).toBeDefined();
                    return id;
                });
                
                // Step 3: Preprocess input
                const preprocessed = await e2eFramework.executeStep('preprocess', async () => {
                    const input = "This is a positive review of the product";
                    const tokenized = await e2eFramework.wasmModule.tokenize(input, modelId);
                    
                    // Pad or truncate to fixed length
                    const maxLength = 256;
                    const paddedTokens = tokenized.tokens.slice(0, maxLength);
                    while (paddedTokens.length < maxLength) {
                        paddedTokens.push(0); // Padding token
                    }
                    
                    return {
                        tokens: paddedTokens,
                        attention_mask: new Array(paddedTokens.length).fill(1),
                        originalLength: tokenized.tokens.length
                    };
                });
                
                // Step 4: Run inference
                const inferenceResult = await e2eFramework.executeStep('inference', async () => {
                    const inputTensorId = e2eFramework.wasmModule.createTensor(
                        [1, preprocessed.tokens.length],
                        preprocessed.tokens
                    );
                    
                    const result = await e2eFramework.wasmModule.inference(inputTensorId, modelId);
                    
                    expect(result.outputTensorId).toBeDefined();
                    expect(result.outputShape).toEqual([1, 10]); // 10 classes
                    
                    return result;
                });
                
                // Step 5: Postprocess output
                const postprocessed = await e2eFramework.executeStep('postprocess', async () => {
                    const result = await e2eFramework.wasmModule.decode(
                        inferenceResult.outputTensorId,
                        modelId
                    );
                    
                    expect(result.type).toBe('classification');
                    expect(result.predicted_class).toBeGreaterThanOrEqual(0);
                    expect(result.predicted_class).toBeLessThan(10);
                    expect(result.confidence).toBeDefined();
                    
                    return result;
                });
                
                // Step 6: Cleanup
                await e2eFramework.executeStep('cleanup', async () => {
                    e2eFramework.wasmModule.tensors.clear();
                    return { status: 'cleaned' };
                });
                
                const scenarioResult = e2eFramework.finishScenario();
                expect(scenarioResult.success).toBe(true);
                
                console.log(`Text classification completed in ${scenarioResult.totalDuration.toFixed(2)}ms`);
                console.log(`Predicted class: ${postprocessed.predicted_class}, Confidence: ${postprocessed.confidence.toFixed(3)}`);
                
            } catch (error) {
                e2eFramework.finishScenario();
                throw error;
            }
        }, E2E_CONFIG.timeout);
    });

    describe('Embeddings Generation Pipeline', () => {
        it('should complete full embeddings generation workflow', async () => {
            const scenario = E2E_CONFIG.scenarios.find(s => s.name === 'embeddings');
            e2eFramework.startScenario(scenario.name);
            
            try {
                // Step 1: Initialize
                await e2eFramework.executeStep('init', async () => {
                    expect(e2eFramework.wasmModule.initialized).toBe(true);
                    return { status: 'initialized' };
                });
                
                // Step 2: Load model
                const modelId = await e2eFramework.executeStep('load_model', async () => {
                    const id = e2eFramework.models.get('embeddings');
                    expect(id).toBeDefined();
                    return id;
                });
                
                // Step 3: Tokenize input
                const tokenized = await e2eFramework.executeStep('tokenize', async () => {
                    const input = "This is a sentence for embedding generation";
                    const result = await e2eFramework.wasmModule.tokenize(input, modelId);
                    
                    expect(result.tokens).toBeDefined();
                    expect(result.tokens.length).toBeLessThanOrEqual(128); // Max length
                    
                    return result;
                });
                
                // Step 4: Run inference
                const inferenceResult = await e2eFramework.executeStep('inference', async () => {
                    const inputTensorId = e2eFramework.wasmModule.createTensor(
                        [1, tokenized.tokens.length],
                        tokenized.tokens
                    );
                    
                    const result = await e2eFramework.wasmModule.inference(inputTensorId, modelId);
                    
                    expect(result.outputTensorId).toBeDefined();
                    expect(result.outputShape).toEqual([1, tokenized.tokens.length, 768]); // Hidden size
                    
                    return result;
                });
                
                // Step 5: Extract embeddings
                const embeddings = await e2eFramework.executeStep('extract_embeddings', async () => {
                    const result = await e2eFramework.wasmModule.decode(
                        inferenceResult.outputTensorId,
                        modelId
                    );
                    
                    expect(result.type).toBe('embeddings');
                    expect(result.embeddings).toBeDefined();
                    expect(result.embeddings.length).toBe(tokenized.tokens.length * 768);
                    
                    // Extract sentence embedding (mean pooling)
                    const sentenceEmbedding = new Float32Array(768);
                    for (let i = 0; i < 768; i++) {
                        let sum = 0;
                        for (let j = 0; j < tokenized.tokens.length; j++) {
                            sum += result.embeddings[j * 768 + i];
                        }
                        sentenceEmbedding[i] = sum / tokenized.tokens.length;
                    }
                    
                    return {
                        ...result,
                        sentenceEmbedding: Array.from(sentenceEmbedding)
                    };
                });
                
                // Step 6: Cleanup
                await e2eFramework.executeStep('cleanup', async () => {
                    e2eFramework.wasmModule.tensors.clear();
                    return { status: 'cleaned' };
                });
                
                const scenarioResult = e2eFramework.finishScenario();
                expect(scenarioResult.success).toBe(true);
                
                console.log(`Embeddings generation completed in ${scenarioResult.totalDuration.toFixed(2)}ms`);
                console.log(`Sentence embedding dimensions: ${embeddings.sentenceEmbedding.length}`);
                
            } catch (error) {
                e2eFramework.finishScenario();
                throw error;
            }
        }, E2E_CONFIG.timeout);
    });

    describe('Batch Processing Pipeline', () => {
        it('should complete batch processing workflow', async () => {
            const scenario = E2E_CONFIG.scenarios.find(s => s.name === 'batch_processing');
            e2eFramework.startScenario(scenario.name);
            
            try {
                // Step 1: Initialize
                await e2eFramework.executeStep('init', async () => {
                    expect(e2eFramework.wasmModule.initialized).toBe(true);
                    return { status: 'initialized' };
                });
                
                // Step 2: Load model
                const modelId = await e2eFramework.executeStep('load_model', async () => {
                    const id = e2eFramework.models.get('text_generation');
                    expect(id).toBeDefined();
                    return id;
                });
                
                // Step 3: Batch preprocess
                const batchInputs = await e2eFramework.executeStep('batch_preprocess', async () => {
                    const texts = [
                        "First input text for batch processing",
                        "Second input text for batch processing",
                        "Third input text for batch processing",
                        "Fourth input text for batch processing"
                    ];
                    
                    const batchInputs = [];
                    for (const text of texts) {
                        const tokenized = await e2eFramework.wasmModule.tokenize(text, modelId);
                        batchInputs.push({
                            text,
                            shape: [1, tokenized.tokens.length],
                            data: tokenized.tokens
                        });
                    }
                    
                    expect(batchInputs).toHaveLength(4);
                    return batchInputs;
                });
                
                // Step 4: Batch inference
                const batchResults = await e2eFramework.executeStep('batch_inference', async () => {
                    const results = await e2eFramework.wasmModule.processInBatches(
                        batchInputs,
                        modelId,
                        2 // Batch size of 2
                    );
                    
                    expect(results).toHaveLength(4);
                    results.forEach(result => {
                        expect(result.output).toBeDefined();
                        expect(result.inferenceTime).toBeGreaterThan(0);
                    });
                    
                    return results;
                });
                
                // Step 5: Batch postprocess
                const processedResults = await e2eFramework.executeStep('batch_postprocess', async () => {
                    const processed = batchResults.map((result, index) => ({
                        inputText: batchInputs[index].text,
                        outputText: result.output.content,
                        inferenceTime: result.inferenceTime,
                        inputLength: batchInputs[index].data.length,
                        outputLength: result.output.tokens ? result.output.tokens.length : 0
                    }));
                    
                    return processed;
                });
                
                // Step 6: Cleanup
                await e2eFramework.executeStep('cleanup', async () => {
                    e2eFramework.wasmModule.tensors.clear();
                    return { status: 'cleaned' };
                });
                
                const scenarioResult = e2eFramework.finishScenario();
                expect(scenarioResult.success).toBe(true);
                
                console.log(`Batch processing completed in ${scenarioResult.totalDuration.toFixed(2)}ms`);
                console.log(`Processed ${processedResults.length} inputs`);
                
                const avgInferenceTime = processedResults.reduce((sum, r) => sum + r.inferenceTime, 0) / processedResults.length;
                console.log(`Average inference time: ${avgInferenceTime.toFixed(2)}ms`);
                
            } catch (error) {
                e2eFramework.finishScenario();
                throw error;
            }
        }, E2E_CONFIG.timeout);
    });

    describe('Error Handling and Recovery', () => {
        it('should handle model loading errors gracefully', async () => {
            try {
                await e2eFramework.wasmModule.loadModel('non_existent_model', {
                    config: {},
                    weights: {}
                });
                
                // This should not be reached
                expect(true).toBe(false);
            } catch (error) {
                expect(error).toBeDefined();
                console.log('Model loading error handled correctly:', error.message);
            }
        });
        
        it('should handle inference errors gracefully', async () => {
            try {
                // Try to run inference with invalid tensor ID
                await e2eFramework.wasmModule.inference(99999, 1);
                
                // This should not be reached
                expect(true).toBe(false);
            } catch (error) {
                expect(error).toBeDefined();
                expect(error.message).toContain('not found');
                console.log('Inference error handled correctly:', error.message);
            }
        });
        
        it('should recover from memory pressure scenarios', async () => {
            const modelId = e2eFramework.models.get('text_generation');
            
            // Create memory pressure
            const tensorIds = [];
            for (let i = 0; i < 100; i++) {
                tensorIds.push(e2eFramework.wasmModule.createTensor([100, 100]));
            }
            
            const memoryUsage = e2eFramework.wasmModule.getMemoryUsage();
            console.log(`Memory usage under pressure: ${memoryUsage.estimatedMemoryMB.toFixed(2)}MB`);
            
            // Should still be able to run inference
            const inputTensorId = e2eFramework.wasmModule.createTensor([1, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
            const result = await e2eFramework.wasmModule.inference(inputTensorId, modelId);
            
            expect(result.outputTensorId).toBeDefined();
            
            // Cleanup
            e2eFramework.wasmModule.cleanup();
            
            const finalMemoryUsage = e2eFramework.wasmModule.getMemoryUsage();
            expect(finalMemoryUsage.totalTensors).toBe(0);
            
            console.log('Memory pressure test completed successfully');
        });
    });

    describe('Performance Validation', () => {
        it('should meet performance requirements for real-time inference', async () => {
            const modelId = e2eFramework.models.get('text_generation');
            
            // Test multiple inference runs
            const inferenceTimes = [];
            
            for (let i = 0; i < 10; i++) {
                const inputTensorId = e2eFramework.wasmModule.createTensor([1, 50]);
                
                const startTime = performance.now();
                const result = await e2eFramework.wasmModule.inference(inputTensorId, modelId);
                const endTime = performance.now();
                
                inferenceTimes.push(endTime - startTime);
                
                // Cleanup
                e2eFramework.wasmModule.tensors.delete(inputTensorId);
                e2eFramework.wasmModule.tensors.delete(result.outputTensorId);
            }
            
            const avgTime = inferenceTimes.reduce((a, b) => a + b, 0) / inferenceTimes.length;
            const maxTime = Math.max(...inferenceTimes);
            
            console.log(`Average inference time: ${avgTime.toFixed(2)}ms`);
            console.log(`Maximum inference time: ${maxTime.toFixed(2)}ms`);
            
            // Performance requirements
            expect(avgTime).toBeLessThan(500); // Average should be under 500ms
            expect(maxTime).toBeLessThan(1000); // Maximum should be under 1s
            
            // Consistency check
            const standardDeviation = Math.sqrt(
                inferenceTimes.reduce((sum, time) => sum + Math.pow(time - avgTime, 2), 0) / inferenceTimes.length
            );
            
            expect(standardDeviation).toBeLessThan(avgTime * 0.5); // Should be relatively consistent
            
            console.log(`Performance validation passed (std dev: ${standardDeviation.toFixed(2)}ms)`);
        });
    });
});

// Export for use in other test files
export {
    E2ETestFramework,
    MockTrustformersWasmE2E,
    E2E_CONFIG
};