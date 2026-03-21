/**
 * Test Configuration and Environment Detection for TrustformeRS
 *
 * Handles different test environments and provides graceful fallbacks
 */

import path from 'path';
import { fileURLToPath } from 'url';
import process from 'process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class TestEnvironment {
  constructor() {
    this.wasmAvailable = false;
    this.webglAvailable = false;
    this.webgpuAvailable = false;
    this.trustformersInitialized = false;
    this.skipIntegrationTests = false;
    this.mockMode = false;

    this.capabilities = {
      tensor: false,
      models: false,
      pipeline: false,
      webgl: false,
      webgpu: false,
      nodejs: true,
      browser: false
    };

    this.testConfig = {
      timeout: {
        unit: 5000,        // 5 seconds for unit tests
        integration: 15000, // 15 seconds for integration tests
        performance: 30000, // 30 seconds for performance tests
        stress: 60000      // 1 minute for stress tests
      },
      memory: {
        warningThreshold: 500,   // MB
        errorThreshold: 1000,    // MB
        trackingEnabled: true
      },
      performance: {
        benchmarkIterations: 100,
        warmupIterations: 10,
        measureMemory: true,
        profilerEnabled: false
      },
      mock: {
        tensorOperations: true,
        modelInference: true,
        wasmFallback: true
      }
    };
  }

  async initialize() {
    console.log('🔧 Initializing test environment...');

    await this.detectCapabilities();
    await this.setupMockEnvironment();
    this.configureTestBehavior();

    console.log('✅ Test environment initialized');
    this.printEnvironmentInfo();
  }

  async detectCapabilities() {
    // Check if WASM module is available
    try {
      const wasmPath = path.resolve(__dirname, '../pkg/trustformers_wasm.js');
      const { stat } = await import('fs/promises');
      await stat(wasmPath);
      this.wasmAvailable = true;
      console.log('✅ WASM module found');
    } catch (error) {
      this.wasmAvailable = false;
      console.log('⚠️  WASM module not found - using mock mode');
      this.mockMode = true;
    }

    // Try to initialize TrustformeRS if WASM is available
    if (this.wasmAvailable) {
      try {
        const trustformers = await import('../src/index.js');
        await trustformers.initialize();
        this.trustformersInitialized = true;
        this.capabilities.tensor = true;
        this.capabilities.models = true;
        this.capabilities.pipeline = true;
        console.log('✅ TrustformeRS initialized successfully');
      } catch (error) {
        console.log(`⚠️  TrustformeRS initialization failed: ${error.message}`);
        this.mockMode = true;
      }
    }

    // Detect Node.js specific capabilities
    this.capabilities.nodejs = typeof process !== 'undefined';

    // Check for garbage collection availability
    this.gcAvailable = typeof global.gc === 'function';
    if (!this.gcAvailable) {
      console.log('ℹ️  Garbage collection not available (run with --expose-gc for better memory testing)');
    }

    // Set integration test behavior
    if (!this.trustformersInitialized) {
      this.skipIntegrationTests = true;
      console.log('⏭️  Integration tests will be skipped due to missing WASM');
    }
  }

  async setupMockEnvironment() {
    if (this.mockMode) {
      console.log('🎭 Setting up mock environment...');

      // Create global mock objects that tests can use
      global.mockTrustformers = {
        initialized: true,
        tensor: {
          create: (data, shape) => this.createMockTensor(data, shape),
          zeros: (shape) => this.createMockTensor(new Float32Array(this.calculateSize(shape)).fill(0), shape),
          ones: (shape) => this.createMockTensor(new Float32Array(this.calculateSize(shape)).fill(1), shape),
          random: (shape) => {
            const size = this.calculateSize(shape);
            const data = new Float32Array(size);
            for (let i = 0; i < size; i++) {
              data[i] = Math.random();
            }
            return this.createMockTensor(data, shape);
          }
        },
        model: {
          create: (config) => this.createMockModel(config),
          load: async (modelPath) => this.createMockModel({ name: modelPath })
        },
        pipeline: {
          create: (task, model, tokenizer) => this.createMockPipeline(task, model, tokenizer)
        }
      };

      console.log('✅ Mock environment ready');
    }
  }

  createMockTensor(data, shape) {
    const size = this.calculateSize(shape);
    const tensorData = data instanceof Float32Array ? data : new Float32Array(data || size);

    return {
      data: tensorData,
      shape: shape || [tensorData.length],
      dtype: 'float32',
      size,
      ndim: shape ? shape.length : 1,

      // Mock operations
      add: (other) => this.createMockTensor(
        tensorData.map((val, i) => val + (other.data?.[i] || other)),
        shape
      ),
      sub: (other) => this.createMockTensor(
        tensorData.map((val, i) => val - (other.data?.[i] || other)),
        shape
      ),
      mul: (other) => this.createMockTensor(
        tensorData.map((val, i) => val * (other.data?.[i] || other)),
        shape
      ),
      div: (other) => this.createMockTensor(
        tensorData.map((val, i) => val / (other.data?.[i] || other)),
        shape
      ),
      matmul: (other) => {
        // Simple matrix multiplication mock
        if (shape.length === 2 && other.shape.length === 2) {
          const [m, k] = shape;
          const [k2, n] = other.shape;
          if (k !== k2) throw new Error('Shape mismatch for matrix multiplication');

          const resultShape = [m, n];
          const resultSize = m * n;
          const result = new Float32Array(resultSize);

          for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
              let sum = 0;
              for (let l = 0; l < k; l++) {
                sum += tensorData[i * k + l] * other.data[l * n + j];
              }
              result[i * n + j] = sum;
            }
          }

          return this.createMockTensor(result, resultShape);
        }
        throw new Error('Matrix multiplication not supported for these shapes');
      },
      transpose: () => {
        if (shape.length === 2) {
          const [m, n] = shape;
          const result = new Float32Array(size);
          for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
              result[j * m + i] = tensorData[i * n + j];
            }
          }
          return this.createMockTensor(result, [n, m]);
        }
        return this; // For non-2D tensors, return as-is
      },
      reshape: (newShape) => {
        if (this.calculateSize(newShape) !== size) {
          throw new Error('Cannot reshape: size mismatch');
        }
        return this.createMockTensor(tensorData, newShape);
      },
      sum: (axis) => {
        if (axis === undefined) {
          return tensorData.reduce((sum, val) => sum + val, 0);
        }
        // Simplified axis reduction
        return this.createMockTensor([tensorData.reduce((sum, val) => sum + val, 0)], [1]);
      },
      mean: () => {
        const sum = tensorData.reduce((sum, val) => sum + val, 0);
        return sum / tensorData.length;
      },
      toString: () => `MockTensor(shape=[${shape.join(', ')}], dtype=float32)`
    };
  }

  createMockModel(config = {}) {
    return {
      name: config.name || 'mock-model',
      config,
      loaded: true,

      forward: async (inputs) => {
        // Simple mock forward pass
        if (inputs.data) {
          const outputData = new Float32Array(inputs.data.length);
          for (let i = 0; i < inputs.data.length; i++) {
            outputData[i] = Math.tanh(inputs.data[i]); // Simple activation
          }
          return this.createMockTensor(outputData, inputs.shape);
        }
        return this.createMockTensor([0.5, 0.3, 0.2], [3]); // Mock classification output
      },

      generate: async (inputs, options = {}) => {
        // Mock text generation
        const maxLength = options.max_length || 50;
        const tokens = [];
        for (let i = 0; i < maxLength; i++) {
          tokens.push(Math.floor(Math.random() * 1000)); // Random token IDs
        }
        return { tokens, text: 'Mock generated text ' + tokens.slice(0, 5).join(' ') };
      },

      getParameterCount: () => 1000000, // 1M parameters

      getMemoryUsage: () => ({
        model: 10,     // MB
        cache: 5,      // MB
        total: 15      // MB
      })
    };
  }

  createMockPipeline(task, model, tokenizer) {
    return {
      task,
      model: model || this.createMockModel(),
      tokenizer: tokenizer || this.createMockTokenizer(),

      predict: async (inputs, options = {}) => {
        switch (task) {
          case 'text-classification':
            return [
              { label: 'positive', score: 0.8 },
              { label: 'negative', score: 0.2 }
            ];

          case 'question-answering':
            return {
              answer: 'Mock answer',
              start: 0,
              end: 11,
              score: 0.9
            };

          case 'text-generation':
            return {
              generated_text: inputs + ' mock generated continuation'
            };

          case 'feature-extraction':
            return this.createMockTensor(
              Array.from({ length: 768 }, () => Math.random()),
              [768]
            );

          default:
            return { result: 'mock_result' };
        }
      }
    };
  }

  createMockTokenizer() {
    return {
      encode: (text) => {
        // Simple mock tokenization
        return text.split(' ').map((_, i) => i + 1);
      },

      decode: (tokens) => {
        return tokens.map(t => `token_${t}`).join(' ');
      },

      getVocabSize: () => 50000
    };
  }

  calculateSize(shape) {
    return shape.reduce((acc, dim) => acc * dim, 1);
  }

  configureTestBehavior() {
    // Adjust test configuration based on environment
    if (this.mockMode) {
      this.testConfig.timeout.integration = 5000; // Faster for mock tests
      this.testConfig.performance.benchmarkIterations = 50; // Fewer iterations
    }

    if (!this.gcAvailable) {
      this.testConfig.memory.trackingEnabled = false;
    }
  }

  printEnvironmentInfo() {
    console.log('\n🌍 Test Environment Information:');
    console.log('─'.repeat(50));
    console.log(`Node.js: ${process.version}`);
    console.log(`Platform: ${process.platform} ${process.arch}`);
    console.log(`WASM Available: ${this.wasmAvailable ? '✅' : '❌'}`);
    console.log(`TrustformeRS Initialized: ${this.trustformersInitialized ? '✅' : '❌'}`);
    console.log(`Mock Mode: ${this.mockMode ? '✅' : '❌'}`);
    console.log(`GC Available: ${this.gcAvailable ? '✅' : '❌'}`);
    console.log(`Integration Tests: ${this.skipIntegrationTests ? '⏭️ Skipped' : '✅ Enabled'}`);

    console.log('\n🔧 Capabilities:');
    Object.entries(this.capabilities).forEach(([key, value]) => {
      console.log(`  ${key}: ${value ? '✅' : '❌'}`);
    });

    console.log('\n⚙️ Test Configuration:');
    console.log(`  Unit Test Timeout: ${this.testConfig.timeout.unit}ms`);
    console.log(`  Integration Test Timeout: ${this.testConfig.timeout.integration}ms`);
    console.log(`  Performance Test Timeout: ${this.testConfig.timeout.performance}ms`);
    console.log(`  Memory Tracking: ${this.testConfig.memory.trackingEnabled ? '✅' : '❌'}`);
    console.log(`  Benchmark Iterations: ${this.testConfig.performance.benchmarkIterations}`);
    console.log();
  }

  // Helper methods for tests
  shouldSkipTest(requiredCapabilities = []) {
    return requiredCapabilities.some(cap => !this.capabilities[cap]);
  }

  getTestTimeout(testType = 'unit') {
    return this.testConfig.timeout[testType] || this.testConfig.timeout.unit;
  }

  isWasmAvailable() {
    return this.wasmAvailable && this.trustformersInitialized;
  }

  isMockMode() {
    return this.mockMode;
  }

  // Coverage tracking utilities
  getCoverageInfo() {
    return {
      totalTests: 0,
      passedTests: 0,
      failedTests: 0,
      skippedTests: 0,
      categories: {
        tensor: { total: 0, passed: 0, failed: 0, skipped: 0 },
        models: { total: 0, passed: 0, failed: 0, skipped: 0 },
        pipeline: { total: 0, passed: 0, failed: 0, skipped: 0 },
        performance: { total: 0, passed: 0, failed: 0, skipped: 0 },
        integration: { total: 0, passed: 0, failed: 0, skipped: 0 }
      }
    };
  }
}

// Test utilities and helpers
export class TestUtilities {
  static async withTimeout(promise, timeoutMs = 5000) {
    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(() => reject(new Error(`Operation timed out after ${timeoutMs}ms`)), timeoutMs)
    );

    return Promise.race([promise, timeoutPromise]);
  }

  static async measurePerformance(operation, label = 'Operation') {
    const start = performance.now();
    const memBefore = process.memoryUsage?.() || {};

    const result = await operation();

    const end = performance.now();
    const memAfter = process.memoryUsage?.() || {};

    return {
      result,
      duration: end - start,
      memoryDelta: memAfter.heapUsed && memBefore.heapUsed
        ? Math.round((memAfter.heapUsed - memBefore.heapUsed) / 1024 / 1024)
        : 0,
      label
    };
  }

  static createTensorComparisonHelper() {
    return {
      expectTensorsEqual: (tensor1, tensor2, tolerance = 1e-6) => {
        if (!tensor1 || !tensor2) {
          throw new Error('Cannot compare null/undefined tensors');
        }

        if (JSON.stringify(tensor1.shape) !== JSON.stringify(tensor2.shape)) {
          throw new Error(`Shape mismatch: ${JSON.stringify(tensor1.shape)} vs ${JSON.stringify(tensor2.shape)}`);
        }

        if (tensor1.data && tensor2.data) {
          for (let i = 0; i < tensor1.data.length; i++) {
            if (Math.abs(tensor1.data[i] - tensor2.data[i]) > tolerance) {
              throw new Error(`Values differ at index ${i}: ${tensor1.data[i]} vs ${tensor2.data[i]}`);
            }
          }
        }
      },

      expectTensorShape: (tensor, expectedShape) => {
        if (!tensor || !tensor.shape) {
          throw new Error('Tensor is null or has no shape');
        }

        if (JSON.stringify(tensor.shape) !== JSON.stringify(expectedShape)) {
          throw new Error(`Expected shape ${JSON.stringify(expectedShape)}, got ${JSON.stringify(tensor.shape)}`);
        }
      },

      expectTensorValue: (tensor, expectedValue, tolerance = 1e-6) => {
        if (!tensor || tensor.data === undefined) {
          throw new Error('Tensor is null or has no data');
        }

        if (typeof expectedValue === 'number') {
          // Check if all values are close to expected
          for (let i = 0; i < tensor.data.length; i++) {
            if (Math.abs(tensor.data[i] - expectedValue) > tolerance) {
              throw new Error(`Expected all values to be ${expectedValue}, but found ${tensor.data[i]} at index ${i}`);
            }
          }
        }
      }
    };
  }
}

// Global test environment instance
export const testEnv = new TestEnvironment();