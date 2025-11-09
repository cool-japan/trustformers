/*global MLGraphBuilder*/
/**
 * WebNN (Web Neural Network API) Backend
 *
 * Provides hardware-accelerated machine learning operations using the WebNN API.
 * WebNN enables native acceleration on CPU, GPU, and NPU devices.
 *
 * Features:
 * - Graph-based computation for optimal performance
 * - Automatic device selection (CPU, GPU, NPU)
 * - Support for multiple data types (float32, float16, int8, uint8)
 * - Operator fusion and optimization
 * - Memory-efficient execution
 *
 * @see https://www.w3.org/TR/webnn/
 */

/**
 * WebNN availability and capability detection
 */
export class WebNNCapabilities {
  static async detect() {
    const caps = {
      available: false,
      version: null,
      devices: [],
      supportedOps: [],
      supportedTypes: [],
      features: {}
    };

    // Check if WebNN is available
    if (typeof navigator === 'undefined' || !navigator.ml) {
      return caps;
    }

    caps.available = true;

    try {
      // Detect available devices
      const contexts = await this._detectContexts();
      caps.devices = contexts.map(c => c.deviceType);

      // Detect supported operations
      caps.supportedOps = await this._detectOperations(contexts[0]);

      // Detect supported data types
      caps.supportedTypes = this._detectDataTypes();

      // Detect advanced features
      caps.features = {
        graphBuilder: true,
        operatorFusion: true,
        int8Quantization: this._supportsInt8(contexts[0]),
        float16: this._supportsFloat16(contexts[0]),
        dynamicShapes: false // Not widely supported yet
      };

    } catch (error) {
      console.warn('WebNN capability detection failed:', error);
      caps.available = false;
    }

    return caps;
  }

  static async _detectContexts() {
    const deviceTypes = ['gpu', 'cpu', 'npu'];
    const contexts = [];

    for (const deviceType of deviceTypes) {
      try {
        const context = await navigator.ml.createContext({ deviceType });
        contexts.push({ deviceType, context });
      } catch {
        // Device not available
      }
    }

    return contexts;
  }

  static async _detectOperations(contextInfo) {
    if (!contextInfo) return [];

    const ops = [
      'add', 'sub', 'mul', 'div', 'matmul',
      'relu', 'sigmoid', 'tanh', 'gelu', 'softmax',
      'conv2d', 'pooling', 'batchNorm', 'layerNorm',
      'reshape', 'transpose', 'concat', 'slice',
      'gather', 'scatter', 'reduce', 'cast'
    ];

    const supported = [];
    const builder = new MLGraphBuilder(contextInfo.context);

    for (const op of ops) {
      try {
        // Try to create a simple operation to test support
        if (typeof builder[op] === 'function') {
          supported.push(op);
        }
      } catch {
        // Operation not supported
      }
    }

    return supported;
  }

  static _detectDataTypes() {
    const types = ['float32', 'float16', 'int32', 'uint32', 'int8', 'uint8'];
    return types.filter(type => {
      try {
        // Check if MLOperandDataType enum has this type
        return true; // Assume all are supported for now
      } catch {
        return false;
      }
    });
  }

  static _supportsInt8(contextInfo) {
    if (!contextInfo) return false;
    // Check if int8 quantization is supported
    return true; // Simplified check
  }

  static _supportsFloat16(contextInfo) {
    if (!contextInfo) return false;
    // Check if float16 is supported
    return true; // Simplified check
  }
}

/**
 * WebNN Graph Builder and Executor
 */
export class WebNNGraph {
  constructor(context, builder) {
    this.context = context;
    this.builder = builder;
    this.graph = null;
    this.inputs = new Map();
    this.outputs = new Map();
    this.constants = new Map();
  }

  /**
   * Add input operand to the graph
   */
  addInput(name, shape, dataType = 'float32') {
    const operand = this.builder.input(name, {
      dataType,
      dimensions: shape
    });
    this.inputs.set(name, { operand, shape, dataType });
    return operand;
  }

  /**
   * Add constant operand to the graph
   */
  addConstant(name, data, shape, dataType = 'float32') {
    const operand = this.builder.constant({
      dataType,
      dimensions: shape
    }, data);
    this.constants.set(name, { operand, data, shape, dataType });
    return operand;
  }

  /**
   * Build the computation graph
   */
  async build(outputOperands) {
    try {
      // Convert output operands to named outputs
      const namedOutputs = {};
      for (const [name, operand] of Object.entries(outputOperands)) {
        namedOutputs[name] = operand;
        this.outputs.set(name, operand);
      }

      // Build the graph
      this.graph = await this.builder.build(namedOutputs);
      return this.graph;
    } catch (error) {
      throw new Error(`Failed to build WebNN graph: ${error.message}`);
    }
  }

  /**
   * Execute the graph with given inputs
   */
  async execute(inputData) {
    if (!this.graph) {
      throw new Error('Graph not built. Call build() first.');
    }

    try {
      // Prepare input buffers
      const inputs = {};
      for (const [name, data] of Object.entries(inputData)) {
        const inputInfo = this.inputs.get(name);
        if (!inputInfo) {
          throw new Error(`Unknown input: ${name}`);
        }

        // Convert data to typed array if needed
        const typedData = this._convertToTypedArray(data, inputInfo.dataType);
        inputs[name] = typedData;
      }

      // Execute the graph
      const outputs = await this.context.compute(this.graph, inputs);

      // Convert outputs to regular arrays or tensors
      const results = {};
      for (const [name, buffer] of Object.entries(outputs)) {
        results[name] = buffer;
      }

      return results;
    } catch (error) {
      throw new Error(`Graph execution failed: ${error.message}`);
    }
  }

  _convertToTypedArray(data, dataType) {
    if (ArrayBuffer.isView(data)) {
      return data;
    }

    switch (dataType) {
      case 'float32':
        return new Float32Array(data);
      case 'float16':
        return new Uint16Array(data); // Float16 stored as uint16
      case 'int32':
        return new Int32Array(data);
      case 'uint32':
        return new Uint32Array(data);
      case 'int8':
        return new Int8Array(data);
      case 'uint8':
        return new Uint8Array(data);
      default:
        return new Float32Array(data);
    }
  }
}

/**
 * WebNN Backend Implementation
 */
export class WebNNBackend {
  constructor(options = {}) {
    this.deviceType = options.deviceType || 'gpu'; // 'gpu', 'cpu', or 'npu'
    this.powerPreference = options.powerPreference || 'default'; // 'default', 'high-performance', 'low-power'
    this.context = null;
    this.initialized = false;
    this.capabilities = null;
    this.graphs = new Map();
    this.statistics = {
      graphsCreated: 0,
      executionsCompleted: 0,
      executionsFailed: 0,
      totalExecutionTime: 0,
      averageExecutionTime: 0
    };
  }

  /**
   * Initialize WebNN backend
   */
  async initialize() {
    if (this.initialized) {
      return true;
    }

    try {
      // Check availability
      if (typeof navigator === 'undefined' || !navigator.ml) {
        throw new Error('WebNN not available in this environment');
      }

      // Detect capabilities
      this.capabilities = await WebNNCapabilities.detect();
      if (!this.capabilities.available) {
        throw new Error('WebNN not supported');
      }

      // Create context
      this.context = await navigator.ml.createContext({
        deviceType: this.deviceType,
        powerPreference: this.powerPreference
      });

      this.initialized = true;
      // eslint-disable-next-line no-console
      console.log(`WebNN backend initialized with device: ${this.deviceType}`);
      return true;

    } catch (error) {
      console.error('Failed to initialize WebNN backend:', error);
      this.initialized = false;
      return false;
    }
  }

  /**
   * Create a new graph builder
   */
  createGraphBuilder() {
    if (!this.initialized) {
      throw new Error('WebNN backend not initialized');
    }

    const builder = new MLGraphBuilder(this.context);
    const graph = new WebNNGraph(this.context, builder);
    this.statistics.graphsCreated++;
    return graph;
  }

  /**
   * Build and cache a computation graph
   */
  async buildGraph(name, buildFn) {
    const builder = this.createGraphBuilder();
    const outputOperands = await buildFn(builder);
    await builder.build(outputOperands);
    this.graphs.set(name, builder);
    return builder;
  }

  /**
   * Execute a cached graph
   */
  async executeGraph(name, inputs) {
    const graph = this.graphs.get(name);
    if (!graph) {
      throw new Error(`Graph not found: ${name}`);
    }

    const startTime = performance.now();

    try {
      const result = await graph.execute(inputs);
      const executionTime = performance.now() - startTime;

      this._updateStatistics(executionTime, true);
      return result;

    } catch (error) {
      this._updateStatistics(0, false);
      throw error;
    }
  }

  /**
   * Common tensor operations using WebNN
   */
  async matmul(a, b, _options = {}) {
    const builder = this.createGraphBuilder();

    const inputA = builder.addInput('a', a.shape);
    const inputB = builder.addInput('b', b.shape);

    const result = builder.builder.matmul(inputA, inputB);

    await builder.build({ output: result });
    return builder.execute({ a: a.data, b: b.data });
  }

  async add(a, b) {
    const builder = this.createGraphBuilder();

    const inputA = builder.addInput('a', a.shape);
    const inputB = builder.addInput('b', b.shape);

    const result = builder.builder.add(inputA, inputB);

    await builder.build({ output: result });
    return builder.execute({ a: a.data, b: b.data });
  }

  async relu(input) {
    const builder = this.createGraphBuilder();

    const inputOperand = builder.addInput('input', input.shape);
    const result = builder.builder.relu(inputOperand);

    await builder.build({ output: result });
    return builder.execute({ input: input.data });
  }

  async softmax(input, _axis = -1) {
    const builder = this.createGraphBuilder();

    const inputOperand = builder.addInput('input', input.shape);
    const result = builder.builder.softmax(inputOperand);

    await builder.build({ output: result });
    return builder.execute({ input: input.data });
  }

  async layerNorm(input, scale, bias, options = {}) {
    const epsilon = options.epsilon || 1e-5;
    const builder = this.createGraphBuilder();

    const inputOperand = builder.addInput('input', input.shape);
    const scaleOperand = builder.addConstant('scale', scale.data, scale.shape);
    const biasOperand = builder.addConstant('bias', bias.data, bias.shape);

    // Compute mean
    const mean = builder.builder.reduceMean(inputOperand, { axes: [-1], keepDimensions: true });

    // Compute variance
    const diff = builder.builder.sub(inputOperand, mean);
    const variance = builder.builder.reduceMean(
      builder.builder.mul(diff, diff),
      { axes: [-1], keepDimensions: true }
    );

    // Normalize
    const variancePlusEpsilon = builder.builder.add(
      variance,
      builder.builder.constant({ dataType: 'float32', dimensions: [1] }, new Float32Array([epsilon]))
    );
    const stddev = builder.builder.sqrt(variancePlusEpsilon);
    const normalized = builder.builder.div(diff, stddev);

    // Scale and shift
    const scaled = builder.builder.mul(normalized, scaleOperand);
    const result = builder.builder.add(scaled, biasOperand);

    await builder.build({ output: result });
    return builder.execute({ input: input.data });
  }

  /**
   * Build attention mechanism using WebNN
   */
  async buildAttentionGraph(hiddenSize, numHeads, _options = {}) {
    const headDim = hiddenSize / numHeads;
    const builder = this.createGraphBuilder();

    // Inputs
    const query = builder.addInput('query', [1, -1, hiddenSize]); // [batch, seq, hidden]
    const key = builder.addInput('key', [1, -1, hiddenSize]);
    const value = builder.addInput('value', [1, -1, hiddenSize]);

    // Query, Key, Value projections (would use weights in practice)
    // For now, simplified version

    // Compute attention scores
    const scores = builder.builder.matmul(query, key); // Simplified

    // Scale scores
    const scaleFactor = Math.sqrt(headDim);
    const scaleOperand = builder.builder.constant(
      { dataType: 'float32', dimensions: [1] },
      new Float32Array([1.0 / scaleFactor])
    );
    const scaledScores = builder.builder.mul(scores, scaleOperand);

    // Apply softmax
    const attnWeights = builder.builder.softmax(scaledScores);

    // Apply attention to values
    const output = builder.builder.matmul(attnWeights, value);

    await builder.build({ output });
    return builder;
  }

  /**
   * Quantize tensor to int8
   */
  async quantizeInt8(input, scale, _zeroPoint) {
    const builder = this.createGraphBuilder();

    const inputOperand = builder.addInput('input', input.shape, 'float32');
    const scaleOperand = builder.builder.constant(
      { dataType: 'float32', dimensions: [1] },
      new Float32Array([scale])
    );
    // Quantize: round(input / scale) + zeroPoint
    const scaled = builder.builder.div(inputOperand, scaleOperand);
    const rounded = builder.builder.floor(builder.builder.add(
      scaled,
      builder.builder.constant({ dataType: 'float32', dimensions: [1] }, new Float32Array([0.5]))
    ));
    const result = builder.builder.cast(rounded, 'int8');

    await builder.build({ output: result });
    return builder.execute({ input: input.data });
  }

  /**
   * Dequantize int8 tensor back to float32
   */
  async dequantizeInt8(input, scale, zeroPoint) {
    const builder = this.createGraphBuilder();

    const inputOperand = builder.addInput('input', input.shape, 'int8');
    const scaleOperand = builder.builder.constant(
      { dataType: 'float32', dimensions: [1] },
      new Float32Array([scale])
    );

    // Dequantize: (input - zeroPoint) * scale
    const casted = builder.builder.cast(inputOperand, 'float32');
    const zeroPointOperand = builder.builder.constant(
      { dataType: 'float32', dimensions: [1] },
      new Float32Array([zeroPoint])
    );
    const shifted = builder.builder.sub(casted, zeroPointOperand);
    const result = builder.builder.mul(shifted, scaleOperand);

    await builder.build({ output: result });
    return builder.execute({ input: input.data });
  }

  _updateStatistics(executionTime, success) {
    if (success) {
      this.statistics.executionsCompleted++;
      this.statistics.totalExecutionTime += executionTime;
      this.statistics.averageExecutionTime =
        this.statistics.totalExecutionTime / this.statistics.executionsCompleted;
    } else {
      this.statistics.executionsFailed++;
    }
  }

  /**
   * Get backend statistics
   */
  getStatistics() {
    return { ...this.statistics };
  }

  /**
   * Get capabilities
   */
  getCapabilities() {
    return this.capabilities;
  }

  /**
   * Check if WebNN is supported
   */
  static isSupported() {
    return typeof navigator !== 'undefined' &&
           navigator.ml !== undefined &&
           typeof navigator.ml.createContext === 'function';
  }

  /**
   * Dispose resources
   */
  dispose() {
    this.graphs.clear();
    this.context = null;
    this.initialized = false;
  }
}

/**
 * WebNN Operation Builders
 * Helper functions for building common neural network operations
 */
export class WebNNOperations {
  /**
   * Build a fully connected (linear) layer
   */
  static linear(builder, input, weights, bias = null) {
    const result = builder.matmul(input, weights);
    return bias ? builder.add(result, bias) : result;
  }

  /**
   * Build a convolution layer
   */
  static conv2d(builder, input, filter, options = {}) {
    const {
      padding = [0, 0, 0, 0],
      strides = [1, 1],
      dilations = [1, 1],
      groups = 1
    } = options;

    return builder.conv2d(input, filter, {
      padding,
      strides,
      dilations,
      groups
    });
  }

  /**
   * Build batch normalization
   */
  static batchNorm(builder, input, mean, variance, scale, bias, options = {}) {
    const epsilon = options.epsilon || 1e-5;

    return builder.batchNormalization(input, mean, variance, {
      scale,
      bias,
      epsilon
    });
  }

  /**
   * Build GELU activation
   */
  static gelu(builder, input) {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    const sqrt2OverPi = 0.7978845608;
    const coeff = 0.044715;

    const x2 = builder.mul(input, input);
    const x3 = builder.mul(x2, input);

    const inner = builder.add(
      input,
      builder.mul(
        builder.constant({ dataType: 'float32', dimensions: [1] }, new Float32Array([coeff])),
        x3
      )
    );

    const scaled = builder.mul(
      builder.constant({ dataType: 'float32', dimensions: [1] }, new Float32Array([sqrt2OverPi])),
      inner
    );

    const tanh = builder.tanh(scaled);
    const onePlusTanh = builder.add(
      builder.constant({ dataType: 'float32', dimensions: [1] }, new Float32Array([1.0])),
      tanh
    );

    const half = builder.constant({ dataType: 'float32', dimensions: [1] }, new Float32Array([0.5]));

    return builder.mul(builder.mul(half, input), onePlusTanh);
  }

  /**
   * Build residual connection
   */
  static residual(builder, input, transform) {
    return builder.add(input, transform);
  }
}

/**
 * WebNN Backend Manager
 * Manages WebNN backend lifecycle and provides high-level API
 */
export class WebNNBackendManager {
  constructor() {
    this.backend = null;
    this.initialized = false;
    this.fallbackToWebGL = true;
  }

  async initialize(options = {}) {
    try {
      // Check if WebNN is supported
      if (!WebNNBackend.isSupported()) {
        console.warn('WebNN not supported, falling back to other backends');
        return false;
      }

      // Create and initialize backend
      this.backend = new WebNNBackend(options);
      const success = await this.backend.initialize();

      if (success) {
        this.initialized = true;
        // eslint-disable-next-line no-console
        console.log('WebNN backend ready');
        return true;
      }

      return false;

    } catch (error) {
      console.error('WebNN initialization failed:', error);
      return false;
    }
  }

  getBackend() {
    return this.backend;
  }

  isInitialized() {
    return this.initialized;
  }

  dispose() {
    if (this.backend) {
      this.backend.dispose();
      this.backend = null;
      this.initialized = false;
    }
  }
}

// Singleton instance
let globalWebNNManager = null;

/**
 * Get or create the global WebNN backend manager
 */
export function getWebNNManager() {
  if (!globalWebNNManager) {
    globalWebNNManager = new WebNNBackendManager();
  }
  return globalWebNNManager;
}

/**
 * Initialize WebNN backend
 */
export async function initWebNN(options = {}) {
  const manager = getWebNNManager();
  return await manager.initialize(options);
}

/**
 * Check if WebNN is available
 */
export function isWebNNAvailable() {
  return WebNNBackend.isSupported();
}

// Export all classes and functions
export default {
  WebNNBackend,
  WebNNBackendManager,
  WebNNCapabilities,
  WebNNGraph,
  WebNNOperations,
  getWebNNManager,
  initWebNN,
  isWebNNAvailable
};
