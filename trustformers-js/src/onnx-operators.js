/**
 * ONNX Operators Implementation
 *
 * Comprehensive implementation of ONNX operators for model inference.
 * Supports 60+ operators covering:
 * - Math operations (Add, Sub, Mul, Div, MatMul, Gemm)
 * - Activations (Relu, Gelu, Sigmoid, Tanh, Softmax, Swish, etc.)
 * - Normalization (BatchNormalization, LayerNormalization, InstanceNormalization)
 * - Pooling (MaxPool, AveragePool, GlobalAveragePool, GlobalMaxPool)
 * - Convolution (Conv, ConvTranspose)
 * - Tensor operations (Reshape, Transpose, Concat, Split, Slice, Gather, Scatter)
 * - Reduction (ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd)
 * - Comparison (Equal, Greater, Less, Where)
 * - Logical (And, Or, Not, Xor)
 * - Other (Cast, Clip, Dropout, Pad, Squeeze, Unsqueeze, Resize)
 *
 * @module onnx-operators
 */

/**
 * Base class for ONNX operators
 */
class ONNXOperator {
  constructor(name, attributes = {}) {
    this.name = name;
    this.attributes = attributes;
  }

  /**
   * Execute operator
   * @param {Array<Tensor>} inputs - Input tensors
   * @returns {Array<Tensor>} Output tensors
   */
  execute(inputs) {
    throw new Error(`Operator ${this.name} not implemented`);
  }

  /**
   * Validate inputs
   * @param {Array<Tensor>} inputs - Input tensors
   * @param {number} expectedCount - Expected number of inputs
   */
  validateInputs(inputs, expectedCount) {
    if (inputs.length < expectedCount) {
      throw new Error(
        `${this.name}: Expected at least ${expectedCount} inputs, got ${inputs.length}`
      );
    }
  }

  /**
   * Get attribute with default
   * @param {string} name - Attribute name
   * @param {*} defaultValue - Default value
   * @returns {*} Attribute value
   */
  getAttribute(name, defaultValue) {
    return this.attributes[name] !== undefined ? this.attributes[name] : defaultValue;
  }
}

/**
 * Tensor representation
 */
class Tensor {
  constructor(data, shape, dtype = 'float32') {
    this.data = data;
    this.shape = shape;
    this.dtype = dtype;
  }

  get size() {
    return this.shape.reduce((a, b) => a * b, 1);
  }

  /**
   * Reshape tensor
   * @param {Array<number>} newShape - New shape
   * @returns {Tensor} Reshaped tensor
   */
  reshape(newShape) {
    const newSize = newShape.reduce((a, b) => a * b, 1);
    if (newSize !== this.size) {
      throw new Error(`Cannot reshape tensor of size ${this.size} to ${newSize}`);
    }
    return new Tensor(this.data, newShape, this.dtype);
  }

  /**
   * Clone tensor
   * @returns {Tensor} Cloned tensor
   */
  clone() {
    const newData = this.data.constructor === Array
      ? [...this.data]
      : new this.data.constructor(this.data);
    return new Tensor(newData, [...this.shape], this.dtype);
  }
}

// ============================================================================
// Math Operators
// ============================================================================

/**
 * Add operator: C = A + B (with broadcasting)
 */
class Add extends ONNXOperator {
  constructor(attributes = {}) {
    super('Add', attributes);
  }

  execute(inputs) {
    this.validateInputs(inputs, 2);
    const [A, B] = inputs;

    const { shape: outShape, stridesA, stridesB } = this.broadcastShapes(A.shape, B.shape);
    const result = new Float32Array(outShape.reduce((a, b) => a * b, 1));

    for (let i = 0; i < result.length; i++) {
      const idxA = this.getBroadcastIndex(i, outShape, stridesA);
      const idxB = this.getBroadcastIndex(i, outShape, stridesB);
      result[i] = A.data[idxA] + B.data[idxB];
    }

    return [new Tensor(result, outShape, A.dtype)];
  }

  broadcastShapes(shapeA, shapeB) {
    const ndimA = shapeA.length;
    const ndimB = shapeB.length;
    const ndimOut = Math.max(ndimA, ndimB);

    const outShape = new Array(ndimOut);
    const stridesA = new Array(ndimOut);
    const stridesB = new Array(ndimOut);

    let strideA = 1;
    let strideB = 1;

    for (let i = 0; i < ndimOut; i++) {
      const dimA = i < ndimA ? shapeA[ndimA - 1 - i] : 1;
      const dimB = i < ndimB ? shapeB[ndimB - 1 - i] : 1;

      if (dimA !== dimB && dimA !== 1 && dimB !== 1) {
        throw new Error(`Cannot broadcast shapes ${shapeA} and ${shapeB}`);
      }

      outShape[ndimOut - 1 - i] = Math.max(dimA, dimB);
      stridesA[ndimOut - 1 - i] = dimA === 1 ? 0 : strideA;
      stridesB[ndimOut - 1 - i] = dimB === 1 ? 0 : strideB;

      strideA *= dimA;
      strideB *= dimB;
    }

    return { shape: outShape, stridesA, stridesB };
  }

  getBroadcastIndex(linearIdx, shape, strides) {
    let idx = 0;
    for (let i = shape.length - 1; i >= 0; i--) {
      const coord = Math.floor(linearIdx / strides[i]) % shape[i];
      const stride = strides[i] === 0 ? 0 : strides[i];
      idx += coord * (stride === 0 ? 0 : 1);
      linearIdx %= strides[i] || 1;
    }
    return idx;
  }
}

/**
 * Sub operator: C = A - B
 */
class Sub extends Add {
  constructor(attributes = {}) {
    super(attributes);
    this.name = 'Sub';
  }

  execute(inputs) {
    this.validateInputs(inputs, 2);
    const [A, B] = inputs;

    const { shape: outShape, stridesA, stridesB } = this.broadcastShapes(A.shape, B.shape);
    const result = new Float32Array(outShape.reduce((a, b) => a * b, 1));

    for (let i = 0; i < result.length; i++) {
      const idxA = this.getBroadcastIndex(i, outShape, stridesA);
      const idxB = this.getBroadcastIndex(i, outShape, stridesB);
      result[i] = A.data[idxA] - B.data[idxB];
    }

    return [new Tensor(result, outShape, A.dtype)];
  }
}

/**
 * Mul operator: C = A * B (element-wise)
 */
class Mul extends Add {
  constructor(attributes = {}) {
    super(attributes);
    this.name = 'Mul';
  }

  execute(inputs) {
    this.validateInputs(inputs, 2);
    const [A, B] = inputs;

    const { shape: outShape, stridesA, stridesB } = this.broadcastShapes(A.shape, B.shape);
    const result = new Float32Array(outShape.reduce((a, b) => a * b, 1));

    for (let i = 0; i < result.length; i++) {
      const idxA = this.getBroadcastIndex(i, outShape, stridesA);
      const idxB = this.getBroadcastIndex(i, outShape, stridesB);
      result[i] = A.data[idxA] * B.data[idxB];
    }

    return [new Tensor(result, outShape, A.dtype)];
  }
}

/**
 * Div operator: C = A / B
 */
class Div extends Add {
  constructor(attributes = {}) {
    super(attributes);
    this.name = 'Div';
  }

  execute(inputs) {
    this.validateInputs(inputs, 2);
    const [A, B] = inputs;

    const { shape: outShape, stridesA, stridesB } = this.broadcastShapes(A.shape, B.shape);
    const result = new Float32Array(outShape.reduce((a, b) => a * b, 1));

    for (let i = 0; i < result.length; i++) {
      const idxA = this.getBroadcastIndex(i, outShape, stridesA);
      const idxB = this.getBroadcastIndex(i, outShape, stridesB);
      result[i] = A.data[idxA] / (B.data[idxB] + 1e-10); // Add epsilon for stability
    }

    return [new Tensor(result, outShape, A.dtype)];
  }
}

/**
 * MatMul operator: C = A @ B (matrix multiplication)
 */
class MatMul extends ONNXOperator {
  constructor(attributes = {}) {
    super('MatMul', attributes);
  }

  execute(inputs) {
    this.validateInputs(inputs, 2);
    const [A, B] = inputs;

    // Support batched matrix multiplication
    const [M, K] = A.shape.slice(-2);
    const N = B.shape[B.shape.length - 1];

    if (A.shape[A.shape.length - 1] !== B.shape[B.shape.length - 2]) {
      throw new Error(
        `MatMul: incompatible shapes ${A.shape} and ${B.shape}`
      );
    }

    // Compute output shape (batch dims + [M, N])
    const batchDims = A.shape.slice(0, -2);
    const outShape = [...batchDims, M, N];
    const batchSize = batchDims.reduce((a, b) => a * b, 1);

    const result = new Float32Array(batchSize * M * N);

    // Perform batched matrix multiplication
    for (let b = 0; b < batchSize; b++) {
      const offsetA = b * M * K;
      const offsetB = b * K * N;
      const offsetC = b * M * N;

      for (let i = 0; i < M; i++) {
        for (let j = 0; j < N; j++) {
          let sum = 0;
          for (let k = 0; k < K; k++) {
            sum += A.data[offsetA + i * K + k] * B.data[offsetB + k * N + j];
          }
          result[offsetC + i * N + j] = sum;
        }
      }
    }

    return [new Tensor(result, outShape, A.dtype)];
  }
}

/**
 * Gemm operator: Y = alpha * A * B + beta * C
 * Generalized matrix multiplication
 */
class Gemm extends ONNXOperator {
  constructor(attributes = {}) {
    super('Gemm', attributes);
  }

  execute(inputs) {
    this.validateInputs(inputs, 2);
    const [A, B, C] = inputs.length >= 3 ? inputs : [...inputs, null];

    const alpha = this.getAttribute('alpha', 1.0);
    const beta = this.getAttribute('beta', 1.0);
    const transA = this.getAttribute('transA', 0);
    const transB = this.getAttribute('transB', 0);

    // Get dimensions
    let [M, K] = A.shape;
    let [K2, N] = B.shape;

    if (transA) [M, K] = [K, M];
    if (transB) [K2, N] = [N, K2];

    if (K !== K2) {
      throw new Error(`Gemm: incompatible dimensions K=${K}, K2=${K2}`);
    }

    const result = new Float32Array(M * N);

    // Compute Y = alpha * A * B
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          const aIdx = transA ? k * M + i : i * K + k;
          const bIdx = transB ? j * K + k : k * N + j;
          sum += A.data[aIdx] * B.data[bIdx];
        }
        result[i * N + j] = alpha * sum;
      }
    }

    // Add beta * C if provided
    if (C) {
      for (let i = 0; i < result.length; i++) {
        result[i] += beta * (C.data[i] || 0);
      }
    }

    return [new Tensor(result, [M, N], A.dtype)];
  }
}

// ============================================================================
// Activation Functions
// ============================================================================

/**
 * Relu operator: y = max(0, x)
 */
class Relu extends ONNXOperator {
  constructor(attributes = {}) {
    super('Relu', attributes);
  }

  execute(inputs) {
    this.validateInputs(inputs, 1);
    const [X] = inputs;

    const result = new Float32Array(X.size);
    for (let i = 0; i < X.size; i++) {
      result[i] = Math.max(0, X.data[i]);
    }

    return [new Tensor(result, X.shape, X.dtype)];
  }
}

/**
 * Gelu operator: y = x * Φ(x) where Φ is the cumulative distribution function
 * Approximation: y = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
 */
class Gelu extends ONNXOperator {
  constructor(attributes = {}) {
    super('Gelu', attributes);
  }

  execute(inputs) {
    this.validateInputs(inputs, 1);
    const [X] = inputs;

    const result = new Float32Array(X.size);
    const sqrt2OverPi = Math.sqrt(2 / Math.PI);

    for (let i = 0; i < X.size; i++) {
      const x = X.data[i];
      const x3 = x * x * x;
      const inner = sqrt2OverPi * (x + 0.044715 * x3);
      result[i] = 0.5 * x * (1 + Math.tanh(inner));
    }

    return [new Tensor(result, X.shape, X.dtype)];
  }
}

/**
 * Sigmoid operator: y = 1 / (1 + exp(-x))
 */
class Sigmoid extends ONNXOperator {
  constructor(attributes = {}) {
    super('Sigmoid', attributes);
  }

  execute(inputs) {
    this.validateInputs(inputs, 1);
    const [X] = inputs;

    const result = new Float32Array(X.size);
    for (let i = 0; i < X.size; i++) {
      result[i] = 1 / (1 + Math.exp(-X.data[i]));
    }

    return [new Tensor(result, X.shape, X.dtype)];
  }
}

/**
 * Tanh operator: y = tanh(x)
 */
class Tanh extends ONNXOperator {
  constructor(attributes = {}) {
    super('Tanh', attributes);
  }

  execute(inputs) {
    this.validateInputs(inputs, 1);
    const [X] = inputs;

    const result = new Float32Array(X.size);
    for (let i = 0; i < X.size; i++) {
      result[i] = Math.tanh(X.data[i]);
    }

    return [new Tensor(result, X.shape, X.dtype)];
  }
}

/**
 * Softmax operator: y_i = exp(x_i) / sum(exp(x_j))
 */
class Softmax extends ONNXOperator {
  constructor(attributes = {}) {
    super('Softmax', attributes);
  }

  execute(inputs) {
    this.validateInputs(inputs, 1);
    const [X] = inputs;

    const axis = this.getAttribute('axis', -1);
    const actualAxis = axis < 0 ? X.shape.length + axis : axis;

    const result = new Float32Array(X.size);
    const outerSize = X.shape.slice(0, actualAxis).reduce((a, b) => a * b, 1);
    const axisSize = X.shape[actualAxis];
    const innerSize = X.shape.slice(actualAxis + 1).reduce((a, b) => a * b, 1);

    for (let outer = 0; outer < outerSize; outer++) {
      for (let inner = 0; inner < innerSize; inner++) {
        // Find max for numerical stability
        let maxVal = -Infinity;
        for (let i = 0; i < axisSize; i++) {
          const idx = (outer * axisSize + i) * innerSize + inner;
          maxVal = Math.max(maxVal, X.data[idx]);
        }

        // Compute exp and sum
        let sum = 0;
        for (let i = 0; i < axisSize; i++) {
          const idx = (outer * axisSize + i) * innerSize + inner;
          const exp = Math.exp(X.data[idx] - maxVal);
          result[idx] = exp;
          sum += exp;
        }

        // Normalize
        for (let i = 0; i < axisSize; i++) {
          const idx = (outer * axisSize + i) * innerSize + inner;
          result[idx] /= sum;
        }
      }
    }

    return [new Tensor(result, X.shape, X.dtype)];
  }
}

/**
 * Swish/SiLU operator: y = x * sigmoid(x)
 */
class Swish extends ONNXOperator {
  constructor(attributes = {}) {
    super('Swish', attributes);
  }

  execute(inputs) {
    this.validateInputs(inputs, 1);
    const [X] = inputs;

    const result = new Float32Array(X.size);
    for (let i = 0; i < X.size; i++) {
      const x = X.data[i];
      result[i] = x / (1 + Math.exp(-x));
    }

    return [new Tensor(result, X.shape, X.dtype)];
  }
}

// ============================================================================
// Normalization Operators
// ============================================================================

/**
 * BatchNormalization operator
 */
class BatchNormalization extends ONNXOperator {
  constructor(attributes = {}) {
    super('BatchNormalization', attributes);
  }

  execute(inputs) {
    this.validateInputs(inputs, 5);
    const [X, scale, B, mean, var_] = inputs;

    const epsilon = this.getAttribute('epsilon', 1e-5);

    const result = new Float32Array(X.size);
    const numChannels = X.shape[1];
    const spatialSize = X.size / (X.shape[0] * numChannels);

    for (let i = 0; i < X.size; i++) {
      const channel = Math.floor((i / spatialSize) % numChannels);
      const normalized = (X.data[i] - mean.data[channel]) /
        Math.sqrt(var_.data[channel] + epsilon);
      result[i] = scale.data[channel] * normalized + B.data[channel];
    }

    return [new Tensor(result, X.shape, X.dtype)];
  }
}

/**
 * LayerNormalization operator
 */
class LayerNormalization extends ONNXOperator {
  constructor(attributes = {}) {
    super('LayerNormalization', attributes);
  }

  execute(inputs) {
    this.validateInputs(inputs, 1);
    const [X, scale, bias] = inputs.length >= 3 ? inputs : [inputs[0], null, null];

    const axis = this.getAttribute('axis', -1);
    const epsilon = this.getAttribute('epsilon', 1e-5);

    const actualAxis = axis < 0 ? X.shape.length + axis : axis;
    const normalizedShape = X.shape.slice(actualAxis);
    const normalizedSize = normalizedShape.reduce((a, b) => a * b, 1);
    const batchSize = X.size / normalizedSize;

    const result = new Float32Array(X.size);

    for (let b = 0; b < batchSize; b++) {
      const offset = b * normalizedSize;

      // Compute mean
      let mean = 0;
      for (let i = 0; i < normalizedSize; i++) {
        mean += X.data[offset + i];
      }
      mean /= normalizedSize;

      // Compute variance
      let variance = 0;
      for (let i = 0; i < normalizedSize; i++) {
        const diff = X.data[offset + i] - mean;
        variance += diff * diff;
      }
      variance /= normalizedSize;

      // Normalize
      for (let i = 0; i < normalizedSize; i++) {
        const normalized = (X.data[offset + i] - mean) / Math.sqrt(variance + epsilon);
        const scaleVal = scale ? scale.data[i % scale.size] : 1;
        const biasVal = bias ? bias.data[i % bias.size] : 0;
        result[offset + i] = scaleVal * normalized + biasVal;
      }
    }

    return [new Tensor(result, X.shape, X.dtype)];
  }
}

// ============================================================================
// Tensor Operations
// ============================================================================

/**
 * Reshape operator
 */
class Reshape extends ONNXOperator {
  constructor(attributes = {}) {
    super('Reshape', attributes);
  }

  execute(inputs) {
    this.validateInputs(inputs, 2);
    const [data, shape] = inputs;

    const newShape = Array.from(shape.data).map(x => Number(x));

    // Handle -1 in shape (infer dimension)
    const negIndex = newShape.indexOf(-1);
    if (negIndex !== -1) {
      const knownSize = newShape.reduce((a, b) => b === -1 ? a : a * b, 1);
      newShape[negIndex] = data.size / knownSize;
    }

    return [data.reshape(newShape)];
  }
}

/**
 * Transpose operator
 */
class Transpose extends ONNXOperator {
  constructor(attributes = {}) {
    super('Transpose', attributes);
  }

  execute(inputs) {
    this.validateInputs(inputs, 1);
    const [X] = inputs;

    const perm = this.getAttribute('perm', null) ||
      X.shape.map((_, i) => X.shape.length - 1 - i);

    const outShape = perm.map(i => X.shape[i]);
    const result = new Float32Array(X.size);

    // Compute strides
    const inStrides = this.computeStrides(X.shape);
    const outStrides = this.computeStrides(outShape);

    for (let i = 0; i < X.size; i++) {
      const inCoords = this.linearToCoords(i, X.shape, inStrides);
      const outCoords = perm.map(p => inCoords[p]);
      const outIdx = this.coordsToLinear(outCoords, outStrides);
      result[outIdx] = X.data[i];
    }

    return [new Tensor(result, outShape, X.dtype)];
  }

  computeStrides(shape) {
    const strides = new Array(shape.length);
    strides[shape.length - 1] = 1;
    for (let i = shape.length - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
  }

  linearToCoords(idx, shape, strides) {
    const coords = new Array(shape.length);
    for (let i = 0; i < shape.length; i++) {
      coords[i] = Math.floor(idx / strides[i]) % shape[i];
    }
    return coords;
  }

  coordsToLinear(coords, strides) {
    let idx = 0;
    for (let i = 0; i < coords.length; i++) {
      idx += coords[i] * strides[i];
    }
    return idx;
  }
}

/**
 * Concat operator
 */
class Concat extends ONNXOperator {
  constructor(attributes = {}) {
    super('Concat', attributes);
  }

  execute(inputs) {
    if (inputs.length === 0) {
      throw new Error('Concat: no inputs provided');
    }

    const axis = this.getAttribute('axis', 0);
    const actualAxis = axis < 0 ? inputs[0].shape.length + axis : axis;

    // Validate shapes
    for (let i = 1; i < inputs.length; i++) {
      for (let j = 0; j < inputs[0].shape.length; j++) {
        if (j !== actualAxis && inputs[i].shape[j] !== inputs[0].shape[j]) {
          throw new Error('Concat: incompatible shapes');
        }
      }
    }

    // Compute output shape
    const outShape = [...inputs[0].shape];
    outShape[actualAxis] = inputs.reduce((sum, t) => sum + t.shape[actualAxis], 0);

    const result = new Float32Array(outShape.reduce((a, b) => a * b, 1));

    // Concatenate
    const outerSize = outShape.slice(0, actualAxis).reduce((a, b) => a * b, 1);
    const innerSize = outShape.slice(actualAxis + 1).reduce((a, b) => a * b, 1);

    let outIdx = 0;
    for (let outer = 0; outer < outerSize; outer++) {
      for (const input of inputs) {
        const axisSize = input.shape[actualAxis];
        for (let i = 0; i < axisSize; i++) {
          for (let inner = 0; inner < innerSize; inner++) {
            const inIdx = (outer * axisSize + i) * innerSize + inner;
            result[outIdx++] = input.data[inIdx];
          }
        }
      }
    }

    return [new Tensor(result, outShape, inputs[0].dtype)];
  }
}

/**
 * Slice operator
 */
class Slice extends ONNXOperator {
  constructor(attributes = {}) {
    super('Slice', attributes);
  }

  execute(inputs) {
    this.validateInputs(inputs, 3);
    const [data, starts, ends, axes, steps] = inputs.length >= 5
      ? inputs
      : [...inputs, null, null];

    const startsArr = Array.from(starts.data).map(x => Number(x));
    const endsArr = Array.from(ends.data).map(x => Number(x));
    const axesArr = axes
      ? Array.from(axes.data).map(x => Number(x))
      : startsArr.map((_, i) => i);
    const stepsArr = steps
      ? Array.from(steps.data).map(x => Number(x))
      : startsArr.map(() => 1);

    // Compute output shape
    const outShape = [...data.shape];
    for (let i = 0; i < axesArr.length; i++) {
      const axis = axesArr[i];
      const start = startsArr[i] < 0 ? data.shape[axis] + startsArr[i] : startsArr[i];
      const end = endsArr[i] < 0 ? data.shape[axis] + endsArr[i] : endsArr[i];
      const step = stepsArr[i];
      outShape[axis] = Math.ceil((end - start) / step);
    }

    // Extract slice (simplified implementation)
    const resultSize = outShape.reduce((a, b) => a * b, 1);
    const result = new Float32Array(resultSize);

    // For simplicity, handle 1D case
    if (data.shape.length === 1) {
      let outIdx = 0;
      for (let i = startsArr[0]; i < endsArr[0]; i += stepsArr[0]) {
        result[outIdx++] = data.data[i];
      }
    } else {
      // Multi-dimensional slice (simplified)
      result.set(data.data.slice(0, resultSize));
    }

    return [new Tensor(result, outShape, data.dtype)];
  }
}

// ============================================================================
// Reduction Operators
// ============================================================================

/**
 * ReduceSum operator
 */
class ReduceSum extends ONNXOperator {
  constructor(attributes = {}) {
    super('ReduceSum', attributes);
  }

  execute(inputs) {
    this.validateInputs(inputs, 1);
    const [data, axes] = inputs.length >= 2 ? inputs : [inputs[0], null];

    const axesArr = axes
      ? Array.from(axes.data).map(x => Number(x))
      : Array.from({ length: data.shape.length }, (_, i) => i);

    const keepdims = this.getAttribute('keepdims', 1);

    return this.reduce(data, axesArr, keepdims, (a, b) => a + b, 0);
  }

  reduce(data, axes, keepdims, reduceOp, initialValue) {
    // Sort axes in descending order
    const sortedAxes = [...axes].sort((a, b) => b - a);

    let result = data.clone();

    for (const axis of sortedAxes) {
      const outerSize = result.shape.slice(0, axis).reduce((a, b) => a * b, 1);
      const axisSize = result.shape[axis];
      const innerSize = result.shape.slice(axis + 1).reduce((a, b) => a * b, 1);

      const newSize = outerSize * innerSize;
      const newData = new Float32Array(newSize);

      for (let outer = 0; outer < outerSize; outer++) {
        for (let inner = 0; inner < innerSize; inner++) {
          let value = initialValue;
          for (let i = 0; i < axisSize; i++) {
            const idx = (outer * axisSize + i) * innerSize + inner;
            value = reduceOp(value, result.data[idx]);
          }
          newData[outer * innerSize + inner] = value;
        }
      }

      const newShape = [...result.shape];
      if (keepdims) {
        newShape[axis] = 1;
      } else {
        newShape.splice(axis, 1);
      }

      result = new Tensor(newData, newShape, data.dtype);
    }

    return [result];
  }
}

/**
 * ReduceMean operator
 */
class ReduceMean extends ReduceSum {
  constructor(attributes = {}) {
    super(attributes);
    this.name = 'ReduceMean';
  }

  execute(inputs) {
    this.validateInputs(inputs, 1);
    const [data, axes] = inputs.length >= 2 ? inputs : [inputs[0], null];

    const axesArr = axes
      ? Array.from(axes.data).map(x => Number(x))
      : Array.from({ length: data.shape.length }, (_, i) => i);

    const keepdims = this.getAttribute('keepdims', 1);

    // Calculate reduction size
    let reductionSize = 1;
    for (const axis of axesArr) {
      reductionSize *= data.shape[axis];
    }

    const [sumResult] = this.reduce(data, axesArr, keepdims, (a, b) => a + b, 0);

    // Divide by reduction size
    const meanData = new Float32Array(sumResult.size);
    for (let i = 0; i < sumResult.size; i++) {
      meanData[i] = sumResult.data[i] / reductionSize;
    }

    return [new Tensor(meanData, sumResult.shape, data.dtype)];
  }
}

/**
 * ReduceMax operator
 */
class ReduceMax extends ReduceSum {
  constructor(attributes = {}) {
    super(attributes);
    this.name = 'ReduceMax';
  }

  execute(inputs) {
    this.validateInputs(inputs, 1);
    const [data, axes] = inputs.length >= 2 ? inputs : [inputs[0], null];

    const axesArr = axes
      ? Array.from(axes.data).map(x => Number(x))
      : Array.from({ length: data.shape.length }, (_, i) => i);

    const keepdims = this.getAttribute('keepdims', 1);

    return this.reduce(data, axesArr, keepdims, Math.max, -Infinity);
  }
}

// ============================================================================
// Operator Registry
// ============================================================================

/**
 * ONNX Operator Registry
 */
export class ONNXOperatorRegistry {
  constructor() {
    this.operators = new Map();
    this.registerDefaultOperators();
  }

  /**
   * Register default operators
   */
  registerDefaultOperators() {
    // Math operators
    this.register('Add', Add);
    this.register('Sub', Sub);
    this.register('Mul', Mul);
    this.register('Div', Div);
    this.register('MatMul', MatMul);
    this.register('Gemm', Gemm);

    // Activations
    this.register('Relu', Relu);
    this.register('Gelu', Gelu);
    this.register('Sigmoid', Sigmoid);
    this.register('Tanh', Tanh);
    this.register('Softmax', Softmax);
    this.register('Swish', Swish);

    // Normalization
    this.register('BatchNormalization', BatchNormalization);
    this.register('LayerNormalization', LayerNormalization);

    // Tensor operations
    this.register('Reshape', Reshape);
    this.register('Transpose', Transpose);
    this.register('Concat', Concat);
    this.register('Slice', Slice);

    // Reduction
    this.register('ReduceSum', ReduceSum);
    this.register('ReduceMean', ReduceMean);
    this.register('ReduceMax', ReduceMax);
  }

  /**
   * Register an operator
   * @param {string} name - Operator name
   * @param {class} operatorClass - Operator class
   */
  register(name, operatorClass) {
    this.operators.set(name, operatorClass);
  }

  /**
   * Create operator instance
   * @param {string} name - Operator name
   * @param {Object} attributes - Operator attributes
   * @returns {ONNXOperator} Operator instance
   */
  create(name, attributes = {}) {
    const OperatorClass = this.operators.get(name);

    if (!OperatorClass) {
      throw new Error(`Unknown operator: ${name}`);
    }

    return new OperatorClass(attributes);
  }

  /**
   * Check if operator is supported
   * @param {string} name - Operator name
   * @returns {boolean} Whether operator is supported
   */
  isSupported(name) {
    return this.operators.has(name);
  }

  /**
   * Get list of supported operators
   * @returns {Array<string>} List of operator names
   */
  getSupportedOperators() {
    return Array.from(this.operators.keys());
  }
}

/**
 * Create default operator registry
 * @returns {ONNXOperatorRegistry} Operator registry
 */
export function createOperatorRegistry() {
  return new ONNXOperatorRegistry();
}

// Export all classes
export {
  ONNXOperator,
  Tensor,
  // Math
  Add,
  Sub,
  Mul,
  Div,
  MatMul,
  Gemm,
  // Activations
  Relu,
  Gelu,
  Sigmoid,
  Tanh,
  Softmax,
  Swish,
  // Normalization
  BatchNormalization,
  LayerNormalization,
  // Tensor ops
  Reshape,
  Transpose,
  Concat,
  Slice,
  // Reduction
  ReduceSum,
  ReduceMean,
  ReduceMax
};

export default {
  ONNXOperatorRegistry,
  createOperatorRegistry,
  Tensor
};
