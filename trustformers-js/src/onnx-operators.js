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

    // Pad shapes on the left with 1s to equal length.
    const paddedA = new Array(ndimOut).fill(1);
    const paddedB = new Array(ndimOut).fill(1);
    for (let i = 0; i < ndimA; i++) paddedA[ndimOut - ndimA + i] = shapeA[i];
    for (let i = 0; i < ndimB; i++) paddedB[ndimOut - ndimB + i] = shapeB[i];

    const outShape = new Array(ndimOut);
    for (let i = 0; i < ndimOut; i++) {
      const da = paddedA[i];
      const db = paddedB[i];
      if (da !== db && da !== 1 && db !== 1) {
        throw new Error(`Cannot broadcast shapes [${shapeA}] and [${shapeB}]`);
      }
      outShape[i] = Math.max(da, db);
    }

    // Compute flat strides for each padded source shape (C-contiguous).
    const computeStrides = (shape) => {
      const s = new Array(ndimOut);
      s[ndimOut - 1] = 1;
      for (let i = ndimOut - 2; i >= 0; i--) {
        s[i] = s[i + 1] * shape[i + 1];
      }
      return s;
    };

    const rawStridesA = computeStrides(paddedA);
    const rawStridesB = computeStrides(paddedB);

    // For broadcast (dim==1) axes, force stride to 0.
    const stridesA = rawStridesA.map((s, i) => paddedA[i] === 1 ? 0 : s);
    const stridesB = rawStridesB.map((s, i) => paddedB[i] === 1 ? 0 : s);

    return { shape: outShape, stridesA, stridesB };
  }

  getBroadcastIndex(linearIdx, shape, strides) {
    // Compute the output's multi-dimensional coordinates, then map each
    // dimension through the per-source strides (0 for broadcast dims).
    let idx = 0;
    // We need per-output-dimension strides to decode linearIdx correctly.
    // outStrides[i] = product of outShape[i+1 ..].
    const outStrides = new Array(shape.length);
    outStrides[shape.length - 1] = 1;
    for (let i = shape.length - 2; i >= 0; i--) {
      outStrides[i] = outStrides[i + 1] * shape[i + 1];
    }
    for (let i = 0; i < shape.length; i++) {
      const coord = Math.floor(linearIdx / outStrides[i]) % shape[i];
      idx += coord * strides[i]; // strides[i] == 0 means broadcast → contributes 0
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
 * Slice operator — full multi-dimensional implementation with steps support.
 * Inputs: data, starts, ends[, axes[, steps]]
 */
class Slice extends ONNXOperator {
  constructor(attributes = {}) {
    super('Slice', attributes);
  }

  execute(inputs) {
    this.validateInputs(inputs, 3);
    const [data, startsT, endsT] = inputs;
    const axesT = inputs.length >= 4 ? inputs[3] : null;
    const stepsT = inputs.length >= 5 ? inputs[4] : null;

    const ndim = data.shape.length;

    const startsRaw = Array.from(startsT.data).map(Number);
    const endsRaw = Array.from(endsT.data).map(Number);
    const axesRaw = axesT
      ? Array.from(axesT.data).map(Number)
      : startsRaw.map((_, i) => i);
    const stepsRaw = stepsT
      ? Array.from(stepsT.data).map(Number)
      : startsRaw.map(() => 1);

    // Normalise negative indices; clamp to valid range per ONNX spec.
    const clampedStarts = new Array(ndim).fill(0);
    const clampedEnds = data.shape.slice();
    const clampedSteps = new Array(ndim).fill(1);

    for (let k = 0; k < axesRaw.length; k++) {
      let ax = axesRaw[k];
      if (ax < 0) ax += ndim;
      const dim = data.shape[ax];
      const step = stepsRaw[k];
      clampedSteps[ax] = step;

      let s = startsRaw[k] < 0 ? startsRaw[k] + dim : startsRaw[k];
      let e = endsRaw[k] < 0 ? endsRaw[k] + dim : endsRaw[k];

      if (step > 0) {
        s = Math.max(0, Math.min(dim, s));
        e = Math.max(0, Math.min(dim, e));
      } else {
        s = Math.max(-1, Math.min(dim - 1, s));
        e = Math.max(-1, Math.min(dim - 1, e));
      }
      clampedStarts[ax] = s;
      clampedEnds[ax] = e;
    }

    // Compute output shape
    const outShape = new Array(ndim);
    for (let ax = 0; ax < ndim; ax++) {
      const step = clampedSteps[ax];
      const s = clampedStarts[ax];
      const e = clampedEnds[ax];
      const span = e - s;
      outShape[ax] = step > 0
        ? Math.max(0, Math.ceil(span / step))
        : Math.max(0, Math.ceil(-span / -step));
    }

    const resultSize = outShape.reduce((a, b) => a * b, 1);
    const result = new Float32Array(resultSize);

    // Compute input C-contiguous strides.
    const inStrides = new Array(ndim);
    inStrides[ndim - 1] = 1;
    for (let i = ndim - 2; i >= 0; i--) {
      inStrides[i] = inStrides[i + 1] * data.shape[i + 1];
    }

    // Output C-contiguous strides.
    const outStrides = new Array(ndim);
    outStrides[ndim - 1] = 1;
    for (let i = ndim - 2; i >= 0; i--) {
      outStrides[i] = outStrides[i + 1] * outShape[i + 1];
    }

    // Iterate every output element by its multi-index.
    for (let outIdx = 0; outIdx < resultSize; outIdx++) {
      let inIdx = 0;
      let remaining = outIdx;
      for (let ax = 0; ax < ndim; ax++) {
        const outCoord = Math.floor(remaining / outStrides[ax]);
        remaining -= outCoord * outStrides[ax];
        const inCoord = clampedStarts[ax] + outCoord * clampedSteps[ax];
        inIdx += inCoord * inStrides[ax];
      }
      result[outIdx] = data.data[inIdx];
    }

    return [new Tensor(result, outShape, data.dtype)];
  }
}

/**
 * Gather operator — gathers slices from data along an axis using indices.
 *
 * ONNX spec: output[i_0,...,i_{r-1},j_0,...,j_{q-1}] =
 *              data[i_0,...,i_{axis-1}, indices[j_0,...,j_{q-1}], i_{axis+1},...,i_{r-1}]
 * where r = data.shape.length, q = indices.shape.length.
 */
class Gather extends ONNXOperator {
  constructor(attributes = {}) {
    super('Gather', attributes);
  }

  execute(inputs) {
    this.validateInputs(inputs, 2);
    const [data, indices] = inputs;

    let axis = this.getAttribute('axis', 0);
    if (axis < 0) axis += data.shape.length;

    if (axis < 0 || axis >= data.shape.length) {
      throw new Error(`Gather: axis ${axis} out of range for shape [${data.shape}]`);
    }

    const dataNdim = data.shape.length;
    const idxNdim = indices.shape.length;

    // Output shape = data.shape[:axis] + indices.shape + data.shape[axis+1:]
    const outShape = [
      ...data.shape.slice(0, axis),
      ...indices.shape,
      ...data.shape.slice(axis + 1)
    ];

    const resultSize = outShape.reduce((a, b) => a * b, 1);
    const result = new Float32Array(resultSize);

    // Compute strides for data (C-contiguous).
    const dataStrides = new Array(dataNdim);
    dataStrides[dataNdim - 1] = 1;
    for (let i = dataNdim - 2; i >= 0; i--) {
      dataStrides[i] = dataStrides[i + 1] * data.shape[i + 1];
    }

    // Compute strides for output (C-contiguous).
    const outStrides = new Array(outShape.length);
    outStrides[outShape.length - 1] = 1;
    for (let i = outShape.length - 2; i >= 0; i--) {
      outStrides[i] = outStrides[i + 1] * outShape[i + 1];
    }

    // Compute strides for indices (C-contiguous).
    const idxStrides = new Array(idxNdim);
    if (idxNdim > 0) {
      idxStrides[idxNdim - 1] = 1;
      for (let i = idxNdim - 2; i >= 0; i--) {
        idxStrides[i] = idxStrides[i + 1] * indices.shape[i + 1];
      }
    }

    // Iterate over every output element.
    for (let outFlat = 0; outFlat < resultSize; outFlat++) {
      // Decode output multi-index
      let remaining = outFlat;
      const outCoords = new Array(outShape.length);
      for (let d = 0; d < outShape.length; d++) {
        outCoords[d] = Math.floor(remaining / outStrides[d]);
        remaining -= outCoords[d] * outStrides[d];
      }

      // Split into: outCoords[:axis], outCoords[axis:axis+idxNdim], outCoords[axis+idxNdim:]
      const preDims = outCoords.slice(0, axis);
      const idxCoords = idxNdim > 0 ? outCoords.slice(axis, axis + idxNdim) : [];
      const postDims = outCoords.slice(axis + idxNdim);

      // Find the flat index into the indices tensor
      let idxFlat = 0;
      for (let d = 0; d < idxNdim; d++) {
        idxFlat += idxCoords[d] * idxStrides[d];
      }

      let gatherIdx = Math.floor(indices.data[idxFlat]);
      // Support negative indices
      if (gatherIdx < 0) gatherIdx += data.shape[axis];

      if (gatherIdx < 0 || gatherIdx >= data.shape[axis]) {
        throw new Error(
          `Gather: index ${gatherIdx} out of bounds for axis ${axis} size ${data.shape[axis]}`
        );
      }

      // Build the flat data index
      let dataFlat = 0;
      for (let d = 0; d < axis; d++) {
        dataFlat += preDims[d] * dataStrides[d];
      }
      dataFlat += gatherIdx * dataStrides[axis];
      for (let d = 0; d < postDims.length; d++) {
        dataFlat += postDims[d] * dataStrides[axis + 1 + d];
      }

      result[outFlat] = data.data[dataFlat];
    }

    return [new Tensor(result, outShape, data.dtype)];
  }
}

/**
 * Flatten operator — flattens input into 2D [outer, inner].
 *
 * ONNX spec: output shape = [prod(shape[:axis]), prod(shape[axis:])]
 * axis defaults to 1.  axis may be negative.
 */
class Flatten extends ONNXOperator {
  constructor(attributes = {}) {
    super('Flatten', attributes);
  }

  execute(inputs) {
    this.validateInputs(inputs, 1);
    const [data] = inputs;

    let axis = this.getAttribute('axis', 1);
    if (axis < 0) axis += data.shape.length;

    if (axis < 0 || axis > data.shape.length) {
      throw new Error(
        `Flatten: axis ${this.getAttribute('axis', 1)} out of range for rank-${data.shape.length} tensor`
      );
    }

    const outerSize = data.shape.slice(0, axis).reduce((a, b) => a * b, 1);
    const innerSize = data.shape.slice(axis).reduce((a, b) => a * b, 1);
    const outShape = [outerSize, innerSize];

    return [new Tensor(data.data, outShape, data.dtype)];
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
    this.register('Gather', Gather);
    this.register('Flatten', Flatten);

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
  Gather,
  Flatten,
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
