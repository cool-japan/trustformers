/**
 * TrustformeRS Tensor Module
 * Tree-shakable tensor operations and utilities
 */

// Import from main module
import { tensor, zeros, ones, randn, eye, tensor_ops, activations } from '../index.js';

// Re-export tensor creation functions
export { tensor, zeros, ones, randn, eye };

// Re-export math operations from tensor_ops
export const add = tensor_ops.add;
export const subtract = tensor_ops.sub;
export const multiply = tensor_ops.mul;
export const divide = tensor_ops.div;
export const matmul = tensor_ops.matmul;
export const dot = tensor_ops.matmul; // Alias for matmul

// Scalar operations
export const addScalar = tensor_ops.addScalar;
export const mulScalar = tensor_ops.mulScalar;

// Shape operations
export const reshape = tensor_ops.reshape;
export const transpose = tensor_ops.transpose;
export const squeeze = tensor_ops.squeeze;
export const unsqueeze = tensor_ops.unsqueeze;

// Reduction operations
export const sum = tensor_ops.sum;
export const mean = tensor_ops.mean;

// Mathematical functions
export const exp = tensor_ops.exp;
export const log = tensor_ops.log;
export const sqrt = tensor_ops.sqrt;
export const pow = tensor_ops.pow;
export const abs = tensor_ops.abs;

// Normalization operations
export const layerNorm = tensor_ops.layerNorm;
export const batchNorm = tensor_ops.batchNorm;

// Activation functions from activations
export const relu = activations.relu;
export const leakyRelu = activations.leakyRelu;
export const gelu = activations.gelu;
export const swish = activations.swish;
export const sigmoid = activations.sigmoid;
export const tanh = activations.tanh;
export const softmax = activations.softmax;
export const logSoftmax = activations.logSoftmax;

// Tensor-specific utilities
export class TensorUtils {
  /**
   * Check if two tensors have the same shape
   */
  static sameShape(a, b) {
    if (a.shape.length !== b.shape.length) return false;
    return a.shape.every((dim, i) => dim === b.shape[i]);
  }

  /**
   * Check if tensor shapes are broadcastable
   */
  static broadcastable(shapeA, shapeB) {
    const maxLen = Math.max(shapeA.length, shapeB.length);
    const paddedA = [...Array(maxLen - shapeA.length).fill(1), ...shapeA];
    const paddedB = [...Array(maxLen - shapeB.length).fill(1), ...shapeB];

    return paddedA.every((dimA, i) => {
      const dimB = paddedB[i];
      return dimA === dimB || dimA === 1 || dimB === 1;
    });
  }

  /**
   * Calculate output shape after broadcasting
   */
  static broadcastShape(shapeA, shapeB) {
    const maxLen = Math.max(shapeA.length, shapeB.length);
    const paddedA = [...Array(maxLen - shapeA.length).fill(1), ...shapeA];
    const paddedB = [...Array(maxLen - shapeB.length).fill(1), ...shapeB];

    return paddedA.map((dimA, i) => Math.max(dimA, paddedB[i]));
  }

  /**
   * Calculate total number of elements
   */
  static numel(shape) {
    return shape.reduce((acc, dim) => acc * dim, 1);
  }

  /**
   * Validate tensor shape
   */
  static validateShape(shape) {
    if (!Array.isArray(shape)) {
      throw new Error('Shape must be an array');
    }
    if (shape.some(dim => !Number.isInteger(dim) || dim < 0)) {
      throw new Error('Shape dimensions must be non-negative integers');
    }
    return true;
  }

  /**
   * Calculate memory usage of a tensor
   */
  static memoryUsage(shape, dtype = 'f32') {
    const bytesPerElement = {
      f32: 4,
      f64: 8,
      i32: 4,
      i64: 8,
      u8: 1,
      i8: 1,
      bool: 1,
    };

    const bytes = bytesPerElement[dtype] || 4;
    return this.numel(shape) * bytes;
  }
}

// Tensor creation presets
export const TensorPresets = {
  /**
   * Create a tensor with Xavier/Glorot initialization
   */
  async xavier(shape, gain = 1.0) {
    const fanIn = shape[0];
    const fanOut = shape[1] || fanIn;
    const std = gain * Math.sqrt(2.0 / (fanIn + fanOut));
    const tensor = await randn(shape);
    return mulScalar(tensor, std);
  },

  /**
   * Create a tensor with Kaiming/He initialization
   */
  async kaiming(shape, mode = 'fan_in', nonlinearity = 'relu') {
    const fan = mode === 'fan_in' ? shape[0] : shape[1] || shape[0];
    const gain = nonlinearity === 'relu' ? Math.sqrt(2.0) : 1.0;
    const std = gain / Math.sqrt(fan);
    const tensor = await randn(shape);
    return mulScalar(tensor, std);
  },

  /**
   * Create a tensor with uniform initialization
   */
  async uniform(shape, low = 0, high = 1) {
    const tensor = await randn(shape);
    const scaled = mulScalar(tensor, high - low);
    return addScalar(scaled, low);
  },

  /**
   * Create a tensor with truncated normal initialization
   */
  async truncatedNormal(shape, mean = 0, std = 1, a = -2, b = 2) {
    // Simplified truncated normal (actual implementation would be more complex)
    const tensor = await randn(shape);
    const scaled = addScalar(mulScalar(tensor, std), mean);
    // Note: clamp would need to be implemented in tensor_ops for full functionality
    return scaled;
  },
};

export default {
  TensorUtils,
  TensorPresets,
};