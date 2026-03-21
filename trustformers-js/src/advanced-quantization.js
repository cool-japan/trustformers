/**
 * Advanced Quantization Module
 *
 * Provides comprehensive quantization support for model compression and acceleration.
 * Supports multiple quantization schemes:
 * - INT8 (8-bit integer)
 * - INT4 (4-bit integer)
 * - FP16 (16-bit floating point)
 * - Mixed precision
 *
 * Features:
 * - Dynamic and static quantization
 * - Per-tensor and per-channel quantization
 * - Symmetric and asymmetric quantization
 * - Calibration and tuning utilities
 * - GGML-compatible quantization formats
 */

/**
 * Quantization data types
 */
export const QuantizationType = {
  FP32: 'fp32',
  FP16: 'fp16',
  INT8: 'int8',
  INT4: 'int4',
  UINT8: 'uint8',
  UINT4: 'uint4',
  GGML_Q4_0: 'ggml_q4_0',
  GGML_Q4_1: 'ggml_q4_1',
  GGML_Q5_0: 'ggml_q5_0',
  GGML_Q5_1: 'ggml_q5_1',
  GGML_Q8_0: 'ggml_q8_0'
};

/**
 * Quantization schemes
 */
export const QuantizationScheme = {
  SYMMETRIC: 'symmetric',
  ASYMMETRIC: 'asymmetric',
  AFFINE: 'affine'
};

/**
 * Quantization granularity
 */
export const QuantizationGranularity = {
  PER_TENSOR: 'per_tensor',
  PER_CHANNEL: 'per_channel',
  PER_GROUP: 'per_group'
};

/**
 * Float16 Utilities
 */
export class Float16Utils {
  /**
   * Convert Float32 to Float16 (using Uint16 storage)
   */
  static float32ToFloat16(value) {
    const f32 = new Float32Array([value]);
    const u32 = new Uint32Array(f32.buffer);
    const bits = u32[0];

    // Extract sign, exponent, and mantissa
    const sign = (bits >> 31) & 0x1;
    const exponent = (bits >> 23) & 0xff;
    const mantissa = bits & 0x7fffff;

    // Handle special cases
    if (exponent === 0xff) {
      // Infinity or NaN
      return (sign << 15) | 0x7c00 | (mantissa ? 1 : 0);
    }

    if (exponent === 0) {
      // Zero or denormal
      return sign << 15;
    }

    // Compute fp16 exponent
    const fp16Exp = exponent - 127 + 15;

    // Handle overflow/underflow
    if (fp16Exp >= 31) {
      // Overflow to infinity
      return (sign << 15) | 0x7c00;
    }

    if (fp16Exp <= 0) {
      // Underflow to zero
      return sign << 15;
    }

    // Compute fp16 mantissa (round to nearest)
    const fp16Mantissa = (mantissa + 0x1000) >> 13;

    return (sign << 15) | (fp16Exp << 10) | (fp16Mantissa & 0x3ff);
  }

  /**
   * Convert Float16 (Uint16) to Float32
   */
  static float16ToFloat32(value) {
    const sign = (value >> 15) & 0x1;
    const exponent = (value >> 10) & 0x1f;
    const mantissa = value & 0x3ff;

    // Handle special cases
    if (exponent === 0) {
      // Zero or denormal
      if (mantissa === 0) {
        return sign ? -0.0 : 0.0;
      }
      // Denormal numbers
      const denormal = mantissa / 1024.0 * Math.pow(2, -14);
      return sign ? -denormal : denormal;
    }

    if (exponent === 0x1f) {
      // Infinity or NaN
      return mantissa ? NaN : (sign ? -Infinity : Infinity);
    }

    // Normal numbers
    const f32Exp = exponent - 15 + 127;
    const f32Mantissa = mantissa << 13;
    const bits = (sign << 31) | (f32Exp << 23) | f32Mantissa;

    const u32 = new Uint32Array([bits]);
    const f32 = new Float32Array(u32.buffer);
    return f32[0];
  }

  /**
   * Convert Float32Array to Float16 (stored as Uint16Array)
   */
  static float32ArrayToFloat16Array(array) {
    const result = new Uint16Array(array.length);
    for (let i = 0; i < array.length; i++) {
      result[i] = this.float32ToFloat16(array[i]);
    }
    return result;
  }

  /**
   * Convert Float16 (Uint16Array) to Float32Array
   */
  static float16ArrayToFloat32Array(array) {
    const result = new Float32Array(array.length);
    for (let i = 0; i < array.length; i++) {
      result[i] = this.float16ToFloat32(array[i]);
    }
    return result;
  }
}

/**
 * INT8 Quantization
 */
export class Int8Quantizer {
  /**
   * Calibrate quantization parameters from data
   */
  static calibrate(data, scheme = QuantizationScheme.SYMMETRIC) {
    const min = Math.min(...data);
    const max = Math.max(...data);

    if (scheme === QuantizationScheme.SYMMETRIC) {
      const absMax = Math.max(Math.abs(min), Math.abs(max));
      const scale = absMax / 127.0;
      return { scale, zeroPoint: 0, min, max };
    } 
      // Asymmetric quantization
      const scale = (max - min) / 255.0;
      const zeroPoint = Math.round(-min / scale);
      return { scale, zeroPoint, min, max };
    
  }

  /**
   * Quantize Float32Array to Int8Array
   */
  static quantize(data, params = null) {
    const { scale, zeroPoint } = params || this.calibrate(data);
    const result = new Int8Array(data.length);

    for (let i = 0; i < data.length; i++) {
      const quantized = Math.round(data[i] / scale) + zeroPoint;
      result[i] = Math.max(-128, Math.min(127, quantized));
    }

    return { data: result, scale, zeroPoint };
  }

  /**
   * Dequantize Int8Array to Float32Array
   */
  static dequantize(data, scale, zeroPoint = 0) {
    const result = new Float32Array(data.length);

    for (let i = 0; i < data.length; i++) {
      result[i] = (data[i] - zeroPoint) * scale;
    }

    return result;
  }

  /**
   * Per-channel quantization
   */
  static quantizePerChannel(data, shape, axis = 0) {
    const numChannels = shape[axis];
    const channelSize = data.length / numChannels;

    const quantizedData = new Int8Array(data.length);
    const scales = new Float32Array(numChannels);
    const zeroPoints = new Int8Array(numChannels);

    for (let ch = 0; ch < numChannels; ch++) {
      const start = ch * channelSize;
      const end = start + channelSize;
      const channelData = data.slice(start, end);

      const params = this.calibrate(channelData);
      scales[ch] = params.scale;
      zeroPoints[ch] = params.zeroPoint;

      for (let i = 0; i < channelSize; i++) {
        const idx = start + i;
        const quantized = Math.round(data[idx] / params.scale) + params.zeroPoint;
        quantizedData[idx] = Math.max(-128, Math.min(127, quantized));
      }
    }

    return { data: quantizedData, scales, zeroPoints, numChannels };
  }

  /**
   * Dequantize per-channel data
   */
  static dequantizePerChannel(data, scales, zeroPoints, shape, axis = 0) {
    const numChannels = shape[axis];
    const channelSize = data.length / numChannels;
    const result = new Float32Array(data.length);

    for (let ch = 0; ch < numChannels; ch++) {
      const start = ch * channelSize;
      const scale = scales[ch];
      const zeroPoint = zeroPoints[ch];

      for (let i = 0; i < channelSize; i++) {
        const idx = start + i;
        result[idx] = (data[idx] - zeroPoint) * scale;
      }
    }

    return result;
  }
}

/**
 * INT4 Quantization (4-bit)
 */
export class Int4Quantizer {
  /**
   * Quantize Float32Array to 4-bit (packed in Uint8Array)
   */
  static quantize(data, params = null) {
    const { scale, zeroPoint } = params || Int8Quantizer.calibrate(data);

    // Pack two 4-bit values into each byte
    const packedLength = Math.ceil(data.length / 2);
    const result = new Uint8Array(packedLength);

    for (let i = 0; i < data.length; i++) {
      const quantized = Math.round(data[i] / scale) + zeroPoint;
      const clamped = Math.max(-8, Math.min(7, quantized)) + 8; // Map to [0, 15]

      const byteIdx = Math.floor(i / 2);
      if (i % 2 === 0) {
        result[byteIdx] = clamped & 0x0f;
      } else {
        result[byteIdx] |= (clamped << 4);
      }
    }

    return { data: result, scale, zeroPoint, originalLength: data.length };
  }

  /**
   * Dequantize 4-bit data to Float32Array
   */
  static dequantize(packedData, scale, zeroPoint = 8, originalLength) {
    const result = new Float32Array(originalLength);

    for (let i = 0; i < originalLength; i++) {
      const byteIdx = Math.floor(i / 2);
      const nibble = (i % 2 === 0) ?
        (packedData[byteIdx] & 0x0f) :
        ((packedData[byteIdx] >> 4) & 0x0f);

      const value = nibble - 8; // Map back to [-8, 7]
      result[i] = (value - zeroPoint) * scale;
    }

    return result;
  }
}

/**
 * GGML-Compatible Quantization
 * Supports GGML quantization formats used in llama.cpp
 */
export class GGMLQuantizer {
  /**
   * GGML Q4_0: 4-bit quantization with block-wise scaling
   * Block size: 32 values
   */
  static quantizeQ4_0(data) {
    const blockSize = 32;
    const numBlocks = Math.ceil(data.length / blockSize);
    const quantizedSize = numBlocks * (2 + blockSize / 2); // 2 bytes scale + 16 bytes data

    const result = new Uint8Array(quantizedSize);
    let offset = 0;

    for (let block = 0; block < numBlocks; block++) {
      const start = block * blockSize;
      const end = Math.min(start + blockSize, data.length);
      const blockData = data.slice(start, end);

      // Compute block scale
      const absMax = Math.max(...blockData.map(Math.abs));
      const scale = absMax / 8.0;

      // Store scale as float16
      const scaleF16 = Float16Utils.float32ToFloat16(scale);
      result[offset++] = scaleF16 & 0xff;
      result[offset++] = (scaleF16 >> 8) & 0xff;

      // Quantize and pack values
      for (let i = 0; i < blockData.length; i += 2) {
        const q1 = Math.round(blockData[i] / scale);
        const q2 = i + 1 < blockData.length ? Math.round(blockData[i + 1] / scale) : 0;

        const clamped1 = Math.max(-8, Math.min(7, q1)) + 8;
        const clamped2 = Math.max(-8, Math.min(7, q2)) + 8;

        result[offset++] = (clamped1 & 0x0f) | ((clamped2 & 0x0f) << 4);
      }
    }

    return { data: result, blockSize, numBlocks, type: QuantizationType.GGML_Q4_0 };
  }

  /**
   * GGML Q8_0: 8-bit quantization with block-wise scaling
   * Block size: 32 values
   */
  static quantizeQ8_0(data) {
    const blockSize = 32;
    const numBlocks = Math.ceil(data.length / blockSize);
    const quantizedSize = numBlocks * (2 + blockSize); // 2 bytes scale + 32 bytes data

    const result = new Uint8Array(quantizedSize);
    let offset = 0;

    for (let block = 0; block < numBlocks; block++) {
      const start = block * blockSize;
      const end = Math.min(start + blockSize, data.length);
      const blockData = data.slice(start, end);

      // Compute block scale
      const absMax = Math.max(...blockData.map(Math.abs));
      const scale = absMax / 127.0;

      // Store scale as float16
      const scaleF16 = Float16Utils.float32ToFloat16(scale);
      result[offset++] = scaleF16 & 0xff;
      result[offset++] = (scaleF16 >> 8) & 0xff;

      // Quantize values
      for (let i = 0; i < blockData.length; i++) {
        const quantized = Math.round(blockData[i] / scale);
        const clamped = Math.max(-128, Math.min(127, quantized));
        result[offset++] = clamped & 0xff;
      }

      // Pad if needed
      for (let i = blockData.length; i < blockSize; i++) {
        result[offset++] = 0;
      }
    }

    return { data: result, blockSize, numBlocks, type: QuantizationType.GGML_Q8_0 };
  }

  /**
   * Dequantize GGML Q4_0 format
   */
  static dequantizeQ4_0(quantized, originalLength) {
    const blockSize = 32;
    const result = new Float32Array(originalLength);
    let offset = 0;
    let outIdx = 0;

    while (outIdx < originalLength) {
      // Read scale (float16)
      const scaleF16 = quantized[offset++] | (quantized[offset++] << 8);
      const scale = Float16Utils.float16ToFloat32(scaleF16);

      // Dequantize block
      for (let i = 0; i < blockSize / 2 && outIdx < originalLength; i++) {
        const packed = quantized[offset++];
        const q1 = (packed & 0x0f) - 8;
        const q2 = ((packed >> 4) & 0x0f) - 8;

        result[outIdx++] = q1 * scale;
        if (outIdx < originalLength) {
          result[outIdx++] = q2 * scale;
        }
      }
    }

    return result;
  }

  /**
   * Dequantize GGML Q8_0 format
   */
  static dequantizeQ8_0(quantized, originalLength) {
    const blockSize = 32;
    const result = new Float32Array(originalLength);
    let offset = 0;
    let outIdx = 0;

    while (outIdx < originalLength) {
      // Read scale (float16)
      const scaleF16 = quantized[offset++] | (quantized[offset++] << 8);
      const scale = Float16Utils.float16ToFloat32(scaleF16);

      // Dequantize block
      const remaining = Math.min(blockSize, originalLength - outIdx);
      for (let i = 0; i < remaining; i++) {
        const quantized_val = quantized[offset++];
        const signed = quantized_val > 127 ? quantized_val - 256 : quantized_val;
        result[outIdx++] = signed * scale;
      }

      // Skip padding
      offset += blockSize - remaining;
    }

    return result;
  }
}

/**
 * Quantization Calibrator
 * Determines optimal quantization parameters
 */
export class QuantizationCalibrator {
  constructor() {
    this.statistics = new Map();
  }

  /**
   * Collect statistics from data
   */
  collectStatistics(name, data) {
    const min = Math.min(...data);
    const max = Math.max(...data);
    const absMax = Math.max(Math.abs(min), Math.abs(max));

    // Compute histogram
    const bins = 256;
    const histogram = new Uint32Array(bins);
    const binWidth = (max - min) / bins;

    for (const value of data) {
      const bin = Math.min(bins - 1, Math.floor((value - min) / binWidth));
      histogram[bin]++;
    }

    this.statistics.set(name, {
      min, max, absMax,
      mean: data.reduce((a, b) => a + b, 0) / data.length,
      histogram,
      binWidth
    });
  }

  /**
   * Compute optimal scale using KL divergence minimization
   */
  computeOptimalScale(name, targetBits = 8) {
    const stats = this.statistics.get(name);
    if (!stats) {
      throw new Error(`No statistics collected for ${name}`);
    }

    const maxValue = targetBits === 8 ? 127 : 7;
    let bestScale = stats.absMax / maxValue;
    let bestDivergence = Infinity;

    // Try different scales
    for (let i = 0.8; i <= 1.2; i += 0.05) {
      const scale = (stats.absMax / maxValue) * i;
      const divergence = this._computeKLDivergence(stats, scale, maxValue);

      if (divergence < bestDivergence) {
        bestDivergence = divergence;
        bestScale = scale;
      }
    }

    return bestScale;
  }

  _computeKLDivergence(stats, scale, maxValue) {
    // Simplified KL divergence computation
    // In practice, this would be more sophisticated
    let divergence = 0;

    for (let i = 0; i < stats.histogram.length; i++) {
      if (stats.histogram[i] === 0) continue;

      const value = stats.min + i * stats.binWidth;
      const quantized = Math.round(value / scale);
      const clamped = Math.max(-maxValue, Math.min(maxValue, quantized));
      const dequantized = clamped * scale;

      const error = Math.abs(value - dequantized);
      divergence += stats.histogram[i] * error;
    }

    return divergence;
  }
}

/**
 * Mixed Precision Quantization
 * Different layers can use different quantization schemes
 */
export class MixedPrecisionQuantizer {
  constructor() {
    this.layerConfig = new Map();
    this.defaultType = QuantizationType.INT8;
  }

  /**
   * Set quantization type for a specific layer
   */
  setLayerQuantization(layerName, quantizationType) {
    this.layerConfig.set(layerName, quantizationType);
  }

  /**
   * Quantize a layer based on its configuration
   */
  quantizeLayer(layerName, data, _shape = null) {
    const quantType = this.layerConfig.get(layerName) || this.defaultType;

    switch (quantType) {
      case QuantizationType.FP16:
        return {
          data: Float16Utils.float32ArrayToFloat16Array(data),
          type: QuantizationType.FP16
        };

      case QuantizationType.INT8:
        return {
          ...Int8Quantizer.quantize(data),
          type: QuantizationType.INT8
        };

      case QuantizationType.INT4:
        return {
          ...Int4Quantizer.quantize(data),
          type: QuantizationType.INT4
        };

      case QuantizationType.GGML_Q4_0:
        return GGMLQuantizer.quantizeQ4_0(data);

      case QuantizationType.GGML_Q8_0:
        return GGMLQuantizer.quantizeQ8_0(data);

      default:
        return { data, type: QuantizationType.FP32 };
    }
  }

  /**
   * Dequantize a layer
   */
  dequantizeLayer(quantized) {
    switch (quantized.type) {
      case QuantizationType.FP16:
        return Float16Utils.float16ArrayToFloat32Array(quantized.data);

      case QuantizationType.INT8:
        return Int8Quantizer.dequantize(quantized.data, quantized.scale, quantized.zeroPoint);

      case QuantizationType.INT4:
        return Int4Quantizer.dequantize(
          quantized.data,
          quantized.scale,
          quantized.zeroPoint,
          quantized.originalLength
        );

      case QuantizationType.GGML_Q4_0:
        return GGMLQuantizer.dequantizeQ4_0(quantized.data, quantized.originalLength);

      case QuantizationType.GGML_Q8_0:
        return GGMLQuantizer.dequantizeQ8_0(quantized.data, quantized.originalLength);

      default:
        return quantized.data;
    }
  }

  /**
   * Generate quantization report
   */
  generateReport() {
    const report = {
      layers: {},
      totalOriginalSize: 0,
      totalQuantizedSize: 0,
      compressionRatio: 0
    };

    for (const [layer, type] of this.layerConfig.entries()) {
      report.layers[layer] = {
        type,
        bitsPerWeight: this._getBitsPerWeight(type)
      };
    }

    return report;
  }

  _getBitsPerWeight(type) {
    switch (type) {
      case QuantizationType.FP32: return 32;
      case QuantizationType.FP16: return 16;
      case QuantizationType.INT8: return 8;
      case QuantizationType.INT4: return 4;
      case QuantizationType.GGML_Q4_0: return 4.5;
      case QuantizationType.GGML_Q8_0: return 8.5;
      default: return 32;
    }
  }
}

/**
 * Quantization-Aware Training Support
 */
export class QuantizationAwareTraining {
  /**
   * Simulate quantization during forward pass
   */
  static fakeQuantize(data, scale, zeroPoint, numBits = 8) {
    const maxValue = Math.pow(2, numBits - 1) - 1;
    const minValue = -maxValue - 1;

    const result = new Float32Array(data.length);

    for (let i = 0; i < data.length; i++) {
      const quantized = Math.round(data[i] / scale) + zeroPoint;
      const clamped = Math.max(minValue, Math.min(maxValue, quantized));
      result[i] = (clamped - zeroPoint) * scale;
    }

    return result;
  }
}

// Export all classes and utilities
export default {
  QuantizationType,
  QuantizationScheme,
  QuantizationGranularity,
  Float16Utils,
  Int8Quantizer,
  Int4Quantizer,
  GGMLQuantizer,
  QuantizationCalibrator,
  MixedPrecisionQuantizer,
  QuantizationAwareTraining
};
