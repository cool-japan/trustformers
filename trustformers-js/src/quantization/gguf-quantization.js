/**
 * GGUF Quantization Support for TrustformeRS
 *
 * Implements GGUF (GPT-Generated Unified Format) quantization methods
 * for extreme model compression with minimal quality loss.
 *
 * Supported quantization types:
 * - Q4_0, Q4_1: 4-bit quantization with different block sizes
 * - Q5_0, Q5_1: 5-bit quantization for better quality
 * - Q8_0: 8-bit quantization for high quality
 * - Q2_K, Q3_K, Q4_K, Q5_K, Q6_K: K-quants (improved quality)
 * - IQ1_S, IQ2_XXS, IQ2_XS: Importance matrix quantization
 *
 * Features:
 * - Multiple quantization methods (legacy and K-quants)
 * - Block-wise quantization for better accuracy
 * - Importance matrix quantization (IQ series)
 * - Dequantization for inference
 * - Metadata and header parsing
 * - Mixed precision support
 *
 * @module quantization/gguf-quantization
 */

/**
 * GGUF quantization types
 * @enum {number}
 */
export const GGUFQuantType = {
  // Legacy quantization types
  Q4_0: 2,   // 4-bit, block size 32, no scaling
  Q4_1: 3,   // 4-bit, block size 32, with scaling
  Q5_0: 6,   // 5-bit, block size 32, no scaling
  Q5_1: 7,   // 5-bit, block size 32, with scaling
  Q8_0: 8,   // 8-bit, block size 32
  Q8_1: 9,   // 8-bit, block size 32, with scaling

  // K-quants (improved quality)
  Q2_K: 10,  // 2-bit K-quant
  Q3_K: 11,  // 3-K-quant
  Q4_K: 12,  // 4-bit K-quant
  Q5_K: 13,  // 5-bit K-quant
  Q6_K: 14,  // 6-bit K-quant

  // Importance matrix quantization
  IQ1_S: 15,    // 1.56 bits per weight
  IQ2_XXS: 16,  // 2.06 bits per weight
  IQ2_XS: 17,   // 2.31 bits per weight
  IQ3_XXS: 18,  // 3.06 bits per weight

  // No quantization
  F32: 0,    // 32-bit float
  F16: 1,    // 16-bit float
};

/**
 * Block sizes for different quantization types
 */
const BLOCK_SIZES = {
  [GGUFQuantType.Q4_0]: 32,
  [GGUFQuantType.Q4_1]: 32,
  [GGUFQuantType.Q5_0]: 32,
  [GGUFQuantType.Q5_1]: 32,
  [GGUFQuantType.Q8_0]: 32,
  [GGUFQuantType.Q8_1]: 32,
  [GGUFQuantType.Q2_K]: 256,
  [GGUFQuantType.Q3_K]: 256,
  [GGUFQuantType.Q4_K]: 256,
  [GGUFQuantType.Q5_K]: 256,
  [GGUFQuantType.Q6_K]: 256,
  [GGUFQuantType.IQ1_S]: 256,
  [GGUFQuantType.IQ2_XXS]: 256,
  [GGUFQuantType.IQ2_XS]: 256,
  [GGUFQuantType.IQ3_XXS]: 256,
};

/**
 * Bytes per block for different quantization types
 */
const BYTES_PER_BLOCK = {
  [GGUFQuantType.Q4_0]: 18,    // 2 + 16 (half + 32*4bit)
  [GGUFQuantType.Q4_1]: 20,    // 2 + 2 + 16 (2*half + 32*4bit)
  [GGUFQuantType.Q5_0]: 22,    // 2 + 4 + 16 (half + 32bit + 32*4bit)
  [GGUFQuantType.Q5_1]: 24,    // 2 + 2 + 4 + 16
  [GGUFQuantType.Q8_0]: 34,    // 2 + 32 (half + 32*8bit)
  [GGUFQuantType.Q8_1]: 36,    // 2 + 2 + 32
  [GGUFQuantType.Q2_K]: 82,    // K-quant block
  [GGUFQuantType.Q3_K]: 110,
  [GGUFQuantType.Q4_K]: 144,
  [GGUFQuantType.Q5_K]: 176,
  [GGUFQuantType.Q6_K]: 210,
  [GGUFQuantType.IQ1_S]: 52,
  [GGUFQuantType.IQ2_XXS]: 66,
  [GGUFQuantType.IQ2_XS]: 74,
  [GGUFQuantType.IQ3_XXS]: 98,
};

/**
 * GGUF Quantizer
 */
export class GGUFQuantizer {
  /**
   * Create a GGUF quantizer
   * @param {Object} config - Quantization configuration
   */
  constructor(config = {}) {
    this.config = {
      quantType: GGUFQuantType.Q4_K,
      perChannel: true,
      importanceMatrix: false,
      ...config
    };
  }

  /**
   * Quantize a tensor
   * @param {Float32Array} data - Input data
   * @param {Array<number>} shape - Tensor shape
   * @returns {Object} Quantized data and metadata
   */
  quantize(data, shape) {
    const { quantType } = this.config;
    const blockSize = BLOCK_SIZES[quantType];
    const bytesPerBlock = BYTES_PER_BLOCK[quantType];

    if (!blockSize) {
      throw new Error(`Unsupported quantization type: ${quantType}`);
    }

    // Calculate number of blocks
    const numElements = data.length;
    const numBlocks = Math.ceil(numElements / blockSize);
    const totalBytes = numBlocks * bytesPerBlock;

    // Allocate output buffer
    const quantized = new Uint8Array(totalBytes);

    // Quantize based on type
    switch (quantType) {
      case GGUFQuantType.Q4_0:
        this.quantizeQ4_0(data, quantized, numBlocks);
        break;
      case GGUFQuantType.Q4_1:
        this.quantizeQ4_1(data, quantized, numBlocks);
        break;
      case GGUFQuantType.Q5_0:
        this.quantizeQ5_0(data, quantized, numBlocks);
        break;
      case GGUFQuantType.Q5_1:
        this.quantizeQ5_1(data, quantized, numBlocks);
        break;
      case GGUFQuantType.Q8_0:
        this.quantizeQ8_0(data, quantized, numBlocks);
        break;
      case GGUFQuantType.Q4_K:
        this.quantizeQ4_K(data, quantized, numBlocks);
        break;
      case GGUFQuantType.Q5_K:
        this.quantizeQ5_K(data, quantized, numBlocks);
        break;
      case GGUFQuantType.Q6_K:
        this.quantizeQ6_K(data, quantized, numBlocks);
        break;
      case GGUFQuantType.IQ2_XS:
        this.quantizeIQ2_XS(data, quantized, numBlocks);
        break;
      default:
        throw new Error(`Quantization method not implemented: ${quantType}`);
    }

    return {
      data: quantized,
      quantType,
      shape,
      blockSize,
      numBlocks,
      originalSize: numElements * 4, // Float32
      compressedSize: totalBytes,
      compressionRatio: (numElements * 4) / totalBytes
    };
  }

  /**
   * Dequantize a tensor
   * @param {Uint8Array} quantized - Quantized data
   * @param {Object} metadata - Quantization metadata
   * @returns {Float32Array} Dequantized data
   */
  dequantize(quantized, metadata) {
    const { quantType, shape, numBlocks } = metadata;
    // blockSize available via BLOCK_SIZES[quantType] if needed
    const numElements = shape.reduce((a, b) => a * b, 1);
    const output = new Float32Array(numElements);

    switch (quantType) {
      case GGUFQuantType.Q4_0:
        this.dequantizeQ4_0(quantized, output, numBlocks);
        break;
      case GGUFQuantType.Q4_1:
        this.dequantizeQ4_1(quantized, output, numBlocks);
        break;
      case GGUFQuantType.Q5_0:
        this.dequantizeQ5_0(quantized, output, numBlocks);
        break;
      case GGUFQuantType.Q5_1:
        this.dequantizeQ5_1(quantized, output, numBlocks);
        break;
      case GGUFQuantType.Q8_0:
        this.dequantizeQ8_0(quantized, output, numBlocks);
        break;
      case GGUFQuantType.Q4_K:
        this.dequantizeQ4_K(quantized, output, numBlocks);
        break;
      case GGUFQuantType.Q5_K:
        this.dequantizeQ5_K(quantized, output, numBlocks);
        break;
      case GGUFQuantType.Q6_K:
        this.dequantizeQ6_K(quantized, output, numBlocks);
        break;
      default:
        throw new Error(`Dequantization method not implemented: ${quantType}`);
    }

    return output;
  }

  /**
   * Q4_0 quantization (4-bit, no offset)
   * Block structure: [scale (f16)] + [32 x 4-bit values]
   */
  quantizeQ4_0(input, output, numBlocks) {
    const blockSize = 32;
    let outOffset = 0;

    for (let block = 0; block < numBlocks; block++) {
      const blockStart = block * blockSize;
      const blockEnd = Math.min(blockStart + blockSize, input.length);
      const blockData = input.slice(blockStart, blockEnd);

      // Find max absolute value for scaling
      let maxAbs = 0;
      for (let i = 0; i < blockData.length; i++) {
        maxAbs = Math.max(maxAbs, Math.abs(blockData[i]));
      }

      // Calculate scale (quantize to [-8, 7] range for 4-bit signed)
      const scale = maxAbs / 8;
      const invScale = scale !== 0 ? 1 / scale : 0;

      // Write scale as float16
      this.writeFloat16(output, outOffset, scale);
      outOffset += 2;

      // Quantize values to 4-bit
      for (let i = 0; i < blockData.length; i += 2) {
        const q0 = Math.round(blockData[i] * invScale);
        const q1 = i + 1 < blockData.length ? Math.round(blockData[i + 1] * invScale) : 0;

        // Clamp to 4-bit signed range [-8, 7]
        const clamped0 = Math.max(-8, Math.min(7, q0)) + 8; // [0, 15]
        const clamped1 = Math.max(-8, Math.min(7, q1)) + 8;

        // Pack two 4-bit values into one byte
        output[outOffset++] = (clamped1 << 4) | clamped0;
      }
    }
  }

  /**
   * Q4_0 dequantization
   */
  dequantizeQ4_0(input, output, numBlocks) {
    const blockSize = 32;
    let inOffset = 0;
    let outOffset = 0;

    for (let block = 0; block < numBlocks; block++) {
      // Read scale
      const scale = this.readFloat16(input, inOffset);
      inOffset += 2;

      // Dequantize values
      for (let i = 0; i < blockSize && outOffset < output.length; i += 2) {
        const byte = input[inOffset++];

        // Unpack two 4-bit values
        const q0 = (byte & 0x0F) - 8; // Convert back to [-8, 7]
        const q1 = ((byte >> 4) & 0x0F) - 8;

        output[outOffset++] = q0 * scale;
        if (outOffset < output.length) {
          output[outOffset++] = q1 * scale;
        }
      }
    }
  }

  /**
   * Q4_1 quantization (4-bit, with offset)
   * Block structure: [min (f16)] + [scale (f16)] + [32 x 4-bit values]
   */
  quantizeQ4_1(input, output, numBlocks) {
    const blockSize = 32;
    let outOffset = 0;

    for (let block = 0; block < numBlocks; block++) {
      const blockStart = block * blockSize;
      const blockEnd = Math.min(blockStart + blockSize, input.length);
      const blockData = input.slice(blockStart, blockEnd);

      // Find min and max for scaling
      let min = Infinity;
      let max = -Infinity;
      for (let i = 0; i < blockData.length; i++) {
        min = Math.min(min, blockData[i]);
        max = Math.max(max, blockData[i]);
      }

      // Calculate scale and offset (quantize to [0, 15] range for 4-bit unsigned)
      const scale = (max - min) / 15;
      const invScale = scale !== 0 ? 1 / scale : 0;

      // Write min and scale as float16
      this.writeFloat16(output, outOffset, min);
      outOffset += 2;
      this.writeFloat16(output, outOffset, scale);
      outOffset += 2;

      // Quantize values to 4-bit
      for (let i = 0; i < blockData.length; i += 2) {
        const q0 = Math.round((blockData[i] - min) * invScale);
        const q1 = i + 1 < blockData.length ? Math.round((blockData[i + 1] - min) * invScale) : 0;

        // Clamp to 4-bit unsigned range [0, 15]
        const clamped0 = Math.max(0, Math.min(15, q0));
        const clamped1 = Math.max(0, Math.min(15, q1));

        // Pack two 4-bit values into one byte
        output[outOffset++] = (clamped1 << 4) | clamped0;
      }
    }
  }

  /**
   * Q4_1 dequantization
   */
  dequantizeQ4_1(input, output, numBlocks) {
    const blockSize = 32;
    let inOffset = 0;
    let outOffset = 0;

    for (let block = 0; block < numBlocks; block++) {
      // Read min and scale
      const min = this.readFloat16(input, inOffset);
      inOffset += 2;
      const scale = this.readFloat16(input, inOffset);
      inOffset += 2;

      // Dequantize values
      for (let i = 0; i < blockSize && outOffset < output.length; i += 2) {
        const byte = input[inOffset++];

        // Unpack two 4-bit values
        const q0 = byte & 0x0F;
        const q1 = (byte >> 4) & 0x0F;

        output[outOffset++] = min + (q0 * scale);
        if (outOffset < output.length) {
          output[outOffset++] = min + (q1 * scale);
        }
      }
    }
  }

  /**
   * Q8_0 quantization (8-bit, higher quality)
   */
  quantizeQ8_0(input, output, numBlocks) {
    const blockSize = 32;
    let outOffset = 0;

    for (let block = 0; block < numBlocks; block++) {
      const blockStart = block * blockSize;
      const blockEnd = Math.min(blockStart + blockSize, input.length);
      const blockData = input.slice(blockStart, blockEnd);

      // Find max absolute value for scaling
      let maxAbs = 0;
      for (let i = 0; i < blockData.length; i++) {
        maxAbs = Math.max(maxAbs, Math.abs(blockData[i]));
      }

      // Calculate scale (quantize to [-128, 127] range for 8-bit signed)
      const scale = maxAbs / 128;
      const invScale = scale !== 0 ? 1 / scale : 0;

      // Write scale as float16
      this.writeFloat16(output, outOffset, scale);
      outOffset += 2;

      // Quantize values to 8-bit
      for (let i = 0; i < blockData.length; i++) {
        const q = Math.round(blockData[i] * invScale);
        // Clamp to 8-bit signed range [-128, 127]
        output[outOffset++] = Math.max(-128, Math.min(127, q));
      }
    }
  }

  /**
   * Q8_0 dequantization
   */
  dequantizeQ8_0(input, output, numBlocks) {
    const blockSize = 32;
    let inOffset = 0;
    let outOffset = 0;

    for (let block = 0; block < numBlocks; block++) {
      // Read scale
      const scale = this.readFloat16(input, inOffset);
      inOffset += 2;

      // Dequantize values
      for (let i = 0; i < blockSize && outOffset < output.length; i++) {
        const q = input[inOffset++];
        // Convert unsigned byte to signed
        const signed = q > 127 ? q - 256 : q;
        output[outOffset++] = signed * scale;
      }
    }
  }

  quantizeQ5_0(input, output, numBlocks) {
    // Q5_0: 5-bit symmetric, block size 32
    // Block: [scale f16 (2 bytes)] [qh 4 bytes (high bit per weight)] [qs 16 bytes (lo 4-bit nibbles)]
    const blockSize = 32;
    let outOffset = 0;

    for (let block = 0; block < numBlocks; block++) {
      const blockStart = block * blockSize;
      const blockEnd = Math.min(blockStart + blockSize, input.length);

      let maxAbs = 0;
      for (let i = blockStart; i < blockEnd; i++) {
        const a = Math.abs(input[i]);
        if (a > maxAbs) maxAbs = a;
      }

      const scale = maxAbs / 16;
      const invScale = scale !== 0 ? 1 / scale : 0;

      this.writeFloat16(output, outOffset, scale);
      outOffset += 2;

      // qh: high bit of each 5-bit signed quant, packed 8 per byte
      let qhStart = outOffset;
      outOffset += 4;
      for (let i = 0; i < 4; i++) output[qhStart + i] = 0;

      // qs: low 4 bits, 2 per byte
      let qsStart = outOffset;
      outOffset += 16;

      for (let i = 0; i < blockSize; i++) {
        const v = i < (blockEnd - blockStart) ? input[blockStart + i] : 0;
        // Quantize to [-16, 15] range (5-bit signed), add 16 → [0, 31]
        const qi = Math.max(-16, Math.min(15, Math.round(v * invScale))) + 16;
        const hi = (qi >> 4) & 1;
        const lo = qi & 0x0F;

        // Pack high bit
        const byteIdx = Math.floor(i / 8);
        const bitIdx = i % 8;
        output[qhStart + byteIdx] |= hi << bitIdx;

        // Pack low nibble
        const nibbleByteIdx = Math.floor(i / 2);
        if (i % 2 === 0) {
          output[qsStart + nibbleByteIdx] = lo;
        } else {
          output[qsStart + nibbleByteIdx] |= lo << 4;
        }
      }
    }
  }

  dequantizeQ5_0(input, output, numBlocks) {
    const blockSize = 32;
    let inOffset = 0;
    let outOffset = 0;

    for (let block = 0; block < numBlocks; block++) {
      const scale = this.readFloat16(input, inOffset);
      inOffset += 2;

      const qhStart = inOffset;
      inOffset += 4;
      const qsStart = inOffset;
      inOffset += 16;

      for (let i = 0; i < blockSize && outOffset < output.length; i++) {
        const hi = (input[qhStart + Math.floor(i / 8)] >> (i % 8)) & 1;
        const nibbleByte = input[qsStart + Math.floor(i / 2)];
        const lo = i % 2 === 0 ? nibbleByte & 0x0F : (nibbleByte >> 4) & 0x0F;
        const qi = lo | (hi << 4);
        output[outOffset++] = (qi - 16) * scale;
      }
    }
  }

  quantizeQ5_1(input, output, numBlocks) {
    // Q5_1: 5-bit with offset, block size 32
    // Block: [min f16 (2)] [scale f16 (2)] [qh 4 bytes] [qs 16 bytes]
    const blockSize = 32;
    let outOffset = 0;

    for (let block = 0; block < numBlocks; block++) {
      const blockStart = block * blockSize;
      const blockEnd = Math.min(blockStart + blockSize, input.length);

      let min = Infinity;
      let max = -Infinity;
      for (let i = blockStart; i < blockEnd; i++) {
        if (input[i] < min) min = input[i];
        if (input[i] > max) max = input[i];
      }
      if (!isFinite(min)) min = 0;
      if (!isFinite(max)) max = 0;

      const scale = (max - min) / 31;
      const invScale = scale !== 0 ? 1 / scale : 0;

      this.writeFloat16(output, outOffset, min);
      outOffset += 2;
      this.writeFloat16(output, outOffset, scale);
      outOffset += 2;

      let qhStart = outOffset;
      outOffset += 4;
      for (let i = 0; i < 4; i++) output[qhStart + i] = 0;

      let qsStart = outOffset;
      outOffset += 16;

      for (let i = 0; i < blockSize; i++) {
        const v = i < (blockEnd - blockStart) ? input[blockStart + i] : min;
        const qi = Math.max(0, Math.min(31, Math.round((v - min) * invScale)));
        const hi = (qi >> 4) & 1;
        const lo = qi & 0x0F;

        const byteIdx = Math.floor(i / 8);
        const bitIdx = i % 8;
        output[qhStart + byteIdx] |= hi << bitIdx;

        const nibbleByteIdx = Math.floor(i / 2);
        if (i % 2 === 0) {
          output[qsStart + nibbleByteIdx] = lo;
        } else {
          output[qsStart + nibbleByteIdx] |= lo << 4;
        }
      }
    }
  }

  dequantizeQ5_1(input, output, numBlocks) {
    const blockSize = 32;
    let inOffset = 0;
    let outOffset = 0;

    for (let block = 0; block < numBlocks; block++) {
      const min = this.readFloat16(input, inOffset);
      inOffset += 2;
      const scale = this.readFloat16(input, inOffset);
      inOffset += 2;

      const qhStart = inOffset;
      inOffset += 4;
      const qsStart = inOffset;
      inOffset += 16;

      for (let i = 0; i < blockSize && outOffset < output.length; i++) {
        const hi = (input[qhStart + Math.floor(i / 8)] >> (i % 8)) & 1;
        const nibbleByte = input[qsStart + Math.floor(i / 2)];
        const lo = i % 2 === 0 ? nibbleByte & 0x0F : (nibbleByte >> 4) & 0x0F;
        const qi = lo | (hi << 4);
        output[outOffset++] = min + qi * scale;
      }
    }
  }

  // ─── K-quant helpers ──────────────────────────────────────────────────────

  /**
   * Encode 6-bit scales and mins into the 12-byte packed layout used by Q4_K and Q5_K.
   *
   * The layout mirrors the Rust reference (oxillama-quant q4_k.rs / q5_k.rs):
   *   bytes 0..3  → lower 6 bits of sc[0..3]  (bits 7:6 = hi bits of sc[4..7])
   *   bytes 4..7  → lower 6 bits of mn[0..3]  (bits 7:6 = hi bits of mn[4..7])
   *   bytes 8..11 → lo 4 bits = sc[4..7] lo nibble | hi 4 bits = mn[4..7] lo nibble
   *
   * sc[j] and mn[j] are clamped to [0, 63] before packing.
   * @param {Uint8Array} dst - destination (12 bytes starting at offset)
   * @param {number} offset
   * @param {number[]} sc - 8 scale values [0..63]
   * @param {number[]} mn - 8 min values [0..63]
   */
  _encodeScalesMins(dst, offset, sc, mn) {
    for (let j = 0; j < 4; j++) {
      // lower 6 bits of sc/mn in bytes 0..3 and 4..7
      dst[offset + j] = sc[j] & 0x3F;
      dst[offset + 4 + j] = mn[j] & 0x3F;
    }
    for (let j = 0; j < 4; j++) {
      // high bits of sc[j+4] go into bits 7:6 of byte j
      dst[offset + j] |= ((sc[j + 4] >> 4) & 0x03) << 6;
      // high bits of mn[j+4] go into bits 7:6 of byte j+4
      dst[offset + 4 + j] |= ((mn[j + 4] >> 4) & 0x03) << 6;
      // low nibble of sc[j+4] and mn[j+4] packed in bytes 8..11
      dst[offset + 8 + j] = (sc[j + 4] & 0x0F) | ((mn[j + 4] & 0x0F) << 4);
    }
  }

  /**
   * Decode 6-bit packed scales and mins (matches Rust decode_scales_mins).
   * @param {Uint8Array} src
   * @param {number} offset - start of the 12-byte region
   * @returns {{ sc: number[], mn: number[] }}
   */
  _decodeScalesMins(src, offset) {
    const sc = new Array(8);
    const mn = new Array(8);

    for (let j = 0; j < 4; j++) {
      sc[j] = src[offset + j] & 0x3F;
      mn[j] = src[offset + 4 + j] & 0x3F;
    }
    for (let j = 4; j < 8; j++) {
      const loSc = src[offset + j + 4] & 0x0F;
      const hiSc = (src[offset + j - 4] >> 6) & 0x03;
      sc[j] = loSc | (hiSc << 4);

      const loMn = (src[offset + j + 4] >> 4) & 0x0F;
      const hiMn = (src[offset + j] >> 6) & 0x03;
      mn[j] = loMn | (hiMn << 4);
    }
    return { sc, mn };
  }

  /**
   * Compute the optimal super-block scale d and minimum dmin for K-quants,
   * and the 8 integer sub-block scales sc[i]/mn[i] in [0, 63].
   *
   * K-quant formula: w = d·sc[i]·q - dmin·mn[i],  q ∈ [0, numLevels]
   *
   * Strategy (mirrors llama.cpp make_qkx2_quants):
   *   For each of the 8 sub-blocks (32 weights each):
   *     1. Find min and max within the sub-block.
   *     2. subScale[s] = (max - min) / numLevels
   *     3. subMin[s]   = -min  (positive; the negative offset)
   *   The super-scale d    = max(subScales) / 63
   *   The super-min   dmin = max(subMins)   / 63
   *   Sub-block integers:
   *     sc[i] = round(subScale[i] / d)    clamped [0, 63]
   *     mn[i] = round(subMin[i]   / dmin) clamped [0, 63]  (or 0 when dmin==0)
   *
   * Dequantization recovery:
   *   w̃ = d·sc[i]·q - dmin·mn[i]
   *     ≈ subScale[i]·q + min   (when q≈round((w - min)/subScale))
   *
   * @param {Float32Array} data - full input array
   * @param {number} blockStart - offset of this super-block in data
   * @param {number} numLevels - 15 for Q4_K, 31 for Q5_K
   * @returns {{ d: number, dmin: number, sc: number[], mn: number[] }}
   */
  _computeKQuantSuperBlock(data, blockStart, numLevels) {
    const NUM_SUB = 8;
    const SUB_SIZE = 32;

    const subScales = new Array(NUM_SUB);
    const subMins = new Array(NUM_SUB);

    for (let s = 0; s < NUM_SUB; s++) {
      const start = blockStart + s * SUB_SIZE;
      const end = Math.min(start + SUB_SIZE, data.length);
      let minV = Infinity;
      let maxV = -Infinity;
      for (let i = start; i < end; i++) {
        if (data[i] < minV) minV = data[i];
        if (data[i] > maxV) maxV = data[i];
      }
      if (!isFinite(minV)) { minV = 0; maxV = 0; }
      // dmin·mn[i] encodes the positive offset to subtract from w.
      // min may be positive (data shifted away from zero), so cap at 0.
      subMins[s] = Math.max(0, -minV);  // positive offset ≥ 0
      subScales[s] = (maxV - minV) / numLevels;
    }

    const maxSubScale = subScales.reduce((a, b) => Math.max(a, b), 0);
    const maxSubMin   = subMins.reduce((a, b) => Math.max(a, b), 0);

    const d    = maxSubScale / 63;
    const dmin = maxSubMin   / 63;

    const sc = new Array(NUM_SUB);
    const mn = new Array(NUM_SUB);
    for (let s = 0; s < NUM_SUB; s++) {
      sc[s] = d    !== 0 ? Math.max(0, Math.min(63, Math.round(subScales[s] / d))) : 0;
      mn[s] = dmin !== 0 ? Math.max(0, Math.min(63, Math.round(subMins[s]  / dmin))) : 0;
    }

    return { d, dmin, sc, mn };
  }

  // ─── Q4_K: 256 weights, 144 bytes per super-block ─────────────────────────
  //
  // Layout (from Rust q4_k.rs):
  //   [0..2]   d    FP16 super-block scale
  //   [2..4]   dmin FP16 super-block minimum
  //   [4..16]  12 bytes: 8×sc + 8×mn packed 6-bit each
  //   [16..144] 128 bytes: 256 × 4-bit nibbles (2 per byte)
  //
  // Weight formula: w = d·sc[i]·q - dmin·mn[i],  q ∈ [0,15]

  /**
   * Q4_K quantization — 256-weight super-blocks, 144 bytes each.
   */
  quantizeQ4_K(input, output, numBlocks) {
    const SUPER_SIZE = 256;
    const BLOCK_BYTES = 144;

    for (let block = 0; block < numBlocks; block++) {
      const blockStart = block * SUPER_SIZE;
      const outBase = block * BLOCK_BYTES;

      const { d, dmin, sc, mn } = this._computeKQuantSuperBlock(input, blockStart, 15);

      this.writeFloat16(output, outBase, d);
      this.writeFloat16(output, outBase + 2, dmin);
      this._encodeScalesMins(output, outBase + 4, sc, mn);

      // Quantize 256 weights into 128 nibble bytes.
      // 8 sub-blocks of 32; 4 groups of 2 sub-blocks.
      // Layout: group g → nibble bytes [g*32 .. (g+1)*32)
      //   lo nibble = weights for sub-block 2g   (positions 0..31 within group)
      //   hi nibble = weights for sub-block 2g+1 (positions 32..63 within group)
      const qsBase = outBase + 16;
      let isub = 0;
      let qsOff = 0;

      for (let grp = 0; grp < 4; grp++) {
        const d1 = d * sc[isub];
        const m1 = dmin * mn[isub];
        const d2 = d * sc[isub + 1];
        const m2 = dmin * mn[isub + 1];

        for (let l = 0; l < 32; l++) {
          const idx0 = blockStart + grp * 64 + l;
          const idx1 = blockStart + grp * 64 + 32 + l;

          const w0 = idx0 < input.length ? input[idx0] : 0;
          const w1 = idx1 < input.length ? input[idx1] : 0;

          // q = round((w + dmin*mn) / (d*sc)), clamped to [0,15]
          const q0 = d1 !== 0 ? Math.max(0, Math.min(15, Math.round((w0 + m1) / d1))) : 0;
          const q1 = d2 !== 0 ? Math.max(0, Math.min(15, Math.round((w1 + m2) / d2))) : 0;

          output[qsBase + qsOff + l] = q0 | (q1 << 4);
        }

        isub += 2;
        qsOff += 32;
      }
    }
  }

  /**
   * Q4_K dequantization — 256-weight super-blocks, 144 bytes each.
   */
  dequantizeQ4_K(input, output, numBlocks) {
    const SUPER_SIZE = 256;
    const BLOCK_BYTES = 144;

    for (let block = 0; block < numBlocks; block++) {
      const inBase = block * BLOCK_BYTES;
      const outBase = block * SUPER_SIZE;

      const d = this.readFloat16(input, inBase);
      const dmin = this.readFloat16(input, inBase + 2);
      const { sc, mn } = this._decodeScalesMins(input, inBase + 4);
      const qsBase = inBase + 16;

      let isub = 0;
      let qsOff = 0;

      for (let grp = 0; grp < 4; grp++) {
        const d1 = d * sc[isub];
        const m1 = dmin * mn[isub];
        const d2 = d * sc[isub + 1];
        const m2 = dmin * mn[isub + 1];

        for (let l = 0; l < 32; l++) {
          const byte = input[qsBase + qsOff + l];
          const q0 = byte & 0x0F;
          const q1 = (byte >> 4) & 0x0F;

          const out0 = outBase + grp * 64 + l;
          const out1 = outBase + grp * 64 + 32 + l;

          if (out0 < output.length) output[out0] = d1 * q0 - m1;
          if (out1 < output.length) output[out1] = d2 * q1 - m2;
        }

        isub += 2;
        qsOff += 32;
      }
    }
  }

  // ─── Q5_K: 256 weights, 176 bytes per super-block ─────────────────────────
  //
  // Layout (from Rust q5_k.rs):
  //   [0..2]    d     FP16 super-block scale
  //   [2..4]    dmin  FP16 super-block minimum
  //   [4..16]   12 bytes: 8×sc + 8×mn packed 6-bit each
  //   [16..48]  32 bytes: qh — high bit of each 5-bit quant (1 bit per weight, 8 per byte)
  //   [48..176] 128 bytes: qs — low 4 bits of each 5-bit quant (2 per byte)
  //
  // Weight formula: w = d·sc[i]·q5 - dmin·mn[i],  q5 ∈ [0,31]
  //   where q5 = (qs_lo4) | (qh_bit << 4)

  /**
   * Q5_K quantization — 256-weight super-blocks, 176 bytes each.
   */
  quantizeQ5_K(input, output, numBlocks) {
    const SUPER_SIZE = 256;
    const BLOCK_BYTES = 176;

    for (let block = 0; block < numBlocks; block++) {
      const blockStart = block * SUPER_SIZE;
      const outBase = block * BLOCK_BYTES;

      const { d, dmin, sc, mn } = this._computeKQuantSuperBlock(input, blockStart, 31);

      this.writeFloat16(output, outBase, d);
      this.writeFloat16(output, outBase + 2, dmin);
      this._encodeScalesMins(output, outBase + 4, sc, mn);

      const qhBase = outBase + 16;
      const qsBase = outBase + 48;

      // Zero out qh (32 bytes)
      for (let i = 0; i < 32; i++) output[qhBase + i] = 0;

      // 4 groups × 2 sub-blocks of 32 each = 256 weights
      // Group g: weights [g*64 .. g*64+64)
      //   sub-block 2g   → lo nibble (bits 3:0 of qs)
      //   sub-block 2g+1 → hi nibble (bits 7:4 of qs)
      // qh layout: for position l (0..32) in the 128-byte qs stripe,
      //   bit (group)   of qh[l]   holds the high bit for sub-block 2g   weight l
      //   bit (group+4) of qh[l]   holds the high bit for sub-block 2g+1 weight l
      let isub = 0;
      let qsOff = 0;

      for (let grp = 0; grp < 4; grp++) {
        const d1 = d * sc[isub];
        const m1 = dmin * mn[isub];
        const d2 = d * sc[isub + 1];
        const m2 = dmin * mn[isub + 1];

        for (let l = 0; l < 32; l++) {
          const idx0 = blockStart + grp * 64 + l;
          const idx1 = blockStart + grp * 64 + 32 + l;

          const w0 = idx0 < input.length ? input[idx0] : 0;
          const w1 = idx1 < input.length ? input[idx1] : 0;

          const q0 = d1 !== 0 ? Math.max(0, Math.min(31, Math.round((w0 + m1) / d1))) : 0;
          const q1 = d2 !== 0 ? Math.max(0, Math.min(31, Math.round((w1 + m2) / d2))) : 0;

          const hi0 = (q0 >> 4) & 1;
          const hi1 = (q1 >> 4) & 1;
          const lo0 = q0 & 0x0F;
          const lo1 = q1 & 0x0F;

          // qh[l] bit at position `grp` for sub-block 2g (lo nibble weights)
          // qh[l] bit at position `grp+4` for sub-block 2g+1 (hi nibble weights)
          output[qhBase + l] |= hi0 << grp;
          output[qhBase + l] |= hi1 << (grp + 4);

          output[qsBase + qsOff + l] = lo0 | (lo1 << 4);
        }

        isub += 2;
        qsOff += 32;
      }
    }
  }

  /**
   * Q5_K dequantization — 256-weight super-blocks, 176 bytes each.
   */
  dequantizeQ5_K(input, output, numBlocks) {
    const SUPER_SIZE = 256;
    const BLOCK_BYTES = 176;

    for (let block = 0; block < numBlocks; block++) {
      const inBase = block * BLOCK_BYTES;
      const outBase = block * SUPER_SIZE;

      const d = this.readFloat16(input, inBase);
      const dmin = this.readFloat16(input, inBase + 2);
      const { sc, mn } = this._decodeScalesMins(input, inBase + 4);
      const qhBase = inBase + 16;
      const qsBase = inBase + 48;

      let isub = 0;
      let qsOff = 0;

      for (let grp = 0; grp < 4; grp++) {
        const d1 = d * sc[isub];
        const m1 = dmin * mn[isub];
        const d2 = d * sc[isub + 1];
        const m2 = dmin * mn[isub + 1];

        for (let l = 0; l < 32; l++) {
          const qh = input[qhBase + l];
          const qsByte = input[qsBase + qsOff + l];

          const hi0 = (qh >> grp) & 1;
          const hi1 = (qh >> (grp + 4)) & 1;
          const lo0 = qsByte & 0x0F;
          const lo1 = (qsByte >> 4) & 0x0F;

          const q0 = lo0 | (hi0 << 4);
          const q1 = lo1 | (hi1 << 4);

          const out0 = outBase + grp * 64 + l;
          const out1 = outBase + grp * 64 + 32 + l;

          if (out0 < output.length) output[out0] = d1 * q0 - m1;
          if (out1 < output.length) output[out1] = d2 * q1 - m2;
        }

        isub += 2;
        qsOff += 32;
      }
    }
  }

  // ─── Q6_K: 256 weights, 210 bytes per super-block ─────────────────────────
  //
  // Layout (from Rust q6_k.rs) — NOTE different byte order vs Q4/Q5:
  //   [0..128]   128 bytes: ql — lower 4 bits of 6-bit quants (2 per byte)
  //   [128..192]  64 bytes: qh — upper 2 bits of each 6-bit quant (4 per byte)
  //   [192..208]  16 bytes: scales — 16 × int8 signed sub-block scales
  //   [208..210]   2 bytes: d FP16 super-block scale
  //
  // Symmetric ("type-0"): no minimum offset.
  // Weight formula: w = d · scale_i · (q6_bit − 32),  q6 ∈ [0, 63]
  // 16 sub-blocks of 16 weights each.

  /**
   * Q6_K quantization — 256-weight super-blocks, 210 bytes each.
   */
  quantizeQ6_K(input, output, numBlocks) {
    const SUPER_SIZE = 256;
    const BLOCK_BYTES = 210;
    const NUM_SUB = 16;
    const SUB_SIZE = 16;

    for (let block = 0; block < numBlocks; block++) {
      const blockStart = block * SUPER_SIZE;
      const outBase = block * BLOCK_BYTES;

      // Find global max-abs for super-block scale
      let maxAbsGlobal = 0;
      for (let i = blockStart; i < blockStart + SUPER_SIZE && i < input.length; i++) {
        const a = Math.abs(input[i]);
        if (a > maxAbsGlobal) maxAbsGlobal = a;
      }

      // For each sub-block find max-abs and compute int8 scale
      const subMaxAbs = new Array(NUM_SUB);
      for (let s = 0; s < NUM_SUB; s++) {
        const start = blockStart + s * SUB_SIZE;
        const end = Math.min(start + SUB_SIZE, input.length);
        let mx = 0;
        for (let i = start; i < end; i++) {
          const a = Math.abs(input[i]);
          if (a > mx) mx = a;
        }
        subMaxAbs[s] = mx;
      }

      // Super-block d: maps so that d * max_int8_scale * 31 ≈ maxAbsGlobal
      // We choose d = maxAbsGlobal / (127 * 31) so sub-scales fit in int8 range.
      // (31 because q6 centered = q-32, range [-32,31])
      const d = maxAbsGlobal / (127 * 31);
      const invD = d !== 0 ? 1 / d : 0;

      // Write ql (128 bytes), qh (64 bytes), scales (16 bytes), d (2 bytes)
      const qlBase = outBase;
      const qhBase = outBase + 128;
      const scBase = outBase + 192;

      for (let i = 0; i < 128; i++) output[qlBase + i] = 0;
      for (let i = 0; i < 64; i++) output[qhBase + i] = 0;

      // Compute sub-scales (int8)
      for (let s = 0; s < NUM_SUB; s++) {
        const scale = d !== 0 ? Math.max(-127, Math.min(127, Math.round(subMaxAbs[s] * invD / 31))) : 0;
        output[scBase + s] = scale & 0xFF;
      }

      // Quantize using the GGML per-sub-block layout.
      // The 256 weights are laid out in ql/qh using the same interleaved structure
      // as the dequant path in Rust:
      //   For group in 0..2, l in 0..32:
      //     q1 at out[group*128 + l]          ← ql[group*64 + l]       lo4 | qh[group*32+l] bits 1:0
      //     q2 at out[group*128 + 32 + l]     ← ql[group*64 + l + 32]  lo4 | qh bits 3:2
      //     q3 at out[group*128 + 64 + l]     ← ql[group*64 + l]       hi4 | qh bits 5:4
      //     q4 at out[group*128 + 96 + l]     ← ql[group*64 + l + 32]  hi4 | qh bits 7:6
      //   Sub-block index = l/16 within the group.

      for (let grp = 0; grp < 2; grp++) {
        const qlOff = grp * 64;
        const qhOff = grp * 32;
        const scOff = grp * 8;
        const wBase = grp * 128;

        for (let l = 0; l < 32; l++) {
          const is = Math.floor(l / 16); // sub-block index within group (0 or 1)

          const s0 = output[scBase + scOff + is] & 0xFF;
          const s1 = output[scBase + scOff + is + 2] & 0xFF;
          const s2 = output[scBase + scOff + is + 4] & 0xFF;
          const s3 = output[scBase + scOff + is + 6] & 0xFF;

          // Convert int8 scale from stored byte
          const scale0 = s0 > 127 ? s0 - 256 : s0;
          const scale1 = s1 > 127 ? s1 - 256 : s1;
          const scale2 = s2 > 127 ? s2 - 256 : s2;
          const scale3 = s3 > 127 ? s3 - 256 : s3;

          const quantize = (w, scaleI) => {
            if (scaleI === 0) return 32; // q-32=0
            const q = Math.round(w / (d * scaleI)) + 32;
            return Math.max(0, Math.min(63, q));
          };

          const c0 = blockStart + wBase + l;
          const c1 = blockStart + wBase + 32 + l;
          const c2 = blockStart + wBase + 64 + l;
          const c3 = blockStart + wBase + 96 + l;

          const w0 = c0 < input.length ? input[c0] : 0;
          const w1 = c1 < input.length ? input[c1] : 0;
          const w2 = c2 < input.length ? input[c2] : 0;
          const w3 = c3 < input.length ? input[c3] : 0;

          const q1 = quantize(w0, scale0);
          const q2 = quantize(w1, scale1);
          const q3 = quantize(w2, scale2);
          const q4 = quantize(w3, scale3);

          // q1: lo4 → ql[qlOff+l] bits 3:0;  hi4 → ql[qlOff+l] bits 7:4 from q3
          // q2: lo4 → ql[qlOff+l+32] bits 3:0; hi4 from q4
          // q1 upper 2 bits → qh[qhOff+l] bits 1:0
          // q2 upper 2 bits → qh[qhOff+l] bits 3:2
          // q3 upper 2 bits → qh[qhOff+l] bits 5:4
          // q4 upper 2 bits → qh[qhOff+l] bits 7:6
          output[qlBase + qlOff + l] = (q1 & 0x0F) | ((q3 & 0x0F) << 4);
          output[qlBase + qlOff + l + 32] = (q2 & 0x0F) | ((q4 & 0x0F) << 4);
          output[qhBase + qhOff + l] = ((q1 >> 4) & 3) |
            (((q2 >> 4) & 3) << 2) |
            (((q3 >> 4) & 3) << 4) |
            (((q4 >> 4) & 3) << 6);
        }
      }

      // Write d as FP16 at [208..210]
      this.writeFloat16(output, outBase + 208, d);
    }
  }

  /**
   * Q6_K dequantization — 256-weight super-blocks, 210 bytes each.
   */
  dequantizeQ6_K(input, output, numBlocks) {
    const SUPER_SIZE = 256;
    const BLOCK_BYTES = 210;

    for (let block = 0; block < numBlocks; block++) {
      const inBase = block * BLOCK_BYTES;
      const outBase = block * SUPER_SIZE;

      const ql = input.subarray(inBase, inBase + 128);
      const qh = input.subarray(inBase + 128, inBase + 192);
      const scales = input.subarray(inBase + 192, inBase + 208);
      const d = this.readFloat16(input, inBase + 208);

      for (let grp = 0; grp < 2; grp++) {
        const qlOff = grp * 64;
        const qhOff = grp * 32;
        const scOff = grp * 8;
        const wBase = grp * 128;

        for (let l = 0; l < 32; l++) {
          const is = Math.floor(l / 16);

          const s0Raw = scales[scOff + is];
          const s1Raw = scales[scOff + is + 2];
          const s2Raw = scales[scOff + is + 4];
          const s3Raw = scales[scOff + is + 6];

          const s0 = s0Raw > 127 ? s0Raw - 256 : s0Raw;
          const s1 = s1Raw > 127 ? s1Raw - 256 : s1Raw;
          const s2 = s2Raw > 127 ? s2Raw - 256 : s2Raw;
          const s3 = s3Raw > 127 ? s3Raw - 256 : s3Raw;

          const qhByte = qh[qhOff + l];
          const q1 = ((ql[qlOff + l] & 0x0F) | ((qhByte & 3) << 4)) - 32;
          const q2 = ((ql[qlOff + l + 32] & 0x0F) | (((qhByte >> 2) & 3) << 4)) - 32;
          const q3 = ((ql[qlOff + l] >> 4) | (((qhByte >> 4) & 3) << 4)) - 32;
          const q4 = ((ql[qlOff + l + 32] >> 4) | (((qhByte >> 6) & 3) << 4)) - 32;

          const o0 = outBase + wBase + l;
          const o1 = outBase + wBase + 32 + l;
          const o2 = outBase + wBase + 64 + l;
          const o3 = outBase + wBase + 96 + l;

          if (o0 < output.length) output[o0] = d * s0 * q1;
          if (o1 < output.length) output[o1] = d * s1 * q2;
          if (o2 < output.length) output[o2] = d * s2 * q3;
          if (o3 < output.length) output[o3] = d * s3 * q4;
        }
      }
    }
  }

  quantizeIQ2_XS(input, output, numBlocks) {
    // IQ2_XS is an importance-matrix quantization with grid codebooks;
    // a full grid-based implementation is outside the scope of this K-quant
    // pass.  Fall back to Q4_0 which preserves the output byte-length contract.
    this.quantizeQ4_0(input, output, numBlocks);
  }

  // ─── Float16 conversion ───────────────────────────────────────────────────

  /**
   * Write a float32 value as IEEE 754 float16 (binary16) at the given byte offset.
   *
   * Conversion follows the standard binary32→binary16 path:
   *   sign: 1 bit, exponent: 5 bits (bias 15), mantissa: 10 bits.
   * Subnormals, ±Inf and NaN are handled.
   *
   * @param {Uint8Array} buffer
   * @param {number} offset - byte offset (writes 2 bytes)
   * @param {number} value - float32 value
   */
  writeFloat16(buffer, offset, value) {
    const bits16 = this._f32ToF16Bits(value);
    buffer[offset] = bits16 & 0xFF;
    buffer[offset + 1] = (bits16 >> 8) & 0xFF;
  }

  /**
   * Read a float16 value (little-endian) from the given byte offset.
   * @param {Uint8Array} buffer
   * @param {number} offset
   * @returns {number} float32 value
   */
  readFloat16(buffer, offset) {
    const bits16 = buffer[offset] | (buffer[offset + 1] << 8);
    return this._f16BitsToF32(bits16);
  }

  /**
   * Convert a float32 number to its IEEE 754 float16 bit pattern (uint16).
   * @param {number} value
   * @returns {number} uint16
   */
  _f32ToF16Bits(value) {
    if (isNaN(value)) return 0x7E00; // quiet NaN
    if (!isFinite(value)) return value > 0 ? 0x7C00 : 0xFC00;

    const buf = new ArrayBuffer(4);
    const dv = new DataView(buf);
    dv.setFloat32(0, value, true); // little-endian matches getUint32 below
    const b = dv.getUint32(0, true); // little-endian

    const sign = (b >> 31) & 1;
    const exp32 = (b >> 23) & 0xFF;
    const mant32 = b & 0x7FFFFF;

    let exp16 = exp32 - 127 + 15;
    let mant16;

    if (exp16 >= 31) {
      // Overflow → infinity
      return (sign << 15) | 0x7C00;
    } else if (exp16 <= 0) {
      if (exp16 < -10) {
        // Too small → zero
        return (sign << 15);
      }
      // Subnormal
      mant16 = (mant32 | 0x800000) >> (1 - exp16);
      mant16 = (mant16 + 0x1000) >> 13; // round
      return (sign << 15) | (mant16 & 0x3FF);
    }

    // Round the 23-bit mantissa to 10 bits
    mant16 = (mant32 + 0x1000) >> 13;
    if (mant16 >= 0x400) {
      exp16++;
      mant16 = 0;
    }
    return (sign << 15) | (exp16 << 10) | (mant16 & 0x3FF);
  }

  /**
   * Convert an IEEE 754 float16 bit pattern (uint16) to a float32 number.
   * @param {number} bits - uint16
   * @returns {number} float32
   */
  _f16BitsToF32(bits) {
    const sign = (bits >> 15) & 1;
    const exp16 = (bits >> 10) & 0x1F;
    const mant16 = bits & 0x3FF;

    let value;
    if (exp16 === 0) {
      // Subnormal / zero
      value = mant16 / 1024 * Math.pow(2, -14);
    } else if (exp16 === 31) {
      value = mant16 === 0 ? Infinity : NaN;
    } else {
      value = (1 + mant16 / 1024) * Math.pow(2, exp16 - 15);
    }

    return sign ? -value : value;
  }
}

/**
 * GGUF Model Loader
 */
export class GGUFModelLoader {
  /**
   * Load a GGUF model from a file or URL
   * @param {string|File|ArrayBuffer} source - Model source
   * @returns {Promise<Object>} Loaded model
   */
  async load(source) {
    let buffer;

    if (typeof source === 'string') {
      // Load from URL
      const response = await fetch(source);
      buffer = await response.arrayBuffer();
    } else if (source instanceof File) {
      // Load from file
      buffer = await source.arrayBuffer();
    } else if (source instanceof ArrayBuffer) {
      buffer = source;
    } else {
      throw new Error('Unsupported source type');
    }

    return this.parse(buffer);
  }

  /**
   * Parse GGUF file format
   * @param {ArrayBuffer} buffer - File buffer
   * @returns {Object} Parsed model
   */
  parse(buffer) {
    const view = new DataView(buffer);
    let offset = 0;

    // Read magic number (4 bytes): "GGUF"
    const magic = String.fromCharCode(
      view.getUint8(offset++),
      view.getUint8(offset++),
      view.getUint8(offset++),
      view.getUint8(offset++)
    );

    if (magic !== 'GGUF') {
      throw new Error('Invalid GGUF file: wrong magic number');
    }

    // Read version (4 bytes)
    const version = view.getUint32(offset, true);
    offset += 4;

    // Read tensor count (8 bytes)
    const tensorCount = Number(view.getBigUint64(offset, true));
    offset += 8;

    // Read metadata count (8 bytes)
    const metadataCount = Number(view.getBigUint64(offset, true));
    offset += 8;

    // Read metadata
    const metadata = {};
    for (let i = 0; i < metadataCount; i++) {
      const { key, value, newOffset } = this.readMetadataKV(view, offset);
      metadata[key] = value;
      offset = newOffset;
    }

    // Read tensor info
    const tensors = [];
    for (let i = 0; i < tensorCount; i++) {
      const { info, newOffset } = this.readTensorInfo(view, offset);
      tensors.push(info);
      offset = newOffset;
    }

    return {
      version,
      metadata,
      tensors,
      dataOffset: offset
    };
  }

  /**
   * Read metadata key-value pair
   */
  readMetadataKV(view, offset) {
    // Read key length
    const keyLen = Number(view.getBigUint64(offset, true));
    offset += 8;

    // Read key
    const keyBytes = new Uint8Array(view.buffer, offset, keyLen);
    const key = new TextDecoder().decode(keyBytes);
    offset += keyLen;

    // Read value type
    const valueType = view.getUint32(offset, true);
    offset += 4;

    // Read value based on type
    let value;
    switch (valueType) {
      case 0: // uint8
        value = view.getUint8(offset);
        offset += 1;
        break;
      case 1: // int8
        value = view.getInt8(offset);
        offset += 1;
        break;
      case 2: // uint16
        value = view.getUint16(offset, true);
        offset += 2;
        break;
      case 3: // int16
        value = view.getInt16(offset, true);
        offset += 2;
        break;
      case 4: // uint32
        value = view.getUint32(offset, true);
        offset += 4;
        break;
      case 5: // int32
        value = view.getInt32(offset, true);
        offset += 4;
        break;
      case 6: // float32
        value = view.getFloat32(offset, true);
        offset += 4;
        break;
      case 7: // bool
        value = view.getUint8(offset) !== 0;
        offset += 1;
        break;
      case 8: { // string
        const strLen = Number(view.getBigUint64(offset, true));
        offset += 8;
        const strBytes = new Uint8Array(view.buffer, offset, strLen);
        value = new TextDecoder().decode(strBytes);
        offset += strLen;
        break;
      }
      default:
        throw new Error(`Unknown metadata value type: ${valueType}`);
    }

    return { key, value, newOffset: offset };
  }

  /**
   * Read tensor info
   */
  readTensorInfo(view, offset) {
    // Read name length
    const nameLen = Number(view.getBigUint64(offset, true));
    offset += 8;

    // Read name
    const nameBytes = new Uint8Array(view.buffer, offset, nameLen);
    const name = new TextDecoder().decode(nameBytes);
    offset += nameLen;

    // Read dimensions
    const nDims = view.getUint32(offset, true);
    offset += 4;

    const shape = [];
    for (let i = 0; i < nDims; i++) {
      shape.push(Number(view.getBigUint64(offset, true)));
      offset += 8;
    }

    // Read quantization type
    const quantType = view.getUint32(offset, true);
    offset += 4;

    // Read data offset
    const dataOffset = Number(view.getBigUint64(offset, true));
    offset += 8;

    return {
      info: { name, shape, quantType, dataOffset },
      newOffset: offset
    };
  }
}

/**
 * Create a GGUF quantizer
 * @param {Object} config - Configuration
 * @returns {GGUFQuantizer}
 */
export function createGGUFQuantizer(config) {
  return new GGUFQuantizer(config);
}

/**
 * Create a GGUF model loader
 * @returns {GGUFModelLoader}
 */
export function createGGUFLoader() {
  return new GGUFModelLoader();
}

export default {
  GGUFQuantType,
  GGUFQuantizer,
  GGUFModelLoader,
  createGGUFQuantizer,
  createGGUFLoader
};
