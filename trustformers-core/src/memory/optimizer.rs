//! Memory optimizer for training and inference.
//!
//! Strategies: activation offloading, optimizer state compression, swap.
//!
//! This module provides tools for analysing the memory footprint of a
//! transformer model and recommending (or applying) memory reduction strategies
//! such as gradient checkpointing, activation compression, and optimizer-state
//! precision reduction.

use std::fmt;

use crate::grad_checkpoint::selective::SelectiveCheckpointStrategy;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by memory optimisation operations.
#[derive(Debug, Clone, PartialEq)]
pub enum MemOptError {
    /// The compressed byte slice has an unexpected length.
    InvalidCompressedLength { expected_min: usize, got: usize },
    /// A quantisation bit-width is not supported.
    UnsupportedBitWidth(u8),
    /// The decompressed data does not match the expected element count.
    LengthMismatch { expected: usize, got: usize },
    /// A configuration parameter is invalid.
    InvalidConfig(String),
}

impl fmt::Display for MemOptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemOptError::InvalidCompressedLength { expected_min, got } => {
                write!(
                    f,
                    "compressed buffer too short: expected at least {expected_min} bytes, got {got}"
                )
            },
            MemOptError::UnsupportedBitWidth(bits) => {
                write!(f, "unsupported activation quantisation bit width: {bits}")
            },
            MemOptError::LengthMismatch { expected, got } => {
                write!(
                    f,
                    "decompressed length mismatch: expected {expected} elements, got {got}"
                )
            },
            MemOptError::InvalidConfig(msg) => write!(f, "invalid memory optimiser config: {msg}"),
        }
    }
}

impl std::error::Error for MemOptError {}

// ─────────────────────────────────────────────────────────────────────────────
// StatePrecision
// ─────────────────────────────────────────────────────────────────────────────

/// Numeric precision used for optimizer states (e.g. Adam m/v moments).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StatePrecision {
    /// 32-bit float — full precision (standard).
    Fp32,
    /// 16-bit float — halves memory relative to Fp32.
    Fp16,
    /// Brain float 16 — same size as Fp16 but better dynamic range.
    Bf16,
    /// 8-bit quantised optimizer states (Dettmers et al., 2022 — bitsandbytes).
    Int8Quantized,
}

// ─────────────────────────────────────────────────────────────────────────────
// ActivationCompression
// ─────────────────────────────────────────────────────────────────────────────

/// Compression method applied to activations before they are stashed for the
/// backward pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationCompression {
    /// No compression — store raw f32 bytes.
    None,
    /// Down-cast each element to BF16 (2 bytes per element).
    Fp16Cast,
    /// Quantise activations to `bits` bits per element.
    Quantized {
        /// Number of bits per quantised value (1–8 supported).
        bits: u8,
    },
    /// Store only elements with absolute value above `threshold`; zero the rest.
    Sparse {
        /// Sparsity threshold — elements with `|x| <= threshold` are zeroed.
        threshold: f32,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// MemoryOptimizationConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the [`MemoryOptimizer`].
#[derive(Debug, Clone)]
pub struct MemoryOptimizationConfig {
    /// Offload activation tensors to CPU memory during forward pass.
    pub offload_activations: bool,
    /// Offload Adam m/v states to CPU memory.
    pub offload_optimizer_states: bool,
    /// Use memory-efficient (flash) attention.
    pub use_memory_efficient_attention: bool,
    /// Gradient checkpointing strategy.
    pub gradient_checkpointing: SelectiveCheckpointStrategy,
    /// Numeric precision for optimizer states.
    pub optimizer_state_precision: StatePrecision,
    /// Compression method for saved activations.
    pub activation_compression: ActivationCompression,
    /// Optional target peak GPU memory in megabytes.
    pub max_gpu_memory_mb: Option<f32>,
}

impl Default for MemoryOptimizationConfig {
    fn default() -> Self {
        Self {
            offload_activations: false,
            offload_optimizer_states: false,
            use_memory_efficient_attention: true,
            gradient_checkpointing: SelectiveCheckpointStrategy::None,
            optimizer_state_precision: StatePrecision::Fp32,
            activation_compression: ActivationCompression::None,
            max_gpu_memory_mb: Option::None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MemoryBudget
// ─────────────────────────────────────────────────────────────────────────────

/// Breakdown of GPU memory usage across the major consumers.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    /// Total GPU memory available in MB.
    pub total_gpu_memory_mb: f32,
    /// Memory used by model weight tensors.
    pub model_weights_mb: f32,
    /// Memory used by optimizer states (e.g. Adam moments).
    pub optimizer_states_mb: f32,
    /// Memory used by activation tensors during the forward pass.
    pub activations_mb: f32,
    /// Memory used by gradient tensors.
    pub gradients_mb: f32,
    /// Reserved memory (CUDA context, fragmentation headroom, etc.).
    pub reserved_mb: f32,
}

impl MemoryBudget {
    /// Memory available for storing activations (total minus all other consumers).
    pub fn available_for_activations_mb(&self) -> f32 {
        let used =
            self.model_weights_mb + self.optimizer_states_mb + self.gradients_mb + self.reserved_mb;
        (self.total_gpu_memory_mb - used).max(0.0)
    }

    /// Returns `true` when all fields are non-negative and the sum of consumers
    /// does not exceed the total.
    pub fn is_feasible(&self) -> bool {
        let non_negative = self.total_gpu_memory_mb >= 0.0
            && self.model_weights_mb >= 0.0
            && self.optimizer_states_mb >= 0.0
            && self.activations_mb >= 0.0
            && self.gradients_mb >= 0.0
            && self.reserved_mb >= 0.0;
        if !non_negative {
            return false;
        }
        let total_used = self.model_weights_mb
            + self.optimizer_states_mb
            + self.activations_mb
            + self.gradients_mb
            + self.reserved_mb;
        total_used <= self.total_gpu_memory_mb
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Free-standing estimation functions
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate the GPU memory (MB) required to store model weights.
///
/// # Arguments
/// * `num_params`   – number of model parameters
/// * `dtype_bytes`  – bytes per element (e.g. 4 for f32, 2 for f16)
pub fn estimate_model_memory(num_params: u64, dtype_bytes: u32) -> f32 {
    (num_params as f64 * dtype_bytes as f64 / (1024.0 * 1024.0)) as f32
}

/// Estimate the GPU memory (MB) for optimizer states.
///
/// For Adam with two moment vectors (m + v):
/// - Fp32  : 8 bytes per parameter
/// - Fp16/Bf16: 4 bytes per parameter
/// - Int8  : 1 byte per parameter
pub fn estimate_optimizer_state_memory(num_params: u64, precision: StatePrecision) -> f32 {
    let bytes_per_param: u64 = match precision {
        StatePrecision::Fp32 => 8,
        StatePrecision::Fp16 | StatePrecision::Bf16 => 4,
        StatePrecision::Int8Quantized => 1,
    };
    (num_params as f64 * bytes_per_param as f64 / (1024.0 * 1024.0)) as f32
}

/// Estimate the GPU memory (MB) for activations during the forward pass.
///
/// # Formula
///
/// ```text
/// baseline = batch * seq * hidden * 4 (f32) * num_layers * 2  (attn + FFN)
/// with_ckpt = baseline * (1 - checkpoint_ratio)
/// ```
///
/// # Arguments
/// * `batch_size`        – training batch size
/// * `seq_len`           – sequence length
/// * `hidden_size`       – model hidden dimension
/// * `num_layers`        – number of transformer layers
/// * `checkpoint_ratio`  – fraction of layers that are checkpointed (0.0–1.0)
pub fn estimate_activation_memory(
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
    num_layers: usize,
    checkpoint_ratio: f32,
) -> f32 {
    let baseline = batch_size as f64
        * seq_len as f64
        * hidden_size as f64
        * 4.0  // f32 bytes
        * num_layers as f64
        * 2.0; // attention + FFN per layer
    let saved_fraction = (1.0 - checkpoint_ratio as f64).max(0.0);
    (baseline * saved_fraction / (1024.0 * 1024.0)) as f32
}

// ─────────────────────────────────────────────────────────────────────────────
// MemoryOptimizer
// ─────────────────────────────────────────────────────────────────────────────

/// High-level memory optimizer.
///
/// Wraps a [`MemoryOptimizationConfig`] and exposes helpers for budget
/// computation, strategy recommendation, and activation compression.
pub struct MemoryOptimizer {
    /// The active optimisation configuration.
    pub config: MemoryOptimizationConfig,
}

impl MemoryOptimizer {
    /// Create a new optimizer with the given configuration.
    pub fn new(config: MemoryOptimizationConfig) -> Self {
        Self { config }
    }

    /// Compute a memory budget for a given model and hardware configuration.
    ///
    /// Uses a fixed reserved headroom of 512 MB; gradients are assumed equal
    /// in size to the model weights (also f32).
    pub fn compute_memory_budget(
        num_params: u64,
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
        num_layers: usize,
        total_gpu_mb: f32,
    ) -> MemoryBudget {
        let model_weights_mb = estimate_model_memory(num_params, 4); // f32
        let optimizer_states_mb = estimate_optimizer_state_memory(num_params, StatePrecision::Fp32);
        let activations_mb =
            estimate_activation_memory(batch_size, seq_len, hidden_size, num_layers, 0.0);
        let gradients_mb = model_weights_mb; // gradients are the same dtype as weights
        let reserved_mb = 512.0;

        MemoryBudget {
            total_gpu_memory_mb: total_gpu_mb,
            model_weights_mb,
            optimizer_states_mb,
            activations_mb,
            gradients_mb,
            reserved_mb,
        }
    }

    /// Recommend an optimisation strategy based on how far over budget the
    /// model is.
    ///
    /// | Situation                    | Recommendation                            |
    /// |------------------------------|-------------------------------------------|
    /// | Within budget                | `None` checkpointing, defaults            |
    /// | Slightly over (< 25 %)       | `EveryNthLayer { n: 2 }` checkpointing    |
    /// | Moderately over (< 50 %)     | + `Fp16Cast` activation compression        |
    /// | Severely over (≥ 50 %)       | + `offload_activations = true`            |
    pub fn recommend_strategy(
        budget: &MemoryBudget,
        total_gpu_mb: f32,
    ) -> MemoryOptimizationConfig {
        if budget.is_feasible() {
            return MemoryOptimizationConfig::default();
        }

        let total_used = budget.model_weights_mb
            + budget.optimizer_states_mb
            + budget.activations_mb
            + budget.gradients_mb
            + budget.reserved_mb;

        let over_ratio = if total_gpu_mb > 0.0 {
            (total_used - total_gpu_mb) / total_gpu_mb
        } else {
            1.0
        };

        if over_ratio < 0.25 {
            // Slightly over — gradient checkpointing every 2nd layer
            MemoryOptimizationConfig {
                gradient_checkpointing: SelectiveCheckpointStrategy::EveryNthLayer { n: 2 },
                ..MemoryOptimizationConfig::default()
            }
        } else if over_ratio < 0.50 {
            // Moderately over — add Fp16 activation compression
            MemoryOptimizationConfig {
                gradient_checkpointing: SelectiveCheckpointStrategy::EveryNthLayer { n: 2 },
                activation_compression: ActivationCompression::Fp16Cast,
                ..MemoryOptimizationConfig::default()
            }
        } else {
            // Severely over — also offload activations to CPU
            MemoryOptimizationConfig {
                gradient_checkpointing: SelectiveCheckpointStrategy::Full,
                activation_compression: ActivationCompression::Fp16Cast,
                offload_activations: true,
                ..MemoryOptimizationConfig::default()
            }
        }
    }

    /// Compress a slice of f32 activations using the specified method.
    ///
    /// The returned `Vec<u8>` can later be passed to [`Self::decompress_activation`].
    pub fn compress_activation(activation: &[f32], method: ActivationCompression) -> Vec<u8> {
        match method {
            ActivationCompression::None => {
                // Transmute f32 slice to byte slice.
                let mut out = Vec::with_capacity(activation.len() * 4);
                for &v in activation {
                    out.extend_from_slice(&v.to_le_bytes());
                }
                out
            },
            ActivationCompression::Fp16Cast => {
                // Convert each f32 to BF16 (brain float 16) — 2 bytes per element.
                let mut out = Vec::with_capacity(activation.len() * 2);
                for &v in activation {
                    let bf16_bits = f32_to_bf16_bits(v);
                    out.extend_from_slice(&bf16_bits.to_le_bytes());
                }
                out
            },
            ActivationCompression::Quantized { bits } => compress_quantized(activation, bits),
            ActivationCompression::Sparse { threshold } => compress_sparse(activation, threshold),
        }
    }

    /// Decompress a byte buffer produced by [`Self::compress_activation`] back
    /// to a `Vec<f32>`.
    ///
    /// `original_len` is the number of f32 elements in the original activation.
    pub fn decompress_activation(
        compressed: &[u8],
        original_len: usize,
        method: ActivationCompression,
    ) -> Result<Vec<f32>, MemOptError> {
        match method {
            ActivationCompression::None => decompress_none(compressed, original_len),
            ActivationCompression::Fp16Cast => decompress_fp16cast(compressed, original_len),
            ActivationCompression::Quantized { bits } => {
                decompress_quantized(compressed, original_len, bits)
            },
            ActivationCompression::Sparse { .. } => decompress_sparse(compressed, original_len),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Compression / decompression helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Convert an f32 value to a BF16 bit pattern (truncate the lower 16 mantissa bits).
#[inline]
fn f32_to_bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    // Round to nearest even by adding 0x7FFF + bit 16.
    let rounding_bias = 0x7FFF_u32 + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

/// Convert a BF16 bit pattern back to f32 (zero-extend lower 16 bits).
#[inline]
fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ── None ─────────────────────────────────────────────────────────────────────

fn decompress_none(compressed: &[u8], original_len: usize) -> Result<Vec<f32>, MemOptError> {
    let expected_bytes = original_len * 4;
    if compressed.len() < expected_bytes {
        return Err(MemOptError::InvalidCompressedLength {
            expected_min: expected_bytes,
            got: compressed.len(),
        });
    }
    let mut out = Vec::with_capacity(original_len);
    for chunk in compressed[..expected_bytes].chunks_exact(4) {
        let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
        out.push(f32::from_le_bytes(arr));
    }
    Ok(out)
}

// ── Fp16Cast ─────────────────────────────────────────────────────────────────

fn decompress_fp16cast(compressed: &[u8], original_len: usize) -> Result<Vec<f32>, MemOptError> {
    let expected_bytes = original_len * 2;
    if compressed.len() < expected_bytes {
        return Err(MemOptError::InvalidCompressedLength {
            expected_min: expected_bytes,
            got: compressed.len(),
        });
    }
    let mut out = Vec::with_capacity(original_len);
    for chunk in compressed[..expected_bytes].chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(bf16_bits_to_f32(bits));
    }
    Ok(out)
}

// ── Quantized ─────────────────────────────────────────────────────────────────

/// Simple uniform quantisation to `bits` bits per element.
///
/// Layout of the compressed buffer:
/// - 4 bytes: f32 min value (little-endian)
/// - 4 bytes: f32 max value (little-endian)
/// - 4 bytes: u32 element count (little-endian)
/// - packed quantised values: ceil(n * bits / 8) bytes
fn compress_quantized(activation: &[f32], bits: u8) -> Vec<u8> {
    if activation.is_empty() || bits == 0 {
        // Header only.
        let mut out = Vec::with_capacity(12);
        out.extend_from_slice(&0_f32.to_le_bytes());
        out.extend_from_slice(&0_f32.to_le_bytes());
        out.extend_from_slice(&(0_u32).to_le_bytes());
        return out;
    }

    let bits = bits.min(8);
    let min_val = activation.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = activation.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max_val - min_val;
    let levels = ((1u32 << bits) - 1) as f32;

    let n = activation.len();
    let packed_bytes = (n * bits as usize).div_ceil(8);
    let mut out = Vec::with_capacity(12 + packed_bytes);
    out.extend_from_slice(&min_val.to_le_bytes());
    out.extend_from_slice(&max_val.to_le_bytes());
    out.extend_from_slice(&(n as u32).to_le_bytes());

    let mut bit_buf: u64 = 0;
    let mut bits_in_buf: u32 = 0;
    for &v in activation {
        let q = if range < f32::EPSILON {
            0_u64
        } else {
            ((v - min_val) / range * levels).round() as u64
        };
        bit_buf |= q << bits_in_buf;
        bits_in_buf += bits as u32;
        while bits_in_buf >= 8 {
            out.push((bit_buf & 0xFF) as u8);
            bit_buf >>= 8;
            bits_in_buf -= 8;
        }
    }
    if bits_in_buf > 0 {
        out.push((bit_buf & 0xFF) as u8);
    }
    out
}

fn decompress_quantized(
    compressed: &[u8],
    original_len: usize,
    bits: u8,
) -> Result<Vec<f32>, MemOptError> {
    if bits == 0 || bits > 8 {
        return Err(MemOptError::UnsupportedBitWidth(bits));
    }
    if compressed.len() < 12 {
        return Err(MemOptError::InvalidCompressedLength {
            expected_min: 12,
            got: compressed.len(),
        });
    }
    let min_val = f32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]]);
    let max_val = f32::from_le_bytes([compressed[4], compressed[5], compressed[6], compressed[7]]);
    let stored_n =
        u32::from_le_bytes([compressed[8], compressed[9], compressed[10], compressed[11]]) as usize;

    if stored_n != original_len {
        return Err(MemOptError::LengthMismatch {
            expected: original_len,
            got: stored_n,
        });
    }

    let levels = ((1u32 << bits) - 1) as f32;
    let range = max_val - min_val;

    let payload = &compressed[12..];
    let mut out = Vec::with_capacity(original_len);
    let mask = (1u64 << bits) - 1;
    let mut bit_buf: u64 = 0;
    let mut bits_in_buf: u32 = 0;
    let mut byte_idx = 0;

    for _ in 0..original_len {
        while bits_in_buf < bits as u32 {
            if byte_idx < payload.len() {
                bit_buf |= (payload[byte_idx] as u64) << bits_in_buf;
                byte_idx += 1;
            }
            bits_in_buf += 8;
        }
        let q = bit_buf & mask;
        bit_buf >>= bits as u32;
        bits_in_buf -= bits as u32;

        let v = if levels < f32::EPSILON {
            min_val
        } else {
            min_val + (q as f32 / levels) * range
        };
        out.push(v);
    }
    Ok(out)
}

// ── Sparse ───────────────────────────────────────────────────────────────────

/// Sparse compression: store only (index: u32, value: f32) pairs for elements
/// where `|v| > threshold`.  Header = 4 bytes for element count.
fn compress_sparse(activation: &[f32], threshold: f32) -> Vec<u8> {
    let n = activation.len();
    // Header: original element count as u32
    let pairs: Vec<(u32, f32)> = activation
        .iter()
        .enumerate()
        .filter_map(
            |(i, &v)| {
                if v.abs() > threshold {
                    Some((i as u32, v))
                } else {
                    Option::None
                }
            },
        )
        .collect();

    let mut out = Vec::with_capacity(4 + 8 * pairs.len());
    out.extend_from_slice(&(n as u32).to_le_bytes());
    for (idx, val) in pairs {
        out.extend_from_slice(&idx.to_le_bytes());
        out.extend_from_slice(&val.to_le_bytes());
    }
    out
}

fn decompress_sparse(compressed: &[u8], original_len: usize) -> Result<Vec<f32>, MemOptError> {
    if compressed.len() < 4 {
        return Err(MemOptError::InvalidCompressedLength {
            expected_min: 4,
            got: compressed.len(),
        });
    }
    let stored_n =
        u32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]]) as usize;
    if stored_n != original_len {
        return Err(MemOptError::LengthMismatch {
            expected: original_len,
            got: stored_n,
        });
    }

    let mut out = vec![0.0_f32; original_len];
    let payload = &compressed[4..];
    let mut offset = 0;

    while offset + 8 <= payload.len() {
        let idx = u32::from_le_bytes([
            payload[offset],
            payload[offset + 1],
            payload[offset + 2],
            payload[offset + 3],
        ]) as usize;
        let val = f32::from_le_bytes([
            payload[offset + 4],
            payload[offset + 5],
            payload[offset + 6],
            payload[offset + 7],
        ]);
        if idx < original_len {
            out[idx] = val;
        }
        offset += 8;
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test 1: Model memory estimate ────────────────────────────────────

    #[test]
    fn test_model_memory_estimate() {
        // 1M params × 4 bytes = 4 MB
        let mb = estimate_model_memory(1_000_000, 4);
        let expected = 1_000_000.0 * 4.0 / (1024.0 * 1024.0);
        assert!((mb - expected).abs() < 0.01, "got {mb}");
    }

    // ── Test 2: Optimizer state memory (FP16 is half of FP32) ─────────────

    #[test]
    fn test_optimizer_state_memory_fp16_half_fp32() {
        let fp32_mb = estimate_optimizer_state_memory(1_000_000, StatePrecision::Fp32);
        let fp16_mb = estimate_optimizer_state_memory(1_000_000, StatePrecision::Fp16);
        assert!(
            (fp32_mb / fp16_mb - 2.0).abs() < 0.01,
            "fp32={fp32_mb} fp16={fp16_mb}"
        );
    }

    // ── Test 3: Optimizer state memory values ────────────────────────────

    #[test]
    fn test_optimizer_state_memory_values() {
        let n = 1_000_000_u64;
        let fp32 = estimate_optimizer_state_memory(n, StatePrecision::Fp32);
        // 1M * 8 bytes = 8 MB
        let expected = 8.0 * 1_000_000.0 / (1024.0 * 1024.0);
        assert!((fp32 - expected).abs() < 0.01);

        let int8 = estimate_optimizer_state_memory(n, StatePrecision::Int8Quantized);
        // 1M * 1 byte = ~0.954 MB
        assert!(int8 < 1.1 && int8 > 0.9, "int8={int8}");

        let bf16 = estimate_optimizer_state_memory(n, StatePrecision::Bf16);
        let fp16 = estimate_optimizer_state_memory(n, StatePrecision::Fp16);
        // Bf16 and Fp16 should be equal size
        assert!((bf16 - fp16).abs() < 0.001);
    }

    // ── Test 4: Activation memory with checkpointing ──────────────────────

    #[test]
    fn test_activation_memory_with_checkpointing() {
        let no_ckpt = estimate_activation_memory(4, 512, 768, 12, 0.0);
        let full_ckpt = estimate_activation_memory(4, 512, 768, 12, 1.0);
        // Full checkpointing should give 0 MB stored activations.
        assert!(
            full_ckpt.abs() < 1e-6,
            "full_ckpt should be 0, got {full_ckpt}"
        );
        // Half checkpointing should be half of no checkpointing.
        let half_ckpt = estimate_activation_memory(4, 512, 768, 12, 0.5);
        assert!((half_ckpt - no_ckpt * 0.5).abs() < 0.01);
    }

    // ── Test 5: is_feasible check ─────────────────────────────────────────

    #[test]
    fn test_is_feasible() {
        let feasible = MemoryBudget {
            total_gpu_memory_mb: 40_960.0,
            model_weights_mb: 7_000.0,
            optimizer_states_mb: 14_000.0,
            activations_mb: 4_000.0,
            gradients_mb: 7_000.0,
            reserved_mb: 512.0,
        };
        assert!(feasible.is_feasible());

        let infeasible = MemoryBudget {
            total_gpu_memory_mb: 10_000.0,
            model_weights_mb: 7_000.0,
            optimizer_states_mb: 14_000.0,
            activations_mb: 4_000.0,
            gradients_mb: 7_000.0,
            reserved_mb: 512.0,
        };
        assert!(!infeasible.is_feasible());
    }

    // ── Test 6: available_for_activations_mb ─────────────────────────────

    #[test]
    fn test_available_for_activations_mb() {
        let budget = MemoryBudget {
            total_gpu_memory_mb: 40_960.0,
            model_weights_mb: 7_000.0,
            optimizer_states_mb: 14_000.0,
            activations_mb: 4_000.0,
            gradients_mb: 7_000.0,
            reserved_mb: 512.0,
        };
        let avail = budget.available_for_activations_mb();
        // 40960 - (7000 + 14000 + 7000 + 512) = 12448
        assert!((avail - 12_448.0).abs() < 0.01, "got {avail}");
    }

    // ── Test 7: recommend_strategy — feasible budget → default ───────────

    #[test]
    fn test_recommend_strategy_feasible() {
        let budget = MemoryBudget {
            total_gpu_memory_mb: 80_000.0,
            model_weights_mb: 1_000.0,
            optimizer_states_mb: 2_000.0,
            activations_mb: 500.0,
            gradients_mb: 1_000.0,
            reserved_mb: 512.0,
        };
        let config = MemoryOptimizer::recommend_strategy(&budget, 80_000.0);
        assert_eq!(
            config.gradient_checkpointing,
            SelectiveCheckpointStrategy::None
        );
        assert!(!config.offload_activations);
    }

    // ── Test 8: recommend_strategy — slightly over → EveryNthLayer(2) ────

    #[test]
    fn test_recommend_strategy_slightly_over() {
        // Slightly over: total_used = 11_512, total_gpu = 10_000 → over by ~15%
        let budget = MemoryBudget {
            total_gpu_memory_mb: 10_000.0,
            model_weights_mb: 4_000.0,
            optimizer_states_mb: 5_000.0,
            activations_mb: 2_000.0,
            gradients_mb: 512.0,
            reserved_mb: 0.0,
        };
        let config = MemoryOptimizer::recommend_strategy(&budget, 10_000.0);
        assert_eq!(
            config.gradient_checkpointing,
            SelectiveCheckpointStrategy::EveryNthLayer { n: 2 }
        );
        assert!(!config.offload_activations);
    }

    // ── Test 9: recommend_strategy — severely over → offload ─────────────

    #[test]
    fn test_recommend_strategy_severely_over() {
        // Severely over: total_used = 20_000, total_gpu = 10_000 → over by 100%
        let budget = MemoryBudget {
            total_gpu_memory_mb: 10_000.0,
            model_weights_mb: 8_000.0,
            optimizer_states_mb: 8_000.0,
            activations_mb: 4_000.0,
            gradients_mb: 0.0,
            reserved_mb: 0.0,
        };
        let config = MemoryOptimizer::recommend_strategy(&budget, 10_000.0);
        assert!(config.offload_activations);
    }

    // ── Test 10: compress/decompress Fp16Cast round-trip ──────────────────

    #[test]
    fn test_compress_decompress_fp16cast_roundtrip() {
        let original: Vec<f32> = vec![1.0, -2.5, 0.0, std::f32::consts::PI, 100.0, -0.001];
        let compressed =
            MemoryOptimizer::compress_activation(&original, ActivationCompression::Fp16Cast);
        assert_eq!(compressed.len(), original.len() * 2);
        let decompressed = MemoryOptimizer::decompress_activation(
            &compressed,
            original.len(),
            ActivationCompression::Fp16Cast,
        )
        .expect("decompression failed");
        assert_eq!(decompressed.len(), original.len());
        for (orig, decomp) in original.iter().zip(decompressed.iter()) {
            // BF16 has ~2 decimal digits of precision
            let rel_err = if orig.abs() > 1e-6 {
                (orig - decomp).abs() / orig.abs()
            } else {
                (orig - decomp).abs()
            };
            assert!(
                rel_err < 0.02,
                "orig={orig} decomp={decomp} rel_err={rel_err}"
            );
        }
    }

    // ── Test 11: compress None is f32 bytes ──────────────────────────────

    #[test]
    fn test_compress_none_is_f32_bytes() {
        let original: Vec<f32> = vec![1.0_f32, 2.0, 3.0];
        let compressed =
            MemoryOptimizer::compress_activation(&original, ActivationCompression::None);
        assert_eq!(compressed.len(), 12); // 3 × 4 bytes

        let decompressed =
            MemoryOptimizer::decompress_activation(&compressed, 3, ActivationCompression::None)
                .expect("ok");
        assert_eq!(decompressed, original);
    }

    // ── Test 12: compress/decompress quantized round-trip ─────────────────

    #[test]
    fn test_compress_decompress_quantized_roundtrip() {
        let original: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let method = ActivationCompression::Quantized { bits: 8 };
        let compressed = MemoryOptimizer::compress_activation(&original, method);
        let decompressed =
            MemoryOptimizer::decompress_activation(&compressed, original.len(), method)
                .expect("ok");
        assert_eq!(decompressed.len(), original.len());
        for (o, d) in original.iter().zip(decompressed.iter()) {
            // 8-bit quantisation → small absolute error
            assert!((o - d).abs() < 0.1, "orig={o} decomp={d}");
        }
    }

    // ── Test 13: compress/decompress sparse round-trip ────────────────────

    #[test]
    fn test_compress_decompress_sparse_roundtrip() {
        let original: Vec<f32> = vec![0.0, 0.0, 3.0, 0.0, -5.0, 0.01, 0.0];
        let method = ActivationCompression::Sparse { threshold: 0.05 };
        let compressed = MemoryOptimizer::compress_activation(&original, method);
        let decompressed =
            MemoryOptimizer::decompress_activation(&compressed, original.len(), method)
                .expect("ok");
        assert_eq!(decompressed.len(), original.len());
        // Values above threshold should be preserved.
        assert!((decompressed[2] - 3.0).abs() < 1e-5);
        assert!((decompressed[4] - (-5.0)).abs() < 1e-5);
        // Small value (0.01) is below threshold (0.05) — should be zeroed.
        assert_eq!(decompressed[5], 0.0);
    }

    // ── Test 14: empty activation compression ────────────────────────────

    #[test]
    fn test_empty_activation_compression() {
        let empty: Vec<f32> = vec![];
        for method in [
            ActivationCompression::None,
            ActivationCompression::Fp16Cast,
            ActivationCompression::Quantized { bits: 4 },
            ActivationCompression::Sparse { threshold: 0.0 },
        ] {
            let compressed = MemoryOptimizer::compress_activation(&empty, method);
            let decompressed =
                MemoryOptimizer::decompress_activation(&compressed, 0, method).unwrap_or_default();
            assert!(
                decompressed.is_empty(),
                "method {method:?} produced non-empty output for empty input"
            );
        }
    }

    // ── Test 15: multi-config validation ─────────────────────────────────

    #[test]
    fn test_multi_config_validation() {
        let config = MemoryOptimizationConfig {
            offload_activations: true,
            offload_optimizer_states: true,
            use_memory_efficient_attention: true,
            gradient_checkpointing: SelectiveCheckpointStrategy::Full,
            optimizer_state_precision: StatePrecision::Int8Quantized,
            activation_compression: ActivationCompression::Fp16Cast,
            max_gpu_memory_mb: Some(24_576.0),
        };
        assert!(config.offload_activations);
        assert!(config.offload_optimizer_states);
        assert!(config.use_memory_efficient_attention);
        assert_eq!(
            config.gradient_checkpointing,
            SelectiveCheckpointStrategy::Full
        );
        assert_eq!(
            config.optimizer_state_precision,
            StatePrecision::Int8Quantized
        );
        assert_eq!(config.max_gpu_memory_mb, Some(24_576.0));
    }

    // ── Test 16: compute_memory_budget basic sanity ───────────────────────

    #[test]
    fn test_compute_memory_budget_sanity() {
        let budget = MemoryOptimizer::compute_memory_budget(
            7_000_000_000, // 7B params
            1,             // batch size
            512,           // seq len
            4096,          // hidden size
            32,            // layers
            80_000.0,      // 80 GB
        );
        assert!(budget.model_weights_mb > 0.0);
        assert!(budget.optimizer_states_mb > 0.0);
        assert!(budget.activations_mb >= 0.0);
        assert_eq!(budget.gradients_mb, budget.model_weights_mb);
    }
}
