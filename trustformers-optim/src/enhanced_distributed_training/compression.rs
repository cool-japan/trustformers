//! Gradient compression codecs for distributed training.
//!
//! Every codec operates on the tensor's **real f32 values** (via
//! [`Tensor::to_vec_f32`]). An earlier revision read gradients through
//! `Tensor::to_vec_u8()`, which is an `f32 as u8` *saturating value cast* — for
//! gradients in the usual `[-0.1, 0.1]` range that cast maps every element to
//! `0u8`, so compress→decompress silently returned an all-zero gradient. The
//! round-trip tests at the bottom of this file pin the correct behaviour.
//!
//! Codecs:
//!
//! * [`CompressionType::TopK`] — keep the `k` largest-magnitude entries.
//! * [`CompressionType::RandomSparsification`] — keep a random subset.
//! * [`CompressionType::Quantization`] — uniform affine quantization to
//!   `2..=8` bits with a per-tensor scale/offset.
//! * [`CompressionType::PowerSGD`] — genuine rank-`r` factorization by one
//!   power iteration with Gram-Schmidt orthonormalization.
//! * [`CompressionType::OneBitSGD`] — sign vector plus the gradient's L2 norm.
//! * [`CompressionType::Adaptive`] — picks a top-k sparsity from the gradient's
//!   variance.
//!
//! All lossy codecs support *error feedback*: the residual left behind by the
//! codec is accumulated and added back into the next step's gradient, which is
//! what makes aggressive compression converge in practice.

use super::{CompressionConfig, CompressionType};
use std::collections::HashMap;
use std::time::Instant;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;

/// Advanced gradient compression with multiple algorithms
pub struct GradientCompressor {
    config: CompressionConfig,
    error_feedback_state: HashMap<String, Tensor>,
    compression_stats: CompressionStats,
}

/// Aggregate statistics over all compressions performed so far.
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// Total bytes produced by the codecs.
    pub total_compressed_bytes: usize,
    /// Total bytes the same gradients would have occupied dense (f32).
    pub total_uncompressed_bytes: usize,
    /// `total_compressed_bytes / total_uncompressed_bytes`.
    pub average_compression_ratio: f32,
    /// Wall-clock time of the last compression pass.
    pub compression_time_ms: f32,
    /// Wall-clock time of the last decompression pass.
    pub decompression_time_ms: f32,
}

impl Default for CompressionStats {
    fn default() -> Self {
        Self {
            total_compressed_bytes: 0,
            total_uncompressed_bytes: 0,
            average_compression_ratio: 1.0,
            compression_time_ms: 0.0,
            decompression_time_ms: 0.0,
        }
    }
}

/// Deterministic 64-bit LCG, used where every rank must draw the *same*
/// sequence (PowerSGD's shared sketch matrix) without pulling in a seeded RNG
/// dependency.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        // Odd seed keeps the generator full-period.
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        // Numerical Recipes constants.
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }

    /// Uniform in `[-1, 1)`.
    fn next_signed_unit(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as f32; // 24 bits
        bits / 8_388_608.0 - 1.0
    }
}

impl GradientCompressor {
    /// Build a compressor for `config`.
    pub fn new(config: CompressionConfig) -> Self {
        Self {
            config,
            error_feedback_state: HashMap::new(),
            compression_stats: CompressionStats::default(),
        }
    }

    /// Compress every gradient in `gradients`.
    ///
    /// With error feedback enabled the previous step's residual is added to the
    /// gradient before compression and the new residual is retained, so the
    /// information a lossy codec discards is not lost but deferred.
    pub fn compress_gradients(
        &mut self,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, CompressedGradient>> {
        if !self.config.enabled {
            // No compression - convert to "compressed" format for API consistency
            return Ok(gradients
                .iter()
                .map(|(name, grad)| (name.clone(), CompressedGradient::uncompressed(grad.clone())))
                .collect());
        }

        let start_time = Instant::now();
        let mut compressed = HashMap::new();
        let mut compressed_bytes = 0usize;
        let mut dense_bytes = 0usize;

        // Deterministic ordering so repeated runs (and peer ranks) agree.
        let mut names: Vec<&String> = gradients.keys().collect();
        names.sort();

        for name in names {
            let gradient = match gradients.get(name) {
                Some(gradient) => gradient,
                None => continue,
            };

            let corrected = if self.config.error_feedback {
                match self.error_feedback_state.get(name) {
                    Some(residual) => gradient.add(residual)?,
                    None => gradient.clone(),
                }
            } else {
                gradient.clone()
            };

            let compressed_grad = self.compress_one(&corrected)?;

            if self.config.error_feedback {
                let reconstructed = compressed_grad.decompress()?;
                let residual = corrected.sub(&reconstructed)?;
                self.error_feedback_state.insert(name.clone(), residual);
            }

            compressed_bytes += compressed_grad.size_bytes();
            dense_bytes += gradient.len() * std::mem::size_of::<f32>();
            compressed.insert(name.clone(), compressed_grad);
        }

        self.compression_stats.compression_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
        self.compression_stats.total_compressed_bytes += compressed_bytes;
        self.compression_stats.total_uncompressed_bytes += dense_bytes;
        if self.compression_stats.total_uncompressed_bytes > 0 {
            self.compression_stats.average_compression_ratio =
                self.compression_stats.total_compressed_bytes as f32
                    / self.compression_stats.total_uncompressed_bytes as f32;
        }

        Ok(compressed)
    }

    /// Decompress a batch of gradients back into dense tensors.
    pub fn decompress_gradients(
        &mut self,
        compressed: &HashMap<String, CompressedGradient>,
    ) -> Result<HashMap<String, Tensor>> {
        let start_time = Instant::now();
        let mut dense = HashMap::with_capacity(compressed.len());
        for (name, gradient) in compressed {
            dense.insert(name.clone(), gradient.decompress()?);
        }
        self.compression_stats.decompression_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
        Ok(dense)
    }

    fn compress_one(&self, gradient: &Tensor) -> Result<CompressedGradient> {
        match &self.config.algorithm {
            CompressionType::None => Ok(CompressedGradient::uncompressed(gradient.clone())),
            CompressionType::TopK { k } => self.compress_topk(gradient, *k),
            CompressionType::RandomSparsification { ratio } => {
                self.compress_random(gradient, *ratio)
            },
            CompressionType::Quantization { bits } => self.compress_quantization(gradient, *bits),
            CompressionType::PowerSGD { rank } => self.compress_powersgd(gradient, *rank),
            CompressionType::OneBitSGD => self.compress_onebit(gradient),
            CompressionType::Adaptive => self.compress_adaptive(gradient),
        }
    }

    /// Top-K sparsification: retain the `k` entries with the largest magnitude.
    pub fn compress_topk(&self, gradient: &Tensor, k: usize) -> Result<CompressedGradient> {
        let values = gradient.to_vec_f32()?;
        let keep = k.min(values.len());

        if keep == 0 || values.is_empty() {
            return Ok(CompressedGradient {
                compression_type: CompressionType::TopK { k },
                compressed_data: CompressedData::Sparse {
                    indices: Vec::new(),
                    values: Vec::new(),
                },
                original_shape: gradient.shape().to_vec(),
                compression_ratio: 0.0,
            });
        }

        let mut ranked: Vec<(usize, f32)> =
            values.iter().enumerate().map(|(index, value)| (index, value.abs())).collect();
        ranked.select_nth_unstable_by(keep - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))
        });
        ranked.truncate(keep);
        ranked.sort_by_key(|(index, _)| *index);

        let indices: Vec<usize> = ranked.iter().map(|(index, _)| *index).collect();
        let kept: Vec<f32> = indices.iter().map(|&index| values[index]).collect();

        Ok(CompressedGradient {
            compression_type: CompressionType::TopK { k },
            compressed_data: CompressedData::Sparse {
                indices,
                values: kept,
            },
            original_shape: gradient.shape().to_vec(),
            compression_ratio: keep as f32 / values.len() as f32,
        })
    }

    /// Random-k sparsification: retain a uniformly random subset of entries.
    pub fn compress_random(&self, gradient: &Tensor, ratio: f32) -> Result<CompressedGradient> {
        if !(ratio.is_finite() && ratio > 0.0) {
            return Err(TrustformersError::invalid_input(format!(
                "RandomSparsification ratio must be finite and > 0, got {ratio}"
            )));
        }

        let values = gradient.to_vec_f32()?;
        let clamped = ratio.min(1.0);
        let keep = ((values.len() as f32) * clamped).round() as usize;
        let keep = keep.clamp(usize::from(!values.is_empty()), values.len());

        use scirs2_core::random::*; // SciRS2 Integration Policy
        let mut indices: Vec<usize> = (0..values.len()).collect();
        let mut rng = thread_rng();
        indices.shuffle(rng.rng_mut());
        indices.truncate(keep);
        indices.sort_unstable(); // Sort for better cache locality

        let kept: Vec<f32> = indices.iter().map(|&index| values[index]).collect();

        Ok(CompressedGradient {
            compression_type: CompressionType::RandomSparsification { ratio },
            compressed_data: CompressedData::Sparse {
                indices,
                values: kept,
            },
            original_shape: gradient.shape().to_vec(),
            compression_ratio: if values.is_empty() {
                0.0
            } else {
                keep as f32 / values.len() as f32
            },
        })
    }

    /// Uniform affine quantization to `bits` bits.
    ///
    /// The reconstruction error is bounded by half a quantization step, and a
    /// constant tensor (`max == min`) round-trips exactly.
    pub fn compress_quantization(&self, gradient: &Tensor, bits: u8) -> Result<CompressedGradient> {
        if !(2..=8).contains(&bits) {
            return Err(TrustformersError::invalid_input(format!(
                "Quantization bits must be in 2..=8 (the payload is stored as u8), got {bits}"
            )));
        }

        let values = gradient.to_vec_f32()?;
        let levels = 1u32 << bits;

        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &value in &values {
            min_val = min_val.min(value);
            max_val = max_val.max(value);
        }
        if values.is_empty() {
            min_val = 0.0;
            max_val = 0.0;
        }

        let span = max_val - min_val;
        // A degenerate span (constant tensor) would divide by zero; store every
        // element at level 0 so dequantization returns `min_val` exactly.
        let scale = if span > 0.0 && span.is_finite() { span / (levels - 1) as f32 } else { 0.0 };

        let quantized: Vec<u8> = if scale > 0.0 {
            values
                .iter()
                .map(|&value| {
                    ((value - min_val) / scale).round().clamp(0.0, (levels - 1) as f32) as u8
                })
                .collect()
        } else {
            vec![0u8; values.len()]
        };

        Ok(CompressedGradient {
            compression_type: CompressionType::Quantization { bits },
            compressed_data: CompressedData::Quantized {
                data: quantized,
                min_val,
                scale,
                levels,
            },
            original_shape: gradient.shape().to_vec(),
            compression_ratio: bits as f32 / 32.0,
        })
    }

    /// PowerSGD rank-`r` factorization.
    ///
    /// The gradient is viewed as an `m x n` matrix `M` (leading dimension times
    /// the product of the rest). One power iteration is performed against a
    /// deterministic sketch `Q0`:
    ///
    /// 1. `P = M · Q0`            (`m x r`)
    /// 2. orthonormalize `P`      (modified Gram-Schmidt)
    /// 3. `Q = Mᵀ · P`            (`n x r`)
    ///
    /// and `P`, `Q` are transmitted. The receiver reconstructs `M̂ = P · Qᵀ`,
    /// the best rank-`r` approximation in the subspace spanned by `P`. A 1-D
    /// gradient has no meaningful matrix structure, so it is sent uncompressed
    /// (this matches the reference PowerSGD implementation).
    pub fn compress_powersgd(&self, gradient: &Tensor, rank: usize) -> Result<CompressedGradient> {
        let shape = gradient.shape().to_vec();
        let values = gradient.to_vec_f32()?;

        if rank == 0 {
            return Err(TrustformersError::invalid_input(
                "PowerSGD rank must be >= 1".to_string(),
            ));
        }
        if shape.len() < 2 || values.is_empty() {
            return Ok(CompressedGradient::uncompressed(gradient.clone()));
        }

        let rows = shape[0];
        let cols = values.len() / rows.max(1);
        if rows == 0 || cols == 0 || rows * cols != values.len() {
            return Ok(CompressedGradient::uncompressed(gradient.clone()));
        }

        let effective_rank = rank.min(rows).min(cols);
        // Below this size the two factors cost more than the dense matrix.
        if effective_rank * (rows + cols) >= rows * cols {
            return Ok(CompressedGradient::uncompressed(gradient.clone()));
        }

        // Deterministic sketch: identical on every rank, so the factorizations
        // produced by different workers live in comparable subspaces.
        let mut rng = Lcg::new((rows as u64) << 32 | (cols as u64) ^ effective_rank as u64);
        let mut q0 = vec![0.0f32; cols * effective_rank];
        for slot in q0.iter_mut() {
            *slot = rng.next_signed_unit();
        }

        // P = M * Q0
        let mut p = vec![0.0f32; rows * effective_rank];
        for row in 0..rows {
            for component in 0..effective_rank {
                let mut accumulator = 0.0f32;
                for col in 0..cols {
                    accumulator += values[row * cols + col] * q0[col * effective_rank + component];
                }
                p[row * effective_rank + component] = accumulator;
            }
        }

        orthonormalize_columns(&mut p, rows, effective_rank);

        // Q = M^T * P
        let mut q = vec![0.0f32; cols * effective_rank];
        for col in 0..cols {
            for component in 0..effective_rank {
                let mut accumulator = 0.0f32;
                for row in 0..rows {
                    accumulator += values[row * cols + col] * p[row * effective_rank + component];
                }
                q[col * effective_rank + component] = accumulator;
            }
        }

        let compressed_elements = effective_rank * (rows + cols);

        Ok(CompressedGradient {
            compression_type: CompressionType::PowerSGD { rank },
            compressed_data: CompressedData::LowRank {
                p,
                q,
                rows,
                cols,
                rank: effective_rank,
            },
            original_shape: shape,
            compression_ratio: compressed_elements as f32 / values.len() as f32,
        })
    }

    /// 1-bit SGD: transmit the sign of each entry plus the gradient's L2 norm.
    pub fn compress_onebit(&self, gradient: &Tensor) -> Result<CompressedGradient> {
        let values = gradient.to_vec_f32()?;
        let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        let signs: Vec<bool> = values.iter().map(|&value| value >= 0.0).collect();
        let packed_signs = pack_bits(&signs);

        Ok(CompressedGradient {
            compression_type: CompressionType::OneBitSGD,
            compressed_data: CompressedData::OneBit {
                signs: packed_signs,
                norm,
                elements: values.len(),
            },
            original_shape: gradient.shape().to_vec(),
            compression_ratio: 1.0 / 32.0,
        })
    }

    /// Choose a top-k sparsity from the gradient's variance.
    pub fn compress_adaptive(&self, gradient: &Tensor) -> Result<CompressedGradient> {
        let values = gradient.to_vec_f32()?;
        let variance = calculate_variance(&values);

        if variance < self.config.adaptive_threshold {
            // Low variance: the gradient is close to uniform, so a few entries
            // carry most of the signal.
            self.compress_topk(gradient, values.len().div_ceil(20)) // 5% sparsity
        } else {
            self.compress_topk(gradient, values.len().div_ceil(5)) // 20% sparsity
        }
    }

    /// Residual retained for a parameter under error feedback, if any.
    pub fn error_feedback_residual(&self, name: &str) -> Option<&Tensor> {
        self.error_feedback_state.get(name)
    }

    /// Aggregate compression statistics.
    pub fn get_compression_stats(&self) -> &CompressionStats {
        &self.compression_stats
    }
}

/// Modified Gram-Schmidt orthonormalization of the `columns` columns of a
/// row-major `rows x columns` matrix.
fn orthonormalize_columns(matrix: &mut [f32], rows: usize, columns: usize) {
    for column in 0..columns {
        // Subtract the projection onto every previously orthonormalized column.
        for previous in 0..column {
            let mut projection = 0.0f32;
            for row in 0..rows {
                projection += matrix[row * columns + column] * matrix[row * columns + previous];
            }
            for row in 0..rows {
                matrix[row * columns + column] -= projection * matrix[row * columns + previous];
            }
        }

        let mut norm = 0.0f32;
        for row in 0..rows {
            let value = matrix[row * columns + column];
            norm += value * value;
        }
        let norm = norm.sqrt();

        if norm > 1e-8 {
            let inverse = 1.0 / norm;
            for row in 0..rows {
                matrix[row * columns + column] *= inverse;
            }
        } else {
            // Degenerate column: replace with a canonical basis vector so the
            // factor stays full rank instead of collapsing to zero.
            for row in 0..rows {
                matrix[row * columns + column] = f32::from(row == column % rows.max(1));
            }
        }
    }
}

fn pack_bits(bits: &[bool]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(bits.len().div_ceil(8));
    for chunk in bits.chunks(8) {
        let mut byte = 0u8;
        for (index, &bit) in chunk.iter().enumerate() {
            if bit {
                byte |= 1 << index;
            }
        }
        packed.push(byte);
    }
    packed
}

fn calculate_variance(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    data.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / data.len() as f32
}

/// Compressed gradient representation
#[derive(Debug, Clone)]
pub struct CompressedGradient {
    /// Codec that produced this payload.
    pub compression_type: CompressionType,
    /// The payload itself.
    pub compressed_data: CompressedData,
    /// Shape of the dense gradient this payload reconstructs.
    pub original_shape: Vec<usize>,
    /// Compressed element count divided by dense element count.
    pub compression_ratio: f32,
}

/// Codec-specific compressed payloads.
#[derive(Debug, Clone)]
pub enum CompressedData {
    /// Verbatim tensor (no compression).
    Uncompressed(Tensor),
    /// Sparse payload: `values[i]` belongs at flat index `indices[i]`.
    Sparse {
        /// Flat indices of the retained entries, ascending.
        indices: Vec<usize>,
        /// Retained values.
        values: Vec<f32>,
    },
    /// Uniform affine quantization: `value = min_val + level * scale`.
    Quantized {
        /// Quantization levels, one per element.
        data: Vec<u8>,
        /// Value mapped to level 0.
        min_val: f32,
        /// Width of one quantization level (`0.0` for a constant tensor).
        scale: f32,
        /// Number of levels (`2^bits`).
        levels: u32,
    },
    /// Rank-`rank` factorization `M ≈ P · Qᵀ`, both stored row-major.
    LowRank {
        /// `rows x rank` orthonormal factor.
        p: Vec<f32>,
        /// `cols x rank` factor.
        q: Vec<f32>,
        /// Row count of the reconstructed matrix.
        rows: usize,
        /// Column count of the reconstructed matrix.
        cols: usize,
        /// Factorization rank.
        rank: usize,
    },
    /// Sign vector plus the gradient's L2 norm.
    OneBit {
        /// Packed sign bits, LSB first.
        signs: Vec<u8>,
        /// L2 norm of the original gradient.
        norm: f32,
        /// Number of elements encoded (the packing rounds up to whole bytes).
        elements: usize,
    },
}

impl CompressedGradient {
    /// Wrap a tensor without compressing it.
    pub fn uncompressed(tensor: Tensor) -> Self {
        let shape = tensor.shape().to_vec();
        Self {
            compression_type: CompressionType::None,
            compressed_data: CompressedData::Uncompressed(tensor),
            original_shape: shape,
            compression_ratio: 1.0,
        }
    }

    /// Reconstruct the dense gradient.
    pub fn decompress(&self) -> Result<Tensor> {
        match &self.compressed_data {
            CompressedData::Uncompressed(tensor) => Ok(tensor.clone()),
            CompressedData::Sparse { indices, values } => {
                let total_elements: usize = self.original_shape.iter().product();
                let mut data = vec![0.0f32; total_elements];
                for (&index, &value) in indices.iter().zip(values.iter()) {
                    if index >= data.len() {
                        return Err(TrustformersError::invalid_input(format!(
                            "sparse index {index} out of range for {total_elements} elements"
                        )));
                    }
                    data[index] = value;
                }
                Tensor::from_slice(&data, &self.original_shape)
            },
            CompressedData::Quantized {
                data,
                min_val,
                scale,
                ..
            } => {
                let dequantized: Vec<f32> =
                    data.iter().map(|&level| min_val + level as f32 * scale).collect();
                Tensor::from_slice(&dequantized, &self.original_shape)
            },
            CompressedData::LowRank {
                p,
                q,
                rows,
                cols,
                rank,
            } => {
                let mut data = vec![0.0f32; rows * cols];
                for row in 0..*rows {
                    for col in 0..*cols {
                        let mut accumulator = 0.0f32;
                        for component in 0..*rank {
                            accumulator += p[row * rank + component] * q[col * rank + component];
                        }
                        data[row * cols + col] = accumulator;
                    }
                }
                Tensor::from_slice(&data, &self.original_shape)
            },
            CompressedData::OneBit {
                signs,
                norm,
                elements,
            } => {
                let scale = if *elements == 0 { 0.0 } else { norm / (*elements as f32).sqrt() };
                let mut data = Vec::with_capacity(*elements);
                'outer: for &byte in signs {
                    for bit in 0..8 {
                        if data.len() >= *elements {
                            break 'outer;
                        }
                        let sign = if (byte >> bit) & 1 == 1 { 1.0 } else { -1.0 };
                        data.push(sign * scale);
                    }
                }
                data.resize(*elements, 0.0);
                Tensor::from_slice(&data, &self.original_shape)
            },
        }
    }

    /// Wire size of this payload in bytes.
    pub fn size_bytes(&self) -> usize {
        match &self.compressed_data {
            CompressedData::Uncompressed(tensor) => tensor.memory_usage(),
            CompressedData::Sparse { indices, values } => {
                indices.len() * std::mem::size_of::<u32>()
                    + values.len() * std::mem::size_of::<f32>()
            },
            CompressedData::Quantized { data, .. } => {
                data.len() * std::mem::size_of::<u8>()
                    + 2 * std::mem::size_of::<f32>()
                    + std::mem::size_of::<u32>()
            },
            CompressedData::LowRank { p, q, .. } => {
                (p.len() + q.len()) * std::mem::size_of::<f32>() + 3 * std::mem::size_of::<u32>()
            },
            CompressedData::OneBit { signs, .. } => {
                signs.len() * std::mem::size_of::<u8>()
                    + std::mem::size_of::<f32>()
                    + std::mem::size_of::<u32>()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(algorithm: CompressionType) -> CompressionConfig {
        CompressionConfig {
            enabled: true,
            algorithm,
            target_ratio: 0.1,
            error_feedback: false,
            adaptive_threshold: 0.01,
        }
    }

    /// Gradient values in the range real gradients actually occupy. The old
    /// `to_vec_u8()` path mapped every one of these to `0u8`.
    fn small_gradient() -> Tensor {
        Tensor::from_slice(&[0.05, -0.02, 0.13, -0.4, 0.0, 0.31, -0.11, 0.07], &[8])
            .expect("tensor must build in test")
    }

    #[test]
    fn topk_keeps_the_largest_magnitudes_and_zeros_the_rest() {
        let compressor = GradientCompressor::new(config(CompressionType::TopK { k: 3 }));
        let gradient = small_gradient();
        let compressed = compressor.compress_topk(&gradient, 3).expect("compress in test");
        let restored = compressed
            .decompress()
            .expect("decompress in test")
            .to_vec_f32()
            .expect("tensor read in test");

        // |−0.4| > |0.31| > |0.13| are the three largest.
        assert_eq!(restored, vec![0.0, 0.0, 0.13, -0.4, 0.0, 0.31, 0.0, 0.0]);
    }

    #[test]
    fn topk_round_trip_is_not_all_zero() {
        // Direct regression against the `to_vec_u8()` bug: every value here is
        // < 1.0 and would have cast to 0u8.
        let compressor = GradientCompressor::new(config(CompressionType::TopK { k: 8 }));
        let gradient = small_gradient();
        let compressed = compressor.compress_topk(&gradient, 8).expect("compress in test");
        let restored =
            compressed.decompress().expect("decompress in test").to_vec_f32().expect("read");

        assert_eq!(restored, gradient.to_vec_f32().expect("read"));
        assert!(restored.iter().any(|value| *value != 0.0));
    }

    #[test]
    fn random_sparsification_retains_the_requested_fraction_verbatim() {
        let compressor =
            GradientCompressor::new(config(CompressionType::RandomSparsification { ratio: 0.5 }));
        let gradient = small_gradient();
        let original = gradient.to_vec_f32().expect("read");
        let compressed = compressor.compress_random(&gradient, 0.5).expect("compress in test");

        let CompressedData::Sparse { indices, values } = &compressed.compressed_data else {
            panic!("random sparsification must produce a sparse payload");
        };
        assert_eq!(indices.len(), 4);
        for (&index, &value) in indices.iter().zip(values) {
            assert_eq!(value, original[index], "retained values must be verbatim");
        }

        let restored =
            compressed.decompress().expect("decompress in test").to_vec_f32().expect("read");
        assert_eq!(restored.len(), original.len());
    }

    #[test]
    fn quantization_round_trip_is_within_half_a_step() {
        let compressor = GradientCompressor::new(config(CompressionType::Quantization { bits: 8 }));
        let gradient = small_gradient();
        let original = gradient.to_vec_f32().expect("read");
        let compressed = compressor.compress_quantization(&gradient, 8).expect("compress in test");

        let CompressedData::Quantized { scale, .. } = compressed.compressed_data else {
            panic!("quantization must produce a quantized payload");
        };
        let restored =
            compressed.decompress().expect("decompress in test").to_vec_f32().expect("read");

        for (got, want) in restored.iter().zip(&original) {
            assert!(
                (got - want).abs() <= scale * 0.5 + 1e-6,
                "quantization error {} exceeded half a step {}",
                (got - want).abs(),
                scale * 0.5
            );
        }
        // And the reconstruction is genuinely non-trivial.
        assert!(restored.iter().any(|value| *value != 0.0));
    }

    #[test]
    fn quantization_handles_a_constant_gradient_without_nan() {
        let compressor = GradientCompressor::new(config(CompressionType::Quantization { bits: 4 }));
        let gradient = Tensor::from_slice(&[0.25; 6], &[6]).expect("tensor must build in test");
        let compressed = compressor.compress_quantization(&gradient, 4).expect("compress in test");
        let restored =
            compressed.decompress().expect("decompress in test").to_vec_f32().expect("read");

        assert_eq!(restored, vec![0.25f32; 6]);
        assert!(restored.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn quantization_rejects_unsupported_bit_widths() {
        let compressor =
            GradientCompressor::new(config(CompressionType::Quantization { bits: 16 }));
        assert!(compressor.compress_quantization(&small_gradient(), 16).is_err());
        assert!(compressor.compress_quantization(&small_gradient(), 1).is_err());
    }

    #[test]
    fn onebit_round_trip_preserves_signs_and_norm_scale() {
        let compressor = GradientCompressor::new(config(CompressionType::OneBitSGD));
        let gradient = small_gradient();
        let original = gradient.to_vec_f32().expect("read");
        let compressed = compressor.compress_onebit(&gradient).expect("compress in test");
        let restored =
            compressed.decompress().expect("decompress in test").to_vec_f32().expect("read");

        assert_eq!(restored.len(), original.len());
        let norm = original.iter().map(|v| v * v).sum::<f32>().sqrt();
        let expected_scale = norm / (original.len() as f32).sqrt();
        for (got, want) in restored.iter().zip(&original) {
            let expected_sign = if *want >= 0.0 { 1.0 } else { -1.0 };
            approx::assert_relative_eq!(*got, expected_sign * expected_scale, epsilon = 1e-6);
        }
    }

    #[test]
    fn powersgd_reconstructs_an_exactly_low_rank_matrix() {
        // M = u * v^T is exactly rank 1, so a rank-1 factorization must
        // reproduce it (up to floating point).
        let rows = 6;
        let cols = 5;
        let u: Vec<f32> = (0..rows).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        let v: Vec<f32> = (0..cols).map(|j| 0.2 * (j as f32 + 1.0) - 0.3).collect();
        let mut data = vec![0.0f32; rows * cols];
        for (row, u_value) in u.iter().enumerate() {
            for (col, v_value) in v.iter().enumerate() {
                data[row * cols + col] = u_value * v_value;
            }
        }
        let gradient = Tensor::from_slice(&data, &[rows, cols]).expect("tensor must build in test");

        let compressor = GradientCompressor::new(config(CompressionType::PowerSGD { rank: 1 }));
        let compressed = compressor.compress_powersgd(&gradient, 1).expect("compress in test");

        assert!(
            matches!(compressed.compressed_data, CompressedData::LowRank { .. }),
            "PowerSGD must emit a low-rank payload"
        );
        assert!(compressed.compression_ratio < 1.0);

        let restored =
            compressed.decompress().expect("decompress in test").to_vec_f32().expect("read");
        assert_eq!(restored.len(), data.len());
        for (got, want) in restored.iter().zip(&data) {
            approx::assert_relative_eq!(got, want, epsilon = 1e-4);
        }
    }

    #[test]
    fn powersgd_shape_is_preserved_and_reconstruction_is_not_truncated() {
        // The old implementation truncated the byte vector and copied it into
        // the head of a zero buffer, leaving the tail all zeros.
        let rows = 8;
        let cols = 8;
        let data: Vec<f32> = (0..rows * cols).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();
        let gradient = Tensor::from_slice(&data, &[rows, cols]).expect("tensor must build in test");

        let compressor = GradientCompressor::new(config(CompressionType::PowerSGD { rank: 3 }));
        let compressed = compressor.compress_powersgd(&gradient, 3).expect("compress in test");
        let restored = compressed.decompress().expect("decompress in test");

        assert_eq!(restored.shape(), vec![rows, cols]);
        let restored = restored.to_vec_f32().expect("read");
        let tail_energy: f32 = restored[rows * cols / 2..].iter().map(|value| value.abs()).sum();
        assert!(tail_energy > 0.0, "the second half must not be all zeros");
    }

    #[test]
    fn powersgd_falls_back_to_dense_for_one_dimensional_gradients() {
        let compressor = GradientCompressor::new(config(CompressionType::PowerSGD { rank: 2 }));
        let gradient = small_gradient();
        let compressed = compressor.compress_powersgd(&gradient, 2).expect("compress in test");
        assert!(matches!(
            compressed.compressed_data,
            CompressedData::Uncompressed(_)
        ));
        let restored =
            compressed.decompress().expect("decompress in test").to_vec_f32().expect("read");
        assert_eq!(restored, gradient.to_vec_f32().expect("read"));
    }

    #[test]
    fn adaptive_compression_scales_sparsity_with_variance() {
        let compressor = GradientCompressor::new(config(CompressionType::Adaptive));

        let low_variance =
            Tensor::from_slice(&[0.001f32; 40], &[40]).expect("tensor must build in test");
        let high_variance: Vec<f32> =
            (0..40).map(|i| if i % 2 == 0 { -1.0 } else { 1.0 }).collect();
        let high_variance =
            Tensor::from_slice(&high_variance, &[40]).expect("tensor must build in test");

        let low = compressor.compress_adaptive(&low_variance).expect("compress in test");
        let high = compressor.compress_adaptive(&high_variance).expect("compress in test");

        assert!(
            low.compression_ratio < high.compression_ratio,
            "low-variance gradients must compress more aggressively"
        );
    }

    #[test]
    fn compress_gradients_round_trips_every_codec() {
        let codecs = [
            CompressionType::None,
            CompressionType::TopK { k: 8 },
            CompressionType::RandomSparsification { ratio: 1.0 },
            CompressionType::Quantization { bits: 8 },
            CompressionType::OneBitSGD,
            CompressionType::Adaptive,
        ];

        for codec in codecs {
            let mut compressor = GradientCompressor::new(config(codec.clone()));
            let mut gradients = HashMap::new();
            gradients.insert("w".to_string(), small_gradient());

            let compressed = compressor.compress_gradients(&gradients).expect("compress in test");
            let dense = compressor.decompress_gradients(&compressed).expect("decompress in test");

            let restored =
                dense.get("w").expect("gradient must survive").to_vec_f32().expect("read");
            assert_eq!(
                restored.len(),
                8,
                "{codec:?} must preserve the element count"
            );
            assert!(
                restored.iter().any(|value| value.abs() > 1e-6),
                "{codec:?} must not return an all-zero gradient"
            );
        }
    }

    #[test]
    fn error_feedback_accumulates_the_residual() {
        let mut config = config(CompressionType::TopK { k: 1 });
        config.error_feedback = true;
        let mut compressor = GradientCompressor::new(config);

        let mut gradients = HashMap::new();
        gradients.insert("w".to_string(), small_gradient());

        compressor.compress_gradients(&gradients).expect("compress in test");
        let residual = compressor
            .error_feedback_residual("w")
            .expect("residual must be retained")
            .to_vec_f32()
            .expect("read");

        // Top-1 keeps index 3 (-0.4); everything else is deferred.
        assert_eq!(residual[3], 0.0);
        assert!(residual.iter().any(|value| value.abs() > 1e-6));
    }

    #[test]
    fn compression_stats_report_real_byte_counts() {
        let mut compressor = GradientCompressor::new(config(CompressionType::TopK { k: 2 }));
        let mut gradients = HashMap::new();
        gradients.insert("w".to_string(), small_gradient());

        compressor.compress_gradients(&gradients).expect("compress in test");
        let stats = compressor.get_compression_stats();

        assert_eq!(stats.total_uncompressed_bytes, 8 * 4);
        assert!(stats.total_compressed_bytes > 0);
        assert!(stats.average_compression_ratio > 0.0);
    }
}
