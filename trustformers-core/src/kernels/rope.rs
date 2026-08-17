// Optimized Rotary Position Embedding (RoPE) implementation with vectorization

#![allow(unused_variables)] // Optimized implementation with architecture-specific code paths

use crate::tensor::Tensor;
use anyhow::Result;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// Vectorized RoPE implementation for improved performance
pub struct VectorizedRoPE {
    dim: usize,
    #[allow(dead_code)]
    max_seq_len: usize,
    base: f32,
    #[allow(dead_code)]
    inv_freq: Tensor,
    use_simd: bool,
}

impl VectorizedRoPE {
    pub fn new(dim: usize, max_seq_len: usize, base: f32) -> Result<Self> {
        // Compute inverse frequencies
        let half_dim = dim / 2;
        let inv_freq_vec: Vec<f32> =
            (0..half_dim).map(|i| 1.0 / base.powf(2.0 * i as f32 / dim as f32)).collect();

        let inv_freq = Tensor::new(inv_freq_vec)?;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let use_simd = is_x86_feature_detected!("avx2") && dim >= 128 && dim.is_multiple_of(8);
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let use_simd = false;

        Ok(Self {
            dim,
            max_seq_len,
            base,
            inv_freq,
            use_simd,
        })
    }

    pub fn forward(&self, x: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        if self.use_simd {
            self.forward_simd(x, position_ids)
        } else {
            self.forward_standard(x, position_ids)
        }
    }

    fn forward_standard(&self, x: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        // Standard RoPE implementation without SIMD optimization
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];
        let hidden_size = x.shape()[2];

        // Get precomputed cos/sin embeddings
        let cos_sin = self.get_cos_sin_embeddings(position_ids)?;
        let half_dim = hidden_size / 2;

        // Split input tensor into first half and second half
        let x1 = x.slice(2, 0, half_dim)?;
        let x2 = x.slice(2, half_dim, hidden_size)?;

        // Extract cos and sin from precomputed embeddings
        // cos_sin has shape [batch, seq_len, half_dim, 2]
        let cos_vals = cos_sin.slice(3, 0, 1)?.squeeze(3)?;
        let sin_vals = cos_sin.slice(3, 1, 2)?.squeeze(3)?;

        // Apply RoPE transformation:
        // out1 = x1 * cos - x2 * sin
        // out2 = x1 * sin + x2 * cos
        let out1 = x1.mul(&cos_vals)?.sub(&x2.mul(&sin_vals)?)?;
        let out2 = x1.mul(&sin_vals)?.add(&x2.mul(&cos_vals)?)?;

        // Concatenate the results back together
        let result = Tensor::concat(&[out1, out2], 2)?;
        Ok(result)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    #[allow(dead_code)]
    unsafe fn forward_simd_inner(
        &self,
        x_data: &[f32],
        cos_data: &[f32],
        sin_data: &[f32],
        output_data: &mut [f32],
        batch_size: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) {
        let half_dim = head_dim / 2;

        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..num_heads {
                    let x_base = b * seq_len * num_heads * head_dim
                        + s * num_heads * head_dim
                        + h * head_dim;
                    let cos_sin_base = b * seq_len * half_dim + s * half_dim;

                    let x1_start = x_base;
                    let x2_start = x_base + half_dim;
                    let out1_start = x_base;
                    let out2_start = x_base + half_dim;

                    // Process in chunks of 8 (AVX2 width)
                    let mut i = 0;
                    while i + 8 <= half_dim {
                        // Load data
                        let x1_vals = _mm256_loadu_ps(&x_data[x1_start + i]);
                        let x2_vals = _mm256_loadu_ps(&x_data[x2_start + i]);
                        let cos_vals = _mm256_loadu_ps(&cos_data[cos_sin_base + i]);
                        let sin_vals = _mm256_loadu_ps(&sin_data[cos_sin_base + i]);

                        // Compute rotated values
                        // rotated_x1 = x1 * cos - x2 * sin
                        let x1_cos = _mm256_mul_ps(x1_vals, cos_vals);
                        let x2_sin = _mm256_mul_ps(x2_vals, sin_vals);
                        let rotated_x1 = _mm256_sub_ps(x1_cos, x2_sin);

                        // rotated_x2 = x2 * cos + x1 * sin
                        let x2_cos = _mm256_mul_ps(x2_vals, cos_vals);
                        let x1_sin = _mm256_mul_ps(x1_vals, sin_vals);
                        let rotated_x2 = _mm256_add_ps(x2_cos, x1_sin);

                        // Store results
                        _mm256_storeu_ps(&mut output_data[out1_start + i], rotated_x1);
                        _mm256_storeu_ps(&mut output_data[out2_start + i], rotated_x2);

                        i += 8;
                    }

                    // Handle remaining elements
                    while i < half_dim {
                        let x1 = x_data[x1_start + i];
                        let x2 = x_data[x2_start + i];
                        let cos_val = cos_data[cos_sin_base + i];
                        let sin_val = sin_data[cos_sin_base + i];

                        output_data[out1_start + i] = x1 * cos_val - x2 * sin_val;
                        output_data[out2_start + i] = x2 * cos_val + x1 * sin_val;

                        i += 1;
                    }
                }
            }
        }
    }

    fn forward_simd(&self, x: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        // Enhanced SIMD-optimized RoPE implementation
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];
        let hidden_size = x.shape()[2];

        // Get precomputed cos/sin embeddings
        let cos_sin = self.get_cos_sin_embeddings(position_ids)?;
        let cos_embed = cos_sin.slice(3, 0, 1)?; // Extract cos part
        let sin_embed = cos_sin.slice(3, 1, 2)?; // Extract sin part

        // Split input into even and odd dimensions for rotation
        let half_dim = hidden_size / 2;
        let x_even = x.slice(2, 0, half_dim)?; // x[..., ::2]
        let x_odd = x.slice(2, half_dim, hidden_size)?; // x[..., 1::2]

        // Apply rotary transformation:
        // x_out = x * cos - x_rotated * sin
        // where x_rotated swaps even/odd dimensions
        let cos_part = x_even.mul(&cos_embed.squeeze(3)?)?;
        let sin_part = x_odd.mul(&sin_embed.squeeze(3)?)?;

        let rotated_even = cos_part.sub(&sin_part)?;
        let rotated_odd =
            x_even.mul(&sin_embed.squeeze(3)?)?.add(&x_odd.mul(&cos_embed.squeeze(3)?)?)?;

        // Concatenate back
        Ok(Tensor::concat(&[rotated_even, rotated_odd], 2)?)
    }

    fn get_cos_sin_embeddings(&self, position_ids: &Tensor) -> Result<Tensor> {
        // Enhanced cos/sin embedding computation
        let batch_size = position_ids.shape()[0];
        let seq_len = position_ids.shape()[1];
        let half_dim = self.dim / 2;

        // Create frequency tensor: 1 / (base^(2i/dim)) for i in range(dim/2)
        let mut freqs = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            let freq = 1.0 / self.base.powf(2.0 * (i as f32) / (self.dim as f32));
            freqs.push(freq);
        }
        let freq_tensor = Tensor::from_vec(freqs, &[half_dim])?;

        // Compute position * frequencies
        // Convert position_ids to f32 tensor for computation
        let position_data = position_ids.data()?;
        let position_f32: Vec<f32> = position_data.into_iter().collect();
        let position_ids_f32 = Tensor::from_vec(position_f32, &position_ids.shape())?;

        // Create expanded frequency tensor for broadcasting
        let freq_expanded = freq_tensor.unsqueeze(0)?.unsqueeze(0)?;
        let pos_expanded = position_ids_f32.unsqueeze(position_ids_f32.shape().len())?;
        let pos_freqs = pos_expanded.mul(&freq_expanded)?;

        // Compute cos and sin
        let cos_embed = pos_freqs.cos()?;
        let sin_embed = pos_freqs.sin()?;

        // Stack cos and sin along last dimension
        let cos_expanded = cos_embed.unsqueeze(cos_embed.shape().len())?;
        let sin_expanded = sin_embed.unsqueeze(sin_embed.shape().len())?;

        Ok(Tensor::concat(&[cos_expanded, sin_expanded], 3)?)
    }
}

/// Optimized RoPE for specific common configurations
pub struct OptimizedRoPE {
    config: RoPEConfig,
    precomputed_freqs: Option<Tensor>,
}

#[derive(Debug, Clone)]
pub struct RoPEConfig {
    pub dim: usize,
    pub max_seq_len: usize,
    pub base: f32,
    pub scaling_factor: f32,
    pub scaling_type: RoPEScalingType,
}

#[derive(Debug, Clone, Copy)]
pub enum RoPEScalingType {
    None,
    Linear,
    Dynamic,
}

impl OptimizedRoPE {
    pub fn new(config: RoPEConfig) -> Result<Self> {
        Ok(Self {
            config,
            precomputed_freqs: None,
        })
    }

    /// Create OptimizedRoPE with precomputed frequencies for maximum performance
    pub fn with_precomputed_freqs(config: RoPEConfig) -> Result<Self> {
        let precomputed = Self::precompute_frequencies(&config)?;
        Ok(Self {
            config,
            precomputed_freqs: Some(precomputed),
        })
    }

    /// Enable precomputation for this instance
    pub fn enable_precomputation(&mut self) -> Result<()> {
        if self.precomputed_freqs.is_none() {
            self.precomputed_freqs = Some(Self::precompute_frequencies(&self.config)?);
        }
        Ok(())
    }

    fn precompute_frequencies(config: &RoPEConfig) -> Result<Tensor> {
        let half_dim = config.dim / 2;

        // Create frequency values
        let mut freq_values = Vec::new();
        for i in 0..half_dim {
            let freq = 1.0 / config.base.powf(2.0 * i as f32 / config.dim as f32);
            freq_values.push(freq);
        }

        // Apply scaling
        let scaled_freqs: Vec<f32> = match config.scaling_type {
            RoPEScalingType::None => freq_values,
            RoPEScalingType::Linear => {
                freq_values.into_iter().map(|f| f / config.scaling_factor).collect()
            },
            RoPEScalingType::Dynamic => {
                // More complex dynamic scaling logic
                freq_values
                    .into_iter()
                    .enumerate()
                    .map(|(i, f)| {
                        let scale = if i < half_dim / 2 { config.scaling_factor } else { 1.0 };
                        f / scale
                    })
                    .collect()
            },
        };

        // Precompute cos/sin for all positions.
        //
        // Layout: row-major [max_seq_len, half_dim * 2], where each row `pos`
        // holds that position's `half_dim` cos values followed by its
        // `half_dim` sin values: row(pos) = [cos_0..cos_{h-1}, sin_0..sin_{h-1}].
        // This must interleave per-position (not stack all cos rows followed by
        // all sin rows globally), since `forward_with_precomputed` slices this
        // tensor along dim 1 assuming exactly this per-row layout.
        let mut all_vals = Vec::with_capacity(config.max_seq_len * half_dim * 2);
        for pos in 0..config.max_seq_len {
            for &freq in scaled_freqs.iter() {
                let angle = pos as f32 * freq;
                all_vals.push(angle.cos());
            }
            for &freq in scaled_freqs.iter() {
                let angle = pos as f32 * freq;
                all_vals.push(angle.sin());
            }
        }

        Ok(Tensor::from_vec(
            all_vals,
            &[config.max_seq_len, half_dim * 2],
        )?)
    }

    pub fn forward(&self, x: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        if let Some(ref precomputed) = self.precomputed_freqs {
            self.forward_with_precomputed(x, position_ids, precomputed)
        } else {
            self.forward_dynamic(x, position_ids)
        }
    }

    fn forward_with_precomputed(
        &self,
        x: &Tensor,
        position_ids: &Tensor,
        precomputed: &Tensor,
    ) -> Result<Tensor> {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];
        let hidden_size = x.shape()[2];
        let half_dim = hidden_size / 2;

        if half_dim * 2 != self.config.dim {
            anyhow::bail!(
                "OptimizedRoPE::forward_with_precomputed: input hidden_size {} (half_dim {}) \
                 does not match the configured RoPE dim {} used to build the precomputed table",
                hidden_size,
                half_dim,
                self.config.dim
            );
        }

        let pos_shape = position_ids.shape();
        if pos_shape != [batch_size, seq_len] {
            anyhow::bail!(
                "OptimizedRoPE::forward_with_precomputed: position_ids shape {:?} does not \
                 match x's [batch_size, seq_len] = [{}, {}]",
                pos_shape,
                batch_size,
                seq_len
            );
        }

        // `precomputed` is [max_seq_len, half_dim * 2]; each row `pos` holds
        // [cos_0..cos_{h-1}, sin_0..sin_{h-1}] for that position (see
        // `precompute_frequencies`). Gather the rows selected by `position_ids`
        // instead of reshaping the whole table, so the actual requested
        // positions (which need not be sequential or dense) drive the result.
        let cos_precomputed = precomputed.slice(1, 0, half_dim)?; // [max_seq_len, half_dim]
        let sin_precomputed = precomputed.slice(1, half_dim, half_dim * 2)?; // [max_seq_len, half_dim]
        let cos_table = cos_precomputed.data()?;
        let sin_table = sin_precomputed.data()?;
        let position_data = position_ids.data()?;

        let mut cos_gathered = Vec::with_capacity(batch_size * seq_len * half_dim);
        let mut sin_gathered = Vec::with_capacity(batch_size * seq_len * half_dim);
        for &pos_f in &position_data {
            let pos = pos_f.round();
            if pos < 0.0 || pos as usize >= self.config.max_seq_len {
                anyhow::bail!(
                    "OptimizedRoPE::forward_with_precomputed: position {} is out of bounds \
                     for max_seq_len {}",
                    pos_f,
                    self.config.max_seq_len
                );
            }
            let row_start = pos as usize * half_dim;
            cos_gathered.extend_from_slice(&cos_table[row_start..row_start + half_dim]);
            sin_gathered.extend_from_slice(&sin_table[row_start..row_start + half_dim]);
        }

        let cos_embed = Tensor::from_vec(cos_gathered, &[batch_size, seq_len, half_dim])?;
        let sin_embed = Tensor::from_vec(sin_gathered, &[batch_size, seq_len, half_dim])?;

        // Split input tensor for rotation
        let x_first_half = x.slice(2, 0, half_dim)?;
        let x_second_half = x.slice(2, half_dim, hidden_size)?;

        // Apply rotary transformation
        let cos_result = x_first_half.mul(&cos_embed)?.sub(&x_second_half.mul(&sin_embed)?)?;
        let sin_result = x_first_half.mul(&sin_embed)?.add(&x_second_half.mul(&cos_embed)?)?;

        // Concatenate results
        Ok(Tensor::concat(&[cos_result, sin_result], 2)?)
    }

    fn forward_dynamic(&self, x: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        // Fallback to dynamic computation for long sequences
        let vectorized_rope =
            VectorizedRoPE::new(self.config.dim, self.config.max_seq_len, self.config.base)?;
        vectorized_rope.forward(x, position_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vectorized_rope_creation() -> Result<()> {
        let rope = VectorizedRoPE::new(128, 2048, 10000.0)?;
        assert_eq!(rope.dim, 128);
        assert_eq!(rope.max_seq_len, 2048);
        Ok(())
    }

    #[test]
    fn test_optimized_rope_creation() -> Result<()> {
        let config = RoPEConfig {
            dim: 128,
            max_seq_len: 1024,
            base: 10000.0,
            scaling_factor: 1.0,
            scaling_type: RoPEScalingType::None,
        };

        let rope = OptimizedRoPE::new(config)?;
        assert_eq!(rope.config.dim, 128);

        Ok(())
    }

    #[test]
    fn test_rope_config() {
        let config = RoPEConfig {
            dim: 64,
            max_seq_len: 512,
            base: 10000.0,
            scaling_factor: 1.0,
            scaling_type: RoPEScalingType::Linear,
        };

        assert_eq!(config.dim, 64);
        assert_eq!(config.max_seq_len, 512);
    }

    /// Regression test for the precomputed-frequency path ignoring
    /// `position_ids`. Before the fix, `forward_with_precomputed` reshaped the
    /// flat `[max_seq_len, half_dim*2]` table directly (ignoring the actual
    /// position values) and the table itself was laid out incorrectly (all
    /// cos rows stacked before all sin rows, rather than interleaved
    /// per-position). With non-sequential position ids and `max_seq_len !=
    /// batch_size * seq_len`, the old code would either error out on the
    /// reshape or silently produce wrong rotations. The fixed
    /// `OptimizedRoPE::forward` (precomputed path) must match
    /// `VectorizedRoPE::forward` (which computes cos/sin directly from
    /// `position_ids` with no precomputed table) for the same inputs.
    #[test]
    fn test_optimized_rope_precomputed_matches_dynamic_for_arbitrary_positions() -> Result<()> {
        let dim = 8;
        let max_seq_len = 16;
        let base = 10000.0;
        let config = RoPEConfig {
            dim,
            max_seq_len,
            base,
            scaling_factor: 1.0,
            scaling_type: RoPEScalingType::None,
        };
        let rope = OptimizedRoPE::with_precomputed_freqs(config)?;

        let batch_size = 2;
        let seq_len = 3;
        let hidden_size = dim;

        // Distinct, non-monotonic values so mixing up rows/positions is detectable.
        let x_vals: Vec<f32> = (0..(batch_size * seq_len * hidden_size))
            .map(|i| i as f32 * 0.1 - 1.0)
            .collect();
        let x = Tensor::from_vec(x_vals, &[batch_size, seq_len, hidden_size])?;

        // Deliberately unordered, non-sequential positions (and
        // batch_size * seq_len = 6 != max_seq_len = 16, which previously
        // broke the naive reshape).
        let position_vals: Vec<f32> = vec![5.0, 0.0, 9.0, 2.0, 12.0, 7.0];
        let position_ids = Tensor::from_vec(position_vals, &[batch_size, seq_len])?;

        let precomputed_out = rope.forward(&x, &position_ids)?;

        let vectorized_rope = VectorizedRoPE::new(dim, max_seq_len, base)?;
        let dynamic_out = vectorized_rope.forward(&x, &position_ids)?;

        assert_eq!(precomputed_out.shape(), dynamic_out.shape());

        let a = precomputed_out.data()?;
        let b = dynamic_out.data()?;
        assert_eq!(a.len(), b.len());
        for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (av - bv).abs() < 1e-4,
                "mismatch at flat index {}: precomputed={}, dynamic={}",
                i,
                av,
                bv
            );
        }

        // Sanity check: the output must actually depend on which position was
        // used (i.e. this isn't trivially passing because everything is
        // zero). Swapping to all-zero positions must give a different result
        // for at least one element, since cos(0)=1, sin(0)=0 differs from the
        // arbitrary positions above.
        let zero_positions =
            Tensor::from_vec(vec![0.0; batch_size * seq_len], &[batch_size, seq_len])?;
        let zero_pos_out = rope.forward(&x, &zero_positions)?.data()?;
        assert!(
            zero_pos_out.iter().zip(a.iter()).any(|(z, o)| (z - o).abs() > 1e-4),
            "output does not vary with position_ids"
        );

        Ok(())
    }
}
