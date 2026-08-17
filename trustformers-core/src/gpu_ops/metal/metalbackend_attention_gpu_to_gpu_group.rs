//! # MetalBackend - attention_gpu_to_gpu_group Methods
//!
//! This module contains method implementations for `MetalBackend`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
use super::common::*;

use super::metalbackend_type::MetalBackend;
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
use super::types::{BufferCache, BufferId};

#[cfg(all(target_os = "macos", feature = "metal"))]
impl MetalBackend {
    /// Execute full multi-head attention on GPU (ZERO CPU transfers!)
    /// Inputs: Q, K, V buffers [batch, seq_len, hidden_size]
    /// Output: attention output [batch, seq_len, hidden_size]
    /// Performs: For each head: softmax(Q_h @ K_h^T / sqrt(d_k)) @ V_h, then concatenate
    ///
    /// # Buffer lifetimes
    ///
    /// The five intermediates (`q_heads`, `k_heads`, `v_heads`, `k_heads_t`,
    /// `attn_weights`, `output_heads`) are provably dead once the final output buffer
    /// exists, so they are released before returning. They used to be left in the
    /// process-global buffer cache forever - six leaked GPU allocations per attention
    /// call, i.e. per layer per token.
    pub fn attention_gpu_to_gpu(
        &self,
        q_buffer_id: &BufferId,
        k_buffer_id: &BufferId,
        v_buffer_id: &BufferId,
        batch_size: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<BufferId> {
        let hidden_size = num_heads * head_dim;

        tracing::trace!(
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            "metal: multi-head attention (gpu-to-gpu)"
        );

        if batch_size != 1 {
            return Err(TrustformersError::tensor_op_error(
                "GPU attention currently only supports batch_size=1",
                "attention_gpu_to_gpu",
            ));
        }
        if seq_len == 0 || num_heads == 0 || head_dim == 0 {
            return Err(TrustformersError::tensor_op_error(
                "GPU attention requires non-zero seq_len, num_heads and head_dim",
                "attention_gpu_to_gpu",
            ));
        }

        let q_heads = self.reshape_to_heads_gpu(q_buffer_id, seq_len, num_heads, head_dim)?;
        let k_heads = self.reshape_to_heads_gpu(k_buffer_id, seq_len, num_heads, head_dim)?;
        let v_heads = self.reshape_to_heads_gpu(v_buffer_id, seq_len, num_heads, head_dim)?;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Run the pipeline, then release every intermediate regardless of where it
        // failed: a mid-pipeline error must not leak the buffers already allocated.
        let mut scratch: Vec<BufferId> = vec![q_heads, k_heads, v_heads];
        let result = (|| -> Result<BufferId> {
            let k_heads_t =
                self.batched_transpose_gpu_to_gpu(&k_heads, num_heads, seq_len, head_dim)?;
            scratch.push(k_heads_t);

            let attn_weights = self.batched_scaled_matmul_softmax_causal_gpu_to_gpu(
                &q_heads, &k_heads_t, num_heads, seq_len, head_dim, scale,
            )?;
            scratch.push(attn_weights);

            let output_heads_id = self.batched_matmul_gpu_to_gpu(
                &attn_weights,
                &v_heads,
                num_heads,
                seq_len,
                seq_len,
                head_dim,
            )?;
            scratch.push(output_heads_id);

            self.reshape_from_heads_gpu(&output_heads_id, seq_len, num_heads, head_dim)
        })();

        self.release_buffers(&scratch)?;
        let final_output = result?;

        tracing::trace!(
            hidden_size,
            "metal: multi-head attention complete, {} intermediates released",
            scratch.len()
        );

        Ok(final_output)
    }
}
