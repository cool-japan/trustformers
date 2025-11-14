//! # MetalBackend - attention_gpu_to_gpu_group Methods
//!
//! This module contains method implementations for `MetalBackend`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::common::*;
use super::metalbackend_type::MetalBackend;
use super::types::{BufferCache, BufferId};

impl MetalBackend {
    /// Execute full multi-head attention on GPU (ZERO CPU transfers!)
    /// Inputs: Q, K, V buffers [batch, seq_len, hidden_size]
    /// Output: attention output [batch, seq_len, hidden_size]
    /// Performs: For each head: softmax(Q_h @ K_h^T / sqrt(d_k)) @ V_h, then concatenate
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

        #[cfg(debug_assertions)]
        eprintln!(
            "🚀 GPU Multi-Head Attention: batch={}, seq={}, heads={}, head_dim={}",
            batch_size, seq_len, num_heads, head_dim
        );

        if batch_size != 1 {
            return Err(TrustformersError::tensor_op_error(
                "GPU attention currently only supports batch_size=1",
                "attention_gpu_to_gpu",
            ));
        }

        #[cfg(debug_assertions)]
        eprintln!("   Step 1: Reshaping Q, K, V to separate heads");

        let q_heads = self.reshape_to_heads_gpu(q_buffer_id, seq_len, num_heads, head_dim)?;
        let k_heads = self.reshape_to_heads_gpu(k_buffer_id, seq_len, num_heads, head_dim)?;
        let v_heads = self.reshape_to_heads_gpu(v_buffer_id, seq_len, num_heads, head_dim)?;
        let scale = 1.0 / (head_dim as f32).sqrt();

        #[cfg(debug_assertions)]
        eprintln!(
            "   Step 2: 🚀 BATCHED multi-head attention (scale={}, {} heads in parallel)",
            scale, num_heads
        );

        #[cfg(debug_assertions)]
        eprintln!("      2a. Batched transpose K ({} heads)", num_heads);

        let k_heads_t =
            self.batched_transpose_gpu_to_gpu(&k_heads, num_heads, seq_len, head_dim)?;

        #[cfg(debug_assertions)]
        eprintln!(
            "      2b. 🔥 FUSED batched scaled matmul + softmax ({} heads)",
            num_heads
        );

        let attn_weights = self.batched_scaled_matmul_softmax_causal_gpu_to_gpu(
            &q_heads, &k_heads_t, num_heads, seq_len, head_dim, scale,
        )?;

        #[cfg(debug_assertions)]
        eprintln!("      2c. Batched matmul @ V ({} heads)", num_heads);

        let output_heads_id = self.batched_matmul_gpu_to_gpu(
            &attn_weights,
            &v_heads,
            num_heads,
            seq_len,
            seq_len,
            head_dim,
        )?;

        #[cfg(debug_assertions)]
        eprintln!(
            "   ✅ Batched attention complete: {} heads processed in 3 GPU dispatches (vs 112 sequential)",
            num_heads
        );

        #[cfg(debug_assertions)]
        eprintln!(
            "   Step 3: Concatenating heads back to [seq_len, {}]",
            hidden_size
        );

        let final_output =
            self.reshape_from_heads_gpu(&output_heads_id, seq_len, num_heads, head_dim)?;

        #[cfg(debug_assertions)]
        eprintln!("✅ GPU Multi-Head Attention complete!");

        Ok(final_output)
    }
}
