//! # MetalBackend - attention_with_cache_gpu_to_gpu_group Methods
//!
//! This module contains method implementations for `MetalBackend`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::common::*;
use super::metalbackend_type::MetalBackend;
use super::types::{BufferCache, BufferId};

impl MetalBackend {
    /// Execute multi-head attention with KV-cache on GPU (supports different Q vs K/V seq lengths)
    ///
    /// This version takes pre-reshaped tensors in multi-head format and supports
    /// different sequence lengths for Q (current tokens) vs K/V (cached + current).
    ///
    /// # Arguments
    ///
    /// * `q_heads_id` - Query tensor: [batch, num_heads, q_seq_len, head_dim]
    /// * `k_heads_id` - Key tensor: [batch, num_heads, kv_seq_len, head_dim]
    /// * `v_heads_id` - Value tensor: [batch, num_heads, kv_seq_len, head_dim]
    /// * `batch_size` - Batch size
    /// * `q_seq_len` - Query sequence length (typically 1 during generation)
    /// * `kv_seq_len` - Key/Value sequence length (cached + new tokens)
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head
    ///
    /// # Returns
    ///
    /// Buffer ID containing output: [batch, num_heads, q_seq_len, head_dim]
    pub fn attention_with_cache_gpu_to_gpu(
        &self,
        q_heads_id: &BufferId,
        k_heads_id: &BufferId,
        v_heads_id: &BufferId,
        batch_size: usize,
        q_seq_len: usize,
        kv_seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<BufferId> {
        // eprintln!(
        //     "🚀 GPU Multi-Head Attention (with cache): batch={}, q_seq={}, kv_seq={}, heads={}, head_dim={}",
        //     batch_size, q_seq_len, kv_seq_len, num_heads, head_dim
        // );
        if batch_size != 1 {
            return Err(TrustformersError::tensor_op_error(
                "GPU cached attention currently only supports batch_size=1",
                "attention_with_cache_gpu_to_gpu",
            ));
        }
        let scale = 1.0 / (head_dim as f32).sqrt();
        // eprintln!(
        //     "   Step 1: Batched transpose K ({} heads, kv_seq={})",
        //     num_heads, kv_seq_len
        // );
        let k_heads_t =
            self.batched_transpose_gpu_to_gpu(k_heads_id, num_heads, kv_seq_len, head_dim)?;
        // eprintln!(
        //     "   Step 2: 🔥 Batched scaled matmul + softmax (q_seq={}, kv_seq={})",
        //     q_seq_len, kv_seq_len
        // );
        let attn_weights = if q_seq_len == kv_seq_len {
            // Same sequence length: use causal-masked fused kernel
            self.batched_scaled_matmul_softmax_causal_gpu_to_gpu(
                q_heads_id, &k_heads_t, num_heads, q_seq_len, head_dim, scale,
            )?
        } else {
            // Different sequence lengths (generation): use gen-optimized fused kernel
            self.batched_scaled_matmul_softmax_gen_gpu_to_gpu(
                q_heads_id, &k_heads_t, num_heads, q_seq_len, kv_seq_len, head_dim, scale,
            )?
        };
        // eprintln!("   Step 3: Batched matmul @ V ({} heads)", num_heads);
        let output_heads_id = self.batched_matmul_gpu_to_gpu(
            &attn_weights,
            v_heads_id,
            num_heads,
            q_seq_len,
            kv_seq_len,
            head_dim,
        )?;
        // eprintln!("✅ GPU cached attention complete!");
        Ok(output_heads_id)
    }
}
