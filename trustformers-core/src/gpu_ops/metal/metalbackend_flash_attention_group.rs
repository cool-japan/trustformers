// Flash Attention implementation for Metal GPU
// Based on "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
// (Dao et al., 2022)
//
// This implementation uses tiled matrix multiplication with online softmax to achieve:
// - O(N) memory complexity vs O(N²) for standard attention
// - 2-3x speedup through better SRAM utilization
// - Numerically stable softmax computation

use crate::errors::TrustformersError;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::gpu_ops::metal::metalbackend_type::MetalBackend;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::gpu_ops::metal::types::BufferId;
#[cfg(all(target_os = "macos", feature = "metal"))]
use metal::*;

#[cfg(all(target_os = "macos", feature = "metal"))]
use std::ffi::c_void;
#[cfg(all(target_os = "macos", feature = "metal"))]
use std::sync::Arc;
#[allow(dead_code)]
type Result<T> = std::result::Result<T, TrustformersError>;

// Block sizes optimized for M1 Max (48KB SRAM per threadgroup)
// Block_Q: Number of query tokens processed per threadgroup
// Block_KV: Number of key/value tokens processed per inner loop iteration
#[allow(dead_code)]
const BLOCK_Q: usize = 32;
#[allow(dead_code)]
const BLOCK_KV: usize = 64;
#[allow(dead_code)]
#[cfg(all(target_os = "macos", feature = "metal"))]
impl MetalBackend {
    /// Flash Attention forward pass with KV-cache support
    ///
    /// Implements tiled attention with online softmax for memory-efficient computation.
    ///
    /// # Arguments
    /// * `q_heads_id` - Query buffer [batch, num_heads, q_seq_len, head_dim]
    /// * `k_heads_id` - Key buffer [batch, num_heads, kv_seq_len, head_dim]
    /// * `v_heads_id` - Value buffer [batch, num_heads, kv_seq_len, head_dim]
    /// * `batch_size` - Batch size
    /// * `q_seq_len` - Query sequence length
    /// * `kv_seq_len` - Key/Value sequence length (includes cached tokens)
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head (typically 64, 80, or 128)
    ///
    /// # Returns
    /// * `BufferId` - Output buffer [batch, num_heads, q_seq_len, head_dim]
    ///
    /// # Performance
    /// - Memory: O(N) vs O(N²) for standard attention
    /// - Expected speedup: 2.5-3x vs standard attention
    /// - Target ALU utilization: 60-70%
    pub fn flash_attention_with_cache(
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
        // Validate inputs
        if head_dim > 256 {
            return Err(TrustformersError::hardware_error(
                &format!("Head dimension {} exceeds maximum 256", head_dim),
                "flash_attention_with_cache",
            ));
        }

        // Get input buffers
        let q_buffer = self.get_persistent_buffer(q_heads_id)?;
        let k_buffer = self.get_persistent_buffer(k_heads_id)?;
        let v_buffer = self.get_persistent_buffer(v_heads_id)?;

        // Calculate output size: [batch, num_heads, q_seq_len, head_dim]
        let output_size = batch_size * num_heads * q_seq_len * head_dim;

        // Create output buffer
        let output_buffer = self.device.new_buffer(
            (output_size * 4) as u64, // 4 bytes per f32
            MTLResourceOptions::StorageModeShared,
        );

        // Create auxiliary buffer for logsumexp [batch, num_heads, q_seq_len]
        // Used for numerical stability in online softmax
        let logsumexp_size = batch_size * num_heads * q_seq_len;
        let logsumexp_buffer = self.device.new_buffer(
            (logsumexp_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create parameter buffer
        #[repr(C)]
        struct FlashAttentionParams {
            batch_size: u32,
            num_heads: u32,
            q_seq_len: u32,
            kv_seq_len: u32,
            head_dim: u32,
            scale: f32,           // 1/sqrt(head_dim) for scaled dot-product
            use_causal_mask: u32, // 1 for autoregressive, 0 for bidirectional
        }

        let scale = 1.0 / (head_dim as f32).sqrt();
        let params = FlashAttentionParams {
            batch_size: batch_size as u32,
            num_heads: num_heads as u32,
            q_seq_len: q_seq_len as u32,
            kv_seq_len: kv_seq_len as u32,
            head_dim: head_dim as u32,
            scale,
            use_causal_mask: 1, // Always causal for GPT-2 style models
        };

        let params_buffer = self.device.new_buffer_with_data(
            &params as *const _ as *const c_void,
            std::mem::size_of::<FlashAttentionParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Set pipeline
        encoder.set_compute_pipeline_state(&self.flash_attention_pipeline);

        // Bind buffers
        encoder.set_buffer(0, Some(&**q_buffer), 0);
        encoder.set_buffer(1, Some(&**k_buffer), 0);
        encoder.set_buffer(2, Some(&**v_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        encoder.set_buffer(4, Some(&logsumexp_buffer), 0);
        encoder.set_buffer(5, Some(&params_buffer), 0);

        // Calculate threadgroup memory sizes
        // We need SRAM for:
        // - Q block: [BLOCK_Q, head_dim] floats
        // - K block: [BLOCK_KV, head_dim] floats
        // - V block: [BLOCK_KV, head_dim] floats
        let shared_q_size = BLOCK_Q * head_dim * 4; // 4 bytes per float
        let shared_k_size = BLOCK_KV * head_dim * 4;
        let shared_v_size = BLOCK_KV * head_dim * 4;

        encoder.set_threadgroup_memory_length(0, shared_q_size as u64);
        encoder.set_threadgroup_memory_length(1, shared_k_size as u64);
        encoder.set_threadgroup_memory_length(2, shared_v_size as u64);

        // Calculate grid dimensions
        // Each threadgroup processes BLOCK_Q query tokens for one (batch, head) pair
        let num_q_blocks = q_seq_len.div_ceil(BLOCK_Q);
        let grid_size = MTLSize {
            width: num_q_blocks as u64,
            height: num_heads as u64,
            depth: batch_size as u64,
        };

        // Threadgroup size: [BLOCK_Q, 1, 1]
        // Each thread handles one query token in the block
        let threadgroup_size = MTLSize {
            width: BLOCK_Q as u64,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed(); // Wait for GPU to complete

        // Register output buffer in cache
        let output_id = self.register_buffer(output_buffer, output_size)?;
        Ok(output_id)
    }

    /// Helper method to register a Metal buffer and return BufferId
    fn register_buffer(&self, buffer: metal::Buffer, _size: usize) -> Result<BufferId> {
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "register_buffer")
        })?;

        let buffer_id = BufferId::new();
        cache.insert(buffer_id, Arc::new(buffer));

        Ok(buffer_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_ops::metal::functions::get_metal_backend;

    #[test]
    fn test_flash_attention_basic() -> Result<()> {
        let backend = get_metal_backend()?;

        // Test dimensions
        let batch_size = 1;
        let num_heads = 2;
        let seq_len = 8;
        let head_dim = 64;

        // Create test data
        let q_data = vec![0.1f32; batch_size * num_heads * seq_len * head_dim];
        let k_data = vec![0.2f32; batch_size * num_heads * seq_len * head_dim];
        let v_data = vec![0.3f32; batch_size * num_heads * seq_len * head_dim];

        let q_id = backend.create_persistent_buffer(&q_data)?;
        let k_id = backend.create_persistent_buffer(&k_data)?;
        let v_id = backend.create_persistent_buffer(&v_data)?;

        // Run flash attention
        let output_id = backend.flash_attention_with_cache(
            &q_id, &k_id, &v_id, batch_size, seq_len, seq_len, num_heads, head_dim,
        )?;

        // Verify output shape
        let output_data = backend.download_buffer_to_vec(&output_id)?;
        assert_eq!(
            output_data.len(),
            batch_size * num_heads * seq_len * head_dim
        );

        // Basic sanity check: output should not be all zeros
        let sum: f32 = output_data.iter().sum();
        assert!(sum.abs() > 0.001, "Output is all zeros");

        Ok(())
    }

    #[test]
    fn test_flash_attention_vs_standard() -> Result<()> {
        let backend = get_metal_backend()?;

        // Test dimensions
        let batch_size = 1;
        let num_heads = 2;
        let seq_len = 16;
        let head_dim = 64;

        // Create random-ish test data (using simple pattern for reproducibility)
        let mut q_data = Vec::new();
        let mut k_data = Vec::new();
        let mut v_data = Vec::new();

        for i in 0..(batch_size * num_heads * seq_len * head_dim) {
            q_data.push(((i % 100) as f32) * 0.01);
            k_data.push(((i % 100) as f32) * 0.01 + 0.1);
            v_data.push(((i % 100) as f32) * 0.01 + 0.2);
        }

        let q_id = backend.create_persistent_buffer(&q_data)?;
        let k_id = backend.create_persistent_buffer(&k_data)?;
        let v_id = backend.create_persistent_buffer(&v_data)?;

        // Run Flash Attention
        let flash_output_id = backend.flash_attention_with_cache(
            &q_id, &k_id, &v_id, batch_size, seq_len, seq_len, num_heads, head_dim,
        )?;

        // Run Standard Attention
        let standard_output_id = backend.attention_with_cache_gpu_to_gpu(
            &q_id, &k_id, &v_id, batch_size, seq_len, seq_len, num_heads, head_dim,
        )?;

        // Download results
        let flash_output = backend.download_buffer_to_vec(&flash_output_id)?;
        let standard_output = backend.download_buffer_to_vec(&standard_output_id)?;

        // Compare outputs
        assert_eq!(flash_output.len(), standard_output.len());

        let mut max_diff = 0.0f32;
        let mut total_diff = 0.0f32;
        let mut num_elements = 0;

        for (i, (&flash, &standard)) in flash_output.iter().zip(standard_output.iter()).enumerate()
        {
            let diff = (flash - standard).abs();
            max_diff = max_diff.max(diff);
            total_diff += diff;
            num_elements += 1;

            // Check element-wise tolerance
            if diff > 1e-3 {
                println!(
                    "Large difference at index {}: flash={}, standard={}, diff={}",
                    i, flash, standard, diff
                );
            }
        }

        let avg_diff = total_diff / num_elements as f32;

        println!("Flash Attention vs Standard Attention:");
        println!("  Max difference: {}", max_diff);
        println!("  Average difference: {}", avg_diff);
        println!("  Total elements: {}", num_elements);

        // Relaxed tolerance due to numerical differences in online softmax
        assert!(
            max_diff < 1e-3,
            "Max difference {} exceeds tolerance 1e-3",
            max_diff
        );
        assert!(
            avg_diff < 1e-4,
            "Average difference {} exceeds tolerance 1e-4",
            avg_diff
        );

        Ok(())
    }

    #[test]
    fn test_flash_attention_causal_masking() -> Result<()> {
        let backend = get_metal_backend()?;

        // Test dimensions
        let batch_size = 1;
        let num_heads = 1;
        let seq_len = 4;
        let head_dim = 8;

        // Create simple test data
        let q_data = vec![1.0f32; batch_size * num_heads * seq_len * head_dim];
        let k_data = vec![1.0f32; batch_size * num_heads * seq_len * head_dim];
        let mut v_data = vec![0.0f32; batch_size * num_heads * seq_len * head_dim];

        // Set each token's V to a unique value
        for token_idx in 0..seq_len {
            for d in 0..head_dim {
                v_data[token_idx * head_dim + d] = (token_idx + 1) as f32;
            }
        }

        let q_id = backend.create_persistent_buffer(&q_data)?;
        let k_id = backend.create_persistent_buffer(&k_data)?;
        let v_id = backend.create_persistent_buffer(&v_data)?;

        // Run Flash Attention with causal mask
        let output_id = backend.flash_attention_with_cache(
            &q_id, &k_id, &v_id, batch_size, seq_len, seq_len, num_heads, head_dim,
        )?;

        let output = backend.download_buffer_to_vec(&output_id)?;

        // Verify causal property: token i should only see tokens 0..=i
        // With uniform Q and K, attention weights are uniform over visible tokens
        // So token i should output average of V[0..=i]

        for token_idx in 0..seq_len {
            let offset = token_idx * head_dim;
            let token_output = &output[offset..offset + head_dim];

            // Expected: average of V values for tokens 0..=token_idx
            let expected_sum: f32 = (1..=token_idx + 1).sum::<usize>() as f32;
            let expected_avg = expected_sum / (token_idx + 1) as f32;

            // Check first dimension (all dimensions should be the same)
            let actual = token_output[0];
            let diff = (actual - expected_avg).abs();

            println!(
                "Token {}: expected={}, actual={}, diff={}",
                token_idx, expected_avg, actual, diff
            );

            assert!(
                diff < 0.1,
                "Token {} causal mask incorrect: expected {}, got {}",
                token_idx,
                expected_avg,
                actual
            );
        }

        Ok(())
    }

    #[test]
    fn test_flash_attention_various_seq_lengths() -> Result<()> {
        let backend = get_metal_backend()?;

        // Test various sequence lengths
        for seq_len in &[8, 16, 32, 64, 128] {
            println!("Testing seq_len={}", seq_len);

            let batch_size = 1;
            let num_heads = 2;
            let head_dim = 64;

            let q_data = vec![0.1f32; batch_size * num_heads * seq_len * head_dim];
            let k_data = vec![0.2f32; batch_size * num_heads * seq_len * head_dim];
            let v_data = vec![0.3f32; batch_size * num_heads * seq_len * head_dim];

            let q_id = backend.create_persistent_buffer(&q_data)?;
            let k_id = backend.create_persistent_buffer(&k_data)?;
            let v_id = backend.create_persistent_buffer(&v_data)?;

            let output_id = backend.flash_attention_with_cache(
                &q_id, &k_id, &v_id, batch_size, *seq_len, *seq_len, num_heads, head_dim,
            )?;

            let output = backend.download_buffer_to_vec(&output_id)?;
            assert_eq!(output.len(), batch_size * num_heads * seq_len * head_dim);

            // Sanity check: output not all zeros
            let sum: f32 = output.iter().sum();
            assert!(
                sum.abs() > 0.001,
                "Output is all zeros for seq_len={}",
                seq_len
            );
        }

        Ok(())
    }
}
