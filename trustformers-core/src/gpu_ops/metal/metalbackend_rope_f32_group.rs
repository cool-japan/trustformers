//! # MetalBackend - rope_f32_group Methods
//!
//! This module contains method implementations for `MetalBackend`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::common::*;
use super::metalbackend_type::MetalBackend;

impl MetalBackend {
    /// Execute RoPE (Rotary Position Embedding) on GPU
    /// Rotates Q and K tensors for position encoding
    /// Input/output shape: [seq_len, num_heads, head_dim]
    pub fn rope_f32(
        &self,
        input: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        rotary_ndims: usize,
        base: f32,
    ) -> Result<Vec<f32>> {
        let total_size = seq_len * num_heads * head_dim;
        if input.len() != total_size {
            return Err(TrustformersError::shape_error(format!(
                "Input size {} doesn't match seq_len {} * num_heads {} * head_dim {}",
                input.len(),
                seq_len,
                num_heads,
                head_dim
            )));
        }
        let input_buffer = self.create_buffer(input)?;
        let output_buffer = self.device.new_buffer(
            (total_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&*self.rope_pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        let seq_len_u32 = seq_len as u32;
        let num_heads_u32 = num_heads as u32;
        let head_dim_u32 = head_dim as u32;
        let rotary_ndims_u32 = rotary_ndims as u32;
        encoder.set_bytes(
            2,
            mem::size_of::<u32>() as u64,
            &seq_len_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            3,
            mem::size_of::<u32>() as u64,
            &num_heads_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            mem::size_of::<u32>() as u64,
            &head_dim_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            mem::size_of::<u32>() as u64,
            &rotary_ndims_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            mem::size_of::<f32>() as u64,
            &base as *const f32 as *const _,
        );
        let threadgroup_size = metal::MTLSize {
            width: 8,
            height: 4,
            depth: 4,
        };
        let threadgroups = metal::MTLSize {
            width: ((rotary_ndims / 2) as u64 + 7) / 8,
            height: (num_heads as u64 + 3) / 4,
            depth: (seq_len as u64 + 3) / 4,
        };
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let result_ptr = output_buffer.contents() as *const f32;
        let result = unsafe { std::slice::from_raw_parts(result_ptr, total_size) }.to_vec();
        Ok(result)
    }
}
