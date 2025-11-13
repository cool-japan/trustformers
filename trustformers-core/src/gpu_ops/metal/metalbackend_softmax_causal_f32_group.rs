//! # MetalBackend - softmax_causal_f32_group Methods
//!
//! This module contains method implementations for `MetalBackend`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::common::*;
use super::metalbackend_type::MetalBackend;

impl MetalBackend {
    /// Execute Softmax with causal mask on GPU
    /// Applies causal mask: position i can only attend to j <= i
    /// Input/output shape: [seq_len, seq_len]
    pub fn softmax_causal_f32(&self, input: &[f32], seq_len: usize) -> Result<Vec<f32>> {
        let total_size = seq_len * seq_len;
        if input.len() != total_size {
            return Err(TrustformersError::shape_error(format!(
                "Input size {} doesn't match seq_len^2 {}",
                input.len(),
                total_size
            )));
        }
        let input_buffer = self.create_buffer(input)?;
        let output_buffer = self.device.new_buffer(
            (total_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&*self.softmax_causal_pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        let seq_len_u32 = seq_len as u32;
        encoder.set_bytes(
            2,
            mem::size_of::<u32>() as u64,
            &seq_len_u32 as *const u32 as *const _,
        );
        let threadgroup_size = metal::MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };
        let threadgroups = metal::MTLSize {
            width: (seq_len as u64 + 63) / 64,
            height: 1,
            depth: 1,
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
