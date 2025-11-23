//! # MetalBackend - gelu_f32_group Methods
//!
//! This module contains method implementations for `MetalBackend`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
use super::common::*;

use super::metalbackend_type::MetalBackend;

#[cfg(all(target_os = "macos", feature = "metal"))]
impl MetalBackend {
    /// Execute GELU activation on GPU
    /// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    pub fn gelu_f32(&self, input: &[f32]) -> Result<Vec<f32>> {
        let size = input.len();
        let input_buffer = self.create_buffer(input)?;
        let output_buffer = self.device.new_buffer(
            (size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&*self.gelu_pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        let size_u32 = size as u32;
        encoder.set_bytes(
            2,
            mem::size_of::<u32>() as u64,
            &size_u32 as *const u32 as *const _,
        );
        let threadgroup_size = metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let threadgroups = metal::MTLSize {
            width: (size as u64 + 255) / 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        // command_buffer.wait_until_completed(); // Async: Let GPU pipeline operations
        let result_ptr = output_buffer.contents() as *const f32;
        let result = unsafe { std::slice::from_raw_parts(result_ptr, size) }.to_vec();
        Ok(result)
    }
}
