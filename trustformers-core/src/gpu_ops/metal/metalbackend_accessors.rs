//! # MetalBackend - accessors Methods
//!
//! This module contains method implementations for `MetalBackend`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::common::*;
use super::metalbackend_type::MetalBackend;
use super::types::{BufferCache, BufferId};

impl MetalBackend {
    /// Get a persistent buffer by ID
    pub fn get_persistent_buffer(&self, id: &BufferId) -> Result<Arc<Buffer>> {
        let cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "get_persistent_buffer",
            )
        })?;
        cache.get(id).ok_or_else(|| {
            TrustformersError::hardware_error(
                &format!("Buffer {:?} not found in cache", id),
                "get_persistent_buffer",
            )
        })
    }

    /// Perform matrix multiplication with cached weight buffer
    /// This avoids transferring weight data on each forward pass
    pub fn matmul_with_cached_weight(
        &self,
        a: &[f32],
        weight_buffer_id: &BufferId,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        let b_buffer = self.get_persistent_buffer(weight_buffer_id)?;
        let a_buffer = self.create_buffer(a)?;
        let result_size = m * n;
        let c_buffer = self.device.new_buffer(
            (result_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&*self.matmul_pipeline);
        encoder.set_buffer(0, Some(&a_buffer), 0);
        encoder.set_buffer(1, Some(&*b_buffer), 0);
        encoder.set_buffer(2, Some(&c_buffer), 0);
        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_u32 = k as u32;
        encoder.set_bytes(
            3,
            mem::size_of::<u32>() as u64,
            &m_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            mem::size_of::<u32>() as u64,
            &n_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            mem::size_of::<u32>() as u64,
            &k_u32 as *const u32 as *const _,
        );
        let threadgroup_size = metal::MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        };
        let threadgroups = metal::MTLSize {
            width: (n as u64 + 15) / 16,
            height: (m as u64 + 15) / 16,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let result_ptr = c_buffer.contents() as *const f32;
        let result = unsafe { std::slice::from_raw_parts(result_ptr, result_size) }.to_vec();
        Ok(result)
    }
    /// Download a GPU buffer to CPU as a Vec<f32>
    pub fn download_buffer_to_vec(&self, buffer_id: &BufferId) -> Result<Vec<f32>> {
        let buffer = self.get_persistent_buffer(buffer_id)?;
        let size = buffer.length() as usize / mem::size_of::<f32>();
        let ptr = buffer.contents() as *const f32;
        let data_vec = unsafe { std::slice::from_raw_parts(ptr, size) }.to_vec();
        Ok(data_vec)
    }
    /// Insert a single head into reshaped buffer: [seq_len, head_dim] → [num_heads, seq_len, head_dim]
    /// Input is [:, :], inserted at [head_idx, :, :]
    pub fn insert_head_gpu(
        &self,
        heads_buffer_id: &BufferId,
        head_buffer_id: &BufferId,
        head_idx: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<()> {
        let head_size = seq_len * head_dim;
        let offset_elements = head_idx * head_size;
        let offset_bytes = offset_elements * mem::size_of::<f32>();
        let dst_buffer = self.get_persistent_buffer(heads_buffer_id)?;
        let src_buffer = self.get_persistent_buffer(head_buffer_id)?;
        let command_buffer = self.command_queue.new_command_buffer();
        let blit_encoder = command_buffer.new_blit_command_encoder();
        blit_encoder.copy_from_buffer(
            &*src_buffer,
            0,
            &*dst_buffer,
            offset_bytes as u64,
            (head_size * mem::size_of::<f32>()) as u64,
        );
        blit_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(())
    }
}
