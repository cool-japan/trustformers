//! # MetalBackend - accessors Methods
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
    /// Get a persistent buffer by ID, refreshing its LRU position.
    pub fn get_persistent_buffer(&self, id: &BufferId) -> Result<Arc<Buffer>> {
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "get_persistent_buffer",
            )
        })?;
        cache.get(id).ok_or_else(|| {
            TrustformersError::hardware_error(
                &format!(
                    "Buffer {:?} not found in cache (it was released, or evicted under \
                     memory pressure - retain long-lived buffers with `retain_buffer`)",
                    id
                ),
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
        // The cached weight may still be the target of an in-flight async dispatch.
        self.flush()?;
        let a_buffer = self.create_buffer(a)?;
        let result_size = m * n;
        let c_buffer = self.device.new_buffer(
            (result_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.matmul_pipeline);
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
            width: (n as u64).div_ceil(16),
            height: (m as u64).div_ceil(16),
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
    /// Download a GPU buffer to CPU as a `Vec<f32>`.
    ///
    /// Two hazards this guards against, both of which used to be live:
    ///
    /// * **Reading before the GPU finished.** Kernel wrappers commit asynchronously,
    ///   so the producing dispatch may still be in flight. [`flush`](Self::flush)
    ///   waits for every outstanding command buffer first.
    /// * **GPU-private buffers.** `contents()` is documented to return null for
    ///   `StorageModePrivate`, and the old code dereferenced it unconditionally.
    ///   Rather than trust that (Apple Silicon's unified memory hands back a mapping
    ///   that reads as zeroes instead of null - silently wrong data, which is worse
    ///   than a crash), the storage mode itself is checked. `Private` and
    ///   `Memoryless` are refused; use
    ///   [`download_buffer_via_staging`](Self::download_buffer_via_staging) for those.
    pub fn download_buffer_to_vec(&self, buffer_id: &BufferId) -> Result<Vec<f32>> {
        let buffer = self.get_persistent_buffer(buffer_id)?;
        self.flush()?;
        let storage_mode = buffer.storage_mode();
        if matches!(
            storage_mode,
            metal::MTLStorageMode::Private | metal::MTLStorageMode::Memoryless
        ) {
            return Err(TrustformersError::hardware_error(
                &format!(
                    "Buffer {:?} is not CPU-mappable (storage mode {:?}): download it \
                     through `download_buffer_via_staging`, which blits into a \
                     StorageModeShared staging buffer",
                    buffer_id, storage_mode
                ),
                "download_buffer_to_vec",
            ));
        }
        let size = buffer.length() as usize / mem::size_of::<f32>();
        let ptr = buffer.contents() as *const f32;
        if ptr.is_null() {
            return Err(TrustformersError::hardware_error(
                &format!(
                    "Buffer {:?} reports storage mode {:?} but has no CPU mapping",
                    buffer_id, storage_mode
                ),
                "download_buffer_to_vec",
            ));
        }
        // SAFETY: `ptr` is the non-null CPU mapping of a Shared/Managed MTLBuffer of
        // `buffer.length()` bytes, and every committed dispatch that could still be
        // writing it has completed (see `flush` above). `size` is derived from that
        // same length, so the slice stays inside the allocation.
        let data_vec = unsafe { std::slice::from_raw_parts(ptr, size) }.to_vec();
        Ok(data_vec)
    }

    /// Download a GPU buffer that may live in `StorageModePrivate` memory.
    ///
    /// Blits `n_elems` floats into a CPU-mappable staging buffer and reads that,
    /// so it works for every storage mode. Prefer
    /// [`download_buffer_to_vec`](Self::download_buffer_to_vec) when the buffer is
    /// already Shared - this path costs an extra copy.
    pub fn download_buffer_via_staging(
        &self,
        buffer_id: &BufferId,
        n_elems: usize,
    ) -> Result<Vec<f32>> {
        let source = self.get_persistent_buffer(buffer_id)?;
        let bytes = (n_elems * mem::size_of::<f32>()) as u64;
        if bytes > source.length() {
            return Err(TrustformersError::shape_error(format!(
                "download_buffer_via_staging: requested {} bytes from a {} byte buffer",
                bytes,
                source.length()
            )));
        }
        // Every prior async dispatch must land before the blit reads the source.
        self.flush()?;
        let staging = self.device.new_buffer(bytes.max(1), MTLResourceOptions::StorageModeShared);
        let command_buffer = self.command_queue.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_buffer(&source, 0, &staging, 0, bytes);
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let ptr = staging.contents() as *const f32;
        if ptr.is_null() {
            return Err(TrustformersError::hardware_error(
                "Staging buffer has no CPU mapping",
                "download_buffer_via_staging",
            ));
        }
        // SAFETY: `staging` is Shared (CPU-mappable) and holds exactly `n_elems` f32
        // after the blit above completed.
        let data_vec = unsafe { std::slice::from_raw_parts(ptr, n_elems) }.to_vec();
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
            &src_buffer,
            0,
            &dst_buffer,
            offset_bytes as u64,
            (head_size * mem::size_of::<f32>()) as u64,
        );
        blit_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(())
    }
}
