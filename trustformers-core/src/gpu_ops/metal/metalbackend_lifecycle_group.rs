//! # MetalBackend - GPU work synchronisation and buffer lifetime management
//!
//! Two responsibilities that used to be missing entirely:
//!
//! ## 1. Completion tracking
//!
//! Most kernel wrappers `commit()` their command buffer and return immediately so
//! the GPU can pipeline. That is correct for GPU-to-GPU chaining on a single
//! `MTLCommandQueue`, but it is *not* correct when the result has to leave that
//! queue:
//!
//! * a CPU readback (`download_buffer_to_vec`) previously read `MTLBuffer::contents`
//!   with no wait at all, so it could observe a partially written buffer; and
//! * `oxi_resident_gemm` hands the buffers to `oxicuda-metal`, which owns a
//!   **separate** `MTLCommandQueue` (`oxicuda-metal-0.5.5/src/device.rs:49`), and
//!   Metal orders nothing across queues.
//!
//! Every asynchronous commit now goes through `MetalBackend::commit_async`, which
//! records the command buffer. [`MetalBackend::flush`] waits on all recorded
//! buffers, and is called before any CPU readback and before every hand-off to
//! oxicuda's queue.
//!
//! ## 2. Buffer lifetimes
//!
//! Helpers for the byte-capped LRU cache in [`super::types`]: RAII handles, pinning,
//! explicit release of dead intermediates and occupancy statistics.

#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
use super::common::*;

use super::metalbackend_type::MetalBackend;
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
use super::types::{BufferCache, BufferCacheStats, BufferId, BufferTier, MetalBufferHandle};

/// Facts read from the live `MTLDevice`.
///
/// Every field comes from a Metal API call on the device this backend actually
/// opened - nothing here is a lookup table or a hardcoded constant. Consumers that
/// need to report hardware capabilities (pipelines, mobile engines) must source
/// them from here rather than inventing them.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalDeviceInfo {
    /// `MTLDevice.name`, e.g. "Apple M4 Pro".
    pub name: String,
    /// `MTLDevice.registryID` - stable identifier for this GPU.
    pub registry_id: u64,
    /// `MTLDevice.maxBufferLength` in bytes: the largest single allocation.
    pub max_buffer_length: usize,
    /// `MTLDevice.maxThreadsPerThreadgroup` (width, height, depth).
    pub max_threads_per_threadgroup: (usize, usize, usize),
    /// `MTLDevice.maxThreadgroupMemoryLength` in bytes (threadgroup/SRAM budget).
    pub max_threadgroup_memory_length: usize,
    /// `MTLDevice.recommendedMaxWorkingSetSize` in bytes.
    pub recommended_max_working_set_size: u64,
    /// `MTLDevice.currentAllocatedSize` in bytes at the time of the query.
    pub current_allocated_size: usize,
    /// `MTLDevice.hasUnifiedMemory` - true on Apple Silicon.
    pub has_unified_memory: bool,
    /// `MTLDevice.lowPower` - true for an integrated GPU on a dual-GPU Mac.
    pub is_low_power: bool,
    /// `MTLDevice.removable` - true for eGPUs.
    pub is_removable: bool,
    /// `MTLDevice.supportsRaytracing`.
    pub supports_raytracing: bool,
    /// Highest `MTLGPUFamily.Apple*` this device reports support for, if any.
    pub apple_gpu_family: Option<u32>,
    /// `supportsFamily(MTLGPUFamily.Metal3)`.
    pub supports_metal3: bool,
    /// Whether the Pure-Rust oxicuda-metal compute backend initialised successfully.
    pub oxicuda_metal_available: bool,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl MetalBackend {
    /// Query the live `MTLDevice` for its real capabilities.
    ///
    /// Replaces every hardcoded capability table: `supports_neural_engine: true`,
    /// `max_buffer_size: 4 GiB`, `unified_memory: true` and friends were reported
    /// unconditionally by the old pipeline backend regardless of hardware.
    ///
    /// Note what is deliberately *absent*: there is no Neural Engine field. Metal
    /// exposes no API for querying or dispatching to the ANE, so any such flag would
    /// be an invention; ANE access requires CoreML, which this backend does not use.
    pub fn device_info(&self) -> MetalDeviceInfo {
        let threads = self.device.max_threads_per_threadgroup();
        // Probe downward: `supportsFamily` is monotone within the Apple series, so the
        // first hit is the highest supported family.
        let apple_gpu_family = [
            (metal::MTLGPUFamily::Apple9, 9_u32),
            (metal::MTLGPUFamily::Apple8, 8),
            (metal::MTLGPUFamily::Apple7, 7),
            (metal::MTLGPUFamily::Apple6, 6),
            (metal::MTLGPUFamily::Apple5, 5),
            (metal::MTLGPUFamily::Apple4, 4),
            (metal::MTLGPUFamily::Apple3, 3),
            (metal::MTLGPUFamily::Apple2, 2),
            (metal::MTLGPUFamily::Apple1, 1),
        ]
        .into_iter()
        .find(|(family, _)| self.device.supports_family(*family))
        .map(|(_, generation)| generation);

        MetalDeviceInfo {
            name: self.device.name().to_string(),
            registry_id: self.device.registry_id(),
            max_buffer_length: self.device.max_buffer_length() as usize,
            max_threads_per_threadgroup: (
                threads.width as usize,
                threads.height as usize,
                threads.depth as usize,
            ),
            max_threadgroup_memory_length: self.device.max_threadgroup_memory_length() as usize,
            recommended_max_working_set_size: self.device.recommended_max_working_set_size(),
            current_allocated_size: self.device.current_allocated_size() as usize,
            has_unified_memory: self.device.has_unified_memory(),
            is_low_power: self.device.is_low_power(),
            is_removable: self.device.is_removable(),
            supports_raytracing: self.device.supports_raytracing(),
            apple_gpu_family,
            supports_metal3: self.device.supports_family(metal::MTLGPUFamily::Metal3),
            oxicuda_metal_available: self.mps_ops.is_some(),
        }
    }

    /// Commit a command buffer without blocking, recording it so [`flush`](Self::flush)
    /// can wait for it before the data crosses to the CPU or to another queue.
    pub(super) fn commit_async(&self, command_buffer: &metal::CommandBufferRef) {
        command_buffer.commit();
        match self.pending_command_buffers.lock() {
            Ok(mut pending) => {
                // Drop already-finished buffers so the list cannot grow without bound
                // during a long generation loop.
                pending.retain(|cb| {
                    !matches!(
                        cb.status(),
                        metal::MTLCommandBufferStatus::Completed
                            | metal::MTLCommandBufferStatus::Error
                    )
                });
                pending.push(command_buffer.to_owned());
            },
            Err(_) => {
                // The tracking mutex is poisoned: fall back to a synchronous wait so
                // correctness never depends on the bookkeeping succeeding.
                command_buffer.wait_until_completed();
            },
        }
    }

    /// Block until every command buffer this backend committed has completed.
    ///
    /// Waits on each recorded buffer individually rather than relying on
    /// commit-order completion, so it is correct even for command buffers that
    /// overlap in flight. Waiting on an already-completed buffer returns immediately.
    ///
    /// # Cost
    ///
    /// This is a full pipeline barrier and it is deliberately taken before every
    /// hand-off to oxicuda-metal's separate queue (`matmul_gpu_to_gpu_mps` and its
    /// scaled variant) and before every CPU readback. That serialises our kernels
    /// against each GEMM, giving up some overlap that the previous fire-and-forget
    /// code enjoyed - but the previous code was reading operands another queue might
    /// still be writing, and Metal orders nothing across queues, so the "overlap" was
    /// a race. If profiling later shows this barrier dominating, the correct
    /// replacement is a per-buffer `MTLSharedEvent` (signal on our queue, wait on
    /// oxicuda's) or an `oxicuda_metal::MetalBackend::from_queue(...)` constructor
    /// that shares this backend's queue - not removing the wait.
    pub fn flush(&self) -> Result<()> {
        let recorded = {
            let mut pending = self.pending_command_buffers.lock().map_err(|_| {
                TrustformersError::hardware_error(
                    "Failed to lock pending command buffer list",
                    "flush",
                )
            })?;
            std::mem::take(&mut *pending)
        };
        for command_buffer in &recorded {
            command_buffer.wait_until_completed();
        }
        Ok(())
    }

    /// Alias for [`flush`](Self::flush) with the name callers expect from other backends.
    pub fn synchronize(&self) -> Result<()> {
        self.flush()
    }

    /// Take a reference-counted handle to a cached buffer.
    ///
    /// While the handle (or any clone of it) is alive the entry is exempt from LRU
    /// eviction; when the last clone drops, the entry is removed and the `MTLBuffer`
    /// freed. This is the supported way to keep a GPU tensor resident across many
    /// ops without leaking it for the process lifetime.
    pub fn retain_buffer(&self, id: &BufferId) -> Result<MetalBufferHandle> {
        MetalBufferHandle::new(Arc::clone(&self.buffer_cache), *id)
    }

    /// Move a cached buffer into a different reclamation tier.
    ///
    /// Moving something to [`BufferTier::Evictable`] is a promise that the caller can
    /// regenerate its contents: the cache may drop it at any allocation once the byte
    /// cap is exceeded, and a later `get_persistent_buffer` on its id will fail.
    pub fn set_buffer_tier(&self, id: &BufferId, tier: BufferTier) -> Result<()> {
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "set_buffer_tier")
        })?;
        if !cache.set_tier(id, tier) {
            return Err(TrustformersError::hardware_error(
                &format!("Buffer {:?} not found in cache", id),
                "set_buffer_tier",
            ));
        }
        Ok(())
    }

    /// Reclamation tier of a cached buffer, if it is resident.
    pub fn buffer_tier(&self, id: &BufferId) -> Result<Option<BufferTier>> {
        let cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "buffer_tier")
        })?;
        Ok(cache.tier_of(id))
    }

    /// Release a batch of dead intermediates in one lock acquisition.
    ///
    /// Used by the composite ops (`attention_gpu_to_gpu`, ...) to free the
    /// intermediates that are provably dead once the final output exists. Unknown
    /// ids are ignored: releasing twice is not an error.
    pub fn release_buffers(&self, ids: &[BufferId]) -> Result<()> {
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "release_buffers")
        })?;
        for id in ids {
            cache.remove(id);
        }
        Ok(())
    }

    /// Whether `id` is still resident in the buffer cache.
    ///
    /// Useful after an LRU-pressured run: a transient id can legitimately have been
    /// evicted, and callers that need a guarantee should hold a
    /// [`MetalBufferHandle`] instead of probing.
    pub fn has_buffer(&self, id: &BufferId) -> Result<bool> {
        let cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "has_buffer")
        })?;
        Ok(cache.contains(id))
    }

    /// Occupancy and eviction counters for the buffer cache.
    pub fn buffer_cache_stats(&self) -> Result<BufferCacheStats> {
        let cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "buffer_cache_stats")
        })?;
        Ok(cache.stats())
    }

    /// Bytes currently held by the buffer cache.
    pub fn buffer_cache_live_bytes(&self) -> Result<usize> {
        Ok(self.buffer_cache_stats()?.live_bytes)
    }

    /// Validate the shapes handed to the fused scaled-matmul+softmax kernels.
    ///
    /// Both kernels are now unbounded in the sequence dimension (online softmax, no
    /// per-thread score array), so there is no artificial length limit left to
    /// enforce. What still has to hold is that the operand buffers are large enough
    /// for the declared shape and that no dimension is zero - otherwise the kernel
    /// indexes past the allocation, which is exactly the class of silent corruption
    /// the fixed-size `scores[256]` / `scores[512]` arrays used to cause.
    pub(super) fn validate_fused_attention_shapes(
        q_buffer: &Arc<Buffer>,
        k_t_buffer: &Arc<Buffer>,
        num_heads: usize,
        q_seq_len: usize,
        kv_seq_len: usize,
        head_dim: usize,
        op: &'static str,
    ) -> Result<()> {
        if num_heads == 0 || q_seq_len == 0 || kv_seq_len == 0 || head_dim == 0 {
            return Err(TrustformersError::shape_error(format!(
                "{op}: dimensions must be non-zero, got heads={num_heads} \
                 q_seq={q_seq_len} kv_seq={kv_seq_len} head_dim={head_dim}"
            )));
        }
        let elem = mem::size_of::<f32>();
        let q_needed = num_heads * q_seq_len * head_dim;
        let k_needed = num_heads * head_dim * kv_seq_len;
        let q_have = q_buffer.length() as usize / elem;
        let k_have = k_t_buffer.length() as usize / elem;
        if q_have < q_needed {
            return Err(TrustformersError::shape_error(format!(
                "{op}: Q buffer holds {q_have} floats but the declared shape \
                 [{num_heads}, {q_seq_len}, {head_dim}] needs {q_needed}"
            )));
        }
        if k_have < k_needed {
            return Err(TrustformersError::shape_error(format!(
                "{op}: K^T buffer holds {k_have} floats but the declared shape \
                 [{num_heads}, {head_dim}, {kv_seq_len}] needs {k_needed}"
            )));
        }
        // A single dispatch cannot exceed the device's maximum buffer length; the
        // attention-weight output is the largest allocation in the op.
        num_heads
            .checked_mul(q_seq_len)
            .and_then(|v| v.checked_mul(kv_seq_len))
            .and_then(|v| v.checked_mul(elem))
            .ok_or_else(|| {
                TrustformersError::shape_error(format!(
                    "{op}: attention-weight buffer size overflows usize for \
                     heads={num_heads} q_seq={q_seq_len} kv_seq={kv_seq_len}"
                ))
            })?;
        Ok(())
    }

    /// Reject a requested allocation that this Metal device cannot represent.
    pub(super) fn validate_allocation_bytes(&self, bytes: usize, op: &'static str) -> Result<()> {
        let max = self.device.max_buffer_length() as usize;
        if bytes > max {
            return Err(TrustformersError::hardware_error(
                &format!(
                    "{op}: requested {bytes} byte buffer exceeds this device's \
                     maxBufferLength of {max} bytes"
                ),
                op,
            ));
        }
        Ok(())
    }

    /// Change the cache byte cap at runtime, evicting immediately if already over it.
    pub fn set_buffer_cache_capacity_bytes(&self, capacity_bytes: usize) -> Result<()> {
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "set_buffer_cache_capacity_bytes",
            )
        })?;
        cache.set_capacity_bytes(capacity_bytes);
        Ok(())
    }
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
impl MetalBackend {
    /// The live `MTLDevice`, for tests that need to allocate an unusual buffer.
    pub(super) fn device_for_tests(&self) -> &MetalDevice {
        &self.device
    }

    /// Park an externally created buffer in the cache and return its id.
    pub(super) fn insert_buffer_for_tests(&self, buffer: Arc<Buffer>) -> Result<BufferId> {
        let id = BufferId::new();
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "insert_buffer_for_tests",
            )
        })?;
        cache.insert(id, buffer);
        Ok(id)
    }
}
