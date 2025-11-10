//! Metal GPU backend for matrix operations
//!
//! This module provides Metal GPU acceleration for tensor operations on Apple Silicon.
//! Uses metal-rs for direct Metal API access.

use crate::device::Device;
use crate::errors::Result;
use crate::tensor::Tensor;

#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::errors::TrustformersError;
#[cfg(all(target_os = "macos", feature = "metal"))]
use metal::{Buffer, CommandQueue, CompileOptions, Device as MetalDevice, MTLResourceOptions};
#[cfg(all(target_os = "macos", feature = "metal"))]
use std::collections::HashMap;
#[cfg(all(target_os = "macos", feature = "metal"))]
use std::mem;
#[cfg(all(target_os = "macos", feature = "metal"))]
use std::sync::Arc;

/// Buffer ID for persistent GPU buffers
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(u64);

#[cfg(all(target_os = "macos", feature = "metal"))]
impl BufferId {
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        BufferId(COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl Default for BufferId {
    fn default() -> Self {
        Self::new()
    }
}

/// Persistent buffer cache for Metal GPU
#[cfg(all(target_os = "macos", feature = "metal"))]
struct BufferCache {
    buffers: HashMap<BufferId, Arc<Buffer>>,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl BufferCache {
    fn new() -> Self {
        Self {
            buffers: HashMap::new(),
        }
    }

    fn insert(&mut self, id: BufferId, buffer: Arc<Buffer>) {
        self.buffers.insert(id, buffer);
    }

    fn get(&self, id: &BufferId) -> Option<Arc<Buffer>> {
        self.buffers.get(id).cloned()
    }

    fn remove(&mut self, id: &BufferId) -> Option<Arc<Buffer>> {
        self.buffers.remove(id)
    }

    fn clear(&mut self) {
        self.buffers.clear();
    }

    fn len(&self) -> usize {
        self.buffers.len()
    }
}

/// Metal GPU backend for matrix multiplication
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct MetalBackend {
    device: MetalDevice,
    command_queue: CommandQueue,
    buffer_cache: Arc<std::sync::Mutex<BufferCache>>,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl MetalBackend {
    /// Create a new Metal backend
    pub fn new() -> Result<Self> {
        let device = MetalDevice::system_default().ok_or_else(|| {
            TrustformersError::hardware_error("No Metal device found", "MetalBackend::new")
        })?;

        let command_queue = device.new_command_queue();

        Ok(Self {
            device,
            command_queue,
            buffer_cache: Arc::new(std::sync::Mutex::new(BufferCache::new())),
        })
    }

    /// Create a persistent GPU buffer and return its ID
    pub fn create_persistent_buffer(&self, data: &[f32]) -> Result<BufferId> {
        let buffer = Arc::new(self.create_buffer(data)?);
        let buffer_id = BufferId::new();

        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "create_persistent_buffer",
            )
        })?;

        cache.insert(buffer_id, buffer);
        Ok(buffer_id)
    }

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

    /// Remove a persistent buffer from cache
    pub fn remove_persistent_buffer(&self, id: &BufferId) -> Result<()> {
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "remove_persistent_buffer",
            )
        })?;

        cache.remove(id);
        Ok(())
    }

    /// Clear all persistent buffers
    pub fn clear_buffer_cache(&self) -> Result<()> {
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "clear_buffer_cache")
        })?;

        cache.clear();
        Ok(())
    }

    /// Get the number of cached buffers
    pub fn buffer_cache_size(&self) -> Result<usize> {
        let cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "buffer_cache_size")
        })?;

        Ok(cache.len())
    }

    /// Perform matrix multiplication on Metal GPU
    pub fn matmul_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        // Create Metal buffers
        let a_buffer = self.create_buffer(a)?;
        let b_buffer = self.create_buffer(b)?;

        let result_size = m * n;
        let c_buffer = self.device.new_buffer(
            (result_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Metal shader for matrix multiplication
        let shader_source = r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void matmul(
                device const float* a [[buffer(0)]],
                device const float* b [[buffer(1)]],
                device float* c [[buffer(2)]],
                constant uint& M [[buffer(3)]],
                constant uint& N [[buffer(4)]],
                constant uint& K [[buffer(5)]],
                uint2 gid [[thread_position_in_grid]]
            ) {
                uint row = gid.y;
                uint col = gid.x;

                if (row >= M || col >= N) return;

                float sum = 0.0;
                for (uint i = 0; i < K; ++i) {
                    sum += a[row * K + i] * b[i * N + col];
                }
                c[row * N + col] = sum;
            }
        "#;

        // Compile shader
        let library = self
            .device
            .new_library_with_source(shader_source, &CompileOptions::new())
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to compile Metal shader: {}", e),
                    "matmul_f32",
                )
            })?;

        let kernel = library.get_function("matmul", None).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to get kernel function: {}", e),
                "matmul_f32",
            )
        })?;

        let pipeline =
            self.device.new_compute_pipeline_state_with_function(&kernel).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create pipeline: {}", e),
                    "matmul_f32",
                )
            })?;

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&a_buffer), 0);
        encoder.set_buffer(1, Some(&b_buffer), 0);
        encoder.set_buffer(2, Some(&c_buffer), 0);

        // Set dimensions as constants
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

        // Dispatch threads
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

        // Execute
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back to CPU
        let result_ptr = c_buffer.contents() as *const f32;
        let result = unsafe { std::slice::from_raw_parts(result_ptr, result_size) }.to_vec();

        Ok(result)
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
        // Get cached weight buffer
        let b_buffer = self.get_persistent_buffer(weight_buffer_id)?;

        // Create buffer for input (activations change on each forward pass)
        let a_buffer = self.create_buffer(a)?;

        let result_size = m * n;
        let c_buffer = self.device.new_buffer(
            (result_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Metal shader for matrix multiplication (same as matmul_f32)
        let shader_source = r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void matmul(
                device const float* a [[buffer(0)]],
                device const float* b [[buffer(1)]],
                device float* c [[buffer(2)]],
                constant uint& M [[buffer(3)]],
                constant uint& N [[buffer(4)]],
                constant uint& K [[buffer(5)]],
                uint2 gid [[thread_position_in_grid]]
            ) {
                uint row = gid.y;
                uint col = gid.x;

                if (row >= M || col >= N) return;

                float sum = 0.0;
                for (uint i = 0; i < K; ++i) {
                    sum += a[row * K + i] * b[i * N + col];
                }
                c[row * N + col] = sum;
            }
        "#;

        // Compile shader
        let library = self
            .device
            .new_library_with_source(shader_source, &CompileOptions::new())
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to compile Metal shader: {}", e),
                    "matmul_with_cached_weight",
                )
            })?;

        let kernel = library.get_function("matmul", None).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to get kernel function: {}", e),
                "matmul_with_cached_weight",
            )
        })?;

        let pipeline =
            self.device.new_compute_pipeline_state_with_function(&kernel).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create pipeline: {}", e),
                    "matmul_with_cached_weight",
                )
            })?;

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&a_buffer), 0);
        encoder.set_buffer(1, Some(&*b_buffer), 0); // Use cached weight buffer
        encoder.set_buffer(2, Some(&c_buffer), 0);

        // Set dimensions as constants
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

        // Dispatch threads
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

        // Execute
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back to CPU
        let result_ptr = c_buffer.contents() as *const f32;
        let result = unsafe { std::slice::from_raw_parts(result_ptr, result_size) }.to_vec();

        Ok(result)
    }

    fn create_buffer(&self, data: &[f32]) -> Result<Buffer> {
        let byte_size = std::mem::size_of_val(data) as u64;
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_size,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(buffer)
    }
}

/// Global Metal backend cache
#[cfg(all(target_os = "macos", feature = "metal"))]
static METAL_BACKEND: once_cell::sync::Lazy<std::sync::Mutex<Option<MetalBackend>>> =
    once_cell::sync::Lazy::new(|| std::sync::Mutex::new(None));

/// Get or create Metal backend instance
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn get_metal_backend() -> Result<MetalBackend> {
    let mut cache = METAL_BACKEND.lock().map_err(|_| {
        TrustformersError::hardware_error("Failed to lock Metal backend cache", "get_metal_backend")
    })?;

    if cache.is_none() {
        *cache = Some(MetalBackend::new()?);
    }

    // Clone the cached backend (device and command queue are Arc-based internally)
    cache
        .as_ref()
        .ok_or_else(|| {
            TrustformersError::hardware_error("Metal backend not initialized", "get_metal_backend")
        })
        .map(|backend| MetalBackend {
            device: backend.device.clone(),
            command_queue: backend.command_queue.clone(),
            buffer_cache: Arc::clone(&backend.buffer_cache),
        })
}

/// Dispatch matrix multiplication to appropriate backend based on device
#[allow(unused_variables)]
pub fn dispatch_matmul(a: &Tensor, b: &Tensor, device: &Device) -> Result<Tensor> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        if let Device::Metal(_device_id) = device {
            match (a, b) {
                (Tensor::F32(a_arr), Tensor::F32(b_arr)) => {
                    // Convert to 2D arrays
                    if a_arr.ndim() != 2 || b_arr.ndim() != 2 {
                        return Err(TrustformersError::shape_error(
                            "Metal dispatch currently only supports 2D tensors".to_string(),
                        ));
                    }

                    let a_2d = a_arr
                        .clone()
                        .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                        .map_err(|e| {
                            TrustformersError::shape_error(format!(
                                "Failed to convert to 2D: {}",
                                e
                            ))
                        })?;
                    let b_2d = b_arr
                        .clone()
                        .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                        .map_err(|e| {
                            TrustformersError::shape_error(format!(
                                "Failed to convert to 2D: {}",
                                e
                            ))
                        })?;

                    let (m, k) = a_2d.dim();
                    let (k2, n) = b_2d.dim();

                    if k != k2 {
                        return Err(TrustformersError::shape_error(format!(
                            "Matrix dimension mismatch: {}×{} vs {}×{}",
                            m, k, k2, n
                        )));
                    }

                    // Get Metal backend
                    let backend = get_metal_backend()?;

                    // Convert to contiguous slices
                    let a_data: Vec<f32> = a_2d.iter().copied().collect();
                    let b_data: Vec<f32> = b_2d.iter().copied().collect();

                    // Execute Metal matmul
                    let result_data = backend.matmul_f32(&a_data, &b_data, m, k, n)?;

                    // Convert back to tensor
                    let result_2d = scirs2_core::ndarray::Array2::from_shape_vec(
                        (m, n),
                        result_data,
                    )
                    .map_err(|e| {
                        TrustformersError::shape_error(format!("Failed to reshape result: {}", e))
                    })?;

                    let result_dyn = result_2d.into_dyn();
                    return Ok(Tensor::F32(result_dyn));
                },
                _ => {
                    // Fallback to CPU matmul for non-F32 tensors
                    return a.matmul(b);
                },
            }
        }
    }

    // Default: CPU matmul (or if Metal not available/configured)
    a.matmul(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_matmul_cpu() -> Result<()> {
        let a = Tensor::randn(&[2, 3])?;
        let b = Tensor::randn(&[3, 4])?;

        let c = dispatch_matmul(&a, &b, &Device::CPU)?;

        assert_eq!(c.shape(), &[2, 4]);
        Ok(())
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_dispatch_matmul_metal() -> Result<()> {
        let a = Tensor::randn(&[2, 3])?;
        let b = Tensor::randn(&[3, 4])?;

        let c = dispatch_matmul(&a, &b, &Device::Metal(0))?;

        assert_eq!(c.shape(), &[2, 4]);
        Ok(())
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_metal_backend_correctness() -> Result<()> {
        // Test that Metal matmul produces correct results
        let backend = MetalBackend::new()?;

        // Simple 2x2 matrix multiplication
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [[1, 2], [3, 4]]
        let b = vec![5.0, 6.0, 7.0, 8.0]; // [[5, 6], [7, 8]]

        let result = backend.matmul_f32(&a, &b, 2, 2, 2)?;

        // Expected: [[19, 22], [43, 50]]
        let expected = vec![19.0, 22.0, 43.0, 50.0];

        for (i, (&res, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-5,
                "Mismatch at index {}: {} vs {}",
                i,
                res,
                exp
            );
        }

        Ok(())
    }
}
