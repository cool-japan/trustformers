// Copyright (c) 2024 TrustformeRS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Hardware acceleration integration for TrustformeRS
//!
//! This module provides a unified interface for hardware acceleration backends,
//! automatically selecting the best available acceleration method based on system
//! capabilities and user preferences.

#![allow(unused_variables)] // Multi-backend implementation with feature gates

#[allow(unused_imports)] // Used conditionally based on feature gates
use crate::errors::{acceleration_error, hardware_error, tensor_op_error, Result};
use crate::tensor::Tensor;
use std::sync::OnceLock;

/// Hardware acceleration backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccelerationBackend {
    /// NVIDIA CUDA
    Cuda,
    /// AMD ROCm
    Rocm,
    /// Intel OneAPI
    Intel,
    /// Vulkan Compute
    Vulkan,
    /// Apple Metal
    Metal,
    /// CPU fallback
    Cpu,
}

/// Hardware acceleration configuration
#[derive(Debug, Clone)]
pub struct AccelerationConfig {
    /// Preferred backend (if available)
    pub preferred_backend: Option<AccelerationBackend>,
    /// Enable automatic fallback to CPU
    pub auto_fallback: bool,
    /// Memory pool size per device (MB)
    pub memory_pool_size: usize,
    /// Enable kernel caching
    pub enable_kernel_cache: bool,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
}

impl Default for AccelerationConfig {
    fn default() -> Self {
        Self {
            preferred_backend: None,
            auto_fallback: true,
            memory_pool_size: 1024, // 1GB
            enable_kernel_cache: true,
            enable_monitoring: true,
        }
    }
}

/// Hardware acceleration manager
pub struct HardwareAccelerator {
    /// Active backend
    active_backend: AccelerationBackend,
    /// Configuration
    #[allow(dead_code)]
    config: AccelerationConfig,
    /// Performance statistics
    stats: AccelerationStats,
}

/// Performance statistics
#[derive(Debug, Clone, Default)]
pub struct AccelerationStats {
    /// Total operations executed
    pub total_operations: u64,
    /// Total execution time (ms)
    pub total_time_ms: f64,
    /// Memory allocated (bytes)
    pub memory_allocated: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
}

/// Global hardware accelerator instance
static ACCELERATOR: OnceLock<HardwareAccelerator> = OnceLock::new();

impl HardwareAccelerator {
    /// Initialize hardware accelerator with configuration
    pub fn initialize(config: AccelerationConfig) -> Result<&'static HardwareAccelerator> {
        ACCELERATOR.get_or_init(|| {
            Self::new(config).unwrap_or_else(|_| {
                // Fallback to CPU if initialization fails
                Self::new_cpu_fallback()
            })
        });
        Ok(ACCELERATOR.get().unwrap())
    }

    /// Get global hardware accelerator instance
    pub fn global() -> Result<&'static HardwareAccelerator> {
        ACCELERATOR.get().ok_or_else(|| {
            hardware_error("unknown", "Hardware accelerator not initialized")
                .with_operation("global")
                .with_suggestion("Call HardwareAccelerator::initialize() first")
        })
    }

    /// Create new hardware accelerator
    fn new(config: AccelerationConfig) -> Result<Self> {
        let backend = Self::select_backend(&config)?;

        // Initialize the selected backend
        match backend {
            AccelerationBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    crate::kernels::cuda_impl::api::init_cuda()?;
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(
                        acceleration_error("CUDA", "Support not compiled in this build")
                            .with_operation("initialization")
                            .with_suggestion("Rebuild with --features cuda to enable CUDA support"),
                    );
                }
            },
            AccelerationBackend::Rocm => {
                #[cfg(feature = "rocm")]
                {
                    crate::kernels::rocm_impl::api::init_rocm()?;
                }
                #[cfg(not(feature = "rocm"))]
                {
                    return Err(
                        acceleration_error("ROCm", "Support not compiled in this build")
                            .with_operation("initialization")
                            .with_suggestion("Rebuild with --features rocm to enable ROCm support"),
                    );
                }
            },
            AccelerationBackend::Intel => {
                #[cfg(feature = "intel")]
                {
                    crate::kernels::intel_impl::api::init_intel()?;
                }
                #[cfg(not(feature = "intel"))]
                {
                    return Err(acceleration_error(
                        "Intel OneAPI",
                        "Support not compiled in this build",
                    )
                    .with_operation("initialization")
                    .with_suggestion(
                        "Rebuild with --features intel to enable Intel OneAPI support",
                    ));
                }
            },
            AccelerationBackend::Vulkan => {
                #[cfg(feature = "vulkan")]
                {
                    // Vulkan backend initialization is handled in VulkanImpl::new()
                    let _vulkan = crate::kernels::vulkan_impl::VulkanImpl::new()?;
                }
                #[cfg(not(feature = "vulkan"))]
                {
                    return Err(
                        acceleration_error("Vulkan", "Support not compiled in this build")
                            .with_operation("initialization")
                            .with_suggestion(
                                "Rebuild with --features vulkan to enable Vulkan support",
                            ),
                    );
                }
            },
            AccelerationBackend::Metal => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    // Metal backend initialization using Metal Performance Shaders
                    let _metal = crate::kernels::metal_impl::MetalImpl::new()?;
                    log::info!(
                        "Metal backend initialized successfully for Apple Silicon acceleration"
                    );
                }
                #[cfg(not(all(target_os = "macos", feature = "metal")))]
                {
                    return Err(
                        acceleration_error("Metal", "Support not compiled in this build")
                            .with_operation("initialization")
                            .with_suggestion(
                                "Rebuild with --features metal to enable Metal support",
                            )
                            .with_suggestion("Metal backend requires macOS/iOS with Apple Silicon"),
                    );
                }
            },
            AccelerationBackend::Cpu => {
                // CPU backend is always available
            },
        }

        Ok(Self {
            active_backend: backend,
            config,
            stats: AccelerationStats::default(),
        })
    }

    /// Create CPU fallback accelerator
    fn new_cpu_fallback() -> Self {
        Self {
            active_backend: AccelerationBackend::Cpu,
            config: AccelerationConfig::default(),
            stats: AccelerationStats::default(),
        }
    }

    /// Select the best available backend
    fn select_backend(config: &AccelerationConfig) -> Result<AccelerationBackend> {
        // Try preferred backend first
        if let Some(preferred) = config.preferred_backend {
            if Self::is_backend_available(preferred) {
                return Ok(preferred);
            }
        }

        // Auto-select based on availability
        let backends = [
            AccelerationBackend::Cuda,
            AccelerationBackend::Rocm,
            AccelerationBackend::Intel,
            AccelerationBackend::Vulkan,
            AccelerationBackend::Metal,
            AccelerationBackend::Cpu,
        ];

        for backend in backends {
            if Self::is_backend_available(backend) {
                return Ok(backend);
            }
        }

        Err(
            hardware_error("system", "No acceleration backend available on this system")
                .with_operation("backend_selection")
                .with_suggestion("Install GPU drivers (NVIDIA CUDA, AMD ROCm, Intel OneAPI)")
                .with_suggestion("Ensure required features are enabled during compilation")
                .with_suggestion("CPU backend should always be available as fallback"),
        )
    }

    /// Check if backend is available
    fn is_backend_available(backend: AccelerationBackend) -> bool {
        match backend {
            AccelerationBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    crate::kernels::cuda_impl::api::init_cuda().is_ok()
                }
                #[cfg(not(feature = "cuda"))]
                {
                    false
                }
            },
            AccelerationBackend::Rocm => {
                #[cfg(feature = "rocm")]
                {
                    crate::kernels::rocm_impl::api::init_rocm().is_ok()
                }
                #[cfg(not(feature = "rocm"))]
                {
                    false
                }
            },
            AccelerationBackend::Intel => {
                #[cfg(feature = "intel")]
                {
                    crate::kernels::intel_impl::api::is_intel_available()
                }
                #[cfg(not(feature = "intel"))]
                {
                    false
                }
            },
            AccelerationBackend::Vulkan => {
                #[cfg(feature = "vulkan")]
                {
                    crate::kernels::vulkan_impl::VulkanImpl::new().is_ok()
                }
                #[cfg(not(feature = "vulkan"))]
                {
                    false
                }
            },
            AccelerationBackend::Metal => {
                // Check if Metal is available by attempting to create a Metal implementation
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    crate::kernels::metal_impl::MetalImpl::new().is_ok()
                }
                #[cfg(not(all(target_os = "macos", feature = "metal")))]
                {
                    false
                }
            },
            AccelerationBackend::Cpu => true,
        }
    }

    /// Execute matrix multiplication with hardware acceleration
    pub fn matmul(&mut self, a: &Tensor, b: &Tensor, c: &mut Tensor) -> Result<()> {
        let start_time = std::time::Instant::now();

        let result = match self.active_backend {
            AccelerationBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    crate::kernels::cuda_impl::api::cuda_matmul(a, b, c)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    self.cpu_matmul(a, b, c)
                }
            },
            AccelerationBackend::Rocm => {
                #[cfg(feature = "rocm")]
                {
                    crate::kernels::rocm_impl::api::rocm_matmul(a, b, c)
                }
                #[cfg(not(feature = "rocm"))]
                {
                    self.cpu_matmul(a, b, c)
                }
            },
            AccelerationBackend::Intel => {
                #[cfg(feature = "intel")]
                {
                    crate::kernels::intel_impl::api::intel_matmul(a, b, c)
                }
                #[cfg(not(feature = "intel"))]
                {
                    self.cpu_matmul(a, b, c)
                }
            },
            AccelerationBackend::Vulkan => {
                #[cfg(feature = "vulkan")]
                {
                    let mut vulkan = crate::kernels::vulkan_impl::VulkanImpl::new()?;
                    vulkan.matmul(a, b, c)
                }
                #[cfg(not(feature = "vulkan"))]
                {
                    self.cpu_matmul(a, b, c)
                }
            },
            AccelerationBackend::Metal => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    let metal_impl = crate::kernels::metal_impl::MetalImpl::new()?;
                    metal_impl.matrix_multiply(a, b).and_then(|result| {
                        // Copy result to output tensor c
                        match (c, &result) {
                            (Tensor::F32(c_arr), Tensor::F32(result_arr)) => {
                                c_arr.assign(result_arr);
                                Ok(())
                            },
                            _ => Err(tensor_op_error(
                                "Tensor type mismatch in Metal matmul",
                                "matmul",
                            )),
                        }
                    })
                }
                #[cfg(not(all(target_os = "macos", feature = "metal")))]
                {
                    self.cpu_matmul(a, b, c)
                }
            },
            AccelerationBackend::Cpu => self.cpu_matmul(a, b, c),
        };

        // Update statistics
        self.stats.total_operations += 1;
        self.stats.total_time_ms += start_time.elapsed().as_millis() as f64;

        result
    }

    /// Execute Flash Attention with hardware acceleration
    pub fn flash_attention(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        output: &mut Tensor,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();

        let result = match self.active_backend {
            AccelerationBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    crate::kernels::cuda_impl::api::cuda_flash_attention(query, key, value, output)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    self.cpu_flash_attention(query, key, value, output)
                }
            },
            AccelerationBackend::Rocm => {
                #[cfg(feature = "rocm")]
                {
                    crate::kernels::rocm_impl::api::rocm_flash_attention(query, key, value, output)
                }
                #[cfg(not(feature = "rocm"))]
                {
                    self.cpu_flash_attention(query, key, value, output)
                }
            },
            AccelerationBackend::Intel => {
                #[cfg(feature = "intel")]
                {
                    crate::kernels::intel_impl::api::intel_flash_attention(
                        query, key, value, output,
                    )
                }
                #[cfg(not(feature = "intel"))]
                {
                    self.cpu_flash_attention(query, key, value, output)
                }
            },
            AccelerationBackend::Vulkan => {
                #[cfg(feature = "vulkan")]
                {
                    let mut vulkan = crate::kernels::vulkan_impl::VulkanImpl::new()?;
                    let scale = 1.0 / (query.shape()[2] as f32).sqrt();
                    vulkan.flash_attention(query, key, value, output, scale)
                }
                #[cfg(not(feature = "vulkan"))]
                {
                    self.cpu_flash_attention(query, key, value, output)
                }
            },
            AccelerationBackend::Metal => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    let metal_impl = crate::kernels::metal_impl::MetalImpl::new()?;
                    metal_impl.flash_attention(query, key, value, output)
                }
                #[cfg(not(all(target_os = "macos", feature = "metal")))]
                {
                    self.cpu_flash_attention(query, key, value, output)
                }
            },
            AccelerationBackend::Cpu => self.cpu_flash_attention(query, key, value, output),
        };

        // Update statistics
        self.stats.total_operations += 1;
        self.stats.total_time_ms += start_time.elapsed().as_millis() as f64;

        result
    }

    /// CPU fallback for matrix multiplication
    fn cpu_matmul(&self, a: &Tensor, b: &Tensor, c: &mut Tensor) -> Result<()> {
        // Use the tensor's built-in matmul implementation
        let result = a.matmul(b)?;
        *c = result;
        Ok(())
    }

    /// CPU fallback for Flash Attention
    fn cpu_flash_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        output: &mut Tensor,
    ) -> Result<()> {
        // Simplified CPU implementation of Flash Attention
        let q_shape = query.shape();
        let batch_size = q_shape[0];
        let seq_len = q_shape[1];
        let head_dim = q_shape[2];

        // Compute attention scores: Q @ K^T
        let key_transposed = key.transpose(1, 2)?;
        let scores = query.matmul(&key_transposed)?;

        // Apply scaling
        let scale = 1.0 / (head_dim as f32).sqrt();
        let scaled_scores = scores.mul_scalar(scale)?;

        // Apply softmax
        let attention_weights = scaled_scores.softmax(2)?;

        // Apply attention to values: attention_weights @ V
        let result = attention_weights.matmul(value)?;

        *output = result;
        Ok(())
    }

    /// Get active backend
    pub fn active_backend(&self) -> AccelerationBackend {
        self.active_backend
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> &AccelerationStats {
        &self.stats
    }

    /// Reset performance statistics
    pub fn reset_stats(&mut self) {
        self.stats = AccelerationStats::default();
    }

    /// Get device information
    pub fn device_info(&self) -> Result<String> {
        match self.active_backend {
            AccelerationBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    crate::kernels::cuda_impl::api::cuda_device_info()
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Ok("CUDA not available".to_string())
                }
            },
            AccelerationBackend::Rocm => {
                #[cfg(feature = "rocm")]
                {
                    crate::kernels::rocm_impl::api::rocm_device_info()
                }
                #[cfg(not(feature = "rocm"))]
                {
                    Ok("ROCm not available".to_string())
                }
            },
            AccelerationBackend::Intel => {
                #[cfg(feature = "intel")]
                {
                    crate::kernels::intel_impl::api::intel_device_info()
                }
                #[cfg(not(feature = "intel"))]
                {
                    Ok("Intel OneAPI not available".to_string())
                }
            },
            AccelerationBackend::Cpu => Ok(format!("CPU: {} cores", num_cpus::get())),
            _ => Ok(format!("Backend: {:?}", self.active_backend)),
        }
    }

    /// Get memory statistics
    pub fn memory_stats(&self) -> Result<(usize, usize)> {
        match self.active_backend {
            AccelerationBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    crate::kernels::cuda_impl::api::cuda_memory_stats()
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Ok((0, 0))
                }
            },
            AccelerationBackend::Rocm => {
                #[cfg(feature = "rocm")]
                {
                    crate::kernels::rocm_impl::api::rocm_memory_stats()
                }
                #[cfg(not(feature = "rocm"))]
                {
                    Ok((0, 0))
                }
            },
            AccelerationBackend::Intel => {
                #[cfg(feature = "intel")]
                {
                    crate::kernels::intel_impl::api::intel_memory_stats()
                }
                #[cfg(not(feature = "intel"))]
                {
                    Ok((0, 0))
                }
            },
            AccelerationBackend::Cpu => {
                Ok((0, 0)) // CPU doesn't have dedicated memory pool
            },
            _ => Ok((0, 0)),
        }
    }
}

/// Public API for hardware acceleration
pub mod api {
    use super::*;

    /// Initialize hardware acceleration with default configuration
    pub fn init_hardware_acceleration() -> Result<()> {
        HardwareAccelerator::initialize(AccelerationConfig::default())?;
        Ok(())
    }

    /// Initialize hardware acceleration with custom configuration
    pub fn init_hardware_acceleration_with_config(config: AccelerationConfig) -> Result<()> {
        HardwareAccelerator::initialize(config)?;
        Ok(())
    }

    /// Execute accelerated matrix multiplication
    pub fn accelerated_matmul(a: &Tensor, b: &Tensor, c: &mut Tensor) -> Result<()> {
        let accelerator = HardwareAccelerator::global()?;

        // Since we can't get a mutable reference from the static,
        // we need to handle this differently for now
        let result = a.matmul(b)?;
        *c = result;
        Ok(())
    }

    /// Execute accelerated Flash Attention
    pub fn accelerated_flash_attention(
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        output: &mut Tensor,
    ) -> Result<()> {
        let accelerator = HardwareAccelerator::global()?;

        // Since we can't get a mutable reference from the static,
        // we need to handle this differently for now
        // Simplified CPU implementation of Flash Attention
        let q_shape = query.shape();
        let head_dim = q_shape[q_shape.len() - 1];

        // Compute attention scores: Q @ K^T
        let key_transposed = key.transpose(q_shape.len() - 2, q_shape.len() - 1)?;
        let scores = query.matmul(&key_transposed)?;

        // Apply scaling
        let scale = 1.0 / (head_dim as f32).sqrt();
        let scaled_scores = scores.mul_scalar(scale)?;

        // Apply softmax
        let attention_weights = scaled_scores.softmax((q_shape.len() - 1) as i32)?;

        // Apply attention to values: attention_weights @ V
        let result = attention_weights.matmul(value)?;

        *output = result;
        Ok(())
    }

    /// Get active acceleration backend
    pub fn get_active_backend() -> Result<AccelerationBackend> {
        Ok(HardwareAccelerator::global()?.active_backend())
    }

    /// Get device information
    pub fn get_device_info() -> Result<String> {
        HardwareAccelerator::global()?.device_info()
    }

    /// Get performance statistics
    pub fn get_performance_stats() -> Result<AccelerationStats> {
        Ok(HardwareAccelerator::global()?.get_stats().clone())
    }

    /// Get memory statistics
    pub fn get_memory_stats() -> Result<(usize, usize)> {
        HardwareAccelerator::global()?.memory_stats()
    }

    /// Check if a specific backend is available
    pub fn is_backend_available(backend: AccelerationBackend) -> bool {
        HardwareAccelerator::is_backend_available(backend)
    }

    /// List all available backends
    pub fn list_available_backends() -> Vec<AccelerationBackend> {
        let all_backends = [
            AccelerationBackend::Cuda,
            AccelerationBackend::Rocm,
            AccelerationBackend::Intel,
            AccelerationBackend::Vulkan,
            AccelerationBackend::Metal,
            AccelerationBackend::Cpu,
        ];

        all_backends
            .into_iter()
            .filter(|&backend| HardwareAccelerator::is_backend_available(backend))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_hardware_acceleration_initialization() {
        let config = AccelerationConfig::default();
        assert!(HardwareAccelerator::initialize(config).is_ok());
    }

    #[test]
    fn test_backend_selection() {
        let available = api::list_available_backends();
        assert!(!available.is_empty());
        assert!(available.contains(&AccelerationBackend::Cpu));
    }

    #[test]
    fn test_accelerated_matmul() {
        let _ = api::init_hardware_acceleration();

        let a = Tensor::ones(&[4, 4]).expect("Failed to create ones tensor");
        let b = Tensor::ones(&[4, 4]).expect("Failed to create ones tensor");
        let mut c = Tensor::zeros(&[4, 4]).expect("Failed to create zero tensor");

        let result = api::accelerated_matmul(&a, &b, &mut c);
        assert!(result.is_ok());

        // Result should be all 4s
        let data = c.data().unwrap();
        assert!(data.iter().all(|&x| (x - 4.0).abs() < 1e-6));
    }

    #[test]
    fn test_device_info() {
        let _ = api::init_hardware_acceleration();
        let info = api::get_device_info();
        assert!(info.is_ok());
    }

    #[test]
    fn test_performance_stats() {
        let _ = api::init_hardware_acceleration();
        let stats = api::get_performance_stats();
        assert!(stats.is_ok());
    }

    #[test]
    fn test_memory_stats() {
        let _ = api::init_hardware_acceleration();
        let stats = api::get_memory_stats();
        assert!(stats.is_ok());
    }
}
