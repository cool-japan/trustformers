//! Kernel Optimization Module
//!
//! Provides kernel-level optimizations for mobile platforms including:
//! - SIMD vectorization (NEON, AdvSIMD)
//! - GPU kernel optimization (Metal, Vulkan)
//! - Memory access pattern optimization
//! - Loop unrolling and tiling

use super::KernelType;
use crate::MobileBackend;
use std::collections::HashMap;
use trustformers_core::error::Result;

/// Kernel configuration
#[derive(Debug, Clone)]
pub struct KernelConfig {
    pub vectorization: VectorizationConfig,
    pub memory_layout: MemoryLayout,
    pub loop_optimization: LoopOptimization,
    pub precision: PrecisionConfig,
}

/// Vectorization configuration
#[derive(Debug, Clone)]
pub struct VectorizationConfig {
    pub enable_neon: bool,
    pub enable_sve: bool,
    pub vector_width: usize,
    pub unroll_factor: usize,
}

/// Memory layout optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLayout {
    RowMajor,
    ColumnMajor,
    Packed,
    Tiled { tile_m: usize, tile_n: usize },
}

/// Loop optimization strategies
#[derive(Debug, Clone)]
pub struct LoopOptimization {
    pub unroll_factor: usize,
    pub tile_size: usize,
    pub prefetch_distance: usize,
    pub enable_parallelization: bool,
}

/// Precision configuration
#[derive(Debug, Clone, Copy)]
pub struct PrecisionConfig {
    pub use_fp16: bool,
    pub use_bfloat16: bool,
    pub mixed_precision: bool,
}

/// Optimized kernel representation
#[derive(Debug, Clone)]
pub struct OptimizedKernel {
    pub kernel_type: KernelType,
    pub backend: MobileBackend,
    pub config: KernelConfig,
    pub estimated_speedup: f32,
    pub code: Option<String>,
}

/// Kernel optimizer
pub struct KernelOptimizer {
    backend: MobileBackend,
    device_capabilities: DeviceCapabilities,
    optimization_cache: HashMap<String, OptimizedKernel>,
}

/// Device capabilities
#[derive(Debug, Clone)]
struct DeviceCapabilities {
    has_neon: bool,
    has_sve: bool,
    has_fp16: bool,
    has_dot_product: bool,
    has_matrix_multiply: bool,
    simd_width: usize,
    l1_cache_size: usize,
    l2_cache_size: usize,
}

impl KernelOptimizer {
    /// Create new kernel optimizer for backend
    pub fn new(backend: MobileBackend) -> Self {
        let device_capabilities = Self::detect_capabilities(&backend);

        Self {
            backend,
            device_capabilities,
            optimization_cache: HashMap::new(),
        }
    }

    /// Optimize kernel for mobile execution
    pub fn optimize_kernel(
        &mut self,
        kernel: &KernelType,
        input_shapes: &[Vec<usize>],
        output_shape: &[usize],
    ) -> Result<OptimizedKernel> {
        // Check cache first
        let cache_key = format!("{:?}_{:?}_{:?}", kernel, input_shapes, output_shape);
        if let Some(cached) = self.optimization_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Select optimization strategy based on kernel type
        let optimized = match kernel {
            KernelType::Conv2d => self.optimize_conv2d(input_shapes, output_shape)?,
            KernelType::Linear => self.optimize_linear(input_shapes, output_shape)?,
            KernelType::BatchNorm => self.optimize_batchnorm(input_shapes, output_shape)?,
            KernelType::Attention => self.optimize_attention(input_shapes, output_shape)?,
            KernelType::Pooling => self.optimize_pooling(input_shapes, output_shape)?,
            KernelType::Activation => self.optimize_activation(input_shapes, output_shape)?,
            KernelType::Custom(name) => self.optimize_custom(name, input_shapes, output_shape)?,
        };

        // Cache the result
        self.optimization_cache.insert(cache_key, optimized.clone());

        Ok(optimized)
    }

    /// Detect device capabilities
    fn detect_capabilities(backend: &MobileBackend) -> DeviceCapabilities {
        match backend {
            MobileBackend::CPU => DeviceCapabilities {
                has_neon: cfg!(target_arch = "aarch64") || cfg!(target_arch = "arm"),
                has_sve: false, // SVE detection would be more complex
                has_fp16: cfg!(target_arch = "aarch64"),
                has_dot_product: cfg!(target_arch = "aarch64"),
                has_matrix_multiply: false,
                simd_width: if cfg!(target_arch = "aarch64") { 128 } else { 64 },
                l1_cache_size: 32 * 1024,  // 32KB typical
                l2_cache_size: 256 * 1024, // 256KB typical
            },
            MobileBackend::GPU => DeviceCapabilities {
                has_neon: false,
                has_sve: false,
                has_fp16: true,
                has_dot_product: true,
                has_matrix_multiply: true,
                simd_width: 32, // Warp/wavefront size
                l1_cache_size: 16 * 1024,
                l2_cache_size: 512 * 1024,
            },
            _ => DeviceCapabilities {
                has_neon: false,
                has_sve: false,
                has_fp16: false,
                has_dot_product: false,
                has_matrix_multiply: false,
                simd_width: 1,
                l1_cache_size: 32 * 1024,
                l2_cache_size: 256 * 1024,
            },
        }
    }

    /// Optimize Conv2D kernel
    fn optimize_conv2d(
        &self,
        input_shapes: &[Vec<usize>],
        output_shape: &[usize],
    ) -> Result<OptimizedKernel> {
        let config = self.create_conv2d_config(input_shapes, output_shape)?;

        let kernel = match self.backend {
            MobileBackend::CPU if self.device_capabilities.has_neon => {
                NeonKernel::create_conv2d(&config)?
            },
            MobileBackend::GPU => {
                if cfg!(target_os = "ios") {
                    MetalKernel::create_conv2d(&config)?
                } else {
                    VulkanKernel::create_conv2d(&config)?
                }
            },
            _ => self.create_generic_kernel(KernelType::Conv2d, config)?,
        };

        Ok(kernel)
    }

    /// Optimize Linear/Dense kernel
    fn optimize_linear(
        &self,
        input_shapes: &[Vec<usize>],
        output_shape: &[usize],
    ) -> Result<OptimizedKernel> {
        let config = self.create_linear_config(input_shapes, output_shape)?;

        let kernel = match self.backend {
            MobileBackend::CPU if self.device_capabilities.has_neon => {
                NeonKernel::create_gemm(&config)?
            },
            MobileBackend::GPU => {
                if cfg!(target_os = "ios") {
                    MetalKernel::create_gemm(&config)?
                } else {
                    VulkanKernel::create_gemm(&config)?
                }
            },
            _ => self.create_generic_kernel(KernelType::Linear, config)?,
        };

        Ok(kernel)
    }

    /// Optimize BatchNorm kernel
    fn optimize_batchnorm(
        &self,
        input_shapes: &[Vec<usize>],
        output_shape: &[usize],
    ) -> Result<OptimizedKernel> {
        let config = KernelConfig {
            vectorization: VectorizationConfig {
                enable_neon: self.device_capabilities.has_neon,
                enable_sve: false,
                vector_width: self.device_capabilities.simd_width / 32, // Elements per vector
                unroll_factor: 4,
            },
            memory_layout: MemoryLayout::Packed,
            loop_optimization: LoopOptimization {
                unroll_factor: 4,
                tile_size: 64,
                prefetch_distance: 2,
                enable_parallelization: true,
            },
            precision: PrecisionConfig {
                use_fp16: self.device_capabilities.has_fp16,
                use_bfloat16: false,
                mixed_precision: false,
            },
        };

        self.create_generic_kernel(KernelType::BatchNorm, config)
    }

    /// Optimize Attention kernel
    fn optimize_attention(
        &self,
        input_shapes: &[Vec<usize>],
        output_shape: &[usize],
    ) -> Result<OptimizedKernel> {
        // Attention benefits from matrix multiply optimizations
        let config = KernelConfig {
            vectorization: VectorizationConfig {
                enable_neon: self.device_capabilities.has_neon,
                enable_sve: false,
                vector_width: self.device_capabilities.simd_width / 32,
                unroll_factor: 2,
            },
            memory_layout: MemoryLayout::Tiled {
                tile_m: 64,
                tile_n: 64,
            },
            loop_optimization: LoopOptimization {
                unroll_factor: 2,
                tile_size: 64,
                prefetch_distance: 4,
                enable_parallelization: true,
            },
            precision: PrecisionConfig {
                use_fp16: self.device_capabilities.has_fp16,
                use_bfloat16: false,
                mixed_precision: true, // Use FP16 for compute, FP32 for accumulation
            },
        };

        self.create_generic_kernel(KernelType::Attention, config)
    }

    /// Optimize Pooling kernel
    fn optimize_pooling(
        &self,
        input_shapes: &[Vec<usize>],
        output_shape: &[usize],
    ) -> Result<OptimizedKernel> {
        let config = KernelConfig {
            vectorization: VectorizationConfig {
                enable_neon: self.device_capabilities.has_neon,
                enable_sve: false,
                vector_width: self.device_capabilities.simd_width / 32,
                unroll_factor: 2,
            },
            memory_layout: MemoryLayout::RowMajor,
            loop_optimization: LoopOptimization {
                unroll_factor: 2,
                tile_size: 32,
                prefetch_distance: 1,
                enable_parallelization: false,
            },
            precision: PrecisionConfig {
                use_fp16: self.device_capabilities.has_fp16,
                use_bfloat16: false,
                mixed_precision: false,
            },
        };

        self.create_generic_kernel(KernelType::Pooling, config)
    }

    /// Optimize Activation kernel
    fn optimize_activation(
        &self,
        input_shapes: &[Vec<usize>],
        output_shape: &[usize],
    ) -> Result<OptimizedKernel> {
        let config = KernelConfig {
            vectorization: VectorizationConfig {
                enable_neon: self.device_capabilities.has_neon,
                enable_sve: false,
                vector_width: self.device_capabilities.simd_width / 32,
                unroll_factor: 8, // Activations are memory-bound, so unroll more
            },
            memory_layout: MemoryLayout::Packed,
            loop_optimization: LoopOptimization {
                unroll_factor: 8,
                tile_size: 128,
                prefetch_distance: 2,
                enable_parallelization: true,
            },
            precision: PrecisionConfig {
                use_fp16: self.device_capabilities.has_fp16,
                use_bfloat16: false,
                mixed_precision: false,
            },
        };

        self.create_generic_kernel(KernelType::Activation, config)
    }

    /// Optimize custom kernel
    fn optimize_custom(
        &self,
        name: &str,
        input_shapes: &[Vec<usize>],
        output_shape: &[usize],
    ) -> Result<OptimizedKernel> {
        // Default optimization for custom kernels
        let config = KernelConfig {
            vectorization: VectorizationConfig {
                enable_neon: self.device_capabilities.has_neon,
                enable_sve: false,
                vector_width: self.device_capabilities.simd_width / 32,
                unroll_factor: 4,
            },
            memory_layout: MemoryLayout::RowMajor,
            loop_optimization: LoopOptimization {
                unroll_factor: 4,
                tile_size: 64,
                prefetch_distance: 2,
                enable_parallelization: true,
            },
            precision: PrecisionConfig {
                use_fp16: self.device_capabilities.has_fp16,
                use_bfloat16: false,
                mixed_precision: false,
            },
        };

        self.create_generic_kernel(KernelType::Custom(name.to_string()), config)
    }

    /// Create Conv2D-specific configuration
    fn create_conv2d_config(
        &self,
        input_shapes: &[Vec<usize>],
        output_shape: &[usize],
    ) -> Result<KernelConfig> {
        // Extract convolution parameters
        let batch_size = input_shapes[0][0];
        let channels_in = input_shapes[0][1];
        let height = input_shapes[0][2];
        let width = input_shapes[0][3];

        // Optimize tiling based on cache size
        let tile_size = self.compute_optimal_tile_size(height * width * 4)?; // 4 bytes per float

        Ok(KernelConfig {
            vectorization: VectorizationConfig {
                enable_neon: self.device_capabilities.has_neon,
                enable_sve: false,
                vector_width: self.device_capabilities.simd_width / 32,
                unroll_factor: 4,
            },
            memory_layout: MemoryLayout::Tiled {
                tile_m: tile_size,
                tile_n: tile_size,
            },
            loop_optimization: LoopOptimization {
                unroll_factor: 4,
                tile_size,
                prefetch_distance: 4,
                enable_parallelization: batch_size > 1,
            },
            precision: PrecisionConfig {
                use_fp16: self.device_capabilities.has_fp16,
                use_bfloat16: false,
                mixed_precision: true,
            },
        })
    }

    /// Create Linear/GEMM-specific configuration
    fn create_linear_config(
        &self,
        input_shapes: &[Vec<usize>],
        output_shape: &[usize],
    ) -> Result<KernelConfig> {
        let m = input_shapes[0][0]; // Batch size
        let k = input_shapes[0][1]; // Input features
        let n = output_shape[1]; // Output features

        // Use different strategies based on matrix sizes
        let (tile_m, tile_n) = if m * n * k < 1024 * 1024 {
            // Small matrices - use smaller tiles
            (32, 32)
        } else {
            // Large matrices - use larger tiles
            (64, 64)
        };

        Ok(KernelConfig {
            vectorization: VectorizationConfig {
                enable_neon: self.device_capabilities.has_neon,
                enable_sve: false,
                vector_width: self.device_capabilities.simd_width / 32,
                unroll_factor: 4,
            },
            memory_layout: MemoryLayout::Tiled { tile_m, tile_n },
            loop_optimization: LoopOptimization {
                unroll_factor: 4,
                tile_size: tile_m,
                prefetch_distance: 8,
                enable_parallelization: true,
            },
            precision: PrecisionConfig {
                use_fp16: self.device_capabilities.has_fp16,
                use_bfloat16: false,
                mixed_precision: true,
            },
        })
    }

    /// Create generic optimized kernel
    fn create_generic_kernel(
        &self,
        kernel_type: KernelType,
        config: KernelConfig,
    ) -> Result<OptimizedKernel> {
        let estimated_speedup = self.estimate_speedup(&config);

        Ok(OptimizedKernel {
            kernel_type,
            backend: self.backend,
            config,
            estimated_speedup,
            code: None, // Would contain actual kernel code
        })
    }

    /// Compute optimal tile size based on cache
    fn compute_optimal_tile_size(&self, data_size: usize) -> Result<usize> {
        let cache_size = self.device_capabilities.l1_cache_size;

        // Use 80% of L1 cache for tiles
        let available_cache = (cache_size as f32 * 0.8) as usize;

        // Find largest power-of-2 tile that fits
        let mut tile_size = 128;
        while tile_size * tile_size * 4 > available_cache && tile_size > 16 {
            tile_size /= 2;
        }

        Ok(tile_size)
    }

    /// Estimate speedup from optimizations
    fn estimate_speedup(&self, config: &KernelConfig) -> f32 {
        let mut speedup = 1.0;

        // Vectorization speedup
        if config.vectorization.enable_neon {
            speedup *= config.vectorization.vector_width as f32;
        }

        // Unrolling speedup (diminishing returns)
        speedup *= 1.0 + (config.loop_optimization.unroll_factor as f32).log2() * 0.2;

        // FP16 speedup
        if config.precision.use_fp16 {
            speedup *= 1.5; // Typical FP16 speedup
        }

        // Tiling speedup (cache efficiency)
        if let MemoryLayout::Tiled { .. } = config.memory_layout {
            speedup *= 1.3;
        }

        speedup
    }
}

/// SIMD kernel base trait
pub trait SimdKernel {
    fn create_conv2d(config: &KernelConfig) -> Result<OptimizedKernel>;
    fn create_gemm(config: &KernelConfig) -> Result<OptimizedKernel>;
    fn create_activation(config: &KernelConfig) -> Result<OptimizedKernel>;
}

/// NEON kernel implementation (ARM)
pub struct NeonKernel;

impl SimdKernel for NeonKernel {
    fn create_conv2d(config: &KernelConfig) -> Result<OptimizedKernel> {
        Ok(OptimizedKernel {
            kernel_type: KernelType::Conv2d,
            backend: MobileBackend::CPU,
            config: config.clone(),
            estimated_speedup: 4.0, // NEON provides ~4x speedup
            code: Some(Self::generate_neon_conv2d_code(config)),
        })
    }

    fn create_gemm(config: &KernelConfig) -> Result<OptimizedKernel> {
        Ok(OptimizedKernel {
            kernel_type: KernelType::Linear,
            backend: MobileBackend::CPU,
            config: config.clone(),
            estimated_speedup: 3.5,
            code: Some(Self::generate_neon_gemm_code(config)),
        })
    }

    fn create_activation(config: &KernelConfig) -> Result<OptimizedKernel> {
        Ok(OptimizedKernel {
            kernel_type: KernelType::Activation,
            backend: MobileBackend::CPU,
            config: config.clone(),
            estimated_speedup: 4.0,
            code: Some(Self::generate_neon_activation_code(config)),
        })
    }
}

impl NeonKernel {
    fn generate_neon_conv2d_code(config: &KernelConfig) -> String {
        // Simplified NEON intrinsics code generation
        format!(
            r#"
            // NEON-optimized Conv2D kernel
            void neon_conv2d(const float* input, const float* weights, float* output) {{
                // Use float32x4_t for NEON vectors
                const int vec_size = {};

                for (int i = 0; i < height; i += {}) {{
                    for (int j = 0; j < width; j += vec_size) {{
                        float32x4_t sum = vdupq_n_f32(0.0f);

                        // Convolution computation with NEON
                        // ... (actual implementation)

                        vst1q_f32(output + offset, sum);
                    }}
                }}
            }}
            "#,
            config.vectorization.vector_width, config.loop_optimization.unroll_factor
        )
    }

    fn generate_neon_gemm_code(config: &KernelConfig) -> String {
        format!(
            r#"
            // NEON-optimized GEMM kernel
            void neon_gemm(const float* A, const float* B, float* C, int M, int N, int K) {{
                // Tiled GEMM with NEON
                const int tile_m = {};
                const int tile_n = {};

                // Main computation loop
                // ... (actual implementation)
            }}
            "#,
            if let MemoryLayout::Tiled { tile_m, tile_n } = config.memory_layout {
                (tile_m, tile_n)
            } else {
                (64, 64)
            }
            .0,
            if let MemoryLayout::Tiled { tile_m, tile_n } = config.memory_layout {
                (tile_m, tile_n)
            } else {
                (64, 64)
            }
            .1
        )
    }

    fn generate_neon_activation_code(config: &KernelConfig) -> String {
        format!(
            r#"
            // NEON-optimized activation kernel
            void neon_relu(const float* input, float* output, int size) {{
                const float32x4_t zero = vdupq_n_f32(0.0f);

                for (int i = 0; i < size; i += {}) {{
                    float32x4_t x = vld1q_f32(input + i);
                    float32x4_t result = vmaxq_f32(x, zero);
                    vst1q_f32(output + i, result);
                }}
            }}
            "#,
            config.vectorization.vector_width * config.loop_optimization.unroll_factor
        )
    }
}

/// Vulkan kernel implementation (Android GPU)
pub struct VulkanKernel;

impl SimdKernel for VulkanKernel {
    fn create_conv2d(config: &KernelConfig) -> Result<OptimizedKernel> {
        Ok(OptimizedKernel {
            kernel_type: KernelType::Conv2d,
            backend: MobileBackend::GPU,
            config: config.clone(),
            estimated_speedup: 10.0, // GPU can provide significant speedup
            code: Some(Self::generate_vulkan_conv2d_shader(config)),
        })
    }

    fn create_gemm(config: &KernelConfig) -> Result<OptimizedKernel> {
        Ok(OptimizedKernel {
            kernel_type: KernelType::Linear,
            backend: MobileBackend::GPU,
            config: config.clone(),
            estimated_speedup: 8.0,
            code: Some(Self::generate_vulkan_gemm_shader(config)),
        })
    }

    fn create_activation(config: &KernelConfig) -> Result<OptimizedKernel> {
        Ok(OptimizedKernel {
            kernel_type: KernelType::Activation,
            backend: MobileBackend::GPU,
            config: config.clone(),
            estimated_speedup: 15.0,
            code: Some(Self::generate_vulkan_activation_shader(config)),
        })
    }
}

impl VulkanKernel {
    fn generate_vulkan_conv2d_shader(config: &KernelConfig) -> String {
        // Simplified Vulkan compute shader
        format!(
            r#"
            #version 450

            layout(local_size_x = {}, local_size_y = {}, local_size_z = 1) in;

            layout(binding = 0) readonly buffer Input {{ float data[]; }} input_buffer;
            layout(binding = 1) readonly buffer Weight {{ float data[]; }} weight_buffer;
            layout(binding = 2) writeonly buffer Output {{ float data[]; }} output_buffer;

            void main() {{
                // Compute shader for Conv2D
                // ... (actual implementation)
            }}
            "#,
            config.loop_optimization.tile_size / 4,
            config.loop_optimization.tile_size / 4
        )
    }

    fn generate_vulkan_gemm_shader(config: &KernelConfig) -> String {
        r#"
            #version 450

            // Optimized GEMM compute shader
            // ... (shader code)
            "#
        .to_string()
    }

    fn generate_vulkan_activation_shader(config: &KernelConfig) -> String {
        r#"
            #version 450

            // Simple activation compute shader
            // ... (shader code)
            "#
        .to_string()
    }
}

/// Metal kernel implementation (iOS GPU)
pub struct MetalKernel;

impl SimdKernel for MetalKernel {
    fn create_conv2d(config: &KernelConfig) -> Result<OptimizedKernel> {
        Ok(OptimizedKernel {
            kernel_type: KernelType::Conv2d,
            backend: MobileBackend::GPU,
            config: config.clone(),
            estimated_speedup: 12.0, // Metal is highly optimized for Apple hardware
            code: Some(Self::generate_metal_conv2d_kernel(config)),
        })
    }

    fn create_gemm(config: &KernelConfig) -> Result<OptimizedKernel> {
        Ok(OptimizedKernel {
            kernel_type: KernelType::Linear,
            backend: MobileBackend::GPU,
            config: config.clone(),
            estimated_speedup: 10.0,
            code: Some(Self::generate_metal_gemm_kernel(config)),
        })
    }

    fn create_activation(config: &KernelConfig) -> Result<OptimizedKernel> {
        Ok(OptimizedKernel {
            kernel_type: KernelType::Activation,
            backend: MobileBackend::GPU,
            config: config.clone(),
            estimated_speedup: 20.0,
            code: Some(Self::generate_metal_activation_kernel(config)),
        })
    }
}

impl MetalKernel {
    fn generate_metal_conv2d_kernel(config: &KernelConfig) -> String {
        r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void conv2d(
                device const float* input [[buffer(0)]],
                device const float* weights [[buffer(1)]],
                device float* output [[buffer(2)]],
                uint2 gid [[thread_position_in_grid]]
            ) {
                // Metal kernel for Conv2D
                // Optimized for Apple GPU architecture
                // ... (actual implementation)
            }
            "#
        .to_string()
    }

    fn generate_metal_gemm_kernel(config: &KernelConfig) -> String {
        r#"
            #include <metal_stdlib>
            using namespace metal;

            // Optimized GEMM kernel for Metal
            // Uses threadgroup memory for tiling
            // ... (kernel code)
            "#
        .to_string()
    }

    fn generate_metal_activation_kernel(config: &KernelConfig) -> String {
        r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void relu(
                device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                uint id [[thread_position_in_grid]]
            ) {
                output[id] = max(input[id], 0.0f);
            }
            "#
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_optimizer_creation() {
        let optimizer = KernelOptimizer::new(MobileBackend::CPU);
        assert_eq!(optimizer.backend, MobileBackend::CPU);
    }

    #[test]
    fn test_kernel_optimization() {
        let mut optimizer = KernelOptimizer::new(MobileBackend::CPU);

        let kernel = KernelType::Conv2d;
        let input_shapes = vec![vec![1, 3, 224, 224]];
        let output_shape = vec![1, 64, 112, 112];

        let optimized =
            optimizer.optimize_kernel(&kernel, &input_shapes, &output_shape[..]).unwrap();

        assert!(optimized.estimated_speedup >= 1.0);
        assert_eq!(optimized.kernel_type, KernelType::Conv2d);
    }

    #[test]
    fn test_kernel_config() {
        let config = KernelConfig {
            vectorization: VectorizationConfig {
                enable_neon: true,
                enable_sve: false,
                vector_width: 4,
                unroll_factor: 4,
            },
            memory_layout: MemoryLayout::Tiled {
                tile_m: 64,
                tile_n: 64,
            },
            loop_optimization: LoopOptimization {
                unroll_factor: 4,
                tile_size: 64,
                prefetch_distance: 4,
                enable_parallelization: true,
            },
            precision: PrecisionConfig {
                use_fp16: true,
                use_bfloat16: false,
                mixed_precision: true,
            },
        };

        assert!(config.vectorization.enable_neon);
        assert_eq!(config.loop_optimization.tile_size, 64);
    }
}
