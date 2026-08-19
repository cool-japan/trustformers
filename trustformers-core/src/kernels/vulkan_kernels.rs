use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Vulkan compute shader operations for cross-platform GPU acceleration
///
/// This module provides optimized Vulkan compute shaders for transformer operations,
/// offering broad hardware compatibility across vendors while maintaining high performance.
///
/// Features:
/// - Matrix multiplication with various precisions (FP32, FP16, BF16, INT8)
/// - Fused attention operations with memory-efficient implementations
/// - Element-wise operations with compute shader optimization
/// - Custom reduction operations using subgroup operations
/// - Cross-platform compatibility (NVIDIA, AMD, Intel, Mobile GPUs)
///
/// Vulkan kernel handle for managing GPU resources
pub struct VulkanKernel {
    /// Vulkan instance
    instance: Option<VulkanInstance>,
    /// Available GPU devices
    devices: Vec<VulkanDevice>,
    /// Memory pools for different devices
    memory_pools: HashMap<usize, Arc<Mutex<VulkanMemoryPool>>>,
    /// Shader cache for compiled compute shaders. Always empty: this module
    /// has no real compute pipeline wired up (see `matmul`'s docs above),
    /// so nothing ever compiles a shader to cache. Kept, like
    /// `CompiledShader` itself, as part of the shape a future real backend
    /// would populate.
    #[allow(dead_code)]
    shader_cache: HashMap<String, CompiledShader>,
    /// Command pools for different devices
    command_pools: HashMap<usize, VulkanCommandPool>,
}

/// Vulkan device information
#[derive(Debug, Clone)]
pub struct VulkanDevice {
    pub id: usize,
    pub name: String,
    pub vendor_id: u32,
    pub device_type: VulkanDeviceType,
    pub memory_total: u64,
    pub memory_free: u64,
    pub compute_queue_family: u32,
    pub max_workgroup_size: [u32; 3],
    pub max_workgroup_count: [u32; 3],
    pub max_workgroup_invocations: u32,
    pub subgroup_size: u32,
    pub supports_subgroup_ops: bool,
    pub supports_fp16: bool,
    pub supports_int8: bool,
    pub max_memory_allocation_size: u64,
    pub buffer_device_address: bool,
}

/// Vulkan device types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VulkanDeviceType {
    DiscreteGpu,
    IntegratedGpu,
    VirtualGpu,
    Cpu,
    Other,
}

/// Vulkan instance wrapper
#[derive(Debug)]
pub struct VulkanInstance {
    #[allow(dead_code)]
    device_id: usize,
    #[allow(dead_code)]
    logical_device: VulkanLogicalDevice,
    #[allow(dead_code)]
    queue: VulkanQueue,
}

/// Vulkan logical device
#[derive(Debug)]
pub struct VulkanLogicalDevice {
    #[allow(dead_code)]
    id: usize,
    #[allow(dead_code)]
    extensions: Vec<String>,
    #[allow(dead_code)]
    features: VulkanFeatures,
}

/// Vulkan device features
#[derive(Debug, Default)]
pub struct VulkanFeatures {
    pub compute_shader: bool,
    pub storage_buffer_16bit_access: bool,
    pub uniform_and_storage_buffer_16bit_access: bool,
    pub storage_push_constant_16: bool,
    pub storage_input_output_16: bool,
    pub storage_buffer_8bit_access: bool,
    pub uniform_and_storage_buffer_8bit_access: bool,
    pub storage_push_constant_8: bool,
    pub shader_float16: bool,
    pub shader_int8: bool,
    pub subgroup_vote: bool,
    pub subgroup_arithmetic: bool,
    pub subgroup_ballot: bool,
    pub subgroup_shuffle: bool,
    pub subgroup_shuffle_relative: bool,
    pub subgroup_clustered: bool,
    pub subgroup_quad: bool,
}

/// Vulkan compute queue
#[derive(Debug)]
pub struct VulkanQueue {
    #[allow(dead_code)]
    family_index: u32,
    #[allow(dead_code)]
    index: u32,
}

/// Memory pool for efficient GPU memory management
#[derive(Debug)]
pub struct VulkanMemoryPool {
    #[allow(dead_code)]
    device_id: usize,
    #[allow(dead_code)]
    allocated_blocks: HashMap<usize, VulkanMemoryBlock>,
    free_blocks: Vec<VulkanMemoryBlock>,
    total_allocated: u64,
    peak_allocated: u64,
    #[allow(dead_code)]
    memory_type_index: u32,
}

/// Vulkan memory block
#[derive(Debug, Clone)]
pub struct VulkanMemoryBlock {
    #[allow(dead_code)]
    ptr: usize,
    size: u64,
    #[allow(dead_code)]
    device_id: usize,
    #[allow(dead_code)]
    memory_type: VulkanMemoryType,
    #[allow(dead_code)]
    buffer: Option<VulkanBuffer>,
}

/// Vulkan memory types
#[derive(Debug, Clone, Copy)]
pub enum VulkanMemoryType {
    DeviceLocal,
    HostVisible,
    HostCoherent,
    HostCached,
}

/// Vulkan buffer wrapper
#[derive(Debug, Clone)]
pub struct VulkanBuffer {
    #[allow(dead_code)]
    id: usize,
    #[allow(dead_code)]
    size: u64,
    #[allow(dead_code)]
    usage: VulkanBufferUsage,
}

/// Vulkan buffer usage flags
#[derive(Debug, Clone, Copy, Default)]
pub struct VulkanBufferUsage {
    pub storage: bool,
    pub uniform: bool,
    pub transfer_src: bool,
    pub transfer_dst: bool,
}

/// Compiled Vulkan compute shader
#[derive(Debug, Clone)]
pub struct CompiledShader {
    #[allow(dead_code)]
    name: String,
    #[allow(dead_code)]
    spirv_code: Vec<u32>,
    #[allow(dead_code)]
    entry_point: String,
    #[allow(dead_code)]
    workgroup_size: [u32; 3],
    #[allow(dead_code)]
    push_constant_size: u32,
    #[allow(dead_code)]
    descriptor_set_layouts: Vec<VulkanDescriptorSetLayout>,
}

/// Vulkan descriptor set layout
#[derive(Debug, Clone)]
pub struct VulkanDescriptorSetLayout {
    #[allow(dead_code)]
    binding: u32,
    #[allow(dead_code)]
    descriptor_type: VulkanDescriptorType,
    #[allow(dead_code)]
    stage_flags: VulkanShaderStage,
}

/// Vulkan descriptor types
#[derive(Debug, Clone, Copy)]
pub enum VulkanDescriptorType {
    StorageBuffer,
    UniformBuffer,
    StorageImage,
    SampledImage,
}

/// Vulkan shader stages
#[derive(Debug, Clone, Copy)]
pub enum VulkanShaderStage {
    Compute,
}

/// Command pool for recording command buffers
#[derive(Debug)]
pub struct VulkanCommandPool {
    #[allow(dead_code)]
    device_id: usize,
    #[allow(dead_code)]
    queue_family: u32,
    #[allow(dead_code)]
    command_buffers: Vec<VulkanCommandBuffer>,
}

/// Vulkan command buffer
#[derive(Debug)]
pub struct VulkanCommandBuffer {
    #[allow(dead_code)]
    id: usize,
    #[allow(dead_code)]
    recording: bool,
}

/// Vulkan kernel configuration
#[derive(Debug, Clone)]
pub struct VulkanKernelConfig {
    pub workgroup_size: [u32; 3],
    pub workgroup_count: [u32; 3],
    pub push_constants: Vec<u8>,
    pub specialization_constants: HashMap<u32, u32>,
}

impl Default for VulkanKernelConfig {
    fn default() -> Self {
        Self {
            workgroup_size: [256, 1, 1],
            workgroup_count: [1, 1, 1],
            push_constants: Vec::new(),
            specialization_constants: HashMap::new(),
        }
    }
}

/// Precision types supported by Vulkan compute shaders
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VulkanPrecision {
    FP32,
    FP16,
    BF16,
    INT8,
    INT4,
}

impl VulkanKernel {
    /// Create new Vulkan kernel manager
    pub fn new() -> Result<Self> {
        let devices = Self::detect_devices()?;
        let memory_pools = HashMap::new();
        let shader_cache = HashMap::new();
        let command_pools = HashMap::new();

        Ok(Self {
            instance: None,
            devices,
            memory_pools,
            shader_cache,
            command_pools,
        })
    }

    /// Get list of available Vulkan devices
    pub fn enumerate_devices(&self) -> Result<Vec<VulkanDevice>> {
        Ok(self.devices.clone())
    }

    /// Initialize Vulkan for specific device
    pub fn initialize(&mut self, device_id: usize) -> Result<()> {
        let device = self.devices.iter().find(|d| d.id == device_id).ok_or_else(|| {
            TrustformersError::tensor_op_error(
                &format!("Vulkan device {} not found", device_id),
                "VulkanKernels::select_device",
            )
        })?;

        // Create logical device and queues
        let logical_device = VulkanLogicalDevice {
            id: device_id,
            extensions: vec![
                "VK_KHR_storage_buffer_storage_class".to_string(),
                "VK_KHR_16bit_storage".to_string(),
                "VK_KHR_8bit_storage".to_string(),
                "VK_KHR_shader_float16_int8".to_string(),
            ],
            features: VulkanFeatures {
                compute_shader: true,
                storage_buffer_16bit_access: device.supports_fp16,
                uniform_and_storage_buffer_16bit_access: device.supports_fp16,
                storage_buffer_8bit_access: device.supports_int8,
                uniform_and_storage_buffer_8bit_access: device.supports_int8,
                shader_float16: device.supports_fp16,
                shader_int8: device.supports_int8,
                subgroup_vote: device.supports_subgroup_ops,
                subgroup_arithmetic: device.supports_subgroup_ops,
                subgroup_ballot: device.supports_subgroup_ops,
                subgroup_shuffle: device.supports_subgroup_ops,
                subgroup_shuffle_relative: device.supports_subgroup_ops,
                subgroup_clustered: device.supports_subgroup_ops,
                subgroup_quad: device.supports_subgroup_ops,
                ..Default::default()
            },
        };

        let queue = VulkanQueue {
            family_index: device.compute_queue_family,
            index: 0,
        };

        self.instance = Some(VulkanInstance {
            device_id,
            logical_device,
            queue,
        });

        // Initialize memory pool
        let memory_pool = VulkanMemoryPool {
            device_id,
            allocated_blocks: HashMap::new(),
            free_blocks: Vec::new(),
            total_allocated: 0,
            peak_allocated: 0,
            memory_type_index: 0, // Device local memory
        };

        self.memory_pools.insert(device_id, Arc::new(Mutex::new(memory_pool)));

        // Initialize command pool
        let command_pool = VulkanCommandPool {
            device_id,
            queue_family: device.compute_queue_family,
            command_buffers: Vec::new(),
        };

        self.command_pools.insert(device_id, command_pool);

        Ok(())
    }

    /// Detect available Vulkan devices (internal method)
    fn detect_devices() -> Result<Vec<VulkanDevice>> {
        let mut devices = Vec::new();

        // Runtime device detection using vulkano
        // Attempt to enumerate actual Vulkan devices available on the system
        #[cfg(feature = "vulkan")]
        {
            use vulkano::device::physical::PhysicalDevice;
            use vulkano::instance::{Instance, InstanceCreateInfo};
            use vulkano::VulkanLibrary;

            // Try to initialize Vulkan and enumerate devices
            match VulkanLibrary::new() {
                Ok(library) => {
                    match Instance::new(library.clone(), InstanceCreateInfo::default()) {
                        Ok(instance) => {
                            // Enumerate physical devices
                            let physical_devices: Vec<Arc<PhysicalDevice>> =
                                match instance.enumerate_physical_devices() {
                                    Ok(devices) => devices.collect(),
                                    Err(_) => Vec::new(),
                                };
                            for (idx, physical_device) in physical_devices.iter().enumerate() {
                                let properties = physical_device.properties();
                                // Note: In Vulkano 0.35+, limits are part of properties directly
                                let limits = properties;

                                // Determine device type
                                let device_type = match properties.device_type {
                                    vulkano::device::physical::PhysicalDeviceType::DiscreteGpu => VulkanDeviceType::DiscreteGpu,
                                    vulkano::device::physical::PhysicalDeviceType::IntegratedGpu => VulkanDeviceType::IntegratedGpu,
                                    vulkano::device::physical::PhysicalDeviceType::VirtualGpu => VulkanDeviceType::VirtualGpu,
                                    vulkano::device::physical::PhysicalDeviceType::Cpu => VulkanDeviceType::Cpu,
                                    _ => VulkanDeviceType::Other,
                                };

                                // Get memory information
                                let memory_properties = physical_device.memory_properties();
                                let total_memory: u64 = memory_properties
                                    .memory_heaps
                                    .iter()
                                    .map(|heap| heap.size)
                                    .sum();

                                // Find compute queue family
                                // Note: In Vulkano 0.35+, QueueFlags uses contains() method
                                let compute_queue_family = physical_device
                                    .queue_family_properties()
                                    .iter()
                                    .position(|q| {
                                        q.queue_flags
                                            .intersects(vulkano::device::QueueFlags::COMPUTE)
                                    })
                                    .unwrap_or(0)
                                    as u32;

                                // Detect subgroup size based on vendor
                                let subgroup_size = match properties.vendor_id {
                                    0x10de => 32, // NVIDIA warp size
                                    0x1002 => 64, // AMD wavefront size
                                    0x8086 => 16, // Intel EU subgroup size
                                    _ => 32,      // Default
                                };

                                // `subgroup_size`/`supports_fp16`/`supports_int8` are queried
                                // from the real physical device rather than guessed from the
                                // PCI vendor ID or hardcoded `true`: a vendor ID only says who
                                // made the GPU, not what this specific model supports, and
                                // Vulkan 1.1 core already exposes the real values through
                                // `properties.subgroup_size` and `supported_features()`
                                // (`buffer_device_address` two lines below already did this
                                // correctly - the others should too).
                                let supported_features = physical_device.supported_features();

                                devices.push(VulkanDevice {
                                    id: idx,
                                    name: properties.device_name.clone(),
                                    vendor_id: properties.vendor_id,
                                    device_type,
                                    memory_total: total_memory,
                                    // vulkano's base API (no VK_EXT_memory_budget) cannot
                                    // report live free memory; report the real total and
                                    // leave `memory_free` at the same value rather than a
                                    // fabricated "90% free" guess.
                                    memory_free: total_memory,
                                    compute_queue_family,
                                    max_workgroup_size: limits.max_compute_work_group_size,
                                    max_workgroup_count: limits.max_compute_work_group_count,
                                    max_workgroup_invocations: limits
                                        .max_compute_work_group_invocations,
                                    subgroup_size: properties
                                        .subgroup_size
                                        .unwrap_or(subgroup_size),
                                    supports_subgroup_ops: properties.subgroup_size.is_some(),
                                    supports_fp16: supported_features.shader_float16,
                                    supports_int8: supported_features.shader_int8,
                                    max_memory_allocation_size: limits
                                        .max_memory_allocation_size
                                        .unwrap_or(u64::MAX),
                                    buffer_device_address: supported_features.buffer_device_address,
                                });
                            }
                        },
                        Err(e) => {
                            log::warn!("Failed to create Vulkan instance: {e}");
                        },
                    }
                },
                Err(e) => {
                    log::warn!("Failed to load Vulkan library: {e}");
                },
            }
        }

        // No mock/fabricated devices: if no real Vulkan device was found
        // (or the `vulkan` feature is off), honestly report zero devices
        // rather than inventing an NVIDIA GPU or (on Android/iOS) an ARM
        // Mali GPU that may not be the actual hardware present.
        Ok(devices)
    }

    /// Matrix multiplication.
    ///
    /// No real Vulkan compute pipeline is wired up in this module (see the
    /// module docs: `kernels/vulkan_impl.rs` has a working vulkano matmul
    /// shader). Every helper this used to call - `allocate_buffer`,
    /// `copy_to_buffer`/`copy_from_buffer`, `dispatch` - was a stub that
    /// returned `Ok(())` without touching real GPU memory, so this
    /// structurally looked like a complete GPU pipeline while never
    /// actually writing `result`. Returns a structured "not implemented"
    /// error instead.
    pub fn matmul(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        result: &mut Tensor,
        config: Option<VulkanKernelConfig>,
    ) -> Result<()> {
        let _ = config.unwrap_or_default();

        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(TrustformersError::tensor_op_error(
                "Matrix multiplication requires 2D tensors",
                "VulkanKernels::gemm",
            ));
        }

        if a_shape[1] != b_shape[0] {
            return Err(TrustformersError::tensor_op_error(
                "Matrix dimensions incompatible for multiplication",
                "VulkanKernels::gemm",
            ));
        }

        let expected_result_shape = [a_shape[0], b_shape[1]];
        if result.shape() != expected_result_shape {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "result shape {:?} must be {expected_result_shape:?}",
                    result.shape()
                ),
                "VulkanKernels::gemm",
            ));
        }

        self.instance.as_ref().ok_or_else(|| {
            TrustformersError::tensor_op_error("Vulkan not initialized", "VulkanKernels::gemm")
        })?;

        Err(TrustformersError::not_implemented(
            "VulkanKernel::matmul: no real compute pipeline is wired up in this module - use \
             kernels::vulkan_impl::VulkanImpl::matmul, which dispatches a real vulkano GLSL/ \
             SPIR-V shader"
                .to_string(),
        ))
    }

    /// Flash attention. No real compute pipeline is wired up (see `matmul`
    /// docs); returns a structured "not implemented" error instead of
    /// `Ok(())` with `output` left untouched.
    pub fn flash_attention(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        output: &mut Tensor,
        config: Option<VulkanKernelConfig>,
    ) -> Result<()> {
        let _ = config.unwrap_or_default();

        let q_shape = query.shape();
        if q_shape.len() != 3 {
            return Err(TrustformersError::tensor_op_error(
                "Flash attention requires 3D tensors",
                "VulkanKernels::flash_attention",
            ));
        }
        for (name, tensor) in [("key", key), ("value", value), ("output", &*output)] {
            if tensor.shape() != q_shape {
                return Err(TrustformersError::tensor_op_error(
                    &format!(
                        "{name} shape {:?} must match query shape {q_shape:?}",
                        tensor.shape()
                    ),
                    "VulkanKernels::flash_attention",
                ));
            }
        }

        Err(TrustformersError::not_implemented(
            "VulkanKernel::flash_attention: no real compute pipeline is wired up in this module"
                .to_string(),
        ))
    }

    /// Layer normalization. No real compute pipeline is wired up (see
    /// `matmul` docs); returns a structured "not implemented" error
    /// instead of `Ok(())` with `output` left untouched.
    pub fn layer_norm(
        &mut self,
        input: &Tensor,
        gamma: &Tensor,
        beta: Option<&Tensor>,
        output: &mut Tensor,
        epsilon: f32,
        // Reserved for the real backend's shader-variant selection. There is
        // no invariant to check without conflating precision (a compute
        // mode) with dtype (the tensor's storage format) - INT8 precision
        // computed from an F32-stored tensor is quantization, a legitimate,
        // common call, not a mismatch.
        _precision: VulkanPrecision,
    ) -> Result<()> {
        if epsilon <= 0.0 || !epsilon.is_finite() {
            return Err(TrustformersError::tensor_op_error(
                &format!("epsilon {epsilon} must be a finite positive number"),
                "VulkanKernels::layer_norm",
            ));
        }
        let input_shape = input.shape();
        if output.shape() != input_shape {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "output shape {:?} must match input shape {input_shape:?}",
                    output.shape()
                ),
                "VulkanKernels::layer_norm",
            ));
        }
        let Some(&feature_dim) = input_shape.last() else {
            return Err(TrustformersError::tensor_op_error(
                "input must have at least one dimension",
                "VulkanKernels::layer_norm",
            ));
        };
        let mut affine_params = vec![("gamma", gamma)];
        if let Some(beta) = beta {
            affine_params.push(("beta", beta));
        }
        for (name, tensor) in affine_params {
            if tensor.shape() != [feature_dim] {
                return Err(TrustformersError::tensor_op_error(
                    &format!(
                        "{name} shape {:?} must be a 1-D tensor of length {feature_dim} \
                         (input's last dimension)",
                        tensor.shape()
                    ),
                    "VulkanKernels::layer_norm",
                ));
            }
        }

        Err(TrustformersError::not_implemented(
            "VulkanKernel::layer_norm: no real compute pipeline is wired up in this module"
                .to_string(),
        ))
    }

    /// GELU activation. No real compute pipeline is wired up (see `matmul`
    /// docs); returns a structured "not implemented" error instead of
    /// `Ok(())` with `output` left untouched.
    pub fn gelu(
        &mut self,
        input: &Tensor,
        output: &mut Tensor,
        config: Option<VulkanKernelConfig>,
    ) -> Result<()> {
        let _ = config.unwrap_or_default();

        if output.shape() != input.shape() {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "output shape {:?} must match input shape {:?}",
                    output.shape(),
                    input.shape()
                ),
                "VulkanKernels::gelu",
            ));
        }

        Err(TrustformersError::not_implemented(
            "VulkanKernel::gelu: no real compute pipeline is wired up in this module".to_string(),
        ))
    }

    /// Reduce sum. No real compute pipeline is wired up (see `matmul`
    /// docs); returns a structured "not implemented" error instead of
    /// `Ok(())` with `output` left untouched.
    pub fn reduce_sum(
        &mut self,
        input: &Tensor,
        output: &mut Tensor,
        dim: usize,
        config: Option<VulkanKernelConfig>,
    ) -> Result<()> {
        let _ = config.unwrap_or_default();

        let input_shape = input.shape();
        if dim >= input_shape.len() {
            return Err(TrustformersError::tensor_op_error(
                "Reduction dimension out of bounds",
                "VulkanKernels::reduce",
            ));
        }
        let expected_shape: Vec<usize> = input_shape
            .iter()
            .enumerate()
            .filter(|(axis, _)| *axis != dim)
            .map(|(_, &size)| size)
            .collect();
        if output.shape() != expected_shape {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "output shape {:?} must be {expected_shape:?} (input {input_shape:?} with \
                     dim {dim} reduced away)",
                    output.shape()
                ),
                "VulkanKernels::reduce",
            ));
        }

        self.instance.as_ref().ok_or_else(|| {
            TrustformersError::tensor_op_error("Vulkan not initialized", "VulkanKernels::reduce")
        })?;

        Err(TrustformersError::not_implemented(
            "VulkanKernel::reduce_sum: no real compute pipeline is wired up in this module"
                .to_string(),
        ))
    }

    /// Get memory statistics
    pub fn get_memory_stats(&self, device_id: usize) -> Result<(u64, u64, u64)> {
        if let Some(pool) = self.memory_pools.get(&device_id) {
            let pool = pool.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let free_memory = pool.free_blocks.iter().map(|b| b.size).sum();
            Ok((pool.total_allocated, pool.peak_allocated, free_memory))
        } else {
            Ok((0, 0, 0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vulkan_kernel_creation() {
        let kernel = VulkanKernel::new();
        assert!(kernel.is_ok());
    }

    /// Regression test: `detect_devices` used to fabricate a fixed
    /// NVIDIA/ARM Mali device regardless of what hardware (if any) was
    /// actually attached. There is no real Vulkan device on this CI/dev
    /// host (and, without the `vulkan` feature, no runtime probe even
    /// runs), so honest enumeration must report zero - never a phantom
    /// device asserted as always present.
    #[test]
    fn test_device_enumeration_reports_no_phantom_devices_without_real_hardware() {
        let kernel = VulkanKernel::new().expect("operation failed in test");
        let devices = kernel.enumerate_devices().expect("operation failed in test");

        // Whatever is reported must be real: every entry must carry
        // plausible (non-fabricated-placeholder) properties.
        for device in &devices {
            assert!(!device.name.is_empty());
            assert!(device.max_workgroup_size[0] > 0);
        }

        #[cfg(not(feature = "vulkan"))]
        assert!(
            devices.is_empty(),
            "without the vulkan feature there is no real probe, so this must be empty, not a \
             fabricated device"
        );
    }

    #[test]
    fn test_vulkan_config_default() {
        let config = VulkanKernelConfig::default();
        assert_eq!(config.workgroup_size, [256, 1, 1]);
        assert_eq!(config.workgroup_count, [1, 1, 1]);
    }

    /// Regression test: before this fix, `matmul` (and
    /// flash_attention/layer_norm/gelu/reduce_sum) drove an entirely fake
    /// pipeline - `compile_matmul_shader` returned 1024 zero bytes as
    /// "SPIR-V", `allocate_buffer`/`copy_to_buffer`/`copy_from_buffer`/
    /// `dispatch` were all no-op `Ok(())` stubs - and reported success
    /// while never writing `result`. That fake-pipeline machinery
    /// (`compile_matmul_shader` et al.) no longer exists; every op must
    /// error honestly instead.
    #[test]
    fn test_matmul_errors_instead_of_faking_a_pipeline() {
        let mut kernel = VulkanKernel::new().expect("operation failed in test");
        let a = Tensor::ones(&[2, 3]).expect("tensor creation failed");
        let b = Tensor::ones(&[3, 4]).expect("tensor creation failed");
        let mut result = Tensor::zeros(&[2, 4]).expect("tensor creation failed");

        // `matmul` requires `initialize()` to have been called first (it
        // checks `self.instance`); with no real Vulkan device to
        // initialize against, or with the compute pipeline itself
        // unimplemented, this must error either way - never silently
        // leave `result` untouched while returning `Ok`.
        let result_status = kernel.matmul(&a, &b, &mut result, None);
        assert!(
            result_status.is_err(),
            "matmul must error rather than fabricate a completed GPU pipeline"
        );
    }

    #[test]
    fn test_precision_types() {
        assert_eq!(VulkanPrecision::FP32, VulkanPrecision::FP32);
        assert_ne!(VulkanPrecision::FP32, VulkanPrecision::FP16);
    }

    #[test]
    fn test_device_types() {
        assert_eq!(VulkanDeviceType::DiscreteGpu, VulkanDeviceType::DiscreteGpu);
        assert_ne!(
            VulkanDeviceType::DiscreteGpu,
            VulkanDeviceType::IntegratedGpu
        );
    }

    #[test]
    fn test_memory_pool_stats() {
        let kernel = VulkanKernel::new().expect("operation failed in test");
        let stats = kernel.get_memory_stats(0);
        assert!(stats.is_ok());

        // No pool is ever registered for a device without a real backing
        // Vulkan device, so the honest answer is all-zero stats, not a
        // fabricated nonzero pool. `total`/`peak`/`free` are `u64`, so the
        // commented-out `>= 0` checks this replaces were always vacuously
        // true and asserted nothing.
        let (total, peak, free) = stats.expect("operation failed in test");
        assert_eq!(
            (total, peak, free),
            (0, 0, 0),
            "no memory pool was ever registered for device 0 on this host"
        );
    }

    #[test]
    fn test_buffer_usage_flags() {
        let usage = VulkanBufferUsage {
            storage: true,
            uniform: false,
            transfer_src: true,
            transfer_dst: false,
        };

        assert!(usage.storage);
        assert!(!usage.uniform);
        assert!(usage.transfer_src);
        assert!(!usage.transfer_dst);
    }

    #[test]
    fn test_vulkan_features() {
        let features = VulkanFeatures {
            compute_shader: true,
            shader_float16: true,
            subgroup_vote: true,
            ..Default::default()
        };

        assert!(features.compute_shader);
        assert!(features.shader_float16);
        assert!(features.subgroup_vote);
        assert!(!features.storage_buffer_8bit_access);
    }

    /// Regression test: `matmul`'s new `result` shape check used to be an
    /// unread `result` parameter under the file's blanket
    /// `#![allow(unused_variables)]`. A wrong-shaped `result` must be
    /// rejected with a message naming the shape mismatch, distinct from
    /// both the pre-existing dimension checks and the generic "not
    /// initialized"/"not implemented" errors that follow it.
    #[test]
    fn matmul_rejects_a_wrong_result_shape() {
        let mut kernel = VulkanKernel::new().expect("operation failed in test");
        let a = Tensor::ones(&[2, 3]).expect("tensor creation failed");
        let b = Tensor::ones(&[3, 4]).expect("tensor creation failed");
        // Correct product shape is [2, 4]; this is deliberately wrong.
        let mut wrong_result = Tensor::zeros(&[2, 5]).expect("tensor creation failed");

        let err = kernel
            .matmul(&a, &b, &mut wrong_result, None)
            .expect_err("a mismatched result shape must be rejected");
        assert!(
            err.to_string().contains("result shape"),
            "error should name the result shape as the cause, got: {err}"
        );
    }

    /// Regression test: `flash_attention`'s `key`/`value`/`output` shape
    /// check used to be dead code - the parameters were threaded in and
    /// never read before falling straight through to the unconditional
    /// "not implemented" error. A shape mismatch must now be rejected with
    /// its own message rather than being silently accepted only to hit the
    /// same generic error a well-formed call would also hit.
    #[test]
    fn flash_attention_distinguishes_shape_errors_from_not_implemented() {
        let mut kernel = VulkanKernel::new().expect("operation failed in test");
        let query = Tensor::ones(&[1, 2, 4]).expect("tensor creation failed");
        let mismatched_key = Tensor::ones(&[1, 3, 4]).expect("tensor creation failed");
        let value = Tensor::ones(&[1, 2, 4]).expect("tensor creation failed");
        let mut output = Tensor::zeros(&[1, 2, 4]).expect("tensor creation failed");

        let shape_err = kernel
            .flash_attention(&query, &mismatched_key, &value, &mut output, None)
            .expect_err("a mismatched key shape must be rejected");
        assert!(
            shape_err.to_string().contains("key shape"),
            "error should name key's shape as the cause, got: {shape_err}"
        );

        // A well-formed call has nothing left to reject except the honestly
        // unimplemented compute pipeline.
        let matching_key = Tensor::ones(&[1, 2, 4]).expect("tensor creation failed");
        let not_implemented_err = kernel
            .flash_attention(&query, &matching_key, &value, &mut output, None)
            .expect_err("no compute pipeline is wired up yet");
        assert!(
            not_implemented_err.to_string().contains("wired up"),
            "a shape-correct call should fail on the unimplemented pipeline, not a shape check, \
             got: {not_implemented_err}"
        );
    }

    /// Regression test: `layer_norm`'s `epsilon`/`gamma`/`output` checks
    /// used to be dead code for the same reason as `flash_attention`
    /// above.
    #[test]
    fn layer_norm_rejects_bad_epsilon_and_gamma_shape() {
        let mut kernel = VulkanKernel::new().expect("operation failed in test");
        let input = Tensor::ones(&[2, 8]).expect("tensor creation failed");
        let gamma = Tensor::ones(&[8]).expect("tensor creation failed");
        let mut output = Tensor::zeros(&[2, 8]).expect("tensor creation failed");

        let eps_err = kernel
            .layer_norm(
                &input,
                &gamma,
                None,
                &mut output,
                0.0,
                VulkanPrecision::FP32,
            )
            .expect_err("a zero epsilon must be rejected");
        assert!(
            eps_err.to_string().contains("epsilon"),
            "error should name epsilon as the cause, got: {eps_err}"
        );

        let wrong_gamma = Tensor::ones(&[4]).expect("tensor creation failed"); // should be [8]
        let gamma_err = kernel
            .layer_norm(
                &input,
                &wrong_gamma,
                None,
                &mut output,
                1e-5,
                VulkanPrecision::FP32,
            )
            .expect_err("a mismatched gamma shape must be rejected");
        assert!(
            gamma_err.to_string().contains("gamma"),
            "error should name gamma as the cause, got: {gamma_err}"
        );
    }

    /// Regression test: `gelu`'s `output` shape check used to be dead code
    /// for the same reason as `flash_attention` above.
    #[test]
    fn gelu_rejects_an_output_shape_mismatch() {
        let mut kernel = VulkanKernel::new().expect("operation failed in test");
        let input = Tensor::ones(&[2, 8]).expect("tensor creation failed");
        let mut wrong_output = Tensor::zeros(&[2, 4]).expect("tensor creation failed");

        let err = kernel
            .gelu(&input, &mut wrong_output, None)
            .expect_err("a mismatched output shape must be rejected");
        assert!(
            err.to_string().contains("output shape"),
            "error should name the output shape as the cause, got: {err}"
        );
    }

    /// Regression test: `reduce_sum`'s `output` shape check used to be dead
    /// code for the same reason as `flash_attention` above.
    #[test]
    fn reduce_sum_rejects_an_output_shape_mismatch() {
        let mut kernel = VulkanKernel::new().expect("operation failed in test");
        let input = Tensor::ones(&[2, 8]).expect("tensor creation failed");
        let mut wrong_output = Tensor::zeros(&[8]).expect("tensor creation failed"); // should be [2]

        let err = kernel
            .reduce_sum(&input, &mut wrong_output, 1, None)
            .expect_err("a mismatched output shape must be rejected");
        assert!(
            err.to_string().contains("output shape"),
            "error should name the output shape as the cause, got: {err}"
        );
    }
}
