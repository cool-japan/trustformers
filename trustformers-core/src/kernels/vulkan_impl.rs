use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "vulkan")]
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage,
    },
    descriptor_set::{
        allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
        DescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
        QueueFlags,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    sync::{self, GpuFuture},
    VulkanLibrary,
};

/// Real Vulkan implementation using vulkano for cross-platform GPU acceleration
///
/// This module provides production-ready Vulkan compute shaders for transformer operations,
/// offering broad hardware compatibility across vendors while maintaining high performance.
///
/// Features:
/// - Matrix multiplication with various precisions (FP32, FP16, BF16, INT8)
/// - Fused attention operations with memory-efficient implementations
/// - Element-wise operations with compute shader optimization
/// - Custom reduction operations using subgroup operations
/// - Cross-platform compatibility (NVIDIA, AMD, Intel, Mobile GPUs)
pub struct VulkanImpl {
    #[cfg(feature = "vulkan")]
    #[allow(dead_code)]
    instance: Arc<Instance>,
    #[cfg(feature = "vulkan")]
    physical_device: Arc<PhysicalDevice>,
    #[cfg(feature = "vulkan")]
    device: Arc<Device>,
    #[cfg(feature = "vulkan")]
    queue: Arc<Queue>,
    #[cfg(feature = "vulkan")]
    memory_allocator: Arc<StandardMemoryAllocator>,
    #[cfg(feature = "vulkan")]
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    #[cfg(feature = "vulkan")]
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    #[cfg(feature = "vulkan")]
    compute_pipelines: HashMap<String, Arc<ComputePipeline>>,
    #[cfg(not(feature = "vulkan"))]
    _placeholder: (),
}

/// Device information extracted from Vulkan physical device
#[derive(Debug, Clone)]
pub struct VulkanDeviceInfo {
    pub name: String,
    pub device_type: String,
    pub vendor_id: u32,
    pub memory_total: u64,
    pub max_workgroup_size: [u32; 3],
    pub max_workgroup_count: [u32; 3],
    pub max_workgroup_invocations: u32,
    pub subgroup_size: u32,
    pub supports_subgroup_ops: bool,
    pub supports_fp16: bool,
    pub supports_int8: bool,
}

impl VulkanImpl {
    /// Create new Vulkan implementation
    pub fn new() -> Result<Self> {
        #[cfg(feature = "vulkan")]
        {
            Self::new_with_vulkano()
        }

        #[cfg(not(feature = "vulkan"))]
        {
            Ok(Self { _placeholder: () })
        }
    }

    #[cfg(feature = "vulkan")]
    fn new_with_vulkano() -> Result<Self> {
        // Create Vulkan instance
        let library = VulkanLibrary::new().map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to load Vulkan library: {}", e),
                "VulkanImpl::new",
            )
        })?;

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..Default::default()
            },
        )
        .map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to create Vulkan instance: {}", e),
                "VulkanImpl::new",
            )
        })?;

        // Select best physical device
        let physical_device = Self::select_best_device(&instance)?;

        // Create logical device and queue
        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_, q)| q.queue_flags.intersects(QueueFlags::COMPUTE))
            .ok_or_else(|| {
                TrustformersError::hardware_error(
                    "No compute queue family found",
                    "VulkanImpl::new",
                )
            })?;

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: queue_family_index as u32,
                    ..Default::default()
                }],
                enabled_extensions: DeviceExtensions {
                    khr_storage_buffer_storage_class: true,
                    khr_16bit_storage: true,
                    khr_8bit_storage: true,
                    khr_shader_float16_int8: true,
                    ..DeviceExtensions::empty()
                },
                enabled_features: DeviceFeatures {
                    shader_float16: true,
                    shader_int8: true,
                    storage_buffer16_bit_access: true,
                    uniform_and_storage_buffer16_bit_access: true,
                    storage_buffer8_bit_access: true,
                    uniform_and_storage_buffer8_bit_access: true,
                    ..DeviceFeatures::empty()
                },
                ..Default::default()
            },
        )
        .map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to create device: {}", e),
                "VulkanImpl::new",
            )
        })?;

        let queue = queues.next().ok_or_else(|| {
            TrustformersError::hardware_error("No queues available from device", "VulkanImpl::new")
        })?;

        // Create allocators
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        ));

        Ok(Self {
            instance,
            physical_device,
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
            compute_pipelines: HashMap::new(),
        })
    }

    #[cfg(feature = "vulkan")]
    fn select_best_device(instance: &Arc<Instance>) -> Result<Arc<PhysicalDevice>> {
        let physical_devices = instance.enumerate_physical_devices().map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to enumerate devices: {}", e),
                "VulkanImpl::select_best_device",
            )
        })?;

        // Prefer discrete GPU, then integrated GPU, then other types
        let best_device = physical_devices
            .filter(|d| {
                d.queue_family_properties()
                    .iter()
                    .any(|q| q.queue_flags.intersects(QueueFlags::COMPUTE))
            })
            .max_by_key(|d| match d.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 4,
                PhysicalDeviceType::IntegratedGpu => 3,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 1,
                PhysicalDeviceType::Other => 0,
                _ => 0, // Handle any future device types
            })
            .ok_or_else(|| {
                TrustformersError::hardware_error(
                    "No suitable Vulkan device found",
                    "VulkanImpl::select_best_device",
                )
            })?;

        Ok(best_device)
    }

    /// Get device information
    pub fn get_device_info(&self) -> Result<VulkanDeviceInfo> {
        #[cfg(feature = "vulkan")]
        {
            let props = self.physical_device.properties();

            Ok(VulkanDeviceInfo {
                name: props.device_name.clone(),
                device_type: format!("{:?}", props.device_type),
                vendor_id: props.vendor_id,
                memory_total: self
                    .physical_device
                    .memory_properties()
                    .memory_heaps
                    .iter()
                    .map(|heap| heap.size)
                    .max()
                    .unwrap_or(0),
                max_workgroup_size: props.max_compute_work_group_size,
                max_workgroup_count: props.max_compute_work_group_count,
                max_workgroup_invocations: props.max_compute_work_group_invocations,
                subgroup_size: props.subgroup_size.unwrap_or(32),
                supports_subgroup_ops: true, // Vulkan 1.1+ required
                supports_fp16: self.device.enabled_features().shader_float16,
                supports_int8: self.device.enabled_features().shader_int8,
            })
        }

        #[cfg(not(feature = "vulkan"))]
        {
            Err(TrustformersError::hardware_error(
                "Vulkan feature not enabled",
                "VulkanImpl::get_device_info",
            ))
        }
    }

    /// Matrix multiplication using Vulkan compute shaders
    pub fn matmul(&mut self, a: &Tensor, b: &Tensor, result: &mut Tensor) -> Result<()> {
        #[cfg(feature = "vulkan")]
        {
            let a_shape = a.shape();
            let b_shape = b.shape();

            if a_shape.len() != 2 || b_shape.len() != 2 {
                return Err(TrustformersError::tensor_op_error(
                    "Matrix multiplication requires 2D tensors",
                    "VulkanImpl::matmul",
                ));
            }

            if a_shape[1] != b_shape[0] {
                return Err(TrustformersError::tensor_op_error(
                    "Matrix dimensions incompatible for multiplication",
                    "VulkanImpl::matmul",
                ));
            }

            let m = a_shape[0];
            let k = a_shape[1];
            let n = b_shape[1];

            // Get or create compute pipeline
            let pipeline = self.get_or_create_matmul_pipeline()?;

            // Create buffers
            let a_data = a.data()?;
            let b_data = b.data()?;

            let a_buffer = Buffer::from_iter(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                a_data.iter().cloned(),
            )
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create A buffer: {}", e),
                    "VulkanImpl::matmul",
                )
            })?;

            let b_buffer = Buffer::from_iter(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                b_data.iter().cloned(),
            )
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create B buffer: {}", e),
                    "VulkanImpl::matmul",
                )
            })?;

            let result_buffer = Buffer::from_iter(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                (0..m * n).map(|_| 0.0f32),
            )
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create result buffer: {}", e),
                    "VulkanImpl::matmul",
                )
            })?;

            // Create descriptor set
            let layout = pipeline.layout().set_layouts().first().ok_or_else(|| {
                TrustformersError::hardware_error(
                    "Pipeline has no descriptor set layouts",
                    "VulkanImpl::matmul",
                )
            })?;
            let descriptor_set = DescriptorSet::new(
                self.descriptor_set_allocator.clone(),
                layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, a_buffer.clone()),
                    WriteDescriptorSet::buffer(1, b_buffer.clone()),
                    WriteDescriptorSet::buffer(2, result_buffer.clone()),
                ],
                [],
            )
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create descriptor set: {}", e),
                    "VulkanImpl::matmul",
                )
            })?;

            // Create and execute command buffer
            let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
                self.command_buffer_allocator.clone(),
                self.queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create command buffer: {}", e),
                    "VulkanImpl::matmul",
                )
            })?;

            // Push constants for matrix dimensions
            let push_constants = MatmulPushConstants {
                m: m as u32,
                k: k as u32,
                n: n as u32,
            };

            command_buffer_builder
                .bind_pipeline_compute(pipeline.clone())
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to bind pipeline: {}", e),
                        "VulkanImpl::matmul",
                    )
                })?
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pipeline.layout().clone(),
                    0,
                    descriptor_set,
                )
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to bind descriptor sets: {}", e),
                        "VulkanImpl::matmul",
                    )
                })?
                .push_constants(pipeline.layout().clone(), 0, push_constants)
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to push constants: {}", e),
                        "VulkanImpl::matmul",
                    )
                })?;

            // Safety: dispatch is safe when using valid workgroup sizes computed from tensor dimensions
            unsafe {
                command_buffer_builder
                    .dispatch([
                        n.div_ceil(16) as u32, // Workgroup size of 16x16
                        m.div_ceil(16) as u32,
                        1,
                    ])
                    .map_err(|e| {
                        TrustformersError::hardware_error(
                            &format!("Failed to dispatch: {}", e),
                            "VulkanImpl::matmul",
                        )
                    })?;
            }

            let command_buffer = command_buffer_builder.build().map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to build command buffer: {}", e),
                    "VulkanImpl::matmul",
                )
            })?;

            // Submit and wait
            let future = sync::now(self.device.clone())
                .then_execute(self.queue.clone(), command_buffer)
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to execute command buffer: {}", e),
                        "VulkanImpl::matmul",
                    )
                })?
                .then_signal_fence_and_flush()
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to signal fence: {}", e),
                        "VulkanImpl::matmul",
                    )
                })?;

            future.wait(None).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to wait for completion: {}", e),
                    "VulkanImpl::matmul",
                )
            })?;

            // Copy result back to CPU
            let content = result_buffer.read().map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to read result buffer: {}", e),
                    "VulkanImpl::matmul",
                )
            })?;

            // Replace result tensor with new data
            let result_shape = result.shape();
            *result = Tensor::from_vec(content.to_vec(), &result_shape)?;

            Ok(())
        }

        #[cfg(not(feature = "vulkan"))]
        {
            Err(TrustformersError::hardware_error(
                "Vulkan feature not enabled",
                "VulkanImpl::matmul",
            ))
        }
    }

    #[cfg(feature = "vulkan")]
    fn get_or_create_matmul_pipeline(&mut self) -> Result<Arc<ComputePipeline>> {
        const PIPELINE_NAME: &str = "matmul";

        if let Some(pipeline) = self.compute_pipelines.get(PIPELINE_NAME) {
            return Ok(pipeline.clone());
        }

        // Create the shader module
        let shader = matmul_cs::load(self.device.clone()).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to load shader: {}", e),
                "VulkanImpl::get_or_create_matmul_pipeline",
            )
        })?;

        // Get entry point from shader module
        let entry_point = shader.entry_point("main").ok_or_else(|| {
            TrustformersError::hardware_error(
                "Shader entry point 'main' not found",
                "VulkanImpl::get_or_create_matmul_pipeline",
            )
        })?;

        // Create pipeline layout
        let stage = PipelineShaderStageCreateInfo::new(entry_point);
        let layout = PipelineLayout::new(
            self.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(self.device.clone())
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to create pipeline layout: {}", e),
                        "VulkanImpl::get_or_create_matmul_pipeline",
                    )
                })?,
        )
        .map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to create pipeline layout: {}", e),
                "VulkanImpl::get_or_create_matmul_pipeline",
            )
        })?;

        // Create compute pipeline
        let pipeline = ComputePipeline::new(
            self.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to create pipeline: {}", e),
                "VulkanImpl::get_or_create_matmul_pipeline",
            )
        })?;

        self.compute_pipelines.insert(PIPELINE_NAME.to_string(), pipeline.clone());

        Ok(pipeline)
    }

    /// Flash attention using Vulkan compute shaders.
    ///
    /// No attention compute shader is wired up yet (unlike `matmul`, which
    /// has a real GLSL/SPIR-V shader - see `matmul_cs` below). This
    /// returns a structured "not implemented" error rather than `Ok(())`
    /// with `output` left untouched, which is what this used to do while
    /// still reporting success.
    pub fn flash_attention(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        output: &mut Tensor,
        scale: f32,
    ) -> Result<()> {
        #[cfg(feature = "vulkan")]
        {
            let q_shape = query.shape();
            if q_shape.len() != 3 {
                return Err(TrustformersError::tensor_op_error(
                    "Flash attention requires 3D tensors",
                    "VulkanImpl::flash_attention",
                ));
            }
            for (name, tensor) in [("key", key), ("value", value), ("output", &*output)] {
                if tensor.shape() != q_shape {
                    return Err(TrustformersError::tensor_op_error(
                        &format!(
                            "{name} shape {:?} must match query shape {q_shape:?}",
                            tensor.shape()
                        ),
                        "VulkanImpl::flash_attention",
                    ));
                }
            }
            if !scale.is_finite() {
                return Err(TrustformersError::tensor_op_error(
                    &format!("scale {scale} must be finite"),
                    "VulkanImpl::flash_attention",
                ));
            }

            Err(TrustformersError::not_implemented(
                "VulkanImpl::flash_attention: no attention compute shader is wired up yet (see \
                 VulkanImpl::matmul's matmul_cs shader for the pattern to follow)"
                    .to_string(),
            ))
        }

        #[cfg(not(feature = "vulkan"))]
        {
            Err(TrustformersError::hardware_error(
                "Vulkan feature not enabled",
                "VulkanImpl::flash_attention",
            ))
        }
    }

    /// Layer normalization using Vulkan compute shaders.
    ///
    /// No layer-norm compute shader is wired up yet; returns a structured
    /// "not implemented" error instead of `Ok(())` with `output` left
    /// untouched.
    pub fn layer_norm(
        &mut self,
        input: &Tensor,
        gamma: &Tensor,
        beta: Option<&Tensor>,
        output: &mut Tensor,
        epsilon: f32,
    ) -> Result<()> {
        #[cfg(feature = "vulkan")]
        {
            if epsilon <= 0.0 || !epsilon.is_finite() {
                return Err(TrustformersError::tensor_op_error(
                    &format!("epsilon {epsilon} must be a finite positive number"),
                    "VulkanImpl::layer_norm",
                ));
            }
            let input_shape = input.shape();
            if output.shape() != input_shape {
                return Err(TrustformersError::tensor_op_error(
                    &format!(
                        "output shape {:?} must match input shape {input_shape:?}",
                        output.shape()
                    ),
                    "VulkanImpl::layer_norm",
                ));
            }
            let Some(&feature_dim) = input_shape.last() else {
                return Err(TrustformersError::tensor_op_error(
                    "input must have at least one dimension",
                    "VulkanImpl::layer_norm",
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
                        "VulkanImpl::layer_norm",
                    ));
                }
            }

            Err(TrustformersError::not_implemented(
                "VulkanImpl::layer_norm: no layer-norm compute shader is wired up yet (see \
                 VulkanImpl::matmul's matmul_cs shader for the pattern to follow)"
                    .to_string(),
            ))
        }

        #[cfg(not(feature = "vulkan"))]
        {
            Err(TrustformersError::hardware_error(
                "Vulkan feature not enabled",
                "VulkanImpl::layer_norm",
            ))
        }
    }

    /// GELU activation using Vulkan compute shaders.
    ///
    /// No GELU compute shader is wired up yet; returns a structured "not
    /// implemented" error instead of `Ok(())` with `output` left untouched.
    pub fn gelu(&mut self, input: &Tensor, output: &mut Tensor) -> Result<()> {
        #[cfg(feature = "vulkan")]
        {
            if output.shape() != input.shape() {
                return Err(TrustformersError::tensor_op_error(
                    &format!(
                        "output shape {:?} must match input shape {:?}",
                        output.shape(),
                        input.shape()
                    ),
                    "VulkanImpl::gelu",
                ));
            }

            Err(TrustformersError::not_implemented(
                "VulkanImpl::gelu: no GELU compute shader is wired up yet (see \
                 VulkanImpl::matmul's matmul_cs shader for the pattern to follow)"
                    .to_string(),
            ))
        }

        #[cfg(not(feature = "vulkan"))]
        {
            Err(TrustformersError::hardware_error(
                "Vulkan feature not enabled",
                "VulkanImpl::gelu",
            ))
        }
    }

    /// Reduce sum using Vulkan compute shaders.
    ///
    /// No reduction compute shader is wired up yet; returns a structured
    /// "not implemented" error instead of `Ok(())` with `output` left
    /// untouched.
    pub fn reduce_sum(&mut self, input: &Tensor, output: &mut Tensor, dim: usize) -> Result<()> {
        #[cfg(feature = "vulkan")]
        {
            let input_shape = input.shape();
            if dim >= input_shape.len() {
                return Err(TrustformersError::tensor_op_error(
                    &format!(
                        "reduction dim {dim} is out of bounds for a {}-D input",
                        input_shape.len()
                    ),
                    "VulkanImpl::reduce_sum",
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
                        "output shape {:?} must be {expected_shape:?} (input {input_shape:?} \
                         with dim {dim} reduced away)",
                        output.shape()
                    ),
                    "VulkanImpl::reduce_sum",
                ));
            }

            Err(TrustformersError::not_implemented(
                "VulkanImpl::reduce_sum: no reduction compute shader is wired up yet (see \
                 VulkanImpl::matmul's matmul_cs shader for the pattern to follow)"
                    .to_string(),
            ))
        }

        #[cfg(not(feature = "vulkan"))]
        {
            Err(TrustformersError::hardware_error(
                "Vulkan feature not enabled",
                "VulkanImpl::reduce_sum",
            ))
        }
    }

    /// Get memory statistics
    pub fn get_memory_stats(&self) -> Result<(u64, u64, u64)> {
        #[cfg(feature = "vulkan")]
        {
            // In a real implementation, this would query Vulkan memory heaps
            // For now, return placeholder values
            Ok((0, 0, 0))
        }

        #[cfg(not(feature = "vulkan"))]
        {
            Ok((0, 0, 0))
        }
    }
}

#[cfg(feature = "vulkan")]
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulPushConstants {
    m: u32,
    k: u32,
    n: u32,
}

#[cfg(feature = "vulkan")]
#[allow(clippy::incompatible_msrv)] // Generated by vulkano_shaders macro
mod matmul_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 460

            layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

            layout(push_constant) uniform PushConstants {
                uint M;
                uint K;
                uint N;
            } pc;

            layout(set = 0, binding = 0) readonly buffer MatrixA {
                float data[];
            } matrix_a;

            layout(set = 0, binding = 1) readonly buffer MatrixB {
                float data[];
            } matrix_b;

            layout(set = 0, binding = 2) writeonly buffer MatrixC {
                float data[];
            } matrix_c;

            void main() {
                uint row = gl_GlobalInvocationID.y;
                uint col = gl_GlobalInvocationID.x;

                if (row >= pc.M || col >= pc.N) {
                    return;
                }

                float result = 0.0;
                for (uint k = 0; k < pc.K; k++) {
                    float a_val = matrix_a.data[row * pc.K + k];
                    float b_val = matrix_b.data[k * pc.N + col];
                    result += a_val * b_val;
                }

                matrix_c.data[row * pc.N + col] = result;
            }
        "
    }
}

// Note: VulkanImpl does not implement Default because Vulkan initialization
// can fail and we cannot create a meaningful fallback implementation.
// Use VulkanImpl::new() and handle the Result appropriately.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vulkan_impl_creation() {
        let result = VulkanImpl::new();

        #[cfg(feature = "vulkan")]
        {
            // If Vulkan is available, this should succeed
            if std::env::var("CI").is_err() {
                // Only test in non-CI environments where Vulkan might be available
                println!("Vulkan test result: {:?}", result.is_ok());
            }
        }

        #[cfg(not(feature = "vulkan"))]
        {
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_device_info() {
        if let Ok(vulkan) = VulkanImpl::new() {
            let info = vulkan.get_device_info();

            #[cfg(feature = "vulkan")]
            {
                if std::env::var("CI").is_err() {
                    println!("Device info result: {:?}", info);
                }
            }

            #[cfg(not(feature = "vulkan"))]
            {
                assert!(info.is_err());
            }
        }
    }

    #[test]
    fn test_matmul_basic() {
        if let Ok(mut vulkan) = VulkanImpl::new() {
            // Create test matrices
            let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])
                .expect("Tensor from_vec failed");
            let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])
                .expect("Tensor from_vec failed");
            let mut result = Tensor::zeros(&[2, 2]).expect("Failed to create zero tensor");

            let matmul_result = vulkan.matmul(&a, &b, &mut result);

            #[cfg(feature = "vulkan")]
            {
                if std::env::var("CI").is_err() {
                    println!("Matmul test result: {:?}", matmul_result);
                }
            }

            #[cfg(not(feature = "vulkan"))]
            {
                assert!(matmul_result.is_err());
            }
        }
    }

    #[test]
    fn test_memory_stats() {
        if let Ok(vulkan) = VulkanImpl::new() {
            let stats = vulkan.get_memory_stats();
            assert!(stats.is_ok());

            // `get_memory_stats` returns hardcoded placeholder zeros (no real
            // Vulkan memory-heap query is wired up yet). `total`/`peak`/`free`
            // are `u64`, so the commented-out `>= 0` checks this replaces were
            // always vacuously true and asserted nothing.
            let (total, peak, free) = stats.expect("operation failed in test");
            assert_eq!((total, peak, free), (0, 0, 0));
        }
    }

    /// Regression test: before this fix, `flash_attention`/`layer_norm`/
    /// `gelu`/`reduce_sum` returned `Ok(())` while leaving `output`
    /// completely untouched - a silent no-op reported as success. This
    /// asserts the pre-existing (and always correct) `not(feature =
    /// "vulkan")` behavior, so it runs unconditionally regardless of
    /// whether this host has a real Vulkan device.
    #[test]
    #[cfg(not(feature = "vulkan"))]
    fn test_unimplemented_ops_error_without_vulkan_feature() {
        let mut vulkan = VulkanImpl::new().expect("placeholder VulkanImpl always succeeds");
        let t = Tensor::zeros(&[1, 2, 2]).expect("tensor creation failed");
        let mut out = Tensor::zeros(&[1, 2, 2]).expect("tensor creation failed");
        assert!(vulkan.flash_attention(&t, &t, &t, &mut out, 1.0).is_err());
        assert!(vulkan.layer_norm(&t, &t, None, &mut out, 1e-5).is_err());
        assert!(vulkan.gelu(&t, &mut out).is_err());
        assert!(vulkan.reduce_sum(&t, &mut out, 0).is_err());
    }

    /// Regression test (only meaningfully exercised on a host with a real
    /// Vulkan device - it silently skips otherwise, matching this file's
    /// existing hardware-dependent tests): before this fix, these four ops
    /// returned `Ok(())` under `#[cfg(feature = "vulkan")]` without ever
    /// writing their output tensor. They must now error instead of
    /// fabricating a successful result.
    #[test]
    #[cfg(feature = "vulkan")]
    fn test_unimplemented_ops_error_with_real_vulkan_device() {
        if let Ok(mut vulkan) = VulkanImpl::new() {
            let q = Tensor::from_vec(vec![1.0; 8], &[1, 2, 4]).expect("tensor creation failed");
            let mut attn_out = Tensor::zeros(&[1, 2, 4]).expect("tensor creation failed");
            assert!(
                vulkan.flash_attention(&q, &q, &q, &mut attn_out, 1.0).is_err(),
                "flash_attention has no compute shader yet and must error, not silently no-op"
            );

            let input = Tensor::ones(&[2, 4]).expect("tensor creation failed");
            let gamma = Tensor::ones(&[4]).expect("tensor creation failed");
            let mut ln_out = Tensor::zeros(&[2, 4]).expect("tensor creation failed");
            assert!(
                vulkan.layer_norm(&input, &gamma, None, &mut ln_out, 1e-5).is_err(),
                "layer_norm has no compute shader yet and must error"
            );

            let mut gelu_out = Tensor::zeros(&[2, 4]).expect("tensor creation failed");
            assert!(
                vulkan.gelu(&input, &mut gelu_out).is_err(),
                "gelu has no compute shader yet and must error"
            );

            let mut sum_out = Tensor::zeros(&[2]).expect("tensor creation failed");
            assert!(
                vulkan.reduce_sum(&input, &mut sum_out, 1).is_err(),
                "reduce_sum has no compute shader yet and must error"
            );
        }
    }

    /// Regression test (only meaningfully exercised on a host with a real
    /// Vulkan device, matching the test above): the shape/parameter checks
    /// added to `flash_attention`/`layer_norm`/`gelu`/`reduce_sum` used to
    /// be unread parameters under this file's now-removed blanket
    /// `#![allow(unused_variables)]`. A malformed call must be rejected
    /// with a message that names the actual problem, not fall through to
    /// the same generic "not wired up" text a well-formed call also gets -
    /// otherwise a caller bug (e.g. a mismatched `key` shape) would be
    /// indistinguishable from "this backend isn't implemented yet".
    #[test]
    #[cfg(feature = "vulkan")]
    fn test_new_validation_errors_are_distinct_from_not_implemented() {
        if let Ok(mut vulkan) = VulkanImpl::new() {
            let q = Tensor::from_vec(vec![1.0; 8], &[1, 2, 4]).expect("tensor creation failed");
            let mismatched_key =
                Tensor::from_vec(vec![1.0; 12], &[1, 3, 4]).expect("tensor creation failed");
            let mut attn_out = Tensor::zeros(&[1, 2, 4]).expect("tensor creation failed");
            let shape_err = vulkan
                .flash_attention(&q, &mismatched_key, &q, &mut attn_out, 1.0)
                .expect_err("a mismatched key shape must be rejected");
            assert!(
                shape_err.to_string().contains("key shape"),
                "error should name key's shape as the cause, got: {shape_err}"
            );

            let non_finite_err = vulkan
                .flash_attention(&q, &q, &q, &mut attn_out, f32::NAN)
                .expect_err("a non-finite scale must be rejected");
            assert!(
                non_finite_err.to_string().contains("scale"),
                "error should name scale as the cause, got: {non_finite_err}"
            );

            let input = Tensor::ones(&[2, 4]).expect("tensor creation failed");
            let gamma = Tensor::ones(&[4]).expect("tensor creation failed");
            let mut ln_out = Tensor::zeros(&[2, 4]).expect("tensor creation failed");
            let eps_err = vulkan
                .layer_norm(&input, &gamma, None, &mut ln_out, -1.0)
                .expect_err("a negative epsilon must be rejected");
            assert!(
                eps_err.to_string().contains("epsilon"),
                "error should name epsilon as the cause, got: {eps_err}"
            );

            let mut wrong_gelu_out = Tensor::zeros(&[2, 3]).expect("tensor creation failed");
            let gelu_err = vulkan
                .gelu(&input, &mut wrong_gelu_out)
                .expect_err("a mismatched output shape must be rejected");
            assert!(
                gelu_err.to_string().contains("output shape"),
                "error should name the output shape as the cause, got: {gelu_err}"
            );

            let mut wrong_sum_out = Tensor::zeros(&[4]).expect("tensor creation failed");
            let sum_err = vulkan
                .reduce_sum(&input, &mut wrong_sum_out, 1)
                .expect_err("a mismatched output shape must be rejected");
            assert!(
                sum_err.to_string().contains("output shape"),
                "error should name the output shape as the cause, got: {sum_err}"
            );
        }
    }
}
