//! WebGPU type definitions with fallbacks
//!
//! This module provides WebGPU type fallbacks since WebGPU types are not available in web-sys v0.3.81.
//! Once WebGPU types become available in web-sys, these can be replaced with actual types.

// Fallback types - WebGPU types are not available in current web-sys version
pub type Gpu = js_sys::Object;
pub type GpuAdapter = js_sys::Object;
pub type GpuBuffer = js_sys::Object;
pub type GpuBufferDescriptor = js_sys::Object;
pub type GpuBufferUsage = u32;
pub type GpuBindGroup = js_sys::Object;
pub type GpuBindGroupDescriptor = js_sys::Object;
pub type GpuBindGroupEntry = js_sys::Object;
pub type GpuBindGroupLayout = js_sys::Object;
pub type GpuCommandEncoder = js_sys::Object;
pub type GpuComputePassEncoder = js_sys::Object;
pub type GpuComputePipeline = js_sys::Object;
pub type GpuComputePipelineDescriptor = js_sys::Object;
pub type GpuDevice = js_sys::Object;
pub type GpuExtent3dDict = js_sys::Object;
pub type GpuProgrammableStage = js_sys::Object;
pub type GpuQueue = js_sys::Object;
pub type GpuRequestAdapterOptions = js_sys::Object;
pub type GpuShaderModule = js_sys::Object;
pub type GpuShaderModuleDescriptor = js_sys::Object;
pub type GpuTexture = js_sys::Object;
pub type GpuTextureDescriptor = js_sys::Object;
pub type GpuTextureDimension = js_sys::Object;
pub type GpuTextureFormat = js_sys::Object;
pub type GpuTextureUsage = u32;

// Fallback constants for WebGPU usage flags
pub mod buffer_usage {
    pub const MAP_READ: u32 = 0x0001;
    pub const MAP_WRITE: u32 = 0x0002;
    pub const COPY_SRC: u32 = 0x0004;
    pub const COPY_DST: u32 = 0x0008;
    pub const INDEX: u32 = 0x0010;
    pub const VERTEX: u32 = 0x0020;
    pub const UNIFORM: u32 = 0x0040;
    pub const STORAGE: u32 = 0x0080;
    pub const INDIRECT: u32 = 0x0100;
    pub const QUERY_RESOLVE: u32 = 0x0200;
}

pub mod texture_usage {
    pub const COPY_SRC: u32 = 0x01;
    pub const COPY_DST: u32 = 0x02;
    pub const TEXTURE_BINDING: u32 = 0x04;
    pub const STORAGE_BINDING: u32 = 0x08;
    pub const RENDER_ATTACHMENT: u32 = 0x10;
}

/// Helper functions for creating WebGPU descriptors using js_sys::Reflect
/// since web-sys 0.3.81 doesn't have proper WebGPU types
use wasm_bindgen::JsValue;

/// Create a GpuBufferDescriptor with size and usage
pub fn create_buffer_descriptor(
    size: f64,
    usage: u32,
    label: Option<&str>,
    mapped_at_creation: bool,
) -> Result<GpuBufferDescriptor, JsValue> {
    let descriptor = js_sys::Object::new();
    js_sys::Reflect::set(&descriptor, &"size".into(), &JsValue::from_f64(size))?;
    js_sys::Reflect::set(
        &descriptor,
        &"usage".into(),
        &JsValue::from_f64(usage as f64),
    )?;
    js_sys::Reflect::set(
        &descriptor,
        &"mappedAtCreation".into(),
        &JsValue::from_bool(mapped_at_creation),
    )?;
    if let Some(label_str) = label {
        js_sys::Reflect::set(&descriptor, &"label".into(), &JsValue::from_str(label_str))?;
    }
    Ok(descriptor)
}

/// Create a GpuShaderModuleDescriptor
pub fn create_shader_module_descriptor(
    code: &str,
    label: Option<&str>,
) -> Result<GpuShaderModuleDescriptor, JsValue> {
    let descriptor = js_sys::Object::new();
    js_sys::Reflect::set(&descriptor, &"code".into(), &JsValue::from_str(code))?;
    if let Some(label_str) = label {
        js_sys::Reflect::set(&descriptor, &"label".into(), &JsValue::from_str(label_str))?;
    }
    Ok(descriptor)
}

/// Create a GpuProgrammableStage
pub fn create_programmable_stage(
    module: &GpuShaderModule,
    entry_point: &str,
) -> Result<GpuProgrammableStage, JsValue> {
    let stage = js_sys::Object::new();
    js_sys::Reflect::set(&stage, &"module".into(), module)?;
    js_sys::Reflect::set(
        &stage,
        &"entryPoint".into(),
        &JsValue::from_str(entry_point),
    )?;
    Ok(stage)
}

/// Create a GpuComputePipelineDescriptor
pub fn create_compute_pipeline_descriptor(
    compute_stage: &GpuProgrammableStage,
    label: Option<&str>,
) -> Result<GpuComputePipelineDescriptor, JsValue> {
    let descriptor = js_sys::Object::new();
    js_sys::Reflect::set(&descriptor, &"compute".into(), compute_stage)?;
    if let Some(label_str) = label {
        js_sys::Reflect::set(&descriptor, &"label".into(), &JsValue::from_str(label_str))?;
    }
    Ok(descriptor)
}

/// Create a GpuBindGroupEntry
pub fn create_bind_group_entry(
    binding: u32,
    resource: &JsValue,
) -> Result<GpuBindGroupEntry, JsValue> {
    let entry = js_sys::Object::new();
    js_sys::Reflect::set(
        &entry,
        &"binding".into(),
        &JsValue::from_f64(binding as f64),
    )?;
    js_sys::Reflect::set(&entry, &"resource".into(), resource)?;
    Ok(entry)
}

/// Create a GpuBindGroupDescriptor
pub fn create_bind_group_descriptor(
    layout: &GpuBindGroupLayout,
    entries: &js_sys::Array,
) -> Result<GpuBindGroupDescriptor, JsValue> {
    let descriptor = js_sys::Object::new();
    js_sys::Reflect::set(&descriptor, &"layout".into(), layout)?;
    js_sys::Reflect::set(&descriptor, &"entries".into(), entries)?;
    Ok(descriptor)
}

/// Create a GpuExtent3dDict (texture size)
pub fn create_extent_3d(
    width: u32,
    height: u32,
    depth_or_array_layers: u32,
) -> Result<GpuExtent3dDict, JsValue> {
    let extent = js_sys::Object::new();
    js_sys::Reflect::set(&extent, &"width".into(), &JsValue::from_f64(width as f64))?;
    js_sys::Reflect::set(&extent, &"height".into(), &JsValue::from_f64(height as f64))?;
    js_sys::Reflect::set(
        &extent,
        &"depthOrArrayLayers".into(),
        &JsValue::from_f64(depth_or_array_layers as f64),
    )?;
    Ok(extent)
}

/// Create a GpuTextureDescriptor
pub fn create_texture_descriptor(
    format: &str,
    size: &GpuExtent3dDict,
    usage: u32,
    dimension: &str,
    label: Option<&str>,
) -> Result<GpuTextureDescriptor, JsValue> {
    let descriptor = js_sys::Object::new();
    js_sys::Reflect::set(&descriptor, &"format".into(), &JsValue::from_str(format))?;
    js_sys::Reflect::set(&descriptor, &"size".into(), size)?;
    js_sys::Reflect::set(
        &descriptor,
        &"usage".into(),
        &JsValue::from_f64(usage as f64),
    )?;
    js_sys::Reflect::set(
        &descriptor,
        &"dimension".into(),
        &JsValue::from_str(dimension),
    )?;
    if let Some(label_str) = label {
        js_sys::Reflect::set(&descriptor, &"label".into(), &JsValue::from_str(label_str))?;
    }
    Ok(descriptor)
}

// Extension traits for calling WebGPU methods on js_sys::Object types
use wasm_bindgen::JsCast;

/// Extension trait for GpuDevice operations
pub trait GpuDeviceExt {
    fn create_buffer(&self, descriptor: &GpuBufferDescriptor) -> GpuBuffer;
    fn create_shader_module(&self, descriptor: &GpuShaderModuleDescriptor) -> GpuShaderModule;
    fn create_compute_pipeline(
        &self,
        descriptor: &GpuComputePipelineDescriptor,
    ) -> GpuComputePipeline;
    fn create_bind_group(&self, descriptor: &GpuBindGroupDescriptor) -> GpuBindGroup;
    fn create_command_encoder(&self) -> GpuCommandEncoder;
    fn create_texture(&self, descriptor: &GpuTextureDescriptor) -> GpuTexture;
    fn queue(&self) -> GpuQueue;
}

impl GpuDeviceExt for GpuDevice {
    fn create_buffer(&self, descriptor: &GpuBufferDescriptor) -> GpuBuffer {
        let func = js_sys::Reflect::get(self, &"createBuffer".into())
            .expect("createBuffer method should exist");
        let func: js_sys::Function = func.unchecked_into();
        func.call1(self, descriptor)
            .expect("createBuffer should succeed")
            .unchecked_into()
    }

    fn create_shader_module(&self, descriptor: &GpuShaderModuleDescriptor) -> GpuShaderModule {
        let func = js_sys::Reflect::get(self, &"createShaderModule".into())
            .expect("createShaderModule method should exist");
        let func: js_sys::Function = func.unchecked_into();
        func.call1(self, descriptor)
            .expect("createShaderModule should succeed")
            .unchecked_into()
    }

    fn create_compute_pipeline(
        &self,
        descriptor: &GpuComputePipelineDescriptor,
    ) -> GpuComputePipeline {
        let func = js_sys::Reflect::get(self, &"createComputePipeline".into())
            .expect("createComputePipeline method should exist");
        let func: js_sys::Function = func.unchecked_into();
        func.call1(self, descriptor)
            .expect("createComputePipeline should succeed")
            .unchecked_into()
    }

    fn create_bind_group(&self, descriptor: &GpuBindGroupDescriptor) -> GpuBindGroup {
        let func = js_sys::Reflect::get(self, &"createBindGroup".into())
            .expect("createBindGroup method should exist");
        let func: js_sys::Function = func.unchecked_into();
        func.call1(self, descriptor)
            .expect("createBindGroup should succeed")
            .unchecked_into()
    }

    fn create_command_encoder(&self) -> GpuCommandEncoder {
        let func = js_sys::Reflect::get(self, &"createCommandEncoder".into())
            .expect("createCommandEncoder method should exist");
        let func: js_sys::Function = func.unchecked_into();
        func.call0(self).expect("createCommandEncoder should succeed").unchecked_into()
    }

    fn create_texture(&self, descriptor: &GpuTextureDescriptor) -> GpuTexture {
        let func = js_sys::Reflect::get(self, &"createTexture".into())
            .expect("createTexture method should exist");
        let func: js_sys::Function = func.unchecked_into();
        func.call1(self, descriptor)
            .expect("createTexture should succeed")
            .unchecked_into()
    }

    fn queue(&self) -> GpuQueue {
        js_sys::Reflect::get(self, &"queue".into())
            .expect("queue property should exist")
            .unchecked_into()
    }
}

/// Extension trait for Gpu (navigator.gpu) operations
pub trait GpuExt {
    fn request_adapter(&self) -> js_sys::Promise;
}

impl GpuExt for Gpu {
    fn request_adapter(&self) -> js_sys::Promise {
        let func = js_sys::Reflect::get(self, &"requestAdapter".into())
            .expect("requestAdapter method should exist");
        let func: js_sys::Function = func.unchecked_into();
        func.call0(self).expect("requestAdapter should succeed").unchecked_into()
    }
}

/// Extension trait for GpuAdapter operations
pub trait GpuAdapterExt {
    fn request_device(&self) -> js_sys::Promise;
    fn limits(&self) -> js_sys::Object;
}

impl GpuAdapterExt for GpuAdapter {
    fn request_device(&self) -> js_sys::Promise {
        let func = js_sys::Reflect::get(self, &"requestDevice".into())
            .expect("requestDevice method should exist");
        let func: js_sys::Function = func.unchecked_into();
        func.call0(self).expect("requestDevice should succeed").unchecked_into()
    }

    fn limits(&self) -> js_sys::Object {
        js_sys::Reflect::get(self, &"limits".into())
            .expect("limits property should exist")
            .unchecked_into()
    }
}

/// Extension trait for GpuCommandEncoder operations
pub trait GpuCommandEncoderExt {
    fn copy_buffer_to_buffer(
        &self,
        source: &GpuBuffer,
        source_offset: f64,
        destination: &GpuBuffer,
        destination_offset: f64,
        size: f64,
    );
    fn begin_compute_pass(&self) -> GpuComputePassEncoder;
    fn finish(&self) -> js_sys::Object;
}

impl GpuCommandEncoderExt for GpuCommandEncoder {
    fn copy_buffer_to_buffer(
        &self,
        source: &GpuBuffer,
        source_offset: f64,
        destination: &GpuBuffer,
        destination_offset: f64,
        size: f64,
    ) {
        let func = js_sys::Reflect::get(self, &"copyBufferToBuffer".into())
            .expect("copyBufferToBuffer method should exist");
        let func: js_sys::Function = func.unchecked_into();
        let _ = func.call5(
            self,
            source,
            &JsValue::from_f64(source_offset),
            destination,
            &JsValue::from_f64(destination_offset),
            &JsValue::from_f64(size),
        );
    }

    fn begin_compute_pass(&self) -> GpuComputePassEncoder {
        let func = js_sys::Reflect::get(self, &"beginComputePass".into())
            .expect("beginComputePass method should exist");
        let func: js_sys::Function = func.unchecked_into();
        func.call0(self).expect("beginComputePass should succeed").unchecked_into()
    }

    fn finish(&self) -> js_sys::Object {
        let func =
            js_sys::Reflect::get(self, &"finish".into()).expect("finish method should exist");
        let func: js_sys::Function = func.unchecked_into();
        func.call0(self).expect("finish should succeed").unchecked_into()
    }
}

/// Extension trait for GpuQueue operations
pub trait GpuQueueExt {
    fn submit(&self, command_buffers: &js_sys::Array);
}

impl GpuQueueExt for GpuQueue {
    fn submit(&self, command_buffers: &js_sys::Array) {
        let func =
            js_sys::Reflect::get(self, &"submit".into()).expect("submit method should exist");
        let func: js_sys::Function = func.unchecked_into();
        let _ = func.call1(self, command_buffers);
    }
}

/// Extension trait for GpuBuffer operations
pub trait GpuBufferExt {
    fn map_async(&self, mode: u32, offset: f64, size: f64) -> js_sys::Promise;
    fn get_mapped_range(&self) -> js_sys::ArrayBuffer;
    fn unmap(&self);
}

impl GpuBufferExt for GpuBuffer {
    fn map_async(&self, mode: u32, offset: f64, size: f64) -> js_sys::Promise {
        let func =
            js_sys::Reflect::get(self, &"mapAsync".into()).expect("mapAsync method should exist");
        let func: js_sys::Function = func.unchecked_into();
        func.call3(
            self,
            &JsValue::from_f64(mode as f64),
            &JsValue::from_f64(offset),
            &JsValue::from_f64(size),
        )
        .expect("mapAsync should succeed")
        .unchecked_into()
    }

    fn get_mapped_range(&self) -> js_sys::ArrayBuffer {
        let func = js_sys::Reflect::get(self, &"getMappedRange".into())
            .expect("getMappedRange method should exist");
        let func: js_sys::Function = func.unchecked_into();
        func.call0(self).expect("getMappedRange should succeed").unchecked_into()
    }

    fn unmap(&self) {
        let func = js_sys::Reflect::get(self, &"unmap".into()).expect("unmap method should exist");
        let func: js_sys::Function = func.unchecked_into();
        let _ = func.call0(self);
    }
}

/// Extension trait for GpuComputePipeline operations
pub trait GpuComputePipelineExt {
    fn get_bind_group_layout(&self, index: u32) -> GpuBindGroupLayout;
}

impl GpuComputePipelineExt for GpuComputePipeline {
    fn get_bind_group_layout(&self, index: u32) -> GpuBindGroupLayout {
        let func = js_sys::Reflect::get(self, &"getBindGroupLayout".into())
            .expect("getBindGroupLayout method should exist");
        let func: js_sys::Function = func.unchecked_into();
        func.call1(self, &JsValue::from_f64(index as f64))
            .expect("getBindGroupLayout should succeed")
            .unchecked_into()
    }
}

/// Extension trait for GpuComputePassEncoder operations
pub trait GpuComputePassEncoderExt {
    fn set_pipeline(&self, pipeline: &GpuComputePipeline);
    fn set_bind_group(&self, index: u32, bind_group: &GpuBindGroup);
    fn dispatch_workgroups(&self, x: u32, y: u32, z: u32);
    fn end(&self);
}

impl GpuComputePassEncoderExt for GpuComputePassEncoder {
    fn set_pipeline(&self, pipeline: &GpuComputePipeline) {
        let func = js_sys::Reflect::get(self, &"setPipeline".into())
            .expect("setPipeline method should exist");
        let func: js_sys::Function = func.unchecked_into();
        let _ = func.call1(self, pipeline);
    }

    fn set_bind_group(&self, index: u32, bind_group: &GpuBindGroup) {
        let func = js_sys::Reflect::get(self, &"setBindGroup".into())
            .expect("setBindGroup method should exist");
        let func: js_sys::Function = func.unchecked_into();
        let _ = func.call2(self, &JsValue::from_f64(index as f64), bind_group);
    }

    fn dispatch_workgroups(&self, x: u32, y: u32, z: u32) {
        let func = js_sys::Reflect::get(self, &"dispatchWorkgroups".into())
            .expect("dispatchWorkgroups method should exist");
        let func: js_sys::Function = func.unchecked_into();
        let _ = func.call3(
            self,
            &JsValue::from_f64(x as f64),
            &JsValue::from_f64(y as f64),
            &JsValue::from_f64(z as f64),
        );
    }

    fn end(&self) {
        let func = js_sys::Reflect::get(self, &"end".into()).expect("end method should exist");
        let func: js_sys::Function = func.unchecked_into();
        let _ = func.call0(self);
    }
}
