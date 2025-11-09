//! Metal API Bindings for iOS
//!
//! This module provides low-level Metal API bindings for GPU computation on iOS devices.
//! Metal is Apple's low-level graphics and compute API for high-performance computation.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_uint, c_void};
use std::ptr;

#[cfg(target_os = "ios")]
use core_foundation::{
    base::{CFRelease, CFTypeRef},
    string::{CFString, CFStringRef},
};

// Metal API types
#[cfg(target_os = "ios")]
#[repr(C)]
pub struct MTLDevice;

#[cfg(target_os = "ios")]
#[repr(C)]
pub struct MTLLibrary;

#[cfg(target_os = "ios")]
#[repr(C)]
pub struct MTLFunction;

#[cfg(target_os = "ios")]
#[repr(C)]
pub struct MTLComputePipelineState;

#[cfg(target_os = "ios")]
#[repr(C)]
pub struct MTLCommandQueue;

#[cfg(target_os = "ios")]
#[repr(C)]
pub struct MTLCommandBuffer;

#[cfg(target_os = "ios")]
#[repr(C)]
pub struct MTLComputeCommandEncoder;

#[cfg(target_os = "ios")]
#[repr(C)]
pub struct MTLBuffer;

#[cfg(target_os = "ios")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MTLSize {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}

#[cfg(target_os = "ios")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MTLOrigin {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

#[cfg(target_os = "ios")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MTLRegion {
    pub origin: MTLOrigin,
    pub size: MTLSize,
}

// Metal resource options
#[cfg(target_os = "ios")]
pub const MTL_RESOURCE_STORAGE_MODE_SHARED: c_uint = 0;
#[cfg(target_os = "ios")]
pub const MTL_RESOURCE_STORAGE_MODE_MANAGED: c_uint = 1;
#[cfg(target_os = "ios")]
pub const MTL_RESOURCE_STORAGE_MODE_PRIVATE: c_uint = 2;
#[cfg(target_os = "ios")]
pub const MTL_RESOURCE_STORAGE_MODE_MEMORYLESS: c_uint = 3;

// Metal GPU families
#[cfg(target_os = "ios")]
pub const MTL_GPU_FAMILY_APPLE_1: u32 = 1001;
#[cfg(target_os = "ios")]
pub const MTL_GPU_FAMILY_APPLE_2: u32 = 1002;
#[cfg(target_os = "ios")]
pub const MTL_GPU_FAMILY_APPLE_3: u32 = 1003;
#[cfg(target_os = "ios")]
pub const MTL_GPU_FAMILY_APPLE_4: u32 = 1004;
#[cfg(target_os = "ios")]
pub const MTL_GPU_FAMILY_APPLE_5: u32 = 1005;
#[cfg(target_os = "ios")]
pub const MTL_GPU_FAMILY_APPLE_6: u32 = 1006;
#[cfg(target_os = "ios")]
pub const MTL_GPU_FAMILY_APPLE_7: u32 = 1007;
#[cfg(target_os = "ios")]
pub const MTL_GPU_FAMILY_APPLE_8: u32 = 1008;

// Metal C API bindings
#[cfg(target_os = "ios")]
extern "C" {
    // Device creation and management
    pub fn MTLCreateSystemDefaultDevice() -> *mut MTLDevice;
    pub fn MTLCopyAllDevices() -> *mut c_void; // Returns NSArray of MTLDevice objects
    pub fn MTLDevice_getName(device: *mut MTLDevice) -> CFStringRef;
    pub fn MTLDevice_getSupportsFamily(device: *mut MTLDevice, family: u32) -> bool;
    pub fn MTLDevice_getRecommendedMaxWorkingSetSize(device: *mut MTLDevice) -> u64;
    pub fn MTLDevice_getMaxThreadsPerThreadgroup(device: *mut MTLDevice) -> MTLSize;
    pub fn MTLDevice_getRegistryID(device: *mut MTLDevice) -> u64;
    pub fn MTLDevice_getArchitecture(device: *mut MTLDevice) -> CFStringRef;

    // Multi-GPU support
    pub fn NSArray_count(array: *mut c_void) -> usize;
    pub fn NSArray_objectAtIndex(array: *mut c_void, index: usize) -> *mut MTLDevice;

    // Library and function management
    pub fn MTLDevice_newDefaultLibrary(device: *mut MTLDevice) -> *mut MTLLibrary;
    pub fn MTLDevice_newLibraryWithSource(
        device: *mut MTLDevice,
        source: *const c_char,
        options: *mut c_void,
        error: *mut *mut c_void,
    ) -> *mut MTLLibrary;
    pub fn MTLLibrary_newFunctionWithName(
        library: *mut MTLLibrary,
        name: *const c_char,
    ) -> *mut MTLFunction;
    pub fn MTLFunction_setName(function: *mut MTLFunction, name: *const c_char);

    // Pipeline state creation
    pub fn MTLDevice_newComputePipelineStateWithFunction(
        device: *mut MTLDevice,
        function: *mut MTLFunction,
        error: *mut *mut c_void,
    ) -> *mut MTLComputePipelineState;
    pub fn MTLComputePipelineState_getMaxTotalThreadsPerThreadgroup(
        state: *mut MTLComputePipelineState,
    ) -> usize;
    pub fn MTLComputePipelineState_getThreadExecutionWidth(
        state: *mut MTLComputePipelineState,
    ) -> usize;

    // Command queue and buffers
    pub fn MTLDevice_newCommandQueue(device: *mut MTLDevice) -> *mut MTLCommandQueue;
    pub fn MTLDevice_newCommandQueueWithMaxCommandBufferCount(
        device: *mut MTLDevice,
        max_buffer_count: usize,
    ) -> *mut MTLCommandQueue;
    pub fn MTLCommandQueue_commandBuffer(queue: *mut MTLCommandQueue) -> *mut MTLCommandBuffer;
    pub fn MTLCommandQueue_commandBufferWithUnretainedReferences(
        queue: *mut MTLCommandQueue,
    ) -> *mut MTLCommandBuffer;

    // Buffer management
    pub fn MTLDevice_newBufferWithLength(
        device: *mut MTLDevice,
        length: usize,
        options: c_uint,
    ) -> *mut MTLBuffer;
    pub fn MTLDevice_newBufferWithBytes(
        device: *mut MTLDevice,
        pointer: *const c_void,
        length: usize,
        options: c_uint,
    ) -> *mut MTLBuffer;
    pub fn MTLBuffer_contents(buffer: *mut MTLBuffer) -> *mut c_void;
    pub fn MTLBuffer_length(buffer: *mut MTLBuffer) -> usize;
    pub fn MTLBuffer_didModifyRange(
        buffer: *mut MTLBuffer,
        range_location: usize,
        range_length: usize,
    );

    // Compute encoding
    pub fn MTLCommandBuffer_computeCommandEncoder(
        buffer: *mut MTLCommandBuffer,
    ) -> *mut MTLComputeCommandEncoder;
    pub fn MTLComputeCommandEncoder_setComputePipelineState(
        encoder: *mut MTLComputeCommandEncoder,
        state: *mut MTLComputePipelineState,
    );
    pub fn MTLComputeCommandEncoder_setBuffer(
        encoder: *mut MTLComputeCommandEncoder,
        buffer: *mut MTLBuffer,
        offset: usize,
        index: c_uint,
    );
    pub fn MTLComputeCommandEncoder_setBytes(
        encoder: *mut MTLComputeCommandEncoder,
        bytes: *const c_void,
        length: usize,
        index: c_uint,
    );
    pub fn MTLComputeCommandEncoder_dispatchThreadgroups(
        encoder: *mut MTLComputeCommandEncoder,
        threadgroupsPerGrid: MTLSize,
        threadsPerThreadgroup: MTLSize,
    );
    pub fn MTLComputeCommandEncoder_dispatchThreads(
        encoder: *mut MTLComputeCommandEncoder,
        threadsPerGrid: MTLSize,
        threadsPerThreadgroup: MTLSize,
    );
    pub fn MTLComputeCommandEncoder_endEncoding(encoder: *mut MTLComputeCommandEncoder);
    pub fn MTLComputeCommandEncoder_setLabel(
        encoder: *mut MTLComputeCommandEncoder,
        label: *const c_char,
    );

    // Command execution and synchronization
    pub fn MTLCommandBuffer_commit(buffer: *mut MTLCommandBuffer);
    pub fn MTLCommandBuffer_waitUntilCompleted(buffer: *mut MTLCommandBuffer);
    pub fn MTLCommandBuffer_waitUntilScheduled(buffer: *mut MTLCommandBuffer);
    pub fn MTLCommandBuffer_addCompletedHandler(
        buffer: *mut MTLCommandBuffer,
        handler: extern "C" fn(*mut MTLCommandBuffer),
    );
    pub fn MTLCommandBuffer_addScheduledHandler(
        buffer: *mut MTLCommandBuffer,
        handler: extern "C" fn(*mut MTLCommandBuffer),
    );
    pub fn MTLCommandBuffer_enqueue(buffer: *mut MTLCommandBuffer);
    pub fn MTLCommandBuffer_setLabel(buffer: *mut MTLCommandBuffer, label: *const c_char);

    // Performance and debugging
    pub fn MTLCommandBuffer_GPUStartTime(buffer: *mut MTLCommandBuffer) -> f64;
    pub fn MTLCommandBuffer_GPUEndTime(buffer: *mut MTLCommandBuffer) -> f64;
    pub fn MTLCommandBuffer_kernelStartTime(buffer: *mut MTLCommandBuffer) -> f64;
    pub fn MTLCommandBuffer_kernelEndTime(buffer: *mut MTLCommandBuffer) -> f64;

    // Memory management and cleanup
    pub fn MTLDevice_release(device: *mut MTLDevice);
    pub fn MTLLibrary_release(library: *mut MTLLibrary);
    pub fn MTLFunction_release(function: *mut MTLFunction);
    pub fn MTLComputePipelineState_release(state: *mut MTLComputePipelineState);
    pub fn MTLCommandQueue_release(queue: *mut MTLCommandQueue);
    pub fn MTLCommandBuffer_release(buffer: *mut MTLCommandBuffer);
    pub fn MTLComputeCommandEncoder_release(encoder: *mut MTLComputeCommandEncoder);
    pub fn MTLBuffer_release(buffer: *mut MTLBuffer);

    // Memory pressure and optimization
    pub fn MTLDevice_currentAllocatedSize(device: *mut MTLDevice) -> usize;
    pub fn MTLDevice_hasUnifiedMemory(device: *mut MTLDevice) -> bool;
    pub fn MTLDevice_isLowPower(device: *mut MTLDevice) -> bool;
    pub fn MTLDevice_isRemovable(device: *mut MTLDevice) -> bool;
    pub fn MTLDevice_locationNumber(device: *mut MTLDevice) -> usize;
    pub fn MTLDevice_maxTransferRate(device: *mut MTLDevice) -> u64;
}

// High-level Metal wrapper types
#[cfg(target_os = "ios")]
pub struct MetalDevice {
    device: *mut MTLDevice,
    command_queue: *mut MTLCommandQueue,
    device_name: String,
    supports_apple_gpu: bool,
    max_working_set_size: u64,
    max_threads_per_threadgroup: MTLSize,
}

#[cfg(target_os = "ios")]
pub struct MetalBuffer {
    buffer: *mut MTLBuffer,
    length: usize,
}

#[cfg(target_os = "ios")]
pub struct MetalComputePipeline {
    pipeline_state: *mut MTLComputePipelineState,
    max_total_threads: usize,
    thread_execution_width: usize,
}

#[cfg(target_os = "ios")]
pub struct MetalCommandBuffer {
    command_buffer: *mut MTLCommandBuffer,
    label: Option<String>,
}

#[cfg(target_os = "ios")]
impl MetalDevice {
    /// Create Metal device from system default
    pub fn create_system_default() -> Result<Self, String> {
        unsafe {
            let device = MTLCreateSystemDefaultDevice();
            if device.is_null() {
                return Err("Failed to create Metal device".to_string());
            }

            let command_queue = MTLDevice_newCommandQueue(device);
            if command_queue.is_null() {
                MTLDevice_release(device);
                return Err("Failed to create command queue".to_string());
            }

            let name_ref = MTLDevice_getName(device);
            let device_name = if !name_ref.is_null() {
                CFString::from_CFTypeRef(name_ref as CFTypeRef).to_string()
            } else {
                "Unknown Device".to_string()
            };

            let supports_apple_gpu = MTLDevice_getSupportsFamily(device, MTL_GPU_FAMILY_APPLE_1);
            let max_working_set_size = MTLDevice_getRecommendedMaxWorkingSetSize(device);
            let max_threads_per_threadgroup = MTLDevice_getMaxThreadsPerThreadgroup(device);

            Ok(Self {
                device,
                command_queue,
                device_name,
                supports_apple_gpu,
                max_working_set_size,
                max_threads_per_threadgroup,
            })
        }
    }

    /// Get all available Metal devices
    pub fn get_all_devices() -> Result<Vec<Self>, String> {
        unsafe {
            let devices_array = MTLCopyAllDevices();
            if devices_array.is_null() {
                return Err("Failed to get Metal devices".to_string());
            }

            let device_count = NSArray_count(devices_array);
            let mut devices = Vec::with_capacity(device_count);

            for i in 0..device_count {
                let device = NSArray_objectAtIndex(devices_array, i);
                if !device.is_null() {
                    let command_queue = MTLDevice_newCommandQueue(device);
                    if !command_queue.is_null() {
                        let name_ref = MTLDevice_getName(device);
                        let device_name = if !name_ref.is_null() {
                            CFString::from_CFTypeRef(name_ref as CFTypeRef).to_string()
                        } else {
                            format!("Device {}", i)
                        };

                        let supports_apple_gpu =
                            MTLDevice_getSupportsFamily(device, MTL_GPU_FAMILY_APPLE_1);
                        let max_working_set_size =
                            MTLDevice_getRecommendedMaxWorkingSetSize(device);
                        let max_threads_per_threadgroup =
                            MTLDevice_getMaxThreadsPerThreadgroup(device);

                        devices.push(Self {
                            device,
                            command_queue,
                            device_name,
                            supports_apple_gpu,
                            max_working_set_size,
                            max_threads_per_threadgroup,
                        });
                    }
                }
            }

            // Note: devices_array should be released, but we'll let ARC handle it
            Ok(devices)
        }
    }

    /// Create buffer with data
    pub fn create_buffer_with_data(&self, data: &[u8]) -> Result<MetalBuffer, String> {
        unsafe {
            let buffer = MTLDevice_newBufferWithBytes(
                self.device,
                data.as_ptr() as *const c_void,
                data.len(),
                MTL_RESOURCE_STORAGE_MODE_SHARED,
            );

            if buffer.is_null() {
                return Err("Failed to create Metal buffer".to_string());
            }

            Ok(MetalBuffer {
                buffer,
                length: data.len(),
            })
        }
    }

    /// Create buffer with size
    pub fn create_buffer_with_size(&self, size: usize) -> Result<MetalBuffer, String> {
        unsafe {
            let buffer =
                MTLDevice_newBufferWithLength(self.device, size, MTL_RESOURCE_STORAGE_MODE_SHARED);

            if buffer.is_null() {
                return Err("Failed to create Metal buffer".to_string());
            }

            Ok(MetalBuffer {
                buffer,
                length: size,
            })
        }
    }

    /// Create compute pipeline from source
    pub fn create_compute_pipeline_from_source(
        &self,
        source: &str,
        function_name: &str,
    ) -> Result<MetalComputePipeline, String> {
        unsafe {
            let source_cstr = CString::new(source).map_err(|e| format!("Invalid source: {}", e))?;
            let function_name_cstr =
                CString::new(function_name).map_err(|e| format!("Invalid function name: {}", e))?;

            let mut error: *mut c_void = ptr::null_mut();
            let library = MTLDevice_newLibraryWithSource(
                self.device,
                source_cstr.as_ptr(),
                ptr::null_mut(),
                &mut error,
            );

            if library.is_null() {
                return Err("Failed to create Metal library".to_string());
            }

            let function = MTLLibrary_newFunctionWithName(library, function_name_cstr.as_ptr());
            if function.is_null() {
                MTLLibrary_release(library);
                return Err(format!("Failed to find function: {}", function_name));
            }

            let pipeline_state =
                MTLDevice_newComputePipelineStateWithFunction(self.device, function, &mut error);

            MTLFunction_release(function);
            MTLLibrary_release(library);

            if pipeline_state.is_null() {
                return Err("Failed to create compute pipeline state".to_string());
            }

            let max_total_threads =
                MTLComputePipelineState_getMaxTotalThreadsPerThreadgroup(pipeline_state);
            let thread_execution_width =
                MTLComputePipelineState_getThreadExecutionWidth(pipeline_state);

            Ok(MetalComputePipeline {
                pipeline_state,
                max_total_threads,
                thread_execution_width,
            })
        }
    }

    /// Create command buffer
    pub fn create_command_buffer(&self) -> Result<MetalCommandBuffer, String> {
        unsafe {
            let command_buffer = MTLCommandQueue_commandBuffer(self.command_queue);
            if command_buffer.is_null() {
                return Err("Failed to create command buffer".to_string());
            }

            Ok(MetalCommandBuffer {
                command_buffer,
                label: None,
            })
        }
    }

    /// Get device information
    pub fn get_device_info(&self) -> MetalDeviceInfo {
        unsafe {
            MetalDeviceInfo {
                name: self.device_name.clone(),
                supports_apple_gpu: self.supports_apple_gpu,
                max_working_set_size: self.max_working_set_size,
                max_threads_per_threadgroup: self.max_threads_per_threadgroup,
                current_allocated_size: MTLDevice_currentAllocatedSize(self.device),
                has_unified_memory: MTLDevice_hasUnifiedMemory(self.device),
                is_low_power: MTLDevice_isLowPower(self.device),
                is_removable: MTLDevice_isRemovable(self.device),
                location_number: MTLDevice_locationNumber(self.device),
                max_transfer_rate: MTLDevice_maxTransferRate(self.device),
                registry_id: MTLDevice_getRegistryID(self.device),
                architecture: {
                    let arch_ref = MTLDevice_getArchitecture(self.device);
                    if !arch_ref.is_null() {
                        CFString::from_CFTypeRef(arch_ref as CFTypeRef).to_string()
                    } else {
                        "Unknown".to_string()
                    }
                },
            }
        }
    }
}

#[cfg(target_os = "ios")]
impl Drop for MetalDevice {
    fn drop(&mut self) {
        unsafe {
            if !self.command_queue.is_null() {
                MTLCommandQueue_release(self.command_queue);
            }
            if !self.device.is_null() {
                MTLDevice_release(self.device);
            }
        }
    }
}

#[cfg(target_os = "ios")]
impl MetalBuffer {
    /// Get buffer contents as mutable slice
    pub fn contents_mut<T>(&mut self) -> &mut [T] {
        unsafe {
            let ptr = MTLBuffer_contents(self.buffer) as *mut T;
            let len = self.length / std::mem::size_of::<T>();
            std::slice::from_raw_parts_mut(ptr, len)
        }
    }

    /// Get buffer contents as slice
    pub fn contents<T>(&self) -> &[T] {
        unsafe {
            let ptr = MTLBuffer_contents(self.buffer) as *const T;
            let len = self.length / std::mem::size_of::<T>();
            std::slice::from_raw_parts(ptr, len)
        }
    }

    /// Mark buffer range as modified
    pub fn did_modify_range(&self, location: usize, length: usize) {
        unsafe {
            MTLBuffer_didModifyRange(self.buffer, location, length);
        }
    }

    /// Get buffer length
    pub fn length(&self) -> usize {
        self.length
    }
}

#[cfg(target_os = "ios")]
impl Drop for MetalBuffer {
    fn drop(&mut self) {
        unsafe {
            if !self.buffer.is_null() {
                MTLBuffer_release(self.buffer);
            }
        }
    }
}

#[cfg(target_os = "ios")]
impl MetalComputePipeline {
    /// Get maximum total threads per threadgroup
    pub fn max_total_threads_per_threadgroup(&self) -> usize {
        self.max_total_threads
    }

    /// Get thread execution width
    pub fn thread_execution_width(&self) -> usize {
        self.thread_execution_width
    }
}

#[cfg(target_os = "ios")]
impl Drop for MetalComputePipeline {
    fn drop(&mut self) {
        unsafe {
            if !self.pipeline_state.is_null() {
                MTLComputePipelineState_release(self.pipeline_state);
            }
        }
    }
}

#[cfg(target_os = "ios")]
impl MetalCommandBuffer {
    /// Create compute command encoder
    pub fn create_compute_encoder(&self) -> Result<MetalComputeEncoder, String> {
        unsafe {
            let encoder = MTLCommandBuffer_computeCommandEncoder(self.command_buffer);
            if encoder.is_null() {
                return Err("Failed to create compute command encoder".to_string());
            }

            Ok(MetalComputeEncoder { encoder })
        }
    }

    /// Set command buffer label
    pub fn set_label(&mut self, label: &str) {
        self.label = Some(label.to_string());
        let label_cstr = CString::new(label).unwrap();
        unsafe {
            MTLCommandBuffer_setLabel(self.command_buffer, label_cstr.as_ptr());
        }
    }

    /// Commit command buffer
    pub fn commit(&self) {
        unsafe {
            MTLCommandBuffer_commit(self.command_buffer);
        }
    }

    /// Wait until completed
    pub fn wait_until_completed(&self) {
        unsafe {
            MTLCommandBuffer_waitUntilCompleted(self.command_buffer);
        }
    }

    /// Get GPU execution times
    pub fn get_gpu_times(&self) -> (f64, f64) {
        unsafe {
            let start_time = MTLCommandBuffer_GPUStartTime(self.command_buffer);
            let end_time = MTLCommandBuffer_GPUEndTime(self.command_buffer);
            (start_time, end_time)
        }
    }
}

#[cfg(target_os = "ios")]
impl Drop for MetalCommandBuffer {
    fn drop(&mut self) {
        unsafe {
            if !self.command_buffer.is_null() {
                MTLCommandBuffer_release(self.command_buffer);
            }
        }
    }
}

/// Metal compute command encoder wrapper
#[cfg(target_os = "ios")]
pub struct MetalComputeEncoder {
    encoder: *mut MTLComputeCommandEncoder,
}

#[cfg(target_os = "ios")]
impl MetalComputeEncoder {
    /// Set compute pipeline state
    pub fn set_compute_pipeline_state(&self, pipeline: &MetalComputePipeline) {
        unsafe {
            MTLComputeCommandEncoder_setComputePipelineState(self.encoder, pipeline.pipeline_state);
        }
    }

    /// Set buffer at index
    pub fn set_buffer(&self, buffer: &MetalBuffer, offset: usize, index: u32) {
        unsafe {
            MTLComputeCommandEncoder_setBuffer(self.encoder, buffer.buffer, offset, index);
        }
    }

    /// Set bytes at index
    pub fn set_bytes(&self, bytes: &[u8], index: u32) {
        unsafe {
            MTLComputeCommandEncoder_setBytes(
                self.encoder,
                bytes.as_ptr() as *const c_void,
                bytes.len(),
                index,
            );
        }
    }

    /// Dispatch threadgroups
    pub fn dispatch_threadgroups(
        &self,
        threadgroups_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        unsafe {
            MTLComputeCommandEncoder_dispatchThreadgroups(
                self.encoder,
                threadgroups_per_grid,
                threads_per_threadgroup,
            );
        }
    }

    /// Dispatch threads (iOS 11+)
    pub fn dispatch_threads(&self, threads_per_grid: MTLSize, threads_per_threadgroup: MTLSize) {
        unsafe {
            MTLComputeCommandEncoder_dispatchThreads(
                self.encoder,
                threads_per_grid,
                threads_per_threadgroup,
            );
        }
    }

    /// Set encoder label
    pub fn set_label(&self, label: &str) {
        let label_cstr = CString::new(label).unwrap();
        unsafe {
            MTLComputeCommandEncoder_setLabel(self.encoder, label_cstr.as_ptr());
        }
    }

    /// End encoding
    pub fn end_encoding(&self) {
        unsafe {
            MTLComputeCommandEncoder_endEncoding(self.encoder);
        }
    }
}

#[cfg(target_os = "ios")]
impl Drop for MetalComputeEncoder {
    fn drop(&mut self) {
        unsafe {
            if !self.encoder.is_null() {
                MTLComputeCommandEncoder_release(self.encoder);
            }
        }
    }
}

/// Metal device information
#[derive(Debug, Clone)]
pub struct MetalDeviceInfo {
    pub name: String,
    pub supports_apple_gpu: bool,
    pub max_working_set_size: u64,
    pub max_threads_per_threadgroup: MTLSize,
    pub current_allocated_size: usize,
    pub has_unified_memory: bool,
    pub is_low_power: bool,
    pub is_removable: bool,
    pub location_number: usize,
    pub max_transfer_rate: u64,
    pub registry_id: u64,
    pub architecture: String,
}

/// Utility functions for Metal size calculations
impl MTLSize {
    /// Create new MTLSize
    pub fn new(width: usize, height: usize, depth: usize) -> Self {
        Self {
            width,
            height,
            depth,
        }
    }

    /// Create 1D size
    pub fn new_1d(width: usize) -> Self {
        Self {
            width,
            height: 1,
            depth: 1,
        }
    }

    /// Create 2D size
    pub fn new_2d(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            depth: 1,
        }
    }

    /// Get total size
    pub fn total(&self) -> usize {
        self.width * self.height * self.depth
    }
}

impl MTLOrigin {
    /// Create new MTLOrigin
    pub fn new(x: usize, y: usize, z: usize) -> Self {
        Self { x, y, z }
    }

    /// Create zero origin
    pub fn zero() -> Self {
        Self { x: 0, y: 0, z: 0 }
    }
}

impl MTLRegion {
    /// Create new MTLRegion
    pub fn new(origin: MTLOrigin, size: MTLSize) -> Self {
        Self { origin, size }
    }

    /// Create region from size with zero origin
    pub fn from_size(size: MTLSize) -> Self {
        Self {
            origin: MTLOrigin::zero(),
            size,
        }
    }
}

// Non-iOS stub implementations
#[cfg(not(target_os = "ios"))]
pub struct MetalDevice;

#[cfg(not(target_os = "ios"))]
pub struct MetalBuffer;

#[cfg(not(target_os = "ios"))]
pub struct MetalComputePipeline;

#[cfg(not(target_os = "ios"))]
pub struct MetalCommandBuffer;

#[cfg(not(target_os = "ios"))]
pub struct MetalComputeEncoder;

#[cfg(not(target_os = "ios"))]
#[derive(Debug, Clone)]
pub struct MetalDeviceInfo {
    pub name: String,
}

#[cfg(not(target_os = "ios"))]
impl MetalDevice {
    pub fn create_system_default() -> Result<Self, String> {
        Err("Metal not available on this platform".to_string())
    }

    pub fn get_all_devices() -> Result<Vec<Self>, String> {
        Err("Metal not available on this platform".to_string())
    }
}
