//! Enhanced WebGPU backend with buffer pool and memory monitoring

use crate::webgpu::buffer_pool::BufferPool;
use crate::webgpu::simple_ops::SimpleGpuOps;
use js_sys::{Float32Array, SharedArrayBuffer};
use std::collections::VecDeque;
use std::vec::Vec;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
// Use shared WebGPU types
use super::types::{
    buffer_usage, texture_usage, GpuBuffer, GpuBufferExt, GpuCommandEncoderExt, GpuDevice,
    GpuDeviceExt, GpuQueueExt, GpuTexture,
};

/// Enhanced WebGPU backend with memory management
#[wasm_bindgen]
pub struct WebGPUBackend {
    device: GpuDevice,
    ops: SimpleGpuOps,
    buffer_pool: BufferPool,
    peak_memory: usize,
    current_memory: usize,
    texture_cache: std::collections::BTreeMap<String, GpuTexture>,
    pending_operations: VecDeque<js_sys::Promise>,
    shared_memory_enabled: bool,
    shared_buffers: std::collections::BTreeMap<usize, SharedArrayBuffer>,
}

#[wasm_bindgen]
impl WebGPUBackend {
    /// Create a new WebGPU backend
    pub fn new(device: GpuDevice) -> Result<WebGPUBackend, JsValue> {
        let ops = SimpleGpuOps::new(device.clone());
        let shared_memory_enabled = Self::is_shared_memory_supported();

        Ok(WebGPUBackend {
            device,
            ops,
            buffer_pool: BufferPool::new(),
            peak_memory: 0,
            current_memory: 0,
            texture_cache: std::collections::BTreeMap::new(),
            pending_operations: VecDeque::new(),
            shared_memory_enabled,
            shared_buffers: std::collections::BTreeMap::new(),
        })
    }

    /// Get current memory usage in bytes
    #[wasm_bindgen(getter)]
    pub fn current_memory(&self) -> usize {
        self.current_memory
    }

    /// Get peak memory usage in bytes
    #[wasm_bindgen(getter)]
    pub fn peak_memory(&self) -> usize {
        self.peak_memory
    }

    /// Clear buffer pool and reset memory counters
    pub fn cleanup(&mut self) {
        self.buffer_pool.clear();
        self.texture_cache.clear();
        self.pending_operations.clear();
        self.shared_buffers.clear();
        self.current_memory = 0;
    }

    /// Get number of pending asynchronous operations
    pub fn pending_operations_count(&self) -> usize {
        self.pending_operations.len()
    }

    /// Check if WebGPU is available
    pub fn is_available() -> bool {
        web_sys::window()
            .and_then(|w| {
                js_sys::Reflect::get(&w.navigator(), &JsValue::from_str("gpu"))
                    .ok()
                    .filter(|v| !v.is_undefined())
            })
            .is_some()
    }

    /// Get the device (cloned for wasm compatibility)
    pub fn get_device(&self) -> GpuDevice {
        self.device.clone()
    }

    /// Check if shared memory is supported
    pub fn is_shared_memory_supported() -> bool {
        web_sys::window()
            .and_then(|w| js_sys::Reflect::get(&w, &JsValue::from_str("SharedArrayBuffer")).ok())
            .map(|v| !v.is_undefined())
            .unwrap_or(false)
    }

    /// Get shared memory support status
    #[wasm_bindgen(getter)]
    pub fn shared_memory_enabled(&self) -> bool {
        self.shared_memory_enabled
    }

    /// Create a shared buffer for inter-thread communication
    pub fn create_shared_buffer(&mut self, size: usize) -> Result<SharedArrayBuffer, JsValue> {
        if !self.shared_memory_enabled {
            return Err("SharedArrayBuffer not supported".into());
        }

        let byte_size = size * 4; // f32 = 4 bytes
        let shared_buffer = SharedArrayBuffer::new(byte_size as u32);

        // Store reference to the buffer
        self.shared_buffers.insert(size, shared_buffer.clone());

        Ok(shared_buffer)
    }

    /// Get or create a shared buffer for given size
    pub fn get_shared_buffer(&mut self, size: usize) -> Result<SharedArrayBuffer, JsValue> {
        if let Some(buffer) = self.shared_buffers.get(&size) {
            return Ok(buffer.clone());
        }

        self.create_shared_buffer(size)
    }

    /// Create GPU buffer from shared memory
    pub fn create_buffer_from_shared(
        &mut self,
        shared_buffer: &SharedArrayBuffer,
        usage: u32,
    ) -> Result<GpuBuffer, JsValue> {
        let byte_size = shared_buffer.byte_length() as usize;
        let float_size = byte_size / 4;

        // Create Float32Array view of shared buffer
        let shared_array = Float32Array::new(shared_buffer);

        // Create GPU buffer
        let buffer = self.buffer_pool.get_buffer(&self.device, float_size, usage)?;

        // Copy data from shared buffer to GPU buffer
        let mapped_range = buffer.get_mapped_range();
        let mapped_array = Float32Array::new(&mapped_range);
        mapped_array.set(&shared_array.subarray(0, shared_array.length()), 0);
        buffer.unmap();

        // Update memory tracking
        self.current_memory += byte_size;
        if self.current_memory > self.peak_memory {
            self.peak_memory = self.current_memory;
        }

        Ok(buffer)
    }

    /// Clear shared buffers
    pub fn clear_shared_buffers(&mut self) {
        self.shared_buffers.clear();
    }
}

// Non-wasm_bindgen methods for internal use
impl WebGPUBackend {
    pub fn ops(&self) -> &SimpleGpuOps {
        &self.ops
    }

    /// Create a GPU buffer with the given data and usage, using buffer pool
    pub fn create_buffer(&mut self, data: &[f32], usage: u32) -> Result<GpuBuffer, JsValue> {
        let size = data.len();
        let byte_size = size * 4; // f32 = 4 bytes

        // Try to get buffer from pool first
        let buffer = self.buffer_pool.get_buffer(&self.device, size, usage)?;

        // Map buffer and copy data
        let mapped_range = buffer.get_mapped_range();
        let mapped_array = js_sys::Float32Array::new(&mapped_range);
        mapped_array.copy_from(data);
        buffer.unmap();

        // Update memory tracking
        self.current_memory += byte_size;
        if self.current_memory > self.peak_memory {
            self.peak_memory = self.current_memory;
        }

        Ok(buffer)
    }

    /// Create an empty GPU buffer of given size
    pub fn create_empty_buffer(&mut self, size: usize, usage: u32) -> Result<GpuBuffer, JsValue> {
        let byte_size = size * 4; // f32 = 4 bytes

        let buffer = self.buffer_pool.get_buffer(&self.device, size, usage)?;

        // Update memory tracking
        self.current_memory += byte_size;
        if self.current_memory > self.peak_memory {
            self.peak_memory = self.current_memory;
        }

        Ok(buffer)
    }

    /// Return a buffer to the pool
    pub fn return_buffer(&mut self, buffer: GpuBuffer, size: usize) {
        let byte_size = size * 4;
        self.buffer_pool.return_buffer(buffer, size);

        // Update memory tracking
        if self.current_memory >= byte_size {
            self.current_memory -= byte_size;
        }
    }

    /// Read data back from GPU buffer
    pub async fn read_buffer(&self, buffer: &GpuBuffer, size: usize) -> Result<Vec<f32>, JsValue> {
        use super::types::create_buffer_descriptor;

        // Create a staging buffer for reading back data
        let staging_size = (size * 4) as f64; // f32 = 4 bytes
        let staging_descriptor = create_buffer_descriptor(
            staging_size,
            buffer_usage::COPY_DST | buffer_usage::MAP_READ,
            Some("Staging Buffer"),
            false,
        )?;

        let staging_buffer = self.device.create_buffer(&staging_descriptor);

        // Copy data from compute buffer to staging buffer
        let encoder = self.device.create_command_encoder();
        encoder.copy_buffer_to_buffer(buffer, 0.0, &staging_buffer, 0.0, staging_size);
        let command_buffer = encoder.finish();
        self.device.queue().submit(&js_sys::Array::of1(&command_buffer));

        // Map staging buffer for reading
        // GpuMapMode not available in web-sys 0.3.81 - using numeric value
        let map_future = JsFuture::from(staging_buffer.map_async(1u32, 0.0, staging_size)); // READ = 1
        map_future.await?;

        // Read data from mapped buffer
        let mapped_range = staging_buffer.get_mapped_range();
        let mapped_array = js_sys::Float32Array::new(&mapped_range);
        let mut result = vec![0.0f32; size];
        mapped_array.copy_to(&mut result);

        staging_buffer.unmap();

        Ok(result)
    }

    /// Create a texture for efficient weight storage
    pub fn create_weight_texture(
        &mut self,
        _data: &[f32],
        width: u32,
        height: u32,
        key: &str,
    ) -> Result<&GpuTexture, JsValue> {
        // Use entry API to avoid borrow conflict
        use std::collections::btree_map::Entry;

        match self.texture_cache.entry(key.to_string()) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                // Create texture descriptor using helper functions
                let size = super::types::create_extent_3d(width, height, 1)?;

                let descriptor = super::types::create_texture_descriptor(
                    "rgba32float",
                    &size,
                    texture_usage::TEXTURE_BINDING | texture_usage::COPY_DST,
                    "2d",
                    Some(key),
                )?;

                let texture = self.device.create_texture(&descriptor);

                // Write data to texture (simplified - would need proper data conversion)
                // In a real implementation, you'd convert f32 data to RGBA format

                // Update memory tracking
                let texture_size = (width * height * 16) as usize; // 4 components * 4 bytes each
                self.current_memory += texture_size;

                if self.current_memory > self.peak_memory {
                    self.peak_memory = self.current_memory;
                }

                // Insert into cache and return reference
                Ok(entry.insert(texture))
            },
        }
    }

    /// Execute an operation asynchronously and track it
    pub fn execute_async(&mut self, operation: js_sys::Promise) {
        self.pending_operations.push_back(operation);
    }

    /// Wait for all pending operations to complete
    pub async fn wait_for_completion(&mut self) -> Result<(), JsValue> {
        while let Some(operation) = self.pending_operations.pop_front() {
            wasm_bindgen_futures::JsFuture::from(operation).await?;
        }
        Ok(())
    }

    /// Batch multiple buffer operations for efficiency
    pub fn batch_buffer_operations(
        &mut self,
        operations: &[BufferOperation],
    ) -> Result<Vec<GpuBuffer>, JsValue> {
        let mut results = Vec::new();

        for op in operations {
            match op {
                BufferOperation::Create { data, usage } => {
                    let buffer = self.create_buffer(data, *usage)?;
                    results.push(buffer);
                },
                BufferOperation::CreateEmpty { size, usage } => {
                    let buffer = self.create_empty_buffer(*size, *usage)?;
                    results.push(buffer);
                },
            }
        }

        Ok(results)
    }
}

/// Buffer operation types for batching
#[derive(Debug, Clone)]
pub enum BufferOperation {
    Create { data: Vec<f32>, usage: u32 },
    CreateEmpty { size: usize, usage: u32 },
}
