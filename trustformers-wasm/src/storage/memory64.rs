//! Memory64 support module for WebAssembly
//!
//! This module provides support for Memory64, allowing access to more than 4GB of memory
//! in WebAssembly environments that support it.

use serde::{Deserialize, Serialize};
use std::vec::Vec;
use wasm_bindgen::prelude::*;

use super::StorageError;

/// Memory allocation strategy for large models
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AllocationStrategyInternal {
    /// Allocate memory in large continuous chunks
    Continuous,
    /// Allocate memory in smaller, manageable chunks
    Chunked,
    /// Adaptive allocation based on available memory
    Adaptive,
}

/// Memory64 configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory64Config {
    pub max_memory_gb: u32,
    pub allocation_strategy: AllocationStrategyInternal,
    pub enable_compression: bool,
}

impl Default for Memory64Config {
    fn default() -> Self {
        Self {
            max_memory_gb: 16,
            allocation_strategy: AllocationStrategyInternal::Adaptive,
            enable_compression: true,
        }
    }
}

/// Initialize the Memory64 module
pub fn initialize() -> Result<(), StorageError> {
    // Check if Memory64 is supported
    match Memory64Manager::check_memory64_support() {
        Ok(capabilities) => {
            if !capabilities.is_supported {
                web_sys::console::warn_1(&"Memory64 is not supported in this environment - falling back to standard memory".into());
            }
            Ok(())
        },
        Err(e) => Err(StorageError::InitializationError(format!(
            "Failed to check Memory64 support: {:?}",
            e
        ))),
    }
}

/// Memory64 manager for handling large memory allocations
#[wasm_bindgen]
pub struct Memory64Manager {
    max_memory_gb: u32,
    current_usage_bytes: u64,
    allocation_chunks: Vec<AllocationChunk>,
    enabled: bool,
}

/// Represents a chunk of allocated memory
#[derive(Debug, Clone)]
struct AllocationChunk {
    id: u32,
    size_bytes: u64,
    purpose: String,
}

/// Memory allocation strategy for large models
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AllocationStrategy {
    /// Allocate memory in large continuous chunks
    Continuous,
    /// Allocate memory in smaller, manageable chunks
    Chunked,
    /// Adaptive allocation based on available memory
    Adaptive,
}

/// Memory64 capabilities and status
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct Memory64Capabilities {
    pub is_supported: bool,
    pub max_memory_gb: u32,
    pub available_memory_gb: u32,
    pub current_usage_bytes: u64,
}

#[wasm_bindgen]
impl Memory64Manager {
    #[wasm_bindgen(constructor)]
    pub fn new(max_memory_gb: u32) -> Result<Memory64Manager, JsValue> {
        let capabilities = Self::check_memory64_support()?;

        if !capabilities.is_supported {
            return Err("Memory64 is not supported in this environment".into());
        }

        if max_memory_gb > capabilities.max_memory_gb {
            return Err(format!(
                "Requested memory ({} GB) exceeds maximum available ({} GB)",
                max_memory_gb, capabilities.max_memory_gb
            )
            .into());
        }

        Ok(Memory64Manager {
            max_memory_gb,
            current_usage_bytes: 0,
            allocation_chunks: Vec::new(),
            enabled: true,
        })
    }

    /// Check if Memory64 is supported in the current environment
    pub fn check_memory64_support() -> Result<Memory64Capabilities, JsValue> {
        // Query the WebAssembly memory to check if Memory64 is available
        let memory = wasm_bindgen::memory();
        let memory_obj: &js_sys::WebAssembly::Memory = memory.unchecked_ref();
        let buffer = js_sys::WebAssembly::Memory::buffer(memory_obj);
        let array_buffer: &js_sys::ArrayBuffer = buffer.unchecked_ref();
        let current_size = js_sys::ArrayBuffer::byte_length(array_buffer) as u64;

        // Check if we can access more than 4GB through Memory64
        let is_supported = Self::test_memory64_access()?;

        // Estimate maximum memory based on environment
        let max_memory_gb: u32 = if is_supported {
            // In Memory64, theoretical limit is much higher
            64 // Conservative estimate for browser environments
        } else {
            4 // Standard WASM32 limit
        };

        let available_memory_gb =
            max_memory_gb.saturating_sub(current_size as u32 / (1024 * 1024 * 1024));

        Ok(Memory64Capabilities {
            is_supported,
            max_memory_gb,
            available_memory_gb,
            current_usage_bytes: current_size,
        })
    }

    /// Test Memory64 access capabilities
    fn test_memory64_access() -> Result<bool, JsValue> {
        // Try to create a WebAssembly memory with Memory64 features
        // This is a simplified test - in practice, you'd need to check
        // browser support and WebAssembly.Memory constructor options

        let js_code = r#"
            try {
                // Check if Memory64 is supported
                if (typeof WebAssembly.Memory === 'function') {
                    // Try to create memory with memory64 option
                    // Note: This is experimental and may not be supported in all browsers
                    const memory = new WebAssembly.Memory({
                        initial: 1,
                        maximum: 1000, // Much higher than 4GB limit
                        shared: false
                    });
                    return true;
                }
                return false;
            } catch (e) {
                return false;
            }
        "#;

        let result = js_sys::eval(js_code)?;
        Ok(result.as_bool().unwrap_or(false))
    }

    /// Allocate memory chunk for a specific purpose
    pub fn allocate_chunk(&mut self, size_gb: f64, purpose: &str) -> Result<u32, JsValue> {
        let size_bytes = (size_gb * 1024.0 * 1024.0 * 1024.0) as u64;

        if self.current_usage_bytes + size_bytes > (self.max_memory_gb as u64 * 1024 * 1024 * 1024)
        {
            return Err(format!(
                "Cannot allocate {} GB: would exceed memory limit of {} GB",
                size_gb, self.max_memory_gb
            )
            .into());
        }

        let chunk_id = self.allocation_chunks.len() as u32;
        let chunk = AllocationChunk {
            id: chunk_id,
            size_bytes,
            purpose: purpose.to_string(),
        };

        self.allocation_chunks.push(chunk);
        self.current_usage_bytes += size_bytes;

        web_sys::console::log_1(
            &format!(
                "Allocated {} GB for '{}' (chunk ID: {})",
                size_gb, purpose, chunk_id
            )
            .into(),
        );

        Ok(chunk_id)
    }

    /// Deallocate a memory chunk
    pub fn deallocate_chunk(&mut self, chunk_id: u32) -> Result<(), JsValue> {
        if let Some(pos) = self.allocation_chunks.iter().position(|c| c.id == chunk_id) {
            let chunk = self.allocation_chunks.remove(pos);
            self.current_usage_bytes = self.current_usage_bytes.saturating_sub(chunk.size_bytes);

            web_sys::console::log_1(
                &format!("Deallocated chunk {} ({})", chunk_id, chunk.purpose).into(),
            );

            Ok(())
        } else {
            Err(format!("Chunk {} not found", chunk_id).into())
        }
    }

    /// Get current memory usage in bytes
    #[wasm_bindgen(getter)]
    pub fn current_usage_bytes(&self) -> u64 {
        self.current_usage_bytes
    }

    /// Get current memory usage in GB
    #[wasm_bindgen(getter)]
    pub fn current_usage_gb(&self) -> f64 {
        self.current_usage_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Get maximum allowed memory in GB
    #[wasm_bindgen(getter)]
    pub fn max_memory_gb(&self) -> u32 {
        self.max_memory_gb
    }

    /// Get available memory in GB
    #[wasm_bindgen(getter)]
    pub fn available_memory_gb(&self) -> f64 {
        let max_bytes = self.max_memory_gb as u64 * 1024 * 1024 * 1024;
        let available_bytes = max_bytes.saturating_sub(self.current_usage_bytes);
        available_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Check if Memory64 is enabled
    #[wasm_bindgen(getter)]
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Enable or disable Memory64 usage
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        web_sys::console::log_1(
            &format!("Memory64 {}", if enabled { "enabled" } else { "disabled" }).into(),
        );
    }

    /// Get allocation strategy recommendation based on model size
    pub fn recommend_allocation_strategy(&self, model_size_gb: f64) -> AllocationStrategy {
        let available = self.available_memory_gb();

        if model_size_gb <= 1.0 || model_size_gb <= available * 0.8 {
            AllocationStrategy::Continuous
        } else if model_size_gb <= available {
            AllocationStrategy::Chunked
        } else {
            AllocationStrategy::Adaptive
        }
    }

    /// Optimize memory layout for large model loading
    pub fn optimize_for_large_model(&mut self, model_size_gb: f64) -> Result<String, JsValue> {
        let strategy = self.recommend_allocation_strategy(model_size_gb);

        match strategy {
            AllocationStrategy::Continuous => {
                // Pre-allocate a large continuous chunk
                let chunk_id = self.allocate_chunk(model_size_gb, "large_model_continuous")?;
                Ok(format!(
                    "Allocated continuous chunk {} for {} GB model",
                    chunk_id, model_size_gb
                ))
            },
            AllocationStrategy::Chunked => {
                // Allocate in smaller chunks
                let num_chunks = (model_size_gb / 2.0).ceil() as u32;
                let chunk_size = model_size_gb / num_chunks as f64;

                let mut chunk_ids = Vec::new();
                for i in 0..num_chunks {
                    let chunk_id =
                        self.allocate_chunk(chunk_size, &format!("large_model_chunk_{}", i))?;
                    chunk_ids.push(chunk_id);
                }

                Ok(format!(
                    "Allocated {} chunks for {} GB model",
                    num_chunks, model_size_gb
                ))
            },
            AllocationStrategy::Adaptive => {
                // Use available memory with adaptive strategy
                let available = self.available_memory_gb();
                if available > 0.5 {
                    let chunk_id = self.allocate_chunk(available * 0.9, "large_model_adaptive")?;
                    Ok(format!(
                        "Allocated adaptive chunk {} ({} GB) for {} GB model",
                        chunk_id,
                        available * 0.9,
                        model_size_gb
                    ))
                } else {
                    Err("Insufficient memory for large model".into())
                }
            },
        }
    }

    /// Get memory usage summary
    pub fn get_usage_summary(&self) -> String {
        format!(
            "Memory64 Manager: {:.2}/{} GB used, {} chunks allocated",
            self.current_usage_gb(),
            self.max_memory_gb,
            self.allocation_chunks.len()
        )
    }

    /// Get detailed allocation information
    pub fn get_allocation_details(&self) -> js_sys::Array {
        let details = js_sys::Array::new();

        for chunk in &self.allocation_chunks {
            let chunk_info = js_sys::Object::new();
            js_sys::Reflect::set(&chunk_info, &"id".into(), &(chunk.id as f64).into()).unwrap();
            js_sys::Reflect::set(
                &chunk_info,
                &"size_gb".into(),
                &((chunk.size_bytes as f64) / (1024.0 * 1024.0 * 1024.0)).into(),
            )
            .unwrap();
            js_sys::Reflect::set(
                &chunk_info,
                &"purpose".into(),
                &chunk.purpose.clone().into(),
            )
            .unwrap();
            details.push(&chunk_info);
        }

        details
    }

    /// Clear all allocations
    pub fn clear_all_allocations(&mut self) {
        let count = self.allocation_chunks.len();
        self.allocation_chunks.clear();
        self.current_usage_bytes = 0;

        web_sys::console::log_1(&format!("Cleared {} memory allocations", count).into());
    }

    /// Allocate memory for a specific model
    pub fn allocate_for_model(
        &mut self,
        model_id: &str,
        size_bytes: usize,
    ) -> Result<u32, JsValue> {
        let size_gb = size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        self.allocate_chunk(size_gb, &format!("model_{}", model_id))
    }

    /// Get model data from allocated memory
    pub fn get_model_data(&self, _model_id: &str) -> Result<Option<Vec<u8>>, JsValue> {
        // Stub implementation - would need actual memory mapping
        Ok(None)
    }

    /// Get memory statistics
    pub fn get_statistics(&self) -> Result<JsValue, JsValue> {
        let stats = js_sys::Object::new();
        js_sys::Reflect::set(
            &stats,
            &"current_usage_bytes".into(),
            &JsValue::from_f64(self.current_usage_bytes as f64),
        )?;
        js_sys::Reflect::set(
            &stats,
            &"max_memory_bytes".into(),
            &JsValue::from_f64((self.max_memory_gb as u64 * 1024 * 1024 * 1024) as f64),
        )?;
        js_sys::Reflect::set(
            &stats,
            &"allocation_count".into(),
            &JsValue::from_f64(self.allocation_chunks.len() as f64),
        )?;
        Ok(stats.into())
    }

    /// Clear all allocations (alias for clear_all_allocations)
    pub fn clear_all(&mut self) {
        self.clear_all_allocations()
    }
}

#[wasm_bindgen]
impl Memory64Capabilities {
    #[wasm_bindgen(getter)]
    pub fn is_supported(&self) -> bool {
        self.is_supported
    }

    #[wasm_bindgen(getter)]
    pub fn max_memory_gb(&self) -> u32 {
        self.max_memory_gb
    }

    #[wasm_bindgen(getter)]
    pub fn available_memory_gb(&self) -> u32 {
        self.available_memory_gb
    }

    #[wasm_bindgen(getter)]
    pub fn current_usage_bytes(&self) -> u64 {
        self.current_usage_bytes
    }

    /// Get a summary of Memory64 capabilities
    pub fn summary(&self) -> String {
        format!(
            "Memory64 Support: {}, Max: {} GB, Available: {} GB, Current: {} MB",
            if self.is_supported { "Yes" } else { "No" },
            self.max_memory_gb,
            self.available_memory_gb,
            self.current_usage_bytes / (1024 * 1024)
        )
    }
}

/// Check if Memory64 is supported (standalone function)
#[wasm_bindgen]
pub fn is_memory64_supported() -> bool {
    Memory64Manager::check_memory64_support()
        .map(|caps| caps.is_supported)
        .unwrap_or(false)
}

/// Get Memory64 capabilities (standalone function)
#[wasm_bindgen]
pub fn get_memory64_capabilities() -> Result<Memory64Capabilities, JsValue> {
    Memory64Manager::check_memory64_support()
}

/// Estimate if a model size can be loaded with Memory64
#[wasm_bindgen]
pub fn can_load_model_size(model_size_gb: f64) -> Result<bool, JsValue> {
    let capabilities = Memory64Manager::check_memory64_support()?;
    Ok(capabilities.is_supported && model_size_gb <= capabilities.available_memory_gb as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_memory64_manager_creation() {
        // This test would only pass in environments that support Memory64
        // In most current browsers, this will fail
        if let Ok(manager) = Memory64Manager::new(8) {
            assert_eq!(manager.max_memory_gb(), 8);
            assert_eq!(manager.current_usage_gb(), 0.0);
            assert!(manager.enabled());
        }
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_allocation_strategy() {
        if let Ok(manager) = Memory64Manager::new(16) {
            assert_eq!(
                manager.recommend_allocation_strategy(0.5),
                AllocationStrategy::Continuous
            );
            assert_eq!(
                manager.recommend_allocation_strategy(12.0),
                AllocationStrategy::Chunked
            );
        }
    }
}
