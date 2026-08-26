//! Memory64 support module for WebAssembly
//!
//! This module provides support for Memory64, allowing access to more than 4GB of memory
//! in WebAssembly environments that support it.
//!
//! ## Design: JS-free core, thin `wasm_bindgen` wrappers
//!
//! Every `#[wasm_bindgen]`-exposed method here that used to do real work also
//! constructed a `JsValue` (via `Result<_, JsValue>` or `web_sys::console`)
//! somewhere on its success path. `JsValue` construction unconditionally
//! panics on non-wasm32 targets (see `lib.rs`'s `InferenceSession` doc
//! comments for the same constraint), which is exactly why this module's
//! prior tests were `#[cfg(target_arch = "wasm32")]`-gated and therefore
//! never actually ran under `cargo test` / `cargo nextest`.
//!
//! To make the real bookkeeping in this module natively testable, the
//! allocation/storage logic lives in `pub(crate)` "core" methods that return
//! `Result<_, String>` (or plain values) and never touch `JsValue` or
//! `web_sys`. The `#[wasm_bindgen]`-exposed methods are thin wrappers that
//! call the core, convert `String` errors to `JsValue`, and do any
//! browser-console logging.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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
    /// Real byte storage for models allocated via `allocate_for_model`,
    /// keyed by `model_id`. Previously this storage did not exist at all -
    /// `allocate_for_model` recorded only a `size_bytes` count and the
    /// caller's actual bytes were discarded - so `get_model_data` had
    /// nothing to return and unconditionally returned `Ok(None)` regardless
    /// of what had been "allocated". A plain `HashMap` - no browser API
    /// involved - so storing and retrieving is fully testable natively.
    model_data: HashMap<String, Vec<u8>>,
}

/// Represents a chunk of allocated memory
#[derive(Debug, Clone)]
struct AllocationChunk {
    id: u32,
    size_bytes: u64,
    purpose: String,
    /// Set when this chunk backs a model registered via
    /// `allocate_for_model_core`, so deallocating the chunk also evicts the
    /// matching entry from `model_data`.
    model_id: Option<String>,
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

impl Memory64Manager {
    /// Pure, JS-free constructor: real bookkeeping state with no browser
    /// API calls, so it - and everything built on it - is unit-testable on
    /// native targets. Used both by the `#[wasm_bindgen]`-exposed `new`
    /// (after a real `check_memory64_support` probe) and directly by native
    /// tests below.
    pub(crate) fn with_capacity_gb(max_memory_gb: u32) -> Self {
        Self {
            max_memory_gb,
            current_usage_bytes: 0,
            allocation_chunks: Vec::new(),
            enabled: true,
            model_data: HashMap::new(),
        }
    }

    /// Pure core of chunk allocation: real bookkeeping, `String` errors
    /// only (never `JsValue`).
    pub(crate) fn allocate_chunk_core(
        &mut self,
        size_bytes: u64,
        purpose: String,
        model_id: Option<String>,
    ) -> Result<u32, String> {
        let limit_bytes = self.max_memory_gb as u64 * 1024 * 1024 * 1024;
        if self.current_usage_bytes + size_bytes > limit_bytes {
            return Err(format!(
                "Cannot allocate {size_bytes} bytes for '{purpose}': would exceed memory limit of {} GB",
                self.max_memory_gb
            ));
        }

        let chunk_id = self.allocation_chunks.len() as u32;
        self.allocation_chunks.push(AllocationChunk {
            id: chunk_id,
            size_bytes,
            purpose,
            model_id,
        });
        self.current_usage_bytes += size_bytes;

        Ok(chunk_id)
    }

    /// Pure core of chunk deallocation. Also evicts any `model_data` entry
    /// owned by the deallocated chunk, so freeing a model's chunk cannot
    /// leave stale bytes reachable through `get_model_data_core`.
    pub(crate) fn deallocate_chunk_core(&mut self, chunk_id: u32) -> Result<(), String> {
        let pos = self
            .allocation_chunks
            .iter()
            .position(|c| c.id == chunk_id)
            .ok_or_else(|| format!("Chunk {chunk_id} not found"))?;
        let chunk = self.allocation_chunks.remove(pos);
        self.current_usage_bytes = self.current_usage_bytes.saturating_sub(chunk.size_bytes);
        if let Some(ref model_id) = chunk.model_id {
            self.model_data.remove(model_id);
        }
        Ok(())
    }

    /// Pure core of `allocate_for_model`: records real bookkeeping *and*
    /// stores `data` itself, so `get_model_data_core` can return it later.
    pub(crate) fn allocate_for_model_core(
        &mut self,
        model_id: &str,
        data: &[u8],
    ) -> Result<u32, String> {
        let chunk_id = self.allocate_chunk_core(
            data.len() as u64,
            format!("model_{model_id}"),
            Some(model_id.to_string()),
        )?;
        self.model_data.insert(model_id.to_string(), data.to_vec());
        Ok(chunk_id)
    }

    /// Pure core of `get_model_data`: a real lookup into `model_data`,
    /// rather than the unconditional `Ok(None)` this used to return.
    pub(crate) fn get_model_data_core(&self, model_id: &str) -> Option<Vec<u8>> {
        self.model_data.get(model_id).cloned()
    }
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

        Ok(Self::with_capacity_gb(max_memory_gb))
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
        let chunk_id = self
            .allocate_chunk_core(size_bytes, purpose.to_string(), None)
            .map_err(|e| JsValue::from_str(&e))?;

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
        // Look up the purpose before the core call removes the chunk, purely
        // for the log message below.
        let purpose = self
            .allocation_chunks
            .iter()
            .find(|c| c.id == chunk_id)
            .map(|c| c.purpose.clone());

        self.deallocate_chunk_core(chunk_id).map_err(|e| JsValue::from_str(&e))?;

        web_sys::console::log_1(
            &format!(
                "Deallocated chunk {} ({})",
                chunk_id,
                purpose.as_deref().unwrap_or("unknown")
            )
            .into(),
        );

        Ok(())
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
            let _ = js_sys::Reflect::set(&chunk_info, &"id".into(), &(chunk.id as f64).into());
            let _ = js_sys::Reflect::set(
                &chunk_info,
                &"size_gb".into(),
                &((chunk.size_bytes as f64) / (1024.0 * 1024.0 * 1024.0)).into(),
            );
            let _ = js_sys::Reflect::set(
                &chunk_info,
                &"purpose".into(),
                &chunk.purpose.clone().into(),
            );
            details.push(&chunk_info);
        }

        details
    }

    /// Clear all allocations
    pub fn clear_all_allocations(&mut self) {
        let count = self.allocation_chunks.len();
        self.allocation_chunks.clear();
        self.current_usage_bytes = 0;
        self.model_data.clear();

        web_sys::console::log_1(&format!("Cleared {} memory allocations", count).into());
    }

    /// Allocate memory for a specific model and store its bytes so they can
    /// be retrieved later via [`Self::get_model_data`].
    ///
    /// Previously this took only a `size_bytes: usize` count and discarded
    /// the caller's actual data - so nothing was ever available for
    /// `get_model_data` to return. It now takes the real bytes and stores
    /// them (see `Self::allocate_for_model_core`).
    pub fn allocate_for_model(&mut self, model_id: &str, data: &[u8]) -> Result<u32, JsValue> {
        self.allocate_for_model_core(model_id, data).map_err(|e| JsValue::from_str(&e))
    }

    /// Get model data from allocated memory.
    ///
    /// Previously a stub that unconditionally returned `Ok(None)` regardless
    /// of what had been "allocated" via `allocate_for_model`. Now returns
    /// the real bytes stored by [`Self::allocate_for_model`], if any.
    pub fn get_model_data(&self, model_id: &str) -> Result<Option<Vec<u8>>, JsValue> {
        Ok(self.get_model_data_core(model_id))
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

    // -----------------------------------------------------------------
    // All tests below exercise the pure, JS-free "core" methods directly
    // (`with_capacity_gb`, `allocate_chunk_core`, `allocate_for_model_core`,
    // `get_model_data_core`, `deallocate_chunk_core`) so they run natively
    // under `cargo test` / `cargo nextest`, unlike the module's previous
    // tests which were `#[cfg(target_arch = "wasm32")]`-gated (because they
    // called `Memory64Manager::new`, which calls `check_memory64_support`,
    // which calls `js_sys::eval` - unusable off wasm32) and therefore never
    // actually ran in CI.
    // -----------------------------------------------------------------

    #[test]
    fn test_with_capacity_gb_starts_empty() {
        let manager = Memory64Manager::with_capacity_gb(8);
        assert_eq!(manager.max_memory_gb, 8);
        assert_eq!(manager.current_usage_bytes, 0);
        assert!(manager.enabled);
        assert!(manager.allocation_chunks.is_empty());
        assert!(manager.model_data.is_empty());
    }

    #[test]
    fn test_get_model_data_core_round_trips_real_bytes() {
        // Regression test for the old `get_model_data`, which unconditionally
        // returned `Ok(None)` no matter what had been allocated.
        let mut manager = Memory64Manager::with_capacity_gb(4);
        let payload = vec![1u8, 2, 3, 4, 5, 42, 255, 0];

        manager
            .allocate_for_model_core("model-a", &payload)
            .expect("allocation within limit should succeed");

        let retrieved = manager.get_model_data_core("model-a");
        assert_eq!(retrieved, Some(payload));
    }

    #[test]
    fn test_get_model_data_core_unknown_model_returns_none() {
        let manager = Memory64Manager::with_capacity_gb(4);
        assert_eq!(manager.get_model_data_core("does-not-exist"), None);
    }

    #[test]
    fn test_allocate_for_model_core_tracks_usage_bytes() {
        let mut manager = Memory64Manager::with_capacity_gb(4);
        let payload = vec![0u8; 1024];

        manager.allocate_for_model_core("model-a", &payload).expect("should succeed");

        assert_eq!(manager.current_usage_bytes, 1024);
    }

    #[test]
    fn test_allocate_for_model_core_rejects_over_limit() {
        // 1 GB limit; request one byte more than that via the byte-count
        // core directly (no need to actually materialize a multi-GB Vec in
        // a test process).
        let mut manager = Memory64Manager::with_capacity_gb(1);
        let limit_bytes = 1024u64 * 1024 * 1024;
        let result = manager.allocate_chunk_core(limit_bytes + 1, "oversized".to_string(), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_deallocate_chunk_core_evicts_model_data() {
        let mut manager = Memory64Manager::with_capacity_gb(4);
        let payload = vec![9u8; 16];

        let chunk_id = manager
            .allocate_for_model_core("model-a", &payload)
            .expect("allocation should succeed");

        assert_eq!(manager.get_model_data_core("model-a"), Some(payload));

        manager.deallocate_chunk_core(chunk_id).expect("deallocation should succeed");

        // The whole point of this test: after deallocating the chunk that
        // backed "model-a", its bytes must no longer be retrievable.
        assert_eq!(manager.get_model_data_core("model-a"), None);
        assert_eq!(manager.current_usage_bytes, 0);
    }

    #[test]
    fn test_deallocate_chunk_core_unknown_id_errors() {
        let mut manager = Memory64Manager::with_capacity_gb(4);
        assert!(manager.deallocate_chunk_core(999).is_err());
    }

    #[test]
    fn test_multiple_models_stored_independently() {
        let mut manager = Memory64Manager::with_capacity_gb(4);
        manager.allocate_for_model_core("model-a", &[1, 2, 3]).expect("ok");
        manager.allocate_for_model_core("model-b", &[4, 5, 6, 7]).expect("ok");

        assert_eq!(manager.get_model_data_core("model-a"), Some(vec![1, 2, 3]));
        assert_eq!(
            manager.get_model_data_core("model-b"),
            Some(vec![4, 5, 6, 7])
        );
        assert_eq!(manager.current_usage_bytes, 7);
    }

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
