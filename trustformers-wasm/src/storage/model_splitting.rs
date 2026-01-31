//! Model splitting for chunked loading of large models
//!
//! This module provides functionality to split large transformer models into smaller chunks
//! that can be loaded progressively, reducing memory pressure and startup time.

#![allow(dead_code)]

use js_sys::{ArrayBuffer, Object, Uint8Array};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::format;
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;

use super::StorageError;

/// Splitter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitterConfig {
    pub max_chunk_size_mb: f64,
    pub enable_compression: bool,
    pub enable_lazy_loading: bool,
}

impl Default for SplitterConfig {
    fn default() -> Self {
        Self {
            max_chunk_size_mb: 50.0,
            enable_compression: true,
            enable_lazy_loading: true,
        }
    }
}

/// Initialize the model splitting module
pub fn initialize() -> Result<(), StorageError> {
    // Perform any necessary initialization checks
    web_sys::console::log_1(&"Model splitting module initialized".into());
    Ok(())
}

/// Model chunk configuration
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    pub(crate) max_chunk_size_mb: f64,
    pub(crate) overlap_percentage: f64,
    pub(crate) compression_enabled: bool,
    pub(crate) priority_loading: bool,
    pub(crate) lazy_loading: bool,
}

/// Model splitter for large transformers
#[wasm_bindgen]
pub struct ModelSplitter {
    config: ChunkConfig,
    chunks: Vec<ModelChunk>,
    chunk_metadata: ChunkMetadata,
    loaded_chunks: BTreeMap<String, Vec<u8>>,
    loading_order: Vec<String>,
}

/// Individual model chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelChunk {
    pub id: String,
    pub chunk_type: ChunkType,
    pub size_bytes: usize,
    pub dependencies: Vec<String>,
    pub priority: ChunkPriority,
    pub data: Vec<u8>,
    pub compressed: bool,
    pub checksum: u32,
}

/// Types of model chunks
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ChunkType {
    /// Embedding weights
    Embeddings,
    /// Attention layer weights
    Attention,
    /// Feed-forward network weights
    FeedForward,
    /// Layer normalization parameters
    LayerNorm,
    /// Output projection weights
    OutputProjection,
    /// Positional encodings
    PositionalEncoding,
    /// Vocabulary and tokenizer data
    Vocabulary,
    /// Model configuration
    Config,
    /// Custom layer weights
    Custom,
}

/// Priority levels for chunk loading
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum ChunkPriority {
    Critical = 0, // Must be loaded first (config, vocab)
    High = 1,     // Core model components (embeddings, first layers)
    Medium = 2,   // Middle layers
    Low = 3,      // Later layers, optional components
}

/// Metadata for the split model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    pub model_name: String,
    pub model_version: String,
    pub total_chunks: usize,
    pub total_size_bytes: usize,
    pub chunk_manifest: Vec<ChunkInfo>,
    pub loading_strategy: LoadingStrategy,
    pub dependencies: BTreeMap<String, Vec<String>>,
}

/// Chunk information in manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkInfo {
    pub id: String,
    pub chunk_type: ChunkType,
    pub size_bytes: usize,
    pub priority: ChunkPriority,
    pub url: Option<String>,
    pub dependencies: Vec<String>,
    pub checksum: u32,
}

/// Loading strategy for chunks
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LoadingStrategy {
    /// Load all chunks at once
    Eager,
    /// Load chunks as needed
    Lazy,
    /// Load by priority levels
    Priority,
    /// Load based on usage patterns
    Adaptive,
}

/// Model loading session
#[wasm_bindgen]
pub struct ModelLoadingSession {
    splitter: ModelSplitter,
    loaded_components: BTreeMap<ChunkType, bool>,
    loading_progress: f64,
    total_size: usize,
    loaded_size: usize,
    current_strategy: LoadingStrategy,
}

#[wasm_bindgen]
impl ChunkConfig {
    /// Create a new chunk configuration
    #[wasm_bindgen(constructor)]
    pub fn new() -> ChunkConfig {
        ChunkConfig {
            max_chunk_size_mb: 50.0, // 50MB max per chunk
            overlap_percentage: 5.0, // 5% overlap for continuity
            compression_enabled: true,
            priority_loading: true,
            lazy_loading: true,
        }
    }

    /// Set maximum chunk size in MB
    pub fn set_max_chunk_size_mb(&mut self, size_mb: f64) {
        self.max_chunk_size_mb = size_mb.clamp(1.0, 500.0); // Clamp between 1MB and 500MB
    }

    /// Set overlap percentage for chunk boundaries
    pub fn set_overlap_percentage(&mut self, percentage: f64) {
        self.overlap_percentage = percentage.clamp(0.0, 20.0); // Clamp between 0% and 20%
    }

    /// Enable or disable compression
    pub fn set_compression_enabled(&mut self, enabled: bool) {
        self.compression_enabled = enabled;
    }

    /// Enable or disable priority loading
    pub fn set_priority_loading(&mut self, enabled: bool) {
        self.priority_loading = enabled;
    }

    /// Enable or disable lazy loading
    pub fn set_lazy_loading(&mut self, enabled: bool) {
        self.lazy_loading = enabled;
    }

    #[wasm_bindgen(getter)]
    pub fn max_chunk_size_mb(&self) -> f64 {
        self.max_chunk_size_mb
    }

    #[wasm_bindgen(getter)]
    pub fn overlap_percentage(&self) -> f64 {
        self.overlap_percentage
    }

    #[wasm_bindgen(getter)]
    pub fn compression_enabled(&self) -> bool {
        self.compression_enabled
    }

    #[wasm_bindgen(getter)]
    pub fn priority_loading(&self) -> bool {
        self.priority_loading
    }

    #[wasm_bindgen(getter)]
    pub fn lazy_loading(&self) -> bool {
        self.lazy_loading
    }
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl ModelSplitter {
    /// Create a new model splitter
    #[wasm_bindgen(constructor)]
    pub fn new(config: ChunkConfig) -> ModelSplitter {
        ModelSplitter {
            config,
            chunks: Vec::new(),
            chunk_metadata: ChunkMetadata {
                model_name: String::new(),
                model_version: String::new(),
                total_chunks: 0,
                total_size_bytes: 0,
                chunk_manifest: Vec::new(),
                loading_strategy: LoadingStrategy::Priority,
                dependencies: BTreeMap::new(),
            },
            loaded_chunks: BTreeMap::new(),
            loading_order: Vec::new(),
        }
    }

    /// Split a model into chunks
    pub fn split_model(
        &mut self,
        model_data: &[u8],
        model_name: &str,
        model_version: &str,
    ) -> Result<js_sys::Array, JsValue> {
        web_sys::console::log_1(
            &format!(
                "Splitting model '{}' ({} bytes) into chunks",
                model_name,
                model_data.len()
            )
            .into(),
        );

        self.chunk_metadata.model_name = model_name.to_string();
        self.chunk_metadata.model_version = model_version.to_string();
        self.chunk_metadata.total_size_bytes = model_data.len();

        // Analyze model structure to identify components
        let components = self.analyze_model_structure(model_data)?;

        // Split into chunks based on components and size limits
        self.chunks = self.create_chunks_from_components(model_data, components)?;

        // Generate metadata
        self.generate_chunk_metadata()?;

        // Determine optimal loading order
        self.loading_order = self.calculate_loading_order();

        web_sys::console::log_1(
            &format!(
                "Model split into {} chunks, total size: {} bytes",
                self.chunks.len(),
                model_data.len()
            )
            .into(),
        );

        // Return chunk information as JavaScript array
        self.get_chunk_manifest()
    }

    /// Analyze model structure to identify components
    fn analyze_model_structure(&self, model_data: &[u8]) -> Result<Vec<ModelComponent>, JsValue> {
        // This is a simplified implementation
        // In a real scenario, you'd parse the model format (e.g., ONNX, SafeTensors, etc.)

        let mut components = Vec::new();
        let chunk_size = (self.config.max_chunk_size_mb * 1024.0 * 1024.0) as usize;

        // Estimate component boundaries based on typical transformer architecture
        let total_size = model_data.len();

        // Configuration chunk (small, critical)
        components.push(ModelComponent {
            name: "config".to_string(),
            chunk_type: ChunkType::Config,
            start_offset: 0,
            size_bytes: (total_size / 100).min(1024 * 1024), // ~1% or 1MB max
            priority: ChunkPriority::Critical,
        });

        // Vocabulary chunk (medium priority)
        let vocab_start = components
            .last()
            .expect("components has at least one element after config push")
            .end_offset();
        components.push(ModelComponent {
            name: "vocabulary".to_string(),
            chunk_type: ChunkType::Vocabulary,
            start_offset: vocab_start,
            size_bytes: (total_size * 5 / 100).min(chunk_size), // ~5% of model
            priority: ChunkPriority::Critical,
        });

        // Embeddings (high priority)
        let embed_start = components
            .last()
            .expect("components has at least two elements after vocab push")
            .end_offset();
        components.push(ModelComponent {
            name: "embeddings".to_string(),
            chunk_type: ChunkType::Embeddings,
            start_offset: embed_start,
            size_bytes: (total_size * 15 / 100).min(chunk_size), // ~15% of model
            priority: ChunkPriority::High,
        });

        // Split remaining data into attention and FFN layers
        let remaining_start = components
            .last()
            .expect("components has at least three elements after embeddings push")
            .end_offset();
        let remaining_size = total_size - remaining_start;
        let num_layer_chunks = remaining_size.div_ceil(chunk_size);

        for i in 0..num_layer_chunks {
            let start = remaining_start + i * chunk_size;
            let size = (chunk_size).min(total_size - start);

            if size == 0 {
                break;
            }

            let chunk_type = if i % 2 == 0 { ChunkType::Attention } else { ChunkType::FeedForward };
            let priority = match i {
                0..=2 => ChunkPriority::High,
                3..=6 => ChunkPriority::Medium,
                _ => ChunkPriority::Low,
            };

            components.push(ModelComponent {
                name: format!("layer_{}", i),
                chunk_type,
                start_offset: start,
                size_bytes: size,
                priority,
            });
        }

        Ok(components)
    }

    /// Create chunks from identified components
    fn create_chunks_from_components(
        &self,
        model_data: &[u8],
        components: Vec<ModelComponent>,
    ) -> Result<Vec<ModelChunk>, JsValue> {
        let mut chunks = Vec::new();

        for (i, component) in components.iter().enumerate() {
            let start = component.start_offset;
            let end = (start + component.size_bytes).min(model_data.len());
            let chunk_data = &model_data[start..end];

            let mut final_data = chunk_data.to_vec();
            let compressed = if self.config.compression_enabled && chunk_data.len() > 1024 {
                // Simple compression simulation (in real implementation, use actual compression)
                final_data = self.compress_data(chunk_data)?;
                true
            } else {
                false
            };

            let chunk = ModelChunk {
                id: format!("chunk_{:03}_{}", i, component.name),
                chunk_type: component.chunk_type,
                size_bytes: final_data.len(),
                dependencies: self.calculate_chunk_dependencies(i, component),
                priority: component.priority,
                data: final_data,
                compressed,
                checksum: self.calculate_checksum(chunk_data),
            };

            chunks.push(chunk);
        }

        Ok(chunks)
    }

    /// Simple data compression simulation
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>, JsValue> {
        // In a real implementation, you'd use actual compression like gzip, lz4, etc.
        // For now, we'll simulate compression by reducing size by ~30%
        let compressed_size = (data.len() as f64 * 0.7) as usize;
        let mut compressed = vec![0u8; compressed_size];

        // Copy first part of data as a simulation
        let copy_size = compressed_size.min(data.len());
        compressed[..copy_size].copy_from_slice(&data[..copy_size]);

        Ok(compressed)
    }

    /// Calculate dependencies between chunks
    fn calculate_chunk_dependencies(
        &self,
        chunk_index: usize,
        component: &ModelComponent,
    ) -> Vec<String> {
        let mut dependencies = Vec::new();

        // Config and vocabulary are always required first
        if component.chunk_type != ChunkType::Config
            && component.chunk_type != ChunkType::Vocabulary
        {
            dependencies.push("chunk_000_config".to_string());
            dependencies.push("chunk_001_vocabulary".to_string());
        }

        // Embeddings required for most layers
        if matches!(
            component.chunk_type,
            ChunkType::Attention | ChunkType::FeedForward | ChunkType::OutputProjection
        ) {
            dependencies.push("chunk_002_embeddings".to_string());
        }

        // Sequential dependencies for transformer layers
        if chunk_index > 3
            && matches!(
                component.chunk_type,
                ChunkType::Attention | ChunkType::FeedForward
            )
        {
            dependencies.push(format!("chunk_{:03}_{}", chunk_index - 1, "layer"));
        }

        dependencies
    }

    /// Calculate checksum for data integrity
    fn calculate_checksum(&self, data: &[u8]) -> u32 {
        // Simple CRC32-like checksum
        let mut checksum = 0u32;
        for &byte in data {
            checksum = checksum.wrapping_mul(31).wrapping_add(byte as u32);
        }
        checksum
    }

    /// Generate chunk metadata
    fn generate_chunk_metadata(&mut self) -> Result<(), JsValue> {
        self.chunk_metadata.total_chunks = self.chunks.len();
        self.chunk_metadata.chunk_manifest.clear();

        for chunk in &self.chunks {
            let chunk_info = ChunkInfo {
                id: chunk.id.clone(),
                chunk_type: chunk.chunk_type,
                size_bytes: chunk.size_bytes,
                priority: chunk.priority,
                url: None, // Would be set during deployment
                dependencies: chunk.dependencies.clone(),
                checksum: chunk.checksum,
            };

            self.chunk_metadata.chunk_manifest.push(chunk_info);
        }

        // Set loading strategy based on configuration
        self.chunk_metadata.loading_strategy = if self.config.lazy_loading {
            if self.config.priority_loading {
                LoadingStrategy::Priority
            } else {
                LoadingStrategy::Lazy
            }
        } else {
            LoadingStrategy::Eager
        };

        Ok(())
    }

    /// Calculate optimal loading order
    fn calculate_loading_order(&self) -> Vec<String> {
        let mut order = Vec::new();

        // Sort chunks by priority and dependencies
        let mut chunks_by_priority: Vec<_> = self.chunks.iter().collect();
        chunks_by_priority.sort_by_key(|chunk| (chunk.priority as u8, chunk.id.clone()));

        for chunk in chunks_by_priority {
            order.push(chunk.id.clone());
        }

        order
    }

    /// Get chunk manifest as JavaScript array
    pub fn get_chunk_manifest(&self) -> Result<js_sys::Array, JsValue> {
        let manifest = js_sys::Array::new();

        for chunk_info in &self.chunk_metadata.chunk_manifest {
            let chunk_obj = Object::new();

            js_sys::Reflect::set(&chunk_obj, &"id".into(), &chunk_info.id.clone().into())?;
            js_sys::Reflect::set(
                &chunk_obj,
                &"type".into(),
                &format!("{:?}", chunk_info.chunk_type).into(),
            )?;
            js_sys::Reflect::set(
                &chunk_obj,
                &"size_bytes".into(),
                &(chunk_info.size_bytes as f64).into(),
            )?;
            js_sys::Reflect::set(
                &chunk_obj,
                &"priority".into(),
                &(chunk_info.priority as u8 as f64).into(),
            )?;
            js_sys::Reflect::set(
                &chunk_obj,
                &"checksum".into(),
                &(chunk_info.checksum as f64).into(),
            )?;

            let deps_array = js_sys::Array::new();
            for dep in &chunk_info.dependencies {
                deps_array.push(&dep.into());
            }
            js_sys::Reflect::set(&chunk_obj, &"dependencies".into(), &deps_array)?;

            manifest.push(&chunk_obj);
        }

        Ok(manifest)
    }

    /// Get chunk data by ID
    pub fn get_chunk_data(&self, chunk_id: &str) -> Result<Option<js_sys::Uint8Array>, JsValue> {
        if let Some(chunk) = self.chunks.iter().find(|c| c.id == chunk_id) {
            let array_buffer = ArrayBuffer::new(chunk.data.len() as u32);
            let uint8_view = Uint8Array::new(&array_buffer);
            uint8_view.copy_from(&chunk.data);
            Ok(Some(uint8_view))
        } else {
            Ok(None)
        }
    }

    /// Get loading order
    pub fn get_loading_order(&self) -> js_sys::Array {
        let order_array = js_sys::Array::new();
        for chunk_id in &self.loading_order {
            order_array.push(&chunk_id.into());
        }
        order_array
    }

    /// Get total number of chunks
    #[wasm_bindgen(getter)]
    pub fn total_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Get total size in bytes
    #[wasm_bindgen(getter)]
    pub fn total_size_bytes(&self) -> usize {
        self.chunk_metadata.total_size_bytes
    }

    /// Get model name
    #[wasm_bindgen(getter)]
    pub fn model_name(&self) -> String {
        self.chunk_metadata.model_name.clone()
    }

    /// Export chunks as separate files (returns URLs)
    pub fn export_chunks_as_files(&self) -> Result<js_sys::Array, JsValue> {
        let file_urls = js_sys::Array::new();

        for chunk in &self.chunks {
            // Create blob for each chunk
            let uint8_array = Uint8Array::new(&ArrayBuffer::new(chunk.data.len() as u32));
            uint8_array.copy_from(&chunk.data);

            let blob_parts = js_sys::Array::new();
            blob_parts.push(&uint8_array);

            // BlobPropertyBag not available in web-sys 0.3.81 - using default
            let _blob = web_sys::Blob::new_with_u8_array_sequence(&blob_parts)?;

            // Url not available in web-sys 0.3.81 - using placeholder
            let url = format!("blob:data-chunk-{}", chunk.id);

            let file_info = Object::new();
            js_sys::Reflect::set(&file_info, &"chunk_id".into(), &chunk.id.clone().into())?;
            js_sys::Reflect::set(
                &file_info,
                &"filename".into(),
                &format!("{}.chunk", chunk.id).into(),
            )?;
            js_sys::Reflect::set(&file_info, &"url".into(), &url.into())?;
            js_sys::Reflect::set(
                &file_info,
                &"size_bytes".into(),
                &(chunk.size_bytes as f64).into(),
            )?;

            file_urls.push(&file_info);
        }

        Ok(file_urls)
    }
}

/// Model component during analysis
#[derive(Debug, Clone)]
struct ModelComponent {
    name: String,
    chunk_type: ChunkType,
    start_offset: usize,
    size_bytes: usize,
    priority: ChunkPriority,
}

impl ModelComponent {
    fn end_offset(&self) -> usize {
        self.start_offset + self.size_bytes
    }
}

#[wasm_bindgen]
impl ModelLoadingSession {
    /// Create a new loading session
    #[wasm_bindgen(constructor)]
    pub fn new(splitter: ModelSplitter) -> ModelLoadingSession {
        let total_size = splitter.total_size_bytes();

        ModelLoadingSession {
            splitter,
            loaded_components: BTreeMap::new(),
            loading_progress: 0.0,
            total_size,
            loaded_size: 0,
            current_strategy: LoadingStrategy::Priority,
        }
    }

    /// Load chunks by priority
    pub async fn load_by_priority(&mut self) -> Result<f64, JsValue> {
        let loading_order = self.splitter.get_loading_order();
        let mut loaded_count = 0;
        let total_count = loading_order.length() as usize;

        for i in 0..loading_order.length() {
            let chunk_id = loading_order
                .get(i)
                .as_string()
                .expect("loading order should contain string chunk IDs");

            // Simulate loading delay
            self.simulate_chunk_loading(&chunk_id).await?;

            loaded_count += 1;
            self.loading_progress = (loaded_count as f64 / total_count as f64) * 100.0;

            web_sys::console::log_1(
                &format!(
                    "Loaded chunk {}: {:.1}% complete",
                    chunk_id, self.loading_progress
                )
                .into(),
            );
        }

        Ok(self.loading_progress)
    }

    /// Simulate chunk loading with delay
    async fn simulate_chunk_loading(&mut self, chunk_id: &str) -> Result<(), JsValue> {
        // Simulate network delay based on chunk size
        let delay_ms = if let Some(chunk_data) = self.splitter.get_chunk_data(chunk_id)? {
            let size_mb = chunk_data.length() as f64 / (1024.0 * 1024.0);
            (size_mb * 100.0) as u32 // 100ms per MB simulation
        } else {
            100 // Default delay
        };

        // Create a promise that resolves after delay
        let promise = js_sys::Promise::new(&mut |resolve, _reject| {
            let timeout_id = web_sys::window()
                .unwrap()
                .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, delay_ms as i32)
                .unwrap();

            // Store timeout_id if needed for cleanup
            let _ = timeout_id;
        });

        wasm_bindgen_futures::JsFuture::from(promise).await?;
        Ok(())
    }

    /// Get loading progress percentage
    #[wasm_bindgen(getter)]
    pub fn loading_progress(&self) -> f64 {
        self.loading_progress
    }

    /// Check if a specific component type is loaded
    pub fn is_component_loaded(&self, component_type: ChunkType) -> bool {
        self.loaded_components.get(&component_type).copied().unwrap_or(false)
    }

    /// Get summary of loaded components
    pub fn get_loaded_summary(&self) -> String {
        let loaded_count = self.loaded_components.values().filter(|&&loaded| loaded).count();
        let total_types = 9; // Number of ChunkType variants

        format!(
            "Loaded components: {}/{} ({:.1}% complete)",
            loaded_count, total_types, self.loading_progress
        )
    }
}

/// Check if model splitting is beneficial for a given model size
#[wasm_bindgen]
pub fn should_split_model(model_size_mb: f64, available_memory_mb: f64) -> bool {
    // Split if model is larger than 50% of available memory or larger than 100MB
    model_size_mb > available_memory_mb * 0.5 || model_size_mb > 100.0
}

/// Get recommended chunk size based on model size and available memory
#[wasm_bindgen]
pub fn get_recommended_chunk_size_mb(model_size_mb: f64, available_memory_mb: f64) -> f64 {
    let max_chunk_size = available_memory_mb * 0.3; // Use at most 30% of available memory per chunk
    let min_chunk_size = 10.0; // Minimum 10MB per chunk
    let target_chunks = (model_size_mb / 50.0).ceil(); // Target ~50MB per chunk ideally

    let calculated_size = model_size_mb / target_chunks;
    calculated_size.clamp(min_chunk_size, max_chunk_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_config() {
        let mut config = ChunkConfig::new();
        assert_eq!(config.max_chunk_size_mb(), 50.0);

        config.set_max_chunk_size_mb(100.0);
        assert_eq!(config.max_chunk_size_mb(), 100.0);

        config.set_overlap_percentage(10.0);
        assert_eq!(config.overlap_percentage(), 10.0);
    }

    #[test]
    fn test_should_split_model() {
        assert!(should_split_model(200.0, 300.0)); // 200MB model, 300MB memory -> split
        assert!(!should_split_model(50.0, 500.0)); // 50MB model, 500MB memory -> no split
        assert!(should_split_model(150.0, 200.0)); // 150MB model, 200MB memory -> split
    }

    #[test]
    fn test_recommended_chunk_size() {
        let chunk_size = get_recommended_chunk_size_mb(500.0, 1000.0);
        assert!((10.0..=300.0).contains(&chunk_size)); // Within reasonable bounds
    }
}
