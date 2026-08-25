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

/// Initialize the model splitting module.
///
/// `.into()` on a `&str` constructs a `JsValue`, which unconditionally
/// panics off wasm32 (no JS engine backs it there) in a context Rust
/// cannot unwind out of, aborting the whole process — the exact defect
/// class this module's `split_model`/`load_by_priority` already guard
/// against with the same `#[cfg(target_arch = "wasm32")]`.
#[cfg(target_arch = "wasm32")]
pub fn initialize() -> Result<(), StorageError> {
    web_sys::console::log_1(&"Model splitting module initialized".into());
    Ok(())
}

/// Native build of [`initialize`]: no browser console to log to, and
/// nothing else to check, so this is an unconditional success rather
/// than a silent no-op standing in for real work.
#[cfg(not(target_arch = "wasm32"))]
pub fn initialize() -> Result<(), StorageError> {
    Ok(())
}

/// Real DEFLATE compression (pure Rust, via `oxiarc_deflate`). Pure
/// (`String`-erroring, no `JsValue`) so it is unit tested directly; see
/// [`ModelSplitter::compress_data`] for the `JsValue`-wrapping boundary
/// used from `#[wasm_bindgen]` methods.
fn compress_bytes(data: &[u8]) -> Result<Vec<u8>, String> {
    oxiarc_deflate::deflate(data, 6).map_err(|e| format!("chunk compression failed: {e}"))
}

/// Inverse of [`compress_bytes`].
fn decompress_bytes(data: &[u8]) -> Result<Vec<u8>, String> {
    oxiarc_deflate::inflate(data).map_err(|e| format!("chunk decompression failed: {e}"))
}

/// Calculate a checksum for chunk data integrity. Not cryptographic, but
/// order- and content-sensitive (unlike a byte sum or XOR), which is
/// enough to catch the truncation/corruption this module's compression
/// pipeline used to introduce.
fn calculate_checksum(data: &[u8]) -> u32 {
    let mut checksum = 0u32;
    for &byte in data {
        checksum = checksum.wrapping_mul(31).wrapping_add(byte as u32);
    }
    checksum
}

/// Resolve a chunk by id and return its real, decompressed bytes with the
/// checksum verified against them — never the raw, possibly-compressed
/// bytes handed back as-is. `Ok(None)` for an unknown id (matches the
/// public API's optional-return shape); `Err` for a decompression failure
/// or a checksum mismatch — never silently-corrupt bytes.
///
/// This used to not exist at all: `get_chunk_data` returned `chunk.data`
/// directly with no decompression step anywhere in this module, so any
/// chunk marked `compressed: true` was unusable to every caller (and, with
/// the old fake `compress_data`, permanently missing 30% of its content
/// regardless).
fn resolve_chunk_data(chunks: &[ModelChunk], chunk_id: &str) -> Result<Option<Vec<u8>>, String> {
    let Some(chunk) = chunks.iter().find(|c| c.id == chunk_id) else {
        return Ok(None);
    };
    let raw = if chunk.compressed {
        decompress_bytes(&chunk.data)?
    } else {
        chunk.data.clone()
    };

    let actual_checksum = calculate_checksum(&raw);
    if actual_checksum != chunk.checksum {
        return Err(format!(
            "chunk '{chunk_id}' failed integrity check: expected checksum {}, got {actual_checksum}",
            chunk.checksum
        ));
    }
    Ok(Some(raw))
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
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(
            &format!(
                "Splitting model '{}' ({} bytes) into chunks",
                model_name,
                model_data.len()
            )
            .into(),
        );

        self.split_model_inner(model_data, model_name, model_version)?;

        #[cfg(target_arch = "wasm32")]
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

    /// Pure core of [`Self::split_model`]: analyze + chunk + generate
    /// metadata + compute loading order, populating `self.chunks`. Its
    /// `Result` error type is inherited from the `?`-chained calls below
    /// (kept as `JsValue` rather than plumbing a separate `String` error
    /// type through this whole call chain), but none of them construct a
    /// `JsValue` on their success path, so — unlike `split_model` itself,
    /// which also builds real `js_sys::Object`/`Array` values via
    /// `get_chunk_manifest()` (real JS-heap operations only meaningful with
    /// an actual JS engine) — this is safe to call directly from native
    /// tests and is what they use to exercise real chunking + compression +
    /// checksum + reassembly end to end.
    fn split_model_inner(
        &mut self,
        model_data: &[u8],
        model_name: &str,
        model_version: &str,
    ) -> Result<(), JsValue> {
        self.chunk_metadata.model_name = model_name.to_string();
        self.chunk_metadata.model_version = model_version.to_string();
        self.chunk_metadata.total_size_bytes = model_data.len();

        let components = self.analyze_model_structure(model_data)?;
        self.chunks = self.create_chunks_from_components(model_data, components)?;
        self.generate_chunk_metadata()?;
        self.loading_order = self.calculate_loading_order();
        Ok(())
    }

    /// Analyze model structure to identify real component boundaries.
    ///
    /// A previous version invented a transformer layout purely from
    /// `model_data.len()` — "the first 1% is config", "the next 5% is
    /// vocabulary", "the next 15% is embeddings", remaining bytes
    /// alternating `Attention`/`FeedForward` by chunk-index parity —
    /// without reading a single byte of `model_data`. None of those
    /// labels reflected anything actually in the buffer.
    ///
    /// Now: when `model_data` is a recognizable SafeTensors buffer (a
    /// real structural check — [`formats::looks_like_safetensors`], not a
    /// guess), every component below is genuine: one component for the
    /// header itself (real bytes `[0, header_end)`, which must be read
    /// before any tensor's shape/dtype/offset is known), then one
    /// component per tensor with its real name and its exact
    /// `data_offsets` byte range straight from the file's own header —
    /// see [`Self::components_from_safetensors_layout`].
    ///
    /// Otherwise there is nothing in the bytes to read a layout from, and
    /// the honest choice is to say so structurally rather than guess: see
    /// [`Self::uniform_fallback_components`] for the equal-sized,
    /// no-semantic-claim fallback.
    fn analyze_model_structure(&self, model_data: &[u8]) -> Result<Vec<ModelComponent>, JsValue> {
        use crate::core::model::formats;

        if formats::looks_like_safetensors(model_data) {
            // Magic bytes matched, so this buffer claims to be
            // SafeTensors: a header that then fails to parse is real
            // corruption worth surfacing as an error, not something to
            // paper over with the uniform fallback.
            let (header_end, layout) =
                formats::safetensors_component_layout(model_data).map_err(|e| {
                    JsValue::from_str(&format!(
                        "model_data looks like safetensors (valid magic bytes) but its header \
                         failed to parse: {e}"
                    ))
                })?;
            return Ok(Self::components_from_safetensors_layout(
                model_data.len(),
                header_end,
                &layout,
            ));
        }

        Ok(Self::uniform_fallback_components(
            model_data.len(),
            self.config.max_chunk_size_mb,
        ))
    }

    /// Build real components from a genuine SafeTensors byte layout: one
    /// `Config`-typed component for the header (`[0, header_end)`, must
    /// load first — nothing else can be interpreted without it), then one
    /// `Custom`-typed component per tensor, named with the tensor's real
    /// name and sized/offset exactly as declared in the file.
    ///
    /// `chunk_type` stays `Custom` rather than guessing `Embeddings` /
    /// `Attention` / `FeedForward` from the tensor name: naming
    /// conventions vary across architectures (GPT-2, BERT, T5, LLaMA-style
    /// checkpoints all differ), and this module has no per-architecture
    /// tensor-name table to consult — a wrong guess would be exactly the
    /// kind of fabricated semantic label this rewrite removes. `priority`
    /// is real-position-derived (see [`Self::position_priority`]), the
    /// same honest basis the uniform fallback uses.
    fn components_from_safetensors_layout(
        total_size: usize,
        header_end: usize,
        layout: &[(String, usize, usize)],
    ) -> Vec<ModelComponent> {
        let mut components = Vec::with_capacity(layout.len() + 1);

        components.push(ModelComponent {
            name: "safetensors_header".to_string(),
            chunk_type: ChunkType::Config,
            start_offset: 0,
            size_bytes: header_end,
            priority: ChunkPriority::Critical,
        });

        for (name, start, end) in layout {
            components.push(ModelComponent {
                name: name.clone(),
                chunk_type: ChunkType::Custom,
                start_offset: *start,
                size_bytes: end.saturating_sub(*start),
                priority: Self::position_priority(*start, total_size),
            });
        }

        components
    }

    /// Honest fallback for a format this module cannot parse: equal-sized
    /// chunks tiling `[0, total_size)` exactly, with no gaps or overlap
    /// (unlike [`Self::components_from_safetensors_layout`], which
    /// inherits whatever the file's own `data_offsets` declare — the
    /// SafeTensors format permits padding between tensors, so that path
    /// makes no such guarantee). Chunks here are named only by position
    /// (`chunk_0`, `chunk_1`, ... never a semantic label like
    /// "embeddings" this function has no evidence for), typed `Custom`,
    /// with `priority` derived purely from position via
    /// [`Self::position_priority`].
    fn uniform_fallback_components(
        total_size: usize,
        max_chunk_size_mb: f64,
    ) -> Vec<ModelComponent> {
        let chunk_size = ((max_chunk_size_mb * 1024.0 * 1024.0) as usize).max(1);

        let mut components = Vec::new();
        let mut start = 0usize;
        let mut index = 0usize;
        while start < total_size {
            let size = chunk_size.min(total_size - start);
            components.push(ModelComponent {
                name: format!("chunk_{index}"),
                chunk_type: ChunkType::Custom,
                start_offset: start,
                size_bytes: size,
                priority: Self::position_priority(start, total_size),
            });
            start += size;
            index += 1;
        }
        components
    }

    /// Priority derived only from byte position — never a claim about
    /// architectural importance this module has no evidence for. Earlier
    /// bytes are ranked to load first, which is a real, defensible
    /// loading-order strategy (a header/prefix is more likely to gate
    /// interpreting the rest of the file) independent of what the bytes
    /// actually contain.
    fn position_priority(start_offset: usize, total_size: usize) -> ChunkPriority {
        if total_size == 0 {
            return ChunkPriority::Low;
        }
        match start_offset as f64 / total_size as f64 {
            f if f < 0.10 => ChunkPriority::Critical,
            f if f < 0.35 => ChunkPriority::High,
            f if f < 0.70 => ChunkPriority::Medium,
            _ => ChunkPriority::Low,
        }
    }

    /// Create chunks from identified components
    fn create_chunks_from_components(
        &self,
        model_data: &[u8],
        components: Vec<ModelComponent>,
    ) -> Result<Vec<ModelChunk>, JsValue> {
        let mut chunks: Vec<ModelChunk> = Vec::new();

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

            // Computed from `chunks` built so far (real chunk ids), not
            // `component`/`i` alone — see `calculate_chunk_dependencies`.
            let dependencies = Self::calculate_chunk_dependencies(&chunks, component);

            let chunk = ModelChunk {
                id: format!("chunk_{:03}_{}", i, component.name),
                chunk_type: component.chunk_type,
                size_bytes: final_data.len(),
                dependencies,
                priority: component.priority,
                data: final_data,
                compressed,
                checksum: calculate_checksum(chunk_data),
            };

            chunks.push(chunk);
        }

        Ok(chunks)
    }

    /// Real chunk compression via `oxiarc_deflate` (pure-Rust DEFLATE;
    /// COOLJAPAN policy forbids `flate2`/`zstd`/`lz4` directly). Every byte
    /// of `data` is encoded; see [`decompress_bytes`] for the inverse.
    ///
    /// This used to be `let compressed_size = (data.len() as f64 * 0.7) as
    /// usize; compressed[..copy_size].copy_from_slice(&data[..copy_size])`
    /// — an unconditional 30%-of-every-chunk truncation with no inverse
    /// operation anywhere in this module, so any chunk marked `compressed:
    /// true` was silently missing its last 30% forever.
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>, JsValue> {
        compress_bytes(data).map_err(|e| JsValue::from_str(&e))
    }

    /// Inverse of [`Self::compress_data`].
    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>, JsValue> {
        decompress_bytes(data).map_err(|e| JsValue::from_str(&e))
    }

    /// Calculate dependencies between chunks.
    ///
    /// A previous version hardcoded three specific ids —
    /// `"chunk_000_config"`, `"chunk_001_vocabulary"`, `"chunk_002_embeddings"`
    /// — as prerequisites for every other chunk, and a "chunk_index > 3"
    /// rule for a `layer` chunk one index back. Those ids only ever
    /// existed because [`Self::analyze_model_structure`] used to *always*
    /// emit config/vocabulary/embeddings components at exactly those
    /// three positions; now that it emits real, format-derived components
    /// instead, those ids are usually dangling references to chunks that
    /// were never created.
    ///
    /// The one dependency this module can still state honestly: every
    /// SafeTensors tensor chunk needs the `Config`-typed header chunk —
    /// without its shape/dtype/offset table, a tensor's raw bytes cannot
    /// be interpreted — so any non-`Config` chunk depends on whichever
    /// `Config`-typed chunk already exists in this split, if any. The
    /// header chunk itself, and every chunk from the format-unaware
    /// uniform fallback (no header exists to depend on), get no
    /// dependencies: this module has no evidence of any other ordering
    /// constraint between opaque, equally-uninterpreted byte ranges.
    fn calculate_chunk_dependencies(
        chunks_so_far: &[ModelChunk],
        component: &ModelComponent,
    ) -> Vec<String> {
        if component.chunk_type == ChunkType::Config {
            return Vec::new();
        }

        chunks_so_far
            .iter()
            .find(|chunk| chunk.chunk_type == ChunkType::Config)
            .map(|header_chunk| std::vec![header_chunk.id.clone()])
            .unwrap_or_default()
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
        let raw = resolve_chunk_data(&self.chunks, chunk_id).map_err(|e| JsValue::from_str(&e))?;
        let Some(raw) = raw else {
            return Ok(None);
        };
        let array_buffer = ArrayBuffer::new(raw.len() as u32);
        let uint8_view = Uint8Array::new(&array_buffer);
        uint8_view.copy_from(&raw);
        Ok(Some(uint8_view))
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

            // A real `Blob`, kept alive by the object URL created below —
            // `web_sys::Url::create_object_url_with_blob` (used here) is
            // available in the web-sys version this crate depends on (the
            // "Url not available" comment this replaced was stale). The old
            // code discarded the blob (`let _blob = ...`) entirely and
            // fabricated `format!("blob:data-chunk-{}", chunk.id)` — a
            // string that merely looks like a blob URL but was never
            // registered with the browser, so it could never actually be
            // fetched/opened.
            let blob = web_sys::Blob::new_with_u8_array_sequence(&blob_parts)?;
            let url = web_sys::Url::create_object_url_with_blob(&blob)?;

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

    /// Pure-Rust core of [`Self::load_by_priority`]: takes the loading
    /// order and chunk list directly (no `js_sys`/`JsValue`), so it can be
    /// exercised by native unit tests — the same wasm-facing/pure-core
    /// split `split_model`/`split_model_inner` already use elsewhere in
    /// this file.
    ///
    /// Materializes each chunk's real, decompressed bytes (via
    /// [`resolve_chunk_data`], which also verifies the chunk's checksum)
    /// and records the real result in `loaded_components`/`loaded_size` —
    /// a previous version instead ran a `setTimeout` sized at "100ms per
    /// MB" per chunk and drove `loading_progress` to 100% without ever
    /// touching `loaded_components`/`loaded_size`, so the reported
    /// progress contradicted the session's own (untouched) state. The
    /// chunk bytes are already resident in `chunks` (populated by
    /// `split_model`), so there is no real network I/O here — loading is
    /// synchronous CPU work (decompression + checksum verification), never
    /// an invented delay.
    fn load_by_priority_inner(
        loading_order: &[String],
        chunks: &[ModelChunk],
        total_size: usize,
        loaded_components: &mut BTreeMap<ChunkType, bool>,
        loaded_size: &mut usize,
    ) -> Result<f64, String> {
        let mut progress = 0.0;

        for chunk_id in loading_order {
            let chunk = chunks
                .iter()
                .find(|c| &c.id == chunk_id)
                .ok_or_else(|| format!("loading order references unknown chunk id '{chunk_id}'"))?;

            // Real decompression + checksum verification (never a fake
            // delay) — errors (a bad checksum, a corrupt compressed
            // stream) propagate instead of being reported as loaded.
            let data = resolve_chunk_data(chunks, chunk_id)?
                .ok_or_else(|| format!("chunk '{chunk_id}' not found while loading"))?;

            loaded_components.insert(chunk.chunk_type, true);
            *loaded_size += data.len();
            progress = if total_size > 0 {
                ((*loaded_size as f64 / total_size as f64) * 100.0).min(100.0)
            } else {
                100.0
            };
        }

        Ok(progress)
    }

    /// Load chunks by priority. This stays `async fn` only to keep the
    /// existing public API stable for JS callers that already `await` it —
    /// see [`Self::load_by_priority_inner`] for why loading itself does
    /// not need to await anything.
    pub async fn load_by_priority(&mut self) -> Result<f64, JsValue> {
        // A plain `Vec<String>`, read directly off the (same-module) private
        // field rather than through `ModelSplitter::get_loading_order`'s
        // `js_sys::Array` — that indirection buys nothing here and would
        // only add unnecessary JS-value construction to the hot path.
        let loading_order = self.splitter.loading_order.clone();
        let total_count = loading_order.len();

        let progress = Self::load_by_priority_inner(
            &loading_order,
            &self.splitter.chunks,
            self.total_size,
            &mut self.loaded_components,
            &mut self.loaded_size,
        )
        .map_err(|e| JsValue::from_str(&e))?;
        self.loading_progress = progress;

        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(
            &format!(
                "Loaded {total_count} chunk(s) by priority: {progress:.1}% complete \
                 ({loaded_size} of {total_size} bytes)",
                loaded_size = self.loaded_size,
                total_size = self.total_size,
            )
            .into(),
        );
        #[cfg(not(target_arch = "wasm32"))]
        let _ = total_count;

        Ok(self.loading_progress)
    }

    /// Get loading progress percentage
    #[wasm_bindgen(getter)]
    pub fn loading_progress(&self) -> f64 {
        self.loading_progress
    }

    /// Check if a specific component type is loaded.
    ///
    /// `false` covers two different situations this method cannot tell
    /// apart from its return value alone: `component_type` genuinely
    /// hasn't finished loading yet, or this split never produces that
    /// `ChunkType` in the first place. Since
    /// `ModelSplitter::analyze_model_structure`'s honest rewrite, a
    /// SafeTensors split only ever reports `ChunkType::Config` (the
    /// header) and `ChunkType::Custom` (every tensor) — the other seven
    /// variants (`Embeddings`, `Attention`, `Vocabulary`, ...) are never
    /// emitted by this module's own splitter and so can never return
    /// `true` here for that reason alone, not because loading failed.
    pub fn is_component_loaded(&self, component_type: ChunkType) -> bool {
        self.loaded_components.get(&component_type).copied().unwrap_or(false)
    }

    /// Get summary of loaded components.
    ///
    /// `total_types` used to be a hardcoded `9` — the number of
    /// `ChunkType` enum variants — regardless of how many of them this
    /// split actually produced. Even under the old (fabricated) layout
    /// only 5 of the 9 variants were ever emitted, so a "fully loaded"
    /// split could never show anything but a confusing "5/9 (100.0%
    /// complete)". `ModelSplitter::analyze_model_structure`'s honest
    /// rewrite makes this worse if left as-is (a SafeTensors split now
    /// uses only `Config` + `Custom`), so the denominator is now the real
    /// count of distinct `ChunkType`s present in *this* split.
    pub fn get_loaded_summary(&self) -> String {
        let loaded_count = self.loaded_components.values().filter(|&&loaded| loaded).count();
        let total_types: std::collections::BTreeSet<ChunkType> =
            self.splitter.chunks.iter().map(|chunk| chunk.chunk_type).collect();

        format!(
            "Loaded components: {}/{} ({:.1}% complete)",
            loaded_count,
            total_types.len(),
            self.loading_progress
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

    // -----------------------------------------------------------------
    // Real DEFLATE compression (replacing the fake 30%-truncation stub).
    // -----------------------------------------------------------------

    #[test]
    fn test_compress_decompress_bytes_round_trip() {
        let mut state = 4242u32;
        let original: Vec<u8> = (0..3000)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                (state >> 16) as u8
            })
            .collect();

        let compressed = compress_bytes(&original).expect("compression should succeed");
        // Regression guard: the old fake `compress_data` produced exactly
        // `(len as f64 * 0.7) as usize` bytes by construction, silently
        // dropping the rest. Real DEFLATE output length is data-dependent
        // and effectively never lands on that exact naive formula.
        let naive_old_len = (original.len() as f64 * 0.7) as usize;
        assert_ne!(compressed.len(), naive_old_len);

        let decompressed = decompress_bytes(&compressed).expect("decompression should succeed");
        assert_eq!(
            decompressed, original,
            "compress/decompress must round-trip exactly"
        );
    }

    #[test]
    fn test_compress_bytes_shrinks_repetitive_data_for_real() {
        let original = std::vec![0xABu8; 16384];
        let compressed = compress_bytes(&original).expect("compression should succeed");
        assert!(
            compressed.len() < original.len() / 4,
            "highly repetitive data must compress substantially: {} of {} bytes",
            compressed.len(),
            original.len()
        );
        let decompressed = decompress_bytes(&compressed).expect("decompression should succeed");
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_decompress_bytes_rejects_garbage() {
        let garbage = std::vec![0xFFu8; 32];
        assert!(
            decompress_bytes(&garbage).is_err(),
            "non-DEFLATE data must not silently decode"
        );
    }

    // -----------------------------------------------------------------
    // Chunk checksum verification.
    // -----------------------------------------------------------------

    fn make_chunk(id: &str, raw: &[u8], compress: bool) -> ModelChunk {
        let checksum = calculate_checksum(raw);
        let data = if compress {
            compress_bytes(raw).expect("compression should succeed")
        } else {
            raw.to_vec()
        };
        ModelChunk {
            id: id.to_string(),
            chunk_type: ChunkType::Custom,
            size_bytes: data.len(),
            dependencies: Vec::new(),
            priority: ChunkPriority::Medium,
            data,
            compressed: compress,
            checksum,
        }
    }

    #[test]
    fn test_resolve_chunk_data_round_trips_compressed_chunk() {
        let raw = std::vec![7u8, 8, 9, 10, 11, 12, 13, 14];
        let chunk = make_chunk("c1", &raw, true);
        let resolved = resolve_chunk_data(std::slice::from_ref(&chunk), "c1")
            .expect("checksum must match")
            .expect("chunk must be found");
        assert_eq!(resolved, raw);
    }

    #[test]
    fn test_resolve_chunk_data_missing_id_returns_none() {
        let chunk = make_chunk("c1", &[1, 2, 3], false);
        let resolved = resolve_chunk_data(std::slice::from_ref(&chunk), "does-not-exist").unwrap();
        assert!(resolved.is_none());
    }

    #[test]
    fn test_resolve_chunk_data_detects_checksum_mismatch() {
        // Regression guard: with the old fake compressor, a compressed
        // chunk's stored `data` never matched what its checksum (computed
        // over the real pre-compression bytes) expected — but nothing ever
        // checked that. Corrupt a chunk's stored bytes directly here and
        // confirm the mismatch is now caught rather than silently returned.
        let raw = std::vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut chunk = make_chunk("c1", &raw, false);
        chunk.data[0] ^= 0xFF; // corrupt one byte after checksumming

        let err = resolve_chunk_data(std::slice::from_ref(&chunk), "c1")
            .expect_err("corrupted chunk data must fail its checksum check");
        assert!(err.contains("integrity check"));
    }

    // -----------------------------------------------------------------
    // End-to-end: split -> resolve every chunk -> exact reassembly.
    // -----------------------------------------------------------------

    #[test]
    fn test_split_model_reassembles_byte_exact() {
        // The headline regression test for this module: the old
        // `compress_data` silently discarded the last 30% of every chunk
        // over 1KB, so concatenating resolved chunk data could never equal
        // the original model bytes. Build a real (non-trivial, partially
        // compressible) byte buffer, split it, resolve every chunk through
        // the real decompress+checksum path, and verify the concatenation
        // is byte-for-byte identical to the input.
        let mut state = 7u32;
        // `set_max_chunk_size_mb` clamps to a 1.0 MB floor (see its own
        // doc comment), so the buffer must comfortably exceed 1 MB - not
        // merely whatever chunk size is requested below - to genuinely
        // force multiple chunks under the honest (position-tiled)
        // fallback splitter.
        let total_size = 3_000_000usize;
        let model_data: Vec<u8> = (0..total_size)
            .map(|i| {
                if i % 5 == 0 {
                    // Some genuinely repetitive stretches so compression
                    // actually engages for at least some chunks.
                    0x11
                } else {
                    state = state.wrapping_mul(1103515245).wrapping_add(12345);
                    (state >> 16) as u8
                }
            })
            .collect();

        let mut config = ChunkConfig::new();
        config.set_max_chunk_size_mb(1.0); // the real (clamped) floor
        let mut splitter = ModelSplitter::new(config);
        splitter
            .split_model_inner(&model_data, "test-model", "1.0.0")
            .expect("splitting real data must succeed");

        assert!(
            splitter.chunks.len() > 1,
            "expected more than one chunk for this input size"
        );
        assert!(
            splitter.chunks.iter().any(|c| c.compressed),
            "at least one chunk should have engaged real compression"
        );

        let mut reassembled = Vec::with_capacity(model_data.len());
        for chunk in &splitter.chunks {
            let resolved = resolve_chunk_data(&splitter.chunks, &chunk.id)
                .expect("every chunk must pass its checksum check")
                .expect("chunk must be found by its own id");
            reassembled.extend_from_slice(&resolved);
        }

        assert_eq!(
            reassembled.len(),
            model_data.len(),
            "reassembled length must match the original exactly"
        );
        assert_eq!(
            reassembled, model_data,
            "reassembled bytes must be identical to the original"
        );
    }

    #[test]
    fn test_resolved_chunk_length_matches_its_own_component_not_a_70_percent_truncation() {
        // The old bug: `compress_data` unconditionally truncated to 70% of
        // whatever it was given, and nothing ever decompressed it back.
        // Take a single component-sized buffer through compress+decompress
        // directly and confirm the recovered length is the *real* original
        // length, not `(len as f64 * 0.7) as usize`.
        let raw = std::vec![0x5Au8; 4096];
        let compressed = compress_bytes(&raw).expect("compression should succeed");
        let decompressed = decompress_bytes(&compressed).expect("decompression should succeed");
        assert_eq!(decompressed.len(), raw.len());
        assert_ne!(decompressed.len(), (raw.len() as f64 * 0.7) as usize);
    }

    // -----------------------------------------------------------------
    // `ModelLoadingSession::load_by_priority`: real progress, no invented
    // delay.
    // -----------------------------------------------------------------

    fn split_for_test(model_data: &[u8]) -> ModelSplitter {
        let mut config = ChunkConfig::new();
        // Clamped to a 1.0 MB floor regardless of the value requested
        // here (see `set_max_chunk_size_mb`'s doc comment) - callers that
        // need multiple chunks out of the honest (position-tiled)
        // fallback splitter must pass a `model_data` comfortably larger
        // than 1 MB.
        config.set_max_chunk_size_mb(1.0);
        let mut splitter = ModelSplitter::new(config);
        splitter
            .split_model_inner(model_data, "test-model", "1.0.0")
            .expect("splitting real data must succeed");
        splitter
    }

    #[test]
    fn test_load_by_priority_updates_loaded_components_and_size_for_real() {
        // Regression test for the old `simulate_chunk_loading`: it ran a
        // `setTimeout` sized at "100ms per MB" and drove `loading_progress`
        // to 100% while `loaded_components`/`loaded_size` stayed at their
        // initial (empty/zero) values forever. Here, after loading, both
        // must reflect what was actually materialized.
        let mut state = 99u32;
        // See `split_for_test`: must comfortably exceed the 1.0 MB
        // clamped chunk-size floor to produce multiple chunks.
        let model_data: Vec<u8> = (0..3_000_000)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                (state >> 16) as u8
            })
            .collect();
        let splitter = split_for_test(&model_data);
        assert!(splitter.chunks.len() > 1, "expected multiple chunks");
        // `total_size_bytes()` (and `loaded_size`, which accumulates real
        // *decompressed* bytes) are both measured in original/logical
        // bytes — not `chunk.size_bytes`, which is each chunk's on-wire
        // *stored* (possibly DEFLATE-compressed) size and so does not sum
        // back to the original length.
        let expected_total_size = splitter.total_size_bytes();
        assert_eq!(expected_total_size, model_data.len());
        let expected_types: std::collections::BTreeSet<ChunkType> =
            splitter.chunks.iter().map(|c| c.chunk_type).collect();
        let loading_order = splitter.loading_order.clone();
        let chunks = splitter.chunks.clone();

        let mut loaded_components = BTreeMap::new();
        let mut loaded_size = 0usize;
        let progress = ModelLoadingSession::load_by_priority_inner(
            &loading_order,
            &chunks,
            expected_total_size,
            &mut loaded_components,
            &mut loaded_size,
        )
        .expect("loading real, checksum-verified chunks must succeed");

        assert_eq!(
            loaded_size, expected_total_size,
            "loaded_size must equal the sum of every chunk's real size, not stay at 0"
        );
        assert!(
            (progress - 100.0).abs() < 1e-9,
            "progress must reflect real completion, got {progress}"
        );
        for chunk_type in expected_types {
            assert_eq!(
                loaded_components.get(&chunk_type),
                Some(&true),
                "{chunk_type:?} must be marked loaded, not left untouched"
            );
        }
    }

    #[test]
    fn test_load_by_priority_reports_partial_progress_and_rejects_corruption() {
        let raw_a = std::vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let raw_b = std::vec![9u8, 10, 11, 12];
        let mut chunk_a = make_chunk("a", &raw_a, false);
        chunk_a.chunk_type = ChunkType::Embeddings;
        let mut chunk_b = make_chunk("b", &raw_b, false);
        chunk_b.chunk_type = ChunkType::Vocabulary;
        chunk_b.data[0] ^= 0xFF; // corrupt chunk "b" after checksumming
        let total_size = raw_a.len() + raw_b.len();

        let mut loaded_components = BTreeMap::new();
        let mut loaded_size = 0usize;
        let err = ModelLoadingSession::load_by_priority_inner(
            &std::vec!["a".to_string(), "b".to_string()],
            &std::vec![chunk_a, chunk_b],
            total_size,
            &mut loaded_components,
            &mut loaded_size,
        )
        .expect_err("a corrupted chunk must fail its checksum check, not report success");
        assert!(err.contains("integrity check"), "unexpected error: {err}");

        // The first (uncorrupted) chunk must still have been recorded
        // before the second one failed — no invented all-or-nothing
        // progress jump.
        assert_eq!(loaded_size, raw_a.len());
        assert_eq!(loaded_components.get(&ChunkType::Embeddings), Some(&true));
        assert_eq!(loaded_components.get(&ChunkType::Vocabulary), None);
    }

    #[test]
    fn test_load_by_priority_wasm_bindgen_wrapper_runs_synchronously() {
        // Exercises the real `pub async fn load_by_priority` (not just its
        // `_inner` core) end to end on a native target: it must complete
        // without ever needing a real network delay or a JS/wasm32
        // environment on its success path.
        let model_data = std::vec![0x42u8; 150_000];
        let splitter = split_for_test(&model_data);
        // See the comment in the test above: this must be the logical
        // (pre-compression) total, not a sum of stored chunk sizes — the
        // repeated `0x42` bytes here compress extremely well, so the two
        // would differ by orders of magnitude.
        let expected_total_size = splitter.total_size_bytes();
        let mut session = ModelLoadingSession::new(splitter);

        let progress = futures::executor::block_on(session.load_by_priority())
            .expect("loading must succeed natively with no JS/network dependency");

        assert!((progress - 100.0).abs() < 1e-9);
        assert_eq!(session.loaded_size, expected_total_size);
        assert!(session.loaded_components.values().all(|&loaded| loaded));
    }

    // -----------------------------------------------------------------
    // `analyze_model_structure`: real component boundaries for
    // recognized formats, honest position-only naming for everything
    // else — replacing the old invented "config 1% / vocabulary 5% /
    // embeddings 15% / alternating attention-feedforward layers" layout
    // that never read a byte of `model_data`.
    // -----------------------------------------------------------------

    /// Hand-build a minimal, valid SafeTensors buffer (independent of any
    /// writer implementation) — see
    /// `core::model::formats::safetensors::tests::build_safetensors` for
    /// the format this mirrors.
    fn build_test_safetensors(entries: &[(&str, &str, &[usize], &[u8])]) -> Vec<u8> {
        let mut body = Vec::new();
        let mut fields = Vec::new();
        let mut offset = 0usize;
        for (name, dtype, shape, bytes) in entries {
            let start = offset;
            let end = offset + bytes.len();
            offset = end;
            body.extend_from_slice(bytes);
            let shape_str =
                shape.iter().map(std::string::ToString::to_string).collect::<Vec<_>>().join(",");
            fields.push(format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{shape_str}],\
                 \"data_offsets\":[{start},{end}]}}"
            ));
        }
        let header_json = format!("{{{}}}", fields.join(","));
        let mut out = Vec::new();
        out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
        out.extend_from_slice(header_json.as_bytes());
        out.extend_from_slice(&body);
        out
    }

    #[test]
    fn test_split_model_uses_real_tensor_names_for_recognized_safetensors() {
        let tensor_a = [1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat();
        let tensor_b = 3.0f32.to_le_bytes().to_vec();
        let data = build_test_safetensors(&[
            ("layer.attn.q_proj.weight", "F32", &[2], &tensor_a),
            ("layer.mlp.up_proj.weight", "F32", &[1], &tensor_b),
        ]);

        let mut splitter = ModelSplitter::new(ChunkConfig::new());
        splitter
            .split_model_inner(&data, "test-model", "1.0.0")
            .expect("a well-formed safetensors buffer must split successfully");

        // Header chunk + one chunk per real tensor - never the old fixed
        // "config"/"vocabulary"/"embeddings"/"layer_N" fabricated set.
        assert_eq!(splitter.chunks.len(), 3);
        let names: Vec<&str> = splitter.chunks.iter().map(|c| c.id.as_str()).collect();
        assert!(
            names.iter().any(|n| n.ends_with("_safetensors_header")),
            "expected a real header chunk, got {names:?}"
        );
        assert!(
            names.iter().any(|n| n.ends_with("_layer.attn.q_proj.weight")),
            "expected the tensor's real name, got {names:?}"
        );
        assert!(
            names.iter().any(|n| n.ends_with("_layer.mlp.up_proj.weight")),
            "expected the tensor's real name, got {names:?}"
        );
        assert!(
            names.iter().all(|n| !n.contains("vocabulary") && !n.contains("embeddings")),
            "must not contain fabricated semantic labels this buffer never declared: {names:?}"
        );

        let header_chunk =
            splitter.chunks.iter().find(|c| c.id.ends_with("_safetensors_header")).unwrap();
        assert_eq!(header_chunk.chunk_type, ChunkType::Config);
        let tensor_chunk = splitter
            .chunks
            .iter()
            .find(|c| c.id.ends_with("_layer.attn.q_proj.weight"))
            .unwrap();
        assert_eq!(tensor_chunk.chunk_type, ChunkType::Custom);

        // Since this buffer was built by hand with no padding, every
        // chunk's resolved bytes must reassemble it exactly - unlike the
        // real-world case where SafeTensors permits gaps, this is a
        // legitimate byte-exact check because the test controls every
        // byte of the input.
        let mut reassembled = Vec::with_capacity(data.len());
        for chunk in &splitter.chunks {
            let resolved = resolve_chunk_data(&splitter.chunks, &chunk.id)
                .expect("checksum must verify")
                .expect("chunk must be found by its own id");
            reassembled.extend_from_slice(&resolved);
        }
        assert_eq!(reassembled, data);
    }

    #[test]
    fn test_calculate_chunk_dependencies_safetensors_tensors_depend_on_real_header_chunk() {
        let data = build_test_safetensors(&[("w", "F32", &[1], &1.0f32.to_le_bytes())]);
        let mut splitter = ModelSplitter::new(ChunkConfig::new());
        splitter.split_model_inner(&data, "test-model", "1.0.0").expect("must split");

        let header_chunk =
            splitter.chunks.iter().find(|c| c.chunk_type == ChunkType::Config).unwrap();
        assert!(
            header_chunk.dependencies.is_empty(),
            "the header chunk itself must not depend on anything"
        );

        let tensor_chunk =
            splitter.chunks.iter().find(|c| c.chunk_type != ChunkType::Config).unwrap();
        assert_eq!(
            tensor_chunk.dependencies,
            std::vec![header_chunk.id.clone()],
            "a tensor chunk must depend on the real header chunk's own id, not a hardcoded \
             'chunk_000_config' that may not exist"
        );
    }

    #[test]
    fn test_uniform_fallback_names_chunks_by_position_only_and_declares_no_dependencies() {
        // Bytes that cannot be mistaken for a safetensors header (the
        // first 8 bytes decode to a header length far larger than the
        // buffer itself). Comfortably exceeds the 1.0 MB clamped
        // chunk-size floor (see `set_max_chunk_size_mb`'s doc comment) so
        // multiple chunks genuinely get produced.
        let data: Vec<u8> = (0..3_000_000u32).map(|i| (i % 251) as u8).collect();

        let mut config = ChunkConfig::new();
        config.set_max_chunk_size_mb(1.0); // the real (clamped) floor
        let mut splitter = ModelSplitter::new(config);
        splitter.split_model_inner(&data, "test-model", "1.0.0").expect("must split");

        assert!(splitter.chunks.len() > 1, "expected multiple chunks");
        for (i, chunk) in splitter.chunks.iter().enumerate() {
            assert!(
                chunk.id.ends_with(&format!("_chunk_{i}")),
                "chunk name must be position-only ('chunk_N'), got '{}'",
                chunk.id
            );
            assert_eq!(chunk.chunk_type, ChunkType::Custom);
            assert!(
                chunk.dependencies.is_empty(),
                "an unrecognized format has no real dependency evidence to report, got {:?}",
                chunk.dependencies
            );
        }
    }

    #[test]
    fn test_get_loaded_summary_denominator_reflects_real_distinct_chunk_types() {
        // A previous version hardcoded `total_types = 9` (the number of
        // `ChunkType` enum variants) regardless of how many this split
        // actually used, so a fully-loaded split with only two real
        // types in play (as any SafeTensors split now has: `Config` +
        // `Custom`) would report a self-contradictory "2/9 (100.0%
        // complete)".
        let data = build_test_safetensors(&[("w", "F32", &[1], &1.0f32.to_le_bytes())]);
        let mut splitter = ModelSplitter::new(ChunkConfig::new());
        splitter.split_model_inner(&data, "test-model", "1.0.0").expect("must split");
        let distinct_types: std::collections::BTreeSet<ChunkType> =
            splitter.chunks.iter().map(|c| c.chunk_type).collect();
        assert_eq!(
            distinct_types.len(),
            2,
            "expected exactly Config + Custom in this split"
        );

        let mut session = ModelLoadingSession::new(splitter);
        futures::executor::block_on(session.load_by_priority())
            .expect("loading must succeed natively");

        let summary = session.get_loaded_summary();
        assert_eq!(summary, "Loaded components: 2/2 (100.0% complete)");
    }
}
