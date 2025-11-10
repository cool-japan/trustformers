//! Progressive Module Loading for Cold Start Optimization
//!
//! This module implements intelligent progressive loading strategies to minimize
//! time-to-interactive (TTI) and optimize cold start performance for web deployment.
//!
//! Key features:
//! - Priority-based module loading
//! - Lazy initialization with deferred execution
//! - Streaming compilation support
//! - Chunked model loading with prefetching
//! - Cache-aware loading strategies

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;

/// Module loading priority levels
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum LoadPriority {
    /// Critical modules needed for first render (highest priority)
    Critical = 0,
    /// High priority modules for core functionality
    High = 1,
    /// Medium priority for common features
    Medium = 2,
    /// Low priority for optional features
    Low = 3,
    /// Deferred loading for rarely used features
    Deferred = 4,
}

/// Module loading state
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadingState {
    /// Not yet loaded
    Pending,
    /// Currently loading
    Loading,
    /// Successfully loaded
    Loaded,
    /// Failed to load
    Failed,
    /// Loading cached version
    LoadingCached,
}

/// Module metadata for progressive loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleMetadata {
    pub name: String,
    pub priority: LoadPriority,
    pub size_bytes: usize,
    pub dependencies: Vec<String>,
    pub optional: bool,
    pub cache_key: String,
    pub estimated_load_time_ms: f32,
}

/// Progressive loader configuration
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct ProgressiveLoaderConfig {
    enable_streaming: bool,
    enable_prefetch: bool,
    enable_cache: bool,
    max_concurrent_loads: usize,
    prefetch_threshold_ms: f32,
    #[allow(dead_code)]
    cache_version: String,
}

impl Default for ProgressiveLoaderConfig {
    fn default() -> Self {
        Self {
            enable_streaming: true,
            enable_prefetch: true,
            enable_cache: true,
            max_concurrent_loads: 4,
            prefetch_threshold_ms: 100.0,
            cache_version: "v1".to_string(),
        }
    }
}

#[wasm_bindgen]
impl ProgressiveLoaderConfig {
    /// Create a new progressive loader configuration
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create configuration optimized for fast initial load
    pub fn fast_initial() -> Self {
        Self {
            enable_streaming: true,
            enable_prefetch: true,
            enable_cache: true,
            max_concurrent_loads: 6,
            prefetch_threshold_ms: 50.0,
            cache_version: "v1".to_string(),
        }
    }

    /// Create configuration optimized for low bandwidth
    pub fn low_bandwidth() -> Self {
        Self {
            enable_streaming: true,
            enable_prefetch: false,
            enable_cache: true,
            max_concurrent_loads: 2,
            prefetch_threshold_ms: 200.0,
            cache_version: "v1".to_string(),
        }
    }

    /// Enable or disable streaming compilation
    pub fn set_streaming(&mut self, enable: bool) {
        self.enable_streaming = enable;
    }

    /// Enable or disable prefetching
    pub fn set_prefetch(&mut self, enable: bool) {
        self.enable_prefetch = enable;
    }

    /// Set maximum concurrent loads
    pub fn set_max_concurrent(&mut self, max: usize) {
        self.max_concurrent_loads = max;
    }
}

/// Progressive module loader
pub struct ProgressiveLoader {
    config: ProgressiveLoaderConfig,
    modules: HashMap<String, ModuleMetadata>,
    #[allow(dead_code)]
    loading_queue: VecDeque<String>,
    #[allow(dead_code)]
    loaded_modules: HashMap<String, LoadingState>,
    loading_stats: LoadingStats,
}

impl ProgressiveLoader {
    /// Create a new progressive loader
    pub fn new(config: ProgressiveLoaderConfig) -> Self {
        Self {
            config,
            modules: HashMap::new(),
            loading_queue: VecDeque::new(),
            loaded_modules: HashMap::new(),
            loading_stats: LoadingStats::default(),
        }
    }

    /// Register a module for progressive loading
    pub fn register_module(&mut self, metadata: ModuleMetadata) {
        self.modules.insert(metadata.name.clone(), metadata);
    }

    /// Plan loading order based on priorities and dependencies
    pub fn plan_loading_order(&mut self) -> Vec<String> {
        let mut load_order = Vec::new();
        let mut loaded = std::collections::HashSet::new();

        // Sort modules by priority
        let mut sorted_modules: Vec<_> = self.modules.values().collect();
        sorted_modules.sort_by_key(|m| m.priority);

        // Build load order respecting dependencies
        for module in sorted_modules {
            self.add_to_load_order(&module.name, &mut load_order, &mut loaded);
        }

        load_order
    }

    fn add_to_load_order(
        &self,
        module_name: &str,
        load_order: &mut Vec<String>,
        loaded: &mut std::collections::HashSet<String>,
    ) {
        if loaded.contains(module_name) {
            return;
        }

        if let Some(module) = self.modules.get(module_name) {
            // Load dependencies first
            for dep in &module.dependencies {
                self.add_to_load_order(dep, load_order, loaded);
            }

            // Add this module
            load_order.push(module_name.to_string());
            loaded.insert(module_name.to_string());
        }
    }

    /// Estimate total loading time
    pub fn estimate_load_time(&self, modules: &[String]) -> f32 {
        let mut total_time = 0.0;
        let mut concurrent_groups = Vec::new();
        let mut current_group = Vec::new();

        for module_name in modules {
            if let Some(module) = self.modules.get(module_name) {
                current_group.push(module.estimated_load_time_ms);

                if current_group.len() >= self.config.max_concurrent_loads {
                    concurrent_groups.push(current_group);
                    current_group = Vec::new();
                }
            }
        }

        if !current_group.is_empty() {
            concurrent_groups.push(current_group);
        }

        // Sum the maximum time in each concurrent group
        for group in concurrent_groups {
            total_time += group.iter().copied().fold(0.0f32, f32::max);
        }

        total_time
    }

    /// Get modules that should be prefetched
    pub fn get_prefetch_candidates(&self, loaded: &[String]) -> Vec<String> {
        if !self.config.enable_prefetch {
            return Vec::new();
        }

        let mut candidates = Vec::new();
        let loaded_set: std::collections::HashSet<_> = loaded.iter().cloned().collect();

        for (name, module) in &self.modules {
            if loaded_set.contains(name) {
                continue;
            }

            // Check if dependencies are loaded
            let deps_loaded = module.dependencies.iter().all(|dep| loaded_set.contains(dep));

            if deps_loaded && module.estimated_load_time_ms <= self.config.prefetch_threshold_ms {
                candidates.push(name.clone());
            }
        }

        // Sort by priority
        candidates.sort_by_key(|name| {
            self.modules.get(name).map(|m| m.priority).unwrap_or(LoadPriority::Deferred)
        });

        candidates
    }

    /// Generate loading manifest for the loader
    pub fn generate_manifest(&mut self) -> LoadingManifest {
        let load_order = self.plan_loading_order();
        let critical_modules: Vec<_> = self
            .modules
            .values()
            .filter(|m| m.priority == LoadPriority::Critical)
            .map(|m| m.name.clone())
            .collect();

        let total_size: usize = self.modules.values().map(|m| m.size_bytes).sum();
        let estimated_time = self.estimate_load_time(&load_order);

        LoadingManifest {
            load_order,
            critical_modules,
            total_modules: self.modules.len(),
            total_size_bytes: total_size,
            estimated_load_time_ms: estimated_time,
            cache_enabled: self.config.enable_cache,
            streaming_enabled: self.config.enable_streaming,
        }
    }

    /// Get loading statistics
    pub fn get_stats(&self) -> &LoadingStats {
        &self.loading_stats
    }

    /// Update loading statistics
    pub fn update_stats(&mut self, module_name: &str, load_time_ms: f32, from_cache: bool) {
        self.loading_stats.total_modules_loaded += 1;
        self.loading_stats.total_load_time_ms += load_time_ms;

        if from_cache {
            self.loading_stats.cache_hits += 1;
        }

        if let Some(module) = self.modules.get(module_name) {
            self.loading_stats.total_bytes_loaded += module.size_bytes;
        }

        // Update average
        self.loading_stats.average_load_time_ms =
            self.loading_stats.total_load_time_ms / self.loading_stats.total_modules_loaded as f32;
    }
}

/// Loading manifest for client-side loader
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadingManifest {
    load_order: Vec<String>,
    critical_modules: Vec<String>,
    total_modules: usize,
    total_size_bytes: usize,
    estimated_load_time_ms: f32,
    cache_enabled: bool,
    streaming_enabled: bool,
}

#[wasm_bindgen]
impl LoadingManifest {
    /// Get the loading order
    pub fn load_order(&self) -> Vec<String> {
        self.load_order.clone()
    }

    /// Get critical modules
    pub fn critical_modules(&self) -> Vec<String> {
        self.critical_modules.clone()
    }

    /// Get total number of modules
    pub fn total_modules(&self) -> usize {
        self.total_modules
    }

    /// Get total size in bytes
    pub fn total_size_bytes(&self) -> usize {
        self.total_size_bytes
    }

    /// Get estimated load time in milliseconds
    pub fn estimated_load_time_ms(&self) -> f32 {
        self.estimated_load_time_ms
    }

    /// Check if caching is enabled
    pub fn is_cache_enabled(&self) -> bool {
        self.cache_enabled
    }

    /// Check if streaming is enabled
    pub fn is_streaming_enabled(&self) -> bool {
        self.streaming_enabled
    }
}

/// Loading statistics
#[wasm_bindgen]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoadingStats {
    total_modules_loaded: usize,
    total_bytes_loaded: usize,
    total_load_time_ms: f32,
    average_load_time_ms: f32,
    cache_hits: usize,
}

#[wasm_bindgen]
impl LoadingStats {
    /// Get total modules loaded
    pub fn total_modules_loaded(&self) -> usize {
        self.total_modules_loaded
    }

    /// Get total bytes loaded
    pub fn total_bytes_loaded(&self) -> usize {
        self.total_bytes_loaded
    }

    /// Get total load time in milliseconds
    pub fn total_load_time_ms(&self) -> f32 {
        self.total_load_time_ms
    }

    /// Get average load time per module
    pub fn average_load_time_ms(&self) -> f32 {
        self.average_load_time_ms
    }

    /// Get cache hit count
    pub fn cache_hits(&self) -> usize {
        self.cache_hits
    }

    /// Get cache hit rate
    pub fn cache_hit_rate(&self) -> f32 {
        if self.total_modules_loaded == 0 {
            0.0
        } else {
            self.cache_hits as f32 / self.total_modules_loaded as f32
        }
    }
}

/// Chunk loader for splitting large models
pub struct ChunkLoader {
    chunk_size: usize,
    chunks: Vec<ChunkMetadata>,
    loaded_chunks: std::collections::HashSet<usize>,
}

#[derive(Debug, Clone)]
pub struct ChunkMetadata {
    pub chunk_id: usize,
    pub offset: usize,
    pub size: usize,
    pub checksum: String,
}

impl ChunkLoader {
    /// Create a new chunk loader
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunk_size,
            chunks: Vec::new(),
            loaded_chunks: std::collections::HashSet::new(),
        }
    }

    /// Split data into chunks
    pub fn split_into_chunks(&mut self, total_size: usize) -> Vec<ChunkMetadata> {
        let num_chunks = (total_size + self.chunk_size - 1) / self.chunk_size;
        let mut chunks = Vec::new();

        for i in 0..num_chunks {
            let offset = i * self.chunk_size;
            let size = (self.chunk_size).min(total_size - offset);

            chunks.push(ChunkMetadata {
                chunk_id: i,
                offset,
                size,
                checksum: format!("chunk_{}", i), // Placeholder
            });
        }

        self.chunks = chunks.clone();
        chunks
    }

    /// Mark chunk as loaded
    pub fn mark_loaded(&mut self, chunk_id: usize) {
        self.loaded_chunks.insert(chunk_id);
    }

    /// Check if all chunks are loaded
    pub fn is_complete(&self) -> bool {
        self.loaded_chunks.len() == self.chunks.len()
    }

    /// Get loading progress (0.0 to 1.0)
    pub fn progress(&self) -> f32 {
        if self.chunks.is_empty() {
            0.0
        } else {
            self.loaded_chunks.len() as f32 / self.chunks.len() as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progressive_loader_config() {
        let config = ProgressiveLoaderConfig::default();
        assert!(config.enable_streaming);
        assert!(config.enable_prefetch);
        assert_eq!(config.max_concurrent_loads, 4);
    }

    #[test]
    fn test_fast_initial_config() {
        let config = ProgressiveLoaderConfig::fast_initial();
        assert_eq!(config.max_concurrent_loads, 6);
        assert_eq!(config.prefetch_threshold_ms, 50.0);
    }

    #[test]
    fn test_low_bandwidth_config() {
        let config = ProgressiveLoaderConfig::low_bandwidth();
        assert_eq!(config.max_concurrent_loads, 2);
        assert!(!config.enable_prefetch);
    }

    #[test]
    fn test_module_registration() {
        let config = ProgressiveLoaderConfig::default();
        let mut loader = ProgressiveLoader::new(config);

        let metadata = ModuleMetadata {
            name: "core".to_string(),
            priority: LoadPriority::Critical,
            size_bytes: 1024,
            dependencies: vec![],
            optional: false,
            cache_key: "core_v1".to_string(),
            estimated_load_time_ms: 10.0,
        };

        loader.register_module(metadata);
        assert_eq!(loader.modules.len(), 1);
    }

    #[test]
    fn test_loading_order_with_dependencies() {
        let config = ProgressiveLoaderConfig::default();
        let mut loader = ProgressiveLoader::new(config);

        loader.register_module(ModuleMetadata {
            name: "base".to_string(),
            priority: LoadPriority::Critical,
            size_bytes: 1024,
            dependencies: vec![],
            optional: false,
            cache_key: "base_v1".to_string(),
            estimated_load_time_ms: 10.0,
        });

        loader.register_module(ModuleMetadata {
            name: "feature".to_string(),
            priority: LoadPriority::High,
            size_bytes: 2048,
            dependencies: vec!["base".to_string()],
            optional: false,
            cache_key: "feature_v1".to_string(),
            estimated_load_time_ms: 20.0,
        });

        let load_order = loader.plan_loading_order();
        assert_eq!(load_order.len(), 2);
        assert_eq!(load_order[0], "base");
        assert_eq!(load_order[1], "feature");
    }

    #[test]
    fn test_load_time_estimation() {
        let config = ProgressiveLoaderConfig::default();
        let mut loader = ProgressiveLoader::new(config);

        loader.register_module(ModuleMetadata {
            name: "mod1".to_string(),
            priority: LoadPriority::Critical,
            size_bytes: 1024,
            dependencies: vec![],
            optional: false,
            cache_key: "mod1_v1".to_string(),
            estimated_load_time_ms: 10.0,
        });

        loader.register_module(ModuleMetadata {
            name: "mod2".to_string(),
            priority: LoadPriority::High,
            size_bytes: 2048,
            dependencies: vec![],
            optional: false,
            cache_key: "mod2_v1".to_string(),
            estimated_load_time_ms: 15.0,
        });

        let modules = vec!["mod1".to_string(), "mod2".to_string()];
        let estimated_time = loader.estimate_load_time(&modules);
        assert!(estimated_time > 0.0);
    }

    #[test]
    fn test_prefetch_candidates() {
        let config = ProgressiveLoaderConfig::default();
        let mut loader = ProgressiveLoader::new(config);

        loader.register_module(ModuleMetadata {
            name: "loaded".to_string(),
            priority: LoadPriority::Critical,
            size_bytes: 1024,
            dependencies: vec![],
            optional: false,
            cache_key: "loaded_v1".to_string(),
            estimated_load_time_ms: 10.0,
        });

        loader.register_module(ModuleMetadata {
            name: "prefetch".to_string(),
            priority: LoadPriority::High,
            size_bytes: 2048,
            dependencies: vec!["loaded".to_string()],
            optional: false,
            cache_key: "prefetch_v1".to_string(),
            estimated_load_time_ms: 50.0,
        });

        let loaded = vec!["loaded".to_string()];
        let candidates = loader.get_prefetch_candidates(&loaded);
        assert!(!candidates.is_empty());
    }

    #[test]
    fn test_loading_manifest_generation() {
        let config = ProgressiveLoaderConfig::default();
        let mut loader = ProgressiveLoader::new(config);

        loader.register_module(ModuleMetadata {
            name: "core".to_string(),
            priority: LoadPriority::Critical,
            size_bytes: 1024,
            dependencies: vec![],
            optional: false,
            cache_key: "core_v1".to_string(),
            estimated_load_time_ms: 10.0,
        });

        let manifest = loader.generate_manifest();
        assert_eq!(manifest.total_modules(), 1);
        assert!(manifest.is_cache_enabled());
        assert!(manifest.is_streaming_enabled());
    }

    #[test]
    fn test_loading_stats() {
        let config = ProgressiveLoaderConfig::default();
        let mut loader = ProgressiveLoader::new(config);

        loader.register_module(ModuleMetadata {
            name: "test".to_string(),
            priority: LoadPriority::Critical,
            size_bytes: 1024,
            dependencies: vec![],
            optional: false,
            cache_key: "test_v1".to_string(),
            estimated_load_time_ms: 10.0,
        });

        loader.update_stats("test", 10.0, true);

        let stats = loader.get_stats();
        assert_eq!(stats.total_modules_loaded(), 1);
        assert_eq!(stats.cache_hits(), 1);
        assert_eq!(stats.cache_hit_rate(), 1.0);
    }

    #[test]
    fn test_chunk_loader() {
        let mut chunk_loader = ChunkLoader::new(1024);
        let chunks = chunk_loader.split_into_chunks(5000);

        assert_eq!(chunks.len(), 5);
        assert_eq!(chunks[0].size, 1024);
        assert_eq!(chunks[4].size, 5000 - 4 * 1024);
    }

    #[test]
    fn test_chunk_progress() {
        let mut chunk_loader = ChunkLoader::new(1024);
        chunk_loader.split_into_chunks(3000);

        assert_eq!(chunk_loader.progress(), 0.0);

        chunk_loader.mark_loaded(0);
        assert!(chunk_loader.progress() > 0.0);
        assert!(!chunk_loader.is_complete());

        chunk_loader.mark_loaded(1);
        chunk_loader.mark_loaded(2);
        assert_eq!(chunk_loader.progress(), 1.0);
        assert!(chunk_loader.is_complete());
    }

    #[test]
    fn test_load_priority_ordering() {
        assert!(LoadPriority::Critical < LoadPriority::High);
        assert!(LoadPriority::High < LoadPriority::Medium);
        assert!(LoadPriority::Medium < LoadPriority::Low);
        assert!(LoadPriority::Low < LoadPriority::Deferred);
    }
}
