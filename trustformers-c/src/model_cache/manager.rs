//! Model cache manager implementation

use anyhow::{anyhow, Result};
use crossbeam_utils::atomic::AtomicCell;
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::types::*;
use crate::error::TrustformersResult;

/// Main model cache manager with lock-free data structures
pub struct ModelCacheManager {
    config: ModelCacheConfig,
    /// Lock-free cache for concurrent model access
    cache: DashMap<String, ModelCacheEntry>,
    /// Lock-free access tracking for LRU
    access_tracker: DashMap<String, AtomicCell<Instant>>,
    /// Lock-free load queue using DashMap with priority ordering
    load_queue: DashMap<String, ModelPreloadRequest>,
    /// Access order for LRU eviction (protected by mutex)
    access_order: Arc<Mutex<Vec<String>>>,
    /// Atomic statistics for lock-free updates
    total_models: AtomicUsize,
    loaded_models: AtomicUsize,
    memory_usage_mb: AtomicUsize,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    evictions: AtomicU64,
    load_errors: AtomicU64,
    health_check_failures: AtomicU64,
    total_access_count: AtomicU64,
    start_time: AtomicCell<Instant>,
    cleanup_handle: Option<thread::JoinHandle<()>>,
    health_check_handle: Option<thread::JoinHandle<()>>,
}

impl ModelCacheManager {
    pub fn new() -> Self {
        Self {
            config: ModelCacheConfig::default(),
            cache: DashMap::new(),
            access_tracker: DashMap::new(),
            load_queue: DashMap::new(),
            access_order: Arc::new(Mutex::new(Vec::new())),
            total_models: AtomicUsize::new(0),
            loaded_models: AtomicUsize::new(0),
            memory_usage_mb: AtomicUsize::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            load_errors: AtomicU64::new(0),
            health_check_failures: AtomicU64::new(0),
            total_access_count: AtomicU64::new(0),
            start_time: AtomicCell::new(Instant::now()),
            cleanup_handle: None,
            health_check_handle: None,
        }
    }

    /// Configure the model cache
    pub fn configure(&mut self, config: ModelCacheConfig) -> TrustformersResult<()> {
        self.config = config.clone();

        // Start background tasks if enabled
        if config.cleanup_interval_sec > 0 {
            self.start_cleanup_task()?;
        }

        if config.enable_health_checks && config.health_check_interval_sec > 0 {
            self.start_health_check_task()?;
        }

        Ok(())
    }

    /// Load or get a model from cache
    pub fn get_or_load_model(
        &mut self,
        model_id: &str,
        model_path: &str,
        config: &str,
    ) -> TrustformersResult<usize> {
        // Check if model is already cached
        if let Some(mut entry) = self.cache.get_mut(model_id) {
            // Update access info
            entry.last_accessed = Instant::now();
            entry.access_count += 1;

            // Update LRU order
            if let Ok(mut access_order) = self.access_order.lock() {
                if let Some(pos) = access_order.iter().position(|id| id == model_id) {
                    access_order.remove(pos);
                }
                access_order.push(model_id.to_string());
            }

            self.cache_hits.fetch_add(1, Ordering::Relaxed);
            self.total_access_count.fetch_add(1, Ordering::Relaxed);

            return Ok(entry.model_handle);
        }

        // Model not in cache, need to load
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        self.total_access_count.fetch_add(1, Ordering::Relaxed);

        // Check if we need to evict models first
        self.maybe_evict_models()?;

        // Load the model
        let load_start = Instant::now();
        let model_handle = self.load_model_internal(model_path, config)?;
        let load_duration = load_start.elapsed();
        let load_duration_ms = load_duration.as_secs_f64() * 1000.0;

        // Estimate model size
        let size_mb = self.estimate_model_size(model_path)?;

        // Extract metadata
        let metadata = self.extract_model_metadata(config)?;

        // Extract version
        let version = self.extract_model_version(model_path, config)?;

        // Create cache entry
        let entry = ModelCacheEntry {
            model_id: model_id.to_string(),
            model_handle,
            model_path: model_path.to_string(),
            model_config: config.to_string(),
            version,
            size_mb,
            load_time: Instant::now(),
            load_duration_ms,
            last_accessed: Instant::now(),
            access_count: 1,
            warmup_completed: false,
            health_status: ModelHealthStatus::Unknown,
            metadata,
        };

        // Add to cache
        self.cache.insert(model_id.to_string(), entry);

        // Update access order
        if let Ok(mut access_order) = self.access_order.lock() {
            access_order.push(model_id.to_string());
        }

        // Update statistics
        self.total_models.fetch_add(1, Ordering::Relaxed);
        self.loaded_models.fetch_add(1, Ordering::Relaxed);
        self.memory_usage_mb.fetch_add(size_mb, Ordering::Relaxed);

        // Schedule warmup if enabled
        if self.config.enable_preloading {
            self.schedule_model_warmup(model_id)?;
        }

        Ok(model_handle)
    }

    /// Preload models based on priority
    pub fn preload_models(&mut self, requests: Vec<ModelPreloadRequest>) -> TrustformersResult<()> {
        let mut sorted_requests = requests;
        sorted_requests.sort_by(|a, b| b.priority.cmp(&a.priority));

        for request in sorted_requests {
            if !self.cache.contains_key(&request.model_id) {
                self.load_queue.insert(request.model_id.clone(), request.clone());

                // Try to load immediately if there's capacity
                if self.loaded_models.load(Ordering::Relaxed) < self.config.max_models {
                    match self.get_or_load_model(
                        &request.model_id,
                        &request.model_path,
                        &request.config,
                    ) {
                        Ok(_) => {
                            self.load_queue.remove(&request.model_id);
                        },
                        Err(_) => {
                            self.load_errors.fetch_add(1, Ordering::Relaxed);
                        },
                    }
                }
            }
        }

        Ok(())
    }

    /// Remove a model from cache
    pub fn remove_model(&mut self, model_id: &str) -> TrustformersResult<()> {
        if let Some((_, entry)) = self.cache.remove(model_id) {
            // Free model handle
            self.free_model_handle(entry.model_handle)?;

            // Update statistics
            self.loaded_models.fetch_sub(1, Ordering::Relaxed);
            self.memory_usage_mb.fetch_sub(entry.size_mb, Ordering::Relaxed);

            // Remove from access order
            if let Ok(mut access_order) = self.access_order.lock() {
                access_order.retain(|id| id != model_id);
            }

            Ok(())
        } else {
            Err(anyhow!("Model {} not found in cache", model_id).into())
        }
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> ModelCacheStats {
        let total_models = self.total_models.load(Ordering::Relaxed);
        let loaded_models = self.loaded_models.load(Ordering::Relaxed);
        let memory_usage_mb = self.memory_usage_mb.load(Ordering::Relaxed);
        let cache_hits = self.cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.cache_misses.load(Ordering::Relaxed);
        let total_access = self.total_access_count.load(Ordering::Relaxed);

        let cache_hit_ratio =
            if total_access > 0 { cache_hits as f64 / total_access as f64 } else { 0.0 };

        // Calculate average load time from all cached models
        let average_load_time_ms = if loaded_models > 0 {
            let total_load_time: f64 =
                self.cache.iter().map(|entry| entry.value().load_duration_ms).sum();
            total_load_time / loaded_models as f64
        } else {
            0.0
        };

        let uptime_sec = self.start_time.load().elapsed().as_secs();

        ModelCacheStats {
            total_models,
            loaded_models,
            memory_usage_mb,
            memory_limit_mb: self.config.max_memory_mb,
            cache_hits,
            cache_misses,
            cache_hit_ratio,
            evictions: self.evictions.load(Ordering::Relaxed),
            load_errors: self.load_errors.load(Ordering::Relaxed),
            health_check_failures: self.health_check_failures.load(Ordering::Relaxed),
            average_load_time_ms,
            total_access_count: total_access,
            uptime_sec,
        }
    }

    /// List all cached models
    pub fn list_models(&self) -> Vec<String> {
        self.cache.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Get model information
    pub fn get_model_info(&self, model_id: &str) -> Option<ModelCacheEntry> {
        self.cache.get(model_id).map(|entry| entry.clone())
    }

    // Internal helper methods

    fn maybe_evict_models(&mut self) -> TrustformersResult<()> {
        let current_models = self.loaded_models.load(Ordering::Relaxed);
        let current_memory = self.memory_usage_mb.load(Ordering::Relaxed);

        if current_models >= self.config.max_models || current_memory >= self.config.max_memory_mb {
            let evict_count = if current_models >= self.config.max_models {
                1
            } else {
                // Memory-based eviction - evict until under threshold
                let target_memory = self.config.max_memory_mb * 80 / 100; // 80% threshold
                (current_memory - target_memory) / 100 // Rough estimate
            };

            for _ in 0..evict_count {
                self.evict_one_model()?;
            }
        }

        Ok(())
    }

    fn evict_one_model(&mut self) -> TrustformersResult<()> {
        match self.config.eviction_policy {
            EvictionPolicy::LRU => self.evict_lru_model(),
            EvictionPolicy::LFU => self.evict_lfu_model(),
            EvictionPolicy::FIFO => self.evict_fifo_model(),
            _ => self.evict_lru_model(), // Default to LRU
        }
    }

    fn evict_lru_model(&mut self) -> TrustformersResult<()> {
        let model_id = {
            let access_order =
                self.access_order.lock().map_err(|_| anyhow!("Failed to lock access order"))?;
            access_order.first().cloned()
        }; // Lock is released here

        if let Some(model_id) = model_id {
            self.remove_model(&model_id)?;
            self.evictions.fetch_add(1, Ordering::Relaxed);
            Ok(())
        } else {
            Err(anyhow!("No models available for eviction").into())
        }
    }

    fn evict_lfu_model(&mut self) -> TrustformersResult<()> {
        let mut min_access_count = u64::MAX;
        let mut lfu_model_id = String::new();

        for entry in self.cache.iter() {
            if entry.access_count < min_access_count {
                min_access_count = entry.access_count;
                lfu_model_id = entry.model_id.clone();
            }
        }

        if !lfu_model_id.is_empty() {
            self.remove_model(&lfu_model_id)?;
            self.evictions.fetch_add(1, Ordering::Relaxed);
            Ok(())
        } else {
            Err(anyhow!("No models available for eviction").into())
        }
    }

    fn evict_fifo_model(&mut self) -> TrustformersResult<()> {
        let mut oldest_load_time = Instant::now();
        let mut fifo_model_id = String::new();

        for entry in self.cache.iter() {
            if entry.load_time < oldest_load_time {
                oldest_load_time = entry.load_time;
                fifo_model_id = entry.model_id.clone();
            }
        }

        if !fifo_model_id.is_empty() {
            self.remove_model(&fifo_model_id)?;
            self.evictions.fetch_add(1, Ordering::Relaxed);
            Ok(())
        } else {
            Err(anyhow!("No models available for eviction").into())
        }
    }

    fn load_model_internal(&self, model_path: &str, config: &str) -> TrustformersResult<usize> {
        // Simulate model loading - in real implementation this would:
        // 1. Load model using trustformers::AutoModel::from_pretrained()
        // 2. Return a handle to the loaded model

        // Generate a unique model handle (simplified)
        let handle = (model_path.len() + config.len()) * 1000 + std::process::id() as usize;

        // Simulate loading delay
        std::thread::sleep(Duration::from_millis(100));

        Ok(handle)
    }

    fn estimate_model_size(&self, model_path: &str) -> TrustformersResult<usize> {
        // Simplified size estimation - in real implementation this would
        // check actual file sizes, config parameters, etc.
        let base_size = match model_path {
            path if path.contains("7b") || path.contains("7B") => 7000,
            path if path.contains("13b") || path.contains("13B") => 13000,
            path if path.contains("30b") || path.contains("30B") => 30000,
            path if path.contains("65b") || path.contains("65B") => 65000,
            _ => 1000, // Default 1GB
        };

        Ok(base_size)
    }

    fn extract_model_metadata(&self, config: &str) -> TrustformersResult<ModelMetadata> {
        // Simplified metadata extraction - in real implementation this would
        // parse the actual model config JSON
        Ok(ModelMetadata {
            architecture: "transformer".to_string(),
            framework: "trustformers".to_string(),
            input_shape: vec![1, 512], // batch_size, sequence_length
            output_shape: vec![1, 512, 32000], // batch_size, sequence_length, vocab_size
            parameters_count: 7_000_000_000, // 7B parameters
            quantized: config.contains("quantized"),
            precision: if config.contains("fp16") { "fp16" } else { "fp32" }.to_string(),
            supported_backends: vec!["cpu".to_string(), "cuda".to_string(), "rocm".to_string()],
        })
    }

    fn extract_model_version(&self, model_path: &str, _config: &str) -> TrustformersResult<String> {
        // Extract version from path or config
        if let Some(version_start) = model_path.rfind("/v") {
            if let Some(version_end) = model_path[version_start + 2..].find('/') {
                return Ok(
                    model_path[version_start + 2..version_start + 2 + version_end].to_string(),
                );
            }
        }

        // Default version
        Ok("1.0.0".to_string())
    }

    fn schedule_model_warmup(&mut self, model_id: &str) -> TrustformersResult<()> {
        // Simplified warmup - in real implementation this would run inference
        // with sample inputs to warm up the model
        if let Some(mut entry) = self.cache.get_mut(model_id) {
            entry.warmup_completed = true;
        }
        Ok(())
    }

    fn start_cleanup_task(&mut self) -> TrustformersResult<()> {
        // Start background cleanup task
        // In real implementation this would spawn a thread
        Ok(())
    }

    fn start_health_check_task(&mut self) -> TrustformersResult<()> {
        // Start background health check task
        // In real implementation this would spawn a thread
        Ok(())
    }

    fn free_model_handle(&self, handle: usize) -> TrustformersResult<()> {
        // Free model resources
        // In real implementation this would call the appropriate cleanup functions
        Ok(())
    }
}
