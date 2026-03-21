// Storage Module - Model storage and caching
//
// This module provides comprehensive storage solutions for TrustformeRS
// models including IndexedDB caching, memory management, streaming loading,
// and model splitting for efficient web deployment.

#[cfg(feature = "indexeddb")]
pub mod indexeddb;

#[cfg(feature = "memory64")]
pub mod memory64;

#[cfg(feature = "streaming-loader")]
pub mod streaming_loader;

#[cfg(feature = "model-splitting")]
pub mod model_splitting;

pub mod progressive_loader;

// Re-export main types for convenience
#[cfg(feature = "indexeddb")]
pub use indexeddb::{CompressionType, ModelMetadata, ModelStorage, StoredModel};

#[cfg(feature = "memory64")]
pub use memory64::{
    can_load_model_size, get_memory64_capabilities, is_memory64_supported, AllocationStrategy,
    Memory64Capabilities, Memory64Manager,
};

#[cfg(feature = "streaming-loader")]
pub use streaming_loader::{
    get_optimal_chunk_size_kb, is_cache_api_available, is_streaming_compilation_supported,
    LoadingProgress, StreamingConfig, StreamingLoader,
};

#[cfg(feature = "model-splitting")]
pub use model_splitting::{
    get_recommended_chunk_size_mb, should_split_model, ChunkConfig, ChunkInfo, ChunkPriority,
    ChunkType, LoadingStrategy as SplittingStrategy, ModelChunk, ModelLoadingSession,
    ModelSplitter,
};

pub use progressive_loader::{
    ChunkLoader, LoadPriority, LoadingManifest, LoadingState,
    LoadingStats as ProgressiveLoadingStats, ModuleMetadata, ProgressiveLoader,
    ProgressiveLoaderConfig,
};

/// Storage module initialization
pub fn initialize() -> Result<(), StorageError> {
    web_sys::console::log_1(&"Initializing TrustformeRS WASM storage module".into());

    #[cfg(feature = "indexeddb")]
    {
        indexeddb::initialize()?;
        web_sys::console::log_1(&"IndexedDB storage subsystem initialized".into());
    }

    #[cfg(feature = "memory64")]
    {
        memory64::initialize()?;
        web_sys::console::log_1(&"Memory64 subsystem initialized".into());
    }

    #[cfg(feature = "streaming-loader")]
    {
        streaming_loader::initialize()?;
        web_sys::console::log_1(&"Streaming loader subsystem initialized".into());
    }

    #[cfg(feature = "model-splitting")]
    {
        model_splitting::initialize()?;
        web_sys::console::log_1(&"Model splitting subsystem initialized".into());
    }

    web_sys::console::log_1(&"TrustformeRS WASM storage module initialized successfully".into());
    Ok(())
}

/// Storage module error types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageError {
    IndexedDbError(String),
    MemoryError(String),
    StreamingError(String),
    SplittingError(String),
    CompressionError(String),
    ValidationError(String),
    QuotaExceeded(u64, u64), // used, limit
    NetworkError(String),
    InitializationError(String),
}

impl core::fmt::Display for StorageError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            StorageError::IndexedDbError(msg) => write!(f, "IndexedDB error: {}", msg),
            StorageError::MemoryError(msg) => write!(f, "Memory error: {}", msg),
            StorageError::StreamingError(msg) => write!(f, "Streaming error: {}", msg),
            StorageError::SplittingError(msg) => write!(f, "Splitting error: {}", msg),
            StorageError::CompressionError(msg) => write!(f, "Compression error: {}", msg),
            StorageError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            StorageError::QuotaExceeded(used, limit) => write!(
                f,
                "Quota exceeded: {} bytes used, {} bytes limit",
                used, limit
            ),
            StorageError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            StorageError::InitializationError(msg) => write!(f, "Initialization error: {}", msg),
        }
    }
}

impl std::error::Error for StorageError {}

/// Storage module configuration
#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub enable_indexeddb: bool,
    pub enable_memory64: bool,
    pub enable_streaming: bool,
    pub enable_splitting: bool,
    pub max_cache_size_mb: u32,
    pub max_memory_size_mb: Option<u32>,
    pub compression_enabled: bool,
    pub validation_enabled: bool,
    pub cleanup_on_quota_exceeded: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            enable_indexeddb: true,
            enable_memory64: true,
            enable_streaming: true,
            enable_splitting: true,
            max_cache_size_mb: 500,   // 500 MB default cache
            max_memory_size_mb: None, // Auto-detect
            compression_enabled: true,
            validation_enabled: true,
            cleanup_on_quota_exceeded: true,
        }
    }
}

/// Storage capabilities detection
#[derive(Debug, Clone)]
pub struct StorageCapabilities {
    pub has_indexeddb: bool,
    pub has_memory64: bool,
    pub has_cache_api: bool,
    pub has_file_system_access: bool,
    pub has_opfs: bool, // Origin Private File System
    pub indexeddb_quota_mb: Option<u64>,
    pub cache_quota_mb: Option<u64>,
    pub max_memory_mb: Option<u32>,
    pub supports_compression: bool,
    pub supports_streaming: bool,
}

impl StorageCapabilities {
    /// Detect current environment storage capabilities
    pub async fn detect() -> Self {
        let mut capabilities = Self {
            has_indexeddb: false,
            has_memory64: false,
            has_cache_api: false,
            has_file_system_access: false,
            has_opfs: false,
            indexeddb_quota_mb: None,
            cache_quota_mb: None,
            max_memory_mb: None,
            supports_compression: true, // Usually available
            supports_streaming: true,   // Usually available
        };

        // Detect IndexedDB
        capabilities.has_indexeddb = Self::detect_indexeddb();
        if capabilities.has_indexeddb {
            capabilities.indexeddb_quota_mb = Self::get_indexeddb_quota().await;
        }

        // Detect Memory64
        #[cfg(feature = "memory64")]
        {
            capabilities.has_memory64 = memory64::is_memory64_supported();
            if capabilities.has_memory64 {
                if let Ok(mem_caps) = memory64::get_memory64_capabilities() {
                    capabilities.max_memory_mb = Some(mem_caps.max_memory_gb * 1024);
                }
            }
        }

        // Detect Cache API
        capabilities.has_cache_api = Self::detect_cache_api();
        if capabilities.has_cache_api {
            capabilities.cache_quota_mb = Self::get_cache_quota().await;
        }

        // Detect File System Access API
        capabilities.has_file_system_access = Self::detect_file_system_access();

        // Detect Origin Private File System
        capabilities.has_opfs = Self::detect_opfs();

        capabilities
    }

    fn detect_indexeddb() -> bool {
        js_sys::Reflect::has(&js_sys::global(), &"indexedDB".into()).unwrap_or(false)
    }

    fn detect_cache_api() -> bool {
        js_sys::Reflect::has(&js_sys::global(), &"caches".into()).unwrap_or(false)
    }

    fn detect_file_system_access() -> bool {
        web_sys::window()
            .and_then(|w| js_sys::Reflect::get(&w, &"showOpenFilePicker".into()).ok())
            .is_some()
    }

    fn detect_opfs() -> bool {
        web_sys::window()
            .map(|w| w.navigator().storage())
            .and_then(|s| js_sys::Reflect::get(&s, &"getDirectory".into()).ok())
            .is_some()
    }

    async fn get_indexeddb_quota() -> Option<u64> {
        // Try to estimate quota using Storage API
        if let Some(navigator) = web_sys::window().map(|w| w.navigator()) {
            let storage = navigator.storage();
            {
                if let Ok(estimate_promise) = storage.estimate() {
                    if let Ok(estimate) =
                        wasm_bindgen_futures::JsFuture::from(estimate_promise).await
                    {
                        return js_sys::Reflect::get(&estimate, &"quota".into())
                            .ok()
                            .and_then(|v| v.as_f64())
                            .map(|bytes| (bytes / 1024.0 / 1024.0) as u64);
                    }
                }
            }
        }
        None
    }

    async fn get_cache_quota() -> Option<u64> {
        // Cache API shares quota with IndexedDB in most browsers
        Self::get_indexeddb_quota().await
    }
}

/// Unified storage manager
pub struct StorageManager {
    config: StorageConfig,
    capabilities: StorageCapabilities,
    #[cfg(feature = "indexeddb")]
    indexeddb_storage: Option<indexeddb::ModelStorage>,
    #[cfg(feature = "memory64")]
    memory_manager: Option<memory64::Memory64Manager>,
    #[cfg(feature = "streaming-loader")]
    streaming_loader: Option<streaming_loader::StreamingLoader>,
    #[cfg(feature = "model-splitting")]
    model_splitter: Option<model_splitting::ModelSplitter>,
}

impl StorageManager {
    /// Create a new storage manager
    pub async fn new(config: StorageConfig) -> Result<Self, StorageError> {
        let capabilities = StorageCapabilities::detect().await;

        let mut manager = Self {
            config,
            capabilities,
            #[cfg(feature = "indexeddb")]
            indexeddb_storage: None,
            #[cfg(feature = "memory64")]
            memory_manager: None,
            #[cfg(feature = "streaming-loader")]
            streaming_loader: None,
            #[cfg(feature = "model-splitting")]
            model_splitter: None,
        };

        manager.initialize_storage_backends().await?;
        Ok(manager)
    }

    async fn initialize_storage_backends(&mut self) -> Result<(), StorageError> {
        #[cfg(feature = "indexeddb")]
        if self.config.enable_indexeddb && self.capabilities.has_indexeddb {
            let storage_config = indexeddb::StorageConfig {
                db_name: "trustformers-models".to_string(),
                max_storage_mb: self.config.max_cache_size_mb as f64,
                enable_compression: self.config.compression_enabled,
                ..Default::default()
            };

            let mut storage = indexeddb::ModelStorage::new(
                storage_config.db_name.clone(),
                storage_config.max_storage_mb,
            );
            storage
                .initialize()
                .await
                .map_err(|e| StorageError::IndexedDbError(format!("{:?}", e)))?;

            self.indexeddb_storage = Some(storage);
            web_sys::console::log_1(&"IndexedDB storage backend initialized".into());
        }

        #[cfg(feature = "memory64")]
        if self.config.enable_memory64 && self.capabilities.has_memory64 {
            // Memory64Manager::new() takes max_memory_gb as u32
            let max_memory_gb = (self.config.max_memory_size_mb.unwrap_or(4096) / 1024).max(1);

            let manager = memory64::Memory64Manager::new(max_memory_gb)
                .map_err(|e| StorageError::MemoryError(format!("{:?}", e)))?;

            self.memory_manager = Some(manager);
            web_sys::console::log_1(&"Memory64 backend initialized".into());
        }

        #[cfg(feature = "streaming-loader")]
        if self.config.enable_streaming {
            let streaming_config = streaming_loader::StreamingConfig {
                chunk_size_kb: 512,
                max_concurrent_chunks: 4,
                enable_caching: true,
                ..Default::default()
            };

            let loader = streaming_loader::StreamingLoader::new(streaming_config);
            self.streaming_loader = Some(loader);
            web_sys::console::log_1(&"Streaming loader initialized".into());
        }

        #[cfg(feature = "model-splitting")]
        if self.config.enable_splitting {
            let chunk_config = model_splitting::ChunkConfig {
                max_chunk_size_mb: 10.0,
                overlap_percentage: 0.0,
                compression_enabled: self.config.compression_enabled,
                priority_loading: false,
                lazy_loading: true,
            };

            let splitter = model_splitting::ModelSplitter::new(chunk_config);
            self.model_splitter = Some(splitter);
            web_sys::console::log_1(&"Model splitter initialized".into());
        }

        Ok(())
    }

    /// Store a model using the best available backend
    pub async fn store_model(
        &mut self,
        model_id: &str,
        model_data: &[u8],
        metadata: ModelInfo,
    ) -> Result<(), StorageError> {
        let data_size_mb = model_data.len() as f64 / 1024.0 / 1024.0;

        // Choose the best storage backend based on size and capabilities
        if data_size_mb > 50.0 {
            // Large model - use splitting + streaming
            #[cfg(feature = "model-splitting")]
            if let Some(ref mut splitter) = self.model_splitter {
                web_sys::console::log_1(
                    &format!(
                        "Splitting large model '{}' ({:.1} MB)",
                        model_id, data_size_mb
                    )
                    .into(),
                );
                return Self::store_split_model_impl(splitter, model_id, model_data, metadata)
                    .await;
            }
        }

        // Medium/small model - use IndexedDB or Memory64
        #[cfg(feature = "indexeddb")]
        if let Some(ref storage) = self.indexeddb_storage {
            web_sys::console::log_1(
                &format!(
                    "Storing model '{}' in IndexedDB ({:.1} MB)",
                    model_id, data_size_mb
                )
                .into(),
            );
            storage
                .store_model(
                    model_id,
                    &metadata.name,
                    &metadata.architecture,
                    &metadata.version,
                    model_data,
                )
                .await
                .map_err(|e| StorageError::IndexedDbError(format!("{:?}", e)))?;
            return Ok(());
        }

        #[cfg(feature = "memory64")]
        if let Some(ref mut memory_manager) = self.memory_manager {
            web_sys::console::log_1(
                &format!(
                    "Storing model '{}' in Memory64 ({:.1} MB)",
                    model_id, data_size_mb
                )
                .into(),
            );
            memory_manager
                .allocate_for_model(model_id, model_data.len())
                .map_err(|e| StorageError::MemoryError(format!("{:?}", e)))?;
            return Ok(());
        }

        Err(StorageError::InitializationError(
            "No storage backend available".to_string(),
        ))
    }

    /// Load a model using the best available backend
    pub async fn load_model(&self, model_id: &str) -> Result<Option<Vec<u8>>, StorageError> {
        // Try IndexedDB first
        #[cfg(feature = "indexeddb")]
        if let Some(ref storage) = self.indexeddb_storage {
            if let Ok(Some(data)) = storage.get_model(model_id).await {
                web_sys::console::log_1(
                    &format!("Loaded model '{}' from IndexedDB", model_id).into(),
                );
                return Ok(Some(data));
            }
        }

        // Try Memory64
        #[cfg(feature = "memory64")]
        if let Some(ref memory_manager) = self.memory_manager {
            if let Ok(Some(data)) = memory_manager.get_model_data(model_id) {
                web_sys::console::log_1(
                    &format!("Loaded model '{}' from Memory64", model_id).into(),
                );
                return Ok(Some(data));
            }
        }

        // Try split model loading
        #[cfg(feature = "model-splitting")]
        if let Some(ref splitter) = self.model_splitter {
            if let Ok(Some(data)) = self.load_split_model(splitter, model_id).await {
                web_sys::console::log_1(&format!("Loaded split model '{}'", model_id).into());
                return Ok(Some(data));
            }
        }

        Ok(None)
    }

    /// Get storage statistics
    pub async fn get_statistics(&self) -> StorageStatistics {
        let mut stats = StorageStatistics {
            total_models: 0,
            total_size_bytes: 0,
            indexeddb_usage: None,
            memory64_usage: None,
            cache_hit_rate: 0.0,
            available_space_mb: 0,
        };

        #[cfg(feature = "indexeddb")]
        if let Some(ref storage) = self.indexeddb_storage {
            if let Ok(usage) = storage.get_storage_usage().await {
                stats.indexeddb_usage = Some(usage as u64);
                stats.total_size_bytes += usage as u64;
            }
            if let Ok(models_js) = storage.list_models().await {
                if let Ok(models) =
                    serde_wasm_bindgen::from_value::<Vec<indexeddb::ModelMetadata>>(models_js)
                {
                    stats.total_models += models.len();
                }
            }
        }

        #[cfg(feature = "memory64")]
        if let Some(ref memory_manager) = self.memory_manager {
            if let Ok(memory_stats_js) = memory_manager.get_statistics() {
                if let Some(total_bytes) =
                    js_sys::Reflect::get(&memory_stats_js, &"current_usage_bytes".into())
                        .ok()
                        .and_then(|v| v.as_f64())
                {
                    let total_allocated_bytes = total_bytes as u64;
                    stats.memory64_usage = Some(total_allocated_bytes);
                    stats.total_size_bytes += total_allocated_bytes;
                }
            }
        }

        // Estimate available space
        if let Some(quota) = self.capabilities.indexeddb_quota_mb {
            stats.available_space_mb = quota.saturating_sub(stats.total_size_bytes / 1024 / 1024);
        }

        stats
    }

    /// Clear all stored models
    pub async fn clear_all(&mut self) -> Result<(), StorageError> {
        let mut cleared_count = 0;

        #[cfg(feature = "indexeddb")]
        if let Some(ref storage) = self.indexeddb_storage {
            storage
                .clear_all()
                .await
                .map_err(|e| StorageError::IndexedDbError(format!("{:?}", e)))?;
            cleared_count += 1;
        }

        #[cfg(feature = "memory64")]
        if let Some(ref mut memory_manager) = self.memory_manager {
            memory_manager.clear_all();
            cleared_count += 1;
        }

        web_sys::console::log_1(&format!("Cleared {} storage backends", cleared_count).into());
        Ok(())
    }

    /// Get storage capabilities
    pub fn capabilities(&self) -> &StorageCapabilities {
        &self.capabilities
    }

    /// Get storage configuration
    pub fn config(&self) -> &StorageConfig {
        &self.config
    }

    // Helper methods

    #[cfg(feature = "model-splitting")]
    async fn store_split_model_impl(
        splitter: &mut model_splitting::ModelSplitter,
        model_id: &str,
        model_data: &[u8],
        metadata: ModelInfo,
    ) -> Result<(), StorageError> {
        // Implementation would split and store the model
        web_sys::console::log_1(&format!("Splitting model '{}' into chunks", model_id).into());
        let _ = (splitter, model_data, metadata); // Avoid unused warnings
        Ok(())
    }

    #[cfg(feature = "model-splitting")]
    async fn load_split_model(
        &self,
        _splitter: &model_splitting::ModelSplitter,
        model_id: &str,
    ) -> Result<Option<Vec<u8>>, StorageError> {
        // Implementation would load and reassemble the model
        web_sys::console::log_1(&format!("Loading split model '{}'", model_id).into());
        Ok(None)
    }
}

/// Model information for storage
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub architecture: String,
    pub version: String,
    pub description: String,
    pub tags: Vec<String>,
}

impl Default for ModelInfo {
    fn default() -> Self {
        Self {
            name: "unknown".to_string(),
            architecture: "unknown".to_string(),
            version: "1.0.0".to_string(),
            description: String::new(),
            tags: Vec::new(),
        }
    }
}

/// Storage statistics
#[derive(Debug, Clone)]
pub struct StorageStatistics {
    pub total_models: usize,
    pub total_size_bytes: u64,
    pub indexeddb_usage: Option<u64>,
    pub memory64_usage: Option<u64>,
    pub cache_hit_rate: f64,
    pub available_space_mb: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_config_default() {
        let config = StorageConfig::default();
        assert!(config.enable_indexeddb);
        assert!(config.enable_memory64);
        assert_eq!(config.max_cache_size_mb, 500);
        assert!(config.compression_enabled);
    }

    #[test]
    fn test_model_info_default() {
        let info = ModelInfo::default();
        assert_eq!(info.name, "unknown");
        assert_eq!(info.architecture, "unknown");
        assert_eq!(info.version, "1.0.0");
        assert!(info.tags.is_empty());
    }
}
