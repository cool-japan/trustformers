//! Android Content Provider for Model Management
//!
//! This module provides a Content Provider implementation for managing
//! TrustformeRS models on Android, enabling secure model sharing between
//! applications and centralized model management.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use trustformers_core::errors::Result;
use trustformers_core::TrustformersError;

// Real AEAD primitives backing `EncryptionManager` -- RustCrypto's
// `aes-gcm` and `chacha20poly1305`, already workspace dependencies.
// `aead::Generate` sources the nonce from the operating system CSPRNG (the
// `getrandom` cargo feature both crates enable by default), never a seeded
// PRNG or a fixed value.
use aes_gcm::aead::{Aead, Generate, KeyInit};
use aes_gcm::{Aes128Gcm, Aes256Gcm};
use chacha20poly1305::ChaCha20Poly1305;

/// Android Content Provider for TrustformeRS models
pub struct AndroidModelContentProvider {
    config: ContentProviderConfig,
    model_registry: ModelRegistry,
    security_manager: ContentProviderSecurity,
    cache_manager: ModelCacheManager,
    uri_matcher: UriMatcher,
}

/// Configuration for the Content Provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentProviderConfig {
    /// Authority for the content provider
    pub authority: String,
    /// Enable model sharing between apps
    pub enable_sharing: bool,
    /// Maximum number of cached models
    pub max_cached_models: usize,
    /// Cache size limit in MB
    pub cache_size_limit_mb: usize,
    /// Security configuration
    pub security: SecurityConfig,
    /// Performance configuration
    pub performance: PerformanceConfig,
}

/// Security configuration for Content Provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Require signature verification for model access
    pub require_signature_verification: bool,
    /// Allowed package names for model access
    pub allowed_packages: Vec<String>,
    /// Enable per-model permissions
    pub per_model_permissions: bool,
    /// Encryption configuration
    pub encryption: EncryptionConfig,
}

/// Encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Enable at-rest encryption
    pub encrypt_at_rest: bool,
    /// Enable in-transit encryption
    pub encrypt_in_transit: bool,
    /// Encryption algorithm
    pub algorithm: EncryptionAlgorithm,
    /// Key management configuration
    pub key_management: KeyManagementConfig,
}

/// Encryption algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    AES256GCM,
    ChaCha20Poly1305,
    AES128GCM,
}

/// Key management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyManagementConfig {
    /// Use Android Keystore
    pub use_android_keystore: bool,
    /// Key rotation period in days
    pub key_rotation_days: u32,
    /// Backup key configuration
    pub backup_keys: bool,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable background model loading
    pub background_loading: bool,
    /// Number of worker threads
    pub worker_threads: usize,
    /// Connection pool size
    pub connection_pool_size: usize,
    /// Enable compression for model transfer
    pub enable_compression: bool,
    /// Compression algorithm
    pub compression_algorithm: CompressionAlgorithm,
}

/// Compression algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    Gzip,
    Lz4,
    Zstd,
}

/// Model registry for tracking available models
struct ModelRegistry {
    models: HashMap<String, ModelInfo>,
    access_logs: HashMap<String, Vec<AccessLog>>,
    usage_stats: HashMap<String, UsageStats>,
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Unique model identifier
    pub id: String,
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model type
    pub model_type: ModelType,
    /// File path
    pub file_path: PathBuf,
    /// Model size in bytes
    pub size_bytes: u64,
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Access permissions
    pub permissions: ModelPermissions,
    /// Creation timestamp
    pub created_at: u64,
    /// Last modified timestamp
    pub modified_at: u64,
}

/// Model types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    Transformer,
    CNN,
    RNN,
    Custom,
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model description
    pub description: String,
    /// Model architecture
    pub architecture: String,
    /// Training dataset
    pub dataset: Option<String>,
    /// Model accuracy metrics
    pub accuracy: Option<f32>,
    /// Model latency (ms)
    pub latency_ms: Option<f32>,
    /// Memory usage (MB)
    pub memory_mb: Option<f32>,
    /// Tags for categorization
    pub tags: Vec<String>,
}

/// Model access permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPermissions {
    /// Public access allowed
    pub public_access: bool,
    /// Allowed package names
    pub allowed_packages: Vec<String>,
    /// Required permissions
    pub required_permissions: Vec<String>,
    /// Access level
    pub access_level: AccessLevel,
}

/// Access levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessLevel {
    Public,
    Restricted,
    Private,
    System,
}

/// Access log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AccessLog {
    timestamp: u64,
    package_name: String,
    operation: Operation,
    result: AccessResult,
}

/// Content Provider operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operation {
    Query,
    Insert,
    Update,
    Delete,
    Download,
    Stream,
}

/// Access results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessResult {
    Success,
    Denied,
    Error,
    NotFound,
}

/// Usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
struct UsageStats {
    total_accesses: u64,
    successful_accesses: u64,
    failed_accesses: u64,
    average_response_time_ms: f32,
    last_access: u64,
    bandwidth_used_mb: f64,
}

/// Content Provider security manager
struct ContentProviderSecurity {
    config: SecurityConfig,
    signature_verifier: SignatureVerifier,
    encryption_manager: EncryptionManager,
}

/// Signature verification
struct SignatureVerifier {
    trusted_signatures: Vec<String>,
}

/// Encryption manager
struct EncryptionManager {
    config: EncryptionConfig,
    active_keys: HashMap<String, Vec<u8>>,
}

/// Model cache manager
struct ModelCacheManager {
    cache: HashMap<String, CachedModel>,
    cache_size_bytes: u64,
    config: PerformanceConfig,
}

/// Cached model information
struct CachedModel {
    model_info: ModelInfo,
    data: Vec<u8>,
    last_accessed: u64,
    access_count: u64,
}

/// URI matcher for Content Provider
struct UriMatcher {
    patterns: HashMap<String, UriPattern>,
}

/// URI patterns
#[derive(Debug, Clone)]
struct UriPattern {
    pattern: String,
    operation: Operation,
    parameters: Vec<String>,
}

/// Content Provider query parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryParams {
    /// Model type filter
    pub model_type: Option<ModelType>,
    /// Name filter
    pub name_filter: Option<String>,
    /// Version filter
    pub version_filter: Option<String>,
    /// Tags filter
    pub tags: Vec<String>,
    /// Maximum results
    pub limit: Option<usize>,
    /// Offset for pagination
    pub offset: Option<usize>,
    /// Sort order
    pub sort_by: Option<SortOrder>,
}

/// Sort orders
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SortOrder {
    Name,
    Version,
    Size,
    CreatedAt,
    ModifiedAt,
    Usage,
}

/// Query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Found models
    pub models: Vec<ModelInfo>,
    /// Total count (for pagination)
    pub total_count: usize,
    /// Query execution time (ms)
    pub execution_time_ms: f32,
}

impl Default for ContentProviderConfig {
    fn default() -> Self {
        Self {
            authority: "com.trustformers.models".to_string(),
            enable_sharing: true,
            max_cached_models: 10,
            cache_size_limit_mb: 500,
            security: SecurityConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            require_signature_verification: true,
            allowed_packages: vec![],
            per_model_permissions: true,
            encryption: EncryptionConfig::default(),
        }
    }
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            encrypt_at_rest: true,
            encrypt_in_transit: true,
            algorithm: EncryptionAlgorithm::AES256GCM,
            key_management: KeyManagementConfig::default(),
        }
    }
}

impl Default for KeyManagementConfig {
    fn default() -> Self {
        Self {
            use_android_keystore: true,
            key_rotation_days: 90,
            backup_keys: true,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            background_loading: true,
            worker_threads: 4,
            connection_pool_size: 10,
            enable_compression: true,
            compression_algorithm: CompressionAlgorithm::Lz4,
        }
    }
}

impl AndroidModelContentProvider {
    /// Create new Android Content Provider
    pub fn new(config: ContentProviderConfig) -> Result<Self> {
        let model_registry = ModelRegistry::new();
        let security_manager = ContentProviderSecurity::new(config.security.clone())?;
        let cache_manager = ModelCacheManager::new(config.performance.clone());
        let uri_matcher = UriMatcher::new(&config.authority);

        Ok(Self {
            config,
            model_registry,
            security_manager,
            cache_manager,
            uri_matcher,
        })
    }

    /// Register a new model in the Content Provider
    pub fn register_model(&mut self, model_info: ModelInfo) -> Result<()> {
        // Validate model information
        self.validate_model_info(&model_info)?;

        // Check security permissions
        self.security_manager.verify_model_registration(&model_info)?;

        // Register in model registry
        self.model_registry.register_model(model_info.clone())?;

        // Cache model if enabled
        if self.config.performance.background_loading {
            self.cache_manager.preload_model(&model_info)?;
        }

        tracing::info!("Model registered: {} ({})", model_info.name, model_info.id);
        Ok(())
    }

    /// Query models using Content Provider interface
    pub fn query(&mut self, params: QueryParams, calling_package: &str) -> Result<QueryResult> {
        let start_time = std::time::Instant::now();

        // Check security permissions
        self.security_manager.verify_query_access(calling_package)?;

        // Execute query
        let models = self.model_registry.query_models(&params)?;

        // Filter based on permissions
        let filtered_models = self.filter_models_by_permissions(&models, calling_package)?;

        let execution_time = start_time.elapsed().as_millis() as f32;

        // Log access
        self.log_access(calling_package, Operation::Query, AccessResult::Success);

        Ok(QueryResult {
            models: filtered_models.clone(),
            total_count: filtered_models.len(),
            execution_time_ms: execution_time,
        })
    }

    /// Download model data
    pub fn download_model(&mut self, model_id: &str, calling_package: &str) -> Result<Vec<u8>> {
        // Check security permissions
        self.security_manager.verify_download_access(model_id, calling_package)?;

        // Check cache first
        if let Some(cached_data) = self.cache_manager.get_cached_model(model_id)? {
            self.log_access(calling_package, Operation::Download, AccessResult::Success);
            return Ok(cached_data);
        }

        // Load from storage
        let model_info = self.model_registry.get_model(model_id)?.ok_or_else(|| {
            TrustformersError::runtime_error(format!("Model not found: {}", model_id))
        })?;

        let data = self.load_model_data(&model_info)?;

        // Encrypt if required
        let encrypted_data = if self.config.security.encryption.encrypt_in_transit {
            self.security_manager.encrypt_data(&data)?
        } else {
            data
        };

        // Update cache
        self.cache_manager.cache_model(model_id, &encrypted_data)?;

        // Log access
        self.log_access(calling_package, Operation::Download, AccessResult::Success);

        Ok(encrypted_data)
    }

    /// Stream model data (for large models)
    pub fn stream_model(&mut self, model_id: &str, calling_package: &str) -> Result<ModelStream> {
        // Check security permissions
        self.security_manager.verify_download_access(model_id, calling_package)?;

        let model_info = self.model_registry.get_model(model_id)?.ok_or_else(|| {
            TrustformersError::runtime_error(format!("Model not found: {}", model_id))
        })?;

        // Create streaming interface
        let stream = ModelStream::new(model_info, self.config.performance.compression_algorithm)?;

        // Log access
        self.log_access(calling_package, Operation::Stream, AccessResult::Success);

        Ok(stream)
    }

    /// Update model metadata
    pub fn update_model(
        &mut self,
        model_id: &str,
        metadata: ModelMetadata,
        calling_package: &str,
    ) -> Result<()> {
        // Check security permissions
        self.security_manager.verify_update_access(model_id, calling_package)?;

        // Update model in registry
        self.model_registry.update_model_metadata(model_id, metadata)?;

        // Invalidate cache
        self.cache_manager.invalidate_cache(model_id)?;

        // Log access
        self.log_access(calling_package, Operation::Update, AccessResult::Success);

        Ok(())
    }

    /// Delete model
    pub fn delete_model(&mut self, model_id: &str, calling_package: &str) -> Result<()> {
        // Check security permissions
        self.security_manager.verify_delete_access(model_id, calling_package)?;

        // Remove from registry
        self.model_registry.remove_model(model_id)?;

        // Remove from cache
        self.cache_manager.remove_from_cache(model_id)?;

        // Log access
        self.log_access(calling_package, Operation::Delete, AccessResult::Success);

        Ok(())
    }

    /// Get usage statistics
    pub fn get_usage_stats(&self, model_id: Option<&str>) -> Result<Vec<(String, UsageStats)>> {
        match model_id {
            Some(id) => {
                if let Some(stats) = self.model_registry.get_usage_stats(id) {
                    Ok(vec![(id.to_string(), stats)])
                } else {
                    Ok(vec![])
                }
            },
            None => Ok(self.model_registry.get_all_usage_stats()),
        }
    }

    /// Clean up expired cache entries
    pub fn cleanup_cache(&mut self) -> Result<()> {
        self.cache_manager.cleanup_expired_entries()?;
        Ok(())
    }

    // Private helper methods

    fn validate_model_info(&self, model_info: &ModelInfo) -> Result<()> {
        if model_info.id.is_empty() {
            return Err(TrustformersError::config_error(
                "Model ID cannot be empty",
                "validate",
            ));
        }

        if model_info.name.is_empty() {
            return Err(TrustformersError::config_error(
                "Model name cannot be empty",
                "validate",
            ));
        }

        if !model_info.file_path.exists() {
            return Err(TrustformersError::runtime_error(format!(
                "Model file does not exist: {:?}",
                model_info.file_path
            )));
        }

        Ok(())
    }

    fn filter_models_by_permissions(
        &self,
        models: &[ModelInfo],
        calling_package: &str,
    ) -> Result<Vec<ModelInfo>> {
        let mut filtered = Vec::new();

        for model in models {
            if self.security_manager.check_model_access(&model.permissions, calling_package)? {
                filtered.push(model.clone());
            }
        }

        Ok(filtered)
    }

    fn load_model_data(&self, model_info: &ModelInfo) -> Result<Vec<u8>> {
        std::fs::read(&model_info.file_path).map_err(|e| {
            TrustformersError::runtime_error(format!("Failed to read model file: {}", e))
        })
    }

    fn log_access(&mut self, package_name: &str, operation: Operation, result: AccessResult) {
        let log_entry = AccessLog {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            package_name: package_name.to_string(),
            operation,
            result,
        };

        self.model_registry.add_access_log("global".to_string(), log_entry);
    }
}

/// Model streaming interface for large models
pub struct ModelStream {
    model_info: ModelInfo,
    compression: CompressionAlgorithm,
    chunk_size: usize,
    current_position: usize,
}

impl ModelStream {
    fn new(model_info: ModelInfo, compression: CompressionAlgorithm) -> Result<Self> {
        Ok(Self {
            model_info,
            compression,
            chunk_size: 1024 * 1024, // 1MB chunks
            current_position: 0,
        })
    }

    /// Read the next chunk of model data from the real file on disk at
    /// `model_info.file_path`, starting at `current_position`. Previously a
    /// `// Placeholder implementation` that synthesized `vec![0u8; ...]` --
    /// every streamed chunk was zeroed regardless of the file's actual
    /// contents. Uses `Read + Seek` so repeated calls resume from where the
    /// last one left off without re-reading earlier bytes.
    pub fn read_chunk(&mut self) -> Result<Option<Vec<u8>>> {
        use std::io::{Read, Seek, SeekFrom};

        if self.current_position >= self.model_info.size_bytes as usize {
            return Ok(None);
        }

        let chunk_end =
            (self.current_position + self.chunk_size).min(self.model_info.size_bytes as usize);
        let chunk_len = chunk_end - self.current_position;

        let mut file = std::fs::File::open(&self.model_info.file_path).map_err(|e| {
            TrustformersError::runtime_error(format!(
                "failed to open model file {:?} for streaming: {e}",
                self.model_info.file_path
            ))
        })?;
        file.seek(SeekFrom::Start(self.current_position as u64)).map_err(|e| {
            TrustformersError::runtime_error(format!(
                "failed to seek model file {:?} to offset {}: {e}",
                self.model_info.file_path, self.current_position
            ))
        })?;

        let mut chunk = vec![0u8; chunk_len];
        file.read_exact(&mut chunk).map_err(|e| {
            TrustformersError::runtime_error(format!(
                "failed to read {chunk_len} bytes from model file {:?} at offset {}: {e}",
                self.model_info.file_path, self.current_position
            ))
        })?;
        self.current_position = chunk_end;

        Ok(Some(chunk))
    }

    /// Get stream progress
    pub fn progress(&self) -> f32 {
        self.current_position as f32 / self.model_info.size_bytes as f32
    }
}

// Implementation of helper structs

impl ModelRegistry {
    fn new() -> Self {
        Self {
            models: HashMap::new(),
            access_logs: HashMap::new(),
            usage_stats: HashMap::new(),
        }
    }

    fn register_model(&mut self, model_info: ModelInfo) -> Result<()> {
        let id = model_info.id.clone();
        self.models.insert(id.clone(), model_info);
        self.usage_stats.insert(id, UsageStats::new());
        Ok(())
    }

    fn query_models(&self, params: &QueryParams) -> Result<Vec<ModelInfo>> {
        let mut results: Vec<_> = self.models.values().cloned().collect();

        // Apply filters
        if let Some(model_type) = params.model_type {
            results.retain(|m| m.model_type == model_type);
        }

        if let Some(ref name_filter) = params.name_filter {
            results.retain(|m| m.name.contains(name_filter));
        }

        if let Some(ref version_filter) = params.version_filter {
            results.retain(|m| m.version == *version_filter);
        }

        if !params.tags.is_empty() {
            results.retain(|m| params.tags.iter().any(|tag| m.metadata.tags.contains(tag)));
        }

        // Apply sorting
        if let Some(sort_by) = params.sort_by {
            match sort_by {
                SortOrder::Name => results.sort_by_key(|a| a.name.clone()),
                SortOrder::Version => results.sort_by_key(|a| a.version.clone()),
                SortOrder::Size => results.sort_by_key(|a| a.size_bytes),
                SortOrder::CreatedAt => results.sort_by_key(|a| a.created_at),
                SortOrder::ModifiedAt => results.sort_by_key(|a| a.modified_at),
                SortOrder::Usage => {
                    results.sort_by(|a, b| {
                        let usage_a =
                            self.usage_stats.get(&a.id).map(|u| u.total_accesses).unwrap_or(0);
                        let usage_b =
                            self.usage_stats.get(&b.id).map(|u| u.total_accesses).unwrap_or(0);
                        usage_b.cmp(&usage_a) // Descending order
                    });
                },
            }
        }

        // Apply pagination
        if let Some(offset) = params.offset {
            if offset < results.len() {
                results = results.into_iter().skip(offset).collect();
            } else {
                results.clear();
            }
        }

        if let Some(limit) = params.limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    fn get_model(&self, model_id: &str) -> Result<Option<ModelInfo>> {
        Ok(self.models.get(model_id).cloned())
    }

    fn update_model_metadata(&mut self, model_id: &str, metadata: ModelMetadata) -> Result<()> {
        if let Some(model) = self.models.get_mut(model_id) {
            model.metadata = metadata;
            model.modified_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            Ok(())
        } else {
            Err(TrustformersError::runtime_error(format!(
                "Model not found: {}",
                model_id
            )))
        }
    }

    fn remove_model(&mut self, model_id: &str) -> Result<()> {
        self.models.remove(model_id);
        self.access_logs.remove(model_id);
        self.usage_stats.remove(model_id);
        Ok(())
    }

    fn add_access_log(&mut self, model_id: String, log: AccessLog) {
        self.access_logs.entry(model_id).or_default().push(log);
    }

    fn get_usage_stats(&self, model_id: &str) -> Option<UsageStats> {
        self.usage_stats.get(model_id).cloned()
    }

    fn get_all_usage_stats(&self) -> Vec<(String, UsageStats)> {
        self.usage_stats.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }
}

impl UsageStats {
    fn new() -> Self {
        Self {
            total_accesses: 0,
            successful_accesses: 0,
            failed_accesses: 0,
            average_response_time_ms: 0.0,
            last_access: 0,
            bandwidth_used_mb: 0.0,
        }
    }
}

impl ContentProviderSecurity {
    fn new(config: SecurityConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            signature_verifier: SignatureVerifier::new(),
            encryption_manager: EncryptionManager::new(config.encryption)?,
        })
    }

    fn verify_model_registration(&self, _model_info: &ModelInfo) -> Result<()> {
        // Implementation would verify registration permissions
        Ok(())
    }

    fn verify_query_access(&self, _calling_package: &str) -> Result<()> {
        // Implementation would verify query permissions
        Ok(())
    }

    fn verify_download_access(&self, _model_id: &str, _calling_package: &str) -> Result<()> {
        // Implementation would verify download permissions
        Ok(())
    }

    fn verify_update_access(&self, _model_id: &str, _calling_package: &str) -> Result<()> {
        // Implementation would verify update permissions
        Ok(())
    }

    fn verify_delete_access(&self, _model_id: &str, _calling_package: &str) -> Result<()> {
        // Implementation would verify delete permissions
        Ok(())
    }

    fn check_model_access(
        &self,
        permissions: &ModelPermissions,
        calling_package: &str,
    ) -> Result<bool> {
        if permissions.public_access {
            return Ok(true);
        }

        if permissions.allowed_packages.contains(&calling_package.to_string()) {
            return Ok(true);
        }

        Ok(false)
    }

    fn encrypt_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.encryption_manager.encrypt(data)
    }
}

impl SignatureVerifier {
    fn new() -> Self {
        Self {
            trusted_signatures: Vec::new(),
        }
    }
}

/// Key id [`EncryptionManager`] stores its one active symmetric key under.
/// A future revision that needs multiple concurrent keys (e.g. one per
/// model, for independent rotation) can extend `active_keys` to more than
/// one entry; today's callers (`ContentProviderSecurity::encrypt_data`)
/// only ever need one.
const ACTIVE_KEY_ID: &str = "default";

impl EncryptionManager {
    /// Generates one real 32-byte symmetric key from the OS CSPRNG for this
    /// instance's lifetime. Previously `active_keys` was constructed empty
    /// and never populated by anything -- there was no key to encrypt with,
    /// which fit the fact that [`Self::encrypt`] never actually used one.
    fn new(config: EncryptionConfig) -> Result<Self> {
        let mut key = vec![0u8; 32];
        getrandom::fill(&mut key).map_err(|e| {
            TrustformersError::runtime_error(format!(
                "failed to generate an encryption key from the OS CSPRNG: {e}"
            ))
        })?;

        let mut active_keys = HashMap::new();
        active_keys.insert(ACTIVE_KEY_ID.to_string(), key);

        Ok(Self {
            config,
            active_keys,
        })
    }

    fn active_key(&self) -> Result<&[u8]> {
        self.active_keys.get(ACTIVE_KEY_ID).map(Vec::as_slice).ok_or_else(|| {
            TrustformersError::runtime_error(
                "EncryptionManager has no active key (internal error: EncryptionManager::new \
                 should always populate one)"
                    .to_string(),
            )
        })
    }

    /// Real AEAD sealing, algorithm selected by
    /// `self.config.algorithm`. Output layout is `nonce (12 bytes) ||
    /// ciphertext+tag`. Replaces a previous implementation documented `//
    /// Placeholder encryption implementation` that was `Ok(data.to_vec())`
    /// -- content marked for at-rest/in-transit encryption left the
    /// function byte-for-byte identical to its plaintext input, with no
    /// indication to any caller that nothing had actually been protected.
    fn encrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        let key = self.active_key()?;

        let (nonce_bytes, ciphertext): (Vec<u8>, Vec<u8>) = match self.config.algorithm {
            EncryptionAlgorithm::AES256GCM => {
                let cipher = Aes256Gcm::new_from_slice(key).map_err(|e| {
                    TrustformersError::runtime_error(format!(
                        "failed to initialise AES-256-GCM cipher: {e}"
                    ))
                })?;
                let nonce = aes_gcm::Nonce::generate();
                let ct = cipher.encrypt(&nonce, data).map_err(|e| {
                    TrustformersError::runtime_error(format!("AES-256-GCM encryption failed: {e}"))
                })?;
                (nonce.to_vec(), ct)
            },
            EncryptionAlgorithm::AES128GCM => {
                let cipher = Aes128Gcm::new_from_slice(&key[..16]).map_err(|e| {
                    TrustformersError::runtime_error(format!(
                        "failed to initialise AES-128-GCM cipher: {e}"
                    ))
                })?;
                let nonce = aes_gcm::Nonce::generate();
                let ct = cipher.encrypt(&nonce, data).map_err(|e| {
                    TrustformersError::runtime_error(format!("AES-128-GCM encryption failed: {e}"))
                })?;
                (nonce.to_vec(), ct)
            },
            EncryptionAlgorithm::ChaCha20Poly1305 => {
                let cipher = ChaCha20Poly1305::new_from_slice(key).map_err(|e| {
                    TrustformersError::runtime_error(format!(
                        "failed to initialise ChaCha20-Poly1305 cipher: {e}"
                    ))
                })?;
                let nonce = chacha20poly1305::Nonce::generate();
                let ct = cipher.encrypt(&nonce, data).map_err(|e| {
                    TrustformersError::runtime_error(format!(
                        "ChaCha20-Poly1305 encryption failed: {e}"
                    ))
                })?;
                (nonce.to_vec(), ct)
            },
        };

        let mut out = Vec::with_capacity(nonce_bytes.len() + ciphertext.len());
        out.extend_from_slice(&nonce_bytes);
        out.extend_from_slice(&ciphertext);
        Ok(out)
    }

    /// Real AEAD opening matching [`Self::encrypt`]'s output layout and
    /// algorithm selection. Returns an error -- not corrupted plaintext --
    /// on the wrong key or tampered ciphertext, since every algorithm this
    /// manager supports is an authenticated cipher.
    #[allow(dead_code)] // the real counterpart to `encrypt`; not yet called by a decrypt-side caller in this module
    fn decrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        const NONCE_LEN: usize = 12;
        if data.len() < NONCE_LEN {
            return Err(TrustformersError::runtime_error(format!(
                "encrypted data must be at least {NONCE_LEN} bytes (AEAD nonce size), got {}",
                data.len()
            )));
        }
        let key = self.active_key()?;
        let (nonce_bytes, ciphertext) = data.split_at(NONCE_LEN);

        let auth_fail = |e: aes_gcm::aead::Error| -> TrustformersError {
            TrustformersError::runtime_error(format!(
                "decryption failed: wrong key, or the ciphertext was corrupted or tampered with \
                 ({e})"
            ))
        };

        match self.config.algorithm {
            EncryptionAlgorithm::AES256GCM => {
                let cipher = Aes256Gcm::new_from_slice(key).map_err(|e| {
                    TrustformersError::runtime_error(format!(
                        "failed to initialise AES-256-GCM cipher: {e}"
                    ))
                })?;
                let nonce = aes_gcm::Nonce::try_from(nonce_bytes).map_err(|_| {
                    TrustformersError::runtime_error("malformed AES-256-GCM nonce".to_string())
                })?;
                cipher.decrypt(&nonce, ciphertext).map_err(auth_fail)
            },
            EncryptionAlgorithm::AES128GCM => {
                let cipher = Aes128Gcm::new_from_slice(&key[..16]).map_err(|e| {
                    TrustformersError::runtime_error(format!(
                        "failed to initialise AES-128-GCM cipher: {e}"
                    ))
                })?;
                let nonce = aes_gcm::Nonce::try_from(nonce_bytes).map_err(|_| {
                    TrustformersError::runtime_error("malformed AES-128-GCM nonce".to_string())
                })?;
                cipher.decrypt(&nonce, ciphertext).map_err(auth_fail)
            },
            EncryptionAlgorithm::ChaCha20Poly1305 => {
                let cipher = ChaCha20Poly1305::new_from_slice(key).map_err(|e| {
                    TrustformersError::runtime_error(format!(
                        "failed to initialise ChaCha20-Poly1305 cipher: {e}"
                    ))
                })?;
                let nonce = chacha20poly1305::Nonce::try_from(nonce_bytes).map_err(|_| {
                    TrustformersError::runtime_error(
                        "malformed ChaCha20-Poly1305 nonce".to_string(),
                    )
                })?;
                cipher.decrypt(&nonce, ciphertext).map_err(auth_fail)
            },
        }
    }
}

impl ModelCacheManager {
    fn new(config: PerformanceConfig) -> Self {
        Self {
            cache: HashMap::new(),
            cache_size_bytes: 0,
            config,
        }
    }

    fn preload_model(&mut self, _model_info: &ModelInfo) -> Result<()> {
        // Implementation would preload model into cache
        Ok(())
    }

    fn get_cached_model(&mut self, model_id: &str) -> Result<Option<Vec<u8>>> {
        if let Some(cached) = self.cache.get_mut(model_id) {
            cached.last_accessed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            cached.access_count += 1;
            Ok(Some(cached.data.clone()))
        } else {
            Ok(None)
        }
    }

    fn cache_model(&mut self, model_id: &str, data: &[u8]) -> Result<()> {
        // Check cache size limits
        if self.cache_size_bytes + data.len() as u64
            > (self.config.connection_pool_size * 1024 * 1024) as u64
        {
            self.evict_lru_models()?;
        }

        let cached_model = CachedModel {
            model_info: ModelInfo {
                id: model_id.to_string(),
                name: "cached".to_string(),
                version: "1.0".to_string(),
                model_type: ModelType::Custom,
                file_path: PathBuf::new(),
                size_bytes: data.len() as u64,
                metadata: ModelMetadata {
                    description: "Cached model".to_string(),
                    architecture: "Unknown".to_string(),
                    dataset: None,
                    accuracy: None,
                    latency_ms: None,
                    memory_mb: None,
                    tags: vec![],
                },
                permissions: ModelPermissions {
                    public_access: false,
                    allowed_packages: vec![],
                    required_permissions: vec![],
                    access_level: AccessLevel::Private,
                },
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                modified_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            },
            data: data.to_vec(),
            last_accessed: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            access_count: 1,
        };

        self.cache_size_bytes += data.len() as u64;
        self.cache.insert(model_id.to_string(), cached_model);

        Ok(())
    }

    fn invalidate_cache(&mut self, model_id: &str) -> Result<()> {
        if let Some(cached) = self.cache.remove(model_id) {
            self.cache_size_bytes -= cached.data.len() as u64;
        }
        Ok(())
    }

    fn remove_from_cache(&mut self, model_id: &str) -> Result<()> {
        self.invalidate_cache(model_id)
    }

    fn cleanup_expired_entries(&mut self) -> Result<()> {
        // Remove entries that haven't been accessed in a while
        let cutoff_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            - 3600; // 1 hour

        let to_remove: Vec<String> = self
            .cache
            .iter()
            .filter(|(_, cached)| cached.last_accessed < cutoff_time)
            .map(|(id, _)| id.clone())
            .collect();

        for id in to_remove {
            self.invalidate_cache(&id)?;
        }

        Ok(())
    }

    fn evict_lru_models(&mut self) -> Result<()> {
        // Find least recently used model
        if let Some((lru_id, _)) = self
            .cache
            .iter()
            .min_by_key(|(_, cached)| cached.last_accessed)
            .map(|(id, cached)| (id.clone(), cached))
        {
            self.invalidate_cache(&lru_id)?;
        }
        Ok(())
    }
}

impl UriMatcher {
    fn new(authority: &str) -> Self {
        let mut patterns = HashMap::new();

        // Add standard Content Provider URI patterns
        patterns.insert(
            format!("content://{}/models", authority),
            UriPattern {
                pattern: "/models".to_string(),
                operation: Operation::Query,
                parameters: vec![],
            },
        );

        patterns.insert(
            format!("content://{}/models/#", authority),
            UriPattern {
                pattern: "/models/#".to_string(),
                operation: Operation::Download,
                parameters: vec!["model_id".to_string()],
            },
        );

        Self { patterns }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_provider_creation() {
        let config = ContentProviderConfig::default();
        let provider = AndroidModelContentProvider::new(config);
        assert!(provider.is_ok());
    }

    #[test]
    fn test_model_info_validation() {
        let config = ContentProviderConfig::default();
        let provider = AndroidModelContentProvider::new(config).expect("Operation failed");

        let invalid_model = ModelInfo {
            id: "".to_string(), // Invalid: empty ID
            name: "test".to_string(),
            version: "1.0".to_string(),
            model_type: ModelType::Transformer,
            file_path: PathBuf::from("/nonexistent/path"),
            size_bytes: 1000,
            metadata: ModelMetadata {
                description: "Test model".to_string(),
                architecture: "Test".to_string(),
                dataset: None,
                accuracy: None,
                latency_ms: None,
                memory_mb: None,
                tags: vec![],
            },
            permissions: ModelPermissions {
                public_access: true,
                allowed_packages: vec![],
                required_permissions: vec![],
                access_level: AccessLevel::Public,
            },
            created_at: 0,
            modified_at: 0,
        };

        assert!(provider.validate_model_info(&invalid_model).is_err());
    }

    #[test]
    fn test_query_params() {
        let params = QueryParams {
            model_type: Some(ModelType::Transformer),
            name_filter: Some("bert".to_string()),
            version_filter: None,
            tags: vec!["nlp".to_string()],
            limit: Some(10),
            offset: Some(0),
            sort_by: Some(SortOrder::Name),
        };

        assert_eq!(
            params.model_type.expect("Operation failed"),
            ModelType::Transformer
        );
        assert_eq!(params.tags.len(), 1);
    }

    #[test]
    fn test_content_provider_config_default() {
        let config = ContentProviderConfig::default();
        assert!(!config.authority.is_empty());
        assert!(config.enable_sharing);
        assert!(config.max_cached_models > 0);
        assert!(config.cache_size_limit_mb > 0);
    }

    #[test]
    fn test_security_config_default() {
        let config = SecurityConfig::default();
        assert!(config.require_signature_verification);
        assert!(config.per_model_permissions);
    }

    #[test]
    fn test_encryption_config_default() {
        let config = EncryptionConfig::default();
        assert!(config.encrypt_at_rest);
        assert!(config.encrypt_in_transit);
    }

    #[test]
    fn test_key_management_config_default() {
        let config = KeyManagementConfig::default();
        assert!(config.use_android_keystore);
        assert!(config.key_rotation_days > 0);
    }

    #[test]
    fn test_performance_config_default() {
        let config = PerformanceConfig::default();
        assert!(config.background_loading);
        assert!(config.worker_threads > 0);
        assert!(config.connection_pool_size > 0);
    }

    #[test]
    fn test_encryption_algorithm_variants() {
        let algos = vec![
            EncryptionAlgorithm::AES256GCM,
            EncryptionAlgorithm::ChaCha20Poly1305,
            EncryptionAlgorithm::AES128GCM,
        ];
        assert_eq!(algos.len(), 3);
    }

    #[test]
    fn test_compression_algorithm_variants() {
        let algos = vec![
            CompressionAlgorithm::Gzip,
            CompressionAlgorithm::Lz4,
            CompressionAlgorithm::Zstd,
        ];
        assert_eq!(algos.len(), 3);
    }

    #[test]
    fn test_model_type_variants() {
        let types = vec![
            ModelType::Transformer,
            ModelType::CNN,
            ModelType::RNN,
            ModelType::Custom,
        ];
        assert_eq!(types.len(), 4);
    }

    #[test]
    fn test_access_level_variants() {
        let levels = vec![
            AccessLevel::Public,
            AccessLevel::Private,
            AccessLevel::Restricted,
            AccessLevel::System,
        ];
        assert_eq!(levels.len(), 4);
    }

    #[test]
    fn test_sort_order_variants() {
        let orders = vec![SortOrder::Name, SortOrder::Size, SortOrder::CreatedAt];
        assert_eq!(orders.len(), 3);
    }

    #[test]
    fn test_model_info_with_valid_id() {
        let config = ContentProviderConfig::default();
        let provider = AndroidModelContentProvider::new(config).expect("Operation failed");

        // `validate_model_info` checks `file_path.exists()` for real (see its
        // own doc comment), so this must point at an actual file rather than
        // the fictitious `/models/test.bin` this test previously used --
        // that path never existed on the test host and this assertion would
        // fail against a real (non-placeholder) `validate_model_info`.
        let dir = std::env::temp_dir();
        let file_path = dir.join(format!(
            "trustformers_mobile_valid_model_test_{}.bin",
            std::process::id()
        ));
        std::fs::write(&file_path, b"fake model bytes").expect("write temp model file");

        let model = ModelInfo {
            id: "valid_model_id".to_string(),
            name: "Test Model".to_string(),
            version: "1.0".to_string(),
            model_type: ModelType::Transformer,
            file_path: file_path.clone(),
            size_bytes: 10000,
            metadata: ModelMetadata {
                description: "Test".to_string(),
                architecture: "Transformer".to_string(),
                dataset: None,
                accuracy: Some(0.95),
                latency_ms: Some(10.0),
                memory_mb: Some(128.0),
                tags: vec!["nlp".to_string()],
            },
            permissions: ModelPermissions {
                public_access: true,
                allowed_packages: vec![],
                required_permissions: vec![],
                access_level: AccessLevel::Public,
            },
            created_at: 1000,
            modified_at: 1000,
        };

        assert!(provider.validate_model_info(&model).is_ok());
        let _ = std::fs::remove_file(&file_path);
    }

    #[test]
    fn test_query_params_with_no_filters() {
        let params = QueryParams {
            model_type: None,
            name_filter: None,
            version_filter: None,
            tags: vec![],
            limit: None,
            offset: None,
            sort_by: None,
        };
        assert!(params.model_type.is_none());
        assert!(params.tags.is_empty());
    }

    #[test]
    fn test_query_params_with_limit() {
        let params = QueryParams {
            model_type: None,
            name_filter: None,
            version_filter: None,
            tags: vec![],
            limit: Some(50),
            offset: Some(10),
            sort_by: Some(SortOrder::CreatedAt),
        };
        assert_eq!(params.limit, Some(50));
        assert_eq!(params.offset, Some(10));
    }

    #[test]
    fn test_model_metadata_optional_fields() {
        let metadata = ModelMetadata {
            description: "Minimal model".to_string(),
            architecture: "CNN".to_string(),
            dataset: None,
            accuracy: None,
            latency_ms: None,
            memory_mb: None,
            tags: vec![],
        };
        assert!(metadata.dataset.is_none());
        assert!(metadata.accuracy.is_none());
    }

    #[test]
    fn test_model_permissions_private() {
        let perms = ModelPermissions {
            public_access: false,
            allowed_packages: vec!["com.example.app".to_string()],
            required_permissions: vec!["READ_MODELS".to_string()],
            access_level: AccessLevel::Private,
        };
        assert!(!perms.public_access);
        assert_eq!(perms.allowed_packages.len(), 1);
    }

    #[test]
    fn test_model_stream_progress() {
        // Regression test for `ModelStream::read_chunk`: it used to be a
        // `// Placeholder implementation` that returned `vec![0u8; ...]`
        // regardless of the file's real contents. Write real,
        // non-all-zero bytes to a temp file and verify each streamed
        // chunk matches them exactly (which a zero-fill fake would fail),
        // and that `progress()` advances with `current_position`.
        let dir = std::env::temp_dir();
        let file_path = dir.join(format!(
            "trustformers_mobile_stream_test_{}.bin",
            std::process::id()
        ));
        let contents: Vec<u8> = (0..5u8).map(|b| b * 37 + 1).collect(); // non-zero, non-trivial bytes
        std::fs::write(&file_path, &contents).expect("write temp model file");

        let model_info = ModelInfo {
            id: "stream_test".to_string(),
            name: "Stream Test Model".to_string(),
            version: "1.0".to_string(),
            model_type: ModelType::Transformer,
            file_path: file_path.clone(),
            size_bytes: contents.len() as u64,
            metadata: ModelMetadata {
                description: "Test".to_string(),
                architecture: "Transformer".to_string(),
                dataset: None,
                accuracy: None,
                latency_ms: None,
                memory_mb: None,
                tags: vec![],
            },
            permissions: ModelPermissions {
                public_access: true,
                allowed_packages: vec![],
                required_permissions: vec![],
                access_level: AccessLevel::Public,
            },
            created_at: 0,
            modified_at: 0,
        };

        let mut stream = ModelStream::new(model_info, CompressionAlgorithm::Gzip)
            .expect("ModelStream::new should succeed for a real file");
        stream.chunk_size = 2;
        assert_eq!(stream.progress(), 0.0);

        let mut read_back = Vec::new();
        while let Some(chunk) = stream.read_chunk().expect("read_chunk should succeed") {
            read_back.extend_from_slice(&chunk);
        }
        assert_eq!(
            read_back, contents,
            "streamed bytes must match the real file contents"
        );
        assert_eq!(stream.progress(), 1.0);

        let _ = std::fs::remove_file(&file_path);
    }

    #[test]
    fn test_usage_stats_default() {
        let stats = UsageStats::new();
        assert_eq!(stats.total_accesses, 0);
        assert_eq!(stats.successful_accesses, 0);
        assert_eq!(stats.failed_accesses, 0);
    }
}
