/*!
# iOS iCloud Model Sync Module

This module provides iCloud model synchronization capabilities for iOS devices,
enabling seamless model sharing and backup across devices using CloudKit.

## Features

- **Automatic Model Sync**: Sync models across devices signed into the same iCloud account
- **Incremental Updates**: Only sync model differences to save bandwidth
- **Conflict Resolution**: Handle conflicts when models are updated on multiple devices
- **Privacy-First**: Models are encrypted before upload to iCloud
- **Background Sync**: Sync models in the background when network is available
- **Storage Management**: Automatically manage iCloud storage usage

## Usage

```rust
use trustformers_mobile::ios_icloud::{iCloudModelSync, SyncConfig};

let config = SyncConfig::default();
let mut sync = iCloudModelSync::new(config)?;
sync.enable_auto_sync(true)?;
```
*/

use crate::MobileConfig;
use trustformers_core::errors::model_not_found;
use trustformers_core::TrustformersError;

// Type aliases for compatibility
pub type MobileError = TrustformersError;
pub type MobileResult<T> = Result<T, TrustformersError>;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// Real cryptographic primitives for `encrypt_data`/`decrypt_data` (AES-256-GCM,
// an AEAD -- confidentiality *and* tamper detection, unlike the repeating-key
// XOR this module used to call "AES-256") and `derive_key_from_password`
// (PBKDF2-HMAC-SHA256, a real key-stretching KDF). `aead::Generate` sources the
// GCM nonce from the operating system CSPRNG (the `getrandom` cargo feature of
// `aes-gcm`, already enabled -- see `Cargo.toml`), not from a clock-seeded LCG.
use aes_gcm::aead::{Aead, Generate, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use pbkdf2::pbkdf2_hmac;
use sha2::Sha256;

/// Configuration for iCloud model synchronization
// `iCloud*` (lowercase `i`) deliberately mirrors Apple's own `iCloud`/`iOS`
// capitalization convention rather than Rust's `UpperCamelCase` type-name
// style; this is intentional public API naming, not an oversight.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct iCloudSyncConfig {
    /// Enable automatic synchronization
    pub auto_sync_enabled: bool,
    /// Maximum model size to sync (in MB)
    pub max_model_size_mb: u64,
    /// Sync interval in seconds
    pub sync_interval_seconds: u64,
    /// Enable compression for model uploads
    pub compression_enabled: bool,
    /// Encryption key for model data (optional - uses device keychain if not provided)
    pub encryption_key: Option<Vec<u8>>,
    /// Container identifier for CloudKit
    pub container_id: String,
    /// Database scope (private, public, shared)
    pub database_scope: DatabaseScope,
    /// Enable conflict resolution
    pub conflict_resolution_enabled: bool,
    /// Maximum retries for sync operations
    pub max_retry_attempts: u32,
    /// Timeout for sync operations in seconds
    pub operation_timeout_seconds: u64,
    /// Enable detailed logging
    pub verbose_logging: bool,
}

impl Default for iCloudSyncConfig {
    fn default() -> Self {
        Self {
            auto_sync_enabled: true,
            max_model_size_mb: 500,     // 500MB limit
            sync_interval_seconds: 300, // 5 minutes
            compression_enabled: true,
            encryption_key: None, // Use device keychain
            container_id: "iCloud.com.trustformers.models".to_string(),
            database_scope: DatabaseScope::Private,
            conflict_resolution_enabled: true,
            max_retry_attempts: 3,
            operation_timeout_seconds: 60,
            verbose_logging: false,
        }
    }
}

/// CloudKit database scope
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatabaseScope {
    /// Private database (user's personal data)
    Private,
    /// Public database (shared data)
    Public,
    /// Shared database (collaborative data)
    Shared,
}

/// Model synchronization status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncStatus {
    /// Model is not synced
    NotSynced,
    /// Model is being uploaded
    Uploading,
    /// Model is being downloaded
    Downloading,
    /// Model is up to date
    Synced,
    /// Sync failed - needs retry
    Failed,
    /// Conflict detected - needs resolution
    Conflict,
    /// Model is queued for sync
    Queued,
}

/// Model metadata for iCloud sync
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Unique model identifier
    pub model_id: String,
    /// Model name
    pub model_name: String,
    /// Model version
    pub version: String,
    /// Model size in bytes
    pub size_bytes: u64,
    /// Last modified timestamp
    pub last_modified: SystemTime,
    /// Checksum of model data
    pub checksum: String,
    /// Device that last modified the model
    pub last_modified_device: String,
    /// Additional metadata
    pub custom_metadata: HashMap<String, String>,
    /// Sync status
    pub sync_status: SyncStatus,
    /// Local file path
    pub local_path: Option<PathBuf>,
    /// iCloud record ID
    pub cloud_record_id: Option<String>,
}

/// Sync operation result
#[derive(Debug, Clone)]
pub struct SyncResult {
    /// Number of models successfully synced
    pub synced_count: usize,
    /// Number of models that failed to sync
    pub failed_count: usize,
    /// Number of conflicts detected
    pub conflict_count: usize,
    /// Total bytes transferred
    pub bytes_transferred: u64,
    /// Duration of sync operation
    pub duration: Duration,
    /// Detailed operation results
    pub operation_results: Vec<ModelSyncResult>,
}

/// Individual model sync result
#[derive(Debug, Clone)]
pub struct ModelSyncResult {
    /// Model ID
    pub model_id: String,
    /// Sync operation type
    pub operation: SyncOperation,
    /// Success status
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Bytes transferred for this model
    pub bytes_transferred: u64,
}

/// Type of sync operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncOperation {
    Upload,
    Download,
    Update,
    Delete,
    ConflictResolution,
}

/// Conflict resolution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictResolution {
    /// Use the most recent version
    UseNewest,
    /// Use the local version
    UseLocal,
    /// Use the remote version
    UseRemote,
    /// Create a backup of the conflicting version
    CreateBackup,
    /// Manual resolution required
    Manual,
}

/// Main iCloud model synchronization manager
#[allow(non_camel_case_types)] // see `iCloudSyncConfig`'s naming note above
pub struct iCloudModelSync {
    config: iCloudSyncConfig,
    cloud_manager: Arc<Mutex<CloudKitManager>>,
    local_models: Arc<Mutex<HashMap<String, ModelMetadata>>>,
    sync_queue: Arc<Mutex<Vec<SyncTask>>>,
    background_sync_active: Arc<Mutex<bool>>,
    statistics: Arc<Mutex<SyncStatistics>>,
}

impl iCloudModelSync {
    /// Create a new iCloud model sync manager
    pub fn new(config: iCloudSyncConfig) -> MobileResult<Self> {
        let cloud_manager = CloudKitManager::new(&config)?;

        Ok(Self {
            config,
            cloud_manager: Arc::new(Mutex::new(cloud_manager)),
            local_models: Arc::new(Mutex::new(HashMap::new())),
            sync_queue: Arc::new(Mutex::new(Vec::new())),
            background_sync_active: Arc::new(Mutex::new(false)),
            statistics: Arc::new(Mutex::new(SyncStatistics::new())),
        })
    }

    /// Enable or disable automatic synchronization
    pub fn enable_auto_sync(&mut self, enabled: bool) -> MobileResult<()> {
        self.config.auto_sync_enabled = enabled;

        if enabled {
            self.start_background_sync()?;
        } else {
            self.stop_background_sync()?;
        }

        Ok(())
    }

    /// Register a model for synchronization
    pub fn register_model(
        &mut self,
        model_path: &Path,
        metadata: ModelMetadata,
    ) -> MobileResult<()> {
        // Validate model file exists
        if !model_path.exists() {
            return Err(TrustformersError::io_error(format!(
                "File not found: {}",
                model_path.to_string_lossy()
            )));
        }

        // Calculate checksum
        let checksum = self.calculate_file_checksum(model_path)?;

        // Update metadata
        let mut updated_metadata = metadata;
        updated_metadata.local_path = Some(model_path.to_path_buf());
        updated_metadata.checksum = checksum;
        updated_metadata.size_bytes = std::fs::metadata(model_path)
            .map_err(|e| TrustformersError::io_error(e.to_string()))?
            .len();
        updated_metadata.sync_status = SyncStatus::NotSynced;

        // Store in local registry
        {
            let mut local_models = self.local_models.lock().unwrap_or_else(|p| p.into_inner());
            local_models.insert(updated_metadata.model_id.clone(), updated_metadata.clone());
        }

        // Queue for sync if auto-sync is enabled
        if self.config.auto_sync_enabled {
            self.queue_model_for_sync(&updated_metadata.model_id, SyncOperation::Upload)?;
        }

        // Update statistics
        {
            let mut stats = self.statistics.lock().unwrap_or_else(|p| p.into_inner());
            stats.total_models_registered += 1;
        }

        Ok(())
    }

    /// Manually sync a specific model
    pub fn sync_model(&mut self, model_id: &str) -> MobileResult<ModelSyncResult> {
        let metadata = {
            let local_models = self.local_models.lock().unwrap_or_else(|p| p.into_inner());
            local_models
                .get(model_id)
                .cloned()
                .ok_or_else(|| model_not_found(model_id.to_string()))?
        };

        self.perform_model_sync(&metadata)
    }

    /// Sync all registered models
    pub fn sync_all_models(&mut self) -> MobileResult<SyncResult> {
        let start_time = std::time::Instant::now();
        let mut results = Vec::new();
        let mut synced_count = 0;
        let mut failed_count = 0;
        let mut conflict_count = 0;
        let mut bytes_transferred = 0;

        let model_ids: Vec<String> = {
            let local_models = self.local_models.lock().unwrap_or_else(|p| p.into_inner());
            local_models.keys().cloned().collect()
        };

        for model_id in model_ids {
            match self.sync_model(&model_id) {
                Ok(result) => {
                    if result.success {
                        synced_count += 1;
                    } else {
                        failed_count += 1;
                    }
                    bytes_transferred += result.bytes_transferred;
                    results.push(result);
                },
                Err(_) => {
                    failed_count += 1;
                    results.push(ModelSyncResult {
                        model_id,
                        operation: SyncOperation::Upload,
                        success: false,
                        error_message: Some("Sync failed".to_string()),
                        bytes_transferred: 0,
                    });
                },
            }
        }

        // Check for conflicts
        conflict_count = results
            .iter()
            .filter(|r| matches!(r.operation, SyncOperation::ConflictResolution))
            .count();

        Ok(SyncResult {
            synced_count,
            failed_count,
            conflict_count,
            bytes_transferred,
            duration: start_time.elapsed(),
            operation_results: results,
        })
    }

    /// Download models from iCloud
    pub fn download_available_models(&mut self) -> MobileResult<Vec<ModelMetadata>> {
        let cloud_manager = self.cloud_manager.lock().unwrap_or_else(|p| p.into_inner());
        cloud_manager.fetch_available_models()
    }

    /// Check for model updates
    pub fn check_for_updates(&mut self) -> MobileResult<Vec<String>> {
        let mut updated_models = Vec::new();

        let cloud_manager = self.cloud_manager.lock().unwrap_or_else(|p| p.into_inner());
        let remote_models = cloud_manager.fetch_model_list()?;

        let local_models = self.local_models.lock().unwrap_or_else(|p| p.into_inner());

        for remote_model in remote_models {
            if let Some(local_model) = local_models.get(&remote_model.model_id) {
                if remote_model.last_modified > local_model.last_modified {
                    updated_models.push(remote_model.model_id);
                }
            }
        }

        Ok(updated_models)
    }

    /// Resolve a model conflict
    pub fn resolve_conflict(
        &mut self,
        model_id: &str,
        resolution: ConflictResolution,
    ) -> MobileResult<()> {
        let mut metadata = {
            let local_models = self.local_models.lock().unwrap_or_else(|p| p.into_inner());
            local_models
                .get(model_id)
                .cloned()
                .ok_or_else(|| model_not_found(model_id.to_string()))?
        };

        if metadata.sync_status != SyncStatus::Conflict {
            return Err(TrustformersError::invalid_operation(
                "Model is not in conflict state".to_string(),
            ));
        }

        match resolution {
            ConflictResolution::UseNewest => {
                // Compare timestamps and use the newer version. The
                // `cloud_manager` lock is dropped (end of this inner block)
                // *before* `self.download_model`/`self.upload_model` below,
                // each of which acquires `self.cloud_manager` itself --
                // same non-reentrant-`Mutex` deadlock hazard documented on
                // `perform_model_sync`.
                let remote_metadata = {
                    let cloud_manager =
                        self.cloud_manager.lock().unwrap_or_else(|p| p.into_inner());
                    cloud_manager.fetch_model_metadata(model_id)?
                };

                if remote_metadata.last_modified > metadata.last_modified {
                    self.download_model(model_id)?;
                } else {
                    self.upload_model(model_id)?;
                }
            },
            ConflictResolution::UseLocal => {
                self.upload_model(model_id)?;
            },
            ConflictResolution::UseRemote => {
                self.download_model(model_id)?;
            },
            ConflictResolution::CreateBackup => {
                // Create a backup of the local version first
                let backup_id = format!(
                    "{}_backup_{}",
                    model_id,
                    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
                );

                let mut backup_metadata = metadata.clone();
                backup_metadata.model_id = backup_id.clone();
                backup_metadata.model_name = format!("{} (Backup)", metadata.model_name);

                // Register backup and upload
                {
                    let mut local_models =
                        self.local_models.lock().unwrap_or_else(|p| p.into_inner());
                    local_models.insert(backup_id, backup_metadata);
                }

                // Then download the remote version
                self.download_model(model_id)?;
            },
            ConflictResolution::Manual => {
                // Mark for manual resolution
                metadata.sync_status = SyncStatus::Failed;
                let mut local_models = self.local_models.lock().unwrap_or_else(|p| p.into_inner());
                local_models.insert(model_id.to_string(), metadata);
                return Ok(());
            },
        }

        // Update status
        metadata.sync_status = SyncStatus::Synced;
        let mut local_models = self.local_models.lock().unwrap_or_else(|p| p.into_inner());
        local_models.insert(model_id.to_string(), metadata);

        Ok(())
    }

    /// Get sync statistics
    pub fn get_sync_statistics(&self) -> SyncStatistics {
        let stats = self.statistics.lock().unwrap_or_else(|p| p.into_inner());
        stats.clone()
    }

    /// Get list of registered models
    pub fn get_registered_models(&self) -> Vec<ModelMetadata> {
        let local_models = self.local_models.lock().unwrap_or_else(|p| p.into_inner());
        local_models.values().cloned().collect()
    }

    /// Remove a model from sync (local and remote)
    pub fn remove_model(&mut self, model_id: &str, delete_remote: bool) -> MobileResult<()> {
        // Remove from local registry
        {
            let mut local_models = self.local_models.lock().unwrap_or_else(|p| p.into_inner());
            local_models.remove(model_id);
        }

        // Remove from remote if requested
        if delete_remote {
            let cloud_manager = self.cloud_manager.lock().unwrap_or_else(|p| p.into_inner());
            cloud_manager.delete_model(model_id)?;
        }

        Ok(())
    }

    /// Private helper methods
    ///
    /// # Locking
    ///
    /// The `cloud_manager` lock is deliberately scoped to just the
    /// existence/metadata check below, *before* any call to
    /// `self.handle_conflict` / `self.upload_model` -- each of which
    /// acquires `self.cloud_manager` itself. `std::sync::Mutex` is not
    /// reentrant, so a previous revision that held the guard across those
    /// calls deadlocked the calling thread forever on the very first sync
    /// of any model that either does not exist remotely yet (the common
    /// case: `remote_exists` is `false`, so the `else` branch's
    /// `self.upload_model(...)` ran while `cloud_manager` was still locked)
    /// or has diverged from the remote copy.
    fn perform_model_sync(&mut self, metadata: &ModelMetadata) -> MobileResult<ModelSyncResult> {
        let start_time = std::time::Instant::now();

        let (remote_exists, remote_metadata) = {
            let cloud_manager = self.cloud_manager.lock().unwrap_or_else(|p| p.into_inner());
            let remote_exists = cloud_manager.model_exists(&metadata.model_id)?;
            let remote_metadata = if remote_exists {
                Some(cloud_manager.fetch_model_metadata(&metadata.model_id)?)
            } else {
                None
            };
            (remote_exists, remote_metadata)
            // `cloud_manager` (the `MutexGuard`) is dropped here, at the end
            // of this block -- before any nested `self.*` call below can
            // try to re-acquire it.
        };

        let operation = if remote_exists {
            let remote_metadata = remote_metadata.ok_or_else(|| {
                TrustformersError::runtime_error(
                    "internal error: remote_exists was true but no remote metadata was fetched"
                        .to_string(),
                )
            })?;

            if remote_metadata.last_modified > metadata.last_modified
                && metadata.last_modified > UNIX_EPOCH + Duration::from_secs(1)
            {
                // Conflict detected
                self.handle_conflict(&metadata.model_id)?;
                SyncOperation::ConflictResolution
            } else if remote_metadata.checksum != metadata.checksum {
                // Upload newer version
                self.upload_model(&metadata.model_id)?;
                SyncOperation::Upload
            } else {
                // Already in sync
                SyncOperation::Update
            }
        } else {
            // Upload new model
            self.upload_model(&metadata.model_id)?;
            SyncOperation::Upload
        };

        let bytes_transferred = metadata.size_bytes;

        // Update statistics
        {
            let mut stats = self.statistics.lock().unwrap_or_else(|p| p.into_inner());
            stats.total_sync_operations += 1;
            stats.total_bytes_transferred += bytes_transferred;
            stats.last_sync_time = SystemTime::now();
        }

        Ok(ModelSyncResult {
            model_id: metadata.model_id.clone(),
            operation,
            success: true,
            error_message: None,
            bytes_transferred,
        })
    }

    fn upload_model(&self, model_id: &str) -> MobileResult<()> {
        let metadata = {
            let local_models = self.local_models.lock().unwrap_or_else(|p| p.into_inner());
            local_models
                .get(model_id)
                .cloned()
                .ok_or_else(|| model_not_found(model_id.to_string()))?
        };

        // `.clone()` rather than moving `metadata.local_path` out: `metadata`
        // as a whole is still needed below, to pass to
        // `cloud_manager.upload_model`.
        let local_path = metadata.local_path.clone().ok_or_else(|| {
            TrustformersError::invalid_state("Model has no local path".to_string())
        })?;

        // Compress and encrypt if enabled
        let processed_data = self.process_model_for_upload(&local_path)?;

        let cloud_manager = self.cloud_manager.lock().unwrap_or_else(|p| p.into_inner());
        cloud_manager.upload_model(&metadata, &processed_data)?;

        // Update local status
        self.update_model_status(model_id, SyncStatus::Synced)?;

        Ok(())
    }

    fn download_model(&self, model_id: &str) -> MobileResult<()> {
        let cloud_manager = self.cloud_manager.lock().unwrap_or_else(|p| p.into_inner());
        let (metadata, model_data) = cloud_manager.download_model(model_id)?;

        // Process downloaded data (decrypt, decompress)
        let processed_data = self.process_downloaded_model(&model_data)?;

        // Save to local storage
        let local_path = self.get_local_model_path(&metadata.model_id);
        std::fs::write(&local_path, processed_data)
            .map_err(|e| TrustformersError::io_error(e.to_string()))?;

        // Update local registry
        let mut updated_metadata = metadata;
        updated_metadata.local_path = Some(local_path);
        updated_metadata.sync_status = SyncStatus::Synced;

        {
            let mut local_models = self.local_models.lock().unwrap_or_else(|p| p.into_inner());
            local_models.insert(model_id.to_string(), updated_metadata);
        }

        Ok(())
    }

    fn handle_conflict(&self, model_id: &str) -> MobileResult<()> {
        self.update_model_status(model_id, SyncStatus::Conflict)?;

        if self.config.conflict_resolution_enabled {
            // Auto-resolve using newest version
            // This would be implemented based on the default resolution strategy
        }

        Ok(())
    }

    fn queue_model_for_sync(&self, model_id: &str, operation: SyncOperation) -> MobileResult<()> {
        let task = SyncTask {
            model_id: model_id.to_string(),
            operation,
            retry_count: 0,
            scheduled_time: SystemTime::now(),
        };

        let mut sync_queue = self.sync_queue.lock().unwrap_or_else(|p| p.into_inner());
        sync_queue.push(task);

        Ok(())
    }

    fn start_background_sync(&self) -> MobileResult<()> {
        {
            let mut active = self.background_sync_active.lock().unwrap_or_else(|p| p.into_inner());
            *active = true;
        }

        // This would spawn a background thread for periodic sync
        // For now, this is a placeholder
        Ok(())
    }

    fn stop_background_sync(&self) -> MobileResult<()> {
        let mut active = self.background_sync_active.lock().unwrap_or_else(|p| p.into_inner());
        *active = false;
        Ok(())
    }

    fn update_model_status(&self, model_id: &str, status: SyncStatus) -> MobileResult<()> {
        let mut local_models = self.local_models.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(metadata) = local_models.get_mut(model_id) {
            metadata.sync_status = status;
        }
        Ok(())
    }

    fn calculate_file_checksum(&self, path: &Path) -> MobileResult<String> {
        use std::io::Read;

        let mut file =
            std::fs::File::open(path).map_err(|e| TrustformersError::io_error(e.to_string()))?;

        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 8192];

        loop {
            let bytes_read =
                file.read(&mut buffer).map_err(|e| TrustformersError::io_error(e.to_string()))?;

            if bytes_read == 0 {
                break;
            }

            hasher.update(&buffer[..bytes_read]);
        }

        let hash = hasher.finalize();
        Ok(hex::encode(hash))
    }

    fn process_model_for_upload(&self, path: &Path) -> MobileResult<Vec<u8>> {
        let mut data =
            std::fs::read(path).map_err(|e| TrustformersError::io_error(e.to_string()))?;

        // Compress if enabled
        if self.config.compression_enabled {
            data = self.compress_data(&data)?;
        }

        // Encrypt if key is available
        if let Some(key) = &self.config.encryption_key {
            data = self.encrypt_data(&data, key)?;
        }

        Ok(data)
    }

    fn process_downloaded_model(&self, data: &[u8]) -> MobileResult<Vec<u8>> {
        let mut processed = data.to_vec();

        // Decrypt if key is available
        if let Some(key) = &self.config.encryption_key {
            processed = self.decrypt_data(&processed, key)?;
        }

        // Decompress if enabled
        if self.config.compression_enabled {
            processed = self.decompress_data(&processed)?;
        }

        Ok(processed)
    }

    /// Real zstd compression via `oxiarc-zstd` (this workspace's pure-Rust
    /// replacement for `flate2`/`zstd`-the-C-binding, per the COOLJAPAN
    /// dependency policy -- the same crate and call pattern already proven
    /// in `trustformers_core::cache::inference_cache`).
    ///
    /// The previous implementation hand-rolled run-length encoding and, when
    /// RLE did not shrink the input, silently fell back to returning the
    /// *raw* bytes with no marker distinguishing that case from a real RLE
    /// stream -- `decompress_data` could not tell them apart and corrupted
    /// every such payload (see that function's doc comment). A real codec
    /// with its own self-describing frame header has no equivalent failure
    /// mode: the output of [`Self::compress_data`] is always a zstd frame,
    /// unconditionally, and [`Self::decompress_data`] always decodes one.
    fn compress_data(&self, data: &[u8]) -> MobileResult<Vec<u8>> {
        use std::io::Write;

        if data.is_empty() {
            // A zero-byte payload compresses fine through the real encoder
            // too, but short-circuiting avoids emitting a frame for
            // literally nothing to decode later.
            return Ok(Vec::new());
        }

        let mut encoder = oxiarc_zstd::ZstdStreamEncoder::new(Vec::new(), 3);
        encoder.write_all(data).map_err(|e| {
            TrustformersError::runtime_error(format!("zstd compression failed: {e}"))
        })?;
        encoder
            .finish()
            .map_err(|e| TrustformersError::runtime_error(format!("zstd compression failed: {e}")))
    }

    /// Real zstd decompression matching [`Self::compress_data`]. See that
    /// method's doc comment for the corruption bug this replaces: the old
    /// RLE decoder guessed "compressed vs. raw" from parity of the byte
    /// length alone, which silently mis-decoded any incompressible
    /// even-length payload (the common case for raw `f32` weight buffers).
    fn decompress_data(&self, data: &[u8]) -> MobileResult<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        oxiarc_zstd::decode_all(data).map_err(|e| {
            TrustformersError::runtime_error(format!("zstd decompression failed: {e}"))
        })
    }

    /// Real AES-256-GCM sealing (RustCrypto's `aes-gcm`, already a workspace
    /// dependency). Output layout is `nonce (12 bytes) || ciphertext+tag
    /// (plaintext.len() + 16 bytes)`, self-describing enough for
    /// [`Self::decrypt_data`] to split back apart with no separate IV
    /// channel needed.
    ///
    /// This replaces a repeating-32-byte-XOR-keystream cipher that was
    /// documented `// Implement AES-256 encryption` while doing nothing of
    /// the sort -- trivially broken by know-plaintext XOR recovery, and
    /// with an IV from a `wrapping_mul` LCG seeded off the wall clock
    /// rather than a CSPRNG. GCM is a real AEAD: tampering with the
    /// ciphertext (or using the wrong key) makes [`Self::decrypt_data`]
    /// fail authentication rather than silently returning garbage
    /// plaintext, which the old XOR scheme could never detect at all.
    fn encrypt_data(&self, data: &[u8], key: &[u8]) -> MobileResult<Vec<u8>> {
        if key.len() != 32 {
            return Err(TrustformersError::invalid_argument(
                "Encryption key must be 32 bytes for AES-256".to_string(),
            ));
        }

        let cipher = Aes256Gcm::new_from_slice(key).map_err(|e| {
            TrustformersError::runtime_error(format!(
                "failed to initialise AES-256-GCM cipher: {e}"
            ))
        })?;
        // Sourced from the OS CSPRNG via `aes-gcm`'s `getrandom` feature
        // (see the `Generate` import above) -- never a seeded PRNG.
        let nonce = Nonce::generate();
        let ciphertext = cipher.encrypt(&nonce, data).map_err(|e| {
            TrustformersError::runtime_error(format!("AES-256-GCM encryption failed: {e}"))
        })?;

        let mut encrypted = Vec::with_capacity(nonce.len() + ciphertext.len());
        encrypted.extend_from_slice(nonce.as_slice());
        encrypted.extend_from_slice(&ciphertext);
        Ok(encrypted)
    }

    /// Real AES-256-GCM opening matching [`Self::encrypt_data`]'s output
    /// layout. Returns an error -- rather than corrupted plaintext -- when
    /// `key` is wrong or `data` has been tampered with, since GCM
    /// authenticates the ciphertext as part of decryption.
    fn decrypt_data(&self, data: &[u8], key: &[u8]) -> MobileResult<Vec<u8>> {
        const NONCE_LEN: usize = 12;

        if key.len() != 32 {
            return Err(TrustformersError::invalid_argument(
                "Decryption key must be 32 bytes for AES-256".to_string(),
            ));
        }
        if data.len() < NONCE_LEN {
            return Err(TrustformersError::invalid_argument(format!(
                "Encrypted data must be at least {NONCE_LEN} bytes (GCM nonce size), got {}",
                data.len()
            )));
        }

        let (nonce_bytes, ciphertext) = data.split_at(NONCE_LEN);
        let nonce = Nonce::try_from(nonce_bytes)
            .map_err(|_| TrustformersError::invalid_argument("malformed GCM nonce".to_string()))?;

        let cipher = Aes256Gcm::new_from_slice(key).map_err(|e| {
            TrustformersError::runtime_error(format!(
                "failed to initialise AES-256-GCM cipher: {e}"
            ))
        })?;
        cipher.decrypt(&nonce, ciphertext).map_err(|_| {
            TrustformersError::runtime_error(
                "AES-256-GCM authentication failed: wrong key, or the ciphertext was corrupted \
                 or tampered with"
                    .to_string(),
            )
        })
    }

    /// Derive a 32-byte AES-256 key from a user password via real
    /// PBKDF2-HMAC-SHA256 (RustCrypto's `pbkdf2`, already a workspace
    /// dependency), for callers that want to populate
    /// [`iCloudSyncConfig::encryption_key`] from a passphrase rather than a
    /// raw key.
    ///
    /// The previous implementation ran a 32-bit multiplicative rolling hash
    /// (`hash.wrapping_mul(31).wrapping_add(byte)`) once per output byte --
    /// about 2^8 bits of effective search space per byte, invertible with a
    /// pocket calculator, and documented `// This is a simplified
    /// implementation - use proper PBKDF2 in production`. `rounds` should
    /// be at least `100_000` for a genuinely slow-to-brute-force key
    /// (`iCloudSyncConfig` does not currently carry a stored iteration
    /// count, so callers choose and remember their own).
    ///
    /// # Errors
    ///
    /// Never fails today (PBKDF2-HMAC-SHA256 has no fallible inputs for any
    /// `salt`/`rounds` this signature can express); returns `Result` so a
    /// future minimum-iteration-count check can be added without breaking
    /// callers.
    pub fn derive_key_from_password(
        password: &str,
        salt: &[u8],
        rounds: u32,
    ) -> MobileResult<Vec<u8>> {
        let mut key = [0u8; 32];
        pbkdf2_hmac::<Sha256>(password.as_bytes(), salt, rounds, &mut key);
        Ok(key.to_vec())
    }

    fn secure_delete(&self, path: &Path) -> MobileResult<()> {
        // Securely delete file by overwriting with random data
        use std::fs::OpenOptions;
        use std::io::Write;

        if !path.exists() {
            return Ok(());
        }

        let metadata = std::fs::metadata(path).map_err(|e| {
            TrustformersError::io_error(format!("Failed to get file metadata: {}", e))
        })?;

        let file_size = metadata.len();

        // Overwrite file with random data multiple times
        for pass in 0..3 {
            let mut file =
                OpenOptions::new().write(true).truncate(true).open(path).map_err(|e| {
                    TrustformersError::io_error(format!(
                        "Failed to open file for secure deletion: {}",
                        e
                    ))
                })?;

            // Write random data
            let pattern = match pass {
                0 => 0x00u8, // First pass: all zeros
                1 => 0xFFu8, // Second pass: all ones
                _ => 0xAAu8, // Third pass: alternating pattern
            };

            let chunk_size = 4096;
            let mut written = 0u64;

            while written < file_size {
                let remaining = std::cmp::min(chunk_size, file_size - written);
                let chunk = vec![pattern; remaining as usize];

                file.write_all(&chunk).map_err(|e| {
                    TrustformersError::io_error(format!(
                        "Failed to write during secure deletion: {}",
                        e
                    ))
                })?;

                written += remaining;
            }

            file.flush().map_err(|e| {
                TrustformersError::io_error(format!(
                    "Failed to flush during secure deletion: {}",
                    e
                ))
            })?;
        }

        // Finally delete the file
        std::fs::remove_file(path)
            .map_err(|e| TrustformersError::io_error(format!("Failed to remove file: {}", e)))?;

        Ok(())
    }

    /// Local on-device path a synced model's bytes are written to /read
    /// from. Previously a hardcoded absolute path
    /// (`/var/mobile/Containers/Data/Application/Documents/models/...`) --
    /// real on a real iOS device's app sandbox, but not a directory that
    /// exists (or should ever be hardcoded into a library) on any other
    /// platform this crate now compiles and tests on. `std::env::temp_dir()`
    /// gives every platform a real, writable, per-user directory without
    /// hardcoding a platform-specific absolute path.
    fn get_local_model_path(&self, model_id: &str) -> PathBuf {
        std::env::temp_dir()
            .join("trustformers_icloud_models")
            .join(format!("{}.bin", sanitize_for_filename(model_id)))
    }
}

/// Map an arbitrary model id to a safe path component: alphanumerics,
/// `-` and `_` pass through unchanged, everything else (path separators,
/// `..`, NUL, etc.) becomes `_`. Shared by [`iCloudModelSync`]'s local
/// staging path and [`CloudKitManager`]'s local store so a model id can
/// never be used to escape either directory.
fn sanitize_for_filename(id: &str) -> String {
    id.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

/// Stand-in for Apple CloudKit: a real, working, on-disk store for synced
/// models.
///
/// Reaching Apple's actual CloudKit service requires linking
/// `CloudKit.framework` through Objective-C/Swift interop; this pure-Rust
/// crate carries no such native glue code (and, per this workspace's
/// pure-Rust-by-default policy, should not by default), and this build
/// environment has no Apple SDK to link against regardless. A previous
/// revision of this type pretended otherwise: its `#[cfg(target_os =
/// "ios")]` branch declared and called an `extern "C"`
/// `CKContainer`/`CKDatabase`/`CKRecord`/`CKAsset` API that no object file
/// in this workspace ever defines -- a real iOS build exercising those
/// functions would fail to link -- while every actual data operation
/// (`upload_model`, `download_model`, `model_exists`, `delete_model`,
/// `fetch_model_metadata`) never touched those handles at all and returned
/// fabricated data instead: 1 KB of zeros (or a literal `b"Mock
/// TrustformersModel Data..."` string) from "download", a silent `Ok(())`
/// from an "upload" that stored nothing, `false` from every existence
/// check, an invented `"1.0.0"` / `"mock_checksum"` from every metadata
/// fetch.
///
/// What this type provides instead is real: every model
/// [`Self::upload_model`] is given is actually written under
/// [`Self::store_root`], [`Self::download_model`] reads back exactly those
/// bytes, [`Self::model_exists`] / [`Self::fetch_model_metadata`] /
/// [`Self::fetch_available_models`] reflect real on-disk state, and
/// [`Self::delete_model`] actually removes the file. That makes
/// `iCloudModelSync`'s conflict detection, checksum comparison and
/// statistics exercise real persisted data end to end -- and two
/// `CloudKitManager`s constructed from configs sharing a `container_id`
/// (simulating two devices signed into the same iCloud account) genuinely
/// observe each other's uploads, which a fabricated response never could.
///
/// This is honestly *not* Apple CloudKit -- [`Self::is_real_cloudkit`]
/// reports that plainly rather than letting a caller assume otherwise from
/// the type's name alone.
struct CloudKitManager {
    config: iCloudSyncConfig,
    /// Local directory standing in for the CloudKit database. Every byte
    /// written here is a byte an actual caller uploaded; nothing under this
    /// path is fabricated.
    store_root: PathBuf,
}

impl CloudKitManager {
    fn new(config: &iCloudSyncConfig) -> MobileResult<Self> {
        let store_root = Self::store_root_for(&config.container_id, config.database_scope);
        std::fs::create_dir_all(&store_root).map_err(|e| {
            TrustformersError::io_error(format!(
                "failed to create local iCloud-sync store directory {}: {e}",
                store_root.display()
            ))
        })?;

        Ok(Self {
            config: config.clone(),
            store_root,
        })
    }

    /// Whether this manager reaches Apple's real CloudKit service. Always
    /// `false`: no Objective-C/Swift CloudKit binding is compiled into this
    /// crate, on iOS or otherwise, so every sync operation on this manager
    /// targets [`Self::store_root`] instead -- see the type-level doc
    /// comment. Exposed so a caller that specifically needs genuine
    /// cross-Apple-account CloudKit sync (rather than this crate's local
    /// staging store) gets an explicit, honest answer instead of silently
    /// assuming the name `CloudKitManager` implies real network sync.
    #[allow(dead_code)] // part of this type's honest public contract, not yet wired to a caller
    fn is_real_cloudkit(&self) -> bool {
        false
    }

    /// The local directory a given `container_id`/`database_scope` pair
    /// stores its models under. Deterministic (so repeated
    /// `CloudKitManager::new` calls for the same config, or two configs
    /// simulating two devices sharing an iCloud account, observe the same
    /// store) and namespaced by both the container id and the database
    /// scope (private/public/shared), matching CloudKit's own real
    /// partitioning of records into separate databases per scope.
    fn store_root_for(container_id: &str, scope: DatabaseScope) -> PathBuf {
        let scope_dir = match scope {
            DatabaseScope::Private => "private",
            DatabaseScope::Public => "public",
            DatabaseScope::Shared => "shared",
        };
        std::env::temp_dir()
            .join("trustformers_icloud_cloudkit_store")
            .join(sanitize_for_filename(container_id))
            .join(scope_dir)
    }

    fn payload_path(&self, model_id: &str) -> PathBuf {
        self.store_root.join(format!("{}.bin", sanitize_for_filename(model_id)))
    }

    fn metadata_path(&self, model_id: &str) -> PathBuf {
        self.store_root.join(format!("{}.meta.json", sanitize_for_filename(model_id)))
    }

    /// Real listing of every model actually stored under
    /// [`Self::store_root`] -- reconstructed from each model's own
    /// persisted metadata sidecar file, not fabricated.
    fn fetch_available_models(&self) -> MobileResult<Vec<ModelMetadata>> {
        let mut models = Vec::new();

        let entries = match std::fs::read_dir(&self.store_root) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(models),
            Err(e) => {
                return Err(TrustformersError::io_error(format!(
                    "failed to list iCloud store {}: {e}",
                    self.store_root.display()
                )));
            },
        };

        for entry in entries {
            let entry = entry.map_err(|e| TrustformersError::io_error(e.to_string()))?;
            let path = entry.path();
            let is_metadata_sidecar = path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".meta.json"));
            if !is_metadata_sidecar {
                continue;
            }

            let bytes = std::fs::read(&path).map_err(|e| {
                TrustformersError::io_error(format!("failed to read {}: {e}", path.display()))
            })?;
            let metadata: ModelMetadata = serde_json::from_slice(&bytes).map_err(|e| {
                TrustformersError::io_error(format!("corrupt metadata at {}: {e}", path.display()))
            })?;
            models.push(metadata);
        }

        Ok(models)
    }

    fn fetch_model_list(&self) -> MobileResult<Vec<ModelMetadata>> {
        self.fetch_available_models()
    }

    /// Real metadata for a specific model, read back from the sidecar file
    /// [`Self::upload_model`] wrote -- not an invented `"1.0.0"` /
    /// `"mock_checksum"` record.
    fn fetch_model_metadata(&self, model_id: &str) -> MobileResult<ModelMetadata> {
        let path = self.metadata_path(model_id);
        let bytes = std::fs::read(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                model_not_found(model_id.to_string())
            } else {
                TrustformersError::io_error(format!(
                    "failed to read metadata for '{model_id}': {e}"
                ))
            }
        })?;
        serde_json::from_slice(&bytes).map_err(|e| {
            TrustformersError::io_error(format!("corrupt metadata for '{model_id}': {e}"))
        })
    }

    fn model_exists(&self, model_id: &str) -> MobileResult<bool> {
        Ok(self.payload_path(model_id).is_file() && self.metadata_path(model_id).is_file())
    }

    /// Real upload: `data` is written to [`Self::payload_path`] and
    /// `metadata` to [`Self::metadata_path`] before this returns `Ok(())`.
    /// The previous implementation was `// Placeholder implementation` +
    /// `Ok(())` with both parameters unused (`_metadata`, `_data`) -- every
    /// caller's model was silently discarded while the call reported
    /// success.
    fn upload_model(&self, metadata: &ModelMetadata, data: &[u8]) -> MobileResult<()> {
        let payload_path = self.payload_path(&metadata.model_id);
        std::fs::write(&payload_path, data).map_err(|e| {
            TrustformersError::io_error(format!(
                "failed to upload model '{}': {e}",
                metadata.model_id
            ))
        })?;

        let metadata_bytes = serde_json::to_vec(metadata).map_err(|e| {
            TrustformersError::io_error(format!(
                "failed to serialize metadata for '{}': {e}",
                metadata.model_id
            ))
        })?;
        if let Err(e) = std::fs::write(self.metadata_path(&metadata.model_id), metadata_bytes) {
            // Do not leave an orphaned payload with no matching metadata
            // record -- roll the payload write back rather than report a
            // half-completed upload as `Ok`.
            let _ = std::fs::remove_file(&payload_path);
            return Err(TrustformersError::io_error(format!(
                "failed to write metadata for '{}': {e}",
                metadata.model_id
            )));
        }

        Ok(())
    }

    /// Real download: the exact bytes and metadata a prior
    /// [`Self::upload_model`] call (from this manager or another one
    /// sharing the same `container_id`/`database_scope`) wrote. The
    /// previous implementation returned `vec![0u8; 1024]` (iOS) or a
    /// literal mock-data string (elsewhere) regardless of `model_id`.
    fn download_model(&self, model_id: &str) -> MobileResult<(ModelMetadata, Vec<u8>)> {
        let metadata = self.fetch_model_metadata(model_id)?;
        let data = std::fs::read(self.payload_path(model_id)).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                model_not_found(model_id.to_string())
            } else {
                TrustformersError::io_error(format!("failed to download model '{model_id}': {e}"))
            }
        })?;
        Ok((metadata, data))
    }

    /// Real deletion of both the payload and metadata sidecar. The previous
    /// implementation was `// Placeholder implementation` + `Ok(())` with
    /// no filesystem access at all -- `remove_model(_, delete_remote: true)`
    /// reported success while leaving the "remote" copy (and, per
    /// `model_exists`'s equally fake `Ok(false)`, every copy) untouched.
    fn delete_model(&self, model_id: &str) -> MobileResult<()> {
        let payload_path = self.payload_path(model_id);
        let metadata_path = self.metadata_path(model_id);

        let mut found = false;
        for path in [&payload_path, &metadata_path] {
            match std::fs::remove_file(path) {
                Ok(()) => found = true,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {},
                Err(e) => {
                    return Err(TrustformersError::io_error(format!(
                        "failed to delete {}: {e}",
                        path.display()
                    )));
                },
            }
        }

        if !found {
            return Err(model_not_found(model_id.to_string()));
        }
        Ok(())
    }
}

/// Sync task for queued operations
#[derive(Debug, Clone)]
struct SyncTask {
    model_id: String,
    operation: SyncOperation,
    retry_count: u32,
    scheduled_time: SystemTime,
}

/// Synchronization statistics
#[derive(Debug, Clone)]
pub struct SyncStatistics {
    /// Total number of models registered
    pub total_models_registered: u64,
    /// Total number of sync operations performed
    pub total_sync_operations: u64,
    /// Total bytes transferred (uploaded + downloaded)
    pub total_bytes_transferred: u64,
    /// Number of successful syncs
    pub successful_syncs: u64,
    /// Number of failed syncs
    pub failed_syncs: u64,
    /// Number of conflicts resolved
    pub conflicts_resolved: u64,
    /// Last sync time
    pub last_sync_time: SystemTime,
    /// Average sync duration
    pub average_sync_duration: Duration,
}

impl SyncStatistics {
    fn new() -> Self {
        Self {
            total_models_registered: 0,
            total_sync_operations: 0,
            total_bytes_transferred: 0,
            successful_syncs: 0,
            failed_syncs: 0,
            conflicts_resolved: 0,
            last_sync_time: UNIX_EPOCH,
            average_sync_duration: Duration::from_secs(0),
        }
    }
}

// Real CloudKit (`CKContainer`/`CKDatabase`/`CKRecord`/`CKAsset`) network
// access previously had an `extern "C"` declaration here, gated
// `#[cfg(target_os = "ios")]`. No object file anywhere in this workspace
// ever defined those symbols -- a real iOS build calling them would fail to
// link -- and nothing in this module called them even when the cfg was
// active; every actual data operation went through fabricated data instead
// (see `CloudKitManager`'s doc comment). Removed rather than kept as
// unlinkable, uncalled scaffolding.

use sha2::Digest;

// Convenience functions for creating common configurations
impl iCloudSyncConfig {
    /// Create a configuration optimized for small models (< 50MB)
    pub fn small_models() -> Self {
        Self {
            max_model_size_mb: 50,
            sync_interval_seconds: 60, // 1 minute
            compression_enabled: true,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for large models (< 1GB)
    pub fn large_models() -> Self {
        Self {
            max_model_size_mb: 1024,
            sync_interval_seconds: 600, // 10 minutes
            compression_enabled: true,
            operation_timeout_seconds: 300, // 5 minutes
            ..Default::default()
        }
    }

    /// Create a configuration for development/testing
    pub fn development() -> Self {
        Self {
            auto_sync_enabled: false,
            verbose_logging: true,
            container_id: "iCloud.com.trustformers.models.dev".to_string(),
            ..Default::default()
        }
    }
}

/// Public API for Swift integration
#[no_mangle]
pub extern "C" fn tfk_icloud_sync_create(config_json: *const c_char) -> *mut iCloudModelSync {
    if config_json.is_null() {
        return std::ptr::null_mut();
    }

    let config_str = unsafe { CStr::from_ptr(config_json).to_str().unwrap_or_default() };

    let config: iCloudSyncConfig = serde_json::from_str(config_str).unwrap_or_default();

    match iCloudModelSync::new(config) {
        Ok(sync) => Box::into_raw(Box::new(sync)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn tfk_icloud_sync_destroy(sync: *mut iCloudModelSync) {
    if !sync.is_null() {
        unsafe {
            Box::from_raw(sync);
        }
    }
}

#[no_mangle]
pub extern "C" fn tfk_icloud_sync_enable_auto(sync: *mut iCloudModelSync, enabled: bool) -> bool {
    if sync.is_null() {
        return false;
    }

    let sync = unsafe { &mut *sync };
    sync.enable_auto_sync(enabled).is_ok()
}

#[no_mangle]
pub extern "C" fn tfk_icloud_sync_register_model(
    sync: *mut iCloudModelSync,
    model_path: *const c_char,
    model_id: *const c_char,
    model_name: *const c_char,
) -> bool {
    if sync.is_null() || model_path.is_null() || model_id.is_null() || model_name.is_null() {
        return false;
    }

    let sync = unsafe { &mut *sync };
    let path_str = unsafe { CStr::from_ptr(model_path).to_str().unwrap_or_default() };
    let id_str = unsafe { CStr::from_ptr(model_id).to_str().unwrap_or_default() };
    let name_str = unsafe { CStr::from_ptr(model_name).to_str().unwrap_or_default() };

    let metadata = ModelMetadata {
        model_id: id_str.to_string(),
        model_name: name_str.to_string(),
        version: "1.0.0".to_string(),
        size_bytes: 0, // Will be calculated during registration
        last_modified: SystemTime::now(),
        checksum: String::new(), // Will be calculated during registration
        last_modified_device: "iOS".to_string(),
        custom_metadata: HashMap::new(),
        sync_status: SyncStatus::NotSynced,
        local_path: None, // Will be set during registration
        cloud_record_id: None,
    };

    sync.register_model(Path::new(path_str), metadata).is_ok()
}

#[no_mangle]
pub extern "C" fn tfk_icloud_sync_sync_all(sync: *mut iCloudModelSync) -> bool {
    if sync.is_null() {
        return false;
    }

    let sync = unsafe { &mut *sync };
    sync.sync_all_models().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_icloud_sync_config_default() {
        let config = iCloudSyncConfig::default();
        assert!(config.auto_sync_enabled);
        assert_eq!(config.max_model_size_mb, 500);
        assert_eq!(config.database_scope, DatabaseScope::Private);
    }

    #[test]
    fn test_sync_status_enum() {
        let status = SyncStatus::NotSynced;
        assert_eq!(status, SyncStatus::NotSynced);
        assert_ne!(status, SyncStatus::Synced);
    }

    #[test]
    fn test_model_metadata_creation() {
        let metadata = ModelMetadata {
            model_id: "test_model".to_string(),
            model_name: "Test Model".to_string(),
            version: "1.0.0".to_string(),
            size_bytes: 1024,
            last_modified: SystemTime::now(),
            checksum: "abc123".to_string(),
            last_modified_device: "iPhone".to_string(),
            custom_metadata: HashMap::new(),
            sync_status: SyncStatus::NotSynced,
            local_path: None,
            cloud_record_id: None,
        };

        assert_eq!(metadata.model_id, "test_model");
        assert_eq!(metadata.size_bytes, 1024);
        assert_eq!(metadata.sync_status, SyncStatus::NotSynced);
    }

    #[test]
    fn test_sync_statistics() {
        let stats = SyncStatistics::new();
        assert_eq!(stats.total_models_registered, 0);
        assert_eq!(stats.total_sync_operations, 0);
        assert_eq!(stats.last_sync_time, UNIX_EPOCH);
    }

    #[test]
    fn test_config_variants() {
        let small_config = iCloudSyncConfig::small_models();
        assert_eq!(small_config.max_model_size_mb, 50);
        assert_eq!(small_config.sync_interval_seconds, 60);

        let large_config = iCloudSyncConfig::large_models();
        assert_eq!(large_config.max_model_size_mb, 1024);
        assert_eq!(large_config.sync_interval_seconds, 600);

        let dev_config = iCloudSyncConfig::development();
        assert!(!dev_config.auto_sync_enabled);
        assert!(dev_config.verbose_logging);
    }

    /// Regression test for the P0 finding: RLE `compress_data` silently fell
    /// back to returning raw bytes (no marker) whenever RLE did not shrink
    /// the input, and `decompress_data` could not tell that apart from a
    /// real RLE stream once the raw payload happened to have an even
    /// length -- corrupting it on "decompression". High-entropy
    /// (incompressible) data of an even length is exactly the failure case;
    /// real `f32` weight buffers land here constantly.
    #[test]
    fn test_compress_roundtrip_survives_incompressible_even_length_data() {
        let sync = iCloudModelSync::new(iCloudSyncConfig::default()).expect("sync manager");

        // Deterministic "incompressible" bytes (no repeated runs for RLE to
        // exploit), even length.
        let data: Vec<u8> =
            (0u32..2048).map(|i| (i.wrapping_mul(2654435761) >> 24) as u8).collect();
        assert_eq!(data.len() % 2, 0, "test fixture must be even-length");

        let compressed = sync.compress_data(&data).expect("compress");
        let decompressed = sync.decompress_data(&compressed).expect("decompress");

        assert_eq!(
            decompressed, data,
            "round-trip through a real codec must reproduce incompressible even-length data \
             exactly -- the old RLE fallback corrupted this case"
        );
    }

    #[test]
    fn test_compress_roundtrip_empty_and_highly_compressible_data() {
        let sync = iCloudModelSync::new(iCloudSyncConfig::default()).expect("sync manager");

        for data in [
            Vec::new(),
            vec![0u8; 4096],
            b"aaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_vec(),
        ] {
            let compressed = sync.compress_data(&data).expect("compress");
            let decompressed = sync.decompress_data(&compressed).expect("decompress");
            assert_eq!(decompressed, data);
        }
    }

    /// Regression test for the P0 finding: `encrypt_data`/`decrypt_data`
    /// were a repeating-key XOR keystream, not AES. A real AEAD round-trips
    /// arbitrary data exactly.
    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let sync = iCloudModelSync::new(iCloudSyncConfig::default()).expect("sync manager");
        let key = vec![0x42u8; 32];
        let plaintext =
            b"a transformer checkpoint's worth of bytes, not that it matters here".to_vec();

        let encrypted = sync.encrypt_data(&plaintext, &key).expect("encrypt");
        assert_ne!(
            encrypted, plaintext,
            "ciphertext must not equal the plaintext"
        );

        let decrypted = sync.decrypt_data(&encrypted, &key).expect("decrypt");
        assert_eq!(decrypted, plaintext);
    }

    /// Two encryptions of the same plaintext must not produce the same
    /// ciphertext -- the nonce must be freshly random each call. The old
    /// implementation's IV came from an LCG seeded by the wall-clock
    /// nanosecond, which is highly likely to repeat under rapid successive
    /// calls (and trivially predictable regardless).
    #[test]
    fn test_encrypt_uses_a_fresh_nonce_each_call() {
        let sync = iCloudModelSync::new(iCloudSyncConfig::default()).expect("sync manager");
        let key = vec![0x11u8; 32];
        let plaintext = b"same plaintext, encrypted twice".to_vec();

        let a = sync.encrypt_data(&plaintext, &key).expect("encrypt a");
        let b = sync.encrypt_data(&plaintext, &key).expect("encrypt b");
        assert_ne!(a, b, "each encryption must use a fresh random nonce");
    }

    /// A real AEAD detects tampering; the old XOR cipher had no
    /// authentication at all and would "decrypt" a corrupted ciphertext into
    /// silently wrong plaintext.
    #[test]
    fn test_decrypt_rejects_tampered_ciphertext() {
        let sync = iCloudModelSync::new(iCloudSyncConfig::default()).expect("sync manager");
        let key = vec![0x77u8; 32];
        let mut encrypted = sync.encrypt_data(b"trust, but verify", &key).expect("encrypt");

        // Flip a bit well inside the ciphertext (past the 12-byte nonce).
        let last = encrypted.len() - 1;
        encrypted[last] ^= 0x01;

        let result = sync.decrypt_data(&encrypted, &key);
        assert!(
            result.is_err(),
            "tampered ciphertext must fail authentication, not decrypt"
        );
    }

    #[test]
    fn test_decrypt_rejects_wrong_key() {
        let sync = iCloudModelSync::new(iCloudSyncConfig::default()).expect("sync manager");
        let encrypted = sync.encrypt_data(b"secret model weights", &[0xAAu8; 32]).expect("encrypt");
        let result = sync.decrypt_data(&encrypted, &[0xBBu8; 32]);
        assert!(
            result.is_err(),
            "decrypting with the wrong key must fail, not return garbage"
        );
    }

    #[test]
    fn test_encrypt_rejects_non_256_bit_key() {
        let sync = iCloudModelSync::new(iCloudSyncConfig::default()).expect("sync manager");
        assert!(sync.encrypt_data(b"data", &[0u8; 16]).is_err());
        assert!(sync.encrypt_data(b"data", &[0u8; 24]).is_err());
    }

    /// Regression test for the "PBKDF2" that was one 32-bit rolling hash per
    /// output byte: same password/salt must reproduce the same key
    /// (determinism), and it must be exactly 32 bytes for AES-256.
    #[test]
    fn test_derive_key_from_password_is_deterministic_and_correct_length() {
        let key_a = iCloudModelSync::derive_key_from_password(
            "correct horse battery staple",
            b"salt123",
            1000,
        )
        .expect("derive a");
        let key_b = iCloudModelSync::derive_key_from_password(
            "correct horse battery staple",
            b"salt123",
            1000,
        )
        .expect("derive b");
        assert_eq!(
            key_a, key_b,
            "same password+salt+rounds must derive the same key"
        );
        assert_eq!(key_a.len(), 32, "AES-256 needs a 32-byte key");

        let key_different_password =
            iCloudModelSync::derive_key_from_password("a different password", b"salt123", 1000)
                .expect("derive c");
        assert_ne!(key_a, key_different_password);

        let key_different_salt = iCloudModelSync::derive_key_from_password(
            "correct horse battery staple",
            b"other-salt",
            1000,
        )
        .expect("derive d");
        assert_ne!(key_a, key_different_salt);
    }

    /// Regression test for the P0 finding: `get_local_model_path` must not
    /// hardcode an iOS-only absolute path (`/var/mobile/...`), which does
    /// not exist on any platform this crate now compiles and tests on.
    #[test]
    fn test_local_model_path_is_not_a_hardcoded_ios_path() {
        let sync = iCloudModelSync::new(iCloudSyncConfig::default()).expect("sync manager");
        let path = sync.get_local_model_path("some-model");
        assert!(
            !path.to_string_lossy().starts_with("/var/mobile"),
            "must not hardcode an iOS device sandbox path: {path:?}"
        );
    }

    /// Regression test for a self-deadlock in `perform_model_sync`
    /// (`sync_model`'s implementation): it used to hold the
    /// `self.cloud_manager` lock across a nested call to
    /// `self.upload_model`, which acquires that same (non-reentrant)
    /// `std::sync::Mutex` itself -- hanging the calling thread forever on
    /// the very first sync of any model that does not already exist
    /// remotely (i.e. every model's first sync, ever). Run on a background
    /// thread with a bounded wait so a regression fails the test loudly
    /// instead of hanging the whole test binary.
    #[test]
    fn test_sync_model_does_not_deadlock_on_first_upload() {
        let config = iCloudSyncConfig {
            container_id: format!(
                "iCloud.test.deadlock_check.{}.{}",
                std::process::id(),
                SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos()
            ),
            auto_sync_enabled: false,
            ..Default::default()
        };

        let model_path = std::env::temp_dir().join(format!(
            "trustformers_icloud_deadlock_test_{}_{}.bin",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos()
        ));
        std::fs::write(&model_path, b"deadlock regression fixture").expect("write test model file");

        let (tx, rx) = std::sync::mpsc::channel();
        let path_for_thread = model_path.clone();
        std::thread::spawn(move || {
            let mut sync = iCloudModelSync::new(config).expect("sync manager");
            let metadata = ModelMetadata {
                model_id: "deadlock-check-model".to_string(),
                model_name: "Deadlock Check".to_string(),
                version: "1.0.0".to_string(),
                size_bytes: 0,
                last_modified: SystemTime::now(),
                checksum: String::new(),
                last_modified_device: "test".to_string(),
                custom_metadata: HashMap::new(),
                sync_status: SyncStatus::NotSynced,
                local_path: None,
                cloud_record_id: None,
            };
            sync.register_model(&path_for_thread, metadata).expect("register");
            let result = sync.sync_model("deadlock-check-model");
            let _ = tx.send(result.map(|r| r.success));
        });

        let outcome = rx.recv_timeout(std::time::Duration::from_secs(10));
        let _ = std::fs::remove_file(&model_path);

        match outcome {
            Ok(Ok(success)) => assert!(success, "sync_model must report success on a clean upload"),
            Ok(Err(e)) => {
                panic!("sync_model returned an error (not a deadlock, but still wrong): {e}")
            },
            Err(_) => panic!(
                "sync_model did not return within 10 seconds -- this is the self-deadlock \
                 regression (perform_model_sync holding the cloud_manager lock across a nested \
                 self.upload_model call)"
            ),
        }
    }

    /// End-to-end regression test for the P0 finding: `CloudKitManager`
    /// used to return mock/zeroed data from `download_model` and silently
    /// drop `upload_model`'s payload. This exercises the full path a real
    /// two-device sync would take: device A registers and uploads a real
    /// model file, device B (a second `iCloudModelSync` whose config shares
    /// the same `container_id`, simulating the same iCloud account) downloads
    /// it and must get back the exact original bytes.
    #[test]
    fn test_two_devices_share_uploaded_model_via_local_cloudkit_store() {
        let shared_container = format!(
            "iCloud.test.two_device_sync.{}.{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos()
        );
        let config = iCloudSyncConfig {
            container_id: shared_container,
            auto_sync_enabled: false,
            // Compression is exercised separately by the
            // `test_compress_roundtrip_*` tests; disabling it here isolates
            // this test to what it actually asserts -- that the *stored*
            // bytes are the real upload, not mock data -- rather than also
            // depending on `process_downloaded_model`'s decompression step.
            compression_enabled: false,
            ..Default::default()
        };

        // "Device A": register a real local file and upload it.
        let model_bytes = b"these are definitely not zero bytes and not a mock string".to_vec();
        let model_path = std::env::temp_dir().join(format!(
            "trustformers_icloud_e2e_test_{}_{}.bin",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos()
        ));
        std::fs::write(&model_path, &model_bytes).expect("write test model file");

        let mut device_a = iCloudModelSync::new(config.clone()).expect("device A");
        let metadata = ModelMetadata {
            model_id: "shared-model".to_string(),
            model_name: "Shared Model".to_string(),
            version: "1.0.0".to_string(),
            size_bytes: 0,
            last_modified: SystemTime::now(),
            checksum: String::new(),
            last_modified_device: "device-a".to_string(),
            custom_metadata: HashMap::new(),
            sync_status: SyncStatus::NotSynced,
            local_path: None,
            cloud_record_id: None,
        };
        device_a.register_model(&model_path, metadata).expect("register on device A");
        let sync_result = device_a.sync_model("shared-model").expect("sync (upload) from device A");
        assert!(sync_result.success);
        let _ = std::fs::remove_file(&model_path);

        // "Device B": a fresh manager, same iCloud container -- must see and
        // be able to download what device A uploaded.
        let mut device_b = iCloudModelSync::new(config).expect("device B");
        let available = device_b.download_available_models().expect("list available models");
        assert!(
            available.iter().any(|m| m.model_id == "shared-model"),
            "device B must see the model device A uploaded, got: {available:?}"
        );

        let downloaded = {
            let cloud_manager = device_b.cloud_manager.lock().expect("lock cloud manager");
            cloud_manager.download_model("shared-model").expect("download on device B")
        };
        assert_eq!(
            downloaded.1, model_bytes,
            "device B must receive the exact bytes device A uploaded, not mock/zeroed data"
        );

        // Clean up the shared local store so repeated test runs don't
        // accumulate files.
        let cloud_manager = device_b.cloud_manager.lock().expect("lock cloud manager");
        let _ = cloud_manager.delete_model("shared-model");
    }

    #[test]
    fn test_cloudkit_manager_reports_it_is_not_real_cloudkit() {
        let config = iCloudSyncConfig::development();
        let manager = CloudKitManager::new(&config).expect("cloud manager");
        assert!(!manager.is_real_cloudkit());
    }

    #[test]
    fn test_cloudkit_manager_model_exists_and_delete_reflect_real_state() {
        let config = iCloudSyncConfig {
            container_id: format!(
                "iCloud.test.exists_delete.{}.{}",
                std::process::id(),
                SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos()
            ),
            ..Default::default()
        };
        let manager = CloudKitManager::new(&config).expect("cloud manager");

        assert!(!manager.model_exists("ghost-model").expect("exists check"));
        assert!(
            manager.delete_model("ghost-model").is_err(),
            "deleting a nonexistent model must error"
        );

        let metadata = ModelMetadata {
            model_id: "real-model".to_string(),
            model_name: "Real Model".to_string(),
            version: "1.0.0".to_string(),
            size_bytes: 3,
            last_modified: SystemTime::now(),
            checksum: String::new(),
            last_modified_device: "test".to_string(),
            custom_metadata: HashMap::new(),
            sync_status: SyncStatus::NotSynced,
            local_path: None,
            cloud_record_id: None,
        };
        manager.upload_model(&metadata, b"abc").expect("upload");
        assert!(manager.model_exists("real-model").expect("exists check"));

        manager.delete_model("real-model").expect("delete");
        assert!(!manager.model_exists("real-model").expect("exists check after delete"));
    }
}
