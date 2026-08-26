//! Model Manager
//!
//! Handles loading, unloading, and lifecycle management of ML models.

use crate::model_management::{
    config::{LoadingStrategy, ModelManagementConfig, UnloadingStrategy},
    registry::ModelRegistry,
    versioning::VersionManager,
    ModelError, ModelLoadConfig, ModelMetadata, ModelResult, ModelStatus,
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use tokio::{sync::Semaphore, task::JoinHandle, time::timeout};
use trustformers_models::weight_loading::checkpoint::Checkpoint;

/// Represents a loaded model instance
#[derive(Debug)]
pub struct LoadedModel {
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Loading configuration used
    pub load_config: ModelLoadConfig,
    /// Time when model was loaded
    pub loaded_at: Instant,
    /// Last access time for LRU tracking
    pub last_accessed: Arc<RwLock<Instant>>,
    /// Model instance (placeholder - would contain actual model)
    pub instance: ModelInstance,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// GPU memory usage in bytes (if applicable)
    pub gpu_memory_usage: Option<u64>,
}

/// A model whose weights are resident in this process.
///
/// The instance owns the parsed checkpoint tensors, so `memory_bytes` is a
/// measurement of what was actually loaded rather than an estimate.
pub struct ModelInstance {
    pub model_type: String,
    pub device: String,
    pub precision: String,
    /// The loaded checkpoint. `None` only for instances created by
    /// [`ModelInstance::without_weights`], which explicitly refuse inference.
    checkpoint: Option<Arc<Checkpoint>>,
    /// Number of tensors bound from the checkpoint.
    tensor_count: usize,
    /// Total resident size of the loaded tensors, in bytes.
    memory_bytes: u64,
}

impl std::fmt::Debug for ModelInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelInstance")
            .field("model_type", &self.model_type)
            .field("device", &self.device)
            .field("precision", &self.precision)
            .field("tensor_count", &self.tensor_count)
            .field("memory_bytes", &self.memory_bytes)
            .field("weights_loaded", &self.checkpoint.is_some())
            .finish()
    }
}

impl ModelInstance {
    /// Load a real checkpoint from disk.
    ///
    /// `path` may be a checkpoint file or a directory containing one of
    /// `model.safetensors`, `pytorch_model.bin` or `model.bin`.
    ///
    /// # Errors
    ///
    /// Fails when the path does not exist, when no recognised checkpoint is
    /// present, or when the container cannot be parsed. A model is never
    /// invented for a path that holds no weights.
    pub fn load(
        path: &Path,
        model_type: String,
        device: String,
        precision: String,
    ) -> ModelResult<Self> {
        let checkpoint_path = resolve_checkpoint_path(path)?;
        let bytes = std::fs::read(&checkpoint_path).map_err(|e| ModelError::LoadingFailed {
            error: format!("failed to read {}: {}", checkpoint_path.display(), e),
        })?;
        let checkpoint = Checkpoint::from_bytes(&bytes).map_err(|e| ModelError::LoadingFailed {
            error: format!(
                "failed to parse checkpoint {}: {}",
                checkpoint_path.display(),
                e
            ),
        })?;

        let tensor_count = checkpoint.len();
        let memory_bytes = measure_checkpoint_bytes(&checkpoint);
        if tensor_count == 0 {
            return Err(ModelError::LoadingFailed {
                error: format!(
                    "checkpoint {} contains no tensors",
                    checkpoint_path.display()
                ),
            });
        }

        Ok(Self {
            model_type,
            device,
            precision,
            checkpoint: Some(Arc::new(checkpoint)),
            tensor_count,
            memory_bytes,
        })
    }

    /// An instance that carries no weights.
    ///
    /// Used for registry entries whose weights have not been fetched yet; every
    /// inference call against it returns [`ModelError::LoadingFailed`].
    pub fn without_weights(model_type: String, device: String, precision: String) -> Self {
        Self {
            model_type,
            device,
            precision,
            checkpoint: None,
            tensor_count: 0,
            memory_bytes: 0,
        }
    }

    /// Number of tensors bound from the checkpoint.
    pub fn tensor_count(&self) -> usize {
        self.tensor_count
    }

    /// Measured resident size of the loaded weights, in bytes.
    pub fn memory_bytes(&self) -> u64 {
        self.memory_bytes
    }

    /// Whether real weights are resident.
    pub fn has_weights(&self) -> bool {
        self.checkpoint.is_some()
    }

    /// Names of the loaded tensors.
    pub fn tensor_names(&self) -> Vec<String> {
        self.checkpoint.as_ref().map(|c| c.names()).unwrap_or_default()
    }

    /// Verify that the instance is usable for inference.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::LoadingFailed`] when no weights are resident. This
    /// deliberately does not synthesize a response: text generation belongs to
    /// the batching executor, which owns the tokenizer and the decode loop.
    pub async fn infer(&self, input: &str) -> ModelResult<String> {
        let checkpoint = self.checkpoint.as_ref().ok_or_else(|| ModelError::LoadingFailed {
            error: "model instance has no resident weights; load a checkpoint first".to_string(),
        })?;

        if checkpoint.is_empty() {
            return Err(ModelError::LoadingFailed {
                error: "model instance holds an empty checkpoint".to_string(),
            });
        }

        Err(ModelError::InvalidConfig {
            error: format!(
                "ModelInstance holds {} loaded tensors but does not own a decode loop; \
                 route the request for {:?} through the batching executor \
                 (crate::batching::ModelBatchExecutor)",
                self.tensor_count,
                input.chars().take(32).collect::<String>()
            ),
        })
    }
}

/// Resolve a user-supplied model path to an actual checkpoint file.
fn resolve_checkpoint_path(path: &Path) -> ModelResult<PathBuf> {
    if path.as_os_str().is_empty() {
        return Err(ModelError::LoadingFailed {
            error: "no model_path was supplied; the server does not fabricate weights".to_string(),
        });
    }
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    if path.is_dir() {
        for candidate in [
            "model.safetensors",
            "pytorch_model.bin",
            "model.bin",
            "weights.safetensors",
        ] {
            let file = path.join(candidate);
            if file.is_file() {
                return Ok(file);
            }
        }
        return Err(ModelError::LoadingFailed {
            error: format!(
                "directory {} contains no recognised checkpoint (looked for \
                 model.safetensors, pytorch_model.bin, model.bin, weights.safetensors)",
                path.display()
            ),
        });
    }
    Err(ModelError::ModelNotFound {
        id: path.display().to_string(),
    })
}

/// Total resident size of a checkpoint's tensors, in bytes.
fn measure_checkpoint_bytes(checkpoint: &Checkpoint) -> u64 {
    checkpoint
        .names()
        .iter()
        .filter_map(|name| checkpoint.get(name))
        .map(|tensor| {
            let elements: usize = tensor.shape().iter().product();
            (elements * std::mem::size_of::<f32>()) as u64
        })
        .sum()
}

/// What a completed load actually produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct ModelLoadReport {
    /// Tensors bound from the checkpoint.
    pub tensor_count: usize,
    /// Measured resident size of those tensors, in bytes.
    pub memory_bytes: u64,
}

/// Model manager handles loading, unloading, and lifecycle of models
pub struct ModelManager {
    /// Configuration
    config: ModelManagementConfig,

    /// Model registry
    registry: Arc<ModelRegistry>,

    /// Version manager
    version_manager: Arc<VersionManager>,

    /// Currently loaded models
    loaded_models: Arc<RwLock<HashMap<String, Arc<LoadedModel>>>>,

    /// Loading semaphore to limit concurrent loads
    load_semaphore: Arc<Semaphore>,

    /// Background task handles
    background_tasks: Arc<RwLock<Vec<JoinHandle<()>>>>,
}

impl ModelManager {
    /// Create a new model manager
    pub fn new(
        config: ModelManagementConfig,
        registry: Arc<ModelRegistry>,
        version_manager: Arc<VersionManager>,
    ) -> Self {
        let load_semaphore = Arc::new(Semaphore::new(config.max_loaded_models));

        Self {
            config,
            registry,
            version_manager,
            loaded_models: Arc::new(RwLock::new(HashMap::new())),
            load_semaphore,
            background_tasks: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Start the model manager
    pub async fn start(&self) -> ModelResult<()> {
        // Start background health check task
        let health_check_task = self.start_health_check_task().await;
        self.background_tasks
            .write()
            .map_err(|e| ModelError::InvalidConfig {
                error: format!("Lock poisoned: {}", e),
            })?
            .push(health_check_task);

        // Start cleanup task
        let cleanup_task = self.start_cleanup_task().await;
        self.background_tasks
            .write()
            .map_err(|e| ModelError::InvalidConfig {
                error: format!("Lock poisoned: {}", e),
            })?
            .push(cleanup_task);

        Ok(())
    }

    /// Stop the model manager
    pub async fn stop(&self) -> ModelResult<()> {
        // Cancel background tasks
        let tasks = {
            let mut tasks =
                self.background_tasks.write().map_err(|e| ModelError::InvalidConfig {
                    error: format!("Lock poisoned: {}", e),
                })?;
            std::mem::take(&mut *tasks)
        };

        for task in tasks {
            task.abort();
        }

        // Unload all models
        let model_ids: Vec<String> = {
            self.loaded_models
                .read()
                .map_err(|e| ModelError::InvalidConfig {
                    error: format!("Lock poisoned: {}", e),
                })?
                .keys()
                .cloned()
                .collect()
        };

        for model_id in model_ids {
            self.unload_model(&model_id, UnloadingStrategy::Immediate).await?;
        }

        Ok(())
    }

    /// Load a model
    pub async fn load_model(
        &self,
        model_id: &str,
        load_config: ModelLoadConfig,
        strategy: LoadingStrategy,
    ) -> ModelResult<()> {
        // Get model metadata
        let metadata =
            self.registry.get_model(model_id).ok_or_else(|| ModelError::ModelNotFound {
                id: model_id.to_string(),
            })?;

        // Check if already loaded
        if self
            .loaded_models
            .read()
            .map_err(|e| ModelError::InvalidConfig {
                error: format!("Lock poisoned: {}", e),
            })?
            .contains_key(model_id)
        {
            return Ok(());
        }

        // Acquire loading permit
        let _permit = match strategy {
            LoadingStrategy::Parallel { .. } => {
                // For parallel loading, use the global semaphore but don't enforce strict limits
                self.load_semaphore.acquire().await.map_err(|e| ModelError::LoadingFailed {
                    error: format!("Semaphore acquisition failed: {}", e),
                })?
            },
            _ => self.load_semaphore.acquire().await.map_err(|e| ModelError::LoadingFailed {
                error: format!("Semaphore acquisition failed: {}", e),
            })?,
        };

        // Update status to loading
        self.registry.update_status(model_id, ModelStatus::Loading).await?;

        // Perform the actual loading
        match strategy {
            LoadingStrategy::Lazy => {
                self.load_model_impl(model_id, metadata, load_config).await?;
            },
            LoadingStrategy::Eager => {
                self.load_model_impl(model_id, metadata, load_config).await?;
            },
            LoadingStrategy::Preload => {
                let model_id = model_id.to_string();
                let manager = self.clone_for_background();
                tokio::spawn(async move {
                    if let Err(e) = manager.load_model_impl(&model_id, metadata, load_config).await
                    {
                        tracing::error!("Preload failed for model {}: {}", model_id, e);
                    }
                });
            },
            LoadingStrategy::Parallel { .. } => {
                self.load_model_impl(model_id, metadata, load_config).await?;
            },
        }

        Ok(())
    }

    /// Unload a model
    pub async fn unload_model(
        &self,
        model_id: &str,
        strategy: UnloadingStrategy,
    ) -> ModelResult<()> {
        let loaded_model = {
            self.loaded_models
                .read()
                .map_err(|e| ModelError::InvalidConfig {
                    error: format!("Lock poisoned: {}", e),
                })?
                .get(model_id)
                .cloned()
        };

        if let Some(_loaded_model) = loaded_model {
            match strategy {
                UnloadingStrategy::Immediate => {
                    self.unload_model_impl(model_id).await?;
                },
                UnloadingStrategy::Graceful {
                    timeout: timeout_duration,
                } => {
                    // Set status to draining
                    self.registry.update_status(model_id, ModelStatus::Draining).await?;

                    // Wait for timeout or until no active requests
                    let start = Instant::now();
                    while start.elapsed() < timeout_duration {
                        // In a real implementation, check for active requests
                        // For now, just wait a bit
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }

                    self.unload_model_impl(model_id).await?;
                },
                UnloadingStrategy::Cached { ttl } => {
                    // Mark for later cleanup but keep in memory
                    self.registry.update_status(model_id, ModelStatus::Standby).await?;

                    // Schedule cleanup after TTL
                    let model_id = model_id.to_string();
                    let manager = self.clone_for_background();
                    tokio::spawn(async move {
                        tokio::time::sleep(ttl).await;
                        if let Err(e) = manager.unload_model_impl(&model_id).await {
                            tracing::error!("Cached model cleanup failed for {}: {}", model_id, e);
                        }
                    });
                },
            }
        }

        Ok(())
    }

    /// Get a loaded model
    pub fn get_loaded_model(&self, model_id: &str) -> Option<Arc<LoadedModel>> {
        let loaded_model = self.loaded_models.read().ok()?.get(model_id).cloned();

        if let Some(ref loaded_model) = loaded_model {
            // Update last accessed time
            *loaded_model.last_accessed.write().ok()? = Instant::now();
        }

        loaded_model
    }

    /// Look at a loaded model without marking it as used.
    ///
    /// [`Self::get_loaded_model`] refreshes the LRU stamp, which is right for a
    /// caller that is about to run inference and wrong for one that is merely
    /// reporting state: a status query that touched every model would make the
    /// LRU unloading strategy believe the whole set is hot. Read-only reporting
    /// paths use this instead.
    pub fn peek_loaded_model(&self, model_id: &str) -> Option<Arc<LoadedModel>> {
        self.loaded_models.read().ok()?.get(model_id).cloned()
    }

    /// List all loaded models
    pub fn list_loaded_models(&self) -> Vec<String> {
        self.loaded_models
            .read()
            .ok()
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Get memory usage statistics
    pub fn get_memory_usage(&self) -> (u64, Option<u64>) {
        let loaded_models = match self.loaded_models.read() {
            Ok(guard) => guard,
            Err(_) => return (0, None),
        };
        let total_memory: u64 = loaded_models.values().map(|m| m.memory_usage).sum();
        let total_gpu_memory: Option<u64> = {
            let gpu_usages: Vec<u64> =
                loaded_models.values().filter_map(|m| m.gpu_memory_usage).collect();
            if gpu_usages.is_empty() {
                None
            } else {
                Some(gpu_usages.into_iter().sum())
            }
        };

        (total_memory, total_gpu_memory)
    }

    /// Check if resource limits allow loading a new model
    pub fn check_resource_availability(
        &self,
        estimated_memory: u64,
        estimated_gpu_memory: Option<u64>,
    ) -> bool {
        let (current_memory, current_gpu_memory) = self.get_memory_usage();

        // Check memory limit
        let memory_limit = (self.config.resource_limits.max_memory_bytes as f32
            * (1.0 - self.config.resource_limits.memory_safety_buffer))
            as u64;

        if current_memory + estimated_memory > memory_limit {
            return false;
        }

        // Check GPU memory limit if applicable
        if let (Some(estimated_gpu), Some(max_gpu)) = (
            estimated_gpu_memory,
            self.config.resource_limits.max_gpu_memory_bytes,
        ) {
            let current_gpu = current_gpu_memory.unwrap_or(0);
            let gpu_limit =
                (max_gpu as f32 * (1.0 - self.config.resource_limits.memory_safety_buffer)) as u64;

            if current_gpu + estimated_gpu > gpu_limit {
                return false;
            }
        }

        true
    }

    /// Register (if necessary) and load a model by name and version.
    ///
    /// Returns the real load report: how many tensors were bound and how many
    /// bytes they occupy.
    pub async fn load_named_model(
        &self,
        model_name: &str,
        model_version: &str,
        load_config: ModelLoadConfig,
        strategy: LoadingStrategy,
    ) -> ModelResult<ModelLoadReport> {
        // The checkpoint must exist before anything is registered, so a bad
        // request never leaves a phantom entry in the registry.
        let checkpoint_path = resolve_checkpoint_path(Path::new(&load_config.model_path))?;

        // The registry persists metadata to disk; make sure its directory exists
        // and any previously persisted entries are indexed. Idempotent.
        self.registry.initialize().await?;

        let model_id = match self.registry.get_model_by_name_version(model_name, model_version) {
            Some(existing) => existing.id,
            None => {
                self.registry
                    .register_model(
                        model_name.to_string(),
                        model_version.to_string(),
                        checkpoint_path.display().to_string(),
                        "{}".to_string(),
                        crate::model_management::DeploymentStrategy::Replace,
                        HashMap::new(),
                    )
                    .await?
            },
        };

        self.load_model(&model_id, load_config, strategy).await?;

        let loaded = self.get_loaded_model(&model_id).ok_or_else(|| ModelError::LoadingFailed {
            error: format!("model {} did not become resident after loading", model_id),
        })?;

        Ok(ModelLoadReport {
            tensor_count: loaded.instance.tensor_count(),
            memory_bytes: loaded.memory_usage,
        })
    }

    /// Implementation of model loading.
    ///
    /// Reads the checkpoint from disk through
    /// [`trustformers_models::weight_loading::checkpoint::Checkpoint`] and
    /// records the measured resident size of the parsed tensors.
    async fn load_model_impl(
        &self,
        model_id: &str,
        metadata: ModelMetadata,
        load_config: ModelLoadConfig,
    ) -> ModelResult<()> {
        // Prefer the caller's explicit path, falling back to the path recorded
        // in the registry when the request did not carry one.
        let raw_path = if load_config.model_path.is_empty() {
            metadata.path.clone()
        } else {
            load_config.model_path.clone()
        };

        let device = load_config.device.clone();
        let precision = load_config.precision.clone();
        let architecture = metadata.name.clone();

        let load_future = tokio::task::spawn_blocking(move || {
            ModelInstance::load(Path::new(&raw_path), architecture, device, precision)
        });

        let instance = timeout(self.config.load_timeout, load_future)
            .await
            .map_err(|_| ModelError::LoadingFailed {
                error: format!(
                    "loading model {} timed out after {:?}",
                    model_id, self.config.load_timeout
                ),
            })?
            .map_err(|e| ModelError::LoadingFailed {
                error: format!("model loading task failed: {}", e),
            })??;

        // Measured, not assumed.
        let memory_usage = instance.memory_bytes();
        let gpu_memory_usage =
            if load_config.device.starts_with("cuda") { Some(memory_usage) } else { None };

        // Check resource availability against the real footprint
        if !self.check_resource_availability(memory_usage, gpu_memory_usage) {
            return Err(ModelError::ResourceConstraint {
                constraint: format!(
                    "loading {} needs {} bytes, which exceeds the configured limit",
                    model_id, memory_usage
                ),
            });
        }

        let tensor_count = instance.tensor_count();

        // Create loaded model
        let loaded_model = Arc::new(LoadedModel {
            metadata: metadata.clone(),
            load_config,
            loaded_at: Instant::now(),
            last_accessed: Arc::new(RwLock::new(Instant::now())),
            instance,
            memory_usage,
            gpu_memory_usage,
        });

        // Add to loaded models
        {
            let mut loaded_models = self.loaded_models.write().unwrap_or_else(|p| p.into_inner());
            loaded_models.insert(model_id.to_string(), loaded_model);
        }

        // Update status to active
        self.registry.update_status(model_id, ModelStatus::Active).await?;

        tracing::info!(
            "Model {} loaded: {} tensors, {} bytes resident",
            model_id,
            tensor_count,
            memory_usage
        );
        Ok(())
    }

    /// Implementation of model unloading
    async fn unload_model_impl(&self, model_id: &str) -> ModelResult<()> {
        // Remove from loaded models
        let loaded_model = {
            let mut loaded_models = self.loaded_models.write().unwrap_or_else(|p| p.into_inner());
            loaded_models.remove(model_id)
        };

        if loaded_model.is_some() {
            // Update status to unloaded
            self.registry.update_status(model_id, ModelStatus::Unloaded).await?;

            tracing::info!("Model {} unloaded successfully", model_id);
        }

        Ok(())
    }

    /// Start health check background task
    async fn start_health_check_task(&self) -> JoinHandle<()> {
        let manager = self.clone_for_background();
        let interval = self.config.health_check_interval;

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                if let Err(e) = manager.perform_health_check().await {
                    tracing::error!("Health check failed: {}", e);
                }
            }
        })
    }

    /// Start cleanup background task
    async fn start_cleanup_task(&self) -> JoinHandle<()> {
        let manager = self.clone_for_background();
        let interval = self.config.cleanup_interval;
        let max_versions = self.config.max_versions_per_model;

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                if let Err(e) = manager.perform_cleanup(max_versions).await {
                    tracing::error!("Cleanup failed: {}", e);
                }
            }
        })
    }

    /// Perform health check on loaded models
    async fn perform_health_check(&self) -> ModelResult<()> {
        let model_ids: Vec<String> = {
            self.loaded_models
                .read()
                .map_err(|e| ModelError::InvalidConfig {
                    error: format!("Lock poisoned: {}", e),
                })?
                .keys()
                .cloned()
                .collect()
        };

        for model_id in model_ids {
            if let Some(loaded_model) = self.get_loaded_model(&model_id) {
                // A model is healthy when its weights are genuinely resident.
                if loaded_model.instance.has_weights() && loaded_model.instance.tensor_count() > 0 {
                    self.registry.update_status(&model_id, ModelStatus::Active).await?;
                } else {
                    let error = "model has no resident weights".to_string();
                    tracing::warn!("Health check failed for model {}: {}", model_id, error);
                    self.registry.update_status(&model_id, ModelStatus::Failed { error }).await?;
                }
            }
        }

        Ok(())
    }

    /// Perform cleanup of old versions and unused models
    async fn perform_cleanup(&self, max_versions_per_model: usize) -> ModelResult<()> {
        // Cleanup old versions in registry
        let removed_models = self.registry.cleanup_old_versions(max_versions_per_model).await?;

        // Unload any models that were removed from registry
        for model_id in removed_models {
            let should_unload = self
                .loaded_models
                .read()
                .unwrap_or_else(|p| p.into_inner())
                .contains_key(&model_id);
            if should_unload {
                self.unload_model(&model_id, UnloadingStrategy::Immediate).await?;
            }
        }

        Ok(())
    }

    /// Clone for background tasks (simplified for this implementation)
    fn clone_for_background(&self) -> ModelManager {
        ModelManager {
            config: self.config.clone(),
            registry: Arc::clone(&self.registry),
            version_manager: Arc::clone(&self.version_manager),
            loaded_models: Arc::clone(&self.loaded_models),
            load_semaphore: Arc::clone(&self.load_semaphore),
            background_tasks: Arc::clone(&self.background_tasks),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_management::{config::ModelManagementConfig, versioning::VersionManager};
    use tempfile::TempDir;

    /// Write a minimal but genuine safetensors container holding one f32 tensor.
    fn write_safetensors(path: &Path, name: &str, shape: &[usize]) -> u64 {
        let elements: usize = shape.iter().product();
        let data = vec![0.5f32; elements];
        let byte_len = elements * std::mem::size_of::<f32>();
        let header = serde_json::json!({
            name: {
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [0, byte_len],
            }
        });
        let header_bytes = serde_json::to_vec(&header).expect("header serializes");

        let mut out = Vec::new();
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        for value in &data {
            out.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(path, out).expect("fixture writes");
        byte_len as u64
    }

    #[tokio::test]
    async fn test_model_manager() {
        let temp_dir = TempDir::new().expect("test operation should succeed");
        let checkpoint = temp_dir.path().join("model.safetensors");
        let expected_bytes = write_safetensors(&checkpoint, "weight", &[4, 8]);

        let config = ModelManagementConfig::default();
        let registry = Arc::new(ModelRegistry::new(
            temp_dir.path().to_string_lossy().to_string(),
        ));
        let version_manager = Arc::new(VersionManager::new());

        registry.initialize().await.expect("initialization should succeed in test");

        let manager = ModelManager::new(config, registry.clone(), version_manager);

        // Register a model
        let model_id = registry
            .register_model(
                "test-model".to_string(),
                "1.0.0".to_string(),
                checkpoint.display().to_string(),
                "{}".to_string(),
                crate::model_management::DeploymentStrategy::Replace,
                HashMap::new(),
            )
            .await
            .expect("test operation should succeed");

        let load_config = ModelLoadConfig {
            model_path: checkpoint.display().to_string(),
            revision: None,
            precision: "fp32".to_string(),
            device: "cpu".to_string(),
            max_batch_size: 1,
            max_sequence_length: 512,
            kv_cache_size: None,
            config_overrides: HashMap::new(),
        };

        manager
            .load_model(&model_id, load_config, LoadingStrategy::Eager)
            .await
            .expect("test operation should succeed");

        let loaded = manager.get_loaded_model(&model_id).expect("model must be resident");

        // Regression: the footprint must be the measured tensor size, not 1 GiB.
        assert_eq!(loaded.memory_usage, expected_bytes);
        assert_ne!(loaded.memory_usage, 1024 * 1024 * 1024);
        assert_eq!(loaded.instance.tensor_count(), 1);
        assert!(loaded.instance.has_weights());

        manager
            .unload_model(&model_id, UnloadingStrategy::Immediate)
            .await
            .expect("async operation should succeed in test");

        assert!(manager.get_loaded_model(&model_id).is_none());
    }

    /// Regression: loading a path that holds no weights must fail rather than
    /// producing a placeholder instance.
    #[tokio::test]
    async fn loading_a_missing_checkpoint_fails() {
        let temp_dir = TempDir::new().expect("temp dir");
        let config = ModelManagementConfig::default();
        let registry = Arc::new(ModelRegistry::new(
            temp_dir.path().to_string_lossy().to_string(),
        ));
        registry.initialize().await.expect("registry initializes");
        let manager = ModelManager::new(config, registry.clone(), Arc::new(VersionManager::new()));

        let missing = temp_dir.path().join("definitely-absent");
        let model_id = registry
            .register_model(
                "absent".to_string(),
                "1.0.0".to_string(),
                missing.display().to_string(),
                "{}".to_string(),
                crate::model_management::DeploymentStrategy::Replace,
                HashMap::new(),
            )
            .await
            .expect("registration succeeds");

        let load_config = ModelLoadConfig {
            model_path: missing.display().to_string(),
            revision: None,
            precision: "fp32".to_string(),
            device: "cpu".to_string(),
            max_batch_size: 1,
            max_sequence_length: 128,
            kv_cache_size: None,
            config_overrides: HashMap::new(),
        };

        let error = manager
            .load_model(&model_id, load_config, LoadingStrategy::Eager)
            .await
            .expect_err("a missing checkpoint must not load");
        assert!(
            error.to_string().to_lowercase().contains("not found")
                || error.to_string().contains("definitely-absent"),
            "unexpected error: {error}"
        );
        assert!(manager.get_loaded_model(&model_id).is_none());
    }

    /// Regression: `infer` must not synthesize "Generated response for: …".
    #[tokio::test]
    async fn instance_without_weights_refuses_inference() {
        let instance = ModelInstance::without_weights(
            "gpt2".to_string(),
            "cpu".to_string(),
            "fp32".to_string(),
        );
        let error = instance.infer("hello").await.expect_err("must refuse");
        assert!(error.to_string().contains("no resident weights"));
        assert_eq!(instance.memory_bytes(), 0);
    }

    /// Regression: a loaded instance still refuses to fabricate text; generation
    /// belongs to the batching executor.
    #[tokio::test]
    async fn loaded_instance_directs_callers_to_the_executor() {
        let temp_dir = TempDir::new().expect("temp dir");
        let checkpoint = temp_dir.path().join("model.safetensors");
        write_safetensors(&checkpoint, "weight", &[2, 2]);

        let instance = ModelInstance::load(
            &checkpoint,
            "gpt2".to_string(),
            "cpu".to_string(),
            "fp32".to_string(),
        )
        .expect("checkpoint loads");
        assert!(instance.has_weights());
        assert_eq!(instance.memory_bytes(), 16);

        let error = instance.infer("hello").await.expect_err("must not fabricate");
        let message = error.to_string();
        assert!(!message.contains("Generated response for"));
        assert!(message.contains("batching executor"));
    }

    /// Regression: a directory that holds a checkpoint resolves to it.
    #[test]
    fn directory_checkpoints_resolve() {
        let temp_dir = TempDir::new().expect("temp dir");
        let checkpoint = temp_dir.path().join("model.safetensors");
        write_safetensors(&checkpoint, "weight", &[1, 1]);
        assert_eq!(
            resolve_checkpoint_path(temp_dir.path()).expect("resolves"),
            checkpoint
        );

        let empty = TempDir::new().expect("temp dir");
        assert!(resolve_checkpoint_path(empty.path()).is_err());
        assert!(resolve_checkpoint_path(Path::new("")).is_err());
    }
}
