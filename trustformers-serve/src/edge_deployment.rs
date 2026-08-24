// Edge Deployment Infrastructure for TrustformeRS
// Provides comprehensive edge deployment capabilities for distributed inference
// at the edge, including offline mode, model synchronization, and bandwidth optimization

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{mpsc, RwLock};
use uuid::Uuid;

/// Size of one chunk in the edge artifact transfer, in bytes.
///
/// Model artifacts are copied and hashed a chunk at a time so a multi-gigabyte
/// artifact never has to be held in memory.
pub const TRANSFER_CHUNK_BYTES: usize = 1024 * 1024;

/// Edge deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeConfig {
    /// Edge node identifier
    pub node_id: String,
    /// Node location (geographic region)
    pub location: String,
    /// Available storage capacity in MB
    pub storage_capacity_mb: u64,
    /// Available memory in MB
    pub memory_capacity_mb: u64,
    /// CPU cores available
    pub cpu_cores: u32,
    /// GPU memory in MB (0 if no GPU)
    pub gpu_memory_mb: u64,
    /// Network bandwidth in Mbps
    pub bandwidth_mbps: u32,
    /// Latency to central server in ms
    pub latency_to_central_ms: u32,
    /// Operating mode
    pub mode: EdgeMode,
    /// Synchronization settings
    pub sync_config: SyncConfig,
    /// Optimization settings
    pub optimization: EdgeOptimization,
}

/// Edge operating modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeMode {
    /// Always connected to central server
    Connected,
    /// Can operate offline with local models
    Hybrid,
    /// Primarily offline with periodic sync
    Offline,
    /// Emergency mode with minimal functionality
    Emergency,
}

/// Model synchronization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    /// Sync interval in seconds
    pub sync_interval_seconds: u64,
    /// Maximum model size to sync in MB
    pub max_model_size_mb: u64,
    /// Priority models that should always be available
    pub priority_models: Vec<String>,
    /// Compression level (0-9)
    pub compression_level: u8,
    /// Delta sync enabled
    pub delta_sync: bool,
    /// Bandwidth throttle in Mbps
    pub bandwidth_throttle_mbps: Option<u32>,
}

/// Edge optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeOptimization {
    /// Model quantization enabled
    pub quantization: bool,
    /// Pruning enabled
    pub pruning: bool,
    /// Knowledge distillation for model compression
    pub distillation: bool,
    /// Cache optimization
    pub cache_optimization: bool,
    /// Bandwidth optimization strategies
    pub bandwidth_strategies: Vec<BandwidthStrategy>,
}

/// Bandwidth optimization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BandwidthStrategy {
    /// Compress responses
    Compression,
    /// Cache frequently requested results
    Caching,
    /// Use delta updates
    DeltaUpdates,
    /// Prefetch popular models
    Prefetching,
    /// Request batching
    Batching,
}

/// Edge node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeNode {
    pub id: String,
    pub location: String,
    pub status: EdgeNodeStatus,
    pub resources: EdgeResources,
    pub models: Vec<EdgeModel>,
    pub last_sync: SystemTime,
    pub metrics: EdgeMetrics,
}

/// Edge node status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeNodeStatus {
    Online,
    Offline,
    Syncing,
    Degraded,
    Error(String),
}

/// Edge node resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeResources {
    pub storage_used_mb: u64,
    pub storage_available_mb: u64,
    pub memory_used_mb: u64,
    pub memory_available_mb: u64,
    pub cpu_usage_percent: f32,
    pub gpu_usage_percent: f32,
    pub network_usage_mbps: f32,
}

/// Edge model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeModel {
    pub id: String,
    pub name: String,
    pub version: String,
    pub size_mb: u64,
    pub format: ModelFormat,
    pub optimization_level: OptimizationLevel,
    pub last_updated: SystemTime,
    pub usage_count: u64,
    pub priority: ModelPriority,
}

/// Model format for edge deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelFormat {
    /// Original unoptimized model
    Original,
    /// Quantized model (INT8/INT4)
    Quantized,
    /// Pruned model
    Pruned,
    /// Distilled model
    Distilled,
    /// Hybrid optimized
    Hybrid,
}

/// Model optimization level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationLevel {
    None,
    Light,
    Medium,
    Aggressive,
    Custom(HashMap<String, String>),
}

/// Model priority for edge deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelPriority {
    Critical,
    High,
    Medium,
    Low,
    OnDemand,
}

/// Edge deployment metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeMetrics {
    pub requests_served: u64,
    pub cache_hit_rate: f32,
    pub average_latency_ms: f32,
    pub bandwidth_saved_mb: f64,
    pub offline_requests: u64,
    pub sync_success_rate: f32,
    pub model_accuracy: HashMap<String, f32>,
    pub energy_efficiency: f32,
}

/// Synchronization event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncEvent {
    ModelUpdate {
        model_id: String,
        version: String,
        size_mb: u64,
        checksum: String,
    },
    ModelDelete {
        model_id: String,
    },
    ConfigUpdate {
        config: EdgeConfig,
    },
    HealthCheck,
    MetricsSync {
        metrics: EdgeMetrics,
    },
}

/// A model artifact registered with the orchestrator and available for transfer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelArtifact {
    /// Identifier of the model this artifact belongs to.
    pub model_id: String,
    /// Absolute or relative path of the artifact on the orchestrator host.
    pub source_path: PathBuf,
    /// Size of the artifact in bytes, measured by reading it.
    pub size_bytes: u64,
    /// Lowercase hex SHA-256 digest of the artifact contents.
    pub sha256: String,
}

/// A model artifact that has been transferred onto an edge node.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeployedArtifact {
    /// Node the artifact lives on.
    pub node_id: String,
    /// Model the artifact belongs to.
    pub model_id: String,
    /// Path of the artifact inside the node's storage root.
    pub path: PathBuf,
    /// Bytes actually written to the node.
    pub bytes_written: u64,
    /// SHA-256 digest computed while writing, and re-verified by reading back.
    pub sha256: String,
    /// When the transfer completed.
    pub deployed_at: SystemTime,
}

/// Output of a real offline inference run on an edge node.
#[derive(Debug, Clone)]
pub struct OfflineInferenceOutput {
    /// Text produced by the local model.
    pub text: String,
    /// Confidence reported by the engine. Engines that cannot measure a
    /// confidence must say so rather than inventing one — see
    /// [`OfflineInferenceEngine`].
    pub confidence: f32,
    /// Whether the engine answered from its local result cache.
    pub served_from_cache: bool,
}

/// A local inference runtime that an edge node can execute models on.
///
/// The orchestrator itself owns no model weights and cannot generate text, so
/// [`EdgeOrchestrator::handle_offline_request`] refuses the request unless an
/// engine is installed with [`EdgeOrchestrator::with_offline_engine`]. Every
/// field of the resulting [`InferenceResponse`] comes from the engine or is
/// measured by the orchestrator; none of them is a constant.
///
/// Implementations must report a confidence they actually computed. If the
/// model exposes no confidence signal, return an error rather than a
/// placeholder number.
#[async_trait::async_trait]
pub trait OfflineInferenceEngine: Send + Sync + std::fmt::Debug {
    /// Run `request` against the locally deployed `model`.
    async fn infer(
        &self,
        model: &EdgeModel,
        artifact: Option<&DeployedArtifact>,
        request: &InferenceRequest,
    ) -> Result<OfflineInferenceOutput, EdgeError>;
}

/// Edge deployment orchestrator
pub struct EdgeOrchestrator {
    nodes: Arc<RwLock<HashMap<String, EdgeNode>>>,
    config: EdgeConfig,
    sync_sender: mpsc::Sender<SyncEvent>,
    metrics_collector: Arc<RwLock<EdgeMetrics>>,
    /// Registered source artifacts, keyed by model id.
    artifacts: Arc<RwLock<HashMap<String, ModelArtifact>>>,
    /// Filesystem root for each node's local model store, keyed by node id.
    node_storage: Arc<RwLock<HashMap<String, PathBuf>>>,
    /// Artifacts that have really been transferred, keyed by `"{node}/{model}"`.
    deployed: Arc<RwLock<HashMap<String, DeployedArtifact>>>,
    /// Local runtime used to answer offline inference requests, if installed.
    offline_engine: Option<Arc<dyn OfflineInferenceEngine>>,
}

impl std::fmt::Debug for EdgeOrchestrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EdgeOrchestrator")
            .field("config", &self.config)
            .field("offline_engine", &self.offline_engine.is_some())
            .finish_non_exhaustive()
    }
}

impl EdgeOrchestrator {
    /// Create a new edge orchestrator
    pub fn new(config: EdgeConfig) -> (Self, mpsc::Receiver<SyncEvent>) {
        let (sync_sender, sync_receiver) = mpsc::channel(1000);

        let orchestrator = Self {
            nodes: Arc::new(RwLock::new(HashMap::new())),
            config,
            sync_sender,
            metrics_collector: Arc::new(RwLock::new(EdgeMetrics::default())),
            artifacts: Arc::new(RwLock::new(HashMap::new())),
            node_storage: Arc::new(RwLock::new(HashMap::new())),
            deployed: Arc::new(RwLock::new(HashMap::new())),
            offline_engine: None,
        };

        (orchestrator, sync_receiver)
    }

    /// Install the local inference runtime used to serve offline requests.
    #[must_use]
    pub fn with_offline_engine(mut self, engine: Arc<dyn OfflineInferenceEngine>) -> Self {
        self.offline_engine = Some(engine);
        self
    }

    /// Register the on-disk artifact that backs `model_id`.
    ///
    /// The file is read end to end to measure its real size and SHA-256 digest;
    /// nothing is assumed from the [`EdgeModel::size_mb`] metadata.
    ///
    /// # Errors
    ///
    /// Returns [`EdgeError::ArtifactUnavailable`] if the path cannot be read.
    pub async fn register_model_artifact(
        &self,
        model_id: &str,
        source_path: impl AsRef<Path>,
    ) -> Result<ModelArtifact, EdgeError> {
        let source_path = source_path.as_ref().to_path_buf();
        let (size_bytes, sha256) = hash_file(&source_path).await?;
        let artifact = ModelArtifact {
            model_id: model_id.to_string(),
            source_path,
            size_bytes,
            sha256,
        };
        self.artifacts.write().await.insert(model_id.to_string(), artifact.clone());
        Ok(artifact)
    }

    /// Register the directory that `node_id` uses as its local model store.
    ///
    /// The directory (and any missing parents) is created if necessary.
    ///
    /// # Errors
    ///
    /// Returns [`EdgeError::StorageError`] if the directory cannot be created.
    pub async fn register_node_storage(
        &self,
        node_id: &str,
        root: impl AsRef<Path>,
    ) -> Result<(), EdgeError> {
        let root = root.as_ref().to_path_buf();
        tokio::fs::create_dir_all(&root).await.map_err(|e| {
            EdgeError::StorageError(format!(
                "cannot create storage root {}: {e}",
                root.display()
            ))
        })?;
        self.node_storage.write().await.insert(node_id.to_string(), root);
        Ok(())
    }

    /// Look up the artifact registered for a model, if any.
    pub async fn get_model_artifact(&self, model_id: &str) -> Option<ModelArtifact> {
        self.artifacts.read().await.get(model_id).cloned()
    }

    /// Look up an artifact that has been transferred to a node.
    pub async fn get_deployed_artifact(
        &self,
        node_id: &str,
        model_id: &str,
    ) -> Option<DeployedArtifact> {
        self.deployed.read().await.get(&deployment_key(node_id, model_id)).cloned()
    }

    /// Re-read a deployed artifact from the node's storage and verify that its
    /// SHA-256 digest still matches the digest recorded at transfer time.
    ///
    /// # Errors
    ///
    /// * [`EdgeError::ArtifactUnavailable`] if nothing was deployed, or the file
    ///   has since disappeared.
    /// * [`EdgeError::ChecksumMismatch`] if the contents changed.
    pub async fn verify_deployment(
        &self,
        node_id: &str,
        model_id: &str,
    ) -> Result<DeployedArtifact, EdgeError> {
        let artifact = self.get_deployed_artifact(node_id, model_id).await.ok_or_else(|| {
            EdgeError::ArtifactUnavailable(format!(
                "no artifact of model {model_id} has been deployed to node {node_id}"
            ))
        })?;
        let (size, digest) = hash_file(&artifact.path).await?;
        if digest != artifact.sha256 || size != artifact.bytes_written {
            return Err(EdgeError::ChecksumMismatch {
                expected: artifact.sha256.clone(),
                actual: digest,
            });
        }
        Ok(artifact)
    }

    /// Register a new edge node
    pub async fn register_node(&self, node: EdgeNode) -> Result<(), EdgeError> {
        let mut nodes = self.nodes.write().await;
        nodes.insert(node.id.clone(), node);
        Ok(())
    }

    /// Remove an edge node
    pub async fn unregister_node(&self, node_id: &str) -> Result<(), EdgeError> {
        let mut nodes = self.nodes.write().await;
        nodes.remove(node_id);
        Ok(())
    }

    /// Get edge node information
    pub async fn get_node(&self, node_id: &str) -> Option<EdgeNode> {
        let nodes = self.nodes.read().await;
        nodes.get(node_id).cloned()
    }

    /// List all edge nodes
    pub async fn list_nodes(&self) -> Vec<EdgeNode> {
        let nodes = self.nodes.read().await;
        nodes.values().cloned().collect()
    }

    /// Deploy model to edge nodes
    pub async fn deploy_model(
        &self,
        model: EdgeModel,
        target_nodes: Vec<String>,
    ) -> Result<DeploymentResult, EdgeError> {
        let nodes = self.nodes.read().await;
        let mut results = HashMap::new();

        for node_id in target_nodes {
            if let Some(node) = nodes.get(&node_id) {
                let result = self.deploy_model_to_node(&model, node).await?;
                results.insert(node_id, result);
            }
        }

        let total_deployments = results.len();
        let successful_deployments = results.values().filter(|r| r.success).count();

        Ok(DeploymentResult {
            model_id: model.id.clone(),
            node_results: results,
            total_deployments,
            successful_deployments,
        })
    }

    /// Deploy a model to a specific node by really transferring its artifact.
    ///
    /// The artifact registered for `model.id` is copied chunk by chunk into the
    /// node's storage root, hashed while it is written, and then read back and
    /// re-hashed. The returned [`NodeDeploymentResult`] carries the number of
    /// bytes that were actually written and the verified digest. Nothing is
    /// reported as deployed unless the round trip succeeded.
    async fn deploy_model_to_node(
        &self,
        model: &EdgeModel,
        node: &EdgeNode,
    ) -> Result<NodeDeploymentResult, EdgeError> {
        let start_time = SystemTime::now();

        let failure = |error: String| NodeDeploymentResult {
            node_id: node.id.clone(),
            success: false,
            error: Some(error),
            deployment_time_ms: 0,
            bytes_transferred: 0,
            checksum_sha256: None,
        };

        let Some(artifact) = self.get_model_artifact(&model.id).await else {
            return Ok(failure(format!(
                "no artifact registered for model {}: call register_model_artifact() first",
                model.id
            )));
        };

        let Some(root) = self.node_storage.read().await.get(&node.id).cloned() else {
            return Ok(failure(format!(
                "node {} has no storage root: call register_node_storage() first",
                node.id
            )));
        };

        // Check real capacity against the artifact's measured size, not metadata.
        let required_mb = artifact.size_bytes.div_ceil(1024 * 1024);
        if node.resources.storage_available_mb < required_mb {
            return Ok(failure(format!(
                "insufficient storage on node {}: {} MB available, {} MB required",
                node.id, node.resources.storage_available_mb, required_mb
            )));
        }

        let file_name = artifact
            .source_path
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| format!("{}.bin", model.id));
        let target_dir = root.join(&model.id).join(&model.version);
        let target_path = target_dir.join(file_name);

        match self.transfer_artifact(&artifact, &target_path).await {
            Ok((bytes_written, digest)) => {
                let deployed = DeployedArtifact {
                    node_id: node.id.clone(),
                    model_id: model.id.clone(),
                    path: target_path,
                    bytes_written,
                    sha256: digest.clone(),
                    deployed_at: SystemTime::now(),
                };
                self.deployed
                    .write()
                    .await
                    .insert(deployment_key(&node.id, &model.id), deployed);

                Ok(NodeDeploymentResult {
                    node_id: node.id.clone(),
                    success: true,
                    error: None,
                    deployment_time_ms: start_time.elapsed().unwrap_or_default().as_millis() as u64,
                    bytes_transferred: bytes_written,
                    checksum_sha256: Some(digest),
                })
            },
            Err(e) => Ok(failure(e.to_string())),
        }
    }

    /// Copy `artifact` to `target_path` in [`TRANSFER_CHUNK_BYTES`] chunks,
    /// hashing as it goes, then read the result back and verify the digest.
    async fn transfer_artifact(
        &self,
        artifact: &ModelArtifact,
        target_path: &Path,
    ) -> Result<(u64, String), EdgeError> {
        if let Some(parent) = target_path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                EdgeError::StorageError(format!("cannot create {}: {e}", parent.display()))
            })?;
        }

        let mut source = tokio::fs::File::open(&artifact.source_path).await.map_err(|e| {
            EdgeError::ArtifactUnavailable(format!(
                "cannot open source artifact {}: {e}",
                artifact.source_path.display()
            ))
        })?;
        let mut target = tokio::fs::File::create(target_path).await.map_err(|e| {
            EdgeError::StorageError(format!("cannot create {}: {e}", target_path.display()))
        })?;

        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; TRANSFER_CHUNK_BYTES];
        let mut written: u64 = 0;
        loop {
            let read = source.read(&mut buffer).await.map_err(|e| {
                EdgeError::ArtifactUnavailable(format!(
                    "read failed at byte {written} of {}: {e}",
                    artifact.source_path.display()
                ))
            })?;
            if read == 0 {
                break;
            }
            target.write_all(&buffer[..read]).await.map_err(|e| {
                EdgeError::StorageError(format!(
                    "write failed at byte {written} of {}: {e}",
                    target_path.display()
                ))
            })?;
            hasher.update(&buffer[..read]);
            written += read as u64;
        }
        target.flush().await.map_err(|e| {
            EdgeError::StorageError(format!("flush failed for {}: {e}", target_path.display()))
        })?;
        target.sync_all().await.map_err(|e| {
            EdgeError::StorageError(format!("fsync failed for {}: {e}", target_path.display()))
        })?;
        drop(target);

        let digest = hex::encode(hasher.finalize());
        if digest != artifact.sha256 || written != artifact.size_bytes {
            let _ = tokio::fs::remove_file(target_path).await;
            return Err(EdgeError::ChecksumMismatch {
                expected: artifact.sha256.clone(),
                actual: digest,
            });
        }

        // Read the written file back: this catches truncation and storage
        // corruption that the write-side hash cannot see.
        let (verified_size, verified_digest) = hash_file(target_path).await?;
        if verified_digest != artifact.sha256 || verified_size != artifact.size_bytes {
            let _ = tokio::fs::remove_file(target_path).await;
            return Err(EdgeError::ChecksumMismatch {
                expected: artifact.sha256.clone(),
                actual: verified_digest,
            });
        }

        Ok((written, digest))
    }

    /// Synchronize models across edge nodes.
    ///
    /// This re-verifies every artifact that has really been transferred: each
    /// deployed file is read back and re-hashed. `bytes_transferred` is the
    /// number of bytes actually re-read, and `sync_time_ms` is measured, not
    /// derived from a formula.
    ///
    /// # Errors
    ///
    /// Returns [`EdgeError::SyncError`] if the sync event channel is closed.
    pub async fn sync_models(&self) -> Result<SyncResult, EdgeError> {
        self.sync_sender
            .send(SyncEvent::HealthCheck)
            .await
            .map_err(|e| EdgeError::SyncError(format!("Failed to send sync event: {}", e)))?;

        let node_ids: Vec<(String, Vec<String>)> = {
            let nodes = self.nodes.read().await;
            nodes
                .values()
                .map(|node| {
                    (
                        node.id.clone(),
                        node.models.iter().map(|m| m.id.clone()).collect(),
                    )
                })
                .collect()
        };

        let mut sync_results = Vec::with_capacity(node_ids.len());
        for (node_id, model_ids) in node_ids {
            let started = std::time::Instant::now();
            let mut verified = 0usize;
            let mut bytes = 0u64;
            let mut failed = false;
            for model_id in &model_ids {
                match self.verify_deployment(&node_id, model_id).await {
                    Ok(artifact) => {
                        verified += 1;
                        bytes += artifact.bytes_written;
                    },
                    Err(_) => failed = true,
                }
            }
            sync_results.push(NodeSyncResult {
                node_id,
                success: !failed,
                models_synced: verified,
                bytes_transferred: bytes,
                sync_time_ms: started.elapsed().as_millis() as u64,
            });
        }

        Ok(SyncResult {
            timestamp: SystemTime::now(),
            nodes_synced: sync_results.len(),
            total_models_synced: sync_results.iter().map(|r| r.models_synced).sum(),
            total_bytes_transferred: sync_results.iter().map(|r| r.bytes_transferred).sum(),
            node_results: sync_results,
        })
    }

    /// Optimize edge deployment based on usage patterns
    pub async fn optimize_deployment(&self) -> Result<OptimizationResult, EdgeError> {
        let nodes = self.nodes.read().await;
        let mut optimizations = Vec::new();

        for node in nodes.values() {
            // Analyze model usage patterns
            let low_usage_models: Vec<_> =
                node.models.iter().filter(|m| m.usage_count < 10).cloned().collect();

            if !low_usage_models.is_empty() {
                optimizations.push(EdgeOptimizationAction {
                    node_id: node.id.clone(),
                    action: OptimizationAction::RemoveUnusedModels,
                    models_affected: low_usage_models.into_iter().map(|m| m.id).collect(),
                    estimated_savings_mb: node
                        .models
                        .iter()
                        .filter(|m| m.usage_count < 10)
                        .map(|m| m.size_mb)
                        .sum(),
                });
            }

            // Check for optimization opportunities
            let unoptimized_models: Vec<_> = node
                .models
                .iter()
                .filter(|m| matches!(m.optimization_level, OptimizationLevel::None))
                .cloned()
                .collect();

            if !unoptimized_models.is_empty() {
                optimizations.push(EdgeOptimizationAction {
                    node_id: node.id.clone(),
                    action: OptimizationAction::OptimizeModels,
                    models_affected: unoptimized_models.into_iter().map(|m| m.id).collect(),
                    estimated_savings_mb: node.models.iter()
                        .filter(|m| matches!(m.optimization_level, OptimizationLevel::None))
                        .map(|m| m.size_mb / 2) // Assume 50% compression
                        .sum(),
                });
            }
        }

        Ok(OptimizationResult {
            timestamp: SystemTime::now(),
            optimizations,
            total_potential_savings_mb: nodes
                .values()
                .flat_map(|n| &n.models)
                .filter(|m| {
                    m.usage_count < 10 || matches!(m.optimization_level, OptimizationLevel::None)
                })
                .map(|m| m.size_mb / 3)
                .sum(),
        })
    }

    /// Handle an offline inference request on a specific edge node.
    ///
    /// The request is executed by the installed [`OfflineInferenceEngine`]. With
    /// no engine installed the orchestrator has no way to run a model, so it
    /// returns [`EdgeError::InferenceUnavailable`] instead of echoing the input
    /// back with an invented confidence. `processing_time_ms` is measured around
    /// the real call.
    ///
    /// # Errors
    ///
    /// * [`EdgeError::NodeNotFound`] / [`EdgeError::ModelNotFound`] when the
    ///   node or model is unknown to the orchestrator.
    /// * [`EdgeError::InferenceUnavailable`] when no local runtime is installed.
    /// * Whatever the engine returns when execution fails.
    pub async fn handle_offline_request(
        &self,
        node_id: &str,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, EdgeError> {
        let model = {
            let nodes = self.nodes.read().await;
            let node =
                nodes.get(node_id).ok_or_else(|| EdgeError::NodeNotFound(node_id.to_string()))?;
            node.models
                .iter()
                .find(|m| m.id == request.model_id)
                .cloned()
                .ok_or_else(|| EdgeError::ModelNotFound(request.model_id.clone()))?
        };

        let engine = self.offline_engine.as_ref().ok_or_else(|| {
            EdgeError::InferenceUnavailable(format!(
                "node {node_id} has no local inference runtime: install one with \
                 EdgeOrchestrator::with_offline_engine() before serving offline requests"
            ))
        })?;

        let artifact = self.get_deployed_artifact(node_id, &request.model_id).await;

        let started = std::time::Instant::now();
        let output = engine.infer(&model, artifact.as_ref(), &request).await?;
        let elapsed_ms = started.elapsed().as_millis() as u64;

        {
            let mut metrics = self.metrics_collector.write().await;
            metrics.offline_requests += 1;
            metrics.requests_served += 1;
            if output.served_from_cache {
                let served = metrics.requests_served as f32;
                let hits = metrics.cache_hit_rate * (served - 1.0) + 1.0;
                metrics.cache_hit_rate = hits / served;
            } else if metrics.requests_served > 1 {
                let served = metrics.requests_served as f32;
                metrics.cache_hit_rate = metrics.cache_hit_rate * (served - 1.0) / served;
            }
            let served = metrics.requests_served as f32;
            metrics.average_latency_ms =
                (metrics.average_latency_ms * (served - 1.0) + elapsed_ms as f32) / served;
        }

        Ok(InferenceResponse {
            request_id: request.request_id,
            model_id: model.id,
            result: output.text,
            confidence: output.confidence,
            processing_time_ms: elapsed_ms,
            served_from_cache: output.served_from_cache,
            node_id: node_id.to_string(),
        })
    }

    /// Get edge deployment statistics
    pub async fn get_statistics(&self) -> EdgeStatistics {
        let nodes = self.nodes.read().await;
        let metrics = self.metrics_collector.read().await;

        EdgeStatistics {
            total_nodes: nodes.len(),
            online_nodes: nodes
                .values()
                .filter(|n| matches!(n.status, EdgeNodeStatus::Online))
                .count(),
            total_models: nodes.values().map(|n| n.models.len()).sum(),
            total_storage_used_mb: nodes.values().map(|n| n.resources.storage_used_mb).sum(),
            total_storage_capacity_mb: nodes
                .values()
                .map(|n| n.resources.storage_available_mb + n.resources.storage_used_mb)
                .sum(),
            average_latency_ms: metrics.average_latency_ms,
            total_requests_served: metrics.requests_served,
            cache_hit_rate: metrics.cache_hit_rate,
            bandwidth_saved_mb: metrics.bandwidth_saved_mb,
            offline_request_percentage: if metrics.requests_served > 0 {
                (metrics.offline_requests as f32 / metrics.requests_served as f32) * 100.0
            } else {
                0.0
            },
        }
    }
}

/// Edge deployment error types
#[derive(Debug, thiserror::Error)]
pub enum EdgeError {
    #[error("Node not found: {0}")]
    NodeNotFound(String),
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Synchronization error: {0}")]
    SyncError(String),
    #[error("Deployment error: {0}")]
    DeploymentError(String),
    #[error("Configuration error: {0}")]
    ConfigError(String),
    #[error("Resource error: {0}")]
    ResourceError(String),
    /// The model artifact is missing or unreadable on the orchestrator host.
    #[error("Model artifact unavailable: {0}")]
    ArtifactUnavailable(String),
    /// The edge node's local storage could not be written.
    #[error("Edge storage error: {0}")]
    StorageError(String),
    /// A transferred artifact does not hash to the expected digest.
    #[error("Checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },
    /// No local inference runtime is installed on this orchestrator.
    #[error("Offline inference unavailable: {0}")]
    InferenceUnavailable(String),
}

/// Compute the byte length and lowercase hex SHA-256 digest of a file, reading
/// it in [`TRANSFER_CHUNK_BYTES`] chunks so large artifacts never need to fit in
/// memory.
async fn hash_file(path: &Path) -> Result<(u64, String), EdgeError> {
    let mut file = tokio::fs::File::open(path).await.map_err(|e| {
        EdgeError::ArtifactUnavailable(format!("cannot open {}: {e}", path.display()))
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; TRANSFER_CHUNK_BYTES];
    let mut size: u64 = 0;
    loop {
        let read = file.read(&mut buffer).await.map_err(|e| {
            EdgeError::ArtifactUnavailable(format!("cannot read {}: {e}", path.display()))
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        size += read as u64;
    }
    Ok((size, hex::encode(hasher.finalize())))
}

/// Key used to index deployed artifacts by node and model.
fn deployment_key(node_id: &str, model_id: &str) -> String {
    format!("{node_id}/{model_id}")
}

/// Deployment result
#[derive(Debug, Serialize, Deserialize)]
pub struct DeploymentResult {
    pub model_id: String,
    pub node_results: HashMap<String, NodeDeploymentResult>,
    pub total_deployments: usize,
    pub successful_deployments: usize,
}

/// Node deployment result
#[derive(Debug, Serialize, Deserialize)]
pub struct NodeDeploymentResult {
    pub node_id: String,
    pub success: bool,
    pub error: Option<String>,
    /// Wall-clock duration of the real transfer, in milliseconds.
    pub deployment_time_ms: u64,
    /// Bytes actually written to the node's storage.
    pub bytes_transferred: u64,
    /// SHA-256 digest of the transferred artifact, verified by reading it back.
    pub checksum_sha256: Option<String>,
}

/// Synchronization result
#[derive(Debug, Serialize, Deserialize)]
pub struct SyncResult {
    pub timestamp: SystemTime,
    pub nodes_synced: usize,
    pub total_models_synced: usize,
    pub total_bytes_transferred: u64,
    pub node_results: Vec<NodeSyncResult>,
}

/// Node synchronization result
#[derive(Debug, Serialize, Deserialize)]
pub struct NodeSyncResult {
    pub node_id: String,
    pub success: bool,
    pub models_synced: usize,
    pub bytes_transferred: u64,
    pub sync_time_ms: u64,
}

/// Optimization result
#[derive(Debug, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub timestamp: SystemTime,
    pub optimizations: Vec<EdgeOptimizationAction>,
    pub total_potential_savings_mb: u64,
}

/// Edge optimization action
#[derive(Debug, Serialize, Deserialize)]
pub struct EdgeOptimizationAction {
    pub node_id: String,
    pub action: OptimizationAction,
    pub models_affected: Vec<String>,
    pub estimated_savings_mb: u64,
}

/// Optimization action types
#[derive(Debug, Serialize, Deserialize)]
pub enum OptimizationAction {
    RemoveUnusedModels,
    OptimizeModels,
    CachePopularModels,
    CompressResponses,
    BalanceLoad,
}

/// Inference request for edge processing
#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub request_id: String,
    pub model_id: String,
    pub input: String,
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Inference response from edge node
#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub request_id: String,
    pub model_id: String,
    pub result: String,
    pub confidence: f32,
    pub processing_time_ms: u64,
    pub served_from_cache: bool,
    pub node_id: String,
}

/// Edge deployment statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct EdgeStatistics {
    pub total_nodes: usize,
    pub online_nodes: usize,
    pub total_models: usize,
    pub total_storage_used_mb: u64,
    pub total_storage_capacity_mb: u64,
    pub average_latency_ms: f32,
    pub total_requests_served: u64,
    pub cache_hit_rate: f32,
    pub bandwidth_saved_mb: f64,
    pub offline_request_percentage: f32,
}

impl Default for EdgeMetrics {
    fn default() -> Self {
        Self {
            requests_served: 0,
            cache_hit_rate: 0.0,
            average_latency_ms: 0.0,
            bandwidth_saved_mb: 0.0,
            offline_requests: 0,
            sync_success_rate: 100.0,
            model_accuracy: HashMap::new(),
            energy_efficiency: 1.0,
        }
    }
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self {
            node_id: Uuid::new_v4().to_string(),
            location: "unknown".to_string(),
            storage_capacity_mb: 10240, // 10GB
            memory_capacity_mb: 8192,   // 8GB
            cpu_cores: 4,
            gpu_memory_mb: 0,
            bandwidth_mbps: 100,
            latency_to_central_ms: 50,
            mode: EdgeMode::Hybrid,
            sync_config: SyncConfig::default(),
            optimization: EdgeOptimization::default(),
        }
    }
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            sync_interval_seconds: 3600, // 1 hour
            max_model_size_mb: 5120,     // 5GB
            priority_models: Vec::new(),
            compression_level: 6,
            delta_sync: true,
            bandwidth_throttle_mbps: None,
        }
    }
}

impl Default for EdgeOptimization {
    fn default() -> Self {
        Self {
            quantization: true,
            pruning: false,
            distillation: false,
            cache_optimization: true,
            bandwidth_strategies: vec![
                BandwidthStrategy::Compression,
                BandwidthStrategy::Caching,
                BandwidthStrategy::DeltaUpdates,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_edge_orchestrator_creation() {
        let config = EdgeConfig::default();
        let (orchestrator, _receiver) = EdgeOrchestrator::new(config);

        let stats = orchestrator.get_statistics().await;
        assert_eq!(stats.total_nodes, 0);
    }

    #[tokio::test]
    async fn test_node_registration() {
        let config = EdgeConfig::default();
        let (orchestrator, _receiver) = EdgeOrchestrator::new(config);

        let node = EdgeNode {
            id: "test-node".to_string(),
            location: "test-location".to_string(),
            status: EdgeNodeStatus::Online,
            resources: EdgeResources {
                storage_used_mb: 1000,
                storage_available_mb: 9000,
                memory_used_mb: 2000,
                memory_available_mb: 6000,
                cpu_usage_percent: 50.0,
                gpu_usage_percent: 0.0,
                network_usage_mbps: 10.0,
            },
            models: Vec::new(),
            last_sync: SystemTime::now(),
            metrics: EdgeMetrics::default(),
        };

        orchestrator
            .register_node(node)
            .await
            .expect("registration should succeed in test");

        let stats = orchestrator.get_statistics().await;
        assert_eq!(stats.total_nodes, 1);
        assert_eq!(stats.online_nodes, 1);
    }

    #[tokio::test]
    async fn test_model_deployment() {
        let config = EdgeConfig::default();
        let (orchestrator, _receiver) = EdgeOrchestrator::new(config);

        // Register a node
        let node = EdgeNode {
            id: "test-node".to_string(),
            location: "test-location".to_string(),
            status: EdgeNodeStatus::Online,
            resources: EdgeResources {
                storage_used_mb: 1000,
                storage_available_mb: 9000,
                memory_used_mb: 2000,
                memory_available_mb: 6000,
                cpu_usage_percent: 50.0,
                gpu_usage_percent: 0.0,
                network_usage_mbps: 10.0,
            },
            models: Vec::new(),
            last_sync: SystemTime::now(),
            metrics: EdgeMetrics::default(),
        };

        orchestrator
            .register_node(node)
            .await
            .expect("registration should succeed in test");

        // Deploy a model
        let model = EdgeModel {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            version: "1.0.0".to_string(),
            size_mb: 500,
            format: ModelFormat::Quantized,
            optimization_level: OptimizationLevel::Medium,
            last_updated: SystemTime::now(),
            usage_count: 0,
            priority: ModelPriority::High,
        };

        // Without a registered artifact and storage root there is nothing to
        // transfer, so the deployment must fail loudly instead of reporting a
        // success it never performed.
        let result = orchestrator
            .deploy_model(model, vec!["test-node".to_string()])
            .await
            .expect("async operation should succeed in test");
        assert_eq!(result.total_deployments, 1);
        assert_eq!(
            result.successful_deployments, 0,
            "deployment without an artifact must not report success"
        );
        let node_result =
            result.node_results.get("test-node").expect("node result must be present");
        assert!(node_result
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("no artifact registered"));
        assert_eq!(node_result.bytes_transferred, 0);
    }

    #[tokio::test]
    async fn test_offline_inference() {
        let config = EdgeConfig::default();
        let (orchestrator, _receiver) = EdgeOrchestrator::new(config);

        // Register a node with a model
        let model = EdgeModel {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            version: "1.0.0".to_string(),
            size_mb: 500,
            format: ModelFormat::Quantized,
            optimization_level: OptimizationLevel::Medium,
            last_updated: SystemTime::now(),
            usage_count: 0,
            priority: ModelPriority::High,
        };

        let node = EdgeNode {
            id: "test-node".to_string(),
            location: "test-location".to_string(),
            status: EdgeNodeStatus::Online,
            resources: EdgeResources {
                storage_used_mb: 1000,
                storage_available_mb: 9000,
                memory_used_mb: 2000,
                memory_available_mb: 6000,
                cpu_usage_percent: 50.0,
                gpu_usage_percent: 0.0,
                network_usage_mbps: 10.0,
            },
            models: vec![model],
            last_sync: SystemTime::now(),
            metrics: EdgeMetrics::default(),
        };

        orchestrator
            .register_node(node)
            .await
            .expect("registration should succeed in test");

        // Make an offline inference request
        let request = InferenceRequest {
            request_id: "test-request".to_string(),
            model_id: "test-model".to_string(),
            input: "test input".to_string(),
            parameters: HashMap::new(),
        };

        // Regression test: this used to return
        // `format!("Offline inference result for: {input}")` with a hardcoded
        // 0.95 confidence and 150 ms latency. With no local runtime installed,
        // the orchestrator cannot run a model and must say so.
        let err = orchestrator
            .handle_offline_request("test-node", request)
            .await
            .expect_err("offline inference without a runtime must fail");
        assert!(
            matches!(err, EdgeError::InferenceUnavailable(_)),
            "expected InferenceUnavailable, got {err:?}"
        );
    }

    /// Unique per-test scratch directory under the system temp dir.
    fn temp_root(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir()
            .join("trustformers-edge-tests")
            .join(format!("{label}-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("temp dir must be creatable");
        dir
    }

    async fn write_artifact(path: &std::path::Path, bytes: &[u8]) {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.expect("parent dir");
        }
        tokio::fs::write(path, bytes).await.expect("artifact write");
    }

    async fn cleanup(dir: &std::path::Path) {
        let _ = tokio::fs::remove_dir_all(dir).await;
    }

    /// Deterministic local runtime used to exercise the offline path. It is a
    /// test double for a model runtime, declared under `#[cfg(test)]`: the
    /// production path has no built-in engine and refuses the request instead.
    #[derive(Debug)]
    struct EchoLengthEngine;

    #[async_trait::async_trait]
    impl OfflineInferenceEngine for EchoLengthEngine {
        async fn infer(
            &self,
            model: &EdgeModel,
            artifact: Option<&DeployedArtifact>,
            request: &InferenceRequest,
        ) -> Result<OfflineInferenceOutput, EdgeError> {
            let artifact = artifact.ok_or_else(|| {
                EdgeError::ArtifactUnavailable(format!(
                    "model {} is not present on this node",
                    model.id
                ))
            })?;
            Ok(OfflineInferenceOutput {
                text: format!(
                    "{} bytes of {} answered {} chars",
                    artifact.bytes_written,
                    model.id,
                    request.input.len()
                ),
                confidence: 0.5,
                served_from_cache: false,
            })
        }
    }

    fn make_node(id: &str, loc: &str, status: EdgeNodeStatus) -> EdgeNode {
        EdgeNode {
            id: id.to_string(),
            location: loc.to_string(),
            status,
            resources: EdgeResources {
                storage_used_mb: 1000,
                storage_available_mb: 9000,
                memory_used_mb: 2000,
                memory_available_mb: 6000,
                cpu_usage_percent: 30.0,
                gpu_usage_percent: 0.0,
                network_usage_mbps: 5.0,
            },
            models: Vec::new(),
            last_sync: SystemTime::now(),
            metrics: EdgeMetrics::default(),
        }
    }

    fn make_model(id: &str, size_mb: u64) -> EdgeModel {
        EdgeModel {
            id: id.to_string(),
            name: format!("Model {}", id),
            version: "1.0.0".to_string(),
            size_mb,
            format: ModelFormat::Quantized,
            optimization_level: OptimizationLevel::Medium,
            last_updated: SystemTime::now(),
            usage_count: 0,
            priority: ModelPriority::Medium,
        }
    }

    #[test]
    fn test_default_config() {
        let config = EdgeConfig::default();
        assert!(!config.node_id.is_empty());
        assert!(config.storage_capacity_mb > 0);
        assert!(config.memory_capacity_mb > 0);
    }

    #[tokio::test]
    async fn test_register_multiple_nodes() {
        let config = EdgeConfig::default();
        let (orchestrator, _rx) = EdgeOrchestrator::new(config);

        for i in 0..5 {
            let node = make_node(
                &format!("n-{}", i),
                &format!("loc-{}", i),
                EdgeNodeStatus::Online,
            );
            orchestrator.register_node(node).await.expect("register ok");
        }

        let stats = orchestrator.get_statistics().await;
        assert_eq!(stats.total_nodes, 5);
        assert_eq!(stats.online_nodes, 5);
    }

    #[tokio::test]
    async fn test_register_offline_node() {
        let config = EdgeConfig::default();
        let (orchestrator, _rx) = EdgeOrchestrator::new(config);

        let node = make_node("offline-1", "loc", EdgeNodeStatus::Offline);
        orchestrator.register_node(node).await.expect("register ok");

        let stats = orchestrator.get_statistics().await;
        assert_eq!(stats.total_nodes, 1);
    }

    #[tokio::test]
    async fn test_deploy_model_to_nonexistent_node() {
        let config = EdgeConfig::default();
        let (orchestrator, _rx) = EdgeOrchestrator::new(config);

        let model = make_model("m1", 500);
        let result = orchestrator.deploy_model(model, vec!["nonexistent".to_string()]).await;
        assert!(result.is_ok());
        if let Ok(r) = result {
            // Node doesn't exist so it's skipped entirely
            assert_eq!(r.total_deployments, 0);
            assert_eq!(r.successful_deployments, 0);
        }
    }

    #[tokio::test]
    async fn test_deploy_model_to_multiple_nodes() {
        let config = EdgeConfig::default();
        let (orchestrator, _rx) = EdgeOrchestrator::new(config);

        for i in 0..2 {
            let node = make_node(&format!("mn-{}", i), "loc", EdgeNodeStatus::Online);
            orchestrator.register_node(node).await.expect("register ok");
        }

        let temp = temp_root("multi-node");
        let model = make_model("m2", 1); // Tiny model for fast test
        let source = temp.join("m2.bin");
        write_artifact(&source, &vec![7u8; 4096]).await;
        orchestrator
            .register_model_artifact("m2", &source)
            .await
            .expect("register artifact");
        for i in 0..2 {
            orchestrator
                .register_node_storage(&format!("mn-{i}"), temp.join(format!("node-{i}")))
                .await
                .expect("register storage");
        }

        let nodes = vec!["mn-0".to_string(), "mn-1".to_string()];
        let result = orchestrator.deploy_model(model, nodes).await.expect("deploy ok");
        assert_eq!(result.total_deployments, 2);
        assert_eq!(result.successful_deployments, 2);
        for node_result in result.node_results.values() {
            assert_eq!(node_result.bytes_transferred, 4096);
            assert!(node_result.checksum_sha256.is_some());
        }
        cleanup(&temp).await;
    }

    #[test]
    fn test_edge_node_status_debug() {
        let statuses = vec![
            EdgeNodeStatus::Online,
            EdgeNodeStatus::Offline,
            EdgeNodeStatus::Degraded,
        ];
        for s in statuses {
            assert!(!format!("{:?}", s).is_empty());
        }
    }

    #[test]
    fn test_model_format_debug() {
        let formats = vec![
            ModelFormat::Original,
            ModelFormat::Quantized,
            ModelFormat::Pruned,
            ModelFormat::Distilled,
        ];
        for f in formats {
            assert!(!format!("{:?}", f).is_empty());
        }
    }

    #[test]
    fn test_optimization_level_debug() {
        let levels = vec![
            OptimizationLevel::None,
            OptimizationLevel::Light,
            OptimizationLevel::Medium,
            OptimizationLevel::Aggressive,
            OptimizationLevel::Custom(HashMap::new()),
        ];
        for l in levels {
            assert!(!format!("{:?}", l).is_empty());
        }
    }

    #[test]
    fn test_model_priority_debug() {
        let priorities = vec![
            ModelPriority::Low,
            ModelPriority::Medium,
            ModelPriority::High,
            ModelPriority::Critical,
        ];
        for p in priorities {
            assert!(!format!("{:?}", p).is_empty());
        }
    }

    #[test]
    fn test_edge_metrics_default() {
        let metrics = EdgeMetrics::default();
        assert_eq!(metrics.requests_served, 0);
        assert_eq!(metrics.offline_requests, 0);
        assert!((metrics.cache_hit_rate - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_edge_resources_structure() {
        let resources = EdgeResources {
            storage_used_mb: 500,
            storage_available_mb: 9500,
            memory_used_mb: 1000,
            memory_available_mb: 7000,
            cpu_usage_percent: 25.0,
            gpu_usage_percent: 50.0,
            network_usage_mbps: 100.0,
        };
        assert_eq!(
            resources.storage_used_mb + resources.storage_available_mb,
            10000
        );
    }

    #[tokio::test]
    async fn test_offline_request_nonexistent_node() {
        let config = EdgeConfig::default();
        let (orchestrator, _rx) = EdgeOrchestrator::new(config);

        let request = InferenceRequest {
            request_id: "req-1".to_string(),
            model_id: "m1".to_string(),
            input: "test".to_string(),
            parameters: HashMap::new(),
        };

        let result = orchestrator.handle_offline_request("ghost", request).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_statistics_initial() {
        let config = EdgeConfig::default();
        let (orchestrator, _rx) = EdgeOrchestrator::new(config);
        let stats = orchestrator.get_statistics().await;
        assert_eq!(stats.total_nodes, 0);
        assert_eq!(stats.online_nodes, 0);
    }

    #[tokio::test]
    async fn test_deploy_large_model_insufficient_storage() {
        let config = EdgeConfig::default();
        let (orchestrator, _rx) = EdgeOrchestrator::new(config);

        let mut node = make_node("small-node", "loc", EdgeNodeStatus::Online);
        node.resources.storage_available_mb = 10; // Very little storage
        orchestrator.register_node(node).await.expect("register ok");

        let model = make_model("big-model", 50000);
        let result = orchestrator.deploy_model(model, vec!["small-node".to_string()]).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_inference_request_structure() {
        let mut params = HashMap::new();
        params.insert("temperature".to_string(), serde_json::json!(0.7));
        let req = InferenceRequest {
            request_id: "req-42".to_string(),
            model_id: "model-1".to_string(),
            input: "Hello world".to_string(),
            parameters: params,
        };
        assert_eq!(req.request_id, "req-42");
        assert_eq!(req.parameters.len(), 1);
    }

    #[tokio::test]
    async fn test_list_nodes_empty() {
        let config = EdgeConfig::default();
        let (orchestrator, _rx) = EdgeOrchestrator::new(config);
        let nodes = orchestrator.list_nodes().await;
        assert!(nodes.is_empty());
    }

    #[tokio::test]
    async fn test_list_nodes_populated() {
        let config = EdgeConfig::default();
        let (orchestrator, _rx) = EdgeOrchestrator::new(config);

        for i in 0..3 {
            let node = make_node(
                &format!("ln-{}", i),
                &format!("loc-{}", i),
                EdgeNodeStatus::Online,
            );
            orchestrator.register_node(node).await.expect("register ok");
        }

        let nodes = orchestrator.list_nodes().await;
        assert_eq!(nodes.len(), 3);
    }

    #[tokio::test]
    async fn test_get_node() {
        let config = EdgeConfig::default();
        let (orchestrator, _rx) = EdgeOrchestrator::new(config);

        let node = make_node("gn-1", "loc-1", EdgeNodeStatus::Online);
        orchestrator.register_node(node).await.expect("register ok");

        let found = orchestrator.get_node("gn-1").await;
        assert!(found.is_some());
        let not_found = orchestrator.get_node("ghost").await;
        assert!(not_found.is_none());
    }

    #[tokio::test]
    async fn test_unregister_node() {
        let config = EdgeConfig::default();
        let (orchestrator, _rx) = EdgeOrchestrator::new(config);

        let node = make_node("un-1", "loc", EdgeNodeStatus::Online);
        orchestrator.register_node(node).await.expect("register ok");
        orchestrator.unregister_node("un-1").await.expect("unregister ok");
        let nodes = orchestrator.list_nodes().await;
        assert!(nodes.is_empty());
    }

    #[test]
    fn test_model_format_original() {
        let fmt = ModelFormat::Original;
        assert!(format!("{:?}", fmt).contains("Original"));
    }

    #[test]
    fn test_sync_config_default() {
        let config = SyncConfig::default();
        assert_eq!(config.sync_interval_seconds, 3600);
        assert!(config.delta_sync);
        assert!(config.bandwidth_throttle_mbps.is_none());
    }

    #[test]
    fn test_edge_optimization_default() {
        let opt = EdgeOptimization::default();
        assert!(opt.quantization);
        assert!(!opt.pruning);
        assert!(!opt.distillation);
        assert!(opt.cache_optimization);
    }

    #[tokio::test]
    async fn test_optimize_deployment_empty() {
        let config = EdgeConfig::default();
        let (orchestrator, _rx) = EdgeOrchestrator::new(config);
        let result = orchestrator.optimize_deployment().await.expect("optimize ok");
        assert!(result.optimizations.is_empty());
    }
    // ── Real artifact transfer ────────────────────────────────────────────────

    /// Regression test: `deploy_model_to_node` used to `tokio::time::sleep` for
    /// a fraction of an estimated duration and then report `success: true`
    /// without moving a single byte.
    #[tokio::test]
    async fn test_deploy_transfers_the_real_artifact() {
        let temp = temp_root("transfer");
        let (orchestrator, _rx) = EdgeOrchestrator::new(EdgeConfig::default());
        orchestrator
            .register_node(make_node("n1", "loc", EdgeNodeStatus::Online))
            .await
            .expect("register");

        // A payload larger than one transfer chunk, to exercise the loop.
        let payload: Vec<u8> =
            (0..(TRANSFER_CHUNK_BYTES + 12_345)).map(|i| (i % 251) as u8).collect();
        let source = temp.join("source").join("weights.safetensors");
        write_artifact(&source, &payload).await;

        let artifact = orchestrator
            .register_model_artifact("m1", &source)
            .await
            .expect("artifact registration");
        assert_eq!(artifact.size_bytes, payload.len() as u64);

        let node_root = temp.join("node-store");
        orchestrator.register_node_storage("n1", &node_root).await.expect("storage");

        let result = orchestrator
            .deploy_model(make_model("m1", 8), vec!["n1".to_string()])
            .await
            .expect("deploy");
        assert_eq!(result.successful_deployments, 1);
        let node_result = result.node_results.get("n1").expect("node result");
        assert_eq!(node_result.bytes_transferred, payload.len() as u64);
        assert_eq!(
            node_result.checksum_sha256.as_deref(),
            Some(artifact.sha256.as_str())
        );

        // The bytes really are on the node, and they are the right bytes.
        let deployed = orchestrator
            .get_deployed_artifact("n1", "m1")
            .await
            .expect("deployed artifact must be recorded");
        assert!(deployed.path.starts_with(&node_root));
        let on_disk = tokio::fs::read(&deployed.path).await.expect("read back");
        assert_eq!(
            on_disk, payload,
            "transferred bytes must match the source exactly"
        );

        orchestrator.verify_deployment("n1", "m1").await.expect("verification");
        cleanup(&temp).await;
    }

    #[tokio::test]
    async fn test_deploy_without_storage_root_fails_honestly() {
        let temp = temp_root("no-storage");
        let (orchestrator, _rx) = EdgeOrchestrator::new(EdgeConfig::default());
        orchestrator
            .register_node(make_node("n1", "loc", EdgeNodeStatus::Online))
            .await
            .expect("register");
        let source = temp.join("m.bin");
        write_artifact(&source, b"payload").await;
        orchestrator.register_model_artifact("m1", &source).await.expect("artifact");

        let result = orchestrator
            .deploy_model(make_model("m1", 1), vec!["n1".to_string()])
            .await
            .expect("deploy");
        assert_eq!(result.successful_deployments, 0);
        assert!(result.node_results["n1"]
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("no storage root"));
        cleanup(&temp).await;
    }

    #[tokio::test]
    async fn test_register_missing_artifact_errors() {
        let (orchestrator, _rx) = EdgeOrchestrator::new(EdgeConfig::default());
        let missing = std::env::temp_dir().join(format!("absent-{}", Uuid::new_v4()));
        let err = orchestrator
            .register_model_artifact("m1", &missing)
            .await
            .expect_err("missing artifact must error");
        assert!(matches!(err, EdgeError::ArtifactUnavailable(_)), "{err:?}");
    }

    #[tokio::test]
    async fn test_verify_deployment_detects_corruption() {
        let temp = temp_root("corrupt");
        let (orchestrator, _rx) = EdgeOrchestrator::new(EdgeConfig::default());
        orchestrator
            .register_node(make_node("n1", "loc", EdgeNodeStatus::Online))
            .await
            .expect("register");
        let source = temp.join("m.bin");
        write_artifact(&source, b"the original model weights").await;
        orchestrator.register_model_artifact("m1", &source).await.expect("artifact");
        orchestrator
            .register_node_storage("n1", temp.join("store"))
            .await
            .expect("storage");
        orchestrator
            .deploy_model(make_model("m1", 1), vec!["n1".to_string()])
            .await
            .expect("deploy");

        let deployed = orchestrator.get_deployed_artifact("n1", "m1").await.expect("deployed");
        tokio::fs::write(&deployed.path, b"the tampered model weights")
            .await
            .expect("tamper");

        let err = orchestrator
            .verify_deployment("n1", "m1")
            .await
            .expect_err("tampering must be detected");
        assert!(matches!(err, EdgeError::ChecksumMismatch { .. }), "{err:?}");
        cleanup(&temp).await;
    }

    #[tokio::test]
    async fn test_deploy_rejects_artifact_larger_than_free_storage() {
        let temp = temp_root("capacity");
        let (orchestrator, _rx) = EdgeOrchestrator::new(EdgeConfig::default());
        let mut node = make_node("n1", "loc", EdgeNodeStatus::Online);
        node.resources.storage_available_mb = 0;
        orchestrator.register_node(node).await.expect("register");
        let source = temp.join("m.bin");
        write_artifact(&source, &vec![0u8; 2 * 1024 * 1024]).await;
        orchestrator.register_model_artifact("m1", &source).await.expect("artifact");
        orchestrator
            .register_node_storage("n1", temp.join("store"))
            .await
            .expect("storage");

        let result = orchestrator
            .deploy_model(make_model("m1", 2), vec!["n1".to_string()])
            .await
            .expect("deploy");
        assert_eq!(result.successful_deployments, 0);
        assert!(result.node_results["n1"]
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("insufficient storage"));
        cleanup(&temp).await;
    }

    /// Regression test: `sync_models` used to derive `bytes_transferred` from
    /// `size_mb * 1024 * 1024` and `sync_time_ms` from a formula, for models
    /// that had never been transferred anywhere.
    #[tokio::test]
    async fn test_sync_reports_only_verified_artifacts() {
        let temp = temp_root("sync");
        let (orchestrator, mut rx) = EdgeOrchestrator::new(EdgeConfig::default());

        let mut node = make_node("n1", "loc", EdgeNodeStatus::Online);
        node.models = vec![make_model("m1", 1), make_model("m2", 1)];
        orchestrator.register_node(node).await.expect("register");

        let source = temp.join("m1.bin");
        write_artifact(&source, &vec![3u8; 2048]).await;
        orchestrator.register_model_artifact("m1", &source).await.expect("artifact");
        orchestrator
            .register_node_storage("n1", temp.join("store"))
            .await
            .expect("storage");
        orchestrator
            .deploy_model(make_model("m1", 1), vec!["n1".to_string()])
            .await
            .expect("deploy");

        let sync = orchestrator.sync_models().await.expect("sync");
        assert_eq!(sync.nodes_synced, 1);
        assert_eq!(
            sync.total_models_synced, 1,
            "only the model that was really transferred may be counted"
        );
        assert_eq!(sync.total_bytes_transferred, 2048);
        assert!(!sync.node_results[0].success, "m2 was never transferred");
        assert!(rx.try_recv().is_ok(), "a sync event must have been emitted");
        cleanup(&temp).await;
    }

    // ── Offline inference ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_offline_inference_uses_the_installed_engine() {
        let temp = temp_root("offline");
        let (orchestrator, _rx) = EdgeOrchestrator::new(EdgeConfig::default());
        let orchestrator = orchestrator.with_offline_engine(Arc::new(EchoLengthEngine));

        let mut node = make_node("n1", "loc", EdgeNodeStatus::Online);
        node.models = vec![make_model("m1", 1)];
        orchestrator.register_node(node).await.expect("register");

        let source = temp.join("m1.bin");
        write_artifact(&source, &vec![9u8; 1024]).await;
        orchestrator.register_model_artifact("m1", &source).await.expect("artifact");
        orchestrator
            .register_node_storage("n1", temp.join("store"))
            .await
            .expect("storage");
        orchestrator
            .deploy_model(make_model("m1", 1), vec!["n1".to_string()])
            .await
            .expect("deploy");

        let response = orchestrator
            .handle_offline_request(
                "n1",
                InferenceRequest {
                    request_id: "r1".to_string(),
                    model_id: "m1".to_string(),
                    input: "hello".to_string(),
                    parameters: HashMap::new(),
                },
            )
            .await
            .expect("inference");
        assert_eq!(response.result, "1024 bytes of m1 answered 5 chars");
        assert!(
            (response.confidence - 0.5).abs() < f32::EPSILON,
            "confidence must come from the engine, not a constant 0.95"
        );

        let stats = orchestrator.get_statistics().await;
        assert_eq!(stats.total_requests_served, 1);
        cleanup(&temp).await;
    }

    #[tokio::test]
    async fn test_offline_inference_without_deployed_artifact_errors() {
        let (orchestrator, _rx) = EdgeOrchestrator::new(EdgeConfig::default());
        let orchestrator = orchestrator.with_offline_engine(Arc::new(EchoLengthEngine));
        let mut node = make_node("n1", "loc", EdgeNodeStatus::Online);
        node.models = vec![make_model("m1", 1)];
        orchestrator.register_node(node).await.expect("register");

        let err = orchestrator
            .handle_offline_request(
                "n1",
                InferenceRequest {
                    request_id: "r1".to_string(),
                    model_id: "m1".to_string(),
                    input: "hello".to_string(),
                    parameters: HashMap::new(),
                },
            )
            .await
            .expect_err("engine must refuse to answer without the model present");
        assert!(matches!(err, EdgeError::ArtifactUnavailable(_)), "{err:?}");

        // A refused request must not be counted as served.
        assert_eq!(orchestrator.get_statistics().await.total_requests_served, 0);
    }

    #[tokio::test]
    async fn test_offline_inference_unknown_model_errors() {
        let (orchestrator, _rx) = EdgeOrchestrator::new(EdgeConfig::default());
        let orchestrator = orchestrator.with_offline_engine(Arc::new(EchoLengthEngine));
        orchestrator
            .register_node(make_node("n1", "loc", EdgeNodeStatus::Online))
            .await
            .expect("register");
        let err = orchestrator
            .handle_offline_request(
                "n1",
                InferenceRequest {
                    request_id: "r1".to_string(),
                    model_id: "absent".to_string(),
                    input: "hello".to_string(),
                    parameters: HashMap::new(),
                },
            )
            .await
            .expect_err("unknown model must error");
        assert!(matches!(err, EdgeError::ModelNotFound(_)), "{err:?}");
    }
}
