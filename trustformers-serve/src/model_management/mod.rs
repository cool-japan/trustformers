//! Model Management System
//!
//! Provides hot model swapping, versioning, canary deployments, and blue-green deployments
//! for production inference serving.

pub mod config;
pub mod deployment;
pub mod manager;
pub mod registry;
pub mod versioning;

pub use config::*;
pub use deployment::*;
pub use manager::*;
pub use registry::*;
pub use versioning::*;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model deployment strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DeploymentStrategy {
    /// Immediate replacement of the current model
    Replace,
    /// Gradual rollout to a percentage of traffic
    Canary { percentage: f32 },
    /// Blue-green deployment with instant switching
    BlueGreen,
    /// A/B testing between multiple models
    ABTest { variants: Vec<String> },
}

/// Model status in the registry
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelStatus {
    /// Model is being loaded
    Loading,
    /// Model is active and serving traffic
    Active,
    /// Model is in standby (loaded but not serving traffic)
    Standby,
    /// Model is being drained (no new requests, finishing existing ones)
    Draining,
    /// Model has been unloaded
    Unloaded,
    /// Model failed to load or encountered an error
    Failed { error: String },
}

/// Model metadata for tracking and management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Unique model identifier
    pub id: String,
    /// Human-readable model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Path to model files
    pub path: String,
    /// Model configuration as JSON string
    pub config: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last updated timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// Current status
    pub status: ModelStatus,
    /// Deployment strategy
    pub deployment_strategy: DeploymentStrategy,
    /// Tags for categorization
    pub tags: HashMap<String, String>,
    /// Model metrics
    pub metrics: ModelMetrics,
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelMetrics {
    /// Total number of requests served
    pub total_requests: u64,
    /// Total number of successful requests
    pub successful_requests: u64,
    /// Total number of failed requests
    pub failed_requests: u64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f32,
    /// P95 latency in milliseconds
    pub p95_latency_ms: f32,
    /// P99 latency in milliseconds
    pub p99_latency_ms: f32,
    /// Tokens processed per second
    pub tokens_per_second: f32,
    /// Memory usage in bytes
    pub memory_usage_bytes: u64,
    /// GPU memory usage in bytes (if applicable)
    pub gpu_memory_usage_bytes: Option<u64>,
}

/// Model loading configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLoadConfig {
    /// Model path or Hub identifier
    pub model_path: String,
    /// Optional revision/branch
    pub revision: Option<String>,
    /// Model precision (fp32, fp16, int8, etc.)
    pub precision: String,
    /// Device placement (cpu, cuda:0, etc.)
    pub device: String,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// KV cache size
    pub kv_cache_size: Option<usize>,
    /// Additional model-specific configuration
    pub config_overrides: HashMap<String, serde_json::Value>,
}

/// Result type for model management operations
pub type ModelResult<T> = Result<T, ModelError>;

/// Model management errors
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Model not found: {id}")]
    ModelNotFound { id: String },

    #[error("Model already exists: {id}")]
    ModelAlreadyExists { id: String },

    #[error("Model loading failed: {error}")]
    LoadingFailed { error: String },

    #[error("Model deployment failed: {error}")]
    DeploymentFailed { error: String },

    #[error("Invalid model configuration: {error}")]
    InvalidConfig { error: String },

    #[error("Version conflict: {current} vs {requested}")]
    VersionConflict { current: String, requested: String },

    #[error("Deployment strategy not supported: {strategy:?}")]
    UnsupportedStrategy { strategy: DeploymentStrategy },

    #[error("Resource constraint: {constraint}")]
    ResourceConstraint { constraint: String },

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}
