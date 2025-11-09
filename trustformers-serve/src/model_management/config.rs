//! Model Management Configuration

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for the model management system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManagementConfig {
    /// Maximum number of models that can be loaded simultaneously
    pub max_loaded_models: usize,

    /// Default timeout for model loading operations
    pub load_timeout: Duration,

    /// Default timeout for model unloading operations
    pub unload_timeout: Duration,

    /// Interval for health checks and metrics collection
    pub health_check_interval: Duration,

    /// Interval for automatic cleanup of old model versions
    pub cleanup_interval: Duration,

    /// Maximum number of versions to keep per model
    pub max_versions_per_model: usize,

    /// Directory for storing model metadata
    pub metadata_dir: String,

    /// Directory for caching downloaded models
    pub cache_dir: String,

    /// Canary deployment configuration
    pub canary_config: CanaryConfig,

    /// Blue-green deployment configuration
    pub blue_green_config: BlueGreenConfig,

    /// A/B testing configuration
    pub ab_test_config: ABTestConfig,

    /// Resource constraints
    pub resource_limits: ResourceLimits,
}

impl Default for ModelManagementConfig {
    fn default() -> Self {
        Self {
            max_loaded_models: 5,
            load_timeout: Duration::from_secs(300), // 5 minutes
            unload_timeout: Duration::from_secs(60), // 1 minute
            health_check_interval: Duration::from_secs(30),
            cleanup_interval: Duration::from_secs(3600), // 1 hour
            max_versions_per_model: 3,
            metadata_dir: "./model_metadata".to_string(),
            cache_dir: "./model_cache".to_string(),
            canary_config: CanaryConfig::default(),
            blue_green_config: BlueGreenConfig::default(),
            ab_test_config: ABTestConfig::default(),
            resource_limits: ResourceLimits::default(),
        }
    }
}

/// Configuration for canary deployments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanaryConfig {
    /// Default canary percentage
    pub default_percentage: f32,

    /// Minimum percentage for canary deployment
    pub min_percentage: f32,

    /// Maximum percentage for canary deployment
    pub max_percentage: f32,

    /// Step size for gradual rollout
    pub step_size: f32,

    /// Duration to wait between rollout steps
    pub step_duration: Duration,

    /// Success threshold to proceed to next step
    pub success_threshold: f32,

    /// Error threshold to abort canary deployment
    pub error_threshold: f32,

    /// Automatic rollback on failure
    pub auto_rollback: bool,
}

impl Default for CanaryConfig {
    fn default() -> Self {
        Self {
            default_percentage: 5.0,
            min_percentage: 1.0,
            max_percentage: 50.0,
            step_size: 5.0,
            step_duration: Duration::from_secs(300), // 5 minutes
            success_threshold: 0.95,
            error_threshold: 0.05,
            auto_rollback: true,
        }
    }
}

/// Configuration for blue-green deployments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlueGreenConfig {
    /// Warm-up requests to send to green environment
    pub warmup_requests: usize,

    /// Validation checks before switching
    pub validation_checks: Vec<String>,

    /// Timeout for validation
    pub validation_timeout: Duration,

    /// Keep old version for rollback
    pub keep_old_version: bool,

    /// Automatic rollback on validation failure
    pub auto_rollback: bool,
}

impl Default for BlueGreenConfig {
    fn default() -> Self {
        Self {
            warmup_requests: 10,
            validation_checks: vec!["health".to_string(), "latency".to_string()],
            validation_timeout: Duration::from_secs(60),
            keep_old_version: true,
            auto_rollback: true,
        }
    }
}

/// Configuration for A/B testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestConfig {
    /// Maximum number of variants in an A/B test
    pub max_variants: usize,

    /// Default traffic split for variants
    pub default_traffic_split: f32,

    /// Minimum sample size for statistical significance
    pub min_sample_size: usize,

    /// Confidence level for statistical tests
    pub confidence_level: f32,

    /// Duration to run A/B tests
    pub test_duration: Duration,

    /// Metrics to track for comparison
    pub tracked_metrics: Vec<String>,
}

impl Default for ABTestConfig {
    fn default() -> Self {
        Self {
            max_variants: 4,
            default_traffic_split: 0.5,
            min_sample_size: 1000,
            confidence_level: 0.95,
            test_duration: Duration::from_secs(7 * 24 * 3600), // 1 week
            tracked_metrics: vec![
                "latency".to_string(),
                "accuracy".to_string(),
                "throughput".to_string(),
            ],
        }
    }
}

/// Resource constraints for model management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum total memory usage in bytes
    pub max_memory_bytes: u64,

    /// Maximum GPU memory usage in bytes (per GPU)
    pub max_gpu_memory_bytes: Option<u64>,

    /// Maximum number of CPU cores to use
    pub max_cpu_cores: Option<usize>,

    /// Maximum disk space for model cache in bytes
    pub max_cache_size_bytes: u64,

    /// Memory safety buffer (percentage)
    pub memory_safety_buffer: f32,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_bytes: 32 * 1024 * 1024 * 1024, // 32 GB
            max_gpu_memory_bytes: Some(24 * 1024 * 1024 * 1024), // 24 GB
            max_cpu_cores: None,                       // Use all available
            max_cache_size_bytes: 100 * 1024 * 1024 * 1024, // 100 GB
            memory_safety_buffer: 0.1,                 // 10% buffer
        }
    }
}

/// Model loading strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LoadingStrategy {
    /// Load model on-demand when first requested
    Lazy,

    /// Load model immediately when added to registry
    Eager,

    /// Pre-load model in background
    Preload,

    /// Load multiple models in parallel
    Parallel { concurrency: usize },
}

impl Default for LoadingStrategy {
    fn default() -> Self {
        LoadingStrategy::Lazy
    }
}

/// Model unloading strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum UnloadingStrategy {
    /// Immediately unload when requested
    Immediate,

    /// Gracefully drain existing requests before unloading
    Graceful { timeout: Duration },

    /// Keep model in memory for quick reloading
    Cached { ttl: Duration },
}

impl Default for UnloadingStrategy {
    fn default() -> Self {
        UnloadingStrategy::Graceful {
            timeout: Duration::from_secs(30),
        }
    }
}
