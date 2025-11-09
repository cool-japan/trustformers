//! Configuration management for network adaptation parameters.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Network adaptation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkAdaptationConfig {
    /// Enable network condition monitoring
    pub enable_monitoring: bool,
    /// Network monitoring interval (ms)
    pub monitoring_interval_ms: u64,
    /// Enable adaptive scheduling
    pub enable_adaptive_scheduling: bool,
    /// Enable bandwidth optimization
    pub enable_bandwidth_optimization: bool,
    /// Network quality thresholds
    pub quality_thresholds: NetworkQualityThresholds,
    /// Communication strategy settings
    pub communication_strategy: CommunicationStrategy,
    /// Data usage limits
    pub data_usage_limits: DataUsageLimits,
    /// Sync frequency settings
    pub sync_frequency: SyncFrequencyConfig,
    /// Failure recovery settings
    pub failure_recovery: FailureRecoveryConfig,
    /// Network prediction settings
    pub prediction_config: NetworkPredictionConfig,
}

/// Network quality thresholds for different actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkQualityThresholds {
    /// Minimum bandwidth for full sync (Mbps)
    pub min_bandwidth_full_sync_mbps: f32,
    /// Minimum bandwidth for incremental sync (Mbps)
    pub min_bandwidth_incremental_sync_mbps: f32,
    /// Maximum latency for real-time updates (ms)
    pub max_latency_realtime_ms: f32,
    /// Maximum packet loss for reliable sync (%)
    pub max_packet_loss_percent: f32,
    /// Signal strength threshold (dBm)
    pub min_signal_strength_dbm: i32,
    /// Jitter tolerance (ms)
    pub max_jitter_ms: f32,
}

/// Communication strategy for different network conditions
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CommunicationStrategy {
    /// Strategy for WiFi networks
    pub wifi_strategy: WiFiStrategy,
    /// Strategy for cellular networks
    pub cellular_strategy: CellularStrategy,
    /// Strategy for poor network conditions
    pub poor_network_strategy: PoorNetworkStrategy,
    /// Compression settings
    pub compression_config: NetworkCompressionConfig,
    /// Retry settings
    pub retry_config: RetryConfig,
}

/// WiFi-specific communication strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WiFiStrategy {
    /// Enable high-frequency updates
    pub enable_high_frequency_updates: bool,
    /// Maximum model size for full sync (MB)
    pub max_full_sync_size_mb: usize,
    /// Preferred sync window (hours)
    pub preferred_sync_window_hours: Vec<u8>,
    /// Enable background sync
    pub enable_background_sync: bool,
    /// Concurrent connection limit
    pub max_concurrent_connections: usize,
}

/// Cellular-specific communication strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellularStrategy {
    /// Enable cellular sync
    pub enable_cellular_sync: bool,
    /// 5G-specific settings
    pub g5_config: CellularConfig,
    /// 4G-specific settings
    pub g4_config: CellularConfig,
    /// Data usage awareness
    pub data_usage_awareness: DataUsageAwareness,
    /// Time-based scheduling
    pub time_based_scheduling: TimeBasedScheduling,
}

/// Cellular network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellularConfig {
    /// Maximum sync size (MB)
    pub max_sync_size_mb: usize,
    /// Sync frequency (hours)
    pub sync_frequency_hours: u32,
    /// Enable compression
    pub enable_compression: bool,
    /// Compression ratio target
    pub compression_ratio_target: f32,
    /// Enable delta sync only
    pub delta_sync_only: bool,
}

/// Data usage awareness settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataUsageAwareness {
    /// Track daily data usage
    pub track_daily_usage: bool,
    /// Daily data limit (MB)
    pub daily_limit_mb: usize,
    /// Monthly data limit (MB)
    pub monthly_limit_mb: usize,
    /// Warning threshold (%)
    pub warning_threshold_percent: u8,
    /// Adaptive quality based on usage
    pub adaptive_quality: bool,
}

/// Time-based scheduling for cellular networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeBasedScheduling {
    /// Preferred hours for sync (0-23)
    pub preferred_hours: Vec<u8>,
    /// Avoid peak hours
    pub avoid_peak_hours: bool,
    /// Peak hours definition (0-23)
    pub peak_hours: Vec<u8>,
    /// Off-peak bonus multiplier
    pub off_peak_multiplier: f32,
}

/// Poor network condition strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoorNetworkStrategy {
    /// Enable degraded mode
    pub enable_degraded_mode: bool,
    /// Minimum viable update size (KB)
    pub min_update_size_kb: usize,
    /// Extended retry intervals
    pub extended_retry_intervals: Vec<u64>,
    /// Enable store-and-forward
    pub enable_store_and_forward: bool,
    /// Maximum queue size (MB)
    pub max_queue_size_mb: usize,
    /// Fallback to local training only
    pub fallback_local_only: bool,
}

/// Network compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkCompressionConfig {
    /// Enable gradient compression
    pub enable_gradient_compression: bool,
    /// Gradient compression algorithm
    pub gradient_compression_algo: GradientCompressionAlgorithm,
    /// Model compression for sync
    pub model_compression_ratio: f32,
    /// Enable differential compression
    pub enable_differential_compression: bool,
    /// Quantization settings for network transfer
    pub network_quantization: NetworkQuantizationConfig,
}

/// Gradient compression algorithms
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GradientCompressionAlgorithm {
    /// No compression
    None,
    /// Top-K compression
    TopK { k: usize },
    /// Random sparsification
    RandomSparsification { ratio: f32 },
    /// Quantized compression
    Quantized { bits: u8 },
    /// Adaptive compression
    Adaptive,
}

/// Network quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkQuantizationConfig {
    /// Enable quantization for network transfer
    pub enabled: bool,
    /// Bits per parameter
    pub bits_per_parameter: u8,
    /// Dynamic range scaling
    pub dynamic_range_scaling: bool,
    /// Quantization scheme
    pub scheme: QuantizationScheme,
}

/// Quantization schemes
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum QuantizationScheme {
    Uniform,
    NonUniform,
    Logarithmic,
    Adaptive,
}

/// Retry configuration for network operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_attempts: usize,
    /// Base retry interval (ms)
    pub base_interval_ms: u64,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f32,
    /// Maximum retry interval (ms)
    pub max_interval_ms: u64,
    /// Jitter percentage (0.0-1.0)
    pub jitter_percent: f32,
}

/// Data usage limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataUsageLimits {
    /// Daily data usage limit (MB)
    pub daily_limit_mb: usize,
    /// Monthly data usage limit (MB)
    pub monthly_limit_mb: usize,
    /// Per-session limit (MB)
    pub session_limit_mb: usize,
    /// Warning thresholds (%)
    pub warning_thresholds: Vec<u8>,
    /// Emergency stop threshold (%)
    pub emergency_stop_threshold: u8,
}

/// Sync frequency configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncFrequencyConfig {
    /// Base frequency (minutes)
    pub base_frequency_minutes: u32,
    /// Adaptive frequency enabled
    pub adaptive_frequency: bool,
    /// Minimum frequency (minutes)
    pub min_frequency_minutes: u32,
    /// Maximum frequency (minutes)
    pub max_frequency_minutes: u32,
    /// Conditions for frequency adjustment
    pub frequency_conditions: HashMap<String, f32>,
}

/// Failure recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureRecoveryConfig {
    /// Enable automatic recovery
    pub enable_auto_recovery: bool,
    /// Recovery timeout (ms)
    pub recovery_timeout_ms: u64,
    /// Maximum recovery attempts
    pub max_recovery_attempts: usize,
    /// Recovery strategies
    pub recovery_strategies: Vec<RecoveryStrategy>,
    /// Fallback to offline mode
    pub fallback_offline: bool,
}

/// Recovery strategies for network failures
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    Retry,
    Reconnect,
    Degrade,
    Queue,
    Offline,
}

/// Network prediction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPredictionConfig {
    /// Enable prediction
    pub enabled: bool,
    /// Prediction window (minutes)
    pub prediction_window_minutes: u32,
    /// Historical data window (hours)
    pub history_window_hours: u32,
    /// Prediction accuracy threshold
    pub accuracy_threshold: f32,
    /// Model update frequency (hours)
    pub model_update_frequency_hours: u32,
}

// Default implementations
impl Default for NetworkAdaptationConfig {
    fn default() -> Self {
        Self {
            enable_monitoring: true,
            monitoring_interval_ms: 10000, // 10 seconds
            enable_adaptive_scheduling: true,
            enable_bandwidth_optimization: true,
            quality_thresholds: NetworkQualityThresholds::default(),
            communication_strategy: CommunicationStrategy::default(),
            data_usage_limits: DataUsageLimits::default(),
            sync_frequency: SyncFrequencyConfig::default(),
            failure_recovery: FailureRecoveryConfig::default(),
            prediction_config: NetworkPredictionConfig::default(),
        }
    }
}

impl Default for NetworkQualityThresholds {
    fn default() -> Self {
        Self {
            min_bandwidth_full_sync_mbps: 5.0,
            min_bandwidth_incremental_sync_mbps: 1.0,
            max_latency_realtime_ms: 100.0,
            max_packet_loss_percent: 2.0,
            min_signal_strength_dbm: -70,
            max_jitter_ms: 50.0,
        }
    }
}

impl Default for WiFiStrategy {
    fn default() -> Self {
        Self {
            enable_high_frequency_updates: true,
            max_full_sync_size_mb: 100,
            preferred_sync_window_hours: vec![2, 3, 4, 5], // 2-5 AM
            enable_background_sync: true,
            max_concurrent_connections: 3,
        }
    }
}

impl Default for CellularStrategy {
    fn default() -> Self {
        Self {
            enable_cellular_sync: true,
            g5_config: CellularConfig {
                max_sync_size_mb: 50,
                sync_frequency_hours: 6,
                enable_compression: true,
                compression_ratio_target: 0.7,
                delta_sync_only: false,
            },
            g4_config: CellularConfig {
                max_sync_size_mb: 20,
                sync_frequency_hours: 12,
                enable_compression: true,
                compression_ratio_target: 0.5,
                delta_sync_only: true,
            },
            data_usage_awareness: DataUsageAwareness::default(),
            time_based_scheduling: TimeBasedScheduling::default(),
        }
    }
}

impl Default for DataUsageAwareness {
    fn default() -> Self {
        Self {
            track_daily_usage: true,
            daily_limit_mb: 100,
            monthly_limit_mb: 2000,
            warning_threshold_percent: 80,
            adaptive_quality: true,
        }
    }
}

impl Default for TimeBasedScheduling {
    fn default() -> Self {
        Self {
            preferred_hours: vec![2, 3, 4, 5, 6],
            avoid_peak_hours: true,
            peak_hours: vec![8, 9, 10, 17, 18, 19, 20],
            off_peak_multiplier: 1.5,
        }
    }
}

impl Default for PoorNetworkStrategy {
    fn default() -> Self {
        Self {
            enable_degraded_mode: true,
            min_update_size_kb: 10,
            extended_retry_intervals: vec![5000, 10000, 20000, 40000],
            enable_store_and_forward: true,
            max_queue_size_mb: 50,
            fallback_local_only: true,
        }
    }
}

impl Default for NetworkCompressionConfig {
    fn default() -> Self {
        Self {
            enable_gradient_compression: true,
            gradient_compression_algo: GradientCompressionAlgorithm::Adaptive,
            model_compression_ratio: 0.8,
            enable_differential_compression: true,
            network_quantization: NetworkQuantizationConfig::default(),
        }
    }
}

impl Default for NetworkQuantizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            bits_per_parameter: 16,
            dynamic_range_scaling: true,
            scheme: QuantizationScheme::Adaptive,
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_interval_ms: 1000,
            backoff_multiplier: 2.0,
            max_interval_ms: 10000,
            jitter_percent: 0.1,
        }
    }
}

impl Default for DataUsageLimits {
    fn default() -> Self {
        Self {
            daily_limit_mb: 100,
            monthly_limit_mb: 2000,
            session_limit_mb: 50,
            warning_thresholds: vec![50, 75, 90],
            emergency_stop_threshold: 95,
        }
    }
}

impl Default for SyncFrequencyConfig {
    fn default() -> Self {
        Self {
            base_frequency_minutes: 60,
            adaptive_frequency: true,
            min_frequency_minutes: 15,
            max_frequency_minutes: 240,
            frequency_conditions: HashMap::new(),
        }
    }
}

impl Default for FailureRecoveryConfig {
    fn default() -> Self {
        Self {
            enable_auto_recovery: true,
            recovery_timeout_ms: 30000,
            max_recovery_attempts: 3,
            recovery_strategies: vec![
                RecoveryStrategy::Retry,
                RecoveryStrategy::Reconnect,
                RecoveryStrategy::Degrade,
            ],
            fallback_offline: true,
        }
    }
}

impl Default for NetworkPredictionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            prediction_window_minutes: 30,
            history_window_hours: 24,
            accuracy_threshold: 0.8,
            model_update_frequency_hours: 6,
        }
    }
}
