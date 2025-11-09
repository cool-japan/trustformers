//! Core types, enums, and data structures for network adaptation.
//!
//! This module provides comprehensive type definitions for network condition adaptation
//! in federated learning scenarios, including configuration structures, enums for
//! various strategies and states, and core data structures for tracking network
//! performance and task execution.

use crate::{battery::BatteryLevel, profiler::NetworkConnectionType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Instant, SystemTime};

// =============================================================================
// NETWORK QUALITY AND CONDITIONS
// =============================================================================

/// Network quality assessment levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum NetworkQuality {
    Poor,
    Fair,
    Good,
    Excellent,
}

/// Trend directions for network metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Volatile,
}

/// Current network conditions snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConditions {
    /// Timestamp of measurement
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    /// Connection type
    pub connection_type: NetworkConnectionType,
    /// Bandwidth (Mbps)
    pub bandwidth_mbps: f32,
    /// Latency (ms)
    pub latency_ms: f32,
    /// Packet loss (%)
    pub packet_loss_percent: f32,
    /// Signal strength (dBm)
    pub signal_strength_dbm: Option<i32>,
    /// Network jitter (ms)
    pub jitter_ms: f32,
    /// Network stability score (0.0-1.0)
    pub stability_score: f32,
    /// Quality assessment
    pub quality_assessment: NetworkQuality,
    /// Available data allowance (MB)
    pub available_data_mb: Option<usize>,
}

/// Network usage pattern for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPattern {
    /// Time period
    pub time_period: TimePeriod,
    /// Expected bandwidth (Mbps)
    pub expected_bandwidth_mbps: f32,
    /// Expected latency (ms)
    pub expected_latency_ms: f32,
    /// Expected stability (0.0-1.0)
    pub expected_stability: f32,
    /// Confidence level (0.0-1.0)
    pub confidence: f32,
}

/// Time periods for pattern analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimePeriod {
    Hourly,
    Daily,
    Weekly,
    Monthly,
    Seasonal,
}

// =============================================================================
// CONFIGURATION STRUCTURES
// =============================================================================

/// Main network adaptation configuration
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

/// Network-specific quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkQuantizationConfig {
    /// Quantization bits for gradients
    pub gradient_bits: u8,
    /// Quantization bits for model weights
    pub weight_bits: u8,
    /// Enable dynamic quantization
    pub dynamic_quantization: bool,
    /// Error feedback mechanism
    pub error_feedback: bool,
}

/// Retry configuration for network operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_retries: usize,
    /// Base retry interval (ms)
    pub base_interval_ms: u64,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f32,
    /// Maximum retry interval (ms)
    pub max_interval_ms: u64,
    /// Jitter factor (0.0-1.0)
    pub jitter_factor: f32,
}

/// Data usage limits for different network types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataUsageLimits {
    /// Daily limit on WiFi (MB)
    pub wifi_daily_limit_mb: Option<usize>,
    /// Daily limit on cellular (MB)
    pub cellular_daily_limit_mb: Option<usize>,
    /// Monthly limit on cellular (MB)
    pub cellular_monthly_limit_mb: Option<usize>,
    /// Warning thresholds
    pub warning_thresholds: HashMap<NetworkConnectionType, u8>,
    /// Emergency stop thresholds
    pub emergency_thresholds: HashMap<NetworkConnectionType, u8>,
}

/// Synchronization frequency configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncFrequencyConfig {
    /// Base sync frequency (minutes)
    pub base_frequency_minutes: u32,
    /// Network-specific multipliers
    pub network_multipliers: HashMap<NetworkConnectionType, f32>,
    /// Battery-aware adjustments
    pub battery_adjustments: HashMap<BatteryLevel, f32>,
    /// Adaptive frequency enabled
    pub adaptive_frequency: bool,
    /// Min/max frequency bounds (minutes)
    pub min_frequency_minutes: u32,
    pub max_frequency_minutes: u32,
}

/// Failure recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureRecoveryConfig {
    /// Enable automatic recovery
    pub enable_auto_recovery: bool,
    /// Recovery timeout (minutes)
    pub recovery_timeout_minutes: u32,
    /// Enable checkpointing
    pub enable_checkpointing: bool,
    /// Checkpoint frequency (updates)
    pub checkpoint_frequency: u32,
    /// Enable graceful degradation
    pub enable_graceful_degradation: bool,
    /// Failure detection sensitivity
    pub failure_detection_sensitivity: FailureDetectionSensitivity,
}

/// Network prediction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPredictionConfig {
    /// Enable network prediction
    pub enable_prediction: bool,
    /// Prediction window (minutes)
    pub prediction_window_minutes: u32,
    /// Historical data window (hours)
    pub historical_window_hours: u32,
    /// Prediction accuracy threshold
    pub accuracy_threshold: f32,
    /// Enable machine learning predictions
    pub enable_ml_predictions: bool,
}

/// Pruning configuration for network transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningConfig {
    /// Enable pruning for transfer
    pub enable_pruning: bool,
    /// Pruning ratio
    pub pruning_ratio: f32,
    /// Structured pruning
    pub structured_pruning: bool,
    /// Recovery mechanism
    pub recovery_mechanism: PruningRecoveryMechanism,
}

/// Delta compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaCompressionConfig {
    /// Enable delta compression
    pub enable_delta: bool,
    /// Delta computation algorithm
    pub delta_algorithm: DeltaAlgorithm,
    /// Maximum delta size (MB)
    pub max_delta_size_mb: usize,
    /// Fallback to full sync threshold
    pub fallback_threshold: f32,
}

/// Scheduling constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingConstraints {
    /// Maximum concurrent tasks
    pub max_concurrent_tasks: usize,
    /// Time windows for different task types
    pub time_windows: HashMap<FederatedTaskType, TimeWindow>,
    /// Network-specific constraints
    pub network_constraints: HashMap<NetworkConnectionType, NetworkConstraints>,
    /// Battery level constraints
    pub battery_constraints: HashMap<BatteryLevel, BatteryConstraints>,
}

/// Time window for task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindow {
    /// Start hour (0-23)
    pub start_hour: u8,
    /// End hour (0-23)
    pub end_hour: u8,
    /// Days of week (0-6, 0=Sunday)
    pub days_of_week: Vec<u8>,
    /// Timezone offset (hours)
    pub timezone_offset: i8,
}

/// Network-specific constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConstraints {
    /// Maximum bandwidth usage (%)
    pub max_bandwidth_usage_percent: f32,
    /// Rate limiting (MB/s)
    pub rate_limit_mbps: Option<f32>,
    /// Connection limits
    pub max_connections: usize,
    /// Quality requirements
    pub min_quality: NetworkQuality,
}

/// Battery-level specific constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatteryConstraints {
    /// Allow federated learning
    pub allow_federated_learning: bool,
    /// Maximum task duration (minutes)
    pub max_task_duration_minutes: u32,
    /// Reduced frequency multiplier
    pub frequency_multiplier: f32,
    /// Emergency only mode
    pub emergency_only: bool,
}

// =============================================================================
// ENUMS FOR ALGORITHMS AND STRATEGIES
// =============================================================================

/// Gradient compression algorithms
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GradientCompressionAlgorithm {
    /// No compression
    None,
    /// Top-K compression
    TopK { k: usize },
    /// Random sparsification
    RandomSparsification { ratio: f32 },
    /// Threshold-based compression
    ThresholdBased { threshold: f32 },
    /// Quantized compression
    Quantized { bits: u8 },
    /// Adaptive compression
    Adaptive,
}

/// Failure detection sensitivity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureDetectionSensitivity {
    Low,
    Medium,
    High,
    Adaptive,
}

/// Optimization strategies for scheduling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    MinimizeLatency,
    MinimizeDataUsage,
    MaximizeReliability,
    BalancedOptimization,
    BatteryAware,
    NetworkAware,
}

/// Synchronization strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncStrategy {
    Immediate,
    Batched,
    Scheduled,
    Opportunistic,
    Adaptive,
}

/// Types of synchronization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncType {
    FullModel,
    Gradients,
    Incremental,
    Checkpoint,
    Metadata,
}

/// Conflict resolution strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictResolutionStrategy {
    LastWriterWins,
    MergeConflicts,
    UserDecision,
    ServerDecision,
    VersionVector,
}

/// Merge algorithms for conflict resolution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeAlgorithm {
    AverageMerge,
    WeightedMerge,
    SelectiveMerge,
    AdaptiveMerge,
}

/// Checksum algorithms for integrity checking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChecksumAlgorithm {
    MD5,
    SHA256,
    CRC32,
    Custom,
}

/// Recovery actions for integrity failures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryAction {
    Retry,
    RedownloadFull,
    RedownloadPartial,
    UseBackup,
    SkipUpdate,
}

/// Bandwidth allocation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationStrategy {
    EqualShare,
    PriorityBased,
    WeightedFairQueuing,
    AdaptiveAllocation,
}

/// Pruning recovery mechanisms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PruningRecoveryMechanism {
    None,
    LocalRecovery,
    ServerRecovery,
    HybridRecovery,
}

/// Delta compression algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeltaAlgorithm {
    SimpleDiff,
    OptimizedDiff,
    SemanticDiff,
    CompressedDiff,
}

// =============================================================================
// FEDERATED LEARNING TASK TYPES
// =============================================================================

/// Types of federated learning tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FederatedTaskType {
    ModelDownload,
    GradientUpload,
    FullModelSync,
    IncrementalSync,
    Heartbeat,
    Checkpoint,
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
    Critical,
    Emergency,
}

/// Task status tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    Scheduled,
    InProgress,
    Completed,
    Failed,
    Cancelled,
    Deferred,
}

/// Federated learning task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedTask {
    /// Task identifier
    pub task_id: String,
    /// Task type
    pub task_type: FederatedTaskType,
    /// Priority level
    pub priority: TaskPriority,
    /// Estimated data size (MB)
    pub estimated_size_mb: usize,
    /// Network requirements
    pub network_requirements: NetworkRequirements,
    /// Scheduled time
    #[serde(skip, default = "Instant::now")]
    pub scheduled_time: Instant,
    /// Deadline
    #[serde(skip, default = "Instant::now")]
    pub deadline: Instant,
    /// Retry count
    pub retry_count: usize,
    /// Task status
    pub status: TaskStatus,
}

/// Network requirements for tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkRequirements {
    /// Minimum bandwidth (Mbps)
    pub min_bandwidth_mbps: f32,
    /// Maximum acceptable latency (ms)
    pub max_latency_ms: f32,
    /// Required reliability (%)
    pub required_reliability_percent: f32,
    /// Maximum data usage (MB)
    pub max_data_usage_mb: usize,
    /// Preferred connection types
    pub preferred_connections: Vec<NetworkConnectionType>,
}

/// Synchronization task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncTask {
    /// Task identifier
    pub task_id: String,
    /// Sync type
    pub sync_type: SyncType,
    /// Model version
    pub model_version: String,
    /// Estimated size (MB)
    pub estimated_size_mb: usize,
    /// Network requirements
    pub network_requirements: NetworkRequirements,
    /// Retry count
    pub retry_count: usize,
    /// Status
    pub status: TaskStatus,
}

// =============================================================================
// PERFORMANCE AND STATISTICS
// =============================================================================

/// Task performance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskPerformance {
    /// Task type
    pub task_type: FederatedTaskType,
    /// Network conditions during execution
    pub network_conditions: NetworkConditions,
    /// Execution time (ms)
    pub execution_time_ms: u64,
    /// Data transferred (MB)
    pub data_transferred_mb: f32,
    /// Success rate
    pub success_rate: f32,
    /// Battery consumption (mWh)
    pub battery_consumption_mwh: f32,
}

/// Compression statistics tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    /// Original size (bytes)
    pub original_size_bytes: usize,
    /// Compressed size (bytes)
    pub compressed_size_bytes: usize,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Compression time (ms)
    pub compression_time_ms: u64,
    /// Decompression time (ms)
    pub decompression_time_ms: u64,
    /// Quality loss (%)
    pub quality_loss_percent: f32,
}

/// Data usage record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataUsageRecord {
    /// Timestamp
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    /// Connection type
    pub connection_type: NetworkConnectionType,
    /// Data used (bytes)
    pub data_used_bytes: usize,
    /// Task type that used the data
    pub task_type: FederatedTaskType,
    /// Duration (ms)
    pub duration_ms: u64,
}

/// Network adaptation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkAdaptationStats {
    /// Total tasks scheduled
    pub total_tasks_scheduled: usize,
    /// Tasks completed successfully
    pub tasks_completed: usize,
    /// Tasks failed
    pub tasks_failed: usize,
    /// Average completion time (ms)
    pub avg_completion_time_ms: f64,
    /// Data usage by network type
    pub data_usage_by_network: HashMap<NetworkConnectionType, usize>,
    /// Compression effectiveness
    pub compression_stats: CompressionStats,
    /// Network quality distribution
    pub quality_distribution: HashMap<NetworkQuality, u32>,
    /// Adaptation accuracy
    pub adaptation_accuracy: f32,
    /// Battery impact (mWh)
    pub battery_impact_mwh: f32,
}

// =============================================================================
// VERSION MANAGEMENT AND INTEGRITY
// =============================================================================

/// Version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionInfo {
    /// Version identifier
    pub version_id: String,
    /// Timestamp
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    /// Model hash
    pub model_hash: String,
    /// Size (bytes)
    pub size_bytes: usize,
    /// Parent version
    pub parent_version: Option<String>,
}

/// Conflict record for tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictRecord {
    /// Conflict timestamp
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    /// Conflicting versions
    pub conflicting_versions: Vec<String>,
    /// Resolution strategy used
    pub resolution_strategy: ConflictResolutionStrategy,
    /// Resolution time (ms)
    pub resolution_time_ms: u64,
    /// Success flag
    pub success: bool,
}

/// Integrity failure record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityFailure {
    /// Failure timestamp
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    /// Model version
    pub model_version: String,
    /// Expected checksum
    pub expected_checksum: String,
    /// Actual checksum
    pub actual_checksum: String,
    /// Recovery action taken
    pub recovery_action: RecoveryAction,
}

// =============================================================================
// ADDITIONAL TYPES FROM EXISTING FILE
// =============================================================================

/// Battery impact assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatteryImpact {
    Minimal,
    Low,
    Moderate,
    High,
    Severe,
}

/// Scheduling decision for federated tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingDecision {
    /// Task to be executed
    pub task_type: FederatedTaskType,
    /// Scheduled execution time
    pub scheduled_time: SystemTime,
    /// Priority level (higher = more urgent)
    pub priority: TaskPriority,
    /// Network conditions required
    pub required_conditions: NetworkQuality,
    /// Estimated duration (ms)
    pub estimated_duration_ms: u64,
    /// Estimated data transfer (MB)
    pub estimated_data_mb: f32,
    /// Battery impact assessment
    pub battery_impact: BatteryImpact,
}

/// Result of model synchronization attempt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncResult {
    /// Whether sync was successful
    pub success: bool,
    /// Amount of data transferred (MB)
    pub data_transferred_mb: f32,
    /// Time taken for sync (ms)
    pub sync_time_ms: u64,
    /// Strategy used
    pub strategy_used: SyncStrategy,
    /// Compression ratio achieved (if applicable)
    pub compression_ratio: Option<f32>,
    /// Error message (if failed)
    pub error_message: Option<String>,
    /// Network conditions during sync
    pub network_conditions: NetworkConditions,
}

/// Data usage tracking information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataUsageInfo {
    /// Total data used today (MB)
    pub daily_usage_mb: usize,
    /// Total data used this month (MB)
    pub monthly_usage_mb: usize,
    /// Data used in current session (MB)
    pub session_usage_mb: usize,
    /// Breakdown by network type
    pub usage_by_network: HashMap<NetworkConnectionType, usize>,
    /// Remaining daily allowance (MB)
    pub daily_remaining_mb: usize,
    /// Remaining monthly allowance (MB)
    pub monthly_remaining_mb: usize,
}

/// Network metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetricsSnapshot {
    /// Timestamp of measurement
    pub timestamp: SystemTime,
    /// Network conditions
    pub conditions: NetworkConditions,
    /// Throughput measurement (MB/s)
    pub throughput_mbps: f32,
    /// Round-trip time (ms)
    pub rtt_ms: f32,
    /// Connection stability
    pub stability: f32,
    /// Quality of service metrics
    pub qos_metrics: QosMetrics,
}

/// Quality of service metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QosMetrics {
    /// Jitter variation (ms)
    pub jitter_variation_ms: f32,
    /// Packet reordering rate
    pub packet_reordering_rate: f32,
    /// Duplicate packet rate
    pub duplicate_packet_rate: f32,
    /// Out-of-order delivery rate
    pub out_of_order_rate: f32,
}

/// Traffic shaping parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficShapingParams {
    /// Maximum bandwidth allocation (Mbps)
    pub max_bandwidth_mbps: f32,
    /// Burst size allowance (MB)
    pub burst_size_mb: f32,
    /// Priority queue settings
    pub priority_queues: HashMap<TaskPriority, QueueConfig>,
    /// Rate limiting enabled
    pub rate_limiting_enabled: bool,
}

/// Queue configuration for traffic shaping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueConfig {
    /// Maximum queue size (packets)
    pub max_queue_size: usize,
    /// Drop policy
    pub drop_policy: DropPolicy,
    /// Bandwidth allocation (%)
    pub bandwidth_allocation_percent: f32,
}

/// Packet drop policies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DropPolicy {
    DropTail,
    DropHead,
    Random,
    Priority,
}

/// Network adaptation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationEvent {
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Event type
    pub event_type: AdaptationEventType,
    /// Network conditions when event occurred
    pub network_conditions: NetworkConditions,
    /// Action taken
    pub action: AdaptationAction,
    /// Event details
    pub details: String,
}

/// Types of adaptation events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptationEventType {
    QualityChange,
    StrategySwitch,
    CompressionAdjust,
    ScheduleChange,
    FailureRecovery,
    BatteryConstraint,
}

/// Actions taken during adaptation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptationAction {
    IncreaseCompression,
    DecreaseCompression,
    SwitchToWiFi,
    SwitchToCellular,
    EnableDegradedMode,
    DisableDegradedMode,
    ScheduleDelay,
    ScheduleAdvance,
    PauseOperations,
    ResumeOperations,
}

/// Prediction model types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Linear,
    Polynomial,
    ExponentialSmoothing,
    NeuralNetwork,
}

/// Network prediction model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionModelParams {
    /// Model weights
    pub weights: Vec<f32>,
    /// Bias terms
    pub biases: Vec<f32>,
    /// Learning rate
    pub learning_rate: f32,
    /// Regularization parameter
    pub regularization: f32,
    /// Model accuracy
    pub accuracy: f32,
    /// Last update timestamp
    pub last_updated: SystemTime,
}

// =============================================================================
// DEFAULT IMPLEMENTATIONS
// =============================================================================

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
            preferred_hours: vec![2, 3, 4, 5], // 2-5 AM
            avoid_peak_hours: true,
            peak_hours: vec![8, 9, 17, 18, 19, 20], // 8-9 AM, 5-8 PM
            off_peak_multiplier: 1.5,
        }
    }
}

impl Default for PoorNetworkStrategy {
    fn default() -> Self {
        Self {
            enable_degraded_mode: true,
            min_update_size_kb: 100,
            extended_retry_intervals: vec![30000, 60000, 120000, 300000], // 30s, 1m, 2m, 5m
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
            model_compression_ratio: 0.7,
            enable_differential_compression: true,
            network_quantization: NetworkQuantizationConfig::default(),
        }
    }
}

impl Default for NetworkQuantizationConfig {
    fn default() -> Self {
        Self {
            gradient_bits: 8,
            weight_bits: 16,
            dynamic_quantization: true,
            error_feedback: true,
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_interval_ms: 1000,
            backoff_multiplier: 2.0,
            max_interval_ms: 30000,
            jitter_factor: 0.1,
        }
    }
}

impl Default for DataUsageLimits {
    fn default() -> Self {
        let mut warning_thresholds = HashMap::new();
        warning_thresholds.insert(NetworkConnectionType::WiFi, 90);
        warning_thresholds.insert(NetworkConnectionType::Cellular4G, 80);
        warning_thresholds.insert(NetworkConnectionType::Cellular5G, 85);

        let mut emergency_thresholds = HashMap::new();
        emergency_thresholds.insert(NetworkConnectionType::WiFi, 95);
        emergency_thresholds.insert(NetworkConnectionType::Cellular4G, 90);
        emergency_thresholds.insert(NetworkConnectionType::Cellular5G, 95);

        Self {
            wifi_daily_limit_mb: Some(1000),
            cellular_daily_limit_mb: Some(100),
            cellular_monthly_limit_mb: Some(2000),
            warning_thresholds,
            emergency_thresholds,
        }
    }
}

impl Default for SyncFrequencyConfig {
    fn default() -> Self {
        let mut network_multipliers = HashMap::new();
        network_multipliers.insert(NetworkConnectionType::WiFi, 1.0);
        network_multipliers.insert(NetworkConnectionType::Cellular5G, 0.8);
        network_multipliers.insert(NetworkConnectionType::Cellular4G, 0.5);
        network_multipliers.insert(NetworkConnectionType::Ethernet, 1.2);

        let mut battery_adjustments = HashMap::new();
        battery_adjustments.insert(BatteryLevel::Critical, 0.1);
        battery_adjustments.insert(BatteryLevel::Low, 0.3);
        battery_adjustments.insert(BatteryLevel::Medium, 0.7);
        battery_adjustments.insert(BatteryLevel::High, 1.0);
        battery_adjustments.insert(BatteryLevel::Charging, 1.5);

        Self {
            base_frequency_minutes: 60,
            network_multipliers,
            battery_adjustments,
            adaptive_frequency: true,
            min_frequency_minutes: 15,
            max_frequency_minutes: 360, // 6 hours
        }
    }
}

impl Default for FailureRecoveryConfig {
    fn default() -> Self {
        Self {
            enable_auto_recovery: true,
            recovery_timeout_minutes: 30,
            enable_checkpointing: true,
            checkpoint_frequency: 10,
            enable_graceful_degradation: true,
            failure_detection_sensitivity: FailureDetectionSensitivity::Medium,
        }
    }
}

impl Default for NetworkPredictionConfig {
    fn default() -> Self {
        Self {
            enable_prediction: true,
            prediction_window_minutes: 30,
            historical_window_hours: 24,
            accuracy_threshold: 0.8,
            enable_ml_predictions: true,
        }
    }
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            enable_pruning: true,
            pruning_ratio: 0.1,
            structured_pruning: false,
            recovery_mechanism: PruningRecoveryMechanism::HybridRecovery,
        }
    }
}

impl Default for DeltaCompressionConfig {
    fn default() -> Self {
        Self {
            enable_delta: true,
            delta_algorithm: DeltaAlgorithm::OptimizedDiff,
            max_delta_size_mb: 10,
            fallback_threshold: 0.8,
        }
    }
}

impl Default for SchedulingConstraints {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: 3,
            time_windows: HashMap::new(),
            network_constraints: HashMap::new(),
            battery_constraints: HashMap::new(),
        }
    }
}

impl Default for NetworkConditions {
    fn default() -> Self {
        Self {
            timestamp: Instant::now(),
            connection_type: NetworkConnectionType::WiFi,
            bandwidth_mbps: 10.0,
            latency_ms: 50.0,
            packet_loss_percent: 0.1,
            signal_strength_dbm: Some(-50),
            jitter_ms: 5.0,
            stability_score: 0.8,
            quality_assessment: NetworkQuality::Good,
            available_data_mb: Some(1000),
        }
    }
}

impl Default for CompressionStats {
    fn default() -> Self {
        Self {
            original_size_bytes: 0,
            compressed_size_bytes: 0,
            compression_ratio: 1.0,
            compression_time_ms: 0,
            decompression_time_ms: 0,
            quality_loss_percent: 0.0,
        }
    }
}

impl Default for QosMetrics {
    fn default() -> Self {
        Self {
            jitter_variation_ms: 0.0,
            packet_reordering_rate: 0.0,
            duplicate_packet_rate: 0.0,
            out_of_order_rate: 0.0,
        }
    }
}

impl Default for NetworkRequirements {
    fn default() -> Self {
        Self {
            min_bandwidth_mbps: 1.0,
            max_latency_ms: 1000.0,
            required_reliability_percent: 95.0,
            max_data_usage_mb: 100,
            preferred_connections: vec![NetworkConnectionType::WiFi],
        }
    }
}
