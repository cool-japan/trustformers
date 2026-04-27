//! Lifecycle Configuration Types
//!
//! This module contains all configuration types for app lifecycle management,
//! including resource management, background tasks, notifications, and system responses.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Lifecycle management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleConfig {
    /// Enable background execution
    pub enable_background_execution: bool,
    /// Background execution time limit (seconds)
    pub background_execution_limit_seconds: u64,
    /// Enable state persistence
    pub enable_state_persistence: bool,
    /// State save interval (seconds)
    pub state_save_interval_seconds: u64,
    /// Resource management settings
    pub resource_management: ResourceManagementConfig,
    /// Background task settings
    pub background_tasks: BackgroundTaskConfig,
    /// Notification settings
    pub notifications: NotificationConfig,
    /// Memory warning handling
    pub memory_warning_handling: MemoryWarningConfig,
    /// Thermal warning handling
    pub thermal_warning_handling: ThermalWarningConfig,
    /// Network interruption handling
    pub network_interruption_handling: NetworkInterruptionConfig,
}

/// Resource management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceManagementConfig {
    /// Enable automatic resource scaling
    pub enable_auto_scaling: bool,
    /// Memory pressure response
    pub memory_pressure_response: MemoryPressureResponse,
    /// Thermal pressure response
    pub thermal_pressure_response: ThermalPressureResponse,
    /// Battery pressure response
    pub battery_pressure_response: BatteryPressureResponse,
    /// Background resource limits
    pub background_limits: BackgroundResourceLimits,
    /// Foreground resource allocation
    pub foreground_allocation: ForegroundResourceAllocation,
}

/// Memory pressure response strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPressureResponse {
    /// Enable memory cleanup on pressure
    pub enable_cleanup: bool,
    /// Memory pressure thresholds (%)
    pub pressure_thresholds: MemoryPressureThresholds,
    /// Cleanup strategies by pressure level
    pub cleanup_strategies: HashMap<MemoryPressureLevel, CleanupStrategy>,
    /// Model eviction policy
    pub model_eviction_policy: ModelEvictionPolicy,
}

/// Memory pressure levels and thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPressureThresholds {
    /// Warning level threshold (%)
    pub warning_percent: u8,
    /// Critical level threshold (%)
    pub critical_percent: u8,
    /// Emergency level threshold (%)
    pub emergency_percent: u8,
}

/// Memory pressure levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MemoryPressureLevel {
    Normal,
    Warning,
    Critical,
    Emergency,
}

/// Cleanup strategies for memory pressure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupStrategy {
    /// Clear model caches
    pub clear_model_cache: bool,
    /// Clear intermediate tensors
    pub clear_intermediate_tensors: bool,
    /// Reduce batch sizes
    pub reduce_batch_sizes: bool,
    /// Compress models in memory
    pub compress_models: bool,
    /// Offload models to disk
    pub offload_to_disk: bool,
    /// Priority for cleanup operations
    pub cleanup_priority: CleanupPriority,
}

/// Cleanup operation priorities
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CleanupPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Model eviction policies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelEvictionPolicy {
    LeastRecentlyUsed,
    LeastFrequentlyUsed,
    SizeBasedEviction,
    PriorityBasedEviction,
    AdaptiveEviction,
}

/// Thermal pressure response strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalPressureResponse {
    /// Enable thermal response
    pub enable_response: bool,
    /// Thermal throttling levels
    pub throttling_levels: ThermalThrottlingLevels,
    /// Performance scaling strategy
    pub performance_scaling: PerformanceScalingStrategy,
    /// Inference frequency reduction
    pub frequency_reduction: FrequencyReductionConfig,
}

/// Thermal throttling levels and responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalThrottlingLevels {
    /// Light throttling threshold (°C)
    pub light_throttle_celsius: f32,
    /// Moderate throttling threshold (°C)
    pub moderate_throttle_celsius: f32,
    /// Heavy throttling threshold (°C)
    pub heavy_throttle_celsius: f32,
    /// Emergency throttling threshold (°C)
    pub emergency_throttle_celsius: f32,
}

/// Performance scaling strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceScalingStrategy {
    Linear,
    Exponential,
    Adaptive,
    UserDefined,
}

/// Inference frequency reduction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyReductionConfig {
    /// Enable frequency reduction
    pub enable_reduction: bool,
    /// Reduction factors by thermal level
    pub reduction_factors: HashMap<ThermalLevel, f32>,
    /// Minimum inference frequency (Hz)
    pub min_frequency_hz: f32,
}

/// Thermal levels for throttling
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ThermalLevel {
    Normal,
    Light,
    Moderate,
    Heavy,
    Emergency,
}

/// Battery pressure response strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatteryPressureResponse {
    /// Enable battery response
    pub enable_response: bool,
    /// Battery level thresholds
    pub battery_thresholds: BatteryThresholds,
    /// Power saving strategies
    pub power_saving_strategies: HashMap<crate::battery::BatteryLevel, PowerSavingStrategy>,
    /// Background processing limits
    pub background_processing_limits: BackgroundProcessingLimits,
}

/// Battery level thresholds for responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatteryThresholds {
    /// Critical battery level (%)
    pub critical_percent: u8,
    /// Low battery level (%)
    pub low_percent: u8,
    /// Medium battery level (%)
    pub medium_percent: u8,
}

/// Power saving strategies by battery level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerSavingStrategy {
    /// Reduce inference frequency
    pub reduce_inference_frequency: bool,
    /// Lower model precision
    pub lower_model_precision: bool,
    /// Disable background updates
    pub disable_background_updates: bool,
    /// Use CPU-only inference
    pub cpu_only_inference: bool,
    /// Defer non-critical tasks
    pub defer_non_critical_tasks: bool,
}

/// Background processing limits based on battery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundProcessingLimits {
    /// Maximum background time (seconds)
    pub max_background_time_seconds: u64,
    /// Maximum CPU usage (%)
    pub max_cpu_usage_percent: u8,
    /// Maximum memory usage (MB)
    pub max_memory_usage_mb: usize,
    /// Task priority adjustments
    pub priority_adjustments: HashMap<TaskType, i8>,
}

/// Background resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundResourceLimits {
    /// Maximum CPU usage in background (%)
    pub max_cpu_percent: u8,
    /// Maximum memory usage in background (MB)
    pub max_memory_mb: usize,
    /// Maximum network usage in background (MB/min)
    pub max_network_mbps: f32,
    /// Maximum background execution time (seconds)
    pub max_execution_time_seconds: u64,
}

/// Foreground resource allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForegroundResourceAllocation {
    /// CPU allocation (%)
    pub cpu_allocation_percent: u8,
    /// Memory allocation (MB)
    pub memory_allocation_mb: usize,
    /// Network allocation (MB/min)
    pub network_allocation_mbps: f32,
    /// GPU allocation (%)
    pub gpu_allocation_percent: Option<u8>,
}

/// Background task configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundTaskConfig {
    /// Enable background model updates
    pub enable_model_updates: bool,
    /// Enable background federated learning
    pub enable_federated_learning: bool,
    /// Enable background data processing
    pub enable_data_processing: bool,
    /// Background task priorities
    pub task_priorities: HashMap<TaskType, TaskPriority>,
    /// Task scheduling strategies
    pub scheduling_strategies: HashMap<TaskType, SchedulingStrategy>,
    /// Maximum concurrent background tasks
    pub max_concurrent_tasks: usize,
}

/// Types of background tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskType {
    ModelUpdate,
    FederatedLearning,
    DataSync,
    CacheCleanup,
    Analytics,
    Backup,
    Precomputation,
    HealthCheck,
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

/// Task scheduling strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchedulingStrategy {
    Immediate,
    Deferred,
    OpportunisticAgg,
    UserIdle,
    NetworkOptimal,
    BatteryOptimal,
    ThermalOptimal,
}

/// Notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    /// Enable push notifications
    pub enable_push_notifications: bool,
    /// Enable local notifications
    pub enable_local_notifications: bool,
    /// Notification types to enable
    pub enabled_types: Vec<NotificationType>,
    /// Background notification handling
    pub background_handling: BackgroundNotificationHandling,
    /// Notification throttling
    pub throttling: NotificationThrottling,
}

/// Types of notifications
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NotificationType {
    ModelUpdateAvailable,
    FederatedLearningTask,
    SystemAlert,
    PerformanceWarning,
    MaintenanceRequired,
    UserAction,
}

/// Background notification handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundNotificationHandling {
    /// Queue notifications when in background
    pub queue_when_background: bool,
    /// Maximum queued notifications
    pub max_queued_notifications: usize,
    /// Batch notification delivery
    pub batch_delivery: bool,
    /// Delivery strategies by type
    pub delivery_strategies: HashMap<NotificationType, DeliveryStrategy>,
}

/// Notification delivery strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeliveryStrategy {
    Immediate,
    Batched,
    Deferred,
    OnForeground,
    UserTriggered,
}

/// Notification throttling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationThrottling {
    /// Enable throttling
    pub enable_throttling: bool,
    /// Maximum notifications per hour
    pub max_per_hour: u32,
    /// Minimum interval between notifications (seconds)
    pub min_interval_seconds: u64,
    /// Notification type-based rate limits
    pub notification_rate_limits: HashMap<NotificationType, u32>,
}

/// Memory warning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryWarningConfig {
    /// Enable memory warning handling
    pub enable_handling: bool,
    /// Automatic cleanup on warning
    pub automatic_cleanup: bool,
    /// Warning response strategies
    pub response_strategies: Vec<MemoryWarningResponse>,
    /// Grace period before aggressive cleanup (seconds)
    pub grace_period_seconds: u64,
}

/// Memory warning response actions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryWarningResponse {
    ClearCaches,
    ReduceBatchSizes,
    OffloadModels,
    PauseBackgroundTasks,
    NotifyUser,
    ForceGarbageCollection,
}

/// Thermal warning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalWarningConfig {
    /// Enable thermal warning handling
    pub enable_handling: bool,
    /// Automatic throttling on warning
    pub automatic_throttling: bool,
    /// Warning response strategies
    pub response_strategies: Vec<ThermalWarningResponse>,
    /// Cooldown period (seconds)
    pub cooldown_period_seconds: u64,
}

/// Thermal warning response actions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThermalWarningResponse {
    ReduceInferenceFrequency,
    LowerModelPrecision,
    PauseComputeIntensiveTasks,
    SwitchToCpuOnly,
    NotifyUser,
    EnterCooldownMode,
}

/// Network interruption handling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInterruptionConfig {
    /// Enable interruption handling
    pub enable_handling: bool,
    /// Automatic reconnection
    pub automatic_reconnection: bool,
    /// Maximum reconnection attempts
    pub max_reconnection_attempts: u32,
    /// Reconnection backoff strategy
    pub backoff_strategy: ReconnectionBackoffStrategy,
    /// Offline mode configuration
    pub offline_mode: OfflineModeConfig,
}

/// Reconnection backoff strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReconnectionBackoffStrategy {
    Linear,
    Exponential,
    Fibonacci,
    Custom,
}

/// Offline mode configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfflineModeConfig {
    /// Enable offline mode
    pub enable_offline_mode: bool,
    /// Local model fallback
    pub local_model_fallback: bool,
    /// Cache inference results
    pub cache_inference_results: bool,
    /// Queue sync operations
    pub queue_sync_operations: bool,
    /// Maximum offline cache size (MB)
    pub max_offline_cache_mb: usize,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            enable_background_execution: true,
            background_execution_limit_seconds: 30,
            enable_state_persistence: true,
            state_save_interval_seconds: 60,
            resource_management: ResourceManagementConfig::default(),
            background_tasks: BackgroundTaskConfig::default(),
            notifications: NotificationConfig::default(),
            memory_warning_handling: MemoryWarningConfig::default(),
            thermal_warning_handling: ThermalWarningConfig::default(),
            network_interruption_handling: NetworkInterruptionConfig::default(),
        }
    }
}

impl Default for ResourceManagementConfig {
    fn default() -> Self {
        Self {
            enable_auto_scaling: true,
            memory_pressure_response: MemoryPressureResponse::default(),
            thermal_pressure_response: ThermalPressureResponse::default(),
            battery_pressure_response: BatteryPressureResponse::default(),
            background_limits: BackgroundResourceLimits::default(),
            foreground_allocation: ForegroundResourceAllocation::default(),
        }
    }
}

impl Default for MemoryPressureResponse {
    fn default() -> Self {
        let mut cleanup_strategies = HashMap::new();
        cleanup_strategies.insert(
            MemoryPressureLevel::Warning,
            CleanupStrategy {
                clear_model_cache: true,
                clear_intermediate_tensors: false,
                reduce_batch_sizes: false,
                compress_models: false,
                offload_to_disk: false,
                cleanup_priority: CleanupPriority::Low,
            },
        );
        cleanup_strategies.insert(
            MemoryPressureLevel::Critical,
            CleanupStrategy {
                clear_model_cache: true,
                clear_intermediate_tensors: true,
                reduce_batch_sizes: true,
                compress_models: true,
                offload_to_disk: false,
                cleanup_priority: CleanupPriority::Medium,
            },
        );
        cleanup_strategies.insert(
            MemoryPressureLevel::Emergency,
            CleanupStrategy {
                clear_model_cache: true,
                clear_intermediate_tensors: true,
                reduce_batch_sizes: true,
                compress_models: true,
                offload_to_disk: true,
                cleanup_priority: CleanupPriority::Critical,
            },
        );

        Self {
            enable_cleanup: true,
            pressure_thresholds: MemoryPressureThresholds {
                warning_percent: 70,
                critical_percent: 85,
                emergency_percent: 95,
            },
            cleanup_strategies,
            model_eviction_policy: ModelEvictionPolicy::LeastRecentlyUsed,
        }
    }
}

impl Default for ThermalPressureResponse {
    fn default() -> Self {
        let mut reduction_factors = HashMap::new();
        reduction_factors.insert(ThermalLevel::Light, 0.9);
        reduction_factors.insert(ThermalLevel::Moderate, 0.7);
        reduction_factors.insert(ThermalLevel::Heavy, 0.5);
        reduction_factors.insert(ThermalLevel::Emergency, 0.3);

        Self {
            enable_response: true,
            throttling_levels: ThermalThrottlingLevels {
                light_throttle_celsius: 40.0,
                moderate_throttle_celsius: 45.0,
                heavy_throttle_celsius: 50.0,
                emergency_throttle_celsius: 55.0,
            },
            performance_scaling: PerformanceScalingStrategy::Adaptive,
            frequency_reduction: FrequencyReductionConfig {
                enable_reduction: true,
                reduction_factors,
                min_frequency_hz: 1.0,
            },
        }
    }
}

impl Default for BatteryPressureResponse {
    fn default() -> Self {
        use crate::battery::BatteryLevel;
        let mut power_saving_strategies = HashMap::new();

        power_saving_strategies.insert(
            BatteryLevel::Critical,
            PowerSavingStrategy {
                reduce_inference_frequency: true,
                lower_model_precision: true,
                disable_background_updates: true,
                cpu_only_inference: true,
                defer_non_critical_tasks: true,
            },
        );

        power_saving_strategies.insert(
            BatteryLevel::Low,
            PowerSavingStrategy {
                reduce_inference_frequency: true,
                lower_model_precision: false,
                disable_background_updates: true,
                cpu_only_inference: false,
                defer_non_critical_tasks: true,
            },
        );

        let mut priority_adjustments = HashMap::new();
        priority_adjustments.insert(TaskType::ModelUpdate, -1);
        priority_adjustments.insert(TaskType::FederatedLearning, -2);
        priority_adjustments.insert(TaskType::Analytics, -1);

        Self {
            enable_response: true,
            battery_thresholds: BatteryThresholds {
                critical_percent: 15,
                low_percent: 30,
                medium_percent: 50,
            },
            power_saving_strategies,
            background_processing_limits: BackgroundProcessingLimits {
                max_background_time_seconds: 10,
                max_cpu_usage_percent: 20,
                max_memory_usage_mb: 100,
                priority_adjustments,
            },
        }
    }
}

impl Default for BackgroundResourceLimits {
    fn default() -> Self {
        Self {
            max_cpu_percent: 30,
            max_memory_mb: 256,
            max_network_mbps: 1.0,
            max_execution_time_seconds: 30,
        }
    }
}

impl Default for ForegroundResourceAllocation {
    fn default() -> Self {
        Self {
            cpu_allocation_percent: 80,
            memory_allocation_mb: 1024,
            network_allocation_mbps: 10.0,
            gpu_allocation_percent: Some(70),
        }
    }
}

impl Default for BackgroundTaskConfig {
    fn default() -> Self {
        let mut task_priorities = HashMap::new();
        task_priorities.insert(TaskType::ModelUpdate, TaskPriority::Normal);
        task_priorities.insert(TaskType::FederatedLearning, TaskPriority::Low);
        task_priorities.insert(TaskType::DataSync, TaskPriority::Normal);
        task_priorities.insert(TaskType::CacheCleanup, TaskPriority::Low);
        task_priorities.insert(TaskType::Analytics, TaskPriority::Low);
        task_priorities.insert(TaskType::Backup, TaskPriority::Low);
        task_priorities.insert(TaskType::Precomputation, TaskPriority::Normal);
        task_priorities.insert(TaskType::HealthCheck, TaskPriority::High);

        let mut scheduling_strategies = HashMap::new();
        scheduling_strategies.insert(TaskType::ModelUpdate, SchedulingStrategy::NetworkOptimal);
        scheduling_strategies.insert(
            TaskType::FederatedLearning,
            SchedulingStrategy::BatteryOptimal,
        );
        scheduling_strategies.insert(TaskType::DataSync, SchedulingStrategy::NetworkOptimal);
        scheduling_strategies.insert(TaskType::CacheCleanup, SchedulingStrategy::UserIdle);
        scheduling_strategies.insert(TaskType::Analytics, SchedulingStrategy::UserIdle);
        scheduling_strategies.insert(TaskType::Backup, SchedulingStrategy::UserIdle);
        scheduling_strategies.insert(TaskType::Precomputation, SchedulingStrategy::ThermalOptimal);
        scheduling_strategies.insert(TaskType::HealthCheck, SchedulingStrategy::Immediate);

        Self {
            enable_model_updates: true,
            enable_federated_learning: false,
            enable_data_processing: true,
            task_priorities,
            scheduling_strategies,
            max_concurrent_tasks: 3,
        }
    }
}

impl Default for NotificationConfig {
    fn default() -> Self {
        let enabled_types = vec![
            NotificationType::SystemAlert,
            NotificationType::PerformanceWarning,
            NotificationType::MaintenanceRequired,
        ];

        Self {
            enable_push_notifications: true,
            enable_local_notifications: true,
            enabled_types,
            background_handling: BackgroundNotificationHandling::default(),
            throttling: NotificationThrottling::default(),
        }
    }
}

impl Default for BackgroundNotificationHandling {
    fn default() -> Self {
        let mut delivery_strategies = HashMap::new();
        delivery_strategies.insert(
            NotificationType::ModelUpdateAvailable,
            DeliveryStrategy::Batched,
        );
        delivery_strategies.insert(
            NotificationType::FederatedLearningTask,
            DeliveryStrategy::Deferred,
        );
        delivery_strategies.insert(NotificationType::SystemAlert, DeliveryStrategy::Immediate);
        delivery_strategies.insert(
            NotificationType::PerformanceWarning,
            DeliveryStrategy::OnForeground,
        );
        delivery_strategies.insert(
            NotificationType::MaintenanceRequired,
            DeliveryStrategy::Batched,
        );
        delivery_strategies.insert(
            NotificationType::UserAction,
            DeliveryStrategy::UserTriggered,
        );

        Self {
            queue_when_background: true,
            max_queued_notifications: 10,
            batch_delivery: true,
            delivery_strategies,
        }
    }
}

impl Default for NotificationThrottling {
    fn default() -> Self {
        let mut notification_rate_limits = HashMap::new();
        notification_rate_limits.insert(NotificationType::ModelUpdateAvailable, 2);
        notification_rate_limits.insert(NotificationType::FederatedLearningTask, 5);
        notification_rate_limits.insert(NotificationType::SystemAlert, 10);
        notification_rate_limits.insert(NotificationType::PerformanceWarning, 20);
        notification_rate_limits.insert(NotificationType::MaintenanceRequired, 100);

        Self {
            enable_throttling: true,
            max_per_hour: 10,
            min_interval_seconds: 300, // 5 minutes
            notification_rate_limits,
        }
    }
}

impl Default for MemoryWarningConfig {
    fn default() -> Self {
        Self {
            enable_handling: true,
            automatic_cleanup: true,
            response_strategies: vec![
                MemoryWarningResponse::ClearCaches,
                MemoryWarningResponse::ReduceBatchSizes,
                MemoryWarningResponse::PauseBackgroundTasks,
            ],
            grace_period_seconds: 10,
        }
    }
}

impl Default for ThermalWarningConfig {
    fn default() -> Self {
        Self {
            enable_handling: true,
            automatic_throttling: true,
            response_strategies: vec![
                ThermalWarningResponse::ReduceInferenceFrequency,
                ThermalWarningResponse::PauseComputeIntensiveTasks,
            ],
            cooldown_period_seconds: 30,
        }
    }
}

impl Default for NetworkInterruptionConfig {
    fn default() -> Self {
        Self {
            enable_handling: true,
            automatic_reconnection: true,
            max_reconnection_attempts: 5,
            backoff_strategy: ReconnectionBackoffStrategy::Exponential,
            offline_mode: OfflineModeConfig::default(),
        }
    }
}

impl Default for OfflineModeConfig {
    fn default() -> Self {
        Self {
            enable_offline_mode: true,
            local_model_fallback: true,
            cache_inference_results: true,
            queue_sync_operations: true,
            max_offline_cache_mb: 100,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── LifecycleConfig::default() ────────────────────────────────────────

    #[test]
    fn test_lifecycle_config_default_fields() {
        let cfg = LifecycleConfig::default();
        assert!(cfg.enable_background_execution);
        assert!(cfg.background_execution_limit_seconds > 0);
        assert!(cfg.enable_state_persistence);
        assert!(cfg.state_save_interval_seconds > 0);
    }

    // ── MemoryPressureLevel ordering ──────────────────────────────────────

    #[test]
    fn test_memory_pressure_level_ordering() {
        assert!(MemoryPressureLevel::Normal < MemoryPressureLevel::Warning);
        assert!(MemoryPressureLevel::Warning < MemoryPressureLevel::Critical);
        assert!(MemoryPressureLevel::Critical < MemoryPressureLevel::Emergency);
    }

    #[test]
    fn test_memory_pressure_level_equality() {
        assert_eq!(MemoryPressureLevel::Normal, MemoryPressureLevel::Normal);
        assert_ne!(MemoryPressureLevel::Warning, MemoryPressureLevel::Critical);
    }

    // ── CleanupPriority ordering ──────────────────────────────────────────

    #[test]
    fn test_cleanup_priority_ordering() {
        assert!(CleanupPriority::Low < CleanupPriority::Medium);
        assert!(CleanupPriority::Medium < CleanupPriority::High);
        assert!(CleanupPriority::High < CleanupPriority::Critical);
    }

    // ── ModelEvictionPolicy variants ──────────────────────────────────────

    #[test]
    fn test_model_eviction_policy_variants() {
        let policies = [
            ModelEvictionPolicy::LeastRecentlyUsed,
            ModelEvictionPolicy::LeastFrequentlyUsed,
            ModelEvictionPolicy::SizeBasedEviction,
            ModelEvictionPolicy::PriorityBasedEviction,
            ModelEvictionPolicy::AdaptiveEviction,
        ];
        for p in &policies {
            assert!(!format!("{:?}", p).is_empty());
        }
    }

    // ── ThermalLevel ordering ─────────────────────────────────────────────

    #[test]
    fn test_thermal_level_ordering() {
        assert!(ThermalLevel::Normal < ThermalLevel::Light);
        assert!(ThermalLevel::Light < ThermalLevel::Moderate);
        assert!(ThermalLevel::Moderate < ThermalLevel::Heavy);
        assert!(ThermalLevel::Heavy < ThermalLevel::Emergency);
    }

    // ── PerformanceScalingStrategy variants ──────────────────────────────

    #[test]
    fn test_performance_scaling_strategy_variants() {
        let strats = [
            PerformanceScalingStrategy::Linear,
            PerformanceScalingStrategy::Exponential,
            PerformanceScalingStrategy::Adaptive,
            PerformanceScalingStrategy::UserDefined,
        ];
        for s in &strats {
            assert!(!format!("{:?}", s).is_empty());
        }
    }

    // ── TaskType / TaskPriority ───────────────────────────────────────────

    #[test]
    fn test_task_type_variants() {
        let types = [
            TaskType::ModelUpdate,
            TaskType::FederatedLearning,
            TaskType::DataSync,
            TaskType::CacheCleanup,
            TaskType::Analytics,
            TaskType::Backup,
            TaskType::Precomputation,
            TaskType::HealthCheck,
        ];
        for t in &types {
            assert!(!format!("{:?}", t).is_empty());
        }
    }

    #[test]
    fn test_task_priority_ordering() {
        assert!(TaskPriority::Low < TaskPriority::Normal);
        assert!(TaskPriority::Normal < TaskPriority::High);
        assert!(TaskPriority::High < TaskPriority::Critical);
        assert!(TaskPriority::Critical < TaskPriority::Emergency);
    }

    // ── SchedulingStrategy variants ───────────────────────────────────────

    #[test]
    fn test_scheduling_strategy_variants() {
        let strats = [
            SchedulingStrategy::Immediate,
            SchedulingStrategy::Deferred,
            SchedulingStrategy::OpportunisticAgg,
            SchedulingStrategy::UserIdle,
            SchedulingStrategy::NetworkOptimal,
            SchedulingStrategy::BatteryOptimal,
            SchedulingStrategy::ThermalOptimal,
        ];
        for s in &strats {
            assert!(!format!("{:?}", s).is_empty());
        }
    }

    // ── NotificationType variants ─────────────────────────────────────────

    #[test]
    fn test_notification_type_variants() {
        let types = [
            NotificationType::ModelUpdateAvailable,
            NotificationType::FederatedLearningTask,
            NotificationType::SystemAlert,
            NotificationType::PerformanceWarning,
            NotificationType::MaintenanceRequired,
            NotificationType::UserAction,
        ];
        for t in &types {
            assert!(!format!("{:?}", t).is_empty());
        }
    }

    // ── DeliveryStrategy variants ─────────────────────────────────────────

    #[test]
    fn test_delivery_strategy_variants() {
        let strats = [
            DeliveryStrategy::Immediate,
            DeliveryStrategy::Batched,
            DeliveryStrategy::Deferred,
            DeliveryStrategy::OnForeground,
            DeliveryStrategy::UserTriggered,
        ];
        for s in &strats {
            assert!(!format!("{:?}", s).is_empty());
        }
    }

    // ── MemoryWarningResponse variants ────────────────────────────────────

    #[test]
    fn test_memory_warning_response_variants() {
        let responses = [
            MemoryWarningResponse::ClearCaches,
            MemoryWarningResponse::ReduceBatchSizes,
            MemoryWarningResponse::OffloadModels,
            MemoryWarningResponse::PauseBackgroundTasks,
            MemoryWarningResponse::NotifyUser,
            MemoryWarningResponse::ForceGarbageCollection,
        ];
        for r in &responses {
            assert!(!format!("{:?}", r).is_empty());
        }
    }

    // ── ThermalWarningResponse variants ──────────────────────────────────

    #[test]
    fn test_thermal_warning_response_variants() {
        let responses = [
            ThermalWarningResponse::ReduceInferenceFrequency,
            ThermalWarningResponse::LowerModelPrecision,
            ThermalWarningResponse::PauseComputeIntensiveTasks,
            ThermalWarningResponse::SwitchToCpuOnly,
            ThermalWarningResponse::NotifyUser,
            ThermalWarningResponse::EnterCooldownMode,
        ];
        for r in &responses {
            assert!(!format!("{:?}", r).is_empty());
        }
    }

    // ── ReconnectionBackoffStrategy variants ─────────────────────────────

    #[test]
    fn test_reconnection_backoff_strategy_variants() {
        let strats = [
            ReconnectionBackoffStrategy::Linear,
            ReconnectionBackoffStrategy::Exponential,
            ReconnectionBackoffStrategy::Fibonacci,
            ReconnectionBackoffStrategy::Custom,
        ];
        for s in &strats {
            assert!(!format!("{:?}", s).is_empty());
        }
    }

    // ── BackgroundResourceLimits::default() ──────────────────────────────

    #[test]
    fn test_background_resource_limits_default() {
        let limits = BackgroundResourceLimits::default();
        assert!(limits.max_cpu_percent > 0);
        assert!(limits.max_memory_mb > 0);
        assert!(limits.max_network_mbps > 0.0);
        assert!(limits.max_execution_time_seconds > 0);
    }

    // ── ForegroundResourceAllocation::default() ──────────────────────────

    #[test]
    fn test_foreground_resource_allocation_default() {
        let alloc = ForegroundResourceAllocation::default();
        assert!(alloc.cpu_allocation_percent > 0);
        assert!(alloc.memory_allocation_mb > 0);
        assert!(alloc.gpu_allocation_percent.is_some());
    }

    // ── OfflineModeConfig::default() ─────────────────────────────────────

    #[test]
    fn test_offline_mode_config_default() {
        let cfg = OfflineModeConfig::default();
        assert!(cfg.enable_offline_mode);
        assert!(cfg.local_model_fallback);
        assert!(cfg.max_offline_cache_mb > 0);
    }

    // ── NetworkInterruptionConfig::default() ─────────────────────────────

    #[test]
    fn test_network_interruption_config_default() {
        let cfg = NetworkInterruptionConfig::default();
        assert!(cfg.enable_handling);
        assert!(cfg.automatic_reconnection);
        assert!(cfg.max_reconnection_attempts > 0);
    }

    // ── MemoryWarningConfig::default() ───────────────────────────────────

    #[test]
    fn test_memory_warning_config_default() {
        let cfg = MemoryWarningConfig::default();
        assert!(cfg.enable_handling);
        assert!(cfg.automatic_cleanup);
        assert!(!cfg.response_strategies.is_empty());
    }

    // ── NotificationThrottling::default() ────────────────────────────────

    #[test]
    fn test_notification_throttling_default() {
        let throttle = NotificationThrottling::default();
        assert!(throttle.enable_throttling);
        assert!(throttle.max_per_hour > 0);
        assert!(throttle.min_interval_seconds > 0);
    }

    // ── BackgroundTaskConfig::default() ──────────────────────────────────

    #[test]
    fn test_background_task_config_default() {
        let cfg = BackgroundTaskConfig::default();
        assert!(cfg.enable_model_updates);
        assert!(cfg.max_concurrent_tasks > 0);
        assert!(cfg.task_priorities.contains_key(&TaskType::HealthCheck));
        assert_eq!(
            cfg.task_priorities.get(&TaskType::HealthCheck).copied(),
            Some(TaskPriority::High)
        );
    }

    // ── MemoryPressureThresholds ──────────────────────────────────────────

    #[test]
    fn test_memory_pressure_thresholds_ordering() {
        let cfg = LifecycleConfig::default();
        let thresholds = &cfg.resource_management.memory_pressure_response.pressure_thresholds;
        assert!(thresholds.warning_percent < thresholds.critical_percent);
        assert!(thresholds.critical_percent < thresholds.emergency_percent);
    }
}
