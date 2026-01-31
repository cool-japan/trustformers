//! Lifecycle Statistics and Monitoring
//!
//! This module contains types for tracking and analyzing lifecycle statistics,
//! performance metrics, and system monitoring.

use crate::lifecycle::config::{TaskPriority, TaskType};
use crate::lifecycle::state::AppState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Lifecycle statistics tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleStats {
    /// Statistics collection start time
    pub start_timestamp: u64,
    /// App state statistics
    pub app_state_stats: AppStateStats,
    /// Task execution statistics
    pub task_execution_stats: HashMap<TaskType, TaskExecutionStats>,
    /// Resource usage statistics
    pub resource_usage_stats: ResourceUsageStats,
    /// Performance statistics
    pub performance_stats: PerformanceStats,
    /// Error statistics
    pub error_stats: ErrorStats,
    /// User interaction statistics
    pub user_interaction_stats: UserInteractionStats,
    /// System health statistics
    pub system_health_stats: SystemHealthStats,
}

/// App state transition statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppStateStats {
    /// Time spent in each state (seconds)
    pub time_in_state: HashMap<AppState, u64>,
    /// Transition counts between states
    pub transition_counts: HashMap<String, u32>, // "FromState->ToState" format
    /// Average time between transitions (seconds)
    pub avg_transition_interval_seconds: f64,
    /// Total transitions recorded
    pub total_transitions: u64,
    /// Last state change timestamp
    pub last_state_change_timestamp: u64,
    /// State change frequency (transitions/hour)
    pub state_change_frequency_per_hour: f32,
}

/// Task execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskExecutionStats {
    /// Total tasks executed
    pub total_executed: u64,
    /// Successful executions
    pub successful_executions: u64,
    /// Failed executions
    pub failed_executions: u64,
    /// Cancelled executions
    pub cancelled_executions: u64,
    /// Average execution time (seconds)
    pub avg_execution_time_seconds: f64,
    /// Minimum execution time (seconds)
    pub min_execution_time_seconds: f64,
    /// Maximum execution time (seconds)
    pub max_execution_time_seconds: f64,
    /// Success rate percentage
    pub success_rate_percent: f32,
    /// Average resource consumption
    pub avg_resource_consumption: AvgResourceConsumption,
    /// Priority distribution
    pub priority_distribution: HashMap<TaskPriority, u32>,
    /// Queue wait time statistics
    pub queue_wait_stats: QueueWaitStats,
}

/// Average resource consumption statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvgResourceConsumption {
    /// Average CPU usage (%)
    pub avg_cpu_percent: f32,
    /// Average memory usage (MB)
    pub avg_memory_mb: f32,
    /// Average network usage (MB)
    pub avg_network_mb: f32,
    /// Average battery consumption (mAh)
    pub avg_battery_mah: f32,
    /// Average execution time (seconds)
    pub avg_execution_time_seconds: f32,
}

/// Queue wait time statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueWaitStats {
    /// Average wait time (seconds)
    pub avg_wait_time_seconds: f64,
    /// Minimum wait time (seconds)
    pub min_wait_time_seconds: f64,
    /// Maximum wait time (seconds)
    pub max_wait_time_seconds: f64,
    /// 95th percentile wait time (seconds)
    pub p95_wait_time_seconds: f64,
}

/// Resource usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageStats {
    /// CPU usage statistics
    pub cpu_stats: UsageStats,
    /// Memory usage statistics
    pub memory_stats: UsageStats,
    /// Network usage statistics
    pub network_stats: UsageStats,
    /// Battery usage statistics
    pub battery_stats: BatteryUsageStats,
    /// GPU usage statistics (if available)
    pub gpu_stats: Option<UsageStats>,
    /// Storage I/O statistics
    pub storage_stats: StorageStats,
    /// Thermal statistics
    pub thermal_stats: ThermalStats,
}

/// Generic usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStats {
    /// Current usage value
    pub current: f32,
    /// Average usage
    pub average: f32,
    /// Minimum usage recorded
    pub minimum: f32,
    /// Maximum usage recorded
    pub maximum: f32,
    /// 95th percentile usage
    pub p95: f32,
    /// Standard deviation
    pub std_deviation: f32,
    /// Total samples collected
    pub sample_count: u64,
}

/// Battery usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatteryUsageStats {
    /// Current battery level (%)
    pub current_level_percent: u8,
    /// Battery drain rate (%/hour)
    pub drain_rate_percent_per_hour: f32,
    /// Average battery level
    pub avg_battery_level_percent: f32,
    /// Time since last charge (hours)
    pub time_since_last_charge_hours: f32,
    /// Charging cycles in session
    pub charging_cycles: u32,
    /// Low battery events
    pub low_battery_events: u32,
    /// Critical battery events
    pub critical_battery_events: u32,
}

/// Storage I/O statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    /// Read operations count
    pub read_operations: u64,
    /// Write operations count
    pub write_operations: u64,
    /// Total bytes read
    pub bytes_read: u64,
    /// Total bytes written
    pub bytes_written: u64,
    /// Average read speed (MB/s)
    pub avg_read_speed_mbps: f32,
    /// Average write speed (MB/s)
    pub avg_write_speed_mbps: f32,
    /// Storage space usage (MB)
    pub storage_usage_mb: u64,
    /// Available storage space (MB)
    pub available_storage_mb: u64,
}

/// Thermal statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalStats {
    /// Current temperature (°C)
    pub current_temperature_celsius: f32,
    /// Average temperature (°C)
    pub avg_temperature_celsius: f32,
    /// Maximum temperature recorded (°C)
    pub max_temperature_celsius: f32,
    /// Thermal events count
    pub thermal_events: u32,
    /// Throttling events count
    pub throttling_events: u32,
    /// Time spent in thermal warning (seconds)
    pub time_in_thermal_warning_seconds: u64,
    /// Temperature trend
    pub temperature_trend: TemperatureTrend,
}

/// Temperature trend analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemperatureTrend {
    Stable,
    Rising,
    Falling,
    Oscillating,
}

/// Performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    /// Inference performance statistics
    pub inference_stats: InferencePerformanceStats,
    /// Memory performance statistics
    pub memory_performance_stats: MemoryPerformanceStats,
    /// Network performance statistics
    pub network_performance_stats: NetworkPerformanceStats,
    /// Overall system performance score (0-100)
    pub overall_performance_score: f32,
    /// Performance degradation events
    pub performance_degradation_events: u32,
    /// Performance optimization events
    pub performance_optimization_events: u32,
}

/// Inference performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferencePerformanceStats {
    /// Total inferences performed
    pub total_inferences: u64,
    /// Average inference time (ms)
    pub avg_inference_time_ms: f32,
    /// Inference throughput (inferences/second)
    pub throughput_per_second: f32,
    /// Accuracy statistics
    pub accuracy_stats: AccuracyStats,
    /// Model loading time statistics
    pub model_loading_stats: ModelLoadingStats,
    /// Queue backlog statistics
    pub queue_backlog_stats: QueueBacklogStats,
}

/// Accuracy tracking statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyStats {
    /// Average accuracy score (0-100)
    pub avg_accuracy_score: f32,
    /// Accuracy trend over time
    pub accuracy_trend: AccuracyTrend,
    /// Model drift detection events
    pub model_drift_events: u32,
    /// Accuracy degradation events
    pub accuracy_degradation_events: u32,
}

/// Accuracy trend analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccuracyTrend {
    Stable,
    Improving,
    Degrading,
    Fluctuating,
}

/// Model loading statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLoadingStats {
    /// Total model loads
    pub total_loads: u32,
    /// Average loading time (seconds)
    pub avg_loading_time_seconds: f32,
    /// Cache hit rate (%)
    pub cache_hit_rate_percent: f32,
    /// Failed loads count
    pub failed_loads: u32,
    /// Memory usage for loaded models (MB)
    pub loaded_models_memory_mb: usize,
}

/// Queue backlog statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueBacklogStats {
    /// Current queue size
    pub current_queue_size: usize,
    /// Average queue size
    pub avg_queue_size: f32,
    /// Maximum queue size recorded
    pub max_queue_size: usize,
    /// Queue overflow events
    pub queue_overflow_events: u32,
    /// Average processing time per item (ms)
    pub avg_processing_time_ms: f32,
}

/// Memory performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPerformanceStats {
    /// Memory allocation rate (MB/s)
    pub allocation_rate_mbps: f32,
    /// Memory deallocation rate (MB/s)
    pub deallocation_rate_mbps: f32,
    /// Garbage collection events
    pub gc_events: u32,
    /// Average GC pause time (ms)
    pub avg_gc_pause_time_ms: f32,
    /// Memory fragmentation percentage
    pub fragmentation_percent: f32,
    /// Out of memory events
    pub oom_events: u32,
    /// Memory pressure events
    pub memory_pressure_events: u32,
}

/// Network performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPerformanceStats {
    /// Data transfer rate (MB/s)
    pub transfer_rate_mbps: f32,
    /// Connection success rate (%)
    pub connection_success_rate_percent: f32,
    /// Average latency (ms)
    pub avg_latency_ms: f32,
    /// Timeout events
    pub timeout_events: u32,
    /// Retry events
    pub retry_events: u32,
    /// Data usage statistics
    pub data_usage: DataUsageStats,
}

/// Data usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataUsageStats {
    /// Total data sent (MB)
    pub total_sent_mb: f32,
    /// Total data received (MB)
    pub total_received_mb: f32,
    /// Data usage by task type
    pub usage_by_task_type: HashMap<TaskType, f32>,
    /// Peak bandwidth usage (MB/s)
    pub peak_bandwidth_mbps: f32,
}

/// Error statistics tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorStats {
    /// Total errors recorded
    pub total_errors: u64,
    /// Errors by category
    pub errors_by_category: HashMap<String, u32>,
    /// Errors by severity
    pub errors_by_severity: HashMap<ErrorSeverity, u32>,
    /// Recent error patterns
    pub recent_error_patterns: Vec<ErrorPattern>,
    /// Error rate (errors/hour)
    pub error_rate_per_hour: f32,
    /// Error resolution statistics
    pub error_resolution_stats: ErrorResolutionStats,
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Error pattern detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPattern {
    /// Pattern identifier
    pub pattern_id: String,
    /// Error message pattern
    pub message_pattern: String,
    /// Occurrence count
    pub occurrence_count: u32,
    /// First occurrence timestamp
    pub first_occurrence_timestamp: u64,
    /// Last occurrence timestamp
    pub last_occurrence_timestamp: u64,
    /// Affected components
    pub affected_components: Vec<String>,
}

/// Error resolution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResolutionStats {
    /// Automatically resolved errors
    pub auto_resolved_errors: u32,
    /// Manually resolved errors
    pub manually_resolved_errors: u32,
    /// Unresolved errors
    pub unresolved_errors: u32,
    /// Average resolution time (minutes)
    pub avg_resolution_time_minutes: f32,
    /// Resolution success rate (%)
    pub resolution_success_rate_percent: f32,
}

/// User interaction statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInteractionStats {
    /// Total interactions
    pub total_interactions: u64,
    /// Interactions by type
    pub interactions_by_type: HashMap<String, u32>,
    /// Average session duration (minutes)
    pub avg_session_duration_minutes: f32,
    /// User engagement score (0-100)
    pub engagement_score: f32,
    /// Feature usage statistics
    pub feature_usage_stats: FeatureUsageStats,
    /// User satisfaction metrics
    pub satisfaction_metrics: UserSatisfactionMetrics,
}

/// Feature usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureUsageStats {
    /// Feature usage counts
    pub feature_usage_counts: HashMap<String, u32>,
    /// Feature popularity ranking
    pub popularity_ranking: Vec<String>,
    /// Unused features list
    pub unused_features: Vec<String>,
    /// Feature adoption rate (%)
    pub adoption_rate_percent: f32,
}

/// User satisfaction metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSatisfactionMetrics {
    /// Overall satisfaction score (0-100)
    pub overall_satisfaction_score: f32,
    /// Performance satisfaction score (0-100)
    pub performance_satisfaction_score: f32,
    /// User feedback count
    pub feedback_count: u32,
    /// Positive feedback percentage
    pub positive_feedback_percent: f32,
    /// App crashes experienced by user
    pub user_experienced_crashes: u32,
}

/// System health monitoring statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthStats {
    /// Overall health score (0-100)
    pub overall_health_score: f32,
    /// Component health scores
    pub component_health_scores: HashMap<String, f32>,
    /// Health trend over time
    pub health_trend: HealthTrend,
    /// Critical issues count
    pub critical_issues_count: u32,
    /// Warning issues count
    pub warning_issues_count: u32,
    /// System uptime (hours)
    pub system_uptime_hours: f32,
    /// Stability metrics
    pub stability_metrics: StabilityMetrics,
}

/// Health trend analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthTrend {
    Excellent,
    Good,
    Fair,
    Poor,
    Critical,
}

/// System stability metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityMetrics {
    /// Crash-free sessions percentage
    pub crash_free_sessions_percent: f32,
    /// Mean time between failures (hours)
    pub mtbf_hours: f32,
    /// System availability percentage
    pub availability_percent: f32,
    /// Recovery time statistics
    pub recovery_time_stats: RecoveryTimeStats,
}

/// Recovery time statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryTimeStats {
    /// Average recovery time (minutes)
    pub avg_recovery_time_minutes: f32,
    /// Fastest recovery time (minutes)
    pub fastest_recovery_minutes: f32,
    /// Slowest recovery time (minutes)
    pub slowest_recovery_minutes: f32,
    /// Successful recoveries count
    pub successful_recoveries: u32,
    /// Failed recovery attempts
    pub failed_recovery_attempts: u32,
}

impl LifecycleStats {
    /// Create new lifecycle statistics tracker
    pub fn new() -> Self {
        let start_timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs();

        Self {
            start_timestamp,
            app_state_stats: AppStateStats::new(),
            task_execution_stats: HashMap::new(),
            resource_usage_stats: ResourceUsageStats::new(),
            performance_stats: PerformanceStats::new(),
            error_stats: ErrorStats::new(),
            user_interaction_stats: UserInteractionStats::new(),
            system_health_stats: SystemHealthStats::new(),
        }
    }

    /// Update statistics with new data point
    pub fn update_stats(&mut self, update: StatsUpdate) {
        match update {
            StatsUpdate::AppStateTransition {
                from,
                to,
                timestamp,
            } => {
                self.app_state_stats.record_transition(from, to, timestamp);
            },
            StatsUpdate::TaskExecution {
                task_type,
                execution_stats,
            } => {
                self.task_execution_stats
                    .entry(task_type)
                    .or_insert_with(TaskExecutionStats::new)
                    .update(execution_stats);
            },
            StatsUpdate::ResourceUsage {
                cpu,
                memory,
                network,
                battery,
            } => {
                self.resource_usage_stats.update(cpu, memory, network, battery);
            },
            StatsUpdate::Performance {
                inference_time,
                accuracy,
                throughput,
            } => {
                self.performance_stats
                    .update_inference_stats(inference_time, accuracy, throughput);
            },
            StatsUpdate::Error {
                severity,
                category,
                message,
            } => {
                self.error_stats.record_error(severity, category, message);
            },
            StatsUpdate::UserInteraction {
                interaction_type,
                duration,
            } => {
                self.user_interaction_stats.record_interaction(interaction_type, duration);
            },
            StatsUpdate::SystemHealth {
                component,
                health_score,
            } => {
                self.system_health_stats.update_component_health(component, health_score);
            },
        }
    }

    /// Generate statistics summary report
    pub fn generate_summary_report(&self) -> StatsSummaryReport {
        StatsSummaryReport {
            collection_period_hours: self.get_collection_period_hours(),
            overall_performance_score: self.performance_stats.overall_performance_score,
            system_health_score: self.system_health_stats.overall_health_score,
            error_rate_per_hour: self.error_stats.error_rate_per_hour,
            user_engagement_score: self.user_interaction_stats.engagement_score,
            resource_efficiency_score: self.calculate_resource_efficiency_score(),
            key_metrics: self.extract_key_metrics(),
            recommendations: self.generate_recommendations(),
        }
    }

    /// Get collection period in hours
    pub fn get_collection_period_hours(&self) -> f32 {
        let current_timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs();
        (current_timestamp - self.start_timestamp) as f32 / 3600.0
    }

    /// Calculate overall resource efficiency score
    fn calculate_resource_efficiency_score(&self) -> f32 {
        let cpu_efficiency = 100.0 - self.resource_usage_stats.cpu_stats.average;
        let memory_efficiency =
            (1.0 - (self.resource_usage_stats.memory_stats.average / 100.0)) * 100.0;
        let battery_efficiency =
            100.0 - self.resource_usage_stats.battery_stats.drain_rate_percent_per_hour;

        (cpu_efficiency + memory_efficiency + battery_efficiency) / 3.0
    }

    /// Extract key performance metrics
    fn extract_key_metrics(&self) -> Vec<KeyMetric> {
        vec![
            KeyMetric {
                name: "Average Inference Time".to_string(),
                value: self.performance_stats.inference_stats.avg_inference_time_ms,
                unit: "ms".to_string(),
                trend: MetricTrend::Stable,
            },
            KeyMetric {
                name: "Success Rate".to_string(),
                value: self.calculate_overall_success_rate(),
                unit: "%".to_string(),
                trend: MetricTrend::Stable,
            },
            KeyMetric {
                name: "Memory Usage".to_string(),
                value: self.resource_usage_stats.memory_stats.average,
                unit: "MB".to_string(),
                trend: MetricTrend::Stable,
            },
        ]
    }

    /// Calculate overall success rate
    fn calculate_overall_success_rate(&self) -> f32 {
        if self.task_execution_stats.is_empty() {
            return 100.0;
        }

        let total_tasks: u64 = self.task_execution_stats.values().map(|s| s.total_executed).sum();
        let successful_tasks: u64 =
            self.task_execution_stats.values().map(|s| s.successful_executions).sum();

        if total_tasks == 0 {
            100.0
        } else {
            (successful_tasks as f32 / total_tasks as f32) * 100.0
        }
    }

    /// Generate performance recommendations
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if self.resource_usage_stats.memory_stats.average > 80.0 {
            recommendations.push(
                "Consider enabling aggressive memory cleanup to reduce memory usage".to_string(),
            );
        }

        if self.resource_usage_stats.battery_stats.drain_rate_percent_per_hour > 20.0 {
            recommendations
                .push("High battery drain detected. Enable battery optimization mode".to_string());
        }

        if self.error_stats.error_rate_per_hour > 5.0 {
            recommendations
                .push("Error rate is high. Review error patterns and implement fixes".to_string());
        }

        if self.performance_stats.inference_stats.avg_inference_time_ms > 1000.0 {
            recommendations.push(
                "Inference time is high. Consider model optimization or hardware acceleration"
                    .to_string(),
            );
        }

        recommendations
    }
}

/// Statistics update events
#[derive(Debug, Clone)]
pub enum StatsUpdate {
    AppStateTransition {
        from: AppState,
        to: AppState,
        timestamp: u64,
    },
    TaskExecution {
        task_type: TaskType,
        execution_stats: TaskExecutionUpdate,
    },
    ResourceUsage {
        cpu: f32,
        memory: f32,
        network: f32,
        battery: f32,
    },
    Performance {
        inference_time: f32,
        accuracy: f32,
        throughput: f32,
    },
    Error {
        severity: ErrorSeverity,
        category: String,
        message: String,
    },
    UserInteraction {
        interaction_type: String,
        duration: u64,
    },
    SystemHealth {
        component: String,
        health_score: f32,
    },
}

/// Task execution update data
#[derive(Debug, Clone)]
pub struct TaskExecutionUpdate {
    pub execution_time_seconds: f64,
    pub success: bool,
    pub priority: TaskPriority,
    pub resource_usage: AvgResourceConsumption,
    pub wait_time_seconds: f64,
}

/// Statistics summary report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsSummaryReport {
    pub collection_period_hours: f32,
    pub overall_performance_score: f32,
    pub system_health_score: f32,
    pub error_rate_per_hour: f32,
    pub user_engagement_score: f32,
    pub resource_efficiency_score: f32,
    pub key_metrics: Vec<KeyMetric>,
    pub recommendations: Vec<String>,
}

/// Key metric representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyMetric {
    pub name: String,
    pub value: f32,
    pub unit: String,
    pub trend: MetricTrend,
}

/// Metric trend analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricTrend {
    Improving,
    Stable,
    Degrading,
    Volatile,
}

// Implementation blocks for default constructors
impl AppStateStats {
    fn new() -> Self {
        Self {
            time_in_state: HashMap::new(),
            transition_counts: HashMap::new(),
            avg_transition_interval_seconds: 0.0,
            total_transitions: 0,
            last_state_change_timestamp: 0,
            state_change_frequency_per_hour: 0.0,
        }
    }

    fn record_transition(&mut self, from: AppState, to: AppState, timestamp: u64) {
        let transition_key = format!("{:?}->{:?}", from, to);
        *self.transition_counts.entry(transition_key).or_insert(0) += 1;
        self.total_transitions += 1;
        self.last_state_change_timestamp = timestamp;
    }
}

impl TaskExecutionStats {
    fn new() -> Self {
        Self {
            total_executed: 0,
            successful_executions: 0,
            failed_executions: 0,
            cancelled_executions: 0,
            avg_execution_time_seconds: 0.0,
            min_execution_time_seconds: f64::MAX,
            max_execution_time_seconds: 0.0,
            success_rate_percent: 100.0,
            avg_resource_consumption: AvgResourceConsumption::default(),
            priority_distribution: HashMap::new(),
            queue_wait_stats: QueueWaitStats::default(),
        }
    }

    fn update(&mut self, update: TaskExecutionUpdate) {
        self.total_executed += 1;
        if update.success {
            self.successful_executions += 1;
        } else {
            self.failed_executions += 1;
        }

        // Update timing statistics
        self.min_execution_time_seconds =
            self.min_execution_time_seconds.min(update.execution_time_seconds);
        self.max_execution_time_seconds =
            self.max_execution_time_seconds.max(update.execution_time_seconds);

        // Update running average
        let alpha = 0.1;
        self.avg_execution_time_seconds =
            alpha * update.execution_time_seconds + (1.0 - alpha) * self.avg_execution_time_seconds;

        // Update success rate
        self.success_rate_percent =
            (self.successful_executions as f32 / self.total_executed as f32) * 100.0;

        // Update priority distribution
        *self.priority_distribution.entry(update.priority).or_insert(0) += 1;
    }
}

impl Default for AvgResourceConsumption {
    fn default() -> Self {
        Self {
            avg_cpu_percent: 0.0,
            avg_memory_mb: 0.0,
            avg_network_mb: 0.0,
            avg_battery_mah: 0.0,
            avg_execution_time_seconds: 0.0,
        }
    }
}

impl Default for QueueWaitStats {
    fn default() -> Self {
        Self {
            avg_wait_time_seconds: 0.0,
            min_wait_time_seconds: 0.0,
            max_wait_time_seconds: 0.0,
            p95_wait_time_seconds: 0.0,
        }
    }
}

impl ResourceUsageStats {
    fn new() -> Self {
        Self {
            cpu_stats: UsageStats::new(),
            memory_stats: UsageStats::new(),
            network_stats: UsageStats::new(),
            battery_stats: BatteryUsageStats::new(),
            gpu_stats: None,
            storage_stats: StorageStats::new(),
            thermal_stats: ThermalStats::new(),
        }
    }

    fn update(&mut self, cpu: f32, memory: f32, network: f32, battery: f32) {
        self.cpu_stats.update(cpu);
        self.memory_stats.update(memory);
        self.network_stats.update(network);
        self.battery_stats.update(battery);
    }
}

impl UsageStats {
    fn new() -> Self {
        Self {
            current: 0.0,
            average: 0.0,
            minimum: f32::MAX,
            maximum: 0.0,
            p95: 0.0,
            std_deviation: 0.0,
            sample_count: 0,
        }
    }

    fn update(&mut self, value: f32) {
        self.current = value;
        self.minimum = self.minimum.min(value);
        self.maximum = self.maximum.max(value);
        self.sample_count += 1;

        // Update running average
        let alpha = 1.0 / self.sample_count as f32;
        self.average = alpha * value + (1.0 - alpha) * self.average;
    }
}

impl BatteryUsageStats {
    fn new() -> Self {
        Self {
            current_level_percent: 100,
            drain_rate_percent_per_hour: 0.0,
            avg_battery_level_percent: 100.0,
            time_since_last_charge_hours: 0.0,
            charging_cycles: 0,
            low_battery_events: 0,
            critical_battery_events: 0,
        }
    }

    fn update(&mut self, battery_level: f32) {
        self.current_level_percent = battery_level as u8;

        // Update running average
        let alpha = 0.1;
        self.avg_battery_level_percent =
            alpha * battery_level + (1.0 - alpha) * self.avg_battery_level_percent;
    }
}

impl StorageStats {
    fn new() -> Self {
        Self {
            read_operations: 0,
            write_operations: 0,
            bytes_read: 0,
            bytes_written: 0,
            avg_read_speed_mbps: 0.0,
            avg_write_speed_mbps: 0.0,
            storage_usage_mb: 0,
            available_storage_mb: 1000, // Default value
        }
    }
}

impl ThermalStats {
    fn new() -> Self {
        Self {
            current_temperature_celsius: 25.0,
            avg_temperature_celsius: 25.0,
            max_temperature_celsius: 25.0,
            thermal_events: 0,
            throttling_events: 0,
            time_in_thermal_warning_seconds: 0,
            temperature_trend: TemperatureTrend::Stable,
        }
    }
}

impl PerformanceStats {
    fn new() -> Self {
        Self {
            inference_stats: InferencePerformanceStats::new(),
            memory_performance_stats: MemoryPerformanceStats::new(),
            network_performance_stats: NetworkPerformanceStats::new(),
            overall_performance_score: 100.0,
            performance_degradation_events: 0,
            performance_optimization_events: 0,
        }
    }

    fn update_inference_stats(&mut self, inference_time: f32, accuracy: f32, throughput: f32) {
        self.inference_stats.update(inference_time, accuracy, throughput);
    }
}

impl InferencePerformanceStats {
    fn new() -> Self {
        Self {
            total_inferences: 0,
            avg_inference_time_ms: 0.0,
            throughput_per_second: 0.0,
            accuracy_stats: AccuracyStats::new(),
            model_loading_stats: ModelLoadingStats::new(),
            queue_backlog_stats: QueueBacklogStats::new(),
        }
    }

    fn update(&mut self, inference_time: f32, accuracy: f32, throughput: f32) {
        self.total_inferences += 1;

        // Update running averages
        let alpha = 0.1;
        self.avg_inference_time_ms =
            alpha * inference_time + (1.0 - alpha) * self.avg_inference_time_ms;
        self.throughput_per_second =
            alpha * throughput + (1.0 - alpha) * self.throughput_per_second;

        self.accuracy_stats.update(accuracy);
    }
}

impl AccuracyStats {
    fn new() -> Self {
        Self {
            avg_accuracy_score: 100.0,
            accuracy_trend: AccuracyTrend::Stable,
            model_drift_events: 0,
            accuracy_degradation_events: 0,
        }
    }

    fn update(&mut self, accuracy: f32) {
        let alpha = 0.1;
        self.avg_accuracy_score = alpha * accuracy + (1.0 - alpha) * self.avg_accuracy_score;
    }
}

impl ModelLoadingStats {
    fn new() -> Self {
        Self {
            total_loads: 0,
            avg_loading_time_seconds: 0.0,
            cache_hit_rate_percent: 100.0,
            failed_loads: 0,
            loaded_models_memory_mb: 0,
        }
    }
}

impl QueueBacklogStats {
    fn new() -> Self {
        Self {
            current_queue_size: 0,
            avg_queue_size: 0.0,
            max_queue_size: 0,
            queue_overflow_events: 0,
            avg_processing_time_ms: 0.0,
        }
    }
}

impl MemoryPerformanceStats {
    fn new() -> Self {
        Self {
            allocation_rate_mbps: 0.0,
            deallocation_rate_mbps: 0.0,
            gc_events: 0,
            avg_gc_pause_time_ms: 0.0,
            fragmentation_percent: 0.0,
            oom_events: 0,
            memory_pressure_events: 0,
        }
    }
}

impl NetworkPerformanceStats {
    fn new() -> Self {
        Self {
            transfer_rate_mbps: 0.0,
            connection_success_rate_percent: 100.0,
            avg_latency_ms: 0.0,
            timeout_events: 0,
            retry_events: 0,
            data_usage: DataUsageStats::new(),
        }
    }
}

impl DataUsageStats {
    fn new() -> Self {
        Self {
            total_sent_mb: 0.0,
            total_received_mb: 0.0,
            usage_by_task_type: HashMap::new(),
            peak_bandwidth_mbps: 0.0,
        }
    }
}

impl ErrorStats {
    fn new() -> Self {
        Self {
            total_errors: 0,
            errors_by_category: HashMap::new(),
            errors_by_severity: HashMap::new(),
            recent_error_patterns: Vec::new(),
            error_rate_per_hour: 0.0,
            error_resolution_stats: ErrorResolutionStats::new(),
        }
    }

    fn record_error(&mut self, severity: ErrorSeverity, category: String, _message: String) {
        self.total_errors += 1;
        *self.errors_by_category.entry(category).or_insert(0) += 1;
        *self.errors_by_severity.entry(severity).or_insert(0) += 1;
    }
}

impl ErrorResolutionStats {
    fn new() -> Self {
        Self {
            auto_resolved_errors: 0,
            manually_resolved_errors: 0,
            unresolved_errors: 0,
            avg_resolution_time_minutes: 0.0,
            resolution_success_rate_percent: 100.0,
        }
    }
}

impl UserInteractionStats {
    fn new() -> Self {
        Self {
            total_interactions: 0,
            interactions_by_type: HashMap::new(),
            avg_session_duration_minutes: 0.0,
            engagement_score: 100.0,
            feature_usage_stats: FeatureUsageStats::new(),
            satisfaction_metrics: UserSatisfactionMetrics::new(),
        }
    }

    fn record_interaction(&mut self, interaction_type: String, _duration: u64) {
        self.total_interactions += 1;
        *self.interactions_by_type.entry(interaction_type).or_insert(0) += 1;
    }
}

impl FeatureUsageStats {
    fn new() -> Self {
        Self {
            feature_usage_counts: HashMap::new(),
            popularity_ranking: Vec::new(),
            unused_features: Vec::new(),
            adoption_rate_percent: 100.0,
        }
    }
}

impl UserSatisfactionMetrics {
    fn new() -> Self {
        Self {
            overall_satisfaction_score: 100.0,
            performance_satisfaction_score: 100.0,
            feedback_count: 0,
            positive_feedback_percent: 100.0,
            user_experienced_crashes: 0,
        }
    }
}

impl SystemHealthStats {
    fn new() -> Self {
        Self {
            overall_health_score: 100.0,
            component_health_scores: HashMap::new(),
            health_trend: HealthTrend::Excellent,
            critical_issues_count: 0,
            warning_issues_count: 0,
            system_uptime_hours: 0.0,
            stability_metrics: StabilityMetrics::new(),
        }
    }

    fn update_component_health(&mut self, component: String, health_score: f32) {
        self.component_health_scores.insert(component, health_score);

        // Recalculate overall health score
        if !self.component_health_scores.is_empty() {
            let total_score: f32 = self.component_health_scores.values().sum();
            self.overall_health_score = total_score / self.component_health_scores.len() as f32;
        }
    }
}

impl StabilityMetrics {
    fn new() -> Self {
        Self {
            crash_free_sessions_percent: 100.0,
            mtbf_hours: 24.0,
            availability_percent: 100.0,
            recovery_time_stats: RecoveryTimeStats::new(),
        }
    }
}

impl RecoveryTimeStats {
    fn new() -> Self {
        Self {
            avg_recovery_time_minutes: 0.0,
            fastest_recovery_minutes: 0.0,
            slowest_recovery_minutes: 0.0,
            successful_recoveries: 0,
            failed_recovery_attempts: 0,
        }
    }
}

impl Default for LifecycleStats {
    fn default() -> Self {
        Self::new()
    }
}
