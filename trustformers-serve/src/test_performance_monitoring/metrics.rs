//! Performance Metrics Models and Data Structures
//!
//! This module defines the core performance metrics that are collected,
//! analyzed, and monitored throughout the test performance monitoring system.

use super::types::*;
use crate::performance_optimizer::test_characterization::pattern_engine::SeverityLevel;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use trustformers_mobile::network_adaptation::types::TimePeriod;

/// Comprehensive execution performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    pub test_id: String,
    pub execution_time: Duration,
    pub setup_time: Duration,
    pub teardown_time: Duration,
    pub assertion_time: Duration,
    pub total_time: Duration,
    pub memory_peak: u64,
    pub memory_average: u64,
    pub cpu_usage_percent: f64,
    pub thread_count: u32,
    pub context_switches: u64,
    pub page_faults: u64,
    pub io_operations: IoMetrics,
    pub network_operations: NetworkMetrics,
    pub custom_metrics: HashMap<String, MetricValue>,
    pub timestamp: SystemTime,
    pub test_status: TestStatus,
    pub failure_reason: Option<String>,
}

/// Input/Output operation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoMetrics {
    pub read_operations: u64,
    pub write_operations: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub read_latency_avg: Duration,
    pub write_latency_avg: Duration,
    pub read_latency_p95: Duration,
    pub write_latency_p95: Duration,
    pub read_latency_p99: Duration,
    pub write_latency_p99: Duration,
    pub disk_usage_mb: f64,
    pub file_handles_opened: u32,
    pub file_handles_closed: u32,
    pub temporary_files_created: u32,
}

/// Network operation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    pub connections_opened: u32,
    pub connections_closed: u32,
    pub connections_active: u32,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub packets_sent: u64,
    pub packets_received: u64,
    pub request_count: u64,
    pub response_count: u64,
    pub error_count: u64,
    pub timeout_count: u64,
    pub latency_avg: Duration,
    pub latency_p95: Duration,
    pub latency_p99: Duration,
    pub throughput_mbps: f64,
    pub connection_pool_size: u32,
    pub dns_lookup_time: Duration,
}

/// System-level performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub timestamp: SystemTime,
    pub cpu_usage_percent: f64,
    pub cpu_load_avg_1min: f64,
    pub cpu_load_avg_5min: f64,
    pub cpu_load_avg_15min: f64,
    pub memory_total_mb: u64,
    pub memory_used_mb: u64,
    pub memory_available_mb: u64,
    pub memory_cached_mb: u64,
    pub memory_buffers_mb: u64,
    pub swap_total_mb: u64,
    pub swap_used_mb: u64,
    pub disk_usage_percent: f64,
    pub disk_read_iops: u64,
    pub disk_write_iops: u64,
    pub disk_read_throughput_mbps: f64,
    pub disk_write_throughput_mbps: f64,
    pub network_rx_mbps: f64,
    pub network_tx_mbps: f64,
    pub open_file_descriptors: u32,
    pub process_count: u32,
    pub thread_count: u32,
    pub tcp_connections: u32,
    pub system_uptime: Duration,
    pub temperature_celsius: Option<f64>,
}

/// Parallelization and concurrency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelizationMetrics {
    pub test_id: String,
    pub parallel_execution_count: u32,
    pub concurrent_threads: u32,
    pub thread_pool_size: u32,
    pub active_tasks: u32,
    pub queued_tasks: u32,
    pub completed_tasks: u32,
    pub failed_tasks: u32,
    pub task_wait_time_avg: Duration,
    pub task_execution_time_avg: Duration,
    pub thread_utilization_percent: f64,
    pub lock_contention_count: u64,
    pub deadlock_count: u64,
    pub context_switch_rate: f64,
    pub synchronization_overhead: Duration,
    pub parallelization_efficiency: f64,
    pub speedup_ratio: f64,
    pub scalability_factor: f64,
    pub resource_sharing_conflicts: u64,
    pub timestamp: SystemTime,
}

/// Test efficiency and optimization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyMetrics {
    pub test_id: String,
    pub efficiency_score: f64,
    pub resource_utilization_score: f64,
    pub time_efficiency_score: f64,
    pub memory_efficiency_score: f64,
    pub cpu_efficiency_score: f64,
    pub io_efficiency_score: f64,
    pub cache_hit_ratio: f64,
    pub cache_miss_ratio: f64,
    pub optimization_opportunities: Vec<OptimizationSuggestion>,
    pub bottleneck_analysis: BottleneckAnalysis,
    pub performance_regression_risk: RegressionRisk,
    pub baseline_comparison: Option<BaselineComparison>,
    pub trend_analysis: TrendAnalysis,
    pub timestamp: SystemTime,
}

/// Optimization suggestion with priority and impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    pub suggestion_id: String,
    pub category: OptimizationCategory,
    pub description: String,
    pub impact_level: ImpactLevel,
    pub implementation_effort: EffortLevel,
    pub expected_improvement_percent: f64,
    pub affected_metrics: Vec<String>,
    pub implementation_steps: Vec<String>,
    pub risks: Vec<String>,
    pub priority_score: f64,
}

/// Bottleneck identification and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    pub primary_bottleneck: Option<BottleneckType>,
    pub secondary_bottlenecks: Vec<BottleneckType>,
    pub bottleneck_severity: SeverityLevel,
    pub impact_on_performance: f64,
    pub resolution_complexity: ComplexityLevel,
    pub affected_components: Vec<String>,
    pub mitigation_strategies: Vec<String>,
    pub estimated_resolution_time: Duration,
}

/// Performance regression risk assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionRisk {
    pub risk_level: RiskLevel,
    pub confidence_score: f64,
    pub risk_factors: Vec<RiskFactor>,
    pub historical_patterns: Vec<String>,
    pub mitigation_recommendations: Vec<String>,
    pub monitoring_recommendations: Vec<String>,
}

/// Baseline performance comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineComparison {
    pub baseline_id: String,
    pub baseline_timestamp: SystemTime,
    pub current_vs_baseline_percent: f64,
    pub performance_delta: PerformanceDelta,
    pub significant_changes: Vec<SignificantChange>,
    pub regression_detected: bool,
    pub improvement_detected: bool,
    pub stability_assessment: StabilityAssessment,
}

/// Performance change measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceDelta {
    pub execution_time_delta_percent: f64,
    pub memory_usage_delta_percent: f64,
    pub cpu_usage_delta_percent: f64,
    pub throughput_delta_percent: f64,
    pub error_rate_delta_percent: f64,
    pub latency_delta_percent: f64,
    pub overall_performance_delta: f64,
}

/// Significant performance change detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignificantChange {
    pub metric_name: String,
    pub change_type: ChangeType,
    pub magnitude_percent: f64,
    pub statistical_significance: f64,
    pub confidence_interval: (f64, f64),
    pub change_detection_method: DetectionMethod,
    pub impact_assessment: ImpactAssessment,
}

/// Reliability and stability metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityMetrics {
    pub test_id: String,
    pub success_rate_percent: f64,
    pub failure_rate_percent: f64,
    pub flakiness_score: f64,
    pub consistency_score: f64,
    pub stability_index: f64,
    pub mtbf_hours: f64,   // Mean Time Between Failures
    pub mttr_minutes: f64, // Mean Time To Recovery
    pub error_patterns: Vec<ErrorPattern>,
    pub failure_modes: Vec<FailureMode>,
    pub recovery_metrics: RecoveryMetrics,
    pub resilience_score: f64,
    pub fault_tolerance_rating: f64,
    pub timestamp: SystemTime,
}

/// Error pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPattern {
    pub pattern_id: String,
    pub error_type: String,
    pub frequency: u64,
    pub severity: SeverityLevel,
    pub first_occurrence: SystemTime,
    pub last_occurrence: SystemTime,
    pub pattern_description: String,
    pub root_cause_hypothesis: Vec<String>,
    pub correlation_with_metrics: Vec<String>,
    pub resolution_attempts: Vec<String>,
}

/// Failure mode categorization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureMode {
    pub mode_id: String,
    pub failure_type: FailureType,
    pub description: String,
    pub occurrence_probability: f64,
    pub impact_severity: SeverityLevel,
    pub detection_time_avg: Duration,
    pub recovery_time_avg: Duration,
    pub prevention_strategies: Vec<String>,
    pub mitigation_actions: Vec<String>,
}

/// Recovery and fault handling metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryMetrics {
    pub automatic_recovery_success_rate: f64,
    pub manual_intervention_required_rate: f64,
    pub average_recovery_time: Duration,
    pub recovery_attempts_avg: f64,
    pub cascade_failure_prevention_rate: f64,
    pub graceful_degradation_effectiveness: f64,
    pub rollback_success_rate: f64,
    pub data_integrity_preservation_rate: f64,
}

/// Aggregated metrics for time periods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    pub time_period: TimePeriod,
    pub start_time: SystemTime,
    pub end_time: SystemTime,
    pub total_tests_executed: u64,
    pub total_tests_passed: u64,
    pub total_tests_failed: u64,
    pub average_execution_time: Duration,
    pub p50_execution_time: Duration,
    pub p90_execution_time: Duration,
    pub p95_execution_time: Duration,
    pub p99_execution_time: Duration,
    pub max_execution_time: Duration,
    pub min_execution_time: Duration,
    pub execution_time_variance: f64,
    pub execution_time_std_dev: f64,
    pub average_memory_usage: u64,
    pub peak_memory_usage: u64,
    pub average_cpu_usage: f64,
    pub peak_cpu_usage: f64,
    pub throughput_tests_per_second: f64,
    pub error_rate_percent: f64,
    pub flakiness_incidents: u64,
    pub performance_regressions: u64,
    pub optimization_opportunities_identified: u64,
}

/// Real-time streaming metrics for live monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingMetrics {
    pub stream_id: String,
    pub test_id: String,
    pub timestamp: SystemTime,
    pub elapsed_time: Duration,
    pub current_phase: TestPhase,
    pub progress_percent: f64,
    pub instantaneous_cpu: f64,
    pub instantaneous_memory: u64,
    pub instantaneous_io_rate: f64,
    pub instantaneous_network_rate: f64,
    pub live_error_count: u64,
    pub live_warning_count: u64,
    pub performance_indicators: Vec<LivePerformanceIndicator>,
    pub anomaly_flags: Vec<AnomalyFlag>,
    pub prediction_metrics: Option<PredictionMetrics>,
}

/// Live performance indicators for real-time monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LivePerformanceIndicator {
    pub indicator_name: String,
    pub current_value: f64,
    pub threshold_value: f64,
    pub status: IndicatorStatus,
    pub trend_direction: TrendDirection,
    pub deviation_from_baseline: f64,
    pub criticality_level: CriticalityLevel,
}

/// Anomaly detection flags
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyFlag {
    pub anomaly_type: AnomalyType,
    pub description: String,
    pub severity: SeverityLevel,
    pub confidence_score: f64,
    pub affected_metrics: Vec<String>,
    pub detection_timestamp: SystemTime,
    pub expected_resolution_time: Option<Duration>,
}

/// Predictive performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionMetrics {
    pub predicted_completion_time: SystemTime,
    pub predicted_final_duration: Duration,
    pub predicted_peak_memory: u64,
    pub predicted_success_probability: f64,
    pub confidence_intervals: PredictionConfidenceIntervals,
    pub prediction_model_accuracy: f64,
    pub prediction_timestamp: SystemTime,
}

/// Prediction confidence intervals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionConfidenceIntervals {
    pub completion_time_ci: (SystemTime, SystemTime),
    pub duration_ci: (Duration, Duration),
    pub memory_ci: (u64, u64),
    pub success_probability_ci: (f64, f64),
    pub confidence_level: f64,
}

/// Comprehensive metric collection for a single test execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveTestMetrics {
    pub test_id: String,
    pub test_name: String,
    pub test_suite: String,
    pub execution_metrics: ExecutionMetrics,
    pub system_metrics: SystemMetrics,
    pub parallelization_metrics: Option<ParallelizationMetrics>,
    pub efficiency_metrics: EfficiencyMetrics,
    pub reliability_metrics: ReliabilityMetrics,
    pub streaming_metrics: Vec<StreamingMetrics>,
    pub custom_metrics: HashMap<String, MetricValue>,
    pub collection_metadata: MetricCollectionMetadata,
}

/// Metadata about metric collection process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricCollectionMetadata {
    pub collection_version: String,
    pub collector_id: String,
    pub collection_start_time: SystemTime,
    pub collection_end_time: SystemTime,
    pub collection_overhead_ms: u64,
    pub metric_completeness_percent: f64,
    pub sampling_rate: f64,
    pub collection_errors: Vec<String>,
    pub data_quality_score: f64,
}

impl ExecutionMetrics {
    /// Calculate total resource usage score
    pub fn total_resource_score(&self) -> f64 {
        let cpu_score = (self.cpu_usage_percent / 100.0) * 0.3;
        let memory_score = (self.memory_peak as f64 / (1024.0 * 1024.0 * 1024.0)) * 0.3; // GB
        let io_score = ((self.io_operations.bytes_read + self.io_operations.bytes_written) as f64
            / (1024.0 * 1024.0))
            * 0.2; // MB
        let network_score = ((self.network_operations.bytes_sent
            + self.network_operations.bytes_received) as f64
            / (1024.0 * 1024.0))
            * 0.2; // MB

        (cpu_score + memory_score + io_score + network_score).min(1.0)
    }

    /// Check if execution meets performance thresholds
    pub fn meets_performance_thresholds(&self, thresholds: &PerformanceThresholds) -> bool {
        self.execution_time <= thresholds.max_execution_time
            && self.memory_peak <= thresholds.max_memory_usage
            && self.cpu_usage_percent <= thresholds.max_cpu_usage
    }

    /// Generate performance summary
    pub fn performance_summary(&self) -> String {
        format!(
            "Test {}: {}ms execution, {:.2}% CPU, {:.2}MB memory peak",
            self.test_id,
            self.execution_time.as_millis(),
            self.cpu_usage_percent,
            self.memory_peak as f64 / (1024.0 * 1024.0)
        )
    }
}

impl SystemMetrics {
    /// Calculate overall system pressure indicator
    pub fn system_pressure_level(&self) -> PressureLevel {
        let cpu_pressure = self.cpu_usage_percent / 100.0;
        let memory_pressure = self.memory_used_mb as f64 / self.memory_total_mb as f64;
        let disk_pressure = self.disk_usage_percent / 100.0;

        let overall_pressure = (cpu_pressure + memory_pressure + disk_pressure) / 3.0;

        match overall_pressure {
            p if p < 0.3 => PressureLevel::Low,
            p if p < 0.6 => PressureLevel::Medium,
            p if p < 0.8 => PressureLevel::High,
            _ => PressureLevel::Critical,
        }
    }

    /// Check if system is under resource pressure
    pub fn is_under_pressure(&self, thresholds: &SystemPressureThresholds) -> bool {
        self.cpu_usage_percent > thresholds.cpu_pressure_threshold
            || (self.memory_used_mb as f64 / self.memory_total_mb as f64)
                > thresholds.memory_pressure_threshold
            || self.disk_usage_percent > thresholds.disk_pressure_threshold
            || self.open_file_descriptors > thresholds.file_descriptor_threshold
    }
}

impl EfficiencyMetrics {
    /// Calculate weighted efficiency score
    pub fn weighted_efficiency_score(&self) -> f64 {
        ((self.time_efficiency_score / 100.0) * 0.25)
            + ((self.memory_efficiency_score / 100.0) * 0.25)
            + ((self.cpu_efficiency_score / 100.0) * 0.25)
            + ((self.io_efficiency_score / 100.0) * 0.15)
            + (self.cache_hit_ratio * 0.1)
    }

    /// Identify top optimization opportunities
    pub fn top_optimization_opportunities(&self, limit: usize) -> Vec<&OptimizationSuggestion> {
        let mut opportunities = self.optimization_opportunities.iter().collect::<Vec<_>>();
        opportunities.sort_by(|a, b| {
            b.priority_score
                .partial_cmp(&a.priority_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        opportunities.into_iter().take(limit).collect()
    }
}

impl ReliabilityMetrics {
    /// Calculate overall reliability score
    pub fn overall_reliability_score(&self) -> f64 {
        (self.success_rate_percent / 100.0) * 0.4
            + (1.0 - self.flakiness_score) * 0.3
            + (self.consistency_score / 100.0) * 0.2
            + (self.resilience_score / 100.0) * 0.1
    }

    /// Check if test is considered flaky
    pub fn is_flaky(&self, threshold: f64) -> bool {
        self.flakiness_score > threshold
    }

    /// Generate reliability assessment
    pub fn reliability_assessment(&self) -> String {
        let score = self.overall_reliability_score();
        let rating = match score {
            s if s >= 0.9 => "Excellent",
            s if s >= 0.8 => "Good",
            s if s >= 0.7 => "Fair",
            s if s >= 0.6 => "Poor",
            _ => "Critical",
        };

        format!(
            "Reliability: {} ({:.1}%) - Success: {:.1}%, Flakiness: {:.1}%, Consistency: {:.1}%",
            rating,
            score * 100.0,
            self.success_rate_percent,
            self.flakiness_score * 100.0,
            self.consistency_score
        )
    }
}

impl Default for IoMetrics {
    fn default() -> Self {
        Self {
            read_operations: 0,
            write_operations: 0,
            bytes_read: 0,
            bytes_written: 0,
            read_latency_avg: Duration::from_millis(0),
            write_latency_avg: Duration::from_millis(0),
            read_latency_p95: Duration::from_millis(0),
            write_latency_p95: Duration::from_millis(0),
            read_latency_p99: Duration::from_millis(0),
            write_latency_p99: Duration::from_millis(0),
            disk_usage_mb: 0.0,
            file_handles_opened: 0,
            file_handles_closed: 0,
            temporary_files_created: 0,
        }
    }
}

impl Default for NetworkMetrics {
    fn default() -> Self {
        Self {
            connections_opened: 0,
            connections_closed: 0,
            connections_active: 0,
            bytes_sent: 0,
            bytes_received: 0,
            packets_sent: 0,
            packets_received: 0,
            request_count: 0,
            response_count: 0,
            error_count: 0,
            timeout_count: 0,
            latency_avg: Duration::from_millis(0),
            latency_p95: Duration::from_millis(0),
            latency_p99: Duration::from_millis(0),
            throughput_mbps: 0.0,
            connection_pool_size: 0,
            dns_lookup_time: Duration::from_millis(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    #[test]
    fn test_execution_metrics_resource_score() {
        let metrics = ExecutionMetrics {
            test_id: "test-001".to_string(),
            execution_time: Duration::from_millis(1000),
            setup_time: Duration::from_millis(100),
            teardown_time: Duration::from_millis(50),
            assertion_time: Duration::from_millis(850),
            total_time: Duration::from_millis(1000),
            memory_peak: 100 * 1024 * 1024,   // 100MB
            memory_average: 80 * 1024 * 1024, // 80MB
            cpu_usage_percent: 50.0,
            thread_count: 4,
            context_switches: 1000,
            page_faults: 10,
            io_operations: IoMetrics::default(),
            network_operations: NetworkMetrics::default(),
            custom_metrics: HashMap::new(),
            timestamp: SystemTime::now(),
            test_status: TestStatus::Passed,
            failure_reason: None,
        };

        let score = metrics.total_resource_score();
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_system_pressure_calculation() {
        let metrics = SystemMetrics {
            timestamp: SystemTime::now(),
            cpu_usage_percent: 80.0,
            cpu_load_avg_1min: 2.5,
            cpu_load_avg_5min: 2.0,
            cpu_load_avg_15min: 1.8,
            memory_total_mb: 8192,
            memory_used_mb: 6144, // 75% usage
            memory_available_mb: 2048,
            memory_cached_mb: 1024,
            memory_buffers_mb: 512,
            swap_total_mb: 4096,
            swap_used_mb: 1024,
            disk_usage_percent: 70.0,
            disk_read_iops: 100,
            disk_write_iops: 50,
            disk_read_throughput_mbps: 50.0,
            disk_write_throughput_mbps: 25.0,
            network_rx_mbps: 10.0,
            network_tx_mbps: 5.0,
            open_file_descriptors: 1000,
            process_count: 200,
            thread_count: 800,
            tcp_connections: 50,
            system_uptime: Duration::from_secs(86400),
            temperature_celsius: Some(65.0),
        };

        let pressure = metrics.system_pressure_level();
        assert!(matches!(pressure, PressureLevel::High));
    }

    #[test]
    fn test_reliability_metrics_assessment() {
        let metrics = ReliabilityMetrics {
            test_id: "reliability-test".to_string(),
            success_rate_percent: 95.0,
            failure_rate_percent: 5.0,
            flakiness_score: 0.1, // 10% flakiness
            consistency_score: 90.0,
            stability_index: 85.0,
            mtbf_hours: 24.0,
            mttr_minutes: 15.0,
            error_patterns: vec![],
            failure_modes: vec![],
            recovery_metrics: RecoveryMetrics {
                automatic_recovery_success_rate: 80.0,
                manual_intervention_required_rate: 20.0,
                average_recovery_time: Duration::from_secs(10 * 60),
                recovery_attempts_avg: 1.5,
                cascade_failure_prevention_rate: 95.0,
                graceful_degradation_effectiveness: 85.0,
                rollback_success_rate: 99.0,
                data_integrity_preservation_rate: 100.0,
            },
            resilience_score: 80.0,
            fault_tolerance_rating: 75.0,
            timestamp: SystemTime::now(),
        };

        let score = metrics.overall_reliability_score();
        assert!(score >= 0.8); // Should be "Good" or better rating

        let assessment = metrics.reliability_assessment();
        assert!(assessment.contains("Good") || assessment.contains("Excellent"));
    }

    #[test]
    fn test_efficiency_metrics_optimization() {
        let optimization = OptimizationSuggestion {
            suggestion_id: "opt-001".to_string(),
            category: OptimizationCategory::Resource,
            description: "Reduce memory allocation frequency".to_string(),
            impact_level: ImpactLevel::High,
            implementation_effort: EffortLevel::Medium,
            expected_improvement_percent: 25.0,
            affected_metrics: vec!["memory_usage".to_string(), "execution_time".to_string()],
            implementation_steps: vec!["Use object pooling".to_string()],
            risks: vec!["Increased complexity".to_string()],
            priority_score: 85.0,
        };

        let metrics = EfficiencyMetrics {
            test_id: "efficiency-test".to_string(),
            efficiency_score: 75.0,
            resource_utilization_score: 80.0,
            time_efficiency_score: 70.0,
            memory_efficiency_score: 65.0,
            cpu_efficiency_score: 85.0,
            io_efficiency_score: 75.0,
            cache_hit_ratio: 0.9,
            cache_miss_ratio: 0.1,
            optimization_opportunities: vec![optimization],
            bottleneck_analysis: BottleneckAnalysis {
                primary_bottleneck: Some(BottleneckType::Memory),
                secondary_bottlenecks: vec![BottleneckType::IO],
                bottleneck_severity: SeverityLevel::Medium,
                impact_on_performance: 20.0,
                resolution_complexity: ComplexityLevel::Moderate,
                affected_components: vec!["memory_allocator".to_string()],
                mitigation_strategies: vec!["Implement pooling".to_string()],
                estimated_resolution_time: Duration::from_secs(8 * 3600),
            },
            performance_regression_risk: RegressionRisk {
                risk_level: RiskLevel::Low,
                confidence_score: 0.85,
                risk_factors: vec![],
                historical_patterns: vec![],
                mitigation_recommendations: vec![],
                monitoring_recommendations: vec![],
            },
            baseline_comparison: None,
            trend_analysis: TrendAnalysis {
                trend_direction: "Stable".to_string(),
                trend_strength: 0.1,
                confidence: 0.9,
            },
            timestamp: SystemTime::now(),
        };

        let weighted_score = metrics.weighted_efficiency_score();
        assert!(weighted_score > 0.0 && weighted_score <= 1.0);

        let top_opportunities = metrics.top_optimization_opportunities(1);
        assert_eq!(top_opportunities.len(), 1);
        assert_eq!(top_opportunities[0].suggestion_id, "opt-001");
    }
}
