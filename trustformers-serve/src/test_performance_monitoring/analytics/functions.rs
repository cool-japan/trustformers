//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::historical_data::DataQualityMetrics;
use super::super::metrics::*;
use super::types::*;
use crate::performance_optimizer::test_characterization::types::{
    DegradationPattern, FaultToleranceMetrics, OptimalResourceAllocation, ResourceScalingEfficiency,
};
use chrono::Utc;
use std::time::{Duration, SystemTime};

use super::types::{
    CpuProfile, DistributionType, IoProfile, MemoryProfile, NetworkProfile, Percentiles,
    PerformanceCharacteristics, ReliabilityProfile, ScalabilityProfile, StatisticalSummary,
    TimeProfile,
};

pub(super) fn determine_trend_direction(values: &[f64]) -> Trend {
    if values.len() < 2 {
        return Trend::Unknown;
    }
    let first = values.first().copied().unwrap_or(0.0);
    let last = values.last().copied().unwrap_or(0.0);
    let delta = last - first;
    let tolerance = first.abs().max(1.0) * 0.05;
    if delta.abs() <= tolerance {
        Trend::Stable
    } else if delta > 0.0 {
        Trend::Increasing
    } else {
        Trend::Decreasing
    }
}
pub(super) fn extract_metric_name(series_key: &str) -> String {
    series_key
        .rsplit_once('_')
        .map(|(_, metric)| metric.to_string())
        .unwrap_or_else(|| series_key.to_string())
}
pub(super) fn extract_test_id(series_id: &str) -> String {
    series_id.split('_').next().unwrap_or(series_id).to_string()
}
pub(super) fn metric_unit(metric_name: &str) -> &'static str {
    match metric_name {
        name if name.contains("execution") => "seconds",
        name if name.contains("memory") => "bytes",
        name if name.contains("cpu") => "percent",
        _ => "units",
    }
}
pub(super) fn collect_execution_times(metrics_data: &[ComprehensiveTestMetrics]) -> Vec<f64> {
    metrics_data
        .iter()
        .map(|m| m.execution_metrics.execution_time.as_secs_f64())
        .collect()
}
pub(super) fn fallback_summary(values: &[f64]) -> StatisticalSummary {
    let mean = if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    };
    StatisticalSummary {
        mean,
        median: mean,
        mode: None,
        standard_deviation: 0.0,
        variance: 0.0,
        skewness: 0.0,
        kurtosis: 0.0,
        percentiles: Percentiles {
            p1: mean,
            p5: mean,
            p10: mean,
            p25: mean,
            p50: mean,
            p75: mean,
            p90: mean,
            p95: mean,
            p99: mean,
        },
        range: (mean, mean),
        interquartile_range: 0.0,
    }
}
pub(super) fn build_performance_characteristics(
    metrics_data: &[ComprehensiveTestMetrics],
    summary: &StatisticalSummary,
) -> PerformanceCharacteristics {
    let mut memory_values: Vec<f64> =
        metrics_data.iter().map(|m| m.execution_metrics.memory_peak as f64).collect();
    if memory_values.is_empty() {
        memory_values.push(0.0);
    }
    let memory_typical = memory_values.iter().sum::<f64>() / memory_values.len() as f64;
    let memory_min = memory_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let memory_max = memory_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut cpu_values: Vec<f64> =
        metrics_data.iter().map(|m| m.execution_metrics.cpu_usage_percent).collect();
    if cpu_values.is_empty() {
        cpu_values.push(0.0);
    }
    let cpu_mean = cpu_values.iter().sum::<f64>() / cpu_values.len() as f64;
    let cpu_peak = cpu_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let latest = metrics_data.last();
    PerformanceCharacteristics {
        execution_time_profile: TimeProfile {
            typical_execution_time: Duration::from_secs_f64(summary.mean),
            fastest_execution_time: Duration::from_secs_f64(summary.range.0.max(0.0)),
            slowest_execution_time: Duration::from_secs_f64(summary.range.1.max(0.0)),
            execution_time_variance: summary.variance,
            time_distribution: DistributionType::Normal,
            seasonal_patterns: Vec::new(),
        },
        memory_usage_profile: MemoryProfile {
            typical_peak_memory: memory_typical as u64,
            minimum_memory_required: memory_min.max(0.0) as u64,
            maximum_memory_observed: memory_max.max(0.0) as u64,
            memory_growth_pattern: GrowthPattern::default(),
            memory_leak_indicators: Vec::new(),
            gc_impact: None,
        },
        cpu_usage_profile: CpuProfile {
            typical_cpu_usage: cpu_mean,
            peak_cpu_usage: cpu_peak,
            cpu_efficiency_score: 0.0,
            thread_utilization: ThreadUtilization::default(),
            parallelization_effectiveness: 0.0,
            cpu_bound_phases: Vec::new(),
        },
        io_performance_profile: IoProfile {
            typical_io_rate: 0.0,
            io_pattern_type: IoPattern::default(),
            read_write_ratio: 0.0,
            io_latency_characteristics: LatencyCharacteristics::default(),
            disk_usage_pattern: DiskUsagePattern::default(),
            io_bottlenecks: Vec::new(),
        },
        network_performance_profile: NetworkProfile {
            typical_network_usage: 0.0,
            network_pattern_type: NetworkPattern::default(),
            connection_characteristics: ConnectionCharacteristics::default(),
            bandwidth_utilization: BandwidthUtilization::default(),
            network_latency_profile: NetworkLatencyProfile::default(),
            network_reliability: NetworkReliability::default(),
        },
        reliability_profile: ReliabilityProfile {
            success_rate_trend: Trend::Stable,
            failure_pattern_analysis: FailurePatternAnalysis {
                pattern_detected: false,
                pattern_type: String::new(),
                frequency: 0.0,
            },
            recovery_characteristics: RecoveryCharacteristics {
                recovery_time_ms: latest.map(|m| m.reliability_metrics.mttr_minutes).unwrap_or(0.0)
                    * 60.0
                    * 1000.0,
                success_rate: latest
                    .map(|m| m.reliability_metrics.success_rate_percent / 100.0)
                    .unwrap_or(1.0),
                retry_count: 0,
            },
            stability_indicators: StabilityIndicators {
                stability_score: latest
                    .map(|m| m.reliability_metrics.stability_index)
                    .unwrap_or(1.0),
                variance: 0.0,
                trend: "stable".to_string(),
            },
            fault_tolerance_metrics: FaultToleranceMetrics {
                failure_rate: latest
                    .map(|m| m.reliability_metrics.failure_rate_percent / 100.0)
                    .unwrap_or(0.0),
                recovery_time: latest
                    .map(|m| Duration::from_secs_f64(m.reliability_metrics.mttr_minutes / 60.0))
                    .unwrap_or(Duration::from_secs(0)),
                fault_detection_time: Duration::from_secs(0),
                resilience_score: latest
                    .map(|m| m.reliability_metrics.resilience_score)
                    .unwrap_or(0.0),
            },
        },
        scalability_profile: ScalabilityProfile {
            scalability_trend: Trend::Stable,
            performance_degradation_pattern: DegradationPattern {
                degradation_rate: 0.0,
                start_time: Utc::now(),
                severity: "none".to_string(),
            },
            resource_scaling_efficiency: ResourceScalingEfficiency {
                scaling_factor: 1.0,
                efficiency_score: 0.0,
                scalability_limit: 0,
                optimal_resource_count: 0,
            },
            bottleneck_emergence_points: Vec::new(),
            optimal_resource_allocation: OptimalResourceAllocation {
                cpu_allocation: cpu_mean,
                memory_allocation: memory_typical as u64,
                thread_count: latest
                    .map(|m| m.execution_metrics.thread_count as usize)
                    .unwrap_or(0),
            },
        },
    }
}
pub(super) fn percentage_delta(current: f64, baseline: f64) -> f64 {
    if baseline.abs() < f64::EPSILON {
        0.0
    } else {
        ((current - baseline) / baseline) * 100.0
    }
}
pub(super) fn default_data_quality_metrics(now: SystemTime) -> DataQualityMetrics {
    DataQualityMetrics {
        completeness_score: 1.0,
        accuracy_score: 1.0,
        consistency_score: 1.0,
        timeliness_score: 1.0,
        validity_score: 1.0,
        overall_quality_score: 1.0,
        quality_issues: Vec::new(),
        last_quality_check: now,
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analytics_engine_creation() {
        let config = AnalyticsConfig::default();
        let engine = PerformanceAnalyticsEngine::new(config);
        assert_eq!(engine.statistical_analyzer.window_size, 100);
        assert_eq!(engine.statistical_analyzer.confidence_level, 0.95);
    }
    #[test]
    fn test_statistical_summary_calculation() {
        let analyzer = StatisticalAnalyzer {
            window_size: 100,
            confidence_level: 0.95,
            statistical_methods: vec![],
            outlier_detection_config: OutlierDetectionConfig::default(),
        };
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let summary = analyzer.calculate_descriptive_statistics(&data).unwrap();
        assert_eq!(summary.mean, 3.0);
        assert_eq!(summary.median, 3.0);
        assert_eq!(summary.range, (1.0, 5.0));
    }
    #[test]
    fn test_percentile_calculation() {
        let analyzer = StatisticalAnalyzer {
            window_size: 100,
            confidence_level: 0.95,
            statistical_methods: vec![],
            outlier_detection_config: OutlierDetectionConfig::default(),
        };
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(analyzer.percentile(&data, 0.5), 5.0);
        assert_eq!(analyzer.percentile(&data, 0.25), 3.0);
        assert_eq!(analyzer.percentile(&data, 0.75), 8.0);
    }
    #[test]
    fn test_outlier_detection() {
        let analyzer = StatisticalAnalyzer {
            window_size: 100,
            confidence_level: 0.95,
            statistical_methods: vec![],
            outlier_detection_config: OutlierDetectionConfig::default(),
        };
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let outlier_analysis = analyzer.detect_outliers(&data).unwrap();
        assert_eq!(outlier_analysis.outliers.len(), 1);
        assert_eq!(outlier_analysis.outliers[0].value, 100.0);
        assert!(outlier_analysis.outliers[0].score > 0.0);
    }
}
