//! The analysis engines that consume raw metrics and produce bottleneck/regression/trend/alert results.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque};
use std::time::Instant;
use trustformers_core::error::Result;

use super::analysis_types::{
    AlertSeverity, AlertType, BottleneckSeverity, BottleneckType, OptimizationCategory,
    OptimizationDifficulty, OptimizationPriority, OptimizationSuggestion, PerformanceAlert,
    PerformanceBottleneck, PerformancePattern, PerformanceRegression, PerformanceTrend,
};
use super::config_types::{AlertThresholds, BottleneckConfig};
use super::metrics_types::{MemoryPressureLevel, MetricsSnapshot};

/// Alert system for performance monitoring
pub(super) struct AlertSystem {
    pub(super) thresholds: AlertThresholds,
    pub(super) active_alerts: Vec<PerformanceAlert>,
    pub(super) alert_history: VecDeque<PerformanceAlert>,
}
impl AlertSystem {
    pub(super) fn new(thresholds: AlertThresholds) -> Self {
        Self {
            thresholds,
            active_alerts: vec![],
            alert_history: VecDeque::new(),
        }
    }

    pub(super) fn check_thresholds(
        &mut self,
        metrics: &MetricsSnapshot,
    ) -> Result<Vec<PerformanceAlert>> {
        let mut alerts = vec![];

        // CPU threshold check
        if metrics.platform_metrics.cpu_metrics.utilization_percent
            > self.thresholds.cpu_threshold_percent
        {
            alerts.push(PerformanceAlert {
                alert_type: AlertType::HighCpuUsage,
                severity: AlertSeverity::Warning,
                message: format!(
                    "CPU utilization is {:.1}%, exceeding threshold of {:.1}%",
                    metrics.platform_metrics.cpu_metrics.utilization_percent,
                    self.thresholds.cpu_threshold_percent
                ),
                trigger_value: metrics.platform_metrics.cpu_metrics.utilization_percent,
                threshold_value: self.thresholds.cpu_threshold_percent,
                timestamp: Instant::now(),
                duration_ms: 0,
                suggested_actions: vec![
                    "Reduce inference frequency".to_string(),
                    "Enable CPU throttling".to_string(),
                    "Optimize model computation".to_string(),
                ],
            });
        }

        // Memory threshold check
        let memory_usage_percent = if metrics.platform_metrics.memory_metrics.total_usage_mb > 0 {
            (metrics.platform_metrics.memory_metrics.total_usage_mb as f32
                / (metrics.platform_metrics.memory_metrics.total_usage_mb
                    + metrics.platform_metrics.memory_metrics.available_mb)
                    as f32)
                * 100.0
        } else {
            0.0
        };

        if memory_usage_percent > self.thresholds.memory_threshold_percent {
            alerts.push(PerformanceAlert {
                alert_type: AlertType::HighMemoryUsage,
                severity: AlertSeverity::Warning,
                message: format!(
                    "Memory utilization is {:.1}%, exceeding threshold of {:.1}%",
                    memory_usage_percent, self.thresholds.memory_threshold_percent
                ),
                trigger_value: memory_usage_percent,
                threshold_value: self.thresholds.memory_threshold_percent,
                timestamp: Instant::now(),
                duration_ms: 0,
                suggested_actions: vec![
                    "Enable memory optimization".to_string(),
                    "Reduce model size".to_string(),
                    "Clear model cache".to_string(),
                ],
            });
        }

        // Latency threshold check
        if metrics.inference_metrics.latency_ms > self.thresholds.latency_threshold_ms {
            alerts.push(PerformanceAlert {
                alert_type: AlertType::HighLatency,
                severity: AlertSeverity::Error,
                message: format!(
                    "Inference latency is {:.1}ms, exceeding threshold of {:.1}ms",
                    metrics.inference_metrics.latency_ms, self.thresholds.latency_threshold_ms
                ),
                trigger_value: metrics.inference_metrics.latency_ms,
                threshold_value: self.thresholds.latency_threshold_ms,
                timestamp: Instant::now(),
                duration_ms: 0,
                suggested_actions: vec![
                    "Enable hardware acceleration".to_string(),
                    "Optimize model architecture".to_string(),
                    "Reduce batch size".to_string(),
                ],
            });
        }

        // Store alerts in history
        for alert in &alerts {
            self.alert_history.push_back(alert.clone());
        }

        self.active_alerts = alerts.clone();
        Ok(alerts)
    }
}
/// Bottleneck detector
pub(super) struct BottleneckDetector {
    pub(super) config: BottleneckConfig,
    pub(super) analysis_buffer: VecDeque<MetricsSnapshot>,
    pub(super) detected_bottlenecks: Vec<PerformanceBottleneck>,
}
impl BottleneckDetector {
    pub(super) fn new(config: BottleneckConfig) -> Self {
        Self {
            config,
            analysis_buffer: VecDeque::new(),
            detected_bottlenecks: vec![],
        }
    }

    pub(super) fn analyze(
        &mut self,
        metrics: &MetricsSnapshot,
    ) -> Result<Vec<PerformanceBottleneck>> {
        self.analysis_buffer.push_back(metrics.clone());

        if self.analysis_buffer.len() > self.config.analysis_window_samples {
            self.analysis_buffer.pop_front();
        }

        // Analyze for bottlenecks
        let mut bottlenecks = vec![];

        // CPU bottleneck detection
        if self.config.detect_cpu_bottlenecks
            && metrics.platform_metrics.cpu_metrics.utilization_percent
                > self.config.cpu_threshold_percent
        {
            bottlenecks.push(PerformanceBottleneck {
                bottleneck_type: BottleneckType::CPU,
                description: format!(
                    "High CPU utilization detected: {:.1}%",
                    metrics.platform_metrics.cpu_metrics.utilization_percent
                ),
                severity: self.calculate_bottleneck_severity(
                    metrics.platform_metrics.cpu_metrics.utilization_percent,
                    self.config.cpu_threshold_percent,
                ),
                duration_ms: 0, // Would calculate based on analysis window
                performance_impact_percent: metrics
                    .platform_metrics
                    .cpu_metrics
                    .utilization_percent
                    - self.config.cpu_threshold_percent,
                optimizations: vec![OptimizationSuggestion {
                    category: OptimizationCategory::ComputeOptimization,
                    description: "Consider reducing inference frequency or using model compression"
                        .to_string(),
                    expected_improvement_percent: 20.0,
                    difficulty: OptimizationDifficulty::Medium,
                    priority: OptimizationPriority::High,
                }],
                confidence: 0.9,
            });
        }

        // Memory bottleneck detection
        if self.config.detect_memory_bottlenecks
            && matches!(
                metrics.platform_metrics.memory_metrics.pressure_level,
                MemoryPressureLevel::High | MemoryPressureLevel::Critical
            )
        {
            bottlenecks.push(PerformanceBottleneck {
                bottleneck_type: BottleneckType::Memory,
                description: format!(
                    "High memory pressure detected: {:?}",
                    metrics.platform_metrics.memory_metrics.pressure_level
                ),
                severity: match metrics.platform_metrics.memory_metrics.pressure_level {
                    MemoryPressureLevel::High => BottleneckSeverity::High,
                    MemoryPressureLevel::Critical => BottleneckSeverity::Critical,
                    _ => BottleneckSeverity::Medium,
                },
                duration_ms: 0,
                performance_impact_percent: 30.0,
                optimizations: vec![OptimizationSuggestion {
                    category: OptimizationCategory::MemoryOptimization,
                    description: "Enable aggressive memory optimization and model quantization"
                        .to_string(),
                    expected_improvement_percent: 40.0,
                    difficulty: OptimizationDifficulty::Easy,
                    priority: OptimizationPriority::Critical,
                }],
                confidence: 0.95,
            });
        }

        self.detected_bottlenecks = bottlenecks.clone();
        Ok(bottlenecks)
    }

    pub(super) fn calculate_bottleneck_severity(
        &self,
        current_value: f32,
        threshold: f32,
    ) -> BottleneckSeverity {
        let ratio = current_value / threshold;
        if ratio > 1.5 {
            BottleneckSeverity::Critical
        } else if ratio > 1.25 {
            BottleneckSeverity::High
        } else if ratio > 1.1 {
            BottleneckSeverity::Medium
        } else {
            BottleneckSeverity::Low
        }
    }
}
/// Performance analyzer for pattern detection and insights
pub(super) struct PerformanceAnalyzer {
    pub(super) performance_patterns: Vec<PerformancePattern>,
    pub(super) regression_detector: RegressionDetector,
    pub(super) trend_analyzer: TrendAnalyzer,
}
impl PerformanceAnalyzer {
    pub(super) fn new() -> Self {
        Self {
            performance_patterns: vec![],
            regression_detector: RegressionDetector::new(),
            trend_analyzer: TrendAnalyzer::new(),
        }
    }

    pub(super) fn generate_suggestions(
        &mut self,
        metrics: &MetricsSnapshot,
        bottlenecks: &[PerformanceBottleneck],
    ) -> Result<Vec<OptimizationSuggestion>> {
        let mut suggestions = vec![];

        // Generate suggestions based on bottlenecks
        for bottleneck in bottlenecks {
            suggestions.extend(bottleneck.optimizations.clone());
        }

        // Generate suggestions based on metrics
        if metrics.inference_metrics.latency_ms > 200.0 {
            suggestions.push(OptimizationSuggestion {
                category: OptimizationCategory::ComputeOptimization,
                description: "High inference latency detected. Consider model optimization or hardware acceleration".to_string(),
                expected_improvement_percent: 50.0,
                difficulty: OptimizationDifficulty::Medium,
                priority: OptimizationPriority::High,
            });
        }

        Ok(suggestions)
    }
}
/// Performance regression detector
pub(super) struct RegressionDetector {
    pub(super) baseline_metrics: Option<MetricsSnapshot>,
    pub(super) regression_threshold_percent: f32,
    pub(super) detected_regressions: Vec<PerformanceRegression>,
}
impl RegressionDetector {
    pub(super) fn new() -> Self {
        Self {
            baseline_metrics: None,
            regression_threshold_percent: 10.0,
            detected_regressions: vec![],
        }
    }
}
/// Performance trend analyzer
pub(super) struct TrendAnalyzer {
    pub(super) trend_window_size: usize,
    pub(super) performance_trends: HashMap<String, PerformanceTrend>,
}
impl TrendAnalyzer {
    pub(super) fn new() -> Self {
        Self {
            trend_window_size: 50,
            performance_trends: HashMap::new(),
        }
    }
}
