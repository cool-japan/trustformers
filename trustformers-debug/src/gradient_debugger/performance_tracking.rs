//! Performance Tracking and Bottleneck Analysis for Gradient Computation
//!
//! This module provides comprehensive performance tracking capabilities for gradient
//! computation, including bottleneck identification, throughput analysis, and
//! resource utilization monitoring.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Performance tracking for gradient computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientPerformanceTracker {
    pub total_gradient_computations: usize,
    pub average_computation_time: Duration,
    pub memory_usage_bytes: usize,
    pub throughput_gradients_per_second: f64,
    pub bottleneck_layers: Vec<String>,
    pub layer_performance_map: HashMap<String, LayerPerformanceMetrics>,
    pub resource_utilization: ResourceUtilization,
    pub performance_history: Vec<PerformanceSnapshot>,
}

impl Default for GradientPerformanceTracker {
    fn default() -> Self {
        Self {
            total_gradient_computations: 0,
            average_computation_time: Duration::from_millis(0),
            memory_usage_bytes: 0,
            throughput_gradients_per_second: 0.0,
            bottleneck_layers: Vec::new(),
            layer_performance_map: HashMap::new(),
            resource_utilization: ResourceUtilization::default(),
            performance_history: Vec::new(),
        }
    }
}

impl GradientPerformanceTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn start_timing(&mut self, layer_name: &str) -> PerformanceTimer {
        PerformanceTimer::new(layer_name.to_string())
    }

    pub fn record_layer_performance(
        &mut self,
        layer_name: &str,
        computation_time: Duration,
        memory_used: usize,
    ) {
        let metrics = self
            .layer_performance_map
            .entry(layer_name.to_string())
            .or_insert_with(|| LayerPerformanceMetrics::new(layer_name.to_string()));

        metrics.update(computation_time, memory_used);
        self.total_gradient_computations += 1;

        // Update overall averages
        self.update_overall_metrics();
        self.identify_bottlenecks();
    }

    fn update_overall_metrics(&mut self) {
        if self.layer_performance_map.is_empty() {
            return;
        }

        let total_time: Duration =
            self.layer_performance_map.values().map(|m| m.average_computation_time).sum();

        let total_layers = self.layer_performance_map.len();
        self.average_computation_time = total_time / total_layers as u32;

        self.memory_usage_bytes =
            self.layer_performance_map.values().map(|m| m.average_memory_usage).sum();

        // Calculate throughput
        if self.average_computation_time.as_secs_f64() > 0.0 {
            self.throughput_gradients_per_second =
                1.0 / self.average_computation_time.as_secs_f64();
        }
    }

    fn identify_bottlenecks(&mut self) {
        self.bottleneck_layers.clear();

        if self.layer_performance_map.len() < 2 {
            return;
        }

        // Calculate mean and standard deviation of computation times
        let times: Vec<f64> = self
            .layer_performance_map
            .values()
            .map(|m| m.average_computation_time.as_secs_f64())
            .collect();

        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let variance = times.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / times.len() as f64;
        let std_dev = variance.sqrt();

        // Identify layers that are significantly slower than average
        let threshold = mean + 1.5 * std_dev;

        for (layer_name, metrics) in &self.layer_performance_map {
            if metrics.average_computation_time.as_secs_f64() > threshold {
                self.bottleneck_layers.push(layer_name.clone());
            }
        }
    }

    pub fn get_performance_trends(&self) -> PerformanceTrends {
        if self.performance_history.len() < 2 {
            return PerformanceTrends::default();
        }

        let recent_snapshots: Vec<&PerformanceSnapshot> =
            self.performance_history.iter().rev().take(10).collect();

        let older_snapshots: Vec<&PerformanceSnapshot> =
            self.performance_history.iter().rev().skip(10).take(10).collect();

        if older_snapshots.is_empty() {
            return PerformanceTrends::default();
        }

        let recent_avg_throughput = recent_snapshots.iter().map(|s| s.throughput).sum::<f64>()
            / recent_snapshots.len() as f64;

        let older_avg_throughput = older_snapshots.iter().map(|s| s.throughput).sum::<f64>()
            / older_snapshots.len() as f64;

        let recent_avg_memory =
            recent_snapshots.iter().map(|s| s.memory_usage).sum::<usize>() / recent_snapshots.len();

        let older_avg_memory =
            older_snapshots.iter().map(|s| s.memory_usage).sum::<usize>() / older_snapshots.len();

        PerformanceTrends {
            throughput_trend: Self::classify_trend(recent_avg_throughput, older_avg_throughput),
            memory_trend: Self::classify_trend(recent_avg_memory as f64, older_avg_memory as f64),
            bottleneck_stability: self
                .analyze_bottleneck_stability(&recent_snapshots, &older_snapshots),
            overall_performance_direction: self
                .analyze_overall_direction(&recent_snapshots, &older_snapshots),
        }
    }

    fn classify_trend(recent: f64, older: f64) -> TrendDirection {
        let change_ratio = (recent - older) / older.max(1e-10);
        let threshold = 0.05; // 5% change threshold

        if change_ratio > threshold {
            TrendDirection::Improving
        } else if change_ratio < -threshold {
            TrendDirection::Degrading
        } else {
            TrendDirection::Stable
        }
    }

    fn analyze_bottleneck_stability(
        &self,
        recent: &[&PerformanceSnapshot],
        older: &[&PerformanceSnapshot],
    ) -> BottleneckStability {
        let recent_bottlenecks: std::collections::HashSet<&String> =
            recent.iter().flat_map(|s| &s.active_bottlenecks).collect();

        let older_bottlenecks: std::collections::HashSet<&String> =
            older.iter().flat_map(|s| &s.active_bottlenecks).collect();

        let intersection_size = recent_bottlenecks.intersection(&older_bottlenecks).count();
        let union_size = recent_bottlenecks.union(&older_bottlenecks).count();

        if union_size == 0 {
            return BottleneckStability::Stable;
        }

        let stability_ratio = intersection_size as f64 / union_size as f64;

        if stability_ratio > 0.8 {
            BottleneckStability::Stable
        } else if stability_ratio > 0.5 {
            BottleneckStability::Moderate
        } else {
            BottleneckStability::Unstable
        }
    }

    fn analyze_overall_direction(
        &self,
        recent: &[&PerformanceSnapshot],
        older: &[&PerformanceSnapshot],
    ) -> PerformanceDirection {
        let recent_avg_time =
            recent.iter().map(|s| s.average_time.as_secs_f64()).sum::<f64>() / recent.len() as f64;

        let older_avg_time =
            older.iter().map(|s| s.average_time.as_secs_f64()).sum::<f64>() / older.len() as f64;

        if recent_avg_time < older_avg_time * 0.95 {
            PerformanceDirection::Improving
        } else if recent_avg_time > older_avg_time * 1.05 {
            PerformanceDirection::Degrading
        } else {
            PerformanceDirection::Stable
        }
    }

    pub fn generate_optimization_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Analyze bottlenecks
        for layer_name in &self.bottleneck_layers {
            if let Some(metrics) = self.layer_performance_map.get(layer_name) {
                recommendations.push(OptimizationRecommendation {
                    layer_name: layer_name.clone(),
                    issue_type: OptimizationIssue::ComputationalBottleneck,
                    severity: self.calculate_bottleneck_severity(metrics),
                    recommendations: vec![
                        format!("Consider optimizing {} layer computation", layer_name),
                        "Check for inefficient operations or memory access patterns".to_string(),
                        "Consider layer-specific optimizations or hardware acceleration"
                            .to_string(),
                    ],
                    expected_improvement: self.estimate_improvement_potential(metrics),
                });
            }
        }

        // Memory usage analysis
        if self.memory_usage_bytes > 1_000_000_000 {
            // > 1GB
            recommendations.push(OptimizationRecommendation {
                layer_name: "Global".to_string(),
                issue_type: OptimizationIssue::HighMemoryUsage,
                severity: OptimizationSeverity::High,
                recommendations: vec![
                    "Consider gradient checkpointing to reduce memory usage".to_string(),
                    "Optimize batch size and sequence length".to_string(),
                    "Use memory-efficient attention mechanisms".to_string(),
                ],
                expected_improvement: 0.3,
            });
        }

        // Low throughput analysis
        if self.throughput_gradients_per_second < 1.0 {
            recommendations.push(OptimizationRecommendation {
                layer_name: "Global".to_string(),
                issue_type: OptimizationIssue::LowThroughput,
                severity: OptimizationSeverity::Medium,
                recommendations: vec![
                    "Consider mixed precision training".to_string(),
                    "Optimize data loading and preprocessing pipelines".to_string(),
                    "Use gradient accumulation for larger effective batch sizes".to_string(),
                ],
                expected_improvement: 0.4,
            });
        }

        recommendations
    }

    fn calculate_bottleneck_severity(
        &self,
        metrics: &LayerPerformanceMetrics,
    ) -> OptimizationSeverity {
        let relative_slowness = metrics.average_computation_time.as_secs_f64()
            / self.average_computation_time.as_secs_f64();

        if relative_slowness > 3.0 {
            OptimizationSeverity::Critical
        } else if relative_slowness > 2.0 {
            OptimizationSeverity::High
        } else if relative_slowness > 1.5 {
            OptimizationSeverity::Medium
        } else {
            OptimizationSeverity::Low
        }
    }

    fn estimate_improvement_potential(&self, metrics: &LayerPerformanceMetrics) -> f64 {
        let relative_slowness = metrics.average_computation_time.as_secs_f64()
            / self.average_computation_time.as_secs_f64();

        // Estimate potential improvement based on how much slower this layer is
        (relative_slowness - 1.0).min(0.8).max(0.1)
    }

    /// Start monitoring performance
    pub fn start_monitoring(&mut self) {
        // Reset performance tracking state
        self.total_gradient_computations = 0;
        self.average_computation_time = Duration::from_millis(0);
        self.memory_usage_bytes = 0;
        self.throughput_gradients_per_second = 0.0;
        self.bottleneck_layers.clear();
        self.layer_performance_map.clear();

        // Initialize resource utilization monitoring
        self.resource_utilization = ResourceUtilization {
            cpu_usage_percent: 0.0,
            memory_usage_percent: 0.0,
            gpu_usage_percent: 0.0,
            io_wait_percent: 0.0,
        };
    }

    /// Take a performance snapshot
    pub fn take_performance_snapshot(&self) -> PerformanceSnapshot {
        PerformanceSnapshot {
            timestamp: std::time::SystemTime::now(),
            total_computations: self.total_gradient_computations,
            average_time: self.average_computation_time,
            memory_usage: self.memory_usage_bytes,
            throughput: self.throughput_gradients_per_second,
            active_bottlenecks: self.bottleneck_layers.clone(),
            layer_count: self.layer_performance_map.len(),
        }
    }
}

/// Layer-specific performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerPerformanceMetrics {
    pub layer_name: String,
    pub computation_count: usize,
    pub total_computation_time: Duration,
    pub average_computation_time: Duration,
    pub total_memory_usage: usize,
    pub average_memory_usage: usize,
    pub min_computation_time: Duration,
    pub max_computation_time: Duration,
    pub performance_variance: f64,
}

impl LayerPerformanceMetrics {
    pub fn new(layer_name: String) -> Self {
        Self {
            layer_name,
            computation_count: 0,
            total_computation_time: Duration::from_millis(0),
            average_computation_time: Duration::from_millis(0),
            total_memory_usage: 0,
            average_memory_usage: 0,
            min_computation_time: Duration::from_secs(u64::MAX),
            max_computation_time: Duration::from_millis(0),
            performance_variance: 0.0,
        }
    }

    pub fn update(&mut self, computation_time: Duration, memory_used: usize) {
        self.computation_count += 1;
        self.total_computation_time += computation_time;
        self.total_memory_usage += memory_used;

        self.average_computation_time = self.total_computation_time / self.computation_count as u32;
        self.average_memory_usage = self.total_memory_usage / self.computation_count;

        if computation_time < self.min_computation_time {
            self.min_computation_time = computation_time;
        }
        if computation_time > self.max_computation_time {
            self.max_computation_time = computation_time;
        }

        self.update_variance(computation_time);
    }

    fn update_variance(&mut self, new_time: Duration) {
        if self.computation_count < 2 {
            self.performance_variance = 0.0;
            return;
        }

        let mean = self.average_computation_time.as_secs_f64();
        let new_value = new_time.as_secs_f64();

        // Incremental variance calculation
        let old_variance = self.performance_variance;
        let delta = new_value - mean;
        self.performance_variance = ((self.computation_count - 1) as f64 * old_variance
            + delta * delta)
            / self.computation_count as f64;
    }
}

/// Performance timer for measuring gradient computation time
#[derive(Debug)]
pub struct PerformanceTimer {
    layer_name: String,
    start_time: Instant,
}

impl PerformanceTimer {
    pub fn new(layer_name: String) -> Self {
        Self {
            layer_name,
            start_time: Instant::now(),
        }
    }

    pub fn finish(self) -> (String, Duration) {
        (self.layer_name, self.start_time.elapsed())
    }
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    pub cpu_usage_percent: f64,
    pub gpu_usage_percent: f64,
    pub memory_usage_percent: f64,
    pub io_wait_percent: f64,
}

impl Default for ResourceUtilization {
    fn default() -> Self {
        Self {
            cpu_usage_percent: 0.0,
            gpu_usage_percent: 0.0,
            memory_usage_percent: 0.0,
            io_wait_percent: 0.0,
        }
    }
}

/// Performance snapshot at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    pub timestamp: std::time::SystemTime,
    pub total_computations: usize,
    pub average_time: Duration,
    pub memory_usage: usize,
    pub throughput: f64,
    pub active_bottlenecks: Vec<String>,
    pub layer_count: usize,
}

/// Performance trends analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    pub throughput_trend: TrendDirection,
    pub memory_trend: TrendDirection,
    pub bottleneck_stability: BottleneckStability,
    pub overall_performance_direction: PerformanceDirection,
}

impl Default for PerformanceTrends {
    fn default() -> Self {
        Self {
            throughput_trend: TrendDirection::Stable,
            memory_trend: TrendDirection::Stable,
            bottleneck_stability: BottleneckStability::Stable,
            overall_performance_direction: PerformanceDirection::Stable,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckStability {
    Stable,
    Moderate,
    Unstable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceDirection {
    Improving,
    Stable,
    Degrading,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub layer_name: String,
    pub issue_type: OptimizationIssue,
    pub severity: OptimizationSeverity,
    pub recommendations: Vec<String>,
    pub expected_improvement: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationIssue {
    ComputationalBottleneck,
    HighMemoryUsage,
    LowThroughput,
    ResourceUnderutilization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationSeverity {
    Low,
    Medium,
    High,
    Critical,
}
