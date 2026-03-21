//! Performance profiler struct definitions

use super::types::*;
use serde::{Deserialize, Serialize};
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;

/// Profiler configuration
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    enabled: bool,
    detailed_timing: bool,
    resource_monitoring: bool,
    bottleneck_detection: bool,
    #[allow(dead_code)]
    memory_profiling: bool,
    #[allow(dead_code)]
    gpu_profiling: bool,
    #[allow(dead_code)]
    operation_breakdown: bool,
    #[allow(dead_code)]
    comparative_analysis: bool,
    max_samples: usize,
    sampling_interval_ms: u32,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl ProfilerConfig {
    /// Create a new profiler configuration
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            enabled: false,
            detailed_timing: true,
            resource_monitoring: true,
            bottleneck_detection: true,
            memory_profiling: true,
            gpu_profiling: true,
            operation_breakdown: true,
            comparative_analysis: false,
            max_samples: 10000,
            sampling_interval_ms: 10,
        }
    }

    /// Create a development configuration with full profiling
    pub fn development() -> Self {
        Self {
            enabled: true,
            detailed_timing: true,
            resource_monitoring: true,
            bottleneck_detection: true,
            memory_profiling: true,
            gpu_profiling: true,
            operation_breakdown: true,
            comparative_analysis: true,
            max_samples: 50000,
            sampling_interval_ms: 1,
        }
    }

    /// Create a production configuration with minimal overhead
    pub fn production() -> Self {
        Self {
            enabled: true,
            detailed_timing: false,
            resource_monitoring: false,
            bottleneck_detection: true,
            memory_profiling: false,
            gpu_profiling: false,
            operation_breakdown: false,
            comparative_analysis: false,
            max_samples: 1000,
            sampling_interval_ms: 100,
        }
    }

    /// Enable/disable profiling
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Enable/disable detailed timing
    pub fn set_detailed_timing(&mut self, enabled: bool) {
        self.detailed_timing = enabled;
    }

    /// Enable/disable resource monitoring
    pub fn set_resource_monitoring(&mut self, enabled: bool) {
        self.resource_monitoring = enabled;
    }

    /// Enable/disable bottleneck detection
    pub fn set_bottleneck_detection(&mut self, enabled: bool) {
        self.bottleneck_detection = enabled;
    }

    /// Set sampling interval in milliseconds
    pub fn set_sampling_interval(&mut self, interval_ms: u32) {
        self.sampling_interval_ms = interval_ms;
    }

    /// Set maximum number of samples to retain
    pub fn set_max_samples(&mut self, max: usize) {
        self.max_samples = max;
    }

    // Getters
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn detailed_timing(&self) -> bool {
        self.detailed_timing
    }

    pub fn resource_monitoring(&self) -> bool {
        self.resource_monitoring
    }

    pub fn bottleneck_detection(&self) -> bool {
        self.bottleneck_detection
    }

    pub fn max_samples(&self) -> usize {
        self.max_samples
    }

    pub fn sampling_interval_ms(&self) -> u32 {
        self.sampling_interval_ms
    }
}

/// Detailed operation profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationProfile {
    pub operation_type: OperationType,
    pub operation_name: String,
    pub start_time: f64,
    pub end_time: f64,
    pub duration_ms: f64,
    pub cpu_time_ms: f64,
    pub gpu_time_ms: f64,
    pub memory_allocated: usize,
    pub memory_peak: usize,
    pub gpu_memory_used: usize,
    pub flops: u64,
    pub memory_bandwidth_gb_s: f32,
    pub cache_hits: u32,
    pub cache_misses: u32,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
}

/// Resource usage sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSample {
    pub timestamp: f64,
    pub resource_type: ResourceType,
    pub value: f32,
    pub cpu_usage: f32,
    pub gpu_usage: f32,
    pub wasm_memory: usize,
    pub gpu_memory: usize,
    pub network_bytes: usize,
    pub cache_hit_rate: f32,
    pub battery_level: f32,
    pub power_consumption: f32,
    pub thermal_state: f32, // 0.0 = cool, 1.0 = hot
    pub cpu_temperature: f32,
    pub gpu_temperature: f32,
}

/// Performance bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    pub bottleneck_type: BottleneckType,
    pub operation: String,
    pub severity: f32, // 0.0 to 1.0
    pub time_percentage: f32,
    pub description: String,
    pub recommendation: String,
}

/// Performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub total_time_ms: f64,
    pub operation_count: usize,
    pub average_fps: f32,
    pub bottlenecks: Vec<Bottleneck>,
    pub top_operations: Vec<(String, f64)>, // (operation, time_ms)
    pub resource_efficiency: f32,           // 0.0 to 1.0
    pub recommendations: Vec<String>,
}

/// Real-time analytics
#[derive(Debug, Clone)]
pub struct RealTimeAnalytics {
    pub enabled: bool,
    pub window_size: usize,
    pub trend_analysis: bool,
    pub anomaly_detection: bool,
    pub predictive_modeling: bool,
    pub adaptive_optimization: bool,
    pub regression_detection: bool,
}

/// Adaptive optimizer
#[derive(Debug, Clone)]
pub struct AdaptiveOptimizer {
    pub enabled: bool,
    pub learning_rate: f32,
    pub optimization_targets: Vec<OptimizationTarget>,
    pub current_strategy: OptimizationStrategy,
    pub adaptation_history: Vec<AdaptationRecord>,
    pub performance_baselines: Vec<PerformanceBaseline>,
}

/// Record of optimization adaptations
#[derive(Debug, Clone)]
pub struct AdaptationRecord {
    pub timestamp: f64,
    pub old_strategy: OptimizationStrategy,
    pub new_strategy: OptimizationStrategy,
    pub trigger_metric: String,
    pub improvement_ratio: f32,
    pub confidence_score: f32,
}

/// Performance baseline for comparison
#[derive(Debug, Clone)]
pub struct PerformanceBaseline {
    pub name: String,
    pub timestamp: f64,
    pub avg_latency_ms: f64,
    pub avg_throughput: f32,
    pub avg_memory_mb: f32,
    pub avg_accuracy: f32,
}

/// Real-time performance trend
#[derive(Debug, Clone)]
pub struct PerformanceTrend {
    pub metric_name: String,
    pub values: Vec<f64>,
    pub timestamps: Vec<f64>,
    pub trend_direction: TrendDirection,
    pub trend_strength: f32,
    pub predicted_next_value: f64,
}

/// Performance anomaly detection
#[derive(Debug, Clone)]
pub struct PerformanceAnomaly {
    pub timestamp: f64,
    pub metric_name: String,
    pub expected_value: f64,
    pub actual_value: f64,
    pub severity: AnomalySeverity,
    pub description: String,
    pub suggested_action: String,
}
