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

/// Detailed operation profile.
///
/// Only carries what this profiler can genuinely measure from
/// `start_operation`/`end_operation`'s inputs: wall-clock timing (real)
/// and WASM linear-memory growth (real, via
/// [`crate::get_wasm_memory_usage`]). A previous version also reported
/// `cpu_time_ms`/`gpu_time_ms` (an invented fixed 80/20 split of
/// `duration_ms`), `gpu_memory_used`/`flops`/`memory_bandwidth_gb_s`/
/// `cache_hits`/`cache_misses`/`input_shape`/`output_shape` (constant
/// lookup tables keyed only by `operation_type`, unrelated to what the
/// operation actually did) — none of that is derivable from what
/// `start_operation`/`end_operation` receive today (no per-op byte
/// counts, cache instrumentation, or GPU timer), so those fields have
/// been removed rather than kept as permanently-`None` placeholders.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationProfile {
    pub operation_type: OperationType,
    pub operation_name: String,
    pub start_time: f64,
    pub end_time: f64,
    pub duration_ms: f64,
    pub memory_peak: usize,
    /// WASM linear-memory growth (bytes) observed between
    /// `start_operation` and `end_operation`, via
    /// [`crate::get_wasm_memory_usage`]. Real, but a narrower quantity
    /// than "bytes this operation allocated": the allocator only grows
    /// the linear memory in whole pages and never shrinks it, so this is
    /// `0` for any operation served entirely from already-grown heap, and
    /// counts a page grown for unrelated concurrent allocation the same
    /// as one caused by this operation.
    pub wasm_memory_growth_bytes: usize,
}

/// Resource usage sample.
///
/// Only carries what this profiler can genuinely measure: WASM linear
/// memory (real) and, when the browser's Battery Status API is present,
/// the real battery level (see
/// [`crate::performance_profiler::PerformanceProfiler::refresh_battery_status`]).
/// A previous version also reported `cpu_usage`/`gpu_usage`/
/// `cache_hit_rate`/`gpu_memory`/`power_consumption`/`thermal_state`/
/// `cpu_temperature`/`gpu_temperature`/`network_bytes` — no standard
/// browser API exposes process/system CPU or GPU utilization, GPU memory
/// usage, device temperature, or power draw to a web page (WebGPU
/// deliberately omits memory/utilization queries to resist
/// fingerprinting), and this profiler tracks no real cache or network
/// byte counters — so those were either a literal formula over
/// `Date::now()`, a hardcoded constant, or built from other fields in
/// this same list. Removed rather than kept as permanently-`None`
/// placeholders, since none of them is ever measurable under this
/// profiler's current inputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSample {
    pub timestamp: f64,
    pub resource_type: ResourceType,
    /// Headline reading for `resource_type`: memory in MB (not raw
    /// bytes - `f32` only holds integers exactly up to ~16.7 million, so
    /// a byte count would silently round for any heap above ~16MB) for
    /// `WAMSMemory` samples, battery level (0.0-1.0) for `Battery`
    /// samples. See `wasm_memory` for the exact byte count.
    pub value: f32,
    pub wasm_memory: usize,
    /// Real battery level (0.0-1.0) from the browser's Battery Status
    /// API, cached by the most recent
    /// `PerformanceProfiler::refresh_battery_status()` call. `None` when
    /// that has never been called, is still pending, or the API is
    /// unavailable (most browsers besides Firefox for Android have
    /// removed it) - never a fabricated reading.
    pub battery_level: Option<f32>,
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

/// Performance summary.
///
/// `resource_efficiency` (0.0-1.0, "how well CPU/GPU are utilized") has
/// been removed: it was computed purely from `ResourceSample`'s
/// (fabricated, now-deleted) `cpu_usage`/`gpu_usage` fields, so there was
/// no real quantity left to compute it from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub total_time_ms: f64,
    pub operation_count: usize,
    pub average_fps: f32,
    pub bottlenecks: Vec<Bottleneck>,
    pub top_operations: Vec<(String, f64)>, // (operation, time_ms)
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

/// Record of optimization adaptations.
///
/// `improvement_ratio`/`confidence_score` are always `None`: this
/// profiler switches strategy based on a real trigger metric crossing a
/// real threshold (see `check_and_trigger_adaptation`), but never runs
/// the same workload under two strategies to compare, and has no trained
/// model — so there is no measurement to report for "how much better is
/// the new strategy". A previous version multiplied a hardcoded
/// per-transition-pair table (e.g. 2.5x for CPU-preferred to
/// GPU-preferred) by factors derived from fabricated CPU/GPU usage
/// telemetry and reported the result as "ML-powered estimation" with a
/// flat 0.8 "confidence".
#[derive(Debug, Clone)]
pub struct AdaptationRecord {
    pub timestamp: f64,
    pub old_strategy: OptimizationStrategy,
    pub new_strategy: OptimizationStrategy,
    pub trigger_metric: String,
    pub improvement_ratio: Option<f32>,
    pub confidence_score: Option<f32>,
}

/// Performance baseline for comparison.
///
/// `avg_accuracy` is always `None`: this profiler only ever observes
/// timing and memory, never a model's actual output vs. ground truth, so
/// it has no accuracy signal to average - a previous version wrote a
/// flat `0.95` regardless of the operations backing the baseline.
#[derive(Debug, Clone)]
pub struct PerformanceBaseline {
    pub name: String,
    pub timestamp: f64,
    pub avg_latency_ms: f64,
    pub avg_throughput: f32,
    pub avg_memory_mb: f32,
    pub avg_accuracy: Option<f32>,
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
