//! Advanced performance profiler for ML inference optimization
//!
//! This module provides comprehensive performance profiling capabilities including:
//! - Detailed ML operation timing (wall-clock and real WASM memory growth)
//! - Bottleneck detection and analysis
//! - Resource usage monitoring (real WASM memory; real battery level when
//!   the browser's Battery Status API is present - no browser API exposes
//!   CPU/GPU utilization, GPU memory, or device temperature to a web page,
//!   so this module does not report those)
//! - Performance visualization data
//! - Optimization recommendations
//! - Comparative performance analysis

use crate::debug::DebugLogger;
use crate::performance::*;
use js_sys::{Array, Date, Object};
use std::string::{String, ToString};
use std::vec::Vec;
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

/// Advanced performance profiler with real-time analytics
#[wasm_bindgen]
pub struct PerformanceProfiler {
    config: ProfilerConfig,
    real_time_analytics: RealTimeAnalytics,
    adaptive_optimizer: AdaptiveOptimizer,
    performance_trends: Vec<PerformanceTrend>,
    detected_anomalies: Vec<PerformanceAnomaly>,
    current_baselines: Vec<PerformanceBaseline>,
    operation_profiles: Vec<OperationProfile>,
    resource_samples: Vec<ResourceSample>,
    // (name, type, start_time, wasm_memory_at_start) - the memory reading
    // lets `end_operation` report real linear-memory growth.
    active_operations: Vec<(String, OperationType, f64, usize)>,
    baseline_metrics: Option<PerformanceSummary>,
    debug_logger: Option<DebugLogger>,
    /// Cached real battery level, refreshed by
    /// [`Self::refresh_battery_status`]. `None` until that has been
    /// called (successfully) at least once.
    cached_battery_level: Option<f32>,
}

#[wasm_bindgen]
impl PerformanceProfiler {
    /// Create a new performance profiler with real-time analytics
    #[wasm_bindgen(constructor)]
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            config,
            real_time_analytics: RealTimeAnalytics {
                enabled: true,
                window_size: 100,
                trend_analysis: true,
                anomaly_detection: true,
                predictive_modeling: true,
                adaptive_optimization: true,
                regression_detection: true,
            },
            adaptive_optimizer: AdaptiveOptimizer {
                enabled: true,
                learning_rate: 0.1,
                optimization_targets: vec![OptimizationTarget::Balanced],
                current_strategy: OptimizationStrategy::Hybrid,
                adaptation_history: Vec::new(),
                performance_baselines: Vec::new(),
            },
            performance_trends: Vec::new(),
            detected_anomalies: Vec::new(),
            current_baselines: Vec::new(),
            operation_profiles: Vec::new(),
            resource_samples: Vec::new(),
            active_operations: Vec::new(),
            baseline_metrics: None,
            debug_logger: None,
            cached_battery_level: None,
        }
    }

    /// Set debug logger for integration
    pub fn set_debug_logger(&mut self, logger: DebugLogger) {
        self.debug_logger = Some(logger);
    }

    /// Start profiling an operation
    pub fn start_operation(&mut self, name: &str, operation_type: OperationType) {
        if !self.config.enabled() {
            return;
        }

        let start_time = Date::now();
        let wasm_memory_at_start = crate::get_wasm_memory_usage();
        self.active_operations.push((
            name.to_string(),
            operation_type,
            start_time,
            wasm_memory_at_start,
        ));

        if self.config.detailed_timing() {
            web_sys::console::time_with_label(&format!("🔍 {name}"));
        }

        // Log to debug logger if available
        if let Some(ref mut logger) = self.debug_logger {
            logger.start_timer(name);
        }
    }

    /// End profiling an operation
    pub fn end_operation(&mut self, name: &str) -> Option<f64> {
        if !self.config.enabled() {
            return None;
        }

        let end_time = Date::now();

        // Find and remove the operation
        if let Some(pos) =
            self.active_operations.iter().position(|(op_name, _, _, _)| op_name == name)
        {
            let (_, operation_type, start_time, wasm_memory_at_start) =
                self.active_operations.remove(pos);
            let duration_ms = end_time - start_time;
            let memory_peak = crate::get_wasm_memory_usage();

            // Create detailed profile. Only fields genuinely derivable
            // from what this method receives (name/type/timestamps) plus
            // real WASM memory readings - see `OperationProfile`'s doc
            // comment for what was removed and why.
            let profile = OperationProfile {
                operation_type,
                operation_name: name.to_string(),
                start_time,
                end_time,
                duration_ms,
                memory_peak,
                wasm_memory_growth_bytes: Self::wasm_memory_growth(
                    wasm_memory_at_start,
                    memory_peak,
                ),
            };

            self.operation_profiles.push(profile);

            // Maintain maximum samples limit
            if self.operation_profiles.len() > self.config.max_samples() {
                self.operation_profiles.remove(0);
            }

            if self.config.detailed_timing() {
                web_sys::console::time_end_with_label(&format!("🔍 {name}"));
                web_sys::console::log_1(
                    &format!("📊 {name} completed in {duration_ms:.2}ms").into(),
                );
            }

            // Log to debug logger if available
            if let Some(ref mut logger) = self.debug_logger {
                logger.end_timer(name);
            }

            Some(duration_ms)
        } else {
            None
        }
    }

    /// Real WASM linear-memory growth (bytes) between two
    /// [`crate::get_wasm_memory_usage`] readings. Saturating: readings
    /// are monotonic in practice (the allocator only grows pages), but
    /// this must never underflow-panic even if a pair were ever taken
    /// out of order.
    fn wasm_memory_growth(before: usize, after: usize) -> usize {
        after.saturating_sub(before)
    }

    /// Refresh the cached real battery level via the browser's Battery
    /// Status API (`navigator.getBattery()`), read through
    /// `js_sys::Reflect`/`js_sys::Function` the same way as
    /// `device_capability::DeviceCapabilityDetector::get_battery_info` -
    /// so this doesn't need a typed `web_sys::BatteryManager` binding
    /// either. Updates [`Self::cached_battery_level`]'s backing field on
    /// success; leaves it untouched (never fabricates a reading) when the
    /// window/navigator/API is unavailable, which is always the case off
    /// the wasm32 target. A previous version of the removed
    /// `get_battery_level` checked for the API's presence and then
    /// returned a hardcoded `0.8` regardless of what it found.
    pub async fn refresh_battery_status(&mut self) {
        #[cfg(target_arch = "wasm32")]
        {
            let Some(window) = web_sys::window() else {
                return;
            };
            let navigator = window.navigator();
            let Ok(get_battery) = js_sys::Reflect::get(&navigator, &"getBattery".into()) else {
                return;
            };
            let Some(get_battery_fn) = get_battery.dyn_ref::<js_sys::Function>() else {
                return;
            };
            let Ok(promise) = get_battery_fn.call0(&navigator) else {
                return;
            };
            let Ok(promise) = promise.dyn_into::<js_sys::Promise>() else {
                return;
            };
            let Ok(battery_manager) = wasm_bindgen_futures::JsFuture::from(promise).await else {
                return;
            };
            if let Some(level) = js_sys::Reflect::get(&battery_manager, &"level".into())
                .ok()
                .and_then(|v| v.as_f64())
            {
                self.cached_battery_level = Some(level as f32);
            }
        }
    }

    /// Sample current resource usage.
    ///
    /// Only records what is genuinely measured: real WASM linear memory,
    /// plus a real battery sample whenever
    /// [`Self::refresh_battery_status`] has previously cached one. A
    /// previous version also sampled CPU/GPU usage, GPU memory, cache hit
    /// rate, power consumption, and thermal/CPU/GPU temperature here -
    /// all fabricated (see `ResourceSample`'s doc comment) - and used
    /// those fabricated readings to drive a fake thermal-throttling
    /// detector (`check_thermal_throttling`, removed).
    pub fn sample_resources(&mut self) {
        if !self.config.enabled() || !self.config.resource_monitoring() {
            return;
        }

        let timestamp = Date::now();
        let wasm_memory = crate::get_wasm_memory_usage();

        self.resource_samples.push(ResourceSample {
            timestamp,
            resource_type: ResourceType::WAMSMemory,
            // In MB, not raw bytes: `f32` only holds integers exactly up
            // to ~16.7 million, so a `wasm_memory as f32` byte count
            // silently rounds for any heap above ~16MB (a very ordinary
            // size for a loaded model). `wasm_memory` below is the exact
            // `usize` reading for any consumer that needs it.
            value: wasm_memory as f32 / 1_048_576.0,
            wasm_memory,
            battery_level: self.cached_battery_level,
        });

        if let Some(battery_level) = self.cached_battery_level {
            self.resource_samples.push(ResourceSample {
                timestamp,
                resource_type: ResourceType::Battery,
                value: battery_level,
                wasm_memory,
                battery_level: self.cached_battery_level,
            });
        }

        // Maintain maximum samples limit
        while self.resource_samples.len() > self.config.max_samples() {
            self.resource_samples.remove(0);
        }
    }

    /// Analyze performance and detect bottlenecks
    pub fn analyze_performance(&self) -> String {
        let summary = self.analyze_performance_internal();
        serde_json::to_string(&summary).unwrap_or_else(|_| "{}".to_string())
    }

    fn analyze_performance_internal(&self) -> PerformanceSummary {
        let total_time_ms: f64 = self.operation_profiles.iter().map(|p| p.duration_ms).sum();
        let operation_count = self.operation_profiles.len();
        let average_fps = if total_time_ms > 0.0 {
            1000.0 / (total_time_ms / operation_count as f64)
        } else {
            0.0
        };

        let bottlenecks = if self.config.bottleneck_detection() {
            self.detect_bottlenecks()
        } else {
            Vec::new()
        };

        let top_operations = self.get_top_operations(10);
        let recommendations = self.generate_recommendations(&bottlenecks);

        PerformanceSummary {
            total_time_ms,
            operation_count,
            average_fps: average_fps as f32,
            bottlenecks,
            top_operations,
            recommendations,
        }
    }

    /// Get performance summary as JSON string
    pub fn get_performance_summary(&self) -> String {
        let summary = self.analyze_performance_internal();
        serde_json::to_string_pretty(&summary).unwrap_or_else(|_| "{}".to_string())
    }

    /// Get operation breakdown for visualization
    pub fn get_operation_breakdown(&self) -> Array {
        let array = Array::new();

        for profile in &self.operation_profiles {
            let obj = Object::new();
            let _ =
                js_sys::Reflect::set(&obj, &"name".into(), &profile.operation_name.clone().into());
            let _ = js_sys::Reflect::set(
                &obj,
                &"type".into(),
                &format!("{op_type:?}", op_type = profile.operation_type).into(),
            );
            let _ = js_sys::Reflect::set(&obj, &"duration".into(), &profile.duration_ms.into());
            let _ = js_sys::Reflect::set(&obj, &"start_time".into(), &profile.start_time.into());
            let _ = js_sys::Reflect::set(
                &obj,
                &"wasm_memory_growth_bytes".into(),
                &profile.wasm_memory_growth_bytes.into(),
            );
            array.push(&obj);
        }

        array
    }

    /// Get resource usage timeline for visualization
    pub fn get_resource_timeline(&self) -> Array {
        let array = Array::new();

        for sample in &self.resource_samples {
            let obj = Object::new();
            let _ = js_sys::Reflect::set(&obj, &"timestamp".into(), &sample.timestamp.into());
            let _ = js_sys::Reflect::set(
                &obj,
                &"type".into(),
                &format!("{resource_type:?}", resource_type = sample.resource_type).into(),
            );
            let _ = js_sys::Reflect::set(&obj, &"memory".into(), &sample.wasm_memory.into());
            let battery_js: JsValue =
                sample.battery_level.map_or(JsValue::NULL, |level| level.into());
            let _ = js_sys::Reflect::set(&obj, &"battery_level".into(), &battery_js);
            array.push(&obj);
        }

        array
    }

    /// Set baseline metrics for comparison
    pub fn set_baseline(&mut self) {
        self.baseline_metrics = Some(self.analyze_performance_internal());
    }

    /// Compare current performance with baseline
    pub fn compare_with_baseline(&self) -> Option<String> {
        if let Some(ref baseline) = self.baseline_metrics {
            let current = self.analyze_performance_internal();

            let time_change =
                ((current.total_time_ms - baseline.total_time_ms) / baseline.total_time_ms) * 100.0;
            let fps_change = ((current.average_fps as f64 - baseline.average_fps as f64)
                / baseline.average_fps as f64)
                * 100.0;

            Some(format!(
                "Performance Comparison:\n\
                 Total Time: {:.1}% change\n\
                 Average FPS: {:.1}% change\n\
                 Bottlenecks: {} current vs {} baseline",
                time_change,
                fps_change,
                current.bottlenecks.len(),
                baseline.bottlenecks.len()
            ))
        } else {
            None
        }
    }

    /// Clear all profiling data
    pub fn clear(&mut self) {
        self.operation_profiles.clear();
        self.resource_samples.clear();
        self.active_operations.clear();
    }

    /// Export profiling data for external analysis
    pub fn export_data(&self) -> String {
        // Manual JSON construction to avoid json! macro dependency
        let summary = self.analyze_performance();

        format!(
            r#"{{
  "operation_profiles": {},
  "resource_samples": {},
  "performance_summary": {},
  "baseline": {}
}}"#,
            serde_json::to_string(&self.operation_profiles).unwrap_or("[]".to_string()),
            serde_json::to_string(&self.resource_samples).unwrap_or("[]".to_string()),
            serde_json::to_string(&summary).unwrap_or("{}".to_string()),
            serde_json::to_string(&self.baseline_metrics).unwrap_or("null".to_string())
        )
    }

    // Private helper methods

    fn detect_bottlenecks(&self) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();

        if self.operation_profiles.is_empty() {
            return bottlenecks;
        }

        let total_time: f64 = self.operation_profiles.iter().map(|p| p.duration_ms).sum();

        // Detect slow operations
        for profile in &self.operation_profiles {
            let time_percentage = (profile.duration_ms / total_time) * 100.0;

            if time_percentage > 20.0 {
                // Classified from `operation_type`, which the caller
                // genuinely supplies at `start_operation` time - a
                // previous version compared the fabricated
                // `gpu_time_ms`/`cpu_time_ms` split (a fixed 20/80 of
                // `duration_ms` for every operation), so `gpu_time_ms >
                // cpu_time_ms` was always false and this branch could
                // never actually classify anything as `GPUCompute`.
                let bottleneck_type = if profile.operation_type == OperationType::GPUKernel {
                    BottleneckType::GPUCompute
                } else {
                    BottleneckType::CPUCompute
                };

                bottlenecks.push(Bottleneck {
                    bottleneck_type,
                    operation: profile.operation_name.clone(),
                    severity: (time_percentage / 100.0).min(1.0) as f32,
                    time_percentage: time_percentage as f32,
                    description: format!(
                        "{} takes {:.1}% of total time",
                        profile.operation_name, time_percentage
                    ),
                    recommendation: self.get_optimization_recommendation(bottleneck_type),
                });
            }
        }

        // Detect memory bottlenecks
        let max_memory = self.operation_profiles.iter().map(|p| p.memory_peak).max().unwrap_or(0);
        if max_memory > 100 * 1024 * 1024 {
            // >100MB
            bottlenecks.push(Bottleneck {
                bottleneck_type: BottleneckType::MemoryCapacity,
                operation: "Overall".to_string(),
                severity: 0.7,
                time_percentage: 0.0,
                description: format!(
                    "High memory usage: {size}MB",
                    size = max_memory / (1024 * 1024)
                ),
                recommendation: "Consider model quantization or weight compression".to_string(),
            });
        }

        bottlenecks
    }

    fn get_top_operations(&self, limit: usize) -> Vec<(String, f64)> {
        let mut operations: Vec<_> = self
            .operation_profiles
            .iter()
            .map(|p| (p.operation_name.clone(), p.duration_ms))
            .collect();

        operations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        operations.truncate(limit);
        operations
    }

    fn generate_recommendations(&self, bottlenecks: &[Bottleneck]) -> Vec<String> {
        let mut recommendations = Vec::new();

        for bottleneck in bottlenecks {
            recommendations.push(bottleneck.recommendation.clone());
        }

        // General recommendations
        if self.operation_profiles.len() > 1000 {
            recommendations.push("Consider reducing profiling overhead in production".to_string());
        }

        let avg_duration: f64 = self.operation_profiles.iter().map(|p| p.duration_ms).sum::<f64>()
            / self.operation_profiles.len() as f64;
        if avg_duration > 100.0 {
            recommendations
                .push("Consider model optimization techniques like quantization".to_string());
        }

        recommendations
    }

    fn get_optimization_recommendation(&self, bottleneck_type: BottleneckType) -> String {
        match bottleneck_type {
            BottleneckType::CPUCompute => {
                "Consider using WebGPU acceleration or SIMD optimizations".to_string()
            },
            BottleneckType::GPUCompute => {
                "Consider kernel fusion or reducing data transfers".to_string()
            },
            BottleneckType::MemoryBandwidth => {
                "Consider data layout optimizations or caching".to_string()
            },
            BottleneckType::MemoryCapacity => {
                "Consider model compression or quantization".to_string()
            },
            BottleneckType::GPUMemory => "Consider reducing batch size or model size".to_string(),
            BottleneckType::DataTransfer => {
                "Consider batching operations or reducing transfers".to_string()
            },
            BottleneckType::Serialization => "Consider binary formats or streaming".to_string(),
            BottleneckType::JSInterop => "Consider reducing WASM/JS boundary crossings".to_string(),
        }
    }

    // Real-time analytics and adaptive optimization methods

    /// Update real-time performance metrics and trigger adaptation if needed
    pub fn update_real_time_metrics(
        &mut self,
        latency_ms: f64,
        throughput: f32,
        memory_mb: f32,
        accuracy: f32,
    ) {
        if !self.real_time_analytics.enabled {
            return;
        }

        let timestamp = Date::now();

        // Update performance trends
        self.update_performance_trend("latency", latency_ms, timestamp);
        self.update_performance_trend("throughput", throughput as f64, timestamp);
        self.update_performance_trend("memory", memory_mb as f64, timestamp);
        self.update_performance_trend("accuracy", accuracy as f64, timestamp);

        // Detect anomalies
        if self.real_time_analytics.anomaly_detection {
            self.detect_anomalies(timestamp, latency_ms, throughput, memory_mb, accuracy);
        }

        // Trigger adaptive optimization if enabled
        if self.real_time_analytics.adaptive_optimization && self.adaptive_optimizer.enabled {
            self.check_and_trigger_adaptation(
                timestamp, latency_ms, throughput, memory_mb, accuracy,
            );
        }

        web_sys::console::log_1(&format!(
            "📊 Real-time metrics: {:.1}ms latency, {:.1} throughput, {:.1}MB memory, {:.1}% accuracy",
            latency_ms, throughput, memory_mb, accuracy * 100.0
        ).into());
    }

    /// Set optimization target for adaptive optimization
    pub fn set_optimization_target(&mut self, target_name: &str) {
        let target = match target_name {
            "latency" => OptimizationTarget::Latency,
            "throughput" => OptimizationTarget::Throughput,
            "memory" => OptimizationTarget::MemoryUsage,
            "power" => OptimizationTarget::PowerEfficiency,
            "accuracy" => OptimizationTarget::Accuracy,
            "balanced" => OptimizationTarget::Balanced,
            _ => OptimizationTarget::Balanced,
        };

        self.adaptive_optimizer.optimization_targets = vec![target];
        web_sys::console::log_1(&format!("🎯 Optimization target set to: {target:?}").into());
    }

    /// Get current performance trends as JavaScript object
    pub fn get_performance_trends(&self) -> js_sys::Object {
        let trends_obj = js_sys::Object::new();

        for trend in &self.performance_trends {
            let trend_obj = js_sys::Object::new();

            // Convert values and timestamps to arrays
            let values_array = js_sys::Array::new();
            for &value in &trend.values {
                values_array.push(&value.into());
            }

            let timestamps_array = js_sys::Array::new();
            for &timestamp in &trend.timestamps {
                timestamps_array.push(&timestamp.into());
            }

            let _ = js_sys::Reflect::set(&trend_obj, &"values".into(), &values_array);
            let _ = js_sys::Reflect::set(&trend_obj, &"timestamps".into(), &timestamps_array);
            let _ = js_sys::Reflect::set(
                &trend_obj,
                &"direction".into(),
                &format!("{direction:?}", direction = trend.trend_direction).into(),
            );
            let _ =
                js_sys::Reflect::set(&trend_obj, &"strength".into(), &trend.trend_strength.into());
            let _ = js_sys::Reflect::set(
                &trend_obj,
                &"predicted_next".into(),
                &trend.predicted_next_value.into(),
            );

            let _ =
                js_sys::Reflect::set(&trends_obj, &trend.metric_name.clone().into(), &trend_obj);
        }

        trends_obj
    }

    /// Get detected anomalies as JavaScript array
    pub fn get_detected_anomalies(&self) -> js_sys::Array {
        let anomalies_array = js_sys::Array::new();

        for anomaly in &self.detected_anomalies {
            let anomaly_obj = js_sys::Object::new();
            let _ =
                js_sys::Reflect::set(&anomaly_obj, &"timestamp".into(), &anomaly.timestamp.into());
            let _ = js_sys::Reflect::set(
                &anomaly_obj,
                &"metric".into(),
                &anomaly.metric_name.clone().into(),
            );
            let _ = js_sys::Reflect::set(
                &anomaly_obj,
                &"expected".into(),
                &anomaly.expected_value.into(),
            );
            let _ =
                js_sys::Reflect::set(&anomaly_obj, &"actual".into(), &anomaly.actual_value.into());
            let _ = js_sys::Reflect::set(
                &anomaly_obj,
                &"severity".into(),
                &format!("{severity:?}", severity = anomaly.severity).into(),
            );
            let _ = js_sys::Reflect::set(
                &anomaly_obj,
                &"description".into(),
                &anomaly.description.clone().into(),
            );
            let _ = js_sys::Reflect::set(
                &anomaly_obj,
                &"suggested_action".into(),
                &anomaly.suggested_action.clone().into(),
            );

            anomalies_array.push(&anomaly_obj);
        }

        anomalies_array
    }

    /// Get adaptive optimization state
    pub fn get_adaptive_state(&self) -> js_sys::Object {
        let state_obj = js_sys::Object::new();

        let _ = js_sys::Reflect::set(
            &state_obj,
            &"enabled".into(),
            &self.adaptive_optimizer.enabled.into(),
        );
        let _ = js_sys::Reflect::set(
            &state_obj,
            &"learning_rate".into(),
            &self.adaptive_optimizer.learning_rate.into(),
        );
        let _ = js_sys::Reflect::set(
            &state_obj,
            &"current_strategy".into(),
            &format!(
                "{strategy:?}",
                strategy = self.adaptive_optimizer.current_strategy
            )
            .into(),
        );
        let _ = js_sys::Reflect::set(
            &state_obj,
            &"adaptation_count".into(),
            &self.adaptive_optimizer.adaptation_history.len().into(),
        );

        // Add optimization targets
        let targets_array = js_sys::Array::new();
        for target in &self.adaptive_optimizer.optimization_targets {
            targets_array.push(&format!("{target:?}").into());
        }
        let _ = js_sys::Reflect::set(&state_obj, &"optimization_targets".into(), &targets_array);

        state_obj
    }

    /// Enable/disable real-time analytics
    pub fn set_real_time_analytics(&mut self, enabled: bool) {
        self.real_time_analytics.enabled = enabled;
        web_sys::console::log_1(
            &format!(
                "📊 Real-time analytics: {}",
                if enabled { "enabled" } else { "disabled" }
            )
            .into(),
        );
    }

    /// Enable/disable adaptive optimization
    pub fn set_adaptive_optimization(&mut self, enabled: bool) {
        self.adaptive_optimizer.enabled = enabled;
        web_sys::console::log_1(
            &format!(
                "🤖 Adaptive optimization: {}",
                if enabled { "enabled" } else { "disabled" }
            )
            .into(),
        );
    }

    /// Create a performance baseline for comparison
    pub fn create_baseline(&mut self, name: &str) {
        if self.operation_profiles.is_empty() {
            web_sys::console::log_1(
                &"⚠️ No performance data available for baseline creation".into(),
            );
            return;
        }

        let avg_latency = self.operation_profiles.iter().map(|p| p.duration_ms).sum::<f64>()
            / self.operation_profiles.len() as f64;
        // Real per-operation peak WASM memory, averaged - `memory_peak` is
        // the one memory reading `OperationProfile` still carries (see its
        // doc comment); the removed `memory_allocated` was a constant
        // lookup keyed only by operation type.
        let avg_memory = self.operation_profiles.iter().map(|p| p.memory_peak as f64).sum::<f64>()
            / self.operation_profiles.len() as f64
            / 1_048_576.0; // Convert to MB

        let baseline = PerformanceBaseline {
            name: name.to_string(),
            timestamp: Date::now(),
            avg_latency_ms: avg_latency,
            avg_throughput: if avg_latency > 0.0 { 1000.0 / avg_latency as f32 } else { 0.0 },
            avg_memory_mb: avg_memory as f32,
            avg_accuracy: None, // not measurable - see `PerformanceBaseline`'s doc comment
        };

        self.current_baselines.push(baseline);
        web_sys::console::log_1(&format!("📏 Performance baseline '{name}' created").into());
    }

    // Private helper methods for real-time analytics

    fn update_performance_trend(&mut self, metric_name: &str, value: f64, timestamp: f64) {
        // Find existing trend or create new one
        let trend_index =
            match self.performance_trends.iter().position(|t| t.metric_name == metric_name) {
                Some(index) => index,
                None => {
                    self.performance_trends.push(PerformanceTrend {
                        metric_name: metric_name.to_string(),
                        values: Vec::new(),
                        timestamps: Vec::new(),
                        trend_direction: TrendDirection::Stable,
                        trend_strength: 0.0,
                        predicted_next_value: value,
                    });
                    self.performance_trends.len() - 1
                },
            };

        let trend = &mut self.performance_trends[trend_index];

        // Add new data point
        trend.values.push(value);
        trend.timestamps.push(timestamp);

        // Maintain window size
        if trend.values.len() > self.real_time_analytics.window_size {
            trend.values.remove(0);
            trend.timestamps.remove(0);
        }

        // Update trend analysis
        if self.real_time_analytics.trend_analysis && trend.values.len() >= 5 {
            // Extract values needed for analysis to avoid borrowing conflicts
            let values = trend.values.clone();
            let n = values.len();

            if n >= 5 {
                // Simple linear regression for trend detection
                let recent_values = &values[n - 5..];
                let sum_x: f64 = (0..5).map(|i| i as f64).sum();
                let sum_y: f64 = recent_values.iter().sum();
                let sum_xy: f64 =
                    recent_values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
                let sum_x2: f64 = (0..5).map(|i| (i * i) as f64).sum();

                let slope = (5.0 * sum_xy - sum_x * sum_y) / (5.0 * sum_x2 - sum_x * sum_x);
                let variance = recent_values
                    .iter()
                    .map(|&x| {
                        let diff = x - sum_y / 5.0;
                        diff * diff
                    })
                    .sum::<f64>()
                    / 5.0;

                // Determine trend direction and strength
                let trend_threshold = variance.sqrt() * 0.1;

                if slope > trend_threshold {
                    trend.trend_direction = TrendDirection::Improving;
                    trend.trend_strength = (slope / variance.sqrt()).abs().min(1.0) as f32;
                } else if slope < -trend_threshold {
                    trend.trend_direction = TrendDirection::Degrading;
                    trend.trend_strength = (slope / variance.sqrt()).abs().min(1.0) as f32;
                } else if variance > trend_threshold * trend_threshold {
                    trend.trend_direction = TrendDirection::Volatile;
                    trend.trend_strength = (variance.sqrt() / sum_y.abs()).min(1.0) as f32;
                } else {
                    trend.trend_direction = TrendDirection::Stable;
                    trend.trend_strength = 0.1;
                }

                // Predict next value
                let last_value = values[n - 1];
                trend.predicted_next_value = last_value + slope;
            }
        }
    }

    fn detect_anomalies(
        &mut self,
        timestamp: f64,
        latency_ms: f64,
        throughput: f32,
        memory_mb: f32,
        accuracy: f32,
    ) {
        let metrics = [
            ("latency", latency_ms),
            ("throughput", throughput as f64),
            ("memory", memory_mb as f64),
            ("accuracy", accuracy as f64),
        ];

        for (metric_name, value) in &metrics {
            self.check_metric_anomaly(timestamp, metric_name, *value);
        }
    }

    fn check_metric_anomaly(&mut self, timestamp: f64, metric_name: &str, value: f64) {
        if let Some(trend) =
            self.performance_trends.iter().find(|t| t.metric_name == metric_name).cloned()
        {
            if trend.values.len() >= 10 {
                self.process_anomaly_detection(timestamp, metric_name, value, &trend);
            }
        }
    }

    fn process_anomaly_detection(
        &mut self,
        timestamp: f64,
        metric_name: &str,
        value: f64,
        trend: &PerformanceTrend,
    ) {
        let mean = trend.values.iter().sum::<f64>() / trend.values.len() as f64;
        let variance = trend.values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
            / trend.values.len() as f64;
        let std_dev = variance.sqrt();

        let threshold = 2.0 * std_dev;
        let deviation = (value - mean).abs();

        if deviation > threshold {
            let severity = self.determine_anomaly_severity(deviation, std_dev);
            let anomaly = self.create_anomaly(timestamp, metric_name, value, mean, severity);
            self.add_anomaly(anomaly);
        }
    }

    fn determine_anomaly_severity(&self, deviation: f64, std_dev: f64) -> AnomalySeverity {
        if deviation > 3.0 * std_dev {
            AnomalySeverity::Critical
        } else if deviation > 2.5 * std_dev {
            AnomalySeverity::Warning
        } else {
            AnomalySeverity::Info
        }
    }

    fn create_anomaly(
        &self,
        timestamp: f64,
        metric_name: &str,
        actual_value: f64,
        expected_value: f64,
        severity: AnomalySeverity,
    ) -> PerformanceAnomaly {
        PerformanceAnomaly {
            timestamp,
            metric_name: metric_name.to_string(),
            expected_value,
            actual_value,
            severity,
            description: format!(
                "{metric_name} deviated by {actual_value:.2} from expected {expected_value:.2}"
            ),
            suggested_action: self.get_anomaly_suggestion(
                metric_name,
                actual_value,
                expected_value,
            ),
        }
    }

    fn add_anomaly(&mut self, anomaly: PerformanceAnomaly) {
        self.detected_anomalies.push(anomaly);

        // Keep only recent anomalies
        if self.detected_anomalies.len() > 100 {
            self.detected_anomalies.remove(0);
        }
    }

    fn get_anomaly_suggestion(&self, metric_name: &str, actual: f64, expected: f64) -> String {
        match metric_name {
            "latency" if actual > expected => {
                "Consider enabling more aggressive quantization or reducing model complexity"
                    .to_string()
            },
            "throughput" if actual < expected => {
                "Check for CPU/GPU bottlenecks or memory pressure".to_string()
            },
            "memory" if actual > expected => {
                "Enable memory optimization or reduce batch size".to_string()
            },
            "accuracy" if actual < expected => {
                "Review quantization settings or model configuration".to_string()
            },
            _ => "Monitor trend and consider adaptive optimization".to_string(),
        }
    }

    fn check_and_trigger_adaptation(
        &mut self,
        timestamp: f64,
        latency_ms: f64,
        throughput: f32,
        memory_mb: f32,
        accuracy: f32,
    ) {
        // Check if adaptation is needed based on optimization targets
        let mut should_adapt = false;
        let mut trigger_metric = String::new();

        for target in &self.adaptive_optimizer.optimization_targets {
            match target {
                OptimizationTarget::Latency
                    if latency_ms > 100.0 => {
                        // 100ms threshold
                        should_adapt = true;
                        trigger_metric = "latency".to_string();
                    },
                OptimizationTarget::Throughput
                    if throughput < 10.0 => {
                        // 10 ops/sec threshold
                        should_adapt = true;
                        trigger_metric = "throughput".to_string();
                    },
                OptimizationTarget::MemoryUsage
                    if memory_mb > 1000.0 => {
                        // 1GB threshold
                        should_adapt = true;
                        trigger_metric = "memory".to_string();
                    },
                OptimizationTarget::Accuracy
                    if accuracy < 0.9 => {
                        // 90% accuracy threshold
                        should_adapt = true;
                        trigger_metric = "accuracy".to_string();
                    },
                OptimizationTarget::Balanced
                    // Check multiple metrics with relaxed thresholds
                    if (latency_ms > 200.0
                        || throughput < 5.0
                        || memory_mb > 1500.0
                        || accuracy < 0.85)
                    => {
                        should_adapt = true;
                        trigger_metric = "balanced".to_string();
                    },
                _ => {},
            }
        }

        if should_adapt {
            self.apply_adaptive_optimization(
                timestamp,
                &trigger_metric,
                latency_ms,
                throughput,
                memory_mb,
                accuracy,
            );
        }
    }

    /// Apply an adaptive-optimization strategy switch, triggered by
    /// `trigger_metric` crossing a real threshold in
    /// `check_and_trigger_adaptation` (real: it acts on the caller-supplied
    /// `latency_ms`/`throughput`/`memory_mb`/`accuracy` from
    /// `update_real_time_metrics`).
    fn apply_adaptive_optimization(
        &mut self,
        timestamp: f64,
        trigger_metric: &str,
        _latency_ms: f64,
        _throughput: f32,
        _memory_mb: f32,
        _accuracy: f32,
    ) {
        let old_strategy = self.adaptive_optimizer.current_strategy;

        // Select new strategy based on trigger metric
        let new_strategy = match trigger_metric {
            "latency" => OptimizationStrategy::LatencyOptimized,
            "throughput" => OptimizationStrategy::ThroughputOptimized,
            "memory" => OptimizationStrategy::MemoryOptimized,
            _ => OptimizationStrategy::Hybrid,
        };

        if new_strategy != old_strategy {
            // `improvement_ratio`/`confidence_score` are `None` - see
            // `AdaptationRecord`'s doc comment for why: this profiler
            // never measures the same workload under two strategies, so
            // it has no real basis for a numeric estimate here. A
            // previous version computed one anyway from a hardcoded
            // per-transition-pair table multiplied by factors derived
            // from fabricated CPU/GPU usage telemetry.
            let adaptation = AdaptationRecord {
                timestamp,
                old_strategy,
                new_strategy,
                trigger_metric: trigger_metric.to_string(),
                improvement_ratio: None,
                confidence_score: None,
            };

            self.adaptive_optimizer.current_strategy = new_strategy;
            self.adaptive_optimizer.adaptation_history.push(adaptation);

            // Keep only recent adaptations
            if self.adaptive_optimizer.adaptation_history.len() > 50 {
                self.adaptive_optimizer.adaptation_history.remove(0);
            }

            web_sys::console::log_1(
                &format!(
                    "🤖 Adaptive optimization: {old_strategy:?} -> {new_strategy:?} (trigger: {trigger_metric})"
                )
                .into(),
            );
        }
    }

    /// Get power-aware performance recommendations from the real cached
    /// battery level.
    ///
    /// A previous version (`get_thermal_recommendations`) also emitted
    /// "High thermal state detected" / "Combined thermal and battery
    /// stress" recommendations driven by `ResourceType::Thermal` samples
    /// - which were themselves built from `estimate_cpu_temperature`/
    /// `estimate_gpu_temperature`/`get_thermal_state`, all fabricated (no
    /// standard browser API exposes device temperature or thermal
    /// throttling state to a web page; the one check that looked
    /// real - `navigator.thermalState` - is not a real API either).
    /// `sample_resources` no longer produces `Thermal`-typed samples at
    /// all, so that branch is removed along with the sampling rather
    /// than kept alive on data that can never arrive. The same
    /// fabricated temperatures also fed a `check_thermal_throttling`
    /// detector that switched optimization strategy and logged "Thermal
    /// throttling detected" - removed entirely, per the same reasoning.
    pub fn get_power_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if let Some(battery_level) = self.cached_battery_level {
            if battery_level < 0.2 {
                recommendations.push("Low battery level - enable power saving mode".to_string());
                recommendations.push("Reduce inference frequency to conserve battery".to_string());
            }
        }

        recommendations
    }
}

/// Create a performance profiler with development settings
#[wasm_bindgen]
pub fn create_development_profiler() -> PerformanceProfiler {
    PerformanceProfiler::new(ProfilerConfig::development())
}

/// Create a performance profiler with production settings
#[wasm_bindgen]
pub fn create_production_profiler() -> PerformanceProfiler {
    PerformanceProfiler::new(ProfilerConfig::production())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_config() {
        let config = ProfilerConfig::development();
        assert!(config.enabled());
        assert!(config.detailed_timing());

        let prod_config = ProfilerConfig::production();
        assert!(!prod_config.detailed_timing());
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_operation_profiling() {
        let config = ProfilerConfig::development();
        let mut profiler = PerformanceProfiler::new(config);

        profiler.start_operation("test_op", OperationType::MatMul);
        let duration = profiler.end_operation("test_op");

        assert!(duration.is_some());
        assert_eq!(profiler.operation_profiles.len(), 1);
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_performance_analysis() {
        let config = ProfilerConfig::development();
        let mut profiler = PerformanceProfiler::new(config);

        profiler.start_operation("op1", OperationType::Attention);
        profiler.end_operation("op1");

        let summary = profiler.analyze_performance_internal();
        assert_eq!(summary.operation_count, 1);
        assert!(summary.total_time_ms >= 0.0);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_profiler_config_only() {
        // Test only configuration creation for non-WASM targets
        let config = ProfilerConfig::development();
        let profiler = PerformanceProfiler::new(config);
        assert!(profiler.operation_profiles.is_empty());
        assert!(profiler.performance_trends.is_empty());
    }

    // -----------------------------------------------------------------
    // Honesty regression tests. `start_operation`/`end_operation`/
    // `sample_resources`/`create_baseline` all call `js_sys::Date::now()`
    // internally and so panic off the wasm32 target (there is no JS
    // runtime backing it) - this predates this change and is why the two
    // `#[cfg(target_arch = "wasm32")]` tests above already existed. The
    // tests below exercise the same fixes through paths that take a
    // timestamp as a plain parameter or touch no timestamp at all, so
    // they can run natively.
    // -----------------------------------------------------------------

    fn profile_with(operation_type: OperationType, duration_ms: f64) -> OperationProfile {
        OperationProfile {
            operation_type,
            operation_name: "op".to_string(),
            start_time: 0.0,
            end_time: duration_ms,
            duration_ms,
            memory_peak: 0,
            wasm_memory_growth_bytes: 0,
        }
    }

    #[test]
    fn test_wasm_memory_growth_is_saturating_and_correct() {
        assert_eq!(PerformanceProfiler::wasm_memory_growth(100, 150), 50);
        assert_eq!(PerformanceProfiler::wasm_memory_growth(100, 100), 0);
        // Must never underflow-panic even if readings were ever inverted.
        assert_eq!(PerformanceProfiler::wasm_memory_growth(150, 100), 0);
    }

    #[test]
    fn test_detect_bottlenecks_classifies_gpu_kernel_operation_as_gpu_compute() {
        // Regression test: a previous version classified bottleneck_type
        // by comparing the fabricated `gpu_time_ms > cpu_time_ms` (always
        // false, since `gpu_time_ms` was a fixed 20% of `duration_ms` and
        // `cpu_time_ms` the other 80%), so `GPUCompute` could never be
        // produced no matter what actually ran. The real signal is the
        // caller-supplied `operation_type`.
        let config = ProfilerConfig::development();
        let mut profiler = PerformanceProfiler::new(config);
        profiler.operation_profiles.push(profile_with(OperationType::GPUKernel, 1000.0));

        let bottlenecks = profiler.detect_bottlenecks();

        assert_eq!(bottlenecks.len(), 1);
        assert_eq!(bottlenecks[0].bottleneck_type, BottleneckType::GPUCompute);
    }

    #[test]
    fn test_detect_bottlenecks_classifies_non_gpu_operation_as_cpu_compute() {
        let config = ProfilerConfig::development();
        let mut profiler = PerformanceProfiler::new(config);
        profiler.operation_profiles.push(profile_with(OperationType::MatMul, 1000.0));

        let bottlenecks = profiler.detect_bottlenecks();

        assert_eq!(bottlenecks.len(), 1);
        assert_eq!(bottlenecks[0].bottleneck_type, BottleneckType::CPUCompute);
    }

    #[test]
    fn test_refresh_battery_status_leaves_cache_none_off_web() {
        // No Battery API (or any browser API at all) exists off the web
        // platform; the cache must stay `None` rather than the previous
        // `get_battery_level`'s hardcoded fallback.
        let config = ProfilerConfig::development();
        let mut profiler = PerformanceProfiler::new(config);

        futures::executor::block_on(profiler.refresh_battery_status());

        assert_eq!(profiler.cached_battery_level, None);
    }

    #[test]
    fn test_apply_adaptive_optimization_records_no_fabricated_improvement() {
        // A previous version computed `improvement_ratio` from a
        // hardcoded per-transition table multiplied by factors derived
        // from fabricated CPU/GPU telemetry, and set `confidence_score`
        // to a flat 0.8. Neither is measurable, so both must be `None`.
        let config = ProfilerConfig::development();
        let mut profiler = PerformanceProfiler::new(config);
        // The profiler starts in `Hybrid`; "memory" maps to
        // `MemoryOptimized`, so this genuinely triggers a switch (and
        // therefore a pushed `AdaptationRecord`) rather than the
        // early-return no-op path for "already on this strategy".
        profiler.apply_adaptive_optimization(0.0, "memory", 0.0, 0.0, 0.0, 0.0);

        assert_eq!(profiler.adaptive_optimizer.adaptation_history.len(), 1);
        let record = &profiler.adaptive_optimizer.adaptation_history[0];
        assert_eq!(record.improvement_ratio, None);
        assert_eq!(record.confidence_score, None);
        assert_eq!(record.new_strategy, OptimizationStrategy::MemoryOptimized);
    }

    #[test]
    fn test_get_power_recommendations_empty_without_battery_data() {
        // Honest absence: no cached battery reading must never be
        // silently treated as "battery is fine", only as "nothing to
        // recommend".
        let config = ProfilerConfig::development();
        let profiler = PerformanceProfiler::new(config);
        assert!(profiler.get_power_recommendations().is_empty());
    }

    #[test]
    fn test_get_power_recommendations_flags_low_real_battery() {
        let config = ProfilerConfig::development();
        let mut profiler = PerformanceProfiler::new(config);
        profiler.cached_battery_level = Some(0.1);

        let recommendations = profiler.get_power_recommendations();

        assert!(!recommendations.is_empty());
        assert!(recommendations.iter().any(|r| r.contains("battery")));
    }
}
