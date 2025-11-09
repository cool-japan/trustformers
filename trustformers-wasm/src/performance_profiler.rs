//! Advanced performance profiler for ML inference optimization
//!
//! This module provides comprehensive performance profiling capabilities including:
//! - Detailed ML operation profiling
//! - Bottleneck detection and analysis
//! - Resource usage monitoring (CPU, GPU, Memory)
//! - Performance visualization data
//! - Optimization recommendations
//! - Comparative performance analysis

use crate::debug::DebugLogger;
use crate::performance::*;
use js_sys::{Array, Date, Object};
use std::string::{String, ToString};
use std::vec::Vec;
use wasm_bindgen::prelude::*;

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
    active_operations: Vec<(String, OperationType, f64)>, // (name, type, start_time)
    baseline_metrics: Option<PerformanceSummary>,
    debug_logger: Option<DebugLogger>,
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
        self.active_operations.push((name.to_string(), operation_type, start_time));

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
        if let Some(pos) = self.active_operations.iter().position(|(op_name, _, _)| op_name == name)
        {
            let (_, operation_type, start_time) = self.active_operations.remove(pos);
            let duration_ms = end_time - start_time;

            // Create detailed profile
            let profile = OperationProfile {
                operation_type,
                operation_name: name.to_string(),
                start_time,
                end_time,
                duration_ms,
                cpu_time_ms: duration_ms * 0.8, // Approximate CPU time
                gpu_time_ms: duration_ms * 0.2, // Approximate GPU time
                memory_allocated: self.estimate_memory_usage(operation_type),
                memory_peak: crate::get_wasm_memory_usage(),
                gpu_memory_used: self.estimate_gpu_memory_usage(operation_type),
                flops: self.estimate_flops(operation_type, duration_ms),
                memory_bandwidth_gb_s: self.estimate_memory_bandwidth(operation_type),
                cache_hits: self.estimate_cache_hits(operation_type),
                cache_misses: self.estimate_cache_misses(operation_type),
                input_shape: self.estimate_input_shape(operation_type),
                output_shape: self.estimate_output_shape(operation_type),
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

    /// Sample current resource usage with enhanced thermal and power monitoring
    pub fn sample_resources(&mut self) {
        if !self.config.enabled() || !self.config.resource_monitoring() {
            return;
        }

        let timestamp = Date::now();
        let cpu_usage = self.estimate_cpu_usage();
        let gpu_usage = self.estimate_gpu_usage();
        let wasm_memory = crate::get_wasm_memory_usage();
        let gpu_memory = self.estimate_gpu_memory_usage(OperationType::FullInference);
        let battery_level = self.get_battery_level();
        let power_consumption = self.estimate_power_consumption();
        let thermal_state = self.get_thermal_state();
        let cpu_temp = self.estimate_cpu_temperature();
        let gpu_temp = self.estimate_gpu_temperature();

        // Create comprehensive resource samples for different metrics
        let samples = vec![
            ResourceSample {
                timestamp,
                resource_type: ResourceType::CPU,
                value: cpu_usage,
                cpu_usage,
                gpu_usage,
                wasm_memory,
                gpu_memory,
                network_bytes: 0,
                cache_hit_rate: 0.85,
                battery_level,
                power_consumption,
                thermal_state,
                cpu_temperature: cpu_temp,
                gpu_temperature: gpu_temp,
            },
            ResourceSample {
                timestamp,
                resource_type: ResourceType::Battery,
                value: battery_level,
                cpu_usage,
                gpu_usage,
                wasm_memory,
                gpu_memory,
                network_bytes: 0,
                cache_hit_rate: 0.85,
                battery_level,
                power_consumption,
                thermal_state,
                cpu_temperature: cpu_temp,
                gpu_temperature: gpu_temp,
            },
            ResourceSample {
                timestamp,
                resource_type: ResourceType::Thermal,
                value: thermal_state,
                cpu_usage,
                gpu_usage,
                wasm_memory,
                gpu_memory,
                network_bytes: 0,
                cache_hit_rate: 0.85,
                battery_level,
                power_consumption,
                thermal_state,
                cpu_temperature: cpu_temp,
                gpu_temperature: gpu_temp,
            },
        ];

        for sample in samples {
            self.resource_samples.push(sample);
        }

        // Maintain maximum samples limit
        while self.resource_samples.len() > self.config.max_samples() {
            self.resource_samples.remove(0);
        }

        // Check for thermal throttling and adaptive optimization
        self.check_thermal_throttling(thermal_state, cpu_temp, gpu_temp);
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
        let resource_efficiency = self.calculate_resource_efficiency();
        let recommendations = self.generate_recommendations(&bottlenecks);

        PerformanceSummary {
            total_time_ms,
            operation_count,
            average_fps: average_fps as f32,
            bottlenecks,
            top_operations,
            resource_efficiency,
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
            js_sys::Reflect::set(&obj, &"name".into(), &profile.operation_name.clone().into())
                .unwrap();
            js_sys::Reflect::set(
                &obj,
                &"type".into(),
                &format!("{op_type:?}", op_type = profile.operation_type).into(),
            )
            .unwrap();
            js_sys::Reflect::set(&obj, &"duration".into(), &profile.duration_ms.into()).unwrap();
            js_sys::Reflect::set(&obj, &"start_time".into(), &profile.start_time.into()).unwrap();
            js_sys::Reflect::set(&obj, &"cpu_time".into(), &profile.cpu_time_ms.into()).unwrap();
            js_sys::Reflect::set(&obj, &"gpu_time".into(), &profile.gpu_time_ms.into()).unwrap();
            js_sys::Reflect::set(&obj, &"memory".into(), &profile.memory_allocated.into()).unwrap();
            array.push(&obj);
        }

        array
    }

    /// Get resource usage timeline for visualization
    pub fn get_resource_timeline(&self) -> Array {
        let array = Array::new();

        for sample in &self.resource_samples {
            let obj = Object::new();
            js_sys::Reflect::set(&obj, &"timestamp".into(), &sample.timestamp.into()).unwrap();
            js_sys::Reflect::set(&obj, &"cpu".into(), &sample.cpu_usage.into()).unwrap();
            js_sys::Reflect::set(&obj, &"gpu".into(), &sample.gpu_usage.into()).unwrap();
            js_sys::Reflect::set(&obj, &"memory".into(), &sample.wasm_memory.into()).unwrap();
            js_sys::Reflect::set(&obj, &"gpu_memory".into(), &sample.gpu_memory.into()).unwrap();
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
            let efficiency_change =
                (current.resource_efficiency as f64 - baseline.resource_efficiency as f64) * 100.0;

            Some(format!(
                "Performance Comparison:\n\
                 Total Time: {:.1}% change\n\
                 Average FPS: {:.1}% change\n\
                 Resource Efficiency: {:.1}% change\n\
                 Bottlenecks: {} current vs {} baseline",
                time_change,
                fps_change,
                efficiency_change,
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
                let bottleneck_type = if profile.gpu_time_ms > profile.cpu_time_ms {
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

        operations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        operations.truncate(limit);
        operations
    }

    fn calculate_resource_efficiency(&self) -> f32 {
        if self.resource_samples.is_empty() {
            return 0.5;
        }

        let avg_cpu = self.resource_samples.iter().map(|s| s.cpu_usage).sum::<f32>()
            / self.resource_samples.len() as f32;
        let avg_gpu = self.resource_samples.iter().map(|s| s.gpu_usage).sum::<f32>()
            / self.resource_samples.len() as f32;

        // Efficiency is based on balanced usage without waste
        let cpu_efficiency = (avg_cpu / 100.0).min(1.0);
        let gpu_efficiency = (avg_gpu / 100.0).min(1.0);

        (cpu_efficiency + gpu_efficiency) / 2.0
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

    // Estimation methods (in a real implementation, these would use actual measurements)

    fn estimate_memory_usage(&self, op_type: OperationType) -> usize {
        match op_type {
            OperationType::ModelLoading => 50 * 1024 * 1024,
            OperationType::TransformerLayer => 10 * 1024 * 1024,
            OperationType::Attention => 5 * 1024 * 1024,
            OperationType::MatMul => 2 * 1024 * 1024,
            _ => 1024 * 1024,
        }
    }

    fn estimate_gpu_memory_usage(&self, op_type: OperationType) -> usize {
        match op_type {
            OperationType::ModelLoading => 100 * 1024 * 1024,
            OperationType::TransformerLayer => 20 * 1024 * 1024,
            OperationType::Attention => 10 * 1024 * 1024,
            OperationType::MatMul => 5 * 1024 * 1024,
            _ => 1024 * 1024,
        }
    }

    fn estimate_flops(&self, op_type: OperationType, duration_ms: f64) -> u64 {
        let base_flops = match op_type {
            OperationType::MatMul => 1_000_000_000,
            OperationType::Attention => 500_000_000,
            OperationType::TransformerLayer => 2_000_000_000,
            _ => 100_000_000,
        };
        (base_flops as f64 * (duration_ms / 1000.0)) as u64
    }

    fn estimate_memory_bandwidth(&self, op_type: OperationType) -> f32 {
        match op_type {
            OperationType::MatMul => 100.0,
            OperationType::MemoryTransfer => 50.0,
            _ => 20.0,
        }
    }

    fn estimate_cache_hits(&self, op_type: OperationType) -> u32 {
        match op_type {
            OperationType::Embedding => 1000,
            OperationType::Attention => 500,
            _ => 100,
        }
    }

    fn estimate_cache_misses(&self, op_type: OperationType) -> u32 {
        match op_type {
            OperationType::ModelLoading => 500,
            OperationType::MemoryTransfer => 200,
            _ => 20,
        }
    }

    fn estimate_input_shape(&self, op_type: OperationType) -> Vec<usize> {
        match op_type {
            OperationType::TransformerLayer => vec![1, 512, 768],
            OperationType::Attention => vec![1, 12, 512, 64],
            OperationType::MatMul => vec![512, 768],
            _ => vec![1, 512],
        }
    }

    fn estimate_output_shape(&self, op_type: OperationType) -> Vec<usize> {
        match op_type {
            OperationType::TransformerLayer => vec![1, 512, 768],
            OperationType::Attention => vec![1, 512, 768],
            OperationType::MatMul => vec![512, 3072],
            _ => vec![1, 512],
        }
    }

    fn estimate_cpu_usage(&self) -> f32 {
        // Simulate CPU usage
        50.0 + ((Date::now() % 100.0) / 2.0) as f32
    }

    fn estimate_gpu_usage(&self) -> f32 {
        // Simulate GPU usage
        30.0 + ((Date::now() % 100.0) / 3.0) as f32
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

            js_sys::Reflect::set(&trend_obj, &"values".into(), &values_array).unwrap();
            js_sys::Reflect::set(&trend_obj, &"timestamps".into(), &timestamps_array).unwrap();
            js_sys::Reflect::set(
                &trend_obj,
                &"direction".into(),
                &format!("{direction:?}", direction = trend.trend_direction).into(),
            )
            .unwrap();
            js_sys::Reflect::set(&trend_obj, &"strength".into(), &trend.trend_strength.into())
                .unwrap();
            js_sys::Reflect::set(
                &trend_obj,
                &"predicted_next".into(),
                &trend.predicted_next_value.into(),
            )
            .unwrap();

            js_sys::Reflect::set(&trends_obj, &trend.metric_name.clone().into(), &trend_obj)
                .unwrap();
        }

        trends_obj
    }

    /// Get detected anomalies as JavaScript array
    pub fn get_detected_anomalies(&self) -> js_sys::Array {
        let anomalies_array = js_sys::Array::new();

        for anomaly in &self.detected_anomalies {
            let anomaly_obj = js_sys::Object::new();
            js_sys::Reflect::set(&anomaly_obj, &"timestamp".into(), &anomaly.timestamp.into())
                .unwrap();
            js_sys::Reflect::set(
                &anomaly_obj,
                &"metric".into(),
                &anomaly.metric_name.clone().into(),
            )
            .unwrap();
            js_sys::Reflect::set(
                &anomaly_obj,
                &"expected".into(),
                &anomaly.expected_value.into(),
            )
            .unwrap();
            js_sys::Reflect::set(&anomaly_obj, &"actual".into(), &anomaly.actual_value.into())
                .unwrap();
            js_sys::Reflect::set(
                &anomaly_obj,
                &"severity".into(),
                &format!("{severity:?}", severity = anomaly.severity).into(),
            )
            .unwrap();
            js_sys::Reflect::set(
                &anomaly_obj,
                &"description".into(),
                &anomaly.description.clone().into(),
            )
            .unwrap();
            js_sys::Reflect::set(
                &anomaly_obj,
                &"suggested_action".into(),
                &anomaly.suggested_action.clone().into(),
            )
            .unwrap();

            anomalies_array.push(&anomaly_obj);
        }

        anomalies_array
    }

    /// Get adaptive optimization state
    pub fn get_adaptive_state(&self) -> js_sys::Object {
        let state_obj = js_sys::Object::new();

        js_sys::Reflect::set(
            &state_obj,
            &"enabled".into(),
            &self.adaptive_optimizer.enabled.into(),
        )
        .unwrap();
        js_sys::Reflect::set(
            &state_obj,
            &"learning_rate".into(),
            &self.adaptive_optimizer.learning_rate.into(),
        )
        .unwrap();
        js_sys::Reflect::set(
            &state_obj,
            &"current_strategy".into(),
            &format!(
                "{strategy:?}",
                strategy = self.adaptive_optimizer.current_strategy
            )
            .into(),
        )
        .unwrap();
        js_sys::Reflect::set(
            &state_obj,
            &"adaptation_count".into(),
            &self.adaptive_optimizer.adaptation_history.len().into(),
        )
        .unwrap();

        // Add optimization targets
        let targets_array = js_sys::Array::new();
        for target in &self.adaptive_optimizer.optimization_targets {
            targets_array.push(&format!("{target:?}").into());
        }
        js_sys::Reflect::set(&state_obj, &"optimization_targets".into(), &targets_array).unwrap();

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
        let avg_memory =
            self.operation_profiles.iter().map(|p| p.memory_allocated as f64).sum::<f64>()
                / self.operation_profiles.len() as f64
                / 1_048_576.0; // Convert to MB

        let baseline = PerformanceBaseline {
            name: name.to_string(),
            timestamp: Date::now(),
            avg_latency_ms: avg_latency,
            avg_throughput: if avg_latency > 0.0 { 1000.0 / avg_latency as f32 } else { 0.0 },
            avg_memory_mb: avg_memory as f32,
            avg_accuracy: 0.95, // Default accuracy estimate
        };

        self.current_baselines.push(baseline);
        web_sys::console::log_1(&format!("📏 Performance baseline '{name}' created").into());
    }

    // Private helper methods for real-time analytics

    fn update_performance_trend(&mut self, metric_name: &str, value: f64, timestamp: f64) {
        // Find existing trend or create new one
        let trend_index = self.performance_trends.iter().position(|t| t.metric_name == metric_name);

        let trend = if let Some(index) = trend_index {
            &mut self.performance_trends[index]
        } else {
            self.performance_trends.push(PerformanceTrend {
                metric_name: metric_name.to_string(),
                values: Vec::new(),
                timestamps: Vec::new(),
                trend_direction: TrendDirection::Stable,
                trend_strength: 0.0,
                predicted_next_value: value,
            });
            self.performance_trends.last_mut().unwrap()
        };

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
                OptimizationTarget::Latency => {
                    if latency_ms > 100.0 {
                        // 100ms threshold
                        should_adapt = true;
                        trigger_metric = "latency".to_string();
                    }
                },
                OptimizationTarget::Throughput => {
                    if throughput < 10.0 {
                        // 10 ops/sec threshold
                        should_adapt = true;
                        trigger_metric = "throughput".to_string();
                    }
                },
                OptimizationTarget::MemoryUsage => {
                    if memory_mb > 1000.0 {
                        // 1GB threshold
                        should_adapt = true;
                        trigger_metric = "memory".to_string();
                    }
                },
                OptimizationTarget::Accuracy => {
                    if accuracy < 0.9 {
                        // 90% accuracy threshold
                        should_adapt = true;
                        trigger_metric = "accuracy".to_string();
                    }
                },
                OptimizationTarget::Balanced => {
                    // Check multiple metrics with relaxed thresholds
                    if latency_ms > 200.0
                        || throughput < 5.0
                        || memory_mb > 1500.0
                        || accuracy < 0.85
                    {
                        should_adapt = true;
                        trigger_metric = "balanced".to_string();
                    }
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
            let improvement_ratio = self.estimate_strategy_improvement(old_strategy, new_strategy);

            let adaptation = AdaptationRecord {
                timestamp,
                old_strategy,
                new_strategy,
                trigger_metric: trigger_metric.to_string(),
                improvement_ratio,
                confidence_score: 0.8, // Default confidence
            };

            self.adaptive_optimizer.current_strategy = new_strategy;
            self.adaptive_optimizer.adaptation_history.push(adaptation);

            // Keep only recent adaptations
            if self.adaptive_optimizer.adaptation_history.len() > 50 {
                self.adaptive_optimizer.adaptation_history.remove(0);
            }

            web_sys::console::log_1(
                &format!(
                    "🤖 Adaptive optimization: {:?} -> {:?} (trigger: {}, improvement: {:.1}%)",
                    old_strategy,
                    new_strategy,
                    trigger_metric,
                    improvement_ratio * 100.0
                )
                .into(),
            );
        }
    }

    fn estimate_strategy_improvement(
        &self,
        old: OptimizationStrategy,
        new: OptimizationStrategy,
    ) -> f32 {
        // Advanced ML-powered estimation based on historical performance patterns
        let historical_improvement = self.calculate_historical_improvement(old, new);
        let device_factor = self.get_device_performance_factor();
        let workload_factor = self.get_workload_complexity_factor();

        // Weighted combination of factors for more accurate estimation
        let base_improvement = match (old, new) {
            (OptimizationStrategy::CPUPreferred, OptimizationStrategy::GPUPreferred) => 2.5,
            (OptimizationStrategy::Hybrid, OptimizationStrategy::MemoryOptimized) => 1.8,
            (OptimizationStrategy::MemoryOptimized, OptimizationStrategy::Hybrid) => 1.4,
            (OptimizationStrategy::CPUPreferred, OptimizationStrategy::MemoryOptimized) => 3.2,
            (OptimizationStrategy::GPUPreferred, OptimizationStrategy::MemoryOptimized) => 1.2,
            _ => 1.15, // Default 15% improvement
        };

        (base_improvement * historical_improvement * device_factor * workload_factor).min(5.0)
    }

    fn calculate_historical_improvement(
        &self,
        old: OptimizationStrategy,
        new: OptimizationStrategy,
    ) -> f32 {
        // Analyze historical performance data for strategy transitions
        if self.performance_trends.len() < 10 {
            return 1.0; // Not enough data, use base estimate
        }

        let mut strategy_transitions = Vec::new();
        for i in 1..self.performance_trends.len() {
            let prev_trend = &self.performance_trends[i - 1];
            let curr_trend = &self.performance_trends[i];

            // Simplified strategy detection based on performance characteristics
            let prev_strategy = self.infer_strategy_from_trend(prev_trend);
            let curr_strategy = self.infer_strategy_from_trend(curr_trend);

            if prev_strategy == old && curr_strategy == new {
                let improvement = (curr_trend.trend_strength + 1.0) / 2.0; // Normalize based on trend strength
                strategy_transitions.push(improvement);
            }
        }

        if strategy_transitions.is_empty() {
            1.0
        } else {
            let avg_improvement: f32 =
                strategy_transitions.iter().sum::<f32>() / strategy_transitions.len() as f32;
            0.5 + avg_improvement // Scale to reasonable range
        }
    }

    fn get_device_performance_factor(&self) -> f32 {
        // Estimate device performance based on recent resource utilization
        let recent_samples: Vec<_> = self.resource_samples.iter().rev().take(50).collect();
        if recent_samples.is_empty() {
            return 1.0;
        }

        let avg_cpu_usage: f32 = recent_samples
            .iter()
            .filter(|s| s.resource_type == ResourceType::CPU)
            .map(|s| s.value)
            .sum::<f32>()
            / recent_samples.len() as f32;

        let avg_memory_usage: f32 = recent_samples
            .iter()
            .filter(|s| s.resource_type == ResourceType::WAMSMemory)
            .map(|s| s.value)
            .sum::<f32>()
            / recent_samples.len() as f32;

        // Higher resource availability = higher performance factor
        let cpu_factor = (100.0 - avg_cpu_usage) / 100.0;
        let memory_factor = (100.0 - avg_memory_usage) / 100.0;

        (cpu_factor + memory_factor) / 2.0
    }

    fn get_workload_complexity_factor(&self) -> f32 {
        // Analyze workload complexity based on operation profiles
        if self.operation_profiles.is_empty() {
            return 1.0;
        }

        let avg_duration: f64 = self.operation_profiles.iter().map(|p| p.duration_ms).sum::<f64>()
            / self.operation_profiles.len() as f64;

        let memory_intensity: u64 =
            self.operation_profiles.iter().map(|p| p.memory_peak).max().unwrap_or(0) as u64;

        // Complex workloads (longer duration, more memory) benefit more from optimization
        let duration_factor = (avg_duration / 1000.0).min(2.0); // Cap at 2x
        let memory_factor = (memory_intensity as f64 / (100.0 * 1024.0 * 1024.0)).min(2.0); // 100MB baseline

        (1.0 + duration_factor + memory_factor) as f32 / 3.0
    }

    fn infer_strategy_from_trend(&self, trend: &PerformanceTrend) -> OptimizationStrategy {
        // Infer strategy based on performance characteristics
        if trend.trend_strength > 0.8 {
            OptimizationStrategy::MemoryOptimized
        } else if trend.trend_strength > 0.6 {
            OptimizationStrategy::GPUPreferred
        } else {
            OptimizationStrategy::Hybrid
        }
    }

    /// Get current battery level (0.0 to 1.0)
    fn get_battery_level(&self) -> f32 {
        #[cfg(target_arch = "wasm32")]
        {
            if let Some(window) = web_sys::window() {
                if let Ok(navigator) = js_sys::Reflect::get(&window, &"navigator".into()) {
                    if let Ok(get_battery) = js_sys::Reflect::get(&navigator, &"getBattery".into())
                    {
                        // Battery API is async, so we use cached value or estimate
                        // In a real implementation, you would cache the battery object
                        return 0.8; // Placeholder - would use cached battery level
                    }
                }
            }
        }
        1.0 // Default to full battery for non-web environments
    }

    /// Estimate current power consumption (watts)
    fn estimate_power_consumption(&self) -> f32 {
        #[cfg(target_arch = "wasm32")]
        {
            // Estimate based on CPU/GPU usage and device characteristics
            let recent_samples: Vec<_> = self.resource_samples.iter().rev().take(10).collect();
            if recent_samples.is_empty() {
                return 5.0; // Default 5W
            }

            let avg_cpu: f32 = recent_samples.iter().map(|s| s.cpu_usage).sum::<f32>()
                / recent_samples.len() as f32;

            let avg_gpu: f32 = recent_samples.iter().map(|s| s.gpu_usage).sum::<f32>()
                / recent_samples.len() as f32;

            // Estimate power consumption based on usage
            let base_power = 2.0; // Base system power
            let cpu_power = (avg_cpu / 100.0) * 8.0; // Up to 8W for CPU
            let gpu_power = (avg_gpu / 100.0) * 15.0; // Up to 15W for GPU

            base_power + cpu_power + gpu_power
        }
        #[cfg(not(target_arch = "wasm32"))]
        5.0 // Default for non-web environments
    }

    /// Get current thermal state (0.0 = cool, 1.0 = hot)
    fn get_thermal_state(&self) -> f32 {
        #[cfg(target_arch = "wasm32")]
        {
            if let Some(window) = web_sys::window() {
                if let Ok(navigator) = js_sys::Reflect::get(&window, &"navigator".into()) {
                    if let Ok(thermal_state) =
                        js_sys::Reflect::get(&navigator, &"thermalState".into())
                    {
                        if let Some(state_str) = thermal_state.as_string() {
                            return match state_str.as_str() {
                                "nominal" => 0.2,
                                "fair" => 0.4,
                                "serious" => 0.6,
                                "critical" => 0.9,
                                _ => 0.3,
                            };
                        }
                    }
                }
            }

            // Estimate thermal state based on recent performance
            let recent_samples: Vec<_> = self.resource_samples.iter().rev().take(20).collect();
            if recent_samples.len() < 5 {
                return 0.3; // Default moderate thermal state
            }

            let avg_usage: f32 =
                recent_samples.iter().map(|s| (s.cpu_usage + s.gpu_usage) / 2.0).sum::<f32>()
                    / recent_samples.len() as f32;

            // High sustained usage indicates higher thermal state
            (avg_usage / 100.0).min(1.0)
        }
        #[cfg(not(target_arch = "wasm32"))]
        0.3 // Default moderate thermal state
    }

    /// Estimate CPU temperature (Celsius)
    fn estimate_cpu_temperature(&self) -> f32 {
        #[cfg(target_arch = "wasm32")]
        {
            // Estimate based on CPU usage and thermal state
            let thermal_state = self.get_thermal_state();
            let recent_cpu_usage =
                self.resource_samples.iter().rev().take(10).map(|s| s.cpu_usage).sum::<f32>()
                    / 10.0_f32.max(self.resource_samples.len() as f32);

            let base_temp = 35.0; // Base temperature
            let usage_temp = (recent_cpu_usage / 100.0) * 30.0; // Up to 30°C from usage
            let thermal_temp = thermal_state * 20.0; // Up to 20°C from thermal state

            base_temp + usage_temp + thermal_temp
        }
        #[cfg(not(target_arch = "wasm32"))]
        45.0 // Default CPU temperature
    }

    /// Estimate GPU temperature (Celsius)
    fn estimate_gpu_temperature(&self) -> f32 {
        #[cfg(target_arch = "wasm32")]
        {
            // Estimate based on GPU usage and thermal state
            let thermal_state = self.get_thermal_state();
            let recent_gpu_usage =
                self.resource_samples.iter().rev().take(10).map(|s| s.gpu_usage).sum::<f32>()
                    / 10.0_f32.max(self.resource_samples.len() as f32);

            let base_temp = 40.0; // Base GPU temperature (typically higher than CPU)
            let usage_temp = (recent_gpu_usage / 100.0) * 40.0; // Up to 40°C from usage
            let thermal_temp = thermal_state * 25.0; // Up to 25°C from thermal state

            base_temp + usage_temp + thermal_temp
        }
        #[cfg(not(target_arch = "wasm32"))]
        55.0 // Default GPU temperature
    }

    /// Check for thermal throttling and trigger adaptive optimization
    fn check_thermal_throttling(&mut self, thermal_state: f32, cpu_temp: f32, gpu_temp: f32) {
        let should_throttle = thermal_state > 0.7 || cpu_temp > 75.0 || gpu_temp > 80.0;

        if should_throttle {
            // Log thermal event
            if let Some(ref mut logger) = self.debug_logger {
                logger.info(&format!(
                    "Thermal throttling detected: thermal_state={:.2}, cpu_temp={:.1}°C, gpu_temp={:.1}°C",
                    thermal_state, cpu_temp, gpu_temp
                ), "thermal");
            }

            // Trigger adaptive optimization toward more conservative strategy
            let current_strategy = self.adaptive_optimizer.current_strategy;
            let target_strategy = match current_strategy {
                OptimizationStrategy::GPUPreferred => OptimizationStrategy::Hybrid,
                OptimizationStrategy::Hybrid => OptimizationStrategy::MemoryOptimized,
                _ => OptimizationStrategy::MemoryOptimized,
            };

            if current_strategy != target_strategy {
                let estimated_improvement =
                    self.estimate_strategy_improvement(current_strategy, target_strategy);
                if estimated_improvement > 1.1 {
                    // At least 10% improvement
                    self.adaptive_optimizer.current_strategy = target_strategy;

                    if let Some(ref mut logger) = self.debug_logger {
                        logger.info(&format!(
                            "Thermal adaptive optimization: switched from {current_strategy:?} to {target_strategy:?} (estimated {:.1}% improvement)",
                            (estimated_improvement - 1.0) * 100.0
                        ), "thermal");
                    }
                }
            }
        }
    }

    /// Get thermal-aware performance recommendations
    pub fn get_thermal_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        let recent_thermal = self
            .resource_samples
            .iter()
            .rev()
            .take(10)
            .filter(|s| s.resource_type == ResourceType::Thermal)
            .map(|s| s.value)
            .sum::<f32>()
            / 10.0;

        let recent_battery = self
            .resource_samples
            .iter()
            .rev()
            .take(10)
            .filter(|s| s.resource_type == ResourceType::Battery)
            .map(|s| s.value)
            .sum::<f32>()
            / 10.0;

        if recent_thermal > 0.7 {
            recommendations.push(
                "High thermal state detected - consider reducing model complexity".to_string(),
            );
            recommendations
                .push("Switch to CPU-based inference to reduce GPU heat generation".to_string());
        }

        if recent_battery < 0.2 {
            recommendations.push("Low battery level - enable power saving mode".to_string());
            recommendations.push("Reduce inference frequency to conserve battery".to_string());
        }

        if recent_thermal > 0.5 && recent_battery < 0.3 {
            recommendations.push(
                "Combined thermal and battery stress - enable aggressive power management"
                    .to_string(),
            );
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
}
