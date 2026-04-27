//! Performance profiling tools for debugging

pub mod events;
pub mod gpu;
pub mod io_monitor;
pub mod memory;
pub mod report;

// Re-export all public types from sub-modules
pub use events::{
    BottleneckSeverity, BottleneckType, CpuBottleneckAnalysis, CpuProfile, HotFunction,
    MemorySnapshot, PerformanceBottleneck, ProfileEvent, ProfileStats,
};
pub use gpu::{GpuKernelProfile, GpuKernelSummary, GpuMemoryPool, GpuProfiler};
pub use io_monitor::{
    BandwidthSample, IoDeviceType, IoMonitor, IoOperation, IoOperationType, IoPerformanceSummary,
    IoProfile, LayerLatencyProfile,
};
pub use memory::{
    MemoryAllocation, MemoryAllocationType, MemoryEfficiencyAnalysis, MemoryStats, MemoryTracker,
};
pub use report::{
    EnhancedProfilerReport, LayerLatencyAnalysis, MemoryAllocationSummary, PerformanceAnalysis,
    ProfilerReport,
};

use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};
use uuid::Uuid;

use crate::DebugConfig;

/// Performance profiler
#[derive(Debug)]
pub struct Profiler {
    #[allow(dead_code)]
    config: DebugConfig,
    events: Vec<ProfileEvent>,
    active_timers: HashMap<String, Instant>,
    memory_snapshots: Vec<MemorySnapshot>,
    start_time: Option<Instant>,
    layer_profiles: HashMap<String, LayerProfile>,
    bottlenecks: Vec<PerformanceBottleneck>,
    // Enhanced profiling features
    gpu_kernel_profiles: Vec<GpuKernelProfile>,
    memory_allocations: HashMap<Uuid, MemoryAllocation>,
    layer_latency_profiles: HashMap<String, LayerLatencyProfile>,
    io_profiles: Vec<IoProfile>,
    cpu_bottleneck_analysis: Vec<CpuBottleneckAnalysis>,
    memory_tracker: Arc<Mutex<MemoryTracker>>,
    gpu_profiler: Option<GpuProfiler>,
    io_monitor: IoMonitor,
}

#[derive(Debug)]
pub struct LayerProfile {
    #[allow(dead_code)]
    layer_name: String,
    forward_times: Vec<Duration>,
    backward_times: Vec<Duration>,
    memory_usage: Vec<usize>,
    call_count: usize,
}

impl LayerProfile {
    /// Get forward execution times
    pub fn forward_times(&self) -> &Vec<Duration> {
        &self.forward_times
    }

    /// Get backward execution times
    pub fn backward_times(&self) -> &Vec<Duration> {
        &self.backward_times
    }

    /// Get memory usage samples
    pub fn memory_usage(&self) -> &Vec<usize> {
        &self.memory_usage
    }

    /// Get total number of calls
    pub fn call_count(&self) -> usize {
        self.call_count
    }
}

impl Profiler {
    /// Create a new profiler
    pub fn new(config: &DebugConfig) -> Self {
        Self {
            config: config.clone(),
            events: Vec::new(),
            active_timers: HashMap::new(),
            memory_snapshots: Vec::new(),
            start_time: None,
            layer_profiles: HashMap::new(),
            bottlenecks: Vec::new(),
            // Enhanced profiling features
            gpu_kernel_profiles: Vec::new(),
            memory_allocations: HashMap::new(),
            layer_latency_profiles: HashMap::new(),
            io_profiles: Vec::new(),
            cpu_bottleneck_analysis: Vec::new(),
            memory_tracker: Arc::new(Mutex::new(MemoryTracker::new())),
            gpu_profiler: GpuProfiler::new().ok(),
            io_monitor: IoMonitor::new(),
        }
    }

    /// Start profiling session
    pub async fn start(&mut self) -> Result<()> {
        tracing::info!("Starting performance profiler");
        self.start_time = Some(Instant::now());
        self.take_memory_snapshot();
        Ok(())
    }

    /// Get reference to profiling events
    pub fn get_events(&self) -> &Vec<ProfileEvent> {
        &self.events
    }

    /// Start timing a function or operation
    pub fn start_timer(&mut self, name: &str) {
        self.active_timers.insert(name.to_string(), Instant::now());
    }

    /// End timing and record the event
    pub fn end_timer(&mut self, name: &str) -> Option<Duration> {
        if let Some(start_time) = self.active_timers.remove(name) {
            let duration = start_time.elapsed();

            // Record basic function call event
            self.events.push(ProfileEvent::FunctionCall {
                function_name: name.to_string(),
                duration,
                memory_delta: 0, // Would need actual memory tracking
            });

            Some(duration)
        } else {
            tracing::warn!("Timer '{}' was not started", name);
            None
        }
    }

    /// Record layer execution timing
    pub fn record_layer_execution(
        &mut self,
        layer_name: &str,
        layer_type: &str,
        forward_time: Duration,
        backward_time: Option<Duration>,
        memory_usage: usize,
        parameter_count: usize,
    ) {
        // Record event
        self.events.push(ProfileEvent::LayerExecution {
            layer_name: layer_name.to_string(),
            layer_type: layer_type.to_string(),
            forward_time,
            backward_time,
            memory_usage,
            parameter_count,
        });

        // Update layer profile
        let profile =
            self.layer_profiles
                .entry(layer_name.to_string())
                .or_insert_with(|| LayerProfile {
                    layer_name: layer_name.to_string(),
                    forward_times: Vec::new(),
                    backward_times: Vec::new(),
                    memory_usage: Vec::new(),
                    call_count: 0,
                });

        profile.forward_times.push(forward_time);
        if let Some(backward) = backward_time {
            profile.backward_times.push(backward);
        }
        profile.memory_usage.push(memory_usage);
        profile.call_count += 1;
    }

    /// Record tensor operation timing
    pub fn record_tensor_operation(
        &mut self,
        operation: &str,
        tensor_shape: &[usize],
        duration: Duration,
        memory_allocated: usize,
    ) {
        self.events.push(ProfileEvent::TensorOperation {
            operation: operation.to_string(),
            tensor_shape: tensor_shape.to_vec(),
            duration,
            memory_allocated,
        });
    }

    /// Record model inference timing
    pub fn record_model_inference(
        &mut self,
        batch_size: usize,
        sequence_length: usize,
        duration: Duration,
    ) {
        let tokens_per_second = (batch_size * sequence_length) as f64 / duration.as_secs_f64();

        self.events.push(ProfileEvent::ModelInference {
            batch_size,
            sequence_length,
            duration,
            tokens_per_second,
        });
    }

    /// Record gradient computation timing
    pub fn record_gradient_computation(
        &mut self,
        layer_name: &str,
        gradient_norm: f64,
        duration: Duration,
    ) {
        self.events.push(ProfileEvent::GradientComputation {
            layer_name: layer_name.to_string(),
            gradient_norm,
            duration,
        });
    }

    /// Take a memory usage snapshot
    pub fn take_memory_snapshot(&mut self) {
        // Simplified memory tracking - in practice would use system APIs
        let snapshot = MemorySnapshot {
            timestamp: chrono::Utc::now(),
            heap_allocated: 0, // Would get from system
            heap_used: 0,
            stack_size: 0,
            gpu_allocated: None,
            gpu_used: None,
        };

        self.memory_snapshots.push(snapshot);

        // Keep only recent snapshots to prevent memory growth
        if self.memory_snapshots.len() > 1000 {
            self.memory_snapshots.drain(0..500);
        }
    }

    /// Analyze performance and detect bottlenecks
    pub fn analyze_performance(&mut self) -> Vec<PerformanceBottleneck> {
        self.bottlenecks.clear();

        // Analyze layer execution times
        self.analyze_layer_bottlenecks();

        // Analyze memory usage patterns
        self.analyze_memory_bottlenecks();

        // Analyze tensor operation efficiency
        self.analyze_tensor_bottlenecks();

        self.bottlenecks.clone()
    }

    /// Get profiling statistics
    pub fn get_statistics(&self) -> HashMap<String, ProfileStats> {
        let mut stats = HashMap::new();

        // Group events by type
        let mut grouped_events: HashMap<String, Vec<&ProfileEvent>> = HashMap::new();

        for event in &self.events {
            let event_type = match event {
                ProfileEvent::FunctionCall { .. } => "FunctionCall",
                ProfileEvent::LayerExecution { .. } => "LayerExecution",
                ProfileEvent::TensorOperation { .. } => "TensorOperation",
                ProfileEvent::ModelInference { .. } => "ModelInference",
                ProfileEvent::GradientComputation { .. } => "GradientComputation",
            };

            grouped_events.entry(event_type.to_string()).or_default().push(event);
        }

        // Calculate statistics for each event type
        for (event_type, events) in grouped_events {
            let durations: Vec<Duration> = events
                .iter()
                .filter_map(|event| match event {
                    ProfileEvent::FunctionCall { duration, .. } => Some(*duration),
                    ProfileEvent::LayerExecution { forward_time, .. } => Some(*forward_time),
                    ProfileEvent::TensorOperation { duration, .. } => Some(*duration),
                    ProfileEvent::ModelInference { duration, .. } => Some(*duration),
                    ProfileEvent::GradientComputation { duration, .. } => Some(*duration),
                })
                .collect();

            if !durations.is_empty() {
                let total_duration: Duration = durations.iter().sum();
                let avg_duration = total_duration / durations.len() as u32;
                let min_duration = durations.iter().min().copied().unwrap_or_default();
                let max_duration = durations.iter().max().copied().unwrap_or_default();

                stats.insert(
                    event_type.clone(),
                    ProfileStats {
                        event_type,
                        count: durations.len(),
                        total_duration,
                        avg_duration,
                        min_duration,
                        max_duration,
                        total_memory: 0, // Simplified
                        avg_memory: 0.0,
                    },
                );
            }
        }

        stats
    }

    /// Get layer-specific performance profiles
    pub fn get_layer_profiles(&self) -> &HashMap<String, LayerProfile> {
        &self.layer_profiles
    }

    /// Get memory usage over time
    pub fn get_memory_timeline(&self) -> &[MemorySnapshot] {
        &self.memory_snapshots
    }

    /// Generate performance report
    pub async fn generate_report(&self) -> Result<ProfilerReport> {
        let statistics = self.get_statistics();
        let bottlenecks = self.bottlenecks.clone();
        let total_events = self.events.len();

        let total_runtime =
            if let Some(start) = self.start_time { start.elapsed() } else { Duration::ZERO };

        // Calculate slowest layers
        let slowest_layers = self.get_slowest_layers(5);

        // Memory efficiency analysis
        let memory_efficiency = self.analyze_memory_efficiency();

        Ok(ProfilerReport {
            total_events,
            total_runtime,
            statistics,
            bottlenecks,
            slowest_layers,
            memory_efficiency,
            recommendations: self.generate_performance_recommendations(),
        })
    }

    /// Clear all profiling data
    pub fn clear(&mut self) {
        self.events.clear();
        self.active_timers.clear();
        self.memory_snapshots.clear();
        self.layer_profiles.clear();
        self.bottlenecks.clear();
        self.start_time = None;
        // Clear enhanced profiling data
        self.gpu_kernel_profiles.clear();
        self.memory_allocations.clear();
        self.layer_latency_profiles.clear();
        self.io_profiles.clear();
        self.cpu_bottleneck_analysis.clear();
        if let Ok(mut tracker) = self.memory_tracker.lock() {
            *tracker = MemoryTracker::new();
        }
        self.io_monitor = IoMonitor::new();
    }

    // Enhanced profiling methods

    /// Profile GPU kernel execution
    pub fn profile_gpu_kernel(&mut self, kernel_profile: GpuKernelProfile) {
        if let Some(ref mut gpu_profiler) = self.gpu_profiler {
            gpu_profiler.profile_kernel(kernel_profile.clone());
        }
        self.gpu_kernel_profiles.push(kernel_profile);
    }

    /// Track memory allocation
    pub fn track_memory_allocation(
        &mut self,
        size_bytes: usize,
        allocation_type: MemoryAllocationType,
        device_id: Option<i32>,
        stack_trace: Vec<String>,
    ) -> Uuid {
        let allocation_id = Uuid::new_v4();
        let allocation = MemoryAllocation {
            allocation_id,
            size_bytes,
            allocation_type,
            device_id,
            timestamp: SystemTime::now(),
            stack_trace,
            freed: false,
            free_timestamp: None,
        };

        if let Ok(mut tracker) = self.memory_tracker.lock() {
            tracker.track_allocation(allocation.clone());
        }

        self.memory_allocations.insert(allocation_id, allocation);
        allocation_id
    }

    /// Track memory deallocation
    pub fn track_memory_deallocation(&mut self, allocation_id: Uuid) {
        if let Some(allocation) = self.memory_allocations.get_mut(&allocation_id) {
            allocation.freed = true;
            allocation.free_timestamp = Some(SystemTime::now());
        }

        if let Ok(mut tracker) = self.memory_tracker.lock() {
            tracker.track_deallocation(allocation_id);
        }
    }

    /// Profile layer latency with detailed breakdown
    pub fn profile_layer_latency(&mut self, layer_latency: LayerLatencyProfile) {
        self.layer_latency_profiles
            .insert(layer_latency.layer_name.clone(), layer_latency);
    }

    /// Start I/O operation profiling
    pub fn start_io_profiling(
        &mut self,
        operation_type: IoOperationType,
        bytes_expected: usize,
    ) -> Uuid {
        self.io_monitor.start_io_operation(operation_type, bytes_expected)
    }

    /// Finish I/O operation profiling
    pub fn finish_io_profiling(&mut self, operation_id: Uuid, bytes_transferred: usize) {
        if let Some(profile) = self.io_monitor.finish_io_operation(operation_id, bytes_transferred)
        {
            self.io_profiles.push(profile);
        }
    }

    /// Analyze CPU bottlenecks
    pub fn analyze_cpu_bottlenecks(&mut self) -> Vec<CpuBottleneckAnalysis> {
        // Simplified CPU bottleneck analysis
        // In practice, this would use system profiling APIs
        let analysis = CpuBottleneckAnalysis {
            thread_id: 0, // Use 0 as placeholder since thread::current().id().as_u64() is unstable
            cpu_usage: 0.75, // Simplified
            context_switches: 1000,
            cache_misses: 500,
            instructions_per_cycle: 2.5,
            branch_mispredictions: 100,
            hot_functions: vec![
                HotFunction {
                    function_name: "tensor_multiply".to_string(),
                    self_time_percentage: 25.0,
                    call_count: 1000,
                    avg_time_per_call: Duration::from_micros(250),
                },
                HotFunction {
                    function_name: "gradient_computation".to_string(),
                    self_time_percentage: 20.0,
                    call_count: 500,
                    avg_time_per_call: Duration::from_micros(400),
                },
            ],
            bottleneck_score: 0.6,
        };

        self.cpu_bottleneck_analysis.push(analysis.clone());
        vec![analysis]
    }

    /// Get memory allocation statistics
    pub fn get_memory_stats(&self) -> Option<MemoryStats> {
        if let Ok(tracker) = self.memory_tracker.lock() {
            Some(tracker.get_memory_stats())
        } else {
            None
        }
    }

    /// Get GPU utilization metrics
    pub fn get_gpu_utilization(&self, device_id: i32) -> Option<f64> {
        self.gpu_profiler
            .as_ref()
            .map(|profiler| profiler.get_gpu_utilization(device_id))
    }

    /// Get I/O bandwidth statistics
    pub fn get_io_bandwidth_stats(&self) -> HashMap<IoDeviceType, f64> {
        let mut stats = HashMap::new();

        stats.insert(
            IoDeviceType::SSD,
            self.io_monitor.get_average_bandwidth(&IoDeviceType::SSD),
        );
        stats.insert(
            IoDeviceType::HDD,
            self.io_monitor.get_average_bandwidth(&IoDeviceType::HDD),
        );
        stats.insert(
            IoDeviceType::Network,
            self.io_monitor.get_average_bandwidth(&IoDeviceType::Network),
        );
        stats.insert(
            IoDeviceType::Memory,
            self.io_monitor.get_average_bandwidth(&IoDeviceType::Memory),
        );
        stats.insert(
            IoDeviceType::Cache,
            self.io_monitor.get_average_bandwidth(&IoDeviceType::Cache),
        );

        stats
    }

    /// Get layer latency analysis
    pub fn get_layer_latency_analysis(&self) -> Vec<LayerLatencyAnalysis> {
        self.layer_latency_profiles
            .values()
            .map(|profile| LayerLatencyAnalysis {
                layer_name: profile.layer_name.clone(),
                layer_type: profile.layer_type.clone(),
                total_time: profile.cpu_time
                    + profile.gpu_time
                    + profile.memory_copy_time
                    + profile.sync_time,
                cpu_percentage: profile.cpu_time.as_secs_f64()
                    / (profile.cpu_time
                        + profile.gpu_time
                        + profile.memory_copy_time
                        + profile.sync_time)
                        .as_secs_f64()
                    * 100.0,
                gpu_percentage: profile.gpu_time.as_secs_f64()
                    / (profile.cpu_time
                        + profile.gpu_time
                        + profile.memory_copy_time
                        + profile.sync_time)
                        .as_secs_f64()
                    * 100.0,
                memory_copy_percentage: profile.memory_copy_time.as_secs_f64()
                    / (profile.cpu_time
                        + profile.gpu_time
                        + profile.memory_copy_time
                        + profile.sync_time)
                        .as_secs_f64()
                    * 100.0,
                flops_per_second: if profile.gpu_time.as_secs_f64() > 0.0 {
                    profile.flops as f64 / profile.gpu_time.as_secs_f64()
                } else {
                    0.0
                },
                memory_bandwidth_utilization: profile.cache_hit_rate,
                bottleneck_type: self.identify_layer_bottleneck(profile),
            })
            .collect()
    }

    /// Get comprehensive performance analysis
    pub fn get_performance_analysis(&self) -> PerformanceAnalysis {
        let memory_stats = self.get_memory_stats();
        let io_bandwidth_stats = self.get_io_bandwidth_stats();
        let layer_analysis = self.get_layer_latency_analysis();

        let gpu_utilization =
            self.gpu_profiler.as_ref().map(|profiler| profiler.get_gpu_utilization(0));

        PerformanceAnalysis {
            memory_stats,
            io_bandwidth_stats,
            layer_analysis,
            gpu_utilization,
            cpu_bottlenecks: self.cpu_bottleneck_analysis.clone(),
            total_gpu_kernels: self.gpu_kernel_profiles.len(),
            total_io_operations: self.io_profiles.len(),
            performance_score: self.calculate_overall_performance_score(),
            recommendations: self.generate_enhanced_recommendations(),
        }
    }

    fn identify_layer_bottleneck(&self, profile: &LayerLatencyProfile) -> String {
        let total_time =
            profile.cpu_time + profile.gpu_time + profile.memory_copy_time + profile.sync_time;

        if profile.memory_copy_time > total_time / 2 {
            "Memory Bandwidth".to_string()
        } else if profile.sync_time > total_time / 3 {
            "GPU Synchronization".to_string()
        } else if profile.gpu_time > profile.cpu_time * 10 {
            "GPU Compute".to_string()
        } else {
            "CPU Compute".to_string()
        }
    }

    fn calculate_overall_performance_score(&self) -> f64 {
        let mut score: f64 = 100.0;

        // Deduct for bottlenecks
        for bottleneck in &self.bottlenecks {
            match bottleneck.severity {
                BottleneckSeverity::Critical => score -= 20.0,
                BottleneckSeverity::High => score -= 10.0,
                BottleneckSeverity::Medium => score -= 5.0,
                BottleneckSeverity::Low => score -= 2.0,
            }
        }

        // Deduct for poor GPU utilization
        if let Some(gpu_util) = self.get_gpu_utilization(0) {
            if gpu_util < 0.5 {
                score -= 15.0;
            } else if gpu_util < 0.7 {
                score -= 8.0;
            }
        }

        // Deduct for memory inefficiency
        if let Some(memory_stats) = self.get_memory_stats() {
            if memory_stats.memory_efficiency < 0.8 {
                score -= 10.0;
            }
        }

        score.max(0.0)
    }

    fn generate_enhanced_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // GPU utilization recommendations
        if let Some(gpu_util) = self.get_gpu_utilization(0) {
            if gpu_util < 0.5 {
                recommendations.push("Low GPU utilization detected. Consider increasing batch size or optimizing GPU kernels.".to_string());
            }
        }

        // Memory recommendations
        if let Some(memory_stats) = self.get_memory_stats() {
            if memory_stats.memory_efficiency < 0.8 {
                recommendations.push("Memory allocation efficiency is low. Consider memory pooling or reducing allocations.".to_string());
            }

            if memory_stats.active_allocations > 10000 {
                recommendations.push("High number of active memory allocations. Consider batch allocation strategies.".to_string());
            }
        }

        // I/O recommendations
        let io_stats = self.get_io_bandwidth_stats();
        if let Some(&ssd_bandwidth) = io_stats.get(&IoDeviceType::SSD) {
            if ssd_bandwidth < 100.0 {
                // Less than 100 MB/s
                recommendations.push(
                    "Low SSD bandwidth utilization. Consider optimizing file I/O patterns."
                        .to_string(),
                );
            }
        }

        // Layer-specific recommendations
        let layer_analysis = self.get_layer_latency_analysis();
        for analysis in &layer_analysis {
            if analysis.memory_copy_percentage > 50.0 {
                recommendations.push(format!(
                    "Layer '{}' is memory bandwidth bound. Consider data layout optimization.",
                    analysis.layer_name
                ));
            }

            if analysis.cpu_percentage > 80.0 {
                recommendations.push(format!(
                    "Layer '{}' is CPU bound. Consider GPU acceleration.",
                    analysis.layer_name
                ));
            }
        }

        if recommendations.is_empty() {
            recommendations
                .push("Performance appears optimal based on current analysis.".to_string());
        }

        recommendations
    }

    // Private analysis methods

    fn analyze_layer_bottlenecks(&mut self) {
        for (layer_name, profile) in &self.layer_profiles {
            if profile.forward_times.is_empty() {
                continue;
            }

            let avg_forward_time =
                profile.forward_times.iter().sum::<Duration>() / profile.forward_times.len() as u32;

            // Consider a layer slow if it takes more than 100ms on average
            if avg_forward_time.as_millis() > 100 {
                let mut metrics = HashMap::new();
                metrics.insert(
                    "avg_forward_time_ms".to_string(),
                    avg_forward_time.as_millis() as f64,
                );
                metrics.insert("call_count".to_string(), profile.call_count as f64);

                self.bottlenecks.push(PerformanceBottleneck {
                    bottleneck_type: BottleneckType::ModelComputation,
                    location: layer_name.clone(),
                    severity: if avg_forward_time.as_millis() > 500 {
                        BottleneckSeverity::High
                    } else {
                        BottleneckSeverity::Medium
                    },
                    description: format!(
                        "Layer '{}' has slow forward pass: {:.1}ms average",
                        layer_name,
                        avg_forward_time.as_millis()
                    ),
                    suggestion: "Consider optimizing layer implementation or reducing layer size"
                        .to_string(),
                    metrics,
                });
            }
        }
    }

    fn analyze_memory_bottlenecks(&mut self) {
        if self.memory_snapshots.len() < 2 {
            return;
        }

        // Check for memory growth trend
        let recent_snapshots = if self.memory_snapshots.len() > 10 {
            &self.memory_snapshots[self.memory_snapshots.len() - 10..]
        } else {
            &self.memory_snapshots
        };

        if recent_snapshots.len() >= 5 {
            let initial_memory = recent_snapshots[0].heap_allocated;
            let final_memory = recent_snapshots
                .last()
                .expect("recent_snapshots has at least 5 elements")
                .heap_allocated;

            if final_memory > initial_memory * 2 {
                let mut metrics = HashMap::new();
                metrics.insert(
                    "initial_memory_mb".to_string(),
                    initial_memory as f64 / (1024.0 * 1024.0),
                );
                metrics.insert(
                    "final_memory_mb".to_string(),
                    final_memory as f64 / (1024.0 * 1024.0),
                );
                metrics.insert(
                    "growth_ratio".to_string(),
                    final_memory as f64 / initial_memory as f64,
                );

                self.bottlenecks.push(PerformanceBottleneck {
                    bottleneck_type: BottleneckType::MemoryBound,
                    location: "Memory Usage".to_string(),
                    severity: BottleneckSeverity::High,
                    description: "Significant memory growth detected during profiling".to_string(),
                    suggestion: "Check for memory leaks or inefficient memory usage patterns"
                        .to_string(),
                    metrics,
                });
            }
        }
    }

    fn analyze_tensor_bottlenecks(&mut self) {
        // Group tensor operations by type
        let mut operation_groups: HashMap<String, Vec<Duration>> = HashMap::new();

        for event in &self.events {
            if let ProfileEvent::TensorOperation {
                operation,
                duration,
                ..
            } = event
            {
                operation_groups.entry(operation.clone()).or_default().push(*duration);
            }
        }

        // Find slow operations
        for (operation, durations) in operation_groups {
            if durations.is_empty() {
                continue;
            }

            let avg_duration = durations.iter().sum::<Duration>() / durations.len() as u32;
            let total_time = durations.iter().sum::<Duration>();

            // Consider operation slow if it takes more than 10ms on average
            if avg_duration.as_millis() > 10 {
                let mut metrics = HashMap::new();
                metrics.insert(
                    "avg_duration_ms".to_string(),
                    avg_duration.as_millis() as f64,
                );
                metrics.insert("total_time_ms".to_string(), total_time.as_millis() as f64);
                metrics.insert("call_count".to_string(), durations.len() as f64);

                self.bottlenecks.push(PerformanceBottleneck {
                    bottleneck_type: BottleneckType::CpuBound,
                    location: format!("Tensor Operation: {}", operation),
                    severity: if avg_duration.as_millis() > 50 {
                        BottleneckSeverity::High
                    } else {
                        BottleneckSeverity::Medium
                    },
                    description: format!(
                        "Tensor operation '{}' is slow: {:.1}ms average",
                        operation,
                        avg_duration.as_millis()
                    ),
                    suggestion:
                        "Consider optimizing tensor operation or using different data types"
                            .to_string(),
                    metrics,
                });
            }
        }
    }

    fn get_slowest_layers(&self, limit: usize) -> Vec<(String, Duration)> {
        let mut layer_times: Vec<(String, Duration)> = self
            .layer_profiles
            .iter()
            .map(|(name, profile)| {
                let avg_time = if profile.forward_times.is_empty() {
                    Duration::ZERO
                } else {
                    profile.forward_times.iter().sum::<Duration>()
                        / profile.forward_times.len() as u32
                };
                (name.clone(), avg_time)
            })
            .collect();

        layer_times.sort_by_key(|item| std::cmp::Reverse(item.1));
        layer_times.truncate(limit);
        layer_times
    }

    fn analyze_memory_efficiency(&self) -> MemoryEfficiencyAnalysis {
        if self.memory_snapshots.is_empty() {
            return MemoryEfficiencyAnalysis::default();
        }

        let memory_values: Vec<usize> =
            self.memory_snapshots.iter().map(|snapshot| snapshot.heap_allocated).collect();

        let max_memory = memory_values.iter().max().copied().unwrap_or(0);
        let min_memory = memory_values.iter().min().copied().unwrap_or(0);
        let avg_memory = memory_values.iter().sum::<usize>() / memory_values.len();

        MemoryEfficiencyAnalysis {
            peak_memory_mb: max_memory as f64 / (1024.0 * 1024.0),
            min_memory_mb: min_memory as f64 / (1024.0 * 1024.0),
            avg_memory_mb: avg_memory as f64 / (1024.0 * 1024.0),
            memory_variance: self.calculate_memory_variance(&memory_values, avg_memory),
            efficiency_score: self.calculate_memory_efficiency_score(&memory_values),
        }
    }

    fn calculate_memory_variance(&self, values: &[usize], mean: usize) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let variance_sum: f64 = values
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean as f64;
                diff * diff
            })
            .sum();

        variance_sum / (values.len() - 1) as f64
    }

    fn calculate_memory_efficiency_score(&self, values: &[usize]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let max_memory = values.iter().max().copied().unwrap_or(0);
        let min_memory = values.iter().min().copied().unwrap_or(0);

        if max_memory == 0 {
            return 100.0;
        }

        // Efficiency score: closer to 100% means more stable memory usage
        100.0 * (1.0 - (max_memory - min_memory) as f64 / max_memory as f64)
    }

    fn generate_performance_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Analyze bottlenecks for recommendations
        for bottleneck in &self.bottlenecks {
            match bottleneck.bottleneck_type {
                BottleneckType::ModelComputation => {
                    recommendations.push(
                        "Consider model architecture optimizations or layer fusion".to_string(),
                    );
                },
                BottleneckType::MemoryBound => {
                    recommendations.push(
                        "Optimize memory usage with gradient checkpointing or model parallelism"
                            .to_string(),
                    );
                },
                BottleneckType::CpuBound => {
                    recommendations.push(
                        "Consider GPU acceleration or optimized CPU implementations".to_string(),
                    );
                },
                _ => {},
            }
        }

        // General recommendations based on profiling data
        if self.events.len() > 10000 {
            recommendations.push(
                "High number of profiling events - consider reducing profiling overhead"
                    .to_string(),
            );
        }

        let stats = self.get_statistics();
        if let Some(layer_stats) = stats.get("LayerExecution") {
            if layer_stats.avg_duration.as_millis() > 50 {
                recommendations.push(
                    "Average layer execution time is high - consider layer optimization"
                        .to_string(),
                );
            }
        }

        if recommendations.is_empty() {
            recommendations
                .push("Performance appears optimal based on current profiling data".to_string());
        }

        recommendations
    }

    /// Generate enhanced profiler report with advanced metrics
    pub async fn generate_enhanced_report(&self) -> Result<EnhancedProfilerReport> {
        let basic_report = self.generate_report().await?;
        let performance_analysis = self.get_performance_analysis();

        let gpu_kernel_summary = self.generate_gpu_kernel_summary();
        let memory_allocation_summary = self.generate_memory_allocation_summary();
        let io_performance_summary = self.generate_io_performance_summary();

        Ok(EnhancedProfilerReport {
            basic_report,
            performance_analysis,
            gpu_kernel_summary,
            memory_allocation_summary,
            io_performance_summary,
        })
    }

    fn generate_gpu_kernel_summary(&self) -> GpuKernelSummary {
        let total_kernels = self.gpu_kernel_profiles.len();
        let total_execution_time = self.gpu_kernel_profiles.iter().map(|k| k.execution_time).sum();

        let avg_occupancy = if total_kernels > 0 {
            self.gpu_kernel_profiles.iter().map(|k| k.occupancy).sum::<f64>() / total_kernels as f64
        } else {
            0.0
        };

        let avg_compute_utilization = if total_kernels > 0 {
            self.gpu_kernel_profiles.iter().map(|k| k.compute_utilization).sum::<f64>()
                / total_kernels as f64
        } else {
            0.0
        };

        let mut kernels_by_time: Vec<_> = self
            .gpu_kernel_profiles
            .iter()
            .map(|k| (k.kernel_name.clone(), k.execution_time))
            .collect();
        kernels_by_time.sort_by_key(|item| std::cmp::Reverse(item.1));

        let slowest_kernels = kernels_by_time.into_iter().take(5).map(|(name, _)| name).collect();

        GpuKernelSummary {
            total_kernels,
            total_execution_time,
            avg_occupancy,
            avg_compute_utilization,
            slowest_kernels,
        }
    }

    fn generate_memory_allocation_summary(&self) -> MemoryAllocationSummary {
        let total_allocations = self.memory_allocations.len();
        let peak_memory_usage =
            self.memory_allocations.values().map(|a| a.size_bytes).max().unwrap_or(0);

        let memory_efficiency = if let Some(stats) = self.get_memory_stats() {
            stats.memory_efficiency
        } else {
            1.0
        };

        let mut allocations_by_size: Vec<_> = self
            .memory_allocations
            .values()
            .map(|a| (format!("{} bytes", a.size_bytes), a.size_bytes))
            .collect();
        allocations_by_size.sort_by_key(|item| std::cmp::Reverse(item.1));

        let largest_allocations =
            allocations_by_size.into_iter().take(5).map(|(desc, _)| desc).collect();

        let memory_leaks = self.memory_allocations.values().filter(|a| !a.freed).count();

        MemoryAllocationSummary {
            total_allocations,
            peak_memory_usage,
            memory_efficiency,
            largest_allocations,
            memory_leaks,
        }
    }

    fn generate_io_performance_summary(&self) -> IoPerformanceSummary {
        let total_operations = self.io_profiles.len();
        let total_bytes_transferred = self.io_profiles.iter().map(|io| io.bytes_transferred).sum();

        let avg_bandwidth_by_device = self.get_io_bandwidth_stats();

        let mut operations_by_duration: Vec<_> = self
            .io_profiles
            .iter()
            .map(|io| {
                (
                    format!("{:?}: {} bytes", io.operation_type, io.bytes_transferred),
                    io.duration,
                )
            })
            .collect();
        operations_by_duration.sort_by_key(|item| std::cmp::Reverse(item.1));

        let slowest_operations =
            operations_by_duration.into_iter().take(5).map(|(desc, _)| desc).collect();

        IoPerformanceSummary {
            total_operations,
            total_bytes_transferred,
            avg_bandwidth_by_device,
            slowest_operations,
        }
    }
}

/// Scoped timer for automatic timing
pub struct ScopedTimer<'a> {
    profiler: &'a mut Profiler,
    name: String,
}

impl<'a> ScopedTimer<'a> {
    pub fn new(profiler: &'a mut Profiler, name: String) -> Self {
        profiler.start_timer(&name);
        Self { profiler, name }
    }
}

impl<'a> Drop for ScopedTimer<'a> {
    fn drop(&mut self) {
        self.profiler.end_timer(&self.name);
    }
}

/// Macro for convenient timing
#[macro_export]
macro_rules! profile_scope {
    ($profiler:expr, $name:expr) => {
        let _timer = ScopedTimer::new($profiler, $name.to_string());
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> DebugConfig {
        DebugConfig::default()
    }

    // --- Profiler tests ---

    #[test]
    fn test_profiler_new() {
        let config = make_config();
        let profiler = Profiler::new(&config);
        assert!(profiler.events.is_empty());
        assert!(profiler.active_timers.is_empty());
        assert!(profiler.start_time.is_none());
    }

    #[test]
    fn test_profiler_start_end_timer() {
        let config = make_config();
        let mut profiler = Profiler::new(&config);
        profiler.start_timer("test_op");
        let duration = profiler.end_timer("test_op");
        assert!(duration.is_some());
        assert_eq!(profiler.events.len(), 1);
    }

    #[test]
    fn test_profiler_end_timer_not_started() {
        let config = make_config();
        let mut profiler = Profiler::new(&config);
        let duration = profiler.end_timer("nonexistent");
        assert!(duration.is_none());
    }

    #[test]
    fn test_profiler_record_layer_execution() {
        let config = make_config();
        let mut profiler = Profiler::new(&config);
        profiler.record_layer_execution(
            "attention",
            "self_attention",
            Duration::from_millis(50),
            Some(Duration::from_millis(30)),
            1024,
            1000,
        );
        assert_eq!(profiler.events.len(), 1);
        let profiles = profiler.get_layer_profiles();
        assert!(profiles.contains_key("attention"));
        let lp = &profiles["attention"];
        assert_eq!(lp.call_count(), 1);
        assert_eq!(lp.forward_times().len(), 1);
        assert_eq!(lp.backward_times().len(), 1);
    }

    #[test]
    fn test_profiler_record_tensor_operation() {
        let config = make_config();
        let mut profiler = Profiler::new(&config);
        profiler.record_tensor_operation("matmul", &[64, 128], Duration::from_micros(200), 8192);
        assert_eq!(profiler.events.len(), 1);
    }

    #[test]
    fn test_profiler_record_model_inference() {
        let config = make_config();
        let mut profiler = Profiler::new(&config);
        profiler.record_model_inference(32, 512, Duration::from_millis(100));
        assert_eq!(profiler.events.len(), 1);
    }

    #[test]
    fn test_profiler_record_gradient_computation() {
        let config = make_config();
        let mut profiler = Profiler::new(&config);
        profiler.record_gradient_computation("fc1", 0.5, Duration::from_millis(10));
        assert_eq!(profiler.events.len(), 1);
    }

    #[test]
    fn test_profiler_get_statistics_empty() {
        let config = make_config();
        let profiler = Profiler::new(&config);
        let stats = profiler.get_statistics();
        assert!(stats.is_empty());
    }

    #[test]
    fn test_profiler_get_statistics_with_events() {
        let config = make_config();
        let mut profiler = Profiler::new(&config);
        profiler.record_model_inference(8, 256, Duration::from_millis(50));
        profiler.record_model_inference(8, 256, Duration::from_millis(100));
        let stats = profiler.get_statistics();
        assert!(stats.contains_key("ModelInference"));
        let mi_stats = &stats["ModelInference"];
        assert_eq!(mi_stats.count, 2);
    }

    #[test]
    fn test_profiler_clear() {
        let config = make_config();
        let mut profiler = Profiler::new(&config);
        profiler.start_timer("op1");
        profiler.end_timer("op1");
        profiler.take_memory_snapshot();
        profiler.clear();
        assert!(profiler.events.is_empty());
        assert!(profiler.active_timers.is_empty());
        assert!(profiler.memory_snapshots.is_empty());
        assert!(profiler.start_time.is_none());
    }

    #[test]
    fn test_profiler_take_memory_snapshot() {
        let config = make_config();
        let mut profiler = Profiler::new(&config);
        profiler.take_memory_snapshot();
        assert_eq!(profiler.get_memory_timeline().len(), 1);
    }

    #[test]
    fn test_profiler_memory_snapshot_limit() {
        let config = make_config();
        let mut profiler = Profiler::new(&config);
        for _ in 0..1100 {
            profiler.take_memory_snapshot();
        }
        // Should trim to ~500 after exceeding 1000
        assert!(profiler.get_memory_timeline().len() <= 601);
    }

    #[test]
    fn test_profiler_analyze_performance_empty() {
        let config = make_config();
        let mut profiler = Profiler::new(&config);
        let bottlenecks = profiler.analyze_performance();
        assert!(bottlenecks.is_empty());
    }

    #[test]
    fn test_profiler_analyze_performance_slow_layer() {
        let config = make_config();
        let mut profiler = Profiler::new(&config);
        for _ in 0..5 {
            profiler.record_layer_execution(
                "slow_layer",
                "dense",
                Duration::from_millis(600),
                None,
                4096,
                10000,
            );
        }
        let bottlenecks = profiler.analyze_performance();
        assert!(!bottlenecks.is_empty());
    }

    #[test]
    fn test_profiler_get_slowest_layers() {
        let config = make_config();
        let mut profiler = Profiler::new(&config);
        profiler.record_layer_execution(
            "fast_layer",
            "relu",
            Duration::from_millis(1),
            None,
            128,
            0,
        );
        profiler.record_layer_execution(
            "slow_layer",
            "dense",
            Duration::from_millis(200),
            None,
            4096,
            10000,
        );
        let slowest = profiler.get_slowest_layers(2);
        assert_eq!(slowest.len(), 2);
        assert_eq!(slowest[0].0, "slow_layer");
    }

    #[test]
    fn test_profiler_memory_efficiency_empty() {
        let config = make_config();
        let profiler = Profiler::new(&config);
        let analysis = profiler.analyze_memory_efficiency();
        assert!((analysis.efficiency_score - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_profiler_calculate_memory_variance() {
        let config = make_config();
        let profiler = Profiler::new(&config);
        let values = vec![100, 200, 300];
        let mean = 200;
        let variance = profiler.calculate_memory_variance(&values, mean);
        // variance = ((100-200)^2 + (200-200)^2 + (300-200)^2) / 2 = 10000
        assert!((variance - 10000.0).abs() < 1e-3);
    }

    #[test]
    fn test_profiler_calculate_memory_efficiency_score_empty() {
        let config = make_config();
        let profiler = Profiler::new(&config);
        let score = profiler.calculate_memory_efficiency_score(&[]);
        assert!((score - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_profiler_calculate_memory_efficiency_score_stable() {
        let config = make_config();
        let profiler = Profiler::new(&config);
        let values = vec![100, 100, 100];
        let score = profiler.calculate_memory_efficiency_score(&values);
        assert!((score - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_profiler_calculate_memory_efficiency_score_varied() {
        let config = make_config();
        let profiler = Profiler::new(&config);
        let values = vec![50, 100];
        let score = profiler.calculate_memory_efficiency_score(&values);
        // 100 * (1.0 - (100-50)/100) = 50.0
        assert!((score - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_profiler_overall_performance_score_no_bottlenecks() {
        let config = make_config();
        let profiler = Profiler::new(&config);
        let score = profiler.calculate_overall_performance_score();
        // Score starts at 100 but GPU utilization of 0.0 deducts 15 points
        assert!(score >= 50.0);
        assert!(score <= 100.0);
    }

    #[test]
    fn test_profiler_identify_layer_bottleneck_memory() {
        let config = make_config();
        let profiler = Profiler::new(&config);
        let profile = LayerLatencyProfile {
            layer_name: "test".to_string(),
            layer_type: "dense".to_string(),
            input_shapes: vec![vec![32, 128]],
            output_shapes: vec![vec![32, 256]],
            cpu_time: Duration::from_millis(10),
            gpu_time: Duration::from_millis(10),
            memory_copy_time: Duration::from_millis(100),
            sync_time: Duration::from_millis(5),
            parameter_count: 1000,
            flops: 100000,
            memory_footprint_bytes: 4096,
            cache_hit_rate: 0.5,
        };
        let bottleneck = profiler.identify_layer_bottleneck(&profile);
        assert_eq!(bottleneck, "Memory Bandwidth");
    }

    #[test]
    fn test_profiler_identify_layer_bottleneck_sync() {
        let config = make_config();
        let profiler = Profiler::new(&config);
        let profile = LayerLatencyProfile {
            layer_name: "test".to_string(),
            layer_type: "dense".to_string(),
            input_shapes: vec![],
            output_shapes: vec![],
            cpu_time: Duration::from_millis(10),
            gpu_time: Duration::from_millis(10),
            memory_copy_time: Duration::from_millis(5),
            sync_time: Duration::from_millis(50),
            parameter_count: 0,
            flops: 0,
            memory_footprint_bytes: 0,
            cache_hit_rate: 0.0,
        };
        let bottleneck = profiler.identify_layer_bottleneck(&profile);
        assert_eq!(bottleneck, "GPU Synchronization");
    }

    #[test]
    fn test_profiler_io_bandwidth_stats_empty() {
        let config = make_config();
        let profiler = Profiler::new(&config);
        let stats = profiler.get_io_bandwidth_stats();
        assert_eq!(stats.len(), 5);
        for &val in stats.values() {
            assert!((val - 0.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_profiler_track_memory_allocation_and_deallocation() {
        let config = make_config();
        let mut profiler = Profiler::new(&config);
        let alloc_id = profiler.track_memory_allocation(
            4096,
            MemoryAllocationType::Host,
            None,
            vec!["test_frame".to_string()],
        );
        assert!(profiler.memory_allocations.contains_key(&alloc_id));
        profiler.track_memory_deallocation(alloc_id);
        let alloc = profiler.memory_allocations.get(&alloc_id);
        assert!(alloc.is_some());
        assert!(alloc.expect("allocation should exist").freed);
    }

    #[test]
    fn test_profiler_gpu_kernel_summary_empty() {
        let config = make_config();
        let profiler = Profiler::new(&config);
        let summary = profiler.generate_gpu_kernel_summary();
        assert_eq!(summary.total_kernels, 0);
        assert!((summary.avg_occupancy - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_profiler_memory_allocation_summary() {
        let config = make_config();
        let mut profiler = Profiler::new(&config);
        let _id = profiler.track_memory_allocation(
            1024,
            MemoryAllocationType::Device,
            Some(0),
            Vec::new(),
        );
        let summary = profiler.generate_memory_allocation_summary();
        assert_eq!(summary.total_allocations, 1);
        assert_eq!(summary.peak_memory_usage, 1024);
        assert_eq!(summary.memory_leaks, 1);
    }

    #[test]
    fn test_profiler_io_performance_summary_empty() {
        let config = make_config();
        let profiler = Profiler::new(&config);
        let summary = profiler.generate_io_performance_summary();
        assert_eq!(summary.total_operations, 0);
        assert_eq!(summary.total_bytes_transferred, 0);
    }

    #[test]
    fn test_profiler_analyze_cpu_bottlenecks() {
        let config = make_config();
        let mut profiler = Profiler::new(&config);
        let result = profiler.analyze_cpu_bottlenecks();
        assert!(!result.is_empty());
        assert_eq!(result[0].hot_functions.len(), 2);
    }

    #[test]
    fn test_profiler_performance_analysis() {
        let config = make_config();
        let profiler = Profiler::new(&config);
        let analysis = profiler.get_performance_analysis();
        assert!(analysis.performance_score > 0.0);
        assert!(!analysis.recommendations.is_empty());
    }

    #[test]
    fn test_profiler_generate_performance_recommendations_optimal() {
        let config = make_config();
        let profiler = Profiler::new(&config);
        let recs = profiler.generate_performance_recommendations();
        assert!(!recs.is_empty());
        assert!(recs[0].contains("optimal"));
    }

    #[test]
    fn test_layer_profile_accessors() {
        let config = make_config();
        let mut profiler = Profiler::new(&config);
        profiler.record_layer_execution(
            "layer1",
            "conv",
            Duration::from_millis(10),
            Some(Duration::from_millis(5)),
            512,
            100,
        );
        let profiles = profiler.get_layer_profiles();
        let lp = &profiles["layer1"];
        assert_eq!(lp.forward_times().len(), 1);
        assert_eq!(lp.backward_times().len(), 1);
        assert_eq!(lp.memory_usage(), &vec![512]);
        assert_eq!(lp.call_count(), 1);
    }
}
