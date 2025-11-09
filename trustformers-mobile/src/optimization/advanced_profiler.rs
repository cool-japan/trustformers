//! Advanced Performance Profiler for Mobile AI Optimization
//!
//! This module provides comprehensive performance profiling and monitoring
//! capabilities specifically designed for mobile AI inference optimization.

use crate::{MobileBackend, MobileConfig, MobilePlatform};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use trustformers_core::error::{CoreError, Result};
use trustformers_core::TrustformersError;

/// Configuration for advanced performance profiling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedProfilerConfig {
    /// Enable detailed memory tracking
    pub enable_memory_tracking: bool,
    /// Enable thermal monitoring
    pub enable_thermal_monitoring: bool,
    /// Enable power consumption tracking
    pub enable_power_tracking: bool,
    /// Enable operation-level profiling
    pub enable_operation_profiling: bool,
    /// Enable real-time visualization
    pub enable_real_time_viz: bool,
    /// Sampling interval in milliseconds
    pub sampling_interval_ms: u64,
    /// Maximum history length for rolling metrics
    pub max_history_length: usize,
    /// Profiling output format
    pub output_format: ProfilerOutputFormat,
    /// Enable GPU profiling (if available)
    pub enable_gpu_profiling: bool,
    /// Enable network usage tracking
    pub enable_network_tracking: bool,
}

impl Default for AdvancedProfilerConfig {
    fn default() -> Self {
        Self {
            enable_memory_tracking: true,
            enable_thermal_monitoring: true,
            enable_power_tracking: true,
            enable_operation_profiling: true,
            enable_real_time_viz: false,
            sampling_interval_ms: 100, // 100ms sampling
            max_history_length: 1000,  // Keep 1000 samples
            output_format: ProfilerOutputFormat::Json,
            enable_gpu_profiling: true,
            enable_network_tracking: false,
        }
    }
}

/// Output format for profiling data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProfilerOutputFormat {
    Json,
    Csv,
    Flamegraph,
    Chrome,
    Custom(String),
}

/// Real-time performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Timestamp of measurement
    pub timestamp: u64,
    /// CPU usage percentage (0-100)
    pub cpu_usage: f32,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Peak memory usage in bytes
    pub peak_memory: u64,
    /// GPU usage percentage (0-100)
    pub gpu_usage: Option<f32>,
    /// GPU memory usage in bytes
    pub gpu_memory: Option<u64>,
    /// Device temperature in Celsius
    pub temperature: Option<f32>,
    /// Battery level percentage (0-100)
    pub battery_level: Option<f32>,
    /// Power consumption in watts
    pub power_consumption: Option<f32>,
    /// Network bytes sent
    pub network_sent_bytes: Option<u64>,
    /// Network bytes received
    pub network_received_bytes: Option<u64>,
    /// Frame rate (FPS) for real-time applications
    pub fps: Option<f32>,
    /// Inference latency in milliseconds
    pub inference_latency_ms: Option<f32>,
    /// Throughput in inferences per second
    pub throughput_ips: Option<f32>,
}

/// Operation-level profiling data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationProfile {
    /// Operation name
    pub operation_name: String,
    /// Operation type
    pub operation_type: OperationType,
    /// Execution time in microseconds
    pub execution_time_us: u64,
    /// Memory allocated during operation
    pub memory_allocated_bytes: u64,
    /// Memory freed during operation
    pub memory_freed_bytes: u64,
    /// Number of FLOPs (floating point operations)
    pub flops: Option<u64>,
    /// Input tensor shapes
    pub input_shapes: Vec<Vec<usize>>,
    /// Output tensor shape
    pub output_shape: Vec<usize>,
    /// GPU kernel execution time (if applicable)
    pub gpu_kernel_time_us: Option<u64>,
    /// Cache hit rate for this operation
    pub cache_hit_rate: Option<f32>,
}

/// Type of operation being profiled
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationType {
    Convolution,
    LinearTransform,
    Attention,
    Normalization,
    Activation,
    Pooling,
    Quantization,
    Dequantization,
    MemoryCopy,
    DataTransfer,
    Custom(String),
}

/// Thermal analysis data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalAnalysis {
    /// Current temperature zones
    pub temperature_zones: HashMap<String, f32>,
    /// Thermal throttling status
    pub is_throttling: bool,
    /// Predicted temperature trend
    pub temperature_trend: TemperatureTrend,
    /// Time until thermal throttling (if predicted)
    pub time_to_throttling_ms: Option<u64>,
    /// Recommended action
    pub recommended_action: ThermalRecommendation,
}

/// Temperature trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemperatureTrend {
    Cooling,
    Stable,
    Rising,
    Critical,
}

/// Thermal management recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThermalRecommendation {
    Continue,
    ReduceFrequency,
    PauseInference,
    SwitchToLowerPrecision,
    EnableThermalManagement,
}

/// Power analysis data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerAnalysis {
    /// Current power consumption in watts
    pub current_power_w: f32,
    /// Average power consumption over window
    pub average_power_w: f32,
    /// Peak power consumption
    pub peak_power_w: f32,
    /// Estimated battery life remaining in minutes
    pub battery_life_remaining_min: Option<f32>,
    /// Power efficiency (inferences per watt)
    pub power_efficiency_ipw: Option<f32>,
    /// Recommended power mode
    pub recommended_power_mode: PowerMode,
}

/// Power management modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PowerMode {
    MaxPerformance,
    Balanced,
    PowerSaver,
    UltraLowPower,
}

/// Memory analysis data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAnalysis {
    /// Current memory usage
    pub current_usage_bytes: u64,
    /// Peak memory usage
    pub peak_usage_bytes: u64,
    /// Available memory
    pub available_bytes: u64,
    /// Memory fragmentation percentage
    pub fragmentation_percent: f32,
    /// Memory allocation patterns
    pub allocation_patterns: Vec<AllocationPattern>,
    /// Memory leak detection
    pub potential_leaks: Vec<MemoryLeak>,
    /// Garbage collection statistics
    pub gc_stats: Option<GCStats>,
}

/// Memory allocation pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationPattern {
    /// Size of allocation
    pub size_bytes: u64,
    /// Frequency of this allocation size
    pub frequency: u32,
    /// Average lifetime of allocations
    pub average_lifetime_ms: f32,
    /// Allocation source (operation type)
    pub source: String,
}

/// Memory leak detection data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLeak {
    /// Size of potential leak
    pub size_bytes: u64,
    /// Duration since allocation
    pub age_ms: u64,
    /// Suspected source
    pub source: String,
    /// Confidence in leak detection (0-1)
    pub confidence: f32,
}

/// Garbage collection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCStats {
    /// Number of GC cycles
    pub gc_cycles: u32,
    /// Total time spent in GC
    pub total_gc_time_ms: u64,
    /// Average GC pause time
    pub average_pause_ms: f32,
    /// Memory reclaimed by GC
    pub memory_reclaimed_bytes: u64,
}

/// Comprehensive profiling report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingReport {
    /// Report metadata
    pub metadata: ReportMetadata,
    /// System information
    pub system_info: SystemInfo,
    /// Performance summary
    pub performance_summary: PerformanceSummary,
    /// Operation profiles
    pub operation_profiles: Vec<OperationProfile>,
    /// Thermal analysis
    pub thermal_analysis: ThermalAnalysis,
    /// Power analysis
    pub power_analysis: PowerAnalysis,
    /// Memory analysis
    pub memory_analysis: MemoryAnalysis,
    /// Performance metrics timeline
    pub metrics_timeline: Vec<PerformanceMetrics>,
    /// Optimization recommendations
    pub recommendations: Vec<OptimizationRecommendation>,
    /// Bottleneck analysis
    pub bottlenecks: Vec<PerformanceBottleneck>,
}

/// Report metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    /// Profiling session ID
    pub session_id: String,
    /// Start time
    pub start_time: u64,
    /// End time
    pub end_time: u64,
    /// Total profiling duration
    pub duration_ms: u64,
    /// Profiler version
    pub profiler_version: String,
    /// Model information
    pub model_info: Option<String>,
}

/// System information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// Platform type
    pub platform: String,
    /// Device model
    pub device_model: String,
    /// OS version
    pub os_version: String,
    /// CPU architecture
    pub cpu_arch: String,
    /// Total system memory
    pub total_memory_bytes: u64,
    /// GPU information
    pub gpu_info: Option<String>,
    /// Available compute backends
    pub available_backends: Vec<String>,
}

/// Performance summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    /// Average inference latency
    pub avg_inference_latency_ms: f32,
    /// 95th percentile latency
    pub p95_latency_ms: f32,
    /// 99th percentile latency
    pub p99_latency_ms: f32,
    /// Average throughput
    pub avg_throughput_ips: f32,
    /// Peak throughput
    pub peak_throughput_ips: f32,
    /// Average memory usage
    pub avg_memory_usage_mb: f32,
    /// Peak memory usage
    pub peak_memory_usage_mb: f32,
    /// Average power consumption
    pub avg_power_consumption_w: f32,
    /// Total energy consumed
    pub total_energy_consumed_j: f32,
    /// Model efficiency score (0-100)
    pub efficiency_score: f32,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Priority level
    pub priority: RecommendationPriority,
    /// Description
    pub description: String,
    /// Expected improvement
    pub expected_improvement: String,
    /// Implementation difficulty
    pub difficulty: ImplementationDifficulty,
    /// Affected operations
    pub affected_operations: Vec<String>,
}

/// Type of optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    Quantization,
    OperatorFusion,
    MemoryOptimization,
    PowerManagement,
    ThermalManagement,
    ModelCompression,
    BatchSizeOptimization,
    PrecisionTuning,
    CacheOptimization,
    ParallelizationStrategy,
}

/// Priority of recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
}

/// Implementation difficulty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationDifficulty {
    Easy,   // Can be done automatically
    Medium, // Requires configuration changes
    Hard,   // Requires code changes
    Expert, // Requires domain expertise
}

/// Performance bottleneck analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,
    /// Severity (0-100)
    pub severity: f32,
    /// Affected operations
    pub affected_operations: Vec<String>,
    /// Description
    pub description: String,
    /// Potential solutions
    pub solutions: Vec<String>,
    /// Impact on overall performance
    pub performance_impact_percent: f32,
}

/// Type of performance bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    Compute,
    Memory,
    IO,
    Thermal,
    Power,
    Network,
    Synchronization,
}

/// Advanced performance profiler
pub struct AdvancedProfiler {
    config: AdvancedProfilerConfig,
    mobile_config: MobileConfig,
    session_id: String,
    start_time: Instant,
    metrics_history: VecDeque<PerformanceMetrics>,
    operation_profiles: Vec<OperationProfile>,
    current_session: Option<ProfilingSession>,
    baseline_metrics: Option<PerformanceMetrics>,
}

/// Active profiling session
struct ProfilingSession {
    id: String,
    start_time: Instant,
    active_operations: HashMap<String, Instant>,
    memory_tracker: MemoryTracker,
    thermal_monitor: ThermalMonitor,
    power_monitor: PowerMonitor,
}

/// Memory tracking utilities
struct MemoryTracker {
    allocations: HashMap<String, AllocationInfo>,
    peak_usage: u64,
    current_usage: u64,
}

/// Allocation tracking information
struct AllocationInfo {
    size: u64,
    timestamp: Instant,
    source: String,
}

/// Thermal monitoring utilities
struct ThermalMonitor {
    temperature_history: VecDeque<f32>,
    throttling_events: Vec<Instant>,
    baseline_temp: f32,
}

/// Power monitoring utilities
struct PowerMonitor {
    power_readings: VecDeque<f32>,
    baseline_power: f32,
    energy_consumed: f32,
}

impl AdvancedProfiler {
    /// Create a new advanced profiler
    pub fn new(config: AdvancedProfilerConfig, mobile_config: MobileConfig) -> Self {
        let session_id = format!(
            "prof_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );

        Self {
            config,
            mobile_config,
            session_id,
            start_time: Instant::now(),
            metrics_history: VecDeque::with_capacity(1000),
            operation_profiles: Vec::new(),
            current_session: None,
            baseline_metrics: None,
        }
    }

    /// Start a new profiling session
    pub fn start_session(&mut self) -> Result<String> {
        let session_id = format!(
            "session_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );

        let session = ProfilingSession {
            id: session_id.clone(),
            start_time: Instant::now(),
            active_operations: HashMap::new(),
            memory_tracker: MemoryTracker {
                allocations: HashMap::new(),
                peak_usage: 0,
                current_usage: 0,
            },
            thermal_monitor: ThermalMonitor {
                temperature_history: VecDeque::with_capacity(100),
                throttling_events: Vec::new(),
                baseline_temp: self.get_current_temperature(),
            },
            power_monitor: PowerMonitor {
                power_readings: VecDeque::with_capacity(100),
                baseline_power: self.get_current_power_consumption(),
                energy_consumed: 0.0,
            },
        };

        // Capture baseline metrics
        self.baseline_metrics = Some(self.capture_current_metrics()?);
        self.current_session = Some(session);

        Ok(session_id)
    }

    /// Record operation start
    pub fn operation_start(
        &mut self,
        operation_name: &str,
        operation_type: OperationType,
    ) -> Result<()> {
        if let Some(ref mut session) = self.current_session {
            session.active_operations.insert(operation_name.to_string(), Instant::now());
        }
        Ok(())
    }

    /// Record operation end and create profile
    pub fn operation_end(
        &mut self,
        operation_name: &str,
        input_shapes: Vec<Vec<usize>>,
        output_shape: Vec<usize>,
    ) -> Result<()> {
        // Get memory usage before mutable borrow
        let memory_after = self.get_memory_usage();

        if let Some(ref mut session) = self.current_session {
            if let Some(start_time) = session.active_operations.remove(operation_name) {
                let execution_time = start_time.elapsed();

                // Track memory allocation changes
                let memory_before = session.memory_tracker.current_usage;
                let memory_allocated = memory_after.saturating_sub(memory_before);
                let memory_freed = memory_before.saturating_sub(memory_after);

                // Update memory tracker
                session.memory_tracker.current_usage = memory_after;
                if memory_after > session.memory_tracker.peak_usage {
                    session.memory_tracker.peak_usage = memory_after;
                }

                // Create operation profile
                let profile = OperationProfile {
                    operation_name: operation_name.to_string(),
                    operation_type: self.infer_operation_type(operation_name),
                    execution_time_us: execution_time.as_micros() as u64,
                    memory_allocated_bytes: memory_allocated,
                    memory_freed_bytes: memory_freed,
                    flops: self.estimate_flops(&input_shapes, &output_shape),
                    input_shapes,
                    output_shape,
                    gpu_kernel_time_us: self.get_gpu_kernel_time(operation_name, execution_time),
                    cache_hit_rate: self.estimate_cache_hit_rate(operation_name),
                };

                self.operation_profiles.push(profile);
            }
        }
        Ok(())
    }

    /// Capture current system metrics
    pub fn capture_metrics(&mut self) -> Result<PerformanceMetrics> {
        let metrics = self.capture_current_metrics()?;

        // Add to history
        self.metrics_history.push_back(metrics.clone());

        // Maintain history size limit
        while self.metrics_history.len() > self.config.max_history_length {
            self.metrics_history.pop_front();
        }

        Ok(metrics)
    }

    /// Generate comprehensive profiling report
    pub fn generate_report(&self) -> Result<ProfilingReport> {
        let end_time = Instant::now();
        let duration = end_time.duration_since(self.start_time);

        let report = ProfilingReport {
            metadata: ReportMetadata {
                session_id: self.session_id.clone(),
                start_time: self.start_time.elapsed().as_millis() as u64,
                end_time: end_time.elapsed().as_millis() as u64,
                duration_ms: duration.as_millis() as u64,
                profiler_version: "1.0.0".to_string(),
                model_info: None,
            },
            system_info: self.get_system_info(),
            performance_summary: self.calculate_performance_summary(),
            operation_profiles: self.operation_profiles.clone(),
            thermal_analysis: self.analyze_thermal_performance(),
            power_analysis: self.analyze_power_consumption(),
            memory_analysis: self.analyze_memory_usage(),
            metrics_timeline: self.metrics_history.clone().into(),
            recommendations: self.generate_recommendations(),
            bottlenecks: self.identify_bottlenecks(),
        };

        Ok(report)
    }

    /// Export report in specified format
    pub fn export_report(
        &self,
        report: &ProfilingReport,
        format: ProfilerOutputFormat,
    ) -> Result<String> {
        match format {
            ProfilerOutputFormat::Json => serde_json::to_string_pretty(report)
                .map_err(|e| TrustformersError::serialization_error(e.to_string()).into()),
            ProfilerOutputFormat::Csv => {
                // Convert to CSV format
                self.export_csv_report(report)
            },
            ProfilerOutputFormat::Flamegraph => {
                // Generate flamegraph data
                self.export_flamegraph(report)
            },
            ProfilerOutputFormat::Chrome => {
                // Export Chrome DevTools format
                self.export_chrome_format(report)
            },
            ProfilerOutputFormat::Custom(format_name) => Err(TrustformersError::invalid_input(
                format!("Unsupported format: {}", format_name),
            )
            .into()),
        }
    }

    /// Get real-time performance recommendations
    pub fn get_realtime_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        if let Some(current_metrics) = self.metrics_history.back() {
            // Memory pressure check
            if current_metrics.memory_usage > (4 * 1024 * 1024 * 1024) {
                // > 4GB
                recommendations.push(OptimizationRecommendation {
                    recommendation_type: RecommendationType::MemoryOptimization,
                    priority: RecommendationPriority::High,
                    description: "High memory usage detected. Consider enabling memory pooling or reducing batch size.".to_string(),
                    expected_improvement: "20-40% memory reduction".to_string(),
                    difficulty: ImplementationDifficulty::Medium,
                    affected_operations: vec!["All operations".to_string()],
                });
            }

            // Thermal check
            if let Some(temp) = current_metrics.temperature {
                if temp > 70.0 {
                    // > 70°C
                    recommendations.push(OptimizationRecommendation {
                        recommendation_type: RecommendationType::ThermalManagement,
                        priority: RecommendationPriority::Critical,
                        description: "High temperature detected. Enable thermal throttling or reduce precision.".to_string(),
                        expected_improvement: "Temperature reduction and sustained performance".to_string(),
                        difficulty: ImplementationDifficulty::Easy,
                        affected_operations: vec!["Compute-intensive operations".to_string()],
                    });
                }
            }

            // Power consumption check
            if let Some(power) = current_metrics.power_consumption {
                if power > 5.0 {
                    // > 5W
                    recommendations.push(OptimizationRecommendation {
                        recommendation_type: RecommendationType::PowerManagement,
                        priority: RecommendationPriority::Medium,
                        description: "High power consumption detected. Consider switching to power-saving mode.".to_string(),
                        expected_improvement: "20-30% power reduction".to_string(),
                        difficulty: ImplementationDifficulty::Easy,
                        affected_operations: vec!["All operations".to_string()],
                    });
                }
            }
        }

        recommendations
    }

    // Private helper methods

    fn capture_current_metrics(&self) -> Result<PerformanceMetrics> {
        Ok(PerformanceMetrics {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            cpu_usage: self.get_cpu_usage(),
            memory_usage: self.get_memory_usage(),
            peak_memory: self.get_peak_memory(),
            gpu_usage: self.get_gpu_usage(),
            gpu_memory: self.get_gpu_memory(),
            temperature: Some(self.get_current_temperature()),
            battery_level: self.get_battery_level(),
            power_consumption: Some(self.get_current_power_consumption()),
            network_sent_bytes: self.get_network_sent(),
            network_received_bytes: self.get_network_received(),
            fps: self.calculate_fps(),
            inference_latency_ms: self.calculate_average_latency(),
            throughput_ips: self.calculate_throughput(),
        })
    }

    fn get_system_info(&self) -> SystemInfo {
        SystemInfo {
            platform: match self.mobile_config.platform {
                MobilePlatform::Ios => "iOS".to_string(),
                MobilePlatform::Android => "Android".to_string(),
                MobilePlatform::Generic => "Generic".to_string(),
            },
            device_model: self.detect_device_model(),
            os_version: self.detect_os_version(),
            cpu_arch: std::env::consts::ARCH.to_string(),
            total_memory_bytes: (self.mobile_config.max_memory_mb * 1024 * 1024) as u64,
            gpu_info: self.detect_gpu_info(),
            available_backends: vec![format!("{:?}", self.mobile_config.backend)],
        }
    }

    fn calculate_performance_summary(&self) -> PerformanceSummary {
        if self.metrics_history.is_empty() {
            return PerformanceSummary {
                avg_inference_latency_ms: 0.0,
                p95_latency_ms: 0.0,
                p99_latency_ms: 0.0,
                avg_throughput_ips: 0.0,
                peak_throughput_ips: 0.0,
                avg_memory_usage_mb: 0.0,
                peak_memory_usage_mb: 0.0,
                avg_power_consumption_w: 0.0,
                total_energy_consumed_j: 0.0,
                efficiency_score: 0.0,
            };
        }

        let total_metrics = self.metrics_history.len() as f32;

        let avg_memory = self.metrics_history.iter().map(|m| m.memory_usage as f32).sum::<f32>()
            / total_metrics
            / (1024.0 * 1024.0); // Convert to MB

        let peak_memory = self.metrics_history.iter().map(|m| m.peak_memory).max().unwrap_or(0)
            as f32
            / (1024.0 * 1024.0); // Convert to MB

        let avg_power =
            self.metrics_history.iter().filter_map(|m| m.power_consumption).sum::<f32>()
                / total_metrics;

        // Calculate latency statistics from operation profiles
        let latencies: Vec<f32> = self.operation_profiles.iter()
            .map(|op| op.execution_time_us as f32 / 1000.0) // Convert to ms
            .collect();

        let (avg_latency, p95_latency, p99_latency) =
            self.calculate_latency_percentiles(&latencies);

        // Calculate throughput from operation profiles
        let (avg_throughput, peak_throughput) = self.calculate_throughput_stats();

        // Calculate efficiency score based on multiple factors
        let efficiency_score = self.calculate_efficiency_score(avg_latency, avg_power, avg_memory);

        PerformanceSummary {
            avg_inference_latency_ms: avg_latency,
            p95_latency_ms: p95_latency,
            p99_latency_ms: p99_latency,
            avg_throughput_ips: avg_throughput,
            peak_throughput_ips: peak_throughput,
            avg_memory_usage_mb: avg_memory,
            peak_memory_usage_mb: peak_memory,
            avg_power_consumption_w: avg_power,
            total_energy_consumed_j: avg_power * self.start_time.elapsed().as_secs_f32(),
            efficiency_score,
        }
    }

    fn analyze_thermal_performance(&self) -> ThermalAnalysis {
        let current_temp = self.get_current_temperature();
        let is_throttling = current_temp > 80.0; // Simple threshold

        let trend = if current_temp > 75.0 {
            TemperatureTrend::Critical
        } else if current_temp > 65.0 {
            TemperatureTrend::Rising
        } else if current_temp > 45.0 {
            TemperatureTrend::Stable
        } else {
            TemperatureTrend::Cooling
        };

        let recommendation = match trend {
            TemperatureTrend::Critical => ThermalRecommendation::PauseInference,
            TemperatureTrend::Rising => ThermalRecommendation::ReduceFrequency,
            _ => ThermalRecommendation::Continue,
        };

        ThermalAnalysis {
            temperature_zones: {
                let mut zones = HashMap::new();
                zones.insert("CPU".to_string(), current_temp);
                zones.insert("GPU".to_string(), current_temp - 5.0); // Estimate
                zones
            },
            is_throttling,
            temperature_trend: trend,
            time_to_throttling_ms: if current_temp > 70.0 { Some(30000) } else { None },
            recommended_action: recommendation,
        }
    }

    fn analyze_power_consumption(&self) -> PowerAnalysis {
        let current_power = self.get_current_power_consumption();
        let avg_power =
            self.metrics_history.iter().filter_map(|m| m.power_consumption).sum::<f32>()
                / self.metrics_history.len() as f32;
        let peak_power = self
            .metrics_history
            .iter()
            .filter_map(|m| m.power_consumption)
            .fold(0.0f32, |acc, x| acc.max(x));

        PowerAnalysis {
            current_power_w: current_power,
            average_power_w: avg_power,
            peak_power_w: peak_power,
            battery_life_remaining_min: self.estimate_battery_life(),
            power_efficiency_ipw: Some(20.0 / current_power), // Assuming 20 IPS
            recommended_power_mode: if current_power > 5.0 {
                PowerMode::PowerSaver
            } else if current_power > 3.0 {
                PowerMode::Balanced
            } else {
                PowerMode::MaxPerformance
            },
        }
    }

    fn analyze_memory_usage(&self) -> MemoryAnalysis {
        let current_usage = self.get_memory_usage();
        let peak_usage = self.get_peak_memory();
        let available = (self.mobile_config.max_memory_mb * 1024 * 1024) as u64 - current_usage;

        MemoryAnalysis {
            current_usage_bytes: current_usage,
            peak_usage_bytes: peak_usage,
            available_bytes: available,
            fragmentation_percent: self.calculate_memory_fragmentation(),
            allocation_patterns: self.analyze_allocation_patterns(),
            potential_leaks: self.detect_memory_leaks(),
            gc_stats: self.calculate_gc_stats(),
        }
    }

    fn generate_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Add quantization recommendation if using high precision
        recommendations.push(OptimizationRecommendation {
            recommendation_type: RecommendationType::Quantization,
            priority: RecommendationPriority::High,
            description:
                "Consider using INT8 quantization for better performance and lower memory usage."
                    .to_string(),
            expected_improvement: "50% memory reduction, 2x speed improvement".to_string(),
            difficulty: ImplementationDifficulty::Medium,
            affected_operations: ["Linear", "Convolution"].iter().map(|s| s.to_string()).collect(),
        });

        // Add operator fusion recommendation
        recommendations.push(OptimizationRecommendation {
            recommendation_type: RecommendationType::OperatorFusion,
            priority: RecommendationPriority::Medium,
            description: "Fuse consecutive operators to reduce memory transfers and improve cache efficiency.".to_string(),
            expected_improvement: "15-25% latency reduction".to_string(),
            difficulty: ImplementationDifficulty::Easy,
            affected_operations: ["Conv+BatchNorm", "Linear+Activation"].iter().map(|s| s.to_string()).collect(),
        });

        recommendations
    }

    fn identify_bottlenecks(&self) -> Vec<PerformanceBottleneck> {
        let mut bottlenecks = Vec::new();

        // Memory bottleneck analysis
        if self.get_memory_usage() > (self.mobile_config.max_memory_mb * 1024 * 1024 * 3 / 4) as u64
        {
            bottlenecks.push(PerformanceBottleneck {
                bottleneck_type: BottleneckType::Memory,
                severity: 75.0,
                affected_operations: vec!["All operations".to_string()],
                description: "High memory usage may cause performance degradation.".to_string(),
                solutions: vec![
                    "Enable memory pooling".to_string(),
                    "Reduce batch size".to_string(),
                    "Use quantization".to_string(),
                ],
                performance_impact_percent: 25.0,
            });
        }

        // Thermal bottleneck analysis
        if self.get_current_temperature() > 70.0 {
            bottlenecks.push(PerformanceBottleneck {
                bottleneck_type: BottleneckType::Thermal,
                severity: 85.0,
                affected_operations: vec!["Compute-intensive operations".to_string()],
                description: "High temperature causing thermal throttling.".to_string(),
                solutions: vec![
                    "Reduce computation frequency".to_string(),
                    "Enable thermal management".to_string(),
                    "Switch to lower precision".to_string(),
                ],
                performance_impact_percent: 40.0,
            });
        }

        bottlenecks
    }

    // Platform-specific metric collection methods
    fn get_cpu_usage(&self) -> f32 {
        // Simulate CPU usage based on recent operation activity
        let recent_ops = self
            .operation_profiles
            .iter()
            .rev()
            .take(10)
            .map(|op| op.execution_time_us)
            .sum::<u64>();

        // Convert to a percentage (normalize by 100ms window)
        let base_usage = 25.0;
        let activity_factor = (recent_ops as f32 / 100_000.0).min(50.0); // Cap at 50% additional
        base_usage + activity_factor
    }

    fn get_memory_usage(&self) -> u64 {
        // Simulate memory usage based on model configuration and operations
        let base_memory = (self.mobile_config.max_memory_mb as u64 * 1024 * 1024) / 4; // 25% base
        let operation_memory = self
            .operation_profiles
            .iter()
            .map(|op| op.memory_allocated_bytes.saturating_sub(op.memory_freed_bytes))
            .sum::<u64>();

        base_memory + operation_memory
    }

    fn get_peak_memory(&self) -> u64 {
        let current = self.get_memory_usage();
        let historical_peak =
            self.metrics_history.iter().map(|m| m.peak_memory).max().unwrap_or(current);

        current.max(historical_peak)
    }

    fn get_gpu_usage(&self) -> Option<f32> {
        match self.mobile_config.backend {
            MobileBackend::Metal | MobileBackend::Vulkan | MobileBackend::OpenCL => {
                // Simulate GPU usage based on compute operations
                let gpu_ops = self
                    .operation_profiles
                    .iter()
                    .filter(|op| {
                        matches!(
                            op.operation_type,
                            OperationType::Convolution
                                | OperationType::LinearTransform
                                | OperationType::Attention
                        )
                    })
                    .count();

                let base_usage = 20.0;
                let activity_usage = (gpu_ops as f32 * 5.0).min(60.0);
                Some(base_usage + activity_usage)
            },
            _ => None,
        }
    }

    fn get_gpu_memory(&self) -> Option<u64> {
        match self.mobile_config.backend {
            MobileBackend::Metal | MobileBackend::Vulkan | MobileBackend::OpenCL => {
                // Estimate GPU memory based on tensor operations
                let gpu_memory = self
                    .operation_profiles
                    .iter()
                    .filter(|op| {
                        matches!(
                            op.operation_type,
                            OperationType::Convolution | OperationType::LinearTransform
                        )
                    })
                    .map(|op| {
                        let input_size: usize =
                            op.input_shapes.iter().map(|s| s.iter().product::<usize>()).sum();
                        let output_size: usize = op.output_shape.iter().product();
                        ((input_size + output_size) * 4) as u64 // 4 bytes per float
                    })
                    .sum::<u64>();

                Some((200 * 1024 * 1024) + gpu_memory) // 200MB base + operation memory
            },
            _ => None,
        }
    }

    fn get_current_temperature(&self) -> f32 {
        // Simulate temperature based on CPU/GPU usage and duration
        let cpu_usage = self.get_cpu_usage();
        let gpu_usage = self.get_gpu_usage().unwrap_or(0.0);
        let duration_minutes = self.start_time.elapsed().as_secs() as f32 / 60.0;

        let base_temp = 35.0; // Ambient
        let cpu_heat = cpu_usage * 0.3; // CPU contributes to heat
        let gpu_heat = gpu_usage * 0.2; // GPU contributes to heat
        let duration_heat = duration_minutes * 0.5; // Heat buildup over time

        (base_temp + cpu_heat + gpu_heat + duration_heat).min(85.0) // Cap at 85°C
    }

    fn get_battery_level(&self) -> Option<f32> {
        match self.mobile_config.platform {
            MobilePlatform::Ios | MobilePlatform::Android => {
                // Simulate battery drain based on power consumption
                let power = self.get_current_power_consumption();
                let duration_hours = self.start_time.elapsed().as_secs() as f32 / 3600.0;
                let initial_level = 85.0;
                let drain_rate = power * 2.0; // % per hour per watt

                Some((initial_level - (drain_rate * duration_hours)).max(5.0))
            },
            _ => None,
        }
    }

    fn get_current_power_consumption(&self) -> f32 {
        // Calculate power based on CPU/GPU usage and backend
        let cpu_usage = self.get_cpu_usage();
        let gpu_usage = self.get_gpu_usage().unwrap_or(0.0);

        let base_power = 1.0; // Base system power
        let cpu_power = cpu_usage * 0.02; // 2W at 100% CPU
        let gpu_power = gpu_usage * 0.03; // 3W at 100% GPU

        base_power + cpu_power + gpu_power
    }

    fn get_network_sent(&self) -> Option<u64> {
        // Simulate network usage if enabled
        if self.config.enable_network_tracking {
            Some(self.operation_profiles.len() as u64 * 512) // 512 bytes per operation
        } else {
            None
        }
    }

    fn get_network_received(&self) -> Option<u64> {
        // Simulate network usage if enabled
        if self.config.enable_network_tracking {
            Some(self.operation_profiles.len() as u64 * 1024) // 1KB per operation
        } else {
            None
        }
    }

    fn estimate_battery_life(&self) -> Option<f32> {
        if let Some(battery_level) = self.get_battery_level() {
            let power_consumption = self.get_current_power_consumption();
            let battery_capacity_wh = match self.mobile_config.platform {
                MobilePlatform::Ios => 15.0,     // Typical iPhone battery
                MobilePlatform::Android => 20.0, // Typical Android battery
                _ => 10.0,
            };

            let remaining_capacity = (battery_level / 100.0) * battery_capacity_wh;
            let estimated_hours = remaining_capacity / power_consumption;

            Some(estimated_hours * 60.0) // Convert to minutes
        } else {
            None
        }
    }

    fn infer_operation_type(&self, operation_name: &str) -> OperationType {
        let name_lower = operation_name.to_lowercase();
        if name_lower.contains("conv") {
            OperationType::Convolution
        } else if name_lower.contains("linear")
            || name_lower.contains("dense")
            || name_lower.contains("matmul")
        {
            OperationType::LinearTransform
        } else if name_lower.contains("attention") || name_lower.contains("attn") {
            OperationType::Attention
        } else if name_lower.contains("norm")
            || name_lower.contains("batch")
            || name_lower.contains("layer")
        {
            OperationType::Normalization
        } else if name_lower.contains("relu")
            || name_lower.contains("gelu")
            || name_lower.contains("sigmoid")
            || name_lower.contains("tanh")
            || name_lower.contains("softmax")
        {
            OperationType::Activation
        } else if name_lower.contains("pool") {
            OperationType::Pooling
        } else if name_lower.contains("quantize") {
            OperationType::Quantization
        } else if name_lower.contains("dequantize") {
            OperationType::Dequantization
        } else if name_lower.contains("copy") || name_lower.contains("memcpy") {
            OperationType::MemoryCopy
        } else if name_lower.contains("transfer")
            || name_lower.contains("upload")
            || name_lower.contains("download")
        {
            OperationType::DataTransfer
        } else {
            OperationType::Custom(operation_name.to_string())
        }
    }

    fn estimate_flops(&self, input_shapes: &[Vec<usize>], output_shape: &[usize]) -> Option<u64> {
        // Simple FLOP estimation based on shapes
        if input_shapes.is_empty() || output_shape.is_empty() {
            return None;
        }

        let input_size: usize = input_shapes[0].iter().product();
        let output_size: usize = output_shape.iter().product();

        // Estimate as 2 * input_size * output_size (for matrix multiplication)
        Some((2 * input_size * output_size) as u64)
    }

    // Additional helper methods for enhanced profiling functionality

    fn get_gpu_kernel_time(&self, operation_name: &str, cpu_time: Duration) -> Option<u64> {
        // Estimate GPU kernel time based on operation type and CPU time
        if self.get_gpu_usage().is_some() {
            let operation_type = self.infer_operation_type(operation_name);
            let gpu_efficiency = match operation_type {
                OperationType::Convolution => 0.3, // GPU is much faster for conv
                OperationType::LinearTransform => 0.4, // GPU good for matrix ops
                OperationType::Attention => 0.5,   // GPU moderate for attention
                OperationType::Activation => 0.8,  // GPU not much faster for simple ops
                _ => 0.9,                          // Mostly CPU-bound
            };

            Some((cpu_time.as_micros() as f64 * gpu_efficiency) as u64)
        } else {
            None
        }
    }

    fn estimate_cache_hit_rate(&self, operation_name: &str) -> Option<f32> {
        // Estimate cache hit rate based on operation type and recent history
        let operation_type = self.infer_operation_type(operation_name);
        let base_hit_rate = match operation_type {
            OperationType::Convolution => 0.85,     // Good spatial locality
            OperationType::LinearTransform => 0.75, // Sequential access
            OperationType::Attention => 0.60,       // Random access patterns
            OperationType::Activation => 0.90,      // Element-wise, good locality
            OperationType::Normalization => 0.80,   // Sequential with some gathering
            _ => 0.70,                              // Default
        };

        // Adjust based on recent cache pressure (simulate)
        let recent_memory_pressure = self
            .metrics_history
            .iter()
            .rev()
            .take(5)
            .map(|m| {
                m.memory_usage as f32 / (self.mobile_config.max_memory_mb as f32 * 1024.0 * 1024.0)
            })
            .sum::<f32>()
            / 5.0;

        let pressure_penalty = recent_memory_pressure * 0.2; // Up to 20% penalty
        Some((base_hit_rate - pressure_penalty).max(0.1))
    }

    fn calculate_fps(&self) -> Option<f32> {
        if self.operation_profiles.len() < 2 {
            return None;
        }

        // Calculate FPS based on recent operation completion rate
        let recent_operations = self.operation_profiles.iter()
            .rev()
            .take(30) // Last 30 operations
            .collect::<Vec<_>>();

        if recent_operations.len() < 2 {
            return None;
        }

        let total_time_ms = recent_operations
            .iter()
            .map(|op| op.execution_time_us as f32 / 1000.0)
            .sum::<f32>();

        if total_time_ms > 0.0 {
            Some(1000.0 * recent_operations.len() as f32 / total_time_ms)
        } else {
            None
        }
    }

    fn calculate_average_latency(&self) -> Option<f32> {
        if self.operation_profiles.is_empty() {
            return None;
        }

        let total_latency = self.operation_profiles.iter()
            .map(|op| op.execution_time_us as f32 / 1000.0) // Convert to ms
            .sum::<f32>();

        Some(total_latency / self.operation_profiles.len() as f32)
    }

    fn calculate_throughput(&self) -> Option<f32> {
        if self.operation_profiles.is_empty() || self.start_time.elapsed().as_secs() == 0 {
            return None;
        }

        let elapsed_seconds = self.start_time.elapsed().as_secs_f32();
        Some(self.operation_profiles.len() as f32 / elapsed_seconds)
    }

    fn detect_device_model(&self) -> String {
        match self.mobile_config.platform {
            MobilePlatform::Ios => {
                // Simulate iOS device detection
                "iPhone 15 Pro".to_string() // Simulated
            },
            MobilePlatform::Android => {
                // Simulate Android device detection
                "Samsung Galaxy S24".to_string() // Simulated
            },
            MobilePlatform::Generic => {
                std::env::var("DEVICE_MODEL").unwrap_or_else(|_| "Generic Device".to_string())
            },
        }
    }

    fn detect_os_version(&self) -> String {
        match self.mobile_config.platform {
            MobilePlatform::Ios => {
                "iOS 17.5".to_string() // Simulated
            },
            MobilePlatform::Android => {
                "Android 14".to_string() // Simulated
            },
            MobilePlatform::Generic => std::env::consts::OS.to_string(),
        }
    }

    fn detect_gpu_info(&self) -> Option<String> {
        match self.mobile_config.backend {
            MobileBackend::Metal => Some("Apple A17 Pro GPU".to_string()),
            MobileBackend::Vulkan => Some("Adreno 750 GPU".to_string()),
            MobileBackend::OpenCL => Some("Mali-G78 GPU".to_string()),
            _ => None,
        }
    }

    fn calculate_latency_percentiles(&self, latencies: &[f32]) -> (f32, f32, f32) {
        if latencies.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let mut sorted = latencies.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let avg = sorted.iter().sum::<f32>() / sorted.len() as f32;
        let p95_idx = ((sorted.len() as f32 * 0.95) as usize).min(sorted.len() - 1);
        let p99_idx = ((sorted.len() as f32 * 0.99) as usize).min(sorted.len() - 1);

        let p95 = sorted[p95_idx];
        let p99 = sorted[p99_idx];

        (avg, p95, p99)
    }

    fn calculate_throughput_stats(&self) -> (f32, f32) {
        if self.operation_profiles.is_empty() {
            return (0.0, 0.0);
        }

        // Calculate throughput for sliding windows
        let window_size = 10;
        let mut throughputs = Vec::new();

        for window_start in 0..self.operation_profiles.len().saturating_sub(window_size - 1) {
            let window_end = (window_start + window_size).min(self.operation_profiles.len());
            let window_ops = &self.operation_profiles[window_start..window_end];

            let total_time_s = window_ops
                .iter()
                .map(|op| op.execution_time_us as f32 / 1_000_000.0)
                .sum::<f32>();

            if total_time_s > 0.0 {
                throughputs.push(window_ops.len() as f32 / total_time_s);
            }
        }

        if throughputs.is_empty() {
            return (0.0, 0.0);
        }

        let avg_throughput = throughputs.iter().sum::<f32>() / throughputs.len() as f32;
        let peak_throughput = throughputs.iter().fold(0.0f32, |acc, &x| acc.max(x));

        (avg_throughput, peak_throughput)
    }

    fn calculate_efficiency_score(
        &self,
        avg_latency: f32,
        avg_power: f32,
        avg_memory_mb: f32,
    ) -> f32 {
        // Calculate efficiency score based on multiple factors (0-100)
        let mut score = 100.0;

        // Latency penalty (higher latency = lower score)
        let latency_penalty = (avg_latency / 100.0).min(50.0); // Cap at 50% penalty
        score -= latency_penalty;

        // Power penalty (higher power = lower score)
        let power_penalty = (avg_power / 10.0 * 20.0).min(30.0); // Cap at 30% penalty
        score -= power_penalty;

        // Memory penalty (higher memory usage = lower score)
        let memory_ratio = avg_memory_mb / (self.mobile_config.max_memory_mb as f32);
        let memory_penalty = (memory_ratio * 20.0).min(20.0); // Cap at 20% penalty
        score -= memory_penalty;

        score.max(0.0)
    }

    fn calculate_memory_fragmentation(&self) -> f32 {
        // Estimate fragmentation based on allocation patterns
        let total_allocations =
            self.operation_profiles.iter().map(|op| op.memory_allocated_bytes).sum::<u64>();

        let total_deallocations =
            self.operation_profiles.iter().map(|op| op.memory_freed_bytes).sum::<u64>();

        if total_allocations > 0 {
            let allocation_efficiency = total_deallocations as f32 / total_allocations as f32;
            let fragmentation = (1.0 - allocation_efficiency.min(1.0)) * 100.0;
            fragmentation.min(25.0) // Cap at 25% fragmentation
        } else {
            0.0
        }
    }

    fn analyze_allocation_patterns(&self) -> Vec<AllocationPattern> {
        let mut patterns = std::collections::HashMap::new();

        // Group allocations by size ranges
        for op in &self.operation_profiles {
            if op.memory_allocated_bytes > 0 {
                let size_bucket = if op.memory_allocated_bytes < 1024 {
                    "Small (<1KB)".to_string()
                } else if op.memory_allocated_bytes < 1024 * 1024 {
                    "Medium (1KB-1MB)".to_string()
                } else {
                    "Large (>1MB)".to_string()
                };

                let pattern = patterns.entry(size_bucket).or_insert(AllocationPattern {
                    size_bytes: 0,
                    frequency: 0,
                    average_lifetime_ms: 0.0,
                    source: "Unknown".to_string(),
                });

                pattern.frequency += 1;
                pattern.size_bytes = op.memory_allocated_bytes; // Use latest as representative
                pattern.average_lifetime_ms = op.execution_time_us as f32 / 1000.0; // Approximate
                pattern.source = format!("{:?}", op.operation_type);
            }
        }

        patterns.into_values().collect()
    }

    fn detect_memory_leaks(&self) -> Vec<MemoryLeak> {
        let mut potential_leaks = Vec::new();

        // Look for operations with significant allocations but no deallocations
        for op in &self.operation_profiles {
            if op.memory_allocated_bytes > 1024 * 1024 && op.memory_freed_bytes == 0 {
                potential_leaks.push(MemoryLeak {
                    size_bytes: op.memory_allocated_bytes,
                    age_ms: op.execution_time_us / 1000, // Simplified
                    source: op.operation_name.clone(),
                    confidence: if op.memory_allocated_bytes > 10 * 1024 * 1024 {
                        0.8
                    } else {
                        0.4
                    },
                });
            }
        }

        potential_leaks
    }

    fn calculate_gc_stats(&self) -> Option<GCStats> {
        // Simulate GC stats for mobile platforms
        match self.mobile_config.platform {
            MobilePlatform::Android => {
                // Android has GC
                let operations_count = self.operation_profiles.len() as u32;
                let estimated_gc_cycles = operations_count / 50; // Estimate GC every 50 operations

                Some(GCStats {
                    gc_cycles: estimated_gc_cycles,
                    total_gc_time_ms: estimated_gc_cycles as u64 * 5, // 5ms per GC
                    average_pause_ms: 5.0,
                    memory_reclaimed_bytes: self
                        .operation_profiles
                        .iter()
                        .map(|op| op.memory_freed_bytes)
                        .sum::<u64>(),
                })
            },
            _ => None, // iOS doesn't have traditional GC
        }
    }

    fn export_csv_report(&self, report: &ProfilingReport) -> Result<String> {
        let mut csv = String::new();

        // Header
        csv.push_str("timestamp,cpu_usage,memory_usage,temperature,power_consumption\n");

        // Data rows
        for metric in &report.metrics_timeline {
            csv.push_str(&format!(
                "{},{},{},{},{}\n",
                metric.timestamp,
                metric.cpu_usage,
                metric.memory_usage,
                metric.temperature.unwrap_or(0.0),
                metric.power_consumption.unwrap_or(0.0)
            ));
        }

        Ok(csv)
    }

    fn export_flamegraph(&self, report: &ProfilingReport) -> Result<String> {
        // Generate flamegraph data in SVG format
        let mut flamegraph_data = String::new();
        flamegraph_data.push_str("# Flamegraph Data\n");

        // Group operations by type and calculate cumulative times
        let mut stack_traces = std::collections::HashMap::new();

        for op in &report.operation_profiles {
            let stack = format!("{:?};{}", op.operation_type, op.operation_name);
            *stack_traces.entry(stack).or_insert(0u64) += op.execution_time_us;
        }

        // Sort by execution time descending
        let mut sorted_traces: Vec<_> = stack_traces.into_iter().collect();
        sorted_traces.sort_by(|a, b| b.1.cmp(&a.1));

        for (stack, time_us) in sorted_traces {
            flamegraph_data.push_str(&format!("{} {}\n", stack, time_us));
        }

        Ok(flamegraph_data)
    }

    fn export_chrome_format(&self, report: &ProfilingReport) -> Result<String> {
        // Generate Chrome DevTools tracing format (JSON)
        let mut events = Vec::new();

        // Add process info
        events.push(serde_json::json!({
            "name": "process_name",
            "ph": "M",
            "pid": 1,
            "args": {
                "name": "TrustformeRS Mobile Profiler"
            }
        }));

        // Add thread info
        events.push(serde_json::json!({
            "name": "thread_name",
            "ph": "M",
            "pid": 1,
            "tid": 1,
            "args": {
                "name": "Main Thread"
            }
        }));

        // Add operation events
        let mut current_time = 0u64;
        for op in &report.operation_profiles {
            // Begin event
            events.push(serde_json::json!({
                "name": op.operation_name,
                "cat": format!("{:?}", op.operation_type),
                "ph": "B",
                "ts": current_time,
                "pid": 1,
                "tid": 1,
                "args": {
                    "input_shapes": op.input_shapes,
                    "output_shape": op.output_shape,
                    "flops": op.flops
                }
            }));

            // End event
            events.push(serde_json::json!({
                "name": op.operation_name,
                "cat": format!("{:?}", op.operation_type),
                "ph": "E",
                "ts": current_time + op.execution_time_us,
                "pid": 1,
                "tid": 1
            }));

            current_time += op.execution_time_us + 100; // Add small gap
        }

        let chrome_trace = serde_json::json!({
            "traceEvents": events,
            "displayTimeUnit": "ms",
            "otherData": {
                "version": "Chrome Trace Format",
                "creator": "TrustformeRS Mobile Profiler"
            }
        });

        serde_json::to_string_pretty(&chrome_trace)
            .map_err(|e| TrustformersError::serialization_error(e.to_string()).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let config = AdvancedProfilerConfig::default();
        let mobile_config = MobileConfig::default();
        let profiler = AdvancedProfiler::new(config, mobile_config);
        assert!(!profiler.session_id.is_empty());
    }

    #[test]
    fn test_metrics_capture() {
        let config = AdvancedProfilerConfig::default();
        let mobile_config = MobileConfig::default();
        let mut profiler = AdvancedProfiler::new(config, mobile_config);

        let metrics = profiler.capture_metrics();
        assert!(metrics.is_ok());
        assert_eq!(profiler.metrics_history.len(), 1);
    }

    #[test]
    fn test_session_management() {
        let config = AdvancedProfilerConfig::default();
        let mobile_config = MobileConfig::default();
        let mut profiler = AdvancedProfiler::new(config, mobile_config);

        let session_id = profiler.start_session();
        assert!(session_id.is_ok());
        assert!(profiler.current_session.is_some());
    }

    #[test]
    fn test_operation_profiling() {
        let config = AdvancedProfilerConfig::default();
        let mobile_config = MobileConfig::default();
        let mut profiler = AdvancedProfiler::new(config, mobile_config);

        let _ = profiler.start_session();
        let _ = profiler.operation_start("test_op", OperationType::LinearTransform);
        std::thread::sleep(std::time::Duration::from_millis(10));
        let _ = profiler.operation_end("test_op", vec![vec![32, 768]], vec![32, 10]);

        assert_eq!(profiler.operation_profiles.len(), 1);
    }

    #[test]
    fn test_report_generation() {
        let config = AdvancedProfilerConfig::default();
        let mobile_config = MobileConfig::default();
        let mut profiler = AdvancedProfiler::new(config, mobile_config);

        let _ = profiler.capture_metrics();
        let report = profiler.generate_report();
        assert!(report.is_ok());
    }

    #[test]
    fn test_recommendations() {
        let config = AdvancedProfilerConfig::default();
        let mobile_config = MobileConfig::default();
        let mut profiler = AdvancedProfiler::new(config, mobile_config);

        // Manually add metrics that will trigger recommendations
        let high_memory_metrics = PerformanceMetrics {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            cpu_usage: 50.0,
            memory_usage: 5 * 1024 * 1024 * 1024, // 5GB - exceeds 4GB threshold
            peak_memory: 5 * 1024 * 1024 * 1024,
            gpu_usage: Some(30.0),
            gpu_memory: Some(2 * 1024 * 1024 * 1024),
            temperature: Some(75.0), // Exceeds 70°C threshold
            battery_level: Some(50.0),
            power_consumption: Some(6.0), // Exceeds 5W threshold
            network_sent_bytes: Some(1024 * 1024),
            network_received_bytes: Some(1024 * 1024),
            fps: Some(30.0),
            inference_latency_ms: Some(100.0),
            throughput_ips: Some(10.0),
        };

        profiler.metrics_history.push_back(high_memory_metrics);

        let recommendations = profiler.get_realtime_recommendations();
        assert!(!recommendations.is_empty());
    }
}
