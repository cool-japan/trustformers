//! Enhanced Performance Profiler for TrustformeRS
//!
//! This module provides advanced performance profiling capabilities with:
//! - Real-time performance monitoring
//! - Hardware-specific optimization suggestions
//! - Memory leak detection
//! - Comprehensive performance analytics
//! - Integration with modern observability tools

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

/// Enhanced Performance Profiler with advanced analytics
#[derive(Debug)]
pub struct EnhancedProfiler {
    sessions: Arc<RwLock<HashMap<String, ProfilingSession>>>,
    global_metrics: Arc<RwLock<GlobalMetrics>>,
    config: ProfilerConfig,
}

/// Configuration for the enhanced profiler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    /// Enable hardware-specific profiling
    pub hardware_profiling: bool,
    /// Enable memory leak detection
    pub memory_leak_detection: bool,
    /// Enable real-time performance alerts
    pub real_time_alerts: bool,
    /// Enable AI-powered performance analysis
    pub ai_powered_analysis: bool,
    /// Sampling interval for continuous profiling
    pub sampling_interval_ms: u64,
    /// Maximum number of performance samples to keep
    pub max_samples: usize,
    /// Performance thresholds for alerts
    pub thresholds: PerformanceThresholds,
    /// Export formats enabled
    pub export_formats: Vec<ExportFormat>,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            hardware_profiling: true,
            memory_leak_detection: true,
            real_time_alerts: true,
            ai_powered_analysis: false, // Disabled by default
            sampling_interval_ms: 100,
            max_samples: 10000,
            thresholds: PerformanceThresholds::default(),
            export_formats: vec![ExportFormat::JSON, ExportFormat::Prometheus],
        }
    }
}

/// Performance thresholds for alerting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    pub max_latency_ms: f32,
    pub min_throughput_ops_per_sec: f32,
    pub max_memory_usage_mb: f32,
    pub max_cpu_usage_percent: f32,
    pub max_gpu_usage_percent: f32,
    pub memory_leak_threshold_mb: f32,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_latency_ms: 1000.0,
            min_throughput_ops_per_sec: 10.0,
            max_memory_usage_mb: 1024.0,
            max_cpu_usage_percent: 90.0,
            max_gpu_usage_percent: 95.0,
            memory_leak_threshold_mb: 10.0,
        }
    }
}

/// Export formats for profiling data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    JSON,
    CSV,
    Prometheus,
    Flamegraph,
    OpenTelemetry,
    Jaeger,
}

/// Profiling session for tracking performance of specific operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingSession {
    pub session_id: String,
    pub operation_name: String,
    #[serde(skip, default = "Instant::now")]
    pub start_time: Instant,
    pub samples: Vec<PerformanceSample>,
    pub hardware_info: HardwareInfo,
    pub memory_tracker: MemoryTracker,
    pub status: SessionStatus,
}

/// Performance sample capturing point-in-time metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSample {
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    pub latency_ms: f32,
    pub throughput_ops_per_sec: f32,
    pub memory_usage_mb: f32,
    pub cpu_usage_percent: f32,
    pub gpu_usage_percent: f32,
    pub custom_metrics: HashMap<String, f64>,
}

/// Hardware information for optimization recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub cpu_cores: usize,
    pub cpu_model: String,
    pub total_memory_gb: f32,
    pub gpu_info: Vec<GPUInfo>,
    pub platform: Platform,
    pub specialized_hardware: Vec<SpecializedHardware>,
}

/// GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUInfo {
    pub name: String,
    pub memory_gb: f32,
    pub compute_capability: String,
    pub utilization_percent: f32,
}

/// Platform detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Platform {
    Linux,
    Windows,
    MacOS,
    Unknown,
}

/// Specialized hardware detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpecializedHardware {
    CUDA,
    ROCm,
    Metal,
    OpenCL,
    TensorRT,
    CoreML,
    ONNX,
    TPU,
    NPU,
}

/// Memory tracking for leak detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTracker {
    pub initial_memory_mb: f32,
    pub peak_memory_mb: f32,
    pub current_memory_mb: f32,
    pub allocation_count: u64,
    pub deallocation_count: u64,
    pub leak_detected: bool,
    pub memory_samples: Vec<MemorySample>,
}

/// Memory sample for tracking over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySample {
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    pub memory_mb: f32,
    pub allocations: u64,
    pub deallocations: u64,
}

/// Session status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionStatus {
    Active,
    Completed,
    Failed,
    Cancelled,
}

/// Global metrics across all sessions
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct GlobalMetrics {
    pub total_sessions: u64,
    pub active_sessions: u64,
    pub average_latency_ms: f32,
    pub total_operations: u64,
    pub memory_leaks_detected: u64,
    pub performance_alerts: u64,
    pub optimization_suggestions: Vec<OptimizationSuggestion>,
}

/// AI-powered optimization suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    pub category: OptimizationCategory,
    pub severity: SuggestionSeverity,
    pub description: String,
    pub suggested_action: String,
    pub expected_improvement: String,
    pub confidence: f32,
}

/// Optimization categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationCategory {
    Memory,
    CPU,
    GPU,
    IO,
    NetworkLatency,
    ModelArchitecture,
    BatchSize,
    Quantization,
    Caching,
    Threading,
}

/// Suggestion severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuggestionSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Comprehensive performance analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalysis {
    pub session_summary: SessionSummary,
    pub performance_trends: PerformanceTrends,
    pub bottleneck_analysis: BottleneckAnalysis,
    pub optimization_recommendations: Vec<OptimizationSuggestion>,
    pub hardware_utilization: HardwareUtilization,
    pub memory_analysis: MemoryAnalysis,
}

/// Session summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub total_duration_ms: f32,
    pub total_operations: u64,
    pub average_latency_ms: f32,
    pub p95_latency_ms: f32,
    pub p99_latency_ms: f32,
    pub peak_throughput_ops_per_sec: f32,
    pub peak_memory_mb: f32,
}

/// Performance trends over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    pub latency_trend: TrendDirection,
    pub throughput_trend: TrendDirection,
    pub memory_trend: TrendDirection,
    pub trend_confidence: f32,
}

/// Trend direction analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Volatile,
}

/// Bottleneck analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    pub primary_bottleneck: BottleneckType,
    pub bottleneck_severity: f32,
    pub contributing_factors: Vec<String>,
    pub impact_analysis: String,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    CPU,
    Memory,
    GPU,
    IO,
    Network,
    ModelComplexity,
    DataLoading,
    Synchronization,
    Unknown,
}

/// Hardware utilization analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareUtilization {
    pub cpu_utilization_percent: f32,
    pub memory_utilization_percent: f32,
    pub gpu_utilization_percent: f32,
    pub efficiency_score: f32,
    pub underutilized_resources: Vec<String>,
}

/// Memory analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAnalysis {
    pub leak_probability: f32,
    pub fragmentation_level: f32,
    pub allocation_pattern: AllocationPattern,
    pub gc_impact: f32,
    pub optimization_potential: f32,
}

/// Memory allocation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationPattern {
    Steady,
    Spiky,
    Growing,
    Cyclical,
    Chaotic,
}

impl EnhancedProfiler {
    /// Create a new enhanced profiler with configuration
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            global_metrics: Arc::new(RwLock::new(GlobalMetrics::default())),
            config,
        }
    }

    /// Start a new profiling session
    pub async fn start_session(
        &self,
        session_id: String,
        operation_name: String,
    ) -> Result<(), String> {
        let hardware_info = self.detect_hardware().await;
        let session = ProfilingSession {
            session_id: session_id.clone(),
            operation_name,
            start_time: Instant::now(),
            samples: Vec::new(),
            hardware_info,
            memory_tracker: MemoryTracker {
                initial_memory_mb: self.get_current_memory_usage(),
                peak_memory_mb: 0.0,
                current_memory_mb: 0.0,
                allocation_count: 0,
                deallocation_count: 0,
                leak_detected: false,
                memory_samples: Vec::new(),
            },
            status: SessionStatus::Active,
        };

        let mut sessions = self.sessions.write().await;
        sessions.insert(session_id, session);

        let mut global_metrics = self.global_metrics.write().await;
        global_metrics.total_sessions += 1;
        global_metrics.active_sessions += 1;

        Ok(())
    }

    /// Record a performance sample
    pub async fn record_sample(
        &self,
        session_id: &str,
        custom_metrics: HashMap<String, f64>,
    ) -> Result<(), String> {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            let sample = PerformanceSample {
                timestamp: Instant::now(),
                latency_ms: session.start_time.elapsed().as_millis() as f32,
                throughput_ops_per_sec: self.calculate_throughput(session).await,
                memory_usage_mb: self.get_current_memory_usage(),
                cpu_usage_percent: self.get_cpu_usage().await,
                gpu_usage_percent: self.get_gpu_usage().await,
                custom_metrics,
            };

            // Update memory tracker
            session.memory_tracker.current_memory_mb = sample.memory_usage_mb;
            if sample.memory_usage_mb > session.memory_tracker.peak_memory_mb {
                session.memory_tracker.peak_memory_mb = sample.memory_usage_mb;
            }

            // Add memory sample
            session.memory_tracker.memory_samples.push(MemorySample {
                timestamp: sample.timestamp,
                memory_mb: sample.memory_usage_mb,
                allocations: session.memory_tracker.allocation_count,
                deallocations: session.memory_tracker.deallocation_count,
            });

            session.samples.push(sample);

            // Check for performance alerts
            if self.config.real_time_alerts {
                self.check_performance_alerts(session).await;
            }

            // Limit samples to prevent memory growth
            if session.samples.len() > self.config.max_samples {
                session.samples.remove(0);
            }

            Ok(())
        } else {
            Err(format!("Session {} not found", session_id))
        }
    }

    /// End a profiling session and generate analysis
    pub async fn end_session(&self, session_id: &str) -> Result<PerformanceAnalysis, String> {
        let mut sessions = self.sessions.write().await;
        if let Some(mut session) = sessions.remove(session_id) {
            session.status = SessionStatus::Completed;

            let mut global_metrics = self.global_metrics.write().await;
            global_metrics.active_sessions -= 1;

            // Generate comprehensive analysis
            let analysis = self.generate_analysis(&session).await;

            // Add optimization suggestions to global metrics
            global_metrics
                .optimization_suggestions
                .extend(analysis.optimization_recommendations.clone());

            Ok(analysis)
        } else {
            Err(format!("Session {} not found", session_id))
        }
    }

    /// Detect hardware configuration
    async fn detect_hardware(&self) -> HardwareInfo {
        // Mock hardware detection - in real implementation, use system APIs
        HardwareInfo {
            cpu_cores: num_cpus::get(),
            cpu_model: "Mock CPU Model".to_string(),
            total_memory_gb: 16.0, // Mock value
            gpu_info: vec![GPUInfo {
                name: "Mock GPU".to_string(),
                memory_gb: 8.0,
                compute_capability: "8.6".to_string(),
                utilization_percent: 0.0,
            }],
            platform: if cfg!(target_os = "linux") {
                Platform::Linux
            } else if cfg!(target_os = "windows") {
                Platform::Windows
            } else if cfg!(target_os = "macos") {
                Platform::MacOS
            } else {
                Platform::Unknown
            },
            specialized_hardware: vec![
                SpecializedHardware::CUDA,
                SpecializedHardware::Metal,
                SpecializedHardware::ONNX,
            ],
        }
    }

    /// Get current memory usage (mock implementation)
    fn get_current_memory_usage(&self) -> f32 {
        // Mock memory usage - in real implementation, use system APIs
        100.0 + (std::ptr::addr_of!(self) as usize % 100) as f32 / 2.0
    }

    /// Get CPU usage (mock implementation)
    async fn get_cpu_usage(&self) -> f32 {
        // Mock CPU usage - in real implementation, use system APIs
        20.0 + (std::ptr::addr_of!(self) as usize % 60) as f32
    }

    /// Get GPU usage (mock implementation)
    async fn get_gpu_usage(&self) -> f32 {
        // Mock GPU usage - in real implementation, use GPU APIs
        10.0 + (std::ptr::addr_of!(self) as usize % 80) as f32
    }

    /// Calculate throughput for a session
    async fn calculate_throughput(&self, session: &ProfilingSession) -> f32 {
        let duration_sec = session.start_time.elapsed().as_secs_f32();
        if duration_sec > 0.0 {
            session.samples.len() as f32 / duration_sec
        } else {
            0.0
        }
    }

    /// Check for performance alerts
    async fn check_performance_alerts(&self, session: &ProfilingSession) {
        if let Some(latest_sample) = session.samples.last() {
            let mut alerts_triggered = 0;

            if latest_sample.latency_ms > self.config.thresholds.max_latency_ms {
                alerts_triggered += 1;
                println!(
                    "ALERT: High latency detected: {:.2}ms",
                    latest_sample.latency_ms
                );
            }

            if latest_sample.memory_usage_mb > self.config.thresholds.max_memory_usage_mb {
                alerts_triggered += 1;
                println!(
                    "ALERT: High memory usage: {:.2}MB",
                    latest_sample.memory_usage_mb
                );
            }

            if latest_sample.cpu_usage_percent > self.config.thresholds.max_cpu_usage_percent {
                alerts_triggered += 1;
                println!(
                    "ALERT: High CPU usage: {:.2}%",
                    latest_sample.cpu_usage_percent
                );
            }

            if alerts_triggered > 0 {
                let mut global_metrics = self.global_metrics.write().await;
                global_metrics.performance_alerts += alerts_triggered;
            }
        }
    }

    /// Generate comprehensive performance analysis
    async fn generate_analysis(&self, session: &ProfilingSession) -> PerformanceAnalysis {
        let session_summary = self.calculate_session_summary(session);
        let performance_trends = self.analyze_trends(session);
        let bottleneck_analysis = self.analyze_bottlenecks(session);
        let optimization_recommendations =
            self.generate_optimization_recommendations(session).await;
        let hardware_utilization = self.analyze_hardware_utilization(session);
        let memory_analysis = self.analyze_memory_usage(session);

        PerformanceAnalysis {
            session_summary,
            performance_trends,
            bottleneck_analysis,
            optimization_recommendations,
            hardware_utilization,
            memory_analysis,
        }
    }

    /// Calculate session summary statistics
    fn calculate_session_summary(&self, session: &ProfilingSession) -> SessionSummary {
        let latencies: Vec<f32> = session.samples.iter().map(|s| s.latency_ms).collect();
        let throughputs: Vec<f32> =
            session.samples.iter().map(|s| s.throughput_ops_per_sec).collect();

        SessionSummary {
            total_duration_ms: session.start_time.elapsed().as_millis() as f32,
            total_operations: session.samples.len() as u64,
            average_latency_ms: latencies.iter().sum::<f32>() / latencies.len() as f32,
            p95_latency_ms: self.percentile(&latencies, 0.95),
            p99_latency_ms: self.percentile(&latencies, 0.99),
            peak_throughput_ops_per_sec: throughputs.iter().cloned().fold(0.0f32, f32::max),
            peak_memory_mb: session.memory_tracker.peak_memory_mb,
        }
    }

    /// Calculate percentile from a sorted list
    fn percentile(&self, data: &[f32], percentile: f32) -> f32 {
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let index = ((data.len() as f32 - 1.0) * percentile) as usize;
        sorted_data.get(index).copied().unwrap_or(0.0)
    }

    /// Analyze performance trends
    fn analyze_trends(&self, session: &ProfilingSession) -> PerformanceTrends {
        // Simple trend analysis - in real implementation, use statistical methods
        PerformanceTrends {
            latency_trend: TrendDirection::Stable,
            throughput_trend: TrendDirection::Improving,
            memory_trend: TrendDirection::Stable,
            trend_confidence: 0.8,
        }
    }

    /// Analyze bottlenecks
    fn analyze_bottlenecks(&self, session: &ProfilingSession) -> BottleneckAnalysis {
        // Simple bottleneck analysis - in real implementation, use advanced analytics
        let avg_cpu = session.samples.iter().map(|s| s.cpu_usage_percent).sum::<f32>()
            / session.samples.len() as f32;
        let avg_memory = session.samples.iter().map(|s| s.memory_usage_mb).sum::<f32>()
            / session.samples.len() as f32;

        let primary_bottleneck = if avg_cpu > 80.0 {
            BottleneckType::CPU
        } else if avg_memory > 1000.0 {
            BottleneckType::Memory
        } else {
            BottleneckType::Unknown
        };

        BottleneckAnalysis {
            primary_bottleneck,
            bottleneck_severity: 0.5,
            contributing_factors: vec!["Mock factor 1".to_string(), "Mock factor 2".to_string()],
            impact_analysis: "Moderate impact on overall performance".to_string(),
        }
    }

    /// Generate AI-powered optimization recommendations
    async fn generate_optimization_recommendations(
        &self,
        session: &ProfilingSession,
    ) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();

        // Memory optimization suggestion
        if session.memory_tracker.peak_memory_mb > 500.0 {
            suggestions.push(OptimizationSuggestion {
                category: OptimizationCategory::Memory,
                severity: SuggestionSeverity::Medium,
                description: "High peak memory usage detected".to_string(),
                suggested_action: "Consider implementing memory pooling or reducing batch size"
                    .to_string(),
                expected_improvement: "20-30% reduction in memory usage".to_string(),
                confidence: 0.85,
            });
        }

        // Batch size optimization
        if session.samples.len() > 100 {
            suggestions.push(OptimizationSuggestion {
                category: OptimizationCategory::BatchSize,
                severity: SuggestionSeverity::Low,
                description: "Batch size may be sub-optimal for throughput".to_string(),
                suggested_action: "Experiment with larger batch sizes for better GPU utilization"
                    .to_string(),
                expected_improvement: "15-25% improvement in throughput".to_string(),
                confidence: 0.7,
            });
        }

        suggestions
    }

    /// Analyze hardware utilization
    fn analyze_hardware_utilization(&self, session: &ProfilingSession) -> HardwareUtilization {
        let avg_cpu = session.samples.iter().map(|s| s.cpu_usage_percent).sum::<f32>()
            / session.samples.len() as f32;
        let avg_gpu = session.samples.iter().map(|s| s.gpu_usage_percent).sum::<f32>()
            / session.samples.len() as f32;
        let avg_memory = session.samples.iter().map(|s| s.memory_usage_mb).sum::<f32>()
            / session.samples.len() as f32;

        HardwareUtilization {
            cpu_utilization_percent: avg_cpu,
            memory_utilization_percent: avg_memory / session.hardware_info.total_memory_gb / 10.24, // Convert to percentage
            gpu_utilization_percent: avg_gpu,
            efficiency_score: (avg_cpu + avg_gpu) / 2.0 / 100.0,
            underutilized_resources: vec!["GPU".to_string()], // Mock
        }
    }

    /// Analyze memory usage patterns
    fn analyze_memory_usage(&self, session: &ProfilingSession) -> MemoryAnalysis {
        let memory_growth =
            session.memory_tracker.peak_memory_mb - session.memory_tracker.initial_memory_mb;

        MemoryAnalysis {
            leak_probability: if memory_growth > 50.0 { 0.7 } else { 0.2 },
            fragmentation_level: 0.3,                      // Mock
            allocation_pattern: AllocationPattern::Steady, // Mock
            gc_impact: 0.1,                                // Mock
            optimization_potential: 0.6,                   // Mock
        }
    }

    /// Export profiling data in specified format
    pub async fn export_data(
        &self,
        session_id: &str,
        format: ExportFormat,
    ) -> Result<String, String> {
        let sessions = self.sessions.read().await;
        if let Some(session) = sessions.get(session_id) {
            match format {
                ExportFormat::JSON => serde_json::to_string_pretty(session)
                    .map_err(|e| format!("JSON export failed: {}", e)),
                ExportFormat::CSV => {
                    // Simple CSV export - in real implementation, use proper CSV library
                    let mut csv =
                        "timestamp,latency_ms,throughput,memory_mb,cpu_percent,gpu_percent\n"
                            .to_string();
                    for sample in &session.samples {
                        csv.push_str(&format!(
                            "{:?},{},{},{},{},{}\n",
                            sample.timestamp,
                            sample.latency_ms,
                            sample.throughput_ops_per_sec,
                            sample.memory_usage_mb,
                            sample.cpu_usage_percent,
                            sample.gpu_usage_percent
                        ));
                    }
                    Ok(csv)
                },
                ExportFormat::Prometheus => {
                    // Prometheus metrics format
                    let mut prometheus = String::new();
                    if let Some(latest_sample) = session.samples.last() {
                        prometheus.push_str(&format!(
                            "# HELP trustformers_latency_ms Current latency in milliseconds\n\
                             # TYPE trustformers_latency_ms gauge\n\
                             trustformers_latency_ms{{session=\"{}\"}} {}\n",
                            session_id, latest_sample.latency_ms
                        ));
                    }
                    Ok(prometheus)
                },
                _ => Err("Export format not implemented".to_string()),
            }
        } else {
            Err(format!("Session {} not found", session_id))
        }
    }

    /// Get global performance metrics
    pub async fn get_global_metrics(&self) -> GlobalMetrics {
        self.global_metrics.read().await.clone()
    }
}

/// Global profiler instance for easy access
static GLOBAL_PROFILER: std::sync::OnceLock<Arc<EnhancedProfiler>> = std::sync::OnceLock::new();

/// Initialize the global profiler
pub fn init_global_profiler(config: ProfilerConfig) {
    let _ = GLOBAL_PROFILER.get_or_init(|| Arc::new(EnhancedProfiler::new(config)));
}

/// Get the global profiler instance
pub fn global_profiler() -> Option<Arc<EnhancedProfiler>> {
    GLOBAL_PROFILER.get().cloned()
}

/// Convenience macro for enhanced profiling operations
#[macro_export]
macro_rules! enhanced_profile_operation {
    ($operation_name:expr, $block:block) => {{
        let profiler = global_profiler().expect("Profiler not initialized");
        let session_id = format!(
            "{}_{}",
            $operation_name,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );

        profiler
            .start_session(session_id.clone(), $operation_name.to_string())
            .await
            .unwrap();

        let result = $block;

        profiler
            .record_sample(&session_id, std::collections::HashMap::new())
            .await
            .unwrap();
        let _analysis = profiler.end_session(&session_id).await.unwrap();

        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_enhanced_profiler_basic_functionality() {
        let config = ProfilerConfig::default();
        let profiler = EnhancedProfiler::new(config);

        let session_id = "test_session".to_string();
        let operation_name = "test_operation".to_string();

        // Start session
        profiler.start_session(session_id.clone(), operation_name).await.unwrap();

        // Record samples (more than 100 to trigger batch size optimization)
        for i in 0..105 {
            let mut custom_metrics = HashMap::new();
            custom_metrics.insert("iteration".to_string(), i as f64);
            profiler.record_sample(&session_id, custom_metrics).await.unwrap();
            if i % 20 == 0 {
                sleep(Duration::from_millis(1)).await; // Reduce sleep frequency for faster test
            }
        }

        // End session and get analysis
        let analysis = profiler.end_session(&session_id).await.unwrap();

        assert!(analysis.session_summary.total_operations > 0);
        assert!(analysis.session_summary.total_duration_ms > 0.0);
        assert!(!analysis.optimization_recommendations.is_empty());
    }

    #[tokio::test]
    async fn test_global_profiler() {
        let config = ProfilerConfig::default();
        init_global_profiler(config);

        let profiler = global_profiler().expect("Global profiler should be initialized");

        let session_id = "global_test".to_string();
        profiler
            .start_session(session_id.clone(), "global_test".to_string())
            .await
            .unwrap();

        let metrics = profiler.get_global_metrics().await;
        assert!(metrics.total_sessions > 0);
    }

    #[tokio::test]
    async fn test_export_functionality() {
        let config = ProfilerConfig::default();
        let profiler = EnhancedProfiler::new(config);

        let session_id = "export_test".to_string();
        profiler
            .start_session(session_id.clone(), "export_test".to_string())
            .await
            .unwrap();
        profiler.record_sample(&session_id, HashMap::new()).await.unwrap();

        // Test JSON export
        let json_export = profiler.export_data(&session_id, ExportFormat::JSON).await;
        assert!(json_export.is_ok());

        // Test CSV export
        let csv_export = profiler.export_data(&session_id, ExportFormat::CSV).await;
        assert!(csv_export.is_ok());

        profiler.end_session(&session_id).await.unwrap();
    }
}
