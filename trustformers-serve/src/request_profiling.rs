// Allow dead code for infrastructure under development
#![allow(dead_code)]

//! Request Profiling Module
//!
//! Provides comprehensive request profiling capabilities for performance analysis,
//! including detailed timing measurements, resource usage tracking, bottleneck
//! identification, and performance optimization recommendations.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{info, trace, warn};
use uuid::Uuid;

/// Request profiling service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestProfilingConfig {
    /// Enable request profiling
    pub enabled: bool,

    /// Maximum number of profiles to keep in memory
    pub max_profiles_in_memory: usize,

    /// Enable detailed timing measurements
    pub enable_detailed_timing: bool,

    /// Enable resource usage tracking
    pub enable_resource_tracking: bool,

    /// Enable call stack tracking
    pub enable_call_stack_tracking: bool,

    /// Enable memory profiling
    pub enable_memory_profiling: bool,

    /// Enable CPU profiling
    pub enable_cpu_profiling: bool,

    /// Enable I/O profiling
    pub enable_io_profiling: bool,

    /// Sampling rate for profiling (0.0 to 1.0)
    pub sampling_rate: f64,

    /// Minimum request duration to profile (milliseconds)
    pub min_duration_to_profile_ms: u64,

    /// Enable performance recommendations
    pub enable_performance_recommendations: bool,

    /// Enable profile aggregation
    pub enable_profile_aggregation: bool,

    /// Aggregation window in seconds
    pub aggregation_window_secs: u64,

    /// Enable profile export
    pub enable_profile_export: bool,

    /// Profile export format
    pub profile_export_format: ProfileExportFormat,

    /// Enable flame graph generation
    pub enable_flame_graphs: bool,

    /// Enable bottleneck detection
    pub enable_bottleneck_detection: bool,

    /// Bottleneck threshold percentage
    pub bottleneck_threshold_percent: f64,
}

impl Default for RequestProfilingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_profiles_in_memory: 10000,
            enable_detailed_timing: true,
            enable_resource_tracking: true,
            enable_call_stack_tracking: true,
            enable_memory_profiling: true,
            enable_cpu_profiling: true,
            enable_io_profiling: true,
            sampling_rate: 0.1, // 10% sampling
            min_duration_to_profile_ms: 10,
            enable_performance_recommendations: true,
            enable_profile_aggregation: true,
            aggregation_window_secs: 300,
            enable_profile_export: true,
            profile_export_format: ProfileExportFormat::JSON,
            enable_flame_graphs: true,
            enable_bottleneck_detection: true,
            bottleneck_threshold_percent: 20.0,
        }
    }
}

/// Profile export formats
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ProfileExportFormat {
    JSON,
    CSV,
    Protobuf,
    FlameGraph,
    Pprof,
    Jaeger,
}

/// Request profile containing comprehensive performance data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestProfile {
    /// Unique profile identifier
    pub id: String,

    /// Request identifier
    pub request_id: String,

    /// Request type/endpoint
    pub request_type: String,

    /// Request start time
    pub start_time: SystemTime,

    /// Request end time
    pub end_time: Option<SystemTime>,

    /// Total request duration
    pub total_duration: Option<Duration>,

    /// Detailed timing breakdown
    pub timing_breakdown: TimingBreakdown,

    /// Resource usage metrics
    pub resource_usage: ResourceUsage,

    /// Call stack information
    pub call_stack: Vec<CallStackEntry>,

    /// Memory profiling data
    pub memory_profile: Option<MemoryProfile>,

    /// CPU profiling data
    pub cpu_profile: Option<CpuProfile>,

    /// I/O profiling data
    pub io_profile: Option<IoProfile>,

    /// Request metadata
    pub metadata: HashMap<String, String>,

    /// Performance issues detected
    pub performance_issues: Vec<PerformanceIssue>,

    /// Performance recommendations
    pub recommendations: Vec<PerformanceRecommendation>,

    /// Error information if request failed
    pub error_info: Option<ErrorInfo>,

    /// Profile status
    pub status: ProfileStatus,
}

/// Detailed timing breakdown for different phases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingBreakdown {
    /// Request parsing/validation time
    pub parsing_duration: Option<Duration>,

    /// Authentication/authorization time
    pub auth_duration: Option<Duration>,

    /// Model loading time
    pub model_load_duration: Option<Duration>,

    /// Input preprocessing time
    pub preprocessing_duration: Option<Duration>,

    /// Model inference time
    pub inference_duration: Option<Duration>,

    /// Output postprocessing time
    pub postprocessing_duration: Option<Duration>,

    /// Response serialization time
    pub serialization_duration: Option<Duration>,

    /// Network I/O time
    pub network_io_duration: Option<Duration>,

    /// Database query time
    pub database_duration: Option<Duration>,

    /// Cache operations time
    pub cache_duration: Option<Duration>,

    /// Custom timing points
    pub custom_timings: HashMap<String, Duration>,
}

impl Default for TimingBreakdown {
    fn default() -> Self {
        Self {
            parsing_duration: None,
            auth_duration: None,
            model_load_duration: None,
            preprocessing_duration: None,
            inference_duration: None,
            postprocessing_duration: None,
            serialization_duration: None,
            network_io_duration: None,
            database_duration: None,
            cache_duration: None,
            custom_timings: HashMap::new(),
        }
    }
}

/// Resource usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Peak CPU usage percentage
    pub peak_cpu_percent: Option<f64>,

    /// Average CPU usage percentage
    pub avg_cpu_percent: Option<f64>,

    /// Peak memory usage in bytes
    pub peak_memory_bytes: Option<usize>,

    /// Average memory usage in bytes
    pub avg_memory_bytes: Option<usize>,

    /// GPU memory usage in bytes
    pub gpu_memory_bytes: Option<usize>,

    /// GPU utilization percentage
    pub gpu_utilization_percent: Option<f64>,

    /// Disk I/O bytes read
    pub disk_read_bytes: Option<u64>,

    /// Disk I/O bytes written
    pub disk_write_bytes: Option<u64>,

    /// Network bytes received
    pub network_rx_bytes: Option<u64>,

    /// Network bytes transmitted
    pub network_tx_bytes: Option<u64>,

    /// Number of file descriptors used
    pub file_descriptors: Option<u32>,

    /// Number of threads used
    pub thread_count: Option<u32>,
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            peak_cpu_percent: None,
            avg_cpu_percent: None,
            peak_memory_bytes: None,
            avg_memory_bytes: None,
            gpu_memory_bytes: None,
            gpu_utilization_percent: None,
            disk_read_bytes: None,
            disk_write_bytes: None,
            network_rx_bytes: None,
            network_tx_bytes: None,
            file_descriptors: None,
            thread_count: None,
        }
    }
}

/// Call stack entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallStackEntry {
    /// Function/method name
    pub function_name: String,

    /// File path
    pub file_path: Option<String>,

    /// Line number
    pub line_number: Option<u32>,

    /// Module name
    pub module_name: Option<String>,

    /// Entry time
    pub entry_time: SystemTime,

    /// Exit time
    pub exit_time: Option<SystemTime>,

    /// Duration spent in this function
    pub duration: Option<Duration>,

    /// Self time (excluding called functions)
    pub self_time: Option<Duration>,

    /// Number of calls to this function
    pub call_count: u32,

    /// Child function calls
    pub children: Vec<CallStackEntry>,
}

/// Memory profiling data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProfile {
    /// Memory allocations
    pub allocations: Vec<MemoryAllocation>,

    /// Peak heap usage
    pub peak_heap_bytes: usize,

    /// Peak stack usage
    pub peak_stack_bytes: Option<usize>,

    /// Total allocations count
    pub total_allocations: u64,

    /// Total deallocations count
    pub total_deallocations: u64,

    /// Memory leaks detected
    pub memory_leaks: Vec<MemoryLeak>,

    /// Garbage collection statistics
    pub gc_stats: Option<GcStats>,
}

/// Memory allocation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAllocation {
    /// Allocation size
    pub size: usize,

    /// Allocation timestamp
    pub timestamp: SystemTime,

    /// Stack trace at allocation
    pub stack_trace: Vec<String>,

    /// Allocation type/category
    pub allocation_type: String,

    /// Whether allocation was freed
    pub freed: bool,

    /// Free timestamp
    pub free_timestamp: Option<SystemTime>,
}

/// Memory leak information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLeak {
    /// Leaked memory size
    pub size: usize,

    /// Allocation stack trace
    pub allocation_stack: Vec<String>,

    /// Leak detection timestamp
    pub detected_at: SystemTime,
}

/// Garbage collection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcStats {
    /// Number of GC cycles
    pub gc_cycles: u64,

    /// Total GC time
    pub total_gc_time: Duration,

    /// Average GC time
    pub avg_gc_time: Duration,

    /// Objects collected
    pub objects_collected: u64,

    /// Memory freed by GC
    pub memory_freed: usize,
}

/// CPU profiling data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuProfile {
    /// CPU samples
    pub samples: Vec<CpuSample>,

    /// Sampling frequency (Hz)
    pub sampling_frequency: f64,

    /// Total sample count
    pub total_samples: u64,

    /// Hot functions (most CPU time)
    pub hot_functions: Vec<HotFunction>,

    /// CPU usage over time
    pub cpu_usage_timeline: Vec<CpuUsagePoint>,
}

/// CPU sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuSample {
    /// Sample timestamp
    pub timestamp: SystemTime,

    /// Stack trace
    pub stack_trace: Vec<String>,

    /// CPU time spent
    pub cpu_time: Duration,

    /// Thread ID
    pub thread_id: Option<u64>,
}

/// Hot function information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotFunction {
    /// Function name
    pub function_name: String,

    /// Total CPU time
    pub total_time: Duration,

    /// Self CPU time
    pub self_time: Duration,

    /// Percentage of total CPU time
    pub percentage: f64,

    /// Number of samples
    pub sample_count: u64,
}

/// CPU usage point in timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuUsagePoint {
    /// Timestamp
    pub timestamp: SystemTime,

    /// CPU usage percentage
    pub cpu_percent: f64,
}

/// I/O profiling data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoProfile {
    /// File I/O operations
    pub file_operations: Vec<FileOperation>,

    /// Network I/O operations
    pub network_operations: Vec<NetworkOperation>,

    /// Total bytes read
    pub total_bytes_read: u64,

    /// Total bytes written
    pub total_bytes_written: u64,

    /// I/O wait time
    pub io_wait_time: Duration,

    /// I/O bandwidth utilization
    pub bandwidth_utilization: Option<f64>,
}

/// File I/O operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileOperation {
    /// Operation type
    pub operation_type: IoOperationType,

    /// File path
    pub file_path: String,

    /// Operation start time
    pub start_time: SystemTime,

    /// Operation duration
    pub duration: Duration,

    /// Bytes transferred
    pub bytes: u64,

    /// File offset
    pub offset: Option<u64>,
}

/// Network I/O operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkOperation {
    /// Operation type
    pub operation_type: IoOperationType,

    /// Remote address
    pub remote_address: String,

    /// Local port
    pub local_port: Option<u16>,

    /// Operation start time
    pub start_time: SystemTime,

    /// Operation duration
    pub duration: Duration,

    /// Bytes transferred
    pub bytes: u64,

    /// Protocol used
    pub protocol: String,
}

/// I/O operation types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum IoOperationType {
    Read,
    Write,
    Open,
    Close,
    Seek,
    Flush,
    Connect,
    Accept,
    Send,
    Receive,
}

/// Performance issue detected during profiling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceIssue {
    /// Issue type
    pub issue_type: PerformanceIssueType,

    /// Issue severity
    pub severity: IssueSeverity,

    /// Issue description
    pub description: String,

    /// Location where issue was detected
    pub location: Option<String>,

    /// Metric value that triggered the issue
    pub metric_value: Option<f64>,

    /// Threshold that was exceeded
    pub threshold: Option<f64>,

    /// Impact assessment
    pub impact: String,

    /// Detection timestamp
    pub detected_at: SystemTime,
}

/// Performance issue types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum PerformanceIssueType {
    HighLatency,
    HighCpuUsage,
    HighMemoryUsage,
    MemoryLeak,
    SlowIo,
    DeadlockDetected,
    ThrashingDetected,
    IneffientAlgorithm,
    ResourceContention,
    UnoptimizedQuery,
    ExcessiveGarbageCollection,
    HotSpot,
}

/// Issue severity levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum IssueSeverity {
    Critical = 4,
    High = 3,
    Medium = 2,
    Low = 1,
    Info = 0,
}

/// Performance recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,

    /// Priority level
    pub priority: IssueSeverity,

    /// Recommendation description
    pub description: String,

    /// Specific action to take
    pub action: String,

    /// Expected impact
    pub expected_impact: String,

    /// Implementation difficulty
    pub difficulty: DifficultyLevel,

    /// Related performance issues
    pub related_issues: Vec<String>,

    /// Code locations to modify
    pub code_locations: Vec<String>,
}

/// Recommendation types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum RecommendationType {
    AlgorithmOptimization,
    MemoryOptimization,
    CpuOptimization,
    IoOptimization,
    CachingStrategy,
    CodeRefactoring,
    ResourceScaling,
    DatabaseOptimization,
    NetworkOptimization,
    ConfigurationTuning,
}

/// Implementation difficulty levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum DifficultyLevel {
    Easy = 1,
    Medium = 2,
    Hard = 3,
    Expert = 4,
}

/// Error information for failed requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorInfo {
    /// Error type
    pub error_type: String,

    /// Error message
    pub error_message: String,

    /// Stack trace
    pub stack_trace: Vec<String>,

    /// Error location
    pub location: Option<String>,

    /// Error timestamp
    pub timestamp: SystemTime,

    /// Error category
    pub category: ErrorCategory,
}

/// Error categories
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ErrorCategory {
    Timeout,
    ResourceExhaustion,
    InvalidInput,
    InternalError,
    NetworkError,
    DatabaseError,
    AuthenticationError,
    AuthorizationError,
    RateLimitExceeded,
    ServiceUnavailable,
}

/// Profile status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ProfileStatus {
    Active,
    Completed,
    Failed,
    Cancelled,
}

/// Aggregated profile statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedProfileStats {
    /// Time window for aggregation
    pub window_start: SystemTime,
    pub window_end: SystemTime,

    /// Number of profiles in aggregation
    pub profile_count: u64,

    /// Request type statistics
    pub request_type_stats: HashMap<String, RequestTypeStats>,

    /// Overall performance metrics
    pub overall_metrics: OverallMetrics,

    /// Performance trends
    pub trends: PerformanceTrends,

    /// Top performance issues
    pub top_issues: Vec<PerformanceIssue>,

    /// Recommendations summary
    pub recommendations_summary: Vec<PerformanceRecommendation>,
}

/// Statistics for a specific request type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestTypeStats {
    /// Request count
    pub count: u64,

    /// Average duration
    pub avg_duration: Duration,

    /// Median duration
    pub median_duration: Duration,

    /// 95th percentile duration
    pub p95_duration: Duration,

    /// 99th percentile duration
    pub p99_duration: Duration,

    /// Error rate
    pub error_rate: f64,

    /// Average resource usage
    pub avg_resource_usage: ResourceUsage,
}

/// Overall performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallMetrics {
    /// Total requests processed
    pub total_requests: u64,

    /// Successful requests
    pub successful_requests: u64,

    /// Failed requests
    pub failed_requests: u64,

    /// Overall error rate
    pub error_rate: f64,

    /// Average request duration
    pub avg_duration: Duration,

    /// Throughput (requests per second)
    pub throughput: f64,

    /// Resource utilization summary
    pub resource_utilization: ResourceUsage,
}

/// Performance trends over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    /// Latency trend
    pub latency_trend: Trend,

    /// Throughput trend
    pub throughput_trend: Trend,

    /// Error rate trend
    pub error_rate_trend: Trend,

    /// Resource usage trends
    pub cpu_usage_trend: Trend,
    pub memory_usage_trend: Trend,
}

/// Trend direction and magnitude
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Trend {
    Improving,
    Stable,
    Degrading,
    Volatile,
}

/// Profiling statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingStats {
    /// Total profiles created
    pub total_profiles: u64,

    /// Active profiles count
    pub active_profiles: u64,

    /// Completed profiles count
    pub completed_profiles: u64,

    /// Failed profiles count
    pub failed_profiles: u64,

    /// Average profiling overhead
    pub avg_overhead_percent: f64,

    /// Memory usage by profiler
    pub profiler_memory_usage: usize,

    /// Sampling statistics
    pub sampling_rate_actual: f64,

    /// Profile export statistics
    pub exports_count: u64,

    /// Profile aggregations count
    pub aggregations_count: u64,
}

/// Request profiling errors
#[derive(Debug, Error)]
pub enum RequestProfilingError {
    #[error("Profile not found: {0}")]
    ProfileNotFound(String),

    #[error("Profiling disabled")]
    ProfilingDisabled,

    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    #[error("Export failed: {0}")]
    ExportFailed(String),

    #[error("Aggregation failed: {0}")]
    AggregationFailed(String),

    #[error("Sampling error: {0}")]
    SamplingError(String),

    #[error("Internal profiling error: {0}")]
    Internal(#[from] anyhow::Error),
}

/// Request profiling service
pub struct RequestProfilingService {
    config: RequestProfilingConfig,
    active_profiles: Arc<RwLock<HashMap<String, RequestProfile>>>,
    completed_profiles: Arc<RwLock<VecDeque<RequestProfile>>>,
    aggregated_stats: Arc<RwLock<Vec<AggregatedProfileStats>>>,
    stats: Arc<RwLock<ProfilingStats>>,
    profile_counter: AtomicU64,
}

impl RequestProfilingService {
    /// Create a new request profiling service
    pub fn new(config: RequestProfilingConfig) -> Self {
        Self {
            config,
            active_profiles: Arc::new(RwLock::new(HashMap::new())),
            completed_profiles: Arc::new(RwLock::new(VecDeque::new())),
            aggregated_stats: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(ProfilingStats {
                total_profiles: 0,
                active_profiles: 0,
                completed_profiles: 0,
                failed_profiles: 0,
                avg_overhead_percent: 0.0,
                profiler_memory_usage: 0,
                sampling_rate_actual: 0.0,
                exports_count: 0,
                aggregations_count: 0,
            })),
            profile_counter: AtomicU64::new(0),
        }
    }

    /// Start profiling a request
    pub async fn start_profile(
        &self,
        request_id: String,
        request_type: String,
    ) -> Result<String, RequestProfilingError> {
        if !self.config.enabled {
            return Err(RequestProfilingError::ProfilingDisabled);
        }

        // Check sampling rate
        use scirs2_core::random::*;
        let mut rng = thread_rng();
        if rng.random::<f64>() > self.config.sampling_rate {
            return Err(RequestProfilingError::SamplingError(
                "Request not sampled".to_string(),
            ));
        }

        let profile_id = Uuid::new_v4().to_string();
        let profile = RequestProfile {
            id: profile_id.clone(),
            request_id,
            request_type,
            start_time: SystemTime::now(),
            end_time: None,
            total_duration: None,
            timing_breakdown: TimingBreakdown::default(),
            resource_usage: ResourceUsage::default(),
            call_stack: Vec::new(),
            memory_profile: if self.config.enable_memory_profiling {
                Some(MemoryProfile {
                    allocations: Vec::new(),
                    peak_heap_bytes: 0,
                    peak_stack_bytes: None,
                    total_allocations: 0,
                    total_deallocations: 0,
                    memory_leaks: Vec::new(),
                    gc_stats: None,
                })
            } else {
                None
            },
            cpu_profile: if self.config.enable_cpu_profiling {
                Some(CpuProfile {
                    samples: Vec::new(),
                    sampling_frequency: 100.0,
                    total_samples: 0,
                    hot_functions: Vec::new(),
                    cpu_usage_timeline: Vec::new(),
                })
            } else {
                None
            },
            io_profile: if self.config.enable_io_profiling {
                Some(IoProfile {
                    file_operations: Vec::new(),
                    network_operations: Vec::new(),
                    total_bytes_read: 0,
                    total_bytes_written: 0,
                    io_wait_time: Duration::from_secs(0),
                    bandwidth_utilization: None,
                })
            } else {
                None
            },
            metadata: HashMap::new(),
            performance_issues: Vec::new(),
            recommendations: Vec::new(),
            error_info: None,
            status: ProfileStatus::Active,
        };

        // Capture request_id before moving profile
        let request_id = profile.request_id.clone();
        self.active_profiles.write().await.insert(profile_id.clone(), profile);

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.total_profiles += 1;
        stats.active_profiles += 1;
        self.profile_counter.fetch_add(1, Ordering::SeqCst);

        info!(
            "Started profiling request: {} with profile ID: {}",
            request_id, profile_id
        );
        Ok(profile_id)
    }

    /// Complete a profile
    pub async fn complete_profile(&self, profile_id: &str) -> Result<(), RequestProfilingError> {
        let mut active_profiles = self.active_profiles.write().await;

        if let Some(mut profile) = active_profiles.remove(profile_id) {
            profile.end_time = Some(SystemTime::now());
            profile.total_duration =
                profile.end_time.and_then(|end| end.duration_since(profile.start_time).ok());
            profile.status = ProfileStatus::Completed;

            // Check if profile meets minimum duration threshold
            if let Some(duration) = profile.total_duration {
                if duration.as_millis() < self.config.min_duration_to_profile_ms as u128 {
                    // Skip profiling for short requests
                    self.update_completion_stats(false).await;
                    return Ok(());
                }
            }

            // Analyze performance and generate recommendations
            if self.config.enable_performance_recommendations {
                self.analyze_performance(&mut profile).await;
            }

            // Detect bottlenecks
            if self.config.enable_bottleneck_detection {
                self.detect_bottlenecks(&mut profile).await;
            }

            // Store completed profile
            let mut completed_profiles = self.completed_profiles.write().await;
            completed_profiles.push_back(profile);

            // Maintain maximum profiles in memory
            while completed_profiles.len() > self.config.max_profiles_in_memory {
                completed_profiles.pop_front();
            }

            self.update_completion_stats(true).await;

            info!(
                "Completed profiling for request with profile ID: {}",
                profile_id
            );
        } else {
            return Err(RequestProfilingError::ProfileNotFound(
                profile_id.to_string(),
            ));
        }

        Ok(())
    }

    /// Record timing for a specific phase
    pub async fn record_timing(
        &self,
        profile_id: &str,
        phase: &str,
        duration: Duration,
    ) -> Result<(), RequestProfilingError> {
        let mut active_profiles = self.active_profiles.write().await;

        if let Some(profile) = active_profiles.get_mut(profile_id) {
            match phase {
                "parsing" => profile.timing_breakdown.parsing_duration = Some(duration),
                "auth" => profile.timing_breakdown.auth_duration = Some(duration),
                "model_load" => profile.timing_breakdown.model_load_duration = Some(duration),
                "preprocessing" => profile.timing_breakdown.preprocessing_duration = Some(duration),
                "inference" => profile.timing_breakdown.inference_duration = Some(duration),
                "postprocessing" => {
                    profile.timing_breakdown.postprocessing_duration = Some(duration)
                },
                "serialization" => profile.timing_breakdown.serialization_duration = Some(duration),
                "network_io" => profile.timing_breakdown.network_io_duration = Some(duration),
                "database" => profile.timing_breakdown.database_duration = Some(duration),
                "cache" => profile.timing_breakdown.cache_duration = Some(duration),
                _ => {
                    profile.timing_breakdown.custom_timings.insert(phase.to_string(), duration);
                },
            }

            trace!("Recorded timing for phase '{}': {:?}", phase, duration);
        } else {
            return Err(RequestProfilingError::ProfileNotFound(
                profile_id.to_string(),
            ));
        }

        Ok(())
    }

    /// Record resource usage
    pub async fn record_resource_usage(
        &self,
        profile_id: &str,
        resource_usage: ResourceUsage,
    ) -> Result<(), RequestProfilingError> {
        let mut active_profiles = self.active_profiles.write().await;

        if let Some(profile) = active_profiles.get_mut(profile_id) {
            profile.resource_usage = resource_usage;
            trace!("Recorded resource usage for profile: {}", profile_id);
        } else {
            return Err(RequestProfilingError::ProfileNotFound(
                profile_id.to_string(),
            ));
        }

        Ok(())
    }

    /// Add call stack entry
    pub async fn add_call_stack_entry(
        &self,
        profile_id: &str,
        entry: CallStackEntry,
    ) -> Result<(), RequestProfilingError> {
        if !self.config.enable_call_stack_tracking {
            return Ok(());
        }

        let mut active_profiles = self.active_profiles.write().await;

        if let Some(profile) = active_profiles.get_mut(profile_id) {
            profile.call_stack.push(entry);
        } else {
            return Err(RequestProfilingError::ProfileNotFound(
                profile_id.to_string(),
            ));
        }

        Ok(())
    }

    /// Record error information
    pub async fn record_error(
        &self,
        profile_id: &str,
        error_info: ErrorInfo,
    ) -> Result<(), RequestProfilingError> {
        let mut active_profiles = self.active_profiles.write().await;

        if let Some(profile) = active_profiles.get_mut(profile_id) {
            profile.error_info = Some(error_info);
            profile.status = ProfileStatus::Failed;
            warn!("Recorded error for profile: {}", profile_id);
        } else {
            return Err(RequestProfilingError::ProfileNotFound(
                profile_id.to_string(),
            ));
        }

        Ok(())
    }

    /// Get profile by ID
    pub async fn get_profile(&self, profile_id: &str) -> Option<RequestProfile> {
        // Check active profiles first
        if let Some(profile) = self.active_profiles.read().await.get(profile_id) {
            return Some(profile.clone());
        }

        // Check completed profiles
        let completed_profiles = self.completed_profiles.read().await;
        completed_profiles.iter().find(|p| p.id == profile_id).cloned()
    }

    /// List profiles with optional filtering
    pub async fn list_profiles(
        &self,
        request_type: Option<&str>,
        status: Option<ProfileStatus>,
        limit: Option<usize>,
    ) -> Vec<RequestProfile> {
        let mut profiles = Vec::new();

        // Add active profiles
        for profile in self.active_profiles.read().await.values() {
            if self.matches_filter(profile, request_type, status) {
                profiles.push(profile.clone());
            }
        }

        // Add completed profiles
        for profile in self.completed_profiles.read().await.iter() {
            if self.matches_filter(profile, request_type, status) {
                profiles.push(profile.clone());
            }
        }

        // Sort by start time (newest first)
        profiles.sort_by(|a, b| b.start_time.cmp(&a.start_time));

        // Apply limit
        if let Some(limit) = limit {
            profiles.truncate(limit);
        }

        profiles
    }

    /// Check if profile matches filter criteria
    fn matches_filter(
        &self,
        profile: &RequestProfile,
        request_type: Option<&str>,
        status: Option<ProfileStatus>,
    ) -> bool {
        if let Some(req_type) = request_type {
            if profile.request_type != req_type {
                return false;
            }
        }

        if let Some(target_status) = status {
            if profile.status != target_status {
                return false;
            }
        }

        true
    }

    /// Analyze performance and generate recommendations
    async fn analyze_performance(&self, profile: &mut RequestProfile) {
        // Analyze timing breakdown
        if let Some(total_duration) = profile.total_duration {
            let total_ms = total_duration.as_millis() as f64;

            // Check for high latency
            if total_ms > 1000.0 {
                profile.performance_issues.push(PerformanceIssue {
                    issue_type: PerformanceIssueType::HighLatency,
                    severity: IssueSeverity::High,
                    description: format!(
                        "Request took {:.2}ms, which is above acceptable threshold",
                        total_ms
                    ),
                    location: None,
                    metric_value: Some(total_ms),
                    threshold: Some(1000.0),
                    impact: "User experience degradation".to_string(),
                    detected_at: SystemTime::now(),
                });

                profile.recommendations.push(PerformanceRecommendation {
                    recommendation_type: RecommendationType::AlgorithmOptimization,
                    priority: IssueSeverity::High,
                    description: "Optimize request processing to reduce latency".to_string(),
                    action: "Profile individual components and optimize bottlenecks".to_string(),
                    expected_impact: "Reduce request latency by 20-50%".to_string(),
                    difficulty: DifficultyLevel::Medium,
                    related_issues: vec!["HighLatency".to_string()],
                    code_locations: Vec::new(),
                });
            }
        }

        // Analyze resource usage
        if let Some(cpu_percent) = profile.resource_usage.peak_cpu_percent {
            if cpu_percent > 90.0 {
                profile.performance_issues.push(PerformanceIssue {
                    issue_type: PerformanceIssueType::HighCpuUsage,
                    severity: IssueSeverity::Medium,
                    description: format!("Peak CPU usage of {:.1}% detected", cpu_percent),
                    location: None,
                    metric_value: Some(cpu_percent),
                    threshold: Some(90.0),
                    impact: "System performance degradation".to_string(),
                    detected_at: SystemTime::now(),
                });
            }
        }

        if let Some(memory_bytes) = profile.resource_usage.peak_memory_bytes {
            let memory_mb = memory_bytes as f64 / (1024.0 * 1024.0);
            if memory_mb > 500.0 {
                profile.performance_issues.push(PerformanceIssue {
                    issue_type: PerformanceIssueType::HighMemoryUsage,
                    severity: IssueSeverity::Medium,
                    description: format!("Peak memory usage of {:.1}MB detected", memory_mb),
                    location: None,
                    metric_value: Some(memory_mb),
                    threshold: Some(500.0),
                    impact: "Increased memory pressure".to_string(),
                    detected_at: SystemTime::now(),
                });
            }
        }
    }

    /// Detect performance bottlenecks
    async fn detect_bottlenecks(&self, profile: &mut RequestProfile) {
        if let Some(total_duration) = profile.total_duration {
            let total_ms = total_duration.as_millis() as f64;
            let threshold_percent = self.config.bottleneck_threshold_percent;

            // Check each timing phase
            let timings = vec![
                ("inference", profile.timing_breakdown.inference_duration),
                (
                    "preprocessing",
                    profile.timing_breakdown.preprocessing_duration,
                ),
                (
                    "postprocessing",
                    profile.timing_breakdown.postprocessing_duration,
                ),
                ("model_load", profile.timing_breakdown.model_load_duration),
                ("database", profile.timing_breakdown.database_duration),
                ("network_io", profile.timing_breakdown.network_io_duration),
            ];

            for (phase, duration_opt) in timings {
                if let Some(duration) = duration_opt {
                    let phase_ms = duration.as_millis() as f64;
                    let percentage = (phase_ms / total_ms) * 100.0;

                    if percentage > threshold_percent {
                        profile.performance_issues.push(PerformanceIssue {
                            issue_type: PerformanceIssueType::HotSpot,
                            severity: IssueSeverity::High,
                            description: format!(
                                "{} phase consumes {:.1}% of total request time",
                                phase, percentage
                            ),
                            location: Some(phase.to_string()),
                            metric_value: Some(percentage),
                            threshold: Some(threshold_percent),
                            impact: "Major performance bottleneck".to_string(),
                            detected_at: SystemTime::now(),
                        });

                        profile.recommendations.push(PerformanceRecommendation {
                            recommendation_type: RecommendationType::AlgorithmOptimization,
                            priority: IssueSeverity::High,
                            description: format!(
                                "Optimize {} phase which is the main bottleneck",
                                phase
                            ),
                            action: format!("Profile and optimize {} implementation", phase),
                            expected_impact: "Significantly reduce overall request latency"
                                .to_string(),
                            difficulty: DifficultyLevel::Medium,
                            related_issues: vec!["HotSpot".to_string()],
                            code_locations: vec![phase.to_string()],
                        });
                    }
                }
            }
        }
    }

    /// Update completion statistics
    async fn update_completion_stats(&self, success: bool) {
        let mut stats = self.stats.write().await;
        stats.active_profiles = stats.active_profiles.saturating_sub(1);

        if success {
            stats.completed_profiles += 1;
        } else {
            stats.failed_profiles += 1;
        }
    }

    /// Get profiling statistics
    pub async fn get_stats(&self) -> ProfilingStats {
        self.stats.read().await.clone()
    }

    /// Export profiles to specified format
    pub async fn export_profiles(
        &self,
        format: ProfileExportFormat,
        profiles: Vec<RequestProfile>,
    ) -> Result<String, RequestProfilingError> {
        if !self.config.enable_profile_export {
            return Err(RequestProfilingError::ExportFailed(
                "Profile export is disabled".to_string(),
            ));
        }

        match format {
            ProfileExportFormat::JSON => serde_json::to_string_pretty(&profiles)
                .map_err(|e| RequestProfilingError::ExportFailed(e.to_string())),
            ProfileExportFormat::CSV => self.export_to_csv(profiles),
            _ => Err(RequestProfilingError::ExportFailed(format!(
                "Export format {:?} not implemented",
                format
            ))),
        }
    }

    /// Export profiles to CSV format
    fn export_to_csv(
        &self,
        profiles: Vec<RequestProfile>,
    ) -> Result<String, RequestProfilingError> {
        let mut csv = String::new();
        csv.push_str("profile_id,request_id,request_type,total_duration_ms,status,error_type\n");

        for profile in profiles {
            let duration_ms = profile
                .total_duration
                .map(|d| d.as_millis().to_string())
                .unwrap_or_else(|| "N/A".to_string());

            let default_error_type = "None".to_string();
            let error_type = profile
                .error_info
                .as_ref()
                .map(|e| &e.error_type)
                .unwrap_or(&default_error_type);

            csv.push_str(&format!(
                "{},{},{},{},{:?},{}\n",
                profile.id,
                profile.request_id,
                profile.request_type,
                duration_ms,
                profile.status,
                error_type
            ));
        }

        Ok(csv)
    }

    /// Generate flame graph data
    pub async fn generate_flame_graph(
        &self,
        profile_id: &str,
    ) -> Result<String, RequestProfilingError> {
        if !self.config.enable_flame_graphs {
            return Err(RequestProfilingError::ExportFailed(
                "Flame graph generation is disabled".to_string(),
            ));
        }

        let profile = self
            .get_profile(profile_id)
            .await
            .ok_or_else(|| RequestProfilingError::ProfileNotFound(profile_id.to_string()))?;

        // Generate flame graph data from call stack
        let mut flame_data = String::new();

        for entry in &profile.call_stack {
            if let Some(duration) = entry.duration {
                let duration_ms = duration.as_millis();
                flame_data.push_str(&format!("{} {}\n", entry.function_name, duration_ms));
            }
        }

        Ok(flame_data)
    }
}

/// Summary statistics for the profiling service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingStatsSummary {
    /// Total profiles created
    pub total_profiles: u64,

    /// Success rate percentage
    pub success_rate_percent: f64,

    /// Average profiling overhead percentage
    pub avg_overhead_percent: f64,

    /// Currently active profiles
    pub active_profiles: u64,

    /// Memory usage by profiler in MB
    pub profiler_memory_mb: f64,

    /// Actual sampling rate achieved
    pub sampling_rate_actual: f64,
}

impl RequestProfilingService {
    /// Get summary statistics
    pub async fn get_stats_summary(&self) -> ProfilingStatsSummary {
        let stats = self.stats.read().await;

        let success_rate = if stats.total_profiles > 0 {
            (stats.completed_profiles as f64 / stats.total_profiles as f64) * 100.0
        } else {
            0.0
        };

        ProfilingStatsSummary {
            total_profiles: stats.total_profiles,
            success_rate_percent: success_rate,
            avg_overhead_percent: stats.avg_overhead_percent,
            active_profiles: stats.active_profiles,
            profiler_memory_mb: stats.profiler_memory_usage as f64 / (1024.0 * 1024.0),
            sampling_rate_actual: stats.sampling_rate_actual,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_profiling_service_creation() {
        let config = RequestProfilingConfig::default();
        let service = RequestProfilingService::new(config);
        let stats = service.get_stats().await;

        assert_eq!(stats.total_profiles, 0);
        assert_eq!(stats.active_profiles, 0);
    }

    #[tokio::test]
    async fn test_start_and_complete_profile() {
        let mut config = RequestProfilingConfig::default();
        config.sampling_rate = 1.0; // 100% sampling for testing
        config.min_duration_to_profile_ms = 0; // Store all profiles regardless of duration

        let service = RequestProfilingService::new(config);

        let profile_id = service
            .start_profile("req_123".to_string(), "inference".to_string())
            .await
            .unwrap();
        assert!(!profile_id.is_empty());

        let stats = service.get_stats().await;
        assert_eq!(stats.total_profiles, 1);
        assert_eq!(stats.active_profiles, 1);

        service.complete_profile(&profile_id).await.unwrap();

        let stats = service.get_stats().await;
        assert_eq!(stats.active_profiles, 0);
        assert_eq!(stats.completed_profiles, 1);
    }

    #[tokio::test]
    async fn test_record_timing() {
        let mut config = RequestProfilingConfig::default();
        config.sampling_rate = 1.0;

        let service = RequestProfilingService::new(config);
        let profile_id = service
            .start_profile("req_123".to_string(), "inference".to_string())
            .await
            .unwrap();

        let inference_duration = Duration::from_millis(100);
        service
            .record_timing(&profile_id, "inference", inference_duration)
            .await
            .unwrap();

        let profile = service.get_profile(&profile_id).await.unwrap();
        assert_eq!(
            profile.timing_breakdown.inference_duration,
            Some(inference_duration)
        );
    }

    #[tokio::test]
    async fn test_resource_usage_recording() {
        let mut config = RequestProfilingConfig::default();
        config.sampling_rate = 1.0;

        let service = RequestProfilingService::new(config);
        let profile_id = service
            .start_profile("req_123".to_string(), "inference".to_string())
            .await
            .unwrap();

        let resource_usage = ResourceUsage {
            peak_cpu_percent: Some(85.5),
            peak_memory_bytes: Some(1024 * 1024 * 100), // 100MB
            ..Default::default()
        };

        service
            .record_resource_usage(&profile_id, resource_usage.clone())
            .await
            .unwrap();

        let profile = service.get_profile(&profile_id).await.unwrap();
        assert_eq!(profile.resource_usage.peak_cpu_percent, Some(85.5));
        assert_eq!(
            profile.resource_usage.peak_memory_bytes,
            Some(1024 * 1024 * 100)
        );
    }

    #[tokio::test]
    async fn test_error_recording() {
        let mut config = RequestProfilingConfig::default();
        config.sampling_rate = 1.0;

        let service = RequestProfilingService::new(config);
        let profile_id = service
            .start_profile("req_123".to_string(), "inference".to_string())
            .await
            .unwrap();

        let error_info = ErrorInfo {
            error_type: "TimeoutError".to_string(),
            error_message: "Request timed out".to_string(),
            stack_trace: vec!["function1".to_string(), "function2".to_string()],
            location: Some("inference.rs:123".to_string()),
            timestamp: SystemTime::now(),
            category: ErrorCategory::Timeout,
        };

        service.record_error(&profile_id, error_info).await.unwrap();

        let profile = service.get_profile(&profile_id).await.unwrap();
        assert!(profile.error_info.is_some());
        assert_eq!(profile.status, ProfileStatus::Failed);
    }

    #[tokio::test]
    async fn test_profile_filtering() {
        let mut config = RequestProfilingConfig::default();
        config.sampling_rate = 1.0;
        config.min_duration_to_profile_ms = 0; // Store all profiles regardless of duration

        let service = RequestProfilingService::new(config);

        let profile1 = service
            .start_profile("req_1".to_string(), "inference".to_string())
            .await
            .unwrap();
        let profile2 = service
            .start_profile("req_2".to_string(), "preprocessing".to_string())
            .await
            .unwrap();

        service.complete_profile(&profile1).await.unwrap();

        // Test filtering by request type
        let inference_profiles = service.list_profiles(Some("inference"), None, None).await;
        assert_eq!(inference_profiles.len(), 1);
        assert_eq!(inference_profiles[0].request_type, "inference");

        // Test filtering by status
        let active_profiles = service.list_profiles(None, Some(ProfileStatus::Active), None).await;
        assert_eq!(active_profiles.len(), 1);
        assert_eq!(active_profiles[0].id, profile2);
    }

    #[test]
    fn test_issue_severity_ordering() {
        assert!(IssueSeverity::Critical > IssueSeverity::High);
        assert!(IssueSeverity::High > IssueSeverity::Medium);
        assert!(IssueSeverity::Medium > IssueSeverity::Low);
        assert!(IssueSeverity::Low > IssueSeverity::Info);
    }

    #[test]
    fn test_difficulty_level_ordering() {
        assert!(DifficultyLevel::Expert > DifficultyLevel::Hard);
        assert!(DifficultyLevel::Hard > DifficultyLevel::Medium);
        assert!(DifficultyLevel::Medium > DifficultyLevel::Easy);
    }
}
