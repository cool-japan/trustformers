//! Crash report data types: the top-level `CrashReport` and the system/app/thread/memory/network snapshots it's built from.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{
    device_info::{MobileDeviceInfo, ThermalState},
    mobile_performance_profiler::{MobileMetricsSnapshot, PerformanceBottleneck},
    model_debugger::{ModelAnomaly, TensorDebugInfo},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use super::analysis_types::{
    CrashPattern, ErrorLogEntry, MemoryPermissions, MemoryRegionType, ModelPerformanceMetrics,
    NetworkConnectionType, RecoveryImpact, RiskAssessment, RiskLevel, SimilarCrash, SystemEvent,
    ThreadState, UserAction,
};
use super::config::RecoveryStrategy;

/// Application information at crash time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppCrashInfo {
    /// Application version
    pub app_version: String,
    /// Build number
    pub build_number: String,
    /// Framework version
    pub framework_version: String,
    /// Application state
    pub app_state: AppState,
    /// Foreground/background status
    pub foreground_status: ForegroundStatus,
    /// Session duration
    pub session_duration: Duration,
    /// Model information
    pub model_info: Option<ModelCrashInfo>,
    /// Recent operations
    pub recent_operations: Vec<RecentOperation>,
}
/// Application state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AppState {
    Launching,
    Active,
    Background,
    Suspended,
    Terminating,
    Unknown,
}
/// Battery information at crash time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatteryCrashInfo {
    /// Battery level (0-100)
    pub level_percent: f32,
    /// Charging status
    pub charging: bool,
    /// Battery temperature
    pub temperature_c: Option<f32>,
    /// Power usage
    pub power_usage_mw: Option<f32>,
}
/// CPU information at crash time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuCrashInfo {
    /// CPU usage percentage
    pub usage_percent: f32,
    /// CPU frequency
    pub frequency_mhz: u32,
    /// CPU temperature
    pub temperature_c: Option<f32>,
    /// Throttling status
    pub throttling: bool,
    /// Active cores
    pub active_cores: usize,
}
/// Crash analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrashAnalysis {
    /// Root cause analysis
    pub root_cause: Option<String>,
    /// Contributing factors
    pub contributing_factors: Vec<String>,
    /// Similarity to known crashes
    pub similar_crashes: Vec<SimilarCrash>,
    /// Pattern analysis
    pub patterns: Vec<CrashPattern>,
    /// Risk assessment
    pub risk_assessment: RiskAssessment,
    /// Analysis confidence
    pub confidence_score: f32,
    /// Analysis timestamp
    pub analysis_timestamp: u64,
}
/// Crash context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrashContext {
    /// Current operation
    pub current_operation: Option<String>,
    /// User actions leading to crash
    pub user_actions: Vec<UserAction>,
    /// System events
    pub system_events: Vec<SystemEvent>,
    /// Model anomalies detected
    pub model_anomalies: Vec<ModelAnomaly>,
    /// Performance bottlenecks
    pub performance_bottlenecks: Vec<PerformanceBottleneck>,
    /// Error logs
    pub error_logs: Vec<ErrorLogEntry>,
}
/// Crash report structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrashReport {
    /// Unique crash report ID
    pub report_id: String,
    /// Timestamp of crash
    pub timestamp: u64,
    /// Crash type
    pub crash_type: CrashType,
    /// Crash severity
    pub severity: CrashSeverity,
    /// System information
    pub system_info: SystemCrashInfo,
    /// Application information
    pub app_info: AppCrashInfo,
    /// Stack trace information
    pub stack_trace: Option<StackTrace>,
    /// Memory dump
    pub memory_dump: Option<MemoryDump>,
    /// Context information
    pub context: CrashContext,
    /// Analysis results
    pub analysis: Option<CrashAnalysis>,
    /// Recovery suggestions
    pub recovery_suggestions: Vec<RecoverySuggestion>,
    /// Privacy compliant data only
    pub is_privacy_compliant: bool,
}
/// Crash severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CrashSeverity {
    Low,
    Medium,
    High,
    Critical,
}
/// Types of crashes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CrashType {
    /// Segmentation fault
    SegmentationFault,
    /// Memory allocation failure
    OutOfMemory,
    /// Stack overflow
    StackOverflow,
    /// Assertion failure
    AssertionFailure,
    /// Uncaught exception
    UncaughtException,
    /// Signal-based crash
    Signal(i32),
    /// Application hang/ANR
    ApplicationHang,
    /// GPU-related crash
    GPUCrash,
    /// Model-related crash
    ModelCrash,
    /// Unknown crash
    Unknown,
}
/// Exception information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExceptionInfo {
    /// Exception type
    pub exception_type: String,
    /// Exception message
    pub message: String,
    /// Exception code
    pub code: Option<i32>,
    /// Additional info
    pub additional_info: HashMap<String, String>,
}
/// Foreground status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForegroundStatus {
    Foreground,
    Background,
    Unknown,
}
/// GPU information at crash time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuCrashInfo {
    /// GPU usage percentage
    pub usage_percent: f32,
    /// GPU memory usage
    pub memory_usage_mb: f32,
    /// GPU temperature
    pub temperature_c: Option<f32>,
    /// GPU frequency
    pub frequency_mhz: Option<u32>,
    /// GPU vendor
    pub vendor: String,
    /// GPU model
    pub model: String,
}
/// Heap information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeapInfo {
    /// Total heap size
    pub total_size: usize,
    /// Used heap size
    pub used_size: usize,
    /// Free heap size
    pub free_size: usize,
    /// Largest free block
    pub largest_free_block: usize,
    /// Fragmentation ratio
    pub fragmentation_ratio: f32,
}
/// Memory dump information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDump {
    /// Memory regions
    pub memory_regions: Vec<MemoryRegion>,
    /// Heap information
    pub heap_info: HeapInfo,
    /// Stack information
    pub stack_info: StackInfo,
    /// Register values
    pub registers: HashMap<String, u64>,
    /// Binary data (base64 encoded)
    pub binary_data: Option<String>,
}
/// Memory region information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRegion {
    /// Start address
    pub start_address: u64,
    /// End address
    pub end_address: u64,
    /// Size in bytes
    pub size: usize,
    /// Region type
    pub region_type: MemoryRegionType,
    /// Permissions
    pub permissions: MemoryPermissions,
    /// Protection flags
    pub protection: u32,
}
/// Memory usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsageInfo {
    /// Total memory
    pub total_mb: f32,
    /// Used memory
    pub used_mb: f32,
    /// Available memory
    pub available_mb: f32,
    /// Heap usage
    pub heap_mb: f32,
    /// Stack usage
    pub stack_mb: f32,
    /// GPU memory
    pub gpu_mb: Option<f32>,
}
/// Model information at crash time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCrashInfo {
    /// Model ID
    pub model_id: String,
    /// Model type
    pub model_type: String,
    /// Current operation
    pub current_operation: Option<String>,
    /// Input tensors
    pub input_tensors: Vec<TensorDebugInfo>,
    /// Performance metrics
    pub performance_metrics: Option<ModelPerformanceMetrics>,
}
/// Network information at crash time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkCrashInfo {
    /// Connection type
    pub connection_type: NetworkConnectionType,
    /// Network strength
    pub signal_strength: Option<f32>,
    /// Bandwidth
    pub bandwidth_mbps: Option<f32>,
    /// Latency
    pub latency_ms: Option<f32>,
}
/// Recent operation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentOperation {
    /// Operation name
    pub operation: String,
    /// Timestamp
    pub timestamp: u64,
    /// Duration
    pub duration: Duration,
    /// Success status
    pub success: bool,
    /// Error message
    pub error: Option<String>,
}
/// Recovery suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoverySuggestion {
    /// Suggestion type
    pub suggestion_type: RecoveryStrategy,
    /// Description
    pub description: String,
    /// Implementation steps
    pub steps: Vec<String>,
    /// Success probability
    pub success_probability: f32,
    /// Risk level
    pub risk_level: RiskLevel,
    /// Estimated impact
    pub impact: RecoveryImpact,
}
/// Signal information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalInfo {
    /// Signal number
    pub signal: i32,
    /// Signal name
    pub signal_name: String,
    /// Signal code
    pub signal_code: Option<i32>,
    /// Fault address
    pub fault_address: Option<u64>,
}
/// Stack frame information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackFrame {
    /// Frame index
    pub frame_index: usize,
    /// Instruction pointer
    pub instruction_pointer: u64,
    /// Function name
    pub function_name: Option<String>,
    /// File name
    pub file_name: Option<String>,
    /// Line number
    pub line_number: Option<u32>,
    /// Module name
    pub module_name: Option<String>,
    /// Symbol offset
    pub symbol_offset: Option<u64>,
}
/// Stack information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackInfo {
    /// Stack base address
    pub base_address: u64,
    /// Stack size
    pub size: usize,
    /// Stack usage
    pub usage: usize,
    /// Stack overflow detected
    pub overflow_detected: bool,
}
/// Stack trace information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackTrace {
    /// Thread that crashed
    pub crashed_thread: ThreadInfo,
    /// All threads
    pub all_threads: Vec<ThreadInfo>,
    /// Exception information
    pub exception_info: Option<ExceptionInfo>,
    /// Signal information
    pub signal_info: Option<SignalInfo>,
    /// Call stack frames
    pub frames: Vec<StackFrame>,
}
/// System information at crash time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemCrashInfo {
    /// Device information
    pub device_info: MobileDeviceInfo,
    /// Performance metrics
    pub performance_metrics: Option<MobileMetricsSnapshot>,
    /// Memory usage
    pub memory_usage: MemoryUsageInfo,
    /// CPU information
    pub cpu_info: CpuCrashInfo,
    /// GPU information
    pub gpu_info: Option<GpuCrashInfo>,
    /// Thermal state
    pub thermal_state: Option<ThermalState>,
    /// Battery information
    pub battery_info: Option<BatteryCrashInfo>,
    /// Network information
    pub network_info: Option<NetworkCrashInfo>,
}
/// Thread information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadInfo {
    /// Thread ID
    pub thread_id: u64,
    /// Thread name
    pub thread_name: Option<String>,
    /// Thread state
    pub thread_state: ThreadState,
    /// Stack frames
    pub stack_frames: Vec<StackFrame>,
    /// Register values
    pub registers: HashMap<String, u64>,
}
