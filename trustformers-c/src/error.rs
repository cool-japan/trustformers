//! Error handling for TrustformeRS C API

use crate::utils::{c_str_to_string, string_to_c_str};
use std::ffi::CString;
use std::fmt;
use std::os::raw::{c_char, c_int};
use std::ptr;

/// C-compatible error codes for TrustformeRS operations
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TrustformersError {
    /// Operation completed successfully
    Success = 0,
    /// Null pointer passed where valid pointer expected
    NullPointer = -1,
    /// Invalid parameter provided
    InvalidParameter = -2,
    /// Memory allocation failed
    OutOfMemory = -3,
    /// File not found or inaccessible
    FileNotFound = -4,
    /// Invalid file format
    InvalidFormat = -5,
    /// Model loading failed
    ModelLoadError = -6,
    /// Tokenizer error
    TokenizerError = -7,
    /// Pipeline creation failed
    PipelineError = -8,
    /// Inference execution failed
    InferenceError = -9,
    /// Feature not available in this build
    FeatureNotAvailable = -10,
    /// Device not available (e.g., CUDA not available)
    DeviceNotAvailable = -11,
    /// Tensor operation failed
    TensorError = -12,
    /// Generic runtime error
    RuntimeError = -13,
    /// Thread/concurrency error
    ConcurrencyError = -14,
    /// Serialization/deserialization error
    SerializationError = -15,
    /// Network/IO error
    NetworkError = -16,
    /// Configuration error
    ConfigError = -17,
    /// Invalid path provided
    InvalidPath = -18,
    /// Resource limit exceeded
    ResourceLimitExceeded = -19,
    /// Operation timeout
    Timeout = -20,
    /// Validation failed
    ValidationError = -21,
    /// Invalid handle provided
    InvalidHandle = -22,
    /// Compilation error
    CompilationError = -23,
    /// Execution error
    ExecutionError = -24,
    /// Hardware error
    HardwareError = -25,
    /// Initialization error
    InitializationError = -26,
    /// Optimization error
    OptimizationError = -27,
    /// Plugin initialization error
    PluginInitError = -28,
    /// Plugin not found
    PluginNotFound = -29,
    /// Operation not found
    OperationNotFound = -30,
    /// Unknown error
    Unknown = -100,
}

/// Result type used throughout TrustformeRS C API
pub type TrustformersResult<T> = Result<T, TrustformersError>;

impl fmt::Display for TrustformersError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            TrustformersError::Success => "Success",
            TrustformersError::NullPointer => "Null pointer provided",
            TrustformersError::InvalidParameter => "Invalid parameter",
            TrustformersError::OutOfMemory => "Out of memory",
            TrustformersError::FileNotFound => "File not found",
            TrustformersError::InvalidFormat => "Invalid file format",
            TrustformersError::ModelLoadError => "Model loading failed",
            TrustformersError::TokenizerError => "Tokenizer error",
            TrustformersError::PipelineError => "Pipeline error",
            TrustformersError::InferenceError => "Inference failed",
            TrustformersError::FeatureNotAvailable => "Feature not available",
            TrustformersError::DeviceNotAvailable => "Device not available",
            TrustformersError::TensorError => "Tensor operation failed",
            TrustformersError::RuntimeError => "Runtime error",
            TrustformersError::ConcurrencyError => "Concurrency error",
            TrustformersError::SerializationError => "Serialization error",
            TrustformersError::NetworkError => "Network error",
            TrustformersError::ConfigError => "Configuration error",
            TrustformersError::ResourceLimitExceeded => "Resource limit exceeded",
            TrustformersError::Timeout => "Operation timeout",
            TrustformersError::ValidationError => "Validation failed",
            TrustformersError::InvalidHandle => "Invalid handle provided",
            TrustformersError::CompilationError => "Compilation error",
            TrustformersError::ExecutionError => "Execution error",
            TrustformersError::HardwareError => "Hardware error",
            TrustformersError::InitializationError => "Initialization error",
            TrustformersError::OptimizationError => "Optimization error",
            TrustformersError::InvalidPath => "Invalid file path",
            TrustformersError::PluginInitError => "Plugin initialization failed",
            TrustformersError::PluginNotFound => "Plugin not found",
            TrustformersError::OperationNotFound => "Operation not found",
            TrustformersError::Unknown => "Unknown error",
        };
        write!(f, "{}", message)
    }
}

impl Default for TrustformersError {
    fn default() -> Self {
        Self::Success
    }
}

impl std::error::Error for TrustformersError {}

impl From<anyhow::Error> for TrustformersError {
    fn from(error: anyhow::Error) -> Self {
        let error_str = error.to_string().to_lowercase();

        if error_str.contains("null") || error_str.contains("pointer") {
            TrustformersError::NullPointer
        } else if error_str.contains("memory") || error_str.contains("allocation") {
            TrustformersError::OutOfMemory
        } else if error_str.contains("file") || error_str.contains("not found") {
            TrustformersError::FileNotFound
        } else if error_str.contains("format") || error_str.contains("parse") {
            TrustformersError::InvalidFormat
        } else if error_str.contains("model") {
            TrustformersError::ModelLoadError
        } else if error_str.contains("tokenizer") || error_str.contains("token") {
            TrustformersError::TokenizerError
        } else if error_str.contains("pipeline") {
            TrustformersError::PipelineError
        } else if error_str.contains("inference") {
            TrustformersError::InferenceError
        } else if error_str.contains("feature") || error_str.contains("available") {
            TrustformersError::FeatureNotAvailable
        } else if error_str.contains("device")
            || error_str.contains("cuda")
            || error_str.contains("gpu")
        {
            TrustformersError::DeviceNotAvailable
        } else if error_str.contains("tensor") {
            TrustformersError::TensorError
        } else if error_str.contains("thread") || error_str.contains("concurrency") {
            TrustformersError::ConcurrencyError
        } else if error_str.contains("serialize") || error_str.contains("deserialize") {
            TrustformersError::SerializationError
        } else if error_str.contains("network") || error_str.contains("io") {
            TrustformersError::NetworkError
        } else if error_str.contains("config") {
            TrustformersError::ConfigError
        } else if error_str.contains("limit") || error_str.contains("exceeded") {
            TrustformersError::ResourceLimitExceeded
        } else if error_str.contains("timeout") {
            TrustformersError::Timeout
        } else {
            TrustformersError::RuntimeError
        }
    }
}

impl From<serde_json::Error> for TrustformersError {
    fn from(_: serde_json::Error) -> Self {
        TrustformersError::SerializationError
    }
}

impl From<std::io::Error> for TrustformersError {
    fn from(error: std::io::Error) -> Self {
        match error.kind() {
            std::io::ErrorKind::NotFound => TrustformersError::FileNotFound,
            std::io::ErrorKind::PermissionDenied => TrustformersError::InvalidPath,
            std::io::ErrorKind::OutOfMemory => TrustformersError::OutOfMemory,
            std::io::ErrorKind::InvalidData | std::io::ErrorKind::InvalidInput => {
                TrustformersError::InvalidFormat
            },
            std::io::ErrorKind::TimedOut => TrustformersError::Timeout,
            _ => TrustformersError::RuntimeError,
        }
    }
}

#[cfg(feature = "codegen")]
impl From<syn::Error> for TrustformersError {
    fn from(_: syn::Error) -> Self {
        TrustformersError::CompilationError
    }
}

#[cfg(feature = "codegen")]
impl From<regex::Error> for TrustformersError {
    fn from(_: regex::Error) -> Self {
        TrustformersError::ValidationError
    }
}

#[cfg(feature = "codegen")]
impl From<toml::de::Error> for TrustformersError {
    fn from(_: toml::de::Error) -> Self {
        TrustformersError::ConfigError
    }
}

/// Build information structure for C API
#[repr(C)]
#[derive(Debug)]
pub struct TrustformersBuildInfo {
    /// Version string
    pub version: *mut c_char,
    /// Comma-separated list of enabled features
    pub features: *mut c_char,
    /// Build date
    pub build_date: *mut c_char,
    /// Target platform
    pub target: *mut c_char,
}

impl Default for TrustformersBuildInfo {
    fn default() -> Self {
        Self {
            version: std::ptr::null_mut(),
            features: std::ptr::null_mut(),
            build_date: std::ptr::null_mut(),
            target: std::ptr::null_mut(),
        }
    }
}

/// Memory usage statistics for C API
#[repr(C)]
#[derive(Debug, Default)]
pub struct TrustformersMemoryUsage {
    /// Total memory allocated by TrustformeRS (bytes)
    pub total_memory_bytes: u64,
    /// Peak memory usage (bytes)
    pub peak_memory_bytes: u64,
    /// Number of allocated models
    pub allocated_models: u64,
    /// Number of allocated tokenizers
    pub allocated_tokenizers: u64,
    /// Number of allocated pipelines
    pub allocated_pipelines: u64,
    /// Number of allocated tensors
    pub allocated_tensors: u64,
}

/// Enhanced error context for C API
#[repr(C)]
#[derive(Debug, Default)]
pub struct TrustformersErrorContext {
    /// Error code
    pub error_code: TrustformersError,
    /// Primary error message
    pub message: *mut c_char,
    /// Detailed error description
    pub description: *mut c_char,
    /// Error category (e.g., "model", "tokenizer", "pipeline")
    pub category: *mut c_char,
    /// Function where error occurred
    pub function_name: *mut c_char,
    /// File where error occurred
    pub file_name: *mut c_char,
    /// Line number where error occurred
    pub line_number: c_int,
    /// Stack trace (JSON array of strings)
    pub stack_trace: *mut c_char,
    /// Error severity: 0=Info, 1=Warning, 2=Error, 3=Fatal
    pub severity: c_int,
    /// Timestamp when error occurred (Unix timestamp)
    pub timestamp: u64,
    /// Suggested recovery actions (JSON array)
    pub recovery_suggestions: *mut c_char,
    /// Related error codes (for error chains)
    pub related_errors: *mut TrustformersError,
    /// Number of related errors
    pub num_related_errors: usize,
}

/// Error diagnostic information
#[repr(C)]
#[derive(Debug, Default)]
pub struct TrustformersErrorDiagnostics {
    /// Total number of errors recorded
    pub total_errors: u64,
    /// Number of errors by type (JSON object)
    pub error_counts_json: *mut c_char,
    /// Most frequent error
    pub most_frequent_error: TrustformersError,
    /// Recent error history (JSON array)
    pub recent_errors_json: *mut c_char,
    /// Error rate (errors per minute)
    pub error_rate: f64,
    /// System health score (0-100, higher is better)
    pub health_score: f64,
    /// Error patterns detected (JSON array)
    pub error_patterns_json: *mut c_char,
}

/// Error recovery options
#[repr(C)]
#[derive(Debug)]
pub struct TrustformersErrorRecovery {
    /// Recovery strategy: 0=Retry, 1=Reset, 2=Fallback, 3=Abort
    pub strategy: c_int,
    /// Maximum retry attempts
    pub max_retries: c_int,
    /// Retry delay in milliseconds
    pub retry_delay_ms: c_int,
    /// Whether to enable automatic recovery
    pub auto_recovery: c_int,
    /// Fallback configuration (JSON string)
    pub fallback_config: *const c_char,
}

impl Default for TrustformersErrorRecovery {
    fn default() -> Self {
        Self {
            strategy: 0, // Retry
            max_retries: 3,
            retry_delay_ms: 1000,
            auto_recovery: 1,
            fallback_config: std::ptr::null(),
        }
    }
}

/// Performance metrics for C API
#[repr(C)]
#[derive(Debug, Default)]
pub struct TrustformersPerformanceMetrics {
    /// Total number of inferences performed
    pub total_inferences: u64,
    /// Average inference time in milliseconds
    pub avg_inference_time_ms: f64,
    /// Minimum inference time in milliseconds
    pub min_inference_time_ms: f64,
    /// Maximum inference time in milliseconds
    pub max_inference_time_ms: f64,
    /// Total inference time in milliseconds
    pub total_inference_time_ms: f64,
    /// Throughput (inferences per second)
    pub throughput_ips: f64,
}

/// Get error message string for a given error code
#[no_mangle]
pub extern "C" fn trustformers_error_message(error: TrustformersError) -> *const c_char {
    let message = match error {
        TrustformersError::Success => "Success",
        TrustformersError::NullPointer => "Null pointer provided",
        TrustformersError::InvalidParameter => "Invalid parameter",
        TrustformersError::OutOfMemory => "Out of memory",
        TrustformersError::FileNotFound => "File not found",
        TrustformersError::InvalidFormat => "Invalid file format",
        TrustformersError::ModelLoadError => "Model loading failed",
        TrustformersError::TokenizerError => "Tokenizer error",
        TrustformersError::PipelineError => "Pipeline error",
        TrustformersError::InferenceError => "Inference failed",
        TrustformersError::FeatureNotAvailable => "Feature not available",
        TrustformersError::DeviceNotAvailable => "Device not available",
        TrustformersError::TensorError => "Tensor operation failed",
        TrustformersError::RuntimeError => "Runtime error",
        TrustformersError::ConcurrencyError => "Concurrency error",
        TrustformersError::SerializationError => "Serialization error",
        TrustformersError::NetworkError => "Network error",
        TrustformersError::ConfigError => "Configuration error",
        TrustformersError::ResourceLimitExceeded => "Resource limit exceeded",
        TrustformersError::Timeout => "Operation timeout",
        TrustformersError::ValidationError => "Validation failed",
        TrustformersError::InvalidHandle => "Invalid handle provided",
        TrustformersError::CompilationError => "Compilation error",
        TrustformersError::ExecutionError => "Execution error",
        TrustformersError::HardwareError => "Hardware error",
        TrustformersError::InitializationError => "Initialization error",
        TrustformersError::OptimizationError => "Optimization error",
        TrustformersError::InvalidPath => "Invalid path provided",
        TrustformersError::PluginInitError => "Plugin initialization failed",
        TrustformersError::PluginNotFound => "Plugin not found",
        TrustformersError::OperationNotFound => "Operation not found",
        TrustformersError::Unknown => "Unknown error",
    };

    message.as_ptr() as *const c_char
}

/// Check if an error code represents success
#[no_mangle]
pub extern "C" fn trustformers_is_success(error: TrustformersError) -> c_int {
    if error == TrustformersError::Success {
        1
    } else {
        0
    }
}

/// Check if an error code represents a failure
#[no_mangle]
pub extern "C" fn trustformers_is_error(error: TrustformersError) -> c_int {
    if error != TrustformersError::Success {
        1
    } else {
        0
    }
}

/// Utility macro for error handling in C API functions
macro_rules! c_try {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => {
                eprintln!("TrustformeRS C API Error: {}", e);
                return TrustformersError::from(e);
            },
        }
    };
}

/// Utility macro for error handling with return value in C API functions
macro_rules! c_try_with_value {
    ($expr:expr, $error_val:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => {
                eprintln!("TrustformeRS C API Error: {}", e);
                return $error_val;
            },
        }
    };
}

pub(crate) use c_try;
pub(crate) use c_try_with_value;

use once_cell::sync::Lazy;
use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Global error tracking system
static ERROR_TRACKER: Lazy<RwLock<ErrorTracker>> = Lazy::new(|| RwLock::new(ErrorTracker::new()));

/// Internal error tracking structure
#[derive(Debug, Default)]
struct ErrorTracker {
    /// Total error count
    total_errors: u64,
    /// Error counts by type
    error_counts: HashMap<TrustformersError, u64>,
    /// Recent error history (circular buffer)
    recent_errors: VecDeque<ErrorRecord>,
    /// Error patterns detected
    error_patterns: Vec<String>,
    /// System health metrics
    health_score: f64,
    /// Error rate tracking
    error_timestamps: VecDeque<u64>,
}

/// Internal error record
#[derive(Debug, Clone)]
struct ErrorRecord {
    error_code: TrustformersError,
    timestamp: u64,
    function_name: String,
    file_name: String,
    line_number: i32,
    message: String,
    severity: i32,
}

impl ErrorTracker {
    fn new() -> Self {
        Self {
            total_errors: 0,
            error_counts: HashMap::new(),
            recent_errors: VecDeque::with_capacity(100),
            error_patterns: Vec::new(),
            health_score: 100.0,
            error_timestamps: VecDeque::with_capacity(1000),
        }
    }

    fn record_error(&mut self, record: ErrorRecord) {
        self.total_errors += 1;

        // Update error counts
        *self.error_counts.entry(record.error_code).or_insert(0) += 1;

        // Add to recent errors (circular buffer)
        if self.recent_errors.len() >= 100 {
            self.recent_errors.pop_front();
        }
        self.recent_errors.push_back(record.clone());

        // Track error timestamps for rate calculation
        if self.error_timestamps.len() >= 1000 {
            self.error_timestamps.pop_front();
        }
        self.error_timestamps.push_back(record.timestamp);

        // Update health score
        self.update_health_score();

        // Detect error patterns
        self.detect_error_patterns();
    }

    fn update_health_score(&mut self) {
        // Simple health score calculation based on recent error rate
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let recent_errors = self.error_timestamps.iter()
            .filter(|&&ts| now - ts < 300) // Last 5 minutes
            .count();

        // Health score decreases with error rate
        self.health_score = 100.0 - (recent_errors as f64 * 2.0).min(100.0);
    }

    fn detect_error_patterns(&mut self) {
        // Simple pattern detection - look for repeated error sequences
        if self.recent_errors.len() >= 3 {
            let last_three: Vec<_> =
                self.recent_errors.iter().rev().take(3).map(|r| r.error_code).collect();

            if last_three[0] == last_three[1] && last_three[1] == last_three[2] {
                let pattern = format!("Repeated error: {:?}", last_three[0]);
                if !self.error_patterns.contains(&pattern) {
                    self.error_patterns.push(pattern);
                }
            }
        }
    }

    fn calculate_error_rate(&self) -> f64 {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let minute_ago = now - 60;

        let errors_last_minute =
            self.error_timestamps.iter().filter(|&&ts| ts > minute_ago).count();

        errors_last_minute as f64
    }

    fn get_most_frequent_error(&self) -> TrustformersError {
        self.error_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&error, _)| error)
            .unwrap_or(TrustformersError::Success)
    }
}

/// Record an error in the tracking system
#[no_mangle]
pub extern "C" fn trustformers_error_record(
    error_code: TrustformersError,
    function_name: *const c_char,
    file_name: *const c_char,
    line_number: c_int,
    message: *const c_char,
    severity: c_int,
) -> TrustformersError {
    let function_name_str = if function_name.is_null() {
        "unknown".to_string()
    } else {
        match c_str_to_string(function_name) {
            Ok(s) => s,
            Err(_) => "unknown".to_string(),
        }
    };

    let file_name_str = if file_name.is_null() {
        "unknown".to_string()
    } else {
        match c_str_to_string(file_name) {
            Ok(s) => s,
            Err(_) => "unknown".to_string(),
        }
    };

    let message_str = if message.is_null() {
        "".to_string()
    } else {
        match c_str_to_string(message) {
            Ok(s) => s,
            Err(_) => "".to_string(),
        }
    };

    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

    let record = ErrorRecord {
        error_code,
        timestamp,
        function_name: function_name_str,
        file_name: file_name_str,
        line_number,
        message: message_str,
        severity,
    };

    if let Ok(mut tracker) = ERROR_TRACKER.write() {
        tracker.record_error(record);
    }

    TrustformersError::Success
}

/// Get comprehensive error diagnostics
#[no_mangle]
pub extern "C" fn trustformers_error_get_diagnostics(
    diagnostics: *mut TrustformersErrorDiagnostics,
) -> TrustformersError {
    if diagnostics.is_null() {
        return TrustformersError::NullPointer;
    }

    let tracker = match ERROR_TRACKER.read() {
        Ok(t) => t,
        Err(_) => return TrustformersError::ConcurrencyError,
    };

    unsafe {
        let diag = &mut *diagnostics;

        diag.total_errors = tracker.total_errors;
        diag.most_frequent_error = tracker.get_most_frequent_error();
        diag.error_rate = tracker.calculate_error_rate();
        diag.health_score = tracker.health_score;

        // Serialize error counts to JSON
        if let Ok(json) = serde_json::to_string(&tracker.error_counts) {
            diag.error_counts_json = string_to_c_str(json);
        }

        // Serialize recent errors to JSON
        let recent_errors_simplified: Vec<serde_json::Value> = tracker
            .recent_errors
            .iter()
            .map(|r| {
                serde_json::json!({
                    "error_code": r.error_code as i32,
                    "timestamp": r.timestamp,
                    "function": r.function_name,
                    "message": r.message,
                    "severity": r.severity
                })
            })
            .collect();

        if let Ok(json) = serde_json::to_string(&recent_errors_simplified) {
            diag.recent_errors_json = string_to_c_str(json);
        }

        // Serialize error patterns to JSON
        if let Ok(json) = serde_json::to_string(&tracker.error_patterns) {
            diag.error_patterns_json = string_to_c_str(json);
        }
    }

    TrustformersError::Success
}

/// Get detailed error context for a specific error
#[no_mangle]
pub extern "C" fn trustformers_error_get_context(
    error_code: TrustformersError,
    context: *mut TrustformersErrorContext,
) -> TrustformersError {
    if context.is_null() {
        return TrustformersError::NullPointer;
    }

    let tracker = match ERROR_TRACKER.read() {
        Ok(t) => t,
        Err(_) => return TrustformersError::ConcurrencyError,
    };

    // Find the most recent occurrence of this error
    let recent_error = tracker.recent_errors.iter().rev().find(|r| r.error_code == error_code);

    unsafe {
        let ctx = &mut *context;
        ctx.error_code = error_code;

        if let Some(error_record) = recent_error {
            ctx.message = string_to_c_str(error_record.message.clone());
            ctx.function_name = string_to_c_str(error_record.function_name.clone());
            ctx.file_name = string_to_c_str(error_record.file_name.clone());
            ctx.line_number = error_record.line_number;
            ctx.timestamp = error_record.timestamp;
            ctx.severity = error_record.severity;
        } else {
            let msg_ptr = trustformers_error_message(error_code);
            let msg_str = if !msg_ptr.is_null() {
                unsafe { std::ffi::CStr::from_ptr(msg_ptr).to_string_lossy().to_string() }
            } else {
                "Unknown error".to_string()
            };
            ctx.message = string_to_c_str(msg_str);
        }

        // Set category based on error type
        let category = match error_code {
            TrustformersError::ModelLoadError => "model",
            TrustformersError::TokenizerError => "tokenizer",
            TrustformersError::PipelineError => "pipeline",
            TrustformersError::TensorError => "tensor",
            TrustformersError::InferenceError => "inference",
            _ => "general",
        };
        ctx.category = string_to_c_str(category.to_string());

        // Generate recovery suggestions
        let suggestions = generate_recovery_suggestions(error_code);
        if let Ok(json) = serde_json::to_string(&suggestions) {
            ctx.recovery_suggestions = string_to_c_str(json);
        }
    }

    TrustformersError::Success
}

/// Attempt automatic error recovery
#[no_mangle]
pub extern "C" fn trustformers_error_attempt_recovery(
    error_code: TrustformersError,
    recovery_config: *const TrustformersErrorRecovery,
    success: *mut c_int,
) -> TrustformersError {
    if success.is_null() {
        return TrustformersError::NullPointer;
    }

    let config = if recovery_config.is_null() {
        TrustformersErrorRecovery::default()
    } else {
        unsafe { std::ptr::read(recovery_config) }
    };

    // Simulate recovery attempt based on strategy
    let recovery_successful = match config.strategy {
        0 => attempt_retry_recovery(error_code, config.max_retries),
        1 => attempt_reset_recovery(error_code),
        2 => attempt_fallback_recovery(error_code, config.fallback_config),
        3 => false, // Abort - no recovery
        _ => false,
    };

    unsafe {
        *success = if recovery_successful { 1 } else { 0 };
    }

    TrustformersError::Success
}

/// Clear error tracking history
#[no_mangle]
pub extern "C" fn trustformers_error_clear_history() -> TrustformersError {
    if let Ok(mut tracker) = ERROR_TRACKER.write() {
        tracker.recent_errors.clear();
        tracker.error_patterns.clear();
        tracker.error_timestamps.clear();
        tracker.error_counts.clear();
        tracker.total_errors = 0;
        tracker.health_score = 100.0;
    }

    TrustformersError::Success
}

/// Free error diagnostics memory
#[no_mangle]
pub extern "C" fn trustformers_error_diagnostics_free(
    diagnostics: *mut TrustformersErrorDiagnostics,
) {
    if diagnostics.is_null() {
        return;
    }

    unsafe {
        let diag = &mut *diagnostics;

        if !diag.error_counts_json.is_null() {
            let _ = CString::from_raw(diag.error_counts_json);
            diag.error_counts_json = ptr::null_mut();
        }

        if !diag.recent_errors_json.is_null() {
            let _ = CString::from_raw(diag.recent_errors_json);
            diag.recent_errors_json = ptr::null_mut();
        }

        if !diag.error_patterns_json.is_null() {
            let _ = CString::from_raw(diag.error_patterns_json);
            diag.error_patterns_json = ptr::null_mut();
        }
    }
}

/// Free error context memory
#[no_mangle]
pub extern "C" fn trustformers_error_context_free(context: *mut TrustformersErrorContext) {
    if context.is_null() {
        return;
    }

    unsafe {
        let ctx = &mut *context;

        if !ctx.message.is_null() {
            let _ = CString::from_raw(ctx.message);
            ctx.message = ptr::null_mut();
        }

        if !ctx.description.is_null() {
            let _ = CString::from_raw(ctx.description);
            ctx.description = ptr::null_mut();
        }

        if !ctx.category.is_null() {
            let _ = CString::from_raw(ctx.category);
            ctx.category = ptr::null_mut();
        }

        if !ctx.function_name.is_null() {
            let _ = CString::from_raw(ctx.function_name);
            ctx.function_name = ptr::null_mut();
        }

        if !ctx.file_name.is_null() {
            let _ = CString::from_raw(ctx.file_name);
            ctx.file_name = ptr::null_mut();
        }

        if !ctx.stack_trace.is_null() {
            let _ = CString::from_raw(ctx.stack_trace);
            ctx.stack_trace = ptr::null_mut();
        }

        if !ctx.recovery_suggestions.is_null() {
            let _ = CString::from_raw(ctx.recovery_suggestions);
            ctx.recovery_suggestions = ptr::null_mut();
        }

        if !ctx.related_errors.is_null() && ctx.num_related_errors > 0 {
            let layout =
                std::alloc::Layout::array::<TrustformersError>(ctx.num_related_errors).unwrap();
            std::alloc::dealloc(ctx.related_errors as *mut u8, layout);
            ctx.related_errors = ptr::null_mut();
        }
    }
}

/// Helper function to generate recovery suggestions
fn generate_recovery_suggestions(error_code: TrustformersError) -> Vec<String> {
    match error_code {
        TrustformersError::NullPointer => vec![
            "Check that all required parameters are provided".to_string(),
            "Ensure pointers are properly initialized".to_string(),
        ],
        TrustformersError::OutOfMemory => vec![
            "Free unused resources".to_string(),
            "Reduce batch size or model size".to_string(),
            "Use quantization to reduce memory usage".to_string(),
        ],
        TrustformersError::FileNotFound => vec![
            "Check file path is correct".to_string(),
            "Ensure file exists and is accessible".to_string(),
            "Verify permissions to read the file".to_string(),
        ],
        TrustformersError::ModelLoadError => vec![
            "Check model format compatibility".to_string(),
            "Verify model files are not corrupted".to_string(),
            "Try different backend or device".to_string(),
        ],
        TrustformersError::TokenizerError => vec![
            "Check tokenizer configuration".to_string(),
            "Verify input text encoding".to_string(),
            "Ensure vocabulary file is accessible".to_string(),
        ],
        TrustformersError::DeviceNotAvailable => vec![
            "Check CUDA installation and drivers".to_string(),
            "Fall back to CPU device".to_string(),
            "Verify device ID is valid".to_string(),
        ],
        TrustformersError::ValidationError => vec![
            "Check input parameters are valid".to_string(),
            "Verify data format and constraints".to_string(),
            "Review validation rules".to_string(),
        ],
        TrustformersError::InvalidHandle => vec![
            "Ensure handle was created successfully".to_string(),
            "Check handle was not freed previously".to_string(),
            "Verify handle type matches expected type".to_string(),
        ],
        _ => vec![
            "Check logs for more details".to_string(),
            "Retry the operation".to_string(),
        ],
    }
}

/// Helper function for retry recovery
fn attempt_retry_recovery(error_code: TrustformersError, max_retries: c_int) -> bool {
    // Simple retry logic - in practice this would be more sophisticated
    match error_code {
        TrustformersError::NetworkError | TrustformersError::Timeout => max_retries > 0,
        _ => false,
    }
}

/// Helper function for reset recovery
fn attempt_reset_recovery(error_code: TrustformersError) -> bool {
    // Reset recovery is applicable for certain error types
    match error_code {
        TrustformersError::ConcurrencyError | TrustformersError::RuntimeError => true,
        _ => false,
    }
}

/// Helper function for fallback recovery
fn attempt_fallback_recovery(
    error_code: TrustformersError,
    _fallback_config: *const c_char,
) -> bool {
    // Fallback recovery depends on having alternative configurations
    match error_code {
        TrustformersError::DeviceNotAvailable | TrustformersError::FeatureNotAvailable => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_messages() {
        let success_msg = trustformers_error_message(TrustformersError::Success);
        assert!(!success_msg.is_null());

        let error_msg = trustformers_error_message(TrustformersError::RuntimeError);
        assert!(!error_msg.is_null());
    }

    #[test]
    fn test_error_checks() {
        assert_eq!(trustformers_is_success(TrustformersError::Success), 1);
        assert_eq!(trustformers_is_success(TrustformersError::RuntimeError), 0);

        assert_eq!(trustformers_is_error(TrustformersError::Success), 0);
        assert_eq!(trustformers_is_error(TrustformersError::RuntimeError), 1);
    }

    #[test]
    fn test_anyhow_conversion() {
        let anyhow_error = anyhow::anyhow!("Model loading failed");
        let c_error: TrustformersError = anyhow_error.into();
        assert_eq!(c_error, TrustformersError::ModelLoadError);
    }
}
