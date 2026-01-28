/// Interactive Tensor Debugger for TrustformeRS
///
/// This module provides comprehensive debugging tools for tensor operations,
/// gradient flow analysis, and interactive debugging features.
use crate::errors::Result;
use crate::tensor::{DType, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Severity level for debugger issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    /// Informational message
    Info,
    /// Warning about potential issues
    Warning,
    /// Error that should be addressed
    Error,
    /// Critical issue requiring immediate attention
    Critical,
}

/// Type of issue detected by the tensor debugger
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorIssueType {
    /// NaN values detected
    NaN,
    /// Infinite values detected
    Infinity,
    /// Gradient vanishing (very small values)
    VanishingGradient,
    /// Gradient exploding (very large values)
    ExplodingGradient,
    /// All zeros in tensor
    AllZeros,
    /// Unusual value distribution
    UnusualDistribution,
    /// Memory leak suspected
    MemoryLeak,
    /// Dtype mismatch
    DTypeMismatch,
    /// Shape mismatch
    ShapeMismatch,
    /// Operation failure
    OperationFailure,
}

impl fmt::Display for TensorIssueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorIssueType::NaN => write!(f, "NaN Values"),
            TensorIssueType::Infinity => write!(f, "Infinite Values"),
            TensorIssueType::VanishingGradient => write!(f, "Vanishing Gradient"),
            TensorIssueType::ExplodingGradient => write!(f, "Exploding Gradient"),
            TensorIssueType::AllZeros => write!(f, "All Zeros"),
            TensorIssueType::UnusualDistribution => write!(f, "Unusual Distribution"),
            TensorIssueType::MemoryLeak => write!(f, "Memory Leak"),
            TensorIssueType::DTypeMismatch => write!(f, "DType Mismatch"),
            TensorIssueType::ShapeMismatch => write!(f, "Shape Mismatch"),
            TensorIssueType::OperationFailure => write!(f, "Operation Failure"),
        }
    }
}

/// Issue detected by the tensor debugger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDebugIssue {
    /// Type of issue
    pub issue_type: TensorIssueType,
    /// Severity level
    pub severity: Severity,
    /// Human-readable message
    pub message: String,
    /// Tensor name (if available)
    pub tensor_name: Option<String>,
    /// Operation name (if available)
    pub operation: Option<String>,
    /// Location in code (file:line)
    pub location: Option<String>,
    /// Timestamp when issue was detected
    pub timestamp: std::time::SystemTime,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl TensorDebugIssue {
    fn new(issue_type: TensorIssueType, severity: Severity, message: String) -> Self {
        Self {
            issue_type,
            severity,
            message,
            tensor_name: None,
            operation: None,
            location: None,
            timestamp: std::time::SystemTime::now(),
            metadata: HashMap::new(),
        }
    }

    /// Add tensor name to issue
    pub fn with_tensor_name(mut self, name: String) -> Self {
        self.tensor_name = Some(name);
        self
    }

    /// Add operation name to issue
    pub fn with_operation(mut self, op: String) -> Self {
        self.operation = Some(op);
        self
    }

    /// Add source location
    pub fn with_location(mut self, location: String) -> Self {
        self.location = Some(location);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Statistics for a tensor (debugger)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugTensorStats {
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: DType,
    /// Minimum value (if applicable)
    pub min: Option<f64>,
    /// Maximum value (if applicable)
    pub max: Option<f64>,
    /// Mean value (if applicable)
    pub mean: Option<f64>,
    /// Standard deviation (if applicable)
    pub std_dev: Option<f64>,
    /// Number of NaN values
    pub nan_count: usize,
    /// Number of infinite values
    pub inf_count: usize,
    /// Number of zero values
    pub zero_count: usize,
    /// Total number of elements
    pub total_elements: usize,
    /// Memory usage in bytes
    pub memory_bytes: usize,
}

impl DebugTensorStats {
    /// Compute statistics from a tensor
    pub fn from_tensor(tensor: &Tensor) -> Result<Self> {
        let shape = tensor.shape().to_vec();
        let dtype = tensor.dtype();
        let total_elements = shape.iter().product();

        // For simplicity, only compute full stats for F32 tensors
        let (min, max, mean, std_dev, nan_count, inf_count, zero_count) = match tensor {
            Tensor::F32(arr) => {
                let data: Vec<f32> = arr.iter().copied().collect();
                let mut min_val = f64::INFINITY;
                let mut max_val = f64::NEG_INFINITY;
                let mut sum = 0.0;
                let mut nan_count = 0;
                let mut inf_count = 0;
                let mut zero_count = 0;

                for &val in &data {
                    if val.is_nan() {
                        nan_count += 1;
                        continue;
                    }
                    if val.is_infinite() {
                        inf_count += 1;
                        continue;
                    }
                    if val == 0.0 {
                        zero_count += 1;
                    }

                    let val_f64 = val as f64;
                    min_val = min_val.min(val_f64);
                    max_val = max_val.max(val_f64);
                    sum += val_f64;
                }

                let count = (data.len() - nan_count - inf_count) as f64;
                let mean = if count > 0.0 { sum / count } else { 0.0 };

                // Compute std dev
                let mut sum_sq_diff = 0.0;
                for &val in &data {
                    if !val.is_nan() && !val.is_infinite() {
                        let diff = val as f64 - mean;
                        sum_sq_diff += diff * diff;
                    }
                }
                let std_dev = if count > 0.0 { (sum_sq_diff / count).sqrt() } else { 0.0 };

                (
                    Some(min_val),
                    Some(max_val),
                    Some(mean),
                    Some(std_dev),
                    nan_count,
                    inf_count,
                    zero_count,
                )
            },
            Tensor::F64(arr) => {
                let data: Vec<f64> = arr.iter().copied().collect();
                let mut min_val = f64::INFINITY;
                let mut max_val = f64::NEG_INFINITY;
                let mut sum = 0.0;
                let mut nan_count = 0;
                let mut inf_count = 0;
                let mut zero_count = 0;

                for &val in &data {
                    if val.is_nan() {
                        nan_count += 1;
                        continue;
                    }
                    if val.is_infinite() {
                        inf_count += 1;
                        continue;
                    }
                    if val == 0.0 {
                        zero_count += 1;
                    }

                    min_val = min_val.min(val);
                    max_val = max_val.max(val);
                    sum += val;
                }

                let count = (data.len() - nan_count - inf_count) as f64;
                let mean = if count > 0.0 { sum / count } else { 0.0 };

                // Compute std dev
                let mut sum_sq_diff = 0.0;
                for &val in &data {
                    if !val.is_nan() && !val.is_infinite() {
                        let diff = val - mean;
                        sum_sq_diff += diff * diff;
                    }
                }
                let std_dev = if count > 0.0 { (sum_sq_diff / count).sqrt() } else { 0.0 };

                (
                    Some(min_val),
                    Some(max_val),
                    Some(mean),
                    Some(std_dev),
                    nan_count,
                    inf_count,
                    zero_count,
                )
            },
            _ => (None, None, None, None, 0, 0, 0),
        };

        let memory_bytes = total_elements * dtype.size_in_bytes();

        Ok(Self {
            shape,
            dtype,
            min,
            max,
            mean,
            std_dev,
            nan_count,
            inf_count,
            zero_count,
            total_elements,
            memory_bytes,
        })
    }

    /// Check for potential issues
    pub fn detect_issues(&self) -> Vec<TensorDebugIssue> {
        let mut issues = Vec::new();

        // Check for NaN values
        if self.nan_count > 0 {
            issues.push(
                TensorDebugIssue::new(
                    TensorIssueType::NaN,
                    Severity::Error,
                    format!(
                        "Found {} NaN values out of {}",
                        self.nan_count, self.total_elements
                    ),
                )
                .with_metadata("nan_count".to_string(), self.nan_count.to_string())
                .with_metadata(
                    "nan_percentage".to_string(),
                    format!(
                        "{:.2}%",
                        100.0 * self.nan_count as f64 / self.total_elements as f64
                    ),
                ),
            );
        }

        // Check for infinite values
        if self.inf_count > 0 {
            issues.push(
                TensorDebugIssue::new(
                    TensorIssueType::Infinity,
                    Severity::Error,
                    format!(
                        "Found {} infinite values out of {}",
                        self.inf_count, self.total_elements
                    ),
                )
                .with_metadata("inf_count".to_string(), self.inf_count.to_string()),
            );
        }

        // Check for all zeros
        if self.zero_count == self.total_elements {
            issues.push(TensorDebugIssue::new(
                TensorIssueType::AllZeros,
                Severity::Warning,
                "Tensor contains all zeros".to_string(),
            ));
        }

        // Check for vanishing values (very small)
        if let (Some(max_val), Some(mean_val)) = (self.max, self.mean) {
            if max_val.abs() < 1e-7 && mean_val.abs() < 1e-7 {
                issues.push(
                    TensorDebugIssue::new(
                        TensorIssueType::VanishingGradient,
                        Severity::Warning,
                        format!(
                            "Very small values detected (max: {:.2e}, mean: {:.2e})",
                            max_val, mean_val
                        ),
                    )
                    .with_metadata("max_value".to_string(), format!("{:.2e}", max_val))
                    .with_metadata("mean_value".to_string(), format!("{:.2e}", mean_val)),
                );
            }
        }

        // Check for exploding values (very large)
        if let Some(max_val) = self.max {
            if max_val.abs() > 1e6 {
                issues.push(
                    TensorDebugIssue::new(
                        TensorIssueType::ExplodingGradient,
                        Severity::Error,
                        format!("Very large values detected (max: {:.2e})", max_val),
                    )
                    .with_metadata("max_value".to_string(), format!("{:.2e}", max_val)),
                );
            }
        }

        issues
    }
}

/// Operation trace entry
#[derive(Debug, Clone)]
pub struct OperationTrace {
    /// Operation name
    pub operation: String,
    /// Input tensor names
    pub inputs: Vec<String>,
    /// Output tensor name
    pub output: String,
    /// Timestamp (not serializable)
    pub timestamp: Instant,
    /// Duration
    pub duration: std::time::Duration,
    /// Input shapes
    pub input_shapes: Vec<Vec<usize>>,
    /// Output shape
    pub output_shape: Vec<usize>,
}

/// Watchpoint condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WatchCondition {
    /// Watch for NaN values
    HasNaN,
    /// Watch for infinite values
    HasInf,
    /// Watch for values exceeding threshold
    ValueExceeds(f64),
    /// Watch for values below threshold
    ValueBelow(f64),
    /// Watch for specific shape
    ShapeEquals(Vec<usize>),
    /// Custom condition (string description)
    Custom(String),
}

/// Watchpoint on a tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Watchpoint {
    /// Tensor name pattern (supports wildcards)
    pub tensor_pattern: String,
    /// Condition to watch for
    pub condition: WatchCondition,
    /// Whether to break on condition
    pub break_on_trigger: bool,
    /// Number of times triggered
    pub trigger_count: usize,
}

/// Tensor debugger configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDebuggerConfig {
    /// Enable automatic issue detection
    pub auto_detect_issues: bool,
    /// Enable operation tracing
    pub enable_tracing: bool,
    /// Maximum number of trace entries to keep
    pub max_trace_entries: usize,
    /// Enable watchpoints
    pub enable_watchpoints: bool,
    /// Break on errors
    pub break_on_error: bool,
    /// Break on warnings
    pub break_on_warning: bool,
    /// Maximum number of issues to track
    pub max_issues: usize,
}

impl Default for TensorDebuggerConfig {
    fn default() -> Self {
        Self {
            auto_detect_issues: true,
            enable_tracing: true,
            max_trace_entries: 1000,
            enable_watchpoints: true,
            break_on_error: true,
            break_on_warning: false,
            max_issues: 100,
        }
    }
}

/// Interactive tensor debugger
pub struct TensorDebugger {
    config: TensorDebuggerConfig,
    /// Named tensors being tracked
    tensors: Arc<Mutex<HashMap<String, Tensor>>>,
    /// Detected issues
    issues: Arc<Mutex<VecDeque<TensorDebugIssue>>>,
    /// Operation traces
    traces: Arc<Mutex<VecDeque<OperationTrace>>>,
    /// Active watchpoints
    watchpoints: Arc<Mutex<Vec<Watchpoint>>>,
    /// Breakpoint flag
    breakpoint_hit: Arc<Mutex<bool>>,
    /// Statistics cache
    stats_cache: Arc<Mutex<HashMap<String, DebugTensorStats>>>,
}

impl TensorDebugger {
    /// Create a new tensor debugger with default configuration
    pub fn new() -> Self {
        Self::with_config(TensorDebuggerConfig::default())
    }

    /// Create a new tensor debugger with custom configuration
    pub fn with_config(config: TensorDebuggerConfig) -> Self {
        Self {
            config,
            tensors: Arc::new(Mutex::new(HashMap::new())),
            issues: Arc::new(Mutex::new(VecDeque::new())),
            traces: Arc::new(Mutex::new(VecDeque::new())),
            watchpoints: Arc::new(Mutex::new(Vec::new())),
            breakpoint_hit: Arc::new(Mutex::new(false)),
            stats_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Register a tensor for debugging
    pub fn register_tensor(&self, name: String, tensor: Tensor) -> Result<()> {
        let mut tensors = self.tensors.lock().expect("Lock poisoned");
        tensors.insert(name.clone(), tensor.clone());

        // Compute and cache statistics
        let stats = DebugTensorStats::from_tensor(&tensor)?;

        {
            let mut cache = self.stats_cache.lock().expect("Lock poisoned");
            cache.insert(name.clone(), stats.clone());
        }

        // Auto-detect issues if enabled
        if self.config.auto_detect_issues {
            let detected_issues = stats.detect_issues();
            if !detected_issues.is_empty() {
                let mut issues = self.issues.lock().expect("Lock poisoned");
                for mut issue in detected_issues {
                    issue = issue.with_tensor_name(name.clone());

                    // Check if we should break
                    if (issue.severity == Severity::Error && self.config.break_on_error)
                        || (issue.severity == Severity::Warning && self.config.break_on_warning)
                    {
                        *self.breakpoint_hit.lock().expect("Lock poisoned") = true;
                    }

                    issues.push_back(issue);

                    // Limit issue queue
                    while issues.len() > self.config.max_issues {
                        issues.pop_front();
                    }
                }
            }
        }

        // Check watchpoints if enabled
        if self.config.enable_watchpoints {
            self.check_watchpoints(&name, &tensor)?;
        }

        Ok(())
    }

    /// Add a watchpoint
    pub fn add_watchpoint(&self, watchpoint: Watchpoint) {
        let mut watchpoints = self.watchpoints.lock().expect("Lock poisoned");
        watchpoints.push(watchpoint);
    }

    /// Remove all watchpoints matching pattern
    pub fn remove_watchpoint(&self, pattern: &str) {
        let mut watchpoints = self.watchpoints.lock().expect("Lock poisoned");
        watchpoints.retain(|w| w.tensor_pattern != pattern);
    }

    /// Check watchpoints for a tensor
    fn check_watchpoints(&self, name: &str, tensor: &Tensor) -> Result<()> {
        let mut watchpoints = self.watchpoints.lock().expect("Lock poisoned");

        for wp in watchpoints.iter_mut() {
            // Simple pattern matching (exact match for now)
            if name == wp.tensor_pattern || wp.tensor_pattern == "*" {
                let triggered = match &wp.condition {
                    WatchCondition::HasNaN => {
                        let stats = DebugTensorStats::from_tensor(tensor)?;
                        stats.nan_count > 0
                    },
                    WatchCondition::HasInf => {
                        let stats = DebugTensorStats::from_tensor(tensor)?;
                        stats.inf_count > 0
                    },
                    WatchCondition::ValueExceeds(threshold) => {
                        let stats = DebugTensorStats::from_tensor(tensor)?;
                        stats.max.is_some_and(|max| max.abs() > *threshold)
                    },
                    WatchCondition::ValueBelow(threshold) => {
                        let stats = DebugTensorStats::from_tensor(tensor)?;
                        stats.min.is_some_and(|min| min.abs() < *threshold)
                    },
                    WatchCondition::ShapeEquals(expected_shape) => {
                        tensor.shape() == expected_shape.as_slice()
                    },
                    WatchCondition::Custom(_) => false, // Custom conditions not implemented
                };

                if triggered {
                    wp.trigger_count += 1;

                    if wp.break_on_trigger {
                        *self.breakpoint_hit.lock().expect("Lock poisoned") = true;
                    }

                    // Log issue
                    let issue = TensorDebugIssue::new(
                        TensorIssueType::OperationFailure,
                        Severity::Warning,
                        format!("Watchpoint triggered: {:?}", wp.condition),
                    )
                    .with_tensor_name(name.to_string())
                    .with_metadata("trigger_count".to_string(), wp.trigger_count.to_string());

                    let mut issues = self.issues.lock().expect("Lock poisoned");
                    issues.push_back(issue);
                }
            }
        }

        Ok(())
    }

    /// Get tensor by name
    pub fn get_tensor(&self, name: &str) -> Option<Tensor> {
        let tensors = self.tensors.lock().expect("Lock poisoned");
        tensors.get(name).cloned()
    }

    /// Get statistics for a tensor
    pub fn get_stats(&self, name: &str) -> Option<DebugTensorStats> {
        let cache = self.stats_cache.lock().expect("Lock poisoned");
        cache.get(name).cloned()
    }

    /// Get all issues
    pub fn get_issues(&self) -> Vec<TensorDebugIssue> {
        let issues = self.issues.lock().expect("Lock poisoned");
        issues.iter().cloned().collect()
    }

    /// Clear all issues
    pub fn clear_issues(&self) {
        let mut issues = self.issues.lock().expect("Lock poisoned");
        issues.clear();
    }

    /// Get operation traces
    pub fn get_traces(&self) -> Vec<OperationTrace> {
        let traces = self.traces.lock().expect("Lock poisoned");
        traces.iter().cloned().collect()
    }

    /// Clear traces
    pub fn clear_traces(&self) {
        let mut traces = self.traces.lock().expect("Lock poisoned");
        traces.clear();
    }

    /// Check if breakpoint was hit
    pub fn is_breakpoint_hit(&self) -> bool {
        *self.breakpoint_hit.lock().expect("Lock poisoned")
    }

    /// Clear breakpoint flag
    pub fn clear_breakpoint(&self) {
        *self.breakpoint_hit.lock().expect("Lock poisoned") = false;
    }

    /// Print summary of all tracked tensors
    pub fn print_summary(&self) {
        println!("\n=== Tensor Debugger Summary ===\n");

        let cache = self.stats_cache.lock().expect("Lock poisoned");
        println!("Tracked Tensors: {}", cache.len());

        for (name, stats) in cache.iter() {
            println!("\nTensor: {}", name);
            println!("  Shape: {:?}", stats.shape);
            println!("  DType: {:?}", stats.dtype);
            println!("  Elements: {}", stats.total_elements);
            println!("  Memory: {} bytes", stats.memory_bytes);

            if let Some(min) = stats.min {
                println!("  Min: {:.6}", min);
            }
            if let Some(max) = stats.max {
                println!("  Max: {:.6}", max);
            }
            if let Some(mean) = stats.mean {
                println!("  Mean: {:.6}", mean);
            }
            if let Some(std) = stats.std_dev {
                println!("  Std Dev: {:.6}", std);
            }

            if stats.nan_count > 0 {
                println!("  ⚠️  NaN count: {}", stats.nan_count);
            }
            if stats.inf_count > 0 {
                println!("  ⚠️  Inf count: {}", stats.inf_count);
            }
        }

        let issues = self.issues.lock().expect("Lock poisoned");
        if !issues.is_empty() {
            println!("\n=== Issues ({}) ===\n", issues.len());
            for (i, issue) in issues.iter().enumerate() {
                println!(
                    "{}. [{:?}] {}: {}",
                    i + 1,
                    issue.severity,
                    issue.issue_type,
                    issue.message
                );
                if let Some(tensor_name) = &issue.tensor_name {
                    println!("   Tensor: {}", tensor_name);
                }
            }
        }

        println!("\n==============================\n");
    }
}

impl Default for TensorDebugger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debugger_creation() {
        let debugger = TensorDebugger::new();
        assert!(!debugger.is_breakpoint_hit());
        assert_eq!(debugger.get_issues().len(), 0);
    }

    #[test]
    fn test_tensor_registration() -> Result<()> {
        let debugger = TensorDebugger::new();
        let tensor = Tensor::ones(&[2, 3])?;

        debugger.register_tensor("test_tensor".to_string(), tensor.clone())?;

        let retrieved = debugger.get_tensor("test_tensor");
        assert!(retrieved.is_some());

        let stats = debugger.get_stats("test_tensor");
        assert!(stats.is_some());

        let stats = stats.unwrap();
        assert_eq!(stats.shape, vec![2, 3]);
        assert_eq!(stats.total_elements, 6);

        Ok(())
    }

    #[test]
    fn test_nan_detection() -> Result<()> {
        let debugger = TensorDebugger::new();

        // Create tensor with NaN
        let data = vec![1.0, 2.0, f32::NAN, 4.0];
        let tensor = Tensor::from_slice(&data, &[4])?;

        debugger.register_tensor("nan_tensor".to_string(), tensor)?;

        let issues = debugger.get_issues();
        assert!(!issues.is_empty());

        let has_nan_issue = issues.iter().any(|i| i.issue_type == TensorIssueType::NaN);
        assert!(has_nan_issue);

        Ok(())
    }

    #[test]
    fn test_watchpoint() -> Result<()> {
        let debugger = TensorDebugger::new();

        // Add watchpoint for NaN
        let wp = Watchpoint {
            tensor_pattern: "watched".to_string(),
            condition: WatchCondition::HasNaN,
            break_on_trigger: true,
            trigger_count: 0,
        };
        debugger.add_watchpoint(wp);

        // Create tensor with NaN
        let data = vec![1.0, f32::NAN];
        let tensor = Tensor::from_slice(&data, &[2])?;

        debugger.register_tensor("watched".to_string(), tensor)?;

        // Breakpoint should be hit
        assert!(debugger.is_breakpoint_hit());

        Ok(())
    }
}
