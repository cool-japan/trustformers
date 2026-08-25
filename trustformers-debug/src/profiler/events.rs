//! Profiling event types and related data structures

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Profiling event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProfileEvent {
    FunctionCall {
        function_name: String,
        duration: Duration,
        memory_delta: i64,
    },
    LayerExecution {
        layer_name: String,
        layer_type: String,
        forward_time: Duration,
        backward_time: Option<Duration>,
        memory_usage: usize,
        parameter_count: usize,
    },
    TensorOperation {
        operation: String,
        tensor_shape: Vec<usize>,
        duration: Duration,
        memory_allocated: usize,
    },
    ModelInference {
        batch_size: usize,
        sequence_length: usize,
        duration: Duration,
        tokens_per_second: f64,
    },
    GradientComputation {
        layer_name: String,
        gradient_norm: f64,
        duration: Duration,
    },
}

/// Profiling statistics for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileStats {
    pub event_type: String,
    pub count: usize,
    pub total_duration: Duration,
    pub avg_duration: Duration,
    pub min_duration: Duration,
    pub max_duration: Duration,
    pub total_memory: i64,
    pub avg_memory: f64,
}

/// Memory usage snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Resident set size of this process in bytes, from `sysinfo`.
    ///
    /// `None` when the platform's process table does not list this PID. It
    /// used to be a `usize` hardcoded to `0`, i.e. "this process holds no
    /// memory", published as a measurement.
    pub process_rss_bytes: Option<usize>,
    /// Virtual memory size of this process in bytes, from `sysinfo`.
    pub process_virtual_bytes: Option<usize>,
    /// GPU memory allocated. Always `None`: this crate links no GPU driver.
    pub gpu_allocated: Option<usize>,
    /// GPU memory in use. Always `None`, for the same reason.
    pub gpu_used: Option<usize>,
}

/// Performance bottleneck detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    pub bottleneck_type: BottleneckType,
    pub location: String,
    pub severity: BottleneckSeverity,
    pub description: String,
    pub suggestion: String,
    pub metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    CpuBound,
    MemoryBound,
    IoBound,
    GpuBound,
    NetworkBound,
    DataLoading,
    ModelComputation,
    GradientComputation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// CPU profiling information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuProfile {
    pub function_name: String,
    pub self_time: Duration,
    pub total_time: Duration,
    pub call_count: usize,
    pub children: Vec<CpuProfile>,
}

/// CPU bottleneck analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuBottleneckAnalysis {
    /// Id of the process the reading belongs to (this process).
    pub process_id: u32,
    /// Real CPU usage percentage of this process, from `sysinfo`; `None` when
    /// the platform gave no reading. It used to be a hardcoded `0.75`.
    pub cpu_usage_percent: Option<f64>,
    /// Voluntary + involuntary context switches.
    ///
    /// Always `None`: reading them needs `getrusage`/`/proc/self/status`
    /// parsing that this Pure-Rust crate does not do. Previously hardcoded to
    /// `1000`.
    pub context_switches: Option<u64>,
    /// Cache misses. Always `None`: a hardware performance counter, which
    /// needs `perf_event_open`/PMU access. Previously hardcoded to `500`.
    pub cache_misses: Option<u64>,
    /// Instructions per cycle. Always `None`: also a PMU counter. Previously
    /// hardcoded to `2.5`.
    pub instructions_per_cycle: Option<f64>,
    /// Branch mispredictions. Always `None`: also a PMU counter. Previously
    /// hardcoded to `100`.
    pub branch_mispredictions: Option<u64>,
    /// Hottest functions by self time.
    ///
    /// Built from this profiler's own recorded operation timings. It used to
    /// be a two-element literal naming `tensor_multiply` and
    /// `gradient_computation` with invented call counts, regardless of what
    /// had been profiled.
    pub hot_functions: Vec<HotFunction>,
    /// Share of total recorded time spent in the single hottest function, in
    /// `[0, 1]`; `None` when nothing has been profiled.
    pub bottleneck_score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotFunction {
    pub function_name: String,
    pub self_time_percentage: f64,
    pub call_count: usize,
    pub avg_time_per_call: Duration,
}
