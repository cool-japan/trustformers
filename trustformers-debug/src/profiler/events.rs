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
    pub heap_allocated: usize,
    pub heap_used: usize,
    pub stack_size: usize,
    pub gpu_allocated: Option<usize>,
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
    pub thread_id: u64,
    pub cpu_usage: f64,
    pub context_switches: u64,
    pub cache_misses: u64,
    pub instructions_per_cycle: f64,
    pub branch_mispredictions: u64,
    pub hot_functions: Vec<HotFunction>,
    pub bottleneck_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotFunction {
    pub function_name: String,
    pub self_time_percentage: f64,
    pub call_count: usize,
    pub avg_time_per_call: Duration,
}
