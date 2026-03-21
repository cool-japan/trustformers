//! Type definitions for debug utilities
//!
//! This module contains all the data structures used for debugging,
//! profiling, and introspection of models.

use crate::memory_safety::MemorySafetyVerifier;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Debug information about a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelIntrospection {
    pub model_id: String,
    pub model_name: String,
    pub model_type: String,
    pub parameters_count: u64,
    pub layers_count: usize,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub memory_usage_bytes: u64,
    pub created_at: String,
    pub last_used: String,
    pub usage_count: u64,
    pub avg_inference_time_ms: f64,
    pub layer_info: Vec<LayerInfo>,
    pub quantization_info: Option<QuantizationInfo>,
}

/// Information about a model layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    pub name: String,
    pub layer_type: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub parameters_count: u64,
    pub memory_usage_bytes: u64,
    pub activation_function: Option<String>,
    pub is_trainable: bool,
}

/// Quantization information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationInfo {
    pub is_quantized: bool,
    pub quantization_bits: u8,
    pub quantization_method: String,
    pub compression_ratio: f64,
    pub accuracy_loss: Option<f64>,
}

/// Performance profiling data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingData {
    pub session_id: String,
    pub total_duration_ms: f64,
    pub layer_timings: Vec<LayerTiming>,
    pub memory_snapshots: Vec<MemorySnapshot>,
    pub tensor_operations: Vec<TensorOperation>,
    pub bottlenecks: Vec<PerformanceBottleneck>,
}

/// Timing information for individual layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerTiming {
    pub layer_name: String,
    pub forward_pass_ms: f64,
    pub memory_allocation_ms: f64,
    pub compute_utilization: f64,
    pub cache_hit_rate: f64,
}

/// Memory usage snapshot at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub timestamp_ms: f64,
    pub total_allocated_bytes: u64,
    pub peak_allocated_bytes: u64,
    pub gpu_memory_bytes: Option<u64>,
    pub tensor_count: usize,
    pub fragmentation_ratio: f64,
}

/// Information about tensor operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorOperation {
    pub operation_type: String,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shape: Vec<usize>,
    pub duration_ms: f64,
    pub memory_delta_bytes: i64,
    pub flops: Option<u64>,
}

/// Performance bottleneck identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    pub location: String,
    pub bottleneck_type: String,
    pub severity: f64, // 0.0 to 1.0
    pub description: String,
    pub suggested_fix: String,
}

/// Visualization data for model architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVisualization {
    pub nodes: Vec<VisualizationNode>,
    pub edges: Vec<VisualizationEdge>,
    pub layout_hints: HashMap<String, String>,
}

/// Node in the model visualization graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationNode {
    pub id: String,
    pub label: String,
    pub node_type: String,
    pub position: Option<(f64, f64)>,
    pub size: (f64, f64),
    pub color: String,
    pub metadata: HashMap<String, String>,
}

/// Edge in the model visualization graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationEdge {
    pub from: String,
    pub to: String,
    pub label: Option<String>,
    pub edge_type: String,
    pub weight: f64,
    pub color: String,
}

/// Debug session for tracking model behavior
pub struct DebugSession {
    pub session_id: String,
    pub start_time: Instant,
    pub profiling_data: Arc<Mutex<ProfilingData>>,
    pub memory_tracker: Arc<MemorySafetyVerifier>,
    pub is_active: bool,
}
