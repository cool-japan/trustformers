//! Performance profiler type definitions

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Types of ML operations that can be profiled
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationType {
    /// Model loading operations
    ModelLoading,
    /// Tokenization operations
    Tokenization,
    /// Forward pass through transformer layers
    TransformerLayer,
    /// Attention computation
    Attention,
    /// Feed-forward network computation
    FeedForward,
    /// Embedding lookup
    Embedding,
    /// Layer normalization
    LayerNorm,
    /// Activation functions
    Activation,
    /// Matrix multiplication
    MatMul,
    /// Quantization/dequantization
    Quantization,
    /// Memory transfers
    MemoryTransfer,
    /// GPU kernel execution
    GPUKernel,
    /// Complete inference pass
    FullInference,
}

/// Resource types for monitoring
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceType {
    /// CPU utilization
    CPU,
    /// GPU utilization
    GPU,
    /// WASM memory usage
    WAMSMemory,
    /// GPU memory usage
    GPUMemory,
    /// Network bandwidth
    Network,
    /// Cache hit rates
    Cache,
    /// Battery level
    Battery,
    /// Power consumption
    Power,
    /// Thermal state
    Thermal,
    /// CPU temperature
    CPUTemperature,
    /// GPU temperature
    GPUTemperature,
}

/// Performance bottleneck types
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BottleneckType {
    /// CPU computation bound
    CPUCompute,
    /// GPU computation bound
    GPUCompute,
    /// Memory bandwidth bound
    MemoryBandwidth,
    /// Memory capacity bound
    MemoryCapacity,
    /// GPU memory bound
    GPUMemory,
    /// Data transfer bound
    DataTransfer,
    /// Serialization/deserialization bound
    Serialization,
    /// JavaScript interop bound
    JSInterop,
}

/// Optimization targets for adaptive optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationTarget {
    Latency,         // Minimize inference latency
    Throughput,      // Maximize throughput
    MemoryUsage,     // Minimize memory usage
    PowerEfficiency, // Minimize power consumption
    Accuracy,        // Maintain accuracy while optimizing
    Balanced,        // Balance all metrics
}

/// Optimization strategies that can be applied
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationStrategy {
    CPUPreferred,        // Use CPU for most operations
    GPUPreferred,        // Use GPU for most operations
    Hybrid,              // Dynamic CPU/GPU selection
    MemoryOptimized,     // Optimize for memory usage
    LatencyOptimized,    // Optimize for low latency
    ThroughputOptimized, // Optimize for high throughput
}

/// Direction of performance trend
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    Improving,
    Degrading,
    Stable,
    Volatile,
}

/// Severity of performance anomaly
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalySeverity {
    Info,
    Warning,
    Critical,
}
