//! Operation types and fusion patterns for kernel fusion
//!
//! This module defines the fundamental types used in kernel fusion, including
//! operation types, fusion patterns, and constraints.

use serde::{Deserialize, Serialize};

/// Operation types that can be fused
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperationType {
    // Element-wise operations
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,

    // Activation functions
    ReLU,
    GELU,
    Sigmoid,
    Tanh,
    Swish,
    SwiGLU,
    Softmax,
    LayerNorm,
    GroupNorm,
    RMSNorm,

    // Position embedding operations
    RoPE,
    SinusoidalPositionEmbedding,
    LearnablePositionEmbedding,

    // Matrix operations
    MatMul,
    BatchMatMul,
    Transpose,

    // Reduction operations
    Sum,
    Mean,
    Max,
    Min,
    ArgMax,
    ArgMin,

    // Tensor manipulation
    Reshape,
    Slice,
    Concat,
    Split,
    Broadcast,

    // Memory operations
    Copy,
    Cast,

    // Custom operations
    Custom(String),
}

/// Fusion pattern that describes how operations can be combined
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FusionPattern {
    /// Sequence of element-wise operations
    ElementWiseChain(Vec<OperationType>),
    /// Matrix multiplication followed by bias add and activation
    LinearActivation {
        matmul: OperationType,
        bias_add: bool,
        activation: Option<OperationType>,
    },
    /// Batch normalization pattern
    BatchNorm {
        normalize: bool,
        scale: bool,
        shift: bool,
        activation: Option<OperationType>,
    },
    /// Attention computation pattern
    AttentionFusion {
        query_key_matmul: bool,
        softmax: bool,
        value_matmul: bool,
        dropout: bool,
    },
    /// Reduction followed by broadcast
    ReduceBroadcast {
        reduction: OperationType,
        broadcast: OperationType,
    },
    /// RoPE (Rotary Position Embedding) pattern
    RoPEFusion {
        apply_rope: bool,
        cos_sin_cached: bool,
        dimensions: usize,
    },
    /// SwiGLU activation pattern (Swish + GLU)
    SwiGLU {
        gate_projection: bool,
        up_projection: bool,
        swish_activation: bool,
        element_wise_multiply: bool,
    },
    /// Group normalization pattern
    GroupNorm {
        groups: usize,
        normalize: bool,
        scale: bool,
        shift: bool,
        activation: Option<OperationType>,
    },
    /// Flash attention with optimized memory access
    FlashAttentionOptimized {
        query_key_matmul: bool,
        scaled_softmax: bool,
        value_matmul: bool,
        causal_mask: bool,
        dropout: bool,
        block_size: usize,
    },
    /// Custom fusion pattern
    Custom {
        name: String,
        operations: Vec<OperationType>,
        constraints: Vec<FusionConstraint>,
    },
}

/// Constraints for fusion patterns
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FusionConstraint {
    /// Operations must have compatible shapes
    ShapeCompatible,
    /// Operations must use the same data type
    DataTypeCompatible,
    /// Operations must be on the same device
    DeviceCompatible,
    /// Maximum number of operations in fusion
    MaxOperations(usize),
    /// Memory usage constraint
    MaxMemoryUsage(usize),
    /// Operations must be contiguous in computation graph
    Contiguous,
    /// Memory layout must be optimized for cache efficiency
    CacheOptimized,
    /// Memory access pattern must be coalesced for GPU
    CoalescedAccess,
    /// Tensor strides must be aligned for vectorization
    StrideAligned(usize), // alignment in bytes
    /// Memory bandwidth constraint
    MaxMemoryBandwidth(f64), // GB/s
    /// Cache size constraint
    FitsInCache(usize), // cache size in bytes
    /// Prefer operations that maintain data locality
    DataLocality,
}
