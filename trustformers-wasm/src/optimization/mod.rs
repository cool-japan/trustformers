//! Optimization modules for TrustformeRS WASM
//!
//! This module provides optimization features including quantization,
//! weight compression, batch processing, and SIMD-accelerated tensor operations.

pub mod batch_processing;
pub mod memory_pool;
pub mod quantization;
pub mod simd_tensor_ops;
pub mod weight_compression;

// Re-export main types
pub use batch_processing::{
    BatchConfig, BatchProcessor, BatchResponse, BatchingStrategy, Priority,
};
pub use memory_pool::{MemoryPool, MemoryPoolConfig, MemoryPoolStats};
pub use quantization::{
    QuantizationConfig, QuantizationPrecision, QuantizationStrategy, QuantizedModelData,
    WebQuantizer,
};
pub use simd_tensor_ops::SimdTensorOps;
pub use weight_compression::{
    CompressedModelData, CompressionConfig, CompressionLevel, CompressionStrategy, SparsityPattern,
    WeightCompressor,
};
