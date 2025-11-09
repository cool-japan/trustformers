/// Weight Loading Infrastructure
///
/// This module provides comprehensive weight loading support for various formats:
/// - HuggingFace format (PyTorch .bin files and SafeTensors)
/// - Lazy loading for large models
/// - Memory-mapped weight files
/// - Streaming weight loading
/// - Distributed weight loading across multiple nodes
/// - GGUF quantized format support
/// - Automatic format detection
/// - Load balancing and fault tolerance
pub mod config;
pub mod distributed;
pub mod gguf;
pub mod huggingface;
pub mod memory_mapped;
pub mod streaming;
pub mod utils;

// Re-export common types and traits
pub use config::{
    CacheEvictionPolicy, CacheStrategy, ConsistencyLevel, DistributedCacheConfig,
    DistributedConfig, FaultToleranceConfig, LoadBalancingStrategy, NetworkConfig, NodeConfig,
    QuantizationConfig, WeightDataType, WeightFormat, WeightLoadingConfig,
};

pub use huggingface::{
    HuggingFaceIndex, HuggingFaceLoader, HuggingFaceMetadata, LazyTensor, SafeTensorsHeader,
    TensorInfo, TensorMetadata, WeightLoader,
};

pub use memory_mapped::MemoryMappedLoader;

pub use streaming::StreamingLoader;

pub use distributed::{DistributedStats, DistributedWeightLoader};

pub use gguf::{GGMLType, GGUFHeader, GGUFLoader, GGUFTensorInfo};

pub use utils::{
    auto_create_loader, create_distributed_loader, create_gguf_loader, create_huggingface_loader,
    create_memory_mapped_loader,
};

/// Default weight loading configuration
pub fn default_config() -> WeightLoadingConfig {
    WeightLoadingConfig::default()
}

/// Create a weight loading configuration with streaming enabled
pub fn streaming_config() -> WeightLoadingConfig {
    WeightLoadingConfig {
        streaming: true,
        ..Default::default()
    }
}

/// Create a weight loading configuration with memory mapping enabled
pub fn memory_mapped_config() -> WeightLoadingConfig {
    WeightLoadingConfig {
        memory_mapped: true,
        ..Default::default()
    }
}

/// Create a weight loading configuration with lazy loading enabled
pub fn lazy_loading_config() -> WeightLoadingConfig {
    WeightLoadingConfig {
        lazy_loading: true,
        ..Default::default()
    }
}
