//! Unit tests for the weight-loading configuration and metadata types.
//!
//! This file used to sit on disk without ever being declared in
//! `weight_loading/mod.rs`, so none of it was compiled or run. It is wired in
//! now: a test that never runs is indistinguishable from one that does not exist.

use crate::weight_loading::config::{
    CacheEvictionPolicy, CacheStrategy, ConsistencyLevel, DistributedCacheConfig,
    FaultToleranceConfig, LoadBalancingStrategy, NetworkConfig, NodeConfig, QuantizationConfig,
    WeightDataType, WeightLoadingConfig,
};
use crate::weight_loading::huggingface::{SafeTensorsHeader, TensorInfo, TensorMetadata};
use std::path::PathBuf;

#[test]
fn test_weight_loading_config_with_lazy_loading() {
    let config = WeightLoadingConfig {
        lazy_loading: true,
        ..Default::default()
    };
    assert!(config.lazy_loading);
}

#[test]
fn test_weight_loading_config_with_memory_mapped() {
    let config = WeightLoadingConfig {
        memory_mapped: true,
        ..Default::default()
    };
    assert!(config.memory_mapped);
}

#[test]
fn test_weight_loading_config_with_quantization() {
    let config = WeightLoadingConfig {
        quantization: Some(QuantizationConfig {
            bits: 4,
            group_size: Some(128),
            symmetric: true,
        }),
        ..Default::default()
    };
    let quant = config.quantization.as_ref().expect("expected quantization config");
    assert_eq!(quant.bits, 4);
    assert_eq!(quant.group_size, Some(128));
    assert!(quant.symmetric);
}

#[test]
fn test_weight_loading_config_with_cache_dir() {
    let cache_dir = std::env::temp_dir().join("model_cache");
    let config = WeightLoadingConfig {
        cache_dir: Some(cache_dir.clone()),
        ..Default::default()
    };
    assert_eq!(config.cache_dir, Some(cache_dir));
}

// --- WeightDataType Tests ---

#[test]
fn test_weight_data_type_variants() {
    let _ = WeightDataType::Float32;
    let _ = WeightDataType::Float16;
    let _ = WeightDataType::BFloat16;
    let _ = WeightDataType::Int8;
    let _ = WeightDataType::Int4;
}

// --- FaultToleranceConfig Tests ---

#[test]
fn test_fault_tolerance_config_default() {
    let config = FaultToleranceConfig::default();
    assert_eq!(config.max_retries, 3);
    assert!(config.enable_failover);
    assert!(config.backup_nodes.is_empty());
}

// --- NetworkConfig Tests ---

#[test]
fn test_network_config_default() {
    let config = NetworkConfig::default();
    assert_eq!(config.max_concurrent_connections, 10);
    assert!(config.enable_keepalive);
    assert_eq!(config.chunk_size, 8192);
}

// --- DistributedCacheConfig Tests ---

#[test]
fn test_distributed_cache_config_default() {
    let config = DistributedCacheConfig::default();
    assert_eq!(config.cache_strategy, CacheStrategy::ReadThrough);
    assert_eq!(config.replication_factor, 2);
    assert_eq!(config.eviction_policy, CacheEvictionPolicy::LRU);
    assert_eq!(config.consistency_level, ConsistencyLevel::Eventual);
}

// --- LoadBalancingStrategy Tests ---

#[test]
fn test_load_balancing_strategy_equality() {
    assert_eq!(
        LoadBalancingStrategy::RoundRobin,
        LoadBalancingStrategy::RoundRobin
    );
    assert_ne!(
        LoadBalancingStrategy::RoundRobin,
        LoadBalancingStrategy::LeastLoaded
    );
}

// --- TensorInfo Tests ---

#[test]
fn test_tensor_info_creation() {
    let info = TensorInfo {
        dtype: "F32".to_string(),
        shape: vec![768, 512],
        data_offsets: [0, 1572864],
    };
    assert_eq!(info.dtype, "F32");
    assert_eq!(info.shape, vec![768, 512]);
    assert_eq!(info.data_offsets[0], 0);
    assert_eq!(info.data_offsets[1], 1572864);
}

// --- TensorMetadata Tests ---

#[test]
fn test_tensor_metadata_creation() {
    let meta = TensorMetadata {
        shape: vec![768, 512],
        dtype: WeightDataType::Float32,
        size_bytes: 1572864,
        offset: 0,
    };
    assert_eq!(meta.shape, vec![768, 512]);
    assert_eq!(meta.size_bytes, 1572864);
    assert_eq!(meta.offset, 0);
}

// --- SafeTensorsHeader Tests ---

#[test]
fn test_safetensors_header_deserialization() {
    let json = r#"{"__metadata__": {"format": "pt"}, "model.weight": {"dtype": "F32", "shape": [768, 512], "data_offsets": [0, 1572864]}}"#;
    let header: Result<SafeTensorsHeader, _> = serde_json::from_str(json);
    assert!(header.is_ok());
    let header = header.expect("Failed to deserialize header");
    assert!(header.metadata.is_some());
    assert_eq!(header.tensors.len(), 1);
    assert!(header.tensors.contains_key("model.weight"));
}

#[test]
fn test_safetensors_header_no_metadata() {
    let json = r#"{"model.weight": {"dtype": "F32", "shape": [64], "data_offsets": [0, 256]}}"#;
    let header: Result<SafeTensorsHeader, _> = serde_json::from_str(json);
    assert!(header.is_ok());
    let header = header.expect("Failed to deserialize header");
    assert_eq!(header.tensors.len(), 1);
}

// --- NodeConfig Tests ---

#[test]
fn test_node_config_creation() {
    let node = NodeConfig {
        id: "node-1".to_string(),
        address: "192.168.1.1".to_string(),
        port: 8080,
        weight_capacity: 1024 * 1024 * 1024,
        bandwidth: 100.0,
        priority: 128,
        storage_paths: vec![PathBuf::from("/data/models")],
    };
    assert_eq!(node.id, "node-1");
    assert_eq!(node.port, 8080);
    assert_eq!(node.priority, 128);
}

// --- QuantizationConfig Tests ---

#[test]
fn test_quantization_config_4bit() {
    let config = QuantizationConfig {
        bits: 4,
        group_size: Some(128),
        symmetric: true,
    };
    assert_eq!(config.bits, 4);
    assert!(config.symmetric);
}

#[test]
fn test_quantization_config_8bit_asymmetric() {
    let config = QuantizationConfig {
        bits: 8,
        group_size: None,
        symmetric: false,
    };
    assert_eq!(config.bits, 8);
    assert!(config.group_size.is_none());
    assert!(!config.symmetric);
}

// --- Integration: SafeTensors file parsing ---

#[test]
fn test_safetensors_file_creation_and_parsing() {
    let json_content = r#"{"__metadata__": {"format": "pt"}, "encoder.weight": {"dtype": "F32", "shape": [128, 64], "data_offsets": [0, 32768]}, "decoder.weight": {"dtype": "F32", "shape": [64, 128], "data_offsets": [32768, 65536]}}"#;

    // A fixed name under `temp_dir()` would collide between concurrent test
    // processes; `write_temp_file` keys the name on the pid and a counter.
    let test_file = crate::weight_loading::test_support::write_temp_file(
        "header",
        "json",
        json_content.as_bytes(),
    );

    let content = std::fs::read_to_string(&test_file).expect("Failed to read test file");
    let header: SafeTensorsHeader = serde_json::from_str(&content).expect("Failed to parse header");

    assert_eq!(header.tensors.len(), 2);
    assert!(header.tensors.contains_key("encoder.weight"));
    assert!(header.tensors.contains_key("decoder.weight"));

    let encoder_info = &header.tensors["encoder.weight"];
    assert_eq!(encoder_info.shape, vec![128, 64]);

    let _ = std::fs::remove_file(&test_file);
}
