//! Checkpoint format conversion and management
//!
//! This module provides utilities for converting model checkpoints between
//! different deep learning frameworks (PyTorch, TensorFlow, JAX).

pub mod converter;
pub mod formats;
pub mod mapping;
pub mod utils;

pub use converter::{
    CheckpointConverter, CheckpointConverterBuilder, ConversionConfig, ConversionResult,
};
pub use formats::{
    Checkpoint, CheckpointFormat, JaxCheckpoint, PyTorchCheckpoint, TensorFlowCheckpoint,
    TrustformersCheckpoint, WeightTensor,
};
pub use mapping::{LayerMapping, ModelType, WeightMapping, WeightMappingRule};
pub use utils::{
    compare_checkpoints, detect_format, get_checkpoint_info, load_checkpoint, merge_checkpoints,
    save_checkpoint, shard_checkpoint,
};

use anyhow::Result;
use std::path::Path;

/// Quick conversion function for common use cases
pub async fn convert_checkpoint(
    source_path: &Path,
    target_path: &Path,
    target_format: CheckpointFormat,
) -> Result<ConversionResult> {
    let converter = CheckpointConverter::new();
    converter.convert(source_path, target_path, target_format).await
}
