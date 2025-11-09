/// Utility functions for weight loading
///
/// This module provides convenience functions for creating different types of weight loaders.
use std::path::Path;
use trustformers_core::errors::{ErrorKind, Result, TrustformersError};

use super::config::{DistributedConfig, WeightLoadingConfig};
use super::distributed::DistributedWeightLoader;
use super::gguf::GGUFLoader;
use super::huggingface::{HuggingFaceLoader, WeightLoader};
use super::memory_mapped::MemoryMappedLoader;

/// Create a HuggingFace weight loader
pub fn create_huggingface_loader(
    model_dir: impl AsRef<Path>,
    config: Option<WeightLoadingConfig>,
) -> Result<Box<dyn WeightLoader>> {
    let config = config.unwrap_or_default();
    let loader = HuggingFaceLoader::new(model_dir, config)?;
    Ok(Box::new(loader))
}

/// Create a memory-mapped weight loader
pub fn create_memory_mapped_loader(path: impl AsRef<Path>) -> Result<Box<dyn WeightLoader>> {
    let loader = MemoryMappedLoader::new(path)?;
    Ok(Box::new(loader))
}

/// Create a GGUF loader
pub fn create_gguf_loader(path: impl AsRef<Path>) -> Result<Box<dyn WeightLoader>> {
    let loader = GGUFLoader::new(path)?;
    Ok(Box::new(loader))
}

/// Create a distributed weight loader
pub fn create_distributed_loader(
    config: WeightLoadingConfig,
    distributed_config: DistributedConfig,
) -> Result<Box<dyn WeightLoader>> {
    let loader = DistributedWeightLoader::new(config, distributed_config)?;
    Ok(Box::new(loader))
}

/// Auto-detect format and create appropriate loader
pub fn auto_create_loader(
    path: impl AsRef<Path>,
    config: Option<WeightLoadingConfig>,
) -> Result<Box<dyn WeightLoader>> {
    let path = path.as_ref();
    let config = config.unwrap_or_default();

    // Check if distributed loading is configured
    if let Some(distributed_config) = config.distributed.clone() {
        return create_distributed_loader(config, distributed_config);
    }

    if path.is_dir() {
        // Directory - assume HuggingFace format
        create_huggingface_loader(path, Some(config))
    } else if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
        // Single SafeTensors file
        if config.memory_mapped {
            create_memory_mapped_loader(path)
        } else {
            // Create single-file HuggingFace loader
            create_huggingface_loader(path.parent().unwrap_or(path), Some(config))
        }
    } else if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
        // GGUF file
        create_gguf_loader(path)
    } else {
        Err(TrustformersError::new(ErrorKind::InvalidFormat {
            expected: "HuggingFace directory, .safetensors, or .gguf".to_string(),
            actual: "Unknown weight format".to_string(),
        }))
    }
}
