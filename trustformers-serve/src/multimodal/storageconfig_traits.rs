//! # StorageConfig - Trait Implementations
//!
//! This module contains trait implementations for `StorageConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::path::PathBuf;

use super::types::{StorageBackend, StorageConfig};

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            backend: StorageBackend::Local,
            local_path: PathBuf::from("storage/multimodal"),
            temp_path: PathBuf::from("tmp/multimodal"),
            retention_hours: 24,
            enable_compression: false,
            compression_level: 6,
            enable_encryption: false,
            encryption_key: None,
        }
    }
}
