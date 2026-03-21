//! Model caching and management for efficient serving
//!
//! This module provides advanced model caching, hot-swapping, and memory management
//! capabilities for production model serving with TrustformeRS.
//!
//! The module is organized into several sub-modules for better maintainability:
//! - `types`: Type definitions for cache configuration, entries, and statistics
//! - `manager`: Main model cache manager implementation with lock-free data structures
//! - `c_api`: C Foreign Function Interface exports

use once_cell::sync::Lazy;
use std::sync::RwLock;

// Sub-modules
pub mod c_api;
pub mod manager;
pub mod types;

// Re-export public types and functions
pub use manager::ModelCacheManager;
pub use types::*;

// C API functions are automatically exported via #[no_mangle]
pub use c_api::*;

/// Global model cache manager
static MODEL_CACHE_MANAGER: Lazy<RwLock<ModelCacheManager>> =
    Lazy::new(|| RwLock::new(ModelCacheManager::new()));
