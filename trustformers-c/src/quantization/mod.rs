//! Quantization Support for TrustformeRS C API
//!
//! This module provides comprehensive quantization capabilities including INT8, INT4, dynamic quantization,
//! mixed-precision, quantization-aware training, and post-training quantization techniques.
//!
//! ## Refactoring Summary
//!
//! Previously this was a single 3,118-line file containing all quantization functionality.
//! It has been split into focused modules:
//!
//! - `types.rs` - Core type definitions and enums
//! - `config.rs` - Configuration structures and settings
//! - `engine.rs` - Main quantization engine implementation
//! - `utils.rs` - Utility functions and helpers
//! - `c_api.rs` - C API functions for external interfaces
//!
//! This refactoring improves:
//! - Code maintainability and readability
//! - Module compilation times
//! - Test isolation
//! - Code reuse through focused modules
//! - Developer experience when working on specific algorithms

pub mod advanced_techniques;
pub mod c_api;
pub mod config;
pub mod engine;
pub mod types;
pub mod utils;

// Re-export core types and functionality for convenience
pub use config::*;
pub use engine::*;
pub use types::*;
pub use utils::*;

// Re-export C API for external use
pub use c_api::{
    trustformers_quantization_calibrate, trustformers_quantization_create_engine,
    trustformers_quantization_destroy_engine, trustformers_quantization_export_model,
    trustformers_quantization_get_stats, trustformers_quantization_get_version,
    trustformers_quantization_quantize_model, trustformers_quantization_recommend_config,
    trustformers_quantization_set_calibration_data, trustformers_quantization_validate_config,
    QuantizationHandle, TrustformersQuantizationConfig, TrustformersQuantizationStats,
};

// Re-export advanced techniques for external use
pub use advanced_techniques::{
    trustformers_advanced_quantization_create, trustformers_advanced_quantization_report,
    trustformers_nas_quantization_optimize, AdvancedQuantizationConfig, AdvancedQuantizationEngine,
};
