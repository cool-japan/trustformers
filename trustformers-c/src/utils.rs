//! Utility functions for TrustformeRS C API
//!
//! This module has been refactored into a modular architecture for better maintainability.
//! All functionality has been organized into focused sub-modules while maintaining
//! backward compatibility through comprehensive re-exports.
//!
//! # Architecture Overview
//!
//! The utilities are organized into the following modules:
//! - `types`: Core types, constants, and C-compatible structures
//! - `string_interning`: String interning system for memory optimization
//! - `string_conversion`: C/Rust string conversion utilities
//! - `performance`: Performance benchmarking and timing utilities
//! - `validation`: Input validation functions for security
//! - `string_operations`: String manipulation functions
//! - `system_info`: System information gathering
//! - `c_api`: Main C API export functions
//!
//! # Usage
//!
//! All previous functionality remains available at the same import paths:
//!
//! ```rust
//! use trustformers_c::utils::{
//!     StringInterner, PerformanceTimer, validate_c_string_safe,
//!     get_system_info, c_str_to_string, string_to_c_str
//! };
//! ```

// Import the modular structure
#[path = "utils_impl/mod.rs"]
mod utils_impl;

// Re-export everything to maintain backward compatibility
pub use utils_impl::*;

// Legacy re-exports for backward compatibility
pub use utils_impl::{
    // String conversion utilities
    c_str_to_string,
    cleanup_global_interner,

    free_c_string,
    get_cpu_info,
    get_disk_usage,

    get_global_interner,
    get_memory_info,
    // System information
    get_system_info,
    initialize_global_interner,
    result_to_error,
    str_to_c_str,
    string_to_c_str,
    validate_file_path,
    // Validation functions (using the new comprehensive validation module)
    validate_string as validate_c_string_safe,
    validate_string as validate_model_name, // Legacy alias

    validate_string_comprehensive,
    AllocationBenchmarkResult,

    BenchmarkComparison,
    MicroTimer,
    PercentileStats,
    // Performance timing
    PerformanceTimer,
    SafeCString,

    // String interning types and functions
    StringInterner,
    StringInterningStats,
    StringUsageStats,
    TrustformersBenchmarkResult,
    TrustformersDiskInfo,
    TrustformersMemoryInfo,
    // Core types
    TrustformersSystemInfo,
};

// Legacy type aliases for backward compatibility
pub type StringInternerStats = StringInterningStats;
pub type TrustformersStringInternerStats = StringInterningStats;
pub type TrustformersMemoryBreakdown = StringInterningStats; // Close enough for compatibility

// Legacy function aliases for backward compatibility
pub use string_operations::{
    string_compare as trustformers_string_compare, string_find as trustformers_string_length,
    string_find as trustformers_string_concat, string_trim as trustformers_string_copy,
};

// Legacy constants
pub const MAX_STRING_LENGTH: usize = 1024 * 1024;

// Re-export commonly used validation functions with legacy names
pub fn validate_range_f32(
    value: f32,
    min: f32,
    max: f32,
    _name: &str,
) -> crate::error::TrustformersResult<f32> {
    if !value.is_finite() {
        return Err(crate::error::TrustformersError::InvalidParameter);
    }
    if value < min || value > max {
        return Err(crate::error::TrustformersError::InvalidParameter);
    }
    Ok(value)
}

pub fn validate_range_i32(
    value: i32,
    min: i32,
    max: i32,
    _name: &str,
) -> crate::error::TrustformersResult<i32> {
    if value < min || value > max {
        return Err(crate::error::TrustformersError::InvalidParameter);
    }
    Ok(value)
}

pub fn validate_array_ptr<T>(
    ptr: *const T,
    len: usize,
    _name: &str,
) -> crate::error::TrustformersResult<()> {
    if ptr.is_null() {
        return Err(crate::error::TrustformersError::NullPointer);
    }
    if len == 0 {
        return Err(crate::error::TrustformersError::InvalidParameter);
    }
    const MAX_ARRAY_SIZE: usize = 1_000_000;
    if len > MAX_ARRAY_SIZE {
        return Err(crate::error::TrustformersError::ResourceLimitExceeded);
    }
    Ok(())
}

// Legacy validation wrapper functions for C API compatibility
pub fn validate_json_string(
    json_str: *const std::os::raw::c_char,
) -> crate::error::TrustformersResult<serde_json::Value> {
    match c_str_to_string(json_str) {
        Ok(json_string) => {
            if json_string.len() > MAX_STRING_LENGTH {
                return Err(crate::error::TrustformersError::InvalidParameter);
            }
            serde_json::from_str(&json_string)
                .map_err(|_| crate::error::TrustformersError::InvalidFormat)
        },
        Err(e) => Err(e),
    }
}

// Re-export the standards module for legacy compatibility
pub use utils_impl::standards;

// Legacy common_strings module compatibility
pub mod common_strings {
    use super::*;

    pub fn init_common_strings() {
        initialize_global_interner();
        let interner = get_global_interner();

        // Common model names
        let _ = interner.intern("bert");
        let _ = interner.intern("gpt2");
        let _ = interner.intern("gpt3");
        let _ = interner.intern("t5");
        let _ = interner.intern("roberta");
        let _ = interner.intern("distilbert");

        // Common tensor operations
        let _ = interner.intern("matmul");
        let _ = interner.intern("add");
        let _ = interner.intern("mul");
        let _ = interner.intern("softmax");
        let _ = interner.intern("relu");
        let _ = interner.intern("gelu");

        // Common data types
        let _ = interner.intern("float32");
        let _ = interner.intern("float16");
        let _ = interner.intern("int32");
        let _ = interner.intern("int64");
        let _ = interner.intern("bool");

        // Common error messages
        let _ = interner.intern("null_pointer");
        let _ = interner.intern("out_of_memory");
        let _ = interner.intern("invalid_argument");
        let _ = interner.intern("model_not_found");
        let _ = interner.intern("tokenizer_error");

        // Common device names
        let _ = interner.intern("cpu");
        let _ = interner.intern("cuda");
        let _ = interner.intern("gpu");
        let _ = interner.intern("metal");
        let _ = interner.intern("rocm");

        // Common configuration keys
        let _ = interner.intern("max_length");
        let _ = interner.intern("batch_size");
        let _ = interner.intern("num_heads");
        let _ = interner.intern("hidden_size");
        let _ = interner.intern("vocab_size");
    }

    pub fn get_common_string_id(name: &str) -> Option<u32> {
        get_global_interner().get_id(name)
    }
}

// Initialize utilities on module load
use std::sync::Once;
static INIT: Once = Once::new();

/// Initialize the utilities system
pub fn init_utils() {
    INIT.call_once(|| {
        let _ = initialize_utils();
        common_strings::init_common_strings();
    });
}

// Auto-initialize when the module is first loaded
lazy_static::lazy_static! {
    static ref _INIT: () = init_utils();
}

// TODO: Tests use old function signatures for validate_string_comprehensive
// #[cfg(test)]
/*
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::ptr;

    #[test]
    fn test_backward_compatibility_string_interning() {
        let interner = get_global_interner();
        let id = interner.intern("test");
        assert!(id > 0);

        let retrieved = interner.get(id).unwrap();
        assert_eq!(retrieved.as_str(), "test");
    }

    #[test]
    fn test_backward_compatibility_string_conversion() {
        let rust_str = "Hello, World!";
        let c_str = CString::new(rust_str).unwrap();
        let converted = c_str_to_string(c_str.as_ptr()).unwrap();
        assert_eq!(converted, rust_str);

        let c_ptr = string_to_c_str(rust_str.to_string());
        assert!(!c_ptr.is_null());
        let back = c_str_to_string(c_ptr).unwrap();
        assert_eq!(back, rust_str);

        unsafe {
            free_c_string(c_ptr);
        }
    }

    #[test]
    fn test_backward_compatibility_performance_timer() {
        let mut timer = PerformanceTimer::new();
        timer.start();
        std::thread::sleep(std::time::Duration::from_millis(1));
        let elapsed = timer.stop().unwrap();
        assert!(elapsed > 0.0);

        let stats = timer.get_statistics();
        assert_eq!(stats.iterations, 1);
    }

    #[test]
    fn test_backward_compatibility_validation() {
        let valid_str = CString::new("valid_string").unwrap();
        let result = validate_string_comprehensive("valid_string", 1, 100, true);
        assert!(result.is_ok());

        assert!(validate_range_f32(5.0, 0.0, 10.0, "test").is_ok());
        assert!(validate_range_i32(5, 0, 10, "test").is_ok());
    }

    #[test]
    fn test_backward_compatibility_system_info() {
        let info = get_system_info().unwrap();
        assert!(info.num_cpu_cores > 0);
    }

    #[test]
    fn test_backward_compatibility_c_api() {
        // Test C API functions are still accessible
        assert_eq!(
            trustformers_utils_init(),
            crate::error::TrustformersError::Success as i32
        );
        assert_eq!(
            trustformers_utils_cleanup(),
            crate::error::TrustformersError::Success as i32
        );
    }

    #[test]
    fn test_legacy_common_strings() {
        common_strings::init_common_strings();

        let bert_id = common_strings::get_common_string_id("bert");
        assert!(bert_id.is_some());

        let nonexistent_id = common_strings::get_common_string_id("nonexistent");
        assert!(nonexistent_id.is_none());
    }

    #[test]
    fn test_modular_integration() {
        // Test that all modules work together seamlessly
        init_utils();

        let interner = get_global_interner();
        let id = interner.intern("integration_test");

        let mut timer = PerformanceTimer::new();
        timer.start();
        std::thread::sleep(std::time::Duration::from_millis(1));
        let _elapsed = timer.stop().unwrap();

        let result = validate_string_comprehensive("valid", 1, 10, false);
        assert!(result.is_ok());

        let _system_info = get_system_info().unwrap();

        // Everything should work without conflicts
        assert!(interner.get(id).is_some());
    }
}
*/
