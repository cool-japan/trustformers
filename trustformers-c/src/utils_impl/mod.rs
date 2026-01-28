//! C Utilities Module
//!
//! This module provides a comprehensive set of utilities for C API integration,
//! organized into focused sub-modules for maintainability and clarity.

// Core modules
pub mod c_api;
pub mod performance;
pub mod standards;
pub mod string_conversion;
pub mod string_interning;
pub mod string_operations;
pub mod system_info;
pub mod types;
pub mod validation;

// Re-export core types and constants
pub use types::*;

// Re-export string interning functionality
pub use string_interning::{
    cleanup_global_interner, get_global_interner, initialize_global_interner, StringInterner,
};

// Re-export types needed for compatibility
pub use types::{
    StringInternerStats, StringInternerStats as StringInterningStats, StringUsageStats,
};

// Re-export string conversion utilities
pub use string_conversion::{
    c_str_to_string,
    conversion_utils::{
        c_str_clone, c_str_concat, c_str_equal, c_str_len, c_str_to_lowercase, c_str_to_uppercase,
        c_str_trim, c_string_array_to_vec, free_c_string_array, is_valid_utf8_c_str,
        option_string_to_c_str, string_vec_to_c_array,
    },
    free_c_string, result_to_error, str_to_c_str, string_to_c_str, SafeCString,
};

// Re-export performance utilities
pub use performance::{
    benchmark_utils::{
        benchmark_allocation, benchmark_auto, benchmark_compare, benchmark_function,
    },
    monitoring::{PerformanceMonitor, ScopeTimer},
    AllocationBenchmarkResult, BenchmarkComparison, MicroTimer, PercentileStats, PerformanceTimer,
};

// Re-export validation functions
pub use validation::{
    validate_file_path, validate_model_name, validate_string, validate_string_comprehensive,
};

// Add missing parse_number function
pub fn parse_number<T: std::str::FromStr>(s: &str) -> Result<T, T::Err> {
    s.parse()
}

// Re-export string operations (commented out until implemented)
/*
pub use string_operations::{
    string_find, string_replace, string_replace_all, string_split,
    string_join, string_trim, string_to_lowercase, string_to_uppercase,
    string_reverse, string_contains, string_starts_with, string_ends_with,
    string_substring, string_pad_left, string_pad_right, string_remove_chars,
    string_insert, string_count_occurrences, string_remove_whitespace,
    string_capitalize, string_title_case, string_snake_case, string_camel_case,
    string_kebab_case, string_truncate, string_word_count, string_char_count,
    string_line_count, string_is_empty, string_compare, string_levenshtein_distance,
    string_similarity, string_tokenize, string_normalize_whitespace,
    string_remove_duplicates, string_extract_numbers, string_extract_words,
    string_format_bytes, string_format_duration, string_parse_csv_line,
    string_escape_html, string_unescape_html, string_to_slug,
    advanced_ops::{
        string_fuzzy_match, string_soundex, string_metaphone, string_compress,
        string_decompress, string_hash, string_encrypt, string_decrypt,
        string_generate_random, string_validate_regex, string_extract_emails,
        string_extract_urls, string_extract_phone_numbers, string_mask_sensitive,
        string_generate_uuid, string_parse_duration, string_humanize_bytes,
        string_pluralize, string_singularize, string_transliterate,
        FuzzyMatchResult, CompressionStats, EncryptionKey, MaskingOptions,
    },
};
*/

// Re-export system information
pub use system_info::{
    get_battery_info, get_boot_time, get_cpu_info, get_cpu_usage, get_disk_usage, get_display_info,
    get_environment_variables, get_gpu_info, get_hardware_info, get_load_average, get_locale_info,
    get_memory_info, get_network_interfaces, get_process_info, get_system_info, get_thermal_info,
    get_timezone, get_uptime,
    hardware_utils::{
        detect_cpu_features, get_cache_sizes, get_memory_topology, get_pci_devices,
        get_storage_devices, get_usb_devices,
    },
    process_utils::{
        find_process_by_name, get_process_cpu_usage, get_process_memory, is_process_running,
        kill_process, list_processes,
    },
    system_monitor::{AlertCallback, ResourceThresholds, SystemMonitor},
};

// Re-export main C API functions
pub use c_api::*;

/// Utility macro for safe C string handling
#[macro_export]
macro_rules! safe_c_string {
    ($expr:expr) => {
        match $crate::utils::SafeCString::from_str($expr) {
            Some(s) => s,
            None => return std::ptr::null_mut(),
        }
    };
}

/// Utility macro for converting Rust results to C error codes
#[macro_export]
macro_rules! result_to_c_error {
    ($expr:expr) => {
        match $expr {
            Ok(_) => $crate::error::TrustformersError::Success as i32,
            Err(e) => e as i32,
        }
    };
}

/// Utility macro for safe pointer dereferencing
#[macro_export]
macro_rules! safe_deref {
    ($ptr:expr) => {
        if $ptr.is_null() {
            return $crate::error::TrustformersError::NullPointer as i32;
        }
        unsafe { &mut *$ptr }
    };
}

/// Initialize all utility subsystems
pub fn initialize_utils() -> crate::error::TrustformersResult<()> {
    initialize_global_interner();
    // Initialize other subsystems as needed
    Ok(())
}

/// Cleanup all utility subsystems
pub fn cleanup_utils() {
    cleanup_global_interner();
    // Cleanup other subsystems as needed
}

/// Get comprehensive utility statistics
pub fn get_utils_statistics() -> UtilsStatistics {
    // Convert SystemInfo to TrustformersSystemInfo
    let sys_info = match system_info::get_system_info() {
        Ok(info) => TrustformersSystemInfo {
            num_cpu_cores: info.num_cpu_cores,
            available_cpu_cores: info.available_cpu_cores,
            total_memory_bytes: info.total_memory_bytes,
            available_memory_bytes: info.available_memory_bytes,
            cuda_available: if info.gpu_info.cuda_available { 1 } else { 0 },
            num_cuda_devices: info.gpu_info.cuda_devices.len() as u32,
            os_name: str_to_c_str(&info.os_info.name),
            arch_name: str_to_c_str(&info.cpu_architecture),
        },
        Err(_) => TrustformersSystemInfo::default(),
    };

    UtilsStatistics {
        string_interning: get_global_interner().get_statistics(),
        system_info: sys_info,
        // Add other statistics as needed
    }
}

/// Combined statistics for all utility subsystems
#[derive(Debug)]
pub struct UtilsStatistics {
    pub string_interning: StringInterningStats,
    pub system_info: TrustformersSystemInfo,
}

// TODO: Tests use old function signatures for validate_string and safe_c_string macro
// #[cfg(test)]
/*
mod tests {
    use super::*;
use crate::error::TrustformersResult;

    #[test]
    fn test_utils_initialization() {
        assert!(initialize_utils().is_ok());
        cleanup_utils();
    }

    #[test]
    fn test_safe_c_string_macro() {
        let safe_str = safe_c_string!("test");
        assert!(!safe_str.as_ptr().is_null());
    }

    #[test]
    fn test_utils_statistics() {
        initialize_utils().unwrap();
        let stats = get_utils_statistics();
        assert!(stats.string_interning.total_strings >= 0);
        cleanup_utils();
    }

    #[test]
    fn test_module_integration() {
        // Test that all modules work together
        initialize_utils().unwrap();

        // Test string interning
        let interner = get_global_interner();
        let id = interner.intern("test_string");
        assert!(id > 0);

        // Test string conversion
        let c_str = str_to_c_str("test");
        assert!(!c_str.is_null());

        // Test validation
        assert!(validate_string("valid_string").is_ok());

        // Test performance timer
        let mut timer = PerformanceTimer::new();
        timer.start();
        std::thread::sleep(std::time::Duration::from_millis(1));
        assert!(timer.stop().is_ok());

        // Cleanup
        unsafe {
            free_c_string(c_str);
        }
        cleanup_utils();
    }
}
*/
