//! Core Types and Structures for C API Utilities
//!
//! This module provides fundamental types, structures, and constants used
//! throughout the C API utilities for TrustformeRS.

use std::os::raw::{c_char, c_double, c_int};
use std::ptr;
use std::time::Instant;

/// Maximum string length for safety (1MB)
pub const MAX_STRING_LENGTH: usize = 1024 * 1024;

/// Maximum array size to prevent DoS attacks
pub const MAX_ARRAY_SIZE: usize = 1_000_000;

/// Default maximum string length for model names
pub const MAX_MODEL_NAME_LENGTH: usize = 256;

/// Default maximum file path length
pub const MAX_FILE_PATH_LENGTH: usize = 4096;

/// Maximum JSON string size (1MB)
pub const MAX_JSON_STRING_LENGTH: usize = 1024 * 1024;

/// String usage statistics for optimization
#[derive(Debug, Clone)]
pub struct StringUsageStats {
    /// Number of times this string was accessed
    pub access_count: u64,
    /// Last time this string was accessed
    pub last_access: Instant,
    /// Size of the string in bytes
    pub size_bytes: usize,
}

impl StringUsageStats {
    pub fn new(size_bytes: usize) -> Self {
        Self {
            access_count: 1,
            last_access: Instant::now(),
            size_bytes,
        }
    }

    pub fn update_access(&mut self) {
        self.access_count += 1;
        self.last_access = Instant::now();
    }
}

/// Statistics about the string interner
#[derive(Debug)]
pub struct StringInternerStats {
    pub total_strings: usize,
    pub total_memory_bytes: usize,
    pub total_accesses: u64,
    pub avg_accesses_per_string: f64,
    pub most_frequent_strings: Vec<(u32, u64)>, // (id, access_count)
}

// Type aliases for compatibility
pub type StringInterningStats = StringInternerStats;

/// Memory usage breakdown for string interner
#[derive(Debug)]
pub struct MemoryBreakdown {
    pub total_memory_bytes: usize,
    pub small_strings_count: usize,  // < 64 bytes
    pub medium_strings_count: usize, // 64-512 bytes
    pub large_strings_count: usize,  // > 512 bytes
}

/// C-compatible performance benchmarking result
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TrustformersBenchmarkResult {
    /// Number of iterations performed
    pub iterations: u64,
    /// Total time in milliseconds
    pub total_time_ms: c_double,
    /// Average time per iteration in milliseconds
    pub avg_time_ms: c_double,
    /// Minimum time in milliseconds
    pub min_time_ms: c_double,
    /// Maximum time in milliseconds
    pub max_time_ms: c_double,
    /// Standard deviation in milliseconds
    pub std_dev_ms: c_double,
    /// Throughput (operations per second)
    pub throughput_ops: c_double,
}

impl Default for TrustformersBenchmarkResult {
    fn default() -> Self {
        Self {
            iterations: 0,
            total_time_ms: 0.0,
            avg_time_ms: 0.0,
            min_time_ms: 0.0,
            max_time_ms: 0.0,
            std_dev_ms: 0.0,
            throughput_ops: 0.0,
        }
    }
}

/// C-compatible system information structure
#[repr(C)]
#[derive(Debug)]
pub struct TrustformersSystemInfo {
    /// Number of CPU cores
    pub num_cpu_cores: u32,
    /// Available CPU cores
    pub available_cpu_cores: u32,
    /// Total system memory in bytes
    pub total_memory_bytes: u64,
    /// Available memory in bytes
    pub available_memory_bytes: u64,
    /// Whether CUDA is available
    pub cuda_available: c_int,
    /// Number of CUDA devices
    pub num_cuda_devices: u32,
    /// Operating system name
    pub os_name: *mut c_char,
    /// Architecture name
    pub arch_name: *mut c_char,
}

impl Default for TrustformersSystemInfo {
    fn default() -> Self {
        Self {
            num_cpu_cores: 0,
            available_cpu_cores: 0,
            total_memory_bytes: 0,
            available_memory_bytes: 0,
            cuda_available: 0,
            num_cuda_devices: 0,
            os_name: ptr::null_mut(),
            arch_name: ptr::null_mut(),
        }
    }
}

/// C-compatible string interner statistics
#[repr(C)]
#[derive(Debug, Default)]
pub struct TrustformersStringInternerStats {
    pub total_strings: usize,
    pub total_memory_bytes: usize,
    pub total_accesses: u64,
    pub avg_accesses_per_string: c_double,
}

/// C-compatible memory breakdown
#[repr(C)]
#[derive(Debug, Default)]
pub struct TrustformersMemoryBreakdown {
    pub total_memory_bytes: usize,
    pub small_strings_count: usize,
    pub medium_strings_count: usize,
    pub large_strings_count: usize,
}

/// C-compatible memory information structure
#[repr(C)]
#[derive(Debug, Default)]
pub struct TrustformersMemoryInfo {
    /// Total memory in bytes
    pub total_memory_bytes: u64,
    /// Available memory in bytes
    pub available_memory_bytes: u64,
    /// Used memory in bytes
    pub used_memory_bytes: u64,
    /// Memory usage percentage
    pub memory_usage_percent: c_double,
}

/// C-compatible disk information structure
#[repr(C)]
#[derive(Debug, Default)]
pub struct TrustformersDiskInfo {
    /// Total disk space in bytes
    pub total_space_bytes: u64,
    /// Available disk space in bytes
    pub available_space_bytes: u64,
    /// Used disk space in bytes
    pub used_space_bytes: u64,
    /// Disk usage percentage
    pub usage_percent: c_double,
}

/// Memory size categories for string classification
pub mod memory_categories {
    /// Small strings threshold (64 bytes)
    pub const SMALL_STRING_THRESHOLD: usize = 64;
    /// Medium strings threshold (512 bytes)
    pub const MEDIUM_STRING_THRESHOLD: usize = 512;
    // Large strings are above medium threshold
}

/// Common validation patterns
pub mod validation_patterns {
    /// Suspicious file path patterns
    pub const SUSPICIOUS_PATH_PATTERNS: &[&str] = &["//", "\\\\", "<", ">", "|", ":", "*", "?"];

    /// Valid model name characters (alphanumeric plus limited special chars)
    pub fn is_valid_model_name_char(c: char) -> bool {
        c.is_alphanumeric() || "-_/.".contains(c)
    }

    /// Check for control characters that could cause issues
    pub fn is_safe_control_char(c: char) -> bool {
        !c.is_control() || c == '\n' || c == '\r' || c == '\t'
    }
}
