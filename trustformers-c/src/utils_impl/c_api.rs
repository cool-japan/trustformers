//! C API Export Functions
//!
//! This module provides the main C API exports that combine functionality from other
//! utility modules to provide a comprehensive C interface.

use super::parse_number;
use super::performance::*;
use super::string_conversion::*;
use super::string_interning::*;
use super::string_operations::*;
use super::system_info::*;
use super::types::*;
use super::validation::*;
use crate::error::{TrustformersError, TrustformersResult};
use std::os::raw::c_char;

/// Simple string validation helper function
fn validate_string_simple(
    s: &str,
    min_length: usize,
    max_length: usize,
    allow_special_chars: bool,
) -> Result<(), TrustformersError> {
    if s.len() < min_length {
        return Err(TrustformersError::ValidationError);
    }
    if s.len() > max_length {
        return Err(TrustformersError::ValidationError);
    }
    if !allow_special_chars
        && s.chars().any(|c| !c.is_alphanumeric() && c != ' ' && c != '_' && c != '-')
    {
        return Err(TrustformersError::ValidationError);
    }
    Ok(())
}
use std::ptr;

/// Initialize the C utilities system
#[no_mangle]
pub extern "C" fn trustformers_utils_init() -> i32 {
    // Initialize common strings in global interner
    super::string_interning::common_strings::init_common_strings();
    TrustformersError::Success as i32
}

/// Clean up the C utilities system
#[no_mangle]
pub extern "C" fn trustformers_utils_cleanup() -> i32 {
    // Global string interner is cleaned up automatically on program exit
    // No explicit cleanup needed for lazy static
    TrustformersError::Success as i32
}

/// Convert SystemInfo to TrustformersSystemInfo
fn convert_system_info(info: super::system_info::SystemInfo) -> TrustformersSystemInfo {
    TrustformersSystemInfo {
        num_cpu_cores: info.num_cpu_cores,
        available_cpu_cores: info.available_cpu_cores,
        total_memory_bytes: info.total_memory_bytes,
        available_memory_bytes: info.available_memory_bytes,
        cuda_available: if info.gpu_info.cuda_available { 1 } else { 0 },
        num_cuda_devices: info.gpu_info.cuda_devices.len() as u32,
        os_name: str_to_c_str(&info.os_info.name),
        arch_name: str_to_c_str(&info.cpu_architecture),
    }
}

/// Get comprehensive system information
#[no_mangle]
pub extern "C" fn trustformers_get_system_info() -> *mut TrustformersSystemInfo {
    match super::system_info::get_system_info() {
        Ok(info) => Box::into_raw(Box::new(convert_system_info(info))),
        Err(_) => ptr::null_mut(),
    }
}

/// Free system information structure
#[no_mangle]
pub unsafe extern "C" fn trustformers_free_system_info(info: *mut TrustformersSystemInfo) {
    if !info.is_null() {
        let _ = Box::from_raw(info);
    }
}

/// Validate a C string with comprehensive security checks
#[no_mangle]
pub extern "C" fn trustformers_validate_string(
    input: *const c_char,
    min_length: usize,
    max_length: usize,
    allow_special_chars: bool,
) -> i32 {
    if input.is_null() {
        return TrustformersError::NullPointer as i32;
    }

    match c_str_to_string(input) {
        Ok(string) => {
            match validate_string_simple(&string, min_length, max_length, allow_special_chars) {
                Ok(_) => TrustformersError::Success as i32,
                Err(_) => TrustformersError::InvalidParameter as i32,
            }
        },
        Err(e) => e as i32,
    }
}

/// Validate a file path for security
#[no_mangle]
pub extern "C" fn trustformers_validate_file_path(path: *const c_char) -> i32 {
    if path.is_null() {
        return TrustformersError::NullPointer as i32;
    }

    match validate_file_path(path) {
        Ok(_) => TrustformersError::Success as i32,
        Err(e) => e as i32,
    }
}

/// Intern a string and return its ID
#[no_mangle]
pub extern "C" fn trustformers_intern_string(input: *const c_char) -> u32 {
    if input.is_null() {
        return 0; // Invalid ID
    }

    match c_str_to_string(input) {
        Ok(string) => get_global_interner().intern(&string),
        Err(_) => 0, // Invalid ID
    }
}

/// Get an interned string by ID
#[no_mangle]
pub extern "C" fn trustformers_get_interned_string(id: u32) -> *mut c_char {
    match get_global_interner().get(id) {
        Some(string) => string_to_c_str(string.to_string()),
        None => ptr::null_mut(),
    }
}

/// Create a performance timer
#[no_mangle]
pub extern "C" fn trustformers_create_timer() -> *mut PerformanceTimer {
    Box::into_raw(Box::new(PerformanceTimer::new()))
}

/// Start a performance timer
#[no_mangle]
pub unsafe extern "C" fn trustformers_timer_start(timer: *mut PerformanceTimer) -> i32 {
    if timer.is_null() {
        return TrustformersError::NullPointer as i32;
    }

    (*timer).start();
    TrustformersError::Success as i32
}

/// Stop a performance timer and get the elapsed time
#[no_mangle]
pub unsafe extern "C" fn trustformers_timer_stop(
    timer: *mut PerformanceTimer,
    elapsed_ms: *mut f64,
) -> i32 {
    if timer.is_null() || elapsed_ms.is_null() {
        return TrustformersError::NullPointer as i32;
    }

    match (*timer).stop() {
        Ok(time) => {
            *elapsed_ms = time;
            TrustformersError::Success as i32
        },
        Err(e) => e as i32,
    }
}

/// Get performance timer statistics
#[no_mangle]
pub unsafe extern "C" fn trustformers_timer_get_stats(
    timer: *mut PerformanceTimer,
) -> *mut TrustformersBenchmarkResult {
    if timer.is_null() {
        return ptr::null_mut();
    }

    let stats = (*timer).get_statistics();
    Box::into_raw(Box::new(stats))
}

/// Free a performance timer
#[no_mangle]
pub unsafe extern "C" fn trustformers_free_timer(timer: *mut PerformanceTimer) {
    if !timer.is_null() {
        let _ = Box::from_raw(timer);
    }
}

/// Free benchmark result structure
#[no_mangle]
pub unsafe extern "C" fn trustformers_free_benchmark_result(
    result: *mut TrustformersBenchmarkResult,
) {
    if !result.is_null() {
        let _ = Box::from_raw(result);
    }
}

/// Concatenate two C strings
#[no_mangle]
pub extern "C" fn trustformers_string_concat(s1: *const c_char, s2: *const c_char) -> *mut c_char {
    match super::string_conversion::conversion_utils::c_str_concat(s1, s2) {
        Ok(result) => result,
        Err(_) => ptr::null_mut(),
    }
}

/// Trim whitespace from a C string
#[no_mangle]
pub extern "C" fn trustformers_string_trim(input: *const c_char) -> *mut c_char {
    match super::string_conversion::conversion_utils::c_str_trim(input) {
        Ok(result) => result,
        Err(_) => ptr::null_mut(),
    }
}

/// Convert C string to lowercase
#[no_mangle]
pub extern "C" fn trustformers_string_to_lowercase(input: *const c_char) -> *mut c_char {
    match super::string_conversion::conversion_utils::c_str_to_lowercase(input) {
        Ok(result) => result,
        Err(_) => ptr::null_mut(),
    }
}

/// Convert C string to uppercase
#[no_mangle]
pub extern "C" fn trustformers_string_to_uppercase(input: *const c_char) -> *mut c_char {
    match super::string_conversion::conversion_utils::c_str_to_uppercase(input) {
        Ok(result) => result,
        Err(_) => ptr::null_mut(),
    }
}

/// Check if two C strings are equal
#[no_mangle]
pub extern "C" fn trustformers_string_equal(s1: *const c_char, s2: *const c_char) -> i32 {
    match super::string_conversion::conversion_utils::c_str_equal(s1, s2) {
        Ok(true) => 1,
        Ok(false) => 0,
        Err(_) => -1,
    }
}

/// Get the length of a C string
#[no_mangle]
pub extern "C" fn trustformers_string_length(input: *const c_char) -> i32 {
    match super::string_conversion::conversion_utils::c_str_len(input) {
        Ok(len) => len as i32,
        Err(_) => -1,
    }
}

/// Parse a C string as an integer
#[no_mangle]
pub extern "C" fn trustformers_parse_int(input: *const c_char, result: *mut i64) -> i32 {
    if input.is_null() || result.is_null() {
        return TrustformersError::NullPointer as i32;
    }

    match c_str_to_string(input) {
        Ok(string) => match parse_number::<i64>(&string) {
            Ok(value) => {
                unsafe {
                    *result = value;
                }
                TrustformersError::Success as i32
            },
            Err(_) => TrustformersError::InvalidParameter as i32,
        },
        Err(e) => e as i32,
    }
}

/// Parse a C string as a float
#[no_mangle]
pub extern "C" fn trustformers_parse_float(input: *const c_char, result: *mut f64) -> i32 {
    if input.is_null() || result.is_null() {
        return TrustformersError::NullPointer as i32;
    }

    match c_str_to_string(input) {
        Ok(string) => match parse_number::<f64>(&string) {
            Ok(value) => {
                unsafe {
                    *result = value;
                }
                TrustformersError::Success as i32
            },
            Err(_) => TrustformersError::InvalidParameter as i32,
        },
        Err(e) => e as i32,
    }
}

/// Format a number as a C string
#[no_mangle]
pub extern "C" fn trustformers_format_number(value: f64, precision: u8) -> *mut c_char {
    let formatted = format!("{:.1$}", value, precision as usize);
    string_to_c_str(formatted)
}

/// Get CPU usage percentage
#[no_mangle]
pub extern "C" fn trustformers_get_cpu_usage() -> f64 {
    get_cpu_usage().unwrap_or(-1.0)
}

/// Get memory usage information
#[no_mangle]
pub extern "C" fn trustformers_get_memory_usage() -> *mut TrustformersMemoryInfo {
    match super::system_info::get_system_info() {
        Ok(sys_info) => {
            let memory_info = TrustformersMemoryInfo {
                total_memory_bytes: sys_info.total_memory_bytes,
                available_memory_bytes: sys_info.available_memory_bytes,
                used_memory_bytes: sys_info.total_memory_bytes - sys_info.available_memory_bytes,
                memory_usage_percent: ((sys_info.total_memory_bytes
                    - sys_info.available_memory_bytes)
                    as f64
                    / sys_info.total_memory_bytes as f64
                    * 100.0),
            };
            Box::into_raw(Box::new(memory_info))
        },
        Err(_) => ptr::null_mut(),
    }
}

/// Free memory usage information
#[no_mangle]
pub unsafe extern "C" fn trustformers_free_memory_info(info: *mut TrustformersMemoryInfo) {
    if !info.is_null() {
        let _ = Box::from_raw(info);
    }
}

/// Get disk usage for a path
#[no_mangle]
pub extern "C" fn trustformers_get_disk_usage(path: *const c_char) -> *mut TrustformersDiskInfo {
    if path.is_null() {
        return ptr::null_mut();
    }

    match c_str_to_string(path) {
        Ok(path_str) => match get_disk_usage(&path_str) {
            Ok(usage_percent) => {
                let disk_info = TrustformersDiskInfo {
                    total_space_bytes: 1000000000000, // 1TB placeholder
                    available_space_bytes: (1000000000000_u64 as f64 * (100.0 - usage_percent)
                        / 100.0) as u64,
                    used_space_bytes: (1000000000000_u64 as f64 * usage_percent / 100.0) as u64,
                    usage_percent,
                };
                Box::into_raw(Box::new(disk_info))
            },
            Err(_) => ptr::null_mut(),
        },
        Err(_) => ptr::null_mut(),
    }
}

/// Free disk usage information
#[no_mangle]
pub unsafe extern "C" fn trustformers_free_disk_info(info: *mut TrustformersDiskInfo) {
    if !info.is_null() {
        let _ = Box::from_raw(info);
    }
}

/// Benchmark a function pointer
#[no_mangle]
pub extern "C" fn trustformers_benchmark_function(
    func: extern "C" fn(),
    iterations: usize,
    warmup_iterations: usize,
) -> *mut TrustformersBenchmarkResult {
    let result = super::performance::benchmark_utils::benchmark_function(
        || func(),
        iterations,
        warmup_iterations,
    );
    Box::into_raw(Box::new(result))
}

/// Get string interning statistics
#[no_mangle]
pub extern "C" fn trustformers_get_interning_stats() -> *mut StringInterningStats {
    let stats = get_global_interner().get_statistics();
    Box::into_raw(Box::new(stats))
}

/// Free string interning statistics
#[no_mangle]
pub unsafe extern "C" fn trustformers_free_interning_stats(stats: *mut StringInterningStats) {
    if !stats.is_null() {
        let _ = Box::from_raw(stats);
    }
}

/// Free a C string allocated by this library
#[no_mangle]
pub unsafe extern "C" fn trustformers_free_string(ptr: *mut c_char) {
    free_c_string(ptr);
}

/// Get library version information
#[no_mangle]
pub extern "C" fn trustformers_get_version() -> *mut c_char {
    string_to_c_str("1.0.0".to_string())
}

/// Get build information
#[no_mangle]
pub extern "C" fn trustformers_get_build_info() -> *mut c_char {
    let rustc_version = std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string());
    let pkg_version = std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "unknown".to_string());
    let build_info = format!(
        "TrustformeRS C Utils - Built: {} - Rust: {}",
        rustc_version, pkg_version
    );
    string_to_c_str(build_info)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_c_api_initialization() {
        assert_eq!(trustformers_utils_init(), TrustformersError::Success as i32);
        assert_eq!(
            trustformers_utils_cleanup(),
            TrustformersError::Success as i32
        );
    }

    #[test]
    fn test_string_operations_c_api() {
        let test_str = CString::new("Hello World").unwrap();
        let result = trustformers_string_to_uppercase(test_str.as_ptr());
        assert!(!result.is_null());

        let back_to_rust = unsafe { c_str_to_string(result) }.unwrap();
        assert_eq!(back_to_rust, "HELLO WORLD");

        unsafe {
            trustformers_free_string(result);
        }
    }

    #[test]
    fn test_validation_c_api() {
        let valid_str = CString::new("ValidString123").unwrap();
        let result = trustformers_validate_string(valid_str.as_ptr(), 5, 50, true);
        assert_eq!(result, TrustformersError::Success as i32);

        let invalid_str = CString::new("").unwrap();
        let result = trustformers_validate_string(invalid_str.as_ptr(), 5, 50, false);
        assert_ne!(result, TrustformersError::Success as i32);
    }

    #[test]
    fn test_performance_timer_c_api() {
        let timer = trustformers_create_timer();
        assert!(!timer.is_null());

        unsafe {
            assert_eq!(
                trustformers_timer_start(timer),
                TrustformersError::Success as i32
            );

            // Simulate some work
            std::thread::sleep(std::time::Duration::from_millis(1));

            let mut elapsed = 0.0;
            assert_eq!(
                trustformers_timer_stop(timer, &mut elapsed),
                TrustformersError::Success as i32
            );
            assert!(elapsed > 0.0);

            trustformers_free_timer(timer);
        }
    }
}
