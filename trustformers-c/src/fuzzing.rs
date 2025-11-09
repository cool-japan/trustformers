//! Fuzzing utilities for security testing

use crate::error::{TrustformersError, TrustformersResult};
use scirs2_core::random::*; // SciRS2 Integration Policy (was: use rand::{thread_rng, Rng};)
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_uint};
use std::ptr;

/// Fuzzing configuration
#[derive(Debug, Clone)]
pub struct FuzzConfig {
    /// Maximum string length to generate
    pub max_string_length: usize,
    /// Maximum array size to generate
    pub max_array_size: usize,
    /// Probability of generating null pointers (0.0 to 1.0)
    pub null_probability: f64,
    /// Probability of generating invalid UTF-8 (0.0 to 1.0)
    pub invalid_utf8_probability: f64,
    /// Number of test iterations
    pub iterations: usize,
}

impl Default for FuzzConfig {
    fn default() -> Self {
        Self {
            max_string_length: 1024,
            max_array_size: 100,
            null_probability: 0.1,
            invalid_utf8_probability: 0.05,
            iterations: 1000,
        }
    }
}

/// Fuzzing test result
#[derive(Debug)]
pub struct FuzzResult {
    /// Number of tests passed
    pub passed: usize,
    /// Number of tests failed
    pub failed: usize,
    /// Number of crashes detected
    pub crashes: usize,
    /// Errors encountered during testing
    pub errors: Vec<String>,
}

impl FuzzResult {
    pub fn new() -> Self {
        Self {
            passed: 0,
            failed: 0,
            crashes: 0,
            errors: Vec::new(),
        }
    }

    pub fn total_tests(&self) -> usize {
        self.passed + self.failed + self.crashes
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_tests() == 0 {
            0.0
        } else {
            self.passed as f64 / self.total_tests() as f64
        }
    }
}

/// Generate random string for fuzzing
pub fn generate_fuzz_string(config: &FuzzConfig) -> Option<CString> {
    let mut rng = thread_rng();

    // Randomly return null
    if rng.gen::<f64>() < config.null_probability {
        return None;
    }

    let length = rng.gen_range(0..=config.max_string_length);
    let mut bytes = Vec::with_capacity(length);

    for _ in 0..length {
        if rng.gen::<f64>() < config.invalid_utf8_probability {
            // Generate invalid UTF-8 byte
            bytes.push(rng.gen_range(0x80..=0xFF));
        } else {
            // Generate valid ASCII/UTF-8 byte
            let byte = match rng.gen_range(0..4) {
                0 => rng.gen_range(0x20..=0x7E), // Printable ASCII
                1 => rng.gen_range(0x00..=0x1F), // Control characters
                2 => b'\0',                      // Null terminator (dangerous)
                _ => rng.gen_range(0x01..=0xFF), // Any byte
            };
            bytes.push(byte);
        }
    }

    // Try to create a CString, but handle null bytes
    match CString::new(bytes) {
        Ok(s) => Some(s),
        Err(_) => {
            // Contains null bytes, create a string without them
            let safe_string = format!("fuzz_string_{}", rng.gen::<u32>());
            CString::new(safe_string).ok()
        },
    }
}

/// Generate random array of integers
pub fn generate_fuzz_array(config: &FuzzConfig) -> Option<Vec<c_int>> {
    let mut rng = thread_rng();

    if rng.gen::<f64>() < config.null_probability {
        return None;
    }

    let length = rng.gen_range(0..=config.max_array_size);
    let mut array = Vec::with_capacity(length);

    for _ in 0..length {
        let value = match rng.gen_range(0..6) {
            0 => c_int::MAX,
            1 => c_int::MIN,
            2 => 0,
            3 => -1,
            4 => rng.gen::<c_int>(),
            _ => rng.gen_range(-1000..=1000),
        };
        array.push(value);
    }

    Some(array)
}

/// Fuzz test for string validation functions
pub fn fuzz_string_validation(config: &FuzzConfig) -> FuzzResult {
    let mut result = FuzzResult::new();

    for _ in 0..config.iterations {
        let test_string = generate_fuzz_string(config);

        match test_string {
            Some(s) => {
                let c_str = s.as_ptr();

                // Test string validation
                let validation_result = unsafe { trustformers_validate_string_fuzz(c_str) };

                match validation_result {
                    0 => result.passed += 1, // Success (string is valid)
                    _ => result.failed += 1, // Expected failure (string is invalid)
                }
            },
            None => {
                // Test null pointer handling
                let validation_result = unsafe { trustformers_validate_string_fuzz(ptr::null()) };

                if validation_result == TrustformersError::NullPointer as c_int {
                    result.passed += 1;
                } else {
                    result.failed += 1;
                    result.errors.push("Null pointer not handled correctly".to_string());
                }
            },
        }
    }

    result
}

/// Fuzz test for memory allocation functions
pub fn fuzz_memory_operations(config: &FuzzConfig) -> FuzzResult {
    let mut result = FuzzResult::new();

    for _ in 0..config.iterations {
        let mut rng = thread_rng();
        let size = rng.gen_range(0..=config.max_array_size * 1024);

        // Test memory allocation with various sizes
        let alloc_result = unsafe { trustformers_alloc_memory(size as c_uint) };

        if !alloc_result.is_null() {
            // Memory allocated successfully, test freeing it
            unsafe {
                trustformers_free_memory(alloc_result);
            }
            result.passed += 1;
        } else if size == 0 {
            // Zero-size allocation might legitimately fail
            result.passed += 1;
        } else {
            result.failed += 1;
            result.errors.push(format!("Memory allocation failed for size {}", size));
        }
    }

    result
}

/// Comprehensive fuzzing test suite
pub fn run_fuzz_suite(config: &FuzzConfig) -> FuzzResult {
    let mut total_result = FuzzResult::new();

    // Run individual fuzz tests
    let string_result = fuzz_string_validation(config);
    let memory_result = fuzz_memory_operations(config);

    // Combine results
    total_result.passed = string_result.passed + memory_result.passed;
    total_result.failed = string_result.failed + memory_result.failed;
    total_result.crashes = string_result.crashes + memory_result.crashes;

    total_result.errors.extend(string_result.errors);
    total_result.errors.extend(memory_result.errors);

    total_result
}

/// C API for fuzzing
#[no_mangle]
pub extern "C" fn trustformers_run_fuzz_tests(iterations: c_uint) -> c_int {
    let config = FuzzConfig {
        iterations: iterations as usize,
        ..Default::default()
    };

    let result = run_fuzz_suite(&config);

    // Return success rate as percentage
    (result.success_rate() * 100.0) as c_int
}

/// Get the number of fuzzing errors found
#[no_mangle]
pub extern "C" fn trustformers_get_fuzz_error_count() -> c_uint {
    // In a real implementation, this would return the count from the last run
    // For now, return 0 to indicate no critical errors
    0
}

/// Check if fuzzing is available
#[no_mangle]
pub extern "C" fn trustformers_fuzzing_available() -> c_int {
    1 // Always available
}

// Placeholder C API functions that would be tested by fuzzing
// These are dummy implementations for demonstration

#[no_mangle]
pub extern "C" fn trustformers_validate_string_fuzz(s: *const c_char) -> c_int {
    if s.is_null() {
        return TrustformersError::NullPointer as c_int;
    }

    unsafe {
        match CStr::from_ptr(s).to_str() {
            Ok(_) => TrustformersError::Success as c_int,
            Err(_) => TrustformersError::InvalidParameter as c_int,
        }
    }
}

#[no_mangle]
pub extern "C" fn trustformers_alloc_memory(size: c_uint) -> *mut std::ffi::c_void {
    if size == 0 {
        return ptr::null_mut();
    }

    let layout = std::alloc::Layout::from_size_align(size as usize, 1).unwrap();
    unsafe { std::alloc::alloc(layout) as *mut std::ffi::c_void }
}

#[no_mangle]
pub extern "C" fn trustformers_free_memory(ptr: *mut std::ffi::c_void) {
    if !ptr.is_null() {
        // In a real implementation, we'd need to know the size
        // This is a simplified version
        unsafe {
            let layout = std::alloc::Layout::from_size_align(1, 1).unwrap();
            std::alloc::dealloc(ptr as *mut u8, layout);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuzz_config_default() {
        let config = FuzzConfig::default();
        assert!(config.max_string_length > 0);
        assert!(config.iterations > 0);
        assert!(config.null_probability >= 0.0 && config.null_probability <= 1.0);
    }

    #[test]
    fn test_generate_fuzz_string() {
        let config = FuzzConfig::default();

        // Test multiple generations
        for _ in 0..10 {
            let result = generate_fuzz_string(&config);
            // Should either be Some(CString) or None
            match result {
                Some(_s) => {
                    // String is valid, no need to check as_ptr() for null (it's never null)
                },
                None => {
                    // Null generation is expected sometimes
                },
            }
        }
    }

    #[test]
    fn test_fuzz_result() {
        let mut result = FuzzResult::new();
        result.passed = 80;
        result.failed = 20;

        assert_eq!(result.total_tests(), 100);
        assert_eq!(result.success_rate(), 0.8);
    }

    #[test]
    fn test_string_validation_fuzzing() {
        let config = FuzzConfig {
            iterations: 10,
            ..Default::default()
        };

        let result = fuzz_string_validation(&config);
        assert_eq!(result.total_tests(), 10);
    }
}
