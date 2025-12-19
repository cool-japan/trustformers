//! C/Rust String Conversion Utilities
//!
//! This module provides safe and efficient utilities for converting between
//! C strings and Rust strings, with comprehensive error handling and memory management.

use crate::error::{TrustformersError, TrustformersResult};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

/// Utility function to safely convert C string to Rust string
pub fn c_str_to_string(c_str: *const c_char) -> TrustformersResult<String> {
    if c_str.is_null() {
        return Err(TrustformersError::NullPointer);
    }

    unsafe {
        Ok(CStr::from_ptr(c_str)
            .to_str()
            .map_err(|_| TrustformersError::InvalidFormat)?
            .to_string())
    }
}

/// Utility function to safely convert Rust string to C string
pub fn string_to_c_str(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(c_string) => c_string.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Utility function to safely convert Rust string slice to C string
pub fn str_to_c_str(s: &str) -> *mut c_char {
    match CString::new(s) {
        Ok(c_string) => c_string.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Utility function to convert Rust Result to C error code with optional value extraction
pub fn result_to_error<T>(result: TrustformersResult<T>) -> (TrustformersError, Option<T>) {
    match result {
        Ok(value) => (TrustformersError::Success, Some(value)),
        Err(e) => {
            eprintln!("TrustformeRS Error: {:?}", e);
            (e, None)
        },
    }
}

/// Free a C string that was allocated by this library
pub unsafe fn free_c_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        let _ = CString::from_raw(ptr);
    }
}

/// Safe wrapper for C string operations
pub struct SafeCString {
    ptr: *mut c_char,
}

impl SafeCString {
    /// Create a new SafeCString from a Rust string
    pub fn new(s: String) -> Option<Self> {
        let ptr = string_to_c_str(s);
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Create a new SafeCString from a Rust string slice
    pub fn from_str(s: &str) -> Option<Self> {
        let ptr = str_to_c_str(s);
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Get the raw C string pointer
    pub fn as_ptr(&self) -> *const c_char {
        self.ptr
    }

    /// Get the mutable raw C string pointer
    pub fn as_mut_ptr(&mut self) -> *mut c_char {
        self.ptr
    }

    /// Convert back to Rust string (consumes self)
    pub fn into_string(self) -> TrustformersResult<String> {
        let result = c_str_to_string(self.ptr);
        // Prevent drop from freeing the memory since we're consuming it
        std::mem::forget(self);
        result
    }

    /// Get a copy of the string content without consuming self
    pub fn to_string(&self) -> TrustformersResult<String> {
        c_str_to_string(self.ptr)
    }

    /// Release ownership of the C string pointer
    pub fn into_raw(self) -> *mut c_char {
        let ptr = self.ptr;
        std::mem::forget(self);
        ptr
    }
}

impl Drop for SafeCString {
    fn drop(&mut self) {
        unsafe {
            free_c_string(self.ptr);
        }
    }
}

/// String conversion utilities for common patterns
pub mod conversion_utils {
    use super::*;

    /// Convert an optional string to C string
    pub fn option_string_to_c_str(s: Option<String>) -> *mut c_char {
        match s {
            Some(string) => string_to_c_str(string),
            None => ptr::null_mut(),
        }
    }

    /// Convert a vector of strings to array of C strings
    pub fn string_vec_to_c_array(strings: Vec<String>) -> Vec<*mut c_char> {
        strings.into_iter().map(string_to_c_str).collect()
    }

    /// Free an array of C strings
    pub unsafe fn free_c_string_array(array: Vec<*mut c_char>) {
        for ptr in array {
            free_c_string(ptr);
        }
    }

    /// Convert C string array to Rust string vector
    pub unsafe fn c_string_array_to_vec(
        array: *const *const c_char,
        len: usize,
    ) -> TrustformersResult<Vec<String>> {
        if array.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            let c_str_ptr = *array.add(i);
            if c_str_ptr.is_null() {
                return Err(TrustformersError::NullPointer);
            }
            result.push(c_str_to_string(c_str_ptr)?);
        }
        Ok(result)
    }

    /// Check if a C string is valid UTF-8 without converting
    pub fn is_valid_utf8_c_str(c_str: *const c_char) -> bool {
        if c_str.is_null() {
            return false;
        }

        unsafe {
            CStr::from_ptr(c_str).to_str().is_ok()
        }
    }

    /// Get the length of a C string safely
    pub fn c_str_len(c_str: *const c_char) -> TrustformersResult<usize> {
        if c_str.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        unsafe { Ok(CStr::from_ptr(c_str).to_bytes().len()) }
    }

    /// Copy a C string to a new allocation
    pub fn c_str_clone(c_str: *const c_char) -> TrustformersResult<*mut c_char> {
        if c_str.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        unsafe {
            let rust_str =
                CStr::from_ptr(c_str).to_str().map_err(|_| TrustformersError::InvalidFormat)?;
            Ok(str_to_c_str(rust_str))
        }
    }

    /// Compare two C strings for equality
    pub fn c_str_equal(s1: *const c_char, s2: *const c_char) -> TrustformersResult<bool> {
        if s1.is_null() || s2.is_null() {
            return Ok(s1 == s2); // Both null = equal, one null = not equal
        }

        unsafe {
            let str1 = CStr::from_ptr(s1).to_str().map_err(|_| TrustformersError::InvalidFormat)?;
            let str2 = CStr::from_ptr(s2).to_str().map_err(|_| TrustformersError::InvalidFormat)?;
            Ok(str1 == str2)
        }
    }

    /// Concatenate two C strings
    pub fn c_str_concat(s1: *const c_char, s2: *const c_char) -> TrustformersResult<*mut c_char> {
        if s1.is_null() || s2.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        unsafe {
            let str1 = CStr::from_ptr(s1).to_str().map_err(|_| TrustformersError::InvalidFormat)?;
            let str2 = CStr::from_ptr(s2).to_str().map_err(|_| TrustformersError::InvalidFormat)?;
            let concatenated = format!("{}{}", str1, str2);
            Ok(string_to_c_str(concatenated))
        }
    }

    /// Trim whitespace from a C string and return new allocation
    pub fn c_str_trim(c_str: *const c_char) -> TrustformersResult<*mut c_char> {
        if c_str.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        unsafe {
            let rust_str =
                CStr::from_ptr(c_str).to_str().map_err(|_| TrustformersError::InvalidFormat)?;
            let trimmed = rust_str.trim();
            Ok(str_to_c_str(trimmed))
        }
    }

    /// Convert C string to lowercase
    pub fn c_str_to_lowercase(c_str: *const c_char) -> TrustformersResult<*mut c_char> {
        if c_str.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        unsafe {
            let rust_str =
                CStr::from_ptr(c_str).to_str().map_err(|_| TrustformersError::InvalidFormat)?;
            let lowercase = rust_str.to_lowercase();
            Ok(string_to_c_str(lowercase))
        }
    }

    /// Convert C string to uppercase
    pub fn c_str_to_uppercase(c_str: *const c_char) -> TrustformersResult<*mut c_char> {
        if c_str.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        unsafe {
            let rust_str =
                CStr::from_ptr(c_str).to_str().map_err(|_| TrustformersError::InvalidFormat)?;
            let uppercase = rust_str.to_uppercase();
            Ok(string_to_c_str(uppercase))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_c_string() {
        let safe_str = SafeCString::from_str("Hello, World!").unwrap();
        let back_to_rust = safe_str.to_string().unwrap();
        assert_eq!(back_to_rust, "Hello, World!");
    }

    #[test]
    fn test_string_conversion() {
        let original = "Test String";
        let c_str = str_to_c_str(original);
        assert!(!c_str.is_null());

        let converted = c_str_to_string(c_str).unwrap();
        assert_eq!(converted, original);

        unsafe {
            free_c_string(c_str);
        }
    }

    #[test]
    fn test_null_pointer_handling() {
        assert!(c_str_to_string(ptr::null()).is_err());
        assert_eq!(
            string_to_c_str("test\0embedded_null".to_string()),
            ptr::null_mut()
        );
    }
}
