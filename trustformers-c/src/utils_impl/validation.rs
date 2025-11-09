//! Input Validation Functions for Security
//!
//! This module provides comprehensive input validation utilities to ensure
//! security and safety when processing external inputs, particularly from C APIs.

use super::string_conversion::c_str_to_string;
use super::types::*;
use crate::error::{TrustformersError, TrustformersResult};
use std::os::raw::c_char;

// Re-export commonly used validation functions for compatibility
pub use input_validation::validate_c_string_safe as validate_string;
pub use model_validation::validate_model_name;
pub use path_validation::validate_file_path;

/// Comprehensive string validation with all security checks
pub fn validate_string_comprehensive(
    s: *const c_char,
    max_len: Option<usize>,
    allow_unicode: bool,
) -> TrustformersResult<String> {
    if allow_unicode {
        input_validation::validate_c_string_safe(s, max_len)
    } else {
        input_validation::validate_c_string_ascii_only(s, max_len)
    }
}

/// Enhanced input validation utilities with security focus
pub mod input_validation {
    use super::*;

    /// Validate C string with comprehensive length and content checks
    pub fn validate_c_string_safe(
        s: *const c_char,
        max_len: Option<usize>,
    ) -> TrustformersResult<String> {
        if s.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        let max_length = max_len.unwrap_or(MAX_STRING_LENGTH);
        let rust_str = c_str_to_string(s)?;

        if rust_str.len() > max_length {
            return Err(TrustformersError::InvalidParameter);
        }

        // Check for null bytes within string (security measure)
        if rust_str.contains('\0') {
            return Err(TrustformersError::InvalidFormat);
        }

        // Check for control characters that could cause issues
        if rust_str.chars().any(|c| !validation_patterns::is_safe_control_char(c)) {
            return Err(TrustformersError::InvalidFormat);
        }

        Ok(rust_str)
    }

    /// Validate C string with stricter ASCII-only requirements
    pub fn validate_c_string_ascii_only(
        s: *const c_char,
        max_len: Option<usize>,
    ) -> TrustformersResult<String> {
        let validated_str = validate_c_string_safe(s, max_len)?;

        // Check for ASCII-only characters
        if !validated_str.is_ascii() {
            return Err(TrustformersError::InvalidFormat);
        }

        Ok(validated_str)
    }

    /// Validate and normalize whitespace in C string
    pub fn validate_and_normalize_whitespace(
        s: *const c_char,
        max_len: Option<usize>,
    ) -> TrustformersResult<String> {
        let validated_str = validate_c_string_safe(s, max_len)?;

        // Normalize consecutive whitespace to single spaces
        let normalized = validated_str.split_whitespace().collect::<Vec<&str>>().join(" ");

        Ok(normalized)
    }
}

/// Model name validation with security considerations
pub mod model_validation {
    use super::*;

    /// Validate model name for security and format compliance
    pub fn validate_model_name(name: *const c_char) -> TrustformersResult<String> {
        let model_name =
            input_validation::validate_c_string_safe(name, Some(MAX_MODEL_NAME_LENGTH))?;

        // Model names should be alphanumeric with limited special characters
        if !model_name.chars().all(validation_patterns::is_valid_model_name_char) {
            return Err(TrustformersError::InvalidParameter);
        }

        // Prevent path traversal
        if model_name.contains("..") || model_name.starts_with('/') {
            return Err(TrustformersError::ValidationError);
        }

        // Prevent overly long path segments
        if model_name.split('/').any(|segment| segment.len() > 64) {
            return Err(TrustformersError::InvalidParameter);
        }

        // Check for reserved model names
        if is_reserved_model_name(&model_name) {
            return Err(TrustformersError::InvalidParameter);
        }

        Ok(model_name)
    }

    /// Check if a model name is reserved
    fn is_reserved_model_name(name: &str) -> bool {
        const RESERVED_NAMES: &[&str] = &[
            "con",
            "prn",
            "aux",
            "nul", // Windows reserved
            ".",
            "..", // Path navigation
            "null",
            "undefined", // Common programming nulls
        ];

        RESERVED_NAMES.iter().any(|&reserved| name.eq_ignore_ascii_case(reserved))
    }

    /// Validate and normalize model version string
    pub fn validate_model_version(version: *const c_char) -> TrustformersResult<String> {
        let version_str = input_validation::validate_c_string_safe(version, Some(32))?;

        // Basic semver-like validation: major.minor.patch or similar
        if !version_str.chars().all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-') {
            return Err(TrustformersError::InvalidParameter);
        }

        // Must not start or end with special characters
        if version_str.starts_with('.')
            || version_str.ends_with('.')
            || version_str.starts_with('-')
            || version_str.ends_with('-')
        {
            return Err(TrustformersError::InvalidParameter);
        }

        Ok(version_str)
    }
}

/// File path validation with security considerations
pub mod path_validation {
    use super::*;

    /// Validate file path for security
    pub fn validate_file_path(path: *const c_char) -> TrustformersResult<String> {
        let file_path = input_validation::validate_c_string_safe(path, Some(MAX_FILE_PATH_LENGTH))?;

        // Basic path validation
        if file_path.is_empty() {
            return Err(TrustformersError::InvalidParameter);
        }

        // Check for path traversal attempts
        if file_path.contains("..") {
            return Err(TrustformersError::ValidationError);
        }

        // Check for suspicious patterns
        for pattern in validation_patterns::SUSPICIOUS_PATH_PATTERNS {
            if file_path.contains(pattern) {
                return Err(TrustformersError::ValidationError);
            }
        }

        // Prevent absolute paths in certain contexts
        if is_absolute_path_dangerous(&file_path) {
            return Err(TrustformersError::ValidationError);
        }

        Ok(file_path)
    }

    /// Check if absolute path is dangerous in current context
    fn is_absolute_path_dangerous(path: &str) -> bool {
        // Block absolute paths to system directories
        const DANGEROUS_PREFIXES: &[&str] = &[
            "/etc/",
            "/proc/",
            "/sys/",
            "/dev/",
            "C:\\Windows\\",
            "C:\\System32\\",
            "/usr/bin/",
            "/bin/",
            "/sbin/",
        ];

        DANGEROUS_PREFIXES.iter().any(|&prefix| path.starts_with(prefix))
    }

    /// Validate directory path
    pub fn validate_directory_path(path: *const c_char) -> TrustformersResult<String> {
        let dir_path = validate_file_path(path)?;

        // Directories should end with separator or not contain file extensions
        if dir_path.contains('.') && !dir_path.ends_with('/') && !dir_path.ends_with('\\') {
            // Might be a file, check if it has a suspicious extension
            if let Some(extension) = dir_path.split('.').last() {
                if is_executable_extension(extension) {
                    return Err(TrustformersError::ValidationError);
                }
            }
        }

        Ok(dir_path)
    }

    /// Check if file extension indicates executable content
    fn is_executable_extension(ext: &str) -> bool {
        const EXECUTABLE_EXTENSIONS: &[&str] = &[
            "exe", "bat", "cmd", "com", "scr", "pif", "sh", "bash", "zsh", "fish", "py", "pl",
            "rb", "js", "vbs", "ps1",
        ];

        EXECUTABLE_EXTENSIONS.iter().any(|&exec_ext| ext.eq_ignore_ascii_case(exec_ext))
    }
}

/// JSON validation utilities
pub mod json_validation {
    use super::*;

    /// Validate JSON string for parsing safety
    pub fn validate_json_string(json_str: *const c_char) -> TrustformersResult<serde_json::Value> {
        let json_string =
            input_validation::validate_c_string_safe(json_str, Some(MAX_JSON_STRING_LENGTH))?;

        // Parse JSON to validate structure
        let parsed: serde_json::Value =
            serde_json::from_str(&json_string).map_err(|_| TrustformersError::InvalidFormat)?;

        // Additional security checks
        validate_json_content(&parsed)?;

        Ok(parsed)
    }

    /// Validate JSON content for security issues
    fn validate_json_content(value: &serde_json::Value) -> TrustformersResult<()> {
        match value {
            serde_json::Value::Object(obj) => {
                // Limit object size to prevent DoS
                if obj.len() > 1000 {
                    return Err(TrustformersError::ResourceLimitExceeded);
                }

                // Check for suspicious keys
                for key in obj.keys() {
                    if is_suspicious_json_key(key) {
                        return Err(TrustformersError::ValidationError);
                    }
                }

                // Recursively validate nested objects
                for val in obj.values() {
                    validate_json_content(val)?;
                }
            },
            serde_json::Value::Array(arr) => {
                // Limit array size
                if arr.len() > 10000 {
                    return Err(TrustformersError::ResourceLimitExceeded);
                }

                // Recursively validate array elements
                for item in arr {
                    validate_json_content(item)?;
                }
            },
            serde_json::Value::String(s) => {
                // Check string length and content
                if s.len() > 10000 {
                    return Err(TrustformersError::ResourceLimitExceeded);
                }

                if contains_suspicious_content(s) {
                    return Err(TrustformersError::ValidationError);
                }
            },
            _ => {}, // Numbers, booleans, null are generally safe
        }

        Ok(())
    }

    /// Check for suspicious JSON object keys
    fn is_suspicious_json_key(key: &str) -> bool {
        const SUSPICIOUS_KEYS: &[&str] = &[
            "__proto__",
            "constructor",
            "prototype",
            "eval",
            "function",
            "script",
            "onload",
        ];

        SUSPICIOUS_KEYS.iter().any(|&suspicious| key.contains(suspicious))
    }

    /// Check for suspicious content in JSON strings
    fn contains_suspicious_content(content: &str) -> bool {
        const SUSPICIOUS_PATTERNS: &[&str] = &[
            "<script",
            "javascript:",
            "data:text/html",
            "onload=",
            "onerror=",
            "onclick=",
        ];

        let lowercase_content = content.to_lowercase();
        SUSPICIOUS_PATTERNS.iter().any(|&pattern| lowercase_content.contains(pattern))
    }
}

/// Numerical validation utilities
pub mod numeric_validation {
    use super::*;

    /// Validate f32 value within specified range
    pub fn validate_range_f32(
        value: f32,
        min: f32,
        max: f32,
        _name: &str,
    ) -> TrustformersResult<f32> {
        if !value.is_finite() {
            return Err(TrustformersError::InvalidParameter);
        }

        if value < min || value > max {
            return Err(TrustformersError::InvalidParameter);
        }

        Ok(value)
    }

    /// Validate f64 value within specified range
    pub fn validate_range_f64(
        value: f64,
        min: f64,
        max: f64,
        _name: &str,
    ) -> TrustformersResult<f64> {
        if !value.is_finite() {
            return Err(TrustformersError::InvalidParameter);
        }

        if value < min || value > max {
            return Err(TrustformersError::InvalidParameter);
        }

        Ok(value)
    }

    /// Validate i32 value within specified range
    pub fn validate_range_i32(
        value: i32,
        min: i32,
        max: i32,
        _name: &str,
    ) -> TrustformersResult<i32> {
        if value < min || value > max {
            return Err(TrustformersError::InvalidParameter);
        }

        Ok(value)
    }

    /// Validate i64 value within specified range
    pub fn validate_range_i64(
        value: i64,
        min: i64,
        max: i64,
        _name: &str,
    ) -> TrustformersResult<i64> {
        if value < min || value > max {
            return Err(TrustformersError::InvalidParameter);
        }

        Ok(value)
    }

    /// Validate unsigned integer within range
    pub fn validate_range_usize(
        value: usize,
        min: usize,
        max: usize,
        _name: &str,
    ) -> TrustformersResult<usize> {
        if value < min || value > max {
            return Err(TrustformersError::InvalidParameter);
        }

        Ok(value)
    }

    /// Validate probability value (0.0 to 1.0)
    pub fn validate_probability(value: f64) -> TrustformersResult<f64> {
        validate_range_f64(value, 0.0, 1.0, "probability")
    }

    /// Validate percentage value (0.0 to 100.0)
    pub fn validate_percentage(value: f64) -> TrustformersResult<f64> {
        validate_range_f64(value, 0.0, 100.0, "percentage")
    }

    /// Validate positive integer
    pub fn validate_positive_i32(value: i32) -> TrustformersResult<i32> {
        if value <= 0 {
            return Err(TrustformersError::InvalidParameter);
        }
        Ok(value)
    }

    /// Validate non-negative integer
    pub fn validate_non_negative_i32(value: i32) -> TrustformersResult<i32> {
        if value < 0 {
            return Err(TrustformersError::InvalidParameter);
        }
        Ok(value)
    }
}

/// Array and pointer validation utilities
pub mod array_validation {
    use super::*;

    /// Validate array pointer and length
    pub fn validate_array_ptr<T>(ptr: *const T, len: usize, _name: &str) -> TrustformersResult<()> {
        if ptr.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        if len == 0 {
            return Err(TrustformersError::InvalidParameter);
        }

        // Check for reasonable array size (prevent DoS)
        if len > MAX_ARRAY_SIZE {
            return Err(TrustformersError::ResourceLimitExceeded);
        }

        Ok(())
    }

    /// Validate mutable array pointer and length
    pub fn validate_array_ptr_mut<T>(
        ptr: *mut T,
        len: usize,
        _name: &str,
    ) -> TrustformersResult<()> {
        if ptr.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        if len == 0 {
            return Err(TrustformersError::InvalidParameter);
        }

        if len > MAX_ARRAY_SIZE {
            return Err(TrustformersError::ResourceLimitExceeded);
        }

        Ok(())
    }

    /// Validate buffer size for operations
    pub fn validate_buffer_size(
        size: usize,
        min_size: usize,
        max_size: usize,
    ) -> TrustformersResult<usize> {
        if size < min_size {
            return Err(TrustformersError::InvalidParameter);
        }

        if size > max_size {
            return Err(TrustformersError::ResourceLimitExceeded);
        }

        Ok(size)
    }

    /// Validate slice bounds
    pub fn validate_slice_bounds(
        start: usize,
        end: usize,
        array_len: usize,
    ) -> TrustformersResult<()> {
        if start > end {
            return Err(TrustformersError::InvalidParameter);
        }

        if end > array_len {
            return Err(TrustformersError::InvalidParameter);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::input_validation::*;
    use super::model_validation::*;
    use super::numeric_validation::*;
    use super::path_validation::*;
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_model_name_validation() {
        let valid_name = CString::new("bert-base-uncased").unwrap();
        assert!(validate_model_name(valid_name.as_ptr()).is_ok());

        let invalid_name = CString::new("../../../etc/passwd").unwrap();
        assert!(validate_model_name(invalid_name.as_ptr()).is_err());
    }

    #[test]
    fn test_path_validation() {
        let valid_path = CString::new("models/bert/config.json").unwrap();
        assert!(validate_file_path(valid_path.as_ptr()).is_ok());

        let invalid_path = CString::new("../../../etc/passwd").unwrap();
        assert!(validate_file_path(invalid_path.as_ptr()).is_err());
    }

    #[test]
    fn test_numeric_validation() {
        assert!(validate_range_f32(0.5, 0.0, 1.0, "test").is_ok());
        assert!(validate_range_f32(1.5, 0.0, 1.0, "test").is_err());
        assert!(validate_range_f32(f32::NAN, 0.0, 1.0, "test").is_err());
    }

    #[test]
    fn test_json_validation() {
        let valid_json = CString::new(r#"{"key": "value", "number": 42}"#).unwrap();
        assert!(json_validation::validate_json_string(valid_json.as_ptr()).is_ok());

        let invalid_json = CString::new("not json").unwrap();
        assert!(json_validation::validate_json_string(invalid_json.as_ptr()).is_err());
    }

    #[test]
    fn test_array_validation() {
        let array = [1, 2, 3, 4, 5];
        assert!(array_validation::validate_array_ptr(array.as_ptr(), array.len(), "test").is_ok());
        assert!(array_validation::validate_array_ptr(std::ptr::null::<i32>(), 5, "test").is_err());
    }
}
