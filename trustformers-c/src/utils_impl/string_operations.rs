//! String Manipulation and Operations
//!
//! This module provides comprehensive string manipulation utilities optimized for
//! C API usage, including safe operations, comparisons, and transformations.

use super::string_conversion::*;
use super::validation::input_validation::*;
use crate::error::{TrustformersError, TrustformersResult};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;

// Re-export commonly used functions for legacy compatibility
pub use core_operations::{
    compare_strings as string_compare, concatenate_strings as string_concat,
    copy_string as string_copy, get_string_length as string_length,
};

pub use search_operations::{contains_substring, find_substring as string_find};

pub use transformations::{to_lowercase, to_uppercase, trim_whitespace as string_trim};

/// Core string operations for C API
pub mod core_operations {
    use super::*;

    /// Get the length of a C string safely
    pub fn get_string_length(s: *const c_char) -> TrustformersResult<usize> {
        if s.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        unsafe { Ok(CStr::from_ptr(s).to_bytes().len()) }
    }

    /// Copy C string to a new buffer
    pub fn copy_string(src: *const c_char) -> TrustformersResult<*mut c_char> {
        if src.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        let rust_str = c_str_to_string(src)?;
        Ok(string_to_c_str(rust_str))
    }

    /// Compare two C strings for equality
    pub fn compare_strings(s1: *const c_char, s2: *const c_char) -> TrustformersResult<c_int> {
        if s1.is_null() || s2.is_null() {
            return Ok(if s1 == s2 { 0 } else { -2 }); // Both null = equal, one null = error
        }

        unsafe {
            let str1 = CStr::from_ptr(s1).to_str().map_err(|_| TrustformersError::InvalidFormat)?;
            let str2 = CStr::from_ptr(s2).to_str().map_err(|_| TrustformersError::InvalidFormat)?;

            Ok(match str1.cmp(str2) {
                std::cmp::Ordering::Equal => 0,
                std::cmp::Ordering::Less => -1,
                std::cmp::Ordering::Greater => 1,
            })
        }
    }

    /// Compare two C strings ignoring case
    pub fn compare_strings_ignore_case(
        s1: *const c_char,
        s2: *const c_char,
    ) -> TrustformersResult<c_int> {
        if s1.is_null() || s2.is_null() {
            return Ok(if s1 == s2 { 0 } else { -2 });
        }

        let str1 = c_str_to_string(s1)?.to_lowercase();
        let str2 = c_str_to_string(s2)?.to_lowercase();

        Ok(match str1.cmp(&str2) {
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Greater => 1,
        })
    }

    /// Check if two C strings are equal
    pub fn strings_equal(s1: *const c_char, s2: *const c_char) -> bool {
        match compare_strings(s1, s2) {
            Ok(0) => true,
            _ => false,
        }
    }

    /// Concatenate two C strings
    pub fn concatenate_strings(
        s1: *const c_char,
        s2: *const c_char,
    ) -> TrustformersResult<*mut c_char> {
        if s1.is_null() || s2.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        let str1 = c_str_to_string(s1)?;
        let str2 = c_str_to_string(s2)?;
        let concatenated = format!("{}{}", str1, str2);

        Ok(string_to_c_str(concatenated))
    }

    /// Concatenate multiple C strings
    pub fn concatenate_multiple(strings: &[*const c_char]) -> TrustformersResult<*mut c_char> {
        if strings.is_empty() {
            return Ok(string_to_c_str(String::new()));
        }

        let mut result = String::new();
        for &s in strings {
            if s.is_null() {
                return Err(TrustformersError::NullPointer);
            }
            result.push_str(&c_str_to_string(s)?);
        }

        Ok(string_to_c_str(result))
    }

    /// Join C strings with a separator
    pub fn join_strings(
        strings: &[*const c_char],
        separator: *const c_char,
    ) -> TrustformersResult<*mut c_char> {
        if strings.is_empty() {
            return Ok(string_to_c_str(String::new()));
        }

        let sep = if separator.is_null() { String::new() } else { c_str_to_string(separator)? };

        let mut rust_strings = Vec::new();
        for &s in strings {
            if s.is_null() {
                return Err(TrustformersError::NullPointer);
            }
            rust_strings.push(c_str_to_string(s)?);
        }

        let joined = rust_strings.join(&sep);
        Ok(string_to_c_str(joined))
    }
}

/// String transformation operations
pub mod transformations {
    use super::*;

    /// Convert C string to lowercase
    pub fn to_lowercase(s: *const c_char) -> TrustformersResult<*mut c_char> {
        if s.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        let rust_str = c_str_to_string(s)?;
        let lowercase = rust_str.to_lowercase();
        Ok(string_to_c_str(lowercase))
    }

    /// Convert C string to uppercase
    pub fn to_uppercase(s: *const c_char) -> TrustformersResult<*mut c_char> {
        if s.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        let rust_str = c_str_to_string(s)?;
        let uppercase = rust_str.to_uppercase();
        Ok(string_to_c_str(uppercase))
    }

    /// Trim whitespace from C string
    pub fn trim_whitespace(s: *const c_char) -> TrustformersResult<*mut c_char> {
        if s.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        let rust_str = c_str_to_string(s)?;
        let trimmed = rust_str.trim().to_string();
        Ok(string_to_c_str(trimmed))
    }

    /// Trim whitespace from start of C string
    pub fn trim_start(s: *const c_char) -> TrustformersResult<*mut c_char> {
        if s.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        let rust_str = c_str_to_string(s)?;
        let trimmed = rust_str.trim_start().to_string();
        Ok(string_to_c_str(trimmed))
    }

    /// Trim whitespace from end of C string
    pub fn trim_end(s: *const c_char) -> TrustformersResult<*mut c_char> {
        if s.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        let rust_str = c_str_to_string(s)?;
        let trimmed = rust_str.trim_end().to_string();
        Ok(string_to_c_str(trimmed))
    }

    /// Replace all occurrences of a pattern in a string
    pub fn replace_all(
        s: *const c_char,
        pattern: *const c_char,
        replacement: *const c_char,
    ) -> TrustformersResult<*mut c_char> {
        if s.is_null() || pattern.is_null() || replacement.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        let rust_str = c_str_to_string(s)?;
        let pattern_str = c_str_to_string(pattern)?;
        let replacement_str = c_str_to_string(replacement)?;

        let result = rust_str.replace(&pattern_str, &replacement_str);
        Ok(string_to_c_str(result))
    }

    /// Reverse a string
    pub fn reverse_string(s: *const c_char) -> TrustformersResult<*mut c_char> {
        if s.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        let rust_str = c_str_to_string(s)?;
        let reversed: String = rust_str.chars().rev().collect();
        Ok(string_to_c_str(reversed))
    }

    /// Repeat a string n times
    pub fn repeat_string(s: *const c_char, n: usize) -> TrustformersResult<*mut c_char> {
        if s.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        if n > 10000 {
            return Err(TrustformersError::ResourceLimitExceeded);
        }

        let rust_str = c_str_to_string(s)?;
        let repeated = rust_str.repeat(n);
        Ok(string_to_c_str(repeated))
    }

    /// Pad string to specified length with character
    pub fn pad_string(
        s: *const c_char,
        target_len: usize,
        pad_char: c_char,
        pad_left: bool,
    ) -> TrustformersResult<*mut c_char> {
        if s.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        if target_len > 100000 {
            return Err(TrustformersError::ResourceLimitExceeded);
        }

        let rust_str = c_str_to_string(s)?;
        let pad_character = char::from(pad_char as u8);

        if rust_str.len() >= target_len {
            return Ok(string_to_c_str(rust_str));
        }

        let padding_needed = target_len - rust_str.len();
        let padding: String = pad_character.to_string().repeat(padding_needed);

        let result = if pad_left {
            format!("{}{}", padding, rust_str)
        } else {
            format!("{}{}", rust_str, padding)
        };

        Ok(string_to_c_str(result))
    }
}

/// String search and pattern matching operations
pub mod search_operations {
    use super::*;

    /// Check if string contains a substring
    pub fn contains_substring(
        haystack: *const c_char,
        needle: *const c_char,
    ) -> TrustformersResult<bool> {
        if haystack.is_null() || needle.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        let haystack_str = c_str_to_string(haystack)?;
        let needle_str = c_str_to_string(needle)?;

        Ok(haystack_str.contains(&needle_str))
    }

    /// Check if string starts with a prefix
    pub fn starts_with(s: *const c_char, prefix: *const c_char) -> TrustformersResult<bool> {
        if s.is_null() || prefix.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        let string = c_str_to_string(s)?;
        let prefix_str = c_str_to_string(prefix)?;

        Ok(string.starts_with(&prefix_str))
    }

    /// Check if string ends with a suffix
    pub fn ends_with(s: *const c_char, suffix: *const c_char) -> TrustformersResult<bool> {
        if s.is_null() || suffix.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        let string = c_str_to_string(s)?;
        let suffix_str = c_str_to_string(suffix)?;

        Ok(string.ends_with(&suffix_str))
    }

    /// Find first occurrence of substring
    pub fn find_substring(
        haystack: *const c_char,
        needle: *const c_char,
    ) -> TrustformersResult<i32> {
        if haystack.is_null() || needle.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        let haystack_str = c_str_to_string(haystack)?;
        let needle_str = c_str_to_string(needle)?;

        match haystack_str.find(&needle_str) {
            Some(index) => Ok(index as i32),
            None => Ok(-1),
        }
    }

    /// Count occurrences of substring
    pub fn count_occurrences(
        haystack: *const c_char,
        needle: *const c_char,
    ) -> TrustformersResult<usize> {
        if haystack.is_null() || needle.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        let haystack_str = c_str_to_string(haystack)?;
        let needle_str = c_str_to_string(needle)?;

        if needle_str.is_empty() {
            return Ok(0);
        }

        Ok(haystack_str.matches(&needle_str).count())
    }

    /// Split string by delimiter
    pub fn split_string(
        s: *const c_char,
        delimiter: *const c_char,
    ) -> TrustformersResult<Vec<String>> {
        if s.is_null() || delimiter.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        let string = c_str_to_string(s)?;
        let delim = c_str_to_string(delimiter)?;

        let parts: Vec<String> = string.split(&delim).map(|s| s.to_string()).collect();

        // Limit number of parts to prevent DoS
        if parts.len() > 10000 {
            return Err(TrustformersError::ResourceLimitExceeded);
        }

        Ok(parts)
    }

    /// Check if string matches a simple glob pattern
    pub fn matches_glob_pattern(
        s: *const c_char,
        pattern: *const c_char,
    ) -> TrustformersResult<bool> {
        if s.is_null() || pattern.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        let string = c_str_to_string(s)?;
        let pattern_str = c_str_to_string(pattern)?;

        Ok(simple_glob_match(&string, &pattern_str))
    }

    /// Simple glob pattern matching implementation
    fn simple_glob_match(text: &str, pattern: &str) -> bool {
        let text_chars: Vec<char> = text.chars().collect();
        let pattern_chars: Vec<char> = pattern.chars().collect();

        fn is_match(text: &[char], pattern: &[char], text_idx: usize, pattern_idx: usize) -> bool {
            if pattern_idx == pattern.len() {
                return text_idx == text.len();
            }

            if text_idx == text.len() {
                return pattern[pattern_idx..].iter().all(|&c| c == '*');
            }

            match pattern[pattern_idx] {
                '*' => {
                    // Try matching zero characters or one+ characters
                    is_match(text, pattern, text_idx, pattern_idx + 1)
                        || is_match(text, pattern, text_idx + 1, pattern_idx)
                },
                '?' => {
                    // Match any single character
                    is_match(text, pattern, text_idx + 1, pattern_idx + 1)
                },
                c => {
                    // Match literal character
                    text[text_idx] == c && is_match(text, pattern, text_idx + 1, pattern_idx + 1)
                },
            }
        }

        is_match(&text_chars, &pattern_chars, 0, 0)
    }
}

/// String formatting and building operations
pub mod formatting {
    use super::*;

    /// Format string with printf-style formatting (simplified)
    pub fn format_string(format: *const c_char, args: &[&str]) -> TrustformersResult<*mut c_char> {
        if format.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        let format_str = c_str_to_string(format)?;
        let result = simple_format(&format_str, args)?;

        Ok(string_to_c_str(result))
    }

    /// Simple string formatting implementation
    fn simple_format(format_str: &str, args: &[&str]) -> TrustformersResult<String> {
        let mut result = format_str.to_string();
        let mut arg_index = 0;

        // Replace {} placeholders with arguments
        while let Some(start) = result.find("{}") {
            if arg_index >= args.len() {
                return Err(TrustformersError::InvalidParameter);
            }

            result.replace_range(start..start + 2, args[arg_index]);
            arg_index += 1;
        }

        Ok(result)
    }

    /// Build string by appending multiple parts
    pub fn build_string(parts: &[*const c_char]) -> TrustformersResult<*mut c_char> {
        if parts.is_empty() {
            return Ok(string_to_c_str(String::new()));
        }

        let mut builder = String::new();
        for &part in parts {
            if part.is_null() {
                return Err(TrustformersError::NullPointer);
            }
            builder.push_str(&c_str_to_string(part)?);
        }

        Ok(string_to_c_str(builder))
    }

    /// Create a string filled with a repeated character
    pub fn create_filled_string(c: c_char, count: usize) -> TrustformersResult<*mut c_char> {
        if count > 100000 {
            return Err(TrustformersError::ResourceLimitExceeded);
        }

        let character = char::from(c as u8);
        let filled = character.to_string().repeat(count);

        Ok(string_to_c_str(filled))
    }
}

/// Encoding and decoding utilities
pub mod encoding {
    use super::*;

    /// Escape special characters for JSON
    pub fn escape_json(s: *const c_char) -> TrustformersResult<*mut c_char> {
        if s.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        let rust_str = c_str_to_string(s)?;
        let escaped = rust_str
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t");

        Ok(string_to_c_str(escaped))
    }

    /// Unescape JSON special characters
    pub fn unescape_json(s: *const c_char) -> TrustformersResult<*mut c_char> {
        if s.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        let rust_str = c_str_to_string(s)?;
        let unescaped = rust_str
            .replace("\\\\", "\\")
            .replace("\\\"", "\"")
            .replace("\\n", "\n")
            .replace("\\r", "\r")
            .replace("\\t", "\t");

        Ok(string_to_c_str(unescaped))
    }

    /// URL encode a string
    pub fn url_encode(s: *const c_char) -> TrustformersResult<*mut c_char> {
        if s.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        let rust_str = c_str_to_string(s)?;
        let encoded = simple_url_encode(&rust_str);

        Ok(string_to_c_str(encoded))
    }

    /// Simple URL encoding implementation
    fn simple_url_encode(s: &str) -> String {
        s.chars()
            .map(|c| match c {
                'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
                ' ' => "+".to_string(),
                _ => format!("%{:02X}", c as u8),
            })
            .collect()
    }

    /// Base64 encode (simplified implementation)
    pub fn base64_encode(s: *const c_char) -> TrustformersResult<*mut c_char> {
        if s.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        let rust_str = c_str_to_string(s)?;
        let encoded = simple_base64_encode(rust_str.as_bytes());

        Ok(string_to_c_str(encoded))
    }

    /// Simple base64 encoding
    fn simple_base64_encode(data: &[u8]) -> String {
        const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut result = String::new();

        for chunk in data.chunks(3) {
            let mut buf = [0u8; 3];
            for (i, &byte) in chunk.iter().enumerate() {
                buf[i] = byte;
            }

            let b = ((buf[0] as u32) << 16) | ((buf[1] as u32) << 8) | (buf[2] as u32);

            result.push(ALPHABET[((b >> 18) & 63) as usize] as char);
            result.push(ALPHABET[((b >> 12) & 63) as usize] as char);
            result.push(if chunk.len() > 1 {
                ALPHABET[((b >> 6) & 63) as usize] as char
            } else {
                '='
            });
            result.push(if chunk.len() > 2 { ALPHABET[(b & 63) as usize] as char } else { '=' });
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::core_operations::*;
    use super::search_operations::*;
    use super::transformations::*;
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_string_length() {
        let test_str = CString::new("Hello, World!").unwrap();
        let length = get_string_length(test_str.as_ptr()).unwrap();
        assert_eq!(length, 13);
    }

    #[test]
    fn test_string_copy() {
        let test_str = CString::new("Test").unwrap();
        let copied = copy_string(test_str.as_ptr()).unwrap();

        unsafe {
            let copied_str = CStr::from_ptr(copied);
            assert_eq!(copied_str, test_str.as_c_str());
            let _ = CString::from_raw(copied); // Free the memory
        }
    }

    #[test]
    fn test_string_comparison() {
        let str1 = CString::new("abc").unwrap();
        let str2 = CString::new("def").unwrap();
        let str3 = CString::new("abc").unwrap();

        assert_eq!(compare_strings(str1.as_ptr(), str2.as_ptr()).unwrap(), -1);
        assert_eq!(compare_strings(str1.as_ptr(), str3.as_ptr()).unwrap(), 0);
        assert_eq!(compare_strings(str2.as_ptr(), str1.as_ptr()).unwrap(), 1);
    }

    #[test]
    fn test_string_concatenation() {
        let str1 = CString::new("Hello, ").unwrap();
        let str2 = CString::new("World!").unwrap();
        let result = concatenate_strings(str1.as_ptr(), str2.as_ptr()).unwrap();

        unsafe {
            let result_str = CStr::from_ptr(result);
            assert_eq!(result_str.to_str().unwrap(), "Hello, World!");
            let _ = CString::from_raw(result); // Free the memory
        }
    }

    #[test]
    fn test_case_transformations() {
        let test_str = CString::new("Hello World").unwrap();

        let lower = to_lowercase(test_str.as_ptr()).unwrap();
        let upper = to_uppercase(test_str.as_ptr()).unwrap();

        unsafe {
            assert_eq!(CStr::from_ptr(lower).to_str().unwrap(), "hello world");
            assert_eq!(CStr::from_ptr(upper).to_str().unwrap(), "HELLO WORLD");

            let _ = CString::from_raw(lower);
            let _ = CString::from_raw(upper);
        }
    }

    #[test]
    fn test_substring_search() {
        let haystack = CString::new("The quick brown fox").unwrap();
        let needle = CString::new("quick").unwrap();

        assert!(contains_substring(haystack.as_ptr(), needle.as_ptr()).unwrap());
        assert_eq!(
            find_substring(haystack.as_ptr(), needle.as_ptr()).unwrap(),
            4
        );
    }
}
