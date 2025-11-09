//! ABI stability and deprecation management

use crate::error::{TrustformersError, TrustformersResult};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_uint};
use std::sync::{Arc, Mutex};

/// ABI version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbiVersion {
    /// Major version number
    pub major: u32,
    /// Minor version number
    pub minor: u32,
    /// Patch version number
    pub patch: u32,
    /// ABI compatibility level
    pub abi_level: u32,
}

impl AbiVersion {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
            abi_level: major * 1000 + minor, // Simple ABI level calculation
        }
    }

    pub fn to_string(&self) -> String {
        format!("{}.{}.{}", self.major, self.minor, self.patch)
    }

    /// Check if this version is compatible with another version
    pub fn is_compatible_with(&self, other: &AbiVersion) -> bool {
        // Compatible if ABI level is the same
        self.abi_level == other.abi_level
    }

    /// Check if this version is newer than another
    pub fn is_newer_than(&self, other: &AbiVersion) -> bool {
        if self.major != other.major {
            return self.major > other.major;
        }
        if self.minor != other.minor {
            return self.minor > other.minor;
        }
        self.patch > other.patch
    }
}

/// Current ABI version
pub const CURRENT_ABI_VERSION: AbiVersion = AbiVersion {
    major: 0,
    minor: 1,
    patch: 0,
    abi_level: 1,
};

/// Deprecation level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeprecationLevel {
    /// Function is deprecated but still works
    Deprecated,
    /// Function will be removed in the next major version
    PendingRemoval,
    /// Function is removed (should not be called)
    Removed,
}

/// Information about a deprecated function
#[derive(Debug, Clone)]
pub struct DeprecationInfo {
    /// The deprecated function name
    pub function_name: String,
    /// Deprecation level
    pub level: DeprecationLevel,
    /// Version when it was deprecated
    pub deprecated_since: AbiVersion,
    /// Version when it will be/was removed
    pub removal_version: Option<AbiVersion>,
    /// Replacement function (if any)
    pub replacement: Option<String>,
    /// Deprecation message
    pub message: String,
}

/// Global deprecation tracking
static DEPRECATION_REGISTRY: Lazy<Arc<Mutex<HashMap<String, DeprecationInfo>>>> =
    Lazy::new(|| Arc::new(Mutex::new(HashMap::new())));

/// Warning tracking to avoid spamming
static WARNING_TRACKER: Lazy<Arc<Mutex<HashMap<String, u32>>>> =
    Lazy::new(|| Arc::new(Mutex::new(HashMap::new())));

/// Register a deprecated function
pub fn register_deprecation(info: DeprecationInfo) {
    let mut registry = DEPRECATION_REGISTRY.lock().unwrap();
    registry.insert(info.function_name.clone(), info);
}

/// Check if a function is deprecated and issue warnings
pub fn check_deprecation(function_name: &str) -> Option<&'static str> {
    let registry = DEPRECATION_REGISTRY.lock().unwrap();

    if let Some(info) = registry.get(function_name) {
        // Track warnings to avoid spam
        let mut tracker = WARNING_TRACKER.lock().unwrap();
        let count = tracker.entry(function_name.to_string()).or_insert(0);
        *count += 1;

        // Only warn for the first few calls
        if *count <= 3 {
            eprintln!(
                "WARNING: Function '{}' is deprecated: {}",
                function_name, info.message
            );

            if let Some(replacement) = &info.replacement {
                eprintln!("         Use '{}' instead.", replacement);
            }

            if let Some(removal_version) = &info.removal_version {
                eprintln!(
                    "         Will be removed in version {}.",
                    removal_version.to_string()
                );
            }
        }

        match info.level {
            DeprecationLevel::Deprecated => Some("deprecated"),
            DeprecationLevel::PendingRemoval => Some("pending_removal"),
            DeprecationLevel::Removed => Some("removed"),
        }
    } else {
        None
    }
}

/// Initialize deprecation registry with known deprecated functions
pub fn init_deprecation_registry() {
    // Example deprecated functions
    register_deprecation(DeprecationInfo {
        function_name: "trustformers_old_api_function".to_string(),
        level: DeprecationLevel::Deprecated,
        deprecated_since: AbiVersion::new(0, 1, 0),
        removal_version: Some(AbiVersion::new(1, 0, 0)),
        replacement: Some("trustformers_new_api_function".to_string()),
        message: "This function has been replaced with a more efficient implementation".to_string(),
    });

    register_deprecation(DeprecationInfo {
        function_name: "trustformers_legacy_tokenizer".to_string(),
        level: DeprecationLevel::PendingRemoval,
        deprecated_since: AbiVersion::new(0, 1, 0),
        removal_version: Some(AbiVersion::new(0, 2, 0)),
        replacement: Some("trustformers_tokenizer_create".to_string()),
        message: "Legacy tokenizer API will be removed soon".to_string(),
    });
}

/// Macro for deprecated function calls
#[macro_export]
macro_rules! deprecated_function {
    ($func_name:expr, $body:block) => {{
        crate::abi_stability::check_deprecation($func_name);
        $body
    }};
}

/// Version negotiation for client compatibility
pub fn negotiate_version(client_version: &AbiVersion) -> TrustformersResult<AbiVersion> {
    if client_version.is_compatible_with(&CURRENT_ABI_VERSION) {
        Ok(CURRENT_ABI_VERSION.clone())
    } else if client_version.is_newer_than(&CURRENT_ABI_VERSION) {
        Err(TrustformersError::ValidationError)
    } else {
        // Try to find a compatible version
        // For now, we only support the current version
        Err(TrustformersError::ValidationError)
    }
}

/// C API functions for ABI management

/// Get the current ABI version
#[no_mangle]
pub extern "C" fn trustformers_get_abi_version(
    major: *mut c_uint,
    minor: *mut c_uint,
    patch: *mut c_uint,
) -> c_int {
    if major.is_null() || minor.is_null() || patch.is_null() {
        return crate::error::TrustformersError::NullPointer as c_int;
    }

    unsafe {
        *major = CURRENT_ABI_VERSION.major;
        *minor = CURRENT_ABI_VERSION.minor;
        *patch = CURRENT_ABI_VERSION.patch;
    }

    crate::error::TrustformersError::Success as c_int
}

/// Check version compatibility
#[no_mangle]
pub extern "C" fn trustformers_check_version_compatibility(
    client_major: c_uint,
    client_minor: c_uint,
    client_patch: c_uint,
) -> c_int {
    let client_version = AbiVersion::new(client_major, client_minor, client_patch);

    match negotiate_version(&client_version) {
        Ok(_) => 1,  // Compatible
        Err(_) => 0, // Not compatible
    }
}

/// Get version compatibility information
#[no_mangle]
pub extern "C" fn trustformers_get_version_info() -> *const c_char {
    let version_info = format!(
        "TrustformeRS C API v{} (ABI level {})",
        CURRENT_ABI_VERSION.to_string(),
        CURRENT_ABI_VERSION.abi_level
    );

    match CString::new(version_info) {
        Ok(c_string) => {
            // Note: This leaks memory. In production, you'd want to manage this differently
            let ptr = c_string.as_ptr();
            std::mem::forget(c_string);
            ptr
        },
        Err(_) => std::ptr::null(),
    }
}

/// Initialize the ABI system
#[no_mangle]
pub extern "C" fn trustformers_init_abi_system() -> c_int {
    init_deprecation_registry();
    crate::error::TrustformersError::Success as c_int
}

/// Check if a function is deprecated
#[no_mangle]
pub extern "C" fn trustformers_is_function_deprecated(function_name: *const c_char) -> c_int {
    if function_name.is_null() {
        return -1;
    }

    let name = unsafe {
        match CStr::from_ptr(function_name).to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        }
    };

    match check_deprecation(name) {
        Some("deprecated") => 1,
        Some("pending_removal") => 2,
        Some("removed") => 3,
        _ => 0,
    }
}

/// Example of a deprecated function
#[no_mangle]
pub extern "C" fn trustformers_old_api_function() -> c_int {
    deprecated_function!("trustformers_old_api_function", {
        // Old implementation
        crate::error::TrustformersError::Success as c_int
    })
}

/// Example of the new replacement function
#[no_mangle]
pub extern "C" fn trustformers_new_api_function() -> c_int {
    // New improved implementation
    crate::error::TrustformersError::Success as c_int
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abi_version() {
        let v1 = AbiVersion::new(1, 0, 0);
        let v2 = AbiVersion::new(1, 0, 1);
        let v3 = AbiVersion::new(1, 1, 0);

        assert!(!v1.is_newer_than(&v1));
        assert!(v2.is_newer_than(&v1));
        assert!(v3.is_newer_than(&v2));

        assert!(v1.is_compatible_with(&v2)); // Same ABI level
        assert!(!v1.is_compatible_with(&v3)); // Different ABI level
    }

    #[test]
    fn test_version_negotiation() {
        let current = CURRENT_ABI_VERSION.clone();
        let compatible = AbiVersion::new(current.major, current.minor, current.patch + 1);
        let incompatible = AbiVersion::new(current.major + 1, 0, 0);

        assert!(negotiate_version(&current).is_ok());
        assert!(negotiate_version(&compatible).is_ok());
        assert!(negotiate_version(&incompatible).is_err());
    }

    #[test]
    fn test_deprecation_registry() {
        init_deprecation_registry();

        let status = check_deprecation("trustformers_old_api_function");
        assert!(status.is_some());

        let status = check_deprecation("nonexistent_function");
        assert!(status.is_none());
    }

    #[test]
    fn test_c_api_version() {
        let mut major = 0u32;
        let mut minor = 0u32;
        let mut patch = 0u32;

        let result = trustformers_get_abi_version(
            &mut major as *mut c_uint,
            &mut minor as *mut c_uint,
            &mut patch as *mut c_uint,
        );

        assert_eq!(result, crate::error::TrustformersError::Success as c_int);
        assert_eq!(major, CURRENT_ABI_VERSION.major);
        assert_eq!(minor, CURRENT_ABI_VERSION.minor);
        assert_eq!(patch, CURRENT_ABI_VERSION.patch);
    }
}
