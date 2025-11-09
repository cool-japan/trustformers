//! C API functions for debug utilities
//!
//! This module provides the C-compatible interface for the debug utilities.

use super::debug_manager::DebugUtilities;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};

// Global debug utilities instance
static mut DEBUG_UTILITIES: Option<DebugUtilities> = None;

/// Initialize debug utilities
#[no_mangle]
pub unsafe extern "C" fn trustformers_debug_init() -> c_int {
    DEBUG_UTILITIES = Some(DebugUtilities::new());
    0 // Success
}

/// Start a debug session
#[no_mangle]
pub unsafe extern "C" fn trustformers_debug_start_session(session_id: *const c_char) -> c_int {
    if session_id.is_null() {
        return 1; // Error
    }

    let session_id_str = match CStr::from_ptr(session_id).to_str() {
        Ok(s) => s,
        Err(_) => return 1,
    };

    if let Some(ref mut debug_utils) = DEBUG_UTILITIES {
        match debug_utils.start_debug_session(session_id_str) {
            Ok(_) => 0,
            Err(_) => 1,
        }
    } else {
        1
    }
}

/// Introspect a model
#[no_mangle]
pub unsafe extern "C" fn trustformers_debug_introspect_model(
    model_id: *const c_char,
    model_ptr: *const c_void,
    result_json: *mut *mut c_char,
) -> c_int {
    if model_id.is_null() || result_json.is_null() {
        return 1;
    }

    let model_id_str = match CStr::from_ptr(model_id).to_str() {
        Ok(s) => s,
        Err(_) => return 1,
    };

    if let Some(ref mut debug_utils) = DEBUG_UTILITIES {
        match debug_utils.introspect_model(model_id_str, model_ptr) {
            Ok(introspection) => match serde_json::to_string(&introspection) {
                Ok(json) => match CString::new(json) {
                    Ok(c_string) => {
                        *result_json = c_string.into_raw();
                        0
                    },
                    Err(_) => 1,
                },
                Err(_) => 1,
            },
            Err(_) => 1,
        }
    } else {
        1
    }
}

/// Generate model visualization
#[no_mangle]
pub unsafe extern "C" fn trustformers_debug_generate_visualization(
    model_id: *const c_char,
    result_json: *mut *mut c_char,
) -> c_int {
    if model_id.is_null() || result_json.is_null() {
        return 1;
    }

    let model_id_str = match CStr::from_ptr(model_id).to_str() {
        Ok(s) => s,
        Err(_) => return 1,
    };

    if let Some(ref debug_utils) = DEBUG_UTILITIES {
        match debug_utils.generate_model_visualization(model_id_str) {
            Ok(visualization) => match serde_json::to_string(&visualization) {
                Ok(json) => match CString::new(json) {
                    Ok(c_string) => {
                        *result_json = c_string.into_raw();
                        0
                    },
                    Err(_) => 1,
                },
                Err(_) => 1,
            },
            Err(_) => 1,
        }
    } else {
        1
    }
}

/// Generate performance report
#[no_mangle]
pub unsafe extern "C" fn trustformers_debug_generate_report(
    session_id: *const c_char,
    result: *mut *mut c_char,
) -> c_int {
    if session_id.is_null() || result.is_null() {
        return 1;
    }

    let session_id_str = match CStr::from_ptr(session_id).to_str() {
        Ok(s) => s,
        Err(_) => return 1,
    };

    if let Some(ref debug_utils) = DEBUG_UTILITIES {
        match debug_utils.generate_performance_report(session_id_str) {
            Ok(report) => match CString::new(report) {
                Ok(c_string) => {
                    *result = c_string.into_raw();
                    0
                },
                Err(_) => 1,
            },
            Err(_) => 1,
        }
    } else {
        1
    }
}

/// Free debug string
#[no_mangle]
pub unsafe extern "C" fn trustformers_debug_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}
