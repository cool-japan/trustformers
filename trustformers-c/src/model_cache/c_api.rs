//! C API functions for model cache

use std::os::raw::{c_char, c_float, c_int};

use super::types::*;
use super::MODEL_CACHE_MANAGER;
use crate::error::TrustformersError;
use crate::{c_str_to_string, string_to_c_str};

/// Configure model cache
#[no_mangle]
pub extern "C" fn trustformers_model_cache_configure(
    config_json: *const c_char,
) -> TrustformersError {
    if config_json.is_null() {
        return TrustformersError::NullPointer;
    }

    let config_str = match c_str_to_string(config_json) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let config: ModelCacheConfig = match serde_json::from_str(&config_str) {
        Ok(config) => config,
        Err(_) => return TrustformersError::SerializationError,
    };

    let mut manager = match MODEL_CACHE_MANAGER.write() {
        Ok(manager) => manager,
        Err(_) => return TrustformersError::RuntimeError,
    };

    match manager.configure(config) {
        Ok(_) => TrustformersError::Success,
        Err(_) => TrustformersError::RuntimeError,
    }
}

/// Get or load model from cache
#[no_mangle]
pub extern "C" fn trustformers_model_cache_get_or_load(
    model_id: *const c_char,
    model_path: *const c_char,
    config: *const c_char,
    model_handle: *mut usize,
) -> TrustformersError {
    if model_id.is_null() || model_path.is_null() || config.is_null() || model_handle.is_null() {
        return TrustformersError::NullPointer;
    }

    let id_str = match c_str_to_string(model_id) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let path_str = match c_str_to_string(model_path) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let config_str = match c_str_to_string(config) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let mut manager = match MODEL_CACHE_MANAGER.write() {
        Ok(manager) => manager,
        Err(_) => return TrustformersError::RuntimeError,
    };

    match manager.get_or_load_model(&id_str, &path_str, &config_str) {
        Ok(handle) => {
            unsafe {
                *model_handle = handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::RuntimeError,
    }
}

/// Get model cache statistics
#[no_mangle]
pub extern "C" fn trustformers_model_cache_get_stats(
    stats_json: *mut *mut c_char,
) -> TrustformersError {
    if stats_json.is_null() {
        return TrustformersError::NullPointer;
    }

    let manager = match MODEL_CACHE_MANAGER.read() {
        Ok(manager) => manager,
        Err(_) => return TrustformersError::RuntimeError,
    };

    let stats = manager.get_stats();
    let stats_json_str = match serde_json::to_string_pretty(&stats) {
        Ok(json) => json,
        Err(_) => return TrustformersError::SerializationError,
    };

    unsafe {
        *stats_json = string_to_c_str(stats_json_str);
    }

    TrustformersError::Success
}

/// Remove model from cache
#[no_mangle]
pub extern "C" fn trustformers_model_cache_remove(model_id: *const c_char) -> TrustformersError {
    if model_id.is_null() {
        return TrustformersError::NullPointer;
    }

    let id_str = match c_str_to_string(model_id) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let mut manager = match MODEL_CACHE_MANAGER.write() {
        Ok(manager) => manager,
        Err(_) => return TrustformersError::RuntimeError,
    };

    match manager.remove_model(&id_str) {
        Ok(_) => TrustformersError::Success,
        Err(_) => TrustformersError::InvalidParameter,
    }
}
