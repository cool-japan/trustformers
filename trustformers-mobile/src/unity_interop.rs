//! Unity C# interoperability layer for TrustformeRS mobile
//!
//! This module provides C-compatible FFI functions that can be called from Unity's C# scripts.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_void};
use std::ptr;
use std::sync::{Arc, Mutex};

use serde_json;
use trustformers_core::Tensor;
use trustformers_core::TrustformersError;

use crate::{
    device_info::MobileDeviceDetector, inference::MobileInferenceEngine, MobileConfig, MobileStats,
};

// Engine instance storage
use std::sync::OnceLock;

static ENGINE_STORAGE: OnceLock<Mutex<HashMap<usize, Arc<Mutex<MobileInferenceEngine>>>>> =
    OnceLock::new();
static NEXT_ENGINE_ID: OnceLock<Mutex<usize>> = OnceLock::new();

fn get_engine_storage() -> &'static Mutex<HashMap<usize, Arc<Mutex<MobileInferenceEngine>>>> {
    ENGINE_STORAGE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn get_next_engine_id() -> &'static Mutex<usize> {
    NEXT_ENGINE_ID.get_or_init(|| Mutex::new(1))
}

// Callback function types for IL2CPP compatibility
type LogCallback = extern "C" fn(*const c_char);
type ProgressCallback = extern "C" fn(c_float);
type ErrorCallback = extern "C" fn(c_int, *const c_char);
type InferenceCompleteCallback = extern "C" fn(*const c_float, c_int);

// Global callback storage
static mut LOG_CALLBACK: Option<LogCallback> = None;
static mut PROGRESS_CALLBACK: Option<ProgressCallback> = None;
static mut ERROR_CALLBACK: Option<ErrorCallback> = None;
static mut INFERENCE_CALLBACK: Option<InferenceCompleteCallback> = None;

// Helper macro for C string creation
macro_rules! c_string {
    ($s:expr) => {
        CString::new($s)
            .unwrap_or_else(|_| CString::new("Invalid string").unwrap_or_default())
            .into_raw()
    };
}

// Helper macro for error handling with callbacks
macro_rules! handle_error {
    ($result:expr, $default:expr) => {
        match $result {
            Ok(val) => val,
            Err(e) => {
                let error_msg = format!("TrustformeRS Error: {}", e);
                let c_msg = CString::new(error_msg).unwrap_or_default();
                unsafe {
                    if let Some(callback) = ERROR_CALLBACK {
                        callback(-1, c_msg.as_ptr());
                    }
                }
                return $default;
            },
        }
    };
}

/// Initialize IL2CPP support
#[no_mangle]
pub extern "C" fn trustformers_initialize_il2cpp_support() -> c_int {
    log::info("Initializing IL2CPP support for TrustformeRS");

    // Initialize logging (Unity-specific logging is handled via callbacks)
    // No additional initialization needed for Unity logging

    0 // Success
}

/// Cleanup IL2CPP support
#[no_mangle]
pub extern "C" fn trustformers_cleanup_il2cpp_support() {
    log::info("Cleaning up IL2CPP support");

    // Clear all engine instances
    if let Ok(mut storage) = get_engine_storage().lock() {
        storage.clear();
    }

    // Clear callbacks
    unsafe {
        LOG_CALLBACK = None;
        PROGRESS_CALLBACK = None;
        ERROR_CALLBACK = None;
        INFERENCE_CALLBACK = None;
    }
}

/// Set log callback for IL2CPP
#[no_mangle]
pub extern "C" fn trustformers_set_log_callback(callback: LogCallback) {
    unsafe {
        LOG_CALLBACK = Some(callback);
    }
}

/// Set progress callback for IL2CPP
#[no_mangle]
pub extern "C" fn trustformers_set_progress_callback(callback: ProgressCallback) {
    unsafe {
        PROGRESS_CALLBACK = Some(callback);
    }
}

/// Set error callback for IL2CPP
#[no_mangle]
pub extern "C" fn trustformers_set_error_callback(callback: ErrorCallback) {
    unsafe {
        ERROR_CALLBACK = Some(callback);
    }
}

/// Set inference complete callback for IL2CPP
#[no_mangle]
pub extern "C" fn trustformers_set_inference_callback(callback: InferenceCompleteCallback) {
    unsafe {
        INFERENCE_CALLBACK = Some(callback);
    }
}

/// Create a new TrustformeRS engine
#[no_mangle]
pub extern "C" fn trustformers_create_engine(config_json: *const c_char) -> *mut c_void {
    if config_json.is_null() {
        return ptr::null_mut();
    }

    let config_str = unsafe {
        match CStr::from_ptr(config_json).to_str() {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        }
    };

    let config: MobileConfig = handle_error!(
        serde_json::from_str(config_str).map_err(|e| {
            trustformers_core::error::CoreError::from(TrustformersError::config_error(
                &e.to_string(),
                "parse_config",
            ))
        }),
        ptr::null_mut()
    );

    tracing::info!("Creating TrustformeRS engine with config: {:?}", config);

    let engine = handle_error!(MobileInferenceEngine::new(config), ptr::null_mut());

    // Store engine and return ID as pointer
    let engine_arc = Arc::new(Mutex::new(engine));
    let mut storage = handle_error!(
        get_engine_storage()
            .lock()
            .map_err(|_| trustformers_core::error::CoreError::from(
                TrustformersError::runtime_error("Lock poisoned".into())
            )),
        ptr::null_mut()
    );
    let mut next_id = handle_error!(
        get_next_engine_id()
            .lock()
            .map_err(|_| trustformers_core::error::CoreError::from(
                TrustformersError::runtime_error("Lock poisoned".into())
            )),
        ptr::null_mut()
    );

    let engine_id = *next_id;
    *next_id += 1;

    storage.insert(engine_id, engine_arc);

    tracing::info!("Engine created with ID: {}", engine_id);
    engine_id as *mut c_void
}

/// Destroy a TrustformeRS engine
#[no_mangle]
pub extern "C" fn trustformers_destroy_engine(engine_ptr: *mut c_void) {
    if engine_ptr.is_null() {
        return;
    }

    let engine_id = engine_ptr as usize;

    if let Ok(mut storage) = get_engine_storage().lock() {
        if storage.remove(&engine_id).is_some() {
            log::info(&format!("Engine {} destroyed", engine_id));
        }
    }
}

/// Load a model into the engine
#[no_mangle]
pub extern "C" fn trustformers_load_model(
    engine_ptr: *mut c_void,
    model_path: *const c_char,
) -> c_int {
    if engine_ptr.is_null() || model_path.is_null() {
        return -1;
    }

    let engine_id = engine_ptr as usize;
    let path_str = unsafe {
        match CStr::from_ptr(model_path).to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        }
    };

    let storage = handle_error!(
        get_engine_storage()
            .lock()
            .map_err(|_| trustformers_core::error::CoreError::from(
                TrustformersError::runtime_error("Lock poisoned".into())
            )),
        -1
    );

    let engine_arc = match storage.get(&engine_id) {
        Some(arc) => arc.clone(),
        None => return -1,
    };

    drop(storage); // Release lock early

    let mut engine = handle_error!(
        engine_arc.lock().map_err(|_| trustformers_core::error::CoreError::from(
            TrustformersError::runtime_error("Lock poisoned".into())
        )),
        -1
    );

    tracing::info!("Loading model: {}", path_str);

    // Report progress
    unsafe {
        if let Some(callback) = PROGRESS_CALLBACK {
            callback(0.0);
        }
    }

    let result = engine.load_model_from_file(path_str);

    unsafe {
        if let Some(callback) = PROGRESS_CALLBACK {
            callback(1.0);
        }
    }

    match result {
        Ok(_) => {
            log::info(&format!("Model loaded successfully: {}", path_str));
            0
        },
        Err(e) => {
            let error_msg = format!("Failed to load model: {}", e);
            log::error(&error_msg);
            unsafe {
                if let Some(callback) = ERROR_CALLBACK {
                    let c_msg = CString::new(error_msg).unwrap_or_default();
                    callback(-2, c_msg.as_ptr());
                }
            }
            -2
        },
    }
}

/// Perform inference
#[no_mangle]
pub extern "C" fn trustformers_inference(
    engine_ptr: *mut c_void,
    input_data: *const c_float,
    input_length: c_int,
    output_data: *mut c_float,
    output_length: c_int,
) -> c_int {
    if engine_ptr.is_null() || input_data.is_null() || output_data.is_null() {
        return -1;
    }

    let engine_id = engine_ptr as usize;
    let input_slice = unsafe { std::slice::from_raw_parts(input_data, input_length as usize) };
    let output_slice =
        unsafe { std::slice::from_raw_parts_mut(output_data, output_length as usize) };

    let storage = handle_error!(
        get_engine_storage()
            .lock()
            .map_err(|_| trustformers_core::error::CoreError::from(
                TrustformersError::runtime_error("Lock poisoned".into())
            )),
        -1
    );

    let engine_arc = match storage.get(&engine_id) {
        Some(arc) => arc.clone(),
        None => return -1,
    };

    drop(storage); // Release lock early

    let mut engine = handle_error!(
        engine_arc.lock().map_err(|_| trustformers_core::error::CoreError::from(
            TrustformersError::runtime_error("Lock poisoned".into())
        )),
        -1
    );

    // Create input tensor
    let input_tensor = handle_error!(
        Tensor::from_slice(input_slice, &[input_length as usize]),
        -1
    );

    // Perform inference
    let result_tensor = handle_error!(engine.inference(&input_tensor), -1);

    // Copy results to output buffer
    let result_data = handle_error!(result_tensor.data(), -1);
    let copy_length = std::cmp::min(result_data.len(), output_slice.len());

    for i in 0..copy_length {
        output_slice[i] = result_data[i];
    }

    // Trigger async callback if available
    unsafe {
        if let Some(callback) = INFERENCE_CALLBACK {
            callback(output_data, copy_length as c_int);
        }
    }

    0 // Success
}

/// Perform batch inference
#[no_mangle]
pub extern "C" fn trustformers_batch_inference(
    engine_ptr: *mut c_void,
    input_data: *const c_float,
    batch_size: c_int,
    input_length: c_int,
    output_data: *mut c_float,
    output_length: c_int,
) -> c_int {
    if engine_ptr.is_null() || input_data.is_null() || output_data.is_null() {
        return -1;
    }

    let engine_id = engine_ptr as usize;
    let total_input_size = (batch_size * input_length) as usize;
    let total_output_size = (batch_size * output_length) as usize;

    let input_slice = unsafe { std::slice::from_raw_parts(input_data, total_input_size) };
    let output_slice = unsafe { std::slice::from_raw_parts_mut(output_data, total_output_size) };

    let storage = handle_error!(
        get_engine_storage()
            .lock()
            .map_err(|_| trustformers_core::error::CoreError::from(
                TrustformersError::runtime_error("Lock poisoned".into())
            )),
        -1
    );

    let engine_arc = match storage.get(&engine_id) {
        Some(arc) => arc.clone(),
        None => return -1,
    };

    drop(storage); // Release lock early

    let mut engine = handle_error!(
        engine_arc.lock().map_err(|_| trustformers_core::error::CoreError::from(
            TrustformersError::runtime_error("Lock poisoned".into())
        )),
        -1
    );

    // Create batch input tensors
    let mut input_tensors = Vec::new();
    for i in 0..batch_size as usize {
        let start_idx = i * input_length as usize;
        let end_idx = start_idx + input_length as usize;
        let batch_input = &input_slice[start_idx..end_idx];

        let tensor = handle_error!(
            Tensor::from_slice(batch_input, &[input_length as usize]),
            -1
        );
        input_tensors.push(tensor);
    }

    // Perform batch inference
    let result_tensors = handle_error!(engine.batch_inference(input_tensors), -1);

    // Copy results to output buffer
    for (i, result_tensor) in result_tensors.iter().enumerate() {
        let result_data = handle_error!(result_tensor.data(), -1);
        let start_idx = i * output_length as usize;
        let copy_length = std::cmp::min(result_data.len(), output_length as usize);

        for j in 0..copy_length {
            if start_idx + j < output_slice.len() {
                output_slice[start_idx + j] = result_data[j];
            }
        }
    }

    0 // Success
}

/// Get engine statistics
#[no_mangle]
pub extern "C" fn trustformers_get_stats(engine_ptr: *mut c_void, stats: *mut MobileStats) {
    if engine_ptr.is_null() || stats.is_null() {
        return;
    }

    let engine_id = engine_ptr as usize;

    let storage = match get_engine_storage().lock() {
        Ok(s) => s,
        Err(_) => return,
    };

    let engine_arc = match storage.get(&engine_id) {
        Some(arc) => arc.clone(),
        None => return,
    };

    drop(storage); // Release lock early

    let engine = match engine_arc.lock() {
        Ok(e) => e,
        Err(_) => return,
    };

    let engine_stats = engine.get_stats();

    unsafe {
        *stats = engine_stats.clone();
    }
}

/// Set performance mode
#[no_mangle]
pub extern "C" fn trustformers_set_performance_mode(engine_ptr: *mut c_void, mode: c_int) -> c_int {
    if engine_ptr.is_null() {
        return -1;
    }

    let engine_id = engine_ptr as usize;

    let storage = handle_error!(
        get_engine_storage()
            .lock()
            .map_err(|_| trustformers_core::error::CoreError::from(
                TrustformersError::runtime_error("Lock poisoned".into())
            )),
        -1
    );

    let engine_arc = match storage.get(&engine_id) {
        Some(arc) => arc.clone(),
        None => return -1,
    };

    drop(storage); // Release lock early

    let mut engine = handle_error!(
        engine_arc.lock().map_err(|_| trustformers_core::error::CoreError::from(
            TrustformersError::runtime_error("Lock poisoned".into())
        )),
        -1
    );

    // Set performance mode using the engine's method
    match engine.set_performance_mode(mode) {
        Ok(_) => {
            tracing::info!("Performance mode set to: {}", mode);
            0
        },
        Err(e) => {
            let error_msg = format!("Failed to set performance mode: {}", e);
            log::error(&error_msg);
            unsafe {
                if let Some(callback) = ERROR_CALLBACK {
                    let c_msg = CString::new(error_msg).unwrap_or_default();
                    callback(-3, c_msg.as_ptr());
                }
            }
            -3
        },
    }
}

/// Get device information
#[no_mangle]
pub extern "C" fn trustformers_get_device_info() -> *mut c_char {
    match MobileDeviceDetector::detect() {
        Ok(device_info) => {
            let info_json = match serde_json::to_string_pretty(&device_info) {
                Ok(json) => json,
                Err(e) => format!("Error serializing device info: {}", e),
            };
            c_string!(info_json)
        },
        Err(e) => {
            let error_msg = format!("Error getting device info: {}", e);
            c_string!(error_msg)
        },
    }
}

/// Free a string allocated by the native library
#[no_mangle]
pub extern "C" fn trustformers_free_string(str_ptr: *mut c_char) {
    if !str_ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(str_ptr);
        }
    }
}

/// Warm up the engine
#[no_mangle]
pub extern "C" fn trustformers_warm_up(engine_ptr: *mut c_void) -> c_int {
    if engine_ptr.is_null() {
        return -1;
    }

    let engine_id = engine_ptr as usize;

    let storage = handle_error!(
        get_engine_storage()
            .lock()
            .map_err(|_| trustformers_core::error::CoreError::from(
                TrustformersError::runtime_error("Lock poisoned".into())
            )),
        -1
    );

    let engine_arc = match storage.get(&engine_id) {
        Some(arc) => arc.clone(),
        None => return -1,
    };

    drop(storage); // Release lock early

    let mut engine = handle_error!(
        engine_arc.lock().map_err(|_| trustformers_core::error::CoreError::from(
            TrustformersError::runtime_error("Lock poisoned".into())
        )),
        -1
    );

    // Warm up the engine
    log::info("Starting engine warm-up");
    match engine.warm_up() {
        Ok(_) => {
            log::info("Engine warm-up completed successfully");
            0
        },
        Err(e) => {
            let error_msg = format!("Failed to warm up engine: {}", e);
            log::error(&error_msg);
            unsafe {
                if let Some(callback) = ERROR_CALLBACK {
                    let c_msg = CString::new(error_msg).unwrap_or_default();
                    callback(-3, c_msg.as_ptr());
                }
            }
            -3
        },
    }
}

// Memory management functions for IL2CPP compatibility
//
// `std::alloc::dealloc` requires the `Layout` passed to it to be *exactly*
// the one the matching `alloc` call used (same size, same alignment) -- the
// allocator is free to use that size/align pair to pick a size class,
// coalesce free blocks, etc., and giving it a different one back is
// undefined behaviour (heap corruption in a real allocator, not just a
// theoretical concern). This C ABI hands a bare pointer to Unity/IL2CPP and
// gets a bare pointer back on free with no size -- there is nowhere else to
// recover the original `Layout` from. The previous implementation
// deallocated with a hardcoded `Layout::new::<u8>()` (size 1, align 1)
// regardless of what was actually allocated: instant UB on every call.
//
// The fix is the standard C-ABI-safe pattern: prefix the returned block with
// a small fixed-size header holding the *exact* size and alignment that were
// requested, allocate `header + padding-for-alignment + requested bytes` as
// one block up front, and hand the caller a pointer just past the header
// (aligned to their requested alignment). `free` walks backward from that
// pointer to find the header, reconstructs the identical `Layout` the
// allocation used, and deallocates that -- which is provably correct because
// it is the same `Layout` value, not a size/align pair that merely happens
// to describe the same bytes.

/// Header stored immediately before every pointer this module hands to
/// Unity/IL2CPP. `repr(C)` so its own layout is stable and independent of
/// field-reordering optimizations; `align(16)` so the header itself never
/// under-aligns whatever payload alignment is requested next to it (16
/// covers every alignment this API accepts, since `size` is clamped to
/// `MAX_SUPPORTED_ALIGN` below).
#[repr(C, align(16))]
struct AllocationHeader {
    /// The `size` argument originally passed to
    /// [`trustformers_allocate_managed_memory`] (the *usable* payload size,
    /// not the header+padding+payload block size).
    size: usize,
    /// The alignment the payload was allocated with.
    align: usize,
}

/// Largest alignment this allocator will honour. `Layout::from_size_align`
/// requires the alignment to be a power of two and requires `size`, rounded
/// up to that alignment, not to overflow `isize::MAX`; capping the
/// alignment we ever request keeps the header-plus-payload arithmetic below
/// safely far from that ceiling for any `c_int`-bounded `size`.
const MAX_SUPPORTED_ALIGN: usize = 16;

/// Build the `Layout` for the full header+payload block backing a
/// `payload_size`-byte, `MAX_SUPPORTED_ALIGN`-aligned allocation.
///
/// The header is placed at the very start of an `align`-aligned block, so
/// the payload (which starts at `header_size` bytes in) is `align`-aligned
/// too precisely when `header_size` is itself a multiple of `align` --
/// which it is, since `AllocationHeader` is declared `align(16)` and
/// `align <= 16` here. Returns `None` if the requested size would overflow
/// the block-size computation.
fn block_layout(payload_size: usize) -> Option<(std::alloc::Layout, usize)> {
    let header_size = std::mem::size_of::<AllocationHeader>();
    let total = header_size.checked_add(payload_size)?;
    let layout = std::alloc::Layout::from_size_align(total, MAX_SUPPORTED_ALIGN).ok()?;
    Some((layout, header_size))
}

/// Allocate managed memory. Returns a pointer to `size` usable bytes,
/// aligned to `MAX_SUPPORTED_ALIGN`, or null on failure (zero/negative
/// size, arithmetic overflow, or allocator failure).
#[no_mangle]
pub extern "C" fn trustformers_allocate_managed_memory(size: c_int) -> *mut c_void {
    if size <= 0 {
        return ptr::null_mut();
    }
    let payload_size = size as usize;

    let Some((layout, header_size)) = block_layout(payload_size) else {
        return ptr::null_mut();
    };

    // SAFETY: `layout` has non-zero size (it includes the non-zero-sized
    // header even if `payload_size` were 0, though that path is excluded
    // above since `size <= 0` returns early).
    let block = unsafe { std::alloc::alloc(layout) };
    if block.is_null() {
        return ptr::null_mut();
    }

    // SAFETY: `block` is a fresh allocation of at least
    // `header_size + payload_size` bytes, sized and aligned by `layout`,
    // which reserves room for one `AllocationHeader` at offset 0 (`layout`
    // was built from `header_size + payload_size`, and `AllocationHeader`'s
    // own alignment requirement, 16, does not exceed the block's alignment,
    // also `MAX_SUPPORTED_ALIGN` = 16, so offset 0 is a valid, correctly
    // aligned place to write one).
    unsafe {
        let header = block as *mut AllocationHeader;
        header.write(AllocationHeader {
            size: payload_size,
            align: MAX_SUPPORTED_ALIGN,
        });
    }

    // SAFETY: `header_size <= layout.size()`, so this stays within the
    // allocated block.
    let payload = unsafe { block.add(header_size) };
    payload as *mut c_void
}

/// Free memory obtained from [`trustformers_allocate_managed_memory`]. A
/// null pointer is a documented no-op (matches `free(3)`/`std::alloc`
/// convention); any other pointer must be one this module itself returned
/// and not already freed -- passing a foreign or double-freed pointer is
/// undefined behaviour in the same way it would be for `free()`, and no FFI
/// boundary can check for that from the callee side.
#[no_mangle]
pub extern "C" fn trustformers_free_managed_memory(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }

    let header_size = std::mem::size_of::<AllocationHeader>();

    // SAFETY: by this function's contract, `ptr` is the payload pointer
    // returned by a prior `trustformers_allocate_managed_memory` call,
    // which places its `AllocationHeader` exactly `header_size` bytes
    // before that pointer (see the SAFETY comment there).
    let block = unsafe { (ptr as *mut u8).sub(header_size) };

    // SAFETY: `block` points to a live `AllocationHeader` written by the
    // matching allocate call and not yet freed (single-free contract above).
    let header = unsafe { &*(block as *const AllocationHeader) };
    let payload_size = header.size;
    let align = header.align;

    // Reconstruct the *exact* Layout the allocation used: same total size
    // (header + payload) and same alignment. `block_layout` is the single
    // source of truth for that computation, shared with the allocate side,
    // so this cannot drift from what was actually allocated.
    let Some((layout, _)) = block_layout(payload_size) else {
        // Unreachable in practice (the same computation succeeded when this
        // block was allocated), but never guess a Layout to deallocate with.
        debug_assert!(
            false,
            "trustformers_free_managed_memory: could not reconstruct the original Layout \
             (size={payload_size}, align={align}); leaking rather than risking a mismatched \
             dealloc"
        );
        return;
    };
    debug_assert_eq!(
        layout.align(),
        align,
        "stored alignment must match MAX_SUPPORTED_ALIGN"
    );

    // SAFETY: `block` was allocated by `std::alloc::alloc` with exactly
    // this `Layout` (reconstructed via the same `block_layout` function
    // from the header's own recorded `size`), and this is the first and
    // only time it is freed (per this function's single-free contract).
    unsafe {
        std::alloc::dealloc(block, layout);
    }
}

/// Copy data to managed memory
#[no_mangle]
pub extern "C" fn trustformers_copy_to_managed_memory(
    dest: *mut c_void,
    source: *const c_float,
    length: c_int,
) {
    if dest.is_null() || source.is_null() || length <= 0 {
        return;
    }

    let src_slice = unsafe { std::slice::from_raw_parts(source, length as usize) };

    let dest_slice =
        unsafe { std::slice::from_raw_parts_mut(dest as *mut c_float, length as usize) };

    dest_slice.copy_from_slice(src_slice);
}

/// Copy data from managed memory
#[no_mangle]
pub extern "C" fn trustformers_copy_from_managed_memory(
    dest: *mut c_float,
    source: *const c_void,
    length: c_int,
) {
    if dest.is_null() || source.is_null() || length <= 0 {
        return;
    }

    let src_slice =
        unsafe { std::slice::from_raw_parts(source as *const c_float, length as usize) };

    let dest_slice = unsafe { std::slice::from_raw_parts_mut(dest, length as usize) };

    dest_slice.copy_from_slice(src_slice);
}

// Utility functions for logging
mod log {
    use super::*;

    pub fn info(message: &str) {
        let c_msg = CString::new(message).unwrap_or_default();
        unsafe {
            if let Some(callback) = LOG_CALLBACK {
                callback(c_msg.as_ptr());
            }
        }
        println!("[INFO] {}", message);
    }

    pub fn error(message: &str) {
        let c_msg = CString::new(message).unwrap_or_default();
        unsafe {
            if let Some(callback) = ERROR_CALLBACK {
                callback(-1, c_msg.as_ptr());
            }
        }
        eprintln!("[ERROR] {}", message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation_and_destruction() {
        let config = MobileConfig::default();
        let config_json = serde_json::to_string(&config).expect("Operation failed");
        let c_config = CString::new(config_json).expect("Operation failed");

        let engine_ptr = trustformers_create_engine(c_config.as_ptr());
        assert!(!engine_ptr.is_null());

        trustformers_destroy_engine(engine_ptr);
    }

    #[test]
    fn test_memory_management() {
        let size = 1024;
        let ptr = trustformers_allocate_managed_memory(size);
        assert!(!ptr.is_null());

        trustformers_free_managed_memory(ptr);
    }

    /// Regression test for the alloc/dealloc `Layout` mismatch: allocates a
    /// range of sizes (including ones deliberately *not* a multiple of the
    /// allocator's natural alignment, so the header/payload arithmetic is
    /// exercised for real), writes a byte pattern across the *entire*
    /// usable payload of each, reads it back, and frees it.
    ///
    /// Against the old implementation (`dealloc` with a hardcoded
    /// `Layout::new::<u8>()`, i.e. size 1 / align 1, instead of the Layout
    /// actually used to `alloc`), this exact sequence is undefined
    /// behaviour per `std::alloc`'s contract -- a real allocator is free to
    /// corrupt its own bookkeeping in response, and under Miri (run via
    /// `cargo +nightly miri test -p trustformers-mobile --lib
    /// unity_interop::tests::test_memory_management_round_trip_and_miri_safety`)
    /// it is caught deterministically as "deallocating with a Layout that
    /// does not match the original allocation". This test is the harness
    /// that would have failed against that code; it passes now because
    /// `trustformers_free_managed_memory` reconstructs the exact original
    /// `Layout` from a stored header instead of guessing one.
    #[test]
    fn test_memory_management_round_trip_and_miri_safety() {
        // Sizes chosen to span "smaller than the header", "not a multiple
        // of the allocator's alignment", and "much larger than the header",
        // so the header/payload split is exercised at its edges, not just
        // in the common case.
        for &size in &[1usize, 3, 8, 15, 16, 17, 256, 1024, 4095] {
            let ptr = trustformers_allocate_managed_memory(size as c_int);
            assert!(!ptr.is_null(), "allocation of {size} bytes must succeed");

            // Every returned pointer must be usable for the full requested
            // size: write a distinct pattern across all of it (this would
            // corrupt the allocator's own header if the header/payload
            // offset arithmetic under- or over-shot) and read it back.
            let bytes = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, size) };
            for (i, b) in bytes.iter_mut().enumerate() {
                *b = (i % 256) as u8;
            }
            let bytes_ro = unsafe { std::slice::from_raw_parts(ptr as *const u8, size) };
            for (i, &b) in bytes_ro.iter().enumerate() {
                assert_eq!(
                    b,
                    (i % 256) as u8,
                    "byte {i} of a {size}-byte allocation was corrupted"
                );
            }

            trustformers_free_managed_memory(ptr);
        }
    }

    /// Invalid sizes (zero, negative) must return null rather than
    /// attempting an allocation that could underflow the header+payload
    /// size computation.
    #[test]
    fn test_memory_management_rejects_invalid_sizes() {
        assert!(trustformers_allocate_managed_memory(0).is_null());
        assert!(trustformers_allocate_managed_memory(-1).is_null());
        assert!(trustformers_allocate_managed_memory(i32::MIN).is_null());
    }

    /// Freeing a null pointer must be a documented no-op, matching
    /// `free(3)` convention -- callers should not have to special-case it.
    #[test]
    fn test_memory_management_free_null_is_noop() {
        trustformers_free_managed_memory(ptr::null_mut());
    }

    #[test]
    fn test_device_info() {
        let info_ptr = trustformers_get_device_info();
        assert!(!info_ptr.is_null());

        trustformers_free_string(info_ptr);
    }
}
