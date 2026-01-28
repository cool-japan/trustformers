//! Async C API for TrustformeRS
//!
//! This module provides asynchronous C API bindings for non-blocking operations.
//! Callbacks are used to notify when operations complete.

use crate::error::TrustformersError;
use crate::utils::string_to_c_str;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;
use std::sync::Arc;
use std::thread;

/// Async operation handle
pub type TrustformersAsyncHandle = usize;

/// Async operation status
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrustformersAsyncStatus {
    /// Operation is pending
    Pending = 0,
    /// Operation is running
    Running = 1,
    /// Operation completed successfully
    Completed = 2,
    /// Operation failed with error
    Failed = 3,
    /// Operation was cancelled
    Cancelled = 4,
}

/// Async operation callback function type
///
/// Called when an async operation completes.
/// - `handle`: The async operation handle
/// - `status`: The final status of the operation
/// - `result`: Operation-specific result data (can be null)
/// - `user_data`: User-provided data passed during operation creation
pub type TrustformersAsyncCallback = extern "C" fn(
    handle: TrustformersAsyncHandle,
    status: TrustformersAsyncStatus,
    result: *const c_void,
    user_data: *mut c_void,
);

/// Progress callback function type for async operations
///
/// Called periodically to report progress of long-running operations.
/// - `handle`: The async operation handle
/// - `progress`: Progress percentage (0.0 to 1.0)
/// - `message`: Optional progress message
/// - `user_data`: User-provided data
pub type TrustformersAsyncProgressCallback = extern "C" fn(
    handle: TrustformersAsyncHandle,
    progress: f32,
    message: *const c_char,
    user_data: *mut c_void,
);

/// Internal async operation state
struct AsyncOperation {
    handle: TrustformersAsyncHandle,
    status: TrustformersAsyncStatus,
    callback: Option<TrustformersAsyncCallback>,
    progress_callback: Option<TrustformersAsyncProgressCallback>,
    user_data: *mut c_void,
    error_message: Option<String>,
    thread_handle: Option<thread::JoinHandle<()>>,
}

impl AsyncOperation {
    fn new(
        handle: TrustformersAsyncHandle,
        callback: Option<TrustformersAsyncCallback>,
        progress_callback: Option<TrustformersAsyncProgressCallback>,
        user_data: *mut c_void,
    ) -> Self {
        Self {
            handle,
            status: TrustformersAsyncStatus::Pending,
            callback,
            progress_callback,
            user_data,
            error_message: None,
            thread_handle: None,
        }
    }

    fn set_status(&mut self, status: TrustformersAsyncStatus) {
        self.status = status;
    }

    fn set_error(&mut self, message: String) {
        self.status = TrustformersAsyncStatus::Failed;
        self.error_message = Some(message);
    }

    fn report_progress(&self, progress: f32, message: &str) {
        if let Some(progress_cb) = self.progress_callback {
            let msg_ptr = string_to_c_str(message.to_string());
            progress_cb(self.handle, progress, msg_ptr, self.user_data);
            if !msg_ptr.is_null() {
                unsafe {
                    let _ = CString::from_raw(msg_ptr as *mut c_char);
                }
            }
        }
    }

    fn complete(&self, result: *const c_void) {
        if let Some(callback) = self.callback {
            callback(self.handle, self.status, result, self.user_data);
        }
    }
}

// SAFETY: AsyncOperation is protected by RwLock and all pointer access is synchronized
unsafe impl Send for AsyncOperation {}
unsafe impl Sync for AsyncOperation {}

/// Global async operation registry
static ASYNC_REGISTRY: Lazy<RwLock<AsyncRegistry>> =
    Lazy::new(|| RwLock::new(AsyncRegistry::new()));

struct AsyncRegistry {
    operations: HashMap<TrustformersAsyncHandle, Arc<RwLock<AsyncOperation>>>,
    next_handle: TrustformersAsyncHandle,
}

impl AsyncRegistry {
    fn new() -> Self {
        Self {
            operations: HashMap::new(),
            next_handle: 1,
        }
    }

    fn create_operation(
        &mut self,
        callback: Option<TrustformersAsyncCallback>,
        progress_callback: Option<TrustformersAsyncProgressCallback>,
        user_data: *mut c_void,
    ) -> (TrustformersAsyncHandle, Arc<RwLock<AsyncOperation>>) {
        let handle = self.next_handle;
        self.next_handle += 1;

        let operation = Arc::new(RwLock::new(AsyncOperation::new(
            handle,
            callback,
            progress_callback,
            user_data,
        )));

        self.operations.insert(handle, operation.clone());
        (handle, operation)
    }

    fn get_operation(
        &self,
        handle: TrustformersAsyncHandle,
    ) -> Option<Arc<RwLock<AsyncOperation>>> {
        self.operations.get(&handle).cloned()
    }

    fn remove_operation(&mut self, handle: TrustformersAsyncHandle) -> bool {
        self.operations.remove(&handle).is_some()
    }

    fn cancel_all(&mut self) {
        for operation in self.operations.values() {
            let mut op = operation.write();
            if op.status == TrustformersAsyncStatus::Pending
                || op.status == TrustformersAsyncStatus::Running
            {
                op.status = TrustformersAsyncStatus::Cancelled;
            }
        }
    }
}

/// Create an async operation for model loading
///
/// # Parameters
/// - `model_path`: Path to the model to load
/// - `callback`: Completion callback (can be null)
/// - `progress_callback`: Progress callback (can be null)
/// - `user_data`: User data to pass to callbacks
/// - `handle`: Output parameter for async handle
///
/// # Returns
/// Error code indicating success or failure
#[no_mangle]
pub extern "C" fn trustformers_async_load_model(
    model_path: *const c_char,
    callback: TrustformersAsyncCallback,
    progress_callback: TrustformersAsyncProgressCallback,
    user_data: *mut c_void,
    handle: *mut TrustformersAsyncHandle,
) -> TrustformersError {
    if model_path.is_null() || handle.is_null() {
        return TrustformersError::NullPointer;
    }

    let path_str = match crate::utils::c_str_to_string(model_path) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let (async_handle, operation) =
        ASYNC_REGISTRY
            .write()
            .create_operation(Some(callback), Some(progress_callback), user_data);

    unsafe {
        *handle = async_handle;
    }

    // Spawn background thread for async loading
    let op_clone = operation.clone();
    let thread_handle = thread::spawn(move || {
        // Simulate async model loading
        {
            let mut op = op_clone.write();
            op.set_status(TrustformersAsyncStatus::Running);
        }

        // Report progress
        {
            let op = op_clone.read();
            op.report_progress(0.1, "Initializing model loader");
        }

        // Simulate work
        thread::sleep(std::time::Duration::from_millis(100));

        {
            let op = op_clone.read();
            op.report_progress(0.5, "Loading model weights");
        }

        thread::sleep(std::time::Duration::from_millis(100));

        {
            let op = op_clone.read();
            op.report_progress(0.9, "Finalizing model");
        }

        // Complete
        {
            let mut op = op_clone.write();
            op.set_status(TrustformersAsyncStatus::Completed);
        }

        {
            let op = op_clone.read();
            op.complete(ptr::null());
        }
    });

    // Store thread handle
    {
        let mut op = operation.write();
        op.thread_handle = Some(thread_handle);
    }

    TrustformersError::Success
}

/// Get the status of an async operation
#[no_mangle]
pub extern "C" fn trustformers_async_get_status(
    handle: TrustformersAsyncHandle,
) -> TrustformersAsyncStatus {
    let registry = ASYNC_REGISTRY.read();
    if let Some(operation) = registry.get_operation(handle) {
        let op = operation.read();
        op.status
    } else {
        TrustformersAsyncStatus::Failed
    }
}

/// Cancel an async operation
#[no_mangle]
pub extern "C" fn trustformers_async_cancel(handle: TrustformersAsyncHandle) -> TrustformersError {
    let registry = ASYNC_REGISTRY.read();
    if let Some(operation) = registry.get_operation(handle) {
        let mut op = operation.write();
        if op.status == TrustformersAsyncStatus::Pending
            || op.status == TrustformersAsyncStatus::Running
        {
            op.status = TrustformersAsyncStatus::Cancelled;
            TrustformersError::Success
        } else {
            TrustformersError::InvalidHandle
        }
    } else {
        TrustformersError::InvalidHandle
    }
}

/// Wait for an async operation to complete (blocking)
#[no_mangle]
pub extern "C" fn trustformers_async_wait(
    handle: TrustformersAsyncHandle,
    timeout_ms: c_int,
) -> TrustformersError {
    let start = std::time::Instant::now();
    let timeout = if timeout_ms < 0 {
        None
    } else {
        Some(std::time::Duration::from_millis(timeout_ms as u64))
    };

    loop {
        let status = trustformers_async_get_status(handle);
        match status {
            TrustformersAsyncStatus::Completed => return TrustformersError::Success,
            TrustformersAsyncStatus::Failed => return TrustformersError::RuntimeError,
            TrustformersAsyncStatus::Cancelled => return TrustformersError::RuntimeError,
            _ => {
                if let Some(timeout_duration) = timeout {
                    if start.elapsed() > timeout_duration {
                        return TrustformersError::Timeout;
                    }
                }
                thread::sleep(std::time::Duration::from_millis(10));
            },
        }
    }
}

/// Get error message for a failed async operation
#[no_mangle]
pub extern "C" fn trustformers_async_get_error(handle: TrustformersAsyncHandle) -> *const c_char {
    let registry = ASYNC_REGISTRY.read();
    if let Some(operation) = registry.get_operation(handle) {
        let op = operation.read();
        if let Some(ref error_msg) = op.error_message {
            return string_to_c_str(error_msg.clone());
        }
    }
    ptr::null()
}

/// Free an async operation handle
#[no_mangle]
pub extern "C" fn trustformers_async_free(handle: TrustformersAsyncHandle) -> TrustformersError {
    // Wait for operation to complete if still running
    let status = trustformers_async_get_status(handle);
    if status == TrustformersAsyncStatus::Running {
        // Try to wait with timeout
        let _ = trustformers_async_wait(handle, 5000);
    }

    let removed = ASYNC_REGISTRY.write().remove_operation(handle);
    if removed {
        TrustformersError::Success
    } else {
        TrustformersError::InvalidHandle
    }
}

/// Cancel all pending async operations
#[no_mangle]
pub extern "C" fn trustformers_async_cancel_all() -> TrustformersError {
    ASYNC_REGISTRY.write().cancel_all();
    TrustformersError::Success
}

/// Get the number of active async operations
#[no_mangle]
pub extern "C" fn trustformers_async_get_active_count() -> c_int {
    let registry = ASYNC_REGISTRY.read();
    registry.operations.len() as c_int
}

#[cfg(test)]
mod tests {
    use super::*;

    extern "C" fn test_callback(
        _handle: TrustformersAsyncHandle,
        _status: TrustformersAsyncStatus,
        _result: *const c_void,
        _user_data: *mut c_void,
    ) {
        // Test callback
    }

    extern "C" fn test_progress_callback(
        _handle: TrustformersAsyncHandle,
        _progress: f32,
        _message: *const c_char,
        _user_data: *mut c_void,
    ) {
        // Test progress callback
    }

    #[test]
    fn test_async_operation_lifecycle() {
        let model_path = CString::new("/path/to/model").unwrap();
        let mut handle: TrustformersAsyncHandle = 0;

        let err = trustformers_async_load_model(
            model_path.as_ptr(),
            test_callback,
            test_progress_callback,
            ptr::null_mut(),
            &mut handle,
        );

        assert_eq!(err, TrustformersError::Success);
        assert_ne!(handle, 0);

        // Wait for completion
        let err = trustformers_async_wait(handle, 5000);
        assert_eq!(err, TrustformersError::Success);

        // Check status
        let status = trustformers_async_get_status(handle);
        assert_eq!(status, TrustformersAsyncStatus::Completed);

        // Free
        let err = trustformers_async_free(handle);
        assert_eq!(err, TrustformersError::Success);
    }

    #[test]
    fn test_async_cancel() {
        let model_path = CString::new("/path/to/model").unwrap();
        let mut handle: TrustformersAsyncHandle = 0;

        trustformers_async_load_model(
            model_path.as_ptr(),
            test_callback,
            test_progress_callback,
            ptr::null_mut(),
            &mut handle,
        );

        // Cancel immediately
        let err = trustformers_async_cancel(handle);
        assert_eq!(err, TrustformersError::Success);

        // Check status
        let status = trustformers_async_get_status(handle);
        assert_eq!(status, TrustformersAsyncStatus::Cancelled);

        trustformers_async_free(handle);
    }
}
