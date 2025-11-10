//! Streaming inference C API for TrustformeRS
//!
//! This module provides streaming inference capabilities with callbacks
//! for progressive result generation, useful for text generation, translation, etc.

use crate::error::{TrustformersError, TrustformersResult};
use crate::utils::string_to_c_str;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::thread;

/// Stream handle type
pub type TrustformersStreamHandle = usize;

/// Token callback function type
///
/// Parameters:
/// - user_data: User-provided data pointer
/// - token: Generated token string
/// - token_id: Token ID
/// - probability: Token probability (0.0 to 1.0)
/// - is_final: Whether this is the final token
///
/// Returns: 0 to continue, non-zero to stop generation
pub type TrustformersTokenCallback = extern "C" fn(
    user_data: *mut c_void,
    token: *const c_char,
    token_id: i32,
    probability: f32,
    is_final: c_int,
) -> c_int;

/// Chunk callback function type for text chunks
pub type TrustformersChunkCallback = extern "C" fn(
    user_data: *mut c_void,
    chunk: *const c_char,
    chunk_len: usize,
    is_final: c_int,
) -> c_int;

/// Progress callback function type
///
/// Parameters:
/// - user_data: User-provided data pointer
/// - current: Current step
/// - total: Total steps
/// - elapsed_ms: Elapsed time in milliseconds
pub type TrustformersProgressCallback =
    extern "C" fn(user_data: *mut c_void, current: usize, total: usize, elapsed_ms: u64) -> c_int;

/// Streaming configuration
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TrustformersStreamingConfig {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling (0.0 to 2.0, default: 1.0)
    pub temperature: f32,
    /// Top-p sampling parameter (0.0 to 1.0)
    pub top_p: f32,
    /// Top-k sampling parameter
    pub top_k: usize,
    /// Repetition penalty (1.0 = no penalty)
    pub repetition_penalty: f32,
    /// Chunk size for streaming (tokens)
    pub chunk_size: usize,
    /// Enable token-level callbacks
    pub enable_token_callback: c_int,
    /// Enable chunk-level callbacks
    pub enable_chunk_callback: c_int,
    /// Enable progress callbacks
    pub enable_progress_callback: c_int,
    /// Token callback
    pub token_callback: Option<TrustformersTokenCallback>,
    /// Chunk callback
    pub chunk_callback: Option<TrustformersChunkCallback>,
    /// Progress callback
    pub progress_callback: Option<TrustformersProgressCallback>,
    /// User data pointer passed to callbacks
    pub user_data: *mut c_void,
}

impl Default for TrustformersStreamingConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 1.0,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.0,
            chunk_size: 10,
            enable_token_callback: 0,
            enable_chunk_callback: 0,
            enable_progress_callback: 0,
            token_callback: None,
            chunk_callback: None,
            progress_callback: None,
            user_data: ptr::null_mut(),
        }
    }
}

// SAFETY: TrustformersStreamingConfig contains callbacks and user_data pointer.
// The Send/Sync implementation is safe because:
// 1. Callbacks are stateless function pointers
// 2. user_data lifetime is managed by the caller
// 3. Access is synchronized through StreamRegistry's RwLock
unsafe impl Send for TrustformersStreamingConfig {}
unsafe impl Sync for TrustformersStreamingConfig {}

/// Streaming state
#[derive(Debug)]
pub struct StreamState {
    config: TrustformersStreamingConfig,
    is_active: Arc<AtomicBool>,
    tokens_generated: Arc<AtomicU64>,
    start_time: std::time::Instant,
    accumulated_text: String,
}

/// Global stream registry
static STREAM_REGISTRY: Lazy<RwLock<StreamRegistry>> =
    Lazy::new(|| RwLock::new(StreamRegistry::new()));

struct StreamRegistry {
    streams: HashMap<usize, Arc<RwLock<StreamState>>>,
    next_handle: usize,
}

impl StreamRegistry {
    fn new() -> Self {
        Self {
            streams: HashMap::new(),
            next_handle: 1,
        }
    }

    fn register(&mut self, state: StreamState) -> usize {
        let handle = self.next_handle;
        self.next_handle += 1;
        self.streams.insert(handle, Arc::new(RwLock::new(state)));
        handle
    }

    fn get(&self, handle: usize) -> Option<Arc<RwLock<StreamState>>> {
        self.streams.get(&handle).cloned()
    }

    fn remove(&mut self, handle: usize) -> bool {
        self.streams.remove(&handle).is_some()
    }
}

/// Create a streaming session
#[no_mangle]
pub extern "C" fn trustformers_streaming_create(
    config: *const TrustformersStreamingConfig,
    handle: *mut TrustformersStreamHandle,
) -> TrustformersError {
    if handle.is_null() {
        return TrustformersError::NullPointer;
    }

    let config = if config.is_null() {
        TrustformersStreamingConfig::default()
    } else {
        unsafe { std::ptr::read(config) }
    };

    let state = StreamState {
        config,
        is_active: Arc::new(AtomicBool::new(false)),
        tokens_generated: Arc::new(AtomicU64::new(0)),
        start_time: std::time::Instant::now(),
        accumulated_text: String::new(),
    };

    let stream_handle = STREAM_REGISTRY.write().register(state);

    unsafe {
        *handle = stream_handle;
    }

    TrustformersError::Success
}

/// Start streaming inference
#[no_mangle]
pub extern "C" fn trustformers_streaming_start(
    stream: TrustformersStreamHandle,
    model_handle: usize,
    prompt: *const c_char,
) -> TrustformersError {
    if prompt.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = STREAM_REGISTRY.read();
    let Some(state_arc) = registry.get(stream) else {
        return TrustformersError::InvalidHandle;
    };

    let prompt_str = unsafe {
        match CStr::from_ptr(prompt).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return TrustformersError::InvalidParameter,
        }
    };

    // Set stream as active
    state_arc.read().is_active.store(true, Ordering::SeqCst);

    // Clone necessary data for the thread
    let state_clone = state_arc.clone();

    // Spawn streaming thread
    thread::spawn(move || {
        simulate_streaming_generation(state_clone, model_handle, prompt_str);
    });

    TrustformersError::Success
}

/// Stop streaming inference
#[no_mangle]
pub extern "C" fn trustformers_streaming_stop(
    stream: TrustformersStreamHandle,
) -> TrustformersError {
    let registry = STREAM_REGISTRY.read();
    let Some(state_arc) = registry.get(stream) else {
        return TrustformersError::InvalidHandle;
    };

    state_arc.read().is_active.store(false, Ordering::SeqCst);

    TrustformersError::Success
}

/// Check if stream is active
#[no_mangle]
pub extern "C" fn trustformers_streaming_is_active(
    stream: TrustformersStreamHandle,
    is_active: *mut c_int,
) -> TrustformersError {
    if is_active.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = STREAM_REGISTRY.read();
    let Some(state_arc) = registry.get(stream) else {
        return TrustformersError::InvalidHandle;
    };

    let active = state_arc.read().is_active.load(Ordering::SeqCst);

    unsafe {
        *is_active = if active { 1 } else { 0 };
    }

    TrustformersError::Success
}

/// Get number of tokens generated so far
#[no_mangle]
pub extern "C" fn trustformers_streaming_get_token_count(
    stream: TrustformersStreamHandle,
    count: *mut u64,
) -> TrustformersError {
    if count.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = STREAM_REGISTRY.read();
    let Some(state_arc) = registry.get(stream) else {
        return TrustformersError::InvalidHandle;
    };

    let token_count = state_arc.read().tokens_generated.load(Ordering::SeqCst);

    unsafe {
        *count = token_count;
    }

    TrustformersError::Success
}

/// Get accumulated text so far
#[no_mangle]
pub extern "C" fn trustformers_streaming_get_text(
    stream: TrustformersStreamHandle,
    text: *mut *mut c_char,
) -> TrustformersError {
    if text.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = STREAM_REGISTRY.read();
    let Some(state_arc) = registry.get(stream) else {
        return TrustformersError::InvalidHandle;
    };

    let accumulated = state_arc.read().accumulated_text.clone();
    let c_string = string_to_c_str(accumulated);

    unsafe {
        *text = c_string;
    }

    TrustformersError::Success
}

/// Wait for streaming to complete
#[no_mangle]
pub extern "C" fn trustformers_streaming_wait(
    stream: TrustformersStreamHandle,
    timeout_ms: u64,
) -> TrustformersError {
    let registry = STREAM_REGISTRY.read();
    let Some(state_arc) = registry.get(stream) else {
        return TrustformersError::InvalidHandle;
    };

    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_millis(timeout_ms);

    while state_arc.read().is_active.load(Ordering::SeqCst) {
        if timeout_ms > 0 && start.elapsed() > timeout {
            return TrustformersError::Timeout;
        }

        thread::sleep(std::time::Duration::from_millis(10));
    }

    TrustformersError::Success
}

/// Free streaming session
#[no_mangle]
pub extern "C" fn trustformers_streaming_free(
    stream: TrustformersStreamHandle,
) -> TrustformersError {
    if stream == 0 {
        return TrustformersError::InvalidHandle;
    }

    // Stop streaming if active
    let _ = trustformers_streaming_stop(stream);

    let removed = STREAM_REGISTRY.write().remove(stream);

    if removed {
        TrustformersError::Success
    } else {
        TrustformersError::InvalidHandle
    }
}

/// Simulate streaming generation (placeholder implementation)
fn simulate_streaming_generation(
    state: Arc<RwLock<StreamState>>,
    _model_handle: usize,
    prompt: String,
) {
    // Clone config to avoid borrow conflicts
    let config = state.read().config.clone();
    let start_time = state.read().start_time;

    // Simulate token-by-token generation
    let sample_tokens = vec![
        "Hello",
        " world",
        "!",
        " This",
        " is",
        " a",
        " test",
        " of",
        " streaming",
        " generation",
        ".",
    ];

    for (i, token) in sample_tokens.iter().enumerate() {
        if !state.read().is_active.load(Ordering::SeqCst) {
            break;
        }

        let token_id = i as i32;
        let probability = 0.9 - (i as f32 * 0.05);
        let is_final = i == sample_tokens.len() - 1;

        // Update accumulated text
        state.write().accumulated_text.push_str(token);

        // Token callback
        if config.enable_token_callback != 0 {
            if let Some(callback) = config.token_callback {
                let token_cstr = CString::new(*token).unwrap();
                let should_stop = callback(
                    config.user_data,
                    token_cstr.as_ptr(),
                    token_id,
                    probability,
                    if is_final { 1 } else { 0 },
                );

                if should_stop != 0 {
                    break;
                }
            }
        }

        // Chunk callback (every chunk_size tokens)
        if config.enable_chunk_callback != 0 && (i + 1) % config.chunk_size == 0 {
            if let Some(callback) = config.chunk_callback {
                let chunk = state.read().accumulated_text.clone();
                let chunk_cstr = CString::new(chunk.clone()).unwrap();
                let should_stop = callback(
                    config.user_data,
                    chunk_cstr.as_ptr(),
                    chunk.len(),
                    if is_final { 1 } else { 0 },
                );

                if should_stop != 0 {
                    break;
                }
            }
        }

        // Progress callback
        if config.enable_progress_callback != 0 {
            if let Some(callback) = config.progress_callback {
                let elapsed = start_time.elapsed().as_millis() as u64;
                let should_stop = callback(config.user_data, i + 1, config.max_tokens, elapsed);

                if should_stop != 0 {
                    break;
                }
            }
        }

        state.read().tokens_generated.fetch_add(1, Ordering::SeqCst);

        // Simulate processing time
        thread::sleep(std::time::Duration::from_millis(50));

        if i >= config.max_tokens {
            break;
        }
    }

    // Mark as inactive
    state.read().is_active.store(false, Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_create() {
        let mut handle: TrustformersStreamHandle = 0;
        let err = trustformers_streaming_create(ptr::null(), &mut handle);

        assert_eq!(err, TrustformersError::Success);
        assert_ne!(handle, 0);

        // Free
        let err = trustformers_streaming_free(handle);
        assert_eq!(err, TrustformersError::Success);
    }

    #[test]
    fn test_streaming_lifecycle() {
        let mut handle: TrustformersStreamHandle = 0;
        trustformers_streaming_create(ptr::null(), &mut handle);

        // Check initial state
        let mut is_active: c_int = 0;
        trustformers_streaming_is_active(handle, &mut is_active);
        assert_eq!(is_active, 0);

        // Check token count
        let mut count: u64 = 0;
        trustformers_streaming_get_token_count(handle, &mut count);
        assert_eq!(count, 0);

        trustformers_streaming_free(handle);
    }
}
