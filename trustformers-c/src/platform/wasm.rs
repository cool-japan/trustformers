//! WebAssembly-specific optimizations and utilities for TrustformeRS-C
//!
//! This module provides WebAssembly-specific implementations and optimizations
//! for running TrustformeRS in browser and Node.js environments.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int};
use wasm_bindgen::prelude::*;

/// WASM-specific optimization configuration
#[derive(Debug, Clone)]
pub struct WasmOptimizer {
    memory_limit_mb: usize,
    enable_simd: bool,
    enable_threads: bool,
    chunk_size: usize,
}

impl WasmOptimizer {
    pub fn new() -> Self {
        Self {
            memory_limit_mb: 512, // Conservative default for WASM
            enable_simd: Self::detect_simd_support(),
            enable_threads: Self::detect_thread_support(),
            chunk_size: 1024, // Process data in smaller chunks for WASM
        }
    }

    /// Configure WASM optimizer with specific parameters
    pub fn with_config(memory_limit_mb: usize, enable_simd: bool, enable_threads: bool) -> Self {
        Self {
            memory_limit_mb,
            enable_simd,
            enable_threads,
            chunk_size: if memory_limit_mb > 256 { 2048 } else { 512 },
        }
    }

    /// Detect SIMD support in WASM runtime
    fn detect_simd_support() -> bool {
        #[cfg(target_feature = "simd128")]
        {
            true
        }
        #[cfg(not(target_feature = "simd128"))]
        {
            false
        }
    }

    /// Detect thread support in WASM runtime
    fn detect_thread_support() -> bool {
        #[cfg(target_feature = "atomics")]
        {
            true
        }
        #[cfg(not(target_feature = "atomics"))]
        {
            false
        }
    }

    /// WASM-optimized matrix multiplication
    pub fn matrix_multiply(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        if self.enable_simd && cfg!(target_feature = "simd128") {
            self.simd_matrix_multiply(a, b, c, m, n, k);
        } else {
            self.chunked_matrix_multiply(a, b, c, m, n, k);
        }
    }

    #[cfg(target_feature = "simd128")]
    fn simd_matrix_multiply(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        use std::arch::wasm32::*;

        // SIMD-accelerated matrix multiplication for WASM
        for i in 0..m {
            for j in (0..n).step_by(4) {
                let mut sum = f32x4_splat(0.0);

                for ki in 0..k {
                    let a_val = f32x4_splat(a[i * k + ki]);

                    if j + 3 < n {
                        let b_vals = v128_load(&b[ki * n + j] as *const f32 as *const v128);
                        let b_f32x4 = f32x4(b_vals);
                        sum = f32x4_add(sum, f32x4_mul(a_val, b_f32x4));
                    } else {
                        // Handle remaining elements
                        for jj in j..n.min(j + 4) {
                            let idx = i * n + jj;
                            if idx < c.len() {
                                c[idx] += a[i * k + ki] * b[ki * n + jj];
                            }
                        }
                        break;
                    }
                }

                if j + 3 < n {
                    v128_store(&mut c[i * n + j] as *mut f32 as *mut v128, sum.0);
                }
            }
        }
    }

    #[cfg(not(target_feature = "simd128"))]
    fn simd_matrix_multiply(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        // Fallback to chunked multiplication
        self.chunked_matrix_multiply(a, b, c, m, n, k);
    }

    /// Memory-efficient chunked matrix multiplication for WASM
    fn chunked_matrix_multiply(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        let chunk_size = self.chunk_size;

        for i_chunk in (0..m).step_by(chunk_size) {
            let i_end = (i_chunk + chunk_size).min(m);

            for j_chunk in (0..n).step_by(chunk_size) {
                let j_end = (j_chunk + chunk_size).min(n);

                for k_chunk in (0..k).step_by(chunk_size) {
                    let k_end = (k_chunk + chunk_size).min(k);

                    // Process this chunk
                    for i in i_chunk..i_end {
                        for j in j_chunk..j_end {
                            let mut sum = 0.0;
                            for ki in k_chunk..k_end {
                                sum += a[i * k + ki] * b[ki * n + j];
                            }
                            c[i * n + j] += sum;
                        }
                    }
                }
            }
        }
    }

    /// WASM-optimized tensor operations
    pub fn tensor_add(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        if self.enable_simd && cfg!(target_feature = "simd128") {
            self.simd_tensor_add(a, b, result);
        } else {
            for ((a_val, b_val), result_val) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
                *result_val = a_val + b_val;
            }
        }
    }

    #[cfg(target_feature = "simd128")]
    fn simd_tensor_add(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        use std::arch::wasm32::*;

        let len = a.len().min(b.len()).min(result.len());
        let simd_len = len & !3; // Round down to multiple of 4

        // Process 4 elements at a time with SIMD
        for i in (0..simd_len).step_by(4) {
            let a_vec = v128_load(&a[i] as *const f32 as *const v128);
            let b_vec = v128_load(&b[i] as *const f32 as *const v128);
            let result_vec = f32x4_add(f32x4(a_vec), f32x4(b_vec));
            v128_store(&mut result[i] as *mut f32 as *mut v128, result_vec.0);
        }

        // Handle remaining elements
        for i in simd_len..len {
            result[i] = a[i] + b[i];
        }
    }

    /// Get memory usage statistics optimized for WASM
    pub fn get_memory_stats(&self) -> WasmMemoryStats {
        WasmMemoryStats {
            memory_limit_mb: self.memory_limit_mb,
            estimated_usage_mb: Self::estimate_memory_usage(),
            chunk_size: self.chunk_size,
            simd_enabled: self.enable_simd,
            threads_enabled: self.enable_threads,
        }
    }

    fn estimate_memory_usage() -> usize {
        // Simplified memory estimation for WASM
        // In a real implementation, this would query the WASM runtime
        64 // Default estimate
    }
}

/// WASM-specific memory statistics
#[derive(Debug, Clone)]
#[repr(C)]
pub struct WasmMemoryStats {
    pub memory_limit_mb: usize,
    pub estimated_usage_mb: usize,
    pub chunk_size: usize,
    pub simd_enabled: bool,
    pub threads_enabled: bool,
}

/// WASM-specific tokenizer optimization
pub struct WasmTokenizer {
    chunk_size: usize,
    use_streaming: bool,
}

impl WasmTokenizer {
    pub fn new() -> Self {
        Self {
            chunk_size: 512,
            use_streaming: true,
        }
    }

    /// Process text in chunks to avoid memory pressure in WASM
    pub fn tokenize_chunked(&self, text: &str, max_length: usize) -> Vec<u32> {
        let mut tokens = Vec::new();

        if text.len() <= self.chunk_size {
            // Process directly if small enough
            return self.tokenize_simple(text, max_length);
        }

        // Process in chunks
        let chars: Vec<char> = text.chars().collect();
        for chunk in chars.chunks(self.chunk_size) {
            let chunk_str: String = chunk.iter().collect();
            let mut chunk_tokens = self.tokenize_simple(&chunk_str, max_length);
            tokens.append(&mut chunk_tokens);

            if tokens.len() >= max_length {
                tokens.truncate(max_length);
                break;
            }
        }

        tokens
    }

    fn tokenize_simple(&self, text: &str, max_length: usize) -> Vec<u32> {
        // Simplified tokenization for demo
        // In a real implementation, this would use the actual tokenizer
        text.chars().take(max_length).map(|c| c as u32).collect()
    }
}

// C API exports for WASM-specific functionality

/// Initialize WASM optimizer with default settings
#[no_mangle]
pub extern "C" fn trustformers_wasm_init() -> *mut WasmOptimizer {
    Box::into_raw(Box::new(WasmOptimizer::new()))
}

/// Initialize WASM optimizer with custom configuration
#[no_mangle]
pub extern "C" fn trustformers_wasm_init_with_config(
    memory_limit_mb: usize,
    enable_simd: c_int,
    enable_threads: c_int,
) -> *mut WasmOptimizer {
    let optimizer =
        WasmOptimizer::with_config(memory_limit_mb, enable_simd != 0, enable_threads != 0);
    Box::into_raw(Box::new(optimizer))
}

/// Free WASM optimizer
#[no_mangle]
pub extern "C" fn trustformers_wasm_free(optimizer: *mut WasmOptimizer) {
    if !optimizer.is_null() {
        unsafe {
            let _ = Box::from_raw(optimizer);
        }
    }
}

/// Get WASM memory statistics
#[no_mangle]
pub extern "C" fn trustformers_wasm_get_memory_stats(
    optimizer: *const WasmOptimizer,
    stats: *mut WasmMemoryStats,
) -> c_int {
    if optimizer.is_null() || stats.is_null() {
        return -1;
    }

    unsafe {
        let opt = &*optimizer;
        let memory_stats = opt.get_memory_stats();
        *stats = memory_stats;
    }

    0
}

/// Check if SIMD is supported
#[no_mangle]
pub extern "C" fn trustformers_wasm_has_simd() -> c_int {
    if WasmOptimizer::detect_simd_support() {
        1
    } else {
        0
    }
}

/// Check if threads are supported
#[no_mangle]
pub extern "C" fn trustformers_wasm_has_threads() -> c_int {
    if WasmOptimizer::detect_thread_support() {
        1
    } else {
        0
    }
}

/// WASM-optimized matrix multiplication
#[no_mangle]
pub extern "C" fn trustformers_wasm_matrix_multiply(
    optimizer: *const WasmOptimizer,
    a: *const c_float,
    b: *const c_float,
    c: *mut c_float,
    m: usize,
    n: usize,
    k: usize,
) -> c_int {
    if optimizer.is_null() || a.is_null() || b.is_null() || c.is_null() {
        return -1;
    }

    unsafe {
        let opt = &*optimizer;
        let a_slice = std::slice::from_raw_parts(a, m * k);
        let b_slice = std::slice::from_raw_parts(b, k * n);
        let c_slice = std::slice::from_raw_parts_mut(c, m * n);

        opt.matrix_multiply(a_slice, b_slice, c_slice, m, n, k);
    }

    0
}

/// WASM-optimized tensor addition
#[no_mangle]
pub extern "C" fn trustformers_wasm_tensor_add(
    optimizer: *const WasmOptimizer,
    a: *const c_float,
    b: *const c_float,
    result: *mut c_float,
    len: usize,
) -> c_int {
    if optimizer.is_null() || a.is_null() || b.is_null() || result.is_null() {
        return -1;
    }

    unsafe {
        let opt = &*optimizer;
        let a_slice = std::slice::from_raw_parts(a, len);
        let b_slice = std::slice::from_raw_parts(b, len);
        let result_slice = std::slice::from_raw_parts_mut(result, len);

        opt.tensor_add(a_slice, b_slice, result_slice);
    }

    0
}

// WASM-bindgen exports for JavaScript integration

#[wasm_bindgen]
pub struct WasmTrustformers {
    optimizer: WasmOptimizer,
    tokenizer: WasmTokenizer,
}

#[wasm_bindgen]
impl WasmTrustformers {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmTrustformers {
        Self {
            optimizer: WasmOptimizer::new(),
            tokenizer: WasmTokenizer::new(),
        }
    }

    #[wasm_bindgen]
    pub fn get_version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    #[wasm_bindgen]
    pub fn has_simd(&self) -> bool {
        self.optimizer.enable_simd
    }

    #[wasm_bindgen]
    pub fn has_threads(&self) -> bool {
        self.optimizer.enable_threads
    }

    #[wasm_bindgen]
    pub fn tokenize(&self, text: &str, max_length: usize) -> Vec<u32> {
        self.tokenizer.tokenize_chunked(text, max_length)
    }

    #[wasm_bindgen]
    pub fn set_memory_limit(&mut self, limit_mb: usize) {
        self.optimizer.memory_limit_mb = limit_mb;
    }

    #[wasm_bindgen]
    pub fn get_memory_limit(&self) -> usize {
        self.optimizer.memory_limit_mb
    }

    #[wasm_bindgen]
    pub fn get_chunk_size(&self) -> usize {
        self.optimizer.chunk_size
    }

    #[wasm_bindgen]
    pub fn set_chunk_size(&mut self, size: usize) {
        self.optimizer.chunk_size = size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_optimizer_creation() {
        let optimizer = WasmOptimizer::new();
        assert_eq!(optimizer.memory_limit_mb, 512);
        assert_eq!(optimizer.chunk_size, 1024);
    }

    #[test]
    fn test_wasm_optimizer_with_config() {
        let optimizer = WasmOptimizer::with_config(256, false, false);
        assert_eq!(optimizer.memory_limit_mb, 256);
        assert!(!optimizer.enable_simd);
        assert!(!optimizer.enable_threads);
        assert_eq!(optimizer.chunk_size, 512);
    }

    #[test]
    fn test_chunked_matrix_multiply() {
        let optimizer = WasmOptimizer::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];

        optimizer.chunked_matrix_multiply(&a, &b, &mut c, 2, 2, 2);

        // Expected result for 2x2 matrix multiplication
        assert!((c[0] - 19.0).abs() < 1e-6);
        assert!((c[1] - 22.0).abs() < 1e-6);
        assert!((c[2] - 43.0).abs() < 1e-6);
        assert!((c[3] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_add() {
        let optimizer = WasmOptimizer::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut result = vec![0.0; 4];

        optimizer.tensor_add(&a, &b, &mut result);

        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_wasm_tokenizer() {
        let tokenizer = WasmTokenizer::new();
        let tokens = tokenizer.tokenize_chunked("hello", 10);

        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0], 'h' as u32);
        assert_eq!(tokens[4], 'o' as u32);
    }

    #[test]
    fn test_memory_stats() {
        let optimizer = WasmOptimizer::new();
        let stats = optimizer.get_memory_stats();

        assert_eq!(stats.memory_limit_mb, 512);
        assert_eq!(stats.chunk_size, 1024);
    }

    #[test]
    fn test_wasm_trustformers_js_interface() {
        let mut wasm_tf = WasmTrustformers::new();

        assert!(!WasmTrustformers::get_version().is_empty());
        assert_eq!(wasm_tf.get_memory_limit(), 512);

        wasm_tf.set_memory_limit(256);
        assert_eq!(wasm_tf.get_memory_limit(), 256);

        let tokens = wasm_tf.tokenize("test", 10);
        assert_eq!(tokens.len(), 4);
    }
}
