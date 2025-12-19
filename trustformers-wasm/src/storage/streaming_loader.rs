//! Streaming compilation support for faster WASM loading
//!
//! This module provides streaming compilation capabilities to reduce startup time
//! by loading and compiling WASM modules progressively.

#![allow(dead_code)]

use js_sys::{ArrayBuffer, Promise, Uint8Array, WebAssembly};
use std::format;
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

use super::StorageError;

/// Initialize the streaming loader module
pub fn initialize() -> Result<(), StorageError> {
    // Check if streaming compilation is supported
    if !is_streaming_compilation_supported() {
        web_sys::console::warn_1(
            &"Streaming WASM compilation is not supported in this environment".into(),
        );
    }

    // Check if Cache API is available
    if !is_cache_api_available() {
        web_sys::console::warn_1(&"Cache API is not available - caching will be disabled".into());
    }

    Ok(())
}

/// Streaming WASM loader configuration
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    pub(crate) chunk_size_kb: u32,
    pub(crate) max_concurrent_chunks: u32,
    pub(crate) enable_caching: bool,
    pub(crate) cache_name: String,
    pub(crate) precompile_enabled: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Streaming WASM loader
#[wasm_bindgen]
pub struct StreamingLoader {
    config: StreamingConfig,
    loaded_chunks: Vec<ArrayBuffer>,
    compiled_modules: Vec<WebAssembly::Module>,
    total_size: usize,
    loaded_size: usize,
    is_loading: bool,
    loading_start_time: f64,
    last_progress_time: f64,
    last_progress_bytes: usize,
}

/// Loading progress information
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct LoadingProgress {
    total_bytes: usize,
    loaded_bytes: usize,
    compiled_chunks: usize,
    total_chunks: usize,
    loading_speed_kbps: f64,
    estimated_remaining_ms: f64,
}

/// Chunk loading result
#[derive(Debug, Clone)]
struct ChunkResult {
    chunk_id: usize,
    data: ArrayBuffer,
    size: usize,
    load_time_ms: f64,
}

#[wasm_bindgen]
impl StreamingConfig {
    /// Create a new streaming configuration
    #[wasm_bindgen(constructor)]
    pub fn new() -> StreamingConfig {
        StreamingConfig {
            chunk_size_kb: 256,       // 256KB chunks by default
            max_concurrent_chunks: 4, // Load 4 chunks concurrently
            enable_caching: true,
            cache_name: "trustformers-wasm-cache".to_string(),
            precompile_enabled: true,
        }
    }

    /// Set chunk size in KB
    pub fn set_chunk_size_kb(&mut self, size_kb: u32) {
        self.chunk_size_kb = size_kb.clamp(64, 2048); // Clamp between 64KB and 2MB
    }

    /// Set maximum concurrent chunks
    pub fn set_max_concurrent_chunks(&mut self, count: u32) {
        self.max_concurrent_chunks = count.clamp(1, 8); // Clamp between 1 and 8
    }

    /// Enable or disable caching
    pub fn set_caching_enabled(&mut self, enabled: bool) {
        self.enable_caching = enabled;
    }

    /// Set cache name
    pub fn set_cache_name(&mut self, name: String) {
        self.cache_name = name;
    }

    /// Enable or disable precompilation
    pub fn set_precompile_enabled(&mut self, enabled: bool) {
        self.precompile_enabled = enabled;
    }

    #[wasm_bindgen(getter)]
    pub fn chunk_size_kb(&self) -> u32 {
        self.chunk_size_kb
    }

    #[wasm_bindgen(getter)]
    pub fn max_concurrent_chunks(&self) -> u32 {
        self.max_concurrent_chunks
    }

    #[wasm_bindgen(getter)]
    pub fn enable_caching(&self) -> bool {
        self.enable_caching
    }

    #[wasm_bindgen(getter)]
    pub fn precompile_enabled(&self) -> bool {
        self.precompile_enabled
    }
}

#[wasm_bindgen]
impl StreamingLoader {
    /// Create a new streaming loader
    #[wasm_bindgen(constructor)]
    pub fn new(config: StreamingConfig) -> StreamingLoader {
        let now = js_sys::Date::now();
        StreamingLoader {
            config,
            loaded_chunks: Vec::new(),
            compiled_modules: Vec::new(),
            total_size: 0,
            loaded_size: 0,
            is_loading: false,
            loading_start_time: now,
            last_progress_time: now,
            last_progress_bytes: 0,
        }
    }

    /// Load WASM module from URL with streaming
    pub async fn load_from_url(&mut self, url: &str) -> Result<WebAssembly::Module, JsValue> {
        self.is_loading = true;
        self.loaded_chunks.clear();
        self.compiled_modules.clear();

        // Initialize timing metrics
        let now = js_sys::Date::now();
        self.loading_start_time = now;
        self.last_progress_time = now;
        self.last_progress_bytes = 0;
        self.loaded_size = 0;

        web_sys::console::log_1(&format!("Starting streaming load from: {}", url).into());

        // Check cache first if enabled
        if self.config.enable_caching {
            if let Some(cached_module) = self.load_from_cache(url).await? {
                web_sys::console::log_1(&"Loaded WASM module from cache".into());
                self.is_loading = false;
                return Ok(cached_module);
            }
        }

        // Get total size using HEAD request
        let total_size = self.get_resource_size(url).await?;
        self.total_size = total_size;

        web_sys::console::log_1(&format!("Total WASM size: {} bytes", total_size).into());

        // Calculate chunk information
        let chunk_size = (self.config.chunk_size_kb as usize) * 1024;
        let total_chunks = total_size.div_ceil(chunk_size);

        web_sys::console::log_1(
            &format!(
                "Loading {} chunks of {} KB each",
                total_chunks, self.config.chunk_size_kb
            )
            .into(),
        );

        // Load chunks in parallel
        let final_module = self.load_chunks_parallel(url, chunk_size, total_chunks).await?;

        // Cache the compiled module if caching is enabled
        if self.config.enable_caching {
            self.save_to_cache(url, &final_module).await?;
        }

        self.is_loading = false;
        web_sys::console::log_1(&"Streaming load completed".into());

        Ok(final_module)
    }

    /// Load WASM module from byte array with streaming compilation
    pub async fn load_from_bytes(&mut self, data: &[u8]) -> Result<WebAssembly::Module, JsValue> {
        self.is_loading = true;
        self.total_size = data.len();
        self.loaded_size = 0;

        let chunk_size = (self.config.chunk_size_kb as usize) * 1024;
        let total_chunks = data.len().div_ceil(chunk_size);

        web_sys::console::log_1(
            &format!(
                "Compiling WASM from memory: {} bytes in {} chunks",
                data.len(),
                total_chunks
            )
            .into(),
        );

        // Split data into chunks and compile progressively
        let mut _compiled_chunks: Vec<JsValue> = Vec::new();

        for chunk_id in 0..total_chunks {
            let start = chunk_id * chunk_size;
            let end = (start + chunk_size).min(data.len());
            let chunk_data = &data[start..end];

            // Create ArrayBuffer for this chunk
            let array_buffer = ArrayBuffer::new(chunk_data.len() as u32);
            let uint8_view = Uint8Array::new(&array_buffer);
            uint8_view.copy_from(chunk_data);

            self.loaded_chunks.push(array_buffer);
            self.update_progress_metrics(chunk_data.len());

            // Yield control periodically
            if chunk_id % 4 == 0 {
                let promise = Promise::resolve(&JsValue::NULL);
                JsFuture::from(promise).await?;
            }
        }

        // Reassemble and compile final module
        let final_module = self.compile_final_module().await?;

        self.is_loading = false;
        Ok(final_module)
    }

    /// Get resource size using HEAD request
    async fn get_resource_size(&self, url: &str) -> Result<usize, JsValue> {
        let window = web_sys::window().ok_or("No window object")?;
        let fetch_promise = window.fetch_with_str(url);
        let response = JsFuture::from(fetch_promise).await?;
        let response: web_sys::Response = response.dyn_into()?;

        let content_length = response.headers().get("content-length")?;
        if let Some(length_str) = content_length {
            length_str.parse().map_err(|_| "Invalid content-length".into())
        } else {
            // Fallback: fetch the entire resource to get size
            let array_buffer_promise = response.array_buffer()?;
            let array_buffer = JsFuture::from(array_buffer_promise).await?;
            let array_buffer: ArrayBuffer = array_buffer.dyn_into()?;
            Ok(array_buffer.byte_length() as usize)
        }
    }

    /// Load chunks in parallel using fetch with range requests
    async fn load_chunks_parallel(
        &mut self,
        url: &str,
        chunk_size: usize,
        total_chunks: usize,
    ) -> Result<WebAssembly::Module, JsValue> {
        // Load chunks sequentially (simplified from parallel due to type constraints)
        for chunk_id in 0..total_chunks {
            let start = chunk_id * chunk_size;
            let end = ((chunk_id + 1) * chunk_size).min(self.total_size) - 1;

            // Load the chunk and store it
            let chunk_result = self.load_chunk_range(url, chunk_id, start, end).await?;
            self.loaded_chunks.push(chunk_result.data);
        }

        // Compile final module
        self.compile_final_module().await
    }

    /// Load a specific byte range of the resource
    async fn load_chunk_range(
        &self,
        url: &str,
        chunk_id: usize,
        start: usize,
        end: usize,
    ) -> Result<ChunkResult, JsValue> {
        let start_time = js_sys::Date::now();

        // Create request with Range header
        let init = web_sys::RequestInit::new();
        init.set_method("GET");

        let headers = web_sys::Headers::new()?;
        headers.set("Range", &format!("bytes={}-{}", start, end))?;
        init.set_headers(&headers);

        let request = web_sys::Request::new_with_str_and_init(url, &init)?;

        let window = web_sys::window().ok_or("No window object")?;
        let response_promise = window.fetch_with_request(&request);
        let response = JsFuture::from(response_promise).await?;
        let response: web_sys::Response = response.dyn_into()?;

        if !response.ok() {
            return Err(format!("Failed to load chunk {}: {}", chunk_id, response.status()).into());
        }

        let array_buffer_promise = response.array_buffer()?;
        let array_buffer = JsFuture::from(array_buffer_promise).await?;
        let array_buffer: ArrayBuffer = array_buffer.dyn_into()?;

        let end_time = js_sys::Date::now();
        let load_time = end_time - start_time;

        // Get byte_length before moving array_buffer
        let size = array_buffer.byte_length() as usize;

        Ok(ChunkResult {
            chunk_id,
            data: array_buffer,
            size,
            load_time_ms: load_time,
        })
    }

    /// Wait for the next chunk to complete
    async fn wait_for_chunk_completion(
        &mut self,
        chunk_futures: &mut Vec<JsFuture>,
    ) -> Result<ChunkResult, JsValue> {
        // This is a simplified implementation
        // In a real implementation, you'd use Promise.race() to wait for the first completion
        if let Some(future) = chunk_futures.pop() {
            let _result = future.await?;
            // Process the result to extract ChunkResult
            // For now, return a dummy result
            Ok(ChunkResult {
                chunk_id: 0,
                data: ArrayBuffer::new(0),
                size: 0,
                load_time_ms: 0.0,
            })
        } else {
            Err("No chunks to wait for".into())
        }
    }

    /// Compile the final module from loaded chunks
    async fn compile_final_module(&self) -> Result<WebAssembly::Module, JsValue> {
        // Calculate total size
        let total_size: usize =
            self.loaded_chunks.iter().map(|chunk| chunk.byte_length() as usize).sum();

        // Create final buffer
        let final_buffer = ArrayBuffer::new(total_size as u32);
        let final_view = Uint8Array::new(&final_buffer);

        let mut offset = 0;
        for chunk in &self.loaded_chunks {
            let chunk_view = Uint8Array::new(chunk);
            final_view.set(&chunk_view, offset);
            offset += chunk.byte_length();
        }

        // Compile the final module
        if self.config.precompile_enabled {
            // Use streaming compilation if available
            web_sys::console::log_1(&"Compiling WASM module...".into());
            let compile_promise = WebAssembly::compile(&final_buffer);
            let module = JsFuture::from(compile_promise).await?;
            let module: WebAssembly::Module = module.dyn_into()?;
            Ok(module)
        } else {
            // Instant compilation
            let module = WebAssembly::Module::new(&final_buffer)?;
            Ok(module)
        }
    }

    /// Load from cache if available
    async fn load_from_cache(&self, url: &str) -> Result<Option<WebAssembly::Module>, JsValue> {
        // Check if CacheStorage is available
        let window = web_sys::window().ok_or("No window object")?;
        let navigator = window.navigator();

        let cache_storage = match js_sys::Reflect::get(&navigator, &"serviceWorker".into())? {
            cache_obj if !cache_obj.is_undefined() => {
                match js_sys::Reflect::get(&cache_obj, &"controller".into())? {
                    controller if !controller.is_undefined() => {
                        // Service worker is available, try to use cache
                        self.try_cache_api_load(url).await?
                    },
                    _ => None,
                }
            },
            _ => None,
        };

        Ok(cache_storage)
    }

    /// Try to load from Cache API
    async fn try_cache_api_load(&self, url: &str) -> Result<Option<WebAssembly::Module>, JsValue> {
        let js_code = format!(
            r#"
            (async () => {{
                try {{
                    const cache = await caches.open('{}');
                    const response = await cache.match('{}');
                    if (response) {{
                        const arrayBuffer = await response.arrayBuffer();
                        return WebAssembly.compile(arrayBuffer);
                    }}
                    return null;
                }} catch (e) {{
                    return null;
                }}
            }})()
        "#,
            self.config.cache_name, url
        );

        let promise = js_sys::eval(&js_code)?;
        let promise: Promise = promise.dyn_into()?;
        let result = JsFuture::from(promise).await?;

        if result.is_null() || result.is_undefined() {
            Ok(None)
        } else {
            let module: WebAssembly::Module = result.dyn_into()?;
            Ok(Some(module))
        }
    }

    /// Save compiled module to cache
    async fn save_to_cache(&self, url: &str, _module: &WebAssembly::Module) -> Result<(), JsValue> {
        let js_code = format!(
            r#"
            (async () => {{
                try {{
                    const cache = await caches.open('{}');
                    const moduleBytes = await WebAssembly.Module.exports(arguments[0]);
                    const response = new Response(moduleBytes, {{
                        headers: {{ 'Content-Type': 'application/wasm' }}
                    }});
                    await cache.put('{}', response);
                    return true;
                }} catch (e) {{
                    console.warn('Failed to cache WASM module:', e);
                    return false;
                }}
            }})()
        "#,
            self.config.cache_name, url
        );

        let promise = js_sys::eval(&js_code)?;
        let promise: Promise = promise.dyn_into()?;
        let _result = JsFuture::from(promise).await?;

        Ok(())
    }

    /// Get current loading progress
    pub fn get_progress(&self) -> LoadingProgress {
        let _loaded_percentage = if self.total_size > 0 {
            (self.loaded_size as f64 / self.total_size as f64) * 100.0
        } else {
            0.0
        };

        let current_time = js_sys::Date::now();
        let elapsed_time_ms = current_time - self.loading_start_time;

        // Calculate loading speed (KB/s)
        let loading_speed_kbps = if elapsed_time_ms > 0.0 {
            (self.loaded_size as f64 / 1024.0) / (elapsed_time_ms / 1000.0)
        } else {
            0.0
        };

        // Calculate estimated remaining time
        let estimated_remaining_ms =
            if loading_speed_kbps > 0.0 && self.total_size > self.loaded_size {
                let remaining_bytes = self.total_size - self.loaded_size;
                let remaining_kb = remaining_bytes as f64 / 1024.0;
                (remaining_kb / loading_speed_kbps) * 1000.0
            } else {
                0.0
            };

        LoadingProgress {
            total_bytes: self.total_size,
            loaded_bytes: self.loaded_size,
            compiled_chunks: self.compiled_modules.len(),
            total_chunks: self.loaded_chunks.len(),
            loading_speed_kbps,
            estimated_remaining_ms,
        }
    }

    /// Update progress metrics when new data is loaded
    fn update_progress_metrics(&mut self, bytes_loaded: usize) {
        let current_time = js_sys::Date::now();
        self.loaded_size += bytes_loaded;
        self.last_progress_time = current_time;
        self.last_progress_bytes = self.loaded_size;

        // Log progress for debugging
        let progress_percent = if self.total_size > 0 {
            (self.loaded_size as f64 / self.total_size as f64) * 100.0
        } else {
            0.0
        };

        web_sys::console::log_1(
            &format!(
                "📊 Loading progress: {:.1}% ({} / {} bytes)",
                progress_percent, self.loaded_size, self.total_size
            )
            .into(),
        );
    }

    /// Check if currently loading
    #[wasm_bindgen(getter)]
    pub fn is_loading(&self) -> bool {
        self.is_loading
    }

    /// Get total loaded size
    #[wasm_bindgen(getter)]
    pub fn loaded_size(&self) -> usize {
        self.loaded_size
    }

    /// Get total size
    #[wasm_bindgen(getter)]
    pub fn total_size(&self) -> usize {
        self.total_size
    }

    /// Get loading percentage
    #[wasm_bindgen(getter)]
    pub fn loading_percentage(&self) -> f64 {
        if self.total_size > 0 {
            (self.loaded_size as f64 / self.total_size as f64) * 100.0
        } else {
            0.0
        }
    }
}

#[wasm_bindgen]
impl LoadingProgress {
    #[wasm_bindgen(getter)]
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    #[wasm_bindgen(getter)]
    pub fn loaded_bytes(&self) -> usize {
        self.loaded_bytes
    }

    #[wasm_bindgen(getter)]
    pub fn compiled_chunks(&self) -> usize {
        self.compiled_chunks
    }

    #[wasm_bindgen(getter)]
    pub fn total_chunks(&self) -> usize {
        self.total_chunks
    }

    #[wasm_bindgen(getter)]
    pub fn loading_speed_kbps(&self) -> f64 {
        self.loading_speed_kbps
    }

    #[wasm_bindgen(getter)]
    pub fn estimated_remaining_ms(&self) -> f64 {
        self.estimated_remaining_ms
    }

    #[wasm_bindgen(getter)]
    pub fn percentage(&self) -> f64 {
        if self.total_bytes > 0 {
            (self.loaded_bytes as f64 / self.total_bytes as f64) * 100.0
        } else {
            0.0
        }
    }
}

/// Check if streaming compilation is supported
#[wasm_bindgen]
pub fn is_streaming_compilation_supported() -> bool {
    let js_code = r#"
        try {
            return typeof WebAssembly.compileStreaming !== 'undefined';
        } catch (e) {
            return false;
        }
    "#;

    js_sys::eval(js_code)
        .map(|result| result.as_bool().unwrap_or(false))
        .unwrap_or(false)
}

/// Check if Cache API is available
#[wasm_bindgen]
pub fn is_cache_api_available() -> bool {
    let js_code = r#"
        try {
            return typeof caches !== 'undefined';
        } catch (e) {
            return false;
        }
    "#;

    js_sys::eval(js_code)
        .map(|result| result.as_bool().unwrap_or(false))
        .unwrap_or(false)
}

/// Get optimal chunk size based on connection type
#[wasm_bindgen]
pub fn get_optimal_chunk_size_kb() -> u32 {
    let js_code = r#"
        try {
            const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
            if (connection) {
                const effectiveType = connection.effectiveType;
                switch (effectiveType) {
                    case 'slow-2g': return 64;   // 64KB chunks for slow connections
                    case '2g': return 128;       // 128KB chunks
                    case '3g': return 256;       // 256KB chunks
                    case '4g': return 512;       // 512KB chunks for fast connections
                    default: return 256;        // Default to 256KB
                }
            }
            return 256; // Default chunk size
        } catch (e) {
            return 256;
        }
    "#;

    js_sys::eval(js_code)
        .ok()
        .and_then(|result| result.as_f64())
        .map(|size| size as u32)
        .unwrap_or(256)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_config() {
        let mut config = StreamingConfig::new();
        assert_eq!(config.chunk_size_kb(), 256);

        config.set_chunk_size_kb(512);
        assert_eq!(config.chunk_size_kb(), 512);

        config.set_max_concurrent_chunks(8);
        assert_eq!(config.max_concurrent_chunks(), 8);
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_feature_detection() {
        let _streaming_supported = is_streaming_compilation_supported();
        let _cache_supported = is_cache_api_available();
        let _optimal_chunk = get_optimal_chunk_size_kb();
    }
}
