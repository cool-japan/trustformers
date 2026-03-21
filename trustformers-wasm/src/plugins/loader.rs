// Plugin Loader - Dynamic plugin loading and management
//
// This module handles the dynamic loading of plugins from various sources
// including URLs, local storage, and embedded plugins.

use super::interface::{Plugin, PluginCapabilities, PluginConfig, PluginContext, PluginError};
use super::registry::PluginMetadata;
use std::boxed::Box;
use std::string::{String, ToString};
use std::vec::Vec;
use core::fmt;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

/// Plugin loader for dynamic loading of plugins
#[derive(Debug)]
pub struct PluginLoader {
    config: LoaderConfig,
    cache: std::collections::HashMap<String, CachedPlugin>,
    loading_strategies: std::collections::HashMap<PluginSource, Box<dyn LoadingStrategy>>,
}

impl PluginLoader {
    /// Create a new plugin loader with default configuration
    pub fn new() -> Self {
        let mut loader = Self {
            config: LoaderConfig::default(),
            cache: std::collections::HashMap::new(),
            loading_strategies: std::collections::HashMap::new(),
        };

        // Register default loading strategies
        loader.loading_strategies.insert(PluginSource::Url, Box::new(UrlLoadingStrategy::new()));
        loader.loading_strategies.insert(PluginSource::Local, Box::new(LocalLoadingStrategy::new()));
        loader.loading_strategies.insert(PluginSource::Embedded, Box::new(EmbeddedLoadingStrategy::new()));

        loader
    }

    /// Create a plugin loader with custom configuration
    pub fn with_config(config: LoaderConfig) -> Self {
        let mut loader = Self::new();
        loader.config = config;
        loader
    }

    /// Load a plugin from metadata
    pub async fn load_plugin(&mut self, metadata: &PluginMetadata) -> Result<Box<dyn Plugin>, LoadingError> {
        let plugin_id = &metadata.id;

        // Check cache first
        if let Some(cached) = self.cache.get(plugin_id) {
            if self.is_cache_valid(cached) {
                web_sys::console::log_1(&format!("Loading plugin '{plugin_id}' from cache").into());
                return self.create_plugin_from_cache(cached).await;
            } else {
                // Remove invalid cache entry
                self.cache.remove(plugin_id);
            }
        }

        // Determine plugin source
        let source = self.determine_plugin_source(metadata)?;

        // Get loading strategy
        let strategy = self.loading_strategies.get(&source)
            .ok_or_else(|| LoadingError::UnsupportedSource(source))?;

        web_sys::console::log_1(&format!("Loading plugin '{plugin_id}' from source: {source:?}").into());

        // Load plugin code
        let plugin_code = strategy.load_plugin_code(metadata).await?;

        // Validate plugin code
        self.validate_plugin_code(&plugin_code, metadata)?;

        // Create plugin instance
        let plugin = self.create_plugin_instance(&plugin_code, metadata).await?;

        // Cache the plugin if caching is enabled
        if self.config.enable_cache {
            let cached_plugin = CachedPlugin {
                metadata: metadata.clone(),
                code: plugin_code,
                loaded_at: js_sys::Date::now(),
                access_count: 1,
            };
            self.cache.insert(plugin_id.clone(), cached_plugin);
        }

        web_sys::console::log_1(&format!("Successfully loaded plugin '{plugin_id}'").into());
        Ok(plugin)
    }

    /// Unload a plugin from cache
    pub fn unload_plugin(&mut self, plugin_id: &str) -> Result<(), LoadingError> {
        self.cache.remove(plugin_id);
        web_sys::console::log_1(&format!("Unloaded plugin '{plugin_id}' from cache").into());
        Ok(())
    }

    /// Check if a plugin is cached
    pub fn is_plugin_cached(&self, plugin_id: &str) -> bool {
        self.cache.contains_key(plugin_id)
    }

    /// Get cache statistics
    pub fn get_cache_statistics(&self) -> CacheStatistics {
        let total_size = self.cache.values().map(|p| p.code.len()).sum();
        let mut access_distribution = std::collections::HashMap::new();

        for cached in self.cache.values() {
            let range = match cached.access_count {
                1..=5 => "1-5",
                6..=20 => "6-20",
                21..=100 => "21-100",
                _ => "100+",
            };
            *access_distribution.entry(range.to_string()).or_insert(0) += 1;
        }

        CacheStatistics {
            cached_plugins: self.cache.len(),
            total_cache_size: total_size,
            access_distribution,
            oldest_entry: self.cache.values().map(|p| p.loaded_at).min(),
            newest_entry: self.cache.values().map(|p| p.loaded_at).max(),
        }
    }

    /// Clear plugin cache
    pub fn clear_cache(&mut self) {
        let count = self.cache.len();
        self.cache.clear();
        web_sys::console::log_1(&format!("Cleared {count} plugins from cache").into());
    }

    /// Preload plugins
    pub async fn preload_plugins(&mut self, plugin_ids: Vec<String>, registry: &super::registry::PluginRegistry) -> Result<usize, LoadingError> {
        let mut loaded_count = 0;

        for plugin_id in plugin_ids {
            if let Some(metadata) = registry.get_metadata(&plugin_id) {
                match self.load_plugin(metadata).await {
                    Ok(_) => {
                        loaded_count += 1;
                        web_sys::console::log_1(&format!("Preloaded plugin '{plugin_id}'").into());
                    }
                    Err(e) => {
                        web_sys::console::warn_1(&format!("Failed to preload plugin '{plugin_id}': {e:?}").into());
                    }
                }
            } else {
                web_sys::console::warn_1(&format!("Plugin '{plugin_id}' not found in registry").into());
            }
        }

        Ok(loaded_count)
    }

    /// Register a custom loading strategy
    pub fn register_loading_strategy(&mut self, source: PluginSource, strategy: Box<dyn LoadingStrategy>) {
        self.loading_strategies.insert(source, strategy);
        web_sys::console::log_1(&format!("Registered loading strategy for source: {source:?}").into());
    }

    /// Remove a loading strategy
    pub fn remove_loading_strategy(&mut self, source: PluginSource) -> Option<Box<dyn LoadingStrategy>> {
        self.loading_strategies.remove(&source)
    }

    // Helper methods

    fn determine_plugin_source(&self, metadata: &PluginMetadata) -> Result<PluginSource, LoadingError> {
        if let Some(ref url) = metadata.source_url {
            if url.starts_with("http://") || url.starts_with("https://") {
                Ok(PluginSource::Url)
            } else if url.starts_with("file://") || url.starts_with("/") {
                Ok(PluginSource::Local)
            } else {
                Err(LoadingError::InvalidSource(url.clone()))
            }
        } else {
            // Check if it's an embedded plugin
            Ok(PluginSource::Embedded)
        }
    }

    fn is_cache_valid(&self, cached: &CachedPlugin) -> bool {
        if let Some(max_age) = self.config.cache_max_age_ms {
            let age = js_sys::Date::now() - cached.loaded_at;
            age < max_age
        } else {
            true
        }
    }

    async fn create_plugin_from_cache(&self, cached: &CachedPlugin) -> Result<Box<dyn Plugin>, LoadingError> {
        // In a real implementation, this would deserialize the cached plugin
        // For now, we'll recreate it from the cached code
        self.create_plugin_instance(&cached.code, &cached.metadata).await
    }

    fn validate_plugin_code(&self, code: &[u8], metadata: &PluginMetadata) -> Result<(), LoadingError> {
        // Validate checksum if provided
        if let Some(ref expected_checksum) = metadata.checksum {
            let actual_checksum = self.calculate_checksum(code);
            if actual_checksum != *expected_checksum {
                return Err(LoadingError::ChecksumMismatch(expected_checksum.clone(), actual_checksum));
            }
        }

        // Validate code size
        if code.len() > self.config.max_plugin_size {
            return Err(LoadingError::PluginTooLarge(code.len(), self.config.max_plugin_size));
        }

        // Basic security checks
        if self.config.enable_security_checks {
            self.perform_security_checks(code)?;
        }

        Ok(())
    }

    async fn create_plugin_instance(&self, code: &[u8], metadata: &PluginMetadata) -> Result<Box<dyn Plugin>, LoadingError> {
        // In a real implementation, this would:
        // 1. Compile the WebAssembly module
        // 2. Instantiate the plugin
        // 3. Initialize it with the metadata

        // For now, we'll create a basic plugin implementation
        Ok(Box::new(BasicPlugin::new(metadata.clone())))
    }

    fn calculate_checksum(&self, data: &[u8]) -> String {
        // Simple hash calculation (in production, use a proper hash function)
        let mut hash = 0u32;
        for byte in data {
            hash = hash.wrapping_mul(31).wrapping_add(*byte as u32);
        }
format!("{hash:08x}")
    }

    fn perform_security_checks(&self, code: &[u8]) -> Result<(), LoadingError> {
        // Basic security checks
        // In production, this would include:
        // - Code signing verification
        // - Malware scanning
        // - Resource usage limits
        // - API access validation

        if code.is_empty() {
            return Err(LoadingError::SecurityCheckFailed("Empty plugin code".to_string()));
        }

        // Check for basic WASM magic number
        if code.len() >= 4 && &code[0..4] == b"\\0asm" {
            // Looks like a WASM module
            Ok(())
        } else {
            Err(LoadingError::SecurityCheckFailed("Invalid WASM format".to_string()))
        }
    }
}

impl Default for PluginLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Plugin loading strategies
#[async_trait::async_trait(?Send)]
pub trait LoadingStrategy: fmt::Debug + Send + Sync {
    /// Load plugin code from the specified source
    async fn load_plugin_code(&self, metadata: &PluginMetadata) -> Result<Vec<u8>, LoadingError>;

    /// Check if the source is available
    async fn is_source_available(&self, metadata: &PluginMetadata) -> bool;

    /// Get the source priority (lower is higher priority)
    fn get_priority(&self) -> u8;
}

/// URL-based loading strategy
#[derive(Debug)]
pub struct UrlLoadingStrategy {
    client: web_sys::Window,
}

impl UrlLoadingStrategy {
    pub fn new() -> Self {
        Self {
            client: web_sys::window().expect("should have a window in WASM"),
        }
    }
}

#[async_trait::async_trait(?Send)]
impl LoadingStrategy for UrlLoadingStrategy {
    async fn load_plugin_code(&self, metadata: &PluginMetadata) -> Result<Vec<u8>, LoadingError> {
        let url = metadata.source_url.as_ref()
            .ok_or_else(|| LoadingError::MissingSource)?;

        // Use fetch API to download the plugin
        let request = web_sys::Request::new_with_str(url)
            .map_err(|_| LoadingError::NetworkError("Failed to create request".to_string()))?;

        let response_promise = self.client.fetch_with_request(&request);
        let response = JsFuture::from(response_promise).await
            .map_err(|_| LoadingError::NetworkError("Fetch failed".to_string()))?;

        let response: web_sys::Response = response.dyn_into()
            .map_err(|_| LoadingError::NetworkError("Invalid response".to_string()))?;

        if !response.ok() {
            return Err(LoadingError::NetworkError(format!("HTTP {status}", status = response.status())));
        }

        let array_buffer_promise = response.array_buffer()
            .map_err(|_| LoadingError::NetworkError("Failed to get array buffer".to_string()))?;

        let array_buffer = JsFuture::from(array_buffer_promise).await
            .map_err(|_| LoadingError::NetworkError("Failed to read array buffer".to_string()))?;

        let array_buffer: js_sys::ArrayBuffer = array_buffer.dyn_into()
            .map_err(|_| LoadingError::NetworkError("Invalid array buffer".to_string()))?;

        let uint8_array = js_sys::Uint8Array::new(&array_buffer);
        let mut data = vec![0u8; uint8_array.length() as usize];
        uint8_array.copy_to(&mut data);

        Ok(data)
    }

    async fn is_source_available(&self, metadata: &PluginMetadata) -> bool {
        metadata.source_url.is_some()
    }

    fn get_priority(&self) -> u8 {
        1 // High priority
    }
}

/// Local storage loading strategy
#[derive(Debug)]
pub struct LocalLoadingStrategy;

impl LocalLoadingStrategy {
    pub fn new() -> Self {
        Self
    }

    /// Load plugin code from IndexedDB
    async fn load_from_indexeddb(&self, _path: &str, plugin_id: &str) -> Result<Vec<u8>, LoadingError> {
        // Get window and indexedDB
        let window = web_sys::window().ok_or_else(|| LoadingError::BrowserApiUnavailable("window".to_string()))?;
        let indexed_db = window
            .indexed_db()
            .map_err(|_| LoadingError::BrowserApiUnavailable("indexedDB".to_string()))?
            .ok_or_else(|| LoadingError::BrowserApiUnavailable("indexedDB support".to_string()))?;

        // Open database
        let open_request = indexed_db
            .open_with_u32("TrustformersPlugins", 1)
            .map_err(|e| LoadingError::DatabaseError(format!("Failed to open IndexedDB: {e:?}")))?;

        // Convert to promise and await
        let promise = js_sys::Promise::resolve(&open_request);
        let db_result = JsFuture::from(promise).await
            .map_err(|e| LoadingError::DatabaseError(format!("IndexedDB open failed: {e:?}")))?;

        let db: web_sys::IdbDatabase = db_result.dyn_into()
            .map_err(|_| LoadingError::DatabaseError("Invalid database object".to_string()))?;

        // Create transaction
        let transaction = db
            .transaction_with_str_and_mode("plugins", web_sys::IdbTransactionMode::Readonly)
            .map_err(|e| LoadingError::DatabaseError(format!("Failed to create transaction: {e:?}")))?;

        let object_store = transaction
            .object_store("plugins")
            .map_err(|e| LoadingError::DatabaseError(format!("Failed to get object store: {e:?}")))?;

        // Get plugin data
        let get_request = object_store
            .get(&JsValue::from_str(plugin_id))
            .map_err(|e| LoadingError::DatabaseError(format!("Failed to get plugin: {e:?}")))?;

        let promise = js_sys::Promise::resolve(&get_request);
        let result = JsFuture::from(promise).await
            .map_err(|e| LoadingError::DatabaseError(format!("Plugin retrieval failed: {e:?}")))?;

        if result.is_null() || result.is_undefined() {
            return Err(LoadingError::FileNotFound(format!("Plugin '{plugin_id}' not found in IndexedDB")));
        }

        // Extract plugin code from result
        let plugin_data = js_sys::Reflect::get(&result, &JsValue::from_str("code"))
            .map_err(|_| LoadingError::DatabaseError("Failed to extract plugin code".to_string()))?;

        // Convert to Vec<u8>
        if let Ok(array_buffer) = plugin_data.dyn_into::<js_sys::ArrayBuffer>() {
            let uint8_array = js_sys::Uint8Array::new(&array_buffer);
            let mut code = vec![0u8; uint8_array.length() as usize];
            uint8_array.copy_to(&mut code);
            Ok(code)
        } else if plugin_data.is_string() {
            // Handle string data (e.g., JavaScript code)
            let code_str = plugin_data.as_string().unwrap_or_default();
            Ok(code_str.into_bytes())
        } else {
            Err(LoadingError::DatabaseError("Invalid plugin data format".to_string()))
        }
    }

    /// Load plugin code from localStorage (for smaller plugins)
    async fn load_from_localstorage(&self, _path: &str, plugin_id: &str) -> Result<Vec<u8>, LoadingError> {
        let window = web_sys::window().ok_or_else(|| LoadingError::BrowserApiUnavailable("window".to_string()))?;
        let storage = window
            .local_storage()
            .map_err(|_| LoadingError::BrowserApiUnavailable("localStorage".to_string()))?
            .ok_or_else(|| LoadingError::BrowserApiUnavailable("localStorage support".to_string()))?;

        let key = format!("trustformers_plugin_{plugin_id}");
        let plugin_data = storage
            .get_item(&key)
            .map_err(|e| LoadingError::DatabaseError(format!("localStorage access failed: {e:?}")))?
            .ok_or_else(|| LoadingError::FileNotFound(format!("Plugin '{plugin_id}' not found in localStorage")))?;

        // Decode base64 if the data is base64 encoded, otherwise treat as plain text
        if plugin_data.starts_with("data:application/wasm;base64,") {
            let base64_data = &plugin_data[29..]; // Remove the data URI prefix
            self.decode_base64(base64_data)
        } else if plugin_data.starts_with("data:text/javascript;base64,") {
            let base64_data = &plugin_data[29..]; // Remove the data URI prefix
            self.decode_base64(base64_data)
        } else {
            // Plain text JavaScript code
            Ok(plugin_data.into_bytes())
        }
    }

    /// Check if plugin is available in IndexedDB
    async fn check_indexeddb_availability(&self, _path: &str, plugin_id: &str) -> bool {
        match self.load_from_indexeddb(_path, plugin_id).await {
            Ok(_) => true,
            Err(_) => false,
        }
    }

    /// Check if plugin is available in localStorage
    async fn check_localstorage_availability(&self, _path: &str, plugin_id: &str) -> bool {
        let window = web_sys::window();
        if let Some(window) = window {
            if let Ok(Some(storage)) = window.local_storage() {
                let key = format!("trustformers_plugin_{plugin_id}");
                return storage.get_item(&key).unwrap_or(None).is_some();
            }
        }
        false
    }

    /// Decode base64 string to bytes
    fn decode_base64(&self, base64_data: &str) -> Result<Vec<u8>, LoadingError> {
        // Use browser's atob function for base64 decoding
        let window = web_sys::window().ok_or_else(|| LoadingError::BrowserApiUnavailable("window".to_string()))?;
        let decoded = window
            .atob(base64_data)
            .map_err(|e| LoadingError::DecodingError(format!("Base64 decoding failed: {e:?}")))?;

        Ok(decoded.into_bytes())
    }

    /// Store plugin in IndexedDB for future use
    pub async fn store_plugin_in_indexeddb(&self, plugin_id: &str, code: &[u8], metadata: &PluginMetadata) -> Result<(), LoadingError> {
        let window = web_sys::window().ok_or_else(|| LoadingError::BrowserApiUnavailable("window".to_string()))?;
        let indexed_db = window
            .indexed_db()
            .map_err(|_| LoadingError::BrowserApiUnavailable("indexedDB".to_string()))?
            .ok_or_else(|| LoadingError::BrowserApiUnavailable("indexedDB support".to_string()))?;

        // Open database
        let open_request = indexed_db
            .open_with_u32("TrustformersPlugins", 1)
            .map_err(|e| LoadingError::DatabaseError(format!("Failed to open IndexedDB: {e:?}")))?;

        let promise = js_sys::Promise::resolve(&open_request);
        let db_result = JsFuture::from(promise).await
            .map_err(|e| LoadingError::DatabaseError(format!("IndexedDB open failed: {e:?}")))?;

        let db: web_sys::IdbDatabase = db_result.dyn_into()
            .map_err(|_| LoadingError::DatabaseError("Invalid database object".to_string()))?;

        // Create transaction
        let transaction = db
            .transaction_with_str_and_mode("plugins", web_sys::IdbTransactionMode::Readwrite)
            .map_err(|e| LoadingError::DatabaseError(format!("Failed to create transaction: {e:?}")))?;

        let object_store = transaction
            .object_store("plugins")
            .map_err(|e| LoadingError::DatabaseError(format!("Failed to get object store: {e:?}")))?;

        // Create plugin record
        let plugin_record = js_sys::Object::new();
        js_sys::Reflect::set(&plugin_record, &JsValue::from_str("id"), &JsValue::from_str(plugin_id))
            .map_err(|_| LoadingError::DatabaseError("Failed to set plugin id".to_string()))?;

        let array_buffer = js_sys::ArrayBuffer::new(code.len() as u32);
        let uint8_array = js_sys::Uint8Array::new(&array_buffer);
        uint8_array.copy_from(code);

        js_sys::Reflect::set(&plugin_record, &JsValue::from_str("code"), &array_buffer)
            .map_err(|_| LoadingError::DatabaseError("Failed to set plugin code".to_string()))?;

        js_sys::Reflect::set(&plugin_record, &JsValue::from_str("metadata"), &serde_wasm_bindgen::to_value(metadata).unwrap_or(JsValue::NULL))
            .map_err(|_| LoadingError::DatabaseError("Failed to set plugin metadata".to_string()))?;

        js_sys::Reflect::set(&plugin_record, &JsValue::from_str("stored_at"), &JsValue::from_f64(js_sys::Date::now()))
            .map_err(|_| LoadingError::DatabaseError("Failed to set storage timestamp".to_string()))?;

        // Store the plugin
        let put_request = object_store
            .put_with_key(&plugin_record, &JsValue::from_str(plugin_id))
            .map_err(|e| LoadingError::DatabaseError(format!("Failed to store plugin: {e:?}")))?;

        let promise = js_sys::Promise::resolve(&put_request);
        JsFuture::from(promise).await
            .map_err(|e| LoadingError::DatabaseError(format!("Plugin storage failed: {e:?}")))?;

        web_sys::console::log_1(&format!("Successfully stored plugin '{plugin_id}' in IndexedDB").into());
        Ok(())
    }

    /// Store plugin in localStorage (for smaller plugins)
    pub async fn store_plugin_in_localstorage(&self, plugin_id: &str, code: &[u8]) -> Result<(), LoadingError> {
        let window = web_sys::window().ok_or_else(|| LoadingError::BrowserApiUnavailable("window".to_string()))?;
        let storage = window
            .local_storage()
            .map_err(|_| LoadingError::BrowserApiUnavailable("localStorage".to_string()))?
            .ok_or_else(|| LoadingError::BrowserApiUnavailable("localStorage support".to_string()))?;

        // Encode as base64 data URI
        let base64_data = window
            .btoa(&String::from_utf8_lossy(code))
            .map_err(|e| LoadingError::EncodingError(format!("Base64 encoding failed: {e:?}")))?;

        let data_uri = if code.len() > 4 && &code[0..4] == b"\0asm" {
            format!("data:application/wasm;base64,{base64_data}")
        } else {
            format!("data:text/javascript;base64,{base64_data}")
        };

        let key = format!("trustformers_plugin_{plugin_id}");
        storage
            .set_item(&key, &data_uri)
            .map_err(|e| LoadingError::DatabaseError(format!("localStorage storage failed: {e:?}")))?;

        web_sys::console::log_1(&format!("Successfully stored plugin '{plugin_id}' in localStorage").into());
        Ok(())
    }
}

#[async_trait::async_trait(?Send)]
impl LoadingStrategy for LocalLoadingStrategy {
    async fn load_plugin_code(&self, metadata: &PluginMetadata) -> Result<Vec<u8>, LoadingError> {
        let path = metadata.source_url.as_ref()
            .ok_or_else(|| LoadingError::MissingSource)?;

        web_sys::console::log_1(&format!("Loading plugin from local storage: {path}").into());

        // Try IndexedDB first for persistent local storage
        match self.load_from_indexeddb(path, &metadata.id).await {
            Ok(code) => {
                web_sys::console::log_1(&format!("Successfully loaded plugin '{}' from IndexedDB", id = metadata.id).into());
                return Ok(code);
            }
            Err(e) => {
                web_sys::console::warn_1(&format!("IndexedDB loading failed for '{id}': {e:?}", id = metadata.id).into());
            }
        }

        // Try localStorage as fallback for smaller plugins
        match self.load_from_localstorage(path, &metadata.id).await {
            Ok(code) => {
                web_sys::console::log_1(&format!("Successfully loaded plugin '{id}' from localStorage", id = metadata.id).into());
                return Ok(code);
            }
            Err(e) => {
                web_sys::console::warn_1(&format!("localStorage loading failed for '{id}': {e:?}", id = metadata.id).into());
            }
        }

        // If all local loading methods fail, return appropriate error
        Err(LoadingError::FileNotFound(format!("Plugin '{id}' not found in local storage", id = metadata.id)))
    }

    async fn is_source_available(&self, metadata: &PluginMetadata) -> bool {
        let path = metadata.source_url.as_ref();
        if let Some(path) = path {
            // Check if plugin exists in IndexedDB or localStorage
            self.check_indexeddb_availability(path, &metadata.id).await ||
            self.check_localstorage_availability(path, &metadata.id).await
        } else {
            false
        }
    }

    fn get_priority(&self) -> u8 {
        2 // Medium priority
    }
}

/// Embedded plugin loading strategy
#[derive(Debug)]
pub struct EmbeddedLoadingStrategy {
    plugins: std::collections::HashMap<String, Vec<u8>>,
}

impl EmbeddedLoadingStrategy {
    pub fn new() -> Self {
        Self {
            plugins: std::collections::HashMap::new(),
        }
    }

    /// Register an embedded plugin
    pub fn register_embedded_plugin(&mut self, plugin_id: String, code: Vec<u8>) {
        self.plugins.insert(plugin_id, code);
    }
}

#[async_trait::async_trait(?Send)]
impl LoadingStrategy for EmbeddedLoadingStrategy {
    async fn load_plugin_code(&self, metadata: &PluginMetadata) -> Result<Vec<u8>, LoadingError> {
        self.plugins.get(&metadata.id)
            .cloned()
            .ok_or_else(|| LoadingError::PluginNotFound(metadata.id.clone()))
    }

    async fn is_source_available(&self, metadata: &PluginMetadata) -> bool {
        self.plugins.contains_key(&metadata.id)
    }

    fn get_priority(&self) -> u8 {
        0 // Highest priority
    }
}

/// Basic plugin implementation for testing
#[derive(Debug)]
struct BasicPlugin {
    metadata: PluginMetadata,
    initialized: bool,
    config: PluginConfig,
}

impl BasicPlugin {
    fn new(metadata: PluginMetadata) -> Self {
        let config = PluginConfig::new(
            metadata.id.clone(),
            metadata.name.clone(),
            metadata.version.clone(),
            metadata.plugin_type,
        );

        Self {
            metadata,
            initialized: false,
            config,
        }
    }
}

#[async_trait::async_trait(?Send)]
impl Plugin for BasicPlugin {
    fn capabilities(&self) -> PluginCapabilities {
        self.metadata.capabilities.clone()
    }

    fn config(&self) -> &PluginConfig {
        &self.config
    }

    async fn initialize(&mut self, _context: PluginContext) -> Result<(), PluginError> {
        self.initialized = true;
        Ok(())
    }

    async fn execute(&mut self, function_name: &str, _context: PluginContext) -> Result<serde_json::Value, PluginError> {
        if !self.initialized {
            return Err(PluginError::NotInitialized);
        }

        match function_name {
            "test" => Ok(serde_json::Value::String("Hello from plugin!".to_string())),
            _ => Err(PluginError::InvalidFunction(function_name.to_string())),
        }
    }

    async fn cleanup(&mut self) -> Result<(), PluginError> {
        self.initialized = false;
        Ok(())
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }

    fn version(&self) -> &str {
        &self.metadata.version
    }

    fn dependencies(&self) -> Vec<String> {
        self.metadata.dependencies.clone()
    }
}

/// Plugin source types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PluginSource {
    Url,
    Local,
    Embedded,
    Custom(u8),
}

/// Cached plugin data
#[derive(Debug, Clone)]
struct CachedPlugin {
    metadata: PluginMetadata,
    code: Vec<u8>,
    loaded_at: f64,
    access_count: u32,
}

/// Loader configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoaderConfig {
    pub enable_cache: bool,
    pub cache_max_age_ms: Option<f64>,
    pub max_cache_size: Option<usize>,
    pub max_plugin_size: usize,
    pub enable_security_checks: bool,
    pub allowed_sources: Vec<String>,
    pub timeout_ms: u32,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            enable_cache: true,
            cache_max_age_ms: Some(3600000.0), // 1 hour
            max_cache_size: Some(100 * 1024 * 1024), // 100 MB
            max_plugin_size: 10 * 1024 * 1024, // 10 MB
            enable_security_checks: true,
            allowed_sources: Vec::new(),
            timeout_ms: 30000, // 30 seconds
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    pub cached_plugins: usize,
    pub total_cache_size: usize,
    pub access_distribution: std::collections::HashMap<String, usize>,
    pub oldest_entry: Option<f64>,
    pub newest_entry: Option<f64>,
}

/// Loading errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadingError {
    PluginNotFound(String),
    MissingSource,
    InvalidSource(String),
    UnsupportedSource(PluginSource),
    NetworkError(String),
    SecurityCheckFailed(String),
    ChecksumMismatch(String, String), // expected, actual
    PluginTooLarge(usize, usize), // actual, max
    CompilationFailed(String),
    InstantiationFailed(String),
    NotImplemented(String),
    CacheError(String),
    TimeoutError,
    // New error types for local file loading
    BrowserApiUnavailable(String),
    DatabaseError(String),
    FileNotFound(String),
    DecodingError(String),
    EncodingError(String),
}

impl fmt::Display for LoadingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadingError::PluginNotFound(id) => write!(f, "Plugin not found: {}", id),
            LoadingError::MissingSource => write!(f, "Missing plugin source"),
            LoadingError::InvalidSource(src) => write!(f, "Invalid plugin source: {}", src),
            LoadingError::UnsupportedSource(src) => write!(f, "Unsupported plugin source: {:?}", src),
            LoadingError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            LoadingError::SecurityCheckFailed(msg) => write!(f, "Security check failed: {}", msg),
            LoadingError::ChecksumMismatch(exp, act) => write!(f, "Checksum mismatch: expected {}, actual {}", exp, act),
            LoadingError::PluginTooLarge(actual, max) => write!(f, "Plugin too large: {} bytes (max {})", actual, max),
            LoadingError::CompilationFailed(msg) => write!(f, "Compilation failed: {}", msg),
            LoadingError::InstantiationFailed(msg) => write!(f, "Instantiation failed: {}", msg),
            LoadingError::NotImplemented(feature) => write!(f, "Not implemented: {}", feature),
            LoadingError::CacheError(msg) => write!(f, "Cache error: {}", msg),
            LoadingError::TimeoutError => write!(f, "Loading timeout"),
            LoadingError::BrowserApiUnavailable(api) => write!(f, "Browser API unavailable: {}", api),
            LoadingError::DatabaseError(msg) => write!(f, "Database error: {}", msg),
            LoadingError::FileNotFound(path) => write!(f, "File not found: {}", path),
            LoadingError::DecodingError(msg) => write!(f, "Decoding error: {}", msg),
            LoadingError::EncodingError(msg) => write!(f, "Encoding error: {}", msg),
        }
    }
}

impl std::error::Error for LoadingError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugins::interface::PluginType;

    #[test]
    fn test_loader_creation() {
        let loader = PluginLoader::new();
        assert!(loader.loading_strategies.contains_key(&PluginSource::Url));
        assert!(loader.loading_strategies.contains_key(&PluginSource::Local));
        assert!(loader.loading_strategies.contains_key(&PluginSource::Embedded));
    }

    #[test]
    fn test_cache_statistics() {
        let loader = PluginLoader::new();
        let stats = loader.get_cache_statistics();
        assert_eq!(stats.cached_plugins, 0);
        assert_eq!(stats.total_cache_size, 0);
    }

    #[test]
    fn test_basic_plugin() {
        let metadata = PluginMetadata::new(
            "test".to_string(),
            "Test Plugin".to_string(),
            "1.0.0".to_string(),
            PluginType::Extension,
        );

        let plugin = BasicPlugin::new(metadata);
        assert!(!plugin.is_initialized());
        assert_eq!(plugin.version(), "1.0.0");
    }
}