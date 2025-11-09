//! Service Worker integration for offline capabilities and caching
//!
//! This module provides Service Worker support for:
//! - Model caching and offline access
//! - Background model loading and preprocessing
//! - Request/response interception for optimization
//! - Progressive Web App (PWA) features

#![allow(dead_code)]

use js_sys::{Function, Object, Promise};
use serde::{Deserialize, Serialize};
use serde_wasm_bindgen::{from_value, to_value};
use std::boxed::Box;
use std::collections::BTreeMap;
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{CacheStorage, MessageEvent, Navigator, ServiceWorkerRegistration};

/// Service Worker message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceWorkerMessage {
    /// Cache a model file
    CacheModel {
        model_url: String,
        cache_key: String,
        priority: CachePriority,
    },
    /// Load a model from cache
    LoadModel { cache_key: String },
    /// Preprocess input data
    PreprocessInput {
        input_data: Vec<u8>,
        preprocessing_type: String,
    },
    /// Background inference
    BackgroundInference {
        model_key: String,
        input_data: Vec<f32>,
        config: InferenceConfig,
    },
    /// Update cache statistics
    UpdateCacheStats,
    /// Clear cache
    ClearCache { cache_pattern: Option<String> },
}

/// Cache priority levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[wasm_bindgen]
pub enum CachePriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Inference configuration for background processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub batch_size: Option<usize>,
    pub use_cache: bool,
}

/// Service Worker response types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceWorkerResponse {
    /// Model cached successfully
    ModelCached {
        cache_key: String,
        size_bytes: usize,
        success: bool,
    },
    /// Model loaded from cache
    ModelLoaded {
        cache_key: String,
        data: Vec<u8>,
        success: bool,
    },
    /// Preprocessing completed
    PreprocessingComplete {
        processed_data: Vec<f32>,
        processing_time_ms: f64,
        success: bool,
    },
    /// Background inference completed
    InferenceComplete {
        result: Vec<f32>,
        execution_time_ms: f64,
        success: bool,
    },
    /// Cache statistics updated
    CacheStatsUpdated {
        total_size_bytes: usize,
        item_count: usize,
        hit_rate: f64,
    },
    /// Cache cleared
    CacheCleared {
        items_removed: usize,
        size_freed_bytes: usize,
    },
    /// Error occurred
    Error { message: String, error_type: String },
}

/// Service Worker manager
#[wasm_bindgen]
pub struct ServiceWorkerManager {
    registration: Option<ServiceWorkerRegistration>,
    cache_storage: Option<CacheStorage>,
    cache_name: String,
    message_handlers: BTreeMap<String, Function>,
}

#[wasm_bindgen]
impl ServiceWorkerManager {
    /// Create a new Service Worker manager
    #[wasm_bindgen(constructor)]
    pub fn new(cache_name: String) -> ServiceWorkerManager {
        ServiceWorkerManager {
            registration: None,
            cache_storage: None,
            cache_name,
            message_handlers: BTreeMap::new(),
        }
    }

    /// Register a Service Worker
    pub async fn register(&mut self, worker_url: String) -> Result<(), JsValue> {
        let window = web_sys::window().ok_or("No window available")?;
        let navigator = window.navigator();

        if !Self::is_service_worker_supported(&navigator) {
            return Err("Service Worker not supported".into());
        }

        let service_worker = navigator.service_worker();
        let registration_promise = service_worker.register(&worker_url);
        let registration = JsFuture::from(registration_promise).await?;

        self.registration = Some(registration.into());

        // Initialize cache storage
        self.cache_storage = js_sys::Reflect::get(&window, &JsValue::from_str("caches"))
            .ok()
            .and_then(|v| v.dyn_into::<CacheStorage>().ok());

        Ok(())
    }

    /// Check if Service Worker is supported
    pub fn is_service_worker_supported(navigator: &Navigator) -> bool {
        js_sys::Reflect::get(navigator, &JsValue::from_str("serviceWorker"))
            .map(|v| !v.is_undefined())
            .unwrap_or(false)
    }

    /// Send a message to the Service Worker (private - use specific methods instead)
    async fn send_message_internal(
        &self,
        message: ServiceWorkerMessage,
    ) -> Result<ServiceWorkerResponse, JsValue> {
        let registration = self.registration.as_ref().ok_or("Service Worker not registered")?;

        let active_worker = registration.active().ok_or("No active Service Worker")?;

        let message_js = to_value(&message)?;
        active_worker.post_message(&message_js)?;

        // Wait for response
        let response_promise = Promise::new(&mut |resolve, _reject| {
            let closure = Closure::wrap(Box::new(move |event: MessageEvent| {
                resolve.call1(&JsValue::UNDEFINED, &event.data()).unwrap();
            }) as Box<dyn FnMut(_)>);

            // Set up message listener
            // In a real implementation, this would properly handle the response
            closure.forget();
        });

        let response_data = JsFuture::from(response_promise).await?;
        let response: ServiceWorkerResponse = from_value(response_data)?;

        Ok(response)
    }

    /// Cache a model file (private - use JS-compatible method instead)
    async fn cache_model_internal(
        &self,
        model_url: String,
        cache_key: String,
        priority: CachePriority,
    ) -> Result<(), JsValue> {
        let message = ServiceWorkerMessage::CacheModel {
            model_url,
            cache_key,
            priority,
        };

        let response = self.send_message_internal(message).await?;

        match response {
            ServiceWorkerResponse::ModelCached { success, .. } => {
                if success {
                    Ok(())
                } else {
                    Err("Failed to cache model".into())
                }
            },
            ServiceWorkerResponse::Error { message, .. } => Err(message.into()),
            _ => Err("Unexpected response type".into()),
        }
    }

    /// Load a model from cache
    pub async fn load_cached_model(&self, cache_key: String) -> Result<Vec<u8>, JsValue> {
        let message = ServiceWorkerMessage::LoadModel { cache_key };

        let response = self.send_message_internal(message).await?;

        match response {
            ServiceWorkerResponse::ModelLoaded { data, success, .. } => {
                if success {
                    Ok(data)
                } else {
                    Err("Failed to load cached model".into())
                }
            },
            ServiceWorkerResponse::Error { message, .. } => Err(message.into()),
            _ => Err("Unexpected response type".into()),
        }
    }

    /// Preprocess input data in the background
    pub async fn preprocess_input(
        &self,
        input_data: Vec<u8>,
        preprocessing_type: String,
    ) -> Result<Vec<f32>, JsValue> {
        let message = ServiceWorkerMessage::PreprocessInput {
            input_data,
            preprocessing_type,
        };

        let response = self.send_message_internal(message).await?;

        match response {
            ServiceWorkerResponse::PreprocessingComplete {
                processed_data,
                success,
                ..
            } => {
                if success {
                    Ok(processed_data)
                } else {
                    Err("Failed to preprocess input".into())
                }
            },
            ServiceWorkerResponse::Error { message, .. } => Err(message.into()),
            _ => Err("Unexpected response type".into()),
        }
    }

    /// Run background inference (private - use JS-compatible method instead)
    async fn background_inference_internal(
        &self,
        model_key: String,
        input_data: Vec<f32>,
        config: InferenceConfig,
    ) -> Result<Vec<f32>, JsValue> {
        let message = ServiceWorkerMessage::BackgroundInference {
            model_key,
            input_data,
            config,
        };

        let response = self.send_message_internal(message).await?;

        match response {
            ServiceWorkerResponse::InferenceComplete {
                result, success, ..
            } => {
                if success {
                    Ok(result)
                } else {
                    Err("Failed to run background inference".into())
                }
            },
            ServiceWorkerResponse::Error { message, .. } => Err(message.into()),
            _ => Err("Unexpected response type".into()),
        }
    }

    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> Result<CacheStats, JsValue> {
        let message = ServiceWorkerMessage::UpdateCacheStats;

        let response = self.send_message_internal(message).await?;

        match response {
            ServiceWorkerResponse::CacheStatsUpdated {
                total_size_bytes,
                item_count,
                hit_rate,
            } => Ok(CacheStats {
                total_size_bytes,
                item_count,
                hit_rate,
            }),
            ServiceWorkerResponse::Error { message, .. } => Err(message.into()),
            _ => Err("Unexpected response type".into()),
        }
    }

    /// Clear cache
    pub async fn clear_cache(
        &self,
        cache_pattern: Option<String>,
    ) -> Result<ClearCacheResult, JsValue> {
        let message = ServiceWorkerMessage::ClearCache { cache_pattern };

        let response = self.send_message_internal(message).await?;

        match response {
            ServiceWorkerResponse::CacheCleared {
                items_removed,
                size_freed_bytes,
            } => Ok(ClearCacheResult {
                items_removed,
                size_freed_bytes,
            }),
            ServiceWorkerResponse::Error { message, .. } => Err(message.into()),
            _ => Err("Unexpected response type".into()),
        }
    }

    /// Add message handler
    pub fn add_message_handler(&mut self, message_type: String, handler: Function) {
        self.message_handlers.insert(message_type, handler);
    }

    /// Remove message handler
    pub fn remove_message_handler(&mut self, message_type: String) {
        self.message_handlers.remove(&message_type);
    }

    /// Unregister the Service Worker
    pub async fn unregister(&mut self) -> Result<(), JsValue> {
        if let Some(registration) = &self.registration {
            let unregister_promise = registration.unregister()?;
            let success = JsFuture::from(unregister_promise).await?;

            if success.as_bool().unwrap_or(false) {
                self.registration = None;
                Ok(())
            } else {
                Err("Failed to unregister Service Worker".into())
            }
        } else {
            Err("No Service Worker registered".into())
        }
    }
}

/// Cache statistics
#[wasm_bindgen]
pub struct CacheStats {
    total_size_bytes: usize,
    item_count: usize,
    hit_rate: f64,
}

#[wasm_bindgen]
impl CacheStats {
    #[wasm_bindgen(getter)]
    pub fn total_size_bytes(&self) -> usize {
        self.total_size_bytes
    }

    #[wasm_bindgen(getter)]
    pub fn item_count(&self) -> usize {
        self.item_count
    }

    #[wasm_bindgen(getter)]
    pub fn hit_rate(&self) -> f64 {
        self.hit_rate
    }
}

/// Clear cache result
#[wasm_bindgen]
pub struct ClearCacheResult {
    items_removed: usize,
    size_freed_bytes: usize,
}

#[wasm_bindgen]
impl ClearCacheResult {
    #[wasm_bindgen(getter)]
    pub fn items_removed(&self) -> usize {
        self.items_removed
    }

    #[wasm_bindgen(getter)]
    pub fn size_freed_bytes(&self) -> usize {
        self.size_freed_bytes
    }
}

/// Utility functions for Service Worker integration
/// Check if Service Worker is supported
#[wasm_bindgen]
pub fn is_service_worker_supported() -> bool {
    web_sys::window()
        .map(|w| w.navigator())
        .map(|nav| ServiceWorkerManager::is_service_worker_supported(&nav))
        .unwrap_or(false)
}

/// Create inference config
#[wasm_bindgen]
pub fn create_inference_config(
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    batch_size: Option<usize>,
    use_cache: bool,
) -> JsValue {
    let config = InferenceConfig {
        max_tokens,
        temperature,
        batch_size,
        use_cache,
    };
    to_value(&config).unwrap_or(JsValue::NULL)
}

/// Service Worker installation helper
#[wasm_bindgen]
pub async fn install_service_worker(
    worker_url: String,
    cache_name: String,
) -> Result<ServiceWorkerManager, JsValue> {
    let mut manager = ServiceWorkerManager::new(cache_name);
    manager.register(worker_url).await?;
    Ok(manager)
}

/// PWA installation prompt
#[wasm_bindgen]
pub struct PWAInstaller {
    deferred_prompt: Option<Object>,
}

#[wasm_bindgen]
impl PWAInstaller {
    #[wasm_bindgen(constructor)]
    pub fn new() -> PWAInstaller {
        PWAInstaller {
            deferred_prompt: None,
        }
    }

    /// Set up PWA installation prompt
    pub fn setup_install_prompt(&mut self) -> Result<(), JsValue> {
        let window = web_sys::window().ok_or("No window available")?;

        // Listen for beforeinstallprompt event
        let closure = Closure::wrap(Box::new(|event: web_sys::Event| {
            // Prevent default installation prompt
            event.prevent_default();
            // Store the event for later use
            web_sys::console::log_1(&"PWA install prompt ready".into());
        }) as Box<dyn FnMut(_)>);

        window.add_event_listener_with_callback(
            "beforeinstallprompt",
            closure.as_ref().unchecked_ref(),
        )?;
        closure.forget();

        Ok(())
    }

    /// Show PWA installation prompt
    pub async fn show_install_prompt(&self) -> Result<bool, JsValue> {
        if let Some(_prompt) = &self.deferred_prompt {
            // In a real implementation, this would trigger the installation prompt
            // For now, return true to indicate the prompt was shown
            Ok(true)
        } else {
            Err("No deferred prompt available".into())
        }
    }

    /// Check if PWA is installable
    pub fn is_installable(&self) -> bool {
        self.deferred_prompt.is_some()
    }
}

impl Default for PWAInstaller {
    fn default() -> Self {
        Self::new()
    }
}
