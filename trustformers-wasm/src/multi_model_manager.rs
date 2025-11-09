//! Multi-model management system for efficient model loading and switching
//!
//! This module provides comprehensive multi-model management capabilities including:
//! - Model registry and metadata management
//! - Dynamic model loading/unloading with memory optimization
//! - Model switching and intelligent routing
//! - Performance optimization for multi-model scenarios
//! - Model versioning and A/B testing support
//! - Resource allocation and cleanup

use crate::debug::DebugLogger;
use crate::quantization::WebQuantizer;
#[cfg(feature = "indexeddb")]
use crate::storage::ModelStorage;
use js_sys::{Array, Date, Object};
use serde::{Deserialize, Serialize};
use std::string::{String, ToString};
use std::vec::Vec;
use std::{format, vec};
use wasm_bindgen::prelude::*;

/// Model status in the management system
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelStatus {
    /// Model is not loaded
    NotLoaded,
    /// Model is currently loading
    Loading,
    /// Model is loaded and ready for inference
    Ready,
    /// Model is currently unloading
    Unloading,
    /// Model encountered an error
    Error,
    /// Model is warming up (first inference)
    WarmingUp,
}

/// Model priority for memory management
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ModelPriority {
    /// Low priority - can be unloaded first
    Low = 0,
    /// Normal priority
    Normal = 1,
    /// High priority - kept in memory longer
    High = 2,
    /// Critical priority - never auto-unloaded
    Critical = 3,
}

/// Model deployment environment
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeploymentEnvironment {
    /// Development environment
    Development,
    /// Staging environment
    Staging,
    /// Production environment
    Production,
    /// A/B testing environment
    Testing,
}

/// Model metadata and configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub id: String,
    pub name: String,
    pub version: String,
    pub description: String,
    pub model_type: String,
    pub architecture: String,
    pub size_bytes: usize,
    pub priority: ModelPriority,
    pub tags: Vec<String>,
    pub environment: DeploymentEnvironment,
    pub created_at: f64,
    pub last_used: f64,
    pub usage_count: u32,
    pub capabilities: Vec<String>,
    pub requirements: ModelRequirements,
    pub download_url: Option<String>,
}

/// Model resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRequirements {
    pub min_memory_mb: u32,
    pub min_gpu_memory_mb: u32,
    pub requires_gpu: bool,
    pub requires_webgpu: bool,
    pub min_cpu_cores: u32,
    pub recommended_batch_size: u32,
}

/// Loaded model instance
pub struct LoadedModel {
    pub metadata: ModelMetadata,
    pub status: ModelStatus,
    pub session: Option<crate::InferenceSession>,
    pub load_time: f64,
    pub memory_usage: usize,
    pub gpu_memory_usage: usize,
    pub warmup_completed: bool,
    pub performance_stats: ModelPerformanceStats,
}

/// Performance statistics for a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceStats {
    pub inference_count: u32,
    pub total_inference_time_ms: f64,
    pub average_inference_time_ms: f64,
    pub last_inference_time_ms: f64,
    pub errors: u32,
    pub cache_hits: u32,
    pub cache_misses: u32,
}

/// Model routing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRouting {
    pub route_id: String,
    pub condition: RoutingCondition,
    pub target_model_id: String,
    pub weight: f32, // For A/B testing
    pub enabled: bool,
}

/// Routing conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingCondition {
    /// Route based on input size
    InputSize {
        min_tokens: Option<u32>,
        max_tokens: Option<u32>,
    },
    /// Route based on user segment
    UserSegment { segment: String },
    /// Route based on capability requirement
    Capability { required_capability: String },
    /// Route based on performance requirement
    Performance { max_latency_ms: f64 },
    /// Random routing for A/B testing
    Random { percentage: f32 },
    /// Always route (default)
    Always,
}

/// Multi-model manager configuration
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct MultiModelConfig {
    max_concurrent_models: usize,
    max_memory_usage_mb: u32,
    auto_unload_inactive: bool,
    inactive_timeout_ms: u32,
    enable_preloading: bool,
    enable_model_warming: bool,
    #[allow(dead_code)]
    cache_enabled: bool,
    #[allow(dead_code)]
    performance_monitoring: bool,
}

impl Default for MultiModelConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl MultiModelConfig {
    /// Create a new multi-model configuration
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            max_concurrent_models: 3,
            max_memory_usage_mb: 500,
            auto_unload_inactive: true,
            inactive_timeout_ms: 300_000, // 5 minutes
            enable_preloading: true,
            enable_model_warming: true,
            cache_enabled: true,
            performance_monitoring: true,
        }
    }

    /// Create a configuration optimized for development
    pub fn development() -> Self {
        Self {
            max_concurrent_models: 2,
            max_memory_usage_mb: 200,
            auto_unload_inactive: false,
            inactive_timeout_ms: 600_000, // 10 minutes
            enable_preloading: false,
            enable_model_warming: false,
            cache_enabled: true,
            performance_monitoring: true,
        }
    }

    /// Create a configuration optimized for production
    pub fn production() -> Self {
        Self {
            max_concurrent_models: 5,
            max_memory_usage_mb: 1000,
            auto_unload_inactive: true,
            inactive_timeout_ms: 180_000, // 3 minutes
            enable_preloading: true,
            enable_model_warming: true,
            cache_enabled: true,
            performance_monitoring: false,
        }
    }

    /// Set maximum concurrent models
    pub fn set_max_concurrent_models(mut self, max: usize) -> Self {
        self.max_concurrent_models = max;
        self
    }

    /// Set maximum memory usage in MB
    pub fn set_max_memory_usage_mb(mut self, mb: u32) -> Self {
        self.max_memory_usage_mb = mb;
        self
    }

    /// Enable/disable auto-unloading of inactive models
    pub fn set_auto_unload_inactive(mut self, enabled: bool) -> Self {
        self.auto_unload_inactive = enabled;
        self
    }

    /// Set inactive timeout in milliseconds
    pub fn set_inactive_timeout_ms(mut self, timeout: u32) -> Self {
        self.inactive_timeout_ms = timeout;
        self
    }

    /// Enable/disable model preloading
    pub fn set_enable_preloading(mut self, enabled: bool) -> Self {
        self.enable_preloading = enabled;
        self
    }
}

/// Multi-model manager
#[wasm_bindgen]
pub struct MultiModelManager {
    config: MultiModelConfig,
    models: Vec<LoadedModel>,
    model_registry: Vec<ModelMetadata>,
    routing_rules: Vec<ModelRouting>,
    default_model_id: Option<String>,
    #[cfg(feature = "indexeddb")]
    storage: Option<ModelStorage>,
    quantizer: Option<WebQuantizer>,
    debug_logger: Option<DebugLogger>,
}

#[wasm_bindgen]
impl MultiModelManager {
    /// Create a new multi-model manager
    #[wasm_bindgen(constructor)]
    pub fn new(config: MultiModelConfig) -> Self {
        Self {
            config,
            models: Vec::new(),
            model_registry: Vec::new(),
            routing_rules: Vec::new(),
            default_model_id: None,
            #[cfg(feature = "indexeddb")]
            storage: None,
            quantizer: None,
            debug_logger: None,
        }
    }

    /// Initialize storage for model caching
    #[cfg(feature = "indexeddb")]
    pub async fn initialize_storage(&mut self, max_storage_mb: f64) -> Result<(), JsValue> {
        let mut storage = ModelStorage::new("trustformers-models".to_string(), max_storage_mb);
        storage.initialize().await?;
        self.storage = Some(storage);

        if let Some(ref mut logger) = self.debug_logger {
            logger.info(
                &format!("Initialized model storage ({}MB)", max_storage_mb),
                "multi_model",
            );
        }

        Ok(())
    }

    /// Set debug logger
    pub fn set_debug_logger(&mut self, logger: DebugLogger) {
        self.debug_logger = Some(logger);
    }

    /// Set quantizer for model optimization
    pub fn set_quantizer(&mut self, quantizer: WebQuantizer) {
        self.quantizer = Some(quantizer);
    }

    /// Register a model in the system
    pub fn register_model(&mut self, metadata: &str) -> Result<(), JsValue> {
        let model_metadata: ModelMetadata = serde_json::from_str(metadata)
            .map_err(|e| JsValue::from_str(&format!("Invalid metadata: {}", e)))?;

        // Check if model already exists
        if self.model_registry.iter().any(|m| m.id == model_metadata.id) {
            return Err(JsValue::from_str("Model already registered"));
        }

        self.model_registry.push(model_metadata.clone());

        if let Some(ref mut logger) = self.debug_logger {
            logger.info(
                &format!(
                    "Registered model: {} ({})",
                    model_metadata.name, model_metadata.id
                ),
                "multi_model",
            );
        }

        Ok(())
    }

    /// Load a model by ID
    pub async fn load_model(&mut self, model_id: &str) -> Result<(), JsValue> {
        // Check if model is already loaded
        if self.models.iter().any(|m| m.metadata.id == model_id) {
            return Ok(());
        }

        // Find model metadata
        let metadata = self
            .model_registry
            .iter()
            .find(|m| m.id == model_id)
            .ok_or_else(|| JsValue::from_str("Model not found in registry"))?
            .clone();

        // Check resource availability
        self.ensure_resources_available(&metadata)?;

        if let Some(ref mut logger) = self.debug_logger {
            logger.start_timer(&format!("load_model_{}", model_id));
            logger.info(&format!("Loading model: {}", metadata.name), "multi_model");
        }

        // Create inference session
        let mut session = crate::InferenceSession::new(metadata.model_type.clone())?;
        session.initialize_with_auto_device().await?;

        // Load model data (from cache or URL)
        #[cfg(feature = "indexeddb")]
        if let Some(ref storage) = self.storage {
            if let Some(cached_data) = storage.get_model(model_id).await? {
                session.load_model(&cached_data).await?;
            } else {
                // Load from URL and cache
                if let Some(url) = &metadata.download_url {
                    let model_data = self.fetch_model_from_url(url).await.map_err(|e| {
                        JsValue::from_str(&format!("Failed to fetch model from {}: {:?}", url, e))
                    })?;
                    session.load_model(&model_data).await?;
                    storage
                        .store_model(
                            model_id,
                            &metadata.name,
                            &metadata.architecture,
                            &metadata.version,
                            &model_data,
                        )
                        .await?;
                } else {
                    return Err(JsValue::from_str("No download URL provided for model"));
                }
            }
        } else {
            // Load from URL without caching
            if let Some(url) = &metadata.download_url {
                let model_data = self.fetch_model_from_url(url).await.map_err(|e| {
                    JsValue::from_str(&format!("Failed to fetch model from {}: {:?}", url, e))
                })?;
                session.load_model(&model_data).await?;
            } else {
                return Err(JsValue::from_str("No download URL provided for model"));
            }
        }

        #[cfg(not(feature = "indexeddb"))]
        {
            // Load from URL without caching
            if let Some(url) = &metadata.download_url {
                let model_data = self.fetch_model_from_url(url).await.map_err(|e| {
                    JsValue::from_str(&format!("Failed to fetch model from {}: {:?}", url, e))
                })?;
                session.load_model(&model_data).await?;
            } else {
                return Err(JsValue::from_str("No download URL provided for model"));
            }
        }

        let load_time = Date::now();

        let loaded_model = LoadedModel {
            metadata: metadata.clone(),
            status: ModelStatus::Ready,
            session: Some(session),
            load_time,
            memory_usage: metadata.size_bytes,
            gpu_memory_usage: 0, // Would be calculated
            warmup_completed: false,
            performance_stats: ModelPerformanceStats {
                inference_count: 0,
                total_inference_time_ms: 0.0,
                average_inference_time_ms: 0.0,
                last_inference_time_ms: 0.0,
                errors: 0,
                cache_hits: 0,
                cache_misses: 0,
            },
        };

        self.models.push(loaded_model);

        if let Some(ref mut logger) = self.debug_logger {
            logger.end_timer(&format!("load_model_{}", model_id));
            logger.info(
                &format!("Model loaded successfully: {}", metadata.name),
                "multi_model",
            );
        }

        // Perform warmup if enabled
        if self.config.enable_model_warming {
            self.warmup_model(model_id).await?;
        }

        Ok(())
    }

    /// Unload a model by ID
    pub fn unload_model(&mut self, model_id: &str) -> Result<(), JsValue> {
        if let Some(pos) = self.models.iter().position(|m| m.metadata.id == model_id) {
            let model = self.models.remove(pos);

            if let Some(ref mut logger) = self.debug_logger {
                logger.info(
                    &format!("Unloaded model: {}", model.metadata.name),
                    "multi_model",
                );
            }

            Ok(())
        } else {
            Err(JsValue::from_str("Model not found"))
        }
    }

    /// Get a model for inference by ID
    pub fn get_model_index(&self, model_id: &str) -> Option<usize> {
        self.models.iter().position(|m| m.metadata.id == model_id)
    }

    /// Route a request to the appropriate model
    pub fn route_request(&self, input_context: &str) -> Result<String, JsValue> {
        // Parse input context (simplified)
        let context: serde_json::Value = serde_json::from_str(input_context)
            .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()));

        // Apply routing rules
        for rule in &self.routing_rules {
            if !rule.enabled {
                continue;
            }

            if self.matches_routing_condition(&rule.condition, &context)? {
                return Ok(rule.target_model_id.clone());
            }
        }

        // Use default model if no rules match
        self.default_model_id
            .clone()
            .ok_or_else(|| JsValue::from_str("No default model configured"))
    }

    /// Add a routing rule
    pub fn add_routing_rule(&mut self, rule_json: &str) -> Result<(), JsValue> {
        let rule: ModelRouting = serde_json::from_str(rule_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid routing rule: {}", e)))?;

        self.routing_rules.push(rule);
        Ok(())
    }

    /// Set the default model
    pub fn set_default_model(&mut self, model_id: &str) {
        self.default_model_id = Some(model_id.to_string());
    }

    /// Get list of loaded models
    pub fn get_loaded_models(&self) -> Array {
        let array = Array::new();

        for model in &self.models {
            let obj = Object::new();
            js_sys::Reflect::set(&obj, &"id".into(), &model.metadata.id.clone().into()).unwrap();
            js_sys::Reflect::set(&obj, &"name".into(), &model.metadata.name.clone().into())
                .unwrap();
            js_sys::Reflect::set(
                &obj,
                &"status".into(),
                &format!("{:?}", model.status).into(),
            )
            .unwrap();
            js_sys::Reflect::set(&obj, &"memory_usage".into(), &model.memory_usage.into()).unwrap();
            js_sys::Reflect::set(&obj, &"last_used".into(), &model.metadata.last_used.into())
                .unwrap();
            js_sys::Reflect::set(
                &obj,
                &"usage_count".into(),
                &model.metadata.usage_count.into(),
            )
            .unwrap();
            array.push(&obj);
        }

        array
    }

    /// Get performance statistics for all models
    pub fn get_performance_stats(&self) -> String {
        let stats: Vec<_> = self
            .models
            .iter()
            .map(|m| (m.metadata.id.clone(), m.performance_stats.clone()))
            .collect();

        serde_json::to_string_pretty(&stats).unwrap_or_else(|_| "{}".to_string())
    }

    /// Optimize memory usage by unloading inactive models
    pub fn optimize_memory(&mut self) -> Result<(), JsValue> {
        if !self.config.auto_unload_inactive {
            return Ok(());
        }

        let current_time = Date::now();
        let mut models_to_unload = Vec::new();

        // Find models to unload based on priority and last usage
        for (i, model) in self.models.iter().enumerate() {
            let inactive_time = current_time - model.metadata.last_used;

            // Don't unload critical priority models
            if model.metadata.priority == ModelPriority::Critical {
                continue;
            }

            // Unload if inactive for too long
            if inactive_time > self.config.inactive_timeout_ms as f64 {
                models_to_unload.push((i, model.metadata.id.clone()));
            }
        }

        // Sort by priority (unload low priority first)
        models_to_unload.sort_by(|a, b| {
            let model_a = &self.models[a.0];
            let model_b = &self.models[b.0];
            model_a.metadata.priority.cmp(&model_b.metadata.priority)
        });

        // Unload models
        for (_, model_id) in models_to_unload {
            self.unload_model(&model_id)?;

            if let Some(ref mut logger) = self.debug_logger {
                logger.info(
                    &format!("Auto-unloaded inactive model: {}", model_id),
                    "multi_model",
                );
            }
        }

        Ok(())
    }

    /// Preload models based on usage patterns
    pub async fn preload_models(&mut self) -> Result<(), JsValue> {
        if !self.config.enable_preloading {
            return Ok(());
        }

        // Sort models by usage frequency and priority
        let mut candidates: Vec<_> = self
            .model_registry
            .iter()
            .filter(|m| !self.models.iter().any(|loaded| loaded.metadata.id == m.id))
            .cloned()
            .collect();

        candidates.sort_by(|a, b| {
            let a_score = (a.usage_count as f32) * (a.priority as u8 as f32);
            let b_score = (b.usage_count as f32) * (b.priority as u8 as f32);
            b_score.partial_cmp(&a_score).unwrap()
        });

        // Preload top candidates if we have capacity
        for candidate in
            candidates.iter().take(self.config.max_concurrent_models - self.models.len())
        {
            if self.has_capacity_for_model(candidate) {
                self.load_model(&candidate.id).await?;

                if let Some(ref mut logger) = self.debug_logger {
                    logger.info(
                        &format!("Preloaded model: {}", candidate.name),
                        "multi_model",
                    );
                }
            }
        }

        Ok(())
    }

    /// Get current memory usage across all models
    pub fn get_total_memory_usage(&self) -> usize {
        self.models.iter().map(|m| m.memory_usage).sum()
    }

    /// Get system status summary
    pub fn get_system_status(&self) -> String {
        let total_memory = self.get_total_memory_usage();
        let total_memory_mb = total_memory / (1024 * 1024);

        // Manual JSON construction
        format!(
            r#"{{
  "loaded_models": {},
  "registered_models": {},
  "routing_rules": {},
  "total_memory_usage_mb": {},
  "max_memory_mb": {},
  "memory_utilization": {:.3},
  "default_model": {}
}}"#,
            self.models.len(),
            self.model_registry.len(),
            self.routing_rules.len(),
            total_memory_mb,
            self.config.max_memory_usage_mb,
            (total_memory_mb as f32) / (self.config.max_memory_usage_mb as f32),
            if let Some(ref default) = self.default_model_id {
                format!("\"{}\"", default)
            } else {
                "null".to_string()
            }
        )
    }

    // Private helper methods

    async fn warmup_model(&mut self, model_id: &str) -> Result<(), JsValue> {
        if let Some(model) = self.models.iter_mut().find(|m| m.metadata.id == model_id) {
            model.status = ModelStatus::WarmingUp;

            // Perform a dummy inference to warm up the model
            // This would use actual inference in a real implementation

            model.status = ModelStatus::Ready;
            model.warmup_completed = true;

            if let Some(ref mut logger) = self.debug_logger {
                logger.info(
                    &format!("Model warmed up: {}", model.metadata.name),
                    "multi_model",
                );
            }
        }

        Ok(())
    }

    fn ensure_resources_available(&self, metadata: &ModelMetadata) -> Result<(), JsValue> {
        // Check if we're at capacity
        if self.models.len() >= self.config.max_concurrent_models {
            return Err(JsValue::from_str("Maximum concurrent models reached"));
        }

        // Check memory requirements
        let current_memory = self.get_total_memory_usage() / (1024 * 1024); // Convert to MB
        let required_memory = current_memory + (metadata.size_bytes / (1024 * 1024));

        if required_memory > self.config.max_memory_usage_mb as usize {
            return Err(JsValue::from_str("Insufficient memory for model"));
        }

        Ok(())
    }

    fn has_capacity_for_model(&self, metadata: &ModelMetadata) -> bool {
        if self.models.len() >= self.config.max_concurrent_models {
            return false;
        }

        let current_memory = self.get_total_memory_usage() / (1024 * 1024);
        let required_memory = current_memory + (metadata.size_bytes / (1024 * 1024));

        required_memory <= self.config.max_memory_usage_mb as usize
    }

    fn matches_routing_condition(
        &self,
        condition: &RoutingCondition,
        context: &serde_json::Value,
    ) -> Result<bool, JsValue> {
        match condition {
            RoutingCondition::InputSize {
                min_tokens,
                max_tokens,
            } => {
                let token_count =
                    context.get("token_count").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

                if let Some(min) = min_tokens {
                    if token_count < *min {
                        return Ok(false);
                    }
                }

                if let Some(max) = max_tokens {
                    if token_count > *max {
                        return Ok(false);
                    }
                }

                Ok(true)
            },
            RoutingCondition::UserSegment { segment } => {
                let user_segment =
                    context.get("user_segment").and_then(|v| v.as_str()).unwrap_or("");
                Ok(user_segment == segment)
            },
            RoutingCondition::Capability {
                required_capability,
            } => {
                let empty_vec = vec![];
                let capabilities = context
                    .get("required_capabilities")
                    .and_then(|v| v.as_array())
                    .unwrap_or(&empty_vec);

                Ok(capabilities
                    .iter()
                    .any(|cap| cap.as_str().is_some_and(|s| s == required_capability)))
            },
            RoutingCondition::Performance { max_latency_ms } => {
                let required_latency =
                    context.get("max_latency_ms").and_then(|v| v.as_f64()).unwrap_or(f64::INFINITY);
                Ok(required_latency <= *max_latency_ms)
            },
            RoutingCondition::Random { percentage } => {
                let random_value = (Date::now() % 100.0) / 100.0;
                Ok(random_value < (*percentage / 100.0) as f64)
            },
            RoutingCondition::Always => Ok(true),
        }
    }

    /// Fetch model data from a remote URL using the Fetch API
    async fn fetch_model_from_url(&self, url: &str) -> Result<Vec<u8>, JsValue> {
        use wasm_bindgen::JsCast;
        use wasm_bindgen_futures::JsFuture;

        // Create fetch request
        let request = web_sys::Request::new_with_str(url)?;
        request.headers().set("Accept", "application/octet-stream")?;

        // Get the global window object
        let window = web_sys::window().ok_or("No global window object")?;

        // Perform the fetch
        let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;
        let resp: web_sys::Response = resp_value.dyn_into()?;

        // Check if the response is ok
        if !resp.ok() {
            return Err(JsValue::from_str(&format!(
                "HTTP error: {} {}",
                resp.status(),
                resp.status_text()
            )));
        }

        // Get the response as ArrayBuffer
        let array_buffer = JsFuture::from(resp.array_buffer()?).await?;
        let uint8_array = js_sys::Uint8Array::new(&array_buffer);

        // Convert to Vec<u8>
        let mut data = vec![0u8; uint8_array.length() as usize];
        uint8_array.copy_to(&mut data);

        web_sys::console::log_1(
            &format!(
                "Successfully downloaded {} bytes for MultiModelManager",
                data.len()
            )
            .into(),
        );

        Ok(data)
    }
}

/// Create a multi-model manager with development settings
#[wasm_bindgen]
pub fn create_development_multi_model_manager() -> MultiModelManager {
    MultiModelManager::new(MultiModelConfig::development())
}

/// Create a multi-model manager with production settings
#[wasm_bindgen]
pub fn create_production_multi_model_manager() -> MultiModelManager {
    MultiModelManager::new(MultiModelConfig::production())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_model_config() {
        let config = MultiModelConfig::development();
        assert_eq!(config.max_concurrent_models, 2);
        assert_eq!(config.max_memory_usage_mb, 200);

        let prod_config = MultiModelConfig::production();
        assert_eq!(prod_config.max_concurrent_models, 5);
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_model_registration() {
        let config = MultiModelConfig::new();
        let mut manager = MultiModelManager::new(config);

        let metadata = ModelMetadata {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            version: "1.0".to_string(),
            description: "A test model".to_string(),
            model_type: "transformer".to_string(),
            architecture: "bert".to_string(),
            size_bytes: 1024 * 1024,
            priority: ModelPriority::Normal,
            tags: vec!["test".to_string()],
            environment: DeploymentEnvironment::Development,
            created_at: Date::now(),
            last_used: Date::now(),
            usage_count: 0,
            capabilities: vec!["text-generation".to_string()],
            requirements: ModelRequirements {
                min_memory_mb: 100,
                min_gpu_memory_mb: 0,
                requires_gpu: false,
                requires_webgpu: false,
                min_cpu_cores: 1,
                recommended_batch_size: 1,
            },
            download_url: Some("https://example.com/models/test-model.bin".to_string()),
        };

        let metadata_json = serde_json::to_string(&metadata).unwrap();
        let result = manager.register_model(&metadata_json);
        assert!(result.is_ok());
        assert_eq!(manager.model_registry.len(), 1);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_model_manager_config() {
        // Test only configuration creation for non-WASM targets
        let config = MultiModelConfig::new();
        let manager = MultiModelManager::new(config);
        assert_eq!(manager.model_registry.len(), 0);
        assert!(manager.routing_rules.is_empty());
    }

    #[test]
    fn test_routing_conditions() {
        let manager = MultiModelManager::new(MultiModelConfig::new());

        let condition = RoutingCondition::InputSize {
            min_tokens: Some(10),
            max_tokens: Some(100),
        };
        let context: serde_json::Value = serde_json::from_str(r#"{"token_count": 50}"#).unwrap();

        let matches = manager.matches_routing_condition(&condition, &context).unwrap();
        assert!(matches);

        let context_too_small: serde_json::Value =
            serde_json::from_str(r#"{"token_count": 5}"#).unwrap();
        let matches_small =
            manager.matches_routing_condition(&condition, &context_too_small).unwrap();
        assert!(!matches_small);
    }
}
