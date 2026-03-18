// Plugin Framework for TrustformeRS WASM
// Enables community extensions and custom functionality

#![allow(clippy::missing_enforced_import_renames)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wasm_bindgen::prelude::*;

/// Plugin trait that all plugins must implement
pub trait Plugin: Send + Sync {
    /// Plugin metadata
    fn metadata(&self) -> PluginMetadata;

    /// Initialize the plugin
    fn initialize(&mut self, config: PluginConfig) -> Result<(), PluginError>;

    /// Execute plugin functionality
    fn execute(&self, context: &PluginContext) -> Result<PluginResult, PluginError>;

    /// Cleanup resources when plugin is unloaded
    fn cleanup(&mut self);
}

/// Plugin metadata information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    pub name: String,
    pub version: String,
    pub author: String,
    pub description: String,
    pub plugin_type: PluginType,
    pub dependencies: Vec<String>,
    pub permissions: Vec<PluginPermission>,
}

/// Types of plugins supported
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PluginType {
    /// Preprocessing plugins (tokenization, data processing)
    Preprocessor,
    /// Model inference plugins (custom model formats)
    InferenceEngine,
    /// Postprocessing plugins (output formatting, analysis)
    Postprocessor,
    /// Optimization plugins (quantization, pruning)
    Optimizer,
    /// Visualization plugins (charts, graphs)
    Visualizer,
    /// Storage plugins (custom backends)
    Storage,
    /// Network plugins (custom protocols)
    Network,
    /// Utility plugins (general purpose)
    Utility,
}

/// Plugin permissions system
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PluginPermission {
    /// Access to read model data
    ReadModelData,
    /// Access to modify model data
    WriteModelData,
    /// Access to network operations
    NetworkAccess,
    /// Access to local storage
    StorageAccess,
    /// Access to GPU resources
    GpuAccess,
    /// Access to performance profiling
    ProfilingAccess,
    /// Access to debug information
    DebugAccess,
    /// Access to user interface
    UiAccess,
}

/// Plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    pub settings: HashMap<String, String>,
    pub enabled_features: Vec<String>,
    pub resource_limits: ResourceLimits,
}

/// Resource limits for plugins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_memory_mb: Option<usize>,
    pub max_execution_time_ms: Option<u64>,
    pub max_network_requests: Option<usize>,
    pub max_gpu_memory_mb: Option<usize>,
}

/// Plugin execution context
#[derive(Debug)]
pub struct PluginContext {
    pub plugin_id: String,
    pub session_id: String,
    pub request_data: HashMap<String, String>,
    pub model_metadata: Option<ModelMetadata>,
    pub performance_budget: PerformanceBudget,
}

/// Serializable version of PluginContext for JavaScript interop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializablePluginContext {
    pub plugin_id: String,
    pub session_id: String,
    pub request_data: HashMap<String, String>,
    pub model_metadata: Option<SerializableModelMetadata>,
    pub performance_budget: SerializablePerformanceBudget,
}

/// Model metadata for plugin context
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub model_type: String,
    pub size_mb: f64,
    pub architecture: String,
    pub precision: String,
}

/// Serializable version of ModelMetadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableModelMetadata {
    pub model_type: String,
    pub size_mb: f64,
    pub architecture: String,
    pub precision: String,
}

/// Performance budget for plugin execution
#[derive(Debug, Clone)]
pub struct PerformanceBudget {
    pub max_latency_ms: u64,
    pub max_memory_mb: usize,
    pub priority: ExecutionPriority,
}

/// Serializable version of PerformanceBudget
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializablePerformanceBudget {
    pub max_latency_ms: u64,
    pub max_memory_mb: usize,
    pub priority: SerializableExecutionPriority,
}

/// Execution priority levels
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionPriority {
    Critical,
    High,
    Normal,
    Low,
    Background,
}

/// Serializable version of ExecutionPriority
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SerializableExecutionPriority {
    Critical,
    High,
    Normal,
    Low,
    Background,
}

// Conversion implementations
impl From<ExecutionPriority> for SerializableExecutionPriority {
    fn from(priority: ExecutionPriority) -> Self {
        match priority {
            ExecutionPriority::Critical => SerializableExecutionPriority::Critical,
            ExecutionPriority::High => SerializableExecutionPriority::High,
            ExecutionPriority::Normal => SerializableExecutionPriority::Normal,
            ExecutionPriority::Low => SerializableExecutionPriority::Low,
            ExecutionPriority::Background => SerializableExecutionPriority::Background,
        }
    }
}

impl From<ModelMetadata> for SerializableModelMetadata {
    fn from(metadata: ModelMetadata) -> Self {
        SerializableModelMetadata {
            model_type: metadata.model_type,
            size_mb: metadata.size_mb,
            architecture: metadata.architecture,
            precision: metadata.precision,
        }
    }
}

impl From<PerformanceBudget> for SerializablePerformanceBudget {
    fn from(budget: PerformanceBudget) -> Self {
        SerializablePerformanceBudget {
            max_latency_ms: budget.max_latency_ms,
            max_memory_mb: budget.max_memory_mb,
            priority: budget.priority.into(),
        }
    }
}

impl From<PluginContext> for SerializablePluginContext {
    fn from(context: PluginContext) -> Self {
        SerializablePluginContext {
            plugin_id: context.plugin_id,
            session_id: context.session_id,
            request_data: context.request_data,
            model_metadata: context.model_metadata.map(|m| m.into()),
            performance_budget: context.performance_budget.into(),
        }
    }
}

/// Plugin execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginResult {
    pub success: bool,
    pub data: HashMap<String, String>,
    pub metrics: ExecutionMetrics,
    pub messages: Vec<String>,
}

/// Execution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    pub execution_time_ms: f64,
    pub memory_used_mb: f64,
    pub cpu_usage_percent: f64,
    pub gpu_memory_used_mb: Option<f64>,
}

/// Plugin errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginError {
    pub code: PluginErrorCode,
    pub message: String,
    pub details: Option<String>,
}

/// Plugin error codes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PluginErrorCode {
    InitializationFailed,
    ExecutionFailed,
    PermissionDenied,
    ResourceExhausted,
    InvalidConfiguration,
    DependencyMissing,
    UnsupportedOperation,
    Internal,
}

/// Plugin registry for managing loaded plugins
pub struct PluginRegistry {
    plugins: Arc<Mutex<HashMap<String, Box<dyn Plugin>>>>,
    enabled_plugins: Arc<Mutex<Vec<String>>>,
    plugin_configs: Arc<Mutex<HashMap<String, PluginConfig>>>,
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginRegistry {
    /// Create a new plugin registry
    pub fn new() -> Self {
        Self {
            plugins: Arc::new(Mutex::new(HashMap::new())),
            enabled_plugins: Arc::new(Mutex::new(Vec::new())),
            plugin_configs: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Register a new plugin
    pub fn register_plugin(
        &self,
        plugin_id: String,
        plugin: Box<dyn Plugin>,
        config: PluginConfig,
    ) -> Result<(), PluginError> {
        // Validate plugin metadata
        let metadata = plugin.metadata();
        self.validate_plugin_metadata(&metadata)?;

        // Check dependencies
        self.check_plugin_dependencies(&metadata.dependencies)?;

        // Store plugin and configuration
        {
            let mut plugins = self.plugins.lock().expect("lock should not be poisoned");
            plugins.insert(plugin_id.clone(), plugin);
        }

        {
            let mut configs = self.plugin_configs.lock().expect("lock should not be poisoned");
            configs.insert(plugin_id, config);
        }

        Ok(())
    }

    /// Enable a plugin
    pub fn enable_plugin(&self, plugin_id: &str) -> Result<(), PluginError> {
        // Check if plugin exists
        {
            let plugins = self.plugins.lock().expect("lock should not be poisoned");
            if !plugins.contains_key(plugin_id) {
                return Err(PluginError {
                    code: PluginErrorCode::DependencyMissing,
                    message: format!("Plugin '{plugin_id}' not found"),
                    details: None,
                });
            }
        }

        // Initialize plugin
        {
            let mut plugins = self.plugins.lock().expect("lock should not be poisoned");
            let configs = self.plugin_configs.lock().expect("lock should not be poisoned");

            if let (Some(plugin), Some(config)) =
                (plugins.get_mut(plugin_id), configs.get(plugin_id))
            {
                plugin.initialize(config.clone())?;
            }
        }

        // Add to enabled list
        {
            let mut enabled = self.enabled_plugins.lock().expect("lock should not be poisoned");
            if !enabled.contains(&plugin_id.to_string()) {
                enabled.push(plugin_id.to_string());
            }
        }

        Ok(())
    }

    /// Disable a plugin
    pub fn disable_plugin(&self, plugin_id: &str) -> Result<(), PluginError> {
        // Remove from enabled list
        {
            let mut enabled = self.enabled_plugins.lock().expect("lock should not be poisoned");
            enabled.retain(|id| id != plugin_id);
        }

        // Cleanup plugin
        {
            let mut plugins = self.plugins.lock().expect("lock should not be poisoned");
            if let Some(plugin) = plugins.get_mut(plugin_id) {
                plugin.cleanup();
            }
        }

        Ok(())
    }

    /// Execute plugins of a specific type
    pub fn execute_plugins(
        &self,
        plugin_type: PluginType,
        context: &PluginContext,
    ) -> Vec<Result<PluginResult, PluginError>> {
        let enabled = self.enabled_plugins.lock().expect("lock should not be poisoned").clone();
        let plugins = self.plugins.lock().expect("lock should not be poisoned");

        let mut results = Vec::new();

        for plugin_id in enabled {
            if let Some(plugin) = plugins.get(&plugin_id) {
                let metadata = plugin.metadata();
                if metadata.plugin_type == plugin_type {
                    let result = plugin.execute(context);
                    results.push(result);
                }
            }
        }

        results
    }

    /// Get enabled plugins
    pub fn get_enabled_plugins(&self) -> Vec<String> {
        self.enabled_plugins.lock().expect("lock should not be poisoned").clone()
    }

    /// Get plugin metadata
    pub fn get_plugin_metadata(&self, plugin_id: &str) -> Option<PluginMetadata> {
        let plugins = self.plugins.lock().expect("lock should not be poisoned");
        plugins.get(plugin_id).map(|plugin| plugin.metadata())
    }

    /// List all registered plugins
    pub fn list_plugins(&self) -> Vec<PluginMetadata> {
        let plugins = self.plugins.lock().expect("lock should not be poisoned");
        plugins.values().map(|plugin| plugin.metadata()).collect()
    }

    fn validate_plugin_metadata(&self, metadata: &PluginMetadata) -> Result<(), PluginError> {
        if metadata.name.is_empty() {
            return Err(PluginError {
                code: PluginErrorCode::InvalidConfiguration,
                message: "Plugin name cannot be empty".to_string(),
                details: None,
            });
        }

        if metadata.version.is_empty() {
            return Err(PluginError {
                code: PluginErrorCode::InvalidConfiguration,
                message: "Plugin version cannot be empty".to_string(),
                details: None,
            });
        }

        Ok(())
    }

    fn check_plugin_dependencies(&self, dependencies: &[String]) -> Result<(), PluginError> {
        let plugins = self.plugins.lock().expect("lock should not be poisoned");

        for dep in dependencies {
            if !plugins.contains_key(dep) {
                return Err(PluginError {
                    code: PluginErrorCode::DependencyMissing,
                    message: format!("Missing dependency: {dep}"),
                    details: None,
                });
            }
        }

        Ok(())
    }
}

/// WASM bindings for plugin framework
#[wasm_bindgen]
pub struct PluginManager {
    registry: PluginRegistry,
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl PluginManager {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            registry: PluginRegistry::new(),
        }
    }

    /// Get list of enabled plugins as JSON
    #[wasm_bindgen(getter)]
    pub fn enabled_plugins(&self) -> String {
        serde_json::to_string(&self.registry.get_enabled_plugins()).unwrap_or_default()
    }

    /// Get list of all plugins as JSON
    pub fn list_plugins(&self) -> String {
        serde_json::to_string(&self.registry.list_plugins()).unwrap_or_default()
    }

    /// Enable a plugin by ID
    pub fn enable_plugin(&self, plugin_id: &str) -> Result<(), JsValue> {
        self.registry
            .enable_plugin(plugin_id)
            .map_err(|e| JsValue::from_str(&e.message))
    }

    /// Disable a plugin by ID
    pub fn disable_plugin(&self, plugin_id: &str) -> Result<(), JsValue> {
        self.registry
            .disable_plugin(plugin_id)
            .map_err(|e| JsValue::from_str(&e.message))
    }

    /// Get plugin metadata as JSON
    pub fn get_plugin_metadata(&self, plugin_id: &str) -> Option<String> {
        self.registry
            .get_plugin_metadata(plugin_id)
            .and_then(|metadata| serde_json::to_string(&metadata).ok())
    }
}

/// Default resource limits for plugins
impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: Some(100),
            max_execution_time_ms: Some(5000),
            max_network_requests: Some(10),
            max_gpu_memory_mb: Some(50),
        }
    }
}

/// Default plugin configuration
impl Default for PluginConfig {
    fn default() -> Self {
        Self {
            settings: HashMap::new(),
            enabled_features: Vec::new(),
            resource_limits: ResourceLimits::default(),
        }
    }
}

/// Utility functions for creating plugin configs
#[wasm_bindgen]
pub fn create_default_plugin_config() -> String {
    serde_json::to_string(&PluginConfig::default()).unwrap_or_default()
}

#[wasm_bindgen]
pub fn create_plugin_context(plugin_id: &str, session_id: &str, request_data: &str) -> String {
    let request_map: HashMap<String, String> =
        serde_json::from_str(request_data).unwrap_or_default();

    let context = PluginContext {
        plugin_id: plugin_id.to_string(),
        session_id: session_id.to_string(),
        request_data: request_map,
        model_metadata: Some(ModelMetadata {
            model_type: "transformer".to_string(),
            size_mb: 125.5,
            architecture: "gpt-2".to_string(),
            precision: "fp16".to_string(),
        }),
        performance_budget: PerformanceBudget {
            max_latency_ms: 1000,
            max_memory_mb: 50,
            priority: ExecutionPriority::Normal,
        },
    };

    // Convert to serializable version and serialize to JSON
    let serializable_context: SerializablePluginContext = context.into();
    serde_json::to_string(&serializable_context).unwrap_or_else(|e| {
        web_sys::console::log_1(&format!("Failed to serialize plugin context: {}", e).into());
        format!(
            r#"{{"plugin_id":"{}","session_id":"{}","error":"serialization_failed"}}"#,
            plugin_id, session_id
        )
    })
}
