// Plugin Interface - Core plugin traits and types
//
// This module defines the core interfaces that all plugins must implement
// to integrate with the TrustformeRS WASM plugin system.

use std::boxed::Box;
use std::string::{String, ToString};
use std::vec::Vec;
use core::fmt;
use serde::{Deserialize, Serialize};

/// Core plugin trait that all plugins must implement
#[async_trait::async_trait(?Send)]
pub trait Plugin: fmt::Debug {
    /// Get plugin capabilities
    fn capabilities(&self) -> PluginCapabilities;

    /// Get plugin configuration
    fn config(&self) -> &PluginConfig;

    /// Initialize the plugin with given context
    async fn initialize(&mut self, context: PluginContext) -> Result<(), PluginError>;

    /// Execute a plugin function
    async fn execute(&mut self, function_name: &str, context: PluginContext) -> Result<serde_json::Value, PluginError>;

    /// Cleanup plugin resources
    async fn cleanup(&mut self) -> Result<(), PluginError>;

    /// Check if plugin is initialized
    fn is_initialized(&self) -> bool;

    /// Get plugin version
    fn version(&self) -> &str;

    /// Get plugin dependencies
    fn dependencies(&self) -> Vec<String>;
}

/// Plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    pub id: String,
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub plugin_type: PluginType,
    pub capabilities: PluginCapabilities,
    pub settings: serde_json::Value,
    pub dependencies: Vec<String>,
    pub optional_dependencies: Vec<String>,
}

impl PluginConfig {
    /// Create a new plugin configuration
    pub fn new(
        id: String,
        name: String,
        version: String,
        plugin_type: PluginType,
    ) -> Self {
        Self {
            id,
            name,
            version,
            description: String::new(),
            author: String::new(),
            plugin_type,
            capabilities: PluginCapabilities::default(),
            settings: serde_json::Value::Null,
            dependencies: Vec::new(),
            optional_dependencies: Vec::new(),
        }
    }

    /// Set plugin description
    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }

    /// Set plugin author
    pub fn with_author(mut self, author: String) -> Self {
        self.author = author;
        self
    }

    /// Set plugin capabilities
    pub fn with_capabilities(mut self, capabilities: PluginCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Set plugin settings
    pub fn with_settings(mut self, settings: serde_json::Value) -> Self {
        self.settings = settings;
        self
    }

    /// Add a dependency
    pub fn with_dependency(mut self, dependency: String) -> Self {
        self.dependencies.push(dependency);
        self
    }

    /// Add an optional dependency
    pub fn with_optional_dependency(mut self, dependency: String) -> Self {
        self.optional_dependencies.push(dependency);
        self
    }
}

/// Plugin capabilities and features
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PluginCapabilities {
    pub functions: Vec<String>,
    pub permissions: Vec<PluginPermission>,
    pub supported_formats: Vec<String>,
    pub hardware_requirements: HardwareRequirements,
    pub memory_requirements: MemoryRequirements,
    pub api_version: String,
}

impl Default for PluginCapabilities {
    fn default() -> Self {
        Self {
            functions: Vec::new(),
            permissions: Vec::new(),
            supported_formats: Vec::new(),
            hardware_requirements: HardwareRequirements::default(),
            memory_requirements: MemoryRequirements::default(),
            api_version: "1.0.0".to_string(),
        }
    }
}

impl PluginCapabilities {
    /// Create new capabilities
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a function capability
    pub fn with_function(mut self, function: String) -> Self {
        self.functions.push(function);
        self
    }

    /// Add a permission requirement
    pub fn with_permission(mut self, permission: PluginPermission) -> Self {
        self.permissions.push(permission);
        self
    }

    /// Add a supported format
    pub fn with_format(mut self, format: String) -> Self {
        self.supported_formats.push(format);
        self
    }

    /// Set hardware requirements
    pub fn with_hardware_requirements(mut self, requirements: HardwareRequirements) -> Self {
        self.hardware_requirements = requirements;
        self
    }

    /// Set memory requirements
    pub fn with_memory_requirements(mut self, requirements: MemoryRequirements) -> Self {
        self.memory_requirements = requirements;
        self
    }

    /// Check if function is supported
    pub fn supports_function(&self, function: &str) -> bool {
        self.functions.iter().any(|f| f == function)
    }

    /// Check if permission is required
    pub fn requires_permission(&self, permission: &PluginPermission) -> bool {
        self.permissions.contains(permission)
    }

    /// Check if format is supported
    pub fn supports_format(&self, format: &str) -> bool {
        self.supported_formats.iter().any(|f| f == format)
    }
}

/// Plugin types for categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PluginType {
    ModelLoader,
    Tokenizer,
    Optimizer,
    Preprocessor,
    Postprocessor,
    Visualizer,
    ExportHandler,
    CustomOperation,
    UIComponent,
    DataLoader,
    Middleware,
    Extension,
}

impl fmt::Display for PluginType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            PluginType::ModelLoader => "model-loader",
            PluginType::Tokenizer => "tokenizer",
            PluginType::Optimizer => "optimizer",
            PluginType::Preprocessor => "preprocessor",
            PluginType::Postprocessor => "postprocessor",
            PluginType::Visualizer => "visualizer",
            PluginType::ExportHandler => "export-handler",
            PluginType::CustomOperation => "custom-operation",
            PluginType::UIComponent => "ui-component",
            PluginType::DataLoader => "data-loader",
            PluginType::Middleware => "middleware",
            PluginType::Extension => "extension",
        };
        write!(f, "{}", name)
    }
}

/// Plugin permissions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PluginPermission {
    FileAccess,
    NetworkAccess,
    StorageAccess,
    WebGLAccess,
    WebGPUAccess,
    CameraAccess,
    MicrophoneAccess,
    NotificationAccess,
    GeolocationAccess,
    DeviceMemoryAccess,
    SystemInfoAccess,
    CryptographyAccess,
}

/// Hardware requirements for plugins
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HardwareRequirements {
    pub webgl_required: bool,
    pub webgpu_required: bool,
    pub simd_required: bool,
    pub threads_required: bool,
    pub min_memory_mb: u32,
    pub gpu_memory_mb: Option<u32>,
    pub supported_architectures: Vec<String>,
}

impl Default for HardwareRequirements {
    fn default() -> Self {
        Self {
            webgl_required: false,
            webgpu_required: false,
            simd_required: false,
            threads_required: false,
            min_memory_mb: 128,
            gpu_memory_mb: None,
            supported_architectures: vec!["wasm32".to_string()],
        }
    }
}

/// Memory requirements for plugins
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryRequirements {
    pub min_heap_mb: u32,
    pub max_heap_mb: Option<u32>,
    pub stack_size_kb: u32,
    pub shared_memory: bool,
    pub persistent_storage_mb: Option<u32>,
}

impl Default for MemoryRequirements {
    fn default() -> Self {
        Self {
            min_heap_mb: 64,
            max_heap_mb: None,
            stack_size_kb: 512,
            shared_memory: false,
            persistent_storage_mb: None,
        }
    }
}

/// Context passed to plugin functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginContext {
    pub session_id: String,
    pub request_id: String,
    pub parameters: serde_json::Value,
    pub metadata: serde_json::Value,
    pub permissions: Vec<PluginPermission>,
    pub resources: ContextResources,
}

impl PluginContext {
    /// Create a new plugin context
    pub fn new(session_id: String, request_id: String) -> Self {
        Self {
            session_id,
            request_id,
            parameters: serde_json::Value::Object(serde_json::Map::new()),
            metadata: serde_json::Value::Object(serde_json::Map::new()),
            permissions: Vec::new(),
            resources: ContextResources::default(),
        }
    }

    /// Set parameters
    pub fn with_parameters(mut self, parameters: serde_json::Value) -> Self {
        self.parameters = parameters;
        self
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// Add a permission
    pub fn with_permission(mut self, permission: PluginPermission) -> Self {
        self.permissions.push(permission);
        self
    }

    /// Set resources
    pub fn with_resources(mut self, resources: ContextResources) -> Self {
        self.resources = resources;
        self
    }

    /// Get parameter by key
    pub fn get_parameter(&self, key: &str) -> Option<&serde_json::Value> {
        self.parameters.get(key)
    }

    /// Get metadata by key
    pub fn get_metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.metadata.get(key)
    }

    /// Check if permission is granted
    pub fn has_permission(&self, permission: &PluginPermission) -> bool {
        self.permissions.contains(permission)
    }
}

/// Resources available to plugins
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ContextResources {
    pub memory_limit_mb: Option<u32>,
    pub time_limit_ms: Option<u32>,
    pub temporary_storage_mb: Option<u32>,
    pub concurrent_requests: Option<u32>,
    pub allowed_origins: Vec<String>,
}

/// Plugin error types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginError {
    NotInitialized,
    AlreadyInitialized,
    InvalidFunction(String),
    PermissionDenied(PluginPermission),
    ResourceExhausted(String),
    InvalidParameter(String),
    ExecutionFailed(String),
    InitializationFailed(String),
    CleanupFailed(String),
    DependencyMissing(String),
    VersionMismatch(String, String), // required, actual
    HardwareUnsupported(String),
    MemoryInsufficient(u32, u32), // required, available
}

impl fmt::Display for PluginError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PluginError::NotInitialized => write!(f, "Plugin not initialized"),
            PluginError::AlreadyInitialized => write!(f, "Plugin already initialized"),
            PluginError::InvalidFunction(func) => write!(f, "Invalid function: {}", func),
            PluginError::PermissionDenied(perm) => write!(f, "Permission denied: {:?}", perm),
            PluginError::ResourceExhausted(resource) => write!(f, "Resource exhausted: {}", resource),
            PluginError::InvalidParameter(param) => write!(f, "Invalid parameter: {}", param),
            PluginError::ExecutionFailed(msg) => write!(f, "Execution failed: {}", msg),
            PluginError::InitializationFailed(msg) => write!(f, "Initialization failed: {}", msg),
            PluginError::CleanupFailed(msg) => write!(f, "Cleanup failed: {}", msg),
            PluginError::DependencyMissing(dep) => write!(f, "Dependency missing: {}", dep),
            PluginError::VersionMismatch(req, actual) => write!(f, "Version mismatch: required {}, actual {}", req, actual),
            PluginError::HardwareUnsupported(hw) => write!(f, "Hardware unsupported: {}", hw),
            PluginError::MemoryInsufficient(req, avail) => write!(f, "Memory insufficient: required {} MB, available {} MB", req, avail),
        }
    }
}

impl std::error::Error for PluginError {}

/// Macro for creating plugin configurations
#[macro_export]
macro_rules! plugin_config {
    ($id:expr, $name:expr, $version:expr, $type:expr) => {
        PluginConfig::new($id.to_string(), $name.to_string(), $version.to_string(), $type)
    };
    ($id:expr, $name:expr, $version:expr, $type:expr, { $($key:ident: $value:expr),* }) => {
        {
            let mut config = PluginConfig::new($id.to_string(), $name.to_string(), $version.to_string(), $type);
            $(
                config = match stringify!($key) {
                    "description" => config.with_description($value.to_string()),
                    "author" => config.with_author($value.to_string()),
                    _ => config,
                };
            )*
            config
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_config_creation() {
        let config = PluginConfig::new(
            "test-plugin".to_string(),
            "Test Plugin".to_string(),
            "1.0.0".to_string(),
            PluginType::Extension,
        );

        assert_eq!(config.id, "test-plugin");
        assert_eq!(config.name, "Test Plugin");
        assert_eq!(config.version, "1.0.0");
        assert_eq!(config.plugin_type, PluginType::Extension);
    }

    #[test]
    fn test_plugin_capabilities() {
        let capabilities = PluginCapabilities::new()
            .with_function("process".to_string())
            .with_permission(PluginPermission::FileAccess)
            .with_format("json".to_string());

        assert!(capabilities.supports_function("process"));
        assert!(capabilities.requires_permission(&PluginPermission::FileAccess));
        assert!(capabilities.supports_format("json"));
        assert!(!capabilities.supports_function("invalid"));
    }

    #[test]
    fn test_plugin_context() {
        let context = PluginContext::new("session-1".to_string(), "request-1".to_string())
            .with_permission(PluginPermission::NetworkAccess);

        assert_eq!(context.session_id, "session-1");
        assert_eq!(context.request_id, "request-1");
        assert!(context.has_permission(&PluginPermission::NetworkAccess));
        assert!(!context.has_permission(&PluginPermission::FileAccess));
    }

    #[test]
    fn test_plugin_type_display() {
        assert_eq!(format!("{}", PluginType::ModelLoader), "model-loader");
        assert_eq!(format!("{}", PluginType::Tokenizer), "tokenizer");
        assert_eq!(format!("{}", PluginType::Extension), "extension");
    }
}