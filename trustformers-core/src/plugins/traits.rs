//! Core plugin traits and interfaces.

use crate::errors::Result;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;

/// Core trait for plugin implementations.
///
/// The `Plugin` trait provides metadata, configuration, and lifecycle management
/// capabilities for dynamic plugin systems. Plugins can provide Layer functionality
/// through the `forward` method.
///
/// # Requirements
///
/// Plugins must be:
/// - Thread-safe (`Send + Sync`)
/// - Cloneable for multi-instance usage
///
/// # Example
///
/// ```no_run
/// use trustformers_core::plugins::Plugin;
/// use trustformers_core::tensor::Tensor;
/// use trustformers_core::error::Result;
/// use std::collections::HashMap;
///
/// #[derive(Debug, Clone)]
/// struct CustomAttentionPlugin {
///     hidden_size: usize,
///     num_heads: usize,
///     config: HashMap<String, serde_json::Value>,
/// }
///
/// impl Plugin for CustomAttentionPlugin {
///     fn name(&self) -> &str {
///         "custom_attention"
///     }
///
///     fn version(&self) -> &str {
///         "1.0.0"
///     }
///
///     fn description(&self) -> &str {
///         "A custom attention mechanism with optimized kernels"
///     }
///
///     fn configure(&mut self, config: HashMap<String, serde_json::Value>) -> Result<()> {
///         self.config = config;
///         Ok(())
///     }
///
///     fn get_config(&self) -> &HashMap<String, serde_json::Value> {
///         &self.config
///     }
///
///     fn forward(&self, input: Tensor) -> Result<Tensor> {
///         // Custom attention implementation
///         Ok(input) // Simplified
///     }
///
///     fn as_any(&self) -> &dyn std::any::Any {
///         self
///     }
/// }
/// ```
pub trait Plugin: Send + Sync + ClonePlugin + std::fmt::Debug {
    /// Returns the plugin's unique name.
    ///
    /// The name should be unique across all plugins and follow a consistent
    /// naming convention (e.g., "vendor.component_name" or "custom_attention").
    ///
    /// # Returns
    ///
    /// A string slice containing the plugin name.
    fn name(&self) -> &str;

    /// Returns the plugin's version string.
    ///
    /// Should follow semantic versioning (e.g., "1.2.3").
    ///
    /// # Returns
    ///
    /// A string slice containing the version.
    fn version(&self) -> &str;

    /// Returns a human-readable description of the plugin.
    ///
    /// # Returns
    ///
    /// A string slice describing the plugin's functionality.
    fn description(&self) -> &str;

    /// Configures the plugin with the provided parameters.
    ///
    /// # Arguments
    ///
    /// * `config` - A map of configuration parameters
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on successful configuration, or an error if
    /// the configuration is invalid.
    ///
    /// # Errors
    ///
    /// May return errors for:
    /// - Invalid configuration parameters
    /// - Missing required parameters
    /// - Parameter validation failures
    fn configure(&mut self, config: HashMap<String, serde_json::Value>) -> Result<()>;

    /// Returns the current plugin configuration.
    ///
    /// # Returns
    ///
    /// A reference to the plugin's configuration map.
    fn get_config(&self) -> &HashMap<String, serde_json::Value>;

    /// Validates the current configuration.
    ///
    /// This method should check that all configuration parameters are valid
    /// and compatible with each other.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the configuration is valid, or an error describing
    /// the validation failure.
    fn validate_config(&self) -> Result<()> {
        Ok(())
    }

    /// Initializes the plugin for use.
    ///
    /// This method is called after configuration and before the plugin
    /// is used for computation. It should set up any internal state,
    /// allocate resources, or perform other initialization tasks.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on successful initialization, or an error if
    /// initialization fails.
    ///
    /// # Errors
    ///
    /// May return errors for:
    /// - Resource allocation failures
    /// - Invalid configuration
    /// - Hardware compatibility issues
    fn initialize(&mut self) -> Result<()> {
        self.validate_config()
    }

    /// Cleans up plugin resources.
    ///
    /// This method is called when the plugin is no longer needed.
    /// It should release any allocated resources, close connections,
    /// or perform other cleanup tasks.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on successful cleanup, or an error if
    /// cleanup fails.
    fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }

    /// Returns the plugin as an `Any` trait object for downcasting.
    ///
    /// This enables type-safe downcasting to concrete plugin types
    /// when needed for advanced functionality.
    ///
    /// # Returns
    ///
    /// A reference to self as an `Any` trait object.
    fn as_any(&self) -> &dyn Any;

    /// Returns plugin dependencies.
    ///
    /// Lists other plugins or system components that this plugin
    /// requires to function correctly.
    ///
    /// # Returns
    ///
    /// A vector of dependency specifications (e.g., "plugin_name >= 1.0.0").
    fn dependencies(&self) -> Vec<String> {
        Vec::new()
    }

    /// Returns plugin capabilities.
    ///
    /// Describes what features or operations this plugin supports.
    /// This can be used for plugin discovery and compatibility checking.
    ///
    /// # Returns
    ///
    /// A vector of capability strings.
    fn capabilities(&self) -> Vec<String> {
        Vec::new()
    }

    /// Returns plugin tags for categorization.
    ///
    /// Tags help organize and discover plugins by functionality
    /// (e.g., "attention", "optimization", "quantization").
    ///
    /// # Returns
    ///
    /// A vector of tag strings.
    fn tags(&self) -> Vec<String> {
        Vec::new()
    }

    /// Performs the forward computation of this plugin.
    ///
    /// This method provides the core computational functionality of the plugin,
    /// accepting and returning tensors.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor to process
    ///
    /// # Returns
    ///
    /// Returns `Ok(output)` containing the plugin's output tensor, or an error
    /// if computation fails.
    ///
    /// # Errors
    ///
    /// May return errors for:
    /// - Invalid input dimensions
    /// - Numerical errors during computation
    /// - Resource allocation failures
    fn forward(&self, input: Tensor) -> Result<Tensor>;
}

/// Helper trait for cloning plugin trait objects.
///
/// This trait provides a way to clone boxed plugin instances,
/// which is needed for plugin registry management.
pub trait ClonePlugin {
    /// Creates a clone of the plugin.
    ///
    /// # Returns
    ///
    /// A boxed clone of the plugin.
    fn clone_plugin(&self) -> Box<dyn Plugin>;
}

impl<T> ClonePlugin for T
where
    T: Plugin + Clone + 'static,
{
    fn clone_plugin(&self) -> Box<dyn Plugin> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Plugin> {
    fn clone(&self) -> Self {
        self.clone_plugin()
    }
}

/// Plugin lifecycle events.
///
/// These events are fired during different stages of a plugin's lifecycle,
/// allowing for monitoring, logging, and custom handling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginEvent {
    /// Plugin is being loaded.
    Loading { name: String, version: String },
    /// Plugin has been successfully loaded.
    Loaded { name: String, version: String },
    /// Plugin configuration is being updated.
    Configuring {
        name: String,
        config: HashMap<String, serde_json::Value>,
    },
    /// Plugin is being initialized.
    Initializing { name: String },
    /// Plugin has been successfully initialized.
    Initialized { name: String },
    /// Plugin is being unloaded.
    Unloading { name: String },
    /// Plugin has been unloaded.
    Unloaded { name: String },
    /// Plugin encountered an error.
    Error { name: String, error: String },
}

/// Trait for handling plugin lifecycle events.
///
/// Implement this trait to receive notifications about plugin
/// lifecycle events for monitoring, logging, or custom handling.
pub trait PluginEventHandler: Send + Sync {
    /// Handles a plugin lifecycle event.
    ///
    /// # Arguments
    ///
    /// * `event` - The lifecycle event that occurred
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on successful handling, or an error if
    /// handling fails.
    fn handle_event(&self, event: &PluginEvent) -> Result<()>;
}

/// Plugin execution context.
///
/// Provides runtime information and utilities to plugins during execution.
/// This includes access to shared resources, configuration, and monitoring.
#[derive(Debug)]
pub struct PluginContext {
    /// Plugin name
    pub name: String,
    /// Runtime configuration
    pub config: HashMap<String, serde_json::Value>,
    /// Shared resources
    pub resources: HashMap<String, Box<dyn Any + Send + Sync>>,
    /// Performance metrics
    pub metrics: HashMap<String, f64>,
}

impl PluginContext {
    /// Creates a new plugin context.
    ///
    /// # Arguments
    ///
    /// * `name` - The plugin name
    /// * `config` - Initial configuration
    ///
    /// # Returns
    ///
    /// A new plugin context instance.
    pub fn new(name: String, config: HashMap<String, serde_json::Value>) -> Self {
        Self {
            name,
            config,
            resources: HashMap::new(),
            metrics: HashMap::new(),
        }
    }

    /// Adds a shared resource to the context.
    ///
    /// # Arguments
    ///
    /// * `key` - The resource key
    /// * `resource` - The resource to add
    pub fn add_resource<T: Any + Send + Sync>(&mut self, key: String, resource: T) {
        self.resources.insert(key, Box::new(resource));
    }

    /// Gets a shared resource from the context.
    ///
    /// # Arguments
    ///
    /// * `key` - The resource key
    ///
    /// # Returns
    ///
    /// An optional reference to the resource.
    pub fn get_resource<T: Any + Send + Sync>(&self, key: &str) -> Option<&T> {
        self.resources.get(key).and_then(|r| r.downcast_ref::<T>())
    }

    /// Updates a performance metric.
    ///
    /// # Arguments
    ///
    /// * `key` - The metric name
    /// * `value` - The metric value
    pub fn update_metric(&mut self, key: String, value: f64) {
        self.metrics.insert(key, value);
    }

    /// Gets a performance metric.
    ///
    /// # Arguments
    ///
    /// * `key` - The metric name
    ///
    /// # Returns
    ///
    /// The metric value if it exists.
    pub fn get_metric(&self, key: &str) -> Option<f64> {
        self.metrics.get(key).copied()
    }
}
