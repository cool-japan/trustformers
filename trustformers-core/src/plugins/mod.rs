//! Plugin system for TrustformeRS.
//!
//! This module provides a flexible plugin architecture that allows custom layers
//! and components to be dynamically loaded and integrated into TrustformeRS models.
//! The plugin system supports version compatibility, discovery, and runtime loading.
//!
//! # Overview
//!
//! The plugin system consists of:
//!
//! - [`Plugin`]: Core trait for plugin implementations
//! - [`PluginRegistry`]: Central registry for plugin discovery and management
//! - [`PluginInfo`]: Metadata about plugin capabilities and compatibility
//! - [`PluginLoader`]: Dynamic loading of plugin implementations
//!
//! # Example
//!
//! ```no_run
//! use trustformers_core::plugins::{Plugin, PluginRegistry, PluginInfo};
//! use trustformers_core::tensor::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Register a custom plugin
//! let mut registry = PluginRegistry::new();
//! registry.register("custom_attention", PluginInfo::new(
//!     "1.0.0",
//!     "Custom attention mechanism",
//!     &["trustformers-core >= 0.1.0"]
//! ))?;
//!
//! // Load and use the plugin
//! let plugin = registry.load("custom_attention")?;
//! let input = Tensor::randn(&[2, 128, 768])?;
//! let output = plugin.forward(input)?;
//! # Ok(())
//! # }
//! ```

pub mod info;
pub mod loader;
pub mod registry;
pub mod traits;

pub use info::{Dependency, GpuRequirements, PluginInfo, SystemRequirements};
pub use loader::PluginLoader;
pub use registry::PluginRegistry;
pub use traits::{Plugin, PluginContext, PluginEvent, PluginEventHandler};

use crate::errors::Result;
use std::collections::HashMap;

/// Plugin discovery and management interface.
///
/// This trait provides methods for discovering available plugins,
/// checking compatibility, and loading plugin instances.
pub trait PluginManager {
    /// Discovers all available plugins in the system.
    ///
    /// # Returns
    ///
    /// A map of plugin names to their information metadata.
    fn discover_plugins(&self) -> Result<HashMap<String, PluginInfo>>;

    /// Checks if a plugin is compatible with the current system.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the plugin to check
    /// * `version` - The required version specification
    ///
    /// # Returns
    ///
    /// Returns `true` if the plugin is compatible, `false` otherwise.
    fn is_compatible(&self, name: &str, version: &str) -> Result<bool>;

    /// Loads a plugin by name.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the plugin to load
    ///
    /// # Returns
    ///
    /// Returns a boxed plugin instance ready for use.
    fn load_plugin(&self, name: &str) -> Result<Box<dyn Plugin>>;

    /// Lists all available plugin names.
    ///
    /// # Returns
    ///
    /// A vector of plugin names that can be loaded.
    fn list_plugins(&self) -> Vec<String>;
}
