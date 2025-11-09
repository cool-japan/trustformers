//! Plugin registry for discovery and management.

use crate::errors::{Result, TrustformersError};
use crate::plugins::{Plugin, PluginInfo, PluginLoader, PluginManager};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Central registry for plugin discovery and management.
///
/// The `PluginRegistry` maintains a catalog of available plugins,
/// handles plugin loading and unloading, and provides compatibility
/// checking and version management.
///
/// # Thread Safety
///
/// The registry is thread-safe and can be shared across multiple threads.
/// It uses read-write locks to allow concurrent reads while ensuring
/// exclusive access for modifications.
///
/// # Example
///
/// ```no_run
/// use trustformers_core::plugins::{PluginRegistry, PluginInfo};
/// use std::path::Path;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut registry = PluginRegistry::new();
///
/// // Register a plugin
/// let info = PluginInfo::new(
///     "custom_attention",
///     "1.0.0",
///     "Custom attention mechanism",
///     &["trustformers-core >= 0.1.0"]
/// );
/// registry.register("custom_attention", info)?;
///
/// // Load the plugin
/// let plugin = registry.load_plugin("custom_attention")?;
///
/// // List all available plugins
/// let plugins = registry.list_plugins();
/// assert!(plugins.contains(&"custom_attention".to_string()));
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct PluginRegistry {
    /// Plugin information cache
    plugins: Arc<RwLock<HashMap<String, PluginInfo>>>,
    /// Loaded plugin instances
    loaded: Arc<RwLock<HashMap<String, Box<dyn Plugin>>>>,
    /// Plugin search paths
    search_paths: Arc<RwLock<Vec<PathBuf>>>,
    /// Plugin loader
    loader: Arc<PluginLoader>,
    /// Registry configuration
    config: RegistryConfig,
}

impl PluginRegistry {
    /// Creates a new plugin registry.
    ///
    /// # Returns
    ///
    /// A new registry instance with default configuration.
    pub fn new() -> Self {
        Self {
            plugins: Arc::new(RwLock::new(HashMap::new())),
            loaded: Arc::new(RwLock::new(HashMap::new())),
            search_paths: Arc::new(RwLock::new(Vec::new())),
            loader: Arc::new(PluginLoader::new()),
            config: RegistryConfig::default(),
        }
    }

    /// Creates a new plugin registry with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Registry configuration
    ///
    /// # Returns
    ///
    /// A new registry instance.
    pub fn with_config(config: RegistryConfig) -> Self {
        Self {
            plugins: Arc::new(RwLock::new(HashMap::new())),
            loaded: Arc::new(RwLock::new(HashMap::new())),
            search_paths: Arc::new(RwLock::new(Vec::new())),
            loader: Arc::new(PluginLoader::new()),
            config,
        }
    }

    /// Registers a plugin in the registry.
    ///
    /// # Arguments
    ///
    /// * `name` - Plugin name (must be unique)
    /// * `info` - Plugin metadata
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, error if registration fails.
    ///
    /// # Errors
    ///
    /// - Plugin name already exists
    /// - Invalid plugin information
    pub fn register(&self, name: &str, info: PluginInfo) -> Result<()> {
        info.validate()?;

        let mut plugins = self.plugins.write().map_err(|_| {
            TrustformersError::lock_error("Failed to acquire write lock".to_string())
        })?;

        if plugins.contains_key(name) {
            return Err(TrustformersError::plugin_error(format!(
                "Plugin '{}' is already registered",
                name
            )));
        }

        plugins.insert(name.to_string(), info);
        Ok(())
    }

    /// Unregisters a plugin from the registry.
    ///
    /// # Arguments
    ///
    /// * `name` - Plugin name to unregister
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, error if the plugin is not found.
    pub fn unregister(&self, name: &str) -> Result<()> {
        // Unload the plugin if it's currently loaded
        if self.is_loaded(name) {
            self.unload_plugin(name)?;
        }

        let mut plugins = self.plugins.write().map_err(|_| {
            TrustformersError::lock_error("Failed to acquire write lock".to_string())
        })?;

        plugins.remove(name).ok_or_else(|| {
            TrustformersError::plugin_error(format!("Plugin '{}' not found", name))
        })?;

        Ok(())
    }

    /// Gets information about a registered plugin.
    ///
    /// # Arguments
    ///
    /// * `name` - Plugin name
    ///
    /// # Returns
    ///
    /// Plugin information if found.
    pub fn get_plugin_info(&self, name: &str) -> Result<PluginInfo> {
        let plugins = self.plugins.read().map_err(|_| {
            TrustformersError::lock_error("Failed to acquire read lock".to_string())
        })?;

        plugins
            .get(name)
            .cloned()
            .ok_or_else(|| TrustformersError::plugin_error(format!("Plugin '{}' not found", name)))
    }

    /// Checks if a plugin is currently loaded.
    ///
    /// # Arguments
    ///
    /// * `name` - Plugin name
    ///
    /// # Returns
    ///
    /// `true` if the plugin is loaded, `false` otherwise.
    pub fn is_loaded(&self, name: &str) -> bool {
        self.loaded.read().map(|loaded| loaded.contains_key(name)).unwrap_or(false)
    }

    /// Unloads a plugin from memory.
    ///
    /// # Arguments
    ///
    /// * `name` - Plugin name to unload
    ///
    /// # Returns
    ///
    /// `Ok(())` on success.
    pub fn unload_plugin(&self, name: &str) -> Result<()> {
        let mut loaded = self.loaded.write().map_err(|_| {
            TrustformersError::lock_error("Failed to acquire write lock".to_string())
        })?;

        if let Some(mut plugin) = loaded.remove(name) {
            plugin.cleanup()?;
        }

        Ok(())
    }

    /// Adds a directory to the plugin search path.
    ///
    /// # Arguments
    ///
    /// * `path` - Directory path to add
    pub fn add_search_path<P: AsRef<Path>>(&self, path: P) {
        if let Ok(mut paths) = self.search_paths.write() {
            paths.push(path.as_ref().to_path_buf());
        }
    }

    /// Removes a directory from the plugin search path.
    ///
    /// # Arguments
    ///
    /// * `path` - Directory path to remove
    pub fn remove_search_path<P: AsRef<Path>>(&self, path: P) {
        if let Ok(mut paths) = self.search_paths.write() {
            paths.retain(|p| p != path.as_ref());
        }
    }

    /// Scans all search paths for plugins and registers them.
    ///
    /// # Returns
    ///
    /// Number of plugins discovered and registered.
    pub fn scan_for_plugins(&self) -> Result<usize> {
        let search_paths = self
            .search_paths
            .read()
            .map_err(|_| TrustformersError::lock_error("Failed to acquire read lock".to_string()))?
            .clone();

        let mut count = 0;

        for path in &search_paths {
            if let Ok(entries) = std::fs::read_dir(path) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() && self.is_plugin_file(&path) {
                        if let Ok(info) = self.loader.load_plugin_info(&path) {
                            let name = info.name().to_string();
                            if self.register(&name, info).is_ok() {
                                count += 1;
                            }
                        }
                    }
                }
            }
        }

        Ok(count)
    }

    /// Checks if a file is a plugin file based on extension.
    ///
    /// # Arguments
    ///
    /// * `path` - File path to check
    ///
    /// # Returns
    ///
    /// `true` if it's a plugin file.
    fn is_plugin_file(&self, path: &Path) -> bool {
        if let Some(ext) = path.extension() {
            let ext = ext.to_string_lossy().to_lowercase();
            matches!(ext.as_str(), "so" | "dll" | "dylib" | "wasm")
        } else {
            false
        }
    }

    /// Validates plugin dependencies.
    ///
    /// # Arguments
    ///
    /// * `name` - Plugin name to validate
    ///
    /// # Returns
    ///
    /// `Ok(())` if all dependencies are satisfied.
    pub fn validate_dependencies(&self, name: &str) -> Result<()> {
        let info = self.get_plugin_info(name)?;

        for dep in info.dependencies() {
            if !dep.optional {
                let dep_info = self.get_plugin_info(&dep.name)?;
                if !dep.requirement.matches(dep_info.version()) {
                    return Err(TrustformersError::plugin_error(format!(
                        "Plugin '{}' requires '{}' {} but found {}",
                        name,
                        dep.name,
                        dep.requirement,
                        dep_info.version()
                    )));
                }
            }
        }

        Ok(())
    }

    /// Exports the registry to a configuration file.
    ///
    /// # Arguments
    ///
    /// * `path` - File path to write the configuration
    ///
    /// # Returns
    ///
    /// `Ok(())` on success.
    pub fn export_config<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let plugins = self.plugins.read().map_err(|_| {
            TrustformersError::lock_error("Failed to acquire read lock".to_string())
        })?;

        let config = RegistryConfig {
            plugins: plugins.clone(),
            ..self.config.clone()
        };

        let json = serde_json::to_string_pretty(&config)
            .map_err(|e| TrustformersError::serialization_error(e.to_string()))?;

        std::fs::write(path, json).map_err(|e| TrustformersError::io_error(e.to_string()))?;

        Ok(())
    }

    /// Imports registry configuration from a file.
    ///
    /// # Arguments
    ///
    /// * `path` - File path to read the configuration from
    ///
    /// # Returns
    ///
    /// `Ok(())` on success.
    pub fn import_config<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| TrustformersError::io_error(e.to_string()))?;

        let config: RegistryConfig = serde_json::from_str(&json)
            .map_err(|e| TrustformersError::serialization_error(e.to_string()))?;

        let mut plugins = self.plugins.write().map_err(|_| {
            TrustformersError::lock_error("Failed to acquire write lock".to_string())
        })?;

        for (name, info) in config.plugins {
            plugins.insert(name, info);
        }

        Ok(())
    }

    /// Gets registry statistics.
    ///
    /// # Returns
    ///
    /// Registry statistics.
    pub fn stats(&self) -> Result<RegistryStats> {
        let plugins = self.plugins.read().map_err(|_| {
            TrustformersError::lock_error("Failed to acquire read lock".to_string())
        })?;
        let loaded = self.loaded.read().map_err(|_| {
            TrustformersError::lock_error("Failed to acquire read lock".to_string())
        })?;

        Ok(RegistryStats {
            total_plugins: plugins.len(),
            loaded_plugins: loaded.len(),
            search_paths: self.search_paths.read().map(|paths| paths.len()).unwrap_or(0),
        })
    }
}

impl PluginManager for PluginRegistry {
    fn discover_plugins(&self) -> Result<HashMap<String, PluginInfo>> {
        let plugins = self.plugins.read().map_err(|_| {
            TrustformersError::lock_error("Failed to acquire read lock".to_string())
        })?;
        Ok(plugins.clone())
    }

    fn is_compatible(&self, name: &str, version: &str) -> Result<bool> {
        let info = self.get_plugin_info(name)?;
        Ok(info.is_compatible_with("trustformers-core", version))
    }

    fn load_plugin(&self, name: &str) -> Result<Box<dyn Plugin>> {
        // Check if already loaded
        {
            let loaded = self.loaded.read().map_err(|_| {
                TrustformersError::lock_error("Failed to acquire read lock".to_string())
            })?;
            if let Some(plugin) = loaded.get(name) {
                return Ok(plugin.clone());
            }
        }

        // Validate dependencies
        self.validate_dependencies(name)?;

        // Get plugin info
        let info = self.get_plugin_info(name)?;

        // Load the plugin
        let mut plugin = self.loader.load_plugin(&info)?;
        plugin.initialize()?;

        // Store in loaded cache
        let plugin_clone = plugin.clone();
        {
            let mut loaded = self.loaded.write().map_err(|_| {
                TrustformersError::lock_error("Failed to acquire write lock".to_string())
            })?;
            loaded.insert(name.to_string(), plugin);
        }

        Ok(plugin_clone)
    }

    fn list_plugins(&self) -> Vec<String> {
        self.plugins
            .read()
            .map(|plugins| plugins.keys().cloned().collect())
            .unwrap_or_default()
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Registry configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    /// Registered plugins
    #[serde(default)]
    pub plugins: HashMap<String, PluginInfo>,
    /// Maximum number of loaded plugins
    #[serde(default = "default_max_loaded")]
    pub max_loaded_plugins: usize,
    /// Enable automatic plugin discovery
    #[serde(default = "default_auto_discovery")]
    pub auto_discovery: bool,
    /// Plugin cache directory
    #[serde(default)]
    pub cache_dir: Option<PathBuf>,
    /// Plugin load timeout in seconds
    #[serde(default = "default_load_timeout")]
    pub load_timeout_secs: u64,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            plugins: HashMap::new(),
            max_loaded_plugins: default_max_loaded(),
            auto_discovery: default_auto_discovery(),
            cache_dir: None,
            load_timeout_secs: default_load_timeout(),
        }
    }
}

fn default_max_loaded() -> usize {
    100
}
fn default_auto_discovery() -> bool {
    true
}
fn default_load_timeout() -> u64 {
    30
}

/// Registry statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryStats {
    /// Total number of registered plugins
    pub total_plugins: usize,
    /// Number of currently loaded plugins
    pub loaded_plugins: usize,
    /// Number of search paths
    pub search_paths: usize,
}
