// Plugin Registry - Plugin registration and metadata management
//
// This module manages plugin registration, discovery, and metadata storage
// for the TrustformeRS WASM plugin system.

use super::interface::{PluginCapabilities, PluginConfig, PluginType, PluginPermission};
use std::collections::BTreeMap;
use std::string::{String, ToString};
use std::vec::Vec;
use core::fmt;
use serde::{Deserialize, Serialize};

/// Plugin registry for managing plugin metadata and discovery
#[derive(Debug)]
pub struct PluginRegistry {
    plugins: BTreeMap<String, PluginMetadata>,
    config: RegistryConfig,
    type_index: BTreeMap<PluginType, Vec<String>>,
    dependency_graph: BTreeMap<String, Vec<String>>,
}

impl PluginRegistry {
    /// Create a new plugin registry with default configuration
    pub fn new() -> Self {
        Self {
            plugins: BTreeMap::new(),
            config: RegistryConfig::default(),
            type_index: BTreeMap::new(),
            dependency_graph: BTreeMap::new(),
        }
    }

    /// Create a plugin registry with custom configuration
    pub fn with_config(config: RegistryConfig) -> Self {
        Self {
            plugins: BTreeMap::new(),
            config,
            type_index: BTreeMap::new(),
            dependency_graph: BTreeMap::new(),
        }
    }

    /// Register a plugin
    pub fn register(&mut self, metadata: PluginMetadata) -> Result<(), PluginError> {
        // Validate plugin metadata
        self.validate_metadata(&metadata)?;

        let plugin_id = metadata.id.clone();

        // Check for ID conflicts
        if self.plugins.contains_key(&plugin_id) && !self.config.allow_updates {
            return Err(PluginError::PluginAlreadyExists(plugin_id));
        }

        // Update type index
        self.type_index
            .entry(metadata.plugin_type)
            .or_insert_with(Vec::new)
            .push(plugin_id.clone());

        // Update dependency graph
        if !metadata.dependencies.is_empty() {
            self.dependency_graph.insert(plugin_id.clone(), metadata.dependencies.clone());
        }

        // Register the plugin
        self.plugins.insert(plugin_id.clone(), metadata);

        web_sys::console::log_1(&format!("Registered plugin: {plugin_id}").into());
        Ok(())
    }

    /// Unregister a plugin
    pub fn unregister(&mut self, plugin_id: &str) -> Result<PluginMetadata, PluginError> {
        let metadata = self.plugins.remove(plugin_id)
            .ok_or_else(|| PluginError::PluginNotFound(plugin_id.to_string()))?;

        // Remove from type index
        if let Some(plugins) = self.type_index.get_mut(&metadata.plugin_type) {
            plugins.retain(|id| id != plugin_id);
            if plugins.is_empty() {
                self.type_index.remove(&metadata.plugin_type);
            }
        }

        // Remove from dependency graph
        self.dependency_graph.remove(plugin_id);

        // Remove dependencies on this plugin from other plugins
        for deps in self.dependency_graph.values_mut() {
            deps.retain(|dep| dep != plugin_id);
        }

        web_sys::console::log_1(&format!("Unregistered plugin: {plugin_id}").into());
        Ok(metadata)
    }

    /// Get plugin metadata by ID
    pub fn get_metadata(&self, plugin_id: &str) -> Option<&PluginMetadata> {
        self.plugins.get(plugin_id)
    }

    /// Check if a plugin is registered
    pub fn is_registered(&self, plugin_id: &str) -> bool {
        self.plugins.contains_key(plugin_id)
    }

    /// List all registered plugins
    pub fn list_plugins(&self) -> Vec<&PluginMetadata> {
        self.plugins.values().collect()
    }

    /// Find plugins by type
    pub fn find_plugins_by_type(&self, plugin_type: PluginType) -> Vec<&PluginMetadata> {
        self.type_index
            .get(&plugin_type)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.plugins.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find plugins by capability
    pub fn find_plugins_by_capability(&self, capability: &str) -> Vec<&PluginMetadata> {
        self.plugins
            .values()
            .filter(|metadata| metadata.capabilities.supports_function(capability))
            .collect()
    }

    /// Find plugins by permission
    pub fn find_plugins_by_permission(&self, permission: PluginPermission) -> Vec<&PluginMetadata> {
        self.plugins
            .values()
            .filter(|metadata| metadata.capabilities.requires_permission(&permission))
            .collect()
    }

    /// Get plugin dependencies
    pub fn get_dependencies(&self, plugin_id: &str) -> Vec<String> {
        self.dependency_graph
            .get(plugin_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Get plugins that depend on the given plugin
    pub fn get_dependents(&self, plugin_id: &str) -> Vec<String> {
        self.dependency_graph
            .iter()
            .filter_map(|(id, deps)| {
                if deps.contains(&plugin_id.to_string()) {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Check for circular dependencies
    pub fn has_circular_dependencies(&self, plugin_id: &str) -> bool {
        let mut visited = std::collections::HashSet::new();
        let mut stack = std::collections::HashSet::new();
        self.has_cycle_helper(plugin_id, &mut visited, &mut stack)
    }

    /// Get dependency resolution order
    pub fn resolve_dependencies(&self, plugin_id: &str) -> Result<Vec<String>, PluginError> {
        let mut resolved = Vec::new();
        let mut visiting = std::collections::HashSet::new();
        let mut visited = std::collections::HashSet::new();

        self.resolve_dependencies_helper(plugin_id, &mut resolved, &mut visiting, &mut visited)?;
        Ok(resolved)
    }

    /// Validate plugin compatibility
    pub fn validate_compatibility(&self, plugin_id: &str) -> Result<(), PluginError> {
        let metadata = self.get_metadata(plugin_id)
            .ok_or_else(|| PluginError::PluginNotFound(plugin_id.to_string()))?;

        // Check API version compatibility
        if !self.is_api_version_compatible(&metadata.capabilities.api_version) {
            return Err(PluginError::ApiVersionIncompatible(
                metadata.capabilities.api_version.clone(),
                self.config.api_version.clone(),
            ));
        }

        // Check dependencies
        for dep in &metadata.dependencies {
            if !self.is_registered(dep) {
                return Err(PluginError::DependencyNotFound(dep.clone()));
            }

            // Recursively check dependency compatibility
            self.validate_compatibility(dep)?;
        }

        Ok(())
    }

    /// Get registry statistics
    pub fn get_statistics(&self) -> RegistryStatistics {
        let mut type_distribution = BTreeMap::new();
        for metadata in self.plugins.values() {
            *type_distribution.entry(metadata.plugin_type).or_insert(0) += 1;
        }

        RegistryStatistics {
            total_plugins: self.plugins.len(),
            plugins_by_type: type_distribution,
            total_dependencies: self.dependency_graph.len(),
            plugins_with_dependencies: self.dependency_graph.values().filter(|deps| !deps.is_empty()).count(),
        }
    }

    /// Count registered plugins
    pub fn count(&self) -> usize {
        self.plugins.len()
    }

    /// Get type distribution
    pub fn get_type_distribution(&self) -> std::collections::HashMap<PluginType, usize> {
        let mut distribution = std::collections::HashMap::new();
        for metadata in self.plugins.values() {
            *distribution.entry(metadata.plugin_type).or_insert(0) += 1;
        }
        distribution
    }

    /// Clear all registered plugins
    pub fn clear(&mut self) {
        self.plugins.clear();
        self.type_index.clear();
        self.dependency_graph.clear();
        web_sys::console::log_1(&"Cleared plugin registry".into());
    }

    /// Export registry to JSON
    pub fn export_to_json(&self) -> Result<String, PluginError> {
        let export_data = RegistryExport {
            plugins: self.plugins.values().cloned().collect(),
            config: self.config.clone(),
            version: "1.0.0".to_string(),
        };

        serde_json::to_string_pretty(&export_data)
            .map_err(|e| PluginError::SerializationFailed(e.to_string()))
    }

    /// Import registry from JSON
    pub fn import_from_json(&mut self, json: &str) -> Result<(), PluginError> {
        let import_data: RegistryExport = serde_json::from_str(json)
            .map_err(|e| PluginError::DeserializationFailed(e.to_string()))?;

        // Clear existing registry
        self.clear();

        // Import plugins
        for metadata in import_data.plugins {
            self.register(metadata)?;
        }

        web_sys::console::log_1(&"Imported plugin registry from JSON".into());
        Ok(())
    }

    // Helper methods

    fn validate_metadata(&self, metadata: &PluginMetadata) -> Result<(), PluginError> {
        // Validate ID
        if metadata.id.is_empty() {
            return Err(PluginError::InvalidMetadata("Plugin ID cannot be empty".to_string()));
        }

        // Validate name
        if metadata.name.is_empty() {
            return Err(PluginError::InvalidMetadata("Plugin name cannot be empty".to_string()));
        }

        // Validate version
        if metadata.version.is_empty() {
            return Err(PluginError::InvalidMetadata("Plugin version cannot be empty".to_string()));
        }

        // Validate source URL if provided
        if let Some(ref url) = metadata.source_url {
            if url.is_empty() {
                return Err(PluginError::InvalidMetadata("Source URL cannot be empty".to_string()));
            }
        }

        Ok(())
    }

    fn has_cycle_helper(
        &self,
        plugin_id: &str,
        visited: &mut std::collections::HashSet<String>,
        stack: &mut std::collections::HashSet<String>,
    ) -> bool {
        if stack.contains(plugin_id) {
            return true;
        }

        if visited.contains(plugin_id) {
            return false;
        }

        visited.insert(plugin_id.to_string());
        stack.insert(plugin_id.to_string());

        if let Some(deps) = self.dependency_graph.get(plugin_id) {
            for dep in deps {
                if self.has_cycle_helper(dep, visited, stack) {
                    return true;
                }
            }
        }

        stack.remove(plugin_id);
        false
    }

    fn resolve_dependencies_helper(
        &self,
        plugin_id: &str,
        resolved: &mut Vec<String>,
        visiting: &mut std::collections::HashSet<String>,
        visited: &mut std::collections::HashSet<String>,
    ) -> Result<(), PluginError> {
        if visiting.contains(plugin_id) {
            return Err(PluginError::CircularDependency(plugin_id.to_string()));
        }

        if visited.contains(plugin_id) {
            return Ok(());
        }

        visiting.insert(plugin_id.to_string());

        if let Some(deps) = self.dependency_graph.get(plugin_id) {
            for dep in deps {
                self.resolve_dependencies_helper(dep, resolved, visiting, visited)?;
            }
        }

        visiting.remove(plugin_id);
        visited.insert(plugin_id.to_string());
        resolved.push(plugin_id.to_string());

        Ok(())
    }

    fn is_api_version_compatible(&self, plugin_api_version: &str) -> bool {
        // Simple semantic version compatibility check
        let registry_version = &self.config.api_version;

        // Extract major version numbers
        let registry_major = registry_version.split('.').next().unwrap_or("0").parse::<u32>().unwrap_or(0);
        let plugin_major = plugin_api_version.split('.').next().unwrap_or("0").parse::<u32>().unwrap_or(0);

        // Compatible if major versions match
        registry_major == plugin_major
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Plugin metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    pub id: String,
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub license: String,
    pub plugin_type: PluginType,
    pub capabilities: PluginCapabilities,
    pub dependencies: Vec<String>,
    pub optional_dependencies: Vec<String>,
    pub source_url: Option<String>,
    pub checksum: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub tags: Vec<String>,
}

impl PluginMetadata {
    /// Create new plugin metadata
    pub fn new(
        id: String,
        name: String,
        version: String,
        plugin_type: PluginType,
    ) -> Self {
        let now = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();

        Self {
            id,
            name,
            version,
            description: String::new(),
            author: String::new(),
            license: String::new(),
            plugin_type,
            capabilities: PluginCapabilities::default(),
            dependencies: Vec::new(),
            optional_dependencies: Vec::new(),
            source_url: None,
            checksum: None,
            created_at: now.clone(),
            updated_at: now,
            tags: Vec::new(),
        }
    }

    /// Set description
    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }

    /// Set author
    pub fn with_author(mut self, author: String) -> Self {
        self.author = author;
        self
    }

    /// Set license
    pub fn with_license(mut self, license: String) -> Self {
        self.license = license;
        self
    }

    /// Set capabilities
    pub fn with_capabilities(mut self, capabilities: PluginCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Set source URL
    pub fn with_source_url(mut self, url: String) -> Self {
        self.source_url = Some(url);
        self
    }

    /// Set checksum
    pub fn with_checksum(mut self, checksum: String) -> Self {
        self.checksum = Some(checksum);
        self
    }

    /// Add a dependency
    pub fn with_dependency(mut self, dependency: String) -> Self {
        self.dependencies.push(dependency);
        self
    }

    /// Add a tag
    pub fn with_tag(mut self, tag: String) -> Self {
        self.tags.push(tag);
        self
    }

    /// Update timestamp
    pub fn touch(&mut self) {
        self.updated_at = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();
    }
}

/// Registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    pub allow_updates: bool,
    pub max_plugins: Option<usize>,
    pub api_version: String,
    pub trusted_sources: Vec<String>,
    pub blocked_plugins: Vec<String>,
    pub auto_resolve_dependencies: bool,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            allow_updates: true,
            max_plugins: None,
            api_version: "1.0.0".to_string(),
            trusted_sources: Vec::new(),
            blocked_plugins: Vec::new(),
            auto_resolve_dependencies: true,
        }
    }
}

/// Registry statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryStatistics {
    pub total_plugins: usize,
    pub plugins_by_type: BTreeMap<PluginType, usize>,
    pub total_dependencies: usize,
    pub plugins_with_dependencies: usize,
}

/// Registry export format
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RegistryExport {
    plugins: Vec<PluginMetadata>,
    config: RegistryConfig,
    version: String,
}

/// Plugin registry errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginError {
    PluginNotFound(String),
    PluginAlreadyExists(String),
    InvalidMetadata(String),
    DependencyNotFound(String),
    CircularDependency(String),
    ApiVersionIncompatible(String, String), // plugin version, registry version
    SerializationFailed(String),
    DeserializationFailed(String),
    RegistryFull,
    PluginBlocked(String),
    UntrustedSource(String),
}

impl fmt::Display for PluginError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PluginError::PluginNotFound(id) => write!(f, "Plugin not found: {}", id),
            PluginError::PluginAlreadyExists(id) => write!(f, "Plugin already exists: {}", id),
            PluginError::InvalidMetadata(msg) => write!(f, "Invalid metadata: {}", msg),
            PluginError::DependencyNotFound(dep) => write!(f, "Dependency not found: {}", dep),
            PluginError::CircularDependency(id) => write!(f, "Circular dependency detected: {}", id),
            PluginError::ApiVersionIncompatible(plugin, registry) => {
                write!(f, "API version incompatible: plugin {}, registry {}", plugin, registry)
            }
            PluginError::SerializationFailed(msg) => write!(f, "Serialization failed: {}", msg),
            PluginError::DeserializationFailed(msg) => write!(f, "Deserialization failed: {}", msg),
            PluginError::RegistryFull => write!(f, "Registry is full"),
            PluginError::PluginBlocked(id) => write!(f, "Plugin is blocked: {}", id),
            PluginError::UntrustedSource(url) => write!(f, "Untrusted source: {}", url),
        }
    }
}

impl std::error::Error for PluginError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = PluginRegistry::new();
        assert_eq!(registry.count(), 0);
        assert!(registry.list_plugins().is_empty());
    }

    #[test]
    fn test_plugin_registration() {
        let mut registry = PluginRegistry::new();
        let metadata = PluginMetadata::new(
            "test-plugin".to_string(),
            "Test Plugin".to_string(),
            "1.0.0".to_string(),
            PluginType::Extension,
        );

        assert!(registry.register(metadata).is_ok());
        assert_eq!(registry.count(), 1);
        assert!(registry.is_registered("test-plugin"));
    }

    #[test]
    fn test_plugin_discovery() {
        let mut registry = PluginRegistry::new();
        let metadata = PluginMetadata::new(
            "test-plugin".to_string(),
            "Test Plugin".to_string(),
            "1.0.0".to_string(),
            PluginType::ModelLoader,
        );

        registry.register(metadata).unwrap();

        let plugins = registry.find_plugins_by_type(PluginType::ModelLoader);
        assert_eq!(plugins.len(), 1);
        assert_eq!(plugins[0].id, "test-plugin");
    }

    #[test]
    fn test_dependency_resolution() {
        let mut registry = PluginRegistry::new();

        // Register dependency first
        let dep_metadata = PluginMetadata::new(
            "dep-plugin".to_string(),
            "Dependency Plugin".to_string(),
            "1.0.0".to_string(),
            PluginType::Extension,
        );
        registry.register(dep_metadata).unwrap();

        // Register plugin with dependency
        let metadata = PluginMetadata::new(
            "main-plugin".to_string(),
            "Main Plugin".to_string(),
            "1.0.0".to_string(),
            PluginType::Extension,
        ).with_dependency("dep-plugin".to_string());

        registry.register(metadata).unwrap();

        let deps = registry.get_dependencies("main-plugin");
        assert_eq!(deps, vec!["dep-plugin".to_string()]);

        let resolved = registry.resolve_dependencies("main-plugin").unwrap();
        assert_eq!(resolved, vec!["dep-plugin".to_string(), "main-plugin".to_string()]);
    }
}