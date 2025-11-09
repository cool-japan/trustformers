//! Plugin metadata and information structures.

use crate::errors::{Result, TrustformersError};
use semver::{Version, VersionReq};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Plugin metadata and compatibility information.
///
/// `PluginInfo` contains all the metadata needed to identify, validate,
/// and load a plugin. This includes version information, dependencies,
/// capabilities, and compatibility requirements.
///
/// # Example
///
/// ```no_run
/// use trustformers_core::plugins::PluginInfo;
///
/// let info = PluginInfo::new(
///     "custom_attention",
///     "1.2.0",
///     "Optimized multi-head attention with flash attention support",
///     &["trustformers-core >= 0.1.0", "cuda >= 11.0"]
/// );
///
/// assert_eq!(info.name(), "custom_attention");
/// assert_eq!(info.version().to_string(), "1.2.0");
/// assert!(info.is_compatible_with("trustformers-core", "0.2.0"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginInfo {
    /// Plugin name
    name: String,
    /// Plugin version
    version: Version,
    /// Human-readable description
    description: String,
    /// Author information
    author: Option<String>,
    /// Plugin homepage or repository URL
    homepage: Option<String>,
    /// License identifier (e.g., "MIT", "Apache-2.0")
    license: Option<String>,
    /// Compatibility requirements
    dependencies: Vec<Dependency>,
    /// Plugin capabilities
    capabilities: Vec<String>,
    /// Plugin tags for categorization
    tags: Vec<String>,
    /// Minimum system requirements
    requirements: SystemRequirements,
    /// Plugin entry point or library path
    entry_point: String,
    /// Additional metadata
    metadata: HashMap<String, serde_json::Value>,
}

impl PluginInfo {
    /// Creates a new plugin info instance.
    ///
    /// # Arguments
    ///
    /// * `name` - Plugin name
    /// * `version` - Plugin version string (must be valid semver)
    /// * `description` - Plugin description
    /// * `dependencies` - Array of dependency specifications
    ///
    /// # Returns
    ///
    /// A new `PluginInfo` instance.
    ///
    /// # Panics
    ///
    /// Panics if the version string is not valid semver.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use trustformers_core::plugins::PluginInfo;
    ///
    /// let info = PluginInfo::new(
    ///     "my_plugin",
    ///     "1.0.0",
    ///     "A sample plugin",
    ///     &["trustformers-core >= 0.1.0"]
    /// );
    /// ```
    pub fn new(name: &str, version: &str, description: &str, dependencies: &[&str]) -> Self {
        let version = Version::parse(version).expect("Invalid version string");

        let deps = dependencies.iter().filter_map(|dep| Dependency::parse(dep).ok()).collect();

        Self {
            name: name.to_string(),
            version,
            description: description.to_string(),
            author: None,
            homepage: None,
            license: None,
            dependencies: deps,
            capabilities: Vec::new(),
            tags: Vec::new(),
            requirements: SystemRequirements::default(),
            entry_point: String::new(),
            metadata: HashMap::new(),
        }
    }

    /// Returns the plugin name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the plugin version.
    pub fn version(&self) -> &Version {
        &self.version
    }

    /// Returns the plugin description.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Returns the plugin author.
    pub fn author(&self) -> Option<&str> {
        self.author.as_deref()
    }

    /// Sets the plugin author.
    pub fn set_author(&mut self, author: String) {
        self.author = Some(author);
    }

    /// Returns the plugin homepage URL.
    pub fn homepage(&self) -> Option<&str> {
        self.homepage.as_deref()
    }

    /// Sets the plugin homepage URL.
    pub fn set_homepage(&mut self, homepage: String) {
        self.homepage = Some(homepage);
    }

    /// Returns the plugin license.
    pub fn license(&self) -> Option<&str> {
        self.license.as_deref()
    }

    /// Sets the plugin license.
    pub fn set_license(&mut self, license: String) {
        self.license = Some(license);
    }

    /// Returns the plugin dependencies.
    pub fn dependencies(&self) -> &[Dependency] {
        &self.dependencies
    }

    /// Adds a dependency requirement.
    pub fn add_dependency(&mut self, dependency: Dependency) {
        self.dependencies.push(dependency);
    }

    /// Returns the plugin capabilities.
    pub fn capabilities(&self) -> &[String] {
        &self.capabilities
    }

    /// Adds a capability.
    pub fn add_capability(&mut self, capability: String) {
        self.capabilities.push(capability);
    }

    /// Returns the plugin tags.
    pub fn tags(&self) -> &[String] {
        &self.tags
    }

    /// Adds a tag.
    pub fn add_tag(&mut self, tag: String) {
        self.tags.push(tag);
    }

    /// Returns the system requirements.
    pub fn requirements(&self) -> &SystemRequirements {
        &self.requirements
    }

    /// Sets the system requirements.
    pub fn set_requirements(&mut self, requirements: SystemRequirements) {
        self.requirements = requirements;
    }

    /// Returns the plugin entry point.
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }

    /// Sets the plugin entry point.
    pub fn set_entry_point(&mut self, entry_point: String) {
        self.entry_point = entry_point;
    }

    /// Returns the plugin metadata.
    pub fn metadata(&self) -> &HashMap<String, serde_json::Value> {
        &self.metadata
    }

    /// Adds metadata.
    pub fn add_metadata(&mut self, key: String, value: serde_json::Value) {
        self.metadata.insert(key, value);
    }

    /// Checks if this plugin is compatible with a given dependency.
    ///
    /// # Arguments
    ///
    /// * `name` - The dependency name
    /// * `version` - The dependency version
    ///
    /// # Returns
    ///
    /// `true` if compatible, `false` otherwise.
    pub fn is_compatible_with(&self, name: &str, version: &str) -> bool {
        if let Ok(dep_version) = Version::parse(version) {
            for dep in &self.dependencies {
                if dep.name == name {
                    return dep.requirement.matches(&dep_version);
                }
            }
        }
        true // No dependency found means compatible
    }

    /// Validates the plugin info for completeness and correctness.
    ///
    /// # Returns
    ///
    /// `Ok(())` if valid, error otherwise.
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(TrustformersError::invalid_config(
                "Plugin name cannot be empty".to_string(),
            ));
        }

        if self.description.is_empty() {
            return Err(TrustformersError::invalid_config(
                "Plugin description cannot be empty".to_string(),
            ));
        }

        if self.entry_point.is_empty() {
            return Err(TrustformersError::invalid_config(
                "Plugin entry point cannot be empty".to_string(),
            ));
        }

        // Validate dependencies
        for dep in &self.dependencies {
            dep.validate()?;
        }

        // Validate system requirements
        self.requirements.validate()?;

        Ok(())
    }
}

/// Plugin dependency specification.
///
/// Represents a dependency on another plugin or system component
/// with version requirements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    /// Dependency name
    pub name: String,
    /// Version requirement
    pub requirement: VersionReq,
    /// Whether this dependency is optional
    pub optional: bool,
    /// Dependency features required
    pub features: Vec<String>,
}

impl Dependency {
    /// Creates a new dependency.
    ///
    /// # Arguments
    ///
    /// * `name` - Dependency name
    /// * `requirement` - Version requirement string
    ///
    /// # Returns
    ///
    /// A new dependency instance.
    pub fn new(name: &str, requirement: &str) -> Result<Self> {
        let req = VersionReq::parse(requirement).map_err(|e| {
            TrustformersError::invalid_config(format!(
                "Invalid version requirement '{}': {}",
                requirement, e
            ))
        })?;

        Ok(Self {
            name: name.to_string(),
            requirement: req,
            optional: false,
            features: Vec::new(),
        })
    }

    /// Parses a dependency string.
    ///
    /// Supports formats like:
    /// - "name >= 1.0.0"
    /// - "name = 1.2.3"
    /// - "name"
    ///
    /// # Arguments
    ///
    /// * `spec` - Dependency specification string
    ///
    /// # Returns
    ///
    /// A parsed dependency.
    pub fn parse(spec: &str) -> Result<Self> {
        let parts: Vec<&str> = spec.splitn(2, ' ').collect();
        let name = parts[0].to_string();

        let requirement = if parts.len() > 1 {
            VersionReq::parse(parts[1].trim()).map_err(|e| {
                TrustformersError::invalid_config(format!(
                    "Invalid dependency spec '{}': {}",
                    spec, e
                ))
            })?
        } else {
            VersionReq::STAR
        };

        Ok(Self {
            name,
            requirement,
            optional: false,
            features: Vec::new(),
        })
    }

    /// Sets the dependency as optional.
    pub fn optional(mut self) -> Self {
        self.optional = true;
        self
    }

    /// Adds required features.
    pub fn with_features(mut self, features: Vec<String>) -> Self {
        self.features = features;
        self
    }

    /// Validates the dependency.
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(TrustformersError::invalid_config(
                "Dependency name cannot be empty".to_string(),
            ));
        }
        Ok(())
    }
}

/// System requirements for running a plugin.
///
/// Specifies minimum hardware and software requirements
/// needed for the plugin to function correctly.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SystemRequirements {
    /// Minimum RAM in MB
    pub min_memory_mb: Option<u64>,
    /// Minimum free disk space in MB
    pub min_disk_mb: Option<u64>,
    /// Required CPU features (e.g., "avx2", "sse4.1")
    pub cpu_features: Vec<String>,
    /// GPU requirements
    pub gpu: Option<GpuRequirements>,
    /// Operating system requirements
    pub os: Vec<String>,
    /// Architecture requirements (e.g., "x86_64", "aarch64")
    pub arch: Vec<String>,
}

impl SystemRequirements {
    /// Creates new system requirements.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets minimum memory requirement.
    pub fn min_memory_mb(mut self, memory_mb: u64) -> Self {
        self.min_memory_mb = Some(memory_mb);
        self
    }

    /// Sets minimum disk space requirement.
    pub fn min_disk_mb(mut self, disk_mb: u64) -> Self {
        self.min_disk_mb = Some(disk_mb);
        self
    }

    /// Adds CPU feature requirement.
    pub fn cpu_feature(mut self, feature: String) -> Self {
        self.cpu_features.push(feature);
        self
    }

    /// Sets GPU requirements.
    pub fn gpu(mut self, gpu: GpuRequirements) -> Self {
        self.gpu = Some(gpu);
        self
    }

    /// Adds OS requirement.
    pub fn os(mut self, os: String) -> Self {
        self.os.push(os);
        self
    }

    /// Adds architecture requirement.
    pub fn arch(mut self, arch: String) -> Self {
        self.arch.push(arch);
        self
    }

    /// Validates the system requirements.
    pub fn validate(&self) -> Result<()> {
        if let Some(gpu) = &self.gpu {
            gpu.validate()?;
        }
        Ok(())
    }
}

/// GPU requirements specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuRequirements {
    /// Minimum GPU memory in MB
    pub min_memory_mb: u64,
    /// Required compute capability (for CUDA)
    pub compute_capability: Option<String>,
    /// Required GPU vendors (e.g., "nvidia", "amd", "intel")
    pub vendors: Vec<String>,
    /// Required GPU APIs (e.g., "cuda", "opencl", "vulkan")
    pub apis: Vec<String>,
}

impl GpuRequirements {
    /// Creates new GPU requirements.
    pub fn new(min_memory_mb: u64) -> Self {
        Self {
            min_memory_mb,
            compute_capability: None,
            vendors: Vec::new(),
            apis: Vec::new(),
        }
    }

    /// Sets compute capability requirement.
    pub fn compute_capability(mut self, capability: String) -> Self {
        self.compute_capability = Some(capability);
        self
    }

    /// Adds vendor requirement.
    pub fn vendor(mut self, vendor: String) -> Self {
        self.vendors.push(vendor);
        self
    }

    /// Adds API requirement.
    pub fn api(mut self, api: String) -> Self {
        self.apis.push(api);
        self
    }

    /// Validates GPU requirements.
    pub fn validate(&self) -> Result<()> {
        if self.min_memory_mb == 0 {
            return Err(TrustformersError::invalid_config(
                "GPU memory requirement must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }
}
