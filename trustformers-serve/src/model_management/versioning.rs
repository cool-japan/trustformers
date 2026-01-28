//! Model Versioning System
//!
//! Handles semantic versioning, compatibility checks, and version resolution.

use crate::model_management::{ModelError, ModelResult};
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap},
    fmt,
    str::FromStr,
};

/// Semantic version structure
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SemanticVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub pre_release: Option<String>,
    pub build: Option<String>,
}

impl SemanticVersion {
    /// Create a new semantic version
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
            pre_release: None,
            build: None,
        }
    }

    /// Create with pre-release tag
    pub fn with_pre_release(mut self, pre_release: String) -> Self {
        self.pre_release = Some(pre_release);
        self
    }

    /// Create with build metadata
    pub fn with_build(mut self, build: String) -> Self {
        self.build = Some(build);
        self
    }

    /// Check if this is a pre-release version
    pub fn is_pre_release(&self) -> bool {
        self.pre_release.is_some()
    }

    /// Check if this version is compatible with another (same major version)
    pub fn is_compatible_with(&self, other: &SemanticVersion) -> bool {
        // Versions are compatible if they have the same major version
        // In semantic versioning, same major version means backward compatible
        self.major == other.major
    }

    /// Get the next major version
    pub fn next_major(&self) -> SemanticVersion {
        SemanticVersion::new(self.major + 1, 0, 0)
    }

    /// Get the next minor version
    pub fn next_minor(&self) -> SemanticVersion {
        SemanticVersion::new(self.major, self.minor + 1, 0)
    }

    /// Get the next patch version
    pub fn next_patch(&self) -> SemanticVersion {
        SemanticVersion::new(self.major, self.minor, self.patch + 1)
    }
}

impl fmt::Display for SemanticVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;

        if let Some(ref pre_release) = self.pre_release {
            write!(f, "-{}", pre_release)?;
        }

        if let Some(ref build) = self.build {
            write!(f, "+{}", build)?;
        }

        Ok(())
    }
}

impl FromStr for SemanticVersion {
    type Err = ModelError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut parts = s.split('+');
        let version_part = parts.next().unwrap();
        let build = parts.next().map(|s| s.to_string());

        let mut parts = version_part.split('-');
        let version_core = parts.next().unwrap();
        let pre_release = parts.next().map(|s| s.to_string());

        let version_nums: Vec<&str> = version_core.split('.').collect();
        if version_nums.len() != 3 {
            return Err(ModelError::InvalidConfig {
                error: format!("Invalid version format: {}", s),
            });
        }

        let major = version_nums[0].parse().map_err(|_| ModelError::InvalidConfig {
            error: format!("Invalid major version: {}", version_nums[0]),
        })?;

        let minor = version_nums[1].parse().map_err(|_| ModelError::InvalidConfig {
            error: format!("Invalid minor version: {}", version_nums[1]),
        })?;

        let patch = version_nums[2].parse().map_err(|_| ModelError::InvalidConfig {
            error: format!("Invalid patch version: {}", version_nums[2]),
        })?;

        Ok(SemanticVersion {
            major,
            minor,
            patch,
            pre_release,
            build,
        })
    }
}

impl PartialOrd for SemanticVersion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SemanticVersion {
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare major, minor, patch
        match (
            self.major.cmp(&other.major),
            self.minor.cmp(&other.minor),
            self.patch.cmp(&other.patch),
        ) {
            (Ordering::Equal, Ordering::Equal, Ordering::Equal) => {
                // Compare pre-release versions
                match (&self.pre_release, &other.pre_release) {
                    (None, None) => Ordering::Equal,
                    (None, Some(_)) => Ordering::Greater, // Release > pre-release
                    (Some(_), None) => Ordering::Less,    // Pre-release < release
                    (Some(a), Some(b)) => a.cmp(b),       // Compare pre-release strings
                }
            },
            (Ordering::Equal, Ordering::Equal, patch_cmp) => patch_cmp,
            (Ordering::Equal, minor_cmp, _) => minor_cmp,
            (major_cmp, _, _) => major_cmp,
        }
    }
}

/// Version constraint for dependency resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VersionConstraint {
    /// Exact version match
    Exact(SemanticVersion),
    /// Compatible version (same major, >= minor.patch)
    Compatible(SemanticVersion),
    /// Greater than or equal
    GreaterOrEqual(SemanticVersion),
    /// Less than
    LessThan(SemanticVersion),
    /// Version range
    Range {
        min: SemanticVersion,
        max: SemanticVersion,
        include_min: bool,
        include_max: bool,
    },
    /// Any version
    Any,
}

impl VersionConstraint {
    /// Check if a version satisfies this constraint
    pub fn satisfies(&self, version: &SemanticVersion) -> bool {
        match self {
            VersionConstraint::Exact(v) => version == v,
            VersionConstraint::Compatible(v) => version.is_compatible_with(v),
            VersionConstraint::GreaterOrEqual(v) => version >= v,
            VersionConstraint::LessThan(v) => version < v,
            VersionConstraint::Range {
                min,
                max,
                include_min,
                include_max,
            } => {
                let min_check = if *include_min { version >= min } else { version > min };
                let max_check = if *include_max { version <= max } else { version < max };
                min_check && max_check
            },
            VersionConstraint::Any => true,
        }
    }
}

/// Model dependency information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDependency {
    /// Name of the dependency
    pub name: String,
    /// Version constraint
    pub version_constraint: VersionConstraint,
    /// Whether this dependency is optional
    pub optional: bool,
    /// Features required from this dependency
    pub features: Vec<String>,
}

/// Version compatibility information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityInfo {
    /// Minimum compatible version
    pub min_version: SemanticVersion,
    /// Maximum compatible version (if any)
    pub max_version: Option<SemanticVersion>,
    /// Breaking changes from this version
    pub breaking_changes: Vec<String>,
    /// Deprecated features in this version
    pub deprecated_features: Vec<String>,
    /// Migration guide URL or text
    pub migration_guide: Option<String>,
}

/// Model version metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionMetadata {
    /// Version number
    pub version: SemanticVersion,
    /// Release date
    pub release_date: chrono::DateTime<chrono::Utc>,
    /// Release notes
    pub release_notes: String,
    /// Model dependencies
    pub dependencies: Vec<ModelDependency>,
    /// Compatibility information
    pub compatibility: CompatibilityInfo,
    /// Performance benchmarks for this version
    pub benchmarks: HashMap<String, f64>,
    /// Model size in bytes
    pub model_size: u64,
    /// Checksum for integrity verification
    pub checksum: String,
}

/// Version manager for handling model versioning
pub struct VersionManager {
    /// Version metadata indexed by model name and version
    versions: HashMap<String, BTreeMap<SemanticVersion, VersionMetadata>>,
}

impl VersionManager {
    /// Create a new version manager
    pub fn new() -> Self {
        Self {
            versions: HashMap::new(),
        }
    }

    /// Register version metadata
    pub fn register_version(
        &mut self,
        model_name: &str,
        metadata: VersionMetadata,
    ) -> ModelResult<()> {
        self.versions
            .entry(model_name.to_string())
            .or_default()
            .insert(metadata.version.clone(), metadata);
        Ok(())
    }

    /// Get version metadata
    pub fn get_version_metadata(
        &self,
        model_name: &str,
        version: &SemanticVersion,
    ) -> Option<&VersionMetadata> {
        self.versions.get(model_name)?.get(version)
    }

    /// Get latest version for a model
    pub fn get_latest_version(&self, model_name: &str) -> Option<&VersionMetadata> {
        self.versions.get(model_name)?.iter().last().map(|(_, metadata)| metadata)
    }

    /// Get latest stable version (non-pre-release)
    pub fn get_latest_stable_version(&self, model_name: &str) -> Option<&VersionMetadata> {
        self.versions
            .get(model_name)?
            .iter()
            .rev()
            .find(|(version, _)| !version.is_pre_release())
            .map(|(_, metadata)| metadata)
    }

    /// Resolve version constraint to specific version
    pub fn resolve_version(
        &self,
        model_name: &str,
        constraint: &VersionConstraint,
    ) -> Option<&VersionMetadata> {
        let versions = self.versions.get(model_name)?;

        versions
            .iter()
            .rev() // Start from latest version
            .find(|(version, _)| constraint.satisfies(version))
            .map(|(_, metadata)| metadata)
    }

    /// List all versions for a model
    pub fn list_versions(&self, model_name: &str) -> Vec<&VersionMetadata> {
        if let Some(versions) = self.versions.get(model_name) {
            versions.values().collect()
        } else {
            Vec::new()
        }
    }

    /// Check version compatibility
    pub fn check_compatibility(
        &self,
        model_name: &str,
        from_version: &SemanticVersion,
        to_version: &SemanticVersion,
    ) -> ModelResult<bool> {
        let from_metadata =
            self.get_version_metadata(model_name, from_version).ok_or_else(|| {
                ModelError::ModelNotFound {
                    id: format!("{}:{}", model_name, from_version),
                }
            })?;

        let _to_metadata = self.get_version_metadata(model_name, to_version).ok_or_else(|| {
            ModelError::ModelNotFound {
                id: format!("{}:{}", model_name, to_version),
            }
        })?;

        // Check if versions are compatible
        let compatible = from_version.is_compatible_with(to_version)
            && to_version >= &from_metadata.compatibility.min_version
            && from_metadata
                .compatibility
                .max_version
                .as_ref()
                .map_or(true, |max| to_version <= max);

        Ok(compatible)
    }

    /// Get upgrade path between versions
    pub fn get_upgrade_path(
        &self,
        model_name: &str,
        from_version: &SemanticVersion,
        to_version: &SemanticVersion,
    ) -> ModelResult<Vec<SemanticVersion>> {
        if !self.check_compatibility(model_name, from_version, to_version)? {
            return Err(ModelError::VersionConflict {
                current: from_version.to_string(),
                requested: to_version.to_string(),
            });
        }

        let versions = self.versions.get(model_name).ok_or_else(|| ModelError::ModelNotFound {
            id: model_name.to_string(),
        })?;

        let mut path = Vec::new();
        for (version, _) in versions.range(from_version..=to_version) {
            if version > from_version && version <= to_version {
                path.push(version.clone());
            }
        }

        Ok(path)
    }

    /// Validate dependencies for a version
    pub fn validate_dependencies(
        &self,
        model_name: &str,
        version: &SemanticVersion,
        available_models: &HashMap<String, SemanticVersion>,
    ) -> ModelResult<Vec<String>> {
        let metadata = self.get_version_metadata(model_name, version).ok_or_else(|| {
            ModelError::ModelNotFound {
                id: format!("{}:{}", model_name, version),
            }
        })?;

        let mut missing_deps = Vec::new();

        for dep in &metadata.dependencies {
            if dep.optional {
                continue;
            }

            if let Some(available_version) = available_models.get(&dep.name) {
                if !dep.version_constraint.satisfies(available_version) {
                    missing_deps.push(format!(
                        "{} (required: {:?}, available: {})",
                        dep.name, dep.version_constraint, available_version
                    ));
                }
            } else {
                missing_deps.push(format!(
                    "{} (required: {:?}, not available)",
                    dep.name, dep.version_constraint
                ));
            }
        }

        Ok(missing_deps)
    }
}

impl Default for VersionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_version() {
        let v1 = SemanticVersion::new(1, 2, 3);
        let v2 = SemanticVersion::new(1, 2, 4);
        let v3 = SemanticVersion::new(2, 0, 0);

        assert!(v1 < v2);
        assert!(v2 < v3);
        assert!(v1.is_compatible_with(&v2));
        assert!(!v1.is_compatible_with(&v3));
    }

    #[test]
    fn test_version_parsing() {
        let version: SemanticVersion = "1.2.3-alpha+build".parse().unwrap();
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 2);
        assert_eq!(version.patch, 3);
        assert_eq!(version.pre_release, Some("alpha".to_string()));
        assert_eq!(version.build, Some("build".to_string()));
    }

    #[test]
    fn test_version_constraints() {
        let version = SemanticVersion::new(1, 2, 3);

        let exact = VersionConstraint::Exact(version.clone());
        assert!(exact.satisfies(&version));

        let compatible = VersionConstraint::Compatible(SemanticVersion::new(1, 2, 0));
        assert!(compatible.satisfies(&version));

        let range = VersionConstraint::Range {
            min: SemanticVersion::new(1, 0, 0),
            max: SemanticVersion::new(2, 0, 0),
            include_min: true,
            include_max: false,
        };
        assert!(range.satisfies(&version));
    }

    #[test]
    fn test_version_manager() {
        let mut manager = VersionManager::new();

        let version = SemanticVersion::new(1, 0, 0);
        let metadata = VersionMetadata {
            version: version.clone(),
            release_date: chrono::Utc::now(),
            release_notes: "Initial release".to_string(),
            dependencies: Vec::new(),
            compatibility: CompatibilityInfo {
                min_version: version.clone(),
                max_version: None,
                breaking_changes: Vec::new(),
                deprecated_features: Vec::new(),
                migration_guide: None,
            },
            benchmarks: HashMap::new(),
            model_size: 1024,
            checksum: "abc123".to_string(),
        };

        manager.register_version("test-model", metadata).unwrap();

        let retrieved = manager.get_version_metadata("test-model", &version);
        assert!(retrieved.is_some());

        let latest = manager.get_latest_version("test-model");
        assert!(latest.is_some());
    }
}
