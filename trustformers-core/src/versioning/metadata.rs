//! Model metadata and version definitions

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Model metadata containing version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Human-readable description
    pub description: String,
    /// Creator/author of this version
    pub created_by: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Model type (e.g., "transformer", "cnn", "rnn")
    pub model_type: String,
    /// Model architecture (e.g., "gpt2", "bert", "llama")
    pub architecture: Option<String>,
    /// Model size/parameters (e.g., "125M", "1.3B", "7B")
    pub size: Option<String>,
    /// Training configuration used
    pub training_config: Option<serde_json::Value>,
    /// Performance metrics
    pub metrics: HashMap<String, f64>,
    /// Tags for categorization
    pub tags: Vec<ModelTag>,
    /// Custom attributes
    pub attributes: HashMap<String, serde_json::Value>,
    /// Source information (dataset, training run, etc.)
    pub source: Option<ModelSource>,
    /// Checksum for integrity verification
    pub checksum: Option<String>,
    /// Model size in bytes
    pub size_bytes: Option<u64>,
    /// Compatible framework versions
    pub framework_versions: Vec<String>,
}

impl ModelMetadata {
    /// Create a new metadata builder
    pub fn builder() -> ModelMetadataBuilder {
        ModelMetadataBuilder::new()
    }

    /// Add a metric
    pub fn add_metric(&mut self, name: String, value: f64) {
        self.metrics.insert(name, value);
    }

    /// Add a tag
    pub fn add_tag(&mut self, tag: ModelTag) {
        self.tags.push(tag);
    }

    /// Get metric value
    pub fn get_metric(&self, name: &str) -> Option<f64> {
        self.metrics.get(name).copied()
    }

    /// Check if model has tag
    pub fn has_tag(&self, tag_name: &str) -> bool {
        self.tags.iter().any(|t| t.name == tag_name)
    }
}

/// Builder for model metadata
pub struct ModelMetadataBuilder {
    description: Option<String>,
    created_by: Option<String>,
    model_type: Option<String>,
    architecture: Option<String>,
    size: Option<String>,
    training_config: Option<serde_json::Value>,
    metrics: HashMap<String, f64>,
    tags: Vec<ModelTag>,
    attributes: HashMap<String, serde_json::Value>,
    source: Option<ModelSource>,
    checksum: Option<String>,
    size_bytes: Option<u64>,
    framework_versions: Vec<String>,
}

impl ModelMetadataBuilder {
    fn new() -> Self {
        Self {
            description: None,
            created_by: None,
            model_type: None,
            architecture: None,
            size: None,
            training_config: None,
            metrics: HashMap::new(),
            tags: Vec::new(),
            attributes: HashMap::new(),
            source: None,
            checksum: None,
            size_bytes: None,
            framework_versions: Vec::new(),
        }
    }

    pub fn description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    pub fn created_by(mut self, created_by: String) -> Self {
        self.created_by = Some(created_by);
        self
    }

    pub fn model_type(mut self, model_type: String) -> Self {
        self.model_type = Some(model_type);
        self
    }

    pub fn architecture(mut self, architecture: String) -> Self {
        self.architecture = Some(architecture);
        self
    }

    pub fn size(mut self, size: String) -> Self {
        self.size = Some(size);
        self
    }

    pub fn training_config(mut self, config: serde_json::Value) -> Self {
        self.training_config = Some(config);
        self
    }

    pub fn metric(mut self, name: String, value: f64) -> Self {
        self.metrics.insert(name, value);
        self
    }

    pub fn tag(mut self, tag: ModelTag) -> Self {
        self.tags.push(tag);
        self
    }

    pub fn attribute(mut self, key: String, value: serde_json::Value) -> Self {
        self.attributes.insert(key, value);
        self
    }

    pub fn source(mut self, source: ModelSource) -> Self {
        self.source = Some(source);
        self
    }

    pub fn checksum(mut self, checksum: String) -> Self {
        self.checksum = Some(checksum);
        self
    }

    pub fn size_bytes(mut self, size_bytes: u64) -> Self {
        self.size_bytes = Some(size_bytes);
        self
    }

    pub fn framework_version(mut self, version: String) -> Self {
        self.framework_versions.push(version);
        self
    }

    pub fn build(self) -> ModelMetadata {
        ModelMetadata {
            description: self.description.unwrap_or_default(),
            created_by: self.created_by.unwrap_or_default(),
            created_at: Utc::now(),
            model_type: self.model_type.unwrap_or_default(),
            architecture: self.architecture,
            size: self.size,
            training_config: self.training_config,
            metrics: self.metrics,
            tags: self.tags,
            attributes: self.attributes,
            source: self.source,
            checksum: self.checksum,
            size_bytes: self.size_bytes,
            framework_versions: self.framework_versions,
        }
    }
}

/// Model source information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSource {
    /// Source type (e.g., "training", "fine_tuning", "conversion")
    pub source_type: String,
    /// Training dataset name/identifier
    pub dataset: Option<String>,
    /// Training run identifier
    pub training_run_id: Option<String>,
    /// Base model (for fine-tuned models)
    pub base_model: Option<String>,
    /// Training configuration reference
    pub config_ref: Option<String>,
    /// Additional source metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Model tag for categorization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelTag {
    pub name: String,
    pub value: Option<String>,
    pub category: Option<String>,
}

impl ModelTag {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            value: None,
            category: None,
        }
    }

    pub fn with_value(name: &str, value: &str) -> Self {
        Self {
            name: name.to_string(),
            value: Some(value.to_string()),
            category: None,
        }
    }

    pub fn with_category(name: &str, value: &str, category: &str) -> Self {
        Self {
            name: name.to_string(),
            value: Some(value.to_string()),
            category: Some(category.to_string()),
        }
    }
}

/// A versioned model containing metadata and artifact references
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedModel {
    /// Unique identifier
    id: Uuid,
    /// Model name
    model_name: String,
    /// Version string (e.g., "1.0.0", "v2.1-beta")
    version: String,
    /// Model metadata
    metadata: ModelMetadata,
    /// Artifact IDs
    artifact_ids: Vec<Uuid>,
    /// Parent version (for incremental versions)
    parent_version: Option<Uuid>,
    /// Child versions (derived from this version)
    child_versions: Vec<Uuid>,
}

impl VersionedModel {
    /// Create a new versioned model
    pub fn new(
        model_name: String,
        version: String,
        metadata: ModelMetadata,
        artifact_ids: Vec<Uuid>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            model_name,
            version,
            metadata,
            artifact_ids,
            parent_version: None,
            child_versions: Vec::new(),
        }
    }

    /// Create a versioned model with parent
    pub fn with_parent(
        model_name: String,
        version: String,
        metadata: ModelMetadata,
        artifact_ids: Vec<Uuid>,
        parent_id: Uuid,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            model_name,
            version,
            metadata,
            artifact_ids,
            parent_version: Some(parent_id),
            child_versions: Vec::new(),
        }
    }

    /// Get unique identifier
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Get model name
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Get version string
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Get metadata
    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// Get artifact IDs
    pub fn artifact_ids(&self) -> &[Uuid] {
        &self.artifact_ids
    }

    /// Get parent version ID
    pub fn parent_version(&self) -> Option<Uuid> {
        self.parent_version
    }

    /// Get child version IDs
    pub fn child_versions(&self) -> &[Uuid] {
        &self.child_versions
    }

    /// Add child version
    pub fn add_child(&mut self, child_id: Uuid) {
        if !self.child_versions.contains(&child_id) {
            self.child_versions.push(child_id);
        }
    }

    /// Remove child version
    pub fn remove_child(&mut self, child_id: Uuid) {
        self.child_versions.retain(|&id| id != child_id);
    }

    /// Check if this is a root version (no parent)
    pub fn is_root(&self) -> bool {
        self.parent_version.is_none()
    }

    /// Check if this is a leaf version (no children)
    pub fn is_leaf(&self) -> bool {
        self.child_versions.is_empty()
    }

    /// Get full qualified name
    pub fn qualified_name(&self) -> String {
        format!("{}:{}", self.model_name, self.version)
    }

    /// Validate version format
    pub fn validate_version_format(&self) -> Result<()> {
        // Basic semantic version validation
        if self.version.is_empty() {
            anyhow::bail!("Version cannot be empty");
        }

        // Allow semver, git tags, or custom formats
        if !self.is_valid_version_format() {
            anyhow::bail!("Invalid version format: {}", self.version);
        }

        Ok(())
    }

    fn is_valid_version_format(&self) -> bool {
        // Accept semver (1.0.0), git-style (v1.0.0), or custom formats
        let version = &self.version;

        // Semver pattern
        if regex::Regex::new(r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$")
            .unwrap()
            .is_match(version)
        {
            return true;
        }

        // Git tag pattern
        if regex::Regex::new(r"^v?\d+\.\d+(\.\d+)?(-[a-zA-Z0-9.-]+)?$")
            .unwrap()
            .is_match(version)
        {
            return true;
        }

        // Custom format (alphanumeric, dots, dashes, underscores)
        if regex::Regex::new(r"^[a-zA-Z0-9._-]+$").unwrap().is_match(version) {
            return true;
        }

        false
    }
}

/// Version comparison for sorting
impl PartialOrd for VersionedModel {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for VersionedModel {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // First compare by model name
        match self.model_name.cmp(&other.model_name) {
            std::cmp::Ordering::Equal => {
                // Then by creation time
                self.metadata.created_at.cmp(&other.metadata.created_at)
            },
            other => other,
        }
    }
}

impl PartialEq for VersionedModel {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for VersionedModel {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_builder() {
        let metadata = ModelMetadata::builder()
            .description("Test model".to_string())
            .created_by("test_user".to_string())
            .model_type("transformer".to_string())
            .architecture("gpt2".to_string())
            .metric("accuracy".to_string(), 0.95)
            .tag(ModelTag::new("experimental"))
            .build();

        assert_eq!(metadata.description, "Test model");
        assert_eq!(metadata.created_by, "test_user");
        assert_eq!(metadata.model_type, "transformer");
        assert_eq!(metadata.architecture, Some("gpt2".to_string()));
        assert_eq!(metadata.get_metric("accuracy"), Some(0.95));
        assert!(metadata.has_tag("experimental"));
    }

    #[test]
    fn test_versioned_model() {
        let metadata = ModelMetadata::builder()
            .description("Test model".to_string())
            .created_by("test_user".to_string())
            .model_type("transformer".to_string())
            .build();

        let model = VersionedModel::new(
            "test_model".to_string(),
            "1.0.0".to_string(),
            metadata,
            vec![Uuid::new_v4()],
        );

        assert_eq!(model.model_name(), "test_model");
        assert_eq!(model.version(), "1.0.0");
        assert_eq!(model.qualified_name(), "test_model:1.0.0");
        assert!(model.is_root());
        assert!(model.is_leaf());
        assert!(model.validate_version_format().is_ok());
    }

    #[test]
    fn test_version_format_validation() {
        let test_cases = vec![
            ("1.0.0", true),
            ("v1.0.0", true),
            ("2.1.3-beta", true),
            ("1.0.0+build.1", true),
            ("main", true),
            ("experimental-v2", true),
            ("", false),
            ("1.0", true), // Should pass custom format
            ("invalid version!", false),
        ];

        for (version, should_be_valid) in test_cases {
            let metadata = ModelMetadata::builder()
                .description("Test".to_string())
                .created_by("test".to_string())
                .model_type("test".to_string())
                .build();

            let model =
                VersionedModel::new("test".to_string(), version.to_string(), metadata, vec![]);

            let is_valid = model.validate_version_format().is_ok();
            assert_eq!(
                is_valid, should_be_valid,
                "Version '{}' validation failed",
                version
            );
        }
    }

    #[test]
    fn test_model_tags() {
        let tag1 = ModelTag::new("production");
        let tag2 = ModelTag::with_value("environment", "staging");
        let tag3 = ModelTag::with_category("model_type", "llm", "architecture");

        assert_eq!(tag1.name, "production");
        assert_eq!(tag1.value, None);
        assert_eq!(tag1.category, None);

        assert_eq!(tag2.name, "environment");
        assert_eq!(tag2.value, Some("staging".to_string()));
        assert_eq!(tag2.category, None);

        assert_eq!(tag3.name, "model_type");
        assert_eq!(tag3.value, Some("llm".to_string()));
        assert_eq!(tag3.category, Some("architecture".to_string()));
    }
}
