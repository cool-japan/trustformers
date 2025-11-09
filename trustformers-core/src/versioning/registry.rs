//! Model version registry for tracking and querying versions

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tokio::sync::RwLock;
use uuid::Uuid;

use super::metadata::VersionedModel;

/// Model version registry
pub struct ModelRegistry {
    /// Map of version ID to versioned model
    versions: RwLock<HashMap<Uuid, VersionedModel>>,
    /// Index by model name -> version string -> version ID
    name_index: RwLock<HashMap<String, HashMap<String, Uuid>>>,
    /// Index by tag -> version IDs
    tag_index: RwLock<HashMap<String, HashSet<Uuid>>>,
    /// Index by creation time (sorted)
    time_index: RwLock<Vec<(DateTime<Utc>, Uuid)>>,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new() -> Self {
        Self {
            versions: RwLock::new(HashMap::new()),
            name_index: RwLock::new(HashMap::new()),
            tag_index: RwLock::new(HashMap::new()),
            time_index: RwLock::new(Vec::new()),
        }
    }

    /// Register a new model version
    pub async fn register(&self, model: VersionedModel) -> Result<Uuid> {
        let version_id = model.id();
        let model_name = model.model_name().to_string();
        let version_string = model.version().to_string();

        // Check for duplicate versions
        {
            let name_index = self.name_index.read().await;
            if let Some(versions) = name_index.get(&model_name) {
                if versions.contains_key(&version_string) {
                    anyhow::bail!("Version {}:{} already exists", model_name, version_string);
                }
            }
        }

        // Update version storage
        {
            let mut versions = self.versions.write().await;
            versions.insert(version_id, model.clone());
        }

        // Update name index
        {
            let mut name_index = self.name_index.write().await;
            name_index.entry(model_name).or_default().insert(version_string, version_id);
        }

        // Update tag index
        {
            let mut tag_index = self.tag_index.write().await;
            for tag in &model.metadata().tags {
                tag_index.entry(tag.name.clone()).or_default().insert(version_id);
            }
        }

        // Update time index
        {
            let mut time_index = self.time_index.write().await;
            time_index.push((model.metadata().created_at, version_id));
            time_index.sort_by_key(|(time, _)| *time);
        }

        tracing::debug!("Registered model version: {}", version_id);
        Ok(version_id)
    }

    /// Get a model version by ID
    pub async fn get_version(&self, version_id: Uuid) -> Result<Option<VersionedModel>> {
        let versions = self.versions.read().await;
        Ok(versions.get(&version_id).cloned())
    }

    /// Get a model version by name and version string
    pub async fn get_version_by_name(
        &self,
        model_name: &str,
        version: &str,
    ) -> Result<Option<VersionedModel>> {
        let name_index = self.name_index.read().await;
        if let Some(versions) = name_index.get(model_name) {
            if let Some(&version_id) = versions.get(version) {
                let versions_map = self.versions.read().await;
                return Ok(versions_map.get(&version_id).cloned());
            }
        }
        Ok(None)
    }

    /// List all versions for a model
    pub async fn list_versions(&self, model_name: &str) -> Result<Vec<VersionedModel>> {
        let name_index = self.name_index.read().await;
        let versions_map = self.versions.read().await;

        if let Some(versions) = name_index.get(model_name) {
            let mut models: Vec<VersionedModel> =
                versions.values().filter_map(|&id| versions_map.get(&id).cloned()).collect();

            // Sort by creation time (newest first)
            models.sort_by(|a, b| b.metadata().created_at.cmp(&a.metadata().created_at));
            Ok(models)
        } else {
            Ok(vec![])
        }
    }

    /// List all model names
    pub async fn list_models(&self) -> Result<Vec<String>> {
        let name_index = self.name_index.read().await;
        let mut names: Vec<String> = name_index.keys().cloned().collect();
        names.sort();
        Ok(names)
    }

    /// Query versions with filters
    pub async fn query_versions(&self, query: VersionQuery) -> Result<Vec<VersionedModel>> {
        let versions_map = self.versions.read().await;
        let mut results = Vec::new();

        for model in versions_map.values() {
            if self.matches_query(model, &query).await {
                results.push(model.clone());
            }
        }

        // Apply sorting
        self.sort_results(&mut results, &query.sort_by);

        // Apply pagination
        if let Some(limit) = query.limit {
            let offset = query.offset.unwrap_or(0);
            let end = std::cmp::min(offset + limit, results.len());
            results = results[offset..end].to_vec();
        }

        Ok(results)
    }

    /// Remove a version from the registry
    pub async fn remove_version(&self, version_id: Uuid) -> Result<Option<VersionedModel>> {
        // Get the model to remove from indices
        let model = {
            let mut versions = self.versions.write().await;
            versions.remove(&version_id)
        };

        if let Some(ref model) = model {
            // Remove from name index
            {
                let mut name_index = self.name_index.write().await;
                if let Some(versions) = name_index.get_mut(model.model_name()) {
                    versions.remove(model.version());
                    if versions.is_empty() {
                        name_index.remove(model.model_name());
                    }
                }
            }

            // Remove from tag index
            {
                let mut tag_index = self.tag_index.write().await;
                for tag in &model.metadata().tags {
                    if let Some(tag_set) = tag_index.get_mut(&tag.name) {
                        tag_set.remove(&version_id);
                        if tag_set.is_empty() {
                            tag_index.remove(&tag.name);
                        }
                    }
                }
            }

            // Remove from time index
            {
                let mut time_index = self.time_index.write().await;
                time_index.retain(|(_, id)| *id != version_id);
            }

            tracing::debug!("Removed model version: {}", version_id);
        }

        Ok(model)
    }

    /// Get versions by tag
    pub async fn get_versions_by_tag(&self, tag_name: &str) -> Result<Vec<VersionedModel>> {
        let tag_index = self.tag_index.read().await;
        let versions_map = self.versions.read().await;

        if let Some(version_ids) = tag_index.get(tag_name) {
            let models: Vec<VersionedModel> =
                version_ids.iter().filter_map(|&id| versions_map.get(&id).cloned()).collect();
            Ok(models)
        } else {
            Ok(vec![])
        }
    }

    /// Get latest version for a model
    pub async fn get_latest_version(&self, model_name: &str) -> Result<Option<VersionedModel>> {
        let versions = self.list_versions(model_name).await?;
        Ok(versions.into_iter().next()) // Already sorted by creation time (newest first)
    }

    /// Get registry statistics
    pub async fn get_statistics(&self) -> Result<RegistryStatistics> {
        let versions_map = self.versions.read().await;
        let name_index = self.name_index.read().await;
        let tag_index = self.tag_index.read().await;

        let total_versions = versions_map.len();
        let total_models = name_index.len();
        let total_tags = tag_index.len();

        // Calculate storage statistics
        let mut total_artifacts = 0;
        let mut total_size_bytes = 0;

        for model in versions_map.values() {
            total_artifacts += model.artifact_ids().len();
            if let Some(size) = model.metadata().size_bytes {
                total_size_bytes += size;
            }
        }

        Ok(RegistryStatistics {
            total_versions,
            total_models,
            total_tags,
            total_artifacts,
            total_size_bytes,
        })
    }

    // Helper methods

    async fn matches_query(&self, model: &VersionedModel, query: &VersionQuery) -> bool {
        // Model name filter
        if let Some(ref pattern) = query.model_name_pattern {
            if !self.matches_pattern(model.model_name(), pattern) {
                return false;
            }
        }

        // Version filter
        if let Some(ref version_filter) = query.version_filter {
            if !self.matches_version_filter(model, version_filter) {
                return false;
            }
        }

        // Tag filter
        if !query.tags.is_empty() {
            let model_tags: HashSet<String> =
                model.metadata().tags.iter().map(|tag| tag.name.clone()).collect();

            match query.tag_mode {
                TagMatchMode::Any => {
                    if !query.tags.iter().any(|tag| model_tags.contains(tag)) {
                        return false;
                    }
                },
                TagMatchMode::All => {
                    if !query.tags.iter().all(|tag| model_tags.contains(tag)) {
                        return false;
                    }
                },
            }
        }

        // Created date range
        if let Some(ref date_range) = query.created_date_range {
            let created_at = model.metadata().created_at;
            if let Some(start) = date_range.start {
                if created_at < start {
                    return false;
                }
            }
            if let Some(end) = date_range.end {
                if created_at > end {
                    return false;
                }
            }
        }

        // Model type filter
        if let Some(ref model_type) = query.model_type {
            if model.metadata().model_type != *model_type {
                return false;
            }
        }

        true
    }

    fn matches_pattern(&self, text: &str, pattern: &str) -> bool {
        // Simple pattern matching - could be enhanced with regex
        if pattern.contains('*') {
            // Wildcard matching
            let parts: Vec<&str> = pattern.split('*').collect();
            if parts.len() == 2 {
                let prefix = parts[0];
                let suffix = parts[1];
                return text.starts_with(prefix) && text.ends_with(suffix);
            }
        }
        text.contains(pattern)
    }

    fn matches_version_filter(&self, model: &VersionedModel, filter: &VersionFilter) -> bool {
        match filter {
            VersionFilter::Exact(version) => model.version() == version,
            VersionFilter::Prefix(prefix) => model.version().starts_with(prefix),
            VersionFilter::Regex(regex_str) => {
                if let Ok(regex) = regex::Regex::new(regex_str) {
                    regex.is_match(model.version())
                } else {
                    false
                }
            },
        }
    }

    fn sort_results(&self, results: &mut [VersionedModel], sort_by: &SortBy) {
        match sort_by {
            SortBy::CreatedAt(order) => {
                results.sort_by(|a, b| {
                    let cmp = a.metadata().created_at.cmp(&b.metadata().created_at);
                    match order {
                        SortOrder::Ascending => cmp,
                        SortOrder::Descending => cmp.reverse(),
                    }
                });
            },
            SortBy::ModelName(order) => {
                results.sort_by(|a, b| {
                    let cmp = a.model_name().cmp(b.model_name());
                    match order {
                        SortOrder::Ascending => cmp,
                        SortOrder::Descending => cmp.reverse(),
                    }
                });
            },
            SortBy::Version(order) => {
                results.sort_by(|a, b| {
                    let cmp = a.version().cmp(b.version());
                    match order {
                        SortOrder::Ascending => cmp,
                        SortOrder::Descending => cmp.reverse(),
                    }
                });
            },
        }
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Query for searching model versions
#[derive(Debug, Clone)]
pub struct VersionQuery {
    /// Model name pattern (supports wildcards)
    pub model_name_pattern: Option<String>,
    /// Version filter
    pub version_filter: Option<VersionFilter>,
    /// Tags to match
    pub tags: Vec<String>,
    /// Tag matching mode
    pub tag_mode: TagMatchMode,
    /// Created date range
    pub created_date_range: Option<DateRange>,
    /// Model type filter
    pub model_type: Option<String>,
    /// Sort order
    pub sort_by: SortBy,
    /// Pagination offset
    pub offset: Option<usize>,
    /// Pagination limit
    pub limit: Option<usize>,
}

impl Default for VersionQuery {
    fn default() -> Self {
        Self {
            model_name_pattern: None,
            version_filter: None,
            tags: Vec::new(),
            tag_mode: TagMatchMode::Any,
            created_date_range: None,
            model_type: None,
            sort_by: SortBy::CreatedAt(SortOrder::Descending),
            offset: None,
            limit: None,
        }
    }
}

impl VersionQuery {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn model_name_pattern(mut self, pattern: String) -> Self {
        self.model_name_pattern = Some(pattern);
        self
    }

    pub fn version_filter(mut self, filter: VersionFilter) -> Self {
        self.version_filter = Some(filter);
        self
    }

    pub fn with_tag(mut self, tag: String) -> Self {
        self.tags.push(tag);
        self
    }

    pub fn tag_mode(mut self, mode: TagMatchMode) -> Self {
        self.tag_mode = mode;
        self
    }

    pub fn created_after(mut self, date: DateTime<Utc>) -> Self {
        let range = self.created_date_range.get_or_insert(DateRange::default());
        range.start = Some(date);
        self
    }

    pub fn created_before(mut self, date: DateTime<Utc>) -> Self {
        let range = self.created_date_range.get_or_insert(DateRange::default());
        range.end = Some(date);
        self
    }

    pub fn model_type(mut self, model_type: String) -> Self {
        self.model_type = Some(model_type);
        self
    }

    pub fn sort_by(mut self, sort_by: SortBy) -> Self {
        self.sort_by = sort_by;
        self
    }

    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }
}

/// Version filter options
#[derive(Debug, Clone)]
pub enum VersionFilter {
    /// Exact version match
    Exact(String),
    /// Version prefix match
    Prefix(String),
    /// Regex pattern match
    Regex(String),
}

/// Tag matching mode
#[derive(Debug, Clone)]
pub enum TagMatchMode {
    /// Match any of the specified tags
    Any,
    /// Match all of the specified tags
    All,
}

/// Date range filter
#[derive(Debug, Clone, Default)]
pub struct DateRange {
    pub start: Option<DateTime<Utc>>,
    pub end: Option<DateTime<Utc>>,
}

/// Sort options
#[derive(Debug, Clone)]
pub enum SortBy {
    CreatedAt(SortOrder),
    ModelName(SortOrder),
    Version(SortOrder),
}

/// Sort order
#[derive(Debug, Clone)]
pub enum SortOrder {
    Ascending,
    Descending,
}

/// Registry statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryStatistics {
    pub total_versions: usize,
    pub total_models: usize,
    pub total_tags: usize,
    pub total_artifacts: usize,
    pub total_size_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::versioning::metadata::{ModelMetadata, ModelTag};

    async fn create_test_model(name: &str, version: &str, tags: Vec<ModelTag>) -> VersionedModel {
        let mut metadata_builder = ModelMetadata::builder()
            .description(format!("Test model {}", name))
            .created_by("test_user".to_string())
            .model_type("transformer".to_string());

        for tag in tags {
            metadata_builder = metadata_builder.tag(tag);
        }

        let metadata = metadata_builder.build();

        VersionedModel::new(name.to_string(), version.to_string(), metadata, vec![])
    }

    #[tokio::test]
    async fn test_registry_operations() {
        let registry = ModelRegistry::new();

        // Register a model
        let model = create_test_model("gpt2", "1.0.0", vec![ModelTag::new("production")]).await;
        let version_id = registry.register(model.clone()).await.unwrap();
        assert_eq!(version_id, model.id());

        // Get by ID
        let retrieved = registry.get_version(version_id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().version(), "1.0.0");

        // Get by name and version
        let retrieved = registry.get_version_by_name("gpt2", "1.0.0").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().model_name(), "gpt2");

        // List versions
        let versions = registry.list_versions("gpt2").await.unwrap();
        assert_eq!(versions.len(), 1);

        // List models
        let models = registry.list_models().await.unwrap();
        assert_eq!(models, vec!["gpt2"]);
    }

    #[tokio::test]
    async fn test_query_functionality() {
        let registry = ModelRegistry::new();

        // Register multiple models
        let models = vec![
            create_test_model("gpt2", "1.0.0", vec![ModelTag::new("production")]).await,
            create_test_model("gpt2", "1.1.0", vec![ModelTag::new("staging")]).await,
            create_test_model("bert", "1.0.0", vec![ModelTag::new("production")]).await,
        ];

        for model in models {
            registry.register(model).await.unwrap();
        }

        // Query by model name pattern
        let query = VersionQuery::new().model_name_pattern("gpt*".to_string());
        let results = registry.query_versions(query).await.unwrap();
        assert_eq!(results.len(), 2);

        // Query by tag
        let query = VersionQuery::new().with_tag("production".to_string());
        let results = registry.query_versions(query).await.unwrap();
        assert_eq!(results.len(), 2);

        // Query with limit
        let query = VersionQuery::new().limit(1);
        let results = registry.query_versions(query).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_tag_operations() {
        let registry = ModelRegistry::new();

        let model = create_test_model(
            "test",
            "1.0.0",
            vec![ModelTag::new("production"), ModelTag::new("gpu")],
        )
        .await;

        registry.register(model).await.unwrap();

        // Get by tag
        let results = registry.get_versions_by_tag("production").await.unwrap();
        assert_eq!(results.len(), 1);

        let results = registry.get_versions_by_tag("nonexistent").await.unwrap();
        assert_eq!(results.len(), 0);
    }

    #[tokio::test]
    async fn test_duplicate_prevention() {
        let registry = ModelRegistry::new();

        let model1 = create_test_model("test", "1.0.0", vec![]).await;
        let model2 = create_test_model("test", "1.0.0", vec![]).await;

        registry.register(model1).await.unwrap();
        let result = registry.register(model2).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_registry_statistics() {
        let registry = ModelRegistry::new();

        let model = create_test_model("test", "1.0.0", vec![ModelTag::new("test")]).await;
        registry.register(model).await.unwrap();

        let stats = registry.get_statistics().await.unwrap();
        assert_eq!(stats.total_versions, 1);
        assert_eq!(stats.total_models, 1);
        assert_eq!(stats.total_tags, 1);
    }
}
