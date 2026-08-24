//! The `ModelRepository` domain type and its query/mutation methods.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use super::types::{
    AccessControl, ChangeType, FileChange, ModelVersion, PerformanceDiff, RepositoryMetadata,
    RepositoryStats, VersionComparison, Visibility,
};

/// Model repository for managing versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRepository {
    /// Model ID
    pub model_id: String,
    /// All versions of the model
    pub versions: HashMap<String, ModelVersion>,
    /// Version history (ordered by creation time)
    pub version_history: Vec<String>,
    /// Repository metadata
    pub metadata: RepositoryMetadata,
    /// Access control
    pub access_control: AccessControl,
}
impl ModelRepository {
    /// Create a new model repository
    pub fn new(model_id: String, owner: String) -> Self {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        Self {
            model_id: model_id.clone(),
            versions: HashMap::new(),
            version_history: Vec::new(),
            metadata: RepositoryMetadata {
                name: model_id.clone(),
                description: None,
                owner,
                visibility: Visibility::Public,
                default_version: "main".to_string(),
                tags: Vec::new(),
                created_at: now,
                updated_at: now,
                stats: RepositoryStats {
                    version_count: 0,
                    total_downloads: 0,
                    stars: 0,
                    forks: 0,
                    contributors: 1,
                    last_activity: now,
                },
            },
            access_control: AccessControl {
                permissions: HashMap::new(),
                api_key: None,
                whitelist: Vec::new(),
                blacklist: Vec::new(),
            },
        }
    }
    /// Get the latest version
    pub fn latest_version(&self) -> Option<&ModelVersion> {
        self.version_history.last().and_then(|v| self.versions.get(v))
    }
    /// Get version by identifier
    pub fn get_version(&self, version: &str) -> Option<&ModelVersion> {
        self.versions.get(version)
    }
    /// List all versions
    pub fn list_versions(&self) -> Vec<&ModelVersion> {
        self.version_history.iter().filter_map(|v| self.versions.get(v)).collect()
    }
    /// Compare two versions
    pub fn compare_versions(&self, from: &str, to: &str) -> Option<VersionComparison> {
        let from_version = self.versions.get(from)?;
        let to_version = self.versions.get(to)?;
        Some(VersionComparison {
            from_version: from.to_string(),
            to_version: to.to_string(),
            size_diff: to_version.size_bytes as i64 - from_version.size_bytes as i64,
            changes: self.compute_changes(from_version, to_version),
            performance_diff: self.compute_performance_diff(from_version, to_version),
            created_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
        })
    }
    /// Compute what changed between two versions.
    ///
    /// `ModelVersion` tracks one aggregate checksum/size per version, not a
    /// per-file manifest (unlike a real Hub repo's git tree), so this can
    /// only ever detect "the model's contents changed as a whole" — never
    /// attribute that to a specific file. An earlier revision fabricated a
    /// hardcoded `"model.safetensors"` path regardless of what (if
    /// anything) actually changed, which claimed a specificity this data
    /// model doesn't have. `path` here is honestly a whole-model marker,
    /// never a guessed filename, and `description` explains what evidence
    /// (checksum vs. size) triggered the entry.
    pub(super) fn compute_changes(
        &self,
        from: &ModelVersion,
        to: &ModelVersion,
    ) -> Vec<FileChange> {
        let mut changes = Vec::new();
        let checksum_changed = from.checksum != to.checksum;
        let size_changed = from.size_bytes != to.size_bytes;
        if checksum_changed || size_changed {
            let description = match (from.checksum.as_deref(), to.checksum.as_deref()) {
                (None, Some(_)) => "Integrity checksum added".to_string(),
                (Some(_), None) => "Integrity checksum removed".to_string(),
                (Some(old), Some(new)) if old != new => {
                    "Model contents changed (checksum differs)".to_string()
                },
                _ if size_changed => "Model size changed".to_string(),
                _ => "Model version updated".to_string(),
            };
            changes.push(FileChange {
                path: "(entire model)".to_string(),
                change_type: ChangeType::Modified,
                old_size: Some(from.size_bytes),
                new_size: to.size_bytes,
                checksum: to.checksum.clone(),
                description: Some(description),
            });
        }
        changes
    }
    pub(super) fn compute_performance_diff(
        &self,
        from: &ModelVersion,
        to: &ModelVersion,
    ) -> PerformanceDiff {
        let from_metrics = from.metrics.as_ref();
        let to_metrics = to.metrics.as_ref();
        PerformanceDiff {
            accuracy_diff: match (from_metrics, to_metrics) {
                (Some(from), Some(to)) => to.accuracy.zip(from.accuracy).map(|(a, b)| a - b),
                _ => None,
            },
            loss_diff: match (from_metrics, to_metrics) {
                (Some(from), Some(to)) => to.loss.zip(from.loss).map(|(a, b)| a - b),
                _ => None,
            },
            speed_diff: match (from_metrics, to_metrics) {
                (Some(from), Some(to)) => {
                    to.inference_speed.zip(from.inference_speed).map(|(a, b)| a - b)
                },
                _ => None,
            },
            memory_diff: match (from_metrics, to_metrics) {
                (Some(from), Some(to)) => {
                    to.memory_usage.zip(from.memory_usage).map(|(a, b)| a - b)
                },
                _ => None,
            },
        }
    }
}
