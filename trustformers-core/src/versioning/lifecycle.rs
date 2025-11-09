//! Version lifecycle management

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Version lifecycle manager
pub struct VersionLifecycle {
    /// Current status of each version
    version_status: RwLock<HashMap<Uuid, VersionStatus>>,
    /// Lifecycle history for each version
    lifecycle_history: RwLock<HashMap<Uuid, Vec<LifecycleEvent>>>,
    /// Lifecycle policies
    policies: RwLock<LifecyclePolicies>,
}

impl VersionLifecycle {
    /// Create a new version lifecycle manager
    pub fn new() -> Self {
        Self {
            version_status: RwLock::new(HashMap::new()),
            lifecycle_history: RwLock::new(HashMap::new()),
            policies: RwLock::new(LifecyclePolicies::default()),
        }
    }

    /// Initialize a new version in the lifecycle
    pub async fn initialize_version(&self, version_id: Uuid) -> Result<()> {
        {
            let mut status_map = self.version_status.write().await;
            status_map.insert(version_id, VersionStatus::Development);
        }

        {
            let mut history_map = self.lifecycle_history.write().await;
            let event = LifecycleEvent {
                transition: VersionTransition::Initialize,
                from_status: None,
                to_status: VersionStatus::Development,
                timestamp: Utc::now(),
                reason: "Version initialized".to_string(),
                triggered_by: "system".to_string(),
            };
            history_map.insert(version_id, vec![event]);
        }

        tracing::info!("Initialized version {} in Development status", version_id);
        Ok(())
    }

    /// Get current status of a version
    pub async fn get_status(&self, version_id: Uuid) -> Result<VersionStatus> {
        let status_map = self.version_status.read().await;
        status_map
            .get(&version_id)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Version {} not found", version_id))
    }

    /// Transition a version to a new status
    pub async fn transition(&self, version_id: Uuid, transition: VersionTransition) -> Result<()> {
        self.transition_with_reason(version_id, transition, "Manual transition", "user")
            .await
    }

    /// Transition with reason and triggerer
    pub async fn transition_with_reason(
        &self,
        version_id: Uuid,
        transition: VersionTransition,
        reason: &str,
        triggered_by: &str,
    ) -> Result<()> {
        let current_status = self.get_status(version_id).await?;
        let new_status = self.validate_transition(current_status, &transition)?;

        // Check policies
        let policies = self.policies.read().await;
        if !policies.allows_transition(current_status, new_status) {
            anyhow::bail!(
                "Transition from {:?} to {:?} is not allowed by policy",
                current_status,
                new_status
            );
        }

        // Update status
        {
            let mut status_map = self.version_status.write().await;
            status_map.insert(version_id, new_status);
        }

        // Record history
        {
            let mut history_map = self.lifecycle_history.write().await;
            let event = LifecycleEvent {
                transition,
                from_status: Some(current_status),
                to_status: new_status,
                timestamp: Utc::now(),
                reason: reason.to_string(),
                triggered_by: triggered_by.to_string(),
            };

            history_map.entry(version_id).or_default().push(event);
        }

        tracing::info!(
            "Transitioned version {} from {:?} to {:?}: {}",
            version_id,
            current_status,
            new_status,
            reason
        );

        Ok(())
    }

    /// Get lifecycle history for a version
    pub async fn get_history(&self, version_id: Uuid) -> Result<Vec<LifecycleEvent>> {
        let history_map = self.lifecycle_history.read().await;
        Ok(history_map.get(&version_id).cloned().unwrap_or_default())
    }

    /// Get all versions in a specific status
    pub async fn get_versions_by_status(&self, status: VersionStatus) -> Result<Vec<Uuid>> {
        let status_map = self.version_status.read().await;
        let versions: Vec<Uuid> =
            status_map.iter().filter(|(_, &s)| s == status).map(|(&id, _)| id).collect();
        Ok(versions)
    }

    /// Check if a version can be promoted
    pub async fn can_promote(&self, version_id: Uuid) -> Result<bool> {
        let current_status = self.get_status(version_id).await?;
        Ok(matches!(current_status, VersionStatus::Staging))
    }

    /// Check if a version can be archived
    pub async fn can_archive(&self, version_id: Uuid) -> Result<bool> {
        let current_status = self.get_status(version_id).await?;
        Ok(!matches!(current_status, VersionStatus::Production))
    }

    /// Auto-archive old versions based on policies
    pub async fn auto_archive(&self) -> Result<Vec<Uuid>> {
        let policies = self.policies.read().await;
        let mut archived_versions = Vec::new();

        if let Some(max_age_days) = policies.auto_archive_after_days {
            let cutoff_date = Utc::now() - chrono::Duration::days(max_age_days as i64);

            // Collect versions to archive first
            let versions_to_archive = {
                let history_map = self.lifecycle_history.read().await;
                let mut to_archive = Vec::new();

                for (&version_id, history) in history_map.iter() {
                    // Find the creation event
                    if let Some(creation_event) = history.first() {
                        if creation_event.timestamp < cutoff_date {
                            to_archive.push(version_id);
                        }
                    }
                }
                to_archive
            };

            // Now archive the versions
            for version_id in versions_to_archive {
                if self.can_archive(version_id).await? {
                    self.transition_with_reason(
                        version_id,
                        VersionTransition::Archive,
                        "Auto-archived due to age policy",
                        "system",
                    )
                    .await?;
                    archived_versions.push(version_id);
                }
            }
        }

        if !archived_versions.is_empty() {
            tracing::info!("Auto-archived {} versions", archived_versions.len());
        }

        Ok(archived_versions)
    }

    /// Update lifecycle policies
    pub async fn update_policies(&self, policies: LifecyclePolicies) -> Result<()> {
        let mut current_policies = self.policies.write().await;
        *current_policies = policies;
        tracing::info!("Updated lifecycle policies");
        Ok(())
    }

    /// Get current policies
    pub async fn get_policies(&self) -> Result<LifecyclePolicies> {
        let policies = self.policies.read().await;
        Ok(policies.clone())
    }

    /// Cleanup version from lifecycle tracking
    pub async fn cleanup_version(&self, version_id: Uuid) -> Result<()> {
        {
            let mut status_map = self.version_status.write().await;
            status_map.remove(&version_id);
        }

        {
            let mut history_map = self.lifecycle_history.write().await;
            history_map.remove(&version_id);
        }

        tracing::debug!("Cleaned up lifecycle tracking for version {}", version_id);
        Ok(())
    }

    /// Get lifecycle statistics
    pub async fn get_statistics(&self) -> Result<LifecycleStatistics> {
        let status_map = self.version_status.read().await;

        let mut counts = HashMap::new();
        for status in [
            VersionStatus::Development,
            VersionStatus::Staging,
            VersionStatus::Production,
            VersionStatus::Archived,
            VersionStatus::Deprecated,
        ] {
            counts.insert(status, 0);
        }

        for &status in status_map.values() {
            *counts.entry(status).or_insert(0) += 1;
        }

        Ok(LifecycleStatistics {
            development_count: counts[&VersionStatus::Development],
            staging_count: counts[&VersionStatus::Staging],
            production_count: counts[&VersionStatus::Production],
            archived_count: counts[&VersionStatus::Archived],
            deprecated_count: counts[&VersionStatus::Deprecated],
            total_versions: status_map.len(),
        })
    }

    // Helper methods

    fn validate_transition(
        &self,
        current_status: VersionStatus,
        transition: &VersionTransition,
    ) -> Result<VersionStatus> {
        let new_status = match (current_status, transition) {
            (VersionStatus::Development, VersionTransition::ToStaging) => VersionStatus::Staging,
            (VersionStatus::Staging, VersionTransition::Promote) => VersionStatus::Production,
            (VersionStatus::Staging, VersionTransition::ToTesting) => VersionStatus::Testing,
            (VersionStatus::Testing, VersionTransition::ToStaging) => VersionStatus::Staging,
            (VersionStatus::Production, VersionTransition::Deprecate) => VersionStatus::Deprecated,
            (_, VersionTransition::Archive) => VersionStatus::Archived,
            (_, VersionTransition::ToTesting) => VersionStatus::Testing,
            (VersionStatus::Archived, VersionTransition::Restore) => VersionStatus::Staging,
            _ => {
                anyhow::bail!(
                    "Invalid transition {:?} from status {:?}",
                    transition,
                    current_status
                );
            },
        };

        Ok(new_status)
    }
}

impl Default for VersionLifecycle {
    fn default() -> Self {
        Self::new()
    }
}

/// Version status in the lifecycle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VersionStatus {
    /// Under development
    Development,
    /// In testing phase
    Testing,
    /// Ready for production testing
    Staging,
    /// Currently deployed in production
    Production,
    /// No longer recommended for use
    Deprecated,
    /// Archived (kept for historical purposes)
    Archived,
}

impl std::fmt::Display for VersionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VersionStatus::Development => write!(f, "Development"),
            VersionStatus::Testing => write!(f, "Testing"),
            VersionStatus::Staging => write!(f, "Staging"),
            VersionStatus::Production => write!(f, "Production"),
            VersionStatus::Deprecated => write!(f, "Deprecated"),
            VersionStatus::Archived => write!(f, "Archived"),
        }
    }
}

/// Version lifecycle transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VersionTransition {
    /// Initialize a new version
    Initialize,
    /// Move to testing
    ToTesting,
    /// Move to staging
    ToStaging,
    /// Promote to production
    Promote,
    /// Deprecate version
    Deprecate,
    /// Archive version
    Archive,
    /// Restore from archive
    Restore,
}

/// Lifecycle event record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleEvent {
    pub transition: VersionTransition,
    pub from_status: Option<VersionStatus>,
    pub to_status: VersionStatus,
    pub timestamp: DateTime<Utc>,
    pub reason: String,
    pub triggered_by: String,
}

/// Lifecycle policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecyclePolicies {
    /// Allowed transitions between statuses
    pub allowed_transitions: HashMap<VersionStatus, Vec<VersionStatus>>,
    /// Maximum number of production versions allowed
    pub max_production_versions: Option<usize>,
    /// Auto-archive versions after this many days
    pub auto_archive_after_days: Option<usize>,
    /// Require approval for production promotion
    pub require_approval_for_production: bool,
    /// Allow rollback from production
    pub allow_production_rollback: bool,
}

impl Default for LifecyclePolicies {
    fn default() -> Self {
        let mut allowed_transitions = HashMap::new();

        allowed_transitions.insert(
            VersionStatus::Development,
            vec![
                VersionStatus::Testing,
                VersionStatus::Staging,
                VersionStatus::Archived,
            ],
        );
        allowed_transitions.insert(
            VersionStatus::Testing,
            vec![
                VersionStatus::Staging,
                VersionStatus::Development,
                VersionStatus::Archived,
            ],
        );
        allowed_transitions.insert(
            VersionStatus::Staging,
            vec![
                VersionStatus::Production,
                VersionStatus::Testing,
                VersionStatus::Archived,
            ],
        );
        allowed_transitions.insert(VersionStatus::Production, vec![VersionStatus::Deprecated]);
        allowed_transitions.insert(VersionStatus::Deprecated, vec![VersionStatus::Archived]);
        allowed_transitions.insert(VersionStatus::Archived, vec![VersionStatus::Staging]);

        Self {
            allowed_transitions,
            max_production_versions: Some(3),
            auto_archive_after_days: Some(365),
            require_approval_for_production: true,
            allow_production_rollback: true,
        }
    }
}

impl LifecyclePolicies {
    /// Check if a transition is allowed
    pub fn allows_transition(&self, from: VersionStatus, to: VersionStatus) -> bool {
        self.allowed_transitions.get(&from).is_some_and(|allowed| allowed.contains(&to))
    }

    /// Set allowed transitions for a status
    pub fn set_allowed_transitions(&mut self, from: VersionStatus, to: Vec<VersionStatus>) {
        self.allowed_transitions.insert(from, to);
    }
}

/// Lifecycle statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleStatistics {
    pub development_count: usize,
    pub staging_count: usize,
    pub production_count: usize,
    pub archived_count: usize,
    pub deprecated_count: usize,
    pub total_versions: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_version_lifecycle() {
        let lifecycle = VersionLifecycle::new();
        let version_id = Uuid::new_v4();

        // Initialize version
        lifecycle.initialize_version(version_id).await.unwrap();
        let status = lifecycle.get_status(version_id).await.unwrap();
        assert_eq!(status, VersionStatus::Development);

        // Transition to staging
        lifecycle.transition(version_id, VersionTransition::ToStaging).await.unwrap();
        let status = lifecycle.get_status(version_id).await.unwrap();
        assert_eq!(status, VersionStatus::Staging);

        // Promote to production
        lifecycle.transition(version_id, VersionTransition::Promote).await.unwrap();
        let status = lifecycle.get_status(version_id).await.unwrap();
        assert_eq!(status, VersionStatus::Production);

        // Check history
        let history = lifecycle.get_history(version_id).await.unwrap();
        assert_eq!(history.len(), 3); // Initialize + 2 transitions
    }

    #[tokio::test]
    async fn test_invalid_transition() {
        let lifecycle = VersionLifecycle::new();
        let version_id = Uuid::new_v4();

        lifecycle.initialize_version(version_id).await.unwrap();

        // Try invalid transition (Development -> Production)
        let result = lifecycle.transition(version_id, VersionTransition::Promote).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_versions_by_status() {
        let lifecycle = VersionLifecycle::new();

        let version1 = Uuid::new_v4();
        let version2 = Uuid::new_v4();

        lifecycle.initialize_version(version1).await.unwrap();
        lifecycle.initialize_version(version2).await.unwrap();

        lifecycle.transition(version1, VersionTransition::ToStaging).await.unwrap();

        let dev_versions =
            lifecycle.get_versions_by_status(VersionStatus::Development).await.unwrap();
        assert_eq!(dev_versions.len(), 1);
        assert!(dev_versions.contains(&version2));

        let staging_versions =
            lifecycle.get_versions_by_status(VersionStatus::Staging).await.unwrap();
        assert_eq!(staging_versions.len(), 1);
        assert!(staging_versions.contains(&version1));
    }

    #[tokio::test]
    async fn test_lifecycle_policies() {
        let mut policies = LifecyclePolicies::default();

        // Test default policy
        assert!(policies.allows_transition(VersionStatus::Development, VersionStatus::Staging));
        assert!(!policies.allows_transition(VersionStatus::Development, VersionStatus::Production));

        // Modify policy
        policies.set_allowed_transitions(
            VersionStatus::Development,
            vec![VersionStatus::Production], // Allow direct promotion
        );

        assert!(policies.allows_transition(VersionStatus::Development, VersionStatus::Production));
        assert!(!policies.allows_transition(VersionStatus::Development, VersionStatus::Staging));
    }

    #[tokio::test]
    async fn test_lifecycle_statistics() {
        let lifecycle = VersionLifecycle::new();

        let version1 = Uuid::new_v4();
        let version2 = Uuid::new_v4();

        lifecycle.initialize_version(version1).await.unwrap();
        lifecycle.initialize_version(version2).await.unwrap();
        lifecycle.transition(version1, VersionTransition::ToStaging).await.unwrap();

        let stats = lifecycle.get_statistics().await.unwrap();
        assert_eq!(stats.development_count, 1);
        assert_eq!(stats.staging_count, 1);
        assert_eq!(stats.total_versions, 2);
    }

    #[tokio::test]
    async fn test_promotion_capability() {
        let lifecycle = VersionLifecycle::new();
        let version_id = Uuid::new_v4();

        lifecycle.initialize_version(version_id).await.unwrap();

        // Cannot promote from development
        assert!(!lifecycle.can_promote(version_id).await.unwrap());

        // Move to staging
        lifecycle.transition(version_id, VersionTransition::ToStaging).await.unwrap();

        // Can promote from staging
        assert!(lifecycle.can_promote(version_id).await.unwrap());
    }
}
