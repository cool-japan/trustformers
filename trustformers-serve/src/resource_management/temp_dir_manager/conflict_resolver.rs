//! Directory Conflict Resolver Implementation
//!
//! This module detects and resolves conflicts that may arise when multiple tests
//! or processes attempt to access the same resources simultaneously.

use super::types::*;

use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};
use tracing::{debug, error, info, instrument, warn};

// ================================================================================================
// Directory Conflict Resolver
// ================================================================================================

/// Resolves conflicts in directory access and allocation
///
/// This component detects and resolves conflicts that may arise when multiple
/// tests or processes attempt to access the same resources simultaneously.
#[derive(Debug)]
pub struct DirectoryConflictResolver {
    /// Active conflicts being tracked
    active_conflicts: Arc<Mutex<HashMap<String, DirectoryConflict>>>,

    /// Conflict resolution history
    resolution_history: Arc<Mutex<Vec<ConflictResolutionRecord>>>,

    /// Conflict detection settings
    detection_settings: ConflictDetectionSettings,

    /// Statistics for conflict resolution
    resolution_stats: Arc<Mutex<ConflictResolutionStatistics>>,
}

/// Settings for conflict detection
#[derive(Debug, Clone)]
pub struct ConflictDetectionSettings {
    /// Enable automatic conflict detection
    pub enable_detection: bool,
    /// Maximum number of directories per test before conflict detection
    pub max_directories_per_test: usize,
    /// Maximum total directories before conflict detection
    pub max_total_directories: usize,
    /// Minimum available disk space threshold
    pub min_disk_space_threshold: u64,
    /// Detection sensitivity level
    pub sensitivity_level: ConflictSensitivity,
}

impl Default for ConflictDetectionSettings {
    fn default() -> Self {
        Self {
            enable_detection: true,
            max_directories_per_test: 10,
            max_total_directories: 100,
            min_disk_space_threshold: 1024 * 1024 * 1024, // 1GB
            sensitivity_level: ConflictSensitivity::Normal,
        }
    }
}

/// Sensitivity levels for conflict detection
#[derive(Debug, Clone, PartialEq)]
pub enum ConflictSensitivity {
    /// Low sensitivity - only detect obvious conflicts
    Low,
    /// Normal sensitivity - standard conflict detection
    Normal,
    /// High sensitivity - aggressive conflict detection
    High,
    /// Custom sensitivity with specific thresholds
    Custom {
        disk_space_threshold: u64,
        concurrency_threshold: usize,
    },
}

/// Record of a conflict resolution attempt
#[derive(Debug, Clone)]
pub struct ConflictResolutionRecord {
    /// Unique resolution ID
    pub resolution_id: String,
    /// Conflict that was resolved
    pub conflict: DirectoryConflict,
    /// Resolution strategy used
    pub strategy: ConflictResolutionStrategy,
    /// Resolution start time
    pub started_at: DateTime<Utc>,
    /// Resolution completion time
    pub completed_at: Option<DateTime<Utc>>,
    /// Resolution outcome
    pub outcome: ResolutionOutcome,
    /// Additional details
    pub details: HashMap<String, String>,
}

/// Outcome of a conflict resolution
#[derive(Debug, Clone, PartialEq)]
pub enum ResolutionOutcome {
    /// Successfully resolved
    Success,
    /// Failed to resolve
    Failed { reason: String },
    /// Resolution timed out
    TimedOut,
    /// Resolution was cancelled
    Cancelled,
    /// Partial resolution achieved
    Partial { description: String },
}

/// Statistics for conflict resolution
#[derive(Debug, Clone, Default)]
pub struct ConflictResolutionStatistics {
    /// Total conflicts detected
    pub total_conflicts_detected: u64,
    /// Total conflicts resolved successfully
    pub successful_resolutions: u64,
    /// Total failed resolutions
    pub failed_resolutions: u64,
    /// Total timed out resolutions
    pub timed_out_resolutions: u64,
    /// Average resolution time
    pub average_resolution_time: Duration,
    /// Most common conflict type
    pub most_common_conflict_type: Option<String>,
    /// Resolution efficiency (success rate)
    pub resolution_efficiency: f32,
}

impl DirectoryConflictResolver {
    /// Create a new conflict resolver
    pub fn new() -> Self {
        Self::with_settings(ConflictDetectionSettings::default())
    }

    /// Create a new conflict resolver with custom settings
    pub fn with_settings(settings: ConflictDetectionSettings) -> Self {
        Self {
            active_conflicts: Arc::new(Mutex::new(HashMap::new())),
            resolution_history: Arc::new(Mutex::new(Vec::new())),
            detection_settings: settings,
            resolution_stats: Arc::new(Mutex::new(ConflictResolutionStatistics::default())),
        }
    }

    /// Check for conflicts before allocating directories
    ///
    /// # Arguments
    ///
    /// * `test_id` - Test ID requesting allocation
    /// * `count` - Number of directories requested
    ///
    /// # Returns
    ///
    /// Returns Some(conflict) if a conflict is detected, None otherwise
    #[instrument(skip(self))]
    pub async fn check_conflicts(&self, test_id: &str, count: usize) -> Option<DirectoryConflict> {
        if !self.detection_settings.enable_detection {
            return None;
        }

        debug!(
            test_id = %test_id,
            count = %count,
            "Checking for conflicts"
        );

        // Check for existing active conflicts
        {
            let active_conflicts = self.active_conflicts.lock();
            for (_, conflict) in active_conflicts.iter() {
                if let ConflictType::MultipleTestAccess { test_ids } = &conflict.conflict_type {
                    if test_ids.contains(&test_id.to_string()) {
                        debug!("Found existing conflict for test: {}", test_id);
                        return Some(conflict.clone());
                    }
                }
            }
        }

        // Check system resource constraints
        if let Some(conflict) = self.check_resource_conflicts(test_id, count).await {
            self.register_conflict(conflict.clone()).await;
            return Some(conflict);
        }

        // Check concurrency limits
        if let Some(conflict) = self.check_concurrency_conflicts(test_id, count).await {
            self.register_conflict(conflict.clone()).await;
            return Some(conflict);
        }

        // Check disk space conflicts
        if let Some(conflict) = self.check_disk_space_conflicts(test_id, count).await {
            self.register_conflict(conflict.clone()).await;
            return Some(conflict);
        }

        None
    }

    /// Resolve a detected conflict
    ///
    /// # Arguments
    ///
    /// * `conflict` - The conflict to resolve
    ///
    /// # Errors
    ///
    /// This function will return an error if conflict resolution fails
    #[instrument(skip(self))]
    pub async fn resolve_conflict(&self, conflict: &DirectoryConflict) -> TempDirResult<()> {
        let resolution_id = uuid::Uuid::new_v4().to_string();
        let started_at = Utc::now();

        info!(
            conflict_id = %conflict.conflict_id,
            resolution_id = %resolution_id,
            strategy = ?conflict.resolution_strategy,
            "Starting conflict resolution"
        );

        let outcome = match &conflict.resolution_strategy {
            ConflictResolutionStrategy::Wait { timeout } => {
                self.resolve_with_wait(conflict, *timeout).await
            },
            ConflictResolutionStrategy::CreateAlternative => {
                self.resolve_with_alternative(conflict).await
            },
            ConflictResolutionStrategy::ForceTerminate => {
                self.resolve_with_force_termination(conflict).await
            },
            ConflictResolutionStrategy::Fail => ResolutionOutcome::Failed {
                reason: "Strategy set to fail".to_string(),
            },
        };

        let completed_at = Utc::now();
        let resolution_record = ConflictResolutionRecord {
            resolution_id: resolution_id.clone(),
            conflict: conflict.clone(),
            strategy: conflict.resolution_strategy.clone(),
            started_at,
            completed_at: Some(completed_at),
            outcome: outcome.clone(),
            details: HashMap::new(),
        };

        // Update statistics and history
        self.update_resolution_statistics(&resolution_record).await;
        self.resolution_history.lock().push(resolution_record);

        // Mark conflict as resolved if successful
        if matches!(
            outcome,
            ResolutionOutcome::Success | ResolutionOutcome::Partial { .. }
        ) {
            self.mark_conflict_resolved(conflict).await;
        }

        match outcome {
            ResolutionOutcome::Success => {
                info!(
                    conflict_id = %conflict.conflict_id,
                    resolution_id = %resolution_id,
                    "Successfully resolved conflict"
                );
                Ok(())
            },
            ResolutionOutcome::Failed { reason } => {
                error!(
                    conflict_id = %conflict.conflict_id,
                    resolution_id = %resolution_id,
                    reason = %reason,
                    "Failed to resolve conflict"
                );
                Err(TempDirError::AccessConflict {
                    path: conflict.path.display().to_string(),
                    message: format!("Failed to resolve conflict: {}", reason),
                })
            },
            ResolutionOutcome::TimedOut => {
                warn!(
                    conflict_id = %conflict.conflict_id,
                    resolution_id = %resolution_id,
                    "Conflict resolution timed out"
                );
                Err(TempDirError::AccessConflict {
                    path: conflict.path.display().to_string(),
                    message: "Conflict resolution timed out".to_string(),
                })
            },
            ResolutionOutcome::Cancelled => {
                info!(
                    conflict_id = %conflict.conflict_id,
                    resolution_id = %resolution_id,
                    "Conflict resolution was cancelled"
                );
                Ok(())
            },
            ResolutionOutcome::Partial { description } => {
                warn!(
                    conflict_id = %conflict.conflict_id,
                    resolution_id = %resolution_id,
                    description = %description,
                    "Partial conflict resolution achieved"
                );
                Ok(())
            },
        }
    }

    /// Get current active conflicts
    pub async fn get_active_conflicts(&self) -> Vec<DirectoryConflict> {
        self.active_conflicts.lock().values().cloned().collect()
    }

    /// Get conflict resolution statistics
    pub async fn get_resolution_statistics(&self) -> ConflictResolutionStatistics {
        self.resolution_stats.lock().clone()
    }

    /// Get resolution history (last N records)
    pub async fn get_resolution_history(
        &self,
        limit: Option<usize>,
    ) -> Vec<ConflictResolutionRecord> {
        let history = self.resolution_history.lock();
        let limit = limit.unwrap_or(100);

        if history.len() <= limit {
            history.clone()
        } else {
            history[history.len() - limit..].to_vec()
        }
    }

    /// Update conflict detection settings
    pub async fn update_settings(&mut self, new_settings: ConflictDetectionSettings) {
        self.detection_settings = new_settings;
        info!("Conflict detection settings updated");
    }

    /// Clear resolution history
    pub async fn clear_history(&self) {
        self.resolution_history.lock().clear();
        info!("Conflict resolution history cleared");
    }

    /// Force resolve all active conflicts
    pub async fn force_resolve_all_conflicts(&self) -> TempDirResult<usize> {
        let active_conflicts: Vec<DirectoryConflict> = {
            let active = self.active_conflicts.lock();
            active.values().cloned().collect()
        };

        let mut resolved_count = 0;

        for conflict in active_conflicts {
            match self.resolve_conflict(&conflict).await {
                Ok(_) => resolved_count += 1,
                Err(e) => {
                    warn!(
                        conflict_id = %conflict.conflict_id,
                        error = %e,
                        "Failed to force resolve conflict"
                    );
                },
            }
        }

        if resolved_count > 0 {
            info!(resolved_count = %resolved_count, "Force resolved conflicts");
        }

        Ok(resolved_count)
    }

    // ============================================================================================
    // Private Implementation Methods
    // ============================================================================================

    /// Check for resource-based conflicts
    async fn check_resource_conflicts(
        &self,
        test_id: &str,
        count: usize,
    ) -> Option<DirectoryConflict> {
        match self.detection_settings.sensitivity_level {
            ConflictSensitivity::Low => {
                if count > self.detection_settings.max_directories_per_test * 2 {
                    Some(self.create_resource_conflict(test_id, count, "Excessive directory count"))
                } else {
                    None
                }
            },
            ConflictSensitivity::Normal => {
                if count > self.detection_settings.max_directories_per_test {
                    Some(self.create_resource_conflict(
                        test_id,
                        count,
                        "Directory count limit exceeded",
                    ))
                } else {
                    None
                }
            },
            ConflictSensitivity::High => {
                if count > self.detection_settings.max_directories_per_test / 2 {
                    Some(self.create_resource_conflict(
                        test_id,
                        count,
                        "High directory count detected",
                    ))
                } else {
                    None
                }
            },
            ConflictSensitivity::Custom {
                concurrency_threshold,
                ..
            } => {
                if count > concurrency_threshold {
                    Some(self.create_resource_conflict(test_id, count, "Custom threshold exceeded"))
                } else {
                    None
                }
            },
        }
    }

    /// Check for concurrency-based conflicts
    async fn check_concurrency_conflicts(
        &self,
        _test_id: &str,
        _count: usize,
    ) -> Option<DirectoryConflict> {
        // In a real implementation, this would check for process locks, file locks, etc.
        // For this example, we'll implement basic logic

        let active_count = self.active_conflicts.lock().len();
        if active_count >= 5 {
            Some(DirectoryConflict::new(
                PathBuf::from("/tmp/concurrency_conflict"),
                ConflictType::MultipleTestAccess {
                    test_ids: vec!["multiple_tests".to_string()],
                },
                ConflictResolutionStrategy::Wait {
                    timeout: Duration::from_secs(30),
                },
            ))
        } else {
            None
        }
    }

    /// Check for disk space conflicts
    async fn check_disk_space_conflicts(
        &self,
        test_id: &str,
        count: usize,
    ) -> Option<DirectoryConflict> {
        // Estimate required space
        let estimated_space = (count as u64) * 1024 * 1024 * 100; // 100MB per directory estimate

        let threshold = match self.detection_settings.sensitivity_level {
            ConflictSensitivity::Custom {
                disk_space_threshold,
                ..
            } => disk_space_threshold,
            _ => self.detection_settings.min_disk_space_threshold,
        };

        if estimated_space > threshold {
            Some(DirectoryConflict::new(
                PathBuf::from(format!("/tmp/test_{}", test_id)),
                ConflictType::DiskSpaceConflict {
                    required: estimated_space,
                    available: threshold,
                },
                ConflictResolutionStrategy::CreateAlternative,
            ))
        } else {
            None
        }
    }

    /// Create a resource conflict
    fn create_resource_conflict(
        &self,
        test_id: &str,
        _count: usize,
        _reason: &str,
    ) -> DirectoryConflict {
        DirectoryConflict::new(
            PathBuf::from(format!("/tmp/test_{}", test_id)),
            ConflictType::MultipleTestAccess {
                test_ids: vec![test_id.to_string()],
            },
            ConflictResolutionStrategy::Wait {
                timeout: Duration::from_secs(60),
            },
        )
    }

    /// Register a new conflict
    async fn register_conflict(&self, conflict: DirectoryConflict) {
        let mut active_conflicts = self.active_conflicts.lock();
        active_conflicts.insert(conflict.conflict_id.clone(), conflict.clone());

        let mut stats = self.resolution_stats.lock();
        stats.total_conflicts_detected += 1;

        debug!(
            conflict_id = %conflict.conflict_id,
            conflict_type = ?conflict.conflict_type,
            "Registered new conflict"
        );
    }

    /// Mark a conflict as resolved
    pub async fn mark_conflict_resolved(&self, conflict: &DirectoryConflict) {
        let mut active_conflicts = self.active_conflicts.lock();
        active_conflicts.remove(&conflict.conflict_id);

        debug!(
            conflict_id = %conflict.conflict_id,
            "Marked conflict as resolved"
        );
    }

    /// Resolve conflict by waiting
    async fn resolve_with_wait(
        &self,
        conflict: &DirectoryConflict,
        timeout: Duration,
    ) -> ResolutionOutcome {
        warn!(
            conflict_id = %conflict.conflict_id,
            timeout_secs = %timeout.as_secs(),
            "Waiting for conflict resolution"
        );

        tokio::time::sleep(timeout).await;

        // After waiting, assume the conflict is resolved
        ResolutionOutcome::Success
    }

    /// Resolve conflict by creating alternative resources
    async fn resolve_with_alternative(&self, conflict: &DirectoryConflict) -> ResolutionOutcome {
        info!(
            conflict_id = %conflict.conflict_id,
            path = %conflict.path.display(),
            "Creating alternative path to resolve conflict"
        );

        // In a real implementation, this would create alternative directories or resources
        ResolutionOutcome::Success
    }

    /// Resolve conflict by force termination
    async fn resolve_with_force_termination(
        &self,
        conflict: &DirectoryConflict,
    ) -> ResolutionOutcome {
        warn!(
            conflict_id = %conflict.conflict_id,
            "Attempting force termination to resolve conflict"
        );

        // In a real implementation, this might terminate processes or force unlock resources
        // For this example, we'll simulate success
        ResolutionOutcome::Success
    }

    /// Update resolution statistics
    async fn update_resolution_statistics(&self, record: &ConflictResolutionRecord) {
        let mut stats = self.resolution_stats.lock();

        match &record.outcome {
            ResolutionOutcome::Success => stats.successful_resolutions += 1,
            ResolutionOutcome::Failed { .. } => stats.failed_resolutions += 1,
            ResolutionOutcome::TimedOut => stats.timed_out_resolutions += 1,
            ResolutionOutcome::Cancelled => {}, // Don't count cancelled as success or failure
            ResolutionOutcome::Partial { .. } => stats.successful_resolutions += 1, // Count partial as success
        }

        // Update average resolution time
        if let Some(completed_at) = record.completed_at {
            let resolution_time = completed_at.signed_duration_since(record.started_at);
            let resolution_duration =
                Duration::from_secs(resolution_time.num_seconds().max(0) as u64);

            let total_resolutions = stats.successful_resolutions
                + stats.failed_resolutions
                + stats.timed_out_resolutions;
            if total_resolutions > 1 {
                let total_time =
                    stats.average_resolution_time.as_secs() as f64 * (total_resolutions - 1) as f64;
                let new_average =
                    (total_time + resolution_duration.as_secs() as f64) / total_resolutions as f64;
                stats.average_resolution_time = Duration::from_secs(new_average as u64);
            } else {
                stats.average_resolution_time = resolution_duration;
            }
        }

        // Update efficiency
        let total_attempts =
            stats.successful_resolutions + stats.failed_resolutions + stats.timed_out_resolutions;
        if total_attempts > 0 {
            stats.resolution_efficiency =
                stats.successful_resolutions as f32 / total_attempts as f32;
        }
    }
}

impl Clone for DirectoryConflictResolver {
    fn clone(&self) -> Self {
        Self {
            active_conflicts: self.active_conflicts.clone(),
            resolution_history: self.resolution_history.clone(),
            detection_settings: self.detection_settings.clone(),
            resolution_stats: self.resolution_stats.clone(),
        }
    }
}

impl Default for DirectoryConflictResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ConflictDetector for DirectoryConflictResolver {
    fn check_conflicts(&self, test_id: &str, count: usize) -> Option<DirectoryConflict> {
        futures::executor::block_on(self.check_conflicts(test_id, count))
    }

    fn resolve_conflict(&self, conflict: &DirectoryConflict) -> Result<(), TempDirError> {
        futures::executor::block_on(self.resolve_conflict(conflict))
    }
}

// ================================================================================================
// Helper Functions
// ================================================================================================

/// Create a basic conflict with wait strategy
pub fn create_wait_conflict(
    path: PathBuf,
    test_ids: Vec<String>,
    timeout: Duration,
) -> DirectoryConflict {
    DirectoryConflict::new(
        path,
        ConflictType::MultipleTestAccess { test_ids },
        ConflictResolutionStrategy::Wait { timeout },
    )
}

/// Create a disk space conflict
pub fn create_disk_space_conflict(
    path: PathBuf,
    required: u64,
    available: u64,
) -> DirectoryConflict {
    DirectoryConflict::new(
        path,
        ConflictType::DiskSpaceConflict {
            required,
            available,
        },
        ConflictResolutionStrategy::CreateAlternative,
    )
}

/// Create a permission conflict
pub fn create_permission_conflict(
    path: PathBuf,
    required_perms: String,
    actual_perms: String,
) -> DirectoryConflict {
    DirectoryConflict::new(
        path,
        ConflictType::PermissionConflict {
            required_perms,
            actual_perms,
        },
        ConflictResolutionStrategy::ForceTerminate,
    )
}
