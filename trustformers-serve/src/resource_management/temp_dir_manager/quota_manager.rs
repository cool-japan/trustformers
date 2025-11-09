//! Directory Quota Manager Implementation
//!
//! This module manages disk space quotas and monitoring for temporary directories,
//! tracking disk space usage and enforcing quotas to prevent temporary directories
//! from consuming excessive disk space.

use super::types::*;
use super::utils::*;

use parking_lot::Mutex;
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};
use tracing::{debug, error, info, warn};

use crate::resource_management::types::*;

// ================================================================================================
// Directory Quota Manager
// ================================================================================================

/// Manages disk space quotas and monitoring for temporary directories
///
/// This component tracks disk space usage and enforces quotas to prevent
/// temporary directories from consuming excessive disk space.
#[derive(Debug)]
pub struct DirectoryQuotaManager {
    /// Total quota limit in bytes
    total_quota_bytes: u64,

    /// Currently used space in bytes
    used_space: Arc<AtomicU64>,

    /// Per-directory quotas
    directory_quotas: Arc<Mutex<HashMap<PathBuf, DirectoryQuota>>>,

    /// Base directory for quota monitoring
    base_directory: PathBuf,

    /// Reserved space (allocated but not yet used)
    reserved_space: Arc<AtomicU64>,

    /// Quota monitoring enabled
    monitoring_enabled: bool,

    /// Quota enforcement level
    enforcement_level: QuotaEnforcementLevel,
}

/// Levels of quota enforcement
#[derive(Debug, Clone, PartialEq)]
pub enum QuotaEnforcementLevel {
    /// No enforcement, monitoring only
    None,
    /// Warn when approaching limits
    Warning,
    /// Prevent allocations that would exceed limits
    Strict,
    /// Aggressive enforcement with cleanup
    Aggressive,
}

impl DirectoryQuotaManager {
    /// Create a new quota manager
    pub fn new(config: &TempDirPoolConfig) -> Self {
        let total_quota = config.max_directories as u64 * config.max_directory_size_bytes;

        Self {
            total_quota_bytes: total_quota,
            used_space: Arc::new(AtomicU64::new(0)),
            directory_quotas: Arc::new(Mutex::new(HashMap::new())),
            base_directory: config.base_path.clone(),
            reserved_space: Arc::new(AtomicU64::new(0)),
            monitoring_enabled: true,
            enforcement_level: QuotaEnforcementLevel::Strict,
        }
    }

    /// Create a new quota manager with custom settings
    pub fn with_settings(
        config: &TempDirPoolConfig,
        enforcement_level: QuotaEnforcementLevel,
        monitoring_enabled: bool,
    ) -> Self {
        let total_quota = config.max_directories as u64 * config.max_directory_size_bytes;

        Self {
            total_quota_bytes: total_quota,
            used_space: Arc::new(AtomicU64::new(0)),
            directory_quotas: Arc::new(Mutex::new(HashMap::new())),
            base_directory: config.base_path.clone(),
            reserved_space: Arc::new(AtomicU64::new(0)),
            monitoring_enabled,
            enforcement_level,
        }
    }

    /// Reserve space for a directory
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the directory
    /// * `size_bytes` - Amount of space to reserve
    ///
    /// # Errors
    ///
    /// This function will return an error if the quota would be exceeded
    pub async fn reserve_space(&self, path: &Path, size_bytes: u64) -> TempDirResult<()> {
        if !self.monitoring_enabled {
            debug!("Quota monitoring disabled, allowing reservation");
            return Ok(());
        }

        let current_used = self.used_space.load(Ordering::SeqCst);
        let current_reserved = self.reserved_space.load(Ordering::SeqCst);
        let total_committed = current_used + current_reserved;

        if total_committed + size_bytes > self.total_quota_bytes {
            match self.enforcement_level {
                QuotaEnforcementLevel::None => {
                    info!(
                        path = %path.display(),
                        requested = %size_bytes,
                        total_available = %(self.total_quota_bytes - total_committed),
                        "Quota would be exceeded but enforcement is disabled"
                    );
                },
                QuotaEnforcementLevel::Warning => {
                    warn!(
                        path = %path.display(),
                        requested = %size_bytes,
                        total_available = %(self.total_quota_bytes - total_committed),
                        "Quota exceeded - warning only"
                    );
                },
                QuotaEnforcementLevel::Strict | QuotaEnforcementLevel::Aggressive => {
                    return Err(TempDirError::QuotaExceeded {
                        requested: size_bytes,
                        available: self.total_quota_bytes - total_committed,
                    });
                },
            }
        }

        // Create directory quota
        let quota = DirectoryQuota {
            max_total_size: size_bytes,
            max_file_count: 10000, // Default file count limit
            max_depth: 10,
            current_size: Arc::new(AtomicU64::new(0)),
            current_file_count: Arc::new(AtomicU64::new(0)),
        };

        {
            let mut directory_quotas = self.directory_quotas.lock();
            directory_quotas.insert(path.to_path_buf(), quota);
        }

        // Update reserved space
        self.reserved_space.fetch_add(size_bytes, Ordering::SeqCst);

        debug!(
            path = %path.display(),
            reserved_bytes = %size_bytes,
            total_reserved = %self.reserved_space.load(Ordering::SeqCst),
            "Reserved space for directory"
        );

        Ok(())
    }

    /// Release space for a directory
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the directory to release space for
    ///
    /// # Errors
    ///
    /// This function will return an error if the directory is not tracked
    pub async fn release_space(&self, path: &Path) -> TempDirResult<()> {
        let mut directory_quotas = self.directory_quotas.lock();

        if let Some(quota) = directory_quotas.remove(path) {
            let reserved_bytes = quota.max_total_size;
            let actual_used = quota.current_size.load(Ordering::SeqCst);

            // Update counters
            self.reserved_space.fetch_sub(reserved_bytes, Ordering::SeqCst);
            if actual_used > 0 {
                self.used_space.fetch_sub(actual_used, Ordering::SeqCst);
            }

            debug!(
                path = %path.display(),
                released_reserved = %reserved_bytes,
                released_used = %actual_used,
                "Released space for directory"
            );

            Ok(())
        } else {
            warn!(path = %path.display(), "Attempted to release space for untracked directory");
            Ok(()) // Don't fail, just warn
        }
    }

    /// Get available space
    ///
    /// # Returns
    ///
    /// Returns the amount of available space in bytes
    ///
    /// # Errors
    ///
    /// This function will return an error if disk space cannot be determined
    pub async fn get_available_space(&self) -> TempDirResult<u64> {
        if !self.monitoring_enabled {
            return Ok(u64::MAX);
        }

        // Get actual disk space available
        let disk_available = self.get_disk_space_available().await?;

        // Get quota available
        let current_used = self.used_space.load(Ordering::SeqCst);
        let current_reserved = self.reserved_space.load(Ordering::SeqCst);
        let total_committed = current_used + current_reserved;
        let quota_available = self.total_quota_bytes.saturating_sub(total_committed);

        // Return the minimum of disk and quota availability
        Ok(disk_available.min(quota_available))
    }

    /// Get total quota in bytes
    pub async fn get_total_quota(&self) -> u64 {
        self.total_quota_bytes
    }

    /// Get currently used space in bytes
    pub async fn get_used_space(&self) -> u64 {
        self.used_space.load(Ordering::SeqCst)
    }

    /// Get currently reserved space in bytes
    pub async fn get_reserved_space(&self) -> u64 {
        self.reserved_space.load(Ordering::SeqCst)
    }

    /// Update directory usage
    pub async fn update_directory_usage(
        &self,
        path: &Path,
        current_size: u64,
        file_count: usize,
    ) -> TempDirResult<()> {
        let directory_quotas = self.directory_quotas.lock();

        if let Some(quota) = directory_quotas.get(path) {
            let previous_size = quota.current_size.swap(current_size, Ordering::SeqCst);
            quota.current_file_count.store(file_count as u64, Ordering::SeqCst);

            // Update total used space
            if current_size > previous_size {
                self.used_space.fetch_add(current_size - previous_size, Ordering::SeqCst);
            } else if previous_size > current_size {
                self.used_space.fetch_sub(previous_size - current_size, Ordering::SeqCst);
            }

            // Check quota enforcement
            if self.monitoring_enabled && self.enforcement_level != QuotaEnforcementLevel::None {
                if current_size > quota.max_total_size {
                    let message = format!(
                        "Directory size {} exceeds quota {}",
                        current_size, quota.max_total_size
                    );

                    match self.enforcement_level {
                        QuotaEnforcementLevel::Warning => {
                            warn!(path = %path.display(), message = %message);
                        },
                        QuotaEnforcementLevel::Strict | QuotaEnforcementLevel::Aggressive => {
                            return Err(TempDirError::QuotaExceeded {
                                requested: current_size,
                                available: quota.max_total_size,
                            });
                        },
                        QuotaEnforcementLevel::None => {},
                    }
                }

                if file_count > quota.max_file_count {
                    let message = format!(
                        "File count {} exceeds limit {}",
                        file_count, quota.max_file_count
                    );

                    match self.enforcement_level {
                        QuotaEnforcementLevel::Warning => {
                            warn!(path = %path.display(), message = %message);
                        },
                        QuotaEnforcementLevel::Strict | QuotaEnforcementLevel::Aggressive => {
                            return Err(TempDirError::AllocationFailed { message });
                        },
                        QuotaEnforcementLevel::None => {},
                    }
                }
            }

            debug!(
                path = %path.display(),
                current_size = %current_size,
                file_count = %file_count,
                "Updated directory usage"
            );
        }

        Ok(())
    }

    /// Get quota utilization percentage
    pub async fn get_quota_utilization(&self) -> f32 {
        if self.total_quota_bytes == 0 {
            return 0.0;
        }

        let current_used = self.used_space.load(Ordering::SeqCst);
        let current_reserved = self.reserved_space.load(Ordering::SeqCst);
        let total_committed = current_used + current_reserved;

        total_committed as f32 / self.total_quota_bytes as f32
    }

    /// Get directory-specific usage information
    pub async fn get_directory_usage(&self, path: &Path) -> Option<DirectoryUsageInfo> {
        let directory_quotas = self.directory_quotas.lock();

        directory_quotas.get(path).map(|quota| DirectoryUsageInfo {
            path: path.to_path_buf(),
            current_size: quota.current_size.load(Ordering::SeqCst),
            max_size: quota.max_total_size,
            file_count: quota.current_file_count.load(Ordering::SeqCst) as usize,
            max_file_count: quota.max_file_count,
            utilization: if quota.max_total_size == 0 {
                0.0
            } else {
                quota.current_size.load(Ordering::SeqCst) as f32 / quota.max_total_size as f32
            },
        })
    }

    /// Get all tracked directories and their usage
    pub async fn get_all_directory_usage(&self) -> Vec<DirectoryUsageInfo> {
        let directory_quotas = self.directory_quotas.lock();

        directory_quotas
            .iter()
            .map(|(path, quota)| DirectoryUsageInfo {
                path: path.clone(),
                current_size: quota.current_size.load(Ordering::SeqCst),
                max_size: quota.max_total_size,
                file_count: quota.current_file_count.load(Ordering::SeqCst) as usize,
                max_file_count: quota.max_file_count,
                utilization: if quota.max_total_size == 0 {
                    0.0
                } else {
                    quota.current_size.load(Ordering::SeqCst) as f32 / quota.max_total_size as f32
                },
            })
            .collect()
    }

    /// Recalculate usage for a specific directory
    pub async fn recalculate_directory_usage(
        &self,
        path: &Path,
    ) -> TempDirResult<DirectorySpaceUsage> {
        if !path.exists() {
            return Err(TempDirError::DirectoryNotFound {
                path: path.display().to_string(),
            });
        }

        let usage = calculate_directory_usage(path)?;

        // Update tracked usage
        self.update_directory_usage(path, usage.total_size, usage.file_count).await?;

        Ok(usage)
    }

    /// Perform quota cleanup (for aggressive enforcement)
    pub async fn perform_quota_cleanup(&self) -> TempDirResult<Vec<PathBuf>> {
        if self.enforcement_level != QuotaEnforcementLevel::Aggressive {
            return Ok(vec![]);
        }

        let utilization = self.get_quota_utilization().await;
        if utilization < 0.9 {
            // Only cleanup when utilization is above 90%
            return Ok(vec![]);
        }

        warn!(utilization = %utilization, "High quota utilization detected, performing cleanup");

        let directory_quotas = self.directory_quotas.lock();
        let mut cleanup_candidates = Vec::new();

        // Find directories that exceed their quotas
        for (path, quota) in directory_quotas.iter() {
            let current_size = quota.current_size.load(Ordering::SeqCst);
            if current_size > quota.max_total_size {
                cleanup_candidates.push(path.clone());
            }
        }

        drop(directory_quotas); // Release lock

        // Perform cleanup on candidates
        let mut cleaned_paths = Vec::new();
        for path in cleanup_candidates {
            match self.cleanup_directory(&path).await {
                Ok(_) => {
                    cleaned_paths.push(path);
                },
                Err(e) => {
                    error!(path = %path.display(), error = %e, "Failed to cleanup directory");
                },
            }
        }

        if !cleaned_paths.is_empty() {
            info!(
                cleaned_count = cleaned_paths.len(),
                "Performed aggressive quota cleanup"
            );
        }

        Ok(cleaned_paths)
    }

    /// Set quota enforcement level
    pub async fn set_enforcement_level(&mut self, level: QuotaEnforcementLevel) {
        let level_clone = level.clone();
        self.enforcement_level = level;
        info!(level = ?level_clone, "Quota enforcement level updated");
    }

    /// Enable or disable quota monitoring
    pub async fn set_monitoring_enabled(&mut self, enabled: bool) {
        self.monitoring_enabled = enabled;
        info!(enabled = %enabled, "Quota monitoring setting updated");
    }

    /// Update total quota limit
    pub async fn update_total_quota(&mut self, new_quota: u64) -> TempDirResult<()> {
        let current_used = self.used_space.load(Ordering::SeqCst);
        let current_reserved = self.reserved_space.load(Ordering::SeqCst);
        let total_committed = current_used + current_reserved;

        if new_quota < total_committed {
            return Err(TempDirError::ConfigurationError {
                message: format!(
                    "New quota {} is less than currently committed space {}",
                    new_quota, total_committed
                ),
            });
        }

        self.total_quota_bytes = new_quota;
        info!(new_quota = %new_quota, "Total quota limit updated");

        Ok(())
    }

    /// Generate quota report
    pub async fn generate_quota_report(&self) -> String {
        let used_space = self.get_used_space().await;
        let reserved_space = self.get_reserved_space().await;
        let available_space = self.get_available_space().await.unwrap_or(0);
        let utilization = self.get_quota_utilization().await;
        let all_usage = self.get_all_directory_usage().await;

        format!(
            "=== Quota Manager Report ===\n\
             Total quota: {} bytes\n\
             Used space: {} bytes\n\
             Reserved space: {} bytes\n\
             Available space: {} bytes\n\
             Utilization: {:.1}%\n\
             Monitoring enabled: {}\n\
             Enforcement level: {:?}\n\
             Tracked directories: {}\n\
             \n\
             == Directory Usage ==\n\
             {}",
            self.total_quota_bytes,
            used_space,
            reserved_space,
            available_space,
            utilization * 100.0,
            self.monitoring_enabled,
            self.enforcement_level,
            all_usage.len(),
            self.format_directory_usage(&all_usage)
        )
    }

    // ============================================================================================
    // Private Implementation Methods
    // ============================================================================================

    /// Get actual disk space available on the filesystem
    async fn get_disk_space_available(&self) -> TempDirResult<u64> {
        get_available_disk_space(&self.base_directory)
    }

    /// Cleanup a directory (remove old files, compress, etc.)
    async fn cleanup_directory(&self, path: &Path) -> TempDirResult<u64> {
        if !path.exists() {
            return Ok(0);
        }

        let _initial_usage = calculate_directory_usage(path)?;

        // Simple cleanup: remove files older than 1 hour
        let cutoff_time = std::time::SystemTime::now()
            .checked_sub(std::time::Duration::from_secs(3600))
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

        let mut bytes_cleaned = 0u64;

        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries {
                if let Ok(entry) = entry {
                    if let Ok(metadata) = entry.metadata() {
                        if let Ok(modified) = metadata.modified() {
                            if modified < cutoff_time && metadata.is_file() {
                                if let Ok(()) = fs::remove_file(entry.path()) {
                                    bytes_cleaned += metadata.len();
                                }
                            }
                        }
                    }
                }
            }
        }

        // Update usage after cleanup
        let final_usage = calculate_directory_usage(path)?;
        self.update_directory_usage(path, final_usage.total_size, final_usage.file_count)
            .await?;

        Ok(bytes_cleaned)
    }

    /// Format directory usage information for reporting
    fn format_directory_usage(&self, usage_list: &[DirectoryUsageInfo]) -> String {
        if usage_list.is_empty() {
            return "No directories tracked".to_string();
        }

        usage_list
            .iter()
            .map(|usage| {
                format!(
                    "{}: {} / {} bytes ({:.1}%), {} files",
                    usage.path.display(),
                    usage.current_size,
                    usage.max_size,
                    usage.utilization * 100.0,
                    usage.file_count
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ================================================================================================
// Additional Types
// ================================================================================================

/// Information about directory usage
#[derive(Debug, Clone)]
pub struct DirectoryUsageInfo {
    /// Directory path
    pub path: PathBuf,
    /// Current size in bytes
    pub current_size: u64,
    /// Maximum allowed size in bytes
    pub max_size: u64,
    /// Current file count
    pub file_count: usize,
    /// Maximum allowed file count
    pub max_file_count: usize,
    /// Utilization percentage (0.0 to 1.0)
    pub utilization: f32,
}

impl DiskSpaceProvider for DirectoryQuotaManager {
    fn get_available_space(&self) -> Result<u64, TempDirError> {
        futures::executor::block_on(self.get_available_space())
    }

    fn get_total_space(&self) -> Result<u64, TempDirError> {
        Ok(self.total_quota_bytes)
    }

    fn get_used_space(&self) -> Result<u64, TempDirError> {
        Ok(self.used_space.load(Ordering::SeqCst))
    }
}
