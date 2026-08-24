//! Data model types for model versions, repositories, and access control in the Hub UI.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::hub::DownloadStats;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Access control for repositories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControl {
    /// Repository permissions
    pub permissions: HashMap<String, Permission>,
    /// API key for access
    pub api_key: Option<String>,
    /// Access whitelist
    pub whitelist: Vec<String>,
    /// Access blacklist
    pub blacklist: Vec<String>,
}
/// Benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Benchmark name
    pub name: String,
    /// Score
    pub score: f64,
    /// Unit of measurement
    pub unit: String,
    /// Benchmark timestamp
    pub timestamp: u64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}
/// Type of file change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    /// File was added
    Added,
    /// File was modified
    Modified,
    /// File was deleted
    Deleted,
    /// File was renamed
    Renamed { old_path: String },
}
/// Compatibility information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityInfo {
    /// Framework compatibility (e.g., "trustformers>=0.1.0")
    pub framework_version: Option<String>,
    /// Python version requirements
    pub python_version: Option<String>,
    /// CUDA version requirements
    pub cuda_version: Option<String>,
    /// Hardware requirements
    pub hardware_requirements: Vec<String>,
    /// Breaking changes from previous version
    pub breaking_changes: Vec<String>,
    /// Migration notes
    pub migration_notes: Option<String>,
}
/// File change information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileChange {
    /// File path
    pub path: String,
    /// Change type
    pub change_type: ChangeType,
    /// Old file size (for modifications)
    pub old_size: Option<u64>,
    /// New file size
    pub new_size: u64,
    /// File checksum
    pub checksum: Option<String>,
    /// Change description
    pub description: Option<String>,
}
/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Accuracy metrics
    pub accuracy: Option<f64>,
    /// Loss metrics
    pub loss: Option<f64>,
    /// Inference speed (tokens/second)
    pub inference_speed: Option<f64>,
    /// Memory usage (MB)
    pub memory_usage: Option<f64>,
    /// Model size (parameters)
    pub parameter_count: Option<u64>,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
    /// Benchmark results
    pub benchmarks: Vec<BenchmarkResult>,
}
/// Model version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    /// Version identifier (e.g., "v1.0.0", "main", commit hash)
    pub version: String,
    /// Human-readable version name
    pub name: Option<String>,
    /// Version description
    pub description: Option<String>,
    /// Creation timestamp
    pub created_at: u64,
    /// Last modified timestamp
    pub modified_at: u64,
    /// Version author
    pub author: Option<String>,
    /// Version tags (e.g., "stable", "experimental", "deprecated")
    pub tags: Vec<String>,
    /// Model performance metrics
    pub metrics: Option<ModelMetrics>,
    /// File changes from previous version
    pub changes: Vec<FileChange>,
    /// Parent version (for tracking lineage)
    pub parent_version: Option<String>,
    /// Download statistics
    pub download_stats: Option<DownloadStats>,
    /// Model size in bytes
    pub size_bytes: u64,
    /// Checksum for integrity verification
    pub checksum: Option<String>,
    /// Status of the version
    pub status: VersionStatus,
    /// Compatibility information
    pub compatibility: CompatibilityInfo,
}
/// Performance differences between versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceDiff {
    /// Accuracy difference
    pub accuracy_diff: Option<f64>,
    /// Loss difference
    pub loss_diff: Option<f64>,
    /// Speed difference
    pub speed_diff: Option<f64>,
    /// Memory usage difference
    pub memory_diff: Option<f64>,
}
/// Permission levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Permission {
    /// Read-only access
    Read,
    /// Read and write access
    Write,
    /// Full administrative access
    Admin,
}
/// Repository metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryMetadata {
    /// Repository name
    pub name: String,
    /// Repository description
    pub description: Option<String>,
    /// Repository owner
    pub owner: String,
    /// Repository visibility
    pub visibility: Visibility,
    /// Default branch/version
    pub default_version: String,
    /// Repository tags
    pub tags: Vec<String>,
    /// Creation timestamp
    pub created_at: u64,
    /// Last update timestamp
    pub updated_at: u64,
    /// Repository statistics
    pub stats: RepositoryStats,
}
/// Repository statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryStats {
    /// Total number of versions
    pub version_count: usize,
    /// Total downloads across all versions
    pub total_downloads: u64,
    /// Repository stars/likes
    pub stars: u64,
    /// Repository forks
    pub forks: u64,
    /// Active contributors
    pub contributors: u64,
    /// Last activity timestamp
    pub last_activity: u64,
}
/// Version comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionComparison {
    /// Source version
    pub from_version: String,
    /// Target version
    pub to_version: String,
    /// Size difference in bytes
    pub size_diff: i64,
    /// File changes
    pub changes: Vec<FileChange>,
    /// Performance differences
    pub performance_diff: PerformanceDiff,
    /// Comparison timestamp
    pub created_at: u64,
}
/// Version status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VersionStatus {
    /// Version is in development
    Development,
    /// Version is stable and ready for use
    Stable,
    /// Version is experimental
    Experimental,
    /// Version is deprecated
    Deprecated,
    /// Version has been archived
    Archived,
}
/// Repository visibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Visibility {
    /// Public repository
    Public,
    /// Private repository
    Private,
    /// Organization-only repository
    Organization,
}
