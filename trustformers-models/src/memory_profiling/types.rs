//! Memory profiling types and data structures
//!
//! This module contains all the core data structures used for memory profiling
//! including metrics, alerts, configurations, and allocation information.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;
use uuid::Uuid;

/// Configuration for the memory profiler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    /// Maximum number of data points to keep in memory
    pub max_data_points: usize,
    /// Data collection interval in milliseconds
    pub collection_interval_ms: u64,
    /// Enable memory leak detection
    pub enable_leak_detection: bool,
    /// Enable allocation pattern analysis
    pub enable_pattern_analysis: bool,
    /// Memory threshold for alerts (in MB)
    pub memory_alert_threshold_mb: f64,
    /// Enable automatic GC suggestions
    pub enable_gc_suggestions: bool,
    /// Output directory for reports
    pub output_dir: String,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            max_data_points: 10000,
            collection_interval_ms: 1000,
            enable_leak_detection: true,
            enable_pattern_analysis: true,
            memory_alert_threshold_mb: 1024.0,
            enable_gc_suggestions: true,
            output_dir: "./memory_reports".to_string(),
        }
    }
}

/// Real-time memory metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    pub timestamp: SystemTime,
    pub total_memory_mb: f64,
    pub heap_memory_mb: f64,
    pub stack_memory_mb: f64,
    pub gpu_memory_mb: Option<f64>,
    pub peak_memory_mb: f64,
    pub allocated_objects: u64,
    pub deallocated_objects: u64,
    pub active_allocations: u64,
    pub memory_fragmentation_ratio: f64,
    pub gc_collections: u64,
    pub gc_time_ms: f64,
    pub memory_growth_rate_mb_per_sec: f64,
}

/// Memory allocation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationInfo {
    pub id: Uuid,
    pub timestamp: SystemTime,
    pub size_bytes: usize,
    pub location: String,
    pub stack_trace: Vec<String>,
    pub object_type: String,
    pub is_leaked: bool,
}

/// Memory profiling alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAlert {
    pub id: Uuid,
    pub timestamp: SystemTime,
    pub alert_type: MemoryAlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub details: HashMap<String, String>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryAlertType {
    HighMemoryUsage,
    MemoryLeak,
    RapidGrowth,
    FragmentationHigh,
    GcPressure,
    OutOfMemory,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Memory pattern analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPattern {
    pub pattern_type: PatternType,
    pub frequency: u64,
    pub average_size: f64,
    pub total_size: f64,
    pub locations: Vec<String>,
    pub trend: PatternTrend,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    LargeAllocations,
    FrequentSmallAllocations,
    GrowingAllocations,
    TemporaryAllocations,
    LeakedAllocations,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternTrend {
    Increasing,
    Decreasing,
    Stable,
    Cyclical,
}

/// Memory dashboard report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDashboardReport {
    pub timestamp: SystemTime,
    pub summary: MemoryUsageSummary,
    pub metrics_over_time: Vec<MemoryMetrics>,
    pub alerts: Vec<MemoryAlert>,
    pub patterns: Vec<MemoryPattern>,
    pub recommendations: Vec<String>,
    pub system_info: SystemInfo,
}

/// Summary of memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsageSummary {
    pub total_runtime_seconds: f64,
    pub peak_memory_mb: f64,
    pub average_memory_mb: f64,
    pub memory_efficiency_score: f64,
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub leaked_allocations: u64,
    pub fragmentation_events: u64,
    pub gc_pressure_events: u64,
    pub alert_count_by_severity: HashMap<AlertSeverity, u64>,
}

/// System information for the report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub total_system_memory_gb: f64,
    pub available_system_memory_gb: f64,
    pub cpu_count: usize,
    pub os_info: String,
    pub rust_version: String,
}

/// System memory information
#[derive(Debug, Clone)]
pub struct SystemMemoryInfo {
    pub total_memory: u64,
    pub available_memory: u64,
    pub used_memory: u64,
    pub free_memory: u64,
    pub cached_memory: u64,
    pub buffer_memory: u64,
}

/// Process memory information
#[derive(Debug, Clone)]
pub struct ProcessMemoryInfo {
    pub rss: u64,    // Resident Set Size
    pub vms: u64,    // Virtual Memory Size
    pub shared: u64, // Shared memory
    pub text: u64,   // Text (code) segment
    pub data: u64,   // Data segment
    pub heap: u64,   // Heap size
    pub stack: u64,  // Stack size
}
