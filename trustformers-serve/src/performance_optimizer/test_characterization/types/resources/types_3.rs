//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::core::{IntensityCalculationMethod, TestCharacterizationResult, TestPhase};
use super::super::patterns::{PatternSignature, PatternVariation, SharingStrategy};
use super::functions::duration_zero;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSharingCapabilities {
    /// Read-only sharing support
    pub supports_read_sharing: bool,
    /// Write sharing support
    pub supports_write_sharing: bool,
    /// Maximum concurrent readers
    pub max_concurrent_readers: Option<usize>,
    /// Maximum concurrent writers
    pub max_concurrent_writers: Option<usize>,
    /// Sharing performance impact
    pub sharing_overhead: f64,
    /// Consistency guarantees
    pub consistency_guarantees: Vec<String>,
    /// Isolation requirements
    pub isolation_requirements: Vec<String>,
    /// Recommended sharing strategy
    pub recommended_strategy: SharingStrategy,
    /// Sharing safety assessment
    pub safety_assessment: f64,
    /// Performance trade-offs
    pub performance_tradeoffs: HashMap<String, f64>,
    /// Performance overhead percentage
    pub performance_overhead: f64,
    /// Implementation complexity score
    pub implementation_complexity: f64,
    /// Sharing mode
    pub sharing_mode: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ResourceUsageSnapshot {
    /// Snapshot timestamp
    #[serde(skip)]
    pub timestamp: Instant,
    /// CPU usage percentage (0.0 - 1.0)
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Available memory in bytes
    pub available_memory: usize,
    /// I/O read rate in bytes/second
    pub io_read_rate: f64,
    /// I/O write rate in bytes/second
    pub io_write_rate: f64,
    /// Network incoming rate in bytes/second (alias: network_rx_rate)
    pub network_in_rate: f64,
    /// Network outgoing rate in bytes/second (alias: network_tx_rate)
    pub network_out_rate: f64,
    /// Network receive rate in bytes/second
    pub network_rx_rate: f64,
    /// Network transmit rate in bytes/second
    pub network_tx_rate: f64,
    /// GPU utilization (0.0 - 1.0)
    pub gpu_utilization: f64,
    /// GPU usage percentage (0.0 - 1.0) (alias for gpu_utilization)
    pub gpu_usage: f64,
    /// GPU memory usage in bytes
    pub gpu_memory_usage: usize,
    /// Disk usage percentage (0.0 - 1.0)
    pub disk_usage: f64,
    /// System load average
    pub load_average: [f64; 3],
    /// Active process count
    pub process_count: usize,
    /// Thread count
    pub thread_count: usize,
    /// Memory pressure level (0.0 - 1.0)
    pub memory_pressure: f64,
    /// I/O wait percentage
    pub io_wait: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageAnalysisResults {
    pub total_storage_available: usize,
    pub total_storage_used: usize,
    pub storage_utilization: f64,
    pub read_throughput: f64,
    pub write_throughput: f64,
    pub storage_performance_score: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageDeviceAnalyzer {
    pub devices_analyzed: Vec<String>,
    pub analysis_enabled: bool,
    pub performance_benchmarks: HashMap<String, f64>,
    pub optimization_recommendations: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ResourceUsageDataPoint {
    /// Data point timestamp
    #[serde(skip)]
    pub timestamp: Instant,
    /// Resource type
    pub resource_type: String,
    /// Usage value
    pub value: f64,
    /// Usage rate (change per second)
    pub rate: f64,
    /// Percentile rank
    pub percentile: f64,
    /// Anomaly score
    pub anomaly_score: f64,
    /// Data quality indicator
    pub quality: f64,
    /// Associated test phase
    pub test_phase: Option<TestPhase>,
    /// Measurement confidence
    pub confidence: f64,
    /// Baseline deviation
    pub baseline_deviation: f64,
    /// Resource usage snapshot
    pub snapshot: ResourceUsageSnapshot,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilizationMetrics {
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub io_utilization: f64,
    pub network_utilization: f64,
    pub gpu_utilization: f64,
    pub overall_utilization: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuMonitor {
    /// CPU usage percentage
    pub usage_percent: f64,
    /// Number of cores
    pub core_count: usize,
    /// Sampling interval
    #[serde(default = "duration_zero")]
    pub sample_interval: std::time::Duration,
}
impl CpuMonitor {
    /// Create a new CpuMonitor with default settings
    pub fn new() -> Self {
        Self {
            usage_percent: 0.0,
            core_count: 1,
            sample_interval: Duration::from_millis(100),
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceIntensity {
    /// CPU intensity (0.0 - 1.0)
    pub cpu_intensity: f64,
    /// Memory intensity (0.0 - 1.0)
    pub memory_intensity: f64,
    /// I/O intensity (0.0 - 1.0)
    pub io_intensity: f64,
    /// Network intensity (0.0 - 1.0)
    pub network_intensity: f64,
    /// GPU intensity (0.0 - 1.0)
    pub gpu_intensity: f64,
    /// Overall intensity score
    pub overall_intensity: f64,
    /// Peak resource usage periods
    #[serde(skip)]
    pub peak_periods: Vec<(Instant, Duration)>,
    /// Resource usage variance
    pub usage_variance: f64,
    /// Baseline comparison
    pub baseline_comparison: f64,
    /// Intensity calculation method used
    pub calculation_method: IntensityCalculationMethod,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilizationProcessorConfig {
    pub processing_enabled: bool,
    pub sample_rate: f64,
    #[serde(default = "duration_zero")]
    pub aggregation_window: std::time::Duration,
    pub outlier_detection: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsagePattern {
    /// Pattern identifier
    pub pattern_id: String,
    /// Pattern signature
    pub signature: PatternSignature,
    /// Resource types involved
    pub resource_types: Vec<String>,
    /// Pattern duration
    #[serde(default = "duration_zero")]
    pub typical_duration: Duration,
    /// Intensity levels
    pub intensity_levels: HashMap<String, f64>,
    /// Frequency of occurrence
    pub frequency: f64,
    /// Confidence in pattern
    pub confidence: f64,
    /// Pattern variations
    pub variations: Vec<PatternVariation>,
    /// Performance characteristics
    pub performance_characteristics: HashMap<String, f64>,
    /// Optimization recommendations
    pub optimizations: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemResourceSnapshot {
    pub snapshot_timestamp: chrono::DateTime<chrono::Utc>,
    pub cpu_usage: f64,
    pub memory_usage: usize,
    pub io_activity: f64,
    pub network_activity: f64,
    pub disk_usage: usize,
    pub network_usage: usize,
    pub io_capacity: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TempDirPoolConfig {
    pub pool_size: usize,
    pub temp_dir_path: String,
    pub auto_cleanup: bool,
    #[serde(default = "duration_zero")]
    pub max_lifetime: std::time::Duration,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceOrderingStrategy {
    pub enabled: bool,
    pub priorities: std::collections::HashMap<String, i32>,
}
impl ResourceOrderingStrategy {
    /// Create a new ResourceOrderingStrategy with default settings
    pub fn new() -> TestCharacterizationResult<Self> {
        Ok(Self {
            enabled: true,
            priorities: HashMap::new(),
        })
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuBoundPhase {
    pub phase_name: String,
    pub cpu_usage_percentage: f64,
    #[serde(default = "duration_zero")]
    pub duration: std::time::Duration,
    pub thread_count: usize,
}
