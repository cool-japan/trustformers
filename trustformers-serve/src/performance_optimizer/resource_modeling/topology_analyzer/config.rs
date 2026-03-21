//! Configuration types for topology analysis

use super::types::*;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for topology analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyAnalysisConfig {
    /// Enable NUMA topology detection
    pub enable_numa_detection: bool,

    /// Enable cache hierarchy analysis
    pub enable_cache_analysis: bool,

    /// Enable memory topology analysis
    pub enable_memory_analysis: bool,

    /// Enable I/O topology analysis
    pub enable_io_analysis: bool,

    /// Enable interconnect analysis
    pub enable_interconnect_analysis: bool,

    /// Validation level for analysis results
    pub validation_level: ValidationLevel,

    /// Cache analysis results for performance
    pub enable_result_caching: bool,

    /// Analysis timeout duration
    pub analysis_timeout: Duration,

    /// Enable vendor-specific optimizations
    pub enable_vendor_optimizations: bool,

    /// Enable advanced bandwidth analysis
    pub enable_bandwidth_analysis: bool,

    /// Number of measurement iterations for precision
    pub measurement_iterations: u32,

    /// Enable parallel analysis where possible
    pub enable_parallel_analysis: bool,

    /// Enable detailed metrics collection
    pub enable_detailed_metrics: bool,

    /// Enable topology change monitoring
    pub enable_topology_monitoring: bool,

    /// Monitoring interval for topology changes
    pub monitoring_interval: Duration,
}

impl Default for TopologyAnalysisConfig {
    fn default() -> Self {
        Self {
            enable_numa_detection: true,
            enable_cache_analysis: true,
            enable_memory_analysis: true,
            enable_io_analysis: true,
            enable_interconnect_analysis: true,
            validation_level: ValidationLevel::Comprehensive,
            enable_result_caching: true,
            analysis_timeout: Duration::from_secs(300), // 5 minutes
            enable_vendor_optimizations: true,
            enable_bandwidth_analysis: true,
            measurement_iterations: 10,
            enable_parallel_analysis: true,
            enable_detailed_metrics: true,
            enable_topology_monitoring: false,
            monitoring_interval: Duration::from_secs(60), // 1 minute
        }
    }
}

/// Configuration for NUMA topology detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaDetectionConfig {
    /// Precision level for NUMA detection
    pub precision_level: PrecisionLevel,

    /// Enable cross-NUMA latency measurement
    pub enable_latency_measurement: bool,

    /// Enable NUMA bandwidth measurement
    pub enable_bandwidth_measurement: bool,

    /// Enable NUMA affinity analysis
    pub enable_affinity_analysis: bool,

    /// Measurement sample count for averaging
    pub measurement_samples: u32,

    /// Enable advanced NUMA features detection
    pub enable_advanced_features: bool,

    /// Enable NUMA memory interleaving detection
    pub enable_interleaving_detection: bool,

    /// Enable NUMA balancing analysis
    pub enable_balancing_analysis: bool,
}

impl Default for NumaDetectionConfig {
    fn default() -> Self {
        Self {
            precision_level: PrecisionLevel::Balanced,
            enable_latency_measurement: true,
            enable_bandwidth_measurement: true,
            enable_affinity_analysis: true,
            measurement_samples: 100,
            enable_advanced_features: true,
            enable_interleaving_detection: true,
            enable_balancing_analysis: true,
        }
    }
}
