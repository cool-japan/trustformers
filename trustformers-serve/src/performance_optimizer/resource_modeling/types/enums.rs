//! Enumeration types for resource modeling
//!
//! Enums for thermal states, health status, severity levels,
//! detection algorithms, profiling methodologies, and optimization targets.

use serde::{Deserialize, Serialize};

/// Thermal management state enumeration
///
/// Current thermal state of the system for thermal management
/// and throttling decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThermalState {
    /// Normal operation temperature
    Normal,

    /// Warning level temperature
    Warning,

    /// Critical level temperature
    Critical,

    /// Emergency shutdown required
    Emergency,
}

impl Default for ThermalState {
    fn default() -> Self {
        Self::Normal
    }
}

/// Health status enumeration
///
/// Health status classification for system components
/// and performance indicators.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Healthy status
    Healthy,

    /// Warning status
    Warning,

    /// Critical status
    Critical,

    /// Failed status
    Failed,

    /// Unknown status
    Unknown,
}

/// Severity level enumeration
///
/// Severity classification for alerts, issues, and
/// system events requiring attention.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    /// Informational severity
    Info,

    /// Low severity
    Low,

    /// Medium severity
    Medium,

    /// High severity
    High,

    /// Critical severity
    Critical,
}

/// Detection algorithm types
///
/// Classification of different detection algorithms
/// for hardware and performance analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DetectionAlgorithmType {
    /// Statistical detection
    Statistical,

    /// Machine learning detection
    MachineLearning,

    /// Threshold-based detection
    ThresholdBased,

    /// Pattern recognition detection
    PatternRecognition,

    /// Custom detection algorithm
    Custom(String),
}

/// Profiling methodology types
///
/// Classification of different profiling methodologies
/// for performance characterization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProfilingMethodology {
    /// Microbenchmark profiling
    Microbenchmark,

    /// Application profiling
    Application,

    /// Synthetic workload profiling
    SyntheticWorkload,

    /// Real workload profiling
    RealWorkload,

    /// Hybrid profiling approach
    Hybrid,
}

/// Resource optimization target
///
/// Target optimization objectives for resource
/// optimization algorithms and strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationTarget {
    /// Maximize throughput
    Throughput,

    /// Minimize latency
    Latency,

    /// Optimize resource efficiency
    ResourceEfficiency,

    /// Balance performance and power
    PowerEfficiency,

    /// Custom optimization target
    Custom(String),
}
