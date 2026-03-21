//! Monitoring types for resource modeling
//!
//! Types for temperature monitoring, utilization tracking, benchmarking,
//! and performance analysis.

use crate::performance_optimizer::types::TemperatureMetrics;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

/// Temperature reading with timestamp and metadata
///
/// Individual temperature measurement including thermal metrics,
/// timestamp, and optional metadata for trend analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureReading {
    /// Temperature metrics
    pub metrics: TemperatureMetrics,

    /// Reading timestamp
    pub timestamp: DateTime<Utc>,
}

/// Utilization history tracking with bounded storage
///
/// Generic utilization history tracker with configurable capacity
/// and automatic cleanup for efficient memory usage.
#[derive(Debug)]
pub struct UtilizationHistory<T> {
    /// Maximum history size
    pub max_size: usize,

    /// Sample values with timestamps
    pub samples: Vec<(T, DateTime<Utc>)>,
}

/// Comprehensive utilization report
///
/// Detailed utilization report covering all system resources
/// with statistical analysis and trend information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtilizationReport {
    /// Monitoring duration
    pub duration: Duration,

    /// CPU utilization statistics
    pub cpu_utilization: UtilizationStats,

    /// Memory utilization statistics
    pub memory_utilization: UtilizationStats,

    /// I/O utilization statistics
    pub io_utilization: UtilizationStats,

    /// Network utilization statistics
    pub network_utilization: UtilizationStats,

    /// GPU utilization statistics
    pub gpu_utilization: Option<UtilizationStats>,

    /// Report timestamp
    pub timestamp: DateTime<Utc>,
}

/// Statistical analysis of utilization data
///
/// Comprehensive statistical analysis including central tendencies,
/// variability measures, and percentile calculations for utilization data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtilizationStats {
    /// Average utilization
    pub average: f32,

    /// Minimum utilization
    pub minimum: f32,

    /// Maximum utilization
    pub maximum: f32,

    /// Standard deviation
    pub std_deviation: f32,

    /// 95th percentile
    pub percentile_95: f32,

    /// 99th percentile
    pub percentile_99: f32,
}

/// Performance benchmarking framework
///
/// Comprehensive benchmarking system for evaluating hardware performance
/// across multiple dimensions with result validation and comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBenchmark {
    /// Benchmark identifier
    pub id: String,

    /// Benchmark name
    pub name: String,

    /// Benchmark description
    pub description: String,

    /// Benchmark parameters
    pub parameters: HashMap<String, String>,

    /// Benchmark duration
    pub duration: Duration,

    /// Benchmark results
    pub results: BenchmarkResults,

    /// Benchmark timestamp
    pub timestamp: DateTime<Utc>,
}

/// Benchmark execution results
///
/// Results from benchmark execution including performance metrics,
/// resource utilization, and validation status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// Performance score
    pub performance_score: f64,

    /// Resource efficiency score
    pub efficiency_score: f32,

    /// Benchmark passed validation
    pub validation_passed: bool,

    /// Detailed metrics
    pub detailed_metrics: HashMap<String, f64>,

    /// Error messages (if any)
    pub errors: Vec<String>,
}

/// System-wide benchmark suite
///
/// Comprehensive benchmark suite covering all system components
/// with coordinated execution and result aggregation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemBenchmark {
    /// CPU benchmarks
    pub cpu_benchmarks: Vec<PerformanceBenchmark>,

    /// Memory benchmarks
    pub memory_benchmarks: Vec<PerformanceBenchmark>,

    /// I/O benchmarks
    pub io_benchmarks: Vec<PerformanceBenchmark>,

    /// Network benchmarks
    pub network_benchmarks: Vec<PerformanceBenchmark>,

    /// GPU benchmarks
    pub gpu_benchmarks: Vec<PerformanceBenchmark>,

    /// Overall benchmark score
    pub overall_score: f64,

    /// Benchmark suite timestamp
    pub timestamp: DateTime<Utc>,
}
