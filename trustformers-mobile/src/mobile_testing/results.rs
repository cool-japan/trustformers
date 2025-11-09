//! Mobile Testing Results
//!
//! This module contains all result structures for the mobile testing framework.

use super::config::{DeviceType, PowerMode, PrecisionMode, ThermalCondition};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Complete test suite results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSuiteResults {
    /// Test execution timestamp
    pub timestamp: SystemTime,
    /// Total test duration
    pub duration: Duration,
    /// Benchmark results
    pub benchmark_results: Vec<BenchmarkResult>,
    /// Battery test results
    pub battery_results: Vec<BatteryTestResult>,
    /// Stress test results
    pub stress_results: Vec<StressTestResult>,
    /// Memory test results
    pub memory_results: Vec<MemoryTestResult>,
    /// Overall success rate
    pub success_rate: f32,
}

/// Benchmark test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Test configuration identifier
    pub config_id: String,
    /// Input tensor shape
    pub input_shape: Vec<usize>,
    /// Precision mode used
    pub precision_mode: PrecisionMode,
    /// Power mode during test
    pub power_mode: PowerMode,
    /// Thermal condition during test
    pub thermal_condition: ThermalCondition,
    /// Average inference latency in milliseconds
    pub avg_latency_ms: f32,
    /// P95 inference latency in milliseconds
    pub p95_latency_ms: f32,
    /// P99 inference latency in milliseconds
    pub p99_latency_ms: f32,
    /// Throughput in inferences per second
    pub throughput_fps: f32,
    /// Memory usage during inference in MB
    pub memory_usage_mb: usize,
    /// Accuracy metrics
    pub accuracy_metrics: AccuracyMetrics,
    /// Power consumption statistics
    pub power_stats: PowerConsumptionStats,
}

/// Battery test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatteryTestResult {
    /// Test duration
    pub duration: Duration,
    /// Initial battery level (0.0-1.0)
    pub initial_battery_level: f32,
    /// Final battery level (0.0-1.0)
    pub final_battery_level: f32,
    /// Total energy consumed in milliwatt-hours
    pub energy_consumed_mwh: f32,
    /// Average power consumption in milliwatts
    pub avg_power_consumption_mw: f32,
    /// Peak power consumption in milliwatts
    pub peak_power_consumption_mw: f32,
    /// Total inferences performed
    pub total_inferences: usize,
    /// Energy per inference in millijoules
    pub energy_per_inference_mj: f32,
    /// Battery drain rate percentage per hour
    pub battery_drain_rate_percent_per_hour: f32,
}

/// Stress test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestResult {
    /// Test duration
    pub duration: Duration,
    /// Stress type applied
    pub stress_type: StressType,
    /// Success rate under stress
    pub success_rate: f32,
    /// Average latency under stress
    pub avg_latency_ms: f32,
    /// Memory pressure level (0.0-1.0)
    pub memory_pressure_level: f32,
    /// CPU utilization (0.0-1.0)
    pub cpu_utilization: f32,
    /// GPU utilization (0.0-1.0)
    pub gpu_utilization: f32,
    /// Thermal throttling events
    pub thermal_throttling_events: usize,
    /// Memory allocation failures
    pub memory_allocation_failures: usize,
    /// Error rate (0.0-1.0)
    pub error_rate: f32,
}

/// Memory test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTestResult {
    /// Test duration
    pub duration: Duration,
    /// Memory test type
    pub test_type: MemoryTestType,
    /// Peak memory usage in MB
    pub peak_memory_usage_mb: usize,
    /// Average memory usage in MB
    pub avg_memory_usage_mb: usize,
    /// Memory leaks detected
    pub memory_leaks_detected: usize,
    /// Memory usage statistics
    pub memory_stats: MemoryUsageStats,
    /// Garbage collection statistics (Android only)
    pub gc_stats: Option<HashMap<String, f32>>,
    /// Memory allocation success rate
    pub allocation_success_rate: f32,
}

/// Stress test types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StressType {
    CPU,
    Memory,
    Thermal,
    Battery,
    Network,
    Storage,
    Combined,
}

/// Memory test types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryTestType {
    LeakDetection,
    PressureTesting,
    AllocationStress,
    FragmentationAnalysis,
    GCPerformance,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsageStats {
    /// Total allocated memory in MB
    pub total_allocated_mb: usize,
    /// Peak allocated memory in MB
    pub peak_allocated_mb: usize,
    /// Memory fragmentation percentage
    pub fragmentation_percent: f32,
    /// Large allocation count
    pub large_allocations: usize,
    /// Small allocation count
    pub small_allocations: usize,
}

/// Power consumption statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerConsumptionStats {
    /// CPU power consumption in milliwatts
    pub cpu_power_mw: f32,
    /// GPU power consumption in milliwatts
    pub gpu_power_mw: f32,
    /// Memory power consumption in milliwatts
    pub memory_power_mw: f32,
    /// Total system power consumption in milliwatts
    pub total_power_mw: f32,
    /// Power efficiency score (0.0-1.0)
    pub efficiency_score: f32,
}

/// Accuracy metrics for model evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    /// Top-1 accuracy percentage
    pub top1_accuracy: f32,
    /// Top-5 accuracy percentage
    pub top5_accuracy: f32,
    /// F1 score
    pub f1_score: f32,
    /// Precision
    pub precision: f32,
    /// Recall
    pub recall: f32,
    /// Mean Average Precision (mAP)
    pub mean_average_precision: f32,
}

/// Device farm session result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceFarmSessionResult {
    /// Session identifier
    pub session_id: String,
    /// Session start time
    pub start_time: SystemTime,
    /// Session duration
    pub duration: Duration,
    /// Device test results
    pub device_results: Vec<DeviceTestResult>,
    /// Aggregated results across all devices
    pub aggregated_results: AggregatedTestResults,
}

/// Individual device test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceTestResult {
    /// Device identifier
    pub device_id: String,
    /// Device information
    pub device_info: DeviceInfo,
    /// Test suite results for this device
    pub test_results: TestSuiteResults,
    /// Device execution metrics
    pub execution_metrics: DeviceExecutionMetrics,
    /// Test artifacts (logs, videos, screenshots)
    pub artifacts: Vec<TestArtifact>,
}

/// Device information structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Device name
    pub device_name: String,
    /// Operating system
    pub os_name: String,
    /// OS version
    pub os_version: String,
    /// Device type
    pub device_type: DeviceType,
    /// Hardware model
    pub hardware_model: String,
    /// CPU architecture
    pub cpu_architecture: String,
    /// Total RAM in MB
    pub ram_mb: usize,
    /// Storage capacity in GB
    pub storage_gb: usize,
    /// Screen resolution
    pub screen_resolution: (u32, u32),
    /// Available sensors
    pub sensors: Vec<String>,
}

/// Device execution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceExecutionMetrics {
    /// Total execution time
    pub execution_time: Duration,
    /// Test setup time
    pub setup_time: Duration,
    /// Test cleanup time
    pub cleanup_time: Duration,
    /// Network transfer time
    pub network_time: Duration,
    /// Device availability time
    pub availability_time: Duration,
}

/// Test artifact (logs, videos, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestArtifact {
    /// Artifact identifier
    pub id: String,
    /// Artifact type
    pub artifact_type: ArtifactType,
    /// File path or URL
    pub location: String,
    /// File size in bytes
    pub size_bytes: usize,
    /// Creation timestamp
    pub created_at: SystemTime,
}

/// Test artifact types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactType {
    Video,
    Screenshot,
    DeviceLog,
    AppLog,
    PerformanceProfile,
    NetworkTrace,
    CrashReport,
    TestReport,
    RawData,
}

/// Aggregated test results across multiple devices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedTestResults {
    /// Number of devices tested
    pub device_count: usize,
    /// Overall success rate
    pub overall_success_rate: f32,
    /// Aggregated metrics
    pub metrics: AggregatedMetrics,
    /// Cross-device analysis
    pub cross_device_analysis: CrossDeviceAnalysis,
}

/// Aggregated performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    /// Average latency across all devices
    pub avg_latency_ms: f32,
    /// Standard deviation of latency
    pub latency_std_dev: f32,
    /// Average throughput
    pub avg_throughput_fps: f32,
    /// Average memory usage
    pub avg_memory_usage_mb: f32,
    /// Average power consumption
    pub avg_power_consumption_mw: f32,
    /// Statistical summary
    pub statistical_summary: StatisticalSummary,
}

/// Cross-device analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDeviceAnalysis {
    /// Performance variance across devices
    pub performance_variance: f32,
    /// Best performing device
    pub best_device: String,
    /// Worst performing device
    pub worst_device: String,
    /// Device compatibility rate
    pub compatibility_rate: f32,
}

/// Statistical summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSummary {
    pub mean: f32,
    pub median: f32,
    pub std_deviation: f32,
    pub min: f32,
    pub max: f32,
    pub percentiles: HashMap<String, f32>,
}

/// Device farm session metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceFarmSessionMetadata {
    /// Session configuration
    pub configuration: String,
    /// Requested device count
    pub requested_devices: usize,
    /// Successfully allocated devices
    pub allocated_devices: usize,
    /// Failed device allocations
    pub failed_allocations: usize,
    /// Total cost (if applicable)
    pub total_cost: Option<f32>,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
}

/// Resource utilization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// Device utilization percentage
    pub device_utilization: f32,
    /// Network bandwidth used in MB
    pub network_usage_mb: f32,
    /// Storage used in MB
    pub storage_usage_mb: f32,
    /// Compute time used in minutes
    pub compute_time_minutes: f32,
}

/// Execution summary for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSummary {
    /// Total tests executed
    pub total_tests: usize,
    /// Successful tests
    pub successful_tests: usize,
    /// Failed tests
    pub failed_tests: usize,
    /// Skipped tests
    pub skipped_tests: usize,
    /// Average execution time per test
    pub avg_execution_time: Duration,
    /// Performance highlights
    pub performance_highlights: Vec<String>,
    /// Issues encountered
    pub issues: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

impl BenchmarkResult {
    /// Calculate performance score based on multiple factors
    pub fn performance_score(&self) -> f32 {
        let latency_score = 1.0 - (self.avg_latency_ms / 1000.0).min(1.0);
        let throughput_score = (self.throughput_fps / 100.0).min(1.0);
        let memory_score = 1.0 - (self.memory_usage_mb as f32 / 1024.0).min(1.0);
        let accuracy_score = self.accuracy_metrics.top1_accuracy / 100.0;
        let power_score = self.power_stats.efficiency_score;

        (latency_score * 0.25
            + throughput_score * 0.25
            + memory_score * 0.2
            + accuracy_score * 0.2
            + power_score * 0.1)
            .max(0.0)
            .min(1.0)
    }

    /// Check if result meets target metrics
    pub fn meets_targets(
        &self,
        target_latency_ms: f32,
        target_throughput_fps: f32,
        target_accuracy: f32,
    ) -> bool {
        self.avg_latency_ms <= target_latency_ms
            && self.throughput_fps >= target_throughput_fps
            && self.accuracy_metrics.top1_accuracy >= target_accuracy
    }
}
