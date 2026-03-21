//! Trait definitions and supporting types for resource modeling
//!
//! Extensible interfaces for vendor detectors, benchmarks, sensors,
//! analyzers, optimization strategies, and all supporting types including
//! hardware profiles, vendor features, and validation results.

use super::{config::*, detection::*, enums::*, error::*, monitoring::*, utility::*};
use crate::performance_optimizer::types::{
    CacheHierarchy, CpuPerformanceCharacteristics, GpuDeviceModel, MemoryType, NetworkInterface,
    StorageDevice,
};
use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

// Re-export types moved to traits_profiling and traits_analysis modules for backward compatibility
pub use super::traits_analysis::*;
pub use super::traits_profiling::*;

// =============================================================================
// TRAIT DEFINITIONS (8 types)
// =============================================================================

/// Vendor-specific hardware detector interface
///
/// Interface for implementing vendor-specific hardware detection
/// with GPU detection capabilities and vendor identification.
#[async_trait]
pub trait VendorDetector: std::fmt::Debug {
    /// Detect GPU devices for this vendor
    async fn detect_gpu_devices(&self) -> Result<Vec<GpuDeviceModel>>;

    /// Get vendor name
    fn vendor_name(&self) -> &str;

    /// Check if vendor hardware is present
    async fn is_vendor_hardware_present(&self) -> Result<bool> {
        let devices = self.detect_gpu_devices().await?;
        Ok(!devices.is_empty())
    }

    /// Get vendor-specific capabilities
    async fn get_vendor_capabilities(&self) -> Result<Vec<String>> {
        Ok(Vec::new())
    }
}

/// Performance benchmark interface
///
/// Interface for implementing different types of performance benchmarks
/// with validation and result reporting capabilities.
#[async_trait]
pub trait PerformanceBenchmarkRunner {
    /// Execute the benchmark
    async fn execute(&self) -> Result<BenchmarkResults>;

    /// Get benchmark name
    fn name(&self) -> &str;

    /// Get benchmark description
    fn description(&self) -> &str;

    /// Validate benchmark prerequisites
    async fn validate_prerequisites(&self) -> Result<bool>;

    /// Get expected execution duration
    fn expected_duration(&self) -> Duration;
}

/// Monitoring sensor interface
///
/// Interface for implementing different types of monitoring sensors
/// with reading collection and status reporting capabilities.
#[async_trait]
pub trait MonitoringSensor {
    /// Read current sensor value
    async fn read_value(&self) -> Result<f64>;

    /// Get sensor name
    fn sensor_name(&self) -> &str;

    /// Get sensor units
    fn units(&self) -> &str;

    /// Check sensor status
    async fn check_status(&self) -> Result<HealthStatus>;

    /// Get sensor metadata
    fn metadata(&self) -> HashMap<String, String>;
}

/// System analyzer interface
///
/// Interface for implementing different types of system analyzers
/// with analysis execution and result reporting capabilities.
#[async_trait]
pub trait SystemAnalyzer {
    /// Perform system analysis
    async fn analyze(&self) -> Result<Box<dyn AnalysisResults + Send + Sync>>;

    /// Get analyzer name
    fn name(&self) -> &str;

    /// Check if analysis is applicable
    async fn is_applicable(&self) -> Result<bool>;

    /// Get analysis dependencies
    fn dependencies(&self) -> Vec<String>;
}

/// Generic analysis results
///
/// Generic interface for analysis results with common
/// metadata and serialization capabilities.
pub trait AnalysisResults {
    /// Get analysis timestamp
    fn timestamp(&self) -> DateTime<Utc>;

    /// Get analysis confidence
    fn confidence(&self) -> f32;

    /// Get analysis summary
    fn summary(&self) -> String;

    /// Serialize results to JSON
    fn to_json(&self) -> Result<String>;
}

/// Resource optimizer interface
///
/// Interface for implementing different resource optimization algorithms
/// with strategy execution and effectiveness measurement.
#[async_trait]
pub trait ResourceOptimizer {
    /// Execute optimization strategy
    async fn optimize(
        &self,
        strategy: &OptimizationStrategy,
    ) -> Result<Box<dyn OptimizationResults + Send + Sync>>;

    /// Get optimizer name
    fn name(&self) -> &str;

    /// Check if strategy is supported
    fn supports_strategy(&self, strategy: &OptimizationStrategy) -> bool;

    /// Get optimization capabilities
    fn capabilities(&self) -> Vec<String>;
}

/// Generic optimization results
///
/// Generic interface for optimization results with effectiveness
/// measurement and improvement tracking capabilities.
pub trait OptimizationResults {
    /// Get optimization effectiveness
    fn effectiveness(&self) -> f32;

    /// Get performance improvement
    fn improvement(&self) -> f32;

    /// Get optimization timestamp
    fn timestamp(&self) -> DateTime<Utc>;

    /// Get optimization details
    fn details(&self) -> HashMap<String, String>;
}

/// Alert handler interface
///
/// Interface for implementing alert handling mechanisms
/// with notification and escalation capabilities.
#[async_trait]
pub trait AlertHandler {
    /// Handle alert
    async fn handle_alert(&self, alert: &SystemAlert) -> Result<()>;

    /// Get handler name
    fn name(&self) -> &str;

    /// Check if handler supports alert type
    fn supports_alert_type(&self, alert_type: &str) -> bool;

    /// Get handler configuration
    fn configuration(&self) -> HashMap<String, String>;
}

// =============================================================================
// SUPPORTING TYPES FOR TRAITS
// =============================================================================

/// System alert information
///
/// System alert with severity, description, and metadata
/// for alert handling and notification systems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemAlert {
    /// Alert ID
    pub id: String,

    /// Alert severity
    pub severity: Severity,

    /// Alert description
    pub description: String,

    /// Alert source
    pub source: String,

    /// Alert timestamp
    pub timestamp: DateTime<Utc>,

    /// Alert metadata
    pub metadata: HashMap<String, String>,
}

// =============================================================================
// IMPLEMENTATION HELPERS
// =============================================================================

impl<T> UtilizationHistory<T> {
    /// Create new utilization history with specified capacity
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size,
            samples: Vec::with_capacity(max_size),
        }
    }

    /// Add new sample to history
    pub fn add_sample(&mut self, value: T, timestamp: DateTime<Utc>) {
        self.samples.push((value, timestamp));
        if self.samples.len() > self.max_size {
            self.samples.remove(0);
        }
    }

    /// Get all samples
    pub fn get_samples(&self) -> &[(T, DateTime<Utc>)] {
        &self.samples
    }

    /// Get sample count
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Clear all samples
    pub fn clear(&mut self) {
        self.samples.clear();
    }
}

impl UtilizationStats {
    /// Create utilization statistics from sample data
    pub fn from_samples(samples: &[f32]) -> Self {
        if samples.is_empty() {
            return Self {
                average: 0.0,
                minimum: 0.0,
                maximum: 0.0,
                std_deviation: 0.0,
                percentile_95: 0.0,
                percentile_99: 0.0,
            };
        }

        let sum: f32 = samples.iter().sum();
        let average = sum / samples.len() as f32;

        let minimum = samples.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let maximum = samples.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let variance =
            samples.iter().map(|&x| (x - average).powi(2)).sum::<f32>() / samples.len() as f32;
        let std_deviation = variance.sqrt();

        let mut sorted_samples = samples.to_vec();
        sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let percentile_95_idx = ((samples.len() as f32 * 0.95) as usize).min(samples.len() - 1);
        let percentile_99_idx = ((samples.len() as f32 * 0.99) as usize).min(samples.len() - 1);

        Self {
            average,
            minimum,
            maximum,
            std_deviation,
            percentile_95: sorted_samples[percentile_95_idx],
            percentile_99: sorted_samples[percentile_99_idx],
        }
    }
}

impl HardwareDetectionCache {
    /// Create new hardware detection cache
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear all cached data
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Check if cache has data for specific component
    pub fn has_cpu_data(&self) -> bool {
        self.cpu_frequencies.is_some() && self.cache_hierarchy.is_some()
    }

    /// Check if cache has memory data
    pub fn has_memory_data(&self) -> bool {
        self.memory_characteristics.is_some()
    }

    /// Check if cache has storage data
    pub fn has_storage_data(&self) -> bool {
        self.storage_devices.is_some()
    }

    /// Check if cache has network data
    pub fn has_network_data(&self) -> bool {
        self.network_interfaces.is_some()
    }

    /// Check if cache has GPU data
    pub fn has_gpu_data(&self) -> bool {
        self.gpu_devices.is_some()
    }
}

// =============================================================================
// DEFAULT IMPLEMENTATIONS
// =============================================================================

// ResourceModelingConfig Default and impl blocks removed - use manager.rs version

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            cpu_benchmark_iterations: 3,
            enable_gpu_profiling: true,
            cache_results: true,
            profiling_timeout: Duration::from_secs(60),
        }
    }
}

impl Default for TemperatureThresholds {
    fn default() -> Self {
        Self {
            warning_temperature: 75.0,
            critical_temperature: 85.0,
            shutdown_temperature: 95.0,
        }
    }
}

impl Default for UtilizationTrackingConfig {
    fn default() -> Self {
        Self {
            sample_interval: Duration::from_secs(1),
            history_size: 3600, // 1 hour at 1-second intervals
            detailed_tracking: true,
        }
    }
}

impl Default for HardwareDetectionConfig {
    fn default() -> Self {
        Self {
            enable_intel_detection: true,
            enable_amd_detection: true,
            enable_nvidia_detection: true,
            cache_detection_results: true,
            detection_timeout: Duration::from_secs(30),
        }
    }
}

impl Default for IntelDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl IntelDetector {
    /// Create new Intel detector
    pub fn new() -> Self {
        Self
    }
}

impl Default for AmdDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl AmdDetector {
    /// Create new AMD detector
    pub fn new() -> Self {
        Self
    }
}

impl Default for NvidiaDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl NvidiaDetector {
    /// Create new NVIDIA detector
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl VendorDetector for IntelDetector {
    async fn detect_gpu_devices(&self) -> Result<Vec<GpuDeviceModel>> {
        // Intel GPU detection implementation would go here
        Ok(Vec::new())
    }

    fn vendor_name(&self) -> &str {
        "Intel"
    }
}

#[async_trait]
impl VendorDetector for AmdDetector {
    async fn detect_gpu_devices(&self) -> Result<Vec<GpuDeviceModel>> {
        // AMD GPU detection implementation would go here
        Ok(Vec::new())
    }

    fn vendor_name(&self) -> &str {
        "AMD"
    }
}

#[async_trait]
impl VendorDetector for NvidiaDetector {
    async fn detect_gpu_devices(&self) -> Result<Vec<GpuDeviceModel>> {
        // NVIDIA GPU detection implementation would go here
        Ok(Vec::new())
    }

    fn vendor_name(&self) -> &str {
        "NVIDIA"
    }
}

// =============================================================================
// CONVERSION IMPLEMENTATIONS
// =============================================================================

// Removed redundant From<ResourceModelingError> for anyhow::Error
// anyhow already provides this via blanket impl for std::error::Error

impl From<DetectionError> for ResourceModelingError {
    fn from(err: DetectionError) -> Self {
        ResourceModelingError::HardwareDetection(err.to_string())
    }
}

impl From<ProfilingError> for ResourceModelingError {
    fn from(err: ProfilingError) -> Self {
        ResourceModelingError::PerformanceProfiling(err.to_string())
    }
}

impl From<MonitoringError> for ResourceModelingError {
    fn from(err: MonitoringError) -> Self {
        ResourceModelingError::TemperatureMonitoring(err.to_string())
    }
}

impl From<AnalysisError> for ResourceModelingError {
    fn from(err: AnalysisError) -> Self {
        ResourceModelingError::TopologyAnalysis(err.to_string())
    }
}

// =============================================================================
// HARDWARE DETECTOR SPECIFIC TYPES (added for hardware_detector.rs module)
// =============================================================================

/// Complete hardware profile containing all detected hardware information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompleteHardwareProfile {
    /// CPU hardware profile
    pub cpu_profile: CpuHardwareProfile,

    /// Memory hardware profile
    pub memory_profile: MemoryHardwareProfile,

    /// Storage hardware profile
    pub storage_profile: StorageHardwareProfile,

    /// Network hardware profile
    pub network_profile: NetworkHardwareProfile,

    /// GPU hardware profile
    pub gpu_profile: GpuHardwareProfile,

    /// Motherboard hardware profile
    pub motherboard_profile: MotherboardHardwareProfile,

    /// Vendor-specific optimizations
    pub vendor_optimizations: VendorOptimizations,

    /// System capability assessment
    pub capability_assessment: SystemCapabilityAssessment,

    /// Detection timestamp
    pub detection_timestamp: DateTime<Utc>,

    /// Detection duration
    pub detection_duration: Duration,
}

/// CPU hardware profile with vendor-specific features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuHardwareProfile {
    /// CPU vendor (Intel, AMD, ARM, etc.)
    pub vendor: String,

    /// CPU model
    pub model: String,

    /// Number of physical cores
    pub core_count: usize,

    /// Number of logical threads
    pub thread_count: usize,

    /// Base frequency in MHz
    pub base_frequency_mhz: u32,

    /// Maximum frequency in MHz
    pub max_frequency_mhz: u32,

    /// Cache hierarchy
    pub cache_hierarchy: CacheHierarchy,

    /// Vendor-specific features
    pub vendor_features: CpuVendorFeatures,

    /// Performance characteristics
    pub performance_characteristics: CpuPerformanceCharacteristics,

    /// Detection timestamp
    pub detection_timestamp: DateTime<Utc>,

    /// Detection duration
    pub detection_duration: Duration,
}

/// Memory hardware profile with timing and performance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryHardwareProfile {
    /// Total memory in MB
    pub total_memory_mb: u64,

    /// Available memory in MB
    pub available_memory_mb: u64,

    /// Memory type (DDR3, DDR4, DDR5, etc.)
    pub memory_type: MemoryType,

    /// Memory speed in MHz
    pub memory_speed_mhz: u32,

    /// Memory bandwidth in GB/s
    pub bandwidth_gbps: f32,

    /// Memory latency
    pub latency: Duration,

    /// Page size in KB
    pub page_size_kb: u32,

    /// Memory modules/DIMMs
    pub memory_modules: Vec<MemoryModule>,

    /// Memory performance metrics
    pub performance_metrics: MemoryPerformanceMetrics,

    /// Detection timestamp
    pub detection_timestamp: DateTime<Utc>,

    /// Detection duration
    pub detection_duration: Duration,
}

/// Storage hardware profile with performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StorageHardwareProfile {
    /// Storage devices
    pub storage_devices: Vec<StorageDevice>,

    /// Total storage capacity in GB
    pub total_capacity_gb: u64,

    /// Storage performance metrics
    pub performance_metrics: StoragePerformanceMetrics,

    /// Detection timestamp
    pub detection_timestamp: DateTime<Utc>,

    /// Detection duration
    pub detection_duration: Duration,
}

/// Network hardware profile with capability analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkHardwareProfile {
    /// Network interfaces
    pub network_interfaces: Vec<NetworkInterface>,

    /// Total bandwidth in Mbps
    pub total_bandwidth_mbps: f32,

    /// Network performance metrics
    pub performance_metrics: NetworkPerformanceMetrics,

    /// Detection timestamp
    pub detection_timestamp: DateTime<Utc>,

    /// Detection duration
    pub detection_duration: Duration,
}

/// GPU hardware profile with compute capabilities
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GpuHardwareProfile {
    /// GPU devices
    pub gpu_devices: Vec<GpuDeviceModel>,

    /// Total GPU memory in MB
    pub total_memory_mb: u64,

    /// Compute capabilities
    pub compute_capabilities: GpuComputeCapabilities,

    /// Vendor-specific features
    pub vendor_features: GpuVendorFeatures,

    /// Detection timestamp
    pub detection_timestamp: DateTime<Utc>,

    /// Detection duration
    pub detection_duration: Duration,
}

/// Motherboard hardware profile with chipset information
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MotherboardHardwareProfile {
    /// Motherboard information
    pub motherboard_info: MotherboardInfo,

    /// Chipset information
    pub chipset_info: ChipsetInfo,

    /// BIOS/UEFI information
    pub firmware_info: FirmwareInfo,

    /// Detection timestamp
    pub detection_timestamp: DateTime<Utc>,

    /// Detection duration
    pub detection_duration: Duration,
}

/// Memory module information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryModule {
    /// DIMM slot number
    pub slot: usize,

    /// Module size in MB
    pub size_mb: u64,

    /// Module speed in MHz
    pub speed_mhz: u32,

    /// Memory type
    pub memory_type: MemoryType,

    /// Manufacturer
    pub manufacturer: String,

    /// Part number
    pub part_number: String,
}

/// Memory performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPerformanceMetrics {
    /// Read bandwidth in GB/s
    pub read_bandwidth_gbps: f32,

    /// Write bandwidth in GB/s
    pub write_bandwidth_gbps: f32,

    /// Memory latency
    pub latency: Duration,

    /// Random access performance (relative score)
    pub random_access_performance: f32,
}

/// Storage performance metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StoragePerformanceMetrics {
    /// Sequential read performance in MB/s
    pub sequential_read_mbps: f32,

    /// Sequential write performance in MB/s
    pub sequential_write_mbps: f32,

    /// Random read IOPS
    pub random_read_iops: u32,

    /// Random write IOPS
    pub random_write_iops: u32,
}

/// Network performance metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkPerformanceMetrics {
    /// Network latency
    pub latency: Duration,

    /// Packet loss rate
    pub packet_loss_rate: f32,

    /// Maximum throughput in Mbps
    pub max_throughput_mbps: f32,
}

/// CPU vendor-specific features
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CpuVendorFeatures {
    /// Supported instruction sets
    pub instruction_sets: Vec<String>,

    /// Virtualization support
    pub virtualization_support: bool,

    /// Security features
    pub security_features: Vec<String>,

    /// Power management features
    pub power_management: Vec<String>,
}

/// GPU vendor-specific features
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GpuVendorFeatures {
    /// Compute APIs supported
    pub compute_apis: Vec<String>,

    /// Ray tracing support
    pub ray_tracing_support: bool,

    /// AI/ML acceleration features
    pub ml_acceleration: Vec<String>,

    /// Video encoding/decoding support
    pub video_codecs: Vec<String>,
}

/// GPU compute capabilities
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GpuComputeCapabilities {
    /// Compute performance in GFLOPS
    pub compute_performance_gflops: f64,

    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f32,

    /// Compute units/cores
    pub compute_units: u32,

    /// Shader ALUs
    pub shader_alus: u32,
}

/// Motherboard information
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MotherboardInfo {
    /// Manufacturer
    pub manufacturer: String,

    /// Model
    pub model: String,

    /// Version/revision
    pub version: String,

    /// Serial number
    pub serial_number: String,
}

/// Chipset information
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChipsetInfo {
    /// Chipset name
    pub name: String,

    /// Vendor
    pub vendor: String,

    /// Revision
    pub revision: String,

    /// Supported features
    pub features: Vec<String>,
}

/// Firmware information
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FirmwareInfo {
    /// Firmware type (BIOS/UEFI)
    pub firmware_type: String,

    /// Vendor
    pub vendor: String,

    /// Version
    pub version: String,

    /// Release date
    pub release_date: String,
}

/// Vendor-specific optimizations and recommendations
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VendorOptimizations {
    /// CPU optimizations
    pub cpu_optimizations: Vec<OptimizationRecommendation>,

    /// GPU optimizations
    pub gpu_optimizations: Vec<OptimizationRecommendation>,

    /// Memory optimizations
    pub memory_optimizations: Vec<OptimizationRecommendation>,

    /// System-level optimizations
    pub system_optimizations: Vec<OptimizationRecommendation>,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Optimization category
    pub category: String,

    /// Description
    pub description: String,

    /// Expected performance gain
    pub performance_gain: f32,

    /// Implementation difficulty (1-5 scale)
    pub difficulty: u8,

    /// Required configuration changes
    pub configuration_changes: Vec<String>,
}

/// System capability assessment
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SystemCapabilityAssessment {
    /// Overall system score (0-100)
    pub overall_score: f32,

    /// CPU capability score
    pub cpu_score: f32,

    /// Memory capability score
    pub memory_score: f32,

    /// Storage capability score
    pub storage_score: f32,

    /// Network capability score
    pub network_score: f32,

    /// GPU capability score
    pub gpu_score: Option<f32>,

    /// Capability breakdown
    pub capability_breakdown: CapabilityBreakdown,

    /// Performance bottlenecks
    pub bottlenecks: Vec<PerformanceBottleneck>,
}

/// Detailed capability breakdown
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CapabilityBreakdown {
    /// Compute capabilities
    pub compute_capabilities: ComputeCapabilities,

    /// Memory capabilities
    pub memory_capabilities: MemoryCapabilities,

    /// I/O capabilities
    pub io_capabilities: IoCapabilities,

    /// Specialized capabilities
    pub specialized_capabilities: SpecializedCapabilities,
}

/// Compute capabilities assessment
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ComputeCapabilities {
    /// Single-threaded performance score
    pub single_thread_score: f32,

    /// Multi-threaded performance score
    pub multi_thread_score: f32,

    /// Vector processing score
    pub vector_processing_score: f32,

    /// Branch prediction efficiency
    pub branch_prediction_score: f32,
}

/// Memory capabilities assessment
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryCapabilities {
    /// Memory bandwidth score
    pub bandwidth_score: f32,

    /// Memory latency score
    pub latency_score: f32,

    /// Memory capacity score
    pub capacity_score: f32,

    /// Cache efficiency score
    pub cache_efficiency_score: f32,
}

/// I/O capabilities assessment
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IoCapabilities {
    /// Storage performance score
    pub storage_score: f32,

    /// Network performance score
    pub network_score: f32,

    /// I/O latency score
    pub latency_score: f32,

    /// Concurrent I/O score
    pub concurrent_io_score: f32,
}

/// Specialized capabilities assessment
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SpecializedCapabilities {
    /// GPU compute score
    pub gpu_compute_score: Option<f32>,

    /// AI/ML acceleration score
    pub ml_acceleration_score: Option<f32>,

    /// Cryptographic acceleration score
    pub crypto_acceleration_score: Option<f32>,

    /// Video processing score
    pub video_processing_score: Option<f32>,
}

/// Performance bottleneck identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    /// Bottleneck component
    pub component: String,

    /// Severity (1-5 scale)
    pub severity: u8,

    /// Description
    pub description: String,

    /// Impact on performance
    pub performance_impact: f32,

    /// Recommended solutions
    pub solutions: Vec<String>,

    /// Subsystem affected
    pub subsystem: String,

    /// Bottleneck type
    pub bottleneck_type: String,

    /// Severity score
    pub severity_score: f64,

    /// Impact percentage
    pub impact_percentage: f64,

    /// Recommended actions
    pub recommended_actions: Vec<String>,
}

/// Hardware validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether the hardware profile is valid
    pub is_valid: bool,

    /// Validation errors
    pub errors: Vec<ValidationError>,

    /// Validation warnings
    pub warnings: Vec<ValidationWarning>,
}

/// Hardware validation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// Error code
    pub code: String,

    /// Error message
    pub message: String,

    /// Affected component
    pub component: String,
}

/// Hardware validation warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    /// Warning code
    pub code: String,

    /// Warning message
    pub message: String,

    /// Affected component
    pub component: String,
}

/// Validation rule result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRuleResult {
    /// Rule name
    pub rule_name: String,

    /// Whether the rule passed
    pub passed: bool,

    /// Error message if failed
    pub error_message: Option<String>,

    /// Warning message if applicable
    pub warning_message: Option<String>,
}

/// Vendor optimization rules
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VendorOptimizationRules {
    /// CPU optimization rules
    pub cpu_rules: Vec<OptimizationRule>,

    /// GPU optimization rules
    pub gpu_rules: Vec<OptimizationRule>,

    /// Memory optimization rules
    pub memory_rules: Vec<OptimizationRule>,
}

/// Optimization rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRule {
    /// Rule name
    pub name: String,

    /// Condition for applying the rule
    pub condition: String,

    /// Optimization recommendation
    pub recommendation: OptimizationRecommendation,
}
