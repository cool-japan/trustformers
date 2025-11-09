//! Mobile Testing Configuration
//!
//! This module contains all configuration structures for the mobile testing framework.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Mobile testing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileTestingConfig {
    /// Enable testing framework
    pub enabled: bool,
    /// Maximum test duration in seconds
    pub max_test_duration: Duration,
    /// Performance benchmark configuration
    pub benchmark_config: BenchmarkConfig,
    /// Battery testing configuration
    pub battery_test_config: BatteryTestConfig,
    /// Stress testing configuration
    pub stress_test_config: StressTestConfig,
    /// Memory testing configuration
    pub memory_test_config: MemoryTestConfig,
    /// Enable detailed logging
    pub enable_detailed_logging: bool,
    /// Output directory for test results
    pub output_directory: String,
    /// Device farm configuration
    pub device_farm_config: Option<DeviceFarmConfig>,
}

/// Performance benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub warmup_iterations: usize,
    pub benchmark_iterations: usize,
    pub input_sizes: Vec<Vec<usize>>,
    pub target_latency_ms: f32,
    pub target_throughput: f32,
    pub precision_modes: Vec<PrecisionMode>,
    pub power_modes: Vec<PowerMode>,
    pub thermal_conditions: Vec<ThermalCondition>,
}

/// Battery testing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatteryTestConfig {
    pub test_duration: Duration,
    pub inference_frequency: Duration,
    pub power_measurement_interval: Duration,
    pub target_power_consumption_mw: f32,
    pub target_battery_drain_percent_per_hour: f32,
}

/// Stress testing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestConfig {
    pub test_duration: Duration,
    pub concurrent_threads: usize,
    pub memory_pressure_enabled: bool,
    pub thermal_stress_enabled: bool,
    pub cpu_stress_level: f32,
    pub memory_stress_level: f32,
}

/// Memory testing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTestConfig {
    pub enable_leak_detection: bool,
    pub memory_stress_duration: Duration,
    pub max_memory_usage_mb: usize,
    pub gc_stress_enabled: bool,
}

/// Device farm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceFarmConfig {
    /// Selected device farm provider
    pub provider: DeviceFarmProvider,
    /// Provider-specific credentials
    pub credentials: DeviceFarmCredentials,
    /// Execution settings
    pub execution_settings: DeviceFarmExecutionSettings,
    /// Device selection criteria
    pub device_selection: DeviceSelectionCriteria,
    /// Parallel execution settings
    pub parallelism: DeviceFarmParallelism,
    /// Timeout settings
    pub timeouts: DeviceFarmTimeouts,
    /// Result aggregation settings
    pub result_aggregation: ResultAggregationSettings,
}

/// Device farm providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceFarmProvider {
    AWS {
        region: String,
        project_name: String,
    },
    Firebase {
        project_id: String,
        test_lab_id: String,
    },
    BrowserStack {
        username: String,
        build_name: String,
    },
    SauceLabs {
        datacenter: String,
        build_name: String,
    },
    AppCenter {
        owner_name: String,
        app_name: String,
    },
    Local {
        device_pool_size: usize,
        devices: Vec<String>,
    },
}

/// Device farm credentials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceFarmCredentials {
    pub access_key: Option<String>,
    pub secret_key: Option<String>,
    pub api_token: Option<String>,
    pub username: Option<String>,
    pub password: Option<String>,
    pub service_account_json: Option<String>,
}

/// Device farm execution settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceFarmExecutionSettings {
    /// Maximum parallel device executions
    pub max_parallel_devices: usize,
    /// Test execution timeout per device
    pub execution_timeout: Duration,
    /// Retry failed tests
    pub retry_failed_tests: bool,
    /// Maximum retry attempts
    pub max_retry_attempts: usize,
    /// Video recording enabled
    pub video_recording: bool,
    /// Screenshot capture enabled
    pub screenshot_capture: bool,
    /// Performance monitoring enabled
    pub performance_monitoring: bool,
    /// Device logs collection enabled
    pub device_logs: bool,
    /// Network logs collection enabled
    pub network_logs: bool,
    /// App crash logs collection enabled
    pub app_crash_logs: bool,
}

/// Device selection criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceSelectionCriteria {
    /// Target device types
    pub device_types: Vec<DeviceType>,
    /// OS version requirements
    pub os_versions: Vec<String>,
    /// Hardware requirements
    pub hardware_requirements: HardwareRequirements,
    /// GPU requirements
    pub gpu_requirements: GpuRequirements,
    /// Geographic regions
    pub regions: Vec<String>,
    /// Device availability requirements
    pub availability_requirements: Vec<String>,
}

/// Device type enum
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    Phone,
    Tablet,
    Watch,
    TV,
    Auto,
    Generic,
}

/// Hardware requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareRequirements {
    /// Minimum RAM in MB
    pub min_ram_mb: usize,
    /// Minimum CPU cores
    pub min_cpu_cores: usize,
    /// Minimum storage in GB
    pub min_storage_gb: usize,
    /// Required sensors
    pub required_sensors: Vec<String>,
    /// Required connectivity
    pub required_connectivity: Vec<String>,
}

/// GPU requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuRequirements {
    /// GPU vendor requirements
    pub vendor: Option<String>,
    /// Minimum GPU memory in MB
    pub min_memory_mb: Option<usize>,
    /// Required GPU features
    pub required_features: Vec<String>,
    /// Compute capability requirements
    pub min_compute_capability: Option<f32>,
}

/// Device farm parallelism settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceFarmParallelism {
    /// Test distribution strategy
    pub distribution_strategy: TestDistributionStrategy,
    /// Load balancing settings
    pub load_balancing: LoadBalancingSettings,
}

/// Test distribution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestDistributionStrategy {
    RoundRobin,
    LeastLoaded,
    Random,
    DeviceCapabilityBased,
    GeographicOptimal,
    CostOptimal,
    PerformanceOptimal,
    AvailabilityBased,
    CustomWeighted { weights: Vec<(String, f32)> },
}

/// Load balancing settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingSettings {
    /// Enable dynamic rebalancing
    pub dynamic_rebalancing: bool,
    /// Rebalancing interval
    pub rebalancing_interval: Duration,
}

/// Device farm timeout settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceFarmTimeouts {
    /// Device allocation timeout
    pub device_allocation_timeout: Duration,
    /// Test execution timeout
    pub test_execution_timeout: Duration,
    /// Result collection timeout
    pub result_collection_timeout: Duration,
    /// Device cleanup timeout
    pub device_cleanup_timeout: Duration,
    /// Overall session timeout
    pub overall_session_timeout: Duration,
}

/// Result aggregation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultAggregationSettings {
    /// Statistical analysis settings
    pub statistical_analysis: StatisticalAnalysisSettings,
    /// Report generation settings
    pub report_generation: ReportGenerationSettings,
}

/// Statistical analysis settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysisSettings {
    /// Calculate percentiles
    pub calculate_percentiles: bool,
    /// Percentile levels to calculate
    pub percentile_levels: Vec<f32>,
    /// Statistical confidence level
    pub confidence_level: f32,
    /// Outlier detection enabled
    pub outlier_detection: bool,
}

/// Report generation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportGenerationSettings {
    /// Output formats
    pub formats: Vec<ReportFormat>,
    /// Include detailed device info
    pub include_device_details: bool,
    /// Include performance charts
    pub include_charts: bool,
    /// Include raw data
    pub include_raw_data: bool,
}

/// Report format enum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    JSON,
    XML,
    HTML,
    PDF,
    CSV,
    Markdown,
}

/// Precision mode for benchmarks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrecisionMode {
    FP32,
    FP16,
    INT8,
    INT4,
    Mixed,
}

/// Power mode for testing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PowerMode {
    HighPerformance,
    Balanced,
    PowerSaver,
    Adaptive,
}

/// Thermal condition for testing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThermalCondition {
    Cool,
    Nominal,
    Warm,
    Hot,
    Critical,
}

/// Target performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetMetrics {
    pub target_latency_ms: f32,
    pub target_throughput_ops_per_sec: f32,
    pub target_accuracy_percent: f32,
    pub target_memory_usage_mb: usize,
    pub target_power_consumption_mw: f32,
    pub target_thermal_efficiency: f32,
}

// Default implementations
impl Default for MobileTestingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_test_duration: Duration::from_secs(300), // 5 minutes
            benchmark_config: BenchmarkConfig::default(),
            battery_test_config: BatteryTestConfig::default(),
            stress_test_config: StressTestConfig::default(),
            memory_test_config: MemoryTestConfig::default(),
            enable_detailed_logging: true,
            output_directory: "./test_results".to_string(),
            device_farm_config: None,
        }
    }
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            benchmark_iterations: 100,
            input_sizes: vec![
                vec![1, 224, 224, 3], // Image input
                vec![1, 512],         // Text input
                vec![1, 1024],        // Feature vector
            ],
            target_latency_ms: 100.0,
            target_throughput: 10.0,
            precision_modes: vec![PrecisionMode::FP32, PrecisionMode::FP16],
            power_modes: vec![PowerMode::Balanced],
            thermal_conditions: vec![ThermalCondition::Nominal],
        }
    }
}

impl Default for BatteryTestConfig {
    fn default() -> Self {
        Self {
            test_duration: Duration::from_secs(3600),        // 1 hour
            inference_frequency: Duration::from_millis(100), // 10 FPS
            power_measurement_interval: Duration::from_secs(1),
            target_power_consumption_mw: 500.0,
            target_battery_drain_percent_per_hour: 5.0,
        }
    }
}

impl Default for StressTestConfig {
    fn default() -> Self {
        Self {
            test_duration: Duration::from_secs(600), // 10 minutes
            concurrent_threads: 4,
            memory_pressure_enabled: true,
            thermal_stress_enabled: true,
            cpu_stress_level: 0.8,
            memory_stress_level: 0.8,
        }
    }
}

impl Default for MemoryTestConfig {
    fn default() -> Self {
        Self {
            enable_leak_detection: true,
            memory_stress_duration: Duration::from_secs(300), // 5 minutes
            max_memory_usage_mb: 512,
            gc_stress_enabled: true,
        }
    }
}
