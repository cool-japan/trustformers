//! Mobile Testing Infrastructure
//!
//! This module provides comprehensive testing utilities specifically designed for mobile
//! environments, including performance benchmarks, battery tests, stress testing,
//! device farm integration, and result aggregation.

pub mod config;
pub mod device_farm;
pub mod framework;
pub mod providers;
pub mod results;

// `device_farm_tests.rs` existed in the tree without a `mod` declaration, so
// none of its tests were ever compiled or run. Wired in here.
#[cfg(test)]
#[path = "device_farm_tests.rs"]
mod device_farm_tests;

// Re-export key types from submodules
pub use config::*;
pub use device_farm::{
    AggregationRules, DeviceFarmManager, DeviceFarmSession, ResultAggregator, StatisticalMethod,
};
pub use framework::MobileTestingFramework;
pub use providers::*;
pub use results::*;

// Add convenience imports
use trustformers_core::error::Result;

/// Create a mobile testing framework with default configuration
pub fn create_mobile_testing_framework() -> Result<MobileTestingFramework> {
    let config = MobileTestingConfig::default();
    MobileTestingFramework::new(config)
}

/// Create a device farm manager with the specified provider
pub fn create_device_farm_manager(provider: DeviceFarmProvider) -> Result<DeviceFarmManager> {
    let config = DeviceFarmConfig {
        provider,
        credentials: DeviceFarmCredentials {
            access_key: None,
            secret_key: None,
            api_token: None,
            username: None,
            password: None,
            service_account_json: None,
        },
        execution_settings: DeviceFarmExecutionSettings {
            max_parallel_devices: 5,
            execution_timeout: std::time::Duration::from_secs(1800), // 30 minutes
            retry_failed_tests: true,
            max_retry_attempts: 3,
            video_recording: true,
            screenshot_capture: true,
            performance_monitoring: true,
            device_logs: true,
            network_logs: false,
            app_crash_logs: true,
        },
        device_selection: DeviceSelectionCriteria {
            device_types: vec![DeviceType::Phone],
            os_versions: vec!["14".to_string(), "17.0".to_string()],
            hardware_requirements: HardwareRequirements {
                min_ram_mb: 4096,
                min_cpu_cores: 4,
                min_storage_gb: 32,
                required_sensors: vec!["camera".to_string()],
                required_connectivity: vec!["wifi".to_string()],
            },
            gpu_requirements: GpuRequirements {
                vendor: None,
                min_memory_mb: Some(2048),
                required_features: vec!["opencl".to_string()],
                min_compute_capability: Some(3.0),
            },
            regions: vec!["us-west-2".to_string()],
            availability_requirements: vec!["available".to_string()],
        },
        parallelism: DeviceFarmParallelism {
            distribution_strategy: TestDistributionStrategy::PerformanceOptimal,
            load_balancing: LoadBalancingSettings {
                dynamic_rebalancing: true,
                rebalancing_interval: std::time::Duration::from_secs(300),
            },
        },
        timeouts: DeviceFarmTimeouts {
            device_allocation_timeout: std::time::Duration::from_secs(300),
            test_execution_timeout: std::time::Duration::from_secs(1800),
            result_collection_timeout: std::time::Duration::from_secs(300),
            device_cleanup_timeout: std::time::Duration::from_secs(120),
            overall_session_timeout: std::time::Duration::from_secs(3600),
        },
        result_aggregation: ResultAggregationSettings {
            statistical_analysis: StatisticalAnalysisSettings {
                calculate_percentiles: true,
                percentile_levels: vec![50.0, 75.0, 90.0, 95.0, 99.0],
                confidence_level: 0.95,
                outlier_detection: true,
            },
            report_generation: ReportGenerationSettings {
                formats: vec![ReportFormat::JSON, ReportFormat::HTML],
                include_device_details: true,
                include_charts: true,
                include_raw_data: false,
            },
        },
    };

    DeviceFarmManager::new(config)
}

/// Create a result aggregator with default rules
pub fn create_result_aggregator() -> ResultAggregator {
    let rules = AggregationRules {
        statistical_methods: vec![
            StatisticalMethod::Mean,
            StatisticalMethod::Median,
            StatisticalMethod::Percentile(95),
            StatisticalMethod::StandardDeviation,
        ],
        outlier_detection: true,
        confidence_level: 0.95,
        minimum_sample_size: 3,
    };

    ResultAggregator::new(rules)
}

/// Quick test runner for basic mobile testing
pub async fn run_quick_mobile_test(mobile_config: crate::MobileConfig) -> Result<TestSuiteResults> {
    let mut framework = create_mobile_testing_framework()?;
    framework.initialize(mobile_config)?;
    framework.run_test_suite().await
}

/// Device information detection utility
pub struct DeviceInfo {
    pub device_name: String,
    pub os_name: String,
    pub os_version: String,
    pub device_type: DeviceType,
    pub hardware_model: String,
    pub cpu_architecture: String,
    pub ram_mb: usize,
    pub storage_gb: usize,
    pub screen_resolution: (u32, u32),
    pub sensors: Vec<String>,
}

impl DeviceInfo {
    /// Describe the machine this process is running on, measured.
    ///
    /// RAM comes from the same real detector the rest of the crate uses
    /// ([`crate::device_info::MobileDeviceDetector`]); total storage from
    /// `sysinfo::Disks`; OS and architecture from the build target. Screen
    /// resolution and the sensor list have no portable source, so they are
    /// reported as absent/empty.
    ///
    /// The previous body returned one of three fixed device descriptions per
    /// compilation target (an "iPhone" with 6 GB / 128 GB / 1170x2532, a
    /// "Generic Android" with 8 GB / 256 GB / 1080x2340, or a 4 GB / 64 GB
    /// fallback) regardless of the actual machine, which
    /// [`Self::meets_requirements`] and [`Self::get_performance_tier`] then
    /// answered from.
    pub fn detect_current_device() -> Result<Self> {
        let detected = crate::device_info::MobileDeviceDetector::detect()?;

        let disks = sysinfo::Disks::new_with_refreshed_list();
        let storage_gb = (disks.iter().map(|disk| disk.total_space()).max().unwrap_or(0)
            / (1024 * 1024 * 1024)) as usize;

        Ok(Self {
            device_name: detected.basic_info.hardware_id.clone(),
            os_name: std::env::consts::OS.to_string(),
            os_version: detected.basic_info.os_version.clone(),
            device_type: DeviceType::Generic,
            hardware_model: detected.basic_info.model.clone(),
            cpu_architecture: std::env::consts::ARCH.to_string(),
            ram_mb: detected.memory_info.total_mb,
            storage_gb,
            // No portable API reports these for a host process.
            screen_resolution: (0, 0),
            sensors: Vec::new(),
        })
    }

    /// Check if device meets minimum requirements
    pub fn meets_requirements(&self, requirements: &HardwareRequirements) -> bool {
        self.ram_mb >= requirements.min_ram_mb
            && self.storage_gb >= requirements.min_storage_gb
            && requirements.required_sensors.iter().all(|sensor| self.sensors.contains(sensor))
    }

    /// Get device performance tier based on specs
    pub fn get_performance_tier(&self) -> crate::device_info::PerformanceTier {
        if self.ram_mb >= 12288 {
            crate::device_info::PerformanceTier::Flagship
        } else if self.ram_mb >= 8192 {
            crate::device_info::PerformanceTier::High
        } else if self.ram_mb >= 6144 {
            crate::device_info::PerformanceTier::Mid
        } else {
            crate::device_info::PerformanceTier::Budget
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mobile_testing_config_defaults() {
        let config = MobileTestingConfig::default();
        assert!(config.enabled);
        assert!(config.enable_detailed_logging);
        assert_eq!(config.benchmark_config.warmup_iterations, 10);
        assert_eq!(
            config.battery_test_config.target_battery_drain_percent_per_hour,
            5.0
        );
    }

    #[test]
    fn test_benchmark_config_defaults() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.warmup_iterations, 10);
        assert_eq!(config.benchmark_iterations, 100);
        assert!(!config.input_sizes.is_empty());
        assert!(config.precision_modes.contains(&PrecisionMode::FP32));
    }

    #[test]
    fn test_battery_test_config_defaults() {
        let config = BatteryTestConfig::default();
        assert_eq!(config.test_duration, std::time::Duration::from_secs(3600));
        assert_eq!(config.target_power_consumption_mw, 500.0);
        assert_eq!(config.target_battery_drain_percent_per_hour, 5.0);
    }

    #[test]
    fn test_stress_test_config_defaults() {
        let config = StressTestConfig::default();
        assert_eq!(config.concurrent_threads, 4);
        assert!(config.memory_pressure_enabled);
        assert!(config.thermal_stress_enabled);
        assert_eq!(config.cpu_stress_level, 0.8);
    }

    #[test]
    fn test_memory_test_config_defaults() {
        let config = MemoryTestConfig::default();
        assert!(config.enable_leak_detection);
        assert_eq!(config.max_memory_usage_mb, 512);
        assert!(config.gc_stress_enabled);
    }

    #[tokio::test]
    async fn test_mobile_testing_framework_creation() {
        let framework = create_mobile_testing_framework();
        assert!(framework.is_ok());
    }

    #[test]
    fn test_device_farm_manager_creation() {
        let provider = DeviceFarmProvider::Local {
            device_pool_size: 2,
            devices: vec!["device1".to_string(), "device2".to_string()],
        };
        let manager = create_device_farm_manager(provider);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_result_aggregator_creation() {
        let aggregator = create_result_aggregator();
        // Should create successfully
        let _ = aggregator;
    }

    #[test]
    fn test_device_info_detection() {
        let device_info = DeviceInfo::detect_current_device();
        assert!(device_info.is_ok());

        let info = device_info.expect("operation failed in test");
        assert!(!info.device_name.is_empty());
        assert!(info.ram_mb > 0);
        assert!(info.storage_gb > 0);
    }

    #[test]
    fn test_device_requirements_check() {
        let device_info = DeviceInfo {
            device_name: "test-device".to_string(),
            os_name: "iOS".to_string(),
            os_version: "17.0".to_string(),
            device_type: DeviceType::Phone,
            hardware_model: "iPhone".to_string(),
            cpu_architecture: "arm64".to_string(),
            ram_mb: 8192,
            storage_gb: 128,
            screen_resolution: (1170, 2532),
            sensors: vec!["camera".to_string(), "accelerometer".to_string()],
        };

        let requirements = HardwareRequirements {
            min_ram_mb: 4096,
            min_cpu_cores: 2,
            min_storage_gb: 64,
            required_sensors: vec!["camera".to_string()],
            required_connectivity: vec![],
        };

        assert!(device_info.meets_requirements(&requirements));

        let high_requirements = HardwareRequirements {
            min_ram_mb: 16384, // 16GB RAM - higher than device has
            min_cpu_cores: 2,
            min_storage_gb: 64,
            required_sensors: vec!["camera".to_string()],
            required_connectivity: vec![],
        };

        assert!(!device_info.meets_requirements(&high_requirements));
    }

    #[test]
    fn test_performance_tier_classification() {
        let flagship_device = DeviceInfo {
            device_name: "flagship".to_string(),
            os_name: "iOS".to_string(),
            os_version: "17.0".to_string(),
            device_type: DeviceType::Phone,
            hardware_model: "iPhone 15 Pro".to_string(),
            cpu_architecture: "arm64".to_string(),
            ram_mb: 12288, // 12GB RAM
            storage_gb: 512,
            screen_resolution: (1179, 2556),
            sensors: vec!["camera".to_string()],
        };

        assert_eq!(
            flagship_device.get_performance_tier(),
            crate::device_info::PerformanceTier::Flagship
        );

        let budget_device = DeviceInfo {
            device_name: "budget".to_string(),
            os_name: "Android".to_string(),
            os_version: "13".to_string(),
            device_type: DeviceType::Phone,
            hardware_model: "Budget Phone".to_string(),
            cpu_architecture: "aarch64".to_string(),
            ram_mb: 4096, // 4GB RAM
            storage_gb: 64,
            screen_resolution: (720, 1520),
            sensors: vec!["camera".to_string()],
        };

        assert_eq!(
            budget_device.get_performance_tier(),
            crate::device_info::PerformanceTier::Budget
        );
    }
}
