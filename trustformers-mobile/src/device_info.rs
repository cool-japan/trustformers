//! Mobile Device Information and Capability Detection
//!
//! This module provides comprehensive device detection and system information
//! gathering for mobile devices, enabling optimized inference configuration
//! based on actual hardware capabilities.

use crate::{MemoryOptimization, MobileBackend, MobileConfig, MobilePlatform};
use serde::{Deserialize, Serialize};
use trustformers_core::error::Result;

/// Comprehensive mobile device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileDeviceInfo {
    /// Basic device information
    pub basic_info: BasicDeviceInfo,
    /// Platform (alias for basic_info.platform)
    pub platform: MobilePlatform,
    /// CPU information and capabilities
    pub cpu_info: CpuInfo,
    /// Memory information
    pub memory_info: MemoryInfo,
    /// GPU information (if available)
    pub gpu_info: Option<GpuInfo>,
    /// Neural processing unit information
    pub npu_info: Option<NpuInfo>,
    /// Thermal management capabilities
    pub thermal_info: ThermalInfo,
    /// Power management information
    pub power_info: PowerInfo,
    /// Available backends for inference
    pub available_backends: Vec<MobileBackend>,
    /// Performance benchmarks
    pub performance_scores: PerformanceScores,
}

impl Default for MobileDeviceInfo {
    fn default() -> Self {
        Self {
            basic_info: BasicDeviceInfo {
                platform: MobilePlatform::Generic,
                manufacturer: "Generic".to_string(),
                model: "Test Device".to_string(),
                os_version: "1.0.0".to_string(),
                hardware_id: "test-device-001".to_string(),
                device_generation: Some(2023),
            },
            platform: MobilePlatform::Generic,
            cpu_info: CpuInfo {
                architecture: "arm64".to_string(),
                core_count: 4,
                performance_cores: 2,
                efficiency_cores: 2,
                total_cores: 4,
                max_frequency_mhz: Some(2400),
                l1_cache_kb: Some(64),
                l2_cache_kb: Some(512),
                l3_cache_kb: Some(2048),
                simd_support: SimdSupport::Basic,
                features: vec!["neon".to_string()],
            },
            memory_info: MemoryInfo {
                total_mb: 4096,
                available_mb: 2048,
                total_memory: 4096,
                available_memory: 2048,
                bandwidth_mbps: Some(25600),
                memory_type: "LPDDR4".to_string(),
                frequency_mhz: Some(1600),
                is_low_memory_device: false,
            },
            gpu_info: None,
            npu_info: None,
            thermal_info: ThermalInfo {
                current_state: ThermalState::Nominal,
                state: ThermalState::Nominal,
                throttling_supported: true,
                temperature_sensors: Vec::new(),
                thermal_zones: Vec::new(),
            },
            power_info: PowerInfo {
                battery_capacity_mah: Some(3000),
                battery_level_percent: Some(80),
                battery_level: Some(80),
                battery_health_percent: Some(100),
                charging_status: ChargingStatus::NotCharging,
                is_charging: false,
                power_save_mode: false,
                low_power_mode_available: true,
            },
            available_backends: vec![MobileBackend::CPU],
            performance_scores: PerformanceScores::default(),
        }
    }
}

/// Basic device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicDeviceInfo {
    /// Device platform
    pub platform: MobilePlatform,
    /// Device manufacturer
    pub manufacturer: String,
    /// Device model
    pub model: String,
    /// OS version
    pub os_version: String,
    /// Hardware identifier
    pub hardware_id: String,
    /// Device generation/year
    pub device_generation: Option<u32>,
}

/// CPU information and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    /// CPU architecture (arm64, x86_64, etc.)
    pub architecture: String,
    /// Total number of cores
    pub total_cores: usize,
    /// Core count (alias for total_cores)
    pub core_count: usize,
    /// Number of performance cores
    pub performance_cores: usize,
    /// Number of efficiency cores
    pub efficiency_cores: usize,
    /// Maximum CPU frequency (MHz)
    pub max_frequency_mhz: Option<usize>,
    /// L1 cache size per core (KB)
    pub l1_cache_kb: Option<usize>,
    /// L2 cache size (KB)
    pub l2_cache_kb: Option<usize>,
    /// L3 cache size (KB)
    pub l3_cache_kb: Option<usize>,
    /// CPU features (NEON, AVX, etc.)
    pub features: Vec<String>,
    /// SIMD support level
    pub simd_support: SimdSupport,
}

/// SIMD support levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimdSupport {
    None,
    Basic,    // ARM NEON or x86 SSE
    Advanced, // ARM NEON with FP16 or x86 AVX
    Cutting,  // Latest SIMD extensions
}

/// Memory information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    /// Total system memory (MB)
    pub total_mb: usize,
    /// Available memory for apps (MB)
    pub available_mb: usize,
    /// Total memory (alias for total_mb)
    pub total_memory: usize,
    /// Available memory (alias for available_mb)
    pub available_memory: usize,
    /// Memory bandwidth (MB/s)
    pub bandwidth_mbps: Option<usize>,
    /// Memory type (LPDDR4, LPDDR5, etc.)
    pub memory_type: String,
    /// Memory frequency (MHz)
    pub frequency_mhz: Option<usize>,
    /// Low memory device flag
    pub is_low_memory_device: bool,
}

/// GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// GPU vendor
    pub vendor: String,
    /// GPU model/name
    pub model: String,
    /// GPU driver version
    pub driver_version: String,
    /// GPU memory (MB, if available)
    pub memory_mb: Option<usize>,
    /// GPU compute units/cores
    pub compute_units: Option<usize>,
    /// Supported APIs
    pub supported_apis: Vec<GpuApi>,
    /// GPU performance tier
    pub performance_tier: GpuPerformanceTier,
}

/// GPU APIs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuApi {
    OpenGLES2,
    OpenGLES3,
    OpenGLES31,
    OpenCL,
    Vulkan,
    Vulkan10,
    Vulkan11,
    Vulkan12,
    Metal2,
    Metal3,
}

/// GPU performance tiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuPerformanceTier {
    Low,
    Medium,
    High,
    Flagship,
}

/// Neural Processing Unit information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpuInfo {
    /// NPU vendor/type
    pub vendor: String,
    /// NPU model
    pub model: String,
    /// NPU version
    pub version: String,
    /// TOPS (Trillions of Operations Per Second)
    pub tops: Option<f32>,
    /// Supported precision formats
    pub supported_precisions: Vec<NpuPrecision>,
    /// Memory bandwidth (MB/s)
    pub memory_bandwidth_mbps: Option<usize>,
}

/// NPU precision formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NpuPrecision {
    FP32,
    FP16,
    BF16,
    INT8,
    INT4,
    INT1,
}

/// Thermal management information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalInfo {
    /// Current thermal state
    pub current_state: ThermalState,
    /// State (alias for current_state)
    pub state: ThermalState,
    /// Thermal throttling support
    pub throttling_supported: bool,
    /// Temperature sensors available
    pub temperature_sensors: Vec<TemperatureSensor>,
    /// Thermal zones
    pub thermal_zones: Vec<String>,
}

/// Thermal states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThermalState {
    Nominal,
    Fair,
    Serious,
    Critical,
    Emergency,
    Shutdown,
}

/// Temperature sensor information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureSensor {
    /// Sensor name
    pub name: String,
    /// Current temperature (Celsius)
    pub temperature_celsius: Option<f32>,
    /// Maximum safe temperature
    pub max_temperature_celsius: Option<f32>,
}

/// Power management information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerInfo {
    /// Battery capacity (mAh)
    pub battery_capacity_mah: Option<usize>,
    /// Current battery level (0-100)
    pub battery_level_percent: Option<u8>,
    /// Battery level (alias for battery_level_percent)
    pub battery_level: Option<u8>,
    /// Battery health (0-100)
    pub battery_health_percent: Option<u8>,
    /// Charging status
    pub charging_status: ChargingStatus,
    /// Is charging (derived from charging_status)
    pub is_charging: bool,
    /// Power save mode active
    pub power_save_mode: bool,
    /// Low power mode available
    pub low_power_mode_available: bool,
}

/// Battery charging status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChargingStatus {
    Unknown,
    Charging,
    Discharging,
    NotCharging,
    Full,
}

/// Device performance scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceScores {
    /// CPU single-core score
    pub cpu_single_core: Option<u32>,
    /// CPU multi-core score
    pub cpu_multi_core: Option<u32>,
    /// GPU score
    pub gpu_score: Option<u32>,
    /// Memory bandwidth score
    pub memory_score: Option<u32>,
    /// Overall performance tier
    pub overall_tier: PerformanceTier,
    /// Performance tier (alias for overall_tier)
    pub tier: PerformanceTier,
}

impl Default for PerformanceScores {
    fn default() -> Self {
        Self {
            cpu_single_core: None,
            cpu_multi_core: None,
            gpu_score: None,
            memory_score: None,
            overall_tier: PerformanceTier::Budget,
            tier: PerformanceTier::Budget,
        }
    }
}

/// Performance tiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PerformanceTier {
    VeryLow,  // Very low-end devices
    Low,      // Low-end devices
    Budget,   // Entry-level devices
    Medium,   // Medium-range devices
    Mid,      // Mid-range devices
    High,     // High-end devices
    VeryHigh, // Very high-end devices
    Flagship, // Premium flagship devices
}

/// Device detector for gathering comprehensive device information
pub struct MobileDeviceDetector;

impl MobileDeviceDetector {
    /// Detect comprehensive device information
    pub fn detect() -> Result<MobileDeviceInfo> {
        let basic_info = Self::detect_basic_info()?;
        let cpu_info = Self::detect_cpu_info()?;
        let memory_info = Self::detect_memory_info()?;
        let gpu_info = Self::detect_gpu_info();
        let npu_info = Self::detect_npu_info();
        let thermal_info = Self::detect_thermal_info()?;
        let power_info = Self::detect_power_info()?;
        let available_backends = Self::detect_available_backends(&basic_info, &gpu_info, &npu_info);
        let performance_scores = Self::benchmark_performance(&cpu_info, &memory_info, &gpu_info)?;

        Ok(MobileDeviceInfo {
            basic_info: basic_info.clone(),
            platform: basic_info.platform,
            cpu_info,
            memory_info,
            gpu_info,
            npu_info,
            thermal_info,
            power_info,
            available_backends,
            performance_scores,
        })
    }

    /// Generate optimized mobile configuration based on device capabilities
    pub fn generate_optimized_config(device_info: &MobileDeviceInfo) -> MobileConfig {
        let mut config = match device_info.basic_info.platform {
            MobilePlatform::Ios => MobileConfig::ios_optimized(),
            MobilePlatform::Android => MobileConfig::android_optimized(),
            MobilePlatform::Generic => MobileConfig::default(),
        };

        // Adjust based on performance tier
        match device_info.performance_scores.overall_tier {
            PerformanceTier::VeryLow | PerformanceTier::Low => {
                Self::configure_for_budget_device(&mut config, device_info)
            },
            PerformanceTier::Budget => Self::configure_for_budget_device(&mut config, device_info),
            PerformanceTier::Medium | PerformanceTier::Mid => {
                Self::configure_for_mid_device(&mut config, device_info)
            },
            PerformanceTier::High => Self::configure_for_high_device(&mut config, device_info),
            PerformanceTier::VeryHigh | PerformanceTier::Flagship => {
                Self::configure_for_flagship_device(&mut config, device_info)
            },
        }

        // Adjust for thermal state
        Self::adjust_for_thermal_state(&mut config, device_info.thermal_info.current_state);

        // Adjust for power state
        Self::adjust_for_power_state(&mut config, &device_info.power_info);

        // Select optimal backend
        Self::select_optimal_backend(&mut config, &device_info.available_backends);

        config
    }

    // Platform-specific detection methods

    #[cfg(target_os = "android")]
    fn detect_basic_info() -> Result<BasicDeviceInfo> {
        // Use Android system properties and APIs
        Ok(BasicDeviceInfo {
            platform: MobilePlatform::Android,
            manufacturer: Self::get_android_manufacturer(),
            model: Self::get_android_model(),
            os_version: Self::get_android_version(),
            hardware_id: Self::get_android_hardware_id(),
            device_generation: Self::estimate_android_generation(),
        })
    }

    #[cfg(target_os = "ios")]
    fn detect_basic_info() -> Result<BasicDeviceInfo> {
        // Use iOS system APIs
        Ok(BasicDeviceInfo {
            platform: MobilePlatform::Ios,
            manufacturer: "Apple".to_string(),
            model: Self::get_ios_model(),
            os_version: Self::get_ios_version(),
            hardware_id: Self::get_ios_hardware_id(),
            device_generation: Self::estimate_ios_generation(),
        })
    }

    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    fn detect_basic_info() -> Result<BasicDeviceInfo> {
        Ok(BasicDeviceInfo {
            platform: MobilePlatform::Generic,
            manufacturer: "Unknown".to_string(),
            model: "Generic Device".to_string(),
            os_version: "Unknown".to_string(),
            hardware_id: "unknown".to_string(),
            device_generation: None,
        })
    }

    fn detect_cpu_info() -> Result<CpuInfo> {
        #[cfg(any(target_os = "android", target_os = "ios"))]
        {
            let total_cores = num_cpus::get();
            let architecture = std::env::consts::ARCH.to_string();

            // Platform-specific CPU detection
            #[cfg(target_os = "android")]
            let (perf_cores, eff_cores, features) = Self::detect_android_cpu_details(total_cores);

            #[cfg(target_os = "ios")]
            let (perf_cores, eff_cores, features) = Self::detect_ios_cpu_details(total_cores);

            let simd_support = Self::detect_simd_support(&architecture, &features);

            Ok(CpuInfo {
                architecture,
                total_cores,
                performance_cores: perf_cores,
                efficiency_cores: eff_cores,
                max_frequency_mhz: Self::detect_max_cpu_frequency(),
                l1_cache_kb: Self::detect_l1_cache_size(),
                l2_cache_kb: Self::detect_l2_cache_size(),
                l3_cache_kb: Self::detect_l3_cache_size(),
                features,
                simd_support,
            })
        }

        #[cfg(not(any(target_os = "android", target_os = "ios")))]
        {
            Ok(CpuInfo {
                architecture: std::env::consts::ARCH.to_string(),
                total_cores: num_cpus::get(),
                core_count: num_cpus::get(),
                performance_cores: num_cpus::get() / 2,
                efficiency_cores: num_cpus::get() / 2,
                max_frequency_mhz: None,
                l1_cache_kb: None,
                l2_cache_kb: None,
                l3_cache_kb: None,
                features: vec![],
                simd_support: SimdSupport::Basic,
            })
        }
    }

    fn detect_memory_info() -> Result<MemoryInfo> {
        #[cfg(any(target_os = "android", target_os = "ios"))]
        {
            let (total_mb, available_mb) = Self::get_memory_info_platform_specific()?;
            let bandwidth_mbps = Self::estimate_memory_bandwidth();
            let memory_type = Self::detect_memory_type();
            let frequency_mhz = Self::detect_memory_frequency();
            let is_low_memory_device = total_mb < 2048; // Less than 2GB

            Ok(MemoryInfo {
                total_mb,
                available_mb,
                bandwidth_mbps,
                memory_type,
                frequency_mhz,
                is_low_memory_device,
            })
        }

        #[cfg(not(any(target_os = "android", target_os = "ios")))]
        {
            Ok(MemoryInfo {
                total_mb: 4096, // Default assumption
                available_mb: 2048,
                total_memory: 4096,
                available_memory: 2048,
                bandwidth_mbps: None,
                memory_type: "Unknown".to_string(),
                frequency_mhz: None,
                is_low_memory_device: false,
            })
        }
    }

    fn detect_gpu_info() -> Option<GpuInfo> {
        #[cfg(target_os = "android")]
        {
            Self::detect_android_gpu_info()
        }

        #[cfg(target_os = "ios")]
        {
            Self::detect_ios_gpu_info()
        }

        #[cfg(not(any(target_os = "android", target_os = "ios")))]
        {
            None
        }
    }

    fn detect_npu_info() -> Option<NpuInfo> {
        #[cfg(target_os = "android")]
        {
            Self::detect_android_npu_info()
        }

        #[cfg(target_os = "ios")]
        {
            Self::detect_ios_neural_engine_info()
        }

        #[cfg(not(any(target_os = "android", target_os = "ios")))]
        {
            None
        }
    }

    fn detect_thermal_info() -> Result<ThermalInfo> {
        let current_state = Self::get_current_thermal_state();
        let throttling_supported = Self::is_thermal_throttling_supported();
        let temperature_sensors = Self::enumerate_temperature_sensors();
        let thermal_zones = Self::enumerate_thermal_zones();

        Ok(ThermalInfo {
            current_state,
            state: current_state,
            throttling_supported,
            temperature_sensors,
            thermal_zones,
        })
    }

    fn detect_power_info() -> Result<PowerInfo> {
        let battery_info = Self::get_battery_info();
        let charging_status = Self::get_charging_status();
        let power_save_mode = Self::is_power_save_mode_active();
        let low_power_mode_available = Self::is_low_power_mode_available();

        Ok(PowerInfo {
            battery_capacity_mah: battery_info.0,
            battery_level_percent: battery_info.1,
            battery_level: battery_info.1,
            battery_health_percent: battery_info.2,
            charging_status,
            is_charging: matches!(charging_status, ChargingStatus::Charging),
            power_save_mode,
            low_power_mode_available,
        })
    }

    fn detect_available_backends(
        basic_info: &BasicDeviceInfo,
        gpu_info: &Option<GpuInfo>,
        npu_info: &Option<NpuInfo>,
    ) -> Vec<MobileBackend> {
        let mut backends = vec![MobileBackend::CPU]; // CPU always available

        match basic_info.platform {
            MobilePlatform::Ios => {
                if npu_info.is_some() {
                    backends.push(MobileBackend::CoreML);
                }
                if gpu_info.is_some() {
                    backends.push(MobileBackend::GPU);
                }
            },
            MobilePlatform::Android => {
                if npu_info.is_some() {
                    backends.push(MobileBackend::NNAPI);
                }
                if gpu_info.is_some() {
                    backends.push(MobileBackend::GPU);
                }
            },
            MobilePlatform::Generic => {
                // Only CPU for generic platforms
            },
        }

        backends
    }

    fn benchmark_performance(
        cpu_info: &CpuInfo,
        memory_info: &MemoryInfo,
        gpu_info: &Option<GpuInfo>,
    ) -> Result<PerformanceScores> {
        // Run micro-benchmarks to assess performance
        let cpu_single_core = Self::benchmark_cpu_single_core();
        let cpu_multi_core = Self::benchmark_cpu_multi_core(cpu_info.total_cores);
        let gpu_score = gpu_info.as_ref().map(|_| Self::benchmark_gpu());
        let memory_score = Self::benchmark_memory_bandwidth();

        let overall_tier = Self::calculate_overall_tier(
            cpu_single_core,
            cpu_multi_core,
            gpu_score,
            memory_info.total_mb,
        );

        Ok(PerformanceScores {
            cpu_single_core,
            cpu_multi_core,
            gpu_score,
            memory_score,
            overall_tier,
            tier: overall_tier,
        })
    }

    // Configuration adjustment methods

    fn configure_for_budget_device(config: &mut MobileConfig, device_info: &MobileDeviceInfo) {
        config.memory_optimization = MemoryOptimization::Maximum;
        config.max_memory_mb = (device_info.memory_info.total_mb / 6).max(128); // Very conservative
        config.num_threads = 1;
        config.enable_batching = false;
        config.max_batch_size = 1;
        if let Some(ref mut quant) = config.quantization.as_mut() {
            quant.scheme = crate::MobileQuantizationScheme::Int4; // Aggressive quantization
            quant.dynamic = true;
        }
    }

    fn configure_for_mid_device(config: &mut MobileConfig, device_info: &MobileDeviceInfo) {
        config.memory_optimization = MemoryOptimization::Balanced;
        config.max_memory_mb = (device_info.memory_info.total_mb / 4).max(256);
        config.num_threads = (device_info.cpu_info.performance_cores).max(1);
        config.enable_batching = device_info.memory_info.total_mb >= 3072;
        config.max_batch_size = if config.enable_batching { 2 } else { 1 };
    }

    fn configure_for_high_device(config: &mut MobileConfig, device_info: &MobileDeviceInfo) {
        config.memory_optimization = MemoryOptimization::Balanced;
        config.max_memory_mb = (device_info.memory_info.total_mb / 3).max(512);
        config.num_threads = device_info.cpu_info.performance_cores + 1;
        config.enable_batching = true;
        config.max_batch_size = 4;
        if let Some(ref mut quant) = config.quantization.as_mut() {
            quant.scheme = crate::MobileQuantizationScheme::FP16; // Higher quality
        }
    }

    fn configure_for_flagship_device(config: &mut MobileConfig, device_info: &MobileDeviceInfo) {
        config.memory_optimization = MemoryOptimization::Minimal;
        config.max_memory_mb = (device_info.memory_info.total_mb / 2).max(1024);
        config.num_threads = device_info.cpu_info.total_cores;
        config.enable_batching = true;
        config.max_batch_size = 8;
        if let Some(ref mut quant) = config.quantization.as_mut() {
            quant.scheme = crate::MobileQuantizationScheme::FP16;
            quant.per_channel = true; // Higher quality quantization
        }
    }

    fn adjust_for_thermal_state(config: &mut MobileConfig, thermal_state: ThermalState) {
        match thermal_state {
            ThermalState::Critical | ThermalState::Emergency => {
                config.memory_optimization = MemoryOptimization::Maximum;
                config.num_threads = 1;
                config.enable_batching = false;
                config.max_batch_size = 1;
            },
            ThermalState::Serious => {
                config.num_threads = (config.num_threads / 2).max(1);
                config.max_batch_size = (config.max_batch_size / 2).max(1);
            },
            ThermalState::Fair => {
                config.num_threads = (config.num_threads * 3 / 4).max(1);
            },
            _ => {
                // No thermal adjustments needed
            },
        }
    }

    fn adjust_for_power_state(config: &mut MobileConfig, power_info: &PowerInfo) {
        if power_info.power_save_mode || power_info.battery_level_percent.unwrap_or(100) < 20 {
            // Aggressive power saving
            config.memory_optimization = MemoryOptimization::Maximum;
            config.num_threads = 1;
            config.enable_batching = false;
            config.backend = MobileBackend::CPU; // Prefer CPU over GPU/NPU
        } else if power_info.battery_level_percent.unwrap_or(100) < 50 {
            // Moderate power saving
            config.num_threads = (config.num_threads / 2).max(1);
            config.max_batch_size = (config.max_batch_size / 2).max(1);
        }
    }

    fn select_optimal_backend(config: &mut MobileConfig, available_backends: &[MobileBackend]) {
        // Prefer specialized hardware if available
        for &backend in available_backends {
            match backend {
                MobileBackend::CoreML | MobileBackend::NNAPI => {
                    config.backend = backend;
                    return;
                },
                MobileBackend::GPU => {
                    if config.backend == MobileBackend::CPU {
                        config.backend = backend;
                    }
                },
                _ => {},
            }
        }
    }

    // Platform-specific implementation helpers
    // These would be implemented with actual platform APIs

    #[cfg(target_os = "android")]
    fn get_android_manufacturer() -> String {
        // Use Android Build.MANUFACTURER
        "Unknown".to_string()
    }

    #[cfg(target_os = "android")]
    fn get_android_model() -> String {
        // Use Android Build.MODEL
        "Android Device".to_string()
    }

    #[cfg(target_os = "android")]
    fn get_android_version() -> String {
        // Use Android Build.VERSION.RELEASE
        "Unknown".to_string()
    }

    #[cfg(target_os = "android")]
    fn get_android_hardware_id() -> String {
        // Use Android Build.HARDWARE
        "unknown".to_string()
    }

    #[cfg(target_os = "android")]
    fn estimate_android_generation() -> Option<u32> {
        // Estimate based on model and year
        None
    }

    #[cfg(target_os = "android")]
    fn detect_android_cpu_details(total_cores: usize) -> (usize, usize, Vec<String>) {
        // Parse /proc/cpuinfo and detect big.LITTLE configuration
        let perf_cores = if total_cores >= 8 { 4 } else { total_cores / 2 };
        let eff_cores = total_cores - perf_cores;
        let features = vec!["neon".to_string()]; // ARM NEON
        (perf_cores, eff_cores, features)
    }

    #[cfg(target_os = "android")]
    fn detect_android_gpu_info() -> Option<GpuInfo> {
        // Use OpenGL ES queries
        Some(GpuInfo {
            vendor: "Unknown".to_string(),
            model: "Android GPU".to_string(),
            driver_version: "Unknown".to_string(),
            memory_mb: None,
            compute_units: None,
            supported_apis: vec![GpuApi::OpenGLES3],
            performance_tier: GpuPerformanceTier::Medium,
        })
    }

    #[cfg(target_os = "android")]
    fn detect_android_npu_info() -> Option<NpuInfo> {
        // Check for Qualcomm Hexagon, MediaTek APU, etc.
        None
    }

    #[cfg(target_os = "ios")]
    fn get_ios_model() -> String {
        // Use iOS APIs to get device model
        "iPhone".to_string()
    }

    #[cfg(target_os = "ios")]
    fn get_ios_version() -> String {
        // Use iOS APIs
        "Unknown".to_string()
    }

    #[cfg(target_os = "ios")]
    fn get_ios_hardware_id() -> String {
        // Use iOS APIs
        "unknown".to_string()
    }

    #[cfg(target_os = "ios")]
    fn estimate_ios_generation() -> Option<u32> {
        // Estimate based on device model
        None
    }

    #[cfg(target_os = "ios")]
    fn detect_ios_cpu_details(total_cores: usize) -> (usize, usize, Vec<String>) {
        // iOS devices typically have performance + efficiency cores
        let perf_cores = if total_cores >= 6 { 2 } else { total_cores / 2 };
        let eff_cores = total_cores - perf_cores;
        let features = vec!["neon".to_string(), "fp16".to_string()];
        (perf_cores, eff_cores, features)
    }

    #[cfg(target_os = "ios")]
    fn detect_ios_gpu_info() -> Option<GpuInfo> {
        // Use Metal APIs
        Some(GpuInfo {
            vendor: "Apple".to_string(),
            model: "Apple GPU".to_string(),
            driver_version: "Unknown".to_string(),
            memory_mb: None,
            compute_units: None,
            supported_apis: vec![GpuApi::Metal3],
            performance_tier: GpuPerformanceTier::High,
        })
    }

    #[cfg(target_os = "ios")]
    fn detect_ios_neural_engine_info() -> Option<NpuInfo> {
        // Detect Apple Neural Engine
        Some(NpuInfo {
            vendor: "Apple".to_string(),
            model: "Neural Engine".to_string(),
            version: "Unknown".to_string(),
            tops: Some(15.8), // Approximate for A15/A16
            supported_precisions: vec![NpuPrecision::FP16, NpuPrecision::INT8],
            memory_bandwidth_mbps: None,
        })
    }

    // Helper methods for detection

    fn detect_simd_support(architecture: &str, features: &[String]) -> SimdSupport {
        if architecture.contains("arm") || architecture.contains("aarch64") {
            if features.iter().any(|f| f.contains("fp16")) {
                SimdSupport::Advanced
            } else if features.iter().any(|f| f.contains("neon")) {
                SimdSupport::Basic
            } else {
                SimdSupport::None
            }
        } else if architecture.contains("x86") {
            if features.iter().any(|f| f.contains("avx")) {
                SimdSupport::Advanced
            } else if features.iter().any(|f| f.contains("sse")) {
                SimdSupport::Basic
            } else {
                SimdSupport::None
            }
        } else {
            SimdSupport::None
        }
    }

    fn detect_max_cpu_frequency() -> Option<usize> {
        // Read from /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq (Android)
        // or use iOS APIs
        None
    }

    fn detect_l1_cache_size() -> Option<usize> {
        // Parse CPU cache information
        None
    }

    fn detect_l2_cache_size() -> Option<usize> {
        None
    }

    fn detect_l3_cache_size() -> Option<usize> {
        None
    }

    #[cfg(any(target_os = "android", target_os = "ios"))]
    fn get_memory_info_platform_specific() -> Result<(usize, usize)> {
        // Platform-specific memory detection
        Ok((4096, 2048)) // Default: 4GB total, 2GB available
    }

    fn estimate_memory_bandwidth() -> Option<usize> {
        // Estimate based on memory type and frequency
        None
    }

    fn detect_memory_type() -> String {
        "Unknown".to_string()
    }

    fn detect_memory_frequency() -> Option<usize> {
        None
    }

    fn get_current_thermal_state() -> ThermalState {
        // Platform-specific thermal state detection
        ThermalState::Nominal
    }

    fn is_thermal_throttling_supported() -> bool {
        // Check if platform supports thermal throttling
        true
    }

    fn enumerate_temperature_sensors() -> Vec<TemperatureSensor> {
        // Enumerate available temperature sensors
        vec![]
    }

    fn enumerate_thermal_zones() -> Vec<String> {
        // Enumerate thermal zones
        vec![]
    }

    fn get_battery_info() -> (Option<usize>, Option<u8>, Option<u8>) {
        // Get battery capacity, level, and health
        (None, None, None)
    }

    fn get_charging_status() -> ChargingStatus {
        ChargingStatus::Unknown
    }

    fn is_power_save_mode_active() -> bool {
        false
    }

    fn is_low_power_mode_available() -> bool {
        true
    }

    // Performance benchmarking methods

    fn benchmark_cpu_single_core() -> Option<u32> {
        // Run single-core CPU benchmark
        Some(1000) // Placeholder score
    }

    fn benchmark_cpu_multi_core(cores: usize) -> Option<u32> {
        // Run multi-core CPU benchmark
        Some((1000 * cores) as u32) // Placeholder
    }

    fn benchmark_gpu() -> u32 {
        // Run GPU benchmark
        2000 // Placeholder
    }

    fn benchmark_memory_bandwidth() -> Option<u32> {
        // Benchmark memory bandwidth
        Some(1500) // Placeholder
    }

    fn calculate_overall_tier(
        cpu_single: Option<u32>,
        cpu_multi: Option<u32>,
        gpu_score: Option<u32>,
        memory_mb: usize,
    ) -> PerformanceTier {
        let cpu_score = cpu_single.unwrap_or(0) + cpu_multi.unwrap_or(0) / 4;
        let gpu_score = gpu_score.unwrap_or(0);
        let memory_factor = if memory_mb >= 8192 {
            1.2
        } else if memory_mb >= 4096 {
            1.0
        } else {
            0.8
        };

        let overall_score = ((cpu_score + gpu_score) as f32 * memory_factor) as u32;

        if overall_score >= 4000 {
            PerformanceTier::Flagship
        } else if overall_score >= 2500 {
            PerformanceTier::High
        } else if overall_score >= 1500 {
            PerformanceTier::Mid
        } else {
            PerformanceTier::Budget
        }
    }
}

impl MobileDeviceInfo {
    /// Check if device supports a specific feature
    pub fn supports_feature(&self, feature: &str) -> bool {
        match feature {
            "fp16" => {
                self.cpu_info.features.iter().any(|f| f.contains("fp16"))
                    || self
                        .npu_info
                        .as_ref()
                        .is_some_and(|npu| npu.supported_precisions.contains(&NpuPrecision::FP16))
            },
            "int8" => true, // Most devices support int8
            "int4" => self.npu_info.is_some(),
            "simd" => matches!(
                self.cpu_info.simd_support,
                SimdSupport::Basic | SimdSupport::Advanced | SimdSupport::Cutting
            ),
            "gpu" => self.gpu_info.is_some(),
            "npu" => self.npu_info.is_some(),
            "vulkan" => self.gpu_info.as_ref().is_some_and(|gpu| {
                gpu.supported_apis.iter().any(|api| {
                    matches!(api, GpuApi::Vulkan10 | GpuApi::Vulkan11 | GpuApi::Vulkan12)
                })
            }),
            "metal" => self.gpu_info.as_ref().is_some_and(|gpu| {
                gpu.supported_apis
                    .iter()
                    .any(|api| matches!(api, GpuApi::Metal2 | GpuApi::Metal3))
            }),
            _ => false,
        }
    }

    /// Get recommended memory allocation for inference
    pub fn get_recommended_memory_allocation(&self) -> usize {
        let base_allocation = match self.performance_scores.overall_tier {
            PerformanceTier::VeryLow => self.memory_info.total_mb / 16,
            PerformanceTier::Low => self.memory_info.total_mb / 12,
            PerformanceTier::Budget => self.memory_info.total_mb / 8,
            PerformanceTier::Medium => self.memory_info.total_mb / 6,
            PerformanceTier::Mid => self.memory_info.total_mb / 4,
            PerformanceTier::High => self.memory_info.total_mb / 3,
            PerformanceTier::VeryHigh => self.memory_info.total_mb / 2,
            PerformanceTier::Flagship => self.memory_info.total_mb / 2,
        };

        // Adjust for current memory pressure
        let available_ratio =
            self.memory_info.available_mb as f32 / self.memory_info.total_mb as f32;
        let adjusted = (base_allocation as f32 * available_ratio) as usize;

        adjusted.clamp(128, 2048) // Min 128MB, max 2GB
    }

    /// Check if device is suitable for on-device training
    pub fn supports_on_device_training(&self) -> bool {
        self.memory_info.total_mb >= 3072 && // At least 3GB RAM
        self.cpu_info.total_cores >= 4 && // At least 4 cores
        matches!(self.performance_scores.overall_tier, PerformanceTier::High | PerformanceTier::Flagship)
    }

    /// Get a summary string of device information
    pub fn summary(&self) -> String {
        format!(
            "Device: {} {} | Platform: {:?} | Memory: {}MB/{} MB | CPU: {} cores ({} perf + {} eff) | GPU: {} | NPU: {} | Performance: {:?}",
            self.basic_info.manufacturer,
            self.basic_info.model,
            self.basic_info.platform,
            self.memory_info.available_mb,
            self.memory_info.total_mb,
            self.cpu_info.total_cores,
            self.cpu_info.performance_cores,
            self.cpu_info.efficiency_cores,
            self.gpu_info.as_ref().map_or("None".to_string(), |gpu| format!("{} {}", gpu.vendor, gpu.model)),
            self.npu_info.as_ref().map_or("None".to_string(), |npu| npu.model.clone()),
            self.performance_scores.overall_tier
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_detection() {
        let device_info = MobileDeviceDetector::detect();
        assert!(device_info.is_ok());

        let info = device_info.unwrap();
        assert!(info.cpu_info.total_cores > 0);
        assert!(info.memory_info.total_mb > 0);
        assert!(!info.available_backends.is_empty());
    }

    #[test]
    fn test_config_generation() {
        let device_info = MobileDeviceDetector::detect().unwrap();
        let config = MobileDeviceDetector::generate_optimized_config(&device_info);

        assert!(config.validate().is_ok());
        assert!(config.max_memory_mb > 0);
        assert!(config.get_thread_count() > 0);
    }

    #[test]
    fn test_performance_tiers() {
        let budget_tier = PerformanceTier::Budget;
        let flagship_tier = PerformanceTier::Flagship;

        assert_ne!(budget_tier, flagship_tier);
        assert!(matches!(budget_tier, PerformanceTier::Budget));
    }

    #[test]
    fn test_simd_support() {
        let support = SimdSupport::Advanced;
        assert!(matches!(support, SimdSupport::Advanced));
    }

    #[test]
    fn test_thermal_states() {
        let state = ThermalState::Nominal;
        assert_eq!(state, ThermalState::Nominal);
    }

    #[test]
    fn test_feature_support() {
        let device_info = MobileDeviceDetector::detect().unwrap();

        // Test basic feature support
        let _simd_support = device_info.supports_feature("simd");
        let _int8_support = device_info.supports_feature("int8");
    }

    #[test]
    fn test_memory_allocation() {
        let device_info = MobileDeviceDetector::detect().unwrap();
        let allocation = device_info.get_recommended_memory_allocation();

        assert!(allocation >= 128);
        assert!(allocation <= 2048);
    }
}
