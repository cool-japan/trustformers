//! Device detection and hardware capability tests for trustformers-mobile
//!
//! Tests device info structures, backend availability, and capability flags
//! using constructed data without actual hardware calls.

use trustformers_mobile::device_info::{
    GpuApi, GpuPerformanceTier, NpuPrecision, TemperatureSensor,
};
use trustformers_mobile::{
    BasicDeviceInfo, ChargingStatus, CpuInfo, GpuInfo, MemoryInfo, MobileBackend, MobileDeviceInfo,
    MobilePlatform, NpuInfo, PerformanceScores, PerformanceTier, PowerInfo, SimdSupport,
    ThermalInfo, ThermalState,
};

fn make_test_device_info() -> MobileDeviceInfo {
    MobileDeviceInfo::default()
}

fn make_flagship_device() -> MobileDeviceInfo {
    MobileDeviceInfo {
        basic_info: BasicDeviceInfo {
            platform: MobilePlatform::Ios,
            manufacturer: "Apple".to_string(),
            model: "iPhone 15 Pro".to_string(),
            os_version: "17.0".to_string(),
            hardware_id: "iphone15pro-001".to_string(),
            device_generation: Some(2023),
        },
        platform: MobilePlatform::Ios,
        cpu_info: CpuInfo {
            architecture: "arm64e".to_string(),
            core_count: 6,
            performance_cores: 2,
            efficiency_cores: 4,
            total_cores: 6,
            max_frequency_mhz: Some(3800),
            l1_cache_kb: Some(128),
            l2_cache_kb: Some(1024),
            l3_cache_kb: Some(16384),
            simd_support: SimdSupport::Cutting,
            features: vec![
                "neon".to_string(),
                "fp16".to_string(),
                "dotprod".to_string(),
            ],
        },
        memory_info: MemoryInfo {
            total_mb: 8192,
            available_mb: 5000,
            total_memory: 8192,
            available_memory: 5000,
            bandwidth_mbps: Some(102400),
            memory_type: "LPDDR5".to_string(),
            frequency_mhz: Some(6400),
            is_low_memory_device: false,
        },
        gpu_info: Some(GpuInfo {
            vendor: "Apple".to_string(),
            model: "Apple GPU 6-core".to_string(),
            driver_version: "MetalKit 3.0".to_string(),
            memory_mb: None, // Unified memory
            compute_units: Some(6),
            supported_apis: vec![GpuApi::Metal3],
            performance_tier: GpuPerformanceTier::Flagship,
        }),
        npu_info: Some(NpuInfo {
            vendor: "Apple".to_string(),
            model: "Apple Neural Engine 3rd gen".to_string(),
            version: "3.0".to_string(),
            tops: Some(35.0),
            supported_precisions: vec![NpuPrecision::FP16, NpuPrecision::INT8, NpuPrecision::INT4],
            memory_bandwidth_mbps: Some(51200),
        }),
        thermal_info: ThermalInfo {
            current_state: ThermalState::Nominal,
            state: ThermalState::Nominal,
            throttling_supported: true,
            temperature_sensors: vec![TemperatureSensor {
                name: "CPU".to_string(),
                temperature_celsius: Some(35.0),
                max_temperature_celsius: Some(80.0),
            }],
            thermal_zones: vec!["cpu".to_string(), "gpu".to_string()],
        },
        power_info: PowerInfo {
            battery_capacity_mah: Some(4422),
            battery_level_percent: Some(90),
            battery_level: Some(90),
            battery_health_percent: Some(100),
            charging_status: ChargingStatus::NotCharging,
            is_charging: false,
            power_save_mode: false,
            low_power_mode_available: true,
        },
        available_backends: vec![
            MobileBackend::CPU,
            MobileBackend::CoreML,
            MobileBackend::Metal,
        ],
        performance_scores: PerformanceScores {
            cpu_single_core: Some(2500),
            cpu_multi_core: Some(8000),
            gpu_score: Some(15000),
            memory_score: Some(12000),
            overall_tier: PerformanceTier::Flagship,
            tier: PerformanceTier::Flagship,
        },
    }
}

#[test]
fn test_default_device_info_is_generic_platform() {
    let info = make_test_device_info();
    assert_eq!(info.platform, MobilePlatform::Generic);
}

#[test]
fn test_default_device_info_has_cpu_backend() {
    let info = make_test_device_info();
    assert!(info.available_backends.contains(&MobileBackend::CPU));
}

#[test]
fn test_default_device_has_no_gpu() {
    let info = make_test_device_info();
    assert!(info.gpu_info.is_none());
}

#[test]
fn test_flagship_device_has_npu() {
    let info = make_flagship_device();
    assert!(info.npu_info.is_some());
}

#[test]
fn test_flagship_device_has_metal_backend() {
    let info = make_flagship_device();
    assert!(info.available_backends.contains(&MobileBackend::Metal));
}

#[test]
fn test_flagship_device_performance_tier_is_flagship() {
    let info = make_flagship_device();
    assert_eq!(
        info.performance_scores.overall_tier,
        PerformanceTier::Flagship
    );
}

#[test]
fn test_performance_tier_ordering() {
    assert!(PerformanceTier::VeryLow < PerformanceTier::Low);
    assert!(PerformanceTier::Low < PerformanceTier::Budget);
    assert!(PerformanceTier::Budget < PerformanceTier::Medium);
    assert!(PerformanceTier::Medium < PerformanceTier::High);
    assert!(PerformanceTier::High < PerformanceTier::Flagship);
}

#[test]
fn test_simd_support_variants_exist() {
    let _none = SimdSupport::None;
    let _basic = SimdSupport::Basic;
    let _advanced = SimdSupport::Advanced;
    let _cutting = SimdSupport::Cutting;
}

#[test]
fn test_charging_status_variants_exist() {
    let _unknown = ChargingStatus::Unknown;
    let _charging = ChargingStatus::Charging;
    let _discharging = ChargingStatus::Discharging;
    let _not_charging = ChargingStatus::NotCharging;
    let _full = ChargingStatus::Full;
}

#[test]
fn test_mobile_backend_variants_include_hardware_accelerators() {
    let _cpu = MobileBackend::CPU;
    let _coreml = MobileBackend::CoreML;
    let _nnapi = MobileBackend::NNAPI;
    let _gpu = MobileBackend::GPU;
    let _metal = MobileBackend::Metal;
    let _vulkan = MobileBackend::Vulkan;
}

#[test]
fn test_device_info_serialization_roundtrip() {
    let info = make_flagship_device();
    let json = serde_json::to_string(&info).expect("serialization should succeed");
    let restored: MobileDeviceInfo =
        serde_json::from_str(&json).expect("deserialization should succeed");
    assert_eq!(restored.platform, info.platform);
    assert_eq!(restored.basic_info.model, info.basic_info.model);
    assert_eq!(restored.cpu_info.total_cores, info.cpu_info.total_cores,);
}

#[test]
fn test_npu_precision_variants() {
    let _fp32 = NpuPrecision::FP32;
    let _fp16 = NpuPrecision::FP16;
    let _bf16 = NpuPrecision::BF16;
    let _int8 = NpuPrecision::INT8;
    let _int4 = NpuPrecision::INT4;
    let _int1 = NpuPrecision::INT1;
}

#[test]
fn test_thermal_state_eq_and_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(ThermalState::Nominal);
    set.insert(ThermalState::Critical);
    set.insert(ThermalState::Nominal); // duplicate
    assert_eq!(set.len(), 2);
}

#[test]
fn test_gpu_performance_tier_variants() {
    let _low = GpuPerformanceTier::Low;
    let _medium = GpuPerformanceTier::Medium;
    let _high = GpuPerformanceTier::High;
    let _flagship = GpuPerformanceTier::Flagship;
}

#[test]
fn test_mobile_platform_variants() {
    let _ios = MobilePlatform::Ios;
    let _android = MobilePlatform::Android;
    let _generic = MobilePlatform::Generic;
    assert_ne!(MobilePlatform::Ios, MobilePlatform::Android);
}
