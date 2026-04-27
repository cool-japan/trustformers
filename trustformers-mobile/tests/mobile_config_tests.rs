//! MobileConfig and platform configuration tests for trustformers-mobile
//!
//! Tests mobile configuration creation, platform-specific defaults,
//! quantization configs, and serialization.

use trustformers_mobile::{
    MemoryOptimization, MobileBackend, MobileConfig, MobilePlatform, MobileQuantizationConfig,
    MobileQuantizationScheme,
};

#[test]
fn test_mobile_config_default_platform_is_generic() {
    let config = MobileConfig::default();
    assert_eq!(config.platform, MobilePlatform::Generic);
}

#[test]
fn test_mobile_config_default_backend_is_cpu() {
    let config = MobileConfig::default();
    assert_eq!(config.backend, MobileBackend::CPU);
}

#[test]
fn test_mobile_config_default_memory_is_conservative() {
    let config = MobileConfig::default();
    // Default should be conservative for mobile
    assert!(
        config.max_memory_mb <= 1024,
        "default memory should be <= 1024MB for mobile, got {}",
        config.max_memory_mb
    );
}

#[test]
fn test_mobile_config_default_has_quantization() {
    let config = MobileConfig::default();
    assert!(
        config.quantization.is_some(),
        "default config should enable quantization for mobile"
    );
}

#[test]
fn test_mobile_config_ios_optimized_uses_coreml() {
    let config = MobileConfig::ios_optimized();
    assert_eq!(config.platform, MobilePlatform::Ios);
    assert_eq!(config.backend, MobileBackend::CoreML);
}

#[test]
fn test_mobile_config_android_optimized_uses_nnapi() {
    let config = MobileConfig::android_optimized();
    assert_eq!(config.platform, MobilePlatform::Android);
    assert_eq!(config.backend, MobileBackend::NNAPI);
}

#[test]
fn test_mobile_config_ultra_low_memory_uses_int4() {
    let config = MobileConfig::ultra_low_memory();
    let quant = config.quantization.expect("ultra-low memory should have quantization");
    assert_eq!(quant.scheme, MobileQuantizationScheme::Int4);
}

#[test]
fn test_mobile_config_ultra_low_memory_is_256mb() {
    let config = MobileConfig::ultra_low_memory();
    assert!(
        config.max_memory_mb <= 256,
        "ultra-low memory should be <= 256MB, got {}",
        config.max_memory_mb
    );
}

#[test]
fn test_mobile_config_ios_has_more_memory_than_android() {
    let ios = MobileConfig::ios_optimized();
    let android = MobileConfig::android_optimized();
    // iOS devices typically have more RAM available
    assert!(ios.max_memory_mb >= android.max_memory_mb);
}

#[test]
fn test_memory_optimization_variants_exist() {
    let _minimal = MemoryOptimization::Minimal;
    let _balanced = MemoryOptimization::Balanced;
    let _maximum = MemoryOptimization::Maximum;
}

#[test]
fn test_mobile_quantization_scheme_variants_exist() {
    let _int8 = MobileQuantizationScheme::Int8;
    let _int4 = MobileQuantizationScheme::Int4;
    let _fp16 = MobileQuantizationScheme::FP16;
    let _dynamic = MobileQuantizationScheme::Dynamic;
}

#[test]
fn test_mobile_config_serialization_roundtrip() {
    let config = MobileConfig::default();
    let json = serde_json::to_string(&config).expect("serialization should succeed");
    let restored: MobileConfig =
        serde_json::from_str(&json).expect("deserialization should succeed");
    assert_eq!(restored.platform, config.platform);
    assert_eq!(restored.backend, config.backend);
    assert_eq!(restored.max_memory_mb, config.max_memory_mb);
    assert_eq!(restored.use_fp16, config.use_fp16);
}

#[test]
fn test_quantization_config_creation() {
    let quant = MobileQuantizationConfig {
        scheme: MobileQuantizationScheme::Int8,
        dynamic: true,
        per_channel: false,
    };
    assert_eq!(quant.scheme, MobileQuantizationScheme::Int8);
    assert!(quant.dynamic);
    assert!(!quant.per_channel);
}

#[test]
fn test_mobile_backend_variants_exist() {
    let _cpu = MobileBackend::CPU;
    let _coreml = MobileBackend::CoreML;
    let _nnapi = MobileBackend::NNAPI;
    let _gpu = MobileBackend::GPU;
    let _metal = MobileBackend::Metal;
    let _vulkan = MobileBackend::Vulkan;
    let _opencl = MobileBackend::OpenCL;
    let _custom = MobileBackend::Custom;
}

#[test]
fn test_mobile_config_fp16_enabled_by_default() {
    let config = MobileConfig::default();
    assert!(
        config.use_fp16,
        "FP16 should be enabled by default for mobile efficiency"
    );
}

#[test]
fn test_mobile_platform_hash_usable_as_map_key() {
    use std::collections::HashMap;
    let mut map: HashMap<MobilePlatform, &str> = HashMap::new();
    map.insert(MobilePlatform::Ios, "ios");
    map.insert(MobilePlatform::Android, "android");
    map.insert(MobilePlatform::Generic, "generic");
    assert_eq!(map.len(), 3);
    assert_eq!(map[&MobilePlatform::Ios], "ios");
}
