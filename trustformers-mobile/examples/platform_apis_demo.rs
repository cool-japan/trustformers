//! Platform APIs Demo
//!
//! Demonstrates Core ML and NNAPI integration for iOS and Android platforms

#[cfg(any(feature = "coreml", feature = "nnapi"))]
use std::path::Path;
use trustformers_mobile::{MobileBackend, MobileConfig, MobilePlatform};

#[cfg(all(target_os = "ios", feature = "coreml"))]
use trustformers_mobile::coreml::{CoreMLComputeUnits, CoreMLConfig, CoreMLEngine};

#[cfg(feature = "coreml")]
use trustformers_mobile::coreml_converter::{
    CoreMLConverterConfig, CoreMLFormat, CoreMLModelConverter, CoreMLQuantizationConfig,
    HardwareTarget, OptimizationLevel, PruningConfig, PruningMethod, QuantizationBits,
    QuantizationMethod,
};

#[cfg(all(target_os = "android", feature = "nnapi"))]
use trustformers_mobile::nnapi::{
    NNAPIConfig, NNAPIDeviceType, NNAPIEngine, NNAPIExecutionPreference,
};

use trustformers_core::Result;
#[cfg(feature = "nnapi")]
use trustformers_mobile::nnapi_converter::{
    CalibrationConfig, CalibrationMethod, FallbackStrategy, NNAPIConverterConfig, NNAPIFormat,
    NNAPIModelConverter, NNAPIOptimizationConfig, NNAPIQuantizationConfig, NNAPIQuantizationScheme,
    NNAPITargetDevice,
};

fn main() -> Result<()> {
    println!("TrustformeRS Platform APIs Demo");
    println!("===============================\n");

    // Detect platform
    let platform = detect_platform();
    println!("Detected platform: {:?}\n", platform);

    match platform {
        MobilePlatform::Ios => {
            #[cfg(feature = "coreml")]
            run_coreml_demo()?;
        },
        MobilePlatform::Android => {
            #[cfg(feature = "nnapi")]
            run_nnapi_demo()?;
        },
        MobilePlatform::Generic => {
            println!("Running generic mobile demo");
            run_generic_demo()?;
        },
    }

    Ok(())
}

/// Detect the current platform
fn detect_platform() -> MobilePlatform {
    #[cfg(target_os = "ios")]
    return MobilePlatform::Ios;

    #[cfg(target_os = "android")]
    return MobilePlatform::Android;

    #[cfg(not(any(target_os = "ios", target_os = "android")))]
    return MobilePlatform::Generic;
}

/// Run Core ML demo for iOS
#[cfg(all(target_os = "ios", feature = "coreml"))]
fn run_coreml_demo() -> Result<()> {
    println!("Core ML Demo (iOS)");
    println!("------------------\n");

    // 1. Model Conversion
    println!("1. Converting model to Core ML format...");
    convert_model_to_coreml()?;

    // 2. Runtime Inference
    println!("\n2. Running Core ML inference...");
    run_coreml_inference()?;

    // 3. Performance Optimization
    println!("\n3. Optimizing for device...");
    optimize_coreml_for_device()?;

    Ok(())
}

#[cfg(all(not(target_os = "ios"), feature = "coreml"))]
fn run_coreml_demo() -> Result<()> {
    println!("Core ML Demo (iOS)");
    println!("------------------\n");
    println!("Note: Core ML APIs only available on iOS devices\n");

    // Only run conversion demo which doesn't need iOS-specific runtime
    println!("1. Converting model to Core ML format...");
    convert_model_to_coreml()?;

    Ok(())
}

/// Convert model to Core ML format
#[cfg(feature = "coreml")]
fn convert_model_to_coreml() -> Result<()> {
    // Configure converter
    let converter_config = CoreMLConverterConfig {
        target_ios_version: "15.0".to_string(),
        optimization_level: OptimizationLevel::Aggressive,
        enable_compression: true,
        quantization: Some(CoreMLQuantizationConfig {
            weight_bits: QuantizationBits::Bit8,
            activation_bits: Some(QuantizationBits::Bit8),
            method: QuantizationMethod::Linear,
            calibration_size: 1000,
            per_channel: true,
        }),
        pruning: Some(PruningConfig {
            target_sparsity: 0.5,
            method: PruningMethod::Magnitude,
            structured: false,
            exclude_layers: vec!["output".to_string()],
        }),
        output_format: CoreMLFormat::MLPackage,
        hardware_target: HardwareTarget::NeuralEngine,
    };

    // Create converter
    let _converter = CoreMLModelConverter::new(converter_config);

    // Convert model (using placeholder paths)
    let _model_path = Path::new("model.tfm");
    let _output_path = Path::new("model.mlpackage");

    println!("Converter configuration:");
    println!("- Target iOS: 15.0");
    println!("- Optimization: Aggressive");
    println!("- Quantization: INT8");
    println!("- Pruning: 50% sparsity");
    println!("- Target: Neural Engine");

    // In production, would actually convert
    // let result = converter.convert(model_path, output_path)?;
    // println!("Conversion complete!");
    // println!("- Output: {:?}", result.output_path);
    // println!("- Size: {:.2} MB", result.model_size_mb);
    // println!("- Compression ratio: {:.2}x", result.compression_ratio);

    Ok(())
}

/// Run Core ML inference
#[cfg(all(target_os = "ios", feature = "coreml"))]
fn run_coreml_inference() -> Result<()> {
    use trustformers_mobile::coreml::mobile_config_to_coreml;

    // Create mobile config
    let mobile_config = MobileConfig::ios_optimized();

    // Convert to Core ML config
    let coreml_config = mobile_config_to_coreml(&mobile_config);

    // Create Core ML engine
    let mut engine = CoreMLEngine::new(coreml_config)?;

    // Load model (placeholder)
    let model_data = vec![0u8; 1024];
    // engine.load_model(&model_data)?;

    println!("Core ML Engine initialized");
    println!(
        "- Compute units: {:?}",
        engine.get_device_info().has_neural_engine
    );
    println!("- iOS version: {}", engine.get_device_info().ios_version);
    println!(
        "- Available memory: {} MB",
        engine.get_device_info().available_memory_mb
    );

    // Create sample input
    let mut input = HashMap::new();
    input.insert("input".to_string(), Tensor::randn(&[1, 3, 224, 224])?);

    // Run inference (placeholder)
    // let output = engine.predict(&input)?;
    // println!("Inference completed!");

    // Get statistics
    let stats = engine.get_stats();
    println!("\nPerformance Statistics:");
    println!("- Total predictions: {}", stats.total_predictions);
    println!(
        "- Avg prediction time: {:.2} ms",
        stats.avg_prediction_time_ms
    );
    println!("- Model load time: {:.2} ms", stats.model_load_time_ms);

    Ok(())
}

/// Optimize Core ML for specific device
#[cfg(all(target_os = "ios", feature = "coreml"))]
fn optimize_coreml_for_device() -> Result<()> {
    use trustformers_mobile::coreml::CoreMLConfig;

    // Device-specific configurations
    let device_configs = vec![
        ("iPhone 14 Pro", CoreMLConfig::for_device("iPhone14,2")),
        ("iPhone 12", CoreMLConfig::for_device("iPhone12,1")),
        ("iPad Pro", CoreMLConfig::for_device("iPad13,4")),
    ];

    for (device_name, config) in device_configs {
        println!("\nOptimization for {}:", device_name);
        println!("- Compute units: {:?}", config.compute_units);
        println!("- Max batch size: {}", config.max_batch_size);
        println!("- Memory handling: {:?}", config.memory_pressure_handling);
        println!("- Reduced precision: {}", config.use_reduced_precision);
    }

    Ok(())
}

/// Run NNAPI demo for Android
#[cfg(all(target_os = "android", feature = "nnapi"))]
fn run_nnapi_demo() -> Result<()> {
    println!("NNAPI Demo (Android)");
    println!("-------------------\n");

    // 1. Model Conversion
    println!("1. Converting model to NNAPI format...");
    convert_model_to_nnapi()?;

    // 2. Runtime Inference
    println!("\n2. Running NNAPI inference...");
    run_nnapi_inference()?;

    // 3. Device Optimization
    println!("\n3. Optimizing for Android devices...");
    optimize_nnapi_for_devices()?;

    Ok(())
}

#[cfg(all(not(target_os = "android"), feature = "nnapi"))]
fn run_nnapi_demo() -> Result<()> {
    println!("NNAPI Demo (Android)");
    println!("-------------------\n");
    println!("Note: NNAPI APIs only available on Android devices\n");

    // Only run conversion demo which doesn't need Android-specific runtime
    println!("1. Converting model to NNAPI format...");
    convert_model_to_nnapi()?;

    Ok(())
}

/// Convert model to NNAPI format
#[cfg(feature = "nnapi")]
fn convert_model_to_nnapi() -> Result<()> {
    // Configure converter
    let converter_config = NNAPIConverterConfig {
        target_api_level: 30, // Android 11
        target_devices: vec![
            NNAPITargetDevice::GPU,
            NNAPITargetDevice::HexagonDSP,
            NNAPITargetDevice::CPU,
        ],
        enable_partitioning: true,
        quantization: Some(NNAPIQuantizationConfig {
            scheme: NNAPIQuantizationScheme::FullInteger,
            calibration: CalibrationConfig {
                num_samples: 1000,
                method: CalibrationMethod::MinMax,
                dataset_path: None,
            },
            per_channel: true,
            symmetric: false,
            quantize_io: true,
        }),
        optimization: NNAPIOptimizationConfig {
            enable_fusion: true,
            optimize_layout: true,
            constant_folding: true,
            dead_code_elimination: true,
            device_optimizations: true,
        },
        fallback_strategy: FallbackStrategy::Partition,
        output_format: NNAPIFormat::Binary,
    };

    // Create converter
    let _converter = NNAPIModelConverter::new(converter_config);

    // Convert model (using placeholder paths)
    let _model_path = Path::new("model.tfm");
    let _output_path = Path::new("model.nnapi");

    println!("Converter configuration:");
    println!("- Target API: 30 (Android 11)");
    println!("- Target devices: GPU, Hexagon DSP, CPU");
    println!("- Quantization: Full Integer");
    println!("- Partitioning: Enabled");
    println!("- Optimizations: All enabled");

    // In production, would actually convert
    // let result = converter.convert(model_path, output_path)?;
    // println!("Conversion complete!");
    // println!("- Output: {:?}", result.output_path);
    // println!("- Size: {:.2} MB", result.model_size_mb);
    // println!("- Operations: {}", result.num_operations);
    // println!("- Partitions: {}", result.num_partitions);

    Ok(())
}

/// Run NNAPI inference
#[cfg(all(target_os = "android", feature = "nnapi"))]
fn run_nnapi_inference() -> Result<()> {
    use trustformers_mobile::nnapi::mobile_config_to_nnapi;

    // Create mobile config
    let mobile_config = MobileConfig::android_optimized();

    // Convert to NNAPI config
    let nnapi_config = mobile_config_to_nnapi(&mobile_config);

    // Create NNAPI engine
    let mut engine = NNAPIEngine::new(nnapi_config)?;

    // Load model (placeholder)
    let model_data = vec![0u8; 1024];
    // engine.load_model(&model_data)?;

    println!("NNAPI Engine initialized");
    println!(
        "- Android API level: {}",
        engine.get_device_info().android_api_level
    );
    println!(
        "- Device: {} {}",
        engine.get_device_info().manufacturer,
        engine.get_device_info().device_model
    );
    println!(
        "- Available devices: {} accelerators",
        engine.get_device_info().available_devices.len()
    );

    // Create sample input
    let mut input = HashMap::new();
    input.insert("input".to_string(), Tensor::randn(&[1, 3, 224, 224])?);

    // Run inference (placeholder)
    // let output = engine.execute(&input)?;
    // println!("Inference completed!");

    // Get statistics
    let stats = engine.get_stats();
    println!("\nPerformance Statistics:");
    println!("- Total executions: {}", stats.total_executions);
    println!(
        "- Avg execution time: {:.2} ms",
        stats.avg_execution_time_ms
    );
    println!("- Compilation time: {:.2} ms", stats.compilation_time_ms);

    Ok(())
}

/// Optimize NNAPI for different Android devices
#[cfg(all(target_os = "android", feature = "nnapi"))]
fn optimize_nnapi_for_devices() -> Result<()> {
    // Different optimization strategies
    let strategies = vec![
        ("Power Optimized", NNAPIConfig::power_optimized()),
        (
            "Performance Optimized",
            NNAPIConfig::performance_optimized(),
        ),
        ("Default", NNAPIConfig::default()),
    ];

    for (name, config) in strategies {
        println!("\n{} Configuration:", name);
        println!("- Preferred devices: {:?}", config.preferred_devices);
        println!("- Execution preference: {:?}", config.execution_preference);
        println!("- Max concurrent: {}", config.max_concurrent_executions);
        println!("- Memory mapping: {}", config.use_memory_mapping);
        println!("- Timeout: {} ms", config.operation_timeout_ms);
    }

    // Device-specific optimizations
    println!("\nDevice-Specific Optimizations:");

    let devices = vec![
        (
            "Qualcomm Snapdragon",
            vec![NNAPIDeviceType::DSP, NNAPIDeviceType::GPU],
        ),
        (
            "MediaTek Dimensity",
            vec![NNAPIDeviceType::Accelerator, NNAPIDeviceType::GPU],
        ),
        (
            "Google Tensor",
            vec![NNAPIDeviceType::NPU, NNAPIDeviceType::GPU],
        ),
        (
            "Samsung Exynos",
            vec![NNAPIDeviceType::NPU, NNAPIDeviceType::GPU],
        ),
    ];

    for (chipset, preferred_devices) in devices {
        println!("\n{}:", chipset);
        println!("- Preferred accelerators: {:?}", preferred_devices);
    }

    Ok(())
}

/// Run generic mobile demo
fn run_generic_demo() -> Result<()> {
    println!("Generic Mobile Demo");
    println!("------------------\n");

    println!("Platform-specific APIs not available.");
    println!("Using CPU-based inference.\n");

    // Create generic mobile config
    let config = MobileConfig {
        platform: MobilePlatform::Generic,
        backend: MobileBackend::CPU,
        ..Default::default()
    };

    println!("Configuration:");
    println!("- Platform: {:?}", config.platform);
    println!("- Backend: {:?}", config.backend);
    println!("- Memory optimization: {:?}", config.memory_optimization);
    println!("- Max memory: {} MB", config.max_memory_mb);

    Ok(())
}

/// Compare Core ML and NNAPI features
#[allow(dead_code)]
fn compare_platform_apis() {
    println!("\nPlatform APIs Comparison");
    println!("========================\n");

    println!("Core ML (iOS):");
    println!("- Hardware: Neural Engine, GPU, CPU");
    println!("- Formats: MLModel, MLPackage");
    println!("- Quantization: INT8, INT4, INT2, Binary");
    println!("- iOS Version: 11.0+");
    println!("- Key Features: On-device training, model encryption");

    println!("\nNNAPI (Android):");
    println!("- Hardware: NPU, DSP, GPU, CPU");
    println!("- Formats: Binary, FlatBuffer, TFLite");
    println!("- Quantization: INT8, FP16, Dynamic");
    println!("- Android Version: 8.1+ (API 27+)");
    println!("- Key Features: Multi-device support, compilation caching");

    println!("\nRecommendations:");
    println!("- Use platform APIs for best performance");
    println!("- Implement fallback for unsupported operations");
    println!("- Test on actual devices for accurate benchmarks");
    println!("- Consider model partitioning for complex models");
}

// Mock implementations for demo (none needed currently)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_detection() {
        let platform = detect_platform();
        assert!(matches!(
            platform,
            MobilePlatform::Ios | MobilePlatform::Android | MobilePlatform::Generic
        ));
    }
}
