# Platform-Specific Neural Network APIs

This guide covers Core ML (iOS) and NNAPI (Android) integration in TrustformeRS, enabling hardware-accelerated inference on mobile devices.

## Table of Contents

1. [Overview](#overview)
2. [Core ML (iOS)](#core-ml-ios)
3. [NNAPI (Android)](#nnapi-android)
4. [Model Conversion](#model-conversion)
5. [Performance Optimization](#performance-optimization)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## Overview

TrustformeRS provides comprehensive support for platform-specific neural network APIs:

- **Core ML**: Apple's framework for on-device inference on iOS, macOS, tvOS, and watchOS
- **NNAPI**: Android Neural Networks API for hardware-accelerated inference on Android devices

### Key Benefits

- **Hardware Acceleration**: Leverage Neural Engine, GPU, DSP, and NPU
- **Power Efficiency**: Optimized for mobile battery life
- **Low Latency**: On-device inference without network round trips
- **Privacy**: Data stays on device

## Core ML (iOS)

### Overview

Core ML provides optimized inference on Apple devices using:
- Neural Engine (A12 Bionic and later)
- GPU (Metal Performance Shaders)
- CPU (Accelerate and BNNS)

### Model Conversion

Convert TrustformeRS models to Core ML format:

```rust
use trustformers_mobile::coreml_converter::{
    CoreMLModelConverter, CoreMLConverterConfig, 
    OptimizationLevel, CoreMLFormat, HardwareTarget
};

// Configure converter
let config = CoreMLConverterConfig {
    target_ios_version: "15.0".to_string(),
    optimization_level: OptimizationLevel::Aggressive,
    enable_compression: true,
    output_format: CoreMLFormat::MLPackage,
    hardware_target: HardwareTarget::NeuralEngine,
    ..Default::default()
};

// Create converter
let converter = CoreMLModelConverter::new(config);

// Convert model
let result = converter.convert(
    Path::new("model.tfm"),
    Path::new("output/model.mlpackage")
)?;

println!("Model converted: {} MB", result.model_size_mb);
```

### Quantization Options

Core ML supports various quantization schemes:

```rust
use trustformers_mobile::coreml_converter::{
    CoreMLQuantizationConfig, QuantizationBits, QuantizationMethod
};

let quantization = CoreMLQuantizationConfig {
    weight_bits: QuantizationBits::Bit8,        // INT8 weights
    activation_bits: Some(QuantizationBits::Bit8), // INT8 activations
    method: QuantizationMethod::Linear,
    calibration_size: 1000,
    per_channel: true,
};
```

Supported bit widths:
- **Binary (1-bit)**: Maximum compression, suitable for binary networks
- **INT2**: 16x compression
- **INT4**: 8x compression  
- **INT8**: 4x compression, good accuracy
- **INT16**: 2x compression

### Runtime Inference

Run inference using Core ML:

```rust
use trustformers_mobile::coreml::{
    CoreMLEngine, CoreMLConfig, CoreMLComputeUnits
};

// Create configuration
let config = CoreMLConfig {
    compute_units: CoreMLComputeUnits::All,
    enable_batch_prediction: true,
    max_batch_size: 4,
    use_reduced_precision: true,
    ..Default::default()
};

// Initialize engine
let mut engine = CoreMLEngine::new(config)?;

// Load model
engine.load_model(&model_data)?;

// Run inference
let output = engine.predict(&input_tensors)?;
```

### Device-Specific Optimization

Optimize for specific iOS devices:

```rust
// iPhone 14 Pro (A16 Bionic)
let config = CoreMLConfig::for_device("iPhone14,2");

// iPad Pro (M2)
let config = CoreMLConfig::for_device("iPad14,3");

// Automatic optimization
engine.optimize_for_device()?;
```

## NNAPI (Android)

### Overview

NNAPI provides hardware acceleration on Android devices through:
- Qualcomm Hexagon DSP
- MediaTek APU
- Google Edge TPU
- Samsung NPU
- GPU (Vulkan/OpenGL)
- CPU fallback

### Model Conversion

Convert TrustformeRS models to NNAPI format:

```rust
use trustformers_mobile::nnapi_converter::{
    NNAPIModelConverter, NNAPIConverterConfig,
    NNAPITargetDevice, NNAPIFormat
};

// Configure converter
let config = NNAPIConverterConfig {
    target_api_level: 30,  // Android 11
    target_devices: vec![
        NNAPITargetDevice::GPU,
        NNAPITargetDevice::HexagonDSP,
        NNAPITargetDevice::CPU,
    ],
    enable_partitioning: true,
    output_format: NNAPIFormat::Binary,
    ..Default::default()
};

// Create converter
let converter = NNAPIModelConverter::new(config);

// Convert model
let result = converter.convert(
    Path::new("model.tfm"),
    Path::new("output/model.nnapi")
)?;
```

### Quantization Options

NNAPI quantization schemes:

```rust
use trustformers_mobile::nnapi_converter::{
    NNAPIQuantizationConfig, NNAPIQuantizationScheme,
    CalibrationConfig, CalibrationMethod
};

let quantization = NNAPIQuantizationConfig {
    scheme: NNAPIQuantizationScheme::FullInteger,
    calibration: CalibrationConfig {
        num_samples: 1000,
        method: CalibrationMethod::MinMax,
        dataset_path: Some(Path::new("calibration_data")),
    },
    per_channel: true,
    symmetric: false,
    quantize_io: true,
};
```

Supported schemes:
- **Dynamic**: Quantize at runtime
- **Full Integer**: INT8 throughout
- **Integer with Float**: INT8 with FP32 fallback
- **Float16**: Half precision

### Runtime Inference

Run inference using NNAPI:

```rust
use trustformers_mobile::nnapi::{
    NNAPIEngine, NNAPIConfig, NNAPIDeviceType,
    NNAPIExecutionPreference
};

// Create configuration
let config = NNAPIConfig {
    preferred_devices: vec![
        NNAPIDeviceType::NPU,
        NNAPIDeviceType::GPU,
        NNAPIDeviceType::CPU,
    ],
    execution_preference: NNAPIExecutionPreference::SustainedSpeed,
    enable_compilation_caching: true,
    ..Default::default()
};

// Initialize engine
let mut engine = NNAPIEngine::new(config)?;

// Initialize with Android context
engine.init_with_jvm(jvm)?;

// Load and compile model
engine.load_model(&model_data)?;

// Run inference
let output = engine.execute(&input_tensors)?;
```

### Device-Specific Optimization

Optimize for different Android devices:

```rust
// Power-efficient configuration
let config = NNAPIConfig::power_optimized();

// Performance-optimized configuration  
let config = NNAPIConfig::performance_optimized();

// Auto-optimize for current device
engine.optimize_for_device()?;
```

## Model Conversion

### Conversion Pipeline

1. **Load TrustformeRS Model**
   ```rust
   let model = load_trustformers_model("model.tfm")?;
   ```

2. **Validate Compatibility**
   - Check supported operations
   - Verify tensor shapes
   - Validate data types

3. **Apply Optimizations**
   - Operator fusion
   - Constant folding
   - Layout optimization
   - Dead code elimination

4. **Quantization** (Optional)
   - Calibrate with representative data
   - Apply quantization scheme
   - Validate accuracy

5. **Export**
   - Core ML: `.mlmodel`, `.mlpackage`
   - NNAPI: `.nnapi`, `.tflite`

### Example: Full Conversion Pipeline

```rust
// Core ML conversion with all optimizations
let coreml_config = CoreMLConverterConfig {
    target_ios_version: "16.0".to_string(),
    optimization_level: OptimizationLevel::Maximum,
    enable_compression: true,
    quantization: Some(CoreMLQuantizationConfig {
        weight_bits: QuantizationBits::Bit4,
        activation_bits: Some(QuantizationBits::Bit8),
        method: QuantizationMethod::KMeans,
        calibration_size: 2000,
        per_channel: true,
    }),
    pruning: Some(PruningConfig {
        target_sparsity: 0.7,  // 70% sparse
        method: PruningMethod::Magnitude,
        structured: true,
        exclude_layers: vec!["output".to_string()],
    }),
    output_format: CoreMLFormat::MLPackage,
    hardware_target: HardwareTarget::All,
};

// NNAPI conversion with partitioning
let nnapi_config = NNAPIConverterConfig {
    target_api_level: 31,  // Android 12
    target_devices: vec![
        NNAPITargetDevice::HexagonDSP,
        NNAPITargetDevice::GPU,
        NNAPITargetDevice::CPU,
    ],
    enable_partitioning: true,
    quantization: Some(NNAPIQuantizationConfig {
        scheme: NNAPIQuantizationScheme::IntegerWithFloat,
        // ... configuration
    }),
    fallback_strategy: FallbackStrategy::Partition,
    output_format: NNAPIFormat::TFLite,
    ..Default::default()
};
```

## Performance Optimization

### Core ML Optimization

1. **Neural Engine Optimization**
   ```rust
   config.hardware_target = HardwareTarget::NeuralEngine;
   ```

2. **Batch Processing**
   ```rust
   config.enable_batch_prediction = true;
   config.max_batch_size = 8;
   ```

3. **Memory Optimization**
   ```rust
   config.memory_pressure_handling = CoreMLMemoryHandling::Aggressive;
   ```

### NNAPI Optimization

1. **Multi-Device Execution**
   ```rust
   config.preferred_devices = vec![
       NNAPIDeviceType::NPU,
       NNAPIDeviceType::DSP,
       NNAPIDeviceType::GPU,
   ];
   ```

2. **Compilation Caching**
   ```rust
   config.enable_compilation_caching = true;
   ```

3. **Execution Preference**
   ```rust
   // For real-time apps
   config.execution_preference = NNAPIExecutionPreference::FastSingleAnswer;
   
   // For batch processing
   config.execution_preference = NNAPIExecutionPreference::SustainedSpeed;
   
   // For battery efficiency
   config.execution_preference = NNAPIExecutionPreference::LowPower;
   ```

## Best Practices

### 1. Platform Detection

```rust
let config = match detect_platform() {
    MobilePlatform::iOS => MobileConfig::ios_optimized(),
    MobilePlatform::Android => MobileConfig::android_optimized(),
    MobilePlatform::Generic => MobileConfig::default(),
};
```

### 2. Fallback Strategy

```rust
// Try platform API first, fallback to CPU
match platform {
    MobilePlatform::iOS => {
        match run_coreml_inference() {
            Ok(result) => result,
            Err(_) => run_cpu_inference()?,
        }
    }
    MobilePlatform::Android => {
        match run_nnapi_inference() {
            Ok(result) => result,
            Err(_) => run_cpu_inference()?,
        }
    }
    _ => run_cpu_inference()?,
}
```

### 3. Model Validation

Always validate converted models:

```rust
// Validate Core ML model
let validation_result = coreml_converter.validate_model(&model)?;
assert!(validation_result.is_valid);

// Validate NNAPI model
let compatibility = nnapi_converter.check_compatibility(&model)?;
assert!(compatibility.is_compatible);
```

### 4. Performance Monitoring

```rust
// Monitor Core ML performance
let stats = coreml_engine.get_stats();
if stats.avg_prediction_time_ms > 50.0 {
    // Adjust configuration
    config.use_reduced_precision = true;
}

// Monitor NNAPI performance
let stats = nnapi_engine.get_stats();
if stats.avg_execution_time_ms > 100.0 {
    // Switch to lower precision
    config.allow_relaxed_computation = true;
}
```

## Troubleshooting

### Core ML Issues

1. **Model Load Failure**
   - Check iOS version compatibility
   - Verify model format
   - Ensure sufficient memory

2. **Poor Performance**
   - Enable Neural Engine
   - Reduce precision
   - Decrease batch size

3. **Unsupported Operations**
   - Use CPU fallback
   - Split model into subgraphs
   - Implement custom operators

### NNAPI Issues

1. **Compilation Failure**
   - Check Android API level
   - Verify device support
   - Enable CPU fallback

2. **Execution Timeout**
   - Increase timeout value
   - Reduce model complexity
   - Use partitioning

3. **Device Not Available**
   - Check device capabilities
   - Use generic accelerator
   - Fallback to CPU

### Common Solutions

```rust
// Robust initialization with fallback
fn initialize_platform_engine() -> Result<Box<dyn InferenceEngine>> {
    #[cfg(target_os = "ios")]
    {
        if let Ok(engine) = CoreMLEngine::new(CoreMLConfig::default()) {
            return Ok(Box::new(engine));
        }
    }
    
    #[cfg(target_os = "android")]
    {
        if let Ok(engine) = NNAPIEngine::new(NNAPIConfig::default()) {
            return Ok(Box::new(engine));
        }
    }
    
    // Fallback to CPU engine
    Ok(Box::new(CPUEngine::new()?))
}
```

## Platform Comparison

| Feature | Core ML | NNAPI |
|---------|---------|--------|
| Minimum OS | iOS 11.0 | Android 8.1 (API 27) |
| Hardware | Neural Engine, GPU, CPU | NPU, DSP, GPU, CPU |
| Quantization | INT8/4/2/1-bit | INT8, FP16, Dynamic |
| Batch Support | Native | Via iteration |
| Model Format | MLModel, MLPackage | Binary, FlatBuffer |
| Compilation Cache | Automatic | Manual |
| On-device Training | Yes | No |
| Custom Operators | Limited | Yes |

## Examples

See the [examples](examples/) directory for complete working examples:

- [`platform_apis_demo.rs`](examples/platform_apis_demo.rs) - Platform API demonstration
- [`coreml_optimization.rs`](examples/coreml_optimization.rs) - Core ML optimization techniques
- [`nnapi_devices.rs`](examples/nnapi_devices.rs) - NNAPI multi-device execution

## Conclusion

Platform-specific APIs provide significant performance benefits:

- **Core ML**: Up to 15x faster on Neural Engine vs CPU
- **NNAPI**: Up to 10x faster on NPU/DSP vs CPU

Always test on real devices and implement appropriate fallback strategies for maximum compatibility.