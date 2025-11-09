# TrustformersKit - iOS Framework for TrustformeRS

TrustformersKit is a comprehensive iOS framework that brings the power of TrustformeRS to iOS applications. It provides optimized mobile inference with support for Core ML, Metal GPU acceleration, and Neural Engine.

## Features

- 🚀 **High Performance**: Optimized for iOS devices with Metal and Neural Engine support
- 🧠 **Core ML Integration**: Seamless integration with Apple's Core ML framework
- 📱 **Device Optimization**: Automatic configuration based on device capabilities
- 🔋 **Battery Efficient**: Power-aware inference with thermal throttling
- 📊 **Performance Monitoring**: Built-in profiling and statistics
- 🎯 **Type Safe**: Full Swift API with comprehensive error handling

## Requirements

- iOS 11.0+ (iOS 13.0+ for SwiftUI examples)
- Xcode 12.0+
- Swift 5.3+
- ARM64 architecture (iPhone 5S and later)

## Installation

### Swift Package Manager

Add TrustformersKit to your project using Swift Package Manager:

1. In Xcode, select File > Add Packages
2. Enter the repository URL: `https://github.com/trustformers/trustformers-mobile`
3. Select the version and add to your target

### Manual Installation

1. Build the framework:
```bash
cd trustformers-mobile
./build-ios.sh
```

2. Drag `TrustformersKit.xcframework` into your Xcode project
3. Ensure "Embed & Sign" is selected in the frameworks settings

### CocoaPods

```ruby
pod 'TrustformersKit', '~> 1.0'
```

## Quick Start

### Basic Inference

```swift
import TrustformersKit

// Initialize the inference engine
let config = TFKModelConfig.optimizedConfig()
let engine = TFKInferenceEngine(config: config)

// Load a model
let model = try engine.loadModel(
    bundleResource: "model",
    type: "tfm",
    config: config
)

// Create input tensor
let input = TFKTensor(floats: inputData, shape: [1, 768])

// Perform inference
let result = engine.performInference(model, input: input)

if result.success {
    print("Output: \(result.output.floatData())")
    print("Inference time: \(result.inferenceTimeMs) ms")
}
```

### Text Classification

```swift
// Configure for text processing
let config = TFKModelConfig.textGenerationConfig()
let model = try engine.loadModel(at: modelPath, config: config)

// Process text
let tokens = tokenize(text)
let input = TFKTensor(floats: tokens, shape: [1, tokens.count])

// Get predictions
let result = engine.performInference(model, input: input)
let predictions = result.topKPredictions(k: 3, labels: labels)

for pred in predictions {
    print("\(pred.label): \(pred.score * 100)%")
}
```

### Image Classification

```swift
// Configure for image processing
let config = TFKModelConfig.imageClassificationConfig()
config.enableBatching = true
config.maxBatchSize = 4

let model = try engine.loadModel(at: modelPath, config: config)

// Process image
let input = preprocessImage(image, size: CGSize(width: 224, height: 224))

// Classify
let result = engine.performInference(model, input: input)
let classification = result.getClassification(labels: imageNetLabels)

print("Predicted: \(classification.label) (\(classification.score * 100)%)")
```

## Configuration

### Model Backends

TrustformersKit supports multiple inference backends:

```swift
// CPU-only inference
config.backend = .cpu

// Core ML (recommended for iOS)
config.backend = .coreML

// Metal GPU acceleration
config.backend = .gpu

// Custom backend
config.backend = .custom
```

### Memory Optimization

Control memory usage based on your app's needs:

```swift
// Minimal memory usage (slowest)
config.memoryOptimization = .maximum

// Balanced (default)
config.memoryOptimization = .balanced

// Maximum performance (uses more memory)
config.memoryOptimization = .minimal
```

### Quantization

Enable quantization for smaller models:

```swift
// INT8 quantization
config.enableQuantization = true
config.quantizationScheme = .int8

// INT4 for extreme compression
config.quantizationScheme = .int4

// FP16 for balanced compression
config.quantizationScheme = .fp16
```

## Device Detection

TrustformersKit automatically detects device capabilities:

```swift
let deviceInfo = TFKDeviceInfo.currentDevice()

// Check capabilities
if deviceInfo.hasNeuralEngine {
    print("Neural Engine available!")
}

// Get recommended configuration
let config = deviceInfo.recommendedConfig()

// Check performance tier
switch deviceInfo.performanceTier() {
case .flagship:
    // Enable all features
case .high:
    // Good performance
case .medium:
    // Balanced settings
case .low:
    // Conservative settings
}
```

## Performance Monitoring

Track inference performance:

```swift
// Get performance statistics
let stats = engine.performanceStats
print(stats.performanceSummary())

// Monitor in real-time
let snapshot = stats.takeSnapshot()
print("Average inference: \(snapshot.averageInferenceTime * 1000) ms")
print("Memory usage: \(snapshot.currentMemoryUsageMB) MB")

// Export statistics
let data = try stats.exportJSON()
```

## Memory Management

Handle memory pressure gracefully:

```swift
// Set memory pressure handler
engine.setMemoryPressureHandler { level in
    switch level {
    case .maximum:
        // Critical memory pressure
        model.unload()
    case .balanced:
        // Reduce batch size
    case .minimal:
        // Normal operation
    }
}

// Enable thermal throttling
engine.setThermalThrottling(true)

// Optimize for battery life
engine.optimizeForBatteryLife(true)
```

## Core ML Integration

Use Core ML models directly:

```swift
// Load Core ML model
let mlModel = try MLModel(contentsOf: modelURL)
let model = TFKModel(coreMLModel: mlModel, config: config)

// Convert between formats
let mlArray = try MLMultiArray(shape: [1, 3, 224, 224], dataType: .float32)
let tensor = try TFKTensor(mlMultiArray: mlArray)

// Perform inference
let result = engine.performInference(model, input: tensor)
let outputArray = try result.output.toMLMultiArray()
```

## Batch Processing

Process multiple inputs efficiently:

```swift
// Enable batching
config.enableBatching = true
config.maxBatchSize = 8

// Process batch
let inputs = images.map { preprocessImage($0) }
let results = engine.performBatchInference(model, inputs: inputs)

// Aggregate results
let aggregated = TFKInferenceResult.aggregate(results)
print("Success rate: \(aggregated.successRate * 100)%")
print("Average time: \(aggregated.averageInferenceTime * 1000) ms")
```

## Logging

Configure logging for debugging:

```swift
// Set log level
TFKLogger.setLogLevel(.debug)

// Enable file logging
TFKLogger.enableFileLogging()

// Custom log handler
TFKLogger.setLogHandler { level, message in
    print("[\(level)] \(message)")
}

// Log performance
TFKLogger.logPerformance(
    operation: "Image Classification",
    time: 0.045,
    memory: 128
)
```

## Error Handling

Comprehensive error handling:

```swift
do {
    let model = try engine.loadModel(at: path, config: config)
    let result = engine.performInference(model, input: input)
    
    if !result.success {
        if let error = result.error {
            print("Inference failed: \(error.localizedDescription)")
            
            // Check specific errors
            if case TFKError.outOfMemory = error {
                // Handle memory error
            }
        }
    }
} catch TFKError.modelNotFound(let path) {
    print("Model not found at: \(path)")
} catch TFKError.unsupportedBackend(let backend) {
    print("Backend not supported: \(backend)")
} catch {
    print("Unexpected error: \(error)")
}
```

## Examples

See the `Example` directory for complete examples:

- `TrustformersKitExample.swift` - Comprehensive examples
- SwiftUI demo app
- Objective-C compatibility examples

## Building from Source

1. Install Rust and cargo-lipo:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install cargo-lipo
```

2. Add iOS targets:
```bash
rustup target add aarch64-apple-ios
rustup target add x86_64-apple-ios
rustup target add aarch64-apple-ios-sim
```

3. Build the framework:
```bash
./build-ios.sh
```

## Performance Tips

1. **Use Core ML Backend**: When possible, use the Core ML backend for best performance
2. **Enable FP16**: Use FP16 precision for 2x memory savings with minimal accuracy loss
3. **Batch Processing**: Process multiple inputs together when possible
4. **Preload Models**: Load models during app startup to avoid delays
5. **Monitor Thermals**: Enable thermal throttling to prevent device overheating
6. **Profile Performance**: Use the built-in profiler to identify bottlenecks

## Troubleshooting

### Common Issues

**Model fails to load**
- Ensure the model file exists in your app bundle
- Check that the model format is supported (.tfm, .onnx, .mlmodel)
- Verify device has enough memory

**Poor performance**
- Check thermal state with `deviceInfo.thermalState`
- Ensure you're using the optimal backend for your device
- Consider enabling quantization

**Memory warnings**
- Implement memory pressure handler
- Use more aggressive quantization (INT4)
- Reduce batch size or disable batching

## License

TrustformersKit is released under the MIT License. See LICENSE for details.

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.