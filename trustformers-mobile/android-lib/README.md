# TrustformeRS Android Library

TrustformeRS Android is a comprehensive library that brings the power of TrustformeRS to Android applications. It provides optimized mobile inference with support for NNAPI, CPU, and GPU backends.

## Features

- 🚀 **High Performance**: Optimized for Android devices with NNAPI and GPU support
- 🤖 **NNAPI Integration**: Leverages Android Neural Networks API for hardware acceleration
- 📱 **Device Optimization**: Automatic configuration based on device capabilities
- 🔋 **Battery Efficient**: Power-aware inference with thermal management
- 📊 **Performance Monitoring**: Built-in profiling and statistics
- 🎯 **Type Safe**: Full Java/Kotlin API with comprehensive error handling
- ⚡ **Coroutine Support**: First-class Kotlin coroutines integration

## Requirements

- Android 5.0 (API level 21) or higher
- Android NDK r21 or higher (for building)
- Android Studio Arctic Fox or newer

## Installation

### Gradle (Recommended)

Add the TrustformeRS Android dependency to your app's `build.gradle`:

```gradle
dependencies {
    implementation 'com.trustformers:trustformers-android:1.0.0'
}
```

### Manual Installation

1. Build the library:
```bash
cd trustformers-mobile
./build-android.sh
```

2. Copy the AAR file from `android-lib/build/outputs/aar/` to your project's `libs` directory

3. Add to your `build.gradle`:
```gradle
dependencies {
    implementation files('libs/trustformers-android-release.aar')
}
```

## Quick Start

### Java Example

```java
import com.trustformers.TrustformersEngine;
import com.trustformers.TrustformersEngine.EngineConfig;
import com.trustformers.Model;
import com.trustformers.Tensor;

// Initialize the engine
EngineConfig config = EngineConfig.createOptimized(context);
TrustformersEngine engine = new TrustformersEngine(context, config);

// Load a model
Model model = engine.loadModel("/path/to/model.tfm");

// Create input tensor
float[] inputData = new float[768];
Tensor input = new Tensor(inputData, new int[]{1, 768});

// Perform inference
Tensor output = engine.inference(model, input);

// Process results
float[] results = output.getData();

// Clean up
engine.close();
```

### Kotlin Example

```kotlin
import com.trustformers.*
import kotlinx.coroutines.*

// Using Kotlin DSL
val engine = trustformersKt(context) {
    backend = EngineConfig.Backend.NNAPI
    useFP16 = true
    enableBatching = true
}

lifecycleScope.launch {
    // Load model asynchronously
    val model = engine.loadModel("/path/to/model.tfm")
    
    // Create input
    val input = tensorOf(1.0f, 2.0f, 3.0f, shape = intArrayOf(1, 3))
    
    // Perform inference
    val output = engine.inference(model, input)
    
    // Get results
    val predictions = output.softmax().topK(5)
}

// Automatic cleanup with use
engine.use { engine ->
    // Use the engine
}
```

## Configuration

### Backend Selection

TrustformeRS Android supports multiple inference backends:

```java
// CPU backend (default)
config.setBackend(EngineConfig.Backend.CPU);

// GPU backend (requires OpenGL ES 3.0+)
config.setBackend(EngineConfig.Backend.GPU);

// NNAPI backend (requires Android 8.1+)
config.setBackend(EngineConfig.Backend.NNAPI);

// Automatic selection based on device
config.setBackend(EngineConfig.Backend.AUTO);
```

### Memory Optimization

Control memory usage based on your app's needs:

```java
// Minimal memory usage (slowest)
config.setMemoryOptimization(EngineConfig.MemoryOptimization.MAXIMUM);

// Balanced (default)
config.setMemoryOptimization(EngineConfig.MemoryOptimization.BALANCED);

// Maximum performance (uses more memory)
config.setMemoryOptimization(EngineConfig.MemoryOptimization.MINIMAL);

// Set memory limit
config.setMaxMemoryMB(512);
```

### Quantization

Enable quantization for smaller models and faster inference:

```java
// INT8 quantization
config.setQuantizationScheme(EngineConfig.QuantizationScheme.INT8);

// INT4 for extreme compression
config.setQuantizationScheme(EngineConfig.QuantizationScheme.INT4);

// FP16 for balanced compression
config.setQuantizationScheme(EngineConfig.QuantizationScheme.FP16);

// Dynamic quantization
config.setQuantizationScheme(EngineConfig.QuantizationScheme.DYNAMIC);
```

## Device Detection

TrustformeRS automatically detects device capabilities:

```java
DeviceInfo deviceInfo = engine.getDeviceInfo();

// Check capabilities
if (deviceInfo.hasNNAPI()) {
    // NNAPI is available
}

// Get performance class
DeviceInfo.PerformanceClass perfClass = deviceInfo.getPerformanceClass();
switch (perfClass) {
    case HIGH_END:
        // Enable all features
        break;
    case MID_RANGE:
        // Balanced settings
        break;
    case ENTRY_LEVEL:
        // Conservative settings
        break;
}

// Monitor thermal status
DeviceInfo.ThermalStatus thermal = deviceInfo.getThermalStatus();
if (thermal == DeviceInfo.ThermalStatus.CRITICAL) {
    // Reduce workload
}
```

## Performance Monitoring

Track inference performance:

```java
// Enable profiling
config.setEnableProfiling(true);

// Get performance statistics
PerformanceStats stats = engine.getPerformanceStats();
System.out.println(stats.getSummary());

// Monitor specific metrics
double avgTime = stats.getAvgInferenceTimeMs();
double throughput = stats.getInferencesPerSecond();
long p95Time = stats.getP95InferenceTimeMs();
```

## Loading Models

### From File System

```java
Model model = engine.loadModel("/sdcard/model.tfm");
```

### From Assets

```java
Model model = engine.loadModelFromAssets("models/text_classifier.tfm");
```

### Model Validation

```java
// Validate input shape
if (model.validateInputShape(inputTensor)) {
    // Shape is compatible
}

// Get expected shapes
int[] inputShape = model.getExpectedInputShape();
int[] outputShape = model.getExpectedOutputShape();

// Get model info
Model.ModelInfo info = model.getModelInfo();
System.out.println("Model: " + info.getName());
System.out.println("Version: " + info.getVersion());
System.out.println("Size: " + info.getModelSizeBytes());
```

## Tensor Operations

### Creating Tensors

```java
// From array
float[] data = {1.0f, 2.0f, 3.0f, 4.0f};
Tensor tensor = new Tensor(data, new int[]{2, 2});

// Zeros
Tensor zeros = Tensor.zeros(new int[]{10, 10});

// Ones
Tensor ones = Tensor.ones(new int[]{5, 5});

// Constant
Tensor constant = Tensor.constant(new int[]{3, 3}, 0.5f);

// From 2D array
float[][] matrix = {{1, 2}, {3, 4}};
Tensor tensor2d = Tensor.fromArray2D(matrix);
```

### Tensor Operations

```java
// Reshape
Tensor reshaped = tensor.reshape(new int[]{4, 1});

// Flatten
Tensor flattened = tensor.flatten();

// Softmax
Tensor probs = output.softmax();

// Argmax
int[] maxIndices = output.argmax();

// Top-K
Tensor.TopKResult topK = output.topK(5);
float[] topValues = topK.getValues();
int[] topIndices = topK.getIndices();
```

## Batch Processing

Process multiple inputs efficiently:

```java
// Enable batching
config.setEnableBatching(true);
config.setMaxBatchSize(8);

// Create batch input
List<Tensor> inputs = Arrays.asList(
    tensor1, tensor2, tensor3, tensor4
);

// Batch inference
List<Tensor> outputs = engine.batchInference(model, inputs);
```

## Error Handling

Comprehensive error handling:

```java
try {
    Model model = engine.loadModel(modelPath);
    Tensor result = engine.inference(model, input);
} catch (IOException e) {
    // Handle loading errors
    Log.e(TAG, "Failed to load model", e);
} catch (RuntimeException e) {
    // Handle inference errors
    Log.e(TAG, "Inference failed", e);
}
```

## Memory Management

Handle memory constraints:

```java
// Monitor memory pressure
float memoryPressure = deviceInfo.getMemoryPressure();
if (memoryPressure > 0.8f) {
    // High memory pressure
    config.setMemoryOptimization(EngineConfig.MemoryOptimization.MAXIMUM);
}

// Get available memory
long availableMemoryMB = deviceInfo.getAvailableMemoryMB();

// Clean up resources
engine.close(); // Always close when done
```

## Kotlin Coroutines

First-class coroutines support:

```kotlin
// Async model loading
val model = withContext(Dispatchers.IO) {
    engine.loadModel(modelPath)
}

// Parallel inference
val results = coroutineScope {
    inputs.map { input ->
        async {
            engine.inference(model, input)
        }
    }.awaitAll()
}

// Flow-based streaming
model.inferenceFlow(inputFlow)
    .collect { output ->
        // Process each output
    }
```

## Advanced Features

### Custom Configuration

```java
// Device-specific optimization
config.configureForDevice(deviceInfo);

// Thread configuration
config.setNumThreads(Runtime.getRuntime().availableProcessors() / 2);

// Custom backend settings
if (deviceInfo.getGPUInfo().isHighPerformance()) {
    config.setBackend(EngineConfig.Backend.GPU);
}
```

### Model Metadata

```java
// Set custom metadata
model.setMetadata("description", "Text classification model");
model.setMetadata("labels", Arrays.asList("positive", "negative", "neutral"));

// Retrieve metadata
Map<String, Object> metadata = model.getMetadata();
```

## ProGuard Configuration

If using ProGuard/R8, add these rules to your `proguard-rules.pro`:

```proguard
-keep class com.trustformers.** { *; }
-keepclassmembers class * {
    native <methods>;
}
```

## Troubleshooting

### Common Issues

**UnsatisfiedLinkError**
- Ensure the native library is properly included
- Check that the correct ABI is supported
- Verify Android NDK is properly configured

**Model loading fails**
- Check file permissions
- Verify model format compatibility
- Ensure sufficient storage space

**Poor performance**
- Check thermal throttling status
- Verify optimal backend selection
- Consider enabling quantization

**Out of memory**
- Enable memory optimization
- Reduce batch size
- Use quantization

## Building from Source

1. Install prerequisites:
   - Android Studio
   - Android NDK r21+
   - Rust toolchain
   - cargo-ndk

2. Clone the repository:
```bash
git clone https://github.com/trustformers/trustformers-mobile
cd trustformers-mobile
```

3. Build the library:
```bash
./build-android.sh
```

4. Find the AAR in `android-lib/build/outputs/aar/`

## Performance Tips

1. **Use NNAPI**: When available, NNAPI provides hardware acceleration
2. **Enable FP16**: Reduces memory usage with minimal accuracy loss
3. **Batch Processing**: Process multiple inputs together
4. **Quantization**: Use INT8 for 4x model size reduction
5. **Thread Tuning**: Adjust thread count based on device
6. **Memory Monitoring**: React to memory pressure events
7. **Thermal Awareness**: Reduce workload during thermal events

## License

TrustformeRS Android is released under the MIT License. See LICENSE for details.

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Support

- GitHub Issues: https://github.com/trustformers/trustformers-mobile/issues
- Documentation: https://trustformers.dev/android
- Community: https://discord.gg/trustformers