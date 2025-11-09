# TrustformeRS Mobile Unity Package

High-performance mobile machine learning inference for Unity applications with comprehensive cross-platform support.

## Features

- ✅ **Cross-Platform Support**: iOS, Android, and standalone builds
- ✅ **High Performance**: Native acceleration with Core ML, NNAPI, and GPU backends
- ✅ **AR Foundation Integration**: Real-time AR inference with object detection, pose estimation, and plane classification
- ✅ **IL2CPP Compatibility**: Full support for Unity's IL2CPP compilation backend
- ✅ **Performance Optimization**: Adaptive quality, thermal throttling, and battery optimization
- ✅ **Memory Management**: Advanced memory pooling and automatic cleanup
- ✅ **Quantization Support**: INT4, INT8, FP16, and dynamic quantization
- ✅ **Async Operations**: Non-blocking inference with Unity's async/await pattern

## Installation

### Package Manager (Recommended)

1. Open Unity Package Manager
2. Click "+" → "Add package from git URL"
3. Enter: `https://github.com/your-org/trustformers-mobile.git#unity-package`

### Manual Installation

1. Download the package from releases
2. Import the `.unitypackage` file into your project
3. Ensure AR Foundation is installed if using AR features

## Quick Start

### Basic Inference

```csharp
using TrustformersMobile;

public class BasicInferenceExample : MonoBehaviour
{
    public TrustformersEngine engine;
    
    async void Start()
    {
        // Engine initializes automatically
        await engine.LoadModelAsync("path/to/your/model");
        
        // Perform inference
        float[] input = { /* your input data */ };
        float[] result = await engine.InferenceAsync(input);
        
        Debug.Log($"Inference result: {string.Join(", ", result)}");
    }
}
```

### AR Integration

```csharp
using TrustformersMobile.AR;

public class ARInferenceExample : MonoBehaviour
{
    public TrustformersARManager arManager;
    
    void Start()
    {
        arManager.OnObjectsDetected += OnObjectsDetected;
        arManager.config.enableObjectDetection = true;
        arManager.objectDetectionModelPath = "path/to/detection/model";
    }
    
    void OnObjectsDetected(ObjectDetectionResult[] objects)
    {
        foreach (var obj in objects)
        {
            Debug.Log($"Detected: {obj.className} ({obj.confidence:F2})");
        }
    }
}
```

### Performance Optimization

```csharp
using TrustformersMobile.Performance;

public class PerformanceExample : MonoBehaviour
{
    public TrustformersPerformanceOptimizer optimizer;
    
    void Start()
    {
        optimizer.config.targetFPS = 60;
        optimizer.config.enableAdaptiveQuality = true;
        optimizer.config.enableThermalThrottling = true;
        
        optimizer.OnMetricsUpdated += OnMetricsUpdated;
    }
    
    void OnMetricsUpdated(PerformanceMetrics metrics)
    {
        if (metrics.currentFPS < 30)
        {
            optimizer.ForceApplyOptimization(OptimizationType.ReduceInferenceFrequency);
        }
    }
}
```

## Configuration

### Engine Configuration

```csharp
var config = new TrustformersEngine.EngineConfig
{
    platform = MobilePlatform.Auto,        // Auto-detect platform
    backend = MobileBackend.Auto,          // Auto-select optimal backend
    memoryOptimization = MemoryOptimization.Balanced,
    maxMemoryMB = 512,
    useFP16 = true,
    quantization = new QuantizationConfig
    {
        enabled = true,
        scheme = QuantizationScheme.Int8,
        dynamic = true
    },
    performance = new PerformanceConfig
    {
        adaptivePerformance = true,
        targetFPS = 60.0f,
        thermalThrottling = true,
        batteryOptimization = true
    }
};
```

### AR Configuration

```csharp
var arConfig = new TrustformersARManager.ARInferenceConfig
{
    inferenceInterval = 0.1f,              // 10 FPS inference
    enableObjectDetection = true,
    enablePoseEstimation = false,
    enablePlaneClassification = true,
    maxConcurrentInferences = 1,
    adaptiveQuality = true,
    inputResolution = new Vector2Int(640, 480)
};
```

## Platform-Specific Setup

### iOS Setup

1. **Xcode Project Settings**:
   - Enable Metal API
   - Add Core ML framework
   - Set minimum iOS version to 12.0+

2. **Build Settings**:
   ```csharp
   PlayerSettings.iOS.targetOSVersionString = "12.0";
   PlayerSettings.iOS.sdkVersion = iOSSdkVersion.DeviceSDK;
   ```

3. **Info.plist Additions**:
   ```xml
   <key>NSCameraUsageDescription</key>
   <string>This app uses the camera for AR inference</string>
   ```

### Android Setup

1. **Gradle Settings**:
   ```gradle
   android {
       compileSdkVersion 30
       defaultConfig {
           minSdkVersion 21
           targetSdkVersion 30
       }
   }
   ```

2. **AndroidManifest.xml**:
   ```xml
   <uses-permission android:name="android.permission.CAMERA" />
   <uses-feature android:name="android.hardware.camera.ar" android:required="true" />
   ```

3. **Build Settings**:
   ```csharp
   PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel21;
   PlayerSettings.Android.targetSdkVersion = AndroidSdkVersions.AndroidApiLevel30;
   ```

## Model Management

### Supported Formats

- ✅ **ONNX**: Cross-platform inference
- ✅ **Core ML**: iOS-specific optimization
- ✅ **TensorFlow Lite**: Android NNAPI support
- ✅ **Custom**: TrustformeRS native format

### Model Loading

```csharp
// Synchronous loading
bool success = engine.LoadModel("StreamingAssets/model.onnx");

// Asynchronous loading
bool success = await engine.LoadModelAsync("StreamingAssets/model.onnx");

// From Resources
bool success = await engine.LoadModelAsync(Resources.Load<TextAsset>("model").bytes);
```

### Model Optimization

```csharp
// Enable quantization
engine.config.quantization.enabled = true;
engine.config.quantization.scheme = QuantizationScheme.Int8;

// Memory optimization
engine.config.memoryOptimization = MemoryOptimization.Maximum;

// Platform-specific optimization
if (Application.platform == RuntimePlatform.IPhonePlayer)
{
    engine.config.backend = MobileBackend.CoreML;
}
```

## Performance Optimization

### Automatic Optimization

The performance optimizer automatically adjusts quality based on:
- **FPS**: Reduces inference frequency when FPS drops
- **Memory**: Enables memory optimization when usage is high
- **Thermal**: Throttles performance to prevent overheating
- **Battery**: Reduces quality on low battery

### Manual Optimization

```csharp
// Reduce inference frequency
optimizer.ForceApplyOptimization(OptimizationType.ReduceInferenceFrequency);

// Lower model precision
optimizer.ForceApplyOptimization(OptimizationType.LowerModelPrecision);

// Enable memory optimization
optimizer.ForceApplyOptimization(OptimizationType.EnableMemoryOptimization);

// Reset to defaults
optimizer.ResetOptimizations();
```

## Error Handling

```csharp
engine.OnError += (error) => {
    Debug.LogError($"TrustformeRS Error: {error}");
    // Handle error appropriately
};

arManager.OnARError += (error) => {
    Debug.LogError($"AR Error: {error}");
    // Disable AR features or show fallback UI
};
```

## Best Practices

### Memory Management
- Use memory pools for frequent allocations
- Enable aggressive GC for low-memory devices
- Monitor memory usage with the performance optimizer

### Performance
- Start with balanced settings and adjust based on device capabilities
- Use adaptive quality for varying performance requirements
- Profile regularly with Unity Profiler

### Battery Life
- Enable battery optimization for mobile builds
- Reduce inference frequency on low battery
- Use lower precision models when appropriate

## Debugging

### Performance UI
```csharp
optimizer.showPerformanceUI = true; // Shows debug overlay
```

### Logging
```csharp
engine.showDebugInfo = true;       // Engine debug info
optimizer.enableLogging = true;    // Performance logging
```

### Profiling
Use Unity Profiler with custom markers:
- `TrustformeRS.Inference`
- `TrustformeRS.ModelLoad`
- `TrustformeRS.ARProcessing`

## API Reference

### TrustformersEngine
- `LoadModel(path)` - Load model synchronously
- `LoadModelAsync(path)` - Load model asynchronously
- `Inference(input)` - Perform single inference
- `InferenceAsync(input)` - Perform async inference
- `BatchInference(inputs)` - Batch processing
- `GetStats()` - Get performance statistics

### TrustformersARManager
- `SetFeatureEnabled(feature, enabled)` - Enable/disable AR features
- `SetInferenceInterval(interval)` - Adjust inference frequency
- `GetARStats()` - Get AR processing statistics

### TrustformersPerformanceOptimizer
- `GetMetrics()` - Get current performance metrics
- `ForceApplyOptimization(type)` - Apply specific optimization
- `ResetOptimizations()` - Reset to default state
- `UpdatePerformanceTargets(fps, memory)` - Update targets

## Troubleshooting

### Common Issues

**Model Loading Fails**
- Ensure model is in correct format
- Check file path and permissions
- Verify model compatibility with target platform

**Poor Performance**
- Enable performance optimizer
- Check device thermal state
- Reduce model complexity or input resolution

**AR Features Not Working**
- Verify AR Foundation is installed
- Check camera permissions
- Ensure device supports ARCore/ARKit

**IL2CPP Build Errors**
- Enable "Allow 'unsafe' Code" in player settings
- Verify all native libraries are properly included
- Check for missing AOT stubs

### Support

For additional support:
- GitHub Issues: [Report bugs and feature requests](https://github.com/your-org/trustformers-mobile/issues)
- Documentation: [Full API documentation](https://docs.trustformers.ai/unity)
- Community: [Discord community](https://discord.gg/trustformers)

## License

This package is licensed under the MIT License. See LICENSE file for details.

## Changelog

### v1.0.0
- Initial release with full Unity integration
- iOS and Android platform support
- AR Foundation integration
- IL2CPP compatibility
- Performance optimization system