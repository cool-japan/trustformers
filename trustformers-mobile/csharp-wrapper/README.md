# TrustformeRS Mobile - C# Bindings

High-performance C# bindings for TrustformeRS Mobile, enabling ML inference on mobile and desktop platforms with .NET.

## Features

- **Cross-platform**: Windows, macOS, Linux, iOS, Android
- **Multiple backends**: CPU, CoreML (iOS), NNAPI (Android), GPU acceleration
- **Memory optimized**: Configurable memory usage and optimization levels
- **Zero-copy operations**: Span<T> support for high-performance scenarios
- **Thread-safe**: Safe for concurrent usage
- **Native performance**: Direct P/Invoke to Rust native library

## Installation

```bash
dotnet add package TrustformersMobile
```

## Quick Start

### Basic Usage

```csharp
using TrustformersMobile;

// Initialize the library
TrustformersMobile.TrustformersMobile.Initialize();

// Create a simple inference engine
using var engine = TrustformersMobile.TrustformersMobile.CreateQuickEngine("path/to/model.bin");

// Prepare input data
float[] input = { 1.0f, 2.0f, 3.0f, 4.0f };

// Run inference
float[] output = engine.InferenceF32(input);

Console.WriteLine($"Output: [{string.Join(", ", output)}]");
```

### Platform-Optimized Usage

```csharp
using TrustformersMobile;

// Create platform-optimized engine (automatically detects iOS/Android)
using var engine = TrustformersMobile.TrustformersMobile.CreateOptimizedEngine("path/to/model.bin");

// Run inference
float[] input = { /* your input data */ };
float[] output = engine.InferenceF32(input);
```

### Advanced Configuration

```csharp
using TrustformersMobile;

// Create custom configuration
using var config = new MobileConfig();

// Configure for iOS with CoreML
config.SetPlatform(MobilePlatform.iOS);
config.SetBackend(MobileBackend.CoreML);
config.SetMemoryOptimization(MemoryOptimization.Balanced);
config.SetMaxMemoryMb(512);
config.SetUseFp16(true);
config.SetNumThreads(4);

// Validate configuration
config.Validate();

// Create engine with custom configuration
using var engine = new MobileInferenceEngine(config, "path/to/model.bin");

// Run inference
float[] input = { /* your input data */ };
float[] output = engine.InferenceF32(input);
```

### Zero-Copy Operations with Span<T>

```csharp
using TrustformersMobile;

using var engine = TrustformersMobile.TrustformersMobile.CreateQuickEngine("path/to/model.bin");

// Use Span<T> for zero-copy operations
Span<float> inputSpan = stackalloc float[4] { 1.0f, 2.0f, 3.0f, 4.0f };
Span<float> outputSpan = stackalloc float[10]; // Pre-allocated output buffer

// Run inference without copying data
int actualOutputSize = engine.InferenceF32(inputSpan, outputSpan);

// Use only the actual output data
var result = outputSpan[..actualOutputSize];
```

### Device Information

```csharp
using TrustformersMobile;

// Get device information
var deviceInfo = TrustformersMobile.TrustformersMobile.GetDeviceInfo();

Console.WriteLine($"Device: {deviceInfo.Name}");
Console.WriteLine($"OS: {deviceInfo.OS}");
Console.WriteLine($"CPU Cores: {deviceInfo.CpuCores}");
Console.WriteLine($"Memory: {deviceInfo.TotalMemoryMB} MB");
Console.WriteLine($"GPU: {deviceInfo.GPU}");
Console.WriteLine($"Performance Tier: {deviceInfo.PerformanceTier}");

// Check platform/backend support
bool isIOSSupported = TrustformersMobile.TrustformersMobile.IsPlatformSupported(MobilePlatform.iOS);
bool isCoreMLSupported = TrustformersMobile.TrustformersMobile.IsBackendSupported(MobileBackend.CoreML);
```

## Configuration Options

### Platforms
- `iOS`: Apple iOS devices
- `Android`: Android devices  
- `Generic`: Cross-platform (desktop/other)

### Backends
- `CPU`: CPU-only inference (always available)
- `CoreML`: Apple's CoreML framework (iOS only)
- `NNAPI`: Android Neural Networks API (Android only)
- `GPU`: Generic GPU acceleration
- `Metal`: Apple Metal (iOS/macOS)
- `Vulkan`: Cross-platform GPU acceleration
- `OpenCL`: Cross-platform GPU acceleration
- `Custom`: Custom backend

### Memory Optimization
- `Minimal`: Fastest inference, higher memory usage
- `Balanced`: Balance between speed and memory
- `Maximum`: Lowest memory usage, slower inference

## Pre-built Configurations

### iOS Optimized
```csharp
using var config = MobileConfig.CreateiOSOptimized();
// Uses: CoreML backend, FP16 precision, batching enabled
```

### Android Optimized  
```csharp
using var config = MobileConfig.CreateAndroidOptimized();
// Uses: NNAPI backend, INT8 quantization, conservative memory
```

### Ultra Low Memory
```csharp
using var config = MobileConfig.CreateUltraLowMemory();
// Uses: INT4 quantization, single-threaded, minimal memory
```

## Error Handling

```csharp
using TrustformersMobile;

try
{
    using var engine = TrustformersMobile.TrustformersMobile.CreateQuickEngine("invalid/path.bin");
}
catch (TrustformersMobileException ex)
{
    Console.WriteLine($"Error: {ex.Message}");
    Console.WriteLine($"Error Code: {ex.ErrorCode}");
    
    switch (ex.ErrorCode)
    {
        case TrustformersMobileError.ModelLoadError:
            Console.WriteLine("Check model file path and format");
            break;
        case TrustformersMobileError.OutOfMemory:
            Console.WriteLine("Reduce memory usage or use lower precision");
            break;
        case TrustformersMobileError.PlatformNotSupported:
            Console.WriteLine("Try a different backend");
            break;
    }
}
```

## Threading and Concurrency

The library is thread-safe. You can use the same configuration from multiple threads, but each thread should have its own `MobileInferenceEngine` instance:

```csharp
using var config = new MobileConfig();

// Each thread gets its own engine
var tasks = Enumerable.Range(0, Environment.ProcessorCount)
    .Select(_ => Task.Run(() =>
    {
        using var engine = new MobileInferenceEngine(config, "model.bin");
        // Perform inference...
    }));

await Task.WhenAll(tasks);
```

## Performance Tips

1. **Reuse engines**: Creating engines is expensive, reuse them when possible
2. **Use Span<T>**: Avoid allocations with `ReadOnlySpan<float>` and `Span<float>`
3. **Pre-allocate buffers**: Reuse output buffers to avoid GC pressure
4. **Choose appropriate precision**: FP16 uses half the memory of FP32
5. **Configure thread count**: Set to number of CPU cores for CPU inference
6. **Platform-specific backends**: Use CoreML on iOS, NNAPI on Android

## Requirements

- .NET 8.0 or later
- Native library for your platform:
  - Windows: `trustformers_mobile.dll`
  - macOS: `libtrusformers_mobile.dylib`
  - Linux: `libtrusformers_mobile.so`

## Building from Source

1. Build the Rust native library:
```bash
cd trustformers-mobile
cargo build --release --features "coreml,nnapi"
```

2. Build the C# wrapper:
```bash
cd csharp-wrapper
dotnet build
```

3. Run tests:
```bash
dotnet test
```

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- GitHub Issues: https://github.com/cool-japan/trustformers/issues
- Documentation: https://docs.trustformers.ai/mobile/csharp