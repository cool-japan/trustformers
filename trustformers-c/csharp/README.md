# TrustformeRS C# Bindings

This package provides C# bindings for TrustformeRS, a high-performance transformer model inference library written in Rust.

## Features

- **High Performance**: Native Rust backend with optimized memory management
- **Multi-Platform**: Support for Windows, Linux, and macOS (x64 and ARM64)
- **Hardware Acceleration**: CUDA, ROCm, and Metal support for GPU acceleration
- **Type Safety**: Strongly-typed C# API with comprehensive error handling
- **Memory Safety**: Automatic resource management with proper disposal patterns
- **HTTP Server**: Built-in model serving capabilities with REST API
- **Comprehensive**: Support for text generation, classification, and custom pipelines

## Installation

### NuGet Package (Recommended)

```bash
dotnet add package TrustformeRS
```

### Manual Installation

1. Download the native libraries for your platform
2. Add reference to `TrustformeRS.dll`
3. Ensure native libraries are in your application's output directory

## Quick Start

### Basic Text Generation

```csharp
using TrustformeRS;

// Initialize the library
TrustformeRS.Initialize();

try
{
    // Load model and tokenizer
    using var model = Model.Load("path/to/model");
    using var tokenizer = Tokenizer.Create("path/to/tokenizer");
    
    // Create pipeline
    using var pipeline = Pipeline.CreateTextGeneration(model, tokenizer);
    
    // Generate text
    var result = pipeline.GenerateText("The future of AI is", new TextGenerationOptions
    {
        MaxLength = 100,
        Temperature = 0.7,
        TopK = 50
    });
    
    Console.WriteLine($"Generated: {result.GeneratedText[0]}");
}
finally
{
    TrustformeRS.Shutdown();
}
```

### Text Classification

```csharp
using TrustformeRS;

// Initialize and load model
TrustformeRS.Initialize();
using var model = Model.Load("path/to/classifier");
using var tokenizer = Tokenizer.Create("path/to/tokenizer");
using var pipeline = Pipeline.CreateTextClassification(model, tokenizer);

// Classify text
var result = pipeline.ClassifyText("I love this product!");
Console.WriteLine($"Classification: {result.TopClassification.Label} ({result.TopClassification.Score:F2})");
```

### HTTP Server

```csharp
using TrustformeRS;

// Create and configure server
var config = new HttpServerConfig
{
    Host = "0.0.0.0",
    Port = 8080,
    EnableCors = true
};

using var server = HttpServer.Create(config);

// Add model endpoint
server.AddModelEndpoint(new ModelEndpoint
{
    Name = "text-generator",
    ModelPath = "path/to/model",
    EndpointPath = "/generate",
    MaxBatchSize = 4
});

// Start server
server.Start();
Console.WriteLine("Server started on http://localhost:8080");

// Keep running
Console.ReadKey();
```

## API Reference

### Core Classes

#### `TrustformeRS` (Static)
Main entry point for library initialization and global operations.

- `Initialize()` - Initialize the library
- `Shutdown()` - Cleanup and shutdown
- `GetVersion()` - Get library version
- `GetMemoryUsage()` - Get memory usage information
- `IsCudaAvailable()` - Check CUDA availability
- `IsRocmAvailable()` - Check ROCm availability

#### `Model`
Represents a loaded transformer model.

- `Load(path, config)` - Load model from file
- `Unload()` - Unload model and free memory
- `Info` - Model metadata and information

#### `Tokenizer`
Handles text tokenization and encoding/decoding.

- `Create(path)` - Create tokenizer from file
- `Encode(text)` - Encode text to token IDs
- `Decode(tokens)` - Decode token IDs to text
- `GetTokenCount(text)` - Get number of tokens
- `SplitIntoChunks(text, maxTokens)` - Split text into chunks

#### `Pipeline`
High-level interface for model inference.

- `CreateTextGeneration(model, tokenizer)` - Create text generation pipeline
- `CreateTextClassification(model, tokenizer)` - Create classification pipeline
- `GenerateText(prompt, options)` - Generate text
- `ClassifyText(text)` - Classify text

#### `HttpServer`
Built-in HTTP server for model serving.

- `Create(config)` - Create server with configuration
- `AddModelEndpoint(endpoint)` - Add model endpoint
- `Start()` / `Stop()` - Start/stop server
- `GetMetrics()` - Get server metrics

### Configuration Options

#### `ModelLoadOptions`
```csharp
new ModelLoadOptions
{
    Device = "cuda:0",           // Device selection
    Precision = "float16",       // Precision mode
    EnableQuantization = true,   // Enable quantization
    OptimizationLevel = 2        // Optimization level
}
```

#### `TextGenerationOptions`
```csharp
new TextGenerationOptions
{
    MaxLength = 200,             // Maximum output length
    Temperature = 0.8,           // Sampling temperature
    TopK = 50,                   // Top-k sampling
    TopP = 0.95,                 // Top-p (nucleus) sampling
    RepetitionPenalty = 1.1,     // Repetition penalty
    DoSample = true,             // Enable sampling
    StopSequences = [".", "!"]   // Stop sequences
}
```

## Hardware Acceleration

### CUDA Support

```csharp
if (TrustformeRS.IsCudaAvailable())
{
    var model = Model.Load("path/to/model", new ModelLoadOptions
    {
        Device = "cuda:0"  // Use first CUDA device
    });
}
```

### ROCm Support (AMD GPUs)

```csharp
if (TrustformeRS.IsRocmAvailable())
{
    var model = Model.Load("path/to/model", new ModelLoadOptions
    {
        Device = "rocm:0"  // Use first ROCm device
    });
}
```

### Metal Support (Apple Silicon)

```csharp
var model = Model.Load("path/to/model", new ModelLoadOptions
{
    Device = "metal"  // Use Metal Performance Shaders
});
```

## Error Handling

All operations can throw `TrustformersException` with detailed error information:

```csharp
try
{
    var model = Model.Load("invalid/path");
}
catch (TrustformersException ex)
{
    Console.WriteLine($"Error: {ex.ErrorCode}");
    Console.WriteLine($"Message: {ex.Message}");
    Console.WriteLine($"Severity: {TrustformersException.GetSeverity(ex.ErrorCode)}");
    Console.WriteLine($"Recoverable: {TrustformersException.IsRecoverable(ex.ErrorCode)}");
}
```

## Memory Management

The library uses RAII patterns with automatic resource cleanup:

```csharp
// Resources are automatically cleaned up when disposed
using var model = Model.Load("path/to/model");
using var tokenizer = Tokenizer.Create("path/to/tokenizer");
using var pipeline = Pipeline.CreateTextGeneration(model, tokenizer);

// Manual cleanup (optional)
model.Unload();
tokenizer.Destroy();
pipeline.Destroy();
```

## Platform Support

| Platform | Architecture | Support |
|----------|-------------|---------|
| Windows  | x64         | ✅       |
| Windows  | x86         | ✅       |
| Windows  | ARM64       | ✅       |
| Linux    | x64         | ✅       |
| Linux    | ARM64       | ✅       |
| macOS    | x64         | ✅       |
| macOS    | ARM64       | ✅       |

## Performance Tips

1. **Reuse objects**: Keep models and pipelines loaded for multiple operations
2. **Batch processing**: Use batch operations when processing multiple inputs
3. **Hardware acceleration**: Use CUDA/ROCm/Metal when available
4. **Memory management**: Dispose objects properly to avoid memory leaks
5. **Optimization**: Enable model optimization for better performance

## Examples

See the `examples/` directory for complete working examples:

- `BasicUsageExample.cs` - Basic model loading and text generation
- `HttpServerExample.cs` - HTTP server setup and usage
- `BatchProcessingExample.cs` - Batch processing techniques
- `HardwareAccelerationExample.cs` - Using GPU acceleration

## Requirements

- .NET 6.0 or higher (or .NET Standard 2.0+ for older frameworks)
- Native TrustformeRS library for your platform
- Optional: CUDA/ROCm/Metal drivers for hardware acceleration

## Troubleshooting

### Common Issues

1. **DllNotFoundException**: Ensure native library is in the correct location
2. **Hardware not available**: Check driver installation for CUDA/ROCm
3. **Memory errors**: Ensure proper disposal of resources
4. **Path issues**: Use absolute paths for models and tokenizers

### Debugging

Enable debugging mode for more detailed error information:

```csharp
var model = Model.Load("path/to/model", new ModelLoadOptions
{
    EnableDebugging = true
});
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Make your changes
4. Add tests for your changes
5. Submit a pull request

## Support

- Documentation: [TrustformeRS Docs](https://docs.trustformers.ai)
- Issues: [GitHub Issues](https://github.com/cool-japan/trustformers/issues)
- Community: [Discord](https://discord.gg/trustformers)