# TrustformeRS Go Bindings

High-performance Go bindings for the TrustformeRS-C library, providing native Go access to advanced transformer models and natural language processing capabilities.

## Features

- **Complete C API Coverage**: Full access to all TrustformeRS-C functionality
- **Memory Safe**: Automatic resource management with Go finalizers
- **Type Safe**: Strong typing with comprehensive error handling
- **Performance Optimized**: Zero-copy operations where possible
- **Cross-Platform**: Support for Windows, macOS, and Linux
- **Production Ready**: Advanced memory tracking and performance monitoring

## Installation

### Prerequisites

1. **TrustformeRS-C Library**: Build and install the TrustformeRS-C library first
2. **Go 1.19+**: Ensure you have Go 1.19 or later installed
3. **C Compiler**: Required for CGO (GCC, Clang, or MSVC)

### Building the Library

```bash
# Clone the repository
git clone https://github.com/trustformers/trustformers-c
cd trustformers-c

# Build the C library
cargo build --release

# Set library path (adjust path as needed)
export TRUSTFORMERS_C_LIB_PATH="$(pwd)/target/release"
```

### Go Module Setup

```bash
cd golang
go mod tidy
```

## Quick Start

### Basic Usage

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/trustformers/trustformers-c/golang/trustformers"
)

func main() {
    // Initialize TrustformeRS
    tf, err := trustformers.NewTrustformeRS()
    if err != nil {
        log.Fatal(err)
    }
    defer tf.Cleanup()
    
    // Display version and features
    fmt.Printf("TrustformeRS Version: %s\n", tf.Version())
    fmt.Printf("GPU Support: %v\n", tf.HasFeature("gpu"))
    
    // Get memory usage
    memUsage, _ := tf.GetMemoryUsage()
    fmt.Printf("Memory Usage: %d bytes\n", memUsage.TotalMemoryBytes)
}
```

### Text Generation

```go
// Load model and tokenizer
model, err := tf.LoadModelFromHub("gpt2")
if err != nil {
    log.Fatal(err)
}
defer model.Free()

tokenizer, err := tf.LoadTokenizerFromHub("gpt2")
if err != nil {
    log.Fatal(err)
}
defer tokenizer.Free()

// Create pipeline
pipeline, err := tf.CreateTextGenerationPipeline(model, tokenizer)
if err != nil {
    log.Fatal(err)
}
defer pipeline.Free()

// Generate text
prompt := "The future of AI is"
result, err := pipeline.GenerateText(prompt)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Generated: %s\n", result)
```

### Text Classification

```go
// Load classification model
model, err := tf.LoadModelFromHub("distilbert-base-uncased-finetuned-sst-2-english")
if err != nil {
    log.Fatal(err)
}
defer model.Free()

tokenizer, err := tf.LoadTokenizerFromHub("distilbert-base-uncased-finetuned-sst-2-english")
if err != nil {
    log.Fatal(err)
}
defer tokenizer.Free()

// Create classification pipeline
pipeline, err := tf.CreateTextClassificationPipeline(model, tokenizer)
if err != nil {
    log.Fatal(err)
}
defer pipeline.Free()

// Classify text
text := "I love this product!"
results, err := pipeline.ClassifyText(text)
if err != nil {
    log.Fatal(err)
}

for _, result := range results {
    fmt.Printf("Label: %s, Score: %.4f\n", result.Label, result.Score)
}
```

## API Reference

### Core Types

#### `TrustformeRS`
Main library interface for initialization and global operations.

**Methods:**
- `NewTrustformeRS() (*TrustformeRS, error)` - Create new instance
- `Init() error` - Initialize library
- `Cleanup() error` - Cleanup resources
- `Version() string` - Get version
- `HasFeature(feature string) bool` - Check feature availability
- `SetLogLevel(level LogLevel) error` - Set logging level

#### `Model`
Represents a loaded transformer model.

**Methods:**
- `LoadModelFromHub(modelName string) (*Model, error)` - Load from Hugging Face Hub
- `LoadModelFromPath(modelPath string) (*Model, error)` - Load from local path
- `Free() error` - Release resources
- `GetInfo() (ModelInfo, error)` - Get model information
- `SetQuantization(bits int) error` - Set quantization level
- `Validate() (bool, error)` - Validate model integrity

#### `Tokenizer`
Text tokenization interface.

**Methods:**
- `LoadTokenizerFromHub(modelName string) (*Tokenizer, error)` - Load from Hub
- `LoadTokenizerFromPath(path string) (*Tokenizer, error)` - Load from path
- `Encode(text string) ([]int, error)` - Encode text to tokens
- `Decode(tokens []int) (string, error)` - Decode tokens to text
- `EncodeBatch(texts []string) ([][]int, error)` - Batch encoding
- `DecodeBatch(tokenBatches [][]int) ([]string, error)` - Batch decoding
- `GetVocabSize() (int, error)` - Get vocabulary size

#### `Pipeline`
High-level inference interface for different tasks.

**Methods:**
- `CreateTextGenerationPipeline(model, tokenizer) (*Pipeline, error)`
- `CreateTextClassificationPipeline(model, tokenizer) (*Pipeline, error)`
- `CreateQuestionAnsweringPipeline(model, tokenizer) (*Pipeline, error)`
- `CreateConversationalPipeline(model, tokenizer) (*Pipeline, error)`
- `GenerateText(prompt string) (string, error)`
- `ClassifyText(text string) ([]ClassificationResult, error)`
- `AnswerQuestion(context, question string) (AnswerResult, error)`

### Configuration Types

#### `OptimizationConfig`
Performance optimization settings.

```go
type OptimizationConfig struct {
    EnableTracking          bool
    EnableCaching           bool
    CacheSizeMB            int
    NumThreads             int
    EnableSIMD             bool
    OptimizeBatchSize      bool
    MemoryOptimizationLevel int
}
```

#### `GenerationOptions`
Text generation configuration.

```go
type GenerationOptions struct {
    MaxLength         int
    MinLength         int
    Temperature       float64
    TopK              int
    TopP              float64
    RepetitionPenalty float64
    DoSample          bool
    EarlyStopping     bool
    NumBeams          int
    NumReturnSequences int
}
```

## Memory Management

The Go bindings provide automatic memory management through Go's finalizers, but explicit cleanup is recommended for better performance:

```go
// Automatic cleanup (not recommended for production)
model, _ := tf.LoadModelFromHub("gpt2")
// model will be cleaned up by finalizer

// Explicit cleanup (recommended)
model, _ := tf.LoadModelFromHub("gpt2")
defer model.Free() // Immediate cleanup
```

### Memory Monitoring

```go
// Basic memory usage
memUsage, _ := tf.GetMemoryUsage()
fmt.Printf("Total: %d bytes\n", memUsage.TotalMemoryBytes)

// Advanced memory statistics
advMemUsage, _ := tf.GetAdvancedMemoryUsage()
fmt.Printf("Pressure Level: %d\n", advMemUsage.PressureLevel)
fmt.Printf("Fragmentation: %.2f\n", advMemUsage.FragmentationRatio)

// Memory leak detection
leakReport, _ := tf.CheckMemoryLeaks()
fmt.Printf("Leak Report: %+v\n", leakReport)
```

## Performance Optimization

### SIMD Acceleration

```go
config := trustformers.OptimizationConfig{
    EnableSIMD: true,
    EnableCaching: true,
    CacheSizeMB: 512,
}
tf.ApplyOptimizations(config)
```

### Performance Profiling

```go
// Start profiling
tf.StartProfiling()

// ... perform operations ...

// Stop and get report
report, _ := tf.StopProfiling()
fmt.Printf("Performance Report: %+v\n", report)
```

### Batch Processing

```go
// Batch text classification for better throughput
texts := []string{"text1", "text2", "text3"}
results, _ := pipeline.ClassifyTextBatch(texts)
```

## Error Handling

The bindings provide comprehensive error handling:

```go
import "errors"

// Standard error types
var (
    ErrNullPointer        = errors.New("null pointer")
    ErrInvalidParameter   = errors.New("invalid parameter")
    ErrRuntimeError       = errors.New("runtime error")
    ErrMemoryError        = errors.New("memory error")
    ErrNotImplemented     = errors.New("not implemented")
)

// Error handling example
model, err := tf.LoadModelFromHub("invalid-model")
if err != nil {
    if errors.Is(err, trustformers.ErrRuntimeError) {
        log.Printf("Runtime error loading model: %v", err)
    }
    return err
}
```

## Examples

Complete examples are available in the `examples/` directory:

- `basic_usage.go` - Library initialization and basic operations
- `text_generation.go` - Text generation with GPT models
- `text_classification.go` - Sentiment analysis and classification
- `question_answering.go` - Question answering systems
- `conversational.go` - Chatbot and conversational AI

## Building Examples

```bash
cd examples

# Basic usage
go run basic_usage.go

# Text generation (requires model download)
go run text_generation.go

# Text classification
go run text_classification.go
```

## Cross-Platform Notes

### Windows
- Requires MSVC or MinGW for CGO compilation
- Library path: `trustformers_c.dll`
- Set `CGO_ENABLED=1` if not already set

### macOS
- Uses Accelerate framework for optimized operations
- Universal binary support for Intel and Apple Silicon
- Library path: `libtrustformers_c.dylib`

### Linux
- Requires GCC or Clang
- Library path: `libtrustformers_c.so`
- pkg-config support available

## Troubleshooting

### Library Not Found

```bash
# Set library path environment variable
export TRUSTFORMERS_C_LIB_PATH="/path/to/trustformers-c/target/release"

# Or use LD_LIBRARY_PATH on Linux
export LD_LIBRARY_PATH="/path/to/trustformers-c/target/release:$LD_LIBRARY_PATH"
```

### CGO Compilation Issues

```bash
# Enable CGO if disabled
export CGO_ENABLED=1

# Set C compiler (if needed)
export CC=gcc
```

### Performance Issues

1. **Enable optimizations**: Use `OptimizationConfig` to enable SIMD and caching
2. **Batch operations**: Use batch methods for multiple inputs
3. **Memory management**: Call `Free()` explicitly on resources
4. **Threading**: Configure appropriate thread count for your system

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all examples work
5. Submit a pull request

## License

This project is licensed under the same terms as the main TrustformeRS project.

## Support

For issues and questions:
- GitHub Issues: [Repository Issues](https://github.com/trustformers/trustformers-c/issues)
- Documentation: [TrustformeRS Docs](https://trustformers.github.io/docs/)
- Community: [Discord/Forum Link]