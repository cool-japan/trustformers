# TrustformeRS C API

High-performance transformer library C bindings for cross-language integration.

## Overview

The TrustformeRS C API provides C-compatible bindings for the TrustformeRS transformer library, enabling integration with other programming languages through Foreign Function Interface (FFI). This crate exposes the core functionality of TrustformeRS including tokenizers, models, pipelines, and ONNX backend support.

## Features

- 🚀 **High Performance**: Native Rust performance with C ABI compatibility
- 🔧 **Cross-Language**: Use from C, C++, Python, JavaScript, and other languages
- 📦 **Complete API**: Full access to TrustformeRS functionality
- 🛡️ **Memory Safe**: Robust memory management and error handling
- 🎯 **Production Ready**: Designed for production deployment
- 📊 **Benchmarking**: Built-in performance monitoring and profiling

### Core Components

- **Pipelines**: Text classification, generation, Q&A, and custom pipelines
- **Tokenizers**: Pre-trained and custom tokenizers with full encoding/decoding
- **Models**: AutoModel loading with support for popular architectures
- **ONNX Backend**: Optimized inference using ONNX Runtime
- **Tensors**: Low-level tensor operations and manipulation
- **Utilities**: Performance timing, memory monitoring, and system info

## Installation

### Building from Source

```bash
# Clone the repository
git clone https://github.com/your-org/trustformers.git
cd trustformers/trustformers-c

# Build the C library
cargo build --release

# Generate C headers
cargo build

# The library and headers will be in target/release/
```

### Using with pkg-config

After building, the library provides pkg-config support:

```bash
# Check if the library is found
pkg-config --exists trustformers-c

# Get compile flags
pkg-config --cflags trustformers-c

# Get linking flags  
pkg-config --libs trustformers-c
```

### Using with CMake

The library provides CMake configuration files:

```cmake
find_package(TrustformeRS REQUIRED)

target_link_libraries(your_target TrustformeRS::trustformers_c)
```

## Quick Start

### 1. Basic C Usage

```c
#include "trustformers.h"
#include <stdio.h>

int main() {
    // Initialize the library
    TrustformersError error = trustformers_init();
    if (error != TrustformersError_Success) {
        fprintf(stderr, "Failed to initialize TrustformeRS\n");
        return 1;
    }
    
    // Create a text classification pipeline
    TrustformersPipelineConfig config = {0};
    config.task = "text-classification";
    config.model = "distilbert-base-uncased";
    config.backend_type = 0; // Native backend
    config.device_type = 0;  // CPU
    
    TrustformersPipeline pipeline;
    error = trustformers_pipeline_create(&config, &pipeline);
    if (error != TrustformersError_Success) {
        printf("Failed to create pipeline\n");
        return 1;
    }
    
    // Perform inference
    const char* text = "I love this product!";
    TrustformersInferenceResult result = {0};
    error = trustformers_pipeline_infer(pipeline, text, &result);
    
    if (error == TrustformersError_Success) {
        printf("Input: %s\n", text);
        printf("Result: %s\n", result.result_json);
        printf("Time: %.2f ms\n", result.inference_time_ms);
        
        // Free result memory
        trustformers_inference_result_free(&result);
    }
    
    // Cleanup
    trustformers_pipeline_destroy(pipeline);
    trustformers_cleanup();
    return 0;
}
```

### 2. Compilation

```bash
# Using pkg-config (recommended)
gcc -o example example.c $(pkg-config --cflags --libs trustformers-c)

# Manual compilation
gcc -o example example.c -I/path/to/headers -L/path/to/lib -ltrustformers_c
```

### 3. Python Integration (ctypes)

```python
import ctypes
import json

# Load the library
lib = ctypes.CDLL('./target/release/libtrustformers_c.so')

# Initialize
lib.trustformers_init()

# Create pipeline
config = PipelineConfig()
config.task = b"text-classification"
config.model = b"distilbert-base-uncased"

pipeline = ctypes.c_ulong()
error = lib.trustformers_pipeline_create(ctypes.byref(config), ctypes.byref(pipeline))

if error == 0:  # Success
    # Perform inference
    result = InferenceResult()
    error = lib.trustformers_pipeline_infer(
        pipeline.value, 
        b"Great product!", 
        ctypes.byref(result)
    )
    
    if error == 0:
        # Parse JSON result
        result_str = ctypes.string_at(result.result_json).decode()
        data = json.loads(result_str)
        print(f"Prediction: {data}")

# Cleanup
lib.trustformers_cleanup()
```

## API Reference

### Error Handling

All functions return `TrustformersError` codes:

```c
typedef enum {
    TrustformersError_Success = 0,
    TrustformersError_NullPointer = -1,
    TrustformersError_InvalidParameter = -2,
    TrustformersError_OutOfMemory = -3,
    TrustformersError_FileNotFound = -4,
    TrustformersError_ModelLoadError = -6,
    TrustformersError_InferenceError = -9,
    // ... more error codes
} TrustformersError;

// Get error message
const char* trustformers_error_message(TrustformersError error);
```

### Initialization

```c
// Initialize the library (call once at startup)
TrustformersError trustformers_init();

// Cleanup resources (call once at shutdown)
TrustformersError trustformers_cleanup();

// Get version information
const char* trustformers_version();
TrustformersError trustformers_build_info(TrustformersBuildInfo* info);
```

### Pipelines

```c
// Pipeline configuration
typedef struct {
    const char* task;              // "text-classification", "text-generation", etc.
    const char* model;             // Model name or path
    int backend_type;              // 0=Native, 1=ONNX
    const char* onnx_model_path;   // Path to ONNX model (if using ONNX backend)
    int device_type;               // 0=CPU, 1=CUDA
    int batch_size;                // Batch size for inference
    int max_length;                // Maximum sequence length
    // ... more configuration options
} TrustformersPipelineConfig;

// Create pipeline
TrustformersError trustformers_pipeline_create(
    const TrustformersPipelineConfig* config,
    TrustformersPipeline* pipeline_handle
);

// Single inference
TrustformersError trustformers_pipeline_infer(
    TrustformersPipeline pipeline_handle,
    const char* input,
    TrustformersInferenceResult* result
);

// Batch inference
TrustformersError trustformers_pipeline_batch_infer(
    TrustformersPipeline pipeline_handle,
    const char* const* inputs,
    size_t num_inputs,
    TrustformersBatchResult* result
);

// Cleanup
TrustformersError trustformers_pipeline_destroy(TrustformersPipeline pipeline_handle);
```

### Tokenizers

```c
// Load tokenizer
TrustformersError trustformers_tokenizer_from_pretrained(
    const char* model_name,
    TrustformersTokenizer* tokenizer_handle
);

// Encode text
TrustformersError trustformers_tokenizer_encode(
    TrustformersTokenizer tokenizer_handle,
    const char* text,
    const TrustformersTokenizerConfig* config,
    TrustformersEncoding* encoding
);

// Decode tokens
TrustformersError trustformers_tokenizer_decode(
    TrustformersTokenizer tokenizer_handle,
    const unsigned int* token_ids,
    size_t num_tokens,
    int skip_special_tokens,
    char** decoded_text
);
```

### ONNX Backend

```c
// Create ONNX text classification pipeline
TrustformersError trustformers_onnx_text_classification_pipeline_create(
    const char* model_path,
    const char* tokenizer_name,
    TrustformersPipeline* pipeline_handle
);

// Create ONNX text generation pipeline  
TrustformersError trustformers_onnx_text_generation_pipeline_create(
    const char* model_path,
    const char* tokenizer_name,
    TrustformersPipeline* pipeline_handle
);
```

### Performance Monitoring

```c
// Create performance timer
void* trustformers_timer_create();

// Start/stop timing
TrustformersError trustformers_timer_start(void* timer);
TrustformersError trustformers_timer_stop(void* timer, double* elapsed_ms);

// Get statistics
TrustformersError trustformers_timer_stats(
    void* timer, 
    TrustformersBenchmarkResult* stats
);
```

### Memory Management

```c
// Get memory usage
TrustformersError trustformers_get_memory_usage(TrustformersMemoryUsage* usage);

// Force garbage collection
TrustformersError trustformers_gc();

// Free specific result types
void trustformers_inference_result_free(TrustformersInferenceResult* result);
void trustformers_batch_result_free(TrustformersBatchResult* result);
void trustformers_encoding_free(TrustformersEncoding* encoding);
```

## Examples

The `examples/` directory contains:

- **`basic_usage.c`**: Complete basic usage example
- **`advanced_pipeline.c`**: Advanced features and ONNX backend
- **`Makefile`**: Build instructions for examples

To build and run examples:

```bash
cd examples
make
./basic_usage
./advanced_pipeline
```

## Language Bindings

### Python

```python
# Using ctypes (basic)
import ctypes
lib = ctypes.CDLL('libtrustformers_c.so')

# Using cffi (recommended)
from cffi import FFI
ffi = FFI()
ffi.cdef(open('trustformers.h').read())
lib = ffi.dlopen('libtrustformers_c.so')
```

### Node.js

```javascript
const ffi = require('ffi-napi');
const ref = require('ref-napi');

const lib = ffi.Library('libtrustformers_c', {
  'trustformers_init': ['int', []],
  'trustformers_pipeline_create': ['int', ['pointer', 'pointer']],
  // ... more function definitions
});
```

### C++

```cpp
extern "C" {
#include "trustformers.h"
}

class TrustformersPipelineWrapper {
    TrustformersPipeline pipeline_;
public:
    TrustformersPipelineWrapper(const TrustformersPipelineConfig& config) {
        if (trustformers_pipeline_create(&config, &pipeline_) != TrustformersError_Success) {
            throw std::runtime_error("Failed to create pipeline");
        }
    }
    
    ~TrustformersPipelineWrapper() {
        trustformers_pipeline_destroy(pipeline_);
    }
    
    std::string infer(const std::string& input) {
        TrustformersInferenceResult result = {0};
        auto error = trustformers_pipeline_infer(pipeline_, input.c_str(), &result);
        if (error != TrustformersError_Success) {
            throw std::runtime_error("Inference failed");
        }
        
        std::string json_result(result.result_json);
        trustformers_inference_result_free(&result);
        return json_result;
    }
};
```

## Build Configuration

### Features

Control compilation features in `Cargo.toml`:

```toml
[features]
default = ["tokenizers", "models", "pipelines"]
tokenizers = ["trustformers-tokenizers"]
models = ["trustformers-models"]
pipelines = ["trustformers/pipeline"] 
gpu = ["trustformers-core/gpu"]
onnx = ["trustformers-core/onnx"]
quantization = ["trustformers-core/quantization"]
debug = ["trustformers-debug"]
```

### Cross-Compilation

```bash
# For ARM64
cargo build --target aarch64-unknown-linux-gnu --release

# For Windows
cargo build --target x86_64-pc-windows-gnu --release

# For macOS
cargo build --target x86_64-apple-darwin --release
```

### Custom cbindgen Configuration

Modify `cbindgen.toml` to customize header generation:

```toml
language = "C"
style = "both" 
cpp_compat = true
include_guard = "TRUSTFORMERS_H"

[export]
include = ["TrustformersError", "TrustformersPipeline"]
exclude = ["InternalType"]
```

## Performance Optimization

### CPU Optimization

```c
// Configure for CPU performance
TrustformersPipelineConfig config = {0};
config.device_type = 0; // CPU
config.num_threads = 8; // Use 8 threads
config.batch_size = 16; // Larger batches for throughput
```

### GPU Acceleration

```c
// Configure for CUDA
TrustformersPipelineConfig config = {0};
config.device_type = 1; // CUDA
config.device_id = 0;   // GPU 0
config.backend_type = 1; // ONNX backend for optimized GPU inference
```

### ONNX Runtime Optimization

```c
// Use ONNX backend for production deployment
config.backend_type = 1; // ONNX
config.onnx_model_path = "optimized_model.onnx";
// ONNX Runtime will automatically use:
// - Graph optimizations
// - Kernel fusion
// - Memory pattern optimization
```

## Troubleshooting

### Common Issues

1. **Library not found**: Check `LD_LIBRARY_PATH` or use `pkg-config`
2. **Symbol errors**: Ensure correct library version and ABI compatibility
3. **Memory leaks**: Always call cleanup functions for results
4. **Performance issues**: Use batch processing and appropriate device settings

### Debug Mode

Build with debug features for troubleshooting:

```bash
cargo build --features debug
export RUST_LOG=debug
./your_program
```

### Memory Debugging

```c
// Monitor memory usage
TrustformersMemoryUsage usage;
trustformers_get_memory_usage(&usage);
printf("Allocated: %llu models, %llu pipelines\n", 
       usage.allocated_models, usage.allocated_pipelines);
```

## Contributing

We welcome contributions! Please see the main [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

### C API Specific Guidelines

1. **Memory Safety**: All functions must handle null pointers gracefully
2. **Error Handling**: Use consistent error codes and messages
3. **Documentation**: Document all public functions and structures
4. **Testing**: Add both unit tests and integration examples
5. **Compatibility**: Maintain ABI compatibility across versions

## License

This project is licensed under the same terms as TrustformeRS. See [LICENSE](../LICENSE) for details.

## Support

- 📝 **Documentation**: [docs.rs/trustformers-c](https://docs.rs/trustformers-c)
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-org/trustformers/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-org/trustformers/discussions)
- 📧 **Email**: support@trustformers.ai

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and breaking changes.