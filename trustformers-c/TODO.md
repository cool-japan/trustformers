# trustformers-c TODO List

## Overview

The `trustformers-c` crate provides C API bindings for TrustformeRS, enabling integration with C, C++, and other languages via FFI. It includes comprehensive hardware support and safe FFI boundaries.

**Key Responsibilities:**
- C API with extern "C" functions
- FFI-safe data structures
- Memory management helpers
- Error handling for C callers
- Hardware acceleration APIs (CUDA, ROCm, TPU, Intel AI, ASICs)
- Thread-safe operation
- ABI stability guarantees

---

## Current Status

### Implementation Status
✅ **PRODUCTION-READY** - Complete C API
✅ **ZERO COMPILATION ERRORS** - Clean compilation
✅ **ABI STABLE** - Stable binary interface
✅ **THREAD SAFE** - Safe for multi-threaded use
✅ **COMPREHENSIVE** - Full feature coverage

### Feature Coverage
- **API:** Model loading, inference, tokenization
- **Hardware:** CUDA, ROCm, Metal, TPU, Intel AI, Qualcomm, Apple Silicon
- **Safety:** Memory safety, error handling, null-pointer checks
- **Compatibility:** C89 compatible headers

---

## Completed Features

### Core C API

#### Model Operations

**Model loading and inference**

- ✅ **Functions**
  - `trustformers_model_load()` - Load model from path
  - `trustformers_model_forward()` - Run inference
  - `trustformers_model_free()` - Free model memory
  - `trustformers_model_num_parameters()` - Get parameter count

**Example:**
```c
#include <trustformers.h>

// Load model
TrustformersModel* model = NULL;
TrustformersError error = trustformers_model_load("gpt2", &model);
if (error != TRUSTFORMERS_SUCCESS) {
    fprintf(stderr, "Error: %s\n", trustformers_error_message(error));
    return 1;
}

// Run inference
float* input_data = ...; // Input tensor data
size_t input_shape[] = {1, 10};
TrustformersTensor* output = NULL;

error = trustformers_model_forward(
    model,
    input_data,
    input_shape,
    2,  // ndim
    &output
);

// Free resources
trustformers_tensor_free(output);
trustformers_model_free(model);
```

---

#### Tokenizer Operations

**Text tokenization from C**

- ✅ **Functions**
  - `trustformers_tokenizer_load()` - Load tokenizer
  - `trustformers_tokenizer_encode()` - Encode text
  - `trustformers_tokenizer_decode()` - Decode tokens
  - `trustformers_tokenizer_free()` - Free tokenizer

**Example:**
```c
// Load tokenizer
TrustformersTokenizer* tokenizer = NULL;
trustformers_tokenizer_load("gpt2", &tokenizer);

// Encode text
const char* text = "Hello, world!";
int32_t* token_ids = NULL;
size_t num_tokens = 0;

trustformers_tokenizer_encode(
    tokenizer,
    text,
    &token_ids,
    &num_tokens
);

// Use tokens...

// Free resources
trustformers_free(token_ids);
trustformers_tokenizer_free(tokenizer);
```

---

### Tensor Operations

#### Tensor API

**Tensor creation and manipulation**

- ✅ **Functions**
  - `trustformers_tensor_create()` - Create tensor
  - `trustformers_tensor_zeros()` - Create zeros tensor
  - `trustformers_tensor_ones()` - Create ones tensor
  - `trustformers_tensor_randn()` - Create random tensor
  - `trustformers_tensor_shape()` - Get tensor shape
  - `trustformers_tensor_data()` - Access tensor data

**Example:**
```c
// Create tensor
float data[] = {1.0, 2.0, 3.0, 4.0};
size_t shape[] = {2, 2};
TrustformersTensor* tensor = trustformers_tensor_create(
    data,
    shape,
    2,  // ndim
    TRUSTFORMERS_DTYPE_F32
);

// Get shape
size_t ndim;
const size_t* tensor_shape = trustformers_tensor_shape(tensor, &ndim);

// Access data
const float* tensor_data = (const float*)trustformers_tensor_data(tensor);

// Free
trustformers_tensor_free(tensor);
```

---

### Hardware Acceleration

#### CUDA API

**NVIDIA GPU support**

- ✅ **Functions**
  - `trustformers_cuda_is_available()` - Check CUDA availability
  - `trustformers_cuda_device_count()` - Get GPU count
  - `trustformers_cuda_set_device()` - Set active GPU
  - `trustformers_cuda_synchronize()` - Synchronize device

**Example:**
```c
// Check CUDA availability
if (trustformers_cuda_is_available()) {
    int device_count = trustformers_cuda_device_count();
    printf("Found %d CUDA devices\n", device_count);

    // Set device
    trustformers_cuda_set_device(0);

    // Load model on GPU
    TrustformersModel* model = NULL;
    trustformers_model_load_on_device("gpt2", "cuda:0", &model);
}
```

---

#### ROCm API

**AMD GPU support**

- ✅ **Functions**
  - `trustformers_rocm_is_available()` - Check ROCm availability
  - `trustformers_rocm_device_count()` - Get GPU count
  - `trustformers_rocm_set_device()` - Set active GPU

---

#### Metal API

**Apple Silicon support**

- ✅ **Functions**
  - `trustformers_metal_is_available()` - Check Metal availability
  - `trustformers_metal_set_device()` - Set Metal device

---

#### TPU API

**Google TPU support**

- ✅ **Functions**
  - `trustformers_tpu_is_available()` - Check TPU availability
  - `trustformers_tpu_initialize()` - Initialize TPU
  - `trustformers_tpu_finalize()` - Cleanup TPU

---

### Error Handling

#### Error Management

**Comprehensive error handling**

- ✅ **Error Codes**
  - `TRUSTFORMERS_SUCCESS` - Success (0)
  - `TRUSTFORMERS_ERROR_INVALID_ARGUMENT` - Invalid argument
  - `TRUSTFORMERS_ERROR_OUT_OF_MEMORY` - Out of memory
  - `TRUSTFORMERS_ERROR_FILE_NOT_FOUND` - File not found
  - `TRUSTFORMERS_ERROR_MODEL_LOAD_FAILED` - Model load failed
  - `TRUSTFORMERS_ERROR_INFERENCE_FAILED` - Inference failed

- ✅ **Functions**
  - `trustformers_error_message()` - Get error message
  - `trustformers_last_error()` - Get last error code
  - `trustformers_clear_error()` - Clear error state

**Example:**
```c
TrustformersError error = trustformers_model_load("invalid-path", &model);
if (error != TRUSTFORMERS_SUCCESS) {
    const char* msg = trustformers_error_message(error);
    fprintf(stderr, "Error %d: %s\n", error, msg);
}
```

---

### Memory Management

#### Memory Safety

**Safe memory management for C**

- ✅ **Functions**
  - `trustformers_malloc()` - Allocate memory
  - `trustformers_free()` - Free memory
  - `trustformers_ref_count()` - Get reference count
  - `trustformers_retain()` - Increment reference
  - `trustformers_release()` - Decrement reference

- ✅ **Features**
  - Reference counting
  - Automatic cleanup
  - Memory leak detection (debug mode)
  - Thread-safe allocation

---

### Thread Safety

#### Concurrency Support

**Thread-safe operations**

- ✅ **Features**
  - Thread-safe model loading
  - Concurrent inference
  - Lock-free operations where possible
  - Thread-local error handling

**Example:**
```c
// Thread-safe inference
void* inference_thread(void* arg) {
    TrustformersModel* model = (TrustformersModel*)arg;

    // Each thread can use the same model safely
    TrustformersTensor* output = NULL;
    trustformers_model_forward(model, input, shape, ndim, &output);

    trustformers_tensor_free(output);
    return NULL;
}
```

---

## Known Limitations

- C API may not expose all Rust features
- Manual memory management required
- Error handling via return codes (no exceptions)
- Some operations may allocate on heap
- ABI stability requires careful versioning

---

## Future Enhancements

### High Priority
- Enhanced error messages
- More tensor operations
- Streaming inference support
- Async C API

### Performance
- Zero-copy operations where possible
- Better memory pooling
- Reduced allocations

### Features
- More hardware backends
- Plugin system
- Configuration API
- Profiling hooks

---

## Development Guidelines

### Code Standards
- **C Compatibility:** C89 compatible headers
- **Safety:** All pointers checked for null
- **Naming:** Prefix all symbols with `trustformers_`
- **Documentation:** Doxygen-style comments

### Build Commands

```bash
# Build C library
cargo build --release -p trustformers-c

# Generate C headers
cbindgen --config cbindgen.toml --crate trustformers-c --output trustformers.h

# Build example
cd examples/c
gcc -o example example.c -ltrustformers -L../../target/release

# Run example
./example
```

---

## Header Example

```c
/**
 * @file trustformers.h
 * @brief TrustformeRS C API
 */

#ifndef TRUSTFORMERS_H
#define TRUSTFORMERS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/** Error codes */
typedef enum {
    TRUSTFORMERS_SUCCESS = 0,
    TRUSTFORMERS_ERROR_INVALID_ARGUMENT = 1,
    TRUSTFORMERS_ERROR_OUT_OF_MEMORY = 2,
    // ... more error codes
} TrustformersError;

/** Opaque model handle */
typedef struct TrustformersModel TrustformersModel;

/** Opaque tensor handle */
typedef struct TrustformersTensor TrustformersTensor;

/**
 * Load a model from path
 * @param path Model path
 * @param model Output model pointer
 * @return Error code
 */
TrustformersError trustformers_model_load(
    const char* path,
    TrustformersModel** model
);

/**
 * Free a model
 * @param model Model to free
 */
void trustformers_model_free(TrustformersModel* model);

#ifdef __cplusplus
}
#endif

#endif /* TRUSTFORMERS_H */
```

---

## Usage Examples

### C Example

```c
#include "trustformers.h"
#include <stdio.h>

int main() {
    // Load model
    TrustformersModel* model = NULL;
    TrustformersError error = trustformers_model_load("gpt2", &model);

    if (error != TRUSTFORMERS_SUCCESS) {
        fprintf(stderr, "Failed to load model: %s\n",
                trustformers_error_message(error));
        return 1;
    }

    // Create input
    float input_data[] = {1.0, 2.0, 3.0};
    size_t shape[] = {1, 3};

    // Run inference
    TrustformersTensor* output = NULL;
    error = trustformers_model_forward(model, input_data, shape, 2, &output);

    if (error != TRUSTFORMERS_SUCCESS) {
        fprintf(stderr, "Inference failed: %s\n",
                trustformers_error_message(error));
        trustformers_model_free(model);
        return 1;
    }

    // Use output...

    // Cleanup
    trustformers_tensor_free(output);
    trustformers_model_free(model);

    return 0;
}
```

### C++ Example

```cpp
#include "trustformers.h"
#include <memory>
#include <stdexcept>

class Model {
    std::unique_ptr<TrustformersModel, decltype(&trustformers_model_free)> model_;

public:
    Model(const char* path)
        : model_(nullptr, trustformers_model_free) {
        TrustformersModel* raw_model = nullptr;
        auto error = trustformers_model_load(path, &raw_model);
        if (error != TRUSTFORMERS_SUCCESS) {
            throw std::runtime_error(trustformers_error_message(error));
        }
        model_.reset(raw_model);
    }

    // Use model...
};

int main() {
    try {
        Model model("gpt2");
        // Use model...
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
```

---

**Last Updated:** Refactored for alpha.1 release
**Status:** Production-ready C API
**ABI:** Stable binary interface
**Platforms:** Linux, macOS, Windows
