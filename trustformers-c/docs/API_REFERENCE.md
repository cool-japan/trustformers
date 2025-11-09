# TrustformeRS-C API Reference

This document provides a comprehensive reference for the TrustformeRS-C API, including all functions, data structures, and usage examples.

## Table of Contents

1. [Overview](#overview)
2. [Core API](#core-api)
3. [Model API](#model-api)
4. [Tokenizer API](#tokenizer-api)
5. [Pipeline API](#pipeline-api)
6. [Memory Management](#memory-management)
7. [Performance Optimization](#performance-optimization)
8. [CUDA Backend](#cuda-backend)
9. [HTTP Server](#http-server)
10. [Error Handling](#error-handling)
11. [Language Bindings](#language-bindings)
12. [Examples](#examples)

## Overview

TrustformeRS-C provides a C-compatible API for high-performance transformer model inference and serving. The library is designed for production use with comprehensive error handling, memory management, and cross-platform support.

### Key Features

- **High Performance**: Optimized for CPU and GPU inference
- **Memory Safe**: Comprehensive memory management and leak detection
- **Cross-Platform**: Support for Linux, macOS, and Windows
- **Language Bindings**: Native bindings for Go, Python, and JavaScript/Node.js
- **Production Ready**: Built-in HTTP server for model serving
- **Extensible**: Modular architecture with feature flags

### Library Initialization

All applications must initialize the library before use:

```c
#include "trustformers_c.h"

int main() {
    // Initialize the library
    TrustformersError error = trustformers_init();
    if (error != TRUSTFORMERS_SUCCESS) {
        fprintf(stderr, "Failed to initialize TrustformeRS\n");
        return 1;
    }
    
    // Use the library...
    
    // Cleanup before exit
    trustformers_cleanup();
    return 0;
}
```

## Core API

### Data Types

#### TrustformersError

Error codes returned by all API functions:

```c
typedef enum {
    TRUSTFORMERS_SUCCESS = 0,
    TRUSTFORMERS_NULL_POINTER = 1,
    TRUSTFORMERS_INVALID_PARAMETER = 2,
    TRUSTFORMERS_RUNTIME_ERROR = 3,
    TRUSTFORMERS_SERIALIZATION_ERROR = 4,
    TRUSTFORMERS_MEMORY_ERROR = 5,
    TRUSTFORMERS_IO_ERROR = 6,
    TRUSTFORMERS_NOT_IMPLEMENTED = 7,
    TRUSTFORMERS_UNKNOWN_ERROR = 8,
} TrustformersError;
```

#### TrustformersMemoryUsage

Memory usage statistics:

```c
typedef struct {
    uint64_t total_memory_bytes;
    uint64_t peak_memory_bytes;
    uint64_t allocated_models;
    uint64_t allocated_tokenizers;
    uint64_t allocated_pipelines;
    uint64_t allocated_tensors;
} TrustformersMemoryUsage;
```

#### TrustformersBuildInfo

Build and version information:

```c
typedef struct {
    char* version;
    char* features;
    char* build_date;
    char* target;
} TrustformersBuildInfo;
```

### Core Functions

#### trustformers_init

Initialize the TrustformeRS library.

```c
TrustformersError trustformers_init(void);
```

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

**Example**:
```c
TrustformersError error = trustformers_init();
if (error != TRUSTFORMERS_SUCCESS) {
    // Handle initialization error
}
```

#### trustformers_cleanup

Cleanup and shutdown the TrustformeRS library.

```c
TrustformersError trustformers_cleanup(void);
```

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

#### trustformers_version

Get the library version string.

```c
const char* trustformers_version(void);
```

**Returns**: Version string (e.g., "0.1.0").

#### trustformers_build_info

Get detailed build information.

```c
TrustformersError trustformers_build_info(TrustformersBuildInfo* info);
```

**Parameters**:
- `info`: Pointer to `TrustformersBuildInfo` structure to fill

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

**Example**:
```c
TrustformersBuildInfo info;
TrustformersError error = trustformers_build_info(&info);
if (error == TRUSTFORMERS_SUCCESS) {
    printf("Version: %s\n", info.version);
    printf("Features: %s\n", info.features);
    printf("Build Date: %s\n", info.build_date);
    printf("Target: %s\n", info.target);
    
    // Free allocated strings
    trustformers_free_string(info.version);
    trustformers_free_string(info.features);
    trustformers_free_string(info.build_date);
    trustformers_free_string(info.target);
}
```

#### trustformers_has_feature

Check if a specific feature is available.

```c
int trustformers_has_feature(const char* feature);
```

**Parameters**:
- `feature`: Feature name to check (e.g., "cuda", "serving")

**Returns**: 1 if feature is available, 0 otherwise.

**Example**:
```c
if (trustformers_has_feature("cuda")) {
    printf("CUDA support is available\n");
}
```

#### trustformers_set_log_level

Set the logging level.

```c
TrustformersError trustformers_set_log_level(int level);
```

**Parameters**:
- `level`: Log level (0=off, 1=error, 2=warn, 3=info, 4=debug, 5=trace)

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

#### trustformers_free_string

Free a string allocated by the library.

```c
void trustformers_free_string(char* ptr);
```

**Parameters**:
- `ptr`: Pointer to string to free

## Model API

### Functions

#### trustformers_load_model_from_hub

Load a model from Hugging Face Hub.

```c
void* trustformers_load_model_from_hub(const char* model_name, TrustformersError* error);
```

**Parameters**:
- `model_name`: Name of the model to load (e.g., "gpt2", "bert-base-uncased")
- `error`: Pointer to error code variable

**Returns**: Model handle on success, NULL on failure.

**Example**:
```c
TrustformersError error;
void* model = trustformers_load_model_from_hub("gpt2", &error);
if (error != TRUSTFORMERS_SUCCESS) {
    fprintf(stderr, "Failed to load model\n");
    return 1;
}

// Use model...

// Free model when done
trustformers_model_free(model);
```

#### trustformers_load_model_from_path

Load a model from a local path.

```c
void* trustformers_load_model_from_path(const char* model_path, TrustformersError* error);
```

**Parameters**:
- `model_path`: Path to model files
- `error`: Pointer to error code variable

**Returns**: Model handle on success, NULL on failure.

#### trustformers_model_free

Free a model and its resources.

```c
TrustformersError trustformers_model_free(void* model);
```

**Parameters**:
- `model`: Model handle to free

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

#### trustformers_model_get_info

Get model information.

```c
TrustformersError trustformers_model_get_info(void* model, char** info_json);
```

**Parameters**:
- `model`: Model handle
- `info_json`: Pointer to receive JSON string with model information

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

**Example**:
```c
char* info_json;
TrustformersError error = trustformers_model_get_info(model, &info_json);
if (error == TRUSTFORMERS_SUCCESS) {
    printf("Model info: %s\n", info_json);
    trustformers_free_string(info_json);
}
```

#### trustformers_model_set_quantization

Set model quantization level.

```c
TrustformersError trustformers_model_set_quantization(void* model, int quantization_bits);
```

**Parameters**:
- `model`: Model handle
- `quantization_bits`: Number of bits for quantization (8, 16, 32)

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

#### trustformers_model_validate

Validate model integrity.

```c
TrustformersError trustformers_model_validate(void* model, int* is_valid);
```

**Parameters**:
- `model`: Model handle
- `is_valid`: Pointer to receive validation result (1=valid, 0=invalid)

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

## Tokenizer API

### Functions

#### trustformers_load_tokenizer_from_hub

Load a tokenizer from Hugging Face Hub.

```c
void* trustformers_load_tokenizer_from_hub(const char* model_name, TrustformersError* error);
```

**Parameters**:
- `model_name`: Name of the tokenizer to load
- `error`: Pointer to error code variable

**Returns**: Tokenizer handle on success, NULL on failure.

#### trustformers_load_tokenizer_from_path

Load a tokenizer from a local path.

```c
void* trustformers_load_tokenizer_from_path(const char* tokenizer_path, TrustformersError* error);
```

**Parameters**:
- `tokenizer_path`: Path to tokenizer files
- `error`: Pointer to error code variable

**Returns**: Tokenizer handle on success, NULL on failure.

#### trustformers_tokenizer_free

Free a tokenizer and its resources.

```c
TrustformersError trustformers_tokenizer_free(void* tokenizer);
```

**Parameters**:
- `tokenizer`: Tokenizer handle to free

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

#### trustformers_tokenizer_encode

Encode text to tokens.

```c
TrustformersError trustformers_tokenizer_encode(void* tokenizer, const char* text, int** tokens, int* num_tokens);
```

**Parameters**:
- `tokenizer`: Tokenizer handle
- `text`: Text to encode
- `tokens`: Pointer to receive token array
- `num_tokens`: Pointer to receive number of tokens

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

**Example**:
```c
int* tokens;
int num_tokens;
TrustformersError error = trustformers_tokenizer_encode(
    tokenizer, "Hello, world!", &tokens, &num_tokens);
if (error == TRUSTFORMERS_SUCCESS) {
    printf("Encoded %d tokens\n", num_tokens);
    for (int i = 0; i < num_tokens; i++) {
        printf("Token %d: %d\n", i, tokens[i]);
    }
    trustformers_free_tokens(tokens);
}
```

#### trustformers_tokenizer_decode

Decode tokens to text.

```c
TrustformersError trustformers_tokenizer_decode(void* tokenizer, const int* tokens, int num_tokens, char** text);
```

**Parameters**:
- `tokenizer`: Tokenizer handle
- `tokens`: Array of tokens to decode
- `num_tokens`: Number of tokens
- `text`: Pointer to receive decoded text

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

#### trustformers_tokenizer_encode_batch

Encode multiple texts to tokens.

```c
TrustformersError trustformers_tokenizer_encode_batch(void* tokenizer, const char** texts, int num_texts, int*** tokens, int** num_tokens);
```

**Parameters**:
- `tokenizer`: Tokenizer handle
- `texts`: Array of texts to encode
- `num_texts`: Number of texts
- `tokens`: Pointer to receive array of token arrays
- `num_tokens`: Pointer to receive array of token counts

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

#### trustformers_tokenizer_get_vocab_size

Get tokenizer vocabulary size.

```c
TrustformersError trustformers_tokenizer_get_vocab_size(void* tokenizer, int* vocab_size);
```

**Parameters**:
- `tokenizer`: Tokenizer handle
- `vocab_size`: Pointer to receive vocabulary size

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

## Pipeline API

Pipelines provide high-level interfaces for common NLP tasks.

### Functions

#### trustformers_create_text_generation_pipeline

Create a text generation pipeline.

```c
void* trustformers_create_text_generation_pipeline(void* model, void* tokenizer, TrustformersError* error);
```

**Parameters**:
- `model`: Model handle
- `tokenizer`: Tokenizer handle
- `error`: Pointer to error code variable

**Returns**: Pipeline handle on success, NULL on failure.

#### trustformers_create_text_classification_pipeline

Create a text classification pipeline.

```c
void* trustformers_create_text_classification_pipeline(void* model, void* tokenizer, TrustformersError* error);
```

**Parameters**:
- `model`: Model handle
- `tokenizer`: Tokenizer handle
- `error`: Pointer to error code variable

**Returns**: Pipeline handle on success, NULL on failure.

#### trustformers_create_question_answering_pipeline

Create a question answering pipeline.

```c
void* trustformers_create_question_answering_pipeline(void* model, void* tokenizer, TrustformersError* error);
```

**Parameters**:
- `model`: Model handle
- `tokenizer`: Tokenizer handle
- `error`: Pointer to error code variable

**Returns**: Pipeline handle on success, NULL on failure.

#### trustformers_create_conversational_pipeline

Create a conversational AI pipeline.

```c
void* trustformers_create_conversational_pipeline(void* model, void* tokenizer, TrustformersError* error);
```

**Parameters**:
- `model`: Model handle
- `tokenizer`: Tokenizer handle
- `error`: Pointer to error code variable

**Returns**: Pipeline handle on success, NULL on failure.

#### trustformers_pipeline_free

Free a pipeline and its resources.

```c
TrustformersError trustformers_pipeline_free(void* pipeline);
```

**Parameters**:
- `pipeline`: Pipeline handle to free

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

#### trustformers_pipeline_generate_text

Generate text using a text generation pipeline.

```c
TrustformersError trustformers_pipeline_generate_text(void* pipeline, const char* prompt, char** generated_text);
```

**Parameters**:
- `pipeline`: Text generation pipeline handle
- `prompt`: Input prompt
- `generated_text`: Pointer to receive generated text

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

**Example**:
```c
char* generated_text;
TrustformersError error = trustformers_pipeline_generate_text(
    pipeline, "The future of AI is", &generated_text);
if (error == TRUSTFORMERS_SUCCESS) {
    printf("Generated: %s\n", generated_text);
    trustformers_free_string(generated_text);
}
```

#### trustformers_pipeline_generate_text_with_options

Generate text with custom options.

```c
TrustformersError trustformers_pipeline_generate_text_with_options(void* pipeline, const char* prompt, const char* options_json, char** generated_text);
```

**Parameters**:
- `pipeline`: Text generation pipeline handle
- `prompt`: Input prompt
- `options_json`: JSON string with generation options
- `generated_text`: Pointer to receive generated text

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

**Example**:
```c
const char* options = "{"
    "\"max_length\": 100,"
    "\"temperature\": 0.8,"
    "\"top_k\": 50,"
    "\"top_p\": 0.9"
"}";

char* generated_text;
TrustformersError error = trustformers_pipeline_generate_text_with_options(
    pipeline, "Once upon a time", options, &generated_text);
if (error == TRUSTFORMERS_SUCCESS) {
    printf("Generated: %s\n", generated_text);
    trustformers_free_string(generated_text);
}
```

#### trustformers_pipeline_classify_text

Classify text using a text classification pipeline.

```c
TrustformersError trustformers_pipeline_classify_text(void* pipeline, const char* text, char** classification_result);
```

**Parameters**:
- `pipeline`: Text classification pipeline handle
- `text`: Text to classify
- `classification_result`: Pointer to receive classification results (JSON)

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

#### trustformers_pipeline_answer_question

Answer a question using a question answering pipeline.

```c
TrustformersError trustformers_pipeline_answer_question(void* pipeline, const char* context, const char* question, char** answer);
```

**Parameters**:
- `pipeline`: Question answering pipeline handle
- `context`: Context text
- `question`: Question to answer
- `answer`: Pointer to receive answer (JSON)

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

## Memory Management

### Functions

#### trustformers_get_memory_usage

Get current memory usage statistics.

```c
TrustformersError trustformers_get_memory_usage(TrustformersMemoryUsage* usage);
```

**Parameters**:
- `usage`: Pointer to `TrustformersMemoryUsage` structure to fill

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

#### trustformers_memory_cleanup

Force memory cleanup and garbage collection.

```c
TrustformersError trustformers_memory_cleanup(void);
```

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

#### trustformers_set_memory_limits

Set memory limits and thresholds.

```c
TrustformersError trustformers_set_memory_limits(uint64_t max_memory_mb, uint64_t warning_threshold_mb);
```

**Parameters**:
- `max_memory_mb`: Maximum memory usage in MB
- `warning_threshold_mb`: Warning threshold in MB

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

#### trustformers_check_memory_leaks

Check for memory leaks and get report.

```c
TrustformersError trustformers_check_memory_leaks(char** leak_report);
```

**Parameters**:
- `leak_report`: Pointer to receive leak report (JSON)

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

## Performance Optimization

### Data Types

#### TrustformersOptimizationConfig

Performance optimization configuration:

```c
typedef struct {
    int enable_tracking;
    int enable_caching;
    int cache_size_mb;
    int num_threads;
    int enable_simd;
    int optimize_batch_size;
    int memory_optimization_level;
} TrustformersOptimizationConfig;
```

#### TrustformersPerformanceMetrics

Performance metrics:

```c
typedef struct {
    uint64_t total_operations;
    double avg_operation_time_ms;
    double min_operation_time_ms;
    double max_operation_time_ms;
    double cache_hit_rate;
    double performance_score;
    int num_optimization_hints;
    char* optimization_hints_json;
} TrustformersPerformanceMetrics;
```

### Functions

#### trustformers_get_performance_metrics

Get current performance metrics.

```c
TrustformersError trustformers_get_performance_metrics(TrustformersPerformanceMetrics* metrics);
```

**Parameters**:
- `metrics`: Pointer to `TrustformersPerformanceMetrics` structure to fill

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

#### trustformers_apply_optimizations

Apply performance optimizations.

```c
TrustformersError trustformers_apply_optimizations(const TrustformersOptimizationConfig* config);
```

**Parameters**:
- `config`: Pointer to optimization configuration

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

**Example**:
```c
TrustformersOptimizationConfig config = {
    .enable_tracking = 1,
    .enable_caching = 1,
    .cache_size_mb = 256,
    .num_threads = 4,
    .enable_simd = 1,
    .optimize_batch_size = 1,
    .memory_optimization_level = 2
};

TrustformersError error = trustformers_apply_optimizations(&config);
if (error == TRUSTFORMERS_SUCCESS) {
    printf("Optimizations applied successfully\n");
}
```

#### trustformers_start_profiling

Start a performance profiling session.

```c
TrustformersError trustformers_start_profiling(void);
```

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

#### trustformers_stop_profiling

Stop profiling and get report.

```c
TrustformersError trustformers_stop_profiling(char** report);
```

**Parameters**:
- `report`: Pointer to receive profiling report (JSON)

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

## CUDA Backend

CUDA backend functions are available when compiled with CUDA support.

### Functions

#### trustformers_cuda_init

Initialize CUDA backend.

```c
TrustformersError trustformers_cuda_init(void);
```

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

#### trustformers_cuda_is_available

Check if CUDA is available.

```c
int trustformers_cuda_is_available(void);
```

**Returns**: 1 if CUDA is available, 0 otherwise.

#### trustformers_cuda_get_device_count

Get number of CUDA devices.

```c
int trustformers_cuda_get_device_count(void);
```

**Returns**: Number of CUDA devices, -1 on error.

#### trustformers_cuda_get_device_info

Get CUDA device information.

```c
TrustformersError trustformers_cuda_get_device_info(int device_id, char** info_json);
```

**Parameters**:
- `device_id`: CUDA device ID
- `info_json`: Pointer to receive device information (JSON)

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

#### trustformers_cuda_set_device

Set current CUDA device.

```c
TrustformersError trustformers_cuda_set_device(int device_id);
```

**Parameters**:
- `device_id`: CUDA device ID to set as current

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

#### trustformers_cuda_get_memory_info

Get CUDA memory information.

```c
TrustformersError trustformers_cuda_get_memory_info(int device_id, uint64_t* free_memory, uint64_t* total_memory);
```

**Parameters**:
- `device_id`: CUDA device ID
- `free_memory`: Pointer to receive free memory in bytes
- `total_memory`: Pointer to receive total memory in bytes

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

#### trustformers_cuda_synchronize

Synchronize CUDA operations.

```c
TrustformersError trustformers_cuda_synchronize(void);
```

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

## HTTP Server

HTTP server functions are available when compiled with serving support.

### Functions

#### trustformers_http_server_create

Create HTTP server with default configuration.

```c
TrustformersError trustformers_http_server_create(char** server_id);
```

**Parameters**:
- `server_id`: Pointer to receive server ID string

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

#### trustformers_http_server_create_with_config

Create HTTP server with custom configuration.

```c
TrustformersError trustformers_http_server_create_with_config(const char* config_json, char** server_id);
```

**Parameters**:
- `config_json`: JSON configuration string
- `server_id`: Pointer to receive server ID string

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

**Example**:
```c
const char* config = "{"
    "\"host\": \"127.0.0.1\","
    "\"port\": 8080,"
    "\"max_connections\": 100,"
    "\"enable_cors\": true"
"}";

char* server_id;
TrustformersError error = trustformers_http_server_create_with_config(config, &server_id);
if (error == TRUSTFORMERS_SUCCESS) {
    printf("Server created with ID: %s\n", server_id);
}
```

#### trustformers_http_server_add_model

Add model endpoint to HTTP server.

```c
TrustformersError trustformers_http_server_add_model(const char* server_id, const char* endpoint_json);
```

**Parameters**:
- `server_id`: Server ID string
- `endpoint_json`: JSON endpoint configuration

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

#### trustformers_http_server_start

Start HTTP server.

```c
TrustformersError trustformers_http_server_start(const char* server_id);
```

**Parameters**:
- `server_id`: Server ID string

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

#### trustformers_http_server_stop

Stop HTTP server.

```c
TrustformersError trustformers_http_server_stop(const char* server_id);
```

**Parameters**:
- `server_id`: Server ID string

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

#### trustformers_http_server_get_metrics

Get HTTP server metrics.

```c
TrustformersError trustformers_http_server_get_metrics(const char* server_id, char** metrics_json);
```

**Parameters**:
- `server_id`: Server ID string
- `metrics_json`: Pointer to receive metrics (JSON)

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

#### trustformers_http_server_destroy

Destroy HTTP server.

```c
TrustformersError trustformers_http_server_destroy(const char* server_id);
```

**Parameters**:
- `server_id`: Server ID string

**Returns**: `TRUSTFORMERS_SUCCESS` on success, error code on failure.

## Error Handling

### Best Practices

1. **Always Check Return Values**: Every function returns a `TrustformersError` code that should be checked.

2. **Use Proper Error Handling**: Handle errors appropriately in your application.

3. **Free Allocated Memory**: Always free strings and structures allocated by the library.

4. **Initialize Before Use**: Call `trustformers_init()` before using any other functions.

5. **Cleanup on Exit**: Call `trustformers_cleanup()` before program termination.

### Example Error Handling

```c
#include "trustformers_c.h"
#include <stdio.h>
#include <stdlib.h>

const char* error_to_string(TrustformersError error) {
    switch (error) {
        case TRUSTFORMERS_SUCCESS: return "Success";
        case TRUSTFORMERS_NULL_POINTER: return "Null pointer";
        case TRUSTFORMERS_INVALID_PARAMETER: return "Invalid parameter";
        case TRUSTFORMERS_RUNTIME_ERROR: return "Runtime error";
        case TRUSTFORMERS_SERIALIZATION_ERROR: return "Serialization error";
        case TRUSTFORMERS_MEMORY_ERROR: return "Memory error";
        case TRUSTFORMERS_IO_ERROR: return "I/O error";
        case TRUSTFORMERS_NOT_IMPLEMENTED: return "Not implemented";
        case TRUSTFORMERS_UNKNOWN_ERROR: return "Unknown error";
        default: return "Unrecognized error";
    }
}

int main() {
    // Initialize library
    TrustformersError error = trustformers_init();
    if (error != TRUSTFORMERS_SUCCESS) {
        fprintf(stderr, "Failed to initialize TrustformeRS: %s\n", error_to_string(error));
        return 1;
    }

    // Load model with error handling
    void* model = trustformers_load_model_from_hub("gpt2", &error);
    if (error != TRUSTFORMERS_SUCCESS) {
        fprintf(stderr, "Failed to load model: %s\n", error_to_string(error));
        trustformers_cleanup();
        return 1;
    }

    // Use model...
    
    // Cleanup
    trustformers_model_free(model);
    trustformers_cleanup();
    return 0;
}
```

## Language Bindings

TrustformeRS-C provides native bindings for multiple programming languages.

### Go Bindings

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

    // Load model
    model, err := tf.LoadModelFromHub("gpt2")
    if err != nil {
        log.Fatal(err)
    }
    defer model.Free()

    fmt.Printf("Model loaded successfully\n")
}
```

### Python Bindings

```python
import trustformers_c

# Initialize library
tf = trustformers_c.TrustformersC()

# Load model
model = tf.load_model_from_hub("gpt2")

# Load tokenizer
tokenizer = tf.load_tokenizer_from_hub("gpt2")

# Create pipeline
pipeline = tf.create_text_generation_pipeline(model, tokenizer)

# Generate text
result = pipeline.generate_text("The future of AI is")
print(f"Generated: {result}")
```

### JavaScript/Node.js Bindings

```javascript
const trustformers = require('trustformers-c');

async function main() {
    // Initialize library
    const tf = new trustformers.TrustformeRS();
    
    // Load model and tokenizer
    const model = await tf.loadModelFromHub('gpt2');
    const tokenizer = await tf.loadTokenizerFromHub('gpt2');
    
    // Create pipeline
    const pipeline = tf.createTextGenerationPipeline(model, tokenizer);
    
    // Generate text
    const result = await pipeline.generateText('The future of AI is');
    console.log('Generated:', result);
    
    // Cleanup
    tf.cleanup();
}

main().catch(console.error);
```

## Examples

### Complete C Example

```c
#include "trustformers_c.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    // Initialize library
    TrustformersError error = trustformers_init();
    if (error != TRUSTFORMERS_SUCCESS) {
        fprintf(stderr, "Failed to initialize TrustformeRS\n");
        return 1;
    }

    printf("TrustformeRS initialized successfully\n");
    printf("Version: %s\n", trustformers_version());

    // Check features
    if (trustformers_has_feature("cuda")) {
        printf("CUDA support is available\n");
    }

    // Load model and tokenizer
    void* model = trustformers_load_model_from_hub("gpt2", &error);
    if (error != TRUSTFORMERS_SUCCESS) {
        fprintf(stderr, "Failed to load model\n");
        trustformers_cleanup();
        return 1;
    }

    void* tokenizer = trustformers_load_tokenizer_from_hub("gpt2", &error);
    if (error != TRUSTFORMERS_SUCCESS) {
        fprintf(stderr, "Failed to load tokenizer\n");
        trustformers_model_free(model);
        trustformers_cleanup();
        return 1;
    }

    // Create text generation pipeline
    void* pipeline = trustformers_create_text_generation_pipeline(model, tokenizer, &error);
    if (error != TRUSTFORMERS_SUCCESS) {
        fprintf(stderr, "Failed to create pipeline\n");
        trustformers_tokenizer_free(tokenizer);
        trustformers_model_free(model);
        trustformers_cleanup();
        return 1;
    }

    // Generate text
    char* generated_text;
    error = trustformers_pipeline_generate_text(pipeline, "The future of AI is", &generated_text);
    if (error == TRUSTFORMERS_SUCCESS) {
        printf("Generated text: %s\n", generated_text);
        trustformers_free_string(generated_text);
    } else {
        fprintf(stderr, "Failed to generate text\n");
    }

    // Get memory usage
    TrustformersMemoryUsage usage;
    error = trustformers_get_memory_usage(&usage);
    if (error == TRUSTFORMERS_SUCCESS) {
        printf("Memory usage: %lu bytes\n", usage.total_memory_bytes);
        printf("Models: %lu, Tokenizers: %lu, Pipelines: %lu\n",
               usage.allocated_models, usage.allocated_tokenizers, usage.allocated_pipelines);
    }

    // Cleanup
    trustformers_pipeline_free(pipeline);
    trustformers_tokenizer_free(tokenizer);
    trustformers_model_free(model);
    trustformers_cleanup();

    printf("TrustformeRS cleanup completed\n");
    return 0;
}
```

### HTTP Server Example

```c
#include "trustformers_c.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
    // Initialize library
    TrustformersError error = trustformers_init();
    if (error != TRUSTFORMERS_SUCCESS) {
        fprintf(stderr, "Failed to initialize TrustformeRS\n");
        return 1;
    }

    // Create HTTP server
    char* server_id;
    error = trustformers_http_server_create(&server_id);
    if (error != TRUSTFORMERS_SUCCESS) {
        fprintf(stderr, "Failed to create HTTP server\n");
        trustformers_cleanup();
        return 1;
    }

    printf("HTTP server created with ID: %s\n", server_id);

    // Add model endpoint
    const char* endpoint_config = "{"
        "\"name\": \"gpt2-generator\","
        "\"model_path\": \"/models/gpt2\","
        "\"endpoint_path\": \"/generate\","
        "\"max_batch_size\": 16,"
        "\"timeout_ms\": 30000"
    "}";

    error = trustformers_http_server_add_model(server_id, endpoint_config);
    if (error != TRUSTFORMERS_SUCCESS) {
        fprintf(stderr, "Failed to add model endpoint\n");
        trustformers_http_server_destroy(server_id);
        trustformers_free_string(server_id);
        trustformers_cleanup();
        return 1;
    }

    // Start server
    error = trustformers_http_server_start(server_id);
    if (error != TRUSTFORMERS_SUCCESS) {
        fprintf(stderr, "Failed to start HTTP server\n");
        trustformers_http_server_destroy(server_id);
        trustformers_free_string(server_id);
        trustformers_cleanup();
        return 1;
    }

    printf("HTTP server started successfully\n");
    printf("Server is running. Press Ctrl+C to stop.\n");

    // Run for 30 seconds (in real application, handle signals)
    sleep(30);

    // Stop server
    trustformers_http_server_stop(server_id);
    trustformers_http_server_destroy(server_id);
    trustformers_free_string(server_id);
    trustformers_cleanup();

    printf("Server stopped and cleaned up\n");
    return 0;
}
```

### CUDA Example

```c
#include "trustformers_c.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    // Initialize library
    TrustformersError error = trustformers_init();
    if (error != TRUSTFORMERS_SUCCESS) {
        fprintf(stderr, "Failed to initialize TrustformeRS\n");
        return 1;
    }

    // Check CUDA availability
    if (!trustformers_cuda_is_available()) {
        printf("CUDA is not available\n");
        trustformers_cleanup();
        return 0;
    }

    // Initialize CUDA
    error = trustformers_cuda_init();
    if (error != TRUSTFORMERS_SUCCESS) {
        fprintf(stderr, "Failed to initialize CUDA\n");
        trustformers_cleanup();
        return 1;
    }

    // Get device count
    int device_count = trustformers_cuda_get_device_count();
    printf("CUDA devices available: %d\n", device_count);

    // Get device information
    for (int i = 0; i < device_count; i++) {
        char* device_info;
        error = trustformers_cuda_get_device_info(i, &device_info);
        if (error == TRUSTFORMERS_SUCCESS) {
            printf("Device %d info: %s\n", i, device_info);
            trustformers_free_string(device_info);
        }
    }

    // Set device 0 as current
    if (device_count > 0) {
        error = trustformers_cuda_set_device(0);
        if (error == TRUSTFORMERS_SUCCESS) {
            printf("Set device 0 as current\n");
            
            // Get memory info
            uint64_t free_memory, total_memory;
            error = trustformers_cuda_get_memory_info(0, &free_memory, &total_memory);
            if (error == TRUSTFORMERS_SUCCESS) {
                printf("Device 0 memory: %lu MB free / %lu MB total\n",
                       free_memory / (1024*1024), total_memory / (1024*1024));
            }
        }
    }

    // Cleanup
    trustformers_cleanup();
    return 0;
}
```

This completes the comprehensive API reference for TrustformeRS-C. The API provides a complete solution for transformer model inference with support for multiple programming languages, GPU acceleration, HTTP serving, and production-ready features like memory management and performance optimization.