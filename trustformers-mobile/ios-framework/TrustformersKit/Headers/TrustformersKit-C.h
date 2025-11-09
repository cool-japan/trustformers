//
//  TrustformersKit-C.h
//  TrustformersKit
//
//  C Interface for Rust FFI Bridge
//  Copyright (c) 2024 TrustformeRS. All rights reserved.
//

#ifndef TrustformersKit_C_h
#define TrustformersKit_C_h

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// MARK: - Enums

/// Mobile platform types
typedef enum {
    TFKPlatformIOS = 0,
    TFKPlatformAndroid = 1,
    TFKPlatformGeneric = 2
} TFKPlatform;

/// Mobile backend types
typedef enum {
    TFKBackendCPU = 0,
    TFKBackendCoreML = 1,
    TFKBackendNNAPI = 2,
    TFKBackendGPU = 3,
    TFKBackendCustom = 4
} TFKBackend;

/// Memory optimization levels
typedef enum {
    TFKMemoryOptimizationMinimal = 0,
    TFKMemoryOptimizationBalanced = 1,
    TFKMemoryOptimizationMaximum = 2
} TFKMemoryOptimization;

/// Quantization schemes
typedef enum {
    TFKQuantizationInt8 = 0,
    TFKQuantizationInt4 = 1,
    TFKQuantizationFP16 = 2,
    TFKQuantizationDynamic = 3
} TFKQuantizationScheme;

/// Thermal states
typedef enum {
    TFKThermalStateNominal = 0,
    TFKThermalStateFair = 1,
    TFKThermalStateSerious = 2,
    TFKThermalStateCritical = 3
} TFKThermalState;

/// Log levels
typedef enum {
    TFKLogLevelTrace = 0,
    TFKLogLevelDebug = 1,
    TFKLogLevelInfo = 2,
    TFKLogLevelWarn = 3,
    TFKLogLevelError = 4
} TFKLogLevel;

// Opaque types
typedef struct TFKEngine TFKEngine;
typedef struct TFKTensorC TFKTensorC;

// Configuration structures
typedef struct {
    uint32_t platform;          // 0=iOS, 1=Android, 2=Generic
    uint32_t backend;           // 0=CPU, 1=CoreML, 2=NNAPI, 3=GPU, 4=Custom
    uint32_t memory_optimization; // 0=Minimal, 1=Balanced, 2=Maximum
    uint32_t max_memory_mb;
    bool use_fp16;
    uint32_t num_threads;
    bool enable_batching;
    uint32_t max_batch_size;
} CTrustformersConfig;

// Legacy compatibility
typedef CTrustformersConfig TFKConfigC;

// Inference result structure
typedef struct {
    float* data;
    size_t length;
    uint32_t memory_used_mb;
    bool success;
    const char* error_message;
} CTrustformersInferenceResult;

// Legacy compatibility
typedef CTrustformersInferenceResult TFKInferenceResultC;

// Tensor structure
typedef struct {
    float* data;
    size_t* shape;
    size_t shape_len;
    size_t data_len;
} CTrustformersTensor;

// Device info structure
typedef struct {
    const char* device_model;
    const char* ios_version;
    uint32_t total_memory_mb;
    uint32_t cpu_cores;
    bool has_gpu;
    bool has_coreml;
    bool has_neural_engine;
    uint32_t thermal_state; // 0=Nominal, 1=Fair, 2=Serious, 3=Critical
} TFKDeviceInfoC;

// Performance stats structure
typedef struct {
    uint64_t total_inferences;
    double avg_inference_time_ms;
    double min_inference_time_ms;
    double max_inference_time_ms;
    uint32_t current_memory_mb;
    uint32_t peak_memory_mb;
    float avg_cpu_usage;
    float avg_gpu_usage;
} TFKPerformanceStatsC;

// Engine management
TFKEngine* tfk_create_engine(const CTrustformersConfig* config);
void tfk_destroy_engine(TFKEngine* engine);
bool tfk_load_model(TFKEngine* engine, const char* model_path);
bool tfk_unload_model(TFKEngine* engine);

// Inference
CTrustformersInferenceResult* tfk_inference(TFKEngine* engine, const CTrustformersTensor* input_tensor);
void tfk_free_inference_result(CTrustformersInferenceResult* result);

// Additional functions from Rust implementation
const char* tfk_get_last_error(void);
uint32_t tfk_get_current_memory_usage(TFKEngine* engine);
void tfk_set_log_callback(void (*callback)(uint32_t level, const char* message));

// Tensor management
TFKTensorC* tfk_create_tensor(const float* data, const int64_t* shape, size_t rank);
void tfk_destroy_tensor(TFKTensorC* tensor);
size_t tfk_tensor_element_count(const TFKTensorC* tensor);
const float* tfk_tensor_data(const TFKTensorC* tensor);
bool tfk_tensor_get_shape(const TFKTensorC* tensor, int64_t* shape, size_t* rank);

// Inference
TFKInferenceResultC* tfk_inference(TFKEngine* engine, const TFKTensorC* input);
void tfk_free_inference_result(TFKInferenceResultC* result);

// Batch inference
TFKInferenceResultC** tfk_batch_inference(
    TFKEngine* engine,
    const TFKTensorC** inputs,
    size_t batch_size
);
void tfk_free_batch_results(TFKInferenceResultC** results, size_t batch_size);

// Device information
TFKDeviceInfoC* tfk_get_device_info(void);
void tfk_free_device_info(TFKDeviceInfoC* info);
TFKConfigC* tfk_get_recommended_config(void);
void tfk_free_config(TFKConfigC* config);

// Performance monitoring
TFKPerformanceStatsC* tfk_get_performance_stats(TFKEngine* engine);
void tfk_free_performance_stats(TFKPerformanceStatsC* stats);
void tfk_reset_performance_stats(TFKEngine* engine);

// Configuration helpers
TFKConfigC* tfk_create_default_config(void);
TFKConfigC* tfk_create_optimized_config(void);
TFKConfigC* tfk_create_low_memory_config(void);
bool tfk_validate_config(const TFKConfigC* config, char* error_buffer, size_t buffer_size);

// Memory management
void tfk_set_memory_limit(TFKEngine* engine, uint32_t limit_mb);
uint32_t tfk_get_current_memory_usage(TFKEngine* engine);
void tfk_enable_memory_warnings(TFKEngine* engine, bool enable);

// Thermal management
void tfk_set_thermal_throttling(TFKEngine* engine, bool enable);
uint32_t tfk_get_thermal_state(void);

// Logging
typedef void (*TFKLogCallback)(int level, const char* message);
void tfk_set_log_callback(TFKLogCallback callback);
void tfk_set_log_level(int level); // 0=Verbose, 1=Debug, 2=Info, 3=Warning, 4=Error

// Version information
const char* tfk_get_version(void);
uint32_t tfk_get_version_major(void);
uint32_t tfk_get_version_minor(void);
uint32_t tfk_get_version_patch(void);

// Error handling
const char* tfk_get_last_error(void);
void tfk_clear_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* TrustformersKit_C_h */