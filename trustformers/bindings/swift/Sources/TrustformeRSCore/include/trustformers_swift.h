#ifndef TRUSTFORMERS_SWIFT_H
#define TRUSTFORMERS_SWIFT_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct TrustformersHandle TrustformersHandle;
typedef struct ModelHandle ModelHandle;
typedef struct TokenizerHandle TokenizerHandle;
typedef struct PipelineHandle PipelineHandle;

// Error handling
typedef enum {
    TRUSTFORMERS_OK = 0,
    TRUSTFORMERS_ERROR_INVALID_ARGUMENT = 1,
    TRUSTFORMERS_ERROR_MODEL_NOT_FOUND = 2,
    TRUSTFORMERS_ERROR_TOKENIZER_ERROR = 3,
    TRUSTFORMERS_ERROR_INFERENCE_ERROR = 4,
    TRUSTFORMERS_ERROR_MEMORY_ERROR = 5,
    TRUSTFORMERS_ERROR_IO_ERROR = 6,
    TRUSTFORMERS_ERROR_UNKNOWN = 999
} TrustformersErrorCode;

typedef struct {
    TrustformersErrorCode code;
    char* message;
} TrustformersError;

// Memory management
typedef struct {
    void* data;
    size_t length;
} TrustformersBuffer;

typedef struct {
    float* data;
    size_t* shape;
    size_t ndim;
} TrustformersTensor;

// Configuration structures
typedef struct {
    bool use_gpu;
    char* device;
    int num_threads;
    bool enable_logging;
    char* cache_dir;
} TrustformersConfig;

typedef struct {
    char* model_type;
    char* model_path;
    char* tokenizer_path;
    bool use_fast_tokenizer;
    int max_length;
    bool do_lower_case;
} ModelConfig;

typedef struct {
    char* task;
    char* model_id;
    char* device;
    bool use_auth_token;
    char* revision;
    float temperature;
    int max_new_tokens;
    int top_k;
    float top_p;
    bool do_sample;
} PipelineConfig;

// Core functions
TrustformersHandle* trustformers_init(const TrustformersConfig* config, TrustformersError* error);
void trustformers_free(TrustformersHandle* handle);
char* trustformers_get_version(void);
bool trustformers_is_gpu_available(void);

// Model functions
ModelHandle* model_load(TrustformersHandle* handle, const ModelConfig* config, TrustformersError* error);
void model_free(ModelHandle* model);
TrustformersTensor* model_forward(ModelHandle* model, const TrustformersTensor* input, TrustformersError* error);
char** model_generate(ModelHandle* model, const char* input, int max_length, TrustformersError* error);
size_t model_get_vocab_size(ModelHandle* model);
char* model_get_model_type(ModelHandle* model);

// Tokenizer functions
TokenizerHandle* tokenizer_load(TrustformersHandle* handle, const char* tokenizer_path, TrustformersError* error);
void tokenizer_free(TokenizerHandle* tokenizer);
int32_t* tokenizer_encode(TokenizerHandle* tokenizer, const char* text, size_t* length, TrustformersError* error);
char* tokenizer_decode(TokenizerHandle* tokenizer, const int32_t* tokens, size_t length, TrustformersError* error);
size_t tokenizer_get_vocab_size(TokenizerHandle* tokenizer);
char** tokenizer_get_vocab(TokenizerHandle* tokenizer, size_t* size);

// Pipeline functions
PipelineHandle* pipeline_create(TrustformersHandle* handle, const PipelineConfig* config, TrustformersError* error);
void pipeline_free(PipelineHandle* pipeline);
char** pipeline_text_generation(PipelineHandle* pipeline, const char* input, TrustformersError* error);
float* pipeline_text_classification(PipelineHandle* pipeline, const char* input, size_t* num_labels, TrustformersError* error);
char* pipeline_question_answering(PipelineHandle* pipeline, const char* question, const char* context, TrustformersError* error);
char* pipeline_summarization(PipelineHandle* pipeline, const char* input, TrustformersError* error);
char* pipeline_translation(PipelineHandle* pipeline, const char* input, const char* target_lang, TrustformersError* error);

// Utility functions
void trustformers_free_string(char* str);
void trustformers_free_string_array(char** array, size_t length);
void trustformers_free_tensor(TrustformersTensor* tensor);
void trustformers_free_buffer(TrustformersBuffer* buffer);
void trustformers_free_error(TrustformersError* error);

// Async functions (callbacks)
typedef void (*TrustformersCallback)(void* user_data, const char* result, TrustformersError* error);

void pipeline_text_generation_async(PipelineHandle* pipeline, const char* input, 
                                   TrustformersCallback callback, void* user_data);
void model_generate_async(ModelHandle* model, const char* input, int max_length,
                         TrustformersCallback callback, void* user_data);

// Advanced features
typedef struct {
    bool enable_streaming;
    int chunk_size;
    float timeout_seconds;
    bool enable_caching;
} AdvancedConfig;

PipelineHandle* pipeline_create_advanced(TrustformersHandle* handle, const PipelineConfig* config, 
                                        const AdvancedConfig* advanced, TrustformersError* error);

// Metal/CoreML specific functions (iOS/macOS)
#if defined(__APPLE__)
bool trustformers_is_metal_available(void);
bool trustformers_is_coreml_available(void);
bool trustformers_enable_metal_backend(TrustformersHandle* handle, TrustformersError* error);
bool trustformers_enable_coreml_backend(TrustformersHandle* handle, TrustformersError* error);
#endif

#ifdef __cplusplus
}
#endif

#endif // TRUSTFORMERS_SWIFT_H