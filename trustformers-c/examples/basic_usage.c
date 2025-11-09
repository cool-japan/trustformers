/*
 * Basic usage example for TrustformeRS C API
 * 
 * This example demonstrates:
 * - API initialization and cleanup
 * - Creating a text classification pipeline
 * - Performing inference
 * - Error handling
 * - Memory management
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "trustformers.h"

void print_error(TrustformersError error) {
    if (error != TrustformersError_Success) {
        const char* error_msg = trustformers_error_message(error);
        fprintf(stderr, "Error: %s\n", error_msg);
    }
}

void print_build_info() {
    printf("=== TrustformeRS Build Information ===\n");
    
    const char* version = trustformers_version();
    printf("Version: %s\n", version);
    
    TrustformersBuildInfo build_info = {0};
    TrustformersError error = trustformers_build_info(&build_info);
    if (error == TrustformersError_Success) {
        printf("Features: %s\n", build_info.features);
        printf("Build Date: %s\n", build_info.build_date);
        printf("Target: %s\n", build_info.target);
        
        // Free allocated strings
        trustformers_free_string(build_info.version);
        trustformers_free_string(build_info.features);
        trustformers_free_string(build_info.build_date);
        trustformers_free_string(build_info.target);
    } else {
        print_error(error);
    }
    
    printf("\n");
}

void print_system_info() {
    printf("=== System Information ===\n");
    
    TrustformersSystemInfo sys_info = {0};
    TrustformersError error = trustformers_get_system_info(&sys_info);
    if (error == TrustformersError_Success) {
        printf("CPU Cores: %u\n", sys_info.num_cpu_cores);
        printf("Available CPU Cores: %u\n", sys_info.available_cpu_cores);
        printf("Total Memory: %llu GB\n", sys_info.total_memory_bytes / (1024*1024*1024));
        printf("Available Memory: %llu GB\n", sys_info.available_memory_bytes / (1024*1024*1024));
        printf("CUDA Available: %s\n", sys_info.cuda_available ? "Yes" : "No");
        printf("CUDA Devices: %u\n", sys_info.num_cuda_devices);
        printf("OS: %s\n", sys_info.os_name);
        printf("Architecture: %s\n", sys_info.arch_name);
        
        // Free allocated strings
        trustformers_system_info_free(&sys_info);
    } else {
        print_error(error);
    }
    
    printf("\n");
}

void demonstrate_tokenizer() {
    printf("=== Tokenizer Demo ===\n");
    
    // Check if tokenizers feature is available
    if (!trustformers_has_feature("tokenizers")) {
        printf("Tokenizers feature not available in this build.\n\n");
        return;
    }
    
    TrustformersTokenizer tokenizer;
    TrustformersError error = trustformers_tokenizer_from_pretrained("bert-base-uncased", &tokenizer);
    
    if (error != TrustformersError_Success) {
        printf("Failed to load tokenizer (this is expected in demo without model files)\n");
        print_error(error);
        printf("\n");
        return;
    }
    
    // Get vocabulary size
    unsigned long vocab_size;
    error = trustformers_tokenizer_vocab_size(tokenizer, &vocab_size);
    if (error == TrustformersError_Success) {
        printf("Vocabulary size: %lu\n", vocab_size);
    }
    
    // Encode text
    const char* text = "Hello, world! This is a test.";
    TrustformersTokenizerConfig config = {0};
    config.max_length = 512;
    config.padding = 1;
    config.truncation = 1;
    config.return_attention_mask = 1;
    config.add_special_tokens = 1;
    
    TrustformersEncoding encoding = {0};
    error = trustformers_tokenizer_encode(tokenizer, text, &config, &encoding);
    
    if (error == TrustformersError_Success) {
        printf("Input text: \"%s\"\n", text);
        printf("Token IDs (%zu tokens): ", encoding.input_ids.length);
        
        for (size_t i = 0; i < encoding.input_ids.length && i < 10; i++) {
            printf("%u ", encoding.input_ids.ids[i]);
        }
        if (encoding.input_ids.length > 10) {
            printf("...");
        }
        printf("\n");
        
        // Decode back to text
        char* decoded_text;
        error = trustformers_tokenizer_decode(tokenizer, encoding.input_ids.ids, 
                                            encoding.input_ids.length, 1, &decoded_text);
        if (error == TrustformersError_Success) {
            printf("Decoded text: \"%s\"\n", decoded_text);
            trustformers_free_string(decoded_text);
        }
        
        // Free encoding memory
        trustformers_encoding_free(&encoding);
    } else {
        print_error(error);
    }
    
    // Clean up tokenizer
    trustformers_tokenizer_destroy(tokenizer);
    printf("\n");
}

void demonstrate_pipeline() {
    printf("=== Pipeline Demo ===\n");
    
    // Check if pipelines feature is available
    if (!trustformers_has_feature("pipelines")) {
        printf("Pipelines feature not available in this build.\n\n");
        return;
    }
    
    // Configure pipeline
    TrustformersPipelineConfig config = {0};
    config.task = "text-classification";
    config.model = "bert-base-uncased";
    config.backend_type = 0; // Native backend
    config.device_type = 0;  // CPU
    config.batch_size = 4;
    config.max_length = 512;
    config.enable_profiling = 0;
    config.num_threads = 0; // Auto-detect
    
    TrustformersPipeline pipeline;
    TrustformersError error = trustformers_pipeline_create(&config, &pipeline);
    
    if (error != TrustformersError_Success) {
        printf("Failed to create pipeline (this is expected in demo without model files)\n");
        print_error(error);
        printf("\n");
        return;
    }
    
    // Test single inference
    const char* test_input = "I love this product! It's amazing!";
    TrustformersInferenceResult result = {0};
    
    error = trustformers_pipeline_infer(pipeline, test_input, &result);
    if (error == TrustformersError_Success) {
        printf("Input: \"%s\"\n", test_input);
        printf("Result: %s\n", result.result_json);
        printf("Inference time: %.2f ms\n", result.inference_time_ms);
        printf("Memory used: %llu bytes\n", result.memory_used_bytes);
        
        // Free result memory
        trustformers_inference_result_free(&result);
    } else {
        print_error(error);
    }
    
    // Test batch inference
    const char* batch_inputs[] = {
        "This is great!",
        "I hate this.",
        "It's okay, I guess.",
        "Absolutely fantastic!"
    };
    size_t num_inputs = sizeof(batch_inputs) / sizeof(batch_inputs[0]);
    
    TrustformersBatchResult batch_result = {0};
    error = trustformers_pipeline_batch_infer(pipeline, batch_inputs, num_inputs, &batch_result);
    
    if (error == TrustformersError_Success) {
        printf("\nBatch inference results:\n");
        for (size_t i = 0; i < batch_result.num_results; i++) {
            printf("Input %zu: \"%s\"\n", i+1, batch_inputs[i]);
            printf("Result %zu: %s\n", i+1, batch_result.results[i]);
            printf("Confidence %zu: %.3f\n", i+1, batch_result.confidences[i]);
        }
        printf("Total batch time: %.2f ms\n", batch_result.total_time_ms);
        printf("Average time per item: %.2f ms\n", batch_result.avg_time_per_item_ms);
        
        // Free batch result memory
        trustformers_batch_result_free(&batch_result);
    } else {
        print_error(error);
    }
    
    // Get pipeline info
    char* info_json;
    error = trustformers_pipeline_info(pipeline, &info_json);
    if (error == TrustformersError_Success) {
        printf("\nPipeline info: %s\n", info_json);
        trustformers_free_string(info_json);
    }
    
    // Clean up pipeline
    trustformers_pipeline_destroy(pipeline);
    printf("\n");
}

void demonstrate_performance_timing() {
    printf("=== Performance Timing Demo ===\n");
    
    void* timer = trustformers_timer_create();
    if (!timer) {
        printf("Failed to create timer\n");
        return;
    }
    
    // Simulate some work with timing
    for (int i = 0; i < 5; i++) {
        trustformers_timer_start(timer);
        
        // Simulate work (simple computation)
        volatile double sum = 0.0;
        for (int j = 0; j < 1000000; j++) {
            sum += j * 0.001;
        }
        
        double elapsed_ms;
        TrustformersError error = trustformers_timer_stop(timer, &elapsed_ms);
        if (error == TrustformersError_Success) {
            printf("Iteration %d: %.2f ms\n", i+1, elapsed_ms);
        }
    }
    
    // Get statistics
    TrustformersBenchmarkResult stats = {0};
    TrustformersError error = trustformers_timer_stats(timer, &stats);
    if (error == TrustformersError_Success) {
        printf("\nPerformance Statistics:\n");
        printf("Iterations: %llu\n", stats.iterations);
        printf("Total time: %.2f ms\n", stats.total_time_ms);
        printf("Average time: %.2f ms\n", stats.avg_time_ms);
        printf("Min time: %.2f ms\n", stats.min_time_ms);
        printf("Max time: %.2f ms\n", stats.max_time_ms);
        printf("Std deviation: %.2f ms\n", stats.std_dev_ms);
        printf("Throughput: %.2f ops/sec\n", stats.throughput_ops);
    }
    
    trustformers_timer_destroy(timer);
    printf("\n");
}

void demonstrate_memory_monitoring() {
    printf("=== Memory Monitoring Demo ===\n");
    
    TrustformersMemoryUsage usage = {0};
    TrustformersError error = trustformers_get_memory_usage(&usage);
    
    if (error == TrustformersError_Success) {
        printf("Memory Usage:\n");
        printf("Total memory: %llu bytes\n", usage.total_memory_bytes);
        printf("Peak memory: %llu bytes\n", usage.peak_memory_bytes);
        printf("Allocated models: %llu\n", usage.allocated_models);
        printf("Allocated tokenizers: %llu\n", usage.allocated_tokenizers);
        printf("Allocated pipelines: %llu\n", usage.allocated_pipelines);
        printf("Allocated tensors: %llu\n", usage.allocated_tensors);
    } else {
        print_error(error);
    }
    
    printf("\n");
}

int main() {
    printf("TrustformeRS C API Basic Usage Example\n");
    printf("=====================================\n\n");
    
    // Initialize the API
    TrustformersError error = trustformers_init();
    if (error != TrustformersError_Success) {
        print_error(error);
        return 1;
    }
    
    printf("TrustformeRS initialized successfully!\n\n");
    
    // Demonstrate various features
    print_build_info();
    print_system_info();
    demonstrate_performance_timing();
    demonstrate_memory_monitoring();
    demonstrate_tokenizer();
    demonstrate_pipeline();
    
    // Clean up
    error = trustformers_cleanup();
    if (error != TrustformersError_Success) {
        print_error(error);
        return 1;
    }
    
    printf("TrustformeRS cleaned up successfully!\n");
    printf("Demo completed.\n");
    
    return 0;
}