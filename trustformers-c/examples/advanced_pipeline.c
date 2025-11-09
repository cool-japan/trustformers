/*
 * Advanced pipeline example for TrustformeRS C API
 * 
 * This example demonstrates:
 * - ONNX backend usage
 * - Advanced pipeline configuration
 * - Performance benchmarking
 * - Batch processing optimization
 * - Error handling and recovery
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "trustformers.h"

void print_error(TrustformersError error) {
    if (error != TrustformersError_Success) {
        const char* error_msg = trustformers_error_message(error);
        fprintf(stderr, "Error: %s\n", error_msg);
    }
}

void print_separator(const char* title) {
    printf("\n=== %s ===\n", title);
}

void check_feature_availability() {
    print_separator("Feature Availability Check");
    
    const char* features[] = {
        "tokenizers", "models", "pipelines", "gpu", "onnx", "quantization", "debug"
    };
    size_t num_features = sizeof(features) / sizeof(features[0]);
    
    printf("Available features:\n");
    for (size_t i = 0; i < num_features; i++) {
        int available = trustformers_has_feature(features[i]);
        printf("  %s: %s\n", features[i], available ? "✓" : "✗");
    }
}

void demonstrate_onnx_text_classification() {
    print_separator("ONNX Text Classification Pipeline");
    
    if (!trustformers_has_feature("onnx")) {
        printf("ONNX feature not available in this build.\n");
        return;
    }
    
    // Note: In a real scenario, you would have actual ONNX model files
    const char* model_path = "models/bert-base-uncased.onnx";
    const char* tokenizer_name = "bert-base-uncased";
    
    TrustformersPipeline pipeline;
    TrustformersError error = trustformers_onnx_text_classification_pipeline_create(
        model_path, tokenizer_name, &pipeline
    );
    
    if (error != TrustformersError_Success) {
        printf("Failed to create ONNX pipeline (expected without model files):\n");
        print_error(error);
        printf("In practice, you would:\n");
        printf("1. Export your model to ONNX format\n");
        printf("2. Place it at: %s\n", model_path);
        printf("3. Run this example\n");
        return;
    }
    
    // Test sentiment analysis
    const char* sentiment_texts[] = {
        "I absolutely love this product! It's fantastic!",
        "This is the worst thing I've ever bought.",
        "It's okay, nothing special but works fine.",
        "Amazing quality and great customer service!",
        "Terrible experience, would not recommend."
    };
    size_t num_texts = sizeof(sentiment_texts) / sizeof(sentiment_texts[0]);
    
    printf("Testing sentiment analysis with %zu texts...\n", num_texts);
    
    // Performance timing
    void* timer = trustformers_timer_create();
    
    for (size_t i = 0; i < num_texts; i++) {
        trustformers_timer_start(timer);
        
        TrustformersInferenceResult result = {0};
        error = trustformers_pipeline_infer(pipeline, sentiment_texts[i], &result);
        
        double elapsed_ms;
        trustformers_timer_stop(timer, &elapsed_ms);
        
        if (error == TrustformersError_Success) {
            printf("\nText %zu: \"%.50s%s\"\n", i+1, sentiment_texts[i], 
                   strlen(sentiment_texts[i]) > 50 ? "..." : "");
            printf("Result: %s\n", result.result_json);
            printf("Confidence: %.3f\n", result.confidence);
            printf("Inference time: %.2f ms\n", elapsed_ms);
            
            trustformers_inference_result_free(&result);
        } else {
            print_error(error);
        }
    }
    
    // Print performance statistics
    TrustformersBenchmarkResult stats = {0};
    trustformers_timer_stats(timer, &stats);
    printf("\nPerformance Summary:\n");
    printf("Average inference time: %.2f ms\n", stats.avg_time_ms);
    printf("Min/Max: %.2f / %.2f ms\n", stats.min_time_ms, stats.max_time_ms);
    printf("Throughput: %.2f inferences/sec\n", stats.throughput_ops);
    
    trustformers_timer_destroy(timer);
    trustformers_pipeline_destroy(pipeline);
}

void demonstrate_onnx_text_generation() {
    print_separator("ONNX Text Generation Pipeline");
    
    if (!trustformers_has_feature("onnx")) {
        printf("ONNX feature not available in this build.\n");
        return;
    }
    
    const char* model_path = "models/gpt2.onnx";
    const char* tokenizer_name = "gpt2";
    
    TrustformersPipeline pipeline;
    TrustformersError error = trustformers_onnx_text_generation_pipeline_create(
        model_path, tokenizer_name, &pipeline
    );
    
    if (error != TrustformersError_Success) {
        printf("Failed to create ONNX generation pipeline (expected without model files):\n");
        print_error(error);
        printf("To use this feature:\n");
        printf("1. Export GPT-2 or similar model to ONNX\n");
        printf("2. Place it at: %s\n", model_path);
        printf("3. Configure generation parameters\n");
        return;
    }
    
    // Test text generation
    const char* prompts[] = {
        "The future of artificial intelligence is",
        "In a world where technology",
        "Scientists have discovered",
        "The most important lesson in life is"
    };
    size_t num_prompts = sizeof(prompts) / sizeof(prompts[0]);
    
    printf("Testing text generation with %zu prompts...\n", num_prompts);
    
    for (size_t i = 0; i < num_prompts; i++) {
        TrustformersInferenceResult result = {0};
        error = trustformers_pipeline_infer(pipeline, prompts[i], &result);
        
        if (error == TrustformersError_Success) {
            printf("\nPrompt %zu: \"%s\"\n", i+1, prompts[i]);
            printf("Generated: %s\n", result.result_json);
            printf("Time: %.2f ms\n", result.inference_time_ms);
            
            trustformers_inference_result_free(&result);
        } else {
            print_error(error);
        }
    }
    
    trustformers_pipeline_destroy(pipeline);
}

void demonstrate_batch_processing() {
    print_separator("Optimized Batch Processing");
    
    // Create a native pipeline for comparison
    TrustformersPipelineConfig config = {0};
    config.task = "text-classification";
    config.model = "distilbert-base-uncased";
    config.backend_type = 0; // Native
    config.device_type = 0;  // CPU
    config.batch_size = 8;   // Optimized batch size
    config.max_length = 256;
    config.enable_profiling = 1;
    config.num_threads = 4;
    
    TrustformersPipeline pipeline;
    TrustformersError error = trustformers_pipeline_create(&config, &pipeline);
    
    if (error != TrustformersError_Success) {
        printf("Failed to create pipeline for batch processing demo:\n");
        print_error(error);
        return;
    }
    
    // Generate test data
    const char* test_texts[] = {
        "This product exceeded my expectations!",
        "Terrible quality, waste of money.",
        "Good value for the price.",
        "Outstanding customer service experience.",
        "Product arrived damaged and late.",
        "Exactly what I was looking for.",
        "Poor build quality, fell apart quickly.",
        "Great features and easy to use.",
        "Not worth the high price point.",
        "Fantastic design and performance.",
        "Customer support was unhelpful.",
        "Best purchase I've made this year!",
        "Average product, nothing special.",
        "Highly recommended for anyone.",
        "Complete disappointment, avoid.",
        "Perfect for my needs."
    };
    size_t num_texts = sizeof(test_texts) / sizeof(test_texts[0]);
    
    printf("Processing %zu texts in batch...\n", num_texts);
    
    // Time batch processing
    void* timer = trustformers_timer_create();
    trustformers_timer_start(timer);
    
    TrustformersBatchResult batch_result = {0};
    error = trustformers_pipeline_batch_infer(pipeline, test_texts, num_texts, &batch_result);
    
    double batch_elapsed;
    trustformers_timer_stop(timer, &batch_elapsed);
    
    if (error == TrustformersError_Success) {
        printf("\nBatch processing completed successfully!\n");
        printf("Results:\n");
        
        // Show first few results
        size_t max_display = batch_result.num_results < 5 ? batch_result.num_results : 5;
        for (size_t i = 0; i < max_display; i++) {
            printf("  %zu. \"%.40s...\" -> %s (%.3f)\n", 
                   i+1, test_texts[i], batch_result.results[i], batch_result.confidences[i]);
        }
        
        if (batch_result.num_results > max_display) {
            printf("  ... and %zu more results\n", batch_result.num_results - max_display);
        }
        
        printf("\nBatch Performance:\n");
        printf("Total time: %.2f ms\n", batch_result.total_time_ms);
        printf("Average per item: %.2f ms\n", batch_result.avg_time_per_item_ms);
        printf("Effective throughput: %.2f items/sec\n", 
               num_texts / (batch_result.total_time_ms / 1000.0));
        
        // Compare with individual processing estimate
        double estimated_individual = batch_result.avg_time_per_item_ms * num_texts;
        double speedup = estimated_individual / batch_result.total_time_ms;
        printf("Estimated speedup vs individual: %.2fx\n", speedup);
        
        trustformers_batch_result_free(&batch_result);
    } else {
        print_error(error);
    }
    
    // Time individual processing for comparison
    printf("\nComparing with individual processing...\n");
    trustformers_timer_reset(timer);
    
    double total_individual_time = 0.0;
    for (size_t i = 0; i < (num_texts < 5 ? num_texts : 5); i++) { // Just test first 5
        trustformers_timer_start(timer);
        
        TrustformersInferenceResult result = {0};
        error = trustformers_pipeline_infer(pipeline, test_texts[i], &result);
        
        double elapsed;
        trustformers_timer_stop(timer, &elapsed);
        total_individual_time += elapsed;
        
        if (error == TrustformersError_Success) {
            trustformers_inference_result_free(&result);
        }
    }
    
    double avg_individual = total_individual_time / 5.0;
    printf("Individual processing average: %.2f ms\n", avg_individual);
    printf("Batch processing average: %.2f ms\n", batch_result.avg_time_per_item_ms);
    printf("Actual speedup: %.2fx\n", avg_individual / batch_result.avg_time_per_item_ms);
    
    trustformers_timer_destroy(timer);
    trustformers_pipeline_destroy(pipeline);
}

void demonstrate_error_handling_and_recovery() {
    print_separator("Error Handling and Recovery");
    
    printf("Testing various error conditions...\n");
    
    // Test null pointer handling
    TrustformersError error = trustformers_pipeline_create(NULL, NULL);
    printf("Null config test: %s\n", 
           error == TrustformersError_NullPointer ? "✓ Handled correctly" : "✗ Unexpected result");
    
    // Test invalid model path
    TrustformersPipelineConfig bad_config = {0};
    bad_config.task = "text-classification";
    bad_config.model = "nonexistent-model-12345";
    bad_config.backend_type = 0;
    bad_config.device_type = 0;
    
    TrustformersPipeline pipeline;
    error = trustformers_pipeline_create(&bad_config, &pipeline);
    printf("Invalid model test: %s\n", 
           error != TrustformersError_Success ? "✓ Handled correctly" : "✗ Should have failed");
    
    if (error != TrustformersError_Success) {
        print_error(error);
    }
    
    // Test invalid tokenizer
    TrustformersTokenizer tokenizer;
    error = trustformers_tokenizer_from_pretrained("invalid-tokenizer-name", &tokenizer);
    printf("Invalid tokenizer test: %s\n", 
           error != TrustformersError_Success ? "✓ Handled correctly" : "✗ Should have failed");
    
    // Test string validation
    int valid = trustformers_validate_string("valid string");
    int invalid = trustformers_validate_string(NULL);
    printf("String validation: %s\n", 
           (valid == 1 && invalid == 0) ? "✓ Working correctly" : "✗ Unexpected behavior");
    
    printf("\nError handling demonstration completed.\n");
}

void demonstrate_memory_management() {
    print_separator("Memory Management and Monitoring");
    
    printf("Initial memory state:\n");
    TrustformersMemoryUsage initial_usage = {0};
    trustformers_get_memory_usage(&initial_usage);
    printf("Allocated objects: %llu models, %llu tokenizers, %llu pipelines, %llu tensors\n",
           initial_usage.allocated_models, initial_usage.allocated_tokenizers,
           initial_usage.allocated_pipelines, initial_usage.allocated_tensors);
    
    // Create and destroy multiple resources to test memory management
    printf("\nCreating multiple resources...\n");
    
    // Try to create multiple tokenizers (will likely fail without models, but tests the API)
    TrustformersTokenizer tokenizers[3];
    const char* tokenizer_names[] = {"bert-base-uncased", "gpt2", "roberta-base"};
    int created_tokenizers = 0;
    
    for (int i = 0; i < 3; i++) {
        TrustformersError error = trustformers_tokenizer_from_pretrained(tokenizer_names[i], &tokenizers[i]);
        if (error == TrustformersError_Success) {
            created_tokenizers++;
            printf("Created tokenizer %d: %s\n", i+1, tokenizer_names[i]);
        }
    }
    
    // Check memory usage after creation
    TrustformersMemoryUsage after_creation = {0};
    trustformers_get_memory_usage(&after_creation);
    printf("\nAfter creation:\n");
    printf("Allocated tokenizers: %llu (delta: +%lld)\n", 
           after_creation.allocated_tokenizers, 
           (long long)(after_creation.allocated_tokenizers - initial_usage.allocated_tokenizers));
    
    // Clean up created resources
    printf("\nCleaning up resources...\n");
    for (int i = 0; i < created_tokenizers; i++) {
        TrustformersError error = trustformers_tokenizer_destroy(tokenizers[i]);
        if (error == TrustformersError_Success) {
            printf("Destroyed tokenizer %d\n", i+1);
        }
    }
    
    // Force garbage collection
    trustformers_gc();
    
    // Check final memory usage
    TrustformersMemoryUsage final_usage = {0};
    trustformers_get_memory_usage(&final_usage);
    printf("\nAfter cleanup:\n");
    printf("Allocated tokenizers: %llu\n", final_usage.allocated_tokenizers);
    printf("Memory management test: %s\n",
           final_usage.allocated_tokenizers == initial_usage.allocated_tokenizers ? 
           "✓ Cleaned up correctly" : "⚠ Some resources may remain");
}

int main() {
    printf("TrustformeRS C API Advanced Pipeline Example\n");
    printf("===========================================\n");
    
    // Initialize
    TrustformersError error = trustformers_init();
    if (error != TrustformersError_Success) {
        print_error(error);
        return 1;
    }
    
    printf("TrustformeRS initialized successfully!\n");
    
    // Set log level for more detailed output
    trustformers_set_log_level(3); // Info level
    
    // Run demonstrations
    check_feature_availability();
    demonstrate_error_handling_and_recovery();
    demonstrate_memory_management();
    demonstrate_onnx_text_classification();
    demonstrate_onnx_text_generation();
    demonstrate_batch_processing();
    
    // Final cleanup
    error = trustformers_cleanup();
    if (error != TrustformersError_Success) {
        print_error(error);
        return 1;
    }
    
    printf("\nAdvanced pipeline demonstration completed successfully!\n");
    printf("\nNote: Many operations in this demo will fail without actual model files.\n");
    printf("To run with real models:\n");
    printf("1. Export your models to ONNX format\n");
    printf("2. Place them in the 'models/' directory\n");
    printf("3. Update the model paths in this example\n");
    printf("4. Recompile and run\n");
    
    return 0;
}