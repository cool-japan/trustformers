/*
 * Debug Utilities Demo for TrustformeRS C API
 * 
 * This example demonstrates how to use the debug utilities for:
 * - Model introspection and analysis
 * - Performance profiling and monitoring
 * - Visualization of model architecture
 * - Memory usage tracking
 * - Bottleneck identification
 * - Report generation in multiple formats
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include "trustformers.h"

void print_error(TrustformersError error) {
    if (error != TrustformersError_Success) {
        const char* error_msg = trustformers_error_message(error);
        fprintf(stderr, "[ERROR] %s\n", error_msg);
    }
}

void print_separator(const char* title) {
    printf("\n=== %s ===\n", title);
}

void save_to_file(const char* content, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (file) {
        fprintf(file, "%s", content);
        fclose(file);
        printf("[INFO] Saved to %s\n", filename);
    } else {
        printf("[ERROR] Failed to save to %s\n", filename);
    }
}

void demonstrate_model_introspection(void* model) {
    print_separator("Model Introspection");
    
    // Initialize debug utilities
    if (trustformers_debug_init() != 0) {
        printf("[ERROR] Failed to initialize debug utilities\n");
        return;
    }
    
    printf("[INFO] Debug utilities initialized successfully\n");
    
    // Introspect the model
    char* introspection_json = NULL;
    int result = trustformers_debug_introspect_model("demo_model", model, &introspection_json);
    
    if (result == 0 && introspection_json) {
        printf("[INFO] Model introspection completed successfully\n");
        printf("\nModel Information:\n");
        
        // Pretty print some key information (simplified JSON parsing)
        if (strstr(introspection_json, "parameters_count")) {
            printf("✓ Model structure analyzed\n");
            printf("✓ Layer information extracted\n");
            printf("✓ Memory usage calculated\n");
            printf("✓ Parameter counts determined\n");
        }
        
        // Save full introspection to file
        save_to_file(introspection_json, "model_introspection.json");
        
        // Show sample of the data
        printf("\nSample of introspection data:\n");
        char* sample = strndup(introspection_json, 300);
        printf("%s...\n", sample);
        free(sample);
        
        trustformers_debug_free_string(introspection_json);
    } else {
        printf("[ERROR] Model introspection failed\n");
    }
}

void demonstrate_model_visualization(void* model) {
    print_separator("Model Architecture Visualization");
    
    // Generate visualization data
    char* visualization_json = NULL;
    int result = trustformers_debug_generate_visualization("demo_model", &visualization_json);
    
    if (result == 0 && visualization_json) {
        printf("[INFO] Model visualization generated successfully\n");
        
        // Parse and display basic information
        if (strstr(visualization_json, "nodes") && strstr(visualization_json, "edges")) {
            printf("✓ Architecture graph generated\n");
            printf("✓ Node and edge information extracted\n");
            printf("✓ Layout hints provided\n");
        }
        
        // Save visualization data
        save_to_file(visualization_json, "model_visualization.json");
        
        // Generate HTML visualization
        printf("\n[INFO] Generating HTML visualization...\n");
        
        char html_content[8192];
        snprintf(html_content, sizeof(html_content),
            "<!DOCTYPE html>\n"
            "<html>\n"
            "<head>\n"
            "    <title>TrustformeRS Model Visualization</title>\n"
            "    <script src=\"https://unpkg.com/vis-network/standalone/umd/vis-network.min.js\"></script>\n"
            "    <style>\n"
            "        body { font-family: Arial, sans-serif; margin: 20px; }\n"
            "        #network { width: 100%%; height: 600px; border: 1px solid #ccc; }\n"
            "        .info { background: #f5f5f5; padding: 10px; margin: 10px 0; }\n"
            "    </style>\n"
            "</head>\n"
            "<body>\n"
            "    <h1>TrustformeRS Model Architecture</h1>\n"
            "    <div class=\"info\">\n"
            "        <strong>Visualization Features:</strong>\n"
            "        <ul>\n"
            "            <li>Interactive node exploration</li>\n"
            "            <li>Layer-by-layer analysis</li>\n"
            "            <li>Data flow visualization</li>\n"
            "            <li>Parameter and memory information</li>\n"
            "        </ul>\n"
            "    </div>\n"
            "    <div id=\"network\"></div>\n"
            "    <script>\n"
            "        const data = %s;\n"
            "        const options = {\n"
            "            layout: { hierarchical: { direction: 'UD', sortMethod: 'directed' } },\n"
            "            physics: { enabled: false },\n"
            "            nodes: { shape: 'box', margin: 10 },\n"
            "            edges: { arrows: 'to' }\n"
            "        };\n"
            "        const network = new vis.Network(document.getElementById('network'), data, options);\n"
            "        network.on('click', function(params) {\n"
            "            if (params.nodes.length > 0) {\n"
            "                const nodeId = params.nodes[0];\n"
            "                const node = data.nodes.find(n => n.id === nodeId);\n"
            "                alert('Node: ' + node.label + '\\\\nType: ' + node.node_type);\n"
            "            }\n"
            "        });\n"
            "    </script>\n"
            "</body>\n"
            "</html>",
            visualization_json
        );
        
        save_to_file(html_content, "model_visualization.html");
        printf("[INFO] Open model_visualization.html in a web browser to view interactive model\n");
        
        trustformers_debug_free_string(visualization_json);
    } else {
        printf("[ERROR] Model visualization failed\n");
    }
}

void demonstrate_performance_profiling(void* pipeline) {
    print_separator("Performance Profiling");
    
    // Start a debug session
    const char* session_id = "performance_demo";
    if (trustformers_debug_start_session(session_id) != 0) {
        printf("[ERROR] Failed to start debug session\n");
        return;
    }
    
    printf("[INFO] Started debug session: %s\n", session_id);
    
    // Simulate some inference operations with profiling
    printf("[INFO] Running profiled inference operations...\n");
    
    const char* test_inputs[] = {
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world in unprecedented ways.",
        "Machine learning models require careful optimization for production deployment.",
        "Deep neural networks can learn complex patterns from large datasets.",
        "Natural language processing enables computers to understand human language."
    };
    
    int num_inputs = sizeof(test_inputs) / sizeof(test_inputs[0]);
    
    for (int i = 0; i < num_inputs; i++) {
        printf("  Processing input %d: \"%.40s%s\"\n", 
               i + 1, test_inputs[i], strlen(test_inputs[i]) > 40 ? "..." : "");
        
        // Simulate inference (in real code, this would call the actual pipeline)
        TrustformersInferenceResult result = {0};
        TrustformersError error = trustformers_pipeline_infer(pipeline, test_inputs[i], &result);
        
        if (error == TrustformersError_Success) {
            printf("    ✓ Completed in %.2f ms\n", result.inference_time_ms);
            trustformers_inference_result_free(&result);
        } else {
            printf("    ✗ Failed (expected in demo without real model)\n");
        }
        
        // Simulate some processing delay
        usleep(100000); // 100ms
    }
    
    // Simulate memory snapshots and tensor operations
    printf("\n[INFO] Taking memory snapshots and recording tensor operations...\n");
    
    // In a real implementation, these would be called automatically during inference
    printf("  📸 Memory snapshot 1: Initial state\n");
    printf("  🔧 Tensor operation: Input tokenization\n");
    printf("  🔧 Tensor operation: Embedding lookup\n");
    printf("  📸 Memory snapshot 2: After embedding\n");
    printf("  🔧 Tensor operation: Multi-head attention\n");
    printf("  🔧 Tensor operation: Feed-forward network\n");
    printf("  📸 Memory snapshot 3: After transformer blocks\n");
    printf("  🔧 Tensor operation: Output projection\n");
    printf("  📸 Memory snapshot 4: Final state\n");
    
    printf("\n[INFO] Profiling session completed, generating reports...\n");
}

void demonstrate_report_generation() {
    print_separator("Report Generation");
    
    const char* session_id = "performance_demo";
    
    // Generate performance report
    char* report = NULL;
    int result = trustformers_debug_generate_report(session_id, &report);
    
    if (result == 0 && report) {
        printf("[INFO] Performance report generated successfully\n");
        
        // Save the report
        save_to_file(report, "performance_report.md");
        
        // Display a summary
        printf("\nPerformance Report Summary:\n");
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        
        // Extract key lines from the report (simplified parsing)
        char* line = strtok(strdup(report), "\n");
        int line_count = 0;
        while (line != NULL && line_count < 15) {
            if (strstr(line, "Session ID:") || 
                strstr(line, "Total Duration:") ||
                strstr(line, "Tensor Operations:") ||
                strstr(line, "Bottlenecks Found:") ||
                strstr(line, "Initial Memory:") ||
                strstr(line, "Peak Memory:") ||
                strstr(line, "Recommendations")) {
                printf("%s\n", line);
            }
            line = strtok(NULL, "\n");
            line_count++;
        }
        
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("\n📋 Full report saved to performance_report.md\n");
        
        trustformers_debug_free_string(report);
    } else {
        printf("[ERROR] Failed to generate performance report\n");
    }
}

void demonstrate_advanced_debugging() {
    print_separator("Advanced Debugging Features");
    
    printf("[INFO] Advanced debugging capabilities:\n\n");
    
    printf("🔍 Model Analysis:\n");
    printf("  • Layer-by-layer parameter analysis\n");
    printf("  • Quantization impact assessment\n");
    printf("  • Memory usage breakdown\n");
    printf("  • Computational complexity estimation\n\n");
    
    printf("⚡ Performance Monitoring:\n");
    printf("  • Real-time inference timing\n");
    printf("  • Memory allocation tracking\n");
    printf("  • Cache hit rate analysis\n");
    printf("  • GPU utilization monitoring\n\n");
    
    printf("🎯 Bottleneck Detection:\n");
    printf("  • Slow layer identification\n");
    printf("  • Memory leak detection\n");
    printf("  • Cache performance issues\n");
    printf("  • Computational inefficiencies\n\n");
    
    printf("📊 Visualization Options:\n");
    printf("  • Interactive model graphs\n");
    printf("  • Performance timelines\n");
    printf("  • Memory usage charts\n");
    printf("  • Attention weight visualizations\n\n");
    
    printf("📈 Export Formats:\n");
    printf("  • JSON data for custom analysis\n");
    printf("  • CSV for spreadsheet import\n");
    printf("  • HTML for web viewing\n");
    printf("  • Markdown reports\n\n");
    
    printf("🔧 Integration Features:\n");
    printf("  • CI/CD pipeline integration\n");
    printf("  • Prometheus metrics export\n");
    printf("  • Custom callback hooks\n");
    printf("  • Real-time monitoring APIs\n");
}

void demonstrate_best_practices() {
    print_separator("Debugging Best Practices");
    
    printf("📚 Best Practices for Model Debugging:\n\n");
    
    printf("1. 🎯 Strategic Profiling:\n");
    printf("   • Profile representative workloads\n");
    printf("   • Use consistent input sizes\n");
    printf("   • Measure multiple iterations\n");
    printf("   • Include warm-up runs\n\n");
    
    printf("2. 🧠 Memory Management:\n");
    printf("   • Monitor peak memory usage\n");
    printf("   • Track allocation patterns\n");
    printf("   • Check for memory leaks\n");
    printf("   • Optimize tensor lifecycles\n\n");
    
    printf("3. ⚡ Performance Optimization:\n");
    printf("   • Identify computational hotspots\n");
    printf("   • Analyze cache performance\n");
    printf("   • Consider quantization options\n");
    printf("   • Evaluate batch size impact\n\n");
    
    printf("4. 🔍 Model Analysis:\n");
    printf("   • Understand layer contributions\n");
    printf("   • Analyze parameter distributions\n");
    printf("   • Monitor activation patterns\n");
    printf("   • Validate numerical stability\n\n");
    
    printf("5. 📊 Visualization Usage:\n");
    printf("   • Use interactive tools for exploration\n");
    printf("   • Generate regular performance reports\n");
    printf("   • Share visualizations with team\n");
    printf("   • Archive debugging sessions\n\n");
    
    printf("6. 🔄 Continuous Monitoring:\n");
    printf("   • Set up automated profiling\n");
    printf("   • Monitor production performance\n");
    printf("   • Track performance regressions\n");
    printf("   • Maintain debugging documentation\n");
}

int main() {
    printf("TrustformeRS Debug Utilities Demo\n");
    printf("=================================\n");
    printf("\nThis demo showcases the comprehensive debugging and profiling\n");
    printf("capabilities of the TrustformeRS C API for model analysis and\n");
    printf("performance optimization.\n");
    
    // Initialize TrustformeRS
    TrustformersError error = trustformers_init();
    if (error != TrustformersError_Success) {
        print_error(error);
        return 1;
    }
    
    printf("\n[INFO] TrustformeRS initialized successfully\n");
    
    // Load a model for demonstration
    void* model = NULL;
    void* tokenizer = NULL;
    void* pipeline = NULL;
    
    // Try to load a model (will fail in demo, but that's expected)
    model = trustformers_load_model_from_hub("gpt2", &error);
    if (error == TrustformersError_Success) {
        printf("[INFO] Model loaded successfully\n");
        
        tokenizer = trustformers_load_tokenizer_from_hub("gpt2", &error);
        if (error == TrustformersError_Success) {
            printf("[INFO] Tokenizer loaded successfully\n");
            
            pipeline = trustformers_create_text_generation_pipeline(model, tokenizer, &error);
            if (error == TrustformersError_Success) {
                printf("[INFO] Pipeline created successfully\n");
            }
        }
    } else {
        printf("[INFO] Model loading failed (expected in demo) - using mock objects\n");
        // Use placeholder pointers for demonstration
        model = (void*)0x1000;
        pipeline = (void*)0x2000;
    }
    
    // Demonstrate debug utilities
    demonstrate_model_introspection(model);
    demonstrate_model_visualization(model);
    demonstrate_performance_profiling(pipeline);
    demonstrate_report_generation();
    demonstrate_advanced_debugging();
    demonstrate_best_practices();
    
    print_separator("Demo Summary");
    
    printf("🎉 Debug utilities demonstration completed!\n\n");
    
    printf("Generated files:\n");
    printf("  📄 model_introspection.json - Detailed model analysis\n");
    printf("  📄 model_visualization.json - Visualization data\n");
    printf("  🌐 model_visualization.html - Interactive model viewer\n");
    printf("  📋 performance_report.md - Performance analysis report\n\n");
    
    printf("Next steps:\n");
    printf("  1. 🌐 Open model_visualization.html in your browser\n");
    printf("  2. 📖 Review the performance report\n");
    printf("  3. 🔧 Apply optimization recommendations\n");
    printf("  4. 📊 Integrate debugging into your workflow\n\n");
    
    printf("For production use:\n");
    printf("  • Set up continuous profiling\n");
    printf("  • Monitor performance metrics\n");
    printf("  • Use debugging data for optimization\n");
    printf("  • Share insights with your team\n");
    
    // Cleanup
    if (pipeline && pipeline != (void*)0x2000) {
        trustformers_pipeline_free(pipeline);
    }
    if (tokenizer) {
        trustformers_tokenizer_free(tokenizer);
    }
    if (model && model != (void*)0x1000) {
        trustformers_model_free(model);
    }
    
    trustformers_cleanup();
    printf("\n[INFO] Demo cleanup completed\n");
    
    return 0;
}