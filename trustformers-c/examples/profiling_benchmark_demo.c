/*
 * Profiling and Benchmarking Demo for TrustformeRS C API
 * 
 * This example demonstrates comprehensive profiling and benchmarking capabilities:
 * - High-resolution timing and performance measurement
 * - Statistical analysis of benchmark results
 * - Continuous profiling and monitoring
 * - Regression testing and performance tracking
 * - Performance comparison and optimization guidance
 * - Export capabilities for analysis and reporting
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include "trustformers.h"

void print_error(TrustformersError error) {
    if (error != TrustformersError_Success) {
        const char* error_msg = trustformers_error_message(error);
        fprintf(stderr, "[ERROR] %s\n", error_msg);
    }
}

void print_separator(const char* title) {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║ %-86s ║\n", title);
    printf("╚════════════════════════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

void save_to_file(const char* content, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (file) {
        fprintf(file, "%s", content);
        fclose(file);
        printf("💾 Saved to %s\n", filename);
    } else {
        printf("❌ Failed to save to %s\n", filename);
    }
}

// Simulate CPU-intensive work
void simulate_cpu_work(int intensity, int duration_ms) {
    clock_t start = clock();
    double target_duration = (double)duration_ms / 1000.0;
    
    volatile double result = 0.0;
    while (((double)(clock() - start) / CLOCKS_PER_SEC) < target_duration) {
        for (int i = 0; i < intensity; i++) {
            result += sin(i) * cos(i) * sqrt(i + 1);
        }
    }
}

// Simulate memory-intensive work
void simulate_memory_work(int size_mb, int duration_ms) {
    size_t size = size_mb * 1024 * 1024;
    char* buffer = malloc(size);
    
    if (buffer) {
        clock_t start = clock();
        double target_duration = (double)duration_ms / 1000.0;
        
        while (((double)(clock() - start) / CLOCKS_PER_SEC) < target_duration) {
            // Random memory access pattern
            for (int i = 0; i < 1000; i++) {
                size_t index = rand() % size;
                buffer[index] = (char)(rand() % 256);
            }
        }
        
        free(buffer);
    }
}

// Simulate inference workload
void simulate_inference(const char* model_size, int batch_size) {
    printf("🧠 Simulating %s inference (batch size: %d)\n", model_size, batch_size);
    
    if (strcmp(model_size, "small") == 0) {
        simulate_cpu_work(10000, 20 + (rand() % 10)); // 20-30ms
    } else if (strcmp(model_size, "medium") == 0) {
        simulate_cpu_work(50000, 80 + (rand() % 20)); // 80-100ms
    } else if (strcmp(model_size, "large") == 0) {
        simulate_cpu_work(100000, 200 + (rand() % 50)); // 200-250ms
    }
    
    // Add batch processing overhead
    if (batch_size > 1) {
        simulate_cpu_work(1000 * batch_size, 5);
    }
}

void demonstrate_basic_timing() {
    print_separator("Basic Timing and Measurement");
    
    // Initialize profiling tools
    if (trustformers_profiling_init() != 0) {
        printf("❌ Failed to initialize profiling tools\n");
        return;
    }
    
    printf("✅ Profiling tools initialized successfully\n\n");
    
    // Create timers for different operations
    const char* timer_names[] = {"tokenization", "inference", "postprocessing", "total"};
    int num_timers = sizeof(timer_names) / sizeof(timer_names[0]);
    
    printf("🔧 Creating timers...\n");
    for (int i = 0; i < num_timers; i++) {
        if (trustformers_profiling_create_timer(timer_names[i]) == 0) {
            printf("  ✅ Created timer: %s\n", timer_names[i]);
        } else {
            printf("  ❌ Failed to create timer: %s\n", timer_names[i]);
        }
    }
    
    printf("\n📏 Running timed operations...\n");
    
    // Simulate a complete inference pipeline with timing
    for (int run = 1; run <= 3; run++) {
        printf("\n  Run %d:\n", run);
        
        // Start total timer
        trustformers_profiling_start_timer("total");
        
        // Tokenization phase
        trustformers_profiling_start_timer("tokenization");
        printf("    🔤 Tokenizing input... ");
        simulate_cpu_work(5000, 10 + (rand() % 5));
        double tokenization_time;
        trustformers_profiling_stop_timer("tokenization", &tokenization_time);
        printf("%.2f ms\n", tokenization_time);
        
        // Inference phase
        trustformers_profiling_start_timer("inference");
        printf("    🧠 Running inference... ");
        simulate_inference("medium", 1);
        double inference_time;
        trustformers_profiling_stop_timer("inference", &inference_time);
        printf("%.2f ms\n", inference_time);
        
        // Post-processing phase
        trustformers_profiling_start_timer("postprocessing");
        printf("    🔧 Post-processing... ");
        simulate_cpu_work(3000, 8 + (rand() % 4));
        double postprocessing_time;
        trustformers_profiling_stop_timer("postprocessing", &postprocessing_time);
        printf("%.2f ms\n", postprocessing_time);
        
        // Stop total timer
        double total_time;
        trustformers_profiling_stop_timer("total", &total_time);
        printf("    ⏱️  Total time: %.2f ms\n", total_time);
    }
}

void demonstrate_statistical_benchmarking() {
    print_separator("Statistical Benchmarking and Analysis");
    
    printf("📊 Running comprehensive benchmarks with statistical analysis...\n\n");
    
    // Benchmark different model sizes
    const char* model_sizes[] = {"small", "medium", "large"};
    int batch_sizes[] = {1, 4, 8, 16};
    
    for (int m = 0; m < 3; m++) {
        for (int b = 0; b < 4; b++) {
            const char* model_size = model_sizes[m];
            int batch_size = batch_sizes[b];
            
            printf("🎯 Benchmarking %s model with batch size %d\n", model_size, batch_size);
            
            char timer_name[64];
            snprintf(timer_name, sizeof(timer_name), "benchmark_%s_batch_%d", model_size, batch_size);
            
            // Create and run timer multiple times for statistics
            trustformers_profiling_create_timer(timer_name);
            
            int iterations = 20; // Number of benchmark iterations
            printf("  Running %d iterations: ", iterations);
            
            for (int i = 0; i < iterations; i++) {
                trustformers_profiling_start_timer(timer_name);
                simulate_inference(model_size, batch_size);
                double elapsed;
                trustformers_profiling_stop_timer(timer_name, &elapsed);
                
                if (i % 5 == 0) {
                    printf(".");
                    fflush(stdout);
                }
            }
            printf(" done\n");
            
            usleep(100000); // Small delay between benchmarks
        }
    }
    
    printf("\n✅ All benchmarks completed\n");
}

void demonstrate_continuous_profiling() {
    print_separator("Continuous Performance Monitoring");
    
    printf("📈 Starting continuous profiling session...\n");
    
    // Start continuous profiling (1000 data points, sample every 50ms)
    if (trustformers_profiling_start_continuous(1000, 50) == 0) {
        printf("✅ Continuous profiling started (sampling every 50ms)\n");
    } else {
        printf("❌ Failed to start continuous profiling\n");
        return;
    }
    
    printf("\n🏃 Running workload while profiling...\n");
    
    // Simulate a varying workload
    const char* workload_phases[] = {
        "warmup", "steady_state", "peak_load", "memory_intensive", "cooldown"
    };
    
    for (int phase = 0; phase < 5; phase++) {
        printf("  Phase %d: %s\n", phase + 1, workload_phases[phase]);
        
        for (int i = 0; i < 10; i++) {
            switch (phase) {
                case 0: // warmup
                    simulate_inference("small", 1);
                    usleep(200000); // 200ms delay
                    break;
                case 1: // steady_state
                    simulate_inference("medium", 4);
                    usleep(100000); // 100ms delay
                    break;
                case 2: // peak_load
                    simulate_inference("large", 8);
                    usleep(50000); // 50ms delay
                    break;
                case 3: // memory_intensive
                    simulate_memory_work(10, 100);
                    usleep(150000); // 150ms delay
                    break;
                case 4: // cooldown
                    simulate_inference("small", 1);
                    usleep(300000); // 300ms delay
                    break;
            }
            
            if (i % 3 == 0) {
                printf("    ⚡ Processing...\n");
            }
        }
    }
    
    // Stop continuous profiling
    if (trustformers_profiling_stop_continuous() == 0) {
        printf("\n✅ Continuous profiling stopped\n");
        printf("📊 Profiling data collected and ready for analysis\n");
    }
}

void demonstrate_report_generation() {
    print_separator("Performance Analysis and Reporting");
    
    printf("📋 Generating comprehensive performance reports...\n\n");
    
    // Generate performance report
    char* report = NULL;
    if (trustformers_profiling_generate_report(&report) == 0 && report) {
        printf("✅ Performance report generated successfully\n");
        
        // Save full report
        save_to_file(report, "performance_analysis.md");
        
        // Display report summary
        printf("\n📊 Performance Report Summary:\n");
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        
        // Extract key sections (simplified parsing)
        char* summary_start = strstr(report, "## Summary");
        if (summary_start) {
            char* next_section = strstr(summary_start + 1, "##");
            if (next_section) {
                int len = next_section - summary_start;
                char summary[1024];
                strncpy(summary, summary_start, len);
                summary[len] = '\0';
                printf("%s\n", summary);
            }
        }
        
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        
        trustformers_profiling_free_string(report);
    } else {
        printf("❌ Failed to generate performance report\n");
    }
    
    // Export data in different formats
    printf("\n💾 Exporting benchmark data...\n");
    
    const char* formats[] = {"json", "csv"};
    const char* extensions[] = {"json", "csv"};
    
    for (int i = 0; i < 2; i++) {
        char* data = NULL;
        if (trustformers_profiling_export_data(formats[i], &data) == 0 && data) {
            char filename[64];
            snprintf(filename, sizeof(filename), "benchmark_data.%s", extensions[i]);
            save_to_file(data, filename);
            trustformers_profiling_free_string(data);
        } else {
            printf("❌ Failed to export %s data\n", formats[i]);
        }
    }
}

void demonstrate_performance_analysis() {
    print_separator("Advanced Performance Analysis");
    
    printf("🔍 Performance Analysis Insights:\n\n");
    
    printf("📈 Key Metrics to Monitor:\n");
    printf("  • ⏱️  Latency (p50, p95, p99 percentiles)\n");
    printf("  • 🚀 Throughput (operations per second)\n");
    printf("  • 💾 Memory usage (peak, average, fragmentation)\n");
    printf("  • 🔄 CPU utilization and efficiency\n");
    printf("  • 🎮 GPU utilization (if applicable)\n");
    printf("  • 📊 Statistical variance and stability\n\n");
    
    printf("🎯 Optimization Opportunities:\n");
    printf("  • 🔧 Model quantization for speed/memory tradeoffs\n");
    printf("  • 📦 Batch size optimization for throughput\n");
    printf("  • 🧠 Layer fusion and computational graph optimization\n");
    printf("  • 💾 Memory pooling and allocation strategies\n");
    printf("  • ⚡ Hardware-specific optimizations (SIMD, GPU kernels)\n");
    printf("  • 🔄 Pipeline parallelization and async processing\n\n");
    
    printf("📋 Benchmarking Best Practices:\n");
    printf("  • 🎯 Use representative workloads and data\n");
    printf("  • 🔥 Include warmup iterations to stabilize performance\n");
    printf("  • 📊 Run sufficient iterations for statistical significance\n");
    printf("  • 🏷️  Tag and version your benchmarks for tracking\n");
    printf("  • 📈 Monitor long-term performance trends\n");
    printf("  • 🔄 Automate regression testing in CI/CD\n\n");
    
    printf("⚠️  Common Performance Pitfalls:\n");
    printf("  • 🐌 Not warming up models before benchmarking\n");
    printf("  • 📏 Using unrealistic input sizes or batch configurations\n");
    printf("  • 🔄 Ignoring memory allocation overhead\n");
    printf("  • 📊 Focusing only on average performance, ignoring variance\n");
    printf("  • 🎮 Not considering hardware-specific optimizations\n");
    printf("  • 📈 Benchmarking in debug mode instead of release builds\n");
}

void demonstrate_integration_examples() {
    print_separator("Integration and Production Usage");
    
    printf("🏗️  Integration Examples:\n\n");
    
    printf("1. 🔄 CI/CD Integration:\n");
    printf("   ```bash\n");
    printf("   # Add to your CI pipeline\n");
    printf("   ./run_benchmarks --baseline production_v1.2.3 \\\n");
    printf("                    --current feature_branch \\\n");
    printf("                    --threshold 5%% \\\n");
    printf("                    --export-format json\n");
    printf("   ```\n\n");
    
    printf("2. 📊 Monitoring Dashboard:\n");
    printf("   ```c\n");
    printf("   // Production monitoring hook\n");
    printf("   trustformers_profiling_start_continuous(10000, 100);\n");
    printf("   \n");
    printf("   // Your inference loop\n");
    printf("   for (requests) {\n");
    printf("       trustformers_profiling_start_timer(\"inference\");\n");
    printf("       process_request(request);\n");
    printf("       trustformers_profiling_stop_timer(\"inference\", &time);\n");
    printf("       \n");
    printf("       if (time > SLA_THRESHOLD) {\n");
    printf("           alert_slow_request(time);\n");
    printf("       }\n");
    printf("   }\n");
    printf("   ```\n\n");
    
    printf("3. 🎯 A/B Testing:\n");
    printf("   ```c\n");
    printf("   // Compare model configurations\n");
    printf("   benchmark_config_a = run_benchmark(\"model_v1\", test_data);\n");
    printf("   benchmark_config_b = run_benchmark(\"model_v2\", test_data);\n");
    printf("   \n");
    printf("   if (statistical_significance(config_a, config_b) < 0.05) {\n");
    printf("       deploy_better_model();\n");
    printf("   }\n");
    printf("   ```\n\n");
    
    printf("4. 📈 Performance Regression Detection:\n");
    printf("   ```c\n");
    printf("   // Automated regression testing\n");
    printf("   current_perf = measure_inference_time();\n");
    printf("   regression_result = check_regression(\"inference\", current_perf, 5.0);\n");
    printf("   \n");
    printf("   if (regression_result.is_regression) {\n");
    printf("       send_alert(regression_result.details);\n");
    printf("       block_deployment();\n");
    printf("   }\n");
    printf("   ```\n\n");
    
    printf("📊 Visualization and Analysis Tools:\n");
    printf("  • 📈 Grafana dashboards for real-time monitoring\n");
    printf("  • 📊 Jupyter notebooks for detailed analysis\n");
    printf("  • 🔧 Custom analysis scripts using exported data\n");
    printf("  • 📋 Automated report generation and distribution\n");
}

int main() {
    printf("╔════════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                        TrustformeRS Profiling & Benchmarking Demo                     ║\n");
    printf("╚════════════════════════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("This comprehensive demo showcases advanced profiling and benchmarking capabilities\n");
    printf("for performance analysis, optimization, and production monitoring.\n");
    
    // Initialize TrustformeRS
    TrustformersError error = trustformers_init();
    if (error != TrustformersError_Success) {
        print_error(error);
        return 1;
    }
    
    printf("\n✅ TrustformeRS initialized successfully\n");
    
    // Seed random number generator for consistent simulation
    srand(time(NULL));
    
    // Run all demonstrations
    demonstrate_basic_timing();
    demonstrate_statistical_benchmarking();
    demonstrate_continuous_profiling();
    demonstrate_report_generation();
    demonstrate_performance_analysis();
    demonstrate_integration_examples();
    
    print_separator("Demo Summary and Next Steps");
    
    printf("🎉 Profiling and benchmarking demonstration completed!\n\n");
    
    printf("📁 Generated Files:\n");
    printf("  📄 performance_analysis.md - Comprehensive performance report\n");
    printf("  📊 benchmark_data.json - Detailed benchmark results in JSON format\n");
    printf("  📈 benchmark_data.csv - Benchmark data for spreadsheet analysis\n\n");
    
    printf("🚀 Next Steps:\n");
    printf("  1. 📖 Review the generated performance reports\n");
    printf("  2. 📊 Import CSV data into your preferred analysis tool\n");
    printf("  3. 🔧 Integrate profiling into your development workflow\n");
    printf("  4. 📈 Set up continuous monitoring for production systems\n");
    printf("  5. 🎯 Implement automated regression testing\n\n");
    
    printf("💡 Pro Tips:\n");
    printf("  • Use profiling data to guide optimization efforts\n");
    printf("  • Set up alerts for performance regressions\n");
    printf("  • Regularly benchmark with production-like data\n");
    printf("  • Share performance insights with your team\n");
    printf("  • Archive benchmarks for historical analysis\n\n");
    
    printf("🔗 Integration Resources:\n");
    printf("  • CI/CD pipeline templates\n");
    printf("  • Monitoring dashboard configurations\n");
    printf("  • Performance analysis notebooks\n");
    printf("  • Alerting and notification setups\n");
    
    // Cleanup
    trustformers_cleanup();
    printf("\n✅ Demo cleanup completed\n");
    
    return 0;
}