//! Example demonstrating the Mobile Integration Testing Framework
//!
//! This example shows how to set up and run comprehensive integration tests
//! for TrustformersRS mobile implementations across different platforms and backends.

use trustformers_mobile::integration_testing::{DataSizeVariant, InputDataType, TestDataConfig};
use trustformers_mobile::{
    BackendTestingConfig, CompatibilityTestingConfig, IntegrationTestConfig,
    MobileIntegrationTestFramework, PerformanceTestingConfig, PlatformTestingConfig, ReportFormat,
    TestConfiguration, TestReportingConfig,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 TrustformersRS Mobile Integration Testing Framework Example");
    println!("============================================================");

    // Configure integration testing
    let test_config = create_comprehensive_test_config();

    // Create the test framework
    let mut test_framework = MobileIntegrationTestFramework::new(test_config)?;

    println!("📝 Configuration created successfully");
    println!("🚀 Starting comprehensive integration tests...\n");

    // Run comprehensive integration tests
    let test_results = test_framework.run_integration_tests().await?;

    println!("✅ Integration tests completed!");
    println!("📊 Generating comprehensive test report...\n");

    // Generate and display test report
    let report = test_framework.generate_test_report(&test_results)?;

    // Print summary to console
    print_test_summary(&test_results);

    // Save detailed report to file
    std::fs::write("integration_test_report.json", &report)?;
    println!("📄 Detailed test report saved to: integration_test_report.json");

    // Print recommendations
    print_recommendations(&test_results);

    Ok(())
}

/// Create a comprehensive test configuration
fn create_comprehensive_test_config() -> IntegrationTestConfig {
    IntegrationTestConfig {
        enabled: true,
        test_config: TestConfiguration {
            timeout_seconds: 600, // 10 minutes
            iterations: 5,
            parallel_execution: true,
            max_concurrent_tests: 8,
            test_data: TestDataConfig {
                use_synthetic_data: true,
                data_size_variants: vec![
                    DataSizeVariant::Small,
                    DataSizeVariant::Medium,
                    DataSizeVariant::Large,
                ],
                input_data_types: vec![
                    InputDataType::Float32,
                    InputDataType::Float16,
                    InputDataType::Int8,
                ],
                batch_size_variants: vec![1, 4, 8, 16, 32],
                sequence_length_variants: vec![64, 128, 256, 512],
            },
            resource_constraints: trustformers_mobile::integration_testing::ResourceConstraints {
                max_memory_mb: 4096,
                max_cpu_usage: 85.0,
                max_test_duration: 3600, // 1 hour
                max_disk_usage: 2048,
                network_limits: trustformers_mobile::integration_testing::NetworkLimits {
                    max_bandwidth_mbps: 1000.0,
                    max_requests_per_second: 200,
                    timeout_seconds: 60,
                },
            },
        },
        platform_testing: PlatformTestingConfig {
            test_ios: true,
            test_android: true,
            test_generic: true,
            ios_config: trustformers_mobile::integration_testing::IOsTestConfig {
                test_coreml_integration: true,
                test_metal_acceleration: true,
                test_arkit_integration: true,
                test_app_extensions: true,
                test_background_processing: true,
                test_icloud_sync: true,
                ios_version_range: trustformers_mobile::integration_testing::VersionRange {
                    min_version: "14.0".to_string(),
                    max_version: "17.0".to_string(),
                    include_prereleases: false,
                },
                device_compatibility: Vec::new(),
            },
            android_config: trustformers_mobile::integration_testing::AndroidTestConfig {
                test_nnapi_integration: true,
                test_gpu_acceleration: true,
                test_edge_tpu: true,
                test_work_manager: true,
                test_content_provider: true,
                test_doze_compatibility: true,
                api_level_range: trustformers_mobile::integration_testing::ApiLevelRange {
                    min_api_level: 21,
                    max_api_level: 34,
                },
                device_compatibility: Vec::new(),
            },
            cross_platform_config:
                trustformers_mobile::integration_testing::CrossPlatformTestConfig {
                    test_data_consistency: true,
                    test_api_consistency: true,
                    test_performance_parity: true,
                    test_behavior_consistency: true,
                    test_serialization_compatibility: true,
                },
        },
        backend_testing: BackendTestingConfig {
            test_cpu: true,
            test_coreml: true,
            test_nnapi: true,
            test_gpu: true,
            test_custom: false,
            test_backend_switching: true,
            test_fallback_mechanisms: true,
        },
        performance_testing: PerformanceTestingConfig {
            enabled: true,
            memory_testing: trustformers_mobile::integration_testing::MemoryTestConfig {
                test_memory_patterns: true,
                test_memory_leaks: true,
                test_memory_pressure: true,
                test_optimization_levels: true,
                memory_thresholds: trustformers_mobile::integration_testing::MemoryThresholds {
                    max_usage_mb: 2048,
                    leak_threshold_mb: 100,
                    pressure_threshold_percentage: 90.0,
                },
            },
            latency_testing: trustformers_mobile::integration_testing::LatencyTestConfig {
                test_inference_latency: true,
                test_initialization_latency: true,
                test_model_loading_latency: true,
                test_backend_switching_latency: true,
                latency_thresholds: trustformers_mobile::integration_testing::LatencyThresholds {
                    max_inference_ms: 50.0,
                    max_initialization_ms: 3000.0,
                    max_model_loading_ms: 8000.0,
                },
            },
            throughput_testing: trustformers_mobile::integration_testing::ThroughputTestConfig {
                test_inference_throughput: true,
                test_batch_throughput: true,
                test_concurrent_throughput: true,
                throughput_thresholds:
                    trustformers_mobile::integration_testing::ThroughputThresholds {
                        min_inferences_per_second: 20.0,
                        min_batch_throughput: 100.0,
                        min_concurrent_throughput: 50.0,
                    },
            },
            power_testing: trustformers_mobile::integration_testing::PowerTestConfig {
                test_power_consumption: true,
                test_battery_impact: true,
                test_thermal_impact: true,
                test_power_optimization: true,
                power_thresholds: trustformers_mobile::integration_testing::PowerThresholds {
                    max_power_consumption_mw: 3000.0,
                    max_battery_drain_percentage_per_hour: 8.0,
                    max_thermal_impact_celsius: 50.0,
                },
            },
            thermal_testing: trustformers_mobile::integration_testing::ThermalTestConfig {
                test_thermal_management: true,
                test_throttling_behavior: true,
                test_thermal_recovery: true,
                thermal_thresholds: trustformers_mobile::integration_testing::ThermalThresholds {
                    max_temperature_celsius: 85.0,
                    throttling_threshold_celsius: 75.0,
                    recovery_threshold_celsius: 65.0,
                },
            },
            load_testing: trustformers_mobile::integration_testing::LoadTestConfig {
                test_sustained_load: true,
                test_peak_load: true,
                test_load_distribution: true,
                test_stress_scenarios: true,
                load_parameters: trustformers_mobile::integration_testing::LoadTestParameters {
                    concurrent_users: 20,
                    requests_per_second: 100.0,
                    test_duration_seconds: 600,
                    ramp_up_time_seconds: 120,
                },
            },
        },
        compatibility_testing: CompatibilityTestingConfig {
            framework_compatibility:
                trustformers_mobile::integration_testing::FrameworkCompatibilityConfig {
                    test_react_native: true,
                    test_flutter: true,
                    test_unity: true,
                    test_native: true,
                    framework_versions: std::collections::HashMap::new(),
                },
            version_compatibility:
                trustformers_mobile::integration_testing::VersionCompatibilityConfig {
                    test_backward_compatibility: true,
                    test_forward_compatibility: true,
                    test_version_migration: true,
                    version_range: trustformers_mobile::integration_testing::VersionRange {
                        min_version: "1.0.0".to_string(),
                        max_version: "2.0.0".to_string(),
                        include_prereleases: false,
                    },
                },
            model_compatibility:
                trustformers_mobile::integration_testing::ModelCompatibilityConfig {
                    test_model_formats: true,
                    test_quantization_variants: true,
                    test_size_variants: true,
                    test_custom_models: true,
                    model_parameters:
                        trustformers_mobile::integration_testing::ModelCompatibilityParameters {
                            supported_formats: vec![
                                "tflite".to_string(),
                                "onnx".to_string(),
                                "coreml".to_string(),
                                "pytorch".to_string(),
                            ],
                            supported_quantizations: vec![
                                "fp32".to_string(),
                                "fp16".to_string(),
                                "int8".to_string(),
                                "int4".to_string(),
                            ],
                            max_model_size_mb: 1000,
                            min_model_size_kb: 50,
                        },
                },
            api_compatibility: trustformers_mobile::integration_testing::ApiCompatibilityConfig {
                test_api_consistency: true,
                test_parameter_validation: true,
                test_error_handling: true,
                test_return_value_consistency: true,
                api_version_compatibility: trustformers_mobile::integration_testing::VersionRange {
                    min_version: "1.0.0".to_string(),
                    max_version: "2.0.0".to_string(),
                    include_prereleases: false,
                },
            },
        },
        reporting: TestReportingConfig {
            output_format: ReportFormat::JSON,
            include_metrics: true,
            include_graphs: true,
            include_error_analysis: true,
            export_to_file: true,
            report_file_path: "comprehensive_integration_test_report.json".to_string(),
            include_recommendations: true,
        },
    }
}

/// Print a summary of test results to the console
fn print_test_summary(results: &trustformers_mobile::IntegrationTestResults) {
    println!("📈 Test Results Summary");
    println!("======================");
    println!("Total Tests: {}", results.summary.total_tests);
    println!("Passed: {} ✅", results.summary.passed_tests);
    println!("Failed: {} ❌", results.summary.failed_tests);
    println!("Skipped: {} ⏭️", results.summary.skipped_tests);
    println!("Success Rate: {:.1}% 📊", results.summary.success_rate);
    println!(
        "Total Duration: {:.2}s ⏱️",
        results.summary.total_duration.as_secs_f64()
    );
    println!();

    // Platform-specific results
    println!("📱 Platform Results:");
    for (platform, platform_result) in &results.platform_results {
        println!("  {:?}:", platform);
        println!("    Tests: {}", platform_result.test_results.len());
        println!(
            "    Avg Latency: {:.1}ms",
            platform_result.performance_metrics.avg_inference_latency_ms
        );
        println!(
            "    Avg Memory: {:.1}MB",
            platform_result.performance_metrics.avg_memory_usage_mb
        );
        println!(
            "    Compatibility: {:.1}%",
            platform_result.compatibility_scores.overall_compatibility
        );
    }
    println!();

    // Backend-specific results
    println!("⚙️ Backend Results:");
    for (backend, backend_result) in &results.backend_results {
        println!("  {:?}:", backend);
        println!("    Tests: {}", backend_result.test_results.len());
        println!(
            "    Avg Latency: {:.1}ms",
            backend_result.performance_metrics.avg_inference_latency_ms
        );
        println!(
            "    Throughput: {:.1} inf/s",
            backend_result.performance_metrics.throughput_inferences_per_second
        );
        println!(
            "    Efficiency: {:.1}%",
            backend_result.performance_metrics.power_efficiency_score
        );
    }
    println!();

    // Performance benchmarks
    println!("🚀 Performance Benchmarks:");
    println!("  Memory:");
    println!(
        "    Peak Usage: {:.1}MB",
        results.performance_results.memory_benchmarks.peak_memory_usage_mb
    );
    println!(
        "    Leaks Detected: {}",
        results.performance_results.memory_benchmarks.memory_leaks_detected
    );
    println!(
        "    Efficiency: {:.1}%",
        results.performance_results.memory_benchmarks.memory_efficiency_score
    );
    println!("  Latency:");
    println!(
        "    Avg Inference: {:.1}ms",
        results.performance_results.latency_benchmarks.avg_inference_latency_ms
    );
    println!(
        "    P95 Inference: {:.1}ms",
        results.performance_results.latency_benchmarks.p95_inference_latency_ms
    );
    println!(
        "    P99 Inference: {:.1}ms",
        results.performance_results.latency_benchmarks.p99_inference_latency_ms
    );
    println!("  Throughput:");
    println!(
        "    Max: {:.1} inf/s",
        results
            .performance_results
            .throughput_benchmarks
            .max_throughput_inferences_per_second
    );
    println!(
        "    Sustained: {:.1} inf/s",
        results
            .performance_results
            .throughput_benchmarks
            .sustained_throughput_inferences_per_second
    );
    println!();

    // Cross-platform comparison
    println!("🔄 Cross-Platform Compatibility:");
    println!(
        "  Data Consistency: {:.1}%",
        results.cross_platform_comparison.data_consistency_score
    );
    println!(
        "  API Consistency: {:.1}%",
        results.cross_platform_comparison.api_consistency_score
    );
    println!(
        "  Performance Parity: {:.1}%",
        results.cross_platform_comparison.performance_parity_score
    );
    println!(
        "  Behavior Consistency: {:.1}%",
        results.cross_platform_comparison.behavior_consistency_score
    );
    println!();
}

/// Print recommendations based on test results
fn print_recommendations(results: &trustformers_mobile::IntegrationTestResults) {
    if results.recommendations.is_empty() {
        println!("🎉 No specific recommendations - all tests passed with good performance!");
        return;
    }

    println!("💡 Recommendations");
    println!("==================");

    for (i, recommendation) in results.recommendations.iter().enumerate() {
        let priority_emoji = match recommendation.priority {
            trustformers_mobile::RecommendationPriority::Critical => "🚨",
            trustformers_mobile::RecommendationPriority::High => "⚠️",
            trustformers_mobile::RecommendationPriority::Medium => "📋",
            trustformers_mobile::RecommendationPriority::Low => "💭",
        };

        let type_emoji = match recommendation.recommendation_type {
            trustformers_mobile::RecommendationType::Performance => "🚀",
            trustformers_mobile::RecommendationType::Compatibility => "🔗",
            trustformers_mobile::RecommendationType::Reliability => "🛡️",
            trustformers_mobile::RecommendationType::Security => "🔒",
            trustformers_mobile::RecommendationType::Usability => "👤",
            trustformers_mobile::RecommendationType::Maintenance => "🔧",
        };

        println!(
            "{}. {} {} {}",
            i + 1,
            priority_emoji,
            type_emoji,
            recommendation.title
        );
        println!("   Priority: {:?}", recommendation.priority);
        println!("   Description: {}", recommendation.description);
        println!("   Effort: {:?}", recommendation.implementation_effort);
        println!("   Expected Impact: {:?}", recommendation.expected_impact);
        println!("   Platforms: {:?}", recommendation.platforms_affected);

        if !recommendation.actions.is_empty() {
            println!("   Actions:");
            for action in &recommendation.actions {
                println!("     • {}", action);
            }
        }
        println!();
    }
}

/// Example of running focused tests for specific scenarios
#[allow(dead_code)]
async fn run_focused_ios_coreml_tests() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Running focused iOS Core ML tests...");

    let mut config = IntegrationTestConfig::default();

    // Focus on iOS and Core ML only
    config.platform_testing.test_android = false;
    config.platform_testing.test_generic = false;
    config.backend_testing.test_cpu = false;
    config.backend_testing.test_nnapi = false;
    config.backend_testing.test_gpu = false;

    // Enable detailed iOS-specific tests
    config.platform_testing.ios_config.test_coreml_integration = true;
    config.platform_testing.ios_config.test_metal_acceleration = true;
    config.platform_testing.ios_config.test_arkit_integration = true;

    let mut test_framework = MobileIntegrationTestFramework::new(config)?;
    let results = test_framework.run_integration_tests().await?;

    println!("✅ Focused iOS Core ML tests completed!");
    print_test_summary(&results);

    Ok(())
}

/// Example of running performance-only benchmarks
#[allow(dead_code)]
async fn run_performance_benchmarks_only() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Running performance benchmarks only...");

    let mut config = IntegrationTestConfig::default();

    // Disable non-performance tests
    config.platform_testing.test_ios = false;
    config.platform_testing.test_android = false;
    config.platform_testing.test_generic = false;
    config.backend_testing.test_backend_switching = false;
    config.backend_testing.test_fallback_mechanisms = false;
    config.compatibility_testing.framework_compatibility.test_react_native = false;
    config.compatibility_testing.framework_compatibility.test_flutter = false;
    config.compatibility_testing.framework_compatibility.test_unity = false;

    // Focus on performance testing
    config.performance_testing.enabled = true;
    config.performance_testing.memory_testing.test_memory_patterns = true;
    config.performance_testing.latency_testing.test_inference_latency = true;
    config.performance_testing.throughput_testing.test_inference_throughput = true;
    config.performance_testing.power_testing.test_power_consumption = true;
    config.performance_testing.load_testing.test_sustained_load = true;

    let mut test_framework = MobileIntegrationTestFramework::new(config)?;
    let results = test_framework.run_integration_tests().await?;

    println!("✅ Performance benchmarks completed!");

    // Focus on performance metrics in the output
    println!("🏆 Performance Results:");
    println!(
        "  Peak Memory: {:.1}MB",
        results.performance_results.memory_benchmarks.peak_memory_usage_mb
    );
    println!(
        "  Avg Latency: {:.1}ms",
        results.performance_results.latency_benchmarks.avg_inference_latency_ms
    );
    println!(
        "  Max Throughput: {:.1} inf/s",
        results
            .performance_results
            .throughput_benchmarks
            .max_throughput_inferences_per_second
    );
    println!(
        "  Avg Power: {:.1}mW",
        results.performance_results.power_benchmarks.avg_power_consumption_mw
    );

    Ok(())
}
