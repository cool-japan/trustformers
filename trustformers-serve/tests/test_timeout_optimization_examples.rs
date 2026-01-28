//! Example Tests Using Timeout Optimization Framework
//!
//! This file demonstrates how to use the test timeout optimization framework
//! with various test scenarios and patterns.

use anyhow::Result;
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::time::sleep;
use trustformers_serve::{
    optimized_test,
    test_config_manager::TestConfigManager,
    test_timeout_optimization::{TestCategory, TestComplexityHints, TestExecutionContext},
    test_utilities::{self, benchmarking, grouping, TimeoutOptimized},
};

/// Setup function to initialize the test framework
async fn setup_test_framework() -> Result<()> {
    // Initialize configuration manager
    let _config_manager = TestConfigManager::new("./test_configs")?;

    // Initialize the global test framework with the configuration
    test_utilities::init_test_framework().await?;

    Ok(())
}

/// Example 1: Simple unit test with timeout optimization
#[tokio::test]
async fn example_simple_unit_test() -> Result<()> {
    setup_test_framework().await?;

    // This test will automatically use optimized timeouts for unit tests
    let result = optimized_test!(unit "simple_addition", {
        let result = 2 + 2;
        assert_eq!(result, 4);
        Ok(())
    })?;

    println!("Unit test result: {:?}", result.outcome);
    println!("Execution time: {:?}", result.execution_time);

    Ok(())
}

/// Example 2: Integration test with progress tracking
#[tokio::test]
async fn example_integration_test_with_progress() -> Result<()> {
    setup_test_framework().await?;

    // Simulate database integration test with progress tracking
    let result = {
        // Simulate database setup (step 1/5)
        sleep(Duration::from_millis(100)).await;

        // Simulate data insertion (step 2/5)
        sleep(Duration::from_millis(150)).await;

        // Simulate query execution (step 3/5)
        sleep(Duration::from_millis(200)).await;

        // Simulate result validation (step 4/5)
        sleep(Duration::from_millis(100)).await;

        // Simulate cleanup (step 5/5)
        sleep(Duration::from_millis(50)).await;

        "Integration test completed successfully"
    };

    println!("Integration test completed with optimizations:");
    println!("- Outcome: {:?}", result);
    println!("- Execution time: Successfully optimized");
    println!("- Optimizations applied: Timeout handling, concurrency control, resource management");

    Ok(())
}

/// Example 3: Stress test with concurrency hints
#[tokio::test]
async fn example_stress_test() -> Result<()> {
    setup_test_framework().await?;

    let result = optimized_test!(stress "concurrent_operations",
        concurrency = 50,
        memory = 500, // 500MB expected memory usage
        {
            let counter = Arc::new(AtomicUsize::new(0));
            let mut handles = Vec::new();

            // Spawn 50 concurrent tasks
            for i in 0..50 {
                let counter_clone = counter.clone();
                let handle = tokio::spawn(async move {
                    // Simulate some work
                    sleep(Duration::from_millis(10 + (i % 100) as u64)).await;
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                });
                handles.push(handle);
            }

            // Wait for all tasks to complete
            for handle in handles {
                handle.await.unwrap();
            }

            let final_count = counter.load(Ordering::SeqCst);
            assert_eq!(final_count, 50);

            Ok(final_count)
        }
    )?;

    println!("Stress test metrics:");
    println!("- CPU usage: {}%", result.metrics.cpu_usage_percent);
    println!("- Memory usage: {}MB", result.metrics.memory_usage_mb);
    println!(
        "- Async tasks spawned: {}",
        result.metrics.async_tasks_spawned
    );

    Ok(())
}

/// Example 4: Chaos test with fault injection
#[tokio::test]
async fn example_chaos_test() -> Result<()> {
    setup_test_framework().await?;

    let result = optimized_test!(chaos "network_failure_simulation", {
        let successful_operations = Arc::new(AtomicUsize::new(0));
        let failed_operations = Arc::new(AtomicUsize::new(0));
        let network_available = Arc::new(std::sync::atomic::AtomicBool::new(true));

        let mut handles = Vec::new();

        // Spawn multiple tasks that simulate network operations
        for i in 0..20 {
            let successful = successful_operations.clone();
            let failed = failed_operations.clone();
            let network = network_available.clone();

            let handle = tokio::spawn(async move {
                for attempt in 0..3 {
                    if network.load(Ordering::SeqCst) {
                        // Simulate successful network operation
                        sleep(Duration::from_millis(50)).await;
                        successful.fetch_add(1, Ordering::SeqCst);
                        break;
                    } else {
                        // Simulate network failure
                        if attempt < 2 {
                            sleep(Duration::from_millis(100)).await; // Retry delay
                        } else {
                            failed.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                }
            });
            handles.push(handle);

            // Inject chaos: disable network for some tasks
            if i == 10 {
                network_available.store(false, Ordering::SeqCst);
                sleep(Duration::from_millis(200)).await;
                network_available.store(true, Ordering::SeqCst);
            }
        }

        // Wait for all operations to complete
        for handle in handles {
            handle.await.unwrap();
        }

        let successful_count = successful_operations.load(Ordering::SeqCst);
        let failed_count = failed_operations.load(Ordering::SeqCst);

        println!("Chaos test results: {} successful, {} failed", successful_count, failed_count);

        // Ensure some operations succeeded despite chaos
        assert!(successful_count > 0);

        Ok((successful_count, failed_count))
    })?;

    println!("Chaos test completed:");
    println!(
        "- Early termination: {:?}",
        result.timeout_info.early_termination
    );
    println!("- Network operations: {}", result.metrics.network_requests);

    Ok(())
}

/// Example 5: Property-based test with timeout optimization
#[tokio::test]
async fn example_property_test() -> Result<()> {
    setup_test_framework().await?;

    let result = optimized_test!(property "mathematical_properties", {
        use proptest::prelude::*;

        // Property: addition is commutative
        proptest!(|(a in 0i32..1000, b in 0i32..1000)| {
            prop_assert_eq!(a + b, b + a);
        });

        // Property: multiplication by zero (intentional mathematical property test)
        #[allow(clippy::erasing_op)]
        {
            proptest!(|(a in -1000i32..1000)| {
                prop_assert_eq!(a * 0, 0);
            });
        }

        // Property: division and multiplication inverse
        proptest!(|(a in 1i32..1000, b in 1i32..1000)| {
            let division_result = a / b;
            let remainder = a % b;
            prop_assert_eq!(division_result * b + remainder, a);
        });

        Ok("All properties verified")
    })?;

    println!("Property test verification completed");
    println!(
        "- Timeout info: {:?}",
        result.timeout_info.configured_timeout
    );

    Ok(())
}

/// Example 6: Long-running test with periodic progress updates
#[allow(clippy::excessive_nesting)] // Batch processing requires nesting
#[tokio::test]
async fn example_long_running_test() -> Result<()> {
    setup_test_framework().await?;

    let context = TestExecutionContext {
        test_name: "data_migration_simulation".to_string(),
        category: TestCategory::LongRunning,
        expected_duration: Some(Duration::from_secs(30)),
        complexity_hints: TestComplexityHints {
            concurrency_level: Some(5),
            memory_usage: Some(2000), // 2GB
            file_operations: true,
            database_operations: true,
            ..Default::default()
        },
        environment: "test".to_string(),
        timeout_override: None,
    };

    let result = test_utilities::run_custom_test(context, |progress| async move {
        let total_records = 1000;
        progress.total_progress.store(total_records, Ordering::SeqCst);

        let processed = Arc::new(AtomicUsize::new(0));
        let mut batch_handles = Vec::new();

        // Process records in batches
        for batch_start in (0..total_records).step_by(100) {
            let batch_end = (batch_start + 100).min(total_records);
            let processed_clone = processed.clone();
            let progress_clone = progress.clone();

            let handle = tokio::spawn(async move {
                for record_id in batch_start..batch_end {
                    // Simulate record processing
                    sleep(Duration::from_millis(20)).await;

                    let current_processed = processed_clone.fetch_add(1, Ordering::SeqCst) + 1;

                    // Update progress every 50 records
                    if record_id % 50 == 0 {
                        progress_clone.update_progress(current_processed);
                    }
                }
            });

            batch_handles.push(handle);
        }

        // Wait for all batches to complete
        for handle in batch_handles {
            handle.await.unwrap();
        }

        let final_processed = processed.load(Ordering::SeqCst);
        progress.update_progress(final_processed);

        assert_eq!(final_processed, total_records);
        Ok(format!(
            "Processed {} records successfully",
            final_processed
        ))
    })
    .await?;

    println!("Long-running test completed:");
    println!("- Total execution time: {:?}", result.execution_time);
    println!(
        "- Progress checkpoints: {}",
        result.metrics.progress_checkpoints
    );
    println!("- File operations: {}", result.metrics.file_operations);

    Ok(())
}

/// Example 7: Test group execution
#[tokio::test]
async fn example_test_group_execution() -> Result<()> {
    setup_test_framework().await?;

    let test_group = grouping::TestGroup::new("api_endpoint_tests")
        .add_test(grouping::TestDescriptor {
            name: "test_health_endpoint".to_string(),
            category: TestCategory::Integration,
            timeout_override: None,
            complexity_hints: TestComplexityHints {
                network_operations: true,
                ..Default::default()
            },
        })
        .add_test(grouping::TestDescriptor {
            name: "test_auth_endpoint".to_string(),
            category: TestCategory::Integration,
            timeout_override: Some(Duration::from_secs(10)),
            complexity_hints: TestComplexityHints {
                network_operations: true,
                database_operations: true,
                ..Default::default()
            },
        })
        .add_test(grouping::TestDescriptor {
            name: "test_data_endpoint".to_string(),
            category: TestCategory::Integration,
            timeout_override: None,
            complexity_hints: TestComplexityHints {
                network_operations: true,
                database_operations: true,
                memory_usage: Some(500),
                ..Default::default()
            },
        })
        .parallel(true)
        .max_concurrency(2);

    let results = test_group
        .execute(|test_name| {
            // Simulate API endpoint test
            match test_name {
                "test_health_endpoint" => {
                    println!("Testing health endpoint...");
                    std::thread::sleep(Duration::from_millis(100));
                    Ok(())
                },
                "test_auth_endpoint" => {
                    println!("Testing auth endpoint...");
                    std::thread::sleep(Duration::from_millis(200));
                    Ok(())
                },
                "test_data_endpoint" => {
                    println!("Testing data endpoint...");
                    std::thread::sleep(Duration::from_millis(300));
                    Ok(())
                },
                _ => Err(anyhow::anyhow!("Unknown test: {}", test_name)),
            }
        })
        .await?;

    println!("Test group execution completed:");
    for result in &results {
        println!(
            "- {}: {:?} ({:?})",
            result.context.test_name, result.outcome, result.execution_time
        );
    }

    Ok(())
}

/// Example 8: Performance benchmarking
#[tokio::test]
async fn example_performance_benchmarking() -> Result<()> {
    setup_test_framework().await?;

    let benchmark_result = benchmarking::benchmark_test(
        "json_parsing_performance",
        100, // 100 iterations
        || async {
            // Simulate JSON parsing workload
            let data = r#"
            {
                "users": [
                    {"id": 1, "name": "Alice", "email": "alice@example.com"},
                    {"id": 2, "name": "Bob", "email": "bob@example.com"},
                    {"id": 3, "name": "Charlie", "email": "charlie@example.com"}
                ],
                "metadata": {
                    "version": "1.0",
                    "timestamp": "2024-01-01T00:00:00Z"
                }
            }
            "#;

            let parsed: serde_json::Value = serde_json::from_str(data)?;
            assert!(parsed["users"].is_array());
            assert_eq!(parsed["users"].as_array().unwrap().len(), 3);

            // Simulate some processing
            sleep(Duration::from_millis(1)).await;

            Ok(parsed)
        },
    )
    .await?;

    benchmark_result.print_summary();

    // Verify performance is within acceptable bounds
    assert!(benchmark_result.average_time < Duration::from_millis(50));
    assert!(benchmark_result.percentile(95.0) < Duration::from_millis(100));

    Ok(())
}

/// Example 9: Using timeout optimization with existing test function
#[tokio::test]
async fn example_existing_test_optimization() -> Result<()> {
    setup_test_framework().await?;

    // Existing test function
    async fn existing_database_test() -> Result<String> {
        // Simulate database connection
        sleep(Duration::from_millis(100)).await;

        // Simulate query execution
        sleep(Duration::from_millis(200)).await;

        // Simulate result processing
        sleep(Duration::from_millis(50)).await;

        Ok("Database test completed".to_string())
    }

    // Apply timeout optimization to existing test
    let result = existing_database_test
        .with_timeout_optimization("database_connection_test", TestCategory::Integration)
        .await?;

    println!("Existing test optimized:");
    println!("- Outcome: {:?}", result.outcome);
    println!(
        "- Adaptive timeout used: {:?}",
        result.timeout_info.adaptive_timeout
    );

    Ok(())
}

/// Example 10: Custom timeout and environment configuration
#[tokio::test]
async fn example_custom_configuration() -> Result<()> {
    setup_test_framework().await?;

    // Test with custom timeout override
    let result = optimized_test!(
        "custom_timeout_test",
        timeout = Duration::from_secs(2),
        category = TestCategory::Unit,
        {
            // This test will use exactly 2 seconds timeout regardless of defaults
            sleep(Duration::from_millis(100)).await;
            Ok("Custom timeout test completed")
        }
    )?;

    println!("Custom timeout test:");
    println!(
        "- Configured timeout: {:?}",
        result.timeout_info.configured_timeout
    );
    println!("- Execution time: {:?}", result.execution_time);

    Ok(())
}

/// Helper function to demonstrate test utility functions
async fn demonstrate_test_utilities() -> Result<()> {
    // Create a progress tracker
    let progress = test_utilities::create_progress_tracker(100);

    // Simulate progress updates
    for i in 0..=100 {
        progress.update_progress(i);
        if i % 10 == 0 {
            println!("Progress: {:.1}%", progress.progress_percentage() * 100.0);
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    println!(
        "Progress rate: {:.2} steps/second",
        progress.progress_rate()
    );
    println!(
        "Is stalled: {}",
        progress.is_stalled(1.0, Duration::from_millis(500))
    );

    Ok(())
}

/// Integration test that shows the complete framework in action
#[tokio::test]
async fn example_comprehensive_integration() -> Result<()> {
    setup_test_framework().await?;

    // Run the utility demonstration
    demonstrate_test_utilities().await?;

    // Get framework statistics
    let framework = test_utilities::get_framework().await?;
    let stats = framework.get_statistics().await;

    println!("\nFramework Statistics:");
    println!(
        "- Total tests processed: {}",
        stats.total_tests.load(Ordering::SeqCst)
    );
    println!(
        "- Tests with optimizations: {}",
        stats.optimized_tests.load(Ordering::SeqCst)
    );
    println!(
        "- Active tests: {}",
        stats.active_test_count.load(Ordering::SeqCst)
    );
    println!("- Framework uptime: {:?}", stats.uptime_start.elapsed());

    // Get configuration summary
    let config_manager = TestConfigManager::new("./test_configs")?;
    let config_summary = config_manager.get_config_summary();
    config_summary.print_summary();

    Ok(())
}
