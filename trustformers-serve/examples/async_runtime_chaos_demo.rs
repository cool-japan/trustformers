//! Async Runtime Chaos Testing Demo
#![allow(unused_variables)]
//!
//! This example demonstrates how to use the async runtime chaos testing framework
//! to test resilience under various async runtime edge cases and failure scenarios.

use anyhow::Result;
use std::time::{Duration, Instant};
use trustformers_serve::async_runtime_chaos::{
    AsyncMemoryPressureConfig, AsyncRuntimeChaosFramework, CancellationStrategy, DeadlockConfig,
    PanicRecoveryConfig, PanicType, RuntimeShutdownConfig, TaskCancellationConfig,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🔬 TrustformeRS Async Runtime Chaos Testing Demo");
    println!("==============================================");

    // Create async runtime chaos testing framework
    let framework = AsyncRuntimeChaosFramework::new();
    framework.start().await?;

    println!("✅ Initialized Async Runtime Chaos Testing Framework");

    // Demo 1: Task Cancellation Chaos Testing
    println!("\n🔄 Demo 1: Task Cancellation Chaos Testing");
    println!("------------------------------------------");

    demo_task_cancellation(&framework).await?;

    // Demo 2: Runtime Shutdown Testing
    println!("\n💻 Demo 2: Runtime Shutdown Testing");
    println!("-----------------------------------");

    demo_runtime_shutdown(&framework).await?;

    // Demo 3: Deadlock Detection Testing
    println!("\n🔒 Demo 3: Deadlock Detection Testing");
    println!("------------------------------------");

    demo_deadlock_detection(&framework).await?;

    // Demo 4: Memory Pressure During Async Operations
    println!("\n💾 Demo 4: Memory Pressure During Async Operations");
    println!("------------------------------------------------");

    demo_async_memory_pressure(&framework).await?;

    // Demo 5: Panic Recovery Testing
    println!("\n⚡ Demo 5: Panic Recovery Testing");
    println!("--------------------------------");

    demo_panic_recovery(&framework).await?;

    // Demo 6: Comprehensive Test Suite
    println!("\n🎯 Demo 6: Comprehensive Test Suite");
    println!("----------------------------------");

    demo_comprehensive_suite(&framework).await?;

    // Demo 7: Real-world Integration Examples
    println!("\n🌍 Demo 7: Real-world Integration Examples");
    println!("------------------------------------------");

    demo_real_world_integration(&framework).await?;

    println!("\n🎉 Async runtime chaos testing demo completed!");
    println!("📊 All chaos testing scenarios have been demonstrated");
    println!("💡 Use these patterns to test your async code resilience");

    Ok(())
}

async fn demo_task_cancellation(framework: &AsyncRuntimeChaosFramework) -> Result<()> {
    println!("Testing task cancellation resilience...");

    // Test 1: Broadcast cancellation
    println!("  📡 Testing broadcast cancellation strategy");
    let broadcast_config = TaskCancellationConfig {
        task_count: 100,
        task_duration: Duration::from_secs(5),
        cancellation_delay: Duration::from_millis(100),
        cancellation_strategy: CancellationStrategy::BroadcastCancel,
        graceful_timeout: Duration::from_secs(2),
        completion_timeout: Duration::from_secs(10),
    };

    let result = framework.test_task_cancellation(broadcast_config).await?;
    print_experiment_results("Broadcast Cancellation", &result);

    // Test 2: Selective cancellation
    println!("  🎯 Testing selective cancellation strategy (50% of tasks)");
    let selective_config = TaskCancellationConfig {
        task_count: 80,
        cancellation_strategy: CancellationStrategy::SelectiveCancel(50.0),
        ..Default::default()
    };

    let result = framework.test_task_cancellation(selective_config).await?;
    print_experiment_results("Selective Cancellation", &result);

    // Test 3: Graceful shutdown
    println!("  🛑 Testing graceful shutdown strategy");
    let graceful_config = TaskCancellationConfig {
        task_count: 60,
        cancellation_strategy: CancellationStrategy::GracefulShutdown,
        graceful_timeout: Duration::from_secs(3),
        ..Default::default()
    };

    let result = framework.test_task_cancellation(graceful_config).await?;
    print_experiment_results("Graceful Shutdown", &result);

    Ok(())
}

async fn demo_runtime_shutdown(framework: &AsyncRuntimeChaosFramework) -> Result<()> {
    println!("Testing runtime shutdown during active operations...");

    // Test different runtime configurations
    let configs = vec![
        (
            "Small Runtime",
            RuntimeShutdownConfig {
                worker_threads: 2,
                concurrent_operations: 10,
                operation_duration: Duration::from_secs(5),
                startup_delay: Duration::from_millis(100),
                graceful_shutdown_timeout: Duration::from_secs(2),
            },
        ),
        (
            "Medium Runtime",
            RuntimeShutdownConfig {
                worker_threads: 4,
                concurrent_operations: 25,
                operation_duration: Duration::from_secs(8),
                startup_delay: Duration::from_millis(200),
                graceful_shutdown_timeout: Duration::from_secs(3),
            },
        ),
        (
            "Large Runtime",
            RuntimeShutdownConfig {
                worker_threads: 8,
                concurrent_operations: 50,
                operation_duration: Duration::from_secs(10),
                startup_delay: Duration::from_millis(300),
                graceful_shutdown_timeout: Duration::from_secs(5),
            },
        ),
    ];

    for (test_name, config) in configs {
        println!(
            "  🏗️  Testing {} shutdown scenario",
            test_name.to_lowercase()
        );
        let result = framework.test_runtime_shutdown(config).await?;
        print_experiment_results(test_name, &result);
    }

    Ok(())
}

async fn demo_deadlock_detection(framework: &AsyncRuntimeChaosFramework) -> Result<()> {
    println!("Testing deadlock detection in async code...");

    // Test different deadlock scenarios
    let scenarios = vec![
        (
            "Quick Deadlock",
            DeadlockConfig {
                deadlock_delay: Duration::from_millis(10),
                deadlock_timeout: Duration::from_millis(50),
                detection_timeout: Duration::from_millis(200),
            },
        ),
        (
            "Moderate Deadlock",
            DeadlockConfig {
                deadlock_delay: Duration::from_millis(100),
                deadlock_timeout: Duration::from_millis(500),
                detection_timeout: Duration::from_secs(2),
            },
        ),
        (
            "Slow Deadlock",
            DeadlockConfig {
                deadlock_delay: Duration::from_millis(500),
                deadlock_timeout: Duration::from_secs(2),
                detection_timeout: Duration::from_secs(5),
            },
        ),
    ];

    for (scenario_name, config) in scenarios {
        println!("  🔍 Testing {} scenario", scenario_name.to_lowercase());
        let result = framework.test_deadlock_detection(config).await?;
        print_experiment_results(scenario_name, &result);
    }

    Ok(())
}

async fn demo_async_memory_pressure(framework: &AsyncRuntimeChaosFramework) -> Result<()> {
    println!("Testing memory pressure during async operations...");

    // Test different memory pressure levels
    let pressure_tests = vec![
        (
            "Light Pressure",
            AsyncMemoryPressureConfig {
                memory_pressure_mb: 100,
                concurrent_async_tasks: 10,
                pressure_duration: Duration::from_secs(2),
            },
        ),
        (
            "Moderate Pressure",
            AsyncMemoryPressureConfig {
                memory_pressure_mb: 250,
                concurrent_async_tasks: 20,
                pressure_duration: Duration::from_secs(3),
            },
        ),
        (
            "Heavy Pressure",
            AsyncMemoryPressureConfig {
                memory_pressure_mb: 500,
                concurrent_async_tasks: 30,
                pressure_duration: Duration::from_secs(4),
            },
        ),
    ];

    for (test_name, config) in pressure_tests {
        println!(
            "  💾 Testing {} ({}MB pressure)",
            test_name.to_lowercase(),
            config.memory_pressure_mb
        );
        let result = framework.test_async_memory_pressure(config).await?;
        print_experiment_results(test_name, &result);
    }

    Ok(())
}

async fn demo_panic_recovery(framework: &AsyncRuntimeChaosFramework) -> Result<()> {
    println!("Testing panic recovery in async tasks...");

    // Test different panic scenarios
    let panic_tests = vec![
        (
            "Immediate Panics",
            PanicRecoveryConfig {
                total_tasks: 30,
                panic_task_count: 5,
                panic_type: PanicType::Immediate,
            },
        ),
        (
            "Delayed Panics",
            PanicRecoveryConfig {
                total_tasks: 40,
                panic_task_count: 8,
                panic_type: PanicType::Delayed,
            },
        ),
        (
            "Conditional Panics",
            PanicRecoveryConfig {
                total_tasks: 50,
                panic_task_count: 10,
                panic_type: PanicType::ConditionalPanic,
            },
        ),
    ];

    for (test_name, config) in panic_tests {
        println!("  ⚡ Testing {} scenario", test_name.to_lowercase());
        let result = framework.test_async_panic_recovery(config).await?;
        print_experiment_results(test_name, &result);
    }

    Ok(())
}

async fn demo_comprehensive_suite(framework: &AsyncRuntimeChaosFramework) -> Result<()> {
    println!("Running comprehensive async runtime chaos test suite...");

    let start_time = Instant::now();
    let suite_result = framework.run_comprehensive_test_suite().await?;
    let total_duration = start_time.elapsed();

    println!("  📊 Suite Results Summary:");
    println!("    • Total Duration: {:.2}s", total_duration.as_secs_f64());
    println!("    • Tests Run: {}", suite_result.results.len());
    println!(
        "    • Success Rate: {:.1}%",
        suite_result.success_rate * 100.0
    );

    println!("  📈 Individual Test Results:");
    for (test_name, result) in &suite_result.results {
        let status = if result.success { "✅ PASS" } else { "❌ FAIL" };
        let duration = result
            .metrics
            .get("experiment_duration_ms")
            .map(|d| format!(" ({:.0}ms)", d))
            .unwrap_or_default();

        println!("    • {}: {}{}", test_name, status, duration);

        if !result.errors.is_empty() {
            for error in &result.errors {
                println!("      ⚠️  {}", error);
            }
        }
    }

    // Analyze results
    let successful_tests = suite_result.results.values().filter(|r| r.success).count();
    let failed_tests = suite_result.results.len() - successful_tests;

    println!("  🎯 Analysis:");
    println!("    • Successful Tests: {}", successful_tests);
    println!("    • Failed Tests: {}", failed_tests);

    if suite_result.success_rate >= 0.8 {
        println!("    • 🎉 Excellent async runtime resilience!");
    } else if suite_result.success_rate >= 0.6 {
        println!("    • 👍 Good async runtime resilience with room for improvement");
    } else {
        println!("    • ⚠️  Async runtime resilience needs attention");
    }

    Ok(())
}

async fn demo_real_world_integration(framework: &AsyncRuntimeChaosFramework) -> Result<()> {
    println!("Demonstrating real-world integration patterns...");

    // Simulate integration with streaming service
    println!("  🌊 Simulating streaming service chaos testing");
    demo_streaming_service_chaos().await?;

    // Simulate integration with batching service
    println!("  📦 Simulating batching service chaos testing");
    demo_batching_service_chaos().await?;

    // Simulate integration with memory pressure handler
    println!("  💾 Simulating memory pressure handler chaos testing");
    demo_memory_pressure_handler_chaos(framework).await?;

    Ok(())
}

async fn demo_streaming_service_chaos() -> Result<()> {
    println!("    Creating streaming connections under chaos conditions...");

    // Simulate chaos testing with streaming service
    let connections = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut handles = Vec::new();

    for i in 0..20 {
        let conn_count = connections.clone();
        handles.push(tokio::spawn(async move {
            // Simulate streaming connection
            conn_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

            // Simulate streaming with potential failures
            for chunk in 0..10 {
                tokio::time::sleep(Duration::from_millis(10)).await;

                // Simulate random failures
                if fastrand::f32() < 0.1 {
                    // 10% failure rate
                    return Err(format!("Stream {} failed at chunk {}", i, chunk));
                }
            }

            conn_count.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }));
    }

    // Wait for completion
    let mut successful_streams = 0;
    let mut failed_streams = 0;

    for handle in handles {
        match handle.await? {
            Ok(_) => successful_streams += 1,
            Err(_) => failed_streams += 1,
        }
    }

    println!(
        "    📊 Streaming chaos results: {} successful, {} failed",
        successful_streams, failed_streams
    );

    Ok(())
}

async fn demo_batching_service_chaos() -> Result<()> {
    println!("    Testing batching service under chaos conditions...");

    // Simulate batching with chaos
    let batch_queue = std::sync::Arc::new(tokio::sync::Mutex::new(Vec::<u32>::new()));
    let processed_batches = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

    // Producer tasks
    let queue_clone = batch_queue.clone();
    let producer_handle = tokio::spawn(async move {
        for i in 0..100 {
            let mut queue = queue_clone.lock().await;
            queue.push(i);

            // Simulate producer chaos (occasional delays)
            if fastrand::f32() < 0.2 {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }
    });

    // Consumer/processor tasks
    let mut consumer_handles = Vec::new();
    for worker_id in 0..3 {
        let queue_clone = batch_queue.clone();
        let processed_clone = processed_batches.clone();

        consumer_handles.push(tokio::spawn(async move {
            loop {
                let batch = {
                    let mut queue = queue_clone.lock().await;
                    if queue.is_empty() {
                        Vec::new()
                    } else {
                        let batch_size = std::cmp::min(10, queue.len());
                        queue.drain(0..batch_size).collect()
                    }
                };

                if batch.is_empty() {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    continue;
                }

                // Process batch with potential chaos
                if fastrand::f32() < 0.15 {
                    // 15% chance of processing failure
                    println!("    ⚠️  Worker {} failed to process batch", worker_id);
                } else {
                    processed_clone.fetch_add(batch.len(), std::sync::atomic::Ordering::SeqCst);
                }

                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }));
    }

    // Wait for producer to finish
    producer_handle.await?;

    // Let consumers process remaining items
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Stop consumers
    for handle in consumer_handles {
        handle.abort();
    }

    let final_processed = processed_batches.load(std::sync::atomic::Ordering::SeqCst);
    println!(
        "    📊 Batching chaos results: {} items processed",
        final_processed
    );

    Ok(())
}

async fn demo_memory_pressure_handler_chaos(framework: &AsyncRuntimeChaosFramework) -> Result<()> {
    println!("    Testing memory pressure handler integration...");

    // Run memory pressure test while simulating actual memory pressure handler
    let config = AsyncMemoryPressureConfig {
        memory_pressure_mb: 200,
        concurrent_async_tasks: 15,
        pressure_duration: Duration::from_secs(3),
    };

    // Simulate memory pressure handler running in background
    let cleanup_trigger = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let trigger_clone = cleanup_trigger.clone();

    let cleanup_handle = tokio::spawn(async move {
        while !trigger_clone.load(std::sync::atomic::Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(100)).await;

            // Simulate memory cleanup activities
            let _cleanup_data = vec![0u8; 1024 * 1024]; // Allocate 1MB
            tokio::time::sleep(Duration::from_millis(10)).await;
            // Data automatically deallocated when dropped
        }
    });

    // Run chaos test
    let result = framework.test_async_memory_pressure(config).await?;

    // Stop cleanup handler
    cleanup_trigger.store(true, std::sync::atomic::Ordering::SeqCst);
    cleanup_handle.await?;

    print_experiment_results("Memory Pressure Handler Integration", &result);

    Ok(())
}

fn print_experiment_results(
    test_name: &str,
    result: &trustformers_serve::async_runtime_chaos::AsyncExperimentResult,
) {
    let status = if result.success { "✅ SUCCESS" } else { "❌ FAILED" };
    println!("    📊 {}: {}", test_name, status);

    // Print key metrics
    for (metric, value) in &result.metrics {
        if metric.contains("duration") {
            println!("      • {}: {:.0}ms", metric, value);
        } else if metric.contains("rate") || metric.contains("percentage") {
            println!("      • {}: {:.1}%", metric, value);
        } else {
            println!("      • {}: {:.0}", metric, value);
        }
    }

    // Print errors if any
    if !result.errors.is_empty() {
        println!("      ⚠️  Errors:");
        for error in &result.errors {
            println!("        - {}", error);
        }
    }

    println!();
}
