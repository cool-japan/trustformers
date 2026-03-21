//! Dynamic Batching Demo for TrustformeRS Inference Server
#![allow(unused_variables)]
//!
//! This example demonstrates how to use the dynamic batching system
//! to optimize inference throughput while maintaining low latency.

use anyhow::Result;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use trustformers_serve::{
    batching::{
        aggregator::RequestInput, config::Priority, BatchingConfig, BatchingMode,
        DynamicBatchConfig, OptimizationTarget, Request, RequestId,
    },
    DynamicBatchingService,
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("TrustformeRS Dynamic Batching Demo");
    println!("==================================\n");

    // Example 1: Basic dynamic batching
    example_basic_batching().await?;

    // Example 2: Priority-based scheduling
    example_priority_scheduling().await?;

    // Example 3: Adaptive batching
    example_adaptive_batching().await?;

    // Example 4: Memory-aware batching
    example_memory_aware_batching().await?;

    // Example 5: Continuous batching for LLMs
    example_continuous_batching().await?;

    // Example 6: Performance monitoring
    example_performance_monitoring().await?;

    Ok(())
}

async fn example_basic_batching() -> Result<()> {
    println!("Example 1: Basic Dynamic Batching");
    println!("---------------------------------");

    // Configure basic dynamic batching
    let config = BatchingConfig {
        max_batch_size: 16,
        min_batch_size: 4,
        max_wait_time: Duration::from_millis(50),
        mode: BatchingMode::Dynamic,
        optimization_target: OptimizationTarget::Throughput,
        ..Default::default()
    };

    let service = DynamicBatchingService::new(config);
    service.start().await?;

    println!("Submitting 20 requests...");

    // Submit multiple requests
    let mut handles = vec![];
    for i in 0..20 {
        let service_clone = service.clone();
        let handle = tokio::spawn(async move {
            let request = Request {
                id: RequestId::new(),
                input: RequestInput::Text {
                    text: format!("Request #{}", i),
                    max_length: Some(100),
                },
                priority: Priority::Normal,
                submitted_at: Instant::now(),
                deadline: None,
                metadata: Default::default(),
            };

            service_clone.submit_request(request).await
        });
        handles.push(handle);

        // Add small delay to simulate real-world scenario
        sleep(Duration::from_millis(5)).await;
    }

    // Wait for all requests
    for handle in handles {
        let result = handle.await??;
        println!("Received result for request: {:?}", result.request_id);
    }

    // Show statistics
    let stats = service.get_stats().await;
    println!("\nBatching Statistics:");
    println!(
        "  Average batch size: {:.1}",
        stats.aggregator_stats.avg_batch_size
    );
    println!(
        "  Total batches formed: {}",
        stats.aggregator_stats.total_batches_formed
    );
    println!(
        "  Average latency: {:.1}ms",
        stats.metrics_summary.avg_latency_ms
    );
    println!(
        "  Throughput: {:.1} req/s",
        stats.metrics_summary.throughput_rps
    );

    println!();
    Ok(())
}

async fn example_priority_scheduling() -> Result<()> {
    println!("Example 2: Priority-based Scheduling");
    println!("-----------------------------------");

    let config = BatchingConfig {
        max_batch_size: 8,
        max_wait_time: Duration::from_millis(100),
        enable_priority_scheduling: true,
        ..Default::default()
    };

    let service = DynamicBatchingService::new(config);
    service.start().await?;

    // Submit requests with different priorities
    let priorities = [
        Priority::Low,
        Priority::Critical,
        Priority::Normal,
        Priority::High,
        Priority::Normal,
        Priority::Critical,
    ];

    println!("Submitting requests with different priorities...");

    let mut handles = vec![];
    for (i, priority) in priorities.iter().enumerate() {
        let service_clone = service.clone();
        let priority = *priority;

        let handle = tokio::spawn(async move {
            let request = Request {
                id: RequestId::new(),
                input: RequestInput::Text {
                    text: format!("Priority {:?} request #{}", priority, i),
                    max_length: None,
                },
                priority,
                submitted_at: Instant::now(),
                deadline: Some(Instant::now() + Duration::from_secs(1)),
                metadata: Default::default(),
            };

            let start = Instant::now();
            let result = service_clone.submit_request(request).await?;
            let latency = start.elapsed();

            Ok::<_, anyhow::Error>((priority, latency, result))
        });

        handles.push(handle);
        sleep(Duration::from_millis(10)).await;
    }

    // Collect results
    for handle in handles {
        let (priority, latency, result) = handle.await??;
        println!(
            "Priority {:?}: latency={:.1}ms",
            priority,
            latency.as_millis()
        );
    }

    println!();
    Ok(())
}

async fn example_adaptive_batching() -> Result<()> {
    println!("Example 3: Adaptive Batching");
    println!("---------------------------");

    let config = BatchingConfig {
        max_batch_size: 32,
        enable_adaptive_batching: true,
        mode: BatchingMode::Adaptive,
        optimization_target: OptimizationTarget::Balanced,
        dynamic_config: DynamicBatchConfig {
            latency_slo_ms: Some(100),
            target_utilization: 0.8,
            ..Default::default()
        },
        ..Default::default()
    };

    let service = DynamicBatchingService::new(config);
    service.start().await?;

    println!("Simulating varying load patterns...");

    // Phase 1: Low load
    println!("\nPhase 1: Low load (5 req/s)");
    for i in 0..10 {
        submit_request(&service, i, Priority::Normal).await?;
        sleep(Duration::from_millis(200)).await;
    }

    let stats = service.get_stats().await;
    println!(
        "  Avg batch size: {:.1}",
        stats.aggregator_stats.avg_batch_size
    );
    println!(
        "  Avg latency: {:.1}ms",
        stats.metrics_summary.avg_latency_ms
    );

    // Phase 2: High load
    println!("\nPhase 2: High load (50 req/s)");
    let mut handles = vec![];
    for i in 10..60 {
        let service_clone = service.clone();
        handles.push(tokio::spawn(async move {
            submit_request(&service_clone, i, Priority::Normal).await
        }));
        sleep(Duration::from_millis(20)).await;
    }

    for handle in handles {
        handle.await??;
    }

    let stats = service.get_stats().await;
    println!(
        "  Avg batch size: {:.1}",
        stats.aggregator_stats.avg_batch_size
    );
    println!(
        "  Avg latency: {:.1}ms",
        stats.metrics_summary.avg_latency_ms
    );

    // Show optimization suggestions
    if !stats.metrics_summary.optimization_suggestions.is_empty() {
        println!("\nOptimization Suggestions:");
        for suggestion in &stats.metrics_summary.optimization_suggestions {
            println!("  - {}", suggestion);
        }
    }

    println!();
    Ok(())
}

async fn example_memory_aware_batching() -> Result<()> {
    println!("Example 4: Memory-aware Batching");
    println!("-------------------------------");

    let config = BatchingConfig {
        max_batch_size: 64,
        memory_limit: Some(1024 * 1024 * 100), // 100MB limit
        dynamic_config: DynamicBatchConfig {
            memory_aware: true,
            enable_bucketing: true,
            bucket_boundaries: vec![128, 256, 512, 1024],
            ..Default::default()
        },
        ..Default::default()
    };

    let service = DynamicBatchingService::new(config);
    service.start().await?;

    println!("Submitting requests with varying sizes...");

    // Submit requests with different sequence lengths
    let sequence_lengths = [50, 200, 100, 500, 150, 1000, 75, 250];

    for (i, seq_len) in sequence_lengths.iter().enumerate() {
        let request = Request {
            id: RequestId::new(),
            input: RequestInput::Text {
                text: "x".repeat(*seq_len),
                max_length: Some(*seq_len),
            },
            priority: Priority::Normal,
            submitted_at: Instant::now(),
            deadline: None,
            metadata: Default::default(),
        };

        println!("Request {} - sequence length: {}", i, seq_len);
        service.submit_request(request).await?;
    }

    let stats = service.get_stats().await;
    println!("\nMemory-aware batching results:");
    println!("  Queue depth: {}", stats.metrics_summary.queue_depth);

    println!();
    Ok(())
}

async fn example_continuous_batching() -> Result<()> {
    println!("Example 5: Continuous Batching for LLMs");
    println!("--------------------------------------");

    let config = BatchingConfig {
        max_batch_size: 16,
        mode: BatchingMode::Continuous,
        optimization_target: OptimizationTarget::Throughput,
        ..Default::default()
    };

    let service = DynamicBatchingService::new(config);
    service.start().await?;

    println!("Simulating text generation requests...");

    // Simulate generation requests with different lengths
    let prompts = [
        ("Short prompt", 50),
        ("Medium length prompt for testing", 150),
        (
            "This is a longer prompt that will generate more tokens",
            300,
        ),
        ("Quick", 20),
    ];

    let mut handles = vec![];
    for (i, (prompt, max_tokens)) in prompts.iter().enumerate() {
        let service_clone = service.clone();
        let prompt = prompt.to_string();
        let max_tokens = *max_tokens;

        let handle = tokio::spawn(async move {
            let request = Request {
                id: RequestId::new(),
                input: RequestInput::Text {
                    text: prompt.clone(),
                    max_length: Some(max_tokens),
                },
                priority: Priority::Normal,
                submitted_at: Instant::now(),
                deadline: None,
                metadata: Default::default(),
            };

            let start = Instant::now();
            let result = service_clone.submit_request(request).await?;
            let duration = start.elapsed();

            println!("Request {} completed in {:.1}ms", i, duration.as_millis());
            Ok::<_, anyhow::Error>(())
        });

        handles.push(handle);

        // Stagger requests
        sleep(Duration::from_millis(100)).await;
    }

    for handle in handles {
        handle.await??;
    }

    println!();
    Ok(())
}

async fn example_performance_monitoring() -> Result<()> {
    println!("Example 6: Performance Monitoring");
    println!("--------------------------------");

    let config = BatchingConfig {
        max_batch_size: 32,
        enable_adaptive_batching: true,
        optimization_target: OptimizationTarget::Balanced,
        ..Default::default()
    };

    let service = DynamicBatchingService::new(config);
    service.start().await?;

    println!("Running performance test...");

    // Generate load for 5 seconds
    let duration = Duration::from_secs(5);
    let start = Instant::now();
    let mut request_count = 0;

    while start.elapsed() < duration {
        submit_request(&service, request_count, Priority::Normal).await?;
        request_count += 1;

        // Variable load
        let delay = if request_count % 10 < 5 { 10 } else { 50 };
        sleep(Duration::from_millis(delay)).await;
    }

    // Get final statistics
    let stats = service.get_stats().await;

    println!("\nPerformance Summary:");
    println!("  Total requests: {}", request_count);
    println!("  Duration: {:.1}s", duration.as_secs_f64());
    println!(
        "  Throughput: {:.1} req/s",
        request_count as f64 / duration.as_secs_f64()
    );
    println!(
        "  Average batch size: {:.1}",
        stats.aggregator_stats.avg_batch_size
    );
    println!(
        "  Average latency: {:.1}ms",
        stats.metrics_summary.avg_latency_ms
    );
    println!("  Queue depth: {}", stats.metrics_summary.queue_depth);

    // Show optimization suggestions
    if !stats.metrics_summary.optimization_suggestions.is_empty() {
        println!("\nOptimization Suggestions:");
        for suggestion in &stats.metrics_summary.optimization_suggestions {
            println!("  - {}", suggestion);
        }
    }

    Ok(())
}

// Helper function to submit a request
async fn submit_request(
    service: &DynamicBatchingService,
    id: usize,
    priority: Priority,
) -> Result<()> {
    let request = Request {
        id: RequestId::new(),
        input: RequestInput::Text {
            text: format!("Request #{}", id),
            max_length: Some(100),
        },
        priority,
        submitted_at: Instant::now(),
        deadline: None,
        metadata: Default::default(),
    };

    service.submit_request(request).await?;
    Ok(())
}
