use anyhow::Result;
#![allow(unused_variables)]
use std::collections::HashMap;
use trustformers_core::leaderboard::*;
use trustformers_core::performance::{
    BenchmarkDSL, BenchmarkRunnerBuilder, RunMode,
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("TrustformeRS Leaderboard System Demo");
    println!("====================================\n");

    // Create leaderboard client
    let client = ClientBuilder::new()
        .local_dir("./leaderboard_data")
        .default_category(LeaderboardCategory::Inference)
        .default_limit(10)
        .build_with_file_storage()
        .await?;

    // Example 1: Submit benchmark results
    println!("=== Example 1: Submitting Benchmark Results ===");
    submit_benchmark_results(&client).await?;

    // Example 2: Query leaderboard
    println!("\n=== Example 2: Querying Leaderboard ===");
    query_leaderboard(&client).await?;

    // Example 3: Compare models
    println!("\n=== Example 3: Comparing Models ===");
    compare_models(&client).await?;

    // Example 4: View trends
    println!("\n=== Example 4: Performance Trends ===");
    view_trends(&client).await?;

    // Example 5: Get statistics
    println!("\n=== Example 5: Leaderboard Statistics ===");
    view_statistics(&client).await?;

    // Example 6: Integration with custom benchmarks
    println!("\n=== Example 6: Integration with Custom Benchmarks ===");
    benchmark_integration(&client).await?;

    Ok(())
}

async fn submit_benchmark_results(client: &LeaderboardClient) -> Result<()> {
    // Create sample submissions
    let submissions = vec![
        create_submission(
            "BERT-Base",
            "1.0",
            "text_classification",
            10.5,
            95.2,
            1024.0,
            0.92,
        ),
        create_submission(
            "BERT-Base",
            "1.1",
            "text_classification",
            8.3,
            120.5,
            980.0,
            0.93,
        ),
        create_submission(
            "DistilBERT",
            "1.0",
            "text_classification",
            5.2,
            192.3,
            512.0,
            0.90,
        ),
        create_submission(
            "RoBERTa",
            "1.0",
            "text_classification",
            11.8,
            84.7,
            1536.0,
            0.94,
        ),
    ];

    for submission in submissions {
        let id = client.submit(submission).await?;
        println!("Submitted entry with ID: {}", id);
    }

    Ok(())
}

async fn query_leaderboard(client: &LeaderboardClient) -> Result<()> {
    // Get top models by latency
    println!("Top models by latency:");
    let top_by_latency = client
        .get_top_models(
            LeaderboardCategory::Inference,
            RankingMetric::Latency,
            5,
        )
        .await?;

    for (i, entry) in top_by_latency.iter().enumerate() {
        println!(
            "{}. {} v{} - {:.1}ms latency, {:.1} throughput",
            i + 1,
            entry.model_name,
            entry.model_version,
            entry.metrics.latency_ms,
            entry.metrics.throughput.unwrap_or(0.0)
        );
    }

    // Get leaderboard for specific benchmark
    println!("\nLeaderboard for 'text_classification' benchmark:");
    let benchmark_results = client
        .get_benchmark_leaderboard("text_classification", Some(5))
        .await?;

    for entry in benchmark_results {
        println!(
            "- {} v{}: {:.1}ms, accuracy: {:.2}",
            entry.model_name,
            entry.model_version,
            entry.metrics.latency_ms,
            entry.metrics.accuracy.unwrap_or(0.0)
        );
    }

    // Search with filters
    println!("\nSearch results (accuracy > 0.91, latency < 10ms):");
    let search_results = client
        .search(SearchParams {
            min_accuracy: Some(0.91),
            max_latency_ms: Some(10.0),
            ..Default::default()
        })
        .await?;

    for entry in search_results {
        println!(
            "- {}: {:.1}ms, accuracy: {:.2}",
            entry.model_name,
            entry.metrics.latency_ms,
            entry.metrics.accuracy.unwrap_or(0.0)
        );
    }

    Ok(())
}

async fn compare_models(client: &LeaderboardClient) -> Result<()> {
    let comparison = client
        .compare_models("BERT-Base", "DistilBERT", Some(LeaderboardCategory::Inference))
        .await?;

    println!("Model Comparison: {} vs {}", comparison.model1.name, comparison.model2.name);
    println!("\nMetric Winners:");
    for (metric, winner) in &comparison.winner_by_metric {
        println!("  {}: {}", metric, winner);
    }

    println!("\nRelative Performance:");
    for (metric, change) in &comparison.relative_performance {
        println!("  {}: {:.1}%", metric, change);
    }

    Ok(())
}

async fn view_trends(client: &LeaderboardClient) -> Result<()> {
    // Get model history
    let history = client.get_model_history("BERT-Base").await?;

    println!("BERT-Base version history:");
    for entry in history {
        println!(
            "  v{} - {}: {:.1}ms latency, {:.2} accuracy",
            entry.model_version,
            entry.timestamp.format("%Y-%m-%d"),
            entry.metrics.latency_ms,
            entry.metrics.accuracy.unwrap_or(0.0)
        );
    }

    // Analyze trends
    let trend = client.get_trends("BERT-Base", "latency", 30).await?;

    println!("\nLatency trend for BERT-Base:");
    println!("  Trend: {:?}", trend.trend);
    println!("  Daily change: {:.2}%", trend.daily_change_percent);
    println!("  R-squared: {:.3}", trend.r_squared);

    Ok(())
}

async fn view_statistics(client: &LeaderboardClient) -> Result<()> {
    let stats = client.get_stats(Some(LeaderboardCategory::Inference)).await?;

    println!("Leaderboard Statistics (Inference):");
    println!("  Total entries: {}", stats.total_entries);
    println!("  Unique models: {}", stats.unique_models);
    println!("  Unique submitters: {}", stats.unique_submitters);

    println!("\nAverage Metrics:");
    println!("  Latency: {:.1}ms", stats.average_metrics.latency_ms);
    if let Some(throughput) = stats.average_metrics.throughput {
        println!("  Throughput: {:.1} items/sec", throughput);
    }
    if let Some(accuracy) = stats.average_metrics.accuracy {
        println!("  Accuracy: {:.3}", accuracy);
    }

    println!("\nBest Metrics:");
    println!(
        "  Lowest latency: {:.1}ms ({})",
        stats.best_metrics.lowest_latency.value,
        stats.best_metrics.lowest_latency.model_name
    );

    if let Some(highest_throughput) = &stats.best_metrics.highest_throughput {
        println!(
            "  Highest throughput: {:.1} items/sec ({})",
            highest_throughput.value,
            highest_throughput.model_name
        );
    }

    println!("\nTop Submitters:");
    for (name, count) in stats.top_submitters.iter().take(3) {
        println!("  {}: {} submissions", name, count);
    }

    Ok(())
}

async fn benchmark_integration(client: &LeaderboardClient) -> Result<()> {
    // Run a benchmark using the custom benchmark framework
    let benchmark = BenchmarkDSL::latency_benchmark("model_inference")
        .measure("forward_pass", || {
            let start = std::time::Instant::now();
            // Simulate model inference
            std::thread::sleep(std::time::Duration::from_millis(7));
            Ok(start.elapsed())
        })
        .build()?;

    let results = BenchmarkRunnerBuilder::new()
        .mode(RunMode::Quick)
        .add_benchmark(Box::new(benchmark))
        .run()?;

    // Submit the benchmark results to the leaderboard
    let report = &results[0];
    let submission_id = client
        .submit_from_report(
            report,
            "MyModel".to_string(),
            "2.0".to_string(),
            "Demo User".to_string(),
        )
        .await?;

    println!("Submitted benchmark results to leaderboard:");
    println!("  Entry ID: {}", submission_id);
    println!("  Model: MyModel v2.0");
    println!("  Average latency: {:.2}ms", report.summary.avg_latency_ms);

    // Verify submission
    if let Some(entry) = client.get_entry(submission_id).await? {
        println!("\nVerified submission:");
        println!("  Status: {}", if entry.validated { "Validated" } else { "Pending" });
        println!("  Category: {}", entry.category);
    }

    Ok(())
}

// Helper function to create submissions
fn create_submission(
    model_name: &str,
    version: &str,
    benchmark: &str,
    latency: f64,
    throughput: f64,
    memory: f64,
    accuracy: f64,
) -> LeaderboardSubmission {
    LeaderboardSubmission {
        model_name: model_name.to_string(),
        model_version: version.to_string(),
        benchmark_name: benchmark.to_string(),
        category: LeaderboardCategory::Inference,
        hardware: HardwareInfo {
            cpu: "Intel Xeon Gold 6248".to_string(),
            cpu_cores: 40,
            gpu: Some("NVIDIA A100 40GB".to_string()),
            gpu_count: Some(1),
            memory_gb: 256.0,
            accelerator: Some(AcceleratorType::CUDA),
            platform: "x86_64".to_string(),
        },
        software: SoftwareInfo {
            framework_version: "0.1.0".to_string(),
            rust_version: "1.75.0".to_string(),
            os: "Ubuntu 22.04".to_string(),
            optimization_level: OptimizationLevel::O3,
            precision: Precision::FP16,
            quantization: None,
            compiler_flags: vec!["-C target-cpu=native".to_string()],
        },
        metrics: PerformanceMetrics {
            latency_ms: latency,
            latency_percentiles: LatencyPercentiles {
                p50: latency * 0.9,
                p90: latency * 1.1,
                p95: latency * 1.2,
                p99: latency * 1.5,
                p999: latency * 2.0,
            },
            throughput: Some(throughput),
            tokens_per_second: Some(throughput * 512.0), // Assuming 512 tokens per item
            memory_mb: Some(memory),
            peak_memory_mb: Some(memory * 1.2),
            gpu_utilization: Some(85.0),
            accuracy: Some(accuracy),
            energy_watts: None,
            custom_metrics: HashMap::new(),
        },
        metadata: HashMap::new(),
        submitter: SubmitterInfo {
            name: "Demo User".to_string(),
            organization: Some("TrustformeRS".to_string()),
            email: None,
            github: Some("trustformers".to_string()),
        },
        tags: vec!["demo".to_string(), "transformer".to_string()],
        benchmark_report: None,
    }
}

// Advanced example: Custom ranking algorithm
struct AccuracyEfficiencyRanking;

impl RankingAlgorithm for AccuracyEfficiencyRanking {
    fn rank(
        &self,
        mut entries: Vec<LeaderboardEntry>,
        _criteria: &RankingCriteria,
    ) -> Result<Vec<LeaderboardEntry>> {
        // Sort by accuracy/latency ratio (accuracy per millisecond)
        entries.sort_by(|a, b| {
            let a_score = a.metrics.accuracy.unwrap_or(0.0) / a.metrics.latency_ms;
            let b_score = b.metrics.accuracy.unwrap_or(0.0) / b.metrics.latency_ms;
            b_score.partial_cmp(&a_score).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(entries)
    }
}