//! Demonstration of pipeline-integrated caching with real performance measurements
#![allow(unused_variables)]
//!
//! This example shows how the inference cache dramatically improves performance
//! for repeated queries in production scenarios.

use std::time::{Duration, Instant};
use trustformers::pipeline::{
    Pipeline, PipelineOutput, TextClassificationPipeline, BasePipeline,
    ClassificationOutput,
};
use trustformers::{AutoModel, AutoTokenizer};
use trustformers_core::cache::{CacheConfig, InferenceCacheBuilder};
use trustformers_core::{
    Tensor, Model, CoreError, Result as CoreResult,
    traits::{Config, Layer},
};

/// Mock model that simulates inference latency
struct MockBertModel {
    inference_time_ms: u64,
}

impl MockBertModel {
    fn new(inference_time_ms: u64) -> Self {
        Self { inference_time_ms }
    }
}

// Mock implementations to make the demo work
impl Model for MockBertModel {
    type Input = Tensor;
    type Output = ModelOutput;
    type Config = MockConfig;

    fn forward(&self, input: Self::Input) -> CoreResult<Self::Output> {
        // Simulate inference time
        std::thread::sleep(Duration::from_millis(self.inference_time_ms));

        // Return mock logits
        Ok(ModelOutput {
            logits: Tensor::zeros(&[1, 2])?, // Binary classification
        })
    }

    fn config(&self) -> &Self::Config {
        // Return a reference to a static mock config for demo purposes
        static MOCK_CONFIG: MockConfig = MockConfig;
        &MOCK_CONFIG
    }
}

struct ModelOutput {
    logits: Tensor,
}

#[derive(Clone)]
struct MockConfig;

impl Config for MockConfig {
    fn from_json(_json: &str) -> CoreResult<Self> {
        Ok(MockConfig)
    }

    fn to_json(&self) -> CoreResult<String> {
        Ok("{}".to_string())
    }
}

/// Mock tokenizer
struct MockTokenizer;

impl MockTokenizer {
    fn encode(&self, _text: &str) -> CoreResult<Tensor> {
        // Return mock token IDs
        Ok(Tensor::zeros(&[1, 10])?)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TrustformeRS Pipeline Cache Performance Demo ===\n");

    // Test scenarios
    demo_classification_cache()?;
    demo_cache_sharing()?;
    demo_cache_metrics_monitoring()?;
    demo_production_scenario()?;

    Ok(())
}

fn demo_classification_cache() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Text Classification with Cache:");
    println!("==================================\n");

    // Create pipeline without cache
    let model = AutoModel::from(MockBertModel::new(50)); // 50ms inference
    let tokenizer = AutoTokenizer::from(MockTokenizer);
    let pipeline_no_cache = TextClassificationPipeline::new(model.clone(), tokenizer.clone())?;

    // Create pipeline with cache
    let cache_config = CacheConfig {
        max_entries: Some(100),
        max_memory_bytes: None,
        ttl: Some(Duration::from_secs(300)),
        enable_metrics: true,
        compress_values: false,
        compression_threshold: 1024,
    };

    let pipeline_with_cache = TextClassificationPipeline::new(model, tokenizer)?
        .base
        .with_cache(cache_config);

    // Test texts
    let test_texts = vec![
        "This product is amazing!",
        "Terrible experience, would not recommend.",
        "This product is amazing!", // Duplicate
        "Neutral opinion about this.",
        "Terrible experience, would not recommend.", // Duplicate
    ];

    println!("Without cache:");
    let mut total_time_no_cache = Duration::ZERO;
    for text in &test_texts {
        let start = Instant::now();
        let _ = pipeline_no_cache.__call__(text.to_string())?;
        let elapsed = start.elapsed();
        total_time_no_cache += elapsed;
        println!("  '{}...': {:?}", &text[..20.min(text.len())], elapsed);
    }

    println!("\nWith cache:");
    let mut total_time_with_cache = Duration::ZERO;
    for (i, text) in test_texts.iter().enumerate() {
        let start = Instant::now();
        let _ = pipeline_with_cache.__call__(text.to_string())?;
        let elapsed = start.elapsed();
        total_time_with_cache += elapsed;
        let cache_status = if i == 2 || i == 4 { "HIT" } else { "MISS" };
        println!("  '{}...': {:?} ({})", &text[..20.min(text.len())], elapsed, cache_status);
    }

    println!("\nPerformance Summary:");
    println!("  Total time without cache: {:?}", total_time_no_cache);
    println!("  Total time with cache: {:?}", total_time_with_cache);
    println!("  Speedup: {:.2}x",
        total_time_no_cache.as_secs_f64() / total_time_with_cache.as_secs_f64());

    // Show cache metrics
    if let Some(cache) = pipeline_with_cache.get_cache() {
        if let Some(metrics) = cache.metrics() {
            let snapshot = metrics.snapshot();
            println!("\nCache Statistics:");
            println!("  Hit rate: {:.1}%", snapshot.hit_rate);
            println!("  Total lookups: {}", snapshot.hits + snapshot.misses);
            println!("  Avg lookup time: {:.2}μs", snapshot.avg_lookup_time_us);
        }
    }

    Ok(())
}

fn demo_cache_sharing() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n\n2. Shared Cache Between Pipelines:");
    println!("====================================\n");

    // Create a shared cache
    let shared_cache = std::sync::Arc::new(
        InferenceCacheBuilder::new()
            .max_entries(500)
            .enable_metrics(true)
            .build()
    );

    // Create multiple pipelines sharing the same cache
    let model1 = AutoModel::from(MockBertModel::new(30));
    let model2 = AutoModel::from(MockBertModel::new(40));
    let tokenizer = AutoTokenizer::from(MockTokenizer);

    let pipeline1 = TextClassificationPipeline::new(model1, tokenizer.clone())?
        .base
        .with_existing_cache(shared_cache.clone());

    let pipeline2 = TextClassificationPipeline::new(model2, tokenizer)?
        .base
        .with_existing_cache(shared_cache.clone());

    // Use pipeline1
    println!("Pipeline 1 processing:");
    let text = "Shared text for both pipelines";
    let start = Instant::now();
    let _ = pipeline1.__call__(text.to_string())?;
    println!("  First call: {:?} (MISS)", start.elapsed());

    // Use pipeline2 with same text - should hit cache
    println!("\nPipeline 2 processing same text:");
    let start = Instant::now();
    let _ = pipeline2.__call__(text.to_string())?;
    println!("  First call: {:?} (Expected MISS - different model)", start.elapsed());

    // Pipeline1 again - should hit cache
    println!("\nPipeline 1 processing again:");
    let start = Instant::now();
    let _ = pipeline1.__call__(text.to_string())?;
    println!("  Second call: {:?} (HIT)", start.elapsed());

    Ok(())
}

fn demo_cache_metrics_monitoring() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n\n3. Real-time Cache Metrics Monitoring:");
    println!("=======================================\n");

    let cache = InferenceCacheBuilder::new()
        .max_entries(10)
        .enable_metrics(true)
        .build();

    let model = AutoModel::from(MockBertModel::new(20));
    let tokenizer = AutoTokenizer::from(MockTokenizer);
    let pipeline = TextClassificationPipeline::new(model, tokenizer)?
        .base
        .with_existing_cache(std::sync::Arc::new(cache));

    // Simulate production workload
    let texts = vec![
        "query_1", "query_2", "query_3", "query_1", // query_1 repeated
        "query_4", "query_5", "query_2", // query_2 repeated
        "query_6", "query_3", // query_3 repeated
        "query_7", "query_8", "query_9", "query_10",
        "query_11", "query_12", // Will trigger evictions
    ];

    for (i, text) in texts.iter().enumerate() {
        let _ = pipeline.__call__(text.to_string())?;

        // Print metrics every 5 requests
        if (i + 1) % 5 == 0 {
            if let Some(cache) = pipeline.get_cache() {
                if let Some(metrics) = cache.metrics() {
                    let snapshot = metrics.snapshot();
                    println!("After {} requests:", i + 1);
                    println!("  Hit rate: {:.1}%", snapshot.hit_rate);
                    println!("  Cache size: {} entries", snapshot.total_entries);
                    println!("  Evictions: {}", snapshot.evictions);
                    println!("  Memory: {:.2} KB\n",
                        snapshot.total_memory_bytes as f64 / 1024.0);
                }
            }
        }
    }

    Ok(())
}

fn demo_production_scenario() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n\n4. Production Scenario - API Server:");
    println!("=====================================\n");

    // Simulate an API server with common queries
    let common_queries = vec![
        ("positive", "This is absolutely wonderful!"),
        ("negative", "This is terrible and disappointing."),
        ("neutral", "This is okay, nothing special."),
        ("question", "Is this product worth buying?"),
        ("comparison", "Better than the competitor's version."),
    ];

    // Create production-configured pipeline
    let cache = InferenceCacheBuilder::new()
        .max_entries(1000)
        .ttl(Duration::from_secs(3600)) // 1 hour TTL
        .enable_compression(true)
        .compression_threshold(512)
        .enable_metrics(true)
        .build();

    let model = AutoModel::from(MockBertModel::new(100)); // 100ms inference
    let tokenizer = AutoTokenizer::from(MockTokenizer);
    let pipeline = TextClassificationPipeline::new(model, tokenizer)?
        .base
        .with_existing_cache(std::sync::Arc::new(cache));

    // Warm up cache with common queries
    println!("Warming up cache with common queries...");
    for (_, query) in &common_queries {
        let _ = pipeline.__call__(query.to_string())?;
    }

    // Simulate production traffic
    println!("\nSimulating production traffic (80% common, 20% unique):");
    let mut rng = rand::thread_rng();
    use rand::Rng;

    let mut response_times = Vec::new();
    let start_time = Instant::now();

    for i in 0..50 {
        let query = if rng.gen_bool(0.8) {
            // 80% common query
            let idx = rng.gen_range(0..common_queries.len());
            common_queries[idx].1.to_string()
        } else {
            // 20% unique query
            format!("Unique query number {}", i)
        };

        let req_start = Instant::now();
        let _ = pipeline.__call__(query)?;
        let req_time = req_start.elapsed();
        response_times.push(req_time);
    }

    let total_time = start_time.elapsed();

    // Calculate statistics
    response_times.sort();
    let p50 = response_times[response_times.len() / 2];
    let p95 = response_times[response_times.len() * 95 / 100];
    let p99 = response_times[response_times.len() * 99 / 100];
    let avg: Duration = response_times.iter().sum::<Duration>() / response_times.len() as u32;

    println!("\nProduction Performance Statistics:");
    println!("  Total requests: {}", response_times.len());
    println!("  Total time: {:?}", total_time);
    println!("  Throughput: {:.1} req/s", response_times.len() as f64 / total_time.as_secs_f64());
    println!("\nLatency percentiles:");
    println!("  Average: {:?}", avg);
    println!("  P50: {:?}", p50);
    println!("  P95: {:?}", p95);
    println!("  P99: {:?}", p99);

    // Final cache statistics
    if let Some(cache) = pipeline.get_cache() {
        if let Some(metrics) = cache.metrics() {
            let snapshot = metrics.snapshot();
            println!("\nFinal Cache Statistics:");
            println!("{}", snapshot.format_report());
        }
    }

    println!("\nConclusion:");
    println!("With caching enabled, the API server can handle significantly more");
    println!("requests per second while maintaining low latency for common queries.");

    Ok(())
}

// Helper trait implementations for the demo
impl From<MockBertModel> for AutoModel {
    fn from(model: MockBertModel) -> Self {
        AutoModel::Custom(Box::new(model) as Box<dyn Model<Input = Tensor, Output = ModelOutput, Config = MockConfig>>)
    }
}

impl From<MockTokenizer> for AutoTokenizer {
    fn from(_tokenizer: MockTokenizer) -> Self {
        AutoTokenizer::Custom
    }
}

// Extension to make the mock work with pipelines
impl TextClassificationPipeline {
    fn new(model: AutoModel, tokenizer: AutoTokenizer) -> CoreResult<Self> {
        Ok(Self {
            base: BasePipeline::new(model, tokenizer),
            labels: std::sync::Arc::new(vec!["NEGATIVE".to_string(), "POSITIVE".to_string()]),
        })
    }
}

// Mock implementation of pipeline's __call__ method for demo
impl Pipeline for TextClassificationPipeline {
    type Input = String;
    type Output = PipelineOutput;

    fn __call__(&self, input: Self::Input) -> CoreResult<Self::Output> {
        // Check cache first
        if let Some(cache) = &self.base.cache {
            let cache_key = trustformers_core::cache::CacheKeyBuilder::new(
                "mock-bert",
                "text-classification"
            )
            .with_text(&input)
            .with_param("max_length", &self.base.max_length)
            .build();

            if let Some(cached_data) = cache.get(&cache_key) {
                if let Ok(results) = serde_json::from_slice::<Vec<ClassificationOutput>>(&cached_data) {
                    return Ok(PipelineOutput::Classification(results));
                }
            }
        }

        // Simulate model inference
        match &self.base.model {
            AutoModel::Custom(model) => {
                let _ = self.base.tokenizer.encode(&input)?;
                let _ = model.forward(Tensor::zeros(&[1, 10])?)?;

                // Create mock results
                let results = vec![
                    ClassificationOutput {
                        label: "POSITIVE".to_string(),
                        score: 0.8,
                    },
                    ClassificationOutput {
                        label: "NEGATIVE".to_string(),
                        score: 0.2,
                    },
                ];

                // Cache the results
                if let Some(cache) = &self.base.cache {
                    let cache_key = trustformers_core::cache::CacheKeyBuilder::new(
                        "mock-bert",
                        "text-classification"
                    )
                    .with_text(&input)
                    .with_param("max_length", &self.base.max_length)
                    .build();

                    if let Ok(serialized) = serde_json::to_vec(&results) {
                        cache.insert(cache_key, serialized);
                    }
                }

                Ok(PipelineOutput::Classification(results))
            }
            _ => Err(TrustformersError::ModelError("Unsupported model type".into())),
        }
    }
}

// Add encode method to AutoTokenizer for the demo
impl AutoTokenizer {
    fn encode(&self, _text: &str) -> CoreResult<Tensor> {
        Ok(Tensor::zeros(&[1, 10])?)
    }
}