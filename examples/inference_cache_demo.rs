//! Demonstration of the inference caching system
#![allow(unused_variables)]
//!
//! This example shows how to use the cache for different pipeline types
//! and demonstrates the performance improvements.

use std::time::Instant;
use trustformers::{
    pipeline::{pipeline, PipelineOptions, TextClassificationPipeline},
    AutoModel, AutoTokenizer,
};
use trustformers_core::cache::{InferenceCache, CacheConfig, InferenceCacheBuilder};

/// Simulate a model that takes time to process
struct MockModel;

impl MockModel {
    fn classify(&self, _text: &str) -> Vec<(String, f32)> {
        // Simulate processing time
        std::thread::sleep(std::time::Duration::from_millis(100));
        vec![
            ("POSITIVE".to_string(), 0.8),
            ("NEGATIVE".to_string(), 0.2),
        ]
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TrustformeRS Inference Cache Demo ===\n");

    // 1. Create cache with different configurations
    demo_cache_configurations()?;

    // 2. Demonstrate cache hit/miss performance
    demo_cache_performance()?;

    // 3. Show cache metrics
    demo_cache_metrics()?;

    // 4. Demonstrate eviction policies
    demo_eviction_policies()?;

    // 5. Show compression benefits
    demo_compression()?;

    Ok(())
}

fn demo_cache_configurations() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Cache Configuration Examples:");
    println!("================================\n");

    // Default configuration
    let default_cache = InferenceCache::new(CacheConfig::default());
    println!("Default cache created with:");
    println!("  - Max entries: 1000");
    println!("  - Max memory: 1GB");
    println!("  - TTL: 1 hour");
    println!("  - Compression: enabled\n");

    // Custom configuration using builder
    let custom_cache = InferenceCacheBuilder::new()
        .max_entries(500)
        .max_memory_mb(256)
        .ttl(std::time::Duration::from_secs(300))
        .enable_compression(true)
        .compression_threshold(512)
        .enable_metrics(true)
        .build();

    println!("Custom cache created with:");
    println!("  - Max entries: 500");
    println!("  - Max memory: 256MB");
    println!("  - TTL: 5 minutes");
    println!("  - Compression threshold: 512 bytes\n");

    Ok(())
}

fn demo_cache_performance() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Cache Performance Demonstration:");
    println!("===================================\n");

    // Create a mock classification pipeline with cache
    let cache_config = CacheConfig {
        max_entries: Some(100),
        max_memory_bytes: Some(10 * 1024 * 1024), // 10MB
        ttl: Some(std::time::Duration::from_secs(300)),
        enable_metrics: true,
        compress_values: false, // Disable for performance testing
        compression_threshold: 1024,
    };

    let cache = InferenceCache::new(cache_config);
    let model = MockModel;

    // Test inputs
    let test_texts = vec![
        "This movie is absolutely fantastic!",
        "I really enjoyed watching this film.",
        "The plot was terrible and boring.",
        "This movie is absolutely fantastic!", // Duplicate
        "I really enjoyed watching this film.", // Duplicate
    ];

    println!("Processing {} texts (with duplicates)...\n", test_texts.len());

    for (i, text) in test_texts.iter().enumerate() {
        let start = Instant::now();

        // Build cache key
        let cache_key = trustformers_core::cache::CacheKeyBuilder::new(
            "mock-model",
            "text-classification"
        )
        .with_text(text)
        .build();

        // Check cache first
        let result = if let Some(cached) = cache.get(&cache_key) {
            println!("Text {}: CACHE HIT", i + 1);
            // Deserialize from cache
            String::from_utf8(cached)?
        } else {
            println!("Text {}: CACHE MISS - Computing...", i + 1);
            // Compute result
            let classification = model.classify(text);
            let result = format!("{:?}", classification);

            // Store in cache
            cache.insert(cache_key, result.as_bytes().to_vec());
            result
        };

        let elapsed = start.elapsed();
        println!("  Time: {:?}", elapsed);
        println!("  Result: {}\n", result);
    }

    Ok(())
}

fn demo_cache_metrics() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Cache Metrics:");
    println!("=================\n");

    let cache = InferenceCacheBuilder::new()
        .max_entries(50)
        .enable_metrics(true)
        .build();

    // Simulate some cache operations
    for i in 0..20 {
        let key = trustformers_core::cache::CacheKeyBuilder::new("model", "task")
            .with_text(&format!("text-{}", i))
            .build();

        let value = format!("result-{}", i).into_bytes();
        cache.insert(key, value);
    }

    // Access some entries multiple times
    for i in 0..10 {
        let key = trustformers_core::cache::CacheKeyBuilder::new("model", "task")
            .with_text(&format!("text-{}", i))
            .build();

        let _ = cache.get(&key);
        let _ = cache.get(&key); // Second access
    }

    // Access non-existent entries
    for i in 20..25 {
        let key = trustformers_core::cache::CacheKeyBuilder::new("model", "task")
            .with_text(&format!("text-{}", i))
            .build();

        let _ = cache.get(&key);
    }

    // Get metrics
    if let Some(metrics) = cache.metrics() {
        let snapshot = metrics.snapshot();
        println!("{}", snapshot.format_report());
    }

    Ok(())
}

fn demo_eviction_policies() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Eviction Policy Demonstration:");
    println!("=================================\n");

    // LRU eviction
    println!("LRU Eviction (max 3 entries):");
    let lru_cache = InferenceCacheBuilder::new()
        .max_entries(3)
        .enable_metrics(true)
        .build();

    for i in 0..5 {
        let key = trustformers_core::cache::CacheKeyBuilder::new("model", "task")
            .with_text(&format!("lru-text-{}", i))
            .build();

        lru_cache.insert(key, format!("result-{}", i).into_bytes());
        println!("  Inserted entry {}. Cache size: {}", i, lru_cache.len());
    }

    println!("\n  Final cache contains:");
    for i in 0..5 {
        let key = trustformers_core::cache::CacheKeyBuilder::new("model", "task")
            .with_text(&format!("lru-text-{}", i))
            .build();

        if lru_cache.get(&key).is_some() {
            println!("    - Entry {} is present", i);
        }
    }

    // Size-based eviction
    println!("\nSize-based Eviction (max 1KB):");
    let size_cache = InferenceCacheBuilder::new()
        .max_memory_mb(0) // Will use 1KB minimum
        .enable_metrics(true)
        .build();

    // Note: The actual implementation would need adjustment for such small sizes
    println!("  (Size-based eviction demonstration omitted for brevity)\n");

    Ok(())
}

fn demo_compression() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Compression Benefits:");
    println!("========================\n");

    let cache_no_compress = InferenceCacheBuilder::new()
        .enable_compression(false)
        .enable_metrics(true)
        .build();

    let cache_compress = InferenceCacheBuilder::new()
        .enable_compression(true)
        .compression_threshold(100)
        .enable_metrics(true)
        .build();

    // Create a large, compressible result
    let large_result = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(50);
    let key = trustformers_core::cache::CacheKeyBuilder::new("model", "task")
        .with_text("compression-test")
        .build();

    // Insert into non-compressed cache
    cache_no_compress.insert(key.clone(), large_result.as_bytes().to_vec());

    // Insert into compressed cache
    cache_compress.insert(key, large_result.as_bytes().to_vec());

    println!("Original size: {} bytes", large_result.len());

    if let (Some(metrics_no_compress), Some(metrics_compress)) =
        (cache_no_compress.metrics(), cache_compress.metrics())
    {
        let snapshot_no_compress = metrics_no_compress.snapshot();
        let snapshot_compress = metrics_compress.snapshot();

        println!("Uncompressed cache memory: {} bytes", snapshot_no_compress.total_memory_bytes);
        println!("Compressed cache memory: {} bytes", snapshot_compress.total_memory_bytes);
        println!("Compression ratio: {:.2}x",
            snapshot_no_compress.total_memory_bytes as f64 / snapshot_compress.total_memory_bytes as f64);
    }

    Ok(())
}

// Pipeline integration example
#[cfg(feature = "pipeline-cache-integration")]
fn demo_pipeline_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("6. Pipeline Integration:");
    println!("========================\n");

    // Create pipeline with cache
    let options = PipelineOptions {
        cache_config: Some(CacheConfig {
            max_entries: Some(1000),
            enable_metrics: true,
            ..Default::default()
        }),
        ..Default::default()
    };

    let pipeline = pipeline("text-classification", Some("bert-base-uncased"), Some(options))?;

    // Process some texts
    let texts = vec![
        "I love this product!".to_string(),
        "This is terrible.".to_string(),
        "I love this product!".to_string(), // Duplicate - should hit cache
    ];

    for (i, text) in texts.iter().enumerate() {
        let start = Instant::now();
        let result = pipeline.__call__(text.clone())?;
        let elapsed = start.elapsed();

        println!("Text {}: {:?}", i + 1, elapsed);
        println!("Result: {:?}\n", result);
    }

    Ok(())
}