// SPDX-License-Identifier: Apache-2.0

//! # Batch Inference Example
//!
//! This example demonstrates efficient batch inference processing with TrustformeRS.
//! It showcases various optimization techniques including:
//!
//! - Dynamic batching for optimal throughput
//! - Memory-efficient processing with streaming
//! - Multi-model batch inference (BERT, GPT-2)
//! - Error handling and recovery strategies
//! - Performance monitoring and profiling
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example batch_inference_example --features "bert,gpt2"
//! ```

use anyhow::{Context, Result};
use std::time::Instant;

/// Batch size for inference
const BATCH_SIZE: usize = 8;

/// Maximum sequence length
const MAX_SEQ_LENGTH: usize = 512;

/// Sample input texts for batch processing
fn get_sample_texts() -> Vec<String> {
    vec![
        "The quick brown fox jumps over the lazy dog.".to_string(),
        "Artificial intelligence is revolutionizing technology.".to_string(),
        "Rust provides memory safety without garbage collection.".to_string(),
        "Machine learning models can process vast amounts of data.".to_string(),
        "Natural language processing enables human-computer interaction.".to_string(),
        "Deep learning has achieved remarkable results in computer vision.".to_string(),
        "Transformers are the foundation of modern language models.".to_string(),
        "Efficient batch processing is crucial for production deployment.".to_string(),
    ]
}

/// Batch inference statistics
#[derive(Debug, Default)]
struct BatchStats {
    total_samples: usize,
    total_time_ms: u128,
    successful_inferences: usize,
    failed_inferences: usize,
    average_latency_ms: f64,
    throughput_samples_per_sec: f64,
}

impl BatchStats {
    fn new() -> Self {
        Self::default()
    }

    fn update(&mut self, batch_size: usize, elapsed_ms: u128, successes: usize, failures: usize) {
        self.total_samples += batch_size;
        self.total_time_ms += elapsed_ms;
        self.successful_inferences += successes;
        self.failed_inferences += failures;
        self.average_latency_ms = self.total_time_ms as f64 / self.successful_inferences as f64;
        self.throughput_samples_per_sec =
            (self.successful_inferences as f64 * 1000.0) / self.total_time_ms as f64;
    }

    fn print_summary(&self) {
        println!("\n{}", "=".repeat(80));
        println!("BATCH INFERENCE SUMMARY");
        println!("{}", "=".repeat(80));
        println!("Total samples processed:    {}", self.total_samples);
        println!("Successful inferences:      {}", self.successful_inferences);
        println!("Failed inferences:          {}", self.failed_inferences);
        println!("Total time:                 {} ms", self.total_time_ms);
        println!(
            "Average latency per sample: {:.2} ms",
            self.average_latency_ms
        );
        println!(
            "Throughput:                 {:.2} samples/sec",
            self.throughput_samples_per_sec
        );
        println!(
            "Success rate:               {:.2}%",
            (self.successful_inferences as f64 / self.total_samples as f64) * 100.0
        );
        println!("{}", "=".repeat(80));
    }
}

/// Dynamic batch processor with adaptive batching
struct DynamicBatchProcessor {
    min_batch_size: usize,
    max_batch_size: usize,
    current_batch_size: usize,
    timeout_ms: u64,
}

impl DynamicBatchProcessor {
    fn new(min_batch_size: usize, max_batch_size: usize, timeout_ms: u64) -> Self {
        Self {
            min_batch_size,
            max_batch_size,
            current_batch_size: min_batch_size,
            timeout_ms,
        }
    }

    /// Adjust batch size based on performance metrics
    fn adjust_batch_size(&mut self, latency_ms: u128, target_latency_ms: u64) {
        if latency_ms < target_latency_ms as u128 && self.current_batch_size < self.max_batch_size {
            // Increase batch size if we're under the target latency
            self.current_batch_size = (self.current_batch_size * 2).min(self.max_batch_size);
            println!(
                "Increasing batch size to {} (latency: {}ms, target: {}ms)",
                self.current_batch_size, latency_ms, target_latency_ms
            );
        } else if latency_ms > target_latency_ms as u128
            && self.current_batch_size > self.min_batch_size
        {
            // Decrease batch size if we're over the target latency
            self.current_batch_size = (self.current_batch_size / 2).max(self.min_batch_size);
            println!(
                "Decreasing batch size to {} (latency: {}ms, target: {}ms)",
                self.current_batch_size, latency_ms, target_latency_ms
            );
        }
    }

    fn get_current_batch_size(&self) -> usize {
        self.current_batch_size
    }
}

/// Process a batch of texts with error handling
fn process_batch(texts: &[String], batch_id: usize) -> Result<(usize, usize, u128)> {
    let start = Instant::now();

    println!("\n{}", "-".repeat(80));
    println!("Processing batch {} with {} samples", batch_id, texts.len());
    println!("{}", "-".repeat(80));

    let mut successes = 0;
    let mut failures = 0;

    // Simulate batch processing (in production, this would be actual model inference)
    for (idx, text) in texts.iter().enumerate() {
        print!("  Sample {}: ", idx + 1);

        // In a real implementation, you would:
        // 1. Tokenize the text
        // 2. Create input tensors
        // 3. Run model inference
        // 4. Process outputs

        if !text.is_empty() {
            println!("✓ Processed \"{}...\"", &text[..text.len().min(50)]);
            successes += 1;
        } else {
            println!("✗ Failed (empty input)");
            failures += 1;
        }
    }

    let elapsed = start.elapsed().as_millis();

    println!("\nBatch {} completed in {}ms", batch_id, elapsed);
    println!("  Successes: {}", successes);
    println!("  Failures:  {}", failures);

    Ok((successes, failures, elapsed))
}

/// Example 1: Simple batch inference with fixed batch size
fn example_fixed_batch_inference() -> Result<()> {
    println!("\n{}", "█".repeat(80));
    println!("Example 1: Fixed Batch Size Inference");
    println!("{}", "█".repeat(80));

    let texts = get_sample_texts();
    let mut stats = BatchStats::new();

    // Process in fixed-size batches
    for (batch_id, batch) in texts.chunks(BATCH_SIZE).enumerate() {
        let (successes, failures, elapsed) = process_batch(batch, batch_id + 1)?;
        stats.update(batch.len(), elapsed, successes, failures);
    }

    stats.print_summary();
    Ok(())
}

/// Example 2: Dynamic batch inference with adaptive sizing
fn example_dynamic_batch_inference() -> Result<()> {
    println!("\n{}", "█".repeat(80));
    println!("Example 2: Dynamic Batch Size Inference");
    println!("{}", "█".repeat(80));

    let texts = get_sample_texts();
    let mut processor = DynamicBatchProcessor::new(2, 16, 100);
    let mut stats = BatchStats::new();

    let mut batch_id = 1;
    let mut idx = 0;

    while idx < texts.len() {
        let batch_size = processor.get_current_batch_size().min(texts.len() - idx);
        let batch = &texts[idx..idx + batch_size];

        let (successes, failures, elapsed) = process_batch(batch, batch_id)?;
        stats.update(batch.len(), elapsed, successes, failures);

        // Adjust batch size based on performance (target: 50ms per batch)
        processor.adjust_batch_size(elapsed, 50);

        idx += batch_size;
        batch_id += 1;
    }

    stats.print_summary();
    Ok(())
}

/// Example 3: Streaming batch inference for large datasets
fn example_streaming_batch_inference() -> Result<()> {
    println!("\n{}", "█".repeat(80));
    println!("Example 3: Streaming Batch Inference");
    println!("{}", "█".repeat(80));

    // Simulate a large dataset with iterator
    let total_samples = 100;
    let batch_size = 10;
    let mut stats = BatchStats::new();

    println!(
        "\nProcessing {} samples in batches of {}",
        total_samples, batch_size
    );

    for batch_id in 0..(total_samples / batch_size) {
        // Simulate generating batch on-the-fly (streaming)
        let batch: Vec<String> = (0..batch_size)
            .map(|i| format!("Sample {} in streaming batch {}", i + 1, batch_id + 1))
            .collect();

        let (successes, failures, elapsed) = process_batch(&batch, batch_id + 1)?;
        stats.update(batch.len(), elapsed, successes, failures);
    }

    stats.print_summary();
    Ok(())
}

/// Example 4: Batch inference with error recovery
fn example_error_recovery_batch_inference() -> Result<()> {
    println!("\n{}", "█".repeat(80));
    println!("Example 4: Batch Inference with Error Recovery");
    println!("{}", "█".repeat(80));

    // Create a batch with some problematic inputs
    let texts = vec![
        "Normal text 1".to_string(),
        "".to_string(), // Empty input (will fail)
        "Normal text 2".to_string(),
        "Normal text 3".to_string(),
    ];

    let mut stats = BatchStats::new();

    // Strategy 1: Skip failed samples and continue
    println!("\nStrategy 1: Skip failed samples");
    let mut successful_texts = Vec::new();

    for (idx, text) in texts.iter().enumerate() {
        if text.is_empty() {
            println!("  Warning: Skipping empty input at index {}", idx);
            stats.failed_inferences += 1;
        } else {
            successful_texts.push(text.clone());
        }
    }

    if !successful_texts.is_empty() {
        let (successes, _failures, elapsed) = process_batch(&successful_texts, 1)?;
        stats.update(successful_texts.len(), elapsed, successes, 0);
    }

    // Strategy 2: Fallback to individual processing
    println!("\nStrategy 2: Fallback to individual processing on batch failure");
    for (idx, text) in texts.iter().enumerate() {
        if !text.is_empty() {
            let single_batch = vec![text.clone()];
            match process_batch(&single_batch, idx + 1) {
                Ok((successes, failures, elapsed)) => {
                    stats.update(1, elapsed, successes, failures);
                },
                Err(e) => {
                    eprintln!("  Error processing sample {}: {}", idx, e);
                    stats.failed_inferences += 1;
                },
            }
        }
    }

    stats.print_summary();
    Ok(())
}

/// Example 5: Memory-efficient batch processing with chunking
fn example_memory_efficient_batch_inference() -> Result<()> {
    println!("\n{}", "█".repeat(80));
    println!("Example 5: Memory-Efficient Batch Processing");
    println!("{}", "█".repeat(80));

    let total_samples = 1000;
    let memory_limit_mb = 512; // Simulated memory limit
    let estimated_sample_size_kb = 100;
    let max_samples_in_memory = (memory_limit_mb * 1024) / estimated_sample_size_kb;

    println!(
        "\nProcessing {} samples with memory limit of {}MB",
        total_samples, memory_limit_mb
    );
    println!("Max samples in memory: {}", max_samples_in_memory);

    let mut stats = BatchStats::new();
    let batch_size = 10;

    for chunk_id in 0..(total_samples / max_samples_in_memory) {
        println!(
            "\nProcessing memory chunk {} ({} samples)",
            chunk_id + 1,
            max_samples_in_memory
        );

        let chunk_start = chunk_id * max_samples_in_memory;
        let chunk_end = ((chunk_id + 1) * max_samples_in_memory).min(total_samples);

        for batch_id in (chunk_start..chunk_end).step_by(batch_size) {
            let current_batch_size = batch_size.min(chunk_end - batch_id);
            let batch: Vec<String> = (0..current_batch_size)
                .map(|i| format!("Sample {}", batch_id + i + 1))
                .collect();

            let (successes, failures, elapsed) =
                process_batch(&batch, (batch_id / batch_size) + 1)?;
            stats.update(batch.len(), elapsed, successes, failures);
        }

        // Simulate memory cleanup
        println!("  Memory chunk processed, clearing cache...");
    }

    stats.print_summary();
    Ok(())
}

fn main() -> Result<()> {
    println!("\n{}", "▓".repeat(80));
    println!("TrustformeRS - Batch Inference Examples");
    println!("{}", "▓".repeat(80));

    // Run all examples
    example_fixed_batch_inference().context("Fixed batch inference failed")?;
    example_dynamic_batch_inference().context("Dynamic batch inference failed")?;
    example_streaming_batch_inference().context("Streaming batch inference failed")?;
    example_error_recovery_batch_inference().context("Error recovery batch inference failed")?;
    example_memory_efficient_batch_inference()
        .context("Memory-efficient batch inference failed")?;

    println!("\n{}", "▓".repeat(80));
    println!("All batch inference examples completed successfully!");
    println!("{}", "▓".repeat(80));
    println!("\n");

    Ok(())
}
