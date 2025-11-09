//! Benchmark utilities for tokenization performance measurement
//!
//! This module provides simple utilities for measuring tokenization performance,
//! allowing users to benchmark their tokenizers with their own data.

use std::time::Instant;
use trustformers_core::traits::Tokenizer;

/// Simple benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    pub measurement_iterations: usize,
    /// Whether to include detailed statistics
    pub detailed_stats: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            measurement_iterations: 100,
            detailed_stats: true,
        }
    }
}

/// Benchmark results for tokenization performance
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Average tokens per second
    pub tokens_per_second: f64,
    /// Average characters per second
    pub characters_per_second: f64,
    /// Average latency per text (microseconds)
    pub average_latency_us: f64,
    /// Minimum latency (microseconds)
    pub min_latency_us: f64,
    /// Maximum latency (microseconds)
    pub max_latency_us: f64,
    /// Total texts processed
    pub total_texts: usize,
    /// Total tokens produced
    pub total_tokens: usize,
    /// Total characters processed
    pub total_characters: usize,
}

impl BenchmarkResult {
    /// Create a simple summary string
    pub fn summary(&self) -> String {
        format!(
            "Performance: {:.0} tokens/sec, {:.0} chars/sec, {:.2}μs avg latency",
            self.tokens_per_second, self.characters_per_second, self.average_latency_us
        )
    }

    /// Create a detailed report
    pub fn detailed_report(&self) -> String {
        format!(
            r#"Tokenization Benchmark Results
==============================
Throughput:
  - Tokens per second: {:.2}
  - Characters per second: {:.2}
  - Texts per second: {:.2}

Latency (per text):
  - Average: {:.2} μs
  - Minimum: {:.2} μs
  - Maximum: {:.2} μs

Volume:
  - Total texts: {}
  - Total tokens: {}
  - Total characters: {}
  - Average tokens per text: {:.1}
  - Average characters per text: {:.1}"#,
            self.tokens_per_second,
            self.characters_per_second,
            self.total_texts as f64 / (self.average_latency_us / 1_000_000.0),
            self.average_latency_us,
            self.min_latency_us,
            self.max_latency_us,
            self.total_texts,
            self.total_tokens,
            self.total_characters,
            self.total_tokens as f64 / self.total_texts as f64,
            self.total_characters as f64 / self.total_texts as f64
        )
    }
}

/// Simple benchmark runner for tokenizers
pub struct TokenizerBenchmark;

impl TokenizerBenchmark {
    /// Benchmark a tokenizer with the given texts
    pub fn benchmark<T: Tokenizer>(
        tokenizer: &T,
        texts: &[String],
        config: BenchmarkConfig,
    ) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        if texts.is_empty() {
            return Err("No texts provided for benchmarking".into());
        }

        // Warmup phase
        for _ in 0..config.warmup_iterations {
            for text in texts.iter().take(std::cmp::min(texts.len(), 10)) {
                let _ = tokenizer.encode(text)?;
            }
        }

        // Measurement phase
        let mut latencies = Vec::new();
        let mut total_tokens = 0;
        let mut total_characters = 0;

        for _ in 0..config.measurement_iterations {
            for text in texts {
                let start = Instant::now();
                let result = tokenizer.encode(text)?;
                let elapsed = start.elapsed();

                latencies.push(elapsed.as_micros() as f64);
                total_tokens += result.input_ids.len();
                total_characters += text.len();
            }
        }

        // Calculate statistics
        let total_time_seconds = latencies.iter().sum::<f64>() / 1_000_000.0;
        let average_latency_us = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let min_latency_us = latencies.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_latency_us = latencies.iter().cloned().fold(0.0, f64::max);

        let tokens_per_second = total_tokens as f64 / total_time_seconds;
        let characters_per_second = total_characters as f64 / total_time_seconds;
        let total_texts = texts.len() * config.measurement_iterations;

        Ok(BenchmarkResult {
            tokens_per_second,
            characters_per_second,
            average_latency_us,
            min_latency_us,
            max_latency_us,
            total_texts,
            total_tokens,
            total_characters,
        })
    }

    /// Quick benchmark with a single text repeated multiple times
    pub fn quick_benchmark<T: Tokenizer>(
        tokenizer: &T,
        text: &str,
        iterations: usize,
    ) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        let texts = vec![text.to_string(); 1];
        let config = BenchmarkConfig {
            warmup_iterations: 5,
            measurement_iterations: iterations,
            detailed_stats: false,
        };

        Self::benchmark(tokenizer, &texts, config)
    }

    /// Benchmark with sample texts of different lengths
    pub fn multi_length_benchmark<T: Tokenizer>(
        tokenizer: &T,
    ) -> Result<Vec<(String, BenchmarkResult)>, Box<dyn std::error::Error>> {
        let test_cases = vec![
            ("Short text", "Hello world!".to_string()),
            ("Medium text", "This is a longer text that contains multiple sentences and should give us a better idea of tokenization performance on medium-length inputs.".to_string()),
            ("Long text", "This is a much longer text that contains many sentences and words. It is designed to test the performance of tokenization on longer inputs that might be more representative of real-world usage scenarios. The text includes various punctuation marks, numbers like 123 and 456, and different types of content that a tokenizer might encounter in practice. This should help identify any performance differences between short and long text processing.".to_string()),
        ];

        let mut results = Vec::new();
        for (name, text) in test_cases {
            let result = Self::quick_benchmark(tokenizer, &text, 100)?;
            results.push((name.to_string(), result));
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::char::CharTokenizer;
    use std::collections::HashMap;

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.warmup_iterations, 10);
        assert_eq!(config.measurement_iterations, 100);
        assert!(config.detailed_stats);
    }

    #[test]
    fn test_quick_benchmark() {
        let tokenizer = CharTokenizer::new(HashMap::new());
        let result = TokenizerBenchmark::quick_benchmark(&tokenizer, "Hello world!", 10);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.tokens_per_second > 0.0);
        assert!(result.characters_per_second > 0.0);
        assert!(result.total_texts > 0);
    }

    #[test]
    fn test_benchmark_with_multiple_texts() {
        let tokenizer = CharTokenizer::new(HashMap::new());
        let texts = vec![
            "Hello world!".to_string(),
            "This is a test.".to_string(),
            "Another test text.".to_string(),
        ];

        let config = BenchmarkConfig {
            warmup_iterations: 2,
            measurement_iterations: 5,
            detailed_stats: true,
        };

        let result = TokenizerBenchmark::benchmark(&tokenizer, &texts, config);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.tokens_per_second > 0.0);
        assert_eq!(result.total_texts, 15); // 3 texts * 5 iterations
    }

    #[test]
    fn test_multi_length_benchmark() {
        let tokenizer = CharTokenizer::new(HashMap::new());
        let results = TokenizerBenchmark::multi_length_benchmark(&tokenizer);
        assert!(results.is_ok());

        let results = results.unwrap();
        assert_eq!(results.len(), 3); // Short, medium, long

        for (name, result) in results {
            assert!(!name.is_empty());
            assert!(result.tokens_per_second > 0.0);
        }
    }

    #[test]
    fn test_benchmark_result_summary() {
        let result = BenchmarkResult {
            tokens_per_second: 1000.0,
            characters_per_second: 5000.0,
            average_latency_us: 50.0,
            min_latency_us: 30.0,
            max_latency_us: 80.0,
            total_texts: 100,
            total_tokens: 500,
            total_characters: 2500,
        };

        let summary = result.summary();
        assert!(summary.contains("1000"));
        assert!(summary.contains("5000"));
        assert!(summary.contains("50.00"));
    }

    #[test]
    fn test_benchmark_empty_texts() {
        let tokenizer = CharTokenizer::new(HashMap::new());
        let texts = vec![];
        let config = BenchmarkConfig::default();

        let result = TokenizerBenchmark::benchmark(&tokenizer, &texts, config);
        assert!(result.is_err());
    }
}
