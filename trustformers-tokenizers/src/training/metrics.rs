//! Training metrics and incremental training functionality.
//!
//! This module provides comprehensive metrics tracking for tokenizer training,
//! including training progress, performance statistics, and incremental
//! training capabilities for updating existing tokenizers.

use crate::bpe::BPETokenizer;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Write};
use std::path::Path;
use std::time::Instant;
use trustformers_core::errors::Result;

use super::config::AdvancedTrainingConfig;
use super::corpus::CorpusProcessor;

/// Training metrics and statistics for tracking training progress.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainingMetrics {
    /// Final vocabulary size
    pub vocab_size: usize,
    /// Total number of tokens processed
    pub total_tokens: usize,
    /// Number of unique tokens encountered
    pub unique_tokens: usize,
    /// Coverage percentage (1.0 - OOV rate)
    pub coverage: f64,
    /// Compression ratio (tokens per character)
    pub compression_ratio: f64,
    /// Out-of-vocabulary rate
    pub oov_rate: f64,
    /// Total training time in seconds
    pub training_time: f64,
    /// Number of training iterations
    pub iterations: usize,
}

impl TrainingMetrics {
    /// Create a new empty metrics instance.
    pub fn new() -> Self {
        Self {
            vocab_size: 0,
            total_tokens: 0,
            unique_tokens: 0,
            coverage: 0.0,
            compression_ratio: 0.0,
            oov_rate: 0.0,
            training_time: 0.0,
            iterations: 0,
        }
    }

    /// Update metrics with new values.
    pub fn update(
        &mut self,
        vocab_size: usize,
        total_tokens: usize,
        unique_tokens: usize,
        coverage: f64,
        compression_ratio: f64,
        oov_rate: f64,
        training_time: f64,
        iterations: usize,
    ) {
        self.vocab_size = vocab_size;
        self.total_tokens = total_tokens;
        self.unique_tokens = unique_tokens;
        self.coverage = coverage;
        self.compression_ratio = compression_ratio;
        self.oov_rate = oov_rate;
        self.training_time = training_time;
        self.iterations = iterations;
    }

    /// Calculate efficiency score based on coverage and compression.
    pub fn efficiency_score(&self) -> f64 {
        // Weighted combination of coverage and compression ratio
        0.7 * self.coverage + 0.3 * (1.0 - self.compression_ratio).max(0.0)
    }

    /// Generate a summary report of the metrics.
    pub fn summary(&self) -> String {
        format!(
            "Training Metrics Summary:\n\
             - Vocabulary Size: {}\n\
             - Total Tokens: {}\n\
             - Unique Tokens: {}\n\
             - Coverage: {:.2}%\n\
             - Compression Ratio: {:.3}\n\
             - OOV Rate: {:.2}%\n\
             - Training Time: {:.2}s\n\
             - Iterations: {}\n\
             - Efficiency Score: {:.3}",
            self.vocab_size,
            self.total_tokens,
            self.unique_tokens,
            self.coverage * 100.0,
            self.compression_ratio,
            self.oov_rate * 100.0,
            self.training_time,
            self.iterations,
            self.efficiency_score()
        )
    }
}

/// Incremental trainer for updating existing tokenizers with new data.
pub struct IncrementalTrainer {
    config: AdvancedTrainingConfig,
    metrics: TrainingMetrics,
}

impl IncrementalTrainer {
    /// Create a new incremental trainer with the given configuration.
    pub fn new(config: AdvancedTrainingConfig) -> Self {
        Self {
            config,
            metrics: TrainingMetrics::new(),
        }
    }

    /// Update an existing BPE tokenizer with new training data.
    ///
    /// This method extends the vocabulary and merge rules of an existing
    /// BPE tokenizer by processing new texts and learning additional
    /// frequent patterns.
    pub fn update_bpe(
        &mut self,
        tokenizer: &BPETokenizer,
        new_texts: &[String],
    ) -> Result<BPETokenizer> {
        let start_time = Instant::now();

        // Extract current vocabulary and merge rules
        let mut existing_vocab = tokenizer.get_vocab_map().clone();
        let mut existing_merges = tokenizer.get_merge_rules().clone();

        // Process new texts using corpus processor
        let _processor = CorpusProcessor::new()
            .with_chunk_size(self.config.base_config.vocab_size / 10)
            .with_lowercase(true);

        let mut word_freqs = HashMap::new();
        for text in new_texts {
            for word in text.split_whitespace() {
                *word_freqs.entry(word.to_string()).or_insert(0) += 1;
            }
        }

        // Filter by frequency threshold
        word_freqs.retain(|_, &mut freq| freq >= self.config.base_config.min_frequency);

        // Add new character vocabulary
        let mut char_freqs = HashMap::new();
        for (word, freq) in &word_freqs {
            for ch in word.chars() {
                *char_freqs.entry(ch.to_string()).or_insert(0) += freq;
            }
        }

        let mut next_id = existing_vocab.len() as u32;
        for (ch, freq) in char_freqs {
            if freq >= self.config.base_config.min_frequency && !existing_vocab.contains_key(&ch) {
                existing_vocab.insert(ch, next_id);
                next_id += 1;
            }
        }

        // Continue BPE training with new data
        let mut splits = HashMap::new();
        for (word, freq) in word_freqs {
            if word.chars().count() <= self.config.base_config.max_input_chars_per_word {
                let split: Vec<String> = word.chars().map(|c| c.to_string()).collect();
                splits.insert(word, (split, freq));
            }
        }

        let mut iterations = 0;
        while existing_vocab.len() < self.config.base_config.vocab_size && iterations < 1000 {
            let mut pair_freqs = HashMap::new();

            for (split, freq) in splits.values() {
                for i in 0..split.len().saturating_sub(1) {
                    let pair = (split[i].clone(), split[i + 1].clone());
                    *pair_freqs.entry(pair).or_insert(0) += freq;
                }
            }

            if pair_freqs.is_empty() {
                break;
            }

            let best_pair = pair_freqs
                .iter()
                .max_by_key(|(_, &freq)| freq)
                .map(|(pair, _)| pair.clone())
                .expect("pair_freqs validated to be non-empty");

            let merged_token = format!("{}{}", best_pair.0, best_pair.1);
            existing_vocab.insert(merged_token, next_id);
            next_id += 1;

            existing_merges.push(best_pair.clone());

            let mut new_splits = HashMap::new();
            for (word, (split, freq)) in splits {
                let new_split = self.merge_word(&split, &best_pair);
                new_splits.insert(word, (new_split, freq));
            }
            splits = new_splits;
            iterations += 1;
        }

        self.metrics.training_time = start_time.elapsed().as_secs_f64();
        self.metrics.iterations = iterations;
        self.metrics.vocab_size = existing_vocab.len();

        Ok(BPETokenizer::new(existing_vocab, existing_merges))
    }

    /// Helper method to merge word splits based on a pair.
    fn merge_word(&self, word: &[String], pair: &(String, String)) -> Vec<String> {
        let mut new_word = Vec::new();
        let mut i = 0;

        while i < word.len() {
            if i < word.len() - 1 && word[i] == pair.0 && word[i + 1] == pair.1 {
                new_word.push(format!("{}{}", pair.0, pair.1));
                i += 2;
            } else {
                new_word.push(word[i].clone());
                i += 1;
            }
        }

        new_word
    }

    /// Calculate comprehensive training metrics for a tokenizer.
    pub fn calculate_metrics<T: trustformers_core::traits::Tokenizer>(
        &mut self,
        tokenizer: &T,
        texts: &[String],
    ) -> Result<TrainingMetrics> {
        let mut total_tokens = 0;
        let mut total_chars = 0;
        let mut oov_count = 0;
        let mut unique_tokens = HashSet::new();

        for text in texts {
            let tokenized = tokenizer.encode(text)?;
            total_tokens += tokenized.input_ids.len();
            total_chars += text.chars().count();

            for &id in &tokenized.input_ids {
                unique_tokens.insert(id);
                // Check if it's UNK token (assuming UNK has ID 1)
                if id == 1 {
                    oov_count += 1;
                }
            }
        }

        self.metrics.total_tokens = total_tokens;
        self.metrics.unique_tokens = unique_tokens.len();
        self.metrics.vocab_size = tokenizer.vocab_size();
        self.metrics.compression_ratio =
            if total_chars > 0 { total_tokens as f64 / total_chars as f64 } else { 0.0 };
        self.metrics.oov_rate =
            if total_tokens > 0 { oov_count as f64 / total_tokens as f64 } else { 0.0 };
        self.metrics.coverage = 1.0 - self.metrics.oov_rate;

        Ok(self.metrics.clone())
    }

    /// Save training checkpoint to disk.
    pub fn save_checkpoint<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let checkpoint_data = serde_json::to_string_pretty(&self.metrics)?;
        let mut file = File::create(path)?;
        file.write_all(checkpoint_data.as_bytes())?;
        Ok(())
    }

    /// Load training checkpoint from disk.
    pub fn load_checkpoint<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        self.metrics = serde_json::from_reader(reader)?;
        Ok(())
    }

    /// Get the current training metrics.
    pub fn get_metrics(&self) -> &TrainingMetrics {
        &self.metrics
    }

    /// Get mutable access to the configuration.
    pub fn get_config_mut(&mut self) -> &mut AdvancedTrainingConfig {
        &mut self.config
    }

    /// Get read-only access to the configuration.
    pub fn get_config(&self) -> &AdvancedTrainingConfig {
        &self.config
    }

    /// Reset metrics to initial state.
    pub fn reset_metrics(&mut self) {
        self.metrics = TrainingMetrics::new();
    }

    /// Check if early stopping criteria are met.
    pub fn should_stop_early(
        &self,
        current_score: f64,
        best_score: f64,
        patience_count: usize,
    ) -> bool {
        if current_score < best_score + self.config.min_improvement {
            patience_count >= self.config.early_stopping_patience
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_metrics_new() {
        let metrics = TrainingMetrics::new();
        assert_eq!(metrics.vocab_size, 0);
        assert_eq!(metrics.total_tokens, 0);
        assert_eq!(metrics.unique_tokens, 0);
        assert_eq!(metrics.coverage, 0.0);
        assert_eq!(metrics.compression_ratio, 0.0);
        assert_eq!(metrics.oov_rate, 0.0);
        assert_eq!(metrics.training_time, 0.0);
        assert_eq!(metrics.iterations, 0);
    }

    #[test]
    fn test_training_metrics_update() {
        let mut metrics = TrainingMetrics::new();
        metrics.update(1000, 50000, 800, 0.95, 0.6, 0.05, 120.5, 100);

        assert_eq!(metrics.vocab_size, 1000);
        assert_eq!(metrics.total_tokens, 50000);
        assert_eq!(metrics.unique_tokens, 800);
        assert_eq!(metrics.coverage, 0.95);
        assert_eq!(metrics.compression_ratio, 0.6);
        assert_eq!(metrics.oov_rate, 0.05);
        assert_eq!(metrics.training_time, 120.5);
        assert_eq!(metrics.iterations, 100);
    }

    #[test]
    fn test_efficiency_score() {
        let mut metrics = TrainingMetrics::new();
        metrics.coverage = 0.9;
        metrics.compression_ratio = 0.7;

        let expected_score = 0.7 * 0.9 + 0.3 * (1.0 - 0.7);
        assert!((metrics.efficiency_score() - expected_score).abs() < f64::EPSILON);
    }

    #[test]
    fn test_summary_generation() {
        let metrics = TrainingMetrics {
            vocab_size: 1000,
            total_tokens: 50000,
            unique_tokens: 800,
            coverage: 0.95,
            compression_ratio: 0.6,
            oov_rate: 0.05,
            training_time: 120.5,
            iterations: 100,
        };

        let summary = metrics.summary();
        assert!(summary.contains("Vocabulary Size: 1000"));
        assert!(summary.contains("Total Tokens: 50000"));
        assert!(summary.contains("Coverage: 95.00%"));
        assert!(summary.contains("Training Time: 120.50s"));
    }

    #[test]
    fn test_incremental_trainer_creation() {
        let config = AdvancedTrainingConfig::default();
        let trainer = IncrementalTrainer::new(config);

        assert_eq!(trainer.get_metrics().vocab_size, 0);
        assert_eq!(trainer.get_metrics().total_tokens, 0);
        assert_eq!(trainer.get_metrics().iterations, 0);
    }

    #[test]
    fn test_early_stopping_logic() {
        let config = AdvancedTrainingConfig {
            early_stopping_patience: 3,
            min_improvement: 0.01,
            ..Default::default()
        };
        let trainer = IncrementalTrainer::new(config);

        // No improvement, should stop after patience
        assert!(!trainer.should_stop_early(0.9, 0.95, 2));
        assert!(trainer.should_stop_early(0.9, 0.95, 3));

        // Sufficient improvement, should not stop
        assert!(!trainer.should_stop_early(0.96, 0.95, 5));
    }

    #[test]
    fn test_metrics_serialization() {
        let metrics = TrainingMetrics {
            vocab_size: 1000,
            total_tokens: 50000,
            unique_tokens: 800,
            coverage: 0.95,
            compression_ratio: 0.6,
            oov_rate: 0.05,
            training_time: 120.5,
            iterations: 100,
        };

        let serialized = serde_json::to_string(&metrics).unwrap();
        let deserialized: TrainingMetrics = serde_json::from_str(&serialized).unwrap();

        assert_eq!(metrics.vocab_size, deserialized.vocab_size);
        assert_eq!(metrics.total_tokens, deserialized.total_tokens);
        assert_eq!(metrics.unique_tokens, deserialized.unique_tokens);
        assert_eq!(metrics.coverage, deserialized.coverage);
        assert_eq!(metrics.compression_ratio, deserialized.compression_ratio);
        assert_eq!(metrics.oov_rate, deserialized.oov_rate);
        assert_eq!(metrics.training_time, deserialized.training_time);
        assert_eq!(metrics.iterations, deserialized.iterations);
    }
}
