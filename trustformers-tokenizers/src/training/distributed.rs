//! Distributed and concurrent training coordination.
//!
//! This module provides capabilities for training tokenizers on large corpora
//! using streaming processing, distributed coordination across multiple nodes,
//! and efficient checkpointing for resuming interrupted training sessions.

use crate::bpe::BPETokenizer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;
use trustformers_core::errors::{Result, TrustformersError};

use super::config::AdvancedTrainingConfig;

/// Streaming trainer for handling very large corpora without loading everything into memory.
pub struct StreamingTrainer {
    config: AdvancedTrainingConfig,
    chunk_size: usize,
    buffer_size: usize,
    save_progress_every: usize,
    temp_dir: Option<String>,
}

impl StreamingTrainer {
    /// Create a new streaming trainer with the given configuration.
    pub fn new(config: AdvancedTrainingConfig) -> Self {
        Self {
            config,
            chunk_size: 10000,
            buffer_size: 64 * 1024,      // 64KB
            save_progress_every: 100000, // Save progress every 100k lines
            temp_dir: None,
        }
    }

    /// Set the chunk size for processing.
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Set the buffer size for file I/O.
    pub fn with_buffer_size(mut self, buffer_size: usize) -> Self {
        self.buffer_size = buffer_size;
        self
    }

    /// Set the frequency for saving progress.
    pub fn with_progress_saving(mut self, save_every: usize) -> Self {
        self.save_progress_every = save_every;
        self
    }

    /// Set the temporary directory for intermediate files.
    pub fn with_temp_dir<P: AsRef<Path>>(mut self, temp_dir: P) -> Self {
        self.temp_dir = Some(temp_dir.as_ref().to_string_lossy().to_string());
        self
    }

    /// Train a BPE tokenizer from a large corpus file using streaming processing.
    ///
    /// This method processes the corpus in multiple passes without loading
    /// the entire file into memory, making it suitable for very large datasets.
    pub fn train_bpe_streaming<P: AsRef<Path>>(&self, corpus_path: P) -> Result<BPETokenizer> {
        let start_time = Instant::now();
        let file = File::open(&corpus_path).map_err(|e| {
            TrustformersError::io_error(format!("Failed to open corpus file: {}", e))
        })?;
        let reader = BufReader::with_capacity(self.buffer_size, file);

        // Phase 1: Stream through corpus to collect character frequencies
        let mut char_freqs = HashMap::new();
        let mut word_freqs = HashMap::new();
        let mut lines_processed = 0;

        println!("Phase 1: Collecting character and word frequencies...");
        for line in reader.lines() {
            let line = line
                .map_err(|e| TrustformersError::io_error(format!("Failed to read line: {}", e)))?;
            lines_processed += 1;

            if lines_processed % self.save_progress_every == 0 {
                println!("Processed {} lines", lines_processed);
            }

            for word in line.split_whitespace() {
                *word_freqs.entry(word.to_string()).or_insert(0) += 1;
                for ch in word.chars() {
                    *char_freqs.entry(ch.to_string()).or_insert(0) += 1;
                }
            }
        }

        println!(
            "Phase 1 complete: {} lines, {} unique words, {} unique chars",
            lines_processed,
            word_freqs.len(),
            char_freqs.len()
        );

        // Phase 2: Initialize vocabulary with frequent characters
        let mut vocab = HashMap::new();
        let mut next_id = 0u32;

        // Add special tokens first
        for token in &self.config.base_config.special_tokens {
            vocab.insert(token.clone(), next_id);
            next_id += 1;
        }

        // Add frequent characters
        for (ch, freq) in char_freqs {
            if freq >= self.config.base_config.min_frequency {
                vocab.insert(ch, next_id);
                next_id += 1;
            }
        }

        println!(
            "Phase 2: Initialized vocabulary with {} tokens",
            vocab.len()
        );

        // Phase 3: Streaming BPE merge learning
        let mut merge_rules = Vec::new();
        let target_vocab_size = self.config.base_config.vocab_size;

        println!(
            "Phase 3: Learning BPE merges (target vocab size: {})...",
            target_vocab_size
        );

        while vocab.len() < target_vocab_size && merge_rules.len() < target_vocab_size {
            // Stream through corpus again to find most frequent pair
            let file = File::open(&corpus_path).map_err(|e| {
                TrustformersError::io_error(format!("Failed to reopen corpus file: {}", e))
            })?;
            let reader = BufReader::with_capacity(self.buffer_size, file);

            let mut pair_freqs = HashMap::new();
            let mut chunk_count = 0;

            for line in reader.lines() {
                let line = line.map_err(|e| {
                    TrustformersError::io_error(format!("Failed to read line: {}", e))
                })?;
                chunk_count += 1;

                if chunk_count % self.chunk_size == 0 {
                    println!(
                        "  Processing chunk {}, vocab size: {}",
                        chunk_count / self.chunk_size,
                        vocab.len()
                    );
                }

                for word in line.split_whitespace() {
                    let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();

                    // Apply existing merge rules
                    for (a, b) in &merge_rules {
                        let mut i = 0;
                        while i < chars.len() - 1 {
                            if chars[i] == *a && chars[i + 1] == *b {
                                chars[i] = format!("{}{}", a, b);
                                chars.remove(i + 1);
                            } else {
                                i += 1;
                            }
                        }
                    }

                    // Count pairs
                    for i in 0..chars.len().saturating_sub(1) {
                        let pair = (chars[i].clone(), chars[i + 1].clone());
                        *pair_freqs.entry(pair).or_insert(0) += 1;
                    }
                }
            }

            // Find most frequent pair
            if let Some(((a, b), _)) = pair_freqs.iter().max_by_key(|(_, freq)| *freq) {
                let merged = format!("{}{}", a, b);
                vocab.insert(merged.clone(), next_id);
                merge_rules.push((a.clone(), b.clone()));
                next_id += 1;

                if merge_rules.len() % 1000 == 0 {
                    println!(
                        "  Learned {} merge rules, vocab size: {}",
                        merge_rules.len(),
                        vocab.len()
                    );
                }
            } else {
                break; // No more pairs to merge
            }
        }

        let training_time = start_time.elapsed().as_secs_f64();
        println!(
            "Training complete: {} vocab, {} merges, {:.2}s",
            vocab.len(),
            merge_rules.len(),
            training_time
        );

        Ok(BPETokenizer::new(vocab, merge_rules))
    }

    /// Train multiple tokenizers from different corpus files concurrently.
    ///
    /// This method spawns multiple training threads to process different
    /// corpora simultaneously, returning a vector of trained tokenizers.
    pub fn train_multi_corpus_bpe<P: AsRef<Path>>(
        &self,
        corpus_paths: &[P],
    ) -> Result<Vec<BPETokenizer>> {
        use std::sync::mpsc;
        use std::thread;

        if corpus_paths.is_empty() {
            return Err(TrustformersError::invalid_input(
                "No corpus paths provided".to_string(),
            ));
        }

        let (tx, rx) = mpsc::channel();
        let mut handles = Vec::new();

        println!("Starting training on {} corpora...", corpus_paths.len());

        for (idx, path) in corpus_paths.iter().enumerate() {
            let path = path.as_ref().to_path_buf();
            let config = self.config.clone();
            let chunk_size = self.chunk_size;
            let buffer_size = self.buffer_size;
            let tx = tx.clone();

            let handle = thread::spawn(move || {
                let trainer = StreamingTrainer {
                    config,
                    chunk_size,
                    buffer_size,
                    save_progress_every: 50000,
                    temp_dir: None,
                };

                println!("Training tokenizer {} on {:?}", idx, path);
                let result = trainer.train_bpe_streaming(&path);
                tx.send((idx, result)).expect("receiver should be available in training loop");
            });

            handles.push(handle);
        }

        drop(tx); // Close the sending side

        // Collect results
        let mut results: Vec<Option<Result<BPETokenizer>>> =
            (0..corpus_paths.len()).map(|_| None).collect();
        for (idx, result) in rx {
            results[idx] = Some(result);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().expect("training thread should not panic");
        }

        // Extract successful results
        let mut tokenizers = Vec::new();
        for (idx, result_opt) in results.into_iter().enumerate() {
            match result_opt {
                Some(Ok(tokenizer)) => {
                    println!("Successfully trained tokenizer {}", idx);
                    tokenizers.push(tokenizer);
                },
                Some(Err(e)) => {
                    return Err(TrustformersError::other(format!(
                        "Failed to train tokenizer {}: {}",
                        idx, e
                    )));
                },
                None => {
                    return Err(TrustformersError::other(format!(
                        "No result for tokenizer {}",
                        idx
                    )));
                },
            }
        }

        println!(
            "Multi-corpus training complete: {} tokenizers",
            tokenizers.len()
        );
        Ok(tokenizers)
    }

    /// Save training checkpoint to disk for resuming interrupted training.
    pub fn save_checkpoint<P: AsRef<Path>>(
        &self,
        path: P,
        vocab: &HashMap<String, u32>,
        merge_rules: &[(String, String)],
    ) -> Result<()> {
        use std::fs;

        let checkpoint = TrainingCheckpoint {
            vocab: vocab.clone(),
            merge_rules: merge_rules.to_vec(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs(),
            config: self.config.clone(),
        };

        let serialized = serde_json::to_string_pretty(&checkpoint).map_err(|e| {
            TrustformersError::serialization_error(format!("Failed to serialize checkpoint: {}", e))
        })?;

        fs::write(&path, serialized).map_err(|e| {
            TrustformersError::io_error(format!("Failed to write checkpoint: {}", e))
        })?;

        Ok(())
    }

    /// Load training checkpoint from disk to resume training.
    pub fn load_checkpoint<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<(HashMap<String, u32>, Vec<(String, String)>)> {
        use std::fs;

        let content = fs::read_to_string(&path).map_err(|e| {
            TrustformersError::io_error(format!("Failed to read checkpoint: {}", e))
        })?;

        let checkpoint: TrainingCheckpoint = serde_json::from_str(&content).map_err(|e| {
            TrustformersError::serialization_error(format!(
                "Failed to deserialize checkpoint: {}",
                e
            ))
        })?;

        Ok((checkpoint.vocab, checkpoint.merge_rules))
    }

    /// Estimate memory usage for processing a corpus.
    pub fn estimate_memory_usage<P: AsRef<Path>>(&self, corpus_path: P) -> Result<MemoryEstimate> {
        let file = File::open(&corpus_path).map_err(|e| {
            TrustformersError::io_error(format!("Failed to open corpus file: {}", e))
        })?;
        let reader = BufReader::new(file);

        let mut _total_chars = 0;
        let mut _total_words = 0;
        let mut unique_chars = std::collections::HashSet::new();
        let mut unique_words = std::collections::HashSet::new();
        let mut lines_sampled = 0;
        const SAMPLE_SIZE: usize = 10000;

        for line in reader.lines().take(SAMPLE_SIZE) {
            let line = line?;
            lines_sampled += 1;

            for word in line.split_whitespace() {
                unique_words.insert(word.to_string());
                _total_words += 1;

                for ch in word.chars() {
                    unique_chars.insert(ch);
                    _total_chars += 1;
                }
            }
        }

        let char_set_size = unique_chars.len() * 4; // Estimate 4 bytes per character entry
        let word_freq_size = unique_words.len() * 16; // Estimate 16 bytes per word frequency entry
        let pair_freq_size = unique_chars.len().pow(2) * 12; // Estimate for pair frequencies

        let estimated_peak_memory =
            char_set_size + word_freq_size + pair_freq_size + self.buffer_size;

        Ok(MemoryEstimate {
            unique_chars: unique_chars.len(),
            unique_words: unique_words.len(),
            estimated_peak_memory_bytes: estimated_peak_memory,
            lines_sampled,
            buffer_size: self.buffer_size,
        })
    }
}

/// Training checkpoint for resuming interrupted training sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    pub vocab: HashMap<String, u32>,
    pub merge_rules: Vec<(String, String)>,
    pub timestamp: u64,
    pub config: AdvancedTrainingConfig,
}

impl TrainingCheckpoint {
    /// Get the age of this checkpoint in seconds.
    pub fn age_seconds(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be after UNIX epoch")
            .as_secs()
            - self.timestamp
    }

    /// Check if this checkpoint is compatible with a given config.
    pub fn is_compatible(&self, config: &AdvancedTrainingConfig) -> bool {
        self.config.base_config.vocab_size == config.base_config.vocab_size
            && self.config.base_config.special_tokens == config.base_config.special_tokens
    }
}

/// Memory usage estimation for streaming training.
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    pub unique_chars: usize,
    pub unique_words: usize,
    pub estimated_peak_memory_bytes: usize,
    pub lines_sampled: usize,
    pub buffer_size: usize,
}

impl MemoryEstimate {
    /// Get estimated peak memory usage in megabytes.
    pub fn peak_memory_mb(&self) -> f64 {
        self.estimated_peak_memory_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Generate a memory usage report.
    pub fn report(&self) -> String {
        format!(
            "Memory Usage Estimate:\n\
             - Unique Characters: {}\n\
             - Unique Words: {}\n\
             - Peak Memory: {:.1} MB\n\
             - Sample Size: {} lines\n\
             - Buffer Size: {} bytes",
            self.unique_chars,
            self.unique_words,
            self.peak_memory_mb(),
            self.lines_sampled,
            self.buffer_size
        )
    }
}

/// Distributed training coordinator for training across multiple machines.
pub struct DistributedTrainingCoordinator {
    node_id: usize,
    total_nodes: usize,
    coordination_endpoint: String,
}

impl DistributedTrainingCoordinator {
    /// Create a new distributed training coordinator.
    pub fn new(node_id: usize, total_nodes: usize, coordination_endpoint: String) -> Self {
        Self {
            node_id,
            total_nodes,
            coordination_endpoint,
        }
    }

    /// Partition corpus for this node using round-robin assignment.
    pub fn partition_corpus<P: AsRef<Path>>(
        &self,
        corpus_path: P,
        output_dir: P,
    ) -> Result<String> {
        let file = File::open(&corpus_path)
            .map_err(|e| TrustformersError::io_error(format!("Failed to open corpus: {}", e)))?;
        let reader = BufReader::new(file);

        let output_path = output_dir.as_ref().join(format!("partition_{}.txt", self.node_id));
        let output_file = File::create(&output_path).map_err(|e| {
            TrustformersError::io_error(format!("Failed to create partition file: {}", e))
        })?;
        let mut writer = BufWriter::new(output_file);

        let mut line_count = 0;
        for line in reader.lines() {
            let line = line
                .map_err(|e| TrustformersError::io_error(format!("Failed to read line: {}", e)))?;

            // Assign lines to nodes in round-robin fashion
            if line_count % self.total_nodes == self.node_id {
                writeln!(writer, "{}", line).map_err(|e| {
                    TrustformersError::io_error(format!("Failed to write line: {}", e))
                })?;
            }
            line_count += 1;
        }

        writer
            .flush()
            .map_err(|e| TrustformersError::io_error(format!("Failed to flush writer: {}", e)))?;

        println!(
            "Node {} processed {} total lines, wrote partition to {:?}",
            self.node_id, line_count, output_path
        );

        Ok(output_path.to_string_lossy().to_string())
    }

    /// Merge vocabularies from multiple nodes.
    ///
    /// This is a simplified implementation that combines vocabularies
    /// by frequency. In a production system, this would involve more
    /// sophisticated coordination protocols.
    pub fn merge_distributed_vocabularies(
        &self,
        vocab_files: &[String],
    ) -> Result<HashMap<String, u32>> {
        let mut combined_vocab = HashMap::new();
        let mut combined_frequencies = HashMap::new();

        // Load all vocabularies and their frequencies
        for vocab_file in vocab_files {
            let content = std::fs::read_to_string(vocab_file).map_err(|e| {
                TrustformersError::io_error(format!(
                    "Failed to read vocab file {}: {}",
                    vocab_file, e
                ))
            })?;

            // Parse simple format: token frequency
            for line in content.lines() {
                let parts: Vec<&str> = line.split('\t').collect();
                if parts.len() == 2 {
                    let token = parts[0].to_string();
                    let freq: usize = parts[1].parse().map_err(|e| {
                        TrustformersError::other(format!("Invalid frequency: {}", e))
                    })?;
                    *combined_frequencies.entry(token).or_insert(0) += freq;
                }
            }
        }

        // Create final vocabulary with most frequent tokens
        let mut sorted_tokens: Vec<_> = combined_frequencies.into_iter().collect();
        sorted_tokens.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by frequency descending

        for (idx, (token, _freq)) in sorted_tokens.into_iter().take(50000).enumerate() {
            combined_vocab.insert(token, idx as u32);
        }

        println!(
            "Merged vocabularies from {} nodes: {} tokens",
            vocab_files.len(),
            combined_vocab.len()
        );
        Ok(combined_vocab)
    }

    /// Get node information.
    pub fn get_node_info(&self) -> NodeInfo {
        NodeInfo {
            node_id: self.node_id,
            total_nodes: self.total_nodes,
            coordination_endpoint: self.coordination_endpoint.clone(),
        }
    }
}

/// Information about a node in distributed training.
#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub node_id: usize,
    pub total_nodes: usize,
    pub coordination_endpoint: String,
}

impl NodeInfo {
    /// Check if this is the master node (node 0).
    pub fn is_master(&self) -> bool {
        self.node_id == 0
    }

    /// Get the expected partition for this node (for debugging).
    pub fn get_partition_info(&self, total_lines: usize) -> PartitionInfo {
        let lines_per_node = total_lines / self.total_nodes;
        let remainder = total_lines % self.total_nodes;

        let expected_lines =
            if self.node_id < remainder { lines_per_node + 1 } else { lines_per_node };

        PartitionInfo {
            node_id: self.node_id,
            expected_lines,
            total_lines,
        }
    }
}

/// Information about data partitioning for a node.
#[derive(Debug, Clone)]
pub struct PartitionInfo {
    pub node_id: usize,
    pub expected_lines: usize,
    pub total_lines: usize,
}

impl PartitionInfo {
    /// Get the percentage of data assigned to this node.
    pub fn percentage(&self) -> f64 {
        if self.total_lines > 0 {
            (self.expected_lines as f64 / self.total_lines as f64) * 100.0
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_trainer_creation() {
        let config = AdvancedTrainingConfig::default();
        let trainer = StreamingTrainer::new(config)
            .with_chunk_size(5000)
            .with_buffer_size(32 * 1024)
            .with_progress_saving(50000);

        assert_eq!(trainer.chunk_size, 5000);
        assert_eq!(trainer.buffer_size, 32 * 1024);
        assert_eq!(trainer.save_progress_every, 50000);
    }

    #[test]
    fn test_memory_estimate() {
        let estimate = MemoryEstimate {
            unique_chars: 100,
            unique_words: 10000,
            estimated_peak_memory_bytes: 1024 * 1024, // 1 MB
            lines_sampled: 1000,
            buffer_size: 64 * 1024,
        };

        assert_eq!(estimate.peak_memory_mb(), 1.0);
        let report = estimate.report();
        assert!(report.contains("1.0 MB"));
        assert!(report.contains("100"));
        assert!(report.contains("10000"));
    }

    #[test]
    fn test_training_checkpoint() {
        let config = AdvancedTrainingConfig::default();
        let vocab = HashMap::new();
        let merge_rules = vec![("a".to_string(), "b".to_string())];

        let checkpoint = TrainingCheckpoint {
            vocab,
            merge_rules,
            timestamp: 1234567890,
            config: config.clone(),
        };

        assert!(checkpoint.is_compatible(&config));
        assert!(checkpoint.age_seconds() > 0);
    }

    #[test]
    fn test_distributed_coordinator() {
        let coordinator = DistributedTrainingCoordinator::new(1, 4, "localhost:8080".to_string());

        let node_info = coordinator.get_node_info();
        assert_eq!(node_info.node_id, 1);
        assert_eq!(node_info.total_nodes, 4);
        assert!(!node_info.is_master());

        let partition_info = node_info.get_partition_info(1000);
        assert_eq!(partition_info.expected_lines, 250);
        assert_eq!(partition_info.percentage(), 25.0);
    }

    #[test]
    fn test_node_info_master() {
        let master_info = NodeInfo {
            node_id: 0,
            total_nodes: 3,
            coordination_endpoint: "localhost:8080".to_string(),
        };

        assert!(master_info.is_master());

        let worker_info = NodeInfo {
            node_id: 1,
            total_nodes: 3,
            coordination_endpoint: "localhost:8080".to_string(),
        };

        assert!(!worker_info.is_master());
    }

    #[test]
    fn test_partition_info_calculation() {
        let node_info = NodeInfo {
            node_id: 0,
            total_nodes: 3,
            coordination_endpoint: "localhost:8080".to_string(),
        };

        // Test with lines that divide evenly
        let partition = node_info.get_partition_info(300);
        assert_eq!(partition.expected_lines, 100);
        assert!((partition.percentage() - (100.0 / 3.0)).abs() < 1e-10);

        // Test with remainder
        let partition_with_remainder = node_info.get_partition_info(301);
        assert_eq!(partition_with_remainder.expected_lines, 101); // Node 0 gets extra line
    }
}
