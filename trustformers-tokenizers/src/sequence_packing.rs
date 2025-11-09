use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::TokenizedInput;

/// Configuration for sequence packing
#[derive(Debug, Clone)]
pub struct PackingConfig {
    /// Maximum sequence length after packing
    pub max_packed_length: usize,

    /// Padding token ID
    pub pad_token_id: u32,

    /// Separator token ID (used between packed sequences)
    pub sep_token_id: Option<u32>,

    /// Whether to add separator tokens between sequences
    pub add_separators: bool,

    /// Minimum sequence length to consider for packing
    pub min_sequence_length: usize,

    /// Maximum number of sequences to pack together
    pub max_sequences_per_pack: usize,

    /// Packing strategy to use
    pub strategy: PackingStrategy,

    /// Whether to preserve sequence boundaries in attention masks
    pub preserve_boundaries: bool,
}

impl Default for PackingConfig {
    fn default() -> Self {
        Self {
            max_packed_length: 512,
            pad_token_id: 0,
            sep_token_id: None,
            add_separators: false,
            min_sequence_length: 10,
            max_sequences_per_pack: 4,
            strategy: PackingStrategy::FirstFit,
            preserve_boundaries: true,
        }
    }
}

/// Different strategies for packing sequences
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackingStrategy {
    /// Pack sequences in order, fitting as many as possible
    FirstFit,

    /// Sort by length and pack optimally
    BestFit,

    /// Group similar length sequences together
    SimilarLength,

    /// Random shuffling before packing
    Random,
}

/// Information about how sequences were packed
#[derive(Debug, Clone)]
pub struct PackingInfo {
    /// Original sequence indices that were packed together
    pub original_indices: Vec<usize>,

    /// Start and end positions of each sequence in the packed sequence
    pub sequence_boundaries: Vec<(usize, usize)>,

    /// Number of sequences packed together
    pub num_sequences: usize,

    /// Total length of the packed sequence (excluding padding)
    pub packed_length: usize,

    /// Efficiency ratio (used length / max length)
    pub efficiency: f32,
}

/// A packed sequence with metadata
#[derive(Debug, Clone)]
pub struct PackedSequence {
    /// The packed tokenized input
    pub tokenized_input: TokenizedInput,

    /// Information about how this was packed
    pub packing_info: PackingInfo,

    /// Token type IDs for each token (0 for first sequence, 1 for second, etc.)
    pub sequence_ids: Vec<u32>,
}

/// Statistics about the packing process
#[derive(Debug, Clone)]
pub struct PackingStats {
    /// Total number of original sequences
    pub total_sequences: usize,

    /// Number of packed sequences produced
    pub num_packed_sequences: usize,

    /// Average number of sequences per pack
    pub avg_sequences_per_pack: f32,

    /// Average efficiency (utilization ratio)
    pub avg_efficiency: f32,

    /// Number of sequences that couldn't be packed
    pub unpacked_sequences: usize,

    /// Total tokens saved through packing
    pub tokens_saved: usize,

    /// Compression ratio (original tokens / packed tokens)
    pub compression_ratio: f32,
}

/// Main sequence packing utility
pub struct SequencePacker {
    config: PackingConfig,
}

impl SequencePacker {
    /// Create a new sequence packer with the given configuration
    pub fn new(config: PackingConfig) -> Self {
        Self { config }
    }

    /// Pack a batch of tokenized inputs
    pub fn pack_sequences(
        &self,
        sequences: &[TokenizedInput],
    ) -> Result<(Vec<PackedSequence>, PackingStats)> {
        if sequences.is_empty() {
            return Ok((vec![], PackingStats::default()));
        }

        // Prepare sequences for packing
        let mut seq_items: Vec<SequenceItem> = sequences
            .iter()
            .enumerate()
            .map(|(idx, seq)| SequenceItem {
                index: idx,
                length: seq.input_ids.len(),
                tokenized_input: seq.clone(),
            })
            .collect();

        // Filter out sequences that are too long or too short
        seq_items.retain(|item| {
            item.length >= self.config.min_sequence_length
                && item.length <= self.config.max_packed_length
        });

        // Apply packing strategy
        self.apply_packing_strategy(&mut seq_items);

        // Pack sequences
        let packed_sequences = self.pack_sequences_greedy(&seq_items)?;

        // Calculate statistics
        let stats = self.calculate_stats(sequences.len(), &packed_sequences);

        Ok((packed_sequences, stats))
    }

    /// Unpack a packed sequence back to individual sequences
    pub fn unpack_sequence(&self, packed: &PackedSequence) -> Result<Vec<TokenizedInput>> {
        let mut sequences = Vec::new();

        for (start, end) in &packed.packing_info.sequence_boundaries {
            if *end > packed.tokenized_input.input_ids.len() {
                return Err(TrustformersError::invalid_input(
                    "Invalid sequence boundary in packed sequence".to_string(),
                ));
            }

            let input_ids = packed.tokenized_input.input_ids[*start..*end].to_vec();
            let attention_mask = packed.tokenized_input.attention_mask[*start..*end].to_vec();

            let token_type_ids = packed
                .tokenized_input
                .token_type_ids
                .as_ref()
                .map(|ttids| ttids[*start..*end].to_vec());

            sequences.push(TokenizedInput {
                input_ids,
                attention_mask,
                token_type_ids,
                special_tokens_mask: None,
                offset_mapping: None,
                overflowing_tokens: None,
            });
        }

        Ok(sequences)
    }

    /// Apply the configured packing strategy to sort sequences
    fn apply_packing_strategy(&self, seq_items: &mut [SequenceItem]) {
        match self.config.strategy {
            PackingStrategy::FirstFit => {
                // No sorting needed, use original order
            },
            PackingStrategy::BestFit => {
                // Sort by length descending for better bin packing
                seq_items.sort_by(|a, b| b.length.cmp(&a.length));
            },
            PackingStrategy::SimilarLength => {
                // Sort by length ascending to group similar lengths
                seq_items.sort_by(|a, b| a.length.cmp(&b.length));
            },
            PackingStrategy::Random => {
                // Shuffle randomly
                use scirs2_core::random::*;  // SciRS2 Integration Policy
                use scirs2_core::random::SliceRandom;  // Explicit for trait methods
                let mut rng = thread_rng();
                seq_items.shuffle(rng.rng_mut());
            },
        }
    }

    /// Pack sequences using a greedy algorithm
    fn pack_sequences_greedy(&self, seq_items: &[SequenceItem]) -> Result<Vec<PackedSequence>> {
        let mut packed_sequences = Vec::new();
        let mut used = vec![false; seq_items.len()];

        for i in 0..seq_items.len() {
            if used[i] {
                continue;
            }

            let mut current_pack = vec![i];
            let mut current_length = seq_items[i].length;
            used[i] = true;

            // Add separators if configured
            if self.config.add_separators && self.config.sep_token_id.is_some() {
                current_length += 1; // Space for separator
            }

            // Try to fit more sequences
            for j in (i + 1)..seq_items.len() {
                if used[j] || current_pack.len() >= self.config.max_sequences_per_pack {
                    continue;
                }

                let additional_length = seq_items[j].length;
                let separator_length =
                    if self.config.add_separators && self.config.sep_token_id.is_some() {
                        1
                    } else {
                        0
                    };

                if current_length + additional_length + separator_length
                    <= self.config.max_packed_length
                {
                    current_pack.push(j);
                    current_length += additional_length + separator_length;
                    used[j] = true;
                }
            }

            // Create packed sequence
            let packed = self.create_packed_sequence(&current_pack, seq_items)?;
            packed_sequences.push(packed);
        }

        Ok(packed_sequences)
    }

    /// Create a packed sequence from a group of sequence indices
    fn create_packed_sequence(
        &self,
        indices: &[usize],
        seq_items: &[SequenceItem],
    ) -> Result<PackedSequence> {
        let mut packed_input_ids = Vec::new();
        let mut packed_attention_mask = Vec::new();
        let mut packed_token_type_ids: Vec<u32> = Vec::new();
        let mut sequence_ids = Vec::new();
        let mut sequence_boundaries = Vec::new();

        for (seq_idx, &item_idx) in indices.iter().enumerate() {
            let item = &seq_items[item_idx];
            let start_pos = packed_input_ids.len();

            // Add the sequence
            packed_input_ids.extend(&item.tokenized_input.input_ids);
            packed_attention_mask.extend(&item.tokenized_input.attention_mask);

            // Add token type IDs
            if let Some(ref ttids) = item.tokenized_input.token_type_ids {
                packed_token_type_ids.extend(ttids);
            } else {
                packed_token_type_ids.extend(vec![0u32; item.tokenized_input.input_ids.len()]);
            }

            // Add sequence IDs for tracking
            sequence_ids.extend(vec![seq_idx as u32; item.tokenized_input.input_ids.len()]);

            let end_pos = packed_input_ids.len();
            sequence_boundaries.push((start_pos, end_pos));

            // Add separator if not the last sequence and separators are enabled
            if seq_idx < indices.len() - 1 && self.config.add_separators {
                if let Some(sep_token_id) = self.config.sep_token_id {
                    packed_input_ids.push(sep_token_id);
                    packed_attention_mask.push(1);
                    packed_token_type_ids.push(0u32);
                    sequence_ids.push(seq_idx as u32);
                }
            }
        }

        // Pad to max length if needed
        let current_length = packed_input_ids.len();
        if current_length < self.config.max_packed_length {
            let padding_length = self.config.max_packed_length - current_length;
            packed_input_ids.extend(vec![self.config.pad_token_id; padding_length]);
            packed_attention_mask.extend(vec![0u8; padding_length]);
            packed_token_type_ids.extend(vec![0u32; padding_length]);
            sequence_ids.extend(vec![u32::MAX; padding_length]); // Use MAX to indicate padding
        }

        let packing_info = PackingInfo {
            original_indices: indices.iter().map(|&i| seq_items[i].index).collect(),
            sequence_boundaries,
            num_sequences: indices.len(),
            packed_length: current_length,
            efficiency: current_length as f32 / self.config.max_packed_length as f32,
        };

        let tokenized_input = TokenizedInput {
            input_ids: packed_input_ids,
            attention_mask: packed_attention_mask,
            token_type_ids: Some(packed_token_type_ids),
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        };

        Ok(PackedSequence {
            tokenized_input,
            packing_info,
            sequence_ids,
        })
    }

    /// Calculate packing statistics
    fn calculate_stats(
        &self,
        original_count: usize,
        packed_sequences: &[PackedSequence],
    ) -> PackingStats {
        let total_packed_sequences = packed_sequences.len();
        let total_sequences_packed: usize =
            packed_sequences.iter().map(|p| p.packing_info.num_sequences).sum();

        let avg_sequences_per_pack = if total_packed_sequences > 0 {
            total_sequences_packed as f32 / total_packed_sequences as f32
        } else {
            0.0
        };

        let avg_efficiency = if total_packed_sequences > 0 {
            packed_sequences.iter().map(|p| p.packing_info.efficiency).sum::<f32>()
                / total_packed_sequences as f32
        } else {
            0.0
        };

        let unpacked_sequences = original_count.saturating_sub(total_sequences_packed);

        // Calculate token savings
        let original_tokens_if_padded = original_count * self.config.max_packed_length;
        let actual_tokens_used: usize =
            packed_sequences.iter().map(|_p| self.config.max_packed_length).sum();
        let tokens_saved = original_tokens_if_padded.saturating_sub(actual_tokens_used);

        let compression_ratio = if actual_tokens_used > 0 {
            original_tokens_if_padded as f32 / actual_tokens_used as f32
        } else {
            1.0
        };

        PackingStats {
            total_sequences: original_count,
            num_packed_sequences: total_packed_sequences,
            avg_sequences_per_pack,
            avg_efficiency,
            unpacked_sequences,
            tokens_saved,
            compression_ratio,
        }
    }
}

impl Default for PackingStats {
    fn default() -> Self {
        Self {
            total_sequences: 0,
            num_packed_sequences: 0,
            avg_sequences_per_pack: 0.0,
            avg_efficiency: 0.0,
            unpacked_sequences: 0,
            tokens_saved: 0,
            compression_ratio: 1.0,
        }
    }
}

/// Internal representation of a sequence for packing
#[derive(Debug, Clone)]
struct SequenceItem {
    index: usize,
    length: usize,
    tokenized_input: TokenizedInput,
}

/// Advanced sequence packer with additional features
pub struct AdvancedSequencePacker {
    base_packer: SequencePacker,
    length_histogram: HashMap<usize, usize>,
    #[allow(dead_code)]
    packing_cache: HashMap<Vec<usize>, PackedSequence>,
}

impl AdvancedSequencePacker {
    /// Create a new advanced sequence packer
    pub fn new(config: PackingConfig) -> Self {
        Self {
            base_packer: SequencePacker::new(config),
            length_histogram: HashMap::new(),
            packing_cache: HashMap::new(),
        }
    }

    /// Pack sequences with length-aware optimization
    pub fn pack_with_optimization(
        &mut self,
        sequences: &[TokenizedInput],
    ) -> Result<(Vec<PackedSequence>, PackingStats)> {
        // Update length histogram
        self.update_length_histogram(sequences);

        // Use the base packer but with optimized strategy
        self.base_packer.pack_sequences(sequences)
    }

    /// Update the length histogram for optimization
    fn update_length_histogram(&mut self, sequences: &[TokenizedInput]) {
        for seq in sequences {
            let length = seq.input_ids.len();
            *self.length_histogram.entry(length).or_insert(0) += 1;
        }
    }

    /// Get length distribution statistics
    pub fn get_length_stats(&self) -> Vec<(usize, usize)> {
        let mut stats: Vec<_> =
            self.length_histogram.iter().map(|(&len, &count)| (len, count)).collect();
        stats.sort_by_key(|&(len, _)| len);
        stats
    }

    /// Suggest optimal packing configuration based on observed data
    pub fn suggest_config(&self) -> PackingConfig {
        let mut config = self.base_packer.config.clone();

        if !self.length_histogram.is_empty() {
            // Calculate percentiles
            let total_sequences: usize = self.length_histogram.values().sum();
            let mut cumulative = 0;
            let mut percentile_95 = 0;

            for (&length, &count) in &self.length_histogram {
                cumulative += count;
                if cumulative >= (total_sequences * 95) / 100 {
                    percentile_95 = length;
                    break;
                }
            }

            // Suggest max length based on 95th percentile
            if percentile_95 > 0 {
                config.max_packed_length = (percentile_95 * 2).max(512);
            }

            // Suggest strategy based on length distribution
            let length_variance = self.calculate_length_variance();
            if length_variance < 100.0 {
                config.strategy = PackingStrategy::SimilarLength;
            } else {
                config.strategy = PackingStrategy::BestFit;
            }
        }

        config
    }

    /// Calculate variance in sequence lengths
    fn calculate_length_variance(&self) -> f64 {
        if self.length_histogram.is_empty() {
            return 0.0;
        }

        let total_sequences: usize = self.length_histogram.values().sum();
        let mean: f64 = self
            .length_histogram
            .iter()
            .map(|(&len, &count)| len as f64 * count as f64)
            .sum::<f64>()
            / total_sequences as f64;

        let variance: f64 = self
            .length_histogram
            .iter()
            .map(|(&len, &count)| {
                let diff = len as f64 - mean;
                diff * diff * count as f64
            })
            .sum::<f64>()
            / total_sequences as f64;

        variance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_sequence(length: usize) -> TokenizedInput {
        TokenizedInput {
            input_ids: (0..length).map(|i| i as u32).collect(),
            attention_mask: vec![1u8; length],
            token_type_ids: Some(vec![0u32; length]),
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        }
    }

    #[test]
    fn test_basic_packing() {
        let config = PackingConfig {
            max_packed_length: 100,
            pad_token_id: 0,
            ..Default::default()
        };
        let packer = SequencePacker::new(config);

        let sequences = vec![
            create_test_sequence(30),
            create_test_sequence(25),
            create_test_sequence(40),
        ];

        let (packed, stats) = packer.pack_sequences(&sequences).unwrap();

        assert!(!packed.is_empty());
        assert_eq!(stats.total_sequences, 3);
    }

    #[test]
    fn test_packing_with_separators() {
        let config = PackingConfig {
            max_packed_length: 100,
            pad_token_id: 0,
            sep_token_id: Some(999),
            add_separators: true,
            ..Default::default()
        };
        let packer = SequencePacker::new(config);

        let sequences = vec![create_test_sequence(20), create_test_sequence(20)];

        let (packed, _) = packer.pack_sequences(&sequences).unwrap();

        assert!(!packed.is_empty());
        // Should have separator between sequences
        assert!(packed[0].tokenized_input.input_ids.contains(&999));
    }

    #[test]
    fn test_unpacking() {
        let config = PackingConfig {
            max_packed_length: 100,
            pad_token_id: 0,
            ..Default::default()
        };
        let packer = SequencePacker::new(config);

        let original_sequences = vec![create_test_sequence(30), create_test_sequence(25)];

        let (packed, _) = packer.pack_sequences(&original_sequences).unwrap();
        let unpacked = packer.unpack_sequence(&packed[0]).unwrap();

        assert_eq!(unpacked.len(), packed[0].packing_info.num_sequences);
    }

    #[test]
    fn test_packing_strategies() {
        let config = PackingConfig {
            max_packed_length: 100,
            strategy: PackingStrategy::BestFit,
            ..Default::default()
        };
        let packer = SequencePacker::new(config);

        let sequences = vec![
            create_test_sequence(80),
            create_test_sequence(10),
            create_test_sequence(15),
            create_test_sequence(20),
        ];

        let (packed, stats) = packer.pack_sequences(&sequences).unwrap();

        assert!(!packed.is_empty());
        assert!(stats.avg_efficiency > 0.0);
    }

    #[test]
    fn test_advanced_packer() {
        let config = PackingConfig::default();
        let mut advanced_packer = AdvancedSequencePacker::new(config);

        let sequences = vec![
            create_test_sequence(50),
            create_test_sequence(55),
            create_test_sequence(48),
            create_test_sequence(52),
        ];

        let (packed, stats) = advanced_packer.pack_with_optimization(&sequences).unwrap();

        assert!(!packed.is_empty());
        assert_eq!(stats.total_sequences, 4);

        let length_stats = advanced_packer.get_length_stats();
        assert!(!length_stats.is_empty());

        let suggested_config = advanced_packer.suggest_config();
        assert!(suggested_config.max_packed_length > 0);
    }

    #[test]
    fn test_efficiency_calculation() {
        let config = PackingConfig {
            max_packed_length: 100,
            ..Default::default()
        };
        let packer = SequencePacker::new(config);

        // Perfect packing scenario
        let sequences = vec![create_test_sequence(50), create_test_sequence(50)];

        let (packed, stats) = packer.pack_sequences(&sequences).unwrap();

        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0].packing_info.efficiency, 1.0); // Perfect efficiency
        assert!(stats.avg_efficiency > 0.9);
    }

    #[test]
    fn test_max_sequences_per_pack() {
        let config = PackingConfig {
            max_packed_length: 1000,
            max_sequences_per_pack: 2,
            ..Default::default()
        };
        let packer = SequencePacker::new(config);

        let sequences = vec![
            create_test_sequence(10),
            create_test_sequence(10),
            create_test_sequence(10),
            create_test_sequence(10),
        ];

        let (packed, _) = packer.pack_sequences(&sequences).unwrap();

        // Should create 2 packs with max 2 sequences each
        assert_eq!(packed.len(), 2);
        for pack in packed {
            assert!(pack.packing_info.num_sequences <= 2);
        }
    }
}
