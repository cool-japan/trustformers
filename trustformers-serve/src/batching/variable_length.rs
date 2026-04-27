//! Variable-Length Batching for TrustformeRS Inference Server
//!
//! Efficient batching for generation tasks where sequences have different
//! lengths.  Sequences are bucketed by length range, padded within each bucket
//! to minimise wasted compute, and emitted as `PaddedBatch` values when the
//! token budget or batch-size limit is reached.

use std::collections::VecDeque;

// ── Configuration ──────────────────────────────────────────────────────────────

/// Configuration for [`VariableLengthBatcher`].
#[derive(Debug, Clone)]
pub struct VariableLengthBatchConfig {
    /// Bucket upper-boundaries (exclusive), e.g. `[32, 64, 128, 256, 512]`.
    ///
    /// Sequences longer than the last boundary land in an overflow bucket.
    pub bucket_boundaries: Vec<usize>,
    /// Maximum number of *padded* tokens across all sequences in one batch.
    pub max_batch_tokens: usize,
    /// Maximum number of sequences in a single batch.
    pub max_batch_size: usize,
    /// When true, pad every sequence to the bucket boundary instead of the
    /// maximum length actually present in the batch.
    pub pad_to_bucket: bool,
    /// When true, [`flush_all`][VariableLengthBatcher::flush_all] drops the
    /// last incomplete batch instead of emitting it.
    pub drop_last: bool,
}

impl Default for VariableLengthBatchConfig {
    fn default() -> Self {
        Self {
            bucket_boundaries: vec![32, 64, 128, 256, 512],
            max_batch_tokens: 4096,
            max_batch_size: 32,
            pad_to_bucket: true,
            drop_last: false,
        }
    }
}

// ── SequenceItem ───────────────────────────────────────────────────────────────

/// A single sequence to be batched.
#[derive(Debug, Clone)]
pub struct SequenceItem {
    /// Caller-assigned identifier (opaque).
    pub id: u64,
    /// Number of tokens in this sequence.
    pub length: usize,
    /// Simulated token embeddings (may be shorter/longer than `length` in
    /// practice; treated as raw payload here).
    pub data: Vec<f32>,
}

// ── PaddedBatch ────────────────────────────────────────────────────────────────

/// A batch of sequences that have been padded to a common length.
#[derive(Debug, Clone)]
pub struct PaddedBatch {
    /// Sequences included in this batch.
    pub items: Vec<SequenceItem>,
    /// All sequences are padded to this length.
    pub padded_length: usize,
    /// Fraction of total allocated tokens that are padding.
    ///
    /// `padding_ratio = total_padding / total_allocated_tokens`
    pub padding_ratio: f64,
    /// Index of the bucket these sequences came from.
    pub bucket_index: usize,
}

// ── BatcherStats ───────────────────────────────────────────────────────────────

/// Aggregate statistics for a [`VariableLengthBatcher`] instance.
#[derive(Debug, Clone, Default)]
pub struct BatcherStats {
    /// Total number of sequences that have been added.
    pub total_items: usize,
    /// Total number of batches that have been emitted.
    pub total_batches: usize,
    /// Average padding ratio across all emitted batches.
    pub avg_padding_ratio: f64,
    /// Number of sequences currently waiting in each bucket.
    pub bucket_counts: Vec<usize>,
}

// ── VariableLengthBatcher ─────────────────────────────────────────────────────

/// Batches sequences of varying lengths with minimal padding waste.
pub struct VariableLengthBatcher {
    config: VariableLengthBatchConfig,
    /// One queue per bucket (including the overflow bucket at the end).
    buckets: Vec<VecDeque<SequenceItem>>,
    // ── statistics ──────────────────────────────────────────────────────────
    total_items: usize,
    total_batches: usize,
    sum_padding_ratio: f64,
}

impl VariableLengthBatcher {
    /// Create a new batcher with the given configuration.
    pub fn new(config: VariableLengthBatchConfig) -> Self {
        // Extra bucket at the end for sequences that exceed all boundaries.
        let num_buckets = config.bucket_boundaries.len() + 1;
        Self {
            buckets: (0..num_buckets).map(|_| VecDeque::new()).collect(),
            config,
            total_items: 0,
            total_batches: 0,
            sum_padding_ratio: 0.0,
        }
    }

    /// Add a sequence to the appropriate bucket.
    pub fn add(&mut self, item: SequenceItem) {
        let bucket = self.bucket_index_for(item.length);
        self.buckets[bucket].push_back(item);
        self.total_items += 1;
    }

    /// Try to form one batch from the first non-empty bucket that has enough
    /// items to fill a batch, or enough tokens.  Returns `None` when no bucket
    /// has sufficient sequences to produce a batch under the current limits.
    ///
    /// The algorithm:
    /// 1. Iterate buckets in order.
    /// 2. For each bucket, greedily collect sequences until `max_batch_size`
    ///    or `max_batch_tokens` would be exceeded.
    /// 3. Emit the first bucket that produced at least one sequence whose
    ///    padded token total meets the `max_batch_tokens` budget (greedy fill)
    ///    **or** whose item count reached `max_batch_size`.
    ///
    /// For use with `drop_last: false`, partial buckets are also returned by
    /// [`flush_all`].
    pub fn next_batch(&mut self) -> Option<PaddedBatch> {
        for bucket_idx in 0..self.buckets.len() {
            if self.buckets[bucket_idx].is_empty() {
                continue;
            }
            let padded_len = self.padded_length_for_bucket(bucket_idx);
            let max_by_tokens = self.config.max_batch_tokens / padded_len.max(1);
            let limit = max_by_tokens.min(self.config.max_batch_size);
            if limit == 0 {
                // A single sequence already exceeds token budget — emit it
                // alone to avoid starvation.
                let item = self.buckets[bucket_idx].pop_front()?;
                let batch = self.make_batch(vec![item], padded_len, bucket_idx);
                return Some(batch);
            }
            // Only emit a full batch here (partial batches are deferred to
            // flush_all).  A bucket is "full" when we can extract exactly
            // `limit` items.
            if self.buckets[bucket_idx].len() >= limit {
                let items: Vec<SequenceItem> = self.buckets[bucket_idx]
                    .drain(..limit)
                    .collect();
                let batch = self.make_batch(items, padded_len, bucket_idx);
                return Some(batch);
            }
        }
        None
    }

    /// Emit all remaining sequences, even if batches are not full.
    ///
    /// When `drop_last` is set, partial batches are discarded.
    pub fn flush_all(&mut self) -> Vec<PaddedBatch> {
        let mut out = Vec::new();
        for bucket_idx in 0..self.buckets.len() {
            if self.buckets[bucket_idx].is_empty() {
                continue;
            }
            let padded_len = self.padded_length_for_bucket(bucket_idx);
            let max_by_tokens = self.config.max_batch_tokens / padded_len.max(1);
            let chunk_size = max_by_tokens.min(self.config.max_batch_size).max(1);

            loop {
                if self.buckets[bucket_idx].is_empty() {
                    break;
                }
                let available = self.buckets[bucket_idx].len();
                let take = available.min(chunk_size);
                let is_partial = take < chunk_size;

                if is_partial && self.config.drop_last {
                    // Discard the remainder.
                    self.buckets[bucket_idx].clear();
                    break;
                }

                let items: Vec<SequenceItem> =
                    self.buckets[bucket_idx].drain(..take).collect();
                let batch = self.make_batch(items, padded_len, bucket_idx);
                out.push(batch);
            }
        }
        out
    }

    /// Return current aggregate statistics.
    pub fn stats(&self) -> BatcherStats {
        let bucket_counts: Vec<usize> = self.buckets.iter().map(|b| b.len()).collect();
        let avg_padding_ratio = if self.total_batches == 0 {
            0.0
        } else {
            self.sum_padding_ratio / self.total_batches as f64
        };
        BatcherStats {
            total_items: self.total_items,
            total_batches: self.total_batches,
            avg_padding_ratio,
            bucket_counts,
        }
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    /// Return the bucket index for a sequence of the given length.
    fn bucket_index_for(&self, length: usize) -> usize {
        for (i, &boundary) in self.config.bucket_boundaries.iter().enumerate() {
            if length <= boundary {
                return i;
            }
        }
        // Overflow bucket.
        self.config.bucket_boundaries.len()
    }

    /// Compute the padded length for all sequences in `bucket_idx`.
    ///
    /// When `pad_to_bucket` is true this is the bucket's upper boundary;
    /// otherwise it is the maximum actual length among queued sequences.
    fn padded_length_for_bucket(&self, bucket_idx: usize) -> usize {
        if self.config.pad_to_bucket {
            // Use the bucket boundary, or a sentinel for the overflow bucket.
            self.config
                .bucket_boundaries
                .get(bucket_idx)
                .copied()
                .unwrap_or_else(|| {
                    // Overflow: use the largest sequence currently queued.
                    self.buckets[bucket_idx]
                        .iter()
                        .map(|s| s.length)
                        .max()
                        .unwrap_or(1)
                })
        } else {
            // Dynamic: pad only to the max length in the queue.
            self.buckets[bucket_idx]
                .iter()
                .map(|s| s.length)
                .max()
                .unwrap_or(1)
        }
    }

    /// Build a [`PaddedBatch`] from a collected set of items, computing the
    /// padding ratio, and updating internal statistics.
    fn make_batch(
        &mut self,
        items: Vec<SequenceItem>,
        padded_length: usize,
        bucket_index: usize,
    ) -> PaddedBatch {
        let n = items.len();
        let total_allocated = n * padded_length;
        let total_real: usize = items.iter().map(|s| s.length).sum();
        let total_padding = total_allocated.saturating_sub(total_real);
        let padding_ratio = if total_allocated == 0 {
            0.0
        } else {
            total_padding as f64 / total_allocated as f64
        };

        self.total_batches += 1;
        self.sum_padding_ratio += padding_ratio;

        PaddedBatch {
            items,
            padded_length,
            padding_ratio,
            bucket_index,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_item(id: u64, length: usize) -> SequenceItem {
        SequenceItem {
            id,
            length,
            data: vec![0.0f32; length],
        }
    }

    fn default_batcher() -> VariableLengthBatcher {
        VariableLengthBatcher::new(VariableLengthBatchConfig::default())
    }

    // 1. Empty batcher returns None from next_batch
    #[test]
    fn test_empty_batcher_next_batch_is_none() {
        let mut batcher = default_batcher();
        assert!(batcher.next_batch().is_none());
    }

    // 2. flush_all on empty batcher returns empty vec
    #[test]
    fn test_empty_batcher_flush_all_is_empty() {
        let mut batcher = default_batcher();
        assert!(batcher.flush_all().is_empty());
    }

    // 3. Bucket assignment for sequence <= first boundary
    #[test]
    fn test_bucket_assignment_first_bucket() {
        let mut batcher = default_batcher();
        batcher.add(make_item(1, 16));
        let stats = batcher.stats();
        assert_eq!(stats.bucket_counts[0], 1);
    }

    // 4. Bucket assignment for sequence exactly at boundary
    #[test]
    fn test_bucket_assignment_at_boundary() {
        let mut batcher = default_batcher();
        batcher.add(make_item(1, 32)); // first boundary is 32
        let stats = batcher.stats();
        assert_eq!(stats.bucket_counts[0], 1, "length 32 belongs to bucket 0");
    }

    // 5. Bucket assignment for sequence just above first boundary
    #[test]
    fn test_bucket_assignment_above_first_boundary() {
        let mut batcher = default_batcher();
        batcher.add(make_item(1, 33)); // just above 32, should land in bucket 1 (<=64)
        let stats = batcher.stats();
        assert_eq!(stats.bucket_counts[1], 1);
    }

    // 6. Overflow bucket for sequence beyond all boundaries
    #[test]
    fn test_overflow_bucket_for_very_long_sequence() {
        let mut batcher = default_batcher();
        batcher.add(make_item(1, 1024)); // beyond 512
        let stats = batcher.stats();
        let overflow_idx = batcher.config.bucket_boundaries.len();
        assert_eq!(stats.bucket_counts[overflow_idx], 1);
    }

    // 7. next_batch emits full batch when limit is reached
    #[test]
    fn test_next_batch_emits_full_batch() {
        let config = VariableLengthBatchConfig {
            bucket_boundaries: vec![32],
            max_batch_tokens: 32 * 4, // room for 4 sequences of length 32
            max_batch_size: 4,
            pad_to_bucket: true,
            drop_last: false,
        };
        let mut batcher = VariableLengthBatcher::new(config);
        for i in 0..4 {
            batcher.add(make_item(i, 20)); // length 20, bucket 0 (<=32)
        }
        let batch = batcher.next_batch().expect("should emit batch");
        assert_eq!(batch.items.len(), 4);
        assert_eq!(batch.padded_length, 32); // pad_to_bucket => bucket boundary
        assert_eq!(batch.bucket_index, 0);
    }

    // 8. padding_ratio calculation
    #[test]
    fn test_padding_ratio_calculation() {
        let config = VariableLengthBatchConfig {
            bucket_boundaries: vec![32],
            max_batch_tokens: 32 * 2,
            max_batch_size: 2,
            pad_to_bucket: true,
            drop_last: false,
        };
        let mut batcher = VariableLengthBatcher::new(config);
        // Two sequences of length 16; padded to 32 each => 2*16 padding out of 2*32 total
        batcher.add(make_item(1, 16));
        batcher.add(make_item(2, 16));
        let batch = batcher.next_batch().expect("batch");
        // padding = (32-16)*2 = 32; total = 32*2 = 64; ratio = 0.5
        let expected = 0.5_f64;
        assert!(
            (batch.padding_ratio - expected).abs() < 1e-9,
            "expected {expected}, got {}",
            batch.padding_ratio
        );
    }

    // 9. max_batch_tokens constrains batch size
    #[test]
    fn test_max_batch_tokens_constrains_size() {
        let config = VariableLengthBatchConfig {
            bucket_boundaries: vec![64],
            max_batch_tokens: 128, // room for only 2 sequences padded to 64
            max_batch_size: 10,
            pad_to_bucket: true,
            drop_last: false,
        };
        let mut batcher = VariableLengthBatcher::new(config);
        for i in 0..10 {
            batcher.add(make_item(i, 50));
        }
        let batch = batcher.next_batch().expect("batch");
        assert!(
            batch.items.len() <= 2,
            "expected at most 2 items, got {}",
            batch.items.len()
        );
    }

    // 10. flush_all emits all partial batches when drop_last=false
    #[test]
    fn test_flush_all_emits_partial_batches() {
        let config = VariableLengthBatchConfig {
            bucket_boundaries: vec![32],
            max_batch_tokens: 32 * 5,
            max_batch_size: 5,
            pad_to_bucket: true,
            drop_last: false,
        };
        let mut batcher = VariableLengthBatcher::new(config);
        // Only 3 sequences; not enough for a full batch of 5.
        for i in 0..3 {
            batcher.add(make_item(i, 20));
        }
        let batches = batcher.flush_all();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].items.len(), 3);
    }

    // 11. flush_all drops partial batch when drop_last=true
    #[test]
    fn test_flush_all_drops_partial_when_drop_last() {
        let config = VariableLengthBatchConfig {
            bucket_boundaries: vec![32],
            max_batch_tokens: 32 * 5,
            max_batch_size: 5,
            pad_to_bucket: true,
            drop_last: true,
        };
        let mut batcher = VariableLengthBatcher::new(config);
        for i in 0..3 {
            batcher.add(make_item(i, 20));
        }
        let batches = batcher.flush_all();
        assert!(batches.is_empty(), "drop_last should discard partial batch");
    }

    // 12. stats total_items and total_batches are correct
    #[test]
    fn test_stats_total_items_and_batches() {
        let config = VariableLengthBatchConfig {
            bucket_boundaries: vec![32],
            max_batch_tokens: 32 * 2,
            max_batch_size: 2,
            pad_to_bucket: true,
            drop_last: false,
        };
        let mut batcher = VariableLengthBatcher::new(config);
        for i in 0..4 {
            batcher.add(make_item(i, 20));
        }
        batcher.next_batch();
        batcher.next_batch();
        let stats = batcher.stats();
        assert_eq!(stats.total_items, 4);
        assert_eq!(stats.total_batches, 2);
    }

    // 13. avg_padding_ratio is updated correctly
    #[test]
    fn test_avg_padding_ratio_after_batches() {
        let config = VariableLengthBatchConfig {
            bucket_boundaries: vec![32],
            max_batch_tokens: 32 * 2,
            max_batch_size: 2,
            pad_to_bucket: true,
            drop_last: false,
        };
        let mut batcher = VariableLengthBatcher::new(config);
        for i in 0..2 {
            batcher.add(make_item(i, 16)); // ratio 0.5 each
        }
        batcher.next_batch();
        let stats = batcher.stats();
        assert!(
            (stats.avg_padding_ratio - 0.5).abs() < 1e-9,
            "expected 0.5, got {}",
            stats.avg_padding_ratio
        );
    }

    // 14. pad_to_bucket=false pads to max item length in batch
    #[test]
    fn test_pad_to_max_item_length_when_not_pad_to_bucket() {
        let config = VariableLengthBatchConfig {
            bucket_boundaries: vec![64],
            max_batch_tokens: 64 * 3,
            max_batch_size: 3,
            pad_to_bucket: false,
            drop_last: false,
        };
        let mut batcher = VariableLengthBatcher::new(config);
        batcher.add(make_item(1, 10));
        batcher.add(make_item(2, 25));
        batcher.add(make_item(3, 40)); // max in this bucket
        let batch = batcher.next_batch().expect("batch");
        // With pad_to_bucket=false padded_length should be 40 (the max in queue).
        assert_eq!(batch.padded_length, 40);
    }

    // 15. bucket_counts reflect items in flight
    #[test]
    fn test_bucket_counts_reflect_in_flight_items() {
        let mut batcher = default_batcher();
        batcher.add(make_item(1, 10));  // bucket 0
        batcher.add(make_item(2, 50));  // bucket 1
        batcher.add(make_item(3, 100)); // bucket 2
        let stats = batcher.stats();
        assert_eq!(stats.bucket_counts[0], 1);
        assert_eq!(stats.bucket_counts[1], 1);
        assert_eq!(stats.bucket_counts[2], 1);
    }

    // 16. sequences are assigned to correct buckets across all boundaries
    #[test]
    fn test_all_bucket_boundaries_assigned_correctly() {
        let boundaries = vec![32usize, 64, 128, 256, 512];
        let test_lengths: Vec<(usize, usize)> = vec![
            (1, 0),   // length 1 -> bucket 0 (<=32)
            (32, 0),  // length 32 -> bucket 0 (<=32)
            (33, 1),  // length 33 -> bucket 1 (<=64)
            (64, 1),  // length 64 -> bucket 1 (<=64)
            (65, 2),  // bucket 2 (<=128)
            (128, 2), // bucket 2
            (129, 3), // bucket 3 (<=256)
            (256, 3), // bucket 3
            (257, 4), // bucket 4 (<=512)
            (512, 4), // bucket 4
            (513, 5), // overflow
        ];
        let config = VariableLengthBatchConfig {
            bucket_boundaries: boundaries,
            ..Default::default()
        };
        let batcher = VariableLengthBatcher::new(config);
        for (length, expected_bucket) in test_lengths {
            let got = batcher.bucket_index_for(length);
            assert_eq!(
                got, expected_bucket,
                "length {length} should map to bucket {expected_bucket}, got {got}"
            );
        }
    }
}
