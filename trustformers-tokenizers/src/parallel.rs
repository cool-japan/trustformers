use scirs2_core::parallel_ops::*; // SciRS2 Integration Policy - replaces rayon
use std::sync::Arc;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// Parallel batch tokenization utilities for improved throughput
pub struct ParallelTokenizer<T: Tokenizer + Sync> {
    tokenizer: Arc<T>,
    chunk_size: usize,
}

impl<T: Tokenizer + Sync> ParallelTokenizer<T> {
    /// Create a new parallel tokenizer wrapper
    pub fn new(tokenizer: T) -> Self {
        Self {
            tokenizer: Arc::new(tokenizer),
            chunk_size: 1000, // Default chunk size
        }
    }

    /// Create a new parallel tokenizer wrapper with custom chunk size
    pub fn with_chunk_size(tokenizer: T, chunk_size: usize) -> Self {
        Self {
            tokenizer: Arc::new(tokenizer),
            chunk_size,
        }
    }

    /// Encode a batch of texts in parallel
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<TokenizedInput>> {
        texts
            .par_chunks(self.chunk_size)
            .map(|chunk| {
                chunk.iter().map(|text| self.tokenizer.encode(text)).collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<Vec<_>>>>()
            .map(|batches| batches.into_iter().flatten().collect())
    }

    /// Encode pairs of texts in parallel
    pub fn encode_pair_batch(&self, text_pairs: &[(&str, &str)]) -> Result<Vec<TokenizedInput>> {
        text_pairs
            .par_chunks(self.chunk_size)
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|(text1, text2)| self.tokenizer.encode_pair(text1, text2))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<Vec<_>>>>()
            .map(|batches| batches.into_iter().flatten().collect())
    }

    /// Decode a batch of token IDs in parallel
    pub fn decode_batch(&self, ids_batch: &[&[u32]]) -> Result<Vec<String>> {
        ids_batch
            .par_chunks(self.chunk_size)
            .map(|chunk| {
                chunk.iter().map(|ids| self.tokenizer.decode(ids)).collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<Vec<_>>>>()
            .map(|batches| batches.into_iter().flatten().collect())
    }

    /// Get the underlying tokenizer
    pub fn tokenizer(&self) -> &T {
        &self.tokenizer
    }

    /// Get the chunk size
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Set the chunk size for batching
    pub fn set_chunk_size(&mut self, chunk_size: usize) {
        self.chunk_size = chunk_size;
    }
}

/// HuggingFace-style truncation strategy for sequence-pair batches (mirrors
/// `tokenizers`' `TruncationStrategy`). Truncation always removes tokens
/// from the *end* of whichever segment(s) it targets, and never removes a
/// leading/trailing run of special tokens identified by
/// [`TokenizedInput::special_tokens_mask`] (when the wrapped tokenizer
/// populates it) -- see [`BatchTokenizer::encode_batch_padded`] and
/// [`BatchTokenizer::encode_pair_batch_padded`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TruncationStrategy {
    /// Alternately truncate whichever of the two sequences is currently
    /// longer, one token at a time, until both fit. For a single
    /// (non-pair) input this behaves identically to `OnlyFirst` (there is
    /// only one sequence to truncate).
    #[default]
    LongestFirst,
    /// Only ever truncate the first sequence. Errors if truncating it
    /// alone cannot make the pair fit within `max_length`.
    OnlyFirst,
    /// Only ever truncate the second sequence. Errors on a single
    /// (non-pair) input, which has no second sequence to truncate.
    OnlySecond,
}

/// Which side padding tokens are added to (mirrors HF's `padding_side`).
/// Encoder models conventionally pad on the right (this crate's original,
/// and still default, behavior); decoder-only models generally require
/// left padding so the *last* real token of every sequence in a batch
/// lines up for next-token generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PaddingSide {
    #[default]
    Right,
    Left,
}

/// A real, usable special-tokens mask for `input` (its own, if present and
/// the right length; otherwise an all-zero mask, meaning "nothing is
/// protected" -- the honest answer when the wrapped tokenizer does not
/// populate this field).
fn usable_special_mask(input: &TokenizedInput, len: usize) -> Vec<u8> {
    match &input.special_tokens_mask {
        Some(m) if m.len() == len => m.clone(),
        _ => vec![0u8; len],
    }
}

/// Remove every position where `keep[i]` is `false` from every present
/// per-token field of `input` (`input_ids`, `attention_mask`, and, when
/// their length currently matches `input_ids`,
/// `token_type_ids`/`special_tokens_mask`/`offset_mapping`), keeping every
/// field in lockstep. `keep.len()` must equal `input.input_ids.len()`.
fn apply_keep_mask(input: &mut TokenizedInput, keep: &[bool]) {
    let len = input.input_ids.len();
    debug_assert_eq!(keep.len(), len, "keep mask must cover every position");

    retain_by_mask(&mut input.input_ids, keep);
    retain_by_mask(&mut input.attention_mask, keep);
    if let Some(v) = input.token_type_ids.as_mut() {
        if v.len() == len {
            retain_by_mask(v, keep);
        }
    }
    if let Some(v) = input.special_tokens_mask.as_mut() {
        if v.len() == len {
            retain_by_mask(v, keep);
        }
    }
    if let Some(v) = input.offset_mapping.as_mut() {
        if v.len() == len {
            retain_by_mask(v, keep);
        }
    }
}

/// Keep only the elements of `vec` at positions where `keep[i]` is `true`.
fn retain_by_mask<V>(vec: &mut Vec<V>, keep: &[bool]) {
    let mut i = 0usize;
    vec.retain(|_| {
        let k = keep[i];
        i += 1;
        k
    });
}

/// Truncate a single-sequence `TokenizedInput` to `max_len`, never
/// removing a position its `special_tokens_mask` marks as special
/// (`[CLS]`/`[SEP]`-style wrapper tokens, wherever they occur), and
/// returns the dropped content tokens (with up to `stride` tokens of
/// overlap from the kept portion prepended, mirroring HF's `stride`
/// semantics) when truncation occurred.
///
/// Without a usable `special_tokens_mask` this degrades to plain tail
/// truncation (the previous behavior) -- there is no way to protect
/// tokens whose special/regular status is unknown.
fn truncate_single_preserving_special_tokens(
    input: &mut TokenizedInput,
    max_len: usize,
    stride: usize,
) -> Option<Vec<u32>> {
    let len = input.input_ids.len();
    if len <= max_len {
        return None;
    }

    let mask = usable_special_mask(input, len);
    // Every removable (non-special) position, in original index order.
    let removable: Vec<usize> = (0..len).filter(|&i| mask[i] == 0).collect();
    let protected_count = len - removable.len();

    let content_budget = max_len.saturating_sub(protected_count);
    let kept_content_len = content_budget.min(removable.len());
    if kept_content_len == removable.len() {
        return None; // Protected positions alone already meet max_len.
    }

    // Drop the *last* `removable.len() - kept_content_len` removable
    // positions (tail truncation of the content, matching HF's default).
    let dropped_positions = &removable[kept_content_len..];

    // Stride overlap: the last `stride` *kept* removable positions, plus
    // every dropped position, in original index order.
    let stride_actual = stride.min(kept_content_len);
    let overflow_positions = &removable[kept_content_len - stride_actual..];
    let overflow: Vec<u32> = overflow_positions.iter().map(|&i| input.input_ids[i]).collect();

    let drop_set: std::collections::HashSet<usize> = dropped_positions.iter().copied().collect();
    let keep: Vec<bool> = (0..len).map(|i| !drop_set.contains(&i)).collect();
    apply_keep_mask(input, &keep);

    Some(overflow)
}

/// Truncate a sequence-pair's flattened `TokenizedInput` (as produced by
/// `Tokenizer::encode_pair`) to `max_len`, per `strategy`, never removing
/// a position its `special_tokens_mask` marks as special -- including a
/// separator sitting *between* the two segments, not just the outermost
/// wrapper tokens. Returns the dropped tokens (segment 1's drop followed
/// by segment 2's drop, each in original order) when truncation occurred.
///
/// Segment assignment for each *removable* (non-special) position comes
/// from `TokenizedInput::token_type_ids` (0 or 1), matching every
/// `encode_pair` implementation in this crate. Without usable
/// `token_type_ids`, every removable position is treated as segment 0 --
/// `LongestFirst` and `OnlyFirst` then behave identically (there is no
/// positional information to find a second segment at all).
///
/// Unlike the single-sequence path, the returned overflow does not
/// include `stride`-based overlap from the kept portion: with two
/// independently-truncatable segments there is no single unambiguous
/// "preceding window" to draw the overlap from, so this reports the raw
/// dropped tokens only.
fn truncate_pair_preserving_special_tokens(
    input: &mut TokenizedInput,
    max_len: usize,
    strategy: TruncationStrategy,
) -> Result<Option<Vec<u32>>> {
    let len = input.input_ids.len();
    if len <= max_len {
        return Ok(None);
    }

    let special_mask = usable_special_mask(input, len);
    // A same-length `token_type_ids` gives each removable position's real
    // segment (0 or 1); otherwise every removable position is segment 0
    // (see this function's docs).
    let token_types: Vec<u32> = match &input.token_type_ids {
        Some(tti) if tti.len() == len => tti.clone(),
        _ => vec![0u32; len],
    };

    let mut seg_a: Vec<usize> = Vec::new();
    let mut seg_b: Vec<usize> = Vec::new();
    for (i, &is_special) in special_mask.iter().enumerate() {
        if is_special == 1 {
            continue;
        }
        if token_types[i] == 0 {
            seg_a.push(i);
        } else {
            seg_b.push(i);
        }
    }

    let protected_count = len - seg_a.len() - seg_b.len();
    let content_len = seg_a.len() + seg_b.len();
    let content_budget = max_len.saturating_sub(protected_count);
    if content_budget >= content_len {
        return Ok(None);
    }
    let to_remove = content_len - content_budget;

    match strategy {
        TruncationStrategy::OnlySecond if seg_b.is_empty() => {
            return Err(TrustformersError::invalid_config(
                "TruncationStrategy::OnlySecond requires a second sequence (segment-1 \
                 token_type_ids); this input has only one segment"
                    .to_string(),
            ));
        },
        TruncationStrategy::OnlyFirst if to_remove > seg_a.len() => {
            return Err(TrustformersError::invalid_config(format!(
                "TruncationStrategy::OnlyFirst cannot remove {} tokens from a first \
                 sequence of only {} content tokens; the pair does not fit within \
                 max_length={} without truncating the second sequence",
                to_remove,
                seg_a.len(),
                max_len
            )));
        },
        TruncationStrategy::OnlySecond if to_remove > seg_b.len() => {
            return Err(TrustformersError::invalid_config(format!(
                "TruncationStrategy::OnlySecond cannot remove {} tokens from a second \
                 sequence of only {} content tokens; the pair does not fit within \
                 max_length={} without truncating the first sequence",
                to_remove,
                seg_b.len(),
                max_len
            )));
        },
        _ => {},
    }

    let mut remaining_a = seg_a.len();
    let mut remaining_b = seg_b.len();
    for _ in 0..to_remove {
        match strategy {
            TruncationStrategy::OnlyFirst => remaining_a -= 1,
            TruncationStrategy::OnlySecond => remaining_b -= 1,
            TruncationStrategy::LongestFirst => {
                if remaining_a >= remaining_b {
                    remaining_a -= 1;
                } else {
                    remaining_b -= 1;
                }
            },
        }
    }

    // Keep the first `remaining_*` (by index order) removable positions of
    // each segment; drop the rest (tail truncation within each segment).
    let dropped_a = &seg_a[remaining_a..];
    let dropped_b = &seg_b[remaining_b..];

    let mut dropped: Vec<u32> = dropped_a.iter().map(|&i| input.input_ids[i]).collect();
    dropped.extend(dropped_b.iter().map(|&i| input.input_ids[i]));

    let drop_set: std::collections::HashSet<usize> =
        dropped_a.iter().chain(dropped_b.iter()).copied().collect();
    let keep: Vec<bool> = (0..len).map(|i| !drop_set.contains(&i)).collect();
    apply_keep_mask(input, &keep);

    Ok(Some(dropped))
}

/// Batch tokenization with padding and truncation support
#[derive(Debug, Clone)]
pub struct BatchTokenizer<T: Tokenizer + Sync> {
    tokenizer: Arc<T>,
    max_length: Option<usize>,
    padding: bool,
    truncation: bool,
    pad_token_id: u32,
    truncation_strategy: TruncationStrategy,
    padding_side: PaddingSide,
    /// Number of tokens of overlap `encode_batch_padded` carries into
    /// `overflowing_tokens`. Unused by `encode_pair_batch_padded` -- see
    /// [`truncate_pair_preserving_special_tokens`].
    stride: usize,
}

impl<T: Tokenizer + Sync> BatchTokenizer<T> {
    /// Create a new batch tokenizer
    pub fn new(tokenizer: T) -> Self {
        Self {
            tokenizer: Arc::new(tokenizer),
            max_length: None,
            padding: false,
            truncation: false,
            pad_token_id: 0, // Default pad token ID
            truncation_strategy: TruncationStrategy::default(),
            padding_side: PaddingSide::default(),
            stride: 0,
        }
    }

    /// Set the maximum sequence length
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = Some(max_length);
        self
    }

    /// Enable padding to max length
    pub fn with_padding(mut self, pad_token_id: u32) -> Self {
        self.padding = true;
        self.pad_token_id = pad_token_id;
        self
    }

    /// Enable truncation to max length
    pub fn with_truncation(mut self) -> Self {
        self.truncation = true;
        self
    }

    /// Select which sequence(s) `encode_pair_batch_padded` truncates
    /// (default [`TruncationStrategy::LongestFirst`]).
    pub fn with_truncation_strategy(mut self, strategy: TruncationStrategy) -> Self {
        self.truncation_strategy = strategy;
        self
    }

    /// Select which side padding tokens are added to (default
    /// [`PaddingSide::Right`]).
    pub fn with_padding_side(mut self, side: PaddingSide) -> Self {
        self.padding_side = side;
        self
    }

    /// Set the number of tokens of overlap carried from the kept portion
    /// into `overflowing_tokens` when `encode_batch_padded` truncates
    /// (default `0`, matching the previous no-overflow behavior).
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Encode a batch with padding and truncation.
    ///
    /// Truncation (when enabled) preserves any leading/trailing special
    /// tokens the wrapped tokenizer's `special_tokens_mask` identifies,
    /// and records the tokens it drops (with `stride` tokens of overlap)
    /// in the returned batch's `overflowing_tokens`.
    pub fn encode_batch_padded(&self, texts: &[&str]) -> Result<BatchedTokenizedInput> {
        if self.truncation && self.truncation_strategy == TruncationStrategy::OnlySecond {
            return Err(TrustformersError::invalid_config(
                "TruncationStrategy::OnlySecond requires sequence-pair input; use \
                 `encode_pair_batch_padded` instead of `encode_batch_padded`"
                    .to_string(),
            ));
        }

        // First, encode all texts in parallel
        let mut processed: Vec<TokenizedInput> = texts
            .par_iter()
            .map(|text| self.tokenizer.encode(text))
            .collect::<Result<Vec<_>>>()?;

        let mut overflow: Vec<Option<Vec<u32>>> = vec![None; processed.len()];
        if let (true, Some(max_len)) = (self.truncation, self.max_length) {
            for (input, slot) in processed.iter_mut().zip(overflow.iter_mut()) {
                *slot = truncate_single_preserving_special_tokens(input, max_len, self.stride);
            }
        }

        self.apply_padding_to_batch(&mut processed);

        Ok(BatchedTokenizedInput::from_parts(processed, overflow))
    }

    /// Encode a batch of sequence pairs with padding and truncation,
    /// honoring [`Self::with_truncation_strategy`] to choose which
    /// sequence(s) truncation removes tokens from.
    pub fn encode_pair_batch_padded(
        &self,
        pairs: &[(&str, &str)],
    ) -> Result<BatchedTokenizedInput> {
        let mut processed: Vec<TokenizedInput> = pairs
            .par_iter()
            .map(|(a, b)| self.tokenizer.encode_pair(a, b))
            .collect::<Result<Vec<_>>>()?;

        let mut overflow: Vec<Option<Vec<u32>>> = vec![None; processed.len()];
        if let (true, Some(max_len)) = (self.truncation, self.max_length) {
            for (input, slot) in processed.iter_mut().zip(overflow.iter_mut()) {
                *slot = truncate_pair_preserving_special_tokens(
                    input,
                    max_len,
                    self.truncation_strategy,
                )?;
            }
        }

        self.apply_padding_to_batch(&mut processed);

        Ok(BatchedTokenizedInput::from_parts(processed, overflow))
    }

    /// Pad every sequence in `processed` to a single, rectangular target
    /// length, per [`Self::padding_side`].
    ///
    /// The target is `max(self.max_length, the longest sequence actually
    /// present)`: if truncation is disabled (or `max_length` is unset) and
    /// some sequence naturally exceeds `max_length`, the target grows to
    /// accommodate it rather than leaving that one sequence unpadded next
    /// to shorter, padded ones -- a batch whose rows have different
    /// lengths cannot be stacked into a tensor, which defeats the purpose
    /// of padding at all.
    fn apply_padding_to_batch(&self, processed: &mut [TokenizedInput]) {
        if !self.padding {
            return;
        }

        let longest_actual = processed.iter().map(|input| input.input_ids.len()).max().unwrap_or(0);
        let target_len = match self.max_length {
            Some(max_len) => max_len.max(longest_actual),
            None => longest_actual,
        };

        for input in processed.iter_mut() {
            let current_len = input.input_ids.len();
            if current_len >= target_len {
                continue;
            }
            let pad_len = target_len - current_len;
            match self.padding_side {
                PaddingSide::Right => {
                    input.input_ids.extend(std::iter::repeat_n(self.pad_token_id, pad_len));
                    input.attention_mask.extend(std::iter::repeat_n(0u8, pad_len));
                    if let Some(type_ids) = input.token_type_ids.as_mut() {
                        type_ids.extend(std::iter::repeat_n(0u32, pad_len));
                    }
                    if let Some(mask) = input.special_tokens_mask.as_mut() {
                        mask.extend(std::iter::repeat_n(0u8, pad_len));
                    }
                    if let Some(offsets) = input.offset_mapping.as_mut() {
                        offsets.extend(std::iter::repeat_n((0, 0), pad_len));
                    }
                },
                PaddingSide::Left => {
                    prepend(&mut input.input_ids, self.pad_token_id, pad_len);
                    prepend(&mut input.attention_mask, 0u8, pad_len);
                    if let Some(type_ids) = input.token_type_ids.as_mut() {
                        prepend(type_ids, 0u32, pad_len);
                    }
                    if let Some(mask) = input.special_tokens_mask.as_mut() {
                        prepend(mask, 0u8, pad_len);
                    }
                    if let Some(offsets) = input.offset_mapping.as_mut() {
                        prepend(offsets, (0, 0), pad_len);
                    }
                },
            }
        }
    }

    /// Get the underlying tokenizer
    pub fn tokenizer(&self) -> &T {
        &self.tokenizer
    }
}

/// Prepend `count` copies of `value` to the front of `vec`, in place.
fn prepend<V: Clone>(vec: &mut Vec<V>, value: V, count: usize) {
    let mut new_vec = Vec::with_capacity(count + vec.len());
    new_vec.extend(std::iter::repeat_n(value, count));
    new_vec.append(vec);
    *vec = new_vec;
}

/// Batched tokenized input with convenient access methods
#[derive(Debug, Clone)]
pub struct BatchedTokenizedInput {
    pub input_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u8>>,
    pub token_type_ids: Option<Vec<Vec<u32>>>,
    /// Present whenever at least one row's source `TokenizedInput` carried
    /// a `special_tokens_mask`; rows without one contribute an empty
    /// `Vec` for that position.
    pub special_tokens_mask: Option<Vec<Vec<u8>>>,
    /// Present whenever at least one row's source `TokenizedInput` carried
    /// an `offset_mapping`; rows without one contribute an empty `Vec` for
    /// that position.
    pub offset_mapping: Option<Vec<Vec<(usize, usize)>>>,
    /// Present whenever truncation actually dropped tokens from at least
    /// one row; rows that were not truncated contribute an empty `Vec`.
    pub overflowing_tokens: Option<Vec<Vec<u32>>>,
}

impl BatchedTokenizedInput {
    /// Create from a batch of TokenizedInput
    pub fn from_batch(batch: Vec<TokenizedInput>) -> Self {
        let len = batch.len();
        Self::from_parts(batch, vec![None; len])
    }

    /// Create from a batch of `TokenizedInput` plus, for each row, the
    /// overflowing tokens truncation dropped from it (if any).
    fn from_parts(batch: Vec<TokenizedInput>, overflow: Vec<Option<Vec<u32>>>) -> Self {
        let mut input_ids = Vec::with_capacity(batch.len());
        let mut attention_mask = Vec::with_capacity(batch.len());
        let mut token_type_ids = Vec::with_capacity(batch.len());
        let mut special_tokens_mask = Vec::with_capacity(batch.len());
        let mut offset_mapping = Vec::with_capacity(batch.len());
        let mut overflowing_tokens = Vec::with_capacity(batch.len());

        let has_token_type_ids = batch.iter().any(|input| input.token_type_ids.is_some());
        let has_special_tokens_mask = batch.iter().any(|input| input.special_tokens_mask.is_some());
        let has_offset_mapping = batch.iter().any(|input| input.offset_mapping.is_some());
        let has_overflow = overflow.iter().any(|o| o.is_some());

        for (input, overflow_item) in batch.into_iter().zip(overflow) {
            input_ids.push(input.input_ids);
            attention_mask.push(input.attention_mask);
            if has_token_type_ids {
                token_type_ids.push(input.token_type_ids.unwrap_or_default());
            }
            if has_special_tokens_mask {
                special_tokens_mask.push(input.special_tokens_mask.unwrap_or_default());
            }
            if has_offset_mapping {
                offset_mapping.push(input.offset_mapping.unwrap_or_default());
            }
            if has_overflow {
                overflowing_tokens.push(overflow_item.unwrap_or_default());
            }
        }

        Self {
            input_ids,
            attention_mask,
            token_type_ids: if has_token_type_ids { Some(token_type_ids) } else { None },
            special_tokens_mask: if has_special_tokens_mask {
                Some(special_tokens_mask)
            } else {
                None
            },
            offset_mapping: if has_offset_mapping { Some(offset_mapping) } else { None },
            overflowing_tokens: if has_overflow { Some(overflowing_tokens) } else { None },
        }
    }

    /// Get the batch size
    pub fn batch_size(&self) -> usize {
        self.input_ids.len()
    }

    /// Get the sequence length for each sample
    pub fn sequence_lengths(&self) -> Vec<usize> {
        self.input_ids.iter().map(|ids| ids.len()).collect()
    }

    /// Convert to individual TokenizedInput items
    pub fn to_individual(self) -> Vec<TokenizedInput> {
        let mut result = Vec::with_capacity(self.input_ids.len());

        for i in 0..self.input_ids.len() {
            let token_type_ids = self.token_type_ids.as_ref().map(|types| types[i].clone());
            let special_tokens_mask =
                self.special_tokens_mask.as_ref().map(|masks| masks[i].clone());
            let offset_mapping = self.offset_mapping.as_ref().map(|offsets| offsets[i].clone());
            let overflowing_tokens =
                self.overflowing_tokens.as_ref().map(|overflow| overflow[i].clone());

            result.push(TokenizedInput {
                input_ids: self.input_ids[i].clone(),
                attention_mask: self.attention_mask[i].clone(),
                token_type_ids,
                special_tokens_mask,
                offset_mapping,
                overflowing_tokens,
            });
        }

        result
    }

    /// Get input IDs as a flat tensor-like structure
    pub fn input_ids_tensor(&self) -> &Vec<Vec<u32>> {
        &self.input_ids
    }

    /// Get attention mask as a flat tensor-like structure
    pub fn attention_mask_tensor(&self) -> &Vec<Vec<u8>> {
        &self.attention_mask
    }

    /// Get token type IDs as a flat tensor-like structure
    pub fn token_type_ids_tensor(&self) -> Option<&Vec<Vec<u32>>> {
        self.token_type_ids.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::char::CharTokenizer;
    use std::collections::HashMap;

    fn create_test_tokenizer() -> CharTokenizer {
        let mut vocab = HashMap::new();
        vocab.insert("a".to_string(), 0);
        vocab.insert("b".to_string(), 1);
        vocab.insert("c".to_string(), 2);
        vocab.insert(" ".to_string(), 3);
        vocab.insert("[UNK]".to_string(), 4);
        vocab.insert("[PAD]".to_string(), 5);
        vocab.insert("[CLS]".to_string(), 6);
        vocab.insert("[SEP]".to_string(), 7);
        CharTokenizer::new(vocab)
    }

    #[test]
    fn test_parallel_tokenizer() {
        let tokenizer = create_test_tokenizer();
        let parallel_tokenizer = ParallelTokenizer::new(tokenizer);

        let texts = vec!["hello world", "goodbye world", "test text"];
        let results = parallel_tokenizer.encode_batch(&texts).expect("Operation failed in test");

        assert_eq!(results.len(), 3);
        for result in results {
            assert!(!result.input_ids.is_empty());
            assert!(!result.attention_mask.is_empty());
        }
    }

    #[test]
    fn test_parallel_encode_pairs() {
        let tokenizer = create_test_tokenizer();
        let parallel_tokenizer = ParallelTokenizer::new(tokenizer);

        let pairs = vec![("hello", "world"), ("good", "bye"), ("test", "text")];
        let results =
            parallel_tokenizer.encode_pair_batch(&pairs).expect("Operation failed in test");

        assert_eq!(results.len(), 3);
        for result in results {
            assert!(!result.input_ids.is_empty());
            assert!(!result.attention_mask.is_empty());
        }
    }

    #[test]
    fn test_batch_tokenizer_with_padding() {
        let tokenizer = create_test_tokenizer();
        let batch_tokenizer = BatchTokenizer::new(tokenizer)
            .with_max_length(10)
            .with_padding(0)
            .with_truncation();

        let texts = vec!["short", "this is a longer text", "medium"];
        let result = batch_tokenizer.encode_batch_padded(&texts).expect("Operation failed in test");

        assert_eq!(result.batch_size(), 3);

        // All sequences should have the same length (10) due to padding/truncation
        for seq_len in result.sequence_lengths() {
            assert_eq!(seq_len, 10);
        }
    }

    #[test]
    fn test_batched_tokenized_input() {
        let input1 = TokenizedInput {
            input_ids: vec![1, 2, 3],
            attention_mask: vec![1, 1, 1],
            token_type_ids: Some(vec![0, 0, 0]),
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        };
        let input2 = TokenizedInput {
            input_ids: vec![4, 5],
            attention_mask: vec![1, 1],
            token_type_ids: Some(vec![1, 1]),
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        };

        let batched = BatchedTokenizedInput::from_batch(vec![input1, input2]);

        assert_eq!(batched.batch_size(), 2);
        assert_eq!(batched.sequence_lengths(), vec![3, 2]);
        assert!(batched.token_type_ids.is_some());

        // Test conversion back to individual
        let individual = batched.to_individual();
        assert_eq!(individual.len(), 2);
        assert_eq!(individual[0].input_ids, vec![1, 2, 3]);
        assert_eq!(individual[1].input_ids, vec![4, 5]);
    }

    #[test]
    fn test_parallel_decode_batch() {
        let tokenizer = create_test_tokenizer();
        let parallel_tokenizer = ParallelTokenizer::new(tokenizer);

        let ids1 = vec![0, 1, 2]; // a, b, c
        let ids2 = vec![3, 0]; // space, a
        let ids_batch = vec![ids1.as_slice(), ids2.as_slice()];

        let results =
            parallel_tokenizer.decode_batch(&ids_batch).expect("Operation failed in test");
        assert_eq!(results.len(), 2);
        assert!(!results[0].is_empty());
        assert!(!results[1].is_empty());
    }

    #[test]
    fn test_chunk_size_configuration() {
        let tokenizer = create_test_tokenizer();
        let mut parallel_tokenizer = ParallelTokenizer::with_chunk_size(tokenizer, 500);

        assert_eq!(parallel_tokenizer.chunk_size(), 500);

        parallel_tokenizer.set_chunk_size(1000);
        assert_eq!(parallel_tokenizer.chunk_size(), 1000);
    }

    /// Hand-computed regression test for the destroyed-trailing-special-token
    /// bug: `[CLS] a b c d e [SEP]` (7 tokens) truncated to `max_length=4`
    /// used to become `[CLS] a b c` (`truncate(4)`), silently dropping the
    /// `[SEP]`. It must instead keep both the `[CLS]` and the `[SEP]` and
    /// truncate only the content in between.
    #[test]
    fn test_single_truncation_preserves_trailing_special_token() {
        let mut input = TokenizedInput {
            input_ids: vec![100, 1, 2, 3, 4, 5, 101], // [CLS] a b c d e [SEP]
            attention_mask: vec![1; 7],
            token_type_ids: None,
            special_tokens_mask: Some(vec![1, 0, 0, 0, 0, 0, 1]),
            offset_mapping: None,
            overflowing_tokens: None,
        };

        let dropped = truncate_single_preserving_special_tokens(&mut input, 4, 1);

        // leading=1 ([CLS]), trailing=1 ([SEP]), content_budget=4-2=2, so
        // the first 2 content tokens (a, b) are kept and c,d,e are
        // dropped: [CLS] a b [SEP].
        assert_eq!(input.input_ids, vec![100, 1, 2, 101]);
        assert_eq!(input.special_tokens_mask, Some(vec![1, 0, 0, 1]));
        // stride=1: the overflow includes the last kept content token (b)
        // plus every dropped token (c, d, e).
        assert_eq!(dropped, Some(vec![2, 3, 4, 5]));
    }

    /// Without a usable `special_tokens_mask`, truncation cannot know what
    /// to protect and must degrade to plain tail truncation -- this pins
    /// that documented fallback rather than silently doing nothing.
    #[test]
    fn test_single_truncation_without_mask_falls_back_to_tail_truncation() {
        let mut input = TokenizedInput {
            input_ids: vec![100, 1, 2, 3, 4, 5, 101],
            attention_mask: vec![1; 7],
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        };

        let dropped = truncate_single_preserving_special_tokens(&mut input, 4, 0);

        assert_eq!(input.input_ids, vec![100, 1, 2, 3]);
        assert_eq!(dropped, Some(vec![4, 5, 101]));
    }

    #[test]
    fn test_encode_batch_padded_reports_overflowing_tokens() {
        let tokenizer = create_test_tokenizer();
        let batch_tokenizer = BatchTokenizer::new(tokenizer).with_max_length(3).with_truncation();

        // "a b c" tokenizes (via CharTokenizer) to 5 characters: a, space,
        // b, space, c -- longer than max_length=3, with no special-tokens
        // mask (CharTokenizer's default bos/eos tokens are not in this
        // fixture's vocab), so this exercises the plain-truncation
        // fallback path end to end.
        let texts = vec!["a b c"];
        let result = batch_tokenizer.encode_batch_padded(&texts).expect("Operation failed in test");

        assert_eq!(result.input_ids[0].len(), 3);
        let overflow = result.overflowing_tokens.expect("truncation must report overflow");
        assert!(!overflow[0].is_empty());
    }

    /// Hand-computed regression test for `TruncationStrategy::OnlyFirst` on
    /// a pair: `[CLS] a b c [SEP] x y [SEP]` (8 tokens, segment 0 = a b c,
    /// segment 1 = x y) truncated to `max_length=6` must remove exactly 2
    /// tokens from segment 0 only, leaving segment 1 untouched.
    #[test]
    fn test_pair_truncation_only_first() {
        let mut input = TokenizedInput {
            input_ids: vec![100, 1, 2, 3, 101, 10, 11, 101], // CLS a b c SEP x y SEP
            attention_mask: vec![1; 8],
            token_type_ids: Some(vec![0, 0, 0, 0, 0, 1, 1, 1]),
            special_tokens_mask: Some(vec![1, 0, 0, 0, 1, 0, 0, 1]),
            offset_mapping: None,
            overflowing_tokens: None,
        };

        let dropped =
            truncate_pair_preserving_special_tokens(&mut input, 6, TruncationStrategy::OnlyFirst)
                .expect("Operation failed in test");

        // protected_count=3 (CLS, mid-SEP, trailing SEP), content_len=5
        // (a,b,c,x,y), content_budget = 6-3 = 3, to_remove = 5-3 = 2.
        // `OnlyFirst` removes both from segment 0 (a,b,c), dropping its
        // last 2 (b, c) and keeping only `a`; segment 1 (x, y) and both
        // SEPs are untouched.
        assert_eq!(input.input_ids, vec![100, 1, 101, 10, 11, 101]);
        assert_eq!(dropped, Some(vec![2, 3]));
    }

    /// The `LongestFirst` strategy alternates, shrinking whichever segment
    /// is currently longer, one token at a time, until the pair fits.
    #[test]
    fn test_pair_truncation_longest_first_balances_segments() {
        let mut input = TokenizedInput {
            // CLS a b c d SEP x SEP: 3 protected positions (CLS, the
            // mid-pair SEP, the trailing SEP); segment 0 (content) has 4
            // tokens (a b c d), segment 1 (content) has 1 (x). The mid-pair
            // SEP is marked special, so -- unlike its `token_type_ids`
            // value of `0` -- it is never counted as (or removable as)
            // segment-0 content.
            input_ids: vec![100, 1, 2, 3, 4, 101, 10, 101],
            attention_mask: vec![1; 8],
            token_type_ids: Some(vec![0, 0, 0, 0, 0, 0, 1, 1]),
            special_tokens_mask: Some(vec![1, 0, 0, 0, 0, 1, 0, 1]),
            offset_mapping: None,
            overflowing_tokens: None,
        };

        // protected_count=3, content_len=5 (4 + 1), content_budget =
        // max_length(4) - 3 = 1, to_remove = 5 - 1 = 4. Alternating
        // "shrink whichever is currently longer" starting from (4, 1)
        // removes all 4 tokens from segment 0 (it stays >= segment 1's
        // length at every step) and none from segment 1.
        let dropped = truncate_pair_preserving_special_tokens(
            &mut input,
            4,
            TruncationStrategy::LongestFirst,
        )
        .expect("Operation failed in test");

        assert_eq!(input.input_ids, vec![100, 101, 10, 101]); // CLS SEP-mid x SEP
        assert_eq!(dropped, Some(vec![1, 2, 3, 4])); // a b c d
    }

    /// `OnlySecond` on effectively-single-segment input (no second segment
    /// in `token_type_ids`) is a configuration error, not a silent no-op.
    #[test]
    fn test_pair_truncation_only_second_errors_without_second_segment() {
        let mut input = TokenizedInput {
            input_ids: vec![100, 1, 2, 3, 4, 5, 101],
            attention_mask: vec![1; 7],
            token_type_ids: Some(vec![0; 7]),
            special_tokens_mask: Some(vec![1, 0, 0, 0, 0, 0, 1]),
            offset_mapping: None,
            overflowing_tokens: None,
        };

        let result =
            truncate_pair_preserving_special_tokens(&mut input, 4, TruncationStrategy::OnlySecond);
        assert!(result.is_err());
    }

    /// `encode_batch_padded` (single-sequence) must reject `OnlySecond`
    /// rather than silently truncating the one sequence it has anyway.
    #[test]
    fn test_encode_batch_padded_rejects_only_second_strategy() {
        let tokenizer = create_test_tokenizer();
        let batch_tokenizer = BatchTokenizer::new(tokenizer)
            .with_max_length(3)
            .with_truncation()
            .with_truncation_strategy(TruncationStrategy::OnlySecond);

        let texts = vec!["a b c"];
        assert!(batch_tokenizer.encode_batch_padded(&texts).is_err());
    }

    /// Regression test for "when max_length is set without truncation,
    /// leaves sequences longer than max_len unpadded, producing a ragged
    /// batch": padding must always converge to one rectangular length.
    #[test]
    fn test_padding_without_truncation_stays_rectangular() {
        let tokenizer = create_test_tokenizer();
        // max_length=2 but truncation NOT enabled: "medium" is naturally
        // longer than 2 (`CharTokenizer::encode` wraps every input in this
        // fixture's [CLS]/[SEP], so "medium" -> 6 chars + 2 wrapper tokens
        // = 8) and must not be left unpadded next to "a" (1 char + 2
        // wrapper tokens = 3, padded).
        let batch_tokenizer = BatchTokenizer::new(tokenizer).with_max_length(2).with_padding(5);

        let texts = vec!["a", "medium"];
        let result = batch_tokenizer.encode_batch_padded(&texts).expect("Operation failed in test");

        let lengths = result.sequence_lengths();
        assert_eq!(
            lengths[0], lengths[1],
            "every row in a batch must share one length"
        );
        assert_eq!(
            lengths[0], 8,
            "the target must grow to fit the naturally-longer row"
        );
    }

    /// `PaddingSide::Left` must add padding to the *front*, keeping the
    /// real tokens right-aligned (required for decoder-only generation).
    #[test]
    fn test_padding_side_left_pads_the_front() {
        let tokenizer = create_test_tokenizer();
        let batch_tokenizer = BatchTokenizer::new(tokenizer)
            .with_max_length(5)
            .with_padding(9)
            .with_padding_side(PaddingSide::Left);

        // `CharTokenizer::encode` wraps "a" (id 0) in this fixture's
        // [CLS]=6/[SEP]=7, producing the 3 real tokens [6, 0, 7]; left
        // padding must add exactly 2 pad tokens to the *front*.
        let texts = vec!["a"];
        let result = batch_tokenizer.encode_batch_padded(&texts).expect("Operation failed in test");

        assert_eq!(result.input_ids[0], vec![9, 9, 6, 0, 7]);
        assert_eq!(result.attention_mask[0], vec![0, 0, 1, 1, 1]);
    }

    #[test]
    fn test_encode_pair_batch_padded_basic() {
        let tokenizer = create_test_tokenizer();
        let batch_tokenizer = BatchTokenizer::new(tokenizer).with_padding(5);

        let pairs = vec![("a", "b"), ("a b c", "a")];
        let result = batch_tokenizer
            .encode_pair_batch_padded(&pairs)
            .expect("Operation failed in test");

        assert_eq!(result.batch_size(), 2);
        let lengths = result.sequence_lengths();
        assert_eq!(lengths[0], lengths[1]);
    }
}
