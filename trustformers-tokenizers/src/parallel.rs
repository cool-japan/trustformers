use scirs2_core::parallel_ops::*; // SciRS2 Integration Policy - replaces rayon
use std::sync::Arc;
use trustformers_core::errors::Result;
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

/// Batch tokenization with padding and truncation support
#[derive(Debug, Clone)]
pub struct BatchTokenizer<T: Tokenizer + Sync> {
    tokenizer: Arc<T>,
    max_length: Option<usize>,
    padding: bool,
    truncation: bool,
    pad_token_id: u32,
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

    /// Encode a batch with padding and truncation
    pub fn encode_batch_padded(&self, texts: &[&str]) -> Result<BatchedTokenizedInput> {
        // First, encode all texts in parallel
        let encoded: Vec<TokenizedInput> = texts
            .par_iter()
            .map(|text| self.tokenizer.encode(text))
            .collect::<Result<Vec<_>>>()?;

        // Apply truncation if enabled
        let mut processed = if let (true, Some(max_len)) = (self.truncation, self.max_length) {
            encoded
                .into_iter()
                .map(|mut input| {
                    if input.input_ids.len() > max_len {
                        input.input_ids.truncate(max_len);
                        input.attention_mask.truncate(max_len);
                        if let Some(ref mut type_ids) = input.token_type_ids {
                            type_ids.truncate(max_len);
                        }
                    }
                    input
                })
                .collect()
        } else {
            encoded
        };

        // Apply padding if enabled
        if self.padding {
            let max_len = if let Some(max_len) = self.max_length {
                max_len
            } else {
                processed.iter().map(|input| input.input_ids.len()).max().unwrap_or(0)
            };

            for input in &mut processed {
                let current_len = input.input_ids.len();
                if current_len < max_len {
                    let pad_len = max_len - current_len;
                    input.input_ids.extend(vec![self.pad_token_id; pad_len]);
                    input.attention_mask.extend(vec![0u8; pad_len]);
                    if let Some(ref mut type_ids) = input.token_type_ids {
                        type_ids.extend(vec![0u32; pad_len]);
                    }
                }
            }
        }

        Ok(BatchedTokenizedInput::from_batch(processed))
    }

    /// Get the underlying tokenizer
    pub fn tokenizer(&self) -> &T {
        &self.tokenizer
    }
}

/// Batched tokenized input with convenient access methods
#[derive(Debug, Clone)]
pub struct BatchedTokenizedInput {
    pub input_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u8>>,
    pub token_type_ids: Option<Vec<Vec<u32>>>,
}

impl BatchedTokenizedInput {
    /// Create from a batch of TokenizedInput
    pub fn from_batch(batch: Vec<TokenizedInput>) -> Self {
        let mut input_ids = Vec::with_capacity(batch.len());
        let mut attention_mask = Vec::with_capacity(batch.len());
        let mut token_type_ids = Vec::with_capacity(batch.len());

        let has_token_type_ids = batch.iter().any(|input| input.token_type_ids.is_some());

        for input in batch {
            input_ids.push(input.input_ids);
            attention_mask.push(input.attention_mask);
            if has_token_type_ids {
                token_type_ids.push(input.token_type_ids.unwrap_or_default());
            }
        }

        Self {
            input_ids,
            attention_mask,
            token_type_ids: if has_token_type_ids { Some(token_type_ids) } else { None },
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

            result.push(TokenizedInput {
                input_ids: self.input_ids[i].clone(),
                attention_mask: self.attention_mask[i].clone(),
                token_type_ids,
                special_tokens_mask: None,
                offset_mapping: None,
                overflowing_tokens: None,
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
        let results = parallel_tokenizer.encode_batch(&texts).unwrap();

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
        let results = parallel_tokenizer.encode_pair_batch(&pairs).unwrap();

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
        let result = batch_tokenizer.encode_batch_padded(&texts).unwrap();

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

        let results = parallel_tokenizer.decode_batch(&ids_batch).unwrap();
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
}
