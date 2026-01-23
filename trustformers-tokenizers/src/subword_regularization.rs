use scirs2_core::random::*; // SciRS2 Integration Policy - Replaces rand
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::errors::Result;
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// Configuration for subword regularization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubwordRegularizationConfig {
    /// Alpha parameter for controlling randomness (0.0 = no randomness, 1.0 = maximum randomness)
    pub alpha: f32,
    /// Number of alternative segmentations to sample
    pub num_samples: usize,
    /// Seed for reproducible randomness
    pub seed: Option<u64>,
    /// Enable debugging output
    pub debug: bool,
}

impl Default for SubwordRegularizationConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            num_samples: 1,
            seed: None,
            debug: false,
        }
    }
}

/// Subword regularization wrapper that adds randomness to tokenization
pub struct SubwordRegularizer<T: Tokenizer> {
    tokenizer: T,
    config: SubwordRegularizationConfig,
    rng: StdRng,
}

impl<T: Tokenizer> SubwordRegularizer<T> {
    pub fn new(tokenizer: T, config: SubwordRegularizationConfig) -> Self {
        let rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            // Generate random seed from thread_rng
            let seed = thread_rng().random();
            StdRng::seed_from_u64(seed)
        };

        Self {
            tokenizer,
            config,
            rng,
        }
    }

    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.config.alpha = alpha;
        self
    }

    pub fn with_num_samples(mut self, num_samples: usize) -> Self {
        self.config.num_samples = num_samples;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self.rng = StdRng::seed_from_u64(seed);
        self
    }

    /// Generate multiple tokenizations with regularization
    pub fn encode_with_regularization(&mut self, text: &str) -> Result<Vec<TokenizedInput>> {
        let mut results = Vec::new();

        for _ in 0..self.config.num_samples {
            let regularized_text = self.apply_regularization(text);
            let tokenized = self.tokenizer.encode(&regularized_text)?;
            results.push(tokenized);
        }

        Ok(results)
    }

    /// Apply regularization to text (simplified version)
    fn apply_regularization(&mut self, text: &str) -> String {
        if self.config.alpha <= 0.0 {
            return text.to_string();
        }

        let mut result = String::new();
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let char = chars[i];

            // Add some randomness to character processing
            if self.rng.random::<f32>() < self.config.alpha {
                // Skip character with some probability
                if self.rng.random::<f32>() < 0.1 {
                    i += 1;
                    continue;
                }

                // Duplicate character with some probability
                if self.rng.random::<f32>() < 0.05 {
                    result.push(char);
                    result.push(char);
                    i += 1;
                    continue;
                }
            }

            result.push(char);
            i += 1;
        }

        result
    }

    /// Get the underlying tokenizer
    pub fn inner(&self) -> &T {
        &self.tokenizer
    }

    /// Get the configuration
    pub fn config(&self) -> &SubwordRegularizationConfig {
        &self.config
    }
}

impl<T: Tokenizer> Tokenizer for SubwordRegularizer<T> {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        // For the basic interface, just use the underlying tokenizer
        self.tokenizer.encode(text)
    }

    fn encode_pair(&self, text: &str, text2: &str) -> Result<TokenizedInput> {
        self.tokenizer.encode_pair(text, text2)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer.decode(ids)
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.tokenizer.get_vocab()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }
}

/// Unigram-specific subword regularization implementation
pub struct UnigramSubwordRegularizer {
    vocab: HashMap<String, f32>,
    config: SubwordRegularizationConfig,
    rng: StdRng,
}

impl UnigramSubwordRegularizer {
    pub fn new(vocab: HashMap<String, f32>, config: SubwordRegularizationConfig) -> Self {
        let rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            // Generate random seed from thread_rng
            let seed = thread_rng().random();
            StdRng::seed_from_u64(seed)
        };

        Self { vocab, config, rng }
    }

    /// Sample a segmentation using the Unigram language model with regularization
    pub fn sample_segmentation(&mut self, text: &str) -> Result<Vec<String>> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();

        if n == 0 {
            return Ok(vec![]);
        }

        // Dynamic programming with sampling
        let mut dp = vec![vec![0.0; n + 1]; n + 1];
        let mut best_seg = vec![vec![None; n + 1]; n + 1];

        // Initialize
        for (i, dp_row) in dp.iter_mut().enumerate().take(n + 1) {
            dp_row[i] = 0.0;
        }

        // Fill DP table with regularization
        for length in 1..=n {
            for start in 0..=n - length {
                let end = start + length;
                let substring: String = chars[start..end].iter().collect();

                if let Some(&score) = self.vocab.get(&substring) {
                    // Apply regularization to the score
                    let regularized_score = if self.config.alpha > 0.0 {
                        let noise = self.rng.random::<f32>() * self.config.alpha;
                        score + noise - self.config.alpha / 2.0
                    } else {
                        score
                    };

                    if dp[start][end] < regularized_score {
                        dp[start][end] = regularized_score;
                        best_seg[start][end] = Some(substring);
                    }
                }

                // Try splitting at intermediate points
                for mid in start + 1..end {
                    let combined_score = dp[start][mid] + dp[mid][end];
                    if dp[start][end] < combined_score {
                        dp[start][end] = combined_score;
                        best_seg[start][end] = None; // Mark as split
                    }
                }
            }
        }

        // Backtrack to get the segmentation
        self.backtrack_segmentation(&best_seg, 0, n, &chars)
    }

    #[allow(clippy::only_used_in_recursion)]
    fn backtrack_segmentation(
        &self,
        best_seg: &[Vec<Option<String>>],
        start: usize,
        end: usize,
        chars: &[char],
    ) -> Result<Vec<String>> {
        if start == end {
            return Ok(vec![]);
        }

        if let Some(ref segment) = best_seg[start][end] {
            return Ok(vec![segment.clone()]);
        }

        // Find the best split point
        let mut best_split = start + 1;
        let mut best_score = f32::NEG_INFINITY;

        for (mid, _) in best_seg.iter().enumerate().take(end).skip(start + 1) {
            let score = best_seg[start][mid].as_ref().map(|_| 1.0).unwrap_or(0.0)
                + best_seg[mid][end].as_ref().map(|_| 1.0).unwrap_or(0.0);
            if score > best_score {
                best_score = score;
                best_split = mid;
            }
        }

        let mut result = self.backtrack_segmentation(best_seg, start, best_split, chars)?;
        let mut right_part = self.backtrack_segmentation(best_seg, best_split, end, chars)?;
        result.append(&mut right_part);

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::char::CharTokenizer;

    #[test]
    fn test_subword_regularization_config() {
        let config = SubwordRegularizationConfig::default();
        assert_eq!(config.alpha, 0.1);
        assert_eq!(config.num_samples, 1);
        assert_eq!(config.seed, None);
        assert!(!config.debug);
    }

    #[test]
    fn test_subword_regularizer_creation() {
        let tokenizer = CharTokenizer::from_text("hello world", 1000);
        let config = SubwordRegularizationConfig::default();
        let regularizer = SubwordRegularizer::new(tokenizer, config);

        assert_eq!(regularizer.config().alpha, 0.1);
        assert_eq!(regularizer.config().num_samples, 1);
    }

    #[test]
    fn test_subword_regularizer_encode() {
        let tokenizer = CharTokenizer::from_text("hello world", 1000);
        let config = SubwordRegularizationConfig::default();
        let regularizer = SubwordRegularizer::new(tokenizer, config);

        let result = regularizer.encode("hello");
        assert!(result.is_ok());

        let tokenized = result.unwrap();
        assert!(!tokenized.input_ids.is_empty());
    }

    #[test]
    fn test_subword_regularizer_with_seed() {
        let tokenizer = CharTokenizer::from_text("hello world", 1000);
        let config = SubwordRegularizationConfig::default();
        let mut regularizer = SubwordRegularizer::new(tokenizer, config).with_seed(42);

        let result1 = regularizer.encode_with_regularization("hello world");
        assert!(result1.is_ok());

        // Reset with same seed
        let tokenizer2 = CharTokenizer::from_text("hello world", 1000);
        let config2 = SubwordRegularizationConfig::default();
        let mut regularizer2 = SubwordRegularizer::new(tokenizer2, config2).with_seed(42);

        let result2 = regularizer2.encode_with_regularization("hello world");
        assert!(result2.is_ok());
    }

    #[test]
    fn test_subword_regularizer_multiple_samples() {
        let tokenizer = CharTokenizer::from_text("hello world", 1000);
        let config = SubwordRegularizationConfig::default();
        let mut regularizer =
            SubwordRegularizer::new(tokenizer, config).with_num_samples(3).with_alpha(0.2);

        let results = regularizer.encode_with_regularization("hello world");
        assert!(results.is_ok());

        let tokenized_results = results.unwrap();
        assert_eq!(tokenized_results.len(), 3);

        for result in tokenized_results {
            assert!(!result.input_ids.is_empty());
        }
    }

    #[test]
    fn test_unigram_subword_regularizer() {
        let mut vocab = HashMap::new();
        vocab.insert("hello".to_string(), 1.0);
        vocab.insert("world".to_string(), 1.0);
        vocab.insert("h".to_string(), 0.5);
        vocab.insert("e".to_string(), 0.5);
        vocab.insert("l".to_string(), 0.5);
        vocab.insert("o".to_string(), 0.5);

        let config = SubwordRegularizationConfig::default();
        let mut regularizer = UnigramSubwordRegularizer::new(vocab, config);

        let result = regularizer.sample_segmentation("hello");
        assert!(result.is_ok());

        let segmentation = result.unwrap();
        assert!(!segmentation.is_empty());
    }

    #[test]
    fn test_unigram_regularizer_with_alpha() {
        let mut vocab = HashMap::new();
        vocab.insert("test".to_string(), 1.0);
        vocab.insert("t".to_string(), 0.3);
        vocab.insert("e".to_string(), 0.3);
        vocab.insert("s".to_string(), 0.3);

        let config = SubwordRegularizationConfig {
            alpha: 0.5,
            num_samples: 1,
            seed: Some(123),
            debug: false,
        };

        let mut regularizer = UnigramSubwordRegularizer::new(vocab, config);

        let result1 = regularizer.sample_segmentation("test");
        assert!(result1.is_ok());

        // Results should be different due to regularization
        let result2 = regularizer.sample_segmentation("test");
        assert!(result2.is_ok());
    }

    #[test]
    fn test_regularization_config_serialization() {
        let config = SubwordRegularizationConfig {
            alpha: 0.3,
            num_samples: 5,
            seed: Some(42),
            debug: true,
        };

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: SubwordRegularizationConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config.alpha, deserialized.alpha);
        assert_eq!(config.num_samples, deserialized.num_samples);
        assert_eq!(config.seed, deserialized.seed);
        assert_eq!(config.debug, deserialized.debug);
    }

    #[test]
    fn test_apply_regularization() {
        let tokenizer = CharTokenizer::from_text("hello world", 1000);
        let config = SubwordRegularizationConfig {
            alpha: 0.0, // No regularization
            num_samples: 1,
            seed: Some(42),
            debug: false,
        };

        let mut regularizer = SubwordRegularizer::new(tokenizer, config);
        let result = regularizer.apply_regularization("hello");
        assert_eq!(result, "hello");

        // With regularization
        regularizer.config.alpha = 0.5;
        let result_with_reg = regularizer.apply_regularization("hello");
        // Result might be different due to randomness
        assert!(!result_with_reg.is_empty());
    }
}
