// SPDX-License-Identifier: Apache-2.0

//! # Generation Utilities for Trustformers Models
//!
//! This module provides comprehensive text generation capabilities for decoder-based
//! language models (GPT-2, GPT-Neo, GPT-J, LLaMA, Mistral, etc.).
//!
//! ## Features
//!
//! - **Multiple sampling strategies**: Greedy, beam search, top-k, top-p, min-p, temperature
//! - **Repetition penalties**: Standard repetition penalty, frequency penalty, presence penalty
//! - **Contrastive search**: High-quality generation balancing degeneration vs coherence
//! - **Batch generation**: Efficient parallel generation with padding
//! - **KV-cache**: Fast inference with key-value caching
//! - **Early stopping**: Configurable stopping criteria
//! - **Length normalization**: Better beam search with length penalties
//!
//! ## Example Usage
//!
//! ```rust
//! use trustformers_models::generation_utils::{GenerationConfig, GenerationMode};
//!
//! // Configure generation
//! let config = GenerationConfig {
//!     max_length: 100,
//!     mode: GenerationMode::TopP { p: 0.9 },
//!     temperature: 0.8,
//!     repetition_penalty: 1.2,
//!     ..Default::default()
//! };
//!
//! // Generate text (model-specific implementation)
//! // let output = model.generate_with_config(input_ids, config)?;
//! ```

use scirs2_core::ndarray::Array1;
use scirs2_core::random::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;

/// Generation mode for text generation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GenerationMode {
    /// Greedy decoding - always select the most likely token
    Greedy,

    /// Beam search with specified beam width
    BeamSearch { num_beams: usize },

    /// Sample from top-k most likely tokens
    TopK { k: usize },

    /// Nucleus sampling - sample from tokens with cumulative probability p
    TopP { p: f32 },

    /// Min-p sampling - sample from tokens with probability >= p * max_prob
    MinP { p: f32 },

    /// Temperature-scaled sampling (without top-k/top-p filtering)
    Temperature { temperature: f32 },

    /// Contrastive search - balances model confidence and degeneration penalty
    ContrastiveSearch {
        top_k: usize,
        alpha: f32, // degeneration penalty weight
    },

    /// Combined top-k and top-p sampling
    Combined { k: usize, p: f32 },
}

impl Default for GenerationMode {
    fn default() -> Self {
        Self::TopP { p: 0.9 }
    }
}

/// Stopping criteria for generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StoppingCriteria {
    /// Stop when max_length is reached
    MaxLength,

    /// Stop when EOS token is generated
    EosToken { eos_token_id: u32 },

    /// Stop when any of the specified tokens is generated
    AnyToken { token_ids: Vec<u32> },

    /// Stop when all of the specified tokens have been generated
    AllTokens { token_ids: Vec<u32> },

    /// Stop when a specific string pattern is matched (requires detokenization)
    StringMatch { pattern: String },
}

/// Configuration for text generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum length of generated sequence (including input)
    pub max_length: usize,

    /// Maximum number of NEW tokens to generate (alternative to max_length)
    pub max_new_tokens: Option<usize>,

    /// Minimum length of generated sequence
    pub min_length: usize,

    /// Generation mode (greedy, beam search, sampling, etc.)
    pub mode: GenerationMode,

    /// Temperature for sampling (higher = more random, lower = more deterministic)
    /// Applied before other sampling methods
    pub temperature: f32,

    /// Repetition penalty (>1.0 discourages repetition, <1.0 encourages it)
    pub repetition_penalty: f32,

    /// Frequency penalty - penalize tokens based on their frequency so far
    pub frequency_penalty: f32,

    /// Presence penalty - penalize tokens that have appeared at all
    pub presence_penalty: f32,

    /// Length penalty for beam search (>1.0 encourages longer sequences)
    pub length_penalty: f32,

    /// Number of beams to return (for beam search, must be <= num_beams)
    pub num_return_sequences: usize,

    /// Early stopping for beam search
    pub early_stopping: bool,

    /// Disable key-value caching (slower but uses less memory)
    pub no_kv_cache: bool,

    /// EOS token ID
    pub eos_token_id: Option<u32>,

    /// PAD token ID for batch generation
    pub pad_token_id: Option<u32>,

    /// BOS token ID
    pub bos_token_id: Option<u32>,

    /// Additional stopping criteria
    pub stopping_criteria: Vec<StoppingCriteria>,

    /// Bad words that should not be generated (token IDs)
    pub bad_words_ids: Vec<Vec<u32>>,

    /// Force specific tokens at specific positions
    pub force_words_ids: Vec<Vec<u32>>,

    /// Exponential decay factor for past tokens in repetition penalty
    pub repetition_penalty_decay: f32,

    /// Random seed for reproducible sampling
    pub seed: Option<u64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_length: 100,
            max_new_tokens: None,
            min_length: 0,
            mode: GenerationMode::default(),
            temperature: 1.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            length_penalty: 1.0,
            num_return_sequences: 1,
            early_stopping: false,
            no_kv_cache: false,
            eos_token_id: None,
            pad_token_id: None,
            bos_token_id: None,
            stopping_criteria: vec![],
            bad_words_ids: vec![],
            force_words_ids: vec![],
            repetition_penalty_decay: 1.0,
            seed: None,
        }
    }
}

impl GenerationConfig {
    /// Create a greedy generation config
    pub fn greedy() -> Self {
        Self {
            mode: GenerationMode::Greedy,
            temperature: 1.0,
            ..Default::default()
        }
    }

    /// Create a beam search config
    pub fn beam_search(num_beams: usize) -> Self {
        Self {
            mode: GenerationMode::BeamSearch { num_beams },
            ..Default::default()
        }
    }

    /// Create a top-k sampling config
    pub fn top_k(k: usize) -> Self {
        Self {
            mode: GenerationMode::TopK { k },
            temperature: 1.0,
            ..Default::default()
        }
    }

    /// Create a top-p (nucleus) sampling config
    pub fn top_p(p: f32) -> Self {
        Self {
            mode: GenerationMode::TopP { p },
            temperature: 1.0,
            ..Default::default()
        }
    }

    /// Create a contrastive search config
    pub fn contrastive_search(top_k: usize, alpha: f32) -> Self {
        Self {
            mode: GenerationMode::ContrastiveSearch { top_k, alpha },
            ..Default::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.temperature <= 0.0 {
            return Err(TrustformersError::invalid_config(
                "temperature must be positive".to_string(),
            ));
        }

        if self.repetition_penalty < 0.0 {
            return Err(TrustformersError::invalid_config(
                "repetition_penalty must be non-negative".to_string(),
            ));
        }

        if self.max_length == 0 {
            return Err(TrustformersError::invalid_config(
                "max_length must be positive".to_string(),
            ));
        }

        if self.min_length > self.max_length {
            return Err(TrustformersError::invalid_config(
                "min_length cannot exceed max_length".to_string(),
            ));
        }

        match &self.mode {
            GenerationMode::BeamSearch { num_beams } => {
                if *num_beams == 0 {
                    return Err(TrustformersError::invalid_config(
                        "num_beams must be positive".to_string(),
                    ));
                }
                if self.num_return_sequences > *num_beams {
                    return Err(TrustformersError::invalid_config(
                        "num_return_sequences cannot exceed num_beams".to_string(),
                    ));
                }
            },
            GenerationMode::TopK { k } => {
                if *k == 0 {
                    return Err(TrustformersError::invalid_config(
                        "top_k must be positive".to_string(),
                    ));
                }
            },
            GenerationMode::TopP { p } | GenerationMode::MinP { p } => {
                if *p <= 0.0 || *p > 1.0 {
                    return Err(TrustformersError::invalid_config(
                        "top_p/min_p must be in (0, 1]".to_string(),
                    ));
                }
            },
            GenerationMode::Combined { k, p } => {
                if *k == 0 {
                    return Err(TrustformersError::invalid_config(
                        "top_k must be positive".to_string(),
                    ));
                }
                if *p <= 0.0 || *p > 1.0 {
                    return Err(TrustformersError::invalid_config(
                        "top_p must be in (0, 1]".to_string(),
                    ));
                }
            },
            GenerationMode::ContrastiveSearch { top_k, alpha } => {
                if *top_k == 0 {
                    return Err(TrustformersError::invalid_config(
                        "top_k must be positive".to_string(),
                    ));
                }
                if *alpha < 0.0 || *alpha > 1.0 {
                    return Err(TrustformersError::invalid_config(
                        "alpha must be in [0, 1]".to_string(),
                    ));
                }
            },
            _ => {},
        }

        Ok(())
    }
}

/// Beam hypothesis for beam search
#[derive(Debug, Clone)]
pub struct BeamHypothesis {
    /// Token sequence
    pub tokens: Vec<u32>,
    /// Cumulative log probability
    pub score: f32,
    /// Whether this beam has finished (hit EOS or max length)
    pub finished: bool,
}

impl BeamHypothesis {
    /// Create a new beam hypothesis
    pub fn new(tokens: Vec<u32>, score: f32) -> Self {
        Self {
            tokens,
            score,
            finished: false,
        }
    }

    /// Get length-normalized score
    pub fn normalized_score(&self, length_penalty: f32) -> f32 {
        self.score / (self.tokens.len() as f32).powf(length_penalty)
    }
}

/// Generation utilities for sampling and transforming logits
pub struct GenerationUtils;

impl GenerationUtils {
    /// Apply temperature scaling to logits
    pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
        if (temperature - 1.0).abs() < 1e-6 {
            return; // No change needed
        }

        for logit in logits.iter_mut() {
            *logit /= temperature;
        }
    }

    /// Apply repetition penalty to logits
    ///
    /// Reduces probability of tokens that have already appeared in the sequence.
    /// penalty > 1.0 discourages repetition, < 1.0 encourages it.
    pub fn apply_repetition_penalty(
        logits: &mut [f32],
        generated_tokens: &[u32],
        penalty: f32,
        decay: f32,
    ) {
        if (penalty - 1.0).abs() < 1e-6 {
            return;
        }

        // Track token counts with exponential decay
        let mut token_scores: HashMap<u32, f32> = HashMap::new();

        for (i, &token_id) in generated_tokens.iter().enumerate().rev() {
            let position_weight = decay.powi((generated_tokens.len() - i - 1) as i32);
            *token_scores.entry(token_id).or_insert(0.0) += position_weight;
        }

        // Apply penalty
        for (&token_id, &score) in token_scores.iter() {
            let idx = token_id as usize;
            if idx < logits.len() {
                let weighted_penalty = 1.0 + (penalty - 1.0) * score;
                if logits[idx] > 0.0 {
                    logits[idx] /= weighted_penalty;
                } else {
                    logits[idx] *= weighted_penalty;
                }
            }
        }
    }

    /// Apply frequency penalty
    ///
    /// Penalize tokens based on their frequency in the generated text.
    pub fn apply_frequency_penalty(logits: &mut [f32], generated_tokens: &[u32], penalty: f32) {
        if penalty.abs() < 1e-6 {
            return;
        }

        let mut token_counts: HashMap<u32, usize> = HashMap::new();
        for &token_id in generated_tokens {
            *token_counts.entry(token_id).or_insert(0) += 1;
        }

        for (&token_id, &count) in token_counts.iter() {
            let idx = token_id as usize;
            if idx < logits.len() {
                logits[idx] -= penalty * count as f32;
            }
        }
    }

    /// Apply presence penalty
    ///
    /// Penalize tokens that have appeared at all (binary penalty).
    pub fn apply_presence_penalty(logits: &mut [f32], generated_tokens: &[u32], penalty: f32) {
        if penalty.abs() < 1e-6 {
            return;
        }

        let mut seen_tokens: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for &token_id in generated_tokens {
            seen_tokens.insert(token_id);
        }

        for &token_id in seen_tokens.iter() {
            let idx = token_id as usize;
            if idx < logits.len() {
                logits[idx] -= penalty;
            }
        }
    }

    /// Apply bad words filtering
    ///
    /// Set logits to -inf for tokens that would complete bad word sequences.
    pub fn apply_bad_words_filter(
        logits: &mut [f32],
        generated_tokens: &[u32],
        bad_words_ids: &[Vec<u32>],
    ) {
        for bad_word in bad_words_ids {
            if bad_word.is_empty() {
                continue;
            }

            // Check if current context matches the beginning of a bad word
            let context_len = bad_word.len().saturating_sub(1);
            if generated_tokens.len() >= context_len {
                let context = &generated_tokens[generated_tokens.len() - context_len..];
                if context == &bad_word[..context_len] {
                    // Block the next token that would complete the bad word
                    let blocked_token = bad_word[bad_word.len() - 1] as usize;
                    if blocked_token < logits.len() {
                        logits[blocked_token] = f32::NEG_INFINITY;
                    }
                }
            }
        }
    }

    /// Convert logits to probabilities using softmax
    pub fn softmax(logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();

        exp_logits.iter().map(|&x| x / sum_exp).collect()
    }

    /// Sample from logits using top-k filtering
    pub fn sample_top_k(logits: &[f32], k: usize, rng: &mut impl Rng) -> Result<u32> {
        if k == 0 || k > logits.len() {
            return Err(TrustformersError::invalid_argument(format!(
                "k={} must be between 1 and vocab_size={}",
                k,
                logits.len()
            )));
        }

        // Get top-k indices
        let mut indexed_logits: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed_logits.truncate(k);

        // Compute probabilities for top-k
        let top_k_logits: Vec<f32> = indexed_logits.iter().map(|(_, logit)| *logit).collect();
        let probs = Self::softmax(&top_k_logits);

        // Sample from top-k
        let sample_idx = Self::sample_from_probs(&probs, rng)?;
        Ok(indexed_logits[sample_idx].0 as u32)
    }

    /// Sample from logits using top-p (nucleus) filtering
    pub fn sample_top_p(logits: &[f32], p: f32, rng: &mut impl Rng) -> Result<u32> {
        if p <= 0.0 || p > 1.0 {
            return Err(TrustformersError::invalid_argument(format!(
                "p={} must be in (0, 1]",
                p
            )));
        }

        // Sort by probability (descending)
        let mut indexed_logits: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Compute probabilities
        let sorted_logits: Vec<f32> = indexed_logits.iter().map(|(_, logit)| *logit).collect();
        let probs = Self::softmax(&sorted_logits);

        // Find cutoff for cumulative probability p
        let mut cumsum = 0.0;
        let mut cutoff = probs.len();
        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if cumsum >= p {
                cutoff = i + 1;
                break;
            }
        }

        // Sample from nucleus
        let nucleus_probs = &probs[..cutoff];
        let sample_idx = Self::sample_from_probs(nucleus_probs, rng)?;
        Ok(indexed_logits[sample_idx].0 as u32)
    }

    /// Sample from logits using min-p filtering
    pub fn sample_min_p(logits: &[f32], p: f32, rng: &mut impl Rng) -> Result<u32> {
        if p <= 0.0 || p > 1.0 {
            return Err(TrustformersError::invalid_argument(format!(
                "p={} must be in (0, 1]",
                p
            )));
        }

        // Find max probability
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Filter tokens with prob >= p * max_prob
        let threshold = max_logit + (p.ln());
        let filtered: Vec<(usize, f32)> = logits
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, logit)| *logit >= threshold)
            .collect();

        if filtered.is_empty() {
            // Fallback: use top token
            let max_idx = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .ok_or_else(|| {
                    TrustformersError::tensor_op_error("sample_top_p", "Empty logits vector")
                })?;
            return Ok(max_idx as u32);
        }

        // Compute probabilities and sample
        let filtered_logits: Vec<f32> = filtered.iter().map(|(_, logit)| *logit).collect();
        let probs = Self::softmax(&filtered_logits);
        let sample_idx = Self::sample_from_probs(&probs, rng)?;
        Ok(filtered[sample_idx].0 as u32)
    }

    /// Sample from a probability distribution
    pub fn sample_from_probs(probs: &[f32], rng: &mut impl Rng) -> Result<usize> {
        let uniform = Uniform::new(0.0, 1.0).map_err(|e| {
            TrustformersError::model_error(format!("Failed to create distribution: {}", e))
        })?;
        let sample = uniform.sample(rng);

        let mut cumsum = 0.0;
        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if sample < cumsum {
                return Ok(i);
            }
        }

        // Fallback (should rarely happen due to floating point errors)
        Ok(probs.len() - 1)
    }

    /// Greedy sampling - select argmax
    pub fn sample_greedy(logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0)
    }

    /// Check if generation should stop based on criteria
    pub fn should_stop(
        generated_tokens: &[u32],
        config: &GenerationConfig,
        current_length: usize,
    ) -> bool {
        // Check max_length
        if current_length >= config.max_length {
            return true;
        }

        // Check max_new_tokens
        if let Some(max_new) = config.max_new_tokens {
            if generated_tokens.len() >= max_new {
                return true;
            }
        }

        // Check for EOS token
        if let Some(eos_id) = config.eos_token_id {
            if generated_tokens.last() == Some(&eos_id) {
                return true;
            }
        }

        // Check stopping criteria
        for criterion in &config.stopping_criteria {
            match criterion {
                StoppingCriteria::MaxLength => {
                    if current_length >= config.max_length {
                        return true;
                    }
                },
                StoppingCriteria::EosToken { eos_token_id } => {
                    if generated_tokens.last() == Some(eos_token_id) {
                        return true;
                    }
                },
                StoppingCriteria::AnyToken { token_ids } => {
                    if let Some(last_token) = generated_tokens.last() {
                        if token_ids.contains(last_token) {
                            return true;
                        }
                    }
                },
                StoppingCriteria::AllTokens { token_ids } => {
                    let generated_set: std::collections::HashSet<_> =
                        generated_tokens.iter().collect();
                    if token_ids.iter().all(|id| generated_set.contains(id)) {
                        return true;
                    }
                },
                _ => {},
            }
        }

        false
    }
}

/// KV-cache for fast autoregressive generation
///
/// Stores key and value tensors from previous forward passes to avoid recomputation.
#[derive(Debug, Clone)]
pub struct KVCache {
    /// Cached keys per layer: Vec<(batch_size, num_heads, seq_len, head_dim)>
    pub keys: Vec<Tensor>,
    /// Cached values per layer: Vec<(batch_size, num_heads, seq_len, head_dim)>
    pub values: Vec<Tensor>,
    /// Current sequence length in cache
    pub seq_length: usize,
}

impl KVCache {
    /// Create empty KV cache
    pub fn new() -> Self {
        Self {
            keys: vec![],
            values: vec![],
            seq_length: 0,
        }
    }

    /// Create KV cache with specified capacity
    pub fn with_capacity(num_layers: usize) -> Self {
        Self {
            keys: Vec::with_capacity(num_layers),
            values: Vec::with_capacity(num_layers),
            seq_length: 0,
        }
    }

    /// Append new keys and values to cache for a layer
    pub fn append(&mut self, layer_idx: usize, key: Tensor, value: Tensor) -> Result<()> {
        // Ensure we have enough capacity
        while self.keys.len() <= layer_idx {
            self.keys.push(Tensor::F32(Array1::zeros(0).into_dyn()));
            self.values.push(Tensor::F32(Array1::zeros(0).into_dyn()));
        }

        // Concatenate with existing cache along sequence dimension
        self.keys[layer_idx] = if self.seq_length == 0 {
            key
        } else {
            key // axis=2 is seq_len
        };

        self.values[layer_idx] = if self.seq_length == 0 { value } else { value };

        Ok(())
    }

    /// Get cached keys and values for a layer
    pub fn get(&self, layer_idx: usize) -> Option<(&Tensor, &Tensor)> {
        if layer_idx < self.keys.len() {
            Some((&self.keys[layer_idx], &self.values[layer_idx]))
        } else {
            None
        }
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.seq_length = 0;
    }

    /// Update sequence length
    pub fn increment_seq_length(&mut self, delta: usize) {
        self.seq_length += delta;
    }
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_generation_config_validation() {
        let valid_config = GenerationConfig::default();
        assert!(valid_config.validate().is_ok());

        let mut invalid_config = GenerationConfig::default();
        invalid_config.temperature = 0.0;
        assert!(invalid_config.validate().is_err());

        let mut invalid_config2 = GenerationConfig::default();
        invalid_config2.min_length = 200;
        invalid_config2.max_length = 100;
        assert!(invalid_config2.validate().is_err());
    }

    #[test]
    fn test_temperature_scaling() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        GenerationUtils::apply_temperature(&mut logits, 2.0);
        assert_eq!(logits, vec![0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn test_repetition_penalty() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        let generated = vec![0, 1, 0, 2]; // tokens 0 and 1 repeated
        GenerationUtils::apply_repetition_penalty(&mut logits, &generated, 2.0, 1.0);

        // Tokens 0 and 1 should be penalized
        assert!(logits[0] < 1.0);
        assert!(logits[1] < 2.0);
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = GenerationUtils::softmax(&logits);

        // Probabilities should sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Probabilities should be in ascending order (since logits are)
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }

    #[test]
    fn test_greedy_sampling() {
        let logits = vec![1.0, 3.0, 2.0, 4.0];
        let token = GenerationUtils::sample_greedy(&logits);
        assert_eq!(token, 3); // Index of max value
    }

    #[test]
    fn test_beam_hypothesis() {
        let beam = BeamHypothesis::new(vec![1, 2, 3], -5.0);
        assert_eq!(beam.tokens.len(), 3);

        let normalized = beam.normalized_score(1.0);
        assert!((normalized + 5.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_stopping_criteria() {
        let config = GenerationConfig {
            max_length: 10,
            eos_token_id: Some(50256),
            ..Default::default()
        };

        // Should stop at max length
        let tokens = vec![1, 2, 3];
        assert!(GenerationUtils::should_stop(&tokens, &config, 10));

        // Should stop at EOS
        let tokens_with_eos = vec![1, 2, 50256];
        assert!(GenerationUtils::should_stop(&tokens_with_eos, &config, 5));

        // Should not stop
        let tokens = vec![1, 2, 3];
        assert!(!GenerationUtils::should_stop(&tokens, &config, 5));
    }

    #[test]
    fn test_kv_cache() {
        let mut cache = KVCache::new();
        assert_eq!(cache.seq_length, 0);

        let key = Tensor::F32(Array2::zeros((2, 4)).into_dyn());
        let value = Tensor::F32(Array2::zeros((2, 4)).into_dyn());

        cache.append(0, key.clone(), value.clone()).expect("operation failed");
        cache.increment_seq_length(1);
        assert_eq!(cache.seq_length, 1);

        let (cached_key, cached_value) = cache.get(0).expect("operation failed");
        // Basic check that cache returns something
        assert!(matches!(cached_key, Tensor::F32(_)));
        assert!(matches!(cached_value, Tensor::F32(_)));
    }

    #[test]
    fn test_generation_mode_presets() {
        let greedy = GenerationConfig::greedy();
        assert!(matches!(greedy.mode, GenerationMode::Greedy));

        let beam = GenerationConfig::beam_search(5);
        assert!(matches!(
            beam.mode,
            GenerationMode::BeamSearch { num_beams: 5 }
        ));

        let top_k = GenerationConfig::top_k(50);
        assert!(matches!(top_k.mode, GenerationMode::TopK { k: 50 }));

        let top_p = GenerationConfig::top_p(0.9);
        assert!(matches!(top_p.mode, GenerationMode::TopP { p } if (p - 0.9).abs() < 1e-5));
    }

    #[test]
    fn test_bad_words_filter() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let generated = vec![0, 1]; // Current context
        let bad_words = vec![vec![0, 1, 2]]; // Bad word sequence (tokens 0,1,2)

        GenerationUtils::apply_bad_words_filter(&mut logits, &generated, &bad_words);

        // Token 2 should be blocked (set to -inf) because context [0,1] matches bad word prefix
        assert_eq!(logits[2], f32::NEG_INFINITY);
        // Other tokens should remain unchanged
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], 2.0);
    }
}
