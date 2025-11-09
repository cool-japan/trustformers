#![allow(deprecated)] // Using rand legacy API, will migrate to scirs2_core

use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use rand::{thread_rng, Rng};

use super::cache::KVCache;
use super::config::{GenerationConfig, GenerationStrategy};

/// Core text generator with various generation strategies
pub struct TextGenerator {
    pub config: GenerationConfig,
    pub vocab_size: usize,
}

impl TextGenerator {
    pub fn new(config: GenerationConfig, vocab_size: usize) -> Self {
        Self { config, vocab_size }
    }

    /// Generate text using the configured strategy
    pub fn generate(
        &self,
        input_ids: &[usize],
        logits_fn: impl Fn(&[usize], Option<&KVCache>) -> Result<(Tensor, Option<KVCache>)>,
    ) -> Result<Vec<Vec<usize>>> {
        match self.config.strategy {
            GenerationStrategy::Greedy => self.generate_greedy(input_ids, logits_fn),
            GenerationStrategy::BeamSearch { num_beams } => {
                self.generate_beam_search(input_ids, num_beams, logits_fn)
            },
            GenerationStrategy::Sampling { temperature } => {
                self.generate_sampling(input_ids, temperature, logits_fn)
            },
            GenerationStrategy::TopK { k, temperature } => {
                self.generate_top_k(input_ids, k, temperature, logits_fn)
            },
            GenerationStrategy::TopP { p, temperature } => {
                self.generate_top_p(input_ids, p, temperature, logits_fn)
            },
            GenerationStrategy::ContrastiveSearch {
                penalty_alpha,
                top_k,
            } => self.generate_contrastive_search(input_ids, penalty_alpha, top_k, logits_fn),
        }
    }

    /// Get maximum generation length
    pub fn get_max_length(&self, input_length: usize) -> usize {
        if let Some(max_new_tokens) = self.config.max_new_tokens {
            input_length + max_new_tokens
        } else if let Some(max_length) = self.config.max_length {
            max_length
        } else {
            input_length + 100 // Default fallback
        }
    }

    /// Check if generation should stop
    pub fn should_stop(
        &self,
        sequence: &[usize],
        last_token: usize,
        current_length: usize,
    ) -> bool {
        // Check EOS token
        if Some(last_token) == self.config.eos_token_id {
            return true;
        }

        // Check maximum length
        let max_length = self.get_max_length(sequence.len() - current_length);
        if current_length >= max_length {
            return true;
        }

        // Check minimum length
        if let Some(min_length) = self.config.min_length {
            if current_length < min_length {
                return false;
            }
        }

        false
    }

    /// Sample a token from logits
    pub fn sample_token(&self, logits: &Tensor) -> Result<usize> {
        match logits {
            Tensor::F32(arr) => {
                let data: Vec<f32> = arr.iter().cloned().collect();
                self.greedy_select_from_data(&data)
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor type for sampling",
                "sample_token",
            )),
        }
    }

    /// Simple greedy generation implementation
    fn generate_greedy(
        &self,
        input_ids: &[usize],
        logits_fn: impl Fn(&[usize], Option<&KVCache>) -> Result<(Tensor, Option<KVCache>)>,
    ) -> Result<Vec<Vec<usize>>> {
        let mut sequences = vec![input_ids.to_vec()];
        let mut cache = if self.config.use_cache { Some(KVCache::new()) } else { None };

        let max_length = self.get_max_length(input_ids.len());

        for step in 0..max_length {
            let (logits, new_cache) = logits_fn(&sequences[0], cache.as_ref())?;
            cache = new_cache;

            let next_token = self.sample_token(&logits)?;
            sequences[0].push(next_token);

            if self.should_stop(&sequences[0], next_token, step + 1) {
                break;
            }
        }

        Ok(sequences)
    }

    /// Placeholder for other generation methods
    fn generate_beam_search(
        &self,
        input_ids: &[usize],
        _num_beams: usize,
        logits_fn: impl Fn(&[usize], Option<&KVCache>) -> Result<(Tensor, Option<KVCache>)>,
    ) -> Result<Vec<Vec<usize>>> {
        // For now, fallback to greedy
        self.generate_greedy(input_ids, logits_fn)
    }

    fn generate_sampling(
        &self,
        input_ids: &[usize],
        _temperature: f32,
        logits_fn: impl Fn(&[usize], Option<&KVCache>) -> Result<(Tensor, Option<KVCache>)>,
    ) -> Result<Vec<Vec<usize>>> {
        // For now, fallback to greedy
        self.generate_greedy(input_ids, logits_fn)
    }

    fn generate_top_k(
        &self,
        input_ids: &[usize],
        _k: usize,
        _temperature: f32,
        logits_fn: impl Fn(&[usize], Option<&KVCache>) -> Result<(Tensor, Option<KVCache>)>,
    ) -> Result<Vec<Vec<usize>>> {
        // For now, fallback to greedy
        self.generate_greedy(input_ids, logits_fn)
    }

    fn generate_top_p(
        &self,
        input_ids: &[usize],
        _p: f32,
        _temperature: f32,
        logits_fn: impl Fn(&[usize], Option<&KVCache>) -> Result<(Tensor, Option<KVCache>)>,
    ) -> Result<Vec<Vec<usize>>> {
        // For now, fallback to greedy
        self.generate_greedy(input_ids, logits_fn)
    }

    fn generate_contrastive_search(
        &self,
        input_ids: &[usize],
        _penalty_alpha: f32,
        _top_k: usize,
        logits_fn: impl Fn(&[usize], Option<&KVCache>) -> Result<(Tensor, Option<KVCache>)>,
    ) -> Result<Vec<Vec<usize>>> {
        // For now, fallback to greedy
        self.generate_greedy(input_ids, logits_fn)
    }

    /// Simple greedy selection from probability data
    fn greedy_select_from_data(&self, data: &[f32]) -> Result<usize> {
        data.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .ok_or_else(|| TrustformersError::invalid_input("Empty logits".to_string()))
    }

    /// Apply softmax to logits
    pub fn softmax(&self, logits: &[f32]) -> Result<Vec<f32>> {
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();

        if sum_exp == 0.0 {
            return Err(TrustformersError::tensor_op_error(
                "Zero sum in softmax",
                "softmax",
            ));
        }

        Ok(exp_logits.iter().map(|&x| x / sum_exp).collect())
    }

    /// Sample from probability distribution
    pub fn sample_from_probs(&self, probs: &[f32]) -> Result<usize> {
        let mut rng = thread_rng();
        let sample: f32 = rng.gen();
        let mut cumsum = 0.0;

        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if sample <= cumsum {
                return Ok(i);
            }
        }

        // Fallback to last token
        Ok(probs.len().saturating_sub(1))
    }
}
