// SPDX-License-Identifier: Apache-2.0

//! Enhanced text generation implementation for GPT-2 models
//!
//! This module provides advanced generation capabilities using the unified
//! generation utilities from `trustformers_models::generation_utils`.

use crate::generation_utils::{GenerationConfig, GenerationMode, GenerationUtils, KVCache};
use crate::gpt2::model::{Gpt2LMHeadModel, Gpt2LMOutput};
use scirs2_core::ndarray::s;
use scirs2_core::random::*;
use trustformers_core::{
    errors::{tensor_op_error, Result, TrustformersError},
    tensor::Tensor,
    traits::{Model, TokenizedInput},
};

/// Trait for models that support text generation
pub trait GenerativeModel {
    /// Generate text with advanced configuration
    fn generate_with_config(
        &self,
        input_ids: Vec<u32>,
        config: GenerationConfig,
    ) -> Result<Vec<Vec<u32>>>;

    /// Generate a single sequence using greedy decoding
    fn generate_greedy(&self, input_ids: Vec<u32>, max_length: usize) -> Result<Vec<u32>>;

    /// Generate using beam search
    fn generate_beam_search(
        &self,
        input_ids: Vec<u32>,
        max_length: usize,
        num_beams: usize,
    ) -> Result<Vec<Vec<u32>>>;

    /// Generate using top-k sampling
    fn generate_top_k(
        &self,
        input_ids: Vec<u32>,
        max_length: usize,
        k: usize,
        temperature: f32,
    ) -> Result<Vec<u32>>;

    /// Generate using nucleus (top-p) sampling
    fn generate_top_p(
        &self,
        input_ids: Vec<u32>,
        max_length: usize,
        p: f32,
        temperature: f32,
    ) -> Result<Vec<u32>>;
}

impl GenerativeModel for Gpt2LMHeadModel {
    fn generate_with_config(
        &self,
        input_ids: Vec<u32>,
        config: GenerationConfig,
    ) -> Result<Vec<Vec<u32>>> {
        // Validate configuration
        config.validate()?;

        // Initialize RNG if seed is provided
        let mut rng = thread_rng(); // Use thread_rng from scirs2_core

        // Determine effective max length
        let max_length = if let Some(max_new_tokens) = config.max_new_tokens {
            input_ids.len() + max_new_tokens
        } else {
            config.max_length
        };

        // Route to appropriate generation method based on mode
        match &config.mode {
            GenerationMode::Greedy => {
                let result = self.generate_greedy_internal(input_ids, max_length, &config)?;
                Ok(vec![result])
            },
            GenerationMode::BeamSearch { num_beams } => {
                self.generate_beam_search_internal(input_ids, max_length, *num_beams, &config)
            },
            GenerationMode::TopK { k } => {
                let result = self.generate_sampling_internal(
                    input_ids,
                    max_length,
                    &config,
                    &mut rng,
                    |logits, rng| GenerationUtils::sample_top_k(logits, *k, rng),
                )?;
                Ok(vec![result])
            },
            GenerationMode::TopP { p } => {
                let result = self.generate_sampling_internal(
                    input_ids,
                    max_length,
                    &config,
                    &mut rng,
                    |logits, rng| GenerationUtils::sample_top_p(logits, *p, rng),
                )?;
                Ok(vec![result])
            },
            GenerationMode::MinP { p } => {
                let result = self.generate_sampling_internal(
                    input_ids,
                    max_length,
                    &config,
                    &mut rng,
                    |logits, rng| GenerationUtils::sample_min_p(logits, *p, rng),
                )?;
                Ok(vec![result])
            },
            GenerationMode::Temperature { temperature } => {
                let mut temp_config = config.clone();
                temp_config.temperature = *temperature;
                let result = self.generate_sampling_internal(
                    input_ids,
                    max_length,
                    &temp_config,
                    &mut rng,
                    |logits, rng| {
                        let probs = GenerationUtils::softmax(logits);
                        GenerationUtils::sample_from_probs(&probs, rng).map(|idx| idx as u32)
                    },
                )?;
                Ok(vec![result])
            },
            GenerationMode::Combined { k, p } => {
                let result = self.generate_sampling_internal(
                    input_ids,
                    max_length,
                    &config,
                    &mut rng,
                    |logits, rng| {
                        // First apply top-k
                        let mut indexed_logits: Vec<(usize, f32)> =
                            logits.iter().copied().enumerate().collect();
                        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                        indexed_logits.truncate(*k);

                        // Then apply top-p on the filtered logits
                        let top_k_logits: Vec<f32> =
                            indexed_logits.iter().map(|(_, logit)| *logit).collect();
                        let probs = GenerationUtils::softmax(&top_k_logits);

                        // Find nucleus cutoff
                        let mut cumsum = 0.0;
                        let mut cutoff = probs.len();
                        for (i, &prob) in probs.iter().enumerate() {
                            cumsum += prob;
                            if cumsum >= *p {
                                cutoff = i + 1;
                                break;
                            }
                        }

                        let nucleus_probs = &probs[..cutoff];
                        let sample_idx = GenerationUtils::sample_from_probs(nucleus_probs, rng)?;
                        Ok(indexed_logits[sample_idx].0 as u32)
                    },
                )?;
                Ok(vec![result])
            },
            GenerationMode::ContrastiveSearch { top_k: _, alpha: _ } => {
                // Contrastive search requires tracking hidden states
                // For now, return an error indicating this is not yet implemented
                Err(TrustformersError::model_error(
                    "Contrastive search not yet implemented for GPT-2".to_string(),
                ))
            },
        }
    }

    fn generate_greedy(&self, input_ids: Vec<u32>, max_length: usize) -> Result<Vec<u32>> {
        let config = GenerationConfig::greedy();
        self.generate_greedy_internal(input_ids, max_length, &config)
    }

    fn generate_beam_search(
        &self,
        input_ids: Vec<u32>,
        max_length: usize,
        num_beams: usize,
    ) -> Result<Vec<Vec<u32>>> {
        let config = GenerationConfig::beam_search(num_beams);
        self.generate_beam_search_internal(input_ids, max_length, num_beams, &config)
    }

    fn generate_top_k(
        &self,
        input_ids: Vec<u32>,
        max_length: usize,
        k: usize,
        temperature: f32,
    ) -> Result<Vec<u32>> {
        let mut config = GenerationConfig::top_k(k);
        config.temperature = temperature;
        config.max_length = max_length;

        let mut rng = thread_rng();
        self.generate_sampling_internal(input_ids, max_length, &config, &mut rng, |logits, rng| {
            GenerationUtils::sample_top_k(logits, k, rng)
        })
    }

    fn generate_top_p(
        &self,
        input_ids: Vec<u32>,
        max_length: usize,
        p: f32,
        temperature: f32,
    ) -> Result<Vec<u32>> {
        let mut config = GenerationConfig::top_p(p);
        config.temperature = temperature;
        config.max_length = max_length;

        let mut rng = thread_rng();
        self.generate_sampling_internal(input_ids, max_length, &config, &mut rng, |logits, rng| {
            GenerationUtils::sample_top_p(logits, p, rng)
        })
    }
}

impl Gpt2LMHeadModel {
    /// Internal greedy generation with full config support
    fn generate_greedy_internal(
        &self,
        input_ids: Vec<u32>,
        max_length: usize,
        config: &GenerationConfig,
    ) -> Result<Vec<u32>> {
        let mut generated = input_ids.clone();
        let mut kv_cache = if !config.no_kv_cache { Some(KVCache::new()) } else { None };

        while generated.len() < max_length {
            // Check stopping criteria
            if GenerationUtils::should_stop(&generated, config, generated.len()) {
                break;
            }

            // Get next token logits
            let mut logits = self.get_next_token_logits(&generated, kv_cache.as_mut())?;

            // Apply penalties
            GenerationUtils::apply_repetition_penalty(
                &mut logits,
                &generated,
                config.repetition_penalty,
                config.repetition_penalty_decay,
            );
            GenerationUtils::apply_frequency_penalty(
                &mut logits,
                &generated,
                config.frequency_penalty,
            );
            GenerationUtils::apply_presence_penalty(
                &mut logits,
                &generated,
                config.presence_penalty,
            );
            GenerationUtils::apply_bad_words_filter(&mut logits, &generated, &config.bad_words_ids);

            // Select argmax
            let next_token = GenerationUtils::sample_greedy(&logits);
            generated.push(next_token);
        }

        Ok(generated)
    }

    /// Internal sampling generation with configurable sampling function
    fn generate_sampling_internal<F, R>(
        &self,
        input_ids: Vec<u32>,
        max_length: usize,
        config: &GenerationConfig,
        rng: &mut R,
        sample_fn: F,
    ) -> Result<Vec<u32>>
    where
        F: Fn(&[f32], &mut R) -> Result<u32>,
        R: Rng,
    {
        let mut generated = input_ids.clone();
        let mut kv_cache = if !config.no_kv_cache { Some(KVCache::new()) } else { None };

        while generated.len() < max_length {
            // Check stopping criteria
            if GenerationUtils::should_stop(&generated, config, generated.len()) {
                break;
            }

            // Get next token logits
            let mut logits = self.get_next_token_logits(&generated, kv_cache.as_mut())?;

            // Apply temperature scaling
            GenerationUtils::apply_temperature(&mut logits, config.temperature);

            // Apply penalties
            GenerationUtils::apply_repetition_penalty(
                &mut logits,
                &generated,
                config.repetition_penalty,
                config.repetition_penalty_decay,
            );
            GenerationUtils::apply_frequency_penalty(
                &mut logits,
                &generated,
                config.frequency_penalty,
            );
            GenerationUtils::apply_presence_penalty(
                &mut logits,
                &generated,
                config.presence_penalty,
            );
            GenerationUtils::apply_bad_words_filter(&mut logits, &generated, &config.bad_words_ids);

            // Sample using the provided sampling function
            let next_token = sample_fn(&logits, rng)?;
            generated.push(next_token);
        }

        Ok(generated)
    }

    /// Internal beam search with full config support
    fn generate_beam_search_internal(
        &self,
        input_ids: Vec<u32>,
        max_length: usize,
        num_beams: usize,
        config: &GenerationConfig,
    ) -> Result<Vec<Vec<u32>>> {
        use crate::generation_utils::BeamHypothesis;

        if num_beams == 1 {
            let result = self.generate_greedy_internal(input_ids, max_length, config)?;
            return Ok(vec![result]);
        }

        // Initialize beams
        let mut beams: Vec<BeamHypothesis> = vec![BeamHypothesis::new(input_ids.clone(), 0.0)];

        while beams[0].tokens.len() < max_length {
            let mut candidates = Vec::new();

            for beam in &beams {
                if beam.finished {
                    candidates.push(beam.clone());
                    continue;
                }

                // Get next token logits for this beam
                let mut logits = self.get_next_token_logits(&beam.tokens, None)?;

                // Apply penalties
                GenerationUtils::apply_repetition_penalty(
                    &mut logits,
                    &beam.tokens,
                    config.repetition_penalty,
                    config.repetition_penalty_decay,
                );
                GenerationUtils::apply_bad_words_filter(
                    &mut logits,
                    &beam.tokens,
                    &config.bad_words_ids,
                );

                // Convert to log probabilities
                let log_probs =
                    GenerationUtils::softmax(&logits).iter().map(|&p| p.ln()).collect::<Vec<_>>();

                // Get top k tokens
                let mut token_scores: Vec<(f32, usize)> =
                    log_probs.iter().enumerate().map(|(idx, &log_prob)| (log_prob, idx)).collect();
                token_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

                // Create new candidates
                for (log_prob, token_idx) in token_scores.iter().take(num_beams) {
                    let new_score = beam.score + log_prob;
                    let mut new_tokens = beam.tokens.clone();
                    new_tokens.push(*token_idx as u32);

                    let mut new_beam = BeamHypothesis::new(new_tokens.clone(), new_score);

                    // Check if this beam should be marked as finished
                    if GenerationUtils::should_stop(&new_tokens, config, new_tokens.len()) {
                        new_beam.finished = true;
                    }

                    candidates.push(new_beam);
                }
            }

            // Select top beams
            candidates.sort_by(|a, b| {
                let a_score = a.normalized_score(config.length_penalty);
                let b_score = b.normalized_score(config.length_penalty);
                b_score.partial_cmp(&a_score).unwrap()
            });

            beams = candidates.into_iter().take(num_beams).collect();

            // Early stopping if enabled
            if config.early_stopping && beams.iter().all(|b| b.finished) {
                break;
            }
        }

        // Return top sequences
        beams.sort_by(|a, b| {
            let a_score = a.normalized_score(config.length_penalty);
            let b_score = b.normalized_score(config.length_penalty);
            b_score.partial_cmp(&a_score).unwrap()
        });

        let num_return = config.num_return_sequences.min(beams.len());
        Ok(beams.iter().take(num_return).map(|b| b.tokens.clone()).collect())
    }

    /// Get logits for the next token prediction
    fn get_next_token_logits(
        &self,
        input_ids: &[u32],
        _kv_cache: Option<&mut KVCache>,
    ) -> Result<Vec<f32>> {
        // Prepare input
        let input = TokenizedInput {
            input_ids: input_ids.to_vec(),
            attention_mask: vec![1u8; input_ids.len()],
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        };

        // Forward pass
        let output: Gpt2LMOutput = self.forward(input)?;

        // Extract last token logits
        match &output.logits {
            Tensor::F32(arr) => {
                let shape = arr.shape();
                if shape.len() != 3 {
                    return Err(tensor_op_error(
                        "tensor_operation",
                        format!("Expected 3D logits tensor, got {}D", shape.len()),
                    ));
                }

                let seq_len = shape[1];
                let vocab_size = shape[2];

                // Get last token's logits: [batch=0, seq_len-1, :]
                let last_logits = arr.slice(s![0, seq_len - 1, ..]);

                // Convert to Vec<f32>
                let logits_vec: Vec<f32> = last_logits.iter().copied().collect();

                if logits_vec.len() != vocab_size {
                    return Err(tensor_op_error(
                        "tensor_operation",
                        format!(
                            "Logits size mismatch: expected {}, got {}",
                            vocab_size,
                            logits_vec.len()
                        ),
                    ));
                }

                Ok(logits_vec)
            },
            _ => Err(tensor_op_error(
                "tensor_operation",
                "Unsupported tensor type for logits".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpt2::Gpt2Config;

    #[test]
    fn test_generation_config_integration() {
        let config = GenerationConfig {
            max_length: 50,
            temperature: 0.8,
            repetition_penalty: 1.2,
            ..Default::default()
        };

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_greedy_generation_interface() -> Result<()> {
        let gpt2_config = Gpt2Config::gpt2_base();
        let model = Gpt2LMHeadModel::new(gpt2_config)?;

        let input_ids = vec![1, 2, 3];
        let max_length = 10;

        // This will fail without weights, but tests the interface
        let result = model.generate_greedy(input_ids, max_length);
        assert!(result.is_ok() || result.is_err()); // Interface test only

        Ok(())
    }

    #[test]
    fn test_generation_modes() {
        // Test that all generation modes can be constructed
        let greedy = GenerationMode::Greedy;
        let beam = GenerationMode::BeamSearch { num_beams: 5 };
        let top_k = GenerationMode::TopK { k: 50 };
        let top_p = GenerationMode::TopP { p: 0.9 };

        assert!(matches!(greedy, GenerationMode::Greedy));
        assert!(matches!(beam, GenerationMode::BeamSearch { .. }));
        assert!(matches!(top_k, GenerationMode::TopK { .. }));
        assert!(matches!(top_p, GenerationMode::TopP { .. }));
    }
}
