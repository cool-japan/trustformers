// SPDX-License-Identifier: Apache-2.0

//! Enhanced text generation implementation for GPT-2 models
//!
//! This module provides advanced generation capabilities using the unified
//! generation utilities from `trustformers_models::generation_utils`.

use crate::generation_utils::{GenerationConfig, GenerationMode, GenerationUtils, KVCache};
use crate::gpt2::model::{last_token_logits, Gpt2LMHeadModel, Gpt2LMOutput};
use scirs2_core::random::*;
use trustformers_core::{
    errors::{Result, TrustformersError},
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
                        indexed_logits.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
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
            GenerationMode::ContrastiveSearch { top_k, alpha } => {
                let result = self.generate_contrastive_internal(
                    input_ids, max_length, *top_k, *alpha, &config,
                )?;
                Ok(vec![result])
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

/// Cosine similarity between two equal-length vectors.
///
/// Zero vectors have no direction, so their similarity is defined as 0 rather
/// than producing a NaN that would silently win an `argmax`.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denominator = norm_a.sqrt() * norm_b.sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        dot / denominator
    }
}

impl Gpt2LMHeadModel {
    /// Contrastive search decoding (Su et al., 2022).
    ///
    /// At every step the `top_k` most likely continuations are re-scored with a
    /// degeneration penalty:
    ///
    /// ```text
    ///   score(v) = (1 - alpha) * p(v | prefix)
    ///            - alpha * max_j cos(h_v, h_j)
    /// ```
    ///
    /// where `h_v` is the representation the model gives the candidate token once
    /// it is appended and `h_j` ranges over the representations of the tokens
    /// already in the prefix. A candidate that merely repeats something already
    /// present therefore loses to a slightly less likely but more novel one.
    ///
    /// This used to be an error arm: the `ContrastiveSearch` variant was publicly
    /// constructible and documented, but calling it returned
    /// `"Contrastive search not yet implemented for GPT-2"`.
    ///
    /// # Errors
    ///
    /// Fails when `top_k` is zero, when `alpha` is outside `[0, 1]`, when the
    /// prompt is empty, or when a forward pass fails.
    fn generate_contrastive_internal(
        &self,
        input_ids: Vec<u32>,
        max_length: usize,
        top_k: usize,
        alpha: f32,
        config: &GenerationConfig,
    ) -> Result<Vec<u32>> {
        if top_k == 0 {
            return Err(TrustformersError::model_error(
                "contrastive search requires top_k >= 1".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&alpha) {
            return Err(TrustformersError::model_error(format!(
                "contrastive search requires alpha in [0, 1], got {alpha}"
            )));
        }
        if input_ids.is_empty() {
            return Err(TrustformersError::model_error(
                "contrastive search requires a non-empty prompt".to_string(),
            ));
        }

        // `should_stop` measures the *completion* against `max_new_tokens`, so it
        // needs the prompt boundary — not the current length, which would make
        // the completion look empty on every iteration and disable both
        // `max_new_tokens` and the stop-sequence match entirely.
        let prompt_length = input_ids.len();
        let mut generated = input_ids;

        while generated.len() < max_length {
            if GenerationUtils::should_stop(&generated, config, prompt_length) {
                break;
            }

            let (mut logits, context_states) = self.logits_and_hidden_states(&generated)?;

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

            let probabilities = GenerationUtils::softmax(&logits);
            let mut ranked: Vec<(usize, f32)> = probabilities.iter().copied().enumerate().collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            ranked.truncate(top_k.min(ranked.len()));

            let mut best: Option<(u32, f32)> = None;
            for (token_id, probability) in ranked {
                let candidate_token = token_id as u32;
                let mut candidate_sequence = generated.clone();
                candidate_sequence.push(candidate_token);

                let (_, candidate_states) = self.logits_and_hidden_states(&candidate_sequence)?;
                let candidate_state = candidate_states.last().ok_or_else(|| {
                    TrustformersError::model_error(
                        "forward pass produced no hidden states for the candidate".to_string(),
                    )
                })?;

                let max_similarity = context_states
                    .iter()
                    .map(|state| cosine_similarity(candidate_state, state))
                    .fold(f32::NEG_INFINITY, f32::max);
                let max_similarity = if max_similarity.is_finite() { max_similarity } else { 0.0 };

                let score = (1.0 - alpha) * probability - alpha * max_similarity;
                if best.is_none_or(|(_, best_score)| score > best_score) {
                    best = Some((candidate_token, score));
                }
            }

            let (next_token, _) = best.ok_or_else(|| {
                TrustformersError::model_error(
                    "contrastive search found no candidate tokens".to_string(),
                )
            })?;
            generated.push(next_token);

            if config.eos_token_id.is_some_and(|eos| eos == next_token) {
                break;
            }
        }

        Ok(generated)
    }

    /// Internal greedy generation with full config support
    fn generate_greedy_internal(
        &self,
        input_ids: Vec<u32>,
        max_length: usize,
        config: &GenerationConfig,
    ) -> Result<Vec<u32>> {
        // The prompt boundary, not the running length: `should_stop` counts the
        // completion against `max_new_tokens`.
        let prompt_length = input_ids.len();
        let mut generated = input_ids.clone();
        let mut kv_cache = if !config.no_kv_cache { Some(KVCache::new()) } else { None };

        while generated.len() < max_length {
            // Check stopping criteria
            if GenerationUtils::should_stop(&generated, config, prompt_length) {
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
        // The prompt boundary, not the running length: `should_stop` counts the
        // completion against `max_new_tokens`.
        let prompt_length = input_ids.len();
        let mut generated = input_ids.clone();
        let mut kv_cache = if !config.no_kv_cache { Some(KVCache::new()) } else { None };

        while generated.len() < max_length {
            // Check stopping criteria
            if GenerationUtils::should_stop(&generated, config, prompt_length) {
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

        // Every beam starts from the prompt, so the prompt boundary is shared.
        let prompt_length = input_ids.len();

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
                token_scores
                    .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

                // Create new candidates
                for (log_prob, token_idx) in token_scores.iter().take(num_beams) {
                    let new_score = beam.score + log_prob;
                    let mut new_tokens = beam.tokens.clone();
                    new_tokens.push(*token_idx as u32);

                    let mut new_beam = BeamHypothesis::new(new_tokens.clone(), new_score);

                    // Check if this beam should be marked as finished
                    if GenerationUtils::should_stop(&new_tokens, config, prompt_length) {
                        new_beam.finished = true;
                    }

                    candidates.push(new_beam);
                }
            }

            // Select top beams
            candidates.sort_by(|a, b| {
                let a_score = a.normalized_score(config.length_penalty);
                let b_score = b.normalized_score(config.length_penalty);
                b_score.partial_cmp(&a_score).unwrap_or(std::cmp::Ordering::Equal)
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
            b_score.partial_cmp(&a_score).unwrap_or(std::cmp::Ordering::Equal)
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

        // Extract last token logits.
        //
        // Shared with `Gpt2LMHeadModel`'s own generators (`generate`,
        // `generate_greedy`, `generate_greedy_with_cache`, `generate_beam_search`):
        // this used to be a private copy that matched `Tensor::F32` only, so every
        // `GenerativeModel` method - `generate_greedy`, `generate_top_k`,
        // `generate_top_p`, `generate_contrastive`, `generate_beam_search` - failed
        // with "Unsupported tensor type for logits" on a model whose weights had been
        // moved to the GPU with `weights_to_gpu`, while the inherent methods of the
        // same model happily generated. One decoder now serves both.
        last_token_logits(&output.logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation_utils::StoppingCriteria;
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
        // Use very small config to reduce memory pressure
        let mut gpt2_config = Gpt2Config::small();
        gpt2_config.vocab_size = 50; // Reduce from default
        gpt2_config.n_positions = 32; // Reduce from default
        gpt2_config.n_embd = 32; // Reduce from default
        gpt2_config.n_layer = 1; // Reduce from default
        gpt2_config.n_head = 2; // Reduce from default

        let model = Gpt2LMHeadModel::new(gpt2_config)?;

        let input_ids = vec![1, 2];
        let max_length = 5; // Reduce from 10

        // Test the interface
        let result = model.generate_greedy(input_ids, max_length);
        assert!(
            result.is_ok(),
            "Greedy generation failed: {:?}",
            result.err()
        );

        let generated = result?;
        assert!(generated.len() <= max_length);

        // Explicit cleanup
        drop(generated);
        drop(model);
        std::hint::black_box(());

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

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert!(config.max_length > 0);
        assert!(config.temperature > 0.0);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_generation_config_validation_valid() {
        let config = GenerationConfig {
            max_length: 100,
            temperature: 1.0,
            repetition_penalty: 1.0,
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_generation_config_custom_temperature() {
        let config = GenerationConfig {
            temperature: 0.5,
            ..Default::default()
        };
        assert!(config.validate().is_ok());
        assert!((config.temperature - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_generation_config_with_repetition_penalty() {
        let config = GenerationConfig {
            repetition_penalty: 1.5,
            ..Default::default()
        };
        assert!(config.validate().is_ok());
        assert!((config.repetition_penalty - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_generation_mode_greedy_debug() {
        let mode = GenerationMode::Greedy;
        let dbg = format!("{:?}", mode);
        assert!(dbg.contains("Greedy"));
    }

    #[test]
    fn test_generation_mode_beam_search_params() {
        let mode = GenerationMode::BeamSearch { num_beams: 5 };
        match mode {
            GenerationMode::BeamSearch { num_beams } => assert_eq!(num_beams, 5),
            _ => panic!("Expected BeamSearch"),
        }
    }

    #[test]
    fn test_generation_mode_top_k_params() {
        let mode = GenerationMode::TopK { k: 50 };
        match mode {
            GenerationMode::TopK { k } => assert_eq!(k, 50),
            _ => panic!("Expected TopK"),
        }
    }

    #[test]
    fn test_generation_mode_top_p_params() {
        let mode = GenerationMode::TopP { p: 0.95 };
        match mode {
            GenerationMode::TopP { p } => assert!((p - 0.95).abs() < f32::EPSILON),
            _ => panic!("Expected TopP"),
        }
    }

    #[test]
    fn test_gpt2_small_config() {
        let config = Gpt2Config::small();
        assert!(config.vocab_size > 0);
        assert!(config.n_embd > 0);
        assert!(config.n_layer > 0);
        assert!(config.n_head > 0);
    }

    #[test]
    fn test_gpt2_model_creation_tiny() -> Result<()> {
        let mut config = Gpt2Config::small();
        config.vocab_size = 20;
        config.n_positions = 16;
        config.n_embd = 16;
        config.n_layer = 1;
        config.n_head = 2;
        let model = Gpt2LMHeadModel::new(config)?;
        assert!(model.num_parameters() > 0);
        Ok(())
    }

    #[test]
    fn test_generation_config_max_new_tokens() {
        let config = GenerationConfig {
            max_new_tokens: Some(10),
            ..Default::default()
        };
        assert_eq!(config.max_new_tokens, Some(10));
    }

    #[test]
    fn test_greedy_generation_tiny_model() -> Result<()> {
        let mut config = Gpt2Config::small();
        config.vocab_size = 30;
        config.n_positions = 16;
        config.n_embd = 16;
        config.n_layer = 1;
        config.n_head = 2;
        let model = Gpt2LMHeadModel::new(config)?;
        let input_ids = vec![1, 2, 3];
        let result = model.generate_greedy(input_ids, 5);
        assert!(result.is_ok());
        let generated = result?;
        assert!(generated.len() <= 5);
        drop(generated);
        drop(model);
        std::hint::black_box(());
        Ok(())
    }

    #[test]
    fn test_generation_config_clone() {
        let config = GenerationConfig {
            max_length: 200,
            temperature: 0.7,
            repetition_penalty: 1.3,
            ..Default::default()
        };
        let cloned = config.clone();
        assert_eq!(cloned.max_length, 200);
        assert!((cloned.temperature - 0.7).abs() < f32::EPSILON);
        assert!((cloned.repetition_penalty - 1.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_generation_mode_all_variants_constructable() {
        let modes: Vec<GenerationMode> = vec![
            GenerationMode::Greedy,
            GenerationMode::BeamSearch { num_beams: 4 },
            GenerationMode::TopK { k: 40 },
            GenerationMode::TopP { p: 0.9 },
        ];
        assert_eq!(modes.len(), 4);
    }

    #[test]
    fn test_generation_config_with_beam_search() {
        let config = GenerationConfig {
            mode: GenerationMode::BeamSearch { num_beams: 4 },
            max_length: 50,
            ..Default::default()
        };
        assert!(matches!(
            config.mode,
            GenerationMode::BeamSearch { num_beams: 4 }
        ));
    }

    // ── Prompt boundary passed to `should_stop` ─────────────────────────────

    /// A GPT-2 small enough to run a generation loop in a unit test.
    fn tiny_lm_head_model() -> Gpt2LMHeadModel {
        let mut config = Gpt2Config::small();
        config.vocab_size = 16;
        config.n_positions = 32;
        config.n_embd = 16;
        config.n_layer = 1;
        config.n_head = 2;
        Gpt2LMHeadModel::new(config).expect("tiny GPT-2 must build")
    }

    /// Regression: every generation loop passed `generated.len()` — the running
    /// length — as `should_stop`'s `prompt_length`. The completion slice
    /// `sequence[prompt_length..]` was therefore *always empty*, so no
    /// [`StoppingCriteria`] that inspects generated tokens could ever fire and
    /// generation ran to `max_length` regardless.
    ///
    /// `AnyToken` over the whole vocabulary matches whatever the model emits, so
    /// the assertion holds for any weights: generation must stop after exactly
    /// one new token.
    #[test]
    fn greedy_generation_honours_a_stopping_criterion_on_the_first_new_token() {
        let model = tiny_lm_head_model();
        let prompt = vec![1u32, 2, 3];
        let config = GenerationConfig {
            max_length: prompt.len() + 8,
            mode: GenerationMode::Greedy,
            stopping_criteria: vec![StoppingCriteria::AnyToken {
                token_ids: (0..16u32).collect(),
            }],
            ..GenerationConfig::greedy()
        };

        let sequences = model
            .generate_with_config(prompt.clone(), config)
            .expect("generation must succeed");
        let generated = sequences.first().expect("one sequence must be returned");

        assert_eq!(
            generated.len(),
            prompt.len() + 1,
            "the stopping criterion must fire on the first generated token, got {} tokens",
            generated.len()
        );
    }

    /// The same defect on the sampling loop.
    #[test]
    fn sampled_generation_honours_a_stopping_criterion_on_the_first_new_token() {
        let model = tiny_lm_head_model();
        let prompt = vec![4u32, 5];
        let config = GenerationConfig {
            max_length: prompt.len() + 8,
            mode: GenerationMode::TopK { k: 4 },
            stopping_criteria: vec![StoppingCriteria::AnyToken {
                token_ids: (0..16u32).collect(),
            }],
            ..GenerationConfig::greedy()
        };

        let sequences = model
            .generate_with_config(prompt.clone(), config)
            .expect("generation must succeed");
        let generated = sequences.first().expect("one sequence must be returned");

        assert_eq!(
            generated.len(),
            prompt.len() + 1,
            "the stopping criterion must fire on the first sampled token, got {} tokens",
            generated.len()
        );
    }

    /// Contrastive search shares the loop shape and shared the defect.
    #[test]
    fn contrastive_generation_honours_a_stopping_criterion_on_the_first_new_token() {
        let model = tiny_lm_head_model();
        let prompt = vec![6u32, 7];
        let config = GenerationConfig {
            max_length: prompt.len() + 6,
            mode: GenerationMode::ContrastiveSearch {
                top_k: 3,
                alpha: 0.5,
            },
            stopping_criteria: vec![StoppingCriteria::AnyToken {
                token_ids: (0..16u32).collect(),
            }],
            ..GenerationConfig::greedy()
        };

        let sequences = model
            .generate_with_config(prompt.clone(), config)
            .expect("generation must succeed");
        let generated = sequences.first().expect("one sequence must be returned");

        assert_eq!(
            generated.len(),
            prompt.len() + 1,
            "the stopping criterion must fire on the first contrastive token, got {} tokens",
            generated.len()
        );
    }

    /// A prompt token that happens to sit in the stop set must not end
    /// generation before anything is produced — that is the other half of the
    /// prompt/completion split.
    #[test]
    fn a_stop_token_inside_the_prompt_does_not_end_generation_immediately() {
        let model = tiny_lm_head_model();
        let prompt = vec![9u32, 9, 9];
        let config = GenerationConfig {
            max_length: prompt.len() + 3,
            mode: GenerationMode::Greedy,
            stopping_criteria: vec![StoppingCriteria::AnyToken { token_ids: vec![9] }],
            ..GenerationConfig::greedy()
        };

        let sequences = model
            .generate_with_config(prompt.clone(), config)
            .expect("generation must succeed");
        let generated = sequences.first().expect("one sequence must be returned");

        assert!(
            generated.len() > prompt.len(),
            "a stop token inside the prompt must not suppress the completion"
        );
    }
}
