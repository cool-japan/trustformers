use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;

use super::cache::KVCache;
use super::config::{CFGConfig, GenerationConfig};

/// A boxed model callback returning next-token logits and the updated cache.
pub type BoxedLogitsFn<'a> =
    Box<dyn Fn(&[usize], Option<&KVCache>) -> Result<(Tensor, Option<KVCache>)> + 'a>;

/// Classifier-Free Guidance generator for improved text generation
pub struct CFGGenerator {
    /// The underlying strategy-aware text generator.
    base_generator: super::core::TextGenerator,
    /// Classifier-free guidance settings extracted from the generation config.
    cfg_config: Option<CFGConfig>,
}

impl CFGGenerator {
    pub fn new(config: GenerationConfig, vocab_size: usize) -> Result<Self> {
        let cfg_config = config.guided_generation.as_ref().and_then(|g| g.cfg.clone());

        Ok(Self {
            base_generator: super::core::TextGenerator::new(config, vocab_size),
            cfg_config,
        })
    }

    /// Create a CFG generator whose sampling is driven by a deterministic seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.base_generator = self.base_generator.with_seed(seed);
        self
    }

    /// Access the underlying text generator.
    pub fn base_generator(&self) -> &super::core::TextGenerator {
        &self.base_generator
    }

    /// Generate text using Classifier-Free Guidance.
    ///
    /// The guided logits are turned into a token with the *configured*
    /// generation strategy, so temperature / top-k / top-p sampling now takes
    /// effect here (this path used to argmax unconditionally).  Beam search and
    /// contrastive search need multi-step lookahead that classifier-free
    /// guidance cannot supply through this interface, so selecting either of
    /// them returns a `NotImplemented` error instead of quietly decoding
    /// greedily.
    pub fn generate_with_cfg(
        &self,
        input_ids: &[usize],
        conditional_logits_fn: impl Fn(&[usize], Option<&KVCache>) -> Result<(Tensor, Option<KVCache>)>,
        unconditional_logits_fn: impl Fn(
            &[usize],
            Option<&KVCache>,
        ) -> Result<(Tensor, Option<KVCache>)>,
    ) -> Result<Vec<Vec<usize>>> {
        let cfg_config = match self.cfg_config.as_ref() {
            Some(config) => config,
            None => return self.base_generator.generate(input_ids, conditional_logits_fn),
        };
        let mut sequences = vec![input_ids.to_vec()];
        let mut conditional_cache =
            if self.base_generator.config.use_cache { Some(KVCache::new()) } else { None };
        let mut unconditional_cache =
            if self.base_generator.config.use_cache { Some(KVCache::new()) } else { None };

        let budget = self.base_generator.max_new_tokens(input_ids.len());
        let mut rng = self.base_generator.new_rng();

        for step in 0..budget {
            // Get conditional logits (with prompt)
            let (conditional_logits, new_conditional_cache) =
                conditional_logits_fn(&sequences[0], conditional_cache.as_ref())?;
            conditional_cache = new_conditional_cache;

            // Get unconditional logits (without prompt or with unconditional prompt)
            let (unconditional_logits, new_unconditional_cache) =
                unconditional_logits_fn(&sequences[0], unconditional_cache.as_ref())?;
            unconditional_cache = new_unconditional_cache;

            // Apply Classifier-Free Guidance
            let guided_logits = self.apply_cfg_guidance(
                &conditional_logits,
                &unconditional_logits,
                cfg_config.guidance_scale,
                cfg_config.dynamic_thresholding,
                cfg_config.threshold_percentile,
            )?;

            // Sample from the guided logits
            let next_token =
                self.base_generator.select_next_token(&guided_logits, &mut rng, &sequences[0])?;
            sequences[0].push(next_token);

            // Check if generation should stop
            if self.base_generator.should_stop(&sequences[0], next_token, step + 1) {
                break;
            }
        }

        Ok(sequences)
    }

    /// Apply CFG guidance to combine conditional and unconditional logits
    fn apply_cfg_guidance(
        &self,
        conditional_logits: &Tensor,
        unconditional_logits: &Tensor,
        guidance_scale: f32,
        dynamic_thresholding: bool,
        threshold_percentile: f32,
    ) -> Result<Tensor> {
        match (conditional_logits, unconditional_logits) {
            (Tensor::F32(cond_arr), Tensor::F32(uncond_arr)) => {
                let cond_data: Vec<f32> = cond_arr.iter().cloned().collect();
                let uncond_data: Vec<f32> = uncond_arr.iter().cloned().collect();

                if cond_data.len() != uncond_data.len() {
                    return Err(TrustformersError::tensor_op_error(
                        "Conditional and unconditional logits must have same length",
                        "apply_cfg_guidance",
                    ));
                }

                // Apply CFG formula: logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                let mut guided_logits: Vec<f32> = uncond_data
                    .iter()
                    .zip(cond_data.iter())
                    .map(|(&uncond, &cond)| uncond + guidance_scale * (cond - uncond))
                    .collect();

                // Apply dynamic thresholding if enabled
                if dynamic_thresholding {
                    guided_logits =
                        self.apply_dynamic_thresholding(guided_logits, threshold_percentile)?;
                }

                Tensor::from_vec(guided_logits, &[cond_data.len()])
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor types for CFG guidance",
                "apply_cfg_guidance",
            )),
        }
    }

    /// Apply dynamic thresholding to prevent extreme values
    fn apply_dynamic_thresholding(
        &self,
        mut logits: Vec<f32>,
        percentile: f32,
    ) -> Result<Vec<f32>> {
        if logits.is_empty() {
            return Err(TrustformersError::invalid_input(
                "dynamic thresholding requires at least one logit".to_string(),
            ));
        }
        if !percentile.is_finite() || !(0.0..=1.0).contains(&percentile) {
            return Err(TrustformersError::invalid_input(format!(
                "threshold_percentile must lie in [0, 1], got {percentile}"
            )));
        }

        // Calculate the percentile threshold
        let mut sorted_abs_logits: Vec<f32> = logits.iter().map(|&x| x.abs()).collect();
        sorted_abs_logits.sort_by(|a, b| a.total_cmp(b));

        let threshold_idx = ((sorted_abs_logits.len() as f32 * percentile) as usize)
            .min(sorted_abs_logits.len() - 1);
        let threshold = sorted_abs_logits[threshold_idx];

        // Clamp values that exceed the threshold
        for logit in &mut logits {
            *logit = logit.clamp(-threshold, threshold);
        }

        Ok(logits)
    }

    /// Generate text with negative prompting (avoiding certain content)
    pub fn generate_with_negative_prompt(
        &self,
        input_ids: &[usize],
        positive_logits_fn: impl Fn(&[usize], Option<&KVCache>) -> Result<(Tensor, Option<KVCache>)>,
        negative_logits_fn: impl Fn(&[usize], Option<&KVCache>) -> Result<(Tensor, Option<KVCache>)>,
        negative_scale: f32,
    ) -> Result<Vec<Vec<usize>>> {
        let mut sequences = vec![input_ids.to_vec()];
        let mut positive_cache =
            if self.base_generator.config.use_cache { Some(KVCache::new()) } else { None };
        let mut negative_cache =
            if self.base_generator.config.use_cache { Some(KVCache::new()) } else { None };

        let budget = self.base_generator.max_new_tokens(input_ids.len());
        let mut rng = self.base_generator.new_rng();

        for step in 0..budget {
            // Get positive logits (what we want)
            let (positive_logits, new_positive_cache) =
                positive_logits_fn(&sequences[0], positive_cache.as_ref())?;
            positive_cache = new_positive_cache;

            // Get negative logits (what we want to avoid)
            let (negative_logits, new_negative_cache) =
                negative_logits_fn(&sequences[0], negative_cache.as_ref())?;
            negative_cache = new_negative_cache;

            // Apply negative guidance: positive_logits - negative_scale * negative_logits
            let guided_logits =
                self.apply_negative_guidance(&positive_logits, &negative_logits, negative_scale)?;

            // Sample from the guided logits
            let next_token =
                self.base_generator.select_next_token(&guided_logits, &mut rng, &sequences[0])?;
            sequences[0].push(next_token);

            // Check if generation should stop
            if self.base_generator.should_stop(&sequences[0], next_token, step + 1) {
                break;
            }
        }

        Ok(sequences)
    }

    /// Apply negative guidance to subtract unwanted content
    fn apply_negative_guidance(
        &self,
        positive_logits: &Tensor,
        negative_logits: &Tensor,
        negative_scale: f32,
    ) -> Result<Tensor> {
        match (positive_logits, negative_logits) {
            (Tensor::F32(pos_arr), Tensor::F32(neg_arr)) => {
                let pos_data: Vec<f32> = pos_arr.iter().cloned().collect();
                let neg_data: Vec<f32> = neg_arr.iter().cloned().collect();

                if pos_data.len() != neg_data.len() {
                    return Err(TrustformersError::tensor_op_error(
                        "Positive and negative logits must have same length",
                        "apply_negative_guidance",
                    ));
                }

                // Apply negative guidance formula: pos_logits - negative_scale * neg_logits
                let guided_logits: Vec<f32> = pos_data
                    .iter()
                    .zip(neg_data.iter())
                    .map(|(&pos, &neg)| pos - negative_scale * neg)
                    .collect();

                Tensor::from_vec(guided_logits, &[pos_data.len()])
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor types for negative guidance",
                "apply_negative_guidance",
            )),
        }
    }

    /// Advanced CFG with multiple conditions and dynamic scaling
    pub fn generate_with_multi_condition_cfg(
        &self,
        input_ids: &[usize],
        condition_logits_fns: Vec<BoxedLogitsFn<'_>>,
        condition_scales: Vec<f32>,
        unconditional_logits_fn: impl Fn(
            &[usize],
            Option<&KVCache>,
        ) -> Result<(Tensor, Option<KVCache>)>,
    ) -> Result<Vec<Vec<usize>>> {
        if condition_logits_fns.len() != condition_scales.len() {
            return Err(TrustformersError::invalid_input(
                "Number of condition functions must match number of scales".to_string(),
            ));
        }

        let mut sequences = vec![input_ids.to_vec()];
        let mut condition_caches: Vec<Option<KVCache>> = (0..condition_logits_fns.len())
            .map(|_| if self.base_generator.config.use_cache { Some(KVCache::new()) } else { None })
            .collect();
        let mut unconditional_cache =
            if self.base_generator.config.use_cache { Some(KVCache::new()) } else { None };

        let budget = self.base_generator.max_new_tokens(input_ids.len());
        let mut rng = self.base_generator.new_rng();

        for step in 0..budget {
            // Get unconditional logits
            let (unconditional_logits, new_unconditional_cache) =
                unconditional_logits_fn(&sequences[0], unconditional_cache.as_ref())?;
            unconditional_cache = new_unconditional_cache;

            // Get all conditional logits
            let mut condition_logits = Vec::new();
            for (i, condition_fn) in condition_logits_fns.iter().enumerate() {
                let (logits, new_cache) =
                    condition_fn(&sequences[0], condition_caches[i].as_ref())?;
                condition_caches[i] = new_cache;
                condition_logits.push(logits);
            }

            // Apply multi-condition CFG
            let guided_logits = self.apply_multi_condition_cfg(
                &unconditional_logits,
                &condition_logits,
                &condition_scales,
            )?;

            // Sample from the guided logits
            let next_token =
                self.base_generator.select_next_token(&guided_logits, &mut rng, &sequences[0])?;
            sequences[0].push(next_token);

            // Check if generation should stop
            if self.base_generator.should_stop(&sequences[0], next_token, step + 1) {
                break;
            }
        }

        Ok(sequences)
    }

    /// Apply multi-condition CFG with weighted conditions
    fn apply_multi_condition_cfg(
        &self,
        unconditional_logits: &Tensor,
        condition_logits: &[Tensor],
        condition_scales: &[f32],
    ) -> Result<Tensor> {
        match unconditional_logits {
            Tensor::F32(uncond_arr) => {
                let uncond_data: Vec<f32> = uncond_arr.iter().cloned().collect();
                let mut guided_logits = uncond_data.clone();

                // Apply each condition with its scale
                for (condition_tensor, &scale) in
                    condition_logits.iter().zip(condition_scales.iter())
                {
                    match condition_tensor {
                        Tensor::F32(cond_arr) => {
                            let cond_data: Vec<f32> = cond_arr.iter().cloned().collect();

                            if cond_data.len() != uncond_data.len() {
                                return Err(TrustformersError::tensor_op_error(
                                    "All logits must have same length",
                                    "apply_multi_condition_cfg",
                                ));
                            }

                            // Add scaled difference: guided += scale * (cond - uncond)
                            for (i, &cond) in cond_data.iter().enumerate() {
                                guided_logits[i] += scale * (cond - uncond_data[i]);
                            }
                        },
                        _ => {
                            return Err(TrustformersError::tensor_op_error(
                                "Unsupported tensor type for condition",
                                "apply_multi_condition_cfg",
                            ))
                        },
                    }
                }

                Tensor::from_vec(guided_logits, &[uncond_data.len()])
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor type for unconditional logits",
                "apply_multi_condition_cfg",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::config::{CFGConfig, GenerationStrategy, GuidedGenerationConfig};

    fn tensor_1d(values: &[f32]) -> Tensor {
        Tensor::from_vec(values.to_vec(), &[values.len()]).expect("logits tensor")
    }

    fn constant_model(
        logits: Vec<f32>,
    ) -> impl Fn(&[usize], Option<&KVCache>) -> Result<(Tensor, Option<KVCache>)> {
        move |_tokens: &[usize], _cache: Option<&KVCache>| Ok((tensor_1d(&logits), None))
    }

    fn base_config(strategy: GenerationStrategy, max_new_tokens: usize) -> GenerationConfig {
        GenerationConfig {
            strategy,
            max_length: None,
            max_new_tokens: Some(max_new_tokens),
            min_length: None,
            ..Default::default()
        }
    }

    fn guided(cfg: CFGConfig) -> GuidedGenerationConfig {
        GuidedGenerationConfig {
            regex_pattern: None,
            grammar: None,
            json_schema: None,
            choice_list: None,
            max_violations: None,
            backtrack_on_violation: false,
            cfg: Some(cfg),
        }
    }

    fn cfg_config(guidance_scale: f32) -> CFGConfig {
        CFGConfig {
            guidance_scale,
            ..Default::default()
        }
    }

    #[test]
    fn test_cfg_generator_constructs_with_all_its_fields() {
        // Regression: `CFGGenerator::new` used to initialise fields the struct
        // did not declare, so the whole module failed to compile and was
        // commented out of `generation/mod.rs`.
        let mut config = base_config(GenerationStrategy::Greedy, 2);
        config.guided_generation = Some(guided(cfg_config(1.5)));
        let generator = CFGGenerator::new(config, 4).expect("construct");
        assert_eq!(generator.base_generator().vocab_size, 4);
    }

    #[test]
    fn test_guidance_moves_the_decoded_token_away_from_the_unconditional_argmax() {
        // Conditional prefers token 1, unconditional strongly prefers token 2.
        // guided = uncond + scale * (cond - uncond); with scale = 3 token 1 wins.
        let mut config = base_config(GenerationStrategy::Greedy, 1);
        config.guided_generation = Some(guided(cfg_config(3.0)));
        let generator = CFGGenerator::new(config, 4).expect("construct");

        let sequences = generator
            .generate_with_cfg(
                &[0],
                constant_model(vec![0.0, 2.0, 1.9, 0.0]),
                constant_model(vec![0.0, 0.0, 5.0, 0.0]),
            )
            .expect("generate");
        assert_eq!(sequences[0], vec![0, 1]);

        // With a guidance scale of 0 the guided logits are the unconditional
        // ones, so the unconditional argmax must win instead.
        let mut neutral = base_config(GenerationStrategy::Greedy, 1);
        neutral.guided_generation = Some(guided(cfg_config(0.0)));
        let neutral_generator = CFGGenerator::new(neutral, 4).expect("construct");
        let sequences = neutral_generator
            .generate_with_cfg(
                &[0],
                constant_model(vec![0.0, 2.0, 1.9, 0.0]),
                constant_model(vec![0.0, 0.0, 5.0, 0.0]),
            )
            .expect("generate");
        assert_eq!(sequences[0], vec![0, 2]);
    }

    #[test]
    fn test_cfg_honours_the_configured_sampling_strategy() {
        // Regression: this path used to argmax unconditionally, so a top-k
        // configuration silently decoded greedily.
        let mut config = base_config(
            GenerationStrategy::TopK {
                k: 2,
                temperature: 1.0,
            },
            120,
        );
        config.guided_generation = Some(guided(cfg_config(1.0)));
        let generator = CFGGenerator::new(config, 4).expect("construct").with_seed(4242);

        let sequences = generator
            .generate_with_cfg(
                &[0],
                constant_model(vec![0.1, 3.0, 2.9, 0.2]),
                constant_model(vec![0.1, 3.0, 2.9, 0.2]),
            )
            .expect("generate");

        let generated = &sequences[0][1..];
        assert!(
            generated.iter().all(|&token| token == 1 || token == 2),
            "sampled outside the top-2 set: {generated:?}"
        );
        assert!(
            generated.contains(&2),
            "top-k collapsed to greedy decoding: {generated:?}"
        );
    }

    #[test]
    fn test_cfg_rejects_strategies_it_cannot_express() {
        let mut config = base_config(GenerationStrategy::BeamSearch { num_beams: 2 }, 2);
        config.guided_generation = Some(guided(cfg_config(1.5)));
        let generator = CFGGenerator::new(config, 4).expect("construct");
        assert!(generator
            .generate_with_cfg(
                &[0],
                constant_model(vec![0.0, 1.0, 0.0, 0.0]),
                constant_model(vec![0.0, 1.0, 0.0, 0.0]),
            )
            .is_err());
    }

    #[test]
    fn test_without_cfg_config_the_base_strategy_runs_unchanged() {
        let config = base_config(GenerationStrategy::Greedy, 3);
        let generator = CFGGenerator::new(config, 4).expect("construct");
        let sequences = generator
            .generate_with_cfg(
                &[0],
                constant_model(vec![0.0, 0.1, 5.0, 0.0]),
                constant_model(vec![9.0, 0.0, 0.0, 0.0]),
            )
            .expect("generate");
        assert_eq!(sequences[0], vec![0, 2, 2, 2]);
    }

    #[test]
    fn test_negative_prompt_subtracts_the_unwanted_distribution() {
        let config = base_config(GenerationStrategy::Greedy, 1);
        let generator = CFGGenerator::new(config, 4).expect("construct");

        // Positive likes tokens 1 and 2 equally; negative likes token 1, so
        // token 2 must survive.
        let sequences = generator
            .generate_with_negative_prompt(
                &[0],
                constant_model(vec![0.0, 2.0, 2.0, 0.0]),
                constant_model(vec![0.0, 3.0, 0.0, 0.0]),
                1.0,
            )
            .expect("generate");
        assert_eq!(sequences[0], vec![0, 2]);
    }

    #[test]
    fn test_multi_condition_cfg_accumulates_every_condition() {
        let config = base_config(GenerationStrategy::Greedy, 1);
        let generator = CFGGenerator::new(config, 4).expect("construct");

        let conditions: Vec<BoxedLogitsFn<'_>> = vec![
            Box::new(constant_model(vec![0.0, 1.0, 0.0, 0.0])),
            Box::new(constant_model(vec![0.0, 0.0, 3.0, 0.0])),
        ];
        let sequences = generator
            .generate_with_multi_condition_cfg(
                &[0],
                conditions,
                vec![1.0, 1.0],
                constant_model(vec![0.0, 0.0, 0.0, 0.0]),
            )
            .expect("generate");
        assert_eq!(sequences[0], vec![0, 2], "the stronger condition wins");
    }

    #[test]
    fn test_multi_condition_cfg_rejects_mismatched_scale_count() {
        let config = base_config(GenerationStrategy::Greedy, 1);
        let generator = CFGGenerator::new(config, 4).expect("construct");
        let conditions: Vec<BoxedLogitsFn<'_>> =
            vec![Box::new(constant_model(vec![0.0, 1.0, 0.0, 0.0]))];
        assert!(generator
            .generate_with_multi_condition_cfg(
                &[0],
                conditions,
                vec![1.0, 1.0],
                constant_model(vec![0.0; 4]),
            )
            .is_err());
    }

    #[test]
    fn test_dynamic_thresholding_clamps_and_validates() {
        let config = base_config(GenerationStrategy::Greedy, 1);
        let generator = CFGGenerator::new(config, 4).expect("construct");

        // |logits| sorted: [1, 2, 3, 10]; percentile 0.5 -> index 2 -> 3.0.
        let clamped = generator
            .apply_dynamic_thresholding(vec![10.0, -2.0, 3.0, 1.0], 0.5)
            .expect("threshold");
        assert_eq!(clamped, vec![3.0, -2.0, 3.0, 1.0]);

        assert!(generator.apply_dynamic_thresholding(vec![1.0], 1.5).is_err());
        assert!(generator.apply_dynamic_thresholding(vec![1.0], f32::NAN).is_err());
        assert!(generator.apply_dynamic_thresholding(Vec::new(), 0.5).is_err());
    }

    #[test]
    fn test_guidance_rejects_mismatched_logit_lengths() {
        let config = base_config(GenerationStrategy::Greedy, 1);
        let generator = CFGGenerator::new(config, 4).expect("construct");
        assert!(generator
            .apply_cfg_guidance(
                &tensor_1d(&[0.0, 1.0, 2.0, 3.0]),
                &tensor_1d(&[0.0, 1.0]),
                1.5,
                false,
                0.99,
            )
            .is_err());
    }
}
