use crate::errors::{TrustformersError, Result};
use crate::tensor::Tensor;

use super::config::{CFGConfig, GenerationConfig};
use super::cache::KVCache;

/// Classifier-Free Guidance generator for improved text generation
pub struct CFGGenerator {
    // We'll define this after creating the core module
    cfg_config: Option<CFGConfig>,
    vocab_size: usize,
    config: GenerationConfig,
}

impl CFGGenerator {
    pub fn new(config: GenerationConfig, vocab_size: usize) -> Result<Self> {
        let cfg_config = config.guided_generation.as_ref().and_then(|g| g.cfg.clone());

        Ok(Self {
            base_generator: super::core::TextGenerator::new(config, vocab_size),
            cfg_config,
        })
    }

    /// Generate text using Classifier-Free Guidance
    pub fn generate_with_cfg(
        &self,
        input_ids: &[usize],
        conditional_logits_fn: impl Fn(&[usize], Option<&KVCache>) -> Result<(Tensor, Option<KVCache>)>,
        unconditional_logits_fn: impl Fn(
            &[usize],
            Option<&KVCache>,
        ) -> Result<(Tensor, Option<KVCache>)>,
    ) -> Result<Vec<Vec<usize>>> {
        if self.cfg_config.is_none() {
            return self.base_generator.generate(input_ids, conditional_logits_fn);
        }

        let cfg_config = self.cfg_config.as_ref().unwrap();
        let mut sequences = vec![input_ids.to_vec()];
        let mut conditional_cache =
            if self.base_generator.config.use_cache { Some(KVCache::new()) } else { None };
        let mut unconditional_cache =
            if self.base_generator.config.use_cache { Some(KVCache::new()) } else { None };

        let max_length = self.base_generator.get_max_length(input_ids.len());

        for step in 0..max_length {
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
            let next_token = self.base_generator.sample_token(&guided_logits)?;
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
        // Calculate the percentile threshold
        let mut sorted_abs_logits: Vec<f32> = logits.iter().map(|&x| x.abs()).collect();
        sorted_abs_logits.sort_by(|a, b| a.partial_cmp(b).unwrap());

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

        let max_length = self.base_generator.get_max_length(input_ids.len());

        for step in 0..max_length {
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
            let next_token = self.base_generator.sample_token(&guided_logits)?;
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
        condition_logits_fns: Vec<
            Box<dyn Fn(&[usize], Option<&KVCache>) -> Result<(Tensor, Option<KVCache>)>>,
        >,
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

        let max_length = self.base_generator.get_max_length(input_ids.len());

        for step in 0..max_length {
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
            let next_token = self.base_generator.sample_token(&guided_logits)?;
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