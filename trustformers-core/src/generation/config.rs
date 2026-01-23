use serde::{Deserialize, Serialize};

/// Generation strategies available in TrustformeRS
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum GenerationStrategy {
    /// Greedy decoding - always select highest probability token
    #[default]
    Greedy,
    /// Random sampling with temperature
    Sampling { temperature: f32 },
    /// Top-k sampling
    TopK { k: usize, temperature: f32 },
    /// Top-p (nucleus) sampling
    TopP { p: f32, temperature: f32 },
    /// Beam search
    BeamSearch { num_beams: usize },
    /// Contrastive search
    ContrastiveSearch { penalty_alpha: f32, top_k: usize },
}

/// Configuration for text generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub strategy: GenerationStrategy,
    pub max_length: Option<usize>,
    pub max_new_tokens: Option<usize>,
    pub min_length: Option<usize>,
    pub do_sample: bool,
    pub early_stopping: bool,
    pub num_return_sequences: usize,
    pub pad_token_id: Option<usize>,
    pub eos_token_id: Option<usize>,
    pub bos_token_id: Option<usize>,
    pub repetition_penalty: f32,
    pub length_penalty: f32,
    pub no_repeat_ngram_size: Option<usize>,
    pub use_cache: bool,
    pub streaming: bool,
    pub guided_generation: Option<GuidedGenerationConfig>,
    pub watermarking: Option<WatermarkingConfig>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            strategy: GenerationStrategy::default(),
            max_length: Some(100),
            max_new_tokens: None,
            min_length: Some(1),
            do_sample: false,
            early_stopping: false,
            num_return_sequences: 1,
            pad_token_id: None,
            eos_token_id: None,
            bos_token_id: None,
            repetition_penalty: 1.0,
            length_penalty: 1.0,
            no_repeat_ngram_size: None,
            use_cache: true,
            streaming: false,
            guided_generation: None,
            watermarking: None,
        }
    }
}

/// Configuration for guided generation (constrained generation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidedGenerationConfig {
    pub regex_pattern: Option<String>,
    pub grammar: Option<String>,
    pub json_schema: Option<String>,
    pub choice_list: Option<Vec<String>>,
    pub max_violations: Option<usize>,
    pub backtrack_on_violation: bool,
    pub cfg: Option<CFGConfig>,
}

impl Default for GuidedGenerationConfig {
    fn default() -> Self {
        Self {
            regex_pattern: None,
            grammar: None,
            json_schema: None,
            choice_list: None,
            max_violations: Some(3),
            backtrack_on_violation: true,
            cfg: None,
        }
    }
}

/// Configuration for Classifier-Free Guidance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CFGConfig {
    pub guidance_scale: f32,
    pub unconditional_prompt: Option<String>,
    pub negative_prompt: Option<String>,
    pub dynamic_thresholding: bool,
    pub threshold_percentile: f32,
}

impl Default for CFGConfig {
    fn default() -> Self {
        Self {
            guidance_scale: 7.5,
            unconditional_prompt: None,
            negative_prompt: None,
            dynamic_thresholding: false,
            threshold_percentile: 0.95,
        }
    }
}

/// Watermarking algorithms available
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum WatermarkingAlgorithm {
    GreenList,
    SoftRedList,
    ExponentialMinimum,
}

/// Configuration for text watermarking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatermarkingConfig {
    pub algorithm: WatermarkingAlgorithm,
    pub gamma: f32,
    pub delta: f32,
    pub hash_key: Option<u64>,
    pub vocab_size: usize,
    pub context_length: usize,
    pub detection_threshold: f32,
}

impl Default for WatermarkingConfig {
    fn default() -> Self {
        Self {
            algorithm: WatermarkingAlgorithm::GreenList,
            gamma: 0.5,
            delta: 2.0,
            hash_key: None,
            vocab_size: 50257, // GPT-2 default
            context_length: 4,
            detection_threshold: 4.0,
        }
    }
}

/// Configuration for assisted generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistedGenerationConfig {
    pub draft_model_name: String,
    pub candidate_length: usize,
    pub acceptance_threshold: f32,
    pub max_draft_tokens: usize,
    pub use_dynamic_speculation: bool,
    pub speculation_depth: usize,
}

impl Default for AssistedGenerationConfig {
    fn default() -> Self {
        Self {
            draft_model_name: "distilbert-base-uncased".to_string(),
            candidate_length: 5,
            acceptance_threshold: 0.8,
            max_draft_tokens: 20,
            use_dynamic_speculation: true,
            speculation_depth: 3,
        }
    }
}

/// Builder for GenerationConfig using standardized patterns
pub mod builder {
    use super::*;
    use crate::errors::Result;
    use crate::patterns::{validators, StandardConfig, ValidatedBuilder};

    // Implement StandardConfig for our configuration types
    impl StandardConfig for GenerationConfig {
        fn validate(&self) -> Result<()> {
            // Validate max_length and max_new_tokens relationship
            if let (Some(max_length), Some(max_new_tokens)) = (self.max_length, self.max_new_tokens)
            {
                if max_new_tokens > max_length {
                    return Err(crate::errors::TrustformersError::config_error(
                        "max_new_tokens cannot be greater than max_length",
                        "generation_config_validation",
                    ));
                }
            }

            // Validate min_length vs max_length
            if let (Some(min_length), Some(max_length)) = (self.min_length, self.max_length) {
                if min_length > max_length {
                    return Err(crate::errors::TrustformersError::config_error(
                        "min_length cannot be greater than max_length",
                        "generation_config_validation",
                    ));
                }
            }

            // Validate repetition penalty
            validators::positive(self.repetition_penalty, "repetition_penalty")?;

            // Validate length penalty
            validators::positive(self.length_penalty, "length_penalty")?;

            // Validate num_return_sequences
            validators::positive(self.num_return_sequences, "num_return_sequences")?;

            // Validate strategy-specific parameters
            match self.strategy {
                GenerationStrategy::Sampling { temperature } => {
                    validators::positive(temperature, "temperature")?;
                },
                GenerationStrategy::TopK { k, temperature } => {
                    validators::positive(k, "k")?;
                    validators::positive(temperature, "temperature")?;
                },
                GenerationStrategy::TopP { p, temperature } => {
                    validators::numeric_range(p, 0.0, 1.0, "p")?;
                    validators::positive(temperature, "temperature")?;
                },
                GenerationStrategy::BeamSearch { num_beams } => {
                    validators::positive(num_beams, "num_beams")?;
                },
                GenerationStrategy::ContrastiveSearch {
                    penalty_alpha,
                    top_k,
                } => {
                    validators::positive(penalty_alpha, "penalty_alpha")?;
                    validators::positive(top_k, "top_k")?;
                },
                GenerationStrategy::Greedy => {}, // No validation needed
            }

            Ok(())
        }
    }

    impl StandardConfig for GuidedGenerationConfig {}
    impl StandardConfig for CFGConfig {}
    impl StandardConfig for WatermarkingConfig {}
    impl StandardConfig for AssistedGenerationConfig {}

    /// Builder for GenerationConfig
    pub type GenerationConfigBuilder = ValidatedBuilder<GenerationConfig>;

    impl GenerationConfigBuilder {
        /// Create a new GenerationConfig builder with validation
        pub fn with_validation() -> Self {
            ValidatedBuilder::new().add_validator(|config: &GenerationConfig| config.validate())
        }

        /// Set the generation strategy
        pub fn strategy(mut self, strategy: GenerationStrategy) -> Self {
            self.data_mut().strategy = strategy;
            self
        }

        /// Set max length
        pub fn max_length(mut self, max_length: usize) -> Self {
            self.data_mut().max_length = Some(max_length);
            self
        }

        /// Set max new tokens
        pub fn max_new_tokens(mut self, max_new_tokens: usize) -> Self {
            self.data_mut().max_new_tokens = Some(max_new_tokens);
            self
        }

        /// Set min length
        pub fn min_length(mut self, min_length: usize) -> Self {
            self.data_mut().min_length = Some(min_length);
            self
        }

        /// Enable sampling
        pub fn enable_sampling(mut self, do_sample: bool) -> Self {
            self.data_mut().do_sample = do_sample;
            self
        }

        /// Enable early stopping
        pub fn early_stopping(mut self, early_stopping: bool) -> Self {
            self.data_mut().early_stopping = early_stopping;
            self
        }

        /// Set number of return sequences
        pub fn num_return_sequences(mut self, num_sequences: usize) -> Self {
            self.data_mut().num_return_sequences = num_sequences;
            self
        }

        /// Set pad token ID
        pub fn pad_token_id(mut self, token_id: usize) -> Self {
            self.data_mut().pad_token_id = Some(token_id);
            self
        }

        /// Set EOS token ID
        pub fn eos_token_id(mut self, token_id: usize) -> Self {
            self.data_mut().eos_token_id = Some(token_id);
            self
        }

        /// Set BOS token ID
        pub fn bos_token_id(mut self, token_id: usize) -> Self {
            self.data_mut().bos_token_id = Some(token_id);
            self
        }

        /// Set repetition penalty
        pub fn repetition_penalty(mut self, penalty: f32) -> Self {
            self.data_mut().repetition_penalty = penalty;
            self
        }

        /// Set length penalty
        pub fn length_penalty(mut self, penalty: f32) -> Self {
            self.data_mut().length_penalty = penalty;
            self
        }

        /// Set no repeat ngram size
        pub fn no_repeat_ngram_size(mut self, size: usize) -> Self {
            self.data_mut().no_repeat_ngram_size = Some(size);
            self
        }

        /// Enable/disable caching
        pub fn use_cache(mut self, use_cache: bool) -> Self {
            self.data_mut().use_cache = use_cache;
            self
        }

        /// Enable/disable streaming
        pub fn streaming(mut self, streaming: bool) -> Self {
            self.data_mut().streaming = streaming;
            self
        }

        /// Set guided generation config
        pub fn guided_generation(mut self, config: GuidedGenerationConfig) -> Self {
            self.data_mut().guided_generation = Some(config);
            self
        }

        /// Set watermarking config
        pub fn watermarking(mut self, config: WatermarkingConfig) -> Self {
            self.data_mut().watermarking = Some(config);
            self
        }

        /// Quick setup for greedy decoding
        pub fn greedy(mut self) -> Self {
            self.data_mut().strategy = GenerationStrategy::Greedy;
            self.data_mut().do_sample = false;
            self
        }

        /// Quick setup for sampling with temperature
        pub fn sampling_with_temperature(mut self, temperature: f32) -> Self {
            self.data_mut().strategy = GenerationStrategy::Sampling { temperature };
            self.data_mut().do_sample = true;
            self
        }

        /// Quick setup for top-k sampling
        pub fn top_k_sampling(mut self, k: usize, temperature: f32) -> Self {
            self.data_mut().strategy = GenerationStrategy::TopK { k, temperature };
            self.data_mut().do_sample = true;
            self
        }

        /// Quick setup for top-p sampling
        pub fn top_p_sampling(mut self, p: f32, temperature: f32) -> Self {
            self.data_mut().strategy = GenerationStrategy::TopP { p, temperature };
            self.data_mut().do_sample = true;
            self
        }

        /// Quick setup for beam search
        pub fn beam_search(mut self, num_beams: usize) -> Self {
            self.data_mut().strategy = GenerationStrategy::BeamSearch { num_beams };
            self.data_mut().do_sample = false;
            self
        }
    }

    // Convenience function for creating a builder
    pub fn generation_config() -> GenerationConfigBuilder {
        GenerationConfigBuilder::with_validation()
    }
}

#[cfg(test)]
mod tests {
    use super::builder::*;
    use super::*;
    use crate::patterns::builder::Builder;

    #[test]
    fn test_generation_config_builder() {
        let config = generation_config()
            .greedy()
            .max_length(100)
            .early_stopping(true)
            .build()
            .unwrap();

        assert_eq!(config.strategy, GenerationStrategy::Greedy);
        assert_eq!(config.max_length, Some(100));
        assert!(config.early_stopping);
    }

    #[test]
    fn test_generation_config_validation() {
        // This should fail validation (min_length > max_length)
        let result = generation_config().min_length(200).max_length(100).build();

        assert!(result.is_err());
    }

    #[test]
    fn test_sampling_config() {
        let config = generation_config()
            .sampling_with_temperature(0.8)
            .max_new_tokens(50)
            .repetition_penalty(1.1)
            .build()
            .unwrap();

        if let GenerationStrategy::Sampling { temperature } = config.strategy {
            assert_eq!(temperature, 0.8);
        } else {
            panic!("Expected sampling strategy but got {:?}", config.strategy);
        }

        assert_eq!(config.max_new_tokens, Some(50));
        assert_eq!(config.repetition_penalty, 1.1);
    }

    #[test]
    fn test_beam_search_config() {
        let config = generation_config()
            .beam_search(4)
            .max_length(200)
            .length_penalty(0.8)
            .build()
            .unwrap();

        if let GenerationStrategy::BeamSearch { num_beams } = config.strategy {
            assert_eq!(num_beams, 4);
        } else {
            panic!(
                "Expected beam search strategy but got {:?}",
                config.strategy
            );
        }
    }
}
