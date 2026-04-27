// ================================================================================================
// CONFIGURATION VALIDATION
// ================================================================================================

use crate::error::{Result, TrustformersError};
use crate::pipeline::conversational::types::{
    ConversationalConfig, MemoryConfig, RepairConfig, StreamingConfig, SummarizationConfig,
};

/// Configuration validation rules
#[derive(Debug, Clone)]
pub struct ValidationRules {
    /// Minimum temperature value
    pub min_temperature: f32,
    /// Maximum temperature value
    pub max_temperature: f32,
    /// Minimum top_p value
    pub min_top_p: f32,
    /// Maximum top_p value
    pub max_top_p: f32,
    /// Minimum top_k value
    pub min_top_k: usize,
    /// Maximum top_k value
    pub max_top_k: usize,
    /// Minimum response tokens
    pub min_response_tokens: usize,
    /// Maximum response tokens
    pub max_response_tokens: usize,
    /// Minimum context tokens
    pub min_context_tokens: usize,
    /// Maximum context tokens
    pub max_context_tokens: usize,
    /// Minimum history turns
    pub min_history_turns: usize,
    /// Maximum history turns
    pub max_history_turns: usize,
    /// Maximum system prompt length
    pub max_system_prompt_length: usize,
}

impl Default for ValidationRules {
    fn default() -> Self {
        Self {
            min_temperature: 0.0,
            max_temperature: 2.0,
            min_top_p: 0.0,
            max_top_p: 1.0,
            min_top_k: 1,
            max_top_k: 1000,
            min_response_tokens: 1,
            max_response_tokens: 8192,
            min_context_tokens: 1,
            max_context_tokens: 32768,
            min_history_turns: 1,
            max_history_turns: 1000,
            max_system_prompt_length: 10000,
        }
    }
}

/// Configuration validator
pub struct ConfigurationValidator;

impl ConfigurationValidator {
    /// Create a new validator
    pub fn new() -> Self {
        Self
    }

    /// Validate a configuration against rules
    pub fn validate(&self, config: &ConversationalConfig, rules: &ValidationRules) -> Result<()> {
        // Validate temperature
        if config.temperature < rules.min_temperature || config.temperature > rules.max_temperature
        {
            return Err(TrustformersError::invalid_input(
                format!(
                    "Temperature {} is out of range [{}, {}]",
                    config.temperature, rules.min_temperature, rules.max_temperature
                ),
                Some("temperature"),
                Some(format!(
                    "value between {} and {}",
                    rules.min_temperature, rules.max_temperature
                )),
                Some(config.temperature.to_string()),
            ));
        }

        // Validate top_p
        if config.top_p < rules.min_top_p || config.top_p > rules.max_top_p {
            return Err(TrustformersError::invalid_input(
                format!(
                    "Top-p {} is out of range [{}, {}]",
                    config.top_p, rules.min_top_p, rules.max_top_p
                ),
                Some("top_p"),
                Some(format!(
                    "value between {} and {}",
                    rules.min_top_p, rules.max_top_p
                )),
                Some(config.top_p.to_string()),
            ));
        }

        // Validate top_k
        if let Some(top_k) = config.top_k {
            if top_k < rules.min_top_k || top_k > rules.max_top_k {
                return Err(TrustformersError::invalid_input(
                    format!(
                        "Top-k {} is out of range [{}, {}]",
                        top_k, rules.min_top_k, rules.max_top_k
                    ),
                    Some("top_k"),
                    Some(format!(
                        "value between {} and {}",
                        rules.min_top_k, rules.max_top_k
                    )),
                    Some(top_k.to_string()),
                ));
            }
        }

        // Validate response tokens
        if config.max_response_tokens < rules.min_response_tokens
            || config.max_response_tokens > rules.max_response_tokens
        {
            return Err(TrustformersError::invalid_input(
                format!(
                    "Max response tokens {} is out of range [{}, {}]",
                    config.max_response_tokens,
                    rules.min_response_tokens,
                    rules.max_response_tokens
                ),
                Some("max_response_tokens"),
                Some(format!(
                    "value between {} and {}",
                    rules.min_response_tokens, rules.max_response_tokens
                )),
                Some(config.max_response_tokens.to_string()),
            ));
        }

        // Validate context tokens
        if config.max_context_tokens < rules.min_context_tokens
            || config.max_context_tokens > rules.max_context_tokens
        {
            return Err(TrustformersError::invalid_input(
                format!(
                    "Max context tokens {} is out of range [{}, {}]",
                    config.max_context_tokens, rules.min_context_tokens, rules.max_context_tokens
                ),
                Some("max_context_tokens"),
                Some(format!(
                    "value between {} and {}",
                    rules.min_context_tokens, rules.max_context_tokens
                )),
                Some(config.max_context_tokens.to_string()),
            ));
        }

        // Validate history turns
        if config.max_history_turns < rules.min_history_turns
            || config.max_history_turns > rules.max_history_turns
        {
            return Err(TrustformersError::invalid_input(
                format!(
                    "Max history turns {} is out of range [{}, {}]",
                    config.max_history_turns, rules.min_history_turns, rules.max_history_turns
                ),
                Some("max_history_turns"),
                Some(format!(
                    "value between {} and {}",
                    rules.min_history_turns, rules.max_history_turns
                )),
                Some(config.max_history_turns.to_string()),
            ));
        }

        // Validate system prompt length
        if let Some(ref prompt) = config.system_prompt {
            if prompt.len() > rules.max_system_prompt_length {
                return Err(TrustformersError::invalid_input(
                    format!(
                        "System prompt length {} exceeds maximum {}",
                        prompt.len(),
                        rules.max_system_prompt_length
                    ),
                    Some("system_prompt"),
                    Some(format!("length <= {}", rules.max_system_prompt_length)),
                    Some(prompt.len().to_string()),
                ));
            }
        }

        // Validate logical relationships
        if config.max_response_tokens > config.max_context_tokens {
            return Err(TrustformersError::invalid_input_simple(
                "Max response tokens cannot exceed max context tokens".to_string(),
            ));
        }

        // Validate memory configuration
        Self::validate_memory_config(&config.memory_config)?;

        // Validate summarization configuration
        Self::validate_summarization_config(&config.summarization_config)?;

        // Validate streaming configuration
        Self::validate_streaming_config(&config.streaming_config)?;

        // Validate repair configuration
        Self::validate_repair_config(&config.repair_config)?;

        Ok(())
    }

    /// Validate memory configuration
    fn validate_memory_config(config: &MemoryConfig) -> Result<()> {
        if config.compression_threshold < 0.0 || config.compression_threshold > 1.0 {
            return Err(TrustformersError::invalid_input(
                format!(
                    "Memory compression threshold {} must be between 0.0 and 1.0",
                    config.compression_threshold
                ),
                Some("compression_threshold"),
                Some("value between 0.0 and 1.0"),
                Some(config.compression_threshold.to_string()),
            ));
        }

        if config.decay_rate < 0.0 || config.decay_rate > 1.0 {
            return Err(TrustformersError::invalid_input(
                format!(
                    "Memory decay rate {} must be between 0.0 and 1.0",
                    config.decay_rate
                ),
                Some("decay_rate"),
                Some("value between 0.0 and 1.0"),
                Some(config.decay_rate.to_string()),
            ));
        }

        if config.max_memories == 0 {
            return Err(TrustformersError::invalid_input_simple(
                "Max memories must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate summarization configuration
    fn validate_summarization_config(config: &SummarizationConfig) -> Result<()> {
        if config.trigger_threshold == 0 {
            return Err(TrustformersError::invalid_input_simple(
                "Summarization trigger threshold must be greater than 0".to_string(),
            ));
        }

        if config.target_length == 0 {
            return Err(TrustformersError::invalid_input_simple(
                "Summarization target length must be greater than 0".to_string(),
            ));
        }

        if config.target_length >= config.trigger_threshold {
            return Err(TrustformersError::invalid_input_simple(
                "Summarization target length should be less than trigger threshold".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate streaming configuration
    fn validate_streaming_config(config: &StreamingConfig) -> Result<()> {
        if config.chunk_size == 0 {
            return Err(TrustformersError::invalid_input_simple(
                "Streaming chunk size must be greater than 0".to_string(),
            ));
        }

        if config.buffer_size == 0 {
            return Err(TrustformersError::invalid_input_simple(
                "Streaming buffer size must be greater than 0".to_string(),
            ));
        }

        if config.buffer_size < config.chunk_size {
            return Err(TrustformersError::invalid_input_simple(
                "Streaming buffer size should be at least as large as chunk size".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate repair configuration
    fn validate_repair_config(config: &RepairConfig) -> Result<()> {
        if config.max_repair_attempts == 0 {
            return Err(TrustformersError::invalid_input_simple(
                "Max repair attempts must be greater than 0".to_string(),
            ));
        }

        if config.repair_strategies.is_empty() {
            return Err(TrustformersError::invalid_input_simple(
                "At least one repair strategy must be specified".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate a configuration with default rules
    pub fn validate_config(&self, config: &ConversationalConfig) -> Result<()> {
        let rules = ValidationRules::default();
        self.validate(config, &rules)
    }
}

impl Default for ConfigurationValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> ConversationalConfig {
        ConversationalConfig::default()
    }

    fn validator() -> ConfigurationValidator {
        ConfigurationValidator::new()
    }

    fn rules() -> ValidationRules {
        ValidationRules::default()
    }

    // ---- Temperature range tests ----

    #[test]
    fn test_valid_config_passes_validation() {
        let result = validator().validate(&valid_config(), &rules());
        assert!(result.is_ok(), "default config must pass validation");
    }

    #[test]
    fn test_temperature_above_max_fails() {
        let mut config = valid_config();
        config.temperature = 2.1;
        let result = validator().validate(&config, &rules());
        assert!(result.is_err(), "temperature 2.1 must fail validation");
    }

    #[test]
    fn test_temperature_below_min_fails() {
        let mut config = valid_config();
        config.temperature = -0.1;
        let result = validator().validate(&config, &rules());
        assert!(result.is_err(), "negative temperature must fail validation");
    }

    #[test]
    fn test_temperature_at_boundary_zero_passes() {
        let mut config = valid_config();
        config.temperature = 0.0;
        let result = validator().validate(&config, &rules());
        assert!(result.is_ok(), "temperature 0.0 must pass validation");
    }

    #[test]
    fn test_temperature_at_boundary_two_passes() {
        let mut config = valid_config();
        config.temperature = 2.0;
        let result = validator().validate(&config, &rules());
        assert!(result.is_ok(), "temperature 2.0 must pass validation");
    }

    // ---- top_p range tests ----

    #[test]
    fn test_top_p_above_one_fails() {
        let mut config = valid_config();
        config.top_p = 1.1;
        let result = validator().validate(&config, &rules());
        assert!(result.is_err(), "top_p 1.1 must fail validation");
    }

    #[test]
    fn test_top_p_negative_fails() {
        let mut config = valid_config();
        config.top_p = -0.1;
        let result = validator().validate(&config, &rules());
        assert!(result.is_err(), "negative top_p must fail validation");
    }

    #[test]
    fn test_top_p_zero_passes() {
        let mut config = valid_config();
        config.top_p = 0.0;
        let result = validator().validate(&config, &rules());
        assert!(result.is_ok(), "top_p 0.0 must pass validation");
    }

    // ---- top_k range tests ----

    #[test]
    fn test_top_k_above_max_fails() {
        let mut config = valid_config();
        config.top_k = Some(1001);
        let result = validator().validate(&config, &rules());
        assert!(
            result.is_err(),
            "top_k 1001 must fail validation (max is 1000)"
        );
    }

    #[test]
    fn test_top_k_zero_fails() {
        let mut config = valid_config();
        config.top_k = Some(0);
        let result = validator().validate(&config, &rules());
        assert!(result.is_err(), "top_k 0 must fail validation (min is 1)");
    }

    #[test]
    fn test_top_k_none_passes() {
        let mut config = valid_config();
        config.top_k = None;
        let result = validator().validate(&config, &rules());
        assert!(result.is_ok(), "top_k None must pass validation");
    }

    // ---- max_response_tokens vs max_context_tokens cross-field test ----

    #[test]
    fn test_response_tokens_exceed_context_tokens_fails() {
        let mut config = valid_config();
        config.max_context_tokens = 100;
        config.max_response_tokens = 200; // response > context
        let result = validator().validate(&config, &rules());
        assert!(
            result.is_err(),
            "response tokens exceeding context tokens must fail"
        );
    }

    // ---- system prompt length test ----

    #[test]
    fn test_system_prompt_too_long_fails() {
        let mut config = valid_config();
        config.system_prompt = Some("x".repeat(10_001)); // exceeds 10000 limit
        let result = validator().validate(&config, &rules());
        assert!(
            result.is_err(),
            "system prompt exceeding 10000 chars must fail"
        );
    }

    #[test]
    fn test_system_prompt_at_max_length_passes() {
        let mut config = valid_config();
        config.system_prompt = Some("x".repeat(10_000));
        let result = validator().validate(&config, &rules());
        assert!(
            result.is_ok(),
            "system prompt at exactly 10000 chars must pass"
        );
    }

    // ---- memory config tests ----

    #[test]
    fn test_memory_compression_threshold_out_of_range_fails() {
        let mut config = valid_config();
        config.memory_config.compression_threshold = 1.5;
        let result = validator().validate(&config, &rules());
        assert!(
            result.is_err(),
            "memory compression_threshold > 1.0 must fail"
        );
    }

    #[test]
    fn test_memory_decay_rate_negative_fails() {
        let mut config = valid_config();
        config.memory_config.decay_rate = -0.1;
        let result = validator().validate(&config, &rules());
        assert!(result.is_err(), "negative decay_rate must fail");
    }

    #[test]
    fn test_memory_max_memories_zero_fails() {
        let mut config = valid_config();
        config.memory_config.max_memories = 0;
        let result = validator().validate(&config, &rules());
        assert!(result.is_err(), "max_memories 0 must fail");
    }

    // ---- streaming config tests ----

    #[test]
    fn test_streaming_chunk_size_zero_fails() {
        let mut config = valid_config();
        config.streaming_config.chunk_size = 0;
        let result = validator().validate(&config, &rules());
        assert!(result.is_err(), "chunk_size 0 must fail");
    }

    #[test]
    fn test_streaming_buffer_smaller_than_chunk_fails() {
        let mut config = valid_config();
        config.streaming_config.chunk_size = 50;
        config.streaming_config.buffer_size = 10;
        let result = validator().validate(&config, &rules());
        assert!(
            result.is_err(),
            "buffer_size smaller than chunk_size must fail"
        );
    }

    // ---- ValidationRules defaults ----

    #[test]
    fn test_default_rules_temperature_range() {
        let r = ValidationRules::default();
        assert!((r.min_temperature - 0.0).abs() < f32::EPSILON);
        assert!((r.max_temperature - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_validate_config_convenience_method_passes_default() {
        let result = validator().validate_config(&valid_config());
        assert!(
            result.is_ok(),
            "validate_config must pass for default config"
        );
    }
}
