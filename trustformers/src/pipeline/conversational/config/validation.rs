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
