//! Central configuration manager for conversational pipeline configurations.
//!
//! This module provides high-level configuration management including validation, merging,
//! environment variable support, and serialization.

use super::merging::ConfigurationMerger;
use super::presets::{ConfigurationPreset, ConfigurationPresets};
use super::validation::{ConfigurationValidator, ValidationRules};
use crate::error::{Result, TrustformersError};
use crate::pipeline::conversational::types::*;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;

/// Central configuration manager for conversational pipeline configurations.
///
/// Provides high-level configuration management including validation, merging,
/// environment variable support, and serialization.
#[derive(Debug, Clone)]
pub struct ConfigurationManager {
    /// Current active configuration
    config: ConversationalConfig,
    /// Configuration validation rules
    validation_rules: ValidationRules,
    /// Environment variable mappings
    env_mappings: HashMap<String, String>,
}

impl ConfigurationManager {
    /// Create a new configuration manager with default settings
    pub fn new() -> Self {
        Self {
            config: ConversationalConfig::default(),
            validation_rules: ValidationRules::default(),
            env_mappings: Self::default_env_mappings(),
        }
    }

    /// Create configuration manager with custom config
    pub fn with_config(config: ConversationalConfig) -> Result<Self> {
        let mut manager = Self::new();
        manager.set_config(config)?;
        Ok(manager)
    }

    /// Load configuration from environment variables
    pub fn from_environment() -> Result<Self> {
        let mut config = ConversationalConfig::default();

        // Load basic parameters
        if let Ok(temp) = env::var("TRUSTFORMERS_TEMPERATURE") {
            config.temperature = temp.parse().map_err(|e| {
                TrustformersError::invalid_input_simple(format!("Invalid temperature value: {}", e))
            })?;
        }

        if let Ok(max_tokens) = env::var("TRUSTFORMERS_MAX_RESPONSE_TOKENS") {
            config.max_response_tokens = max_tokens.parse().map_err(|e| {
                TrustformersError::invalid_input_simple(format!(
                    "Invalid max_response_tokens value: {}",
                    e
                ))
            })?;
        }

        if let Ok(max_context) = env::var("TRUSTFORMERS_MAX_CONTEXT_TOKENS") {
            config.max_context_tokens = max_context.parse().map_err(|e| {
                TrustformersError::invalid_input_simple(format!(
                    "Invalid max_context_tokens value: {}",
                    e
                ))
            })?;
        }

        if let Ok(max_history) = env::var("TRUSTFORMERS_MAX_HISTORY_TURNS") {
            config.max_history_turns = max_history.parse().map_err(|e| {
                TrustformersError::invalid_input_simple(format!(
                    "Invalid max_history_turns value: {}",
                    e
                ))
            })?;
        }

        if let Ok(top_p) = env::var("TRUSTFORMERS_TOP_P") {
            config.top_p = top_p.parse().map_err(|e| {
                TrustformersError::invalid_input_simple(format!("Invalid top_p value: {}", e))
            })?;
        }

        if let Ok(top_k) = env::var("TRUSTFORMERS_TOP_K") {
            config.top_k = Some(top_k.parse().map_err(|e| {
                TrustformersError::invalid_input_simple(format!("Invalid top_k value: {}", e))
            })?);
        }

        if let Ok(system_prompt) = env::var("TRUSTFORMERS_SYSTEM_PROMPT") {
            config.system_prompt = Some(system_prompt);
        }

        if let Ok(safety_filter) = env::var("TRUSTFORMERS_ENABLE_SAFETY_FILTER") {
            config.enable_safety_filter =
                safety_filter.to_lowercase() == "true" || safety_filter == "1";
        }

        if let Ok(summarization) = env::var("TRUSTFORMERS_ENABLE_SUMMARIZATION") {
            config.enable_summarization =
                summarization.to_lowercase() == "true" || summarization == "1";
        }

        if let Ok(persistence) = env::var("TRUSTFORMERS_ENABLE_PERSISTENCE") {
            config.enable_persistence = persistence.to_lowercase() == "true" || persistence == "1";
        }

        // Load conversation mode
        if let Ok(mode) = env::var("TRUSTFORMERS_CONVERSATION_MODE") {
            config.conversation_mode = match mode.to_lowercase().as_str() {
                "chat" => ConversationMode::Chat,
                "assistant" => ConversationMode::Assistant,
                "instruction" | "instruction_following" => ConversationMode::InstructionFollowing,
                "qa" | "question_answering" => ConversationMode::QuestionAnswering,
                "roleplay" | "role_play" => ConversationMode::RolePlay,
                "educational" => ConversationMode::Educational,
                _ => {
                    return Err(TrustformersError::invalid_input_simple(format!(
                        "Invalid conversation mode: {}",
                        mode
                    )))
                },
            };
        }

        // Load memory configuration
        if let Ok(memory_enabled) = env::var("TRUSTFORMERS_MEMORY_ENABLED") {
            config.memory_config.enabled =
                memory_enabled.to_lowercase() == "true" || memory_enabled == "1";
        }

        if let Ok(max_memories) = env::var("TRUSTFORMERS_MAX_MEMORIES") {
            config.memory_config.max_memories = max_memories.parse().map_err(|e| {
                TrustformersError::invalid_input_simple(format!(
                    "Invalid max_memories value: {}",
                    e
                ))
            })?;
        }

        if let Ok(decay_rate) = env::var("TRUSTFORMERS_MEMORY_DECAY_RATE") {
            config.memory_config.decay_rate = decay_rate.parse().map_err(|e| {
                TrustformersError::invalid_input_simple(format!(
                    "Invalid memory_decay_rate value: {}",
                    e
                ))
            })?;
        }

        // Load streaming configuration
        if let Ok(streaming_enabled) = env::var("TRUSTFORMERS_STREAMING_ENABLED") {
            config.streaming_config.enabled =
                streaming_enabled.to_lowercase() == "true" || streaming_enabled == "1";
        }

        if let Ok(chunk_size) = env::var("TRUSTFORMERS_STREAMING_CHUNK_SIZE") {
            config.streaming_config.chunk_size = chunk_size.parse().map_err(|e| {
                TrustformersError::invalid_input_simple(format!(
                    "Invalid streaming_chunk_size value: {}",
                    e
                ))
            })?;
        }

        if let Ok(typing_delay) = env::var("TRUSTFORMERS_TYPING_DELAY_MS") {
            config.streaming_config.typing_delay_ms = typing_delay.parse().map_err(|e| {
                TrustformersError::invalid_input_simple(format!(
                    "Invalid typing_delay_ms value: {}",
                    e
                ))
            })?;
        }

        Self::with_config(config)
    }

    /// Load configuration from JSON file
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path).map_err(|e| {
            TrustformersError::invalid_input_simple(format!("Failed to read config file: {}", e))
        })?;

        let config: ConversationalConfig = serde_json::from_str(&content).map_err(|e| {
            TrustformersError::runtime_error(format!("Failed to parse JSON config: {}", e))
        })?;

        Self::with_config(config)
    }

    /// Load configuration from YAML file
    pub fn from_yaml_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path).map_err(|e| {
            TrustformersError::invalid_input_simple(format!("Failed to read config file: {}", e))
        })?;

        let config: ConversationalConfig = serde_yaml::from_str(&content).map_err(|e| {
            TrustformersError::runtime_error(format!("Failed to parse YAML config: {}", e))
        })?;

        Self::with_config(config)
    }

    /// Save configuration to JSON file
    pub fn save_to_json_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = serde_json::to_string_pretty(&self.config).map_err(|e| {
            TrustformersError::runtime_error(format!("Failed to serialize config to JSON: {}", e))
        })?;

        fs::write(path, content).map_err(|e| {
            TrustformersError::invalid_input_simple(format!("Failed to write config file: {}", e))
        })
    }

    /// Save configuration to YAML file
    pub fn save_to_yaml_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = serde_yaml::to_string(&self.config).map_err(|e| {
            TrustformersError::runtime_error(format!("Failed to serialize config to YAML: {}", e))
        })?;

        fs::write(path, content).map_err(|e| {
            TrustformersError::invalid_input_simple(format!("Failed to write config file: {}", e))
        })
    }

    /// Get current configuration
    pub fn config(&self) -> &ConversationalConfig {
        &self.config
    }

    /// Set configuration with validation
    pub fn set_config(&mut self, config: ConversationalConfig) -> Result<()> {
        self.validate_config(&config)?;
        self.config = config;
        Ok(())
    }

    /// Update configuration with partial changes
    pub fn update_config<F>(&mut self, updater: F) -> Result<()>
    where
        F: FnOnce(&mut ConversationalConfig),
    {
        let mut new_config = self.config.clone();
        updater(&mut new_config);
        self.set_config(new_config)
    }

    /// Merge another configuration into current one
    pub fn merge_config(&mut self, other: &ConversationalConfig) -> Result<()> {
        let merged = ConfigurationMerger::merge(&self.config, other)?;
        self.set_config(merged)
    }

    /// Validate configuration
    pub fn validate_config(&self, config: &ConversationalConfig) -> Result<()> {
        ConfigurationValidator::new().validate(config, &self.validation_rules)
    }

    /// Get configuration as JSON string
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.config).map_err(|e| {
            TrustformersError::runtime_error(format!("Failed to serialize config to JSON: {}", e))
        })
    }

    /// Get configuration as YAML string
    pub fn to_yaml(&self) -> Result<String> {
        serde_yaml::to_string(&self.config).map_err(|e| {
            TrustformersError::runtime_error(format!("Failed to serialize config to YAML: {}", e))
        })
    }

    /// Create a configuration preset for specific use case
    pub fn with_preset(preset: ConfigurationPreset) -> Self {
        let config = ConfigurationPresets::get_preset(preset);
        Self::with_config(config).unwrap_or_else(|_| Self::new())
    }

    /// Get default environment variable mappings
    fn default_env_mappings() -> HashMap<String, String> {
        let mut mappings = HashMap::new();
        mappings.insert(
            "temperature".to_string(),
            "TRUSTFORMERS_TEMPERATURE".to_string(),
        );
        mappings.insert(
            "max_response_tokens".to_string(),
            "TRUSTFORMERS_MAX_RESPONSE_TOKENS".to_string(),
        );
        mappings.insert(
            "max_context_tokens".to_string(),
            "TRUSTFORMERS_MAX_CONTEXT_TOKENS".to_string(),
        );
        mappings.insert(
            "max_history_turns".to_string(),
            "TRUSTFORMERS_MAX_HISTORY_TURNS".to_string(),
        );
        mappings.insert("top_p".to_string(), "TRUSTFORMERS_TOP_P".to_string());
        mappings.insert("top_k".to_string(), "TRUSTFORMERS_TOP_K".to_string());
        mappings.insert(
            "system_prompt".to_string(),
            "TRUSTFORMERS_SYSTEM_PROMPT".to_string(),
        );
        mappings.insert(
            "enable_safety_filter".to_string(),
            "TRUSTFORMERS_ENABLE_SAFETY_FILTER".to_string(),
        );
        mappings.insert(
            "enable_summarization".to_string(),
            "TRUSTFORMERS_ENABLE_SUMMARIZATION".to_string(),
        );
        mappings.insert(
            "enable_persistence".to_string(),
            "TRUSTFORMERS_ENABLE_PERSISTENCE".to_string(),
        );
        mappings.insert(
            "conversation_mode".to_string(),
            "TRUSTFORMERS_CONVERSATION_MODE".to_string(),
        );
        mappings
    }
}

impl Default for ConfigurationManager {
    fn default() -> Self {
        Self::new()
    }
}
