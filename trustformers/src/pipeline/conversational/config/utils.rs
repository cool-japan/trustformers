//! Utility functions for conversational configuration management.
//!
//! This module provides helper functions for validation, conversion,
//! and other common configuration management tasks.

use crate::error::{Result, TrustformersError};
use crate::pipeline::conversational::config::builder::{
    ConversationalConfigBuilder, MemoryConfigBuilder, StreamingConfigBuilder,
    SummarizationConfigBuilder,
};
use crate::pipeline::conversational::config::presets::ConfigurationPreset;
use crate::pipeline::conversational::types::*;

/// Utility functions for configuration management
pub mod utils {
    use super::*;

    /// Create a quick configuration for testing
    pub fn test_config() -> ConversationalConfig {
        ConversationalConfigBuilder::new()
            .temperature(0.5)
            .max_response_tokens(256)
            .max_context_tokens(1024)
            .enable_safety_filter(false)
            .build_unchecked()
    }

    /// Create a minimal configuration
    pub fn minimal_config() -> ConversationalConfig {
        ConversationalConfigBuilder::new()
            .temperature(0.7)
            .max_response_tokens(100)
            .max_context_tokens(500)
            .max_history_turns(5)
            .enable_summarization(false)
            .enable_persistence(false)
            .memory_config(MemoryConfigBuilder::new().enabled(false).build())
            .streaming_config(StreamingConfigBuilder::new().enabled(false).build())
            .build_unchecked()
    }

    /// Create a high-performance configuration
    pub fn performance_config() -> ConversationalConfig {
        ConversationalConfigBuilder::new()
            .temperature(0.1)
            .top_p(0.5)
            .max_response_tokens(200)
            .max_context_tokens(1000)
            .max_history_turns(10)
            .enable_summarization(true)
            .summarization_config(
                SummarizationConfigBuilder::new()
                    .enabled(true)
                    .trigger_threshold(800)
                    .target_length(100)
                    .preserve_recent_turns(2)
                    .build(),
            )
            .memory_config(MemoryConfigBuilder::new().enabled(false).build())
            .build_unchecked()
    }

    /// Validate configuration compatibility
    pub fn validate_compatibility(
        config1: &ConversationalConfig,
        config2: &ConversationalConfig,
    ) -> Result<()> {
        // Check if configurations are compatible for merging
        if config1.conversation_mode != config2.conversation_mode {
            return Err(TrustformersError::invalid_input_simple(
                "Conversation modes are incompatible for merging".to_string(),
            ));
        }

        Ok(())
    }

    /// Calculate configuration complexity score
    pub fn complexity_score(config: &ConversationalConfig) -> f32 {
        let mut score = 0.0;

        // Base complexity from token limits
        score += (config.max_context_tokens as f32 / 1000.0) * 0.1;
        score += (config.max_response_tokens as f32 / 100.0) * 0.1;

        // Memory complexity
        if config.memory_config.enabled {
            score += (config.memory_config.max_memories as f32 / 100.0) * 0.2;
        }

        // Summarization complexity
        if config.summarization_config.enabled {
            score += 0.3;
        }

        // Streaming complexity
        if config.streaming_config.enabled {
            score += 0.2;
        }

        // Repair complexity
        if config.repair_config.enabled {
            score += config.repair_config.repair_strategies.len() as f32 * 0.1;
        }

        // Persona complexity
        if config.persona.is_some() {
            score += 0.4;
        }

        score.min(10.0) // Cap at 10.0
    }

    /// Get recommended presets for use case
    pub fn recommended_presets_for_use_case(use_case: &str) -> Vec<ConfigurationPreset> {
        match use_case.to_lowercase().as_str() {
            "education" | "learning" | "tutoring" => vec![
                ConfigurationPreset::Educational,
                ConfigurationPreset::LanguageLearning,
            ],
            "customer_service" | "support" | "help" => vec![
                ConfigurationPreset::CustomerSupport,
                ConfigurationPreset::Professional,
            ],
            "creative" | "writing" | "storytelling" => {
                vec![ConfigurationPreset::Creative, ConfigurationPreset::Writing]
            },
            "technical" | "programming" | "code" => {
                vec![ConfigurationPreset::Technical, ConfigurationPreset::Coding]
            },
            "medical" | "health" | "healthcare" => vec![ConfigurationPreset::Medical],
            "legal" | "law" => vec![ConfigurationPreset::Legal],
            "research" | "analysis" => {
                vec![ConfigurationPreset::Research, ConfigurationPreset::Focused]
            },
            "casual" | "chat" | "social" => {
                vec![ConfigurationPreset::Casual, ConfigurationPreset::Default]
            },
            "business" | "professional" => vec![
                ConfigurationPreset::Professional,
                ConfigurationPreset::Focused,
            ],
            "gaming" | "entertainment" => {
                vec![ConfigurationPreset::Gaming, ConfigurationPreset::Creative]
            },
            _ => vec![ConfigurationPreset::Default],
        }
    }
}

// TODO: These tests use types that don't exist in current module structure
// (PersonaConfigBuilder, ConfigurationPresets, etc.). Need to be rewritten.
// #[cfg(test)]
#[cfg(test_disabled)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_configuration_builder() {
        let config = ConversationalConfigBuilder::new()
            .temperature(0.8)
            .max_response_tokens(1024)
            .conversation_mode(ConversationMode::Educational)
            .build()
            .unwrap();

        assert_eq!(config.temperature, 0.8);
        assert_eq!(config.max_response_tokens, 1024);
        assert!(matches!(
            config.conversation_mode,
            ConversationMode::Educational
        ));
    }

    #[test]
    fn test_persona_builder() {
        let persona = PersonaConfigBuilder::new()
            .name("Einstein")
            .personality("Curious and wise")
            .add_expertise("Physics")
            .add_expertise("Mathematics")
            .build();

        assert_eq!(persona.name, "Einstein");
        assert_eq!(persona.personality, "Curious and wise");
        assert_eq!(persona.expertise.len(), 2);
        assert!(persona.expertise.contains(&"Physics".to_string()));
    }

    #[test]
    fn test_configuration_validation() {
        let config = ConversationalConfigBuilder::new()
            .temperature(5.0) // Invalid - too high
            .build();

        assert!(config.is_err());
    }

    #[test]
    fn test_configuration_presets() {
        let creative_config = ConfigurationPresets::creative_config();
        assert!(creative_config.temperature > 0.8);

        let focused_config = ConfigurationPresets::focused_config();
        assert!(focused_config.temperature < 0.5);
    }

    #[test]
    fn test_configuration_merging() {
        let base = ConversationalConfig::default();
        let override_config = ConversationalConfigBuilder::new()
            .temperature(0.9)
            .max_response_tokens(2048)
            .build_unchecked();

        let merged = ConfigurationMerger::merge(&base, &override_config).unwrap();
        assert_eq!(merged.temperature, 0.9);
        assert_eq!(merged.max_response_tokens, 2048);
        // Other values should remain from base
        assert_eq!(merged.top_p, base.top_p);
    }

    #[test]
    fn test_environment_configuration() {
        // Set test environment variables
        env::set_var("TRUSTFORMERS_TEMPERATURE", "0.6");
        env::set_var("TRUSTFORMERS_MAX_RESPONSE_TOKENS", "800");
        env::set_var("TRUSTFORMERS_CONVERSATION_MODE", "educational");

        let manager = ConfigurationManager::from_environment().unwrap();
        assert_eq!(manager.config().temperature, 0.6);
        assert_eq!(manager.config().max_response_tokens, 800);
        assert!(matches!(
            manager.config().conversation_mode,
            ConversationMode::Educational
        ));

        // Clean up
        env::remove_var("TRUSTFORMERS_TEMPERATURE");
        env::remove_var("TRUSTFORMERS_MAX_RESPONSE_TOKENS");
        env::remove_var("TRUSTFORMERS_CONVERSATION_MODE");
    }

    #[test]
    fn test_memory_config_validation() {
        let config = MemoryConfigBuilder::new()
            .decay_rate(1.5) // Invalid - too high
            .build();

        let conversational_config =
            ConversationalConfigBuilder::new().memory_config(config).build();

        assert!(conversational_config.is_err());
    }

    #[test]
    fn test_streaming_config_validation() {
        let config = StreamingConfigBuilder::new()
            .chunk_size(0) // Invalid - must be > 0
            .build();

        let conversational_config =
            ConversationalConfigBuilder::new().streaming_config(config).build();

        assert!(conversational_config.is_err());
    }

    #[test]
    fn test_configuration_serialization() {
        let config = ConfigurationPresets::educational_config();

        // Test JSON serialization
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ConversationalConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.temperature, deserialized.temperature);

        // Test YAML serialization
        let yaml = serde_yaml::to_string(&config).unwrap();
        let deserialized: ConversationalConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(config.temperature, deserialized.temperature);
    }

    #[test]
    fn test_complexity_score() {
        let simple_config = utils::minimal_config();
        let complex_config = ConfigurationPresets::educational_config();

        let simple_score = utils::complexity_score(&simple_config);
        let complex_score = utils::complexity_score(&complex_config);

        assert!(complex_score > simple_score);
    }

    #[test]
    fn test_use_case_recommendations() {
        let education_presets = utils::recommended_presets_for_use_case("education");
        assert!(education_presets.contains(&ConfigurationPreset::Educational));

        let coding_presets = utils::recommended_presets_for_use_case("programming");
        assert!(coding_presets.contains(&ConfigurationPreset::Coding));
    }

    #[test]
    fn test_configuration_manager_update() {
        let mut manager = ConfigurationManager::new();

        manager
            .update_config(|config| {
                config.temperature = 0.9;
                config.max_response_tokens = 1500;
            })
            .unwrap();

        assert_eq!(manager.config().temperature, 0.9);
        assert_eq!(manager.config().max_response_tokens, 1500);
    }
}
