//! Configuration merging for conversational AI pipeline.
//!
//! This module provides utilities for merging multiple configuration objects
//! with intelligent override logic and conflict resolution.

use crate::error::Result;
use crate::pipeline::conversational::config::validation::{
    ConfigurationValidator, ValidationRules,
};
use crate::pipeline::conversational::types::*;

/// Configuration merger for combining configurations
pub struct ConfigurationMerger;

/// Field-level merge for the non-`Option` sub-configs: none of their fields
/// have a distinct "unset" representation, so a field is treated as
/// explicitly overridden when it differs from the *struct's* default for
/// that field (`default_value`), and the base value is kept otherwise. This
/// can't distinguish "explicitly set to the default" from "never touched",
/// the same inherent limitation any zero-value-means-unset config merge has.
fn pick<T: PartialEq>(base: T, override_value: T, default_value: &T) -> T {
    if &override_value == default_value {
        base
    } else {
        override_value
    }
}

impl ConfigurationMerger {
    /// Merge two configurations, with 'override_config' taking precedence
    pub fn merge(
        base: &ConversationalConfig,
        override_config: &ConversationalConfig,
    ) -> Result<ConversationalConfig> {
        let mut merged = base.clone();

        // Merge basic parameters - override wins
        merged.max_history_turns = override_config.max_history_turns;
        merged.max_context_tokens = override_config.max_context_tokens;
        merged.enable_summarization = override_config.enable_summarization;
        merged.temperature = override_config.temperature;
        merged.top_p = override_config.top_p;
        merged.top_k = override_config.top_k;
        merged.max_response_tokens = override_config.max_response_tokens;
        merged.enable_safety_filter = override_config.enable_safety_filter;
        merged.conversation_mode = override_config.conversation_mode.clone();
        merged.enable_persistence = override_config.enable_persistence;

        // Merge optional fields
        if override_config.system_prompt.is_some() {
            merged.system_prompt = override_config.system_prompt.clone();
        }

        if override_config.persona.is_some() {
            merged.persona = override_config.persona.clone();
        }

        // Merge sub-configurations
        merged.summarization_config = Self::merge_summarization_config(
            &merged.summarization_config,
            &override_config.summarization_config,
        )?;

        merged.memory_config =
            Self::merge_memory_config(&merged.memory_config, &override_config.memory_config)?;

        merged.repair_config =
            Self::merge_repair_config(&merged.repair_config, &override_config.repair_config)?;

        merged.streaming_config = Self::merge_streaming_config(
            &merged.streaming_config,
            &override_config.streaming_config,
        )?;

        // Validate merged configuration
        let validator = ConfigurationValidator::new();
        validator.validate(&merged, &ValidationRules::default())?;

        Ok(merged)
    }

    /// Merge summarization configurations: a field takes `override_config`'s
    /// value when it differs from [`SummarizationConfig::default`], and
    /// `base`'s value otherwise (see [`pick`]).
    fn merge_summarization_config(
        base: &SummarizationConfig,
        override_config: &SummarizationConfig,
    ) -> Result<SummarizationConfig> {
        let default = SummarizationConfig::default();
        Ok(SummarizationConfig {
            enabled: pick(base.enabled, override_config.enabled, &default.enabled),
            trigger_threshold: pick(
                base.trigger_threshold,
                override_config.trigger_threshold,
                &default.trigger_threshold,
            ),
            target_length: pick(
                base.target_length,
                override_config.target_length,
                &default.target_length,
            ),
            strategy: pick(
                base.strategy.clone(),
                override_config.strategy.clone(),
                &default.strategy,
            ),
            preserve_recent_turns: pick(
                base.preserve_recent_turns,
                override_config.preserve_recent_turns,
                &default.preserve_recent_turns,
            ),
        })
    }

    /// Merge memory configurations: a field takes `override_config`'s value
    /// when it differs from [`MemoryConfig::default`], and `base`'s value
    /// otherwise (see [`pick`]).
    fn merge_memory_config(
        base: &MemoryConfig,
        override_config: &MemoryConfig,
    ) -> Result<MemoryConfig> {
        let default = MemoryConfig::default();
        Ok(MemoryConfig {
            enabled: pick(base.enabled, override_config.enabled, &default.enabled),
            compression_threshold: pick(
                base.compression_threshold,
                override_config.compression_threshold,
                &default.compression_threshold,
            ),
            persist_important_memories: pick(
                base.persist_important_memories,
                override_config.persist_important_memories,
                &default.persist_important_memories,
            ),
            decay_rate: pick(
                base.decay_rate,
                override_config.decay_rate,
                &default.decay_rate,
            ),
            max_memories: pick(
                base.max_memories,
                override_config.max_memories,
                &default.max_memories,
            ),
        })
    }

    /// Merge repair configurations: a field takes `override_config`'s value
    /// when it differs from [`RepairConfig::default`], and `base`'s value
    /// otherwise (see [`pick`]).
    fn merge_repair_config(
        base: &RepairConfig,
        override_config: &RepairConfig,
    ) -> Result<RepairConfig> {
        let default = RepairConfig::default();
        Ok(RepairConfig {
            enabled: pick(base.enabled, override_config.enabled, &default.enabled),
            detect_breakdowns: pick(
                base.detect_breakdowns,
                override_config.detect_breakdowns,
                &default.detect_breakdowns,
            ),
            max_repair_attempts: pick(
                base.max_repair_attempts,
                override_config.max_repair_attempts,
                &default.max_repair_attempts,
            ),
            repair_strategies: pick(
                base.repair_strategies.clone(),
                override_config.repair_strategies.clone(),
                &default.repair_strategies,
            ),
        })
    }

    /// Merge streaming configurations: a field takes `override_config`'s
    /// value when it differs from [`StreamingConfig::default`], and `base`'s
    /// value otherwise (see [`pick`]).
    fn merge_streaming_config(
        base: &StreamingConfig,
        override_config: &StreamingConfig,
    ) -> Result<StreamingConfig> {
        let default = StreamingConfig::default();
        Ok(StreamingConfig {
            enabled: pick(base.enabled, override_config.enabled, &default.enabled),
            chunk_size: pick(
                base.chunk_size,
                override_config.chunk_size,
                &default.chunk_size,
            ),
            buffer_size: pick(
                base.buffer_size,
                override_config.buffer_size,
                &default.buffer_size,
            ),
            typing_delay_ms: pick(
                base.typing_delay_ms,
                override_config.typing_delay_ms,
                &default.typing_delay_ms,
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_base_config() -> ConversationalConfig {
        ConversationalConfig::default()
    }

    fn make_override_config() -> ConversationalConfig {
        let mut config = ConversationalConfig::default();
        config.max_history_turns = 10;
        config.max_context_tokens = 2048;
        config.temperature = 0.5;
        config.top_p = 0.8;
        config.max_response_tokens = 256;
        config
    }

    #[test]
    fn test_merge_basic_parameters_override_wins() {
        let base = make_base_config();
        let override_cfg = make_override_config();
        let result = ConfigurationMerger::merge(&base, &override_cfg);
        if let Ok(merged) = result {
            assert_eq!(merged.max_history_turns, 10);
            assert_eq!(merged.max_context_tokens, 2048);
            assert!((merged.temperature - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_merge_preserves_base_when_override_is_same() {
        let base = make_base_config();
        let override_cfg = make_base_config();
        let result = ConfigurationMerger::merge(&base, &override_cfg);
        if let Ok(merged) = result {
            assert_eq!(merged.max_history_turns, base.max_history_turns);
        }
    }

    #[test]
    fn test_merge_system_prompt_override_takes_effect() {
        let base = make_base_config();
        let mut override_cfg = make_base_config();
        override_cfg.system_prompt = Some("New prompt".to_string());
        let result = ConfigurationMerger::merge(&base, &override_cfg);
        if let Ok(merged) = result {
            assert_eq!(merged.system_prompt, Some("New prompt".to_string()));
        }
    }

    #[test]
    fn test_merge_system_prompt_base_kept_if_override_none() {
        let mut base = make_base_config();
        base.system_prompt = Some("Base prompt".to_string());
        let mut override_cfg = make_base_config();
        override_cfg.system_prompt = None;
        let result = ConfigurationMerger::merge(&base, &override_cfg);
        if let Ok(merged) = result {
            // When override system_prompt is None, base is kept
            assert_eq!(merged.system_prompt, Some("Base prompt".to_string()));
        }
    }

    #[test]
    fn test_merge_streaming_config_override_wins() {
        let base = make_base_config();
        let mut override_cfg = make_base_config();
        override_cfg.streaming_config.enabled = true;
        override_cfg.streaming_config.chunk_size = 20;
        let result = ConfigurationMerger::merge(&base, &override_cfg);
        if let Ok(merged) = result {
            assert!(merged.streaming_config.enabled);
            assert_eq!(merged.streaming_config.chunk_size, 20);
        }
    }

    #[test]
    fn test_merge_memory_config_override_wins() {
        let base = make_base_config();
        let mut override_cfg = make_base_config();
        override_cfg.memory_config.max_memories = 50;
        override_cfg.memory_config.decay_rate = 0.8;
        let result = ConfigurationMerger::merge(&base, &override_cfg);
        if let Ok(merged) = result {
            assert_eq!(merged.memory_config.max_memories, 50);
            assert!((merged.memory_config.decay_rate - 0.8).abs() < 1e-6);
        }
    }

    #[test]
    fn test_merge_repair_config_override_wins() {
        let base = make_base_config();
        let mut override_cfg = make_base_config();
        override_cfg.repair_config.max_repair_attempts = 5;
        override_cfg.repair_config.enabled = false;
        let result = ConfigurationMerger::merge(&base, &override_cfg);
        if let Ok(merged) = result {
            assert_eq!(merged.repair_config.max_repair_attempts, 5);
            assert!(!merged.repair_config.enabled);
        }
    }

    /// Regression test: the sub-config merge functions (summarization,
    /// memory, repair, streaming) used to ignore `base` entirely and always
    /// return a clone of `override_config`, even for fields the override
    /// left at their default (i.e. never touched). A `base` customization
    /// must now survive being merged against an all-default override.
    #[test]
    fn test_merge_sub_configs_preserve_base_when_override_is_default() {
        let mut base = make_base_config();
        base.summarization_config.target_length = 999;
        base.memory_config.max_memories = 12345;
        base.repair_config.max_repair_attempts = 42;
        base.streaming_config.buffer_size = 900;
        base.streaming_config.chunk_size = 77;

        // An override that never touches these sub-configs, so every field
        // is at `T::default()`.
        let override_cfg = make_base_config();

        let merged = ConfigurationMerger::merge(&base, &override_cfg)
            .expect("merging two default-shaped configs must succeed");

        assert_eq!(
            merged.summarization_config.target_length, 999,
            "base's non-default summarization target_length must survive an all-default override"
        );
        assert_eq!(
            merged.memory_config.max_memories, 12345,
            "base's non-default memory max_memories must survive an all-default override"
        );
        assert_eq!(
            merged.repair_config.max_repair_attempts, 42,
            "base's non-default repair max_repair_attempts must survive an all-default override"
        );
        assert_eq!(
            merged.streaming_config.chunk_size, 77,
            "base's non-default streaming chunk_size must survive an all-default override"
        );
    }

    #[test]
    fn test_merge_conversation_mode_override() {
        let base = make_base_config();
        let mut override_cfg = make_base_config();
        override_cfg.conversation_mode = ConversationMode::Educational;
        let result = ConfigurationMerger::merge(&base, &override_cfg);
        if let Ok(merged) = result {
            assert_eq!(merged.conversation_mode, ConversationMode::Educational);
        }
    }

    #[test]
    fn test_merge_enable_safety_filter_override() {
        let base = make_base_config();
        let mut override_cfg = make_base_config();
        override_cfg.enable_safety_filter = false;
        let result = ConfigurationMerger::merge(&base, &override_cfg);
        if let Ok(merged) = result {
            assert!(!merged.enable_safety_filter);
        }
    }

    #[test]
    fn test_merge_summarization_config_override() {
        let base = make_base_config();
        let mut override_cfg = make_base_config();
        override_cfg.summarization_config.target_length = 500;
        let result = ConfigurationMerger::merge(&base, &override_cfg);
        if let Ok(merged) = result {
            assert_eq!(merged.summarization_config.target_length, 500);
        }
    }

    #[test]
    fn test_default_config_is_valid() {
        let config = ConversationalConfig::default();
        assert!(config.temperature >= 0.0);
        assert!(config.temperature <= 2.0);
        assert!(config.top_p >= 0.0);
        assert!(config.top_p <= 1.0);
    }
}
