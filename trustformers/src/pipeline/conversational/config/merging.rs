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

    /// Merge summarization configurations
    fn merge_summarization_config(
        base: &SummarizationConfig,
        override_config: &SummarizationConfig,
    ) -> Result<SummarizationConfig> {
        Ok(SummarizationConfig {
            enabled: override_config.enabled,
            trigger_threshold: override_config.trigger_threshold,
            target_length: override_config.target_length,
            strategy: override_config.strategy.clone(),
            preserve_recent_turns: override_config.preserve_recent_turns,
        })
    }

    /// Merge memory configurations
    fn merge_memory_config(
        base: &MemoryConfig,
        override_config: &MemoryConfig,
    ) -> Result<MemoryConfig> {
        Ok(MemoryConfig {
            enabled: override_config.enabled,
            compression_threshold: override_config.compression_threshold,
            persist_important_memories: override_config.persist_important_memories,
            decay_rate: override_config.decay_rate,
            max_memories: override_config.max_memories,
        })
    }

    /// Merge repair configurations
    fn merge_repair_config(
        base: &RepairConfig,
        override_config: &RepairConfig,
    ) -> Result<RepairConfig> {
        Ok(RepairConfig {
            enabled: override_config.enabled,
            detect_breakdowns: override_config.detect_breakdowns,
            max_repair_attempts: override_config.max_repair_attempts,
            repair_strategies: override_config.repair_strategies.clone(),
        })
    }

    /// Merge streaming configurations
    fn merge_streaming_config(
        base: &StreamingConfig,
        override_config: &StreamingConfig,
    ) -> Result<StreamingConfig> {
        Ok(StreamingConfig {
            enabled: override_config.enabled,
            chunk_size: override_config.chunk_size,
            buffer_size: override_config.buffer_size,
            typing_delay_ms: override_config.typing_delay_ms,
        })
    }
}
