//! Configuration builder patterns for conversational AI pipeline.
//!
//! This module provides fluent builder APIs for creating and configuring
//! conversational pipeline configurations with type-safe, chainable methods.

use crate::error::Result;
use crate::pipeline::conversational::config::validation::{
    ConfigurationValidator, ValidationRules,
};
use crate::pipeline::conversational::config::{ConfigurationPreset, ConfigurationPresets};
use crate::pipeline::conversational::types::*;
use trustformers_core::generation::GenerationConfig;

/// Builder for ConversationalConfig with fluent API
#[derive(Debug, Default)]
pub struct ConversationalConfigBuilder {
    config: ConversationalConfig,
}

impl ConversationalConfigBuilder {
    /// Create a new builder with default values
    pub fn new() -> Self {
        Self {
            config: ConversationalConfig::default(),
        }
    }

    /// Create builder from existing config
    pub fn from_config(config: ConversationalConfig) -> Self {
        Self { config }
    }

    /// Set maximum history turns
    pub fn max_history_turns(mut self, max_turns: usize) -> Self {
        self.config.max_history_turns = max_turns;
        self
    }

    /// Set maximum context tokens
    pub fn max_context_tokens(mut self, max_tokens: usize) -> Self {
        self.config.max_context_tokens = max_tokens;
        self
    }

    /// Enable or disable summarization
    pub fn enable_summarization(mut self, enabled: bool) -> Self {
        self.config.enable_summarization = enabled;
        self
    }

    /// Set temperature for generation
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = temperature;
        self
    }

    /// Set top-p for nucleus sampling
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.config.top_p = top_p;
        self
    }

    /// Set top-k for top-k sampling
    pub fn top_k(mut self, top_k: Option<usize>) -> Self {
        self.config.top_k = top_k;
        self
    }

    /// Set maximum response tokens
    pub fn max_response_tokens(mut self, max_tokens: usize) -> Self {
        self.config.max_response_tokens = max_tokens;
        self
    }

    /// Set system prompt
    pub fn system_prompt<S: Into<String>>(mut self, prompt: Option<S>) -> Self {
        self.config.system_prompt = prompt.map(|s| s.into());
        self
    }

    /// Enable or disable safety filter
    pub fn enable_safety_filter(mut self, enabled: bool) -> Self {
        self.config.enable_safety_filter = enabled;
        self
    }

    /// Set conversation mode
    pub fn conversation_mode(mut self, mode: ConversationMode) -> Self {
        self.config.conversation_mode = mode;
        self
    }

    /// Enable or disable persistence
    pub fn enable_persistence(mut self, enabled: bool) -> Self {
        self.config.enable_persistence = enabled;
        self
    }

    /// Set persona configuration
    pub fn persona(mut self, persona: Option<PersonaConfig>) -> Self {
        self.config.persona = persona;
        self
    }

    /// Set summarization configuration
    pub fn summarization_config(mut self, config: SummarizationConfig) -> Self {
        self.config.summarization_config = config;
        self
    }

    /// Set memory configuration
    pub fn memory_config(mut self, config: MemoryConfig) -> Self {
        self.config.memory_config = config;
        self
    }

    /// Set generation configuration
    pub fn generation_config(mut self, config: GenerationConfig) -> Self {
        self.config.generation_config = config;
        self
    }

    /// Set repair configuration
    pub fn repair_config(mut self, config: RepairConfig) -> Self {
        self.config.repair_config = config;
        self
    }

    /// Set streaming configuration
    pub fn streaming_config(mut self, config: StreamingConfig) -> Self {
        self.config.streaming_config = config;
        self
    }

    /// Apply preset configuration
    pub fn with_preset(mut self, preset: ConfigurationPreset) -> Self {
        let preset_config = ConfigurationPresets::get_preset(preset);
        self.config = preset_config;
        self
    }

    /// Build the configuration with validation
    pub fn build(self) -> Result<ConversationalConfig> {
        let validator = ConfigurationValidator::new();
        validator.validate(&self.config, &ValidationRules::default())?;
        Ok(self.config)
    }

    /// Build the configuration without validation
    pub fn build_unchecked(self) -> ConversationalConfig {
        self.config
    }
}

/// Builder for PersonaConfig
#[derive(Debug, Default)]
pub struct PersonaConfigBuilder {
    name: String,
    personality: String,
    background: String,
    speaking_style: String,
    expertise: Vec<String>,
    constraints: Vec<String>,
}

impl PersonaConfigBuilder {
    /// Create a new persona builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set persona name
    pub fn name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = name.into();
        self
    }

    /// Set personality description
    pub fn personality<S: Into<String>>(mut self, personality: S) -> Self {
        self.personality = personality.into();
        self
    }

    /// Set background information
    pub fn background<S: Into<String>>(mut self, background: S) -> Self {
        self.background = background.into();
        self
    }

    /// Set speaking style
    pub fn speaking_style<S: Into<String>>(mut self, style: S) -> Self {
        self.speaking_style = style.into();
        self
    }

    /// Add expertise area
    pub fn add_expertise<S: Into<String>>(mut self, area: S) -> Self {
        self.expertise.push(area.into());
        self
    }

    /// Set expertise areas
    pub fn expertise(mut self, areas: Vec<String>) -> Self {
        self.expertise = areas;
        self
    }

    /// Add constraint
    pub fn add_constraint<S: Into<String>>(mut self, constraint: S) -> Self {
        self.constraints.push(constraint.into());
        self
    }

    /// Set constraints
    pub fn constraints(mut self, constraints: Vec<String>) -> Self {
        self.constraints = constraints;
        self
    }

    /// Build the persona configuration
    pub fn build(self) -> PersonaConfig {
        PersonaConfig {
            name: self.name,
            personality: self.personality,
            background: self.background,
            speaking_style: self.speaking_style,
            expertise: self.expertise,
            constraints: self.constraints,
        }
    }
}

/// Builder for SummarizationConfig
#[derive(Debug)]
pub struct SummarizationConfigBuilder {
    config: SummarizationConfig,
}

impl SummarizationConfigBuilder {
    /// Create a new summarization config builder
    pub fn new() -> Self {
        Self {
            config: SummarizationConfig::default(),
        }
    }

    /// Enable or disable summarization
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    /// Set trigger threshold
    pub fn trigger_threshold(mut self, threshold: usize) -> Self {
        self.config.trigger_threshold = threshold;
        self
    }

    /// Set target length
    pub fn target_length(mut self, length: usize) -> Self {
        self.config.target_length = length;
        self
    }

    /// Set summarization strategy
    pub fn strategy(mut self, strategy: SummarizationStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Set preserve recent turns
    pub fn preserve_recent_turns(mut self, turns: usize) -> Self {
        self.config.preserve_recent_turns = turns;
        self
    }

    /// Build the summarization configuration
    pub fn build(self) -> SummarizationConfig {
        self.config
    }
}

impl Default for SummarizationConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for MemoryConfig
#[derive(Debug)]
pub struct MemoryConfigBuilder {
    config: MemoryConfig,
}

impl MemoryConfigBuilder {
    /// Create a new memory config builder
    pub fn new() -> Self {
        Self {
            config: MemoryConfig::default(),
        }
    }

    /// Enable or disable memory
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    /// Set compression threshold
    pub fn compression_threshold(mut self, threshold: f32) -> Self {
        self.config.compression_threshold = threshold;
        self
    }

    /// Enable or disable persistent important memories
    pub fn persist_important_memories(mut self, persist: bool) -> Self {
        self.config.persist_important_memories = persist;
        self
    }

    /// Set decay rate
    pub fn decay_rate(mut self, rate: f32) -> Self {
        self.config.decay_rate = rate;
        self
    }

    /// Set maximum memories
    pub fn max_memories(mut self, max: usize) -> Self {
        self.config.max_memories = max;
        self
    }

    /// Build the memory configuration
    pub fn build(self) -> MemoryConfig {
        self.config
    }
}

impl Default for MemoryConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for StreamingConfig
#[derive(Debug)]
pub struct StreamingConfigBuilder {
    config: StreamingConfig,
}

impl StreamingConfigBuilder {
    /// Create a new streaming config builder
    pub fn new() -> Self {
        Self {
            config: StreamingConfig::default(),
        }
    }

    /// Enable or disable streaming
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    /// Set chunk size
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.config.chunk_size = size;
        self
    }

    /// Set buffer size
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.config.buffer_size = size;
        self
    }

    /// Set typing delay in milliseconds
    pub fn typing_delay_ms(mut self, delay: u64) -> Self {
        self.config.typing_delay_ms = delay;
        self
    }

    /// Build the streaming configuration
    pub fn build(self) -> StreamingConfig {
        self.config
    }
}

impl Default for StreamingConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for RepairConfig
#[derive(Debug)]
pub struct RepairConfigBuilder {
    config: RepairConfig,
}

impl RepairConfigBuilder {
    /// Create a new repair config builder
    pub fn new() -> Self {
        Self {
            config: RepairConfig::default(),
        }
    }

    /// Enable or disable repair
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    /// Enable or disable breakdown detection
    pub fn detect_breakdowns(mut self, detect: bool) -> Self {
        self.config.detect_breakdowns = detect;
        self
    }

    /// Set maximum repair attempts
    pub fn max_repair_attempts(mut self, max: usize) -> Self {
        self.config.max_repair_attempts = max;
        self
    }

    /// Set repair strategies
    pub fn strategies(mut self, strategies: Vec<RepairStrategy>) -> Self {
        self.config.repair_strategies = strategies;
        self
    }

    /// Add a repair strategy
    pub fn add_strategy(mut self, strategy: RepairStrategy) -> Self {
        self.config.repair_strategies.push(strategy);
        self
    }

    /// Build the repair configuration
    pub fn build(self) -> RepairConfig {
        self.config
    }
}

impl Default for RepairConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ConversationalConfigBuilder tests ----

    #[test]
    fn test_builder_new_produces_default_config() {
        let config = ConversationalConfigBuilder::new()
            .build()
            .expect("default builder must produce valid config");
        let default = ConversationalConfig::default();
        assert_eq!(config.max_history_turns, default.max_history_turns);
        assert_eq!(config.max_context_tokens, default.max_context_tokens);
    }

    #[test]
    fn test_builder_temperature_method_chaining() {
        let config = ConversationalConfigBuilder::new()
            .temperature(1.0)
            .build()
            .expect("temperature 1.0 must produce valid config");
        assert!(
            (config.temperature - 1.0).abs() < f32::EPSILON,
            "temperature must be 1.0"
        );
    }

    #[test]
    fn test_builder_max_history_turns() {
        let config = ConversationalConfigBuilder::new()
            .max_history_turns(10)
            .build()
            .expect("max_history_turns 10 must produce valid config");
        assert_eq!(config.max_history_turns, 10, "max_history_turns must be 10");
    }

    #[test]
    fn test_builder_top_k_none() {
        let config = ConversationalConfigBuilder::new()
            .top_k(None)
            .build()
            .expect("top_k None must produce valid config");
        assert!(config.top_k.is_none(), "top_k must be None");
    }

    #[test]
    fn test_builder_top_k_some() {
        let config = ConversationalConfigBuilder::new()
            .top_k(Some(100))
            .build()
            .expect("top_k Some(100) must produce valid config");
        assert_eq!(config.top_k, Some(100), "top_k must be Some(100)");
    }

    #[test]
    fn test_builder_system_prompt_set() {
        let config = ConversationalConfigBuilder::new()
            .system_prompt(Some("Custom prompt"))
            .build()
            .expect("custom system_prompt must produce valid config");
        assert_eq!(
            config.system_prompt.as_deref(),
            Some("Custom prompt"),
            "system_prompt must match the supplied value"
        );
    }

    #[test]
    fn test_builder_system_prompt_none() {
        let config = ConversationalConfigBuilder::new()
            .system_prompt(None::<String>)
            .build()
            .expect("None system_prompt must produce valid config");
        assert!(config.system_prompt.is_none(), "system_prompt must be None");
    }

    #[test]
    fn test_builder_conversation_mode() {
        let config = ConversationalConfigBuilder::new()
            .conversation_mode(ConversationMode::Assistant)
            .build()
            .expect("conversation mode change must produce valid config");
        assert_eq!(config.conversation_mode, ConversationMode::Assistant);
    }

    #[test]
    fn test_builder_enable_safety_filter_false() {
        let config = ConversationalConfigBuilder::new()
            .enable_safety_filter(false)
            .build()
            .expect("disabling safety filter must produce valid config");
        assert!(
            !config.enable_safety_filter,
            "safety filter should be disabled"
        );
    }

    #[test]
    fn test_builder_max_response_tokens() {
        let config = ConversationalConfigBuilder::new()
            .max_response_tokens(256)
            .build()
            .expect("max_response_tokens 256 must produce valid config");
        assert_eq!(
            config.max_response_tokens, 256,
            "max_response_tokens must be 256"
        );
    }

    #[test]
    fn test_builder_method_chaining_order_irrelevant() {
        let c1 = ConversationalConfigBuilder::new()
            .temperature(0.5)
            .max_history_turns(5)
            .build()
            .expect("chained build must succeed");
        let c2 = ConversationalConfigBuilder::new()
            .max_history_turns(5)
            .temperature(0.5)
            .build()
            .expect("reversed chain build must succeed");
        assert!((c1.temperature - c2.temperature).abs() < f32::EPSILON);
        assert_eq!(c1.max_history_turns, c2.max_history_turns);
    }

    #[test]
    fn test_builder_build_validates_temperature_too_high() {
        let result = ConversationalConfigBuilder::new()
            .temperature(3.0) // out of range [0, 2]
            .build();
        assert!(result.is_err(), "temperature 3.0 should fail validation");
    }

    #[test]
    fn test_builder_build_validates_temperature_negative() {
        let result = ConversationalConfigBuilder::new().temperature(-0.1).build();
        assert!(
            result.is_err(),
            "negative temperature should fail validation"
        );
    }

    #[test]
    fn test_two_builders_produce_independent_configs() {
        let config_a = ConversationalConfigBuilder::new()
            .max_history_turns(5)
            .build()
            .expect("config A must succeed");
        let config_b = ConversationalConfigBuilder::new()
            .max_history_turns(15)
            .build()
            .expect("config B must succeed");
        assert_ne!(
            config_a.max_history_turns, config_b.max_history_turns,
            "two builder instances must produce independent configs"
        );
    }

    // ---- PersonaConfigBuilder tests ----

    #[test]
    fn test_persona_builder_name() {
        let persona = PersonaConfigBuilder::new().name("Alice").build();
        assert_eq!(
            persona.name, "Alice",
            "persona name must match supplied value"
        );
    }

    #[test]
    fn test_persona_builder_expertise_accumulation() {
        let persona = PersonaConfigBuilder::new().add_expertise("Rust").add_expertise("AI").build();
        assert_eq!(
            persona.expertise.len(),
            2,
            "two expertise areas must be stored"
        );
        assert!(persona.expertise.contains(&"Rust".to_string()));
        assert!(persona.expertise.contains(&"AI".to_string()));
    }

    // ---- SummarizationConfigBuilder tests ----

    #[test]
    fn test_summarization_builder_disabled() {
        let config = SummarizationConfigBuilder::new().enabled(false).build();
        assert!(!config.enabled, "summarization must be disabled");
    }

    // ---- StreamingConfigBuilder tests ----

    #[test]
    fn test_streaming_builder_enabled() {
        let config = StreamingConfigBuilder::new().enabled(true).build();
        assert!(
            config.enabled,
            "streaming must be enabled after builder set"
        );
    }

    #[test]
    fn test_streaming_builder_chunk_size() {
        let config = StreamingConfigBuilder::new().chunk_size(20).build();
        assert_eq!(
            config.chunk_size, 20,
            "chunk_size must match supplied value"
        );
    }
}
