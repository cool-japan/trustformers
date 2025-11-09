//! Conversational Pipeline for Advanced Dialogue Management
//!
//! This module provides a comprehensive conversational AI pipeline system with
//! sophisticated dialogue management, memory systems, safety filtering, and
//! real-time streaming capabilities.
//!
//! ## Architecture
//!
//! The conversational pipeline is organized into focused modules:
//! - [`types`] - Core types, configurations, and shared data structures
//! - [`config`] - Configuration management and builder patterns
//! - [`pipeline`] - Main ConversationalPipeline coordinating all components
//! - [`memory`] - Memory management and conversation context handling
//! - [`analysis`] - Conversation analysis, health assessment, and metrics
//! - [`safety`] - Content filtering and safety assessment
//! - [`summarization`] - Context summarization and compression
//! - [`streaming`] - Real-time response streaming and delivery
//! - [`generation`] - Response generation strategies and optimization
//! - [`utils`] - Utility functions and helper implementations
//!
//! ## Features
//!
//! - **Multi-turn Dialogue**: Sophisticated conversation state management
//! - **Memory System**: Long-term memory with intelligent relevance scoring
//! - **Safety Filtering**: Comprehensive content moderation and risk assessment
//! - **Real-time Streaming**: Natural typing simulation with backpressure handling
//! - **Quality Analysis**: Conversation health monitoring and optimization
//! - **Multiple Modes**: Chat, Assistant, Educational, Role-playing, and more
//! - **Persona Support**: Consistent character personalities and behavior
//! - **Context Summarization**: Intelligent context compression for long conversations
//! - **Advanced Generation**: Multiple generation strategies with adaptive optimization

pub mod analysis;
pub mod config;
pub mod generation;
pub mod memory;
pub mod pipeline;
pub mod reasoning;
pub mod safety;
pub mod state;
pub mod streaming;
pub mod summarization;
pub mod types;
pub mod utils;

// Re-export main types for backward compatibility and convenience
pub use pipeline::ConversationalPipeline;
pub use types::*;

// Re-export component types for easy access
pub use analysis::{ConversationAnalyzer, LinguisticAnalysis};
pub use config::{
    ConfigurationManager, ConfigurationPresets, ConfigurationValidator,
    ConversationalConfigBuilder, ConversationalConfigBuilder as ConfigBuilder,
};
pub use generation::{
    ContextBuilder as GenContextBuilder, GenerationOptimizer, GenerationStrategyManager,
    PromptFormatter, QualityValidator, ResponseGenerator,
};
pub use memory::{
    ConversationMemoryManager, LongTermMemoryManager, MemoryAnalysis, MemoryManager, MemoryUtils,
};
pub use safety::{
    EnhancedSafetyAssessment, EnhancedSafetyFilter, ExtendedSafetyConfig, SafetyAssessment,
    SafetyConfig, SafetyFilter,
};
pub use streaming::{
    ConversationalStreamingPipeline, QualityAnalyzer, ResponseChunker, StreamingCoordinator,
    StreamingManager, StreamingMetrics, TypingSimulator,
};
pub use summarization::{
    ContextSummarizer, SummarizationEngine, SummarizationMetadata, SummarizationResult,
};
pub use utils::{
    ConversationFormatter, ConversationHealthTracker, ConversationSerializer,
    MemoryUtils as UtilsMemory, StringUtils, TextAnalyzer, TextProcessor, TimeUtils,
};

// Legacy compatibility types (maintain exact same interface as before)
pub type ConversationalSystem = ConversationalPipeline<
    Box<
        dyn crate::core::traits::Model<
            Config = crate::AutoConfig,
            Input = crate::core::tensor::Tensor,
            Output = crate::core::tensor::Tensor,
        >,
    >,
    Box<dyn crate::core::traits::Tokenizer>,
>;

// Initialization functions for easy setup
use crate::core::traits::{Model, Tokenizer};
use crate::error::Result;
use trustformers_models::GenerativeModel;

/// Initialize a conversational pipeline with default configuration
pub fn init_conversational_pipeline<M, T>(
    model: M,
    tokenizer: T,
) -> Result<ConversationalPipeline<M, T>>
where
    M: Model + Send + Sync + GenerativeModel + 'static,
    T: Tokenizer + Send + Sync + 'static,
{
    ConversationalPipeline::new(model, tokenizer)
}

/// Initialize a conversational pipeline with custom configuration
pub fn init_conversational_pipeline_with_config<M, T>(
    model: M,
    tokenizer: T,
    config: ConversationalConfig,
) -> Result<ConversationalPipeline<M, T>>
where
    M: Model + Send + Sync + GenerativeModel + 'static,
    T: Tokenizer + Send + Sync + 'static,
{
    Ok(ConversationalPipeline::new(model, tokenizer)?.with_config(config))
}

/// Validate conversational configuration
pub fn validate_conversational_config(config: &ConversationalConfig) -> Result<()> {
    config::ConfigurationValidator::new().validate_config(config)
}

/// Create a conversational pipeline optimized for chat
pub fn create_chat_pipeline<M, T>(model: M, tokenizer: T) -> Result<ConversationalPipeline<M, T>>
where
    M: Model + Send + Sync + GenerativeModel + 'static,
    T: Tokenizer + Send + Sync + 'static,
{
    let config = config::ConfigurationPresets::chat_config();
    Ok(ConversationalPipeline::new(model, tokenizer)?.with_config(config))
}

/// Create a conversational pipeline optimized for assistant tasks
pub fn create_assistant_pipeline<M, T>(
    model: M,
    tokenizer: T,
) -> Result<ConversationalPipeline<M, T>>
where
    M: Model + Send + Sync + GenerativeModel + 'static,
    T: Tokenizer + Send + Sync + 'static,
{
    let config = config::ConfigurationPresets::assistant_config();
    Ok(ConversationalPipeline::new(model, tokenizer)?.with_config(config))
}

/// Create a conversational pipeline optimized for educational use
pub fn create_educational_pipeline<M, T>(
    model: M,
    tokenizer: T,
) -> Result<ConversationalPipeline<M, T>>
where
    M: Model + Send + Sync + GenerativeModel + 'static,
    T: Tokenizer + Send + Sync + 'static,
{
    let config = config::ConfigurationPresets::educational_config();
    Ok(ConversationalPipeline::new(model, tokenizer)?.with_config(config))
}

/// Create a conversational pipeline with custom persona
pub fn create_persona_pipeline<M, T>(
    model: M,
    tokenizer: T,
    persona: PersonaConfig,
) -> Result<ConversationalPipeline<M, T>>
where
    M: Model + Send + Sync + GenerativeModel + 'static,
    T: Tokenizer + Send + Sync + 'static,
{
    let mut config = ConversationalConfig::default();
    config.conversation_mode = ConversationMode::RolePlay;
    config.persona = Some(persona);
    Ok(ConversationalPipeline::new(model, tokenizer)?.with_config(config))
}

/// Create a conversational pipeline with streaming enabled
pub fn create_streaming_pipeline<M, T>(
    model: M,
    tokenizer: T,
) -> Result<ConversationalPipeline<M, T>>
where
    M: Model + Send + Sync + GenerativeModel + 'static,
    T: Tokenizer + Send + Sync + 'static,
{
    let config = config::ConfigurationPresets::streaming_optimized_config();
    Ok(ConversationalPipeline::new(model, tokenizer)?.with_config(config))
}

/// Get conversational pipeline capabilities
pub fn get_conversational_capabilities() -> ConversationalCapabilities {
    get_pipeline_capabilities()
}

/// Conversational pipeline capabilities
#[derive(Debug, Clone)]
pub struct ConversationalCapabilities {
    pub supported_conversation_modes: Vec<String>,
    pub supported_generation_strategies: Vec<String>,
    pub supported_summarization_strategies: Vec<String>,
    pub memory_management: bool,
    pub safety_filtering: bool,
    pub real_time_streaming: bool,
    pub persona_support: bool,
    pub context_summarization: bool,
    pub conversation_analysis: bool,
    pub multi_turn_dialogue: bool,
    pub conversation_repair: bool,
    pub quality_monitoring: bool,
}

/// Get pipeline capabilities
pub fn get_pipeline_capabilities() -> ConversationalCapabilities {
    ConversationalCapabilities {
        supported_conversation_modes: vec![
            "Chat".to_string(),
            "Assistant".to_string(),
            "InstructionFollowing".to_string(),
            "QuestionAnswering".to_string(),
            "RolePlay".to_string(),
            "Educational".to_string(),
        ],
        supported_generation_strategies: vec![
            "Sampling".to_string(),
            "TopK".to_string(),
            "TopP".to_string(),
            "Beam".to_string(),
            "Diverse".to_string(),
            "Contrastive".to_string(),
        ],
        supported_summarization_strategies: vec![
            "Extractive".to_string(),
            "Abstractive".to_string(),
            "Hybrid".to_string(),
        ],
        memory_management: true,
        safety_filtering: true,
        real_time_streaming: true,
        persona_support: true,
        context_summarization: true,
        conversation_analysis: true,
        multi_turn_dialogue: true,
        conversation_repair: true,
        quality_monitoring: true,
    }
}

/// Utility functions for common conversational patterns

/// Quick conversation assessment for immediate decision making
pub async fn quick_conversation_assessment<M, T>(
    pipeline: &ConversationalPipeline<M, T>,
    conversation_id: &str,
) -> Result<ConversationHealth>
where
    M: Model + Send + Sync + GenerativeModel,
    T: Tokenizer + Send + Sync,
{
    pipeline.assess_conversation_health(conversation_id).await
}

/// Start conversation with smart defaults based on conversation mode
pub async fn start_smart_conversation<M, T>(
    model: M,
    tokenizer: T,
    mode: ConversationMode,
) -> Result<ConversationalPipeline<M, T>>
where
    M: Model + Send + Sync + GenerativeModel + 'static,
    T: Tokenizer + Send + Sync + 'static,
{
    let config = match mode {
        ConversationMode::Chat => config::ConfigurationPresets::chat_config(),
        ConversationMode::Assistant => config::ConfigurationPresets::assistant_config(),
        ConversationMode::Educational => config::ConfigurationPresets::educational_config(),
        ConversationMode::RolePlay => config::ConfigurationPresets::roleplay_config(),
        ConversationMode::QuestionAnswering => config::ConfigurationPresets::qa_config(),
        ConversationMode::InstructionFollowing => {
            config::ConfigurationPresets::instruction_config()
        },
    };

    Ok(ConversationalPipeline::new(model, tokenizer)?.with_config(config))
}

/// Get conversation recommendations based on current state
pub async fn get_conversation_recommendations<M, T>(
    pipeline: &ConversationalPipeline<M, T>,
    conversation_id: &str,
) -> Result<Vec<String>>
where
    M: Model + Send + Sync + GenerativeModel,
    T: Tokenizer + Send + Sync,
{
    let health = pipeline.assess_conversation_health(conversation_id).await?;
    let mut recommendations = Vec::new();

    if health.overall_score < 0.7 {
        recommendations.push("Consider conversation repair or topic shift".to_string());
    }

    if health.coherence_score < 0.6 {
        recommendations.push("Focus on maintaining conversation coherence".to_string());
    }

    if health.engagement_score < 0.5 {
        recommendations.push(
            "Try to increase user engagement with questions or interesting topics".to_string(),
        );
    }

    if health.safety_score < 0.8 {
        recommendations.push("Review conversation for safety concerns".to_string());
    }

    if recommendations.is_empty() {
        recommendations.push("Conversation is healthy, continue current approach".to_string());
    }

    Ok(recommendations)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversational_capabilities() {
        let capabilities = get_conversational_capabilities();
        assert!(!capabilities.supported_conversation_modes.is_empty());
        assert!(capabilities.memory_management);
        assert!(capabilities.safety_filtering);
        assert!(capabilities.real_time_streaming);
        assert!(capabilities.persona_support);
    }

    #[test]
    fn test_configuration_validation() {
        let config = ConversationalConfig::default();
        let result = validate_conversational_config(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_configuration_presets() {
        let chat_config = config::ConfigurationPresets::chat_config();
        assert_eq!(chat_config.conversation_mode, ConversationMode::Chat);

        let assistant_config = config::ConfigurationPresets::assistant_config();
        assert_eq!(
            assistant_config.conversation_mode,
            ConversationMode::Assistant
        );

        let educational_config = config::ConfigurationPresets::educational_config();
        assert_eq!(
            educational_config.conversation_mode,
            ConversationMode::Educational
        );
    }

    #[test]
    fn test_builder_patterns() {
        let config = config::ConversationalConfigBuilder::new()
            .temperature(0.8)
            .max_response_tokens(1024)
            .conversation_mode(ConversationMode::Chat)
            .build();

        assert!(config.is_ok());
        let config = config.unwrap();
        assert_eq!(config.temperature, 0.8);
        assert_eq!(config.max_response_tokens, 1024);
        assert_eq!(config.conversation_mode, ConversationMode::Chat);
    }

    #[test]
    fn test_legacy_compatibility() {
        // Test that legacy type aliases work
        fn _test_conversational_system(_: ConversationalSystem) {}
    }

    #[test]
    fn test_module_integration() {
        // Test that all modules can be imported
        use super::analysis::*;
        use super::config::*;
        use super::generation::*;
        use super::memory::*;
        use super::safety::*;
        use super::streaming::*;
        use super::summarization::*;
        use super::types::*;
        use super::utils::*;

        // Basic integration test
        let config = ConversationalConfig::default();
        assert_eq!(config.conversation_mode, ConversationMode::Chat);
    }

    #[test]
    fn test_capabilities_completeness() {
        let capabilities = get_pipeline_capabilities();

        // Verify all expected conversation modes are supported
        let expected_modes = vec![
            "Chat",
            "Assistant",
            "InstructionFollowing",
            "QuestionAnswering",
            "RolePlay",
            "Educational",
        ];
        for mode in expected_modes {
            assert!(capabilities.supported_conversation_modes.contains(&mode.to_string()));
        }

        // Verify all expected generation strategies are supported
        let expected_strategies =
            vec!["Sampling", "TopK", "TopP", "Beam", "Diverse", "Contrastive"];
        for strategy in expected_strategies {
            assert!(capabilities.supported_generation_strategies.contains(&strategy.to_string()));
        }

        // Verify all expected summarization strategies are supported
        let expected_summarization = vec!["Extractive", "Abstractive", "Hybrid"];
        for strategy in expected_summarization {
            assert!(capabilities
                .supported_summarization_strategies
                .contains(&strategy.to_string()));
        }
    }

    #[test]
    fn test_conversation_modes() {
        // Test all conversation modes are valid
        let modes = [
            ConversationMode::Chat,
            ConversationMode::Assistant,
            ConversationMode::InstructionFollowing,
            ConversationMode::QuestionAnswering,
            ConversationMode::RolePlay,
            ConversationMode::Educational,
        ];

        for mode in modes {
            let mut config = ConversationalConfig::default();
            config.conversation_mode = mode;
            assert!(validate_conversational_config(&config).is_ok());
        }
    }
}
