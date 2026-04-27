// ================================================================================================
// CONFIGURATION PRESETS
// ================================================================================================

use crate::pipeline::conversational::config::builder::{
    ConversationalConfigBuilder, MemoryConfigBuilder, RepairConfigBuilder, StreamingConfigBuilder,
};
use crate::pipeline::conversational::types::{
    ConversationMode, ConversationalConfig, RepairStrategy,
};

/// Predefined configuration presets for common use cases
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigurationPreset {
    /// Default balanced configuration
    Default,
    /// Creative conversation mode with higher temperature
    Creative,
    /// Focused and precise responses
    Focused,
    /// Educational tutor configuration
    Educational,
    /// Customer support assistant
    CustomerSupport,
    /// Medical assistant with safety focus
    Medical,
    /// Legal assistant with conservative settings
    Legal,
    /// Technical documentation helper
    Technical,
    /// Casual chat companion
    Casual,
    /// Professional business assistant
    Professional,
    /// Research and analysis helper
    Research,
    /// Creative writing assistant
    Writing,
    /// Code assistant for programming
    Coding,
    /// Gaming and entertainment
    Gaming,
    /// Language learning tutor
    LanguageLearning,
}

/// Factory for creating predefined configurations
pub struct ConfigurationPresets;

impl ConfigurationPresets {
    /// Get a preset configuration
    pub fn get_preset(preset: ConfigurationPreset) -> ConversationalConfig {
        match preset {
            ConfigurationPreset::Default => Self::default_config(),
            ConfigurationPreset::Creative => Self::creative_config(),
            ConfigurationPreset::Focused => Self::focused_config(),
            ConfigurationPreset::Educational => Self::educational_config(),
            ConfigurationPreset::CustomerSupport => Self::customer_support_config(),
            ConfigurationPreset::Medical => Self::medical_config(),
            ConfigurationPreset::Legal => Self::legal_config(),
            ConfigurationPreset::Technical => Self::technical_config(),
            ConfigurationPreset::Casual => Self::casual_config(),
            ConfigurationPreset::Professional => Self::professional_config(),
            ConfigurationPreset::Research => Self::research_config(),
            ConfigurationPreset::Writing => Self::writing_config(),
            ConfigurationPreset::Coding => Self::coding_config(),
            ConfigurationPreset::Gaming => Self::gaming_config(),
            ConfigurationPreset::LanguageLearning => Self::language_learning_config(),
        }
    }

    /// Default balanced configuration
    pub fn default_config() -> ConversationalConfig {
        ConversationalConfig::default()
    }

    /// Creative conversation with higher temperature and creativity
    pub fn creative_config() -> ConversationalConfig {
        ConversationalConfigBuilder::new()
            .temperature(0.9)
            .top_p(0.95)
            .top_k(Some(80))
            .max_response_tokens(1024)
            .conversation_mode(ConversationMode::Chat)
            .system_prompt(Some("You are a creative and imaginative AI assistant. Feel free to think outside the box and provide innovative responses.".to_string()))
            .memory_config(
                MemoryConfigBuilder::new()
                    .enabled(true)
                    .max_memories(150)
                    .persist_important_memories(true)
                    .build()
            )
            .streaming_config(
                StreamingConfigBuilder::new()
                    .enabled(true)
                    .chunk_size(8)
                    .typing_delay_ms(40)
                    .build()
            )
            .build_unchecked()
    }

    /// Focused and precise responses
    pub fn focused_config() -> ConversationalConfig {
        ConversationalConfigBuilder::new()
            .temperature(0.3)
            .top_p(0.7)
            .top_k(Some(20))
            .max_response_tokens(512)
            .conversation_mode(ConversationMode::QuestionAnswering)
            .system_prompt(Some("You are a precise and focused AI assistant. Provide accurate, concise, and well-structured responses.".to_string()))
            .build_unchecked()
    }

    /// Educational tutor configuration
    pub fn educational_config() -> ConversationalConfig {
        ConversationalConfigBuilder::new()
            .temperature(0.6)
            .top_p(0.85)
            .max_response_tokens(800)
            .conversation_mode(ConversationMode::Educational)
            .system_prompt(Some("You are an educational tutor. Explain concepts clearly, ask clarifying questions, and adapt your teaching style to the student's level.".to_string()))
            .memory_config(
                MemoryConfigBuilder::new()
                    .enabled(true)
                    .max_memories(200)
                    .persist_important_memories(true)
                    .decay_rate(0.98) // Slower decay for educational context
                    .build()
            )
            .repair_config(
                RepairConfigBuilder::new()
                    .enabled(true)
                    .detect_breakdowns(true)
                    .max_repair_attempts(5)
                    .strategies(vec![
                        RepairStrategy::Clarification,
                        RepairStrategy::Rephrase,
                        RepairStrategy::Redirect,
                    ])
                    .build()
            )
            .build_unchecked()
    }

    /// Customer support assistant
    pub fn customer_support_config() -> ConversationalConfig {
        ConversationalConfigBuilder::new()
            .temperature(0.5)
            .top_p(0.8)
            .max_response_tokens(600)
            .conversation_mode(ConversationMode::Assistant)
            .system_prompt(Some("You are a helpful customer support assistant. Be patient, empathetic, and solution-focused. Always try to understand the customer's needs.".to_string()))
            .enable_safety_filter(true)
            .memory_config(
                MemoryConfigBuilder::new()
                    .enabled(true)
                    .max_memories(100)
                    .persist_important_memories(true)
                    .build()
            )
            .repair_config(
                RepairConfigBuilder::new()
                    .enabled(true)
                    .detect_breakdowns(true)
                    .max_repair_attempts(3)
                    .strategies(vec![
                        RepairStrategy::Clarification,
                        RepairStrategy::Rephrase,
                    ])
                    .build()
            )
            .build_unchecked()
    }

    /// Medical assistant with enhanced safety
    pub fn medical_config() -> ConversationalConfig {
        ConversationalConfigBuilder::new()
            .temperature(0.2)
            .top_p(0.6)
            .max_response_tokens(1000)
            .conversation_mode(ConversationMode::Assistant)
            .system_prompt(Some("You are a medical information assistant. Provide accurate health information but always recommend consulting healthcare professionals for medical advice.".to_string()))
            .enable_safety_filter(true)
            .memory_config(
                MemoryConfigBuilder::new()
                    .enabled(true)
                    .max_memories(120)
                    .persist_important_memories(true)
                    .build()
            )
            .build_unchecked()
    }

    /// Legal assistant with conservative settings
    pub fn legal_config() -> ConversationalConfig {
        ConversationalConfigBuilder::new()
            .temperature(0.1)
            .top_p(0.5)
            .max_response_tokens(1200)
            .conversation_mode(ConversationMode::Assistant)
            .system_prompt(Some("You are a legal information assistant. Provide general legal information but always advise consulting qualified legal professionals for specific legal advice.".to_string()))
            .enable_safety_filter(true)
            .build_unchecked()
    }

    /// Technical documentation helper
    pub fn technical_config() -> ConversationalConfig {
        ConversationalConfigBuilder::new()
            .temperature(0.4)
            .top_p(0.75)
            .max_response_tokens(1500)
            .conversation_mode(ConversationMode::InstructionFollowing)
            .system_prompt(Some("You are a technical documentation assistant. Provide clear, accurate, and well-structured technical information with examples where appropriate.".to_string()))
            .memory_config(
                MemoryConfigBuilder::new()
                    .enabled(true)
                    .max_memories(180)
                    .build()
            )
            .build_unchecked()
    }

    /// Casual chat companion
    pub fn casual_config() -> ConversationalConfig {
        ConversationalConfigBuilder::new()
            .temperature(0.8)
            .top_p(0.9)
            .max_response_tokens(400)
            .conversation_mode(ConversationMode::Chat)
            .system_prompt(Some("You are a friendly and casual conversation partner. Be warm, engaging, and conversational.".to_string()))
            .streaming_config(
                StreamingConfigBuilder::new()
                    .enabled(true)
                    .chunk_size(5)
                    .typing_delay_ms(60)
                    .build()
            )
            .build_unchecked()
    }

    /// Professional business assistant
    pub fn professional_config() -> ConversationalConfig {
        ConversationalConfigBuilder::new()
            .temperature(0.5)
            .top_p(0.8)
            .max_response_tokens(800)
            .conversation_mode(ConversationMode::Assistant)
            .system_prompt(Some("You are a professional business assistant. Communicate clearly, professionally, and efficiently while maintaining a helpful attitude.".to_string()))
            .memory_config(
                MemoryConfigBuilder::new()
                    .enabled(true)
                    .max_memories(150)
                    .persist_important_memories(true)
                    .build()
            )
            .build_unchecked()
    }

    /// Research and analysis helper
    pub fn research_config() -> ConversationalConfig {
        ConversationalConfigBuilder::new()
            .temperature(0.3)
            .top_p(0.7)
            .max_response_tokens(2000)
            .conversation_mode(ConversationMode::QuestionAnswering)
            .system_prompt(Some("You are a research assistant. Provide thorough, well-researched responses with clear reasoning and evidence-based information.".to_string()))
            .memory_config(
                MemoryConfigBuilder::new()
                    .enabled(true)
                    .max_memories(250)
                    .persist_important_memories(true)
                    .decay_rate(0.99) // Very slow decay for research context
                    .build()
            )
            .build_unchecked()
    }

    /// Creative writing assistant
    pub fn writing_config() -> ConversationalConfig {
        ConversationalConfigBuilder::new()
            .temperature(0.8)
            .top_p(0.95)
            .top_k(Some(100))
            .max_response_tokens(1500)
            .conversation_mode(ConversationMode::Chat)
            .system_prompt(Some("You are a creative writing assistant. Help with storytelling, character development, plot ideas, and writing techniques. Be imaginative and supportive.".to_string()))
            .memory_config(
                MemoryConfigBuilder::new()
                    .enabled(true)
                    .max_memories(200)
                    .persist_important_memories(true)
                    .build()
            )
            .build_unchecked()
    }

    /// Code assistant for programming
    pub fn coding_config() -> ConversationalConfig {
        ConversationalConfigBuilder::new()
            .temperature(0.2)
            .top_p(0.8)
            .max_response_tokens(2000)
            .conversation_mode(ConversationMode::InstructionFollowing)
            .system_prompt(Some("You are a programming assistant. Provide clear, well-commented code examples, explain programming concepts, and help debug issues.".to_string()))
            .memory_config(
                MemoryConfigBuilder::new()
                    .enabled(true)
                    .max_memories(150)
                    .build()
            )
            .build_unchecked()
    }

    /// Gaming and entertainment
    pub fn gaming_config() -> ConversationalConfig {
        ConversationalConfigBuilder::new()
            .temperature(0.9)
            .top_p(0.95)
            .max_response_tokens(800)
            .conversation_mode(ConversationMode::RolePlay)
            .system_prompt(Some("You are an entertaining gaming companion. Be fun, engaging, and creative. Adapt to different gaming scenarios and roleplay situations.".to_string()))
            .memory_config(
                MemoryConfigBuilder::new()
                    .enabled(true)
                    .max_memories(120)
                    .build()
            )
            .streaming_config(
                StreamingConfigBuilder::new()
                    .enabled(true)
                    .chunk_size(6)
                    .typing_delay_ms(50)
                    .build()
            )
            .build_unchecked()
    }

    /// Language learning tutor
    pub fn language_learning_config() -> ConversationalConfig {
        ConversationalConfigBuilder::new()
            .temperature(0.6)
            .top_p(0.85)
            .max_response_tokens(600)
            .conversation_mode(ConversationMode::Educational)
            .system_prompt(Some("You are a language learning tutor. Help students practice languages, explain grammar, provide corrections, and encourage language use.".to_string()))
            .memory_config(
                MemoryConfigBuilder::new()
                    .enabled(true)
                    .max_memories(180)
                    .persist_important_memories(true)
                    .decay_rate(0.97)
                    .build()
            )
            .repair_config(
                RepairConfigBuilder::new()
                    .enabled(true)
                    .detect_breakdowns(true)
                    .max_repair_attempts(4)
                    .strategies(vec![
                        RepairStrategy::Clarification,
                        RepairStrategy::Rephrase,
                    ])
                    .build()
            )
            .build_unchecked()
    }

    /// Chat configuration for casual conversation
    pub fn chat_config() -> ConversationalConfig {
        ConversationalConfigBuilder::new()
            .temperature(0.7)
            .top_p(0.9)
            .max_response_tokens(800)
            .conversation_mode(ConversationMode::Chat)
            .system_prompt(Some("You are a friendly and helpful AI assistant. Engage in natural, conversational dialogue while being informative and supportive.".to_string()))
            .memory_config(
                MemoryConfigBuilder::new()
                    .enabled(true)
                    .max_memories(100)
                    .persist_important_memories(true)
                    .build()
            )
            .streaming_config(
                StreamingConfigBuilder::new()
                    .enabled(true)
                    .chunk_size(10)
                    .typing_delay_ms(50)
                    .build()
            )
            .build_unchecked()
    }

    /// Assistant configuration for task-oriented interaction
    pub fn assistant_config() -> ConversationalConfig {
        ConversationalConfigBuilder::new()
            .temperature(0.6)
            .top_p(0.85)
            .max_response_tokens(1000)
            .conversation_mode(ConversationMode::Assistant)
            .system_prompt(Some("You are a capable AI assistant focused on helping users complete tasks efficiently and accurately. Provide clear, actionable guidance.".to_string()))
            .memory_config(
                MemoryConfigBuilder::new()
                    .enabled(true)
                    .max_memories(120)
                    .persist_important_memories(true)
                    .build()
            )
            // TODO: Implement analysis_config when AnalysisConfigBuilder is available
            // .analysis_config(
            //     AnalysisConfigBuilder::new()
            //         .enabled(true)
            //         .sentiment_analysis(true)
            //         .topic_tracking(true)
            //         .entity_recognition(true)
            //         .build()
            // )
            .build_unchecked()
    }

    /// Roleplay configuration for character-based conversation
    pub fn roleplay_config() -> ConversationalConfig {
        ConversationalConfigBuilder::new()
            .temperature(0.8)
            .top_p(0.92)
            .max_response_tokens(900)
            .conversation_mode(ConversationMode::RolePlay)
            .system_prompt(Some("You are engaging in roleplay conversation. Stay in character while maintaining appropriate boundaries and being entertaining.".to_string()))
            .memory_config(
                MemoryConfigBuilder::new()
                    .enabled(true)
                    .max_memories(200)
                    .persist_important_memories(true)
                    .decay_rate(0.95)
                    .build()
            )
            .streaming_config(
                StreamingConfigBuilder::new()
                    .enabled(true)
                    .chunk_size(5)
                    .typing_delay_ms(40)
                    .build()
            )
            .build_unchecked()
    }

    /// Question answering configuration for factual responses
    pub fn qa_config() -> ConversationalConfig {
        ConversationalConfigBuilder::new()
            .temperature(0.5)
            .top_p(0.8)
            .max_response_tokens(600)
            .conversation_mode(ConversationMode::QuestionAnswering)
            .system_prompt(Some("You are a knowledgeable AI that provides accurate, well-sourced answers to questions. Focus on factual information and cite reasoning when helpful.".to_string()))
            .memory_config(
                MemoryConfigBuilder::new()
                    .enabled(true)
                    .max_memories(80)
                    .persist_important_memories(false)
                    .build()
            )
            // TODO: Implement analysis_config when AnalysisConfigBuilder is available
            // .analysis_config(
            //     AnalysisConfigBuilder::new()
            //         .enabled(true)
            //         .topic_tracking(true)
            //         .entity_recognition(true)
            //         .build()
            // )
            .build_unchecked()
    }

    /// Instruction following configuration for task completion
    pub fn instruction_config() -> ConversationalConfig {
        ConversationalConfigBuilder::new()
            .temperature(0.4)
            .top_p(0.75)
            .max_response_tokens(1200)
            .conversation_mode(ConversationMode::InstructionFollowing)
            .system_prompt(Some("You are an AI assistant that follows instructions carefully and precisely. Break down complex tasks into clear steps.".to_string()))
            .memory_config(
                MemoryConfigBuilder::new()
                    .enabled(true)
                    .max_memories(60)
                    .persist_important_memories(true)
                    .build()
            )
            // TODO: Implement reasoning_config when ReasoningConfigBuilder is available
            // .reasoning_config(
            //     ReasoningConfigBuilder::new()
            //         .enabled(true)
            //         .timeout_ms(3000)
            //         .build()
            // )
            .build_unchecked()
    }

    /// Streaming optimized configuration for real-time interaction
    pub fn streaming_optimized_config() -> ConversationalConfig {
        ConversationalConfigBuilder::new()
            .temperature(0.7)
            .top_p(0.9)
            .max_response_tokens(600)
            .conversation_mode(ConversationMode::Chat)
            .system_prompt(Some(
                "You are optimized for streaming conversation with natural, flowing responses."
                    .to_string(),
            ))
            .streaming_config(
                StreamingConfigBuilder::new()
                    .enabled(true)
                    .chunk_size(3)
                    .typing_delay_ms(20)
                    .buffer_size(1024)
                    .build(),
            )
            .build_unchecked()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Preset existence tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_preset_exists() {
        let cfg = ConfigurationPresets::get_preset(ConfigurationPreset::Default);
        assert!(
            cfg.temperature >= 0.0,
            "default temperature must be non-negative"
        );
    }

    #[test]
    fn test_creative_preset_exists() {
        let cfg = ConfigurationPresets::get_preset(ConfigurationPreset::Creative);
        assert!(
            cfg.temperature > 0.0,
            "creative temperature must be positive"
        );
    }

    #[test]
    fn test_focused_preset_exists() {
        let cfg = ConfigurationPresets::get_preset(ConfigurationPreset::Focused);
        assert!(
            cfg.temperature > 0.0,
            "focused temperature must be positive"
        );
    }

    #[test]
    fn test_educational_preset_exists() {
        let cfg = ConfigurationPresets::get_preset(ConfigurationPreset::Educational);
        assert!(
            cfg.max_response_tokens > 0,
            "educational max_response_tokens must be positive"
        );
    }

    #[test]
    fn test_customer_support_preset_exists() {
        let cfg = ConfigurationPresets::get_preset(ConfigurationPreset::CustomerSupport);
        assert!(
            cfg.max_response_tokens > 0,
            "customer support max_response_tokens must be positive"
        );
    }

    #[test]
    fn test_medical_preset_exists() {
        let cfg = ConfigurationPresets::get_preset(ConfigurationPreset::Medical);
        assert!(
            cfg.enable_safety_filter,
            "medical preset must enable safety filter"
        );
    }

    #[test]
    fn test_legal_preset_exists() {
        let cfg = ConfigurationPresets::get_preset(ConfigurationPreset::Legal);
        assert!(
            cfg.enable_safety_filter,
            "legal preset must enable safety filter"
        );
    }

    #[test]
    fn test_technical_preset_exists() {
        let cfg = ConfigurationPresets::get_preset(ConfigurationPreset::Technical);
        assert!(
            cfg.max_response_tokens > 0,
            "technical max_response_tokens must be positive"
        );
    }

    #[test]
    fn test_casual_preset_exists() {
        let cfg = ConfigurationPresets::get_preset(ConfigurationPreset::Casual);
        assert!(cfg.temperature > 0.0, "casual temperature must be positive");
    }

    #[test]
    fn test_professional_preset_exists() {
        let cfg = ConfigurationPresets::get_preset(ConfigurationPreset::Professional);
        assert!(
            cfg.max_response_tokens > 0,
            "professional max_response_tokens must be positive"
        );
    }

    #[test]
    fn test_research_preset_exists() {
        let cfg = ConfigurationPresets::get_preset(ConfigurationPreset::Research);
        assert!(
            cfg.max_response_tokens >= 2000,
            "research should allow long responses"
        );
    }

    #[test]
    fn test_writing_preset_exists() {
        let cfg = ConfigurationPresets::get_preset(ConfigurationPreset::Writing);
        assert!(
            cfg.temperature > 0.0,
            "writing temperature must be positive"
        );
    }

    #[test]
    fn test_coding_preset_exists() {
        let cfg = ConfigurationPresets::get_preset(ConfigurationPreset::Coding);
        assert!(
            cfg.max_response_tokens >= 2000,
            "coding should allow long responses"
        );
    }

    #[test]
    fn test_gaming_preset_exists() {
        let cfg = ConfigurationPresets::get_preset(ConfigurationPreset::Gaming);
        assert!(cfg.temperature > 0.0, "gaming temperature must be positive");
    }

    #[test]
    fn test_language_learning_preset_exists() {
        let cfg = ConfigurationPresets::get_preset(ConfigurationPreset::LanguageLearning);
        assert!(
            cfg.max_response_tokens > 0,
            "language learning max_response_tokens must be positive"
        );
    }

    // -----------------------------------------------------------------------
    // Temperature ordering tests: creative/casual must be higher than focused/coding
    // -----------------------------------------------------------------------

    #[test]
    fn test_creative_temperature_higher_than_focused() {
        let creative = ConfigurationPresets::creative_config();
        let focused = ConfigurationPresets::focused_config();
        assert!(
            creative.temperature > focused.temperature,
            "creative ({}) should have higher temperature than focused ({})",
            creative.temperature,
            focused.temperature
        );
    }

    #[test]
    fn test_writing_temperature_higher_than_coding() {
        let writing = ConfigurationPresets::writing_config();
        let coding = ConfigurationPresets::coding_config();
        assert!(
            writing.temperature > coding.temperature,
            "writing ({}) should have higher temperature than coding ({})",
            writing.temperature,
            coding.temperature
        );
    }

    #[test]
    fn test_casual_temperature_higher_than_legal() {
        let casual = ConfigurationPresets::casual_config();
        let legal = ConfigurationPresets::legal_config();
        assert!(
            casual.temperature > legal.temperature,
            "casual ({}) should have higher temperature than legal ({})",
            casual.temperature,
            legal.temperature
        );
    }

    #[test]
    fn test_medical_temperature_lower_than_creative() {
        let medical = ConfigurationPresets::medical_config();
        let creative = ConfigurationPresets::creative_config();
        assert!(
            medical.temperature < creative.temperature,
            "medical ({}) should have lower temperature than creative ({})",
            medical.temperature,
            creative.temperature
        );
    }

    // -----------------------------------------------------------------------
    // max_tokens limits
    // -----------------------------------------------------------------------

    #[test]
    fn test_research_max_tokens_large() {
        let cfg = ConfigurationPresets::research_config();
        assert!(
            cfg.max_response_tokens >= 2000,
            "research should allow >=2000 tokens"
        );
    }

    #[test]
    fn test_coding_max_tokens_large() {
        let cfg = ConfigurationPresets::coding_config();
        assert!(
            cfg.max_response_tokens >= 2000,
            "coding should allow >=2000 tokens"
        );
    }

    #[test]
    fn test_casual_max_tokens_smaller_than_research() {
        let casual = ConfigurationPresets::casual_config();
        let research = ConfigurationPresets::research_config();
        assert!(
            casual.max_response_tokens < research.max_response_tokens,
            "casual max_tokens ({}) should be less than research ({})",
            casual.max_response_tokens,
            research.max_response_tokens
        );
    }

    // -----------------------------------------------------------------------
    // System prompt non-empty
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_presets_have_system_prompt() {
        let all_presets = [
            ConfigurationPreset::Default,
            ConfigurationPreset::Creative,
            ConfigurationPreset::Focused,
            ConfigurationPreset::Educational,
            ConfigurationPreset::CustomerSupport,
            ConfigurationPreset::Medical,
            ConfigurationPreset::Legal,
            ConfigurationPreset::Technical,
            ConfigurationPreset::Casual,
            ConfigurationPreset::Professional,
            ConfigurationPreset::Research,
            ConfigurationPreset::Writing,
            ConfigurationPreset::Coding,
            ConfigurationPreset::Gaming,
            ConfigurationPreset::LanguageLearning,
        ];

        for preset in all_presets {
            let cfg = ConfigurationPresets::get_preset(preset);
            let prompt = cfg.system_prompt.expect("every preset must have a system prompt");
            assert!(
                !prompt.is_empty(),
                "system prompt for {:?} must not be empty",
                preset
            );
        }
    }

    // -----------------------------------------------------------------------
    // Safety filter
    // -----------------------------------------------------------------------

    #[test]
    fn test_medical_safety_filter_enabled() {
        let cfg = ConfigurationPresets::medical_config();
        assert!(
            cfg.enable_safety_filter,
            "medical preset must enable safety filter"
        );
    }

    #[test]
    fn test_legal_safety_filter_enabled() {
        let cfg = ConfigurationPresets::legal_config();
        assert!(
            cfg.enable_safety_filter,
            "legal preset must enable safety filter"
        );
    }

    #[test]
    fn test_customer_support_safety_filter_enabled() {
        let cfg = ConfigurationPresets::customer_support_config();
        assert!(
            cfg.enable_safety_filter,
            "customer support preset must enable safety filter"
        );
    }

    // -----------------------------------------------------------------------
    // Memory configuration
    // -----------------------------------------------------------------------

    #[test]
    fn test_educational_memory_enabled() {
        let cfg = ConfigurationPresets::educational_config();
        assert!(
            cfg.memory_config.enabled,
            "educational preset memory must be enabled"
        );
    }

    #[test]
    fn test_research_memory_max_entries_large() {
        let cfg = ConfigurationPresets::research_config();
        assert!(
            cfg.memory_config.max_memories >= 200,
            "research preset should store at least 200 memories"
        );
    }

    #[test]
    fn test_educational_memory_persist_important() {
        let cfg = ConfigurationPresets::educational_config();
        assert!(
            cfg.memory_config.persist_important_memories,
            "educational preset must persist important memories"
        );
    }

    // -----------------------------------------------------------------------
    // Streaming configuration
    // -----------------------------------------------------------------------

    #[test]
    fn test_streaming_optimized_config_has_streaming() {
        let cfg = ConfigurationPresets::streaming_optimized_config();
        assert!(
            cfg.streaming_config.enabled,
            "streaming optimized config must have streaming enabled"
        );
    }

    #[test]
    fn test_casual_streaming_enabled() {
        let cfg = ConfigurationPresets::casual_config();
        assert!(
            cfg.streaming_config.enabled,
            "casual config must have streaming enabled"
        );
    }

    // -----------------------------------------------------------------------
    // Conversation mode assignment
    // -----------------------------------------------------------------------

    #[test]
    fn test_gaming_is_roleplay_mode() {
        let cfg = ConfigurationPresets::gaming_config();
        assert_eq!(
            cfg.conversation_mode,
            ConversationMode::RolePlay,
            "gaming preset should use RolePlay mode"
        );
    }

    #[test]
    fn test_educational_is_educational_mode() {
        let cfg = ConfigurationPresets::educational_config();
        assert_eq!(
            cfg.conversation_mode,
            ConversationMode::Educational,
            "educational preset should use Educational mode"
        );
    }

    #[test]
    fn test_coding_is_instruction_following_mode() {
        let cfg = ConfigurationPresets::coding_config();
        assert_eq!(
            cfg.conversation_mode,
            ConversationMode::InstructionFollowing,
            "coding preset should use InstructionFollowing mode"
        );
    }

    // -----------------------------------------------------------------------
    // top_p in valid range
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_presets_top_p_in_range() {
        let all_presets = [
            ConfigurationPreset::Default,
            ConfigurationPreset::Creative,
            ConfigurationPreset::Focused,
            ConfigurationPreset::Educational,
            ConfigurationPreset::CustomerSupport,
            ConfigurationPreset::Medical,
            ConfigurationPreset::Legal,
            ConfigurationPreset::Technical,
            ConfigurationPreset::Casual,
            ConfigurationPreset::Professional,
            ConfigurationPreset::Research,
            ConfigurationPreset::Writing,
            ConfigurationPreset::Coding,
            ConfigurationPreset::Gaming,
            ConfigurationPreset::LanguageLearning,
        ];

        for preset in all_presets {
            let cfg = ConfigurationPresets::get_preset(preset);
            assert!(
                cfg.top_p >= 0.0 && cfg.top_p <= 1.0,
                "top_p={} for {:?} must be in [0.0, 1.0]",
                cfg.top_p,
                preset
            );
        }
    }

    // -----------------------------------------------------------------------
    // Repair configuration
    // -----------------------------------------------------------------------

    #[test]
    fn test_customer_support_repair_enabled() {
        let cfg = ConfigurationPresets::customer_support_config();
        assert!(
            cfg.repair_config.enabled,
            "customer support must have repair enabled"
        );
    }

    #[test]
    fn test_educational_repair_has_strategies() {
        let cfg = ConfigurationPresets::educational_config();
        assert!(
            !cfg.repair_config.repair_strategies.is_empty(),
            "educational preset must specify repair strategies"
        );
    }

    // -----------------------------------------------------------------------
    // Utility / named constructors
    // -----------------------------------------------------------------------

    #[test]
    fn test_chat_config_chat_mode() {
        let cfg = ConfigurationPresets::chat_config();
        assert_eq!(
            cfg.conversation_mode,
            ConversationMode::Chat,
            "chat_config must use Chat mode"
        );
    }

    #[test]
    fn test_assistant_config_assistant_mode() {
        let cfg = ConfigurationPresets::assistant_config();
        assert_eq!(
            cfg.conversation_mode,
            ConversationMode::Assistant,
            "assistant_config must use Assistant mode"
        );
    }

    #[test]
    fn test_qa_config_question_answering_mode() {
        let cfg = ConfigurationPresets::qa_config();
        assert_eq!(
            cfg.conversation_mode,
            ConversationMode::QuestionAnswering,
            "qa_config must use QuestionAnswering mode"
        );
    }

    #[test]
    fn test_instruction_config_instruction_following_mode() {
        let cfg = ConfigurationPresets::instruction_config();
        assert_eq!(
            cfg.conversation_mode,
            ConversationMode::InstructionFollowing,
            "instruction_config must use InstructionFollowing mode"
        );
    }
}
