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
