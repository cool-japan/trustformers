//! Response generation system for conversational AI pipeline.
//!
//! This module contains all functionality related to generating conversational responses,
//! including context preparation, prompt formatting, generation strategies, response
//! post-processing, quality validation, and streaming capabilities.

use super::types::*;
use crate::core::traits::{Model, Tokenizer};
use crate::error::{Result, TrustformersError};
use async_stream;
use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use trustformers_models::common_patterns::{
    GenerationConfig as ModelsGenerationConfig, GenerativeModel,
};

// ================================================================================================
// MAIN GENERATION SYSTEM
// ================================================================================================

/// Main response generation coordinator
#[derive(Debug)]
pub struct ResponseGenerator<M, T>
where
    M: Model + Send + Sync + GenerativeModel,
    T: Tokenizer + Send + Sync,
{
    model: Arc<M>,
    tokenizer: Arc<T>,
    context_builder: ContextBuilder,
    prompt_formatter: PromptFormatter,
    strategy_manager: GenerationStrategyManager,
    post_processor: ResponsePostProcessor,
    streaming_generator: StreamingGenerator<M, T>,
    quality_validator: QualityValidator,
    fallback_handler: FallbackHandler,
    generation_optimizer: GenerationOptimizer,
}

impl<M, T> ResponseGenerator<M, T>
where
    M: Model + Send + Sync + GenerativeModel + 'static,
    T: Tokenizer + Send + Sync,
{
    /// Create a new response generator
    pub fn new(model: M, tokenizer: T) -> Self {
        let model_arc = Arc::new(model);
        let tokenizer_arc = Arc::new(tokenizer);

        Self {
            model: model_arc.clone(),
            tokenizer: tokenizer_arc.clone(),
            context_builder: ContextBuilder::new(),
            prompt_formatter: PromptFormatter::new(),
            strategy_manager: GenerationStrategyManager::new(),
            post_processor: ResponsePostProcessor::new(),
            streaming_generator: StreamingGenerator::new(model_arc.clone(), tokenizer_arc.clone()),
            quality_validator: QualityValidator::new(),
            fallback_handler: FallbackHandler::new(),
            generation_optimizer: GenerationOptimizer::new(),
        }
    }

    /// Generate a response for the given conversation state and input
    pub async fn generate_response(
        &self,
        state: &ConversationState,
        input: &ConversationalInput,
        config: &ConversationalConfig,
    ) -> Result<GenerationResult> {
        let start_time = std::time::Instant::now();

        // Build enhanced context with memories and persona
        let context = self.context_builder.build_enhanced_context(state, config, &input.message)?;

        // Format prompt for the specific conversation mode
        let formatted_prompt =
            self.prompt_formatter.format_prompt(&context, config, &input.message)?;

        // Optimize generation parameters for the current context
        let optimized_config =
            self.generation_optimizer.optimize_parameters(config, state, &input.message)?;

        // Select and configure generation strategy
        let generation_strategy =
            self.strategy_manager.select_strategy(&optimized_config, state)?;

        // Generate response with retry mechanism
        let raw_response = match self
            .generate_with_strategy(&formatted_prompt, &generation_strategy, &optimized_config)
            .await
        {
            Ok(response) => response,
            Err(e) => {
                // Attempt fallback generation
                self.fallback_handler
                    .handle_generation_failure(
                        &formatted_prompt,
                        &e,
                        &optimized_config,
                        self.model.clone(),
                    )
                    .await?
            },
        };

        // Post-process and enhance the response
        let processed_response =
            self.post_processor.process_response(&raw_response, config, state)?;

        // Validate response quality
        let quality_score =
            self.quality_validator
                .validate_response(&processed_response, &input.message, state)?;

        let generation_time = start_time.elapsed().as_millis() as f64;
        let tokens_generated = self.estimate_token_count(&processed_response)?;

        Ok(GenerationResult {
            response: processed_response,
            generation_stats: GenerationStats {
                generation_time_ms: generation_time,
                tokens_generated,
                tokens_per_second: if generation_time > 0.0 {
                    (tokens_generated as f64) / (generation_time / 1000.0)
                } else {
                    0.0
                },
                confidence: quality_score.overall_confidence,
                truncated: quality_score.was_truncated,
            },
            quality_metrics: quality_score,
            strategy_used: generation_strategy,
        })
    }

    /// Generate streaming response
    pub async fn generate_streaming_response(
        &self,
        state: &ConversationState,
        input: &ConversationalInput,
        config: &ConversationalConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse>> + Send + '_>>> {
        self.streaming_generator
            .generate_streaming(
                state,
                input,
                config,
                &self.context_builder,
                &self.prompt_formatter,
                &self.generation_optimizer,
            )
            .await
    }

    /// Generate response using specific strategy
    async fn generate_with_strategy(
        &self,
        prompt: &str,
        strategy: &GenerationStrategyConfig,
        config: &ConversationalConfig,
    ) -> Result<String> {
        // Tokenize the prompt
        let tokenized = (*self.tokenizer).encode(prompt)?;

        // Convert to models generation config
        let models_config = self.create_models_config(strategy, config)?;

        // Generate using the model
        let response = (*self.model).generate(prompt, &models_config)?;

        Ok(response)
    }

    /// Create models generation config from strategy
    fn create_models_config(
        &self,
        strategy: &GenerationStrategyConfig,
        config: &ConversationalConfig,
    ) -> Result<ModelsGenerationConfig> {
        Ok(ModelsGenerationConfig {
            max_new_tokens: strategy.max_tokens,
            temperature: strategy.temperature,
            top_p: strategy.top_p,
            top_k: strategy.top_k,
            repetition_penalty: strategy.repetition_penalty,
            length_penalty: strategy.length_penalty,
            do_sample: strategy.do_sample,
            early_stopping: strategy.early_stopping,
            ..ModelsGenerationConfig::default()
        })
    }

    /// Estimate token count for text
    fn estimate_token_count(&self, text: &str) -> Result<usize> {
        match (*self.tokenizer).encode(text) {
            Ok(tokenized) => Ok(tokenized.input_ids.len()),
            Err(_) => Ok(text.len() / 4), // Fallback estimation
        }
    }
}

// ================================================================================================
// CONTEXT BUILDING
// ================================================================================================

/// Builds conversation context with memories, persona, and history
#[derive(Debug)]
pub struct ContextBuilder {
    memory_integrator: MemoryIntegrator,
    persona_formatter: PersonaFormatter,
    history_compiler: HistoryCompiler,
}

impl Default for ContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextBuilder {
    pub fn new() -> Self {
        Self {
            memory_integrator: MemoryIntegrator::new(),
            persona_formatter: PersonaFormatter::new(),
            history_compiler: HistoryCompiler::new(),
        }
    }

    /// Build enhanced conversation context
    pub fn build_enhanced_context(
        &self,
        state: &ConversationState,
        config: &ConversationalConfig,
        current_input: &str,
    ) -> Result<String> {
        let mut context = String::new();

        // Add system prompt if available
        if let Some(system_prompt) = &config.system_prompt {
            context.push_str(&format!("System: {}\n\n", system_prompt));
        }

        // Add persona information if available
        if let Some(persona_context) = self.persona_formatter.format_persona(config)? {
            context.push_str(&persona_context);
            context.push_str("\n\n");
        }

        // Add conversation summary if available
        if let Some(summary) = &state.context_summary {
            context.push_str(&format!("Previous conversation summary: {}\n\n", summary));
        }

        // Add relevant memories
        let memory_context = self.memory_integrator.integrate_memories(
            state,
            current_input,
            config.memory_config.max_memories.min(3),
        )?;
        if !memory_context.is_empty() {
            context.push_str(&memory_context);
            context.push('\n');
        }

        // Add recent conversation history
        let history_context = self
            .history_compiler
            .compile_history(state, config.max_context_tokens - context.len())?;
        context.push_str(&history_context);

        Ok(context)
    }
}

/// Integrates relevant memories into context
#[derive(Debug)]
pub struct MemoryIntegrator;

impl Default for MemoryIntegrator {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryIntegrator {
    pub fn new() -> Self {
        Self
    }

    pub fn integrate_memories(
        &self,
        state: &ConversationState,
        current_input: &str,
        max_memories: usize,
    ) -> Result<String> {
        let relevant_memories = state.get_relevant_memories(current_input, max_memories);

        if relevant_memories.is_empty() {
            return Ok(String::new());
        }

        let mut memory_context = String::from("Relevant context from previous conversations:\n");
        for memory in relevant_memories {
            memory_context.push_str(&format!("- {}\n", memory.content));
        }

        Ok(memory_context)
    }
}

/// Formats persona information for context
#[derive(Debug)]
pub struct PersonaFormatter;

impl Default for PersonaFormatter {
    fn default() -> Self {
        Self::new()
    }
}

impl PersonaFormatter {
    pub fn new() -> Self {
        Self
    }

    pub fn format_persona(&self, config: &ConversationalConfig) -> Result<Option<String>> {
        if let Some(persona) = &config.persona {
            let persona_context = format!(
                "You are {}. {}\n\nBackground: {}\n\nSpeaking style: {}",
                persona.name, persona.personality, persona.background, persona.speaking_style
            );
            Ok(Some(persona_context))
        } else {
            Ok(None)
        }
    }
}

/// Compiles conversation history into context
#[derive(Debug)]
pub struct HistoryCompiler;

impl Default for HistoryCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl HistoryCompiler {
    pub fn new() -> Self {
        Self
    }

    pub fn compile_history(
        &self,
        state: &ConversationState,
        max_context_length: usize,
    ) -> Result<String> {
        let recent_turns = state.get_recent_context(max_context_length);
        let mut history = String::new();

        for turn in recent_turns {
            let role_str = match turn.role {
                ConversationRole::User => "User",
                ConversationRole::Assistant => "Assistant",
                ConversationRole::System => "System",
            };
            history.push_str(&format!("{}: {}\n", role_str, turn.content));
        }

        Ok(history)
    }
}

// ================================================================================================
// PROMPT FORMATTING
// ================================================================================================

/// Formats prompts for different conversation modes
pub struct PromptFormatter {
    mode_formatters: HashMap<ConversationMode, Box<dyn ModeFormatter + Send + Sync>>,
}

impl std::fmt::Debug for PromptFormatter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PromptFormatter")
            .field(
                "mode_formatters",
                &format!("{} formatters", self.mode_formatters.len()),
            )
            .finish()
    }
}

impl Default for PromptFormatter {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptFormatter {
    pub fn new() -> Self {
        let mut mode_formatters: HashMap<ConversationMode, Box<dyn ModeFormatter + Send + Sync>> =
            HashMap::new();

        mode_formatters.insert(ConversationMode::Chat, Box::new(ChatFormatter));
        mode_formatters.insert(ConversationMode::Assistant, Box::new(AssistantFormatter));
        mode_formatters.insert(
            ConversationMode::InstructionFollowing,
            Box::new(InstructionFormatter),
        );
        mode_formatters.insert(ConversationMode::QuestionAnswering, Box::new(QAFormatter));
        mode_formatters.insert(ConversationMode::RolePlay, Box::new(RolePlayFormatter));
        mode_formatters.insert(
            ConversationMode::Educational,
            Box::new(EducationalFormatter),
        );

        Self { mode_formatters }
    }

    pub fn format_prompt(
        &self,
        context: &str,
        config: &ConversationalConfig,
        current_input: &str,
    ) -> Result<String> {
        let formatter = self.mode_formatters.get(&config.conversation_mode).ok_or_else(|| {
            TrustformersError::invalid_input(
                format!("No formatter for mode: {:?}", config.conversation_mode),
                Some("conversation_mode"),
                Some("supported conversation mode"),
                Some(format!("{:?}", config.conversation_mode)),
            )
        })?;

        formatter.format(context, config, current_input)
    }
}

/// Trait for mode-specific prompt formatting
trait ModeFormatter {
    fn format(&self, context: &str, config: &ConversationalConfig, input: &str) -> Result<String>;
}

/// Chat mode formatter
struct ChatFormatter;
impl ModeFormatter for ChatFormatter {
    fn format(
        &self,
        context: &str,
        _config: &ConversationalConfig,
        _input: &str,
    ) -> Result<String> {
        Ok(format!(
            "{}\nContinue the conversation naturally and helpfully.\n\nAssistant:",
            context
        ))
    }
}

/// Assistant mode formatter
struct AssistantFormatter;
impl ModeFormatter for AssistantFormatter {
    fn format(
        &self,
        context: &str,
        _config: &ConversationalConfig,
        _input: &str,
    ) -> Result<String> {
        Ok(format!(
            "{}\nProvide helpful assistance with the user's request.\n\nAssistant:",
            context
        ))
    }
}

/// Instruction-following mode formatter
struct InstructionFormatter;
impl ModeFormatter for InstructionFormatter {
    fn format(
        &self,
        context: &str,
        _config: &ConversationalConfig,
        _input: &str,
    ) -> Result<String> {
        Ok(format!(
            "{}\nFollow the user's instructions carefully and accurately.\n\nAssistant:",
            context
        ))
    }
}

/// Question-answering mode formatter
struct QAFormatter;
impl ModeFormatter for QAFormatter {
    fn format(
        &self,
        context: &str,
        _config: &ConversationalConfig,
        _input: &str,
    ) -> Result<String> {
        Ok(format!(
            "{}\nAnswer the user's question accurately and concisely.\n\nAssistant:",
            context
        ))
    }
}

/// Role-play mode formatter
struct RolePlayFormatter;
impl ModeFormatter for RolePlayFormatter {
    fn format(
        &self,
        context: &str,
        _config: &ConversationalConfig,
        _input: &str,
    ) -> Result<String> {
        Ok(format!(
            "{}\nStay in character and respond appropriately to the scenario.\n\nAssistant:",
            context
        ))
    }
}

/// Educational mode formatter
struct EducationalFormatter;
impl ModeFormatter for EducationalFormatter {
    fn format(
        &self,
        context: &str,
        _config: &ConversationalConfig,
        _input: &str,
    ) -> Result<String> {
        Ok(format!("{}\nProvide educational and informative responses to help the user learn.\n\nAssistant:", context))
    }
}

// ================================================================================================
// GENERATION STRATEGY MANAGEMENT
// ================================================================================================

/// Manages generation strategies and parameter selection
#[derive(Debug)]
pub struct GenerationStrategyManager {
    strategy_selector: StrategySelector,
    parameter_optimizer: ParameterOptimizer,
}

impl Default for GenerationStrategyManager {
    fn default() -> Self {
        Self::new()
    }
}

impl GenerationStrategyManager {
    pub fn new() -> Self {
        Self {
            strategy_selector: StrategySelector::new(),
            parameter_optimizer: ParameterOptimizer::new(),
        }
    }

    pub fn select_strategy(
        &self,
        config: &ConversationalConfig,
        state: &ConversationState,
    ) -> Result<GenerationStrategyConfig> {
        let base_strategy = self.strategy_selector.select_base_strategy(config)?;
        let optimized_strategy =
            self.parameter_optimizer.optimize_strategy(base_strategy, config, state)?;

        Ok(optimized_strategy)
    }
}

/// Configuration for a generation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStrategyConfig {
    pub strategy_type: GenerationStrategyType,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub max_tokens: usize,
    pub repetition_penalty: f32,
    pub length_penalty: f32,
    pub do_sample: bool,
    pub early_stopping: bool,
    pub diversity_penalty: f32,
    pub context_awareness: f32,
}

/// Types of generation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GenerationStrategyType {
    Sampling,
    TopK,
    TopP,
    Beam,
    Diverse,
    Contrastive,
}

/// Selects appropriate strategy based on context
#[derive(Debug)]
pub struct StrategySelector;

impl Default for StrategySelector {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategySelector {
    pub fn new() -> Self {
        Self
    }

    pub fn select_base_strategy(
        &self,
        config: &ConversationalConfig,
    ) -> Result<GenerationStrategyConfig> {
        let strategy = match config.conversation_mode {
            ConversationMode::QuestionAnswering => GenerationStrategyConfig {
                strategy_type: GenerationStrategyType::TopP,
                temperature: 0.3,
                top_p: 0.8,
                top_k: None,
                max_tokens: config.max_response_tokens,
                repetition_penalty: 1.1,
                length_penalty: 1.0,
                do_sample: true,
                early_stopping: true,
                diversity_penalty: 0.0,
                context_awareness: 0.9,
            },
            ConversationMode::RolePlay => GenerationStrategyConfig {
                strategy_type: GenerationStrategyType::Sampling,
                temperature: 0.8,
                top_p: 0.9,
                top_k: None,
                max_tokens: config.max_response_tokens,
                repetition_penalty: 1.2,
                length_penalty: 1.0,
                do_sample: true,
                early_stopping: false,
                diversity_penalty: 0.3,
                context_awareness: 0.8,
            },
            ConversationMode::Educational => GenerationStrategyConfig {
                strategy_type: GenerationStrategyType::TopP,
                temperature: 0.5,
                top_p: 0.85,
                top_k: None,
                max_tokens: config.max_response_tokens,
                repetition_penalty: 1.1,
                length_penalty: 1.2,
                do_sample: true,
                early_stopping: true,
                diversity_penalty: 0.1,
                context_awareness: 0.9,
            },
            ConversationMode::InstructionFollowing => GenerationStrategyConfig {
                strategy_type: GenerationStrategyType::TopK,
                temperature: 0.4,
                top_p: 0.9,
                top_k: Some(40),
                max_tokens: config.max_response_tokens,
                repetition_penalty: 1.05,
                length_penalty: 1.0,
                do_sample: true,
                early_stopping: true,
                diversity_penalty: 0.0,
                context_awareness: 0.95,
            },
            _ => GenerationStrategyConfig {
                strategy_type: GenerationStrategyType::TopP,
                temperature: config.temperature,
                top_p: config.top_p,
                top_k: config.top_k,
                max_tokens: config.max_response_tokens,
                repetition_penalty: 1.1,
                length_penalty: 1.0,
                do_sample: true,
                early_stopping: true,
                diversity_penalty: 0.1,
                context_awareness: 0.8,
            },
        };

        Ok(strategy)
    }
}

/// Optimizes strategy parameters based on context
#[derive(Debug)]
pub struct ParameterOptimizer;

impl Default for ParameterOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl ParameterOptimizer {
    pub fn new() -> Self {
        Self
    }

    pub fn optimize_strategy(
        &self,
        mut strategy: GenerationStrategyConfig,
        config: &ConversationalConfig,
        state: &ConversationState,
    ) -> Result<GenerationStrategyConfig> {
        // Adjust based on conversation health
        if state.health.engagement_score < 0.5 {
            strategy.temperature += 0.1; // Increase creativity for low engagement
            strategy.diversity_penalty += 0.1;
        }

        // Adjust based on conversation length
        if state.turns.len() > 10 {
            strategy.context_awareness += 0.1; // Increase context awareness for longer conversations
            strategy.repetition_penalty += 0.05; // Reduce repetition in long conversations
        }

        // Adjust based on persona requirements
        if config.persona.is_some() {
            strategy.context_awareness += 0.1; // Higher context awareness for persona consistency
        }

        // Ensure parameters are within valid ranges
        strategy.temperature = strategy.temperature.clamp(0.1, 2.0);
        strategy.top_p = strategy.top_p.clamp(0.1, 1.0);
        strategy.repetition_penalty = strategy.repetition_penalty.clamp(0.5, 2.0);
        strategy.diversity_penalty = strategy.diversity_penalty.clamp(0.0, 1.0);
        strategy.context_awareness = strategy.context_awareness.clamp(0.0, 1.0);

        Ok(strategy)
    }
}

// ================================================================================================
// GENERATION OPTIMIZATION
// ================================================================================================

/// Optimizes generation parameters for specific contexts
#[derive(Debug)]
pub struct GenerationOptimizer {
    performance_analyzer: PerformanceAnalyzer,
    adaptive_tuner: AdaptiveTuner,
}

impl Default for GenerationOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl GenerationOptimizer {
    pub fn new() -> Self {
        Self {
            performance_analyzer: PerformanceAnalyzer::new(),
            adaptive_tuner: AdaptiveTuner::new(),
        }
    }

    pub fn optimize_parameters(
        &self,
        config: &ConversationalConfig,
        state: &ConversationState,
        current_input: &str,
    ) -> Result<ConversationalConfig> {
        let mut optimized_config = config.clone();

        // Analyze performance metrics
        let performance_metrics = self.performance_analyzer.analyze_performance(state)?;

        // Apply adaptive tuning
        self.adaptive_tuner.tune_parameters(
            &mut optimized_config,
            &performance_metrics,
            current_input,
        )?;

        Ok(optimized_config)
    }
}

/// Analyzes conversation performance metrics
#[derive(Debug)]
pub struct PerformanceAnalyzer;

impl Default for PerformanceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceAnalyzer {
    pub fn new() -> Self {
        Self
    }

    pub fn analyze_performance(&self, state: &ConversationState) -> Result<PerformanceMetrics> {
        let mut metrics = PerformanceMetrics::default();

        // Calculate average response quality
        let quality_scores: Vec<f32> = state
            .turns
            .iter()
            .filter_map(|turn| turn.metadata.as_ref().map(|m| m.quality_score))
            .collect();

        if !quality_scores.is_empty() {
            metrics.avg_quality = quality_scores.iter().sum::<f32>() / quality_scores.len() as f32;
        }

        // Calculate engagement consistency
        let engagement_scores: Vec<f32> = state
            .turns
            .iter()
            .filter_map(|turn| {
                turn.metadata.as_ref().map(|m| match m.engagement_level {
                    EngagementLevel::VeryHigh => 1.0,
                    EngagementLevel::High => 0.8,
                    EngagementLevel::Medium => 0.6,
                    EngagementLevel::Low => 0.4,
                })
            })
            .collect();

        if !engagement_scores.is_empty() {
            metrics.avg_engagement =
                engagement_scores.iter().sum::<f32>() / engagement_scores.len() as f32;
        }

        // Calculate coherence metrics
        metrics.coherence_score = state.health.coherence_score;
        metrics.conversation_length = state.turns.len();
        metrics.memory_utilization = state.memories.len() as f32 / 100.0; // Normalize to 0-1

        Ok(metrics)
    }
}

/// Performance metrics for optimization
#[derive(Debug, Default)]
pub struct PerformanceMetrics {
    pub avg_quality: f32,
    pub avg_engagement: f32,
    pub coherence_score: f32,
    pub conversation_length: usize,
    pub memory_utilization: f32,
}

/// Adaptively tunes parameters based on performance
#[derive(Debug)]
pub struct AdaptiveTuner;

impl Default for AdaptiveTuner {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveTuner {
    pub fn new() -> Self {
        Self
    }

    pub fn tune_parameters(
        &self,
        config: &mut ConversationalConfig,
        metrics: &PerformanceMetrics,
        current_input: &str,
    ) -> Result<()> {
        // Adjust temperature based on engagement
        if metrics.avg_engagement < 0.5 {
            config.temperature = (config.temperature + 0.1).min(1.0);
        } else if metrics.avg_engagement > 0.8 {
            config.temperature = (config.temperature - 0.05).max(0.3);
        }

        // Adjust response length based on conversation flow
        if metrics.conversation_length > 20 && metrics.coherence_score < 0.7 {
            config.max_response_tokens = ((config.max_response_tokens as f32) * 0.8) as usize;
        }

        // Adjust memory usage based on utilization
        if metrics.memory_utilization > 0.8 {
            config.memory_config.max_memories =
                ((config.memory_config.max_memories as f32) * 0.9) as usize;
        }

        // Input-specific adjustments
        if current_input.contains('?') && current_input.len() < 50 {
            // Short questions - more focused responses
            config.top_p = (config.top_p - 0.1).max(0.7);
        }

        Ok(())
    }
}

// ================================================================================================
// RESPONSE POST-PROCESSING
// ================================================================================================

/// Processes and enhances generated responses
#[derive(Debug)]
pub struct ResponsePostProcessor {
    response_cleaner: ResponseCleaner,
    quality_enhancer: QualityEnhancer,
    safety_filter: ResponseSafetyFilter,
}

impl Default for ResponsePostProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl ResponsePostProcessor {
    pub fn new() -> Self {
        Self {
            response_cleaner: ResponseCleaner::new(),
            quality_enhancer: QualityEnhancer::new(),
            safety_filter: ResponseSafetyFilter::new(),
        }
    }

    pub fn process_response(
        &self,
        raw_response: &str,
        config: &ConversationalConfig,
        state: &ConversationState,
    ) -> Result<String> {
        // Clean the response
        let cleaned = self.response_cleaner.clean_response(raw_response)?;

        // Apply safety filtering
        let safe_response = self.safety_filter.filter_response(&cleaned, config)?;

        // Enhance quality
        let enhanced = self.quality_enhancer.enhance_response(&safe_response, config, state)?;

        Ok(enhanced)
    }
}

/// Cleans and formats generated responses
#[derive(Debug)]
pub struct ResponseCleaner;

impl Default for ResponseCleaner {
    fn default() -> Self {
        Self::new()
    }
}

impl ResponseCleaner {
    pub fn new() -> Self {
        Self
    }

    pub fn clean_response(&self, response: &str) -> Result<String> {
        let mut cleaned = response.trim().to_string();

        // Remove common generation artifacts
        cleaned = cleaned.replace("<|endoftext|>", "");
        cleaned = cleaned.replace("<|end|>", "");
        cleaned = cleaned.replace("<eos>", "");
        cleaned = cleaned.replace("<pad>", "");

        // Clean up whitespace
        cleaned = cleaned.replace("\n\n\n", "\n\n");
        cleaned = cleaned.trim().to_string();

        // Ensure proper sentence ending
        if !cleaned.is_empty() && !cleaned.ends_with(['.', '!', '?', ':', ';']) {
            cleaned.push('.');
        }

        // Remove incomplete sentences at the end
        if let Some(last_sentence_start) = cleaned.rfind(['.', '!', '?']) {
            let remaining = &cleaned[last_sentence_start + 1..].trim();
            if remaining.len() < 10 && !remaining.is_empty() {
                cleaned.truncate(last_sentence_start + 1);
            }
        }

        Ok(cleaned)
    }
}

/// Enhances response quality
#[derive(Debug)]
pub struct QualityEnhancer;

impl Default for QualityEnhancer {
    fn default() -> Self {
        Self::new()
    }
}

impl QualityEnhancer {
    pub fn new() -> Self {
        Self
    }

    pub fn enhance_response(
        &self,
        response: &str,
        config: &ConversationalConfig,
        state: &ConversationState,
    ) -> Result<String> {
        let mut enhanced = response.to_string();

        // Add persona-specific enhancements
        if let Some(persona) = &config.persona {
            enhanced = self.apply_persona_style(&enhanced, persona)?;
        }

        // Apply mode-specific enhancements
        enhanced = self.apply_mode_enhancements(&enhanced, &config.conversation_mode)?;

        // Ensure appropriate length
        enhanced = self.enforce_length_constraints(&enhanced, config)?;

        Ok(enhanced)
    }

    fn apply_persona_style(&self, response: &str, persona: &PersonaConfig) -> Result<String> {
        // This would apply persona-specific style adjustments
        // For now, just return the response as-is
        Ok(response.to_string())
    }

    fn apply_mode_enhancements(&self, response: &str, mode: &ConversationMode) -> Result<String> {
        match mode {
            ConversationMode::Educational => {
                // Could add educational formatting like bullet points, examples, etc.
                Ok(response.to_string())
            },
            ConversationMode::QuestionAnswering => {
                // Could ensure the response directly addresses the question
                Ok(response.to_string())
            },
            _ => Ok(response.to_string()),
        }
    }

    fn enforce_length_constraints(
        &self,
        response: &str,
        config: &ConversationalConfig,
    ) -> Result<String> {
        let max_chars = config.max_response_tokens * 4; // Rough estimation

        if response.len() > max_chars {
            let mut truncated = response.chars().take(max_chars - 3).collect::<String>();

            // Try to end at a sentence boundary
            if let Some(last_sentence) = truncated.rfind(['.', '!', '?']) {
                truncated.truncate(last_sentence + 1);
            } else {
                truncated.push_str("...");
            }

            Ok(truncated)
        } else {
            Ok(response.to_string())
        }
    }
}

/// Filters responses for safety
#[derive(Debug)]
pub struct ResponseSafetyFilter;

impl Default for ResponseSafetyFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl ResponseSafetyFilter {
    pub fn new() -> Self {
        Self
    }

    pub fn filter_response(&self, response: &str, config: &ConversationalConfig) -> Result<String> {
        if !config.enable_safety_filter {
            return Ok(response.to_string());
        }

        // Simple safety checks (would be replaced with more sophisticated filtering)
        let safety_violations = ["violence", "harmful", "inappropriate", "offensive"];

        let response_lower = response.to_lowercase();
        for violation in &safety_violations {
            if response_lower.contains(violation) {
                return Ok("I apologize, but I can't provide that response. Let me try a different approach.".to_string());
            }
        }

        Ok(response.to_string())
    }
}

// ================================================================================================
// QUALITY VALIDATION
// ================================================================================================

/// Validates and scores response quality
#[derive(Debug)]
pub struct QualityValidator {
    coherence_checker: CoherenceChecker,
    relevance_scorer: RelevanceScorer,
    fluency_analyzer: FluencyAnalyzer,
}

impl Default for QualityValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl QualityValidator {
    pub fn new() -> Self {
        Self {
            coherence_checker: CoherenceChecker::new(),
            relevance_scorer: RelevanceScorer::new(),
            fluency_analyzer: FluencyAnalyzer::new(),
        }
    }

    pub fn validate_response(
        &self,
        response: &str,
        input: &str,
        state: &ConversationState,
    ) -> Result<QualityMetrics> {
        let coherence_score = self.coherence_checker.check_coherence(response, state)?;
        let relevance_score = self.relevance_scorer.score_relevance(response, input)?;
        let fluency_score = self.fluency_analyzer.analyze_fluency(response)?;

        let overall_confidence = (coherence_score + relevance_score + fluency_score) / 3.0;

        Ok(QualityMetrics {
            coherence_score,
            relevance_score,
            fluency_score,
            overall_confidence,
            was_truncated: response.len() > 1000, // Simple heuristic
            safety_compliant: true,               // Would be determined by safety filter
        })
    }
}

/// Quality metrics for generated responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub coherence_score: f32,
    pub relevance_score: f32,
    pub fluency_score: f32,
    pub overall_confidence: f32,
    pub was_truncated: bool,
    pub safety_compliant: bool,
}

/// Checks response coherence
#[derive(Debug)]
pub struct CoherenceChecker;

impl Default for CoherenceChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl CoherenceChecker {
    pub fn new() -> Self {
        Self
    }

    pub fn check_coherence(&self, response: &str, state: &ConversationState) -> Result<f32> {
        let mut score: f32 = 0.7; // Base score

        // Check if response maintains context from conversation
        if let Some(last_turn) = state.turns.last() {
            if self.has_contextual_continuity(response, &last_turn.content) {
                score += 0.2;
            }
        }

        // Check for internal consistency
        if self.is_internally_consistent(response) {
            score += 0.1;
        }

        Ok(score.min(1.0_f32))
    }

    fn has_contextual_continuity(&self, response: &str, previous_content: &str) -> bool {
        // Simple keyword overlap check
        let response_lower = response.to_lowercase();
        let previous_lower = previous_content.to_lowercase();
        let response_words: Vec<&str> = response_lower.split_whitespace().collect();
        let previous_words: Vec<&str> = previous_lower.split_whitespace().collect();

        let overlap = response_words
            .iter()
            .filter(|word| previous_words.contains(word) && word.len() > 3)
            .count();

        overlap > 0
    }

    fn is_internally_consistent(&self, response: &str) -> bool {
        // Check for contradictory statements (simplified)
        let sentences: Vec<&str> = response.split(['.', '!', '?']).collect();
        sentences.len() > 1 && !response.contains("but") // Very simple heuristic
    }
}

/// Scores response relevance to input
#[derive(Debug)]
pub struct RelevanceScorer;

impl Default for RelevanceScorer {
    fn default() -> Self {
        Self::new()
    }
}

impl RelevanceScorer {
    pub fn new() -> Self {
        Self
    }

    pub fn score_relevance(&self, response: &str, input: &str) -> Result<f32> {
        let response_lower = response.to_lowercase();
        let input_lower = input.to_lowercase();
        let response_words: Vec<&str> = response_lower.split_whitespace().collect();
        let input_words: Vec<&str> = input_lower.split_whitespace().collect();

        let overlap = response_words
            .iter()
            .filter(|word| input_words.contains(word) && word.len() > 2)
            .count();

        let relevance = overlap as f32 / input_words.len().max(1) as f32;
        Ok(relevance.min(1.0))
    }
}

/// Analyzes response fluency
#[derive(Debug)]
pub struct FluencyAnalyzer;

impl Default for FluencyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl FluencyAnalyzer {
    pub fn new() -> Self {
        Self
    }

    pub fn analyze_fluency(&self, response: &str) -> Result<f32> {
        let mut score: f32 = 0.5;

        // Check for proper capitalization
        if response.chars().next().is_some_and(|c| c.is_uppercase()) {
            score += 0.1;
        }

        // Check for proper punctuation
        if response.contains(['.', '!', '?']) {
            score += 0.1;
        }

        // Check for reasonable sentence structure
        let words = response.split_whitespace().count();
        if (3..=100).contains(&words) {
            score += 0.2;
        }

        // Check for grammatical indicators
        if !response.contains("  ") && !response.contains("\t") {
            score += 0.1;
        }

        Ok(score.min(1.0_f32))
    }
}

// ================================================================================================
// STREAMING GENERATION
// ================================================================================================

/// Handles streaming response generation
#[derive(Debug)]
pub struct StreamingGenerator<M, T>
where
    M: Model + Send + Sync + GenerativeModel,
    T: Tokenizer + Send + Sync,
{
    model: Arc<M>,
    tokenizer: Arc<T>,
    chunk_processor: ChunkProcessor,
    stream_controller: StreamController,
}

impl<M, T> StreamingGenerator<M, T>
where
    M: Model + Send + Sync + GenerativeModel,
    T: Tokenizer + Send + Sync,
{
    pub fn new(model: Arc<M>, tokenizer: Arc<T>) -> Self {
        Self {
            model,
            tokenizer,
            chunk_processor: ChunkProcessor::new(),
            stream_controller: StreamController::new(),
        }
    }

    pub async fn generate_streaming(
        &self,
        state: &ConversationState,
        input: &ConversationalInput,
        config: &ConversationalConfig,
        context_builder: &ContextBuilder,
        prompt_formatter: &PromptFormatter,
        optimizer: &GenerationOptimizer,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse>> + Send + '_>>> {
        // Build context and prompt
        let context = context_builder.build_enhanced_context(state, config, &input.message)?;
        let prompt = prompt_formatter.format_prompt(&context, config, &input.message)?;
        let optimized_config = optimizer.optimize_parameters(config, state, &input.message)?;

        // Create streaming configuration
        let stream_config = self.create_stream_config(&optimized_config)?;

        // Generate streaming response
        let model = self.model.clone();
        let chunk_processor = self.chunk_processor.clone();
        let stream_controller = self.stream_controller.clone();

        let stream = async_stream::stream! {
            // Generate full response first (in a real implementation, this would be truly streaming)
            let models_config = ModelsGenerationConfig {
                max_new_tokens: optimized_config.max_response_tokens,
                temperature: optimized_config.temperature,
                top_p: optimized_config.top_p,
                top_k: optimized_config.top_k,
                do_sample: true,
                ..ModelsGenerationConfig::default()
            };

            let full_response = match (*model).generate(&prompt, &models_config) {
                Ok(response) => response,
                Err(e) => {
                    yield Err(e.into());
                    return;
                }
            };

            // Stream response in chunks
            let chunks = chunk_processor.create_chunks(&full_response, &stream_config)?;
            let total_chunks = chunks.len();

            for (index, chunk) in chunks.into_iter().enumerate() {
                let streaming_response = StreamingResponse {
                    chunk: chunk.clone(),
                    is_final: index == total_chunks - 1,
                    chunk_index: index,
                    total_chunks: Some(total_chunks),
                    metadata: None, // Could include metadata for each chunk
                };

                yield Ok(streaming_response);

                // Simulate typing delay
                if let Some(delay) = stream_controller.calculate_delay(&chunk, &stream_config) {
                    tokio::time::sleep(tokio::time::Duration::from_millis(delay)).await;
                }
            }
        };

        Ok(Box::pin(stream))
    }

    fn create_stream_config(&self, config: &ConversationalConfig) -> Result<StreamConfig> {
        Ok(StreamConfig {
            chunk_size: config.streaming_config.chunk_size,
            typing_delay_ms: config.streaming_config.typing_delay_ms,
            buffer_size: config.streaming_config.buffer_size,
            adaptive_chunking: true,
            preserve_word_boundaries: true,
        })
    }
}

/// Configuration for streaming
#[derive(Debug, Clone)]
pub struct StreamConfig {
    pub chunk_size: usize,
    pub typing_delay_ms: u64,
    pub buffer_size: usize,
    pub adaptive_chunking: bool,
    pub preserve_word_boundaries: bool,
}

/// Processes text into streaming chunks
#[derive(Debug, Clone)]
pub struct ChunkProcessor;

impl Default for ChunkProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl ChunkProcessor {
    pub fn new() -> Self {
        Self
    }

    pub fn create_chunks(&self, text: &str, config: &StreamConfig) -> Result<Vec<String>> {
        if config.preserve_word_boundaries {
            self.create_word_boundary_chunks(text, config)
        } else {
            self.create_character_chunks(text, config)
        }
    }

    fn create_word_boundary_chunks(
        &self,
        text: &str,
        config: &StreamConfig,
    ) -> Result<Vec<String>> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();

        for word in words {
            if current_chunk.split_whitespace().count() >= config.chunk_size
                && !current_chunk.is_empty()
            {
                chunks.push(current_chunk.clone());
                current_chunk.clear();
            }

            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(word);
        }

        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        Ok(chunks)
    }

    fn create_character_chunks(&self, text: &str, config: &StreamConfig) -> Result<Vec<String>> {
        let chunk_size = config.chunk_size * 5; // Approximate characters per word
        Ok(text
            .chars()
            .collect::<Vec<char>>()
            .chunks(chunk_size)
            .map(|chunk| chunk.iter().collect())
            .collect())
    }
}

/// Controls streaming flow and timing
#[derive(Debug, Clone)]
pub struct StreamController;

impl Default for StreamController {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamController {
    pub fn new() -> Self {
        Self
    }

    pub fn calculate_delay(&self, chunk: &str, config: &StreamConfig) -> Option<u64> {
        if config.typing_delay_ms == 0 {
            return None;
        }

        // Adjust delay based on chunk characteristics
        let base_delay = config.typing_delay_ms;
        let word_count = chunk.split_whitespace().count();

        // Longer chunks get slightly more delay
        let adjusted_delay = base_delay + (word_count as u64 * 10);

        Some(adjusted_delay)
    }
}

// ================================================================================================
// FALLBACK HANDLING
// ================================================================================================

/// Handles generation failures and provides fallbacks
pub struct FallbackHandler {
    fallback_strategies: Vec<Box<dyn FallbackStrategy + Send + Sync>>,
    error_analyzer: ErrorAnalyzer,
}

impl std::fmt::Debug for FallbackHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FallbackHandler")
            .field(
                "fallback_strategies",
                &format!("{} strategies", self.fallback_strategies.len()),
            )
            .field("error_analyzer", &self.error_analyzer)
            .finish()
    }
}

impl Default for FallbackHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl FallbackHandler {
    pub fn new() -> Self {
        let mut strategies: Vec<Box<dyn FallbackStrategy + Send + Sync>> = Vec::new();
        strategies.push(Box::new(SimpleRetryStrategy));
        strategies.push(Box::new(SimplifiedPromptStrategy));
        strategies.push(Box::new(DefaultResponseStrategy));

        Self {
            fallback_strategies: strategies,
            error_analyzer: ErrorAnalyzer::new(),
        }
    }

    pub async fn handle_generation_failure(
        &self,
        prompt: &str,
        error: &TrustformersError,
        config: &ConversationalConfig,
        model: Arc<dyn GenerativeModel + Send + Sync>,
    ) -> Result<String> {
        let error_type = self.error_analyzer.analyze_error(error)?;

        for strategy in &self.fallback_strategies {
            if strategy.can_handle(&error_type) {
                match strategy.attempt_fallback(prompt, config, model.clone()).await {
                    Ok(response) => return Ok(response),
                    Err(_) => continue, // Try next strategy
                }
            }
        }

        // If all strategies fail, return a default response
        Ok("I apologize, but I'm having trouble generating a response right now. Please try rephrasing your request.".to_string())
    }
}

/// Types of generation errors
#[derive(Debug, Clone)]
pub enum GenerationErrorType {
    ModelFailure,
    TokenizationError,
    ContextTooLong,
    SafetyViolation,
    Timeout,
    Unknown,
}

/// Analyzes generation errors
#[derive(Debug)]
pub struct ErrorAnalyzer;

impl Default for ErrorAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorAnalyzer {
    pub fn new() -> Self {
        Self
    }

    pub fn analyze_error(&self, error: &TrustformersError) -> Result<GenerationErrorType> {
        // Analyze the error and categorize it
        match error {
            crate::error::TrustformersError::Core(_) => Ok(GenerationErrorType::ModelFailure),
            crate::error::TrustformersError::Model { .. } => Ok(GenerationErrorType::ModelFailure),
            crate::error::TrustformersError::Pipeline { .. } => {
                Ok(GenerationErrorType::TokenizationError)
            },
            _ => Ok(GenerationErrorType::Unknown),
        }
    }
}

/// Trait for fallback strategies
#[async_trait]
trait FallbackStrategy {
    fn can_handle(&self, error_type: &GenerationErrorType) -> bool;
    async fn attempt_fallback(
        &self,
        prompt: &str,
        config: &ConversationalConfig,
        model: Arc<dyn GenerativeModel + Send + Sync>,
    ) -> Result<String>;
}

/// Simple retry strategy
struct SimpleRetryStrategy;

#[async_trait]
impl FallbackStrategy for SimpleRetryStrategy {
    fn can_handle(&self, error_type: &GenerationErrorType) -> bool {
        matches!(
            error_type,
            GenerationErrorType::ModelFailure | GenerationErrorType::Timeout
        )
    }

    async fn attempt_fallback(
        &self,
        prompt: &str,
        config: &ConversationalConfig,
        model: Arc<dyn GenerativeModel + Send + Sync>,
    ) -> Result<String> {
        // Retry with simplified parameters
        let simple_config = ModelsGenerationConfig {
            max_new_tokens: config.max_response_tokens.min(256),
            temperature: 0.7,
            top_p: 0.9,
            do_sample: true,
            ..ModelsGenerationConfig::default()
        };

        model.generate(prompt, &simple_config).map_err(Into::into)
    }
}

/// Simplified prompt strategy
struct SimplifiedPromptStrategy;

#[async_trait]
impl FallbackStrategy for SimplifiedPromptStrategy {
    fn can_handle(&self, error_type: &GenerationErrorType) -> bool {
        matches!(error_type, GenerationErrorType::ContextTooLong)
    }

    async fn attempt_fallback(
        &self,
        prompt: &str,
        config: &ConversationalConfig,
        model: Arc<dyn GenerativeModel + Send + Sync>,
    ) -> Result<String> {
        // Simplify prompt by taking only the last part
        let simplified_prompt = if prompt.len() > 1000 {
            let start = prompt.len() - 800;
            &prompt[start..]
        } else {
            prompt
        };

        let simple_config = ModelsGenerationConfig {
            max_new_tokens: config.max_response_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
            do_sample: true,
            ..ModelsGenerationConfig::default()
        };

        model.generate(simplified_prompt, &simple_config).map_err(Into::into)
    }
}

/// Default response strategy
struct DefaultResponseStrategy;

#[async_trait]
impl FallbackStrategy for DefaultResponseStrategy {
    fn can_handle(&self, _error_type: &GenerationErrorType) -> bool {
        true // Can handle any error as last resort
    }

    async fn attempt_fallback(
        &self,
        _prompt: &str,
        _config: &ConversationalConfig,
        _model: Arc<dyn GenerativeModel + Send + Sync>,
    ) -> Result<String> {
        Ok("I understand your message, but I'm having some technical difficulties generating a response right now. Could you please try again?".to_string())
    }
}

// ================================================================================================
// RESULT TYPES
// ================================================================================================

/// Result of response generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    pub response: String,
    pub generation_stats: GenerationStats,
    pub quality_metrics: QualityMetrics,
    pub strategy_used: GenerationStrategyConfig,
}

// ================================================================================================
// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_builder_creation() {
        let builder = ContextBuilder::new();
        // Just verify it can be created without panicking
        assert!(true);
    }

    #[test]
    fn test_prompt_formatter_creation() {
        let formatter = PromptFormatter::new();
        // Verify all conversation modes have formatters
        assert_eq!(formatter.mode_formatters.len(), 6);
    }

    #[test]
    fn test_generation_strategy_config_default() {
        let strategy = GenerationStrategyConfig {
            strategy_type: GenerationStrategyType::TopP,
            temperature: 0.7,
            top_p: 0.9,
            top_k: None,
            max_tokens: 512,
            repetition_penalty: 1.1,
            length_penalty: 1.0,
            do_sample: true,
            early_stopping: true,
            diversity_penalty: 0.1,
            context_awareness: 0.8,
        };

        assert_eq!(strategy.temperature, 0.7);
        assert_eq!(strategy.max_tokens, 512);
    }

    #[test]
    fn test_response_cleaner() {
        let cleaner = ResponseCleaner::new();
        let dirty_response = "Hello world<|endoftext|>\n\n\nThis is a test";
        let cleaned = cleaner.clean_response(dirty_response).unwrap();

        assert!(!cleaned.contains("<|endoftext|>"));
        assert!(!cleaned.contains("\n\n\n"));
        assert!(cleaned.ends_with('.'));
    }

    #[test]
    fn test_quality_metrics_creation() {
        let metrics = QualityMetrics {
            coherence_score: 0.8,
            relevance_score: 0.9,
            fluency_score: 0.85,
            overall_confidence: 0.85,
            was_truncated: false,
            safety_compliant: true,
        };

        assert_eq!(metrics.overall_confidence, 0.85);
        assert!(!metrics.was_truncated);
    }

    #[test]
    fn test_chunk_processor() {
        let processor = ChunkProcessor::new();
        let config = StreamConfig {
            chunk_size: 3,
            typing_delay_ms: 50,
            buffer_size: 100,
            adaptive_chunking: true,
            preserve_word_boundaries: true,
        };

        let text = "This is a test of the chunking system";
        let chunks = processor.create_chunks(text, &config).unwrap();

        assert!(!chunks.is_empty());
        // Each chunk should have roughly the specified number of words
        for chunk in &chunks[..chunks.len() - 1] {
            // Exclude last chunk which might be shorter
            assert!(chunk.split_whitespace().count() <= config.chunk_size + 1); // Allow some flexibility
        }
    }

    #[test]
    fn test_parameter_optimizer() {
        let optimizer = ParameterOptimizer::new();
        let strategy = GenerationStrategyConfig {
            strategy_type: GenerationStrategyType::TopP,
            temperature: 0.7,
            top_p: 0.9,
            top_k: None,
            max_tokens: 512,
            repetition_penalty: 1.1,
            length_penalty: 1.0,
            do_sample: true,
            early_stopping: true,
            diversity_penalty: 0.1,
            context_awareness: 0.8,
        };

        let config = ConversationalConfig::default();
        let state = ConversationState::new("test".to_string());
        let metrics = PerformanceMetrics::default();

        let optimized = optimizer.optimize_strategy(strategy.clone(), &config, &state).unwrap();

        // Verify parameters are within valid ranges
        assert!(optimized.temperature >= 0.1 && optimized.temperature <= 2.0);
        assert!(optimized.top_p >= 0.1 && optimized.top_p <= 1.0);
        assert!(optimized.repetition_penalty >= 0.5 && optimized.repetition_penalty <= 2.0);
    }

    #[test]
    fn test_relevance_scorer() {
        let scorer = RelevanceScorer::new();

        let input = "What is machine learning?";
        let relevant_response = "Machine learning is a subset of artificial intelligence";
        let irrelevant_response = "The weather is nice today";

        let relevant_score = scorer.score_relevance(relevant_response, input).unwrap();
        let irrelevant_score = scorer.score_relevance(irrelevant_response, input).unwrap();

        assert!(relevant_score > irrelevant_score);
    }

    #[test]
    fn test_fluency_analyzer() {
        let analyzer = FluencyAnalyzer::new();

        let good_text =
            "This is a well-formed sentence with proper capitalization and punctuation.";
        let poor_text = "this is bad text  with no punctuation and double spaces";

        let good_score = analyzer.analyze_fluency(good_text).unwrap();
        let poor_score = analyzer.analyze_fluency(poor_text).unwrap();

        assert!(good_score > poor_score);
    }

    #[test]
    fn test_safety_filter() {
        let filter = ResponseSafetyFilter::new();
        let config = ConversationalConfig::default();

        let safe_text = "This is a helpful and appropriate response.";
        let unsafe_text = "This contains violence and harmful content.";

        let safe_result = filter.filter_response(safe_text, &config).unwrap();
        let unsafe_result = filter.filter_response(unsafe_text, &config).unwrap();

        assert_eq!(safe_result, safe_text);
        assert_ne!(unsafe_result, unsafe_text); // Should be filtered
    }

    #[test]
    fn test_stream_controller() {
        let controller = StreamController::new();
        let config = StreamConfig {
            chunk_size: 5,
            typing_delay_ms: 50,
            buffer_size: 100,
            adaptive_chunking: true,
            preserve_word_boundaries: true,
        };

        let short_chunk = "Hello";
        let long_chunk = "This is a much longer chunk with many words";

        let short_delay = controller.calculate_delay(short_chunk, &config);
        let long_delay = controller.calculate_delay(long_chunk, &config);

        if let (Some(short), Some(long)) = (short_delay, long_delay) {
            assert!(long > short); // Longer chunks should have more delay
        }
    }

    #[test]
    fn test_error_analyzer() {
        let analyzer = ErrorAnalyzer::new();

        let model_error = crate::error::TrustformersError::Core(
            trustformers_core::errors::TrustformersError::runtime_error("Test error".to_string()),
        );
        let error_type = analyzer.analyze_error(&model_error).unwrap();

        assert!(matches!(error_type, GenerationErrorType::ModelFailure));
    }
}
