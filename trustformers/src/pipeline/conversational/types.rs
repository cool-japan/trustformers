//! Core types and data structures for conversational AI pipeline.
//!
//! This module contains all the essential types used throughout the conversational pipeline
//! system, including configuration types, data structures for conversation management,
//! analysis metadata, and enums for various classifications.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::generation::GenerationConfig;

// ================================================================================================
// CONFIGURATION TYPES
// ================================================================================================

/// Configuration for conversational pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationalConfig {
    /// Maximum number of turns to keep in conversation history
    pub max_history_turns: usize,
    /// Maximum total tokens for conversation context
    pub max_context_tokens: usize,
    /// Whether to enable context summarization when history is too long
    pub enable_summarization: bool,
    /// Temperature for response generation
    pub temperature: f32,
    /// Top-p for nucleus sampling
    pub top_p: f32,
    /// Top-k for top-k sampling
    pub top_k: Option<usize>,
    /// Maximum tokens to generate for response
    pub max_response_tokens: usize,
    /// System prompt/personality for the conversation
    pub system_prompt: Option<String>,
    /// Whether to enable safety filtering
    pub enable_safety_filter: bool,
    /// Conversation mode (chat, assistant, instruction-following)
    pub conversation_mode: ConversationMode,
    /// Whether to enable conversation state persistence
    pub enable_persistence: bool,
    /// Persona configuration for role-playing
    pub persona: Option<PersonaConfig>,
    /// Context summarization configuration
    pub summarization_config: SummarizationConfig,
    /// Memory management settings
    pub memory_config: MemoryConfig,
    /// Advanced generation parameters
    pub generation_config: GenerationConfig,
    /// Conversation repair settings
    pub repair_config: RepairConfig,
    /// Streaming response configuration
    pub streaming_config: StreamingConfig,
}

impl Default for ConversationalConfig {
    fn default() -> Self {
        Self {
            max_history_turns: 20,
            max_context_tokens: 4096,
            enable_summarization: true,
            temperature: 0.7,
            top_p: 0.9,
            top_k: Some(50),
            max_response_tokens: 512,
            system_prompt: Some(
                "You are a helpful, harmless, and honest AI assistant.".to_string(),
            ),
            enable_safety_filter: true,
            conversation_mode: ConversationMode::Chat,
            enable_persistence: false,
            persona: None,
            summarization_config: SummarizationConfig::default(),
            memory_config: MemoryConfig::default(),
            generation_config: GenerationConfig::default(),
            repair_config: RepairConfig::default(),
            streaming_config: StreamingConfig::default(),
        }
    }
}

/// Persona configuration for role-playing conversations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaConfig {
    /// Name of the persona
    pub name: String,
    /// Description of the persona's personality
    pub personality: String,
    /// Background information
    pub background: String,
    /// Speaking style guidelines
    pub speaking_style: String,
    /// Knowledge areas and expertise
    pub expertise: Vec<String>,
    /// Behavioral constraints
    pub constraints: Vec<String>,
}

/// Configuration for context summarization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummarizationConfig {
    /// Enable automatic summarization
    pub enabled: bool,
    /// Threshold for triggering summarization (tokens)
    pub trigger_threshold: usize,
    /// Target length for summaries (tokens)
    pub target_length: usize,
    /// Summarization strategy
    pub strategy: SummarizationStrategy,
    /// Preserve recent turns (don't summarize)
    pub preserve_recent_turns: usize,
}

impl Default for SummarizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            trigger_threshold: 3000,
            target_length: 200,
            strategy: SummarizationStrategy::Hybrid,
            preserve_recent_turns: 3,
        }
    }
}

/// Memory management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Enable conversation memory optimization
    pub enabled: bool,
    /// Memory compression threshold
    pub compression_threshold: f32,
    /// Important memory persistence
    pub persist_important_memories: bool,
    /// Memory decay rate
    pub decay_rate: f32,
    /// Maximum memory entries
    pub max_memories: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            compression_threshold: 0.7,
            persist_important_memories: true,
            decay_rate: 0.95,
            max_memories: 100,
        }
    }
}

/// Conversation repair configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairConfig {
    /// Enable conversation repair
    pub enabled: bool,
    /// Detect conversation breakdowns
    pub detect_breakdowns: bool,
    /// Auto-repair attempts
    pub max_repair_attempts: usize,
    /// Repair strategies
    pub repair_strategies: Vec<RepairStrategy>,
}

impl Default for RepairConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            detect_breakdowns: true,
            max_repair_attempts: 3,
            repair_strategies: vec![
                RepairStrategy::Clarification,
                RepairStrategy::Rephrase,
                RepairStrategy::Redirect,
            ],
        }
    }
}

/// Streaming response configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Enable streaming responses
    pub enabled: bool,
    /// Chunk size for streaming
    pub chunk_size: usize,
    /// Buffer size
    pub buffer_size: usize,
    /// Typing delay simulation (ms)
    pub typing_delay_ms: u64,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            chunk_size: 10,
            buffer_size: 100,
            typing_delay_ms: 50,
        }
    }
}

// ================================================================================================
// CORE DATA TYPES
// ================================================================================================

/// A single turn in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    /// Role of the speaker (user, assistant, system)
    pub role: ConversationRole,
    /// Content of the message
    pub content: String,
    /// Timestamp of the turn
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Optional metadata (emotions, intent, etc.)
    pub metadata: Option<ConversationMetadata>,
    /// Token count for this turn
    pub token_count: usize,
}

/// Metadata for conversation analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConversationMetadata {
    /// Detected emotion/sentiment
    pub sentiment: Option<String>,
    /// Intent classification
    pub intent: Option<String>,
    /// Confidence scores
    #[serde(default)]
    pub confidence: f32,
    /// Topic classifications
    pub topics: Vec<String>,
    /// Safety flags
    pub safety_flags: Vec<String>,
    /// Named entities
    pub entities: Vec<EntityMention>,
    /// Conversation quality score
    #[serde(default)]
    pub quality_score: f32,
    /// Engagement level
    pub engagement_level: EngagementLevel,
    /// Reasoning type
    pub reasoning_type: Option<ReasoningType>,
}

/// Named entity mention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityMention {
    pub text: String,
    pub entity_type: String,
    pub confidence: f32,
    pub start_pos: usize,
    pub end_pos: usize,
}

/// Conversation state management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationState {
    /// Unique conversation ID
    pub conversation_id: String,
    /// History of conversation turns
    pub turns: Vec<ConversationTurn>,
    /// Current context summary
    pub context_summary: Option<String>,
    /// Total token count in conversation
    pub total_tokens: usize,
    /// Conversation statistics
    pub stats: ConversationStats,
    /// Custom state variables for context tracking
    pub variables: HashMap<String, String>,
    /// Conversation memories
    pub memories: Vec<ConversationMemory>,
    /// Current conversation health
    pub health: ConversationHealth,
    /// Multi-turn reasoning context
    pub reasoning_context: Option<ReasoningContext>,
}

impl ConversationState {
    pub fn new(conversation_id: String) -> Self {
        let now = chrono::Utc::now();
        Self {
            conversation_id,
            turns: Vec::new(),
            context_summary: None,
            total_tokens: 0,
            stats: ConversationStats {
                user_turns: 0,
                assistant_turns: 0,
                avg_response_length: 0.0,
                start_time: now,
                last_interaction: now,
                topics_discussed: Vec::new(),
            },
            variables: HashMap::new(),
            memories: Vec::new(),
            health: ConversationHealth {
                overall_score: 1.0,
                coherence_score: 1.0,
                engagement_score: 1.0,
                safety_score: 1.0,
                responsiveness_score: 1.0,
                context_relevance_score: 1.0,
                recommendations: Vec::new(),
                issues: Vec::new(),
                last_breakdown: None,
                repair_attempts: 0,
            },
            reasoning_context: None,
        }
    }

    /// Add a new turn to the conversation
    pub fn add_turn(&mut self, turn: ConversationTurn) {
        self.total_tokens += turn.token_count;

        match turn.role {
            ConversationRole::User => self.stats.user_turns += 1,
            ConversationRole::Assistant => {
                self.stats.assistant_turns += 1;
                // Update average response length
                let total_length: usize = self
                    .turns
                    .iter()
                    .filter(|t| matches!(t.role, ConversationRole::Assistant))
                    .map(|t| t.content.len())
                    .sum();
                self.stats.avg_response_length =
                    total_length as f32 / self.stats.assistant_turns as f32;
            },
            ConversationRole::System => {},
        }

        self.stats.last_interaction = turn.timestamp;

        // Add topics if available
        if let Some(metadata) = &turn.metadata {
            for topic in &metadata.topics {
                if !self.stats.topics_discussed.contains(topic) {
                    self.stats.topics_discussed.push(topic.clone());
                }
            }
        }

        self.turns.push(turn);
    }

    /// Get recent turns within token limit
    pub fn get_recent_context(&self, max_tokens: usize) -> Vec<&ConversationTurn> {
        let mut context = Vec::new();
        let mut token_count = 0;

        for turn in self.turns.iter().rev() {
            if token_count + turn.token_count > max_tokens {
                break;
            }
            token_count += turn.token_count;
            context.push(turn);
        }

        context.reverse();
        context
    }

    /// Trim history to keep within limits
    pub fn trim_history(&mut self, max_turns: usize, max_tokens: usize) {
        // Remove old turns if exceeding turn limit
        if self.turns.len() > max_turns {
            let keep_count = max_turns;
            self.turns = self.turns.split_off(self.turns.len() - keep_count);
        }

        // Remove old turns if exceeding token limit
        while self.total_tokens > max_tokens && !self.turns.is_empty() {
            let removed = self.turns.remove(0);
            self.total_tokens -= removed.token_count;
        }
    }

    /// Set a context variable
    pub fn set_variable(&mut self, key: String, value: String) {
        self.variables.insert(key, value);
    }

    /// Get a context variable
    pub fn get_variable(&self, key: &str) -> Option<&String> {
        self.variables.get(key)
    }

    /// Add a memory to the conversation
    pub fn add_memory(&mut self, memory: ConversationMemory) {
        self.memories.push(memory);

        // Sort by importance and keep only the most important ones
        self.memories.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());
        if self.memories.len() > 100 {
            // Max memories limit
            self.memories.truncate(100);
        }
    }

    /// Update conversation health
    pub fn update_health(&mut self, coherence: f32, engagement: f32, safety: f32) {
        self.health.coherence_score = coherence;
        self.health.engagement_score = engagement;
        self.health.safety_score = safety;
        self.health.overall_score = (coherence + engagement + safety) / 3.0;
    }

    /// Check if conversation needs repair
    pub fn needs_repair(&self) -> bool {
        self.health.overall_score < 0.6
            || self.health.coherence_score < 0.5
            || self.health.engagement_score < 0.4
    }

    /// Start reasoning context
    pub fn start_reasoning(&mut self, goal: Option<String>) {
        self.reasoning_context = Some(ReasoningContext {
            reasoning_chain: Vec::new(),
            current_goal: goal,
            evidence: Vec::new(),
            assumptions: Vec::new(),
            confidence: 1.0,
        });
    }

    /// Add reasoning step
    pub fn add_reasoning_step(&mut self, step: ReasoningStep) {
        if let Some(ref mut context) = self.reasoning_context {
            context.reasoning_chain.push(step);
        }
    }

    /// Get relevant memories for current context
    pub fn get_relevant_memories(&self, query: &str, limit: usize) -> Vec<&ConversationMemory> {
        let mut scored_memories: Vec<_> = self
            .memories
            .iter()
            .map(|memory| {
                let relevance = self.calculate_memory_relevance(memory, query);
                (memory, relevance)
            })
            .collect();

        scored_memories.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored_memories.into_iter().take(limit).map(|(memory, _)| memory).collect()
    }

    /// Calculate memory relevance to current query
    fn calculate_memory_relevance(&self, memory: &ConversationMemory, query: &str) -> f32 {
        // Simple relevance calculation based on keyword overlap
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();
        let memory_lower = memory.content.to_lowercase();
        let memory_words: Vec<&str> = memory_lower.split_whitespace().collect();

        let overlap = query_words.iter().filter(|word| memory_words.contains(word)).count();

        let relevance = overlap as f32 / query_words.len().max(1) as f32;

        // Weight by importance and recency
        relevance * memory.importance * 0.5
            + (1.0 - (chrono::Utc::now() - memory.last_accessed).num_hours() as f32 / (24.0 * 7.0))
                * 0.3
    }
}

/// Conversation memory for long-term context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMemory {
    pub id: String,
    pub content: String,
    pub importance: f32,
    pub last_accessed: chrono::DateTime<chrono::Utc>,
    pub access_count: usize,
    pub memory_type: MemoryType,
    pub tags: Vec<String>,
}

/// Conversation health tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationHealth {
    pub overall_score: f32,
    pub coherence_score: f32,
    pub engagement_score: f32,
    pub safety_score: f32,
    pub responsiveness_score: f32,
    pub context_relevance_score: f32,
    pub last_breakdown: Option<chrono::DateTime<chrono::Utc>>,
    pub repair_attempts: usize,
    pub recommendations: Vec<String>,
    pub issues: Vec<String>,
}

/// Multi-turn reasoning context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningContext {
    pub reasoning_chain: Vec<ReasoningStep>,
    pub current_goal: Option<String>,
    pub evidence: Vec<String>,
    pub assumptions: Vec<String>,
    pub confidence: f32,
}

/// Individual reasoning step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub step_type: ReasoningType,
    pub description: String,
    pub inputs: Vec<String>,
    pub output: String,
    pub confidence: f32,
}

/// Statistics about the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationStats {
    /// Number of turns by user
    pub user_turns: usize,
    /// Number of turns by assistant
    pub assistant_turns: usize,
    /// Average response length
    pub avg_response_length: f32,
    /// Conversation start time
    pub start_time: chrono::DateTime<chrono::Utc>,
    /// Last interaction time
    pub last_interaction: chrono::DateTime<chrono::Utc>,
    /// Topics discussed
    pub topics_discussed: Vec<String>,
}

/// Input for conversational pipeline
#[derive(Debug, Clone)]
pub struct ConversationalInput {
    /// User's message
    pub message: String,
    /// Optional conversation ID (if continuing existing conversation)
    pub conversation_id: Option<String>,
    /// Optional additional context
    pub context: Option<String>,
    /// Override configuration for this specific input
    pub config_override: Option<ConversationalConfig>,
}

/// Output from conversational pipeline
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConversationalOutput {
    /// Assistant's response
    pub response: String,
    /// Conversation ID
    pub conversation_id: String,
    /// Updated conversation state
    pub conversation_state: ConversationState,
    /// Response metadata
    pub metadata: ConversationMetadata,
    /// Generation statistics
    pub generation_stats: GenerationStats,
}

/// Statistics about the response generation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GenerationStats {
    /// Time taken to generate response
    pub generation_time_ms: f64,
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Generation speed (tokens per second)
    pub tokens_per_second: f64,
    /// Model confidence in the response
    pub confidence: f32,
    /// Whether response was truncated
    pub truncated: bool,
}

// ================================================================================================
// ENUMS
// ================================================================================================

/// Different conversation modes
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConversationMode {
    /// Free-form chat conversation
    Chat,
    /// Task-oriented assistant
    Assistant,
    /// Instruction-following mode
    InstructionFollowing,
    /// Question-answering mode
    QuestionAnswering,
    /// Role-playing conversation
    RolePlay,
    /// Educational/tutoring mode
    Educational,
}

/// Strategy for summarizing conversation context
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SummarizationStrategy {
    /// Extract key points and topics
    Extractive,
    /// Generate abstractive summary
    Abstractive,
    /// Hybrid approach
    Hybrid,
}

/// Strategies for repairing conversation flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RepairStrategy {
    /// Ask clarifying questions
    Clarification,
    /// Rephrase or restart
    Rephrase,
    /// Topic redirection
    Redirect,
    /// Context reset
    Reset,
}

/// Role in conversation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConversationRole {
    User,
    Assistant,
    System,
}

/// Engagement level classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum EngagementLevel {
    Low,
    #[default]
    Medium,
    High,
    VeryHigh,
}

/// Type of reasoning detected
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReasoningType {
    Logical,
    Causal,
    Analogical,
    Creative,
    Mathematical,
    Emotional,
}

/// Types of memories that can be stored
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryType {
    Fact,
    Preference,
    Experience,
    Goal,
    Relationship,
}

/// Health assessment status values
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthStatus {
    Excellent,
    Good,
    Fair,
    Poor,
    Critical,
}

/// Direction of trends in metrics
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TrendDirection {
    Improving,
    Stable,
    Declining,
    Volatile,
}

// ================================================================================================
// ADDITIONAL CONFIGURATION TYPES
// ================================================================================================

/// Configuration for conversation analysis features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Enable health analysis
    pub enable_health_analysis: bool,
    /// Enable engagement tracking
    pub enable_engagement_tracking: bool,
    /// Enable quality assessment
    pub enable_quality_assessment: bool,
    /// Enable linguistic analysis
    pub enable_linguistic_analysis: bool,
    /// Minimum confidence threshold for analysis
    pub min_confidence_threshold: f32,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            enable_health_analysis: true,
            enable_engagement_tracking: true,
            enable_quality_assessment: true,
            enable_linguistic_analysis: true,
            min_confidence_threshold: 0.5,
        }
    }
}

/// Configuration for safety filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyFilterConfig {
    /// Enable safety filtering
    pub enabled: bool,
    /// Safety rules to apply
    pub rules: Vec<SafetyRule>,
    /// Moderation level
    pub moderation_level: ModerationLevel,
    /// Block threshold
    pub block_threshold: f32,
    /// Enable toxicity detection
    pub toxicity_detection: bool,
    /// Enable harmful content detection
    pub harmful_content_detection: bool,
    /// Enable bias detection
    pub bias_detection: bool,
    /// Custom safety rules
    pub custom_rules: Vec<SafetyRule>,
}

impl Default for SafetyFilterConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rules: vec![
                SafetyRule {
                    name: "toxicity".to_string(),
                    pattern: ".*".to_string(),
                    threshold: 0.8,
                    action: SafetyAction::Block,
                    severity: SafetySeverity::High,
                },
                SafetyRule {
                    name: "harassment".to_string(),
                    pattern: ".*".to_string(),
                    threshold: 0.7,
                    action: SafetyAction::Block,
                    severity: SafetySeverity::High,
                },
                SafetyRule {
                    name: "explicit_content".to_string(),
                    pattern: ".*".to_string(),
                    threshold: 0.9,
                    action: SafetyAction::Block,
                    severity: SafetySeverity::Medium,
                },
            ],
            moderation_level: ModerationLevel::Moderate,
            block_threshold: 0.7,
            toxicity_detection: true,
            harmful_content_detection: true,
            bias_detection: true,
            custom_rules: vec![],
        }
    }
}

/// Individual safety rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyRule {
    /// Rule name/identifier
    pub name: String,
    /// Pattern to match (regex or keywords)
    pub pattern: String,
    /// Detection threshold (0.0-1.0)
    pub threshold: f32,
    /// Action to take when triggered
    pub action: SafetyAction,
    /// Severity level
    pub severity: SafetySeverity,
}

/// Actions that can be taken by safety filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetyAction {
    /// Block the content completely
    Block,
    /// Warn but allow through
    Warn,
    /// Request clarification
    Clarify,
    /// Log for review
    Log,
    /// Modify the content
    Modify,
}

/// Severity levels for safety violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Content moderation levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModerationLevel {
    Permissive,
    Moderate,
    Strict,
    VeryStrict,
    Custom,
}

/// Configuration for reasoning features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    /// Enable reasoning system
    pub enabled: bool,
    /// Enable multi-hop reasoning
    pub enable_multi_hop: bool,
    /// Enable evidence tracking
    pub enable_evidence_tracking: bool,
    /// Maximum reasoning depth
    pub max_reasoning_depth: usize,
    /// Confidence threshold for reasoning
    pub reasoning_confidence_threshold: f32,
    /// Enable reasoning chain visualization
    pub enable_reasoning_chains: bool,
    /// Timeout in milliseconds
    pub timeout_ms: u64,
}

impl Default for ReasoningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            enable_multi_hop: true,
            enable_evidence_tracking: true,
            max_reasoning_depth: 5,
            reasoning_confidence_threshold: 0.6,
            enable_reasoning_chains: false,
            timeout_ms: 5000,
        }
    }
}

/// Types of profiling events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    ConversationStart,
    ConversationEnd,
    TurnProcessed,
    MemoryUpdated,
    HealthCheck,
    SafetyAlert,
    PerformanceAlert,
    ErrorOccurred,
}

/// Data associated with profiling events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventData {
    ConversationStarted {
        config: String,
    },
    ConversationEnded {
        duration_ms: u64,
        total_turns: usize,
    },
    TurnProcessed {
        processing_time_ms: f64,
        token_count: usize,
    },
    MemoryUpdated {
        memory_count: usize,
        importance_threshold: f32,
    },
    HealthChecked {
        health_score: f32,
        issues: Vec<String>,
    },
    SafetyAlerted {
        violation_type: String,
        content_snippet: String,
    },
    PerformanceAlerted {
        metric: String,
        value: f64,
        threshold: f64,
    },
    ErrorOccurred {
        error_type: String,
        message: String,
    },
}

/// Response streaming state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamingState {
    NotStarted,
    Streaming,
    Paused,
    Completed,
    Error(String),
}

// ================================================================================================
// EXTENDED TYPES (From existing types.rs)
// ================================================================================================

/// Trending metrics for conversation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendingMetrics {
    pub engagement_trend: TrendDirection,
    pub quality_trend: TrendDirection,
    pub safety_trend: TrendDirection,
    pub coherence_trend: TrendDirection,
    pub response_time_trend: TrendDirection,
}

/// Metric trend information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricTrend {
    pub direction: TrendDirection,
    pub rate_of_change: f32,
    pub confidence: f32,
    pub time_window_hours: u32,
}

/// System health assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub status: HealthStatus,
    pub score: f32,
    pub trending_metrics: TrendingMetrics,
    pub last_assessment: chrono::DateTime<chrono::Utc>,
    pub assessment_confidence: f32,
}

/// Events that occur during profiling sessions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingEvent {
    pub event_type: EventType,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub data: EventData,
    pub conversation_id: String,
}

/// Session metadata for profiling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    pub session_id: String,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
    pub configuration_snapshot: String,
    pub participant_count: usize,
    pub environment: String,
}

/// Streaming response chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingResponse {
    pub chunk: String,
    pub is_final: bool,
    pub chunk_index: usize,
    pub total_chunks: Option<usize>,
    pub metadata: Option<ConversationMetadata>,
}
