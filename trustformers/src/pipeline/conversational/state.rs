//! Conversation state management and turn tracking.

use super::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
                last_breakdown: None,
                repair_attempts: 0,
                recommendations: Vec::new(),
                issues: Vec::new(),
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

    /// Get conversation summary statistics
    pub fn get_summary_stats(&self) -> ConversationSummaryStats {
        let total_turns = self.turns.len();
        let avg_turn_length = if total_turns > 0 {
            self.turns.iter().map(|t| t.content.len()).sum::<usize>() as f32 / total_turns as f32
        } else {
            0.0
        };

        let duration = if let Some(last_turn) = self.turns.last() {
            (last_turn.timestamp - self.stats.start_time).num_minutes() as f32
        } else {
            0.0
        };

        ConversationSummaryStats {
            total_turns,
            total_tokens: self.total_tokens,
            duration_minutes: duration,
            avg_turn_length,
            memory_count: self.memories.len(),
            health_score: self.health.overall_score,
            topics_count: self.stats.topics_discussed.len(),
        }
    }

    /// Archive old memories based on access patterns
    pub fn archive_old_memories(&mut self, archive_threshold_days: i64) {
        let cutoff_date = chrono::Utc::now() - chrono::Duration::days(archive_threshold_days);

        self.memories
            .retain(|memory| memory.last_accessed > cutoff_date || memory.importance > 0.8);
    }

    /// Get conversation flow analysis
    pub fn analyze_conversation_flow(&self) -> ConversationFlowAnalysis {
        let mut user_response_times = Vec::new();
        let mut assistant_response_times = Vec::new();
        let mut topic_transitions = 0;
        let last_topics: Vec<String> = Vec::new();

        for i in 1..self.turns.len() {
            let current = &self.turns[i];
            let previous = &self.turns[i - 1];

            let response_time = (current.timestamp - previous.timestamp).num_seconds() as f32;

            match (&previous.role, &current.role) {
                (ConversationRole::User, ConversationRole::Assistant) => {
                    assistant_response_times.push(response_time);
                },
                (ConversationRole::Assistant, ConversationRole::User) => {
                    user_response_times.push(response_time);
                },
                _ => {},
            }

            // Analyze topic transitions
            if let (Some(prev_meta), Some(curr_meta)) = (&previous.metadata, &current.metadata) {
                if !prev_meta.topics.iter().any(|t| curr_meta.topics.contains(t))
                    && !prev_meta.topics.is_empty()
                    && !curr_meta.topics.is_empty()
                {
                    topic_transitions += 1;
                }
            }
        }

        let avg_user_response_time = if user_response_times.is_empty() {
            0.0
        } else {
            user_response_times.iter().sum::<f32>() / user_response_times.len() as f32
        };

        let avg_assistant_response_time = if assistant_response_times.is_empty() {
            0.0
        } else {
            assistant_response_times.iter().sum::<f32>() / assistant_response_times.len() as f32
        };

        ConversationFlowAnalysis {
            avg_user_response_time_seconds: avg_user_response_time,
            avg_assistant_response_time_seconds: avg_assistant_response_time,
            topic_transitions,
            conversation_pace: if avg_user_response_time > 0.0 && avg_assistant_response_time > 0.0
            {
                ConversationPace::from_response_times(
                    avg_user_response_time,
                    avg_assistant_response_time,
                )
            } else {
                ConversationPace::Unknown
            },
        }
    }
}

/// Summary statistics for a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSummaryStats {
    pub total_turns: usize,
    pub total_tokens: usize,
    pub duration_minutes: f32,
    pub avg_turn_length: f32,
    pub memory_count: usize,
    pub health_score: f32,
    pub topics_count: usize,
}

/// Analysis of conversation flow patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationFlowAnalysis {
    pub avg_user_response_time_seconds: f32,
    pub avg_assistant_response_time_seconds: f32,
    pub topic_transitions: usize,
    pub conversation_pace: ConversationPace,
}

/// Classification of conversation pace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConversationPace {
    Rapid,     // Fast exchanges
    Normal,    // Moderate pace
    Slow,      // Longer pauses
    Irregular, // Inconsistent timing
    Unknown,   // Insufficient data
}

impl ConversationPace {
    fn from_response_times(user_avg: f32, assistant_avg: f32) -> Self {
        let combined_avg = (user_avg + assistant_avg) / 2.0;

        match combined_avg {
            t if t < 5.0 => ConversationPace::Rapid,
            t if t < 30.0 => ConversationPace::Normal,
            t if t < 120.0 => ConversationPace::Slow,
            _ => ConversationPace::Irregular,
        }
    }
}
