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
        self.memories.sort_by(|a, b| {
            b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal)
        });
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

        scored_memories.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_turn(role: ConversationRole, content: &str, tokens: usize) -> ConversationTurn {
        ConversationTurn {
            role,
            content: content.to_string(),
            timestamp: chrono::Utc::now(),
            metadata: None,
            token_count: tokens,
        }
    }

    #[test]
    fn test_new_state_has_empty_turns() {
        let state = ConversationState::new("id-1".to_string());
        assert!(
            state.turns.is_empty(),
            "new ConversationState must have no turns"
        );
    }

    #[test]
    fn test_new_state_id_preserved() {
        let id = "test-conversation-99";
        let state = ConversationState::new(id.to_string());
        assert_eq!(
            state.conversation_id, id,
            "conversation_id must match the supplied id"
        );
    }

    #[test]
    fn test_add_user_turn_increments_counter() {
        let mut state = ConversationState::new("s1".to_string());
        state.add_turn(make_turn(ConversationRole::User, "hi", 2));
        assert_eq!(state.stats.user_turns, 1, "user_turns must increment by 1");
    }

    #[test]
    fn test_add_assistant_turn_increments_counter() {
        let mut state = ConversationState::new("s2".to_string());
        state.add_turn(make_turn(ConversationRole::Assistant, "hello", 4));
        assert_eq!(
            state.stats.assistant_turns, 1,
            "assistant_turns must increment by 1"
        );
    }

    #[test]
    fn test_token_accumulation_across_turns() {
        let mut state = ConversationState::new("s3".to_string());
        state.add_turn(make_turn(ConversationRole::User, "msg1", 5));
        state.add_turn(make_turn(ConversationRole::User, "msg2", 8));
        assert_eq!(
            state.total_tokens, 13,
            "total_tokens must sum all turn token counts"
        );
    }

    #[test]
    fn test_get_recent_context_returns_within_limit() {
        let mut state = ConversationState::new("s4".to_string());
        for i in 0..10u32 {
            state.add_turn(make_turn(ConversationRole::User, &format!("m{i}"), 50));
        }
        // With 500 total tokens, requesting max 150 should give at most 3 turns
        let ctx = state.get_recent_context(150);
        assert!(
            ctx.len() <= 3,
            "context must not exceed token budget (50 per turn, 150 max)"
        );
    }

    #[test]
    fn test_get_recent_context_ordering_preserved() {
        let mut state = ConversationState::new("s5".to_string());
        state.add_turn(make_turn(ConversationRole::User, "first", 10));
        state.add_turn(make_turn(ConversationRole::User, "second", 10));
        let ctx = state.get_recent_context(1000);
        assert_eq!(
            ctx[0].content, "first",
            "context must preserve chronological order"
        );
        assert_eq!(ctx[1].content, "second", "second message must be last");
    }

    #[test]
    fn test_trim_history_keeps_last_n_turns() {
        let mut state = ConversationState::new("s6".to_string());
        for i in 0..8u32 {
            state.add_turn(make_turn(ConversationRole::User, &format!("t{i}"), 1));
        }
        state.trim_history(4, usize::MAX);
        assert_eq!(
            state.turns.len(),
            4,
            "trim_history(4) must keep only the last 4 turns"
        );
    }

    #[test]
    fn test_trim_history_by_tokens() {
        let mut state = ConversationState::new("s7".to_string());
        for _ in 0..5u32 {
            state.add_turn(make_turn(ConversationRole::User, "x", 20));
        }
        // total = 100; trim to 60 means at most 3 turns remain
        state.trim_history(usize::MAX, 60);
        assert!(
            state.total_tokens <= 60,
            "total_tokens must not exceed 60 after token trim"
        );
    }

    #[test]
    fn test_set_and_get_variable_roundtrip() {
        let mut state = ConversationState::new("s8".to_string());
        state.set_variable("lang".to_string(), "en".to_string());
        assert_eq!(
            state.get_variable("lang").map(String::as_str),
            Some("en"),
            "get_variable must return the value set by set_variable"
        );
    }

    #[test]
    fn test_set_variable_overwrite() {
        let mut state = ConversationState::new("s9".to_string());
        state.set_variable("key".to_string(), "v1".to_string());
        state.set_variable("key".to_string(), "v2".to_string());
        assert_eq!(
            state.get_variable("key").map(String::as_str),
            Some("v2"),
            "second set_variable must overwrite the previous value"
        );
    }

    #[test]
    fn test_update_health_averages_correctly() {
        let mut state = ConversationState::new("s10".to_string());
        state.update_health(0.6, 0.9, 1.0);
        let expected = (0.6 + 0.9 + 1.0) / 3.0;
        assert!(
            (state.health.overall_score - expected).abs() < 1e-5,
            "overall_score must be the mean of the three supplied scores"
        );
    }

    #[test]
    fn test_needs_repair_threshold_coherence() {
        let mut state = ConversationState::new("s11".to_string());
        state.update_health(0.4, 0.9, 0.9);
        assert!(
            state.needs_repair(),
            "low coherence (< 0.5) must trigger needs_repair"
        );
    }

    #[test]
    fn test_no_repair_needed_when_scores_high() {
        let mut state = ConversationState::new("s12".to_string());
        state.update_health(0.95, 0.95, 0.95);
        assert!(
            !state.needs_repair(),
            "healthy state must not require repair"
        );
    }

    #[test]
    fn test_start_reasoning_initialises_context() {
        let mut state = ConversationState::new("s13".to_string());
        assert!(
            state.reasoning_context.is_none(),
            "reasoning context should be None initially"
        );
        state.start_reasoning(Some("goal".to_string()));
        assert!(
            state.reasoning_context.is_some(),
            "reasoning context must be Some after start"
        );
    }

    #[test]
    fn test_add_reasoning_step_appended() {
        let mut state = ConversationState::new("s14".to_string());
        state.start_reasoning(None);
        let step = ReasoningStep {
            step_type: ReasoningType::Causal,
            description: "A causes B".to_string(),
            inputs: vec!["A".to_string()],
            output: "B".to_string(),
            confidence: 0.75,
        };
        state.add_reasoning_step(step);
        let ctx = state.reasoning_context.as_ref().expect("context must exist");
        assert_eq!(
            ctx.reasoning_chain.len(),
            1,
            "one reasoning step must be in the chain"
        );
    }

    #[test]
    fn test_memory_sorting_by_importance() {
        let mut state = ConversationState::new("s15".to_string());
        for i in 0..5u32 {
            state.add_memory(ConversationMemory {
                id: format!("m{i}"),
                content: "fact".to_string(),
                importance: i as f32 * 0.2,
                last_accessed: chrono::Utc::now(),
                access_count: 1,
                memory_type: MemoryType::Fact,
                tags: vec![],
            });
        }
        // After sorting by importance desc, first element should have highest importance
        assert!(
            state.memories[0].importance >= state.memories[1].importance,
            "memories should be sorted by descending importance"
        );
    }

    #[test]
    fn test_get_summary_stats_reflects_turns() {
        let mut state = ConversationState::new("s16".to_string());
        state.add_turn(make_turn(ConversationRole::User, "hello", 2));
        state.add_turn(make_turn(ConversationRole::Assistant, "hi there", 4));
        let stats = state.get_summary_stats();
        assert_eq!(stats.total_turns, 2, "summary stats must count both turns");
        assert_eq!(
            stats.total_tokens, 6,
            "summary stats total_tokens must sum all tokens"
        );
    }
}
