//! Memory utilities for conversation management.
//!
//! This module provides comprehensive memory management utilities including
//! importance calculation, memory type classification, relevance scoring,
//! decay algorithms, and memory organization and retrieval.

use chrono::Utc;
use std::collections::HashSet;

use super::super::types::{
    ConversationMemory, ConversationState, ConversationTurn, EngagementLevel, MemoryType,
};

/// Memory utilities for conversation management
pub struct MemoryUtils;

impl MemoryUtils {
    /// Calculate memory importance based on content
    pub fn calculate_importance(turn: &ConversationTurn) -> f32 {
        let mut importance = 0.5; // Base importance

        // Increase importance for questions
        if turn.content.contains('?') {
            importance += 0.2;
        }

        // Increase for personal information
        if ["i am", "my name", "i like", "i prefer", "i work", "i live"]
            .iter()
            .any(|&pattern| turn.content.to_lowercase().contains(pattern))
        {
            importance += 0.3;
        }

        // Increase for goals and preferences
        if ["want", "need", "goal", "prefer", "like", "dislike", "plan"]
            .iter()
            .any(|&pattern| turn.content.to_lowercase().contains(pattern))
        {
            importance += 0.2;
        }

        // Increase for emotional content
        if ["feel", "emotion", "happy", "sad", "excited", "worried"]
            .iter()
            .any(|&pattern| turn.content.to_lowercase().contains(pattern))
        {
            importance += 0.15;
        }

        // Adjust based on metadata if available
        if let Some(metadata) = &turn.metadata {
            importance += metadata.confidence * 0.1;
            match metadata.engagement_level {
                EngagementLevel::High | EngagementLevel::VeryHigh => importance += 0.2,
                EngagementLevel::Medium => importance += 0.1,
                _ => {},
            }
        }

        // Adjust based on content length (very short or very long content might be less important)
        let length = turn.content.len();
        if length < 10 {
            importance -= 0.1;
        } else if length > 500 {
            importance += 0.1;
        }

        importance.clamp(0.0, 1.0)
    }

    /// Classify memory type based on content
    pub fn classify_memory_type(turn: &ConversationTurn) -> MemoryType {
        let content = turn.content.to_lowercase();

        if ["prefer", "like", "dislike", "favorite", "hate", "love"]
            .iter()
            .any(|&pattern| content.contains(pattern))
        {
            MemoryType::Preference
        } else if ["goal", "want", "plan", "will", "hope", "aim", "intend"]
            .iter()
            .any(|&pattern| content.contains(pattern))
        {
            MemoryType::Goal
        } else if [
            "friend",
            "family",
            "colleague",
            "know",
            "relationship",
            "partner",
        ]
        .iter()
        .any(|&pattern| content.contains(pattern))
        {
            MemoryType::Relationship
        } else if [
            "happened",
            "did",
            "went",
            "experience",
            "remember",
            "recall",
        ]
        .iter()
        .any(|&pattern| content.contains(pattern))
        {
            MemoryType::Experience
        } else {
            MemoryType::Fact
        }
    }

    /// Extract tags from conversation turn
    pub fn extract_tags(turn: &ConversationTurn) -> Vec<String> {
        let mut tags = Vec::new();

        // Add metadata-based tags
        if let Some(metadata) = &turn.metadata {
            tags.extend(metadata.topics.clone());
            if let Some(sentiment) = &metadata.sentiment {
                tags.push(format!("sentiment:{}", sentiment));
            }
            if let Some(intent) = &metadata.intent {
                tags.push(format!("intent:{}", intent));
            }
        }

        // Extract keyword-based tags
        let keyword_tags = [
            "work",
            "family",
            "hobby",
            "food",
            "travel",
            "technology",
            "health",
            "education",
            "entertainment",
            "sports",
            "music",
            "books",
            "movies",
        ];

        let content_lower = turn.content.to_lowercase();
        for keyword in keyword_tags {
            if content_lower.contains(keyword) {
                tags.push(keyword.to_string());
            }
        }

        // Add role-based tag
        tags.push(format!("role:{:?}", turn.role).to_lowercase());

        tags
    }

    /// Calculate memory relevance to a query
    pub fn calculate_memory_relevance(memory: &ConversationMemory, query: &str) -> f32 {
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();
        let memory_lower = memory.content.to_lowercase();
        let memory_words: Vec<&str> = memory_lower.split_whitespace().collect();

        // Calculate word overlap
        let overlap = query_words.iter().filter(|word| memory_words.contains(word)).count();

        let word_relevance = overlap as f32 / query_words.len().max(1) as f32;

        // Check tag relevance
        let tag_relevance = memory.tags.iter().any(|tag| query_lower.contains(&tag.to_lowercase()))
            as i32 as f32
            * 0.3;

        // Calculate time decay factor
        let time_decay = {
            let hours_since_access = (Utc::now() - memory.last_accessed).num_hours() as f32;
            let days_since_access = hours_since_access / 24.0;
            // Exponential decay over 30 days
            (-days_since_access / 30.0).exp()
        };

        // Combine factors
        let base_relevance = (word_relevance * 0.6 + tag_relevance * 0.4).min(1.0);
        let importance_factor = memory.importance;
        let access_factor = (memory.access_count as f32).ln().max(1.0) / 10.0;

        (base_relevance * importance_factor + access_factor * 0.1 + time_decay * 0.2).min(1.0)
    }

    /// Decay memory importance over time
    pub fn apply_decay(memory: &mut ConversationMemory, decay_rate: f32) {
        let hours_since_access = (Utc::now() - memory.last_accessed).num_hours() as f32;
        let time_factor = hours_since_access / (24.0 * 7.0); // Weekly decay
        memory.importance *= decay_rate.powf(time_factor);
        memory.importance = memory.importance.max(0.0);
    }

    /// Get relevant memories for current context (enhanced algorithm)
    pub fn get_relevant_memories_for_context<'a>(
        state: &'a ConversationState,
        query: &str,
        limit: usize,
    ) -> Vec<&'a ConversationMemory> {
        let mut scored_memories: Vec<_> = state
            .memories
            .iter()
            .map(|memory| {
                let relevance = Self::calculate_memory_relevance_enhanced(memory, query);
                (memory, relevance)
            })
            .collect();

        scored_memories.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored_memories.into_iter().take(limit).map(|(memory, _)| memory).collect()
    }

    /// Enhanced memory relevance calculation with multiple factors
    pub fn calculate_memory_relevance_enhanced(memory: &ConversationMemory, query: &str) -> f32 {
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();
        let memory_lower = memory.content.to_lowercase();
        let memory_words: Vec<&str> = memory_lower.split_whitespace().collect();

        // Word overlap calculation
        let overlap = query_words.iter().filter(|word| memory_words.contains(word)).count();

        let word_relevance = overlap as f32 / query_words.len().max(1) as f32;

        // Enhanced tag relevance
        let tag_relevance = memory
            .tags
            .iter()
            .map(|tag| {
                if query_lower.contains(&tag.to_lowercase()) {
                    0.5
                } else if tag.contains(':') {
                    // Handle structured tags like "sentiment:positive"
                    let tag_parts: Vec<&str> = tag.split(':').collect();
                    if tag_parts.len() == 2 && query_lower.contains(tag_parts[1]) {
                        0.3
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            })
            .sum::<f32>();

        // Semantic similarity (simple implementation)
        let semantic_similarity = Self::calculate_semantic_similarity(&memory_lower, &query_lower);

        // Time decay factor with improved calculation
        let time_decay = {
            let hours_since_access = (Utc::now() - memory.last_accessed).num_hours() as f32;
            let days_since_access = hours_since_access / 24.0;
            // Sigmoid decay function for more gradual decline
            1.0 / (1.0 + (days_since_access / 14.0).exp())
        };

        // Access frequency boost
        let access_boost = (memory.access_count as f32).ln().max(1.0) / 20.0;

        // Memory type relevance
        let type_relevance = match memory.memory_type {
            MemoryType::Preference | MemoryType::Goal => 0.2,
            MemoryType::Experience => 0.15,
            MemoryType::Relationship => 0.1,
            MemoryType::Fact => 0.05,
        };

        // Combine all factors with weights
        let base_relevance =
            word_relevance * 0.4 + tag_relevance * 0.2 + semantic_similarity * 0.2 + type_relevance;

        let final_relevance =
            base_relevance * memory.importance + access_boost * 0.1 + time_decay * 0.3;

        final_relevance.min(1.0)
    }

    /// Simple semantic similarity calculation
    fn calculate_semantic_similarity(text1: &str, text2: &str) -> f32 {
        let words1: HashSet<&str> = text1.split_whitespace().collect();
        let words2: HashSet<&str> = text2.split_whitespace().collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Batch memory decay for performance
    pub fn batch_decay_memories(memories: &mut [ConversationMemory], decay_rate: f32) {
        let now = Utc::now();
        for memory in memories.iter_mut() {
            let hours_since_access = (now - memory.last_accessed).num_hours() as f32;
            let time_factor = hours_since_access / (24.0 * 7.0);
            memory.importance *= decay_rate.powf(time_factor);
            memory.importance = memory.importance.max(0.0);
        }
    }

    /// Prune low-importance memories
    pub fn prune_memories(memories: &mut Vec<ConversationMemory>, min_importance: f32) {
        memories.retain(|memory| memory.importance >= min_importance);
    }

    /// Sort memories by combined importance and relevance
    pub fn sort_memories_by_priority(memories: &mut [ConversationMemory]) {
        memories.sort_by(|a, b| {
            let priority_a = a.importance + (a.access_count as f32 * 0.01);
            let priority_b = b.importance + (b.access_count as f32 * 0.01);
            priority_b.partial_cmp(&priority_a).unwrap()
        });
    }
}
