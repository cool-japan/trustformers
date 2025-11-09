//! Conversation formatting utilities.
//!
//! This module provides utilities for formatting conversation turns, metadata,
//! durations, and other conversational data for display and processing.

use super::super::types::{ConversationMetadata, ConversationRole, ConversationTurn};

/// Conversation formatting utilities
pub struct ConversationFormatter;

impl ConversationFormatter {
    /// Format role name for display
    pub fn format_role(role: &ConversationRole) -> &'static str {
        match role {
            ConversationRole::User => "User",
            ConversationRole::Assistant => "Assistant",
            ConversationRole::System => "System",
        }
    }

    /// Format conversation turn for context building
    pub fn format_turn_for_context(turn: &ConversationTurn) -> String {
        format!("{}: {}", Self::format_role(&turn.role), turn.content)
    }

    /// Build conversation history string
    pub fn build_history_string(turns: &[ConversationTurn]) -> String {
        turns.iter().map(Self::format_turn_for_context).collect::<Vec<_>>().join("\n")
    }

    /// Format metadata for display
    pub fn format_metadata(metadata: &ConversationMetadata) -> String {
        let mut parts = Vec::new();

        if let Some(sentiment) = &metadata.sentiment {
            parts.push(format!("Sentiment: {}", sentiment));
        }

        if let Some(intent) = &metadata.intent {
            parts.push(format!("Intent: {}", intent));
        }

        if !metadata.topics.is_empty() {
            parts.push(format!("Topics: {}", metadata.topics.join(", ")));
        }

        parts.push(format!("Confidence: {:.2}", metadata.confidence));
        parts.push(format!("Quality: {:.2}", metadata.quality_score));
        parts.push(format!("Engagement: {:?}", metadata.engagement_level));

        parts.join(" | ")
    }

    /// Truncate text to specified length with ellipsis
    pub fn truncate_text(text: &str, max_length: usize) -> String {
        if text.len() <= max_length {
            text.to_string()
        } else {
            format!("{}...", &text[..max_length.saturating_sub(3)])
        }
    }

    /// Format duration for display
    pub fn format_duration(duration: std::time::Duration) -> String {
        let total_seconds = duration.as_secs();
        let hours = total_seconds / 3600;
        let minutes = (total_seconds % 3600) / 60;
        let seconds = total_seconds % 60;

        if hours > 0 {
            format!("{}h {}m {}s", hours, minutes, seconds)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, seconds)
        } else {
            format!("{}s", seconds)
        }
    }
}
