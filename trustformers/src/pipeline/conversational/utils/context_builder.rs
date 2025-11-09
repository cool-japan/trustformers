//! Context building utilities for enhanced conversation context.
//!
//! This module provides functionality to build rich conversation contexts
//! incorporating memories, persona information, conversation history, and
//! mode-specific instructions.

use super::super::types::{
    ConversationMode, ConversationState, ConversationTurn, ConversationalConfig,
};
use super::{formatting::ConversationFormatter, memory::MemoryUtils};
use crate::core::error::Result;

/// Context building utilities for enhanced conversation context
pub struct ContextBuilder;

impl ContextBuilder {
    /// Build enhanced conversation context with memories and persona
    pub fn build_enhanced_context(
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
        if let Some(persona) = &config.persona {
            context.push_str(&format!(
                "You are {}. {}\n\nBackground: {}\n\nSpeaking style: {}\n\n",
                persona.name, persona.personality, persona.background, persona.speaking_style
            ));
        }

        // Add conversation summary if available
        if let Some(summary) = &state.context_summary {
            context.push_str(&format!("Previous conversation summary: {}\n\n", summary));
        }

        // Add relevant memories
        let relevant_memories =
            MemoryUtils::get_relevant_memories_for_context(state, current_input, 3);
        if !relevant_memories.is_empty() {
            context.push_str("Relevant context from previous conversations:\n");
            for memory in relevant_memories {
                context.push_str(&format!("- {}\n", memory.content));
            }
            context.push('\n');
        }

        // Add recent conversation turns
        let recent_turns =
            Self::get_recent_context_within_limit(state, config.max_context_tokens - context.len());
        for turn in recent_turns {
            let role_str = ConversationFormatter::format_role(&turn.role);
            context.push_str(&format!("{}: {}\n", role_str, turn.content));
        }

        // Add conversation mode specific instructions
        match config.conversation_mode {
            ConversationMode::Chat => {
                context.push_str("\nContinue the conversation naturally and helpfully.\n");
            },
            ConversationMode::Assistant => {
                context.push_str("\nProvide helpful assistance with the user's request.\n");
            },
            ConversationMode::InstructionFollowing => {
                context.push_str("\nFollow the user's instructions carefully and accurately.\n");
            },
            ConversationMode::QuestionAnswering => {
                context.push_str("\nAnswer the user's question accurately and concisely.\n");
            },
            ConversationMode::RolePlay => {
                context
                    .push_str("\nStay in character and respond appropriately to the scenario.\n");
            },
            ConversationMode::Educational => {
                context.push_str(
                    "\nProvide educational and informative responses to help the user learn.\n",
                );
            },
        }

        context.push_str("\nAssistant:");
        Ok(context)
    }

    /// Get recent conversation turns within token limit
    pub fn get_recent_context_within_limit(
        state: &ConversationState,
        max_tokens: usize,
    ) -> Vec<&ConversationTurn> {
        let mut context = Vec::new();
        let mut token_count = 0;

        for turn in state.turns.iter().rev() {
            if token_count + turn.token_count > max_tokens {
                break;
            }
            token_count += turn.token_count;
            context.push(turn);
        }

        context.reverse();
        context
    }

    /// Build context for summarization
    pub fn build_summarization_context(turns: &[ConversationTurn]) -> String {
        turns
            .iter()
            .map(|t| {
                format!(
                    "{}: {}",
                    ConversationFormatter::format_role(&t.role),
                    t.content
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}
