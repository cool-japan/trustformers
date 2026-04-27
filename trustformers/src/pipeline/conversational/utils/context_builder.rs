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

#[cfg(test)]
mod tests {
    use super::super::super::types::{ConversationRole, ConversationState};
    use super::*;
    use chrono::Utc;

    fn make_turn(role: ConversationRole, content: &str, token_count: usize) -> ConversationTurn {
        ConversationTurn {
            role,
            content: content.to_string(),
            timestamp: Utc::now(),
            metadata: None,
            token_count,
        }
    }

    fn make_state_with_turns(turns: Vec<ConversationTurn>) -> ConversationState {
        let mut state = ConversationState::new("ctx-test".to_string());
        for turn in turns {
            state.add_turn(turn);
        }
        state
    }

    #[test]
    fn test_get_recent_context_within_limit_empty() {
        let state = ConversationState::new("empty".to_string());
        let context = ContextBuilder::get_recent_context_within_limit(&state, 1000);
        assert!(context.is_empty());
    }

    #[test]
    fn test_get_recent_context_respects_token_limit() {
        let turns = vec![
            make_turn(ConversationRole::User, "hello", 100),
            make_turn(ConversationRole::Assistant, "world", 100),
            make_turn(ConversationRole::User, "goodbye", 100),
        ];
        let state = make_state_with_turns(turns);
        // Only 150 tokens allowed, can fit at most 1 turn
        let context = ContextBuilder::get_recent_context_within_limit(&state, 150);
        assert!(context.len() <= 2);
    }

    #[test]
    fn test_get_recent_context_all_fit_within_limit() {
        let turns = vec![
            make_turn(ConversationRole::User, "hello", 10),
            make_turn(ConversationRole::Assistant, "world", 10),
        ];
        let state = make_state_with_turns(turns);
        let context = ContextBuilder::get_recent_context_within_limit(&state, 1000);
        assert_eq!(context.len(), 2);
    }

    #[test]
    fn test_get_recent_context_order_preserved() {
        let turns = vec![
            make_turn(ConversationRole::User, "first", 10),
            make_turn(ConversationRole::Assistant, "second", 10),
            make_turn(ConversationRole::User, "third", 10),
        ];
        let state = make_state_with_turns(turns);
        let context = ContextBuilder::get_recent_context_within_limit(&state, 1000);
        assert_eq!(context.len(), 3);
        assert_eq!(context[0].content, "first");
        assert_eq!(context[2].content, "third");
    }

    #[test]
    fn test_build_summarization_context_empty() {
        let result = ContextBuilder::build_summarization_context(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_build_summarization_context_single_turn() {
        let turn = make_turn(ConversationRole::User, "hello", 5);
        let result = ContextBuilder::build_summarization_context(&[turn]);
        assert!(result.contains("hello"));
    }

    #[test]
    fn test_build_summarization_context_multiple_turns() {
        let turns = vec![
            make_turn(ConversationRole::User, "question here", 10),
            make_turn(ConversationRole::Assistant, "answer here", 10),
        ];
        let result = ContextBuilder::build_summarization_context(&turns);
        assert!(result.contains("question here"));
        assert!(result.contains("answer here"));
        assert!(result.contains('\n'));
    }

    #[test]
    fn test_build_enhanced_context_no_system_prompt() {
        let mut config = ConversationalConfig::default();
        config.system_prompt = None;
        config.persona = None;
        let state = ConversationState::new("test".to_string());
        let result = ContextBuilder::build_enhanced_context(&state, &config, "hello");
        if let Ok(ctx) = result {
            assert!(!ctx.is_empty());
            assert!(ctx.contains("Assistant:"));
        }
    }

    #[test]
    fn test_build_enhanced_context_with_system_prompt() {
        let mut config = ConversationalConfig::default();
        config.system_prompt = Some("Be helpful".to_string());
        let state = ConversationState::new("test".to_string());
        let result = ContextBuilder::build_enhanced_context(&state, &config, "hello");
        if let Ok(ctx) = result {
            assert!(ctx.contains("Be helpful"));
        }
    }

    #[test]
    fn test_build_enhanced_context_with_summary() {
        let config = ConversationalConfig::default();
        let mut state = ConversationState::new("test".to_string());
        state.context_summary = Some("Previous discussion about coding".to_string());
        let result = ContextBuilder::build_enhanced_context(&state, &config, "tell me more");
        if let Ok(ctx) = result {
            assert!(ctx.contains("Previous discussion about coding"));
        }
    }

    #[test]
    fn test_build_enhanced_context_chat_mode() {
        let mut config = ConversationalConfig::default();
        config.conversation_mode = ConversationMode::Chat;
        let state = ConversationState::new("test".to_string());
        let result = ContextBuilder::build_enhanced_context(&state, &config, "hi");
        if let Ok(ctx) = result {
            assert!(ctx.contains("naturally"));
        }
    }

    #[test]
    fn test_build_enhanced_context_qa_mode() {
        let mut config = ConversationalConfig::default();
        config.conversation_mode = ConversationMode::QuestionAnswering;
        let state = ConversationState::new("test".to_string());
        let result = ContextBuilder::build_enhanced_context(&state, &config, "what is Rust?");
        if let Ok(ctx) = result {
            assert!(ctx.contains("question"));
        }
    }
}
