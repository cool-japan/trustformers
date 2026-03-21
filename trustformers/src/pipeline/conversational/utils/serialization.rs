//! Conversation serialization utilities for export/import.
//!
//! This module provides utilities for serializing and deserializing conversation
//! data to/from various formats including JSON, compact formats, and backups.

use chrono::Utc;
use serde_json;
use std::collections::HashMap;

use super::super::types::{ConversationState, ConversationStats, ConversationTurn};
use crate::error::{Result, TrustformersError};

/// Conversation serialization utilities for export/import
pub struct ConversationSerializer;

impl ConversationSerializer {
    /// Export conversation state to JSON
    pub fn export_conversation(state: &ConversationState) -> Result<String> {
        serde_json::to_string_pretty(state)
            .map_err(|e| TrustformersError::runtime_error(e.to_string()))
    }

    /// Import conversation state from JSON
    pub fn import_conversation(json: &str) -> Result<ConversationState> {
        serde_json::from_str(json).map_err(|e| TrustformersError::runtime_error(e.to_string()))
    }

    /// Export conversation to compact format (for storage efficiency)
    pub fn export_conversation_compact(state: &ConversationState) -> Result<String> {
        serde_json::to_string(state).map_err(|e| TrustformersError::runtime_error(e.to_string()))
    }

    /// Export only conversation history (without state)
    pub fn export_conversation_history(turns: &[ConversationTurn]) -> Result<String> {
        serde_json::to_string_pretty(turns)
            .map_err(|e| TrustformersError::runtime_error(e.to_string()))
    }

    /// Export conversation statistics
    pub fn export_conversation_stats(stats: &ConversationStats) -> Result<String> {
        serde_json::to_string_pretty(stats)
            .map_err(|e| TrustformersError::runtime_error(e.to_string()))
    }

    /// Create conversation backup with metadata
    pub fn create_conversation_backup(
        state: &ConversationState,
        backup_metadata: Option<HashMap<String, String>>,
    ) -> Result<String> {
        let mut backup_data = HashMap::new();
        backup_data.insert(
            "conversation".to_string(),
            serde_json::to_value(state)
                .map_err(|e| TrustformersError::runtime_error(e.to_string()))?,
        );
        backup_data.insert(
            "backup_timestamp".to_string(),
            serde_json::to_value(Utc::now())
                .map_err(|e| TrustformersError::runtime_error(e.to_string()))?,
        );

        if let Some(metadata) = backup_metadata {
            backup_data.insert(
                "metadata".to_string(),
                serde_json::to_value(metadata)
                    .map_err(|e| TrustformersError::runtime_error(e.to_string()))?,
            );
        }

        serde_json::to_string_pretty(&backup_data)
            .map_err(|e| TrustformersError::runtime_error(e.to_string()))
    }

    /// Restore conversation from backup
    pub fn restore_from_backup(backup_json: &str) -> Result<ConversationState> {
        let backup_data: HashMap<String, serde_json::Value> = serde_json::from_str(backup_json)
            .map_err(|e| TrustformersError::runtime_error(e.to_string()))?;

        let conversation_data = backup_data.get("conversation").ok_or_else(|| {
            TrustformersError::invalid_input(
                "No conversation data found in backup",
                Some("backup_data"),
                Some("conversation object"),
                Some("empty or missing conversation data"),
            )
        })?;

        serde_json::from_value(conversation_data.clone())
            .map_err(|e| TrustformersError::runtime_error(e.to_string()))
    }
}
