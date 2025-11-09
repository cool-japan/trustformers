//! Conversational utility modules.
//!
//! This module provides a collection of specialized utility modules for
//! conversational AI systems, each focused on a specific aspect of
//! conversation processing and management.

pub mod context_builder;
pub mod formatting;
pub mod health_tracker;
pub mod memory;
pub mod safety;
pub mod serialization;
pub mod string;
pub mod text_analyzer;
pub mod text_processing;
pub mod time;

// Re-export main utilities for backward compatibility and convenience
pub use context_builder::ContextBuilder;
pub use formatting::ConversationFormatter;
pub use health_tracker::ConversationHealthTracker;
pub use memory::MemoryUtils;
pub use safety::{EnhancedSafetyFilter, FilterResult, RiskLevel, SafetyAnalysis};
pub use serialization::ConversationSerializer;
pub use string::StringUtils;
pub use text_analyzer::TextAnalyzer;
pub use text_processing::TextProcessor;
pub use time::TimeUtils;
