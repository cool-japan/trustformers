//! Conversational analysis modules.
//!
//! This module provides comprehensive analysis capabilities for conversational AI,
//! organized into focused submodules for better maintainability and performance.

pub mod analyzer;
pub mod types;

// Re-export main components for backward compatibility and convenience
pub use analyzer::ConversationAnalyzer;
pub use types::{
    AnalysisPerformance, ContextualMetrics, DetailedHealthMetrics, EnhancedAnalysisConfig,
    HealthAssessment, HealthIssue, HealthIssueType, LinguisticAnalysis, QualityStrictness,
    SafetySensitivity, TurnAnalysisResult,
};
