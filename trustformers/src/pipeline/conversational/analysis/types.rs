//! Analysis types and configurations.
//!
//! This module contains all the type definitions, configurations, and enums
//! used throughout the conversational analysis system.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use super::super::types::EngagementLevel;

/// Result of analyzing a conversation turn
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnAnalysisResult {
    /// Overall quality score (0.0 to 1.0)
    pub quality_score: f32,
    /// Engagement level assessment
    pub engagement_level: EngagementLevel,
    /// Detected sentiment
    pub sentiment: Option<String>,
    /// Intent classification
    pub intent: Option<String>,
    /// Extracted topics
    pub topics: Vec<String>,
    /// Confidence in the analysis
    pub confidence: f32,
    /// Processing time
    pub processing_time: Duration,
}

/// Linguistic analysis results
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LinguisticAnalysis {
    /// Word count
    pub word_count: usize,
    /// Sentence count
    pub sentence_count: usize,
    /// Average sentence length
    pub avg_sentence_length: f32,
    /// Vocabulary richness (unique words / total words)
    pub vocabulary_richness: f32,
    /// Reading level estimate
    pub reading_level: f32,
    /// Language formality score
    pub formality_score: f32,
    /// Grammar quality score
    pub grammar_score: f32,
}

/// Contextual metrics for conversation analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContextualMetrics {
    /// Topic consistency across turns
    pub topic_consistency: f32,
    /// Context relevance score
    pub context_relevance: f32,
    /// Conversation flow quality
    pub flow_quality: f32,
    /// Turn-taking balance
    pub turn_balance: f32,
    /// Response appropriateness
    pub response_appropriateness: f32,
}

/// Performance metrics for analysis operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisPerformance {
    /// Total analysis time
    pub total_time: Duration,
    /// Analysis start time
    #[serde(skip, default = "Instant::now")]
    pub start_time: Instant,
    /// Number of turns analyzed
    pub turns_analyzed: usize,
    /// Average time per turn
    pub avg_time_per_turn: Duration,
    /// Memory usage estimate
    pub memory_usage_mb: f32,
}

/// Health assessment results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthAssessment {
    /// Overall health score (0.0 to 1.0)
    pub overall_score: f32,
    /// Individual component scores
    pub component_scores: DetailedHealthMetrics,
    /// Identified issues
    pub issues: Vec<HealthIssue>,
    /// Trend analysis
    pub trend: Option<String>,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// Detailed health metrics breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedHealthMetrics {
    /// Coherence score
    pub coherence: f32,
    /// Engagement score
    pub engagement: f32,
    /// Safety score
    pub safety: f32,
    /// Responsiveness score
    pub responsiveness: f32,
    /// Context relevance score
    pub context_relevance: f32,
    /// Emotional balance score
    pub emotional_balance: f32,
    /// Information density score
    pub information_density: f32,
}

/// Individual health issue identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthIssue {
    /// Issue type
    pub issue_type: HealthIssueType,
    /// Severity level (0.0 to 1.0)
    pub severity: f32,
    /// Description of the issue
    pub description: String,
    /// Suggested resolution
    pub resolution: Option<String>,
    /// Turn indices where issue occurs
    pub affected_turns: Vec<usize>,
}

/// Types of health issues that can be detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthIssueType {
    /// Low coherence in responses
    LowCoherence,
    /// Poor engagement patterns
    PoorEngagement,
    /// Safety concerns
    SafetyConcerns,
    /// Slow response times
    SlowResponse,
    /// Irrelevant context usage
    IrrelevantContext,
    /// Repetitive responses
    RepetitiveResponses,
    /// Inconsistent persona
    InconsistentPersona,
    /// Poor topic management
    PoorTopicManagement,
}

/// Enhanced analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedAnalysisConfig {
    /// Enable detailed linguistic analysis
    pub enable_linguistic_analysis: bool,
    /// Enable contextual metrics calculation
    pub enable_contextual_metrics: bool,
    /// Enable health assessment
    pub enable_health_assessment: bool,
    /// Enable performance tracking
    pub enable_performance_tracking: bool,
    /// Safety sensitivity level
    pub safety_sensitivity: SafetySensitivity,
    /// Quality assessment strictness
    pub quality_strictness: QualityStrictness,
    /// Minimum confidence threshold
    pub min_confidence_threshold: f32,
    /// Maximum analysis time per turn
    pub max_analysis_time: Duration,
    /// Enable real-time monitoring
    pub enable_real_time_monitoring: bool,
    /// Batch analysis size
    pub batch_size: usize,
}

/// Safety sensitivity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetySensitivity {
    /// Low sensitivity - minimal filtering
    Low,
    /// Medium sensitivity - balanced approach
    Medium,
    /// High sensitivity - strict filtering
    High,
    /// Maximum sensitivity - very conservative
    Maximum,
}

/// Quality assessment strictness levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityStrictness {
    /// Lenient quality standards
    Lenient,
    /// Standard quality requirements
    Standard,
    /// Strict quality enforcement
    Strict,
    /// Maximum quality standards
    Maximum,
}
