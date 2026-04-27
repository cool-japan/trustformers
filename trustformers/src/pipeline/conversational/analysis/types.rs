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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn test_linguistic_analysis_default() {
        let analysis = LinguisticAnalysis::default();
        assert_eq!(analysis.word_count, 0);
        assert_eq!(analysis.sentence_count, 0);
        assert_eq!(analysis.avg_sentence_length, 0.0);
        assert_eq!(analysis.vocabulary_richness, 0.0);
    }

    #[test]
    fn test_contextual_metrics_default() {
        let metrics = ContextualMetrics::default();
        assert_eq!(metrics.topic_consistency, 0.0);
        assert_eq!(metrics.context_relevance, 0.0);
        assert_eq!(metrics.flow_quality, 0.0);
    }

    #[test]
    fn test_health_issue_type_variants_distinct() {
        assert_ne!(
            HealthIssueType::LowCoherence,
            HealthIssueType::PoorEngagement
        );
        assert_ne!(
            HealthIssueType::SafetyConcerns,
            HealthIssueType::SlowResponse
        );
        assert_ne!(
            HealthIssueType::RepetitiveResponses,
            HealthIssueType::InconsistentPersona
        );
    }

    #[test]
    fn test_safety_sensitivity_variants() {
        assert_ne!(SafetySensitivity::Low, SafetySensitivity::High);
        assert_ne!(SafetySensitivity::Medium, SafetySensitivity::Maximum);
    }

    #[test]
    fn test_quality_strictness_variants() {
        assert_ne!(QualityStrictness::Lenient, QualityStrictness::Strict);
        assert_ne!(QualityStrictness::Standard, QualityStrictness::Maximum);
    }

    #[test]
    fn test_health_issue_construction() {
        let issue = HealthIssue {
            issue_type: HealthIssueType::LowCoherence,
            severity: 0.75,
            description: "Poor coherence detected".to_string(),
            resolution: Some("Restructure responses".to_string()),
            affected_turns: vec![0, 1, 2],
        };
        assert_eq!(issue.issue_type, HealthIssueType::LowCoherence);
        assert!((issue.severity - 0.75).abs() < 1e-6);
        assert_eq!(issue.affected_turns.len(), 3);
    }

    #[test]
    fn test_detailed_health_metrics_construction() {
        let metrics = DetailedHealthMetrics {
            coherence: 0.8,
            engagement: 0.7,
            safety: 1.0,
            responsiveness: 0.9,
            context_relevance: 0.75,
            emotional_balance: 0.6,
            information_density: 0.5,
        };
        assert!((metrics.coherence - 0.8).abs() < 1e-6);
        assert!((metrics.safety - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_health_assessment_construction() {
        let assessment = HealthAssessment {
            overall_score: 0.85,
            component_scores: DetailedHealthMetrics {
                coherence: 0.8,
                engagement: 0.9,
                safety: 1.0,
                responsiveness: 0.85,
                context_relevance: 0.7,
                emotional_balance: 0.8,
                information_density: 0.7,
            },
            issues: Vec::new(),
            trend: Some("Improving".to_string()),
            recommendations: vec!["Keep it up".to_string()],
        };
        assert!((assessment.overall_score - 0.85).abs() < 1e-6);
        assert!(assessment.issues.is_empty());
        assert_eq!(assessment.recommendations.len(), 1);
    }

    #[test]
    fn test_turn_analysis_result_construction() {
        let result = TurnAnalysisResult {
            quality_score: 0.9,
            engagement_level: EngagementLevel::High,
            sentiment: Some("positive".to_string()),
            intent: Some("question".to_string()),
            topics: vec!["technology".to_string()],
            confidence: 0.85,
            processing_time: Duration::from_millis(10),
        };
        assert!((result.quality_score - 0.9).abs() < 1e-6);
        assert_eq!(result.topics.len(), 1);
    }

    #[test]
    fn test_analysis_performance_construction() {
        let perf = AnalysisPerformance {
            total_time: Duration::from_secs(1),
            start_time: Instant::now(),
            turns_analyzed: 10,
            avg_time_per_turn: Duration::from_millis(100),
            memory_usage_mb: 12.5,
        };
        assert_eq!(perf.turns_analyzed, 10);
        assert!((perf.memory_usage_mb - 12.5).abs() < 1e-6);
    }

    #[test]
    fn test_enhanced_analysis_config_construction() {
        let config = EnhancedAnalysisConfig {
            enable_linguistic_analysis: true,
            enable_contextual_metrics: false,
            enable_health_assessment: true,
            enable_performance_tracking: false,
            safety_sensitivity: SafetySensitivity::High,
            quality_strictness: QualityStrictness::Strict,
            min_confidence_threshold: 0.6,
            max_analysis_time: Duration::from_secs(5),
            enable_real_time_monitoring: false,
            batch_size: 32,
        };
        assert!(config.enable_linguistic_analysis);
        assert!(!config.enable_contextual_metrics);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.safety_sensitivity, SafetySensitivity::High);
    }

    #[test]
    fn test_health_issue_type_all_variants_covered() {
        let variants = [
            HealthIssueType::LowCoherence,
            HealthIssueType::PoorEngagement,
            HealthIssueType::SafetyConcerns,
            HealthIssueType::SlowResponse,
            HealthIssueType::IrrelevantContext,
            HealthIssueType::RepetitiveResponses,
            HealthIssueType::InconsistentPersona,
            HealthIssueType::PoorTopicManagement,
        ];
        // Ensure all 8 variants are distinct
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j]);
            }
        }
    }

    #[test]
    fn test_linguistic_analysis_custom_values() {
        let analysis = LinguisticAnalysis {
            word_count: 100,
            sentence_count: 10,
            avg_sentence_length: 10.0,
            vocabulary_richness: 0.6,
            reading_level: 8.5,
            formality_score: 0.7,
            grammar_score: 0.9,
        };
        assert_eq!(analysis.word_count, 100);
        assert!((analysis.vocabulary_richness - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_safety_sensitivity_copy() {
        let s1 = SafetySensitivity::Medium;
        let s2 = s1;
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_quality_strictness_copy() {
        let q1 = QualityStrictness::Standard;
        let q2 = q1;
        assert_eq!(q1, q2);
    }
}
