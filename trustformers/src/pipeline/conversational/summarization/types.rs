//! Data types for summarization: results, weights, thresholds, and public type aliases.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::pipeline::conversational::types::{ConversationRole, SummarizationStrategy};
use serde::{Deserialize, Serialize};

use super::engine::ContextSummarizer;

// Legacy compatibility type (from the original single-file module): kept for
// backward compatibility rather than being part of the current core API.
/// Summary with specific constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstrainedSummary {
    pub summary: String,
    pub topics: Option<Vec<String>>,
    pub sentiment_analysis: Option<SentimentAnalysis>,
    pub original_turn_count: usize,
    pub compression_ratio: f32,
}
// Legacy compatibility type (from the original single-file module).
/// A segment of conversation with its summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSegment {
    pub start_turn: usize,
    pub end_turn: usize,
    pub summary: String,
    pub topics: Vec<String>,
    pub turn_count: usize,
}
// Legacy compatibility type (from the original single-file module).
/// Hierarchical summary with segments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalSummary {
    pub overall_summary: String,
    pub main_topics: Vec<String>,
    pub segments: Vec<ConversationSegment>,
    pub total_turns: usize,
}
/// Weights for importance scoring, set via
/// [`ContextSummarizer::with_importance_weights`].
#[derive(Debug, Clone)]
pub struct ImportanceWeights {
    /// Weight for questions in importance calculation
    pub question_weight: f32,
    /// Weight for personal information
    pub personal_info_weight: f32,
    /// Weight for topical relevance
    pub topic_relevance_weight: f32,
    /// Weight for emotional content
    pub emotional_weight: f32,
    /// Weight for reasoning chains
    pub reasoning_weight: f32,
    /// Weight for engagement level
    pub engagement_weight: f32,
    /// Weight for recency (more recent = more important)
    pub recency_weight: f32,
}
impl Default for ImportanceWeights {
    fn default() -> Self {
        Self {
            question_weight: 0.3,
            personal_info_weight: 0.4,
            topic_relevance_weight: 0.25,
            emotional_weight: 0.2,
            reasoning_weight: 0.35,
            engagement_weight: 0.15,
            recency_weight: 0.1,
        }
    }
}
/// Quality assessment result
#[derive(Debug, Clone)]
pub(super) struct QualityAssessment {
    pub(super) quality_score: f32,
    pub(super) confidence: f32,
    pub(super) coherence_score: f32,
}
/// Thresholds for quality assessment, set via
/// [`ContextSummarizer::with_quality_thresholds`]. A summary that falls
/// outside these bounds is still returned (summarization never fails purely
/// on quality), but is logged via `tracing::warn!`.
#[derive(Debug, Clone)]
pub struct QualityThresholds {
    /// Minimum quality score for acceptable summaries
    pub min_quality_score: f32,
    /// Minimum compression ratio to be worthwhile
    pub min_compression_ratio: f32,
    /// Maximum allowable information loss
    pub max_information_loss: f32,
    /// Minimum coherence score
    pub min_coherence_score: f32,
}
impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_quality_score: 0.6,
            min_compression_ratio: 0.3,
            max_information_loss: 0.4,
            min_coherence_score: 0.5,
        }
    }
}
/// Sentence importance score and metadata
#[derive(Debug, Clone)]
pub(super) struct SentenceScore {
    /// The sentence text
    pub(super) sentence: String,
    /// Importance score (0.0 to 1.0)
    pub(super) score: f32,
    /// Position in original text
    pub(super) position: usize,
    /// Turn index this sentence belongs to
    pub(super) turn_index: usize,
    /// Topics this sentence covers
    pub(super) topics: Vec<String>,
    /// Named entities in this sentence
    pub(super) entities: Vec<String>,
    /// Role of the speaker
    pub(super) speaker_role: ConversationRole,
}
// Legacy compatibility type (from the original single-file module).
/// Sentiment analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentAnalysis {
    pub dominant_sentiment: String,
    pub positive_ratio: f32,
    pub negative_ratio: f32,
    pub neutral_ratio: f32,
    pub confidence: f32,
}
/// Result of summarization with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummarizationResult {
    /// Generated summary text
    pub summary: String,
    /// Original token count before summarization
    pub original_tokens: usize,
    /// Summary token count after summarization
    pub summary_tokens: usize,
    /// Compression ratio achieved
    pub compression_ratio: f32,
    /// Quality score of the summary
    pub quality_score: f32,
    /// Strategy used for summarization
    pub strategy_used: SummarizationStrategy,
    /// Key topics preserved
    pub preserved_topics: Vec<String>,
    /// Important entities preserved
    pub preserved_entities: Vec<String>,
    /// Confidence in summary quality
    pub confidence: f32,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
}
/// Topic clustering result for extractive summarization
#[derive(Debug, Clone)]
pub(super) struct TopicCluster {
    /// Central topic/theme
    pub(super) topic: String,
    /// Sentences belonging to this cluster
    pub(super) sentences: Vec<SentenceScore>,
    /// Importance score of this cluster
    pub(super) cluster_score: f32,
    /// Representative sentence for this cluster
    pub(super) representative_sentence: Option<String>,
}
/// Summarization engine alias
pub type SummarizationEngine = ContextSummarizer;

/// Summarization metadata alias
pub type SummarizationMetadata = SummarizationResult;
