//! Context summarization module for conversational AI pipeline.
//!
//! This module provides comprehensive conversation context summarization capabilities,
//! including multiple summarization strategies, quality assessment, token management,
//! and optimization for different conversation types and requirements.
//!
//! # Features
//!
//! - **Multiple Strategies**: Extractive, abstractive, and hybrid summarization
//! - **Context Compression**: Intelligent compression while preserving key information
//! - **Quality Assessment**: Automatic summary quality scoring and validation
//! - **Token Management**: Precise token counting and context window management
//! - **Adaptive Algorithms**: Different algorithms for different conversation types
//! - **Performance Optimization**: Efficient summarization with minimal latency
//! - **Error Recovery**: Robust error handling and fallback mechanisms

use crate::error::{Result, TrustformersError};
use crate::pipeline::conversational::types::{
    ConversationRole, ConversationTurn, EngagementLevel, ReasoningType, SummarizationConfig,
    SummarizationStrategy,
};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

// ================================================================================================
// CORE SUMMARIZATION TYPES
// ================================================================================================

// ================================================================================================
// TYPE ALIASES
// ================================================================================================

/// Summarization engine alias
pub type SummarizationEngine = ContextSummarizer;

/// Summarization metadata alias
pub type SummarizationMetadata = SummarizationResult;

/// Advanced context summarization component for conversation compression
pub struct ContextSummarizer {
    /// Summarization strategy configuration
    pub config: SummarizationConfig,
    /// Token counting function for accurate estimation
    pub token_counter: Option<Arc<dyn Fn(&str) -> usize + Send + Sync>>,
    /// Cache for frequently used regex patterns
    regex_cache: HashMap<String, Regex>,
    /// Importance scoring weights
    importance_weights: ImportanceWeights,
    /// Quality assessment thresholds
    quality_thresholds: QualityThresholds,
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

/// Weights for importance scoring
#[derive(Debug, Clone)]
struct ImportanceWeights {
    /// Weight for questions in importance calculation
    question_weight: f32,
    /// Weight for personal information
    personal_info_weight: f32,
    /// Weight for topical relevance
    topic_relevance_weight: f32,
    /// Weight for emotional content
    emotional_weight: f32,
    /// Weight for reasoning chains
    reasoning_weight: f32,
    /// Weight for engagement level
    engagement_weight: f32,
    /// Weight for recency (more recent = more important)
    recency_weight: f32,
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

/// Thresholds for quality assessment
#[derive(Debug, Clone)]
struct QualityThresholds {
    /// Minimum quality score for acceptable summaries
    min_quality_score: f32,
    /// Minimum compression ratio to be worthwhile
    min_compression_ratio: f32,
    /// Maximum allowable information loss
    max_information_loss: f32,
    /// Minimum coherence score
    min_coherence_score: f32,
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
struct SentenceScore {
    /// The sentence text
    sentence: String,
    /// Importance score (0.0 to 1.0)
    score: f32,
    /// Position in original text
    position: usize,
    /// Turn index this sentence belongs to
    turn_index: usize,
    /// Topics this sentence covers
    topics: Vec<String>,
    /// Named entities in this sentence
    entities: Vec<String>,
    /// Role of the speaker
    speaker_role: ConversationRole,
}

/// Topic clustering result for extractive summarization
#[derive(Debug, Clone)]
struct TopicCluster {
    /// Central topic/theme
    topic: String,
    /// Sentences belonging to this cluster
    sentences: Vec<SentenceScore>,
    /// Importance score of this cluster
    cluster_score: f32,
    /// Representative sentence for this cluster
    representative_sentence: Option<String>,
}

// ================================================================================================
// LEGACY COMPATIBILITY TYPES (From original file)
// ================================================================================================

/// Hierarchical summary with segments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalSummary {
    pub overall_summary: String,
    pub main_topics: Vec<String>,
    pub segments: Vec<ConversationSegment>,
    pub total_turns: usize,
}

/// A segment of conversation with its summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSegment {
    pub start_turn: usize,
    pub end_turn: usize,
    pub summary: String,
    pub topics: Vec<String>,
    pub turn_count: usize,
}

/// Summary with specific constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstrainedSummary {
    pub summary: String,
    pub topics: Option<Vec<String>>,
    pub sentiment_analysis: Option<SentimentAnalysis>,
    pub original_turn_count: usize,
    pub compression_ratio: f32,
}

/// Sentiment analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentAnalysis {
    pub dominant_sentiment: String,
    pub positive_ratio: f32,
    pub negative_ratio: f32,
    pub neutral_ratio: f32,
    pub confidence: f32,
}

// ================================================================================================
// IMPLEMENTATION
// ================================================================================================

impl std::fmt::Debug for ContextSummarizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContextSummarizer")
            .field("config", &self.config)
            .field(
                "token_counter",
                &self.token_counter.as_ref().map(|_| "<function>"),
            )
            .field(
                "regex_cache",
                &format!("{} cached patterns", self.regex_cache.len()),
            )
            .field("importance_weights", &self.importance_weights)
            .field("quality_thresholds", &self.quality_thresholds)
            .finish()
    }
}

impl Clone for ContextSummarizer {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            token_counter: self.token_counter.clone(),
            regex_cache: self.regex_cache.clone(),
            importance_weights: self.importance_weights.clone(),
            quality_thresholds: self.quality_thresholds.clone(),
        }
    }
}

impl ContextSummarizer {
    /// Create a new context summarizer with configuration
    pub fn new(config: SummarizationConfig) -> Self {
        Self {
            config,
            token_counter: None,
            regex_cache: HashMap::new(),
            importance_weights: ImportanceWeights::default(),
            quality_thresholds: QualityThresholds::default(),
        }
    }

    /// Create context summarizer with simple strategy and target length (legacy compatibility)
    pub fn with_strategy(strategy: SummarizationStrategy, target_length: usize) -> Self {
        let mut config = SummarizationConfig::default();
        config.strategy = strategy;
        config.target_length = target_length;
        Self::new(config)
    }

    /// Create context summarizer with custom token counter
    pub fn with_token_counter<F>(mut self, token_counter: F) -> Self
    where
        F: Fn(&str) -> usize + Send + Sync + 'static,
    {
        self.token_counter = Some(Arc::new(token_counter));
        self
    }

    /// Set custom importance weights
    pub fn with_importance_weights(mut self, weights: ImportanceWeights) -> Self {
        self.importance_weights = weights;
        self
    }

    /// Set custom quality thresholds
    pub fn with_quality_thresholds(mut self, thresholds: QualityThresholds) -> Self {
        self.quality_thresholds = thresholds;
        self
    }

    /// Summarize conversation history - legacy method for compatibility
    pub fn summarize_context(&mut self, turns: &[ConversationTurn]) -> Result<String> {
        let result = self.summarize_context_enhanced(turns)?;
        Ok(result.summary)
    }

    /// Summarize conversation history with comprehensive analysis
    pub fn summarize_context_enhanced(
        &mut self,
        turns: &[ConversationTurn],
    ) -> Result<SummarizationResult> {
        let start_time = std::time::Instant::now();

        if turns.is_empty() {
            return Ok(SummarizationResult {
                summary: String::new(),
                original_tokens: 0,
                summary_tokens: 0,
                compression_ratio: 1.0,
                quality_score: 1.0,
                strategy_used: self.config.strategy.clone(),
                preserved_topics: Vec::new(),
                preserved_entities: Vec::new(),
                confidence: 1.0,
                processing_time_ms: start_time.elapsed().as_millis() as f64,
            });
        }

        // Calculate original token count
        let original_tokens = self.calculate_total_tokens(turns);

        // Check if summarization is necessary
        if original_tokens <= self.config.target_length {
            let summary = self.build_full_context(turns);
            return Ok(SummarizationResult {
                summary: summary.clone(),
                original_tokens,
                summary_tokens: self.count_tokens(&summary),
                compression_ratio: 1.0,
                quality_score: 1.0,
                strategy_used: self.config.strategy.clone(),
                preserved_topics: self.extract_all_topics(turns),
                preserved_entities: self.extract_all_entities(turns),
                confidence: 1.0,
                processing_time_ms: start_time.elapsed().as_millis() as f64,
            });
        }

        // Perform summarization based on strategy
        let summary = match self.config.strategy {
            SummarizationStrategy::Extractive => self.extractive_summary(turns)?,
            SummarizationStrategy::Abstractive => self.abstractive_summary(turns)?,
            SummarizationStrategy::Hybrid => self.hybrid_summary(turns)?,
        };

        let summary_tokens = self.count_tokens(&summary);
        let compression_ratio = summary_tokens as f32 / original_tokens as f32;

        // Assess summary quality
        let quality_assessment = self.assess_summary_quality(&summary, turns, compression_ratio);

        // Extract preserved information
        let preserved_topics = self.extract_preserved_topics(&summary, turns);
        let preserved_entities = self.extract_preserved_entities(&summary, turns);

        let processing_time = start_time.elapsed().as_millis() as f64;

        Ok(SummarizationResult {
            summary,
            original_tokens,
            summary_tokens,
            compression_ratio,
            quality_score: quality_assessment.quality_score,
            strategy_used: self.config.strategy.clone(),
            preserved_topics,
            preserved_entities,
            confidence: quality_assessment.confidence,
            processing_time_ms: processing_time,
        })
    }

    /// Perform extractive summarization using sentence scoring and clustering
    fn extractive_summary(&mut self, turns: &[ConversationTurn]) -> Result<String> {
        // Score all sentences for importance
        let scored_sentences = self.score_sentences(turns)?;

        // Cluster sentences by topic for better coverage
        let topic_clusters = self.cluster_by_topics(&scored_sentences);

        // Select representative sentences from each cluster
        let mut selected_sentences = Vec::new();
        let mut current_tokens = 0;
        let target_tokens = self.config.target_length;

        // Sort clusters by importance
        let mut sorted_clusters = topic_clusters;
        sorted_clusters.sort_by(|a, b| {
            b.cluster_score.partial_cmp(&a.cluster_score).unwrap_or(Ordering::Equal)
        });

        // Select sentences from most important clusters first
        for cluster in sorted_clusters {
            for sentence_score in cluster.sentences {
                let sentence_tokens = self.count_tokens(&sentence_score.sentence);
                if current_tokens + sentence_tokens <= target_tokens {
                    selected_sentences.push(sentence_score);
                    current_tokens += sentence_tokens;
                } else if selected_sentences.is_empty() {
                    // Ensure we include at least one sentence even if it exceeds target
                    selected_sentences.push(sentence_score);
                    break;
                }

                if current_tokens >= target_tokens {
                    break;
                }
            }

            if current_tokens >= target_tokens {
                break;
            }
        }

        // Sort selected sentences by original position to maintain coherence
        selected_sentences.sort_by_key(|s| (s.turn_index, s.position));

        // Build coherent summary
        self.build_coherent_summary(selected_sentences)
    }

    /// Perform abstractive summarization using template-based generation
    fn abstractive_summary(&self, turns: &[ConversationTurn]) -> Result<String> {
        // Extract key information for abstraction
        let key_topics = self.extract_key_topics(turns, 5);
        let key_entities = self.extract_key_entities(turns, 10);
        let conversation_flow = self.analyze_conversation_flow(turns);
        let emotional_arc = self.analyze_emotional_arc(turns);

        // Count turns by role for context
        let user_turns = turns.iter().filter(|t| matches!(t.role, ConversationRole::User)).count();
        let assistant_turns =
            turns.iter().filter(|t| matches!(t.role, ConversationRole::Assistant)).count();

        // Build abstractive summary using templates
        let mut summary_parts = Vec::new();

        // Add conversation overview
        summary_parts.push(format!(
            "Conversation summary ({} user messages, {} assistant responses):",
            user_turns, assistant_turns
        ));

        // Add key topics
        if !key_topics.is_empty() {
            summary_parts.push(format!("Main topics discussed: {}", key_topics.join(", ")));
        }

        // Add key entities
        if !key_entities.is_empty() {
            summary_parts.push(format!(
                "Key entities mentioned: {}",
                key_entities.join(", ")
            ));
        }

        // Add conversation flow insights
        if let Some(flow_summary) = conversation_flow {
            summary_parts.push(flow_summary);
        }

        // Add emotional context if significant
        if let Some(emotional_summary) = emotional_arc {
            summary_parts.push(emotional_summary);
        }

        // Add specific important exchanges
        let important_exchanges = self.extract_important_exchanges(turns, 2);
        for exchange in important_exchanges {
            summary_parts.push(exchange);
        }

        let summary = summary_parts.join(" ");

        // Ensure summary fits within target length
        self.trim_to_target_length(summary)
    }

    /// Perform hybrid summarization combining extractive and abstractive approaches
    fn hybrid_summary(&mut self, turns: &[ConversationTurn]) -> Result<String> {
        // Use 60% of target for extractive, 40% for abstractive
        let extractive_target = (self.config.target_length as f32 * 0.6) as usize;
        let abstractive_target = self.config.target_length - extractive_target;

        // Create temporary config for extractive phase
        let mut extractive_config = self.config.clone();
        extractive_config.target_length = extractive_target;
        let original_config = std::mem::replace(&mut self.config, extractive_config);

        // Get extractive summary
        let extractive_part = self.extractive_summary(turns)?;

        // Restore original config and set for abstractive
        self.config = original_config;
        self.config.target_length = abstractive_target;

        // Get abstractive summary
        let abstractive_part = self.abstractive_summary(turns)?;

        // Restore original target length
        self.config.target_length = extractive_target + abstractive_target;

        // Combine both summaries intelligently
        let combined = if extractive_part.is_empty() {
            abstractive_part
        } else if abstractive_part.is_empty() {
            extractive_part
        } else {
            format!("{} {}", abstractive_part, extractive_part)
        };

        // Final trimming to ensure target length
        self.trim_to_target_length(combined)
    }

    /// Score individual sentences for importance
    fn score_sentences(&self, turns: &[ConversationTurn]) -> Result<Vec<SentenceScore>> {
        let mut scored_sentences = Vec::new();

        for (turn_index, turn) in turns.iter().enumerate() {
            let sentences = self.split_into_sentences(&turn.content);

            for (position, sentence) in sentences.into_iter().enumerate() {
                if sentence.trim().is_empty() {
                    continue;
                }

                let score = self.calculate_sentence_importance(&sentence, turn, turn_index);
                let topics = self.extract_sentence_topics(&sentence);
                let entities = self.extract_sentence_entities(&sentence);

                scored_sentences.push(SentenceScore {
                    sentence,
                    score,
                    position,
                    turn_index,
                    topics,
                    entities,
                    speaker_role: turn.role.clone(),
                });
            }
        }

        // Sort by importance score
        scored_sentences.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

        Ok(scored_sentences)
    }

    /// Calculate importance score for a sentence
    fn calculate_sentence_importance(
        &self,
        sentence: &str,
        turn: &ConversationTurn,
        turn_index: usize,
    ) -> f32 {
        let mut importance = 0.0;
        let sentence_lower = sentence.to_lowercase();

        // Question bonus
        if sentence.contains('?') {
            importance += self.importance_weights.question_weight;
        }

        // Personal information bonus
        if self.contains_personal_info(&sentence_lower) {
            importance += self.importance_weights.personal_info_weight;
        }

        // Topic relevance (based on metadata)
        if let Some(metadata) = &turn.metadata {
            if !metadata.topics.is_empty() {
                importance += self.importance_weights.topic_relevance_weight;
            }

            // Engagement level bonus
            let engagement_bonus = match metadata.engagement_level {
                EngagementLevel::VeryHigh => 0.4,
                EngagementLevel::High => 0.3,
                EngagementLevel::Medium => 0.1,
                EngagementLevel::Low => 0.0,
            };
            importance += engagement_bonus * self.importance_weights.engagement_weight;

            // Reasoning type bonus
            if let Some(reasoning_type) = &metadata.reasoning_type {
                let reasoning_bonus = match reasoning_type {
                    ReasoningType::Logical | ReasoningType::Mathematical => 0.3,
                    ReasoningType::Causal | ReasoningType::Analogical => 0.25,
                    ReasoningType::Creative | ReasoningType::Emotional => 0.2,
                };
                importance += reasoning_bonus * self.importance_weights.reasoning_weight;
            }
        }

        // Emotional content bonus
        if self.contains_emotional_content(&sentence_lower) {
            importance += self.importance_weights.emotional_weight;
        }

        // Length factor (not too short, not too long)
        let length_factor = self.calculate_length_factor(sentence);
        importance *= length_factor;

        // Recency factor (more recent turns are slightly more important)
        let recency_factor = 1.0 - (turn_index as f32 * 0.05).min(0.5);
        importance += recency_factor * self.importance_weights.recency_weight;

        importance.min(1.0).max(0.0)
    }

    /// Check if sentence contains personal information
    fn contains_personal_info(&self, sentence: &str) -> bool {
        let personal_patterns = [
            "i am",
            "my name",
            "i like",
            "i prefer",
            "i want",
            "i need",
            "i work",
            "i live",
            "my job",
            "my family",
            "my hobby",
        ];

        personal_patterns.iter().any(|&pattern| sentence.contains(pattern))
    }

    /// Check if sentence contains emotional content
    fn contains_emotional_content(&self, sentence: &str) -> bool {
        let emotional_words = [
            "love",
            "hate",
            "happy",
            "sad",
            "angry",
            "excited",
            "frustrated",
            "disappointed",
            "pleased",
            "worried",
            "nervous",
            "confident",
            "feel",
            "feeling",
            "emotion",
            "heart",
            "soul",
        ];

        emotional_words.iter().any(|&word| sentence.contains(word))
    }

    /// Calculate length factor for sentence scoring
    fn calculate_length_factor(&self, sentence: &str) -> f32 {
        let word_count = sentence.split_whitespace().count();

        match word_count {
            0..=3 => 0.3,   // Too short
            4..=8 => 0.8,   // Short but meaningful
            9..=20 => 1.0,  // Good length
            21..=30 => 0.9, // A bit long
            31..=50 => 0.7, // Long
            _ => 0.5,       // Very long
        }
    }

    /// Cluster sentences by topics for better coverage
    fn cluster_by_topics(&self, sentences: &[SentenceScore]) -> Vec<TopicCluster> {
        let mut topic_map: HashMap<String, Vec<SentenceScore>> = HashMap::new();
        let mut uncategorized = Vec::new();

        // Group sentences by topics
        for sentence in sentences {
            if sentence.topics.is_empty() {
                uncategorized.push(sentence.clone());
            } else {
                for topic in &sentence.topics {
                    topic_map.entry(topic.clone()).or_default().push(sentence.clone());
                }
            }
        }

        let mut clusters = Vec::new();

        // Create clusters for each topic
        for (topic, mut topic_sentences) in topic_map {
            // Sort sentences within topic by importance
            topic_sentences
                .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

            // Calculate cluster score as average of top sentences
            let cluster_score = if topic_sentences.is_empty() {
                0.0
            } else {
                let top_count = (topic_sentences.len() / 2).max(1).min(3);
                topic_sentences.iter().take(top_count).map(|s| s.score).sum::<f32>()
                    / top_count as f32
            };

            // Find representative sentence
            let representative_sentence = topic_sentences.first().map(|s| s.sentence.clone());

            clusters.push(TopicCluster {
                topic,
                sentences: topic_sentences,
                cluster_score,
                representative_sentence,
            });
        }

        // Add uncategorized sentences as a general cluster
        if !uncategorized.is_empty() {
            uncategorized.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
            let cluster_score = uncategorized.iter().take(3).map(|s| s.score).sum::<f32>() / 3.0;

            clusters.push(TopicCluster {
                topic: "general".to_string(),
                sentences: uncategorized,
                cluster_score,
                representative_sentence: None,
            });
        }

        clusters
    }

    /// Build coherent summary from selected sentences
    fn build_coherent_summary(&self, sentences: Vec<SentenceScore>) -> Result<String> {
        if sentences.is_empty() {
            return Ok(String::new());
        }

        let mut summary_parts = Vec::new();
        let mut current_role: Option<ConversationRole> = None;

        for sentence in sentences {
            // Add role marker if role changes
            if current_role.as_ref() != Some(&sentence.speaker_role) {
                let role_marker = match sentence.speaker_role {
                    ConversationRole::User => "User:",
                    ConversationRole::Assistant => "Assistant:",
                    ConversationRole::System => "System:",
                };

                if !summary_parts.is_empty() {
                    summary_parts.push(" ".to_string());
                }
                summary_parts.push(format!("{} ", role_marker));
                current_role = Some(sentence.speaker_role);
            }

            summary_parts.push(sentence.sentence);
            summary_parts.push(" ".to_string());
        }

        Ok(summary_parts.concat().trim().to_string())
    }

    /// Extract key topics from conversation
    fn extract_key_topics(&self, turns: &[ConversationTurn], limit: usize) -> Vec<String> {
        let mut topic_counts: HashMap<String, usize> = HashMap::new();

        for turn in turns {
            if let Some(metadata) = &turn.metadata {
                for topic in &metadata.topics {
                    *topic_counts.entry(topic.clone()).or_insert(0) += 1;
                }
            }
        }

        let mut topics: Vec<_> = topic_counts.into_iter().collect();
        topics.sort_by(|a, b| b.1.cmp(&a.1));

        topics.into_iter().take(limit).map(|(topic, _)| topic).collect()
    }

    /// Extract key entities from conversation
    fn extract_key_entities(&self, turns: &[ConversationTurn], limit: usize) -> Vec<String> {
        let mut entity_counts: HashMap<String, usize> = HashMap::new();

        for turn in turns {
            if let Some(metadata) = &turn.metadata {
                for entity in &metadata.entities {
                    *entity_counts.entry(entity.text.clone()).or_insert(0) += 1;
                }
            }
        }

        let mut entities: Vec<_> = entity_counts.into_iter().collect();
        entities.sort_by(|a, b| b.1.cmp(&a.1));

        entities.into_iter().take(limit).map(|(entity, _)| entity).collect()
    }

    /// Analyze conversation flow for abstractive summary
    fn analyze_conversation_flow(&self, turns: &[ConversationTurn]) -> Option<String> {
        if turns.len() < 3 {
            return None;
        }

        let question_count = turns.iter().filter(|t| t.content.contains('?')).count();
        let total_turns = turns.len();
        let question_ratio = question_count as f32 / total_turns as f32;

        let flow_type = if question_ratio > 0.4 {
            "inquiry-heavy discussion"
        } else if question_ratio > 0.2 {
            "interactive conversation"
        } else {
            "informational exchange"
        };

        Some(format!("The conversation followed a {} pattern", flow_type))
    }

    /// Analyze emotional arc for abstractive summary
    fn analyze_emotional_arc(&self, turns: &[ConversationTurn]) -> Option<String> {
        let mut sentiment_progression = Vec::new();

        for turn in turns {
            if let Some(metadata) = &turn.metadata {
                if let Some(sentiment) = &metadata.sentiment {
                    sentiment_progression.push(sentiment.clone());
                }
            }
        }

        if sentiment_progression.len() < 2 {
            return None;
        }

        let initial_sentiment = &sentiment_progression[0];
        let final_sentiment = sentiment_progression.last().unwrap();

        if initial_sentiment != final_sentiment {
            Some(format!(
                "The emotional tone shifted from {} to {} throughout the conversation",
                initial_sentiment, final_sentiment
            ))
        } else {
            Some(format!(
                "The conversation maintained a {} tone",
                initial_sentiment
            ))
        }
    }

    /// Extract important exchanges for abstractive summary
    fn extract_important_exchanges(&self, turns: &[ConversationTurn], limit: usize) -> Vec<String> {
        let mut exchanges = Vec::new();

        for i in 0..turns.len().saturating_sub(1) {
            let current_turn = &turns[i];
            let next_turn = &turns[i + 1];

            // Look for question-answer pairs
            if current_turn.content.contains('?')
                && matches!(current_turn.role, ConversationRole::User)
                && matches!(next_turn.role, ConversationRole::Assistant)
            {
                let exchange = format!(
                    "User asked about {}, Assistant responded with {}",
                    self.extract_question_topic(&current_turn.content),
                    self.extract_response_summary(&next_turn.content)
                );
                exchanges.push(exchange);
            }
        }

        exchanges.into_iter().take(limit).collect()
    }

    /// Extract topic from a question
    fn extract_question_topic(&self, content: &str) -> String {
        // Simple keyword extraction for question topics
        let keywords = ["what", "how", "why", "when", "where", "who"];
        let content_lower = content.to_lowercase();

        for keyword in keywords {
            if let Some(start) = content_lower.find(keyword) {
                let rest = &content[start..];
                if let Some(end) = rest.find('?') {
                    let question_part = &rest[..end + 1];
                    return question_part.trim().to_string();
                }
            }
        }

        "a topic".to_string()
    }

    /// Extract summary from a response
    fn extract_response_summary(&self, content: &str) -> String {
        let words: Vec<&str> = content.split_whitespace().take(10).collect();
        if words.len() < 10 {
            content.to_string()
        } else {
            format!("{}...", words.join(" "))
        }
    }

    /// Split text into sentences
    fn split_into_sentences(&self, text: &str) -> Vec<String> {
        // Simple sentence splitting on common punctuation
        let sentences: Vec<String> = text
            .split(&['.', '!', '?'])
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && s.len() > 5)
            .collect();

        if sentences.is_empty() {
            vec![text.to_string()]
        } else {
            sentences
        }
    }

    /// Extract topics from a sentence
    fn extract_sentence_topics(&self, sentence: &str) -> Vec<String> {
        let mut topics = Vec::new();
        let sentence_lower = sentence.to_lowercase();

        let topic_keywords = [
            (
                "technology",
                &["computer", "software", "tech", "ai", "programming", "code"] as &[&str],
            ),
            (
                "sports",
                &[
                    "football",
                    "basketball",
                    "soccer",
                    "tennis",
                    "game",
                    "sport",
                ],
            ),
            (
                "food",
                &["restaurant", "cooking", "recipe", "eat", "meal", "food"],
            ),
            (
                "travel",
                &["trip", "vacation", "visit", "country", "hotel", "travel"],
            ),
            (
                "work",
                &["job", "career", "office", "meeting", "project", "work"],
            ),
            (
                "health",
                &[
                    "doctor", "medicine", "exercise", "wellness", "fitness", "health",
                ],
            ),
            (
                "education",
                &[
                    "school",
                    "university",
                    "learn",
                    "study",
                    "education",
                    "teacher",
                ],
            ),
            (
                "family",
                &["family", "parents", "children", "kids", "relatives", "home"],
            ),
        ];

        for (topic, keywords) in topic_keywords {
            if keywords.iter().any(|keyword| sentence_lower.contains(keyword)) {
                topics.push(topic.to_string());
            }
        }

        topics
    }

    /// Extract entities from a sentence (simplified)
    fn extract_sentence_entities(&self, sentence: &str) -> Vec<String> {
        let mut entities = Vec::new();

        // Simple patterns for common entity types
        let patterns = [
            (r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", "PERSON"),
            (r"\b\d{1,2}/\d{1,2}/\d{4}\b", "DATE"),
            (r"\b\d{4}-\d{2}-\d{2}\b", "DATE"),
            (r"\$\d+(?:\.\d{2})?\b", "MONEY"),
        ];

        for (pattern, _entity_type) in patterns {
            if let Ok(regex) = Regex::new(pattern) {
                for mat in regex.find_iter(sentence) {
                    entities.push(mat.as_str().to_string());
                }
            }
        }

        entities
    }

    /// Build full context without summarization
    fn build_full_context(&self, turns: &[ConversationTurn]) -> String {
        turns
            .iter()
            .map(|turn| {
                let role_str = match turn.role {
                    ConversationRole::User => "User",
                    ConversationRole::Assistant => "Assistant",
                    ConversationRole::System => "System",
                };
                format!("{}: {}", role_str, turn.content)
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Calculate total tokens across all turns
    fn calculate_total_tokens(&self, turns: &[ConversationTurn]) -> usize {
        turns.iter().map(|turn| self.count_tokens(&turn.content)).sum()
    }

    /// Count tokens in text
    fn count_tokens(&self, text: &str) -> usize {
        if let Some(ref counter) = self.token_counter {
            counter(text)
        } else {
            // Fallback estimation: ~4 characters per token
            text.len() / 4
        }
    }

    /// Trim summary to target length
    fn trim_to_target_length(&self, summary: String) -> Result<String> {
        let current_tokens = self.count_tokens(&summary);

        if current_tokens <= self.config.target_length {
            return Ok(summary);
        }

        // Calculate target character count
        let target_chars = (summary.len() as f32 * self.config.target_length as f32
            / current_tokens as f32) as usize;

        // Trim at word boundary
        if let Some(truncated) = self.truncate_at_word_boundary(&summary, target_chars) {
            Ok(truncated)
        } else {
            Ok(summary)
        }
    }

    /// Truncate text at word boundary
    fn truncate_at_word_boundary(&self, text: &str, max_chars: usize) -> Option<String> {
        if text.len() <= max_chars {
            return Some(text.to_string());
        }

        let truncated = &text[..max_chars];
        truncated
            .rfind(' ')
            .map(|last_space| format!("{}...", &truncated[..last_space]))
    }

    /// Extract all topics from conversation
    fn extract_all_topics(&self, turns: &[ConversationTurn]) -> Vec<String> {
        let mut topics = HashSet::new();

        for turn in turns {
            if let Some(metadata) = &turn.metadata {
                for topic in &metadata.topics {
                    topics.insert(topic.clone());
                }
            }
        }

        topics.into_iter().collect()
    }

    /// Extract all entities from conversation
    fn extract_all_entities(&self, turns: &[ConversationTurn]) -> Vec<String> {
        let mut entities = HashSet::new();

        for turn in turns {
            if let Some(metadata) = &turn.metadata {
                for entity in &metadata.entities {
                    entities.insert(entity.text.clone());
                }
            }
        }

        entities.into_iter().collect()
    }

    /// Extract topics preserved in summary
    fn extract_preserved_topics(
        &self,
        summary: &str,
        original_turns: &[ConversationTurn],
    ) -> Vec<String> {
        let original_topics = self.extract_all_topics(original_turns);
        let summary_lower = summary.to_lowercase();

        original_topics
            .into_iter()
            .filter(|topic| summary_lower.contains(&topic.to_lowercase()))
            .collect()
    }

    /// Extract entities preserved in summary
    fn extract_preserved_entities(
        &self,
        summary: &str,
        original_turns: &[ConversationTurn],
    ) -> Vec<String> {
        let original_entities = self.extract_all_entities(original_turns);
        let summary_lower = summary.to_lowercase();

        original_entities
            .into_iter()
            .filter(|entity| summary_lower.contains(&entity.to_lowercase()))
            .collect()
    }

    /// Assess summary quality
    fn assess_summary_quality(
        &self,
        summary: &str,
        original_turns: &[ConversationTurn],
        compression_ratio: f32,
    ) -> QualityAssessment {
        let mut quality_score = 0.0;
        let mut confidence: f32 = 1.0;

        // Length appropriateness (0.2 weight)
        let length_score = if summary.trim().is_empty() {
            0.0
        } else if compression_ratio > 0.8 {
            0.5 // Little compression
        } else if compression_ratio < 0.1 {
            0.3 // Too much compression
        } else {
            1.0 // Good compression
        };
        quality_score += length_score * 0.2;

        // Topic preservation (0.3 weight)
        let original_topics = self.extract_all_topics(original_turns);
        let preserved_topics = self.extract_preserved_topics(summary, original_turns);
        let topic_preservation = if original_topics.is_empty() {
            1.0
        } else {
            preserved_topics.len() as f32 / original_topics.len() as f32
        };
        quality_score += topic_preservation * 0.3;

        // Entity preservation (0.2 weight)
        let original_entities = self.extract_all_entities(original_turns);
        let preserved_entities = self.extract_preserved_entities(summary, original_turns);
        let entity_preservation = if original_entities.is_empty() {
            1.0
        } else {
            preserved_entities.len() as f32 / original_entities.len() as f32
        };
        quality_score += entity_preservation * 0.2;

        // Coherence (0.2 weight) - simplified heuristic
        let coherence_score = self.assess_coherence(summary);
        quality_score += coherence_score * 0.2;

        // Readability (0.1 weight)
        let readability_score = self.assess_readability(summary);
        quality_score += readability_score * 0.1;

        // Adjust confidence based on various factors
        if compression_ratio < 0.2 {
            confidence *= 0.8; // Less confident with high compression
        }
        if original_turns.len() < 3 {
            confidence *= 0.9; // Less confident with few turns
        }

        QualityAssessment {
            quality_score: quality_score.min(1.0).max(0.0),
            confidence: confidence.min(1.0).max(0.0),
        }
    }

    /// Assess summary coherence
    fn assess_coherence(&self, summary: &str) -> f32 {
        if summary.trim().is_empty() {
            return 0.0;
        }

        let mut coherence_score: f32 = 0.5; // Base score

        // Check for proper sentence structure
        let sentence_endings = summary.matches(&['.', '!', '?']).count();
        let sentences = self.split_into_sentences(summary).len();
        if sentences > 0 && sentence_endings > 0 {
            coherence_score += 0.2;
        }

        // Check for role markers (indicates conversation structure preserved)
        if summary.contains("User:") || summary.contains("Assistant:") {
            coherence_score += 0.2;
        }

        // Check for transition words
        let transitions = [
            "however",
            "therefore",
            "meanwhile",
            "additionally",
            "furthermore",
        ];
        if transitions.iter().any(|&word| summary.to_lowercase().contains(word)) {
            coherence_score += 0.1;
        }

        coherence_score.min(1.0)
    }

    /// Assess summary readability
    fn assess_readability(&self, summary: &str) -> f32 {
        if summary.trim().is_empty() {
            return 0.0;
        }

        let word_count = summary.split_whitespace().count();
        let sentence_count = summary.matches(&['.', '!', '?']).count().max(1);
        let avg_sentence_length = word_count as f32 / sentence_count as f32;

        // Optimal sentence length is around 15-20 words

        if avg_sentence_length < 5.0 {
            0.6 // Too short
        } else if avg_sentence_length <= 25.0 {
            1.0 // Good length
        } else if avg_sentence_length <= 35.0 {
            0.8 // A bit long
        } else {
            0.5 // Too long
        }
    }

    // ================================================================================================
    // LEGACY COMPATIBILITY METHODS (From original file)
    // ================================================================================================

    /// Generate topic-focused summary (legacy compatibility)
    pub fn summarize_by_topic(
        &self,
        turns: &[ConversationTurn],
        target_topic: &str,
    ) -> Result<String> {
        let relevant_turns: Vec<_> = turns
            .iter()
            .filter(|turn| {
                if let Some(metadata) = &turn.metadata {
                    metadata.topics.iter().any(|topic| topic.contains(target_topic))
                } else {
                    turn.content.to_lowercase().contains(&target_topic.to_lowercase())
                }
            })
            .collect();

        if relevant_turns.is_empty() {
            return Ok(format!("No discussion found about topic: {}", target_topic));
        }

        let cloned_turns = relevant_turns.into_iter().cloned().collect::<Vec<_>>();
        let mut cloned_summarizer = self.clone();
        cloned_summarizer.summarize_context(&cloned_turns)
    }

    /// Generate time-based summary (legacy compatibility)
    pub fn summarize_time_window(
        &self,
        turns: &[ConversationTurn],
        start_time: chrono::DateTime<chrono::Utc>,
        end_time: chrono::DateTime<chrono::Utc>,
    ) -> Result<String> {
        let windowed_turns: Vec<_> = turns
            .iter()
            .filter(|turn| turn.timestamp >= start_time && turn.timestamp <= end_time)
            .cloned()
            .collect();

        if windowed_turns.is_empty() {
            return Ok("No conversation activity in the specified time window.".to_string());
        }

        let mut cloned_summarizer = self.clone();
        cloned_summarizer.summarize_context(&windowed_turns)
    }

    /// Generate hierarchical summary (legacy compatibility)
    pub fn hierarchical_summary(&self, turns: &[ConversationTurn]) -> Result<HierarchicalSummary> {
        let total_turns = turns.len();

        // Divide into segments
        let segment_size = (total_turns / 3).max(1);
        let mut segments = Vec::new();

        for i in (0..total_turns).step_by(segment_size) {
            let end = (i + segment_size).min(total_turns);
            let segment_turns = &turns[i..end];

            if !segment_turns.is_empty() {
                let mut cloned_summarizer = self.clone();
                let segment_summary = cloned_summarizer.summarize_context(segment_turns)?;
                let segment_topics = self.extract_segment_topics(segment_turns);

                segments.push(ConversationSegment {
                    start_turn: i,
                    end_turn: end - 1,
                    summary: segment_summary,
                    topics: segment_topics,
                    turn_count: segment_turns.len(),
                });
            }
        }

        // Generate overall summary
        let mut cloned_summarizer = self.clone();
        let overall_summary = cloned_summarizer.summarize_context(turns)?;
        let main_topics = self.extract_main_topics(turns);

        Ok(HierarchicalSummary {
            overall_summary,
            main_topics,
            segments,
            total_turns,
        })
    }

    /// Extract main topics from conversation (legacy compatibility)
    fn extract_main_topics(&self, turns: &[ConversationTurn]) -> Vec<String> {
        let mut topic_counts = HashMap::new();

        for turn in turns {
            if let Some(metadata) = &turn.metadata {
                for topic in &metadata.topics {
                    *topic_counts.entry(topic.clone()).or_insert(0) += 1;
                }
            }
        }

        let mut topics: Vec<_> = topic_counts.into_iter().collect();
        topics.sort_by(|a, b| b.1.cmp(&a.1));

        topics.into_iter()
            .take(5) // Top 5 topics
            .map(|(topic, _)| topic)
            .collect()
    }

    /// Extract topics from a conversation segment (legacy compatibility)
    fn extract_segment_topics(&self, turns: &[ConversationTurn]) -> Vec<String> {
        let mut topics = HashSet::new();

        for turn in turns {
            if let Some(metadata) = &turn.metadata {
                topics.extend(metadata.topics.iter().cloned());
            }
        }

        topics.into_iter().collect()
    }

    /// Generate summary with specified constraints (legacy compatibility)
    pub fn constrained_summary(
        &self,
        turns: &[ConversationTurn],
        max_length: usize,
        include_topics: bool,
        include_sentiment: bool,
    ) -> Result<ConstrainedSummary> {
        let mut cloned_summarizer = self.clone();
        let base_summary = cloned_summarizer.summarize_context(turns)?;

        let mut final_summary = base_summary;
        if final_summary.len() > max_length {
            final_summary.truncate(max_length - 3);
            final_summary.push_str("...");
        }

        let topics = if include_topics { Some(self.extract_main_topics(turns)) } else { None };

        let sentiment_analysis = if include_sentiment {
            Some(self.analyze_overall_sentiment(turns))
        } else {
            None
        };

        Ok(ConstrainedSummary {
            summary: final_summary.clone(),
            topics,
            sentiment_analysis,
            original_turn_count: turns.len(),
            compression_ratio: turns.iter().map(|t| t.content.len()).sum::<usize>() as f32
                / final_summary.len() as f32,
        })
    }

    /// Analyze overall sentiment of conversation (legacy compatibility)
    fn analyze_overall_sentiment(&self, turns: &[ConversationTurn]) -> SentimentAnalysis {
        let mut positive_count = 0;
        let mut negative_count = 0;
        let mut neutral_count = 0;
        let mut total_confidence = 0.0;

        for turn in turns {
            if let Some(metadata) = &turn.metadata {
                total_confidence += metadata.confidence;

                if let Some(sentiment) = &metadata.sentiment {
                    match sentiment.as_str() {
                        "positive" => positive_count += 1,
                        "negative" => negative_count += 1,
                        _ => neutral_count += 1,
                    }
                }
            }
        }

        let total_turns = turns.len();
        let avg_confidence =
            if total_turns > 0 { total_confidence / total_turns as f32 } else { 0.0 };

        let dominant_sentiment =
            if positive_count > negative_count && positive_count > neutral_count {
                "positive".to_string()
            } else if negative_count > positive_count && negative_count > neutral_count {
                "negative".to_string()
            } else {
                "neutral".to_string()
            };

        SentimentAnalysis {
            dominant_sentiment,
            positive_ratio: positive_count as f32 / total_turns as f32,
            negative_ratio: negative_count as f32 / total_turns as f32,
            neutral_ratio: neutral_count as f32 / total_turns as f32,
            confidence: avg_confidence,
        }
    }
}

/// Quality assessment result
#[derive(Debug, Clone)]
struct QualityAssessment {
    quality_score: f32,
    confidence: f32,
}

// ================================================================================================
// ADDITIONAL HELPER FUNCTIONS
// ================================================================================================

/// Validate summarization configuration
pub fn validate_summarization_config(config: &SummarizationConfig) -> Result<()> {
    if config.target_length == 0 {
        return Err(TrustformersError::invalid_input_simple(
            "Target length must be greater than 0".to_string(),
        ));
    }

    if config.trigger_threshold <= config.target_length {
        return Err(TrustformersError::invalid_input_simple(
            "Trigger threshold must be greater than target length".to_string(),
        ));
    }

    Ok(())
}

/// Create a default context summarizer
pub fn create_default_summarizer() -> ContextSummarizer {
    ContextSummarizer::new(SummarizationConfig::default())
}

/// Create a high-compression summarizer for memory-constrained environments
pub fn create_high_compression_summarizer() -> ContextSummarizer {
    let mut config = SummarizationConfig::default();
    config.target_length = 100;
    config.trigger_threshold = 500;
    config.strategy = SummarizationStrategy::Hybrid;

    ContextSummarizer::new(config)
}

/// Create a topic-focused extractive summarizer
pub fn create_extractive_summarizer() -> ContextSummarizer {
    let mut config = SummarizationConfig::default();
    config.strategy = SummarizationStrategy::Extractive;
    config.target_length = 300;

    ContextSummarizer::new(config)
}

/// Create an abstractive summarizer for detailed overviews
pub fn create_abstractive_summarizer() -> ContextSummarizer {
    let mut config = SummarizationConfig::default();
    config.strategy = SummarizationStrategy::Abstractive;
    config.target_length = 250;

    ContextSummarizer::new(config)
}

#[cfg(test)]
mod tests {
    use super::super::types::ConversationMetadata;
    use super::*;
    use chrono::Utc;

    fn create_test_turn(role: ConversationRole, content: &str) -> ConversationTurn {
        ConversationTurn {
            role,
            content: content.to_string(),
            timestamp: Utc::now(),
            metadata: None,
            token_count: content.len() / 4, // Simple estimation
        }
    }

    fn create_test_turn_with_metadata(
        role: ConversationRole,
        content: &str,
        topics: Vec<String>,
    ) -> ConversationTurn {
        ConversationTurn {
            role,
            content: content.to_string(),
            timestamp: Utc::now(),
            metadata: Some(ConversationMetadata {
                sentiment: Some("neutral".to_string()),
                intent: Some("statement".to_string()),
                confidence: 0.8,
                topics,
                safety_flags: Vec::new(),
                entities: Vec::new(),
                quality_score: 0.8,
                engagement_level: EngagementLevel::Medium,
                reasoning_type: None,
            }),
            token_count: content.len() / 4,
        }
    }

    #[test]
    fn test_context_summarizer_creation() {
        let config = SummarizationConfig::default();
        let summarizer = ContextSummarizer::new(config.clone());

        assert_eq!(summarizer.config.strategy, config.strategy);
        assert_eq!(summarizer.config.target_length, config.target_length);
    }

    #[test]
    fn test_legacy_constructor() {
        let summarizer = ContextSummarizer::with_strategy(SummarizationStrategy::Extractive, 200);
        assert_eq!(
            summarizer.config.strategy,
            SummarizationStrategy::Extractive
        );
        assert_eq!(summarizer.config.target_length, 200);
    }

    #[test]
    fn test_empty_conversation_summarization() {
        let mut summarizer = create_default_summarizer();
        let result = summarizer.summarize_context_enhanced(&[]).unwrap();

        assert!(result.summary.is_empty());
        assert_eq!(result.original_tokens, 0);
        assert_eq!(result.summary_tokens, 0);
        assert_eq!(result.compression_ratio, 1.0);
    }

    #[test]
    fn test_legacy_summarization() {
        let mut summarizer = create_default_summarizer();
        let turns = vec![
            create_test_turn(ConversationRole::User, "Hello!"),
            create_test_turn(ConversationRole::Assistant, "Hi there!"),
        ];

        let result = summarizer.summarize_context(&turns).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_short_conversation_no_summarization() {
        let mut summarizer = create_default_summarizer();
        let turns = vec![
            create_test_turn(ConversationRole::User, "Hello!"),
            create_test_turn(ConversationRole::Assistant, "Hi there!"),
        ];

        let result = summarizer.summarize_context_enhanced(&turns).unwrap();

        // Should not summarize if under target length
        assert!(result.summary.contains("Hello"));
        assert!(result.summary.contains("Hi there"));
        assert_eq!(result.compression_ratio, 1.0);
    }

    #[test]
    fn test_extractive_summarization() {
        let mut config = SummarizationConfig::default();
        config.strategy = SummarizationStrategy::Extractive;
        config.target_length = 20; // Force summarization
        config.trigger_threshold = 10;

        let mut summarizer = ContextSummarizer::new(config);

        let turns = vec![
            create_test_turn_with_metadata(
                ConversationRole::User,
                "I really need help with my Python programming project. It's about machine learning algorithms.",
                vec!["technology".to_string(), "programming".to_string()]
            ),
            create_test_turn(
                ConversationRole::Assistant,
                "I'd be happy to help you with your Python machine learning project. What specific aspect are you working on?"
            ),
            create_test_turn(
                ConversationRole::User,
                "I'm trying to implement a neural network from scratch but I'm getting confused about backpropagation."
            ),
        ];

        let result = summarizer.summarize_context_enhanced(&turns).unwrap();

        assert!(!result.summary.is_empty());
        assert!(result.compression_ratio < 1.0);
        assert!(result.quality_score > 0.0);
        assert_eq!(result.strategy_used, SummarizationStrategy::Extractive);
    }

    #[test]
    fn test_abstractive_summarization() {
        let mut config = SummarizationConfig::default();
        config.strategy = SummarizationStrategy::Abstractive;
        config.target_length = 30;
        config.trigger_threshold = 10;

        let mut summarizer = ContextSummarizer::new(config);

        let turns = vec![
            create_test_turn_with_metadata(
                ConversationRole::User,
                "What's the weather like today?",
                vec!["weather".to_string()]
            ),
            create_test_turn(
                ConversationRole::Assistant,
                "I don't have access to current weather data, but I can help you find weather information."
            ),
            create_test_turn_with_metadata(
                ConversationRole::User,
                "How can I check the weather?",
                vec!["weather".to_string()]
            ),
        ];

        let result = summarizer.summarize_context_enhanced(&turns).unwrap();

        assert!(!result.summary.is_empty());
        assert!(result.summary.contains("Conversation summary"));
        assert_eq!(result.strategy_used, SummarizationStrategy::Abstractive);
    }

    #[test]
    fn test_hybrid_summarization() {
        let mut config = SummarizationConfig::default();
        config.strategy = SummarizationStrategy::Hybrid;
        config.target_length = 40;
        config.trigger_threshold = 10;

        let mut summarizer = ContextSummarizer::new(config);

        let turns = vec![
            create_test_turn(
                ConversationRole::User,
                "I'm interested in learning about artificial intelligence and machine learning.",
            ),
            create_test_turn(
                ConversationRole::Assistant,
                "AI and ML are fascinating fields! What specific area interests you most?",
            ),
            create_test_turn(
                ConversationRole::User,
                "I'd like to understand neural networks and deep learning applications.",
            ),
        ];

        let result = summarizer.summarize_context_enhanced(&turns).unwrap();

        assert!(!result.summary.is_empty());
        assert!(result.compression_ratio < 1.0);
        assert_eq!(result.strategy_used, SummarizationStrategy::Hybrid);
    }

    #[test]
    fn test_sentence_importance_scoring() {
        let summarizer = create_default_summarizer();
        let turn = create_test_turn(
            ConversationRole::User,
            "I really need help with this important question.",
        );

        let score = summarizer.calculate_sentence_importance(
            "I really need help with this important question.",
            &turn,
            0,
        );

        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_personal_info_detection() {
        let summarizer = create_default_summarizer();

        assert!(summarizer.contains_personal_info("i am john and i work as a developer"));
        assert!(summarizer.contains_personal_info("my name is alice"));
        assert!(!summarizer.contains_personal_info("the weather is nice today"));
    }

    #[test]
    fn test_emotional_content_detection() {
        let summarizer = create_default_summarizer();

        assert!(summarizer.contains_emotional_content("i love this amazing product"));
        assert!(summarizer.contains_emotional_content("i feel frustrated about this"));
        assert!(!summarizer.contains_emotional_content("the technical specifications are correct"));
    }

    #[test]
    fn test_token_counting() {
        let summarizer = create_default_summarizer();

        let short_text = "Hello world";
        let long_text = "This is a much longer text with many more words and characters";

        let short_tokens = summarizer.count_tokens(short_text);
        let long_tokens = summarizer.count_tokens(long_text);

        assert!(long_tokens > short_tokens);
        assert!(short_tokens > 0);
    }

    #[test]
    fn test_topic_extraction() {
        let summarizer = create_default_summarizer();

        let tech_sentence = "I need help with programming and software development";
        let food_sentence = "Let's go to a restaurant for dinner";
        let mixed_sentence = "I work in tech but love cooking food";

        let tech_topics = summarizer.extract_sentence_topics(tech_sentence);
        let food_topics = summarizer.extract_sentence_topics(food_sentence);
        let mixed_topics = summarizer.extract_sentence_topics(mixed_sentence);

        assert!(tech_topics.contains(&"technology".to_string()));
        assert!(food_topics.contains(&"food".to_string()));
        assert!(mixed_topics.len() >= 2);
    }

    #[test]
    fn test_quality_assessment() {
        let summarizer = create_default_summarizer();
        let turns = vec![create_test_turn_with_metadata(
            ConversationRole::User,
            "What is machine learning?",
            vec!["technology".to_string()],
        )];

        let good_summary = "User asked about machine learning technology";
        let assessment = summarizer.assess_summary_quality(good_summary, &turns, 0.5);

        assert!(assessment.quality_score > 0.0);
        assert!(assessment.confidence > 0.0);
    }

    #[test]
    fn test_configuration_validation() {
        let mut config = SummarizationConfig::default();
        assert!(validate_summarization_config(&config).is_ok());

        config.target_length = 0;
        assert!(validate_summarization_config(&config).is_err());

        config.target_length = 100;
        config.trigger_threshold = 50;
        assert!(validate_summarization_config(&config).is_err());
    }

    #[test]
    fn test_specialized_summarizers() {
        let high_compression = create_high_compression_summarizer();
        let extractive = create_extractive_summarizer();
        let abstractive = create_abstractive_summarizer();

        assert_eq!(high_compression.config.target_length, 100);
        assert_eq!(
            extractive.config.strategy,
            SummarizationStrategy::Extractive
        );
        assert_eq!(
            abstractive.config.strategy,
            SummarizationStrategy::Abstractive
        );
    }

    #[test]
    fn test_topic_clustering() {
        let summarizer = create_default_summarizer();

        let sentences = vec![
            SentenceScore {
                sentence: "I love programming in Python".to_string(),
                score: 0.8,
                position: 0,
                turn_index: 0,
                topics: vec!["technology".to_string()],
                entities: vec![],
                speaker_role: ConversationRole::User,
            },
            SentenceScore {
                sentence: "Let's discuss machine learning algorithms".to_string(),
                score: 0.9,
                position: 1,
                turn_index: 0,
                topics: vec!["technology".to_string()],
                entities: vec![],
                speaker_role: ConversationRole::User,
            },
            SentenceScore {
                sentence: "I had pizza for dinner".to_string(),
                score: 0.3,
                position: 2,
                turn_index: 1,
                topics: vec!["food".to_string()],
                entities: vec![],
                speaker_role: ConversationRole::User,
            },
        ];

        let clusters = summarizer.cluster_by_topics(&sentences);

        assert!(clusters.len() >= 2); // Should have at least technology and food clusters

        let tech_cluster = clusters.iter().find(|c| c.topic == "technology");
        assert!(tech_cluster.is_some());
        assert_eq!(tech_cluster.unwrap().sentences.len(), 2);
    }

    #[test]
    fn test_conversation_flow_analysis() {
        let summarizer = create_default_summarizer();

        let question_heavy_turns = vec![
            create_test_turn(ConversationRole::User, "What is AI?"),
            create_test_turn(
                ConversationRole::Assistant,
                "AI is artificial intelligence.",
            ),
            create_test_turn(ConversationRole::User, "How does it work?"),
            create_test_turn(ConversationRole::Assistant, "It uses algorithms."),
            create_test_turn(ConversationRole::User, "Can you give examples?"),
        ];

        let flow_analysis = summarizer.analyze_conversation_flow(&question_heavy_turns);
        assert!(flow_analysis.is_some());
        assert!(flow_analysis.unwrap().contains("inquiry-heavy"));

        let statement_heavy_turns = vec![
            create_test_turn(ConversationRole::User, "I work in tech."),
            create_test_turn(ConversationRole::Assistant, "That's interesting."),
            create_test_turn(ConversationRole::User, "I develop software applications."),
        ];

        let flow_analysis2 = summarizer.analyze_conversation_flow(&statement_heavy_turns);
        assert!(flow_analysis2.is_some());
        assert!(flow_analysis2.unwrap().contains("informational"));
    }

    #[test]
    fn test_legacy_compatibility() {
        let summarizer = create_default_summarizer();
        let turns = vec![
            create_test_turn_with_metadata(
                ConversationRole::User,
                "Let's talk about technology and programming",
                vec!["technology".to_string()],
            ),
            create_test_turn(
                ConversationRole::Assistant,
                "Sure, what would you like to know?",
            ),
        ];

        // Test topic-focused summary
        let topic_summary = summarizer.summarize_by_topic(&turns, "technology").unwrap();
        assert!(!topic_summary.is_empty());

        // Test hierarchical summary
        let hierarchical = summarizer.hierarchical_summary(&turns).unwrap();
        assert!(!hierarchical.overall_summary.is_empty());
        assert_eq!(hierarchical.total_turns, 2);

        // Test constrained summary
        let constrained = summarizer.constrained_summary(&turns, 100, true, true).unwrap();
        assert!(!constrained.summary.is_empty());
        assert!(constrained.topics.is_some());
        assert!(constrained.sentiment_analysis.is_some());
    }
}
