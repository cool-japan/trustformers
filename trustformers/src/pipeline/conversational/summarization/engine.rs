//! The `ContextSummarizer` engine: strategy dispatch, extractive/abstractive/hybrid summarization, quality assessment, and token/context management.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use crate::pipeline::conversational::types::{
    ConversationRole, ConversationTurn, EngagementLevel, ReasoningType, SummarizationConfig,
    SummarizationStrategy,
};
use regex::Regex;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::types::{
    ConstrainedSummary, ConversationSegment, HierarchicalSummary, ImportanceWeights,
    QualityAssessment, QualityThresholds, SentenceScore, SentimentAnalysis, SummarizationResult,
    TopicCluster,
};

/// Advanced context summarization component for conversation compression
pub struct ContextSummarizer {
    /// Summarization strategy configuration
    pub config: SummarizationConfig,
    /// Token counting function for accurate estimation
    pub token_counter: Option<Arc<dyn Fn(&str) -> usize + Send + Sync>>,
    /// Cache for frequently used regex patterns
    pub(super) regex_cache: HashMap<String, Regex>,
    /// Importance scoring weights
    pub(super) importance_weights: ImportanceWeights,
    /// Quality assessment thresholds
    pub(super) quality_thresholds: QualityThresholds,
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
        self.warn_if_below_quality_thresholds(&quality_assessment, compression_ratio);

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
    pub(super) fn extractive_summary(&mut self, turns: &[ConversationTurn]) -> Result<String> {
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
            tracing::trace!(
                topic = %cluster.topic,
                cluster_score = cluster.cluster_score,
                representative = cluster.representative_sentence.as_deref().unwrap_or(""),
                "considering topic cluster for extractive summary"
            );
            for sentence_score in cluster.sentences {
                let sentence_tokens = self.count_tokens(&sentence_score.sentence);
                if current_tokens + sentence_tokens <= target_tokens {
                    tracing::trace!(
                        entity_count = sentence_score.entities.len(),
                        score = sentence_score.score,
                        "selecting sentence for extractive summary"
                    );
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
    pub(super) fn abstractive_summary(&self, turns: &[ConversationTurn]) -> Result<String> {
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
    pub(super) fn hybrid_summary(&mut self, turns: &[ConversationTurn]) -> Result<String> {
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
    pub(super) fn score_sentences(&self, turns: &[ConversationTurn]) -> Result<Vec<SentenceScore>> {
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
    pub(super) fn calculate_sentence_importance(
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
    pub(super) fn contains_personal_info(&self, sentence: &str) -> bool {
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
    pub(super) fn contains_emotional_content(&self, sentence: &str) -> bool {
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
    pub(super) fn calculate_length_factor(&self, sentence: &str) -> f32 {
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
    pub(super) fn cluster_by_topics(&self, sentences: &[SentenceScore]) -> Vec<TopicCluster> {
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
    pub(super) fn build_coherent_summary(&self, sentences: Vec<SentenceScore>) -> Result<String> {
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
    pub(super) fn extract_key_topics(
        &self,
        turns: &[ConversationTurn],
        limit: usize,
    ) -> Vec<String> {
        let mut topic_counts: HashMap<String, usize> = HashMap::new();

        for turn in turns {
            if let Some(metadata) = &turn.metadata {
                for topic in &metadata.topics {
                    *topic_counts.entry(topic.clone()).or_insert(0) += 1;
                }
            }
        }

        let mut topics: Vec<_> = topic_counts.into_iter().collect();
        topics.sort_by_key(|item| std::cmp::Reverse(item.1));

        topics.into_iter().take(limit).map(|(topic, _)| topic).collect()
    }

    /// Extract key entities from conversation
    pub(super) fn extract_key_entities(
        &self,
        turns: &[ConversationTurn],
        limit: usize,
    ) -> Vec<String> {
        let mut entity_counts: HashMap<String, usize> = HashMap::new();

        for turn in turns {
            if let Some(metadata) = &turn.metadata {
                for entity in &metadata.entities {
                    *entity_counts.entry(entity.text.clone()).or_insert(0) += 1;
                }
            }
        }

        let mut entities: Vec<_> = entity_counts.into_iter().collect();
        entities.sort_by_key(|item| std::cmp::Reverse(item.1));

        entities.into_iter().take(limit).map(|(entity, _)| entity).collect()
    }

    /// Analyze conversation flow for abstractive summary
    pub(super) fn analyze_conversation_flow(&self, turns: &[ConversationTurn]) -> Option<String> {
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
    pub(super) fn analyze_emotional_arc(&self, turns: &[ConversationTurn]) -> Option<String> {
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
        let final_sentiment = sentiment_progression.last()?;

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
    pub(super) fn extract_important_exchanges(
        &self,
        turns: &[ConversationTurn],
        limit: usize,
    ) -> Vec<String> {
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
    pub(super) fn extract_question_topic(&self, content: &str) -> String {
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
    pub(super) fn extract_response_summary(&self, content: &str) -> String {
        let words: Vec<&str> = content.split_whitespace().take(10).collect();
        if words.len() < 10 {
            content.to_string()
        } else {
            format!("{}...", words.join(" "))
        }
    }

    /// Split text into sentences
    pub(super) fn split_into_sentences(&self, text: &str) -> Vec<String> {
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
    pub(super) fn extract_sentence_topics(&self, sentence: &str) -> Vec<String> {
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
    pub(super) fn extract_sentence_entities(&self, sentence: &str) -> Vec<String> {
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
    pub(super) fn build_full_context(&self, turns: &[ConversationTurn]) -> String {
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
    pub(super) fn calculate_total_tokens(&self, turns: &[ConversationTurn]) -> usize {
        turns.iter().map(|turn| self.count_tokens(&turn.content)).sum()
    }

    /// Count tokens in text
    pub(super) fn count_tokens(&self, text: &str) -> usize {
        if let Some(ref counter) = self.token_counter {
            counter(text)
        } else {
            // Fallback estimation: ~4 characters per token
            text.len() / 4
        }
    }

    /// Trim summary to target length
    pub(super) fn trim_to_target_length(&self, summary: String) -> Result<String> {
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
    pub(super) fn truncate_at_word_boundary(&self, text: &str, max_chars: usize) -> Option<String> {
        if text.len() <= max_chars {
            return Some(text.to_string());
        }

        let truncated = &text[..max_chars];
        truncated
            .rfind(' ')
            .map(|last_space| format!("{}...", &truncated[..last_space]))
    }

    /// Extract all topics from conversation
    pub(super) fn extract_all_topics(&self, turns: &[ConversationTurn]) -> Vec<String> {
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
    pub(super) fn extract_all_entities(&self, turns: &[ConversationTurn]) -> Vec<String> {
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
    pub(super) fn extract_preserved_topics(
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
    pub(super) fn extract_preserved_entities(
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
    pub(super) fn assess_summary_quality(
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
            coherence_score,
        }
    }

    /// Logs a warning for every dimension of `assessment`/`compression_ratio`
    /// that falls outside `self.quality_thresholds`. This doesn't change
    /// `summarize`'s return value (a poor-quality summary is still a real,
    /// honestly-labeled result, not an error), but it makes the configured
    /// thresholds — previously dead configuration nobody consulted —
    /// actually observable.
    pub(super) fn warn_if_below_quality_thresholds(
        &self,
        assessment: &QualityAssessment,
        compression_ratio: f32,
    ) {
        let thresholds = &self.quality_thresholds;
        if assessment.quality_score < thresholds.min_quality_score {
            tracing::warn!(
                quality_score = assessment.quality_score,
                min_quality_score = thresholds.min_quality_score,
                "summarization quality below configured threshold"
            );
        }
        if compression_ratio < thresholds.min_compression_ratio {
            tracing::warn!(
                compression_ratio,
                min_compression_ratio = thresholds.min_compression_ratio,
                "summarization compression ratio below configured threshold"
            );
        }
        if assessment.coherence_score < thresholds.min_coherence_score {
            tracing::warn!(
                coherence_score = assessment.coherence_score,
                min_coherence_score = thresholds.min_coherence_score,
                "summarization coherence below configured threshold"
            );
        }
        // Fraction of tokens removed by compression, as a proxy for how much
        // content was cut; not a semantic information-loss measure.
        let information_loss = 1.0 - compression_ratio.clamp(0.0, 1.0);
        if information_loss > thresholds.max_information_loss {
            tracing::warn!(
                information_loss,
                max_information_loss = thresholds.max_information_loss,
                "summarization information loss above configured threshold"
            );
        }
    }

    /// Assess summary coherence
    pub(super) fn assess_coherence(&self, summary: &str) -> f32 {
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
    pub(super) fn assess_readability(&self, summary: &str) -> f32 {
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
    pub(super) fn extract_main_topics(&self, turns: &[ConversationTurn]) -> Vec<String> {
        let mut topic_counts = HashMap::new();

        for turn in turns {
            if let Some(metadata) = &turn.metadata {
                for topic in &metadata.topics {
                    *topic_counts.entry(topic.clone()).or_insert(0) += 1;
                }
            }
        }

        let mut topics: Vec<_> = topic_counts.into_iter().collect();
        topics.sort_by_key(|item| std::cmp::Reverse(item.1));

        topics.into_iter()
            .take(5) // Top 5 topics
            .map(|(topic, _)| topic)
            .collect()
    }

    /// Extract topics from a conversation segment (legacy compatibility)
    pub(super) fn extract_segment_topics(&self, turns: &[ConversationTurn]) -> Vec<String> {
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
    pub(super) fn analyze_overall_sentiment(
        &self,
        turns: &[ConversationTurn],
    ) -> SentimentAnalysis {
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
