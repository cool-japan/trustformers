//! Main conversation analyzer implementation.
//!
//! This module contains the core ConversationAnalyzer that orchestrates
//! all analysis operations for conversational AI systems.

use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

use super::super::types::{ConversationState, ConversationTurn, EngagementLevel};
use super::types::{
    AnalysisPerformance, EnhancedAnalysisConfig, HealthAssessment, TurnAnalysisResult,
};
use crate::core::error::Result;

/// Main conversation analyzer with comprehensive analysis capabilities
pub struct ConversationAnalyzer {
    /// Analysis configuration
    config: EnhancedAnalysisConfig,
    /// Performance metrics tracking
    performance_metrics: Arc<RwLock<AnalysisPerformance>>,
    /// Analysis cache for optimization
    analysis_cache: Arc<RwLock<std::collections::HashMap<String, TurnAnalysisResult>>>,
}

impl ConversationAnalyzer {
    /// Create a new conversation analyzer
    pub fn new(config: EnhancedAnalysisConfig) -> Self {
        Self {
            config,
            performance_metrics: Arc::new(RwLock::new(AnalysisPerformance {
                total_time: std::time::Duration::from_secs(0),
                start_time: Instant::now(),
                turns_analyzed: 0,
                avg_time_per_turn: std::time::Duration::from_millis(0),
                memory_usage_mb: 0.0,
            })),
            analysis_cache: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Analyze a single conversation turn
    pub async fn analyze_turn(&self, turn: &ConversationTurn) -> Result<TurnAnalysisResult> {
        let start_time = Instant::now();

        // Check cache first
        let cache_key = format!(
            "{}-{}",
            turn.timestamp.timestamp_nanos_opt().unwrap_or(0),
            turn.content.len()
        );
        if let Some(cached_result) = self.analysis_cache.read().await.get(&cache_key) {
            return Ok(cached_result.clone());
        }

        let mut result = TurnAnalysisResult {
            quality_score: 0.7,
            engagement_level: EngagementLevel::Medium,
            sentiment: None,
            intent: None,
            topics: Vec::new(),
            confidence: 0.8,
            processing_time: std::time::Duration::from_millis(0),
        };

        // Perform analysis based on configuration
        if self.config.enable_linguistic_analysis {
            self.analyze_linguistic_features(&turn.content, &mut result);
        }

        // Analyze sentiment and intent
        result.sentiment = self.analyze_sentiment(&turn.content);
        result.intent = self.classify_intent(&turn.content);
        result.topics = self.extract_topics(&turn.content);

        // Calculate engagement level
        result.engagement_level = self.assess_engagement(&turn.content);

        // Calculate overall quality score
        result.quality_score = self.calculate_quality_score(turn, &result);

        // Record processing time
        result.processing_time = start_time.elapsed();

        // Cache the result
        self.analysis_cache.write().await.insert(cache_key, result.clone());

        // Update performance metrics
        self.update_performance_metrics(start_time.elapsed()).await;

        Ok(result)
    }

    /// Analyze entire conversation state
    pub async fn analyze_conversation(
        &self,
        state: &ConversationState,
    ) -> Result<HealthAssessment> {
        let start_time = Instant::now();

        let mut assessment = HealthAssessment {
            overall_score: 0.75,
            component_scores: super::types::DetailedHealthMetrics {
                coherence: 0.8,
                engagement: 0.7,
                safety: 1.0,
                responsiveness: 0.8,
                context_relevance: 0.75,
                emotional_balance: 0.7,
                information_density: 0.6,
            },
            issues: Vec::new(),
            trend: Some("stable".to_string()),
            recommendations: Vec::new(),
        };

        // Analyze recent turns for patterns
        if !state.turns.is_empty() {
            assessment.component_scores.coherence = self.calculate_coherence_score(&state.turns);
            assessment.component_scores.engagement = self.calculate_engagement_score(&state.turns);
            assessment.component_scores.safety = self.calculate_safety_score(&state.turns);
        }

        // Calculate overall score
        assessment.overall_score = (assessment.component_scores.coherence * 0.2
            + assessment.component_scores.engagement * 0.2
            + assessment.component_scores.safety * 0.3
            + assessment.component_scores.responsiveness * 0.15
            + assessment.component_scores.context_relevance * 0.15)
            .min(1.0);

        // Generate recommendations
        assessment.recommendations = self.generate_recommendations(&assessment);

        Ok(assessment)
    }

    /// Analyze linguistic features of content
    fn analyze_linguistic_features(&self, content: &str, result: &mut TurnAnalysisResult) {
        let words: Vec<&str> = content.split_whitespace().collect();
        let sentences: Vec<&str> = content.split(&['.', '!', '?']).collect();

        // Update quality score based on linguistic features
        if words.len() < 3 {
            result.quality_score *= 0.8; // Penalize very short responses
        }

        if sentences.len() > 1 {
            result.quality_score *= 1.1; // Reward structured responses
        }

        // Adjust confidence based on content complexity
        result.confidence = (result.confidence + (words.len().min(20) as f32 / 20.0)) / 2.0;
    }

    /// Analyze sentiment of content
    fn analyze_sentiment(&self, content: &str) -> Option<String> {
        let positive_words = ["good", "great", "excellent", "happy", "wonderful"];
        let negative_words = ["bad", "terrible", "sad", "angry", "awful"];

        let content_lower = content.to_lowercase();
        let pos_count = positive_words.iter().filter(|&w| content_lower.contains(w)).count();
        let neg_count = negative_words.iter().filter(|&w| content_lower.contains(w)).count();

        if pos_count > neg_count {
            Some("positive".to_string())
        } else if neg_count > pos_count {
            Some("negative".to_string())
        } else {
            Some("neutral".to_string())
        }
    }

    /// Classify intent of content
    fn classify_intent(&self, content: &str) -> Option<String> {
        let content_lower = content.to_lowercase();

        if content.contains('?')
            || content_lower.starts_with("what")
            || content_lower.starts_with("how")
        {
            Some("question".to_string())
        } else if ["please", "can you", "help"].iter().any(|&p| content_lower.contains(p)) {
            Some("request".to_string())
        } else if ["thank", "thanks"].iter().any(|&p| content_lower.contains(p)) {
            Some("gratitude".to_string())
        } else {
            Some("statement".to_string())
        }
    }

    /// Extract topics from content
    fn extract_topics(&self, content: &str) -> Vec<String> {
        let topic_keywords = [
            (
                "technology",
                &["computer", "software", "tech", "ai"] as &[&str],
            ),
            ("work", &["job", "career", "office", "business"]),
            ("health", &["doctor", "medicine", "exercise", "wellness"]),
        ];

        let content_lower = content.to_lowercase();
        let mut topics = Vec::new();

        for (topic, keywords) in topic_keywords {
            if keywords.iter().any(|&keyword| content_lower.contains(keyword)) {
                topics.push(topic.to_string());
            }
        }

        topics
    }

    /// Assess engagement level
    fn assess_engagement(&self, content: &str) -> EngagementLevel {
        let engagement_indicators = content.matches(['!', '?']).count()
            + if content.to_lowercase().contains("interesting") { 1 } else { 0 }
            + if content.len() > 100 { 1 } else { 0 };

        match engagement_indicators {
            0..=1 => EngagementLevel::Low,
            2..=3 => EngagementLevel::Medium,
            4..=6 => EngagementLevel::High,
            _ => EngagementLevel::VeryHigh,
        }
    }

    /// Calculate quality score for a turn
    fn calculate_quality_score(&self, turn: &ConversationTurn, result: &TurnAnalysisResult) -> f32 {
        let mut score = 0.5_f32;

        // Length factor
        let length = turn.content.len();
        if (10..=1000).contains(&length) {
            score += 0.2;
        }

        // Grammar indicators (simplified)
        if turn.content.chars().any(|c| c.is_uppercase()) {
            score += 0.1;
        }

        // Engagement bonus
        match result.engagement_level {
            EngagementLevel::High | EngagementLevel::VeryHigh => score += 0.2,
            EngagementLevel::Medium => score += 0.1,
            _ => {},
        }

        score.min(1.0)
    }

    /// Calculate coherence score for conversation turns
    fn calculate_coherence_score(&self, turns: &[ConversationTurn]) -> f32 {
        if turns.is_empty() {
            return 1.0;
        }

        // Simplified coherence calculation based on turn quality
        let avg_quality = turns
            .iter()
            .filter_map(|turn| turn.metadata.as_ref().map(|m| m.quality_score))
            .sum::<f32>()
            / turns.len().max(1) as f32;

        avg_quality
    }

    /// Calculate engagement score for conversation turns
    fn calculate_engagement_score(&self, turns: &[ConversationTurn]) -> f32 {
        if turns.is_empty() {
            return 1.0;
        }

        let recent_turns = if turns.len() > 5 { &turns[turns.len() - 5..] } else { turns };

        let high_engagement_count = recent_turns
            .iter()
            .filter_map(|turn| turn.metadata.as_ref())
            .filter(|metadata| {
                matches!(
                    metadata.engagement_level,
                    EngagementLevel::High | EngagementLevel::VeryHigh
                )
            })
            .count();

        high_engagement_count as f32 / recent_turns.len().max(1) as f32
    }

    /// Calculate safety score for conversation turns
    fn calculate_safety_score(&self, turns: &[ConversationTurn]) -> f32 {
        if turns.is_empty() {
            return 1.0;
        }

        let recent_turns = if turns.len() > 10 { &turns[turns.len() - 10..] } else { turns };

        let unsafe_count = recent_turns
            .iter()
            .filter(|turn| turn.metadata.as_ref().is_some_and(|m| !m.safety_flags.is_empty()))
            .count();

        1.0 - (unsafe_count as f32 / recent_turns.len().max(1) as f32)
    }

    /// Generate recommendations based on assessment
    fn generate_recommendations(&self, assessment: &HealthAssessment) -> Vec<String> {
        let mut recommendations = Vec::new();

        if assessment.component_scores.coherence < 0.5 {
            recommendations.push("Focus on clearer, more structured responses".to_string());
        }

        if assessment.component_scores.engagement < 0.3 {
            recommendations.push("Try asking engaging questions".to_string());
        }

        if assessment.component_scores.safety < 0.9 {
            recommendations.push("Review content for safety compliance".to_string());
        }

        recommendations
    }

    /// Update performance metrics
    async fn update_performance_metrics(&self, processing_time: std::time::Duration) {
        let mut metrics = self.performance_metrics.write().await;
        metrics.turns_analyzed += 1;
        metrics.total_time += processing_time;
        metrics.avg_time_per_turn = metrics.total_time / metrics.turns_analyzed.max(1) as u32;
    }

    /// Get current performance metrics
    pub async fn get_performance_metrics(&self) -> AnalysisPerformance {
        self.performance_metrics.read().await.clone()
    }
}

impl Default for EnhancedAnalysisConfig {
    fn default() -> Self {
        Self {
            enable_linguistic_analysis: true,
            enable_contextual_metrics: true,
            enable_health_assessment: true,
            enable_performance_tracking: true,
            safety_sensitivity: super::types::SafetySensitivity::Medium,
            quality_strictness: super::types::QualityStrictness::Standard,
            min_confidence_threshold: 0.7,
            max_analysis_time: std::time::Duration::from_millis(100),
            enable_real_time_monitoring: false,
            batch_size: 10,
        }
    }
}
