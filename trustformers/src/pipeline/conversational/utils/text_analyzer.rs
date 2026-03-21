//! Text analysis utilities for content understanding.
//!
//! This module provides comprehensive text analysis capabilities including
//! sentiment analysis, intent classification, topic extraction, entity recognition,
//! quality assessment, and safety checks.

use regex::Regex;
use std::collections::HashSet;

use super::super::types::{EngagementLevel, EntityMention, ReasoningType};

/// Text analysis utilities for content understanding
pub struct TextAnalyzer;

impl TextAnalyzer {
    /// Analyze sentiment of text content
    pub fn analyze_sentiment(content: &str) -> Option<String> {
        let positive_words = [
            "good",
            "great",
            "excellent",
            "happy",
            "pleased",
            "wonderful",
            "amazing",
            "fantastic",
            "brilliant",
            "awesome",
            "love",
            "like",
            "enjoy",
            "appreciate",
            "grateful",
            "thankful",
            "positive",
            "optimistic",
            "excited",
            "thrilled",
        ];

        let negative_words = [
            "bad",
            "terrible",
            "awful",
            "sad",
            "angry",
            "frustrated",
            "disappointed",
            "hate",
            "dislike",
            "horrible",
            "disgusting",
            "annoying",
            "upset",
            "worried",
            "anxious",
            "depressed",
            "negative",
            "pessimistic",
            "miserable",
            "furious",
        ];

        let neutral_words = [
            "okay", "fine", "alright", "average", "normal", "standard", "regular", "moderate",
            "typical", "ordinary", "usual", "common",
        ];

        let content_lower = content.to_lowercase();

        let pos_count = positive_words.iter().filter(|word| content_lower.contains(*word)).count();

        let neg_count = negative_words.iter().filter(|word| content_lower.contains(*word)).count();

        let neu_count = neutral_words.iter().filter(|word| content_lower.contains(*word)).count();

        if pos_count > neg_count && pos_count > neu_count {
            Some("positive".to_string())
        } else if neg_count > pos_count && neg_count > neu_count {
            Some("negative".to_string())
        } else {
            Some("neutral".to_string())
        }
    }

    /// Classify intent of the message
    pub fn classify_intent(content: &str) -> Option<String> {
        let content_lower = content.to_lowercase();

        // Question patterns
        if content.contains('?')
            || content_lower.starts_with("what")
            || content_lower.starts_with("how")
            || content_lower.starts_with("why")
            || content_lower.starts_with("when")
            || content_lower.starts_with("where")
            || content_lower.starts_with("who")
            || content_lower.starts_with("which")
        {
            return Some("question".to_string());
        }

        // Request patterns
        if [
            "please",
            "can you",
            "could you",
            "would you",
            "help me",
            "assist me",
        ]
        .iter()
        .any(|&pattern| content_lower.contains(pattern))
        {
            return Some("request".to_string());
        }

        // Gratitude patterns
        if ["thank", "thanks", "appreciate", "grateful"]
            .iter()
            .any(|&pattern| content_lower.contains(pattern))
        {
            return Some("gratitude".to_string());
        }

        // Greeting patterns
        if [
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
        ]
        .iter()
        .any(|&pattern| content_lower.contains(pattern))
        {
            return Some("greeting".to_string());
        }

        // Farewell patterns
        if ["goodbye", "bye", "see you", "farewell", "take care"]
            .iter()
            .any(|&pattern| content_lower.contains(pattern))
        {
            return Some("farewell".to_string());
        }

        // Help seeking patterns
        if ["help", "assist", "support", "guidance", "advice"]
            .iter()
            .any(|&pattern| content_lower.contains(pattern))
        {
            return Some("help_seeking".to_string());
        }

        // Complaint patterns
        if ["complain", "issue", "problem", "trouble", "error", "wrong"]
            .iter()
            .any(|&pattern| content_lower.contains(pattern))
        {
            return Some("complaint".to_string());
        }

        // Information sharing
        if ["i think", "i believe", "in my opinion", "i feel", "i know"]
            .iter()
            .any(|&pattern| content_lower.contains(pattern))
        {
            return Some("information_sharing".to_string());
        }

        Some("statement".to_string())
    }

    /// Extract topics from content
    pub fn extract_topics(content: &str) -> Vec<String> {
        let mut topics = Vec::new();
        let content_lower = content.to_lowercase();

        let topic_keywords = [
            (
                "technology",
                &[
                    "computer",
                    "software",
                    "tech",
                    "ai",
                    "programming",
                    "code",
                    "algorithm",
                    "data",
                    "internet",
                    "web",
                    "mobile",
                    "app",
                ] as &[&str],
            ),
            (
                "sports",
                &[
                    "football",
                    "basketball",
                    "soccer",
                    "tennis",
                    "game",
                    "match",
                    "team",
                    "player",
                    "score",
                    "league",
                    "championship",
                ],
            ),
            (
                "food",
                &[
                    "restaurant",
                    "cooking",
                    "recipe",
                    "eat",
                    "meal",
                    "dish",
                    "cuisine",
                    "chef",
                    "ingredient",
                    "flavor",
                    "taste",
                ],
            ),
            (
                "travel",
                &[
                    "trip",
                    "vacation",
                    "visit",
                    "country",
                    "hotel",
                    "flight",
                    "airport",
                    "tourism",
                    "destination",
                    "journey",
                ],
            ),
            (
                "work",
                &[
                    "job",
                    "career",
                    "office",
                    "meeting",
                    "project",
                    "company",
                    "business",
                    "colleague",
                    "manager",
                    "salary",
                ],
            ),
            (
                "health",
                &[
                    "doctor",
                    "medicine",
                    "exercise",
                    "wellness",
                    "fitness",
                    "hospital",
                    "treatment",
                    "therapy",
                    "diet",
                    "nutrition",
                ],
            ),
            (
                "education",
                &[
                    "school",
                    "university",
                    "student",
                    "teacher",
                    "learn",
                    "study",
                    "course",
                    "degree",
                    "education",
                    "knowledge",
                ],
            ),
            (
                "entertainment",
                &[
                    "movie",
                    "music",
                    "book",
                    "show",
                    "concert",
                    "theater",
                    "film",
                    "song",
                    "artist",
                    "performance",
                ],
            ),
            (
                "finance",
                &[
                    "money",
                    "bank",
                    "investment",
                    "stock",
                    "financial",
                    "economy",
                    "budget",
                    "savings",
                    "loan",
                    "credit",
                ],
            ),
            (
                "science",
                &[
                    "research",
                    "experiment",
                    "discovery",
                    "theory",
                    "physics",
                    "chemistry",
                    "biology",
                    "mathematics",
                    "scientific",
                ],
            ),
            (
                "politics",
                &[
                    "government",
                    "election",
                    "policy",
                    "political",
                    "democracy",
                    "vote",
                    "politician",
                    "law",
                    "regulation",
                ],
            ),
            (
                "family",
                &[
                    "family",
                    "parent",
                    "child",
                    "sibling",
                    "relative",
                    "marriage",
                    "relationship",
                    "home",
                    "domestic",
                ],
            ),
        ];

        for (topic, keywords) in topic_keywords {
            if keywords.iter().any(|keyword| content_lower.contains(keyword)) {
                topics.push(topic.to_string());
            }
        }

        topics
    }

    /// Extract named entities from text
    pub fn extract_entities(content: &str) -> Vec<EntityMention> {
        let mut entities = Vec::new();

        // Define regex patterns for common entity types
        let patterns = [
            (r"\b[A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", "PERSON"),
            (r"\b\d{1,2}/\d{1,2}/\d{4}\b", "DATE"),
            (r"\b\d{4}-\d{2}-\d{2}\b", "DATE"),
            (
                r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
                "DATE",
            ),
            (r"\$\d+(?:\.\d{2})?\b", "MONEY"),
            (
                r"\b\d+(?:\.\d+)?\s*(?:dollars?|euros?|pounds?|yen)\b",
                "MONEY",
            ),
            (r"\b\d{3}-\d{3}-\d{4}\b", "PHONE"),
            (
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "EMAIL",
            ),
            (r"\bhttps?://[^\s]+\b", "URL"),
            (
                r"\b\d{1,5}\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln)\b",
                "ADDRESS",
            ),
            (
                r"\b[A-Z][a-z]+\s+(?:University|College|School|Hospital|Corporation|Corp|Inc|LLC)\b",
                "ORGANIZATION",
            ),
        ];

        for (pattern, entity_type) in patterns {
            if let Ok(regex) = Regex::new(pattern) {
                for mat in regex.find_iter(content) {
                    entities.push(EntityMention {
                        text: mat.as_str().to_string(),
                        entity_type: entity_type.to_string(),
                        confidence: 0.8,
                        start_pos: mat.start(),
                        end_pos: mat.end(),
                    });
                }
            }
        }

        entities
    }

    /// Calculate confidence score for content
    pub fn calculate_confidence(content: &str) -> f32 {
        let mut confidence: f32 = 0.7;

        // Length factor (longer content generally more confident)
        if content.len() > 20 {
            confidence += 0.1;
        }
        if content.len() > 100 {
            confidence += 0.1;
        }

        // Uncertainty indicators
        let uncertainty_words = [
            "maybe", "perhaps", "might", "possibly", "probably", "seems", "appears", "could be",
        ];
        if !uncertainty_words.iter().any(|&word| content.to_lowercase().contains(word)) {
            confidence += 0.1;
        }

        // Confidence indicators
        let confidence_words = [
            "definitely",
            "certainly",
            "absolutely",
            "clearly",
            "obviously",
            "undoubtedly",
        ];
        if confidence_words.iter().any(|&word| content.to_lowercase().contains(word)) {
            confidence += 0.1;
        }

        // Grammar and structure indicators
        if content.chars().any(|c| c.is_uppercase()) {
            confidence += 0.05;
        }

        if [".", "!", "?"].iter().any(|&punct| content.contains(punct)) {
            confidence += 0.05;
        }

        confidence.min(1.0)
    }

    /// Calculate quality score for content
    pub fn calculate_quality_score(content: &str) -> f32 {
        let mut score = 0.5;

        // Length factor
        let length = content.len();
        if (10..=1000).contains(&length) {
            score += 0.2;
        }

        // Grammar indicators
        if content.chars().any(|c| c.is_uppercase()) {
            score += 0.1;
        }

        if [".", "!", "?"].iter().any(|&punct| content.contains(punct)) {
            score += 0.1;
        }

        // Coherence indicators (no filler words)
        if !["uhh", "umm", "err", "uh", "um"]
            .iter()
            .any(|&filler| content.to_lowercase().contains(filler))
        {
            score += 0.1;
        }

        // Vocabulary diversity
        let words: HashSet<&str> = content.split_whitespace().collect();
        let unique_ratio = words.len() as f32 / content.split_whitespace().count().max(1) as f32;
        score += unique_ratio * 0.1;

        // Complete sentences
        let sentence_count = content.matches(['.', '!', '?']).count();
        if sentence_count > 0 {
            score += 0.1;
        }

        score.min(1.0)
    }

    /// Assess engagement level
    pub fn assess_engagement(content: &str) -> EngagementLevel {
        let content_lower = content.to_lowercase();

        // Count engagement indicators
        let engagement_indicators = content.matches(['!', '?']).count()
            + [
                "wow",
                "really",
                "interesting",
                "amazing",
                "fantastic",
                "incredible",
                "awesome",
            ]
            .iter()
            .map(|&word| content_lower.matches(word).count())
            .sum::<usize>()
            + if content_lower.contains("very") || content_lower.contains("extremely") {
                1
            } else {
                0
            }
            + if content.len() > 100 { 1 } else { 0 };

        // Assess emotional intensity
        let emotional_words = [
            "love",
            "hate",
            "excited",
            "thrilled",
            "devastated",
            "overjoyed",
        ];
        let emotional_intensity = emotional_words
            .iter()
            .map(|&word| content_lower.matches(word).count())
            .sum::<usize>();

        let total_score = engagement_indicators + emotional_intensity;

        match total_score {
            0..=1 => EngagementLevel::Low,
            2..=3 => EngagementLevel::Medium,
            4..=6 => EngagementLevel::High,
            _ => EngagementLevel::VeryHigh,
        }
    }

    /// Detect reasoning type in content
    pub fn detect_reasoning_type(content: &str) -> Option<ReasoningType> {
        let content_lower = content.to_lowercase();

        // Logical reasoning
        if [
            "because",
            "therefore",
            "thus",
            "consequently",
            "hence",
            "so",
            "since",
            "as a result",
        ]
        .iter()
        .any(|&pattern| content_lower.contains(pattern))
        {
            return Some(ReasoningType::Logical);
        }

        // Causal reasoning
        if [
            "causes",
            "leads to",
            "results in",
            "due to",
            "caused by",
            "effect of",
        ]
        .iter()
        .any(|&pattern| content_lower.contains(pattern))
        {
            return Some(ReasoningType::Causal);
        }

        // Analogical reasoning
        if [
            "like",
            "similar to",
            "analogous",
            "comparable",
            "just as",
            "in the same way",
        ]
        .iter()
        .any(|&pattern| content_lower.contains(pattern))
        {
            return Some(ReasoningType::Analogical);
        }

        // Mathematical reasoning
        if [
            "calculate",
            "equation",
            "formula",
            "math",
            "number",
            "statistics",
            "probability",
            "algorithm",
        ]
        .iter()
        .any(|&pattern| content_lower.contains(pattern))
        {
            return Some(ReasoningType::Mathematical);
        }

        // Emotional reasoning
        if [
            "feel",
            "emotion",
            "heart",
            "intuition",
            "gut feeling",
            "emotional",
        ]
        .iter()
        .any(|&pattern| content_lower.contains(pattern))
        {
            return Some(ReasoningType::Emotional);
        }

        // Creative reasoning
        if [
            "imagine",
            "creative",
            "innovative",
            "brainstorm",
            "think outside",
            "original",
        ]
        .iter()
        .any(|&pattern| content_lower.contains(pattern))
        {
            return Some(ReasoningType::Creative);
        }

        None
    }

    /// Detect safety issues in content
    pub fn detect_safety_issues(content: &str) -> Vec<String> {
        let mut flags = Vec::new();
        let content_lower = content.to_lowercase();

        let safety_patterns = [
            (
                "violence",
                &[
                    "kill", "hurt", "harm", "attack", "violence", "weapon", "fight", "murder",
                ] as &[&str],
            ),
            (
                "inappropriate",
                &[
                    "inappropriate",
                    "offensive",
                    "rude",
                    "insulting",
                    "harassment",
                ],
            ),
            (
                "personal_info",
                &[
                    "password",
                    "ssn",
                    "social security",
                    "credit card",
                    "bank account",
                    "phone number",
                ],
            ),
            (
                "hate_speech",
                &["hate", "racist", "sexist", "discrimination", "prejudice"],
            ),
            (
                "self_harm",
                &[
                    "suicide",
                    "self-harm",
                    "cut myself",
                    "kill myself",
                    "end it all",
                ],
            ),
            (
                "illegal",
                &["illegal", "drugs", "steal", "fraud", "scam", "criminal"],
            ),
            (
                "adult_content",
                &["sexual", "explicit", "pornographic", "adult content"],
            ),
        ];

        for (flag, patterns) in safety_patterns {
            if patterns.iter().any(|pattern| content_lower.contains(pattern)) {
                flags.push(flag.to_string());
            }
        }

        flags
    }
}
