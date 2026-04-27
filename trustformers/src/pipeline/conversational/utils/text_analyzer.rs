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

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Sentiment analysis tests ----

    #[test]
    fn test_sentiment_positive_text() {
        let sentiment = TextAnalyzer::analyze_sentiment("This is great and amazing work!");
        assert_eq!(
            sentiment.as_deref(),
            Some("positive"),
            "clearly positive text should detect positive sentiment"
        );
    }

    #[test]
    fn test_sentiment_negative_text() {
        let sentiment =
            TextAnalyzer::analyze_sentiment("I hate this terrible and awful experience.");
        assert_eq!(
            sentiment.as_deref(),
            Some("negative"),
            "clearly negative text should detect negative sentiment"
        );
    }

    #[test]
    fn test_sentiment_neutral_text() {
        let sentiment = TextAnalyzer::analyze_sentiment("okay fine normal standard");
        assert_eq!(
            sentiment.as_deref(),
            Some("neutral"),
            "neutral filler words should detect neutral sentiment"
        );
    }

    #[test]
    fn test_sentiment_empty_text_returns_some() {
        // Empty text: no matches → defaults to neutral
        let sentiment = TextAnalyzer::analyze_sentiment("");
        assert!(
            sentiment.is_some(),
            "empty text must still return Some variant"
        );
    }

    // ---- Intent classification tests ----

    #[test]
    fn test_intent_question_with_question_mark() {
        let intent = TextAnalyzer::classify_intent("What is the meaning of life?");
        assert_eq!(
            intent.as_deref(),
            Some("question"),
            "text with '?' should classify as question"
        );
    }

    #[test]
    fn test_intent_question_starts_with_how() {
        let intent = TextAnalyzer::classify_intent("How does this work?");
        assert_eq!(
            intent.as_deref(),
            Some("question"),
            "'how' prefix should classify as question"
        );
    }

    #[test]
    fn test_intent_request_contains_please() {
        let intent = TextAnalyzer::classify_intent("Please help me with this.");
        assert_eq!(
            intent.as_deref(),
            Some("request"),
            "'please' should classify as request"
        );
    }

    #[test]
    fn test_intent_gratitude_thanks() {
        let intent = TextAnalyzer::classify_intent("Thanks for your help.");
        assert_eq!(
            intent.as_deref(),
            Some("gratitude"),
            "'thanks' should classify as gratitude"
        );
    }

    #[test]
    fn test_intent_greeting_hello() {
        let intent = TextAnalyzer::classify_intent("Hello there!");
        assert_eq!(
            intent.as_deref(),
            Some("greeting"),
            "'hello' should classify as greeting"
        );
    }

    #[test]
    fn test_intent_farewell() {
        let intent = TextAnalyzer::classify_intent("Goodbye, see you later.");
        assert_eq!(
            intent.as_deref(),
            Some("farewell"),
            "farewell keywords should classify as farewell"
        );
    }

    #[test]
    fn test_intent_complaint_with_problem() {
        // Use a string that does not accidentally trigger earlier patterns (e.g. contains "hi" → greeting)
        let intent = TextAnalyzer::classify_intent("I am experiencing a problem.");
        assert_eq!(
            intent.as_deref(),
            Some("complaint"),
            "'problem' should classify as complaint"
        );
    }

    #[test]
    fn test_intent_statement_for_generic_text() {
        let intent = TextAnalyzer::classify_intent("The sky is blue.");
        assert_eq!(
            intent.as_deref(),
            Some("statement"),
            "generic sentence should classify as statement"
        );
    }

    // ---- Topic extraction tests ----

    #[test]
    fn test_extract_topics_technology() {
        let topics = TextAnalyzer::extract_topics("I love programming and software development.");
        assert!(
            topics.contains(&"technology".to_string()),
            "tech keywords should extract technology topic"
        );
    }

    #[test]
    fn test_extract_topics_health() {
        let topics = TextAnalyzer::extract_topics("I exercise daily for fitness and wellness.");
        assert!(
            topics.contains(&"health".to_string()),
            "health keywords should extract health topic"
        );
    }

    #[test]
    fn test_extract_topics_empty_text_returns_no_topics() {
        let topics = TextAnalyzer::extract_topics("xyz_obscure_none");
        assert!(topics.is_empty(), "obscure text should yield no topics");
    }

    // ---- Entity extraction tests ----

    #[test]
    fn test_extract_entities_finds_email() {
        let entities =
            TextAnalyzer::extract_entities("Contact me at user@example.com for details.");
        let emails: Vec<_> = entities.iter().filter(|e| e.entity_type == "EMAIL").collect();
        assert!(
            !emails.is_empty(),
            "email address should be extracted as EMAIL entity"
        );
    }

    #[test]
    fn test_extract_entities_finds_money() {
        let entities = TextAnalyzer::extract_entities("The price is $42.99 today.");
        let money: Vec<_> = entities.iter().filter(|e| e.entity_type == "MONEY").collect();
        assert!(
            !money.is_empty(),
            "dollar amount should be extracted as MONEY entity"
        );
    }

    #[test]
    fn test_extract_entities_finds_url() {
        let entities = TextAnalyzer::extract_entities("Visit https://example.com for more info.");
        let urls: Vec<_> = entities.iter().filter(|e| e.entity_type == "URL").collect();
        assert!(!urls.is_empty(), "URL should be extracted as URL entity");
    }

    // ---- Quality / confidence tests ----

    #[test]
    fn test_confidence_increases_with_length() {
        let short_conf = TextAnalyzer::calculate_confidence("hi");
        let long_conf = TextAnalyzer::calculate_confidence(
            "This is a much longer statement that contains more information.",
        );
        assert!(
            long_conf > short_conf,
            "longer text should yield higher confidence"
        );
    }

    #[test]
    fn test_quality_score_reasonable_text() {
        let score = TextAnalyzer::calculate_quality_score("This is a complete sentence.");
        assert!(score > 0.5, "well-formed sentence should score above 0.5");
    }

    // ---- Engagement assessment tests ----

    #[test]
    fn test_engagement_low_for_plain_text() {
        let level = TextAnalyzer::assess_engagement("hello");
        assert_eq!(
            level,
            EngagementLevel::Low,
            "minimal text should have low engagement"
        );
    }

    #[test]
    fn test_engagement_high_with_exclamations_and_keywords() {
        let level = TextAnalyzer::assess_engagement(
            "Wow! This is absolutely amazing and incredible!! I love it so much!",
        );
        assert!(
            matches!(level, EngagementLevel::High | EngagementLevel::VeryHigh),
            "enthusiastic text should have high engagement"
        );
    }

    // ---- Reasoning type detection tests ----

    #[test]
    fn test_detect_reasoning_logical() {
        let rt = TextAnalyzer::detect_reasoning_type("Because it rained, the ground is wet.");
        assert_eq!(
            rt,
            Some(ReasoningType::Logical),
            "'because' should detect logical reasoning"
        );
    }

    #[test]
    fn test_detect_reasoning_causal() {
        let rt = TextAnalyzer::detect_reasoning_type("Stress causes health problems.");
        assert_eq!(
            rt,
            Some(ReasoningType::Causal),
            "'causes' should detect causal reasoning"
        );
    }

    #[test]
    fn test_detect_reasoning_mathematical() {
        let rt = TextAnalyzer::detect_reasoning_type("Please calculate the equation.");
        assert_eq!(
            rt,
            Some(ReasoningType::Mathematical),
            "math keywords should detect mathematical reasoning"
        );
    }

    // ---- Safety detection tests ----

    #[test]
    fn test_safety_detects_violence_keywords() {
        let flags = TextAnalyzer::detect_safety_issues("Someone wants to hurt others.");
        assert!(
            flags.contains(&"violence".to_string()),
            "'hurt' should trigger violence flag"
        );
    }

    #[test]
    fn test_safety_clean_text_returns_empty() {
        let flags = TextAnalyzer::detect_safety_issues("I love sunny days and good food.");
        assert!(flags.is_empty(), "clean text should return no safety flags");
    }
}
