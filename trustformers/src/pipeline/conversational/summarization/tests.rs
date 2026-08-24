//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use crate::pipeline::conversational::types::{
    ConversationMetadata, ConversationRole, ConversationTurn, EngagementLevel, SummarizationConfig,
    SummarizationStrategy,
};

use super::*;

#[cfg(test)]
mod tests_2 {
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
        let result = summarizer.summarize_context_enhanced(&[]).expect("operation failed in test");

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

        let result = summarizer.summarize_context(&turns).expect("operation failed in test");
        assert!(!result.is_empty());
    }

    #[test]
    fn test_short_conversation_no_summarization() {
        let mut summarizer = create_default_summarizer();
        let turns = vec![
            create_test_turn(ConversationRole::User, "Hello!"),
            create_test_turn(ConversationRole::Assistant, "Hi there!"),
        ];

        let result =
            summarizer.summarize_context_enhanced(&turns).expect("operation failed in test");

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

        let result =
            summarizer.summarize_context_enhanced(&turns).expect("operation failed in test");

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

        let result =
            summarizer.summarize_context_enhanced(&turns).expect("operation failed in test");

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

        let result =
            summarizer.summarize_context_enhanced(&turns).expect("operation failed in test");

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
        assert_eq!(
            tech_cluster.expect("operation failed in test").sentences.len(),
            2
        );
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
        assert!(flow_analysis.expect("operation failed in test").contains("inquiry-heavy"));

        let statement_heavy_turns = vec![
            create_test_turn(ConversationRole::User, "I work in tech."),
            create_test_turn(ConversationRole::Assistant, "That's interesting."),
            create_test_turn(ConversationRole::User, "I develop software applications."),
        ];

        let flow_analysis2 = summarizer.analyze_conversation_flow(&statement_heavy_turns);
        assert!(flow_analysis2.is_some());
        assert!(flow_analysis2.expect("operation failed in test").contains("informational"));
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
        let topic_summary = summarizer
            .summarize_by_topic(&turns, "technology")
            .expect("operation failed in test");
        assert!(!topic_summary.is_empty());

        // Test hierarchical summary
        let hierarchical =
            summarizer.hierarchical_summary(&turns).expect("operation failed in test");
        assert!(!hierarchical.overall_summary.is_empty());
        assert_eq!(hierarchical.total_turns, 2);

        // Test constrained summary
        let constrained = summarizer
            .constrained_summary(&turns, 100, true, true)
            .expect("operation failed in test");
        assert!(!constrained.summary.is_empty());
        assert!(constrained.topics.is_some());
        assert!(constrained.sentiment_analysis.is_some());
    }
}
