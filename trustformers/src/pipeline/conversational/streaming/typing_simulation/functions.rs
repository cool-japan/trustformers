//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use std::time::{Duration, Instant};

use super::types::{
    PerformanceTracker, TypingCharacteristics, TypingEvent, TypingEventType, TypingPattern,
    TypingPatterns, TypingPersonality,
};
use super::types_3::TypingSimulator;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::conversational::streaming::types::{
        AdvancedStreamingConfig, ChunkMetadata, ChunkTiming, ChunkType, StreamChunk,
    };
    /// Simple LCG for deterministic test values (avoids rand/ndarray dependency)
    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
        fn next_u64(&mut self) -> u64 {
            self.state =
                self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.state
        }
        fn next_f32_unit(&mut self) -> f32 {
            (self.next_u64() >> 32) as f32 / u32::MAX as f32
        }
    }
    fn make_chunk(content: &str, complexity: f32) -> StreamChunk {
        StreamChunk {
            content: content.to_string(),
            index: 0,
            chunk_type: ChunkType::Content,
            timing: ChunkTiming::default(),
            metadata: ChunkMetadata::with_complexity(complexity),
        }
    }
    fn make_default_config() -> AdvancedStreamingConfig {
        AdvancedStreamingConfig::default()
    }
    #[test]
    fn test_typing_simulator_new() {
        let config = make_default_config();
        let sim = TypingSimulator::new(config);
        let metrics = sim.performance_tracker.get_metrics();
        assert_eq!(metrics.total_bursts_generated, 0);
        assert_eq!(metrics.total_content_processed, 0);
    }
    #[test]
    fn test_typing_simulator_with_personality() {
        let config = make_default_config();
        let personality = TypingPersonality::fast_typist();
        let sim = TypingSimulator::with_personality(config, personality);
        assert!(
            sim.personality.speed_multiplier > 1.0,
            "fast_typist should have speed_multiplier > 1.0"
        );
    }
    #[test]
    fn test_typing_delay_non_variable_returns_base_delay() {
        let mut config = make_default_config();
        config.variable_typing_speed = false;
        config.base_config.typing_delay_ms = 42;
        let sim = TypingSimulator::new(config);
        let chunk = make_chunk("hello world", 0.5);
        let delay = sim.calculate_typing_delay(&chunk);
        assert_eq!(
            delay.as_millis(),
            42,
            "non-variable speed should return exactly the base delay"
        );
    }
    #[test]
    fn test_typing_delay_variable_speed_positive() {
        let config = make_default_config();
        let sim = TypingSimulator::new(config);
        let chunk = make_chunk("hello world this is a test", 0.5);
        let delay = sim.calculate_typing_delay(&chunk);
        assert!(
            delay.as_millis() > 0,
            "variable speed typing delay should be positive"
        );
    }
    #[test]
    fn test_typing_delay_scales_with_content_length() {
        let config = make_default_config();
        let sim = TypingSimulator::new(config);
        let short_chunk = make_chunk("hi", 0.5);
        let long_chunk = make_chunk(
            "This is a much longer piece of text that should take more time to type",
            0.5,
        );
        let short_delay = sim.calculate_typing_delay(&short_chunk);
        let long_delay = sim.calculate_typing_delay(&long_chunk);
        assert!(
            long_delay > short_delay,
            "longer content should produce larger typing delay"
        );
    }
    #[test]
    fn test_typing_delay_high_complexity_increases_delay() {
        let config = make_default_config();
        let sim = TypingSimulator::new(config);
        let low_chunk = make_chunk("simple test content", 0.1);
        let high_chunk = make_chunk("simple test content", 0.9);
        let low_delay = sim.calculate_typing_delay(&low_chunk);
        let high_delay = sim.calculate_typing_delay(&high_chunk);
        assert!(
            high_delay >= low_delay,
            "higher complexity should produce >= typing delay"
        );
    }
    #[test]
    fn test_natural_pause_disabled_returns_base_pause() {
        let mut config = make_default_config();
        config.natural_pausing = false;
        let sim = TypingSimulator::new(config);
        let mut chunk = make_chunk("Hello world.", 0.5);
        chunk.timing.pause_ms = 75;
        let pause = sim.calculate_natural_pause(&chunk);
        assert_eq!(
            pause.as_millis(),
            75,
            "disabled natural pausing should return the base pause_ms unchanged"
        );
    }
    #[test]
    fn test_natural_pause_enabled_adds_punctuation_pause() {
        let config = make_default_config();
        let sim = TypingSimulator::new(config);
        let mut chunk_no_punct = make_chunk("hello world without punctuation", 0.5);
        chunk_no_punct.timing.pause_ms = 0;
        let mut chunk_with_punct = make_chunk("Hello world. A sentence ends here.", 0.5);
        chunk_with_punct.timing.pause_ms = 0;
        let no_punct_pause = sim.calculate_natural_pause(&chunk_no_punct);
        let with_punct_pause = sim.calculate_natural_pause(&chunk_with_punct);
        assert!(
            with_punct_pause >= no_punct_pause,
            "punctuation should add natural pause time"
        );
    }
    #[test]
    fn test_natural_pause_high_complexity_adds_thinking_pause() {
        let config = make_default_config();
        let sim = TypingSimulator::new(config);
        let mut chunk_low = make_chunk("simple content here", 0.1);
        chunk_low.timing.pause_ms = 0;
        let mut chunk_high = make_chunk("simple content here", 0.9);
        chunk_high.timing.pause_ms = 0;
        let low_pause = sim.calculate_natural_pause(&chunk_low);
        let high_pause = sim.calculate_natural_pause(&chunk_high);
        assert!(
            high_pause > low_pause,
            "high complexity content should trigger thinking pause"
        );
    }
    #[test]
    fn test_generate_typing_burst_empty_content() {
        let config = make_default_config();
        let sim = TypingSimulator::new(config);
        let chunk = make_chunk("", 0.5);
        let events = sim.generate_typing_burst(&chunk);
        assert!(
            events.is_empty(),
            "empty content should produce no typing events"
        );
    }
    #[test]
    fn test_generate_typing_burst_produces_events() {
        let config = make_default_config();
        let sim = TypingSimulator::new(config);
        let chunk = make_chunk("Hello world! This is a test sentence.", 0.5);
        let events = sim.generate_typing_burst(&chunk);
        assert!(
            !events.is_empty(),
            "non-empty content should produce typing events"
        );
    }
    #[test]
    fn test_generate_typing_burst_has_start_typing_events() {
        let config = make_default_config();
        let sim = TypingSimulator::new(config);
        let chunk = make_chunk("Hello world this is test content for typing", 0.5);
        let events = sim.generate_typing_burst(&chunk);
        let start_events =
            events.iter().filter(|e| e.event_type == TypingEventType::StartTyping).count();
        assert!(
            start_events > 0,
            "burst should contain at least one StartTyping event"
        );
    }
    #[test]
    fn test_generate_typing_burst_event_char_indices_non_decreasing() {
        let config = make_default_config();
        let sim = TypingSimulator::new(config);
        let chunk = make_chunk(
            "Hello world. This is a multi-sentence test for character index tracking.",
            0.4,
        );
        let events = sim.generate_typing_burst(&chunk);
        let content_events: Vec<_> = events.iter().filter(|e| e.produces_content()).collect();
        for window in content_events.windows(2) {
            assert!(
                window[1].char_index >= window[0].char_index,
                "char_index should be non-decreasing across content events: {} >= {}",
                window[1].char_index,
                window[0].char_index
            );
        }
    }
    #[test]
    fn test_generate_typing_burst_all_delays_positive() {
        let config = make_default_config();
        let sim = TypingSimulator::new(config);
        let chunk = make_chunk("Test typing content with various patterns", 0.5);
        let events = sim.generate_typing_burst(&chunk);
        for event in &events {
            assert!(
                event.delay.as_millis() > 0,
                "every event delay should be positive, got {:?} for {:?}",
                event.delay,
                event.event_type
            );
        }
    }
    #[test]
    fn test_typing_event_is_pause_detection() {
        let pause_event = TypingEvent::new(
            TypingEventType::Pause,
            0,
            String::new(),
            Duration::from_millis(100),
        );
        let hesitation_event = TypingEvent::new(
            TypingEventType::Hesitation,
            0,
            String::new(),
            Duration::from_millis(200),
        );
        let typing_event = TypingEvent::new(
            TypingEventType::StartTyping,
            0,
            "hi".to_string(),
            Duration::from_millis(50),
        );
        assert!(
            pause_event.is_pause(),
            "Pause event should be detected as pause"
        );
        assert!(
            hesitation_event.is_pause(),
            "Hesitation event should be detected as pause"
        );
        assert!(
            !typing_event.is_pause(),
            "StartTyping should not be a pause"
        );
    }
    #[test]
    fn test_typing_event_produces_content() {
        let start_event = TypingEvent::new(
            TypingEventType::StartTyping,
            0,
            "hi".to_string(),
            Duration::from_millis(50),
        );
        let correction_event = TypingEvent::new(
            TypingEventType::Correction,
            0,
            "abc".to_string(),
            Duration::from_millis(150),
        );
        let pause_event = TypingEvent::new(
            TypingEventType::Pause,
            0,
            String::new(),
            Duration::from_millis(100),
        );
        assert!(
            start_event.produces_content(),
            "StartTyping should produce content"
        );
        assert!(
            correction_event.produces_content(),
            "Correction should produce content"
        );
        assert!(
            !pause_event.produces_content(),
            "Pause should not produce content"
        );
    }
    #[test]
    fn test_fast_typist_speed_multiplier_greater_than_balanced() {
        let balanced = TypingPersonality::balanced();
        let fast = TypingPersonality::fast_typist();
        assert!(
            fast.speed_multiplier > balanced.speed_multiplier,
            "fast typist should have higher speed_multiplier than balanced"
        );
    }
    #[test]
    fn test_careful_typist_pause_multiplier_greater_than_balanced() {
        let balanced = TypingPersonality::balanced();
        let careful = TypingPersonality::careful_typist();
        assert!(
            careful.pause_multiplier > balanced.pause_multiplier,
            "careful typist should have higher pause_multiplier than balanced"
        );
    }
    #[test]
    fn test_fast_typist_has_lower_hesitation_than_careful() {
        let fast = TypingPersonality::fast_typist();
        let careful = TypingPersonality::careful_typist();
        assert!(
            fast.hesitation_frequency < careful.hesitation_frequency,
            "fast typist should hesitate less than careful typist"
        );
    }
    #[test]
    fn test_typing_patterns_new() {
        let patterns = TypingPatterns::new();
        let analysis = patterns.analyze_content("algorithm implementation class");
        assert!(
            analysis.complexity_score >= 0.0 && analysis.complexity_score <= 1.0,
            "complexity score should be in [0.0, 1.0]"
        );
    }
    #[test]
    fn test_typing_patterns_technical_content_detected() {
        let patterns = TypingPatterns::new();
        let analysis = patterns.analyze_content(
            "The algorithm implementation uses a class method to execute the function",
        );
        assert!(
            analysis.pattern_scores.contains_key("technical"),
            "technical content should produce a 'technical' pattern score"
        );
        let tech_score = analysis.pattern_scores["technical"];
        assert!(
            tech_score > 0.0,
            "technical content should have positive technical score"
        );
    }
    #[test]
    fn test_typing_patterns_question_content_detected() {
        let patterns = TypingPatterns::new();
        let analysis =
            patterns.analyze_content("What is the best way to how and why does this work when");
        assert!(
            analysis.pattern_scores.contains_key("question"),
            "question content should produce a 'question' pattern score"
        );
        let q_score = analysis.pattern_scores["question"];
        assert!(
            q_score > 0.0,
            "question content should have positive question score"
        );
    }
    #[test]
    fn test_typing_patterns_analysis_cache() {
        let patterns = TypingPatterns::new();
        let content = "Testing cache behavior with unique content xyz";
        let analysis1 = patterns.analyze_content(content);
        let analysis2 = patterns.analyze_content(content);
        assert!(
            (analysis1.complexity_score - analysis2.complexity_score).abs() < 1e-6,
            "cached analysis should return identical complexity scores"
        );
    }
    #[test]
    fn test_typing_patterns_clear_cache() {
        let patterns = TypingPatterns::new();
        patterns.analyze_content("some content to cache");
        patterns.clear_cache();
        let stats = patterns.get_learning_stats();
        let _ = stats.total_analyses;
    }
    #[test]
    fn test_typing_analysis_primary_pattern_score_non_negative() {
        let patterns = TypingPatterns::new();
        let analysis = patterns.analyze_content("hello world simple text");
        let score = analysis.primary_pattern_score();
        assert!(score >= 0.0, "primary pattern score should be non-negative");
    }
    #[test]
    fn test_typing_analysis_complexity_score_bounded() {
        let patterns = TypingPatterns::new();
        let mut lcg = Lcg::new(12345);
        let test_cases = [
            "simple",
            "This is a more complex sentence with multiple clauses, however the algorithm implementation uses advanced optimization methodology architecture.",
            "1 + 2 = 3",
            "because although while since unless and but or nor for yet so",
        ];
        for content in &test_cases {
            let analysis = patterns.analyze_content(content);
            let _ = lcg.next_f32_unit();
            assert!(
                analysis.complexity_score >= 0.0 && analysis.complexity_score <= 1.0,
                "complexity_score should be in [0.0, 1.0] for '{}'",
                content
            );
        }
    }
    #[test]
    fn test_typing_pattern_calculate_score_empty_content() {
        let pattern = TypingPattern::new(
            "test",
            vec!["algorithm", "function"],
            0.5,
            TypingCharacteristics::slow_and_careful(),
        );
        let score = pattern.calculate_score("");
        assert_eq!(
            score, 0.0,
            "empty content should produce zero pattern score"
        );
    }
    #[test]
    fn test_typing_pattern_calculate_score_matching_keywords() {
        let pattern = TypingPattern::new(
            "technical",
            vec!["algorithm", "function", "class"],
            0.8,
            TypingCharacteristics::slow_and_careful(),
        );
        let score = pattern.calculate_score("the algorithm uses a function and class definition");
        assert!(
            score > 0.0,
            "content with matching keywords should have positive score"
        );
    }
    #[test]
    fn test_performance_tracker_initial_metrics() {
        let tracker = PerformanceTracker::new();
        let metrics = tracker.get_metrics();
        assert_eq!(metrics.total_bursts_generated, 0);
        assert_eq!(metrics.total_content_processed, 0);
        assert_eq!(metrics.average_burst_size, 0.0);
    }
    #[test]
    fn test_performance_tracker_record_burst_generation() {
        let tracker = PerformanceTracker::new();
        tracker.record_burst_generation(5, 100);
        let metrics = tracker.get_metrics();
        assert_eq!(metrics.total_bursts_generated, 5);
        assert_eq!(metrics.total_content_processed, 100);
        assert!(
            (metrics.average_burst_size - 20.0).abs() < 1e-3,
            "average_burst_size should be total_content / total_bursts = 20.0"
        );
    }
    #[test]
    fn test_performance_tracker_average_burst_size_accumulation() {
        let tracker = PerformanceTracker::new();
        tracker.record_burst_generation(4, 80);
        tracker.record_burst_generation(6, 120);
        let metrics = tracker.get_metrics();
        assert_eq!(metrics.total_bursts_generated, 10);
        assert_eq!(metrics.total_content_processed, 200);
        assert!((metrics.average_burst_size - 20.0).abs() < 1e-3);
    }
    #[test]
    fn test_performance_tracker_analyze_patterns_scores_in_range() {
        let tracker = PerformanceTracker::new();
        let summary = tracker.analyze_patterns();
        assert!(summary.efficiency_score >= 0.0 && summary.efficiency_score <= 1.0);
        assert!(summary.consistency_score >= 0.0 && summary.consistency_score <= 1.0);
        assert!(summary.adaptability_score >= 0.0 && summary.adaptability_score <= 1.0);
    }
    #[test]
    fn test_typing_characteristics_slow_and_careful_has_low_speed() {
        let chars = TypingCharacteristics::slow_and_careful();
        assert!(
            chars.speed_factor < 1.0,
            "slow_and_careful should have speed_factor < 1.0"
        );
    }
    #[test]
    fn test_typing_characteristics_fast_and_natural_has_high_speed() {
        let chars = TypingCharacteristics::fast_and_natural();
        assert!(
            chars.speed_factor > 1.0,
            "fast_and_natural should have speed_factor > 1.0"
        );
    }
    #[test]
    fn test_simple_prng_deterministic() {
        let config = make_default_config();
        let sim = TypingSimulator::new(config);
        let result1 = sim.simple_prng(0xDEADBEEF);
        let result2 = sim.simple_prng(0xDEADBEEF);
        assert_eq!(
            result1, result2,
            "simple_prng must be deterministic for same input"
        );
    }
    #[test]
    fn test_word_by_word_vs_char_mode_segment_count() {
        let config = make_default_config();
        let big_burst_personality = TypingPersonality {
            burst_size_multiplier: 3.0,
            ..TypingPersonality::balanced()
        };
        let small_burst_personality = TypingPersonality {
            burst_size_multiplier: 0.5,
            ..TypingPersonality::balanced()
        };
        let big_sim = TypingSimulator::with_personality(config.clone(), big_burst_personality);
        let small_sim = TypingSimulator::with_personality(config, small_burst_personality);
        let content = "one two three four five six seven eight nine ten eleven twelve";
        let chunk = make_chunk(content, 0.3);
        let big_events = big_sim.generate_typing_burst(&chunk);
        let small_events = small_sim.generate_typing_burst(&chunk);
        let big_starts = big_events
            .iter()
            .filter(|e| e.event_type == TypingEventType::StartTyping)
            .count();
        let small_starts = small_events
            .iter()
            .filter(|e| e.event_type == TypingEventType::StartTyping)
            .count();
        assert!(
            big_starts <= small_starts,
            "larger burst size multiplier should produce fewer or equal segments: {} <= {}",
            big_starts,
            small_starts
        );
    }
}
