//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::types::{
    PerformanceMetrics, PerformanceTracker, TypingAnalysis, TypingEvent, TypingEventType,
    TypingPatternSummary, TypingPatterns, TypingPersonality,
};
use crate::pipeline::conversational::streaming::types::{
    AdvancedStreamingConfig, ChunkMetadata, StreamChunk,
};

/// Natural typing simulator for human-like response delivery
///
/// The TypingSimulator creates realistic typing patterns by analyzing content
/// complexity, simulating natural pauses, and introducing human-like variations
/// in typing speed and timing. It supports different typing personalities and
/// can adapt to various content types.
#[derive(Debug)]
pub struct TypingSimulator {
    /// Configuration for typing simulation
    config: AdvancedStreamingConfig,
    /// Random number generator state for natural variation
    rng_state: Mutex<u64>,
    /// Typing patterns analyzer for content-aware simulation
    patterns: TypingPatterns,
    /// Performance tracking for adaptive optimization
    pub(super) performance_tracker: PerformanceTracker,
    /// Typing personality for consistent behavior
    pub(super) personality: TypingPersonality,
}
impl TypingSimulator {
    /// Create a new typing simulator with specified configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Advanced streaming configuration containing typing parameters
    ///
    /// # Returns
    ///
    /// A new TypingSimulator instance ready for content simulation
    pub fn new(config: AdvancedStreamingConfig) -> Self {
        Self {
            config,
            rng_state: Mutex::new(0xDEADBEEF),
            patterns: TypingPatterns::new(),
            performance_tracker: PerformanceTracker::new(),
            personality: TypingPersonality::default(),
        }
    }
    /// Create typing simulator with specific personality
    ///
    /// # Arguments
    ///
    /// * `config` - Advanced streaming configuration
    /// * `personality` - Typing personality to use for consistent behavior
    pub fn with_personality(
        config: AdvancedStreamingConfig,
        personality: TypingPersonality,
    ) -> Self {
        Self {
            config,
            rng_state: Mutex::new(0xCAFEBABE),
            patterns: TypingPatterns::new(),
            performance_tracker: PerformanceTracker::new(),
            personality,
        }
    }
    /// Calculate natural typing delay for a chunk
    ///
    /// This method analyzes the chunk content and metadata to determine
    /// realistic typing timing based on:
    /// - Content complexity and length
    /// - Natural typing speed variations
    /// - Personality-based adjustments
    /// - Historical performance patterns
    ///
    /// # Arguments
    ///
    /// * `chunk` - The stream chunk to calculate timing for
    ///
    /// # Returns
    ///
    /// Duration representing natural typing delay
    pub fn calculate_typing_delay(&self, chunk: &StreamChunk) -> Duration {
        let base_delay = Duration::from_millis(self.config.base_config.typing_delay_ms);
        if !self.config.variable_typing_speed {
            return base_delay;
        }
        let char_count = chunk.content.chars().count();
        let base_speed = self.config.base_typing_speed * self.personality.speed_multiplier;
        let chars_per_ms = base_speed / 1000.0;
        let base_duration_ms = char_count as f32 / chars_per_ms;
        let complexity_factor =
            0.8 + chunk.metadata.complexity * 0.4 * self.personality.complexity_sensitivity;
        let adjusted_duration_ms = base_duration_ms * complexity_factor;
        let variation = if self.config.speed_variation > 0.0 {
            let mut rng_state = self.rng_state.lock().expect("lock should not be poisoned");
            *rng_state = self.simple_prng(*rng_state);
            let random_val = (*rng_state as f32) / (u64::MAX as f32);
            let base_variation = self.config.speed_variation * self.personality.variation_intensity;
            let factor: f32 = 1.0 + (random_val - 0.5) * 2.0 * base_variation;
            factor.max(0.5).min(1.5)
        } else {
            1.0
        };
        let personality_adjustment =
            self.personality.calculate_adjustment(&chunk.content, chunk.metadata.complexity);
        let final_duration_ms =
            (adjusted_duration_ms * variation * personality_adjustment).max(10.0);
        Duration::from_millis(final_duration_ms as u64)
    }
    /// Simulate natural pauses based on content structure
    ///
    /// Analyzes content for natural pause points including:
    /// - Punctuation marks (periods, commas, etc.)
    /// - Complex concept boundaries
    /// - Sentence transitions
    /// - Thinking pauses for difficult content
    ///
    /// # Arguments
    ///
    /// * `chunk` - The stream chunk to analyze for pauses
    ///
    /// # Returns
    ///
    /// Duration representing natural pause time
    pub fn calculate_natural_pause(&self, chunk: &StreamChunk) -> Duration {
        if !self.config.natural_pausing {
            return Duration::from_millis(chunk.timing.pause_ms);
        }
        let content = &chunk.content;
        let base_pause = Duration::from_millis(chunk.timing.pause_ms);
        let punctuation_pause = self.calculate_punctuation_pause(content);
        let thinking_pause = if chunk.metadata.complexity > 0.7 {
            let base_thinking = (chunk.metadata.complexity * 200.0) as u64;
            let personality_thinking =
                (base_thinking as f32 * self.personality.thinking_pause_multiplier) as u64;
            Duration::from_millis(personality_thinking)
        } else {
            Duration::ZERO
        };
        let content_pause = self.calculate_content_specific_pause(content);
        base_pause + punctuation_pause + thinking_pause + content_pause
    }
    /// Generate typing burst pattern for natural flow
    ///
    /// Creates a sequence of typing events that simulate natural human typing:
    /// - Realistic typing bursts followed by pauses
    /// - Hesitation points at complex content
    /// - Natural corrections and backtracking
    /// - Adaptive segment sizing based on content
    ///
    /// # Arguments
    ///
    /// * `chunk` - The stream chunk to generate events for
    ///
    /// # Returns
    ///
    /// Vector of typing events representing natural typing flow
    pub fn generate_typing_burst(&self, chunk: &StreamChunk) -> Vec<TypingEvent> {
        let mut events = Vec::new();
        let content = &chunk.content;
        if content.is_empty() {
            return events;
        }
        let analysis = self.patterns.analyze_content(content);
        let segments = self.split_into_typing_segments(content, &analysis);
        let mut char_index = 0;
        for (i, segment) in segments.iter().enumerate() {
            if self.should_add_hesitation(segment, &analysis) {
                events.push(TypingEvent {
                    event_type: TypingEventType::Hesitation,
                    char_index,
                    content: String::new(),
                    delay: self.calculate_hesitation_delay(&analysis),
                });
            }
            events.push(TypingEvent {
                event_type: TypingEventType::StartTyping,
                char_index,
                content: segment.clone(),
                delay: self.calculate_segment_delay(i, segments.len(), &chunk.metadata, &analysis),
            });
            char_index += segment.chars().count();
            if i < segments.len() - 1 {
                events.push(TypingEvent {
                    event_type: TypingEventType::Pause,
                    char_index,
                    content: String::new(),
                    delay: self.calculate_inter_segment_pause(segment),
                });
            }
            if self.should_add_correction(segment, &analysis) {
                events.push(TypingEvent {
                    event_type: TypingEventType::Correction,
                    char_index: char_index.saturating_sub(3),
                    content: segment
                        .chars()
                        .rev()
                        .take(3)
                        .collect::<String>()
                        .chars()
                        .rev()
                        .collect(),
                    delay: Duration::from_millis(150),
                });
            }
        }
        self.performance_tracker.record_burst_generation(events.len(), content.len());
        events
    }
    /// Split content into natural typing segments
    ///
    /// Intelligently divides content into segments that represent natural
    /// typing bursts, considering:
    /// - Word boundaries and semantic units
    /// - Punctuation as natural break points
    /// - Content complexity and density
    /// - Personality-based preferences
    fn split_into_typing_segments(&self, content: &str, analysis: &TypingAnalysis) -> Vec<String> {
        let base_segment_size = (8.0 * self.personality.burst_size_multiplier) as usize;
        let words: Vec<&str> = content.split_whitespace().collect();
        let mut segments = Vec::new();
        let mut current_segment = String::new();
        let mut word_count = 0;
        let target_segment_size = if analysis.complexity_score > 0.7 {
            base_segment_size / 2
        } else {
            base_segment_size
        };
        for word in words {
            if word_count > 0 {
                current_segment.push(' ');
            }
            current_segment.push_str(word);
            word_count += 1;
            let should_break = word_count >= target_segment_size
                || self.is_natural_break_point(word)
                || self.should_break_at_word(word, analysis);
            if should_break {
                segments.push(current_segment.trim().to_string());
                current_segment.clear();
                word_count = 0;
            }
        }
        if !current_segment.trim().is_empty() {
            segments.push(current_segment.trim().to_string());
        }
        if segments.is_empty() {
            segments.push(content.to_string());
        }
        segments
    }
    /// Calculate delay for a typing segment with enhanced factors
    fn calculate_segment_delay(
        &self,
        segment_index: usize,
        total_segments: usize,
        metadata: &ChunkMetadata,
        analysis: &TypingAnalysis,
    ) -> Duration {
        let base_delay = 50;
        let position_factor = if segment_index == 0 {
            1.5 * self.personality.initial_delay_multiplier
        } else if segment_index == total_segments - 1 {
            0.8
        } else {
            1.0
        };
        let complexity_factor =
            0.8 + metadata.complexity * 0.4 * self.personality.complexity_sensitivity;
        let pattern_factor = if let Some(pattern) = &analysis.dominant_pattern {
            match pattern.as_str() {
                "technical" => 1.3,
                "emotional" => 0.9,
                "question" => 1.1,
                "explanation" => 1.2,
                _ => 1.0,
            }
        } else {
            1.0
        };
        let delay_ms =
            (base_delay as f32 * position_factor * complexity_factor * pattern_factor) as u64;
        Duration::from_millis(delay_ms.max(20).min(800))
    }
    /// Calculate pause between typing segments with content awareness
    fn calculate_inter_segment_pause(&self, segment: &str) -> Duration {
        let base_pause = (30.0 * self.personality.pause_multiplier) as u64;
        let punctuation_factor =
            if segment.ends_with('.') || segment.ends_with('!') || segment.ends_with('?') {
                2.0
            } else if segment.ends_with(',') || segment.ends_with(';') {
                1.5
            } else if segment.ends_with(':') {
                1.8
            } else {
                1.0
            };
        let content_factor = if self.contains_complex_terms(segment) {
            1.4
        } else if self.contains_emotional_markers(segment) {
            0.8
        } else {
            1.0
        };
        let pause_ms = (base_pause as f32 * punctuation_factor * content_factor) as u64;
        Duration::from_millis(pause_ms.max(10).min(400))
    }
    /// Calculate punctuation-specific pause timing
    fn calculate_punctuation_pause(&self, content: &str) -> Duration {
        let base_punctuation_pause = self.config.punctuation_pause_ms;
        if content.contains('.') || content.contains('!') || content.contains('?') {
            Duration::from_millis(
                (base_punctuation_pause as f32 * self.personality.pause_multiplier) as u64,
            )
        } else if content.contains(',') || content.contains(';') {
            Duration::from_millis(
                (base_punctuation_pause as f32 * self.personality.pause_multiplier / 2.0) as u64,
            )
        } else if content.contains(':') {
            Duration::from_millis(
                (base_punctuation_pause as f32 * self.personality.pause_multiplier * 0.8) as u64,
            )
        } else {
            Duration::ZERO
        }
    }
    /// Calculate content-specific pauses for special terms or concepts
    fn calculate_content_specific_pause(&self, content: &str) -> Duration {
        let mut pause_ms = 0u64;
        if self.contains_technical_terms(content) {
            pause_ms += (100.0 * self.personality.thinking_pause_multiplier) as u64;
        }
        if self.contains_emotional_markers(content) {
            pause_ms += (50.0 * self.personality.emotional_sensitivity) as u64;
        }
        if self.contains_numbers_or_calculations(content) {
            pause_ms += (80.0 * self.personality.calculation_pause_multiplier) as u64;
        }
        Duration::from_millis(pause_ms)
    }
    /// Determine if hesitation should be added before typing
    fn should_add_hesitation(&self, segment: &str, analysis: &TypingAnalysis) -> bool {
        let mut rng_state = self.rng_state.lock().expect("lock should not be poisoned");
        *rng_state = self.simple_prng(*rng_state);
        let random_val = (*rng_state as f32) / (u64::MAX as f32);
        let base_probability = 0.1 * self.personality.hesitation_frequency;
        let complexity_boost = analysis.complexity_score * 0.2;
        let technical_boost = if self.contains_technical_terms(segment) { 0.15 } else { 0.0 };
        let total_probability = (base_probability + complexity_boost + technical_boost).min(0.5);
        random_val < total_probability
    }
    /// Calculate hesitation delay duration
    fn calculate_hesitation_delay(&self, analysis: &TypingAnalysis) -> Duration {
        let base_hesitation = 200.0;
        let complexity_factor = 1.0 + analysis.complexity_score * 0.5;
        let personality_factor = self.personality.hesitation_intensity;
        let hesitation_ms = (base_hesitation * complexity_factor * personality_factor) as u64;
        Duration::from_millis(hesitation_ms.max(50).min(1000))
    }
    /// Determine if a correction should be simulated
    fn should_add_correction(&self, segment: &str, analysis: &TypingAnalysis) -> bool {
        let mut rng_state = self.rng_state.lock().expect("lock should not be poisoned");
        *rng_state = self.simple_prng(*rng_state);
        let random_val = (*rng_state as f32) / (u64::MAX as f32);
        let base_probability = 0.02 * self.personality.correction_frequency;
        let length_factor = (segment.len() as f32 / 20.0).min(0.1);
        let total_probability = (base_probability + length_factor).min(0.1);
        random_val < total_probability && segment.len() > 5
    }
    /// Simple pseudo-random number generator (Linear Congruential Generator)
    pub(super) fn simple_prng(&self, seed: u64) -> u64 {
        seed.wrapping_mul(1103515245).wrapping_add(12345)
    }
    /// Check if word represents a natural break point
    fn is_natural_break_point(&self, word: &str) -> bool {
        word.ends_with('.')
            || word.ends_with('!')
            || word.ends_with('?')
            || word.ends_with(',')
            || word.ends_with(';')
            || word.ends_with(':')
    }
    /// Check if should break at specific word based on analysis
    fn should_break_at_word(&self, word: &str, analysis: &TypingAnalysis) -> bool {
        if analysis.complexity_score > 0.6 {
            return ["and", "but", "however", "therefore", "because"]
                .contains(&word.to_lowercase().as_str());
        }
        false
    }
    /// Check if content contains complex terms
    fn contains_complex_terms(&self, content: &str) -> bool {
        let complex_indicators = [
            "algorithm",
            "implementation",
            "architecture",
            "optimization",
            "methodology",
            "paradigm",
            "infrastructure",
            "specification",
        ];
        let content_lower = content.to_lowercase();
        complex_indicators.iter().any(|&term| content_lower.contains(term))
    }
    /// Check if content contains technical terms
    fn contains_technical_terms(&self, content: &str) -> bool {
        let technical_terms = [
            "function",
            "variable",
            "parameter",
            "return",
            "class",
            "method",
            "object",
            "array",
            "string",
            "integer",
            "boolean",
            "compile",
        ];
        let content_lower = content.to_lowercase();
        technical_terms.iter().any(|&term| content_lower.contains(term))
    }
    /// Check if content contains emotional markers
    fn contains_emotional_markers(&self, content: &str) -> bool {
        let emotional_words = [
            "feel",
            "emotion",
            "happy",
            "sad",
            "angry",
            "excited",
            "worried",
            "concerned",
            "pleased",
            "disappointed",
        ];
        let content_lower = content.to_lowercase();
        emotional_words.iter().any(|&word| content_lower.contains(word))
    }
    /// Check if content contains numbers or calculations
    fn contains_numbers_or_calculations(&self, content: &str) -> bool {
        content.chars().any(|c| c.is_ascii_digit())
            || content.contains('+')
            || content.contains('-')
            || content.contains('*')
            || content.contains('/')
            || content.contains('=')
            || content.contains('%')
    }
    /// Get current performance metrics
    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        self.performance_tracker.get_metrics()
    }
    /// Update typing personality
    pub fn update_personality(&mut self, personality: TypingPersonality) {
        self.personality = personality;
    }
    /// Analyze recent typing patterns for optimization
    pub fn analyze_recent_patterns(&self) -> TypingPatternSummary {
        self.performance_tracker.analyze_patterns()
    }
}
/// Detailed content metrics for analysis
#[derive(Debug, Clone)]
pub struct ContentMetrics {
    pub word_count: usize,
    pub character_count: usize,
    pub sentence_count: usize,
    pub paragraph_count: usize,
    pub punctuation_density: f32,
    pub lexical_diversity: f32,
    pub readability_score: f32,
}
/// Typing recommendations based on content analysis
#[derive(Debug, Clone)]
pub struct TypingRecommendations {
    pub suggested_speed_multiplier: f32,
    pub pause_emphasis: PauseEmphasis,
    pub hesitation_likelihood: f32,
    pub burst_size_adjustment: f32,
    pub special_handling: Vec<String>,
}
/// Pause emphasis levels
#[derive(Debug, Clone)]
pub enum PauseEmphasis {
    Low,
    Normal,
    High,
    VeryHigh,
}
