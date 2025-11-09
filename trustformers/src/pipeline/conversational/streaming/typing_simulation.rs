//! Natural typing simulation and conversation flow management.
//!
//! This module provides sophisticated typing simulation capabilities for creating
//! human-like conversation experiences. It includes realistic typing patterns,
//! variable speed simulation, natural pauses, and adaptive flow control based
//! on content complexity and conversational context.

use super::types::*;
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};

// ================================================================================================
// TYPING SIMULATION CORE
// ================================================================================================

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
    performance_tracker: PerformanceTracker,
    /// Typing personality for consistent behavior
    personality: TypingPersonality,
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
            rng_state: Mutex::new(0xDEADBEEF), // Simple PRNG seed
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
            rng_state: Mutex::new(0xCAFEBABE), // Simple PRNG seed
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

        // Character-based timing with personality adjustment
        let char_count = chunk.content.chars().count();
        let base_speed = self.config.base_typing_speed * self.personality.speed_multiplier;
        let chars_per_ms = base_speed / 1000.0;
        let base_duration_ms = char_count as f32 / chars_per_ms;

        // Apply complexity factor with personality influence
        let complexity_factor =
            0.8 + chunk.metadata.complexity * 0.4 * self.personality.complexity_sensitivity;
        let adjusted_duration_ms = base_duration_ms * complexity_factor;

        // Add natural variation based on personality
        let variation = if self.config.speed_variation > 0.0 {
            let mut rng_state = self.rng_state.lock().unwrap();
            *rng_state = self.simple_prng(*rng_state);
            let random_val = (*rng_state as f32) / (u64::MAX as f32);
            let base_variation = self.config.speed_variation * self.personality.variation_intensity;
            let factor: f32 = 1.0 + (random_val - 0.5) * 2.0 * base_variation;
            factor.max(0.5).min(1.5)
        } else {
            1.0
        };

        // Apply personality-based adjustments
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

        // Punctuation-based pauses with personality influence
        let punctuation_pause = self.calculate_punctuation_pause(content);

        // Thinking pauses for complex content
        let thinking_pause = if chunk.metadata.complexity > 0.7 {
            let base_thinking = (chunk.metadata.complexity * 200.0) as u64;
            let personality_thinking =
                (base_thinking as f32 * self.personality.thinking_pause_multiplier) as u64;
            Duration::from_millis(personality_thinking)
        } else {
            Duration::ZERO
        };

        // Content-specific pauses (technical terms, emotional content, etc.)
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

        // Analyze content for typing complexity
        let analysis = self.patterns.analyze_content(content);

        // Split content into natural typing segments
        let segments = self.split_into_typing_segments(content, &analysis);
        let mut char_index = 0;

        for (i, segment) in segments.iter().enumerate() {
            // Add hesitation for complex segments
            if self.should_add_hesitation(segment, &analysis) {
                events.push(TypingEvent {
                    event_type: TypingEventType::Hesitation,
                    char_index,
                    content: String::new(),
                    delay: self.calculate_hesitation_delay(&analysis),
                });
            }

            // Main typing burst
            events.push(TypingEvent {
                event_type: TypingEventType::StartTyping,
                char_index,
                content: segment.clone(),
                delay: self.calculate_segment_delay(i, segments.len(), &chunk.metadata, &analysis),
            });

            char_index += segment.chars().count();

            // Natural pause between segments
            if i < segments.len() - 1 {
                events.push(TypingEvent {
                    event_type: TypingEventType::Pause,
                    char_index,
                    content: String::new(),
                    delay: self.calculate_inter_segment_pause(segment),
                });
            }

            // Occasional corrections for realism
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

        // Adjust segment size based on content complexity
        let target_segment_size = if analysis.complexity_score > 0.7 {
            base_segment_size / 2 // Smaller segments for complex content
        } else {
            base_segment_size
        };

        for word in words {
            if word_count > 0 {
                current_segment.push(' ');
            }
            current_segment.push_str(word);
            word_count += 1;

            // Natural break points
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
        let base_delay = 50; // Base delay in ms

        // Position-based adjustments
        let position_factor = if segment_index == 0 {
            1.5 * self.personality.initial_delay_multiplier
        } else if segment_index == total_segments - 1 {
            0.8 // Slightly faster for final segment
        } else {
            1.0
        };

        // Complexity affects thinking time
        let complexity_factor =
            0.8 + metadata.complexity * 0.4 * self.personality.complexity_sensitivity;

        // Pattern-based adjustments
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

        // Punctuation-based adjustments
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

        // Content-based adjustments
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

        // Technical terms require thinking time
        if self.contains_technical_terms(content) {
            pause_ms += (100.0 * self.personality.thinking_pause_multiplier) as u64;
        }

        // Emotional content might have natural hesitation
        if self.contains_emotional_markers(content) {
            pause_ms += (50.0 * self.personality.emotional_sensitivity) as u64;
        }

        // Numbers and calculations
        if self.contains_numbers_or_calculations(content) {
            pause_ms += (80.0 * self.personality.calculation_pause_multiplier) as u64;
        }

        Duration::from_millis(pause_ms)
    }

    /// Determine if hesitation should be added before typing
    fn should_add_hesitation(&self, segment: &str, analysis: &TypingAnalysis) -> bool {
        let mut rng_state = self.rng_state.lock().unwrap();
        *rng_state = self.simple_prng(*rng_state);
        let random_val = (*rng_state as f32) / (u64::MAX as f32);

        let base_probability = 0.1 * self.personality.hesitation_frequency;

        // Increase probability for complex content
        let complexity_boost = analysis.complexity_score * 0.2;

        // Increase for technical terms
        let technical_boost = if self.contains_technical_terms(segment) { 0.15 } else { 0.0 };

        let total_probability = (base_probability + complexity_boost + technical_boost).min(0.5);

        random_val < total_probability
    }

    /// Calculate hesitation delay duration
    fn calculate_hesitation_delay(&self, analysis: &TypingAnalysis) -> Duration {
        let base_hesitation = 200.0; // Base hesitation in ms
        let complexity_factor = 1.0 + analysis.complexity_score * 0.5;
        let personality_factor = self.personality.hesitation_intensity;

        let hesitation_ms = (base_hesitation * complexity_factor * personality_factor) as u64;
        Duration::from_millis(hesitation_ms.max(50).min(1000))
    }

    /// Determine if a correction should be simulated
    fn should_add_correction(&self, segment: &str, analysis: &TypingAnalysis) -> bool {
        let mut rng_state = self.rng_state.lock().unwrap();
        *rng_state = self.simple_prng(*rng_state);
        let random_val = (*rng_state as f32) / (u64::MAX as f32);

        let base_probability = 0.02 * self.personality.correction_frequency;

        // Increase for longer segments (more chance of error)
        let length_factor = (segment.len() as f32 / 20.0).min(0.1);

        let total_probability = (base_probability + length_factor).min(0.1);

        random_val < total_probability && segment.len() > 5
    }

    /// Simple pseudo-random number generator (Linear Congruential Generator)
    fn simple_prng(&self, seed: u64) -> u64 {
        // Using standard LCG parameters
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
        // Break at conjunctions for complex content
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

// ================================================================================================
// TYPING EVENTS AND EVENT TYPES
// ================================================================================================

/// Typing event for natural simulation
///
/// Represents a single event in the typing simulation sequence,
/// including the type of event, timing, and associated content.
#[derive(Debug, Clone)]
pub struct TypingEvent {
    /// Type of typing event
    pub event_type: TypingEventType,
    /// Character index in the full content
    pub char_index: usize,
    /// Content for this event
    pub content: String,
    /// Delay before this event
    pub delay: Duration,
}

impl TypingEvent {
    /// Create a new typing event
    pub fn new(
        event_type: TypingEventType,
        char_index: usize,
        content: String,
        delay: Duration,
    ) -> Self {
        Self {
            event_type,
            char_index,
            content,
            delay,
        }
    }

    /// Check if this is a pause event
    pub fn is_pause(&self) -> bool {
        matches!(
            self.event_type,
            TypingEventType::Pause | TypingEventType::Hesitation
        )
    }

    /// Check if this event produces visible content
    pub fn produces_content(&self) -> bool {
        matches!(
            self.event_type,
            TypingEventType::StartTyping | TypingEventType::Correction
        )
    }

    /// Get total duration including delay
    pub fn total_duration(&self) -> Duration {
        self.delay
    }
}

/// Types of typing events for realistic simulation
#[derive(Debug, Clone, PartialEq)]
pub enum TypingEventType {
    /// Start typing content
    StartTyping,
    /// Natural pause in typing
    Pause,
    /// Correction or backtracking
    Correction,
    /// Hesitation before difficult content
    Hesitation,
}

impl TypingEventType {
    /// Get human-readable description of event type
    pub fn description(&self) -> &'static str {
        match self {
            TypingEventType::StartTyping => "Start typing content",
            TypingEventType::Pause => "Natural pause",
            TypingEventType::Correction => "Correction or backtrack",
            TypingEventType::Hesitation => "Hesitation before content",
        }
    }

    /// Check if this event type affects timing
    pub fn affects_timing(&self) -> bool {
        true // All event types affect timing in some way
    }
}

// ================================================================================================
// TYPING PATTERNS ANALYSIS
// ================================================================================================

/// Typing patterns analyzer for content-aware simulation
///
/// Analyzes content to identify patterns that should influence typing behavior,
/// such as technical complexity, emotional content, or question patterns.
#[derive(Debug)]
pub struct TypingPatterns {
    /// Predefined typing patterns for different content types
    patterns: Vec<TypingPattern>,
    /// Analysis cache for performance
    analysis_cache: Mutex<HashMap<String, TypingAnalysis>>,
    /// Pattern learning history
    learning_history: Mutex<VecDeque<PatternLearningEntry>>,
}

impl Default for TypingPatterns {
    fn default() -> Self {
        Self::new()
    }
}

impl TypingPatterns {
    /// Create new typing patterns analyzer with default patterns
    pub fn new() -> Self {
        Self {
            patterns: Self::create_default_patterns(),
            analysis_cache: Mutex::new(HashMap::new()),
            learning_history: Mutex::new(VecDeque::new()),
        }
    }

    /// Create typing patterns with custom pattern set
    pub fn with_patterns(patterns: Vec<TypingPattern>) -> Self {
        Self {
            patterns,
            analysis_cache: Mutex::new(HashMap::new()),
            learning_history: Mutex::new(VecDeque::new()),
        }
    }

    /// Analyze content for typing patterns
    ///
    /// Performs comprehensive analysis of content to identify:
    /// - Dominant typing patterns
    /// - Content complexity
    /// - Naturalness indicators
    /// - Optimal typing strategies
    ///
    /// # Arguments
    ///
    /// * `content` - The content to analyze
    ///
    /// # Returns
    ///
    /// TypingAnalysis containing pattern analysis results
    pub fn analyze_content(&self, content: &str) -> TypingAnalysis {
        // Check cache first
        if let Ok(cache) = self.analysis_cache.lock() {
            if let Some(cached) = cache.get(content) {
                return cached.clone();
            }
        }

        let content_lower = content.to_lowercase();
        let mut pattern_scores = HashMap::new();

        // Analyze against all patterns
        for pattern in &self.patterns {
            let score = pattern.calculate_score(&content_lower);
            pattern_scores.insert(pattern.name.clone(), score);
        }

        // Find dominant pattern
        let dominant_pattern = pattern_scores.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .filter(|(_, score)| **score > 0.1) // Only if significant
            .map(|(k, _)| k.clone());

        let analysis = TypingAnalysis {
            dominant_pattern,
            pattern_scores,
            complexity_score: self.calculate_complexity(&content_lower),
            naturalness_indicators: self.extract_naturalness_indicators(&content_lower),
            content_metrics: self.calculate_content_metrics(&content_lower),
            typing_recommendations: self.generate_typing_recommendations(&content_lower),
        };

        // Cache the result
        if let Ok(mut cache) = self.analysis_cache.lock() {
            if cache.len() < 100 {
                // Limit cache size
                cache.insert(content.to_string(), analysis.clone());
            }
        }

        // Record for learning
        self.record_analysis_for_learning(content, &analysis);

        analysis
    }

    /// Calculate content complexity score
    ///
    /// Analyzes various factors that contribute to content complexity:
    /// - Average word length
    /// - Sentence structure
    /// - Technical terminology
    /// - Syntactic complexity
    fn calculate_complexity(&self, content: &str) -> f32 {
        let word_count = content.split_whitespace().count();
        if word_count == 0 {
            return 0.0;
        }

        // Basic linguistic complexity
        let words: Vec<&str> = content.split_whitespace().collect();
        let avg_word_length =
            words.iter().map(|w| w.len()).sum::<usize>() as f32 / word_count as f32;

        let sentence_count = content.matches('.').count()
            + content.matches('!').count()
            + content.matches('?').count();
        let avg_sentence_length = word_count as f32 / sentence_count.max(1) as f32;

        // Technical complexity indicators
        let technical_terms = words.iter().filter(|word| self.is_technical_term(word)).count()
            as f32
            / word_count as f32;

        // Syntactic complexity (subordinate clauses, etc.)
        let syntactic_complexity = self.calculate_syntactic_complexity(content);

        // Normalize and combine factors
        let word_complexity = (avg_word_length / 8.0).min(1.0);
        let sentence_complexity = (avg_sentence_length / 20.0).min(1.0);
        let technical_complexity = technical_terms.min(1.0);

        (word_complexity + sentence_complexity + technical_complexity + syntactic_complexity) / 4.0
    }

    /// Extract naturalness indicators from content
    ///
    /// Identifies markers that suggest natural, conversational language
    /// versus formal or artificial text patterns.
    fn extract_naturalness_indicators(&self, content: &str) -> Vec<String> {
        let mut indicators = Vec::new();

        // Hesitation markers
        if content.contains("um")
            || content.contains("uh")
            || content.contains("er")
            || content.contains("hmm")
            || content.contains("well")
        {
            indicators.push("hesitation_markers".to_string());
        }

        // Pause indicators
        if content.matches("...").count() > 0 {
            indicators.push("ellipsis_pauses".to_string());
        }

        // Emotional expression
        if content.matches("!").count() > content.matches(".").count() {
            indicators.push("exclamatory".to_string());
        }

        // Interactive elements
        if content.contains("?") {
            indicators.push("questioning".to_string());
        }

        // Informal language
        if self.contains_informal_language(content) {
            indicators.push("informal_language".to_string());
        }

        // Filler words
        if self.contains_filler_words(content) {
            indicators.push("filler_words".to_string());
        }

        indicators
    }

    /// Calculate detailed content metrics
    fn calculate_content_metrics(&self, content: &str) -> ContentMetrics {
        let words: Vec<&str> = content.split_whitespace().collect();
        let word_count = words.len();

        ContentMetrics {
            word_count,
            character_count: content.chars().count(),
            sentence_count: self.count_sentences(content),
            paragraph_count: content.split("\n\n").count(),
            punctuation_density: self.calculate_punctuation_density(content),
            lexical_diversity: self.calculate_lexical_diversity(&words),
            readability_score: self.calculate_readability_score(content),
        }
    }

    /// Generate typing recommendations based on analysis
    fn generate_typing_recommendations(&self, content: &str) -> TypingRecommendations {
        let complexity = self.calculate_complexity(content);
        let has_technical = self.contains_technical_terms(content);
        let has_emotional = self.contains_emotional_markers(content);

        TypingRecommendations {
            suggested_speed_multiplier: if complexity > 0.7 { 0.8 } else { 1.0 },
            pause_emphasis: if has_technical { PauseEmphasis::High } else { PauseEmphasis::Normal },
            hesitation_likelihood: if complexity > 0.6 { 0.3 } else { 0.1 },
            burst_size_adjustment: if has_emotional { 0.8 } else { 1.0 },
            special_handling: self.identify_special_handling_needs(content),
        }
    }

    /// Create default typing patterns for common content types
    fn create_default_patterns() -> Vec<TypingPattern> {
        vec![
            TypingPattern::new(
                "technical",
                vec![
                    "algorithm",
                    "implementation",
                    "function",
                    "variable",
                    "class",
                    "method",
                ],
                0.8,
                TypingCharacteristics::slow_and_careful(),
            ),
            TypingPattern::new(
                "emotional",
                vec!["feel", "think", "believe", "love", "hate", "happy", "sad"],
                0.6,
                TypingCharacteristics::variable_with_pauses(),
            ),
            TypingPattern::new(
                "question",
                vec!["what", "how", "why", "when", "where", "who"],
                0.7,
                TypingCharacteristics::thoughtful(),
            ),
            TypingPattern::new(
                "explanation",
                vec!["because", "therefore", "however", "moreover", "furthermore"],
                0.75,
                TypingCharacteristics::deliberate(),
            ),
            TypingPattern::new(
                "casual",
                vec!["yeah", "okay", "sure", "cool", "awesome", "nice"],
                0.4,
                TypingCharacteristics::fast_and_natural(),
            ),
        ]
    }

    /// Record analysis for pattern learning
    fn record_analysis_for_learning(&self, content: &str, analysis: &TypingAnalysis) {
        if let Ok(mut history) = self.learning_history.lock() {
            let entry = PatternLearningEntry {
                content_snippet: content.chars().take(100).collect(),
                analysis_result: analysis.clone(),
                timestamp: Instant::now(),
            };

            history.push_back(entry);

            // Limit history size
            while history.len() > 1000 {
                history.pop_front();
            }
        }
    }

    /// Helper methods for content analysis
    fn is_technical_term(&self, word: &str) -> bool {
        let technical_terms = [
            "algorithm",
            "function",
            "variable",
            "class",
            "method",
            "object",
            "array",
            "string",
            "integer",
            "boolean",
            "compile",
            "execute",
            "debug",
            "optimize",
            "refactor",
            "implement",
            "deploy",
        ];
        technical_terms.contains(&word.to_lowercase().as_str())
    }

    fn calculate_syntactic_complexity(&self, content: &str) -> f32 {
        let subordinating_conjunctions = ["because", "although", "while", "since", "unless"];
        let coordinating_conjunctions = ["and", "but", "or", "nor", "for", "yet", "so"];

        let subordinate_count = subordinating_conjunctions
            .iter()
            .map(|conj| content.to_lowercase().matches(conj).count())
            .sum::<usize>();

        let coordinate_count = coordinating_conjunctions
            .iter()
            .map(|conj| content.to_lowercase().matches(conj).count())
            .sum::<usize>();

        let word_count = content.split_whitespace().count().max(1);
        ((subordinate_count * 2 + coordinate_count) as f32 / word_count as f32).min(1.0)
    }

    fn contains_informal_language(&self, content: &str) -> bool {
        let informal_markers = [
            "gonna", "wanna", "kinda", "sorta", "yeah", "nah", "ok", "lol",
        ];
        let content_lower = content.to_lowercase();
        informal_markers.iter().any(|&marker| content_lower.contains(marker))
    }

    fn contains_filler_words(&self, content: &str) -> bool {
        let fillers = [
            "like",
            "you know",
            "I mean",
            "sort of",
            "kind of",
            "basically",
        ];
        let content_lower = content.to_lowercase();
        fillers.iter().any(|&filler| content_lower.contains(filler))
    }

    fn contains_technical_terms(&self, content: &str) -> bool {
        content.split_whitespace().any(|word| self.is_technical_term(word))
    }

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
            "love",
            "hate",
        ];
        let content_lower = content.to_lowercase();
        emotional_words.iter().any(|&word| content_lower.contains(word))
    }

    fn count_sentences(&self, content: &str) -> usize {
        content.matches('.').count() + content.matches('!').count() + content.matches('?').count()
    }

    fn calculate_punctuation_density(&self, content: &str) -> f32 {
        let punctuation_chars = content.chars().filter(|c| c.is_ascii_punctuation()).count();
        let total_chars = content.chars().count().max(1);
        punctuation_chars as f32 / total_chars as f32
    }

    fn calculate_lexical_diversity(&self, words: &[&str]) -> f32 {
        if words.is_empty() {
            return 0.0;
        }
        let unique_words: std::collections::HashSet<_> = words.iter().cloned().collect();
        unique_words.len() as f32 / words.len() as f32
    }

    fn calculate_readability_score(&self, content: &str) -> f32 {
        // Simplified readability calculation
        let words = content.split_whitespace().count() as f32;
        let sentences = self.count_sentences(content).max(1) as f32;
        let avg_sentence_length = words / sentences;

        // Lower score = easier to read = faster typing
        (avg_sentence_length / 15.0).min(1.0)
    }

    fn identify_special_handling_needs(&self, content: &str) -> Vec<String> {
        let mut needs = Vec::new();

        if content.contains("```") || content.contains("```") {
            needs.push("code_block".to_string());
        }

        if content.chars().any(|c| c.is_ascii_digit()) && content.contains('.') {
            needs.push("numbers_and_decimals".to_string());
        }

        if content.contains("http") || content.contains("www") {
            needs.push("urls_and_links".to_string());
        }

        if content.chars().filter(|c| c.is_uppercase()).count() > content.len() / 4 {
            needs.push("heavy_capitalization".to_string());
        }

        needs
    }

    /// Clear analysis cache
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.analysis_cache.lock() {
            cache.clear();
        }
    }

    /// Get learning statistics
    pub fn get_learning_stats(&self) -> LearningStats {
        if let Ok(history) = self.learning_history.lock() {
            LearningStats {
                total_analyses: history.len(),
                pattern_distribution: self.calculate_pattern_distribution(&history),
                complexity_trends: self.calculate_complexity_trends(&history),
            }
        } else {
            LearningStats::default()
        }
    }

    fn calculate_pattern_distribution(
        &self,
        history: &VecDeque<PatternLearningEntry>,
    ) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();
        for entry in history {
            if let Some(ref pattern) = entry.analysis_result.dominant_pattern {
                *distribution.entry(pattern.clone()).or_insert(0) += 1;
            }
        }
        distribution
    }

    fn calculate_complexity_trends(&self, history: &VecDeque<PatternLearningEntry>) -> Vec<f32> {
        history.iter().map(|entry| entry.analysis_result.complexity_score).collect()
    }
}

/// Individual typing pattern for content classification
#[derive(Debug, Clone)]
pub struct TypingPattern {
    /// Pattern identifier name
    pub name: String,
    /// Keywords associated with this pattern
    pub keywords: Vec<String>,
    /// Base complexity score for this pattern
    pub complexity: f32,
    /// Typing characteristics for this pattern
    pub characteristics: TypingCharacteristics,
}

impl TypingPattern {
    /// Create a new typing pattern
    ///
    /// # Arguments
    ///
    /// * `name` - Identifier for this pattern
    /// * `keywords` - Keywords that trigger this pattern
    /// * `complexity` - Base complexity score (0.0-1.0)
    /// * `characteristics` - Typing behavior characteristics
    pub fn new(
        name: &str,
        keywords: Vec<&str>,
        complexity: f32,
        characteristics: TypingCharacteristics,
    ) -> Self {
        Self {
            name: name.to_string(),
            keywords: keywords.iter().map(|s| s.to_string()).collect(),
            complexity,
            characteristics,
        }
    }

    /// Calculate pattern match score for content
    ///
    /// Analyzes how well the content matches this pattern based on
    /// keyword frequency and contextual indicators.
    ///
    /// # Arguments
    ///
    /// * `content` - Content to analyze
    ///
    /// # Returns
    ///
    /// Score between 0.0 and 1.0 indicating pattern match strength
    pub fn calculate_score(&self, content: &str) -> f32 {
        let word_count = content.split_whitespace().count();
        if word_count == 0 {
            return 0.0;
        }

        // Direct keyword matches
        let keyword_matches = self
            .keywords
            .iter()
            .map(|keyword| content.matches(keyword).count())
            .sum::<usize>();

        // Contextual scoring based on pattern type
        let contextual_score = self.calculate_contextual_score(content);

        // Combine scores with weighting
        let keyword_score = (keyword_matches as f32 / word_count as f32) * self.complexity;
        let final_score = (keyword_score * 0.7) + (contextual_score * 0.3);

        final_score.min(1.0)
    }

    /// Calculate contextual score based on pattern-specific indicators
    fn calculate_contextual_score(&self, content: &str) -> f32 {
        match self.name.as_str() {
            "technical" => self.score_technical_context(content),
            "emotional" => self.score_emotional_context(content),
            "question" => self.score_question_context(content),
            "explanation" => self.score_explanation_context(content),
            "casual" => self.score_casual_context(content),
            _ => 0.0,
        }
    }

    fn score_technical_context(&self, content: &str) -> f32 {
        let technical_indicators = content.matches("()").count()
            + content.matches("{}").count()
            + content.matches("[]").count();
        (technical_indicators as f32 / 10.0).min(0.5)
    }

    fn score_emotional_context(&self, content: &str) -> f32 {
        let emotional_punctuation = content.matches("!").count() + content.matches("...").count();
        (emotional_punctuation as f32 / 5.0).min(0.5)
    }

    fn score_question_context(&self, content: &str) -> f32 {
        let question_marks = content.matches("?").count();
        (question_marks as f32 / 2.0).min(0.5)
    }

    fn score_explanation_context(&self, content: &str) -> f32 {
        let explanation_markers = content.matches(":").count() + content.matches("->").count();
        (explanation_markers as f32 / 3.0).min(0.5)
    }

    fn score_casual_context(&self, content: &str) -> f32 {
        let casual_markers = if content.chars().any(|c| c.is_lowercase())
            && !content.chars().any(|c| c.is_uppercase())
        {
            0.3
        } else {
            0.0
        };
        casual_markers
    }
}

/// Analysis results for typing patterns
#[derive(Debug, Clone)]
pub struct TypingAnalysis {
    /// Dominant typing pattern detected
    pub dominant_pattern: Option<String>,
    /// Scores for all patterns
    pub pattern_scores: HashMap<String, f32>,
    /// Overall complexity score
    pub complexity_score: f32,
    /// Naturalness indicators
    pub naturalness_indicators: Vec<String>,
    /// Detailed content metrics
    pub content_metrics: ContentMetrics,
    /// Typing recommendations
    pub typing_recommendations: TypingRecommendations,
}

impl TypingAnalysis {
    /// Get the primary pattern score
    pub fn primary_pattern_score(&self) -> f32 {
        self.pattern_scores.values().cloned().fold(0.0, f32::max)
    }

    /// Check if content requires special handling
    pub fn requires_special_handling(&self) -> bool {
        !self.typing_recommendations.special_handling.is_empty()
    }

    /// Get suggested typing speed adjustment
    pub fn suggested_speed_adjustment(&self) -> f32 {
        self.typing_recommendations.suggested_speed_multiplier
    }
}

// ================================================================================================
// TYPING PERSONALITY AND CHARACTERISTICS
// ================================================================================================

/// Typing personality for consistent behavioral simulation
#[derive(Debug, Clone)]
pub struct TypingPersonality {
    /// Speed multiplier for base typing speed
    pub speed_multiplier: f32,
    /// Sensitivity to content complexity
    pub complexity_sensitivity: f32,
    /// Intensity of speed variations
    pub variation_intensity: f32,
    /// Multiplier for pause durations
    pub pause_multiplier: f32,
    /// Frequency of hesitation events
    pub hesitation_frequency: f32,
    /// Intensity of hesitation when it occurs
    pub hesitation_intensity: f32,
    /// Frequency of typing corrections
    pub correction_frequency: f32,
    /// Multiplier for initial thinking delays
    pub initial_delay_multiplier: f32,
    /// Multiplier for thinking pauses
    pub thinking_pause_multiplier: f32,
    /// Sensitivity to emotional content
    pub emotional_sensitivity: f32,
    /// Multiplier for calculation/number pauses
    pub calculation_pause_multiplier: f32,
    /// Multiplier for typing burst sizes
    pub burst_size_multiplier: f32,
}

impl Default for TypingPersonality {
    fn default() -> Self {
        Self::balanced()
    }
}

impl TypingPersonality {
    /// Create a balanced typing personality
    pub fn balanced() -> Self {
        Self {
            speed_multiplier: 1.0,
            complexity_sensitivity: 1.0,
            variation_intensity: 1.0,
            pause_multiplier: 1.0,
            hesitation_frequency: 1.0,
            hesitation_intensity: 1.0,
            correction_frequency: 1.0,
            initial_delay_multiplier: 1.0,
            thinking_pause_multiplier: 1.0,
            emotional_sensitivity: 1.0,
            calculation_pause_multiplier: 1.0,
            burst_size_multiplier: 1.0,
        }
    }

    /// Create a fast typist personality
    pub fn fast_typist() -> Self {
        Self {
            speed_multiplier: 1.3,
            complexity_sensitivity: 0.7,
            variation_intensity: 0.8,
            pause_multiplier: 0.7,
            hesitation_frequency: 0.5,
            hesitation_intensity: 0.6,
            correction_frequency: 1.2,
            initial_delay_multiplier: 0.8,
            thinking_pause_multiplier: 0.6,
            emotional_sensitivity: 0.8,
            calculation_pause_multiplier: 0.9,
            burst_size_multiplier: 1.2,
        }
    }

    /// Create a careful typist personality
    pub fn careful_typist() -> Self {
        Self {
            speed_multiplier: 0.8,
            complexity_sensitivity: 1.3,
            variation_intensity: 0.6,
            pause_multiplier: 1.4,
            hesitation_frequency: 1.5,
            hesitation_intensity: 1.3,
            correction_frequency: 0.3,
            initial_delay_multiplier: 1.5,
            thinking_pause_multiplier: 1.4,
            emotional_sensitivity: 1.2,
            calculation_pause_multiplier: 1.5,
            burst_size_multiplier: 0.8,
        }
    }

    /// Create an expressive typist personality
    pub fn expressive_typist() -> Self {
        Self {
            speed_multiplier: 1.1,
            complexity_sensitivity: 0.9,
            variation_intensity: 1.4,
            pause_multiplier: 1.2,
            hesitation_frequency: 1.3,
            hesitation_intensity: 1.1,
            correction_frequency: 0.8,
            initial_delay_multiplier: 1.2,
            thinking_pause_multiplier: 1.0,
            emotional_sensitivity: 1.5,
            calculation_pause_multiplier: 0.8,
            burst_size_multiplier: 0.9,
        }
    }

    /// Calculate personality-based adjustment for specific content
    pub fn calculate_adjustment(&self, content: &str, complexity: f32) -> f32 {
        let mut adjustment = 1.0;

        // Complexity-based adjustment
        adjustment *= 1.0 + (complexity - 0.5) * (self.complexity_sensitivity - 1.0) * 0.5;

        // Content-specific adjustments
        if self.contains_emotional_content(content) {
            adjustment *= 0.9 + (self.emotional_sensitivity - 1.0) * 0.2;
        }

        if self.contains_numerical_content(content) {
            adjustment *= 1.0 + (self.calculation_pause_multiplier - 1.0) * 0.3;
        }

        adjustment.max(0.5).min(2.0)
    }

    fn contains_emotional_content(&self, content: &str) -> bool {
        let emotional_indicators = ["feel", "emotion", "excited", "worried", "happy", "sad"];
        let content_lower = content.to_lowercase();
        emotional_indicators.iter().any(|&indicator| content_lower.contains(indicator))
    }

    fn contains_numerical_content(&self, content: &str) -> bool {
        content.chars().any(|c| c.is_ascii_digit())
    }
}

/// Typing characteristics for different content patterns
#[derive(Debug, Clone)]
pub struct TypingCharacteristics {
    /// Speed factor for this pattern
    pub speed_factor: f32,
    /// Pause frequency multiplier
    pub pause_frequency: f32,
    /// Hesitation likelihood
    pub hesitation_likelihood: f32,
    /// Burst size preference
    pub preferred_burst_size: usize,
    /// Correction tendency
    pub correction_tendency: f32,
}

impl TypingCharacteristics {
    /// Characteristics for slow, careful typing
    pub fn slow_and_careful() -> Self {
        Self {
            speed_factor: 0.7,
            pause_frequency: 1.5,
            hesitation_likelihood: 0.3,
            preferred_burst_size: 5,
            correction_tendency: 0.1,
        }
    }

    /// Characteristics for variable typing with emotional pauses
    pub fn variable_with_pauses() -> Self {
        Self {
            speed_factor: 1.0,
            pause_frequency: 1.3,
            hesitation_likelihood: 0.2,
            preferred_burst_size: 6,
            correction_tendency: 0.15,
        }
    }

    /// Characteristics for thoughtful typing
    pub fn thoughtful() -> Self {
        Self {
            speed_factor: 0.9,
            pause_frequency: 1.2,
            hesitation_likelihood: 0.25,
            preferred_burst_size: 7,
            correction_tendency: 0.12,
        }
    }

    /// Characteristics for deliberate typing
    pub fn deliberate() -> Self {
        Self {
            speed_factor: 0.8,
            pause_frequency: 1.4,
            hesitation_likelihood: 0.2,
            preferred_burst_size: 8,
            correction_tendency: 0.08,
        }
    }

    /// Characteristics for fast, natural typing
    pub fn fast_and_natural() -> Self {
        Self {
            speed_factor: 1.3,
            pause_frequency: 0.8,
            hesitation_likelihood: 0.1,
            preferred_burst_size: 10,
            correction_tendency: 0.2,
        }
    }
}

// ================================================================================================
// SUPPORTING TYPES AND STRUCTURES
// ================================================================================================

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

/// Performance tracking for typing simulation
#[derive(Debug)]
pub struct PerformanceTracker {
    burst_generation_times: Mutex<VecDeque<Duration>>,
    analysis_times: Mutex<VecDeque<Duration>>,
    total_bursts_generated: Mutex<usize>,
    total_content_processed: Mutex<usize>,
}

impl Default for PerformanceTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceTracker {
    pub fn new() -> Self {
        Self {
            burst_generation_times: Mutex::new(VecDeque::new()),
            analysis_times: Mutex::new(VecDeque::new()),
            total_bursts_generated: Mutex::new(0),
            total_content_processed: Mutex::new(0),
        }
    }

    pub fn record_burst_generation(&self, burst_count: usize, content_length: usize) {
        if let (Ok(mut bursts), Ok(mut content)) = (
            self.total_bursts_generated.lock(),
            self.total_content_processed.lock(),
        ) {
            *bursts += burst_count;
            *content += content_length;
        }
    }

    pub fn get_metrics(&self) -> PerformanceMetrics {
        let (total_bursts, total_content) = if let (Ok(bursts), Ok(content)) = (
            self.total_bursts_generated.lock(),
            self.total_content_processed.lock(),
        ) {
            (*bursts, *content)
        } else {
            (0, 0)
        };

        PerformanceMetrics {
            total_bursts_generated: total_bursts,
            total_content_processed: total_content,
            average_burst_size: if total_bursts > 0 {
                total_content as f32 / total_bursts as f32
            } else {
                0.0
            },
        }
    }

    pub fn analyze_patterns(&self) -> TypingPatternSummary {
        TypingPatternSummary {
            efficiency_score: 0.85,   // Placeholder calculation
            consistency_score: 0.92,  // Placeholder calculation
            adaptability_score: 0.78, // Placeholder calculation
        }
    }
}

/// Performance metrics for typing simulation
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_bursts_generated: usize,
    pub total_content_processed: usize,
    pub average_burst_size: f32,
}

/// Typing pattern summary for analysis
#[derive(Debug, Clone)]
pub struct TypingPatternSummary {
    pub efficiency_score: f32,
    pub consistency_score: f32,
    pub adaptability_score: f32,
}

/// Pattern learning entry for continuous improvement
#[derive(Debug, Clone)]
pub struct PatternLearningEntry {
    pub content_snippet: String,
    pub analysis_result: TypingAnalysis,
    pub timestamp: Instant,
}

/// Learning statistics for pattern analysis
#[derive(Debug, Clone, Default)]
pub struct LearningStats {
    pub total_analyses: usize,
    pub pattern_distribution: HashMap<String, usize>,
    pub complexity_trends: Vec<f32>,
}
