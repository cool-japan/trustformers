//! Response chunking and token streaming implementation for conversational AI.
//!
//! This module provides comprehensive chunking functionality for streaming responses,
//! including multiple chunking strategies, adaptive sizing, quality assessment,
//! and natural delivery timing.

use super::types::*;
use crate::pipeline::conversational::types::{ConversationMetadata, ReasoningType};
use std::collections::HashSet;
use std::time::{Duration, Instant};

// ================================================================================================
// ADDITIONAL TYPES FOR CHUNKING MODULE
// ================================================================================================

/// Direction of quality trends
#[derive(Debug, Clone, PartialEq, Default)]
pub enum TrendDirection {
    Improving,
    #[default]
    Stable,
    Declining,
}

/// Simple quality trends for chunking analysis
#[derive(Debug, Clone)]
pub struct SimpleQualityTrends {
    pub trend_direction: TrendDirection,
    pub recent_average: f32,
    pub change_magnitude: f32,
}

impl Default for SimpleQualityTrends {
    fn default() -> Self {
        Self {
            trend_direction: TrendDirection::Stable,
            recent_average: 0.8,
            change_magnitude: 0.0,
        }
    }
}

// ================================================================================================
// RESPONSE CHUNKER IMPLEMENTATION
// ================================================================================================

/// Advanced response chunker with multiple strategies
#[derive(Debug)]
pub struct ResponseChunker {
    /// Chunking strategy
    strategy: ChunkingStrategy,
    /// Configuration
    config: AdvancedStreamingConfig,
    /// Quality analyzer
    quality_analyzer: QualityAnalyzer,
}

impl ResponseChunker {
    /// Create a new response chunker
    pub fn new(strategy: ChunkingStrategy, config: AdvancedStreamingConfig) -> Self {
        Self {
            strategy,
            config,
            quality_analyzer: QualityAnalyzer::new(),
        }
    }

    /// Chunk response text into streaming pieces
    pub fn chunk_response(&self, text: &str, metadata: &ConversationMetadata) -> Vec<StreamChunk> {
        match &self.strategy {
            ChunkingStrategy::FixedSize(size) => self.chunk_fixed_size(text, *size),
            ChunkingStrategy::WordBoundary => self.chunk_word_boundary(text),
            ChunkingStrategy::SentenceBoundary => self.chunk_sentence_boundary(text),
            ChunkingStrategy::Adaptive => self.chunk_adaptive(text, metadata),
            ChunkingStrategy::Semantic => self.chunk_semantic(text, metadata),
        }
    }

    /// Fixed size chunking
    fn chunk_fixed_size(&self, text: &str, chunk_size: usize) -> Vec<StreamChunk> {
        let mut chunks = Vec::new();
        let mut chunk_index = 0;

        for chunk_text in text.chars().collect::<Vec<_>>().chunks(chunk_size) {
            let chunk_str: String = chunk_text.iter().collect();
            chunks.push(StreamChunk {
                content: chunk_str,
                index: chunk_index,
                chunk_type: ChunkType::Content,
                timing: ChunkTiming::default(),
                metadata: ChunkMetadata::default(),
            });
            chunk_index += 1;
        }

        chunks
    }

    /// Word boundary chunking for natural word breaks
    fn chunk_word_boundary(&self, text: &str) -> Vec<StreamChunk> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut chunks = Vec::new();
        let mut chunk_index = 0;
        let chunk_size = self.config.base_config.chunk_size;

        for word_chunk in words.chunks(chunk_size) {
            let chunk_text = word_chunk.join(" ");
            chunks.push(StreamChunk {
                content: chunk_text + " ",
                index: chunk_index,
                chunk_type: ChunkType::Content,
                timing: ChunkTiming::default(),
                metadata: ChunkMetadata::default(),
            });
            chunk_index += 1;
        }

        chunks
    }

    /// Sentence boundary chunking for natural sentence breaks
    fn chunk_sentence_boundary(&self, text: &str) -> Vec<StreamChunk> {
        let sentences = self.split_sentences(text);
        let mut chunks = Vec::new();
        let mut chunk_index = 0;

        for sentence in sentences {
            chunks.push(StreamChunk {
                content: sentence,
                index: chunk_index,
                chunk_type: ChunkType::Sentence,
                timing: ChunkTiming::with_pause(self.config.punctuation_pause_ms),
                metadata: ChunkMetadata::default(),
            });
            chunk_index += 1;
        }

        chunks
    }

    /// Adaptive chunking based on content analysis
    fn chunk_adaptive(&self, text: &str, metadata: &ConversationMetadata) -> Vec<StreamChunk> {
        let mut chunks = Vec::new();
        let mut chunk_index = 0;

        // Analyze content complexity
        let complexity = self.analyze_content_complexity(text, metadata);

        // Adjust chunk size based on complexity
        let base_size = self.config.base_config.chunk_size;
        let adjusted_size = if complexity > 0.7 {
            (base_size as f32 * 0.7) as usize // Smaller chunks for complex content
        } else if complexity < 0.3 {
            (base_size as f32 * 1.3) as usize // Larger chunks for simple content
        } else {
            base_size
        }
        .max(self.config.min_chunk_size)
        .min(self.config.max_chunk_size);

        // Use word boundary chunking with adjusted size
        let words: Vec<&str> = text.split_whitespace().collect();
        for word_chunk in words.chunks(adjusted_size) {
            let chunk_text = word_chunk.join(" ");
            chunks.push(StreamChunk {
                content: chunk_text + " ",
                index: chunk_index,
                chunk_type: ChunkType::Adaptive,
                timing: ChunkTiming::adaptive(complexity),
                metadata: ChunkMetadata::with_complexity(complexity),
            });
            chunk_index += 1;
        }

        chunks
    }

    /// Semantic chunking based on meaning and structure
    fn chunk_semantic(&self, text: &str, metadata: &ConversationMetadata) -> Vec<StreamChunk> {
        // Split by paragraphs first for basic semantic structure
        let paragraphs: Vec<&str> = text.split("\n\n").collect();
        let mut chunks = Vec::new();
        let mut chunk_index = 0;

        for paragraph in paragraphs {
            if paragraph.trim().is_empty() {
                continue;
            }

            // Further split long paragraphs to maintain readability
            if paragraph.len() > self.config.max_chunk_size * 2 {
                let sentences = self.split_sentences(paragraph);
                let mut current_chunk = String::new();

                for sentence in sentences {
                    if current_chunk.len() + sentence.len() > self.config.max_chunk_size {
                        if !current_chunk.is_empty() {
                            chunks.push(StreamChunk {
                                content: current_chunk.trim().to_string(),
                                index: chunk_index,
                                chunk_type: ChunkType::Semantic,
                                timing: ChunkTiming::default(),
                                metadata: ChunkMetadata::semantic(),
                            });
                            chunk_index += 1;
                        }
                        current_chunk = sentence;
                    } else {
                        current_chunk.push_str(&sentence);
                    }
                }

                if !current_chunk.is_empty() {
                    chunks.push(StreamChunk {
                        content: current_chunk.trim().to_string(),
                        index: chunk_index,
                        chunk_type: ChunkType::Semantic,
                        timing: ChunkTiming::default(),
                        metadata: ChunkMetadata::semantic(),
                    });
                    chunk_index += 1;
                }
            } else {
                chunks.push(StreamChunk {
                    content: paragraph.trim().to_string(),
                    index: chunk_index,
                    chunk_type: ChunkType::Semantic,
                    timing: ChunkTiming::default(),
                    metadata: ChunkMetadata::semantic(),
                });
                chunk_index += 1;
            }
        }

        chunks
    }

    /// Split text into sentences using basic punctuation rules
    fn split_sentences(&self, text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current_sentence = String::new();

        for char in text.chars() {
            current_sentence.push(char);

            if matches!(char, '.' | '!' | '?') {
                let trimmed = current_sentence.trim();
                if !trimmed.is_empty() {
                    sentences.push(trimmed.to_string() + " ");
                }
                current_sentence.clear();
            }
        }

        if !current_sentence.trim().is_empty() {
            sentences.push(current_sentence.trim().to_string());
        }

        sentences
    }

    /// Analyze content complexity for adaptive chunking
    fn analyze_content_complexity(&self, text: &str, metadata: &ConversationMetadata) -> f32 {
        let mut complexity = 0.0;

        // Length factor - longer text tends to be more complex
        complexity += (text.len() as f32 / 1000.0).min(1.0) * 0.2;

        // Vocabulary complexity - more unique words indicate complexity
        let unique_words: HashSet<&str> = text.split_whitespace().collect();
        let vocab_ratio = unique_words.len() as f32 / text.split_whitespace().count().max(1) as f32;
        complexity += vocab_ratio * 0.3;

        // Technical content indicators
        let technical_indicators = [
            "algorithm",
            "implementation",
            "function",
            "method",
            "class",
            "variable",
        ];
        let technical_count = technical_indicators
            .iter()
            .map(|&term| text.to_lowercase().matches(term).count())
            .sum::<usize>();
        complexity += (technical_count as f32 / 10.0).min(1.0) * 0.3;

        // Reasoning type complexity
        if let Some(reasoning_type) = &metadata.reasoning_type {
            complexity += match reasoning_type {
                ReasoningType::Mathematical => 0.4,
                ReasoningType::Logical => 0.3,
                ReasoningType::Creative => 0.2,
                ReasoningType::Analogical => 0.25,
                ReasoningType::Causal => 0.2,
                ReasoningType::Emotional => 0.1,
            };
        }

        complexity.min(1.0)
    }

    /// Detect code blocks and handle them specially
    fn detect_code_blocks(&self, text: &str) -> Vec<CodeBlock> {
        let mut code_blocks = Vec::new();
        let lines: Vec<&str> = text.lines().collect();
        let mut in_code_block = false;
        let mut code_start = 0;
        let mut code_end = 0;

        for (i, line) in lines.iter().enumerate() {
            if line.trim().starts_with("```") {
                if in_code_block {
                    code_end = i;
                    code_blocks.push(CodeBlock {
                        start_line: code_start,
                        end_line: code_end,
                        language: detect_language(lines[code_start]),
                        content: lines[code_start..=code_end].join("\n"),
                    });
                    in_code_block = false;
                } else {
                    code_start = i;
                    in_code_block = true;
                }
            }
        }

        code_blocks
    }

    /// Handle structured data differently from prose
    fn detect_structured_data(&self, text: &str) -> Vec<StructuredBlock> {
        let mut structured_blocks = Vec::new();

        // Detect lists
        if self.is_list(text) {
            structured_blocks.push(StructuredBlock {
                block_type: StructuredType::List,
                content: text.to_string(),
                should_chunk_items: true,
            });
        }

        // Detect tables (simple markdown tables)
        if text.contains('|') && text.matches('|').count() > 2 {
            structured_blocks.push(StructuredBlock {
                block_type: StructuredType::Table,
                content: text.to_string(),
                should_chunk_items: false, // Keep tables intact
            });
        }

        structured_blocks
    }

    /// Check if text represents a list
    fn is_list(&self, text: &str) -> bool {
        let lines: Vec<&str> = text.lines().collect();
        if lines.len() < 2 {
            return false;
        }

        let list_indicators = ["-", "*", "+"];

        lines.iter().take(3).all(|line| {
            let trimmed = line.trim();
            list_indicators.iter().any(|&indicator| trimmed.starts_with(indicator))
                || self.is_numbered_list_item(trimmed)
        })
    }

    /// Check if a line is a numbered list item (e.g., "1.", "2.", etc.)
    fn is_numbered_list_item(&self, line: &str) -> bool {
        let chars: Vec<char> = line.chars().collect();
        if chars.len() < 2 {
            return false;
        }

        let mut digit_count = 0;
        for &ch in &chars {
            if ch.is_ascii_digit() {
                digit_count += 1;
            } else {
                return ch == '.' && digit_count > 0;
            }
        }
        false
    }

    /// Get access to quality analyzer for external use
    pub fn quality_analyzer(&self) -> &QualityAnalyzer {
        &self.quality_analyzer
    }

    /// Update chunking strategy dynamically
    pub fn update_strategy(&mut self, strategy: ChunkingStrategy) {
        self.strategy = strategy;
    }

    /// Update configuration
    pub fn update_config(&mut self, config: AdvancedStreamingConfig) {
        self.config = config;
    }

    /// Get current strategy
    pub fn current_strategy(&self) -> &ChunkingStrategy {
        &self.strategy
    }

    /// Get current config
    pub fn current_config(&self) -> &AdvancedStreamingConfig {
        &self.config
    }
}

// ================================================================================================
// QUALITY ANALYZER IMPLEMENTATION
// ================================================================================================

impl QualityAnalyzer {
    /// Analyze chunk quality
    pub async fn analyze_chunk_quality(
        &self,
        chunk: &StreamChunk,
        delivery_time: Duration,
    ) -> QualityMeasurement {
        let measurement = QualityMeasurement {
            timestamp: Instant::now(),
            smoothness: self.calculate_smoothness(chunk),
            naturalness: self.calculate_naturalness(chunk),
            responsiveness: self.calculate_responsiveness(delivery_time),
            coherence: self.calculate_coherence(chunk),
            latency_ms: delivery_time.as_millis() as f64,
            chunk_consistency: self.calculate_chunk_consistency(chunk).await,
        };

        // Add to metrics window
        let mut window = self.metrics_window().write().await;
        window.push_back(measurement.clone());

        // Keep window size
        if window.len() > self.window_size() {
            window.pop_front();
        }

        measurement
    }

    /// Calculate overall streaming quality
    pub async fn calculate_overall_quality(&self) -> StreamingQuality {
        let window = self.metrics_window().read().await;
        if window.is_empty() {
            return StreamingQuality::default();
        }

        let count = window.len() as f32;
        let smoothness = window.iter().map(|m| m.smoothness).sum::<f32>() / count;
        let naturalness = window.iter().map(|m| m.naturalness).sum::<f32>() / count;
        let responsiveness = window.iter().map(|m| m.responsiveness).sum::<f32>() / count;
        let coherence = window.iter().map(|m| m.coherence).sum::<f32>() / count;
        let overall_quality = (smoothness + naturalness + responsiveness + coherence) / 4.0;

        StreamingQuality {
            smoothness,
            naturalness,
            responsiveness,
            coherence,
            overall_quality,
        }
    }

    /// Check if quality meets thresholds
    pub async fn meets_quality_thresholds(&self) -> bool {
        let quality = self.calculate_overall_quality().await;
        quality.smoothness >= self.thresholds().min_smoothness
            && quality.naturalness >= self.thresholds().min_naturalness
            && quality.responsiveness >= self.thresholds().min_responsiveness
            && quality.coherence >= self.thresholds().min_coherence
            && quality.overall_quality >= self.thresholds().min_overall_quality
    }

    /// Calculate smoothness based on delivery timing
    fn calculate_smoothness(&self, chunk: &StreamChunk) -> f32 {
        // Simple smoothness calculation based on chunk properties
        let base_smoothness = 0.8;

        // Penalize very short or very long chunks
        let length_factor =
            if chunk.content.len() < 5 || chunk.content.len() > 100 { 0.8 } else { 1.0 };

        base_smoothness * length_factor
    }

    /// Calculate naturalness of chunk content and timing
    fn calculate_naturalness(&self, chunk: &StreamChunk) -> f32 {
        let mut naturalness: f32 = 0.8;

        // Check for natural word boundaries
        if chunk.content.ends_with(' ') || chunk.content.ends_with('\n') {
            naturalness += 0.1;
        }

        // Check for sentence completion
        if chunk.content.ends_with('.')
            || chunk.content.ends_with('!')
            || chunk.content.ends_with('?')
        {
            naturalness += 0.1;
        }

        // Penalize awkward breaks mid-word
        if !chunk.content.is_empty()
            && !chunk.content.chars().last().expect("non-empty content").is_whitespace()
        {
            let words: Vec<&str> = chunk.content.split_whitespace().collect();
            if let Some(last_word) = words.last() {
                if last_word.len() > 2 && !last_word.ends_with('.') && !last_word.ends_with(',') {
                    naturalness -= 0.2; // Likely mid-word break
                }
            }
        }

        naturalness.max(0.0).min(1.0)
    }

    /// Calculate responsiveness based on delivery timing
    fn calculate_responsiveness(&self, delivery_time: Duration) -> f32 {
        let target_latency = 100.0; // 100ms target
        let latency_ms = delivery_time.as_millis() as f32;

        if latency_ms <= target_latency {
            1.0
        } else {
            (target_latency / latency_ms).max(0.1)
        }
    }

    /// Calculate coherence of chunk in context
    fn calculate_coherence(&self, chunk: &StreamChunk) -> f32 {
        let mut coherence: f32 = 0.8;

        // Check chunk type consistency
        match chunk.chunk_type {
            ChunkType::Sentence => {
                if chunk.content.contains('.')
                    || chunk.content.contains('!')
                    || chunk.content.contains('?')
                {
                    coherence += 0.1;
                }
            },
            ChunkType::Semantic => {
                if chunk.content.contains('\n') || chunk.content.len() > 50 {
                    coherence += 0.1;
                }
            },
            _ => {},
        }

        coherence.max(0.0).min(1.0)
    }

    /// Calculate chunk consistency across the stream
    async fn calculate_chunk_consistency(&self, _chunk: &StreamChunk) -> f32 {
        let window = self.metrics_window().read().await;
        if window.len() < 5 {
            return 0.8; // Default for insufficient data
        }

        // Calculate variance in chunk sizes from recent history
        let recent_chunks: Vec<_> = window.iter().rev().take(10).collect();
        let sizes: Vec<f32> = recent_chunks.iter().map(|m| m.latency_ms as f32).collect();

        if sizes.is_empty() {
            return 0.8;
        }

        let mean_size = sizes.iter().sum::<f32>() / sizes.len() as f32;
        let variance =
            sizes.iter().map(|&size| (size - mean_size).powi(2)).sum::<f32>() / sizes.len() as f32;

        // Lower variance means higher consistency
        let consistency: f32 = 1.0 / (1.0 + variance / (mean_size * mean_size));
        consistency.max(0.0).min(1.0)
    }

    /// Get quality trends over time
    pub async fn get_quality_trends(&self) -> SimpleQualityTrends {
        let window = self.metrics_window().read().await;
        if window.len() < 10 {
            return SimpleQualityTrends::default();
        }

        let recent = &window.as_slices().0[window.len() - 5..];
        let earlier = &window.as_slices().0[window.len() - 10..window.len() - 5];

        let recent_avg = recent
            .iter()
            .map(|m| m.smoothness + m.naturalness + m.responsiveness + m.coherence)
            .sum::<f32>()
            / (recent.len() as f32 * 4.0);

        let earlier_avg = earlier
            .iter()
            .map(|m| m.smoothness + m.naturalness + m.responsiveness + m.coherence)
            .sum::<f32>()
            / (earlier.len() as f32 * 4.0);

        let trend_direction = if recent_avg > earlier_avg + 0.05 {
            TrendDirection::Improving
        } else if recent_avg < earlier_avg - 0.05 {
            TrendDirection::Declining
        } else {
            TrendDirection::Stable
        };

        SimpleQualityTrends {
            trend_direction,
            recent_average: recent_avg,
            change_magnitude: (recent_avg - earlier_avg).abs(),
        }
    }

    /// Clear metrics window
    pub async fn clear_metrics(&self) {
        let mut window = self.metrics_window().write().await;
        window.clear();
    }

    /// Update quality thresholds
    pub fn update_thresholds(&mut self, thresholds: QualityThresholds) {
        self.thresholds = thresholds;
    }

    /// Get current thresholds
    pub fn current_thresholds(&self) -> &QualityThresholds {
        &self.thresholds
    }
}

// ================================================================================================
// HELPER TYPES AND FUNCTIONS
// ================================================================================================

/// Code block information for special handling
#[derive(Debug, Clone)]
pub struct CodeBlock {
    pub start_line: usize,
    pub end_line: usize,
    pub language: String,
    pub content: String,
}

/// Structured data block for special chunking
#[derive(Debug, Clone)]
pub struct StructuredBlock {
    pub block_type: StructuredType,
    pub content: String,
    pub should_chunk_items: bool,
}

/// Types of structured content
#[derive(Debug, Clone, PartialEq)]
pub enum StructuredType {
    List,
    Table,
    CodeBlock,
    Quote,
}

/// Detect programming language from code block header
fn detect_language(line: &str) -> String {
    if line.starts_with("```") {
        let lang = line.trim_start_matches("```").trim();
        if lang.is_empty() {
            "text".to_string()
        } else {
            lang.to_string()
        }
    } else {
        "text".to_string()
    }
}

// ================================================================================================
// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::conversational::types::{ConversationMetadata, EngagementLevel};
    use chrono::Utc;

    #[test]
    fn test_fixed_size_chunking() {
        let config = AdvancedStreamingConfig::default();
        let chunker = ResponseChunker::new(ChunkingStrategy::FixedSize(10), config);
        let metadata = ConversationMetadata::default();

        let text = "This is a long sentence that should be split into multiple chunks";
        let chunks = chunker.chunk_response(text, &metadata);

        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|chunk| chunk.content.chars().count() <= 10));
    }

    #[test]
    fn test_word_boundary_chunking() {
        let config = AdvancedStreamingConfig::default();
        let chunker = ResponseChunker::new(ChunkingStrategy::WordBoundary, config);
        let metadata = ConversationMetadata::default();

        let text = "This is a test sentence for word boundary chunking";
        let chunks = chunker.chunk_response(text, &metadata);

        assert!(!chunks.is_empty());
        // Each chunk should end with a space (except possibly the last)
        for chunk in &chunks[..chunks.len() - 1] {
            assert!(chunk.content.ends_with(' '));
        }
    }

    #[test]
    fn test_sentence_boundary_chunking() {
        let config = AdvancedStreamingConfig::default();
        let chunker = ResponseChunker::new(ChunkingStrategy::SentenceBoundary, config);
        let metadata = ConversationMetadata::default();

        let text = "First sentence. Second sentence! Third sentence?";
        let chunks = chunker.chunk_response(text, &metadata);

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].chunk_type, ChunkType::Sentence);
        assert!(chunks[0].timing.pause_ms > 0);
    }

    #[tokio::test]
    async fn test_adaptive_chunking() {
        let config = AdvancedStreamingConfig::default();
        let chunker = ResponseChunker::new(ChunkingStrategy::Adaptive, config);

        // Simple metadata
        let simple_metadata = ConversationMetadata {
            entities: Vec::new(),
            quality_score: 0.5,
            engagement_level: EngagementLevel::Medium,
            reasoning_type: None,
            ..Default::default()
        };

        // Complex metadata
        let complex_metadata = ConversationMetadata {
            entities: Vec::new(),
            quality_score: 0.9,
            engagement_level: EngagementLevel::High,
            reasoning_type: Some(ReasoningType::Mathematical),
            ..Default::default()
        };

        let simple_text = "This is simple text.";
        let complex_text = "The algorithm implementation requires careful consideration of data structures and computational complexity analysis.";

        let simple_chunks = chunker.chunk_response(simple_text, &simple_metadata);
        let complex_chunks = chunker.chunk_response(complex_text, &complex_metadata);

        assert!(!simple_chunks.is_empty());
        assert!(!complex_chunks.is_empty());

        // Verify adaptive chunking produces different results
        for chunk in &complex_chunks {
            assert_eq!(chunk.chunk_type, ChunkType::Adaptive);
            assert!(chunk.metadata.complexity > 0.0);
        }
    }

    #[tokio::test]
    async fn test_quality_analyzer() {
        let analyzer = QualityAnalyzer::new();
        let chunk = StreamChunk {
            content: "This is a well-formed sentence.".to_string(),
            index: 0,
            chunk_type: ChunkType::Sentence,
            timing: ChunkTiming::default(),
            metadata: ChunkMetadata::default(),
        };

        let delivery_time = Duration::from_millis(100);
        let measurement = analyzer.analyze_chunk_quality(&chunk, delivery_time).await;

        assert!(measurement.smoothness > 0.0);
        assert!(measurement.naturalness > 0.0);
        assert!(measurement.responsiveness > 0.0);
        assert!(measurement.coherence > 0.0);
        assert_eq!(measurement.latency_ms, 100.0);

        let overall_quality = analyzer.calculate_overall_quality().await;
        assert!(overall_quality.overall_quality > 0.0);
    }

    #[test]
    fn test_content_complexity_analysis() {
        let config = AdvancedStreamingConfig::default();
        let chunker = ResponseChunker::new(ChunkingStrategy::Adaptive, config);

        let simple_metadata = ConversationMetadata::default();
        let technical_metadata = ConversationMetadata {
            reasoning_type: Some(ReasoningType::Mathematical),
            ..Default::default()
        };

        let simple_text = "Hello world";
        let technical_text =
            "The algorithm implementation requires function optimization and variable analysis";

        let simple_complexity = chunker.analyze_content_complexity(simple_text, &simple_metadata);
        let technical_complexity =
            chunker.analyze_content_complexity(technical_text, &technical_metadata);

        assert!(technical_complexity > simple_complexity);
        assert!(technical_complexity > 0.3); // Should be fairly complex
    }

    #[test]
    fn test_sentence_splitting() {
        let config = AdvancedStreamingConfig::default();
        let chunker = ResponseChunker::new(ChunkingStrategy::SentenceBoundary, config);

        let text = "First sentence. Second sentence! Third sentence? Fourth sentence";
        let sentences = chunker.split_sentences(text);

        assert_eq!(sentences.len(), 4);
        assert!(sentences[0].contains("First sentence."));
        assert!(sentences[1].contains("Second sentence!"));
        assert!(sentences[2].contains("Third sentence?"));
        assert!(sentences[3].contains("Fourth sentence"));
    }
}
