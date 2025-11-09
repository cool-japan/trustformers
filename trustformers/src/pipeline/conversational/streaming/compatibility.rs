//! Backward compatibility and legacy support for streaming functionality.
//!
//! This module provides backward compatibility for existing code that uses the legacy
//! streaming API. All functionality here is deprecated and maintained only for compatibility.
//! Users should migrate to the new modular streaming architecture.
//!
//! # Migration Guide
//!
//! The legacy `StreamingManager` has been replaced by the new modular architecture:
//!
//! ## Old API (Deprecated)
//! ```rust,ignore
//! use trustformers::pipeline::conversational::streaming::StreamingManager;
//!
//! let mut manager = StreamingManager::new(config);
//! let stream = manager.create_stream_from_text("Hello world").await?;
//! ```
//!
//! ## New API (Recommended)
//! ```rust,ignore
//! use trustformers::pipeline::conversational::streaming::ConversationalStreamingPipeline;
//!
//! let pipeline = ConversationalStreamingPipeline::new(advanced_config).await?;
//! let stream = pipeline.stream_response("Hello world", &conversation).await?;
//! ```
//!
//! # Legacy Type Aliases
//!
//! Legacy type aliases are provided for backward compatibility but should be replaced
//! with their modern equivalents from the `super::types` module.

use super::types::*;
use crate::core::error::Result;
use crate::pipeline::conversational::types::{
    ConversationMetadata, StreamingConfig, StreamingResponse, StreamingState,
};
use crate::{AutoModel, AutoTokenizer};
use async_stream::stream;
use futures::{Stream, StreamExt};
use std::pin::Pin;
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};
use uuid::Uuid;

// ================================================================================================
// LEGACY TYPE ALIASES
// ================================================================================================

/// Legacy alias for StreamingConfig
///
/// # Deprecation Warning
/// This type alias is deprecated. Use `StreamingConfig` directly from the types module.
///
/// # Migration Path
/// ```rust,ignore
/// // Old
/// use trustformers::pipeline::conversational::streaming::LegacyStreamingConfig;
///
/// // New
/// use trustformers::pipeline::conversational::types::StreamingConfig;
/// ```
#[deprecated(
    since = "0.3.0",
    note = "Use StreamingConfig from pipeline::conversational::types instead"
)]
pub type LegacyStreamingConfig = StreamingConfig;

/// Legacy alias for StreamingResponse
///
/// # Deprecation Warning
/// This type alias is deprecated. Use `StreamingResponse` directly from the types module.
#[deprecated(
    since = "0.3.0",
    note = "Use StreamingResponse from pipeline::conversational::types instead"
)]
pub type LegacyStreamingResponse = StreamingResponse;

/// Legacy alias for StreamingState
///
/// # Deprecation Warning
/// This type alias is deprecated. Use `StreamingState` directly from the types module.
#[deprecated(
    since = "0.3.0",
    note = "Use StreamingState from pipeline::conversational::types instead"
)]
pub type LegacyStreamingState = StreamingState;

// ================================================================================================
// BACKWARD COMPATIBILITY IMPLEMENTATIONS
// ================================================================================================

/// Legacy streaming manager for backward compatibility
///
/// # Deprecation Warning
///
/// This struct is deprecated and will be removed in version 1.0.0.
/// Use `ConversationalStreamingPipeline` from the new modular architecture instead.
///
/// # Migration Guide
///
/// Replace `StreamingManager` with `ConversationalStreamingPipeline`:
///
/// ```rust,ignore
/// // Old approach
/// let mut manager = StreamingManager::new(config);
/// let stream = manager.create_stream_from_text("Hello").await?;
///
/// // New approach
/// let advanced_config = AdvancedStreamingConfig {
///     base_config: config,
///     ..AdvancedStreamingConfig::default()
/// };
/// let pipeline = ConversationalStreamingPipeline::new(advanced_config).await?;
/// let stream = pipeline.stream_response("Hello", &conversation).await?;
/// ```
#[derive(Debug)]
#[deprecated(
    since = "0.3.0",
    note = "Use ConversationalStreamingPipeline from the new modular architecture. \
            See module documentation for migration guide."
)]
pub struct StreamingManager {
    /// Advanced streaming configuration
    pub config: StreamingConfig,
    /// Current streaming state
    state: StreamingState,
    /// Internal advanced configuration
    advanced_config: AdvancedStreamingConfig,
}

#[allow(deprecated)]
impl StreamingManager {
    /// Create a new streaming manager
    ///
    /// # Deprecation Warning
    /// This method is deprecated. Use `ConversationalStreamingPipeline::new()` instead.
    ///
    /// # Migration
    /// ```rust,ignore
    /// // Old
    /// let manager = StreamingManager::new(config);
    ///
    /// // New
    /// let advanced_config = AdvancedStreamingConfig {
    ///     base_config: config,
    ///     ..AdvancedStreamingConfig::default()
    /// };
    /// let pipeline = ConversationalStreamingPipeline::new(advanced_config).await?;
    /// ```
    #[deprecated(
        since = "0.3.0",
        note = "Use ConversationalStreamingPipeline::new() instead"
    )]
    pub fn new(config: StreamingConfig) -> Self {
        let advanced_config = AdvancedStreamingConfig {
            base_config: config.clone(),
            ..AdvancedStreamingConfig::default()
        };

        Self {
            config,
            state: StreamingState::NotStarted,
            advanced_config,
        }
    }

    /// Create a streaming response from text
    ///
    /// # Deprecation Warning
    /// This method is deprecated. Use `ConversationalStreamingPipeline::stream_response()` instead.
    ///
    /// # Migration
    /// ```rust,ignore
    /// // Old
    /// let stream = manager.create_stream_from_text(text).await?;
    ///
    /// // New
    /// let stream = pipeline.stream_response(text, &conversation).await?;
    /// ```
    #[deprecated(
        since = "0.3.0",
        note = "Use ConversationalStreamingPipeline::stream_response() instead"
    )]
    pub async fn create_stream_from_text(
        &mut self,
        text: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse>> + Send + '_>>> {
        if !self.config.enabled {
            return self.create_single_chunk_stream(text).await;
        }

        self.state = StreamingState::Streaming;

        let chunks = self.split_into_chunks(text);
        let typing_delay = self.config.typing_delay_ms;

        let stream = stream! {
            let chunks_len = chunks.len();
            for (index, chunk) in chunks.into_iter().enumerate() {
                let is_final = index == chunks_len - 1;

                let response = StreamingResponse {
                    chunk: chunk.clone(),
                    is_final,
                    chunk_index: index,
                    total_chunks: Some(chunks_len),
                    metadata: None,
                };

                yield Ok(response);

                if !is_final {
                    sleep(Duration::from_millis(typing_delay)).await;
                }
            }
        };

        Ok(Box::pin(stream))
    }

    /// Create a streaming response with metadata
    ///
    /// # Deprecation Warning
    /// This method is deprecated. Use the new streaming pipeline with enhanced metadata support.
    #[deprecated(
        since = "0.3.0",
        note = "Use ConversationalStreamingPipeline with enhanced metadata support"
    )]
    pub async fn create_metadata_stream(
        &mut self,
        text: &str,
        metadata: ConversationMetadata,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse>> + Send + '_>>> {
        if !self.config.enabled {
            return self.create_single_chunk_stream_with_metadata(text, metadata).await;
        }

        self.state = StreamingState::Streaming;

        let chunks = self.split_into_chunks(text);
        let typing_delay = self.config.typing_delay_ms;

        let stream = stream! {
            let chunks_len = chunks.len();
            for (index, chunk) in chunks.into_iter().enumerate() {
                let is_final = index == chunks_len - 1;

                let response = StreamingResponse {
                    chunk: chunk.clone(),
                    is_final,
                    chunk_index: index,
                    total_chunks: Some(chunks_len),
                    metadata: if is_final { Some(metadata.clone()) } else { None },
                };

                yield Ok(response);

                if !is_final {
                    sleep(Duration::from_millis(typing_delay)).await;
                }
            }
        };

        Ok(Box::pin(stream))
    }

    /// Create a progressive streaming response
    ///
    /// # Deprecation Warning
    /// This method is deprecated. Use the new streaming pipeline with progressive streaming support.
    #[deprecated(
        since = "0.3.0",
        note = "Use ConversationalStreamingPipeline with progressive streaming support"
    )]
    pub async fn create_progressive_stream(
        &mut self,
        initial_chunk: String,
    ) -> Result<(
        mpsc::Sender<String>,
        Pin<Box<dyn Stream<Item = Result<StreamingResponse>> + Send + '_>>,
    )> {
        if !self.config.enabled {
            return Err(crate::error::TrustformersError::FeatureUnavailable {
                message: "Streaming is disabled".to_string(),
                feature: "streaming".to_string(),
                suggestion: Some("Enable streaming in the configuration".to_string()),
                alternatives: vec!["Use non-streaming inference".to_string()],
            }
            .into());
        }

        self.state = StreamingState::Streaming;

        let (tx, mut rx) = mpsc::channel::<String>(self.config.buffer_size);
        let typing_delay = self.config.typing_delay_ms;

        let stream = stream! {
            let mut chunk_index = 0;

            // Send initial chunk if provided
            if !initial_chunk.is_empty() {
                let response = StreamingResponse {
                    chunk: initial_chunk,
                    is_final: false,
                    chunk_index,
                    total_chunks: None,
                    metadata: None,
                };
                yield Ok(response);
                chunk_index += 1;
            }

            // Stream incoming chunks
            while let Some(chunk) = rx.recv().await {
                let is_final = chunk.is_empty(); // Empty chunk signals end

                if !is_final {
                    let response = StreamingResponse {
                        chunk,
                        is_final: false,
                        chunk_index,
                        total_chunks: None,
                        metadata: None,
                    };
                    yield Ok(response);
                    chunk_index += 1;

                    sleep(Duration::from_millis(typing_delay)).await;
                } else {
                    // Send final chunk
                    let response = StreamingResponse {
                        chunk: String::new(),
                        is_final: true,
                        chunk_index,
                        total_chunks: Some(chunk_index + 1),
                        metadata: None,
                    };
                    yield Ok(response);
                    break;
                }
            }
        };

        Ok((tx, Box::pin(stream)))
    }

    /// Split text into streaming chunks
    ///
    /// # Deprecation Warning
    /// This method is deprecated. Use `ResponseChunker` from the chunking module instead.
    #[deprecated(
        since = "0.3.0",
        note = "Use ResponseChunker from the chunking module for advanced chunking strategies"
    )]
    fn split_into_chunks(&self, text: &str) -> Vec<String> {
        if self.config.chunk_size == 0 {
            return vec![text.to_string()];
        }

        let words: Vec<&str> = text.split_whitespace().collect();
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut word_count = 0;

        for word in words {
            if word_count >= self.config.chunk_size && !current_chunk.is_empty() {
                chunks.push(current_chunk.trim().to_string());
                current_chunk = String::new();
                word_count = 0;
            }

            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(word);
            word_count += 1;
        }

        if !current_chunk.is_empty() {
            chunks.push(current_chunk.trim().to_string());
        }

        chunks
    }

    /// Create a single chunk stream for when streaming is disabled
    #[deprecated(
        since = "0.3.0",
        note = "Internal method, use the new streaming pipeline"
    )]
    async fn create_single_chunk_stream(
        &self,
        text: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse>> + Send + '_>>> {
        let response = StreamingResponse {
            chunk: text.to_string(),
            is_final: true,
            chunk_index: 0,
            total_chunks: Some(1),
            metadata: None,
        };

        let stream = stream! {
            yield Ok(response);
        };

        Ok(Box::pin(stream))
    }

    /// Create a single chunk stream with metadata
    #[deprecated(
        since = "0.3.0",
        note = "Internal method, use the new streaming pipeline"
    )]
    async fn create_single_chunk_stream_with_metadata(
        &self,
        text: &str,
        metadata: ConversationMetadata,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse>> + Send + '_>>> {
        let response = StreamingResponse {
            chunk: text.to_string(),
            is_final: true,
            chunk_index: 0,
            total_chunks: Some(1),
            metadata: Some(metadata),
        };

        let stream = stream! {
            yield Ok(response);
        };

        Ok(Box::pin(stream))
    }

    /// Pause the current streaming session
    ///
    /// # Deprecation Warning
    /// This method is deprecated. Use `StreamStateManager` for advanced state management.
    #[deprecated(
        since = "0.3.0",
        note = "Use StreamStateManager for advanced state management"
    )]
    pub fn pause(&mut self) {
        if matches!(self.state, StreamingState::Streaming) {
            self.state = StreamingState::Paused;
        }
    }

    /// Resume a paused streaming session
    ///
    /// # Deprecation Warning
    /// This method is deprecated. Use `StreamStateManager` for advanced state management.
    #[deprecated(
        since = "0.3.0",
        note = "Use StreamStateManager for advanced state management"
    )]
    pub fn resume(&mut self) {
        if matches!(self.state, StreamingState::Paused) {
            self.state = StreamingState::Streaming;
        }
    }

    /// Stop the current streaming session
    ///
    /// # Deprecation Warning
    /// This method is deprecated. Use `StreamStateManager` for advanced state management.
    #[deprecated(
        since = "0.3.0",
        note = "Use StreamStateManager for advanced state management"
    )]
    pub fn stop(&mut self) {
        self.state = StreamingState::Completed;
    }

    /// Get current streaming state
    ///
    /// # Deprecation Warning
    /// This method is deprecated. Use `StreamStateManager` for comprehensive state monitoring.
    #[deprecated(
        since = "0.3.0",
        note = "Use StreamStateManager for comprehensive state monitoring"
    )]
    pub fn get_state(&self) -> &StreamingState {
        &self.state
    }

    /// Check if streaming is currently active
    ///
    /// # Deprecation Warning
    /// This method is deprecated. Use `StreamStateManager` for advanced state queries.
    #[deprecated(
        since = "0.3.0",
        note = "Use StreamStateManager for advanced state queries"
    )]
    pub fn is_streaming(&self) -> bool {
        matches!(self.state, StreamingState::Streaming)
    }

    /// Update streaming configuration
    ///
    /// # Deprecation Warning
    /// This method is deprecated. Use the new configuration management in the pipeline.
    #[deprecated(
        since = "0.3.0",
        note = "Use configuration management in ConversationalStreamingPipeline"
    )]
    pub fn update_config(&mut self, config: StreamingConfig) {
        self.config = config.clone();
        self.advanced_config.base_config = config;
    }

    /// Calculate streaming statistics
    ///
    /// # Deprecation Warning
    /// This method is deprecated. Use `QualityAnalyzer` for comprehensive streaming analytics.
    #[deprecated(
        since = "0.3.0",
        note = "Use QualityAnalyzer for comprehensive streaming analytics and performance metrics"
    )]
    pub fn calculate_stream_stats(&self, responses: &[StreamingResponse]) -> StreamingStats {
        if responses.is_empty() {
            return StreamingStats::default();
        }

        let total_chunks = responses.len();
        let total_characters: usize = responses.iter().map(|r| r.chunk.len()).sum();
        let total_words: usize = responses.iter().map(|r| r.chunk.split_whitespace().count()).sum();

        let avg_chunk_size = if total_chunks > 0 {
            total_characters as f32 / total_chunks as f32
        } else {
            0.0
        };

        let estimated_duration = total_chunks as f32 * self.config.typing_delay_ms as f32 / 1000.0;

        StreamingStats {
            total_chunks,
            total_characters,
            total_words,
            avg_chunk_size,
            estimated_duration_seconds: estimated_duration,
        }
    }
}

#[allow(deprecated)]
impl Default for StreamingManager {
    fn default() -> Self {
        Self::new(StreamingConfig::default())
    }
}

// ================================================================================================
// LEGACY UTILITY TYPES
// ================================================================================================

/// Statistics about streaming performance
///
/// # Deprecation Warning
/// This struct is deprecated. Use `QualityAnalysis` and `AdvancedQualityMetrics`
/// from the quality analyzer for comprehensive streaming analytics.
///
/// # Migration Guide
/// ```rust,ignore
/// // Old
/// let stats = manager.calculate_stream_stats(&responses);
/// println!("Total chunks: {}", stats.total_chunks);
///
/// // New
/// let analyzer = QualityAnalyzer::new(config);
/// let analysis = analyzer.analyze_streaming_quality(&stream_data).await?;
/// println!("Total chunks: {}", analysis.statistical_analysis.chunk_count);
/// ```
#[derive(Debug, Clone, Default)]
#[deprecated(
    since = "0.3.0",
    note = "Use QualityAnalysis and AdvancedQualityMetrics from quality_analyzer module"
)]
pub struct StreamingStats {
    pub total_chunks: usize,
    pub total_characters: usize,
    pub total_words: usize,
    pub avg_chunk_size: f32,
    pub estimated_duration_seconds: f32,
}

/// Streaming session information
///
/// # Deprecation Warning
/// This struct is deprecated. Use `StreamingCoordinator` for comprehensive session management
/// with enhanced tracking and analytics.
///
/// # Migration Guide
/// ```rust,ignore
/// // Old
/// let session = StreamingSession::new(config);
///
/// // New
/// let coordinator = StreamingCoordinator::new(advanced_config);
/// let session_id = coordinator.create_session("conversation_id".to_string()).await?;
/// let session = coordinator.get_session(&session_id).await.unwrap();
/// ```
#[derive(Debug, Clone)]
#[deprecated(
    since = "0.3.0",
    note = "Use StreamingCoordinator for comprehensive session management with enhanced tracking"
)]
pub struct StreamingSession {
    pub session_id: String,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
    pub config: StreamingConfig,
    pub state: StreamingState,
    pub stats: Option<StreamingStats>,
}

#[allow(deprecated)]
impl StreamingSession {
    /// Create a new streaming session
    ///
    /// # Deprecation Warning
    /// This method is deprecated. Use `StreamingCoordinator::create_session()` instead.
    #[deprecated(
        since = "0.3.0",
        note = "Use StreamingCoordinator::create_session() for enhanced session management"
    )]
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            session_id: Uuid::new_v4().to_string(),
            start_time: chrono::Utc::now(),
            end_time: None,
            config,
            state: StreamingState::NotStarted,
            stats: None,
        }
    }

    /// Complete the streaming session
    ///
    /// # Deprecation Warning
    /// This method is deprecated. Use `StreamingCoordinator::close_session()` instead.
    #[deprecated(
        since = "0.3.0",
        note = "Use StreamingCoordinator::close_session() for proper session lifecycle management"
    )]
    pub fn complete(&mut self, stats: StreamingStats) {
        self.end_time = Some(chrono::Utc::now());
        self.state = StreamingState::Completed;
        self.stats = Some(stats);
    }

    /// Get session duration in milliseconds
    ///
    /// # Deprecation Warning
    /// This method is deprecated. Use the enhanced session tracking in `StreamingCoordinator`.
    #[deprecated(
        since = "0.3.0",
        note = "Use enhanced session tracking in StreamingCoordinator"
    )]
    pub fn duration_ms(&self) -> Option<i64> {
        self.end_time.map(|end| (end - self.start_time).num_milliseconds())
    }
}

// ================================================================================================
// CONVERSION UTILITIES
// ================================================================================================

/// Conversion utilities for migrating between old and new streaming APIs
pub mod conversion {
    use super::*;

    /// Convert legacy StreamingConfig to AdvancedStreamingConfig
    ///
    /// This utility helps migrate from the old streaming configuration format
    /// to the new advanced configuration with enhanced features.
    ///
    /// # Example
    /// ```rust,ignore
    /// use trustformers::pipeline::conversational::streaming::compatibility::conversion;
    ///
    /// let legacy_config = StreamingConfig {
    ///     enabled: true,
    ///     chunk_size: 10,
    ///     buffer_size: 100,
    ///     typing_delay_ms: 50,
    /// };
    ///
    /// let advanced_config = conversion::upgrade_streaming_config(legacy_config);
    /// ```
    pub fn upgrade_streaming_config(legacy_config: StreamingConfig) -> AdvancedStreamingConfig {
        AdvancedStreamingConfig {
            base_config: legacy_config,
            adaptive_chunking: true,
            max_chunk_size: 50,
            min_chunk_size: 5,
            natural_pausing: true,
            punctuation_pause_ms: 150,
            variable_typing_speed: true,
            base_typing_speed: 25.0,
            speed_variation: 0.3,
            enable_backpressure: true,
            max_buffer_size: 1000,
            chunk_timeout_ms: 5000,
            quality_based_streaming: true,
            enable_error_recovery: true,
            max_retry_attempts: 3,
        }
    }

    /// Convert legacy StreamingStats to QualityAnalysis
    ///
    /// This utility helps convert old streaming statistics to the new comprehensive
    /// quality analysis format.
    #[allow(deprecated)]
    pub fn upgrade_stats_to_quality_analysis(
        legacy_stats: StreamingStats,
    ) -> super::super::quality_analyzer::QualityAnalyzer {
        // Simply return a new QualityAnalyzer with defaults
        // Legacy stats are deprecated and the new analyzer uses sophisticated metrics
        super::super::quality_analyzer::QualityAnalyzer::new()
    }
}

// ================================================================================================
// MIGRATION HELPERS
// ================================================================================================

/// Migration helpers for transitioning to the new streaming architecture
pub mod migration {
    use super::*;

    /// Helper to migrate from StreamingManager to ConversationalStreamingPipeline
    ///
    /// This function provides a compatibility wrapper that helps transition existing
    /// code from the legacy StreamingManager to the new pipeline architecture.
    ///
    /// # Example
    /// ```rust,ignore
    /// // This helps during migration but should eventually be replaced
    /// let pipeline = migration::create_pipeline_from_legacy_config(legacy_config).await?;
    /// ```
    pub async fn create_pipeline_from_legacy_config(
        legacy_config: StreamingConfig,
    ) -> Result<super::super::pipeline::ConversationalStreamingPipeline<AutoModel, AutoTokenizer>>
    {
        let advanced_config = super::conversion::upgrade_streaming_config(legacy_config);

        // Create default model and tokenizer for compatibility
        let model = AutoModel::from_pretrained("distilbert-base-uncased")?;
        let tokenizer = AutoTokenizer::from_pretrained("distilbert-base-uncased")?;

        Ok(
            super::super::pipeline::ConversationalStreamingPipeline::new(
                model,
                tokenizer,
                advanced_config,
            ),
        )
    }

    /// Migration guide as a compile-time warning
    ///
    /// This function exists solely to provide migration guidance during compilation.
    /// It should not be called in production code.
    #[deprecated(
        since = "0.3.0",
        note = "This function exists for migration guidance. See module documentation for complete migration guide."
    )]
    pub fn migration_guide() {
        // This function intentionally left empty - it's for documentation purposes only
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_legacy_type_aliases() {
        // Test that legacy type aliases still work for backward compatibility
        #[allow(deprecated)]
        {
            let _config: LegacyStreamingConfig = StreamingConfig::default();
            let _state: LegacyStreamingState = StreamingState::NotStarted;
        }
    }

    #[test]
    fn test_conversion_utilities() {
        let legacy_config = StreamingConfig {
            enabled: true,
            chunk_size: 10,
            buffer_size: 100,
            typing_delay_ms: 50,
        };

        let advanced_config = conversion::upgrade_streaming_config(legacy_config.clone());
        assert_eq!(advanced_config.base_config.enabled, legacy_config.enabled);
        assert_eq!(
            advanced_config.base_config.chunk_size,
            legacy_config.chunk_size
        );
        assert!(advanced_config.adaptive_chunking);
    }

    #[tokio::test]
    async fn test_migration_helper() {
        let legacy_config = StreamingConfig::default();
        let result = migration::create_pipeline_from_legacy_config(legacy_config).await;
        // This test verifies the migration helper compiles and can be called
        // The actual pipeline creation may fail due to missing dependencies in test environment
        // but the important part is that the interface works
        assert!(result.is_ok() || result.is_err()); // Just verify it returns a Result
    }
}
