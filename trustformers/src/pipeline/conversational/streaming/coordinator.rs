//! Streaming coordinator and manager implementations for conversational AI pipeline.
//!
//! This module provides the core coordination logic for streaming responses, including
//! session management, resource coordination, and event handling for natural
//! conversational experiences.

use super::super::types::*;
use super::types::*;
use crate::error::{Result, TrustformersError};
use async_stream::stream;
use futures::{Stream, StreamExt};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tokio::time::sleep;
use uuid::Uuid;

// ================================================================================================
// STREAMING COORDINATOR AND MANAGER
// ================================================================================================

/// Central coordinator for streaming responses
#[derive(Debug)]
pub struct StreamingCoordinator {
    /// Configuration for streaming
    config: AdvancedStreamingConfig,
    /// Active streams registry
    active_streams: Arc<RwLock<HashMap<String, StreamSession>>>,
    /// Global metrics
    global_metrics: Arc<RwLock<GlobalStreamingMetrics>>,
    /// Quality analyzer
    quality_analyzer: QualityAnalyzer,
    /// Error recovery manager
    error_recovery: ErrorRecoveryManager,
}

/// Individual stream session
#[derive(Debug, Clone)]
pub struct StreamSession {
    /// Session ID
    pub session_id: String,
    /// Conversation ID
    pub conversation_id: String,
    /// Current state
    pub state: StreamConnection,
    /// Session metrics
    pub metrics: StreamingMetrics,
    /// Start time
    pub start_time: Instant,
    /// Last activity
    pub last_activity: Instant,
    /// Buffer state
    pub buffer_state: BufferState,
}

/// Buffer state for backpressure management
#[derive(Debug, Clone)]
pub struct BufferState {
    /// Current buffer size
    pub current_size: usize,
    /// Maximum buffer size
    pub max_size: usize,
    /// Buffer utilization percentage
    pub utilization: f32,
    /// Pending chunks
    pub pending_chunks: usize,
}

/// Global streaming metrics across all sessions
#[derive(Debug, Clone)]
pub struct GlobalStreamingMetrics {
    /// Total active streams
    pub active_streams: usize,
    /// Total streams created
    pub total_streams_created: usize,
    /// Average stream duration
    pub avg_stream_duration_ms: f64,
    /// Total chunks streamed
    pub total_chunks_streamed: usize,
    /// Total bytes streamed
    pub total_bytes_streamed: usize,
    /// Global error rate
    pub global_error_rate: f32,
    /// System performance metrics
    pub system_performance: SystemPerformanceMetrics,
}

/// System performance metrics for streaming
#[derive(Debug, Clone)]
pub struct SystemPerformanceMetrics {
    /// CPU usage during streaming
    pub cpu_usage: f32,
    /// Memory usage
    pub memory_usage_mb: f64,
    /// Network utilization
    pub network_utilization: f32,
    /// Average latency
    pub avg_latency_ms: f64,
    /// Throughput (chunks per second)
    pub throughput: f32,
}

impl StreamingCoordinator {
    /// Create a new streaming coordinator
    pub fn new(config: AdvancedStreamingConfig) -> Self {
        Self {
            config,
            active_streams: Arc::new(RwLock::new(HashMap::new())),
            global_metrics: Arc::new(RwLock::new(GlobalStreamingMetrics::default())),
            quality_analyzer: QualityAnalyzer::new(),
            error_recovery: ErrorRecoveryManager::new(),
        }
    }

    /// Create a new streaming session
    pub async fn create_session(&self, conversation_id: String) -> Result<String> {
        let session_id = Uuid::new_v4().to_string();
        let session = StreamSession {
            session_id: session_id.clone(),
            conversation_id,
            state: StreamConnection::Connecting,
            metrics: StreamingMetrics::default(),
            start_time: Instant::now(),
            last_activity: Instant::now(),
            buffer_state: BufferState {
                current_size: 0,
                max_size: self.config.max_buffer_size,
                utilization: 0.0,
                pending_chunks: 0,
            },
        };

        let mut streams = self.active_streams.write().await;
        streams.insert(session_id.clone(), session);

        let mut global_metrics = self.global_metrics.write().await;
        global_metrics.active_streams = streams.len();
        global_metrics.total_streams_created += 1;

        Ok(session_id)
    }

    /// Update session state
    pub async fn update_session_state(
        &self,
        session_id: &str,
        state: StreamConnection,
    ) -> Result<()> {
        let mut streams = self.active_streams.write().await;
        if let Some(session) = streams.get_mut(session_id) {
            session.state = state;
            session.last_activity = Instant::now();
        }
        Ok(())
    }

    /// Get session information
    pub async fn get_session(&self, session_id: &str) -> Option<StreamSession> {
        self.active_streams.read().await.get(session_id).cloned()
    }

    /// Close streaming session
    pub async fn close_session(&self, session_id: &str) -> Result<()> {
        let mut streams = self.active_streams.write().await;
        if let Some(session) = streams.remove(session_id) {
            let duration = session.start_time.elapsed().as_millis() as f64;

            let mut global_metrics = self.global_metrics.write().await;
            global_metrics.active_streams = streams.len();

            // Update average duration
            let total_completed =
                global_metrics.total_streams_created - global_metrics.active_streams;
            if total_completed > 0 {
                global_metrics.avg_stream_duration_ms = (global_metrics.avg_stream_duration_ms
                    * (total_completed - 1) as f64
                    + duration)
                    / total_completed as f64;
            }
        }
        Ok(())
    }

    /// Get global metrics
    pub async fn get_global_metrics(&self) -> GlobalStreamingMetrics {
        self.global_metrics.read().await.clone()
    }

    /// Clean up expired sessions
    pub async fn cleanup_expired_sessions(&self, max_age_minutes: u64) -> usize {
        let cutoff = Instant::now() - Duration::from_secs(max_age_minutes * 60);
        let mut streams = self.active_streams.write().await;
        let initial_count = streams.len();

        streams.retain(|_, session| session.last_activity > cutoff);

        let removed_count = initial_count - streams.len();
        if removed_count > 0 {
            let mut global_metrics = self.global_metrics.write().await;
            global_metrics.active_streams = streams.len();
        }

        removed_count
    }

    /// Update buffer state for a session
    pub async fn update_buffer_state(
        &self,
        session_id: &str,
        buffer_state: BufferState,
    ) -> Result<()> {
        let mut streams = self.active_streams.write().await;
        if let Some(session) = streams.get_mut(session_id) {
            session.buffer_state = buffer_state;
            session.last_activity = Instant::now();
        }
        Ok(())
    }

    /// Get active session count
    pub async fn get_active_session_count(&self) -> usize {
        self.active_streams.read().await.len()
    }

    /// Check if session exists
    pub async fn session_exists(&self, session_id: &str) -> bool {
        self.active_streams.read().await.contains_key(session_id)
    }

    /// Update session metrics
    pub async fn update_session_metrics(
        &self,
        session_id: &str,
        metrics: StreamingMetrics,
    ) -> Result<()> {
        let mut streams = self.active_streams.write().await;
        if let Some(session) = streams.get_mut(session_id) {
            session.metrics = metrics;
            session.last_activity = Instant::now();
        }
        Ok(())
    }

    /// Get sessions by conversation ID
    pub async fn get_sessions_by_conversation(&self, conversation_id: &str) -> Vec<StreamSession> {
        self.active_streams
            .read()
            .await
            .values()
            .filter(|session| session.conversation_id == conversation_id)
            .cloned()
            .collect()
    }

    /// Update global metrics with session data
    pub async fn update_global_metrics_from_session(&self, session: &StreamSession) {
        let mut global_metrics = self.global_metrics.write().await;
        global_metrics.total_chunks_streamed += session.metrics.total_chunks;
        global_metrics.total_bytes_streamed += session.metrics.bytes_streamed;
    }
}

/// Legacy streaming manager for backward compatibility
#[derive(Debug)]
pub struct StreamingManager {
    /// Advanced streaming configuration
    pub config: StreamingConfig,
    /// Current streaming state
    state: StreamingState,
    /// Internal advanced pipeline
    advanced_config: AdvancedStreamingConfig,
}

impl StreamingManager {
    /// Create a new streaming manager
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
    pub async fn create_progressive_stream(
        &mut self,
        initial_chunk: String,
    ) -> Result<(
        mpsc::Sender<String>,
        Pin<Box<dyn Stream<Item = Result<StreamingResponse>> + Send + '_>>,
    )> {
        if !self.config.enabled {
            return Err(TrustformersError::invalid_input(
                "Streaming is disabled".to_string(),
                Some("streaming_config.enabled".to_string()),
                Some("true".to_string()),
                Some("false".to_string()),
            ));
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
    pub fn pause(&mut self) {
        if matches!(self.state, StreamingState::Streaming) {
            self.state = StreamingState::Paused;
        }
    }

    /// Resume a paused streaming session
    pub fn resume(&mut self) {
        if matches!(self.state, StreamingState::Paused) {
            self.state = StreamingState::Streaming;
        }
    }

    /// Stop the current streaming session
    pub fn stop(&mut self) {
        self.state = StreamingState::Completed;
    }

    /// Get current streaming state
    pub fn get_state(&self) -> &StreamingState {
        &self.state
    }

    /// Check if streaming is currently active
    pub fn is_streaming(&self) -> bool {
        matches!(self.state, StreamingState::Streaming)
    }

    /// Update streaming configuration
    pub fn update_config(&mut self, config: StreamingConfig) {
        self.config = config.clone();
        self.advanced_config.base_config = config;
    }

    /// Calculate streaming statistics
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

    /// Check if manager is in a valid state for streaming
    pub fn can_start_streaming(&self) -> bool {
        matches!(
            self.state,
            StreamingState::NotStarted | StreamingState::Completed
        )
    }

    /// Reset streaming state
    pub fn reset(&mut self) {
        self.state = StreamingState::NotStarted;
    }

    /// Get advanced configuration
    pub fn get_advanced_config(&self) -> &AdvancedStreamingConfig {
        &self.advanced_config
    }

    /// Update advanced configuration
    pub fn update_advanced_config(&mut self, config: AdvancedStreamingConfig) {
        self.advanced_config = config;
        self.config = self.advanced_config.base_config.clone();
    }
}

/// Statistics about streaming performance
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    pub total_chunks: usize,
    pub total_characters: usize,
    pub total_words: usize,
    pub avg_chunk_size: f32,
    pub estimated_duration_seconds: f32,
}

/// Streaming session information
#[derive(Debug, Clone)]
pub struct StreamingSession {
    pub session_id: String,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
    pub config: StreamingConfig,
    pub state: StreamingState,
    pub stats: Option<StreamingStats>,
}

impl StreamingSession {
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            session_id: uuid::Uuid::new_v4().to_string(),
            start_time: chrono::Utc::now(),
            end_time: None,
            config,
            state: StreamingState::NotStarted,
            stats: None,
        }
    }

    pub fn complete(&mut self, stats: StreamingStats) {
        self.end_time = Some(chrono::Utc::now());
        self.state = StreamingState::Completed;
        self.stats = Some(stats);
    }

    pub fn duration_ms(&self) -> Option<i64> {
        self.end_time.map(|end| (end - self.start_time).num_milliseconds())
    }
}

// ================================================================================================
// DEFAULT IMPLEMENTATIONS
// ================================================================================================

impl Default for GlobalStreamingMetrics {
    fn default() -> Self {
        Self {
            active_streams: 0,
            total_streams_created: 0,
            avg_stream_duration_ms: 0.0,
            total_chunks_streamed: 0,
            total_bytes_streamed: 0,
            global_error_rate: 0.0,
            system_performance: SystemPerformanceMetrics::default(),
        }
    }
}

impl Default for SystemPerformanceMetrics {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage_mb: 0.0,
            network_utilization: 0.0,
            avg_latency_ms: 0.0,
            throughput: 0.0,
        }
    }
}

impl Default for StreamingManager {
    fn default() -> Self {
        Self::new(StreamingConfig::default())
    }
}

impl Default for BufferState {
    fn default() -> Self {
        Self {
            current_size: 0,
            max_size: 1024,
            utilization: 0.0,
            pending_chunks: 0,
        }
    }
}

// ================================================================================================
// SESSION UTILITIES
// ================================================================================================

impl StreamSession {
    /// Create a new stream session
    pub fn new(session_id: String, conversation_id: String, max_buffer_size: usize) -> Self {
        Self {
            session_id,
            conversation_id,
            state: StreamConnection::Connecting,
            metrics: StreamingMetrics::default(),
            start_time: Instant::now(),
            last_activity: Instant::now(),
            buffer_state: BufferState {
                current_size: 0,
                max_size: max_buffer_size,
                utilization: 0.0,
                pending_chunks: 0,
            },
        }
    }

    /// Update the last activity time
    pub fn touch(&mut self) {
        self.last_activity = Instant::now();
    }

    /// Check if session is expired based on timeout
    pub fn is_expired(&self, timeout_minutes: u64) -> bool {
        let timeout_duration = Duration::from_secs(timeout_minutes * 60);
        self.last_activity.elapsed() > timeout_duration
    }

    /// Get session duration
    pub fn duration(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Update buffer utilization
    pub fn update_buffer_utilization(&mut self, current_size: usize, pending_chunks: usize) {
        self.buffer_state.current_size = current_size;
        self.buffer_state.pending_chunks = pending_chunks;
        self.buffer_state.utilization = if self.buffer_state.max_size > 0 {
            current_size as f32 / self.buffer_state.max_size as f32
        } else {
            0.0
        };
        self.touch();
    }
}

impl BufferState {
    /// Create a new buffer state with specified max size
    pub fn new(max_size: usize) -> Self {
        Self {
            current_size: 0,
            max_size,
            utilization: 0.0,
            pending_chunks: 0,
        }
    }

    /// Check if buffer is near capacity
    pub fn is_near_capacity(&self, threshold: f32) -> bool {
        self.utilization >= threshold
    }

    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.current_size >= self.max_size
    }

    /// Get available buffer space
    pub fn available_space(&self) -> usize {
        self.max_size.saturating_sub(self.current_size)
    }
}
