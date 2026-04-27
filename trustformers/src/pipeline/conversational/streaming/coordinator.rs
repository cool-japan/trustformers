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

#[cfg(test)]
mod tests {
    use super::*;

    fn default_coordinator() -> StreamingCoordinator {
        StreamingCoordinator::new(AdvancedStreamingConfig::default())
    }

    // ---- BufferState tests ----

    #[test]
    fn test_buffer_state_new_empty() {
        let buf = BufferState::new(1024);
        assert_eq!(buf.current_size, 0, "new buffer should have current_size 0");
        assert_eq!(
            buf.max_size, 1024,
            "new buffer max_size must match supplied value"
        );
        assert!(
            (buf.utilization - 0.0).abs() < f32::EPSILON,
            "new buffer utilization should be 0"
        );
    }

    #[test]
    fn test_buffer_state_is_not_near_capacity_when_empty() {
        let buf = BufferState::new(100);
        assert!(
            !buf.is_near_capacity(0.5),
            "empty buffer must not be near capacity at 50%"
        );
    }

    #[test]
    fn test_buffer_state_is_near_capacity_at_full_utilization() {
        let mut buf = BufferState::new(100);
        buf.current_size = 100;
        buf.utilization = 1.0;
        assert!(
            buf.is_near_capacity(0.9),
            "buffer at 100% utilization must be near capacity at 90% threshold"
        );
    }

    #[test]
    fn test_buffer_state_is_full_when_at_max() {
        let mut buf = BufferState::new(50);
        buf.current_size = 50;
        assert!(buf.is_full(), "buffer at max_size must report is_full");
    }

    #[test]
    fn test_buffer_state_is_not_full_when_below_max() {
        let mut buf = BufferState::new(100);
        buf.current_size = 50;
        assert!(
            !buf.is_full(),
            "buffer below max_size must not report is_full"
        );
    }

    #[test]
    fn test_buffer_state_available_space() {
        let mut buf = BufferState::new(200);
        buf.current_size = 75;
        assert_eq!(
            buf.available_space(),
            125,
            "available_space must be max_size - current_size"
        );
    }

    #[test]
    fn test_buffer_state_available_space_saturating() {
        let mut buf = BufferState::new(50);
        buf.current_size = 100; // over-full (should not happen but must not panic)
        assert_eq!(
            buf.available_space(),
            0,
            "available_space must be 0 when current_size exceeds max"
        );
    }

    // ---- StreamSession tests ----

    #[test]
    fn test_stream_session_new_initial_state() {
        let session = StreamSession::new("s1".to_string(), "conv-1".to_string(), 1000);
        assert_eq!(
            session.session_id, "s1",
            "session_id must match supplied value"
        );
        assert_eq!(
            session.conversation_id, "conv-1",
            "conversation_id must match supplied value"
        );
        assert_eq!(
            session.state,
            StreamConnection::Connecting,
            "new session state must be Connecting"
        );
    }

    #[test]
    fn test_stream_session_touch_updates_activity() {
        let mut session = StreamSession::new("s2".to_string(), "conv-2".to_string(), 500);
        let before = session.last_activity;
        std::thread::sleep(std::time::Duration::from_millis(1));
        session.touch();
        assert!(
            session.last_activity >= before,
            "touch must update last_activity to current time"
        );
    }

    #[test]
    fn test_stream_session_not_expired_immediately() {
        let session = StreamSession::new("s3".to_string(), "conv-3".to_string(), 500);
        assert!(
            !session.is_expired(60),
            "freshly created session must not be expired after 60 min timeout"
        );
    }

    #[test]
    fn test_stream_session_update_buffer_utilization() {
        let mut session = StreamSession::new("s4".to_string(), "conv-4".to_string(), 100);
        session.update_buffer_utilization(50, 5);
        assert_eq!(
            session.buffer_state.current_size, 50,
            "buffer current_size must be 50"
        );
        assert_eq!(
            session.buffer_state.pending_chunks, 5,
            "pending_chunks must be 5"
        );
        assert!(
            (session.buffer_state.utilization - 0.5).abs() < 1e-5,
            "utilization must be 0.5 for 50/100"
        );
    }

    // ---- StreamingManager tests ----

    #[test]
    fn test_streaming_manager_initial_state_not_started() {
        let manager = StreamingManager::default();
        assert!(
            matches!(manager.get_state(), StreamingState::NotStarted),
            "default manager state must be NotStarted"
        );
    }

    #[test]
    fn test_streaming_manager_can_start_when_not_started() {
        let manager = StreamingManager::default();
        assert!(
            manager.can_start_streaming(),
            "should be able to start when NotStarted"
        );
    }

    #[test]
    fn test_streaming_manager_pause_and_resume() {
        let mut manager = StreamingManager::new(StreamingConfig {
            enabled: true,
            ..StreamingConfig::default()
        });
        manager.state = StreamingState::Streaming;
        manager.pause();
        assert!(
            matches!(manager.get_state(), StreamingState::Paused),
            "state must be Paused after pause()"
        );
        manager.resume();
        assert!(
            matches!(manager.get_state(), StreamingState::Streaming),
            "state must be Streaming after resume()"
        );
    }

    #[test]
    fn test_streaming_manager_stop_sets_completed() {
        let mut manager = StreamingManager::new(StreamingConfig {
            enabled: true,
            ..StreamingConfig::default()
        });
        manager.stop();
        assert!(
            matches!(manager.get_state(), StreamingState::Completed),
            "stop() must set Completed state"
        );
    }

    #[test]
    fn test_streaming_manager_reset() {
        let mut manager = StreamingManager::default();
        manager.stop();
        manager.reset();
        assert!(
            matches!(manager.get_state(), StreamingState::NotStarted),
            "reset() must restore NotStarted state"
        );
    }

    #[test]
    fn test_streaming_manager_calculate_stats_empty() {
        let manager = StreamingManager::default();
        let stats = manager.calculate_stream_stats(&[]);
        assert_eq!(
            stats.total_chunks, 0,
            "empty responses should yield 0 total_chunks"
        );
    }

    #[test]
    fn test_streaming_manager_calculate_stats_with_responses() {
        let manager = StreamingManager::default();
        let responses = vec![
            StreamingResponse {
                chunk: "hello world".to_string(),
                is_final: false,
                chunk_index: 0,
                total_chunks: None,
                metadata: None,
            },
            StreamingResponse {
                chunk: "goodbye".to_string(),
                is_final: true,
                chunk_index: 1,
                total_chunks: Some(2),
                metadata: None,
            },
        ];
        let stats = manager.calculate_stream_stats(&responses);
        assert_eq!(
            stats.total_chunks, 2,
            "two responses must yield total_chunks 2"
        );
        assert_eq!(stats.total_words, 3, "three words across two chunks");
    }

    // ---- StreamingCoordinator tests (async) ----

    #[tokio::test]
    async fn test_coordinator_initial_session_count_zero() {
        let coord = default_coordinator();
        assert_eq!(
            coord.get_active_session_count().await,
            0,
            "new coordinator should have 0 active sessions"
        );
    }

    #[tokio::test]
    async fn test_coordinator_create_session_increments_count() {
        let coord = default_coordinator();
        let _id = coord
            .create_session("conv-1".to_string())
            .await
            .expect("create_session must succeed");
        assert_eq!(
            coord.get_active_session_count().await,
            1,
            "one created session should yield count 1"
        );
    }

    #[tokio::test]
    async fn test_coordinator_create_session_returns_unique_ids() {
        let coord = default_coordinator();
        let id1 = coord.create_session("c1".to_string()).await.expect("session 1 must succeed");
        let id2 = coord.create_session("c2".to_string()).await.expect("session 2 must succeed");
        assert_ne!(id1, id2, "each session must have a unique id");
    }

    #[tokio::test]
    async fn test_coordinator_close_session_decrements_count() {
        let coord = default_coordinator();
        let id = coord
            .create_session("c1".to_string())
            .await
            .expect("create_session must succeed");
        coord.close_session(&id).await.expect("close_session must succeed");
        assert_eq!(
            coord.get_active_session_count().await,
            0,
            "closed session count must be 0"
        );
    }

    #[tokio::test]
    async fn test_coordinator_session_exists_true() {
        let coord = default_coordinator();
        let id = coord
            .create_session("c1".to_string())
            .await
            .expect("create_session must succeed");
        assert!(
            coord.session_exists(&id).await,
            "session must exist after creation"
        );
    }

    #[tokio::test]
    async fn test_coordinator_session_exists_false_for_unknown() {
        let coord = default_coordinator();
        assert!(
            !coord.session_exists("not-a-real-id").await,
            "unknown session must not exist"
        );
    }

    #[tokio::test]
    async fn test_coordinator_update_session_state() {
        let coord = default_coordinator();
        let id = coord
            .create_session("c1".to_string())
            .await
            .expect("create_session must succeed");
        coord
            .update_session_state(&id, StreamConnection::Connected)
            .await
            .expect("update_session_state must succeed");
        let session = coord.get_session(&id).await.expect("session must be retrievable");
        assert_eq!(
            session.state,
            StreamConnection::Connected,
            "session state must be Connected"
        );
    }
}
