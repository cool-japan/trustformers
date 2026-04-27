// Allow dead code for infrastructure under development
#![allow(dead_code)]

//! Streaming Support for Real-time Inference
//!
//! This module provides streaming capabilities for real-time model inference,
//! including token streaming, Server-Sent Events (SSE), and WebSocket support.

pub mod chunk_stream;
pub mod sse;
pub mod token_sse;
pub mod token_stream;
pub mod websocket;

pub use sse::{SseConfig, SseConnection, SseError, SseEvent, SseEventType, SseHandler, SseMetrics};
pub use token_sse::{
    FinishReason, OpenAiStreamChunk, SseError as TokenSseError, SseEvent as TokenSseEvent,
    SseStream, SseStreamConfig, StreamChoice, StreamDelta, TokenChunk,
};

pub use websocket::{
    WebSocketHandler, WsConfig, WsConnection, WsError, WsMessage, WsMessageType, WsMetrics,
};

pub use token_stream::{
    StreamEvent, StreamingResponse, StreamingStats, TokenBuffer, TokenStream, TokenStreamConfig,
};

pub use chunk_stream::{
    ChunkConfig, ChunkStream, ChunkedResponse, ChunkingStrategy, ResponseChunk,
};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc;
use uuid::Uuid;

/// Main streaming service
#[derive(Clone)]
pub struct StreamingService {
    config: StreamingConfig,
    metrics: Arc<StreamingMetrics>,
    active_streams: Arc<tokio::sync::RwLock<std::collections::HashMap<Uuid, ActiveStream>>>,
}

impl StreamingService {
    /// Create a new streaming service
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(StreamingMetrics::new()),
            active_streams: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Start a new streaming session
    pub async fn start_stream(
        &self,
        stream_type: StreamType,
        request_id: Uuid,
    ) -> Result<StreamHandle> {
        let stream_id = Uuid::new_v4();
        let (tx, rx) = mpsc::channel(self.config.buffer_size);

        let stream = ActiveStream {
            id: stream_id,
            stream_type,
            request_id,
            sender: tx,
            started_at: std::time::Instant::now(),
            last_activity: std::time::Instant::now(),
        };

        self.active_streams.write().await.insert(stream_id, stream);
        self.metrics.record_stream_started(stream_type).await;

        Ok(StreamHandle {
            id: stream_id,
            receiver: rx,
            stream_type,
        })
    }

    /// Send data to a stream
    pub async fn send_to_stream(&self, stream_id: Uuid, data: StreamData) -> Result<()> {
        if let Some(stream) = self.active_streams.write().await.get_mut(&stream_id) {
            stream.last_activity = std::time::Instant::now();
            stream.sender.send(data).await?;
            self.metrics.record_data_sent(stream.stream_type).await;
        }

        Ok(())
    }

    /// Close a stream
    pub async fn close_stream(&self, stream_id: Uuid) -> Result<()> {
        if let Some(stream) = self.active_streams.write().await.remove(&stream_id) {
            let duration = stream.started_at.elapsed();
            self.metrics.record_stream_closed(stream.stream_type, duration).await;
        }

        Ok(())
    }

    /// Get streaming statistics
    pub async fn get_stats(&self) -> GlobalStreamingStats {
        let active_count = self.active_streams.read().await.len();

        GlobalStreamingStats {
            active_streams: active_count,
            total_streams_started: self.metrics.total_streams_started().await,
            total_data_sent: self.metrics.total_data_sent().await,
            avg_stream_duration_ms: self.metrics.avg_stream_duration_ms().await,
        }
    }

    /// Cleanup inactive streams
    pub async fn cleanup_inactive_streams(&self) -> Result<()> {
        let timeout = self.config.stream_timeout;
        let now = std::time::Instant::now();

        let mut to_remove = Vec::new();

        {
            let streams = self.active_streams.read().await;
            for (id, stream) in streams.iter() {
                if now.duration_since(stream.last_activity) > timeout {
                    to_remove.push(*id);
                }
            }
        }

        for id in to_remove {
            self.close_stream(id).await?;
        }

        Ok(())
    }
}

/// Streaming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Buffer size for streaming channels
    pub buffer_size: usize,

    /// Stream timeout duration
    pub stream_timeout: std::time::Duration,

    /// Maximum concurrent streams
    pub max_concurrent_streams: usize,

    /// Enable compression for streams
    pub enable_compression: bool,

    /// Chunk size for data streaming
    pub chunk_size: usize,

    /// Heartbeat interval
    pub heartbeat_interval: std::time::Duration,

    /// SSE configuration
    pub sse_config: SseConfig,

    /// WebSocket configuration
    pub ws_config: WsConfig,

    /// Token streaming configuration
    pub token_config: TokenStreamConfig,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1000,
            stream_timeout: std::time::Duration::from_secs(300), // 5 minutes
            max_concurrent_streams: 1000,
            enable_compression: true,
            chunk_size: 8192,
            heartbeat_interval: std::time::Duration::from_secs(30),
            sse_config: SseConfig::default(),
            ws_config: WsConfig::default(),
            token_config: TokenStreamConfig::default(),
        }
    }
}

/// Stream type enumeration
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum StreamType {
    /// Server-Sent Events
    Sse,
    /// WebSocket
    WebSocket,
    /// Token streaming for LLM generation
    TokenStream,
    /// Chunked HTTP response
    ChunkedHttp,
}

/// Stream handle for managing active streams
#[derive(Debug)]
pub struct StreamHandle {
    pub id: Uuid,
    pub receiver: mpsc::Receiver<StreamData>,
    pub stream_type: StreamType,
}

/// Active stream information
#[derive(Debug)]
struct ActiveStream {
    id: Uuid,
    stream_type: StreamType,
    request_id: Uuid,
    sender: mpsc::Sender<StreamData>,
    started_at: std::time::Instant,
    last_activity: std::time::Instant,
}

/// Stream data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamData {
    /// Text token for LLM generation
    Token(String),
    /// Binary data chunk
    Chunk(Vec<u8>),
    /// JSON structured data
    Json(serde_json::Value),
    /// Heartbeat ping
    Heartbeat,
    /// Stream completion
    End,
    /// Error in stream
    Error(String),
}

/// Global streaming statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalStreamingStats {
    pub active_streams: usize,
    pub total_streams_started: u64,
    pub total_data_sent: u64,
    pub avg_stream_duration_ms: f64,
}

/// Streaming metrics collector
#[derive(Debug)]
pub struct StreamingMetrics {
    streams_started: Arc<tokio::sync::RwLock<std::collections::HashMap<StreamType, u64>>>,
    data_sent: Arc<tokio::sync::RwLock<std::collections::HashMap<StreamType, u64>>>,
    stream_durations: Arc<tokio::sync::RwLock<Vec<std::time::Duration>>>,
}

impl Default for StreamingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingMetrics {
    pub fn new() -> Self {
        Self {
            streams_started: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
            data_sent: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
            stream_durations: Arc::new(tokio::sync::RwLock::new(Vec::new())),
        }
    }

    pub async fn record_stream_started(&self, stream_type: StreamType) {
        *self.streams_started.write().await.entry(stream_type).or_insert(0) += 1;
    }

    pub async fn record_data_sent(&self, stream_type: StreamType) {
        *self.data_sent.write().await.entry(stream_type).or_insert(0) += 1;
    }

    pub async fn record_stream_closed(
        &self,
        _stream_type: StreamType,
        duration: std::time::Duration,
    ) {
        self.stream_durations.write().await.push(duration);
    }

    pub async fn total_streams_started(&self) -> u64 {
        self.streams_started.read().await.values().sum()
    }

    pub async fn total_data_sent(&self) -> u64 {
        self.data_sent.read().await.values().sum()
    }

    pub async fn avg_stream_duration_ms(&self) -> f64 {
        let durations = self.stream_durations.read().await;
        if durations.is_empty() {
            0.0
        } else {
            let total: f64 = durations.iter().map(|d| d.as_millis() as f64).sum();
            total / durations.len() as f64
        }
    }
}

// ── StreamAccumulator ─────────────────────────────────────────────────────────

/// Accumulates streamed tokens for partial output tracking and throughput metrics.
pub struct StreamAccumulator {
    /// Complete text accumulated so far (all tokens concatenated).
    pub full_text: String,
    /// Number of tokens received.
    pub token_count: usize,
    /// Elapsed milliseconds when the first token arrived, if any.
    pub first_token_ms: Option<f64>,
    /// Total elapsed milliseconds (set by [`Self::mark_complete`]).
    pub total_ms: f64,
    /// Wall-clock offset (ms) when this accumulator was created.
    started_at_ms: f64,
    /// Whether the stream has been explicitly completed.
    completed: bool,
}

impl StreamAccumulator {
    /// Create a new accumulator.  `started_at_ms` is the wall-clock millisecond
    /// offset at which streaming began (used for latency-to-first-token).
    pub fn new(started_at_ms: f64) -> Self {
        Self {
            full_text: String::new(),
            token_count: 0,
            first_token_ms: None,
            total_ms: 0.0,
            started_at_ms,
            completed: false,
        }
    }

    /// Append a token to the accumulator.
    ///
    /// `elapsed_ms` is the number of milliseconds since `started_at_ms`.
    /// The first call sets `first_token_ms`; subsequent calls only append text.
    pub fn push_token(&mut self, token: &str, elapsed_ms: f64) {
        if self.first_token_ms.is_none() {
            self.first_token_ms = Some(elapsed_ms);
        }
        self.full_text.push_str(token);
        self.token_count += 1;
    }

    /// Tokens generated per second.  Returns 0.0 if `total_ms` is zero.
    pub fn tokens_per_second(&self) -> f64 {
        if self.total_ms <= 0.0 || self.token_count == 0 {
            0.0
        } else {
            self.token_count as f64 / (self.total_ms / 1000.0)
        }
    }

    /// Returns true once `mark_complete` has been called.
    pub fn is_complete(&self) -> bool {
        self.completed
    }

    /// Record end-of-stream, storing the total elapsed time in milliseconds.
    pub fn mark_complete(&mut self, total_ms: f64) {
        self.total_ms = total_ms;
        self.completed = true;
    }
}

// ── SSE format helpers ────────────────────────────────────────────────────────

/// Format an SSE data line: `"data: {json}\n\n"`.
pub fn format_sse_data(json: &str) -> String {
    format!("data: {}\n\n", json)
}

/// Format the SSE end-of-stream signal.
pub fn format_sse_done() -> String {
    "data: [DONE]\n\n".to_string()
}

// ── BackpressureTracker ───────────────────────────────────────────────────────

/// Tracks channel backpressure to allow graceful degradation under load.
pub struct BackpressureTracker {
    /// Load level at which the channel is considered backpressured.
    pub high_watermark: usize,
    /// Current estimated number of in-flight items.
    pub current_load: usize,
    /// Total number of items dropped due to backpressure.
    pub drop_count: u64,
}

impl BackpressureTracker {
    /// Create a new tracker with the given high-watermark threshold.
    pub fn new(high_watermark: usize) -> Self {
        Self {
            high_watermark,
            current_load: 0,
            drop_count: 0,
        }
    }

    /// Increment the in-flight load counter (call when sending an item).
    pub fn record_send(&mut self) {
        self.current_load += 1;
    }

    /// Decrement the in-flight load counter (call when an item is consumed).
    pub fn record_consume(&mut self) {
        self.current_load = self.current_load.saturating_sub(1);
    }

    /// Returns `true` if `current_load >= high_watermark`.
    pub fn is_backpressured(&self) -> bool {
        self.current_load >= self.high_watermark
    }

    /// Record a dropped item (increments `drop_count`, decrements load).
    pub fn record_drop(&mut self) {
        self.drop_count += 1;
        self.current_load = self.current_load.saturating_sub(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_streaming_service() {
        let config = StreamingConfig::default();
        let service = StreamingService::new(config);

        let handle = service
            .start_stream(StreamType::TokenStream, Uuid::new_v4())
            .await
            .expect("async operation should succeed in test");

        service
            .send_to_stream(handle.id, StreamData::Token("Hello".to_string()))
            .await
            .expect("test operation should succeed");
        service
            .close_stream(handle.id)
            .await
            .expect("async operation should succeed in test");

        let stats = service.get_stats().await;
        assert_eq!(stats.active_streams, 0);
    }

    #[test]
    fn test_stream_data_serialization() {
        let data = StreamData::Token("test".to_string());
        let json = serde_json::to_string(&data).expect("JSON serialization should succeed");
        let deserialized: StreamData =
            serde_json::from_str(&json).expect("JSON parsing should succeed for valid test input");

        match deserialized {
            StreamData::Token(s) => assert_eq!(s, "test"),
            other => {
                // Use assert! to provide better error message in tests
                assert!(
                    matches!(other, StreamData::Token(_)),
                    "Expected Token deserialization, got: {:?}",
                    other
                );
            },
        }
    }

    // ── StreamAccumulator tests ───────────────────────────────────────────────

    // 1. new initializes correctly
    #[test]
    fn test_accumulator_new_initializes() {
        let acc = StreamAccumulator::new(0.0);
        assert!(acc.full_text.is_empty());
        assert_eq!(acc.token_count, 0);
        assert!(acc.first_token_ms.is_none());
        assert_eq!(acc.total_ms, 0.0);
        assert!(!acc.is_complete());
    }

    // 2. push_token sets first_token_ms on first push
    #[test]
    fn test_accumulator_first_token_ms_set_on_first_push() {
        let mut acc = StreamAccumulator::new(0.0);
        acc.push_token("hello", 15.0);
        assert_eq!(acc.first_token_ms, Some(15.0));
    }

    // 3. push_token does NOT update first_token_ms on second push
    #[test]
    fn test_accumulator_first_token_ms_not_updated_on_second_push() {
        let mut acc = StreamAccumulator::new(0.0);
        acc.push_token("first", 10.0);
        acc.push_token("second", 20.0);
        assert_eq!(acc.first_token_ms, Some(10.0));
    }

    // 4. push_token appends to full_text correctly
    #[test]
    fn test_accumulator_full_text_accumulates() {
        let mut acc = StreamAccumulator::new(0.0);
        acc.push_token("Hello", 5.0);
        acc.push_token(", ", 10.0);
        acc.push_token("world", 15.0);
        assert_eq!(acc.full_text, "Hello, world");
    }

    // 5. push_token increments token_count
    #[test]
    fn test_accumulator_token_count_increments() {
        let mut acc = StreamAccumulator::new(0.0);
        acc.push_token("a", 1.0);
        acc.push_token("b", 2.0);
        acc.push_token("c", 3.0);
        assert_eq!(acc.token_count, 3);
    }

    // 6. tokens_per_second returns 0 when total_ms is 0
    #[test]
    fn test_accumulator_tokens_per_second_zero_when_no_total_ms() {
        let acc = StreamAccumulator::new(0.0);
        assert_eq!(acc.tokens_per_second(), 0.0);
    }

    // 7. tokens_per_second returns correct value when total_ms > 0
    #[test]
    fn test_accumulator_tokens_per_second_correct() {
        let mut acc = StreamAccumulator::new(0.0);
        acc.push_token("a", 10.0);
        acc.push_token("b", 20.0);
        acc.mark_complete(2000.0); // 2 seconds
                                   // 2 tokens / 2 seconds = 1.0 token/sec
        let tps = acc.tokens_per_second();
        assert!((tps - 1.0).abs() < 1e-9, "expected 1.0 tps, got {}", tps);
    }

    // 8. is_complete returns false initially
    #[test]
    fn test_accumulator_not_complete_initially() {
        let acc = StreamAccumulator::new(0.0);
        assert!(!acc.is_complete());
    }

    // 9. is_complete returns true after mark_complete
    #[test]
    fn test_accumulator_complete_after_mark_complete() {
        let mut acc = StreamAccumulator::new(0.0);
        acc.mark_complete(500.0);
        assert!(acc.is_complete());
    }

    // 10. mark_complete sets total_ms
    #[test]
    fn test_accumulator_mark_complete_sets_total_ms() {
        let mut acc = StreamAccumulator::new(0.0);
        acc.mark_complete(1234.5);
        assert!((acc.total_ms - 1234.5).abs() < 1e-9);
    }

    // 11. Multiple token accumulation
    #[test]
    fn test_accumulator_multiple_tokens() {
        let mut acc = StreamAccumulator::new(100.0);
        for i in 0..10 {
            acc.push_token("x", (i * 50) as f64);
        }
        assert_eq!(acc.token_count, 10);
        assert_eq!(acc.full_text, "xxxxxxxxxx");
        assert_eq!(acc.first_token_ms, Some(0.0));
    }

    // ── SSE format helper tests ───────────────────────────────────────────────

    // 12. format_sse_data produces "data: {...}\n\n" format
    #[test]
    fn test_format_sse_data_format() {
        let result = format_sse_data("{\"token\":\"hi\"}");
        assert!(result.starts_with("data: "));
        assert!(result.ends_with("\n\n"));
        assert!(result.contains("{\"token\":\"hi\"}"));
    }

    // 13. format_sse_done produces "data: [DONE]\n\n"
    #[test]
    fn test_format_sse_done() {
        let result = format_sse_done();
        assert_eq!(result, "data: [DONE]\n\n");
    }

    // 14. SSE format ends with double newline
    #[test]
    fn test_sse_format_ends_with_double_newline() {
        let result = format_sse_data("{}");
        assert!(result.ends_with("\n\n"));
        let result_done = format_sse_done();
        assert!(result_done.ends_with("\n\n"));
    }

    // ── BackpressureTracker tests ─────────────────────────────────────────────

    // 15. BackpressureTracker::new initializes correctly
    #[test]
    fn test_backpressure_tracker_new() {
        let tracker = BackpressureTracker::new(10);
        assert_eq!(tracker.high_watermark, 10);
        assert_eq!(tracker.current_load, 0);
        assert_eq!(tracker.drop_count, 0);
    }

    // 16. record_send increments current_load
    #[test]
    fn test_backpressure_record_send_increments() {
        let mut tracker = BackpressureTracker::new(10);
        tracker.record_send();
        tracker.record_send();
        assert_eq!(tracker.current_load, 2);
    }

    // 17. record_consume decrements current_load
    #[test]
    fn test_backpressure_record_consume_decrements() {
        let mut tracker = BackpressureTracker::new(10);
        tracker.record_send();
        tracker.record_send();
        tracker.record_consume();
        assert_eq!(tracker.current_load, 1);
    }

    // 18. record_consume does not underflow below 0
    #[test]
    fn test_backpressure_record_consume_no_underflow() {
        let mut tracker = BackpressureTracker::new(10);
        tracker.record_consume(); // already at 0
        assert_eq!(tracker.current_load, 0);
    }

    // 19. is_backpressured returns false below watermark
    #[test]
    fn test_backpressure_false_below_watermark() {
        let mut tracker = BackpressureTracker::new(5);
        tracker.record_send();
        tracker.record_send();
        assert!(!tracker.is_backpressured());
    }

    // 20. is_backpressured returns true at or above watermark
    #[test]
    fn test_backpressure_true_at_watermark() {
        let mut tracker = BackpressureTracker::new(3);
        tracker.record_send();
        tracker.record_send();
        tracker.record_send();
        assert!(tracker.is_backpressured());
    }

    // 21. record_drop increments drop_count
    #[test]
    fn test_backpressure_record_drop_increments_drop_count() {
        let mut tracker = BackpressureTracker::new(5);
        tracker.record_send();
        tracker.record_drop();
        assert_eq!(tracker.drop_count, 1);
        assert_eq!(tracker.current_load, 0);
    }

    // 22. StreamingConfig default has reasonable values
    #[test]
    fn test_streaming_config_default_reasonable() {
        let cfg = StreamingConfig::default();
        assert!(cfg.buffer_size > 0);
        assert!(cfg.max_concurrent_streams > 0);
        assert!(cfg.chunk_size > 0);
    }

    // 23. StreamData::Heartbeat serializes and deserializes correctly
    #[test]
    fn test_stream_data_heartbeat_serialization() {
        let data = StreamData::Heartbeat;
        let json = serde_json::to_string(&data).expect("serialize heartbeat");
        let back: StreamData = serde_json::from_str(&json).expect("deserialize heartbeat");
        assert!(matches!(back, StreamData::Heartbeat));
    }

    // 24. StreamData::End serializes and deserializes correctly
    #[test]
    fn test_stream_data_end_serialization() {
        let data = StreamData::End;
        let json = serde_json::to_string(&data).expect("serialize end");
        let back: StreamData = serde_json::from_str(&json).expect("deserialize end");
        assert!(matches!(back, StreamData::End));
    }

    // 25. GlobalStreamingStats fields are accessible
    #[test]
    fn test_global_streaming_stats_fields() {
        let stats = GlobalStreamingStats {
            active_streams: 3,
            total_streams_started: 100,
            total_data_sent: 500,
            avg_stream_duration_ms: 1234.5,
        };
        assert_eq!(stats.active_streams, 3);
        assert_eq!(stats.total_streams_started, 100);
        assert_eq!(stats.total_data_sent, 500);
        assert!((stats.avg_stream_duration_ms - 1234.5).abs() < 1e-9);
    }
}
