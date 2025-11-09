// Allow dead code for infrastructure under development
#![allow(dead_code)]

//! Streaming Support for Real-time Inference
//!
//! This module provides streaming capabilities for real-time model inference,
//! including token streaming, Server-Sent Events (SSE), and WebSocket support.

pub mod chunk_stream;
pub mod sse;
pub mod token_stream;
pub mod websocket;

pub use sse::{SseConfig, SseConnection, SseError, SseEvent, SseEventType, SseHandler, SseMetrics};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_streaming_service() {
        let config = StreamingConfig::default();
        let service = StreamingService::new(config);

        let handle = service.start_stream(StreamType::TokenStream, Uuid::new_v4()).await.unwrap();

        service
            .send_to_stream(handle.id, StreamData::Token("Hello".to_string()))
            .await
            .unwrap();
        service.close_stream(handle.id).await.unwrap();

        let stats = service.get_stats().await;
        assert_eq!(stats.active_streams, 0);
    }

    #[test]
    fn test_stream_data_serialization() {
        let data = StreamData::Token("test".to_string());
        let json = serde_json::to_string(&data).unwrap();
        let deserialized: StreamData = serde_json::from_str(&json).unwrap();

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
}
