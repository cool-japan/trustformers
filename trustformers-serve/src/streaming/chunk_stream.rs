// Allow dead code for infrastructure under development
#![allow(dead_code)]

//! Chunked Response Streaming
//!
//! Provides chunked HTTP response streaming for efficient data transfer
//! of large inference results and progressive response delivery.

use anyhow::{anyhow, Result};
use axum::{
    body::Body,
    http::{header, StatusCode},
    response::Response,
};
use bytes::Bytes;
use futures::stream::{self};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, RwLock};
use uuid::Uuid;

/// Chunk stream for progressive response delivery
#[derive(Debug)]
pub struct ChunkStream {
    config: ChunkConfig,
    buffer: Arc<RwLock<ChunkBuffer>>,
    sender: mpsc::Sender<ResponseChunk>,
    receiver: Option<mpsc::Receiver<ResponseChunk>>,
    stats: Arc<RwLock<ChunkStats>>,
    stream_id: Uuid,
}

impl ChunkStream {
    /// Create a new chunk stream
    pub fn new(config: ChunkConfig, stream_id: Uuid) -> Self {
        let (tx, rx) = mpsc::channel(config.buffer_size);

        Self {
            buffer: Arc::new(RwLock::new(ChunkBuffer::new(config.max_buffer_size))),
            config,
            sender: tx,
            receiver: Some(rx),
            stats: Arc::new(RwLock::new(ChunkStats::new())),
            stream_id,
        }
    }

    /// Take the receiver for consuming chunks
    pub fn take_receiver(&mut self) -> Option<mpsc::Receiver<ResponseChunk>> {
        self.receiver.take()
    }

    /// Add data chunk to the stream
    pub async fn add_chunk(&self, data: Vec<u8>, chunk_type: ChunkType) -> Result<()> {
        let chunk = ResponseChunk {
            id: Uuid::new_v4(),
            data: data.clone(),
            chunk_type,
            timestamp: chrono::Utc::now(),
            size: data.len(),
            stream_id: self.stream_id,
        };

        // Add to buffer
        self.buffer.write().await.add_chunk(chunk.clone());

        // Send to stream
        self.sender.send(chunk).await?;

        // Update stats
        self.stats.write().await.record_chunk_sent(data.len());

        Ok(())
    }

    /// Add text chunk
    pub async fn add_text(&self, text: String) -> Result<()> {
        self.add_chunk(text.into_bytes(), ChunkType::Text).await
    }

    /// Add JSON chunk
    pub async fn add_json(&self, value: serde_json::Value) -> Result<()> {
        let json_bytes = serde_json::to_vec(&value)?;
        self.add_chunk(json_bytes, ChunkType::Json).await
    }

    /// Add binary chunk
    pub async fn add_binary(&self, data: Vec<u8>) -> Result<()> {
        self.add_chunk(data, ChunkType::Binary).await
    }

    /// Signal end of stream
    pub async fn end_stream(&self) -> Result<()> {
        let end_chunk = ResponseChunk {
            id: Uuid::new_v4(),
            data: Vec::new(),
            chunk_type: ChunkType::End,
            timestamp: chrono::Utc::now(),
            size: 0,
            stream_id: self.stream_id,
        };

        self.sender.send(end_chunk).await?;
        self.stats.write().await.record_stream_ended();

        Ok(())
    }

    /// Get current stats
    pub async fn get_stats(&self) -> ChunkStats {
        self.stats.read().await.clone()
    }

    /// Get buffer state
    pub async fn get_buffer_state(&self) -> ChunkBufferState {
        let buffer = self.buffer.read().await;
        ChunkBufferState {
            chunk_count: buffer.chunk_count(),
            total_size: buffer.total_size(),
            buffer_usage: buffer.buffer_usage(),
        }
    }
}

/// Chunk configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkConfig {
    /// Buffer size for chunks
    pub buffer_size: usize,

    /// Maximum buffer size in bytes
    pub max_buffer_size: usize,

    /// Chunk size for splitting large data
    pub chunk_size: usize,

    /// Compression settings
    pub compression: CompressionConfig,

    /// Chunking strategy
    pub strategy: ChunkingStrategy,

    /// Flush interval
    pub flush_interval: Duration,

    /// Enable checksums
    pub enable_checksums: bool,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            buffer_size: 100,
            max_buffer_size: 10 * 1024 * 1024, // 10MB
            chunk_size: 64 * 1024,             // 64KB
            compression: CompressionConfig::default(),
            strategy: ChunkingStrategy::FixedSize,
            flush_interval: Duration::from_millis(10),
            enable_checksums: false,
        }
    }
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Enable compression
    pub enabled: bool,

    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,

    /// Compression level (1-9)
    pub level: u8,

    /// Minimum size to compress
    pub min_size: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: CompressionAlgorithm::Gzip,
            level: 6,
            min_size: 1024, // 1KB
        }
    }
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    Gzip,
    Deflate,
    Brotli,
    Zstd,
}

/// Chunking strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkingStrategy {
    /// Fixed size chunks
    FixedSize,
    /// Adaptive size based on content
    Adaptive,
    /// Line-based chunking for text
    LineBased,
    /// Sentence-based chunking
    SentenceBased,
    /// Token-based chunking for LLM output
    TokenBased,
}

/// Response chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseChunk {
    pub id: Uuid,
    pub data: Vec<u8>,
    pub chunk_type: ChunkType,
    #[serde(with = "chrono::serde::ts_milliseconds")]
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub size: usize,
    pub stream_id: Uuid,
}

impl ResponseChunk {
    /// Get data as string
    pub fn as_string(&self) -> Result<String> {
        String::from_utf8(self.data.clone()).map_err(|e| anyhow!("Invalid UTF-8: {}", e))
    }

    /// Get data as JSON
    pub fn as_json<T: for<'de> Deserialize<'de>>(&self) -> Result<T> {
        serde_json::from_slice(&self.data).map_err(|e| anyhow!("Invalid JSON: {}", e))
    }

    /// Get data as bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}

/// Chunk types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkType {
    /// Text data
    Text,
    /// JSON data
    Json,
    /// Binary data
    Binary,
    /// End of stream marker
    End,
    /// Metadata chunk
    Metadata,
}

/// Chunk buffer for accumulating chunks
#[derive(Debug)]
pub struct ChunkBuffer {
    chunks: VecDeque<ResponseChunk>,
    max_size: usize,
    current_size: usize,
}

impl ChunkBuffer {
    pub fn new(max_size: usize) -> Self {
        Self {
            chunks: VecDeque::new(),
            max_size,
            current_size: 0,
        }
    }

    pub fn add_chunk(&mut self, chunk: ResponseChunk) {
        self.current_size += chunk.size;
        self.chunks.push_back(chunk);

        // Remove old chunks if buffer is full
        while self.current_size > self.max_size && !self.chunks.is_empty() {
            if let Some(old_chunk) = self.chunks.pop_front() {
                self.current_size -= old_chunk.size;
            }
        }
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn total_size(&self) -> usize {
        self.current_size
    }

    pub fn buffer_usage(&self) -> f32 {
        self.current_size as f32 / self.max_size as f32
    }

    pub fn get_chunks(&self) -> &VecDeque<ResponseChunk> {
        &self.chunks
    }

    pub fn clear(&mut self) {
        self.chunks.clear();
        self.current_size = 0;
    }
}

/// Chunked response wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkedResponse {
    pub stream_id: Uuid,
    pub chunks: Vec<ResponseChunk>,
    pub total_size: usize,
    pub chunk_count: usize,
    #[serde(skip, default = "chrono::Utc::now")]
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub content_type: String,
}

impl ChunkedResponse {
    pub fn new(stream_id: Uuid, content_type: String) -> Self {
        Self {
            stream_id,
            chunks: Vec::new(),
            total_size: 0,
            chunk_count: 0,
            created_at: chrono::Utc::now(),
            content_type,
        }
    }

    pub fn add_chunk(&mut self, chunk: ResponseChunk) {
        self.total_size += chunk.size;
        self.chunk_count += 1;
        self.chunks.push(chunk);
    }

    /// Create an HTTP response with chunked transfer encoding
    pub fn to_http_response(self) -> Result<Response<Body>> {
        let chunks = self.chunks;
        let stream = stream::iter(
            chunks.into_iter().map(|chunk| Ok::<_, std::io::Error>(Bytes::from(chunk.data))),
        );

        let body = Body::from_stream(stream);

        Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, self.content_type)
            .header(header::TRANSFER_ENCODING, "chunked")
            .header(header::CACHE_CONTROL, "no-cache")
            .body(body.into())?)
    }
}

/// Chunk buffer state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkBufferState {
    pub chunk_count: usize,
    pub total_size: usize,
    pub buffer_usage: f32,
}

/// Chunk statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkStats {
    pub chunks_sent: usize,
    pub bytes_sent: usize,
    #[serde(skip)]
    pub stream_started: Option<Instant>,
    #[serde(skip)]
    pub stream_ended: Option<Instant>,
    pub avg_chunk_size: f64,
    pub total_duration_ms: Option<f64>,
    pub throughput_mbps: Option<f64>,
}

impl ChunkStats {
    pub fn new() -> Self {
        Self {
            chunks_sent: 0,
            bytes_sent: 0,
            stream_started: Some(Instant::now()),
            stream_ended: None,
            avg_chunk_size: 0.0,
            total_duration_ms: None,
            throughput_mbps: None,
        }
    }

    pub fn record_chunk_sent(&mut self, size: usize) {
        self.chunks_sent += 1;
        self.bytes_sent += size;
        self.avg_chunk_size = self.bytes_sent as f64 / self.chunks_sent as f64;

        if let Some(start) = self.stream_started {
            let elapsed = start.elapsed().as_millis() as f64;
            self.throughput_mbps =
                Some((self.bytes_sent as f64 / 1_000_000.0) / (elapsed / 1000.0));
        }
    }

    pub fn record_stream_ended(&mut self) {
        self.stream_ended = Some(Instant::now());

        if let (Some(start), Some(end)) = (self.stream_started, self.stream_ended) {
            self.total_duration_ms = Some(end.duration_since(start).as_millis() as f64);
        }
    }
}

/// Helper function to create a streaming response
pub async fn create_streaming_response(
    content_type: &str,
    rx: mpsc::Receiver<ResponseChunk>,
) -> Result<Response<Body>> {
    let stream = tokio_stream::wrappers::ReceiverStream::new(rx)
        .map(|chunk| Ok::<_, std::io::Error>(Bytes::from(chunk.data)));

    let body = Body::from_stream(stream);

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type)
        .header(header::TRANSFER_ENCODING, "chunked")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(body.into())?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_chunk_stream() {
        let config = ChunkConfig::default();
        let stream_id = Uuid::new_v4();
        let stream = ChunkStream::new(config, stream_id);

        stream.add_text("Hello".to_string()).await.unwrap();
        stream.add_text(" World".to_string()).await.unwrap();
        stream.end_stream().await.unwrap();

        let stats = stream.get_stats().await;
        assert_eq!(stats.chunks_sent, 2);
        assert!(stats.bytes_sent > 0);
    }

    #[test]
    fn test_chunk_buffer() {
        let mut buffer = ChunkBuffer::new(100);

        let chunk = ResponseChunk {
            id: Uuid::new_v4(),
            data: vec![1, 2, 3, 4, 5],
            chunk_type: ChunkType::Binary,
            timestamp: chrono::Utc::now(),
            size: 5,
            stream_id: Uuid::new_v4(),
        };

        buffer.add_chunk(chunk);

        assert_eq!(buffer.chunk_count(), 1);
        assert_eq!(buffer.total_size(), 5);
    }

    #[test]
    fn test_chunked_response() {
        let stream_id = Uuid::new_v4();
        let mut response = ChunkedResponse::new(stream_id, "text/plain".to_string());

        let chunk = ResponseChunk {
            id: Uuid::new_v4(),
            data: "test".as_bytes().to_vec(),
            chunk_type: ChunkType::Text,
            timestamp: chrono::Utc::now(),
            size: 4,
            stream_id,
        };

        response.add_chunk(chunk);

        assert_eq!(response.chunk_count, 1);
        assert_eq!(response.total_size, 4);
    }
}
