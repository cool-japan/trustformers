//! Server-Sent Events (SSE) Implementation
//!
//! Provides real-time streaming capabilities using Server-Sent Events protocol
//! for live model inference results and progress updates.

use anyhow::Result;
use axum::response::{IntoResponse, Response, Sse};
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    convert::Infallible,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, RwLock};
use uuid::Uuid;

/// SSE handler for managing Server-Sent Event connections
#[derive(Clone)]
pub struct SseHandler {
    config: SseConfig,
    connections: Arc<RwLock<HashMap<Uuid, SseConnection>>>,
    metrics: Arc<SseMetrics>,
}

impl SseHandler {
    /// Create a new SSE handler
    pub fn new(config: SseConfig) -> Self {
        Self {
            config,
            connections: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(SseMetrics::new()),
        }
    }

    /// Handle new SSE connection
    pub async fn handle_connection(&self, request_id: Option<String>) -> Result<Response> {
        let connection_id = Uuid::new_v4();
        let (tx, rx) = mpsc::channel(self.config.buffer_size);

        let connection = SseConnection {
            id: connection_id,
            request_id: request_id.clone(),
            sender: tx,
            created_at: Instant::now(),
            last_ping: Instant::now(),
        };

        self.connections.write().await.insert(connection_id, connection);
        self.metrics.record_connection_opened().await;

        // Create SSE stream
        let stream = self.create_event_stream(rx).await;

        // Start heartbeat task
        self.start_heartbeat_task(connection_id).await;

        Ok(Sse::new(stream)
            .keep_alive(
                axum::response::sse::KeepAlive::new()
                    .interval(self.config.heartbeat_interval)
                    .text("heartbeat"),
            )
            .into_response())
    }

    /// Send event to specific connection
    pub async fn send_to_connection(&self, connection_id: Uuid, event: SseEvent) -> Result<()> {
        if let Some(connection) = self.connections.read().await.get(&connection_id) {
            connection.sender.send(event).await?;
            self.metrics.record_event_sent().await;
        }

        Ok(())
    }

    /// Send event to all connections for a request
    pub async fn send_to_request(&self, request_id: &str, event: SseEvent) -> Result<()> {
        let connections = self.connections.read().await;
        let mut futures = Vec::new();

        for connection in connections.values() {
            if connection.request_id.as_ref() == Some(&request_id.to_string()) {
                let tx = connection.sender.clone();
                let event_clone = event.clone();
                futures.push(async move {
                    let _ = tx.send(event_clone).await;
                });
            }
        }

        // Send to all matching connections concurrently
        futures::future::join_all(futures).await;

        Ok(())
    }

    /// Broadcast event to all connections
    pub async fn broadcast(&self, event: SseEvent) -> Result<()> {
        let connections = self.connections.read().await;
        let mut futures = Vec::new();

        for connection in connections.values() {
            let tx = connection.sender.clone();
            let event_clone = event.clone();
            futures.push(async move {
                let _ = tx.send(event_clone).await;
            });
        }

        futures::future::join_all(futures).await;
        self.metrics.record_broadcast().await;

        Ok(())
    }

    /// Close connection
    pub async fn close_connection(&self, connection_id: Uuid) -> Result<()> {
        if let Some(connection) = self.connections.write().await.remove(&connection_id) {
            let duration = connection.created_at.elapsed();
            self.metrics.record_connection_closed(duration).await;
        }

        Ok(())
    }

    /// Get connection statistics
    pub async fn get_stats(&self) -> SseStats {
        let active_connections = self.connections.read().await.len();

        SseStats {
            active_connections,
            total_connections: self.metrics.total_connections_opened().await,
            total_events_sent: self.metrics.total_events_sent().await,
            total_broadcasts: self.metrics.total_broadcasts().await,
            avg_connection_duration_ms: self.metrics.avg_connection_duration_ms().await,
        }
    }

    /// Cleanup disconnected connections
    pub async fn cleanup_connections(&self) -> Result<()> {
        let timeout = self.config.connection_timeout;
        let now = Instant::now();
        let mut to_remove = Vec::new();

        {
            let connections = self.connections.read().await;
            for (id, connection) in connections.iter() {
                if now.duration_since(connection.last_ping) > timeout {
                    to_remove.push(*id);
                }
            }
        }

        for id in to_remove {
            self.close_connection(id).await?;
        }

        Ok(())
    }

    /// Create event stream for SSE
    async fn create_event_stream(
        &self,
        rx: mpsc::Receiver<SseEvent>,
    ) -> impl Stream<Item = Result<axum::response::sse::Event, Infallible>> {
        stream::unfold(rx, |mut rx| async {
            match rx.recv().await {
                Some(event) => {
                    let sse_event = axum::response::sse::Event::default()
                        .event(event.event_type.to_string())
                        .data(event.data)
                        .id(event.id.to_string());

                    Some((Ok(sse_event), rx))
                },
                None => None,
            }
        })
    }

    /// Start heartbeat task for connection
    async fn start_heartbeat_task(&self, connection_id: Uuid) {
        let connections = self.connections.clone();
        let interval = self.config.heartbeat_interval;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(interval);

            loop {
                interval.tick().await;

                // Update last ping time
                if let Some(connection) = connections.write().await.get_mut(&connection_id) {
                    connection.last_ping = Instant::now();

                    // Send heartbeat event
                    let heartbeat = SseEvent {
                        id: Uuid::new_v4(),
                        event_type: SseEventType::Heartbeat,
                        data: "ping".to_string(),
                        timestamp: chrono::Utc::now(),
                    };

                    if connection.sender.send(heartbeat).await.is_err() {
                        break; // Connection closed
                    }
                } else {
                    break; // Connection no longer exists
                }
            }
        });
    }
}

/// SSE configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SseConfig {
    /// Buffer size for event channel
    pub buffer_size: usize,

    /// Heartbeat interval
    pub heartbeat_interval: Duration,

    /// Connection timeout
    pub connection_timeout: Duration,

    /// Maximum concurrent connections
    pub max_connections: usize,

    /// Enable compression
    pub enable_compression: bool,

    /// CORS settings
    pub cors_origins: Vec<String>,
}

impl Default for SseConfig {
    fn default() -> Self {
        Self {
            buffer_size: 100,
            heartbeat_interval: Duration::from_secs(30),
            connection_timeout: Duration::from_secs(300),
            max_connections: 1000,
            enable_compression: true,
            cors_origins: vec!["*".to_string()],
        }
    }
}

/// SSE connection information
#[derive(Debug)]
pub struct SseConnection {
    pub id: Uuid,
    pub request_id: Option<String>,
    pub sender: mpsc::Sender<SseEvent>,
    pub created_at: Instant,
    pub last_ping: Instant,
}

/// SSE event structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SseEvent {
    pub id: Uuid,
    pub event_type: SseEventType,
    pub data: String,
    #[serde(with = "chrono::serde::ts_milliseconds")]
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl SseEvent {
    /// Create a new SSE event
    pub fn new(event_type: SseEventType, data: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            event_type,
            data,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Create a token event for streaming LLM generation
    pub fn token(token: String) -> Self {
        Self::new(SseEventType::Token, token)
    }

    /// Create a progress event
    pub fn progress(percentage: f32, message: String) -> Self {
        let data = serde_json::json!({
            "percentage": percentage,
            "message": message
        })
        .to_string();

        Self::new(SseEventType::Progress, data)
    }

    /// Create a completion event
    pub fn completion(result: String) -> Self {
        Self::new(SseEventType::Completion, result)
    }

    /// Create an error event
    pub fn error(error: String) -> Self {
        Self::new(SseEventType::Error, error)
    }
}

/// SSE event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SseEventType {
    /// Token generation for LLM streaming
    Token,
    /// Progress update
    Progress,
    /// Task completion
    Completion,
    /// Error occurred
    Error,
    /// Heartbeat ping
    Heartbeat,
    /// Custom event
    Custom(String),
}

impl ToString for SseEventType {
    fn to_string(&self) -> String {
        match self {
            Self::Token => "token".to_string(),
            Self::Progress => "progress".to_string(),
            Self::Completion => "completion".to_string(),
            Self::Error => "error".to_string(),
            Self::Heartbeat => "heartbeat".to_string(),
            Self::Custom(name) => name.clone(),
        }
    }
}

/// SSE statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SseStats {
    pub active_connections: usize,
    pub total_connections: u64,
    pub total_events_sent: u64,
    pub total_broadcasts: u64,
    pub avg_connection_duration_ms: f64,
}

/// SSE metrics collector
#[derive(Debug)]
pub struct SseMetrics {
    connections_opened: Arc<RwLock<u64>>,
    events_sent: Arc<RwLock<u64>>,
    broadcasts: Arc<RwLock<u64>>,
    connection_durations: Arc<RwLock<Vec<Duration>>>,
}

impl Default for SseMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl SseMetrics {
    pub fn new() -> Self {
        Self {
            connections_opened: Arc::new(RwLock::new(0)),
            events_sent: Arc::new(RwLock::new(0)),
            broadcasts: Arc::new(RwLock::new(0)),
            connection_durations: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn record_connection_opened(&self) {
        *self.connections_opened.write().await += 1;
    }

    pub async fn record_event_sent(&self) {
        *self.events_sent.write().await += 1;
    }

    pub async fn record_broadcast(&self) {
        *self.broadcasts.write().await += 1;
    }

    pub async fn record_connection_closed(&self, duration: Duration) {
        self.connection_durations.write().await.push(duration);
    }

    pub async fn total_connections_opened(&self) -> u64 {
        *self.connections_opened.read().await
    }

    pub async fn total_events_sent(&self) -> u64 {
        *self.events_sent.read().await
    }

    pub async fn total_broadcasts(&self) -> u64 {
        *self.broadcasts.read().await
    }

    pub async fn avg_connection_duration_ms(&self) -> f64 {
        let durations = self.connection_durations.read().await;
        if durations.is_empty() {
            0.0
        } else {
            let total: f64 = durations.iter().map(|d| d.as_millis() as f64).sum();
            total / durations.len() as f64
        }
    }
}

/// SSE error types
#[derive(Debug, thiserror::Error)]
pub enum SseError {
    #[error("Connection not found: {0}")]
    ConnectionNotFound(Uuid),

    #[error("Channel send error: {0}")]
    ChannelSend(#[from] mpsc::error::SendError<SseEvent>),

    #[error("Maximum connections reached")]
    MaxConnectionsReached,

    #[error("Invalid event data: {0}")]
    InvalidEventData(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sse_handler() {
        let config = SseConfig::default();
        let handler = SseHandler::new(config);

        // This would be a full integration test with actual HTTP connections
        let stats = handler.get_stats().await;
        assert_eq!(stats.active_connections, 0);
    }

    #[test]
    fn test_sse_event_creation() {
        let event = SseEvent::token("hello".to_string());
        assert!(matches!(event.event_type, SseEventType::Token));
        assert_eq!(event.data, "hello");
    }

    #[test]
    fn test_event_type_to_string() {
        assert_eq!(SseEventType::Token.to_string(), "token");
        assert_eq!(SseEventType::Progress.to_string(), "progress");
        assert_eq!(SseEventType::Custom("test".to_string()).to_string(), "test");
    }
}
