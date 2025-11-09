//! WebSocket Streaming Implementation
//!
//! Provides bidirectional WebSocket streaming for interactive model inference
//! and real-time communication between clients and the inference server.

use anyhow::Result;
use axum::{
    extract::{ws::WebSocket, WebSocketUpgrade},
    response::Response,
};
use futures::{sink::SinkExt, stream::StreamExt};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, RwLock};
use uuid::Uuid;

/// WebSocket handler for managing WebSocket connections
#[derive(Clone)]
pub struct WebSocketHandler {
    config: WsConfig,
    connections: Arc<RwLock<HashMap<Uuid, WsConnection>>>,
    metrics: Arc<WsMetrics>,
}

impl WebSocketHandler {
    /// Create a new WebSocket handler
    pub fn new(config: WsConfig) -> Self {
        Self {
            config,
            connections: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(WsMetrics::new()),
        }
    }

    /// Handle WebSocket upgrade
    pub async fn handle_upgrade(
        &self,
        ws: WebSocketUpgrade,
        request_id: Option<String>,
    ) -> Response {
        let handler = self.clone();

        ws.on_upgrade(move |socket| async move { handler.handle_socket(socket, request_id).await })
    }

    /// Handle WebSocket connection
    async fn handle_socket(&self, socket: WebSocket, request_id: Option<String>) {
        let connection_id = Uuid::new_v4();
        let (tx, mut rx) = mpsc::channel(self.config.buffer_size);

        let connection = WsConnection {
            id: connection_id,
            request_id: request_id.clone(),
            sender: tx,
            created_at: Instant::now(),
            last_activity: Instant::now(),
        };

        self.connections.write().await.insert(connection_id, connection);
        self.metrics.record_connection_opened().await;

        // Split the socket
        let (mut ws_sender, mut ws_receiver) = socket.split();

        // Spawn receiver task
        let connections_recv = self.connections.clone();
        let recv_task = tokio::spawn(async move {
            while let Some(msg) = ws_receiver.next().await {
                match msg {
                    Ok(axum::extract::ws::Message::Text(text)) => {
                        // Handle incoming text message
                        if let Some(connection) =
                            connections_recv.write().await.get_mut(&connection_id)
                        {
                            connection.last_activity = Instant::now();
                        }

                        // Process message (could emit events here)
                        tracing::debug!("Received WebSocket text: {}", text);
                    },
                    Ok(axum::extract::ws::Message::Binary(data)) => {
                        // Handle incoming binary message
                        if let Some(connection) =
                            connections_recv.write().await.get_mut(&connection_id)
                        {
                            connection.last_activity = Instant::now();
                        }

                        tracing::debug!("Received WebSocket binary: {} bytes", data.len());
                    },
                    Ok(axum::extract::ws::Message::Ping(_)) => {
                        // Handle ping - pong will be sent automatically
                        if let Some(connection) =
                            connections_recv.write().await.get_mut(&connection_id)
                        {
                            connection.last_activity = Instant::now();
                        }
                    },
                    Ok(axum::extract::ws::Message::Close(_)) => {
                        break;
                    },
                    Err(e) => {
                        tracing::error!("WebSocket error: {}", e);
                        break;
                    },
                    _ => {},
                }
            }
        });

        // Spawn sender task
        let _connections_send = self.connections.clone();
        let metrics = self.metrics.clone();
        let send_task = tokio::spawn(async move {
            while let Some(message) = rx.recv().await {
                let ws_message = match message.message_type {
                    WsMessageType::Text => axum::extract::ws::Message::Text(message.data.into()),
                    WsMessageType::Binary => {
                        axum::extract::ws::Message::Binary(message.data.into_bytes().into())
                    },
                    WsMessageType::Json => axum::extract::ws::Message::Text(message.data.into()),
                    WsMessageType::Ping => axum::extract::ws::Message::Ping(Vec::new().into()),
                    WsMessageType::Close => axum::extract::ws::Message::Close(None),
                };

                if ws_sender.send(ws_message).await.is_err() {
                    break;
                }

                metrics.record_message_sent().await;
            }
        });

        // Wait for either task to complete
        tokio::select! {
            _ = recv_task => {},
            _ = send_task => {},
        }

        // Clean up connection
        let duration =
            if let Some(connection) = self.connections.write().await.remove(&connection_id) {
                connection.created_at.elapsed()
            } else {
                Duration::from_secs(0)
            };

        self.metrics.record_connection_closed(duration).await;
    }

    /// Send message to specific connection
    pub async fn send_to_connection(&self, connection_id: Uuid, message: WsMessage) -> Result<()> {
        if let Some(connection) = self.connections.read().await.get(&connection_id) {
            connection.sender.send(message).await?;
        }

        Ok(())
    }

    /// Send message to all connections for a request
    pub async fn send_to_request(&self, request_id: &str, message: WsMessage) -> Result<()> {
        let connections = self.connections.read().await;
        let mut futures = Vec::new();

        for connection in connections.values() {
            if connection.request_id.as_ref() == Some(&request_id.to_string()) {
                let tx = connection.sender.clone();
                let msg_clone = message.clone();
                futures.push(async move {
                    let _ = tx.send(msg_clone).await;
                });
            }
        }

        futures::future::join_all(futures).await;

        Ok(())
    }

    /// Broadcast message to all connections
    pub async fn broadcast(&self, message: WsMessage) -> Result<()> {
        let connections = self.connections.read().await;
        let mut futures = Vec::new();

        for connection in connections.values() {
            let tx = connection.sender.clone();
            let msg_clone = message.clone();
            futures.push(async move {
                let _ = tx.send(msg_clone).await;
            });
        }

        futures::future::join_all(futures).await;
        self.metrics.record_broadcast().await;

        Ok(())
    }

    /// Get connection statistics
    pub async fn get_stats(&self) -> WsStats {
        let active_connections = self.connections.read().await.len();

        WsStats {
            active_connections,
            total_connections: self.metrics.total_connections_opened().await,
            total_messages_sent: self.metrics.total_messages_sent().await,
            total_broadcasts: self.metrics.total_broadcasts().await,
            avg_connection_duration_ms: self.metrics.avg_connection_duration_ms().await,
        }
    }

    /// Cleanup inactive connections
    pub async fn cleanup_connections(&self) -> Result<()> {
        let timeout = self.config.connection_timeout;
        let now = Instant::now();
        let mut to_remove = Vec::new();

        {
            let connections = self.connections.read().await;
            for (id, connection) in connections.iter() {
                if now.duration_since(connection.last_activity) > timeout {
                    to_remove.push(*id);
                }
            }
        }

        for id in to_remove {
            if let Some(connection) = self.connections.write().await.remove(&id) {
                let duration = connection.created_at.elapsed();
                self.metrics.record_connection_closed(duration).await;
            }
        }

        Ok(())
    }
}

/// WebSocket configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsConfig {
    /// Buffer size for message channel
    pub buffer_size: usize,

    /// Connection timeout
    pub connection_timeout: Duration,

    /// Maximum concurrent connections
    pub max_connections: usize,

    /// Message size limit
    pub max_message_size: usize,

    /// Enable compression
    pub enable_compression: bool,

    /// Ping interval
    pub ping_interval: Duration,

    /// Maximum frame size
    pub max_frame_size: usize,
}

impl Default for WsConfig {
    fn default() -> Self {
        Self {
            buffer_size: 100,
            connection_timeout: Duration::from_secs(300),
            max_connections: 1000,
            max_message_size: 1024 * 1024, // 1MB
            enable_compression: true,
            ping_interval: Duration::from_secs(30),
            max_frame_size: 64 * 1024, // 64KB
        }
    }
}

/// WebSocket connection information
#[derive(Debug)]
pub struct WsConnection {
    pub id: Uuid,
    pub request_id: Option<String>,
    pub sender: mpsc::Sender<WsMessage>,
    pub created_at: Instant,
    pub last_activity: Instant,
}

/// WebSocket message structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsMessage {
    pub id: Uuid,
    pub message_type: WsMessageType,
    pub data: String,
    pub timestamp: u64, // Unix timestamp in seconds
}

impl WsMessage {
    /// Create a new WebSocket message
    pub fn new(message_type: WsMessageType, data: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            message_type,
            data,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    /// Create a text message
    pub fn text(data: String) -> Self {
        Self::new(WsMessageType::Text, data)
    }

    /// Create a JSON message
    pub fn json(value: serde_json::Value) -> Self {
        let data = serde_json::to_string(&value).unwrap_or_default();
        Self::new(WsMessageType::Json, data)
    }

    /// Create a binary message
    pub fn binary(data: Vec<u8>) -> Self {
        let data_str = String::from_utf8_lossy(&data).to_string();
        Self::new(WsMessageType::Binary, data_str)
    }

    /// Create a ping message
    pub fn ping() -> Self {
        Self::new(WsMessageType::Ping, String::new())
    }

    /// Create a close message
    pub fn close() -> Self {
        Self::new(WsMessageType::Close, String::new())
    }

    /// Create a token message for streaming
    pub fn token(token: String) -> Self {
        let value = serde_json::json!({
            "type": "token",
            "token": token,
            "timestamp": chrono::Utc::now().timestamp_millis()
        });
        Self::json(value)
    }

    /// Create a completion message
    pub fn completion(text: String, reason: String) -> Self {
        let value = serde_json::json!({
            "type": "completion",
            "text": text,
            "reason": reason,
            "timestamp": chrono::Utc::now().timestamp_millis()
        });
        Self::json(value)
    }

    /// Create an error message
    pub fn error(error: String) -> Self {
        let value = serde_json::json!({
            "type": "error",
            "error": error,
            "timestamp": chrono::Utc::now().timestamp_millis()
        });
        Self::json(value)
    }
}

/// WebSocket message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WsMessageType {
    /// Plain text message
    Text,
    /// Binary message
    Binary,
    /// JSON formatted message
    Json,
    /// Ping frame
    Ping,
    /// Close frame
    Close,
}

/// WebSocket statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsStats {
    pub active_connections: usize,
    pub total_connections: u64,
    pub total_messages_sent: u64,
    pub total_broadcasts: u64,
    pub avg_connection_duration_ms: f64,
}

/// WebSocket metrics collector
#[derive(Debug)]
pub struct WsMetrics {
    connections_opened: Arc<RwLock<u64>>,
    messages_sent: Arc<RwLock<u64>>,
    broadcasts: Arc<RwLock<u64>>,
    connection_durations: Arc<RwLock<Vec<Duration>>>,
}

impl WsMetrics {
    pub fn new() -> Self {
        Self {
            connections_opened: Arc::new(RwLock::new(0)),
            messages_sent: Arc::new(RwLock::new(0)),
            broadcasts: Arc::new(RwLock::new(0)),
            connection_durations: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn record_connection_opened(&self) {
        *self.connections_opened.write().await += 1;
    }

    pub async fn record_message_sent(&self) {
        *self.messages_sent.write().await += 1;
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

    pub async fn total_messages_sent(&self) -> u64 {
        *self.messages_sent.read().await
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

/// WebSocket error types
#[derive(Debug, thiserror::Error)]
pub enum WsError {
    #[error("Connection not found: {0}")]
    ConnectionNotFound(Uuid),

    #[error("Channel send error: {0}")]
    ChannelSend(#[from] mpsc::error::SendError<WsMessage>),

    #[error("Maximum connections reached")]
    MaxConnectionsReached,

    #[error("Message too large: {0} bytes")]
    MessageTooLarge(usize),

    #[error("Invalid message format: {0}")]
    InvalidMessageFormat(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ws_handler() {
        let config = WsConfig::default();
        let handler = WebSocketHandler::new(config);

        let stats = handler.get_stats().await;
        assert_eq!(stats.active_connections, 0);
    }

    #[test]
    fn test_ws_message_creation() {
        let msg = WsMessage::text("hello".to_string());
        assert!(matches!(msg.message_type, WsMessageType::Text));
        assert_eq!(msg.data, "hello");

        let token_msg = WsMessage::token("world".to_string());
        assert!(matches!(token_msg.message_type, WsMessageType::Json));
    }

    #[test]
    fn test_ws_message_serialization() {
        let msg = WsMessage::text("test".to_string());
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: WsMessage = serde_json::from_str(&json).unwrap();

        assert_eq!(msg.data, deserialized.data);
    }
}
