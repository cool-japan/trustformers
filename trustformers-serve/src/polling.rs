// Allow dead code for infrastructure under development
#![allow(dead_code)]

//! Long Polling Support
//!
//! Provides long polling functionality for real-time updates and notifications
//! without the complexity of WebSocket connections.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use tokio::{
    sync::{broadcast, oneshot, Mutex},
    time::sleep,
};
use uuid::Uuid;

/// Long polling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongPollingConfig {
    /// Maximum time to wait for new data (in seconds)
    pub timeout_seconds: u64,

    /// Maximum number of concurrent polling connections
    pub max_concurrent_connections: usize,

    /// Cleanup interval for expired connections (in seconds)
    pub cleanup_interval_seconds: u64,

    /// Buffer size for event channels
    pub event_buffer_size: usize,
}

impl Default for LongPollingConfig {
    fn default() -> Self {
        Self {
            timeout_seconds: 30,
            max_concurrent_connections: 1000,
            cleanup_interval_seconds: 60,
            event_buffer_size: 100,
        }
    }
}

/// Event types for long polling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PollEvent {
    /// Inference job completed
    InferenceComplete {
        job_id: String,
        result: String,
        processing_time_ms: f64,
    },

    /// Model deployment status changed
    ModelDeployment {
        model_id: String,
        status: String,
        message: String,
    },

    /// System health status update
    HealthStatusUpdate {
        status: String,
        timestamp: String,
        details: String,
    },

    /// Batch processing completed
    BatchComplete {
        batch_id: String,
        processed_count: usize,
        failed_count: usize,
    },

    /// Custom event
    Custom {
        event_type: String,
        data: serde_json::Value,
    },
}

/// Long polling request
#[derive(Debug, Deserialize)]
pub struct LongPollRequest {
    /// Event types to subscribe to
    pub event_types: Vec<String>,

    /// Client ID for tracking
    pub client_id: Option<String>,

    /// Custom timeout override
    pub timeout_seconds: Option<u64>,

    /// Last event ID received (for resuming)
    pub last_event_id: Option<String>,
}

/// Long polling response
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct LongPollResponse {
    /// Events received
    pub events: Vec<PollEventWithId>,

    /// Whether this is a timeout response
    pub timeout: bool,

    /// Next polling URL or token
    pub next_token: Option<String>,

    /// Server timestamp
    pub timestamp: String,
}

/// Event with ID for tracking
#[derive(Debug, Clone, Serialize)]
pub struct PollEventWithId {
    /// Event ID
    pub id: String,

    /// Event type
    pub event_type: String,

    /// Event data
    pub event: PollEvent,

    /// Event timestamp
    pub timestamp: String,
}

/// Active polling connection
#[derive(Debug)]
struct PollingConnection {
    /// Connection ID
    id: String,

    /// Client ID
    client_id: Option<String>,

    /// Event types subscribed to
    event_types: Vec<String>,

    /// Response sender
    response_sender: oneshot::Sender<LongPollResponse>,

    /// Connection start time
    start_time: Instant,

    /// Connection timeout
    timeout: Duration,

    /// Last event ID
    last_event_id: Option<String>,
}

/// Long polling service
pub struct LongPollingService {
    config: LongPollingConfig,

    /// Active polling connections
    connections: Arc<RwLock<HashMap<String, PollingConnection>>>,

    /// Event broadcaster
    event_sender: broadcast::Sender<PollEventWithId>,

    /// Event receiver for service
    _event_receiver: broadcast::Receiver<PollEventWithId>,

    /// Cleanup task handle
    cleanup_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,

    /// Event history for missed events
    event_history: Arc<Mutex<Vec<PollEventWithId>>>,

    /// Statistics
    stats: Arc<RwLock<LongPollingStats>>,
}

/// Long polling statistics
#[derive(Debug, Default, Clone, Serialize, utoipa::ToSchema)]
pub struct LongPollingStats {
    /// Total active connections
    pub active_connections: usize,

    /// Total events published
    pub total_events_published: u64,

    /// Total connections served
    pub total_connections_served: u64,

    /// Total timeouts
    pub total_timeouts: u64,

    /// Average connection duration
    pub avg_connection_duration_ms: f64,

    /// Events per minute
    pub events_per_minute: f64,
}

impl LongPollingService {
    /// Create a new long polling service
    pub fn new(config: LongPollingConfig) -> Self {
        let (event_sender, event_receiver) = broadcast::channel(config.event_buffer_size);

        let service = Self {
            config,
            connections: Arc::new(RwLock::new(HashMap::new())),
            event_sender,
            _event_receiver: event_receiver,
            cleanup_handle: Arc::new(Mutex::new(None)),
            event_history: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(RwLock::new(LongPollingStats::default())),
        };

        service
    }

    /// Start the long polling service
    pub async fn start(&self) -> Result<()> {
        // Start cleanup task
        let connections = self.connections.clone();
        let cleanup_interval = Duration::from_secs(self.config.cleanup_interval_seconds);

        let cleanup_handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(cleanup_interval);

            loop {
                interval.tick().await;

                // Clean up expired connections
                let mut connections_lock = connections.write().unwrap();
                let now = Instant::now();

                let expired_connections: Vec<String> = connections_lock
                    .iter()
                    .filter(|(_, conn)| now.duration_since(conn.start_time) > conn.timeout)
                    .map(|(id, _)| id.clone())
                    .collect();

                for conn_id in expired_connections {
                    if let Some(conn) = connections_lock.remove(&conn_id) {
                        // Send timeout response
                        let _ = conn.response_sender.send(LongPollResponse {
                            events: vec![],
                            timeout: true,
                            next_token: None,
                            timestamp: chrono::Utc::now().to_rfc3339(),
                        });
                    }
                }
            }
        });

        *self.cleanup_handle.lock().await = Some(cleanup_handle);

        Ok(())
    }

    /// Stop the long polling service
    pub async fn stop(&self) -> Result<()> {
        if let Some(handle) = self.cleanup_handle.lock().await.take() {
            handle.abort();
        }

        // Close all active connections
        let mut connections_lock = self.connections.write().unwrap();
        for (_, conn) in connections_lock.drain() {
            let _ = conn.response_sender.send(LongPollResponse {
                events: vec![],
                timeout: true,
                next_token: None,
                timestamp: chrono::Utc::now().to_rfc3339(),
            });
        }

        Ok(())
    }

    /// Handle a long polling request
    pub async fn poll(&self, request: LongPollRequest) -> Result<LongPollResponse> {
        // Check connection limits
        {
            let connections_lock = self.connections.read().unwrap();
            if connections_lock.len() >= self.config.max_concurrent_connections {
                return Ok(LongPollResponse {
                    events: vec![],
                    timeout: true,
                    next_token: None,
                    timestamp: chrono::Utc::now().to_rfc3339(),
                });
            }
        }

        // Create connection
        let connection_id = Uuid::new_v4().to_string();
        let timeout_duration =
            Duration::from_secs(request.timeout_seconds.unwrap_or(self.config.timeout_seconds));

        let (response_sender, _response_receiver) = oneshot::channel();

        let connection = PollingConnection {
            id: connection_id.clone(),
            client_id: request.client_id,
            event_types: request.event_types,
            response_sender,
            start_time: Instant::now(),
            timeout: timeout_duration,
            last_event_id: request.last_event_id,
        };

        // Check for missed events
        let missed_events = self.get_missed_events(&connection).await;

        if !missed_events.is_empty() {
            // Return missed events immediately
            return Ok(LongPollResponse {
                events: missed_events,
                timeout: false,
                next_token: Some(connection_id),
                timestamp: chrono::Utc::now().to_rfc3339(),
            });
        }

        // Store connection
        {
            let mut connections_lock = self.connections.write().unwrap();
            connections_lock.insert(connection_id.clone(), connection);
        }

        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_connections_served += 1;
            stats.active_connections = self.connections.read().unwrap().len();
        }

        // Subscribe to events
        let mut event_receiver = self.event_sender.subscribe();

        // Wait for events or timeout
        let response = tokio::select! {
            // Wait for events
            result = async {
                let mut events = Vec::new();

                while let Ok(event) = event_receiver.recv().await {
                    // Check if connection is still active
                    let connection_active = {
                        let connections_lock = self.connections.read().unwrap();
                        connections_lock.contains_key(&connection_id)
                    };

                    if !connection_active {
                        break;
                    }

                    // Check if event matches subscription
                    let connections_lock = self.connections.read().unwrap();
                    if let Some(conn) = connections_lock.get(&connection_id) {
                        if conn.event_types.contains(&event.event_type) ||
                           conn.event_types.contains(&"*".to_string()) {
                            events.push(event);
                            break; // Send immediately on first matching event
                        }
                    }
                }

                LongPollResponse {
                    events,
                    timeout: false,
                    next_token: Some(connection_id.clone()),
                    timestamp: chrono::Utc::now().to_rfc3339(),
                }
            } => result,

            // Wait for timeout
            _ = sleep(timeout_duration) => {
                let mut stats = self.stats.write().unwrap();
                stats.total_timeouts += 1;

                LongPollResponse {
                    events: vec![],
                    timeout: true,
                    next_token: None,
                    timestamp: chrono::Utc::now().to_rfc3339(),
                }
            }
        };

        // Clean up connection
        {
            let mut connections_lock = self.connections.write().unwrap();
            connections_lock.remove(&connection_id);
        }

        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.active_connections = self.connections.read().unwrap().len();
        }

        Ok(response)
    }

    /// Publish an event to all subscribers
    pub async fn publish_event(&self, event: PollEvent) -> Result<()> {
        let event_id = Uuid::new_v4().to_string();
        let event_type = match &event {
            PollEvent::InferenceComplete { .. } => "inference_complete",
            PollEvent::ModelDeployment { .. } => "model_deployment",
            PollEvent::HealthStatusUpdate { .. } => "health_status",
            PollEvent::BatchComplete { .. } => "batch_complete",
            PollEvent::Custom { event_type, .. } => event_type,
        };

        let event_with_id = PollEventWithId {
            id: event_id,
            event_type: event_type.to_string(),
            event,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        // Add to history
        {
            let mut history = self.event_history.lock().await;
            history.push(event_with_id.clone());

            // Keep only last 1000 events
            if history.len() > 1000 {
                history.remove(0);
            }
        }

        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_events_published += 1;
        }

        // Broadcast event
        let _ = self.event_sender.send(event_with_id);

        Ok(())
    }

    /// Get missed events since last event ID
    async fn get_missed_events(&self, connection: &PollingConnection) -> Vec<PollEventWithId> {
        let history = self.event_history.lock().await;

        if let Some(last_event_id) = &connection.last_event_id {
            // Find events after the last event ID
            let mut found_last = false;
            let mut missed_events = Vec::new();

            for event in history.iter() {
                if found_last {
                    if connection.event_types.contains(&event.event_type)
                        || connection.event_types.contains(&"*".to_string())
                    {
                        missed_events.push(event.clone());
                    }
                }

                if event.id == *last_event_id {
                    found_last = true;
                }
            }

            missed_events
        } else {
            vec![]
        }
    }

    /// Get service statistics
    pub async fn get_stats(&self) -> LongPollingStats {
        let stats = self.stats.read().unwrap();
        stats.clone()
    }

    /// Get active connection count
    pub fn get_active_connections(&self) -> usize {
        self.connections.read().unwrap().len()
    }
}

impl Drop for LongPollingService {
    fn drop(&mut self) {
        // Use try_lock instead of blocking_lock to avoid blocking in async context
        if let Ok(mut handle_guard) = self.cleanup_handle.try_lock() {
            if let Some(handle) = handle_guard.take() {
                handle.abort();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_long_polling_service_creation() {
        let config = LongPollingConfig::default();
        let service = LongPollingService::new(config);

        assert_eq!(service.get_active_connections(), 0);
    }

    #[tokio::test]
    async fn test_event_publishing() {
        let config = LongPollingConfig::default();
        let service = LongPollingService::new(config);

        let event = PollEvent::InferenceComplete {
            job_id: "test-job".to_string(),
            result: "Test result".to_string(),
            processing_time_ms: 100.0,
        };

        service.publish_event(event).await.unwrap();

        let stats = service.get_stats().await;
        assert_eq!(stats.total_events_published, 1);
    }

    #[tokio::test]
    async fn test_polling_timeout() {
        let mut config = LongPollingConfig::default();
        config.timeout_seconds = 1; // Short timeout for testing

        let service = LongPollingService::new(config);

        let request = LongPollRequest {
            event_types: vec!["inference_complete".to_string()],
            client_id: Some("test-client".to_string()),
            timeout_seconds: Some(1),
            last_event_id: None,
        };

        let response = service.poll(request).await.unwrap();

        assert!(response.timeout);
        assert!(response.events.is_empty());
    }
}
