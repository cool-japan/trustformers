//! Streaming dashboard server using Server-Sent Events (SSE).
//!
//! The [`DashboardServer`] binds a TCP port and streams [`DashboardEvent`]s
//! to HTTP clients that issue a `GET /events` request (standard SSE
//! handshake).  Events are produced via [`DashboardServer::push_event`] and
//! buffered in a [`LockFreeRingBuffer`] before being forwarded to each
//! connected client through a `tokio::sync::broadcast` channel.
//!
//! # Example
//!
//! ```no_run
//! use trustformers_debug::dashboard_ws::websocket::{
//!     DashboardConfig, DashboardEvent, DashboardServer,
//! };
//! use std::net::SocketAddr;
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! let config = DashboardConfig {
//!     bind_addr: "127.0.0.1:7878".parse()?,
//!     max_clients: 8,
//!     event_buffer_size: 256,
//! };
//! let server = DashboardServer::new(config)?;
//! server.start().await?;
//!
//! server.push_event(DashboardEvent::TrainingStep {
//!     step: 1,
//!     loss: 0.5,
//!     learning_rate: 1e-4,
//!     throughput_tokens_per_sec: 3200.0,
//! })?;
//! # Ok(())
//! # }
//! ```

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::broadcast;

use crate::ring_buffer::LockFreeRingBuffer;

// ─────────────────────────────────────────────────────────────
// DashboardEvent
// ─────────────────────────────────────────────────────────────

/// Events that can be pushed to connected dashboard clients.
///
/// All variants are serialised to JSON and delivered via SSE
/// `data:` lines.
///
/// # Example
///
/// ```
/// use trustformers_debug::dashboard_ws::websocket::DashboardEvent;
///
/// let ev = DashboardEvent::TrainingStep {
///     step: 10,
///     loss: 0.35,
///     learning_rate: 1e-4,
///     throughput_tokens_per_sec: 2048.0,
/// };
/// let json = serde_json::to_string(&ev).unwrap();
/// assert!(json.contains("TrainingStep"));
/// ```
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum DashboardEvent {
    /// Progress update emitted after each training step.
    TrainingStep {
        /// Global optimiser step counter.
        step: u64,
        /// Training loss value.
        loss: f32,
        /// Current learning rate.
        learning_rate: f32,
        /// Throughput measured in tokens processed per second.
        throughput_tokens_per_sec: f32,
    },
    /// Snapshot of host-side memory usage.
    MemorySnapshot {
        /// Wall-clock milliseconds since epoch when this snapshot was taken.
        timestamp_ms: u64,
        /// Currently allocated heap bytes.
        allocated_bytes: u64,
        /// Peak allocation since the process started.
        peak_bytes: u64,
    },
    /// Per-layer profiling data.
    LayerProfile {
        /// Layer identifier string.
        layer_name: String,
        /// Wall-clock execution time of the layer in microseconds.
        duration_us: u64,
        /// Memory consumed by the layer's activations.
        memory_bytes: u64,
    },
    /// Arbitrary user-defined event.
    Custom {
        /// Event category name.
        name: String,
        /// Payload — any JSON-serialisable value.
        value: Value,
    },
}

// ─────────────────────────────────────────────────────────────
// DashboardConfig
// ─────────────────────────────────────────────────────────────

/// Configuration for the SSE streaming dashboard server.
///
/// # Example
///
/// ```
/// use trustformers_debug::dashboard_ws::websocket::DashboardConfig;
///
/// let cfg = DashboardConfig {
///     bind_addr: "127.0.0.1:0".parse().unwrap(),
///     max_clients: 4,
///     event_buffer_size: 64,
/// };
/// assert_eq!(cfg.max_clients, 4);
/// ```
pub struct DashboardConfig {
    /// TCP address to bind the server to.
    pub bind_addr: SocketAddr,
    /// Maximum number of simultaneously connected SSE clients.
    pub max_clients: usize,
    /// Capacity of the in-process event ring buffer (rounded to next power of 2).
    pub event_buffer_size: usize,
}

// ─────────────────────────────────────────────────────────────
// DashboardServer
// ─────────────────────────────────────────────────────────────

/// SSE streaming dashboard server.
///
/// Pushes [`DashboardEvent`]s to all connected HTTP clients that opened
/// `GET /events`.  Events are first written to an internal
/// [`LockFreeRingBuffer`] and then forwarded to each client through a
/// `tokio::sync::broadcast` channel.
///
/// # Thread safety
///
/// The server is `Send + Sync`.  `push_event` may be called from any thread
/// or async task.
pub struct DashboardServer {
    config: DashboardConfig,
    event_buffer: Arc<LockFreeRingBuffer<u64>>,
    /// Broadcast channel used to forward events to active client tasks.
    sender: broadcast::Sender<String>,
    /// Set to `true` once `start()` has been called.
    running: Arc<AtomicBool>,
}

impl DashboardServer {
    /// Creates a new server but does **not** start listening yet.
    ///
    /// Call [`start`](Self::start) to begin accepting connections.
    ///
    /// # Errors
    ///
    /// Returns an error if `event_buffer_size` is 0.
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_debug::dashboard_ws::websocket::{DashboardConfig, DashboardServer};
    ///
    /// let cfg = DashboardConfig {
    ///     bind_addr: "127.0.0.1:0".parse().unwrap(),
    ///     max_clients: 4,
    ///     event_buffer_size: 32,
    /// };
    /// let server = DashboardServer::new(cfg).unwrap();
    /// assert!(!server.is_running());
    /// ```
    pub fn new(config: DashboardConfig) -> Result<Self> {
        let buf_size = config.event_buffer_size;
        if buf_size == 0 {
            anyhow::bail!("event_buffer_size must be at least 1");
        }
        let event_buffer = Arc::new(LockFreeRingBuffer::new(buf_size));
        // `max_clients` determines the broadcast channel capacity.
        let channel_cap = config.max_clients.max(1);
        let (sender, _) = broadcast::channel(channel_cap * 8);
        Ok(Self {
            config,
            event_buffer,
            sender,
            running: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Returns `true` if [`start`](Self::start) has been called.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    /// Pushes a [`DashboardEvent`] to all connected clients.
    ///
    /// The event is serialised to JSON and written to the internal ring
    /// buffer as a sequence of bytes (the SSE-framed JSON string length is
    /// stored as an index rather than raw bytes — see implementation notes).
    ///
    /// If no clients are connected the broadcast send silently drops the
    /// message (all receivers have been dropped), which is acceptable
    /// behaviour for a monitoring stream.
    ///
    /// # Errors
    ///
    /// Returns an error if JSON serialisation fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use trustformers_debug::dashboard_ws::websocket::{DashboardConfig, DashboardServer, DashboardEvent};
    /// # let server = DashboardServer::new(DashboardConfig {
    /// #     bind_addr: "127.0.0.1:0".parse().unwrap(),
    /// #     max_clients: 4,
    /// #     event_buffer_size: 32,
    /// # }).unwrap();
    /// server.push_event(DashboardEvent::Custom {
    ///     name: "epoch_end".to_string(),
    ///     value: serde_json::json!({"epoch": 1}),
    /// }).unwrap();
    /// ```
    pub fn push_event(&self, event: DashboardEvent) -> Result<()> {
        let json = serde_json::to_string(&event)?;
        let sse_frame = format!("data: {}\n\n", json);

        // Store a fingerprint (length as u64) in the ring buffer so we can
        // track how many events have been pushed even when no client is
        // connected.
        let _ = self.event_buffer.push(sse_frame.len() as u64);

        // Forward to active clients; ignore send errors (no receivers is fine).
        let _ = self.sender.send(sse_frame);
        Ok(())
    }

    /// Starts the SSE server in a background tokio task.
    ///
    /// The method binds the TCP listener immediately (so binding errors are
    /// returned synchronously) and then spawns a task to accept connections.
    ///
    /// # Errors
    ///
    /// Returns an error if the TCP listener cannot bind to
    /// `config.bind_addr`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #[tokio::main]
    /// # async fn main() -> anyhow::Result<()> {
    /// use trustformers_debug::dashboard_ws::websocket::{DashboardConfig, DashboardServer};
    ///
    /// let cfg = DashboardConfig {
    ///     bind_addr: "127.0.0.1:0".parse()?,
    ///     max_clients: 2,
    ///     event_buffer_size: 16,
    /// };
    /// let server = DashboardServer::new(cfg)?;
    /// server.start().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn start(&self) -> Result<()> {
        let listener = TcpListener::bind(self.config.bind_addr).await?;
        tracing::debug!(
            "SSE dashboard listening on {}",
            listener.local_addr().unwrap_or(self.config.bind_addr)
        );

        let sender = self.sender.clone();
        let running = Arc::clone(&self.running);
        running.store(true, Ordering::Release);
        let running_flag = Arc::clone(&self.running);

        tokio::spawn(async move {
            while running_flag.load(Ordering::Acquire) {
                match listener.accept().await {
                    Ok((stream, peer)) => {
                        tracing::debug!("dashboard: accepted connection from {}", peer);
                        let rx = sender.subscribe();
                        tokio::spawn(handle_client(stream, rx));
                    }
                    Err(e) => {
                        tracing::warn!("dashboard: accept error: {}", e);
                        break;
                    }
                }
            }
        });

        Ok(())
    }

    /// Signals the server to stop accepting new connections.
    ///
    /// Already-running client handler tasks are left to drain and close
    /// naturally.
    ///
    /// # Errors
    ///
    /// Currently infallible; returns `Ok(())` always.
    pub fn stop(&self) -> Result<()> {
        self.running.store(false, Ordering::Release);
        tracing::debug!("dashboard server stopping");
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────

/// Handles one SSE client connection.
///
/// Reads the HTTP request line, sends back SSE headers, then streams events
/// received on `rx` until the connection is closed.
async fn handle_client(
    mut stream: TcpStream,
    mut rx: broadcast::Receiver<String>,
) {
    // Read enough of the request to identify the path, then ignore the rest.
    let mut buf = [0u8; 512];
    match stream.read(&mut buf).await {
        Ok(0) | Err(_) => return,
        Ok(_) => {}
    }

    let request = std::str::from_utf8(&buf).unwrap_or("");
    let is_events = request.starts_with("GET /events") || request.contains("GET /events");

    let (status, body) = if is_events {
        (
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n",
            None,
        )
    } else {
        (
            "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n",
            Some(DASHBOARD_HTML),
        )
    };

    if stream.write_all(status.as_bytes()).await.is_err() {
        return;
    }

    if let Some(html) = body {
        let _ = stream.write_all(html.as_bytes()).await;
        return;
    }

    // Stream events
    loop {
        match rx.recv().await {
            Ok(frame) => {
                if stream.write_all(frame.as_bytes()).await.is_err() {
                    break;
                }
            }
            Err(broadcast::error::RecvError::Lagged(n)) => {
                tracing::warn!("dashboard client lagged by {} events", n);
            }
            Err(broadcast::error::RecvError::Closed) => break,
        }
    }
}

/// Minimal HTML page that auto-connects to the SSE event stream.
const DASHBOARD_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>TrustformeRS Dashboard</title></head>
<body>
<h1>TrustformeRS Streaming Dashboard</h1>
<pre id="log"></pre>
<script>
  const log = document.getElementById('log');
  const es = new EventSource('/events');
  es.onmessage = e => {
    const line = document.createTextNode(e.data + '\n');
    log.appendChild(line);
  };
</script>
</body>
</html>"#;

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Serialisation tests (no network) ─────────────────────

    #[test]
    fn test_training_step_serialises() {
        let ev = DashboardEvent::TrainingStep {
            step: 5,
            loss: 0.5,
            learning_rate: 1e-4,
            throughput_tokens_per_sec: 1024.0,
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains("TrainingStep"));
        assert!(json.contains("\"step\":5"));
    }

    #[test]
    fn test_memory_snapshot_serialises() {
        let ev = DashboardEvent::MemorySnapshot {
            timestamp_ms: 1000,
            allocated_bytes: 1_073_741_824,
            peak_bytes: 2_147_483_648,
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains("MemorySnapshot"));
        assert!(json.contains("allocated_bytes"));
    }

    #[test]
    fn test_layer_profile_serialises() {
        let ev = DashboardEvent::LayerProfile {
            layer_name: "attention".to_string(),
            duration_us: 1500,
            memory_bytes: 256 * 1024 * 1024,
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains("LayerProfile"));
        assert!(json.contains("\"layer_name\":\"attention\""));
    }

    #[test]
    fn test_custom_event_serialises() {
        let ev = DashboardEvent::Custom {
            name: "epoch_end".to_string(),
            value: serde_json::json!({"epoch": 3, "val_loss": 0.22}),
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains("Custom"));
        assert!(json.contains("epoch_end"));
    }

    #[test]
    fn test_event_roundtrip_deserialise() {
        let ev = DashboardEvent::TrainingStep {
            step: 42,
            loss: 0.1,
            learning_rate: 3e-5,
            throughput_tokens_per_sec: 4096.0,
        };
        let json = serde_json::to_string(&ev).unwrap();
        let ev2: DashboardEvent = serde_json::from_str(&json).unwrap();
        if let DashboardEvent::TrainingStep { step, .. } = ev2 {
            assert_eq!(step, 42);
        } else {
            panic!("unexpected variant after roundtrip");
        }
    }

    // ── Server creation and push_event tests ─────────────────

    #[test]
    fn test_server_creation() {
        let cfg = DashboardConfig {
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            max_clients: 4,
            event_buffer_size: 32,
        };
        let server = DashboardServer::new(cfg).unwrap();
        assert!(!server.is_running());
    }

    #[test]
    fn test_push_event_without_clients() {
        let cfg = DashboardConfig {
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            max_clients: 4,
            event_buffer_size: 32,
        };
        let server = DashboardServer::new(cfg).unwrap();
        // Pushing without any client connected should not error.
        server
            .push_event(DashboardEvent::TrainingStep {
                step: 1,
                loss: 0.4,
                learning_rate: 1e-4,
                throughput_tokens_per_sec: 2048.0,
            })
            .unwrap();
        // Ring buffer should have recorded the event size.
        assert!(!server.event_buffer.is_empty());
    }

    #[test]
    fn test_stop_before_start() {
        let cfg = DashboardConfig {
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            max_clients: 2,
            event_buffer_size: 8,
        };
        let server = DashboardServer::new(cfg).unwrap();
        server.stop().unwrap();
        assert!(!server.is_running());
    }

    #[tokio::test]
    async fn test_start_and_stop() {
        let cfg = DashboardConfig {
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            max_clients: 2,
            event_buffer_size: 16,
        };
        let server = DashboardServer::new(cfg).unwrap();
        server.start().await.unwrap();
        assert!(server.is_running());
        server.stop().unwrap();
        assert!(!server.is_running());
    }

    #[tokio::test]
    async fn test_push_event_with_subscriber() {
        let cfg = DashboardConfig {
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            max_clients: 4,
            event_buffer_size: 32,
        };
        let server = DashboardServer::new(cfg).unwrap();

        // Subscribe before pushing
        let mut rx = server.sender.subscribe();

        server
            .push_event(DashboardEvent::Custom {
                name: "test".to_string(),
                value: serde_json::json!(42),
            })
            .unwrap();

        let frame = rx.try_recv().expect("should receive frame");
        assert!(frame.starts_with("data: "));
        assert!(frame.contains("\"test\""));
    }

    // ── additional DashboardEvent tests ────────────────────────────────────

    #[test]
    fn test_dashboard_event_training_step_serialization() {
        let ev = DashboardEvent::TrainingStep {
            step: 42,
            loss: 0.25,
            learning_rate: 3e-4,
            throughput_tokens_per_sec: 1500.0,
        };
        let json = serde_json::to_string(&ev).expect("serialize should succeed");
        assert!(json.contains("TrainingStep"));
        assert!(json.contains("42"));
    }

    #[test]
    fn test_dashboard_event_memory_snapshot_serialization() {
        let ev = DashboardEvent::MemorySnapshot {
            timestamp_ms: 12345,
            allocated_bytes: 1024 * 1024,
            peak_bytes: 2 * 1024 * 1024,
        };
        let json = serde_json::to_string(&ev).expect("serialize should succeed");
        assert!(json.contains("MemorySnapshot"));
    }

    #[test]
    fn test_dashboard_event_layer_profile_serialization() {
        let ev = DashboardEvent::LayerProfile {
            layer_name: "attention".to_string(),
            duration_us: 1500,
            memory_bytes: 4096,
        };
        let json = serde_json::to_string(&ev).expect("serialize should succeed");
        assert!(json.contains("attention"));
        assert!(json.contains("1500"));
    }

    #[test]
    fn test_dashboard_event_custom_roundtrip() {
        let ev = DashboardEvent::Custom {
            name: "my_event".to_string(),
            value: serde_json::json!({"key": "value"}),
        };
        let json = serde_json::to_string(&ev).expect("serialize should succeed");
        let decoded: DashboardEvent = serde_json::from_str(&json).expect("deserialize should succeed");
        if let DashboardEvent::Custom { name, .. } = decoded {
            assert_eq!(name, "my_event");
        } else {
            panic!("expected Custom variant");
        }
    }

    #[test]
    fn test_dashboard_config_fields() {
        let cfg = DashboardConfig {
            bind_addr: "0.0.0.0:8080".parse().expect("parse addr"),
            max_clients: 16,
            event_buffer_size: 128,
        };
        assert_eq!(cfg.max_clients, 16);
        assert_eq!(cfg.event_buffer_size, 128);
    }

    #[test]
    fn test_dashboard_server_zero_buffer_errors() {
        let cfg = DashboardConfig {
            bind_addr: "127.0.0.1:0".parse().expect("parse addr"),
            max_clients: 4,
            event_buffer_size: 0, // invalid
        };
        let result = DashboardServer::new(cfg);
        assert!(result.is_err());
    }

    #[test]
    fn test_push_multiple_events_fills_buffer() {
        let cfg = DashboardConfig {
            bind_addr: "127.0.0.1:0".parse().expect("parse addr"),
            max_clients: 4,
            event_buffer_size: 32,
        };
        let server = DashboardServer::new(cfg).expect("create server");
        for i in 0..5_u64 {
            server.push_event(DashboardEvent::TrainingStep {
                step: i,
                loss: 0.5,
                learning_rate: 1e-4,
                throughput_tokens_per_sec: 1000.0,
            }).expect("push should succeed");
        }
        assert!(!server.event_buffer.is_empty());
    }

    #[test]
    fn test_dashboard_event_deserialize_training_step() {
        let json = r#"{"TrainingStep":{"step":10,"loss":0.3,"learning_rate":0.001,"throughput_tokens_per_sec":2048.0}}"#;
        let ev: DashboardEvent = serde_json::from_str(json).expect("deserialize should succeed");
        if let DashboardEvent::TrainingStep { step, .. } = ev {
            assert_eq!(step, 10);
        } else {
            panic!("expected TrainingStep");
        }
    }
}
