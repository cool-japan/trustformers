//! Metric history tracking and dashboard message types.
//!
//! This module provides:
//! - [`DashboardMessage`] — a typed enum of all real-time training events, with
//!   manual JSON serialisation (no `serde` dependency).
//! - [`MetricHistory`] — a fixed-capacity rolling window with EMA, trend, and
//!   linear-regression utilities.
//! - [`DashboardServerExt`] — an extended server that wraps the core SSE server
//!   and keeps in-memory metric histories plus CSV/JSON summary generation.

use std::fmt::Write as FmtWrite;

// ─────────────────────────────────────────────────────────────────────────────
// DashboardError
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur in the dashboard layer.
#[derive(Debug, Clone, PartialEq)]
pub enum DashboardError {
    /// The network connection to a client could not be established.
    ConnectionFailed,
    /// Failed to serialise a [`DashboardMessage`] to JSON.
    SerializationError,
    /// The internal message buffer is full; the payload size is given.
    BufferFull(usize),
    /// A configuration value is invalid.
    ConfigError(String),
}

impl std::fmt::Display for DashboardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConnectionFailed => write!(f, "dashboard connection failed"),
            Self::SerializationError => write!(f, "failed to serialise dashboard message"),
            Self::BufferFull(n) => write!(f, "dashboard message buffer full (size={n})"),
            Self::ConfigError(msg) => write!(f, "dashboard configuration error: {msg}"),
        }
    }
}

impl std::error::Error for DashboardError {}

// ─────────────────────────────────────────────────────────────────────────────
// DashboardMessage
// ─────────────────────────────────────────────────────────────────────────────

/// Typed messages emitted by a training run to connected dashboard clients.
///
/// All variants are serialised to compact JSON via the hand-written
/// [`DashboardMessage::to_json`] method (no `serde` macro dependency).
///
/// # Example
///
/// ```
/// use trustformers_debug::dashboard_ws::metrics::DashboardMessage;
///
/// let msg = DashboardMessage::TrainingMetrics {
///     step: 1,
///     loss: 2.3,
///     learning_rate: 1e-4,
///     throughput_samples_per_sec: 128.0,
/// };
/// let json = msg.to_json();
/// assert!(json.contains("\"type\":\"training_metrics\""));
/// assert!(json.contains("\"step\":1"));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum DashboardMessage {
    /// Per-step training metrics.
    TrainingMetrics {
        step: u64,
        loss: f32,
        learning_rate: f32,
        throughput_samples_per_sec: f32,
    },
    /// Per-step validation metrics.
    ValidationMetrics {
        step: u64,
        val_loss: f32,
        val_accuracy: f32,
    },
    /// Global and per-layer gradient norms.
    GradientNorm {
        step: u64,
        global_norm: f32,
        per_layer: Vec<(String, f32)>,
    },
    /// Host and device memory snapshot.
    MemoryUsage {
        step: u64,
        gpu_mb: f32,
        cpu_mb: f32,
        peak_gpu_mb: f32,
    },
    /// A checkpoint was written to disk.
    CheckpointSaved {
        step: u64,
        path: String,
    },
    /// Training finished successfully.
    TrainingComplete {
        total_steps: u64,
        final_loss: f32,
        duration_secs: f64,
    },
    /// Periodic keep-alive ping.
    Heartbeat {
        timestamp_ms: u64,
    },
    /// Server-side error notification.
    Error {
        code: u32,
        message: String,
    },
}

impl DashboardMessage {
    /// Returns the `type` field value used in the JSON envelope.
    pub fn message_type(&self) -> &str {
        match self {
            Self::TrainingMetrics { .. } => "training_metrics",
            Self::ValidationMetrics { .. } => "validation_metrics",
            Self::GradientNorm { .. } => "gradient_norm",
            Self::MemoryUsage { .. } => "memory_usage",
            Self::CheckpointSaved { .. } => "checkpoint_saved",
            Self::TrainingComplete { .. } => "training_complete",
            Self::Heartbeat { .. } => "heartbeat",
            Self::Error { .. } => "error",
        }
    }

    /// Serialise the message to a compact JSON string.
    ///
    /// Uses manual formatting to avoid pulling in `serde` inside this module.
    /// The output always contains a `"type"` key plus variant-specific fields.
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_debug::dashboard_ws::metrics::DashboardMessage;
    ///
    /// let msg = DashboardMessage::Heartbeat { timestamp_ms: 12345 };
    /// let json = msg.to_json();
    /// assert!(json.contains("\"type\":\"heartbeat\""));
    /// assert!(json.contains("\"timestamp_ms\":12345"));
    /// ```
    pub fn to_json(&self) -> String {
        match self {
            Self::TrainingMetrics { step, loss, learning_rate, throughput_samples_per_sec } => {
                format!(
                    r#"{{"type":"training_metrics","step":{step},"loss":{loss},"learning_rate":{learning_rate},"throughput_samples_per_sec":{throughput_samples_per_sec}}}"#,
                )
            }
            Self::ValidationMetrics { step, val_loss, val_accuracy } => {
                format!(
                    r#"{{"type":"validation_metrics","step":{step},"val_loss":{val_loss},"val_accuracy":{val_accuracy}}}"#,
                )
            }
            Self::GradientNorm { step, global_norm, per_layer } => {
                let layers_json = Self::per_layer_to_json(per_layer);
                format!(
                    r#"{{"type":"gradient_norm","step":{step},"global_norm":{global_norm},"per_layer":{layers_json}}}"#,
                )
            }
            Self::MemoryUsage { step, gpu_mb, cpu_mb, peak_gpu_mb } => {
                format!(
                    r#"{{"type":"memory_usage","step":{step},"gpu_mb":{gpu_mb},"cpu_mb":{cpu_mb},"peak_gpu_mb":{peak_gpu_mb}}}"#,
                )
            }
            Self::CheckpointSaved { step, path } => {
                let escaped = escape_json_string(path);
                format!(r#"{{"type":"checkpoint_saved","step":{step},"path":"{escaped}"}}"#)
            }
            Self::TrainingComplete { total_steps, final_loss, duration_secs } => {
                format!(
                    r#"{{"type":"training_complete","total_steps":{total_steps},"final_loss":{final_loss},"duration_secs":{duration_secs}}}"#,
                )
            }
            Self::Heartbeat { timestamp_ms } => {
                format!(r#"{{"type":"heartbeat","timestamp_ms":{timestamp_ms}}}"#)
            }
            Self::Error { code, message } => {
                let escaped = escape_json_string(message);
                format!(r#"{{"type":"error","code":{code},"message":"{escaped}"}}"#)
            }
        }
    }

    // ── private helpers ──────────────────────────────────────────────────────

    fn per_layer_to_json(per_layer: &[(String, f32)]) -> String {
        let mut out = String::from('[');
        for (i, (name, norm)) in per_layer.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let escaped = escape_json_string(name);
            let _ = write!(out, r#"["{escaped}",{norm}]"#);
        }
        out.push(']');
        out
    }
}

/// Escapes a string for safe embedding inside a JSON string literal.
fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// DashboardConfig (extended)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the extended training-metrics dashboard.
///
/// Unlike the SSE server configuration ([`super::websocket::DashboardConfig`]),
/// this struct captures parameters for heartbeat scheduling and metric buffering.
#[derive(Debug, Clone)]
pub struct DashboardConfig {
    pub host: String,
    pub port: u16,
    pub max_connections: usize,
    pub heartbeat_interval_ms: u64,
    pub message_buffer_size: usize,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 7878,
            max_connections: 16,
            heartbeat_interval_ms: 5_000,
            message_buffer_size: 512,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MetricHistory
// ─────────────────────────────────────────────────────────────────────────────

/// A fixed-capacity rolling window of `(step, value)` pairs.
///
/// Older entries are silently dropped when `max_history` is exceeded.
/// Provides EMA smoothing and linear-regression trend estimation.
///
/// # Example
///
/// ```
/// use trustformers_debug::dashboard_ws::metrics::MetricHistory;
///
/// let mut h = MetricHistory::new(100);
/// h.push(0, 1.0);
/// h.push(1, 0.8);
/// h.push(2, 0.6);
/// assert_eq!(h.latest(), Some((2, 0.6)));
/// let trend = h.trend(3);
/// assert!(trend.is_some());
/// assert!(trend.unwrap() < 0.0, "loss should be trending down");
/// ```
#[derive(Debug, Clone)]
pub struct MetricHistory {
    pub steps: Vec<u64>,
    pub values: Vec<f32>,
    pub max_history: usize,
}

impl MetricHistory {
    /// Creates a new empty history with the given maximum capacity.
    pub fn new(max_history: usize) -> Self {
        Self {
            steps: Vec::with_capacity(max_history.min(4096)),
            values: Vec::with_capacity(max_history.min(4096)),
            max_history,
        }
    }

    /// Appends a new `(step, value)` pair, evicting the oldest when full.
    pub fn push(&mut self, step: u64, value: f32) {
        if self.steps.len() >= self.max_history {
            self.steps.remove(0);
            self.values.remove(0);
        }
        self.steps.push(step);
        self.values.push(value);
    }

    /// Returns the most-recently pushed `(step, value)` pair, or `None` if empty.
    pub fn latest(&self) -> Option<(u64, f32)> {
        self.steps
            .last()
            .copied()
            .zip(self.values.last().copied())
    }

    /// Estimates the linear-regression slope over the last `window` data points.
    ///
    /// Returns `None` when fewer than two points are available in the window.
    /// A negative slope means the metric is decreasing (good for loss).
    ///
    /// Uses the standard ordinary-least-squares formula:
    /// `slope = (N·Σ(x·y) − Σx·Σy) / (N·Σx² − (Σx)²)`
    pub fn trend(&self, window: usize) -> Option<f32> {
        let n = self.values.len();
        if n < 2 {
            return None;
        }
        let start = n.saturating_sub(window);
        let xs = &self.steps[start..];
        let ys = &self.values[start..];
        let k = xs.len();
        if k < 2 {
            return None;
        }

        let sum_x: f64 = xs.iter().map(|&x| x as f64).sum();
        let sum_y: f64 = ys.iter().map(|&y| y as f64).sum();
        let sum_xx: f64 = xs.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let sum_xy: f64 = xs.iter().zip(ys.iter()).map(|(&x, &y)| (x as f64) * (y as f64)).sum();
        let kf = k as f64;

        let denom = kf * sum_xx - sum_x * sum_x;
        if denom.abs() < f64::EPSILON {
            return None;
        }

        let slope = (kf * sum_xy - sum_x * sum_y) / denom;
        Some(slope as f32)
    }

    /// Computes an exponential moving average (EMA) over all stored values.
    ///
    /// `alpha` controls the smoothing factor (0 < alpha ≤ 1).  Higher `alpha`
    /// gives more weight to recent observations.  Returns an empty vec if the
    /// history is empty.
    ///
    /// # Panics
    ///
    /// Does not panic, but clamps `alpha` to `[1e-6, 1.0]` internally.
    pub fn smooth(&self, alpha: f32) -> Vec<f32> {
        if self.values.is_empty() {
            return Vec::new();
        }
        let alpha = alpha.max(1e-6_f32).min(1.0_f32);
        let mut out = Vec::with_capacity(self.values.len());
        let mut ema = self.values[0];
        out.push(ema);
        for &v in &self.values[1..] {
            ema = alpha * v + (1.0 - alpha) * ema;
            out.push(ema);
        }
        out
    }

    /// Returns the number of data points currently stored.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns `true` when no data has been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DashboardServer (extended)
// ─────────────────────────────────────────────────────────────────────────────

/// In-process training dashboard that records metric histories and can generate
/// CSV / JSON summaries suitable for logging or downstream tools.
///
/// This is an *in-process* implementation that does not bind network sockets.
/// For SSE streaming, see [`super::websocket::DashboardServer`].
///
/// # Example
///
/// ```
/// use trustformers_debug::dashboard_ws::metrics::{DashboardConfig, DashboardServerExt};
///
/// let mut server = DashboardServerExt::new(DashboardConfig::default());
/// server.record_metrics(10, 1.5, 1e-3, 0.9);
/// server.record_metrics(20, 1.2, 9e-4, 0.85);
/// let csv = server.format_metric_csv();
/// assert!(csv.contains("step"));
/// ```
pub struct DashboardServerExt {
    pub config: DashboardConfig,
    pub loss_history: MetricHistory,
    pub lr_history: MetricHistory,
    pub grad_norm_history: MetricHistory,
    pub connected_clients: usize,
    pub messages_sent: u64,
    /// In-process broadcast buffer (bounded by `config.message_buffer_size`).
    message_log: Vec<DashboardMessage>,
}

impl DashboardServerExt {
    /// Creates a new in-process dashboard server with the given configuration.
    pub fn new(config: DashboardConfig) -> Self {
        let cap = config.message_buffer_size;
        Self {
            loss_history: MetricHistory::new(cap),
            lr_history: MetricHistory::new(cap),
            grad_norm_history: MetricHistory::new(cap),
            connected_clients: 0,
            messages_sent: 0,
            message_log: Vec::with_capacity(cap.min(4096)),
            config,
        }
    }

    /// Records one training step into the internal metric histories.
    pub fn record_metrics(&mut self, step: u64, loss: f32, lr: f32, grad_norm: f32) {
        self.loss_history.push(step, loss);
        self.lr_history.push(step, lr);
        self.grad_norm_history.push(step, grad_norm);
    }

    /// Appends a [`DashboardMessage`] to the internal broadcast buffer.
    ///
    /// Returns [`DashboardError::BufferFull`] when the buffer is at capacity.
    pub fn broadcast_message(&mut self, msg: &DashboardMessage) -> Result<(), DashboardError> {
        if self.message_log.len() >= self.config.message_buffer_size {
            return Err(DashboardError::BufferFull(self.message_log.len()));
        }
        self.message_log.push(msg.clone());
        self.messages_sent += 1;
        Ok(())
    }

    /// Returns all messages currently in the broadcast buffer (in order).
    pub fn buffered_messages(&self) -> &[DashboardMessage] {
        &self.message_log
    }

    /// Clears the broadcast buffer.
    pub fn clear_buffer(&mut self) {
        self.message_log.clear();
    }

    /// Returns a JSON summary of all metric histories.
    ///
    /// Includes the latest value, a 10-point trend (slope), and EMA
    /// (α = 0.1) for each of `loss`, `lr`, and `grad_norm`.
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_debug::dashboard_ws::metrics::{DashboardConfig, DashboardServerExt};
    ///
    /// let mut s = DashboardServerExt::new(DashboardConfig::default());
    /// s.record_metrics(1, 2.0, 1e-3, 0.5);
    /// let json = s.generate_summary_json();
    /// assert!(json.contains("\"loss\""));
    /// ```
    pub fn generate_summary_json(&self) -> String {
        let loss_latest = self
            .loss_history
            .latest()
            .map(|(_, v)| format!("{v}"))
            .unwrap_or_else(|| "null".to_string());
        let lr_latest = self
            .lr_history
            .latest()
            .map(|(_, v)| format!("{v}"))
            .unwrap_or_else(|| "null".to_string());
        let grad_latest = self
            .grad_norm_history
            .latest()
            .map(|(_, v)| format!("{v}"))
            .unwrap_or_else(|| "null".to_string());

        let loss_trend = self
            .loss_history
            .trend(10)
            .map(|v| format!("{v}"))
            .unwrap_or_else(|| "null".to_string());
        let lr_trend = self
            .lr_history
            .trend(10)
            .map(|v| format!("{v}"))
            .unwrap_or_else(|| "null".to_string());
        let grad_trend = self
            .grad_norm_history
            .trend(10)
            .map(|v| format!("{v}"))
            .unwrap_or_else(|| "null".to_string());

        let loss_ema = format_f32_slice(&self.loss_history.smooth(0.1));
        let lr_ema = format_f32_slice(&self.lr_history.smooth(0.1));
        let grad_ema = format_f32_slice(&self.grad_norm_history.smooth(0.1));

        format!(
            r#"{{"connected_clients":{clients},"messages_sent":{sent},"loss":{{"latest":{loss_latest},"trend_slope":{loss_trend},"ema_alpha0.1":{loss_ema}}},"lr":{{"latest":{lr_latest},"trend_slope":{lr_trend},"ema_alpha0.1":{lr_ema}}},"grad_norm":{{"latest":{grad_latest},"trend_slope":{grad_trend},"ema_alpha0.1":{grad_ema}}}}}"#,
            clients = self.connected_clients,
            sent = self.messages_sent,
        )
    }

    /// Exports all recorded metrics as a CSV string.
    ///
    /// Columns: `step,loss,lr,grad_norm`
    /// Rows are aligned by index — if one history is shorter it writes empty cells.
    pub fn format_metric_csv(&self) -> String {
        let mut out = String::from("step,loss,lr,grad_norm\n");
        let len = self
            .loss_history
            .len()
            .max(self.lr_history.len())
            .max(self.grad_norm_history.len());

        for i in 0..len {
            let step = self
                .loss_history
                .steps
                .get(i)
                .or_else(|| self.lr_history.steps.get(i))
                .or_else(|| self.grad_norm_history.steps.get(i))
                .copied()
                .unwrap_or(i as u64);

            let loss = self
                .loss_history
                .values
                .get(i)
                .map(|v| format!("{v}"))
                .unwrap_or_default();
            let lr = self
                .lr_history
                .values
                .get(i)
                .map(|v| format!("{v}"))
                .unwrap_or_default();
            let grad = self
                .grad_norm_history
                .values
                .get(i)
                .map(|v| format!("{v}"))
                .unwrap_or_default();

            let _ = writeln!(out, "{step},{loss},{lr},{grad}");
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

fn format_f32_slice(values: &[f32]) -> String {
    let mut out = String::from('[');
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(out, "{v}");
    }
    out.push(']');
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── DashboardMessage::to_json ────────────────────────────────────────────

    #[test]
    fn test_training_metrics_json() {
        let msg = DashboardMessage::TrainingMetrics {
            step: 42,
            loss: 1.5,
            learning_rate: 1e-4,
            throughput_samples_per_sec: 256.0,
        };
        let json = msg.to_json();
        assert!(json.contains("\"type\":\"training_metrics\""));
        assert!(json.contains("\"step\":42"));
        assert!(json.contains("\"loss\":1.5"));
        assert_eq!(msg.message_type(), "training_metrics");
    }

    #[test]
    fn test_validation_metrics_json() {
        let msg = DashboardMessage::ValidationMetrics {
            step: 10,
            val_loss: 0.9,
            val_accuracy: 0.85,
        };
        let json = msg.to_json();
        assert!(json.contains("\"type\":\"validation_metrics\""));
        assert!(json.contains("\"step\":10"));
        assert!(json.contains("\"val_loss\":0.9"));
        assert!(json.contains("\"val_accuracy\":0.85"));
    }

    #[test]
    fn test_gradient_norm_json_with_per_layer() {
        let msg = DashboardMessage::GradientNorm {
            step: 5,
            global_norm: 1.2,
            per_layer: vec![
                ("layer_0.weight".to_string(), 0.3),
                ("layer_1.weight".to_string(), 0.9),
            ],
        };
        let json = msg.to_json();
        assert!(json.contains("\"type\":\"gradient_norm\""));
        assert!(json.contains("\"global_norm\":1.2"));
        assert!(json.contains("layer_0.weight"));
        assert!(json.contains("layer_1.weight"));
    }

    #[test]
    fn test_memory_usage_json() {
        let msg = DashboardMessage::MemoryUsage {
            step: 100,
            gpu_mb: 4096.0,
            cpu_mb: 8192.0,
            peak_gpu_mb: 5000.0,
        };
        let json = msg.to_json();
        assert!(json.contains("\"type\":\"memory_usage\""));
        assert!(json.contains("\"gpu_mb\":4096"));
        assert!(json.contains("\"peak_gpu_mb\":5000"));
    }

    #[test]
    fn test_checkpoint_saved_json_with_escape() {
        let msg = DashboardMessage::CheckpointSaved {
            step: 50,
            path: "/tmp/ckpt/step_50/model\"s.bin".to_string(),
        };
        let json = msg.to_json();
        assert!(json.contains("\"type\":\"checkpoint_saved\""));
        assert!(json.contains("\\\""));
    }

    #[test]
    fn test_training_complete_json() {
        let msg = DashboardMessage::TrainingComplete {
            total_steps: 1000,
            final_loss: 0.05,
            duration_secs: 3600.5,
        };
        let json = msg.to_json();
        assert!(json.contains("\"type\":\"training_complete\""));
        assert!(json.contains("\"total_steps\":1000"));
        assert!(json.contains("3600.5"));
    }

    #[test]
    fn test_heartbeat_json() {
        let msg = DashboardMessage::Heartbeat { timestamp_ms: 999_999 };
        let json = msg.to_json();
        assert!(json.contains("\"type\":\"heartbeat\""));
        assert!(json.contains("\"timestamp_ms\":999999"));
    }

    #[test]
    fn test_error_json() {
        let msg = DashboardMessage::Error { code: 500, message: "internal error".to_string() };
        let json = msg.to_json();
        assert!(json.contains("\"type\":\"error\""));
        assert!(json.contains("\"code\":500"));
        assert!(json.contains("internal error"));
    }

    // ── message_type ────────────────────────────────────────────────────────

    #[test]
    fn test_message_type_variants() {
        assert_eq!(
            DashboardMessage::Heartbeat { timestamp_ms: 0 }.message_type(),
            "heartbeat"
        );
        assert_eq!(
            DashboardMessage::Error { code: 0, message: String::new() }.message_type(),
            "error"
        );
        assert_eq!(
            DashboardMessage::CheckpointSaved { step: 0, path: String::new() }.message_type(),
            "checkpoint_saved"
        );
    }

    // ── MetricHistory ────────────────────────────────────────────────────────

    #[test]
    fn test_metric_history_push_and_latest() {
        let mut h = MetricHistory::new(5);
        assert!(h.latest().is_none());
        h.push(1, 2.0);
        h.push(2, 1.5);
        h.push(3, 1.0);
        assert_eq!(h.latest(), Some((3, 1.0)));
        assert_eq!(h.len(), 3);
    }

    #[test]
    fn test_metric_history_evicts_oldest_when_full() {
        let mut h = MetricHistory::new(3);
        h.push(1, 1.0);
        h.push(2, 2.0);
        h.push(3, 3.0);
        h.push(4, 4.0); // should evict step=1
        assert_eq!(h.len(), 3);
        assert_eq!(h.steps[0], 2);
        assert_eq!(h.latest(), Some((4, 4.0)));
    }

    #[test]
    fn test_metric_history_trend_decreasing() {
        let mut h = MetricHistory::new(100);
        for i in 0_u64..20 {
            h.push(i, 10.0 - i as f32 * 0.5);
        }
        let slope = h.trend(10).expect("should compute slope");
        assert!(slope < 0.0, "loss is decreasing so slope must be negative: {slope}");
    }

    #[test]
    fn test_metric_history_trend_none_with_single_point() {
        let mut h = MetricHistory::new(100);
        h.push(0, 1.0);
        assert!(h.trend(10).is_none());
    }

    #[test]
    fn test_metric_history_smooth_ema() {
        let mut h = MetricHistory::new(100);
        for i in 0..10_u64 {
            h.push(i, i as f32);
        }
        let smoothed = h.smooth(0.3);
        assert_eq!(smoothed.len(), 10);
        // first value equals original first value
        assert_eq!(smoothed[0], 0.0);
        // EMA should be less than the raw value for rising series (lagging)
        assert!(smoothed.last().copied().unwrap() < 9.0);
    }

    #[test]
    fn test_metric_history_smooth_alpha_one_equals_original() {
        let mut h = MetricHistory::new(10);
        let vals = [1.0f32, 3.0, 2.0, 5.0, 4.0];
        for (i, &v) in vals.iter().enumerate() {
            h.push(i as u64, v);
        }
        let smoothed = h.smooth(1.0);
        for (s, &orig) in smoothed.iter().zip(vals.iter()) {
            assert!((s - orig).abs() < 1e-5, "alpha=1 should pass through: {s} vs {orig}");
        }
    }

    // ── DashboardServerExt ───────────────────────────────────────────────────

    #[test]
    fn test_server_record_and_csv() {
        let mut s = DashboardServerExt::new(DashboardConfig::default());
        s.record_metrics(0, 2.0, 1e-3, 1.0);
        s.record_metrics(1, 1.5, 9e-4, 0.8);
        let csv = s.format_metric_csv();
        assert!(csv.starts_with("step,loss,lr,grad_norm\n"));
        assert!(csv.contains("0,2,"));
        assert!(csv.contains("1,1.5,"));
    }

    #[test]
    fn test_server_generate_summary_json_keys() {
        let mut s = DashboardServerExt::new(DashboardConfig::default());
        s.record_metrics(0, 1.0, 0.001, 0.5);
        let json = s.generate_summary_json();
        assert!(json.contains("\"loss\""));
        assert!(json.contains("\"lr\""));
        assert!(json.contains("\"grad_norm\""));
        assert!(json.contains("\"connected_clients\""));
        assert!(json.contains("\"messages_sent\""));
    }

    #[test]
    fn test_server_broadcast_and_buffer_full() {
        let config = DashboardConfig { message_buffer_size: 3, ..Default::default() };
        let mut s = DashboardServerExt::new(config);
        let msg = DashboardMessage::Heartbeat { timestamp_ms: 1 };
        assert!(s.broadcast_message(&msg).is_ok());
        assert!(s.broadcast_message(&msg).is_ok());
        assert!(s.broadcast_message(&msg).is_ok());
        let err = s.broadcast_message(&msg);
        assert!(matches!(err, Err(DashboardError::BufferFull(_))));
        assert_eq!(s.messages_sent, 3);
    }

    #[test]
    fn test_server_clear_buffer() {
        let mut s = DashboardServerExt::new(DashboardConfig::default());
        let msg = DashboardMessage::Heartbeat { timestamp_ms: 0 };
        s.broadcast_message(&msg).unwrap();
        assert_eq!(s.buffered_messages().len(), 1);
        s.clear_buffer();
        assert!(s.buffered_messages().is_empty());
    }

    #[test]
    fn test_dashboard_error_display() {
        assert_eq!(DashboardError::ConnectionFailed.to_string(), "dashboard connection failed");
        assert_eq!(DashboardError::SerializationError.to_string(), "failed to serialise dashboard message");
        assert!(DashboardError::BufferFull(10).to_string().contains("10"));
        assert!(DashboardError::ConfigError("bad port".to_string()).to_string().contains("bad port"));
    }
}
