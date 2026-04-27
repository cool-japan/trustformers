//! Higher-level operational monitoring for TrustformeRS serving.
//!
//! This module builds on the raw Prometheus-style metrics in
//! [`crate::metrics`] to provide:
//!
//! * **Sliding-window metric tracking** via [`MetricWindow`] — maintains a
//!   time-bounded circular buffer of samples and exposes statistical
//!   aggregates (mean, min, max, rate).
//!
//! * **Alert evaluation** via [`ServiceMonitor`] — continuously evaluates
//!   configurable [`AlertRule`]s against live metric windows, firing
//!   [`Alert`]s with configurable cooldown periods to suppress noise.
//!
//! * **Health assessment** — derives a [`HealthStatus`] from the current
//!   set of active alerts (Healthy / Degraded / Unhealthy).
//!
//! * **Predefined LLM serving rules** via [`default_llm_alert_rules`].
//!
//! The module is intentionally *side-effect-free*: no threads, no global
//! state.  Callers are responsible for calling [`ServiceMonitor::record_metric`]
//! and [`ServiceMonitor::check_alerts`] on their own schedule.

pub mod grafana;

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};

// ---------------------------------------------------------------------------
// Alert condition & severity
// ---------------------------------------------------------------------------

/// Predicate that determines when an [`AlertRule`] fires.
#[derive(Debug, Clone, PartialEq)]
pub enum AlertCondition {
    /// Fire when the most recent metric value exceeds a threshold.
    GreaterThan(f64),
    /// Fire when the most recent metric value is below a threshold.
    LessThan(f64),
    /// Fire when the absolute rate of change per second over `window_secs`
    /// exceeds `value`.
    RateExceeds { value: f64, window_secs: u64 },
    /// Fire when the absolute change in the mean between the first and last
    /// half of a sliding window exceeds `delta`.
    SuddenChange { delta: f64, window_secs: u64 },
}

/// Severity classification for a fired alert.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

impl std::fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            AlertSeverity::Info => "INFO",
            AlertSeverity::Warning => "WARNING",
            AlertSeverity::Critical => "CRITICAL",
            AlertSeverity::Emergency => "EMERGENCY",
        };
        f.write_str(s)
    }
}

// ---------------------------------------------------------------------------
// AlertRule
// ---------------------------------------------------------------------------

/// A declarative rule that maps a named metric to an alert condition.
#[derive(Debug, Clone)]
pub struct AlertRule {
    /// Human-readable name used in [`Alert`] messages.
    pub name: String,
    /// Name of the metric this rule evaluates (must match the key passed to
    /// [`ServiceMonitor::record_metric`]).
    pub metric: String,
    /// Condition under which the alert fires.
    pub condition: AlertCondition,
    /// Severity of the alert when it fires.
    pub severity: AlertSeverity,
    /// Minimum number of seconds between consecutive firings of this rule.
    pub cooldown_secs: u64,
}

impl AlertRule {
    /// Construct a new alert rule.
    pub fn new(
        name: impl Into<String>,
        metric: impl Into<String>,
        condition: AlertCondition,
        severity: AlertSeverity,
        cooldown_secs: u64,
    ) -> Self {
        AlertRule {
            name: name.into(),
            metric: metric.into(),
            condition,
            severity,
            cooldown_secs,
        }
    }
}

// ---------------------------------------------------------------------------
// Alert
// ---------------------------------------------------------------------------

/// A fired alert instance.
#[derive(Debug, Clone)]
pub struct Alert {
    /// Name of the rule that triggered this alert.
    pub rule_name: String,
    /// Severity of this alert.
    pub severity: AlertSeverity,
    /// Wall-clock time at which the alert was triggered.
    pub triggered_at: SystemTime,
    /// The metric value that caused the rule to fire.
    pub metric_value: f64,
    /// Human-readable alert message.
    pub message: String,
}

impl Alert {
    fn new(rule_name: &str, severity: AlertSeverity, metric_value: f64, message: String) -> Self {
        Alert {
            rule_name: rule_name.to_owned(),
            severity,
            triggered_at: SystemTime::now(),
            metric_value,
            message,
        }
    }
}

// ---------------------------------------------------------------------------
// MetricWindow
// ---------------------------------------------------------------------------

/// A sliding time-window buffer of `(Instant, f64)` samples.
///
/// Samples older than `window_duration_secs` are evicted on every [`Self::push`]
/// call. Statistical accessors operate only on the retained samples.
#[derive(Debug)]
pub struct MetricWindow {
    /// Raw samples within the retention window.
    pub samples: VecDeque<(Instant, f64)>,
    /// Duration of the window in seconds.
    pub window_duration_secs: u64,
}

impl MetricWindow {
    /// Create an empty window with the given duration.
    pub fn new(window_duration_secs: u64) -> Self {
        MetricWindow {
            samples: VecDeque::new(),
            window_duration_secs,
        }
    }

    /// Insert a new sample with the current timestamp, then evict stale data.
    pub fn push(&mut self, value: f64) {
        self.samples.push_back((Instant::now(), value));
        self.prune_old();
    }

    /// Remove samples older than the window boundary.
    pub fn prune_old(&mut self) {
        let cutoff = Duration::from_secs(self.window_duration_secs);
        while let Some((ts, _)) = self.samples.front() {
            if ts.elapsed() > cutoff {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }

    /// Arithmetic mean of all samples in the window.  Returns `0.0` when empty.
    pub fn mean(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.samples.iter().map(|(_, v)| v).sum();
        sum / self.samples.len() as f64
    }

    /// Maximum sample value within the window.  Returns `f64::NEG_INFINITY` when empty.
    pub fn max(&self) -> f64 {
        self.samples.iter().map(|(_, v)| *v).fold(f64::NEG_INFINITY, f64::max)
    }

    /// Minimum sample value within the window.  Returns `f64::INFINITY` when empty.
    pub fn min(&self) -> f64 {
        self.samples.iter().map(|(_, v)| *v).fold(f64::INFINITY, f64::min)
    }

    /// Rate of change per second: `(last - first) / elapsed_secs`.
    ///
    /// Returns `0.0` when fewer than two samples are present or the elapsed
    /// time is zero.
    pub fn rate(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let (first_ts, first_val) = self.samples.front().expect("len >= 2");
        let (last_ts, last_val) = self.samples.back().expect("len >= 2");
        let elapsed = last_ts
            .checked_duration_since(*first_ts)
            .unwrap_or(Duration::ZERO)
            .as_secs_f64();
        if elapsed < f64::EPSILON {
            return 0.0;
        }
        (last_val - first_val) / elapsed
    }

    /// Most recent sample value.  Returns `None` when empty.
    pub fn latest(&self) -> Option<f64> {
        self.samples.back().map(|(_, v)| *v)
    }

    /// Number of samples currently retained.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the window contains no samples.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Mean of the first half of the window samples.
    fn first_half_mean(&self) -> f64 {
        let mid = self.samples.len() / 2;
        if mid == 0 {
            return self.mean();
        }
        let sum: f64 = self.samples.iter().take(mid).map(|(_, v)| v).sum();
        sum / mid as f64
    }

    /// Mean of the second half of the window samples.
    fn second_half_mean(&self) -> f64 {
        let mid = self.samples.len() / 2;
        let tail_len = self.samples.len() - mid;
        if tail_len == 0 {
            return self.mean();
        }
        let sum: f64 = self.samples.iter().skip(mid).map(|(_, v)| v).sum();
        sum / tail_len as f64
    }
}

// ---------------------------------------------------------------------------
// HealthStatus
// ---------------------------------------------------------------------------

/// High-level health classification derived from active alerts.
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    /// No active alerts.
    Healthy,
    /// One or more Warning-level active alerts; no Critical or Emergency.
    Degraded { reasons: Vec<String> },
    /// At least one Critical or Emergency active alert.
    Unhealthy { reasons: Vec<String> },
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "HEALTHY"),
            HealthStatus::Degraded { reasons } => {
                write!(f, "DEGRADED ({})", reasons.join("; "))
            },
            HealthStatus::Unhealthy { reasons } => {
                write!(f, "UNHEALTHY ({})", reasons.join("; "))
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Retention configuration
// ---------------------------------------------------------------------------

/// Retention periods for raw samples and alert history.
#[derive(Debug, Clone)]
pub struct RetentionConfig {
    /// How long to retain raw metric samples (seconds).
    pub raw_sample_window_secs: u64,
    /// Maximum number of alerts to keep in the history ring buffer.
    pub max_alert_history: usize,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        RetentionConfig {
            raw_sample_window_secs: 300, // 5 minutes
            max_alert_history: 1000,
        }
    }
}

// ---------------------------------------------------------------------------
// MonitoringConfig
// ---------------------------------------------------------------------------

/// Top-level monitoring configuration.
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Alert rules evaluated by [`ServiceMonitor::check_alerts`].
    pub alert_rules: Vec<AlertRule>,
    /// Suggested scrape interval (not enforced by the library; informational).
    pub scrape_interval_ms: u64,
    /// Retention settings for samples and alerts.
    pub retention_periods: RetentionConfig,
    /// Optional TCP port for a future HTTP dashboard server (not started here).
    pub dashboard_port: Option<u16>,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        MonitoringConfig {
            alert_rules: default_llm_alert_rules(),
            scrape_interval_ms: 5_000,
            retention_periods: RetentionConfig::default(),
            dashboard_port: None,
        }
    }
}

// ---------------------------------------------------------------------------
// ServiceMonitor
// ---------------------------------------------------------------------------

/// Operational monitor for a TrustformeRS serving instance.
///
/// Usage pattern
/// -------------
/// 1. Create via `ServiceMonitor::new(config)`.
/// 2. On each metric observation, call `record_metric(name, value)`.
/// 3. Periodically call `check_alerts()` to evaluate all rules and receive
///    newly-fired alerts (those that are not within cooldown).
/// 4. Inspect `health_status()` or `health_report()` for a summary.
pub struct ServiceMonitor {
    config: MonitoringConfig,
    /// Per-metric sliding windows.
    metric_windows: HashMap<String, MetricWindow>,
    /// Currently active (unresolved) alerts.
    active_alerts: Vec<Alert>,
    /// Ring-buffer of historical alerts (capped by `max_alert_history`).
    alert_history: VecDeque<Alert>,
    /// Tracks the last wall-clock time each rule fired (for cooldown).
    last_alert_times: HashMap<String, Instant>,
}

impl ServiceMonitor {
    /// Create a new `ServiceMonitor` with the given configuration.
    pub fn new(config: MonitoringConfig) -> Self {
        ServiceMonitor {
            config,
            metric_windows: HashMap::new(),
            active_alerts: Vec::new(),
            alert_history: VecDeque::new(),
            last_alert_times: HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Metric recording
    // -----------------------------------------------------------------------

    /// Record a single metric sample.
    ///
    /// If no window exists for `name` yet, one is created using the configured
    /// `raw_sample_window_secs` retention period.
    pub fn record_metric(&mut self, name: &str, value: f64) {
        let window_secs = self.config.retention_periods.raw_sample_window_secs;
        self.metric_windows
            .entry(name.to_owned())
            .or_insert_with(|| MetricWindow::new(window_secs))
            .push(value);
    }

    // -----------------------------------------------------------------------
    // Alert evaluation
    // -----------------------------------------------------------------------

    /// Evaluate all configured alert rules and return newly-fired alerts.
    ///
    /// A rule only fires if:
    /// * Its condition is met by the current window data.
    /// * The cooldown period since the last firing has elapsed.
    ///
    /// Fired alerts are appended to `active_alerts` and `alert_history`.
    pub fn check_alerts(&mut self) -> Vec<Alert> {
        let mut newly_fired: Vec<Alert> = Vec::new();

        // Collect rules to avoid borrow conflict.
        let rules: Vec<AlertRule> = self.config.alert_rules.clone();

        for rule in &rules {
            // Cooldown check.
            if let Some(&last_fired) = self.last_alert_times.get(&rule.name) {
                if last_fired.elapsed().as_secs() < rule.cooldown_secs {
                    continue;
                }
            }

            // Retrieve window for this metric.
            let window = match self.metric_windows.get_mut(&rule.metric) {
                Some(w) => {
                    w.prune_old();
                    w
                },
                None => continue, // no data yet
            };

            if window.is_empty() {
                continue;
            }

            let fired = Self::evaluate_condition(&rule.condition, window);
            if let Some((metric_value, description)) = fired {
                let message = format!(
                    "[{}] {} — metric '{}' = {:.4}: {}",
                    rule.severity, rule.name, rule.metric, metric_value, description
                );
                let alert = Alert::new(&rule.name, rule.severity.clone(), metric_value, message);
                self.last_alert_times.insert(rule.name.clone(), Instant::now());
                self.active_alerts.push(alert.clone());
                // Maintain history cap.
                self.alert_history.push_back(alert.clone());
                let max = self.config.retention_periods.max_alert_history;
                while self.alert_history.len() > max {
                    self.alert_history.pop_front();
                }
                newly_fired.push(alert);
            }
        }

        newly_fired
    }

    /// Evaluate a single [`AlertCondition`] against a [`MetricWindow`].
    ///
    /// Returns `Some((metric_value, description))` if the condition is met.
    fn evaluate_condition(
        condition: &AlertCondition,
        window: &MetricWindow,
    ) -> Option<(f64, String)> {
        match condition {
            AlertCondition::GreaterThan(threshold) => {
                let latest = window.latest()?;
                if latest > *threshold {
                    Some((
                        latest,
                        format!("value {latest:.4} > threshold {threshold:.4}"),
                    ))
                } else {
                    None
                }
            },
            AlertCondition::LessThan(threshold) => {
                let latest = window.latest()?;
                if latest < *threshold {
                    Some((
                        latest,
                        format!("value {latest:.4} < threshold {threshold:.4}"),
                    ))
                } else {
                    None
                }
            },
            AlertCondition::RateExceeds { value, .. } => {
                let rate = window.rate().abs();
                if rate > *value {
                    Some((rate, format!("rate {rate:.4}/s > threshold {value:.4}/s")))
                } else {
                    None
                }
            },
            AlertCondition::SuddenChange { delta, .. } => {
                if window.len() < 4 {
                    return None; // need enough samples to split
                }
                let first_half = window.first_half_mean();
                let second_half = window.second_half_mean();
                let change = (second_half - first_half).abs();
                if change > *delta {
                    Some((
                        change,
                        format!(
                            "sudden change {change:.4} > delta {delta:.4} \
                             (first_half_mean={first_half:.4}, second_half_mean={second_half:.4})"
                        ),
                    ))
                } else {
                    None
                }
            },
        }
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Currently active alerts (may include resolved alerts if not cleared).
    pub fn active_alerts(&self) -> &[Alert] {
        &self.active_alerts
    }

    /// Historical ring-buffer of all ever-fired alerts.
    pub fn alert_history(&self) -> &VecDeque<Alert> {
        &self.alert_history
    }

    /// Clear all active alerts (e.g. after human acknowledgement).
    pub fn clear_active_alerts(&mut self) {
        self.active_alerts.clear();
    }

    /// Derive a [`HealthStatus`] from the current active alert set.
    pub fn health_status(&self) -> HealthStatus {
        if self.active_alerts.is_empty() {
            return HealthStatus::Healthy;
        }

        let mut critical_reasons: Vec<String> = Vec::new();
        let mut warning_reasons: Vec<String> = Vec::new();

        for alert in &self.active_alerts {
            match alert.severity {
                AlertSeverity::Emergency | AlertSeverity::Critical => {
                    critical_reasons.push(alert.rule_name.clone());
                },
                AlertSeverity::Warning | AlertSeverity::Info => {
                    warning_reasons.push(alert.rule_name.clone());
                },
            }
        }

        if !critical_reasons.is_empty() {
            HealthStatus::Unhealthy {
                reasons: critical_reasons,
            }
        } else {
            HealthStatus::Degraded {
                reasons: warning_reasons,
            }
        }
    }

    /// Generate a human-readable text health report.
    pub fn health_report(&self) -> String {
        let status = self.health_status();
        let mut report = format!("=== TrustformeRS Health Report ===\nStatus: {status}\n");

        report.push_str(&format!("Active alerts: {}\n", self.active_alerts.len()));

        if !self.active_alerts.is_empty() {
            report.push_str("\nActive Alerts:\n");
            for alert in &self.active_alerts {
                report.push_str(&format!(
                    "  [{severity}] {name}: {msg}\n",
                    severity = alert.severity,
                    name = alert.rule_name,
                    msg = alert.message
                ));
            }
        }

        report.push_str(&format!(
            "\nMetrics tracked: {}\n",
            self.metric_windows.len()
        ));
        for (name, window) in &self.metric_windows {
            if !window.is_empty() {
                report.push_str(&format!(
                    "  {name}: latest={latest:.4}, mean={mean:.4}, samples={n}\n",
                    latest = window.latest().unwrap_or(0.0),
                    mean = window.mean(),
                    n = window.len()
                ));
            }
        }

        report.push_str(&format!(
            "\nAlert history: {} total\n",
            self.alert_history.len()
        ));
        report
    }

    /// Read access to the underlying metric windows (for external inspection).
    pub fn metric_window(&self, name: &str) -> Option<&MetricWindow> {
        self.metric_windows.get(name)
    }
}

// ---------------------------------------------------------------------------
// Predefined LLM serving alert rules
// ---------------------------------------------------------------------------

/// Returns a set of sensible default alert rules for LLM inference serving.
///
/// Rules
/// -----
/// | Rule name            | Metric             | Condition                   | Severity  |
/// |----------------------|--------------------|---------------------------  |-----------|
/// | `high_p99_latency`   | `p99_latency_ms`   | > 1 000 ms                 | Warning   |
/// | `very_high_latency`  | `p99_latency_ms`   | > 5 000 ms                 | Critical  |
/// | `high_error_rate`    | `error_rate`       | > 0.01 (1 %)               | Warning   |
/// | `critical_errors`    | `error_rate`       | > 0.05 (5 %)               | Critical  |
/// | `gpu_oom_risk`       | `gpu_memory_frac`  | > 0.95                     | Critical  |
/// | `gpu_memory_high`    | `gpu_memory_frac`  | > 0.85                     | Warning   |
/// | `queue_backlog`      | `queue_depth`      | > 100                      | Warning   |
/// | `queue_critical`     | `queue_depth`      | > 500                      | Critical  |
/// | `throughput_drop`    | `tokens_per_sec`   | SuddenChange > 20 %        | Warning   |
/// | `latency_spike`      | `p99_latency_ms`   | RateExceeds 100 ms/s       | Warning   |
pub fn default_llm_alert_rules() -> Vec<AlertRule> {
    vec![
        // Latency — Warning
        AlertRule::new(
            "high_p99_latency",
            "p99_latency_ms",
            AlertCondition::GreaterThan(1_000.0),
            AlertSeverity::Warning,
            60,
        ),
        // Latency — Critical
        AlertRule::new(
            "very_high_latency",
            "p99_latency_ms",
            AlertCondition::GreaterThan(5_000.0),
            AlertSeverity::Critical,
            30,
        ),
        // Error rate — Warning
        AlertRule::new(
            "high_error_rate",
            "error_rate",
            AlertCondition::GreaterThan(0.01),
            AlertSeverity::Warning,
            60,
        ),
        // Error rate — Critical
        AlertRule::new(
            "critical_errors",
            "error_rate",
            AlertCondition::GreaterThan(0.05),
            AlertSeverity::Critical,
            30,
        ),
        // GPU memory — OOM risk
        AlertRule::new(
            "gpu_oom_risk",
            "gpu_memory_frac",
            AlertCondition::GreaterThan(0.95),
            AlertSeverity::Critical,
            30,
        ),
        // GPU memory — Warning
        AlertRule::new(
            "gpu_memory_high",
            "gpu_memory_frac",
            AlertCondition::GreaterThan(0.85),
            AlertSeverity::Warning,
            120,
        ),
        // Queue depth — Warning
        AlertRule::new(
            "queue_backlog",
            "queue_depth",
            AlertCondition::GreaterThan(100.0),
            AlertSeverity::Warning,
            60,
        ),
        // Queue depth — Critical
        AlertRule::new(
            "queue_critical",
            "queue_depth",
            AlertCondition::GreaterThan(500.0),
            AlertSeverity::Critical,
            30,
        ),
        // Throughput sudden drop
        AlertRule::new(
            "throughput_drop",
            "tokens_per_sec",
            AlertCondition::SuddenChange {
                delta: 20.0,
                window_secs: 60,
            },
            AlertSeverity::Warning,
            120,
        ),
        // Latency spike rate
        AlertRule::new(
            "latency_spike",
            "p99_latency_ms",
            AlertCondition::RateExceeds {
                value: 100.0,
                window_secs: 60,
            },
            AlertSeverity::Warning,
            60,
        ),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── MetricWindow ───────────────────────────────────────────────────────

    #[test]
    fn test_metric_window_empty_stats() {
        let w = MetricWindow::new(60);
        assert_eq!(w.mean(), 0.0);
        assert_eq!(w.rate(), 0.0);
        assert!(w.is_empty());
        assert!(w.latest().is_none());
    }

    #[test]
    fn test_metric_window_push_and_len() {
        let mut w = MetricWindow::new(60);
        w.push(1.0);
        w.push(2.0);
        w.push(3.0);
        assert_eq!(w.len(), 3);
    }

    #[test]
    fn test_metric_window_mean() {
        let mut w = MetricWindow::new(60);
        w.push(10.0);
        w.push(20.0);
        w.push(30.0);
        assert!((w.mean() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_metric_window_max_min() {
        let mut w = MetricWindow::new(60);
        w.push(5.0);
        w.push(1.0);
        w.push(9.0);
        assert!((w.max() - 9.0).abs() < 1e-9);
        assert!((w.min() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_metric_window_latest() {
        let mut w = MetricWindow::new(60);
        w.push(7.0);
        w.push(42.0);
        assert_eq!(w.latest(), Some(42.0));
    }

    #[test]
    fn test_metric_window_rate_positive() {
        let mut w = MetricWindow::new(3600);
        // Inject samples with synthetic timestamps by pushing sequentially.
        // We can only test the sign / direction, not exact value, without
        // sleeping.
        w.push(0.0);
        w.push(100.0);
        // Rate should be positive (increasing).
        assert!(w.rate() >= 0.0);
    }

    #[test]
    fn test_metric_window_rate_single_sample() {
        let mut w = MetricWindow::new(60);
        w.push(42.0);
        assert_eq!(w.rate(), 0.0); // need at least 2 samples
    }

    #[test]
    fn test_metric_window_max_empty_returns_neg_infinity() {
        let w = MetricWindow::new(60);
        assert_eq!(w.max(), f64::NEG_INFINITY);
    }

    #[test]
    fn test_metric_window_min_empty_returns_infinity() {
        let w = MetricWindow::new(60);
        assert_eq!(w.min(), f64::INFINITY);
    }

    // ── AlertCondition evaluation ──────────────────────────────────────────

    fn window_with_values(values: &[f64]) -> MetricWindow {
        let mut w = MetricWindow::new(3600);
        for &v in values {
            w.push(v);
        }
        w
    }

    #[test]
    fn test_alert_condition_greater_than_fires() {
        let w = window_with_values(&[50.0, 1500.0]);
        let result = ServiceMonitor::evaluate_condition(&AlertCondition::GreaterThan(1000.0), &w);
        assert!(result.is_some());
    }

    #[test]
    fn test_alert_condition_greater_than_no_fire() {
        let w = window_with_values(&[500.0]);
        let result = ServiceMonitor::evaluate_condition(&AlertCondition::GreaterThan(1000.0), &w);
        assert!(result.is_none());
    }

    #[test]
    fn test_alert_condition_less_than_fires() {
        let w = window_with_values(&[0.005]);
        let result = ServiceMonitor::evaluate_condition(&AlertCondition::LessThan(0.01), &w);
        assert!(result.is_some());
    }

    #[test]
    fn test_alert_condition_less_than_no_fire() {
        let w = window_with_values(&[0.5]);
        let result = ServiceMonitor::evaluate_condition(&AlertCondition::LessThan(0.01), &w);
        assert!(result.is_none());
    }

    #[test]
    fn test_alert_condition_sudden_change_fires() {
        // First half mean ≈ 0, second half mean ≈ 100, change = 100 > delta 50.
        let values = [0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0];
        let w = window_with_values(&values);
        let result = ServiceMonitor::evaluate_condition(
            &AlertCondition::SuddenChange {
                delta: 50.0,
                window_secs: 60,
            },
            &w,
        );
        assert!(result.is_some());
    }

    #[test]
    fn test_alert_condition_sudden_change_no_fire_stable() {
        let values = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
        let w = window_with_values(&values);
        let result = ServiceMonitor::evaluate_condition(
            &AlertCondition::SuddenChange {
                delta: 50.0,
                window_secs: 60,
            },
            &w,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_alert_condition_sudden_change_too_few_samples() {
        // < 4 samples → no fire.
        let w = window_with_values(&[0.0, 100.0]);
        let result = ServiceMonitor::evaluate_condition(
            &AlertCondition::SuddenChange {
                delta: 1.0,
                window_secs: 60,
            },
            &w,
        );
        assert!(result.is_none());
    }

    // ── ServiceMonitor ────────────────────────────────────────────────────

    fn monitor_with_single_rule(
        metric: &str,
        condition: AlertCondition,
        severity: AlertSeverity,
    ) -> ServiceMonitor {
        let rule = AlertRule::new("test_rule", metric, condition, severity, 0);
        let config = MonitoringConfig {
            alert_rules: vec![rule],
            scrape_interval_ms: 1000,
            retention_periods: RetentionConfig::default(),
            dashboard_port: None,
        };
        ServiceMonitor::new(config)
    }

    #[test]
    fn test_service_monitor_record_and_window_created() {
        let mut mon = ServiceMonitor::new(MonitoringConfig::default());
        mon.record_metric("p99_latency_ms", 500.0);
        assert!(mon.metric_window("p99_latency_ms").is_some());
    }

    #[test]
    fn test_service_monitor_check_alerts_fires_when_threshold_exceeded() {
        let mut mon = monitor_with_single_rule(
            "latency",
            AlertCondition::GreaterThan(100.0),
            AlertSeverity::Warning,
        );
        mon.record_metric("latency", 250.0);
        let fired = mon.check_alerts();
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].rule_name, "test_rule");
    }

    #[test]
    fn test_service_monitor_check_alerts_no_fire_below_threshold() {
        let mut mon = monitor_with_single_rule(
            "latency",
            AlertCondition::GreaterThan(1000.0),
            AlertSeverity::Warning,
        );
        mon.record_metric("latency", 50.0);
        let fired = mon.check_alerts();
        assert!(fired.is_empty());
    }

    #[test]
    fn test_service_monitor_cooldown_suppresses_repeated_alerts() {
        let rule = AlertRule::new(
            "cooldown_rule",
            "metric",
            AlertCondition::GreaterThan(0.0),
            AlertSeverity::Warning,
            9999, // very long cooldown
        );
        let config = MonitoringConfig {
            alert_rules: vec![rule],
            scrape_interval_ms: 1000,
            retention_periods: RetentionConfig::default(),
            dashboard_port: None,
        };
        let mut mon = ServiceMonitor::new(config);
        mon.record_metric("metric", 100.0);
        let first = mon.check_alerts();
        assert_eq!(first.len(), 1);
        // Second check should be suppressed by cooldown.
        mon.record_metric("metric", 200.0);
        let second = mon.check_alerts();
        assert!(second.is_empty());
    }

    #[test]
    fn test_service_monitor_health_status_healthy() {
        let mon = ServiceMonitor::new(MonitoringConfig {
            alert_rules: vec![],
            ..Default::default()
        });
        assert_eq!(mon.health_status(), HealthStatus::Healthy);
    }

    #[test]
    fn test_service_monitor_health_status_degraded_on_warning() {
        let mut mon = monitor_with_single_rule(
            "metric",
            AlertCondition::GreaterThan(0.0),
            AlertSeverity::Warning,
        );
        mon.record_metric("metric", 1.0);
        mon.check_alerts();
        match mon.health_status() {
            HealthStatus::Degraded { reasons } => {
                assert!(reasons.contains(&"test_rule".to_owned()));
            },
            other => panic!("expected Degraded, got {other:?}"),
        }
    }

    #[test]
    fn test_service_monitor_health_status_unhealthy_on_critical() {
        let mut mon = monitor_with_single_rule(
            "metric",
            AlertCondition::GreaterThan(0.0),
            AlertSeverity::Critical,
        );
        mon.record_metric("metric", 1.0);
        mon.check_alerts();
        match mon.health_status() {
            HealthStatus::Unhealthy { reasons } => {
                assert!(reasons.contains(&"test_rule".to_owned()));
            },
            other => panic!("expected Unhealthy, got {other:?}"),
        }
    }

    #[test]
    fn test_service_monitor_clear_active_alerts() {
        let mut mon = monitor_with_single_rule(
            "metric",
            AlertCondition::GreaterThan(0.0),
            AlertSeverity::Warning,
        );
        mon.record_metric("metric", 1.0);
        mon.check_alerts();
        assert!(!mon.active_alerts().is_empty());
        mon.clear_active_alerts();
        assert!(mon.active_alerts().is_empty());
        assert_eq!(mon.health_status(), HealthStatus::Healthy);
    }

    #[test]
    fn test_service_monitor_alert_history_retained() {
        let mut mon = monitor_with_single_rule(
            "metric",
            AlertCondition::GreaterThan(0.0),
            AlertSeverity::Info,
        );
        mon.record_metric("metric", 1.0);
        mon.check_alerts();
        assert_eq!(mon.alert_history().len(), 1);
    }

    #[test]
    fn test_service_monitor_health_report_contains_status() {
        let mut mon = monitor_with_single_rule(
            "latency",
            AlertCondition::GreaterThan(0.0),
            AlertSeverity::Critical,
        );
        mon.record_metric("latency", 9999.0);
        mon.check_alerts();
        let report = mon.health_report();
        assert!(report.contains("UNHEALTHY") || report.contains("CRITICAL"));
    }

    #[test]
    fn test_service_monitor_no_data_no_alert() {
        let rule = AlertRule::new(
            "no_data",
            "missing_metric",
            AlertCondition::GreaterThan(0.0),
            AlertSeverity::Critical,
            0,
        );
        let config = MonitoringConfig {
            alert_rules: vec![rule],
            ..Default::default()
        };
        let mut mon = ServiceMonitor::new(config);
        let fired = mon.check_alerts();
        assert!(fired.is_empty());
    }

    // ── Default rules ──────────────────────────────────────────────────────

    #[test]
    fn test_default_llm_alert_rules_not_empty() {
        let rules = default_llm_alert_rules();
        assert!(!rules.is_empty());
    }

    #[test]
    fn test_default_llm_alert_rules_covers_key_metrics() {
        let rules = default_llm_alert_rules();
        let metrics: Vec<&str> = rules.iter().map(|r| r.metric.as_str()).collect();
        assert!(metrics.contains(&"p99_latency_ms"));
        assert!(metrics.contains(&"error_rate"));
        assert!(metrics.contains(&"gpu_memory_frac"));
        assert!(metrics.contains(&"queue_depth"));
    }

    #[test]
    fn test_default_llm_alert_rules_gpu_oom_fires() {
        let rules = default_llm_alert_rules();
        let gpu_rule = rules
            .iter()
            .find(|r| r.name == "gpu_oom_risk")
            .expect("gpu_oom_risk rule must exist");
        let w = window_with_values(&[0.97]);
        let fired = ServiceMonitor::evaluate_condition(&gpu_rule.condition, &w);
        assert!(fired.is_some());
    }

    #[test]
    fn test_default_llm_alert_rules_high_error_rate_fires() {
        let rules = default_llm_alert_rules();
        let rule = rules
            .iter()
            .find(|r| r.name == "high_error_rate")
            .expect("high_error_rate must exist");
        let w = window_with_values(&[0.05]);
        let fired = ServiceMonitor::evaluate_condition(&rule.condition, &w);
        assert!(fired.is_some());
    }

    #[test]
    fn test_alert_severity_ordering() {
        assert!(AlertSeverity::Info < AlertSeverity::Warning);
        assert!(AlertSeverity::Warning < AlertSeverity::Critical);
        assert!(AlertSeverity::Critical < AlertSeverity::Emergency);
    }

    #[test]
    fn test_alert_severity_display() {
        assert_eq!(AlertSeverity::Info.to_string(), "INFO");
        assert_eq!(AlertSeverity::Warning.to_string(), "WARNING");
        assert_eq!(AlertSeverity::Critical.to_string(), "CRITICAL");
        assert_eq!(AlertSeverity::Emergency.to_string(), "EMERGENCY");
    }

    // ── Extended tests ─────────────────────────────────────────────────────

    // Test: MetricWindow::mean with single sample
    #[test]
    fn test_metric_window_mean_single_sample() {
        let mut w = MetricWindow::new(60);
        w.push(99.0);
        assert!((w.mean() - 99.0).abs() < 1e-9);
    }

    // Test: MetricWindow::len is zero after creation
    #[test]
    fn test_metric_window_len_zero_after_creation() {
        let w = MetricWindow::new(60);
        assert_eq!(w.len(), 0);
    }

    // Test: MetricWindow::latest after multiple pushes
    #[test]
    fn test_metric_window_latest_tracks_last_push() {
        let mut w = MetricWindow::new(3600);
        w.push(1.0);
        w.push(2.0);
        w.push(99.0);
        assert_eq!(w.latest(), Some(99.0));
    }

    // Test: MetricWindow::mean after zero-value pushes
    #[test]
    fn test_metric_window_mean_all_zeros() {
        let mut w = MetricWindow::new(3600);
        for _ in 0..5 {
            w.push(0.0);
        }
        assert!((w.mean() - 0.0).abs() < 1e-9);
    }

    // Test: AlertCondition::LessThan fires when value is below threshold
    #[test]
    fn test_alert_condition_less_than_fires_ext() {
        let w = window_with_values(&[500.0, 50.0]);
        let result = ServiceMonitor::evaluate_condition(&AlertCondition::LessThan(100.0), &w);
        assert!(
            result.is_some(),
            "LessThan should fire when value < threshold"
        );
    }

    // Test: AlertCondition::LessThan does not fire when value is above threshold
    #[test]
    fn test_alert_condition_less_than_no_fire_ext() {
        let w = window_with_values(&[200.0]);
        let result = ServiceMonitor::evaluate_condition(&AlertCondition::LessThan(100.0), &w);
        assert!(
            result.is_none(),
            "LessThan should not fire when value >= threshold"
        );
    }

    // Test: AlertCondition::GreaterThan does not fire below threshold
    #[test]
    fn test_alert_condition_greater_than_no_fire_ext() {
        let w = window_with_values(&[50.0]);
        let result = ServiceMonitor::evaluate_condition(&AlertCondition::GreaterThan(100.0), &w);
        assert!(result.is_none());
    }

    // Test: AlertCondition on empty window returns None
    #[test]
    fn test_alert_condition_empty_window_returns_none() {
        let w = MetricWindow::new(60);
        let result = ServiceMonitor::evaluate_condition(&AlertCondition::GreaterThan(0.0), &w);
        assert!(result.is_none(), "empty window should never fire");
    }

    // Test: AlertRule construction stores fields
    #[test]
    fn test_alert_rule_new_stores_fields() {
        let rule = AlertRule::new(
            "latency_high",
            "p99_latency_ms",
            AlertCondition::GreaterThan(500.0),
            AlertSeverity::Warning,
            60,
        );
        assert_eq!(rule.name, "latency_high");
        assert_eq!(rule.metric, "p99_latency_ms");
        assert_eq!(rule.cooldown_secs, 60);
        assert_eq!(rule.severity, AlertSeverity::Warning);
    }

    // Test: HealthStatus::Healthy display
    #[test]
    fn test_health_status_healthy_display() {
        assert_eq!(HealthStatus::Healthy.to_string(), "HEALTHY");
    }

    // Test: HealthStatus::Degraded display contains reason
    #[test]
    fn test_health_status_degraded_display() {
        let s = HealthStatus::Degraded {
            reasons: vec!["latency_spike".to_owned()],
        };
        assert!(s.to_string().contains("latency_spike"));
    }

    // Test: HealthStatus::Unhealthy display contains UNHEALTHY
    #[test]
    fn test_health_status_unhealthy_display() {
        let s = HealthStatus::Unhealthy {
            reasons: vec!["oom".to_owned()],
        };
        assert!(s.to_string().contains("UNHEALTHY"));
    }

    // Test: RetentionConfig default values
    #[test]
    fn test_retention_config_default() {
        let cfg = RetentionConfig::default();
        assert!(cfg.raw_sample_window_secs > 0);
        assert!(cfg.max_alert_history > 0);
    }

    // Test: MonitoringConfig default — alert_rules comes pre-populated with
    // the standard LLM alert rules (from default_llm_alert_rules()).
    #[test]
    fn test_monitoring_config_default_no_rules() {
        let cfg = MonitoringConfig::default();
        // The default configuration ships with predefined LLM serving rules.
        assert!(
            !cfg.alert_rules.is_empty(),
            "default config should include the built-in LLM alert rules"
        );
    }

    // Test: ServiceMonitor has no active alerts initially
    #[test]
    fn test_service_monitor_no_active_alerts_initially() {
        let mon = ServiceMonitor::new(MonitoringConfig::default());
        assert!(mon.active_alerts().is_empty());
    }

    // Test: ServiceMonitor alert history empty initially
    #[test]
    fn test_service_monitor_alert_history_empty_initially() {
        let mon = ServiceMonitor::new(MonitoringConfig::default());
        assert!(mon.alert_history().is_empty());
    }

    // Test: recording metric twice both values appear
    #[test]
    fn test_service_monitor_record_metric_twice() {
        let rule = AlertRule::new(
            "r",
            "cpu",
            AlertCondition::GreaterThan(1000.0), // high threshold to avoid spurious fire
            AlertSeverity::Info,
            0,
        );
        let config = MonitoringConfig {
            alert_rules: vec![rule],
            ..Default::default()
        };
        let mut mon = ServiceMonitor::new(config);
        mon.record_metric("cpu", 10.0);
        mon.record_metric("cpu", 20.0);
        let fired = mon.check_alerts();
        assert!(fired.is_empty(), "below threshold should not fire");
    }

    // Test: RateExceeds does not fire on flat signal
    #[test]
    fn test_alert_condition_rate_exceeds_flat_no_fire() {
        let w = window_with_values(&[5.0, 5.0, 5.0, 5.0]);
        let cond = AlertCondition::RateExceeds {
            value: 10.0,
            window_secs: 60,
        };
        let result = ServiceMonitor::evaluate_condition(&cond, &w);
        // Flat signal has near-zero rate; should not fire
        assert!(
            result.is_none(),
            "flat signal should not exceed rate threshold"
        );
    }

    // Test: SuddenChange does not fire on flat signal
    #[test]
    fn test_alert_condition_sudden_change_flat_no_fire() {
        let w = window_with_values(&[100.0, 100.0, 100.0, 100.0, 100.0, 100.0]);
        let cond = AlertCondition::SuddenChange {
            delta: 50.0,
            window_secs: 60,
        };
        let result = ServiceMonitor::evaluate_condition(&cond, &w);
        assert!(
            result.is_none(),
            "flat signal should not trigger SuddenChange"
        );
    }

    // Test: SuddenChange fires on large step change
    #[test]
    fn test_alert_condition_sudden_change_fires_on_spike() {
        // First half: ~0, second half: ~1000
        let mut values: Vec<f64> = vec![1.0; 10];
        for v in values.iter_mut().take(5) {
            *v = 1.0;
        }
        for v in values.iter_mut().skip(5) {
            *v = 1000.0;
        }
        let w = window_with_values(&values);
        let cond = AlertCondition::SuddenChange {
            delta: 500.0,
            window_secs: 60,
        };
        let result = ServiceMonitor::evaluate_condition(&cond, &w);
        assert!(
            result.is_some(),
            "large step change should trigger SuddenChange"
        );
    }

    // Test: default_llm_alert_rules — each rule has non-empty name and metric
    #[test]
    fn test_default_llm_alert_rules_non_empty_names() {
        let rules = default_llm_alert_rules();
        for rule in &rules {
            assert!(!rule.name.is_empty(), "rule name must be non-empty");
            assert!(!rule.metric.is_empty(), "rule metric must be non-empty");
        }
    }

    // Test: AlertSeverity::Info != AlertSeverity::Critical
    #[test]
    fn test_alert_severity_info_ne_critical() {
        assert_ne!(AlertSeverity::Info, AlertSeverity::Critical);
    }

    // Test: check_alerts — Info-level alert fires (stored in history)
    #[test]
    fn test_info_level_alert_fires() {
        let rule = AlertRule::new(
            "info_rule",
            "metric",
            AlertCondition::GreaterThan(0.0),
            AlertSeverity::Info,
            0,
        );
        let config = MonitoringConfig {
            alert_rules: vec![rule],
            ..Default::default()
        };
        let mut mon = ServiceMonitor::new(config);
        mon.record_metric("metric", 1.0);
        let fired = mon.check_alerts();
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].severity, AlertSeverity::Info);
    }

    // Test: health_report returns non-empty string
    #[test]
    fn test_health_report_non_empty() {
        let mon = ServiceMonitor::new(MonitoringConfig::default());
        assert!(!mon.health_report().is_empty());
    }

    // Test: MetricWindow::is_empty is false after push
    #[test]
    fn test_metric_window_not_empty_after_push() {
        let mut w = MetricWindow::new(60);
        w.push(5.0);
        assert!(!w.is_empty());
    }
}
