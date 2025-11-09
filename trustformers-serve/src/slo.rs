//! Service Level Objectives (SLO) Tracking System
//!
//! Comprehensive SLO monitoring and tracking with error budget calculation,
//! breach detection, and performance analysis for production deployments.

use anyhow::Result;
use prometheus::{
    register_counter_vec, register_gauge_vec, register_histogram_vec, Counter, Gauge, Histogram,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use uuid::Uuid;

/// SLO configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SloConfig {
    /// Enable SLO tracking
    pub enabled: bool,
    /// SLO evaluation interval in seconds
    pub evaluation_interval_seconds: u64,
    /// SLO window duration in seconds (e.g., 30 days)
    pub window_duration_seconds: u64,
    /// Enable error budget alerts
    pub enable_error_budget_alerts: bool,
    /// Error budget alert threshold (0.0-1.0)
    pub error_budget_alert_threshold: f64,
    /// Enable breach notifications
    pub enable_breach_notifications: bool,
    /// SLI data retention period in hours
    pub sli_retention_hours: u64,
    /// Maximum number of SLOs to track
    pub max_slos: usize,
}

impl Default for SloConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            evaluation_interval_seconds: 60,
            window_duration_seconds: 30 * 24 * 3600, // 30 days
            enable_error_budget_alerts: true,
            error_budget_alert_threshold: 0.1, // Alert when 10% error budget remains
            enable_breach_notifications: true,
            sli_retention_hours: 30 * 24, // 30 days
            max_slos: 100,
        }
    }
}

/// Service Level Indicator (SLI) definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliDefinition {
    /// SLI unique identifier
    pub id: String,
    /// SLI name
    pub name: String,
    /// SLI description
    pub description: String,
    /// SLI type
    pub sli_type: SliType,
    /// SLI measurement configuration
    pub measurement: SliMeasurement,
    /// Labels for filtering
    pub labels: HashMap<String, String>,
}

/// Service Level Indicator types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SliType {
    /// Availability SLI (percentage of successful requests)
    Availability,
    /// Latency SLI (percentage of requests under threshold)
    Latency { threshold_ms: f64 },
    /// Throughput SLI (requests per second)
    Throughput { target_rps: f64 },
    /// Error rate SLI (percentage of failed requests)
    ErrorRate,
    /// Quality SLI (custom quality metric)
    Quality { metric_name: String },
    /// Custom SLI with formula
    Custom { formula: String },
}

/// SLI measurement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliMeasurement {
    /// Data source for measurements
    pub data_source: DataSource,
    /// Query to retrieve SLI data
    pub query: String,
    /// Measurement frequency in seconds
    pub frequency_seconds: u64,
    /// Aggregation method
    pub aggregation: AggregationMethod,
}

/// Data sources for SLI measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSource {
    /// Prometheus metrics
    Prometheus,
    /// Custom metrics collector
    CustomMetrics,
    /// Direct application metrics
    Application,
    /// External monitoring system
    External { endpoint: String },
}

/// Aggregation methods for SLI data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationMethod {
    Average,
    Sum,
    Count,
    Rate,
    Percentile { percentile: f64 },
    Max,
    Min,
}

/// Service Level Objective (SLO) definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SloDefinition {
    /// SLO unique identifier
    pub id: String,
    /// SLO name
    pub name: String,
    /// SLO description
    pub description: String,
    /// Associated SLI
    pub sli_id: String,
    /// Target value (e.g., 99.9 for 99.9% availability)
    pub target: f64,
    /// SLO window configuration
    pub window: SloWindow,
    /// Error budget configuration
    pub error_budget: ErrorBudgetConfig,
    /// Alert configuration
    pub alerts: Vec<SloAlert>,
    /// SLO criticality level
    pub criticality: SloCriticality,
}

/// SLO window configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SloWindow {
    /// Window type
    pub window_type: WindowType,
    /// Window duration
    pub duration: Duration,
    /// Rolling window configuration
    pub rolling_config: Option<RollingConfig>,
}

/// SLO window types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowType {
    /// Fixed calendar window (e.g., monthly)
    Calendar,
    /// Rolling window (e.g., last 30 days)
    Rolling,
    /// Custom window with specific boundaries
    Custom { start: SystemTime, end: SystemTime },
}

/// Rolling window configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollingConfig {
    /// Rolling step size
    pub step_size: Duration,
    /// Number of data points to keep
    pub max_data_points: usize,
}

/// Error budget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBudgetConfig {
    /// Error budget calculation method
    pub calculation_method: ErrorBudgetMethod,
    /// Alert thresholds for error budget
    pub alert_thresholds: Vec<f64>, // e.g., [0.5, 0.2, 0.1] for 50%, 20%, 10%
    /// Error budget burn rate alerts
    pub burn_rate_alerts: BurnRateConfig,
}

/// Error budget calculation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorBudgetMethod {
    /// Simple calculation: (1 - target) * total_events
    Simple,
    /// Weighted calculation with time-based weights
    Weighted { weights: Vec<f64> },
    /// Advanced calculation with custom formula
    Custom { formula: String },
}

/// Burn rate alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurnRateConfig {
    /// Enable burn rate alerts
    pub enabled: bool,
    /// Burn rate thresholds (e.g., 5x, 10x normal rate)
    pub thresholds: Vec<f64>,
    /// Look-back windows for burn rate calculation
    pub lookback_windows: Vec<Duration>,
}

/// SLO alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SloAlert {
    /// Alert unique identifier
    pub id: String,
    /// Alert name
    pub name: String,
    /// Alert condition
    pub condition: AlertCondition,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Notification channels
    pub channels: Vec<String>,
    /// Alert cooldown period
    pub cooldown: Duration,
}

/// SLO alert conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    /// SLO breach (SLI below target)
    SloBreach,
    /// Error budget depletion
    ErrorBudgetDepletion { threshold: f64 },
    /// High burn rate
    BurnRate { rate: f64, window: Duration },
    /// SLI trend declining
    TrendDecline { duration: Duration, threshold: f64 },
    /// Custom condition
    Custom { expression: String },
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// SLO criticality levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SloCriticality {
    /// Tier 1 - Critical service (99.99%+)
    Tier1,
    /// Tier 2 - Important service (99.9%+)
    Tier2,
    /// Tier 3 - Standard service (99%+)
    Tier3,
    /// Tier 4 - Best effort service
    Tier4,
}

/// SLI data point
#[derive(Debug, Clone)]
pub struct SliDataPoint {
    /// Timestamp
    pub timestamp: SystemTime,
    /// SLI value
    pub value: f64,
    /// Number of events (for rate calculations)
    pub event_count: u64,
    /// Success count (for availability calculations)
    pub success_count: u64,
    /// Labels associated with this data point
    pub labels: HashMap<String, String>,
}

/// SLO performance data
#[derive(Debug, Clone, Serialize)]
pub struct SloPerformance {
    /// SLO ID
    pub slo_id: String,
    /// Current SLI value
    pub current_sli: f64,
    /// SLO target
    pub target: f64,
    /// SLO compliance (percentage)
    pub compliance: f64,
    /// Error budget remaining (0.0-1.0)
    pub error_budget_remaining: f64,
    /// Error budget consumed
    pub error_budget_consumed: f64,
    /// Current burn rate
    pub burn_rate: f64,
    /// Time to budget exhaustion
    pub time_to_exhaustion: Option<Duration>,
    /// Recent trend
    pub trend: TrendDirection,
    /// Last evaluation time
    pub last_evaluation: SystemTime,
}

/// Trend direction for SLO performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Unknown,
}

/// SLO breach event
#[derive(Debug, Clone, Serialize)]
pub struct SloBreachEvent {
    /// Event ID
    pub id: String,
    /// SLO ID that was breached
    pub slo_id: String,
    /// Breach start time
    pub start_time: SystemTime,
    /// Breach end time (None if ongoing)
    pub end_time: Option<SystemTime>,
    /// Duration of breach
    pub duration: Duration,
    /// Minimum SLI value during breach
    pub min_sli_value: f64,
    /// Impact assessment
    pub impact: BreachImpact,
    /// Breach severity
    pub severity: AlertSeverity,
}

/// Breach impact assessment
#[derive(Debug, Clone, Serialize)]
pub struct BreachImpact {
    /// Affected users (estimated)
    pub affected_users: u64,
    /// Revenue impact (estimated)
    pub revenue_impact: f64,
    /// Reputation impact score
    pub reputation_impact: f64,
    /// Error budget consumed by this breach
    pub error_budget_consumed: f64,
}

/// Main SLO tracking service
#[derive(Clone)]
pub struct SloTracker {
    /// Configuration
    config: SloConfig,
    /// SLI definitions
    sli_definitions: Arc<RwLock<HashMap<String, SliDefinition>>>,
    /// SLO definitions
    slo_definitions: Arc<RwLock<HashMap<String, SloDefinition>>>,
    /// SLI data storage
    sli_data: Arc<RwLock<HashMap<String, VecDeque<SliDataPoint>>>>,
    /// SLO performance cache
    slo_performance: Arc<RwLock<HashMap<String, SloPerformance>>>,
    /// Active breaches
    active_breaches: Arc<RwLock<HashMap<String, SloBreachEvent>>>,
    /// Breach history
    breach_history: Arc<RwLock<Vec<SloBreachEvent>>>,
    /// Prometheus metrics
    prometheus_metrics: Arc<SloPrometheusMetrics>,
    /// Statistics
    stats: Arc<SloStats>,
}

/// Prometheus metrics for SLO tracking
struct SloPrometheusMetrics {
    /// SLI value gauge
    sli_value: Gauge,
    /// SLO compliance gauge
    slo_compliance: Gauge,
    /// Error budget remaining gauge
    error_budget_remaining: Gauge,
    /// Burn rate gauge
    burn_rate: Gauge,
    /// SLO breach counter
    slo_breaches: Counter,
    /// SLI evaluation duration
    sli_evaluation_duration: Histogram,
}

/// SLO tracking statistics
#[derive(Debug, Default)]
pub struct SloStats {
    /// Total SLIs tracked
    pub total_slis: AtomicU64,
    /// Total SLOs tracked
    pub total_slos: AtomicU64,
    /// Total evaluations performed
    pub total_evaluations: AtomicU64,
    /// Total breaches detected
    pub total_breaches: AtomicU64,
    /// Active breaches count
    pub active_breaches: AtomicU64,
    /// Error budget alerts sent
    pub error_budget_alerts: AtomicU64,
}

impl SloTracker {
    /// Create a new SLO tracker
    pub fn new(config: SloConfig) -> Result<Self> {
        Ok(Self {
            config,
            sli_definitions: Arc::new(RwLock::new(HashMap::new())),
            slo_definitions: Arc::new(RwLock::new(HashMap::new())),
            sli_data: Arc::new(RwLock::new(HashMap::new())),
            slo_performance: Arc::new(RwLock::new(HashMap::new())),
            active_breaches: Arc::new(RwLock::new(HashMap::new())),
            breach_history: Arc::new(RwLock::new(Vec::new())),
            prometheus_metrics: Arc::new(SloPrometheusMetrics::new()?),
            stats: Arc::new(SloStats::default()),
        })
    }

    /// Start the SLO tracking service
    pub async fn start(&self) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Start SLI evaluation task
        self.start_sli_evaluation_task().await?;

        // Start SLO evaluation task
        self.start_slo_evaluation_task().await?;

        // Start breach detection task
        self.start_breach_detection_task().await?;

        // Start error budget monitoring task
        self.start_error_budget_monitoring_task().await?;

        // Start cleanup task
        self.start_cleanup_task().await?;

        Ok(())
    }

    /// Register a new SLI
    pub async fn register_sli(&self, sli: SliDefinition) -> Result<()> {
        let mut definitions = self.sli_definitions.write().await;

        if definitions.len() >= self.config.max_slos {
            return Err(anyhow::anyhow!("Maximum number of SLIs reached"));
        }

        definitions.insert(sli.id.clone(), sli);
        self.stats.total_slis.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Register a new SLO
    pub async fn register_slo(&self, slo: SloDefinition) -> Result<()> {
        // Validate that SLI exists
        let sli_definitions = self.sli_definitions.read().await;
        if !sli_definitions.contains_key(&slo.sli_id) {
            return Err(anyhow::anyhow!("SLI {} not found", slo.sli_id));
        }
        drop(sli_definitions);

        let mut definitions = self.slo_definitions.write().await;

        if definitions.len() >= self.config.max_slos {
            return Err(anyhow::anyhow!("Maximum number of SLOs reached"));
        }

        definitions.insert(slo.id.clone(), slo);
        self.stats.total_slos.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Record SLI data point
    pub async fn record_sli_data(&self, sli_id: &str, data_point: SliDataPoint) -> Result<()> {
        let mut data = self.sli_data.write().await;
        let sli_series = data.entry(sli_id.to_string()).or_insert_with(VecDeque::new);

        sli_series.push_back(data_point.clone());

        // Limit series size based on retention
        let max_points = (self.config.sli_retention_hours * 3600) / 60; // Assuming 1 minute granularity
        while sli_series.len() > max_points as usize {
            sli_series.pop_front();
        }

        // Update Prometheus metrics
        self.prometheus_metrics.sli_value.set(data_point.value);

        Ok(())
    }

    /// Evaluate SLO performance
    pub async fn evaluate_slo(&self, slo_id: &str) -> Result<SloPerformance> {
        let start = Instant::now();

        let slo_definitions = self.slo_definitions.read().await;
        let slo = slo_definitions
            .get(slo_id)
            .ok_or_else(|| anyhow::anyhow!("SLO {} not found", slo_id))?
            .clone();
        drop(slo_definitions);

        // Get SLI data for the SLO window
        let sli_data = self.get_sli_data_for_window(&slo.sli_id, &slo.window).await?;

        // Calculate current SLI value
        let current_sli = self.calculate_current_sli(&sli_data, &slo).await?;

        // Calculate compliance
        let compliance = if current_sli >= slo.target {
            100.0
        } else {
            (current_sli / slo.target) * 100.0
        };

        // Calculate error budget
        let (error_budget_remaining, error_budget_consumed) =
            self.calculate_error_budget(&sli_data, &slo).await?;

        // Calculate burn rate
        let burn_rate = self.calculate_burn_rate(&sli_data, &slo).await?;

        // Calculate time to exhaustion
        let time_to_exhaustion = if burn_rate > 0.0 && error_budget_remaining > 0.0 {
            Some(Duration::from_secs(
                (error_budget_remaining / burn_rate) as u64,
            ))
        } else {
            None
        };

        // Determine trend
        let trend = self.calculate_trend(&sli_data).await?;

        let performance = SloPerformance {
            slo_id: slo_id.to_string(),
            current_sli,
            target: slo.target,
            compliance,
            error_budget_remaining,
            error_budget_consumed,
            burn_rate,
            time_to_exhaustion,
            trend,
            last_evaluation: SystemTime::now(),
        };

        // Update performance cache
        self.slo_performance
            .write()
            .await
            .insert(slo_id.to_string(), performance.clone());

        // Update Prometheus metrics
        self.prometheus_metrics.slo_compliance.set(compliance);
        self.prometheus_metrics.error_budget_remaining.set(error_budget_remaining);
        self.prometheus_metrics.burn_rate.set(burn_rate);

        // Record evaluation duration
        let duration = start.elapsed().as_secs_f64();
        self.prometheus_metrics.sli_evaluation_duration.observe(duration);

        self.stats.total_evaluations.fetch_add(1, Ordering::Relaxed);

        Ok(performance)
    }

    /// Get SLO performance for all tracked SLOs
    pub async fn get_all_slo_performance(&self) -> HashMap<String, SloPerformance> {
        self.slo_performance.read().await.clone()
    }

    /// Get current breaches
    pub async fn get_active_breaches(&self) -> Vec<SloBreachEvent> {
        self.active_breaches.read().await.values().cloned().collect()
    }

    /// Get breach history
    pub async fn get_breach_history(&self, limit: Option<usize>) -> Vec<SloBreachEvent> {
        let history = self.breach_history.read().await;
        if let Some(limit) = limit {
            history.iter().rev().take(limit).cloned().collect()
        } else {
            history.clone()
        }
    }

    /// Get SLO tracking statistics
    pub async fn get_stats(&self) -> SloStats {
        SloStats {
            total_slis: AtomicU64::new(self.stats.total_slis.load(Ordering::Relaxed)),
            total_slos: AtomicU64::new(self.stats.total_slos.load(Ordering::Relaxed)),
            total_evaluations: AtomicU64::new(self.stats.total_evaluations.load(Ordering::Relaxed)),
            total_breaches: AtomicU64::new(self.stats.total_breaches.load(Ordering::Relaxed)),
            active_breaches: AtomicU64::new(self.stats.active_breaches.load(Ordering::Relaxed)),
            error_budget_alerts: AtomicU64::new(
                self.stats.error_budget_alerts.load(Ordering::Relaxed),
            ),
        }
    }

    // Private helper methods

    async fn start_sli_evaluation_task(&self) -> Result<()> {
        let tracker = self.clone();
        let interval = Duration::from_secs(tracker.config.evaluation_interval_seconds);

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                if let Err(e) = tracker.evaluate_all_slis().await {
                    eprintln!("SLI evaluation failed: {}", e);
                }
            }
        });

        Ok(())
    }

    async fn start_slo_evaluation_task(&self) -> Result<()> {
        let tracker = self.clone();
        let interval = Duration::from_secs(tracker.config.evaluation_interval_seconds);

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                if let Err(e) = tracker.evaluate_all_slos().await {
                    eprintln!("SLO evaluation failed: {}", e);
                }
            }
        });

        Ok(())
    }

    async fn start_breach_detection_task(&self) -> Result<()> {
        let tracker = self.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));

            loop {
                interval.tick().await;

                if let Err(e) = tracker.detect_breaches().await {
                    eprintln!("Breach detection failed: {}", e);
                }
            }
        });

        Ok(())
    }

    async fn start_error_budget_monitoring_task(&self) -> Result<()> {
        let tracker = self.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300)); // 5 minutes

            loop {
                interval.tick().await;

                if let Err(e) = tracker.monitor_error_budgets().await {
                    eprintln!("Error budget monitoring failed: {}", e);
                }
            }
        });

        Ok(())
    }

    async fn start_cleanup_task(&self) -> Result<()> {
        let tracker = self.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(3600)); // 1 hour

            loop {
                interval.tick().await;

                if let Err(e) = tracker.cleanup_old_data().await {
                    eprintln!("SLO data cleanup failed: {}", e);
                }
            }
        });

        Ok(())
    }

    async fn evaluate_all_slis(&self) -> Result<()> {
        let definitions = self.sli_definitions.read().await.clone();

        for (sli_id, _) in definitions {
            if let Err(e) = self.evaluate_sli(&sli_id).await {
                eprintln!("Failed to evaluate SLI {}: {}", sli_id, e);
            }
        }

        Ok(())
    }

    async fn evaluate_all_slos(&self) -> Result<()> {
        let definitions = self.slo_definitions.read().await.clone();

        for (slo_id, _) in definitions {
            if let Err(e) = self.evaluate_slo(&slo_id).await {
                eprintln!("Failed to evaluate SLO {}: {}", slo_id, e);
            }
        }

        Ok(())
    }

    async fn evaluate_sli(&self, sli_id: &str) -> Result<()> {
        // Simplified SLI evaluation - in practice would query data sources
        let dummy_data_point = SliDataPoint {
            timestamp: SystemTime::now(),
            value: 99.5, // Example: 99.5% availability
            event_count: 1000,
            success_count: 995,
            labels: HashMap::new(),
        };

        self.record_sli_data(sli_id, dummy_data_point).await?;
        Ok(())
    }

    async fn get_sli_data_for_window(
        &self,
        sli_id: &str,
        window: &SloWindow,
    ) -> Result<Vec<SliDataPoint>> {
        let data = self.sli_data.read().await;
        let sli_series = data.get(sli_id).cloned().unwrap_or_default();

        // Filter data based on window
        let cutoff_time = SystemTime::now() - window.duration;
        let filtered_data: Vec<_> =
            sli_series.into_iter().filter(|point| point.timestamp > cutoff_time).collect();

        Ok(filtered_data)
    }

    async fn calculate_current_sli(
        &self,
        data: &[SliDataPoint],
        _slo: &SloDefinition,
    ) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }

        // Simple average calculation - in practice would depend on SLI type
        let sum: f64 = data.iter().map(|point| point.value).sum();
        Ok(sum / data.len() as f64)
    }

    async fn calculate_error_budget(
        &self,
        data: &[SliDataPoint],
        slo: &SloDefinition,
    ) -> Result<(f64, f64)> {
        if data.is_empty() {
            return Ok((100.0, 0.0));
        }

        let total_budget = 100.0 - slo.target;
        let total_events: u64 = data.iter().map(|point| point.event_count).sum();
        let total_errors: u64 =
            data.iter().map(|point| point.event_count - point.success_count).sum();

        let error_rate = if total_events > 0 {
            (total_errors as f64 / total_events as f64) * 100.0
        } else {
            0.0
        };

        let consumed = (error_rate / total_budget) * 100.0;
        let remaining = 100.0 - consumed.min(100.0);

        Ok((remaining.max(0.0), consumed.min(100.0)))
    }

    async fn calculate_burn_rate(
        &self,
        data: &[SliDataPoint],
        _slo: &SloDefinition,
    ) -> Result<f64> {
        if data.len() < 2 {
            return Ok(0.0);
        }

        // Simplified burn rate calculation
        let recent_data = &data[data.len().saturating_sub(10)..];
        let avg_error_rate: f64 = recent_data
            .iter()
            .map(|point| {
                if point.event_count > 0 {
                    ((point.event_count - point.success_count) as f64 / point.event_count as f64)
                        * 100.0
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / recent_data.len() as f64;

        Ok(avg_error_rate)
    }

    async fn calculate_trend(&self, data: &[SliDataPoint]) -> Result<TrendDirection> {
        if data.len() < 3 {
            return Ok(TrendDirection::Unknown);
        }

        let mid_point = data.len() / 2;
        let first_half: f64 =
            data[..mid_point].iter().map(|p| p.value).sum::<f64>() / mid_point as f64;
        let second_half: f64 = data[mid_point..].iter().map(|p| p.value).sum::<f64>()
            / (data.len() - mid_point) as f64;

        let diff = second_half - first_half;
        if diff > 0.1 {
            Ok(TrendDirection::Improving)
        } else if diff < -0.1 {
            Ok(TrendDirection::Degrading)
        } else {
            Ok(TrendDirection::Stable)
        }
    }

    async fn detect_breaches(&self) -> Result<()> {
        let performance_map = self.slo_performance.read().await.clone();
        let mut active_breaches = self.active_breaches.write().await;

        for (slo_id, performance) in performance_map {
            let is_breaching = performance.current_sli < performance.target;

            if is_breaching && !active_breaches.contains_key(&slo_id) {
                // New breach detected
                let breach = SloBreachEvent {
                    id: Uuid::new_v4().to_string(),
                    slo_id: slo_id.clone(),
                    start_time: SystemTime::now(),
                    end_time: None,
                    duration: Duration::from_secs(0),
                    min_sli_value: performance.current_sli,
                    impact: BreachImpact {
                        affected_users: 1000, // Estimated
                        revenue_impact: 0.0,
                        reputation_impact: 0.5,
                        error_budget_consumed: performance.error_budget_consumed,
                    },
                    severity: AlertSeverity::High,
                };

                active_breaches.insert(slo_id, breach);
                self.stats.total_breaches.fetch_add(1, Ordering::Relaxed);
                self.stats.active_breaches.fetch_add(1, Ordering::Relaxed);
                self.prometheus_metrics.slo_breaches.inc();
            } else if !is_breaching && active_breaches.contains_key(&slo_id) {
                // Breach resolved
                if let Some(mut breach) = active_breaches.remove(&slo_id) {
                    breach.end_time = Some(SystemTime::now());
                    breach.duration = breach
                        .end_time
                        .unwrap()
                        .duration_since(breach.start_time)
                        .unwrap_or(Duration::from_secs(0));

                    self.breach_history.write().await.push(breach);
                    self.stats.active_breaches.fetch_sub(1, Ordering::Relaxed);
                }
            }
        }

        Ok(())
    }

    async fn monitor_error_budgets(&self) -> Result<()> {
        if !self.config.enable_error_budget_alerts {
            return Ok(());
        }

        let performance_map = self.slo_performance.read().await.clone();

        for (slo_id, performance) in performance_map {
            if performance.error_budget_remaining < self.config.error_budget_alert_threshold * 100.0
            {
                // Send error budget alert
                println!(
                    "ERROR BUDGET ALERT: SLO {} has {:.1}% error budget remaining",
                    slo_id, performance.error_budget_remaining
                );

                self.stats.error_budget_alerts.fetch_add(1, Ordering::Relaxed);
            }
        }

        Ok(())
    }

    async fn cleanup_old_data(&self) -> Result<()> {
        let cutoff_time =
            SystemTime::now() - Duration::from_secs(self.config.sli_retention_hours * 3600);
        let mut data = self.sli_data.write().await;

        for (_, sli_series) in data.iter_mut() {
            sli_series.retain(|point| point.timestamp > cutoff_time);
        }

        // Clean up old breach history
        let mut history = self.breach_history.write().await;
        history.retain(|breach| breach.start_time > cutoff_time);

        Ok(())
    }
}

impl SloPrometheusMetrics {
    fn new() -> Result<Self> {
        Ok(Self {
            sli_value: register_gauge_vec!(
                "slo_sli_value",
                "Current SLI value",
                &["sli_id", "sli_name"]
            )?
            .with_label_values(&["", ""]),

            slo_compliance: register_gauge_vec!(
                "slo_compliance_percentage",
                "SLO compliance percentage",
                &["slo_id", "slo_name"]
            )?
            .with_label_values(&["", ""]),

            error_budget_remaining: register_gauge_vec!(
                "slo_error_budget_remaining",
                "Error budget remaining percentage",
                &["slo_id", "slo_name"]
            )?
            .with_label_values(&["", ""]),

            burn_rate: register_gauge_vec!(
                "slo_burn_rate",
                "Current error budget burn rate",
                &["slo_id", "slo_name"]
            )?
            .with_label_values(&["", ""]),

            slo_breaches: register_counter_vec!(
                "slo_breaches_total",
                "Total number of SLO breaches",
                &["slo_id", "slo_name", "severity"]
            )?
            .with_label_values(&["", "", ""]),

            sli_evaluation_duration: register_histogram_vec!(
                "slo_sli_evaluation_duration_seconds",
                "SLI evaluation duration in seconds",
                &["sli_id"]
            )?
            .with_label_values(&[""]),
        })
    }
}

/// SLO tracking error types
#[derive(Debug, thiserror::Error)]
pub enum SloError {
    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },

    #[error("SLI not found: {sli_id}")]
    SliNotFound { sli_id: String },

    #[error("SLO not found: {slo_id}")]
    SloNotFound { slo_id: String },

    #[error("Data source error: {message}")]
    DataSourceError { message: String },

    #[error("Calculation error: {message}")]
    CalculationError { message: String },

    #[error("Alert error: {message}")]
    AlertError { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_slo_tracker_creation() {
        let config = SloConfig::default();
        let tracker = SloTracker::new(config).unwrap();
        assert!(tracker.config.enabled);
    }

    #[tokio::test]
    async fn test_sli_registration() {
        let config = SloConfig::default();
        let tracker = SloTracker::new(config).unwrap();

        let sli = SliDefinition {
            id: "test_sli".to_string(),
            name: "Test SLI".to_string(),
            description: "Test availability SLI".to_string(),
            sli_type: SliType::Availability,
            measurement: SliMeasurement {
                data_source: DataSource::Prometheus,
                query: "rate(http_requests_total[5m])".to_string(),
                frequency_seconds: 60,
                aggregation: AggregationMethod::Average,
            },
            labels: HashMap::new(),
        };

        let result = tracker.register_sli(sli).await;
        assert!(result.is_ok());

        let stats = tracker.get_stats().await;
        assert_eq!(stats.total_slis.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_slo_registration() {
        let config = SloConfig::default();
        let tracker = SloTracker::new(config).unwrap();

        // First register SLI
        let sli = SliDefinition {
            id: "test_sli".to_string(),
            name: "Test SLI".to_string(),
            description: "Test SLI".to_string(),
            sli_type: SliType::Availability,
            measurement: SliMeasurement {
                data_source: DataSource::Prometheus,
                query: "test".to_string(),
                frequency_seconds: 60,
                aggregation: AggregationMethod::Average,
            },
            labels: HashMap::new(),
        };
        tracker.register_sli(sli).await.unwrap();

        // Then register SLO
        let slo = SloDefinition {
            id: "test_slo".to_string(),
            name: "Test SLO".to_string(),
            description: "Test availability SLO".to_string(),
            sli_id: "test_sli".to_string(),
            target: 99.9,
            window: SloWindow {
                window_type: WindowType::Rolling,
                duration: Duration::from_secs(30 * 24 * 3600),
                rolling_config: None,
            },
            error_budget: ErrorBudgetConfig {
                calculation_method: ErrorBudgetMethod::Simple,
                alert_thresholds: vec![0.5, 0.2, 0.1],
                burn_rate_alerts: BurnRateConfig {
                    enabled: true,
                    thresholds: vec![5.0, 10.0],
                    lookback_windows: vec![Duration::from_secs(300), Duration::from_secs(900)],
                },
            },
            alerts: Vec::new(),
            criticality: SloCriticality::Tier2,
        };

        let result = tracker.register_slo(slo).await;
        assert!(result.is_ok());

        let stats = tracker.get_stats().await;
        assert_eq!(stats.total_slos.load(Ordering::Relaxed), 1);
    }
}
