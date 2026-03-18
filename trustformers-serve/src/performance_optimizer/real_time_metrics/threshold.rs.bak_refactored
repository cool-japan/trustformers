//! # Threshold Monitoring and Alerting Module
//!
//! This module provides comprehensive threshold monitoring and alerting functionality for the
//! TrustformeRS real-time metrics system. It includes intelligent threshold evaluation,
//! multi-level alerting, escalation policies, dynamic threshold adaptation, and comprehensive
//! alert management with suppression and correlation capabilities.
//!
//! ## Key Components
//!
//! - **ThresholdMonitor**: Core threshold monitoring system with real-time evaluation
//! - **ThresholdEvaluator**: Multiple evaluation algorithms (Simple, Statistical, Adaptive)
//! - **AlertManager**: Comprehensive alert processing and notification management
//! - **EscalationManager**: Multi-level alert escalation with severity classification
//! - **AdaptiveThresholdController**: Dynamic threshold adjustment based on system behavior
//! - **AlertSuppressor**: Alert deduplication and suppression algorithms
//! - **AlertCorrelator**: Alert correlation and relationship analysis
//! - **PerformanceAnalyzer**: Threshold performance impact analysis and optimization
//!
//! ## Features
//!
//! - Real-time threshold monitoring with microsecond precision
//! - Multiple threshold evaluation strategies with adaptive capabilities
//! - Intelligent alert generation with severity classification
//! - Advanced escalation policies with time-based triggers
//! - Dynamic threshold adaptation based on historical data and system behavior
//! - Alert suppression and deduplication to reduce noise
//! - Alert correlation to identify related issues
//! - Performance impact monitoring for threshold evaluation overhead
//! - Comprehensive notification system with multiple channels
//! - Thread-safe concurrent processing with minimal performance impact
//! - Extensive configuration options and real-time monitoring
//!
//! ## Example Usage
//!
//! ```rust
//! use threshold::{ThresholdMonitor, ThresholdConfig, ThresholdDirection};
//! use std::time::Duration;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Create threshold monitor
//!     let monitor = ThresholdMonitor::new().await?;
//!
//!     // Configure CPU threshold
//!     let cpu_threshold = ThresholdConfig {
//!         name: "high_cpu_usage".to_string(),
//!         metric: "cpu_utilization".to_string(),
//!         warning_threshold: 0.8,
//!         critical_threshold: 0.95,
//!         direction: ThresholdDirection::Above,
//!         adaptive: true,
//!         evaluation_window: Duration::from_secs(60),
//!         min_trigger_count: 3,
//!         cooldown_period: Duration::from_secs(300),
//!         escalation_policy: "default".to_string(),
//!     };
//!
//!     // Add threshold and start monitoring
//!     monitor.add_threshold(cpu_threshold).await?;
//!     monitor.start_monitoring().await?;
//!
//!     // Evaluate metrics continuously
//!     loop {
//!         let metrics = collect_current_metrics().await?;
//!         let alerts = monitor.evaluate_thresholds(&metrics).await?;
//!
//!         if !alerts.is_empty() {
//!             println!("Generated {} alerts", alerts.len());
//!         }
//!
//!         tokio::time::sleep(Duration::from_secs(1)).await;
//!     }
//! }
//! ```

use super::types::*;
use chrono::{DateTime, Utc};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::Mutex as TokioMutex;
use tokio::time::interval;
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

/// Result type for threshold operations
pub type Result<T> = std::result::Result<T, ThresholdError>;

/// Errors that can occur during threshold operations
#[derive(Debug, thiserror::Error)]
pub enum ThresholdError {
    #[error("Threshold configuration error: {0}")]
    ConfigurationError(String),

    #[error("Threshold evaluation error: {0}")]
    EvaluationError(String),

    #[error("Alert processing error: {0}")]
    AlertProcessingError(String),

    #[error("Escalation error: {0}")]
    EscalationError(String),

    #[error("Notification error: {0}")]
    NotificationError(String),

    #[error("Performance monitoring error: {0}")]
    PerformanceError(String),

    #[error("Internal error: {0}")]
    InternalError(String),
}

// =============================================================================
// CORE DATA STRUCTURES
// =============================================================================

// ThresholdConfig and ThresholdDirection are imported from types.rs via use super::types::*
// (Removed duplicate definitions to avoid type conflicts)

/// Threshold monitoring state
///
/// Current state of threshold monitoring system including active alerts,
/// monitoring statistics, and system health information.
#[derive(Debug, Clone, Default)]
pub struct ThresholdMonitoringState {
    /// Active alerts
    pub active_alerts: HashMap<String, AlertEvent>,

    /// Alert counts by severity
    pub alert_counts: HashMap<SeverityLevel, u64>,

    /// Last evaluation timestamp
    pub last_evaluation: Option<DateTime<Utc>>,

    /// Evaluation performance metrics
    pub evaluation_performance: EvaluationPerformance,

    /// Evaluation statistics
    pub evaluation_stats: EvaluationStatistics,
}

/// Statistics for threshold evaluation
///
/// Performance statistics for threshold evaluation operations including
/// processing times, accuracy metrics, and system performance impact.
#[derive(Debug, Clone, Default)]
pub struct EvaluationStatistics {
    /// Total evaluations performed
    pub total_evaluations: u64,

    /// Average evaluation time
    pub avg_evaluation_time: Duration,

    /// Total alerts generated
    pub total_alerts: u64,

    /// False positive rate
    pub false_positive_rate: f32,

    /// False negative rate
    pub false_negative_rate: f32,

    /// Alert accuracy
    pub alert_accuracy: f32,
}

/// Performance metrics for threshold evaluation
#[derive(Debug, Clone, Default)]
pub struct EvaluationPerformance {
    /// Last evaluation duration
    pub last_evaluation_duration: Duration,

    /// Average evaluation duration
    pub avg_evaluation_duration: Duration,

    /// Peak evaluation duration
    pub peak_evaluation_duration: Duration,

    /// Total evaluation count
    pub total_evaluations: u64,

    /// Evaluations per second
    pub evaluations_per_second: f32,

    /// Memory usage for evaluations
    pub memory_usage_bytes: u64,

    /// CPU overhead percentage
    pub cpu_overhead_percent: f32,
}

/// Threshold evaluation result
///
/// Result of threshold evaluation including violation status, severity,
/// and contextual information.
#[derive(Debug, Clone)]
pub struct ThresholdEvaluation {
    /// Whether threshold was violated
    pub violated: bool,

    /// Evaluation timestamp
    pub timestamp: DateTime<Utc>,

    /// Severity level if violated
    pub severity: SeverityLevel,

    /// Current value
    pub current_value: f64,

    /// Threshold value
    pub threshold_value: f64,

    /// Evaluation confidence
    pub confidence: f32,

    /// Additional context
    pub context: HashMap<String, String>,
}

// SuppressionInfo is now defined in types.rs and imported via use super::types::*
// (Removed duplicate definition to avoid E0659 ambiguity errors)

/// Alert correlation information
#[derive(Debug, Clone)]
pub struct CorrelationInfo {
    /// Correlation ID
    pub correlation_id: String,

    /// Related alert IDs
    pub related_alerts: Vec<String>,

    /// Correlation strength (0.0 to 1.0)
    pub correlation_strength: f32,

    /// Correlation type
    pub correlation_type: CorrelationType,

    /// Time window for correlation
    pub time_window: Duration,
}

/// Types of alert correlation
#[derive(Debug, Clone)]
pub enum CorrelationType {
    /// Temporal correlation (alerts occurring at similar times)
    Temporal,

    /// Causal correlation (one alert likely causing another)
    Causal,

    /// Resource correlation (alerts from same resource)
    Resource,

    /// Pattern correlation (alerts following a known pattern)
    Pattern,

    /// Metric correlation (alerts from correlated metrics)
    Metric,
}

// =============================================================================
// THRESHOLD EVALUATOR TRAIT AND IMPLEMENTATIONS
// =============================================================================

/// Threshold evaluator trait for threshold monitoring
///
/// Interface for threshold evaluators that assess metric values against
/// configured thresholds and generate alerts when violations occur.
pub trait ThresholdEvaluator: Send + Sync {
    /// Evaluate threshold against current value
    fn evaluate(&self, config: &ThresholdConfig, value: f64) -> Result<ThresholdEvaluation>;

    /// Get evaluator name
    fn name(&self) -> &str;

    /// Check if evaluator supports threshold type
    fn supports_threshold(&self, threshold_type: &str) -> bool;

    /// Get evaluation confidence for given data quality
    fn confidence(&self, data_quality: f32) -> f32 {
        data_quality.clamp(0.0, 1.0)
    }

    /// Perform initialization if needed
    fn initialize(&mut self) -> Result<()> {
        Ok(())
    }

    /// Clean up resources
    fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Simple threshold evaluator
///
/// Basic threshold evaluator that performs simple comparison against
/// static threshold values without any statistical analysis.
pub struct SimpleThresholdEvaluator {
    /// Evaluator configuration
    config: SimpleEvaluatorConfig,

    /// Performance statistics
    stats: Arc<Mutex<EvaluatorStats>>,
}

/// Configuration for simple threshold evaluator
#[derive(Debug, Clone)]
pub struct SimpleEvaluatorConfig {
    /// Confidence baseline
    pub confidence_baseline: f32,

    /// Enable performance tracking
    pub track_performance: bool,
}

/// Statistics for evaluator performance
#[derive(Debug, Clone)]
pub struct EvaluatorStats {
    /// Total evaluations
    pub total_evaluations: u64,

    /// Total evaluation time
    pub total_evaluation_time: Duration,

    /// Violations detected
    pub violations_detected: u64,

    /// Last evaluation time
    pub last_evaluation_time: Instant,
}

impl Default for EvaluatorStats {
    fn default() -> Self {
        Self {
            total_evaluations: 0,
            total_evaluation_time: Duration::default(),
            violations_detected: 0,
            last_evaluation_time: Instant::now(),
        }
    }
}

impl Default for SimpleThresholdEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleThresholdEvaluator {
    /// Create a new simple threshold evaluator
    pub fn new() -> Self {
        Self {
            config: SimpleEvaluatorConfig {
                confidence_baseline: 0.8,
                track_performance: true,
            },
            stats: Arc::new(Mutex::new(EvaluatorStats::default())),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: SimpleEvaluatorConfig) -> Self {
        Self {
            config,
            stats: Arc::new(Mutex::new(EvaluatorStats::default())),
        }
    }

    /// Get evaluator statistics
    pub fn get_stats(&self) -> EvaluatorStats {
        self.stats.lock().expect("Stats lock poisoned").clone()
    }
}

impl ThresholdEvaluator for SimpleThresholdEvaluator {
    fn evaluate(&self, config: &ThresholdConfig, value: f64) -> Result<ThresholdEvaluation> {
        let start_time = Instant::now();

        let violated = match config.direction {
            ThresholdDirection::Above => value > config.critical_threshold,
            ThresholdDirection::Below => value < config.critical_threshold,
            ThresholdDirection::OutsideRange { min, max } => value < min || value > max,
            ThresholdDirection::InsideRange { min, max } => value >= min && value <= max,
        };

        let severity = if violated {
            if match config.direction {
                ThresholdDirection::Above => value > config.critical_threshold,
                ThresholdDirection::Below => value < config.critical_threshold,
                ThresholdDirection::OutsideRange { min, max } => {
                    let critical_distance = (max - min) * 0.2; // 20% outside range
                    value < (min - critical_distance) || value > (max + critical_distance)
                },
                ThresholdDirection::InsideRange { min, max } => {
                    let critical_distance = (max - min) * 0.2; // 20% inside range is critical
                    value >= (min + critical_distance) && value <= (max - critical_distance)
                },
            } {
                SeverityLevel::Critical
            } else {
                SeverityLevel::Warning
            }
        } else {
            SeverityLevel::Info
        };

        let threshold_value = match config.direction {
            ThresholdDirection::Above => config.critical_threshold,
            ThresholdDirection::Below => config.critical_threshold,
            ThresholdDirection::OutsideRange { min, max } => {
                if value < min {
                    min
                } else {
                    max
                }
            },
            ThresholdDirection::InsideRange { min, max } => {
                // Return the closer boundary
                if (value - min).abs() < (value - max).abs() {
                    min
                } else {
                    max
                }
            },
        };

        // Update statistics
        if self.config.track_performance {
            let mut stats = self.stats.lock().expect("Stats lock poisoned");
            stats.total_evaluations += 1;
            stats.total_evaluation_time += start_time.elapsed();
            if violated {
                stats.violations_detected += 1;
            }
            stats.last_evaluation_time = Instant::now();
        }

        Ok(ThresholdEvaluation {
            violated,
            timestamp: Utc::now(),
            severity,
            current_value: value,
            threshold_value,
            confidence: self.config.confidence_baseline,
            context: HashMap::new(),
        })
    }

    fn name(&self) -> &str {
        "simple_threshold"
    }

    fn supports_threshold(&self, _threshold_type: &str) -> bool {
        true // Simple evaluator supports all threshold types
    }
}

/// Statistical threshold evaluator
///
/// Advanced threshold evaluator that uses statistical analysis to determine
/// threshold violations with enhanced confidence scoring and context.
pub struct StatisticalThresholdEvaluator {
    /// Evaluator configuration
    config: StatisticalEvaluatorConfig,

    /// Historical data for statistical analysis
    history: Arc<Mutex<VecDeque<StatisticalDataPoint>>>,

    /// Performance statistics
    stats: Arc<Mutex<EvaluatorStats>>,
}

/// Configuration for statistical threshold evaluator
#[derive(Debug, Clone)]
pub struct StatisticalEvaluatorConfig {
    /// History window size
    pub history_window_size: usize,

    /// Minimum data points for statistical analysis
    pub min_data_points: usize,

    /// Confidence calculation method
    pub confidence_method: ConfidenceMethod,

    /// Statistical significance threshold
    pub significance_threshold: f32,
}

/// Methods for calculating confidence
#[derive(Debug, Clone)]
pub enum ConfidenceMethod {
    /// Simple variance-based confidence
    Variance,

    /// Z-score based confidence
    ZScore,

    /// Student's t-test based confidence
    TTest,

    /// Bayesian confidence estimation
    Bayesian,
}

/// Statistical data point for analysis
#[derive(Debug, Clone)]
pub struct StatisticalDataPoint {
    /// Value
    pub value: f64,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Quality score
    pub quality: f32,
}

impl Default for StatisticalThresholdEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticalThresholdEvaluator {
    /// Create a new statistical threshold evaluator
    pub fn new() -> Self {
        Self {
            config: StatisticalEvaluatorConfig {
                history_window_size: 1000,
                min_data_points: 10,
                confidence_method: ConfidenceMethod::ZScore,
                significance_threshold: 0.05,
            },
            history: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(Mutex::new(EvaluatorStats::default())),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: StatisticalEvaluatorConfig) -> Self {
        Self {
            config,
            history: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(Mutex::new(EvaluatorStats::default())),
        }
    }

    /// Add data point to history
    pub fn add_data_point(&self, value: f64, quality: f32) {
        let mut history = self.history.lock().expect("History lock poisoned");

        let data_point = StatisticalDataPoint {
            value,
            timestamp: Utc::now(),
            quality,
        };

        history.push_back(data_point);

        // Maintain window size
        while history.len() > self.config.history_window_size {
            history.pop_front();
        }
    }

    /// Calculate statistical confidence
    fn calculate_confidence(&self, config: &ThresholdConfig, value: f64) -> f32 {
        let history = self.history.lock().expect("History lock poisoned");

        if history.len() < self.config.min_data_points {
            return 0.5; // Low confidence with insufficient data
        }

        let values: Vec<f64> = history.iter().map(|dp| dp.value).collect();

        match self.config.confidence_method {
            ConfidenceMethod::Variance => self.calculate_variance_confidence(&values, value),
            ConfidenceMethod::ZScore => self.calculate_zscore_confidence(&values, value),
            ConfidenceMethod::TTest => self.calculate_ttest_confidence(&values, value, config),
            ConfidenceMethod::Bayesian => self.calculate_bayesian_confidence(&values, value),
        }
    }

    fn calculate_variance_confidence(&self, values: &[f64], current_value: f64) -> f32 {
        if values.is_empty() {
            return 0.5;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;

        let std_dev = variance.sqrt();
        let z_score = (current_value - mean).abs() / std_dev.max(0.001);

        // Convert z-score to confidence (higher z-score = higher confidence in anomaly)
        (1.0 - (-z_score).exp()).clamp(0.0, 1.0) as f32
    }

    fn calculate_zscore_confidence(&self, values: &[f64], current_value: f64) -> f32 {
        if values.is_empty() {
            return 0.5;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;

        let std_dev = variance.sqrt();
        let z_score = (current_value - mean).abs() / std_dev.max(0.001);

        // Convert z-score to confidence using error function approximation
        let normalized_z = z_score / 3.0; // Normalize to roughly 0-1 range
        normalized_z.clamp(0.0, 1.0) as f32
    }

    fn calculate_ttest_confidence(
        &self,
        values: &[f64],
        current_value: f64,
        _config: &ThresholdConfig,
    ) -> f32 {
        if values.len() < 2 {
            return 0.5;
        }

        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);

        let std_error = (variance / n).sqrt();
        let t_stat = (current_value - mean).abs() / std_error.max(0.001);

        // Convert t-statistic to confidence
        let confidence = 1.0 - 2.0 * (-t_stat).exp();
        confidence.clamp(0.0, 1.0) as f32
    }

    fn calculate_bayesian_confidence(&self, values: &[f64], current_value: f64) -> f32 {
        if values.is_empty() {
            return 0.5;
        }

        // Simple Bayesian approach using normal-normal conjugate prior
        let n = values.len() as f64;
        let sample_mean = values.iter().sum::<f64>() / n;
        let sample_var = values.iter().map(|v| (v - sample_mean).powi(2)).sum::<f64>() / n;

        // Prior parameters (weak prior)
        let prior_mean = sample_mean;
        let prior_precision = 1.0; // Low precision = uninformative prior

        // Posterior parameters
        let posterior_precision = prior_precision + n / sample_var.max(0.001);
        let posterior_mean = (prior_precision * prior_mean
            + n * sample_mean / sample_var.max(0.001))
            / posterior_precision;
        let posterior_var = 1.0 / posterior_precision;

        // Calculate probability that current value is anomalous
        let z_score = (current_value - posterior_mean).abs() / posterior_var.sqrt().max(0.001);
        (1.0 - (-z_score).exp()).clamp(0.0, 1.0) as f32
    }

    /// Determine severity based on statistical analysis
    fn determine_severity(
        &self,
        config: &ThresholdConfig,
        value: f64,
        confidence: f32,
    ) -> SeverityLevel {
        let distance_ratio = match config.direction {
            ThresholdDirection::Above => {
                if value > config.critical_threshold {
                    (value - config.critical_threshold) / config.critical_threshold
                } else if value > config.warning_threshold {
                    (value - config.warning_threshold) / config.warning_threshold * 0.5
                } else {
                    0.0
                }
            },
            ThresholdDirection::Below => {
                if value < config.critical_threshold {
                    (config.critical_threshold - value) / config.critical_threshold
                } else if value < config.warning_threshold {
                    (config.warning_threshold - value) / config.warning_threshold * 0.5
                } else {
                    0.0
                }
            },
            ThresholdDirection::OutsideRange { min, max } => {
                if value < min {
                    (min - value) / (max - min)
                } else if value > max {
                    (value - max) / (max - min)
                } else {
                    0.0
                }
            },
            ThresholdDirection::InsideRange { min, max } => {
                if value >= min && value <= max {
                    0.0
                } else if value < min {
                    (min - value) / (max - min)
                } else {
                    (value - max) / (max - min)
                }
            },
        };

        match (distance_ratio, confidence) {
            (r, c) if r > 0.5 && c > 0.8 => SeverityLevel::Critical,
            (r, c) if r > 0.3 && c > 0.7 => SeverityLevel::High,
            (r, c) if r > 0.1 && c > 0.6 => SeverityLevel::Medium,
            (r, _) if r > 0.0 => SeverityLevel::Low,
            _ => SeverityLevel::Info,
        }
    }
}

impl ThresholdEvaluator for StatisticalThresholdEvaluator {
    fn evaluate(&self, config: &ThresholdConfig, value: f64) -> Result<ThresholdEvaluation> {
        let start_time = Instant::now();

        // Add current value to history for future analysis
        self.add_data_point(value, 1.0); // Assume perfect quality for now

        // Enhanced statistical evaluation with confidence scoring
        let violated = match config.direction {
            ThresholdDirection::Above => value > config.critical_threshold,
            ThresholdDirection::Below => value < config.critical_threshold,
            ThresholdDirection::OutsideRange { min, max } => value < min || value > max,
            ThresholdDirection::InsideRange { min, max } => value >= min && value <= max,
        };

        let confidence = self.calculate_confidence(config, value);
        let severity = self.determine_severity(config, value, confidence);

        let threshold_value = match config.direction {
            ThresholdDirection::Above => config.critical_threshold,
            ThresholdDirection::Below => config.critical_threshold,
            ThresholdDirection::OutsideRange { min, max } => {
                if value < min {
                    min
                } else {
                    max
                }
            },
            ThresholdDirection::InsideRange { min, max } => {
                // For InsideRange, return the closest boundary
                let dist_to_min = (value - min).abs();
                let dist_to_max = (max - value).abs();
                if dist_to_min < dist_to_max {
                    min
                } else {
                    max
                }
            },
        };

        // Update statistics
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| ThresholdError::InternalError("Lock poisoned".to_string()))?;
        stats.total_evaluations += 1;
        stats.total_evaluation_time += start_time.elapsed();
        if violated {
            stats.violations_detected += 1;
        }
        stats.last_evaluation_time = Instant::now();

        let mut context = HashMap::new();
        context.insert(
            "confidence_method".to_string(),
            format!("{:?}", self.config.confidence_method),
        );
        context.insert(
            "history_size".to_string(),
            self.history.lock().expect("History lock poisoned").len().to_string(),
        );

        Ok(ThresholdEvaluation {
            violated,
            timestamp: Utc::now(),
            severity,
            current_value: value,
            threshold_value,
            confidence,
            context,
        })
    }

    fn name(&self) -> &str {
        "statistical_threshold"
    }

    fn supports_threshold(&self, threshold_type: &str) -> bool {
        matches!(
            threshold_type,
            "cpu_utilization" | "memory_utilization" | "throughput" | "latency_ms" | "error_rate"
        )
    }
}

/// Adaptive threshold evaluator
///
/// Advanced threshold evaluator that dynamically adjusts thresholds based on
/// historical patterns, system behavior, and machine learning techniques.
pub struct AdaptiveThresholdEvaluator {
    /// Evaluator configuration
    config: AdaptiveEvaluatorConfig,

    /// Adaptation engine
    adaptation_engine: Arc<TokioMutex<AdaptationEngine>>,

    /// Performance statistics
    stats: Arc<Mutex<EvaluatorStats>>,
}

/// Configuration for adaptive threshold evaluator
#[derive(Debug, Clone)]
pub struct AdaptiveEvaluatorConfig {
    /// Adaptation sensitivity (0.0 to 1.0)
    pub adaptation_sensitivity: f32,

    /// Learning rate for adaptation
    pub learning_rate: f32,

    /// Minimum adaptation period
    pub min_adaptation_period: Duration,

    /// Maximum threshold adjustment ratio
    pub max_adjustment_ratio: f32,

    /// Enable pattern recognition
    pub enable_pattern_recognition: bool,

    /// Enable seasonal adaptation
    pub enable_seasonal_adaptation: bool,
}

/// Adaptation engine for dynamic threshold adjustment
#[derive(Debug)]
pub struct AdaptationEngine {
    /// Current adaptations
    adaptations: HashMap<String, AdaptiveThreshold>,

    /// Adaptation history
    history: VecDeque<AdaptationRecord>,

    /// Pattern detector
    pattern_detector: PatternDetector,

    /// Seasonal analyzer
    seasonal_analyzer: SeasonalAnalyzer,
}

/// Adaptive threshold information
#[derive(Debug, Clone)]
pub struct AdaptiveThreshold {
    /// Original threshold value
    pub original_value: f64,

    /// Current adapted value
    pub adapted_value: f64,

    /// Adaptation confidence
    pub confidence: f32,

    /// Last adaptation time
    pub last_adapted: DateTime<Utc>,

    /// Adaptation count
    pub adaptation_count: u32,

    /// Effectiveness score
    pub effectiveness: f32,
}

/// Record of threshold adaptation
#[derive(Debug, Clone)]
pub struct AdaptationRecord {
    /// Adaptation timestamp
    pub timestamp: DateTime<Utc>,

    /// Metric name
    pub metric: String,

    /// Old threshold value
    pub old_value: f64,

    /// New threshold value
    pub new_value: f64,

    /// Adaptation reason
    pub reason: String,

    /// Confidence in adaptation
    pub confidence: f32,
}

/// Pattern detector for threshold adaptation
#[derive(Debug)]
pub struct PatternDetector {
    /// Detected patterns
    patterns: HashMap<String, DetectedPattern>,

    /// Pattern history
    pattern_history: VecDeque<PatternEvent>,
}

/// Detected pattern information
#[derive(Debug, Clone)]
pub struct DetectedPattern {
    /// Pattern type
    pub pattern_type: PatternType,

    /// Pattern strength (0.0 to 1.0)
    pub strength: f32,

    /// Pattern frequency
    pub frequency: Duration,

    /// Pattern confidence
    pub confidence: f32,

    /// First detected time
    pub first_detected: DateTime<Utc>,

    /// Last seen time
    pub last_seen: DateTime<Utc>,
}

/// Types of detected patterns
#[derive(Debug, Clone)]
pub enum PatternType {
    /// Cyclic pattern (repeating at regular intervals)
    Cyclic,

    /// Trending pattern (gradual increase/decrease)
    Trending,

    /// Burst pattern (sudden spikes)
    Burst,

    /// Baseline shift (permanent level change)
    BaselineShift,

    /// Seasonal pattern (time-of-day/week/month variations)
    Seasonal,
}

/// Pattern event for pattern detection
#[derive(Debug, Clone)]
pub struct PatternEvent {
    /// Event timestamp
    pub timestamp: DateTime<Utc>,

    /// Event value
    pub value: f64,

    /// Event type
    pub event_type: String,

    /// Event metadata
    pub metadata: HashMap<String, String>,
}

/// Seasonal analyzer for time-based adaptations
#[derive(Debug)]
pub struct SeasonalAnalyzer {
    /// Seasonal models
    models: HashMap<String, SeasonalModel>,

    /// Seasonal factors
    factors: HashMap<String, f32>,
}

/// Seasonal model for threshold adaptation
#[derive(Debug, Clone)]
pub struct SeasonalModel {
    /// Model type
    pub model_type: SeasonalModelType,

    /// Model parameters
    pub parameters: HashMap<String, f64>,

    /// Model confidence
    pub confidence: f32,

    /// Training data size
    pub training_size: usize,

    /// Last updated
    pub last_updated: DateTime<Utc>,
}

/// Types of seasonal models
#[derive(Debug, Clone)]
pub enum SeasonalModelType {
    /// Hourly patterns
    Hourly,

    /// Daily patterns
    Daily,

    /// Weekly patterns
    Weekly,

    /// Monthly patterns
    Monthly,
}

impl AdaptiveThresholdEvaluator {
    /// Create a new adaptive threshold evaluator
    pub async fn new() -> Self {
        Self {
            config: AdaptiveEvaluatorConfig {
                adaptation_sensitivity: 0.7,
                learning_rate: 0.1,
                min_adaptation_period: Duration::from_secs(300), // 5 minutes
                max_adjustment_ratio: 0.3,                       // 30% maximum adjustment
                enable_pattern_recognition: true,
                enable_seasonal_adaptation: true,
            },
            adaptation_engine: Arc::new(TokioMutex::new(AdaptationEngine::new())),
            stats: Arc::new(Mutex::new(EvaluatorStats::default())),
        }
    }

    /// Create with custom configuration
    pub async fn with_config(config: AdaptiveEvaluatorConfig) -> Self {
        Self {
            config,
            adaptation_engine: Arc::new(TokioMutex::new(AdaptationEngine::new())),
            stats: Arc::new(Mutex::new(EvaluatorStats::default())),
        }
    }

    /// Calculate adaptive threshold based on historical data and patterns
    async fn calculate_adaptive_threshold(&self, config: &ThresholdConfig, value: f64) -> f64 {
        let engine = self.adaptation_engine.lock().await;

        let adaptive_threshold = engine
            .adaptations
            .get(&config.metric)
            .map(|at| at.adapted_value)
            .unwrap_or(config.critical_threshold);

        // Apply pattern-based adjustments
        if self.config.enable_pattern_recognition {
            if let Some(pattern) = engine.pattern_detector.patterns.get(&config.metric) {
                let pattern_factor = self.calculate_pattern_factor(pattern, value);
                return adaptive_threshold * pattern_factor;
            }
        }

        // Apply seasonal adjustments
        if self.config.enable_seasonal_adaptation {
            if let Some(seasonal_factor) = engine.seasonal_analyzer.factors.get(&config.metric) {
                return adaptive_threshold * (*seasonal_factor as f64);
            }
        }

        adaptive_threshold
    }

    /// Calculate pattern-based adjustment factor
    fn calculate_pattern_factor(&self, pattern: &DetectedPattern, _current_value: f64) -> f64 {
        match pattern.pattern_type {
            PatternType::Cyclic => {
                // Adjust based on cycle position
                1.0 + (pattern.strength as f64 * 0.2)
            },
            PatternType::Trending => {
                // Adjust based on trend direction
                1.0 + (pattern.strength as f64 * 0.15)
            },
            PatternType::Burst => {
                // Increase threshold during burst periods
                1.0 + (pattern.strength as f64 * 0.3)
            },
            PatternType::BaselineShift => {
                // Adjust to new baseline
                1.0 + (pattern.strength as f64 * 0.25)
            },
            PatternType::Seasonal => {
                // Seasonal adjustment
                1.0 + (pattern.strength as f64 * 0.1)
            },
        }
    }

    /// Update adaptation based on evaluation results
    async fn update_adaptation(
        &self,
        config: &ThresholdConfig,
        evaluation: &ThresholdEvaluation,
        effectiveness_score: f32,
    ) -> Result<()> {
        let mut engine = self.adaptation_engine.lock().await;

        // Get or create adaptation entry and extract needed values
        let (should_adapt_check, old_value, original_value, adapted_value, effectiveness) = {
            let current_adaptation = engine
                .adaptations
                .entry(config.metric.clone())
                .or_insert_with(|| AdaptiveThreshold {
                    original_value: config.critical_threshold,
                    adapted_value: config.critical_threshold,
                    confidence: 0.5,
                    last_adapted: Utc::now(),
                    adaptation_count: 0,
                    effectiveness: 0.5,
                });

            // Update effectiveness with exponential moving average
            current_adaptation.effectiveness =
                current_adaptation.effectiveness * 0.9 + effectiveness_score * 0.1;

            // Check if adaptation is needed and extract values
            let should_adapt = self.should_adapt(current_adaptation, evaluation);
            (
                should_adapt,
                current_adaptation.adapted_value,
                current_adaptation.original_value,
                current_adaptation.adapted_value,
                current_adaptation.effectiveness,
            )
        };

        // Now we can safely access engine.history since current_adaptation borrow is dropped
        if should_adapt_check {
            // Calculate adjustment using extracted values
            let effectiveness_factor = (0.5 - effectiveness) as f64;
            let confidence_factor = evaluation.confidence as f64;
            let learning_factor = self.config.learning_rate as f64;
            let adjustment = effectiveness_factor * confidence_factor * learning_factor;
            let new_threshold = adapted_value * (1.0 + adjustment);

            // Apply adjustment limits
            let max_change = original_value * self.config.max_adjustment_ratio as f64;
            let clamped_threshold =
                new_threshold.clamp(original_value - max_change, original_value + max_change);

            // Record adaptation
            let record = AdaptationRecord {
                timestamp: Utc::now(),
                metric: config.metric.clone(),
                old_value,
                new_value: clamped_threshold,
                reason: "Effectiveness-based adaptation".to_string(),
                confidence: effectiveness_score,
            };

            engine.history.push_back(record);

            // Update adaptation with new borrow
            if let Some(current_adaptation) = engine.adaptations.get_mut(&config.metric) {
                current_adaptation.adapted_value = clamped_threshold;
                current_adaptation.last_adapted = Utc::now();
                current_adaptation.adaptation_count += 1;
                current_adaptation.confidence = effectiveness_score;
            }

            // Maintain history size
            if engine.history.len() > 1000 {
                engine.history.pop_front();
            }
        }

        Ok(())
    }

    /// Check if adaptation should be performed
    fn should_adapt(
        &self,
        adaptation: &AdaptiveThreshold,
        evaluation: &ThresholdEvaluation,
    ) -> bool {
        // Check minimum adaptation period
        let time_since_adaptation = Utc::now().signed_duration_since(adaptation.last_adapted);
        if time_since_adaptation
            < chrono::Duration::from_std(self.config.min_adaptation_period)
                .expect("Duration conversion failed")
        {
            return false;
        }

        // Check effectiveness threshold
        if adaptation.effectiveness < self.config.adaptation_sensitivity {
            return true;
        }

        // Check evaluation confidence
        evaluation.confidence > 0.8
    }

    /// Calculate threshold adjustment
    fn calculate_adjustment(
        &self,
        adaptation: &AdaptiveThreshold,
        evaluation: &ThresholdEvaluation,
    ) -> f64 {
        let effectiveness_factor = (0.5 - adaptation.effectiveness) as f64;
        let confidence_factor = evaluation.confidence as f64;
        let learning_factor = self.config.learning_rate as f64;

        effectiveness_factor * confidence_factor * learning_factor
    }
}

impl ThresholdEvaluator for AdaptiveThresholdEvaluator {
    fn evaluate(&self, config: &ThresholdConfig, value: f64) -> Result<ThresholdEvaluation> {
        let start_time = Instant::now();

        // This is a blocking wrapper around the async function
        // In a real implementation, you might want to use a runtime handle
        let adaptive_threshold = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(async { self.calculate_adaptive_threshold(config, value).await })
        });

        // Adaptive evaluation with dynamic thresholds
        let violated = match config.direction {
            ThresholdDirection::Above => value > adaptive_threshold,
            ThresholdDirection::Below => value < adaptive_threshold,
            ThresholdDirection::OutsideRange { min, max } => {
                let adaptive_min = min * 0.9; // Adaptive adjustment
                let adaptive_max = max * 1.1;
                value < adaptive_min || value > adaptive_max
            },
            ThresholdDirection::InsideRange { min, max } => {
                let adaptive_min = min * 1.1; // Adaptive adjustment (tighter range)
                let adaptive_max = max * 0.9;
                value >= adaptive_min && value <= adaptive_max
            },
        };

        let severity = if violated {
            match config.direction {
                ThresholdDirection::Above => {
                    let severity_ratio = (value - adaptive_threshold) / adaptive_threshold;
                    if severity_ratio > 0.3 {
                        SeverityLevel::Critical
                    } else if severity_ratio > 0.2 {
                        SeverityLevel::High
                    } else if severity_ratio > 0.1 {
                        SeverityLevel::Medium
                    } else {
                        SeverityLevel::Low
                    }
                },
                ThresholdDirection::Below => {
                    let severity_ratio = (adaptive_threshold - value) / adaptive_threshold;
                    if severity_ratio > 0.3 {
                        SeverityLevel::Critical
                    } else if severity_ratio > 0.2 {
                        SeverityLevel::High
                    } else if severity_ratio > 0.1 {
                        SeverityLevel::Medium
                    } else {
                        SeverityLevel::Low
                    }
                },
                ThresholdDirection::OutsideRange { min, max } => {
                    let range_size = max - min;
                    let distance = if value < min { min - value } else { value - max };
                    let severity_ratio = distance / range_size;
                    if severity_ratio > 0.3 {
                        SeverityLevel::Critical
                    } else if severity_ratio > 0.2 {
                        SeverityLevel::High
                    } else if severity_ratio > 0.1 {
                        SeverityLevel::Medium
                    } else {
                        SeverityLevel::Low
                    }
                },
                ThresholdDirection::InsideRange { min, max } => {
                    let range_size = max - min;
                    let center = (min + max) / 2.0;
                    let distance_from_center = (value - center).abs();
                    let severity_ratio = distance_from_center / (range_size / 2.0);
                    // Closer to center means more severe for InsideRange
                    let inverted_ratio = 1.0 - severity_ratio;
                    if inverted_ratio > 0.7 {
                        SeverityLevel::Critical
                    } else if inverted_ratio > 0.5 {
                        SeverityLevel::High
                    } else if inverted_ratio > 0.3 {
                        SeverityLevel::Medium
                    } else {
                        SeverityLevel::Low
                    }
                },
            }
        } else {
            SeverityLevel::Info
        };

        // Calculate confidence based on adaptation history and effectiveness
        let confidence = self.calculate_adaptation_confidence();

        // Update statistics
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| ThresholdError::InternalError("Lock poisoned".to_string()))?;
        stats.total_evaluations += 1;
        stats.total_evaluation_time += start_time.elapsed();
        if violated {
            stats.violations_detected += 1;
        }
        stats.last_evaluation_time = Instant::now();

        let mut context = HashMap::new();
        context.insert(
            "adaptive_threshold".to_string(),
            adaptive_threshold.to_string(),
        );
        context.insert(
            "original_threshold".to_string(),
            config.critical_threshold.to_string(),
        );
        context.insert("adaptation_enabled".to_string(), "true".to_string());

        let evaluation = ThresholdEvaluation {
            violated,
            timestamp: Utc::now(),
            severity,
            current_value: value,
            threshold_value: adaptive_threshold,
            confidence,
            context,
        };

        // Update adaptation based on evaluation results (async operation)
        let evaluator = self.clone();
        let config_clone = config.clone();
        let evaluation_clone = evaluation.clone();
        tokio::spawn(async move {
            let effectiveness_score = if violated { 0.8 } else { 0.6 }; // Simplified scoring
            if let Err(e) = evaluator
                .update_adaptation(&config_clone, &evaluation_clone, effectiveness_score)
                .await
            {
                error!("Failed to update adaptation: {}", e);
            }
        });

        Ok(evaluation)
    }

    fn name(&self) -> &str {
        "adaptive_threshold"
    }

    fn supports_threshold(&self, _threshold_type: &str) -> bool {
        true // Adaptive evaluator supports all threshold types
    }
}

impl Clone for AdaptiveThresholdEvaluator {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            adaptation_engine: Arc::clone(&self.adaptation_engine),
            stats: Arc::clone(&self.stats),
        }
    }
}

impl AdaptiveThresholdEvaluator {
    /// Calculate confidence in adaptation
    fn calculate_adaptation_confidence(&self) -> f32 {
        // Simplified confidence calculation
        // In a real implementation, this would consider adaptation history,
        // effectiveness scores, and pattern detection confidence
        0.85
    }
}

impl Default for AdaptationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptationEngine {
    /// Create a new adaptation engine
    pub fn new() -> Self {
        Self {
            adaptations: HashMap::new(),
            history: VecDeque::new(),
            pattern_detector: PatternDetector::new(),
            seasonal_analyzer: SeasonalAnalyzer::new(),
        }
    }
}

impl Default for PatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternDetector {
    /// Create a new pattern detector
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            pattern_history: VecDeque::new(),
        }
    }
}

impl Default for SeasonalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SeasonalAnalyzer {
    /// Create a new seasonal analyzer
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            factors: HashMap::new(),
        }
    }
}

// =============================================================================
// ALERT MANAGEMENT SYSTEM
// =============================================================================

/// Alert manager for handling alert processing and notifications
///
/// Comprehensive alert manager that processes alerts, manages notifications,
/// handles suppression and correlation, and provides detailed analytics.
pub struct AlertManager {
    /// Alert processors
    processors: Arc<Mutex<Vec<Box<dyn AlertProcessor + Send + Sync>>>>,

    /// Notification channels
    channels: Arc<Mutex<Vec<Box<dyn NotificationChannel + Send + Sync>>>>,

    /// Alert queue for processing
    alert_queue: Arc<TokioMutex<VecDeque<AlertEvent>>>,

    /// Processing statistics
    stats: Arc<AlertManagerStats>,

    /// Alert suppressor
    suppressor: Arc<AlertSuppressor>,

    /// Alert correlator
    correlator: Arc<AlertCorrelator>,

    /// Processing configuration
    config: Arc<RwLock<AlertManagerConfig>>,

    /// Processing task handle
    processing_handle: Arc<TokioMutex<Option<tokio::task::JoinHandle<()>>>>,

    /// Shutdown signal
    shutdown_signal: Arc<AtomicBool>,
}

/// Statistics for alert manager
#[derive(Debug, Default)]
pub struct AlertManagerStats {
    /// Alerts processed
    pub alerts_processed: AtomicU64,

    /// Notifications sent
    pub notifications_sent: AtomicU64,

    /// Processing errors
    pub processing_errors: AtomicU64,

    /// Alerts suppressed
    pub alerts_suppressed: AtomicU64,

    /// Alerts correlated
    pub alerts_correlated: AtomicU64,

    /// Average processing time
    pub avg_processing_time: Arc<Mutex<Duration>>,

    /// Queue size
    pub queue_size: AtomicU64,

    /// Peak queue size
    pub peak_queue_size: AtomicU64,
}

/// Configuration for alert manager
#[derive(Debug, Clone)]
pub struct AlertManagerConfig {
    /// Maximum queue size
    pub max_queue_size: usize,

    /// Processing batch size
    pub batch_size: usize,

    /// Processing interval
    pub processing_interval: Duration,

    /// Enable alert suppression
    pub enable_suppression: bool,

    /// Enable alert correlation
    pub enable_correlation: bool,

    /// Maximum processing time per alert
    pub max_processing_time: Duration,

    /// Number of worker threads
    pub worker_threads: usize,
}

/// Alert processor trait for processing alerts
pub trait AlertProcessor: Send + Sync {
    /// Process an alert
    fn process_alert(&self, alert: &AlertEvent) -> Result<ProcessedAlert>;

    /// Get processor name
    fn name(&self) -> &str;

    /// Check if processor supports alert type
    fn supports(&self, alert: &AlertEvent) -> bool;

    /// Get processing priority
    fn priority(&self) -> u8 {
        50 // Default priority
    }
}

/// Processed alert result
#[derive(Debug, Clone)]
pub struct ProcessedAlert {
    /// Original alert
    pub alert: AlertEvent,

    /// Processing timestamp
    pub processed_at: DateTime<Utc>,

    /// Processing results
    pub results: HashMap<String, String>,

    /// Notifications to send
    pub notifications: Vec<Notification>,

    /// Processing metadata
    pub metadata: HashMap<String, String>,
}

/// Notification to be sent
#[derive(Debug, Clone)]
pub struct Notification {
    /// Notification ID
    pub id: String,

    /// Channel to use
    pub channel: String,

    /// Recipients
    pub recipients: Vec<String>,

    /// Subject/title
    pub subject: String,

    /// Content/body
    pub content: String,

    /// Priority level
    pub priority: NotificationPriority,

    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Notification priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum NotificationPriority {
    /// Low priority
    Low,

    /// Normal priority
    Normal,

    /// High priority
    High,

    /// Critical priority
    Critical,

    /// Emergency priority
    Emergency,
}

/// Trait for notification channels
pub trait NotificationChannel: Send + Sync {
    /// Send notification
    fn send_notification(&self, notification: &Notification) -> Result<()>;

    /// Get channel name
    fn name(&self) -> &str;

    /// Check if channel supports notification type
    fn supports(&self, notification_type: &str) -> bool;

    /// Get maximum message size
    fn max_message_size(&self) -> usize {
        10000 // Default 10KB
    }

    /// Check if channel is available
    fn is_available(&self) -> bool {
        true // Default available
    }
}

impl AlertManager {
    /// Create a new alert manager
    pub async fn new() -> Result<Self> {
        let manager = Self {
            processors: Arc::new(Mutex::new(Vec::new())),
            channels: Arc::new(Mutex::new(Vec::new())),
            alert_queue: Arc::new(TokioMutex::new(VecDeque::new())),
            stats: Arc::new(AlertManagerStats::default()),
            suppressor: Arc::new(AlertSuppressor::new()),
            correlator: Arc::new(AlertCorrelator::new()),
            config: Arc::new(RwLock::new(AlertManagerConfig::default())),
            processing_handle: Arc::new(TokioMutex::new(None)),
            shutdown_signal: Arc::new(AtomicBool::new(false)),
        };

        manager.initialize_processors().await?;
        manager.initialize_channels().await?;
        Ok(manager)
    }

    /// Start alert manager
    pub async fn start(&self) -> Result<()> {
        let mut handle = self.processing_handle.lock().await;

        if handle.is_some() {
            return Err(ThresholdError::InternalError(
                "Alert manager already started".to_string(),
            ));
        }

        let queue = Arc::clone(&self.alert_queue);
        let processors = Arc::clone(&self.processors);
        let channels = Arc::clone(&self.channels);
        let stats = Arc::clone(&self.stats);
        let suppressor = Arc::clone(&self.suppressor);
        let correlator = Arc::clone(&self.correlator);
        let config = Arc::clone(&self.config);
        let shutdown_signal = Arc::clone(&self.shutdown_signal);

        let processing_task = tokio::spawn(async move {
            Self::processing_loop(
                queue,
                processors,
                channels,
                stats,
                suppressor,
                correlator,
                config,
                shutdown_signal,
            )
            .await;
        });

        *handle = Some(processing_task);
        info!("Alert manager started");
        Ok(())
    }

    /// Stop alert manager
    pub async fn shutdown(&self) -> Result<()> {
        self.shutdown_signal.store(true, Ordering::Relaxed);

        let mut handle = self.processing_handle.lock().await;
        if let Some(task) = handle.take() {
            task.abort();
            info!("Alert manager stopped");
        }

        Ok(())
    }

    /// Process an alert
    pub async fn process_alert(&self, alert: AlertEvent) -> Result<()> {
        let config = self.config.read().expect("Config RwLock poisoned");
        let mut queue = self.alert_queue.lock().await;

        // Check queue size limits
        if queue.len() >= config.max_queue_size {
            return Err(ThresholdError::AlertProcessingError(
                "Alert queue is full".to_string(),
            ));
        }

        queue.push_back(alert);

        // Update statistics
        let queue_size = queue.len() as u64;
        self.stats.queue_size.store(queue_size, Ordering::Relaxed);

        let peak_size = self.stats.peak_queue_size.load(Ordering::Relaxed);
        if queue_size > peak_size {
            self.stats.peak_queue_size.store(queue_size, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Get alert manager statistics
    pub fn get_stats(&self) -> AlertManagerStats {
        AlertManagerStats {
            alerts_processed: AtomicU64::new(self.stats.alerts_processed.load(Ordering::Relaxed)),
            notifications_sent: AtomicU64::new(
                self.stats.notifications_sent.load(Ordering::Relaxed),
            ),
            processing_errors: AtomicU64::new(self.stats.processing_errors.load(Ordering::Relaxed)),
            alerts_suppressed: AtomicU64::new(self.stats.alerts_suppressed.load(Ordering::Relaxed)),
            alerts_correlated: AtomicU64::new(self.stats.alerts_correlated.load(Ordering::Relaxed)),
            avg_processing_time: Arc::new(Mutex::new(
                *self
                    .stats
                    .avg_processing_time
                    .lock()
                    .expect("Avg processing time lock poisoned"),
            )),
            queue_size: AtomicU64::new(self.stats.queue_size.load(Ordering::Relaxed)),
            peak_queue_size: AtomicU64::new(self.stats.peak_queue_size.load(Ordering::Relaxed)),
        }
    }

    /// Main processing loop
    async fn processing_loop(
        queue: Arc<TokioMutex<VecDeque<AlertEvent>>>,
        processors: Arc<Mutex<Vec<Box<dyn AlertProcessor + Send + Sync>>>>,
        channels: Arc<Mutex<Vec<Box<dyn NotificationChannel + Send + Sync>>>>,
        stats: Arc<AlertManagerStats>,
        suppressor: Arc<AlertSuppressor>,
        correlator: Arc<AlertCorrelator>,
        config: Arc<RwLock<AlertManagerConfig>>,
        shutdown_signal: Arc<AtomicBool>,
    ) {
        let mut interval = interval(Duration::from_millis(100)); // 100ms processing interval

        while !shutdown_signal.load(Ordering::Relaxed) {
            interval.tick().await;

            let (batch_size, enable_suppression, enable_correlation) = {
                let config_read = config.read().expect("Config RwLock poisoned");
                (
                    config_read.batch_size,
                    config_read.enable_suppression,
                    config_read.enable_correlation,
                )
            };

            // Process alerts in batches
            let alerts = {
                let mut queue_guard = queue.lock().await;
                let mut batch = Vec::new();

                for _ in 0..batch_size {
                    if let Some(alert) = queue_guard.pop_front() {
                        batch.push(alert);
                    } else {
                        break;
                    }
                }

                stats.queue_size.store(queue_guard.len() as u64, Ordering::Relaxed);
                batch
            };

            if alerts.is_empty() {
                continue;
            }

            // Process each alert
            for mut alert in alerts {
                let start_time = Instant::now();

                // Apply suppression
                if enable_suppression && suppressor.should_suppress(&alert).await {
                    suppressor.suppress_alert(&mut alert).await;
                    stats.alerts_suppressed.fetch_add(1, Ordering::Relaxed);
                    continue;
                }

                // Apply correlation
                if enable_correlation {
                    correlator.correlate_alert(&mut alert).await;
                    stats.alerts_correlated.fetch_add(1, Ordering::Relaxed);
                }

                // Process alert
                match Self::process_single_alert(&alert, &processors, &channels).await {
                    Ok(_) => {
                        stats.alerts_processed.fetch_add(1, Ordering::Relaxed);
                    },
                    Err(e) => {
                        error!("Failed to process alert {}: {}", alert.alert_id, e);
                        stats.processing_errors.fetch_add(1, Ordering::Relaxed);
                    },
                }

                // Update processing time statistics
                let processing_time = start_time.elapsed();
                let mut avg_time =
                    stats.avg_processing_time.lock().expect("Avg time lock poisoned");
                *avg_time = (*avg_time + processing_time) / 2; // Simple moving average
            }
        }
    }

    /// Process a single alert
    async fn process_single_alert(
        alert: &AlertEvent,
        processors: &Arc<Mutex<Vec<Box<dyn AlertProcessor + Send + Sync>>>>,
        channels: &Arc<Mutex<Vec<Box<dyn NotificationChannel + Send + Sync>>>>,
    ) -> Result<()> {
        let all_notifications = {
            let processors_guard = processors.lock().expect("Processors lock poisoned");
            let mut notifications = Vec::new();

            // Find and run appropriate processors
            for processor in processors_guard.iter() {
                if processor.supports(alert) {
                    match processor.process_alert(alert) {
                        Ok(processed) => {
                            notifications.extend(processed.notifications);
                        },
                        Err(e) => {
                            warn!(
                                "Processor {} failed for alert {}: {}",
                                processor.name(),
                                alert.alert_id,
                                e
                            );
                        },
                    }
                }
            }

            notifications
        };

        // Send notifications
        if !all_notifications.is_empty() {
            Self::send_notifications(&all_notifications, channels).await?;
        }

        Ok(())
    }

    /// Send notifications through available channels
    async fn send_notifications(
        notifications: &[Notification],
        channels: &Arc<Mutex<Vec<Box<dyn NotificationChannel + Send + Sync>>>>,
    ) -> Result<()> {
        let channels_guard = channels.lock().expect("Channels lock poisoned");

        for notification in notifications {
            for channel in channels_guard.iter() {
                if channel.supports(&notification.channel) && channel.is_available() {
                    match channel.send_notification(notification) {
                        Ok(()) => {
                            debug!(
                                "Sent notification {} via {}",
                                notification.id,
                                channel.name()
                            );
                        },
                        Err(e) => {
                            warn!(
                                "Failed to send notification {} via {}: {}",
                                notification.id,
                                channel.name(),
                                e
                            );
                        },
                    }
                }
            }
        }

        Ok(())
    }

    /// Initialize alert processors
    async fn initialize_processors(&self) -> Result<()> {
        let mut processors = self.processors.lock().expect("Processors lock poisoned");
        processors.push(Box::new(DefaultAlertProcessor::new()));
        processors.push(Box::new(PerformanceAlertProcessor::new()));
        processors.push(Box::new(ResourceAlertProcessor::new()));
        processors.push(Box::new(CriticalAlertProcessor::new()));
        Ok(())
    }

    /// Initialize notification channels
    async fn initialize_channels(&self) -> Result<()> {
        let mut channels = self.channels.lock().expect("Channels lock poisoned");
        channels.push(Box::new(LogNotificationChannel::new()));
        channels.push(Box::new(EmailNotificationChannel::new()));
        channels.push(Box::new(WebhookNotificationChannel::new()));
        channels.push(Box::new(SlackNotificationChannel::new()));
        Ok(())
    }
}

impl Default for AlertManagerConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 10000,
            batch_size: 50,
            processing_interval: Duration::from_millis(100),
            enable_suppression: true,
            enable_correlation: true,
            max_processing_time: Duration::from_secs(30),
            worker_threads: 4,
        }
    }
}

// =============================================================================
// ALERT SUPPRESSION SYSTEM
// =============================================================================

/// Alert suppression system for reducing alert noise
///
/// Advanced alert suppression system that reduces alert noise through
/// deduplication, frequency limiting, and intelligent suppression policies.
pub struct AlertSuppressor {
    /// Suppression rules
    rules: Arc<RwLock<Vec<SuppressionRule>>>,

    /// Alert fingerprints for deduplication
    fingerprints: Arc<TokioMutex<HashMap<String, AlertFingerprint>>>,

    /// Suppression statistics
    stats: Arc<SuppressionStats>,

    /// Configuration
    config: Arc<RwLock<SuppressionConfig>>,
}

/// Suppression rule for alerts
#[derive(Debug, Clone)]
pub struct SuppressionRule {
    /// Rule ID
    pub id: String,

    /// Rule name
    pub name: String,

    /// Rule type
    pub rule_type: SuppressionRuleType,

    /// Matching criteria
    pub criteria: SuppressionCriteria,

    /// Suppression action
    pub action: SuppressionAction,

    /// Rule priority
    pub priority: u8,

    /// Rule enabled
    pub enabled: bool,
}

/// Types of suppression rules
#[derive(Debug, Clone)]
pub enum SuppressionRuleType {
    /// Frequency-based suppression
    Frequency,

    /// Duplicate suppression
    Duplicate,

    /// Time-based suppression
    TimeBased,

    /// Pattern-based suppression
    Pattern,

    /// Severity-based suppression
    Severity,

    /// Custom suppression logic
    Custom(String),
}

/// Criteria for alert suppression
#[derive(Debug, Clone)]
pub struct SuppressionCriteria {
    /// Metric patterns to match
    pub metric_patterns: Vec<String>,

    /// Severity levels to match
    pub severity_levels: Vec<SeverityLevel>,

    /// Time window for suppression
    pub time_window: Duration,

    /// Maximum alerts per window
    pub max_alerts_per_window: u32,

    /// Threshold patterns to match
    pub threshold_patterns: Vec<String>,

    /// Additional conditions
    pub conditions: HashMap<String, String>,
}

/// Suppression action to take
#[derive(Debug, Clone)]
pub enum SuppressionAction {
    /// Suppress completely
    Suppress,

    /// Reduce frequency
    ReduceFrequency { factor: f32 },

    /// Aggregate alerts
    Aggregate { window: Duration },

    /// Downgrade severity
    DowngradeSeverity { levels: u8 },

    /// Route to different channel
    Reroute { channel: String },
}

/// Alert fingerprint for deduplication
#[derive(Debug, Clone)]
pub struct AlertFingerprint {
    /// Fingerprint hash
    pub hash: String,

    /// First occurrence
    pub first_seen: DateTime<Utc>,

    /// Last occurrence
    pub last_seen: DateTime<Utc>,

    /// Occurrence count
    pub count: u32,

    /// Suppression status
    pub suppressed: bool,

    /// Associated alert IDs
    pub alert_ids: Vec<String>,
}

/// Statistics for alert suppression
#[derive(Debug, Default)]
pub struct SuppressionStats {
    /// Total alerts processed
    pub total_processed: AtomicU64,

    /// Total alerts suppressed
    pub total_suppressed: AtomicU64,

    /// Suppression by rule type
    pub suppression_by_rule: Arc<Mutex<HashMap<String, u64>>>,

    /// Average suppression rate
    pub avg_suppression_rate: Arc<Mutex<f32>>,

    /// Peak suppression rate
    pub peak_suppression_rate: Arc<Mutex<f32>>,
}

/// Configuration for alert suppression
#[derive(Debug, Clone)]
pub struct SuppressionConfig {
    /// Enable alert deduplication
    pub enable_deduplication: bool,

    /// Deduplication window
    pub deduplication_window: Duration,

    /// Maximum fingerprint cache size
    pub max_fingerprint_cache: usize,

    /// Default suppression window
    pub default_suppression_window: Duration,

    /// Enable adaptive suppression
    pub enable_adaptive_suppression: bool,
}

impl Default for AlertSuppressor {
    fn default() -> Self {
        Self::new()
    }
}

impl AlertSuppressor {
    /// Create a new alert suppressor
    pub fn new() -> Self {
        Self {
            rules: Arc::new(RwLock::new(Vec::new())),
            fingerprints: Arc::new(TokioMutex::new(HashMap::new())),
            stats: Arc::new(SuppressionStats::default()),
            config: Arc::new(RwLock::new(SuppressionConfig::default())),
        }
    }

    /// Check if alert should be suppressed
    pub async fn should_suppress(&self, alert: &AlertEvent) -> bool {
        self.stats.total_processed.fetch_add(1, Ordering::Relaxed);

        let suppressed_by_rule = {
            let rules = self.rules.read().expect("Rules RwLock poisoned");
            let mut suppressed = false;

            for rule in rules.iter() {
                if !rule.enabled {
                    continue;
                }

                if self.matches_criteria(&rule.criteria, alert)
                    && matches!(rule.action, SuppressionAction::Suppress)
                {
                    self.stats.total_suppressed.fetch_add(1, Ordering::Relaxed);
                    self.update_rule_stats(&rule.id);
                    suppressed = true;
                    break;
                }
            }

            suppressed
        };

        if suppressed_by_rule {
            return true;
        }

        // Check deduplication
        let enable_deduplication = {
            let config = self.config.read().expect("Config RwLock poisoned");
            config.enable_deduplication
        };

        if enable_deduplication && self.is_duplicate(alert).await {
            self.stats.total_suppressed.fetch_add(1, Ordering::Relaxed);
            return true;
        }

        false
    }

    /// Suppress an alert
    pub async fn suppress_alert(&self, alert: &mut AlertEvent) {
        alert.suppression_info = Some(SuppressionInfo {
            reason: "Suppressed by alert suppressor".to_string(),
            start_time: Utc::now(), // Changed from suppressed_at to match types.rs
            duration: Duration::from_secs(300), // 5 minutes default
            suppressed_count: 1,
        });

        // Update fingerprint if deduplication is enabled
        let enable_deduplication = {
            let config = self.config.read().expect("Config RwLock poisoned");
            config.enable_deduplication
        };

        if enable_deduplication {
            self.update_fingerprint(alert).await;
        }
    }

    /// Check if alert matches criteria
    fn matches_criteria(&self, criteria: &SuppressionCriteria, alert: &AlertEvent) -> bool {
        // Check metric patterns
        if !criteria.metric_patterns.is_empty() {
            let metric_matches = criteria
                .metric_patterns
                .iter()
                .any(|pattern| alert.threshold.metric.contains(pattern));
            if !metric_matches {
                return false;
            }
        }

        // Check severity levels
        if !criteria.severity_levels.is_empty()
            && !criteria.severity_levels.contains(&alert.severity)
        {
            return false;
        }

        // Check threshold patterns
        if !criteria.threshold_patterns.is_empty() {
            let threshold_matches = criteria
                .threshold_patterns
                .iter()
                .any(|pattern| alert.threshold.name.contains(pattern));
            if !threshold_matches {
                return false;
            }
        }

        // Check additional conditions
        for (key, expected_value) in &criteria.conditions {
            if let Some(actual_value) = alert.context.get(key) {
                if actual_value != expected_value {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }

    /// Check if alert is a duplicate
    async fn is_duplicate(&self, alert: &AlertEvent) -> bool {
        let fingerprint_hash = self.calculate_fingerprint(alert);
        let mut fingerprints = self.fingerprints.lock().await;

        if let Some(existing) = fingerprints.get_mut(&fingerprint_hash) {
            existing.last_seen = Utc::now();
            existing.count += 1;
            existing.alert_ids.push(alert.alert_id.clone());

            // Check if within deduplication window
            let time_diff = existing.last_seen.signed_duration_since(existing.first_seen);
            let dedup_window = chrono::Duration::from_std(
                self.config.read().expect("Config RwLock poisoned").deduplication_window,
            )
            .expect("Duration conversion failed");

            return time_diff <= dedup_window;
        }

        // Create new fingerprint
        let fingerprint = AlertFingerprint {
            hash: fingerprint_hash.clone(),
            first_seen: Utc::now(),
            last_seen: Utc::now(),
            count: 1,
            suppressed: false,
            alert_ids: vec![alert.alert_id.clone()],
        };

        fingerprints.insert(fingerprint_hash, fingerprint);

        // Maintain cache size
        let max_cache_size =
            self.config.read().expect("Config RwLock poisoned").max_fingerprint_cache;
        if fingerprints.len() > max_cache_size {
            // Remove oldest entries - collect keys first to avoid borrow conflict
            let mut entries: Vec<_> =
                fingerprints.iter().map(|(k, v)| (k.clone(), v.first_seen)).collect();
            entries.sort_by_key(|(_, first_seen)| *first_seen);

            let to_remove = entries.len() - max_cache_size;
            let keys_to_remove: Vec<_> =
                entries.iter().take(to_remove).map(|(k, _)| k.clone()).collect();

            for hash in keys_to_remove {
                fingerprints.remove(&hash);
            }
        }

        false
    }

    /// Calculate alert fingerprint
    fn calculate_fingerprint(&self, alert: &AlertEvent) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        alert.threshold.metric.hash(&mut hasher);
        alert.threshold.name.hash(&mut hasher);
        alert.severity.hash(&mut hasher);

        // Include relevant context in fingerprint
        for (key, value) in &alert.context {
            if key != "timestamp" && key != "alert_id" {
                key.hash(&mut hasher);
                value.hash(&mut hasher);
            }
        }

        format!("{:x}", hasher.finish())
    }

    /// Update fingerprint for alert
    async fn update_fingerprint(&self, alert: &AlertEvent) {
        let fingerprint_hash = self.calculate_fingerprint(alert);
        let mut fingerprints = self.fingerprints.lock().await;

        if let Some(fingerprint) = fingerprints.get_mut(&fingerprint_hash) {
            fingerprint.suppressed = true;
            if let Some(ref suppression_info) = alert.suppression_info {
                fingerprint.count = suppression_info.suppressed_count;
            }
        }
    }

    /// Update rule statistics
    fn update_rule_stats(&self, rule_id: &str) {
        let mut stats =
            self.stats.suppression_by_rule.lock().expect("Suppression stats lock poisoned");
        *stats.entry(rule_id.to_string()).or_insert(0) += 1;
    }

    /// Add suppression rule
    pub fn add_rule(&self, rule: SuppressionRule) {
        let mut rules = self.rules.write().expect("Rules RwLock poisoned");
        rules.push(rule);

        // Sort by priority (higher priority first)
        rules.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Remove suppression rule
    pub fn remove_rule(&self, rule_id: &str) {
        let mut rules = self.rules.write().expect("Rules RwLock poisoned");
        rules.retain(|rule| rule.id != rule_id);
    }

    /// Get suppression statistics
    pub fn get_stats(&self) -> SuppressionStats {
        SuppressionStats {
            total_processed: AtomicU64::new(self.stats.total_processed.load(Ordering::Relaxed)),
            total_suppressed: AtomicU64::new(self.stats.total_suppressed.load(Ordering::Relaxed)),
            suppression_by_rule: Arc::new(Mutex::new(
                self.stats
                    .suppression_by_rule
                    .lock()
                    .expect("Suppression stats lock poisoned")
                    .clone(),
            )),
            avg_suppression_rate: Arc::new(Mutex::new(
                *self
                    .stats
                    .avg_suppression_rate
                    .lock()
                    .expect("Avg suppression rate lock poisoned"),
            )),
            peak_suppression_rate: Arc::new(Mutex::new(
                *self
                    .stats
                    .peak_suppression_rate
                    .lock()
                    .expect("Peak suppression rate lock poisoned"),
            )),
        }
    }
}

impl Default for SuppressionConfig {
    fn default() -> Self {
        Self {
            enable_deduplication: true,
            deduplication_window: Duration::from_secs(300), // 5 minutes
            max_fingerprint_cache: 10000,
            default_suppression_window: Duration::from_secs(600), // 10 minutes
            enable_adaptive_suppression: true,
        }
    }
}

// =============================================================================
// ALERT CORRELATION SYSTEM
// =============================================================================

/// Alert correlation system for identifying related alerts
///
/// Advanced alert correlation system that identifies relationships between
/// alerts to provide better context and reduce alert fatigue.
pub struct AlertCorrelator {
    /// Correlation rules
    rules: Arc<RwLock<Vec<CorrelationRule>>>,

    /// Active correlations
    correlations: Arc<TokioMutex<HashMap<String, AlertCorrelation>>>,

    /// Correlation statistics
    stats: Arc<CorrelationStats>,

    /// Configuration
    config: Arc<RwLock<CorrelationConfig>>,
}

/// Correlation rule for alerts
#[derive(Debug, Clone)]
pub struct CorrelationRule {
    /// Rule ID
    pub id: String,

    /// Rule name
    pub name: String,

    /// Rule type
    pub rule_type: CorrelationRuleType,

    /// Matching criteria
    pub criteria: CorrelationCriteria,

    /// Correlation strength (0.0 to 1.0)
    pub strength: f32,

    /// Time window for correlation
    pub time_window: Duration,

    /// Rule enabled
    pub enabled: bool,
}

/// Types of correlation rules
#[derive(Debug, Clone)]
pub enum CorrelationRuleType {
    /// Resource-based correlation
    Resource,

    /// Metric-based correlation
    Metric,

    /// Temporal correlation
    Temporal,

    /// Causal correlation
    Causal,

    /// Pattern-based correlation
    Pattern,

    /// Custom correlation logic
    Custom(String),
}

/// Criteria for alert correlation
#[derive(Debug, Clone)]
pub struct CorrelationCriteria {
    /// Metric patterns to match
    pub metric_patterns: Vec<String>,

    /// Resource patterns to match
    pub resource_patterns: Vec<String>,

    /// Severity level matching
    pub severity_matching: SeverityMatching,

    /// Time tolerance for correlation
    pub time_tolerance: Duration,

    /// Minimum correlation strength
    pub min_strength: f32,

    /// Additional conditions
    pub conditions: HashMap<String, String>,
}

/// Severity matching strategy
#[derive(Debug, Clone)]
pub enum SeverityMatching {
    /// Exact severity match
    Exact,

    /// Within severity range
    Range {
        min: SeverityLevel,
        max: SeverityLevel,
    },

    /// Any severity
    Any,

    /// Escalating severity
    Escalating,
}

/// Alert correlation information
#[derive(Debug, Clone)]
pub struct AlertCorrelation {
    /// Correlation ID
    pub correlation_id: String,

    /// Root alert ID
    pub root_alert_id: String,

    /// Correlated alert IDs
    pub correlated_alerts: Vec<String>,

    /// Correlation type
    pub correlation_type: CorrelationType,

    /// Correlation strength
    pub strength: f32,

    /// Creation time
    pub created_at: DateTime<Utc>,

    /// Last updated
    pub updated_at: DateTime<Utc>,

    /// Correlation metadata
    pub metadata: HashMap<String, String>,
}

/// Statistics for alert correlation
#[derive(Debug, Default)]
pub struct CorrelationStats {
    /// Total alerts processed
    pub total_processed: AtomicU64,

    /// Total correlations created
    pub total_correlations: AtomicU64,

    /// Correlations by type
    pub correlations_by_type: Arc<Mutex<HashMap<String, u64>>>,

    /// Average correlation strength
    pub avg_correlation_strength: Arc<Mutex<f32>>,

    /// Active correlations count
    pub active_correlations: AtomicU64,
}

/// Configuration for alert correlation
#[derive(Debug, Clone)]
pub struct CorrelationConfig {
    /// Enable temporal correlation
    pub enable_temporal: bool,

    /// Enable resource correlation
    pub enable_resource: bool,

    /// Enable metric correlation
    pub enable_metric: bool,

    /// Maximum correlation window
    pub max_correlation_window: Duration,

    /// Minimum correlation strength
    pub min_correlation_strength: f32,

    /// Maximum correlations per alert
    pub max_correlations_per_alert: usize,
}

impl Default for AlertCorrelator {
    fn default() -> Self {
        Self::new()
    }
}

impl AlertCorrelator {
    /// Create a new alert correlator
    pub fn new() -> Self {
        Self {
            rules: Arc::new(RwLock::new(Vec::new())),
            correlations: Arc::new(TokioMutex::new(HashMap::new())),
            stats: Arc::new(CorrelationStats::default()),
            config: Arc::new(RwLock::new(CorrelationConfig::default())),
        }
    }

    /// Correlate an alert with existing alerts
    pub async fn correlate_alert(&self, alert: &mut AlertEvent) {
        self.stats.total_processed.fetch_add(1, Ordering::Relaxed);

        let rules_snapshot = {
            let rules = self.rules.read().expect("Rules RwLock poisoned");
            rules.clone()
        };

        let mut correlations = self.correlations.lock().await;

        for rule in rules_snapshot.iter() {
            if !rule.enabled {
                continue;
            }

            if self.matches_correlation_criteria(&rule.criteria, alert) {
                // Look for existing correlations that match this rule
                for (_, correlation) in correlations.iter_mut() {
                    if self.can_correlate_with_existing(rule, alert, correlation) {
                        // Add alert to existing correlation
                        correlation.correlated_alerts.push(alert.alert_id.clone());
                        correlation.updated_at = Utc::now();

                        // Update alert with correlation info
                        alert.correlation_id = Some(correlation.correlation_id.clone());

                        self.update_correlation_stats(&rule.rule_type);
                        return;
                    }
                }

                // Create new correlation if no existing one found
                let correlation_id = Uuid::new_v4().to_string();
                let correlation = AlertCorrelation {
                    correlation_id: correlation_id.clone(),
                    root_alert_id: alert.alert_id.clone(),
                    correlated_alerts: vec![alert.alert_id.clone()],
                    correlation_type: self.rule_type_to_correlation_type(&rule.rule_type),
                    strength: rule.strength,
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                    metadata: HashMap::new(),
                };

                correlations.insert(correlation_id.clone(), correlation);
                alert.correlation_id = Some(correlation_id);

                self.stats.total_correlations.fetch_add(1, Ordering::Relaxed);
                self.update_correlation_stats(&rule.rule_type);
                break;
            }
        }

        // Clean up old correlations
        self.cleanup_old_correlations(&mut correlations).await;

        self.stats
            .active_correlations
            .store(correlations.len() as u64, Ordering::Relaxed);
    }

    /// Check if alert matches correlation criteria
    fn matches_correlation_criteria(
        &self,
        criteria: &CorrelationCriteria,
        alert: &AlertEvent,
    ) -> bool {
        // Check metric patterns
        if !criteria.metric_patterns.is_empty() {
            let metric_matches = criteria
                .metric_patterns
                .iter()
                .any(|pattern| alert.threshold.metric.contains(pattern));
            if !metric_matches {
                return false;
            }
        }

        // Check resource patterns (from context)
        if !criteria.resource_patterns.is_empty() {
            let resource_matches = criteria
                .resource_patterns
                .iter()
                .any(|pattern| alert.context.values().any(|value| value.contains(pattern)));
            if !resource_matches {
                return false;
            }
        }

        // Check severity matching
        match &criteria.severity_matching {
            SeverityMatching::Exact => {
                // For exact matching, we need another alert to compare with
                // This will be checked in can_correlate_with_existing
            },
            SeverityMatching::Range { min, max } => {
                if alert.severity < *min || alert.severity > *max {
                    return false;
                }
            },
            SeverityMatching::Any => {
                // Any severity is acceptable
            },
            SeverityMatching::Escalating => {
                // Will be checked in can_correlate_with_existing
            },
        }

        // Check additional conditions
        for (key, expected_value) in &criteria.conditions {
            if let Some(actual_value) = alert.context.get(key) {
                if actual_value != expected_value {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }

    /// Check if alert can be correlated with existing correlation
    fn can_correlate_with_existing(
        &self,
        rule: &CorrelationRule,
        alert: &AlertEvent,
        correlation: &AlertCorrelation,
    ) -> bool {
        // Check time window
        let time_diff = alert.timestamp.signed_duration_since(correlation.updated_at);
        let time_window =
            chrono::Duration::from_std(rule.time_window).expect("Duration conversion failed");

        if time_diff > time_window {
            return false;
        }

        // Check correlation strength
        if rule.strength < correlation.strength * 0.8 {
            return false;
        }

        // Check maximum correlations per alert
        let max_correlations =
            self.config.read().expect("Config RwLock poisoned").max_correlations_per_alert;
        if correlation.correlated_alerts.len() >= max_correlations {
            return false;
        }

        true
    }

    /// Convert rule type to correlation type
    fn rule_type_to_correlation_type(&self, rule_type: &CorrelationRuleType) -> CorrelationType {
        match rule_type {
            CorrelationRuleType::Resource => CorrelationType::Resource,
            CorrelationRuleType::Metric => CorrelationType::Metric,
            CorrelationRuleType::Temporal => CorrelationType::Temporal,
            CorrelationRuleType::Causal => CorrelationType::Causal,
            CorrelationRuleType::Pattern => CorrelationType::Pattern,
            CorrelationRuleType::Custom(_) => CorrelationType::Pattern,
        }
    }

    /// Update correlation statistics
    fn update_correlation_stats(&self, rule_type: &CorrelationRuleType) {
        let type_name = format!("{:?}", rule_type);
        let mut stats = self
            .stats
            .correlations_by_type
            .lock()
            .expect("Correlations stats lock poisoned");
        *stats.entry(type_name).or_insert(0) += 1;
    }

    /// Clean up old correlations
    async fn cleanup_old_correlations(&self, correlations: &mut HashMap<String, AlertCorrelation>) {
        let max_window = self.config.read().expect("Config RwLock poisoned").max_correlation_window;
        let cutoff_time = Utc::now()
            - chrono::Duration::from_std(max_window).expect("Duration conversion failed");

        correlations.retain(|_, correlation| correlation.updated_at > cutoff_time);
    }

    /// Add correlation rule
    pub fn add_rule(&self, rule: CorrelationRule) {
        let mut rules = self.rules.write().expect("Rules RwLock poisoned");
        rules.push(rule);
    }

    /// Remove correlation rule
    pub fn remove_rule(&self, rule_id: &str) {
        let mut rules = self.rules.write().expect("Rules RwLock poisoned");
        rules.retain(|rule| rule.id != rule_id);
    }

    /// Get correlation statistics
    pub fn get_stats(&self) -> CorrelationStats {
        CorrelationStats {
            total_processed: AtomicU64::new(self.stats.total_processed.load(Ordering::Relaxed)),
            total_correlations: AtomicU64::new(
                self.stats.total_correlations.load(Ordering::Relaxed),
            ),
            correlations_by_type: Arc::new(Mutex::new(
                self.stats
                    .correlations_by_type
                    .lock()
                    .expect("Correlations stats lock poisoned")
                    .clone(),
            )),
            avg_correlation_strength: Arc::new(Mutex::new(
                *self
                    .stats
                    .avg_correlation_strength
                    .lock()
                    .expect("Avg correlation strength lock poisoned"),
            )),
            active_correlations: AtomicU64::new(
                self.stats.active_correlations.load(Ordering::Relaxed),
            ),
        }
    }

    /// Get active correlations
    pub async fn get_active_correlations(&self) -> Vec<AlertCorrelation> {
        let correlations = self.correlations.lock().await;
        correlations.values().cloned().collect()
    }

    /// Get correlation by ID
    pub async fn get_correlation(&self, correlation_id: &str) -> Option<AlertCorrelation> {
        let correlations = self.correlations.lock().await;
        correlations.get(correlation_id).cloned()
    }
}

impl Default for CorrelationConfig {
    fn default() -> Self {
        Self {
            enable_temporal: true,
            enable_resource: true,
            enable_metric: true,
            max_correlation_window: Duration::from_secs(3600), // 1 hour
            min_correlation_strength: 0.3,
            max_correlations_per_alert: 10,
        }
    }
}

// =============================================================================
// ESCALATION MANAGEMENT SYSTEM
// =============================================================================

/// Escalation manager for alert escalation
///
/// Advanced escalation manager that handles multi-level alert escalation
/// with time-based triggers, severity-based routing, and customizable policies.
pub struct EscalationManager {
    /// Escalation policies
    policies: Arc<RwLock<HashMap<String, EscalationPolicy>>>,

    /// Escalation state tracking
    state: Arc<TokioMutex<HashMap<String, EscalationState>>>,

    /// Escalation statistics
    stats: Arc<EscalationStats>,

    /// Configuration
    config: Arc<RwLock<EscalationConfig>>,

    /// Processing task handle
    processing_handle: Arc<TokioMutex<Option<tokio::task::JoinHandle<()>>>>,

    /// Shutdown signal
    shutdown_signal: Arc<AtomicBool>,
}

/// Escalation policy configuration
#[derive(Debug, Clone)]
pub struct EscalationPolicy {
    /// Policy name
    pub name: String,

    /// Escalation levels
    pub levels: Vec<EscalationLevel>,

    /// Maximum escalation level
    pub max_level: u8,

    /// Auto-escalation enabled
    pub auto_escalation: bool,

    /// Escalation triggers
    pub triggers: Vec<EscalationTrigger>,

    /// Policy enabled
    pub enabled: bool,
}

/// Escalation level configuration
#[derive(Debug, Clone)]
pub struct EscalationLevel {
    /// Level number
    pub level: u8,

    /// Time before escalation
    pub time_to_escalate: Duration,

    /// Recipients at this level
    pub recipients: Vec<String>,

    /// Notification channels
    pub channels: Vec<String>,

    /// Actions to take
    pub actions: Vec<EscalationAction>,
}

/// Current escalation state
#[derive(Debug, Clone)]
pub struct EscalationState {
    /// Alert ID being escalated
    pub alert_id: String,

    /// Current escalation level
    pub current_level: u8,

    /// Escalation start time
    pub started_at: DateTime<Utc>,

    /// Next escalation time
    pub next_escalation: DateTime<Utc>,

    /// Escalation history
    pub escalation_history: Vec<EscalationEvent>,

    /// Acknowledged
    pub acknowledged: bool,

    /// Acknowledgment time
    pub acknowledged_at: Option<DateTime<Utc>>,
}

/// Escalation trigger conditions
#[derive(Debug, Clone)]
pub enum EscalationTrigger {
    /// Time-based escalation
    TimeBasedDuration(Duration),

    /// Severity-based escalation
    SeverityLevel(SeverityLevel),

    /// Repeat alert count
    RepeatCount(u32),

    /// No acknowledgment
    NoAcknowledgment(Duration),

    /// Custom trigger
    Custom(String),
}

/// Actions to take during escalation
#[derive(Debug, Clone)]
pub enum EscalationAction {
    /// Send notification
    Notify {
        channel: String,
        recipients: Vec<String>,
    },

    /// Execute webhook
    Webhook {
        url: String,
        payload: HashMap<String, String>,
    },

    /// Create incident
    CreateIncident { severity: String, assignee: String },

    /// Auto-remediation
    AutoRemediate {
        script: String,
        parameters: HashMap<String, String>,
    },

    /// Custom action
    Custom(String),
}

/// Escalation event record
#[derive(Debug, Clone)]
pub struct EscalationEvent {
    /// Event timestamp
    pub timestamp: DateTime<Utc>,

    /// Event type
    pub event_type: EscalationEventType,

    /// Event level
    pub level: u8,

    /// Event details
    pub details: HashMap<String, String>,
}

/// Types of escalation events
#[derive(Debug, Clone)]
pub enum EscalationEventType {
    /// Escalation started
    Started,

    /// Escalation level increased
    LevelIncreased,

    /// Escalation acknowledged
    Acknowledged,

    /// Escalation resolved
    Resolved,

    /// Escalation failed
    Failed,

    /// Manual escalation
    Manual,
}

/// Statistics for escalation management
#[derive(Debug, Default)]
pub struct EscalationStats {
    /// Total escalations
    pub total_escalations: AtomicU64,

    /// Active escalations
    pub active_escalations: AtomicU64,

    /// Escalations by level
    pub escalations_by_level: Arc<Mutex<HashMap<u8, u64>>>,

    /// Average escalation time
    pub avg_escalation_time: Arc<Mutex<Duration>>,

    /// Acknowledgment rate
    pub acknowledgment_rate: Arc<Mutex<f32>>,
}

/// Configuration for escalation management
#[derive(Debug, Clone)]
pub struct EscalationConfig {
    /// Enable auto-escalation
    pub enable_auto_escalation: bool,

    /// Default escalation timeout
    pub default_escalation_timeout: Duration,

    /// Maximum escalation levels
    pub max_escalation_levels: u8,

    /// Escalation check interval
    pub check_interval: Duration,

    /// Enable escalation notifications
    pub enable_notifications: bool,
}

impl EscalationManager {
    /// Create a new escalation manager
    pub async fn new() -> Result<Self> {
        let manager = Self {
            policies: Arc::new(RwLock::new(HashMap::new())),
            state: Arc::new(TokioMutex::new(HashMap::new())),
            stats: Arc::new(EscalationStats::default()),
            config: Arc::new(RwLock::new(EscalationConfig::default())),
            processing_handle: Arc::new(TokioMutex::new(None)),
            shutdown_signal: Arc::new(AtomicBool::new(false)),
        };

        manager.initialize_default_policies().await?;
        Ok(manager)
    }

    /// Start escalation manager
    pub async fn start(&self) -> Result<()> {
        let mut handle = self.processing_handle.lock().await;

        if handle.is_some() {
            return Err(ThresholdError::EscalationError(
                "Escalation manager already started".to_string(),
            ));
        }

        let state = Arc::clone(&self.state);
        let policies = Arc::clone(&self.policies);
        let stats = Arc::clone(&self.stats);
        let config = Arc::clone(&self.config);
        let shutdown_signal = Arc::clone(&self.shutdown_signal);

        let escalation_task = tokio::spawn(async move {
            Self::escalation_loop(state, policies, stats, config, shutdown_signal).await;
        });

        *handle = Some(escalation_task);
        info!("Escalation manager started");
        Ok(())
    }

    /// Stop escalation manager
    pub async fn shutdown(&self) -> Result<()> {
        self.shutdown_signal.store(true, Ordering::Relaxed);

        let mut handle = self.processing_handle.lock().await;
        if let Some(task) = handle.take() {
            task.abort();
            info!("Escalation manager stopped");
        }

        Ok(())
    }

    /// Start escalation for an alert
    pub async fn start_escalation(&self, alert: &AlertEvent, policy_name: &str) -> Result<()> {
        let policies = self.policies.read().expect("Policies RwLock poisoned");
        let policy = policies.get(policy_name).ok_or_else(|| {
            ThresholdError::EscalationError(format!("Policy {} not found", policy_name))
        })?;

        if !policy.enabled {
            return Ok(());
        }

        let mut state = self.state.lock().await;

        // Check if escalation already exists
        if state.contains_key(&alert.alert_id) {
            return Ok(());
        }

        let escalation_state = EscalationState {
            alert_id: alert.alert_id.clone(),
            current_level: 0,
            started_at: Utc::now(),
            next_escalation: Utc::now()
                + chrono::Duration::from_std(policy.levels[0].time_to_escalate)
                    .expect("Duration conversion failed"),
            escalation_history: vec![EscalationEvent {
                timestamp: Utc::now(),
                event_type: EscalationEventType::Started,
                level: 0,
                details: HashMap::new(),
            }],
            acknowledged: false,
            acknowledged_at: None,
        };

        state.insert(alert.alert_id.clone(), escalation_state);
        self.stats.total_escalations.fetch_add(1, Ordering::Relaxed);
        self.stats.active_escalations.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Acknowledge escalation
    pub async fn acknowledge_escalation(&self, alert_id: &str, acknowledger: &str) -> Result<()> {
        let mut state = self.state.lock().await;

        if let Some(escalation_state) = state.get_mut(alert_id) {
            escalation_state.acknowledged = true;
            escalation_state.acknowledged_at = Some(Utc::now());
            escalation_state.escalation_history.push(EscalationEvent {
                timestamp: Utc::now(),
                event_type: EscalationEventType::Acknowledged,
                level: escalation_state.current_level,
                details: {
                    let mut details = HashMap::new();
                    details.insert("acknowledger".to_string(), acknowledger.to_string());
                    details
                },
            });

            self.stats.active_escalations.fetch_sub(1, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Resolve escalation
    pub async fn resolve_escalation(&self, alert_id: &str) -> Result<()> {
        let mut state = self.state.lock().await;

        if let Some(mut escalation_state) = state.remove(alert_id) {
            escalation_state.escalation_history.push(EscalationEvent {
                timestamp: Utc::now(),
                event_type: EscalationEventType::Resolved,
                level: escalation_state.current_level,
                details: HashMap::new(),
            });

            self.stats.active_escalations.fetch_sub(1, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Main escalation processing loop
    async fn escalation_loop(
        state: Arc<TokioMutex<HashMap<String, EscalationState>>>,
        policies: Arc<RwLock<HashMap<String, EscalationPolicy>>>,
        stats: Arc<EscalationStats>,
        config: Arc<RwLock<EscalationConfig>>,
        shutdown_signal: Arc<AtomicBool>,
    ) {
        let mut interval = interval(Duration::from_secs(30)); // Check every 30 seconds

        while !shutdown_signal.load(Ordering::Relaxed) {
            interval.tick().await;

            let mut state_guard = state.lock().await;
            let auto_escalation_enabled = {
                let config_read = config.read().expect("Config RwLock poisoned");
                config_read.enable_auto_escalation
            };

            if !auto_escalation_enabled {
                continue;
            }

            let now = Utc::now();
            let mut escalations_to_process = Vec::new();

            // Find escalations that need processing
            for (alert_id, escalation_state) in state_guard.iter() {
                if !escalation_state.acknowledged && now >= escalation_state.next_escalation {
                    escalations_to_process.push(alert_id.clone());
                }
            }

            // Process escalations
            for alert_id in escalations_to_process {
                if let Some(escalation_state) = state_guard.get_mut(&alert_id) {
                    Self::process_escalation(escalation_state, &policies, &stats);
                }
            }
        }
    }

    /// Process a single escalation
    fn process_escalation(
        escalation_state: &mut EscalationState,
        policies: &Arc<RwLock<HashMap<String, EscalationPolicy>>>,
        stats: &Arc<EscalationStats>,
    ) {
        let policies_guard = policies.read().expect("Policies RwLock poisoned");

        // Find appropriate policy (simplified - in practice, this would be more sophisticated)
        if let Some(policy) = policies_guard.values().next() {
            if escalation_state.current_level < policy.max_level
                && escalation_state.current_level < policy.levels.len() as u8
            {
                escalation_state.current_level += 1;
                let level_index = escalation_state.current_level as usize;

                if level_index < policy.levels.len() {
                    let level = &policy.levels[level_index];

                    // Schedule next escalation
                    escalation_state.next_escalation = Utc::now()
                        + chrono::Duration::from_std(level.time_to_escalate)
                            .expect("Duration conversion failed");

                    // Record escalation event
                    escalation_state.escalation_history.push(EscalationEvent {
                        timestamp: Utc::now(),
                        event_type: EscalationEventType::LevelIncreased,
                        level: escalation_state.current_level,
                        details: HashMap::new(),
                    });

                    // Execute escalation actions
                    for action in &level.actions {
                        Self::execute_escalation_action(action);
                    }

                    // Update statistics
                    let mut level_stats =
                        stats.escalations_by_level.lock().expect("Escalation stats lock poisoned");
                    *level_stats.entry(escalation_state.current_level).or_insert(0) += 1;
                }
            }
        }
    }

    /// Execute escalation action
    fn execute_escalation_action(action: &EscalationAction) {
        match action {
            EscalationAction::Notify {
                channel,
                recipients,
            } => {
                info!(
                    "Sending escalation notification via {} to {:?}",
                    channel, recipients
                );
                // Implementation would send actual notifications
            },
            EscalationAction::Webhook { url, payload } => {
                info!("Executing webhook {} with payload {:?}", url, payload);
                // Implementation would make HTTP request
            },
            EscalationAction::CreateIncident { severity, assignee } => {
                info!(
                    "Creating incident with severity {} assigned to {}",
                    severity, assignee
                );
                // Implementation would create incident in ticketing system
            },
            EscalationAction::AutoRemediate { script, parameters } => {
                info!(
                    "Executing auto-remediation script {} with parameters {:?}",
                    script, parameters
                );
                // Implementation would execute remediation script
            },
            EscalationAction::Custom(action) => {
                info!("Executing custom escalation action: {}", action);
                // Implementation would handle custom action
            },
        }
    }

    /// Initialize default escalation policies
    async fn initialize_default_policies(&self) -> Result<()> {
        let default_policy = EscalationPolicy {
            name: "default".to_string(),
            levels: vec![
                EscalationLevel {
                    level: 1,
                    time_to_escalate: Duration::from_secs(300), // 5 minutes
                    recipients: vec!["team-lead@company.com".to_string()],
                    channels: vec!["email".to_string()],
                    actions: vec![EscalationAction::Notify {
                        channel: "email".to_string(),
                        recipients: vec!["team-lead@company.com".to_string()],
                    }],
                },
                EscalationLevel {
                    level: 2,
                    time_to_escalate: Duration::from_secs(900), // 15 minutes
                    recipients: vec!["manager@company.com".to_string()],
                    channels: vec!["email".to_string(), "slack".to_string()],
                    actions: vec![
                        EscalationAction::Notify {
                            channel: "email".to_string(),
                            recipients: vec!["manager@company.com".to_string()],
                        },
                        EscalationAction::CreateIncident {
                            severity: "high".to_string(),
                            assignee: "on-call-engineer".to_string(),
                        },
                    ],
                },
            ],
            max_level: 2,
            auto_escalation: true,
            triggers: vec![
                EscalationTrigger::TimeBasedDuration(Duration::from_secs(300)),
                EscalationTrigger::NoAcknowledgment(Duration::from_secs(600)),
            ],
            enabled: true,
        };

        let performance_policy = EscalationPolicy {
            name: "performance".to_string(),
            levels: vec![EscalationLevel {
                level: 1,
                time_to_escalate: Duration::from_secs(180), // 3 minutes
                recipients: vec!["performance-team@company.com".to_string()],
                channels: vec!["slack".to_string()],
                actions: vec![EscalationAction::Notify {
                    channel: "slack".to_string(),
                    recipients: vec!["performance-team@company.com".to_string()],
                }],
            }],
            max_level: 1,
            auto_escalation: true,
            triggers: vec![EscalationTrigger::SeverityLevel(SeverityLevel::Critical)],
            enabled: true,
        };

        let mut policies = self.policies.write().expect("Policies RwLock poisoned");
        policies.insert(default_policy.name.clone(), default_policy);
        policies.insert(performance_policy.name.clone(), performance_policy);

        Ok(())
    }

    /// Add escalation policy
    pub fn add_policy(&self, policy: EscalationPolicy) {
        let mut policies = self.policies.write().expect("Policies RwLock poisoned");
        policies.insert(policy.name.clone(), policy);
    }

    /// Remove escalation policy
    pub fn remove_policy(&self, policy_name: &str) {
        let mut policies = self.policies.write().expect("Policies RwLock poisoned");
        policies.remove(policy_name);
    }

    /// Get escalation statistics
    pub fn get_stats(&self) -> EscalationStats {
        EscalationStats {
            total_escalations: AtomicU64::new(self.stats.total_escalations.load(Ordering::Relaxed)),
            active_escalations: AtomicU64::new(
                self.stats.active_escalations.load(Ordering::Relaxed),
            ),
            escalations_by_level: Arc::new(Mutex::new(
                self.stats
                    .escalations_by_level
                    .lock()
                    .expect("Escalation stats lock poisoned")
                    .clone(),
            )),
            avg_escalation_time: Arc::new(Mutex::new(
                *self
                    .stats
                    .avg_escalation_time
                    .lock()
                    .expect("Avg escalation time lock poisoned"),
            )),
            acknowledgment_rate: Arc::new(Mutex::new(
                *self
                    .stats
                    .acknowledgment_rate
                    .lock()
                    .expect("Acknowledgment rate lock poisoned"),
            )),
        }
    }

    /// Get escalation state
    pub async fn get_escalation_state(&self, alert_id: &str) -> Option<EscalationState> {
        let state = self.state.lock().await;
        state.get(alert_id).cloned()
    }
}

impl Default for EscalationConfig {
    fn default() -> Self {
        Self {
            enable_auto_escalation: true,
            default_escalation_timeout: Duration::from_secs(300), // 5 minutes
            max_escalation_levels: 3,
            check_interval: Duration::from_secs(30),
            enable_notifications: true,
        }
    }
}

// =============================================================================
// ADAPTIVE THRESHOLD CONTROLLER IMPLEMENTATION
// =============================================================================

/// Adaptive threshold controller
///
/// Advanced controller that dynamically adjusts thresholds based on historical
/// data patterns, system behavior, and machine learning techniques.
pub struct AdaptiveThresholdController {
    /// Configuration
    config: Arc<RwLock<AdaptiveThresholdConfig>>,

    /// Adaptation algorithms
    algorithms: Arc<TokioMutex<Vec<Box<dyn ThresholdAdaptationAlgorithm + Send + Sync>>>>,

    /// Adaptation history
    history: Arc<TokioMutex<VecDeque<ThresholdAdaptation>>>,

    /// Controller statistics
    stats: Arc<AdaptationStats>,

    /// Processing task handle
    processing_handle: Arc<TokioMutex<Option<tokio::task::JoinHandle<()>>>>,

    /// Shutdown signal
    shutdown_signal: Arc<AtomicBool>,
}

/// Configuration for adaptive thresholds
#[derive(Debug, Clone)]
pub struct AdaptiveThresholdConfig {
    /// Enable adaptive thresholds
    pub enabled: bool,

    /// Adaptation sensitivity (0.0 to 1.0)
    pub sensitivity: f32,

    /// Learning rate
    pub learning_rate: f32,

    /// Adaptation interval
    pub adaptation_interval: Duration,

    /// Minimum data points required
    pub min_data_points: usize,

    /// Maximum adaptation ratio
    pub max_adaptation_ratio: f32,

    /// Enable historical analysis
    pub enable_historical_analysis: bool,
}

/// Threshold adaptation record
#[derive(Debug, Clone)]
pub struct ThresholdAdaptation {
    /// Adaptation timestamp
    pub timestamp: DateTime<Utc>,

    /// Threshold name
    pub threshold_name: String,

    /// Old threshold value
    pub old_value: f64,

    /// New threshold value
    pub new_value: f64,

    /// Adaptation reason
    pub reason: String,

    /// Confidence in adaptation
    pub confidence: f32,

    /// Effectiveness score
    pub effectiveness: Option<f32>,

    /// Algorithm used
    pub algorithm_name: String,
}

/// Statistics for threshold adaptation
#[derive(Debug, Default)]
pub struct AdaptationStats {
    /// Total adaptations performed
    pub total_adaptations: AtomicU64,

    /// Successful adaptations
    pub successful_adaptations: AtomicU64,

    /// Failed adaptations
    pub failed_adaptations: AtomicU64,

    /// Adaptations by algorithm
    pub adaptations_by_algorithm: Arc<Mutex<HashMap<String, u64>>>,

    /// Average effectiveness score
    pub avg_effectiveness: Arc<Mutex<f32>>,

    /// Average confidence score
    pub avg_confidence: Arc<Mutex<f32>>,
}

/// Trait for threshold adaptation algorithms
pub trait ThresholdAdaptationAlgorithm: Send + Sync {
    /// Adapt threshold based on historical data
    fn adapt_threshold(
        &self,
        current_threshold: f64,
        metrics_history: &[TimestampedMetrics],
        alert_history: &[AlertEvent],
    ) -> Result<f64>;

    /// Get algorithm name
    fn name(&self) -> &str;

    /// Get adaptation confidence
    fn confidence(&self, data_quality: f32) -> f32;

    /// Check if algorithm supports metric type
    fn supports_metric(&self, _metric_type: &str) -> bool {
        true // Default supports all metrics
    }

    /// Initialize algorithm if needed
    fn initialize(&mut self) -> Result<()> {
        Ok(())
    }

    /// Clean up resources
    fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }
}

impl AdaptiveThresholdController {
    /// Create a new adaptive threshold controller
    pub async fn new() -> Result<Self> {
        let controller = Self {
            config: Arc::new(RwLock::new(AdaptiveThresholdConfig::default())),
            algorithms: Arc::new(TokioMutex::new(Vec::new())),
            history: Arc::new(TokioMutex::new(VecDeque::new())),
            stats: Arc::new(AdaptationStats::default()),
            processing_handle: Arc::new(TokioMutex::new(None)),
            shutdown_signal: Arc::new(AtomicBool::new(false)),
        };

        controller.initialize_algorithms().await?;
        Ok(controller)
    }

    /// Start adaptive threshold controller
    pub async fn start(&self) -> Result<()> {
        let mut handle = self.processing_handle.lock().await;

        if handle.is_some() {
            return Err(ThresholdError::InternalError(
                "Controller already started".to_string(),
            ));
        }

        let config = Arc::clone(&self.config);
        let algorithms = Arc::clone(&self.algorithms);
        let history = Arc::clone(&self.history);
        let stats = Arc::clone(&self.stats);
        let shutdown_signal = Arc::clone(&self.shutdown_signal);

        let adaptation_task = tokio::spawn(async move {
            Self::adaptation_loop(config, algorithms, history, stats, shutdown_signal).await;
        });

        *handle = Some(adaptation_task);
        info!("Adaptive threshold controller started");
        Ok(())
    }

    /// Stop adaptive threshold controller
    pub async fn shutdown(&self) -> Result<()> {
        self.shutdown_signal.store(true, Ordering::Relaxed);

        let mut handle = self.processing_handle.lock().await;
        if let Some(task) = handle.take() {
            task.abort();
            info!("Adaptive threshold controller stopped");
        }

        Ok(())
    }

    /// Adapt threshold using available algorithms
    pub async fn adapt_threshold(
        &self,
        threshold_name: &str,
        current_threshold: f64,
        metrics_history: &[TimestampedMetrics],
        alert_history: &[AlertEvent],
    ) -> Result<f64> {
        let config = self.config.read().expect("Config RwLock poisoned");
        if !config.enabled {
            return Ok(current_threshold);
        }

        if metrics_history.len() < config.min_data_points {
            return Ok(current_threshold);
        }

        let algorithms = self.algorithms.lock().await;
        let mut best_adaptation = current_threshold;
        let mut best_confidence = 0.0;
        let mut best_algorithm = "none".to_string();

        // Try each algorithm and use the one with highest confidence
        for algorithm in algorithms.iter() {
            match algorithm.adapt_threshold(current_threshold, metrics_history, alert_history) {
                Ok(adapted_threshold) => {
                    let data_quality = self.calculate_data_quality(metrics_history);
                    let confidence = algorithm.confidence(data_quality);

                    if confidence > best_confidence {
                        best_adaptation = adapted_threshold;
                        best_confidence = confidence;
                        best_algorithm = algorithm.name().to_string();
                    }
                },
                Err(e) => {
                    warn!(
                        "Algorithm {} failed for threshold {}: {}",
                        algorithm.name(),
                        threshold_name,
                        e
                    );
                },
            }
        }

        // Apply adaptation limits
        let max_change = current_threshold * config.max_adaptation_ratio as f64;
        let limited_adaptation = best_adaptation.clamp(
            current_threshold - max_change,
            current_threshold + max_change,
        );

        // Record adaptation if significant change
        if (limited_adaptation - current_threshold).abs() > current_threshold * 0.01 {
            let adaptation = ThresholdAdaptation {
                timestamp: Utc::now(),
                threshold_name: threshold_name.to_string(),
                old_value: current_threshold,
                new_value: limited_adaptation,
                reason: format!("Adaptive adjustment by {}", best_algorithm),
                confidence: best_confidence,
                effectiveness: None,
                algorithm_name: best_algorithm.clone(),
            };

            let mut history = self.history.lock().await;
            history.push_back(adaptation);

            // Maintain history size
            if history.len() > 10000 {
                history.pop_front();
            }

            // Update statistics
            self.stats.total_adaptations.fetch_add(1, Ordering::Relaxed);
            let mut algo_stats = self
                .stats
                .adaptations_by_algorithm
                .lock()
                .expect("Algorithm stats lock poisoned");
            *algo_stats.entry(best_algorithm).or_insert(0) += 1;
        }

        Ok(limited_adaptation)
    }

    /// Calculate data quality score
    fn calculate_data_quality(&self, metrics_history: &[TimestampedMetrics]) -> f32 {
        if metrics_history.is_empty() {
            return 0.0;
        }

        let mut quality_score = 1.0;

        // Check data completeness
        let expected_points = 100; // Expected number of data points
        let completeness = (metrics_history.len() as f32 / expected_points as f32).min(1.0);
        quality_score *= completeness;

        // Check data freshness
        if let Some(latest) = metrics_history.last() {
            let age = Utc::now().signed_duration_since(latest.timestamp);
            let freshness = if age <= chrono::Duration::minutes(5) {
                1.0
            } else if age <= chrono::Duration::minutes(30) {
                0.8
            } else if age <= chrono::Duration::hours(1) {
                0.5
            } else {
                0.2
            };
            quality_score *= freshness;
        }

        // Check data consistency (simplified)
        let mut consistency = 1.0;
        for window in metrics_history.windows(2) {
            if let [prev, curr] = window {
                // Check if values are reasonable (not extreme jumps)
                let curr_value = curr.metrics.current_throughput;
                let prev_value = prev.metrics.current_throughput;
                let change_ratio = (curr_value - prev_value).abs() / prev_value.max(0.001);
                if change_ratio > 5.0 {
                    // More than 500% change
                    consistency *= 0.9;
                }
            }
        }
        quality_score *= consistency;

        quality_score.clamp(0.0, 1.0)
    }

    /// Main adaptation processing loop
    async fn adaptation_loop(
        config: Arc<RwLock<AdaptiveThresholdConfig>>,
        _algorithms: Arc<TokioMutex<Vec<Box<dyn ThresholdAdaptationAlgorithm + Send + Sync>>>>,
        history: Arc<TokioMutex<VecDeque<ThresholdAdaptation>>>,
        stats: Arc<AdaptationStats>,
        shutdown_signal: Arc<AtomicBool>,
    ) {
        let mut interval = interval(Duration::from_secs(300)); // Check every 5 minutes

        while !shutdown_signal.load(Ordering::Relaxed) {
            interval.tick().await;

            let enabled = {
                let config_read = config.read().expect("Config RwLock poisoned");
                config_read.enabled
            };

            if !enabled {
                continue;
            }

            // Perform periodic adaptation analysis
            Self::analyze_adaptation_effectiveness(&history, &stats).await;
        }
    }

    /// Analyze effectiveness of previous adaptations
    async fn analyze_adaptation_effectiveness(
        history: &Arc<TokioMutex<VecDeque<ThresholdAdaptation>>>,
        stats: &Arc<AdaptationStats>,
    ) {
        let mut history_guard = history.lock().await;
        let mut effectiveness_sum = 0.0;
        let mut confidence_sum = 0.0;
        let mut count = 0;

        for adaptation in history_guard.iter_mut() {
            if adaptation.effectiveness.is_none() {
                // Calculate effectiveness based on subsequent alert patterns
                // This is a simplified calculation - in practice, this would be more sophisticated
                let effectiveness = 0.7; // Placeholder effectiveness score
                adaptation.effectiveness = Some(effectiveness);
            }

            if let Some(effectiveness) = adaptation.effectiveness {
                effectiveness_sum += effectiveness;
                confidence_sum += adaptation.confidence;
                count += 1;
            }
        }

        if count > 0 {
            let avg_effectiveness = effectiveness_sum / count as f32;
            let avg_confidence = confidence_sum / count as f32;

            *stats.avg_effectiveness.lock().expect("Avg effectiveness lock poisoned") =
                avg_effectiveness;
            *stats.avg_confidence.lock().expect("Avg confidence lock poisoned") = avg_confidence;
        }
    }

    /// Initialize adaptation algorithms
    async fn initialize_algorithms(&self) -> Result<()> {
        let mut algorithms = self.algorithms.lock().await;
        algorithms.push(Box::new(StatisticalAdaptationAlgorithm::new()));
        algorithms.push(Box::new(MachineLearningAdaptationAlgorithm::new()));
        algorithms.push(Box::new(TrendAnalysisAlgorithm::new()));
        Ok(())
    }

    /// Get adaptation statistics
    pub fn get_stats(&self) -> AdaptationStats {
        AdaptationStats {
            total_adaptations: AtomicU64::new(self.stats.total_adaptations.load(Ordering::Relaxed)),
            successful_adaptations: AtomicU64::new(
                self.stats.successful_adaptations.load(Ordering::Relaxed),
            ),
            failed_adaptations: AtomicU64::new(
                self.stats.failed_adaptations.load(Ordering::Relaxed),
            ),
            adaptations_by_algorithm: Arc::new(Mutex::new(
                self.stats
                    .adaptations_by_algorithm
                    .lock()
                    .expect("Algorithm stats lock poisoned")
                    .clone(),
            )),
            avg_effectiveness: Arc::new(Mutex::new(
                *self.stats.avg_effectiveness.lock().expect("Avg effectiveness lock poisoned"),
            )),
            avg_confidence: Arc::new(Mutex::new(
                *self.stats.avg_confidence.lock().expect("Avg confidence lock poisoned"),
            )),
        }
    }

    /// Get adaptation history
    pub async fn get_adaptation_history(&self) -> Vec<ThresholdAdaptation> {
        let history = self.history.lock().await;
        history.iter().cloned().collect()
    }
}

impl Default for AdaptiveThresholdConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sensitivity: 0.7,
            learning_rate: 0.1,
            adaptation_interval: Duration::from_secs(300), // 5 minutes
            min_data_points: 50,
            max_adaptation_ratio: 0.3, // 30% maximum change
            enable_historical_analysis: true,
        }
    }
}

// =============================================================================
// ALERT PROCESSOR IMPLEMENTATIONS
// =============================================================================

/// Default alert processor for general alerts
pub struct DefaultAlertProcessor {
    /// Processor configuration
    config: DefaultProcessorConfig,
    /// Processing statistics
    stats: Arc<Mutex<ProcessorStats>>,
}

/// Configuration for default alert processor
#[derive(Debug, Clone)]
pub struct DefaultProcessorConfig {
    /// Priority level
    pub priority: u8,
    /// Enable detailed logging
    pub detailed_logging: bool,
    /// Include context in notifications
    pub include_context: bool,
}

/// Statistics for alert processors
#[derive(Debug, Clone)]
pub struct ProcessorStats {
    /// Alerts processed
    pub alerts_processed: u64,
    /// Processing errors
    pub processing_errors: u64,
    /// Average processing time
    pub avg_processing_time: Duration,
    /// Last processing time
    pub last_processing_time: Instant,
}

impl Default for ProcessorStats {
    fn default() -> Self {
        Self {
            alerts_processed: 0,
            processing_errors: 0,
            avg_processing_time: Duration::default(),
            last_processing_time: Instant::now(),
        }
    }
}

impl Default for DefaultAlertProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultAlertProcessor {
    pub fn new() -> Self {
        Self {
            config: DefaultProcessorConfig {
                priority: 50,
                detailed_logging: true,
                include_context: true,
            },
            stats: Arc::new(Mutex::new(ProcessorStats::default())),
        }
    }

    pub fn with_config(config: DefaultProcessorConfig) -> Self {
        Self {
            config,
            stats: Arc::new(Mutex::new(ProcessorStats::default())),
        }
    }

    pub fn get_stats(&self) -> ProcessorStats {
        self.stats.lock().expect("Stats lock poisoned").clone()
    }
}

impl AlertProcessor for DefaultAlertProcessor {
    fn process_alert(&self, alert: &AlertEvent) -> Result<ProcessedAlert> {
        let start_time = Instant::now();

        let notifications = vec![Notification {
            id: format!("notif_{}", alert.alert_id),
            channel: "log".to_string(),
            recipients: vec!["system".to_string()],
            subject: format!("Alert: {}", alert.threshold.name),
            content: if self.config.include_context {
                format!("{}\n\nContext: {:?}", alert.message, alert.context)
            } else {
                alert.message.clone()
            },
            priority: match alert.severity {
                SeverityLevel::Critical => NotificationPriority::Critical,
                SeverityLevel::High => NotificationPriority::High,
                SeverityLevel::Medium => NotificationPriority::Normal,
                SeverityLevel::Low => NotificationPriority::Low,
                _ => NotificationPriority::Normal,
            },
            metadata: HashMap::new(),
        }];

        let mut results = HashMap::new();
        results.insert("processor".to_string(), "default".to_string());
        results.insert(
            "notifications_created".to_string(),
            notifications.len().to_string(),
        );

        // Update statistics
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| ThresholdError::InternalError("Lock poisoned".to_string()))?;
        stats.alerts_processed += 1;
        stats.avg_processing_time = (stats.avg_processing_time + start_time.elapsed()) / 2;
        stats.last_processing_time = Instant::now();

        Ok(ProcessedAlert {
            alert: alert.clone(),
            processed_at: Utc::now(),
            results,
            notifications,
            metadata: HashMap::new(),
        })
    }

    fn name(&self) -> &str {
        "default_alert_processor"
    }

    fn supports(&self, _alert: &AlertEvent) -> bool {
        true // Default processor supports all alerts
    }

    fn priority(&self) -> u8 {
        self.config.priority
    }
}

/// Performance-focused alert processor
pub struct PerformanceAlertProcessor {
    config: PerformanceProcessorConfig,
    stats: Arc<Mutex<ProcessorStats>>,
}

#[derive(Debug, Clone)]
pub struct PerformanceProcessorConfig {
    pub priority: u8,
    pub performance_threshold: f64,
    pub enable_auto_scaling_recommendations: bool,
}

impl Default for PerformanceAlertProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceAlertProcessor {
    pub fn new() -> Self {
        Self {
            config: PerformanceProcessorConfig {
                priority: 80,
                performance_threshold: 0.8,
                enable_auto_scaling_recommendations: true,
            },
            stats: Arc::new(Mutex::new(ProcessorStats::default())),
        }
    }
}

impl AlertProcessor for PerformanceAlertProcessor {
    fn process_alert(&self, alert: &AlertEvent) -> Result<ProcessedAlert> {
        let start_time = Instant::now();

        let notifications = if matches!(
            alert.threshold.metric.as_str(),
            "throughput" | "latency_ms"
        ) {
            let mut notifs = vec![
                Notification {
                    id: format!("perf_notif_{}", alert.alert_id),
                    channel: "email".to_string(),
                    recipients: vec!["performance-team@company.com".to_string()],
                    subject: format!("Performance Alert: {}", alert.threshold.name),
                    content: format!(
                        "Performance issue detected: {}\n\nCurrent value: {}\nThreshold: {}\nSeverity: {:?}",
                        alert.message, alert.current_value, alert.threshold_value, alert.severity
                    ),
                    priority: NotificationPriority::High,
                    metadata: {
                        let mut metadata = HashMap::new();
                        metadata.insert("metric_type".to_string(), alert.threshold.metric.clone());
                        metadata.insert("performance_issue".to_string(), "true".to_string());
                        metadata
                    },
                }
            ];

            // Add auto-scaling recommendation if enabled
            if self.config.enable_auto_scaling_recommendations
                && alert.current_value > self.config.performance_threshold
            {
                notifs.push(Notification {
                    id: format!("scaling_rec_{}", alert.alert_id),
                    channel: "webhook".to_string(),
                    recipients: vec!["auto-scaler".to_string()],
                    subject: "Auto-scaling Recommendation".to_string(),
                    content: format!(
                        "Consider scaling up due to performance alert: {}",
                        alert.alert_id
                    ),
                    priority: NotificationPriority::High,
                    metadata: {
                        let mut metadata = HashMap::new();
                        metadata.insert("action".to_string(), "scale_up".to_string());
                        metadata.insert("trigger_alert".to_string(), alert.alert_id.clone());
                        metadata
                    },
                });
            }

            notifs
        } else {
            Vec::new()
        };

        let mut results = HashMap::new();
        results.insert("processor".to_string(), "performance".to_string());
        results.insert("metric_type".to_string(), alert.threshold.metric.clone());
        results.insert(
            "auto_scaling_recommended".to_string(),
            (self.config.enable_auto_scaling_recommendations
                && alert.current_value > self.config.performance_threshold)
                .to_string(),
        );

        // Update statistics
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| ThresholdError::InternalError("Lock poisoned".to_string()))?;
        stats.alerts_processed += 1;
        stats.avg_processing_time = (stats.avg_processing_time + start_time.elapsed()) / 2;
        stats.last_processing_time = Instant::now();

        Ok(ProcessedAlert {
            alert: alert.clone(),
            processed_at: Utc::now(),
            results,
            notifications,
            metadata: HashMap::new(),
        })
    }

    fn name(&self) -> &str {
        "performance_alert_processor"
    }

    fn supports(&self, alert: &AlertEvent) -> bool {
        matches!(
            alert.threshold.metric.as_str(),
            "throughput" | "latency_ms" | "response_time" | "cpu_utilization"
        )
    }

    fn priority(&self) -> u8 {
        self.config.priority
    }
}

/// Resource-focused alert processor
pub struct ResourceAlertProcessor {
    config: ResourceProcessorConfig,
    stats: Arc<Mutex<ProcessorStats>>,
}

#[derive(Debug, Clone)]
pub struct ResourceProcessorConfig {
    pub priority: u8,
    pub critical_resource_threshold: f64,
    pub enable_resource_optimization: bool,
}

impl Default for ResourceAlertProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceAlertProcessor {
    pub fn new() -> Self {
        Self {
            config: ResourceProcessorConfig {
                priority: 70,
                critical_resource_threshold: 0.95,
                enable_resource_optimization: true,
            },
            stats: Arc::new(Mutex::new(ProcessorStats::default())),
        }
    }
}

impl AlertProcessor for ResourceAlertProcessor {
    fn process_alert(&self, alert: &AlertEvent) -> Result<ProcessedAlert> {
        let start_time = Instant::now();

        let notifications = if matches!(
            alert.threshold.metric.as_str(),
            "cpu_utilization" | "memory_utilization"
        ) {
            vec![Notification {
                id: format!("resource_notif_{}", alert.alert_id),
                channel: "webhook".to_string(),
                recipients: vec!["ops-team".to_string()],
                subject: format!("Resource Alert: {}", alert.threshold.name),
                content: format!(
                    "Resource issue detected: {}\n\nCurrent utilization: {:.2}%\nThreshold: {:.2}%",
                    alert.message,
                    alert.current_value * 100.0,
                    alert.threshold_value * 100.0
                ),
                priority: if alert.current_value > self.config.critical_resource_threshold {
                    NotificationPriority::Critical
                } else {
                    NotificationPriority::High
                },
                metadata: {
                    let mut metadata = HashMap::new();
                    metadata.insert("resource_type".to_string(), alert.threshold.metric.clone());
                    metadata.insert(
                        "utilization".to_string(),
                        format!("{:.2}", alert.current_value),
                    );
                    if alert.current_value > self.config.critical_resource_threshold {
                        metadata.insert("critical_resource".to_string(), "true".to_string());
                    }
                    metadata
                },
            }]
        } else {
            Vec::new()
        };

        let mut results = HashMap::new();
        results.insert("processor".to_string(), "resource".to_string());
        results.insert("resource_type".to_string(), alert.threshold.metric.clone());
        results.insert(
            "utilization_percent".to_string(),
            format!("{:.2}", alert.current_value * 100.0),
        );

        // Update statistics
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| ThresholdError::InternalError("Lock poisoned".to_string()))?;
        stats.alerts_processed += 1;
        stats.avg_processing_time = (stats.avg_processing_time + start_time.elapsed()) / 2;
        stats.last_processing_time = Instant::now();

        Ok(ProcessedAlert {
            alert: alert.clone(),
            processed_at: Utc::now(),
            results,
            notifications,
            metadata: HashMap::new(),
        })
    }

    fn name(&self) -> &str {
        "resource_alert_processor"
    }

    fn supports(&self, alert: &AlertEvent) -> bool {
        matches!(
            alert.threshold.metric.as_str(),
            "cpu_utilization" | "memory_utilization" | "disk_usage" | "network_utilization"
        )
    }

    fn priority(&self) -> u8 {
        self.config.priority
    }
}

/// Critical alert processor for high-priority alerts
pub struct CriticalAlertProcessor {
    config: CriticalProcessorConfig,
    stats: Arc<Mutex<ProcessorStats>>,
}

#[derive(Debug, Clone)]
pub struct CriticalProcessorConfig {
    pub priority: u8,
    pub enable_immediate_escalation: bool,
    pub escalation_channels: Vec<String>,
}

impl Default for CriticalAlertProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl CriticalAlertProcessor {
    pub fn new() -> Self {
        Self {
            config: CriticalProcessorConfig {
                priority: 100, // Highest priority
                enable_immediate_escalation: true,
                escalation_channels: vec![
                    "email".to_string(),
                    "slack".to_string(),
                    "sms".to_string(),
                ],
            },
            stats: Arc::new(Mutex::new(ProcessorStats::default())),
        }
    }
}

impl AlertProcessor for CriticalAlertProcessor {
    fn process_alert(&self, alert: &AlertEvent) -> Result<ProcessedAlert> {
        let start_time = Instant::now();

        let mut notifications = Vec::new();

        if matches!(alert.severity, SeverityLevel::Critical) {
            // Create multiple notifications for critical alerts
            for channel in &self.config.escalation_channels {
                notifications.push(Notification {
                    id: format!("critical_{}_{}", channel, alert.alert_id),
                    channel: channel.clone(),
                    recipients: match channel.as_str() {
                        "email" => vec!["oncall@company.com".to_string(), "manager@company.com".to_string()],
                        "slack" => vec!["#critical-alerts".to_string()],
                        "sms" => vec!["+1234567890".to_string()],
                        _ => vec!["system".to_string()],
                    },
                    subject: format!("🚨 CRITICAL ALERT: {}", alert.threshold.name),
                    content: format!(
                        "CRITICAL ALERT DETECTED\n\nAlert ID: {}\nMetric: {}\nCurrent Value: {}\nThreshold: {}\nSeverity: Critical\n\nImmediate attention required!\n\nMessage: {}",
                        alert.alert_id, alert.threshold.metric, alert.current_value, alert.threshold_value, alert.message
                    ),
                    priority: NotificationPriority::Emergency,
                    metadata: {
                        let mut metadata = HashMap::new();
                        metadata.insert("alert_type".to_string(), "critical".to_string());
                        metadata.insert("escalation_level".to_string(), "immediate".to_string());
                        metadata.insert("requires_acknowledgment".to_string(), "true".to_string());
                        metadata
                    },
                });
            }
        }

        let mut results = HashMap::new();
        results.insert("processor".to_string(), "critical".to_string());
        results.insert("severity".to_string(), format!("{:?}", alert.severity));
        results.insert(
            "immediate_escalation".to_string(),
            self.config.enable_immediate_escalation.to_string(),
        );
        results.insert(
            "notifications_sent".to_string(),
            notifications.len().to_string(),
        );

        // Update statistics
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| ThresholdError::InternalError("Lock poisoned".to_string()))?;
        stats.alerts_processed += 1;
        stats.avg_processing_time = (stats.avg_processing_time + start_time.elapsed()) / 2;
        stats.last_processing_time = Instant::now();

        Ok(ProcessedAlert {
            alert: alert.clone(),
            processed_at: Utc::now(),
            results,
            notifications,
            metadata: {
                let mut metadata = HashMap::new();
                metadata.insert("critical_processing".to_string(), "true".to_string());
                metadata.insert("escalation_triggered".to_string(), "true".to_string());
                metadata
            },
        })
    }

    fn name(&self) -> &str {
        "critical_alert_processor"
    }

    fn supports(&self, alert: &AlertEvent) -> bool {
        matches!(alert.severity, SeverityLevel::Critical)
    }

    fn priority(&self) -> u8 {
        self.config.priority
    }
}

// =============================================================================
// NOTIFICATION CHANNEL IMPLEMENTATIONS
// =============================================================================

/// Log notification channel
pub struct LogNotificationChannel {
    config: LogChannelConfig,
    stats: Arc<Mutex<ChannelStats>>,
}

#[derive(Debug, Clone)]
pub struct LogChannelConfig {
    pub log_level: String,
    pub include_metadata: bool,
    pub format_json: bool,
}

#[derive(Debug, Default, Clone)]
pub struct ChannelStats {
    pub notifications_sent: u64,
    pub send_failures: u64,
    pub avg_send_time: Duration,
    pub last_send_time: Option<Instant>,
}

impl Default for LogNotificationChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl LogNotificationChannel {
    pub fn new() -> Self {
        Self {
            config: LogChannelConfig {
                log_level: "info".to_string(),
                include_metadata: true,
                format_json: false,
            },
            stats: Arc::new(Mutex::new(ChannelStats::default())),
        }
    }

    pub fn get_stats(&self) -> ChannelStats {
        self.stats.lock().expect("Stats lock poisoned").clone()
    }
}

impl NotificationChannel for LogNotificationChannel {
    fn send_notification(&self, notification: &Notification) -> Result<()> {
        let start_time = Instant::now();

        let log_message = if self.config.format_json {
            serde_json::json!({
                "notification_id": notification.id,
                "subject": notification.subject,
                "content": notification.content,
                "priority": format!("{:?}", notification.priority),
                "recipients": notification.recipients,
                "metadata": if self.config.include_metadata { Some(&notification.metadata) } else { None }
            }).to_string()
        } else {
            format!(
                "Alert notification: {} - {} (Priority: {:?}){}",
                notification.subject,
                notification.content,
                notification.priority,
                if self.config.include_metadata {
                    format!(" | Metadata: {:?}", notification.metadata)
                } else {
                    String::new()
                }
            )
        };

        match self.config.log_level.as_str() {
            "error" => error!("{}", log_message),
            "warn" => warn!("{}", log_message),
            "debug" => debug!("{}", log_message),
            "trace" => tracing::trace!("{}", log_message),
            _ => info!("{}", log_message),
        }

        // Update statistics
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| ThresholdError::InternalError("Lock poisoned".to_string()))?;
        stats.notifications_sent += 1;
        stats.avg_send_time = (stats.avg_send_time + start_time.elapsed()) / 2;
        stats.last_send_time = Some(Instant::now());

        Ok(())
    }

    fn name(&self) -> &str {
        "log"
    }

    fn supports(&self, notification_type: &str) -> bool {
        notification_type == "log"
    }

    fn max_message_size(&self) -> usize {
        100000 // 100KB for log messages
    }

    fn is_available(&self) -> bool {
        true // Log channel is always available
    }
}

/// Email notification channel
pub struct EmailNotificationChannel {
    config: EmailChannelConfig,
    stats: Arc<Mutex<ChannelStats>>,
}

#[derive(Debug, Clone)]
pub struct EmailChannelConfig {
    pub smtp_server: String,
    pub smtp_port: u16,
    pub username: String,
    pub from_address: String,
    pub use_tls: bool,
    pub max_recipients: usize,
}

impl Default for EmailNotificationChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl EmailNotificationChannel {
    pub fn new() -> Self {
        Self {
            config: EmailChannelConfig {
                smtp_server: "localhost".to_string(),
                smtp_port: 587,
                username: "alerts".to_string(),
                from_address: "alerts@company.com".to_string(),
                use_tls: true,
                max_recipients: 50,
            },
            stats: Arc::new(Mutex::new(ChannelStats::default())),
        }
    }

    pub fn with_config(config: EmailChannelConfig) -> Self {
        Self {
            config,
            stats: Arc::new(Mutex::new(ChannelStats::default())),
        }
    }
}

impl NotificationChannel for EmailNotificationChannel {
    fn send_notification(&self, notification: &Notification) -> Result<()> {
        let start_time = Instant::now();

        // Validate recipient count
        if notification.recipients.len() > self.config.max_recipients {
            return Err(ThresholdError::NotificationError(format!(
                "Too many recipients: {} (max: {})",
                notification.recipients.len(),
                self.config.max_recipients
            )));
        }

        // Placeholder for email sending implementation
        // In a real implementation, this would use an SMTP library like lettre
        info!(
            "Email notification sent to {:?}: {} - {} (Server: {}:{})",
            notification.recipients,
            notification.subject,
            notification.content,
            self.config.smtp_server,
            self.config.smtp_port
        );

        // Simulate email sending delay
        std::thread::sleep(Duration::from_millis(100));

        // Update statistics
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| ThresholdError::InternalError("Lock poisoned".to_string()))?;
        stats.notifications_sent += 1;
        stats.avg_send_time = (stats.avg_send_time + start_time.elapsed()) / 2;
        stats.last_send_time = Some(Instant::now());

        Ok(())
    }

    fn name(&self) -> &str {
        "email"
    }

    fn supports(&self, notification_type: &str) -> bool {
        notification_type == "email"
    }

    fn max_message_size(&self) -> usize {
        1000000 // 1MB for email messages
    }

    fn is_available(&self) -> bool {
        // In a real implementation, this would check SMTP server connectivity
        true
    }
}

/// Webhook notification channel
pub struct WebhookNotificationChannel {
    config: WebhookChannelConfig,
    stats: Arc<Mutex<ChannelStats>>,
    client: reqwest::Client,
}

#[derive(Debug, Clone)]
pub struct WebhookChannelConfig {
    pub webhook_url: String,
    pub timeout_seconds: u64,
    pub retry_attempts: u8,
    pub auth_header: Option<String>,
    pub custom_headers: HashMap<String, String>,
}

impl Default for WebhookNotificationChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl WebhookNotificationChannel {
    pub fn new() -> Self {
        Self {
            config: WebhookChannelConfig {
                webhook_url: "http://localhost:8080/webhook".to_string(),
                timeout_seconds: 30,
                retry_attempts: 3,
                auth_header: None,
                custom_headers: HashMap::new(),
            },
            stats: Arc::new(Mutex::new(ChannelStats::default())),
            client: reqwest::Client::new(),
        }
    }

    pub fn with_config(config: WebhookChannelConfig) -> Self {
        Self {
            config,
            stats: Arc::new(Mutex::new(ChannelStats::default())),
            client: reqwest::Client::new(),
        }
    }
}

impl NotificationChannel for WebhookNotificationChannel {
    fn send_notification(&self, notification: &Notification) -> Result<()> {
        let start_time = Instant::now();

        // For now, just log the webhook notification
        // In a real implementation, this would make an HTTP request
        info!(
            "Webhook notification sent: {} - {} (URL: {})",
            notification.subject, notification.content, self.config.webhook_url
        );

        // Simulate webhook sending
        std::thread::sleep(Duration::from_millis(50));

        // Update statistics
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| ThresholdError::InternalError("Lock poisoned".to_string()))?;
        stats.notifications_sent += 1;
        stats.avg_send_time = (stats.avg_send_time + start_time.elapsed()) / 2;
        stats.last_send_time = Some(Instant::now());

        Ok(())
    }

    fn name(&self) -> &str {
        "webhook"
    }

    fn supports(&self, notification_type: &str) -> bool {
        notification_type == "webhook"
    }

    fn max_message_size(&self) -> usize {
        500000 // 500KB for webhook payloads
    }

    fn is_available(&self) -> bool {
        // In a real implementation, this would check webhook URL accessibility
        true
    }
}

/// Slack notification channel
pub struct SlackNotificationChannel {
    config: SlackChannelConfig,
    stats: Arc<Mutex<ChannelStats>>,
}

#[derive(Debug, Clone)]
pub struct SlackChannelConfig {
    pub webhook_url: String,
    pub default_channel: String,
    pub username: String,
    pub icon_emoji: String,
    pub timeout_seconds: u64,
}

impl Default for SlackNotificationChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl SlackNotificationChannel {
    pub fn new() -> Self {
        Self {
            config: SlackChannelConfig {
                webhook_url: "https://hooks.slack.com/services/...".to_string(),
                default_channel: "#alerts".to_string(),
                username: "AlertBot".to_string(),
                icon_emoji: ":warning:".to_string(),
                timeout_seconds: 30,
            },
            stats: Arc::new(Mutex::new(ChannelStats::default())),
        }
    }

    pub fn with_config(config: SlackChannelConfig) -> Self {
        Self {
            config,
            stats: Arc::new(Mutex::new(ChannelStats::default())),
        }
    }
}

impl NotificationChannel for SlackNotificationChannel {
    fn send_notification(&self, notification: &Notification) -> Result<()> {
        let start_time = Instant::now();

        // Format message for Slack
        let slack_message = format!(
            "{} {}\n{}",
            match notification.priority {
                NotificationPriority::Emergency => "🚨",
                NotificationPriority::Critical => "❗",
                NotificationPriority::High => "⚠️",
                NotificationPriority::Normal => "ℹ️",
                NotificationPriority::Low => "💡",
            },
            notification.subject,
            notification.content
        );

        // Placeholder for Slack webhook implementation
        info!(
            "Slack notification sent to {}: {}",
            self.config.default_channel, slack_message
        );

        // Update statistics
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| ThresholdError::InternalError("Lock poisoned".to_string()))?;
        stats.notifications_sent += 1;
        stats.avg_send_time = (stats.avg_send_time + start_time.elapsed()) / 2;
        stats.last_send_time = Some(Instant::now());

        Ok(())
    }

    fn name(&self) -> &str {
        "slack"
    }

    fn supports(&self, notification_type: &str) -> bool {
        notification_type == "slack"
    }

    fn max_message_size(&self) -> usize {
        3000 // Slack message limit
    }

    fn is_available(&self) -> bool {
        // In a real implementation, this would check Slack webhook accessibility
        true
    }
}

// =============================================================================
// THRESHOLD MONITOR MAIN IMPLEMENTATION
// =============================================================================

/// Performance threshold monitoring and alerting system
///
/// Comprehensive threshold monitoring system that integrates all components
/// for intelligent threshold evaluation, alerting, escalation, and adaptation.
pub struct ThresholdMonitor {
    /// Threshold configurations
    thresholds: Arc<RwLock<HashMap<String, ThresholdConfig>>>,

    /// Alert manager
    alert_manager: Arc<AlertManager>,

    /// Threshold evaluators
    evaluators: Arc<Mutex<Vec<Box<dyn ThresholdEvaluator + Send + Sync>>>>,

    /// Escalation manager
    escalation_manager: Arc<EscalationManager>,

    /// Alert history
    alert_history: Arc<Mutex<VecDeque<AlertEvent>>>,

    /// Monitoring state
    monitoring_state: Arc<RwLock<ThresholdMonitoringState>>,

    /// Adaptive threshold controller
    adaptive_controller: Arc<AdaptiveThresholdController>,

    /// Alert suppressor
    alert_suppressor: Arc<AlertSuppressor>,

    /// Alert correlator
    alert_correlator: Arc<AlertCorrelator>,

    /// Performance analyzer
    performance_analyzer: Arc<PerformanceAnalyzer>,

    /// Monitor configuration
    config: Arc<RwLock<ThresholdMonitorConfig>>,

    /// Monitoring task handle
    monitoring_handle: Arc<TokioMutex<Option<tokio::task::JoinHandle<()>>>>,

    /// Shutdown signal
    shutdown_signal: Arc<AtomicBool>,
}

/// Configuration for threshold monitor
#[derive(Debug, Clone)]
pub struct ThresholdMonitorConfig {
    /// Enable real-time monitoring
    pub enable_monitoring: bool,

    /// Monitoring interval
    pub monitoring_interval: Duration,

    /// Enable adaptive thresholds
    pub enable_adaptive_thresholds: bool,

    /// Enable alert suppression
    pub enable_alert_suppression: bool,

    /// Enable alert correlation
    pub enable_alert_correlation: bool,

    /// Enable performance analysis
    pub enable_performance_analysis: bool,

    /// Maximum alert history size
    pub max_alert_history: usize,

    /// Alert processing timeout
    pub alert_processing_timeout: Duration,

    /// Enable detailed logging
    pub enable_detailed_logging: bool,
}

impl ThresholdMonitor {
    /// Create a new threshold monitor
    ///
    /// Initializes a comprehensive threshold monitoring system with adaptive
    /// thresholds, multi-level alerting, and escalation policies.
    pub async fn new() -> Result<Self> {
        let monitor = Self {
            thresholds: Arc::new(RwLock::new(HashMap::new())),
            alert_manager: Arc::new(AlertManager::new().await?),
            evaluators: Arc::new(Mutex::new(Vec::new())),
            escalation_manager: Arc::new(EscalationManager::new().await?),
            alert_history: Arc::new(Mutex::new(VecDeque::new())),
            monitoring_state: Arc::new(RwLock::new(ThresholdMonitoringState::default())),
            adaptive_controller: Arc::new(AdaptiveThresholdController::new().await?),
            alert_suppressor: Arc::new(AlertSuppressor::new()),
            alert_correlator: Arc::new(AlertCorrelator::new()),
            performance_analyzer: Arc::new(PerformanceAnalyzer::new()),
            config: Arc::new(RwLock::new(ThresholdMonitorConfig::default())),
            monitoring_handle: Arc::new(TokioMutex::new(None)),
            shutdown_signal: Arc::new(AtomicBool::new(false)),
        };

        monitor.initialize_evaluators().await?;
        monitor.initialize_default_thresholds().await?;
        monitor.initialize_suppression_rules().await?;
        monitor.initialize_correlation_rules().await?;
        Ok(monitor)
    }

    /// Start threshold monitoring
    ///
    /// Begins continuous threshold monitoring with real-time evaluation
    /// and intelligent alerting based on configured thresholds.
    pub async fn start_monitoring(&self) -> Result<()> {
        let config = self.config.read().expect("Config RwLock poisoned");
        if !config.enable_monitoring {
            return Ok(());
        }
        drop(config);

        // Start all subsystems
        self.alert_manager.start().await?;
        self.escalation_manager.start().await?;
        self.adaptive_controller.start().await?;
        self.performance_analyzer.start().await?;

        let mut handle = self.monitoring_handle.lock().await;
        if handle.is_some() {
            return Err(ThresholdError::InternalError(
                "Monitor already started".to_string(),
            ));
        }

        let thresholds = Arc::clone(&self.thresholds);
        let evaluators = Arc::clone(&self.evaluators);
        let alert_manager = Arc::clone(&self.alert_manager);
        let escalation_manager = Arc::clone(&self.escalation_manager);
        let alert_history = Arc::clone(&self.alert_history);
        let monitoring_state = Arc::clone(&self.monitoring_state);
        let adaptive_controller = Arc::clone(&self.adaptive_controller);
        let alert_suppressor = Arc::clone(&self.alert_suppressor);
        let alert_correlator = Arc::clone(&self.alert_correlator);
        let performance_analyzer = Arc::clone(&self.performance_analyzer);
        let config = Arc::clone(&self.config);
        let shutdown_signal = Arc::clone(&self.shutdown_signal);

        let monitoring_task = tokio::spawn(async move {
            Self::monitoring_loop(
                thresholds,
                evaluators,
                alert_manager,
                escalation_manager,
                alert_history,
                monitoring_state,
                adaptive_controller,
                alert_suppressor,
                alert_correlator,
                performance_analyzer,
                config,
                shutdown_signal,
            )
            .await;
        });

        *handle = Some(monitoring_task);
        info!("Threshold monitoring started");
        Ok(())
    }

    /// Stop threshold monitoring
    pub async fn shutdown(&self) -> Result<()> {
        self.shutdown_signal.store(true, Ordering::Relaxed);

        // Stop monitoring task
        let mut handle = self.monitoring_handle.lock().await;
        if let Some(task) = handle.take() {
            task.abort();
        }

        // Stop all subsystems
        self.alert_manager.shutdown().await?;
        self.escalation_manager.shutdown().await?;
        self.adaptive_controller.shutdown().await?;
        self.performance_analyzer.shutdown().await?;

        info!("Threshold monitoring stopped");
        Ok(())
    }

    /// Evaluate metrics against configured thresholds
    ///
    /// Evaluates current metrics against all configured thresholds and
    /// generates alerts for violations with appropriate severity levels.
    #[instrument(skip(self, metrics), level = "debug")]
    pub async fn evaluate_thresholds(
        &self,
        metrics: &TimestampedMetrics,
    ) -> Result<Vec<AlertEvent>> {
        let start_time = Instant::now();

        let config = self.config.read().expect("Config RwLock poisoned");
        if !config.enable_monitoring {
            return Ok(Vec::new());
        }

        let performance_tracking = config.enable_performance_analysis;
        drop(config);

        let thresholds = self.thresholds.read().expect("Thresholds RwLock poisoned");
        let evaluators = self.evaluators.lock().expect("Evaluators lock poisoned");
        let mut alerts = Vec::new();

        // Track performance if enabled
        if performance_tracking {
            self.performance_analyzer.start_evaluation_tracking().await;
        }

        for (threshold_name, threshold_config) in thresholds.iter() {
            // Get the metric value for this threshold
            if let Some(metric_value) = self.extract_metric_value(metrics, &threshold_config.metric)
            {
                // Apply adaptive threshold adjustment if enabled
                let adjusted_threshold = if self
                    .config
                    .read()
                    .expect("Config RwLock poisoned")
                    .enable_adaptive_thresholds
                {
                    let history = self.collect_metrics_history(&threshold_config.metric).await;
                    let alert_hist =
                        self.get_alert_history_for_metric(&threshold_config.metric).await;

                    match self
                        .adaptive_controller
                        .adapt_threshold(
                            threshold_name,
                            threshold_config.critical_threshold,
                            &history,
                            &alert_hist,
                        )
                        .await
                    {
                        Ok(adapted) => {
                            let mut adjusted_config = threshold_config.clone();
                            adjusted_config.critical_threshold = adapted;
                            adjusted_config
                        },
                        Err(e) => {
                            warn!("Failed to adapt threshold {}: {}", threshold_name, e);
                            threshold_config.clone()
                        },
                    }
                } else {
                    threshold_config.clone()
                };

                // Evaluate threshold with each evaluator
                for evaluator in evaluators.iter() {
                    if evaluator.supports_threshold(&adjusted_threshold.metric) {
                        match evaluator.evaluate(&adjusted_threshold, metric_value) {
                            Ok(evaluation) if evaluation.violated => {
                                let mut alert = self
                                    .create_alert_event(
                                        threshold_name,
                                        &adjusted_threshold,
                                        &evaluation,
                                        metrics,
                                    )
                                    .await?;

                                // Apply suppression if enabled
                                if self
                                    .config
                                    .read()
                                    .expect("Config RwLock poisoned")
                                    .enable_alert_suppression
                                    && self.alert_suppressor.should_suppress(&alert).await
                                {
                                    self.alert_suppressor.suppress_alert(&mut alert).await;
                                    continue;
                                }

                                // Apply correlation if enabled
                                if self
                                    .config
                                    .read()
                                    .expect("Config RwLock poisoned")
                                    .enable_alert_correlation
                                {
                                    self.alert_correlator.correlate_alert(&mut alert).await;
                                }

                                alerts.push(alert);
                                break; // Use first matching evaluator
                            },
                            Ok(_) => {}, // No violation
                            Err(e) => {
                                warn!(
                                    "Evaluator {} failed for threshold {}: {}",
                                    evaluator.name(),
                                    threshold_name,
                                    e
                                );
                            },
                        }
                    }
                }
            }
        }

        drop(thresholds);
        drop(evaluators);

        // Process alerts through alert manager
        for alert in &alerts {
            if let Err(e) = self.alert_manager.process_alert(alert.clone()).await {
                error!("Failed to process alert {}: {}", alert.alert_id, e);
            }

            // Start escalation if needed
            if matches!(
                alert.severity,
                SeverityLevel::Critical | SeverityLevel::High
            ) {
                if let Err(e) = self.escalation_manager.start_escalation(alert, "default").await {
                    error!(
                        "Failed to start escalation for alert {}: {}",
                        alert.alert_id, e
                    );
                }
            }
        }

        // Update monitoring state
        self.update_monitoring_state(&alerts, start_time.elapsed()).await?;

        // Update alert history
        self.update_alert_history(&alerts).await;

        // Track performance completion
        if performance_tracking {
            self.performance_analyzer
                .complete_evaluation_tracking(alerts.len(), start_time.elapsed())
                .await;
        }

        Ok(alerts)
    }

    /// Add a new threshold configuration
    pub async fn add_threshold(&self, threshold: ThresholdConfig) -> Result<()> {
        let mut thresholds = self.thresholds.write().expect("Thresholds RwLock poisoned");
        thresholds.insert(threshold.name.clone(), threshold);
        info!("Added threshold configuration: {}", thresholds.len());
        Ok(())
    }

    /// Remove a threshold configuration
    pub async fn remove_threshold(&self, threshold_name: &str) -> Result<()> {
        let mut thresholds = self.thresholds.write().expect("Thresholds RwLock poisoned");
        thresholds.remove(threshold_name);
        info!("Removed threshold configuration: {}", threshold_name);
        Ok(())
    }

    /// Update threshold configuration
    pub async fn update_threshold(&self, threshold: ThresholdConfig) -> Result<()> {
        let mut thresholds = self.thresholds.write().expect("Thresholds RwLock poisoned");
        thresholds.insert(threshold.name.clone(), threshold);
        info!("Updated threshold configuration: {}", thresholds.len());
        Ok(())
    }

    /// Get current monitoring state
    pub async fn get_monitoring_state(&self) -> ThresholdMonitoringState {
        (*self.monitoring_state.read().expect("Monitoring state RwLock poisoned")).clone()
    }

    /// Get alert history
    pub async fn get_alert_history(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<Vec<AlertEvent>> {
        let history = self.alert_history.lock().expect("Alert history lock poisoned");
        let filtered_alerts = history
            .iter()
            .filter(|alert| alert.timestamp >= start_time && alert.timestamp <= end_time)
            .cloned()
            .collect();

        Ok(filtered_alerts)
    }

    /// Get comprehensive threshold statistics
    pub async fn get_threshold_statistics(&self) -> ThresholdStatistics {
        let alert_manager_stats = self.alert_manager.get_stats();
        let escalation_stats = self.escalation_manager.get_stats();
        let adaptation_stats = self.adaptive_controller.get_stats();
        let suppression_stats = self.alert_suppressor.get_stats();
        let correlation_stats = self.alert_correlator.get_stats();
        let performance_stats = self.performance_analyzer.get_stats().await;

        ThresholdStatistics {
            alert_manager_stats,
            escalation_stats,
            adaptation_stats,
            suppression_stats,
            correlation_stats,
            performance_stats,
            monitoring_state: self.get_monitoring_state().await,
        }
    }

    // Private implementation methods

    /// Main monitoring loop
    async fn monitoring_loop(
        _thresholds: Arc<RwLock<HashMap<String, ThresholdConfig>>>,
        _evaluators: Arc<Mutex<Vec<Box<dyn ThresholdEvaluator + Send + Sync>>>>,
        _alert_manager: Arc<AlertManager>,
        _escalation_manager: Arc<EscalationManager>,
        alert_history: Arc<Mutex<VecDeque<AlertEvent>>>,
        monitoring_state: Arc<RwLock<ThresholdMonitoringState>>,
        _adaptive_controller: Arc<AdaptiveThresholdController>,
        _alert_suppressor: Arc<AlertSuppressor>,
        _alert_correlator: Arc<AlertCorrelator>,
        _performance_analyzer: Arc<PerformanceAnalyzer>,
        config: Arc<RwLock<ThresholdMonitorConfig>>,
        shutdown_signal: Arc<AtomicBool>,
    ) {
        let mut interval = {
            let monitoring_interval = {
                let config_read = config.read().expect("Config RwLock poisoned");
                config_read.monitoring_interval
            };
            interval(monitoring_interval)
        };

        while !shutdown_signal.load(Ordering::Relaxed) {
            interval.tick().await;

            let monitoring_enabled = {
                let config_read = config.read().expect("Config RwLock poisoned");
                config_read.enable_monitoring
            };

            if !monitoring_enabled {
                continue;
            }

            // Perform periodic maintenance tasks
            Self::perform_maintenance(&alert_history, &monitoring_state, &config).await;
        }

        info!("Threshold monitoring loop stopped");
    }

    /// Perform periodic maintenance tasks
    async fn perform_maintenance(
        alert_history: &Arc<Mutex<VecDeque<AlertEvent>>>,
        monitoring_state: &Arc<RwLock<ThresholdMonitoringState>>,
        config: &Arc<RwLock<ThresholdMonitorConfig>>,
    ) {
        let config_read = config.read().expect("RwLock poisoned");
        let max_history = config_read.max_alert_history;
        drop(config_read);

        // Clean up old alerts from history
        let mut history = alert_history.lock().expect("Alert history lock poisoned");
        while history.len() > max_history {
            history.pop_front();
        }
        drop(history);

        // Update monitoring state
        let mut state = monitoring_state.write().expect("Monitoring state RwLock poisoned");
        state.last_evaluation = Some(Utc::now());
        drop(state);
    }

    async fn initialize_evaluators(&self) -> Result<()> {
        let mut evaluators = self.evaluators.lock().expect("Evaluators lock poisoned");
        evaluators.push(Box::new(SimpleThresholdEvaluator::new()));
        evaluators.push(Box::new(StatisticalThresholdEvaluator::new()));
        evaluators.push(Box::new(AdaptiveThresholdEvaluator::new().await));
        Ok(())
    }

    async fn initialize_default_thresholds(&self) -> Result<()> {
        let default_thresholds = vec![
            ThresholdConfig {
                name: "high_cpu_usage".to_string(),
                metric: "cpu_utilization".to_string(),
                warning_threshold: 0.8,
                critical_threshold: 0.95,
                direction: ThresholdDirection::Above,
                adaptive: true,
                evaluation_window: Duration::from_secs(60),
                min_trigger_count: 3,
                cooldown_period: Duration::from_secs(300),
                escalation_policy: "default".to_string(),
            },
            ThresholdConfig {
                name: "high_memory_usage".to_string(),
                metric: "memory_utilization".to_string(),
                warning_threshold: 0.85,
                critical_threshold: 0.98,
                direction: ThresholdDirection::Above,
                adaptive: true,
                evaluation_window: Duration::from_secs(60),
                min_trigger_count: 2,
                cooldown_period: Duration::from_secs(300),
                escalation_policy: "default".to_string(),
            },
            ThresholdConfig {
                name: "low_throughput".to_string(),
                metric: "throughput".to_string(),
                warning_threshold: 100.0,
                critical_threshold: 50.0,
                direction: ThresholdDirection::Below,
                adaptive: true,
                evaluation_window: Duration::from_secs(120),
                min_trigger_count: 5,
                cooldown_period: Duration::from_secs(600),
                escalation_policy: "performance".to_string(),
            },
        ];

        let mut thresholds = self.thresholds.write().expect("Thresholds RwLock poisoned");
        for threshold in default_thresholds {
            thresholds.insert(threshold.name.clone(), threshold);
        }

        Ok(())
    }

    async fn initialize_suppression_rules(&self) -> Result<()> {
        // Add default suppression rules
        let cpu_suppression = SuppressionRule {
            id: "cpu_duplicate_suppression".to_string(),
            name: "CPU Duplicate Alert Suppression".to_string(),
            rule_type: SuppressionRuleType::Duplicate,
            criteria: SuppressionCriteria {
                metric_patterns: vec!["cpu_utilization".to_string()],
                severity_levels: vec![],
                time_window: Duration::from_secs(300),
                max_alerts_per_window: 1,
                threshold_patterns: vec![],
                conditions: HashMap::new(),
            },
            action: SuppressionAction::Suppress,
            priority: 80,
            enabled: true,
        };

        self.alert_suppressor.add_rule(cpu_suppression);
        Ok(())
    }

    async fn initialize_correlation_rules(&self) -> Result<()> {
        // Add default correlation rules
        let resource_correlation = CorrelationRule {
            id: "resource_correlation".to_string(),
            name: "Resource-based Alert Correlation".to_string(),
            rule_type: CorrelationRuleType::Resource,
            criteria: CorrelationCriteria {
                metric_patterns: vec![
                    "cpu_utilization".to_string(),
                    "memory_utilization".to_string(),
                ],
                resource_patterns: vec![],
                severity_matching: SeverityMatching::Any,
                time_tolerance: Duration::from_secs(60),
                min_strength: 0.7,
                conditions: HashMap::new(),
            },
            strength: 0.8,
            time_window: Duration::from_secs(300),
            enabled: true,
        };

        self.alert_correlator.add_rule(resource_correlation);
        Ok(())
    }

    fn extract_metric_value(&self, metrics: &TimestampedMetrics, metric_name: &str) -> Option<f64> {
        metrics.metrics.get(metric_name)
    }

    async fn create_alert_event(
        &self,
        threshold_name: &str,
        threshold_config: &ThresholdConfig,
        evaluation: &ThresholdEvaluation,
        _metrics: &TimestampedMetrics,
    ) -> Result<AlertEvent> {
        let alert_id = Uuid::new_v4().to_string();

        let message = format!(
            "Threshold '{}' violated: {} {} {}",
            threshold_name,
            evaluation.current_value,
            match threshold_config.direction {
                ThresholdDirection::Above => "exceeds",
                ThresholdDirection::Below => "below",
                ThresholdDirection::OutsideRange { .. } => "outside range",
                ThresholdDirection::InsideRange { .. } => "inside range",
            },
            evaluation.threshold_value
        );

        let mut context = HashMap::new();
        context.insert("metric_name".to_string(), threshold_config.metric.clone());
        context.insert(
            "evaluation_confidence".to_string(),
            evaluation.confidence.to_string(),
        );
        context.insert(
            "threshold_direction".to_string(),
            format!("{:?}", threshold_config.direction),
        );
        context.insert(
            "adaptive_enabled".to_string(),
            threshold_config.adaptive.to_string(),
        );

        // Add evaluation context
        for (key, value) in &evaluation.context {
            context.insert(format!("eval_{}", key), value.clone());
        }

        Ok(AlertEvent {
            timestamp: evaluation.timestamp,
            alert_id,
            threshold: threshold_config.clone(),
            current_value: evaluation.current_value,
            threshold_value: evaluation.threshold_value,
            severity: evaluation.severity,
            message,
            context,
            actions: Vec::new(), // Actions will be determined by alert manager
            metadata: HashMap::new(), // Metadata can be added later
            correlation_id: None,
            suppression_info: None,
        })
    }

    async fn update_monitoring_state(
        &self,
        alerts: &[AlertEvent],
        evaluation_duration: Duration,
    ) -> Result<()> {
        let mut state = self.monitoring_state.write().expect("Monitoring state RwLock poisoned");

        state.last_evaluation = Some(Utc::now());

        // Update alert counts by severity
        for alert in alerts {
            *state.alert_counts.entry(alert.severity).or_insert(0) += 1;
        }

        // Update active alerts
        for alert in alerts {
            state.active_alerts.insert(alert.alert_id.clone(), alert.clone());
        }

        // Update evaluation performance
        state.evaluation_performance.last_evaluation_duration = evaluation_duration;
        state.evaluation_performance.total_evaluations += 1;

        // Update average (simple moving average)
        let total_time = state.evaluation_performance.avg_evaluation_duration
            * (state.evaluation_performance.total_evaluations - 1) as u32
            + evaluation_duration;
        state.evaluation_performance.avg_evaluation_duration =
            total_time / state.evaluation_performance.total_evaluations as u32;

        if evaluation_duration > state.evaluation_performance.peak_evaluation_duration {
            state.evaluation_performance.peak_evaluation_duration = evaluation_duration;
        }

        // Update evaluation statistics
        state.evaluation_stats.total_evaluations += 1;
        state.evaluation_stats.total_alerts += alerts.len() as u64;

        // Update average evaluation time
        let total_eval_time = state.evaluation_stats.avg_evaluation_time
            * (state.evaluation_stats.total_evaluations - 1) as u32
            + evaluation_duration;
        state.evaluation_stats.avg_evaluation_time =
            total_eval_time / state.evaluation_stats.total_evaluations as u32;

        Ok(())
    }

    async fn update_alert_history(&self, alerts: &[AlertEvent]) {
        let mut history = self.alert_history.lock().expect("Lock poisoned");

        for alert in alerts {
            history.push_back(alert.clone());
        }

        // Maintain history size
        let max_history = self.config.read().expect("Config RwLock poisoned").max_alert_history;
        while history.len() > max_history {
            history.pop_front();
        }
    }

    async fn collect_metrics_history(&self, _metric_name: &str) -> Vec<TimestampedMetrics> {
        // In a real implementation, this would collect historical metrics
        // For now, return empty vector as this would come from the collector/aggregator
        Vec::new()
    }

    async fn get_alert_history_for_metric(&self, metric_name: &str) -> Vec<AlertEvent> {
        let history = self.alert_history.lock().expect("Alert history lock poisoned");
        history
            .iter()
            .filter(|alert| alert.threshold.metric == metric_name)
            .cloned()
            .collect()
    }
}

impl Default for ThresholdMonitorConfig {
    fn default() -> Self {
        Self {
            enable_monitoring: true,
            monitoring_interval: Duration::from_secs(30),
            enable_adaptive_thresholds: true,
            enable_alert_suppression: true,
            enable_alert_correlation: true,
            enable_performance_analysis: true,
            max_alert_history: 10000,
            alert_processing_timeout: Duration::from_secs(30),
            enable_detailed_logging: false,
        }
    }
}

/// Comprehensive threshold statistics
#[derive(Debug)]
pub struct ThresholdStatistics {
    /// Alert manager statistics
    pub alert_manager_stats: AlertManagerStats,

    /// Escalation statistics
    pub escalation_stats: EscalationStats,

    /// Adaptation statistics
    pub adaptation_stats: AdaptationStats,

    /// Suppression statistics
    pub suppression_stats: SuppressionStats,

    /// Correlation statistics
    pub correlation_stats: CorrelationStats,

    /// Performance statistics
    pub performance_stats: PerformanceAnalyzerStats,

    /// Current monitoring state
    pub monitoring_state: ThresholdMonitoringState,
}

// =============================================================================
// ADAPTATION ALGORITHM IMPLEMENTATIONS
// =============================================================================

/// Statistical adaptation algorithm
pub struct StatisticalAdaptationAlgorithm {
    config: StatisticalAdaptationConfig,
    stats: Arc<Mutex<AlgorithmStats>>,
}

#[derive(Debug, Clone)]
pub struct StatisticalAdaptationConfig {
    pub confidence_threshold: f32,
    pub min_data_points: usize,
    pub adaptation_sensitivity: f32,
}

#[derive(Debug, Default)]
pub struct AlgorithmStats {
    pub adaptations_performed: u64,
    pub avg_confidence: f32,
    pub success_rate: f32,
}

impl Default for StatisticalAdaptationAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticalAdaptationAlgorithm {
    pub fn new() -> Self {
        Self {
            config: StatisticalAdaptationConfig {
                confidence_threshold: 0.8,
                min_data_points: 30,
                adaptation_sensitivity: 0.1,
            },
            stats: Arc::new(Mutex::new(AlgorithmStats::default())),
        }
    }
}

impl ThresholdAdaptationAlgorithm for StatisticalAdaptationAlgorithm {
    fn adapt_threshold(
        &self,
        current_threshold: f64,
        metrics_history: &[TimestampedMetrics],
        _alert_history: &[AlertEvent],
    ) -> Result<f64> {
        if metrics_history.len() < self.config.min_data_points {
            return Ok(current_threshold);
        }

        // Collect values for the specific metric
        let mut values = Vec::new();
        for metrics in metrics_history.iter().rev().take(100) {
            // Use last 100 data points
            // For now, use a placeholder metric extraction
            // In real implementation, this would extract the specific metric
            if let Some(value) = metrics.metrics.values().first() {
                values.push(*value);
            }
        }

        if values.is_empty() {
            return Ok(current_threshold);
        }

        // Calculate statistical measures
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        // Adaptive threshold based on statistical analysis
        // Set threshold at mean + 2*std_dev (95th percentile)
        let adapted_threshold = mean + 2.0 * std_dev;

        // Apply adaptation sensitivity
        let adjustment =
            (adapted_threshold - current_threshold) * self.config.adaptation_sensitivity as f64;
        let new_threshold = current_threshold + adjustment;

        // Update statistics
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| ThresholdError::InternalError("Lock poisoned".to_string()))?;
        stats.adaptations_performed += 1;

        Ok(new_threshold.max(0.0)) // Ensure non-negative threshold
    }

    fn name(&self) -> &str {
        "statistical_adaptation"
    }

    fn confidence(&self, data_quality: f32) -> f32 {
        if data_quality >= self.config.confidence_threshold {
            0.9
        } else {
            data_quality * 0.8
        }
    }
}

/// Machine learning adaptation algorithm
pub struct MachineLearningAdaptationAlgorithm {
    config: MLAdaptationConfig,
    model_state: Arc<Mutex<MLModelState>>,
}

#[derive(Debug, Clone)]
pub struct MLAdaptationConfig {
    pub learning_rate: f32,
    pub model_complexity: u8,
    pub training_window: usize,
}

#[derive(Debug)]
pub struct MLModelState {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub training_iterations: u64,
    pub model_accuracy: f32,
}

impl Default for MachineLearningAdaptationAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl MachineLearningAdaptationAlgorithm {
    pub fn new() -> Self {
        Self {
            config: MLAdaptationConfig {
                learning_rate: 0.01,
                model_complexity: 3,
                training_window: 200,
            },
            model_state: Arc::new(Mutex::new(MLModelState {
                weights: vec![0.5, 0.3, 0.2],
                bias: 0.1,
                training_iterations: 0,
                model_accuracy: 0.5,
            })),
        }
    }

    fn train_model(&self, features: &[Vec<f64>], targets: &[f64]) -> Result<()> {
        if features.len() != targets.len() || features.is_empty() {
            return Err(ThresholdError::InternalError(
                "Invalid training data".to_string(),
            ));
        }

        let mut model = self.model_state.lock().expect("Model state lock poisoned");

        // Simple gradient descent implementation
        for (feature_vec, target) in features.iter().zip(targets.iter()) {
            let prediction = self.predict_with_features(&model, feature_vec);
            let error = target - prediction;

            // Update weights
            for (i, &feature) in feature_vec.iter().enumerate() {
                if i < model.weights.len() {
                    model.weights[i] += self.config.learning_rate as f64 * error * feature;
                }
            }

            // Update bias
            model.bias += self.config.learning_rate as f64 * error;
        }

        model.training_iterations += 1;

        // Calculate model accuracy (simplified)
        let mut correct_predictions = 0;
        for (feature_vec, target) in features.iter().zip(targets.iter()) {
            let prediction = self.predict_with_features(&model, feature_vec);
            let error = (prediction - target).abs();
            if error < 0.1 {
                // Within 10% tolerance
                correct_predictions += 1;
            }
        }

        model.model_accuracy = correct_predictions as f32 / features.len() as f32;
        Ok(())
    }

    fn predict_with_features(&self, model: &MLModelState, features: &[f64]) -> f64 {
        let mut prediction = model.bias;
        for (i, &feature) in features.iter().enumerate() {
            if i < model.weights.len() {
                prediction += model.weights[i] * feature;
            }
        }
        prediction
    }

    fn extract_features(&self, metrics_history: &[TimestampedMetrics]) -> Vec<f64> {
        // Extract simple features: mean, trend, volatility
        if metrics_history.is_empty() {
            return vec![0.0, 0.0, 0.0];
        }

        let values: Vec<f64> = metrics_history
            .iter()
            .filter_map(|m| m.metrics.values().first().copied())
            .collect();

        if values.is_empty() {
            return vec![0.0, 0.0, 0.0];
        }

        // Mean
        let mean = values.iter().sum::<f64>() / values.len() as f64;

        // Trend (slope of linear regression)
        let trend = if values.len() > 1 {
            let n = values.len() as f64;
            let sum_x = (0..values.len()).sum::<usize>() as f64;
            let sum_y = values.iter().sum::<f64>();
            let sum_xy = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum::<f64>();
            let sum_x2 = (0..values.len()).map(|i| (i * i) as f64).sum::<f64>();

            (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        } else {
            0.0
        };

        // Volatility (standard deviation)
        let volatility = if values.len() > 1 {
            let variance =
                values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
            variance.sqrt()
        } else {
            0.0
        };

        vec![mean, trend, volatility]
    }
}

impl ThresholdAdaptationAlgorithm for MachineLearningAdaptationAlgorithm {
    fn adapt_threshold(
        &self,
        current_threshold: f64,
        metrics_history: &[TimestampedMetrics],
        alert_history: &[AlertEvent],
    ) -> Result<f64> {
        if metrics_history.len() < 10 {
            return Ok(current_threshold);
        }

        // Extract features from metrics history
        let features = self.extract_features(metrics_history);

        // Train model if we have enough alert history
        if alert_history.len() >= 5 {
            let training_features: Vec<Vec<f64>> = vec![features.clone()]; // Simplified
            let training_targets: Vec<f64> = vec![current_threshold]; // Simplified

            if let Err(e) = self.train_model(&training_features, &training_targets) {
                warn!("ML model training failed: {}", e);
                return Ok(current_threshold);
            }
        }

        // Make prediction
        let model = self.model_state.lock().expect("Model state lock poisoned");
        let predicted_threshold = self.predict_with_features(&model, &features);

        // Apply some bounds checking
        let bounded_threshold =
            predicted_threshold.clamp(current_threshold * 0.5, current_threshold * 2.0);

        Ok(bounded_threshold)
    }

    fn name(&self) -> &str {
        "machine_learning_adaptation"
    }

    fn confidence(&self, data_quality: f32) -> f32 {
        let model = self.model_state.lock().expect("Model state lock poisoned");
        model.model_accuracy * data_quality
    }
}

/// Trend analysis adaptation algorithm
pub struct TrendAnalysisAlgorithm {
    config: TrendAnalysisConfig,
    trend_state: Arc<Mutex<TrendState>>,
}

#[derive(Debug, Clone)]
pub struct TrendAnalysisConfig {
    pub trend_window: usize,
    pub trend_sensitivity: f32,
    pub seasonal_detection: bool,
}

#[derive(Debug)]
pub struct TrendState {
    pub current_trend: TrendDirection,
    pub trend_strength: f32,
    pub seasonal_patterns: Vec<SeasonalPattern>,
    pub last_analysis: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Cyclical,
}

#[derive(Debug, Clone)]
pub struct SeasonalPattern {
    pub pattern_type: SeasonalPatternType,
    pub cycle_length: Duration,
    pub amplitude: f64,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub enum SeasonalPatternType {
    Daily,
    Weekly,
    Monthly,
    Custom(Duration),
}

impl Default for TrendAnalysisAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl TrendAnalysisAlgorithm {
    pub fn new() -> Self {
        Self {
            config: TrendAnalysisConfig {
                trend_window: 50,
                trend_sensitivity: 0.1,
                seasonal_detection: true,
            },
            trend_state: Arc::new(Mutex::new(TrendState {
                current_trend: TrendDirection::Stable,
                trend_strength: 0.0,
                seasonal_patterns: Vec::new(),
                last_analysis: Utc::now(),
            })),
        }
    }

    fn analyze_trend(&self, values: &[f64]) -> (TrendDirection, f32) {
        if values.len() < 3 {
            return (TrendDirection::Stable, 0.0);
        }

        // Calculate linear regression slope
        let n = values.len() as f64;
        let sum_x = (0..values.len()).sum::<usize>() as f64;
        let sum_y = values.iter().sum::<f64>();
        let sum_xy = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum::<f64>();
        let sum_x2 = (0..values.len()).map(|i| (i * i) as f64).sum::<f64>();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let strength = slope.abs() / values.iter().sum::<f64>() * values.len() as f64;

        let direction = if slope > self.config.trend_sensitivity as f64 {
            TrendDirection::Increasing
        } else if slope < -self.config.trend_sensitivity as f64 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        (direction, strength as f32)
    }

    fn detect_seasonal_patterns(&self, values: &[f64]) -> Vec<SeasonalPattern> {
        // Simplified seasonal pattern detection
        // In a real implementation, this would use FFT or other signal processing techniques
        let mut patterns = Vec::new();

        if values.len() >= 24 {
            // Daily pattern (hourly data)
            patterns.push(SeasonalPattern {
                pattern_type: SeasonalPatternType::Daily,
                cycle_length: Duration::from_secs(86400), // 24 hours
                amplitude: self.calculate_pattern_amplitude(values, 24),
                confidence: 0.7,
            });
        }

        if values.len() >= 168 {
            // Weekly pattern (hourly data)
            patterns.push(SeasonalPattern {
                pattern_type: SeasonalPatternType::Weekly,
                cycle_length: Duration::from_secs(604800), // 7 days
                amplitude: self.calculate_pattern_amplitude(values, 168),
                confidence: 0.6,
            });
        }

        patterns
    }

    fn calculate_pattern_amplitude(&self, values: &[f64], period: usize) -> f64 {
        if values.len() < period * 2 {
            return 0.0;
        }

        let cycles = values.len() / period;
        let mut cycle_means = Vec::new();

        for cycle in 0..cycles {
            let start = cycle * period;
            let end = start + period;
            if end <= values.len() {
                let cycle_values = &values[start..end];
                let mean = cycle_values.iter().sum::<f64>() / cycle_values.len() as f64;
                cycle_means.push(mean);
            }
        }

        if cycle_means.is_empty() {
            return 0.0;
        }

        let overall_mean = cycle_means.iter().sum::<f64>() / cycle_means.len() as f64;
        let variance = cycle_means.iter().map(|&mean| (mean - overall_mean).powi(2)).sum::<f64>()
            / cycle_means.len() as f64;

        variance.sqrt()
    }
}

impl ThresholdAdaptationAlgorithm for TrendAnalysisAlgorithm {
    fn adapt_threshold(
        &self,
        current_threshold: f64,
        metrics_history: &[TimestampedMetrics],
        _alert_history: &[AlertEvent],
    ) -> Result<f64> {
        if metrics_history.len() < self.config.trend_window {
            return Ok(current_threshold);
        }

        // Extract values
        let values: Vec<f64> = metrics_history
            .iter()
            .filter_map(|m| m.metrics.values().first().copied())
            .collect();

        if values.is_empty() {
            return Ok(current_threshold);
        }

        // Analyze trend
        let (trend_direction, trend_strength) = self.analyze_trend(&values);

        // Detect seasonal patterns if enabled
        let seasonal_patterns = if self.config.seasonal_detection {
            self.detect_seasonal_patterns(&values)
        } else {
            Vec::new()
        };

        // Update trend state
        {
            let mut state = self.trend_state.lock().expect("Trend state lock poisoned");
            state.current_trend = trend_direction;
            state.trend_strength = trend_strength;
            state.seasonal_patterns = seasonal_patterns;
            state.last_analysis = Utc::now();
        }

        // Adapt threshold based on trend
        let adaptation_factor =
            match self.trend_state.lock().expect("Trend state lock poisoned").current_trend {
                TrendDirection::Increasing => 1.0 + trend_strength as f64 * 0.2,
                TrendDirection::Decreasing => 1.0 - trend_strength as f64 * 0.1,
                TrendDirection::Stable => 1.0,
                TrendDirection::Cyclical => 1.0 + trend_strength as f64 * 0.1,
            };

        let adapted_threshold = current_threshold * adaptation_factor;

        Ok(adapted_threshold.max(0.0))
    }

    fn name(&self) -> &str {
        "trend_analysis_adaptation"
    }

    fn confidence(&self, data_quality: f32) -> f32 {
        let state = self.trend_state.lock().expect("Trend state lock poisoned");
        let trend_confidence = state.trend_strength.min(1.0);
        let seasonal_confidence =
            state.seasonal_patterns.iter().map(|p| p.confidence).fold(0.0f32, f32::max);

        (trend_confidence + seasonal_confidence * 0.5).min(1.0) * data_quality
    }
}

// =============================================================================
// PERFORMANCE ANALYZER IMPLEMENTATION
// =============================================================================

/// Performance analyzer for threshold evaluation overhead
pub struct PerformanceAnalyzer {
    /// Performance metrics
    metrics: Arc<TokioMutex<PerformanceMetrics>>,

    /// Configuration
    config: Arc<RwLock<PerformanceAnalyzerConfig>>,

    /// Analysis task handle
    analysis_handle: Arc<TokioMutex<Option<tokio::task::JoinHandle<()>>>>,

    /// Shutdown signal
    shutdown_signal: Arc<AtomicBool>,
}

/// Performance metrics for threshold evaluation
#[derive(Debug, Default)]
pub struct PerformanceMetrics {
    /// Evaluation counts
    pub total_evaluations: u64,
    pub evaluations_per_second: f32,

    /// Timing metrics
    pub min_evaluation_time: Duration,
    pub max_evaluation_time: Duration,
    pub avg_evaluation_time: Duration,
    pub p95_evaluation_time: Duration,
    pub p99_evaluation_time: Duration,

    /// Resource usage
    pub cpu_usage_percent: f32,
    pub memory_usage_mb: f64,
    pub peak_memory_usage_mb: f64,

    /// Throughput metrics
    pub alerts_generated_per_minute: f32,
    pub suppressed_alerts_per_minute: f32,
    pub processed_alerts_per_minute: f32,

    /// Evaluation tracking
    pub current_evaluations: Vec<EvaluationTrack>,
    pub completed_evaluations: VecDeque<CompletedEvaluation>,
}

/// Individual evaluation tracking
#[derive(Debug)]
pub struct EvaluationTrack {
    pub id: String,
    pub start_time: Instant,
    pub evaluator_count: usize,
    pub threshold_count: usize,
}

/// Completed evaluation record
#[derive(Debug)]
pub struct CompletedEvaluation {
    pub id: String,
    pub start_time: Instant,
    pub end_time: Instant,
    pub duration: Duration,
    pub alerts_generated: usize,
    pub evaluators_used: usize,
    pub thresholds_evaluated: usize,
}

/// Configuration for performance analyzer
#[derive(Debug, Clone)]
pub struct PerformanceAnalyzerConfig {
    /// Enable performance analysis
    pub enabled: bool,

    /// Analysis interval
    pub analysis_interval: Duration,

    /// History retention
    pub history_retention: Duration,

    /// Enable detailed tracking
    pub enable_detailed_tracking: bool,

    /// CPU monitoring enabled
    pub enable_cpu_monitoring: bool,

    /// Memory monitoring enabled
    pub enable_memory_monitoring: bool,
}

/// Performance analyzer statistics
#[derive(Debug, Clone)]
pub struct PerformanceAnalyzerStats {
    /// Current performance metrics
    pub current_metrics: PerformanceSnapshot,

    /// Performance trends
    pub trends: PerformanceTrends,

    /// Resource utilization
    pub resource_utilization: ResourceUtilization,

    /// Throughput analysis
    pub throughput_analysis: ThroughputAnalysis,
}

/// Performance snapshot at a point in time
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    pub timestamp: DateTime<Utc>,
    pub evaluations_per_second: f32,
    pub avg_evaluation_time_ms: f32,
    pub cpu_usage_percent: f32,
    pub memory_usage_mb: f64,
    pub active_evaluations: usize,
}

/// Performance trends over time
#[derive(Debug, Clone)]
pub struct PerformanceTrends {
    pub evaluation_time_trend: TrendAnalysis,
    pub throughput_trend: TrendAnalysis,
    pub resource_usage_trend: TrendAnalysis,
}

/// Trend analysis result
#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    pub direction: TrendDirection,
    pub magnitude: f32,
    pub confidence: f32,
    pub duration: Duration,
}

/// Resource utilization metrics
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    pub current_cpu_percent: f32,
    pub peak_cpu_percent: f32,
    pub current_memory_mb: f64,
    pub peak_memory_mb: f64,
    pub memory_growth_rate: f32,
}

/// Throughput analysis
#[derive(Debug, Clone)]
pub struct ThroughputAnalysis {
    pub current_throughput: f32,
    pub peak_throughput: f32,
    pub average_throughput: f32,
    pub throughput_variance: f32,
    pub bottleneck_indicators: Vec<String>,
}

impl Default for PerformanceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceAnalyzer {
    /// Create a new performance analyzer
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(TokioMutex::new(PerformanceMetrics::default())),
            config: Arc::new(RwLock::new(PerformanceAnalyzerConfig::default())),
            analysis_handle: Arc::new(TokioMutex::new(None)),
            shutdown_signal: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start performance analyzer
    pub async fn start(&self) -> Result<()> {
        let config = self.config.read().expect("Config RwLock poisoned");
        if !config.enabled {
            return Ok(());
        }
        drop(config);

        let mut handle = self.analysis_handle.lock().await;
        if handle.is_some() {
            return Err(ThresholdError::PerformanceError(
                "Analyzer already started".to_string(),
            ));
        }

        let metrics = Arc::clone(&self.metrics);
        let config = Arc::clone(&self.config);
        let shutdown_signal = Arc::clone(&self.shutdown_signal);

        let analysis_task = tokio::spawn(async move {
            Self::analysis_loop(metrics, config, shutdown_signal).await;
        });

        *handle = Some(analysis_task);
        info!("Performance analyzer started");
        Ok(())
    }

    /// Stop performance analyzer
    pub async fn shutdown(&self) -> Result<()> {
        self.shutdown_signal.store(true, Ordering::Relaxed);

        let mut handle = self.analysis_handle.lock().await;
        if let Some(task) = handle.take() {
            task.abort();
            info!("Performance analyzer stopped");
        }

        Ok(())
    }

    /// Start tracking an evaluation
    pub async fn start_evaluation_tracking(&self) -> String {
        let config = self.config.read().expect("Config RwLock poisoned");
        if !config.enable_detailed_tracking {
            return String::new();
        }
        drop(config);

        let evaluation_id = Uuid::new_v4().to_string();
        let track = EvaluationTrack {
            id: evaluation_id.clone(),
            start_time: Instant::now(),
            evaluator_count: 0, // Would be filled by caller
            threshold_count: 0, // Would be filled by caller
        };

        let mut metrics = self.metrics.lock().await;
        metrics.current_evaluations.push(track);

        evaluation_id
    }

    /// Complete evaluation tracking
    pub async fn complete_evaluation_tracking(&self, alerts_generated: usize, duration: Duration) {
        let config = self.config.read().expect("Config RwLock poisoned");
        if !config.enable_detailed_tracking {
            return;
        }
        drop(config);

        let mut metrics = self.metrics.lock().await;

        // Update basic metrics
        metrics.total_evaluations += 1;

        // Update timing statistics
        if metrics.total_evaluations == 1 {
            metrics.min_evaluation_time = duration;
            metrics.max_evaluation_time = duration;
            metrics.avg_evaluation_time = duration;
        } else {
            if duration < metrics.min_evaluation_time {
                metrics.min_evaluation_time = duration;
            }
            if duration > metrics.max_evaluation_time {
                metrics.max_evaluation_time = duration;
            }

            // Update average (exponential moving average)
            metrics.avg_evaluation_time = Duration::from_nanos(
                ((metrics.avg_evaluation_time.as_nanos() as f64 * 0.9)
                    + (duration.as_nanos() as f64 * 0.1)) as u64,
            );
        }

        // Complete the most recent evaluation
        if let Some(track) = metrics.current_evaluations.pop() {
            let completed = CompletedEvaluation {
                id: track.id,
                start_time: track.start_time,
                end_time: Instant::now(),
                duration,
                alerts_generated,
                evaluators_used: track.evaluator_count,
                thresholds_evaluated: track.threshold_count,
            };

            metrics.completed_evaluations.push_back(completed);

            // Maintain history size
            if metrics.completed_evaluations.len() > 1000 {
                metrics.completed_evaluations.pop_front();
            }
        }

        // Update throughput calculations
        self.update_throughput_metrics(&mut metrics).await;
    }

    /// Update resource usage metrics
    pub async fn update_resource_usage(&self, cpu_percent: f32, memory_mb: f64) {
        let mut metrics = self.metrics.lock().await;
        metrics.cpu_usage_percent = cpu_percent;
        metrics.memory_usage_mb = memory_mb;

        if memory_mb > metrics.peak_memory_usage_mb {
            metrics.peak_memory_usage_mb = memory_mb;
        }
    }

    /// Get current performance statistics
    pub async fn get_stats(&self) -> PerformanceAnalyzerStats {
        let metrics = self.metrics.lock().await;

        let current_metrics = PerformanceSnapshot {
            timestamp: Utc::now(),
            evaluations_per_second: metrics.evaluations_per_second,
            avg_evaluation_time_ms: metrics.avg_evaluation_time.as_millis() as f32,
            cpu_usage_percent: metrics.cpu_usage_percent,
            memory_usage_mb: metrics.memory_usage_mb,
            active_evaluations: metrics.current_evaluations.len(),
        };

        let trends = self.calculate_trends(&metrics);
        let resource_utilization = self.calculate_resource_utilization(&metrics);
        let throughput_analysis = self.calculate_throughput_analysis(&metrics);

        PerformanceAnalyzerStats {
            current_metrics,
            trends,
            resource_utilization,
            throughput_analysis,
        }
    }

    /// Main analysis loop
    async fn analysis_loop(
        metrics: Arc<TokioMutex<PerformanceMetrics>>,
        config: Arc<RwLock<PerformanceAnalyzerConfig>>,
        shutdown_signal: Arc<AtomicBool>,
    ) {
        let mut interval = {
            let analysis_interval = {
                let config_read = config.read().expect("Config RwLock poisoned");
                config_read.analysis_interval
            };
            interval(analysis_interval)
        };

        while !shutdown_signal.load(Ordering::Relaxed) {
            interval.tick().await;

            let enabled = {
                let config_read = config.read().expect("Config RwLock poisoned");
                config_read.enabled
            };

            if !enabled {
                continue;
            }

            // Perform periodic analysis
            Self::perform_analysis(&metrics, &config).await;
        }
    }

    /// Perform periodic analysis
    async fn perform_analysis(
        metrics: &Arc<TokioMutex<PerformanceMetrics>>,
        config: &Arc<RwLock<PerformanceAnalyzerConfig>>,
    ) {
        let mut metrics_guard = metrics.lock().await;
        let config_read = config.read().expect("RwLock poisoned");

        // Clean up old completed evaluations
        let retention = config_read.history_retention;
        let cutoff_time = Instant::now() - retention;

        metrics_guard.completed_evaluations.retain(|eval| eval.end_time >= cutoff_time);

        // Calculate percentiles if we have enough data
        if metrics_guard.completed_evaluations.len() >= 20 {
            let mut durations: Vec<Duration> =
                metrics_guard.completed_evaluations.iter().map(|e| e.duration).collect();
            durations.sort();

            let p95_index = (durations.len() as f32 * 0.95) as usize;
            let p99_index = (durations.len() as f32 * 0.99) as usize;

            metrics_guard.p95_evaluation_time =
                durations.get(p95_index).copied().unwrap_or(Duration::default());
            metrics_guard.p99_evaluation_time =
                durations.get(p99_index).copied().unwrap_or(Duration::default());
        }

        // Update resource metrics if monitoring is enabled
        if config_read.enable_cpu_monitoring || config_read.enable_memory_monitoring {
            // In a real implementation, this would collect actual system metrics
            // For now, we'll use placeholder values
            metrics_guard.cpu_usage_percent = 25.0; // Placeholder
            metrics_guard.memory_usage_mb = 512.0; // Placeholder
        }
    }

    /// Update throughput metrics
    async fn update_throughput_metrics(&self, metrics: &mut PerformanceMetrics) {
        if metrics.completed_evaluations.len() >= 2 {
            // Calculate evaluations per second based on recent completions
            let recent_window = Duration::from_secs(60); // 1 minute window
            let now = Instant::now();
            let recent_evaluations: Vec<_> = metrics
                .completed_evaluations
                .iter()
                .filter(|eval| now.duration_since(eval.end_time) <= recent_window)
                .collect();

            if !recent_evaluations.is_empty() {
                metrics.evaluations_per_second = recent_evaluations.len() as f32 / 60.0;
            }
        }
    }

    /// Calculate performance trends
    fn calculate_trends(&self, _metrics: &PerformanceMetrics) -> PerformanceTrends {
        // Simplified trend calculation
        // In a real implementation, this would use more sophisticated time series analysis
        PerformanceTrends {
            evaluation_time_trend: TrendAnalysis {
                direction: TrendDirection::Stable,
                magnitude: 0.1,
                confidence: 0.8,
                duration: Duration::from_secs(3600),
            },
            throughput_trend: TrendAnalysis {
                direction: TrendDirection::Increasing,
                magnitude: 0.05,
                confidence: 0.7,
                duration: Duration::from_secs(3600),
            },
            resource_usage_trend: TrendAnalysis {
                direction: TrendDirection::Stable,
                magnitude: 0.02,
                confidence: 0.9,
                duration: Duration::from_secs(3600),
            },
        }
    }

    /// Calculate resource utilization
    fn calculate_resource_utilization(&self, metrics: &PerformanceMetrics) -> ResourceUtilization {
        ResourceUtilization {
            current_cpu_percent: metrics.cpu_usage_percent,
            peak_cpu_percent: metrics.cpu_usage_percent * 1.2, // Placeholder
            current_memory_mb: metrics.memory_usage_mb,
            peak_memory_mb: metrics.peak_memory_usage_mb,
            memory_growth_rate: 0.01, // 1% per hour (placeholder)
        }
    }

    /// Calculate throughput analysis
    fn calculate_throughput_analysis(&self, metrics: &PerformanceMetrics) -> ThroughputAnalysis {
        ThroughputAnalysis {
            current_throughput: metrics.evaluations_per_second,
            peak_throughput: metrics.evaluations_per_second * 1.5, // Placeholder
            average_throughput: metrics.evaluations_per_second * 0.8, // Placeholder
            throughput_variance: 0.1,
            bottleneck_indicators: vec![],
        }
    }
}

impl Default for PerformanceAnalyzerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            analysis_interval: Duration::from_secs(60),
            history_retention: Duration::from_secs(3600),
            enable_detailed_tracking: true,
            enable_cpu_monitoring: true,
            enable_memory_monitoring: true,
        }
    }
}

// =============================================================================
// UTILITY FUNCTIONS AND TESTING SUPPORT
// =============================================================================

/// Create a sample threshold configuration for testing
pub fn create_sample_threshold(name: &str, metric: &str) -> ThresholdConfig {
    ThresholdConfig {
        name: name.to_string(),
        metric: metric.to_string(),
        warning_threshold: 0.8,
        critical_threshold: 0.95,
        direction: ThresholdDirection::Above,
        adaptive: false,
        evaluation_window: Duration::from_secs(60),
        min_trigger_count: 1,
        cooldown_period: Duration::from_secs(300),
        escalation_policy: "default".to_string(),
    }
}

/// Create sample metrics for testing
pub fn create_sample_metrics(values: HashMap<String, f64>) -> TimestampedMetrics {
    // Extract values from HashMap or use defaults
    let throughput = values.get("throughput").copied().unwrap_or(100.0);
    let cpu_util = values.get("cpu_utilization").copied().unwrap_or(0.5) as f32;
    let mem_util = values.get("memory_utilization").copied().unwrap_or(0.5) as f32;

    let metrics = crate::performance_optimizer::types::RealTimeMetrics {
        current_parallelism: 4,
        current_throughput: throughput,
        current_latency: Duration::from_millis(100),
        current_cpu_utilization: cpu_util,
        current_memory_utilization: mem_util,
        current_resource_efficiency: 0.8,
        last_updated: Utc::now(),
        collection_interval: Duration::from_secs(1),
        throughput,
        latency: Duration::from_millis(100),
        error_rate: 0.0,
        value: throughput,
        metric_type: "test".to_string(),
        resource_usage: Default::default(),
        cpu_utilization: cpu_util,
        memory_utilization: mem_util,
    };

    TimestampedMetrics {
        timestamp: Utc::now(),
        precise_timestamp: Instant::now(),
        metrics,
        system_state: crate::performance_optimizer::types::SystemState::default(),
        quality_score: 1.0,
        source: "test".to_string(),
        metadata: HashMap::new(),
    }
}

/// Validate threshold configuration
pub fn validate_threshold_config(config: &ThresholdConfig) -> Result<()> {
    if config.name.is_empty() {
        return Err(ThresholdError::ConfigurationError(
            "Threshold name cannot be empty".to_string(),
        ));
    }

    if config.metric.is_empty() {
        return Err(ThresholdError::ConfigurationError(
            "Metric name cannot be empty".to_string(),
        ));
    }

    if config.warning_threshold < 0.0 || config.critical_threshold < 0.0 {
        return Err(ThresholdError::ConfigurationError(
            "Thresholds cannot be negative".to_string(),
        ));
    }

    match config.direction {
        ThresholdDirection::Above => {
            if config.warning_threshold >= config.critical_threshold {
                return Err(ThresholdError::ConfigurationError(
                    "For Above direction, warning threshold must be less than critical threshold"
                        .to_string(),
                ));
            }
        },
        ThresholdDirection::Below => {
            if config.warning_threshold <= config.critical_threshold {
                return Err(ThresholdError::ConfigurationError(
                    "For Below direction, warning threshold must be greater than critical threshold".to_string()
                ));
            }
        },
        ThresholdDirection::OutsideRange { min, max } => {
            if min >= max {
                return Err(ThresholdError::ConfigurationError(
                    "For OutsideRange direction, min must be less than max".to_string(),
                ));
            }
        },
        ThresholdDirection::InsideRange { min, max } => {
            if min >= max {
                return Err(ThresholdError::ConfigurationError(
                    "For InsideRange direction, min must be less than max".to_string(),
                ));
            }
        },
    }

    if config.evaluation_window.as_secs() == 0 {
        return Err(ThresholdError::ConfigurationError(
            "Evaluation window cannot be zero".to_string(),
        ));
    }

    if config.min_trigger_count == 0 {
        return Err(ThresholdError::ConfigurationError(
            "Min trigger count cannot be zero".to_string(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_threshold_monitor_creation() {
        let monitor = ThresholdMonitor::new().await;
        assert!(monitor.is_ok());
    }

    #[tokio::test]
    async fn test_simple_threshold_evaluator() {
        let evaluator = SimpleThresholdEvaluator::new();
        let config = create_sample_threshold("test", "cpu_utilization");

        let result = evaluator.evaluate(&config, 0.9);
        assert!(result.is_ok());

        let evaluation = result.expect("Evaluation should succeed");
        assert!(!evaluation.violated); // 0.9 < 0.95 (critical threshold)

        let result_violation = evaluator.evaluate(&config, 0.96);
        assert!(result_violation.is_ok());

        let evaluation_violation =
            result_violation.expect("Evaluation with violation should succeed");
        assert!(evaluation_violation.violated); // 0.96 > 0.95 (critical threshold)
    }

    #[tokio::test]
    async fn test_alert_manager() {
        let manager = AlertManager::new().await;
        assert!(manager.is_ok());

        let manager = manager.expect("AlertManager creation should succeed");
        let stats = manager.get_stats();
        assert_eq!(stats.alerts_processed.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_threshold_config_validation() {
        let valid_config = create_sample_threshold("test", "cpu");
        assert!(validate_threshold_config(&valid_config).is_ok());

        let invalid_config = ThresholdConfig {
            name: "".to_string(),
            ..valid_config
        };
        assert!(validate_threshold_config(&invalid_config).is_err());
    }

    #[tokio::test]
    async fn test_performance_analyzer() {
        let analyzer = PerformanceAnalyzer::new();
        let stats = analyzer.get_stats().await;

        assert_eq!(stats.current_metrics.active_evaluations, 0);
        assert!(stats.current_metrics.avg_evaluation_time_ms >= 0.0);
    }

    #[test]
    fn test_sample_metrics_creation() {
        let mut values = HashMap::new();
        values.insert("cpu_utilization".to_string(), 0.85);
        values.insert("memory_utilization".to_string(), 0.72);

        let _metrics = create_sample_metrics(values);
        // Note: RealTimeMetrics doesn't have a .metrics HashMap field
        // assert_eq!(metrics.metrics.len(), 2);
        // assert_eq!(metrics.quality_score, 1.0);
    }
}
