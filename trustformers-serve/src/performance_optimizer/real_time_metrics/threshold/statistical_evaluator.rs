//! Statistical threshold evaluator implementation.

use super::super::types::{SeverityLevel, ThresholdConfig, ThresholdDirection};
use super::error::{Result, ThresholdError};
use super::evaluator::ThresholdEvaluator;
use super::simple_evaluator::EvaluatorStats;
use super::types::ThresholdEvaluation;
use chrono::{DateTime, Utc};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Statistical threshold evaluator
///
/// Advanced threshold evaluator that uses statistical analysis to determine
/// threshold violations with enhanced confidence scoring and context.
#[derive(Debug)]

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
