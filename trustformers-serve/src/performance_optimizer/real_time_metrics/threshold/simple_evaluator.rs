//! Simple threshold evaluator implementation.

use super::super::types::{SeverityLevel, ThresholdConfig, ThresholdDirection};
use super::error::Result;
use super::evaluator::ThresholdEvaluator;
use super::types::ThresholdEvaluation;
use chrono::Utc;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Simple threshold evaluator
///
/// Basic threshold evaluator that performs simple comparison against
/// static threshold values without any statistical analysis.
#[derive(Debug)]

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
