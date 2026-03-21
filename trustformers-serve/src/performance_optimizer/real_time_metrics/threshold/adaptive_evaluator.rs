//! Adaptive threshold evaluator implementation.

use super::super::types::{SeverityLevel, ThresholdConfig, ThresholdDirection};
use super::error::{Result, ThresholdError};
use super::evaluator::ThresholdEvaluator;
use super::simple_evaluator::EvaluatorStats;
use super::types::ThresholdEvaluation;
use chrono::{DateTime, Utc};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::Mutex as TokioMutex;
use tracing::error;

/// Adaptive threshold evaluator
///
/// Advanced threshold evaluator that dynamically adjusts thresholds based on
/// historical patterns, system behavior, and machine learning techniques.
#[derive(Debug)]

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
