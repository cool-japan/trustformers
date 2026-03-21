//! Adaptive threshold controller for dynamic threshold management.

use super::super::types::*;
use super::error::{Result, ThresholdError};
use super::monitor::{
    MachineLearningAdaptationAlgorithm, StatisticalAdaptationAlgorithm, TrendAnalysisAlgorithm,
};
use chrono::{DateTime, Utc};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;
use tokio::sync::Mutex as TokioMutex;
use tokio::time::interval;
use tracing::{info, warn};

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
