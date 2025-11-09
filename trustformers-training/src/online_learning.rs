use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineLearningConfig {
    pub buffer_size: usize,
    pub batch_size: usize,
    pub update_frequency: Duration,
    pub forgetting_factor: f32,
    pub adaptation_rate: f32,
    pub drift_detection_threshold: f32,
    pub window_size: usize,
    pub min_samples_for_update: usize,
    pub enable_concept_drift_detection: bool,
    pub enable_adaptive_learning_rate: bool,
}

impl Default for OnlineLearningConfig {
    fn default() -> Self {
        Self {
            buffer_size: 10000,
            batch_size: 32,
            update_frequency: Duration::from_secs(60),
            forgetting_factor: 0.99,
            adaptation_rate: 0.01,
            drift_detection_threshold: 0.1,
            window_size: 1000,
            min_samples_for_update: 10,
            enable_concept_drift_detection: true,
            enable_adaptive_learning_rate: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OnlineDataPoint {
    pub features: Vec<f32>,
    pub label: Vec<f32>,
    pub timestamp: Instant,
    pub importance_weight: f32,
}

#[derive(Debug, Clone)]
pub struct ConceptDrift {
    pub detected: bool,
    pub drift_score: f32,
    pub detection_time: Instant,
    pub drift_type: DriftType,
}

#[derive(Debug, Clone)]
pub enum DriftType {
    Gradual,
    Sudden,
    Incremental,
    Recurring,
}

pub struct PerformanceWindow {
    scores: VecDeque<f32>,
    timestamps: VecDeque<Instant>,
    window_size: usize,
}

impl PerformanceWindow {
    pub fn new(window_size: usize) -> Self {
        Self {
            scores: VecDeque::with_capacity(window_size),
            timestamps: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    pub fn add_score(&mut self, score: f32) {
        if self.scores.len() >= self.window_size {
            self.scores.pop_front();
            self.timestamps.pop_front();
        }
        self.scores.push_back(score);
        self.timestamps.push_back(Instant::now());
    }

    pub fn mean(&self) -> f32 {
        if self.scores.is_empty() {
            0.0
        } else {
            self.scores.iter().sum::<f32>() / self.scores.len() as f32
        }
    }

    pub fn variance(&self) -> f32 {
        if self.scores.len() < 2 {
            0.0
        } else {
            let mean = self.mean();
            let variance: f32 = self.scores.iter().map(|score| (score - mean).powi(2)).sum::<f32>()
                / (self.scores.len() - 1) as f32;
            variance
        }
    }

    pub fn is_full(&self) -> bool {
        self.scores.len() >= self.window_size
    }
}

pub struct OnlineLearningManager {
    config: OnlineLearningConfig,
    data_buffer: Arc<RwLock<VecDeque<OnlineDataPoint>>>,
    performance_window: Arc<RwLock<PerformanceWindow>>,
    model_state: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    last_update: Arc<RwLock<Instant>>,
    learning_rate: Arc<RwLock<f32>>,
    drift_detector: Arc<RwLock<ConceptDrift>>,
    statistics: Arc<RwLock<OnlineStatistics>>,
}

#[derive(Debug, Default, Clone)]
pub struct OnlineStatistics {
    pub total_samples_processed: usize,
    pub total_updates: usize,
    pub concept_drifts_detected: usize,
    pub average_latency: Duration,
    pub throughput: f32,
    pub last_performance_score: f32,
}

impl OnlineLearningManager {
    pub fn new(config: OnlineLearningConfig) -> Self {
        Self {
            performance_window: Arc::new(RwLock::new(PerformanceWindow::new(config.window_size))),
            data_buffer: Arc::new(RwLock::new(VecDeque::with_capacity(config.buffer_size))),
            model_state: Arc::new(RwLock::new(HashMap::new())),
            last_update: Arc::new(RwLock::new(Instant::now())),
            learning_rate: Arc::new(RwLock::new(config.adaptation_rate)),
            drift_detector: Arc::new(RwLock::new(ConceptDrift {
                detected: false,
                drift_score: 0.0,
                detection_time: Instant::now(),
                drift_type: DriftType::Gradual,
            })),
            statistics: Arc::new(RwLock::new(OnlineStatistics::default())),
            config,
        }
    }

    pub fn add_data_point(&self, data_point: OnlineDataPoint) -> Result<()> {
        let mut buffer = self
            .data_buffer
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on data buffer"))?;

        if buffer.len() >= self.config.buffer_size {
            buffer.pop_front();
        }
        buffer.push_back(data_point);

        // Update statistics
        {
            let mut stats = self
                .statistics
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on statistics"))?;
            stats.total_samples_processed += 1;
        }

        // Check if we should trigger an update
        let should_update = {
            let last_update = self
                .last_update
                .read()
                .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on last_update"))?;
            last_update.elapsed() >= self.config.update_frequency
                && buffer.len() >= self.config.min_samples_for_update
        };

        if should_update {
            self.trigger_model_update()?;
        }

        Ok(())
    }

    pub fn trigger_model_update(&self) -> Result<()> {
        let start_time = Instant::now();

        // Get batch of data
        let batch = self.get_training_batch()?;

        if batch.is_empty() {
            return Ok(());
        }

        // Perform model update (simplified - in practice would call actual model training)
        self.update_model_weights(&batch)?;

        // Update last update time
        {
            let mut last_update = self
                .last_update
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on last_update"))?;
            *last_update = Instant::now();
        }

        // Update statistics
        {
            let mut stats = self
                .statistics
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on statistics"))?;
            stats.total_updates += 1;
            stats.average_latency = (stats.average_latency + start_time.elapsed()) / 2;
        }

        Ok(())
    }

    fn get_training_batch(&self) -> Result<Vec<OnlineDataPoint>> {
        let buffer = self
            .data_buffer
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on data buffer"))?;

        let batch_size = std::cmp::min(self.config.batch_size, buffer.len());
        let batch: Vec<_> = buffer.iter().rev().take(batch_size).cloned().collect();

        Ok(batch)
    }

    fn update_model_weights(&self, batch: &[OnlineDataPoint]) -> Result<()> {
        let learning_rate = {
            let lr = self
                .learning_rate
                .read()
                .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on learning_rate"))?;
            *lr
        };

        // Simplified weight update - in practice would call actual model training
        let mut model_state = self
            .model_state
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on model_state"))?;

        // Calculate gradients and update weights (simplified)
        for data_point in batch {
            let weight_key = "layer_weights".to_string();
            let weights = model_state.entry(weight_key).or_insert_with(|| vec![0.0; 128]);

            // Simplified gradient computation and weight update
            for (i, feature) in data_point.features.iter().enumerate() {
                if i < weights.len() {
                    weights[i] += learning_rate * feature * data_point.importance_weight;
                }
            }
        }

        Ok(())
    }

    pub fn detect_concept_drift(&self, current_performance: f32) -> Result<bool> {
        if !self.config.enable_concept_drift_detection {
            return Ok(false);
        }

        // Add current performance to window
        {
            let mut window = self.performance_window.write().map_err(|_| {
                anyhow::anyhow!("Failed to acquire write lock on performance window")
            })?;
            window.add_score(current_performance);

            if !window.is_full() {
                return Ok(false);
            }
        }

        let window = self
            .performance_window
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on performance window"))?;

        // Simple drift detection using performance degradation
        let recent_mean = {
            let recent_scores: Vec<_> =
                window.scores.iter().rev().take(window.window_size / 4).cloned().collect();
            if recent_scores.is_empty() {
                return Ok(false);
            }
            recent_scores.iter().sum::<f32>() / recent_scores.len() as f32
        };

        let historical_mean = window.mean();
        let performance_drop = historical_mean - recent_mean;

        let drift_detected = performance_drop > self.config.drift_detection_threshold;

        if drift_detected {
            let mut drift_detector = self
                .drift_detector
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on drift detector"))?;

            drift_detector.detected = true;
            drift_detector.drift_score = performance_drop;
            drift_detector.detection_time = Instant::now();
            drift_detector.drift_type =
                if performance_drop > self.config.drift_detection_threshold * 2.0 {
                    DriftType::Sudden
                } else {
                    DriftType::Gradual
                };

            // Update statistics
            let mut stats = self
                .statistics
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on statistics"))?;
            stats.concept_drifts_detected += 1;

            // Adapt learning rate if enabled
            if self.config.enable_adaptive_learning_rate {
                self.adapt_learning_rate(performance_drop)?;
            }
        }

        Ok(drift_detected)
    }

    fn adapt_learning_rate(&self, drift_score: f32) -> Result<()> {
        let mut learning_rate = self
            .learning_rate
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on learning_rate"))?;

        // Increase learning rate proportionally to drift severity
        let adaptation_factor = 1.0 + drift_score;
        *learning_rate = (*learning_rate * adaptation_factor).min(0.1); // Cap at 0.1

        Ok(())
    }

    pub fn reset_after_drift(&self) -> Result<()> {
        // Clear data buffer to start fresh
        {
            let mut buffer = self
                .data_buffer
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on data buffer"))?;
            buffer.clear();
        }

        // Reset drift detector
        {
            let mut drift_detector = self
                .drift_detector
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on drift detector"))?;
            drift_detector.detected = false;
            drift_detector.drift_score = 0.0;
        }

        // Reset learning rate to initial value
        {
            let mut learning_rate = self
                .learning_rate
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on learning_rate"))?;
            *learning_rate = self.config.adaptation_rate;
        }

        Ok(())
    }

    pub fn get_statistics(&self) -> Result<OnlineStatistics> {
        let stats = self
            .statistics
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on statistics"))?;
        Ok((*stats).clone())
    }

    pub fn get_current_drift_state(&self) -> Result<ConceptDrift> {
        let drift = self
            .drift_detector
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on drift detector"))?;
        Ok(drift.clone())
    }

    pub fn get_buffer_size(&self) -> Result<usize> {
        let buffer = self
            .data_buffer
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on data buffer"))?;
        Ok(buffer.len())
    }
}

#[derive(Debug)]
pub enum OnlineLearningError {
    BufferFull,
    ModelUpdateFailed,
    DriftDetectionFailed,
    ConfigurationError,
}

impl std::fmt::Display for OnlineLearningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OnlineLearningError::BufferFull => write!(f, "Data buffer is full"),
            OnlineLearningError::ModelUpdateFailed => write!(f, "Model update failed"),
            OnlineLearningError::DriftDetectionFailed => write!(f, "Drift detection failed"),
            OnlineLearningError::ConfigurationError => write!(f, "Configuration error"),
        }
    }
}

impl std::error::Error for OnlineLearningError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_online_learning_manager_creation() {
        let config = OnlineLearningConfig::default();
        let manager = OnlineLearningManager::new(config);
        assert_eq!(manager.get_buffer_size().unwrap(), 0);
    }

    #[test]
    fn test_add_data_point() {
        let config = OnlineLearningConfig::default();
        let manager = OnlineLearningManager::new(config);

        let data_point = OnlineDataPoint {
            features: vec![1.0, 2.0, 3.0],
            label: vec![1.0],
            timestamp: Instant::now(),
            importance_weight: 1.0,
        };

        manager.add_data_point(data_point).unwrap();
        assert_eq!(manager.get_buffer_size().unwrap(), 1);
    }

    #[test]
    fn test_performance_window() {
        let mut window = PerformanceWindow::new(3);

        window.add_score(0.8);
        window.add_score(0.7);
        window.add_score(0.9);

        assert_eq!(window.mean(), (0.8 + 0.7 + 0.9) / 3.0);
        assert!(window.is_full());

        window.add_score(0.6);
        assert_eq!(window.scores.len(), 3);
        assert_eq!(window.scores[0], 0.7); // 0.8 should be removed
    }

    #[test]
    fn test_concept_drift_detection() {
        let config = OnlineLearningConfig {
            window_size: 4,
            drift_detection_threshold: 0.1,
            ..Default::default()
        };
        let manager = OnlineLearningManager::new(config);

        // Add some high performance scores
        assert!(!manager.detect_concept_drift(0.9).unwrap());
        assert!(!manager.detect_concept_drift(0.85).unwrap());
        assert!(!manager.detect_concept_drift(0.88).unwrap());
        assert!(!manager.detect_concept_drift(0.87).unwrap());

        // Add a significantly lower score that should trigger drift detection
        assert!(manager.detect_concept_drift(0.6).unwrap());

        let drift_state = manager.get_current_drift_state().unwrap();
        assert!(drift_state.detected);
    }

    #[test]
    fn test_buffer_capacity() {
        let config = OnlineLearningConfig {
            buffer_size: 2,
            ..Default::default()
        };
        let manager = OnlineLearningManager::new(config);

        for i in 0..5 {
            let data_point = OnlineDataPoint {
                features: vec![i as f32],
                label: vec![i as f32],
                timestamp: Instant::now(),
                importance_weight: 1.0,
            };
            manager.add_data_point(data_point).unwrap();
        }

        // Buffer should not exceed capacity
        assert_eq!(manager.get_buffer_size().unwrap(), 2);
    }

    #[test]
    fn test_statistics_tracking() {
        let config = OnlineLearningConfig::default();
        let manager = OnlineLearningManager::new(config);

        let data_point = OnlineDataPoint {
            features: vec![1.0, 2.0],
            label: vec![1.0],
            timestamp: Instant::now(),
            importance_weight: 1.0,
        };

        manager.add_data_point(data_point).unwrap();

        let stats = manager.get_statistics().unwrap();
        assert_eq!(stats.total_samples_processed, 1);
    }
}
