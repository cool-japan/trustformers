use anyhow::Result;
// SciRS2 Integration Policy
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for task boundary detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryDetectionConfig {
    /// Window size for change detection
    pub window_size: usize,
    /// Threshold for boundary detection
    pub threshold: f32,
    /// Method for boundary detection
    pub detection_method: DetectionMethod,
    /// Minimum samples before detecting boundary
    pub min_samples: usize,
    /// Smoothing factor for running averages
    pub smoothing_factor: f32,
}

impl Default for BoundaryDetectionConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            threshold: 0.05,
            detection_method: DetectionMethod::LossIncrease,
            min_samples: 50,
            smoothing_factor: 0.1,
        }
    }
}

/// Methods for detecting task boundaries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DetectionMethod {
    /// Detect based on loss increase
    LossIncrease,
    /// Detect based on gradient magnitude change
    GradientMagnitude,
    /// Detect based on activation pattern change
    ActivationPattern,
    /// Detect based on prediction confidence
    ConfidenceChange,
    /// Combined detection using multiple signals
    Combined,
}

/// Task transition information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskTransition {
    /// Previous task ID
    pub from_task: String,
    /// New task ID
    pub to_task: String,
    /// Timestamp of transition
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Boundary detection confidence score
    pub boundary_score: f32,
}

/// Task boundary detector
#[derive(Debug)]
pub struct TaskBoundaryDetector {
    config: BoundaryDetectionConfig,
    loss_history: VecDeque<f32>,
    gradient_history: VecDeque<f32>,
    confidence_history: VecDeque<f32>,
    running_loss_avg: f32,
    running_gradient_avg: f32,
    running_confidence_avg: f32,
    sample_count: usize,
    last_boundary_sample: usize,
}

impl TaskBoundaryDetector {
    pub fn new(config: BoundaryDetectionConfig) -> Self {
        Self {
            config,
            loss_history: VecDeque::new(),
            gradient_history: VecDeque::new(),
            confidence_history: VecDeque::new(),
            running_loss_avg: 0.0,
            running_gradient_avg: 0.0,
            running_confidence_avg: 0.0,
            sample_count: 0,
            last_boundary_sample: 0,
        }
    }

    /// Update detector with new training sample
    pub fn update(&mut self, loss: f32, gradient_norm: f32, confidence: f32) {
        self.sample_count += 1;

        // Update running averages
        let alpha = self.config.smoothing_factor;
        self.running_loss_avg = alpha * loss + (1.0 - alpha) * self.running_loss_avg;
        self.running_gradient_avg =
            alpha * gradient_norm + (1.0 - alpha) * self.running_gradient_avg;
        self.running_confidence_avg =
            alpha * confidence + (1.0 - alpha) * self.running_confidence_avg;

        // Update history windows
        self.loss_history.push_back(loss);
        self.gradient_history.push_back(gradient_norm);
        self.confidence_history.push_back(confidence);

        // Maintain window size
        if self.loss_history.len() > self.config.window_size {
            self.loss_history.pop_front();
            self.gradient_history.pop_front();
            self.confidence_history.pop_front();
        }
    }

    /// Check if a task boundary is detected
    pub fn detect_boundary(&mut self) -> Result<Option<f32>> {
        if self.sample_count < self.config.min_samples {
            return Ok(None);
        }

        if self.sample_count - self.last_boundary_sample < self.config.min_samples {
            return Ok(None);
        }

        let boundary_score = match self.config.detection_method {
            DetectionMethod::LossIncrease => self.detect_loss_increase()?,
            DetectionMethod::GradientMagnitude => self.detect_gradient_change()?,
            DetectionMethod::ActivationPattern => self.detect_activation_change()?,
            DetectionMethod::ConfidenceChange => self.detect_confidence_change()?,
            DetectionMethod::Combined => self.detect_combined()?,
        };

        if boundary_score > self.config.threshold {
            self.last_boundary_sample = self.sample_count;
            Ok(Some(boundary_score))
        } else {
            Ok(None)
        }
    }

    /// Detect boundary based on loss increase
    fn detect_loss_increase(&self) -> Result<f32> {
        if self.loss_history.len() < self.config.window_size / 2 {
            return Ok(0.0);
        }

        let mid_point = self.loss_history.len() / 2;
        let recent_avg: f32 = self.loss_history.iter().skip(mid_point).sum::<f32>()
            / (self.loss_history.len() - mid_point) as f32;
        let old_avg: f32 = self.loss_history.iter().take(mid_point).sum::<f32>() / mid_point as f32;

        let relative_increase = (recent_avg - old_avg) / old_avg.max(1e-8);
        Ok(relative_increase.max(0.0))
    }

    /// Detect boundary based on gradient magnitude change
    fn detect_gradient_change(&self) -> Result<f32> {
        if self.gradient_history.len() < self.config.window_size / 2 {
            return Ok(0.0);
        }

        let mid_point = self.gradient_history.len() / 2;
        let recent_avg: f32 = self.gradient_history.iter().skip(mid_point).sum::<f32>()
            / (self.gradient_history.len() - mid_point) as f32;
        let old_avg: f32 =
            self.gradient_history.iter().take(mid_point).sum::<f32>() / mid_point as f32;

        let relative_change = (recent_avg - old_avg).abs() / old_avg.max(1e-8);
        Ok(relative_change)
    }

    /// Detect boundary based on activation pattern change
    fn detect_activation_change(&self) -> Result<f32> {
        // Placeholder - in practice would analyze activation patterns
        Ok(0.0)
    }

    /// Detect boundary based on confidence change
    fn detect_confidence_change(&self) -> Result<f32> {
        if self.confidence_history.len() < self.config.window_size / 2 {
            return Ok(0.0);
        }

        let mid_point = self.confidence_history.len() / 2;
        let recent_avg: f32 = self.confidence_history.iter().skip(mid_point).sum::<f32>()
            / (self.confidence_history.len() - mid_point) as f32;
        let old_avg: f32 =
            self.confidence_history.iter().take(mid_point).sum::<f32>() / mid_point as f32;

        let confidence_drop = (old_avg - recent_avg) / old_avg.max(1e-8);
        Ok(confidence_drop.max(0.0))
    }

    /// Combined detection using multiple signals
    fn detect_combined(&self) -> Result<f32> {
        let loss_score = self.detect_loss_increase()?;
        let gradient_score = self.detect_gradient_change()?;
        let confidence_score = self.detect_confidence_change()?;

        // Weighted combination
        let combined_score = 0.4 * loss_score + 0.3 * gradient_score + 0.3 * confidence_score;
        Ok(combined_score)
    }

    /// Get detector statistics
    pub fn get_statistics(&self) -> DetectorStats {
        DetectorStats {
            sample_count: self.sample_count,
            running_loss_avg: self.running_loss_avg,
            running_gradient_avg: self.running_gradient_avg,
            running_confidence_avg: self.running_confidence_avg,
            window_size: self.loss_history.len(),
            last_boundary_sample: self.last_boundary_sample,
        }
    }

    /// Reset detector state
    pub fn reset(&mut self) {
        self.loss_history.clear();
        self.gradient_history.clear();
        self.confidence_history.clear();
        self.running_loss_avg = 0.0;
        self.running_gradient_avg = 0.0;
        self.running_confidence_avg = 0.0;
        self.sample_count = 0;
        self.last_boundary_sample = 0;
    }

    /// Force boundary detection
    pub fn force_boundary(&mut self) -> TaskTransition {
        self.last_boundary_sample = self.sample_count;
        TaskTransition {
            from_task: "unknown".to_string(),
            to_task: "unknown".to_string(),
            timestamp: chrono::Utc::now(),
            boundary_score: 1.0,
        }
    }
}

/// Detector statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorStats {
    pub sample_count: usize,
    pub running_loss_avg: f32,
    pub running_gradient_avg: f32,
    pub running_confidence_avg: f32,
    pub window_size: usize,
    pub last_boundary_sample: usize,
}

/// Adaptive threshold management
#[derive(Debug)]
pub struct AdaptiveThreshold {
    base_threshold: f32,
    adaptation_rate: f32,
    false_positive_count: usize,
    false_negative_count: usize,
    total_boundaries: usize,
}

impl AdaptiveThreshold {
    pub fn new(base_threshold: f32, adaptation_rate: f32) -> Self {
        Self {
            base_threshold,
            adaptation_rate,
            false_positive_count: 0,
            false_negative_count: 0,
            total_boundaries: 0,
        }
    }

    /// Update threshold based on feedback
    pub fn update_threshold(&mut self, is_false_positive: bool, is_false_negative: bool) -> f32 {
        if is_false_positive {
            self.false_positive_count += 1;
            self.base_threshold *= 1.0 + self.adaptation_rate;
        } else if is_false_negative {
            self.false_negative_count += 1;
            self.base_threshold *= 1.0 - self.adaptation_rate;
        }

        self.total_boundaries += 1;
        self.base_threshold = self.base_threshold.clamp(0.01, 1.0);
        self.base_threshold
    }

    /// Get current threshold
    pub fn get_threshold(&self) -> f32 {
        self.base_threshold
    }

    /// Get adaptation statistics
    pub fn get_stats(&self) -> (f32, f32, usize) {
        let false_positive_rate =
            self.false_positive_count as f32 / self.total_boundaries.max(1) as f32;
        let false_negative_rate =
            self.false_negative_count as f32 / self.total_boundaries.max(1) as f32;
        (
            false_positive_rate,
            false_negative_rate,
            self.total_boundaries,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_boundary_detector_creation() {
        let config = BoundaryDetectionConfig::default();
        let detector = TaskBoundaryDetector::new(config);

        let stats = detector.get_statistics();
        assert_eq!(stats.sample_count, 0);
        assert_eq!(stats.window_size, 0);
    }

    #[test]
    fn test_boundary_detector_update() {
        let config = BoundaryDetectionConfig::default();
        let mut detector = TaskBoundaryDetector::new(config);

        // Add samples
        for i in 0..10 {
            detector.update(i as f32, (i as f32) * 0.1, 0.9);
        }

        let stats = detector.get_statistics();
        assert_eq!(stats.sample_count, 10);
        assert_eq!(stats.window_size, 10);
        assert!(stats.running_loss_avg > 0.0);
    }

    #[test]
    fn test_loss_increase_detection() {
        let config = BoundaryDetectionConfig {
            window_size: 20,
            threshold: 0.1,
            detection_method: DetectionMethod::LossIncrease,
            min_samples: 10,
            ..Default::default()
        };
        let mut detector = TaskBoundaryDetector::new(config);

        // Add samples with stable loss
        for _i in 0..15 {
            detector.update(1.0, 0.1, 0.9);
        }

        // Add samples with increased loss
        for _i in 0..15 {
            detector.update(2.0, 0.1, 0.9);
        }

        let boundary = detector.detect_boundary().unwrap();
        assert!(boundary.is_some());
        assert!(boundary.unwrap() > 0.1);
    }

    #[test]
    fn test_adaptive_threshold() {
        let mut threshold = AdaptiveThreshold::new(0.5, 0.1);

        let initial_threshold = threshold.get_threshold();
        assert_abs_diff_eq!(initial_threshold, 0.5, epsilon = 1e-6);

        // Update with false positive
        let new_threshold = threshold.update_threshold(true, false);
        assert!(new_threshold > initial_threshold);

        // Update with false negative
        let newer_threshold = threshold.update_threshold(false, true);
        assert!(newer_threshold < new_threshold);
    }

    #[test]
    fn test_combined_detection() {
        let config = BoundaryDetectionConfig {
            window_size: 10,
            threshold: 0.1,
            detection_method: DetectionMethod::Combined,
            min_samples: 5,
            ..Default::default()
        };
        let mut detector = TaskBoundaryDetector::new(config);

        // Add samples - need to ensure we have enough samples in the window
        for i in 0..15 {
            let loss = if i < 10 { 1.0 } else { 1.5 };
            let gradient = if i < 10 { 0.1 } else { 0.15 };
            let confidence = if i < 10 { 0.9 } else { 0.7 };
            detector.update(loss, gradient, confidence);
        }

        let boundary = detector.detect_boundary().unwrap();
        assert!(boundary.is_some());
    }

    #[test]
    fn test_detector_reset() {
        let config = BoundaryDetectionConfig::default();
        let mut detector = TaskBoundaryDetector::new(config);

        // Add samples
        for i in 0..10 {
            detector.update(i as f32, 0.1, 0.9);
        }

        assert_eq!(detector.get_statistics().sample_count, 10);

        detector.reset();
        assert_eq!(detector.get_statistics().sample_count, 0);
        assert_eq!(detector.get_statistics().window_size, 0);
    }
}
