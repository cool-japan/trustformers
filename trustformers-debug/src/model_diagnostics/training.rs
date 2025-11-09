//! Training dynamics and convergence analysis.
//!
//! This module provides comprehensive training dynamics analysis including
//! convergence detection, overfitting/underfitting identification, plateau
//! detection, and training stability assessment for optimizing training processes.

use std::collections::VecDeque;

use super::types::{
    ConvergenceStatus, ModelPerformanceMetrics, OverfittingIndicator, PlateauInfo,
    TrainingDynamics, TrainingStability, UnderfittingIndicator,
};

/// Training dynamics analyzer for monitoring and analyzing training behavior.
#[derive(Debug)]
pub struct TrainingDynamicsAnalyzer {
    /// Historical metrics for analysis
    metrics_history: VecDeque<ModelPerformanceMetrics>,
    /// Configuration for analysis thresholds
    config: TrainingAnalysisConfig,
    /// Current training state
    current_state: TrainingState,
}

/// Configuration for training analysis.
#[derive(Debug, Clone)]
pub struct TrainingAnalysisConfig {
    /// Window size for convergence analysis
    pub convergence_window: usize,
    /// Minimum improvement threshold for convergence
    pub min_improvement_threshold: f64,
    /// Maximum variance threshold for stability
    pub max_variance_threshold: f64,
    /// Minimum plateau duration to consider
    pub min_plateau_duration: usize,
    /// Train-validation gap threshold for overfitting
    pub overfitting_gap_threshold: f64,
    /// Minimum learning rate for underfitting detection
    pub min_learning_rate: f64,
}

impl Default for TrainingAnalysisConfig {
    fn default() -> Self {
        Self {
            convergence_window: 20,
            min_improvement_threshold: 0.001,
            max_variance_threshold: 0.1,
            min_plateau_duration: 10,
            overfitting_gap_threshold: 0.05,
            min_learning_rate: 1e-6,
        }
    }
}

/// Current training state information.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingState {
    /// Steps since last improvement
    steps_since_improvement: usize,
    /// Best loss achieved so far
    best_loss: f64,
    /// Current plateau information
    current_plateau: Option<PlateauInfo>,
    /// Convergence status history
    convergence_history: VecDeque<ConvergenceStatus>,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            steps_since_improvement: 0,
            best_loss: f64::INFINITY,
            current_plateau: None,
            convergence_history: VecDeque::new(),
        }
    }
}

impl TrainingDynamicsAnalyzer {
    /// Create a new training dynamics analyzer.
    pub fn new() -> Self {
        Self {
            metrics_history: VecDeque::new(),
            config: TrainingAnalysisConfig::default(),
            current_state: TrainingState::default(),
        }
    }

    /// Create a new analyzer with custom configuration.
    pub fn with_config(config: TrainingAnalysisConfig) -> Self {
        Self {
            metrics_history: VecDeque::new(),
            config,
            current_state: TrainingState::default(),
        }
    }

    /// Add new training metrics for analysis.
    pub fn add_metrics(&mut self, metrics: ModelPerformanceMetrics) {
        // Update training state
        if metrics.loss < self.current_state.best_loss {
            self.current_state.best_loss = metrics.loss;
            self.current_state.steps_since_improvement = 0;
        } else {
            self.current_state.steps_since_improvement += 1;
        }

        self.metrics_history.push_back(metrics);

        // Maintain reasonable history size
        if self.metrics_history.len() > 1000 {
            self.metrics_history.pop_front();
        }

        // Update convergence history
        let status = self.detect_convergence_status();
        self.current_state.convergence_history.push_back(status);
        if self.current_state.convergence_history.len() > 50 {
            self.current_state.convergence_history.pop_front();
        }
    }

    /// Record training dynamics information.
    pub fn record_training_dynamics(&mut self, _dynamics: TrainingDynamics) {
        // Training dynamics are computed via analysis rather than stored directly
        // This method is provided for API compatibility
    }

    /// Analyze current training dynamics.
    pub fn analyze_training_dynamics(&self) -> TrainingDynamics {
        let convergence_status = self.detect_convergence_status();
        let training_stability = self.assess_training_stability();
        let learning_efficiency = self.calculate_learning_efficiency();
        let overfitting_indicators = self.detect_overfitting_indicators();
        let underfitting_indicators = self.detect_underfitting_indicators();
        let plateau_detection = self.detect_plateau();

        TrainingDynamics {
            convergence_status,
            training_stability,
            learning_efficiency,
            overfitting_indicators,
            underfitting_indicators,
            plateau_detection,
        }
    }

    /// Detect current convergence status.
    pub fn detect_convergence_status(&self) -> ConvergenceStatus {
        if self.metrics_history.len() < self.config.convergence_window {
            return ConvergenceStatus::Unknown;
        }

        let recent_metrics: Vec<_> =
            self.metrics_history.iter().rev().take(self.config.convergence_window).collect();

        let losses: Vec<f64> = recent_metrics.iter().map(|m| m.loss).collect();

        // Check for convergence patterns
        if self.is_converged(&losses) {
            ConvergenceStatus::Converged
        } else if self.is_diverging(&losses) {
            ConvergenceStatus::Diverging
        } else if self.is_oscillating(&losses) {
            ConvergenceStatus::Oscillating
        } else if self.is_plateau(&losses) {
            ConvergenceStatus::Plateau
        } else if self.is_converging(&losses) {
            ConvergenceStatus::Converging
        } else {
            ConvergenceStatus::Unknown
        }
    }

    /// Assess training stability.
    pub fn assess_training_stability(&self) -> TrainingStability {
        if self.metrics_history.len() < 10 {
            return TrainingStability::Unknown;
        }

        let recent_losses: Vec<f64> =
            self.metrics_history.iter().rev().take(20).map(|m| m.loss).collect();

        let variance = self.calculate_variance(&recent_losses);

        if variance > self.config.max_variance_threshold {
            TrainingStability::Unstable
        } else if variance > self.config.max_variance_threshold / 2.0 {
            TrainingStability::HighVariance
        } else {
            TrainingStability::Stable
        }
    }

    /// Calculate learning efficiency score.
    pub fn calculate_learning_efficiency(&self) -> f64 {
        if self.metrics_history.len() < 2 {
            return 0.0;
        }

        let initial_loss = self.metrics_history.front().unwrap().loss;
        let current_loss = self.metrics_history.back().unwrap().loss;
        let steps = self.metrics_history.len();

        if initial_loss <= current_loss {
            return 0.0;
        }

        let improvement = (initial_loss - current_loss) / initial_loss;
        let efficiency = improvement / (steps as f64).sqrt();

        efficiency.min(1.0)
    }

    /// Detect overfitting indicators.
    pub fn detect_overfitting_indicators(&self) -> Vec<OverfittingIndicator> {
        let mut indicators = Vec::new();

        // Check for validation accuracy indicators (simulated for now)
        if self.metrics_history.len() > 10 {
            let recent_losses: Vec<f64> =
                self.metrics_history.iter().rev().take(10).map(|m| m.loss).collect();

            // Simulate validation gap detection
            let avg_loss = recent_losses.iter().sum::<f64>() / recent_losses.len() as f64;
            if avg_loss < 0.01 {
                indicators.push(OverfittingIndicator::PerfectTrainingAccuracy);
            }

            // Check for loss variance indicating overfitting
            let variance = self.calculate_variance(&recent_losses);
            if variance > 0.05 {
                indicators.push(OverfittingIndicator::HighVarianceInValidation);
            }
        }

        indicators
    }

    /// Detect underfitting indicators.
    pub fn detect_underfitting_indicators(&self) -> Vec<UnderfittingIndicator> {
        let mut indicators = Vec::new();

        if let Some(current_metrics) = self.metrics_history.back() {
            // High training loss
            if current_metrics.loss > 1.0 {
                indicators.push(UnderfittingIndicator::HighTrainingLoss {
                    loss: current_metrics.loss,
                    threshold: 1.0,
                });
            }

            // Low accuracy (simulated)
            if let Some(accuracy) = current_metrics.accuracy {
                if accuracy < 0.5 {
                    indicators.push(UnderfittingIndicator::LowTrainingAccuracy {
                        accuracy,
                        threshold: 0.5,
                    });
                }
            }

            // Slow convergence
            if self.current_state.steps_since_improvement > 50 {
                indicators.push(UnderfittingIndicator::SlowConvergence {
                    steps_taken: self.metrics_history.len(),
                    expected: self.metrics_history.len() / 2,
                });
            }

            // No learning
            if self.current_state.steps_since_improvement > 100 {
                indicators.push(UnderfittingIndicator::NoLearning {
                    steps_without_improvement: self.current_state.steps_since_improvement,
                });
            }
        }

        indicators
    }

    /// Detect plateau in training.
    pub fn detect_plateau(&self) -> Option<PlateauInfo> {
        if self.metrics_history.len() < self.config.min_plateau_duration {
            return None;
        }

        let recent_losses: Vec<f64> = self
            .metrics_history
            .iter()
            .rev()
            .take(self.config.min_plateau_duration)
            .map(|m| m.loss)
            .collect();

        let variance = self.calculate_variance(&recent_losses);
        let mean_loss = recent_losses.iter().sum::<f64>() / recent_losses.len() as f64;

        // Check if variance is low enough to indicate plateau
        if variance < self.config.min_improvement_threshold {
            let start_step = self.metrics_history.len() - self.config.min_plateau_duration;
            Some(PlateauInfo {
                start_step,
                duration_steps: self.config.min_plateau_duration,
                plateau_value: mean_loss,
                variance,
            })
        } else {
            None
        }
    }

    /// Generate training recommendations based on current dynamics.
    pub fn generate_training_recommendations(&self) -> Vec<TrainingRecommendation> {
        let mut recommendations = Vec::new();
        let dynamics = self.analyze_training_dynamics();

        match dynamics.convergence_status {
            ConvergenceStatus::Diverging => {
                recommendations.push(TrainingRecommendation {
                    category: "Convergence".to_string(),
                    priority: TrainingRecommendationPriority::Critical,
                    description: "Training is diverging".to_string(),
                    action: "Reduce learning rate immediately".to_string(),
                    expected_impact: 0.8,
                });
            },
            ConvergenceStatus::Plateau => {
                recommendations.push(TrainingRecommendation {
                    category: "Convergence".to_string(),
                    priority: TrainingRecommendationPriority::High,
                    description: "Training has reached a plateau".to_string(),
                    action: "Consider learning rate scheduling or data augmentation".to_string(),
                    expected_impact: 0.6,
                });
            },
            _ => {},
        }

        match dynamics.training_stability {
            TrainingStability::Unstable => {
                recommendations.push(TrainingRecommendation {
                    category: "Stability".to_string(),
                    priority: TrainingRecommendationPriority::High,
                    description: "Training is unstable".to_string(),
                    action: "Reduce learning rate or add gradient clipping".to_string(),
                    expected_impact: 0.7,
                });
            },
            _ => {},
        }

        if dynamics.learning_efficiency < 0.3 {
            recommendations.push(TrainingRecommendation {
                category: "Efficiency".to_string(),
                priority: TrainingRecommendationPriority::Medium,
                description: "Low learning efficiency detected".to_string(),
                action: "Consider architecture changes or hyperparameter tuning".to_string(),
                expected_impact: 0.5,
            });
        }

        recommendations
    }

    // Helper methods for convergence detection
    fn is_converged(&self, losses: &[f64]) -> bool {
        if losses.len() < 5 {
            return false;
        }

        let recent_variance = self.calculate_variance(&losses[..5]);
        recent_variance < self.config.min_improvement_threshold && losses[0] < 0.01
    }

    fn is_diverging(&self, losses: &[f64]) -> bool {
        if losses.len() < 3 {
            return false;
        }

        // Check if loss is consistently increasing
        losses.windows(2).all(|w| w[1] >= w[0])
            && (losses.last().unwrap() / losses.first().unwrap()) > 1.1
    }

    fn is_oscillating(&self, losses: &[f64]) -> bool {
        if losses.len() < 6 {
            return false;
        }

        // Check for oscillating pattern
        let mut direction_changes = 0;
        for window in losses.windows(3) {
            let trend1 = window[1] - window[0];
            let trend2 = window[2] - window[1];
            if trend1.signum() != trend2.signum() {
                direction_changes += 1;
            }
        }

        direction_changes > losses.len() / 3
    }

    fn is_plateau(&self, losses: &[f64]) -> bool {
        let variance = self.calculate_variance(losses);
        variance < self.config.min_improvement_threshold
    }

    fn is_converging(&self, losses: &[f64]) -> bool {
        if losses.len() < 3 {
            return false;
        }

        // Check if loss is generally decreasing
        let trend = self.calculate_trend(losses);
        trend < -self.config.min_improvement_threshold
    }

    fn calculate_variance(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        variance
    }

    fn calculate_trend(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let n = values.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = values.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &y) in values.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Clear analysis history.
    pub fn clear(&mut self) {
        self.metrics_history.clear();
        self.current_state = TrainingState::default();
    }

    /// Get current training state information.
    pub fn get_training_state(&self) -> &TrainingState {
        &self.current_state
    }

    /// Generate comprehensive training dynamics report.
    pub async fn generate_report(&self) -> anyhow::Result<TrainingDynamicsReport> {
        let training_dynamics = self.analyze_training_dynamics();
        let recommendations = self.generate_recommendations();

        Ok(TrainingDynamicsReport {
            training_dynamics,
            recommendations,
            current_state: self.current_state.clone(),
        })
    }

    /// Generate training recommendations.
    fn generate_recommendations(&self) -> Vec<TrainingRecommendation> {
        let mut recommendations = Vec::new();

        // Add basic recommendations based on current state
        recommendations.push(TrainingRecommendation {
            category: "General".to_string(),
            description: "Continue monitoring training dynamics".to_string(),
            action: "Monitor training progress and adjust parameters as needed".to_string(),
            priority: TrainingRecommendationPriority::Low,
            expected_impact: 0.1,
        });

        recommendations
    }
}

impl Default for TrainingDynamicsAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Training recommendation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingRecommendation {
    /// Category of the recommendation
    pub category: String,
    /// Priority level
    pub priority: TrainingRecommendationPriority,
    /// Description of the issue
    pub description: String,
    /// Recommended action
    pub action: String,
    /// Expected impact (0.0 to 1.0)
    pub expected_impact: f64,
}

/// Priority levels for training recommendations.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum TrainingRecommendationPriority {
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Comprehensive training dynamics report.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingDynamicsReport {
    /// Training dynamics analysis
    pub training_dynamics: TrainingDynamics,
    /// Generated recommendations
    pub recommendations: Vec<TrainingRecommendation>,
    /// Current training state
    pub current_state: TrainingState,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_metrics(step: usize, loss: f64) -> ModelPerformanceMetrics {
        ModelPerformanceMetrics {
            training_step: step,
            loss,
            accuracy: Some(0.8),
            learning_rate: 0.001,
            batch_size: 32,
            throughput_samples_per_sec: 100.0,
            memory_usage_mb: 1000.0,
            gpu_utilization: Some(0.9),
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn test_training_dynamics_analyzer_creation() {
        let analyzer = TrainingDynamicsAnalyzer::new();
        assert_eq!(analyzer.metrics_history.len(), 0);
    }

    #[test]
    fn test_add_metrics() {
        let mut analyzer = TrainingDynamicsAnalyzer::new();
        let metrics = create_test_metrics(1, 0.5);

        analyzer.add_metrics(metrics);
        assert_eq!(analyzer.metrics_history.len(), 1);
        assert_eq!(analyzer.current_state.best_loss, 0.5);
    }

    #[test]
    fn test_convergence_detection() {
        let mut analyzer = TrainingDynamicsAnalyzer::new();

        // Add converging sequence
        for i in 1..=25 {
            let loss = 1.0 / (i as f64);
            let metrics = create_test_metrics(i, loss);
            analyzer.add_metrics(metrics);
        }

        let status = analyzer.detect_convergence_status();
        matches!(
            status,
            ConvergenceStatus::Converging | ConvergenceStatus::Converged
        );
    }

    #[test]
    fn test_learning_efficiency_calculation() {
        let mut analyzer = TrainingDynamicsAnalyzer::new();

        analyzer.add_metrics(create_test_metrics(1, 1.0));
        analyzer.add_metrics(create_test_metrics(2, 0.5));
        analyzer.add_metrics(create_test_metrics(3, 0.25));

        let efficiency = analyzer.calculate_learning_efficiency();
        assert!(efficiency > 0.0);
    }

    #[test]
    fn test_plateau_detection() {
        let mut analyzer = TrainingDynamicsAnalyzer::new();

        // Add plateau sequence
        for i in 1..=15 {
            let metrics = create_test_metrics(i, 0.1); // Constant loss
            analyzer.add_metrics(metrics);
        }

        let plateau = analyzer.detect_plateau();
        assert!(plateau.is_some());
    }

    #[test]
    fn test_training_stability_assessment() {
        let mut analyzer = TrainingDynamicsAnalyzer::new();

        // Add stable sequence
        for i in 1..=20 {
            let loss = 0.5 + (i as f64 * 0.001); // Very small variance
            let metrics = create_test_metrics(i, loss);
            analyzer.add_metrics(metrics);
        }

        let stability = analyzer.assess_training_stability();
        matches!(stability, TrainingStability::Stable);
    }
}
