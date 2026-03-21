//! Model and training health assessment system

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, SystemTime};

use crate::{DashboardMetrics, DebugConfig};

/// Comprehensive health checker for model training
#[derive(Debug)]
pub struct HealthChecker {
    #[allow(dead_code)]
    config: DebugConfig,
    metrics_history: VecDeque<DashboardMetrics>,
    health_assessments: Vec<HealthAssessment>,
    stability_tracker: StabilityTracker,
    convergence_analyzer: ConvergenceAnalyzer,
    overfitting_detector: OverfittingDetector,
    generalization_monitor: GeneralizationMonitor,
    performance_baseline: Option<PerformanceBaseline>,
}

/// Overall health assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthAssessment {
    pub timestamp: SystemTime,
    pub overall_health_score: f64,
    pub training_stability_index: f64,
    pub convergence_probability: f64,
    pub overfitting_risk: OverfittingRisk,
    pub generalization_score: f64,
    pub component_scores: ComponentHealthScores,
    pub health_status: HealthStatus,
    pub alerts: Vec<HealthAlert>,
    pub recommendations: Vec<HealthRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealthScores {
    pub gradient_health: f64,
    pub loss_health: f64,
    pub accuracy_health: f64,
    pub performance_health: f64,
    pub memory_health: f64,
    pub stability_health: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Excellent, // 90-100%
    Good,      // 75-89%
    Fair,      // 60-74%
    Poor,      // 40-59%
    Critical,  // 0-39%
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthAlert {
    pub alert_type: HealthAlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub metric_value: f64,
    pub threshold: f64,
    pub trend: Trend,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthAlertType {
    TrainingStability,
    ConvergenceIssue,
    OverfittingDetected,
    PerformanceDegradation,
    MemoryIssue,
    GradientProblem,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Trend {
    Improving,
    Stable,
    Degrading,
    Volatile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthRecommendation {
    pub category: RecommendationCategory,
    pub title: String,
    pub description: String,
    pub urgency: RecommendationUrgency,
    pub expected_impact: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationCategory {
    Training,
    Architecture,
    Hyperparameters,
    Data,
    Performance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationUrgency {
    Immediate,
    Soon,
    Eventually,
    Optional,
}

/// Training stability tracking
#[derive(Debug)]
pub struct StabilityTracker {
    loss_stability: MetricStability,
    accuracy_stability: MetricStability,
    gradient_stability: MetricStability,
    learning_rate_stability: MetricStability,
    #[allow(dead_code)]
    window_size: usize,
}

#[derive(Debug)]
pub struct MetricStability {
    values: VecDeque<f64>,
    variance_threshold: f64,
    #[allow(dead_code)]
    trend_threshold: f64,
}

impl MetricStability {
    pub fn new(variance_threshold: f64, trend_threshold: f64) -> Self {
        Self {
            values: VecDeque::new(),
            variance_threshold,
            trend_threshold,
        }
    }

    pub fn update(&mut self, value: f64) {
        self.values.push_back(value);
        if self.values.len() > 50 {
            self.values.pop_front();
        }
    }

    pub fn calculate_stability(&self) -> f64 {
        if self.values.len() < 5 {
            return 0.5; // Insufficient data
        }

        let variance = self.calculate_variance();
        let trend_stability = self.calculate_trend_stability();

        let variance_score = if variance < self.variance_threshold {
            1.0
        } else {
            (self.variance_threshold / variance).min(1.0)
        };

        (variance_score + trend_stability) / 2.0
    }

    fn calculate_variance(&self) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }

        let mean = self.values.iter().sum::<f64>() / self.values.len() as f64;
        let variance = self.values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / (self.values.len() - 1) as f64;
        variance
    }

    fn calculate_trend_stability(&self) -> f64 {
        if self.values.len() < 10 {
            return 0.5;
        }

        // Calculate slope changes to detect instability
        let mut slope_changes = 0;
        let values: Vec<f64> = self.values.iter().cloned().collect();

        for i in 2..values.len() {
            let slope1 = values[i - 1] - values[i - 2];
            let slope2 = values[i] - values[i - 1];

            if (slope1 > 0.0) != (slope2 > 0.0) {
                slope_changes += 1;
            }
        }

        let change_rate = slope_changes as f64 / (values.len() - 2) as f64;
        (1.0 - change_rate).max(0.0)
    }
}

/// Convergence analysis
#[derive(Debug)]
pub struct ConvergenceAnalyzer {
    loss_history: VecDeque<f64>,
    accuracy_history: VecDeque<f64>,
    convergence_window: usize,
    convergence_threshold: f64,
}

impl Default for ConvergenceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl ConvergenceAnalyzer {
    pub fn new() -> Self {
        Self {
            loss_history: VecDeque::new(),
            accuracy_history: VecDeque::new(),
            convergence_window: 100,
            convergence_threshold: 0.01,
        }
    }

    pub fn update(&mut self, loss: Option<f64>, accuracy: Option<f64>) {
        if let Some(loss) = loss {
            self.loss_history.push_back(loss);
            if self.loss_history.len() > self.convergence_window * 2 {
                self.loss_history.pop_front();
            }
        }

        if let Some(accuracy) = accuracy {
            self.accuracy_history.push_back(accuracy);
            if self.accuracy_history.len() > self.convergence_window * 2 {
                self.accuracy_history.pop_front();
            }
        }
    }

    pub fn calculate_convergence_probability(&self) -> f64 {
        let loss_convergence = self.analyze_loss_convergence();
        let accuracy_convergence = self.analyze_accuracy_convergence();

        // Weight loss convergence more heavily
        0.7 * loss_convergence + 0.3 * accuracy_convergence
    }

    fn analyze_loss_convergence(&self) -> f64 {
        if self.loss_history.len() < self.convergence_window {
            return 0.3; // Insufficient data
        }

        let recent_window = self.convergence_window / 2;
        let recent_losses: Vec<f64> =
            self.loss_history.iter().rev().take(recent_window).cloned().collect();

        let earlier_losses: Vec<f64> = self
            .loss_history
            .iter()
            .rev()
            .skip(recent_window)
            .take(recent_window)
            .cloned()
            .collect();

        if recent_losses.is_empty() || earlier_losses.is_empty() {
            return 0.3;
        }

        let recent_avg = recent_losses.iter().sum::<f64>() / recent_losses.len() as f64;
        let earlier_avg = earlier_losses.iter().sum::<f64>() / earlier_losses.len() as f64;

        // Check if loss is decreasing
        if recent_avg < earlier_avg {
            let improvement_rate = (earlier_avg - recent_avg) / earlier_avg;

            if improvement_rate > self.convergence_threshold {
                0.8 // Good convergence
            } else {
                0.6 // Slow convergence
            }
        } else {
            // Loss increasing or stagnant
            let variance = self.calculate_variance(&recent_losses);
            if variance < self.convergence_threshold {
                0.4 // Converged but not improving
            } else {
                0.2 // Diverging or unstable
            }
        }
    }

    fn analyze_accuracy_convergence(&self) -> f64 {
        if self.accuracy_history.len() < self.convergence_window {
            return 0.5;
        }

        let recent_window = self.convergence_window / 2;
        let recent_accuracy: Vec<f64> =
            self.accuracy_history.iter().rev().take(recent_window).cloned().collect();

        let variance = self.calculate_variance(&recent_accuracy);

        // Low variance in accuracy suggests convergence
        if variance < 0.01 {
            0.8
        } else if variance < 0.05 {
            0.6
        } else {
            0.4
        }
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
}

/// Overfitting detection
#[derive(Debug)]
pub struct OverfittingDetector {
    train_loss_history: VecDeque<f64>,
    val_loss_history: VecDeque<f64>,
    train_accuracy_history: VecDeque<f64>,
    val_accuracy_history: VecDeque<f64>,
    overfitting_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OverfittingRisk {
    None,
    Low,
    Medium,
    High,
    Severe,
}

impl Default for OverfittingDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl OverfittingDetector {
    pub fn new() -> Self {
        Self {
            train_loss_history: VecDeque::new(),
            val_loss_history: VecDeque::new(),
            train_accuracy_history: VecDeque::new(),
            val_accuracy_history: VecDeque::new(),
            overfitting_threshold: 0.1,
        }
    }

    pub fn update_train_metrics(&mut self, loss: Option<f64>, accuracy: Option<f64>) {
        if let Some(loss) = loss {
            self.train_loss_history.push_back(loss);
            if self.train_loss_history.len() > 100 {
                self.train_loss_history.pop_front();
            }
        }

        if let Some(accuracy) = accuracy {
            self.train_accuracy_history.push_back(accuracy);
            if self.train_accuracy_history.len() > 100 {
                self.train_accuracy_history.pop_front();
            }
        }
    }

    pub fn update_validation_metrics(&mut self, loss: Option<f64>, accuracy: Option<f64>) {
        if let Some(loss) = loss {
            self.val_loss_history.push_back(loss);
            if self.val_loss_history.len() > 100 {
                self.val_loss_history.pop_front();
            }
        }

        if let Some(accuracy) = accuracy {
            self.val_accuracy_history.push_back(accuracy);
            if self.val_accuracy_history.len() > 100 {
                self.val_accuracy_history.pop_front();
            }
        }
    }

    pub fn detect_overfitting(&self) -> OverfittingRisk {
        let loss_gap = self.calculate_loss_gap();
        let accuracy_gap = self.calculate_accuracy_gap();
        let trend_analysis = self.analyze_overfitting_trend();

        let overfitting_score = (loss_gap + accuracy_gap + trend_analysis) / 3.0;

        match overfitting_score {
            score if score > 0.8 => OverfittingRisk::Severe,
            score if score > 0.6 => OverfittingRisk::High,
            score if score > 0.4 => OverfittingRisk::Medium,
            score if score > 0.2 => OverfittingRisk::Low,
            _ => OverfittingRisk::None,
        }
    }

    fn calculate_loss_gap(&self) -> f64 {
        if self.train_loss_history.len() < 10 || self.val_loss_history.len() < 10 {
            return 0.0;
        }

        let recent_train_loss = self.train_loss_history.iter().rev().take(10).sum::<f64>() / 10.0;

        let recent_val_loss = self.val_loss_history.iter().rev().take(10).sum::<f64>() / 10.0;

        if recent_train_loss < recent_val_loss {
            let gap = (recent_val_loss - recent_train_loss) / recent_train_loss;
            (gap / self.overfitting_threshold).min(1.0)
        } else {
            0.0
        }
    }

    fn calculate_accuracy_gap(&self) -> f64 {
        if self.train_accuracy_history.len() < 10 || self.val_accuracy_history.len() < 10 {
            return 0.0;
        }

        let recent_train_acc =
            self.train_accuracy_history.iter().rev().take(10).sum::<f64>() / 10.0;

        let recent_val_acc = self.val_accuracy_history.iter().rev().take(10).sum::<f64>() / 10.0;

        if recent_train_acc > recent_val_acc {
            let gap = recent_train_acc - recent_val_acc;
            (gap / self.overfitting_threshold).min(1.0)
        } else {
            0.0
        }
    }

    fn analyze_overfitting_trend(&self) -> f64 {
        // Analyze if the gap between train and validation is increasing
        if self.train_loss_history.len() < 20 || self.val_loss_history.len() < 20 {
            return 0.0;
        }

        let early_train_loss = self.train_loss_history.iter().take(10).sum::<f64>() / 10.0;

        let recent_train_loss = self.train_loss_history.iter().rev().take(10).sum::<f64>() / 10.0;

        let early_val_loss = self.val_loss_history.iter().take(10).sum::<f64>() / 10.0;

        let recent_val_loss = self.val_loss_history.iter().rev().take(10).sum::<f64>() / 10.0;

        let early_gap = (early_val_loss - early_train_loss).max(0.0);
        let recent_gap = (recent_val_loss - recent_train_loss).max(0.0);

        if recent_gap > early_gap && early_gap > 0.0 {
            ((recent_gap - early_gap) / early_gap).min(1.0)
        } else {
            0.0
        }
    }
}

/// Generalization monitoring
#[derive(Debug)]
pub struct GeneralizationMonitor {
    cross_validation_scores: Vec<f64>,
    holdout_performance: Option<f64>,
    train_performance: Option<f64>,
    complexity_metrics: ComplexityMetrics,
}

#[derive(Debug)]
pub struct ComplexityMetrics {
    parameter_count: usize,
    #[allow(dead_code)]
    effective_capacity: f64,
    data_size: usize,
}

impl Default for GeneralizationMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl GeneralizationMonitor {
    pub fn new() -> Self {
        Self {
            cross_validation_scores: Vec::new(),
            holdout_performance: None,
            train_performance: None,
            complexity_metrics: ComplexityMetrics {
                parameter_count: 0,
                effective_capacity: 0.0,
                data_size: 0,
            },
        }
    }

    pub fn update_performance(&mut self, train_perf: f64, val_perf: Option<f64>) {
        self.train_performance = Some(train_perf);
        if let Some(val_perf) = val_perf {
            self.holdout_performance = Some(val_perf);
        }
    }

    pub fn calculate_generalization_score(&self) -> f64 {
        let performance_consistency = self.calculate_performance_consistency();
        let complexity_penalty = self.calculate_complexity_penalty();
        let cv_consistency = self.calculate_cv_consistency();

        (performance_consistency + cv_consistency + (1.0 - complexity_penalty)) / 3.0
    }

    fn calculate_performance_consistency(&self) -> f64 {
        match (self.train_performance, self.holdout_performance) {
            (Some(train), Some(val)) => {
                let gap = (train - val).abs();
                (1.0 - gap.min(1.0)).max(0.0)
            },
            _ => 0.5, // Unknown
        }
    }

    fn calculate_complexity_penalty(&self) -> f64 {
        // Simplified complexity penalty based on parameter count vs data size
        if self.complexity_metrics.data_size == 0 {
            return 0.0;
        }

        let param_per_sample = self.complexity_metrics.parameter_count as f64
            / self.complexity_metrics.data_size as f64;

        if param_per_sample > 1.0 {
            0.8 // High complexity
        } else if param_per_sample > 0.1 {
            0.4 // Medium complexity
        } else {
            0.1 // Low complexity
        }
    }

    fn calculate_cv_consistency(&self) -> f64 {
        if self.cross_validation_scores.len() < 3 {
            return 0.5;
        }

        let mean = self.cross_validation_scores.iter().sum::<f64>()
            / self.cross_validation_scores.len() as f64;
        let variance = self.cross_validation_scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / (self.cross_validation_scores.len() - 1) as f64;

        (1.0 - variance.sqrt().min(1.0)).max(0.0)
    }
}

/// Performance baseline for comparison
#[derive(Debug, Clone)]
pub struct PerformanceBaseline {
    pub baseline_loss: f64,
    pub baseline_accuracy: f64,
    pub baseline_training_time: Duration,
    pub baseline_memory_usage: f64,
    pub established_at: SystemTime,
}

impl HealthChecker {
    /// Create new health checker
    pub fn new(config: &DebugConfig) -> Self {
        Self {
            config: config.clone(),
            metrics_history: VecDeque::new(),
            health_assessments: Vec::new(),
            stability_tracker: StabilityTracker {
                loss_stability: MetricStability::new(0.1, 0.05),
                accuracy_stability: MetricStability::new(0.01, 0.02),
                gradient_stability: MetricStability::new(1.0, 0.1),
                learning_rate_stability: MetricStability::new(0.0001, 0.001),
                window_size: 50,
            },
            convergence_analyzer: ConvergenceAnalyzer::new(),
            overfitting_detector: OverfittingDetector::new(),
            generalization_monitor: GeneralizationMonitor::new(),
            performance_baseline: None,
        }
    }

    /// Update health checker with new metrics
    pub fn update(&mut self, metrics: DashboardMetrics) {
        self.metrics_history.push_back(metrics.clone());

        // Keep only recent metrics to prevent unbounded growth
        if self.metrics_history.len() > 1000 {
            self.metrics_history.pop_front();
        }

        // Update stability tracking
        if let Some(loss) = metrics.loss {
            self.stability_tracker.loss_stability.update(loss);
        }
        if let Some(accuracy) = metrics.accuracy {
            self.stability_tracker.accuracy_stability.update(accuracy);
        }
        if let Some(grad_norm) = metrics.gradient_norm {
            self.stability_tracker.gradient_stability.update(grad_norm);
        }
        if let Some(lr) = metrics.learning_rate {
            self.stability_tracker.learning_rate_stability.update(lr);
        }

        // Update convergence analysis
        self.convergence_analyzer.update(metrics.loss, metrics.accuracy);

        // Update overfitting detection (assuming these are training metrics)
        self.overfitting_detector.update_train_metrics(metrics.loss, metrics.accuracy);

        // Update generalization monitoring
        if let (Some(accuracy), Some(loss)) = (metrics.accuracy, metrics.loss) {
            self.generalization_monitor.update_performance(accuracy, Some(1.0 - loss));
        }
    }

    /// Perform comprehensive health assessment
    pub fn assess_health(&mut self) -> Result<HealthAssessment> {
        let overall_health_score = self.calculate_overall_health_score();
        let training_stability_index = self.calculate_training_stability_index();
        let convergence_probability = self.convergence_analyzer.calculate_convergence_probability();
        let overfitting_risk = self.overfitting_detector.detect_overfitting();
        let generalization_score = self.generalization_monitor.calculate_generalization_score();

        let component_scores = self.calculate_component_scores();
        let health_status = self.determine_health_status(overall_health_score);
        let alerts = self.generate_health_alerts();
        let recommendations = self.generate_health_recommendations(&alerts);

        let assessment = HealthAssessment {
            timestamp: SystemTime::now(),
            overall_health_score,
            training_stability_index,
            convergence_probability,
            overfitting_risk,
            generalization_score,
            component_scores,
            health_status,
            alerts,
            recommendations,
        };

        self.health_assessments.push(assessment.clone());

        // Keep only recent assessments
        if self.health_assessments.len() > 100 {
            self.health_assessments.drain(0..50);
        }

        Ok(assessment)
    }

    fn calculate_overall_health_score(&self) -> f64 {
        let component_scores = self.calculate_component_scores();

        // Weighted average of component scores
        let weights = [
            ("stability", 0.25),
            ("convergence", 0.20),
            ("gradient", 0.15),
            ("loss", 0.15),
            ("accuracy", 0.10),
            ("performance", 0.10),
            ("memory", 0.05),
        ];

        let mut weighted_sum = 0.0;
        weighted_sum += weights[0].1 * component_scores.stability_health;
        weighted_sum +=
            weights[1].1 * self.convergence_analyzer.calculate_convergence_probability();
        weighted_sum += weights[2].1 * component_scores.gradient_health;
        weighted_sum += weights[3].1 * component_scores.loss_health;
        weighted_sum += weights[4].1 * component_scores.accuracy_health;
        weighted_sum += weights[5].1 * component_scores.performance_health;
        weighted_sum += weights[6].1 * component_scores.memory_health;

        weighted_sum
    }

    fn calculate_training_stability_index(&self) -> f64 {
        let loss_stability = self.stability_tracker.loss_stability.calculate_stability();
        let accuracy_stability = self.stability_tracker.accuracy_stability.calculate_stability();
        let gradient_stability = self.stability_tracker.gradient_stability.calculate_stability();
        let lr_stability = self.stability_tracker.learning_rate_stability.calculate_stability();

        (loss_stability + accuracy_stability + gradient_stability + lr_stability) / 4.0
    }

    fn calculate_component_scores(&self) -> ComponentHealthScores {
        ComponentHealthScores {
            gradient_health: self.stability_tracker.gradient_stability.calculate_stability(),
            loss_health: self.calculate_loss_health(),
            accuracy_health: self.calculate_accuracy_health(),
            performance_health: self.calculate_performance_health(),
            memory_health: self.calculate_memory_health(),
            stability_health: self.calculate_training_stability_index(),
        }
    }

    fn calculate_loss_health(&self) -> f64 {
        if self.metrics_history.len() < 10 {
            return 0.5;
        }

        let recent_losses: Vec<f64> =
            self.metrics_history.iter().rev().take(10).filter_map(|m| m.loss).collect();

        if recent_losses.is_empty() {
            return 0.5;
        }

        // Check if loss is generally decreasing
        let first_half_avg = recent_losses[..recent_losses.len() / 2].iter().sum::<f64>()
            / (recent_losses.len() / 2) as f64;
        let second_half_avg = recent_losses[recent_losses.len() / 2..].iter().sum::<f64>()
            / (recent_losses.len() - recent_losses.len() / 2) as f64;

        if second_half_avg < first_half_avg {
            0.8 // Loss decreasing
        } else if (second_half_avg - first_half_avg).abs() / first_half_avg < 0.05 {
            0.6 // Loss stable
        } else {
            0.3 // Loss increasing
        }
    }

    fn calculate_accuracy_health(&self) -> f64 {
        if self.metrics_history.len() < 10 {
            return 0.5;
        }

        let recent_accuracies: Vec<f64> =
            self.metrics_history.iter().rev().take(10).filter_map(|m| m.accuracy).collect();

        if recent_accuracies.is_empty() {
            return 0.5;
        }

        let avg_accuracy = recent_accuracies.iter().sum::<f64>() / recent_accuracies.len() as f64;

        // Score based on absolute accuracy and stability
        let accuracy_score = avg_accuracy; // Assuming accuracy is 0-1
        let stability_score = self.stability_tracker.accuracy_stability.calculate_stability();

        (accuracy_score + stability_score) / 2.0
    }

    fn calculate_performance_health(&self) -> f64 {
        // Check tokens per second and GPU utilization
        if let Some(last_metrics) = self.metrics_history.back() {
            let mut score = 0.0;
            let mut components = 0;

            if let Some(tps) = last_metrics.tokens_per_second {
                score += if tps > 100.0 { 0.8 } else { tps / 125.0 };
                components += 1;
            }

            if let Some(gpu_util) = last_metrics.gpu_utilization {
                score += gpu_util;
                components += 1;
            }

            if components > 0 {
                score / components as f64
            } else {
                0.5
            }
        } else {
            0.5
        }
    }

    fn calculate_memory_health(&self) -> f64 {
        if let Some(last_metrics) = self.metrics_history.back() {
            let memory_usage = last_metrics.memory_usage_mb;

            // Assume 8GB as reasonable upper limit

            if memory_usage < 4096.0 {
                0.9
            } else if memory_usage < 6144.0 {
                0.7
            } else if memory_usage < 8192.0 {
                0.5
            } else {
                0.2
            }
        } else {
            0.5
        }
    }

    fn determine_health_status(&self, score: f64) -> HealthStatus {
        match score {
            s if s >= 0.9 => HealthStatus::Excellent,
            s if s >= 0.75 => HealthStatus::Good,
            s if s >= 0.6 => HealthStatus::Fair,
            s if s >= 0.4 => HealthStatus::Poor,
            _ => HealthStatus::Critical,
        }
    }

    fn generate_health_alerts(&self) -> Vec<HealthAlert> {
        let mut alerts = Vec::new();

        // Training stability alerts
        let stability_index = self.calculate_training_stability_index();
        if stability_index < 0.3 {
            alerts.push(HealthAlert {
                alert_type: HealthAlertType::TrainingStability,
                severity: AlertSeverity::High,
                message: "Training is highly unstable".to_string(),
                metric_value: stability_index,
                threshold: 0.3,
                trend: Trend::Degrading,
            });
        }

        // Convergence alerts
        let convergence_prob = self.convergence_analyzer.calculate_convergence_probability();
        if convergence_prob < 0.2 {
            alerts.push(HealthAlert {
                alert_type: HealthAlertType::ConvergenceIssue,
                severity: AlertSeverity::Medium,
                message: "Low probability of convergence".to_string(),
                metric_value: convergence_prob,
                threshold: 0.2,
                trend: Trend::Stable,
            });
        }

        // Overfitting alerts
        match self.overfitting_detector.detect_overfitting() {
            OverfittingRisk::High | OverfittingRisk::Severe => {
                alerts.push(HealthAlert {
                    alert_type: HealthAlertType::OverfittingDetected,
                    severity: AlertSeverity::High,
                    message: "Significant overfitting detected".to_string(),
                    metric_value: 0.8,
                    threshold: 0.6,
                    trend: Trend::Degrading,
                });
            },
            OverfittingRisk::Medium => {
                alerts.push(HealthAlert {
                    alert_type: HealthAlertType::OverfittingDetected,
                    severity: AlertSeverity::Medium,
                    message: "Moderate overfitting risk".to_string(),
                    metric_value: 0.5,
                    threshold: 0.4,
                    trend: Trend::Stable,
                });
            },
            _ => {},
        }

        alerts
    }

    fn generate_health_recommendations(&self, alerts: &[HealthAlert]) -> Vec<HealthRecommendation> {
        let mut recommendations = Vec::new();

        for alert in alerts {
            match alert.alert_type {
                HealthAlertType::TrainingStability => {
                    recommendations.push(HealthRecommendation {
                        category: RecommendationCategory::Training,
                        title: "Improve Training Stability".to_string(),
                        description:
                            "Reduce learning rate or increase batch size to stabilize training"
                                .to_string(),
                        urgency: RecommendationUrgency::Soon,
                        expected_impact: 0.3,
                    });
                },
                HealthAlertType::ConvergenceIssue => {
                    recommendations.push(HealthRecommendation {
                        category: RecommendationCategory::Hyperparameters,
                        title: "Adjust Learning Rate Schedule".to_string(),
                        description:
                            "Implement learning rate scheduling or adjust optimizer settings"
                                .to_string(),
                        urgency: RecommendationUrgency::Eventually,
                        expected_impact: 0.2,
                    });
                },
                HealthAlertType::OverfittingDetected => {
                    recommendations.push(HealthRecommendation {
                        category: RecommendationCategory::Training,
                        title: "Add Regularization".to_string(),
                        description: "Implement dropout, weight decay, or early stopping to reduce overfitting".to_string(),
                        urgency: RecommendationUrgency::Soon,
                        expected_impact: 0.25,
                    });
                },
                _ => {},
            }
        }

        // Add general recommendations based on overall health
        let overall_score = self.calculate_overall_health_score();
        if overall_score < 0.6 {
            recommendations.push(HealthRecommendation {
                category: RecommendationCategory::Training,
                title: "Comprehensive Training Review".to_string(),
                description: "Review entire training setup including data, model architecture, and hyperparameters".to_string(),
                urgency: RecommendationUrgency::Immediate,
                expected_impact: 0.4,
            });
        }

        recommendations
    }

    /// Set performance baseline for comparison
    pub fn set_baseline(&mut self, baseline: PerformanceBaseline) {
        self.performance_baseline = Some(baseline);
    }

    /// Get health assessment history
    pub fn get_health_history(&self) -> &[HealthAssessment] {
        &self.health_assessments
    }

    /// Quick health check for simplified interface
    pub async fn quick_health_check(&self) -> Result<crate::QuickHealthSummary> {
        let score = if let Some(assessment) = self.health_assessments.last() {
            assessment.overall_health_score * 100.0
        } else {
            // If no assessments yet, do a basic check
            50.0 // Default fair score
        };

        let status = match score {
            90.0..=100.0 => "Excellent",
            75.0..89.9 => "Good",
            60.0..74.9 => "Fair",
            40.0..59.9 => "Poor",
            _ => "Critical",
        }
        .to_string();

        let mut recommendations = Vec::new();
        if score < 60.0 {
            recommendations.push("Review training configuration and data quality".to_string());
        }
        if score < 40.0 {
            recommendations
                .push("Consider adjusting learning rate and model architecture".to_string());
        }
        if score < 80.0 {
            recommendations.push("Monitor training stability and convergence".to_string());
        }

        Ok(crate::QuickHealthSummary {
            score,
            status,
            recommendations,
        })
    }

    /// Generate health report
    pub async fn generate_report(&self) -> Result<HealthReport> {
        let current_assessment = if let Some(assessment) = self.health_assessments.last() {
            assessment.clone()
        } else {
            return Ok(HealthReport::default());
        };

        let health_trends = self.analyze_health_trends();
        let risk_assessment = self.assess_risks();
        let improvement_suggestions = self.generate_improvement_suggestions();

        Ok(HealthReport {
            current_health: current_assessment,
            health_trends,
            risk_assessment,
            improvement_suggestions,
            baseline_comparison: self.compare_with_baseline(),
            summary: self.generate_health_summary(),
        })
    }

    fn analyze_health_trends(&self) -> HealthTrends {
        if self.health_assessments.len() < 5 {
            return HealthTrends::default();
        }

        let recent_scores: Vec<f64> = self
            .health_assessments
            .iter()
            .rev()
            .take(10)
            .map(|a| a.overall_health_score)
            .collect();

        let first_half_avg = recent_scores[recent_scores.len() / 2..].iter().sum::<f64>()
            / (recent_scores.len() - recent_scores.len() / 2) as f64;
        let second_half_avg = recent_scores[..recent_scores.len() / 2].iter().sum::<f64>()
            / (recent_scores.len() / 2) as f64;

        let trend = if second_half_avg > first_half_avg * 1.05 {
            Trend::Improving
        } else if second_half_avg < first_half_avg * 0.95 {
            Trend::Degrading
        } else {
            Trend::Stable
        };

        HealthTrends {
            overall_trend: trend,
            stability_trend: Trend::Stable, // Simplified
            convergence_trend: Trend::Stable,
            overfitting_trend: Trend::Stable,
        }
    }

    fn assess_risks(&self) -> Vec<HealthRisk> {
        let mut risks = Vec::new();

        if let Some(current) = self.health_assessments.last() {
            if current.overall_health_score < 0.4 {
                risks.push(HealthRisk {
                    risk_type: "Poor Overall Health".to_string(),
                    probability: 0.9,
                    impact: 0.8,
                    description: "Model training is in poor health and may fail".to_string(),
                });
            }

            match current.overfitting_risk {
                OverfittingRisk::High | OverfittingRisk::Severe => {
                    risks.push(HealthRisk {
                        risk_type: "Overfitting".to_string(),
                        probability: 0.8,
                        impact: 0.6,
                        description: "Model is likely overfitting and will generalize poorly"
                            .to_string(),
                    });
                },
                _ => {},
            }

            if current.convergence_probability < 0.3 {
                risks.push(HealthRisk {
                    risk_type: "Training Failure".to_string(),
                    probability: 0.7,
                    impact: 0.9,
                    description: "Training may not converge to a useful solution".to_string(),
                });
            }
        }

        risks
    }

    fn generate_improvement_suggestions(&self) -> Vec<ImprovementSuggestion> {
        let mut suggestions = Vec::new();

        if let Some(current) = self.health_assessments.last() {
            if current.component_scores.stability_health < 0.5 {
                suggestions.push(ImprovementSuggestion {
                    area: "Training Stability".to_string(),
                    suggestion: "Reduce learning rate and increase batch size".to_string(),
                    expected_improvement: 0.3,
                    implementation_effort: "Low".to_string(),
                });
            }

            if current.convergence_probability < 0.5 {
                suggestions.push(ImprovementSuggestion {
                    area: "Convergence".to_string(),
                    suggestion: "Implement learning rate scheduling and gradient clipping"
                        .to_string(),
                    expected_improvement: 0.25,
                    implementation_effort: "Medium".to_string(),
                });
            }

            match current.overfitting_risk {
                OverfittingRisk::Medium | OverfittingRisk::High | OverfittingRisk::Severe => {
                    suggestions.push(ImprovementSuggestion {
                        area: "Overfitting Prevention".to_string(),
                        suggestion: "Add dropout layers, implement early stopping, or increase training data".to_string(),
                        expected_improvement: 0.4,
                        implementation_effort: "Medium".to_string(),
                    });
                },
                _ => {},
            }
        }

        suggestions
    }

    fn compare_with_baseline(&self) -> Option<BaselineComparison> {
        if let (Some(_baseline), Some(current)) =
            (&self.performance_baseline, self.health_assessments.last())
        {
            Some(BaselineComparison {
                health_score_change: current.overall_health_score - 0.8, // Simplified baseline score
                stability_change: current.training_stability_index - 0.7,
                convergence_change: current.convergence_probability - 0.6,
                improvement_percentage: ((current.overall_health_score - 0.8) / 0.8 * 100.0)
                    .max(-100.0),
            })
        } else {
            None
        }
    }

    fn generate_health_summary(&self) -> String {
        if let Some(current) = self.health_assessments.last() {
            match current.health_status {
                HealthStatus::Excellent => "Training is in excellent health with stable convergence and no significant issues detected.".to_string(),
                HealthStatus::Good => "Training is proceeding well with minor optimization opportunities.".to_string(),
                HealthStatus::Fair => "Training shows some concerning patterns that should be addressed.".to_string(),
                HealthStatus::Poor => "Training has significant issues requiring immediate attention.".to_string(),
                HealthStatus::Critical => "Training is in critical condition and may fail without intervention.".to_string(),
            }
        } else {
            "Insufficient data for health assessment.".to_string()
        }
    }
}

// Report structures

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthReport {
    pub current_health: HealthAssessment,
    pub health_trends: HealthTrends,
    pub risk_assessment: Vec<HealthRisk>,
    pub improvement_suggestions: Vec<ImprovementSuggestion>,
    pub baseline_comparison: Option<BaselineComparison>,
    pub summary: String,
}

impl Default for HealthReport {
    fn default() -> Self {
        Self {
            current_health: HealthAssessment {
                timestamp: SystemTime::now(),
                overall_health_score: 0.5,
                training_stability_index: 0.5,
                convergence_probability: 0.5,
                overfitting_risk: OverfittingRisk::None,
                generalization_score: 0.5,
                component_scores: ComponentHealthScores {
                    gradient_health: 0.5,
                    loss_health: 0.5,
                    accuracy_health: 0.5,
                    performance_health: 0.5,
                    memory_health: 0.5,
                    stability_health: 0.5,
                },
                health_status: HealthStatus::Fair,
                alerts: Vec::new(),
                recommendations: Vec::new(),
            },
            health_trends: HealthTrends::default(),
            risk_assessment: Vec::new(),
            improvement_suggestions: Vec::new(),
            baseline_comparison: None,
            summary: "No health data available yet.".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthTrends {
    pub overall_trend: Trend,
    pub stability_trend: Trend,
    pub convergence_trend: Trend,
    pub overfitting_trend: Trend,
}

impl Default for HealthTrends {
    fn default() -> Self {
        Self {
            overall_trend: Trend::Stable,
            stability_trend: Trend::Stable,
            convergence_trend: Trend::Stable,
            overfitting_trend: Trend::Stable,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthRisk {
    pub risk_type: String,
    pub probability: f64,
    pub impact: f64,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementSuggestion {
    pub area: String,
    pub suggestion: String,
    pub expected_improvement: f64,
    pub implementation_effort: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineComparison {
    pub health_score_change: f64,
    pub stability_change: f64,
    pub convergence_change: f64,
    pub improvement_percentage: f64,
}
