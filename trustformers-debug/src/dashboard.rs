//! Interactive dashboards for real-time monitoring and analysis

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};
use uuid::Uuid;

use crate::DebugConfig;

/// Real-time metrics for dashboard display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardMetrics {
    pub timestamp: SystemTime,
    pub loss: Option<f64>,
    pub accuracy: Option<f64>,
    pub learning_rate: Option<f64>,
    pub memory_usage_mb: f64,
    pub gpu_utilization: Option<f64>,
    pub tokens_per_second: Option<f64>,
    pub gradient_norm: Option<f64>,
    pub epoch: Option<u32>,
    pub step: Option<u64>,
}

/// Training monitor for real-time tracking
#[derive(Debug)]
pub struct TrainingMonitor {
    #[allow(dead_code)]
    config: DebugConfig,
    metrics_history: VecDeque<DashboardMetrics>,
    max_history: usize,
    start_time: Instant,
    alert_thresholds: AlertThresholds,
    active_alerts: Vec<TrainingAlert>,
}

/// Alert thresholds for training monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub loss_increase_threshold: f64,
    pub gradient_norm_max: f64,
    pub memory_usage_max_mb: f64,
    pub gpu_utilization_min: f64,
    pub learning_rate_min: f64,
    pub tokens_per_second_min: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            loss_increase_threshold: 1.5,
            gradient_norm_max: 10.0,
            memory_usage_max_mb: 8192.0,
            gpu_utilization_min: 0.7,
            learning_rate_min: 1e-8,
            tokens_per_second_min: 100.0,
        }
    }
}

/// Training alert types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingAlert {
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: SystemTime,
    pub metric_value: f64,
    pub threshold: f64,
    pub suggested_action: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AlertType {
    LossIncrease,
    GradientExplosion,
    MemoryOveruse,
    LowGpuUtilization,
    LearningRateTooLow,
    SlowTokenProcessing,
    ModelDivergence,
    TrainingStalled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

impl TrainingMonitor {
    /// Create a new training monitor
    pub fn new(config: &DebugConfig) -> Self {
        Self {
            config: config.clone(),
            metrics_history: VecDeque::new(),
            max_history: 10000,
            start_time: Instant::now(),
            alert_thresholds: AlertThresholds::default(),
            active_alerts: Vec::new(),
        }
    }

    /// Update metrics and check for alerts
    pub fn update_metrics(&mut self, metrics: DashboardMetrics) {
        // Add to history
        self.metrics_history.push_back(metrics.clone());

        // Trim history if needed
        if self.metrics_history.len() > self.max_history {
            self.metrics_history.pop_front();
        }

        // Check for alerts
        self.check_alerts(&metrics);
    }

    /// Get recent metrics for dashboard
    pub fn get_recent_metrics(&self, count: usize) -> Vec<DashboardMetrics> {
        self.metrics_history.iter().rev().take(count).rev().cloned().collect()
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> &[TrainingAlert] {
        &self.active_alerts
    }

    /// Clear resolved alerts
    pub fn clear_alert(&mut self, _alert_type: AlertType) {
        self.active_alerts.retain(|alert| !matches!(&alert.alert_type, _alert_type));
    }

    /// Set custom alert thresholds
    pub fn set_alert_thresholds(&mut self, thresholds: AlertThresholds) {
        self.alert_thresholds = thresholds;
    }

    /// Generate training summary
    pub fn generate_training_summary(&self) -> TrainingSummary {
        let total_duration = self.start_time.elapsed();
        let total_steps = self.metrics_history.len();

        let avg_loss = self.calculate_average_loss();
        let best_accuracy = self.calculate_best_accuracy();
        let avg_tokens_per_second = self.calculate_average_tokens_per_second();
        let training_stability = self.calculate_training_stability();

        TrainingSummary {
            total_duration,
            total_steps,
            avg_loss,
            best_accuracy,
            avg_tokens_per_second,
            training_stability,
            active_alerts_count: self.active_alerts.len(),
            convergence_status: self.assess_convergence(),
        }
    }

    fn check_alerts(&mut self, metrics: &DashboardMetrics) {
        // Check for loss increase
        if let Some(current_loss) = metrics.loss {
            if let Some(prev_metrics) =
                self.metrics_history.get(self.metrics_history.len().saturating_sub(10))
            {
                if let Some(prev_loss) = prev_metrics.loss {
                    if current_loss > prev_loss * self.alert_thresholds.loss_increase_threshold {
                        self.add_alert(TrainingAlert {
                            alert_type: AlertType::LossIncrease,
                            severity: AlertSeverity::Warning,
                            message: "Loss has increased significantly".to_string(),
                            timestamp: SystemTime::now(),
                            metric_value: current_loss,
                            threshold: prev_loss * self.alert_thresholds.loss_increase_threshold,
                            suggested_action: "Check learning rate or data quality".to_string(),
                        });
                    }
                }
            }
        }

        // Check gradient norm
        if let Some(grad_norm) = metrics.gradient_norm {
            if grad_norm > self.alert_thresholds.gradient_norm_max {
                self.add_alert(TrainingAlert {
                    alert_type: AlertType::GradientExplosion,
                    severity: AlertSeverity::Critical,
                    message: "Gradient explosion detected".to_string(),
                    timestamp: SystemTime::now(),
                    metric_value: grad_norm,
                    threshold: self.alert_thresholds.gradient_norm_max,
                    suggested_action: "Apply gradient clipping or reduce learning rate".to_string(),
                });
            }
        }

        // Check memory usage
        if metrics.memory_usage_mb > self.alert_thresholds.memory_usage_max_mb {
            self.add_alert(TrainingAlert {
                alert_type: AlertType::MemoryOveruse,
                severity: AlertSeverity::Warning,
                message: "High memory usage detected".to_string(),
                timestamp: SystemTime::now(),
                metric_value: metrics.memory_usage_mb,
                threshold: self.alert_thresholds.memory_usage_max_mb,
                suggested_action: "Reduce batch size or enable gradient checkpointing".to_string(),
            });
        }

        // Check GPU utilization
        if let Some(gpu_util) = metrics.gpu_utilization {
            if gpu_util < self.alert_thresholds.gpu_utilization_min {
                self.add_alert(TrainingAlert {
                    alert_type: AlertType::LowGpuUtilization,
                    severity: AlertSeverity::Info,
                    message: "Low GPU utilization".to_string(),
                    timestamp: SystemTime::now(),
                    metric_value: gpu_util,
                    threshold: self.alert_thresholds.gpu_utilization_min,
                    suggested_action: "Increase batch size or check data loading".to_string(),
                });
            }
        }

        // Check tokens per second
        if let Some(tps) = metrics.tokens_per_second {
            if tps < self.alert_thresholds.tokens_per_second_min {
                self.add_alert(TrainingAlert {
                    alert_type: AlertType::SlowTokenProcessing,
                    severity: AlertSeverity::Warning,
                    message: "Slow token processing detected".to_string(),
                    timestamp: SystemTime::now(),
                    metric_value: tps,
                    threshold: self.alert_thresholds.tokens_per_second_min,
                    suggested_action: "Optimize model or increase batch size".to_string(),
                });
            }
        }
    }

    fn add_alert(&mut self, alert: TrainingAlert) {
        // Avoid duplicate alerts of same type
        if !self.active_alerts.iter().any(|a| a.alert_type == alert.alert_type) {
            self.active_alerts.push(alert);
        }
    }

    fn calculate_average_loss(&self) -> Option<f64> {
        let losses: Vec<f64> = self.metrics_history.iter().filter_map(|m| m.loss).collect();

        if losses.is_empty() {
            None
        } else {
            Some(losses.iter().sum::<f64>() / losses.len() as f64)
        }
    }

    fn calculate_best_accuracy(&self) -> Option<f64> {
        self.metrics_history
            .iter()
            .filter_map(|m| m.accuracy)
            .fold(None, |acc, x| match acc {
                None => Some(x),
                Some(y) => Some(x.max(y)),
            })
    }

    fn calculate_average_tokens_per_second(&self) -> Option<f64> {
        let tps_values: Vec<f64> =
            self.metrics_history.iter().filter_map(|m| m.tokens_per_second).collect();

        if tps_values.is_empty() {
            None
        } else {
            Some(tps_values.iter().sum::<f64>() / tps_values.len() as f64)
        }
    }

    fn calculate_training_stability(&self) -> TrainingStability {
        if self.metrics_history.len() < 10 {
            return TrainingStability::Insufficient;
        }

        let recent_losses: Vec<f64> =
            self.metrics_history.iter().rev().take(50).filter_map(|m| m.loss).collect();

        if recent_losses.len() < 10 {
            return TrainingStability::Insufficient;
        }

        // Calculate loss variance
        let mean_loss = recent_losses.iter().sum::<f64>() / recent_losses.len() as f64;
        let variance = recent_losses.iter().map(|&x| (x - mean_loss).powi(2)).sum::<f64>()
            / recent_losses.len() as f64;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = if mean_loss != 0.0 { std_dev / mean_loss } else { 0.0 };

        match coefficient_of_variation {
            cv if cv < 0.1 => TrainingStability::Stable,
            cv if cv < 0.3 => TrainingStability::Moderate,
            _ => TrainingStability::Unstable,
        }
    }

    fn assess_convergence(&self) -> ConvergenceStatus {
        if self.metrics_history.len() < 50 {
            return ConvergenceStatus::TooEarly;
        }

        let recent_losses: Vec<f64> =
            self.metrics_history.iter().rev().take(100).filter_map(|m| m.loss).collect();

        if recent_losses.len() < 50 {
            return ConvergenceStatus::TooEarly;
        }

        // Check if loss is decreasing
        let first_half_avg =
            recent_losses[25..].iter().sum::<f64>() / (recent_losses.len() - 25) as f64;
        let second_half_avg = recent_losses[..25].iter().sum::<f64>() / 25.0;

        if second_half_avg < first_half_avg * 0.95 {
            ConvergenceStatus::Converging
        } else if (second_half_avg - first_half_avg).abs() / first_half_avg < 0.01 {
            ConvergenceStatus::Converged
        } else {
            ConvergenceStatus::Diverging
        }
    }
}

/// Model comparison tool for A/B testing
#[derive(Debug)]
pub struct ModelComparator {
    models: HashMap<String, ModelMetrics>,
    comparison_config: ComparisonConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub model_id: String,
    pub model_name: String,
    pub metrics_history: Vec<DashboardMetrics>,
    pub final_loss: Option<f64>,
    pub final_accuracy: Option<f64>,
    pub training_time: Duration,
    pub parameter_count: usize,
    pub model_size_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonConfig {
    pub primary_metric: String,
    pub comparison_window: usize,
    pub significance_threshold: f64,
}

impl Default for ComparisonConfig {
    fn default() -> Self {
        Self {
            primary_metric: "loss".to_string(),
            comparison_window: 100,
            significance_threshold: 0.05,
        }
    }
}

impl ModelComparator {
    /// Create new model comparator
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            comparison_config: ComparisonConfig::default(),
        }
    }

    /// Add model for comparison
    pub fn add_model(&mut self, model_metrics: ModelMetrics) {
        self.models.insert(model_metrics.model_id.clone(), model_metrics);
    }

    /// Compare models and generate report
    pub fn compare_models(&self) -> ModelComparisonReport {
        let mut comparisons = Vec::new();
        let model_ids: Vec<String> = self.models.keys().cloned().collect();

        for i in 0..model_ids.len() {
            for j in (i + 1)..model_ids.len() {
                let model_a = &self.models[&model_ids[i]];
                let model_b = &self.models[&model_ids[j]];

                let comparison = self.compare_two_models(model_a, model_b);
                comparisons.push(comparison);
            }
        }

        let best_model = self.find_best_model();
        let ranking = self.rank_models();

        ModelComparisonReport {
            comparisons,
            best_model,
            ranking,
            comparison_config: self.comparison_config.clone(),
        }
    }

    fn compare_two_models(
        &self,
        model_a: &ModelMetrics,
        model_b: &ModelMetrics,
    ) -> ModelComparison {
        let performance_diff = self.calculate_performance_difference(model_a, model_b);
        let efficiency_diff = self.calculate_efficiency_difference(model_a, model_b);
        let statistical_significance = self.test_statistical_significance(model_a, model_b);

        ModelComparison {
            model_a_id: model_a.model_id.clone(),
            model_b_id: model_b.model_id.clone(),
            performance_difference: performance_diff,
            efficiency_difference: efficiency_diff,
            statistical_significance,
            recommendation: self.generate_recommendation(model_a, model_b, performance_diff),
        }
    }

    fn calculate_performance_difference(
        &self,
        model_a: &ModelMetrics,
        model_b: &ModelMetrics,
    ) -> f64 {
        match self.comparison_config.primary_metric.as_str() {
            "loss" => {
                if let (Some(loss_a), Some(loss_b)) = (model_a.final_loss, model_b.final_loss) {
                    (loss_b - loss_a) / loss_a // Negative means model_a is better
                } else {
                    0.0
                }
            },
            "accuracy" => {
                if let (Some(acc_a), Some(acc_b)) = (model_a.final_accuracy, model_b.final_accuracy)
                {
                    (acc_b - acc_a) / acc_a // Positive means model_b is better
                } else {
                    0.0
                }
            },
            _ => 0.0,
        }
    }

    fn calculate_efficiency_difference(
        &self,
        model_a: &ModelMetrics,
        model_b: &ModelMetrics,
    ) -> f64 {
        // Compare training time efficiency
        let time_diff =
            model_b.training_time.as_secs_f64() / model_a.training_time.as_secs_f64() - 1.0;

        // Compare model size efficiency
        let size_diff = model_b.model_size_mb / model_a.model_size_mb - 1.0;

        // Combined efficiency score (lower is better)
        (time_diff + size_diff) / 2.0
    }

    fn test_statistical_significance(
        &self,
        _model_a: &ModelMetrics,
        _model_b: &ModelMetrics,
    ) -> bool {
        // Simplified statistical test - in practice would use proper statistical methods
        true // Placeholder
    }

    fn generate_recommendation(
        &self,
        model_a: &ModelMetrics,
        model_b: &ModelMetrics,
        perf_diff: f64,
    ) -> String {
        if perf_diff.abs() < 0.01 {
            "Models perform similarly - choose based on other factors".to_string()
        } else if perf_diff < 0.0 {
            format!(
                "Model {} performs {:.1}% better",
                model_a.model_name,
                perf_diff.abs() * 100.0
            )
        } else {
            format!(
                "Model {} performs {:.1}% better",
                model_b.model_name,
                perf_diff * 100.0
            )
        }
    }

    fn find_best_model(&self) -> Option<String> {
        let mut best_model = None;
        let mut best_score = f64::NEG_INFINITY;

        for model in self.models.values() {
            let score = match self.comparison_config.primary_metric.as_str() {
                "loss" => model.final_loss.map(|l| -l).unwrap_or(f64::NEG_INFINITY),
                "accuracy" => model.final_accuracy.unwrap_or(0.0),
                _ => 0.0,
            };

            if score > best_score {
                best_score = score;
                best_model = Some(model.model_id.clone());
            }
        }

        best_model
    }

    fn rank_models(&self) -> Vec<ModelRanking> {
        let mut rankings: Vec<ModelRanking> = self
            .models
            .values()
            .map(|model| {
                let score = match self.comparison_config.primary_metric.as_str() {
                    "loss" => model.final_loss.map(|l| -l).unwrap_or(f64::NEG_INFINITY),
                    "accuracy" => model.final_accuracy.unwrap_or(0.0),
                    _ => 0.0,
                };

                ModelRanking {
                    model_id: model.model_id.clone(),
                    model_name: model.model_name.clone(),
                    score,
                    rank: 0, // Will be filled below
                }
            })
            .collect();

        rankings.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        for (i, ranking) in rankings.iter_mut().enumerate() {
            ranking.rank = i + 1;
        }

        rankings
    }
}

/// Hyperparameter explorer for optimization guidance
#[derive(Debug)]
#[allow(dead_code)]
pub struct HyperparameterExplorer {
    experiments: HashMap<String, HyperparameterExperiment>,
    #[allow(dead_code)]
    search_space: HyperparameterSearchSpace,
    optimization_history: Vec<OptimizationStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterExperiment {
    pub experiment_id: String,
    pub hyperparameters: HashMap<String, HyperparameterValue>,
    pub results: ExperimentResults,
    pub status: ExperimentStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HyperparameterValue {
    Float(f64),
    Integer(i64),
    String(String),
    Boolean(bool),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResults {
    pub final_loss: Option<f64>,
    pub final_accuracy: Option<f64>,
    pub training_time: Duration,
    pub convergence_epoch: Option<u32>,
    pub best_validation_score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExperimentStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterSearchSpace {
    pub learning_rate: (f64, f64),
    pub batch_size: (i64, i64),
    pub dropout_rate: (f64, f64),
    pub weight_decay: (f64, f64),
    pub num_layers: (i64, i64),
    pub hidden_size: (i64, i64),
}

impl Default for HyperparameterSearchSpace {
    fn default() -> Self {
        Self {
            learning_rate: (1e-5, 1e-1),
            batch_size: (4, 128),
            dropout_rate: (0.0, 0.5),
            weight_decay: (0.0, 1e-2),
            num_layers: (1, 12),
            hidden_size: (64, 2048),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStep {
    pub step: usize,
    pub best_experiment_id: String,
    pub best_score: f64,
    pub exploration_count: usize,
    pub exploitation_count: usize,
}

impl HyperparameterExplorer {
    /// Create new hyperparameter explorer
    pub fn new() -> Self {
        Self {
            experiments: HashMap::new(),
            search_space: HyperparameterSearchSpace::default(),
            optimization_history: Vec::new(),
        }
    }

    /// Add experiment result
    pub fn add_experiment(&mut self, experiment: HyperparameterExperiment) {
        self.experiments.insert(experiment.experiment_id.clone(), experiment);
    }

    /// Get hyperparameter recommendations
    pub fn get_recommendations(&self) -> HyperparameterRecommendations {
        let best_experiments = self.find_best_experiments(5);
        let parameter_importance = self.analyze_parameter_importance();
        let suggested_ranges = self.suggest_search_ranges();
        let next_experiments = self.suggest_next_experiments(3);

        HyperparameterRecommendations {
            best_experiments,
            parameter_importance,
            suggested_ranges,
            next_experiments,
            total_experiments: self.experiments.len(),
        }
    }

    fn find_best_experiments(&self, limit: usize) -> Vec<String> {
        let mut experiments: Vec<_> = self.experiments.values().collect();
        experiments.sort_by(|a, b| {
            let score_a = a.results.final_loss.unwrap_or(f64::INFINITY);
            let score_b = b.results.final_loss.unwrap_or(f64::INFINITY);
            score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
        });

        experiments.iter().take(limit).map(|exp| exp.experiment_id.clone()).collect()
    }

    fn analyze_parameter_importance(&self) -> HashMap<String, f64> {
        // Simplified parameter importance analysis
        let mut importance = HashMap::new();
        importance.insert("learning_rate".to_string(), 0.8);
        importance.insert("batch_size".to_string(), 0.6);
        importance.insert("dropout_rate".to_string(), 0.4);
        importance.insert("weight_decay".to_string(), 0.3);
        importance
    }

    fn suggest_search_ranges(&self) -> HashMap<String, (f64, f64)> {
        // Analyze best experiments to narrow search ranges
        let mut ranges = HashMap::new();
        ranges.insert("learning_rate".to_string(), (1e-4, 1e-2));
        ranges.insert("dropout_rate".to_string(), (0.1, 0.3));
        ranges
    }

    fn suggest_next_experiments(&self, count: usize) -> Vec<HashMap<String, HyperparameterValue>> {
        let mut suggestions = Vec::new();

        for i in 0..count {
            let mut params = HashMap::new();

            // Generate varied parameter combinations based on best results
            params.insert(
                "learning_rate".to_string(),
                HyperparameterValue::Float(0.001 * (1.0 + i as f64 * 0.5)),
            );
            params.insert(
                "batch_size".to_string(),
                HyperparameterValue::Integer(32 * (1 + i as i64)),
            );
            params.insert(
                "dropout_rate".to_string(),
                HyperparameterValue::Float(0.1 + i as f64 * 0.1),
            );

            suggestions.push(params);
        }

        suggestions
    }
}

/// Dashboard aggregator that combines all monitoring tools
#[derive(Debug)]
pub struct InteractiveDashboard {
    #[allow(dead_code)]
    config: DebugConfig,
    training_monitor: TrainingMonitor,
    model_comparator: ModelComparator,
    hyperparameter_explorer: HyperparameterExplorer,
    dashboard_state: DashboardState,
    websocket_server: Option<WebSocketServer>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DashboardState {
    pub active_session_id: Option<Uuid>,
    pub refresh_rate_ms: u64,
    pub auto_alerts: bool,
    pub display_mode: DisplayMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DisplayMode {
    Overview,
    DetailedMetrics,
    ModelComparison,
    HyperparameterOptimization,
    AlertsOnly,
}

/// WebSocket server for real-time dashboard updates
#[derive(Debug)]
#[allow(dead_code)]
pub struct WebSocketServer {
    #[allow(dead_code)]
    port: u16,
    connected_clients: Arc<Mutex<Vec<String>>>,
}

impl InteractiveDashboard {
    /// Create new interactive dashboard
    pub fn new(config: &DebugConfig) -> Self {
        Self {
            config: config.clone(),
            training_monitor: TrainingMonitor::new(config),
            model_comparator: ModelComparator::new(),
            hyperparameter_explorer: HyperparameterExplorer::new(),
            dashboard_state: DashboardState {
                active_session_id: None,
                refresh_rate_ms: 1000,
                auto_alerts: true,
                display_mode: DisplayMode::Overview,
            },
            websocket_server: None,
        }
    }

    /// Start dashboard with WebSocket server
    pub async fn start(&mut self, port: Option<u16>) -> Result<()> {
        let port = port.unwrap_or(8080);

        self.websocket_server = Some(WebSocketServer {
            port,
            connected_clients: Arc::new(Mutex::new(Vec::new())),
        });

        tracing::info!("Interactive dashboard started on port {}", port);
        Ok(())
    }

    /// Update dashboard with new metrics
    pub fn update(&mut self, metrics: DashboardMetrics) {
        self.training_monitor.update_metrics(metrics.clone());

        // Broadcast to connected clients if WebSocket server is running
        if let Some(_ws_server) = &self.websocket_server {
            self.broadcast_update(metrics);
        }
    }

    /// Get current dashboard snapshot
    pub fn get_dashboard_snapshot(&self) -> DashboardSnapshot {
        let training_summary = self.training_monitor.generate_training_summary();
        let recent_metrics = self.training_monitor.get_recent_metrics(100);
        let active_alerts = self.training_monitor.get_active_alerts().to_vec();
        let model_comparison = self.model_comparator.compare_models();
        let hyperparameter_recommendations = self.hyperparameter_explorer.get_recommendations();

        DashboardSnapshot {
            timestamp: SystemTime::now(),
            training_summary,
            recent_metrics,
            active_alerts,
            model_comparison,
            hyperparameter_recommendations,
            dashboard_state: DashboardState {
                active_session_id: self.dashboard_state.active_session_id,
                refresh_rate_ms: self.dashboard_state.refresh_rate_ms,
                auto_alerts: self.dashboard_state.auto_alerts,
                display_mode: self.dashboard_state.display_mode.clone(),
            },
        }
    }

    /// Export dashboard data to file
    pub async fn export_dashboard_data(&self, path: &str) -> Result<()> {
        let snapshot = self.get_dashboard_snapshot();
        let json = serde_json::to_string_pretty(&snapshot)?;
        tokio::fs::write(path, json).await?;
        Ok(())
    }

    fn broadcast_update(&self, _metrics: DashboardMetrics) {
        // In a real implementation, this would send updates to WebSocket clients
        tracing::debug!("Broadcasting dashboard update to connected clients");
    }
}

// Supporting data structures

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSummary {
    pub total_duration: Duration,
    pub total_steps: usize,
    pub avg_loss: Option<f64>,
    pub best_accuracy: Option<f64>,
    pub avg_tokens_per_second: Option<f64>,
    pub training_stability: TrainingStability,
    pub active_alerts_count: usize,
    pub convergence_status: ConvergenceStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainingStability {
    Stable,
    Moderate,
    Unstable,
    Insufficient,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConvergenceStatus {
    TooEarly,
    Converging,
    Converged,
    Diverging,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelComparisonReport {
    pub comparisons: Vec<ModelComparison>,
    pub best_model: Option<String>,
    pub ranking: Vec<ModelRanking>,
    pub comparison_config: ComparisonConfig,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelComparison {
    pub model_a_id: String,
    pub model_b_id: String,
    pub performance_difference: f64,
    pub efficiency_difference: f64,
    pub statistical_significance: bool,
    pub recommendation: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelRanking {
    pub model_id: String,
    pub model_name: String,
    pub score: f64,
    pub rank: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HyperparameterRecommendations {
    pub best_experiments: Vec<String>,
    pub parameter_importance: HashMap<String, f64>,
    pub suggested_ranges: HashMap<String, (f64, f64)>,
    pub next_experiments: Vec<HashMap<String, HyperparameterValue>>,
    pub total_experiments: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DashboardSnapshot {
    pub timestamp: SystemTime,
    pub training_summary: TrainingSummary,
    pub recent_metrics: Vec<DashboardMetrics>,
    pub active_alerts: Vec<TrainingAlert>,
    pub model_comparison: ModelComparisonReport,
    pub hyperparameter_recommendations: HyperparameterRecommendations,
    pub dashboard_state: DashboardState,
}

/// Dashboard report for integration with main debug system
#[derive(Debug, Serialize, Deserialize)]
pub struct DashboardReport {
    pub session_duration: Duration,
    pub total_metrics_recorded: usize,
    pub alerts_triggered: usize,
    pub models_compared: usize,
    pub experiments_tracked: usize,
    pub performance_summary: TrainingSummary,
    pub key_insights: Vec<String>,
    pub recommendations: Vec<String>,
}

impl InteractiveDashboard {
    /// Generate comprehensive dashboard report
    pub async fn generate_report(&self) -> Result<DashboardReport> {
        let training_summary = self.training_monitor.generate_training_summary();
        let total_metrics = self.training_monitor.metrics_history.len();
        let alerts_count = self.training_monitor.active_alerts.len();
        let models_count = self.model_comparator.models.len();
        let experiments_count = self.hyperparameter_explorer.experiments.len();

        let key_insights = self.generate_key_insights();
        let recommendations = self.generate_recommendations();

        Ok(DashboardReport {
            session_duration: training_summary.total_duration,
            total_metrics_recorded: total_metrics,
            alerts_triggered: alerts_count,
            models_compared: models_count,
            experiments_tracked: experiments_count,
            performance_summary: training_summary,
            key_insights,
            recommendations,
        })
    }

    fn generate_key_insights(&self) -> Vec<String> {
        let mut insights = Vec::new();

        // Training stability insights
        match self.training_monitor.generate_training_summary().training_stability {
            TrainingStability::Stable => insights.push("Training is proceeding stably".to_string()),
            TrainingStability::Unstable => insights.push(
                "Training shows high variance - consider adjusting hyperparameters".to_string(),
            ),
            _ => {},
        }

        // Model comparison insights
        if self.model_comparator.models.len() > 1 {
            let comparison = self.model_comparator.compare_models();
            if let Some(best_model) = comparison.best_model {
                insights.push(format!("Best performing model: {}", best_model));
            }
        }

        // Alert insights
        let critical_alerts = self
            .training_monitor
            .active_alerts
            .iter()
            .filter(|alert| matches!(alert.severity, AlertSeverity::Critical))
            .count();

        if critical_alerts > 0 {
            insights.push(format!(
                "{} critical alerts require immediate attention",
                critical_alerts
            ));
        }

        insights
    }

    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Based on active alerts
        for alert in &self.training_monitor.active_alerts {
            if matches!(alert.severity, AlertSeverity::Critical) {
                recommendations.push(alert.suggested_action.clone());
            }
        }

        // Based on hyperparameter exploration
        if self.hyperparameter_explorer.experiments.len() > 5 {
            recommendations.push(
                "Continue hyperparameter optimization with narrowed search ranges".to_string(),
            );
        }

        // Based on model comparison
        if self.model_comparator.models.len() > 1 {
            recommendations
                .push("Focus on the best performing model architecture for production".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("Continue monitoring training progress".to_string());
        }

        recommendations
    }
}
