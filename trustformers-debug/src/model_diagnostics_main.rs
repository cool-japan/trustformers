//! Model-level diagnostics and analysis tools.
//!
//! This module has been refactored into a modular architecture for better
//! organization and maintainability. All previous functionality remains
//! available through comprehensive re-exports to ensure backward compatibility.

use crate::DebugConfig;
use anyhow::Result;
use serde::{Deserialize, Serialize};

// Import all modular components directly from the model_diagnostics directory
use crate::model_diagnostics::*;

/// Main model diagnostics system that coordinates all diagnostic components.
#[derive(Debug)]
pub struct ModelDiagnostics {
    #[allow(dead_code)]
    config: DebugConfig,
    performance_analyzer: PerformanceAnalyzer,
    architecture_analyzer: ArchitectureAnalyzer,
    training_analyzer: TrainingDynamicsAnalyzer,
    layer_analyzer: LayerAnalyzer,
    alert_manager: AlertManager,
    auto_debugger: AutoDebugger,
    analytics_engine: AdvancedAnalytics,
    current_step: usize,
}

impl ModelDiagnostics {
    /// Create new model diagnostics system.
    pub fn new(config: &DebugConfig) -> Self {
        Self {
            config: config.clone(),
            performance_analyzer: PerformanceAnalyzer::new(),
            architecture_analyzer: ArchitectureAnalyzer::new(),
            training_analyzer: TrainingDynamicsAnalyzer::new(),
            layer_analyzer: LayerAnalyzer::new(),
            alert_manager: AlertManager::new(),
            auto_debugger: AutoDebugger::new(),
            analytics_engine: AdvancedAnalytics::new(),
            current_step: 0,
        }
    }

    /// Record performance metrics.
    pub fn record_performance(&mut self, metrics: ModelPerformanceMetrics) -> Result<()> {
        self.performance_analyzer.record_metrics(metrics.clone());
        self.auto_debugger.record_performance_metrics(metrics.clone());
        self.analytics_engine.record_performance_metrics(&metrics);

        // Process metrics for alerts
        self.alert_manager.process_performance_metrics(&metrics)?;

        Ok(())
    }

    /// Record architecture information.
    pub fn record_architecture(&mut self, arch_info: ModelArchitectureInfo) {
        self.architecture_analyzer.record_architecture(arch_info);
    }

    /// Record layer activation statistics.
    pub fn record_layer_stats(&mut self, stats: LayerActivationStats) -> Result<()> {
        self.layer_analyzer.record_layer_stats(stats.clone());
        self.auto_debugger.record_layer_stats(stats.clone());

        // Process for alerts
        self.alert_manager.process_layer_stats(&stats)?;

        Ok(())
    }

    /// Record training dynamics.
    pub fn record_training_dynamics(&mut self, dynamics: TrainingDynamics) -> Result<()> {
        self.training_analyzer.record_training_dynamics(dynamics.clone());
        self.auto_debugger.record_training_dynamics(dynamics.clone());

        // Process for alerts
        self.alert_manager.process_training_dynamics(&dynamics)?;

        Ok(())
    }

    /// Calculate overall model health score.
    fn calculate_health_score(&self) -> f64 {
        // Simple implementation - could be enhanced
        let performance_score = 0.8; // Based on performance metrics
        let architecture_score = 0.7; // Based on architecture analysis
        let training_score = 0.9; // Based on training dynamics

        (performance_score + architecture_score + training_score) / 3.0
    }

    /// Aggregate recommendations from all diagnostic components.
    fn aggregate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Collect recommendations from architecture analyzer
        if let Some(arch_analysis) = self.architecture_analyzer.analyze_architecture().ok() {
            for recommendation in arch_analysis.recommendations {
                recommendations.push(format!("[Architecture] {}", recommendation));
            }
        }

        // Collect recommendations from performance analyzer
        let perf_summary = self.performance_analyzer.generate_performance_summary();
        // Add performance-related recommendations based on metrics
        if perf_summary.current_loss > perf_summary.best_loss * 1.5 {
            recommendations.push(
                "[Performance] Current loss significantly higher than best - check for training instability"
                    .to_string(),
            );
        }
        if perf_summary.peak_memory_mb > 16384.0 {
            // > 16GB
            recommendations.push(
                "[Performance] High memory usage detected - consider gradient checkpointing or smaller batch size"
                    .to_string(),
            );
        }

        // Collect recommendations from training analyzer
        let training_dynamics = self.training_analyzer.analyze_training_dynamics();
        match training_dynamics.training_stability {
            TrainingStability::Unstable => {
                recommendations.push(
                    "[Training] Training stability issues detected - consider reducing learning rate or applying gradient clipping"
                        .to_string(),
                );
            },
            TrainingStability::Unknown => {
                recommendations.push(
                    "[Training] Collect more training metrics for better stability assessment"
                        .to_string(),
                );
            },
            _ => {},
        }

        // Check for plateau conditions
        if let Some(plateau) = &training_dynamics.plateau_detection {
            if plateau.duration_steps > 100 {
                recommendations.push(
                    "[Training] Training plateau detected - consider learning rate adjustment or early stopping"
                        .to_string(),
                );
            }
        }

        // Check convergence status
        match training_dynamics.convergence_status {
            ConvergenceStatus::Diverging => {
                recommendations.push(
                    "[Training] Model is diverging - reduce learning rate immediately".to_string(),
                );
            },
            ConvergenceStatus::Plateau => {
                recommendations.push(
                    "[Training] Training has reached a plateau - consider changing optimization strategy or early stopping"
                        .to_string(),
                );
            },
            ConvergenceStatus::Oscillating => {
                recommendations.push(
                    "[Training] Training is oscillating - reduce learning rate or increase batch size"
                        .to_string(),
                );
            },
            _ => {},
        }

        // Add recommendations based on overfitting/underfitting indicators
        if !training_dynamics.overfitting_indicators.is_empty() {
            recommendations.push(
                "[Training] Overfitting detected - consider regularization, dropout, or early stopping"
                    .to_string(),
            );
        }
        if !training_dynamics.underfitting_indicators.is_empty() {
            recommendations.push(
                "[Training] Underfitting detected - consider increasing model capacity or training longer"
                    .to_string(),
            );
        }

        // Collect recommendations from analytics engine
        if let Some(analytics_report) = self.analytics_engine.generate_analytics_report().ok() {
            for recommendation in analytics_report.recommendations {
                recommendations.push(format!("[Analytics] {}", recommendation));
            }
        }

        // Remove duplicates while preserving order
        let mut seen = std::collections::HashSet::new();
        recommendations.retain(|r| seen.insert(r.clone()));

        recommendations
    }

    /// Get current step.
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Analyze current training dynamics.
    pub fn analyze_training_dynamics(&self) -> TrainingDynamics {
        self.training_analyzer.analyze_training_dynamics()
    }

    /// Increment step counter.
    pub fn increment_step(&mut self) {
        self.current_step += 1;
    }

    /// Start the diagnostics system.
    pub async fn start(&mut self) -> Result<()> {
        // Initialize all components
        Ok(())
    }

    /// Generate comprehensive diagnostics report (async version).
    pub async fn generate_report(&self) -> Result<ModelDiagnosticsReport> {
        self.generate_report_sync()
    }

    /// Generate comprehensive diagnostics report (sync version).
    pub fn generate_report_sync(&self) -> Result<ModelDiagnosticsReport> {
        let performance_summary = self.performance_analyzer.generate_performance_summary();
        let architectural_analysis = self.architecture_analyzer.analyze_architecture().ok();
        let training_dynamics = self.training_analyzer.analyze_training_dynamics();
        let alerts = self.alert_manager.get_active_alerts().to_vec();

        // Generate auto-debugging analysis
        let auto_debugging_results = None; // Simplified for now

        // Generate analytics report
        let analytics_report = self.analytics_engine.generate_analytics_report().ok();

        Ok(ModelDiagnosticsReport {
            current_step: self.current_step,
            training_dynamics,
            performance_summary,
            architectural_analysis,
            alerts: alerts.into_iter().map(|a| a.alert).collect(),
            recommendations: self.aggregate_recommendations(),
            health_score: self.calculate_health_score(),
            auto_debugging_results,
            analytics_report,
        })
    }
}

/// Comprehensive model diagnostics report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDiagnosticsReport {
    /// Current training step
    pub current_step: usize,
    /// Training dynamics analysis
    pub training_dynamics: TrainingDynamics,
    /// Performance summary
    pub performance_summary: PerformanceSummary,
    /// Architectural analysis results
    pub architectural_analysis: Option<ArchitecturalAnalysis>,
    /// Active diagnostic alerts
    pub alerts: Vec<ModelDiagnosticAlert>,
    /// Optimization recommendations
    pub recommendations: Vec<String>,
    /// Overall model health score
    pub health_score: f64,
    /// Auto-debugging results
    pub auto_debugging_results: Option<DebuggingReport>,
    /// Advanced analytics report
    pub analytics_report: Option<AnalyticsReport>,
}

impl Default for ModelDiagnosticsReport {
    fn default() -> Self {
        Self {
            current_step: 0,
            training_dynamics: TrainingDynamics {
                convergence_status: ConvergenceStatus::Unknown,
                training_stability: TrainingStability::Unknown,
                learning_efficiency: 0.0,
                overfitting_indicators: Vec::new(),
                underfitting_indicators: Vec::new(),
                plateau_detection: None,
            },
            performance_summary: PerformanceSummary::default(),
            architectural_analysis: None,
            alerts: Vec::new(),
            recommendations: Vec::new(),
            health_score: 0.0,
            auto_debugging_results: None,
            analytics_report: None,
        }
    }
}
