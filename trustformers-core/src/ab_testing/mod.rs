//! A/B Testing Framework for TrustformeRS
//!
//! This module provides a comprehensive A/B testing framework for comparing
//! different model versions or configurations in production settings.

mod analysis;
mod deployment;
mod experiment;
mod metrics;
mod routing;

pub use analysis::{ConfidenceLevel, StatisticalAnalyzer, TestRecommendation, TestResult};
pub use deployment::{
    DeploymentStrategy, HealthCheck, HealthCheckType, RollbackCondition, RolloutController,
    RolloutStatus,
};
pub use experiment::{Experiment, ExperimentConfig, Variant};
pub use metrics::{MetricCollector, MetricDataPoint, MetricType, MetricValue};
pub use routing::{RoutingStrategy, TrafficSplitter, UserSegment};

use anyhow::Result;
use parking_lot::RwLock;
use std::sync::Arc;

/// Main A/B testing manager
pub struct ABTestManager {
    experiments: Arc<RwLock<Vec<Experiment>>>,
    traffic_splitter: Arc<TrafficSplitter>,
    metric_collector: Arc<MetricCollector>,
    analyzer: Arc<StatisticalAnalyzer>,
    rollout_controller: Arc<RolloutController>,
}

impl Default for ABTestManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ABTestManager {
    /// Create a new A/B test manager
    pub fn new() -> Self {
        Self {
            experiments: Arc::new(RwLock::new(Vec::new())),
            traffic_splitter: Arc::new(TrafficSplitter::new()),
            metric_collector: Arc::new(MetricCollector::new()),
            analyzer: Arc::new(StatisticalAnalyzer::new()),
            rollout_controller: Arc::new(RolloutController::new()),
        }
    }

    /// Create a new experiment
    pub fn create_experiment(&self, config: ExperimentConfig) -> Result<String> {
        let experiment = Experiment::new(config)?;
        let experiment_id = experiment.id().to_string();

        self.experiments.write().push(experiment);
        Ok(experiment_id)
    }

    /// Route a request to appropriate variant
    pub fn route_request(&self, experiment_id: &str, user_id: &str) -> Result<Variant> {
        let experiments = self.experiments.read();
        let experiment_uuid = uuid::Uuid::parse_str(experiment_id)?;
        let experiment = experiments
            .iter()
            .find(|e| *e.id() == experiment_uuid)
            .ok_or_else(|| anyhow::anyhow!("Experiment not found"))?;

        self.traffic_splitter.route(experiment, user_id)
    }

    /// Record a metric for an experiment
    pub fn record_metric(
        &self,
        experiment_id: &str,
        variant: &Variant,
        metric_type: MetricType,
        value: MetricValue,
    ) -> Result<()> {
        self.metric_collector.record(experiment_id, variant, metric_type, value)
    }

    /// Analyze experiment results
    pub fn analyze_experiment(&self, experiment_id: &str) -> Result<TestResult> {
        let metrics = self.metric_collector.get_metrics(experiment_id)?;
        self.analyzer.analyze(metrics)
    }

    /// Get experiment status
    pub fn get_experiment_status(&self, experiment_id: &str) -> Result<ExperimentStatus> {
        let experiments = self.experiments.read();
        let experiment_uuid = uuid::Uuid::parse_str(experiment_id)?;
        let experiment = experiments
            .iter()
            .find(|e| *e.id() == experiment_uuid)
            .ok_or_else(|| anyhow::anyhow!("Experiment not found"))?;

        // Convert experiment::ExperimentStatus to ab_testing::ExperimentStatus
        let status = match experiment.status() {
            crate::ab_testing::experiment::ExperimentStatus::Draft => ExperimentStatus::Draft,
            crate::ab_testing::experiment::ExperimentStatus::Running => ExperimentStatus::Running,
            crate::ab_testing::experiment::ExperimentStatus::Paused => ExperimentStatus::Paused,
            crate::ab_testing::experiment::ExperimentStatus::Concluded => {
                ExperimentStatus::Concluded
            },
            crate::ab_testing::experiment::ExperimentStatus::Cancelled => {
                ExperimentStatus::Cancelled
            },
        };
        Ok(status)
    }

    /// Start an experiment
    pub fn start_experiment(&self, experiment_id: &str) -> Result<()> {
        let mut experiments = self.experiments.write();
        let experiment_uuid = uuid::Uuid::parse_str(experiment_id)?;
        let experiment = experiments
            .iter_mut()
            .find(|e| *e.id() == experiment_uuid)
            .ok_or_else(|| anyhow::anyhow!("Experiment not found"))?;

        experiment.start()
    }

    /// Promote winning variant
    pub fn promote_variant(&self, experiment_id: &str, variant: &Variant) -> Result<()> {
        self.rollout_controller.promote(experiment_id, variant)
    }

    /// Rollback to control variant
    pub fn rollback(&self, experiment_id: &str) -> Result<()> {
        self.rollout_controller.rollback(experiment_id)
    }
}

/// Experiment status
#[derive(Debug, Clone, PartialEq)]
pub enum ExperimentStatus {
    /// Experiment is being configured
    Draft,
    /// Experiment is running
    Running,
    /// Experiment is paused
    Paused,
    /// Experiment has concluded
    Concluded,
    /// Experiment was cancelled
    Cancelled,
}

/// A/B test result summary
#[derive(Debug, Clone)]
pub struct ABTestSummary {
    pub experiment_id: String,
    pub control_metrics: MetricSummary,
    pub treatment_metrics: MetricSummary,
    pub statistical_significance: f64,
    pub confidence_level: ConfidenceLevel,
    pub recommendation: Recommendation,
}

/// Metric summary for a variant
#[derive(Debug, Clone)]
pub struct MetricSummary {
    pub variant: Variant,
    pub sample_size: usize,
    pub mean: f64,
    pub std_dev: f64,
    pub confidence_interval: (f64, f64),
}

/// Recommendation based on test results
#[derive(Debug, Clone, PartialEq)]
pub enum Recommendation {
    /// Keep the control variant
    KeepControl,
    /// Switch to treatment variant
    AdoptTreatment,
    /// Continue testing for more data
    ContinueTesting,
    /// No significant difference
    NoPreference,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_experiment() {
        let manager = ABTestManager::new();
        let config = ExperimentConfig {
            name: "Model v2 Test".to_string(),
            description: "Testing new model architecture".to_string(),
            control_variant: Variant::new("v1", "model-v1"),
            treatment_variants: vec![Variant::new("v2", "model-v2")],
            traffic_percentage: 50.0,
            min_sample_size: 1000,
            max_duration_hours: 168,
        };

        let experiment_id = manager.create_experiment(config).unwrap();
        assert!(!experiment_id.is_empty());

        let status = manager.get_experiment_status(&experiment_id).unwrap();
        assert_eq!(status, ExperimentStatus::Draft);
    }

    #[test]
    fn test_route_request() {
        let manager = ABTestManager::new();
        let config = ExperimentConfig {
            name: "Routing Test".to_string(),
            description: "Test traffic routing".to_string(),
            control_variant: Variant::new("control", "model-v1"),
            treatment_variants: vec![Variant::new("treatment", "model-v2")],
            traffic_percentage: 50.0,
            min_sample_size: 100,
            max_duration_hours: 24,
        };

        let experiment_id = manager.create_experiment(config).unwrap();

        // Start the experiment
        manager.start_experiment(&experiment_id).unwrap();

        // Route multiple requests and verify distribution
        let mut control_count = 0;
        let mut treatment_count = 0;

        for i in 0..1000 {
            let user_id = format!("user-{}", i);
            let variant = manager.route_request(&experiment_id, &user_id).unwrap();

            match variant.name() {
                "control" => control_count += 1,
                "treatment" => treatment_count += 1,
                name => panic!("Unexpected variant name: {}", name),
            }
        }

        // Check that distribution is roughly 50/50
        let ratio = control_count as f64 / (control_count + treatment_count) as f64;
        assert!((ratio - 0.5).abs() < 0.05); // Allow 5% deviation
    }
}
