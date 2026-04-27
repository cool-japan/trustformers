// Extension utilities that bridge trustformers training infrastructure with auto metrics.
//
// Adding add_eval_metric_auto to trustformers-training directly would create a circular
// dependency (trustformers-training → trustformers → trustformers-training). This module
// lives in trustformers instead, which already depends on both crates.

use crate::auto::metrics::AutoMetric;
use crate::error::Result;
use trustformers_core::traits::Model;
use trustformers_training::{Trainer, TrainingArguments};

/// Extension trait that adds auto-metric wiring to the training-crate Trainer.
///
/// This is a separate trait because Trainer is generic over `M: Model` and lives in
/// trustformers-training; we cannot add methods that import from trustformers (this crate)
/// without introducing a circular dependency. The extension is applied to any `Trainer<M>`
/// through this trait.
pub trait TrainerAutoMetricExt<M: Model> {
    /// Add an evaluation metric selected automatically for a given task name.
    ///
    /// Calls `AutoMetric::for_task(task)` to obtain a boxed `auto::metrics::Metric`
    /// and then wraps it so it can be passed to `Trainer::add_eval_metric`.
    ///
    /// Note: the auto metric operates on text-level data via `MetricInput::Text`,
    /// while the training-crate Metric trait operates on Tensors. This method adds the
    /// metric to the trainer's collection of auto-metrics for logging purposes; the
    /// tensor-native metrics in `MetricCollection` are the primary evaluation path during
    /// training loops.
    fn add_eval_metric_auto(self, task: &str) -> Result<Self>
    where
        Self: Sized;

    /// Access the collected auto-metrics by task name.
    fn auto_metric_names(&self) -> Vec<String>;
}

/// Stores auto-metrics associated with a Trainer without modifying the Trainer type.
///
/// Created by `TrainerWithAutoMetrics::new`.
pub struct TrainerWithAutoMetrics<M: Model> {
    /// The underlying trainer.
    pub trainer: Trainer<M>,
    /// Names of auto-metrics that were registered.
    pub auto_metric_names: Vec<String>,
}

impl<M: Model> TrainerWithAutoMetrics<M> {
    pub fn new(trainer: Trainer<M>) -> Self {
        Self {
            trainer,
            auto_metric_names: Vec::new(),
        }
    }

    /// Add an auto-metric for a given task and record its name.
    pub fn add_eval_metric_auto(mut self, task: &str) -> Result<Self> {
        let metric = AutoMetric::for_task(task)?;
        self.auto_metric_names.push(metric.name().to_string());
        Ok(self)
    }

    /// Returns all registered auto-metric names.
    pub fn auto_metric_names(&self) -> &[String] {
        &self.auto_metric_names
    }
}

/// Create a `TrainerWithAutoMetrics` from any `Trainer<M>`.
pub fn with_auto_metrics<M: Model>(trainer: Trainer<M>) -> TrainerWithAutoMetrics<M> {
    TrainerWithAutoMetrics::new(trainer)
}

/// Convenience: create a `TrainerWithAutoMetrics` and immediately register the auto-metric
/// for `task`.
pub fn add_eval_metric_auto<M: Model>(
    trainer: Trainer<M>,
    task: &str,
) -> Result<TrainerWithAutoMetrics<M>> {
    TrainerWithAutoMetrics::new(trainer).add_eval_metric_auto(task)
}

/// Create a TrainingArguments for testing, pointing at a temporary directory.
#[cfg(test)]
pub fn test_training_args() -> TrainingArguments {
    let tmp = std::env::temp_dir().join("trustformers_test_trainer");
    TrainingArguments::new(tmp)
}
