//! Trial management for hyperparameter optimization

use super::{Direction, ParameterValue};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Unique identifier for a trial
pub type TrialId = uuid::Uuid;

/// State of a hyperparameter optimization trial
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TrialState {
    /// Trial is waiting to be executed
    Pending,
    /// Trial is currently running
    Running,
    /// Trial completed successfully
    Complete,
    /// Trial failed with an error
    Failed,
    /// Trial was pruned (stopped early)
    Pruned,
}

/// Metrics collected during a trial
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialMetrics {
    /// Primary objective value (the metric being optimized)
    pub objective_value: f64,
    /// Additional metrics collected during the trial
    pub metrics: HashMap<String, f64>,
    /// Intermediate values recorded during training (for pruning)
    pub intermediate_values: Vec<(usize, f64)>, // (step, value)
}

impl TrialMetrics {
    /// Create new metrics with an objective value
    pub fn new(objective_value: f64) -> Self {
        Self {
            objective_value,
            metrics: HashMap::new(),
            intermediate_values: Vec::new(),
        }
    }

    /// Add an additional metric
    pub fn add_metric(mut self, name: impl Into<String>, value: f64) -> Self {
        self.metrics.insert(name.into(), value);
        self
    }

    /// Record an intermediate value at a specific step
    pub fn add_intermediate_value(&mut self, step: usize, value: f64) {
        self.intermediate_values.push((step, value));
    }

    /// Get the latest intermediate value
    pub fn latest_intermediate_value(&self) -> Option<f64> {
        self.intermediate_values.last().map(|(_, value)| *value)
    }

    /// Get intermediate value at a specific step
    pub fn intermediate_value_at_step(&self, step: usize) -> Option<f64> {
        self.intermediate_values
            .iter()
            .find(|(s, _)| *s == step)
            .map(|(_, value)| *value)
    }
}

/// Result of a completed trial
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialResult {
    /// Final metrics
    pub metrics: TrialMetrics,
    /// Additional information about the trial
    pub metadata: HashMap<String, String>,
    /// Error message if the trial failed
    pub error_message: Option<String>,
}

impl TrialResult {
    /// Create a successful trial result
    pub fn success(metrics: TrialMetrics) -> Self {
        Self {
            metrics,
            metadata: HashMap::new(),
            error_message: None,
        }
    }

    /// Create a failed trial result
    pub fn failure(error: impl Into<String>) -> Self {
        Self {
            metrics: TrialMetrics::new(f64::NAN),
            metadata: HashMap::new(),
            error_message: Some(error.into()),
        }
    }

    /// Add metadata to the result
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Check if the trial was successful
    pub fn is_success(&self) -> bool {
        self.error_message.is_none() && !self.metrics.objective_value.is_nan()
    }
}

/// Complete trial information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trial {
    /// Unique trial identifier
    pub id: TrialId,
    /// Trial number (sequential)
    pub number: usize,
    /// Current state of the trial
    pub state: TrialState,
    /// Hyperparameter configuration for this trial
    pub params: HashMap<String, ParameterValue>,
    /// Trial result (if completed)
    pub result: Option<TrialResult>,
    /// When the trial was created
    pub created_at: DateTime<Utc>,
    /// When the trial started running
    pub started_at: Option<DateTime<Utc>>,
    /// When the trial completed
    pub completed_at: Option<DateTime<Utc>>,
    /// Duration of the trial
    pub duration: Option<Duration>,
    /// User attributes (additional metadata)
    pub user_attrs: HashMap<String, String>,
    /// System attributes (internal metadata)
    pub system_attrs: HashMap<String, String>,
}

impl Trial {
    /// Create a new pending trial
    pub fn new(number: usize, params: HashMap<String, ParameterValue>) -> Self {
        Self {
            id: TrialId::new_v4(),
            number,
            state: TrialState::Pending,
            params,
            result: None,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            duration: None,
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
        }
    }

    /// Mark the trial as started
    pub fn start(&mut self) {
        self.state = TrialState::Running;
        self.started_at = Some(Utc::now());
    }

    /// Complete the trial with a result
    pub fn complete(&mut self, result: TrialResult) {
        self.state = if result.is_success() { TrialState::Complete } else { TrialState::Failed };
        self.result = Some(result);
        self.completed_at = Some(Utc::now());

        if let Some(started) = self.started_at {
            self.duration = Some(
                Utc::now()
                    .signed_duration_since(started)
                    .to_std()
                    .unwrap_or(Duration::from_secs(0)),
            );
        }
    }

    /// Mark the trial as pruned
    pub fn prune(&mut self, reason: impl Into<String>) {
        self.state = TrialState::Pruned;
        self.completed_at = Some(Utc::now());
        self.system_attrs.insert("prune_reason".to_string(), reason.into());

        if let Some(started) = self.started_at {
            self.duration = Some(
                Utc::now()
                    .signed_duration_since(started)
                    .to_std()
                    .unwrap_or(Duration::from_secs(0)),
            );
        }
    }

    /// Get the objective value (if available)
    pub fn objective_value(&self) -> Option<f64> {
        self.result.as_ref().map(|r| r.metrics.objective_value).filter(|v| !v.is_nan())
    }

    /// Get a parameter value by name
    pub fn get_param(&self, name: &str) -> Option<&ParameterValue> {
        self.params.get(name)
    }

    /// Add a user attribute
    pub fn set_user_attr(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.user_attrs.insert(key.into(), value.into());
    }

    /// Add a system attribute
    pub fn set_system_attr(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.system_attrs.insert(key.into(), value.into());
    }

    /// Get a user attribute
    pub fn get_user_attr(&self, key: &str) -> Option<&str> {
        self.user_attrs.get(key).map(|s| s.as_str())
    }

    /// Get a system attribute
    pub fn get_system_attr(&self, key: &str) -> Option<&str> {
        self.system_attrs.get(key).map(|s| s.as_str())
    }

    /// Check if the trial is finished (complete, failed, or pruned)
    pub fn is_finished(&self) -> bool {
        matches!(
            self.state,
            TrialState::Complete | TrialState::Failed | TrialState::Pruned
        )
    }

    /// Check if the trial was successful
    pub fn is_successful(&self) -> bool {
        self.state == TrialState::Complete && self.result.as_ref().is_some_and(|r| r.is_success())
    }

    /// Get trial summary string
    pub fn summary(&self) -> String {
        let param_str = self
            .params
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(", ");

        let objective_str = self
            .objective_value()
            .map(|v| format!("{:.6}", v))
            .unwrap_or_else(|| "N/A".to_string());

        format!(
            "Trial {}: {} | objective={} | params=[{}]",
            self.number,
            format!("{:?}", self.state).to_lowercase(),
            objective_str,
            param_str
        )
    }
}

/// History of trials in a study
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialHistory {
    /// All trials in chronological order
    pub trials: Vec<Trial>,
    /// Direction of optimization
    pub direction: Direction,
}

impl TrialHistory {
    /// Create a new trial history
    pub fn new(direction: Direction) -> Self {
        Self {
            trials: Vec::new(),
            direction,
        }
    }

    /// Add a trial to the history
    pub fn add_trial(&mut self, trial: Trial) {
        self.trials.push(trial);
    }

    /// Get the best trial so far
    pub fn best_trial(&self) -> Option<&Trial> {
        self.trials.iter().filter(|t| t.is_successful()).max_by(|a, b| {
            let a_val = a.objective_value().unwrap_or(f64::NEG_INFINITY);
            let b_val = b.objective_value().unwrap_or(f64::NEG_INFINITY);

            match self.direction {
                Direction::Maximize => {
                    a_val.partial_cmp(&b_val).unwrap_or(std::cmp::Ordering::Equal)
                },
                Direction::Minimize => {
                    b_val.partial_cmp(&a_val).unwrap_or(std::cmp::Ordering::Equal)
                },
            }
        })
    }

    /// Get the best objective value so far
    pub fn best_value(&self) -> Option<f64> {
        self.best_trial().and_then(|t| t.objective_value())
    }

    /// Get all completed trials
    pub fn completed_trials(&self) -> Vec<&Trial> {
        self.trials.iter().filter(|t| t.is_successful()).collect()
    }

    /// Get all failed trials
    pub fn failed_trials(&self) -> Vec<&Trial> {
        self.trials.iter().filter(|t| t.state == TrialState::Failed).collect()
    }

    /// Get all pruned trials
    pub fn pruned_trials(&self) -> Vec<&Trial> {
        self.trials.iter().filter(|t| t.state == TrialState::Pruned).collect()
    }

    /// Get trial by ID
    pub fn get_trial(&self, id: TrialId) -> Option<&Trial> {
        self.trials.iter().find(|t| t.id == id)
    }

    /// Get trial by number
    pub fn get_trial_by_number(&self, number: usize) -> Option<&Trial> {
        self.trials.iter().find(|t| t.number == number)
    }

    /// Get trials that should be used for optimization decisions
    pub fn optimization_trials(&self) -> Vec<&Trial> {
        self.completed_trials()
    }

    /// Calculate statistics
    pub fn statistics(&self) -> TrialStatistics {
        let total = self.trials.len();
        let completed = self.completed_trials().len();
        let failed = self.failed_trials().len();
        let pruned = self.pruned_trials().len();

        let total_duration = self
            .trials
            .iter()
            .filter_map(|t| t.duration)
            .fold(Duration::from_secs(0), |acc, d| acc + d);

        let avg_duration =
            if total > 0 { total_duration / total as u32 } else { Duration::from_secs(0) };

        TrialStatistics {
            total_trials: total,
            completed_trials: completed,
            failed_trials: failed,
            pruned_trials: pruned,
            best_value: self.best_value(),
            total_duration,
            average_trial_duration: avg_duration,
        }
    }
}

/// Statistics about trial execution
#[derive(Debug, Clone)]
pub struct TrialStatistics {
    /// Total number of trials
    pub total_trials: usize,
    /// Number of successfully completed trials
    pub completed_trials: usize,
    /// Number of failed trials
    pub failed_trials: usize,
    /// Number of pruned trials
    pub pruned_trials: usize,
    /// Best objective value found
    pub best_value: Option<f64>,
    /// Total time spent on all trials
    pub total_duration: Duration,
    /// Average duration per trial
    pub average_trial_duration: Duration,
}

impl TrialStatistics {
    /// Get success rate as a percentage
    pub fn success_rate(&self) -> f64 {
        if self.total_trials == 0 {
            0.0
        } else {
            (self.completed_trials as f64 / self.total_trials as f64) * 100.0
        }
    }

    /// Get pruning rate as a percentage
    pub fn pruning_rate(&self) -> f64 {
        if self.total_trials == 0 {
            0.0
        } else {
            (self.pruned_trials as f64 / self.total_trials as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trial_creation() {
        let mut params = HashMap::new();
        params.insert("lr".to_string(), ParameterValue::Float(0.01));
        params.insert("batch_size".to_string(), ParameterValue::Int(32));

        let trial = Trial::new(1, params.clone());

        assert_eq!(trial.number, 1);
        assert_eq!(trial.state, TrialState::Pending);
        assert_eq!(trial.params.len(), 2);
        assert_eq!(trial.get_param("lr"), Some(&ParameterValue::Float(0.01)));
        assert!(!trial.is_finished());
    }

    #[test]
    fn test_trial_lifecycle() {
        let mut trial = Trial::new(1, HashMap::new());

        // Start the trial
        trial.start();
        assert_eq!(trial.state, TrialState::Running);
        assert!(trial.started_at.is_some());

        // Complete successfully
        let metrics = TrialMetrics::new(0.95).add_metric("accuracy", 0.95);
        let result = TrialResult::success(metrics);
        trial.complete(result);

        assert_eq!(trial.state, TrialState::Complete);
        assert!(trial.completed_at.is_some());
        assert!(trial.duration.is_some());
        assert!(trial.is_finished());
        assert!(trial.is_successful());
        assert_eq!(trial.objective_value(), Some(0.95));
    }

    #[test]
    fn test_trial_pruning() {
        let mut trial = Trial::new(1, HashMap::new());
        trial.start();
        trial.prune("Poor performance");

        assert_eq!(trial.state, TrialState::Pruned);
        assert!(trial.is_finished());
        assert!(!trial.is_successful());
        assert_eq!(
            trial.get_system_attr("prune_reason"),
            Some("Poor performance")
        );
    }

    #[test]
    fn test_trial_metrics() {
        let mut metrics = TrialMetrics::new(0.85);
        metrics.add_intermediate_value(10, 0.6);
        metrics.add_intermediate_value(20, 0.75);
        metrics.add_intermediate_value(30, 0.85);

        assert_eq!(metrics.objective_value, 0.85);
        assert_eq!(metrics.latest_intermediate_value(), Some(0.85));
        assert_eq!(metrics.intermediate_value_at_step(20), Some(0.75));
        assert_eq!(metrics.intermediate_values.len(), 3);
    }

    #[test]
    fn test_trial_history() {
        let mut history = TrialHistory::new(Direction::Maximize);

        // Add trials with different objectives
        let mut trial1 = Trial::new(1, HashMap::new());
        trial1.complete(TrialResult::success(TrialMetrics::new(0.8)));

        let mut trial2 = Trial::new(2, HashMap::new());
        trial2.complete(TrialResult::success(TrialMetrics::new(0.9)));

        let mut trial3 = Trial::new(3, HashMap::new());
        trial3.complete(TrialResult::failure("Error"));

        history.add_trial(trial1);
        history.add_trial(trial2);
        history.add_trial(trial3);

        assert_eq!(history.trials.len(), 3);
        assert_eq!(history.completed_trials().len(), 2);
        assert_eq!(history.failed_trials().len(), 1);
        assert_eq!(history.best_value(), Some(0.9));

        let stats = history.statistics();
        assert_eq!(stats.total_trials, 3);
        assert_eq!(stats.completed_trials, 2);
        assert_eq!(stats.failed_trials, 1);
        assert!((stats.success_rate() - 200.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_trial_history_minimize() {
        let mut history = TrialHistory::new(Direction::Minimize);

        let mut trial1 = Trial::new(1, HashMap::new());
        trial1.complete(TrialResult::success(TrialMetrics::new(0.3)));

        let mut trial2 = Trial::new(2, HashMap::new());
        trial2.complete(TrialResult::success(TrialMetrics::new(0.1)));

        history.add_trial(trial1);
        history.add_trial(trial2);

        // For minimization, 0.1 should be better than 0.3
        assert_eq!(history.best_value(), Some(0.1));
    }
}
