// Evaluation framework for transformer models
pub mod benchmarks;
pub mod datasets;
pub mod harness;
pub mod metrics;

pub use benchmarks::*;
pub use datasets::*;
pub use harness::*;
pub use metrics::*;

use anyhow::Result;
use std::collections::HashMap;
// Simplified model trait for evaluation
pub trait EvaluationModel {
    fn forward(&self, input: &str) -> Result<String>;
}

/// Evaluation result for a single task
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    pub task_name: String,
    pub metrics: HashMap<String, f64>,
    pub predictions: Vec<String>,
    pub targets: Vec<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Overall evaluation results for multiple tasks
#[derive(Debug, Clone)]
pub struct EvaluationSuite {
    pub results: Vec<EvaluationResult>,
    pub summary: HashMap<String, f64>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl Default for EvaluationSuite {
    fn default() -> Self {
        Self::new()
    }
}

impl EvaluationSuite {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            summary: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn add_result(&mut self, result: EvaluationResult) {
        self.results.push(result);
        self.update_summary();
    }

    fn update_summary(&mut self) {
        self.summary.clear();

        if self.results.is_empty() {
            return;
        }

        // Compute average scores across all tasks
        let mut metric_sums = HashMap::new();
        let mut metric_counts = HashMap::new();

        for result in &self.results {
            for (metric_name, value) in &result.metrics {
                *metric_sums.entry(metric_name.clone()).or_insert(0.0) += value;
                *metric_counts.entry(metric_name.clone()).or_insert(0) += 1;
            }
        }

        for (metric_name, sum) in metric_sums {
            let count = metric_counts[&metric_name];
            self.summary.insert(format!("avg_{}", metric_name), sum / count as f64);
        }

        // Add task count
        self.summary.insert("num_tasks".to_string(), self.results.len() as f64);
    }

    pub fn get_average_score(&self, metric_name: &str) -> Option<f64> {
        self.summary.get(&format!("avg_{}", metric_name)).copied()
    }

    pub fn print_summary(&self) {
        println!(
            "Evaluation Summary ({})",
            self.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        );
        println!("=================================");
        println!("Total tasks: {}", self.results.len());
        println!();

        for result in &self.results {
            println!("Task: {}", result.task_name);
            for (metric, value) in &result.metrics {
                println!("  {}: {:.4}", metric, value);
            }
            println!();
        }

        println!("Overall Averages:");
        for (metric, value) in &self.summary {
            if metric.starts_with("avg_") {
                println!("  {}: {:.4}", metric, value);
            }
        }
    }
}

/// Configuration for evaluation
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    pub batch_size: usize,
    pub max_length: usize,
    pub num_samples: Option<usize>, // Limit number of samples for quick evaluation
    pub seed: Option<u64>,
    pub output_predictions: bool,
    pub save_results: bool,
    pub output_dir: Option<String>,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            batch_size: 8,
            max_length: 512,
            num_samples: None,
            seed: Some(42),
            output_predictions: false,
            save_results: false,
            output_dir: None,
        }
    }
}

/// Main evaluator trait
pub trait Evaluator {
    fn evaluate(
        &self,
        model: &dyn EvaluationModel,
        config: &EvaluationConfig,
    ) -> Result<EvaluationSuite>;

    fn supported_tasks(&self) -> Vec<String>;

    fn evaluate_single_task(
        &self,
        model: &dyn EvaluationModel,
        task_name: &str,
        config: &EvaluationConfig,
    ) -> Result<EvaluationResult>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluation_result_creation() {
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.85);
        metrics.insert("f1".to_string(), 0.82);

        let result = EvaluationResult {
            task_name: "test_task".to_string(),
            metrics,
            predictions: vec!["pos".to_string(), "neg".to_string()],
            targets: vec!["pos".to_string(), "pos".to_string()],
            metadata: HashMap::new(),
        };

        assert_eq!(result.task_name, "test_task");
        assert_eq!(result.metrics.len(), 2);
        assert_eq!(result.predictions.len(), 2);
        assert_eq!(result.targets.len(), 2);
    }

    #[test]
    fn test_evaluation_suite() {
        let mut suite = EvaluationSuite::new();
        assert_eq!(suite.results.len(), 0);

        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.85);

        let result = EvaluationResult {
            task_name: "task1".to_string(),
            metrics,
            predictions: vec![],
            targets: vec![],
            metadata: HashMap::new(),
        };

        suite.add_result(result);
        assert_eq!(suite.results.len(), 1);
        assert!(suite.get_average_score("accuracy").is_some());
        assert_eq!(suite.get_average_score("accuracy").unwrap(), 0.85);
    }

    #[test]
    fn test_evaluation_config_default() {
        let config = EvaluationConfig::default();
        assert_eq!(config.batch_size, 8);
        assert_eq!(config.max_length, 512);
        assert!(config.num_samples.is_none());
        assert_eq!(config.seed, Some(42));
        assert!(!config.output_predictions);
        assert!(!config.save_results);
        assert!(config.output_dir.is_none());
    }
}
