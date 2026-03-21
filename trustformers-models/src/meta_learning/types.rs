//! Type definitions for meta-learning tasks and results

use std::collections::{HashMap, VecDeque};
use std::time::Duration;
use trustformers_core::tensor::Tensor;

use super::config::MetaAlgorithm;

/// A meta-learning task consisting of support and query sets
#[derive(Debug, Clone)]
pub struct Task {
    pub task_id: String,
    pub support_set: ExampleSet,
    pub query_set: ExampleSet,
    pub task_type: TaskType,
}

/// Batch of tasks for meta-learning
#[derive(Debug, Clone)]
pub struct TaskBatch {
    pub tasks: Vec<Task>,
    pub batch_id: String,
}

/// Set of examples for training or evaluation
#[derive(Debug, Clone)]
pub struct ExampleSet {
    pub examples: Vec<Example>,
    pub num_classes: usize,
}

/// Single training/evaluation example
#[derive(Debug, Clone)]
pub struct Example {
    pub input: Tensor,
    pub label: usize,
    pub metadata: HashMap<String, String>,
}

/// Type of meta-learning task
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    Classification,
    Regression,
    Generation,
    SequenceLabeling,
}

/// Results from a meta-learning episode
#[derive(Debug, Clone)]
pub struct EpisodeResult {
    pub episode: usize,
    pub meta_loss: f64,
    pub meta_accuracy: f64,
    pub num_tasks: usize,
    pub episode_time: Duration,
    pub algorithm: MetaAlgorithm,
}

/// Results from adapting to a single task
#[derive(Debug, Clone)]
pub struct TaskResult {
    pub support_loss: f64,
    pub query_loss: f64,
    pub query_accuracy: f64,
    pub adaptation_time: Duration,
}

/// Evaluation results across multiple tasks
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    pub average_accuracy: f64,
    pub task_results: Vec<TaskResult>,
    pub num_tasks: usize,
}

/// Statistics tracking for meta-learning progress
#[derive(Debug)]
pub struct MetaStatistics {
    pub total_episodes: usize,
    pub average_accuracy: f64,
    pub best_accuracy: f64,
    pub recent_accuracies: VecDeque<f64>,
    pub convergence_rate: f64,
}

impl MetaStatistics {
    pub fn new() -> Self {
        Self {
            total_episodes: 0,
            average_accuracy: 0.0,
            best_accuracy: 0.0,
            recent_accuracies: VecDeque::with_capacity(100),
            convergence_rate: 0.0,
        }
    }

    pub fn update(&mut self, episode_result: &EpisodeResult) {
        self.total_episodes += 1;

        // Update running average
        let alpha = 0.01; // Exponential moving average factor
        self.average_accuracy =
            alpha * episode_result.meta_accuracy + (1.0 - alpha) * self.average_accuracy;

        // Update best accuracy
        if episode_result.meta_accuracy > self.best_accuracy {
            self.best_accuracy = episode_result.meta_accuracy;
        }

        // Track recent accuracies for convergence analysis
        self.recent_accuracies.push_back(episode_result.meta_accuracy);
        if self.recent_accuracies.len() > 100 {
            self.recent_accuracies.pop_front();
        }

        // Estimate convergence rate (simplified)
        if self.recent_accuracies.len() >= 10 {
            let recent_slice: Vec<f64> = self.recent_accuracies.iter().cloned().collect();
            let first_half: f64 = recent_slice[..5].iter().sum::<f64>() / 5.0;
            let second_half: f64 = recent_slice[5..10].iter().sum::<f64>() / 5.0;
            self.convergence_rate = second_half - first_half;
        }
    }

    pub fn is_converged(&self, threshold: f64) -> bool {
        self.convergence_rate.abs() < threshold && self.recent_accuracies.len() >= 50
    }

    pub fn get_recent_average(&self, window: usize) -> f64 {
        let window = window.min(self.recent_accuracies.len());
        if window == 0 {
            return 0.0;
        }

        let recent: Vec<f64> = self.recent_accuracies.iter().rev().take(window).cloned().collect();
        recent.iter().sum::<f64>() / recent.len() as f64
    }
}

impl Default for MetaStatistics {
    fn default() -> Self {
        Self::new()
    }
}