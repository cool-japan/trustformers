use anyhow::Result;
use scirs2_core::ndarray::Array1; // SciRS2 Integration Policy
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for Elastic Weight Consolidation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EWCConfig {
    /// Regularization strength (lambda)
    pub lambda: f32,
    /// Number of samples for Fisher Information estimation
    pub fisher_samples: usize,
    /// Whether to use online EWC variant
    pub online: bool,
    /// Decay factor for online EWC
    pub decay_factor: f32,
    /// Whether to use diagonal Fisher approximation
    pub diagonal_fisher: bool,
}

impl Default for EWCConfig {
    fn default() -> Self {
        Self {
            lambda: 0.4,
            fisher_samples: 1000,
            online: false,
            decay_factor: 0.95,
            diagonal_fisher: true,
        }
    }
}

/// Fisher Information Matrix for a parameter
#[derive(Debug, Clone)]
pub struct FisherInformation {
    /// Parameter name
    pub name: String,
    /// Fisher Information values (diagonal approximation)
    pub values: Array1<f32>,
    /// Optimal parameter values from previous task
    pub optimal_params: Array1<f32>,
    /// Task ID this Fisher information belongs to
    pub task_id: String,
}

impl FisherInformation {
    pub fn new(name: String, size: usize, task_id: String) -> Self {
        Self {
            name,
            values: Array1::zeros(size),
            optimal_params: Array1::zeros(size),
            task_id,
        }
    }

    /// Update Fisher information with gradient squared
    pub fn update(&mut self, gradient: &Array1<f32>) {
        for (f, &g) in self.values.iter_mut().zip(gradient.iter()) {
            *f += g * g;
        }
    }

    /// Normalize Fisher information by number of samples
    pub fn normalize(&mut self, num_samples: usize) {
        self.values /= num_samples as f32;
    }

    /// Compute EWC penalty for current parameters
    pub fn compute_penalty(&self, current_params: &Array1<f32>) -> f32 {
        let diff = current_params - &self.optimal_params;
        let penalty: f32 = (&self.values * &diff * &diff).sum();
        penalty * 0.5
    }
}

/// EWC trainer implementation
#[derive(Debug)]
pub struct EWCTrainer {
    config: EWCConfig,
    fisher_information: HashMap<String, Vec<FisherInformation>>,
    current_task_id: Option<String>,
    sample_count: usize,
}

impl EWCTrainer {
    pub fn new(config: EWCConfig) -> Self {
        Self {
            config,
            fisher_information: HashMap::new(),
            current_task_id: None,
            sample_count: 0,
        }
    }

    /// Start a new task
    pub fn start_task(&mut self, task_id: String) {
        self.current_task_id = Some(task_id.clone());
        self.sample_count = 0;

        // Initialize Fisher information for this task
        self.fisher_information.entry(task_id).or_default();
    }

    /// Add Fisher information for a parameter
    pub fn add_parameter(&mut self, param_name: String, param_size: usize) -> Result<()> {
        let task_id = self
            .current_task_id
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No current task set"))?;

        let fisher_info = FisherInformation::new(param_name.clone(), param_size, task_id.clone());

        if let Some(task_fishers) = self.fisher_information.get_mut(task_id) {
            task_fishers.push(fisher_info);
        }

        Ok(())
    }

    /// Update Fisher information with gradient
    pub fn update_fisher(&mut self, param_name: &str, gradient: &Array1<f32>) -> Result<()> {
        let task_id = self
            .current_task_id
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No current task set"))?;

        if let Some(task_fishers) = self.fisher_information.get_mut(task_id) {
            if let Some(fisher) = task_fishers.iter_mut().find(|f| f.name == param_name) {
                fisher.update(gradient);
            }
        }

        Ok(())
    }

    /// Finalize Fisher information computation for current task
    pub fn finalize_task(&mut self, optimal_params: HashMap<String, Array1<f32>>) -> Result<()> {
        let task_id = self
            .current_task_id
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No current task set"))?;

        if let Some(task_fishers) = self.fisher_information.get_mut(task_id) {
            for fisher in task_fishers.iter_mut() {
                // Normalize by number of samples
                fisher.normalize(self.config.fisher_samples);

                // Store optimal parameters
                if let Some(params) = optimal_params.get(&fisher.name) {
                    fisher.optimal_params = params.clone();
                }
            }
        }

        self.current_task_id = None;
        self.sample_count = 0;
        Ok(())
    }

    /// Compute EWC penalty for all previous tasks
    pub fn compute_penalty(&self, current_params: &HashMap<String, Array1<f32>>) -> f32 {
        let mut total_penalty = 0.0;

        for (task_id, task_fishers) in &self.fisher_information {
            // Skip current task if it's not finalized
            if Some(task_id) == self.current_task_id.as_ref() {
                continue;
            }

            for fisher in task_fishers {
                if let Some(params) = current_params.get(&fisher.name) {
                    total_penalty += fisher.compute_penalty(params);
                }
            }
        }

        total_penalty * self.config.lambda
    }

    /// Get Fisher information for a specific task and parameter
    pub fn get_fisher_info(&self, task_id: &str, param_name: &str) -> Option<&FisherInformation> {
        self.fisher_information.get(task_id)?.iter().find(|f| f.name == param_name)
    }

    /// Get all Fisher information
    pub fn get_all_fisher_info(&self) -> &HashMap<String, Vec<FisherInformation>> {
        &self.fisher_information
    }

    /// Compute importance-weighted parameters for online EWC
    pub fn compute_online_fisher(
        &mut self,
        new_fisher: &HashMap<String, Array1<f32>>,
        task_id: &str,
    ) -> Result<()> {
        if !self.config.online {
            return Ok(());
        }

        // Implement online EWC update
        if let Some(task_fishers) = self.fisher_information.get_mut(task_id) {
            for fisher in task_fishers.iter_mut() {
                if let Some(new_values) = new_fisher.get(&fisher.name) {
                    // Exponential moving average
                    fisher.values = &fisher.values * self.config.decay_factor
                        + new_values * (1.0 - self.config.decay_factor);
                }
            }
        }

        Ok(())
    }

    /// Clear Fisher information for a specific task
    pub fn clear_task(&mut self, task_id: &str) {
        self.fisher_information.remove(task_id);
    }

    /// Get number of stored tasks
    pub fn num_tasks(&self) -> usize {
        self.fisher_information.len()
    }
}

/// Utility functions for Fisher Information computation
pub mod utils {
    use super::*;

    /// Compute diagonal Fisher Information from gradients
    pub fn compute_diagonal_fisher(gradients: &[Array1<f32>]) -> Array1<f32> {
        let size = gradients.first().map(|g| g.len()).unwrap_or(0);
        let mut fisher = Array1::zeros(size);

        for gradient in gradients {
            for (f, &g) in fisher.iter_mut().zip(gradient.iter()) {
                *f += g * g;
            }
        }

        fisher / gradients.len() as f32
    }

    /// Estimate Fisher Information using empirical samples
    pub fn estimate_fisher_empirical(
        log_probs: &Array1<f32>,
        gradients: &[Array1<f32>],
    ) -> Array1<f32> {
        let size = gradients.first().map(|g| g.len()).unwrap_or(0);
        let mut fisher = Array1::zeros(size);

        for (i, gradient) in gradients.iter().enumerate() {
            let prob = log_probs[i].exp();
            for (f, &g) in fisher.iter_mut().zip(gradient.iter()) {
                *f += prob * g * g;
            }
        }

        fisher
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_fisher_information() {
        let mut fisher = FisherInformation::new("test_param".to_string(), 5, "task1".to_string());

        let gradient = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        fisher.update(&gradient);

        // Check that Fisher values are gradient squared
        for (i, &expected) in [1.0, 4.0, 9.0, 16.0, 25.0].iter().enumerate() {
            assert_abs_diff_eq!(fisher.values[i], expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_ewc_trainer() {
        let config = EWCConfig::default();
        let mut trainer = EWCTrainer::new(config);

        trainer.start_task("task1".to_string());
        trainer.add_parameter("weight1".to_string(), 3).unwrap();

        let gradient = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        trainer.update_fisher("weight1", &gradient).unwrap();

        let optimal_params = {
            let mut params = HashMap::new();
            params.insert("weight1".to_string(), Array1::from_vec(vec![0.5, 1.0, 1.5]));
            params
        };

        trainer.finalize_task(optimal_params).unwrap();

        // Test penalty computation
        let current_params = {
            let mut params = HashMap::new();
            params.insert("weight1".to_string(), Array1::from_vec(vec![1.0, 2.0, 3.0]));
            params
        };

        let penalty = trainer.compute_penalty(&current_params);
        assert!(penalty > 0.0);
    }

    #[test]
    fn test_diagonal_fisher_computation() {
        let gradients = vec![
            Array1::from_vec(vec![1.0, 2.0]),
            Array1::from_vec(vec![2.0, 1.0]),
            Array1::from_vec(vec![1.0, 1.0]),
        ];

        let fisher = utils::compute_diagonal_fisher(&gradients);

        // Expected: [(1^2 + 2^2 + 1^2)/3, (2^2 + 1^2 + 1^2)/3] = [2.0, 2.0]
        assert_abs_diff_eq!(fisher[0], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(fisher[1], 2.0, epsilon = 1e-6);
    }
}
