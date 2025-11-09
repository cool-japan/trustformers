use anyhow::Result;
use scirs2_core::ndarray::{Array1, Array2}; // SciRS2 Integration Policy
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for Progressive Networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveConfig {
    /// Number of layers per task column
    pub layers_per_column: usize,
    /// Hidden dimension for each layer
    pub hidden_dim: usize,
    /// Whether to use lateral connections
    pub use_lateral_connections: bool,
    /// Adapter dimension for lateral connections
    pub adapter_dim: usize,
    /// Learning rate for adapters
    pub adapter_lr: f32,
    /// Freeze previous columns during training
    pub freeze_previous_columns: bool,
    /// Maximum number of task columns
    pub max_columns: usize,
}

impl Default for ProgressiveConfig {
    fn default() -> Self {
        Self {
            layers_per_column: 3,
            hidden_dim: 512,
            use_lateral_connections: true,
            adapter_dim: 64,
            adapter_lr: 0.001,
            freeze_previous_columns: true,
            max_columns: 10,
        }
    }
}

/// Task-specific module in a progressive network
#[derive(Debug, Clone)]
pub struct TaskModule {
    /// Task ID this module belongs to
    pub task_id: String,
    /// Column index in the progressive network
    pub column_index: usize,
    /// Layer weights for this task
    pub layers: Vec<Layer>,
    /// Lateral connections from previous tasks
    pub lateral_connections: HashMap<usize, Vec<LateralAdapter>>,
    /// Whether this module is frozen
    pub frozen: bool,
}

impl TaskModule {
    pub fn new(task_id: String, column_index: usize, config: &ProgressiveConfig) -> Self {
        let mut layers = Vec::new();

        // Create layers for this task column
        for layer_idx in 0..config.layers_per_column {
            let layer = Layer::new(
                format!("{}_{}", task_id, layer_idx),
                config.hidden_dim,
                config.hidden_dim,
            );
            layers.push(layer);
        }

        Self {
            task_id,
            column_index,
            layers,
            lateral_connections: HashMap::new(),
            frozen: false,
        }
    }

    /// Add lateral connection from another task
    pub fn add_lateral_connection(
        &mut self,
        source_column: usize,
        layer_idx: usize,
        config: &ProgressiveConfig,
    ) -> Result<()> {
        if layer_idx >= self.layers.len() {
            return Err(anyhow::anyhow!("Layer index out of bounds"));
        }

        let adapter = LateralAdapter::new(config.hidden_dim, config.adapter_dim, config.hidden_dim);

        self.lateral_connections.entry(source_column).or_default().push(adapter);

        Ok(())
    }

    /// Forward pass through this task module
    pub fn forward(&self, input: &Array1<f32>) -> Result<Array1<f32>> {
        let mut output = input.clone();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Apply main layer transformation
            output = layer.forward(&output)?;

            // Add lateral connections from previous tasks
            for adapters in self.lateral_connections.values() {
                if layer_idx < adapters.len() {
                    // This would typically receive activations from the corresponding layer
                    // in the source column, but for simplicity we'll skip that here
                    // In practice, this would be: output += adapter.forward(&source_activations)
                }
            }

            // Apply activation function (ReLU)
            output.mapv_inplace(|x| x.max(0.0));
        }

        Ok(output)
    }

    /// Freeze this module (prevent parameter updates)
    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    /// Unfreeze this module
    pub fn unfreeze(&mut self) {
        self.frozen = false;
    }

    /// Get number of parameters in this module
    pub fn num_parameters(&self) -> usize {
        let layer_params: usize = self.layers.iter().map(|layer| layer.num_parameters()).sum();

        let adapter_params: usize = self
            .lateral_connections
            .values()
            .flatten()
            .map(|adapter| adapter.num_parameters())
            .sum();

        layer_params + adapter_params
    }
}

/// Neural network layer
#[derive(Debug, Clone)]
pub struct Layer {
    pub name: String,
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl Layer {
    pub fn new(name: String, input_dim: usize, output_dim: usize) -> Self {
        // Initialize with small random weights
        let weights = Array2::zeros((output_dim, input_dim));
        let bias = Array1::zeros(output_dim);

        Self {
            name,
            weights,
            bias,
            input_dim,
            output_dim,
        }
    }

    pub fn forward(&self, input: &Array1<f32>) -> Result<Array1<f32>> {
        if input.len() != self.input_dim {
            return Err(anyhow::anyhow!(
                "Input dimension mismatch: expected {}, got {}",
                self.input_dim,
                input.len()
            ));
        }

        let output = self.weights.dot(input) + &self.bias;
        Ok(output)
    }

    pub fn num_parameters(&self) -> usize {
        self.weights.len() + self.bias.len()
    }
}

/// Lateral adapter for connections between task columns
#[derive(Debug, Clone)]
pub struct LateralAdapter {
    pub down_projection: Array2<f32>,
    pub up_projection: Array2<f32>,
    pub input_dim: usize,
    pub adapter_dim: usize,
    pub output_dim: usize,
}

impl LateralAdapter {
    pub fn new(input_dim: usize, adapter_dim: usize, output_dim: usize) -> Self {
        let down_projection = Array2::zeros((adapter_dim, input_dim));
        let up_projection = Array2::zeros((output_dim, adapter_dim));

        Self {
            down_projection,
            up_projection,
            input_dim,
            adapter_dim,
            output_dim,
        }
    }

    pub fn forward(&self, input: &Array1<f32>) -> Result<Array1<f32>> {
        if input.len() != self.input_dim {
            return Err(anyhow::anyhow!("Input dimension mismatch"));
        }

        // Down-project, apply activation, then up-project
        let hidden = self.down_projection.dot(input);
        let activated = hidden.mapv(|x| x.max(0.0)); // ReLU
        let output = self.up_projection.dot(&activated);

        Ok(output)
    }

    pub fn num_parameters(&self) -> usize {
        self.down_projection.len() + self.up_projection.len()
    }
}

/// Progressive Network architecture
#[derive(Debug)]
pub struct ProgressiveNetwork {
    config: ProgressiveConfig,
    task_modules: HashMap<String, TaskModule>,
    column_order: Vec<String>,
    current_task: Option<String>,
}

impl ProgressiveNetwork {
    pub fn new(config: ProgressiveConfig) -> Self {
        Self {
            config,
            task_modules: HashMap::new(),
            column_order: Vec::new(),
            current_task: None,
        }
    }

    /// Add a new task column to the network
    pub fn add_task(&mut self, task_id: String) -> Result<()> {
        if self.task_modules.contains_key(&task_id) {
            return Err(anyhow::anyhow!("Task {} already exists", task_id));
        }

        if self.column_order.len() >= self.config.max_columns {
            return Err(anyhow::anyhow!("Maximum number of columns reached"));
        }

        let column_index = self.column_order.len();
        let mut task_module = TaskModule::new(task_id.clone(), column_index, &self.config);

        // Add lateral connections from all previous tasks
        if self.config.use_lateral_connections {
            for prev_column in 0..column_index {
                for layer_idx in 0..self.config.layers_per_column {
                    task_module.add_lateral_connection(prev_column, layer_idx, &self.config)?;
                }
            }
        }

        self.task_modules.insert(task_id.clone(), task_module);
        self.column_order.push(task_id.clone());

        // Freeze previous columns if configured
        if self.config.freeze_previous_columns {
            self.freeze_previous_columns(&task_id);
        }

        self.current_task = Some(task_id);
        Ok(())
    }

    /// Set current active task
    pub fn set_current_task(&mut self, task_id: String) -> Result<()> {
        if !self.task_modules.contains_key(&task_id) {
            return Err(anyhow::anyhow!("Task {} not found", task_id));
        }

        self.current_task = Some(task_id);
        Ok(())
    }

    /// Forward pass for the current task
    pub fn forward(&self, input: &Array1<f32>) -> Result<Array1<f32>> {
        let task_id = self
            .current_task
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No current task set"))?;

        let task_module = self
            .task_modules
            .get(task_id)
            .ok_or_else(|| anyhow::anyhow!("Task module not found"))?;

        task_module.forward(input)
    }

    /// Forward pass for a specific task
    pub fn forward_task(&self, task_id: &str, input: &Array1<f32>) -> Result<Array1<f32>> {
        let task_module = self
            .task_modules
            .get(task_id)
            .ok_or_else(|| anyhow::anyhow!("Task module not found: {}", task_id))?;

        task_module.forward(input)
    }

    /// Freeze all columns except the current one
    fn freeze_previous_columns(&mut self, current_task: &str) {
        for (task_id, module) in &mut self.task_modules {
            if task_id != current_task {
                module.freeze();
            }
        }
    }

    /// Get network statistics
    pub fn get_network_stats(&self) -> NetworkStats {
        let total_params: usize =
            self.task_modules.values().map(|module| module.num_parameters()).sum();

        let frozen_modules: usize =
            self.task_modules.values().filter(|module| module.frozen).count();

        NetworkStats {
            num_tasks: self.task_modules.len(),
            total_parameters: total_params,
            frozen_modules,
            current_task: self.current_task.clone(),
            column_order: self.column_order.clone(),
        }
    }

    /// Remove a task from the network
    pub fn remove_task(&mut self, task_id: &str) -> Result<()> {
        if !self.task_modules.contains_key(task_id) {
            return Err(anyhow::anyhow!("Task {} not found", task_id));
        }

        self.task_modules.remove(task_id);
        self.column_order.retain(|id| id != task_id);

        if self.current_task.as_ref() == Some(&task_id.to_string()) {
            self.current_task = None;
        }

        Ok(())
    }

    /// Get task module for a specific task
    pub fn get_task_module(&self, task_id: &str) -> Option<&TaskModule> {
        self.task_modules.get(task_id)
    }

    /// Check if network has capacity for more tasks
    pub fn has_capacity(&self) -> bool {
        self.column_order.len() < self.config.max_columns
    }

    /// Get number of tasks
    pub fn num_tasks(&self) -> usize {
        self.task_modules.len()
    }
}

/// Network statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    pub num_tasks: usize,
    pub total_parameters: usize,
    pub frozen_modules: usize,
    pub current_task: Option<String>,
    pub column_order: Vec<String>,
}

/// Utility functions for progressive networks
pub mod utils {
    use super::*;

    /// Compute lateral connection importance
    pub fn compute_lateral_importance(
        source_activations: &[Array1<f32>],
        target_gradients: &[Array1<f32>],
    ) -> f32 {
        let mut importance = 0.0;

        for (activation, gradient) in source_activations.iter().zip(target_gradients.iter()) {
            importance += (activation * gradient).sum().abs();
        }

        importance / source_activations.len() as f32
    }

    /// Prune weak lateral connections
    pub fn prune_lateral_connections(
        _network: &mut ProgressiveNetwork,
        _importance_threshold: f32,
    ) -> Result<usize> {
        let pruned_count = 0;

        // This would require access to activation and gradient history
        // For now, this is a placeholder implementation

        Ok(pruned_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progressive_network_creation() {
        let config = ProgressiveConfig::default();
        let mut network = ProgressiveNetwork::new(config);

        assert!(network.add_task("task1".to_string()).is_ok());
        assert_eq!(network.num_tasks(), 1);
        assert!(network.has_capacity());
    }

    #[test]
    fn test_task_module_forward() {
        let config = ProgressiveConfig {
            layers_per_column: 2,
            hidden_dim: 4,
            ..Default::default()
        };

        let task_module = TaskModule::new("test_task".to_string(), 0, &config);
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let result = task_module.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), config.hidden_dim);
    }

    #[test]
    fn test_lateral_adapter() {
        let adapter = LateralAdapter::new(4, 2, 4);
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let result = adapter.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_multiple_tasks() {
        let config = ProgressiveConfig {
            max_columns: 3,
            ..Default::default()
        };
        let mut network = ProgressiveNetwork::new(config);

        assert!(network.add_task("task1".to_string()).is_ok());
        assert!(network.add_task("task2".to_string()).is_ok());
        assert!(network.add_task("task3".to_string()).is_ok());

        // Should fail when max columns reached
        assert!(network.add_task("task4".to_string()).is_err());

        let stats = network.get_network_stats();
        assert_eq!(stats.num_tasks, 3);
        assert_eq!(stats.column_order.len(), 3);
    }

    #[test]
    fn test_network_forward() {
        let config = ProgressiveConfig {
            hidden_dim: 4,
            ..Default::default()
        };
        let mut network = ProgressiveNetwork::new(config);

        network.add_task("task1".to_string()).unwrap();
        network.set_current_task("task1".to_string()).unwrap();

        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let result = network.forward(&input);

        assert!(result.is_ok());
    }
}
