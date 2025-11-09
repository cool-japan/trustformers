//! # Muon Optimizer
//!
//! Implementation of the Muon optimizer, a second-order optimization algorithm designed for
//! neural network training, particularly with hidden layers having 2D weight matrices.
//!
//! Muon is used in the current training speed records for both NanoGPT and CIFAR-10 speedrunning.
//!
//! ## Key Features
//!
//! - **Second-Order Optimization**: Uses Newton-Schulz iteration for efficient orthogonalization
//! - **Low FLOP Overhead**: Below 1% FLOP overhead for typical LM training scenarios
//! - **2D Parameter Focus**: Designed specifically for 2D weight matrices (linear layers)
//! - **Speed Records**: Achieves state-of-the-art training speed on multiple benchmarks
//!
//! ## Design Philosophy
//!
//! Muon only applies to 2D parameters (weight matrices), while scalar and vector parameters
//! must be optimized using a standard method (e.g., AdamW). This hybrid approach provides
//! the best of both worlds: second-order benefits for main parameters and proven stability
//! for auxiliary parameters.

use crate::common::{OptimizerState, StateMemoryStats};
use crate::traits::StatefulOptimizer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::errors::Result;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

/// Configuration for Muon optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MuonConfig {
    /// Learning rate (default: 0.02)
    pub learning_rate: f32,
    /// Momentum coefficient (default: 0.95)
    pub momentum: f32,
    /// Newton-Schulz iteration steps (default: 5)
    pub ns_steps: usize,
    /// Minimum dimension for 2D optimization (default: 64)
    pub min_dim_2d: usize,
    /// Fallback optimizer learning rate for 1D parameters (default: 1e-3)
    pub fallback_lr: f32,
    /// Fallback momentum for 1D parameters (default: 0.9)
    pub fallback_momentum: f32,
    /// Weight decay coefficient (default: 0.0)
    pub weight_decay: f32,
    /// Whether to use orthogonalization (default: true)
    pub use_orthogonal: bool,
}

impl Default for MuonConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.02,
            momentum: 0.95,
            ns_steps: 5,
            min_dim_2d: 64,
            fallback_lr: 1e-3,
            fallback_momentum: 0.9,
            weight_decay: 0.0,
            use_orthogonal: true,
        }
    }
}

/// Muon optimizer implementation
///
/// Muon uses Newton-Schulz iteration for orthogonalization of 2D weight matrices,
/// providing efficient second-order optimization. For 1D parameters, it falls back
/// to a standard momentum-based update.
#[derive(Debug)]
pub struct Muon {
    config: MuonConfig,
    state: OptimizerState,
    /// Momentum buffers for 2D parameters
    momentum_2d: HashMap<String, Vec<Vec<f32>>>,
    /// Momentum buffers for 1D parameters (AdamW-style fallback)
    momentum_1d: HashMap<String, Vec<f32>>,
    /// Parameter shapes for tracking 2D vs 1D
    param_shapes: HashMap<String, (usize, usize)>,
}

impl Muon {
    /// Create a new Muon optimizer with default configuration
    pub fn new() -> Self {
        Self::with_config(MuonConfig::default())
    }

    /// Create Muon with custom learning rate
    pub fn new_with_lr(learning_rate: f32) -> Self {
        let config = MuonConfig {
            learning_rate,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create Muon optimized for NanoGPT training
    pub fn for_nanogpt() -> Self {
        let config = MuonConfig {
            learning_rate: 0.01,
            momentum: 0.95,
            ns_steps: 5,
            min_dim_2d: 32, // Lower threshold for smaller models
            fallback_lr: 5e-4,
            fallback_momentum: 0.9,
            weight_decay: 0.0,
            use_orthogonal: true,
        };
        Self::with_config(config)
    }

    /// Create Muon optimized for CIFAR-10 training
    pub fn for_cifar10() -> Self {
        let config = MuonConfig {
            learning_rate: 0.03,
            momentum: 0.9,
            ns_steps: 4, // Fewer steps for vision tasks
            min_dim_2d: 64,
            fallback_lr: 1e-3,
            fallback_momentum: 0.9,
            weight_decay: 1e-4,
            use_orthogonal: true,
        };
        Self::with_config(config)
    }

    /// Create Muon optimized for large language models
    pub fn for_large_lm() -> Self {
        let config = MuonConfig {
            learning_rate: 0.015,
            momentum: 0.98,  // Higher momentum for large models
            ns_steps: 6,     // More steps for better approximation
            min_dim_2d: 128, // Higher threshold for large models
            fallback_lr: 3e-4,
            fallback_momentum: 0.95,
            weight_decay: 0.01,
            use_orthogonal: true,
        };
        Self::with_config(config)
    }

    /// Create Muon with custom configuration
    pub fn with_config(config: MuonConfig) -> Self {
        Self {
            config,
            state: OptimizerState::new(),
            momentum_2d: HashMap::new(),
            momentum_1d: HashMap::new(),
            param_shapes: HashMap::new(),
        }
    }

    /// Check if parameter should use 2D optimization
    fn should_use_2d_optimization(&self, rows: usize, cols: usize) -> bool {
        rows >= self.config.min_dim_2d && cols >= self.config.min_dim_2d
    }

    /// Newton-Schulz iteration for matrix orthogonalization
    /// Approximates the orthogonal polar factor of a matrix
    fn newton_schulz_orthogonalize(&self, matrix: &mut [Vec<f32>]) {
        if !self.config.use_orthogonal {
            return;
        }

        let rows = matrix.len();
        let cols = matrix[0].len();

        // Newton-Schulz iteration: X_{k+1} = X_k * (3I - X_k^T * X_k) / 2
        for _ in 0..self.config.ns_steps {
            // Compute X^T * X
            let mut xtx = vec![vec![0.0; cols]; cols];
            for i in 0..cols {
                for j in 0..cols {
                    let mut sum = 0.0;
                    for k in 0..rows {
                        sum += matrix[k][i] * matrix[k][j];
                    }
                    xtx[i][j] = sum;
                }
            }

            // Compute 3I - X^T * X
            for i in 0..cols {
                for j in 0..cols {
                    if i == j {
                        xtx[i][j] = 3.0 - xtx[i][j];
                    } else {
                        xtx[i][j] = -xtx[i][j];
                    }
                }
            }

            // Compute X * (3I - X^T * X) / 2
            let mut new_matrix = vec![vec![0.0; cols]; rows];
            for i in 0..rows {
                for j in 0..cols {
                    let mut sum = 0.0;
                    for k in 0..cols {
                        sum += matrix[i][k] * xtx[k][j];
                    }
                    new_matrix[i][j] = sum * 0.5;
                }
            }

            // Update matrix
            for i in 0..rows {
                for j in 0..cols {
                    matrix[i][j] = new_matrix[i][j];
                }
            }
        }
    }

    /// Update 2D parameter using Muon algorithm
    fn update_2d_parameter(
        &mut self,
        param_data: &mut [f32],
        grad_data: &[f32],
        param_id: &str,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        // Initialize momentum if needed
        if !self.momentum_2d.contains_key(param_id) {
            let momentum = vec![vec![0.0; cols]; rows];
            self.momentum_2d.insert(param_id.to_string(), momentum);
        }

        let momentum = self.momentum_2d.get_mut(param_id).unwrap();

        // Reshape flat arrays to 2D views
        let mut param_matrix = vec![vec![0.0; cols]; rows];
        let mut grad_matrix = vec![vec![0.0; cols]; rows];

        // Convert flat to 2D
        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                param_matrix[i][j] = param_data[idx];
                grad_matrix[i][j] = grad_data[idx];
            }
        }

        // Apply weight decay
        if self.config.weight_decay > 0.0 {
            for i in 0..rows {
                for j in 0..cols {
                    grad_matrix[i][j] += self.config.weight_decay * param_matrix[i][j];
                }
            }
        }

        // Update momentum: m = momentum * m + grad
        for i in 0..rows {
            for j in 0..cols {
                momentum[i][j] = self.config.momentum * momentum[i][j] + grad_matrix[i][j];
            }
        }

        // Create update matrix (copy of momentum for orthogonalization)
        let mut update_matrix = momentum.clone();

        // Apply Newton-Schulz orthogonalization
        self.newton_schulz_orthogonalize(&mut update_matrix);

        // Apply update: param = param - lr * orthogonalized_momentum
        for i in 0..rows {
            for j in 0..cols {
                param_matrix[i][j] -= self.config.learning_rate * update_matrix[i][j];

                // Convert back to flat array
                let idx = i * cols + j;
                param_data[idx] = param_matrix[i][j];
            }
        }

        Ok(())
    }

    /// Update 1D parameter using fallback method (momentum SGD)
    fn update_1d_parameter(
        &mut self,
        param_data: &mut [f32],
        grad_data: &[f32],
        param_id: &str,
    ) -> Result<()> {
        let param_size = param_data.len();

        // Initialize momentum if needed
        if !self.momentum_1d.contains_key(param_id) {
            self.momentum_1d.insert(param_id.to_string(), vec![0.0; param_size]);
        }

        let momentum = self.momentum_1d.get_mut(param_id).unwrap();

        // Apply momentum SGD update
        for i in 0..param_size {
            let mut grad = grad_data[i];

            // Apply weight decay
            if self.config.weight_decay > 0.0 {
                grad += self.config.weight_decay * param_data[i];
            }

            // Update momentum
            momentum[i] = self.config.fallback_momentum * momentum[i] + grad;

            // Update parameter
            param_data[i] -= self.config.fallback_lr * momentum[i];
        }

        Ok(())
    }

    /// Get memory statistics for Muon state (deprecated - use memory_usage instead)
    pub fn memory_stats(&self) -> StateMemoryStats {
        self.memory_usage()
    }

    /// Get optimization statistics
    pub fn optimization_stats(&self) -> (usize, usize, f32) {
        let params_2d = self.momentum_2d.len();
        let params_1d = self.momentum_1d.len();
        let total_params = params_2d + params_1d;
        let ratio_2d = if total_params > 0 { params_2d as f32 / total_params as f32 } else { 0.0 };

        (params_2d, params_1d, ratio_2d)
    }
}

impl Default for Muon {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimizer for Muon {
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        let param_data = parameter.data_mut()?;
        let grad_data = grad.data()?;

        // Generate unique parameter ID based on memory address
        let param_id = format!("param_{:p}", param_data.as_ptr());
        let param_size = param_data.len();

        // Determine parameter shape
        let (rows, cols) = if let Some(&shape) = self.param_shapes.get(&param_id) {
            shape
        } else {
            // Try common factorizations for typical NN layers
            let factors = self.find_good_factorization(param_size);
            self.param_shapes.insert(param_id.clone(), factors);
            factors
        };

        // Choose optimization method based on parameter shape
        if self.should_use_2d_optimization(rows, cols) && rows * cols == param_size {
            self.update_2d_parameter(param_data, &grad_data, &param_id, rows, cols)?;
        } else {
            self.update_1d_parameter(param_data, &grad_data, &param_id)?;
        }

        Ok(())
    }

    fn step(&mut self) {
        self.state.step += 1;
    }

    fn zero_grad(&mut self) {
        // This is typically handled by the training framework
        // No action needed here as gradients are managed externally
    }

    fn get_lr(&self) -> f32 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }
}

impl Muon {
    /// Find a good factorization for a given parameter size
    fn find_good_factorization(&self, size: usize) -> (usize, usize) {
        if size < self.config.min_dim_2d {
            return (1, size);
        }

        // Common neural network layer sizes
        let sqrt_size = (size as f32).sqrt() as usize;

        // Try factors close to square root
        for offset in 0..=sqrt_size / 4 {
            let candidate1 = sqrt_size + offset;
            let candidate2 = sqrt_size - offset;

            if candidate1 > 0 && size % candidate1 == 0 {
                let other = size / candidate1;
                if candidate1 >= self.config.min_dim_2d && other >= self.config.min_dim_2d {
                    return (candidate1, other);
                }
            }

            if candidate2 > 0 && size % candidate2 == 0 {
                let other = size / candidate2;
                if candidate2 >= self.config.min_dim_2d && other >= self.config.min_dim_2d {
                    return (candidate2, other);
                }
            }
        }

        // If no good factorization found, treat as 1D
        (1, size)
    }
}

impl StatefulOptimizer for Muon {
    type Config = MuonConfig;
    type State = OptimizerState;

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn state(&self) -> &Self::State {
        &self.state
    }

    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }

    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut state_dict = HashMap::new();

        // Save step count
        state_dict.insert(
            "step".to_string(),
            Tensor::new(vec![self.state.step as f32])?,
        );

        // Save 2D momentum buffers (flattened)
        for (param_id, momentum) in &self.momentum_2d {
            let mut flattened = Vec::new();
            for row in momentum {
                flattened.extend_from_slice(row);
            }
            state_dict.insert(format!("momentum_2d_{}", param_id), Tensor::new(flattened)?);
        }

        // Save 1D momentum buffers
        for (param_id, momentum) in &self.momentum_1d {
            state_dict.insert(
                format!("momentum_1d_{}", param_id),
                Tensor::new(momentum.clone())?,
            );
        }

        // Save parameter shapes
        for (param_id, &(rows, cols)) in &self.param_shapes {
            state_dict.insert(
                format!("shape_{}", param_id),
                Tensor::new(vec![rows as f32, cols as f32])?,
            );
        }

        Ok(state_dict)
    }

    fn load_state_dict(&mut self, state_dict: HashMap<String, Tensor>) -> Result<()> {
        // Load step count
        if let Some(step_tensor) = state_dict.get("step") {
            let step_data = step_tensor.data()?;
            if !step_data.is_empty() {
                self.state.step = step_data[0] as usize;
            }
        }

        // Load parameter shapes first
        for (key, tensor) in &state_dict {
            if let Some(param_id) = key.strip_prefix("shape_") {
                let shape_data = tensor.data()?;
                if shape_data.len() >= 2 {
                    let rows = shape_data[0] as usize;
                    let cols = shape_data[1] as usize;
                    self.param_shapes.insert(param_id.to_string(), (rows, cols));
                }
            }
        }

        // Load momentum buffers
        for (key, tensor) in &state_dict {
            let data = tensor.data()?;
            if let Some(param_id) = key.strip_prefix("momentum_2d_") {
                if let Some(&(rows, cols)) = self.param_shapes.get(param_id) {
                    let mut momentum = vec![vec![0.0; cols]; rows];
                    for i in 0..rows {
                        for j in 0..cols {
                            let idx = i * cols + j;
                            if idx < data.len() {
                                momentum[i][j] = data[idx];
                            }
                        }
                    }
                    self.momentum_2d.insert(param_id.to_string(), momentum);
                }
            } else if let Some(param_id) = key.strip_prefix("momentum_1d_") {
                self.momentum_1d.insert(param_id.to_string(), data);
            }
        }

        Ok(())
    }

    fn memory_usage(&self) -> StateMemoryStats {
        let mut momentum_elements = 0;
        let mut total_elements = 0;

        // Count 2D momentum elements
        for momentum in self.momentum_2d.values() {
            let param_count = momentum.len() * momentum[0].len();
            momentum_elements += param_count;
            total_elements += param_count;
        }

        // Count 1D momentum elements
        for momentum in self.momentum_1d.values() {
            momentum_elements += momentum.len();
            total_elements += momentum.len();
        }

        let total_bytes = total_elements * std::mem::size_of::<f32>();

        StateMemoryStats {
            momentum_elements,
            variance_elements: 0,
            third_moment_elements: 0,
            total_bytes,
            num_parameters: momentum_elements,
        }
    }

    fn reset_state(&mut self) {
        self.state = OptimizerState::new();
        self.momentum_2d.clear();
        self.momentum_1d.clear();
        self.param_shapes.clear();
    }

    fn num_parameters(&self) -> usize {
        let mut total = 0;
        for momentum in self.momentum_2d.values() {
            total += momentum.len() * momentum[0].len();
        }
        for momentum in self.momentum_1d.values() {
            total += momentum.len();
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_muon_creation() {
        let optimizer = Muon::new();
        assert_eq!(optimizer.config.learning_rate, 0.02);
        assert_eq!(optimizer.config.momentum, 0.95);
        assert_eq!(optimizer.config.ns_steps, 5);
        assert_eq!(optimizer.config.min_dim_2d, 64);
        assert_eq!(optimizer.state.step, 0);
    }

    #[test]
    fn test_muon_with_lr() {
        let optimizer = Muon::new_with_lr(0.01);
        assert_eq!(optimizer.config.learning_rate, 0.01);
    }

    #[test]
    fn test_muon_nanogpt_preset() {
        let optimizer = Muon::for_nanogpt();
        assert_eq!(optimizer.config.learning_rate, 0.01);
        assert_eq!(optimizer.config.min_dim_2d, 32);
        assert_eq!(optimizer.config.fallback_lr, 5e-4);
    }

    #[test]
    fn test_muon_cifar10_preset() {
        let optimizer = Muon::for_cifar10();
        assert_eq!(optimizer.config.learning_rate, 0.03);
        assert_eq!(optimizer.config.ns_steps, 4);
        assert_eq!(optimizer.config.weight_decay, 1e-4);
    }

    #[test]
    fn test_muon_large_lm_preset() {
        let optimizer = Muon::for_large_lm();
        assert_eq!(optimizer.config.learning_rate, 0.015);
        assert_eq!(optimizer.config.momentum, 0.98);
        assert_eq!(optimizer.config.min_dim_2d, 128);
    }

    #[test]
    fn test_should_use_2d_optimization() {
        let optimizer = Muon::new();

        // Should use 2D for large matrices
        assert!(optimizer.should_use_2d_optimization(128, 128));
        assert!(optimizer.should_use_2d_optimization(64, 256));

        // Should not use 2D for small matrices
        assert!(!optimizer.should_use_2d_optimization(32, 32));
        assert!(!optimizer.should_use_2d_optimization(64, 32));
        assert!(!optimizer.should_use_2d_optimization(1, 1000));
    }

    #[test]
    fn test_find_good_factorization() {
        let optimizer = Muon::new();

        // Perfect square
        let (rows, cols) = optimizer.find_good_factorization(64 * 64);
        assert_eq!(rows * cols, 64 * 64);
        assert!(rows >= optimizer.config.min_dim_2d);
        assert!(cols >= optimizer.config.min_dim_2d);

        // Small size should be treated as 1D
        let (rows, cols) = optimizer.find_good_factorization(10);
        assert_eq!((rows, cols), (1, 10));

        // Common NN layer size
        let (rows, cols) = optimizer.find_good_factorization(128 * 256);
        assert_eq!(rows * cols, 128 * 256);
    }

    #[test]
    fn test_optimization_stats() {
        let mut optimizer = Muon::new();

        // Initially no parameters
        let (params_2d, params_1d, ratio) = optimizer.optimization_stats();
        assert_eq!(params_2d, 0);
        assert_eq!(params_1d, 0);
        assert_eq!(ratio, 0.0);

        // Add some 2D and 1D parameters
        optimizer.momentum_2d.insert("param_0".to_string(), vec![vec![0.0; 128]; 128]);
        optimizer.momentum_1d.insert("param_1".to_string(), vec![0.0; 10]);
        optimizer.momentum_1d.insert("param_2".to_string(), vec![0.0; 20]);

        let (params_2d, params_1d, ratio) = optimizer.optimization_stats();
        assert_eq!(params_2d, 1);
        assert_eq!(params_1d, 2);
        assert_relative_eq!(ratio, 1.0 / 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_memory_stats() {
        let mut optimizer = Muon::new();

        // Add momentum buffers
        optimizer.momentum_2d.insert("param_0".to_string(), vec![vec![0.0; 100]; 50]); // 5000 params
        optimizer.momentum_1d.insert("param_1".to_string(), vec![0.0; 1000]); // 1000 params

        let stats = optimizer.memory_stats();
        assert_eq!(stats.num_parameters, 6000);
        assert_eq!(stats.momentum_elements, 6000);
        assert_eq!(stats.variance_elements, 0);
        assert_eq!(stats.total_bytes, 6000 * 4); // 4 bytes per f32
    }

    #[test]
    fn test_state_dict_operations() {
        let mut optimizer = Muon::new();
        optimizer.state.step = 5;

        // Add parameter shapes and momentum
        optimizer.param_shapes.insert("param_0".to_string(), (2, 3));
        optimizer.momentum_2d.insert(
            "param_0".to_string(),
            vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]],
        );
        optimizer.momentum_1d.insert("param_1".to_string(), vec![0.7, 0.8]);

        // Save state
        let state_dict = optimizer.state_dict().unwrap();
        assert!(state_dict.contains_key("step"));
        assert!(state_dict.contains_key("momentum_2d_param_0"));
        assert!(state_dict.contains_key("momentum_1d_param_1"));
        assert!(state_dict.contains_key("shape_param_0"));

        // Create new optimizer and load state
        let mut new_optimizer = Muon::new();
        new_optimizer.load_state_dict(state_dict).unwrap();

        assert_eq!(new_optimizer.state.step, 5);
        assert_eq!(new_optimizer.param_shapes["param_0"], (2, 3));
        assert_eq!(new_optimizer.momentum_1d["param_1"], vec![0.7, 0.8]);
    }

    #[test]
    fn test_lr_setter_getter() {
        let mut optimizer = Muon::new();
        assert_eq!(optimizer.get_lr(), 0.02);

        optimizer.set_lr(0.01);
        assert_eq!(optimizer.get_lr(), 0.01);
        assert_eq!(optimizer.config.learning_rate, 0.01);
    }

    #[test]
    fn test_reset() {
        let mut optimizer = Muon::new();
        optimizer.state.step = 10;
        optimizer.momentum_2d.insert("param_0".to_string(), vec![vec![1.0]]);
        optimizer.momentum_1d.insert("param_1".to_string(), vec![1.0]);
        optimizer.param_shapes.insert("param_0".to_string(), (1, 1));

        optimizer.reset_state();

        assert_eq!(optimizer.state.step, 0);
        assert!(optimizer.momentum_2d.is_empty());
        assert!(optimizer.momentum_1d.is_empty());
        assert!(optimizer.param_shapes.is_empty());
    }

    #[test]
    fn test_config_serialization() {
        let config = MuonConfig {
            learning_rate: 0.01,
            momentum: 0.9,
            ns_steps: 3,
            min_dim_2d: 32,
            fallback_lr: 1e-4,
            fallback_momentum: 0.8,
            weight_decay: 1e-5,
            use_orthogonal: false,
        };

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: MuonConfig = serde_json::from_str(&serialized).unwrap();

        assert_relative_eq!(deserialized.learning_rate, config.learning_rate);
        assert_eq!(deserialized.ns_steps, config.ns_steps);
        assert_eq!(deserialized.use_orthogonal, config.use_orthogonal);
    }
}
