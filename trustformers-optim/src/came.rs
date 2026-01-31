//! # CAME Optimizer
//!
//! Implementation of CAME (Confidence-guided Adaptive Memory Efficient Optimization) from
//! "CAME: Confidence-guided Adaptive Memory Efficient Optimization" (2023).
//!
//! CAME simultaneously achieves two goals: fast convergence as in traditional adaptive methods,
//! and low memory usage as in memory-efficient methods.
//!
//! ## Key Features
//!
//! - **Memory Efficiency**: Significantly reduces memory overhead compared to Adam/LAMB
//! - **Confidence-Guided Strategy**: Uses confidence measures to reduce instability
//! - **Fast Convergence**: Maintains the convergence speed of traditional adaptive methods
//! - **Training Stability**: Improved stability over existing memory efficient optimizers
//!
//! ## Research Results
//!
//! Extensive experiments demonstrate the training stability and superior performance of CAME
//! across various NLP tasks such as BERT and GPT-2 training, achieving memory savings while
//! maintaining or improving convergence speed.

use crate::common::{OptimizerState, StateMemoryStats};
use crate::traits::StatefulOptimizer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::errors::Result;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

/// Configuration for CAME optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CAMEConfig {
    /// Learning rate (default: 1e-3)
    pub learning_rate: f32,
    /// First moment decay rate (default: 0.9)
    pub beta1: f32,
    /// Second moment decay rate (default: 0.999)
    pub beta2: f32,
    /// Small constant for numerical stability (default: 1e-8)
    pub epsilon: f32,
    /// Weight decay coefficient (default: 0.01)
    pub weight_decay: f32,
    /// Confidence threshold for memory efficient mode (default: 0.8)
    pub confidence_threshold: f32,
    /// Memory efficiency factor (default: 0.5)
    pub memory_efficiency: f32,
    /// Whether to use bias correction (default: true)
    pub bias_correction: bool,
    /// Minimum parameter size for factorization (default: 256)
    pub min_factorize_size: usize,
    /// Confidence update rate (default: 0.01)
    pub confidence_update_rate: f32,
}

impl Default for CAMEConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            confidence_threshold: 0.8,
            memory_efficiency: 0.5,
            bias_correction: true,
            min_factorize_size: 256,
            confidence_update_rate: 0.01,
        }
    }
}

/// Confidence state for CAME optimizer
#[derive(Debug, Clone)]
struct ConfidenceState {
    /// Gradient magnitude confidence
    magnitude_confidence: f32,
    /// Direction confidence
    direction_confidence: f32,
    /// Historical variance
    variance_history: f32,
    /// Update count for this parameter
    update_count: usize,
}

impl Default for ConfidenceState {
    fn default() -> Self {
        Self {
            magnitude_confidence: 1.0,
            direction_confidence: 1.0,
            variance_history: 0.0,
            update_count: 0,
        }
    }
}

/// CAME optimizer implementation
///
/// CAME uses a confidence-guided strategy to adaptively choose between full adaptive
/// optimization and memory-efficient approximations based on parameter confidence.
#[derive(Debug)]
pub struct CAME {
    config: CAMEConfig,
    state: OptimizerState,
    /// First moment estimates
    momentum: HashMap<String, Vec<f32>>,
    /// Second moment estimates (full or factorized)
    variance: HashMap<String, Vec<f32>>,
    /// Row factors for factorized second moments
    row_factors: HashMap<String, Vec<f32>>,
    /// Column factors for factorized second moments
    col_factors: HashMap<String, Vec<f32>>,
    /// Confidence states for each parameter
    confidence_states: HashMap<String, ConfidenceState>,
    /// Parameter shapes for factorization decisions
    param_shapes: HashMap<String, (usize, usize)>,
}

impl CAME {
    /// Create a new CAME optimizer with default configuration
    pub fn new() -> Self {
        Self::with_config(CAMEConfig::default())
    }

    /// Create CAME with custom learning rate and weight decay
    pub fn new_with_params(learning_rate: f32, weight_decay: f32) -> Self {
        let config = CAMEConfig {
            learning_rate,
            weight_decay,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create CAME optimized for BERT training
    pub fn for_bert_training() -> Self {
        let config = CAMEConfig {
            learning_rate: 5e-5,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-6,
            weight_decay: 0.01,
            confidence_threshold: 0.85,
            memory_efficiency: 0.6,
            bias_correction: true,
            min_factorize_size: 512,
            confidence_update_rate: 0.005,
        };
        Self::with_config(config)
    }

    /// Create CAME optimized for GPT-2 training
    pub fn for_gpt2_training() -> Self {
        let config = CAMEConfig {
            learning_rate: 6e-4,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1e-8,
            weight_decay: 0.1,
            confidence_threshold: 0.75,
            memory_efficiency: 0.7,
            bias_correction: true,
            min_factorize_size: 1024,
            confidence_update_rate: 0.01,
        };
        Self::with_config(config)
    }

    /// Create CAME for memory-constrained training
    pub fn for_memory_constrained() -> Self {
        let config = CAMEConfig {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            confidence_threshold: 0.9, // Higher threshold for more aggressive memory optimization
            memory_efficiency: 0.8,    // Higher memory efficiency
            bias_correction: true,
            min_factorize_size: 128, // Lower threshold for factorization
            confidence_update_rate: 0.02,
        };
        Self::with_config(config)
    }

    /// Create CAME with custom configuration
    pub fn with_config(config: CAMEConfig) -> Self {
        Self {
            config,
            state: OptimizerState::new(),
            momentum: HashMap::new(),
            variance: HashMap::new(),
            row_factors: HashMap::new(),
            col_factors: HashMap::new(),
            confidence_states: HashMap::new(),
            param_shapes: HashMap::new(),
        }
    }

    /// Update confidence state based on gradient properties
    fn update_confidence(&mut self, param_id: &str, grad_data: &[f32]) {
        let confidence_state = self.confidence_states.entry(param_id.to_string()).or_default();

        confidence_state.update_count += 1;
        let alpha = self.config.confidence_update_rate;

        // Calculate gradient magnitude
        let grad_magnitude = grad_data.iter().map(|g| g * g).sum::<f32>().sqrt();

        // Update magnitude confidence (low if magnitude is very different from history)
        let magnitude_ratio = if confidence_state.variance_history > 0.0 {
            (grad_magnitude / confidence_state.variance_history.sqrt()).min(1.0)
        } else {
            1.0
        };

        confidence_state.magnitude_confidence =
            (1.0 - alpha) * confidence_state.magnitude_confidence + alpha * magnitude_ratio;

        // Update variance history
        confidence_state.variance_history = (1.0 - alpha) * confidence_state.variance_history
            + alpha * grad_magnitude * grad_magnitude;

        // Direction confidence based on gradient variance
        let grad_mean = grad_data.iter().sum::<f32>() / grad_data.len() as f32;
        let grad_var = grad_data.iter().map(|g| (g - grad_mean) * (g - grad_mean)).sum::<f32>()
            / grad_data.len() as f32;

        let normalized_var = if grad_magnitude > 0.0 {
            (grad_var / (grad_magnitude * grad_magnitude)).min(1.0)
        } else {
            1.0
        };

        confidence_state.direction_confidence =
            (1.0 - alpha) * confidence_state.direction_confidence + alpha * (1.0 - normalized_var);
    }

    /// Get overall confidence for a parameter
    fn get_parameter_confidence(&self, param_id: &str) -> f32 {
        if let Some(confidence_state) = self.confidence_states.get(param_id) {
            // Combine magnitude and direction confidence
            (confidence_state.magnitude_confidence * confidence_state.direction_confidence).min(1.0)
        } else {
            1.0 // Full confidence for new parameters
        }
    }

    /// Determine if parameter should use factorized second moment
    fn should_factorize(
        &self,
        param_id: &str,
        param_size: usize,
        rows: usize,
        cols: usize,
    ) -> bool {
        let confidence = self.get_parameter_confidence(param_id);
        let is_large_enough = param_size >= self.config.min_factorize_size;
        let is_2d = rows > 1 && cols > 1;
        let below_confidence_threshold = confidence < self.config.confidence_threshold;

        is_large_enough && is_2d && below_confidence_threshold
    }

    /// Initialize parameter state
    fn init_param_state(&mut self, param_id: &str, param_size: usize) {
        if self.momentum.contains_key(param_id) {
            return;
        }

        // Try to determine parameter shape (simplified approach)
        let (rows, cols) = self.infer_param_shape(param_size);
        self.param_shapes.insert(param_id.to_string(), (rows, cols));

        // Initialize momentum
        self.momentum.insert(param_id.to_string(), vec![0.0; param_size]);

        // Initialize second moment (factorized or full)
        if self.should_factorize(param_id, param_size, rows, cols) {
            // Factorized initialization
            self.row_factors.insert(param_id.to_string(), vec![0.0; rows]);
            self.col_factors.insert(param_id.to_string(), vec![0.0; cols]);
        } else {
            // Full second moment
            self.variance.insert(param_id.to_string(), vec![0.0; param_size]);
        }
    }

    /// Infer parameter shape from size (heuristic)
    fn infer_param_shape(&self, size: usize) -> (usize, usize) {
        if size < self.config.min_factorize_size {
            return (1, size);
        }

        // Try to find a good rectangular factorization
        let sqrt_size = (size as f32).sqrt() as usize;

        for candidate in (sqrt_size.saturating_sub(sqrt_size / 4)..=sqrt_size + sqrt_size / 4).rev()
        {
            if candidate > 0 && size % candidate == 0 {
                let other = size / candidate;
                if candidate >= 8 && other >= 8 {
                    // Minimum reasonable dimensions
                    return (candidate, other);
                }
            }
        }

        (1, size) // Fall back to 1D
    }

    /// Get memory statistics (deprecated - use memory_usage instead)
    pub fn memory_stats(&self) -> StateMemoryStats {
        self.memory_usage()
    }

    /// Calculate memory savings compared to Adam
    pub fn memory_savings_ratio(&self) -> f32 {
        let mut adam_memory = 0;
        let mut came_memory = 0;

        for (param_id, momentum) in &self.momentum {
            let param_size = momentum.len();
            adam_memory += param_size * 2; // momentum + variance for Adam

            // CAME memory
            came_memory += param_size; // momentum

            if self.variance.contains_key(param_id) {
                came_memory += param_size; // full variance
            } else if let (Some(row_factors), Some(col_factors)) = (
                self.row_factors.get(param_id),
                self.col_factors.get(param_id),
            ) {
                came_memory += row_factors.len() + col_factors.len(); // factorized variance
            }
        }

        if adam_memory > 0 {
            1.0 - (came_memory as f32 / adam_memory as f32)
        } else {
            0.0
        }
    }

    /// Get confidence statistics
    pub fn confidence_stats(&self) -> (f32, f32, usize, usize) {
        let total_params = self.confidence_states.len();
        let high_confidence_count = self
            .confidence_states
            .values()
            .filter(|cs| {
                let conf = cs.magnitude_confidence * cs.direction_confidence;
                conf >= self.config.confidence_threshold
            })
            .count();

        let avg_magnitude_conf = if total_params > 0 {
            self.confidence_states.values().map(|cs| cs.magnitude_confidence).sum::<f32>()
                / total_params as f32
        } else {
            0.0
        };

        let avg_direction_conf = if total_params > 0 {
            self.confidence_states.values().map(|cs| cs.direction_confidence).sum::<f32>()
                / total_params as f32
        } else {
            0.0
        };

        (
            avg_magnitude_conf,
            avg_direction_conf,
            high_confidence_count,
            total_params,
        )
    }
}

impl Default for CAME {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimizer for CAME {
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        let param_data = parameter.data_mut()?;
        let grad_data = grad.data()?;

        // Generate unique parameter ID based on memory address
        let param_id = format!("param_{:p}", param_data.as_ptr());
        let param_size = param_data.len();

        // Initialize parameter state if needed
        self.init_param_state(&param_id, param_size);

        // Update confidence
        self.update_confidence(&param_id, &grad_data);

        // Get parameter shape
        let (rows, cols) = self.param_shapes[&param_id];

        // Bias correction factors
        let bias_correction1 = if self.config.bias_correction {
            1.0 - self.config.beta1.powi(self.state.step as i32 + 1)
        } else {
            1.0
        };

        let bias_correction2 = if self.config.bias_correction {
            1.0 - self.config.beta2.powi(self.state.step as i32 + 1)
        } else {
            1.0
        };

        // Apply weight decay
        if self.config.weight_decay > 0.0 {
            for i in 0..param_size {
                param_data[i] *= 1.0 - self.config.learning_rate * self.config.weight_decay;
            }
        }

        // Check if should factorize first
        let should_factorize = self.should_factorize(&param_id, param_size, rows, cols);

        // Update first moment
        let momentum = self
            .momentum
            .get_mut(&param_id)
            .expect("momentum must exist after initialization");
        for i in 0..param_size {
            momentum[i] =
                self.config.beta1 * momentum[i] + (1.0 - self.config.beta1) * grad_data[i];
        }

        // Update second moment (factorized or full)
        if should_factorize {
            // Factorized update
            let row_factors = self
                .row_factors
                .get_mut(&param_id)
                .expect("row_factors must exist after initialization");
            let col_factors = self
                .col_factors
                .get_mut(&param_id)
                .expect("col_factors must exist after initialization");

            // Update row factors
            for i in 0..rows {
                let mut row_grad_sq = 0.0;
                for j in 0..cols {
                    let idx = i * cols + j;
                    row_grad_sq += grad_data[idx] * grad_data[idx];
                }
                row_grad_sq /= cols as f32;
                row_factors[i] =
                    self.config.beta2 * row_factors[i] + (1.0 - self.config.beta2) * row_grad_sq;
            }

            // Update column factors
            for j in 0..cols {
                let mut col_grad_sq = 0.0;
                for i in 0..rows {
                    let idx = i * cols + j;
                    col_grad_sq += grad_data[idx] * grad_data[idx];
                }
                col_grad_sq /= rows as f32;
                col_factors[j] =
                    self.config.beta2 * col_factors[j] + (1.0 - self.config.beta2) * col_grad_sq;
            }

            // Parameter update with factorized second moment
            for i in 0..rows {
                for j in 0..cols {
                    let idx = i * cols + j;
                    let corrected_momentum = momentum[idx] / bias_correction1;

                    // Approximate second moment from factors
                    let approx_variance = (row_factors[i] * col_factors[j]).sqrt();
                    let corrected_variance = approx_variance / bias_correction2.sqrt();

                    let denom = corrected_variance + self.config.epsilon;
                    param_data[idx] -= self.config.learning_rate * corrected_momentum / denom;
                }
            }
        } else {
            // Full second moment update
            let variance = self
                .variance
                .get_mut(&param_id)
                .expect("variance must exist after initialization");
            for i in 0..param_size {
                variance[i] = self.config.beta2 * variance[i]
                    + (1.0 - self.config.beta2) * grad_data[i] * grad_data[i];

                let corrected_momentum = momentum[i] / bias_correction1;
                let corrected_variance = variance[i] / bias_correction2;

                let denom = corrected_variance.sqrt() + self.config.epsilon;
                param_data[i] -= self.config.learning_rate * corrected_momentum / denom;
            }
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

impl StatefulOptimizer for CAME {
    type Config = CAMEConfig;
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

        // Save momentum buffers
        for (param_id, momentum) in &self.momentum {
            state_dict.insert(
                format!("momentum_{}", param_id),
                Tensor::new(momentum.clone())?,
            );
        }

        // Save variance buffers
        for (param_id, variance) in &self.variance {
            state_dict.insert(
                format!("variance_{}", param_id),
                Tensor::new(variance.clone())?,
            );
        }

        // Save factorized buffers
        for (param_id, row_factors) in &self.row_factors {
            state_dict.insert(
                format!("row_factors_{}", param_id),
                Tensor::new(row_factors.clone())?,
            );
        }

        for (param_id, col_factors) in &self.col_factors {
            state_dict.insert(
                format!("col_factors_{}", param_id),
                Tensor::new(col_factors.clone())?,
            );
        }

        // Save parameter shapes
        for (param_id, &(rows, cols)) in &self.param_shapes {
            state_dict.insert(
                format!("shape_{}", param_id),
                Tensor::new(vec![rows as f32, cols as f32])?,
            );
        }

        // Save confidence states
        for (param_id, confidence_state) in &self.confidence_states {
            state_dict.insert(
                format!("confidence_{}", param_id),
                Tensor::new(vec![
                    confidence_state.magnitude_confidence,
                    confidence_state.direction_confidence,
                    confidence_state.variance_history,
                    confidence_state.update_count as f32,
                ])?,
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

        // Load all other states
        for (key, tensor) in &state_dict {
            let data = tensor.data()?;
            if let Some(param_id) = key.strip_prefix("momentum_") {
                self.momentum.insert(param_id.to_string(), data);
            } else if let Some(param_id) = key.strip_prefix("variance_") {
                self.variance.insert(param_id.to_string(), data);
            } else if let Some(param_id) = key.strip_prefix("row_factors_") {
                self.row_factors.insert(param_id.to_string(), data);
            } else if let Some(param_id) = key.strip_prefix("col_factors_") {
                self.col_factors.insert(param_id.to_string(), data);
            } else if let Some(param_id) = key.strip_prefix("confidence_") {
                if data.len() >= 4 {
                    let confidence_state = ConfidenceState {
                        magnitude_confidence: data[0],
                        direction_confidence: data[1],
                        variance_history: data[2],
                        update_count: data[3] as usize,
                    };
                    self.confidence_states.insert(param_id.to_string(), confidence_state);
                }
            }
        }

        Ok(())
    }

    fn memory_usage(&self) -> StateMemoryStats {
        let mut momentum_elements = 0;
        let mut variance_elements = 0;
        let mut total_elements = 0;

        // Count momentum elements
        for momentum in self.momentum.values() {
            momentum_elements += momentum.len();
            total_elements += momentum.len();
        }

        // Count full variance elements
        for variance in self.variance.values() {
            variance_elements += variance.len();
            total_elements += variance.len();
        }

        // Count factorized variance elements
        for row_factors in self.row_factors.values() {
            total_elements += row_factors.len();
        }

        for col_factors in self.col_factors.values() {
            total_elements += col_factors.len();
        }

        let total_bytes = total_elements * std::mem::size_of::<f32>();

        StateMemoryStats {
            momentum_elements,
            variance_elements,
            third_moment_elements: 0,
            total_bytes,
            num_parameters: momentum_elements,
        }
    }

    fn reset_state(&mut self) {
        self.state = OptimizerState::new();
        self.momentum.clear();
        self.variance.clear();
        self.row_factors.clear();
        self.col_factors.clear();
        self.confidence_states.clear();
        self.param_shapes.clear();
    }

    fn num_parameters(&self) -> usize {
        self.momentum.values().map(|v| v.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_came_creation() {
        let optimizer = CAME::new();
        assert_eq!(optimizer.config.learning_rate, 1e-3);
        assert_eq!(optimizer.config.beta1, 0.9);
        assert_eq!(optimizer.config.beta2, 0.999);
        assert_eq!(optimizer.config.confidence_threshold, 0.8);
        assert_eq!(optimizer.state.step, 0);
    }

    #[test]
    fn test_came_with_params() {
        let optimizer = CAME::new_with_params(1e-4, 0.1);
        assert_eq!(optimizer.config.learning_rate, 1e-4);
        assert_eq!(optimizer.config.weight_decay, 0.1);
    }

    #[test]
    fn test_came_bert_preset() {
        let optimizer = CAME::for_bert_training();
        assert_eq!(optimizer.config.learning_rate, 5e-5);
        assert_eq!(optimizer.config.confidence_threshold, 0.85);
        assert_eq!(optimizer.config.min_factorize_size, 512);
    }

    #[test]
    fn test_came_gpt2_preset() {
        let optimizer = CAME::for_gpt2_training();
        assert_eq!(optimizer.config.learning_rate, 6e-4);
        assert_eq!(optimizer.config.beta2, 0.95);
        assert_eq!(optimizer.config.memory_efficiency, 0.7);
    }

    #[test]
    fn test_came_memory_constrained_preset() {
        let optimizer = CAME::for_memory_constrained();
        assert_eq!(optimizer.config.confidence_threshold, 0.9);
        assert_eq!(optimizer.config.memory_efficiency, 0.8);
        assert_eq!(optimizer.config.min_factorize_size, 128);
    }

    #[test]
    fn test_infer_param_shape() {
        let optimizer = CAME::new();

        // Perfect square
        let (rows, cols) = optimizer.infer_param_shape(64 * 64);
        assert_eq!(rows * cols, 64 * 64);
        assert!(rows >= 8 && cols >= 8);

        // Small size
        let (rows, cols) = optimizer.infer_param_shape(10);
        assert_eq!((rows, cols), (1, 10));

        // Rectangular
        let (rows, cols) = optimizer.infer_param_shape(128 * 256);
        assert_eq!(rows * cols, 128 * 256);
    }

    #[test]
    fn test_should_factorize() {
        let mut optimizer = CAME::new();
        let param_id = "test_param";

        // Large parameter with low confidence should factorize
        optimizer.confidence_states.insert(
            param_id.to_string(),
            ConfidenceState {
                magnitude_confidence: 0.5,
                direction_confidence: 0.5,
                ..Default::default()
            },
        );

        assert!(optimizer.should_factorize(param_id, 1000, 20, 50));

        // High confidence should not factorize
        optimizer.confidence_states.get_mut(param_id).unwrap().magnitude_confidence = 0.9;
        optimizer.confidence_states.get_mut(param_id).unwrap().direction_confidence = 0.9;

        assert!(!optimizer.should_factorize(param_id, 1000, 20, 50));

        // Small parameter should not factorize
        assert!(!optimizer.should_factorize(param_id, 100, 10, 10));
    }

    #[test]
    fn test_memory_stats() {
        let mut optimizer = CAME::new();

        // Add full variance parameter
        optimizer.momentum.insert("param_0".to_string(), vec![0.0; 1000]);
        optimizer.variance.insert("param_0".to_string(), vec![0.0; 1000]);

        // Add factorized parameter
        optimizer.momentum.insert("param_1".to_string(), vec![0.0; 500]);
        optimizer.row_factors.insert("param_1".to_string(), vec![0.0; 20]);
        optimizer.col_factors.insert("param_1".to_string(), vec![0.0; 25]);

        let stats = optimizer.memory_stats();
        assert_eq!(stats.num_parameters, 1500); // 1000 + 500 momentum
        assert_eq!(stats.momentum_elements, 1500); // 1000 + 500
        assert_eq!(stats.variance_elements, 1000); // Only full variance counted
        assert_eq!(stats.total_bytes, 10180); // (1500 + 1000 + 20 + 25) * 4 bytes
    }

    #[test]
    fn test_memory_savings_ratio() {
        let mut optimizer = CAME::new();

        // Add parameters with different storage strategies
        optimizer.momentum.insert("param_0".to_string(), vec![0.0; 1000]);
        optimizer.variance.insert("param_0".to_string(), vec![0.0; 1000]); // Full storage

        optimizer.momentum.insert("param_1".to_string(), vec![0.0; 1600]);
        optimizer.row_factors.insert("param_1".to_string(), vec![0.0; 40]); // Factorized storage
        optimizer.col_factors.insert("param_1".to_string(), vec![0.0; 40]);

        let savings = optimizer.memory_savings_ratio();
        assert!(savings > 0.0); // Should have some memory savings
        assert!(savings < 1.0); // But not 100% savings
    }

    #[test]
    fn test_confidence_stats() {
        let mut optimizer = CAME::new();

        // Add confidence states
        optimizer.confidence_states.insert(
            "param_0".to_string(),
            ConfidenceState {
                magnitude_confidence: 0.9,
                direction_confidence: 0.8,
                ..Default::default()
            },
        );

        optimizer.confidence_states.insert(
            "param_1".to_string(),
            ConfidenceState {
                magnitude_confidence: 0.7,
                direction_confidence: 0.6,
                ..Default::default()
            },
        );

        let (avg_mag, avg_dir, high_conf_count, total) = optimizer.confidence_stats();
        assert_relative_eq!(avg_mag, 0.8, epsilon = 1e-6);
        assert_relative_eq!(avg_dir, 0.7, epsilon = 1e-6);
        assert_eq!(high_conf_count, 0); // param_0: 0.9*0.8=0.72 < 0.8, param_1: 0.7*0.6=0.42 < 0.8
        assert_eq!(total, 2);
    }

    #[test]
    fn test_state_dict_operations() {
        let mut optimizer = CAME::new();
        optimizer.state.step = 10;

        // Add various states
        optimizer.momentum.insert("param_0".to_string(), vec![0.1, 0.2]);
        optimizer.variance.insert("param_0".to_string(), vec![0.01, 0.04]);
        optimizer.row_factors.insert("param_1".to_string(), vec![0.1, 0.2]);
        optimizer.col_factors.insert("param_1".to_string(), vec![0.3, 0.4]);
        optimizer.param_shapes.insert("param_0".to_string(), (1, 2));
        optimizer.confidence_states.insert(
            "param_0".to_string(),
            ConfidenceState {
                magnitude_confidence: 0.8,
                direction_confidence: 0.7,
                variance_history: 0.5,
                update_count: 5,
            },
        );

        // Save and load state
        let state_dict = optimizer.state_dict().unwrap();
        let mut new_optimizer = CAME::new();
        new_optimizer.load_state_dict(state_dict).unwrap();

        assert_eq!(new_optimizer.state.step, 10);
        assert_eq!(new_optimizer.momentum["param_0"], vec![0.1, 0.2]);
        assert_eq!(new_optimizer.variance["param_0"], vec![0.01, 0.04]);
        assert_eq!(new_optimizer.row_factors["param_1"], vec![0.1, 0.2]);
        assert_eq!(new_optimizer.param_shapes["param_0"], (1, 2));
        assert_relative_eq!(
            new_optimizer.confidence_states["param_0"].magnitude_confidence,
            0.8
        );
    }

    #[test]
    fn test_lr_setter_getter() {
        let mut optimizer = CAME::new();
        assert_eq!(optimizer.get_lr(), 1e-3);

        optimizer.set_lr(2e-4);
        assert_eq!(optimizer.get_lr(), 2e-4);
    }

    #[test]
    fn test_reset() {
        let mut optimizer = CAME::new();
        optimizer.state.step = 50;
        optimizer.momentum.insert("param_0".to_string(), vec![1.0]);
        optimizer
            .confidence_states
            .insert("param_0".to_string(), ConfidenceState::default());

        optimizer.reset_state();

        assert_eq!(optimizer.state.step, 0);
        assert!(optimizer.momentum.is_empty());
        assert!(optimizer.confidence_states.is_empty());
    }

    #[test]
    fn test_config_serialization() {
        let config = CAMEConfig {
            learning_rate: 1e-4,
            confidence_threshold: 0.75,
            memory_efficiency: 0.6,
            min_factorize_size: 512,
            confidence_update_rate: 0.02,
            ..Default::default()
        };

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: CAMEConfig = serde_json::from_str(&serialized).unwrap();

        assert_relative_eq!(deserialized.learning_rate, config.learning_rate);
        assert_relative_eq!(
            deserialized.confidence_threshold,
            config.confidence_threshold
        );
        assert_eq!(deserialized.min_factorize_size, config.min_factorize_size);
    }
}
