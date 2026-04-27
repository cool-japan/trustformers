//! # Optimizer System for TrustFormeRS
//!
//! This module provides automatic optimizer selection and configuration for various
//! machine learning tasks and model architectures. It follows the design patterns
//! established by HuggingFace Transformers, providing intelligent defaults while
//! allowing for fine-grained control when needed.
//!
//! ## Key Components
//!
//! - **AutoOptimizer**: Main entry point for automatic optimizer creation
//! - **Optimizer trait**: Base interface that all optimizers must implement
//! - **OptimizerGradients/OptimizerUpdate**: Data structures for gradient-based optimization
//! - **LearningRateSchedule**: Various learning rate scheduling strategies
//! - **Concrete Optimizers**: AdamW, Adam, and scheduled optimizer implementations
//!
//! ## Usage Examples
//!
//! ### Automatic Optimizer Selection
//!
//! ```rust,ignore
//! use trustformers::auto::optimizers::AutoOptimizer;
//!
//! // Create optimizer from model configuration
//! let optimizer = AutoOptimizer::from_pretrained("bert-base-uncased")?;
//!
//! // Create optimizer for specific task
//! let task_optimizer = AutoOptimizer::for_task("text-classification", &config)?;
//! ```
//!
//! ### Manual Optimizer Configuration
//!
//! ```rust,ignore
//! use trustformers::auto::optimizers::{AdamWOptimizer, AdamWConfig};
//!
//! let config = AdamWConfig {
//!     learning_rate: 2e-5,
//!     beta1: 0.9,
//!     beta2: 0.999,
//!     weight_decay: 0.01,
//!     eps: 1e-8,
//!     amsgrad: false,
//! };
//! let optimizer = AdamWOptimizer::new(config);
//! ```
//!
//! ### Learning Rate Scheduling
//!
//! ```rust,ignore
//! use trustformers::auto::optimizers::{AutoOptimizer, LearningRateSchedule};
//!
//! let base_optimizer = AutoOptimizer::from_config(&config)?;
//! let schedule = LearningRateSchedule::LinearWarmup {
//!     warmup_steps: 1000,
//!     max_lr: 5e-5,
//! };
//! let scheduled_optimizer = AutoOptimizer::with_schedule(base_optimizer, schedule);
//! ```

use crate::error::Result;
use std::collections::HashMap;

// =============================================================================
// AutoOptimizer - Main Entry Point
// =============================================================================

/// Automatically create optimizers based on model and training configuration
///
/// The AutoOptimizer provides intelligent defaults for different model architectures
/// and tasks, while supporting custom configurations when needed. It follows the
/// principle of "smart defaults, flexible overrides" to minimize configuration
/// overhead while maintaining full control when required.
#[derive(Debug, Clone)]
pub struct AutoOptimizer;

impl AutoOptimizer {
    /// Create an optimizer from model configuration loaded from Hub
    ///
    /// This method loads model configuration from the HuggingFace Hub and selects
    /// an appropriate optimizer based on model characteristics such as parameter
    /// count and architecture type.
    ///
    /// # Arguments
    ///
    /// * `model_name_or_path` - Model identifier from Hub or local path
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    ///    /// let optimizer = AutoOptimizer::from_pretrained("bert-base-uncased")?;

    pub fn from_pretrained(model_name_or_path: &str) -> Result<Box<dyn Optimizer>> {
        let config = crate::hub::load_config_from_hub(model_name_or_path, None)?;
        Self::from_config(&config)
    }

    /// Create an optimizer from configuration object
    ///
    /// Analyzes the model configuration to estimate parameter count and choose
    /// appropriate optimizer settings. Larger models typically benefit from
    /// AdamW with higher weight decay, while smaller models work well with
    /// standard Adam optimization.
    ///
    /// # Parameter Selection Logic
    ///
    /// - **> 1B parameters**: AdamW with lr=1e-5, weight_decay=0.1, beta2=0.95
    /// - **> 100M parameters**: AdamW with lr=2e-5, weight_decay=0.01, beta2=0.999
    /// - **< 100M parameters**: Adam with lr=5e-5, no weight decay
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration as JSON value
    pub fn from_config(config: &serde_json::Value) -> Result<Box<dyn Optimizer>> {
        let model_type = config.get("model_type").and_then(|v| v.as_str()).unwrap_or("default");

        // Choose optimizer based on model characteristics
        let hidden_size =
            config.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(768) as usize;
        let num_layers =
            config.get("num_hidden_layers").and_then(|v| v.as_u64()).unwrap_or(12) as usize;

        // Estimate parameter count (rough approximation)
        let estimated_params = hidden_size * hidden_size * num_layers * 4;

        if estimated_params > 1_000_000_000 {
            // > 1B parameters - Use conservative settings for large models
            Ok(Box::new(AdamWOptimizer::new(AdamWConfig {
                learning_rate: 1e-5,
                beta1: 0.9,
                beta2: 0.95, // Lower beta2 for more stable training
                weight_decay: 0.1,
                eps: 1e-8,
                amsgrad: false,
            })))
        } else if estimated_params > 100_000_000 {
            // > 100M parameters - Standard settings for medium models
            Ok(Box::new(AdamWOptimizer::new(AdamWConfig {
                learning_rate: 2e-5,
                beta1: 0.9,
                beta2: 0.999,
                weight_decay: 0.01,
                eps: 1e-8,
                amsgrad: false,
            })))
        } else {
            // < 100M parameters - Higher learning rate for smaller models
            Ok(Box::new(AdamOptimizer::new(AdamConfig {
                learning_rate: 5e-5,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                amsgrad: false,
            })))
        }
    }

    /// Create an optimizer optimized for a specific task
    ///
    /// Different tasks benefit from different optimization strategies based on
    /// their specific requirements and characteristics.
    ///
    /// # Task-Specific Configurations
    ///
    /// - **Text Generation**: AdamW with beta2=0.95 for stable generation
    /// - **Classification**: Adam with standard settings for faster convergence
    /// - **Question Answering**: AdamW with moderate weight decay for generalization
    ///
    /// # Arguments
    ///
    /// * `task` - Task identifier (e.g., "text-generation", "text-classification")
    /// * `model_config` - Model configuration for fallback parameter estimation
    pub fn for_task(task: &str, model_config: &serde_json::Value) -> Result<Box<dyn Optimizer>> {
        match task {
            "text-generation" | "causal-lm" => {
                // For generation tasks, use AdamW with specific settings
                // Lower beta2 helps with stability during generation
                Ok(Box::new(AdamWOptimizer::new(AdamWConfig {
                    learning_rate: 2e-5,
                    beta1: 0.9,
                    beta2: 0.95,
                    weight_decay: 0.1,
                    eps: 1e-8,
                    amsgrad: false,
                })))
            },
            "text-classification" | "sentiment-analysis" => {
                // For classification, standard Adam often works well
                // Higher learning rate for faster convergence on classification heads
                Ok(Box::new(AdamOptimizer::new(AdamConfig {
                    learning_rate: 2e-5,
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-8,
                    amsgrad: false,
                })))
            },
            "question-answering" => {
                // QA benefits from AdamW with moderate weight decay
                // Balances memorization and generalization
                Ok(Box::new(AdamWOptimizer::new(AdamWConfig {
                    learning_rate: 3e-5,
                    beta1: 0.9,
                    beta2: 0.999,
                    weight_decay: 0.01,
                    eps: 1e-8,
                    amsgrad: false,
                })))
            },
            _ => Self::from_config(model_config),
        }
    }

    /// Create an optimizer with learning rate scheduling
    ///
    /// Wraps any base optimizer with a learning rate schedule for improved
    /// training dynamics. Common schedules include warmup, cosine annealing,
    /// and step decay.
    ///
    /// # Arguments
    ///
    /// * `base_optimizer` - Base optimizer to wrap with scheduling
    /// * `schedule` - Learning rate schedule configuration
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    ///    /// let base = AutoOptimizer::from_config(&config)?;
    /// let schedule = LearningRateSchedule::LinearWarmup {
    ///     warmup_steps: 1000,
    ///     max_lr: 5e-5,
    /// };
    /// let scheduled = AutoOptimizer::with_schedule(base, schedule);

    pub fn with_schedule(
        base_optimizer: Box<dyn Optimizer>,
        schedule: LearningRateSchedule,
    ) -> ScheduledOptimizer {
        ScheduledOptimizer::new(base_optimizer, schedule)
    }
}

// =============================================================================
// Base Optimizer Traits and Types
// =============================================================================

/// Core trait that all optimizers must implement
///
/// This trait defines the essential interface for gradient-based optimization,
/// providing methods for parameter updates, state management, and learning
/// rate control. All concrete optimizer implementations must provide these
/// methods to ensure consistent behavior across the framework.
pub trait Optimizer: Send + Sync + std::fmt::Debug {
    /// Take an optimization step using provided gradients
    ///
    /// This is the core method that performs parameter updates based on
    /// computed gradients. Implementations should update internal state
    /// (momentum, variance estimates, etc.) and return parameter updates.
    ///
    /// # Arguments
    ///
    /// * `gradients` - Gradients for all parameters to be updated
    ///
    /// # Returns
    ///
    /// Parameter updates that should be applied to model weights
    fn step(&mut self, gradients: &OptimizerGradients) -> Result<OptimizerUpdate>;

    /// Zero accumulated gradients
    ///
    /// In frameworks with automatic gradient accumulation, this method
    /// clears any accumulated gradients. Implementation depends on the
    /// specific gradient computation backend.
    fn zero_grad(&mut self);

    /// Get current learning rate
    ///
    /// Returns the current learning rate being used by the optimizer.
    /// This may change over time when using learning rate schedules.
    fn get_lr(&self) -> f64;

    /// Set learning rate
    ///
    /// Updates the optimizer's learning rate. This is typically called
    /// by learning rate schedulers or for manual learning rate adjustments.
    ///
    /// # Arguments
    ///
    /// * `lr` - New learning rate value
    fn set_lr(&mut self, lr: f64);

    /// Get optimizer state for serialization
    ///
    /// Returns a serializable representation of the optimizer's internal
    /// state, including momentum terms, variance estimates, step counts, etc.
    /// This enables saving and loading optimizer state for training resumption.
    fn state_dict(&self) -> HashMap<String, serde_json::Value>;

    /// Load optimizer state from serialized data
    ///
    /// Restores the optimizer's internal state from previously saved data.
    /// This is essential for resuming training from checkpoints.
    ///
    /// # Arguments
    ///
    /// * `state` - Serialized optimizer state
    fn load_state_dict(&mut self, state: HashMap<String, serde_json::Value>) -> Result<()>;
}

/// Container for gradients during optimization
///
/// This structure holds gradients for all model parameters along with
/// their shapes, enabling efficient gradient-based optimization across
/// parameters of different dimensions.
#[derive(Debug, Clone)]
pub struct OptimizerGradients {
    /// Flattened gradients for each named parameter
    pub parameters: HashMap<String, Vec<f32>>,
    /// Original shapes of parameters for reconstruction
    pub parameter_shapes: HashMap<String, Vec<usize>>,
}

/// Container for parameter updates from optimization step
///
/// This structure contains the computed parameter updates along with
/// metadata about the optimization step, such as the effective learning
/// rate and step count.
#[derive(Debug, Clone)]
pub struct OptimizerUpdate {
    /// Parameter updates to be applied to model weights
    pub parameter_updates: HashMap<String, Vec<f32>>,
    /// Learning rate used for this step
    pub learning_rate: f64,
    /// Current step count for tracking training progress
    pub step_count: usize,
}

/// Learning rate scheduling strategies
///
/// Different learning rate schedules can significantly impact training
/// dynamics and final model performance. This enum provides common
/// scheduling strategies used in modern deep learning.
#[derive(Debug, Clone)]
pub enum LearningRateSchedule {
    /// Constant learning rate throughout training
    Constant,

    /// Linear warmup to a maximum learning rate
    ///
    /// Gradually increases learning rate from initial value to max_lr
    /// over warmup_steps, then maintains max_lr
    LinearWarmup { warmup_steps: usize, max_lr: f64 },

    /// Cosine annealing schedule
    ///
    /// Follows a cosine curve from initial learning rate down to eta_min
    /// over t_max steps, providing smooth learning rate decay
    CosineAnnealing { t_max: usize, eta_min: f64 },

    /// Step-wise learning rate decay
    ///
    /// Multiplies learning rate by gamma every step_size steps,
    /// providing periodic learning rate reductions
    StepLR { step_size: usize, gamma: f64 },

    /// Polynomial learning rate decay
    ///
    /// Smoothly decays learning rate from initial value to end_lr
    /// following a polynomial curve with specified power
    PolynomialDecay {
        power: f64,
        end_lr: f64,
        total_steps: usize,
    },
}

// =============================================================================
// Concrete Optimizer Implementations
// =============================================================================
//
// NOTE: These implementations are currently included in this module for
// completeness, but should be refactored into separate files as the
// optimizer system grows:
//
// - adamw.rs: AdamW optimizer implementation
// - adam.rs: Adam optimizer implementation
// - sgd.rs: SGD with momentum implementation
// - scheduled.rs: Learning rate scheduling wrapper
// - lamb.rs: LAMB optimizer for large batch training
// - adafactor.rs: Memory-efficient Adafactor optimizer
//
// This modular structure will improve maintainability and allow for
// easier testing and documentation of individual optimizers.

/// AdamW optimizer implementation
///
/// AdamW (Adam with decoupled Weight decay) is a variant of Adam that
/// separates weight decay from gradient-based optimization, leading to
/// better generalization in many scenarios, especially for transformer models.
///
/// The key difference from Adam is that weight decay is applied directly
/// to parameters rather than being included in the gradient computation,
/// which provides more consistent regularization behavior.
#[derive(Debug, Clone)]
pub struct AdamWOptimizer {
    config: AdamWConfig,
    step_count: usize,
    m: HashMap<String, Vec<f32>>, // First moment estimates
    v: HashMap<String, Vec<f32>>, // Second moment estimates
}

/// Configuration for AdamW optimizer
#[derive(Debug, Clone)]
pub struct AdamWConfig {
    /// Learning rate (alpha)
    pub learning_rate: f64,
    /// Exponential decay rate for first moment estimates
    pub beta1: f64,
    /// Exponential decay rate for second moment estimates
    pub beta2: f64,
    /// Weight decay coefficient for regularization
    pub weight_decay: f64,
    /// Small constant for numerical stability
    pub eps: f64,
    /// Whether to use AMSGrad variant
    pub amsgrad: bool,
}

impl AdamWOptimizer {
    /// Create new AdamW optimizer with given configuration
    pub fn new(config: AdamWConfig) -> Self {
        Self {
            config,
            step_count: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }
}

impl Optimizer for AdamWOptimizer {
    fn step(&mut self, gradients: &OptimizerGradients) -> Result<OptimizerUpdate> {
        self.step_count += 1;
        let mut parameter_updates = HashMap::new();

        for (param_name, grad) in &gradients.parameters {
            // Initialize moment estimates if needed
            if !self.m.contains_key(param_name) {
                self.m.insert(param_name.clone(), vec![0.0; grad.len()]);
                self.v.insert(param_name.clone(), vec![0.0; grad.len()]);
            }

            let m =
                self.m.get_mut(param_name).expect("param_name exists in m after initialization");
            let v =
                self.v.get_mut(param_name).expect("param_name exists in v after initialization");

            let mut updates = Vec::with_capacity(grad.len());

            for i in 0..grad.len() {
                // Update biased first moment estimate
                m[i] = self.config.beta1 as f32 * m[i] + (1.0 - self.config.beta1 as f32) * grad[i];

                // Update biased second raw moment estimate
                v[i] = self.config.beta2 as f32 * v[i]
                    + (1.0 - self.config.beta2 as f32) * grad[i] * grad[i];

                // Compute bias-corrected first moment estimate
                let m_hat = m[i] / (1.0 - (self.config.beta1 as f32).powi(self.step_count as i32));

                // Compute bias-corrected second raw moment estimate
                let v_hat = v[i] / (1.0 - (self.config.beta2 as f32).powi(self.step_count as i32));

                // Compute update (AdamW style weight decay is applied separately)
                let update = -self.config.learning_rate as f32 * m_hat
                    / (v_hat.sqrt() + self.config.eps as f32);
                updates.push(update);
            }

            parameter_updates.insert(param_name.clone(), updates);
        }

        Ok(OptimizerUpdate {
            parameter_updates,
            learning_rate: self.config.learning_rate,
            step_count: self.step_count,
        })
    }

    fn zero_grad(&mut self) {
        // In a real implementation, this would clear accumulated gradients
        // This is typically handled by the training loop or automatic differentiation system
    }

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn state_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut state = HashMap::new();
        state.insert(
            "step_count".to_string(),
            serde_json::Value::Number(self.step_count.into()),
        );
        state.insert(
            "learning_rate".to_string(),
            serde_json::Number::from_f64(self.config.learning_rate)
                .map(serde_json::Value::Number)
                .unwrap_or_else(|| {
                    serde_json::Value::String(format!("{}", self.config.learning_rate))
                }),
        );
        // In a real implementation, would serialize m and v moment estimates
        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, serde_json::Value>) -> Result<()> {
        if let Some(step_count) = state.get("step_count").and_then(|v| v.as_u64()) {
            self.step_count = step_count as usize;
        }
        if let Some(lr) = state.get("learning_rate").and_then(|v| v.as_f64()) {
            self.config.learning_rate = lr;
        }
        Ok(())
    }
}

/// Adam optimizer implementation
///
/// The classic Adam (Adaptive Moment Estimation) optimizer that adapts
/// learning rates for each parameter based on first and second moment
/// estimates of gradients. Works well for many tasks but can sometimes
/// suffer from poor generalization compared to AdamW.
#[derive(Debug, Clone)]
pub struct AdamOptimizer {
    config: AdamConfig,
    step_count: usize,
    m: HashMap<String, Vec<f32>>, // First moment estimates
    v: HashMap<String, Vec<f32>>, // Second moment estimates
}

/// Configuration for Adam optimizer
#[derive(Debug, Clone)]
pub struct AdamConfig {
    /// Learning rate (alpha)
    pub learning_rate: f64,
    /// Exponential decay rate for first moment estimates
    pub beta1: f64,
    /// Exponential decay rate for second moment estimates
    pub beta2: f64,
    /// Small constant for numerical stability
    pub eps: f64,
    /// Whether to use AMSGrad variant
    pub amsgrad: bool,
}

impl AdamOptimizer {
    /// Create new Adam optimizer with given configuration
    pub fn new(config: AdamConfig) -> Self {
        Self {
            config,
            step_count: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }
}

impl Optimizer for AdamOptimizer {
    fn step(&mut self, gradients: &OptimizerGradients) -> Result<OptimizerUpdate> {
        // Similar to AdamW but without weight decay
        self.step_count += 1;
        let mut parameter_updates = HashMap::new();

        for (param_name, grad) in &gradients.parameters {
            if !self.m.contains_key(param_name) {
                self.m.insert(param_name.clone(), vec![0.0; grad.len()]);
                self.v.insert(param_name.clone(), vec![0.0; grad.len()]);
            }

            let m =
                self.m.get_mut(param_name).expect("param_name exists in m after initialization");
            let v =
                self.v.get_mut(param_name).expect("param_name exists in v after initialization");

            let mut updates = Vec::with_capacity(grad.len());

            for i in 0..grad.len() {
                m[i] = self.config.beta1 as f32 * m[i] + (1.0 - self.config.beta1 as f32) * grad[i];
                v[i] = self.config.beta2 as f32 * v[i]
                    + (1.0 - self.config.beta2 as f32) * grad[i] * grad[i];

                let m_hat = m[i] / (1.0 - (self.config.beta1 as f32).powi(self.step_count as i32));
                let v_hat = v[i] / (1.0 - (self.config.beta2 as f32).powi(self.step_count as i32));

                let update = -self.config.learning_rate as f32 * m_hat
                    / (v_hat.sqrt() + self.config.eps as f32);
                updates.push(update);
            }

            parameter_updates.insert(param_name.clone(), updates);
        }

        Ok(OptimizerUpdate {
            parameter_updates,
            learning_rate: self.config.learning_rate,
            step_count: self.step_count,
        })
    }

    fn zero_grad(&mut self) {}

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn state_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut state = HashMap::new();
        state.insert(
            "step_count".to_string(),
            serde_json::Value::Number(self.step_count.into()),
        );
        state.insert(
            "learning_rate".to_string(),
            serde_json::Number::from_f64(self.config.learning_rate)
                .map(serde_json::Value::Number)
                .unwrap_or_else(|| {
                    serde_json::Value::String(format!("{}", self.config.learning_rate))
                }),
        );
        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, serde_json::Value>) -> Result<()> {
        if let Some(step_count) = state.get("step_count").and_then(|v| v.as_u64()) {
            self.step_count = step_count as usize;
        }
        if let Some(lr) = state.get("learning_rate").and_then(|v| v.as_f64()) {
            self.config.learning_rate = lr;
        }
        Ok(())
    }
}

/// Optimizer wrapper that applies learning rate scheduling
///
/// This wrapper can be applied to any base optimizer to provide dynamic
/// learning rate adjustment during training. Different schedules can
/// significantly impact convergence speed and final model quality.
#[derive(Debug)]
pub struct ScheduledOptimizer {
    optimizer: Box<dyn Optimizer>,
    schedule: LearningRateSchedule,
    initial_lr: f64,
    current_step: usize,
}

impl ScheduledOptimizer {
    /// Create new scheduled optimizer
    ///
    /// # Arguments
    ///
    /// * `optimizer` - Base optimizer to wrap with scheduling
    /// * `schedule` - Learning rate schedule to apply
    pub fn new(optimizer: Box<dyn Optimizer>, schedule: LearningRateSchedule) -> Self {
        let initial_lr = optimizer.get_lr();
        Self {
            optimizer,
            schedule,
            initial_lr,
            current_step: 0,
        }
    }

    /// Update learning rate based on current step and schedule
    fn update_learning_rate(&mut self) {
        let new_lr = match &self.schedule {
            LearningRateSchedule::Constant => self.initial_lr,
            LearningRateSchedule::LinearWarmup {
                warmup_steps,
                max_lr,
            } => {
                if self.current_step < *warmup_steps {
                    self.initial_lr
                        + (max_lr - self.initial_lr)
                            * (self.current_step as f64 / *warmup_steps as f64)
                } else {
                    *max_lr
                }
            },
            LearningRateSchedule::CosineAnnealing { t_max, eta_min } => {
                eta_min
                    + (self.initial_lr - eta_min)
                        * (1.0
                            + (std::f64::consts::PI * self.current_step as f64 / *t_max as f64)
                                .cos())
                        / 2.0
            },
            LearningRateSchedule::StepLR { step_size, gamma } => {
                self.initial_lr * gamma.powi((self.current_step / step_size) as i32)
            },
            LearningRateSchedule::PolynomialDecay {
                power,
                end_lr,
                total_steps,
            } => {
                if self.current_step >= *total_steps {
                    *end_lr
                } else {
                    let decay_factor =
                        (1.0 - self.current_step as f64 / *total_steps as f64).powf(*power);
                    end_lr + (self.initial_lr - end_lr) * decay_factor
                }
            },
        };

        self.optimizer.set_lr(new_lr);
    }
}

impl Optimizer for ScheduledOptimizer {
    fn step(&mut self, gradients: &OptimizerGradients) -> Result<OptimizerUpdate> {
        self.current_step += 1;
        self.update_learning_rate();
        self.optimizer.step(gradients)
    }

    fn zero_grad(&mut self) {
        self.optimizer.zero_grad();
    }

    fn get_lr(&self) -> f64 {
        self.optimizer.get_lr()
    }

    fn set_lr(&mut self, lr: f64) {
        self.initial_lr = lr;
        self.optimizer.set_lr(lr);
    }

    fn state_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut state = self.optimizer.state_dict();
        state.insert(
            "current_step".to_string(),
            serde_json::Value::Number(self.current_step.into()),
        );
        state.insert(
            "initial_lr".to_string(),
            serde_json::Number::from_f64(self.initial_lr)
                .map(serde_json::Value::Number)
                .unwrap_or_else(|| serde_json::Value::String(format!("{}", self.initial_lr))),
        );
        state
    }

    fn load_state_dict(&mut self, mut state: HashMap<String, serde_json::Value>) -> Result<()> {
        if let Some(step) = state.remove("current_step").and_then(|v| v.as_u64()) {
            self.current_step = step as usize;
        }
        if let Some(lr) = state.remove("initial_lr").and_then(|v| v.as_f64()) {
            self.initial_lr = lr;
        }
        self.optimizer.load_state_dict(state)
    }
}

// =============================================================================
// Public API
// =============================================================================

// All main components are already public and available for import:
// - AutoOptimizer: Main entry point for automatic optimizer creation
// - Optimizer: Base trait for all optimizers
// - OptimizerGradients/OptimizerUpdate: Data structures for optimization
// - LearningRateSchedule: Learning rate scheduling strategies
// - AdamWOptimizer/AdamOptimizer: Concrete optimizer implementations
// - ScheduledOptimizer: Optimizer wrapper with learning rate scheduling

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // LCG for deterministic pseudo-random numbers (no rand crate)
    // -------------------------------------------------------------------------

    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg { state: seed }
        }

        fn next(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005_u64)
                .wrapping_add(1_442_695_040_888_963_407_u64);
            self.state
        }

        fn next_f32(&mut self) -> f32 {
            (self.next() >> 11) as f32 / (1u64 << 53) as f32
        }
    }

    // -------------------------------------------------------------------------
    // AdamWConfig
    // -------------------------------------------------------------------------

    #[test]
    fn test_adamw_config_creation() {
        let config = AdamWConfig {
            learning_rate: 2e-5,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
            eps: 1e-8,
            amsgrad: false,
        };
        let diff = (config.learning_rate - 2e-5).abs();
        assert!(diff < 1e-10, "learning_rate should be set correctly");
        assert!(!config.amsgrad, "amsgrad should be false");
    }

    #[test]
    fn test_adamw_config_weight_decay() {
        let config = AdamWConfig {
            learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.1,
            eps: 1e-8,
            amsgrad: false,
        };
        let diff = (config.weight_decay - 0.1).abs();
        assert!(diff < 1e-10, "weight_decay should be 0.1");
    }

    // -------------------------------------------------------------------------
    // AdamConfig
    // -------------------------------------------------------------------------

    #[test]
    fn test_adam_config_creation() {
        let config = AdamConfig {
            learning_rate: 5e-5,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            amsgrad: false,
        };
        let diff = (config.learning_rate - 5e-5).abs();
        assert!(diff < 1e-12, "learning_rate should be set correctly");
    }

    #[test]
    fn test_adam_config_beta_values() {
        let config = AdamConfig {
            learning_rate: 1e-3,
            beta1: 0.95,
            beta2: 0.99,
            eps: 1e-6,
            amsgrad: true,
        };
        let b1_diff = (config.beta1 - 0.95).abs();
        let b2_diff = (config.beta2 - 0.99).abs();
        assert!(b1_diff < 1e-10, "beta1 should be 0.95");
        assert!(b2_diff < 1e-10, "beta2 should be 0.99");
        assert!(config.amsgrad, "amsgrad should be true");
    }

    // -------------------------------------------------------------------------
    // AdamWOptimizer
    // -------------------------------------------------------------------------

    #[test]
    fn test_adamw_optimizer_creation() {
        let config = AdamWConfig {
            learning_rate: 2e-5,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
            eps: 1e-8,
            amsgrad: false,
        };
        let optimizer = AdamWOptimizer::new(config);
        let diff = (optimizer.get_lr() - 2e-5).abs();
        assert!(diff < 1e-12, "Initial LR should match config");
    }

    #[test]
    fn test_adamw_get_lr() {
        let optimizer = AdamWOptimizer::new(AdamWConfig {
            learning_rate: 3e-4,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
            eps: 1e-8,
            amsgrad: false,
        });
        let diff = (optimizer.get_lr() - 3e-4).abs();
        assert!(diff < 1e-12, "get_lr should return initial learning rate");
    }

    #[test]
    fn test_adamw_set_lr() {
        let mut optimizer = AdamWOptimizer::new(AdamWConfig {
            learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
            eps: 1e-8,
            amsgrad: false,
        });
        optimizer.set_lr(5e-5);
        let diff = (optimizer.get_lr() - 5e-5).abs();
        assert!(diff < 1e-12, "LR should be updated after set_lr");
    }

    #[test]
    fn test_adamw_step_produces_update() {
        let mut optimizer = AdamWOptimizer::new(AdamWConfig {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
            eps: 1e-8,
            amsgrad: false,
        });
        let mut lcg = Lcg::new(42);
        let grads: Vec<f32> = (0..4).map(|_| lcg.next_f32() - 0.5).collect();
        let mut parameters = HashMap::new();
        parameters.insert("layer.weight".to_string(), grads);
        let mut parameter_shapes = HashMap::new();
        parameter_shapes.insert("layer.weight".to_string(), vec![4]);
        let gradients = OptimizerGradients {
            parameters,
            parameter_shapes,
        };

        let result = optimizer.step(&gradients);
        assert!(result.is_ok(), "step() should succeed");
        if let Ok(update) = result {
            assert_eq!(
                update.step_count, 1,
                "Step count should be 1 after first step"
            );
            assert!(
                update.parameter_updates.contains_key("layer.weight"),
                "Update should contain parameter updates"
            );
        }
    }

    #[test]
    fn test_adamw_step_count_increments() {
        let mut optimizer = AdamWOptimizer::new(AdamWConfig {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.0,
            eps: 1e-8,
            amsgrad: false,
        });
        let grads = vec![0.1_f32, -0.2, 0.05];
        let mut parameters = HashMap::new();
        parameters.insert("w".to_string(), grads);
        let mut shapes = HashMap::new();
        shapes.insert("w".to_string(), vec![3]);
        let gradients = OptimizerGradients {
            parameters,
            parameter_shapes: shapes,
        };

        for expected_step in 1..=3usize {
            let result = optimizer.step(&gradients);
            if let Ok(update) = result {
                assert_eq!(update.step_count, expected_step);
            }
        }
    }

    #[test]
    fn test_adamw_state_dict_contains_step_count() {
        let mut optimizer = AdamWOptimizer::new(AdamWConfig {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
            eps: 1e-8,
            amsgrad: false,
        });
        // Do one step to increment counter
        let mut parameters = HashMap::new();
        parameters.insert("w".to_string(), vec![0.1_f32]);
        let mut shapes = HashMap::new();
        shapes.insert("w".to_string(), vec![1]);
        let _ = optimizer.step(&OptimizerGradients {
            parameters,
            parameter_shapes: shapes,
        });
        let state = optimizer.state_dict();
        assert!(
            state.contains_key("step_count"),
            "state_dict should contain step_count"
        );
    }

    #[test]
    fn test_adamw_load_state_dict_updates_lr() {
        let mut optimizer = AdamWOptimizer::new(AdamWConfig {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
            eps: 1e-8,
            amsgrad: false,
        });
        let mut state = HashMap::new();
        state.insert("learning_rate".to_string(), serde_json::json!(2e-4));
        let result = optimizer.load_state_dict(state);
        assert!(result.is_ok(), "load_state_dict should succeed");
        let diff = (optimizer.get_lr() - 2e-4).abs();
        assert!(diff < 1e-12, "LR should be updated from loaded state");
    }

    // -------------------------------------------------------------------------
    // AdamOptimizer
    // -------------------------------------------------------------------------

    #[test]
    fn test_adam_optimizer_creation() {
        let optimizer = AdamOptimizer::new(AdamConfig {
            learning_rate: 5e-5,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            amsgrad: false,
        });
        let diff = (optimizer.get_lr() - 5e-5).abs();
        assert!(diff < 1e-12, "Initial LR should match config");
    }

    #[test]
    fn test_adam_set_lr() {
        let mut optimizer = AdamOptimizer::new(AdamConfig {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            amsgrad: false,
        });
        optimizer.set_lr(1e-4);
        let diff = (optimizer.get_lr() - 1e-4).abs();
        assert!(diff < 1e-12, "LR should be updated after set_lr");
    }

    #[test]
    fn test_adam_step_produces_update() {
        let mut optimizer = AdamOptimizer::new(AdamConfig {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            amsgrad: false,
        });
        let mut parameters = HashMap::new();
        parameters.insert("bias".to_string(), vec![0.5_f32, -0.3]);
        let mut shapes = HashMap::new();
        shapes.insert("bias".to_string(), vec![2]);
        let gradients = OptimizerGradients {
            parameters,
            parameter_shapes: shapes,
        };

        let result = optimizer.step(&gradients);
        assert!(result.is_ok(), "Adam step should succeed");
        if let Ok(update) = result {
            assert!(
                update.parameter_updates.contains_key("bias"),
                "Update should contain 'bias'"
            );
        }
    }

    #[test]
    fn test_adam_zero_grad_does_not_panic() {
        let mut optimizer = AdamOptimizer::new(AdamConfig {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            amsgrad: false,
        });
        optimizer.zero_grad(); // Should not panic
    }

    // -------------------------------------------------------------------------
    // LearningRateSchedule
    // -------------------------------------------------------------------------

    #[test]
    fn test_lr_schedule_constant_variant() {
        let schedule = LearningRateSchedule::Constant;
        // Just verifying that the variant can be created and matched
        assert!(matches!(schedule, LearningRateSchedule::Constant));
    }

    #[test]
    fn test_lr_schedule_linear_warmup_fields() {
        let schedule = LearningRateSchedule::LinearWarmup {
            warmup_steps: 1000,
            max_lr: 5e-5,
        };
        if let LearningRateSchedule::LinearWarmup {
            warmup_steps,
            max_lr,
        } = schedule
        {
            assert_eq!(warmup_steps, 1000);
            let diff = (max_lr - 5e-5).abs();
            assert!(diff < 1e-12, "max_lr should be 5e-5");
        } else {
            panic!("Expected LinearWarmup variant");
        }
    }

    #[test]
    fn test_lr_schedule_cosine_annealing_fields() {
        let schedule = LearningRateSchedule::CosineAnnealing {
            t_max: 500,
            eta_min: 1e-6,
        };
        if let LearningRateSchedule::CosineAnnealing { t_max, eta_min } = schedule {
            assert_eq!(t_max, 500);
            let diff = (eta_min - 1e-6).abs();
            assert!(diff < 1e-12, "eta_min should be 1e-6");
        } else {
            panic!("Expected CosineAnnealing variant");
        }
    }

    #[test]
    fn test_lr_schedule_step_lr_fields() {
        let schedule = LearningRateSchedule::StepLR {
            step_size: 100,
            gamma: 0.1,
        };
        if let LearningRateSchedule::StepLR { step_size, gamma } = schedule {
            assert_eq!(step_size, 100);
            let diff = (gamma - 0.1).abs();
            assert!(diff < 1e-12, "gamma should be 0.1");
        } else {
            panic!("Expected StepLR variant");
        }
    }

    #[test]
    fn test_lr_schedule_polynomial_decay_fields() {
        let schedule = LearningRateSchedule::PolynomialDecay {
            power: 2.0,
            end_lr: 1e-7,
            total_steps: 10_000,
        };
        if let LearningRateSchedule::PolynomialDecay {
            power,
            end_lr,
            total_steps,
        } = schedule
        {
            let p_diff = (power - 2.0).abs();
            assert!(p_diff < 1e-10, "power should be 2.0");
            let e_diff = (end_lr - 1e-7).abs();
            assert!(e_diff < 1e-14, "end_lr should be 1e-7");
            assert_eq!(total_steps, 10_000);
        } else {
            panic!("Expected PolynomialDecay variant");
        }
    }

    // -------------------------------------------------------------------------
    // ScheduledOptimizer
    // -------------------------------------------------------------------------

    #[test]
    fn test_scheduled_optimizer_constant_lr() {
        let base = AdamWOptimizer::new(AdamWConfig {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
            eps: 1e-8,
            amsgrad: false,
        });
        let mut sched = ScheduledOptimizer::new(Box::new(base), LearningRateSchedule::Constant);
        let initial_lr = sched.get_lr();

        let mut parameters = HashMap::new();
        parameters.insert("w".to_string(), vec![0.1_f32]);
        let mut shapes = HashMap::new();
        shapes.insert("w".to_string(), vec![1]);
        let _ = sched.step(&OptimizerGradients {
            parameters,
            parameter_shapes: shapes,
        });

        let lr_diff = (sched.get_lr() - initial_lr).abs();
        assert!(
            lr_diff < 1e-10,
            "Constant schedule should keep LR unchanged"
        );
    }

    #[test]
    fn test_scheduled_optimizer_warmup_increases_lr() {
        let initial_lr = 1e-5_f64;
        let max_lr = 5e-4_f64;
        let base = AdamWOptimizer::new(AdamWConfig {
            learning_rate: initial_lr,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
            eps: 1e-8,
            amsgrad: false,
        });
        let mut sched = ScheduledOptimizer::new(
            Box::new(base),
            LearningRateSchedule::LinearWarmup {
                warmup_steps: 100,
                max_lr,
            },
        );

        let mut parameters = HashMap::new();
        parameters.insert("w".to_string(), vec![0.01_f32]);
        let mut shapes = HashMap::new();
        shapes.insert("w".to_string(), vec![1]);

        // After several warmup steps, LR should increase
        for _ in 0..50 {
            let _ = sched.step(&OptimizerGradients {
                parameters: parameters.clone(),
                parameter_shapes: shapes.clone(),
            });
        }
        let lr_after = sched.get_lr();
        assert!(lr_after > initial_lr, "LR should increase during warmup");
        assert!(lr_after < max_lr + 1e-10, "LR should not exceed max_lr");
    }

    #[test]
    fn test_scheduled_optimizer_state_dict_contains_step() {
        let base = AdamOptimizer::new(AdamConfig {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            amsgrad: false,
        });
        let mut sched = ScheduledOptimizer::new(Box::new(base), LearningRateSchedule::Constant);
        let mut parameters = HashMap::new();
        parameters.insert("w".to_string(), vec![0.1_f32]);
        let mut shapes = HashMap::new();
        shapes.insert("w".to_string(), vec![1]);
        let _ = sched.step(&OptimizerGradients {
            parameters,
            parameter_shapes: shapes,
        });
        let state = sched.state_dict();
        assert!(
            state.contains_key("current_step"),
            "state_dict should have current_step key"
        );
    }

    // -------------------------------------------------------------------------
    // AutoOptimizer::from_config
    // -------------------------------------------------------------------------

    #[test]
    fn test_auto_optimizer_from_config_small_model() {
        let config = serde_json::json!({
            "model_type": "bert",
            "hidden_size": 128,
            "num_hidden_layers": 2
        });
        let result = AutoOptimizer::from_config(&config);
        assert!(result.is_ok(), "AutoOptimizer::from_config should succeed");
        if let Ok(optimizer) = result {
            assert!(
                optimizer.get_lr() > 0.0,
                "Optimizer should have positive LR"
            );
        }
    }

    #[test]
    fn test_auto_optimizer_from_config_large_model() {
        let config = serde_json::json!({
            "model_type": "gpt2",
            "hidden_size": 1024,
            "num_hidden_layers": 36
        });
        let result = AutoOptimizer::from_config(&config);
        assert!(
            result.is_ok(),
            "AutoOptimizer::from_config should succeed for large model"
        );
        if let Ok(optimizer) = result {
            // Large model should use lower LR
            assert!(
                optimizer.get_lr() <= 2e-5 + 1e-12,
                "Large model should use lower LR"
            );
        }
    }

    #[test]
    fn test_auto_optimizer_for_task_text_generation() {
        let config = serde_json::json!({});
        let result = AutoOptimizer::for_task("text-generation", &config);
        assert!(result.is_ok(), "for_task text-generation should succeed");
    }

    #[test]
    fn test_auto_optimizer_for_task_classification() {
        let config = serde_json::json!({});
        let result = AutoOptimizer::for_task("text-classification", &config);
        assert!(
            result.is_ok(),
            "for_task text-classification should succeed"
        );
    }

    #[test]
    fn test_auto_optimizer_for_task_question_answering() {
        let config = serde_json::json!({});
        let result = AutoOptimizer::for_task("question-answering", &config);
        assert!(result.is_ok(), "for_task question-answering should succeed");
    }

    #[test]
    fn test_auto_optimizer_for_task_unknown_uses_default() {
        let config = serde_json::json!({
            "hidden_size": 256,
            "num_hidden_layers": 4
        });
        let result = AutoOptimizer::for_task("some-unknown-task", &config);
        assert!(result.is_ok(), "Unknown task should fall back to default");
    }

    #[test]
    fn test_auto_optimizer_with_schedule() {
        let base = AutoOptimizer::from_config(&serde_json::json!({}));
        assert!(base.is_ok(), "Base optimizer should be created");
        if let Ok(base_opt) = base {
            let schedule = LearningRateSchedule::LinearWarmup {
                warmup_steps: 500,
                max_lr: 5e-5,
            };
            let sched = AutoOptimizer::with_schedule(base_opt, schedule);
            assert!(
                sched.get_lr() > 0.0,
                "Scheduled optimizer should have positive LR"
            );
        }
    }

    // -------------------------------------------------------------------------
    // OptimizerGradients / OptimizerUpdate
    // -------------------------------------------------------------------------

    #[test]
    fn test_optimizer_gradients_creation() {
        let mut parameters = HashMap::new();
        parameters.insert("layer1.weight".to_string(), vec![0.1_f32, 0.2, -0.3]);
        let mut parameter_shapes = HashMap::new();
        parameter_shapes.insert("layer1.weight".to_string(), vec![3]);
        let gradients = OptimizerGradients {
            parameters,
            parameter_shapes,
        };
        assert_eq!(gradients.parameters.len(), 1);
        assert!(gradients.parameters.contains_key("layer1.weight"));
    }

    #[test]
    fn test_optimizer_update_fields() {
        let mut parameter_updates = HashMap::new();
        parameter_updates.insert("w".to_string(), vec![-0.001_f32, 0.002]);
        let update = OptimizerUpdate {
            parameter_updates,
            learning_rate: 1e-3,
            step_count: 5,
        };
        assert_eq!(update.step_count, 5);
        let lr_diff = (update.learning_rate - 1e-3).abs();
        assert!(lr_diff < 1e-12, "learning_rate should match");
        assert!(update.parameter_updates.contains_key("w"));
    }
}
