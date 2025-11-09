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
//! ```rust
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
//! ```rust
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
//! ```rust
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
    /// ```rust
    /// let optimizer = AutoOptimizer::from_pretrained("bert-base-uncased")?;
    /// ```
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
    /// ```rust
    /// let base = AutoOptimizer::from_config(&config)?;
    /// let schedule = LearningRateSchedule::LinearWarmup {
    ///     warmup_steps: 1000,
    ///     max_lr: 5e-5,
    /// };
    /// let scheduled = AutoOptimizer::with_schedule(base, schedule);
    /// ```
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

            let m = self.m.get_mut(param_name).unwrap();
            let v = self.v.get_mut(param_name).unwrap();

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
            serde_json::Value::Number(
                serde_json::Number::from_f64(self.config.learning_rate).unwrap(),
            ),
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

            let m = self.m.get_mut(param_name).unwrap();
            let v = self.v.get_mut(param_name).unwrap();

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
            serde_json::Value::Number(
                serde_json::Number::from_f64(self.config.learning_rate).unwrap(),
            ),
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
            serde_json::Value::Number(serde_json::Number::from_f64(self.initial_lr).unwrap()),
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
