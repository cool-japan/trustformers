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
        // Selection below is purely a function of the estimated parameter
        // count (see the doc comment above); `model_type` is not read here
        // because it does not currently affect the choice of optimizer.

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
    /// Any gradients previously handed to [`Optimizer::accumulate_gradients`]
    /// since the last [`Optimizer::zero_grad`] are folded in elementwise with
    /// whatever is passed here, matching the usual "accumulate across
    /// micro-batches, then step" training loop pattern.
    ///
    /// # Arguments
    ///
    /// * `gradients` - Gradients for all parameters to be updated
    ///
    /// # Returns
    ///
    /// Parameter updates that should be applied to model weights
    ///
    /// # Errors
    ///
    /// Returns an error if a gradient's length disagrees with the length of
    /// the accumulated gradient (or restored moment state) for the same
    /// parameter, rather than indexing out of bounds or silently truncating.
    fn step(&mut self, gradients: &OptimizerGradients) -> Result<OptimizerUpdate>;

    /// Accumulate gradients into an internal per-parameter buffer without
    /// taking an optimization step.
    ///
    /// Calling this multiple times sums the gradients elementwise (the same
    /// semantics as calling `.backward()` repeatedly without an intervening
    /// `zero_grad()` in a typical autodiff-based trainer): a gradient
    /// accumulation loop can call this once per micro-batch and then call
    /// [`Optimizer::step`] once per effective batch.
    ///
    /// # Errors
    ///
    /// Returns an error if a parameter is accumulated at one length and then
    /// accumulated again at a different length (e.g. the model shape
    /// changed without an intervening [`Optimizer::zero_grad`]).
    fn accumulate_gradients(&mut self, gradients: &OptimizerGradients) -> Result<()>;

    /// Zero accumulated gradients
    ///
    /// Clears the buffer built up by [`Optimizer::accumulate_gradients`].
    /// After this call, [`Optimizer::step`] uses only the gradients passed
    /// to it directly, with nothing carried over from prior accumulation.
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
    /// state, including the first/second moment estimates (for optimizers
    /// that have them) and step count. This enables saving and loading
    /// optimizer state for training resumption.
    ///
    /// # Errors
    ///
    /// Returns an error rather than silently emitting JSON `null` when a
    /// moment estimate holds a non-finite (`NaN`/`Infinity`) value — `null`
    /// would round-trip back as `0.0`, hiding that the optimizer had
    /// diverged.
    fn state_dict(&self) -> Result<HashMap<String, serde_json::Value>>;

    /// Load optimizer state from serialized data
    ///
    /// Restores the optimizer's internal state from previously saved data.
    /// This is essential for resuming training from checkpoints.
    ///
    /// # Arguments
    ///
    /// * `state` - Serialized optimizer state
    ///
    /// # Errors
    ///
    /// Returns an error if a moment-estimate entry is present but is not a
    /// JSON object of parameter name -> array-of-numbers, or contains a
    /// non-numeric entry (including `null`, which a state dict produced by
    /// an unguarded serializer could contain in place of a diverged value).
    fn load_state_dict(&mut self, state: HashMap<String, serde_json::Value>) -> Result<()>;
}

/// Serialize a per-parameter moment-estimate map (`m` or `v`) to JSON,
/// rejecting any non-finite value instead of letting `serde_json` silently
/// turn it into `null`.
///
/// `serde_json::Number::from_f64` returns `None` for `NaN`/`Infinity`, and
/// `serde_json::to_value` on such an `f32` therefore serializes it as JSON
/// `null` with no error. A `null` in a moment estimate would round-trip back
/// through [`moment_map_from_json`] as an error (good), but silently
/// *skipping* the value at serialize time would be worse: it would make a
/// diverged optimizer's checkpoint look like a healthy all-zero one. This
/// helper fails loudly instead.
fn moment_map_to_json(
    moments: &HashMap<String, Vec<f32>>,
    which: &str,
) -> Result<serde_json::Value> {
    let mut object = serde_json::Map::with_capacity(moments.len());
    for (name, values) in moments {
        let mut array = Vec::with_capacity(values.len());
        for &value in values {
            let number = serde_json::Number::from_f64(value as f64).ok_or_else(|| {
                crate::error::TrustformersError::runtime_error(format!(
                    "optimizer state_dict: non-finite value in `{which}` moment estimate for \
                     parameter `{name}` (value = {value}); cannot serialize a diverged \
                     optimizer's state losslessly"
                ))
            })?;
            array.push(serde_json::Value::Number(number));
        }
        object.insert(name.clone(), serde_json::Value::Array(array));
    }
    Ok(serde_json::Value::Object(object))
}

/// Inverse of [`moment_map_to_json`]. Returns an empty map when `value` is
/// `None` (the key was absent from the state dict, e.g. a checkpoint saved
/// before this field existed), and a structured error for anything present
/// but malformed rather than silently defaulting missing entries to zero.
fn moment_map_from_json(
    value: Option<&serde_json::Value>,
    which: &str,
) -> Result<HashMap<String, Vec<f32>>> {
    let mut result = HashMap::new();
    let Some(value) = value else {
        return Ok(result);
    };
    let object = value.as_object().ok_or_else(|| {
        crate::error::TrustformersError::runtime_error(format!(
            "optimizer load_state_dict: `{which}` must be a JSON object mapping parameter names \
             to arrays of floats, got {value}"
        ))
    })?;
    for (name, array_value) in object {
        let array = array_value.as_array().ok_or_else(|| {
            crate::error::TrustformersError::runtime_error(format!(
                "optimizer load_state_dict: `{which}.{name}` must be a JSON array of floats, got \
                 {array_value}"
            ))
        })?;
        let mut values = Vec::with_capacity(array.len());
        for entry in array {
            let f = entry.as_f64().ok_or_else(|| {
                crate::error::TrustformersError::runtime_error(format!(
                    "optimizer load_state_dict: `{which}.{name}` contains a non-numeric entry \
                     ({entry}) -- a JSON `null` here usually means the checkpoint was written by \
                     a serializer that silently dropped a non-finite (NaN/Infinity) value"
                ))
            })?;
            values.push(f as f32);
        }
        result.insert(name.clone(), values);
    }
    Ok(result)
}

/// Compute the effective per-parameter gradient for a `step()` call: the
/// gradient passed to `step` plus whatever was accumulated via
/// `accumulate_gradients` for the same parameter (elementwise sum), in
/// stable order (parameters present only in `gradients`, then parameters
/// present only in `accumulated`).
///
/// # Errors
///
/// Returns an error if a parameter appears in both maps with different
/// lengths.
fn merge_accumulated_gradients(
    gradients: &OptimizerGradients,
    accumulated: &HashMap<String, Vec<f32>>,
) -> Result<HashMap<String, Vec<f32>>> {
    let mut effective = HashMap::with_capacity(gradients.parameters.len().max(accumulated.len()));

    for (name, passed) in &gradients.parameters {
        match accumulated.get(name) {
            Some(acc) => {
                if acc.len() != passed.len() {
                    return Err(crate::error::TrustformersError::runtime_error(format!(
                        "optimizer step: accumulated gradient for `{name}` has {} values but the \
                         step's gradient has {} -- shapes must match (call zero_grad() if the \
                         model shape changed)",
                        acc.len(),
                        passed.len()
                    )));
                }
                effective.insert(
                    name.clone(),
                    passed.iter().zip(acc.iter()).map(|(g, a)| g + a).collect(),
                );
            },
            None => {
                effective.insert(name.clone(), passed.clone());
            },
        }
    }
    for (name, acc) in accumulated {
        effective.entry(name.clone()).or_insert_with(|| acc.clone());
    }

    Ok(effective)
}

/// Accumulate `gradients` elementwise into `accumulated`, in place.
///
/// # Errors
///
/// Returns an error if a parameter was previously accumulated at a
/// different length than the newly supplied gradient.
fn accumulate_into(
    accumulated: &mut HashMap<String, Vec<f32>>,
    gradients: &OptimizerGradients,
) -> Result<()> {
    for (name, grad) in &gradients.parameters {
        match accumulated.get_mut(name) {
            Some(existing) => {
                if existing.len() != grad.len() {
                    return Err(crate::error::TrustformersError::runtime_error(format!(
                        "optimizer accumulate_gradients: `{name}` was previously accumulated at \
                         {} values, new gradient has {} -- call zero_grad() before changing \
                         parameter shape",
                        existing.len(),
                        grad.len()
                    )));
                }
                for (acc, g) in existing.iter_mut().zip(grad.iter()) {
                    *acc += g;
                }
            },
            None => {
                accumulated.insert(name.clone(), grad.clone());
            },
        }
    }
    Ok(())
}

/// Guard against a restored (or freshly initialized) moment-estimate vector
/// whose length disagrees with the current effective gradient -- indexing
/// into a mismatched vector would otherwise panic instead of erroring.
fn ensure_moment_len(
    moment: &[f32],
    expected_len: usize,
    which: &str,
    param_name: &str,
) -> Result<()> {
    if moment.len() != expected_len {
        return Err(crate::error::TrustformersError::runtime_error(format!(
            "optimizer step: restored `{which}` state for `{param_name}` has {} entries but the \
             gradient has {expected_len} -- this checkpoint does not match the current model shape",
            moment.len()
        )));
    }
    Ok(())
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
    /// Per-parameter running maximum of `v`, used only when
    /// `config.amsgrad` is set (Reddi et al., 2018). Kept separate from `v`
    /// so state_dict() can omit it for the common non-AMSGrad case.
    v_max: HashMap<String, Vec<f32>>,
    /// Gradients accumulated via [`Optimizer::accumulate_gradients`] since
    /// the last [`Optimizer::zero_grad`]; folded into the next [`Optimizer::step`].
    accumulated_gradients: HashMap<String, Vec<f32>>,
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
            v_max: HashMap::new(),
            accumulated_gradients: HashMap::new(),
        }
    }
}

impl Optimizer for AdamWOptimizer {
    fn step(&mut self, gradients: &OptimizerGradients) -> Result<OptimizerUpdate> {
        let effective_gradients =
            merge_accumulated_gradients(gradients, &self.accumulated_gradients)?;
        self.step_count += 1;
        let mut parameter_updates = HashMap::new();

        for (param_name, grad) in &effective_gradients {
            // Initialize moment estimates if needed (entry API avoids a fallible lookup)
            let m = self.m.entry(param_name.clone()).or_insert_with(|| vec![0.0; grad.len()]);
            ensure_moment_len(m, grad.len(), "m", param_name)?;
            let v = self.v.entry(param_name.clone()).or_insert_with(|| vec![0.0; grad.len()]);
            ensure_moment_len(v, grad.len(), "v", param_name)?;
            let v_max = if self.config.amsgrad {
                let entry =
                    self.v_max.entry(param_name.clone()).or_insert_with(|| vec![0.0; grad.len()]);
                ensure_moment_len(entry, grad.len(), "v_max", param_name)?;
                Some(entry)
            } else {
                None
            };

            let mut updates = Vec::with_capacity(grad.len());
            let mut v_max = v_max;

            for i in 0..grad.len() {
                // Update biased first moment estimate
                m[i] = self.config.beta1 as f32 * m[i] + (1.0 - self.config.beta1 as f32) * grad[i];

                // Update biased second raw moment estimate
                v[i] = self.config.beta2 as f32 * v[i]
                    + (1.0 - self.config.beta2 as f32) * grad[i] * grad[i];

                // Compute bias-corrected first moment estimate
                let m_hat = m[i] / (1.0 - (self.config.beta1 as f32).powi(self.step_count as i32));

                // Compute bias-corrected second raw moment estimate. Under
                // AMSGrad (Reddi et al., 2018) the denominator uses the
                // running *maximum* of `v_hat`'s numerator instead of the
                // current `v[i]`, which prevents the effective learning rate
                // from increasing late in training and fixes Adam's
                // non-convergence counterexample.
                let v_for_denom = if let Some(v_max) = v_max.as_deref_mut() {
                    v_max[i] = v_max[i].max(v[i]);
                    v_max[i]
                } else {
                    v[i]
                };
                let v_hat =
                    v_for_denom / (1.0 - (self.config.beta2 as f32).powi(self.step_count as i32));

                // AdamW-style decoupled weight decay: `weight_decay` shrinks
                // the parameter directly (scaled by the learning rate, as in
                // Loshchilov & Hutter, 2019), rather than being folded into
                // the gradient the way plain L2 regularization would be. It
                // is therefore added on top of the raw Adam update rather
                // than mixed into `grad[i]` above.
                let adam_update = -self.config.learning_rate as f32 * m_hat
                    / (v_hat.sqrt() + self.config.eps as f32);
                let decay_update =
                    -self.config.learning_rate as f32 * self.config.weight_decay as f32;
                updates.push(adam_update + decay_update);
            }

            parameter_updates.insert(param_name.clone(), updates);
        }

        Ok(OptimizerUpdate {
            parameter_updates,
            learning_rate: self.config.learning_rate,
            step_count: self.step_count,
        })
    }

    fn accumulate_gradients(&mut self, gradients: &OptimizerGradients) -> Result<()> {
        accumulate_into(&mut self.accumulated_gradients, gradients)
    }

    fn zero_grad(&mut self) {
        self.accumulated_gradients.clear();
    }

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn state_dict(&self) -> Result<HashMap<String, serde_json::Value>> {
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
        state.insert("m".to_string(), moment_map_to_json(&self.m, "m")?);
        state.insert("v".to_string(), moment_map_to_json(&self.v, "v")?);
        if self.config.amsgrad {
            state.insert(
                "v_max".to_string(),
                moment_map_to_json(&self.v_max, "v_max")?,
            );
        }
        Ok(state)
    }

    fn load_state_dict(&mut self, state: HashMap<String, serde_json::Value>) -> Result<()> {
        if let Some(step_count) = state.get("step_count").and_then(|v| v.as_u64()) {
            self.step_count = step_count as usize;
        }
        if let Some(lr) = state.get("learning_rate").and_then(|v| v.as_f64()) {
            self.config.learning_rate = lr;
        }
        self.m = moment_map_from_json(state.get("m"), "m")?;
        self.v = moment_map_from_json(state.get("v"), "v")?;
        self.v_max = moment_map_from_json(state.get("v_max"), "v_max")?;
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
    /// Per-parameter running maximum of `v`, used only when
    /// `config.amsgrad` is set (Reddi et al., 2018).
    v_max: HashMap<String, Vec<f32>>,
    /// Gradients accumulated via [`Optimizer::accumulate_gradients`] since
    /// the last [`Optimizer::zero_grad`]; folded into the next [`Optimizer::step`].
    accumulated_gradients: HashMap<String, Vec<f32>>,
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
            v_max: HashMap::new(),
            accumulated_gradients: HashMap::new(),
        }
    }
}

impl Optimizer for AdamOptimizer {
    fn step(&mut self, gradients: &OptimizerGradients) -> Result<OptimizerUpdate> {
        // Similar to AdamW but without weight decay
        let effective_gradients =
            merge_accumulated_gradients(gradients, &self.accumulated_gradients)?;
        self.step_count += 1;
        let mut parameter_updates = HashMap::new();

        for (param_name, grad) in &effective_gradients {
            // Initialize moment estimates if needed (entry API avoids a fallible lookup)
            let m = self.m.entry(param_name.clone()).or_insert_with(|| vec![0.0; grad.len()]);
            ensure_moment_len(m, grad.len(), "m", param_name)?;
            let v = self.v.entry(param_name.clone()).or_insert_with(|| vec![0.0; grad.len()]);
            ensure_moment_len(v, grad.len(), "v", param_name)?;
            let v_max = if self.config.amsgrad {
                let entry =
                    self.v_max.entry(param_name.clone()).or_insert_with(|| vec![0.0; grad.len()]);
                ensure_moment_len(entry, grad.len(), "v_max", param_name)?;
                Some(entry)
            } else {
                None
            };

            let mut updates = Vec::with_capacity(grad.len());
            let mut v_max = v_max;

            for i in 0..grad.len() {
                m[i] = self.config.beta1 as f32 * m[i] + (1.0 - self.config.beta1 as f32) * grad[i];
                v[i] = self.config.beta2 as f32 * v[i]
                    + (1.0 - self.config.beta2 as f32) * grad[i] * grad[i];

                let m_hat = m[i] / (1.0 - (self.config.beta1 as f32).powi(self.step_count as i32));
                // See `AdamWOptimizer::step` for why AMSGrad uses a running
                // maximum of `v` in the denominator instead of `v` itself.
                let v_for_denom = if let Some(v_max) = v_max.as_deref_mut() {
                    v_max[i] = v_max[i].max(v[i]);
                    v_max[i]
                } else {
                    v[i]
                };
                let v_hat =
                    v_for_denom / (1.0 - (self.config.beta2 as f32).powi(self.step_count as i32));

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

    fn accumulate_gradients(&mut self, gradients: &OptimizerGradients) -> Result<()> {
        accumulate_into(&mut self.accumulated_gradients, gradients)
    }

    fn zero_grad(&mut self) {
        self.accumulated_gradients.clear();
    }

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn state_dict(&self) -> Result<HashMap<String, serde_json::Value>> {
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
        state.insert("m".to_string(), moment_map_to_json(&self.m, "m")?);
        state.insert("v".to_string(), moment_map_to_json(&self.v, "v")?);
        if self.config.amsgrad {
            state.insert(
                "v_max".to_string(),
                moment_map_to_json(&self.v_max, "v_max")?,
            );
        }
        Ok(state)
    }

    fn load_state_dict(&mut self, state: HashMap<String, serde_json::Value>) -> Result<()> {
        if let Some(step_count) = state.get("step_count").and_then(|v| v.as_u64()) {
            self.step_count = step_count as usize;
        }
        if let Some(lr) = state.get("learning_rate").and_then(|v| v.as_f64()) {
            self.config.learning_rate = lr;
        }
        self.m = moment_map_from_json(state.get("m"), "m")?;
        self.v = moment_map_from_json(state.get("v"), "v")?;
        self.v_max = moment_map_from_json(state.get("v_max"), "v_max")?;
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

    fn accumulate_gradients(&mut self, gradients: &OptimizerGradients) -> Result<()> {
        self.optimizer.accumulate_gradients(gradients)
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

    fn state_dict(&self) -> Result<HashMap<String, serde_json::Value>> {
        let mut state = self.optimizer.state_dict()?;
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
        Ok(state)
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
mod tests;
