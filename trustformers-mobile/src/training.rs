//! On-Device Fine-Tuning for Mobile Deployment
//!
//! This module provides infrastructure for performing fine-tuning directly
//! on mobile devices with memory and compute constraints.

use crate::{MemoryOptimization, MobileConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::autodiff::{AutodiffEngine, Variable};
use trustformers_core::error::{CoreError, Result};
use trustformers_core::Tensor;
use trustformers_core::TrustformersError;

/// On-device training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnDeviceTrainingConfig {
    /// Learning rate for fine-tuning
    pub learning_rate: f32,
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size (usually 1 for mobile)
    pub batch_size: usize,
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Use gradient checkpointing to save memory
    pub gradient_checkpointing: bool,
    /// Fine-tuning method
    pub method: FineTuningMethod,
    /// Memory optimization for training
    pub memory_optimization: MemoryOptimization,
    /// Maximum memory for training (MB)
    pub max_training_memory_mb: usize,
}

/// Fine-tuning methods suitable for mobile devices
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FineTuningMethod {
    /// Low-Rank Adaptation (LoRA) - memory efficient
    LoRA { rank: usize, alpha: f32 },
    /// Adapter layers - lightweight fine-tuning
    Adapter { bottleneck_size: usize },
    /// Prefix tuning - only tune prefix tokens
    PrefixTuning { prefix_length: usize },
    /// Full fine-tuning (not recommended for mobile)
    Full,
}

impl Default for OnDeviceTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            epochs: 3,
            batch_size: 1,                  // Mobile-friendly
            gradient_accumulation_steps: 8, // Simulate larger batches
            max_sequence_length: 128,       // Conservative for mobile
            gradient_checkpointing: true,
            method: FineTuningMethod::LoRA {
                rank: 8,
                alpha: 16.0,
            },
            memory_optimization: MemoryOptimization::Maximum,
            max_training_memory_mb: 512, // 512MB limit (mobile-friendly)
        }
    }
}

/// On-device trainer for mobile fine-tuning
pub struct OnDeviceTrainer {
    config: OnDeviceTrainingConfig,
    mobile_config: MobileConfig,
    model_params: Option<HashMap<String, Tensor>>,
    trainable_params: HashMap<String, Tensor>,
    optimizer_state: OptimizerState,
    training_stats: OnDeviceTrainingStats,
}

impl OnDeviceTrainer {
    /// Create new on-device trainer
    pub fn new(config: OnDeviceTrainingConfig, mobile_config: MobileConfig) -> Result<Self> {
        // Validate training configuration for mobile constraints
        Self::validate_training_config(&config, &mobile_config)?;

        Ok(Self {
            config,
            mobile_config,
            model_params: None,
            trainable_params: HashMap::new(),
            optimizer_state: OptimizerState::new(),
            training_stats: OnDeviceTrainingStats::new(),
        })
    }

    /// Initialize training with base model parameters
    pub fn initialize_training(&mut self, base_params: HashMap<String, Tensor>) -> Result<()> {
        // Initialize trainable parameters based on fine-tuning method
        match self.config.method {
            FineTuningMethod::LoRA { rank, alpha } => {
                self.initialize_lora_params(&base_params, rank, alpha)?;
            },
            FineTuningMethod::Adapter { bottleneck_size } => {
                self.initialize_adapter_params(&base_params, bottleneck_size)?;
            },
            FineTuningMethod::PrefixTuning { prefix_length } => {
                self.initialize_prefix_params(&base_params, prefix_length)?;
            },
            FineTuningMethod::Full => {
                // Not recommended for mobile - clone all parameters
                self.trainable_params = base_params.clone();
            },
        }

        self.model_params = Some(base_params);

        // Estimate training memory requirements
        let memory_estimate = self.estimate_training_memory()?;
        if memory_estimate > self.config.max_training_memory_mb {
            return Err(TrustformersError::runtime_error(format!(
                "Training requires {}MB but limit is {}MB",
                memory_estimate, self.config.max_training_memory_mb
            ))
            .into());
        }

        tracing::info!(
            "On-device training initialized with {} trainable parameters",
            self.trainable_params.len()
        );
        tracing::info!("Estimated training memory: {}MB", memory_estimate);

        Ok(())
    }

    /// Perform one training step
    pub fn training_step(&mut self, input: &Tensor, target: &Tensor) -> Result<f32> {
        // Fresh engine (and so a fresh, empty computation graph) per step:
        // `forward_with_loss` builds a new graph from `self.trainable_params`
        // every call anyway (each `down`/`up` pair is re-wrapped in a new
        // `Variable` from its current tensor value), so reusing an engine
        // across steps would only accumulate dead nodes from every previous
        // step's graph with no benefit -- `AutodiffEngine::clear_graph`
        // exists for exactly this reason, and creating fresh is simpler
        // than remembering to call it.
        let engine = AutodiffEngine::default();

        // Forward pass with gradient computation
        let (loss_var, loss, param_vars) = self.forward_with_loss(&engine, input, target)?;

        // Backward pass (compute real gradients via autodiff)
        let gradients = self.backward_pass(&engine, &loss_var, &param_vars)?;

        // Update trainable parameters
        self.update_parameters(&gradients)?;

        // Update training statistics
        self.training_stats.update_step(loss);

        Ok(loss)
    }

    /// Train on a dataset for specified epochs
    pub fn train(&mut self, dataset: &[(Tensor, Tensor)]) -> Result<OnDeviceTrainingStats> {
        tracing::info!(
            "Starting on-device training for {} epochs",
            self.config.epochs
        );

        for epoch in 0..self.config.epochs {
            let mut epoch_loss = 0.0;
            let mut step_count = 0;

            // Process dataset in mini-batches
            for batch in dataset.chunks(self.config.batch_size) {
                let mut batch_loss = 0.0;

                // Gradient accumulation
                for step in 0..self.config.gradient_accumulation_steps.min(batch.len()) {
                    if step < batch.len() {
                        let (input, target) = &batch[step];
                        let step_loss = self.training_step(input, target)?;
                        batch_loss += step_loss;
                    }
                }

                epoch_loss += batch_loss;
                step_count += 1;

                // Log progress
                if step_count % 10 == 0 {
                    tracing::debug!(
                        "Epoch {}, Step {}, Loss: {:.4}",
                        epoch,
                        step_count,
                        batch_loss / self.config.gradient_accumulation_steps as f32
                    );
                }

                // Check memory usage
                if self.should_trigger_gc() {
                    self.mobile_gc()?;
                }
            }

            let avg_epoch_loss = epoch_loss / step_count as f32;
            self.training_stats.update_epoch(epoch, avg_epoch_loss);

            tracing::info!(
                "Epoch {} completed. Average loss: {:.4}",
                epoch,
                avg_epoch_loss
            );
        }

        tracing::info!("On-device training completed successfully");
        Ok(self.training_stats.clone())
    }

    /// Get trained parameters (only the fine-tuned ones)
    pub fn get_trained_parameters(&self) -> &HashMap<String, Tensor> {
        &self.trainable_params
    }

    /// Get training statistics
    pub fn get_training_stats(&self) -> &OnDeviceTrainingStats {
        &self.training_stats
    }

    /// Save training checkpoint for resuming
    pub fn save_checkpoint(&self) -> Result<OnDeviceCheckpoint> {
        Ok(OnDeviceCheckpoint {
            trainable_params: self.trainable_params.clone(),
            optimizer_state: self.optimizer_state.clone(),
            training_stats: self.training_stats.clone(),
            config: self.config.clone(),
        })
    }

    /// Load training checkpoint
    pub fn load_checkpoint(&mut self, checkpoint: OnDeviceCheckpoint) -> Result<()> {
        self.trainable_params = checkpoint.trainable_params;
        self.optimizer_state = checkpoint.optimizer_state;
        self.training_stats = checkpoint.training_stats;
        self.config = checkpoint.config;

        tracing::info!("Training checkpoint loaded successfully");
        Ok(())
    }

    // Private implementation methods

    fn validate_training_config(
        config: &OnDeviceTrainingConfig,
        mobile_config: &MobileConfig,
    ) -> Result<()> {
        // Check memory constraints
        if config.max_training_memory_mb > mobile_config.max_memory_mb {
            return Err(TrustformersError::config_error(
                "Training memory limit exceeds mobile memory limit",
                "mobile training validation",
            )
            .into());
        }

        // Check batch size is mobile-friendly
        if config.batch_size > 4 {
            return Err(TrustformersError::config_error(
                "Batch size too large for mobile training",
                "mobile training validation",
            )
            .into());
        }

        // Check sequence length is reasonable
        if config.max_sequence_length > 512 {
            return Err(TrustformersError::config_error(
                "Sequence length too long for mobile training",
                "mobile training validation",
            )
            .into());
        }

        Ok(())
    }

    fn initialize_lora_params(
        &mut self,
        base_params: &HashMap<String, Tensor>,
        rank: usize,
        alpha: f32,
    ) -> Result<()> {
        // Initialize LoRA parameters (A and B matrices)
        for (name, param) in base_params {
            if self.should_apply_lora(name) {
                let shape = param.shape();
                if shape.len() == 2 {
                    // For linear layers, create A and B matrices
                    let lora_a = Tensor::randn(&[shape[0], rank])?;
                    let lora_b = Tensor::zeros(&[rank, shape[1]])?; // Initialize B to zero

                    self.trainable_params.insert(format!("{}.lora_A", name), lora_a);
                    self.trainable_params.insert(format!("{}.lora_B", name), lora_b);
                }
            }
        }

        tracing::info!(
            "LoRA parameters initialized with rank {} and alpha {}",
            rank,
            alpha
        );
        Ok(())
    }

    fn initialize_adapter_params(
        &mut self,
        base_params: &HashMap<String, Tensor>,
        bottleneck_size: usize,
    ) -> Result<()> {
        // Initialize adapter layer parameters
        for (name, param) in base_params {
            if self.should_apply_adapter(name) {
                let shape = param.shape();
                if shape.len() == 2 {
                    // Create bottleneck adapter layers
                    let down_proj = Tensor::randn(&[shape[1], bottleneck_size])?;
                    let up_proj = Tensor::randn(&[bottleneck_size, shape[1]])?;

                    self.trainable_params.insert(format!("{}.adapter_down", name), down_proj);
                    self.trainable_params.insert(format!("{}.adapter_up", name), up_proj);
                }
            }
        }

        tracing::info!(
            "Adapter parameters initialized with bottleneck size {}",
            bottleneck_size
        );
        Ok(())
    }

    fn initialize_prefix_params(
        &mut self,
        base_params: &HashMap<String, Tensor>,
        prefix_length: usize,
    ) -> Result<()> {
        // Initialize prefix tuning parameters
        for (name, param) in base_params {
            if name.contains("embed") {
                let shape = param.shape();
                if shape.len() == 2 {
                    // Create prefix embeddings
                    let prefix_embed = Tensor::randn(&[prefix_length, shape[1]])?;
                    self.trainable_params.insert(format!("{}.prefix", name), prefix_embed);
                }
            }
        }

        tracing::info!(
            "Prefix tuning parameters initialized with prefix length {}",
            prefix_length
        );
        Ok(())
    }

    fn should_apply_lora(&self, param_name: &str) -> bool {
        // Apply LoRA to attention and MLP layers
        param_name.contains("attention")
            || param_name.contains("mlp")
            || param_name.contains("linear")
    }

    fn should_apply_adapter(&self, param_name: &str) -> bool {
        // Apply adapters to transformer layers
        param_name.contains("layer") && param_name.contains("linear")
    }

    fn estimate_training_memory(&self) -> Result<usize> {
        let mut total_memory = 0;

        // Model parameters memory
        if let Some(ref params) = self.model_params {
            for param in params.values() {
                total_memory += param.memory_usage();
            }
        }

        // Trainable parameters memory
        for param in self.trainable_params.values() {
            total_memory += param.memory_usage();
        }

        // Gradient memory (same as parameters)
        for param in self.trainable_params.values() {
            total_memory += param.memory_usage();
        }

        // Optimizer state memory (momentum, etc.)
        total_memory += total_memory / 2; // Estimate 50% overhead

        // Convert to MB
        Ok(total_memory / (1024 * 1024))
    }

    /// Real forward pass and loss, driven by `trustformers_core`'s autodiff
    /// engine ([`AutodiffEngine`]) rather than a placeholder identity/MSE
    /// stand-in. Returns the loss `Variable` (for [`Self::backward_pass`] to
    /// call `.backward()` on), the scalar loss value, and every trainable
    /// parameter's own `Variable` node in that same graph -- needed because
    /// [`AutodiffEngine::get_grad`] looks a gradient up by a `Variable`'s
    /// `node_id`; only the exact `Variable` object created *during this
    /// forward pass* has a `node_id` the backward pass actually populated a
    /// gradient for. A fresh `engine.variable(param.clone(), true)` built
    /// later (e.g. inside `backward_pass`, from `self.trainable_params`
    /// alone) would be a distinct, ungraphed leaf node -- always gradient
    /// `None` -- so the parameter `Variable`s must be threaded through
    /// rather than re-derived.
    ///
    /// # What this can and cannot compute
    ///
    /// `OnDeviceTrainer` has no base-model forward pass -- `model_params`
    /// exists only for the memory-estimation and (for `Full` fine-tuning)
    /// clone-through-as-trainable paths, never as an executable graph this
    /// trainer can run. Only [`FineTuningMethod::LoRA`] and
    /// [`FineTuningMethod::Adapter`] have a forward computation this
    /// function can perform *without* a base model: both are defined as a
    /// **standalone additive correction** to a frozen base layer's output,
    /// `delta = input @ down @ up` (LoRA: `down=A [in,r]`, `up=B [r,out]`;
    /// Adapter: `down=adapter_down [in,bottleneck]`,
    /// `up=adapter_up [bottleneck,in]`) -- so training these parameters
    /// against `(input, target)` pairs where `target` is the *desired
    /// total* correction is a real, self-contained supervised regression
    /// problem this function can and does execute end to end, with real
    /// gradients.
    ///
    /// [`FineTuningMethod::PrefixTuning`] (a prefix embedding has no
    /// `input @ prefix` composition -- it is prepended to a sequence,
    /// which requires the base model's attention mechanism to have any
    /// effect) and [`FineTuningMethod::Full`] (training "all parameters"
    /// requires the full base-model forward pass, which this trainer does
    /// not have) return a structured error instead of a fabricated
    /// forward pass. A previous revision returned `input.clone()` as the
    /// "output" and a constant `0.5` as the "loss" for every method,
    /// including these two -- which looked identical to a real, converged
    /// steady-state loss and gave no indication training had not actually
    /// happened.
    ///
    /// # Errors
    ///
    /// Returns an error if no trainable parameters are loaded, the method
    /// is `PrefixTuning`/`Full`, no LoRA/Adapter pair has a shape
    /// compatible with `input`, or any tensor operation fails (e.g. a
    /// shape mismatch between `input` and every loaded pair).
    fn forward_with_loss(
        &self,
        engine: &AutodiffEngine,
        input: &Tensor,
        target: &Tensor,
    ) -> Result<(Variable, f32, HashMap<String, Variable>)> {
        if self.trainable_params.is_empty() {
            return Err(TrustformersError::runtime_error(
                "no trainable parameters are loaded; call initialize_training first".to_string(),
            )
            .into());
        }

        match self.config.method {
            FineTuningMethod::LoRA { .. } => {
                self.forward_low_rank_loss(engine, input, target, "lora_A", "lora_B")
            },
            FineTuningMethod::Adapter { .. } => {
                self.forward_low_rank_loss(engine, input, target, "adapter_down", "adapter_up")
            },
            FineTuningMethod::PrefixTuning { .. } => Err(TrustformersError::runtime_error(
                "on-device training for PrefixTuning is not supported: a prefix embedding has \
                 no `input @ prefix` composition on its own -- it must be prepended to a \
                 sequence and run through the base model's attention layers, which this \
                 trainer does not have access to. Returning a fabricated forward pass here \
                 (as a previous revision did, via `input.clone()`) would silently corrupt the \
                 prefix embeddings with meaningless gradients."
                    .to_string(),
            )
            .into()),
            FineTuningMethod::Full => Err(TrustformersError::runtime_error(
                "on-device training for FineTuningMethod::Full is not supported: computing \
                 gradients for the full parameter set requires running the actual base-model \
                 forward pass, which this trainer does not have (only the parameter tensors \
                 themselves are held, not an executable model graph)."
                    .to_string(),
            )
            .into()),
        }
    }

    /// The shared forward computation for LoRA and Adapter fine-tuning:
    /// both are a pair of low-rank matrices named `<base>.{down_suffix}`
    /// and `<base>.{up_suffix}` computing `input @ down @ up`, trained by
    /// real backprop against `target` via mean-squared-error, using
    /// `trustformers_core`'s autodiff engine.
    ///
    /// When more than one `(down, up)` pair is loaded (multiple base
    /// layers each got their own LoRA/Adapter pair), every pair is applied
    /// to the *same* `input` and their outputs are summed before the loss
    /// -- consistent with each pair being an independent additive
    /// correction to a different point in the (unavailable) base model,
    /// collapsed here into the only signal this standalone trainer has.
    fn forward_low_rank_loss(
        &self,
        engine: &AutodiffEngine,
        input: &Tensor,
        target: &Tensor,
        down_suffix: &str,
        up_suffix: &str,
    ) -> Result<(Variable, f32, HashMap<String, Variable>)> {
        let pairs = self.find_low_rank_pairs(input.shape().as_slice(), down_suffix, up_suffix)?;

        let input_var = engine.variable(input.clone(), false);
        let target_var = engine.variable(target.clone(), false);

        let mut param_vars: HashMap<String, Variable> = HashMap::new();
        let mut output_var: Option<Variable> = None;
        for (down_name, up_name) in &pairs {
            let down = self.trainable_params.get(down_name).ok_or_else(|| {
                TrustformersError::runtime_error(format!(
                    "internal error: '{down_name}' was found during pairing but is missing from \
                     trainable_params"
                ))
            })?;
            let up = self.trainable_params.get(up_name).ok_or_else(|| {
                TrustformersError::runtime_error(format!(
                    "internal error: '{up_name}' was found during pairing but is missing from \
                     trainable_params"
                ))
            })?;

            let down_var = engine.variable(down.clone(), true);
            let up_var = engine.variable(up.clone(), true);
            let contribution = input_var.matmul(&down_var)?.matmul(&up_var)?;

            output_var = Some(match output_var {
                Some(acc) => acc.add(&contribution)?,
                None => contribution,
            });

            param_vars.insert(down_name.clone(), down_var);
            param_vars.insert(up_name.clone(), up_var);
        }

        // `pairs` is non-empty (checked inside `find_low_rank_pairs`), so
        // this always executes the loop body at least once.
        let output_var = output_var.ok_or_else(|| {
            TrustformersError::runtime_error(
                "internal error: find_low_rank_pairs returned an empty list without erroring"
                    .to_string(),
            )
        })?;

        // Mean squared error: mean((output - target)^2). Both `sub` and
        // `square`/`mean` are real autodiff operations with real gradient
        // functions (see `trustformers_core::autodiff::graph`), so
        // `.backward()` on this loss propagates real gradients back to
        // every `down_var`/`up_var` created above.
        let diff = output_var.sub(&target_var)?;
        let squared = diff.square()?;
        let loss_var = squared.mean(None)?;
        let loss_value = loss_var.item()?;

        Ok((loss_var, loss_value, param_vars))
    }

    /// Find every loaded `(<base>.{down_suffix}, <base>.{up_suffix})` pair
    /// whose composed shape (`input @ down @ up`) is valid for `input_shape`
    /// -- i.e. `down`'s first dimension matches `input_shape`'s last
    /// dimension, and `up`'s first dimension matches `down`'s second
    /// dimension. Pairs missing their other half (a `down` with no matching
    /// `up`, or vice versa -- which should not happen given how
    /// `initialize_lora_params`/`initialize_adapter_params` construct them,
    /// but a checkpoint loaded from disk could be malformed) are skipped
    /// with a warning rather than causing a panic or a fabricated shape.
    ///
    /// # Errors
    ///
    /// Returns an error if no compatible pair is found.
    fn find_low_rank_pairs(
        &self,
        input_shape: &[usize],
        down_suffix: &str,
        up_suffix: &str,
    ) -> Result<Vec<(String, String)>> {
        let Some(&last_dim) = input_shape.last() else {
            return Err(TrustformersError::shape_error(
                "input tensor has no dimensions".to_string(),
            )
            .into());
        };

        let mut pairs = Vec::new();
        let mut base_names: Vec<&str> = self
            .trainable_params
            .keys()
            .filter_map(|name| name.strip_suffix(&format!(".{down_suffix}")))
            .filter(|base| self.trainable_params.contains_key(&format!("{base}.{up_suffix}")))
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        // Deterministic order: `add`ing contributions in a different order
        // each run would make a real (if numerically tiny) difference to
        // floating-point rounding, and an arbitrary HashMap-derived order
        // is exactly the nondeterminism `inference.rs`'s
        // `ordered_weight_names` was introduced to eliminate elsewhere in
        // this crate.
        base_names.sort();

        for base in base_names {
            let down_name = format!("{base}.{down_suffix}");
            let up_name = format!("{base}.{up_suffix}");
            let down_shape = self.trainable_params[&down_name].shape();
            let up_shape = self.trainable_params[&up_name].shape();

            if down_shape.len() != 2 || up_shape.len() != 2 {
                tracing::warn!(
                    "skipping '{base}': expected 2D {down_suffix}/{up_suffix} matrices, got \
                     {down_shape:?}/{up_shape:?}"
                );
                continue;
            }
            if down_shape[0] != last_dim {
                continue;
            }
            if down_shape[1] != up_shape[0] {
                tracing::warn!(
                    "skipping '{base}': {down_suffix} output width {} does not match \
                     {up_suffix} input width {}",
                    down_shape[1],
                    up_shape[0]
                );
                continue;
            }
            pairs.push((down_name, up_name));
        }

        if pairs.is_empty() {
            return Err(TrustformersError::shape_error(format!(
                "no loaded {down_suffix}/{up_suffix} pair has an input dimension matching the \
                 given input's last dimension ({last_dim})"
            ))
            .into());
        }

        Ok(pairs)
    }

    /// Real backward pass: calls `loss.backward()` on the autodiff engine
    /// (reverse-mode automatic differentiation, computing a real gradient
    /// for every `Variable` node that fed into `loss_var`) and reads each
    /// trainable parameter's gradient back out via `param_vars` -- the
    /// *exact* `Variable` objects [`Self::forward_low_rank_loss`] created
    /// for each parameter during the forward pass, whose `node_id`s the
    /// backward pass just populated a gradient for.
    ///
    /// A parameter with `requires_grad == false` (none currently -- every
    /// `down`/`up` pair is created with `requires_grad: true` -- but this
    /// stays robust to that changing) or one the loss's computation never
    /// actually used has `AutodiffEngine::get_grad` return `Ok(None)`,
    /// which this function silently skips rather than treating as an
    /// error: `update_parameters` only updates parameters it has a
    /// gradient for, so an unused parameter is correctly left unchanged
    /// (not corrupted with a fabricated update).
    ///
    /// # Errors
    ///
    /// Propagates any error from the backward pass itself (e.g. a
    /// numerical issue the autodiff engine's anomaly detection flags).
    fn backward_pass(
        &self,
        engine: &AutodiffEngine,
        loss_var: &Variable,
        param_vars: &HashMap<String, Variable>,
    ) -> Result<HashMap<String, Tensor>> {
        engine.backward(loss_var, None)?;

        let mut gradients = HashMap::new();
        for (name, param_var) in param_vars {
            if let Some(grad) = engine.get_grad(param_var)? {
                gradients.insert(name.clone(), grad);
            }
        }

        Ok(gradients)
    }

    fn update_parameters(&mut self, gradients: &HashMap<String, Tensor>) -> Result<()> {
        // Adam-style update with momentum (first moment) and a fixed
        // second-moment-free step -- a real, stateful optimizer using
        // `OptimizerState.momentum`, which the previous implementation
        // declared but never read from or wrote to (a plain, memoryless
        // SGD step disguised as a struct with momentum state). This keeps
        // the field honest: it now actually accumulates and decays.
        const BETA: f32 = 0.9;
        self.optimizer_state.step_count += 1;

        for (name, grad) in gradients {
            let Some(param) = self.trainable_params.get_mut(name) else {
                continue;
            };

            let velocity = match self.optimizer_state.momentum.get(name) {
                Some(prev) => {
                    let decayed_prev = prev.scalar_mul(BETA)?;
                    let scaled_grad = grad.scalar_mul(1.0 - BETA)?;
                    decayed_prev.add(&scaled_grad)?
                },
                None => grad.clone(),
            };

            let step = velocity.scalar_mul(self.config.learning_rate)?;
            *param = param.sub(&step)?;
            self.optimizer_state.momentum.insert(name.clone(), velocity);
        }

        Ok(())
    }

    fn should_trigger_gc(&self) -> bool {
        // Trigger GC based on memory pressure (simplified)
        self.training_stats.current_step.is_multiple_of(50)
    }

    fn mobile_gc(&self) -> Result<()> {
        // Trigger mobile-specific garbage collection
        tracing::debug!("Triggering mobile garbage collection");
        Ok(())
    }
}

/// Optimizer state for on-device training
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OptimizerState {
    #[serde(skip)]
    momentum: HashMap<String, Tensor>,
    step_count: usize,
}

impl OptimizerState {
    fn new() -> Self {
        Self {
            momentum: HashMap::new(),
            step_count: 0,
        }
    }
}

/// Training statistics for on-device fine-tuning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnDeviceTrainingStats {
    /// Current training step
    pub current_step: usize,
    /// Current epoch
    pub current_epoch: usize,
    /// Average loss over all steps
    pub avg_loss: f32,
    /// Loss history per epoch
    pub epoch_losses: Vec<f32>,
    /// Total training time in seconds
    pub total_training_time_seconds: f32,
    /// Memory usage during training (MB)
    pub peak_memory_usage_mb: usize,
}

impl OnDeviceTrainingStats {
    fn new() -> Self {
        Self {
            current_step: 0,
            current_epoch: 0,
            avg_loss: 0.0,
            epoch_losses: Vec::new(),
            total_training_time_seconds: 0.0,
            peak_memory_usage_mb: 0,
        }
    }

    fn update_step(&mut self, loss: f32) {
        self.current_step += 1;

        // Update running average loss
        let alpha = 0.1;
        if self.current_step == 1 {
            self.avg_loss = loss;
        } else {
            self.avg_loss = alpha * loss + (1.0 - alpha) * self.avg_loss;
        }
    }

    fn update_epoch(&mut self, epoch: usize, epoch_loss: f32) {
        self.current_epoch = epoch;
        self.epoch_losses.push(epoch_loss);
    }
}

/// Training checkpoint for saving/loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnDeviceCheckpoint {
    #[serde(skip)]
    pub trainable_params: HashMap<String, Tensor>,
    pub optimizer_state: OptimizerState,
    pub training_stats: OnDeviceTrainingStats,
    pub config: OnDeviceTrainingConfig,
}

/// Mobile-specific training utilities
pub struct MobileTrainingUtils;

impl MobileTrainingUtils {
    /// Create optimized training config for mobile device
    pub fn create_mobile_training_config(
        available_memory_mb: usize,
        device_performance: MobilePerformanceLevel,
    ) -> OnDeviceTrainingConfig {
        match device_performance {
            MobilePerformanceLevel::Low => Self::low_end_config(available_memory_mb),
            MobilePerformanceLevel::Medium => Self::mid_range_config(available_memory_mb),
            MobilePerformanceLevel::High => Self::high_end_config(available_memory_mb),
        }
    }

    fn low_end_config(memory_mb: usize) -> OnDeviceTrainingConfig {
        OnDeviceTrainingConfig {
            learning_rate: 5e-5,
            epochs: 1, // Single epoch for speed
            batch_size: 1,
            gradient_accumulation_steps: 16, // Simulate larger batches
            max_sequence_length: 64,         // Very short sequences
            gradient_checkpointing: true,
            method: FineTuningMethod::LoRA {
                rank: 4,
                alpha: 8.0,
            }, // Small rank
            memory_optimization: MemoryOptimization::Maximum,
            max_training_memory_mb: (memory_mb / 4).max(128), // Very conservative
        }
    }

    fn mid_range_config(memory_mb: usize) -> OnDeviceTrainingConfig {
        OnDeviceTrainingConfig {
            learning_rate: 1e-4,
            epochs: 2,
            batch_size: 1,
            gradient_accumulation_steps: 8,
            max_sequence_length: 128,
            gradient_checkpointing: true,
            method: FineTuningMethod::LoRA {
                rank: 8,
                alpha: 16.0,
            },
            memory_optimization: MemoryOptimization::Balanced,
            max_training_memory_mb: (memory_mb / 2).max(256),
        }
    }

    fn high_end_config(memory_mb: usize) -> OnDeviceTrainingConfig {
        OnDeviceTrainingConfig {
            learning_rate: 2e-4,
            epochs: 3,
            batch_size: 2,
            gradient_accumulation_steps: 4,
            max_sequence_length: 256,
            gradient_checkpointing: false, // Can afford more memory
            method: FineTuningMethod::LoRA {
                rank: 16,
                alpha: 32.0,
            },
            memory_optimization: MemoryOptimization::Balanced,
            max_training_memory_mb: (memory_mb * 3 / 4).max(512),
        }
    }
}

/// Mobile device performance levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MobilePerformanceLevel {
    /// Low-end devices (< 2GB RAM)
    Low,
    /// Mid-range devices (2-6GB RAM)
    Medium,
    /// High-end devices (> 6GB RAM)
    High,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_on_device_training_config() {
        let config = OnDeviceTrainingConfig::default();
        assert_eq!(config.batch_size, 1);
        assert!(config.gradient_checkpointing);
        assert!(matches!(config.method, FineTuningMethod::LoRA { .. }));
    }

    #[test]
    fn test_on_device_trainer_creation() {
        let training_config = OnDeviceTrainingConfig::default();
        let mobile_config = crate::MobileConfig::default();

        let trainer = OnDeviceTrainer::new(training_config, mobile_config);
        assert!(trainer.is_ok());
    }

    #[test]
    fn test_training_initialization() {
        let training_config = OnDeviceTrainingConfig::default();
        let mobile_config = crate::MobileConfig::default();
        let mut trainer =
            OnDeviceTrainer::new(training_config, mobile_config).expect("operation failed in test");

        let mut base_params = HashMap::new();
        base_params.insert(
            "attention.linear".to_string(),
            Tensor::randn(&[128, 128]).expect("tensor operation failed"),
        );

        let result = trainer.initialize_training(base_params);
        assert!(result.is_ok());
        assert!(!trainer.trainable_params.is_empty());
    }

    #[test]
    fn test_mobile_training_utils() {
        let config = MobileTrainingUtils::create_mobile_training_config(
            2048,
            MobilePerformanceLevel::Medium,
        );
        assert_eq!(config.batch_size, 1);
        assert!(config.max_training_memory_mb <= 1024);

        let low_config =
            MobileTrainingUtils::create_mobile_training_config(1024, MobilePerformanceLevel::Low);
        assert_eq!(low_config.epochs, 1);
        assert_eq!(low_config.max_sequence_length, 64);
    }

    #[test]
    fn test_fine_tuning_methods() {
        let lora = FineTuningMethod::LoRA {
            rank: 8,
            alpha: 16.0,
        };
        let adapter = FineTuningMethod::Adapter {
            bottleneck_size: 64,
        };
        let prefix = FineTuningMethod::PrefixTuning { prefix_length: 16 };

        assert!(matches!(lora, FineTuningMethod::LoRA { .. }));
        assert!(matches!(adapter, FineTuningMethod::Adapter { .. }));
        assert!(matches!(prefix, FineTuningMethod::PrefixTuning { .. }));
    }

    #[test]
    fn test_training_stats() {
        let mut stats = OnDeviceTrainingStats::new();
        assert_eq!(stats.current_step, 0);
        assert_eq!(stats.avg_loss, 0.0);

        stats.update_step(1.0);
        assert_eq!(stats.current_step, 1);
        assert_eq!(stats.avg_loss, 1.0);

        stats.update_epoch(0, 0.8);
        assert_eq!(stats.epoch_losses.len(), 1);
        assert_eq!(stats.epoch_losses[0], 0.8);
    }

    // -- Regression tests: real forward/backward/update, not random
    // -- gradients against a constant fake loss.

    /// The direct regression test for the P0 finding: real training on a
    /// tiny, hand-solvable LoRA regression task must actually reduce the
    /// loss over repeated `training_step` calls. Against the previous
    /// implementation (`forward_with_loss` returning a constant `0.5` and
    /// `backward_pass` returning `Tensor::randn` gradients regardless of
    /// the actual loss), the loss trace would be flat noise around 0.5 --
    /// not a real, mostly-monotonic decrease -- and `update_parameters`
    /// applying those random "gradients" would actively degrade the LoRA
    /// weights step by step rather than fit them to `target`.
    #[test]
    fn test_training_step_reduces_loss_on_tiny_lora_task() {
        let training_config = OnDeviceTrainingConfig {
            learning_rate: 0.1,
            method: FineTuningMethod::LoRA {
                rank: 2,
                alpha: 4.0,
            },
            ..OnDeviceTrainingConfig::default()
        };
        let mobile_config = crate::MobileConfig::default();
        let mut trainer = OnDeviceTrainer::new(training_config, mobile_config).expect("trainer");

        let mut base_params = HashMap::new();
        // "linear" makes `should_apply_lora` select this parameter; its
        // shape ([4, 4]) is otherwise irrelevant to LoRA init (only the
        // input dimension, 4, is used to size lora_A).
        base_params.insert(
            "block.linear".to_string(),
            Tensor::zeros(&[4, 4]).expect("param"),
        );
        trainer.initialize_training(base_params).expect("initialize_training");

        // A fixed, tiny supervised task: drive the LoRA delta toward a
        // known target vector for a fixed input. `lora_B` is
        // zero-initialized (see `initialize_lora_params`), so the initial
        // output is exactly zero and the initial loss is
        // `mean(target^2)` -- deterministic, so this is a real, not
        // cherry-picked, check.
        let input = Tensor::from_vec(vec![1.0, 0.5, -0.5, 1.0], &[1, 4]).expect("input");
        let target = Tensor::from_vec(vec![2.0, -1.0, 0.5, 1.5], &[1, 4]).expect("target");

        let first_loss = trainer.training_step(&input, &target).expect("training_step 1");
        assert!(first_loss.is_finite());
        assert!(
            first_loss > 0.0,
            "the target is nonzero and the initial output is zero, so the \
            initial MSE loss must be positive, got {first_loss}"
        );

        let mut last_loss = first_loss;
        for step in 2..=20 {
            let loss = trainer
                .training_step(&input, &target)
                .unwrap_or_else(|e| panic!("training_step {step} failed: {e}"));
            assert!(
                loss.is_finite(),
                "loss became non-finite at step {step}: {loss}"
            );
            last_loss = loss;
        }

        assert!(
            last_loss < first_loss * 0.5,
            "20 real gradient-descent steps at lr=0.1 on a fixed tiny target must substantially \
             reduce the loss (from {first_loss} to well under half that); got {last_loss}, which \
             is what a constant-loss/random-gradient implementation would produce instead"
        );
    }

    /// `OptimizerState.momentum` must actually be populated by training --
    /// the previous implementation declared the field but never wrote to
    /// it (plain memoryless SGD masquerading as a stateful optimizer).
    #[test]
    fn test_training_step_populates_optimizer_momentum() {
        let training_config = OnDeviceTrainingConfig {
            method: FineTuningMethod::LoRA {
                rank: 2,
                alpha: 4.0,
            },
            ..OnDeviceTrainingConfig::default()
        };
        let mobile_config = crate::MobileConfig::default();
        let mut trainer = OnDeviceTrainer::new(training_config, mobile_config).expect("trainer");

        let mut base_params = HashMap::new();
        base_params.insert(
            "block.linear".to_string(),
            Tensor::zeros(&[4, 4]).expect("param"),
        );
        trainer.initialize_training(base_params).expect("initialize_training");

        assert!(
            trainer.optimizer_state.momentum.is_empty(),
            "precondition: no steps taken yet"
        );

        let input = Tensor::from_vec(vec![1.0, 0.5, -0.5, 1.0], &[1, 4]).expect("input");
        let target = Tensor::from_vec(vec![2.0, -1.0, 0.5, 1.5], &[1, 4]).expect("target");
        trainer.training_step(&input, &target).expect("training_step");

        assert!(
            !trainer.optimizer_state.momentum.is_empty(),
            "a real optimizer step must record per-parameter momentum state"
        );
        assert_eq!(trainer.optimizer_state.step_count, 1);
    }

    /// `PrefixTuning` has no `input @ prefix` forward composition available
    /// to this standalone trainer and must be a structured error, not a
    /// silent `input.clone()` identity pass reporting a fake loss.
    #[test]
    fn test_training_step_rejects_prefix_tuning() {
        let training_config = OnDeviceTrainingConfig {
            method: FineTuningMethod::PrefixTuning { prefix_length: 4 },
            ..OnDeviceTrainingConfig::default()
        };
        let mobile_config = crate::MobileConfig::default();
        let mut trainer = OnDeviceTrainer::new(training_config, mobile_config).expect("trainer");

        let mut base_params = HashMap::new();
        base_params.insert(
            "token.embed".to_string(),
            Tensor::zeros(&[8, 4]).expect("param"),
        );
        trainer.initialize_training(base_params).expect("initialize_training");

        let input = Tensor::from_vec(vec![1.0, 0.5, -0.5, 1.0], &[1, 4]).expect("input");
        let target = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[1, 4]).expect("target");
        let result = trainer.training_step(&input, &target);

        assert!(
            result.is_err(),
            "PrefixTuning must refuse rather than fabricate a forward pass"
        );
    }

    /// `Full` fine-tuning has no base-model forward pass available to this
    /// standalone trainer and must be a structured error.
    #[test]
    fn test_training_step_rejects_full_fine_tuning() {
        let training_config = OnDeviceTrainingConfig {
            method: FineTuningMethod::Full,
            ..OnDeviceTrainingConfig::default()
        };
        let mobile_config = crate::MobileConfig::default();
        let mut trainer = OnDeviceTrainer::new(training_config, mobile_config).expect("trainer");

        let mut base_params = HashMap::new();
        base_params.insert(
            "any.weight".to_string(),
            Tensor::zeros(&[4, 4]).expect("param"),
        );
        trainer.initialize_training(base_params).expect("initialize_training");

        let input = Tensor::from_vec(vec![1.0, 0.5, -0.5, 1.0], &[1, 4]).expect("input");
        let target = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[1, 4]).expect("target");
        let result = trainer.training_step(&input, &target);

        assert!(
            result.is_err(),
            "Full fine-tuning must refuse rather than fabricate a forward pass"
        );
    }

    /// `training_step` on a trainer with no initialized parameters must
    /// error, not silently report a fake loss.
    #[test]
    fn test_training_step_errors_without_initialization() {
        let training_config = OnDeviceTrainingConfig::default();
        let mobile_config = crate::MobileConfig::default();
        let mut trainer = OnDeviceTrainer::new(training_config, mobile_config).expect("trainer");

        let input = Tensor::from_vec(vec![1.0, 0.5, -0.5, 1.0], &[1, 4]).expect("input");
        let target = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[1, 4]).expect("target");
        let result = trainer.training_step(&input, &target);

        assert!(
            result.is_err(),
            "training with no initialized parameters must error"
        );
    }

    /// The full `train()` loop (multiple epochs, gradient accumulation)
    /// over a tiny fixed dataset must also show real loss reduction from
    /// the first epoch to the last -- an end-to-end check that `train()`'s
    /// batching/accumulation wiring around `training_step` does not lose
    /// the real gradient signal.
    #[test]
    fn test_train_reduces_loss_across_epochs() {
        let training_config = OnDeviceTrainingConfig {
            learning_rate: 0.1,
            epochs: 5,
            batch_size: 1,
            gradient_accumulation_steps: 1,
            method: FineTuningMethod::LoRA {
                rank: 2,
                alpha: 4.0,
            },
            ..OnDeviceTrainingConfig::default()
        };
        let mobile_config = crate::MobileConfig::default();
        let mut trainer = OnDeviceTrainer::new(training_config, mobile_config).expect("trainer");

        let mut base_params = HashMap::new();
        base_params.insert(
            "block.linear".to_string(),
            Tensor::zeros(&[4, 4]).expect("param"),
        );
        trainer.initialize_training(base_params).expect("initialize_training");

        let input = Tensor::from_vec(vec![1.0, 0.5, -0.5, 1.0], &[1, 4]).expect("input");
        let target = Tensor::from_vec(vec![2.0, -1.0, 0.5, 1.5], &[1, 4]).expect("target");
        let dataset = vec![(input, target)];

        let stats = trainer.train(&dataset).expect("train");

        assert_eq!(stats.epoch_losses.len(), 5);
        let first_epoch_loss = stats.epoch_losses[0];
        let last_epoch_loss = stats.epoch_losses[4];
        assert!(
            last_epoch_loss < first_epoch_loss,
            "5 epochs of real training on a fixed tiny task must reduce the average epoch loss: \
             first={first_epoch_loss}, last={last_epoch_loss}"
        );
    }
}
