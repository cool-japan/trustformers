//! Optimizer Fusion Techniques
//!
//! This module provides advanced optimizer fusion techniques for performance optimization.
//! It combines multiple optimizer operations into fused kernels to reduce memory bandwidth
//! and improve overall training performance.

// reason: research-stage module — reserved API/scaffolding fields and methods
// retained intentionally for in-progress features; not yet on active call paths.
#![allow(dead_code)]

use crate::OptimizerState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::Tensor;

/// Fused optimizer operations for performance optimization
#[derive(Debug, Clone)]
pub enum FusedOperation {
    /// Fused Adam update (parameter, gradient, momentum, velocity)
    FusedAdam {
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    },
    /// Fused AdamW update with decoupled weight decay
    FusedAdamW {
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    },
    /// Fused SGD with momentum
    FusedSGDMomentum {
        lr: f64,
        momentum: f64,
        dampening: f64,
        weight_decay: f64,
        nesterov: bool,
    },
    /// Fused gradient clipping and scaling
    FusedGradientClipping { max_norm: f64, scale_factor: f64 },
    /// Fused batch normalization update
    FusedBatchNorm { eps: f64, momentum: f64 },
}

/// Configuration for fused optimizer operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    /// Enable memory bandwidth optimization
    pub enable_memory_coalescing: bool,
    /// Use vectorized operations when possible
    pub enable_vectorization: bool,
    /// Batch size for parameter updates
    pub batch_size: usize,
    /// Enable kernel fusion for compatible operations
    pub enable_kernel_fusion: bool,
    /// Buffer size for batched operations
    pub buffer_size: usize,
    /// Enable asynchronous updates
    pub enable_async_updates: bool,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            enable_memory_coalescing: true,
            enable_vectorization: true,
            batch_size: 64,
            enable_kernel_fusion: true,
            buffer_size: 1024,
            enable_async_updates: false,
        }
    }
}

/// Fused optimizer state for multiple parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedOptimizerState {
    /// Parameter states indexed by parameter name
    pub parameter_states: HashMap<String, OptimizerState>,
    /// Fused operation buffers
    pub operation_buffers: HashMap<String, Vec<f64>>,
    /// Fusion statistics
    pub fusion_stats: FusionStats,
}

/// Statistics for fusion operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionStats {
    /// Number of fused operations executed
    pub fused_operations: u64,
    /// Memory bandwidth saved (bytes)
    pub memory_bandwidth_saved: u64,
    /// FLOPS saved through fusion
    pub flops_saved: u64,
    /// Average batch size
    pub avg_batch_size: f64,
    /// Fusion efficiency ratio
    pub fusion_efficiency: f64,
}

impl Default for FusionStats {
    fn default() -> Self {
        Self {
            fused_operations: 0,
            memory_bandwidth_saved: 0,
            flops_saved: 0,
            avg_batch_size: 0.0,
            fusion_efficiency: 0.0,
        }
    }
}

/// Fused optimizer that combines multiple optimization operations.
///
/// # Result retrieval contract
///
/// [`queue_operation`](FusedOptimizer::queue_operation) takes **owned** parameter and
/// gradient tensors, so the caller's own bindings are never mutated. The updated
/// parameters (and, for [`FusedOperation::FusedGradientClipping`], the clipped
/// gradients) are stored internally keyed by the `param_name` supplied at queue time and
/// must be collected with [`take_updated_parameters`](FusedOptimizer::take_updated_parameters)
/// / [`take_clipped_gradients`](FusedOptimizer::take_clipped_gradients) after
/// [`flush`](FusedOptimizer::flush).
///
/// Callers that want in-place semantics should use
/// [`apply_in_place`](FusedOptimizer::apply_in_place), which updates a `&mut Tensor`
/// directly and bypasses the batching queue.
#[derive(Debug)]
pub struct FusedOptimizer {
    config: FusionConfig,
    state: Arc<Mutex<FusedOptimizerState>>,
    pending_operations: Arc<Mutex<Vec<(String, FusedOperation, Tensor, Tensor)>>>,
    operation_queue: Arc<Mutex<HashMap<String, Vec<FusedOperation>>>>,
    /// Parameters updated by the most recent batch executions, keyed by `param_name`.
    updated_parameters: Arc<Mutex<HashMap<String, Tensor>>>,
    /// Gradients clipped by the most recent batch executions, keyed by `param_name`.
    clipped_gradients: Arc<Mutex<HashMap<String, Tensor>>>,
}

impl FusedOptimizer {
    /// Create new fused optimizer
    pub fn new(config: FusionConfig) -> Result<Self> {
        let state = FusedOptimizerState {
            parameter_states: HashMap::new(),
            operation_buffers: HashMap::new(),
            fusion_stats: FusionStats::default(),
        };

        Ok(Self {
            config,
            state: Arc::new(Mutex::new(state)),
            pending_operations: Arc::new(Mutex::new(Vec::new())),
            operation_queue: Arc::new(Mutex::new(HashMap::new())),
            updated_parameters: Arc::new(Mutex::new(HashMap::new())),
            clipped_gradients: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Record an updated parameter tensor so the caller can retrieve it after `flush()`.
    fn record_updated_parameter(&self, param_name: String, param: Tensor) -> Result<()> {
        let mut updated = self.updated_parameters.lock().map_err(|_| {
            TrustformersError::lock_error("fusion updated-parameter mutex poisoned".to_string())
        })?;
        updated.insert(param_name, param);
        Ok(())
    }

    /// Takes (and clears) the parameters updated by previously executed batches.
    pub fn take_updated_parameters(&mut self) -> Result<HashMap<String, Tensor>> {
        let mut updated = self.updated_parameters.lock().map_err(|_| {
            TrustformersError::lock_error("fusion updated-parameter mutex poisoned".to_string())
        })?;
        Ok(std::mem::take(&mut *updated))
    }

    /// Returns a clone of the most recent updated value for `param_name`, if any.
    pub fn updated_parameter(&self, param_name: &str) -> Result<Option<Tensor>> {
        let updated = self.updated_parameters.lock().map_err(|_| {
            TrustformersError::lock_error("fusion updated-parameter mutex poisoned".to_string())
        })?;
        Ok(updated.get(param_name).cloned())
    }

    /// Takes (and clears) the gradients clipped by previously executed batches.
    pub fn take_clipped_gradients(&mut self) -> Result<HashMap<String, Tensor>> {
        let mut clipped = self.clipped_gradients.lock().map_err(|_| {
            TrustformersError::lock_error("fusion clipped-gradient mutex poisoned".to_string())
        })?;
        Ok(std::mem::take(&mut *clipped))
    }

    /// Applies a single fused operation directly to a caller-owned parameter tensor.
    ///
    /// Unlike [`queue_operation`](FusedOptimizer::queue_operation) this bypasses the
    /// batching queue and mutates `parameter` (or, for gradient clipping, `gradient`)
    /// in place, so no result retrieval step is needed.
    pub fn apply_in_place(
        &mut self,
        param_name: &str,
        operation: FusedOperation,
        parameter: &mut Tensor,
        gradient: &mut Tensor,
    ) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| TrustformersError::lock_error("fusion mutex poisoned".to_string()))?;
        let opt_state = state.parameter_states.entry(param_name.to_string()).or_insert_with(|| {
            OptimizerState {
                step: 0,
                momentum: HashMap::new(),
                variance: HashMap::new(),
                ..Default::default()
            }
        });

        match operation {
            FusedOperation::FusedAdam {
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
            } => Self::fused_adam_update(
                param_name,
                parameter,
                gradient,
                opt_state,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
            ),
            FusedOperation::FusedAdamW {
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
            } => Self::fused_adamw_update(
                param_name,
                parameter,
                gradient,
                opt_state,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
            ),
            FusedOperation::FusedSGDMomentum {
                lr,
                momentum,
                dampening,
                weight_decay,
                nesterov,
            } => Self::fused_sgd_update(
                param_name,
                parameter,
                gradient,
                opt_state,
                lr,
                momentum,
                dampening,
                weight_decay,
                nesterov,
            ),
            FusedOperation::FusedGradientClipping {
                max_norm,
                scale_factor,
            } => {
                let norm = gradient.norm()? as f64;
                let scale = if norm > max_norm && norm > 0.0 {
                    (max_norm / norm) * scale_factor
                } else {
                    scale_factor
                };
                Self::scale_tensor_in_place(gradient, scale as f32)
            },
            FusedOperation::FusedBatchNorm { .. } => Err(TrustformersError::not_implemented(
                "FusedOperation::FusedBatchNorm has no fused implementation".to_string(),
            )),
        }
    }

    /// Multiplies every element of `tensor` by `scale`, in place.
    fn scale_tensor_in_place(tensor: &mut Tensor, scale: f32) -> Result<()> {
        let mut data = tensor.data()?;
        for value in data.iter_mut() {
            *value *= scale;
        }
        tensor.set_data_f32(&data)
    }

    /// Add operation to fusion queue
    pub fn queue_operation(
        &mut self,
        param_name: String,
        operation: FusedOperation,
        parameter: Tensor,
        gradient: Tensor,
    ) -> Result<()> {
        let should_execute = {
            let mut pending = self
                .pending_operations
                .lock()
                .map_err(|_| TrustformersError::lock_error("fusion mutex poisoned".to_string()))?;
            pending.push((param_name, operation, parameter, gradient));
            pending.len() >= self.config.batch_size
        };

        // Execute batch if buffer is full
        if should_execute {
            self.execute_fused_batch()?;
        }

        Ok(())
    }

    /// Execute all pending operations in a fused manner
    pub fn execute_fused_batch(&mut self) -> Result<()> {
        let mut pending = self
            .pending_operations
            .lock()
            .map_err(|_| TrustformersError::lock_error("fusion mutex poisoned".to_string()))?;
        if pending.is_empty() {
            return Ok(());
        }

        let operations = std::mem::take(&mut *pending);
        drop(pending);

        // Group operations by type for maximum fusion efficiency
        let mut adam_ops = Vec::new();
        let mut adamw_ops = Vec::new();
        let mut sgd_ops = Vec::new();
        let mut clip_ops = Vec::new();

        for (param_name, op, param, grad) in operations {
            match op {
                FusedOperation::FusedAdam { .. } => adam_ops.push((param_name, op, param, grad)),
                FusedOperation::FusedAdamW { .. } => adamw_ops.push((param_name, op, param, grad)),
                FusedOperation::FusedSGDMomentum { .. } => {
                    sgd_ops.push((param_name, op, param, grad))
                },
                FusedOperation::FusedGradientClipping { .. } => {
                    clip_ops.push((param_name, op, param, grad))
                },
                _ => {
                    // Handle other operations individually
                    self.execute_single_operation(param_name, op, param, grad)?;
                },
            }
        }

        // Execute fused batches
        if !adam_ops.is_empty() {
            self.execute_fused_adam_batch(adam_ops)?;
        }
        if !adamw_ops.is_empty() {
            self.execute_fused_adamw_batch(adamw_ops)?;
        }
        if !sgd_ops.is_empty() {
            self.execute_fused_sgd_batch(sgd_ops)?;
        }
        if !clip_ops.is_empty() {
            self.execute_fused_clipping_batch(clip_ops)?;
        }

        Ok(())
    }

    /// Execute fused Adam operations
    fn execute_fused_adam_batch(
        &mut self,
        operations: Vec<(String, FusedOperation, Tensor, Tensor)>,
    ) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| TrustformersError::lock_error("fusion mutex poisoned".to_string()))?;
        let batch_size = operations.len();

        for (param_name, op, mut param, grad) in operations {
            if let FusedOperation::FusedAdam {
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
            } = op
            {
                // Get or create optimizer state
                let opt_state =
                    state.parameter_states.entry(param_name.clone()).or_insert_with(|| {
                        OptimizerState {
                            step: 0,
                            momentum: HashMap::new(),
                            variance: HashMap::new(),
                            ..Default::default()
                        }
                    });

                // Fused Adam update with optimized memory access
                Self::fused_adam_update(
                    &param_name,
                    &mut param,
                    &grad,
                    opt_state,
                    lr,
                    beta1,
                    beta2,
                    eps,
                    weight_decay,
                )?;
                self.record_updated_parameter(param_name, param)?;
            }
        }

        // Update fusion statistics
        state.fusion_stats.fused_operations += 1;
        state.fusion_stats.avg_batch_size = (state.fusion_stats.avg_batch_size
            * (state.fusion_stats.fused_operations - 1) as f64
            + batch_size as f64)
            / state.fusion_stats.fused_operations as f64;

        // Estimate memory bandwidth savings (simplified)
        let bandwidth_saved = batch_size * 4 * 8; // 4 tensors * 8 bytes per element (approximate)
        state.fusion_stats.memory_bandwidth_saved += bandwidth_saved as u64;

        Ok(())
    }

    /// Execute fused AdamW operations
    fn execute_fused_adamw_batch(
        &mut self,
        operations: Vec<(String, FusedOperation, Tensor, Tensor)>,
    ) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| TrustformersError::lock_error("fusion mutex poisoned".to_string()))?;
        let batch_size = operations.len();

        for (param_name, op, mut param, grad) in operations {
            if let FusedOperation::FusedAdamW {
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
            } = op
            {
                let opt_state =
                    state.parameter_states.entry(param_name.clone()).or_insert_with(|| {
                        OptimizerState {
                            step: 0,
                            momentum: HashMap::new(),
                            variance: HashMap::new(),
                            ..Default::default()
                        }
                    });

                // Fused AdamW update with decoupled weight decay
                Self::fused_adamw_update(
                    &param_name,
                    &mut param,
                    &grad,
                    opt_state,
                    lr,
                    beta1,
                    beta2,
                    eps,
                    weight_decay,
                )?;
                self.record_updated_parameter(param_name, param)?;
            }
        }

        // Update statistics
        state.fusion_stats.fused_operations += 1;
        let bandwidth_saved = batch_size * 4 * 8;
        state.fusion_stats.memory_bandwidth_saved += bandwidth_saved as u64;

        Ok(())
    }

    /// Execute fused SGD operations
    fn execute_fused_sgd_batch(
        &mut self,
        operations: Vec<(String, FusedOperation, Tensor, Tensor)>,
    ) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| TrustformersError::lock_error("fusion mutex poisoned".to_string()))?;
        let batch_size = operations.len();

        for (param_name, op, mut param, grad) in operations {
            if let FusedOperation::FusedSGDMomentum {
                lr,
                momentum,
                dampening,
                weight_decay,
                nesterov,
            } = op
            {
                let opt_state =
                    state.parameter_states.entry(param_name.clone()).or_insert_with(|| {
                        OptimizerState {
                            step: 0,
                            momentum: HashMap::new(),
                            ..Default::default()
                        }
                    });

                // Fused SGD with momentum update
                Self::fused_sgd_update(
                    &param_name,
                    &mut param,
                    &grad,
                    opt_state,
                    lr,
                    momentum,
                    dampening,
                    weight_decay,
                    nesterov,
                )?;
                self.record_updated_parameter(param_name, param)?;
            }
        }

        // Update statistics
        state.fusion_stats.fused_operations += 1;
        let bandwidth_saved = batch_size * 2 * 8; // SGD uses fewer tensors
        state.fusion_stats.memory_bandwidth_saved += bandwidth_saved as u64;

        Ok(())
    }

    /// Execute fused gradient clipping operations
    fn execute_fused_clipping_batch(
        &mut self,
        operations: Vec<(String, FusedOperation, Tensor, Tensor)>,
    ) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| TrustformersError::lock_error("fusion mutex poisoned".to_string()))?;
        let batch_size = operations.len();

        // Collect all gradients for global norm computation
        let mut gradients = Vec::new();
        for (_, _, _, grad) in &operations {
            gradients.push(grad.clone());
        }

        // Compute global gradient norm for batch
        let global_norm = self.compute_global_norm(&gradients)?;

        // Scale factor shared by the whole batch: clip only when the *global* norm
        // exceeds `max_norm`, matching `torch.nn.utils.clip_grad_norm_` semantics.
        for (param_name, op, _, mut grad) in operations {
            if let FusedOperation::FusedGradientClipping {
                max_norm,
                scale_factor,
            } = op
            {
                let scale = if global_norm > max_norm && global_norm > 0.0 {
                    (max_norm / global_norm) * scale_factor
                } else {
                    scale_factor
                };
                Self::scale_tensor_in_place(&mut grad, scale as f32)?;

                let mut clipped = self.clipped_gradients.lock().map_err(|_| {
                    TrustformersError::lock_error(
                        "fusion clipped-gradient mutex poisoned".to_string(),
                    )
                })?;
                clipped.insert(param_name, grad);
            }
        }

        // Update statistics
        state.fusion_stats.fused_operations += 1;
        let bandwidth_saved = batch_size * 8; // Single pass through gradients
        state.fusion_stats.memory_bandwidth_saved += bandwidth_saved as u64;

        Ok(())
    }

    /// Execute single operation (fallback for non-batchable operations)
    fn execute_single_operation(
        &mut self,
        _param_name: String,
        _operation: FusedOperation,
        _parameter: Tensor,
        _gradient: Tensor,
    ) -> Result<()> {
        // Implementation for individual operations
        Ok(())
    }

    /// Optimized Adam update with fused operations.
    ///
    /// Writes the updated values back into `param` — the per-parameter momentum and
    /// variance buffers live in `state`, keyed by the caller-supplied `param_name`.
    fn fused_adam_update(
        param_name: &str,
        param: &mut Tensor,
        grad: &Tensor,
        state: &mut OptimizerState,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    ) -> Result<()> {
        state.step += 1;
        let param_id = param_name.to_string();
        let param_len = param.data()?.len();

        // Get or initialize momentum and variance buffers
        let momentum =
            state.momentum.entry(param_id.clone()).or_insert_with(|| vec![0.0; param_len]);
        let variance = state.variance.entry(param_id).or_insert_with(|| vec![0.0; param_len]);

        let grad_data = grad.data()?;
        let mut param_data = param.data()?;

        // Bias correction factors
        let bias_correction1 = 1.0 - beta1.powi(state.step as i32);
        let bias_correction2 = 1.0 - beta2.powi(state.step as i32);

        // Fused update loop - combines all operations in single pass
        for i in 0..param_data.len() {
            let mut grad_val = grad_data[i];

            // Apply weight decay if specified (L2 regularization)
            if weight_decay > 0.0 {
                grad_val += weight_decay as f32 * param_data[i];
            }

            // Update biased first moment estimate (momentum)
            momentum[i] = beta1 as f32 * momentum[i] + (1.0 - beta1 as f32) * grad_val;

            // Update biased second raw moment estimate (variance)
            variance[i] = beta2 as f32 * variance[i] + (1.0 - beta2 as f32) * grad_val * grad_val;

            // Compute bias-corrected first and second moment estimates
            let m_hat = momentum[i] / bias_correction1 as f32;
            let v_hat = variance[i] / bias_correction2 as f32;

            // Update parameter with fused Adam step
            param_data[i] -= lr as f32 * m_hat / (v_hat.sqrt() + eps as f32);
        }

        param.set_data_f32(&param_data)
    }

    /// Optimized AdamW update with fused operations and decoupled weight decay.
    ///
    /// Writes the updated values back into `param`.
    fn fused_adamw_update(
        param_name: &str,
        param: &mut Tensor,
        grad: &Tensor,
        state: &mut OptimizerState,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    ) -> Result<()> {
        state.step += 1;
        let param_id = param_name.to_string();
        let param_len = param.data()?.len();

        // Get or initialize momentum and variance buffers
        let momentum =
            state.momentum.entry(param_id.clone()).or_insert_with(|| vec![0.0; param_len]);
        let variance = state.variance.entry(param_id).or_insert_with(|| vec![0.0; param_len]);

        let grad_data = grad.data()?;
        let mut param_data = param.data()?;

        // Bias correction factors
        let bias_correction1 = 1.0 - beta1.powi(state.step as i32);
        let bias_correction2 = 1.0 - beta2.powi(state.step as i32);

        // Fused AdamW update loop - decoupled weight decay
        for i in 0..param_data.len() {
            let grad_val = grad_data[i];

            // Update biased first moment estimate (momentum)
            momentum[i] = beta1 as f32 * momentum[i] + (1.0 - beta1 as f32) * grad_val;

            // Update biased second raw moment estimate (variance)
            variance[i] = beta2 as f32 * variance[i] + (1.0 - beta2 as f32) * grad_val * grad_val;

            // Compute bias-corrected first and second moment estimates
            let m_hat = momentum[i] / bias_correction1 as f32;
            let v_hat = variance[i] / bias_correction2 as f32;

            // AdamW update: apply weight decay directly to parameters (decoupled)
            let adaptive_step = lr as f32 * m_hat / (v_hat.sqrt() + eps as f32);
            let weight_decay_step = lr as f32 * weight_decay as f32 * param_data[i];

            // Combined update with decoupled weight decay
            param_data[i] -= adaptive_step + weight_decay_step;
        }

        param.set_data_f32(&param_data)
    }

    /// Optimized SGD update with fused momentum.
    ///
    /// Writes the updated values back into `param`.
    fn fused_sgd_update(
        param_name: &str,
        param: &mut Tensor,
        grad: &Tensor,
        state: &mut OptimizerState,
        lr: f64,
        momentum_coef: f64,
        dampening: f64,
        weight_decay: f64,
        nesterov: bool,
    ) -> Result<()> {
        state.step += 1;
        let param_id = param_name.to_string();
        let param_len = param.data()?.len();

        // Get or initialize momentum buffer
        let momentum = state.momentum.entry(param_id).or_insert_with(|| vec![0.0; param_len]);

        let grad_data = grad.data()?;
        let mut param_data = param.data()?;

        // Fused SGD update loop with momentum
        for i in 0..param_data.len() {
            let mut grad_val = grad_data[i];

            // Apply weight decay if specified
            if weight_decay > 0.0 {
                grad_val += weight_decay as f32 * param_data[i];
            }

            // Update momentum buffer
            if momentum_coef > 0.0 {
                if state.step == 1 {
                    // First step: initialize momentum with gradient
                    momentum[i] = grad_val;
                } else {
                    // Update momentum with dampening
                    momentum[i] =
                        momentum_coef as f32 * momentum[i] + (1.0 - dampening as f32) * grad_val;
                }

                // Apply Nesterov momentum if enabled
                let update_direction = if nesterov {
                    grad_val + momentum_coef as f32 * momentum[i]
                } else {
                    momentum[i]
                };

                // Update parameter
                param_data[i] -= lr as f32 * update_direction;
            } else {
                // Simple SGD without momentum
                param_data[i] -= lr as f32 * grad_val;
            }
        }

        param.set_data_f32(&param_data)
    }

    /// Compute global gradient norm for clipping
    fn compute_global_norm(&self, gradients: &[Tensor]) -> Result<f64> {
        let mut total_norm_sq = 0.0;

        for grad in gradients {
            let norm = grad.norm()?;
            total_norm_sq += norm * norm;
        }

        Ok(total_norm_sq.sqrt() as f64)
    }

    /// Flush all pending operations
    pub fn flush(&mut self) -> Result<()> {
        self.execute_fused_batch()
    }

    /// Get fusion statistics
    pub fn get_fusion_stats(&self) -> FusionStats {
        let state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        state.fusion_stats.clone()
    }

    /// Reset fusion statistics
    pub fn reset_stats(&mut self) {
        let mut state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        state.fusion_stats = FusionStats::default();
    }

    /// Update fusion configuration
    pub fn update_config(&mut self, config: FusionConfig) {
        self.config = config;
    }
}

/// SIMD-optimized vectorized operations
#[cfg(target_arch = "x86_64")]
pub mod simd {

    /// SIMD-optimized Adam update
    pub fn simd_adam_update(
        param: &mut [f32],
        grad: &[f32],
        momentum: &mut [f32],
        velocity: &mut [f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        step: i32,
    ) {
        use std::arch::x86_64::*;

        let bias_correction1 = 1.0 - beta1.powi(step);
        let bias_correction2 = 1.0 - beta2.powi(step);
        let corrected_lr = lr * (bias_correction2.sqrt() / bias_correction1);

        unsafe {
            let beta1_vec = _mm256_set1_ps(beta1);
            let beta2_vec = _mm256_set1_ps(beta2);
            let one_minus_beta1 = _mm256_set1_ps(1.0 - beta1);
            let one_minus_beta2 = _mm256_set1_ps(1.0 - beta2);
            let eps_vec = _mm256_set1_ps(eps);
            let lr_vec = _mm256_set1_ps(corrected_lr);

            let chunks = param.len() / 8;
            for i in 0..chunks {
                let idx = i * 8;

                // Load values
                let p = _mm256_loadu_ps(param.as_ptr().add(idx));
                let g = _mm256_loadu_ps(grad.as_ptr().add(idx));
                let m = _mm256_loadu_ps(momentum.as_ptr().add(idx));
                let v = _mm256_loadu_ps(velocity.as_ptr().add(idx));

                // Update momentum: momentum = beta1 * momentum + (1 - beta1) * grad
                let m_new = _mm256_fmadd_ps(beta1_vec, m, _mm256_mul_ps(one_minus_beta1, g));

                // Update velocity: velocity = beta2 * velocity + (1 - beta2) * grad^2
                let g_sq = _mm256_mul_ps(g, g);
                let v_new = _mm256_fmadd_ps(beta2_vec, v, _mm256_mul_ps(one_minus_beta2, g_sq));

                // Update parameter: param = param - lr * momentum / (sqrt(velocity) + eps)
                let v_sqrt = _mm256_sqrt_ps(v_new);
                let v_sqrt_eps = _mm256_add_ps(v_sqrt, eps_vec);
                let update = _mm256_div_ps(m_new, v_sqrt_eps);
                let p_new = _mm256_fnmadd_ps(lr_vec, update, p);

                // Store results
                _mm256_storeu_ps(param.as_mut_ptr().add(idx), p_new);
                _mm256_storeu_ps(momentum.as_mut_ptr().add(idx), m_new);
                _mm256_storeu_ps(velocity.as_mut_ptr().add(idx), v_new);
            }

            // Handle remaining elements
            for i in (chunks * 8)..param.len() {
                let g = grad[i];
                momentum[i] = beta1 * momentum[i] + (1.0 - beta1) * g;
                velocity[i] = beta2 * velocity[i] + (1.0 - beta2) * g * g;
                param[i] -= corrected_lr * momentum[i] / (velocity[i].sqrt() + eps);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::Tensor;

    #[test]
    fn test_fused_optimizer_creation() {
        let config = FusionConfig::default();
        let optimizer = FusedOptimizer::new(config).expect("Failed to create fused optimizer");

        let stats = optimizer.get_fusion_stats();
        assert_eq!(stats.fused_operations, 0);
    }

    #[test]
    fn test_fused_adam_operation() {
        let config = FusionConfig::default();
        let mut optimizer = FusedOptimizer::new(config).expect("Failed to create fused optimizer");

        let param = Tensor::ones(&[10, 10]).expect("Failed to create tensor");
        let grad = Tensor::ones(&[10, 10]).expect("Failed to create tensor");

        let operation = FusedOperation::FusedAdam {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        };

        optimizer
            .queue_operation("param1".to_string(), operation, param, grad)
            .expect("Failed to queue operation");

        optimizer.flush().expect("Flush failed");

        let stats = optimizer.get_fusion_stats();
        assert_eq!(stats.fused_operations, 1);
    }

    #[test]
    fn test_fused_adamw_operation() {
        let config = FusionConfig::default();
        let mut optimizer = FusedOptimizer::new(config).expect("Failed to create fused optimizer");

        let param = Tensor::ones(&[5, 5]).expect("Failed to create tensor");
        let grad = Tensor::ones(&[5, 5]).expect("Failed to create tensor");

        let operation = FusedOperation::FusedAdamW {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        };

        optimizer
            .queue_operation("param2".to_string(), operation, param, grad)
            .expect("Failed to queue operation");

        optimizer.flush().expect("Flush failed");

        let stats = optimizer.get_fusion_stats();
        assert_eq!(stats.fused_operations, 1);
    }

    #[test]
    fn test_fused_sgd_operation() {
        let config = FusionConfig::default();
        let mut optimizer = FusedOptimizer::new(config).expect("Failed to create fused optimizer");

        let param = Tensor::ones(&[3, 3]).expect("Failed to create tensor");
        let grad = Tensor::ones(&[3, 3]).expect("Failed to create tensor");

        let operation = FusedOperation::FusedSGDMomentum {
            lr: 0.01,
            momentum: 0.9,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        };

        optimizer
            .queue_operation("param3".to_string(), operation, param, grad)
            .expect("Failed to queue operation");

        optimizer.flush().expect("Flush failed");

        let stats = optimizer.get_fusion_stats();
        assert_eq!(stats.fused_operations, 1);
    }

    #[test]
    fn test_batch_fusion() {
        let config = FusionConfig {
            batch_size: 2,
            ..FusionConfig::default()
        };
        let mut optimizer = FusedOptimizer::new(config).expect("Failed to create fused optimizer");

        // Queue multiple operations
        for i in 0..3 {
            let param = Tensor::ones(&[2, 2]).expect("Failed to create tensor");
            let grad = Tensor::ones(&[2, 2]).expect("Failed to create tensor");

            let operation = FusedOperation::FusedAdam {
                lr: 0.001,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
            };

            optimizer
                .queue_operation(format!("param_{}", i), operation, param, grad)
                .expect("Operation failed in test");
        }

        // Should have executed batch automatically
        let stats = optimizer.get_fusion_stats();
        assert!(stats.fused_operations > 0);
    }

    #[test]
    fn test_fusion_stats() {
        let config = FusionConfig::default();
        let mut optimizer = FusedOptimizer::new(config).expect("Failed to create fused optimizer");

        let param = Tensor::ones(&[10, 10]).expect("Failed to create tensor");
        let grad = Tensor::ones(&[10, 10]).expect("Failed to create tensor");

        let operation = FusedOperation::FusedAdam {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        };

        optimizer
            .queue_operation("param1".to_string(), operation, param, grad)
            .expect("Failed to queue operation");

        optimizer.flush().expect("Flush failed");

        let stats = optimizer.get_fusion_stats();
        assert_eq!(stats.fused_operations, 1);
        assert!(stats.memory_bandwidth_saved > 0);

        optimizer.reset_stats();
        let reset_stats = optimizer.get_fusion_stats();
        assert_eq!(reset_stats.fused_operations, 0);
        assert_eq!(reset_stats.memory_bandwidth_saved, 0);
    }

    #[test]
    fn test_global_norm_computation() {
        let config = FusionConfig::default();
        let optimizer = FusedOptimizer::new(config).expect("Failed to create fused optimizer");

        let grad1 = Tensor::ones(&[3, 3]).expect("Failed to create tensor");
        let grad2 = Tensor::ones(&[2, 2]).expect("Failed to create tensor");

        let gradients = vec![grad1, grad2];
        let global_norm = optimizer
            .compute_global_norm(&gradients)
            .expect("Failed to compute global norm");

        // Expected: sqrt(9 + 4) = sqrt(13) ≈ 3.606
        assert!((global_norm - 3.606).abs() < 0.01);
    }

    /// Regression: the clipping arm used to compute `clip_coef` and drop the result,
    /// so a gradient with norm above `max_norm` came back unchanged.
    #[test]
    fn test_fused_clipping_actually_clips() {
        let config = FusionConfig::default();
        let mut optimizer = FusedOptimizer::new(config).expect("create fused optimizer");

        // 4 elements of 5.0 => norm = sqrt(4 * 25) = 10.0
        let grad = Tensor::from_vec(vec![5.0_f32; 4], &[4]).expect("grad tensor");
        let param = Tensor::from_vec(vec![0.0_f32; 4], &[4]).expect("param tensor");
        let before = grad.norm().expect("norm before");
        assert!((before - 10.0).abs() < 1e-4, "precondition norm: {before}");

        optimizer
            .queue_operation(
                "w".to_string(),
                FusedOperation::FusedGradientClipping {
                    max_norm: 1.0,
                    scale_factor: 1.0,
                },
                param,
                grad,
            )
            .expect("queue clipping");
        optimizer.flush().expect("flush");

        let clipped = optimizer.take_clipped_gradients().expect("take clipped");
        let out = clipped.get("w").expect("clipped gradient recorded");
        let after = out.norm().expect("norm after");
        assert!(
            (after - 1.0).abs() < 1e-4,
            "gradient norm must be clipped to max_norm, got {after}"
        );
    }

    /// Regression: `scale_factor` was also dropped on the non-clipping path.
    #[test]
    fn test_fused_clipping_applies_scale_factor_below_threshold() {
        let config = FusionConfig::default();
        let mut optimizer = FusedOptimizer::new(config).expect("create fused optimizer");

        let grad = Tensor::from_vec(vec![1.0_f32; 4], &[4]).expect("grad tensor");
        let param = Tensor::from_vec(vec![0.0_f32; 4], &[4]).expect("param tensor");

        optimizer
            .queue_operation(
                "w".to_string(),
                FusedOperation::FusedGradientClipping {
                    max_norm: 100.0,
                    scale_factor: 0.5,
                },
                param,
                grad,
            )
            .expect("queue clipping");
        optimizer.flush().expect("flush");

        let clipped = optimizer.take_clipped_gradients().expect("take clipped");
        let data = clipped.get("w").expect("clipped gradient").data().expect("data");
        for v in data {
            assert!((v - 0.5).abs() < 1e-6, "scale_factor must be applied: {v}");
        }
    }

    /// Regression: `fused_adam_update` wrote into `param.data()?`, a throwaway copy,
    /// so the parameter never moved.
    #[test]
    fn test_fused_adam_moves_parameter() {
        let config = FusionConfig::default();
        let mut optimizer = FusedOptimizer::new(config).expect("create fused optimizer");

        let param = Tensor::from_vec(vec![1.0_f32; 4], &[4]).expect("param tensor");
        let grad = Tensor::from_vec(vec![1.0_f32; 4], &[4]).expect("grad tensor");

        optimizer
            .queue_operation(
                "w".to_string(),
                FusedOperation::FusedAdam {
                    lr: 0.1,
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-8,
                    weight_decay: 0.0,
                },
                param,
                grad,
            )
            .expect("queue adam");
        optimizer.flush().expect("flush");

        let updated = optimizer.take_updated_parameters().expect("take updated");
        let data = updated.get("w").expect("updated parameter").data().expect("data");
        // Step 1 Adam with g=1: m_hat = g, v_hat = g^2 => step ≈ lr.
        for v in data {
            assert!(v < 1.0, "parameter must decrease, got {v}");
            assert!(
                (v - 0.9).abs() < 1e-3,
                "first Adam step should be ≈ lr = 0.1, got {v}"
            );
        }
    }

    /// Adam state must persist across steps: identical gradients give shrinking steps
    /// once the bias correction saturates.
    #[test]
    fn test_fused_adam_in_place_carries_state() {
        let config = FusionConfig::default();
        let mut optimizer = FusedOptimizer::new(config).expect("create fused optimizer");

        let mut param = Tensor::from_vec(vec![0.0_f32; 2], &[2]).expect("param");
        let mut grad = Tensor::from_vec(vec![1.0_f32; 2], &[2]).expect("grad");
        let op = FusedOperation::FusedAdam {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        };

        optimizer
            .apply_in_place("w", op.clone(), &mut param, &mut grad)
            .expect("step 1");
        let after_first = param.data().expect("data")[0];
        assert!(after_first < 0.0, "first step must move the parameter");

        // Second step with a *zero* gradient: only carried momentum can move the
        // parameter now. A stateless implementation would leave it exactly in place.
        let mut zero_grad = Tensor::from_vec(vec![0.0_f32; 2], &[2]).expect("zero grad");
        optimizer.apply_in_place("w", op, &mut param, &mut zero_grad).expect("step 2");
        let after_second = param.data().expect("data")[0];

        assert!(
            (after_second - after_first).abs() > 1e-6,
            "carried momentum must still move the parameter on a zero gradient: \
             {after_first} -> {after_second}"
        );
    }

    /// Regression: SGD wrote into a throwaway copy too.
    #[test]
    fn test_fused_sgd_in_place_moves_parameter() {
        let config = FusionConfig::default();
        let mut optimizer = FusedOptimizer::new(config).expect("create fused optimizer");

        let mut param = Tensor::from_vec(vec![1.0_f32, 2.0], &[2]).expect("param");
        let mut grad = Tensor::from_vec(vec![1.0_f32, 1.0], &[2]).expect("grad");

        optimizer
            .apply_in_place(
                "w",
                FusedOperation::FusedSGDMomentum {
                    lr: 0.5,
                    momentum: 0.0,
                    dampening: 0.0,
                    weight_decay: 0.0,
                    nesterov: false,
                },
                &mut param,
                &mut grad,
            )
            .expect("sgd step");

        let data = param.data().expect("data");
        // Plain SGD: p -= lr * g  =>  1.0 - 0.5 = 0.5, 2.0 - 0.5 = 1.5
        assert!((data[0] - 0.5).abs() < 1e-6, "got {}", data[0]);
        assert!((data[1] - 1.5).abs() < 1e-6, "got {}", data[1]);
    }

    /// Convergence smoke test on a quadratic bowl f(x) = sum(x^2), grad = 2x.
    #[test]
    fn test_fused_adam_converges_on_quadratic() {
        let config = FusionConfig::default();
        let mut optimizer = FusedOptimizer::new(config).expect("create fused optimizer");

        let mut param = Tensor::from_vec(vec![1.0_f32; 4], &[4]).expect("param");
        let initial_loss: f32 = param.data().expect("data").iter().map(|v| v * v).sum();

        for _ in 0..400 {
            let grad_data: Vec<f32> = param.data().expect("data").iter().map(|v| 2.0 * v).collect();
            let mut grad = Tensor::from_vec(grad_data, &[4]).expect("grad");
            optimizer
                .apply_in_place(
                    "w",
                    FusedOperation::FusedAdam {
                        lr: 0.05,
                        beta1: 0.9,
                        beta2: 0.999,
                        eps: 1e-8,
                        weight_decay: 0.0,
                    },
                    &mut param,
                    &mut grad,
                )
                .expect("adam step");
        }

        let final_loss: f32 = param.data().expect("data").iter().map(|v| v * v).sum();
        assert!(
            final_loss < initial_loss * 1e-2,
            "loss must decrease: {initial_loss} -> {final_loss}"
        );
    }
}
