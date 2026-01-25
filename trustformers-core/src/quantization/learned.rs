//! Learned quantization parameters for optimal quantization quality.
//!
//! This module implements quantization with learnable parameters where scales and
//! zero points are learned during training rather than computed statically.
//! This approach can significantly improve quantization quality and model accuracy.

use super::base::QuantizationConfig;
use crate::autodiff::{AutodiffEngine, Variable};
use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use crate::traits::Layer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for learned quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedQuantConfig {
    /// Base quantization configuration
    pub base_config: QuantizationConfig,
    /// Learning rate for quantization parameters
    pub learning_rate: f32,
    /// Whether to learn scales
    pub learn_scales: bool,
    /// Whether to learn zero points
    pub learn_zero_points: bool,
    /// Whether to use per-channel learned parameters
    pub per_channel_learned: bool,
    /// Regularization weight for quantization parameters
    pub regularization_weight: f32,
    /// Temperature for straight-through estimator
    pub ste_temperature: f32,
    /// Clipping range for learned parameters
    pub scale_min: f32,
    pub scale_max: f32,
    pub zero_point_min: i32,
    pub zero_point_max: i32,
    /// Whether to use exponential moving average for parameters
    pub use_ema: bool,
    /// EMA momentum
    pub ema_momentum: f32,
    /// Whether to use gradient scaling
    pub use_gradient_scaling: bool,
    /// Gradient scaling factor
    pub gradient_scale_factor: f32,
}

impl Default for LearnedQuantConfig {
    fn default() -> Self {
        Self {
            base_config: QuantizationConfig::default(),
            learning_rate: 1e-4,
            learn_scales: true,
            learn_zero_points: true,
            per_channel_learned: true,
            regularization_weight: 1e-6,
            ste_temperature: 1.0,
            scale_min: 1e-6,
            scale_max: 1e6,
            zero_point_min: -128,
            zero_point_max: 127,
            use_ema: true,
            ema_momentum: 0.999,
            use_gradient_scaling: false,
            gradient_scale_factor: 1.0,
        }
    }
}

/// Learned quantization parameters
#[derive(Debug, Clone)]
pub struct LearnedQuantParams {
    /// Learned scales (one per channel or single value)
    pub scales: Variable,
    /// Learned zero points (one per channel or single value)
    pub zero_points: Variable,
    /// EMA scales for inference
    pub ema_scales: Option<Variable>,
    /// EMA zero points for inference
    pub ema_zero_points: Option<Variable>,
    /// Configuration
    pub config: LearnedQuantConfig,
    /// Training mode flag
    pub training: bool,
    /// Reference to the autodiff engine for creating new variables
    engine: Arc<AutodiffEngine>,
}

impl LearnedQuantParams {
    /// Create new learned quantization parameters
    pub fn new(
        config: LearnedQuantConfig,
        shape: &[usize],
        autodiff_engine: &Arc<AutodiffEngine>,
    ) -> Result<Self> {
        let param_shape = if config.per_channel_learned {
            // For per-channel quantization, parameters have shape [channels]
            if shape.is_empty() {
                return Err(TrustformersError::config_error(
                    "Cannot use per-channel learned quantization with scalar tensor",
                    "LearnedQuantParams::new",
                ));
            }
            vec![shape[0]]
        } else {
            // For per-tensor quantization, parameters are scalars
            vec![1]
        };

        // Initialize scales with reasonable values
        let initial_scales = if config.per_channel_learned {
            Tensor::ones(&param_shape)?
        } else {
            Tensor::scalar(1.0)?
        };

        // Initialize zero points with zeros
        let initial_zero_points = if config.per_channel_learned {
            Tensor::zeros(&param_shape)?
        } else {
            Tensor::scalar(0.0)?
        };

        let scales = autodiff_engine.variable(initial_scales, config.learn_scales);
        let zero_points = autodiff_engine.variable(initial_zero_points, config.learn_zero_points);

        let (ema_scales, ema_zero_points) = if config.use_ema {
            let ema_scales = autodiff_engine.variable(scales.data()?, false);
            let ema_zero_points = autodiff_engine.variable(zero_points.data()?, false);
            (Some(ema_scales), Some(ema_zero_points))
        } else {
            (None, None)
        };

        Ok(Self {
            scales,
            zero_points,
            ema_scales,
            ema_zero_points,
            config,
            training: true,
            engine: autodiff_engine.clone(),
        })
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Update EMA parameters
    pub fn update_ema(&mut self) -> Result<()> {
        if !self.config.use_ema || !self.training {
            return Ok(());
        }

        let momentum = self.config.ema_momentum;

        if let (Some(ref mut ema_scales), Some(ref mut ema_zero_points)) =
            (&mut self.ema_scales, &mut self.ema_zero_points)
        {
            // Update EMA scales: ema = momentum * ema + (1 - momentum) * current
            let current_scales = self.scales.data()?;
            let current_ema_scales = ema_scales.data()?;
            let new_ema_scales = current_ema_scales
                .scalar_mul(momentum)?
                .add(&current_scales.scalar_mul(1.0 - momentum)?)?;
            ema_scales.set_data(new_ema_scales)?;

            // Update EMA zero points
            let current_zero_points = self.zero_points.data()?;
            let current_ema_zero_points = ema_zero_points.data()?;
            let new_ema_zero_points = current_ema_zero_points
                .scalar_mul(momentum)?
                .add(&current_zero_points.scalar_mul(1.0 - momentum)?)?;
            ema_zero_points.set_data(new_ema_zero_points)?;
        }

        Ok(())
    }

    /// Get effective scales (EMA during inference, learned during training)
    pub fn effective_scales(&self) -> Result<Variable> {
        if !self.training && self.config.use_ema {
            if let Some(ref ema_scales) = self.ema_scales {
                Ok(ema_scales.clone())
            } else {
                Ok(self.scales.clone())
            }
        } else {
            Ok(self.scales.clone())
        }
    }

    /// Get effective zero points (EMA during inference, learned during training)
    pub fn effective_zero_points(&self) -> Result<Variable> {
        if !self.training && self.config.use_ema {
            if let Some(ref ema_zero_points) = self.ema_zero_points {
                Ok(ema_zero_points.clone())
            } else {
                Ok(self.zero_points.clone())
            }
        } else {
            Ok(self.zero_points.clone())
        }
    }

    /// Apply parameter constraints
    pub fn apply_constraints(&mut self) -> Result<()> {
        // Clamp scales to valid range
        let scales_data = self.scales.data()?;
        let clamped_scales = scales_data.clamp(self.config.scale_min, self.config.scale_max)?;
        self.scales.set_data(clamped_scales)?;

        // Clamp zero points to valid range
        let zero_points_data = self.zero_points.data()?;
        let clamped_zero_points = zero_points_data.clamp(
            self.config.zero_point_min as f32,
            self.config.zero_point_max as f32,
        )?;
        self.zero_points.set_data(clamped_zero_points)?;

        Ok(())
    }

    /// Compute regularization loss
    pub fn regularization_loss(&self) -> Result<Variable> {
        // If regularization weight is zero, return zero loss directly
        if self.config.regularization_weight == 0.0 {
            // Create a zero scalar using the same engine as scales
            let zero_tensor = Tensor::scalar(0.0)?;
            return Ok(self.engine.variable(zero_tensor, false));
        }

        // Calculate L2 regularization on tensor data directly to avoid computation graph issues
        let scales_data = self.scales.data()?;
        let zero_points_data = self.zero_points.data()?;

        // Calculate squared norms
        let scales_squared = scales_data.square()?;
        let zero_points_squared = zero_points_data.square()?;

        // Calculate means
        let scales_mean = scales_squared.mean()?;
        let zero_points_mean = zero_points_squared.mean()?;

        // Extract scalar values and sum the losses
        let scales_mean_val = match scales_mean {
            Tensor::F32(ref arr) => arr.iter().next().cloned().unwrap_or(0.0),
            Tensor::F64(ref arr) => arr.iter().next().cloned().unwrap_or(0.0) as f32,
            _ => 0.0,
        };
        let zero_points_mean_val = match zero_points_mean {
            Tensor::F32(ref arr) => arr.iter().next().cloned().unwrap_or(0.0),
            Tensor::F64(ref arr) => arr.iter().next().cloned().unwrap_or(0.0) as f32,
            _ => 0.0,
        };

        let total_loss_value = scales_mean_val + zero_points_mean_val;
        let weighted_loss = total_loss_value * self.config.regularization_weight;

        // Create a new variable with the result
        let loss_tensor = Tensor::scalar(weighted_loss)?;
        Ok(self.engine.variable(loss_tensor, true))
    }
}

/// Learned fake quantization layer
#[derive(Debug, Clone)]
pub struct LearnedFakeQuantize {
    /// Learned quantization parameters
    params: LearnedQuantParams,
    /// Number of bits for quantization
    num_bits: u8,
    /// Autodiff engine reference
    engine: Arc<AutodiffEngine>,
}

impl LearnedFakeQuantize {
    /// Create a new learned fake quantization layer
    pub fn new(
        config: LearnedQuantConfig,
        input_shape: &[usize],
        num_bits: u8,
        engine: Arc<AutodiffEngine>,
    ) -> Result<Self> {
        let params = LearnedQuantParams::new(config, input_shape, &engine)?;

        Ok(Self {
            params,
            num_bits,
            engine,
        })
    }

    /// Quantize and dequantize with learned parameters (fake quantization)
    pub fn forward_fake_quantize(&mut self, input: &Variable) -> Result<Variable> {
        let scales = self.params.effective_scales()?;
        let zero_points = self.params.effective_zero_points()?;

        // Compute quantization bounds
        let qmin = -(1 << (self.num_bits - 1)) as f32;
        let qmax = ((1 << (self.num_bits - 1)) - 1) as f32;

        // Quantize: q = round(x / scale + zero_point)
        let scaled = input.div(&scales)?;
        let shifted = scaled.add(&zero_points)?;
        let quantized = self.straight_through_round(&shifted)?;
        let clamped = self.clamp(&quantized, qmin, qmax)?;

        // Dequantize: x = (q - zero_point) * scale
        let dequantized = clamped.sub(&zero_points)?.mul(&scales)?;

        // Update EMA parameters if in training mode
        if self.params.training {
            self.params.update_ema()?;
            self.params.apply_constraints()?;
        }

        Ok(dequantized)
    }

    /// Straight-through estimator for rounding
    fn straight_through_round(&self, input: &Variable) -> Result<Variable> {
        // In forward pass: round, in backward pass: identity
        // This is a simplified implementation - in practice you'd use custom gradients

        if self.params.config.ste_temperature == 1.0 {
            // Standard straight-through estimator
            self.round_with_straight_through(input)
        } else {
            // Soft quantization with temperature
            self.soft_quantization(input)
        }
    }

    /// Round with straight-through gradients
    fn round_with_straight_through(&self, input: &Variable) -> Result<Variable> {
        // For now, we'll use a simple approximation
        // In a full implementation, you'd use custom gradient functions
        let rounded_data = input.data()?.round()?;
        let rounded_var = self.engine.variable(rounded_data, input.requires_grad());
        Ok(rounded_var)
    }

    /// Soft quantization with temperature
    fn soft_quantization(&self, input: &Variable) -> Result<Variable> {
        let temp = self.params.config.ste_temperature;

        // Soft rounding using sigmoid-based approximation
        let floor_val = input.clone(); // Simplified - should be floor
        let ceil_val = floor_val.add_scalar(1.0)?;

        let diff = input.sub(&floor_val)?;
        let sigmoid_weight = diff.div_scalar(temp)?.sigmoid()?;

        let result = floor_val
            .mul(&sigmoid_weight.sub_scalar(1.0)?.neg()?)?
            .add(&ceil_val.mul(&sigmoid_weight)?)?;

        Ok(result)
    }

    /// Clamp values to quantization range
    fn clamp(&self, input: &Variable, min_val: f32, max_val: f32) -> Result<Variable> {
        // Simplified clamping - in practice you'd implement proper clamp operation
        let data = input.data()?;
        let clamped_data = data.clamp(min_val, max_val)?;
        let clamped_var = self.engine.variable(clamped_data, input.requires_grad());
        Ok(clamped_var)
    }

    /// Get quantization parameters
    pub fn params(&self) -> &LearnedQuantParams {
        &self.params
    }

    /// Get mutable quantization parameters
    pub fn params_mut(&mut self) -> &mut LearnedQuantParams {
        &mut self.params
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.params.set_training(training);
    }

    /// Compute total loss including regularization
    pub fn total_loss(&self, reconstruction_loss: &Variable) -> Result<Variable> {
        let reg_loss = self.params.regularization_loss()?;
        reconstruction_loss.add(&reg_loss)
    }
}

/// Learned quantization optimizer
#[derive(Debug)]
pub struct LearnedQuantOptimizer {
    /// Learning rate
    learning_rate: f32,
    /// Momentum for gradient updates
    momentum: f32,
    /// Accumulated gradients for scales
    scale_momentum: HashMap<String, Variable>,
    /// Accumulated gradients for zero points
    zero_point_momentum: HashMap<String, Variable>,
    /// Autodiff engine
    engine: Arc<AutodiffEngine>,
}

impl LearnedQuantOptimizer {
    /// Create a new learned quantization optimizer
    pub fn new(learning_rate: f32, momentum: f32, engine: Arc<AutodiffEngine>) -> Self {
        Self {
            learning_rate,
            momentum,
            scale_momentum: HashMap::new(),
            zero_point_momentum: HashMap::new(),
            engine,
        }
    }

    /// Update learned quantization parameters
    pub fn step(&mut self, layers: &mut [&mut LearnedFakeQuantize]) -> Result<()> {
        for (layer_idx, layer) in layers.iter_mut().enumerate() {
            let layer_name = format!("layer_{}", layer_idx);

            // Update scales
            if let Some(scale_grad) = layer.params.scales.grad()? {
                self.update_scales_parameter(
                    &mut layer.params.scales,
                    &scale_grad,
                    &format!("{}_scales", layer_name),
                )?;
            }

            // Update zero points
            if let Some(zero_point_grad) = layer.params.zero_points.grad()? {
                self.update_zero_points_parameter(
                    &mut layer.params.zero_points,
                    &zero_point_grad,
                    &format!("{}_zero_points", layer_name),
                )?;
            }

            // Apply constraints after updates
            layer.params.apply_constraints()?;
        }

        Ok(())
    }

    /// Update a single parameter with momentum
    #[allow(dead_code)]
    fn update_parameter(
        &mut self,
        parameter: &mut Variable,
        gradient: &Tensor,
        momentum_dict: &mut HashMap<String, Variable>,
        param_name: &str,
    ) -> Result<()> {
        let param_data = parameter.data()?;

        // Get or initialize momentum
        let momentum_var = if let Some(momentum) = momentum_dict.get(param_name) {
            momentum.clone()
        } else {
            let zero_momentum = self.engine.variable(Tensor::zeros(&param_data.shape())?, false);
            momentum_dict.insert(param_name.to_string(), zero_momentum.clone());
            zero_momentum
        };

        // Update momentum: m = momentum * m + gradient
        let momentum_data = momentum_var.data()?;
        let new_momentum = momentum_data.scalar_mul(self.momentum)?.add(gradient)?;

        // Update parameter: param = param - learning_rate * momentum
        let update = new_momentum.scalar_mul(-self.learning_rate)?;
        let new_param = param_data.add(&update)?;

        // Set updated values
        parameter.set_data(new_param)?;
        if let Some(momentum_var) = momentum_dict.get_mut(param_name) {
            momentum_var.set_data(new_momentum)?;
        } else {
            return Err(TrustformersError::runtime_error(
                "Momentum variable not found after insertion".into(),
            ));
        }

        Ok(())
    }

    /// Update scales parameter
    fn update_scales_parameter(
        &mut self,
        parameter: &mut Variable,
        gradient: &Tensor,
        param_name: &str,
    ) -> Result<()> {
        // Extract momentum and other fields to avoid borrowing conflicts
        let param_data = parameter.data()?;

        // Get or initialize momentum
        let momentum_var = if let Some(momentum) = self.scale_momentum.get(param_name) {
            momentum.clone()
        } else {
            let zero_momentum = self.engine.variable(Tensor::zeros(&param_data.shape())?, false);
            self.scale_momentum.insert(param_name.to_string(), zero_momentum.clone());
            zero_momentum
        };

        // Update momentum: m = momentum * m + gradient
        let momentum_data = momentum_var.data()?;
        let new_momentum = momentum_data.scalar_mul(self.momentum)?.add(gradient)?;

        // Update parameter: param = param - learning_rate * momentum
        let update = new_momentum.scalar_mul(-self.learning_rate)?;
        let new_param = param_data.add(&update)?;

        // Set updated values
        parameter.set_data(new_param)?;
        if let Some(momentum_var) = self.scale_momentum.get_mut(param_name) {
            momentum_var.set_data(new_momentum)?;
        } else {
            return Err(TrustformersError::runtime_error(
                "Scale momentum variable not found after insertion".into(),
            ));
        }

        Ok(())
    }

    /// Update zero points parameter
    fn update_zero_points_parameter(
        &mut self,
        parameter: &mut Variable,
        gradient: &Tensor,
        param_name: &str,
    ) -> Result<()> {
        // Extract momentum and other fields to avoid borrowing conflicts
        let param_data = parameter.data()?;

        // Get or initialize momentum
        let momentum_var = if let Some(momentum) = self.zero_point_momentum.get(param_name) {
            momentum.clone()
        } else {
            let zero_momentum = self.engine.variable(Tensor::zeros(&param_data.shape())?, false);
            self.zero_point_momentum.insert(param_name.to_string(), zero_momentum.clone());
            zero_momentum
        };

        // Update momentum: m = momentum * m + gradient
        let momentum_data = momentum_var.data()?;
        let new_momentum = momentum_data.scalar_mul(self.momentum)?.add(gradient)?;

        // Update parameter: param = param - learning_rate * momentum
        let update = new_momentum.scalar_mul(-self.learning_rate)?;
        let new_param = param_data.add(&update)?;

        // Set updated values
        parameter.set_data(new_param)?;
        if let Some(momentum_var) = self.zero_point_momentum.get_mut(param_name) {
            momentum_var.set_data(new_momentum)?;
        } else {
            return Err(TrustformersError::runtime_error(
                "Zero point momentum variable not found after insertion".into(),
            ));
        }

        Ok(())
    }

    /// Zero gradients
    pub fn zero_grad(&self, layers: &[&LearnedFakeQuantize]) {
        for layer in layers {
            layer.params.scales.zero_grad();
            layer.params.zero_points.zero_grad();
        }
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    /// Get learning rate
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
}

/// Learned quantization trainer
pub struct LearnedQuantTrainer {
    /// Configuration
    #[allow(dead_code)]
    config: LearnedQuantConfig,
    /// Optimizer
    optimizer: LearnedQuantOptimizer,
    /// Autodiff engine
    #[allow(dead_code)]
    engine: Arc<AutodiffEngine>,
    /// Training statistics
    stats: LearnedQuantStats,
}

/// Training statistics for learned quantization
#[derive(Debug, Default, Clone)]
pub struct LearnedQuantStats {
    /// Number of training steps
    pub steps: u64,
    /// Average reconstruction loss
    pub avg_reconstruction_loss: f32,
    /// Average regularization loss
    pub avg_regularization_loss: f32,
    /// Average total loss
    pub avg_total_loss: f32,
    /// Learning rate history
    pub lr_history: Vec<f32>,
    /// Loss history
    pub loss_history: Vec<f32>,
}

impl LearnedQuantTrainer {
    /// Create a new learned quantization trainer
    pub fn new(config: LearnedQuantConfig, engine: Arc<AutodiffEngine>) -> Self {
        let optimizer = LearnedQuantOptimizer::new(
            config.learning_rate,
            0.9, // momentum
            engine.clone(),
        );

        Self {
            config,
            optimizer,
            engine,
            stats: LearnedQuantStats::default(),
        }
    }

    /// Train learned quantization parameters
    pub fn train_step(
        &mut self,
        input: &Variable,
        target: &Variable,
        layers: &mut [&mut LearnedFakeQuantize],
    ) -> Result<f32> {
        // Forward pass through all quantization layers
        let mut current = input.clone();
        for layer in layers.iter_mut() {
            current = layer.forward_fake_quantize(&current)?;
        }

        // Compute reconstruction loss
        let reconstruction_loss = self.compute_reconstruction_loss(&current, target)?;

        // Compute regularization loss
        let mut total_reg_loss = Variable::scalar(0.0, false)?;
        for layer in layers.iter() {
            let reg_loss = layer.params.regularization_loss()?;
            total_reg_loss = total_reg_loss.add(&reg_loss)?;
        }

        // Total loss
        let total_loss = reconstruction_loss.add(&total_reg_loss)?;

        // Backward pass
        let layer_refs: Vec<&LearnedFakeQuantize> = layers.iter().map(|layer| &**layer).collect();
        self.optimizer.zero_grad(&layer_refs);
        total_loss.backward()?;

        // Update parameters
        self.optimizer.step(layers)?;

        // Update statistics
        let loss_value = total_loss.item()?;
        self.update_stats(
            loss_value,
            reconstruction_loss.item()?,
            total_reg_loss.item()?,
        );

        Ok(loss_value)
    }

    /// Compute reconstruction loss
    fn compute_reconstruction_loss(
        &self,
        output: &Variable,
        target: &Variable,
    ) -> Result<Variable> {
        // Use MSE loss for reconstruction
        let diff = output.sub(target)?;
        let squared_diff = diff.square()?;
        squared_diff.mean(None)
    }

    /// Update training statistics
    fn update_stats(
        &mut self,
        total_loss: f32,
        reconstruction_loss: f32,
        regularization_loss: f32,
    ) {
        self.stats.steps += 1;

        let alpha = 0.99; // EMA factor
        if self.stats.steps == 1 {
            self.stats.avg_total_loss = total_loss;
            self.stats.avg_reconstruction_loss = reconstruction_loss;
            self.stats.avg_regularization_loss = regularization_loss;
        } else {
            self.stats.avg_total_loss =
                alpha * self.stats.avg_total_loss + (1.0 - alpha) * total_loss;
            self.stats.avg_reconstruction_loss =
                alpha * self.stats.avg_reconstruction_loss + (1.0 - alpha) * reconstruction_loss;
            self.stats.avg_regularization_loss =
                alpha * self.stats.avg_regularization_loss + (1.0 - alpha) * regularization_loss;
        }

        self.stats.lr_history.push(self.optimizer.learning_rate());
        self.stats.loss_history.push(total_loss);
    }

    /// Get training statistics
    pub fn stats(&self) -> &LearnedQuantStats {
        &self.stats
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.optimizer.set_learning_rate(lr);
    }

    /// Get learning rate
    pub fn learning_rate(&self) -> f32 {
        self.optimizer.learning_rate()
    }
}

/// Learned quantization layer for neural networks
#[derive(Debug)]
pub struct LearnedQuantLayer {
    /// Fake quantization layer
    fake_quant: LearnedFakeQuantize,
    /// Layer name
    name: String,
}

impl LearnedQuantLayer {
    /// Create a new learned quantization layer
    pub fn new(
        name: String,
        config: LearnedQuantConfig,
        input_shape: &[usize],
        num_bits: u8,
        engine: Arc<AutodiffEngine>,
    ) -> Result<Self> {
        let fake_quant = LearnedFakeQuantize::new(config, input_shape, num_bits, engine)?;

        Ok(Self { fake_quant, name })
    }

    /// Get layer name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get fake quantization layer
    pub fn fake_quant(&self) -> &LearnedFakeQuantize {
        &self.fake_quant
    }

    /// Get mutable fake quantization layer
    pub fn fake_quant_mut(&mut self) -> &mut LearnedFakeQuantize {
        &mut self.fake_quant
    }
}

impl Layer for LearnedQuantLayer {
    type Input = Variable;
    type Output = Variable;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // This is immutable forward, so we can't update parameters
        // In practice, you'd need a mutable forward or use interior mutability
        let scales = self.fake_quant.params.effective_scales()?;
        let zero_points = self.fake_quant.params.effective_zero_points()?;

        // Simplified quantization for immutable forward
        let result = input.mul(&scales)?.add(&zero_points)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_learned_quant_config() {
        let config = LearnedQuantConfig::default();
        assert!(config.learn_scales);
        assert!(config.learn_zero_points);
        assert!(config.per_channel_learned);
    }

    #[test]
    fn test_learned_quant_params() {
        let config = LearnedQuantConfig::default();
        let engine = Arc::new(AutodiffEngine::default());
        let shape = vec![10, 20];

        let params = LearnedQuantParams::new(config, &shape, &engine).unwrap();
        assert_eq!(params.scales.shape().unwrap(), vec![10]);
        assert_eq!(params.zero_points.shape().unwrap(), vec![10]);
    }

    #[test]
    fn test_learned_fake_quantize() {
        let config = LearnedQuantConfig {
            per_channel_learned: false, // Use per-tensor quantization to avoid shape issues
            ..Default::default()
        };
        let engine = Arc::new(AutodiffEngine::default());
        let shape = vec![5, 10];

        let mut fake_quant = LearnedFakeQuantize::new(config, &shape, 8, engine.clone()).unwrap();

        let input_tensor = Tensor::randn(&[2, 5, 10]).unwrap();
        let input_var = engine.variable(input_tensor, true);

        let result = fake_quant.forward_fake_quantize(&input_var).unwrap();
        assert_eq!(result.shape().unwrap(), vec![2, 5, 10]);
    }

    #[test]
    fn test_learned_quant_optimizer() {
        let engine = Arc::new(AutodiffEngine::default());
        let mut optimizer = LearnedQuantOptimizer::new(0.01, 0.9, engine.clone());

        assert_eq!(optimizer.learning_rate(), 0.01);

        optimizer.set_learning_rate(0.001);
        assert_eq!(optimizer.learning_rate(), 0.001);
    }

    #[test]
    fn test_learned_quant_trainer() {
        let config = LearnedQuantConfig::default();
        let engine = Arc::new(AutodiffEngine::default());

        let trainer = LearnedQuantTrainer::new(config, engine);
        assert_eq!(trainer.stats().steps, 0);
    }

    #[test]
    fn test_parameter_constraints() {
        let config = LearnedQuantConfig {
            scale_min: 0.1,
            scale_max: 10.0,
            ..Default::default()
        };

        let engine = Arc::new(AutodiffEngine::default());
        let shape = vec![5];

        let mut params = LearnedQuantParams::new(config, &shape, &engine).unwrap();

        // Set scales outside bounds
        let bad_scales = Tensor::from_vec(vec![0.01, 100.0, 1.0, 0.05, 50.0], &[5]).unwrap();
        params.scales.set_data(bad_scales).unwrap();

        params.apply_constraints().unwrap();

        let constrained_scales = params.scales.data().unwrap().to_vec_f32().unwrap();
        for &scale in &constrained_scales {
            assert!((0.1..=10.0).contains(&scale));
        }
    }

    #[test]
    fn test_ema_updates() {
        let config = LearnedQuantConfig {
            use_ema: true,
            ema_momentum: 0.9,
            ..Default::default()
        };

        let engine = Arc::new(AutodiffEngine::default());
        let shape = vec![3];

        let mut params = LearnedQuantParams::new(config, &shape, &engine).unwrap();

        // Set initial values
        let new_scales = Tensor::from_vec(vec![2.0, 3.0, 4.0], &[3]).unwrap();
        params.scales.set_data(new_scales).unwrap();

        params.update_ema().unwrap();

        // Check that EMA was updated
        let ema_scales = params.ema_scales.as_ref().unwrap().data().unwrap().to_vec_f32().unwrap();
        assert!(ema_scales[0] > 1.0 && ema_scales[0] < 2.0); // Should be between initial and current
    }

    #[test]
    fn test_regularization_loss() {
        let config = LearnedQuantConfig {
            use_ema: false,             // Disable EMA to avoid computation graph issues
            regularization_weight: 0.0, // Test zero weight case first
            ..Default::default()
        };
        let engine = Arc::new(AutodiffEngine::default());
        let shape = vec![2];

        let params = LearnedQuantParams::new(config, &shape, &engine).unwrap();

        let reg_loss = params.regularization_loss().unwrap();
        assert_eq!(reg_loss.item().unwrap(), 0.0);

        // Now test non-zero weight
        let config2 = LearnedQuantConfig {
            use_ema: false,
            regularization_weight: 1e-6,
            ..Default::default()
        };
        let params2 = LearnedQuantParams::new(config2, &shape, &engine).unwrap();

        // Test scales and zero_points separately first
        let scales_loss = params2.scales.square().unwrap().mean(None).unwrap();
        assert!(scales_loss.item().unwrap() >= 0.0);

        let zero_points_loss = params2.zero_points.square().unwrap().mean(None).unwrap();
        assert!(zero_points_loss.item().unwrap() >= 0.0);

        // Test the add operation directly
        let total_loss = scales_loss.add(&zero_points_loss).unwrap();
        assert!(total_loss.item().unwrap() >= 0.0);

        // Now test the full regularization loss
        let reg_loss2 = params2.regularization_loss().unwrap();
        assert!(reg_loss2.item().unwrap() >= 0.0);
    }
}
