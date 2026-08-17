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
            Tensor::F32(ref arr) => arr.iter().next().copied().unwrap_or(0.0),
            Tensor::F64(ref arr) => arr.iter().next().copied().unwrap_or(0.0) as f32,
            _ => 0.0,
        };
        let zero_points_mean_val = match zero_points_mean {
            Tensor::F32(ref arr) => arr.iter().next().copied().unwrap_or(0.0),
            Tensor::F64(ref arr) => arr.iter().next().copied().unwrap_or(0.0) as f32,
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

    /// Quantize and dequantize with learned parameters (fake quantization).
    ///
    /// In training mode the EMA statistics are updated afterwards; use
    /// [`LearnedFakeQuantize::fake_quantize`] for the pure, side-effect-free
    /// round trip.
    pub fn forward_fake_quantize(&mut self, input: &Variable) -> Result<Variable> {
        let dequantized = self.fake_quantize(input)?;

        // Update EMA parameters if in training mode
        if self.params.training {
            self.params.update_ema()?;
            self.params.apply_constraints()?;
        }

        Ok(dequantized)
    }

    /// Side-effect-free fake quantization round trip.
    ///
    /// `x -> (clamp(round(x / s + z), qmin, qmax) - z) * s`, with straight-through
    /// gradients through the rounding and the clamp so the learned scales and
    /// zero points stay trainable.
    pub fn fake_quantize(&self, input: &Variable) -> Result<Variable> {
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
        clamped.sub(&zero_points)?.mul(&scales)
    }

    /// Straight-through estimator for rounding.
    ///
    /// At the default temperature (`1.0`) this is the hard STE: forward rounds,
    /// backward is the identity. Any other temperature selects the smooth
    /// `soft_quantization` surrogate, whose gradient is exact for the surrogate
    /// it computes.
    fn straight_through_round(&self, input: &Variable) -> Result<Variable> {
        if self.params.config.ste_temperature == 1.0 {
            // Standard straight-through estimator
            self.round_with_straight_through(input)
        } else {
            // Soft quantization with temperature
            self.soft_quantization(input)
        }
    }

    /// Round with straight-through gradients.
    ///
    /// Registers a [`Variable::round_straight_through`] node so the rounding stays
    /// *in* the autograd graph: forward rounds, backward is the identity. The
    /// previous implementation built a brand-new leaf `Variable` from the rounded
    /// data, which severed the graph and left the learned scales and zero points
    /// with no gradient at all.
    fn round_with_straight_through(&self, input: &Variable) -> Result<Variable> {
        input.round_straight_through()
    }

    /// Soft quantization with temperature.
    ///
    /// `soft_round(x) = floor(x) + sigmoid((frac(x) - 0.5) / T)`, which converges
    /// to `round(x)` as `T -> 0` and is smooth for `T > 0`. `floor(x)` is a
    /// piecewise constant, so it is registered as a constant (its derivative is
    /// zero almost everywhere) and the gradient flows through `frac(x) = x -
    /// floor(x)` and the sigmoid.
    fn soft_quantization(&self, input: &Variable) -> Result<Variable> {
        let temp = self.params.config.ste_temperature;
        if !temp.is_finite() || temp <= 0.0 {
            return Err(TrustformersError::invalid_input(format!(
                "STE temperature must be positive, got {}",
                temp
            )));
        }

        // Constant floor: `floor` has zero derivative almost everywhere.
        let floor_tensor = input.data()?.floor()?;
        let floor_val = self.engine.variable(floor_tensor, false);

        // frac = x - floor(x), in [0, 1)
        let fraction = input.sub(&floor_val)?;
        let sigmoid_weight = fraction.sub_scalar(0.5)?.div_scalar(temp)?.sigmoid()?;

        // floor + w  ==  floor * (1 - w) + (floor + 1) * w
        floor_val.add(&sigmoid_weight)
    }

    /// Clamp values to the quantization range with a clipped STE gradient.
    ///
    /// Saturated entries receive no gradient (they cannot influence the loss);
    /// entries inside the range pass their gradient through. The previous
    /// implementation rebuilt a detached leaf and dropped the gradient entirely.
    fn clamp(&self, input: &Variable, min_val: f32, max_val: f32) -> Result<Variable> {
        input.clamp_straight_through(min_val, max_val)
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

    /// Run the real fake-quantization round trip.
    ///
    /// `Layer::forward` takes `&self`, so the EMA statistics cannot be updated
    /// here; use [`LearnedQuantLayer::fake_quant_mut`] +
    /// [`LearnedFakeQuantize::forward_fake_quantize`] when training with EMA.
    /// The quantization itself is the same computation either way -- this used
    /// to be `input * scale + zero_point`, i.e. an affine rescale that never
    /// quantized anything.
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        self.fake_quant.fake_quantize(&input)
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

        let params = LearnedQuantParams::new(config, &shape, &engine)
            .expect("Failed to create LearnedQuantParams");
        assert_eq!(
            params.scales.shape().expect("Failed to get scales shape"),
            vec![10]
        );
        assert_eq!(
            params.zero_points.shape().expect("Failed to get zero_points shape"),
            vec![10]
        );
    }

    #[test]
    fn test_learned_fake_quantize() {
        let config = LearnedQuantConfig {
            per_channel_learned: false, // Use per-tensor quantization to avoid shape issues
            ..Default::default()
        };
        let engine = Arc::new(AutodiffEngine::default());
        let shape = vec![5, 10];

        let mut fake_quant = LearnedFakeQuantize::new(config, &shape, 8, engine.clone())
            .expect("Failed to create LearnedFakeQuantize");

        let input_tensor = Tensor::randn(&[2, 5, 10]).expect("Failed to create random tensor");
        let input_var = engine.variable(input_tensor, true);

        let result = fake_quant.forward_fake_quantize(&input_var).expect("Forward pass failed");
        assert_eq!(
            result.shape().expect("Failed to get result shape"),
            vec![2, 5, 10]
        );
    }

    /// Regression test for the severed STE graph.
    ///
    /// `round_with_straight_through` used to build a brand-new leaf `Variable`
    /// from the rounded data, so no gradient ever reached the input or the
    /// learned scales. A gradient must arrive at the input after `backward`.
    #[test]
    fn straight_through_estimator_keeps_the_gradient_flowing_to_the_input() {
        let config = LearnedQuantConfig {
            per_channel_learned: false,
            ..Default::default()
        };
        let engine = Arc::new(AutodiffEngine::default());
        let shape = vec![4];

        let fake_quant = LearnedFakeQuantize::new(config, &shape, 8, engine.clone())
            .expect("build LearnedFakeQuantize");

        let input_tensor =
            Tensor::from_vec(vec![0.13f32, -0.42, 0.77, -0.05], &[4]).expect("input tensor");
        let input_var = engine.variable(input_tensor, true);

        let output = fake_quant.fake_quantize(&input_var).expect("fake quantize");
        output
            .backward_with_grad(Tensor::ones(&[4]).expect("upstream gradient"))
            .expect("backward");

        let grad = input_var
            .grad()
            .expect("gradient lookup")
            .expect("STE must deliver a gradient to the input");
        let values = grad.to_vec_f32().expect("gradient values");
        assert_eq!(values.len(), 4);
        assert!(
            values.iter().any(|v| v.abs() > 1e-6),
            "STE produced an all-zero gradient: {:?}",
            values
        );
        assert!(
            values.iter().all(|v| v.is_finite()),
            "STE produced a non-finite gradient: {:?}",
            values
        );
    }

    /// Regression test for `LearnedQuantLayer::forward`, which used to compute
    /// `input * scale + zero_point` -- an affine rescale that never rounded, so
    /// a fine-grained ramp came out unchanged up to that affine map.
    #[test]
    fn quant_layer_forward_actually_quantizes() {
        let config = LearnedQuantConfig {
            per_channel_learned: false,
            ..Default::default()
        };
        let engine = Arc::new(AutodiffEngine::default());
        let shape = vec![64];

        let layer = LearnedQuantLayer::new(
            "test".to_string(),
            config,
            &shape,
            4, // 4 bits => a coarse grid, so distinct inputs must collide
            engine.clone(),
        )
        .expect("build LearnedQuantLayer");

        // A dense ramp inside one quantization step: a real quantizer maps many
        // of these onto the same reconstruction level.
        let values: Vec<f32> = (0..64).map(|i| i as f32 * 1e-3).collect();
        let input_var = engine.variable(
            Tensor::from_vec(values.clone(), &[64]).expect("input"),
            true,
        );

        let output = layer.forward(input_var).expect("layer forward");
        let out_values = output.data().expect("output tensor").to_vec_f32().expect("values");
        assert_eq!(out_values.len(), 64);

        let mut distinct = out_values.clone();
        distinct.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        distinct.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        assert!(
            distinct.len() < out_values.len(),
            "output has {} distinct levels for {} inputs -- nothing was quantized",
            distinct.len(),
            out_values.len()
        );
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

        let mut params = LearnedQuantParams::new(config, &shape, &engine)
            .expect("Failed to create LearnedQuantParams");

        // Set scales outside bounds
        let bad_scales = Tensor::from_vec(vec![0.01, 100.0, 1.0, 0.05, 50.0], &[5])
            .expect("Tensor from_vec failed");
        params.scales.set_data(bad_scales).expect("Failed to set scales data");

        params.apply_constraints().expect("Failed to apply constraints");

        let constrained_scales = params
            .scales
            .data()
            .expect("Failed to get scales data")
            .to_vec_f32()
            .expect("Failed to convert to vec_f32");
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

        let mut params = LearnedQuantParams::new(config, &shape, &engine)
            .expect("Failed to create LearnedQuantParams");

        // Set initial values
        let new_scales =
            Tensor::from_vec(vec![2.0, 3.0, 4.0], &[3]).expect("Tensor from_vec failed");
        params.scales.set_data(new_scales).expect("Failed to set scales data");

        params.update_ema().expect("Failed to update EMA");

        // Check that EMA was updated
        let ema_scales = params
            .ema_scales
            .as_ref()
            .expect("EMA scales not found")
            .data()
            .expect("Failed to get EMA data")
            .to_vec_f32()
            .expect("Failed to convert to vec_f32");
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

        let params = LearnedQuantParams::new(config, &shape, &engine)
            .expect("Failed to create LearnedQuantParams");

        let reg_loss = params.regularization_loss().expect("Failed to compute regularization loss");
        assert_eq!(reg_loss.item().expect("Failed to get item value"), 0.0);

        // Now test non-zero weight
        let config2 = LearnedQuantConfig {
            use_ema: false,
            regularization_weight: 1e-6,
            ..Default::default()
        };
        let params2 = LearnedQuantParams::new(config2, &shape, &engine)
            .expect("Failed to create LearnedQuantParams");

        // Test scales and zero_points separately first
        let scales_loss = params2
            .scales
            .square()
            .expect("Failed to square")
            .mean(None)
            .expect("Mean calculation failed");
        assert!(scales_loss.item().expect("Failed to get item value") >= 0.0);

        let zero_points_loss = params2
            .zero_points
            .square()
            .expect("Failed to square")
            .mean(None)
            .expect("Mean calculation failed");
        assert!(zero_points_loss.item().expect("Failed to get item value") >= 0.0);

        // Test the add operation directly
        let total_loss = scales_loss.add(&zero_points_loss).expect("Addition failed");
        assert!(total_loss.item().expect("Failed to get item value") >= 0.0);

        // Now test the full regularization loss
        let reg_loss2 =
            params2.regularization_loss().expect("Failed to compute regularization loss");
        assert!(reg_loss2.item().expect("Failed to get item value") >= 0.0);
    }
}
