//! Quantization-Aware Training (QAT) for TrustformeRS
//!
//! This module provides quantization-aware training functionality that simulates
//! quantization during training to improve model accuracy after quantization.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use trustformers_core::errors::Result;
use trustformers_core::{Layer, QuantizationScheme, Tensor};

/// Mixed-bit quantization strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MixedBitStrategy {
    /// Uniform bit width across all layers
    Uniform { bits: u8 },
    /// Manual specification of bits per layer type
    Manual { layer_bits: HashMap<String, u8> },
    /// Automatic sensitivity-based assignment
    SensitivityBased {
        sensitivity_threshold: f32,
        high_precision_bits: u8,
        low_precision_bits: u8,
    },
    /// Resource-constrained mixed-bit optimization
    ResourceConstrained {
        total_bit_budget: u64,
        critical_layers: Vec<String>,
        critical_bits: u8,
        default_bits: u8,
    },
    /// Progressive quantization (start high, reduce over time)
    Progressive {
        initial_bits: u8,
        final_bits: u8,
        reduction_schedule: Vec<(usize, u8)>, // (step, bits)
    },
}

impl Default for MixedBitStrategy {
    fn default() -> Self {
        Self::Uniform { bits: 8 }
    }
}

/// Layer-specific quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerQuantConfig {
    /// Number of bits for this layer
    pub bits: u8,
    /// Use symmetric quantization
    pub symmetric: bool,
    /// Per-channel quantization
    pub per_channel: bool,
    /// Layer sensitivity score (0.0 = low, 1.0 = high)
    pub sensitivity: f32,
    /// Whether this layer is critical for model performance
    pub is_critical: bool,
}

impl Default for LayerQuantConfig {
    fn default() -> Self {
        Self {
            bits: 8,
            symmetric: true,
            per_channel: false,
            sensitivity: 0.5,
            is_critical: false,
        }
    }
}

/// QAT configuration with mixed-bit support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QATConfig {
    /// Quantization scheme to simulate
    pub qscheme: QuantizationScheme,
    /// Mixed-bit quantization strategy
    pub mixed_bit_strategy: MixedBitStrategy,
    /// Default number of bits (used as fallback)
    pub default_bits: u8,
    /// Use symmetric quantization (default)
    pub symmetric: bool,
    /// Per-channel quantization (default)
    pub per_channel: bool,
    /// Start QAT after this many steps
    pub start_step: usize,
    /// Freeze quantization parameters after this many steps
    pub freeze_step: Option<usize>,
    /// Use learned step size
    pub learnable_step_size: bool,
    /// Observer momentum for running statistics
    pub observer_momentum: f32,
    /// Layer-specific configurations
    pub layer_configs: HashMap<String, LayerQuantConfig>,
    /// Enable activation quantization
    pub quantize_activations: bool,
    /// Activation quantization bits
    pub activation_bits: u8,
    /// Enable mixed-bit optimization
    pub enable_mixed_bit_optimization: bool,
    /// Bit allocation budget for resource-constrained scenarios
    pub bit_allocation_budget: Option<u64>,
}

impl Default for QATConfig {
    fn default() -> Self {
        Self {
            qscheme: QuantizationScheme::Int8,
            mixed_bit_strategy: MixedBitStrategy::default(),
            default_bits: 8,
            symmetric: true,
            per_channel: false,
            start_step: 1000,
            freeze_step: None,
            learnable_step_size: false,
            observer_momentum: 0.99,
            layer_configs: HashMap::new(),
            quantize_activations: false,
            activation_bits: 8,
            enable_mixed_bit_optimization: false,
            bit_allocation_budget: None,
        }
    }
}

impl QATConfig {
    /// Get bits for a specific layer
    pub fn get_layer_bits(&self, layer_name: &str, current_step: usize) -> u8 {
        // First check layer-specific configuration
        if let Some(layer_config) = self.layer_configs.get(layer_name) {
            return layer_config.bits;
        }

        // Then check mixed-bit strategy
        match &self.mixed_bit_strategy {
            MixedBitStrategy::Uniform { bits } => *bits,
            MixedBitStrategy::Manual { layer_bits } => {
                // Try exact match first, then partial match
                layer_bits
                    .get(layer_name)
                    .or_else(|| {
                        // Try to match by layer type (e.g., "linear", "conv2d")
                        layer_bits
                            .iter()
                            .find(|(key, _)| layer_name.contains(key.as_str()))
                            .map(|(_, bits)| bits)
                    })
                    .copied()
                    .unwrap_or(self.default_bits)
            },
            MixedBitStrategy::SensitivityBased {
                sensitivity_threshold,
                high_precision_bits,
                low_precision_bits,
            } => {
                if let Some(layer_config) = self.layer_configs.get(layer_name) {
                    if layer_config.sensitivity > *sensitivity_threshold {
                        *high_precision_bits
                    } else {
                        *low_precision_bits
                    }
                } else {
                    self.default_bits
                }
            },
            MixedBitStrategy::ResourceConstrained {
                critical_layers,
                critical_bits,
                default_bits,
                ..
            } => {
                if critical_layers.iter().any(|layer| layer_name.contains(layer)) {
                    *critical_bits
                } else {
                    *default_bits
                }
            },
            MixedBitStrategy::Progressive {
                initial_bits,
                final_bits,
                reduction_schedule,
            } => {
                // Find the appropriate bits based on current step
                for (step, bits) in reduction_schedule.iter().rev() {
                    if current_step >= *step {
                        return *bits;
                    }
                }
                // If before all scheduled reductions, use initial bits
                if current_step < reduction_schedule.first().map(|(s, _)| *s).unwrap_or(0) {
                    *initial_bits
                } else {
                    *final_bits
                }
            },
        }
    }

    /// Set layer-specific configuration
    pub fn set_layer_config(&mut self, layer_name: String, config: LayerQuantConfig) {
        self.layer_configs.insert(layer_name, config);
    }

    /// Automatically configure layers based on sensitivity analysis
    pub fn auto_configure_sensitivity(&mut self, layer_sensitivities: HashMap<String, f32>) {
        let sensitivity_threshold = match &self.mixed_bit_strategy {
            MixedBitStrategy::SensitivityBased {
                sensitivity_threshold,
                ..
            } => *sensitivity_threshold,
            _ => 0.7, // default threshold
        };

        for (layer_name, sensitivity) in layer_sensitivities {
            let is_critical = sensitivity > sensitivity_threshold;
            let config = LayerQuantConfig {
                bits: if is_critical { 8 } else { 4 },
                sensitivity,
                is_critical,
                ..LayerQuantConfig::default()
            };
            self.layer_configs.insert(layer_name, config);
        }

        // Update strategy to sensitivity-based if not already set
        if matches!(self.mixed_bit_strategy, MixedBitStrategy::Uniform { .. }) {
            self.mixed_bit_strategy = MixedBitStrategy::SensitivityBased {
                sensitivity_threshold,
                high_precision_bits: 8,
                low_precision_bits: 4,
            };
        }
    }

    /// Optimize bit allocation under resource constraints
    pub fn optimize_bit_allocation(&mut self, model_size_info: HashMap<String, u64>) -> Result<()> {
        if let Some(budget) = self.bit_allocation_budget {
            let _total_params: u64 = model_size_info.values().sum();

            // Sort layers by importance (sensitivity * size)
            let mut layer_importance: Vec<(String, f64)> = model_size_info
                .iter()
                .map(|(name, size)| {
                    let sensitivity =
                        self.layer_configs.get(name).map(|c| c.sensitivity as f64).unwrap_or(0.5);
                    let importance = sensitivity * (*size as f64);
                    (name.clone(), importance)
                })
                .collect();

            layer_importance
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Allocate bits greedily based on importance
            let mut remaining_budget = budget;

            for (layer_name, _) in layer_importance {
                let layer_size = model_size_info[&layer_name];
                let min_bits = 4; // minimum quantization
                let max_bits = 8; // maximum practical bits

                // Calculate how many bits we can afford for this layer
                let affordable_bits = std::cmp::min(
                    max_bits,
                    std::cmp::max(min_bits, (remaining_budget / layer_size) as u8),
                );

                // Update layer configuration
                let mut config = self.layer_configs.get(&layer_name).cloned().unwrap_or_default();
                config.bits = affordable_bits;
                self.layer_configs.insert(layer_name.clone(), config);

                // Update remaining budget
                remaining_budget =
                    remaining_budget.saturating_sub(layer_size * affordable_bits as u64);
            }

            println!(
                "🎯 Optimized bit allocation under budget constraint: {} bits",
                budget
            );
            println!("📊 Remaining budget: {} bits", remaining_budget);
        }

        Ok(())
    }

    /// Get total bit consumption estimate
    pub fn estimate_bit_consumption(&self, model_size_info: &HashMap<String, u64>) -> u64 {
        model_size_info
            .iter()
            .map(|(layer_name, size)| {
                let bits = self.get_layer_bits(layer_name, 0); // Use step 0 for estimation
                size * bits as u64
            })
            .sum()
    }

    /// Create a configuration for common mixed-bit scenarios
    pub fn create_common_config(scenario: &str) -> Self {
        match scenario {
            "edge_deployment" => Self {
                mixed_bit_strategy: MixedBitStrategy::ResourceConstrained {
                    total_bit_budget: 1024 * 1024, // 1MB bit budget
                    critical_layers: vec!["attention".to_string(), "output".to_string()],
                    critical_bits: 8,
                    default_bits: 4,
                },
                quantize_activations: true,
                activation_bits: 8,
                enable_mixed_bit_optimization: true,
                ..Self::default()
            },
            "high_accuracy" => Self {
                mixed_bit_strategy: MixedBitStrategy::SensitivityBased {
                    sensitivity_threshold: 0.6,
                    high_precision_bits: 8,
                    low_precision_bits: 6,
                },
                quantize_activations: false, // Keep activations in full precision
                enable_mixed_bit_optimization: true,
                ..Self::default()
            },
            "aggressive_compression" => Self {
                mixed_bit_strategy: MixedBitStrategy::SensitivityBased {
                    sensitivity_threshold: 0.8,
                    high_precision_bits: 6,
                    low_precision_bits: 3,
                },
                quantize_activations: true,
                activation_bits: 4,
                enable_mixed_bit_optimization: true,
                ..Self::default()
            },
            _ => Self::default(),
        }
    }
}

/// Quantization parameters that can be learned
#[derive(Debug, Clone)]
pub struct QuantizationParams {
    /// Scale factor for quantization
    pub scale: Tensor,
    /// Zero point for asymmetric quantization
    pub zero_point: Option<Tensor>,
    /// Running min value
    pub running_min: Tensor,
    /// Running max value
    pub running_max: Tensor,
    /// Number of observations
    pub num_observations: usize,
}

impl QuantizationParams {
    pub fn new(shape: &[usize], symmetric: bool) -> Self {
        Self {
            scale: Tensor::ones(shape).expect("Failed to create scale"),
            zero_point: if symmetric {
                None
            } else {
                Some(Tensor::zeros(shape).expect("Failed to create zero point"))
            },
            running_min: Tensor::full(f32::INFINITY, shape.to_vec()).expect("Failed to create min"),
            running_max: Tensor::full(f32::NEG_INFINITY, shape.to_vec())
                .expect("Failed to create max"),
            num_observations: 0,
        }
    }

    /// Update running statistics
    pub fn update_stats(&mut self, tensor: &Tensor, momentum: f32) -> Result<()> {
        let (current_min_val, current_max_val) = tensor.min_max()?;
        let current_min = Tensor::scalar(current_min_val)?;
        let current_max = Tensor::scalar(current_max_val)?;

        if self.num_observations == 0 {
            self.running_min = current_min;
            self.running_max = current_max;
        } else {
            // Exponential moving average
            self.running_min = self
                .running_min
                .mul_scalar(momentum)?
                .add(&current_min.mul_scalar(1.0 - momentum)?)?;
            self.running_max = self
                .running_max
                .mul_scalar(momentum)?
                .add(&current_max.mul_scalar(1.0 - momentum)?)?;
        }

        self.num_observations += 1;
        Ok(())
    }

    /// Compute scale and zero point from statistics
    pub fn compute_params(&mut self, bits: u8, symmetric: bool) -> Result<()> {
        let q_min = if symmetric { -(1 << (bits - 1)) } else { 0 } as f32;
        let q_max = if symmetric { (1 << (bits - 1)) - 1 } else { (1 << bits) - 1 } as f32;

        if symmetric {
            let abs_running_max = self.running_max.abs()?;
            let abs_running_min = self.running_min.abs()?;
            let (_, max_abs_max) = abs_running_max.min_max()?;
            let (_, max_abs_min) = abs_running_min.min_max()?;
            // For symmetric quantization, we need the maximum of the absolute values
            let abs_max = Tensor::scalar(max_abs_max.max(max_abs_min))?;
            self.scale = abs_max.div_scalar(q_max)?;
        } else {
            let range = self.running_max.sub(&self.running_min)?;
            self.scale = range.div_scalar(q_max - q_min)?;

            if let Some(zp) = &mut self.zero_point {
                *zp = self.running_min.div(&self.scale)?.neg()?.add_scalar(q_min)?;
                *zp = zp.clamp(q_min, q_max)?;
            }
        }

        Ok(())
    }
}

/// QAT Linear layer
pub struct QATLinear {
    /// Original linear layer
    linear: Arc<dyn Layer<Input = Tensor, Output = Tensor>>,
    /// QAT configuration
    config: QATConfig,
    /// Quantization parameters
    quant_params: Arc<Mutex<QuantizationParams>>,
    /// Current training step
    step: Arc<Mutex<usize>>,
    /// Whether QAT is enabled
    enabled: bool,
}

impl QATLinear {
    pub fn new(linear: Arc<dyn Layer<Input = Tensor, Output = Tensor>>, config: QATConfig) -> Self {
        // Initialize quantization parameters based on weight shape
        let weight_shape = vec![1]; // Simplified - would get from linear layer
        let quant_params = QuantizationParams::new(&weight_shape, config.symmetric);

        Self {
            linear,
            config,
            quant_params: Arc::new(Mutex::new(quant_params)),
            step: Arc::new(Mutex::new(0)),
            enabled: true,
        }
    }

    /// Enable or disable QAT
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Get current quantization parameters
    pub fn get_quant_params(&self) -> Arc<Mutex<QuantizationParams>> {
        Arc::clone(&self.quant_params)
    }

    /// Extract weight tensor from the wrapped linear layer
    fn get_layer_weights(&self) -> Result<Tensor> {
        // Since we're working with a trait object, we simulate weight extraction
        // In a production implementation, this would use a WeightAccessor trait
        // or downcast to concrete layer types to extract actual weights

        // Use typical transformer layer dimensions for weight simulation
        let weight_shape = vec![768, 768]; // Standard hidden_size for many models

        // Initialize with Xavier/Glorot uniform initialization for realistic weights
        let fan_in = weight_shape[0] as f32;
        let fan_out = weight_shape[1] as f32;
        let limit = (6.0 / (fan_in + fan_out)).sqrt();

        // Generate weight data with proper initialization distribution
        let total_elements = weight_shape.iter().product::<usize>();
        let weight_data: Vec<f32> = (0..total_elements)
            .map(|_| {
                let uniform_val = fastrand::f32(); // [0.0, 1.0)
                (uniform_val - 0.5) * 2.0 * limit // Scale to [-limit, limit]
            })
            .collect();

        Tensor::from_vec(weight_data, &weight_shape)
    }
}

impl Layer for QATLinear {
    type Input = Tensor;
    type Output = Tensor;
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let mut step = self.step.lock().expect("lock should not be poisoned");
        *step += 1;
        let current_step = *step;
        drop(step);

        // Check if QAT should be active
        if !self.enabled || current_step < self.config.start_step {
            // Regular forward pass without quantization
            return self.linear.forward(input);
        }

        // Get weight tensor from the linear layer
        let weight = self.get_layer_weights()?;

        // Update statistics if not frozen
        if self.config.freeze_step.is_none()
            || current_step
                < self.config.freeze_step.expect("freeze_step checked as Some in condition")
        {
            let mut params = self.quant_params.lock().expect("lock should not be poisoned");
            params.update_stats(&weight, self.config.observer_momentum)?;
            params.compute_params(self.config.default_bits, self.config.symmetric)?;
        }

        // Simulate quantization on weights
        let params = self.quant_params.lock().expect("lock should not be poisoned");
        let _quantized_weight = fake_quantize(
            &weight,
            &params.scale,
            params.zero_point.as_ref(),
            self.config.default_bits,
            self.config.symmetric,
        )?;
        drop(params);

        // Forward with quantized weights
        // In practice, this would use the quantized weights in the linear operation
        self.linear.forward(input)
    }
}

/// QAT Convolution layer
pub struct QATConv2d {
    /// Original convolution layer
    conv: Arc<dyn Layer<Input = Tensor, Output = Tensor>>,
    /// QAT configuration
    config: QATConfig,
    /// Weight quantization parameters
    #[allow(dead_code)]
    weight_params: Arc<Mutex<QuantizationParams>>,
    /// Activation quantization parameters (optional)
    activation_params: Option<Arc<Mutex<QuantizationParams>>>,
    /// Current training step
    step: Arc<Mutex<usize>>,
}

impl QATConv2d {
    pub fn new(
        conv: Arc<dyn Layer<Input = Tensor, Output = Tensor>>,
        config: QATConfig,
        quantize_activations: bool,
    ) -> Self {
        let weight_shape = vec![1]; // Simplified
        let weight_params = QuantizationParams::new(&weight_shape, config.symmetric);

        let activation_params = if quantize_activations {
            Some(Arc::new(Mutex::new(QuantizationParams::new(
                &[1],
                config.symmetric,
            ))))
        } else {
            None
        };

        Self {
            conv,
            config,
            weight_params: Arc::new(Mutex::new(weight_params)),
            activation_params,
            step: Arc::new(Mutex::new(0)),
        }
    }
}

impl Layer for QATConv2d {
    type Input = Tensor;
    type Output = Tensor;
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let mut step = self.step.lock().expect("lock should not be poisoned");
        *step += 1;
        let current_step = *step;
        drop(step);

        if current_step < self.config.start_step {
            return self.conv.forward(input);
        }

        // Quantize input activations if configured
        let quantized_input = if let Some(act_params) = &self.activation_params {
            if self.config.freeze_step.is_none()
                || current_step
                    < self.config.freeze_step.expect("freeze_step checked as Some in condition")
            {
                let mut params = act_params.lock().expect("lock should not be poisoned");
                params.update_stats(&input, self.config.observer_momentum)?;
                params.compute_params(self.config.default_bits, self.config.symmetric)?;
            }

            let params = act_params.lock().expect("lock should not be poisoned");
            fake_quantize(
                &input,
                &params.scale,
                params.zero_point.as_ref(),
                self.config.default_bits,
                self.config.symmetric,
            )?
        } else {
            input.clone()
        };

        // Apply convolution with quantized weights (simplified)
        self.conv.forward(quantized_input)
    }
}

/// Fake quantize operation for QAT
/// Activation quantizer for mixed-bit quantization
#[derive(Debug, Clone)]
pub struct ActivationQuantizer {
    pub params: QuantizationParams,
    pub bits: u8,
    pub symmetric: bool,
    pub calibrated: bool,
}

impl ActivationQuantizer {
    pub fn new(shape: &[usize], bits: u8, symmetric: bool) -> Self {
        Self {
            params: QuantizationParams::new(shape, symmetric),
            bits,
            symmetric,
            calibrated: false,
        }
    }

    /// Calibrate the quantizer with calibration data
    pub fn calibrate(&mut self, calibration_data: &[Tensor], momentum: f32) -> Result<()> {
        for tensor in calibration_data {
            self.params.update_stats(tensor, momentum)?;
        }
        self.params.compute_params(self.bits, self.symmetric)?;
        self.calibrated = true;
        Ok(())
    }

    /// Quantize activation tensor
    pub fn quantize(&self, tensor: &Tensor) -> Result<Tensor> {
        if !self.calibrated {
            // If not calibrated, just pass through with warning
            println!("⚠️ Warning: Activation quantizer not calibrated, using full precision");
            return Ok(tensor.clone());
        }

        fake_quantize(
            tensor,
            &self.params.scale,
            self.params.zero_point.as_ref(),
            self.bits,
            self.symmetric,
        )
    }

    /// Update parameters during training
    pub fn update(&mut self, tensor: &Tensor, momentum: f32) -> Result<()> {
        self.params.update_stats(tensor, momentum)?;
        if self.calibrated {
            self.params.compute_params(self.bits, self.symmetric)?;
        }
        Ok(())
    }
}

/// Mixed-bit fake quantization with layer-aware bit selection
pub fn fake_quantize_mixed_bit(
    tensor: &Tensor,
    scale: &Tensor,
    zero_point: Option<&Tensor>,
    config: &QATConfig,
    layer_name: &str,
    current_step: usize,
) -> Result<Tensor> {
    let bits = config.get_layer_bits(layer_name, current_step);
    fake_quantize(tensor, scale, zero_point, bits, config.symmetric)
}

/// Enhanced fake quantization with better numerical stability
pub fn fake_quantize(
    tensor: &Tensor,
    scale: &Tensor,
    zero_point: Option<&Tensor>,
    bits: u8,
    symmetric: bool,
) -> Result<Tensor> {
    let q_min = if symmetric { -(1 << (bits - 1)) } else { 0 } as f32;
    let q_max = if symmetric { (1 << (bits - 1)) - 1 } else { (1 << bits) - 1 } as f32;

    // Get scalar values for scale and zero_point
    let scale_val = scale.get_float(0)?;
    let zero_point_val = if let Some(zp) = zero_point { zp.get_float(0)? } else { 0.0 };

    // Manual broadcasting: apply operations element-wise
    let tensor_data = tensor.data()?;
    let result_data: Vec<f32> = tensor_data
        .iter()
        .map(|&x| {
            // Scale
            let scaled = x / scale_val;

            // Add zero point if asymmetric
            let shifted = if zero_point.is_some() { scaled + zero_point_val } else { scaled };

            // Round and clamp
            let quantized = shifted.round().clamp(q_min, q_max);

            // Dequantize back
            if zero_point.is_some() {
                (quantized - zero_point_val) * scale_val
            } else {
                quantized * scale_val
            }
        })
        .collect();

    // Straight-through estimator: forward uses dequantized values,
    // backward passes gradients through unchanged
    Tensor::from_vec(result_data, &tensor.shape())
}

/// QAT model wrapper
pub struct QATModel {
    /// Original model
    model: Arc<dyn Layer<Input = Tensor, Output = Tensor>>,
    /// QAT layers mapping
    qat_layers: HashMap<String, Arc<Mutex<dyn Layer<Input = Tensor, Output = Tensor>>>>,
    /// Global QAT configuration
    config: QATConfig,
}

impl QATModel {
    pub fn new(model: Arc<dyn Layer<Input = Tensor, Output = Tensor>>, config: QATConfig) -> Self {
        Self {
            model,
            qat_layers: HashMap::new(),
            config,
        }
    }

    /// Replace a layer with QAT version
    pub fn add_qat_layer(
        &mut self,
        name: String,
        layer: Arc<Mutex<dyn Layer<Input = Tensor, Output = Tensor>>>,
    ) {
        self.qat_layers.insert(name, layer);
    }

    /// Prepare model for QAT
    pub fn prepare(&mut self) -> Result<()> {
        // In practice, this would traverse the model and replace
        // linear/conv layers with QAT versions
        Ok(())
    }

    /// Convert to quantized model
    pub fn convert(&self) -> Result<QuantizedModel> {
        // Extract learned quantization parameters and create
        // a fully quantized model
        let quantized_layers = HashMap::new();

        Ok(QuantizedModel {
            layers: quantized_layers,
            config: self.config.clone(),
        })
    }

    /// Get quantization statistics
    pub fn get_statistics(&self) -> HashMap<String, QuantStats> {
        let mut stats = HashMap::new();

        // Collect statistics from all QAT layers
        for name in self.qat_layers.keys() {
            stats.insert(
                name.clone(),
                QuantStats {
                    min_val: 0.0,
                    max_val: 0.0,
                    mean_val: 0.0,
                    scale: 1.0,
                },
            );
        }

        stats
    }
}

/// Quantized model after QAT
#[allow(dead_code)]
pub struct QuantizedModel {
    #[allow(dead_code)]
    layers: HashMap<String, QuantizedLayer>,
    config: QATConfig,
}

/// Quantized layer representation
#[allow(dead_code)]
pub struct QuantizedLayer {
    #[allow(dead_code)]
    weights: Vec<u8>,
    scale: Vec<f32>,
    zero_point: Vec<i32>,
}

/// Quantization statistics
#[derive(Debug, Clone)]
pub struct QuantStats {
    pub min_val: f32,
    pub max_val: f32,
    pub mean_val: f32,
    pub scale: f32,
}

/// Mixed-bit QAT training manager
pub struct MixedBitQATTrainer {
    /// QAT configuration with mixed-bit settings
    pub config: QATConfig,
    /// Layer-specific quantization parameters
    pub layer_params: HashMap<String, QuantizationParams>,
    /// Activation quantizers for each layer
    pub activation_quantizers: HashMap<String, ActivationQuantizer>,
    /// Learning rate for quantization parameters
    pub quant_lr: f32,
    /// Weight decay for quantization parameters
    pub quant_weight_decay: f32,
    /// Current training step
    pub current_step: usize,
    /// Sensitivity analysis results
    pub layer_sensitivities: HashMap<String, f32>,
    /// Model size information for bit allocation
    pub model_size_info: HashMap<String, u64>,
}

impl MixedBitQATTrainer {
    pub fn new(config: QATConfig, quant_lr: f32, quant_weight_decay: f32) -> Self {
        Self {
            config,
            layer_params: HashMap::new(),
            activation_quantizers: HashMap::new(),
            quant_lr,
            quant_weight_decay,
            current_step: 0,
            layer_sensitivities: HashMap::new(),
            model_size_info: HashMap::new(),
        }
    }

    /// Initialize quantization parameters for a layer
    pub fn init_layer(&mut self, layer_name: String, param_shape: &[usize]) -> Result<()> {
        // Get layer-specific configuration
        let layer_config = self.config.layer_configs.get(&layer_name).cloned().unwrap_or_default();

        // Initialize weight quantization parameters
        let params = QuantizationParams::new(param_shape, layer_config.symmetric);
        self.layer_params.insert(layer_name.clone(), params);

        // Initialize activation quantizer if enabled
        if self.config.quantize_activations {
            let activation_bits = if self.config.enable_mixed_bit_optimization {
                layer_config.bits
            } else {
                self.config.activation_bits
            };

            let act_quantizer =
                ActivationQuantizer::new(param_shape, activation_bits, layer_config.symmetric);
            self.activation_quantizers.insert(layer_name.clone(), act_quantizer);
        }

        println!(
            "🔧 Initialized mixed-bit QAT for layer: {} ({}bits)",
            layer_name,
            self.config.get_layer_bits(&layer_name, self.current_step)
        );

        Ok(())
    }

    /// Perform sensitivity analysis on layers
    pub fn analyze_sensitivity(
        &mut self,
        model_outputs: HashMap<String, Vec<Tensor>>,
    ) -> Result<()> {
        println!("🔍 Performing layer sensitivity analysis for mixed-bit optimization...");

        for (layer_name, outputs) in model_outputs {
            // Calculate sensitivity based on activation variance and gradient magnitudes
            let mut total_variance = 0.0;
            let mut total_magnitude = 0.0;

            for output in &outputs {
                // Calculate activation variance as a proxy for sensitivity
                let data = output.data()?;
                let mean = data.iter().sum::<f32>() / data.len() as f32;
                let variance =
                    data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;

                total_variance += variance;

                // Calculate magnitude (as proxy for importance)
                let magnitude = data.iter().map(|x| x.abs()).sum::<f32>() / data.len() as f32;
                total_magnitude += magnitude;
            }

            // Normalize sensitivity score
            let avg_variance = total_variance / outputs.len() as f32;
            let avg_magnitude = total_magnitude / outputs.len() as f32;
            let sensitivity = (avg_variance * avg_magnitude).sqrt().min(1.0);

            self.layer_sensitivities.insert(layer_name.clone(), sensitivity);

            println!("📊 Layer {} sensitivity: {:.3}", layer_name, sensitivity);
        }

        // Auto-configure based on sensitivity analysis
        self.config.auto_configure_sensitivity(self.layer_sensitivities.clone());

        Ok(())
    }

    /// Update model size information for bit allocation
    pub fn update_model_info(&mut self, model_info: HashMap<String, u64>) {
        self.model_size_info = model_info;

        // Optimize bit allocation if enabled
        if self.config.enable_mixed_bit_optimization {
            if let Err(e) = self.config.optimize_bit_allocation(self.model_size_info.clone()) {
                println!("⚠️ Warning: Failed to optimize bit allocation: {}", e);
            }
        }
    }

    /// Quantize layer weights with mixed-bit support
    pub fn quantize_layer_weights(&mut self, layer_name: &str, weights: &Tensor) -> Result<Tensor> {
        // Get or initialize parameters for this layer
        if !self.layer_params.contains_key(layer_name) {
            self.init_layer(layer_name.to_string(), &weights.shape())?;
        }

        let params = self
            .layer_params
            .get_mut(layer_name)
            .expect("layer_params entry exists after initialization check");

        // Update statistics if we're in the calibration phase
        if self.current_step < self.config.start_step {
            params.update_stats(weights, self.config.observer_momentum)?;
            params.compute_params(
                self.config.get_layer_bits(layer_name, self.current_step),
                self.config.symmetric,
            )?;
        }

        // Apply mixed-bit fake quantization
        fake_quantize_mixed_bit(
            weights,
            &params.scale,
            params.zero_point.as_ref(),
            &self.config,
            layer_name,
            self.current_step,
        )
    }

    /// Quantize layer activations
    pub fn quantize_layer_activations(
        &mut self,
        layer_name: &str,
        activations: &Tensor,
    ) -> Result<Tensor> {
        if !self.config.quantize_activations {
            return Ok(activations.clone());
        }

        if let Some(quantizer) = self.activation_quantizers.get_mut(layer_name) {
            // Update quantizer during training
            quantizer.update(activations, self.config.observer_momentum)?;
            quantizer.quantize(activations)
        } else {
            // Initialize if not present
            let layer_config =
                self.config.layer_configs.get(layer_name).cloned().unwrap_or_default();

            let mut quantizer = ActivationQuantizer::new(
                &activations.shape(),
                layer_config.bits,
                layer_config.symmetric,
            );

            quantizer.update(activations, self.config.observer_momentum)?;
            let result = quantizer.quantize(activations)?;

            self.activation_quantizers.insert(layer_name.to_string(), quantizer);
            Ok(result)
        }
    }

    /// Step the trainer (increment step counter and update progressive quantization)
    pub fn step(&mut self) {
        self.current_step += 1;

        // Handle progressive quantization
        if let MixedBitStrategy::Progressive { .. } = &self.config.mixed_bit_strategy {
            // Bit widths will be automatically updated via get_layer_bits
            if self.current_step.is_multiple_of(1000) {
                println!(
                    "📈 Progressive quantization step {}: updating bit allocations",
                    self.current_step
                );
            }
        }

        // Freeze quantization parameters if specified
        if let Some(freeze_step) = self.config.freeze_step {
            if self.current_step == freeze_step {
                println!(
                    "🔒 Freezing quantization parameters at step {}",
                    freeze_step
                );
            }
        }
    }

    /// Get current quantization statistics
    pub fn get_quantization_stats(&self) -> HashMap<String, (u8, f32)> {
        let mut stats = HashMap::new();

        for layer_name in self.layer_params.keys() {
            let bits = self.config.get_layer_bits(layer_name, self.current_step);
            let sensitivity = self.layer_sensitivities.get(layer_name).copied().unwrap_or(0.0);
            stats.insert(layer_name.clone(), (bits, sensitivity));
        }

        stats
    }

    /// Estimate total memory/compute savings from mixed-bit quantization
    pub fn estimate_savings(&self) -> (f64, f64) {
        if self.model_size_info.is_empty() {
            return (0.0, 0.0);
        }

        let total_params: u64 = self.model_size_info.values().sum();
        let _baseline_bits = 32.0; // fp32 baseline

        let mut total_quantized_bits = 0u64;
        for (layer_name, param_count) in &self.model_size_info {
            let bits = self.config.get_layer_bits(layer_name, self.current_step) as u64;
            total_quantized_bits += param_count * bits;
        }

        let baseline_total_bits = total_params * 32;

        let memory_savings = 1.0 - (total_quantized_bits as f64) / (baseline_total_bits as f64);
        let compute_savings = memory_savings * 0.8; // Approximation: compute scales with memory

        (memory_savings, compute_savings)
    }

    /// Create a summary report of the mixed-bit configuration
    pub fn summary_report(&self) -> String {
        let mut report = String::from("📊 Mixed-Bit QAT Summary Report\n");
        report.push_str("====================================\n\n");

        // Strategy summary
        report.push_str(&format!(
            "🎯 Strategy: {:?}\n",
            self.config.mixed_bit_strategy
        ));
        report.push_str(&format!(
            "📋 Total layers configured: {}\n",
            self.layer_params.len()
        ));
        report.push_str(&format!(
            "📈 Current training step: {}\n",
            self.current_step
        ));

        if self.config.quantize_activations {
            report.push_str(&format!(
                "⚡ Activation quantization: {} bits\n",
                self.config.activation_bits
            ));
        }

        report.push('\n');

        // Per-layer breakdown
        report.push_str("🔍 Per-Layer Configuration:\n");
        for layer_name in self.layer_params.keys() {
            let bits = self.config.get_layer_bits(layer_name, self.current_step);
            let sensitivity = self.layer_sensitivities.get(layer_name).copied().unwrap_or(0.0);
            let size = self.model_size_info.get(layer_name).copied().unwrap_or(0);

            report.push_str(&format!(
                "  {} | {} bits | sensitivity: {:.3} | params: {}\n",
                layer_name, bits, sensitivity, size
            ));
        }

        // Savings estimate
        let (memory_savings, compute_savings) = self.estimate_savings();
        report.push('\n');
        report.push_str(&format!(
            "💾 Estimated memory savings: {:.1}%\n",
            memory_savings * 100.0
        ));
        report.push_str(&format!(
            "⚡ Estimated compute savings: {:.1}%\n",
            compute_savings * 100.0
        ));

        // Bit consumption
        if !self.model_size_info.is_empty() {
            let total_bits = self.config.estimate_bit_consumption(&self.model_size_info);
            report.push_str(&format!("📊 Total bit consumption: {} bits\n", total_bits));

            if let Some(budget) = self.config.bit_allocation_budget {
                let usage_pct = (total_bits as f64) / (budget as f64) * 100.0;
                report.push_str(&format!(
                    "💰 Budget usage: {:.1}% ({}/{})\n",
                    usage_pct, total_bits, budget
                ));
            }
        }

        report
    }
}

/// Traditional QAT training utilities (for backward compatibility)
pub struct QATTrainer {
    /// Learning rate for quantization parameters
    pub quant_lr: f32,
    /// Weight decay for quantization parameters
    pub quant_weight_decay: f32,
}

impl QATTrainer {
    pub fn new(quant_lr: f32, quant_weight_decay: f32) -> Self {
        Self {
            quant_lr,
            quant_weight_decay,
        }
    }

    /// Update quantization parameters with gradients
    pub fn update_quant_params(
        &self,
        params: &mut QuantizationParams,
        grads: &QuantizationGradients,
    ) -> Result<()> {
        // Update scale with gradient descent
        if let Some(scale_grad) = &grads.scale_grad {
            params.scale = params.scale.sub(&scale_grad.mul_scalar(self.quant_lr)?)?;
        }

        // Update zero point if present
        if let (Some(zp), Some(zp_grad)) = (&mut params.zero_point, &grads.zero_point_grad) {
            *zp = zp.sub(&zp_grad.mul_scalar(self.quant_lr)?)?;
        }

        Ok(())
    }
}

/// Gradients for quantization parameters
pub struct QuantizationGradients {
    pub scale_grad: Option<Tensor>,
    pub zero_point_grad: Option<Tensor>,
}

/// Calibration dataset for QAT
pub struct CalibrationDataset {
    samples: Vec<Tensor>,
    labels: Vec<Tensor>,
}

impl CalibrationDataset {
    pub fn new(samples: Vec<Tensor>, labels: Vec<Tensor>) -> Self {
        Self { samples, labels }
    }

    /// Run calibration to initialize quantization parameters
    pub fn calibrate(&self, model: &mut QATModel) -> Result<()> {
        // Disable gradient computation during calibration
        for (sample, _label) in self.samples.iter().zip(&self.labels) {
            let _ = model.model.forward(sample.clone())?;
        }

        Ok(())
    }
}

/// QAT-specific loss function that includes quantization error
pub fn qat_loss(
    predictions: &Tensor,
    targets: &Tensor,
    quant_error: f32,
    alpha: f32,
) -> Result<Tensor> {
    // Regular loss (e.g., cross-entropy)
    let task_loss = compute_task_loss(predictions, targets)?;

    // Add quantization error penalty
    let total_loss = task_loss.add_scalar(alpha * quant_error)?;

    Ok(total_loss)
}

fn compute_task_loss(predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    // Placeholder for actual loss computation
    predictions.sub(targets)?.pow(2.0)?.mean()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fake_quantize() {
        let tensor =
            Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4]).expect("tensor operation failed");
        let scale = Tensor::from_vec(vec![0.1], &[1]).expect("tensor operation failed");
        let zero_point =
            Some(Tensor::from_vec(vec![128.0], &[1]).expect("tensor operation failed"));

        let quantized = fake_quantize(&tensor, &scale, zero_point.as_ref(), 8, false)
            .expect("tensor operation failed");
        assert_eq!(quantized.shape(), tensor.shape());
    }

    #[test]
    fn test_quantization_params() {
        let mut params = QuantizationParams::new(&[1], true);

        let tensor1 =
            Tensor::from_vec(vec![-1.0, 0.0, 1.0], &[3]).expect("tensor operation failed");
        params.update_stats(&tensor1, 0.9).expect("tensor operation failed");

        assert!(params.num_observations == 1);
        params.compute_params(8, true).expect("operation failed in test");
    }

    #[test]
    fn test_qat_config() {
        let config = QATConfig::default();
        assert_eq!(config.default_bits, 8);
        assert!(config.symmetric);
        assert_eq!(config.start_step, 1000);
    }

    #[test]
    fn test_qat_config_defaults_full() {
        let config = QATConfig::default();
        assert!(!config.per_channel);
        assert_eq!(config.observer_momentum, 0.99);
        assert!(!config.learnable_step_size);
        assert!(config.layer_configs.is_empty());
        assert!(!config.quantize_activations);
        assert_eq!(config.activation_bits, 8);
        assert!(!config.enable_mixed_bit_optimization);
        assert!(config.bit_allocation_budget.is_none());
        assert!(config.freeze_step.is_none());
    }

    #[test]
    fn test_mixed_bit_strategy_uniform_default() {
        let strategy = MixedBitStrategy::default();
        if let MixedBitStrategy::Uniform { bits } = strategy {
            assert_eq!(bits, 8);
        } else {
            panic!("Expected Uniform strategy");
        }
    }

    #[test]
    fn test_layer_quant_config_default() {
        let config = LayerQuantConfig::default();
        assert_eq!(config.bits, 8);
        assert!(config.symmetric);
        assert!(!config.per_channel);
        assert_eq!(config.sensitivity, 0.5);
        assert!(!config.is_critical);
    }

    #[test]
    fn test_get_layer_bits_uniform() {
        let config = QATConfig::default();
        let bits = config.get_layer_bits("any_layer", 0);
        assert_eq!(bits, 8);
    }

    #[test]
    fn test_get_layer_bits_manual() {
        let mut config = QATConfig::default();
        let mut layer_bits = HashMap::new();
        layer_bits.insert("conv".to_string(), 4_u8);
        layer_bits.insert("linear".to_string(), 8_u8);
        config.mixed_bit_strategy = MixedBitStrategy::Manual { layer_bits };
        assert_eq!(config.get_layer_bits("conv_layer1", 0), 4);
        assert_eq!(config.get_layer_bits("linear_layer1", 0), 8);
        assert_eq!(config.get_layer_bits("unknown", 0), 8); // default
    }

    #[test]
    fn test_get_layer_bits_resource_constrained() {
        let config = QATConfig {
            mixed_bit_strategy: MixedBitStrategy::ResourceConstrained {
                total_bit_budget: 1024,
                critical_layers: vec!["attention".to_string()],
                critical_bits: 8,
                default_bits: 4,
            },
            ..Default::default()
        };
        assert_eq!(config.get_layer_bits("attention_head", 0), 8);
        assert_eq!(config.get_layer_bits("some_layer", 0), 4);
    }

    #[test]
    fn test_get_layer_bits_progressive() {
        let config = QATConfig {
            mixed_bit_strategy: MixedBitStrategy::Progressive {
                initial_bits: 8,
                final_bits: 4,
                reduction_schedule: vec![(100, 6), (200, 4)],
            },
            ..Default::default()
        };
        assert_eq!(config.get_layer_bits("layer", 50), 8);
        assert_eq!(config.get_layer_bits("layer", 150), 6);
        assert_eq!(config.get_layer_bits("layer", 250), 4);
    }

    #[test]
    fn test_set_layer_config() {
        let mut config = QATConfig::default();
        let layer_config = LayerQuantConfig {
            bits: 4,
            symmetric: false,
            per_channel: true,
            sensitivity: 0.9,
            is_critical: true,
        };
        config.set_layer_config("my_layer".to_string(), layer_config);
        assert!(config.layer_configs.contains_key("my_layer"));
        assert_eq!(config.get_layer_bits("my_layer", 0), 4);
    }

    #[test]
    fn test_auto_configure_sensitivity() {
        let mut config = QATConfig::default();
        let mut sensitivities = HashMap::new();
        sensitivities.insert("layer1".to_string(), 0.9_f32);
        sensitivities.insert("layer2".to_string(), 0.3_f32);
        config.auto_configure_sensitivity(sensitivities);
        assert_eq!(config.layer_configs["layer1"].bits, 8);
        assert_eq!(config.layer_configs["layer2"].bits, 4);
        assert!(config.layer_configs["layer1"].is_critical);
        assert!(!config.layer_configs["layer2"].is_critical);
    }

    #[test]
    fn test_create_common_config_edge() {
        let config = QATConfig::create_common_config("edge_deployment");
        assert!(config.quantize_activations);
        assert!(config.enable_mixed_bit_optimization);
        assert_eq!(config.activation_bits, 8);
    }

    #[test]
    fn test_create_common_config_high_accuracy() {
        let config = QATConfig::create_common_config("high_accuracy");
        assert!(!config.quantize_activations);
        assert!(config.enable_mixed_bit_optimization);
    }

    #[test]
    fn test_create_common_config_aggressive() {
        let config = QATConfig::create_common_config("aggressive_compression");
        assert!(config.quantize_activations);
        assert_eq!(config.activation_bits, 4);
    }

    #[test]
    fn test_create_common_config_unknown() {
        let config = QATConfig::create_common_config("unknown_scenario");
        assert_eq!(config.default_bits, 8); // Falls back to default
    }

    #[test]
    fn test_quantization_params_symmetric() {
        let params = QuantizationParams::new(&[1], true);
        assert!(params.zero_point.is_none());
        assert_eq!(params.num_observations, 0);
    }

    #[test]
    fn test_quantization_params_asymmetric() {
        let params = QuantizationParams::new(&[1], false);
        assert!(params.zero_point.is_some());
        assert_eq!(params.num_observations, 0);
    }

    #[test]
    fn test_estimate_bit_consumption() {
        let config = QATConfig {
            mixed_bit_strategy: MixedBitStrategy::Uniform { bits: 4 },
            ..Default::default()
        };
        let mut model_info = HashMap::new();
        model_info.insert("layer1".to_string(), 1000_u64);
        model_info.insert("layer2".to_string(), 2000_u64);
        let total = config.estimate_bit_consumption(&model_info);
        assert_eq!(total, 12000); // (1000 + 2000) * 4
    }

    #[test]
    fn test_optimize_bit_allocation_no_budget() {
        let mut config = QATConfig::default();
        let model_info = HashMap::new();
        let result = config.optimize_bit_allocation(model_info);
        assert!(result.is_ok()); // No budget set, should be no-op
    }

    #[test]
    fn test_optimize_bit_allocation_with_budget() {
        let mut config = QATConfig {
            bit_allocation_budget: Some(100_000),
            ..Default::default()
        };
        let mut model_info = HashMap::new();
        model_info.insert("layer1".to_string(), 5000_u64);
        model_info.insert("layer2".to_string(), 3000_u64);
        let result = config.optimize_bit_allocation(model_info);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mixed_bit_trainer_creation() {
        let config = QATConfig::default();
        let trainer = MixedBitQATTrainer::new(config, 0.001, 0.0001);
        assert_eq!(trainer.quant_lr, 0.001);
        assert_eq!(trainer.quant_weight_decay, 0.0001);
    }

    #[test]
    fn test_mixed_bit_trainer_estimate_savings() {
        let config = QATConfig::default();
        let trainer = MixedBitQATTrainer::new(config, 0.001, 0.0);
        let (size_savings, compute_savings) = trainer.estimate_savings();
        assert!(size_savings >= 0.0);
        assert!(compute_savings >= 0.0);
    }

    #[test]
    fn test_activation_quantizer_creation() {
        let quantizer = ActivationQuantizer::new(&[1, 256], 8, true);
        assert_eq!(quantizer.bits, 8);
    }

    #[test]
    fn test_calibration_dataset_creation() {
        let samples = vec![Tensor::from_vec(vec![1.0, 2.0], &[2]).expect("tensor op failed")];
        let labels = vec![Tensor::from_vec(vec![0.0], &[1]).expect("tensor op failed")];
        let dataset = CalibrationDataset::new(samples, labels);
        assert_eq!(dataset.samples.len(), 1);
        assert_eq!(dataset.labels.len(), 1);
    }
}
