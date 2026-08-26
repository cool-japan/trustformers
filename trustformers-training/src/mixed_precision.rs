use anyhow::Result;
use scirs2_core::Complex; // SciRS2 Integration Policy
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::tensor::Tensor;

/// Half-precision storage dtype used by [`AMPManager::to_half_precision`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum HalfPrecisionDType {
    /// IEEE 754 binary16 — 10 mantissa bits, max magnitude 65504.
    #[default]
    F16,
    /// bfloat16 — 7 mantissa bits, `f32`'s exponent range (no overflow at `f32` scale).
    BF16,
}

/// Configuration for mixed precision training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionConfig {
    /// Enable automatic mixed precision
    pub enabled: bool,
    /// Narrow dtype that [`AMPManager::to_half_precision`] casts to.
    #[serde(default)]
    pub half_dtype: HalfPrecisionDType,
    /// Initial loss scale value
    pub init_scale: f32,
    /// Factor to scale loss by when no overflow is detected
    pub scale_factor: f32,
    /// Factor to scale loss by when overflow is detected
    pub backoff_factor: f32,
    /// Number of consecutive steps without overflow before increasing scale
    pub scale_window: usize,
    /// Minimum loss scale value
    pub min_scale: f32,
    /// Maximum loss scale value
    pub max_scale: f32,
    /// Skip optimizer update if loss is inf/nan
    pub skip_inf_nan: bool,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            half_dtype: HalfPrecisionDType::F16,
            init_scale: 2f32.powf(16.0), // 65536
            scale_factor: 2.0,
            backoff_factor: 0.5,
            scale_window: 2000,
            min_scale: 1.0,
            max_scale: 2f32.powf(24.0), // 16M
            skip_inf_nan: true,
        }
    }
}

/// Loss scaler for automatic mixed precision training
#[derive(Debug, Clone)]
pub struct LossScaler {
    config: MixedPrecisionConfig,
    current_scale: f32,
    steps_since_overflow: usize,
    overflow_detected: bool,
}

impl LossScaler {
    pub fn new(config: MixedPrecisionConfig) -> Self {
        Self {
            current_scale: config.init_scale,
            steps_since_overflow: 0,
            overflow_detected: false,
            config,
        }
    }

    /// Get current loss scale
    pub fn get_scale(&self) -> f32 {
        if self.config.enabled {
            self.current_scale
        } else {
            1.0
        }
    }

    /// Scale the loss tensor for backward pass
    pub fn scale_loss(&self, loss: &Tensor) -> Result<Tensor> {
        if !self.config.enabled {
            return Ok(loss.clone());
        }

        loss.scalar_mul(self.current_scale).map_err(|e| anyhow::anyhow!(e))
    }

    /// Unscale gradients after the backward pass, reporting whether the batch overflowed.
    ///
    /// The scan is performed in two phases so that the gradient map is never left in a
    /// half-unscaled state:
    ///
    /// 1. Every tensor is checked for `inf`/`NaN`.
    /// 2. Only if the whole map is finite is any tensor divided by the current loss scale.
    ///
    /// When `true` is returned **no tensor has been modified** — the caller must skip the
    /// optimizer step for this batch and feed the result to [`LossScaler::update_scale`]
    /// so the scale backs off. (Previously the loop unscaled an arbitrary,
    /// `HashMap`-iteration-order-dependent prefix before bailing out, which silently mixed
    /// scaled and unscaled gradients in the next step.)
    pub fn unscale_gradients(&self, gradients: &mut HashMap<String, Tensor>) -> Result<bool> {
        if !self.config.enabled {
            return Ok(false);
        }

        // Phase 1: full scan, no mutation.
        let mut overflow_detected = false;
        for gradient in gradients.values() {
            if self.has_inf_nan(gradient)? {
                overflow_detected = true;
                break;
            }
        }

        // A dirty batch is reported without touching any tensor, so the caller can discard
        // the whole map (or retry) with a well-defined, fully-scaled state.
        if overflow_detected && self.config.skip_inf_nan {
            return Ok(true);
        }

        // Phase 2: the map is clean (or the caller opted out of skipping) — unscale all of it.
        let inv_scale = 1.0 / self.current_scale;
        for gradient in gradients.values_mut() {
            *gradient = gradient.scalar_mul(inv_scale).map_err(|e| anyhow::anyhow!(e))?;
        }

        Ok(overflow_detected)
    }

    /// Update loss scale based on overflow detection
    pub fn update_scale(&mut self, overflow_detected: bool) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        if overflow_detected {
            // Decrease scale and reset counter
            self.current_scale =
                (self.current_scale * self.config.backoff_factor).max(self.config.min_scale);
            self.steps_since_overflow = 0;
            self.overflow_detected = true;
        } else {
            // Increase counter and potentially scale up
            self.steps_since_overflow += 1;
            self.overflow_detected = false;

            if self.steps_since_overflow >= self.config.scale_window {
                self.current_scale =
                    (self.current_scale * self.config.scale_factor).min(self.config.max_scale);
                self.steps_since_overflow = 0;
            }
        }

        Ok(())
    }

    /// Check if overflow was detected in the last step
    pub fn overflow_detected(&self) -> bool {
        self.overflow_detected
    }

    /// Check for inf/nan values in tensor
    fn has_inf_nan(&self, tensor: &Tensor) -> Result<bool> {
        match tensor {
            Tensor::F32(arr) => {
                for &value in arr.iter() {
                    if !value.is_finite() {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            Tensor::F64(arr) => {
                for &value in arr.iter() {
                    if !value.is_finite() {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            Tensor::F16(arr) => {
                for &value in arr.iter() {
                    if !value.to_f32().is_finite() {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            Tensor::BF16(arr) => {
                for &value in arr.iter() {
                    if !value.to_f32().is_finite() {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            Tensor::I64(_) => Ok(false), // Integer tensors can't have inf/nan
            Tensor::C32(arr) => {
                for &value in arr.iter() {
                    if !value.re.is_finite() || !value.im.is_finite() {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            Tensor::C64(arr) => {
                for &value in arr.iter() {
                    if !value.re.is_finite() || !value.im.is_finite() {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            Tensor::CF16(arr) => {
                for &value in arr.iter() {
                    if !value.re.to_f32().is_finite() || !value.im.to_f32().is_finite() {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            Tensor::CBF16(arr) => {
                for &value in arr.iter() {
                    if !value.re.to_f32().is_finite() || !value.im.to_f32().is_finite() {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            _ => Ok(false), // Assume sparse and other tensor types are validated
        }
    }
}

/// Automatic Mixed Precision (AMP) manager
#[derive(Debug)]
pub struct AMPManager {
    pub loss_scaler: LossScaler,
    pub config: MixedPrecisionConfig,
}

impl AMPManager {
    pub fn new(config: MixedPrecisionConfig) -> Self {
        let loss_scaler = LossScaler::new(config.clone());
        Self {
            loss_scaler,
            config,
        }
    }

    /// Cast a tensor down to the configured half-precision dtype.
    ///
    /// This performs a **real** narrowing cast to `half::f16` / `half::bf16` (producing a
    /// [`Tensor::F16`] / [`Tensor::BF16`]), so the result carries genuine IEEE half-precision
    /// semantics — relative mantissa precision, subnormals and `inf` on overflow — and half
    /// the memory footprint. Integer tensors are returned unchanged (they have no
    /// half-precision counterpart in [`Tensor`]).
    ///
    /// Returns the input unchanged when AMP is disabled.
    pub fn to_half_precision(&self, tensor: &Tensor) -> Result<Tensor> {
        if !self.config.enabled {
            return Ok(tensor.clone());
        }

        match (tensor, self.config.half_dtype) {
            (Tensor::F32(arr), HalfPrecisionDType::F16) => {
                Ok(Tensor::F16(arr.mapv(half::f16::from_f32)))
            },
            (Tensor::F32(arr), HalfPrecisionDType::BF16) => {
                Ok(Tensor::BF16(arr.mapv(half::bf16::from_f32)))
            },
            (Tensor::F64(arr), HalfPrecisionDType::F16) => {
                Ok(Tensor::F16(arr.mapv(half::f16::from_f64)))
            },
            (Tensor::F64(arr), HalfPrecisionDType::BF16) => {
                Ok(Tensor::BF16(arr.mapv(half::bf16::from_f64)))
            },
            (Tensor::C32(arr), HalfPrecisionDType::F16) => {
                Ok(Tensor::CF16(arr.mapv(|x| {
                    Complex::new(half::f16::from_f32(x.re), half::f16::from_f32(x.im))
                })))
            },
            (Tensor::C32(arr), HalfPrecisionDType::BF16) => {
                Ok(Tensor::CBF16(arr.mapv(|x| {
                    Complex::new(half::bf16::from_f32(x.re), half::bf16::from_f32(x.im))
                })))
            },
            (Tensor::C64(arr), HalfPrecisionDType::F16) => {
                Ok(Tensor::CF16(arr.mapv(|x| {
                    Complex::new(half::f16::from_f64(x.re), half::f16::from_f64(x.im))
                })))
            },
            (Tensor::C64(arr), HalfPrecisionDType::BF16) => {
                Ok(Tensor::CBF16(arr.mapv(|x| {
                    Complex::new(half::bf16::from_f64(x.re), half::bf16::from_f64(x.im))
                })))
            },
            // Already at (or below) half precision, or an integer / device tensor with no
            // meaningful narrowing: hand it back untouched.
            _ => Ok(tensor.clone()),
        }
    }

    /// Widen a half-precision tensor back to `f32` (or `Complex<f32>`).
    ///
    /// This is the exact inverse of [`AMPManager::to_half_precision`] for the dtypes that
    /// method produces; anything already at full precision is returned unchanged.
    pub fn to_full_precision(&self, tensor: &Tensor) -> Result<Tensor> {
        match tensor {
            Tensor::F16(arr) => Ok(Tensor::F32(arr.mapv(|x| x.to_f32()))),
            Tensor::BF16(arr) => Ok(Tensor::F32(arr.mapv(|x| x.to_f32()))),
            Tensor::CF16(arr) => Ok(Tensor::C32(
                arr.mapv(|x| Complex::new(x.re.to_f32(), x.im.to_f32())),
            )),
            Tensor::CBF16(arr) => Ok(Tensor::C32(
                arr.mapv(|x| Complex::new(x.re.to_f32(), x.im.to_f32())),
            )),
            _ => Ok(tensor.clone()),
        }
    }

    /// Run a forward pass under automatic mixed precision.
    ///
    /// The input is cast down to the configured half dtype, `forward_fn` runs on the
    /// narrowed tensor, and the result is widened back to full precision so the loss is
    /// always computed in `f32`. With AMP disabled the closure sees the untouched input.
    pub fn forward_with_amp<F>(&self, input: &Tensor, forward_fn: F) -> Result<Tensor>
    where
        F: FnOnce(Tensor) -> Result<Tensor>,
    {
        if !self.config.enabled {
            return forward_fn(input.clone());
        }

        let half_input = self.to_half_precision(input)?;
        let output = forward_fn(half_input)?;
        self.to_full_precision(&output)
    }

    /// Run a backward pass under loss scaling.
    ///
    /// The loss is multiplied by the current scale, `backward_fn` is invoked with the
    /// **scaled** loss to produce the raw (scaled) gradients, those gradients are unscaled
    /// in one atomic pass, and the loss scale is updated from the overflow verdict.
    ///
    /// Returns `(gradients, overflow)`. When `overflow` is `true` the gradients are still
    /// scaled (see [`LossScaler::unscale_gradients`]) and the caller **must** skip the
    /// optimizer step for this batch.
    pub fn backward_with_amp<F>(
        &mut self,
        loss: &Tensor,
        backward_fn: F,
    ) -> Result<(HashMap<String, Tensor>, bool)>
    where
        F: FnOnce(&Tensor) -> Result<HashMap<String, Tensor>>,
    {
        let scaled_loss = self.loss_scaler.scale_loss(loss)?;
        let mut gradients = backward_fn(&scaled_loss)?;

        let overflow = self.loss_scaler.unscale_gradients(&mut gradients)?;
        self.loss_scaler.update_scale(overflow)?;

        Ok((gradients, overflow))
    }

    /// Unscale a set of already-computed gradients and update the loss scale.
    ///
    /// Use this when the backward pass is driven elsewhere and only the loss-scaling
    /// bookkeeping is needed. Returns `true` when the batch overflowed, in which case the
    /// gradients were left untouched and the optimizer step must be skipped.
    pub fn unscale_and_update(&mut self, gradients: &mut HashMap<String, Tensor>) -> Result<bool> {
        let overflow = self.loss_scaler.unscale_gradients(gradients)?;
        self.loss_scaler.update_scale(overflow)?;
        Ok(overflow)
    }

    /// Get current loss scale
    pub fn get_loss_scale(&self) -> f32 {
        self.loss_scaler.get_scale()
    }

    /// Check if AMP is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

/// Mixed precision training utilities
pub mod utils {
    use super::*;

    /// Create a default AMP configuration for fp16 training
    pub fn default_fp16_config() -> MixedPrecisionConfig {
        MixedPrecisionConfig {
            enabled: true,
            half_dtype: HalfPrecisionDType::F16,
            init_scale: 2f32.powf(16.0),
            scale_factor: 2.0,
            backoff_factor: 0.5,
            scale_window: 2000,
            min_scale: 1.0,
            max_scale: 2f32.powf(24.0),
            skip_inf_nan: true,
        }
    }

    /// Create a default AMP configuration for bfloat16 training
    pub fn default_bf16_config() -> MixedPrecisionConfig {
        MixedPrecisionConfig {
            enabled: true,
            half_dtype: HalfPrecisionDType::BF16,
            init_scale: 1.0, // bfloat16 doesn't typically need loss scaling
            scale_factor: 1.0,
            backoff_factor: 1.0,
            scale_window: usize::MAX,
            min_scale: 1.0,
            max_scale: 1.0,
            skip_inf_nan: true,
        }
    }

    /// Check if tensor values are within fp16 range
    pub fn is_fp16_safe(tensor: &Tensor) -> Result<bool> {
        match tensor {
            Tensor::F32(arr) => {
                for &value in arr.iter() {
                    if value.abs() > 65504.0 || (!value.is_finite() && value != 0.0) {
                        return Ok(false);
                    }
                }
                Ok(true)
            },
            Tensor::F64(arr) => {
                for &value in arr.iter() {
                    if value.abs() > 65504.0 || (!value.is_finite() && value != 0.0) {
                        return Ok(false);
                    }
                }
                Ok(true)
            },
            Tensor::F16(_) => Ok(true),  // Already fp16, so always safe
            Tensor::BF16(_) => Ok(true), // BF16 has similar range to fp16
            Tensor::I64(_) => Ok(true),
            Tensor::C32(arr) => {
                for &value in arr.iter() {
                    if value.re.abs() > 65504.0
                        || value.im.abs() > 65504.0
                        || (!value.re.is_finite() && value.re != 0.0)
                        || (!value.im.is_finite() && value.im != 0.0)
                    {
                        return Ok(false);
                    }
                }
                Ok(true)
            },
            Tensor::C64(arr) => {
                for &value in arr.iter() {
                    if value.re.abs() > 65504.0
                        || value.im.abs() > 65504.0
                        || (!value.re.is_finite() && value.re != 0.0)
                        || (!value.im.is_finite() && value.im != 0.0)
                    {
                        return Ok(false);
                    }
                }
                Ok(true)
            },
            Tensor::CF16(_) => Ok(true),  // Already fp16, so always safe
            Tensor::CBF16(_) => Ok(true), // BF16 has similar range to fp16
            _ => Ok(true),                // Assume sparse and other tensor types are safe
        }
    }

    /// Calculate the dynamic range of a tensor
    pub fn calculate_dynamic_range(tensor: &Tensor) -> Result<(f32, f32)> {
        match tensor {
            Tensor::F32(arr) => {
                let mut min_val = f32::INFINITY;
                let mut max_val = f32::NEG_INFINITY;

                for &value in arr.iter() {
                    if value.is_finite() {
                        min_val = min_val.min(value);
                        max_val = max_val.max(value);
                    }
                }

                Ok((min_val, max_val))
            },
            Tensor::F64(arr) => {
                let min_val = arr
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .copied()
                    .unwrap_or(0.0) as f32;
                let max_val = arr
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .copied()
                    .unwrap_or(0.0) as f32;
                Ok((min_val, max_val))
            },
            Tensor::F16(arr) => {
                let mut min_val = f32::INFINITY;
                let mut max_val = f32::NEG_INFINITY;

                for &value in arr.iter() {
                    let f32_val = value.to_f32();
                    if f32_val.is_finite() {
                        min_val = min_val.min(f32_val);
                        max_val = max_val.max(f32_val);
                    }
                }

                Ok((min_val, max_val))
            },
            Tensor::BF16(arr) => {
                let mut min_val = f32::INFINITY;
                let mut max_val = f32::NEG_INFINITY;

                for &value in arr.iter() {
                    let f32_val = value.to_f32();
                    if f32_val.is_finite() {
                        min_val = min_val.min(f32_val);
                        max_val = max_val.max(f32_val);
                    }
                }

                Ok((min_val, max_val))
            },
            Tensor::I64(arr) => {
                let min_val = arr.iter().min().copied().unwrap_or(0) as f32;
                let max_val = arr.iter().max().copied().unwrap_or(0) as f32;
                Ok((min_val, max_val))
            },
            Tensor::C32(arr) => {
                let mut min_val = f32::INFINITY;
                let mut max_val = f32::NEG_INFINITY;

                for &value in arr.iter() {
                    let magnitude = value.norm();
                    if magnitude.is_finite() {
                        min_val = min_val.min(magnitude);
                        max_val = max_val.max(magnitude);
                    }
                }

                Ok((min_val, max_val))
            },
            Tensor::C64(arr) => {
                let mut min_val = f32::INFINITY;
                let mut max_val = f32::NEG_INFINITY;

                for &value in arr.iter() {
                    let magnitude = value.norm() as f32;
                    if magnitude.is_finite() {
                        min_val = min_val.min(magnitude);
                        max_val = max_val.max(magnitude);
                    }
                }

                Ok((min_val, max_val))
            },
            Tensor::CF16(arr) => {
                let mut min_val = f32::INFINITY;
                let mut max_val = f32::NEG_INFINITY;

                for &value in arr.iter() {
                    let magnitude = (value.re.to_f32().powi(2) + value.im.to_f32().powi(2)).sqrt();
                    if magnitude.is_finite() {
                        min_val = min_val.min(magnitude);
                        max_val = max_val.max(magnitude);
                    }
                }

                Ok((min_val, max_val))
            },
            Tensor::CBF16(arr) => {
                let mut min_val = f32::INFINITY;
                let mut max_val = f32::NEG_INFINITY;

                for &value in arr.iter() {
                    let magnitude = (value.re.to_f32().powi(2) + value.im.to_f32().powi(2)).sqrt();
                    if magnitude.is_finite() {
                        min_val = min_val.min(magnitude);
                        max_val = max_val.max(magnitude);
                    }
                }

                Ok((min_val, max_val))
            },
            _ => Ok((0.0, 1.0)), // Default range for sparse and other tensor types
        }
    }
}

/// Advanced mixed precision training enhancements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedMixedPrecisionConfig {
    /// Base mixed precision config
    pub base_config: MixedPrecisionConfig,
    /// Enable dynamic loss scaling
    pub enable_dynamic_scaling: bool,
    /// Enable gradient scaling per layer
    pub enable_per_layer_scaling: bool,
    /// Enable automatic precision selection
    pub enable_auto_precision: bool,
    /// Minimum precision (fp16, bf16, fp32)
    pub min_precision: String,
    /// Maximum precision (fp16, bf16, fp32)
    pub max_precision: String,
    /// Precision adaptation rate
    pub precision_adaptation_rate: f32,
    /// Memory usage threshold for precision switching
    pub memory_threshold: f32,
    /// Performance threshold for precision switching
    pub performance_threshold: f32,
}

impl Default for AdvancedMixedPrecisionConfig {
    fn default() -> Self {
        Self {
            base_config: MixedPrecisionConfig::default(),
            enable_dynamic_scaling: true,
            enable_per_layer_scaling: true,
            enable_auto_precision: true,
            min_precision: "fp16".to_string(),
            max_precision: "fp32".to_string(),
            precision_adaptation_rate: 0.1,
            memory_threshold: 0.8,
            performance_threshold: 0.9,
        }
    }
}

/// Per-layer scaling configuration
#[derive(Debug, Clone)]
pub struct LayerScalingConfig {
    /// Layer name
    pub layer_name: String,
    /// Current scale factor
    pub scale_factor: f32,
    /// Gradient norm history
    pub gradient_norm_history: Vec<f32>,
    /// Loss history for this layer
    pub loss_history: Vec<f32>,
    /// Overflow count
    pub overflow_count: usize,
    /// Underflow count
    pub underflow_count: usize,
}

impl LayerScalingConfig {
    pub fn new(layer_name: String) -> Self {
        Self {
            layer_name,
            scale_factor: 1.0,
            gradient_norm_history: Vec::new(),
            loss_history: Vec::new(),
            overflow_count: 0,
            underflow_count: 0,
        }
    }
}

/// Advanced mixed precision manager
#[derive(Debug)]
pub struct AdvancedMixedPrecisionManager {
    config: AdvancedMixedPrecisionConfig,
    base_manager: AMPManager,
    layer_configs: HashMap<String, LayerScalingConfig>,
    current_precision: String,
    precision_history: Vec<(usize, String)>,
    memory_usage_history: Vec<f32>,
    performance_history: Vec<f32>,
    step_count: usize,
}

impl AdvancedMixedPrecisionManager {
    pub fn new(config: AdvancedMixedPrecisionConfig) -> Self {
        let base_manager = AMPManager::new(config.base_config.clone());
        Self {
            config,
            base_manager,
            layer_configs: HashMap::new(),
            current_precision: "fp32".to_string(),
            precision_history: Vec::new(),
            memory_usage_history: Vec::new(),
            performance_history: Vec::new(),
            step_count: 0,
        }
    }

    /// Update with training step information
    pub fn update_step(&mut self, memory_usage: f32, performance_score: f32) {
        self.step_count += 1;
        self.memory_usage_history.push(memory_usage);
        self.performance_history.push(performance_score);

        // Keep only recent history
        if self.memory_usage_history.len() > 100 {
            self.memory_usage_history.remove(0);
            self.performance_history.remove(0);
        }

        // Adapt precision if enabled
        if self.config.enable_auto_precision {
            self.adapt_precision();
        }
    }

    /// Adapt precision based on memory usage and performance
    fn adapt_precision(&mut self) {
        let avg_memory =
            self.memory_usage_history.iter().sum::<f32>() / self.memory_usage_history.len() as f32;
        let avg_performance =
            self.performance_history.iter().sum::<f32>() / self.performance_history.len() as f32;

        let target_precision = if avg_memory > self.config.memory_threshold {
            // High memory usage, switch to lower precision
            match self.current_precision.as_str() {
                "fp32" => "fp16",
                "bf16" => "fp16",
                _ => "fp16",
            }
        } else if avg_performance < self.config.performance_threshold {
            // Low performance, switch to higher precision
            match self.current_precision.as_str() {
                "fp16" => "bf16",
                "bf16" => "fp32",
                _ => "fp32",
            }
        } else {
            &self.current_precision
        };

        if target_precision != self.current_precision {
            self.switch_precision(target_precision.to_string());
        }
    }

    /// Switch to a different precision
    fn switch_precision(&mut self, new_precision: String) {
        self.current_precision = new_precision.clone();
        self.precision_history.push((self.step_count, new_precision));

        // Update base manager configuration
        match self.current_precision.as_str() {
            "fp16" => {
                self.base_manager.config = utils::default_fp16_config();
            },
            "bf16" => {
                self.base_manager.config = utils::default_bf16_config();
            },
            "fp32" => {
                self.base_manager.config = MixedPrecisionConfig {
                    enabled: false,
                    ..Default::default()
                };
            },
            _ => {
                self.base_manager.config = utils::default_fp16_config();
            },
        }
    }

    /// Scale gradients with per-layer scaling
    pub fn scale_gradients_per_layer(
        &mut self,
        gradients: &mut HashMap<String, Tensor>,
    ) -> Result<bool> {
        let mut global_overflow = false;

        for (layer_name, gradient) in gradients.iter_mut() {
            // Get or create layer config
            if !self.layer_configs.contains_key(layer_name) {
                self.layer_configs.insert(
                    layer_name.clone(),
                    LayerScalingConfig::new(layer_name.clone()),
                );
            }

            // Compute gradient norm first
            let grad_norm = self.compute_gradient_norm(gradient)?;

            // Get scale factor before mutably borrowing
            let enable_per_layer_scaling = self.config.enable_per_layer_scaling;

            let layer_config = self.layer_configs.get_mut(layer_name).ok_or_else(|| {
                anyhow::anyhow!("layer config missing for '{layer_name}' after insertion")
            })?;
            layer_config.gradient_norm_history.push(grad_norm);

            // Keep only recent history
            if layer_config.gradient_norm_history.len() > 50 {
                layer_config.gradient_norm_history.remove(0);
            }

            // Adapt layer-specific scaling
            if enable_per_layer_scaling {
                // Inline the adaptation logic to avoid borrow issues
                if layer_config.gradient_norm_history.len() >= 5 {
                    let recent_norms: Vec<f32> =
                        layer_config.gradient_norm_history.iter().rev().take(5).cloned().collect();

                    let avg_norm = recent_norms.iter().sum::<f32>() / recent_norms.len() as f32;

                    // Adjust scaling based on gradient norm
                    if avg_norm > 10.0 {
                        // Large gradients, increase scaling
                        layer_config.scale_factor *= 1.1;
                    } else if avg_norm < 0.01 {
                        // Small gradients, decrease scaling
                        layer_config.scale_factor *= 0.9;
                    }

                    // Clamp scale factor to reasonable range
                    layer_config.scale_factor = layer_config.scale_factor.clamp(0.01, 1000.0);
                }
            }

            let scale_factor = layer_config.scale_factor;

            // Drop the mutable borrow before calling scale_tensor
            let _ = layer_config;

            // Scale gradient using extracted scale factor
            *gradient = self.scale_tensor(gradient, scale_factor)?;

            // Check for overflow and update counters
            let has_overflow = self.has_overflow(gradient)?;

            // Re-acquire mutable borrow to update counters
            let layer_config = self.layer_configs.get_mut(layer_name).ok_or_else(|| {
                anyhow::anyhow!("layer config missing for '{layer_name}' after insertion")
            })?;
            if has_overflow {
                layer_config.overflow_count += 1;
                global_overflow = true;
            } else {
                layer_config.underflow_count += 1;
            }
        }

        Ok(global_overflow)
    }

    /// Adapt scaling for a specific layer
    fn adapt_layer_scaling(&mut self, layer_config: &mut LayerScalingConfig) {
        if layer_config.gradient_norm_history.len() < 5 {
            return;
        }

        let recent_norms: Vec<f32> =
            layer_config.gradient_norm_history.iter().rev().take(5).cloned().collect();

        let avg_norm = recent_norms.iter().sum::<f32>() / recent_norms.len() as f32;

        // Adjust scaling based on gradient norm
        if avg_norm > 10.0 {
            // Large gradients, increase scaling
            layer_config.scale_factor *= 1.1;
        } else if avg_norm < 0.01 {
            // Small gradients, decrease scaling
            layer_config.scale_factor *= 0.9;
        }

        // Clamp scaling factor
        layer_config.scale_factor = layer_config.scale_factor.clamp(0.1, 10.0);
    }

    /// Compute gradient norm
    fn compute_gradient_norm(&self, gradient: &Tensor) -> Result<f32> {
        match gradient {
            Tensor::F32(arr) => {
                let norm = arr.iter().map(|&x| x * x).sum::<f32>().sqrt();
                Ok(norm)
            },
            Tensor::F64(arr) => {
                let norm = arr.iter().map(|&x| x * x).sum::<f64>().sqrt() as f32;
                Ok(norm)
            },
            Tensor::F16(arr) => {
                let norm = arr
                    .iter()
                    .map(|&x| {
                        let f32_val = x.to_f32();
                        f32_val * f32_val
                    })
                    .sum::<f32>()
                    .sqrt();
                Ok(norm)
            },
            Tensor::BF16(arr) => {
                let norm = arr
                    .iter()
                    .map(|&x| {
                        let f32_val = x.to_f32();
                        f32_val * f32_val
                    })
                    .sum::<f32>()
                    .sqrt();
                Ok(norm)
            },
            Tensor::I64(_) => Ok(0.0),
            Tensor::C32(arr) => {
                let norm = arr.iter().map(|&x| x.norm_sqr()).sum::<f32>().sqrt();
                Ok(norm)
            },
            Tensor::C64(arr) => {
                let norm = arr.iter().map(|&x| x.norm_sqr() as f32).sum::<f32>().sqrt();
                Ok(norm)
            },
            Tensor::CF16(arr) => {
                let norm = arr
                    .iter()
                    .map(|&x| {
                        let re = x.re.to_f32();
                        let im = x.im.to_f32();
                        re * re + im * im
                    })
                    .sum::<f32>()
                    .sqrt();
                Ok(norm)
            },
            Tensor::CBF16(arr) => {
                let norm = arr
                    .iter()
                    .map(|&x| {
                        let re = x.re.to_f32();
                        let im = x.im.to_f32();
                        re * re + im * im
                    })
                    .sum::<f32>()
                    .sqrt();
                Ok(norm)
            },
            _ => Ok(1.0), // Default norm for sparse and other tensor types
        }
    }

    /// Scale tensor by factor
    fn scale_tensor(&self, tensor: &Tensor, factor: f32) -> Result<Tensor> {
        match tensor {
            Tensor::F32(arr) => {
                let scaled = arr.mapv(|x| x * factor);
                Ok(Tensor::F32(scaled))
            },
            Tensor::F64(arr) => {
                let scaled = arr.mapv(|x| x * factor as f64);
                Ok(Tensor::F64(scaled))
            },
            Tensor::F16(arr) => {
                let factor_f16 = half::f16::from_f32(factor);
                let scaled = arr.mapv(|x| x * factor_f16);
                Ok(Tensor::F16(scaled))
            },
            Tensor::BF16(arr) => {
                let factor_bf16 = half::bf16::from_f32(factor);
                let scaled = arr.mapv(|x| x * factor_bf16);
                Ok(Tensor::BF16(scaled))
            },
            Tensor::I64(arr) => Ok(Tensor::I64(arr.clone())),
            Tensor::C32(arr) => {
                let scaled = arr.mapv(|x| x * factor);
                Ok(Tensor::C32(scaled))
            },
            Tensor::C64(arr) => {
                let scaled = arr.mapv(|x| x * factor as f64);
                Ok(Tensor::C64(scaled))
            },
            Tensor::CF16(arr) => {
                let factor_f16 = half::f16::from_f32(factor);
                let scaled = arr.mapv(|x| Complex::new(x.re * factor_f16, x.im * factor_f16));
                Ok(Tensor::CF16(scaled))
            },
            Tensor::CBF16(arr) => {
                let factor_bf16 = half::bf16::from_f32(factor);
                let scaled = arr.mapv(|x| Complex::new(x.re * factor_bf16, x.im * factor_bf16));
                Ok(Tensor::CBF16(scaled))
            },
            _ => Ok(tensor.clone()), // Don't scale sparse and other tensor types
        }
    }

    /// Check for overflow in tensor
    fn has_overflow(&self, tensor: &Tensor) -> Result<bool> {
        match tensor {
            Tensor::F32(arr) => {
                for &value in arr.iter() {
                    if !value.is_finite() {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            Tensor::F64(arr) => {
                for &value in arr.iter() {
                    if !value.is_finite() {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            Tensor::F16(arr) => {
                for &value in arr.iter() {
                    if !value.to_f32().is_finite() {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            Tensor::BF16(arr) => {
                for &value in arr.iter() {
                    if !value.to_f32().is_finite() {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            Tensor::I64(_) => Ok(false),
            Tensor::C32(arr) => {
                for &value in arr.iter() {
                    if !value.re.is_finite() || !value.im.is_finite() {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            Tensor::C64(arr) => {
                for &value in arr.iter() {
                    if !value.re.is_finite() || !value.im.is_finite() {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            Tensor::CF16(arr) => {
                for &value in arr.iter() {
                    if !value.re.to_f32().is_finite() || !value.im.to_f32().is_finite() {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            Tensor::CBF16(arr) => {
                for &value in arr.iter() {
                    if !value.re.to_f32().is_finite() || !value.im.to_f32().is_finite() {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            _ => Ok(false), // Assume sparse and other tensor types don't overflow
        }
    }

    /// Get current precision
    pub fn get_current_precision(&self) -> &str {
        &self.current_precision
    }

    /// Get precision history
    pub fn get_precision_history(&self) -> &[(usize, String)] {
        &self.precision_history
    }

    /// Get layer configurations
    pub fn get_layer_configs(&self) -> &HashMap<String, LayerScalingConfig> {
        &self.layer_configs
    }

    /// Forward pass with advanced mixed precision
    pub fn forward_with_advanced_amp<F>(&mut self, forward_fn: F) -> Result<Tensor>
    where
        F: FnOnce() -> Result<Tensor>,
    {
        let output = forward_fn()?;

        // Apply precision-specific optimizations
        match self.current_precision.as_str() {
            "fp16" => self.optimize_for_fp16(&output),
            "bf16" => self.optimize_for_bf16(&output),
            "fp32" => Ok(output),
            _ => Ok(output),
        }
    }

    /// Optimize tensor for fp16
    fn optimize_for_fp16(&self, tensor: &Tensor) -> Result<Tensor> {
        match tensor {
            Tensor::F32(arr) => {
                let optimized = arr.mapv(|x| {
                    // Apply fp16 optimizations
                    let clamped = x.clamp(-65504.0, 65504.0);

                    (clamped * 1024.0).round() / 1024.0
                });
                Ok(Tensor::F32(optimized))
            },
            _ => Ok(tensor.clone()),
        }
    }

    /// Optimize tensor for bf16
    fn optimize_for_bf16(&self, tensor: &Tensor) -> Result<Tensor> {
        match tensor {
            Tensor::F32(arr) => {
                let optimized = arr.mapv(|x| {
                    // Apply bf16 optimizations (wider range, lower precision)

                    (x * 128.0).round() / 128.0
                });
                Ok(Tensor::F32(optimized))
            },
            _ => Ok(tensor.clone()),
        }
    }

    /// Generate mixed precision report
    pub fn generate_report(&self) -> MixedPrecisionReport {
        let total_overflows = self.layer_configs.values().map(|config| config.overflow_count).sum();

        let total_underflows =
            self.layer_configs.values().map(|config| config.underflow_count).sum();

        let avg_memory_usage = if !self.memory_usage_history.is_empty() {
            self.memory_usage_history.iter().sum::<f32>() / self.memory_usage_history.len() as f32
        } else {
            0.0
        };

        let avg_performance = if !self.performance_history.is_empty() {
            self.performance_history.iter().sum::<f32>() / self.performance_history.len() as f32
        } else {
            0.0
        };

        MixedPrecisionReport {
            current_precision: self.current_precision.clone(),
            step_count: self.step_count,
            total_overflows,
            total_underflows,
            avg_memory_usage,
            avg_performance,
            precision_switches: self.precision_history.len(),
            layer_count: self.layer_configs.len(),
            recommendations: self.generate_recommendations(),
        }
    }

    /// Generate recommendations for mixed precision training
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check overflow rates
        let total_overflows =
            self.layer_configs.values().map(|config| config.overflow_count).sum::<usize>();

        if total_overflows > self.step_count / 10 {
            recommendations
                .push("High overflow rate detected - consider reducing learning rate".to_string());
        }

        // Check memory usage
        let avg_memory = if !self.memory_usage_history.is_empty() {
            self.memory_usage_history.iter().sum::<f32>() / self.memory_usage_history.len() as f32
        } else {
            0.0
        };

        if avg_memory > 0.9 {
            recommendations
                .push("High memory usage - consider using fp16 or reducing batch size".to_string());
        }

        // Check performance
        let avg_performance = if !self.performance_history.is_empty() {
            self.performance_history.iter().sum::<f32>() / self.performance_history.len() as f32
        } else {
            0.0
        };

        if avg_performance < 0.5 {
            recommendations.push(
                "Low performance - consider using higher precision or adjusting hyperparameters"
                    .to_string(),
            );
        }

        // Check precision switches
        if self.precision_history.len() > 10 {
            recommendations.push(
                "Frequent precision switches - consider adjusting adaptation thresholds"
                    .to_string(),
            );
        }

        if recommendations.is_empty() {
            recommendations.push("Mixed precision training is working well".to_string());
        }

        recommendations
    }
}

/// Mixed precision training report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionReport {
    pub current_precision: String,
    pub step_count: usize,
    pub total_overflows: usize,
    pub total_underflows: usize,
    pub avg_memory_usage: f32,
    pub avg_performance: f32,
    pub precision_switches: usize,
    pub layer_count: usize,
    pub recommendations: Vec<String>,
}

/// Dynamic batching strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicBatchingConfig {
    /// Initial batch size
    pub initial_batch_size: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Minimum batch size
    pub min_batch_size: usize,
    /// Batch size adaptation rate
    pub adaptation_rate: f32,
    /// Memory threshold for batch size reduction
    pub memory_threshold: f32,
    /// Performance threshold for batch size increase
    pub performance_threshold: f32,
}

impl Default for DynamicBatchingConfig {
    fn default() -> Self {
        Self {
            initial_batch_size: 32,
            max_batch_size: 128,
            min_batch_size: 8,
            adaptation_rate: 0.1,
            memory_threshold: 0.85,
            performance_threshold: 0.9,
        }
    }
}

/// Dynamic batching manager
#[derive(Debug)]
pub struct DynamicBatchingManager {
    config: DynamicBatchingConfig,
    current_batch_size: usize,
    batch_size_history: Vec<(usize, usize)>,
    memory_usage_history: Vec<f32>,
    performance_history: Vec<f32>,
    step_count: usize,
}

impl DynamicBatchingManager {
    pub fn new(config: DynamicBatchingConfig) -> Self {
        Self {
            current_batch_size: config.initial_batch_size,
            config,
            batch_size_history: Vec::new(),
            memory_usage_history: Vec::new(),
            performance_history: Vec::new(),
            step_count: 0,
        }
    }

    /// Update with training step information
    pub fn update_step(&mut self, memory_usage: f32, performance_score: f32) {
        self.step_count += 1;
        self.memory_usage_history.push(memory_usage);
        self.performance_history.push(performance_score);

        // Keep only recent history
        if self.memory_usage_history.len() > 50 {
            self.memory_usage_history.remove(0);
            self.performance_history.remove(0);
        }

        // Adapt batch size
        self.adapt_batch_size();
    }

    /// Adapt batch size based on memory usage and performance
    fn adapt_batch_size(&mut self) {
        let avg_memory =
            self.memory_usage_history.iter().sum::<f32>() / self.memory_usage_history.len() as f32;
        let avg_performance =
            self.performance_history.iter().sum::<f32>() / self.performance_history.len() as f32;

        let old_batch_size = self.current_batch_size;

        if avg_memory > self.config.memory_threshold {
            // Reduce batch size
            let reduction = (self.current_batch_size as f32 * self.config.adaptation_rate) as usize;
            self.current_batch_size =
                (self.current_batch_size - reduction).max(self.config.min_batch_size);
        } else if avg_performance > self.config.performance_threshold {
            // Increase batch size
            let increase = (self.current_batch_size as f32 * self.config.adaptation_rate) as usize;
            self.current_batch_size =
                (self.current_batch_size + increase).min(self.config.max_batch_size);
        }

        if self.current_batch_size != old_batch_size {
            self.batch_size_history.push((self.step_count, self.current_batch_size));
        }
    }

    /// Get current batch size
    pub fn get_current_batch_size(&self) -> usize {
        self.current_batch_size
    }

    /// Get batch size history
    pub fn get_batch_size_history(&self) -> &[(usize, usize)] {
        &self.batch_size_history
    }

    /// Generate batching report
    pub fn generate_report(&self) -> DynamicBatchingReport {
        let avg_memory = if !self.memory_usage_history.is_empty() {
            self.memory_usage_history.iter().sum::<f32>() / self.memory_usage_history.len() as f32
        } else {
            0.0
        };

        let avg_performance = if !self.performance_history.is_empty() {
            self.performance_history.iter().sum::<f32>() / self.performance_history.len() as f32
        } else {
            0.0
        };

        DynamicBatchingReport {
            current_batch_size: self.current_batch_size,
            step_count: self.step_count,
            avg_memory_usage: avg_memory,
            avg_performance,
            batch_size_changes: self.batch_size_history.len(),
            memory_efficiency: 1.0 - avg_memory,
            performance_score: avg_performance,
        }
    }
}

/// Dynamic batching report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicBatchingReport {
    pub current_batch_size: usize,
    pub step_count: usize,
    pub avg_memory_usage: f32,
    pub avg_performance: f32,
    pub batch_size_changes: usize,
    pub memory_efficiency: f32,
    pub performance_score: f32,
}

/// Compute optimization manager
#[derive(Debug)]
pub struct ComputeOptimizationManager {
    mixed_precision_manager: AdvancedMixedPrecisionManager,
    dynamic_batching_manager: DynamicBatchingManager,
    kernel_fusion_enabled: bool,
    pipeline_optimization_enabled: bool,
}

impl ComputeOptimizationManager {
    pub fn new(
        mixed_precision_config: AdvancedMixedPrecisionConfig,
        dynamic_batching_config: DynamicBatchingConfig,
    ) -> Self {
        Self {
            mixed_precision_manager: AdvancedMixedPrecisionManager::new(mixed_precision_config),
            dynamic_batching_manager: DynamicBatchingManager::new(dynamic_batching_config),
            kernel_fusion_enabled: true,
            pipeline_optimization_enabled: true,
        }
    }

    /// Update with training step information
    pub fn update_step(&mut self, memory_usage: f32, performance_score: f32) {
        self.mixed_precision_manager.update_step(memory_usage, performance_score);
        self.dynamic_batching_manager.update_step(memory_usage, performance_score);
    }

    /// Get current batch size
    pub fn get_current_batch_size(&self) -> usize {
        self.dynamic_batching_manager.get_current_batch_size()
    }

    /// Get current precision
    pub fn get_current_precision(&self) -> &str {
        self.mixed_precision_manager.get_current_precision()
    }

    /// Generate comprehensive optimization report
    pub fn generate_report(&self) -> ComputeOptimizationReport {
        ComputeOptimizationReport {
            mixed_precision_report: self.mixed_precision_manager.generate_report(),
            dynamic_batching_report: self.dynamic_batching_manager.generate_report(),
            kernel_fusion_enabled: self.kernel_fusion_enabled,
            pipeline_optimization_enabled: self.pipeline_optimization_enabled,
        }
    }
}

/// Comprehensive compute optimization report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeOptimizationReport {
    pub mixed_precision_report: MixedPrecisionReport,
    pub dynamic_batching_report: DynamicBatchingReport,
    pub kernel_fusion_enabled: bool,
    pub pipeline_optimization_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_mixed_precision_config_default() {
        let config = MixedPrecisionConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.init_scale, 65536.0);
        assert_eq!(config.scale_factor, 2.0);
        assert_eq!(config.backoff_factor, 0.5);
        assert_eq!(config.scale_window, 2000);
    }

    #[test]
    fn test_loss_scaler_creation() {
        let config = MixedPrecisionConfig::default();
        let scaler = LossScaler::new(config);
        assert_eq!(scaler.get_scale(), 1.0); // Disabled by default
    }

    #[test]
    fn test_loss_scaler_enabled() {
        let config = MixedPrecisionConfig {
            enabled: true,
            ..MixedPrecisionConfig::default()
        };
        let scaler = LossScaler::new(config);
        assert_eq!(scaler.get_scale(), 65536.0);
    }

    #[test]
    fn test_loss_scaling() {
        let config = MixedPrecisionConfig {
            enabled: true,
            ..MixedPrecisionConfig::default()
        };
        let scaler = LossScaler::new(config);

        let loss = Tensor::ones(&[2, 2]).expect("tensor operation failed");
        let scaled_loss = scaler.scale_loss(&loss).expect("operation failed in test");

        match (&loss, &scaled_loss) {
            (Tensor::F32(orig), Tensor::F32(scaled)) => {
                // Values should be scaled by the loss scale factor
                assert!((scaled[[0, 0]] / orig[[0, 0]] - 65536.0).abs() < 1e-6);
            },
            _ => panic!("Unexpected tensor types"),
        }
    }

    #[test]
    fn test_gradient_unscaling() {
        let config = MixedPrecisionConfig {
            enabled: true,
            ..MixedPrecisionConfig::default()
        };
        let scaler = LossScaler::new(config);

        let mut gradients = HashMap::new();
        gradients.insert(
            "param1".to_string(),
            Tensor::ones(&[2, 2]).expect("tensor operation failed"),
        );

        let overflow = scaler.unscale_gradients(&mut gradients).expect("operation failed in test");
        assert!(!overflow);

        // Check that gradients are unscaled
        let gradient = gradients.get("param1").expect("expected value not found");
        match gradient {
            Tensor::F32(arr) => {
                assert!((arr[[0, 0]] - 1.0 / 65536.0).abs() < 1e-6);
            },
            _ => panic!("Unexpected tensor type"),
        }
    }

    #[test]
    fn test_amp_manager_creation() {
        let config = MixedPrecisionConfig::default();
        let manager = AMPManager::new(config);
        assert!(!manager.is_enabled());
    }

    #[test]
    fn test_amp_manager_enabled() {
        let config = MixedPrecisionConfig {
            enabled: true,
            ..MixedPrecisionConfig::default()
        };
        let manager = AMPManager::new(config);
        assert!(manager.is_enabled());
        assert_eq!(manager.get_loss_scale(), 65536.0);
    }

    #[test]
    fn test_half_precision_conversion() {
        let config = utils::default_fp16_config();
        let manager = AMPManager::new(config);

        let tensor = Tensor::from_vec(vec![1.0, 2.5, -3.7, 1000.0], &[2, 2])
            .expect("tensor operation failed");
        let half_precision = manager.to_half_precision(&tensor).expect("tensor operation failed");
        let full_precision =
            manager.to_full_precision(&half_precision).expect("operation failed in test");

        // Values should be quantized but still reasonable
        match (&tensor, &full_precision) {
            (Tensor::F32(orig), Tensor::F32(converted)) => {
                for (o, c) in orig.iter().zip(converted.iter()) {
                    assert!((o - c).abs() < 0.1); // Some precision loss expected
                }
            },
            _ => panic!("Unexpected tensor types"),
        }
    }

    #[test]
    fn test_fp16_safety_check() {
        let safe_tensor =
            Tensor::from_vec(vec![1.0, 2.0, -3.0], &[3]).expect("tensor operation failed");
        assert!(utils::is_fp16_safe(&safe_tensor).expect("tensor operation failed"));

        let unsafe_tensor =
            Tensor::from_vec(vec![1.0, 70000.0, -3.0], &[3]).expect("tensor operation failed");
        assert!(!utils::is_fp16_safe(&unsafe_tensor).expect("tensor operation failed"));
    }

    #[test]
    fn test_dynamic_range_calculation() {
        let tensor =
            Tensor::from_vec(vec![1.0, 5.0, -2.0, 3.0], &[2, 2]).expect("tensor operation failed");
        let (min_val, max_val) =
            utils::calculate_dynamic_range(&tensor).expect("tensor operation failed");
        assert_eq!(min_val, -2.0);
        assert_eq!(max_val, 5.0);
    }

    // ── Regression tests for the two-pass unscale and the real half-precision cast ──

    #[test]
    fn test_unscale_leaves_every_gradient_untouched_on_overflow() {
        // Regression: the old loop unscaled a HashMap-iteration-order-dependent prefix
        // before breaking out on the first inf/nan, so a caller that kept the map ended up
        // mixing scaled and unscaled gradients. With many clean entries around one dirty
        // one, the old code left at least one entry divided by 65536 with high probability;
        // the fixed code must leave *all* of them exactly as they were.
        let config = MixedPrecisionConfig {
            enabled: true,
            skip_inf_nan: true,
            ..MixedPrecisionConfig::default()
        };
        let scaler = LossScaler::new(config);

        let mut gradients = HashMap::new();
        for i in 0..16 {
            gradients.insert(
                format!("clean_{i}"),
                Tensor::ones(&[2, 2]).expect("tensor creation failed"),
            );
        }
        gradients.insert(
            "dirty".to_string(),
            Tensor::from_vec(vec![f32::INFINITY, 1.0, 1.0, 1.0], &[2, 2])
                .expect("tensor creation failed"),
        );

        let overflow = scaler.unscale_gradients(&mut gradients).expect("unscale failed");
        assert!(overflow, "an inf gradient must be reported as an overflow");

        for i in 0..16 {
            let g = gradients.get(&format!("clean_{i}")).expect("gradient missing");
            let v = g.get_scalar(&[0, 0]).expect("get_scalar failed");
            assert!(
                (v - 1.0).abs() < 1e-9,
                "gradient clean_{i} must be left fully scaled on overflow, got {v}"
            );
        }
    }

    #[test]
    fn test_unscale_divides_every_gradient_when_clean() {
        let config = MixedPrecisionConfig {
            enabled: true,
            ..MixedPrecisionConfig::default()
        };
        let scaler = LossScaler::new(config);

        let mut gradients = HashMap::new();
        for i in 0..4 {
            gradients.insert(
                format!("p{i}"),
                Tensor::ones(&[2]).expect("tensor creation failed"),
            );
        }

        let overflow = scaler.unscale_gradients(&mut gradients).expect("unscale failed");
        assert!(!overflow);
        for i in 0..4 {
            let g = gradients.get(&format!("p{i}")).expect("gradient missing");
            let v = g.get_scalar(&[0]).expect("get_scalar failed");
            assert!((v - 1.0 / 65536.0).abs() < 1e-12, "p{i} not unscaled: {v}");
        }
    }

    #[test]
    fn test_to_half_precision_yields_an_actual_f16_tensor() {
        // Regression: the old implementation returned Tensor::F32 rounded to a 1/1024 grid,
        // which is neither an fp16 dtype nor fp16 semantics.
        let manager = AMPManager::new(utils::default_fp16_config());
        let tensor =
            Tensor::from_vec(vec![1.0, -2.5, 3.25, 0.5], &[4]).expect("tensor creation failed");
        let half = manager.to_half_precision(&tensor).expect("cast failed");
        assert!(
            matches!(half, Tensor::F16(_)),
            "to_half_precision must produce Tensor::F16, got {half:?}"
        );
        assert_eq!(half.shape(), &[4]);

        let back = manager.to_full_precision(&half).expect("widen failed");
        assert!(matches!(back, Tensor::F32(_)));
        // 1.0, -2.5, 3.25 and 0.5 are all exactly representable in binary16.
        let round_tripped = back.data().expect("data failed");
        for (o, r) in [1.0f32, -2.5, 3.25, 0.5].iter().zip(round_tripped.iter()) {
            assert!((o - r).abs() < 1e-6, "expected {o}, got {r}");
        }
    }

    #[test]
    fn test_to_half_precision_keeps_small_magnitudes() {
        // fp16 has *relative* precision: 1e-4 survives a round trip to ~4 significant
        // digits. The old absolute 1/1024 grid flushed it straight to 0.
        let manager = AMPManager::new(utils::default_fp16_config());
        let tensor = Tensor::from_vec(vec![1e-4], &[1]).expect("tensor creation failed");
        let back = manager
            .to_full_precision(&manager.to_half_precision(&tensor).expect("cast failed"))
            .expect("widen failed");
        let v = back.get_scalar(&[0]).expect("get_scalar failed");
        assert!(v > 0.0, "1e-4 must survive an fp16 round trip, got {v}");
        assert!(
            (v - 1e-4).abs() / 1e-4 < 1e-2,
            "fp16 round trip should be within 1% relative, got {v}"
        );
    }

    #[test]
    fn test_bf16_config_selects_bf16_dtype() {
        let manager = AMPManager::new(utils::default_bf16_config());
        let tensor = Tensor::from_vec(vec![1.0, 2.0], &[2]).expect("tensor creation failed");
        let half = manager.to_half_precision(&tensor).expect("cast failed");
        assert!(
            matches!(half, Tensor::BF16(_)),
            "expected BF16, got {half:?}"
        );
    }

    #[test]
    fn test_forward_with_amp_casts_down_and_back_up() {
        let manager = AMPManager::new(utils::default_fp16_config());
        let input = Tensor::from_vec(vec![1.0, 2.0], &[2]).expect("tensor creation failed");
        let mut seen_half = false;
        let out = manager
            .forward_with_amp(&input, |t| {
                seen_half = matches!(t, Tensor::F16(_));
                Ok(t)
            })
            .expect("forward failed");
        assert!(seen_half, "the closure must observe a half-precision input");
        assert!(
            matches!(out, Tensor::F32(_)),
            "the output must be widened back to f32"
        );
    }

    #[test]
    fn test_backward_with_amp_invokes_the_gradient_hook_with_a_scaled_loss() {
        let mut manager = AMPManager::new(utils::default_fp16_config());
        let loss = Tensor::from_vec(vec![2.0], &[1]).expect("tensor creation failed");

        let (grads, overflow) = manager
            .backward_with_amp(&loss, |scaled| {
                // The hook sees loss * 65536 and returns d(loss)/dp = scaled loss here.
                let v = scaled.get_scalar(&[0]).expect("get_scalar failed");
                assert!(
                    (v - 2.0 * 65536.0).abs() < 1e-3,
                    "hook must receive the scaled loss, got {v}"
                );
                let mut m = HashMap::new();
                m.insert("w".to_string(), scaled.clone());
                Ok(m)
            })
            .expect("backward failed");

        assert!(!overflow);
        let g = grads.get("w").expect("gradient missing");
        let v = g.get_scalar(&[0]).expect("get_scalar failed");
        assert!(
            (v - 2.0).abs() < 1e-4,
            "gradient must come back unscaled, got {v}"
        );
    }

    #[test]
    fn test_overflow_detection_and_scale_update() {
        let config = MixedPrecisionConfig {
            enabled: true,
            backoff_factor: 0.5,
            ..MixedPrecisionConfig::default()
        };
        let mut scaler = LossScaler::new(config);

        let initial_scale = scaler.get_scale();

        // Simulate overflow
        scaler.update_scale(true).expect("operation failed in test");
        assert_eq!(scaler.get_scale(), initial_scale * 0.5);
        assert!(scaler.overflow_detected());

        // Simulate no overflow
        scaler.update_scale(false).expect("operation failed in test");
        assert!(!scaler.overflow_detected());
    }
}
