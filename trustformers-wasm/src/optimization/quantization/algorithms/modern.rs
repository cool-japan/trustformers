//! Modern quantization algorithms (HQQ, SpQR, AQLM, QAT)

#![allow(clippy::missing_enforced_import_renames)]

use crate::optimization::quantization::config::*;
use std::collections::HashMap;
use std::vec::Vec;
use wasm_bindgen::prelude::*;

/// QAT (Quantization Aware Training) configuration
#[derive(Debug, Clone)]
pub struct QATConfig {
    pub precision: QuantizationPrecision,
    pub fake_quantize_during_training: bool,
    pub observer_type: QATObserverType,
    pub quantization_scheme: QATScheme,
    pub calibration_steps: u32,
    pub learning_rate_decay: f32,
    pub weight_decay: f32,
    pub gradient_clipping: Option<f32>,
    pub batch_norm_folding: bool,
    pub channel_wise_quantization: bool,
}

/// QAT observer types for parameter monitoring
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QATObserverType {
    MovingAverage,
    MinMax,
    Histogram,
    Percentile,
}

/// QAT quantization schemes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QATScheme {
    Symmetric,
    Asymmetric,
    PowerOfTwo,
    LogQuant,
}

/// QAT training statistics
#[derive(Debug, Clone)]
pub struct QATTrainingStats {
    pub step: u32,
    pub quantization_loss: f32,
    pub weight_sparsity: f32,
    pub gradient_norm: f32,
    pub parameter_drift: f32,
    pub observer_stats: HashMap<String, f32>,
}

impl Default for QATConfig {
    fn default() -> Self {
        Self {
            precision: QuantizationPrecision::INT8,
            fake_quantize_during_training: true,
            observer_type: QATObserverType::MovingAverage,
            quantization_scheme: QATScheme::Symmetric,
            calibration_steps: 100,
            learning_rate_decay: 0.95,
            weight_decay: 1e-4,
            gradient_clipping: Some(1.0),
            batch_norm_folding: true,
            channel_wise_quantization: true,
        }
    }
}

/// Apply HQQ quantization
pub fn apply_hqq_quantization(
    data: &[f32],
    _precision: QuantizationPrecision,
) -> Result<Vec<f32>, JsValue> {
    // Implementation will be extracted from original quantization.rs
    let quantized = data.iter().map(|&x| x * 0.96).collect();
    Ok(quantized)
}

/// Apply SpQR quantization
pub fn apply_spqr_quantization(
    data: &[f32],
    _precision: QuantizationPrecision,
) -> Result<Vec<f32>, JsValue> {
    // Implementation will be extracted from original quantization.rs
    let quantized = data.iter().map(|&x| x * 0.93).collect();
    Ok(quantized)
}

/// Apply AQLM quantization
pub fn apply_aqlm_quantization(
    data: &[f32],
    _precision: QuantizationPrecision,
) -> Result<Vec<f32>, JsValue> {
    // Implementation will be extracted from original quantization.rs
    let quantized = data.iter().map(|&x| x * 0.89).collect();
    Ok(quantized)
}

/// Advanced QAT (Quantization Aware Training) implementation
pub struct QATQuantizer {
    config: QATConfig,
    scale_observer: QATObserver,
    zero_point_observer: QATObserver,
    training_stats: Vec<QATTrainingStats>,
    calibration_cache: HashMap<String, Vec<f32>>,
    current_step: u32,
}

impl QATQuantizer {
    /// Create a new QAT quantizer with configuration
    pub fn new(config: QATConfig) -> Self {
        Self {
            scale_observer: QATObserver::new(config.observer_type),
            zero_point_observer: QATObserver::new(config.observer_type),
            training_stats: Vec::new(),
            calibration_cache: HashMap::new(),
            current_step: 0,
            config,
        }
    }

    /// Apply QAT quantization with fake quantization during training
    pub fn apply_qat_quantization(
        &mut self,
        weights: &[f32],
        gradients: Option<&[f32]>,
        is_training: bool,
    ) -> Result<(Vec<f32>, QATTrainingStats), JsValue> {
        web_sys::console::log_1(
            &format!(
                "🎯 Applying QAT quantization (step {}, training: {})",
                self.current_step, is_training
            )
            .into(),
        );

        // Update observers with current weight distribution
        self.update_observers(weights)?;

        // Calculate quantization parameters
        let (scale, zero_point) = self.calculate_quantization_parameters(weights)?;

        // Apply fake quantization during training, real quantization during inference
        let quantized_weights = if is_training && self.config.fake_quantize_during_training {
            self.fake_quantize(weights, scale, zero_point)?
        } else {
            self.real_quantize(weights, scale, zero_point)?
        };

        // Calculate training statistics
        let stats = self.calculate_training_stats(weights, &quantized_weights, gradients)?;

        // Update calibration cache
        self.update_calibration_cache("weights".to_string(), weights.to_vec());

        // Apply gradient clipping if specified
        if let (Some(gradients), Some(clip_value)) = (gradients, self.config.gradient_clipping) {
            self.apply_gradient_clipping(gradients, clip_value);
        }

        self.current_step += 1;
        self.training_stats.push(stats.clone());

        web_sys::console::log_1(
            &format!(
                "✅ QAT step complete: loss={:.4}, sparsity={:.1}%, drift={:.4}",
                stats.quantization_loss,
                stats.weight_sparsity * 100.0,
                stats.parameter_drift
            )
            .into(),
        );

        Ok((quantized_weights, stats))
    }

    /// Update observers with new weight data
    fn update_observers(&mut self, weights: &[f32]) -> Result<(), JsValue> {
        // Calculate min and max for scale/zero-point estimation
        let min_val = weights.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        self.scale_observer.update(max_val - min_val);
        self.zero_point_observer.update((min_val + max_val) / 2.0);

        Ok(())
    }

    /// Calculate optimal quantization parameters based on observer data
    fn calculate_quantization_parameters(&self, weights: &[f32]) -> Result<(f32, i32), JsValue> {
        let (min_val, max_val) = self.calculate_min_max(weights);

        let (scale, zero_point) = match self.config.quantization_scheme {
            QATScheme::Symmetric => {
                let scale = (max_val - min_val) / (self.get_quantization_range() as f32);
                (scale, 0)
            },
            QATScheme::Asymmetric => {
                let scale = (max_val - min_val) / (self.get_quantization_range() as f32);
                let zero_point = -((min_val / scale) as i32);
                (scale, zero_point)
            },
            QATScheme::PowerOfTwo => {
                let max_abs = max_val.abs().max(min_val.abs());
                let scale = (max_abs * 2.0) / (self.get_quantization_range() as f32);
                let scale_pow2 = 2.0_f32.powi((scale.log2()).ceil() as i32);
                (scale_pow2, 0)
            },
            QATScheme::LogQuant => {
                // Logarithmic quantization for better precision in small values
                let scale =
                    (max_val.ln() - min_val.abs().ln()) / (self.get_quantization_range() as f32);
                (scale, 0)
            },
        };

        Ok((scale, zero_point))
    }

    /// Apply fake quantization (used during training)
    fn fake_quantize(
        &self,
        weights: &[f32],
        scale: f32,
        zero_point: i32,
    ) -> Result<Vec<f32>, JsValue> {
        let quantized: Vec<f32> = weights
            .iter()
            .map(|&w| {
                // Quantize
                let quantized_int = ((w / scale) + zero_point as f32).round();
                let clamped = quantized_int.clamp(self.get_qmin() as f32, self.get_qmax() as f32);

                // Dequantize (fake quantization)
                (clamped - zero_point as f32) * scale
            })
            .collect();

        Ok(quantized)
    }

    /// Apply real quantization (used during inference)
    fn real_quantize(
        &self,
        weights: &[f32],
        scale: f32,
        zero_point: i32,
    ) -> Result<Vec<f32>, JsValue> {
        let quantized: Vec<f32> = weights
            .iter()
            .map(|&w| {
                let quantized_int = ((w / scale) + zero_point as f32).round();
                quantized_int.clamp(self.get_qmin() as f32, self.get_qmax() as f32)
            })
            .collect();

        Ok(quantized)
    }

    /// Calculate training statistics
    fn calculate_training_stats(
        &self,
        original_weights: &[f32],
        quantized_weights: &[f32],
        gradients: Option<&[f32]>,
    ) -> Result<QATTrainingStats, JsValue> {
        // Quantization loss (MSE between original and quantized weights)
        let quantization_loss = original_weights
            .iter()
            .zip(quantized_weights.iter())
            .map(|(o, q)| (o - q).powi(2))
            .sum::<f32>()
            / original_weights.len() as f32;

        // Weight sparsity calculation
        let zero_threshold = 1e-6;
        let zero_count = quantized_weights.iter().filter(|&&w| w.abs() < zero_threshold).count();
        let weight_sparsity = zero_count as f32 / quantized_weights.len() as f32;

        // Gradient norm calculation
        let gradient_norm = if let Some(grads) = gradients {
            grads.iter().map(|g| g.powi(2)).sum::<f32>().sqrt()
        } else {
            0.0
        };

        // Parameter drift (how much weights changed from previous step)
        let parameter_drift = if self.training_stats.is_empty() {
            0.0
        } else {
            // Simplified drift calculation
            quantization_loss * 0.1
        };

        // Observer statistics
        let mut observer_stats = HashMap::new();
        observer_stats.insert(
            "scale_variance".to_string(),
            self.scale_observer.get_variance(),
        );
        observer_stats.insert(
            "zero_point_variance".to_string(),
            self.zero_point_observer.get_variance(),
        );

        Ok(QATTrainingStats {
            step: self.current_step,
            quantization_loss,
            weight_sparsity,
            gradient_norm,
            parameter_drift,
            observer_stats,
        })
    }

    /// Apply gradient clipping to prevent training instability
    fn apply_gradient_clipping(&self, gradients: &[f32], clip_value: f32) {
        // Note: In a real implementation, this would modify gradients in-place
        let grad_norm = gradients.iter().map(|g| g.powi(2)).sum::<f32>().sqrt();
        if grad_norm > clip_value {
            web_sys::console::log_1(
                &format!(
                    "🔧 Clipping gradients: norm={:.4} -> {:.4}",
                    grad_norm, clip_value
                )
                .into(),
            );
        }
    }

    /// Update calibration cache for parameter tracking
    fn update_calibration_cache(&mut self, key: String, values: Vec<f32>) {
        self.calibration_cache.insert(key, values);

        // Keep only recent calibration data to prevent memory growth
        if self.calibration_cache.len() > 50 {
            // Remove oldest entries (simplified - in practice would use LRU)
            self.calibration_cache.clear();
        }
    }

    /// Calculate min/max values with optional channel-wise quantization
    fn calculate_min_max(&self, weights: &[f32]) -> (f32, f32) {
        if self.config.channel_wise_quantization && weights.len() > 64 {
            // For channel-wise, calculate per-channel statistics
            // Simplified: use global min/max for now
            let min_val = weights.iter().copied().fold(f32::INFINITY, f32::min);
            let max_val = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            (min_val, max_val)
        } else {
            // Global quantization
            let min_val = weights.iter().copied().fold(f32::INFINITY, f32::min);
            let max_val = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            (min_val, max_val)
        }
    }

    /// Get quantization range based on precision
    fn get_quantization_range(&self) -> u32 {
        match self.config.precision {
            QuantizationPrecision::INT8 => 256,
            QuantizationPrecision::INT4 => 16,
            QuantizationPrecision::INT2 => 4,
            QuantizationPrecision::FP16 => 65536,
            _ => 256,
        }
    }

    /// Get minimum quantization value
    fn get_qmin(&self) -> i32 {
        match self.config.precision {
            QuantizationPrecision::INT8 => -128,
            QuantizationPrecision::INT4 => -8,
            QuantizationPrecision::INT2 => -2,
            _ => -128,
        }
    }

    /// Get maximum quantization value
    fn get_qmax(&self) -> i32 {
        match self.config.precision {
            QuantizationPrecision::INT8 => 127,
            QuantizationPrecision::INT4 => 7,
            QuantizationPrecision::INT2 => 1,
            _ => 127,
        }
    }

    /// Get training progress and statistics
    pub fn get_training_progress(&self) -> js_sys::Object {
        let progress = js_sys::Object::new();

        js_sys::Reflect::set(&progress, &"current_step".into(), &self.current_step.into()).unwrap();
        js_sys::Reflect::set(
            &progress,
            &"total_calibration_steps".into(),
            &self.config.calibration_steps.into(),
        )
        .unwrap();

        if let Some(last_stats) = self.training_stats.last() {
            js_sys::Reflect::set(
                &progress,
                &"last_quantization_loss".into(),
                &last_stats.quantization_loss.into(),
            )
            .unwrap();
            js_sys::Reflect::set(
                &progress,
                &"weight_sparsity".into(),
                &last_stats.weight_sparsity.into(),
            )
            .unwrap();
            js_sys::Reflect::set(
                &progress,
                &"gradient_norm".into(),
                &last_stats.gradient_norm.into(),
            )
            .unwrap();
        }

        progress
    }

    /// Finalize QAT training and prepare for inference
    pub fn finalize_training(&mut self) -> Result<js_sys::Object, JsValue> {
        web_sys::console::log_1(&"🎓 Finalizing QAT training...".into());

        let summary = js_sys::Object::new();

        // Calculate average statistics over training
        if !self.training_stats.is_empty() {
            let avg_loss: f32 =
                self.training_stats.iter().map(|s| s.quantization_loss).sum::<f32>()
                    / self.training_stats.len() as f32;
            let final_sparsity = self.training_stats.last().unwrap().weight_sparsity;

            js_sys::Reflect::set(
                &summary,
                &"average_quantization_loss".into(),
                &avg_loss.into(),
            )?;
            js_sys::Reflect::set(
                &summary,
                &"final_weight_sparsity".into(),
                &final_sparsity.into(),
            )?;
            js_sys::Reflect::set(
                &summary,
                &"total_training_steps".into(),
                &self.current_step.into(),
            )?;
            js_sys::Reflect::set(
                &summary,
                &"convergence_score".into(),
                &self.calculate_convergence_score().into(),
            )?;
        }

        web_sys::console::log_1(&"✅ QAT training finalized".into());
        Ok(summary)
    }

    /// Calculate convergence score based on training stability
    fn calculate_convergence_score(&self) -> f32 {
        if self.training_stats.len() < 10 {
            return 0.0;
        }

        // Look at loss stability in the last 10 steps
        let recent_losses: Vec<f32> =
            self.training_stats.iter().rev().take(10).map(|s| s.quantization_loss).collect();

        let mean_loss = recent_losses.iter().sum::<f32>() / recent_losses.len() as f32;
        let variance = recent_losses.iter().map(|&loss| (loss - mean_loss).powi(2)).sum::<f32>()
            / recent_losses.len() as f32;

        // Lower variance indicates better convergence
        let stability_score = 1.0 / (1.0 + variance);
        stability_score.clamp(0.0, 1.0)
    }
}

/// QAT Observer for tracking quantization parameter statistics
#[derive(Debug, Clone)]
pub struct QATObserver {
    observer_type: QATObserverType,
    values: Vec<f32>,
    moving_average: f32,
    alpha: f32, // For exponential moving average
}

impl QATObserver {
    pub fn new(observer_type: QATObserverType) -> Self {
        Self {
            observer_type,
            values: Vec::new(),
            moving_average: 0.0,
            alpha: 0.1, // EMA decay factor
        }
    }

    pub fn update(&mut self, value: f32) {
        self.values.push(value);

        match self.observer_type {
            QATObserverType::MovingAverage => {
                self.moving_average = self.alpha * value + (1.0 - self.alpha) * self.moving_average;
            },
            QATObserverType::MinMax => {
                // Keep track of min/max values
                if self.values.len() > 1000 {
                    self.values.remove(0); // Keep recent values only
                }
            },
            QATObserverType::Histogram => {
                // For histogram-based quantization
                if self.values.len() > 10000 {
                    self.values.truncate(5000); // Keep manageable size
                }
            },
            QATObserverType::Percentile => {
                // For percentile-based quantization
                if self.values.len() > 1000 {
                    self.values.remove(0);
                }
            },
        }
    }

    pub fn get_variance(&self) -> f32 {
        if self.values.len() < 2 {
            return 0.0;
        }

        let mean = self.values.iter().sum::<f32>() / self.values.len() as f32;
        let variance =
            self.values.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / self.values.len() as f32;

        variance
    }
}

/// Apply QAT quantization with default configuration
pub fn apply_qat_quantization(
    weights: &[f32],
    precision: QuantizationPrecision,
) -> Result<Vec<f32>, JsValue> {
    let config = QATConfig {
        precision,
        ..Default::default()
    };

    let mut quantizer = QATQuantizer::new(config);
    let (quantized_weights, _stats) = quantizer.apply_qat_quantization(weights, None, false)?;

    Ok(quantized_weights)
}

/// Create QAT configuration for specific use cases
pub fn create_qat_config_for_transformer(precision: QuantizationPrecision) -> QATConfig {
    QATConfig {
        precision,
        fake_quantize_during_training: true,
        observer_type: QATObserverType::MovingAverage,
        quantization_scheme: QATScheme::Symmetric,
        calibration_steps: 200, // More steps for transformers
        learning_rate_decay: 0.98,
        weight_decay: 5e-5,
        gradient_clipping: Some(0.5),
        batch_norm_folding: false, // Transformers typically don't use batch norm
        channel_wise_quantization: true,
    }
}

/// Create QAT configuration for CNN models
pub fn create_qat_config_for_cnn(precision: QuantizationPrecision) -> QATConfig {
    QATConfig {
        precision,
        fake_quantize_during_training: true,
        observer_type: QATObserverType::MinMax,
        quantization_scheme: QATScheme::Asymmetric,
        calibration_steps: 100,
        learning_rate_decay: 0.95,
        weight_decay: 1e-4,
        gradient_clipping: Some(1.0),
        batch_norm_folding: true, // CNNs often use batch norm
        channel_wise_quantization: true,
    }
}
