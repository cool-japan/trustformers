//! Quantization-Aware Training (QAT) Infrastructure for TrustformeRS
//!
//! This module provides comprehensive quantization-aware training capabilities,
//! including fake quantization layers, QAT schedulers, and training utilities.

use crate::errors::{Result, TrustformersError};
use crate::quantization::{ActivationQuantScheme, QuantizationScheme};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// QAT training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QATConfig {
    /// Weight quantization scheme
    pub weight_scheme: QuantizationScheme,
    /// Activation quantization scheme
    pub activation_scheme: ActivationQuantScheme,
    /// Whether to use symmetric quantization
    pub symmetric: bool,
    /// Number of warmup epochs before enabling quantization
    pub warmup_epochs: usize,
    /// QAT schedule for gradual quantization introduction
    pub schedule: QATSchedule,
    /// Whether to quantize first and last layers
    pub quantize_first_last: bool,
    /// Observer configuration
    pub observer_config: ObserverConfig,
    /// Whether to use straight-through estimator
    pub use_ste: bool,
}

/// QAT training schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QATSchedule {
    /// Immediate quantization from start
    Immediate,
    /// Gradual introduction over epochs
    Gradual {
        start_epoch: usize,
        end_epoch: usize,
        weight_schedule: GradualSchedule,
        activation_schedule: GradualSchedule,
    },
    /// Custom layer-by-layer schedule
    LayerWise {
        schedule: HashMap<String, LayerSchedule>,
    },
    /// Progressive bit reduction
    Progressive {
        start_bits: u8,
        end_bits: u8,
        reduction_epochs: Vec<usize>,
    },
}

/// Gradual quantization schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GradualSchedule {
    /// Linear introduction
    Linear,
    /// Cosine schedule
    Cosine,
    /// Exponential schedule
    Exponential { base: f64 },
    /// Step-wise introduction
    Step { steps: Vec<usize> },
}

/// Layer-specific QAT schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSchedule {
    pub start_epoch: usize,
    pub enable_weights: bool,
    pub enable_activations: bool,
    pub bits: Option<u8>,
}

/// Observer configuration for calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObserverConfig {
    /// Moving average momentum for statistics
    pub momentum: f64,
    /// Whether to use percentile clipping
    pub use_percentile: bool,
    /// Percentile value for clipping (e.g., 0.999)
    pub percentile: f64,
    /// Minimum number of observations before quantization
    pub min_observations: usize,
    /// Whether to freeze observer after warmup
    pub freeze_after_warmup: bool,
}

/// Fake quantization layer for QAT
#[derive(Debug)]
pub struct FakeQuantLayer {
    /// Current bit width
    pub bits: u8,
    /// Whether quantization is enabled
    pub enabled: bool,
    /// Quantization scheme
    pub scheme: QuantizationScheme,
    /// Observer for collecting statistics
    pub observer: MovingAverageObserver,
    /// Quantization parameters
    pub scale: Option<f32>,
    pub zero_point: Option<i32>,
    /// Configuration
    pub config: QATConfig,
    /// Current epoch for schedule tracking
    pub current_epoch: usize,
}

/// Moving average observer for QAT
#[derive(Debug, Clone)]
pub struct MovingAverageObserver {
    /// Running minimum
    pub min_val: f32,
    /// Running maximum
    pub max_val: f32,
    /// Moving average momentum
    pub momentum: f64,
    /// Number of observations
    pub num_observations: usize,
    /// Whether observer is frozen
    pub frozen: bool,
    /// Configuration
    pub config: ObserverConfig,
}

/// QAT trainer for managing the training process
#[derive(Debug)]
pub struct QATTrainer {
    /// QAT configuration
    pub config: QATConfig,
    /// Fake quantization layers
    pub fake_quant_layers: HashMap<String, FakeQuantLayer>,
    /// Current training epoch
    pub current_epoch: usize,
    /// Training statistics
    pub stats: QATStats,
}

/// QAT training statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QATStats {
    /// Current quantization ratio (0.0 to 1.0)
    pub quantization_ratio: f64,
    /// Number of quantized layers
    pub quantized_layers: usize,
    /// Total number of layers
    pub total_layers: usize,
    /// Average bit width across layers
    pub average_bits: f64,
    /// Model size reduction ratio
    pub size_reduction: f64,
    /// Current training loss
    pub training_loss: f64,
    /// Validation accuracy
    pub validation_accuracy: f64,
}

impl Default for QATConfig {
    fn default() -> Self {
        Self {
            weight_scheme: QuantizationScheme::Dynamic,
            activation_scheme: ActivationQuantScheme::Int8,
            symmetric: false,
            warmup_epochs: 5,
            schedule: QATSchedule::Gradual {
                start_epoch: 5,
                end_epoch: 20,
                weight_schedule: GradualSchedule::Linear,
                activation_schedule: GradualSchedule::Linear,
            },
            quantize_first_last: false,
            observer_config: ObserverConfig::default(),
            use_ste: true,
        }
    }
}

impl Default for ObserverConfig {
    fn default() -> Self {
        Self {
            momentum: 0.01,
            use_percentile: true,
            percentile: 0.999,
            min_observations: 100,
            freeze_after_warmup: true,
        }
    }
}

impl MovingAverageObserver {
    /// Create new observer
    pub fn new(config: ObserverConfig) -> Self {
        Self {
            min_val: f32::INFINITY,
            max_val: f32::NEG_INFINITY,
            momentum: config.momentum,
            num_observations: 0,
            frozen: false,
            config,
        }
    }

    /// Update observer with new tensor
    pub fn update(&mut self, tensor: &Tensor) -> Result<()> {
        if self.frozen {
            return Ok(());
        }

        match tensor {
            Tensor::F32(arr) => {
                for &val in arr.iter() {
                    if !val.is_finite() {
                        continue;
                    }

                    if self.num_observations == 0 {
                        self.min_val = val;
                        self.max_val = val;
                    } else {
                        // Track actual min/max values
                        if val < self.min_val {
                            self.min_val = val;
                        }
                        if val > self.max_val {
                            self.max_val = val;
                        }
                    }
                    self.num_observations += 1;
                }
            },
            _ => {
                return Err(TrustformersError::quantization_error(
                    "Unsupported tensor type for observer".into(),
                ))
            },
        }

        Ok(())
    }

    /// Get quantization parameters
    pub fn get_quantization_params(&self, bits: u8, symmetric: bool) -> Result<(f32, i32)> {
        if self.num_observations < self.config.min_observations {
            return Err(TrustformersError::quantization_error(
                "Insufficient observations for quantization".into(),
            ));
        }

        let q_min = if symmetric { -(1 << (bits - 1)) } else { 0 };
        let q_max = if symmetric { (1 << (bits - 1)) - 1 } else { (1 << bits) - 1 };

        let (scale, zero_point) = if symmetric {
            let abs_max = self.max_val.abs().max(self.min_val.abs());
            if abs_max == 0.0 {
                return Ok((1.0, 0));
            }
            let scale = abs_max / (q_max - q_min) as f32;
            (scale, 0)
        } else {
            if self.max_val == self.min_val {
                return Ok((1.0, q_min));
            }
            let scale = (self.max_val - self.min_val) / (q_max - q_min) as f32;
            let zero_point = q_min - (self.min_val / scale).round() as i32;
            let zero_point = zero_point.clamp(q_min, q_max);
            (scale, zero_point)
        };

        Ok((scale, zero_point))
    }

    /// Freeze observer
    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    /// Check if observer is ready
    pub fn is_ready(&self) -> bool {
        self.num_observations >= self.config.min_observations
    }
}

impl FakeQuantLayer {
    /// Create new fake quantization layer
    pub fn new(bits: u8, scheme: QuantizationScheme, config: QATConfig) -> Self {
        Self {
            bits,
            enabled: false,
            scheme,
            observer: MovingAverageObserver::new(config.observer_config.clone()),
            scale: None,
            zero_point: None,
            config,
            current_epoch: 0,
        }
    }

    /// Update layer for current epoch
    pub fn update_epoch(&mut self, epoch: usize) {
        self.current_epoch = epoch;

        // Update quantization state based on schedule
        match &self.config.schedule {
            QATSchedule::Immediate => {
                if epoch >= self.config.warmup_epochs {
                    self.enabled = true;
                }
            },
            QATSchedule::Gradual { start_epoch, .. } => {
                if epoch >= *start_epoch {
                    self.enabled = true;
                }
            },
            QATSchedule::LayerWise { .. } => {
                // Layer-specific logic would be implemented here
                self.enabled = epoch >= self.config.warmup_epochs;
            },
            QATSchedule::Progressive {
                start_bits,
                end_bits,
                reduction_epochs,
            } => {
                self.enabled = epoch >= self.config.warmup_epochs;

                // Progressive bit reduction
                for (i, &reduction_epoch) in reduction_epochs.iter().enumerate() {
                    if epoch >= reduction_epoch {
                        let bits_reduction = (start_bits - end_bits) / reduction_epochs.len() as u8;
                        self.bits = (*start_bits - (i as u8 + 1) * bits_reduction).max(*end_bits);
                    }
                }
            },
        }

        // Freeze observer after warmup if configured
        if self.config.observer_config.freeze_after_warmup && epoch > self.config.warmup_epochs {
            self.observer.freeze();
        }
    }

    /// Apply fake quantization to tensor
    pub fn forward(&mut self, tensor: &Tensor, training: bool) -> Result<Tensor> {
        if training {
            // Update observer during training
            self.observer.update(tensor)?;
        }

        if !self.enabled || !self.observer.is_ready() {
            return Ok(tensor.clone());
        }

        // Get quantization parameters
        if self.scale.is_none() || self.zero_point.is_none() {
            let (scale, zero_point) =
                self.observer.get_quantization_params(self.bits, self.config.symmetric)?;
            self.scale = Some(scale);
            self.zero_point = Some(zero_point);
        }

        // Safe: scale and zero_point are set in the block above if None
        let scale = self.scale.expect("scale should be set after observer initialization");
        let zero_point =
            self.zero_point.expect("zero_point should be set after observer initialization");

        // Apply fake quantization with straight-through estimator
        self.fake_quantize(tensor, scale, zero_point)
    }

    /// Fake quantization with straight-through estimator
    fn fake_quantize(&self, tensor: &Tensor, scale: f32, zero_point: i32) -> Result<Tensor> {
        match tensor {
            Tensor::F32(arr) => {
                let q_min = if self.config.symmetric { -(1 << (self.bits - 1)) } else { 0 };
                let q_max = if self.config.symmetric {
                    (1 << (self.bits - 1)) - 1
                } else {
                    (1 << self.bits) - 1
                };

                let fake_quantized_data: Vec<f32> = arr
                    .iter()
                    .map(|&val| {
                        if self.config.use_ste {
                            // Straight-through estimator: forward pass quantized, backward pass identity
                            let q_val =
                                ((val / scale).round() as i32 + zero_point).clamp(q_min, q_max);
                            (q_val - zero_point) as f32 * scale
                        } else {
                            // Standard fake quantization
                            let q_val =
                                ((val / scale).round() as i32 + zero_point).clamp(q_min, q_max);
                            (q_val - zero_point) as f32 * scale
                        }
                    })
                    .collect();

                Tensor::from_vec(fake_quantized_data, arr.shape())
            },
            _ => Err(TrustformersError::quantization_error(
                "Unsupported tensor type for fake quantization".into(),
            )),
        }
    }

    /// Get current quantization parameters
    pub fn get_params(&self) -> Option<(f32, i32)> {
        if let (Some(scale), Some(zero_point)) = (self.scale, self.zero_point) {
            Some((scale, zero_point))
        } else {
            None
        }
    }
}

impl QATTrainer {
    /// Create new QAT trainer
    pub fn new(config: QATConfig) -> Self {
        Self {
            config,
            fake_quant_layers: HashMap::new(),
            current_epoch: 0,
            stats: QATStats::default(),
        }
    }

    /// Add fake quantization layer
    pub fn add_layer(&mut self, name: String, bits: u8, scheme: QuantizationScheme) {
        let layer = FakeQuantLayer::new(bits, scheme, self.config.clone());
        self.fake_quant_layers.insert(name, layer);
        self.update_stats();
    }

    /// Update epoch for all layers
    pub fn update_epoch(&mut self, epoch: usize) {
        self.current_epoch = epoch;

        for layer in self.fake_quant_layers.values_mut() {
            layer.update_epoch(epoch);
        }

        self.update_stats();
    }

    /// Apply fake quantization to tensor for specific layer
    pub fn quantize_layer(
        &mut self,
        layer_name: &str,
        tensor: &Tensor,
        training: bool,
    ) -> Result<Tensor> {
        if let Some(layer) = self.fake_quant_layers.get_mut(layer_name) {
            layer.forward(tensor, training)
        } else {
            Ok(tensor.clone())
        }
    }

    /// Get current quantization schedule value
    pub fn get_schedule_value(
        &self,
        schedule: &GradualSchedule,
        start_epoch: usize,
        end_epoch: usize,
    ) -> f64 {
        if self.current_epoch < start_epoch {
            return 0.0;
        }
        if self.current_epoch >= end_epoch {
            return 1.0;
        }

        let progress = (self.current_epoch - start_epoch) as f64 / (end_epoch - start_epoch) as f64;

        match schedule {
            GradualSchedule::Linear => progress,
            GradualSchedule::Cosine => 0.5 * (1.0 - (std::f64::consts::PI * progress).cos()),
            GradualSchedule::Exponential { base } => 1.0 - base.powf(progress),
            GradualSchedule::Step { steps } => {
                let current_step =
                    steps.iter().position(|&step| self.current_epoch < step).unwrap_or(steps.len());
                current_step as f64 / steps.len() as f64
            },
        }
    }

    /// Update training statistics
    fn update_stats(&mut self) {
        let total_layers = self.fake_quant_layers.len();
        let quantized_layers =
            self.fake_quant_layers.values().filter(|layer| layer.enabled).count();

        let average_bits = if total_layers > 0 {
            self.fake_quant_layers.values().map(|layer| layer.bits as f64).sum::<f64>()
                / total_layers as f64
        } else {
            0.0
        };

        let quantization_ratio = if total_layers > 0 {
            quantized_layers as f64 / total_layers as f64
        } else {
            0.0
        };

        // Estimate size reduction (simplified)
        let size_reduction = match average_bits as u8 {
            8 => 0.75,  // 32-bit to 8-bit
            16 => 0.5,  // 32-bit to 16-bit
            4 => 0.875, // 32-bit to 4-bit
            _ => 0.0,
        } * quantization_ratio;

        self.stats = QATStats {
            quantization_ratio,
            quantized_layers,
            total_layers,
            average_bits,
            size_reduction,
            training_loss: self.stats.training_loss, // Preserve current values
            validation_accuracy: self.stats.validation_accuracy,
        };
    }

    /// Update training metrics
    pub fn update_metrics(&mut self, training_loss: f64, validation_accuracy: f64) {
        self.stats.training_loss = training_loss;
        self.stats.validation_accuracy = validation_accuracy;
    }

    /// Get current statistics
    pub fn get_stats(&self) -> &QATStats {
        &self.stats
    }

    /// Check if QAT is ready (all observers have enough data)
    pub fn is_ready(&self) -> bool {
        self.fake_quant_layers.values().all(|layer| layer.observer.is_ready())
    }

    /// Export quantized model configuration
    pub fn export_quantized_config(&self) -> HashMap<String, (f32, i32, u8)> {
        self.fake_quant_layers
            .iter()
            .filter_map(|(name, layer)| {
                if let Some((scale, zero_point)) = layer.get_params() {
                    Some((name.clone(), (scale, zero_point, layer.bits)))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Save QAT state
    pub fn save_state(&self, path: &str) -> Result<()> {
        let state = QATState {
            config: self.config.clone(),
            current_epoch: self.current_epoch,
            stats: self.stats.clone(),
            layer_configs: self.export_quantized_config(),
        };

        let json_data = serde_json::to_string_pretty(&state).map_err(|e| {
            TrustformersError::quantization_error(format!("Failed to serialize QAT state: {}", e))
        })?;

        std::fs::write(path, json_data).map_err(|e| {
            TrustformersError::quantization_error(format!("Failed to write file: {}", e))
        })?;

        Ok(())
    }

    /// Load QAT state
    pub fn load_state(&mut self, path: &str) -> Result<()> {
        let json_data = std::fs::read_to_string(path).map_err(|e| {
            TrustformersError::quantization_error(format!("Failed to read file: {}", e))
        })?;

        let state: QATState = serde_json::from_str(&json_data).map_err(|e| {
            TrustformersError::quantization_error(format!("Failed to deserialize QAT state: {}", e))
        })?;

        self.config = state.config;
        self.current_epoch = state.current_epoch;
        self.stats = state.stats;

        // Restore layer configurations
        for (name, (scale, zero_point, bits)) in state.layer_configs {
            if let Some(layer) = self.fake_quant_layers.get_mut(&name) {
                layer.scale = Some(scale);
                layer.zero_point = Some(zero_point);
                layer.bits = bits;
            }
        }

        Ok(())
    }
}

impl Default for QATStats {
    fn default() -> Self {
        Self {
            quantization_ratio: 0.0,
            quantized_layers: 0,
            total_layers: 0,
            average_bits: 32.0,
            size_reduction: 0.0,
            training_loss: 0.0,
            validation_accuracy: 0.0,
        }
    }
}

/// Serializable QAT state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QATState {
    pub config: QATConfig,
    pub current_epoch: usize,
    pub stats: QATStats,
    pub layer_configs: HashMap<String, (f32, i32, u8)>, // (scale, zero_point, bits)
}

/// QAT utilities
pub struct QATUtils;

impl QATUtils {
    /// Create a progressive QAT schedule
    pub fn create_progressive_schedule(
        warmup_epochs: usize,
        total_epochs: usize,
        start_bits: u8,
        end_bits: u8,
    ) -> QATSchedule {
        let reduction_steps = (start_bits - end_bits) as usize;
        let epochs_per_step = (total_epochs - warmup_epochs) / reduction_steps.max(1);

        let reduction_epochs: Vec<usize> = (1..=reduction_steps)
            .map(|step| warmup_epochs + step * epochs_per_step)
            .collect();

        QATSchedule::Progressive {
            start_bits,
            end_bits,
            reduction_epochs,
        }
    }

    /// Create layer-wise schedule
    pub fn create_layerwise_schedule(
        layer_names: &[String],
        start_epoch: usize,
        epochs_between_layers: usize,
    ) -> QATSchedule {
        let mut schedule = HashMap::new();

        for (i, name) in layer_names.iter().enumerate() {
            let layer_start_epoch = start_epoch + i * epochs_between_layers;
            schedule.insert(
                name.clone(),
                LayerSchedule {
                    start_epoch: layer_start_epoch,
                    enable_weights: true,
                    enable_activations: true,
                    bits: Some(8),
                },
            );
        }

        QATSchedule::LayerWise { schedule }
    }

    /// Estimate model size reduction
    pub fn estimate_size_reduction(
        original_bits: u8,
        quantized_bits: u8,
        quantization_ratio: f64,
    ) -> f64 {
        let bit_reduction = 1.0 - (quantized_bits as f64 / original_bits as f64);
        bit_reduction * quantization_ratio
    }

    /// Calculate quantization noise
    pub fn calculate_quantization_noise(original: &Tensor, quantized: &Tensor) -> Result<f64> {
        match (original, quantized) {
            (Tensor::F32(orig_arr), Tensor::F32(quant_arr)) => {
                if orig_arr.len() != quant_arr.len() {
                    return Err(TrustformersError::quantization_error(
                        "Tensor sizes don't match".into(),
                    ));
                }

                let mse: f64 = orig_arr
                    .iter()
                    .zip(quant_arr.iter())
                    .map(|(&orig, &quant)| (orig - quant).powi(2) as f64)
                    .sum::<f64>()
                    / orig_arr.len() as f64;

                Ok(mse.sqrt()) // RMSE
            },
            _ => Err(TrustformersError::quantization_error(
                "Unsupported tensor types for noise calculation".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qat_config_default() {
        let config = QATConfig::default();
        assert_eq!(config.warmup_epochs, 5);
        assert!(!config.quantize_first_last);
        assert!(config.use_ste);
    }

    #[test]
    fn test_moving_average_observer() {
        let config = ObserverConfig::default();
        let mut observer = MovingAverageObserver::new(config);

        let tensor =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).expect("Tensor from_vec failed");
        observer.update(&tensor).unwrap();

        assert_eq!(observer.num_observations, 4);
        assert!(observer.min_val <= 1.0);
        assert!(observer.max_val >= 4.0);
    }

    #[test]
    fn test_fake_quant_layer() {
        let mut config = QATConfig::default();
        config.observer_config.freeze_after_warmup = false; // Don't freeze observer prematurely
        let mut layer = FakeQuantLayer::new(8, QuantizationScheme::DynamicINT8, config);

        // Should not be enabled initially
        assert!(!layer.enabled);

        // Update to after warmup
        layer.update_epoch(10);

        let tensor =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).expect("Tensor from_vec failed");

        // Simulate training to collect statistics
        for _ in 0..100 {
            layer.forward(&tensor, true).expect("Forward pass failed");
        }

        // Should be enabled and ready now
        assert!(layer.enabled);
        assert!(layer.observer.is_ready());
    }

    #[test]
    fn test_qat_trainer() {
        let config = QATConfig::default();
        let mut trainer = QATTrainer::new(config);

        trainer.add_layer("conv1".to_string(), 8, QuantizationScheme::DynamicINT8);
        trainer.add_layer("conv2".to_string(), 8, QuantizationScheme::DynamicINT8);

        let stats = trainer.get_stats();
        assert_eq!(stats.total_layers, 2);
        assert_eq!(stats.quantized_layers, 0); // Not enabled yet

        trainer.update_epoch(10);
        let stats = trainer.get_stats();
        assert_eq!(stats.quantized_layers, 2); // Should be enabled now
    }

    #[test]
    fn test_gradual_schedule() {
        let config = QATConfig::default();
        let trainer = QATTrainer::new(config);

        let schedule = GradualSchedule::Linear;
        let value = trainer.get_schedule_value(&schedule, 5, 15);
        // Should be between 0 and 1 for linear schedule
        assert!((0.0..=1.0).contains(&value));
    }

    #[test]
    fn test_qat_utils_progressive_schedule() {
        let schedule = QATUtils::create_progressive_schedule(5, 25, 16, 8);

        match schedule {
            QATSchedule::Progressive {
                start_bits,
                end_bits,
                reduction_epochs,
            } => {
                assert_eq!(start_bits, 16);
                assert_eq!(end_bits, 8);
                assert!(!reduction_epochs.is_empty());
            },
            _ => panic!("Expected progressive schedule"),
        }
    }

    #[test]
    fn test_size_reduction_estimation() {
        let reduction = QATUtils::estimate_size_reduction(32, 8, 1.0);
        assert_eq!(reduction, 0.75); // 75% reduction from 32-bit to 8-bit
    }

    #[test]
    fn test_quantization_noise_calculation() {
        let original =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).expect("Tensor from_vec failed");
        let quantized =
            Tensor::from_vec(vec![1.1, 1.9, 3.1, 3.9], &[4]).expect("Tensor from_vec failed");

        let noise = QATUtils::calculate_quantization_noise(&original, &quantized).unwrap();
        assert!(noise > 0.0);
        assert!(noise < 1.0); // Should be small for close values
    }

    #[test]
    fn test_layer_wise_schedule() {
        let layer_names = vec!["conv1".to_string(), "conv2".to_string(), "fc1".to_string()];
        let schedule = QATUtils::create_layerwise_schedule(&layer_names, 5, 2);

        match schedule {
            QATSchedule::LayerWise { schedule } => {
                assert_eq!(schedule.len(), 3);
                assert!(schedule.contains_key("conv1"));
                assert!(schedule.contains_key("conv2"));
                assert!(schedule.contains_key("fc1"));
            },
            _ => panic!("Expected layer-wise schedule"),
        }
    }
}
