//! Activation quantization for TrustformeRS
//!
//! This module provides activation quantization functionality, which quantizes intermediate
//! layer outputs during inference and training. Unlike weight quantization which is applied
//! to model parameters, activation quantization is applied dynamically to the data flowing
//! through the network.

use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for activation quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationQuantConfig {
    /// Quantization scheme for activations
    pub scheme: ActivationQuantScheme,
    /// Whether to use symmetric quantization
    pub symmetric: bool,
    /// Number of calibration samples to collect statistics
    pub calibration_samples: usize,
    /// Percentile for outlier-aware quantization (e.g., 0.99)
    pub percentile: f32,
    /// Moving average decay for running statistics
    pub ema_decay: f32,
    /// Whether to apply quantization during training
    pub quantize_during_training: bool,
    /// Layer-specific configurations
    pub layer_configs: HashMap<String, LayerQuantConfig>,
}

/// Activation quantization schemes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationQuantScheme {
    /// 8-bit integer quantization
    Int8,
    /// 16-bit integer quantization
    Int16,
    /// Dynamic range quantization
    Dynamic,
    /// Adaptive quantization based on activation distribution
    Adaptive,
}

/// Layer-specific quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerQuantConfig {
    /// Whether to quantize this layer's activations
    pub enabled: bool,
    /// Custom quantization scheme for this layer
    pub scheme: Option<ActivationQuantScheme>,
    /// Custom bit width (overrides scheme if provided)
    pub bits: Option<u8>,
    /// Whether to use layer-specific calibration
    pub calibrate: bool,
}

/// Statistics for activation quantization calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationStats {
    /// Running minimum value
    pub min_val: f32,
    /// Running maximum value
    pub max_val: f32,
    /// Running sum for mean calculation
    pub sum: f64,
    /// Running sum of squares for variance calculation
    pub sum_squares: f64,
    /// Number of samples observed
    pub count: usize,
    /// Histogram for percentile calculation
    pub histogram: Vec<(f32, usize)>,
    /// EMA of min/max values
    pub ema_min: f32,
    /// EMA of max values
    pub ema_max: f32,
}

/// Quantized activation tensor
#[derive(Debug, Clone)]
pub struct QuantizedActivation {
    /// Quantized data
    pub data: Vec<u8>,
    /// Quantization scale
    pub scale: f32,
    /// Zero point for asymmetric quantization
    pub zero_point: i32,
    /// Original tensor shape
    pub shape: Vec<usize>,
    /// Quantization scheme used
    pub scheme: ActivationQuantScheme,
    /// Number of bits used
    pub bits: u8,
}

/// Activation quantization manager
pub struct ActivationQuantizer {
    config: ActivationQuantConfig,
    /// Layer statistics for calibration
    layer_stats: HashMap<String, ActivationStats>,
    /// Whether calibration phase is active
    calibrating: bool,
    /// Number of calibration samples seen
    calibration_count: usize,
}

impl Default for ActivationQuantConfig {
    fn default() -> Self {
        Self {
            scheme: ActivationQuantScheme::Int8,
            symmetric: false,
            calibration_samples: 100,
            percentile: 0.99,
            ema_decay: 0.01,
            quantize_during_training: false,
            layer_configs: HashMap::new(),
        }
    }
}

impl Default for LayerQuantConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            scheme: None,
            bits: None,
            calibrate: true,
        }
    }
}

impl Default for ActivationStats {
    fn default() -> Self {
        Self::new()
    }
}

impl ActivationStats {
    /// Create new activation statistics
    pub fn new() -> Self {
        Self {
            min_val: f32::INFINITY,
            max_val: f32::NEG_INFINITY,
            sum: 0.0,
            sum_squares: 0.0,
            count: 0,
            histogram: Vec::new(),
            ema_min: f32::INFINITY,
            ema_max: f32::NEG_INFINITY,
        }
    }

    /// Update statistics with new tensor
    pub fn update(&mut self, tensor: &Tensor, ema_decay: f32) -> Result<()> {
        match tensor {
            Tensor::F32(arr) => {
                let data: Vec<f32> = arr.iter().cloned().collect();

                for &val in &data {
                    if !val.is_finite() {
                        continue; // Skip NaN/Inf values
                    }

                    self.min_val = self.min_val.min(val);
                    self.max_val = self.max_val.max(val);
                    self.sum += val as f64;
                    self.sum_squares += (val * val) as f64;
                    self.count += 1;

                    // Update EMA min/max
                    if self.ema_min.is_infinite() {
                        self.ema_min = val;
                        self.ema_max = val;
                    } else {
                        if val < self.ema_min {
                            self.ema_min = self.ema_min * (1.0 - ema_decay) + val * ema_decay;
                        }
                        if val > self.ema_max {
                            self.ema_max = self.ema_max * (1.0 - ema_decay) + val * ema_decay;
                        }
                    }
                }

                // Update histogram (simple binning for percentile calculation)
                let num_bins = 1000;
                let range = self.max_val - self.min_val;
                if range > 0.0 {
                    self.histogram.resize(num_bins, (0.0, 0));
                    for &val in &data {
                        if val.is_finite() {
                            let bin_idx =
                                ((val - self.min_val) / range * (num_bins - 1) as f32) as usize;
                            let bin_idx = bin_idx.min(num_bins - 1);
                            self.histogram[bin_idx].0 = val;
                            self.histogram[bin_idx].1 += 1;
                        }
                    }
                }
            },
            _ => {
                return Err(TrustformersError::quantization_error(
                    "Unsupported tensor type for activation quantization".into(),
                ))
            },
        }

        Ok(())
    }

    /// Get mean of observed values
    pub fn mean(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            (self.sum / self.count as f64) as f32
        }
    }

    /// Get variance of observed values
    pub fn variance(&self) -> f32 {
        if self.count <= 1 {
            0.0
        } else {
            let mean = self.mean() as f64;
            let variance = (self.sum_squares / self.count as f64) - (mean * mean);
            variance.max(0.0) as f32
        }
    }

    /// Get percentile value from histogram
    pub fn percentile(&self, p: f32) -> f32 {
        if self.histogram.is_empty() || self.count == 0 {
            return self.max_val;
        }

        let target_count = (self.count as f32 * p) as usize;
        let mut cumulative_count = 0;

        for &(val, count) in &self.histogram {
            cumulative_count += count;
            if cumulative_count >= target_count {
                return val;
            }
        }

        self.max_val
    }

    /// Get quantization parameters based on statistics
    pub fn get_quantization_params(
        &self,
        symmetric: bool,
        bits: u8,
        percentile: f32,
    ) -> Result<(f32, i32)> {
        if self.count == 0 {
            return Err(TrustformersError::quantization_error(
                "No statistics available for quantization".into(),
            ));
        }

        let q_min = if symmetric { -(1 << (bits - 1)) } else { 0 };
        let q_max = if symmetric { (1 << (bits - 1)) - 1 } else { (1 << bits) - 1 };

        let min_val = if percentile < 1.0 {
            // Use percentile-based clipping for outlier robustness
            -self.percentile(1.0 - percentile)
        } else {
            self.min_val
        };

        let max_val = if percentile < 1.0 { self.percentile(percentile) } else { self.max_val };

        let (scale, zero_point) = if symmetric {
            let abs_max = max_val.abs().max(min_val.abs());
            if abs_max == 0.0 {
                return Ok((1.0, 0));
            }
            let scale = abs_max / (q_max - q_min) as f32;
            (scale, 0)
        } else {
            if max_val == min_val {
                return Ok((1.0, q_min));
            }
            let scale = (max_val - min_val) / (q_max - q_min) as f32;
            let zero_point = q_min - (min_val / scale).round() as i32;
            let zero_point = zero_point.clamp(q_min, q_max);
            (scale, zero_point)
        };

        Ok((scale, zero_point))
    }
}

impl QuantizedActivation {
    /// Create new quantized activation
    pub fn new(
        data: Vec<u8>,
        scale: f32,
        zero_point: i32,
        shape: Vec<usize>,
        scheme: ActivationQuantScheme,
        bits: u8,
    ) -> Self {
        Self {
            data,
            scale,
            zero_point,
            shape,
            scheme,
            bits,
        }
    }

    /// Dequantize back to float tensor
    pub fn dequantize(&self) -> Result<Tensor> {
        let total_elements: usize = self.shape.iter().product();
        let mut result = Vec::with_capacity(total_elements);

        match self.scheme {
            ActivationQuantScheme::Int8 | ActivationQuantScheme::Dynamic => {
                for &quantized_val in &self.data {
                    let int_val = quantized_val as i32 - self.zero_point;
                    let float_val = int_val as f32 * self.scale;
                    result.push(float_val);
                }
            },
            ActivationQuantScheme::Int16 => {
                // For 16-bit, we need to unpack the data differently
                for chunk in self.data.chunks(2) {
                    if chunk.len() == 2 {
                        let int16_val =
                            u16::from_le_bytes([chunk[0], chunk[1]]) as i32 - self.zero_point;
                        let float_val = int16_val as f32 * self.scale;
                        result.push(float_val);
                    }
                }
            },
            ActivationQuantScheme::Adaptive => {
                // Same as Int8 for now
                for &quantized_val in &self.data {
                    let int_val = quantized_val as i32 - self.zero_point;
                    let float_val = int_val as f32 * self.scale;
                    result.push(float_val);
                }
            },
        }

        Tensor::from_vec(result, &self.shape)
    }
}

impl ActivationQuantizer {
    /// Create new activation quantizer
    pub fn new(config: ActivationQuantConfig) -> Self {
        Self {
            config,
            layer_stats: HashMap::new(),
            calibrating: true,
            calibration_count: 0,
        }
    }

    /// Start calibration phase
    pub fn start_calibration(&mut self) {
        self.calibrating = true;
        self.calibration_count = 0;
        self.layer_stats.clear();
    }

    /// End calibration phase
    pub fn end_calibration(&mut self) {
        self.calibrating = false;
    }

    /// Check if calibration is complete
    pub fn is_calibration_complete(&self) -> bool {
        !self.calibrating || self.calibration_count >= self.config.calibration_samples
    }

    /// Quantize activation tensor for a specific layer
    pub fn quantize_activation(
        &mut self,
        tensor: &Tensor,
        layer_name: &str,
        training: bool,
    ) -> Result<Tensor> {
        // Get layer-specific configuration
        let layer_config = self.config.layer_configs.get(layer_name).cloned().unwrap_or_default();

        if !layer_config.enabled {
            return Ok(tensor.clone());
        }

        // Don't quantize during training unless explicitly enabled
        if training && !self.config.quantize_during_training {
            if self.calibrating && layer_config.calibrate {
                self.update_statistics(tensor, layer_name)?;
            }
            return Ok(tensor.clone());
        }

        // Update statistics during calibration
        if self.calibrating && layer_config.calibrate {
            self.update_statistics(tensor, layer_name)?;

            // Return original tensor during calibration
            if self.calibration_count < self.config.calibration_samples {
                return Ok(tensor.clone());
            }
        }

        // Apply quantization
        self.apply_quantization(tensor, layer_name, &layer_config)
    }

    /// Update statistics for a layer
    fn update_statistics(&mut self, tensor: &Tensor, layer_name: &str) -> Result<()> {
        let stats = self.layer_stats.entry(layer_name.to_string()).or_default();

        stats.update(tensor, self.config.ema_decay)?;
        self.calibration_count += 1;

        Ok(())
    }

    /// Apply quantization to tensor
    fn apply_quantization(
        &self,
        tensor: &Tensor,
        layer_name: &str,
        layer_config: &LayerQuantConfig,
    ) -> Result<Tensor> {
        let stats = self.layer_stats.get(layer_name).ok_or_else(|| {
            TrustformersError::quantization_error(format!(
                "No calibration statistics found for layer {}",
                layer_name
            ))
        })?;

        let scheme = layer_config.scheme.unwrap_or(self.config.scheme);
        let bits = layer_config.bits.unwrap_or(match scheme {
            ActivationQuantScheme::Int8
            | ActivationQuantScheme::Dynamic
            | ActivationQuantScheme::Adaptive => 8,
            ActivationQuantScheme::Int16 => 16,
        });

        let (scale, zero_point) =
            stats.get_quantization_params(self.config.symmetric, bits, self.config.percentile)?;

        match scheme {
            ActivationQuantScheme::Int8 | ActivationQuantScheme::Dynamic => {
                self.quantize_int8(tensor, scale, zero_point)
            },
            ActivationQuantScheme::Int16 => self.quantize_int16(tensor, scale, zero_point),
            ActivationQuantScheme::Adaptive => {
                self.quantize_adaptive(tensor, stats, scale, zero_point)
            },
        }
    }

    /// Quantize tensor to 8-bit integers
    fn quantize_int8(&self, tensor: &Tensor, scale: f32, zero_point: i32) -> Result<Tensor> {
        match tensor {
            Tensor::F32(arr) => {
                let quantized_data: Vec<f32> = arr
                    .iter()
                    .map(|&val| {
                        let q_val = ((val / scale).round() as i32 + zero_point).clamp(0, 255) as u8;

                        (q_val as i32 - zero_point) as f32 * scale
                    })
                    .collect();

                Tensor::from_vec(quantized_data, arr.shape())
            },
            _ => Err(TrustformersError::quantization_error(
                "Unsupported tensor type for activation quantization".into(),
            )),
        }
    }

    /// Quantize tensor to 16-bit integers
    fn quantize_int16(&self, tensor: &Tensor, scale: f32, zero_point: i32) -> Result<Tensor> {
        match tensor {
            Tensor::F32(arr) => {
                let quantized_data: Vec<f32> = arr
                    .iter()
                    .map(|&val| {
                        let q_val =
                            ((val / scale).round() as i32 + zero_point).clamp(0, 65535) as u16;

                        (q_val as i32 - zero_point) as f32 * scale
                    })
                    .collect();

                Tensor::from_vec(quantized_data, arr.shape())
            },
            _ => Err(TrustformersError::quantization_error(
                "Unsupported tensor type for activation quantization".into(),
            )),
        }
    }

    /// Adaptive quantization based on activation distribution
    fn quantize_adaptive(
        &self,
        tensor: &Tensor,
        stats: &ActivationStats,
        scale: f32,
        zero_point: i32,
    ) -> Result<Tensor> {
        match tensor {
            Tensor::F32(arr) => {
                let variance = stats.variance();
                let mean = stats.mean();

                // Use different quantization strategies based on distribution characteristics
                let quantized_data: Vec<f32> = arr
                    .iter()
                    .map(|&val| {
                        // For low variance activations, use more aggressive quantization
                        let effective_scale = if variance < 0.1 {
                            scale * 0.5 // Finer quantization for low variance
                        } else {
                            scale
                        };

                        // Apply outlier clipping for values far from mean
                        let clipped_val = if (val - mean).abs() > 3.0 * variance.sqrt() {
                            if val > mean {
                                mean + 3.0 * variance.sqrt()
                            } else {
                                mean - 3.0 * variance.sqrt()
                            }
                        } else {
                            val
                        };

                        let q_val = ((clipped_val / effective_scale).round() as i32 + zero_point)
                            .clamp(0, 255) as u8;

                        (q_val as i32 - zero_point) as f32 * effective_scale
                    })
                    .collect();

                Tensor::from_vec(quantized_data, arr.shape())
            },
            _ => Err(TrustformersError::quantization_error(
                "Unsupported tensor type for adaptive quantization".into(),
            )),
        }
    }

    /// Get statistics for a specific layer
    pub fn get_layer_stats(&self, layer_name: &str) -> Option<&ActivationStats> {
        self.layer_stats.get(layer_name)
    }

    /// Get all layer statistics
    pub fn get_all_stats(&self) -> &HashMap<String, ActivationStats> {
        &self.layer_stats
    }

    /// Save calibration statistics to file
    pub fn save_calibration(&self, path: &str) -> Result<()> {
        let json_data = serde_json::to_string_pretty(&self.layer_stats).map_err(|e| {
            TrustformersError::quantization_error(format!("Failed to serialize statistics: {}", e))
        })?;

        std::fs::write(path, json_data).map_err(|e| {
            TrustformersError::quantization_error(format!("Failed to write file: {}", e))
        })?;

        Ok(())
    }

    /// Load calibration statistics from file
    pub fn load_calibration(&mut self, path: &str) -> Result<()> {
        let json_data = std::fs::read_to_string(path).map_err(|e| {
            TrustformersError::quantization_error(format!("Failed to read file: {}", e))
        })?;

        self.layer_stats = serde_json::from_str(&json_data).map_err(|e| {
            TrustformersError::quantization_error(format!(
                "Failed to deserialize statistics: {}",
                e
            ))
        })?;

        self.calibrating = false;
        Ok(())
    }

    /// Configure quantization for a specific layer
    pub fn configure_layer(&mut self, layer_name: &str, config: LayerQuantConfig) {
        self.config.layer_configs.insert(layer_name.to_string(), config);
    }

    /// Disable quantization for a specific layer
    pub fn disable_layer(&mut self, layer_name: &str) {
        let config = LayerQuantConfig {
            enabled: false,
            ..Default::default()
        };
        self.config.layer_configs.insert(layer_name.to_string(), config);
    }

    /// Get memory savings from activation quantization
    pub fn get_memory_savings(&self) -> f32 {
        // Estimate memory savings based on bit width
        match self.config.scheme {
            ActivationQuantScheme::Int8
            | ActivationQuantScheme::Dynamic
            | ActivationQuantScheme::Adaptive => 0.75, // 32-bit to 8-bit = 75% savings
            ActivationQuantScheme::Int16 => 0.5, // 32-bit to 16-bit = 50% savings
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_stats_update() {
        let mut stats = ActivationStats::new();
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();

        stats.update(&tensor, 0.01).unwrap();

        assert_eq!(stats.count, 5);
        assert_eq!(stats.min_val, 1.0);
        assert_eq!(stats.max_val, 5.0);
        assert_eq!(stats.mean(), 3.0);
    }

    #[test]
    fn test_activation_stats_quantization_params() {
        let mut stats = ActivationStats::new();
        let tensor = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();

        stats.update(&tensor, 0.01).unwrap();

        let (scale, zero_point) = stats.get_quantization_params(true, 8, 1.0).unwrap();
        assert!(scale > 0.0);
        assert_eq!(zero_point, 0); // Symmetric quantization
    }

    #[test]
    fn test_activation_quantizer_calibration() {
        let config = ActivationQuantConfig {
            calibration_samples: 2,
            ..Default::default()
        };
        let mut quantizer = ActivationQuantizer::new(config);

        let tensor1 = Tensor::randn(&[10, 20]).unwrap();
        let tensor2 = Tensor::randn(&[10, 20]).unwrap();

        // Calibration phase
        assert!(quantizer.calibrating);
        quantizer.quantize_activation(&tensor1, "layer1", false).unwrap();
        quantizer.quantize_activation(&tensor2, "layer1", false).unwrap();

        // Should have statistics now
        assert!(quantizer.get_layer_stats("layer1").is_some());
    }

    #[test]
    fn test_activation_quantizer_int8() {
        let mut config = ActivationQuantConfig::default();
        config.calibration_samples = 1;
        config.scheme = ActivationQuantScheme::Int8;

        let mut quantizer = ActivationQuantizer::new(config);

        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();

        // Calibrate
        quantizer.quantize_activation(&tensor, "test_layer", false).unwrap();
        quantizer.end_calibration();

        // Quantize
        let result = quantizer.quantize_activation(&tensor, "test_layer", false).unwrap();
        assert_eq!(result.shape(), tensor.shape());
    }

    #[test]
    fn test_activation_quantizer_layer_config() {
        let config = ActivationQuantConfig::default();
        let mut quantizer = ActivationQuantizer::new(config);

        // Configure a specific layer
        let layer_config = LayerQuantConfig {
            enabled: true,
            scheme: Some(ActivationQuantScheme::Int16),
            bits: Some(16),
            calibrate: true,
        };
        quantizer.configure_layer("special_layer", layer_config);

        // Disable another layer
        quantizer.disable_layer("disabled_layer");

        let tensor = Tensor::randn(&[8, 8]).unwrap();

        // Disabled layer should return original tensor
        let result = quantizer.quantize_activation(&tensor, "disabled_layer", false).unwrap();
        // Should be same reference (no quantization applied)
        assert_eq!(result.shape(), tensor.shape());
    }

    #[test]
    fn test_activation_quantizer_adaptive() {
        let mut config = ActivationQuantConfig::default();
        config.scheme = ActivationQuantScheme::Adaptive;
        config.calibration_samples = 1;

        let mut quantizer = ActivationQuantizer::new(config);

        let tensor = Tensor::from_vec(vec![0.1, 0.2, 0.15, 0.18, 10.0], &[5]).unwrap(); // One outlier

        // Calibrate
        quantizer.quantize_activation(&tensor, "adaptive_layer", false).unwrap();
        quantizer.end_calibration();

        // Quantize with adaptive scheme
        let result = quantizer.quantize_activation(&tensor, "adaptive_layer", false).unwrap();
        assert_eq!(result.shape(), tensor.shape());
    }

    #[test]
    fn test_quantized_activation_dequantization() {
        let _original_data = [1.0, 2.0, 3.0, 4.0];
        let shape = vec![4];

        // Simulate quantized data
        let quantized_data = vec![64, 128, 192, 255]; // 8-bit quantized values
        let scale = 4.0 / 255.0; // Scale for range [0, 4]
        let zero_point = 0;

        let quant_activation = QuantizedActivation::new(
            quantized_data,
            scale,
            zero_point,
            shape.clone(),
            ActivationQuantScheme::Int8,
            8,
        );

        let dequantized = quant_activation.dequantize().unwrap();
        assert_eq!(dequantized.shape(), shape);
    }

    #[test]
    fn test_memory_savings_calculation() {
        let config = ActivationQuantConfig {
            scheme: ActivationQuantScheme::Int8,
            ..Default::default()
        };
        let quantizer = ActivationQuantizer::new(config);

        let savings = quantizer.get_memory_savings();
        assert_eq!(savings, 0.75); // 75% savings for int8
    }

    #[test]
    fn test_percentile_calculation() {
        let mut stats = ActivationStats::new();
        let tensor = Tensor::from_vec((1..=100).map(|x| x as f32).collect(), &[100]).unwrap();

        stats.update(&tensor, 0.01).unwrap();

        let p95 = stats.percentile(0.95);
        assert!((90.0..=100.0).contains(&p95)); // Should be around 95
    }

    #[test]
    fn test_serialization() {
        let config = ActivationQuantConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: ActivationQuantConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config.scheme, deserialized.scheme);
        assert_eq!(config.symmetric, deserialized.symmetric);
    }
}
