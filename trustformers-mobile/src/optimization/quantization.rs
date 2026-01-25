//! Mobile Quantization Module
//!
//! Provides efficient quantization implementations for mobile deployment including:
//! - INT4 quantization for ultra-low memory
//! - INT8 quantization for balanced performance
//! - FP16 quantization for GPU acceleration
//! - Dynamic quantization for runtime adaptation

use half::f16;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use trustformers_core::errors::{invalid_config, runtime_error, tensor_op_error, Result};
use trustformers_core::Tensor;

/// Quantization scheme types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum QuantizationScheme {
    Int4,
    Int8,
    FP16,
    Dynamic,
    /// GGUF Q2_K: 2.5625 bits per weight, ultra-low memory
    GGUF_Q2_K,
    /// GGUF Q3_K: 3.4375 bits per weight, balanced
    GGUF_Q3_K,
    /// GGUF Q4_K: 4.5 bits per weight, high quality
    GGUF_Q4_K,
    /// GGUF Q5_0: 5.5 bits per weight, very high quality
    GGUF_Q5_0,
    /// GGUF Q6_K: 6.5 bits per weight, near-lossless
    GGUF_Q6_K,
}

impl std::fmt::Display for QuantizationScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantizationScheme::Int4 => write!(f, "INT4"),
            QuantizationScheme::Int8 => write!(f, "INT8"),
            QuantizationScheme::FP16 => write!(f, "FP16"),
            QuantizationScheme::Dynamic => write!(f, "Dynamic"),
            QuantizationScheme::GGUF_Q2_K => write!(f, "GGUF_Q2_K"),
            QuantizationScheme::GGUF_Q3_K => write!(f, "GGUF_Q3_K"),
            QuantizationScheme::GGUF_Q4_K => write!(f, "GGUF_Q4_K"),
            QuantizationScheme::GGUF_Q5_0 => write!(f, "GGUF_Q5_0"),
            QuantizationScheme::GGUF_Q6_K => write!(f, "GGUF_Q6_K"),
        }
    }
}

/// Calibration method for quantization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationMethod {
    MinMax,
    Percentile,
    MovingAverage,
    KLDivergence,
}

/// Quantization context with calibration data
#[derive(Debug, Clone)]
pub struct QuantizationContext {
    pub method: CalibrationMethod,
    pub num_calibration_samples: usize,
    pub percentile: f32,    // For percentile method
    pub smooth_factor: f32, // For moving average
}

impl Default for QuantizationContext {
    fn default() -> Self {
        Self {
            method: CalibrationMethod::MinMax,
            num_calibration_samples: 100,
            percentile: 99.9,
            smooth_factor: 0.999,
        }
    }
}

/// Quantization calibration data
#[derive(Debug, Clone, Default)]
pub struct QuantizationCalibration {
    pub min_values: HashMap<String, f32>,
    pub max_values: HashMap<String, f32>,
    pub scales: HashMap<String, f32>,
    pub zero_points: HashMap<String, i32>,
    pub histogram_bins: HashMap<String, Vec<f32>>,
}

/// External quantization scheme configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationSchemeConfig {
    /// Default quantization scheme
    pub default_scheme: QuantizationScheme,
    /// Layer-specific quantization schemes
    pub layer_schemes: HashMap<String, QuantizationScheme>,
    /// Tensor-specific quantization schemes (by hash or name)
    pub tensor_schemes: HashMap<String, QuantizationScheme>,
    /// Model-specific schemes
    pub model_schemes: HashMap<String, QuantizationScheme>,
    /// Performance-based scheme mappings
    pub performance_schemes: HashMap<String, QuantizationScheme>,
}

impl Default for QuantizationSchemeConfig {
    fn default() -> Self {
        Self {
            default_scheme: QuantizationScheme::Int8,
            layer_schemes: HashMap::new(),
            tensor_schemes: HashMap::new(),
            model_schemes: HashMap::new(),
            performance_schemes: HashMap::new(),
        }
    }
}

/// Quantization scheme storage manager
#[derive(Debug, Clone)]
pub struct QuantizationSchemeStorage {
    /// Configuration file path
    pub config_path: Option<PathBuf>,
    /// In-memory configuration
    pub config: QuantizationSchemeConfig,
    /// Cache for recently determined schemes
    pub scheme_cache: HashMap<String, QuantizationScheme>,
}

impl Default for QuantizationSchemeStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizationSchemeStorage {
    /// Create a new storage manager
    pub fn new() -> Self {
        Self {
            config_path: None,
            config: QuantizationSchemeConfig::default(),
            scheme_cache: HashMap::new(),
        }
    }

    /// Create storage manager with config file
    pub fn with_config_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config_path = path.as_ref().to_path_buf();
        let config = Self::load_config(&config_path)?;

        Ok(Self {
            config_path: Some(config_path),
            config,
            scheme_cache: HashMap::new(),
        })
    }

    /// Load configuration from file
    pub fn load_config<P: AsRef<Path>>(path: P) -> Result<QuantizationSchemeConfig> {
        let file = File::open(path.as_ref())
            .map_err(|e| runtime_error(format!("Failed to open config file: {}", e)))?;
        let reader = BufReader::new(file);

        serde_json::from_reader(reader)
            .map_err(|e| invalid_config("load_config", format!("Failed to parse config: {}", e)))
    }

    /// Save configuration to file
    pub fn save_config(&self) -> Result<()> {
        if let Some(ref path) = self.config_path {
            let file = File::create(path)
                .map_err(|e| runtime_error(format!("Failed to create config file: {}", e)))?;

            serde_json::to_writer_pretty(file, &self.config)
                .map_err(|e| runtime_error(format!("Failed to write config: {}", e)))?;
        }
        Ok(())
    }

    /// Determine quantization scheme for a tensor
    pub fn determine_scheme(
        &mut self,
        tensor_id: &str,
        layer_name: Option<&str>,
        model_name: Option<&str>,
    ) -> QuantizationScheme {
        // Check cache first
        if let Some(&scheme) = self.scheme_cache.get(tensor_id) {
            return scheme;
        }

        // Check tensor-specific scheme
        if let Some(&scheme) = self.config.tensor_schemes.get(tensor_id) {
            self.scheme_cache.insert(tensor_id.to_string(), scheme);
            return scheme;
        }

        // Check layer-specific scheme
        if let Some(layer) = layer_name {
            if let Some(&scheme) = self.config.layer_schemes.get(layer) {
                self.scheme_cache.insert(tensor_id.to_string(), scheme);
                return scheme;
            }
        }

        // Check model-specific scheme
        if let Some(model) = model_name {
            if let Some(&scheme) = self.config.model_schemes.get(model) {
                self.scheme_cache.insert(tensor_id.to_string(), scheme);
                return scheme;
            }
        }

        // Use default scheme
        let default_scheme = self.config.default_scheme;
        self.scheme_cache.insert(tensor_id.to_string(), default_scheme);
        default_scheme
    }

    /// Set scheme for specific tensor
    pub fn set_tensor_scheme(&mut self, tensor_id: String, scheme: QuantizationScheme) {
        self.config.tensor_schemes.insert(tensor_id.clone(), scheme);
        self.scheme_cache.insert(tensor_id, scheme);
    }

    /// Set scheme for specific layer
    pub fn set_layer_scheme(&mut self, layer_name: String, scheme: QuantizationScheme) {
        self.config.layer_schemes.insert(layer_name, scheme);
    }

    /// Set scheme for specific model
    pub fn set_model_scheme(&mut self, model_name: String, scheme: QuantizationScheme) {
        self.config.model_schemes.insert(model_name, scheme);
    }

    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.scheme_cache.clear();
    }

    /// Generate tensor ID from tensor properties
    pub fn generate_tensor_id(tensor: &Tensor, layer_name: Option<&str>) -> String {
        let shape_str = tensor.shape().iter().map(|&s| s.to_string()).collect::<Vec<_>>().join("x");

        let data_hash = {
            if let Ok(data) = tensor.data() {
                let sample_size = (data.len() / 100).max(1).min(1000); // Sample for hash
                let mut hash = 0u64;
                for i in (0..data.len()).step_by(sample_size) {
                    hash = hash.wrapping_mul(31).wrapping_add(data[i].to_bits() as u64);
                }
                hash
            } else {
                0u64 // Default hash if data access fails
            }
        };

        match layer_name {
            Some(layer) => format!("{}:{}:{:x}", layer, shape_str, data_hash),
            None => format!("tensor:{}:{:x}", shape_str, data_hash),
        }
    }
}

/// Trait for mobile quantizers
pub trait MobileQuantizer: Send + Sync {
    /// Get the quantization scheme
    fn get_scheme(&self) -> QuantizationScheme;

    /// Check if calibration is required
    fn requires_calibration(&self) -> bool;

    /// Calibrate the quantizer with sample data
    fn calibrate(&self, data: &[Tensor]) -> Result<()>;

    /// Quantize a tensor
    fn quantize_tensor(&self, tensor: &Tensor) -> Result<Tensor>;

    /// Dequantize a tensor
    fn dequantize_tensor(&self, tensor: &Tensor) -> Result<Tensor>;
}

/// INT4 Quantizer - Ultra-low memory for mobile
pub struct Int4Quantizer {
    context: QuantizationContext,
    calibration: std::sync::RwLock<QuantizationCalibration>,
}

impl Default for Int4Quantizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Int4Quantizer {
    pub fn new() -> Self {
        Self {
            context: QuantizationContext::default(),
            calibration: std::sync::RwLock::new(QuantizationCalibration::default()),
        }
    }

    fn compute_scale_zero_point(&self, min_val: f32, max_val: f32) -> (f32, i32) {
        let qmin = -8.0; // 4-bit signed: -8 to 7
        let qmax = 7.0;

        let scale = (max_val - min_val) / (qmax - qmin);
        let zero_point = ((qmin - min_val / scale).round() as i32).clamp(-8, 7);

        (scale, zero_point)
    }

    fn quantize_value(&self, value: f32, scale: f32, zero_point: i32) -> i8 {
        let quantized = (value / scale).round() as i32 + zero_point;
        quantized.clamp(-8, 7) as i8
    }

    fn dequantize_value(&self, quantized: i8, scale: f32, zero_point: i32) -> f32 {
        (quantized as i32 - zero_point) as f32 * scale
    }
}

impl MobileQuantizer for Int4Quantizer {
    fn get_scheme(&self) -> QuantizationScheme {
        QuantizationScheme::Int4
    }

    fn requires_calibration(&self) -> bool {
        true
    }

    fn calibrate(&self, data: &[Tensor]) -> Result<()> {
        let mut calibration = self.calibration.write().expect("RwLock poisoned");

        for tensor in data {
            let tensor_data = tensor.data()?;
            let min_val = tensor_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = tensor_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            let (scale, zero_point) = self.compute_scale_zero_point(min_val, max_val);

            // Store as global calibration parameters (simplified for single tensor case)
            calibration.min_values.insert("global".to_string(), min_val);
            calibration.max_values.insert("global".to_string(), max_val);
            calibration.scales.insert("global".to_string(), scale);
            calibration.zero_points.insert("global".to_string(), zero_point);
        }

        Ok(())
    }

    fn quantize_tensor(&self, tensor: &Tensor) -> Result<Tensor> {
        let calibration = self.calibration.read().expect("RwLock poisoned");
        let tensor_data = tensor.data()?;

        // Get or compute scale and zero point
        let (scale, zero_point) = if let Some(&scale) = calibration.scales.get("global") {
            (
                scale,
                *calibration.zero_points.get("global").expect("No global zero point"),
            )
        } else {
            // Compute on the fly if not calibrated
            let min_val = tensor_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = tensor_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            self.compute_scale_zero_point(min_val, max_val)
        };

        // Quantize to INT4 (stored as individual i8 values for compatibility)
        let quantized_data: Vec<i8> =
            tensor_data.iter().map(|&x| self.quantize_value(x, scale, zero_point)).collect();

        // Convert to f32 for tensor storage (maintain same shape)
        let quantized_f32: Vec<f32> = quantized_data.iter().map(|&x| x as f32).collect();

        // Create quantized tensor with same shape as original
        let quantized_tensor = Tensor::from_vec(quantized_f32, &tensor.shape())?;

        // Note: Quantization parameters stored separately (tensor doesn't support metadata)

        Ok(quantized_tensor)
    }

    fn dequantize_tensor(&self, tensor: &Tensor) -> Result<Tensor> {
        let calibration = self.calibration.read().expect("RwLock poisoned");
        let tensor_data = tensor.data()?;

        // Get quantization parameters from calibration data
        let (scale, zero_point) = if let Some(&scale) = calibration.scales.get("global") {
            (
                scale,
                *calibration.zero_points.get("global").expect("No global zero point"),
            )
        } else {
            // Fallback: estimate from quantized data range
            let min_q = tensor_data.iter().fold(f32::INFINITY, |a, &b| a.min(b)) as i8;
            let max_q = tensor_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as i8;
            let range = (max_q - min_q) as f32;
            let scale = if range > 0.0 { 15.0 / range } else { 1.0 }; // 4-bit range is -8 to 7 = 15
            (scale, 0)
        };

        // Dequantize i8 values back to f32
        let dequantized_data: Vec<f32> = tensor_data
            .iter()
            .map(|&x| self.dequantize_value(x as i8, scale, zero_point))
            .collect();

        Tensor::from_vec(dequantized_data, &tensor.shape())
    }
}

/// INT8 Quantizer - Balanced performance and accuracy
pub struct Int8Quantizer {
    context: QuantizationContext,
    calibration: std::sync::RwLock<QuantizationCalibration>,
    symmetric: bool,
}

impl Default for Int8Quantizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Int8Quantizer {
    pub fn new() -> Self {
        Self {
            context: QuantizationContext::default(),
            calibration: std::sync::RwLock::new(QuantizationCalibration::default()),
            symmetric: true, // Symmetric quantization is often better for mobile
        }
    }

    fn compute_scale_zero_point(&self, min_val: f32, max_val: f32) -> (f32, i32) {
        if self.symmetric {
            // Symmetric quantization (zero point = 0)
            let abs_max = min_val.abs().max(max_val.abs());
            let scale = abs_max / 127.0;
            (scale, 0)
        } else {
            // Asymmetric quantization
            let qmin = -128.0;
            let qmax = 127.0;
            let scale = (max_val - min_val) / (qmax - qmin);
            let zero_point = ((qmin - min_val / scale).round() as i32).clamp(-128, 127);
            (scale, zero_point)
        }
    }
}

impl MobileQuantizer for Int8Quantizer {
    fn get_scheme(&self) -> QuantizationScheme {
        QuantizationScheme::Int8
    }

    fn requires_calibration(&self) -> bool {
        true
    }

    fn calibrate(&self, data: &[Tensor]) -> Result<()> {
        let mut calibration = self.calibration.write().expect("RwLock poisoned");

        for tensor in data {
            let tensor_data = tensor.data()?;

            let (min_val, max_val) = match self.context.method {
                CalibrationMethod::MinMax => {
                    let min = tensor_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                    let max = tensor_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    (min, max)
                },
                CalibrationMethod::Percentile => {
                    let mut sorted = tensor_data.to_vec();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let percentile_idx =
                        (sorted.len() as f32 * self.context.percentile / 100.0) as usize;
                    let min_idx =
                        (sorted.len() as f32 * (100.0 - self.context.percentile) / 100.0) as usize;
                    (
                        sorted[min_idx],
                        sorted[percentile_idx.min(sorted.len() - 1)],
                    )
                },
                _ => {
                    // For other methods, fall back to min-max
                    let min = tensor_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                    let max = tensor_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    (min, max)
                },
            };

            let (scale, zero_point) = self.compute_scale_zero_point(min_val, max_val);

            // Store as global calibration parameters (simplified for single tensor case)
            calibration.min_values.insert("global".to_string(), min_val);
            calibration.max_values.insert("global".to_string(), max_val);
            calibration.scales.insert("global".to_string(), scale);
            calibration.zero_points.insert("global".to_string(), zero_point);
        }

        Ok(())
    }

    fn quantize_tensor(&self, tensor: &Tensor) -> Result<Tensor> {
        let calibration = self.calibration.read().expect("RwLock poisoned");
        let tensor_data = tensor.data()?;

        let (scale, zero_point) = if let Some(&scale) = calibration.scales.get("global") {
            (
                scale,
                *calibration.zero_points.get("global").expect("No global zero point"),
            )
        } else {
            let min_val = tensor_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = tensor_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            self.compute_scale_zero_point(min_val, max_val)
        };

        let quantized_data: Vec<i8> = tensor_data
            .iter()
            .map(|&x| {
                let q = (x / scale).round() as i32 + zero_point;
                q.clamp(-128, 127) as i8
            })
            .collect();

        // Convert to f32 for tensor storage (temporary)
        let quantized_f32: Vec<f32> = quantized_data.iter().map(|&x| x as f32).collect();

        let quantized_tensor = Tensor::from_vec(quantized_f32, &tensor.shape())?;
        // Note: Quantization parameters stored separately (tensor doesn't support metadata)

        Ok(quantized_tensor)
    }

    fn dequantize_tensor(&self, tensor: &Tensor) -> Result<Tensor> {
        let calibration = self.calibration.read().expect("RwLock poisoned");
        let tensor_data = tensor.data()?;

        // Get quantization parameters from calibration data
        let (scale, zero_point) = if let Some(&scale) = calibration.scales.get("global") {
            (
                scale,
                *calibration.zero_points.get("global").expect("No global zero point"),
            )
        } else {
            // Fallback: estimate from quantized data range
            let min_q = tensor_data.iter().fold(f32::INFINITY, |a, &b| a.min(b)) as i32;
            let max_q = tensor_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as i32;
            let range = (max_q - min_q) as f32;
            let scale = if range > 0.0 { 255.0 / range } else { 1.0 }; // 8-bit range is 255
            (scale, 0)
        };

        let dequantized_data: Vec<f32> =
            tensor_data.iter().map(|&x| ((x as i32) - zero_point) as f32 * scale).collect();

        Tensor::from_vec(dequantized_data, &tensor.shape())
    }
}

/// FP16 Quantizer - Hardware-accelerated on modern mobile GPUs
pub struct FP16Quantizer;

impl Default for FP16Quantizer {
    fn default() -> Self {
        Self::new()
    }
}

impl FP16Quantizer {
    pub fn new() -> Self {
        Self
    }
}

impl MobileQuantizer for FP16Quantizer {
    fn get_scheme(&self) -> QuantizationScheme {
        QuantizationScheme::FP16
    }

    fn requires_calibration(&self) -> bool {
        false // FP16 doesn't require calibration
    }

    fn calibrate(&self, _data: &[Tensor]) -> Result<()> {
        Ok(()) // No calibration needed
    }

    fn quantize_tensor(&self, tensor: &Tensor) -> Result<Tensor> {
        let tensor_data = tensor.data()?;

        // Convert to FP16
        let fp16_data: Vec<f16> = tensor_data.iter().map(|&x| f16::from_f32(x)).collect();

        // Convert back to f32 for storage (temporary - in real implementation would store as f16)
        let quantized_data: Vec<f32> = fp16_data.iter().map(|&x| f32::from(x)).collect();

        let quantized_tensor = Tensor::from_vec(quantized_data, &tensor.shape())?;
        // Note: Quantization parameters stored separately (tensor doesn't support metadata)

        Ok(quantized_tensor)
    }

    fn dequantize_tensor(&self, tensor: &Tensor) -> Result<Tensor> {
        // FP16 quantization is lossless within range, so just return clone
        Ok(tensor.clone())
    }
}

/// Dynamic Quantizer - Adapts quantization based on runtime statistics
pub struct DynamicQuantizer {
    int8_quantizer: Int8Quantizer,
    fp16_quantizer: FP16Quantizer,
    selection_threshold: f32,
    scheme_storage: QuantizationSchemeStorage,
    layer_context: Option<String>,
    model_context: Option<String>,
}

impl Default for DynamicQuantizer {
    fn default() -> Self {
        Self::new()
    }
}

impl DynamicQuantizer {
    pub fn new() -> Self {
        Self {
            int8_quantizer: Int8Quantizer::new(),
            fp16_quantizer: FP16Quantizer::new(),
            selection_threshold: 0.1, // 10% error threshold
            scheme_storage: QuantizationSchemeStorage::new(),
            layer_context: None,
            model_context: None,
        }
    }

    pub fn with_config_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let scheme_storage = QuantizationSchemeStorage::with_config_file(path)?;
        Ok(Self {
            int8_quantizer: Int8Quantizer::new(),
            fp16_quantizer: FP16Quantizer::new(),
            selection_threshold: 0.1,
            scheme_storage,
            layer_context: None,
            model_context: None,
        })
    }

    pub fn set_layer_context(&mut self, layer_name: String) {
        self.layer_context = Some(layer_name);
    }

    pub fn set_model_context(&mut self, model_name: String) {
        self.model_context = Some(model_name);
    }

    /// Get mutable reference to scheme storage for configuration
    pub fn scheme_storage_mut(&mut self) -> &mut QuantizationSchemeStorage {
        &mut self.scheme_storage
    }

    /// Get reference to scheme storage
    pub fn scheme_storage(&self) -> &QuantizationSchemeStorage {
        &self.scheme_storage
    }

    fn select_quantization_scheme(&self, tensor: &Tensor) -> Result<QuantizationScheme> {
        let tensor_data = tensor.data()?;

        // Compute dynamic range
        let min_val = tensor_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = tensor_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;

        // Compute variance
        let mean = tensor_data.iter().sum::<f32>() / tensor_data.len() as f32;
        let variance =
            tensor_data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / tensor_data.len() as f32;

        // Decision logic
        if range < 1.0 && variance < 0.01 {
            // Small range and low variance - INT8 is sufficient
            Ok(QuantizationScheme::Int8)
        } else {
            // Large range or high variance - use FP16
            Ok(QuantizationScheme::FP16)
        }
    }
}

impl MobileQuantizer for DynamicQuantizer {
    fn get_scheme(&self) -> QuantizationScheme {
        QuantizationScheme::Dynamic
    }

    fn requires_calibration(&self) -> bool {
        true // INT8 path requires calibration
    }

    fn calibrate(&self, data: &[Tensor]) -> Result<()> {
        // Calibrate INT8 quantizer
        self.int8_quantizer.calibrate(data)
    }

    fn quantize_tensor(&self, tensor: &Tensor) -> Result<Tensor> {
        // Generate tensor ID for scheme determination
        let tensor_id =
            QuantizationSchemeStorage::generate_tensor_id(tensor, self.layer_context.as_deref());

        // First check external storage for quantization scheme
        let mut storage = self.scheme_storage.clone();
        let scheme = storage.determine_scheme(
            &tensor_id,
            self.layer_context.as_deref(),
            self.model_context.as_deref(),
        );

        // If external storage returns Dynamic, fall back to selection logic
        let final_scheme = if scheme == QuantizationScheme::Dynamic {
            self.select_quantization_scheme(tensor)?
        } else {
            scheme
        };

        match final_scheme {
            QuantizationScheme::Int4 => {
                // For int4, we need to create a quantizer instance
                let int4_quantizer = Int4Quantizer::new();
                int4_quantizer.quantize_tensor(tensor)
            },
            QuantizationScheme::Int8 => self.int8_quantizer.quantize_tensor(tensor),
            QuantizationScheme::FP16 => self.fp16_quantizer.quantize_tensor(tensor),
            // GGUF schemes fall back to INT8 (will be handled by dedicated GGUF quantizer)
            QuantizationScheme::GGUF_Q2_K
            | QuantizationScheme::GGUF_Q3_K
            | QuantizationScheme::GGUF_Q4_K
            | QuantizationScheme::GGUF_Q5_0
            | QuantizationScheme::GGUF_Q6_K => {
                // Note: GGUF quantization should use MobileGGUFQuantizer directly
                // For now, fall back to INT8
                self.int8_quantizer.quantize_tensor(tensor)
            },
            QuantizationScheme::Dynamic => {
                // This shouldn't happen after our check above, but handle gracefully
                let selected_scheme = self.select_quantization_scheme(tensor)?;
                match selected_scheme {
                    QuantizationScheme::Int4 => {
                        let int4_quantizer = Int4Quantizer::new();
                        int4_quantizer.quantize_tensor(tensor)
                    },
                    QuantizationScheme::Int8 => self.int8_quantizer.quantize_tensor(tensor),
                    QuantizationScheme::FP16 => self.fp16_quantizer.quantize_tensor(tensor),
                    // GGUF schemes fall back to INT8
                    QuantizationScheme::GGUF_Q2_K
                    | QuantizationScheme::GGUF_Q3_K
                    | QuantizationScheme::GGUF_Q4_K
                    | QuantizationScheme::GGUF_Q5_0
                    | QuantizationScheme::GGUF_Q6_K => self.int8_quantizer.quantize_tensor(tensor),
                    QuantizationScheme::Dynamic => {
                        // If still Dynamic, default to Int8 as fallback
                        self.int8_quantizer.quantize_tensor(tensor)
                    },
                }
            },
        }
    }

    fn dequantize_tensor(&self, tensor: &Tensor) -> Result<Tensor> {
        // Generate tensor ID for scheme determination
        let tensor_id =
            QuantizationSchemeStorage::generate_tensor_id(tensor, self.layer_context.as_deref());

        // Determine scheme from external storage
        let mut storage = self.scheme_storage.clone();
        let scheme = storage.determine_scheme(
            &tensor_id,
            self.layer_context.as_deref(),
            self.model_context.as_deref(),
        );

        match scheme {
            QuantizationScheme::Int8 => self.int8_quantizer.dequantize_tensor(tensor),
            QuantizationScheme::FP16 => self.fp16_quantizer.dequantize_tensor(tensor),
            QuantizationScheme::Int4 => {
                // For int4, we need to create a quantizer instance
                let int4_quantizer = Int4Quantizer::new();
                int4_quantizer.dequantize_tensor(tensor)
            },
            // GGUF schemes fall back to INT8 dequantization
            QuantizationScheme::GGUF_Q2_K
            | QuantizationScheme::GGUF_Q3_K
            | QuantizationScheme::GGUF_Q4_K
            | QuantizationScheme::GGUF_Q5_0
            | QuantizationScheme::GGUF_Q6_K => {
                // Note: GGUF dequantization should use MobileGGUFQuantizer directly
                // For now, fall back to INT8
                self.int8_quantizer.dequantize_tensor(tensor)
            },
            QuantizationScheme::Dynamic => {
                // For dynamic schemes, fall back to the selection logic
                let selected_scheme = self.select_quantization_scheme(tensor)?;
                match selected_scheme {
                    QuantizationScheme::Int8 => self.int8_quantizer.dequantize_tensor(tensor),
                    QuantizationScheme::FP16 => self.fp16_quantizer.dequantize_tensor(tensor),
                    _ => self.int8_quantizer.dequantize_tensor(tensor), // Default fallback
                }
            },
        }
    }
}

/// Quantization utilities
pub struct QuantizationUtils;

impl QuantizationUtils {
    /// Compute quantization error
    pub fn compute_error(original: &Tensor, quantized: &Tensor) -> Result<f32> {
        let orig_data = original.data()?;
        let quant_data = quantized.data()?;

        if orig_data.len() != quant_data.len() {
            return Err(tensor_op_error(
                "compute_error",
                "Tensors must have same size for error computation",
            ));
        }

        let mse = orig_data
            .iter()
            .zip(quant_data.iter())
            .map(|(&o, &q)| (o - q).powi(2))
            .sum::<f32>()
            / orig_data.len() as f32;

        Ok(mse.sqrt())
    }

    /// Get compression ratio
    pub fn compression_ratio(scheme: QuantizationScheme) -> f32 {
        match scheme {
            QuantizationScheme::Int4 => 8.0,                // 32-bit to 4-bit
            QuantizationScheme::Int8 => 4.0,                // 32-bit to 8-bit
            QuantizationScheme::FP16 => 2.0,                // 32-bit to 16-bit
            QuantizationScheme::Dynamic => 3.0,             // Average
            QuantizationScheme::GGUF_Q2_K => 32.0 / 2.5625, // 32-bit to 2.5625-bit
            QuantizationScheme::GGUF_Q3_K => 32.0 / 3.4375, // 32-bit to 3.4375-bit
            QuantizationScheme::GGUF_Q4_K => 32.0 / 4.5,    // 32-bit to 4.5-bit
            QuantizationScheme::GGUF_Q5_0 => 32.0 / 5.5,    // 32-bit to 5.5-bit
            QuantizationScheme::GGUF_Q6_K => 32.0 / 6.5,    // 32-bit to 6.5-bit
        }
    }

    /// Estimate memory savings
    pub fn memory_savings_percent(scheme: QuantizationScheme) -> f32 {
        let ratio = Self::compression_ratio(scheme);
        (1.0 - 1.0 / ratio) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int4_quantization() {
        let quantizer = Int4Quantizer::new();
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4])
            .expect("Failed to create tensor");

        // Calibrate
        quantizer.calibrate(&[tensor.clone()]).expect("Calibration failed");

        // Quantize
        let quantized = quantizer.quantize_tensor(&tensor).expect("Quantization failed");
        assert_eq!(quantized.shape(), tensor.shape());

        // Dequantize
        let dequantized = quantizer.dequantize_tensor(&quantized).expect("Dequantization failed");
        assert_eq!(dequantized.shape(), tensor.shape());

        // Check error is reasonable
        let error = QuantizationUtils::compute_error(&tensor, &dequantized)
            .expect("Error computation failed");
        assert!(error < 1.0); // Error should be small
    }

    #[test]
    fn test_int8_quantization() {
        let quantizer = Int8Quantizer::new();
        let tensor = Tensor::from_vec(vec![-10.0, -5.0, 0.0, 5.0, 10.0], &[5])
            .expect("Failed to create tensor");

        quantizer.calibrate(&[tensor.clone()]).expect("Calibration failed");

        let quantized = quantizer.quantize_tensor(&tensor).expect("Quantization failed");
        let dequantized = quantizer.dequantize_tensor(&quantized).expect("Dequantization failed");

        let error = QuantizationUtils::compute_error(&tensor, &dequantized)
            .expect("Error computation failed");
        assert!(error < 0.1); // INT8 should have very low error
    }

    #[test]
    fn test_fp16_quantization() {
        let quantizer = FP16Quantizer::new();
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        // FP16 doesn't require calibration
        assert!(!quantizer.requires_calibration());

        let quantized = quantizer.quantize_tensor(&tensor).expect("Quantization failed");
        let dequantized = quantizer.dequantize_tensor(&quantized).expect("Dequantization failed");

        // FP16 should have minimal error for normal range values
        let error = QuantizationUtils::compute_error(&tensor, &dequantized)
            .expect("Error computation failed");
        assert!(error < 0.001);
    }

    #[test]
    fn test_dynamic_quantization() {
        let mut quantizer = DynamicQuantizer::new();

        // Small range tensor - should use INT8
        let small_range = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[4]).unwrap();

        quantizer.calibrate(&[small_range.clone()]).unwrap();
        let quantized = quantizer.quantize_tensor(&small_range).unwrap();

        // Test external storage functionality
        let tensor_id = QuantizationSchemeStorage::generate_tensor_id(&small_range, None);

        // Configure external storage to use FP16 for this tensor
        quantizer
            .scheme_storage_mut()
            .set_tensor_scheme(tensor_id.clone(), QuantizationScheme::FP16);

        // Quantize again - should now use FP16 due to external storage
        let quantized_fp16 = quantizer.quantize_tensor(&small_range).unwrap();

        // Verify the storage can retrieve the scheme
        let stored_scheme = quantizer.scheme_storage_mut().determine_scheme(&tensor_id, None, None);
        assert_eq!(stored_scheme, QuantizationScheme::FP16);

        // Test default fallback
        let unknown_tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let unknown_id = QuantizationSchemeStorage::generate_tensor_id(&unknown_tensor, None);
        let default_scheme =
            quantizer.scheme_storage_mut().determine_scheme(&unknown_id, None, None);
        assert_eq!(default_scheme, QuantizationScheme::Int8); // Default scheme
    }

    #[test]
    fn test_compression_ratios() {
        assert_eq!(
            QuantizationUtils::compression_ratio(QuantizationScheme::Int4),
            8.0
        );
        assert_eq!(
            QuantizationUtils::compression_ratio(QuantizationScheme::Int8),
            4.0
        );
        assert_eq!(
            QuantizationUtils::compression_ratio(QuantizationScheme::FP16),
            2.0
        );

        assert_eq!(
            QuantizationUtils::memory_savings_percent(QuantizationScheme::Int4),
            87.5
        );
        assert_eq!(
            QuantizationUtils::memory_savings_percent(QuantizationScheme::Int8),
            75.0
        );
        assert_eq!(
            QuantizationUtils::memory_savings_percent(QuantizationScheme::FP16),
            50.0
        );
    }
}
