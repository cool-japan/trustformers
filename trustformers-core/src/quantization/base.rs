#![allow(unused_variables)] // Quantization base implementation

use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Quantization schemes supported by TrustformeRS
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationScheme {
    /// 8-bit integer quantization
    Int8,
    /// 4-bit integer quantization (weight-only)
    Int4,
    /// Dynamic quantization (runtime quantization)
    Dynamic,
    /// Dynamic 8-bit integer quantization (runtime quantization)
    DynamicINT8,
    /// GPTQ (Gradient-based Post-Training Quantization)
    GPTQ,
    /// AWQ (Activation-aware Weight Quantization)
    AWQ,
    /// BitsAndBytes 8-bit quantization
    BnB8bit,
    /// BitsAndBytes 4-bit NormalFloat quantization
    BnB4bit,
    /// BitsAndBytes 4-bit Float16 quantization
    BnB4bitFP4,
}

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub scheme: QuantizationScheme,
    pub symmetric: bool,
    pub per_channel: bool,
    pub calibration_samples: Option<usize>,
    pub group_size: Option<usize>,     // For grouped quantization
    pub bnb_config: Option<BnBConfig>, // BitsAndBytes specific configuration
}

/// BitsAndBytes quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BnBConfig {
    pub use_double_quant: bool,
    pub quant_type: BnBQuantType,
    pub compute_dtype: BnBComputeType,
    pub bnb_4bit_quant_storage: Option<BnBStorageType>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BnBQuantType {
    NF4,  // NormalFloat 4-bit
    FP4,  // Float 4-bit
    Int8, // Integer 8-bit
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BnBComputeType {
    Float16,
    BFloat16,
    Float32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BnBStorageType {
    UInt8,
    Int8,
    Float16,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            scheme: QuantizationScheme::Int8,
            symmetric: true,
            per_channel: false,
            calibration_samples: Some(128),
            group_size: Some(128),
            bnb_config: None,
        }
    }
}

impl Default for BnBConfig {
    fn default() -> Self {
        Self {
            use_double_quant: false,
            quant_type: BnBQuantType::NF4,
            compute_dtype: BnBComputeType::Float16,
            bnb_4bit_quant_storage: Some(BnBStorageType::UInt8),
        }
    }
}

/// Quantized tensor representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedTensor {
    pub data: Vec<u8>,
    pub scale: Vec<f32>,
    pub zero_point: Vec<i32>,
    pub shape: Vec<usize>,
    pub scheme: QuantizationScheme,
    pub per_channel: bool,
}

impl QuantizedTensor {
    /// Create a new quantized tensor
    pub fn new(
        data: Vec<u8>,
        scale: Vec<f32>,
        zero_point: Vec<i32>,
        shape: Vec<usize>,
        scheme: QuantizationScheme,
        per_channel: bool,
    ) -> Self {
        Self {
            data,
            scale,
            zero_point,
            shape,
            scheme,
            per_channel,
        }
    }

    /// Dequantize back to f32 tensor
    pub fn dequantize(&self) -> Result<Tensor> {
        let total_elements: usize = self.shape.iter().product();
        let mut result = Vec::with_capacity(total_elements);

        match self.scheme {
            QuantizationScheme::Int8 | QuantizationScheme::BnB8bit => {
                if self.per_channel {
                    self.dequantize_per_channel_int8(&mut result)?;
                } else {
                    self.dequantize_per_tensor_int8(&mut result)?;
                }
            },
            QuantizationScheme::Int4 => {
                if self.per_channel {
                    self.dequantize_per_channel_int4(&mut result)?;
                } else {
                    self.dequantize_per_tensor_int4(&mut result)?;
                }
            },
            QuantizationScheme::Dynamic | QuantizationScheme::DynamicINT8 => {
                // Dynamic quantization uses runtime statistics
                if self.per_channel {
                    self.dequantize_per_channel_int8(&mut result)?;
                } else {
                    self.dequantize_per_tensor_int8(&mut result)?;
                }
            },
            QuantizationScheme::GPTQ => {
                // GPTQ (Gradient-based Post-Training Quantization)
                // Uses optimized quantization with gradient information
                if self.per_channel {
                    self.dequantize_gptq_per_channel(&mut result)?;
                } else {
                    self.dequantize_gptq_per_tensor(&mut result)?;
                }
            },
            QuantizationScheme::AWQ => {
                // AWQ (Activation-aware Weight Quantization)
                // Uses activation statistics for better quantization
                if self.per_channel {
                    self.dequantize_awq_per_channel(&mut result)?;
                } else {
                    self.dequantize_awq_per_tensor(&mut result)?;
                }
            },
            QuantizationScheme::BnB4bit => {
                // BitsAndBytes 4-bit NormalFloat quantization
                if self.per_channel {
                    self.dequantize_bnb_4bit_per_channel(&mut result)?;
                } else {
                    self.dequantize_bnb_4bit_per_tensor(&mut result)?;
                }
            },
            QuantizationScheme::BnB4bitFP4 => {
                // BitsAndBytes 4-bit Float16 quantization
                if self.per_channel {
                    self.dequantize_bnb_fp4_per_channel(&mut result)?;
                } else {
                    self.dequantize_bnb_fp4_per_tensor(&mut result)?;
                }
            },
        }

        Tensor::from_vec(result, &self.shape)
    }

    fn dequantize_per_tensor_int8(&self, result: &mut Vec<f32>) -> Result<()> {
        if self.scale.len() != 1 || self.zero_point.len() != 1 {
            return Err(TrustformersError::quantization_error(
                "Per-tensor quantization requires single scale and zero point".into(),
            ));
        }

        let scale = self.scale[0];
        let zero_point = self.zero_point[0];

        for &quantized_val in &self.data {
            let int_val = quantized_val as i32 - zero_point;
            let float_val = int_val as f32 * scale;
            result.push(float_val);
        }

        Ok(())
    }

    fn dequantize_per_channel_int8(&self, result: &mut Vec<f32>) -> Result<()> {
        let channels = self.scale.len();
        let elements_per_channel = self.data.len() / channels;

        for (channel_idx, (&scale, &zero_point)) in
            self.scale.iter().zip(&self.zero_point).enumerate()
        {
            let start_idx = channel_idx * elements_per_channel;
            let end_idx = start_idx + elements_per_channel;

            for &quantized_val in &self.data[start_idx..end_idx] {
                let int_val = quantized_val as i32 - zero_point;
                let float_val = int_val as f32 * scale;
                result.push(float_val);
            }
        }

        Ok(())
    }

    fn dequantize_per_tensor_int4(&self, result: &mut Vec<f32>) -> Result<()> {
        if self.scale.len() != 1 || self.zero_point.len() != 1 {
            return Err(TrustformersError::quantization_error(
                "Per-tensor quantization requires single scale and zero point".into(),
            ));
        }

        let scale = self.scale[0];
        let zero_point = self.zero_point[0];

        // Each byte contains 2 4-bit values
        for &byte in &self.data {
            // Extract high 4 bits
            let high_nibble = (byte >> 4) as i32 - zero_point;
            let high_val = high_nibble as f32 * scale;
            result.push(high_val);

            // Extract low 4 bits
            let low_nibble = (byte & 0x0F) as i32 - zero_point;
            let low_val = low_nibble as f32 * scale;
            result.push(low_val);
        }

        Ok(())
    }

    fn dequantize_per_channel_int4(&self, result: &mut Vec<f32>) -> Result<()> {
        let channels = self.scale.len();
        let bytes_per_channel = self.data.len() / channels;

        for (channel_idx, (&scale, &zero_point)) in
            self.scale.iter().zip(&self.zero_point).enumerate()
        {
            let start_idx = channel_idx * bytes_per_channel;
            let end_idx = start_idx + bytes_per_channel;

            for &byte in &self.data[start_idx..end_idx] {
                // Extract high 4 bits
                let high_nibble = (byte >> 4) as i32 - zero_point;
                let high_val = high_nibble as f32 * scale;
                result.push(high_val);

                // Extract low 4 bits
                let low_nibble = (byte & 0x0F) as i32 - zero_point;
                let low_val = low_nibble as f32 * scale;
                result.push(low_val);
            }
        }

        Ok(())
    }

    /// GPTQ dequantization per tensor
    fn dequantize_gptq_per_tensor(&self, result: &mut Vec<f32>) -> Result<()> {
        if self.scale.len() != 1 || self.zero_point.len() != 1 {
            return Err(TrustformersError::quantization_error(
                "GPTQ per-tensor quantization requires single scale and zero point".into(),
            ));
        }

        let scale = self.scale[0];
        let zero_point = self.zero_point[0];

        // GPTQ uses optimized quantization with gradient information
        // Similar to int8 but with better error compensation
        for &quantized_val in &self.data {
            let int_val = quantized_val as i32 - zero_point;
            let float_val = int_val as f32 * scale;
            result.push(float_val);
        }

        Ok(())
    }

    /// GPTQ dequantization per channel
    fn dequantize_gptq_per_channel(&self, result: &mut Vec<f32>) -> Result<()> {
        let channels = self.scale.len();
        let elements_per_channel = self.data.len() / channels;

        for (channel_idx, (&scale, &zero_point)) in
            self.scale.iter().zip(&self.zero_point).enumerate()
        {
            let start_idx = channel_idx * elements_per_channel;
            let end_idx = start_idx + elements_per_channel;

            for &quantized_val in &self.data[start_idx..end_idx] {
                let int_val = quantized_val as i32 - zero_point;
                let float_val = int_val as f32 * scale;
                result.push(float_val);
            }
        }

        Ok(())
    }

    /// AWQ dequantization per tensor
    fn dequantize_awq_per_tensor(&self, result: &mut Vec<f32>) -> Result<()> {
        if self.scale.len() != 1 || self.zero_point.len() != 1 {
            return Err(TrustformersError::quantization_error(
                "AWQ per-tensor quantization requires single scale and zero point".into(),
            ));
        }

        let scale = self.scale[0];
        let zero_point = self.zero_point[0];

        // AWQ uses activation-aware weight quantization
        // Similar to int8 but optimized for specific activation patterns
        for &quantized_val in &self.data {
            let int_val = quantized_val as i32 - zero_point;
            let float_val = int_val as f32 * scale;
            result.push(float_val);
        }

        Ok(())
    }

    /// AWQ dequantization per channel
    fn dequantize_awq_per_channel(&self, result: &mut Vec<f32>) -> Result<()> {
        let channels = self.scale.len();
        let elements_per_channel = self.data.len() / channels;

        for (channel_idx, (&scale, &zero_point)) in
            self.scale.iter().zip(&self.zero_point).enumerate()
        {
            let start_idx = channel_idx * elements_per_channel;
            let end_idx = start_idx + elements_per_channel;

            for &quantized_val in &self.data[start_idx..end_idx] {
                let int_val = quantized_val as i32 - zero_point;
                let float_val = int_val as f32 * scale;
                result.push(float_val);
            }
        }

        Ok(())
    }

    /// BitsAndBytes 4-bit NormalFloat dequantization per tensor
    fn dequantize_bnb_4bit_per_tensor(&self, result: &mut Vec<f32>) -> Result<()> {
        // NF4 (NormalFloat 4-bit) uses a non-uniform quantization table
        // optimized for normal distributions of weights
        const NF4_LEVELS: [f32; 16] = [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ];

        if self.scale.len() != 1 {
            return Err(TrustformersError::quantization_error(
                "BnB 4-bit per-tensor quantization requires single scale".into(),
            ));
        }

        let scale = self.scale[0];

        for &byte in &self.data {
            // Extract high 4 bits
            let high_nibble = (byte >> 4) & 0x0F;
            let high_val = NF4_LEVELS[high_nibble as usize] * scale;
            result.push(high_val);

            // Extract low 4 bits
            let low_nibble = byte & 0x0F;
            let low_val = NF4_LEVELS[low_nibble as usize] * scale;
            result.push(low_val);
        }

        Ok(())
    }

    /// BitsAndBytes 4-bit NormalFloat dequantization per channel
    fn dequantize_bnb_4bit_per_channel(&self, result: &mut Vec<f32>) -> Result<()> {
        const NF4_LEVELS: [f32; 16] = [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ];

        let channels = self.scale.len();
        let bytes_per_channel = self.data.len() / channels;

        for (channel_idx, &scale) in self.scale.iter().enumerate() {
            let start_idx = channel_idx * bytes_per_channel;
            let end_idx = start_idx + bytes_per_channel;

            for &byte in &self.data[start_idx..end_idx] {
                // Extract high 4 bits
                let high_nibble = (byte >> 4) & 0x0F;
                let high_val = NF4_LEVELS[high_nibble as usize] * scale;
                result.push(high_val);

                // Extract low 4 bits
                let low_nibble = byte & 0x0F;
                let low_val = NF4_LEVELS[low_nibble as usize] * scale;
                result.push(low_val);
            }
        }

        Ok(())
    }

    /// BitsAndBytes 4-bit Float16 dequantization per tensor
    fn dequantize_bnb_fp4_per_tensor(&self, result: &mut Vec<f32>) -> Result<()> {
        // FP4 uses a uniform quantization table for Float16 values
        const FP4_LEVELS: [f32; 16] = [
            -12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0,
        ];

        if self.scale.len() != 1 {
            return Err(TrustformersError::quantization_error(
                "BnB FP4 per-tensor quantization requires single scale".into(),
            ));
        }

        let scale = self.scale[0];

        for &byte in &self.data {
            // Extract high 4 bits
            let high_nibble = (byte >> 4) & 0x0F;
            let high_val = FP4_LEVELS[high_nibble as usize] * scale;
            result.push(high_val);

            // Extract low 4 bits
            let low_nibble = byte & 0x0F;
            let low_val = FP4_LEVELS[low_nibble as usize] * scale;
            result.push(low_val);
        }

        Ok(())
    }

    /// BitsAndBytes 4-bit Float16 dequantization per channel
    fn dequantize_bnb_fp4_per_channel(&self, result: &mut Vec<f32>) -> Result<()> {
        const FP4_LEVELS: [f32; 16] = [
            -12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0,
        ];

        let channels = self.scale.len();
        let bytes_per_channel = self.data.len() / channels;

        for (channel_idx, &scale) in self.scale.iter().enumerate() {
            let start_idx = channel_idx * bytes_per_channel;
            let end_idx = start_idx + bytes_per_channel;

            for &byte in &self.data[start_idx..end_idx] {
                // Extract high 4 bits
                let high_nibble = (byte >> 4) & 0x0F;
                let high_val = FP4_LEVELS[high_nibble as usize] * scale;
                result.push(high_val);

                // Extract low 4 bits
                let low_nibble = byte & 0x0F;
                let low_val = FP4_LEVELS[low_nibble as usize] * scale;
                result.push(low_val);
            }
        }

        Ok(())
    }
}

/// Quantization utilities
pub struct Quantizer;

impl Quantizer {
    /// Quantize a tensor according to the given configuration
    pub fn quantize(tensor: &Tensor, config: &QuantizationConfig) -> Result<QuantizedTensor> {
        match config.scheme {
            QuantizationScheme::Int8 => {
                if config.per_channel {
                    Self::quantize_per_channel_int8(tensor, config.symmetric)
                } else {
                    Self::quantize_per_tensor_int8(tensor, config.symmetric)
                }
            },
            QuantizationScheme::Int4 => {
                if config.per_channel {
                    Self::quantize_per_channel_int4(tensor, config.symmetric, config.group_size)
                } else {
                    Self::quantize_per_tensor_int4(tensor, config.symmetric)
                }
            },
            QuantizationScheme::Dynamic => Self::dynamic_quantize(tensor),
            QuantizationScheme::DynamicINT8 => {
                // Dynamic INT8 quantization - same as Dynamic but explicitly 8-bit
                Self::dynamic_quantize(tensor)
            },
            QuantizationScheme::GPTQ => {
                // GPTQ (Gradient-based Post-Training Quantization)
                // For now, use standard INT4 quantization with optimized settings
                // Full GPTQ implementation would require Hessian computation
                if config.per_channel {
                    Self::quantize_per_channel_int4(tensor, true, config.group_size)
                } else {
                    Self::quantize_per_tensor_int4(tensor, true)
                }
            },
            QuantizationScheme::AWQ => {
                // AWQ (Activation-aware Weight Quantization)
                // For now, use standard INT4 quantization with symmetric mode
                // Full AWQ implementation would use activation statistics
                if config.per_channel {
                    Self::quantize_per_channel_int4(tensor, true, config.group_size)
                } else {
                    Self::quantize_per_tensor_int4(tensor, true)
                }
            },
            QuantizationScheme::BnB8bit => {
                // BitsAndBytes 8-bit quantization
                let bnb_config = config.bnb_config.clone().unwrap_or(BnBConfig {
                    use_double_quant: false,
                    quant_type: BnBQuantType::Int8,
                    compute_dtype: BnBComputeType::Float16,
                    bnb_4bit_quant_storage: None,
                });
                let quantizer = BnBQuantizer::new(bnb_config);
                quantizer.quantize_bnb_int8(tensor)
            },
            QuantizationScheme::BnB4bit => {
                // BitsAndBytes 4-bit NormalFloat quantization
                let bnb_config = config.bnb_config.clone().unwrap_or(BnBConfig {
                    use_double_quant: false,
                    quant_type: BnBQuantType::NF4,
                    compute_dtype: BnBComputeType::Float16,
                    bnb_4bit_quant_storage: Some(BnBStorageType::UInt8),
                });
                let quantizer = BnBQuantizer::new(bnb_config);
                quantizer.quantize_nf4(tensor)
            },
            QuantizationScheme::BnB4bitFP4 => {
                // BitsAndBytes 4-bit Float16 quantization
                let bnb_config = config.bnb_config.clone().unwrap_or(BnBConfig {
                    use_double_quant: false,
                    quant_type: BnBQuantType::FP4,
                    compute_dtype: BnBComputeType::Float16,
                    bnb_4bit_quant_storage: Some(BnBStorageType::UInt8),
                });
                let quantizer = BnBQuantizer::new(bnb_config);
                quantizer.quantize_fp4(tensor)
            },
        }
    }

    /// Per-tensor 8-bit quantization
    fn quantize_per_tensor_int8(tensor: &Tensor, symmetric: bool) -> Result<QuantizedTensor> {
        match tensor {
            Tensor::F32(arr) => {
                let data = arr.iter().cloned().collect::<Vec<f32>>();
                let (scale, zero_point) = Self::compute_quantization_params(&data, symmetric, 8)?;

                let quantized_data: Vec<u8> = data
                    .iter()
                    .map(|&val| Self::quantize_value_int8(val, scale, zero_point))
                    .collect();

                Ok(QuantizedTensor::new(
                    quantized_data,
                    vec![scale],
                    vec![zero_point],
                    arr.shape().to_vec(),
                    QuantizationScheme::Int8,
                    false,
                ))
            },
            _ => Err(TrustformersError::quantization_error(
                "Unsupported tensor type for quantization".into(),
            )),
        }
    }

    /// Per-channel 8-bit quantization
    fn quantize_per_channel_int8(tensor: &Tensor, symmetric: bool) -> Result<QuantizedTensor> {
        match tensor {
            Tensor::F32(arr) => {
                let shape = arr.shape();
                let channels = shape[0]; // Assume first dimension is channels
                let elements_per_channel = arr.len() / channels;

                let mut scales = Vec::with_capacity(channels);
                let mut zero_points = Vec::with_capacity(channels);
                let mut quantized_data = Vec::with_capacity(arr.len());

                for channel in 0..channels {
                    let start_idx = channel * elements_per_channel;
                    let end_idx = start_idx + elements_per_channel;
                    let channel_data = arr
                        .iter()
                        .skip(start_idx)
                        .take(elements_per_channel)
                        .cloned()
                        .collect::<Vec<f32>>();

                    let (scale, zero_point) =
                        Self::compute_quantization_params(&channel_data, symmetric, 8)?;
                    scales.push(scale);
                    zero_points.push(zero_point);

                    let channel_quantized: Vec<u8> = channel_data
                        .iter()
                        .map(|&val| Self::quantize_value_int8(val, scale, zero_point))
                        .collect();

                    quantized_data.extend(channel_quantized);
                }

                Ok(QuantizedTensor::new(
                    quantized_data,
                    scales,
                    zero_points,
                    shape.to_vec(),
                    QuantizationScheme::Int8,
                    true,
                ))
            },
            _ => Err(TrustformersError::quantization_error(
                "Unsupported tensor type for quantization".into(),
            )),
        }
    }

    /// Per-tensor 4-bit quantization (weight-only)
    fn quantize_per_tensor_int4(tensor: &Tensor, symmetric: bool) -> Result<QuantizedTensor> {
        match tensor {
            Tensor::F32(arr) => {
                let data = arr.iter().cloned().collect::<Vec<f32>>();
                let (scale, zero_point) = Self::compute_quantization_params(&data, symmetric, 4)?;

                let quantized_data = Self::pack_int4_values(&data, scale, zero_point)?;

                Ok(QuantizedTensor::new(
                    quantized_data,
                    vec![scale],
                    vec![zero_point],
                    arr.shape().to_vec(),
                    QuantizationScheme::Int4,
                    false,
                ))
            },
            _ => Err(TrustformersError::quantization_error(
                "Unsupported tensor type for quantization".into(),
            )),
        }
    }

    /// Per-channel 4-bit quantization with optional grouping
    fn quantize_per_channel_int4(
        tensor: &Tensor,
        symmetric: bool,
        group_size: Option<usize>,
    ) -> Result<QuantizedTensor> {
        match tensor {
            Tensor::F32(arr) => {
                let shape = arr.shape();
                let total_elements = arr.len();
                let group_size = group_size.unwrap_or(128);
                let num_groups = total_elements.div_ceil(group_size);

                let mut scales = Vec::with_capacity(num_groups);
                let mut zero_points = Vec::with_capacity(num_groups);
                let mut quantized_data = Vec::with_capacity(total_elements / 2); // 4-bit packing

                for group_idx in 0..num_groups {
                    let start_idx = group_idx * group_size;
                    let end_idx = (start_idx + group_size).min(total_elements);

                    let group_data = arr
                        .iter()
                        .skip(start_idx)
                        .take(end_idx - start_idx)
                        .cloned()
                        .collect::<Vec<f32>>();

                    let (scale, zero_point) =
                        Self::compute_quantization_params(&group_data, symmetric, 4)?;
                    scales.push(scale);
                    zero_points.push(zero_point);

                    let group_quantized = Self::pack_int4_values(&group_data, scale, zero_point)?;
                    quantized_data.extend(group_quantized);
                }

                Ok(QuantizedTensor::new(
                    quantized_data,
                    scales,
                    zero_points,
                    shape.to_vec(),
                    QuantizationScheme::Int4,
                    true,
                ))
            },
            _ => Err(TrustformersError::quantization_error(
                "Unsupported tensor type for quantization".into(),
            )),
        }
    }

    /// Dynamic quantization (quantize at runtime)
    fn dynamic_quantize(tensor: &Tensor) -> Result<QuantizedTensor> {
        // For dynamic quantization, we quantize to int8 per-tensor
        Self::quantize_per_tensor_int8(tensor, false)
    }

    /// Compute quantization parameters (scale and zero point)
    fn compute_quantization_params(data: &[f32], symmetric: bool, bits: u8) -> Result<(f32, i32)> {
        if data.is_empty() {
            return Err(TrustformersError::quantization_error(
                "Cannot quantize empty data".into(),
            ));
        }

        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let q_min = if symmetric { -(1 << (bits - 1)) } else { 0 };
        let q_max = if symmetric { (1 << (bits - 1)) - 1 } else { (1 << bits) - 1 };

        let (scale, zero_point) = if symmetric {
            let abs_max = max_val.abs().max(min_val.abs());
            let scale = abs_max / (q_max - q_min) as f32;
            (scale, 0)
        } else {
            let scale = (max_val - min_val) / (q_max - q_min) as f32;
            let zero_point = q_min - (min_val / scale).round() as i32;
            let zero_point = zero_point.clamp(q_min, q_max);
            (scale, zero_point)
        };

        Ok((scale, zero_point))
    }

    /// Quantize a single float value to int8
    fn quantize_value_int8(value: f32, scale: f32, zero_point: i32) -> u8 {
        let quantized = (value / scale).round() as i32 + zero_point;
        quantized.clamp(0, 255) as u8
    }

    /// Pack float values into 4-bit representation
    fn pack_int4_values(data: &[f32], scale: f32, zero_point: i32) -> Result<Vec<u8>> {
        let mut packed = Vec::with_capacity(data.len().div_ceil(2));

        for chunk in data.chunks(2) {
            let val1 = Self::quantize_value_int4(chunk[0], scale, zero_point);
            let val2 = if chunk.len() > 1 {
                Self::quantize_value_int4(chunk[1], scale, zero_point)
            } else {
                0 // Pad with zero
            };

            // Pack two 4-bit values into one byte
            let packed_byte = (val1 << 4) | val2;
            packed.push(packed_byte);
        }

        Ok(packed)
    }

    /// Quantize a single float value to int4
    fn quantize_value_int4(value: f32, scale: f32, zero_point: i32) -> u8 {
        let quantized = (value / scale).round() as i32 + zero_point;
        quantized.clamp(0, 15) as u8
    }

    /// Calibrate quantization parameters using sample data
    pub fn calibrate(
        samples: &[Tensor],
        config: &QuantizationConfig,
    ) -> Result<QuantizationConfig> {
        // This is a simplified calibration - in practice, you'd run the model
        // on representative data and collect activation statistics
        let mut calibrated_config = config.clone();

        if let Some(sample_count) = config.calibration_samples {
            let num_samples = samples.len().min(sample_count);

            // Collect statistics from samples
            let mut all_values = Vec::new();
            for sample in samples.iter().take(num_samples) {
                match sample {
                    Tensor::F32(arr) => {
                        all_values.extend(arr.iter().cloned());
                    },
                    _ => continue,
                }
            }

            if !all_values.is_empty() {
                // Update configuration based on calibration data
                let abs_max = all_values.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));

                // Adjust symmetric flag based on data distribution
                let min_val = all_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max_val = all_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                calibrated_config.symmetric =
                    (min_val.abs() - max_val.abs()).abs() / max_val.abs() < 0.1;
            }
        }

        Ok(calibrated_config)
    }
}

/// GPTQ (Gradient-based Post-Training Quantization) implementation
pub struct GPTQQuantizer {
    config: QuantizationConfig,
}

impl GPTQQuantizer {
    pub fn new(config: QuantizationConfig) -> Self {
        Self { config }
    }

    /// Apply GPTQ quantization to a tensor
    /// This is a simplified version - full GPTQ requires Hessian computation
    pub fn quantize(&self, tensor: &Tensor, hessian: Option<&Tensor>) -> Result<QuantizedTensor> {
        // For now, fall back to standard quantization
        // In a full implementation, this would use the Hessian to minimize quantization error
        Quantizer::quantize(tensor, &self.config)
    }
}

/// AWQ (Activation-aware Weight Quantization) implementation
pub struct AWQQuantizer {
    config: QuantizationConfig,
    activation_scales: Option<Vec<f32>>,
}

impl AWQQuantizer {
    pub fn new(config: QuantizationConfig) -> Self {
        Self {
            config,
            activation_scales: None,
        }
    }

    /// Set activation scales for weight quantization
    pub fn set_activation_scales(&mut self, scales: Vec<f32>) {
        self.activation_scales = Some(scales);
    }

    /// Apply AWQ quantization to a tensor
    pub fn quantize(&self, tensor: &Tensor) -> Result<QuantizedTensor> {
        // For now, fall back to standard quantization
        // In a full implementation, this would use activation scales to improve quantization
        Quantizer::quantize(tensor, &self.config)
    }
}

/// BitsAndBytes quantizer implementation
pub struct BnBQuantizer {
    config: BnBConfig,
}

impl BnBQuantizer {
    pub fn new(config: BnBConfig) -> Self {
        Self { config }
    }

    /// Quantize tensor using BitsAndBytes method
    pub fn quantize(&self, tensor: &Tensor) -> Result<QuantizedTensor> {
        match self.config.quant_type {
            BnBQuantType::NF4 => self.quantize_nf4(tensor),
            BnBQuantType::FP4 => self.quantize_fp4(tensor),
            BnBQuantType::Int8 => self.quantize_bnb_int8(tensor),
        }
    }

    /// NormalFloat 4-bit quantization (BitsAndBytes NF4)
    fn quantize_nf4(&self, tensor: &Tensor) -> Result<QuantizedTensor> {
        match tensor {
            Tensor::F32(arr) => {
                let data = arr.iter().cloned().collect::<Vec<f32>>();
                let block_size = 64; // Standard block size for NF4

                let mut quantized_data = Vec::new();
                let mut scales = Vec::new();
                let mut zero_points = Vec::new();

                for chunk in data.chunks(block_size) {
                    let (block_scale, block_quantized) = self.nf4_quantize_block(chunk)?;
                    scales.push(block_scale);
                    zero_points.push(0); // NF4 is symmetric
                    quantized_data.extend(block_quantized);
                }

                Ok(QuantizedTensor::new(
                    quantized_data,
                    scales,
                    zero_points,
                    arr.shape().to_vec(),
                    QuantizationScheme::BnB4bit,
                    false,
                ))
            },
            _ => Err(TrustformersError::quantization_error(
                "Unsupported tensor type for BnB NF4".into(),
            )),
        }
    }

    /// Float 4-bit quantization (BitsAndBytes FP4)
    fn quantize_fp4(&self, tensor: &Tensor) -> Result<QuantizedTensor> {
        match tensor {
            Tensor::F32(arr) => {
                let data = arr.iter().cloned().collect::<Vec<f32>>();
                let block_size = 64;

                let mut quantized_data = Vec::new();
                let mut scales = Vec::new();
                let mut zero_points = Vec::new();

                for chunk in data.chunks(block_size) {
                    let (block_scale, block_quantized) = self.fp4_quantize_block(chunk)?;
                    scales.push(block_scale);
                    zero_points.push(0); // FP4 is symmetric
                    quantized_data.extend(block_quantized);
                }

                Ok(QuantizedTensor::new(
                    quantized_data,
                    scales,
                    zero_points,
                    arr.shape().to_vec(),
                    QuantizationScheme::BnB4bitFP4,
                    false,
                ))
            },
            _ => Err(TrustformersError::quantization_error(
                "Unsupported tensor type for BnB FP4".into(),
            )),
        }
    }

    /// BitsAndBytes 8-bit integer quantization
    fn quantize_bnb_int8(&self, tensor: &Tensor) -> Result<QuantizedTensor> {
        match tensor {
            Tensor::F32(arr) => {
                let data = arr.iter().cloned().collect::<Vec<f32>>();
                let (scale, zero_point) = Quantizer::compute_quantization_params(&data, false, 8)?;

                let quantized_data: Vec<u8> = data
                    .iter()
                    .map(|&val| Quantizer::quantize_value_int8(val, scale, zero_point))
                    .collect();

                Ok(QuantizedTensor::new(
                    quantized_data,
                    vec![scale],
                    vec![zero_point],
                    arr.shape().to_vec(),
                    QuantizationScheme::BnB8bit,
                    false,
                ))
            },
            _ => Err(TrustformersError::quantization_error(
                "Unsupported tensor type for BnB Int8".into(),
            )),
        }
    }

    /// NF4 block quantization with predefined quantization levels
    fn nf4_quantize_block(&self, data: &[f32]) -> Result<(f32, Vec<u8>)> {
        // NF4 quantization levels (based on normal distribution)
        const NF4_LEVELS: [f32; 16] = [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ];

        if data.is_empty() {
            return Err(TrustformersError::quantization_error(
                "Cannot quantize empty block".into(),
            ));
        }

        // Compute block scale
        let abs_max = data.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        let scale = abs_max;

        if scale == 0.0 {
            return Ok((scale, vec![0; data.len()]));
        }

        // Quantize each value to nearest NF4 level
        let mut quantized = Vec::with_capacity(data.len());
        for &val in data {
            let normalized = val / scale;
            let mut best_idx = 0;
            let mut best_dist = (normalized - NF4_LEVELS[0]).abs();

            for (idx, &level) in NF4_LEVELS.iter().enumerate().skip(1) {
                let dist = (normalized - level).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = idx;
                }
            }

            quantized.push(best_idx as u8);
        }

        Ok((scale, quantized))
    }

    /// FP4 block quantization with floating-point levels
    fn fp4_quantize_block(&self, data: &[f32]) -> Result<(f32, Vec<u8>)> {
        // FP4 quantization levels (exponential distribution)
        const FP4_LEVELS: [f32; 16] = [
            0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.625, 0.75, 0.875, 1.0,
            1.25, 1.5, 2.0,
        ];

        if data.is_empty() {
            return Err(TrustformersError::quantization_error(
                "Cannot quantize empty block".into(),
            ));
        }

        // Compute block scale
        let abs_max = data.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        let scale = abs_max / 2.0; // Scale for signed values

        if scale == 0.0 {
            return Ok((scale, vec![0; data.len()]));
        }

        // Quantize each value
        let mut quantized = Vec::with_capacity(data.len());
        for &val in data {
            let abs_val = val.abs() / scale;
            let sign = if val >= 0.0 { 0 } else { 8 }; // Use bit 3 for sign

            let mut best_idx = 0;
            let mut best_dist = (abs_val - FP4_LEVELS[0]).abs();

            for (idx, &level) in FP4_LEVELS[..8].iter().enumerate().skip(1) {
                let dist = (abs_val - level).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = idx;
                }
            }

            quantized.push((sign | best_idx) as u8);
        }

        Ok((scale, quantized))
    }

    /// Dequantize BitsAndBytes tensor
    pub fn dequantize(&self, tensor: &QuantizedTensor) -> Result<Tensor> {
        match tensor.scheme {
            QuantizationScheme::BnB4bit => self.dequantize_nf4(tensor),
            QuantizationScheme::BnB4bitFP4 => self.dequantize_fp4(tensor),
            QuantizationScheme::BnB8bit => tensor.dequantize(), // Use standard dequantization
            _ => Err(TrustformersError::quantization_error(format!(
                "BnB dequantization not supported for scheme {:?}",
                tensor.scheme
            ))),
        }
    }

    /// Dequantize NF4 tensor
    fn dequantize_nf4(&self, tensor: &QuantizedTensor) -> Result<Tensor> {
        const NF4_LEVELS: [f32; 16] = [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ];

        let block_size = 64;
        let mut result = Vec::with_capacity(tensor.data.len());
        let num_blocks = tensor.scale.len();

        for block_idx in 0..num_blocks {
            let scale = tensor.scale[block_idx];
            let start_idx = block_idx * block_size;
            let end_idx = (start_idx + block_size).min(tensor.data.len());

            for &quantized_val in &tensor.data[start_idx..end_idx] {
                let idx = (quantized_val as usize).min(15);
                let dequantized = NF4_LEVELS[idx] * scale;
                result.push(dequantized);
            }
        }

        Tensor::from_vec(result, &tensor.shape)
    }

    /// Dequantize FP4 tensor
    fn dequantize_fp4(&self, tensor: &QuantizedTensor) -> Result<Tensor> {
        const FP4_LEVELS: [f32; 8] = [0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375];

        let block_size = 64;
        let mut result = Vec::with_capacity(tensor.data.len());
        let num_blocks = tensor.scale.len();

        for block_idx in 0..num_blocks {
            let scale = tensor.scale[block_idx];
            let start_idx = block_idx * block_size;
            let end_idx = (start_idx + block_size).min(tensor.data.len());

            for &quantized_val in &tensor.data[start_idx..end_idx] {
                let sign = if (quantized_val & 8) != 0 { -1.0 } else { 1.0 };
                let idx = (quantized_val & 7) as usize;
                let abs_val = FP4_LEVELS[idx];
                let dequantized = sign * abs_val * scale;
                result.push(dequantized);
            }
        }

        Tensor::from_vec(result, &tensor.shape)
    }
}

/// Quantization-aware training support
pub struct QATConfig {
    pub fake_quantize: bool,
    pub observe: bool,
    pub reduce_range: bool,
    pub qscheme: QuantizationScheme,
}

impl Default for QATConfig {
    fn default() -> Self {
        Self {
            fake_quantize: true,
            observe: true,
            reduce_range: false,
            qscheme: QuantizationScheme::Int8,
        }
    }
}

/// Fake quantization for QAT
pub struct FakeQuantize {
    config: QATConfig,
    observers: Vec<Observer>,
}

/// Observer for collecting statistics during QAT
pub struct Observer {
    min_val: f32,
    max_val: f32,
    count: usize,
}

impl Default for Observer {
    fn default() -> Self {
        Self::new()
    }
}

impl Observer {
    pub fn new() -> Self {
        Self {
            min_val: f32::INFINITY,
            max_val: f32::NEG_INFINITY,
            count: 0,
        }
    }

    pub fn update(&mut self, tensor: &Tensor) {
        if let Tensor::F32(arr) = tensor {
            for &val in arr.iter() {
                self.min_val = self.min_val.min(val);
                self.max_val = self.max_val.max(val);
                self.count += 1;
            }
        }
    }

    pub fn get_quantization_params(&self, symmetric: bool, bits: u8) -> Result<(f32, i32)> {
        if self.count == 0 {
            return Err(TrustformersError::quantization_error(
                "No observations for quantization".into(),
            ));
        }

        let q_min = if symmetric { -(1 << (bits - 1)) } else { 0 };
        let q_max = if symmetric { (1 << (bits - 1)) - 1 } else { (1 << bits) - 1 };

        let (scale, zero_point) = if symmetric {
            let abs_max = self.max_val.abs().max(self.min_val.abs());
            let scale = abs_max / (q_max - q_min) as f32;
            (scale, 0)
        } else {
            let scale = (self.max_val - self.min_val) / (q_max - q_min) as f32;
            let zero_point = q_min - (self.min_val / scale).round() as i32;
            let zero_point = zero_point.clamp(q_min, q_max);
            (scale, zero_point)
        };

        Ok((scale, zero_point))
    }
}

impl FakeQuantize {
    pub fn new(config: QATConfig) -> Self {
        Self {
            config,
            observers: Vec::new(),
        }
    }

    /// Apply fake quantization during training
    pub fn forward(&mut self, tensor: &Tensor) -> Result<Tensor> {
        if self.config.observe {
            // Update observer statistics
            if self.observers.is_empty() {
                self.observers.push(Observer::new());
            }
            self.observers[0].update(tensor);
        }

        if self.config.fake_quantize && !self.observers.is_empty() {
            // Apply fake quantization
            let observer = &self.observers[0];
            let (scale, zero_point) = observer.get_quantization_params(true, 8)?;

            // Quantize and immediately dequantize
            match tensor {
                Tensor::F32(arr) => {
                    let quantized_data: Vec<f32> = arr
                        .iter()
                        .map(|&val| {
                            let q_val = Quantizer::quantize_value_int8(val, scale, zero_point);
                            let int_val = q_val as i32 - zero_point;
                            int_val as f32 * scale
                        })
                        .collect();

                    Tensor::from_vec(quantized_data, arr.shape())
                },
                _ => Ok(tensor.clone()),
            }
        } else {
            Ok(tensor.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int8_per_tensor_quantization() {
        let tensor = Tensor::randn(&[10, 20]).unwrap();
        let config = QuantizationConfig {
            scheme: QuantizationScheme::Int8,
            symmetric: true,
            per_channel: false,
            calibration_samples: None,
            group_size: None,
            bnb_config: None,
        };

        let quantized = Quantizer::quantize(&tensor, &config).unwrap();
        assert_eq!(quantized.scheme, QuantizationScheme::Int8);
        assert!(!quantized.per_channel);
        assert_eq!(quantized.scale.len(), 1);
        assert_eq!(quantized.zero_point.len(), 1);

        let dequantized = quantized.dequantize().unwrap();
        assert_eq!(dequantized.shape(), tensor.shape());
    }

    #[test]
    fn test_int4_per_tensor_quantization() {
        let tensor = Tensor::randn(&[8, 16]).unwrap();
        let config = QuantizationConfig {
            scheme: QuantizationScheme::Int4,
            symmetric: false,
            per_channel: false,
            calibration_samples: None,
            group_size: None,
            bnb_config: None,
        };

        let quantized = Quantizer::quantize(&tensor, &config).unwrap();
        assert_eq!(quantized.scheme, QuantizationScheme::Int4);
        assert!(!quantized.per_channel);

        let dequantized = quantized.dequantize().unwrap();
        assert_eq!(dequantized.shape(), tensor.shape());
    }

    #[test]
    fn test_per_channel_quantization() {
        let tensor = Tensor::randn(&[4, 32]).unwrap();
        let config = QuantizationConfig {
            scheme: QuantizationScheme::Int8,
            symmetric: true,
            per_channel: true,
            calibration_samples: None,
            group_size: None,
            bnb_config: None,
        };

        let quantized = Quantizer::quantize(&tensor, &config).unwrap();
        assert!(quantized.per_channel);
        assert_eq!(quantized.scale.len(), 4); // Number of channels
        assert_eq!(quantized.zero_point.len(), 4);

        let dequantized = quantized.dequantize().unwrap();
        assert_eq!(dequantized.shape(), tensor.shape());
    }

    #[test]
    fn test_dynamic_quantization() {
        let tensor = Tensor::randn(&[16, 32]).unwrap();
        let config = QuantizationConfig {
            scheme: QuantizationScheme::Dynamic,
            symmetric: false,
            per_channel: false,
            calibration_samples: None,
            group_size: None,
            bnb_config: None,
        };

        let quantized = Quantizer::quantize(&tensor, &config).unwrap();
        let dequantized = quantized.dequantize().unwrap();
        assert_eq!(dequantized.shape(), tensor.shape());
    }

    #[test]
    fn test_quantization_params_computation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Symmetric quantization
        let (scale, zero_point) = Quantizer::compute_quantization_params(&data, true, 8).unwrap();
        assert_eq!(zero_point, 0);
        assert!(scale > 0.0);

        // Asymmetric quantization
        let (scale, zero_point) = Quantizer::compute_quantization_params(&data, false, 8).unwrap();
        assert!(scale > 0.0);
    }

    #[test]
    fn test_gptq_quantizer() {
        let tensor = Tensor::randn(&[16, 32]).unwrap();
        let config = QuantizationConfig::default();
        let gptq = GPTQQuantizer::new(config);

        let quantized = gptq.quantize(&tensor, None).unwrap();
        let dequantized = quantized.dequantize().unwrap();
        assert_eq!(dequantized.shape(), tensor.shape());
    }

    #[test]
    fn test_awq_quantizer() {
        let tensor = Tensor::randn(&[16, 32]).unwrap();
        let config = QuantizationConfig::default();
        let mut awq = AWQQuantizer::new(config);

        let scales = vec![1.0; 16];
        awq.set_activation_scales(scales);

        let quantized = awq.quantize(&tensor).unwrap();
        let dequantized = quantized.dequantize().unwrap();
        assert_eq!(dequantized.shape(), tensor.shape());
    }

    #[test]
    fn test_calibration() {
        let samples = vec![
            Tensor::randn(&[16, 32]).unwrap(),
            Tensor::randn(&[16, 32]).unwrap(),
            Tensor::randn(&[16, 32]).unwrap(),
        ];

        let config = QuantizationConfig {
            calibration_samples: Some(2),
            ..Default::default()
        };

        let calibrated = Quantizer::calibrate(&samples, &config).unwrap();
        assert_eq!(calibrated.scheme, config.scheme);
    }

    #[test]
    fn test_bnb_nf4_quantization() {
        let tensor = Tensor::randn(&[128]).unwrap();
        let config = BnBConfig {
            quant_type: BnBQuantType::NF4,
            compute_dtype: BnBComputeType::Float16,
            use_double_quant: false,
            bnb_4bit_quant_storage: Some(BnBStorageType::UInt8),
        };

        let bnb = BnBQuantizer::new(config);
        let quantized = bnb.quantize(&tensor).unwrap();
        assert_eq!(quantized.scheme, QuantizationScheme::BnB4bit);

        let dequantized = bnb.dequantize(&quantized).unwrap();
        assert_eq!(dequantized.shape(), tensor.shape());
    }

    #[test]
    fn test_bnb_fp4_quantization() {
        let tensor = Tensor::randn(&[128]).unwrap();
        let config = BnBConfig {
            quant_type: BnBQuantType::FP4,
            compute_dtype: BnBComputeType::Float16,
            use_double_quant: false,
            bnb_4bit_quant_storage: Some(BnBStorageType::UInt8),
        };

        let bnb = BnBQuantizer::new(config);
        let quantized = bnb.quantize(&tensor).unwrap();
        assert_eq!(quantized.scheme, QuantizationScheme::BnB4bitFP4);

        let dequantized = bnb.dequantize(&quantized).unwrap();
        assert_eq!(dequantized.shape(), tensor.shape());
    }

    #[test]
    fn test_bnb_int8_quantization() {
        let tensor = Tensor::randn(&[64, 64]).unwrap();
        let config = BnBConfig {
            quant_type: BnBQuantType::Int8,
            compute_dtype: BnBComputeType::Float32,
            use_double_quant: false,
            bnb_4bit_quant_storage: None,
        };

        let bnb = BnBQuantizer::new(config);
        let quantized = bnb.quantize(&tensor).unwrap();
        assert_eq!(quantized.scheme, QuantizationScheme::BnB8bit);

        let dequantized = quantized.dequantize().unwrap();
        assert_eq!(dequantized.shape(), tensor.shape());
    }

    #[test]
    fn test_qat_fake_quantize() {
        let tensor = Tensor::randn(&[32, 32]).unwrap();
        let config = QATConfig::default();
        let mut fake_quant = FakeQuantize::new(config);

        // First pass should build observer statistics
        let result1 = fake_quant.forward(&tensor).unwrap();
        assert_eq!(result1.shape(), tensor.shape());

        // Second pass should apply fake quantization
        let result2 = fake_quant.forward(&tensor).unwrap();
        assert_eq!(result2.shape(), tensor.shape());
    }

    #[test]
    fn test_observer_statistics() {
        let mut observer = Observer::new();
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();

        observer.update(&tensor);
        assert_eq!(observer.count, 5);

        let (scale, zero_point) = observer.get_quantization_params(true, 8).unwrap();
        assert!(scale > 0.0);
        assert_eq!(zero_point, 0); // Symmetric quantization
    }

    #[test]
    fn test_bnb_config_serialization() {
        let config = BnBConfig {
            quant_type: BnBQuantType::NF4,
            compute_dtype: BnBComputeType::Float16,
            use_double_quant: true,
            bnb_4bit_quant_storage: Some(BnBStorageType::UInt8),
        };

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: BnBConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config.quant_type, deserialized.quant_type);
        assert_eq!(config.compute_dtype, deserialized.compute_dtype);
        assert_eq!(config.use_double_quant, deserialized.use_double_quant);
    }
}
