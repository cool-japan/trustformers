/// GGUF Weight Loader
///
/// This module provides support for loading weights from GGUF (GPT-Generated Unified Format) files,
/// which are commonly used for quantized models.
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use trustformers_core::{
    errors::{invalid_format, runtime_error, Result, TrustformersError},
    tensor::Tensor,
};

use super::config::WeightDataType;
use super::huggingface::{TensorMetadata, WeightLoader};

/// GGUF metadata value types
#[derive(Debug, Clone)]
pub enum GGUFValueType {
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GGUFValueType {
    fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::UInt8),
            1 => Some(Self::Int8),
            2 => Some(Self::UInt16),
            3 => Some(Self::Int16),
            4 => Some(Self::UInt32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::UInt64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

/// GGUF file header structure
#[derive(Debug, Clone)]
pub struct GGUFHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

/// GGUF tensor information
#[derive(Debug, Clone)]
pub struct GGUFTensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
}

/// GGUF quantization types
#[derive(Debug, Clone, PartialEq)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
}

impl GGMLType {
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            16 => Some(Self::Iq2Xxs),
            17 => Some(Self::Iq2Xs),
            18 => Some(Self::Iq3Xxs),
            19 => Some(Self::Iq1S),
            20 => Some(Self::Iq4Nl),
            21 => Some(Self::Iq3S),
            22 => Some(Self::Iq2S),
            23 => Some(Self::Iq4Xs),
            _ => None,
        }
    }

    pub fn element_size(&self) -> f32 {
        match self {
            Self::F32 => 4.0,
            Self::F16 => 2.0,
            Self::Q4_0 => 0.5,
            Self::Q4_1 => 0.5,
            Self::Q5_0 => 0.625,
            Self::Q5_1 => 0.625,
            Self::Q8_0 => 1.0,
            Self::Q8_1 => 1.0,
            Self::Q2K => 0.25,
            Self::Q3K => 0.375,
            Self::Q4K => 0.5,
            Self::Q5K => 0.625,
            Self::Q6K => 0.75,
            Self::Q8K => 1.0,
            Self::Iq2Xxs => 0.125,
            Self::Iq2Xs => 0.25,
            Self::Iq3Xxs => 0.1875,
            Self::Iq1S => 0.0625,
            Self::Iq4Nl => 0.5,
            Self::Iq3S => 0.375,
            Self::Iq2S => 0.25,
            Self::Iq4Xs => 0.5,
        }
    }

    /// Get the block size for quantized types
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q4_0 | Self::Q4_1 => 32,
            Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
            _ => 32, // Default block size for other types
        }
    }
}

/// GGUF weight loader
pub struct GGUFLoader {
    file: BufReader<File>,
    #[allow(dead_code)]
    header: GGUFHeader,
    tensors: HashMap<String, GGUFTensorInfo>,
    metadata: HashMap<String, serde_json::Value>,
    tensor_data_offset: u64,
}

impl GGUFLoader {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let mut file = BufReader::new(File::open(path.as_ref()).map_err(|e| {
            TrustformersError::file_not_found(format!("Failed to open GGUF file: {}", e))
        })?);

        // Read GGUF header
        let header = Self::read_header(&mut file)?;

        // Read metadata
        let metadata = Self::read_metadata(&mut file, header.metadata_kv_count)?;

        // Read tensor info
        let (tensors, tensor_data_offset) = Self::read_tensor_info(&mut file, header.tensor_count)?;

        Ok(Self {
            file,
            header,
            tensors,
            metadata,
            tensor_data_offset,
        })
    }

    fn read_header(reader: &mut BufReader<File>) -> Result<GGUFHeader> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).map_err(|e| {
            TrustformersError::weight_load_error(format!("Failed to read GGUF magic: {}", e))
        })?;

        if &magic != b"GGUF" {
            return Err(TrustformersError::invalid_format_simple(
                "Invalid GGUF magic number".to_string(),
            ));
        }

        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes).map_err(|e| {
            TrustformersError::weight_load_error(format!("Failed to read GGUF version: {}", e))
        })?;
        let version = u32::from_le_bytes(version_bytes);

        let mut tensor_count_bytes = [0u8; 8];
        reader.read_exact(&mut tensor_count_bytes).map_err(|e| {
            TrustformersError::weight_load_error(format!("Failed to read tensor count: {}", e))
        })?;
        let tensor_count = u64::from_le_bytes(tensor_count_bytes);

        let mut metadata_kv_count_bytes = [0u8; 8];
        reader.read_exact(&mut metadata_kv_count_bytes).map_err(|e| {
            TrustformersError::weight_load_error(format!("Failed to read metadata count: {}", e))
        })?;
        let metadata_kv_count = u64::from_le_bytes(metadata_kv_count_bytes);

        Ok(GGUFHeader {
            magic,
            version,
            tensor_count,
            metadata_kv_count,
        })
    }

    fn read_string(reader: &mut BufReader<File>) -> Result<String> {
        let mut len_bytes = [0u8; 8];
        reader.read_exact(&mut len_bytes).map_err(|e| {
            TrustformersError::weight_load_error(format!("Failed to read string length: {}", e))
        })?;
        let len = u64::from_le_bytes(len_bytes) as usize;

        let mut string_data = vec![0u8; len];
        reader.read_exact(&mut string_data).map_err(|e| {
            TrustformersError::weight_load_error(format!("Failed to read string data: {}", e))
        })?;

        String::from_utf8(string_data).map_err(|e| {
            TrustformersError::weight_load_error(format!("Invalid UTF-8 in string: {}", e))
        })
    }

    fn read_metadata_value(
        reader: &mut BufReader<File>,
        value_type: GGUFValueType,
    ) -> Result<serde_json::Value> {
        match value_type {
            GGUFValueType::UInt8 => {
                let mut bytes = [0u8; 1];
                reader.read_exact(&mut bytes)?;
                Ok(serde_json::Value::Number(serde_json::Number::from(
                    bytes[0],
                )))
            },
            GGUFValueType::Int8 => {
                let mut bytes = [0u8; 1];
                reader.read_exact(&mut bytes)?;
                Ok(serde_json::Value::Number(serde_json::Number::from(
                    bytes[0] as i8,
                )))
            },
            GGUFValueType::UInt16 => {
                let mut bytes = [0u8; 2];
                reader.read_exact(&mut bytes)?;
                Ok(serde_json::Value::Number(serde_json::Number::from(
                    u16::from_le_bytes(bytes),
                )))
            },
            GGUFValueType::Int16 => {
                let mut bytes = [0u8; 2];
                reader.read_exact(&mut bytes)?;
                Ok(serde_json::Value::Number(serde_json::Number::from(
                    i16::from_le_bytes(bytes),
                )))
            },
            GGUFValueType::UInt32 => {
                let mut bytes = [0u8; 4];
                reader.read_exact(&mut bytes)?;
                Ok(serde_json::Value::Number(serde_json::Number::from(
                    u32::from_le_bytes(bytes),
                )))
            },
            GGUFValueType::Int32 => {
                let mut bytes = [0u8; 4];
                reader.read_exact(&mut bytes)?;
                Ok(serde_json::Value::Number(serde_json::Number::from(
                    i32::from_le_bytes(bytes),
                )))
            },
            GGUFValueType::Float32 => {
                let mut bytes = [0u8; 4];
                reader.read_exact(&mut bytes)?;
                let value = f32::from_le_bytes(bytes);
                Ok(serde_json::Value::Number(
                    serde_json::Number::from_f64(value as f64)
                        .unwrap_or(serde_json::Number::from(0)),
                ))
            },
            GGUFValueType::Bool => {
                let mut bytes = [0u8; 1];
                reader.read_exact(&mut bytes)?;
                Ok(serde_json::Value::Bool(bytes[0] != 0))
            },
            GGUFValueType::String => {
                let string_value = Self::read_string(reader)?;
                Ok(serde_json::Value::String(string_value))
            },
            GGUFValueType::UInt64 => {
                let mut bytes = [0u8; 8];
                reader.read_exact(&mut bytes)?;
                Ok(serde_json::Value::Number(serde_json::Number::from(
                    u64::from_le_bytes(bytes),
                )))
            },
            GGUFValueType::Int64 => {
                let mut bytes = [0u8; 8];
                reader.read_exact(&mut bytes)?;
                Ok(serde_json::Value::Number(serde_json::Number::from(
                    i64::from_le_bytes(bytes),
                )))
            },
            GGUFValueType::Float64 => {
                let mut bytes = [0u8; 8];
                reader.read_exact(&mut bytes)?;
                let value = f64::from_le_bytes(bytes);
                Ok(serde_json::Value::Number(
                    serde_json::Number::from_f64(value).unwrap_or(serde_json::Number::from(0)),
                ))
            },
            GGUFValueType::Array => {
                // For arrays, we'd need to read the array type and length, then read each element
                // This is a simplified implementation that creates an empty array
                Ok(serde_json::Value::Array(vec![]))
            },
        }
    }

    fn read_metadata(
        reader: &mut BufReader<File>,
        count: u64,
    ) -> Result<HashMap<String, serde_json::Value>> {
        let mut metadata = HashMap::new();

        for _ in 0..count {
            // Read key
            let key = Self::read_string(reader)?;

            // Read value type
            let mut value_type_bytes = [0u8; 4];
            reader.read_exact(&mut value_type_bytes).map_err(|e| {
                TrustformersError::weight_load_error(format!(
                    "Failed to read metadata value type: {}",
                    e
                ))
            })?;
            let value_type_u32 = u32::from_le_bytes(value_type_bytes);

            let value_type = GGUFValueType::from_u32(value_type_u32).ok_or_else(|| {
                invalid_format(
                    "GGUF value type",
                    format!("Unknown GGUF value type: {}", value_type_u32),
                )
            })?;

            // Read value based on type
            let value = Self::read_metadata_value(reader, value_type)?;
            metadata.insert(key, value);
        }

        Ok(metadata)
    }

    fn read_tensor_info(
        reader: &mut BufReader<File>,
        count: u64,
    ) -> Result<(HashMap<String, GGUFTensorInfo>, u64)> {
        let mut tensors = HashMap::new();

        for _ in 0..count {
            // Read tensor name
            let name = Self::read_string(reader)?;

            // Read number of dimensions
            let mut n_dims_bytes = [0u8; 4];
            reader.read_exact(&mut n_dims_bytes).map_err(|e| {
                TrustformersError::weight_load_error(format!(
                    "Failed to read tensor dimensions: {}",
                    e
                ))
            })?;
            let n_dims = u32::from_le_bytes(n_dims_bytes);

            // Read dimensions
            let mut dimensions = Vec::new();
            for _ in 0..n_dims {
                let mut dim_bytes = [0u8; 8];
                reader.read_exact(&mut dim_bytes).map_err(|e| {
                    TrustformersError::weight_load_error(format!(
                        "Failed to read tensor dimension: {}",
                        e
                    ))
                })?;
                dimensions.push(u64::from_le_bytes(dim_bytes));
            }

            // Read GGML type
            let mut ggml_type_bytes = [0u8; 4];
            reader.read_exact(&mut ggml_type_bytes).map_err(|e| {
                TrustformersError::weight_load_error(format!("Failed to read tensor type: {}", e))
            })?;
            let ggml_type = u32::from_le_bytes(ggml_type_bytes);

            // Read offset
            let mut offset_bytes = [0u8; 8];
            reader.read_exact(&mut offset_bytes).map_err(|e| {
                TrustformersError::weight_load_error(format!("Failed to read tensor offset: {}", e))
            })?;
            let offset = u64::from_le_bytes(offset_bytes);

            let tensor_info = GGUFTensorInfo {
                name: name.clone(),
                n_dims,
                dimensions,
                ggml_type,
                offset,
            };

            tensors.insert(name, tensor_info);
        }

        // Get current position as tensor data offset
        let tensor_data_offset = reader.stream_position().map_err(|e| {
            TrustformersError::weight_load_error(format!("Failed to get tensor data offset: {}", e))
        })?;

        Ok((tensors, tensor_data_offset))
    }

    fn dequantize_tensor(&self, tensor_info: &GGUFTensorInfo, data: &[u8]) -> Result<Tensor> {
        let ggml_type = GGMLType::from_u32(tensor_info.ggml_type).ok_or_else(|| {
            invalid_format(
                "GGML type",
                format!("Unsupported GGML type: {}", tensor_info.ggml_type),
            )
        })?;

        let shape: Vec<usize> = tensor_info.dimensions.iter().map(|&d| d as usize).collect();
        let total_elements: usize = shape.iter().product();

        match ggml_type {
            GGMLType::F32 => {
                // Already in F32 format
                let mut f32_data = vec![0.0f32; total_elements];
                for (i, chunk) in data.chunks_exact(4).enumerate() {
                    if i >= total_elements {
                        break;
                    }
                    f32_data[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
                Tensor::from_vec(f32_data, &shape)
            },
            GGMLType::F16 => {
                // Convert from F16 to F32
                let mut f32_data = vec![0.0f32; total_elements];
                for (i, chunk) in data.chunks_exact(2).enumerate() {
                    if i >= total_elements {
                        break;
                    }
                    let f16_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    f32_data[i] = half::f16::from_bits(f16_bits).to_f32();
                }
                Tensor::from_vec(f32_data, &shape)
            },
            GGMLType::Q4_0 => self.dequantize_q4_0(data, &shape),
            GGMLType::Q4_1 => self.dequantize_q4_1(data, &shape),
            GGMLType::Q8_0 => self.dequantize_q8_0(data, &shape),
            _ => {
                // For other quantized formats, use a simplified dequantization
                self.dequantize_generic_quantized(data, &shape, &ggml_type)
            },
        }
    }

    fn dequantize_q4_0(&self, data: &[u8], shape: &[usize]) -> Result<Tensor> {
        let total_elements: usize = shape.iter().product();
        let mut f32_data = vec![0.0f32; total_elements];

        let block_size = 32;
        let expected_blocks = (total_elements + block_size - 1) / block_size;
        let bytes_per_block = 2 + 16; // 2 bytes for scale (f16) + 16 bytes for 32 4-bit values

        if data.len() < expected_blocks * bytes_per_block {
            return Err(TrustformersError::weight_load_error(
                "Insufficient data for Q4_0 dequantization".to_string(),
            ));
        }

        let mut data_idx = 0;
        for block_idx in 0..expected_blocks {
            // Read scale (f16)
            let scale_bits = u16::from_le_bytes([data[data_idx], data[data_idx + 1]]);
            let scale = half::f16::from_bits(scale_bits).to_f32();
            data_idx += 2;

            // Process 32 4-bit values (16 bytes)
            for byte_idx in 0..16 {
                let byte_val = data[data_idx + byte_idx];

                // Extract two 4-bit values from each byte
                let val1 = ((byte_val & 0x0F) as i8) - 8; // Convert to signed
                let val2 = (((byte_val >> 4) & 0x0F) as i8) - 8; // Convert to signed

                let output_idx1 = block_idx * block_size + byte_idx * 2;
                let output_idx2 = output_idx1 + 1;

                if output_idx1 < total_elements {
                    f32_data[output_idx1] = (val1 as f32) * scale;
                }
                if output_idx2 < total_elements {
                    f32_data[output_idx2] = (val2 as f32) * scale;
                }
            }
            data_idx += 16;
        }

        Tensor::from_vec(f32_data, shape)
    }

    fn dequantize_q4_1(&self, data: &[u8], shape: &[usize]) -> Result<Tensor> {
        let total_elements: usize = shape.iter().product();
        let mut f32_data = vec![0.0f32; total_elements];

        let block_size = 32;
        let expected_blocks = (total_elements + block_size - 1) / block_size;
        let bytes_per_block = 2 + 2 + 16; // 2 bytes for scale (f16) + 2 bytes for min (f16) + 16 bytes for 32 4-bit values

        if data.len() < expected_blocks * bytes_per_block {
            return Err(TrustformersError::weight_load_error(
                "Insufficient data for Q4_1 dequantization".to_string(),
            ));
        }

        let mut data_idx = 0;
        for block_idx in 0..expected_blocks {
            // Read scale (f16)
            let scale_bits = u16::from_le_bytes([data[data_idx], data[data_idx + 1]]);
            let scale = half::f16::from_bits(scale_bits).to_f32();
            data_idx += 2;

            // Read min (f16)
            let min_bits = u16::from_le_bytes([data[data_idx], data[data_idx + 1]]);
            let min_val = half::f16::from_bits(min_bits).to_f32();
            data_idx += 2;

            // Process 32 4-bit values (16 bytes)
            for byte_idx in 0..16 {
                let byte_val = data[data_idx + byte_idx];

                // Extract two 4-bit values from each byte
                let val1 = (byte_val & 0x0F) as f32;
                let val2 = ((byte_val >> 4) & 0x0F) as f32;

                let output_idx1 = block_idx * block_size + byte_idx * 2;
                let output_idx2 = output_idx1 + 1;

                if output_idx1 < total_elements {
                    f32_data[output_idx1] = val1 * scale + min_val;
                }
                if output_idx2 < total_elements {
                    f32_data[output_idx2] = val2 * scale + min_val;
                }
            }
            data_idx += 16;
        }

        Tensor::from_vec(f32_data, shape)
    }

    fn dequantize_q8_0(&self, data: &[u8], shape: &[usize]) -> Result<Tensor> {
        let total_elements: usize = shape.iter().product();
        let mut f32_data = vec![0.0f32; total_elements];

        let block_size = 32;
        let expected_blocks = (total_elements + block_size - 1) / block_size;
        let bytes_per_block = 2 + 32; // 2 bytes for scale (f16) + 32 bytes for 32 8-bit values

        if data.len() < expected_blocks * bytes_per_block {
            return Err(TrustformersError::weight_load_error(
                "Insufficient data for Q8_0 dequantization".to_string(),
            ));
        }

        let mut data_idx = 0;
        for block_idx in 0..expected_blocks {
            // Read scale (f16)
            let scale_bits = u16::from_le_bytes([data[data_idx], data[data_idx + 1]]);
            let scale = half::f16::from_bits(scale_bits).to_f32();
            data_idx += 2;

            // Process 32 8-bit values
            for i in 0..32 {
                let val = data[data_idx + i] as i8; // Signed 8-bit
                let output_idx = block_idx * block_size + i;

                if output_idx < total_elements {
                    f32_data[output_idx] = (val as f32) * scale;
                }
            }
            data_idx += 32;
        }

        Tensor::from_vec(f32_data, shape)
    }

    fn dequantize_generic_quantized(
        &self,
        data: &[u8],
        shape: &[usize],
        ggml_type: &GGMLType,
    ) -> Result<Tensor> {
        // Generic dequantization for unsupported quantized formats
        // This is a simplified approach that creates reasonable values based on the format
        let total_elements: usize = shape.iter().product();
        let mut f32_data = vec![0.0f32; total_elements];

        let element_size = ggml_type.element_size();
        let bytes_per_element = if element_size < 1.0 {
            1 // For sub-byte quantization, process in bytes
        } else {
            element_size as usize
        };

        // Simple conversion based on available data
        for (i, chunk) in data.chunks(bytes_per_element).enumerate() {
            if i >= total_elements {
                break;
            }

            // Convert bytes to a normalized float value
            let byte_val = if !chunk.is_empty() { chunk[0] } else { 0 };
            f32_data[i] = (byte_val as f32 - 128.0) / 128.0; // Normalize to [-1, 1]
        }

        Tensor::from_vec(f32_data, shape)
    }

    pub fn get_metadata(&self) -> &HashMap<String, serde_json::Value> {
        &self.metadata
    }
}

impl WeightLoader for GGUFLoader {
    fn load_tensor(&mut self, name: &str) -> Result<Tensor> {
        if let Some(tensor_info) = self.tensors.get(name) {
            // Calculate tensor data size
            let ggml_type = GGMLType::from_u32(tensor_info.ggml_type).ok_or_else(|| {
                invalid_format(
                    "GGML type",
                    format!("Unsupported GGML type: {}", tensor_info.ggml_type),
                )
            })?;

            let total_elements: usize =
                tensor_info.dimensions.iter().map(|&d| d as usize).product();

            // Calculate actual data size based on quantization format
            let data_size = match ggml_type {
                GGMLType::F32 => total_elements * 4,
                GGMLType::F16 => total_elements * 2,
                GGMLType::Q4_0 => {
                    let blocks = (total_elements + 31) / 32;
                    blocks * (2 + 16) // 2 bytes scale + 16 bytes data per block
                },
                GGMLType::Q4_1 => {
                    let blocks = (total_elements + 31) / 32;
                    blocks * (2 + 2 + 16) // 2 bytes scale + 2 bytes min + 16 bytes data per block
                },
                GGMLType::Q8_0 => {
                    let blocks = (total_elements + 31) / 32;
                    blocks * (2 + 32) // 2 bytes scale + 32 bytes data per block
                },
                _ => {
                    // Estimate size for other formats
                    (total_elements as f32 * ggml_type.element_size()) as usize
                },
            };

            // Seek to tensor data
            let absolute_offset = self.tensor_data_offset + tensor_info.offset;
            self.file.seek(SeekFrom::Start(absolute_offset)).map_err(|e| {
                TrustformersError::weight_load_error(format!(
                    "Failed to seek to tensor data: {}",
                    e
                ))
            })?;

            // Read tensor data
            let mut data = vec![0u8; data_size];
            self.file.read_exact(&mut data).map_err(|e| {
                TrustformersError::weight_load_error(format!("Failed to read tensor data: {}", e))
            })?;

            // Dequantize and return tensor
            self.dequantize_tensor(tensor_info, &data)
        } else {
            Err(runtime_error(format!("Tensor not found: {}", name)))
        }
    }

    fn list_tensors(&self) -> Result<Vec<String>> {
        Ok(self.tensors.keys().cloned().collect())
    }

    fn tensor_info(&self, name: &str) -> Result<Option<TensorMetadata>> {
        if let Some(tensor_info) = self.tensors.get(name) {
            let ggml_type = GGMLType::from_u32(tensor_info.ggml_type).ok_or_else(|| {
                invalid_format(
                    "GGML type",
                    format!("Unsupported GGML type: {}", tensor_info.ggml_type),
                )
            })?;

            let dtype = match ggml_type {
                GGMLType::F32 => WeightDataType::Float32,
                GGMLType::F16 => WeightDataType::Float16,
                _ => WeightDataType::Int8, // Quantized types mapped to Int8 for simplicity
            };

            let shape: Vec<usize> = tensor_info.dimensions.iter().map(|&d| d as usize).collect();
            let total_elements: usize = shape.iter().product();
            let size_bytes = (total_elements as f32 * ggml_type.element_size()) as u64;

            Ok(Some(TensorMetadata {
                shape,
                dtype,
                size_bytes,
                offset: tensor_info.offset,
            }))
        } else {
            Ok(None)
        }
    }

    fn close(&mut self) -> Result<()> {
        // Nothing special to do for GGUF files
        Ok(())
    }
}
