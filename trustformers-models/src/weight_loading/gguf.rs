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
use super::gguf_dequant::{block_geometry, dequantize};
use super::huggingface::{TensorMetadata, WeightLoader};

/// Default tensor-data alignment when `general.alignment` is absent.
const DEFAULT_GGUF_ALIGNMENT: u64 = 32;

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

    /// Average bytes per element for this type.
    ///
    /// Derived from the exact block geometry rather than from a nominal
    /// bits-per-weight figure: a `Q4_K` super-block spends 144 bytes on 256
    /// elements (0.5625 B/elem), not the 0.5 that "4-bit" suggests, because it
    /// also carries `d`, `dmin` and twelve packed 6-bit scale bytes. Sizing a
    /// read with the nominal figure truncates every K-quant tensor.
    pub fn element_size(&self) -> f32 {
        let geometry = block_geometry(self);
        geometry.bytes_per_block as f32 / geometry.elements_per_block as f32
    }

    /// Number of tensor elements encoded by one block of this type.
    pub fn block_size(&self) -> usize {
        block_geometry(self).elements_per_block
    }

    /// Number of bytes one block of this type occupies on disk.
    pub fn bytes_per_block(&self) -> usize {
        block_geometry(self).bytes_per_block
    }

    /// Exact on-disk byte size of a tensor with `elements` values.
    pub fn storage_size(&self, elements: usize) -> usize {
        block_geometry(self).bytes_for(elements)
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
        let (tensors, info_end) = Self::read_tensor_info(&mut file, header.tensor_count)?;

        // The tensor data section starts at the first offset at or after the end
        // of the tensor-info table that is a multiple of `general.alignment`
        // (32 by default). Using the raw stream position instead reads every
        // tensor from a few bytes before its real start.
        let alignment = Self::read_alignment(&metadata)?;
        let tensor_data_offset = info_end.div_ceil(alignment) * alignment;

        Ok(Self {
            file,
            header,
            tensors,
            metadata,
            tensor_data_offset,
        })
    }

    /// Tensor-data alignment declared by the file, or the GGUF default of 32.
    fn read_alignment(metadata: &HashMap<String, serde_json::Value>) -> Result<u64> {
        let Some(value) = metadata.get("general.alignment") else {
            return Ok(DEFAULT_GGUF_ALIGNMENT);
        };
        let alignment = value.as_u64().ok_or_else(|| {
            invalid_format(
                "general.alignment",
                format!("expected an unsigned integer, got {value}"),
            )
        })?;
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(invalid_format(
                "general.alignment",
                format!("expected a power of two, got {alignment}"),
            ));
        }
        Ok(alignment)
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
                // An array value is `[element type: u32][count: u64][elements...]`.
                // Every element must be consumed: skipping them leaves the reader
                // mispositioned, so every metadata key after the first array —
                // and then the whole tensor-info table — would be parsed from
                // the wrong offset.
                let mut element_type_bytes = [0u8; 4];
                reader.read_exact(&mut element_type_bytes).map_err(|e| {
                    TrustformersError::weight_load_error(format!(
                        "Failed to read GGUF array element type: {}",
                        e
                    ))
                })?;
                let element_type_u32 = u32::from_le_bytes(element_type_bytes);
                let element_type = GGUFValueType::from_u32(element_type_u32).ok_or_else(|| {
                    invalid_format(
                        "GGUF value type",
                        format!("Unknown GGUF array element type: {}", element_type_u32),
                    )
                })?;

                let mut count_bytes = [0u8; 8];
                reader.read_exact(&mut count_bytes).map_err(|e| {
                    TrustformersError::weight_load_error(format!(
                        "Failed to read GGUF array length: {}",
                        e
                    ))
                })?;
                let count = u64::from_le_bytes(count_bytes);

                let mut items = Vec::with_capacity(count.min(4096) as usize);
                for index in 0..count {
                    let item =
                        Self::read_metadata_value(reader, element_type.clone()).map_err(|e| {
                            TrustformersError::weight_load_error(format!(
                                "Failed to read GGUF array element {index} of {count}: {e}"
                            ))
                        })?;
                    items.push(item);
                }
                Ok(serde_json::Value::Array(items))
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

    /// Row-major tensor shape for a GGUF tensor descriptor.
    ///
    /// GGUF stores `ne[0]` as the fastest-varying dimension (the row length),
    /// which is the reverse of the row-major shape ndarray expects: a tensor
    /// written as `ne = [n_embd, n_vocab]` is an `[n_vocab, n_embd]` matrix.
    fn row_major_shape(tensor_info: &GGUFTensorInfo) -> Vec<usize> {
        tensor_info.dimensions.iter().rev().map(|&d| d as usize).collect()
    }

    /// Number of elements a GGUF tensor descriptor declares.
    fn element_count(tensor_info: &GGUFTensorInfo) -> usize {
        tensor_info.dimensions.iter().map(|&d| d as usize).product()
    }

    /// Resolve the ggml type id of a tensor descriptor.
    fn ggml_type_of(tensor_info: &GGUFTensorInfo) -> Result<GGMLType> {
        GGMLType::from_u32(tensor_info.ggml_type).ok_or_else(|| {
            invalid_format(
                "GGML type",
                format!(
                    "Unsupported GGML type id {} for tensor {}",
                    tensor_info.ggml_type, tensor_info.name
                ),
            )
        })
    }

    /// Dequantize a tensor payload into an f32 tensor.
    ///
    /// Every ggml type goes through [`super::gguf_dequant::dequantize`], which
    /// implements the real block layouts and returns an error for the types it
    /// does not implement. There is no generic fallback: a previous revision
    /// mapped unhandled types to `(byte - 128) / 128` and reported the result as
    /// a successfully loaded weight.
    fn dequantize_tensor(&self, tensor_info: &GGUFTensorInfo, data: &[u8]) -> Result<Tensor> {
        let ggml_type = Self::ggml_type_of(tensor_info)?;
        let shape = Self::row_major_shape(tensor_info);
        let total_elements = Self::element_count(tensor_info);

        let values = dequantize(&ggml_type, data, total_elements).map_err(|e| {
            TrustformersError::weight_load_error(format!(
                "Failed to dequantize tensor {}: {}",
                tensor_info.name, e
            ))
        })?;

        Tensor::from_vec(values, &shape)
    }

    pub fn get_metadata(&self) -> &HashMap<String, serde_json::Value> {
        &self.metadata
    }
}

impl WeightLoader for GGUFLoader {
    fn load_tensor(&mut self, name: &str) -> Result<Tensor> {
        let tensor_info = self
            .tensors
            .get(name)
            .cloned()
            .ok_or_else(|| runtime_error(format!("Tensor not found: {}", name)))?;

        let ggml_type = Self::ggml_type_of(&tensor_info)?;
        let total_elements = Self::element_count(&tensor_info);

        // Exact on-disk size from the ggml block geometry. An approximation here
        // truncates the payload of every K-quant tensor, because their
        // super-blocks carry scales and mins on top of the packed weights.
        let data_size = ggml_type.storage_size(total_elements);

        // Seek to tensor data
        let absolute_offset = self.tensor_data_offset + tensor_info.offset;
        self.file.seek(SeekFrom::Start(absolute_offset)).map_err(|e| {
            TrustformersError::weight_load_error(format!("Failed to seek to tensor data: {}", e))
        })?;

        // Read tensor data
        let mut data = vec![0u8; data_size];
        self.file.read_exact(&mut data).map_err(|e| {
            TrustformersError::weight_load_error(format!(
                "Failed to read {data_size} bytes of data for tensor {name}: {e}"
            ))
        })?;

        // Dequantize and return tensor
        self.dequantize_tensor(&tensor_info, &data)
    }

    fn list_tensors(&self) -> Result<Vec<String>> {
        Ok(self.tensors.keys().cloned().collect())
    }

    fn tensor_info(&self, name: &str) -> Result<Option<TensorMetadata>> {
        if let Some(tensor_info) = self.tensors.get(name) {
            let ggml_type = Self::ggml_type_of(tensor_info)?;

            let dtype = match ggml_type {
                GGMLType::F32 => WeightDataType::Float32,
                GGMLType::F16 => WeightDataType::Float16,
                // Every quantized ggml type is an integer-coded block format;
                // the loader materialises them as f32 after dequantization.
                _ => WeightDataType::Int8,
            };

            let shape = Self::row_major_shape(tensor_info);
            let total_elements = Self::element_count(tensor_info);
            let size_bytes = ggml_type.storage_size(total_elements) as u64;

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

#[cfg(test)]
mod tests {
    use super::*;

    // ── GGUFValueType tests ──────────────────────────────────────────────────

    #[test]
    fn test_gguf_value_type_from_u32_known_values() {
        let cases: &[(u32, bool)] = &[
            (0, true),   // UInt8
            (1, true),   // Int8
            (2, true),   // UInt16
            (3, true),   // Int16
            (4, true),   // UInt32
            (5, true),   // Int32
            (6, true),   // Float32
            (7, true),   // Bool
            (8, true),   // String
            (9, true),   // Array
            (10, true),  // UInt64
            (11, true),  // Int64
            (12, true),  // Float64
            (99, false), // Unknown
        ];
        for &(v, expected_some) in cases {
            let result = GGUFValueType::from_u32(v);
            assert_eq!(
                result.is_some(),
                expected_some,
                "from_u32({}) unexpected",
                v
            );
        }
    }

    // ── GGMLType tests ───────────────────────────────────────────────────────

    #[test]
    fn test_ggml_type_from_u32_f32() {
        let t = GGMLType::from_u32(0);
        assert!(matches!(t, Some(GGMLType::F32)));
    }

    #[test]
    fn test_ggml_type_from_u32_f16() {
        let t = GGMLType::from_u32(1);
        assert!(matches!(t, Some(GGMLType::F16)));
    }

    #[test]
    fn test_ggml_type_from_u32_quantized_types() {
        let q4_0 = GGMLType::from_u32(2);
        assert!(matches!(q4_0, Some(GGMLType::Q4_0)));
        let q8_0 = GGMLType::from_u32(8);
        assert!(matches!(q8_0, Some(GGMLType::Q8_0)));
    }

    #[test]
    fn test_ggml_type_from_u32_iq_types() {
        let iq2 = GGMLType::from_u32(16);
        assert!(matches!(iq2, Some(GGMLType::Iq2Xxs)));
        let iq4_xs = GGMLType::from_u32(23);
        assert!(matches!(iq4_xs, Some(GGMLType::Iq4Xs)));
    }

    #[test]
    fn test_ggml_type_from_u32_unknown_returns_none() {
        let unknown = GGMLType::from_u32(999);
        assert!(unknown.is_none());
    }

    #[test]
    fn test_ggml_type_element_size_f32() {
        assert!((GGMLType::F32.element_size() - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_ggml_type_element_size_f16() {
        assert!((GGMLType::F16.element_size() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_ggml_type_element_size_quantized_less_than_float() {
        let q4 = GGMLType::Q4_0.element_size();
        let f32_size = GGMLType::F32.element_size();
        assert!(q4 < f32_size, "Quantized type should be smaller than f32");
    }

    #[test]
    fn test_ggml_type_element_size_iq1s_smallest() {
        let iq1s = GGMLType::Iq1S.element_size();
        let f32_size = GGMLType::F32.element_size();
        assert!(iq1s < f32_size, "IQ1S should be much smaller than f32");
        assert!(iq1s > 0.0, "IQ1S element size should be positive");
    }

    #[test]
    fn test_ggml_type_block_size_f32_is_one() {
        assert_eq!(GGMLType::F32.block_size(), 1);
    }

    #[test]
    fn test_ggml_type_block_size_f16_is_one() {
        assert_eq!(GGMLType::F16.block_size(), 1);
    }

    #[test]
    fn test_ggml_type_block_size_q4_is_32() {
        assert_eq!(GGMLType::Q4_0.block_size(), 32);
        assert_eq!(GGMLType::Q4_1.block_size(), 32);
    }

    #[test]
    fn test_ggml_type_block_size_k_types_is_256() {
        assert_eq!(GGMLType::Q4K.block_size(), 256);
        assert_eq!(GGMLType::Q6K.block_size(), 256);
        assert_eq!(GGMLType::Q8K.block_size(), 256);
    }

    // ── GGUFHeader tests ─────────────────────────────────────────────────────

    #[test]
    fn test_gguf_header_construction() {
        let header = GGUFHeader {
            magic: *b"GGUF",
            version: 3,
            tensor_count: 128,
            metadata_kv_count: 10,
        };
        assert_eq!(&header.magic, b"GGUF");
        assert_eq!(header.version, 3);
        assert_eq!(header.tensor_count, 128);
        assert_eq!(header.metadata_kv_count, 10);
    }

    #[test]
    fn test_gguf_header_clone() {
        let header = GGUFHeader {
            magic: *b"GGUF",
            version: 2,
            tensor_count: 32,
            metadata_kv_count: 5,
        };
        let cloned = header.clone();
        assert_eq!(cloned.version, 2);
        assert_eq!(cloned.tensor_count, 32);
    }

    // ── GGUFTensorInfo tests ─────────────────────────────────────────────────

    #[test]
    fn test_gguf_tensor_info_construction() {
        let info = GGUFTensorInfo {
            name: "model.embed_tokens.weight".to_string(),
            n_dims: 2,
            dimensions: vec![32000, 4096],
            ggml_type: 0, // F32
            offset: 0,
        };
        assert_eq!(info.name, "model.embed_tokens.weight");
        assert_eq!(info.n_dims, 2);
        assert_eq!(info.dimensions.len(), 2);
    }

    #[test]
    fn test_gguf_tensor_info_clone() {
        let info = GGUFTensorInfo {
            name: "test_tensor".to_string(),
            n_dims: 1,
            dimensions: vec![1024],
            ggml_type: 8, // Q8_0
            offset: 4096,
        };
        let cloned = info.clone();
        assert_eq!(cloned.name, "test_tensor");
        assert_eq!(cloned.offset, 4096);
    }

    // ── GGUFLoader file-based tests ───────────────────────────────────────────

    #[test]
    fn test_gguf_loader_invalid_file() {
        // Attempting to open a non-existent file should return an error
        let result = GGUFLoader::new("/nonexistent/path/model.gguf");
        assert!(result.is_err(), "Expected error for nonexistent GGUF file");
    }

    #[test]
    fn test_gguf_loader_invalid_magic_bytes() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let path = dir.join("test_invalid_magic.gguf");
        {
            let mut f = std::fs::File::create(&path).expect("could not create temp file");
            // Write wrong magic
            f.write_all(b"BADS").expect("write failed");
            f.write_all(&[0u8; 24]).expect("write failed");
        }
        let result = GGUFLoader::new(&path);
        assert!(
            result.is_err(),
            "Expected error for invalid GGUF magic bytes"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_ggml_type_equality() {
        let t1 = GGMLType::F32;
        let t2 = GGMLType::F32;
        let t3 = GGMLType::Q4_0;
        assert_eq!(t1, t2);
        assert_ne!(t1, t3);
    }

    // ── End-to-end tests against real GGUF byte streams ──────────────────────

    use crate::weight_loading::test_support::{
        build_gguf, write_temp_file, GgufMetaValue, GgufTensor,
    };

    fn f32_tensor_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    struct TempGguf {
        path: std::path::PathBuf,
    }

    impl TempGguf {
        fn new(bytes: &[u8]) -> Self {
            Self {
                path: write_temp_file("gguf", "gguf", bytes),
            }
        }
    }

    impl Drop for TempGguf {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    #[test]
    fn gguf_loads_f32_tensor_with_exact_values_and_row_major_shape() {
        // ne = [3, 2] in ggml order is a [2, 3] row-major matrix.
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes = build_gguf(
            &[(
                "general.architecture".to_string(),
                GgufMetaValue::Str("llama".to_string()),
            )],
            &[GgufTensor {
                name: "token_embd.weight".to_string(),
                ggml_type: 0,
                dimensions: vec![3, 2],
                data: f32_tensor_bytes(&values),
            }],
            32,
        );
        let file = TempGguf::new(&bytes);
        let mut loader = GGUFLoader::new(&file.path).expect("GGUF fixture must load");

        let tensor = loader.load_tensor("token_embd.weight").expect("tensor must load");
        assert_eq!(tensor.shape(), vec![2, 3]);
        match tensor {
            Tensor::F32(arr) => {
                assert_eq!(arr.iter().copied().collect::<Vec<f32>>(), values);
            },
            other => panic!("expected an F32 tensor, got {other:?}"),
        }
    }

    #[test]
    fn gguf_metadata_arrays_are_read_and_keep_the_stream_in_sync() {
        // Regression: array values used to be returned as an empty array without
        // consuming their bytes, which desynchronised every later key and the
        // whole tensor-info table.
        let tokens = vec!["<s>".to_string(), "hello".to_string(), "world".to_string()];
        let scores = vec![-1.0f32, 0.5, 2.25];
        let bytes = build_gguf(
            &[
                (
                    "tokenizer.ggml.tokens".to_string(),
                    GgufMetaValue::StrArray(tokens.clone()),
                ),
                (
                    "tokenizer.ggml.scores".to_string(),
                    GgufMetaValue::F32Array(scores.clone()),
                ),
                (
                    "llama.attention.head_count".to_string(),
                    GgufMetaValue::U32(8),
                ),
            ],
            &[GgufTensor {
                name: "output.weight".to_string(),
                ggml_type: 0,
                dimensions: vec![4],
                data: f32_tensor_bytes(&[9.0, 8.0, 7.0, 6.0]),
            }],
            32,
        );
        let file = TempGguf::new(&bytes);
        let mut loader = GGUFLoader::new(&file.path).expect("GGUF fixture must load");

        let metadata = loader.get_metadata();
        let read_tokens = metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .expect("token array must be present");
        assert_eq!(read_tokens.len(), 3);
        assert_eq!(read_tokens[1].as_str(), Some("hello"));

        let read_scores = metadata
            .get("tokenizer.ggml.scores")
            .and_then(|v| v.as_array())
            .expect("score array must be present");
        assert_eq!(read_scores.len(), 3);
        let decoded: Vec<f32> =
            read_scores.iter().filter_map(|v| v.as_f64()).map(|v| v as f32).collect();
        assert_eq!(decoded, scores);

        // The key written after the arrays must still be readable, which only
        // holds if the array element bytes were consumed.
        assert_eq!(
            metadata.get("llama.attention.head_count").and_then(|v| v.as_u64()),
            Some(8)
        );

        // And the tensor table, which follows the metadata, must still be valid.
        let tensor = loader.load_tensor("output.weight").expect("tensor must load");
        match tensor {
            Tensor::F32(arr) => {
                assert_eq!(
                    arr.iter().copied().collect::<Vec<f32>>(),
                    vec![9.0, 8.0, 7.0, 6.0]
                );
            },
            other => panic!("expected an F32 tensor, got {other:?}"),
        }
    }

    #[test]
    fn gguf_honours_a_non_default_alignment() {
        let values: Vec<f32> = vec![0.5, -0.5, 1.5, -1.5];
        let bytes = build_gguf(
            &[(
                "general.architecture".to_string(),
                GgufMetaValue::Str("test".to_string()),
            )],
            &[GgufTensor {
                name: "w".to_string(),
                ggml_type: 0,
                dimensions: vec![4],
                data: f32_tensor_bytes(&values),
            }],
            64,
        );
        let file = TempGguf::new(&bytes);
        let mut loader = GGUFLoader::new(&file.path).expect("GGUF fixture must load");
        match loader.load_tensor("w").expect("tensor must load") {
            Tensor::F32(arr) => assert_eq!(arr.iter().copied().collect::<Vec<f32>>(), values),
            other => panic!("expected an F32 tensor, got {other:?}"),
        }
    }

    #[test]
    fn gguf_reads_a_q4_k_tensor_with_the_exact_super_block_size() {
        // 256 elements of Q4_K occupy exactly 144 bytes. The old size estimate
        // (0.5 bytes/element = 128) would truncate the read.
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_bits().to_le_bytes());
        block[4] = 2; // scale for the first 32-value group
        block[5] = 1; // scale for the second group
        block[16] = 0x36; // qs[0]: low nibble 6 -> element 0, high nibble 3 -> element 32

        let bytes = build_gguf(
            &[],
            &[GgufTensor {
                name: "blk.0.attn_q.weight".to_string(),
                ggml_type: 12, // Q4_K
                dimensions: vec![256],
                data: block,
            }],
            32,
        );
        let file = TempGguf::new(&bytes);
        let mut loader = GGUFLoader::new(&file.path).expect("GGUF fixture must load");

        let info = loader
            .tensor_info("blk.0.attn_q.weight")
            .expect("tensor info must resolve")
            .expect("tensor must exist");
        assert_eq!(info.size_bytes, 144);

        match loader.load_tensor("blk.0.attn_q.weight").expect("tensor must load") {
            Tensor::F32(arr) => {
                let values: Vec<f32> = arr.iter().copied().collect();
                assert_eq!(values.len(), 256);
                assert_eq!(values[0], 2.0 * 6.0);
                assert_eq!(values[32], 1.0 * 3.0);
            },
            other => panic!("expected an F32 tensor, got {other:?}"),
        }
    }

    #[test]
    fn gguf_rejects_a_quantization_it_cannot_decode() {
        // IQ2_XXS has no dequantizer here; the loader must say so instead of
        // returning normalised byte noise.
        let bytes = build_gguf(
            &[],
            &[GgufTensor {
                name: "blk.0.ffn_down.weight".to_string(),
                ggml_type: 16, // IQ2_XXS
                dimensions: vec![256],
                data: vec![0x5A; 66],
            }],
            32,
        );
        let file = TempGguf::new(&bytes);
        let mut loader = GGUFLoader::new(&file.path).expect("GGUF fixture must load");
        let err = loader
            .load_tensor("blk.0.ffn_down.weight")
            .expect_err("an unimplemented quantization must fail loudly");
        assert!(
            err.to_string().contains("Iq2Xxs"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn gguf_q4_0_round_trips_through_the_file_reader() {
        let mut block = Vec::new();
        block.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        block.push(0x9A); // low nibble 10 -> element 0, high nibble 9 -> element 16
        block.extend_from_slice(&[0x88u8; 15]);

        let bytes = build_gguf(
            &[],
            &[GgufTensor {
                name: "w".to_string(),
                ggml_type: 2, // Q4_0
                dimensions: vec![32],
                data: block,
            }],
            32,
        );
        let file = TempGguf::new(&bytes);
        let mut loader = GGUFLoader::new(&file.path).expect("GGUF fixture must load");
        match loader.load_tensor("w").expect("tensor must load") {
            Tensor::F32(arr) => {
                let values: Vec<f32> = arr.iter().copied().collect();
                assert_eq!(values[0], 2.0);
                assert_eq!(values[16], 1.0);
                assert_eq!(values[1], 0.0);
            },
            other => panic!("expected an F32 tensor, got {other:?}"),
        }
    }
}
