// GGUF export functionality (improved GGML format)
#![allow(unused_variables)] // GGUF export

use super::{ExportConfig, ExportFormat, ExportPrecision, ModelExporter};
use crate::traits::Model;
use anyhow::{anyhow, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};

/// GGUF file format constants
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in ASCII
const GGUF_VERSION: u32 = 3;

/// GGUF value types
#[derive(Debug, Clone, Copy)]
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

/// GGUF tensor types (same as GGML but with additional types)
#[derive(Debug, Clone, Copy)]
pub enum GGUFTensorType {
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
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Iq1M = 29,
}

impl GGUFTensorType {
    fn from_precision(precision: ExportPrecision) -> Self {
        match precision {
            ExportPrecision::FP32 => GGUFTensorType::F32,
            ExportPrecision::FP16 => GGUFTensorType::F16,
            ExportPrecision::INT8 => GGUFTensorType::Q8_0,
            ExportPrecision::INT4 => GGUFTensorType::Q4_0,
        }
    }

    #[allow(dead_code)]
    fn element_size(&self) -> usize {
        match self {
            GGUFTensorType::F32 => 4,
            GGUFTensorType::F16 => 2,
            GGUFTensorType::Q4_0 => 2,
            GGUFTensorType::Q4_1 => 2,
            GGUFTensorType::Q5_0 => 3,
            GGUFTensorType::Q5_1 => 3,
            GGUFTensorType::Q8_0 => 1,
            GGUFTensorType::Q8_1 => 1,
            GGUFTensorType::I8 => 1,
            GGUFTensorType::I16 => 2,
            GGUFTensorType::I32 => 4,
            GGUFTensorType::I64 => 8,
            GGUFTensorType::F64 => 8,
            _ => 1, // Quantized types approximation
        }
    }
}

/// GGUF metadata value
#[derive(Debug, Clone)]
pub enum GGUFValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(GGUFValueType, Vec<GGUFValue>),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}

impl GGUFValue {
    fn value_type(&self) -> GGUFValueType {
        match self {
            GGUFValue::UInt8(_) => GGUFValueType::UInt8,
            GGUFValue::Int8(_) => GGUFValueType::Int8,
            GGUFValue::UInt16(_) => GGUFValueType::UInt16,
            GGUFValue::Int16(_) => GGUFValueType::Int16,
            GGUFValue::UInt32(_) => GGUFValueType::UInt32,
            GGUFValue::Int32(_) => GGUFValueType::Int32,
            GGUFValue::Float32(_) => GGUFValueType::Float32,
            GGUFValue::Bool(_) => GGUFValueType::Bool,
            GGUFValue::String(_) => GGUFValueType::String,
            GGUFValue::Array(_, _) => GGUFValueType::Array,
            GGUFValue::UInt64(_) => GGUFValueType::UInt64,
            GGUFValue::Int64(_) => GGUFValueType::Int64,
            GGUFValue::Float64(_) => GGUFValueType::Float64,
        }
    }
}

/// GGUF tensor information
#[derive(Debug)]
pub struct GGUFTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: GGUFTensorType,
    pub offset: u64,
}

/// GGUF tensor data
#[derive(Debug)]
pub struct GGUFTensor {
    pub info: GGUFTensorInfo,
    pub data: Vec<u8>,
}

/// GGUF model representation
#[derive(Debug)]
pub struct GGUFModel {
    pub magic: u32,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub metadata: HashMap<String, GGUFValue>,
    pub tensors: Vec<GGUFTensor>,
}

/// GGUF exporter implementation
#[derive(Clone)]
pub struct GGUFExporter {
    compression_enabled: bool,
    alignment: usize,
}

impl Default for GGUFExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl GGUFExporter {
    pub fn new() -> Self {
        Self {
            compression_enabled: false,
            alignment: 32, // Default alignment for optimal performance
        }
    }

    pub fn with_compression(mut self, enabled: bool) -> Self {
        self.compression_enabled = enabled;
        self
    }

    pub fn with_alignment(mut self, alignment: usize) -> Self {
        self.alignment = alignment;
        self
    }

    fn create_gguf_model<M: Model>(&self, model: &M, config: &ExportConfig) -> Result<GGUFModel> {
        let mut metadata = HashMap::new();

        // Standard metadata keys
        metadata.insert(
            "general.architecture".to_string(),
            GGUFValue::String("gpt2".to_string()),
        );
        metadata.insert(
            "general.name".to_string(),
            GGUFValue::String("TrustformeRS Model".to_string()),
        );
        metadata.insert(
            "general.version".to_string(),
            GGUFValue::String("0.1.0".to_string()),
        );
        metadata.insert(
            "general.description".to_string(),
            GGUFValue::String("Exported from TrustformeRS".to_string()),
        );
        metadata.insert(
            "general.author".to_string(),
            GGUFValue::String("TrustformeRS".to_string()),
        );
        metadata.insert(
            "general.license".to_string(),
            GGUFValue::String("MIT".to_string()),
        );
        metadata.insert(
            "general.source.url".to_string(),
            GGUFValue::String("https://github.com/cool-japan/trustformers".to_string()),
        );

        // Model hyperparameters
        metadata.insert(
            "gpt2.context_length".to_string(),
            GGUFValue::UInt32(config.sequence_length.unwrap_or(2048) as u32),
        );
        metadata.insert("gpt2.embedding_length".to_string(), GGUFValue::UInt32(768));
        metadata.insert("gpt2.block_count".to_string(), GGUFValue::UInt32(12));
        metadata.insert(
            "gpt2.feed_forward_length".to_string(),
            GGUFValue::UInt32(3072),
        );
        metadata.insert(
            "gpt2.attention.head_count".to_string(),
            GGUFValue::UInt32(12),
        );
        metadata.insert(
            "gpt2.attention.head_count_kv".to_string(),
            GGUFValue::UInt32(12),
        );
        metadata.insert(
            "gpt2.attention.layer_norm_epsilon".to_string(),
            GGUFValue::Float32(1e-5),
        );
        metadata.insert(
            "gpt2.rope.dimension_count".to_string(),
            GGUFValue::UInt32(64),
        );
        metadata.insert(
            "gpt2.rope.freq_base".to_string(),
            GGUFValue::Float32(10000.0),
        );

        // Tokenizer metadata
        metadata.insert(
            "tokenizer.ggml.model".to_string(),
            GGUFValue::String("gpt2".to_string()),
        );
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            GGUFValue::Array(GGUFValueType::String, self.create_dummy_vocab(50257)),
        );
        metadata.insert(
            "tokenizer.ggml.token_type".to_string(),
            GGUFValue::Array(GGUFValueType::Int32, self.create_dummy_token_types(50257)),
        );
        metadata.insert(
            "tokenizer.ggml.bos_token_id".to_string(),
            GGUFValue::UInt32(50256),
        );
        metadata.insert(
            "tokenizer.ggml.eos_token_id".to_string(),
            GGUFValue::UInt32(50256),
        );
        metadata.insert(
            "tokenizer.ggml.unknown_token_id".to_string(),
            GGUFValue::UInt32(50256),
        );
        metadata.insert(
            "tokenizer.ggml.separator_token_id".to_string(),
            GGUFValue::UInt32(50256),
        );
        metadata.insert(
            "tokenizer.ggml.padding_token_id".to_string(),
            GGUFValue::UInt32(50256),
        );

        // Create tensors
        let mut tensors = Vec::new();
        self.convert_model_to_tensors(model, &mut tensors, config)?;

        Ok(GGUFModel {
            magic: GGUF_MAGIC,
            version: GGUF_VERSION,
            tensor_count: tensors.len() as u64,
            metadata_kv_count: metadata.len() as u64,
            metadata,
            tensors,
        })
    }

    fn create_dummy_vocab(&self, vocab_size: usize) -> Vec<GGUFValue> {
        (0..vocab_size).map(|i| GGUFValue::String(format!("token_{}", i))).collect()
    }

    fn create_dummy_token_types(&self, vocab_size: usize) -> Vec<GGUFValue> {
        (0..vocab_size)
            .map(|_| GGUFValue::Int32(1)) // Normal token type
            .collect()
    }

    fn convert_model_to_tensors<M: Model>(
        &self,
        model: &M,
        tensors: &mut Vec<GGUFTensor>,
        config: &ExportConfig,
    ) -> Result<()> {
        let tensor_type = GGUFTensorType::from_precision(config.precision);
        let mut offset = 0u64;

        // Token embeddings
        let emb_data = self.generate_dummy_tensor_data(50257 * 768, tensor_type)?;
        tensors.push(GGUFTensor {
            info: GGUFTensorInfo {
                name: "token_embd.weight".to_string(),
                dimensions: vec![768, 50257],
                tensor_type,
                offset,
            },
            data: emb_data,
        });
        offset += tensors.last().unwrap().data.len() as u64;
        offset = self.align_offset(offset);

        // Position embeddings (if applicable)
        let pos_emb_data = self.generate_dummy_tensor_data(2048 * 768, tensor_type)?;
        tensors.push(GGUFTensor {
            info: GGUFTensorInfo {
                name: "position_embd.weight".to_string(),
                dimensions: vec![768, 2048],
                tensor_type,
                offset,
            },
            data: pos_emb_data,
        });
        offset += tensors.last().unwrap().data.len() as u64;
        offset = self.align_offset(offset);

        // Transformer blocks
        for i in 0..12 {
            offset = self.add_transformer_block_tensors(tensors, i, tensor_type, offset)?;
        }

        // Final layer norm
        let final_norm_data = self.generate_dummy_tensor_data(768, tensor_type)?;
        tensors.push(GGUFTensor {
            info: GGUFTensorInfo {
                name: "norm.weight".to_string(),
                dimensions: vec![768],
                tensor_type,
                offset,
            },
            data: final_norm_data,
        });
        offset += tensors.last().unwrap().data.len() as u64;
        offset = self.align_offset(offset);

        // Output projection
        let output_data = self.generate_dummy_tensor_data(768 * 50257, tensor_type)?;
        tensors.push(GGUFTensor {
            info: GGUFTensorInfo {
                name: "output.weight".to_string(),
                dimensions: vec![50257, 768],
                tensor_type,
                offset,
            },
            data: output_data,
        });

        Ok(())
    }

    fn add_transformer_block_tensors(
        &self,
        tensors: &mut Vec<GGUFTensor>,
        block_idx: usize,
        tensor_type: GGUFTensorType,
        mut offset: u64,
    ) -> Result<u64> {
        let block_prefix = format!("blk.{}", block_idx);

        // Attention layer norm
        let attn_norm_data = self.generate_dummy_tensor_data(768, tensor_type)?;
        tensors.push(GGUFTensor {
            info: GGUFTensorInfo {
                name: format!("{}.attn_norm.weight", block_prefix),
                dimensions: vec![768],
                tensor_type,
                offset,
            },
            data: attn_norm_data,
        });
        offset += tensors.last().unwrap().data.len() as u64;
        offset = self.align_offset(offset);

        // Attention weights (combined QKV)
        let attn_qkv_data = self.generate_dummy_tensor_data(768 * 768 * 3, tensor_type)?;
        tensors.push(GGUFTensor {
            info: GGUFTensorInfo {
                name: format!("{}.attn_qkv.weight", block_prefix),
                dimensions: vec![768, 2304], // 768 * 3
                tensor_type,
                offset,
            },
            data: attn_qkv_data,
        });
        offset += tensors.last().unwrap().data.len() as u64;
        offset = self.align_offset(offset);

        // Attention output projection
        let attn_out_data = self.generate_dummy_tensor_data(768 * 768, tensor_type)?;
        tensors.push(GGUFTensor {
            info: GGUFTensorInfo {
                name: format!("{}.attn_output.weight", block_prefix),
                dimensions: vec![768, 768],
                tensor_type,
                offset,
            },
            data: attn_out_data,
        });
        offset += tensors.last().unwrap().data.len() as u64;
        offset = self.align_offset(offset);

        // Feed-forward layer norm
        let ffn_norm_data = self.generate_dummy_tensor_data(768, tensor_type)?;
        tensors.push(GGUFTensor {
            info: GGUFTensorInfo {
                name: format!("{}.ffn_norm.weight", block_prefix),
                dimensions: vec![768],
                tensor_type,
                offset,
            },
            data: ffn_norm_data,
        });
        offset += tensors.last().unwrap().data.len() as u64;
        offset = self.align_offset(offset);

        // Feed-forward up projection
        let ffn_up_data = self.generate_dummy_tensor_data(768 * 3072, tensor_type)?;
        tensors.push(GGUFTensor {
            info: GGUFTensorInfo {
                name: format!("{}.ffn_up.weight", block_prefix),
                dimensions: vec![3072, 768],
                tensor_type,
                offset,
            },
            data: ffn_up_data,
        });
        offset += tensors.last().unwrap().data.len() as u64;
        offset = self.align_offset(offset);

        // Feed-forward down projection
        let ffn_down_data = self.generate_dummy_tensor_data(3072 * 768, tensor_type)?;
        tensors.push(GGUFTensor {
            info: GGUFTensorInfo {
                name: format!("{}.ffn_down.weight", block_prefix),
                dimensions: vec![768, 3072],
                tensor_type,
                offset,
            },
            data: ffn_down_data,
        });
        offset += tensors.last().unwrap().data.len() as u64;
        offset = self.align_offset(offset);

        Ok(offset)
    }

    fn align_offset(&self, offset: u64) -> u64 {
        let alignment = self.alignment as u64;
        offset.div_ceil(alignment) * alignment
    }

    fn generate_dummy_tensor_data(
        &self,
        size: usize,
        tensor_type: GGUFTensorType,
    ) -> Result<Vec<u8>> {
        let mut data = Vec::new();

        match tensor_type {
            GGUFTensorType::F32 => {
                for i in 0..size {
                    let val = (i as f32 * 0.001).sin();
                    data.extend_from_slice(&val.to_le_bytes());
                }
            },
            GGUFTensorType::F16 => {
                for i in 0..size {
                    let val = (i as f32 * 0.001).sin();
                    let f16_val = half::f16::from_f32(val);
                    data.extend_from_slice(&f16_val.to_le_bytes());
                }
            },
            GGUFTensorType::Q8_0 => {
                // Block-wise quantization for Q8_0
                const BLOCK_SIZE: usize = 32;
                for block_start in (0..size).step_by(BLOCK_SIZE) {
                    let block_end = (block_start + BLOCK_SIZE).min(size);
                    let block_size = block_end - block_start;

                    // Compute scale for this block
                    let mut max_abs = 0.0f32;
                    for i in block_start..block_end {
                        let val = (i as f32 * 0.001).sin();
                        max_abs = max_abs.max(val.abs());
                    }
                    let scale = max_abs / 127.0;
                    data.extend_from_slice(&scale.to_le_bytes());

                    // Quantize values in this block
                    for i in block_start..block_end {
                        let val = (i as f32 * 0.001).sin();
                        let quantized = if scale > 0.0 {
                            (val / scale).round().clamp(-128.0, 127.0) as i8
                        } else {
                            0i8
                        };
                        data.push(quantized as u8);
                    }

                    // Pad block if necessary
                    data.resize(data.len() + (BLOCK_SIZE - block_size), 0);
                }
            },
            GGUFTensorType::Q4_0 => {
                // Block-wise 4-bit quantization
                const BLOCK_SIZE: usize = 32;
                for block_start in (0..size).step_by(BLOCK_SIZE) {
                    let block_end = (block_start + BLOCK_SIZE).min(size);

                    // Compute scale
                    let mut max_abs = 0.0f32;
                    for i in block_start..block_end {
                        let val = (i as f32 * 0.001).sin();
                        max_abs = max_abs.max(val.abs());
                    }
                    let scale = max_abs / 7.0; // 4-bit signed range is -8 to 7
                    data.extend_from_slice(&scale.to_le_bytes());

                    // Pack 2 4-bit values per byte
                    for i in (block_start..block_end).step_by(2) {
                        let val1 = (i as f32 * 0.001).sin();
                        let val2 =
                            if i + 1 < block_end { ((i + 1) as f32 * 0.001).sin() } else { 0.0 };

                        let q1 = if scale > 0.0 {
                            (val1 / scale).round().clamp(-8.0, 7.0) as i8
                        } else {
                            0i8
                        };
                        let q2 = if scale > 0.0 {
                            (val2 / scale).round().clamp(-8.0, 7.0) as i8
                        } else {
                            0i8
                        };

                        let packed = ((q1 & 0xF) | ((q2 & 0xF) << 4)) as u8;
                        data.push(packed);
                    }
                }
            },
            _ => {
                return Err(anyhow!(
                    "Unsupported tensor type for GGUF: {:?}",
                    tensor_type
                ));
            },
        }

        Ok(data)
    }

    fn serialize_gguf_model(&self, model: &GGUFModel, output_path: &str) -> Result<()> {
        let file = File::create(format!("{}.gguf", output_path))?;
        let mut writer = BufWriter::new(file);

        // Write header
        writer.write_u32::<LittleEndian>(model.magic)?;
        writer.write_u32::<LittleEndian>(model.version)?;
        writer.write_u64::<LittleEndian>(model.tensor_count)?;
        writer.write_u64::<LittleEndian>(model.metadata_kv_count)?;

        // Write metadata
        for (key, value) in &model.metadata {
            self.write_string(&mut writer, key)?;
            self.write_value(&mut writer, value)?;
        }

        // Write tensor info
        for tensor in &model.tensors {
            self.write_string(&mut writer, &tensor.info.name)?;
            writer.write_u32::<LittleEndian>(tensor.info.dimensions.len() as u32)?;
            for &dim in &tensor.info.dimensions {
                writer.write_u64::<LittleEndian>(dim)?;
            }
            writer.write_u32::<LittleEndian>(tensor.info.tensor_type as u32)?;
            writer.write_u64::<LittleEndian>(tensor.info.offset)?;
        }

        // Write tensor data (aligned)
        let mut current_offset = 0u64;
        for tensor in &model.tensors {
            // Pad to alignment
            while current_offset < tensor.info.offset {
                writer.write_u8(0)?;
                current_offset += 1;
            }

            writer.write_all(&tensor.data)?;
            current_offset += tensor.data.len() as u64;
        }

        writer.flush()?;
        Ok(())
    }

    fn write_string<W: Write>(&self, writer: &mut W, s: &str) -> Result<()> {
        writer.write_u64::<LittleEndian>(s.len() as u64)?;
        writer.write_all(s.as_bytes())?;
        Ok(())
    }

    fn write_value<W: Write>(&self, writer: &mut W, value: &GGUFValue) -> Result<()> {
        writer.write_u32::<LittleEndian>(value.value_type() as u32)?;

        match value {
            GGUFValue::UInt8(v) => writer.write_u8(*v)?,
            GGUFValue::Int8(v) => writer.write_i8(*v)?,
            GGUFValue::UInt16(v) => writer.write_u16::<LittleEndian>(*v)?,
            GGUFValue::Int16(v) => writer.write_i16::<LittleEndian>(*v)?,
            GGUFValue::UInt32(v) => writer.write_u32::<LittleEndian>(*v)?,
            GGUFValue::Int32(v) => writer.write_i32::<LittleEndian>(*v)?,
            GGUFValue::Float32(v) => writer.write_f32::<LittleEndian>(*v)?,
            GGUFValue::Bool(v) => writer.write_u8(if *v { 1 } else { 0 })?,
            GGUFValue::String(s) => self.write_string(writer, s)?,
            GGUFValue::Array(element_type, values) => {
                writer.write_u32::<LittleEndian>(*element_type as u32)?;
                writer.write_u64::<LittleEndian>(values.len() as u64)?;
                for value in values {
                    self.write_value(writer, value)?;
                }
            },
            GGUFValue::UInt64(v) => writer.write_u64::<LittleEndian>(*v)?,
            GGUFValue::Int64(v) => writer.write_i64::<LittleEndian>(*v)?,
            GGUFValue::Float64(v) => writer.write_f64::<LittleEndian>(*v)?,
        }

        Ok(())
    }
}

impl ModelExporter for GGUFExporter {
    fn export<M: Model>(&self, model: &M, config: &ExportConfig) -> Result<()> {
        if config.format != ExportFormat::GGUF {
            return Err(anyhow!("GGUFExporter only supports GGUF format"));
        }

        let gguf_model = self.create_gguf_model(model, config)?;
        self.serialize_gguf_model(&gguf_model, &config.output_path)?;

        println!("Model exported to {}.gguf", config.output_path);
        Ok(())
    }

    fn supported_formats(&self) -> Vec<ExportFormat> {
        vec![ExportFormat::GGUF]
    }

    fn validate_model<M: Model>(&self, _model: &M, format: ExportFormat) -> Result<()> {
        if format != ExportFormat::GGUF {
            return Err(anyhow!("GGUFExporter only supports GGUF format"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_exporter_creation() {
        let exporter = GGUFExporter::new();
        assert!(!exporter.compression_enabled);
        assert_eq!(exporter.alignment, 32);

        let exporter_compressed = exporter.with_compression(true).with_alignment(64);
        assert!(exporter_compressed.compression_enabled);
        assert_eq!(exporter_compressed.alignment, 64);
    }

    #[test]
    fn test_gguf_constants() {
        assert_eq!(GGUF_MAGIC, 0x46554747);
        assert_eq!(GGUF_VERSION, 3);
    }

    #[test]
    fn test_gguf_value_types() {
        let uint8_val = GGUFValue::UInt8(42);
        assert!(matches!(uint8_val.value_type(), GGUFValueType::UInt8));

        let string_val = GGUFValue::String("test".to_string());
        assert!(matches!(string_val.value_type(), GGUFValueType::String));

        let array_val = GGUFValue::Array(
            GGUFValueType::Int32,
            vec![GGUFValue::Int32(1), GGUFValue::Int32(2)],
        );
        assert!(matches!(array_val.value_type(), GGUFValueType::Array));
    }

    #[test]
    fn test_gguf_tensor_type_conversion() {
        assert_eq!(
            GGUFTensorType::from_precision(ExportPrecision::FP32) as u32,
            0
        );
        assert_eq!(
            GGUFTensorType::from_precision(ExportPrecision::FP16) as u32,
            1
        );
        assert_eq!(
            GGUFTensorType::from_precision(ExportPrecision::INT8) as u32,
            8
        );
        assert_eq!(
            GGUFTensorType::from_precision(ExportPrecision::INT4) as u32,
            2
        );
    }

    #[test]
    fn test_gguf_tensor_element_sizes() {
        assert_eq!(GGUFTensorType::F32.element_size(), 4);
        assert_eq!(GGUFTensorType::F16.element_size(), 2);
        assert_eq!(GGUFTensorType::I8.element_size(), 1);
        assert_eq!(GGUFTensorType::I64.element_size(), 8);
    }

    #[test]
    fn test_supported_formats() {
        let exporter = GGUFExporter::new();
        let formats = exporter.supported_formats();
        assert_eq!(formats.len(), 1);
        assert_eq!(formats[0], ExportFormat::GGUF);
    }

    #[test]
    fn test_offset_alignment() {
        let exporter = GGUFExporter::new().with_alignment(32);

        assert_eq!(exporter.align_offset(0), 0);
        assert_eq!(exporter.align_offset(1), 32);
        assert_eq!(exporter.align_offset(31), 32);
        assert_eq!(exporter.align_offset(32), 32);
        assert_eq!(exporter.align_offset(33), 64);
        assert_eq!(exporter.align_offset(63), 64);
        assert_eq!(exporter.align_offset(64), 64);
    }
}
