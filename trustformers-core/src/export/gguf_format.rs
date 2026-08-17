//! GGUF v3 container: types, block quantization codecs, writer and reader.
//!
//! This module is the single implementation of the GGUF binary format used by the
//! crate; [`super::gguf`] and [`super::gguf_enhanced`] are thin exporters on top of
//! it.
//!
//! GGUF is a *tensor container*: a header, a key/value metadata block, a tensor
//! directory and a data section. It carries no operation graph, so it can be
//! written faithfully from [`Model::named_tensors`](crate::traits::Model::named_tensors)
//! alone — which is exactly what the exporters do. Nothing here synthesises data.
//!
//! # Layout (little endian throughout)
//!
//! ```text
//! u32  magic  = "GGUF"
//! u32  version = 3
//! u64  tensor_count
//! u64  metadata_kv_count
//! [metadata_kv_count] { string key; u32 value_type; value }
//! [tensor_count]      { string name; u32 n_dims; u64 dims[n_dims]; u32 type; u64 offset }
//! padding to `general.alignment`
//! tensor data (each tensor starts at `tensor_data_start + offset`)
//! ```
//!
//! Tensor `offset` is **relative to the start of the data section**, not to the
//! start of the file, and every offset is a multiple of `general.alignment`.
//!
//! # Block quantization
//!
//! `Q8_0` and `Q4_0` follow llama.cpp's reference layout exactly, so files written
//! here are readable by llama.cpp and vice versa:
//!
//! * `Q8_0` — 32 elements per block, `f16 d` followed by 32 `i8`; `d = amax / 127`,
//!   `q = round(x / d)`, dequantised as `x = q * d` (34 bytes/block).
//! * `Q4_0` — 32 elements per block, `f16 d` followed by 16 packed bytes; `d` is
//!   `max / -8` where `max` is the element of largest magnitude *with its sign*,
//!   `q = min(15, floor(x/d + 8.5))`, and byte `j` holds `q[j] | (q[j + 16] << 4)`.
//!   Dequantised as `x = (q - 8) * d` (18 bytes/block).

use crate::errors::unsupported_operation;
use crate::tensor::Tensor;
use anyhow::{anyhow, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// `"GGUF"` in ASCII, little endian.
pub const GGUF_MAGIC: u32 = 0x4655_4747;

/// The GGUF version this module reads and writes.
pub const GGUF_VERSION: u32 = 3;

/// Default value of the `general.alignment` metadata key.
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

/// Upper bound on a single length prefix, to keep a corrupt file from making the
/// reader allocate wildly. 1 GiB is far beyond any legitimate GGUF string or array.
const MAX_LENGTH_PREFIX: u64 = 1 << 30;

/// GGUF metadata value type tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// Decode a wire tag.
    pub fn from_u32(tag: u32) -> Result<Self> {
        Ok(match tag {
            0 => GGUFValueType::UInt8,
            1 => GGUFValueType::Int8,
            2 => GGUFValueType::UInt16,
            3 => GGUFValueType::Int16,
            4 => GGUFValueType::UInt32,
            5 => GGUFValueType::Int32,
            6 => GGUFValueType::Float32,
            7 => GGUFValueType::Bool,
            8 => GGUFValueType::String,
            9 => GGUFValueType::Array,
            10 => GGUFValueType::UInt64,
            11 => GGUFValueType::Int64,
            12 => GGUFValueType::Float64,
            other => return Err(anyhow!("unknown GGUF metadata value type tag {other}")),
        })
    }
}

/// A GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
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
    /// The wire tag for this value.
    pub fn value_type(&self) -> GGUFValueType {
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

    /// Interpret the value as an unsigned integer, when it is one.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GGUFValue::UInt8(v) => Some(u64::from(*v)),
            GGUFValue::UInt16(v) => Some(u64::from(*v)),
            GGUFValue::UInt32(v) => Some(u64::from(*v)),
            GGUFValue::UInt64(v) => Some(*v),
            GGUFValue::Int8(v) if *v >= 0 => Some(*v as u64),
            GGUFValue::Int16(v) if *v >= 0 => Some(*v as u64),
            GGUFValue::Int32(v) if *v >= 0 => Some(*v as u64),
            GGUFValue::Int64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    /// Interpret the value as a string, when it is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GGUFValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Write the *payload* (the type tag is written separately by the caller).
    pub fn write_to_buffer<W: Write>(&self, writer: &mut W) -> Result<()> {
        match self {
            GGUFValue::UInt8(v) => writer.write_u8(*v)?,
            GGUFValue::Int8(v) => writer.write_i8(*v)?,
            GGUFValue::UInt16(v) => writer.write_u16::<LittleEndian>(*v)?,
            GGUFValue::Int16(v) => writer.write_i16::<LittleEndian>(*v)?,
            GGUFValue::UInt32(v) => writer.write_u32::<LittleEndian>(*v)?,
            GGUFValue::Int32(v) => writer.write_i32::<LittleEndian>(*v)?,
            GGUFValue::Float32(v) => writer.write_f32::<LittleEndian>(*v)?,
            GGUFValue::Bool(v) => writer.write_u8(u8::from(*v))?,
            GGUFValue::String(s) => write_gguf_string(writer, s)?,
            GGUFValue::Array(elem_type, values) => {
                writer.write_u32::<LittleEndian>(*elem_type as u32)?;
                writer.write_u64::<LittleEndian>(values.len() as u64)?;
                for value in values {
                    if value.value_type() != *elem_type {
                        return Err(anyhow!(
                            "GGUF array declared element type {:?} but contains {:?}",
                            elem_type,
                            value.value_type()
                        ));
                    }
                    value.write_to_buffer(writer)?;
                }
            },
            GGUFValue::UInt64(v) => writer.write_u64::<LittleEndian>(*v)?,
            GGUFValue::Int64(v) => writer.write_i64::<LittleEndian>(*v)?,
            GGUFValue::Float64(v) => writer.write_f64::<LittleEndian>(*v)?,
        }
        Ok(())
    }

    /// Read a value of the given type from `reader`.
    pub fn read_from<R: Read>(reader: &mut R, value_type: GGUFValueType) -> Result<Self> {
        Ok(match value_type {
            GGUFValueType::UInt8 => GGUFValue::UInt8(reader.read_u8()?),
            GGUFValueType::Int8 => GGUFValue::Int8(reader.read_i8()?),
            GGUFValueType::UInt16 => GGUFValue::UInt16(reader.read_u16::<LittleEndian>()?),
            GGUFValueType::Int16 => GGUFValue::Int16(reader.read_i16::<LittleEndian>()?),
            GGUFValueType::UInt32 => GGUFValue::UInt32(reader.read_u32::<LittleEndian>()?),
            GGUFValueType::Int32 => GGUFValue::Int32(reader.read_i32::<LittleEndian>()?),
            GGUFValueType::Float32 => GGUFValue::Float32(reader.read_f32::<LittleEndian>()?),
            GGUFValueType::Bool => GGUFValue::Bool(reader.read_u8()? != 0),
            GGUFValueType::String => GGUFValue::String(read_gguf_string(reader)?),
            GGUFValueType::UInt64 => GGUFValue::UInt64(reader.read_u64::<LittleEndian>()?),
            GGUFValueType::Int64 => GGUFValue::Int64(reader.read_i64::<LittleEndian>()?),
            GGUFValueType::Float64 => GGUFValue::Float64(reader.read_f64::<LittleEndian>()?),
            GGUFValueType::Array => {
                let elem_type = GGUFValueType::from_u32(reader.read_u32::<LittleEndian>()?)?;
                let count = reader.read_u64::<LittleEndian>()?;
                if count > MAX_LENGTH_PREFIX {
                    return Err(anyhow!("GGUF array length {count} is implausibly large"));
                }
                let mut values = Vec::with_capacity(count.min(4096) as usize);
                for _ in 0..count {
                    values.push(GGUFValue::read_from(reader, elem_type)?);
                }
                GGUFValue::Array(elem_type, values)
            },
        })
    }
}

/// GGUF tensor element types (the `ggml_type` enumeration).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// Decode a `ggml_type` wire tag.
    pub fn from_u32(tag: u32) -> Result<Self> {
        Ok(match tag {
            0 => GGUFTensorType::F32,
            1 => GGUFTensorType::F16,
            2 => GGUFTensorType::Q4_0,
            3 => GGUFTensorType::Q4_1,
            6 => GGUFTensorType::Q5_0,
            7 => GGUFTensorType::Q5_1,
            8 => GGUFTensorType::Q8_0,
            9 => GGUFTensorType::Q8_1,
            10 => GGUFTensorType::Q2K,
            11 => GGUFTensorType::Q3K,
            12 => GGUFTensorType::Q4K,
            13 => GGUFTensorType::Q5K,
            14 => GGUFTensorType::Q6K,
            15 => GGUFTensorType::Q8K,
            16 => GGUFTensorType::Iq2Xxs,
            17 => GGUFTensorType::Iq2Xs,
            18 => GGUFTensorType::Iq3Xxs,
            19 => GGUFTensorType::Iq1S,
            20 => GGUFTensorType::Iq4Nl,
            21 => GGUFTensorType::Iq3S,
            22 => GGUFTensorType::Iq2S,
            23 => GGUFTensorType::Iq4Xs,
            24 => GGUFTensorType::I8,
            25 => GGUFTensorType::I16,
            26 => GGUFTensorType::I32,
            27 => GGUFTensorType::I64,
            28 => GGUFTensorType::F64,
            29 => GGUFTensorType::Iq1M,
            other => return Err(anyhow!("unknown GGUF tensor type tag {other}")),
        })
    }

    /// Number of elements packed into one storage block.
    pub fn block_size(&self) -> usize {
        match self {
            GGUFTensorType::Q4_0
            | GGUFTensorType::Q4_1
            | GGUFTensorType::Q5_0
            | GGUFTensorType::Q5_1
            | GGUFTensorType::Q8_0
            | GGUFTensorType::Q8_1 => 32,
            GGUFTensorType::Q2K
            | GGUFTensorType::Q3K
            | GGUFTensorType::Q4K
            | GGUFTensorType::Q5K
            | GGUFTensorType::Q6K
            | GGUFTensorType::Q8K
            | GGUFTensorType::Iq2Xxs
            | GGUFTensorType::Iq2Xs
            | GGUFTensorType::Iq3Xxs
            | GGUFTensorType::Iq1S
            | GGUFTensorType::Iq4Nl
            | GGUFTensorType::Iq3S
            | GGUFTensorType::Iq2S
            | GGUFTensorType::Iq4Xs
            | GGUFTensorType::Iq1M => 256,
            _ => 1,
        }
    }

    /// Size in bytes of one storage block.
    pub fn type_size(&self) -> usize {
        match self {
            GGUFTensorType::F32 => 4,
            GGUFTensorType::F16 => 2,
            GGUFTensorType::F64 => 8,
            GGUFTensorType::I8 => 1,
            GGUFTensorType::I16 => 2,
            GGUFTensorType::I32 => 4,
            GGUFTensorType::I64 => 8,
            // f16 scale + 16 packed nibble bytes
            GGUFTensorType::Q4_0 => 18,
            // f16 scale + f16 min + 16 packed nibble bytes
            GGUFTensorType::Q4_1 => 20,
            GGUFTensorType::Q5_0 => 22,
            GGUFTensorType::Q5_1 => 24,
            // f16 scale + 32 i8
            GGUFTensorType::Q8_0 => 34,
            GGUFTensorType::Q8_1 => 36,
            GGUFTensorType::Q2K => 84,
            GGUFTensorType::Q3K => 110,
            GGUFTensorType::Q4K => 144,
            GGUFTensorType::Q5K => 176,
            GGUFTensorType::Q6K => 210,
            GGUFTensorType::Q8K => 292,
            GGUFTensorType::Iq2Xxs => 66,
            GGUFTensorType::Iq2Xs => 74,
            GGUFTensorType::Iq3Xxs => 98,
            GGUFTensorType::Iq1S => 50,
            GGUFTensorType::Iq4Nl => 18,
            GGUFTensorType::Iq3S => 110,
            GGUFTensorType::Iq2S => 82,
            GGUFTensorType::Iq4Xs => 136,
            GGUFTensorType::Iq1M => 56,
        }
    }

    /// Whether this type stores quantized blocks rather than plain elements.
    pub fn is_quantized(&self) -> bool {
        self.block_size() > 1
    }

    /// Whether this module can encode and decode the type.
    pub fn is_supported_codec(&self) -> bool {
        matches!(
            self,
            GGUFTensorType::F32
                | GGUFTensorType::F16
                | GGUFTensorType::F64
                | GGUFTensorType::Q8_0
                | GGUFTensorType::Q4_0
        )
    }

    /// Byte length of `element_count` elements stored in this type.
    pub fn data_size(&self, element_count: usize) -> Result<usize> {
        let block = self.block_size();
        if !element_count.is_multiple_of(block) {
            return Err(anyhow!(
                "tensor with {element_count} elements is not a multiple of the {:?} block size \
                 ({block}); GGUF cannot store it in this type",
                self
            ));
        }
        Ok((element_count / block) * self.type_size())
    }

    /// Map an export precision onto the tensor type used for weights.
    pub fn from_precision(precision: super::ExportPrecision) -> Self {
        match precision {
            super::ExportPrecision::FP32 => GGUFTensorType::F32,
            super::ExportPrecision::FP16 => GGUFTensorType::F16,
            super::ExportPrecision::INT8 => GGUFTensorType::Q8_0,
            super::ExportPrecision::INT4 => GGUFTensorType::Q4_0,
        }
    }

    /// The `general.file_type` value that corresponds to this weight type.
    pub fn file_type(&self) -> u32 {
        match self {
            GGUFTensorType::F32 => 0,
            GGUFTensorType::F16 => 1,
            GGUFTensorType::Q4_0 => 2,
            GGUFTensorType::Q4_1 => 3,
            GGUFTensorType::Q8_0 => 7,
            GGUFTensorType::Q5_0 => 8,
            GGUFTensorType::Q5_1 => 9,
            _ => 15,
        }
    }
}

/// Directory entry describing one tensor inside a GGUF file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GGUFTensorInfo {
    /// Tensor name as it appears in the file.
    pub name: String,
    /// Dimensions, in GGUF (fastest-varying first) order.
    pub dimensions: Vec<u64>,
    /// Element/block type.
    pub tensor_type: GGUFTensorType,
    /// Byte offset relative to the start of the data section.
    pub offset: u64,
}

impl GGUFTensorInfo {
    /// Total number of elements.
    pub fn element_count(&self) -> u64 {
        self.dimensions.iter().product::<u64>()
    }
}

/// Fixed header of a GGUF file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GGUFHeader {
    pub magic: u32,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

// ---------------------------------------------------------------------------
// Block quantization codecs
// ---------------------------------------------------------------------------

/// Encode `values` into the byte representation of `tensor_type`.
///
/// # Errors
///
/// Returns [`ErrorKind::UnsupportedOperation`](crate::errors::ErrorKind::UnsupportedOperation)
/// for types this crate has no encoder for, rather than writing something that only
/// looks like the type.
pub fn quantize(values: &[f32], tensor_type: GGUFTensorType) -> Result<Vec<u8>> {
    let block = tensor_type.block_size();
    if !values.len().is_multiple_of(block) {
        return Err(anyhow!(
            "cannot encode {} elements as {:?}: not a multiple of the block size {block}",
            values.len(),
            tensor_type
        ));
    }

    match tensor_type {
        GGUFTensorType::F32 => {
            let mut out = Vec::with_capacity(values.len() * 4);
            for &value in values {
                out.extend_from_slice(&value.to_le_bytes());
            }
            Ok(out)
        },
        GGUFTensorType::F16 => {
            let mut out = Vec::with_capacity(values.len() * 2);
            for &value in values {
                out.extend_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes());
            }
            Ok(out)
        },
        GGUFTensorType::F64 => {
            let mut out = Vec::with_capacity(values.len() * 8);
            for &value in values {
                out.extend_from_slice(&f64::from(value).to_le_bytes());
            }
            Ok(out)
        },
        GGUFTensorType::Q8_0 => Ok(quantize_q8_0(values)),
        GGUFTensorType::Q4_0 => Ok(quantize_q4_0(values)),
        other => Err(unsupported_operation(
            format!("GGUF {other:?} encoding"),
            "TrustformeRS implements the F32, F16, F64, Q8_0 and Q4_0 GGUF codecs; it will not \
             write bytes tagged with a quantization it does not actually produce",
        )
        .into()),
    }
}

/// Decode `element_count` elements of `tensor_type` from `bytes`.
pub fn dequantize(
    bytes: &[u8],
    tensor_type: GGUFTensorType,
    element_count: usize,
) -> Result<Vec<f32>> {
    let expected = tensor_type.data_size(element_count)?;
    if bytes.len() < expected {
        return Err(anyhow!(
            "GGUF tensor data truncated: {:?} needs {expected} bytes for {element_count} elements, \
             got {}",
            tensor_type,
            bytes.len()
        ));
    }
    let bytes = &bytes[..expected];

    match tensor_type {
        GGUFTensorType::F32 => Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        GGUFTensorType::F16 => Ok(bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()),
        GGUFTensorType::F64 => Ok(bytes
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect()),
        GGUFTensorType::Q8_0 => Ok(dequantize_q8_0(bytes)),
        GGUFTensorType::Q4_0 => Ok(dequantize_q4_0(bytes)),
        other => Err(unsupported_operation(
            format!("GGUF {other:?} decoding"),
            "TrustformeRS implements the F32, F16, F64, Q8_0 and Q4_0 GGUF codecs",
        )
        .into()),
    }
}

/// llama.cpp `quantize_row_q8_0_ref`: `f16 d` + 32 `i8` per 32-element block.
fn quantize_q8_0(values: &[f32]) -> Vec<u8> {
    const QK: usize = 32;
    let mut out = Vec::with_capacity((values.len() / QK) * 34);

    for block in values.chunks_exact(QK) {
        let amax = block.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        out.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        for &value in block {
            let q = (value * id).round().clamp(-127.0, 127.0) as i8;
            out.push(q as u8);
        }
    }

    out
}

fn dequantize_q8_0(bytes: &[u8]) -> Vec<f32> {
    const QK: usize = 32;
    let mut out = Vec::with_capacity((bytes.len() / 34) * QK);

    for block in bytes.chunks_exact(34) {
        let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        for &raw in &block[2..] {
            out.push((raw as i8) as f32 * d);
        }
    }

    out
}

/// llama.cpp `quantize_row_q4_0_ref`: `f16 d` + 16 packed nibble bytes per block.
///
/// `d` is derived from the *signed* element of largest magnitude, `d = max / -8`,
/// so the quantized codes land in `0..=15` around a bias of 8. Byte `j` packs
/// element `j` in the low nibble and element `j + 16` in the high nibble.
fn quantize_q4_0(values: &[f32]) -> Vec<u8> {
    const QK: usize = 32;
    let mut out = Vec::with_capacity((values.len() / QK) * 18);

    for block in values.chunks_exact(QK) {
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &value in block {
            if value.abs() > amax {
                amax = value.abs();
                max = value;
            }
        }

        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        out.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        for j in 0..QK / 2 {
            let x0 = block[j] * id;
            let x1 = block[j + QK / 2] * id;
            let q0 = ((x0 + 8.5) as i32).clamp(0, 15) as u8;
            let q1 = ((x1 + 8.5) as i32).clamp(0, 15) as u8;
            out.push(q0 | (q1 << 4));
        }
    }

    out
}

fn dequantize_q4_0(bytes: &[u8]) -> Vec<f32> {
    const QK: usize = 32;
    let mut out = vec![0.0f32; (bytes.len() / 18) * QK];

    for (block_index, block) in bytes.chunks_exact(18).enumerate() {
        let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let base = block_index * QK;
        for j in 0..QK / 2 {
            let packed = block[2 + j];
            let q0 = (packed & 0x0F) as i32 - 8;
            let q1 = ((packed >> 4) & 0x0F) as i32 - 8;
            out[base + j] = q0 as f32 * d;
            out[base + j + QK / 2] = q1 as f32 * d;
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

fn write_gguf_string<W: Write>(writer: &mut W, value: &str) -> Result<()> {
    writer.write_u64::<LittleEndian>(value.len() as u64)?;
    writer.write_all(value.as_bytes())?;
    Ok(())
}

fn read_gguf_string<R: Read>(reader: &mut R) -> Result<String> {
    let len = reader.read_u64::<LittleEndian>()?;
    if len > MAX_LENGTH_PREFIX {
        return Err(anyhow!("GGUF string length {len} is implausibly large"));
    }
    let mut buffer = vec![0u8; len as usize];
    reader.read_exact(&mut buffer)?;
    String::from_utf8(buffer).map_err(|e| anyhow!("GGUF string is not valid UTF-8: {e}"))
}

/// A complete GGUF file held in memory: metadata plus name/type/data per tensor.
#[derive(Debug, Clone, Default)]
pub struct GGUFPayload {
    /// Metadata key/value pairs. A `BTreeMap` so that files are byte-reproducible.
    pub metadata: BTreeMap<String, GGUFValue>,
    /// Tensors in the order they appear in the directory.
    pub tensors: Vec<(GGUFTensorInfo, Vec<u8>)>,
}

impl GGUFPayload {
    /// The alignment declared by `general.alignment`, defaulting to 32.
    pub fn alignment(&self) -> u64 {
        self.metadata
            .get("general.alignment")
            .and_then(GGUFValue::as_u64)
            .filter(|a| a.is_power_of_two() && *a > 0)
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT)
    }

    /// Look up a tensor's directory entry and raw bytes by name.
    pub fn tensor(&self, name: &str) -> Option<&(GGUFTensorInfo, Vec<u8>)> {
        self.tensors.iter().find(|(info, _)| info.name == name)
    }

    /// Decode a tensor to `f32` values by name.
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let (info, data) = self
            .tensor(name)
            .ok_or_else(|| anyhow!("GGUF file has no tensor named '{name}'"))?;
        dequantize(data, info.tensor_type, info.element_count() as usize)
    }
}

/// Build a [`GGUFPayload`] from real model tensors, encoding each into `tensor_type`.
///
/// Tensor dimensions are written in GGUF order (fastest-varying dimension first),
/// which is the reverse of the row-major shape carried by [`Tensor`].
pub fn payload_from_tensors(
    metadata: BTreeMap<String, GGUFValue>,
    tensors: &[(String, Tensor)],
    tensor_type: GGUFTensorType,
) -> Result<GGUFPayload> {
    let alignment = metadata
        .get("general.alignment")
        .and_then(GGUFValue::as_u64)
        .filter(|a| a.is_power_of_two() && *a > 0)
        .unwrap_or(GGUF_DEFAULT_ALIGNMENT);

    let mut payload = GGUFPayload {
        metadata,
        tensors: Vec::with_capacity(tensors.len()),
    };

    let mut offset = 0u64;
    for (name, tensor) in tensors {
        let values = tensor
            .to_vec_f32()
            .map_err(|e| anyhow!("tensor '{name}' cannot be read as f32 for GGUF export: {e}"))?;
        let data = quantize(&values, tensor_type).map_err(|e| anyhow!("tensor '{name}': {e}"))?;

        // GGUF stores dimensions with the fastest-varying axis first.
        let dimensions: Vec<u64> = tensor.shape().iter().rev().map(|&d| d as u64).collect();

        let info = GGUFTensorInfo {
            name: name.clone(),
            dimensions,
            tensor_type,
            offset,
        };
        offset = align_up(offset + data.len() as u64, alignment);
        payload.tensors.push((info, data));
    }

    Ok(payload)
}

/// Round `value` up to the next multiple of `alignment` (a power of two).
pub fn align_up(value: u64, alignment: u64) -> u64 {
    debug_assert!(alignment.is_power_of_two());
    value.div_ceil(alignment) * alignment
}

/// Serialise a payload into GGUF v3 bytes.
pub fn write_gguf_bytes(payload: &GGUFPayload) -> Result<Vec<u8>> {
    let alignment = payload.alignment();
    let mut out: Vec<u8> = Vec::new();

    out.write_u32::<LittleEndian>(GGUF_MAGIC)?;
    out.write_u32::<LittleEndian>(GGUF_VERSION)?;
    out.write_u64::<LittleEndian>(payload.tensors.len() as u64)?;
    out.write_u64::<LittleEndian>(payload.metadata.len() as u64)?;

    for (key, value) in &payload.metadata {
        write_gguf_string(&mut out, key)?;
        out.write_u32::<LittleEndian>(value.value_type() as u32)?;
        value.write_to_buffer(&mut out)?;
    }

    for (info, _) in &payload.tensors {
        write_gguf_string(&mut out, &info.name)?;
        out.write_u32::<LittleEndian>(info.dimensions.len() as u32)?;
        for &dim in &info.dimensions {
            out.write_u64::<LittleEndian>(dim)?;
        }
        out.write_u32::<LittleEndian>(info.tensor_type as u32)?;
        out.write_u64::<LittleEndian>(info.offset)?;
    }

    // Pad the directory so the data section starts on an alignment boundary.
    let data_start = align_up(out.len() as u64, alignment);
    out.resize(data_start as usize, 0);

    for (info, data) in &payload.tensors {
        let absolute = data_start + info.offset;
        if (absolute as usize) < out.len() {
            return Err(anyhow!(
                "GGUF tensor '{}' has offset {} which overlaps the previous tensor",
                info.name,
                info.offset
            ));
        }
        out.resize(absolute as usize, 0);
        out.extend_from_slice(data);
    }

    Ok(out)
}

/// Write a payload to `path` (the caller supplies the full file name).
pub fn write_gguf_file<P: AsRef<Path>>(path: P, payload: &GGUFPayload) -> Result<()> {
    let bytes = write_gguf_bytes(payload)?;
    let file = File::create(path.as_ref())?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&bytes)?;
    writer.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// Parse GGUF v3 bytes into a payload.
pub fn read_gguf_bytes(bytes: &[u8]) -> Result<GGUFPayload> {
    let mut cursor = std::io::Cursor::new(bytes);
    let header = read_header(&mut cursor)?;

    let mut metadata = BTreeMap::new();
    for _ in 0..header.metadata_kv_count {
        let key = read_gguf_string(&mut cursor)?;
        let value_type = GGUFValueType::from_u32(cursor.read_u32::<LittleEndian>()?)?;
        let value = GGUFValue::read_from(&mut cursor, value_type)?;
        metadata.insert(key, value);
    }

    let mut infos = Vec::with_capacity(header.tensor_count.min(4096) as usize);
    for _ in 0..header.tensor_count {
        let name = read_gguf_string(&mut cursor)?;
        let n_dims = cursor.read_u32::<LittleEndian>()?;
        if n_dims > 8 {
            return Err(anyhow!("GGUF tensor '{name}' declares {n_dims} dimensions"));
        }
        let mut dimensions = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dimensions.push(cursor.read_u64::<LittleEndian>()?);
        }
        let tensor_type = GGUFTensorType::from_u32(cursor.read_u32::<LittleEndian>()?)?;
        let offset = cursor.read_u64::<LittleEndian>()?;
        infos.push(GGUFTensorInfo {
            name,
            dimensions,
            tensor_type,
            offset,
        });
    }

    let alignment = metadata
        .get("general.alignment")
        .and_then(GGUFValue::as_u64)
        .filter(|a| a.is_power_of_two() && *a > 0)
        .unwrap_or(GGUF_DEFAULT_ALIGNMENT);
    let data_start = align_up(cursor.position(), alignment) as usize;

    let mut tensors = Vec::with_capacity(infos.len());
    for info in infos {
        let size = info.tensor_type.data_size(info.element_count() as usize)?;
        let begin = data_start
            .checked_add(info.offset as usize)
            .ok_or_else(|| anyhow!("GGUF tensor '{}' offset overflows", info.name))?;
        let end = begin
            .checked_add(size)
            .ok_or_else(|| anyhow!("GGUF tensor '{}' length overflows", info.name))?;
        if end > bytes.len() {
            return Err(anyhow!(
                "GGUF tensor '{}' extends to byte {end} but the file is {} bytes",
                info.name,
                bytes.len()
            ));
        }
        tensors.push((info, bytes[begin..end].to_vec()));
    }

    Ok(GGUFPayload { metadata, tensors })
}

/// Read and parse a GGUF file from disk.
pub fn read_gguf_file<P: AsRef<Path>>(path: P) -> Result<GGUFPayload> {
    let bytes = std::fs::read(path.as_ref())?;
    read_gguf_bytes(&bytes)
}

fn read_header<R: Read>(reader: &mut R) -> Result<GGUFHeader> {
    let magic = reader.read_u32::<LittleEndian>()?;
    if magic != GGUF_MAGIC {
        return Err(anyhow!(
            "not a GGUF file: magic is 0x{magic:08X}, expected 0x{GGUF_MAGIC:08X}"
        ));
    }
    let version = reader.read_u32::<LittleEndian>()?;
    if version != GGUF_VERSION {
        return Err(anyhow!(
            "unsupported GGUF version {version}; this build reads version {GGUF_VERSION}"
        ));
    }
    let tensor_count = reader.read_u64::<LittleEndian>()?;
    let metadata_kv_count = reader.read_u64::<LittleEndian>()?;
    if tensor_count > MAX_LENGTH_PREFIX || metadata_kv_count > MAX_LENGTH_PREFIX {
        return Err(anyhow!("GGUF header declares implausible counts"));
    }
    Ok(GGUFHeader {
        magic,
        version,
        tensor_count,
        metadata_kv_count,
    })
}

/// Read only the fixed header of a GGUF file, without loading the body.
pub fn read_gguf_header<P: AsRef<Path>>(path: P) -> Result<GGUFHeader> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);
    reader.seek(SeekFrom::Start(0))?;
    read_header(&mut reader)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_round_trip_recovers_values_within_quantization_error() {
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.375).collect();
        let encoded = quantize(&values, GGUFTensorType::Q8_0).expect("encode");
        assert_eq!(encoded.len(), 2 * 34, "two blocks of 34 bytes");

        let decoded = dequantize(&encoded, GGUFTensorType::Q8_0, values.len()).expect("decode");
        assert_eq!(decoded.len(), values.len());
        for (original, recovered) in values.iter().zip(decoded.iter()) {
            // Worst case is half a step of amax/127; amax here is 12.
            assert!(
                (original - recovered).abs() < 12.0 / 127.0,
                "q8_0 round trip: {original} -> {recovered}"
            );
        }
    }

    #[test]
    fn q4_0_round_trip_matches_llama_cpp_layout() {
        let values: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.5).collect();
        let encoded = quantize(&values, GGUFTensorType::Q4_0).expect("encode");
        assert_eq!(encoded.len(), 18, "one block: f16 scale + 16 packed bytes");

        // The element of largest magnitude is -8.0 (index 0), so d = -8 / -8 = 1.
        let d = half::f16::from_bits(u16::from_le_bytes([encoded[0], encoded[1]])).to_f32();
        assert!((d - 1.0).abs() < 1e-3, "scale should be 1.0, got {d}");

        // Byte j must pack element j (low nibble) and element j+16 (high nibble).
        let low = encoded[2] & 0x0F;
        let high = (encoded[2] >> 4) & 0x0F;
        assert_eq!(low, 0, "x[0] = -8 -> code 0");
        assert_eq!(high, 8, "x[16] = 0 -> code 8");

        let decoded = dequantize(&encoded, GGUFTensorType::Q4_0, values.len()).expect("decode");
        for (original, recovered) in values.iter().zip(decoded.iter()) {
            assert!(
                (original - recovered).abs() <= 0.5,
                "q4_0 round trip: {original} -> {recovered}"
            );
        }
    }

    #[test]
    fn unsupported_quantization_is_refused_rather_than_faked() {
        let values = vec![0.0f32; 256];
        let err = quantize(&values, GGUFTensorType::Q6K).expect_err("no Q6_K encoder exists");
        assert!(err.to_string().contains("Unsupported operation"), "{err}");
    }

    #[test]
    fn block_sizes_reject_ragged_tensors() {
        let values = vec![1.0f32; 40];
        let err = quantize(&values, GGUFTensorType::Q8_0).expect_err("40 is not a multiple of 32");
        assert!(err.to_string().contains("block size"), "{err}");
    }

    #[test]
    fn file_round_trip_preserves_metadata_and_tensors() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GGUFValue::String("test_transformer".to_string()),
        );
        metadata.insert("general.alignment".to_string(), GGUFValue::UInt32(32));
        metadata.insert(
            "test.tokens".to_string(),
            GGUFValue::Array(
                GGUFValueType::String,
                vec![
                    GGUFValue::String("a".to_string()),
                    GGUFValue::String("bb".to_string()),
                ],
            ),
        );

        let tensors = vec![
            (
                "alpha".to_string(),
                Tensor::from_vec((0..12).map(|i| i as f32).collect(), &[3, 4]).expect("tensor"),
            ),
            (
                "beta".to_string(),
                Tensor::from_vec(vec![-1.5, 2.5, 0.0], &[3]).expect("tensor"),
            ),
        ];

        let payload =
            payload_from_tensors(metadata.clone(), &tensors, GGUFTensorType::F32).expect("payload");
        let bytes = write_gguf_bytes(&payload).expect("serialize");

        assert_eq!(&bytes[0..4], b"GGUF");
        let parsed = read_gguf_bytes(&bytes).expect("parse");
        assert_eq!(parsed.metadata, metadata);
        assert_eq!(parsed.tensors.len(), 2);

        // GGUF stores dimensions fastest-varying first, so [3, 4] becomes [4, 3].
        assert_eq!(parsed.tensors[0].0.dimensions, vec![4, 3]);
        assert_eq!(
            parsed.tensor_f32("alpha").expect("alpha"),
            (0..12).map(|i| i as f32).collect::<Vec<_>>()
        );
        assert_eq!(
            parsed.tensor_f32("beta").expect("beta"),
            vec![-1.5, 2.5, 0.0]
        );

        // Every tensor must start on the declared alignment boundary.
        for (info, _) in &parsed.tensors {
            assert_eq!(info.offset % 32, 0, "tensor '{}' misaligned", info.name);
        }
    }

    #[test]
    fn reader_rejects_non_gguf_bytes() {
        let err = read_gguf_bytes(b"this is not a gguf file at all!!").expect_err("must reject");
        assert!(err.to_string().contains("not a GGUF file"), "{err}");
    }

    #[test]
    fn different_weights_produce_different_bytes() {
        let metadata = BTreeMap::new();
        let make = |offset: f32| {
            vec![(
                "w".to_string(),
                Tensor::from_vec((0..8).map(|i| i as f32 + offset).collect(), &[8])
                    .expect("tensor"),
            )]
        };

        let a = write_gguf_bytes(
            &payload_from_tensors(metadata.clone(), &make(0.0), GGUFTensorType::F32)
                .expect("payload"),
        )
        .expect("bytes");
        let b = write_gguf_bytes(
            &payload_from_tensors(metadata, &make(1.0), GGUFTensorType::F32).expect("payload"),
        )
        .expect("bytes");

        assert_eq!(a.len(), b.len());
        assert_ne!(a, b, "different weights must give different files");
    }
}
