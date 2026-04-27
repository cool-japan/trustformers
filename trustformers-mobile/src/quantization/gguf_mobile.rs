//! GGUF format support for mobile deployment.
//!
//! GGUF is GGML's unified format supporting various quantization levels.
//! This module provides a pure-Rust reader/parser for GGUF files that is
//! suitable for use on constrained mobile devices.

use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::Path;

use serde::{Deserialize, Serialize};
use trustformers_core::errors::{Result, TrustformersError};

use crate::optimization::advanced_quantization::MobilePrecision;

// ─── Constants ────────────────────────────────────────────────────────────────

/// GGUF magic bytes: ASCII "GGUF" = 0x47_47_55_46 (little-endian u32: 0x4655_4747).
const GGUF_MAGIC: u32 = 0x4655_4747;

// ─── Quantization type ────────────────────────────────────────────────────────

/// GGUF quantization type as defined in the GGML spec.
// The _K suffix is part of the official spec naming and must be preserved.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GgufQuantType {
    F32,
    F16,
    /// 4-bit, unsigned, grouped — simplest INT4 scheme.
    Q4_0,
    /// 4-bit, unsigned, grouped + per-group min-value.
    Q4_1,
    Q5_0,
    Q5_1,
    /// 8-bit, grouped.
    Q8_0,
    /// 4-bit K-quant (super-blocks, mixed precision).
    Q4_K,
    /// 5-bit K-quant.
    Q5_K,
    /// 6-bit K-quant (near FP16 quality at ~6 bits/weight).
    Q6_K,
}

impl GgufQuantType {
    /// Average number of bits per weight.
    pub fn bits_per_weight(self) -> f32 {
        match self {
            GgufQuantType::F32 => 32.0,
            GgufQuantType::F16 => 16.0,
            GgufQuantType::Q4_0 | GgufQuantType::Q4_1 | GgufQuantType::Q4_K => 4.0,
            GgufQuantType::Q5_0 | GgufQuantType::Q5_1 | GgufQuantType::Q5_K => 5.0,
            GgufQuantType::Q6_K => 6.0,
            GgufQuantType::Q8_0 => 8.0,
        }
    }

    /// Compression factor relative to FP32.
    pub fn compression_vs_f32(self) -> f32 {
        32.0 / self.bits_per_weight()
    }

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            GgufQuantType::F32 => "F32",
            GgufQuantType::F16 => "F16",
            GgufQuantType::Q4_0 => "Q4_0",
            GgufQuantType::Q4_1 => "Q4_1",
            GgufQuantType::Q5_0 => "Q5_0",
            GgufQuantType::Q5_1 => "Q5_1",
            GgufQuantType::Q8_0 => "Q8_0",
            GgufQuantType::Q4_K => "Q4_K",
            GgufQuantType::Q5_K => "Q5_K",
            GgufQuantType::Q6_K => "Q6_K",
        }
    }

    /// Parse from the on-disk integer discriminant.
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(GgufQuantType::F32),
            1 => Some(GgufQuantType::F16),
            2 => Some(GgufQuantType::Q4_0),
            3 => Some(GgufQuantType::Q4_1),
            6 => Some(GgufQuantType::Q5_0),
            7 => Some(GgufQuantType::Q5_1),
            8 => Some(GgufQuantType::Q8_0),
            12 => Some(GgufQuantType::Q4_K),
            13 => Some(GgufQuantType::Q5_K),
            14 => Some(GgufQuantType::Q6_K),
            _ => None,
        }
    }
}

// ─── Metadata value ───────────────────────────────────────────────────────────

/// A metadata value stored in a GGUF file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GgufMetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufMetadataValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufMetadataValue {
    /// Type discriminant as on-disk u32.
    fn type_id(v: u32) -> bool {
        v <= 12
    }

    /// Parse a single metadata value given its type discriminant.
    fn parse(type_id: u32, cur: &mut Cursor<&[u8]>) -> Result<Self> {
        match type_id {
            0 => Ok(GgufMetadataValue::Uint8(read_u8(cur)?)),
            1 => Ok(GgufMetadataValue::Int8(read_u8(cur)? as i8)),
            2 => Ok(GgufMetadataValue::Uint16(read_u16_le(cur)?)),
            3 => Ok(GgufMetadataValue::Int16(read_u16_le(cur)? as i16)),
            4 => Ok(GgufMetadataValue::Uint32(read_u32_le(cur)?)),
            5 => Ok(GgufMetadataValue::Int32(read_u32_le(cur)? as i32)),
            6 => Ok(GgufMetadataValue::Float32(f32::from_bits(read_u32_le(cur)?))),
            7 => Ok(GgufMetadataValue::Bool(read_u8(cur)? != 0)),
            8 => {
                let s = read_gguf_string(cur)?;
                Ok(GgufMetadataValue::String(s))
            }
            9 => {
                let elem_type = read_u32_le(cur)?;
                let count = read_u64_le(cur)?;
                let mut arr = Vec::with_capacity(count.min(65536) as usize);
                for _ in 0..count {
                    arr.push(GgufMetadataValue::parse(elem_type, cur)?);
                }
                Ok(GgufMetadataValue::Array(arr))
            }
            10 => Ok(GgufMetadataValue::Uint64(read_u64_le(cur)?)),
            11 => Ok(GgufMetadataValue::Int64(read_u64_le(cur)? as i64)),
            12 => Ok(GgufMetadataValue::Float64(f64::from_bits(read_u64_le(cur)?))),
            other => Err(TrustformersError::invalid_input(format!(
                "unknown GGUF metadata type id: {other}"
            ))),
        }
    }

    /// Return the contained string value if this is a `String` variant.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufMetadataValue::String(s) => Some(s),
            _ => None,
        }
    }
}

// ─── File structures ──────────────────────────────────────────────────────────

/// Simplified GGUF file header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufHeader {
    pub magic: u32,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub metadata: HashMap<String, GgufMetadataValue>,
}

/// Descriptor for a single tensor stored in a GGUF file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufTensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: Vec<u64>,
    pub quant_type: GgufQuantType,
    /// Byte offset of the tensor data from the start of the data section.
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Total number of elements.
    pub fn num_elements(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Estimated size of the quantized data in bytes.
    pub fn data_size_bytes(&self) -> u64 {
        let n = self.num_elements() as f64;
        let bpw = self.quant_type.bits_per_weight() as f64;
        (n * bpw / 8.0).ceil() as u64
    }
}

// ─── Reader ───────────────────────────────────────────────────────────────────

/// Pure-Rust GGUF file reader for mobile deployment.
pub struct GgufReader {
    header: GgufHeader,
    tensors: Vec<GgufTensorInfo>,
}

impl GgufReader {
    /// Parse a GGUF file from a byte slice.
    ///
    /// This performs a structural parse only — tensor data bytes are not read.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut cur = Cursor::new(data);
        Self::parse(&mut cur)
    }

    /// Parse a GGUF file from the filesystem.
    pub fn from_file(path: &Path) -> Result<Self> {
        let data = std::fs::read(path).map_err(|e| {
            TrustformersError::invalid_input(format!("cannot read GGUF file '{}': {e}", path.display()))
        })?;
        Self::from_bytes(&data)
    }

    fn parse(cur: &mut Cursor<&[u8]>) -> Result<Self> {
        // Magic
        let magic = read_u32_le(cur)?;
        if magic != GGUF_MAGIC {
            return Err(TrustformersError::invalid_input(format!(
                "invalid GGUF magic: 0x{magic:08X} (expected 0x{GGUF_MAGIC:08X})"
            )));
        }
        let version = read_u32_le(cur)?;
        if version != 3 {
            // We accept versions 2 and 3; log a warning for unexpected versions.
            if !(2..=3).contains(&version) {
                return Err(TrustformersError::invalid_input(format!(
                    "unsupported GGUF version: {version}"
                )));
            }
        }
        let tensor_count = read_u64_le(cur)?;
        let metadata_kv_count = read_u64_le(cur)?;

        // Metadata key-value pairs
        let mut metadata: HashMap<String, GgufMetadataValue> =
            HashMap::with_capacity(metadata_kv_count.min(512) as usize);
        for _ in 0..metadata_kv_count {
            let key = read_gguf_string(cur)?;
            let type_id = read_u32_le(cur)?;
            let value = GgufMetadataValue::parse(type_id, cur)?;
            metadata.insert(key, value);
        }

        // Tensor infos
        let mut tensors: Vec<GgufTensorInfo> =
            Vec::with_capacity(tensor_count.min(65536) as usize);
        for _ in 0..tensor_count {
            let name = read_gguf_string(cur)?;
            let n_dims = read_u32_le(cur)?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(read_u64_le(cur)?);
            }
            let qt_raw = read_u32_le(cur)?;
            let quant_type = GgufQuantType::from_u32(qt_raw).ok_or_else(|| {
                TrustformersError::invalid_input(format!(
                    "unknown quant type {qt_raw} for tensor '{name}'"
                ))
            })?;
            let offset = read_u64_le(cur)?;
            tensors.push(GgufTensorInfo {
                name,
                n_dims,
                dims,
                quant_type,
                offset,
            });
        }

        Ok(Self {
            header: GgufHeader {
                magic,
                version,
                tensor_count,
                metadata_kv_count,
                metadata,
            },
            tensors,
        })
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Get tensor info by name.
    pub fn get_tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    /// Get a metadata value by key.
    pub fn metadata(&self, key: &str) -> Option<&GgufMetadataValue> {
        self.header.metadata.get(key)
    }

    /// Model architecture string from metadata (`general.architecture`).
    pub fn architecture(&self) -> Option<&str> {
        self.header
            .metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
    }

    /// Total model size estimate in megabytes.
    pub fn model_size_mb(&self) -> f64 {
        let total_bytes: u64 = self.tensors.iter().map(|t| t.data_size_bytes()).sum();
        total_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Count of tensors per quantization type name.
    pub fn quant_summary(&self) -> HashMap<String, usize> {
        let mut map: HashMap<String, usize> = HashMap::new();
        for t in &self.tensors {
            *map.entry(t.quant_type.name().to_owned()).or_insert(0) += 1;
        }
        map
    }

    /// Access the raw header.
    pub fn header(&self) -> &GgufHeader {
        &self.header
    }

    /// Iterate over all tensor infos.
    pub fn tensors(&self) -> &[GgufTensorInfo] {
        &self.tensors
    }
}

// ─── Conversion helper ────────────────────────────────────────────────────────

/// Map a GGUF quantization type to the best matching mobile precision.
pub fn gguf_to_mobile_precision(quant_type: GgufQuantType) -> MobilePrecision {
    match quant_type {
        GgufQuantType::F32 => MobilePrecision::FP16,    // downcast for mobile
        GgufQuantType::F16 => MobilePrecision::FP16,
        GgufQuantType::Q4_0 | GgufQuantType::Q4_1 | GgufQuantType::Q4_K => MobilePrecision::INT4,
        GgufQuantType::Q5_0 | GgufQuantType::Q5_1 | GgufQuantType::Q5_K => MobilePrecision::Mixed4_8,
        GgufQuantType::Q6_K => MobilePrecision::Mixed8_16,
        GgufQuantType::Q8_0 => MobilePrecision::INT8,
    }
}

// ─── Private I/O helpers ──────────────────────────────────────────────────────

fn read_u8(cur: &mut Cursor<&[u8]>) -> Result<u8> {
    let mut buf = [0u8; 1];
    cur.read_exact(&mut buf).map_err(|e| {
        TrustformersError::invalid_input(format!("GGUF read_u8 failed: {e}"))
    })?;
    Ok(buf[0])
}

fn read_u16_le(cur: &mut Cursor<&[u8]>) -> Result<u16> {
    let mut buf = [0u8; 2];
    cur.read_exact(&mut buf).map_err(|e| {
        TrustformersError::invalid_input(format!("GGUF read_u16 failed: {e}"))
    })?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32_le(cur: &mut Cursor<&[u8]>) -> Result<u32> {
    let mut buf = [0u8; 4];
    cur.read_exact(&mut buf).map_err(|e| {
        TrustformersError::invalid_input(format!("GGUF read_u32 failed: {e}"))
    })?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_le(cur: &mut Cursor<&[u8]>) -> Result<u64> {
    let mut buf = [0u8; 8];
    cur.read_exact(&mut buf).map_err(|e| {
        TrustformersError::invalid_input(format!("GGUF read_u64 failed: {e}"))
    })?;
    Ok(u64::from_le_bytes(buf))
}

/// Read a GGUF length-prefixed UTF-8 string (u64 length, then bytes).
fn read_gguf_string(cur: &mut Cursor<&[u8]>) -> Result<String> {
    let len = read_u64_le(cur)?;
    if len > 1024 * 1024 {
        return Err(TrustformersError::invalid_input(format!(
            "GGUF string too long: {len} bytes"
        )));
    }
    let mut buf = vec![0u8; len as usize];
    cur.read_exact(&mut buf).map_err(|e| {
        TrustformersError::invalid_input(format!("GGUF string read failed: {e}"))
    })?;
    String::from_utf8(buf).map_err(|e| {
        TrustformersError::invalid_input(format!("GGUF string is not valid UTF-8: {e}"))
    })
}

// ─── Mobile loader types ──────────────────────────────────────────────────────

/// Configuration for GGUF-based mobile model loading.
#[derive(Debug, Clone)]
pub struct GgufMobileConfig {
    /// Maximum model size in megabytes that can be loaded into memory.
    pub max_model_size_mb: f64,
    /// Number of layers to offload to slower storage (e.g. flash) instead of RAM.
    pub offload_layers: usize,
    /// Use memory-mapped file access instead of loading into heap.
    pub mmap: bool,
}

impl Default for GgufMobileConfig {
    fn default() -> Self {
        Self {
            max_model_size_mb: 2048.0,
            offload_layers: 0,
            mmap: true,
        }
    }
}

/// Metadata about a single model layer for mobile memory planning.
#[derive(Debug, Clone)]
pub struct GgufLayerInfo {
    /// Layer / tensor name.
    pub name: String,
    /// Quantization type used for this layer.
    pub quant_type: GgufQuantType,
    /// On-disk / in-memory size in bytes.
    pub size_bytes: usize,
    /// Tensor shape dimensions.
    pub tensor_shape: Vec<usize>,
}

impl GgufLayerInfo {
    /// Construct a layer info record.
    pub fn new(name: impl Into<String>, quant_type: GgufQuantType, size_bytes: usize, tensor_shape: Vec<usize>) -> Self {
        Self {
            name: name.into(),
            quant_type,
            size_bytes,
            tensor_shape,
        }
    }
}

/// Mobile-oriented GGUF model loader with memory budget awareness.
pub struct GgufMobileLoader {
    /// Configuration for this loader instance.
    pub config: GgufMobileConfig,
}

impl GgufMobileLoader {
    /// Create a loader with the given configuration.
    pub fn new(config: GgufMobileConfig) -> Self {
        Self { config }
    }

    /// Create a loader with default configuration.
    pub fn default_config() -> Self {
        Self::new(GgufMobileConfig::default())
    }

    /// Estimate total memory requirement for a slice of layers.
    ///
    /// Returns the sum of `size_bytes` for all layers in bytes.
    pub fn estimate_memory_requirement(layers: &[GgufLayerInfo]) -> u64 {
        layers.iter().map(|l| l.size_bytes as u64).sum()
    }

    /// Return the indices of layers that fit within `budget_mb` megabytes
    /// when loaded cumulatively (greedy, first-fit).
    pub fn layers_that_fit(layers: &[GgufLayerInfo], budget_mb: f64) -> Vec<usize> {
        let budget_bytes = (budget_mb * 1024.0 * 1024.0) as u64;
        let mut cumulative = 0u64;
        let mut indices = Vec::new();
        for (i, layer) in layers.iter().enumerate() {
            let next = cumulative + layer.size_bytes as u64;
            if next <= budget_bytes {
                cumulative = next;
                indices.push(i);
            }
        }
        indices
    }

    /// Effective bits per weight for each GGUF quantization type.
    ///
    /// Unlike the raw `bits_per_weight()` on `GgufQuantType`, these values
    /// account for scale/metadata overhead:
    /// - Q4_0 = 4.5 (4-bit data + 0.5-bit overhead for block scale)
    /// - Q4_1 = 5.0 (4-bit data + 1-bit for min value)
    /// - Q5_0 = 5.5
    /// - Q5_1 = 6.0
    /// - Q8_0 = 8.5
    /// - Q4_K, Q5_K, Q6_K use their nominal bit-widths with K-quant overhead
    /// - F16 = 16.0, F32 = 32.0
    pub fn effective_bits_per_weight(quant_type: GgufQuantType) -> f32 {
        match quant_type {
            GgufQuantType::Q4_0 => 4.5,
            GgufQuantType::Q4_1 => 5.0,
            GgufQuantType::Q5_0 => 5.5,
            GgufQuantType::Q5_1 => 6.0,
            GgufQuantType::Q8_0 => 8.5,
            GgufQuantType::Q4_K => 4.58, // K-quant with super-block overhead
            GgufQuantType::Q5_K => 5.54,
            GgufQuantType::Q6_K => 6.56,
            GgufQuantType::F16 => 16.0,
            GgufQuantType::F32 => 32.0,
        }
    }

    /// Compression ratio versus FP32, using effective bits per weight.
    pub fn compression_ratio_vs_f32(quant_type: GgufQuantType) -> f32 {
        32.0 / Self::effective_bits_per_weight(quant_type)
    }
}

// ─── Test helpers ─────────────────────────────────────────────────────────────

#[cfg(test)]
pub(crate) fn make_minimal_gguf(arch: &str) -> Vec<u8> {
    use std::io::Write;
    let mut buf: Vec<u8> = Vec::new();
    // Magic
    buf.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
    // Version 3
    buf.write_all(&3u32.to_le_bytes()).unwrap();
    // tensor_count = 1
    buf.write_all(&1u64.to_le_bytes()).unwrap();
    // metadata_kv_count = 1
    buf.write_all(&1u64.to_le_bytes()).unwrap();

    // Metadata: key = "general.architecture", type = string (8), value = arch
    let key = "general.architecture";
    buf.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
    buf.write_all(key.as_bytes()).unwrap();
    buf.write_all(&8u32.to_le_bytes()).unwrap(); // type = string
    buf.write_all(&(arch.len() as u64).to_le_bytes()).unwrap();
    buf.write_all(arch.as_bytes()).unwrap();

    // One tensor: "embed.weight" Q4_K 2D [16, 32] offset=0
    let tname = "embed.weight";
    buf.write_all(&(tname.len() as u64).to_le_bytes()).unwrap();
    buf.write_all(tname.as_bytes()).unwrap();
    buf.write_all(&2u32.to_le_bytes()).unwrap(); // n_dims
    buf.write_all(&16u64.to_le_bytes()).unwrap(); // dim0
    buf.write_all(&32u64.to_le_bytes()).unwrap(); // dim1
    buf.write_all(&12u32.to_le_bytes()).unwrap(); // Q4_K = 12
    buf.write_all(&0u64.to_le_bytes()).unwrap(); // offset

    buf
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_gguf() {
        let data = make_minimal_gguf("llama");
        let reader = GgufReader::from_bytes(&data).expect("should parse");
        assert_eq!(reader.architecture(), Some("llama"));
        assert_eq!(reader.tensor_names(), vec!["embed.weight"]);
    }

    #[test]
    fn test_invalid_magic_returns_error() {
        let mut data = make_minimal_gguf("llama");
        data[0] = 0xFF; // corrupt magic
        assert!(GgufReader::from_bytes(&data).is_err());
    }

    #[test]
    fn test_tensor_info_lookup() {
        let data = make_minimal_gguf("mistral");
        let reader = GgufReader::from_bytes(&data).expect("should parse");
        let t = reader.get_tensor("embed.weight").expect("tensor should exist");
        assert_eq!(t.quant_type, GgufQuantType::Q4_K);
        assert_eq!(t.n_dims, 2);
        assert_eq!(t.dims, vec![16, 32]);
    }

    #[test]
    fn test_quant_bits_per_weight() {
        assert_eq!(GgufQuantType::F32.bits_per_weight(), 32.0);
        assert_eq!(GgufQuantType::F16.bits_per_weight(), 16.0);
        assert_eq!(GgufQuantType::Q4_0.bits_per_weight(), 4.0);
        assert_eq!(GgufQuantType::Q8_0.bits_per_weight(), 8.0);
        assert_eq!(GgufQuantType::Q6_K.bits_per_weight(), 6.0);
    }

    #[test]
    fn test_compression_vs_f32() {
        assert_eq!(GgufQuantType::F16.compression_vs_f32(), 2.0);
        assert_eq!(GgufQuantType::Q4_K.compression_vs_f32(), 8.0);
        assert_eq!(GgufQuantType::Q8_0.compression_vs_f32(), 4.0);
    }

    #[test]
    fn test_model_size_estimate() {
        let data = make_minimal_gguf("gpt2");
        let reader = GgufReader::from_bytes(&data).expect("should parse");
        // 16*32 = 512 elements at Q4_K (4 bits = 0.5 bytes each) = 256 bytes
        let size = reader.model_size_mb();
        assert!(size > 0.0 && size < 1.0);
    }

    #[test]
    fn test_quant_summary() {
        let data = make_minimal_gguf("phi");
        let reader = GgufReader::from_bytes(&data).expect("should parse");
        let summary = reader.quant_summary();
        assert_eq!(summary.get("Q4_K").copied(), Some(1));
    }

    #[test]
    fn test_gguf_to_mobile_precision() {
        assert_eq!(gguf_to_mobile_precision(GgufQuantType::Q4_K), MobilePrecision::INT4);
        assert_eq!(gguf_to_mobile_precision(GgufQuantType::Q8_0), MobilePrecision::INT8);
        assert_eq!(gguf_to_mobile_precision(GgufQuantType::F16), MobilePrecision::FP16);
        assert_eq!(gguf_to_mobile_precision(GgufQuantType::Q5_K), MobilePrecision::Mixed4_8);
    }

    #[test]
    fn test_quant_type_names() {
        let names = [
            (GgufQuantType::F32, "F32"),
            (GgufQuantType::Q4_0, "Q4_0"),
            (GgufQuantType::Q4_K, "Q4_K"),
            (GgufQuantType::Q6_K, "Q6_K"),
        ];
        for (qt, expected) in names {
            assert_eq!(qt.name(), expected);
        }
    }

    #[test]
    fn test_from_file_missing_returns_error() {
        let path = std::path::Path::new("/nonexistent/file.gguf");
        assert!(GgufReader::from_file(path).is_err());
    }

    #[test]
    fn test_num_elements_and_data_size() {
        let data = make_minimal_gguf("bert");
        let reader = GgufReader::from_bytes(&data).expect("should parse");
        let t = reader.get_tensor("embed.weight").expect("should exist");
        assert_eq!(t.num_elements(), 16 * 32);
        // Q4_K at 4 bits/weight: 512 * 0.5 = 256 bytes
        assert_eq!(t.data_size_bytes(), 256);
    }
}
