use serde::Deserialize;
/// HuggingFace Weight Loader
///
/// This module provides comprehensive support for loading weights from HuggingFace model formats.
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use trustformers_core::{
    errors::{invalid_format, runtime_error, Result, TrustformersError},
    tensor::Tensor,
    traits::WeightReader,
    utils::weight_loading::{torch, PyTorchReader},
};

use super::config::{WeightDataType, WeightFormat, WeightLoadingConfig};

/// HuggingFace model index structure
#[derive(Debug, Deserialize)]
pub struct HuggingFaceIndex {
    pub metadata: HuggingFaceMetadata,
    pub weight_map: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
pub struct HuggingFaceMetadata {
    pub total_size: u64,
    pub format: String,
}

/// SafeTensors header structure
///
/// Note: SafeTensors format has a FLAT structure where tensor names are keys at the root level,
/// and __metadata__ is a special key (not nested under a "tensors" field).
///
/// Actual format: {"__metadata__": {...}, "tensor.name": {...}, "other.tensor": {...}}
#[derive(Debug)]
pub struct SafeTensorsHeader {
    pub metadata: Option<HashMap<String, String>>,
    pub tensors: HashMap<String, TensorInfo>,
}

impl<'de> serde::Deserialize<'de> for SafeTensorsHeader {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Deserialize as a flat HashMap
        let mut map: HashMap<String, serde_json::Value> = HashMap::deserialize(deserializer)?;

        // Extract __metadata__ if present (special key)
        let metadata = map.remove("__metadata__").and_then(|v| serde_json::from_value(v).ok());

        // All remaining keys are tensor names
        let tensors: HashMap<String, TensorInfo> = map
            .into_iter()
            .filter_map(|(k, v)| serde_json::from_value(v).ok().map(|info| (k, info)))
            .collect();

        Ok(SafeTensorsHeader { metadata, tensors })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [u64; 2],
}

/// Weight loader trait
pub trait WeightLoader {
    fn load_tensor(&mut self, name: &str) -> Result<Tensor>;
    fn list_tensors(&self) -> Result<Vec<String>>;
    fn tensor_info(&self, name: &str) -> Result<Option<TensorMetadata>>;
    fn close(&mut self) -> Result<()>;
}

/// Tensor metadata
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    pub shape: Vec<usize>,
    pub dtype: WeightDataType,
    pub size_bytes: u64,
    pub offset: u64,
}

/// Lazy tensor that loads data on-demand
pub struct LazyTensor {
    name: String,
    #[allow(dead_code)]
    filename: String,
    metadata: TensorMetadata,
    model_dir: PathBuf,
    config: WeightLoadingConfig,
}

/// HuggingFace weight loader
///
/// Reads a model directory in either supported format:
///
/// * **safetensors** — the header is parsed and each tensor is read from its
///   declared byte range.
/// * **PyTorch `.bin` / `.pt`** — handed to
///   [`trustformers_core::utils::weight_loading::PyTorchReader`], which walks the
///   ZIP archive and interprets `data.pkl`. No pickle parsing happens in this
///   crate.
///
/// A previous revision never parsed pickle at all: it guessed a tensor's shape
/// from hardcoded BERT-base constants keyed off the parameter name, then scanned
/// the file for the first four bytes that decoded to a finite float below 100 and
/// treated that offset as the start of the tensor. Every `.bin` load returned a
/// fabricated tensor. That code is gone; an unparsable checkpoint is now an error.
pub struct HuggingFaceLoader {
    config: WeightLoadingConfig,
    index: HuggingFaceIndex,
    model_dir: PathBuf,
    tensor_cache: HashMap<String, Tensor>,
    /// Real per-tensor metadata, read from the files at construction.
    metadata: HashMap<String, TensorMetadata>,
    /// Parsed PyTorch checkpoints, keyed by file name.
    pytorch_readers: HashMap<String, PyTorchReader>,
}

impl HuggingFaceLoader {
    /// Open a HuggingFace model directory.
    ///
    /// # Errors
    ///
    /// Fails when the directory holds no recognised weight file, when an index
    /// file is malformed, or when a weight file cannot be parsed.
    pub fn new(model_dir: impl AsRef<Path>, config: WeightLoadingConfig) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();

        let index = Self::discover_index(&model_dir)?;

        let mut loader = Self {
            config,
            index,
            model_dir,
            tensor_cache: HashMap::new(),
            metadata: HashMap::new(),
            pytorch_readers: HashMap::new(),
        };
        loader.read_all_metadata()?;
        Ok(loader)
    }

    /// Build the tensor-name -> file map for a model directory.
    fn discover_index(model_dir: &Path) -> Result<HuggingFaceIndex> {
        for index_name in [
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ] {
            let index_path = model_dir.join(index_name);
            if index_path.exists() {
                return Self::load_index(&index_path);
            }
        }
        Self::create_single_file_index(model_dir)
    }

    fn load_index(path: &Path) -> Result<HuggingFaceIndex> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).map_err(|e| {
            TrustformersError::weight_load_error(format!(
                "Failed to parse HuggingFace index {}: {}",
                path.display(),
                e
            ))
        })
    }

    /// Build an index for a directory holding a single weight file.
    ///
    /// The tensor names are enumerated from the file itself for both formats.
    /// A previous revision fell back to a `"*"` wildcard entry for `.bin` files,
    /// which made `list_tensors` report a single tensor called `*` and let
    /// `find_tensor_file` resolve any name at all.
    fn create_single_file_index(model_dir: &Path) -> Result<HuggingFaceIndex> {
        let candidates: [(&str, bool); 4] = [
            ("model.safetensors", true),
            ("pytorch_model.bin", false),
            ("pytorch_model.pt", false),
            ("pytorch_model.pth", false),
        ];

        let (weight_file, is_safetensors) = candidates
            .iter()
            .find(|(name, _)| model_dir.join(name).exists())
            .copied()
            .ok_or_else(|| {
                TrustformersError::file_not_found(format!(
                    "No weight files found in {} (looked for model.safetensors, pytorch_model.bin, \
                     pytorch_model.pt, pytorch_model.pth and the sharded index files)",
                    model_dir.display()
                ))
            })?;

        let path = model_dir.join(weight_file);
        let tensor_names = if is_safetensors {
            Self::read_safetensors_header(&path)?
                .tensors
                .keys()
                .cloned()
                .collect::<Vec<_>>()
        } else {
            PyTorchReader::from_file(&path)?.list_tensors()
        };

        let mut weight_map = HashMap::new();
        for name in tensor_names {
            weight_map.insert(name, weight_file.to_string());
        }

        Ok(HuggingFaceIndex {
            metadata: HuggingFaceMetadata {
                total_size: 0,
                format: if is_safetensors { "safetensors" } else { "pytorch" }.to_string(),
            },
            weight_map,
        })
    }

    /// Parse a safetensors header from disk.
    fn read_safetensors_header(path: &Path) -> Result<SafeTensorsHeader> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut header_len_bytes = [0u8; 8];
        reader.read_exact(&mut header_len_bytes)?;
        let header_len = u64::from_le_bytes(header_len_bytes);

        let mut header_bytes = vec![0u8; header_len as usize];
        reader.read_exact(&mut header_bytes)?;
        let header_str = std::str::from_utf8(&header_bytes).map_err(|e| {
            TrustformersError::weight_load_error(format!(
                "Invalid UTF-8 in SafeTensors header of {}: {}",
                path.display(),
                e
            ))
        })?;
        serde_json::from_str(header_str).map_err(|e| {
            TrustformersError::serialization_error(format!(
                "Failed to parse SafeTensors header of {}: {}",
                path.display(),
                e
            ))
        })
    }

    /// Read the real shape, dtype and byte size of every tensor in the index.
    fn read_all_metadata(&mut self) -> Result<()> {
        let filenames: std::collections::BTreeSet<String> =
            self.index.weight_map.values().cloned().collect();

        for filename in filenames {
            let path = self.model_dir.join(&filename);
            match self.detect_format(&filename)? {
                WeightFormat::SafeTensors => {
                    let header = Self::read_safetensors_header(&path)?;
                    for (name, info) in header.tensors {
                        let dtype = Self::weight_dtype(&info.dtype)?;
                        self.metadata.insert(
                            name,
                            TensorMetadata {
                                shape: info.shape,
                                dtype,
                                size_bytes: info.data_offsets[1] - info.data_offsets[0],
                                offset: info.data_offsets[0],
                            },
                        );
                    }
                },
                WeightFormat::HuggingFaceBin => {
                    let reader = PyTorchReader::from_file(&path)?;
                    for name in reader.list_tensors() {
                        let record = reader.state_dict().get(&name).ok_or_else(|| {
                            TrustformersError::weight_load_error(format!(
                                "PyTorch checkpoint {} listed {name} but does not hold it",
                                path.display()
                            ))
                        })?;
                        let shape = record.tensor.shape();
                        let element_count: usize = shape.iter().product();
                        self.metadata.insert(
                            name,
                            TensorMetadata {
                                shape,
                                dtype: Self::torch_dtype(record.dtype),
                                size_bytes: (element_count * record.dtype.size_in_bytes()) as u64,
                                offset: 0,
                            },
                        );
                    }
                    self.pytorch_readers.insert(filename.clone(), reader);
                },
                other => {
                    return Err(invalid_format(
                        "weight format",
                        format!("Unsupported weight format {other:?} for {filename}"),
                    ))
                },
            }
        }
        Ok(())
    }

    /// Map a safetensors dtype string onto this crate's dtype enum.
    fn weight_dtype(dtype: &str) -> Result<WeightDataType> {
        match dtype {
            "F32" => Ok(WeightDataType::Float32),
            "F16" => Ok(WeightDataType::Float16),
            "BF16" => Ok(WeightDataType::BFloat16),
            "I8" | "U8" => Ok(WeightDataType::Int8),
            other => Err(invalid_format(
                "dtype",
                format!("Unsupported safetensors dtype: {other}"),
            )),
        }
    }

    /// Map a torch dtype onto this crate's dtype enum.
    fn torch_dtype(dtype: torch::TorchDType) -> WeightDataType {
        match dtype {
            torch::TorchDType::F64 | torch::TorchDType::F32 => WeightDataType::Float32,
            torch::TorchDType::F16 => WeightDataType::Float16,
            torch::TorchDType::BF16 => WeightDataType::BFloat16,
            _ => WeightDataType::Int8,
        }
    }

    /// Load a tensor from a PyTorch checkpoint.
    ///
    /// # Errors
    ///
    /// Fails when the checkpoint does not hold the tensor. It never reconstructs
    /// one from guessed shapes and scanned offsets.
    fn load_from_pytorch_bin(&mut self, name: &str, filename: &str) -> Result<Tensor> {
        if !self.pytorch_readers.contains_key(filename) {
            let path = self.model_dir.join(filename);
            let reader = PyTorchReader::from_file(&path)?;
            self.pytorch_readers.insert(filename.to_string(), reader);
        }
        let reader = self.pytorch_readers.get_mut(filename).ok_or_else(|| {
            TrustformersError::runtime_error(format!(
                "PyTorch reader for {filename} missing after insertion"
            ))
        })?;
        reader.read_tensor(name)
    }

    /// Load a tensor with lazy loading
    #[allow(dead_code)]
    fn load_lazy(&mut self, name: &str) -> Result<LazyTensor> {
        let filename = self.find_tensor_file(name)?;
        let metadata = self.get_tensor_metadata(name)?;

        Ok(LazyTensor {
            name: name.to_string(),
            filename,
            metadata,
            model_dir: self.model_dir.clone(),
            config: self.config.clone(),
        })
    }

    fn find_tensor_file(&self, name: &str) -> Result<String> {
        self.index
            .weight_map
            .get(name)
            .cloned()
            .ok_or_else(|| runtime_error(format!("Tensor not found: {}", name)))
    }

    /// The tensor's real metadata, as read from the weight file.
    ///
    /// A previous revision returned a hardcoded `[1024, 768]` shape and a
    /// matching invented byte size for every tensor.
    fn get_tensor_metadata(&self, name: &str) -> Result<TensorMetadata> {
        self.metadata
            .get(name)
            .cloned()
            .ok_or_else(|| runtime_error(format!("No metadata for tensor: {}", name)))
    }

    fn detect_format(&self, filename: &str) -> Result<WeightFormat> {
        if filename.ends_with(".bin") || filename.ends_with(".pt") || filename.ends_with(".pth") {
            Ok(WeightFormat::HuggingFaceBin)
        } else if filename.ends_with(".safetensors") {
            Ok(WeightFormat::SafeTensors)
        } else {
            Err(invalid_format(
                "file format",
                format!("Unknown format for file: {}", filename),
            ))
        }
    }

    /// Read a tensor out of a safetensors file.
    ///
    /// A fresh handle is opened per tensor: a cached `BufReader` keeps stale
    /// buffered bytes across a `seek`, which silently returns the wrong data.
    fn load_from_safetensors(&mut self, name: &str, filename: &str) -> Result<Tensor> {
        let file_path = self.model_dir.join(filename);
        let file = File::open(&file_path)?;
        let mut reader = BufReader::new(file);

        let mut header_len_bytes = [0u8; 8];
        reader.read_exact(&mut header_len_bytes)?;
        let header_len = u64::from_le_bytes(header_len_bytes);

        let mut header_bytes = vec![0u8; header_len as usize];
        reader.read_exact(&mut header_bytes)?;

        let header_str = std::str::from_utf8(&header_bytes).map_err(|e| {
            TrustformersError::weight_load_error(format!(
                "Invalid UTF-8 in SafeTensors header: {}",
                e
            ))
        })?;

        let header: SafeTensorsHeader = serde_json::from_str(header_str).map_err(|e| {
            TrustformersError::serialization_error(format!(
                "Failed to parse SafeTensors header of {}: {}",
                file_path.display(),
                e
            ))
        })?;

        let tensor_info = header
            .tensors
            .get(name)
            .ok_or_else(|| runtime_error(format!("Tensor not found: {}", name)))?;

        // Offsets are relative to the start of the tensor data section, which
        // begins right after the 8-byte length and the header itself.
        let tensor_data_start = 8 + header_len;
        reader.seek(SeekFrom::Start(
            tensor_data_start + tensor_info.data_offsets[0],
        ))?;

        let data_len = (tensor_info.data_offsets[1] - tensor_info.data_offsets[0]) as usize;
        let mut data = vec![0u8; data_len];
        reader.read_exact(&mut data)?;

        self.bytes_to_tensor(data, &tensor_info.dtype, &tensor_info.shape)
    }

    fn bytes_to_tensor(&self, data: Vec<u8>, dtype: &str, shape: &[usize]) -> Result<Tensor> {
        let expected: usize = shape.iter().product();
        let floats: Vec<f32> = match dtype {
            "F32" => data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect(),
            "F16" => data
                .chunks_exact(2)
                .map(|chunk| {
                    half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32()
                })
                .collect(),
            "BF16" => data
                .chunks_exact(2)
                .map(|chunk| {
                    half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32()
                })
                .collect(),
            "F64" => data
                .chunks_exact(8)
                .map(|c| {
                    f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect(),
            "I8" => data.iter().map(|&b| b as i8 as f32).collect(),
            "U8" => data.iter().map(|&b| b as f32).collect(),
            other => {
                return Err(invalid_format(
                    "dtype",
                    format!("Unsupported dtype: {}", other),
                ))
            },
        };

        if floats.len() != expected {
            return Err(TrustformersError::shape_error(format!(
                "tensor declares shape {shape:?} ({expected} elements) but its payload decodes to \
                 {} elements",
                floats.len()
            )));
        }

        Tensor::from_vec(floats, shape)
    }
}

impl WeightLoader for HuggingFaceLoader {
    fn load_tensor(&mut self, name: &str) -> Result<Tensor> {
        // Check cache first
        if let Some(tensor) = self.tensor_cache.get(name) {
            return Ok(tensor.clone());
        }

        let filename = self.find_tensor_file(name)?;

        let tensor = match self.detect_format(&filename)? {
            WeightFormat::HuggingFaceBin => self.load_from_pytorch_bin(name, &filename)?,
            WeightFormat::SafeTensors => self.load_from_safetensors(name, &filename)?,
            other => {
                return Err(invalid_format(
                    "weight format",
                    format!("Unsupported weight format: {other:?}"),
                ));
            },
        };

        // Cache if not lazy loading
        if !self.config.lazy_loading {
            self.tensor_cache.insert(name.to_string(), tensor.clone());
        }

        Ok(tensor)
    }

    fn list_tensors(&self) -> Result<Vec<String>> {
        let mut names: Vec<String> = self.index.weight_map.keys().cloned().collect();
        names.sort();
        Ok(names)
    }

    fn tensor_info(&self, name: &str) -> Result<Option<TensorMetadata>> {
        Ok(self.metadata.get(name).cloned())
    }

    fn close(&mut self) -> Result<()> {
        self.pytorch_readers.clear();
        self.tensor_cache.clear();
        Ok(())
    }
}

impl LazyTensor {
    pub fn load(&self) -> Result<Tensor> {
        // Create a temporary loader instance to load this specific tensor
        let mut temp_loader = HuggingFaceLoader::new(&self.model_dir, self.config.clone())?;
        temp_loader.load_tensor(&self.name)
    }

    pub fn metadata(&self) -> &TensorMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_loading::config::{WeightDataType, WeightFormat, WeightLoadingConfig};

    // ── WeightFormat tests ────────────────────────────────────────────────────

    #[test]
    fn test_weight_format_equality() {
        let f1 = WeightFormat::SafeTensors;
        let f2 = WeightFormat::SafeTensors;
        assert_eq!(f1, f2);
    }

    #[test]
    fn test_weight_format_custom() {
        let f = WeightFormat::Custom("msgpack".to_string());
        if let WeightFormat::Custom(name) = &f {
            assert_eq!(name, "msgpack");
        } else {
            panic!("Expected Custom variant");
        }
    }

    #[test]
    fn test_weight_format_hf_bin_ne_safetensors() {
        assert_ne!(WeightFormat::HuggingFaceBin, WeightFormat::SafeTensors);
    }

    // ── WeightLoadingConfig tests ─────────────────────────────────────────────

    #[test]
    fn test_weight_loading_config_default() {
        let cfg = WeightLoadingConfig::default();
        assert!(!cfg.lazy_loading);
        assert!(!cfg.memory_mapped);
        assert!(!cfg.streaming);
        assert_eq!(cfg.device, "cpu");
        assert!(cfg.verify_checksums);
    }

    #[test]
    fn test_weight_loading_config_lazy_loading_flag() {
        let cfg = WeightLoadingConfig {
            lazy_loading: true,
            ..WeightLoadingConfig::default()
        };
        assert!(cfg.lazy_loading);
    }

    #[test]
    fn test_weight_loading_config_with_format() {
        let cfg = WeightLoadingConfig {
            format: Some(WeightFormat::SafeTensors),
            ..WeightLoadingConfig::default()
        };
        if let Some(WeightFormat::SafeTensors) = &cfg.format {
            // expected
        } else {
            panic!("Expected SafeTensors format");
        }
    }

    // ── TensorMetadata tests ──────────────────────────────────────────────────

    #[test]
    fn test_tensor_metadata_construction() {
        let meta = TensorMetadata {
            shape: vec![128, 256],
            dtype: WeightDataType::Float32,
            size_bytes: 128 * 256 * 4,
            offset: 0,
        };
        assert_eq!(meta.shape, vec![128, 256]);
        assert_eq!(meta.size_bytes, 131072);
    }

    #[test]
    fn test_tensor_metadata_clone() {
        let meta = TensorMetadata {
            shape: vec![32, 64],
            dtype: WeightDataType::Float16,
            size_bytes: 32 * 64 * 2,
            offset: 1024,
        };
        let cloned = meta.clone();
        assert_eq!(cloned.shape, meta.shape);
        assert_eq!(cloned.offset, meta.offset);
    }

    // ── TensorInfo tests ──────────────────────────────────────────────────────

    #[test]
    fn test_tensor_info_clone() {
        let info = TensorInfo {
            dtype: "F32".to_string(),
            shape: vec![64, 128],
            data_offsets: [0, 32768],
        };
        let cloned = info.clone();
        assert_eq!(cloned.dtype, "F32");
        assert_eq!(cloned.shape, vec![64, 128]);
        assert_eq!(cloned.data_offsets, [0, 32768]);
    }

    // ── HuggingFaceLoader directory tests ─────────────────────────────────────

    #[test]
    fn test_huggingface_loader_nonexistent_dir() {
        let cfg = WeightLoadingConfig::default();
        let result = HuggingFaceLoader::new("/nonexistent/model/dir", cfg);
        assert!(
            result.is_err(),
            "Expected error for nonexistent model directory"
        );
    }

    #[test]
    fn test_huggingface_loader_empty_dir() {
        let dir = std::env::temp_dir().join("trustformers_hf_test_empty_dir");
        let _ = std::fs::create_dir_all(&dir);
        let cfg = WeightLoadingConfig::default();
        let result = HuggingFaceLoader::new(&dir, cfg);
        // Should fail since no weight files are present
        assert!(
            result.is_err(),
            "Expected error for directory with no weight files"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── SafeTensors format tests ──────────────────────────────────────────────

    #[test]
    fn test_safetensors_header_empty() {
        // Deserializing a minimal empty JSON object
        let json = r#"{"__metadata__": {}}"#;
        let result: std::result::Result<SafeTensorsHeader, _> = serde_json::from_str(json);
        assert!(
            result.is_ok(),
            "Empty SafeTensors header parse failed: {:?}",
            result.err()
        );
        let header = result.expect("expected Ok");
        assert!(header.tensors.is_empty());
    }

    #[test]
    fn test_safetensors_header_with_tensor() {
        let json = r#"{
            "__metadata__": {"format": "pt"},
            "model.weight": {"dtype": "F32", "shape": [64, 128], "data_offsets": [0, 32768]}
        }"#;
        let result: std::result::Result<SafeTensorsHeader, _> = serde_json::from_str(json);
        assert!(
            result.is_ok(),
            "SafeTensors header parse failed: {:?}",
            result.err()
        );
        let header = result.expect("expected Ok");
        assert!(header.tensors.contains_key("model.weight"));
        assert!(header.metadata.is_some());
    }

    #[test]
    fn test_safetensors_header_multiple_tensors() {
        let json = r#"{
            "layer.0.weight": {"dtype": "F16", "shape": [256, 512], "data_offsets": [0, 262144]},
            "layer.0.bias": {"dtype": "F32", "shape": [256], "data_offsets": [262144, 263168]}
        }"#;
        let result: std::result::Result<SafeTensorsHeader, _> = serde_json::from_str(json);
        assert!(result.is_ok());
        let header = result.expect("expected Ok");
        assert_eq!(header.tensors.len(), 2);
        assert!(header.metadata.is_none());
    }
    // ── End-to-end tests against real weight files ───────────────────────────

    use crate::weight_loading::test_support::{build_safetensors, F32Tensor};

    struct TempModelDir {
        path: std::path::PathBuf,
    }

    impl TempModelDir {
        fn new(label: &str) -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "trustformers_hf_{label}_{}_{unique}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).expect("temp model dir must be creatable");
            Self { path }
        }

        fn write(&self, name: &str, bytes: &[u8]) {
            std::fs::write(self.path.join(name), bytes).expect("fixture file must be writable");
        }
    }

    impl Drop for TempModelDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn safetensors_directory_lists_and_loads_real_tensors() {
        let dir = TempModelDir::new("st");
        let tensors = vec![
            F32Tensor::new("a.weight", &[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            F32Tensor::new("b.bias", &[3], vec![-1.0, 0.0, 1.0]),
        ];
        dir.write("model.safetensors", &build_safetensors(&tensors));

        let mut loader = HuggingFaceLoader::new(&dir.path, WeightLoadingConfig::default())
            .expect("loader must open the directory");

        assert_eq!(
            loader.list_tensors().expect("names"),
            vec!["a.weight".to_string(), "b.bias".to_string()]
        );

        let tensor = loader.load_tensor("a.weight").expect("tensor must load");
        assert_eq!(tensor.shape(), vec![2, 3]);
        match tensor {
            Tensor::F32(arr) => assert_eq!(
                arr.iter().copied().collect::<Vec<f32>>(),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            ),
            other => panic!("expected an F32 tensor, got {other:?}"),
        }
    }

    #[test]
    fn tensor_info_reports_the_real_shape_not_a_hardcoded_one() {
        // Regression: `get_tensor_metadata` used to return `[1024, 768]` and a
        // matching invented byte size for every tensor in every model.
        let dir = TempModelDir::new("meta");
        let tensors = vec![F32Tensor::new(
            "w",
            &[4, 5],
            (0..20).map(|i| i as f32).collect(),
        )];
        dir.write("model.safetensors", &build_safetensors(&tensors));

        let loader = HuggingFaceLoader::new(&dir.path, WeightLoadingConfig::default())
            .expect("loader must open the directory");
        let info = loader
            .tensor_info("w")
            .expect("metadata lookup must succeed")
            .expect("tensor must exist");
        assert_eq!(info.shape, vec![4, 5]);
        assert_eq!(info.size_bytes, 20 * 4);
        assert!(matches!(info.dtype, WeightDataType::Float32));
    }

    #[test]
    fn unknown_tensor_names_are_rejected_rather_than_resolved_by_a_wildcard() {
        let dir = TempModelDir::new("unknown");
        let tensors = vec![F32Tensor::new("only", &[1], vec![1.0])];
        dir.write("model.safetensors", &build_safetensors(&tensors));

        let mut loader = HuggingFaceLoader::new(&dir.path, WeightLoadingConfig::default())
            .expect("loader must open the directory");
        let err = loader
            .load_tensor("something.else")
            .expect_err("an unknown tensor must not resolve");
        assert!(
            err.to_string().contains("something.else"),
            "unexpected: {err}"
        );
        assert!(loader.tensor_info("something.else").expect("lookup must succeed").is_none());
    }

    #[test]
    fn pytorch_bin_that_is_not_a_checkpoint_fails_instead_of_fabricating_weights() {
        // The old loader guessed the shape from the parameter name and scanned
        // for bytes that "looked like a float", so this returned a tensor.
        let dir = TempModelDir::new("bin");
        dir.write("pytorch_model.bin", &vec![0x42u8; 8192]);

        let result = HuggingFaceLoader::new(&dir.path, WeightLoadingConfig::default());
        let err = match result {
            Ok(_) => panic!("a file that is not a PyTorch checkpoint must be rejected"),
            Err(err) => err,
        };
        assert!(
            err.to_string().to_lowercase().contains("pytorch"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn f16_tensors_are_decoded_to_f32() {
        let dir = TempModelDir::new("f16");
        let values = [1.0f32, -2.5, 0.5, 8.0];
        let mut payload = Vec::new();
        for value in values {
            payload.extend_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes());
        }
        let header = serde_json::json!({
            "w": {"dtype": "F16", "shape": [4], "data_offsets": [0, payload.len()]}
        });
        let header_bytes = serde_json::to_vec(&header).expect("header serialises");
        let mut bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header_bytes);
        bytes.extend_from_slice(&payload);
        dir.write("model.safetensors", &bytes);

        let mut loader = HuggingFaceLoader::new(&dir.path, WeightLoadingConfig::default())
            .expect("loader must open the directory");
        match loader.load_tensor("w").expect("tensor must load") {
            Tensor::F32(arr) => {
                assert_eq!(arr.iter().copied().collect::<Vec<f32>>(), values.to_vec());
            },
            other => panic!("expected an F32 tensor, got {other:?}"),
        }
    }
}
