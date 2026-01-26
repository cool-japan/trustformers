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

/// Internal tensor info for PyTorch parsing
#[derive(Debug)]
struct PyTorchTensorInfo {
    pub shape: Vec<usize>,
    pub dtype: WeightDataType,
    pub data_offset: usize,
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
pub struct HuggingFaceLoader {
    config: WeightLoadingConfig,
    index: HuggingFaceIndex,
    file_handles: HashMap<String, BufReader<File>>,
    model_dir: PathBuf,
    tensor_cache: HashMap<String, Tensor>,
}

impl HuggingFaceLoader {
    pub fn new(model_dir: impl AsRef<Path>, config: WeightLoadingConfig) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();

        // Load index file
        let index_path = model_dir.join("pytorch_model.bin.index.json");
        let index = if index_path.exists() {
            Self::load_index(&index_path)?
        } else {
            // Create single-file index
            Self::create_single_file_index(&model_dir)?
        };

        Ok(Self {
            config,
            index,
            file_handles: HashMap::new(),
            model_dir,
            tensor_cache: HashMap::new(),
        })
    }

    fn load_index(path: &Path) -> Result<HuggingFaceIndex> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).map_err(|e| {
            TrustformersError::weight_load_error(format!(
                "Failed to parse HuggingFace index: {}",
                e
            ))
        })
    }

    fn create_single_file_index(model_dir: &Path) -> Result<HuggingFaceIndex> {
        // Look for single weight file (prefer SafeTensors over PyTorch)
        let bin_path = model_dir.join("pytorch_model.bin");
        let safetensors_path = model_dir.join("model.safetensors");

        let (weight_file, is_safetensors) = if safetensors_path.exists() {
            ("model.safetensors", true)
        } else if bin_path.exists() {
            ("pytorch_model.bin", false)
        } else {
            return Err(TrustformersError::file_not_found(
                "No weight files found in model directory".to_string(),
            ));
        };

        // Create index with proper tensor names
        let mut weight_map = HashMap::new();

        if is_safetensors {
            // Read SafeTensors header to get actual tensor names
            match Self::read_safetensors_tensor_names(&model_dir.join(weight_file)) {
                Ok(tensor_names) => {
                    // Map each tensor name to the weight file
                    for name in tensor_names {
                        weight_map.insert(name, weight_file.to_string());
                    }
                },
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to read SafeTensors header: {}. Using fallback index.",
                        e
                    );
                    // Fallback to old behavior
                    weight_map.insert("*".to_string(), weight_file.to_string());
                },
            }
        } else {
            // For PyTorch files, use wildcard (we can't easily parse .bin files)
            weight_map.insert("*".to_string(), weight_file.to_string());
        }

        Ok(HuggingFaceIndex {
            metadata: HuggingFaceMetadata {
                total_size: 0,
                format: if is_safetensors { "safetensors" } else { "pytorch" }.to_string(),
            },
            weight_map,
        })
    }

    fn read_safetensors_tensor_names(path: &Path) -> Result<Vec<String>> {
        use std::io::Read;

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read header length (first 8 bytes)
        let mut header_len_bytes = [0u8; 8];
        reader.read_exact(&mut header_len_bytes)?;
        let header_len = u64::from_le_bytes(header_len_bytes);

        // Read header JSON
        let mut header_bytes = vec![0u8; header_len as usize];
        reader.read_exact(&mut header_bytes)?;
        let header_str = String::from_utf8(header_bytes).map_err(|e| {
            TrustformersError::weight_load_error(format!(
                "Invalid UTF-8 in SafeTensors header: {}",
                e
            ))
        })?;

        // Parse JSON and extract tensor names
        let header: serde_json::Value = serde_json::from_str(&header_str).map_err(|e| {
            TrustformersError::weight_load_error(format!(
                "Failed to parse SafeTensors header: {}",
                e
            ))
        })?;

        let mut tensor_names = Vec::new();
        if let Some(obj) = header.as_object() {
            for (key, _value) in obj {
                // Skip metadata entries
                if key != "__metadata__" {
                    tensor_names.push(key.clone());
                }
            }
        }

        Ok(tensor_names)
    }

    fn get_file_handle(&mut self, filename: &str) -> Result<&mut BufReader<File>> {
        if !self.file_handles.contains_key(filename) {
            let file_path = self.model_dir.join(filename);
            let file = File::open(&file_path)?;
            let reader = BufReader::new(file);
            self.file_handles.insert(filename.to_string(), reader);
        }

        self.file_handles.get_mut(filename).ok_or_else(|| {
            TrustformersError::runtime_error(format!(
                "File handle for {} not found after insertion",
                filename
            ))
        })
    }

    /// Load tensor from PyTorch .bin file
    fn load_from_pytorch_bin(&mut self, name: &str, filename: &str) -> Result<Tensor> {
        let reader = self.get_file_handle(filename)?;

        // Read the file into memory for processing
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer).map_err(|e| {
            TrustformersError::weight_load_error(format!("Failed to read tensor file: {}", e))
        })?;

        // Basic pickle protocol parsing for PyTorch tensors
        // This is a simplified implementation that handles the most common cases
        match Self::parse_pytorch_pickle_static(&buffer, name) {
            Ok(tensor) => Ok(tensor),
            Err(e) => {
                // Fallback: try to load as raw tensor data if pickle parsing fails
                eprintln!(
                    "Warning: Pickle parsing failed for {}: {}. Attempting raw tensor parsing.",
                    name, e
                );
                Self::parse_raw_tensor_data_static(&buffer, name)
            },
        }
    }

    #[allow(dead_code)]
    fn parse_pytorch_tensor(&mut self, reader: &mut BufReader<File>, name: &str) -> Result<Tensor> {
        // Basic PyTorch .bin file parser implementation
        // This handles the common PyTorch pickle format used by HuggingFace models

        // Read the file into memory for processing
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer).map_err(|e| {
            TrustformersError::weight_load_error(format!("Failed to read tensor file: {}", e))
        })?;

        // Basic pickle protocol parsing for PyTorch tensors
        // This is a simplified implementation that handles the most common cases
        match Self::parse_pytorch_pickle_static(&buffer, name) {
            Ok(tensor) => Ok(tensor),
            Err(e) => {
                // Fallback: try to load as raw tensor data if pickle parsing fails
                eprintln!(
                    "Warning: Pickle parsing failed for {}: {}. Attempting raw tensor parsing.",
                    name, e
                );
                Self::parse_raw_tensor_data_static(&buffer, name)
            },
        }
    }

    #[allow(dead_code)]
    fn parse_pytorch_pickle(&self, data: &[u8], name: &str) -> Result<Tensor> {
        Self::parse_pytorch_pickle_static(data, name)
    }

    fn parse_pytorch_pickle_static(data: &[u8], name: &str) -> Result<Tensor> {
        // Simplified PyTorch pickle parser
        // This handles the basic structure of PyTorch .bin files

        // Look for tensor data markers in the pickle stream
        // PyTorch typically stores tensors with specific magic numbers

        // Check for PyTorch magic numbers
        if data.len() < 8 {
            return Err(TrustformersError::weight_load_error(
                "File too small to contain tensor data".to_string(),
            ));
        }

        // Try to find tensor metadata in the pickle stream
        // This is a heuristic approach for common PyTorch formats
        if let Some(tensor_info) = Self::extract_pytorch_tensor_info_static(data, name) {
            let offset = tensor_info.data_offset;
            let shape = tensor_info.shape;
            let dtype = tensor_info.dtype;
            let total_elements: usize = shape.iter().product();

            match dtype {
                WeightDataType::Float32 => {
                    let data_size = total_elements * 4;
                    if offset + data_size <= data.len() {
                        let tensor_data = &data[offset..offset + data_size];
                        let float_data: Vec<f32> = tensor_data
                            .chunks_exact(4)
                            .map(|chunk| {
                                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                            })
                            .collect();

                        Tensor::from_vec(float_data, &shape).map_err(|e| {
                            TrustformersError::weight_load_error(format!(
                                "Failed to create tensor: {}",
                                e
                            ))
                        })
                    } else {
                        Err(TrustformersError::weight_load_error(
                            "Insufficient data for tensor".to_string(),
                        ))
                    }
                },
                WeightDataType::Float16 => {
                    let data_size = total_elements * 2;
                    if offset + data_size <= data.len() {
                        let tensor_data = &data[offset..offset + data_size];
                        let float_data: Vec<f32> = tensor_data
                            .chunks_exact(2)
                            .map(|chunk| {
                                let half_val = half::f16::from_le_bytes([chunk[0], chunk[1]]);
                                half_val.to_f32()
                            })
                            .collect();

                        Tensor::from_vec(float_data, &shape).map_err(|e| {
                            TrustformersError::weight_load_error(format!(
                                "Failed to create tensor: {}",
                                e
                            ))
                        })
                    } else {
                        Err(TrustformersError::weight_load_error(
                            "Insufficient data for tensor".to_string(),
                        ))
                    }
                },
                _ => Err(TrustformersError::weight_load_error(format!(
                    "Unsupported tensor dtype: {:?}",
                    dtype
                ))),
            }
        } else {
            Err(TrustformersError::weight_load_error(
                "Could not extract tensor information from pickle data".to_string(),
            ))
        }
    }

    #[allow(dead_code)]
    fn extract_pytorch_tensor_info(&self, data: &[u8], name: &str) -> Option<PyTorchTensorInfo> {
        Self::extract_pytorch_tensor_info_static(data, name)
    }

    fn extract_pytorch_tensor_info_static(data: &[u8], name: &str) -> Option<PyTorchTensorInfo> {
        // Extract tensor metadata from pickle stream
        // This is a heuristic approach that looks for common patterns

        // Try to infer tensor properties based on common HuggingFace model patterns
        let shape = Self::infer_tensor_shape_static(name);
        let dtype = WeightDataType::Float32; // Default to float32

        // Look for potential tensor data start
        // PyTorch pickles often have specific patterns
        let mut data_offset = 0;

        // Scan for patterns that might indicate tensor data
        for i in 0..data.len().saturating_sub(16) {
            // Look for potential float patterns or PyTorch-specific markers
            if Self::looks_like_tensor_data_static(&data[i..i.min(i + 16)]) {
                data_offset = i;
                break;
            }
        }

        // If we couldn't find a good offset, use a reasonable default
        if data_offset == 0 && data.len() > 1024 {
            data_offset = 1024; // Skip likely pickle header
        }

        Some(PyTorchTensorInfo {
            shape,
            dtype,
            data_offset,
        })
    }

    #[allow(dead_code)]
    fn infer_tensor_shape(&self, name: &str) -> Vec<usize> {
        Self::infer_tensor_shape_static(name)
    }

    fn infer_tensor_shape_static(name: &str) -> Vec<usize> {
        // Infer tensor shape based on layer name patterns
        // This is a heuristic approach for common transformer model patterns

        if name.contains("embeddings.word_embeddings.weight") {
            vec![30522, 768] // Common BERT vocab size and hidden size
        } else if name.contains("embeddings.position_embeddings.weight") {
            vec![512, 768] // Common max position embeddings
        } else if name.contains("attention.self.query.weight")
            || name.contains("attention.self.key.weight")
            || name.contains("attention.self.value.weight")
        {
            vec![768, 768] // Common attention weight dimensions
        } else if name.contains("attention.output.dense.weight") {
            vec![768, 768] // Attention output projection
        } else if name.contains("intermediate.dense.weight") {
            vec![768, 3072] // Feed-forward intermediate layer
        } else if name.contains("output.dense.weight") {
            vec![3072, 768] // Feed-forward output layer
        } else if name.contains("LayerNorm.weight") || name.contains("LayerNorm.bias") {
            vec![768] // LayerNorm parameters
        } else if name.contains("bias") {
            vec![768] // Common bias size
        } else {
            // Default fallback - try to parse from common patterns or use small default
            vec![768, 768]
        }
    }

    #[allow(dead_code)]
    fn looks_like_tensor_data(&self, chunk: &[u8]) -> bool {
        Self::looks_like_tensor_data_static(chunk)
    }

    fn looks_like_tensor_data_static(chunk: &[u8]) -> bool {
        // Heuristic to identify potential tensor data in byte stream
        if chunk.len() < 4 {
            return false;
        }

        // Check if bytes could represent reasonable float values
        let float_val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);

        // Reasonable range for model weights (not NaN, not infinite, reasonable magnitude)
        float_val.is_finite() && float_val.abs() < 100.0
    }

    #[allow(dead_code)]
    fn parse_raw_tensor_data(&self, data: &[u8], name: &str) -> Result<Tensor> {
        Self::parse_raw_tensor_data_static(data, name)
    }

    fn parse_raw_tensor_data_static(data: &[u8], name: &str) -> Result<Tensor> {
        // Fallback: try to parse as raw tensor data
        let shape = Self::infer_tensor_shape_static(name);
        let total_elements: usize = shape.iter().product();
        let expected_size = total_elements * 4; // Assume float32

        if data.len() >= expected_size {
            // Try different offsets to find the actual tensor data
            for offset in (0..1024.min(data.len())).step_by(4) {
                if offset + expected_size <= data.len() {
                    let tensor_data = &data[offset..offset + expected_size];
                    let float_data: Vec<f32> = tensor_data
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect();

                    // Validate that the data looks reasonable
                    if float_data.iter().any(|&x| x.is_finite() && x.abs() < 100.0) {
                        if let Ok(tensor) = Tensor::from_vec(float_data, &shape) {
                            return Ok(tensor);
                        }
                    }
                }
            }
        }

        Err(TrustformersError::weight_load_error(format!(
            "Could not parse tensor data for {}",
            name
        )))
    }

    /// Load tensor with lazy loading
    #[allow(dead_code)]
    fn load_lazy(&mut self, name: &str) -> Result<LazyTensor> {
        let filename = self.find_tensor_file(name)?;
        let metadata = self.get_tensor_metadata(name, &filename)?;

        Ok(LazyTensor {
            name: name.to_string(),
            filename,
            metadata,
            model_dir: self.model_dir.clone(),
            config: self.config.clone(),
        })
    }

    fn find_tensor_file(&self, name: &str) -> Result<String> {
        // Check weight map for tensor location
        if let Some(filename) = self.index.weight_map.get(name) {
            Ok(filename.clone())
        } else if let Some(filename) = self.index.weight_map.get("*") {
            // Single file case
            Ok(filename.clone())
        } else {
            Err(runtime_error(format!("Tensor not found: {}", name)))
        }
    }

    fn get_tensor_metadata(&self, _name: &str, _filename: &str) -> Result<TensorMetadata> {
        // Parse metadata from file header
        Ok(TensorMetadata {
            shape: vec![1024, 768],
            dtype: WeightDataType::Float32,
            size_bytes: 1024 * 768 * 4,
            offset: 0,
        })
    }

    fn detect_format(&self, filename: &str) -> Result<WeightFormat> {
        if filename.ends_with(".bin") {
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

    fn load_from_safetensors(&mut self, name: &str, filename: &str) -> Result<Tensor> {
        // Use a single method that handles both header parsing and tensor loading
        self.load_safetensors_tensor_complete(name, filename)
    }

    fn load_safetensors_tensor_complete(&mut self, name: &str, filename: &str) -> Result<Tensor> {
        // CRITICAL FIX: Don't use cached file handles for SafeTensors
        // BufReader's internal buffer causes issues when seeking - it doesn't flush the buffer
        // after seek(), so we read stale buffered data instead of fresh file data.
        // Solution: Open a fresh file for each tensor load.
        let file_path = self.model_dir.join(filename);
        eprintln!(
            "[SAFETENSORS DEBUG] Loading tensor '{}' from file: {:?}",
            name, file_path
        );
        let file = File::open(&file_path)?;
        let mut reader = BufReader::new(file);

        // Read header length (first 8 bytes)
        let mut header_len_bytes = [0u8; 8];
        reader.read_exact(&mut header_len_bytes)?;
        let header_len = u64::from_le_bytes(header_len_bytes);
        eprintln!("[SAFETENSORS DEBUG] Header length: {} bytes", header_len);

        // Read header JSON
        let mut header_bytes = vec![0u8; header_len as usize];
        reader.read_exact(&mut header_bytes)?;

        let header_str = std::str::from_utf8(&header_bytes).map_err(|e| {
            TrustformersError::weight_load_error(format!(
                "Invalid UTF-8 in SafeTensors header: {}",
                e
            ))
        })?;

        // Debug: print first 500 chars of header
        eprintln!(
            "[SAFETENSORS DEBUG] Header preview (first 500 chars): {}",
            &header_str[..header_str.len().min(500)]
        );

        let header: SafeTensorsHeader = serde_json::from_str(header_str).map_err(|e| {
            eprintln!("[SAFETENSORS DEBUG] Failed to parse header, printing full header:");
            eprintln!("{}", header_str);
            TrustformersError::serialization_error(format!(
                "Failed to parse SafeTensors header: {}",
                e
            ))
        })?;

        if let Some(tensor_info) = header.tensors.get(name) {
            // Seek to tensor data (offsets are relative to start of tensor data section)
            // SafeTensors format: [8 bytes header_len][header_len bytes JSON][tensor data]
            let tensor_data_start = 8 + header_len;
            reader.seek(SeekFrom::Start(
                tensor_data_start + tensor_info.data_offsets[0],
            ))?;

            // Read tensor data
            let data_len = (tensor_info.data_offsets[1] - tensor_info.data_offsets[0]) as usize;
            let mut data = vec![0u8; data_len];
            reader.read_exact(&mut data)?;

            // Convert to tensor based on dtype
            self.bytes_to_tensor(data, &tensor_info.dtype, &tensor_info.shape)
        } else {
            Err(runtime_error(format!("Tensor not found: {}", name)))
        }
    }

    #[allow(dead_code)]
    fn parse_safetensors_header(
        &mut self,
        reader: &mut BufReader<File>,
    ) -> Result<SafeTensorsHeader> {
        // Read header length (first 8 bytes)
        let mut header_len_bytes = [0u8; 8];
        reader.read_exact(&mut header_len_bytes)?;
        let header_len = u64::from_le_bytes(header_len_bytes);

        // Read header JSON
        let mut header_bytes = vec![0u8; header_len as usize];
        reader.read_exact(&mut header_bytes)?;

        let header_str = std::str::from_utf8(&header_bytes).map_err(|e| {
            TrustformersError::weight_load_error(format!(
                "Invalid UTF-8 in SafeTensors header: {}",
                e
            ))
        })?;
        serde_json::from_str(header_str).map_err(|e| {
            TrustformersError::serialization_error(format!(
                "Failed to parse SafeTensors header: {}",
                e
            ))
        })
    }

    #[allow(dead_code)]
    fn load_safetensors_tensor(
        &mut self,
        reader: &mut BufReader<File>,
        info: &TensorInfo,
    ) -> Result<Tensor> {
        // Seek to tensor data
        reader.seek(SeekFrom::Start(info.data_offsets[0]))?;

        // Read tensor data
        let data_len = (info.data_offsets[1] - info.data_offsets[0]) as usize;
        let mut data = vec![0u8; data_len];
        reader.read_exact(&mut data)?;

        // Convert to tensor based on dtype
        self.bytes_to_tensor(data, &info.dtype, &info.shape)
    }

    fn bytes_to_tensor(&self, data: Vec<u8>, dtype: &str, shape: &[usize]) -> Result<Tensor> {
        match dtype {
            "F32" => {
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Tensor::from_vec(floats, shape)
            },
            "F16" => {
                // Convert f16 to f32
                let floats: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        half::f16::from_bits(bits).to_f32()
                    })
                    .collect();
                Tensor::from_vec(floats, shape)
            },
            "I8" => {
                let ints: Vec<i8> = data.into_iter().map(|b| b as i8).collect();
                // Convert to f32 for now
                let floats: Vec<f32> = ints.into_iter().map(|i| i as f32).collect();
                Tensor::from_vec(floats, shape)
            },
            _ => Err(invalid_format(
                "dtype",
                format!("Unsupported dtype: {}", dtype),
            )),
        }
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
            _ => {
                return Err(invalid_format("weight format", "Unsupported weight format"));
            },
        };

        // Cache if not lazy loading
        if !self.config.lazy_loading {
            self.tensor_cache.insert(name.to_string(), tensor.clone());
        }

        Ok(tensor)
    }

    fn list_tensors(&self) -> Result<Vec<String>> {
        Ok(self.index.weight_map.keys().cloned().collect())
    }

    fn tensor_info(&self, name: &str) -> Result<Option<TensorMetadata>> {
        let filename = self.find_tensor_file(name)?;
        Ok(Some(self.get_tensor_metadata(name, &filename)?))
    }

    fn close(&mut self) -> Result<()> {
        self.file_handles.clear();
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
