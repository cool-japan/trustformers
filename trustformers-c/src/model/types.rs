//! Model type definitions for TrustformeRS C API

use std::os::raw::{c_char, c_int};
use std::ptr;

/// C-compatible model handle
pub type TrustformersModel = usize;

/// C-compatible tensor handle
pub type TrustformersTensor = usize;

/// C-compatible model configuration
#[repr(C)]
#[derive(Debug)]
pub struct TrustformersModelConfig {
    /// Model name or path
    pub model_name: *const c_char,
    /// Cache directory
    pub cache_dir: *const c_char,
    /// Whether to use authentication token
    pub use_auth_token: c_int,
    /// Authentication token
    pub auth_token: *const c_char,
    /// Device type: 0=CPU, 1=CUDA
    pub device_type: c_int,
    /// Device ID for multi-GPU setups
    pub device_id: c_int,
    /// Precision type: 0=float32, 1=float16, 2=bfloat16
    pub precision_type: c_int,
    /// Whether to use safetensors format
    pub use_safetensors: c_int,
    /// Model format: 0=Auto, 1=PyTorch, 2=ONNX, 3=TensorFlow, 4=Safetensors, 5=GGML
    pub model_format: c_int,
    /// Quantization config
    pub quantization_config: *const TrustformersQuantizationConfig,
    /// Whether to validate model integrity
    pub validate_model: c_int,
    /// Whether to trust remote code
    pub trust_remote_code: c_int,
}

/// C-compatible quantization configuration
#[repr(C)]
#[derive(Debug)]
pub struct TrustformersQuantizationConfig {
    /// Quantization type: 0=None, 1=Dynamic, 2=Static, 3=QAT
    pub quantization_type: c_int,
    /// Quantization bits: 8, 16, 32
    pub bits: c_int,
    /// Whether to quantize weights
    pub quantize_weights: c_int,
    /// Whether to quantize activations
    pub quantize_activations: c_int,
    /// Calibration dataset path (for static quantization)
    pub calibration_dataset: *const c_char,
    /// Number of calibration samples
    pub calibration_samples: c_int,
    /// Quantization backend: 0=Default, 1=FBGEMM, 2=QNNPACK, 3=ONEDNN
    pub backend: c_int,
}

impl Default for TrustformersModelConfig {
    fn default() -> Self {
        Self {
            model_name: ptr::null(),
            cache_dir: ptr::null(),
            use_auth_token: 0,
            auth_token: ptr::null(),
            device_type: 0, // CPU
            device_id: 0,
            precision_type: 0,  // float32
            use_safetensors: 1, // True
            model_format: 0,    // Auto
            quantization_config: ptr::null(),
            validate_model: 1,    // True
            trust_remote_code: 0, // False
        }
    }
}

impl Default for TrustformersQuantizationConfig {
    fn default() -> Self {
        Self {
            quantization_type: 0, // None
            bits: 32,
            quantize_weights: 0,
            quantize_activations: 0,
            calibration_dataset: ptr::null(),
            calibration_samples: 100,
            backend: 0, // Default
        }
    }
}

/// C-compatible tensor information
#[repr(C)]
#[derive(Debug)]
pub struct TrustformersTensorInfo {
    /// Shape of the tensor
    pub shape: *mut usize,
    /// Number of dimensions
    pub ndim: usize,
    /// Data type: 0=float32, 1=float64, 2=int32, 3=int64, 4=uint8, 5=bool
    pub dtype: c_int,
    /// Total number of elements
    pub numel: usize,
    /// Size in bytes
    pub size_bytes: usize,
    /// Device type: 0=CPU, 1=CUDA
    pub device_type: c_int,
    /// Device ID
    pub device_id: c_int,
}

impl Default for TrustformersTensorInfo {
    fn default() -> Self {
        Self {
            shape: ptr::null_mut(),
            ndim: 0,
            dtype: 0,
            numel: 0,
            size_bytes: 0,
            device_type: 0,
            device_id: 0,
        }
    }
}

/// C-compatible model metadata
#[repr(C)]
#[derive(Debug)]
pub struct TrustformersModelMetadata {
    /// Model architecture
    pub architecture: *mut c_char,
    /// Number of parameters
    pub num_parameters: u64,
    /// Model size in bytes
    pub model_size_bytes: u64,
    /// Vocab size
    pub vocab_size: u32,
    /// Hidden size
    pub hidden_size: u32,
    /// Number of layers
    pub num_layers: u32,
    /// Number of attention heads
    pub num_attention_heads: u32,
    /// Maximum sequence length
    pub max_sequence_length: u32,
    /// Model format
    pub model_format: *mut c_char,
    /// Framework (PyTorch, TensorFlow, etc.)
    pub framework: *mut c_char,
    /// Model version
    pub model_version: *mut c_char,
    /// License information
    pub license: *mut c_char,
    /// Whether model is quantized
    pub is_quantized: c_int,
    /// Quantization info
    pub quantization_info: *mut c_char,
}

/// Model validation result
#[repr(C)]
#[derive(Debug)]
pub struct TrustformersModelValidation {
    /// Whether validation passed
    pub is_valid: c_int,
    /// Validation errors (JSON string)
    pub errors: *mut c_char,
    /// Validation warnings (JSON string)
    pub warnings: *mut c_char,
    /// Checksum validation result
    pub checksum_valid: c_int,
    /// File integrity check result
    pub file_integrity_valid: c_int,
    /// Configuration validity
    pub config_valid: c_int,
}

impl Default for TrustformersModelMetadata {
    fn default() -> Self {
        Self {
            architecture: ptr::null_mut(),
            num_parameters: 0,
            model_size_bytes: 0,
            vocab_size: 0,
            hidden_size: 0,
            num_layers: 0,
            num_attention_heads: 0,
            max_sequence_length: 0,
            model_format: ptr::null_mut(),
            framework: ptr::null_mut(),
            model_version: ptr::null_mut(),
            license: ptr::null_mut(),
            is_quantized: 0,
            quantization_info: ptr::null_mut(),
        }
    }
}

impl Default for TrustformersModelValidation {
    fn default() -> Self {
        Self {
            is_valid: 0,
            errors: ptr::null_mut(),
            warnings: ptr::null_mut(),
            checksum_valid: 0,
            file_integrity_valid: 0,
            config_valid: 0,
        }
    }
}
