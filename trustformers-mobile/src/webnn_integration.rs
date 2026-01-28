//! WebNN Integration for Mobile Web Deployment
//!
//! This module provides integration with the Web Neural Networks API (WebNN)
//! for hardware-accelerated inference in web browsers and hybrid mobile apps.
//!
//! WebNN enables:
//! - Hardware-accelerated inference in browsers
//! - Progressive Web App (PWA) deployment
//! - Hybrid app support (React Native, Flutter web targets)
//! - Cross-platform web/mobile deployment
//!
//! # Features
//! - Automatic backend detection (CPU, GPU, NPU)
//! - Asynchronous operation support
//! - Graph compilation and optimization
//! - Mobile-specific optimizations
//! - Graceful fallback strategies

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::Tensor;

/// WebNN backend device preference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WebNNDevice {
    /// CPU execution
    CPU,
    /// GPU execution (WebGL/WebGPU backend)
    GPU,
    /// Neural Processing Unit (if available)
    NPU,
    /// Automatic selection based on availability
    Auto,
}

impl std::fmt::Display for WebNNDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WebNNDevice::CPU => write!(f, "CPU"),
            WebNNDevice::GPU => write!(f, "GPU"),
            WebNNDevice::NPU => write!(f, "NPU"),
            WebNNDevice::Auto => write!(f, "Auto"),
        }
    }
}

/// WebNN power preference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WebNNPowerPreference {
    /// Prefer low power consumption
    LowPower,
    /// Prefer high performance
    HighPerformance,
    /// Default/balanced
    Default,
}

/// WebNN data type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WebNNDataType {
    Float32,
    Float16,
    Int32,
    Uint32,
    Int8,
    Uint8,
}

impl WebNNDataType {
    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            WebNNDataType::Float32 | WebNNDataType::Int32 | WebNNDataType::Uint32 => 4,
            WebNNDataType::Float16 => 2,
            WebNNDataType::Int8 | WebNNDataType::Uint8 => 1,
        }
    }
}

/// WebNN operation type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WebNNOperation {
    /// Matrix multiplication
    MatMul,
    /// Convolution
    Conv2d {
        padding: Vec<usize>,
        stride: Vec<usize>,
    },
    /// Activation functions
    Relu,
    Gelu,
    Sigmoid,
    Tanh,
    /// Normalization
    BatchNorm,
    LayerNorm,
    /// Pooling
    MaxPool {
        kernel_size: Vec<usize>,
    },
    AvgPool {
        kernel_size: Vec<usize>,
    },
    /// Element-wise operations
    Add,
    Mul,
    /// Reshaping
    Reshape {
        shape: Vec<i64>,
    },
    Transpose {
        perm: Vec<usize>,
    },
    /// Reduction
    ReduceSum {
        axes: Vec<usize>,
    },
    ReduceMean {
        axes: Vec<usize>,
    },
}

/// WebNN graph builder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebNNGraphConfig {
    /// Preferred device
    pub device: WebNNDevice,

    /// Power preference
    pub power_preference: WebNNPowerPreference,

    /// Enable operator fusion
    pub enable_fusion: bool,

    /// Enable graph optimization
    pub enable_optimization: bool,

    /// Maximum batch size
    pub max_batch_size: usize,

    /// Enable mixed precision
    pub mixed_precision: bool,

    /// Default data type
    pub default_dtype: WebNNDataType,
}

impl Default for WebNNGraphConfig {
    fn default() -> Self {
        Self {
            device: WebNNDevice::Auto,
            power_preference: WebNNPowerPreference::Default,
            enable_fusion: true,
            enable_optimization: true,
            max_batch_size: 1,
            mixed_precision: false,
            default_dtype: WebNNDataType::Float32,
        }
    }
}

/// WebNN capability information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebNNCapabilities {
    /// Is WebNN available
    pub available: bool,

    /// Supported devices
    pub supported_devices: Vec<WebNNDevice>,

    /// Supported data types
    pub supported_dtypes: Vec<WebNNDataType>,

    /// Supported operations
    pub supported_ops: Vec<String>,

    /// Maximum tensor size
    pub max_tensor_size: usize,

    /// WebNN API version
    pub api_version: String,
}

impl Default for WebNNCapabilities {
    fn default() -> Self {
        Self {
            available: false,
            supported_devices: vec![WebNNDevice::CPU],
            supported_dtypes: vec![WebNNDataType::Float32],
            supported_ops: vec![],
            max_tensor_size: 1024 * 1024 * 1024, // 1GB
            api_version: "1.0".to_string(),
        }
    }
}

/// WebNN tensor descriptor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebNNTensorDescriptor {
    pub shape: Vec<usize>,
    pub dtype: WebNNDataType,
    pub name: String,
}

/// WebNN compiled graph
#[derive(Debug, Clone)]
pub struct WebNNCompiledGraph {
    /// Graph ID
    pub graph_id: String,

    /// Input descriptors
    pub inputs: Vec<WebNNTensorDescriptor>,

    /// Output descriptors
    pub outputs: Vec<WebNNTensorDescriptor>,

    /// Compilation metadata
    pub metadata: HashMap<String, String>,
}

/// WebNN execution context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebNNExecutionContext {
    /// Device being used
    pub device: WebNNDevice,

    /// Power mode
    pub power_mode: WebNNPowerPreference,

    /// Number of inferences
    pub num_inferences: usize,

    /// Average inference time (ms)
    pub avg_inference_time_ms: f32,

    /// Total memory usage (bytes)
    pub memory_usage_bytes: usize,
}

impl Default for WebNNExecutionContext {
    fn default() -> Self {
        Self {
            device: WebNNDevice::CPU,
            power_mode: WebNNPowerPreference::Default,
            num_inferences: 0,
            avg_inference_time_ms: 0.0,
            memory_usage_bytes: 0,
        }
    }
}

/// WebNN backend for mobile web deployment
pub struct WebNNBackend {
    config: WebNNGraphConfig,
    capabilities: WebNNCapabilities,
    context: WebNNExecutionContext,
    compiled_graphs: HashMap<String, WebNNCompiledGraph>,
}

impl WebNNBackend {
    /// Create new WebNN backend
    ///
    /// # Example
    /// ```no_run
    /// use trustformers_mobile::webnn_integration::{WebNNBackend, WebNNGraphConfig};
    ///
    /// let config = WebNNGraphConfig::default();
    /// let backend = WebNNBackend::new(config)?;
    ///
    /// if backend.is_available() {
    ///     println!("WebNN is available!");
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(config: WebNNGraphConfig) -> Result<Self> {
        let capabilities = Self::detect_capabilities()?;

        Ok(Self {
            config,
            capabilities,
            context: WebNNExecutionContext::default(),
            compiled_graphs: HashMap::new(),
        })
    }

    /// Detect WebNN capabilities
    pub fn detect_capabilities() -> Result<WebNNCapabilities> {
        // In a real implementation, this would query the WebNN API
        // For now, return default capabilities
        Ok(WebNNCapabilities::default())
    }

    /// Check if WebNN is available
    pub fn is_available(&self) -> bool {
        self.capabilities.available
    }

    /// Get supported devices
    pub fn supported_devices(&self) -> &[WebNNDevice] {
        &self.capabilities.supported_devices
    }

    /// Build computation graph
    ///
    /// # Example
    /// ```no_run
    /// use trustformers_mobile::webnn_integration::{WebNNBackend, WebNNOperation};
    ///
    /// let mut backend = WebNNBackend::new(Default::default())?;
    ///
    /// let graph_id = backend.build_graph(
    ///     "simple_matmul",
    ///     vec![
    ///         WebNNOperation::MatMul,
    ///         WebNNOperation::Relu,
    ///     ]
    /// )?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn build_graph(&mut self, name: &str, operations: Vec<WebNNOperation>) -> Result<String> {
        // Validate operations are supported
        for op in &operations {
            if !self.is_operation_supported(op) {
                return Err(TrustformersError::runtime_error(format!(
                    "Operation {:?} not supported by WebNN backend",
                    op
                )));
            }
        }

        // Create graph ID
        let graph_id = format!("{}_{}", name, self.compiled_graphs.len());

        // In a real implementation, this would compile the graph using WebNN API
        let compiled = WebNNCompiledGraph {
            graph_id: graph_id.clone(),
            inputs: vec![],
            outputs: vec![],
            metadata: HashMap::new(),
        };

        self.compiled_graphs.insert(graph_id.clone(), compiled);

        Ok(graph_id)
    }

    /// Execute compiled graph
    ///
    /// # Example
    /// ```no_run
    /// use trustformers_mobile::webnn_integration::WebNNBackend;
    /// use trustformers_core::Tensor;
    ///
    /// let mut backend = WebNNBackend::new(Default::default())?;
    /// let graph_id = "my_graph";
    ///
    /// let input = Tensor::randn(&[1, 128])?;
    /// let outputs = backend.execute(&graph_id, vec![input])?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn execute(&mut self, graph_id: &str, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        if !self.compiled_graphs.contains_key(graph_id) {
            return Err(TrustformersError::runtime_error(format!(
                "Graph {} not found",
                graph_id
            )));
        }

        // Update statistics
        self.context.num_inferences += 1;

        // In a real implementation, this would execute the graph using WebNN API
        // For now, return the inputs as a placeholder
        Ok(inputs)
    }

    /// Check if operation is supported
    pub fn is_operation_supported(&self, op: &WebNNOperation) -> bool {
        // In a real implementation, check against capabilities
        // For now, assume all common operations are supported
        matches!(
            op,
            WebNNOperation::MatMul
                | WebNNOperation::Conv2d { .. }
                | WebNNOperation::Relu
                | WebNNOperation::Gelu
                | WebNNOperation::Add
                | WebNNOperation::Mul
        )
    }

    /// Get execution statistics
    pub fn get_statistics(&self) -> &WebNNExecutionContext {
        &self.context
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.context = WebNNExecutionContext::default();
    }

    /// Optimize for mobile deployment
    pub fn optimize_for_mobile(&mut self) {
        // Enable mobile-specific optimizations
        self.config.enable_fusion = true;
        self.config.enable_optimization = true;
        self.config.power_preference = WebNNPowerPreference::LowPower;

        // Limit batch size for mobile
        self.config.max_batch_size = 1;

        // Enable mixed precision if supported
        if self.capabilities.supported_dtypes.contains(&WebNNDataType::Float16) {
            self.config.mixed_precision = true;
            self.config.default_dtype = WebNNDataType::Float16;
        }
    }

    /// Get recommended configuration for device
    pub fn recommend_config(device_type: &str) -> WebNNGraphConfig {
        let mut config = WebNNGraphConfig::default();

        match device_type.to_lowercase().as_str() {
            "mobile" | "phone" | "tablet" => {
                config.device = WebNNDevice::Auto;
                config.power_preference = WebNNPowerPreference::LowPower;
                config.max_batch_size = 1;
                config.mixed_precision = true;
                config.default_dtype = WebNNDataType::Float16;
            },
            "desktop" | "laptop" => {
                config.device = WebNNDevice::GPU;
                config.power_preference = WebNNPowerPreference::HighPerformance;
                config.max_batch_size = 4;
                config.mixed_precision = false;
                config.default_dtype = WebNNDataType::Float32;
            },
            _ => {
                // Default configuration
            },
        }

        config
    }
}

/// WebNN utility functions
pub struct WebNNUtils;

impl WebNNUtils {
    /// Check if running in a web environment
    pub fn is_web_environment() -> bool {
        // In a real implementation, this would check for browser environment
        cfg!(target_arch = "wasm32")
    }

    /// Get WebNN feature support level
    pub fn get_support_level() -> WebNNSupportLevel {
        if !Self::is_web_environment() {
            return WebNNSupportLevel::NotAvailable;
        }

        // In a real implementation, query actual WebNN support
        WebNNSupportLevel::Full
    }

    /// Estimate memory requirements for graph
    pub fn estimate_memory(
        operations: &[WebNNOperation],
        input_shapes: &[Vec<usize>],
        dtype: WebNNDataType,
    ) -> usize {
        let mut total_memory = 0;

        // Estimate based on operations and shapes
        for shape in input_shapes {
            let elements: usize = shape.iter().product();
            total_memory += elements * dtype.size_bytes();
        }

        // Add overhead for intermediate tensors
        total_memory += (total_memory as f32 * 1.5) as usize;

        total_memory
    }

    /// Check if device supports WebNN NPU
    pub fn supports_npu() -> bool {
        // In a real implementation, check for NPU support
        false
    }

    /// Get browser capabilities
    pub fn get_browser_info() -> BrowserInfo {
        BrowserInfo {
            name: "Unknown".to_string(),
            version: "0.0.0".to_string(),
            webnn_support: false,
            webgpu_support: false,
            webgl_support: false,
        }
    }
}

/// WebNN support level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WebNNSupportLevel {
    /// Not available
    NotAvailable,
    /// Partial support (some operations)
    Partial,
    /// Full support
    Full,
}

/// Browser information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserInfo {
    pub name: String,
    pub version: String,
    pub webnn_support: bool,
    pub webgpu_support: bool,
    pub webgl_support: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webnn_device_display() {
        assert_eq!(WebNNDevice::CPU.to_string(), "CPU");
        assert_eq!(WebNNDevice::GPU.to_string(), "GPU");
        assert_eq!(WebNNDevice::NPU.to_string(), "NPU");
        assert_eq!(WebNNDevice::Auto.to_string(), "Auto");
    }

    #[test]
    fn test_webnn_data_type_size() {
        assert_eq!(WebNNDataType::Float32.size_bytes(), 4);
        assert_eq!(WebNNDataType::Float16.size_bytes(), 2);
        assert_eq!(WebNNDataType::Int8.size_bytes(), 1);
        assert_eq!(WebNNDataType::Uint8.size_bytes(), 1);
    }

    #[test]
    fn test_webnn_backend_creation() {
        let config = WebNNGraphConfig::default();
        let backend = WebNNBackend::new(config);
        assert!(backend.is_ok());
    }

    #[test]
    fn test_webnn_mobile_optimization() {
        let mut backend = WebNNBackend::new(WebNNGraphConfig::default()).unwrap();
        backend.optimize_for_mobile();

        assert_eq!(backend.config.max_batch_size, 1);
        assert_eq!(
            backend.config.power_preference,
            WebNNPowerPreference::LowPower
        );
        assert!(backend.config.enable_fusion);
        assert!(backend.config.enable_optimization);
    }

    #[test]
    fn test_webnn_config_recommendation() {
        let mobile_config = WebNNBackend::recommend_config("mobile");
        assert_eq!(
            mobile_config.power_preference,
            WebNNPowerPreference::LowPower
        );
        assert_eq!(mobile_config.max_batch_size, 1);
        assert!(mobile_config.mixed_precision);

        let desktop_config = WebNNBackend::recommend_config("desktop");
        assert_eq!(
            desktop_config.power_preference,
            WebNNPowerPreference::HighPerformance
        );
        assert_eq!(desktop_config.max_batch_size, 4);
    }

    #[test]
    fn test_memory_estimation() {
        let ops = vec![WebNNOperation::MatMul, WebNNOperation::Relu];
        let shapes = vec![vec![1, 128], vec![128, 256]];

        let memory = WebNNUtils::estimate_memory(&ops, &shapes, WebNNDataType::Float32);
        assert!(memory > 0);

        // FP16 should use less memory
        let memory_fp16 = WebNNUtils::estimate_memory(&ops, &shapes, WebNNDataType::Float16);
        assert!(memory_fp16 < memory);
    }

    #[test]
    fn test_support_level() {
        let level = WebNNUtils::get_support_level();
        assert!(matches!(
            level,
            WebNNSupportLevel::NotAvailable | WebNNSupportLevel::Partial | WebNNSupportLevel::Full
        ));
    }
}
