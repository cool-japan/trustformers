//! TensorFlow Lite NNAPI Delegate for Android Integration
//!
//! This module provides a TensorFlow Lite delegate that leverages the Android Neural Networks API (NNAPI)
//! for hardware-accelerated inference, integrating with our existing NNAPI backend.

#[cfg(all(target_os = "android", feature = "tflite-nnapi"))]
use crate::nnapi::{NNAPIConfig, NNAPIEngine, NNAPIDeviceType, NNAPIExecutionPreference};
#[cfg(all(target_os = "android", feature = "tflite-nnapi"))]
use crate::{MemoryOptimization, MobileConfig};
#[cfg(all(target_os = "android", feature = "tflite-nnapi"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_os = "android", feature = "tflite-nnapi"))]
use std::collections::HashMap;
#[cfg(all(target_os = "android", feature = "tflite-nnapi"))]
use std::ffi::{c_void, CStr, CString};
#[cfg(all(target_os = "android", feature = "tflite-nnapi"))]
use std::ptr;
#[cfg(all(target_os = "android", feature = "tflite-nnapi"))]
use std::time::Instant;
#[cfg(all(target_os = "android", feature = "tflite-nnapi"))]
use trustformers_core::{Tensor};
use trustformers_core::error::{CoreError, Result};

/// TensorFlow Lite NNAPI delegate
#[cfg(all(target_os = "android", feature = "tflite-nnapi"))]
pub struct TfLiteNNAPIDelegate {
    config: TfLiteNNAPIConfig,
    nnapi_engine: NNAPIEngine,
    delegate_handle: Option<*mut c_void>,
    interpreter_handle: Option<*mut c_void>,
    supported_ops: Vec<i32>,
    stats: TfLiteNNAPIStats,
}

/// TensorFlow Lite NNAPI delegate configuration
#[cfg(all(target_os = "android", feature = "tflite-nnapi"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TfLiteNNAPIConfig {
    /// NNAPI backend configuration
    pub nnapi_config: NNAPIConfig,
    /// Enable TensorFlow Lite delegate optimization
    pub enable_delegate_optimization: bool,
    /// Use NNAPI for supported operations only
    pub nnapi_fallback_enabled: bool,
    /// Maximum number of NNAPI delegate partitions
    pub max_partitions: usize,
    /// Enable delegate caching
    pub enable_delegate_caching: bool,
    /// Delegate compilation timeout (ms)
    pub compilation_timeout_ms: u32,
    /// Enable verbose logging for delegate operations
    pub enable_verbose_logging: bool,
    /// Use accelerator name instead of device type
    pub accelerator_name: Option<String>,
    /// Enable CPU fallback for unsupported operations
    pub enable_cpu_fallback: bool,
}

/// TensorFlow Lite NNAPI delegate statistics
#[cfg(all(target_os = "android", feature = "tflite-nnapi"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TfLiteNNAPIStats {
    /// Total delegate invocations
    pub total_invocations: usize,
    /// Operations delegated to NNAPI
    pub delegated_operations: usize,
    /// Operations executed on CPU fallback
    pub cpu_fallback_operations: usize,
    /// Average delegate overhead (ms)
    pub avg_delegate_overhead_ms: f32,
    /// Delegate initialization time (ms)
    pub delegate_init_time_ms: f32,
    /// Partition creation time (ms)
    pub partition_creation_time_ms: f32,
    /// NNAPI execution efficiency
    pub nnapi_efficiency_percent: f32,
    /// Memory usage during delegation (MB)
    pub delegation_memory_mb: usize,
}

/// TensorFlow Lite operation support info
#[cfg(all(target_os = "android", feature = "tflite-nnapi"))]
#[derive(Debug, Clone)]
pub struct TfLiteOpSupport {
    /// Operation code
    pub op_code: i32,
    /// Operation name
    pub op_name: String,
    /// Whether NNAPI supports this operation
    pub nnapi_supported: bool,
    /// Reason if not supported
    pub unsupported_reason: Option<String>,
    /// Expected performance improvement
    pub performance_gain_factor: f32,
}

#[cfg(all(target_os = "android", feature = "tflite-nnapi"))]
impl TfLiteNNAPIDelegate {
    /// Create new TensorFlow Lite NNAPI delegate
    pub fn new(config: TfLiteNNAPIConfig) -> Result<Self> {
        config.validate()?;

        let nnapi_engine = NNAPIEngine::new(config.nnapi_config.clone())?;
        let stats = TfLiteNNAPIStats::new();

        Ok(Self {
            config,
            nnapi_engine,
            delegate_handle: None,
            interpreter_handle: None,
            supported_ops: Vec::new(),
            stats,
        })
    }

    /// Initialize delegate with TensorFlow Lite interpreter
    pub fn initialize(&mut self, interpreter: *mut c_void) -> Result<()> {
        let start_time = Instant::now();

        self.interpreter_handle = Some(interpreter);

        // Create NNAPI delegate handle
        let delegate = self.create_nnapi_delegate()?;
        self.delegate_handle = Some(delegate);

        // Apply delegate to interpreter
        self.apply_delegate_to_interpreter(interpreter, delegate)?;

        // Analyze operation support
        self.analyze_operation_support(interpreter)?;

        let init_time = start_time.elapsed().as_millis() as f32;
        self.stats.delegate_init_time_ms = init_time;

        tracing::info!(
            "TensorFlow Lite NNAPI delegate initialized in {:.2}ms with {} supported operations",
            init_time,
            self.supported_ops.len()
        );

        Ok(())
    }

    /// Invoke delegate for inference
    pub fn invoke(&mut self) -> Result<()> {
        if self.interpreter_handle.is_none() {
            return Err(TrustformersError::runtime_error("Delegate not initialized".into()).into());
        }

        let start_time = Instant::now();

        // Invoke TensorFlow Lite interpreter with NNAPI delegate
        let interpreter = self.interpreter_handle.expect("Operation failed");
        self.invoke_interpreter(interpreter)?;

        let invocation_time = start_time.elapsed().as_millis() as f32;
        self.stats.update_invocation(invocation_time);

        Ok(())
    }

    /// Get input tensor information
    pub fn get_input_tensor_info(&self, index: i32) -> Result<TensorInfo> {
        if let Some(interpreter) = self.interpreter_handle {
            self.get_tensor_info(interpreter, index, true)
        } else {
            Err(TrustformersError::runtime_error("Delegate not initialized".into()).into())
        }
    }

    /// Get output tensor information
    pub fn get_output_tensor_info(&self, index: i32) -> Result<TensorInfo> {
        if let Some(interpreter) = self.interpreter_handle {
            self.get_tensor_info(interpreter, index, false)
        } else {
            Err(TrustformersError::runtime_error("Delegate not initialized".into()).into())
        }
    }

    /// Set input tensor data
    pub fn set_input_tensor(&mut self, index: i32, data: &[f32]) -> Result<()> {
        if let Some(interpreter) = self.interpreter_handle {
            self.set_tensor_data(interpreter, index, data)
        } else {
            Err(TrustformersError::runtime_error("Delegate not initialized".into()).into())
        }
    }

    /// Get output tensor data
    pub fn get_output_tensor(&self, index: i32) -> Result<Vec<f32>> {
        if let Some(interpreter) = self.interpreter_handle {
            self.get_tensor_data(interpreter, index)
        } else {
            Err(TrustformersError::runtime_error("Delegate not initialized".into()).into())
        }
    }

    /// Get delegate statistics
    pub fn get_stats(&self) -> &TfLiteNNAPIStats {
        &self.stats
    }

    /// Get operation support analysis
    pub fn get_operation_support(&self) -> Vec<TfLiteOpSupport> {
        if let Some(interpreter) = self.interpreter_handle {
            self.analyze_all_operations(interpreter).unwrap_or_default()
        } else {
            Vec::new()
        }
    }

    /// Optimize delegate configuration for current model
    pub fn optimize_for_model(&mut self) -> Result<()> {
        if self.interpreter_handle.is_none() {
            return Err(TrustformersError::runtime_error("Delegate not initialized".into()).into());
        }

        let op_support = self.get_operation_support();
        let nnapi_supported_count = op_support.iter().filter(|op| op.nnapi_supported).count();
        let total_ops = op_support.len();

        if total_ops == 0 {
            return Ok(());
        }

        let support_ratio = nnapi_supported_count as f32 / total_ops as f32;

        // Adjust configuration based on NNAPI support ratio
        if support_ratio > 0.8 {
            // High NNAPI support - optimize for performance
            self.config.nnapi_config.execution_preference = NNAPIExecutionPreference::SustainedSpeed;
            self.config.max_partitions = 8;
            self.config.enable_cpu_fallback = false;
        } else if support_ratio > 0.5 {
            // Medium NNAPI support - balanced approach
            self.config.nnapi_config.execution_preference = NNAPIExecutionPreference::FastSingleAnswer;
            self.config.max_partitions = 4;
            self.config.enable_cpu_fallback = true;
        } else {
            // Low NNAPI support - minimize delegate overhead
            self.config.nnapi_config.execution_preference = NNAPIExecutionPreference::LowPower;
            self.config.max_partitions = 2;
            self.config.enable_cpu_fallback = true;
        }

        // Update efficiency metric
        self.stats.nnapi_efficiency_percent = support_ratio * 100.0;

        tracing::info!(
            "Optimized delegate for model: {:.1}% NNAPI support ({}/{} ops), efficiency: {:.1}%",
            support_ratio * 100.0,
            nnapi_supported_count,
            total_ops,
            self.stats.nnapi_efficiency_percent
        );

        Ok(())
    }

    /// Reset delegate statistics
    pub fn reset_stats(&mut self) {
        self.stats = TfLiteNNAPIStats::new();
    }

    // Private implementation methods

    fn create_nnapi_delegate(&self) -> Result<*mut c_void> {
        // Create TensorFlow Lite NNAPI delegate options
        let options = self.create_delegate_options()?;

        // Call TensorFlow Lite C API to create delegate
        let delegate = unsafe { tflite_nnapi_delegate_create(options) };

        if delegate.is_null() {
            return Err(TrustformersError::runtime_error(
                "Failed to create TensorFlow Lite NNAPI delegate".into(),
            ).into());
        }

        // Clean up options
        unsafe { tflite_nnapi_delegate_options_delete(options) };

        Ok(delegate)
    }

    fn create_delegate_options(&self) -> Result<*mut c_void> {
        // Create TensorFlow Lite NNAPI delegate options
        let options = unsafe { tflite_nnapi_delegate_options_create() };

        if options.is_null() {
            return Err(TrustformersError::runtime_error(
                "Failed to create delegate options".into(),
            ).into());
        }

        // Configure options based on our config
        unsafe {
            // Set execution preference
            let execution_preference = match self.config.nnapi_config.execution_preference {
                NNAPIExecutionPreference::FastSingleAnswer => 0,
                NNAPIExecutionPreference::SustainedSpeed => 1,
                NNAPIExecutionPreference::LowPower => 2,
            };
            tflite_nnapi_delegate_options_set_execution_preference(options, execution_preference);

            // Set accelerator name if specified
            if let Some(ref accelerator_name) = self.config.accelerator_name {
                let name_cstring = CString::new(accelerator_name.as_str()).expect("Operation failed");
                tflite_nnapi_delegate_options_set_accelerator_name(options, name_cstring.as_ptr());
            }

            // Set caching if enabled
            if self.config.enable_delegate_caching {
                let cache_dir = CString::new("/data/data/com.trustformers/cache").expect("Operation failed");
                let model_token = CString::new("trustformers_model").expect("Operation failed");
                tflite_nnapi_delegate_options_set_cache_dir(options, cache_dir.as_ptr());
                tflite_nnapi_delegate_options_set_model_token(options, model_token.as_ptr());
            }

            // Set CPU fallback
            tflite_nnapi_delegate_options_set_allow_fp16(options, self.config.nnapi_config.allow_relaxed_computation);

            // Set max partitions
            tflite_nnapi_delegate_options_set_max_partitions(options, self.config.max_partitions as i32);
        }

        Ok(options)
    }

    fn apply_delegate_to_interpreter(&self, interpreter: *mut c_void, delegate: *mut c_void) -> Result<()> {
        let result = unsafe { tflite_interpreter_modify_graph_with_delegate(interpreter, delegate) };

        if result != 0 {
            return Err(TrustformersError::runtime_error(
                "Failed to apply NNAPI delegate to interpreter".into(),
            ).into());
        }

        Ok(())
    }

    fn analyze_operation_support(&mut self, interpreter: *mut c_void) -> Result<()> {
        let start_time = Instant::now();

        // Get subgraph information
        let subgraph_count = unsafe { tflite_interpreter_get_subgraph_count(interpreter) };

        for subgraph_index in 0..subgraph_count {
            let op_count = unsafe { tflite_subgraph_get_node_count(interpreter, subgraph_index) };

            for op_index in 0..op_count {
                let op_code = unsafe { tflite_subgraph_get_node_opcode(interpreter, subgraph_index, op_index) };

                // Check if this operation is supported by NNAPI
                if self.is_operation_supported_by_nnapi(op_code) {
                    self.supported_ops.push(op_code);
                    self.stats.delegated_operations += 1;
                } else {
                    self.stats.cpu_fallback_operations += 1;
                }
            }
        }

        let analysis_time = start_time.elapsed().as_millis() as f32;
        self.stats.partition_creation_time_ms = analysis_time;

        Ok(())
    }

    fn analyze_all_operations(&self, interpreter: *mut c_void) -> Result<Vec<TfLiteOpSupport>> {
        let mut operations = Vec::new();
        let subgraph_count = unsafe { tflite_interpreter_get_subgraph_count(interpreter) };

        for subgraph_index in 0..subgraph_count {
            let op_count = unsafe { tflite_subgraph_get_node_count(interpreter, subgraph_index) };

            for op_index in 0..op_count {
                let op_code = unsafe { tflite_subgraph_get_node_opcode(interpreter, subgraph_index, op_index) };
                let op_name = self.get_operation_name(op_code);
                let nnapi_supported = self.is_operation_supported_by_nnapi(op_code);
                let performance_gain = self.estimate_performance_gain(op_code);

                let unsupported_reason = if !nnapi_supported {
                    Some(self.get_unsupported_reason(op_code))
                } else {
                    None
                };

                operations.push(TfLiteOpSupport {
                    op_code,
                    op_name,
                    nnapi_supported,
                    unsupported_reason,
                    performance_gain_factor: performance_gain,
                });
            }
        }

        Ok(operations)
    }

    fn invoke_interpreter(&self, interpreter: *mut c_void) -> Result<()> {
        let result = unsafe { tflite_interpreter_invoke(interpreter) };

        if result != 0 {
            return Err(TrustformersError::runtime_error("TensorFlow Lite inference failed".into()).into());
        }

        Ok(())
    }

    fn get_tensor_info(&self, interpreter: *mut c_void, index: i32, is_input: bool) -> Result<TensorInfo> {
        let tensor = if is_input {
            unsafe { tflite_interpreter_get_input_tensor(interpreter, index) }
        } else {
            unsafe { tflite_interpreter_get_output_tensor(interpreter, index) }
        };

        if tensor.is_null() {
            return Err(TrustformersError::runtime_error("Invalid tensor index".into()).into());
        }

        let dims_count = unsafe { tflite_tensor_get_num_dims(tensor) };
        let mut shape = Vec::with_capacity(dims_count as usize);

        for i in 0..dims_count {
            let dim = unsafe { tflite_tensor_get_dim(tensor, i) };
            shape.push(dim as usize);
        }

        let data_type = unsafe { tflite_tensor_get_type(tensor) };
        let byte_size = unsafe { tflite_tensor_get_byte_size(tensor) };

        Ok(TensorInfo {
            index,
            shape,
            data_type,
            byte_size: byte_size as usize,
        })
    }

    fn set_tensor_data(&self, interpreter: *mut c_void, index: i32, data: &[f32]) -> Result<()> {
        let tensor = unsafe { tflite_interpreter_get_input_tensor(interpreter, index) };

        if tensor.is_null() {
            return Err(TrustformersError::runtime_error("Invalid input tensor index".into()).into());
        }

        let tensor_data = unsafe { tflite_tensor_get_data(tensor) as *mut f32 };
        let tensor_size = unsafe { tflite_tensor_get_byte_size(tensor) } / std::mem::size_of::<f32>() as i32;

        if data.len() != tensor_size as usize {
            return Err(TrustformersError::runtime_error("Tensor data size mismatch".into()).into());
        }

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), tensor_data, data.len());
        }

        Ok(())
    }

    fn get_tensor_data(&self, interpreter: *mut c_void, index: i32) -> Result<Vec<f32>> {
        let tensor = unsafe { tflite_interpreter_get_output_tensor(interpreter, index) };

        if tensor.is_null() {
            return Err(TrustformersError::runtime_error("Invalid output tensor index".into()).into());
        }

        let tensor_data = unsafe { tflite_tensor_get_data(tensor) as *const f32 };
        let tensor_size = unsafe { tflite_tensor_get_byte_size(tensor) } / std::mem::size_of::<f32>() as i32;

        let data = unsafe { std::slice::from_raw_parts(tensor_data, tensor_size as usize) };

        Ok(data.to_vec())
    }

    fn is_operation_supported_by_nnapi(&self, op_code: i32) -> bool {
        // Check if operation is supported by NNAPI
        // This would map TensorFlow Lite operation codes to NNAPI support
        match op_code {
            0 => true,  // ADD
            1 => true,  // AVERAGE_POOL_2D
            2 => true,  // CONV_2D
            3 => true,  // DEPTHWISE_CONV_2D
            4 => true,  // FULLY_CONNECTED
            5 => true,  // LOGISTIC
            6 => true,  // MAX_POOL_2D
            7 => true,  // MUL
            8 => true,  // RELU
            9 => true,  // RESHAPE
            10 => true, // SOFTMAX
            _ => false, // Unsupported operations
        }
    }

    fn get_operation_name(&self, op_code: i32) -> String {
        // Map operation codes to names
        match op_code {
            0 => "ADD".to_string(),
            1 => "AVERAGE_POOL_2D".to_string(),
            2 => "CONV_2D".to_string(),
            3 => "DEPTHWISE_CONV_2D".to_string(),
            4 => "FULLY_CONNECTED".to_string(),
            5 => "LOGISTIC".to_string(),
            6 => "MAX_POOL_2D".to_string(),
            7 => "MUL".to_string(),
            8 => "RELU".to_string(),
            9 => "RESHAPE".to_string(),
            10 => "SOFTMAX".to_string(),
            _ => format!("UNKNOWN_{}", op_code),
        }
    }

    fn estimate_performance_gain(&self, op_code: i32) -> f32 {
        // Estimate performance gain from using NNAPI for this operation
        match op_code {
            2 | 3 => 3.0,  // Convolution operations - high gain
            4 => 2.5,      // Fully connected - medium-high gain
            1 | 6 => 2.0,  // Pooling operations - medium gain
            0 | 7 => 1.5,  // Element-wise operations - low-medium gain
            5 | 8 | 10 => 1.8, // Activation functions - medium gain
            9 => 1.0,      // Reshape - minimal gain
            _ => 1.0,      // Unknown operations
        }
    }

    fn get_unsupported_reason(&self, op_code: i32) -> String {
        // Provide reasons why operations are not supported
        if self.is_operation_supported_by_nnapi(op_code) {
            return "Supported".to_string();
        }

        match op_code {
            11..=20 => "Not implemented in NNAPI".to_string(),
            21..=30 => "Requires newer Android version".to_string(),
            31..=40 => "Custom operation not supported".to_string(),
            _ => "Unknown operation".to_string(),
        }
    }
}

#[cfg(all(target_os = "android", feature = "tflite-nnapi"))]
impl Default for TfLiteNNAPIConfig {
    fn default() -> Self {
        Self {
            nnapi_config: NNAPIConfig::default(),
            enable_delegate_optimization: true,
            nnapi_fallback_enabled: true,
            max_partitions: 4,
            enable_delegate_caching: true,
            compilation_timeout_ms: 5000,
            enable_verbose_logging: false,
            accelerator_name: None,
            enable_cpu_fallback: true,
        }
    }
}

#[cfg(all(target_os = "android", feature = "tflite-nnapi"))]
impl TfLiteNNAPIConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        self.nnapi_config.validate()?;

        if self.max_partitions == 0 {
            return Err(TrustformersError::config_error {
                message: "Max partitions must be > 0".to_string(),
                context: trustformers_core::error::ErrorContext::new(
                    trustformers_core::error::ErrorCode::E4001,
                    "validate".to_string(),
                ),
            });
        }

        if self.max_partitions > 16 {
            return Err(TrustformersError::config_error {
                message: "Too many partitions for delegate".to_string(),
                context: trustformers_core::error::ErrorContext::new(
                    trustformers_core::error::ErrorCode::E4001,
                    "validate".to_string(),
                ),
            });
        }

        if self.compilation_timeout_ms < 1000 {
            return Err(TrustformersError::config_error {
                message: "Compilation timeout too short".to_string(),
                context: trustformers_core::error::ErrorContext::new(
                    trustformers_core::error::ErrorCode::E4001,
                    "validate".to_string(),
                ),
            });
        }

        Ok(())
    }

    /// Create configuration optimized for performance
    pub fn performance_optimized() -> Self {
        Self {
            nnapi_config: NNAPIConfig::performance_optimized(),
            enable_delegate_optimization: true,
            nnapi_fallback_enabled: false,
            max_partitions: 8,
            enable_delegate_caching: true,
            compilation_timeout_ms: 10000,
            enable_verbose_logging: false,
            accelerator_name: None,
            enable_cpu_fallback: false,
        }
    }

    /// Create configuration optimized for power efficiency
    pub fn power_optimized() -> Self {
        Self {
            nnapi_config: NNAPIConfig::power_optimized(),
            enable_delegate_optimization: false,
            nnapi_fallback_enabled: true,
            max_partitions: 2,
            enable_delegate_caching: false,
            compilation_timeout_ms: 3000,
            enable_verbose_logging: false,
            accelerator_name: None,
            enable_cpu_fallback: true,
        }
    }
}

#[cfg(all(target_os = "android", feature = "tflite-nnapi"))]
impl TfLiteNNAPIStats {
    fn new() -> Self {
        Self {
            total_invocations: 0,
            delegated_operations: 0,
            cpu_fallback_operations: 0,
            avg_delegate_overhead_ms: 0.0,
            delegate_init_time_ms: 0.0,
            partition_creation_time_ms: 0.0,
            nnapi_efficiency_percent: 0.0,
            delegation_memory_mb: 0,
        }
    }

    fn update_invocation(&mut self, invocation_time_ms: f32) {
        self.total_invocations += 1;

        // Update running average
        let alpha = 0.1;
        if self.total_invocations == 1 {
            self.avg_delegate_overhead_ms = invocation_time_ms;
        } else {
            self.avg_delegate_overhead_ms =
                alpha * invocation_time_ms + (1.0 - alpha) * self.avg_delegate_overhead_ms;
        }
    }
}

/// Tensor information structure
#[cfg(all(target_os = "android", feature = "tflite-nnapi"))]
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub index: i32,
    pub shape: Vec<usize>,
    pub data_type: i32,
    pub byte_size: usize,
}

/// Convert mobile config to TensorFlow Lite NNAPI delegate config
#[cfg(all(target_os = "android", feature = "tflite-nnapi"))]
pub fn mobile_config_to_tflite_nnapi(mobile_config: &MobileConfig) -> TfLiteNNAPIConfig {
    let nnapi_config = crate::nnapi::mobile_config_to_nnapi(mobile_config);

    let mut tflite_config = TfLiteNNAPIConfig::default();
    tflite_config.nnapi_config = nnapi_config;

    // Map memory optimization to delegate settings
    match mobile_config.memory_optimization {
        MemoryOptimization::Maximum => {
            tflite_config = TfLiteNNAPIConfig::power_optimized();
            tflite_config.max_partitions = 1;
            tflite_config.enable_delegate_caching = false;
        },
        MemoryOptimization::Balanced => {
            tflite_config.max_partitions = 4;
            tflite_config.enable_delegate_optimization = true;
            tflite_config.enable_cpu_fallback = true;
        },
        MemoryOptimization::Minimal => {
            tflite_config = TfLiteNNAPIConfig::performance_optimized();
            tflite_config.max_partitions = 8;
        },
    }

    tflite_config
}

// External C API function declarations for TensorFlow Lite NNAPI delegate
#[cfg(all(target_os = "android", feature = "tflite-nnapi"))]
extern "C" {
    fn tflite_nnapi_delegate_create(options: *mut c_void) -> *mut c_void;
    fn tflite_nnapi_delegate_delete(delegate: *mut c_void);
    fn tflite_nnapi_delegate_options_create() -> *mut c_void;
    fn tflite_nnapi_delegate_options_delete(options: *mut c_void);
    fn tflite_nnapi_delegate_options_set_execution_preference(options: *mut c_void, preference: i32);
    fn tflite_nnapi_delegate_options_set_accelerator_name(options: *mut c_void, name: *const i8);
    fn tflite_nnapi_delegate_options_set_cache_dir(options: *mut c_void, cache_dir: *const i8);
    fn tflite_nnapi_delegate_options_set_model_token(options: *mut c_void, model_token: *const i8);
    fn tflite_nnapi_delegate_options_set_allow_fp16(options: *mut c_void, allow: bool);
    fn tflite_nnapi_delegate_options_set_max_partitions(options: *mut c_void, max_partitions: i32);

    fn tflite_interpreter_modify_graph_with_delegate(interpreter: *mut c_void, delegate: *mut c_void) -> i32;
    fn tflite_interpreter_invoke(interpreter: *mut c_void) -> i32;
    fn tflite_interpreter_get_input_tensor(interpreter: *mut c_void, index: i32) -> *mut c_void;
    fn tflite_interpreter_get_output_tensor(interpreter: *mut c_void, index: i32) -> *mut c_void;
    fn tflite_interpreter_get_subgraph_count(interpreter: *mut c_void) -> i32;

    fn tflite_subgraph_get_node_count(interpreter: *mut c_void, subgraph_index: i32) -> i32;
    fn tflite_subgraph_get_node_opcode(interpreter: *mut c_void, subgraph_index: i32, node_index: i32) -> i32;

    fn tflite_tensor_get_data(tensor: *mut c_void) -> *mut c_void;
    fn tflite_tensor_get_byte_size(tensor: *mut c_void) -> i32;
    fn tflite_tensor_get_num_dims(tensor: *mut c_void) -> i32;
    fn tflite_tensor_get_dim(tensor: *mut c_void, dim_index: i32) -> i32;
    fn tflite_tensor_get_type(tensor: *mut c_void) -> i32;
}

// Stub implementations for non-Android platforms
#[cfg(not(all(target_os = "android", feature = "tflite-nnapi")))]
pub struct TfLiteNNAPIDelegate;

#[cfg(not(all(target_os = "android", feature = "tflite-nnapi")))]
impl TfLiteNNAPIDelegate {
    pub fn new(_config: ()) -> Result<Self, CoreError> {
        Err(TrustformersError::runtime_error(
            "TensorFlow Lite NNAPI delegate only available on Android".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(target_os = "android", feature = "tflite-nnapi"))]
    #[test]
    fn test_tflite_nnapi_config_validation() {
        let config = TfLiteNNAPIConfig::default();
        assert!(config.validate().is_ok());

        let mut invalid_config = config.clone();
        invalid_config.max_partitions = 0;
        assert!(invalid_config.validate().is_err());

        invalid_config.max_partitions = 20;
        assert!(invalid_config.validate().is_err());
    }

    #[cfg(all(target_os = "android", feature = "tflite-nnapi"))]
    #[test]
    fn test_optimized_configs() {
        let perf_config = TfLiteNNAPIConfig::performance_optimized();
        assert_eq!(perf_config.max_partitions, 8);
        assert!(!perf_config.enable_cpu_fallback);
        assert!(perf_config.enable_delegate_optimization);

        let power_config = TfLiteNNAPIConfig::power_optimized();
        assert_eq!(power_config.max_partitions, 2);
        assert!(power_config.enable_cpu_fallback);
        assert!(!power_config.enable_delegate_optimization);
    }

    #[cfg(all(target_os = "android", feature = "tflite-nnapi"))]
    #[test]
    fn test_mobile_to_tflite_nnapi_config_conversion() {
        let mobile_config = crate::MobileConfig {
            memory_optimization: MemoryOptimization::Maximum,
            num_threads: 1,
            use_fp16: true,
            ..Default::default()
        };

        let tflite_config = mobile_config_to_tflite_nnapi(&mobile_config);
        assert_eq!(tflite_config.max_partitions, 1);
        assert!(!tflite_config.enable_delegate_caching);
        assert!(tflite_config.nnapi_config.allow_relaxed_computation);
    }
}
