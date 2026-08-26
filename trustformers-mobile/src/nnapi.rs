//! Neural Networks API (NNAPI) Backend for Android Integration
//!
//! This module provides NNAPI integration for optimized inference on Android devices,
//! leveraging Google's Neural Networks API for hardware-accelerated inference.

#[cfg(all(target_os = "android", feature = "nnapi"))]
use crate::{MemoryOptimization, MobileConfig, MobileStats};
#[cfg(all(target_os = "android", feature = "nnapi"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_os = "android", feature = "nnapi"))]
use std::collections::HashMap;
#[cfg(all(target_os = "android", feature = "nnapi"))]
use std::time::Instant;
use trustformers_core::error::{CoreError, Result};
#[cfg(all(target_os = "android", feature = "nnapi"))]
use trustformers_core::Tensor;
use trustformers_core::TrustformersError;

#[cfg(all(target_os = "android", feature = "nnapi"))]
use jni::{
    objects::{JByteArray, JClass, JObject, JString},
    sys::{jbyteArray, jlong, jobject},
    JNIEnv, JavaVM,
};

/// NNAPI inference engine for Android
#[cfg(all(target_os = "android", feature = "nnapi"))]
pub struct NNAPIEngine {
    config: NNAPIConfig,
    model_handle: Option<usize>,
    stats: NNAPIStats,
    device_info: AndroidDeviceInfo,
    jvm: Option<JavaVM>,
    compilation_handle: Option<usize>,
}

/// NNAPI configuration
#[cfg(all(target_os = "android", feature = "nnapi"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NNAPIConfig {
    /// Preferred device types for execution
    pub preferred_devices: Vec<NNAPIDeviceType>,
    /// Enable relaxed computation for better performance
    pub allow_relaxed_computation: bool,
    /// Cache compilation results
    pub enable_compilation_caching: bool,
    /// Execution preference
    pub execution_preference: NNAPIExecutionPreference,
    /// Maximum number of concurrent executions
    pub max_concurrent_executions: usize,
    /// Memory mapping for large models
    pub use_memory_mapping: bool,
    /// Timeout for operations (milliseconds)
    pub operation_timeout_ms: u32,
}

/// NNAPI device types
#[cfg(all(target_os = "android", feature = "nnapi"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NNAPIDeviceType {
    /// CPU implementation
    CPU,
    /// GPU implementation (Vulkan, OpenGL ES)
    GPU,
    /// Dedicated neural processing unit
    NPU,
    /// Digital Signal Processor
    DSP,
    /// Accelerator (vendor-specific)
    Accelerator,
    /// Any available device
    Any,
}

/// NNAPI execution preferences
#[cfg(all(target_os = "android", feature = "nnapi"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NNAPIExecutionPreference {
    /// Prefer fast single-threaded inference
    FastSingleAnswer,
    /// Prefer sustained throughput
    SustainedSpeed,
    /// Prefer power efficiency
    LowPower,
}

/// Android device information
#[cfg(all(target_os = "android", feature = "nnapi"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AndroidDeviceInfo {
    /// Android version (API level)
    pub android_api_level: u32,
    /// Device manufacturer
    pub manufacturer: String,
    /// Device model
    pub device_model: String,
    /// Available NNAPI devices
    pub available_devices: Vec<NNAPIDeviceInfo>,
    /// Total system memory in MB
    pub total_memory_mb: usize,
    /// Available memory in MB
    pub available_memory_mb: usize,
    /// Vulkan support
    pub has_vulkan: bool,
    /// OpenGL ES version
    pub opengl_es_version: String,
}

/// NNAPI device information
#[cfg(all(target_os = "android", feature = "nnapi"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NNAPIDeviceInfo {
    /// Device name
    pub name: String,
    /// Device type
    pub device_type: NNAPIDeviceType,
    /// Device version
    pub version: String,
    /// Supported operations
    pub supported_operations: Vec<String>,
    /// Performance characteristics
    pub performance_info: NNAPIPerformanceInfo,
}

/// NNAPI performance information
#[cfg(all(target_os = "android", feature = "nnapi"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NNAPIPerformanceInfo {
    /// Execution time scaling factor
    pub exec_time: f32,
    /// Power usage scaling factor
    pub power_usage: f32,
    /// Memory bandwidth
    pub memory_bandwidth_mbps: usize,
    /// Compute throughput
    pub compute_throughput_ops: usize,
}

/// NNAPI statistics
#[cfg(all(target_os = "android", feature = "nnapi"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NNAPIStats {
    /// Total executions performed
    pub total_executions: usize,
    /// Average execution time (ms)
    pub avg_execution_time_ms: f32,
    /// Compilation time (ms)
    pub compilation_time_ms: f32,
    /// Memory usage in MB
    pub memory_usage_mb: usize,
    /// Device utilization percentages
    pub device_utilization: HashMap<String, f32>,
    /// Power consumption estimate
    pub estimated_power_mw: f32,
    /// Cache hit rate for compilations
    pub compilation_cache_hit_rate: f32,
}

#[cfg(all(target_os = "android", feature = "nnapi"))]
impl NNAPIEngine {
    /// Create new NNAPI engine
    pub fn new(config: NNAPIConfig) -> Result<Self> {
        config.validate()?;

        let device_info = Self::detect_device_info()?;
        let stats = NNAPIStats::new();

        Ok(Self {
            config,
            model_handle: None,
            stats,
            device_info,
            jvm: None,
            compilation_handle: None,
        })
    }

    /// Initialize with Android JVM context
    pub fn init_with_jvm(&mut self, jvm: JavaVM) -> Result<()> {
        self.jvm = Some(jvm);

        // Initialize NNAPI through JNI
        self.init_nnapi_context()?;

        Ok(())
    }

    /// Load NNAPI model from data
    pub fn load_model(&mut self, model_data: &[u8]) -> Result<()> {
        let start_time = Instant::now();

        tracing::info!("Loading NNAPI model ({} bytes)", model_data.len());

        // Create NNAPI model
        let model_handle = self.create_nnapi_model(model_data)?;

        // Compile model for target devices
        let compilation_handle = self.compile_model(model_handle)?;

        self.model_handle = Some(model_handle);
        self.compilation_handle = Some(compilation_handle);

        let compilation_time = start_time.elapsed().as_millis() as f32;
        self.stats.compilation_time_ms = compilation_time;

        tracing::info!(
            "NNAPI model compiled successfully in {:.2}ms on {} devices",
            compilation_time,
            self.device_info.available_devices.len()
        );

        Ok(())
    }

    /// Execute NNAPI inference
    pub fn execute(&mut self, input: &HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        if self.compilation_handle.is_none() {
            return Err(TrustformersError::runtime_error("NNAPI model not compiled".into()).into());
        }

        let start_time = Instant::now();

        // Create execution instance
        let execution_handle = self.create_execution()?;

        // Set input tensors
        self.set_input_tensors(execution_handle, input)?;

        // Prepare output tensors
        let output_tensors = self.prepare_output_tensors()?;

        // Execute inference
        self.execute_inference(execution_handle)?;

        // Retrieve results
        let results = self.get_output_tensors(execution_handle, output_tensors)?;

        // Cleanup execution
        self.cleanup_execution(execution_handle)?;

        let execution_time = start_time.elapsed().as_millis() as f32;
        self.stats.update_execution(execution_time);

        Ok(results)
    }

    /// Execute batch inference
    pub fn batch_execute(
        &mut self,
        inputs: &[HashMap<String, Tensor>],
    ) -> Result<Vec<HashMap<String, Tensor>>> {
        let mut results = Vec::with_capacity(inputs.len());

        // NNAPI doesn't have native batch support, so we execute sequentially
        // but can optimize by reusing execution instances
        for input in inputs {
            let result = self.execute(input)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Get NNAPI statistics
    pub fn get_stats(&self) -> &NNAPIStats {
        &self.stats
    }

    /// Get device information
    pub fn get_device_info(&self) -> &AndroidDeviceInfo {
        &self.device_info
    }

    /// Optimize configuration for current device
    pub fn optimize_for_device(&mut self) -> Result<()> {
        // Select best devices based on available hardware
        self.config.preferred_devices = self.select_optimal_devices();

        // Adjust execution preference based on device capabilities
        self.config.execution_preference = self.select_execution_preference();

        // Enable features based on Android API level
        if self.device_info.android_api_level >= 30 {
            // Android 11+ features
            self.config.enable_compilation_caching = true;
            self.config.max_concurrent_executions = 4;
        } else if self.device_info.android_api_level >= 29 {
            // Android 10 features
            self.config.enable_compilation_caching = true;
            self.config.max_concurrent_executions = 2;
        } else {
            // Older Android versions
            self.config.enable_compilation_caching = false;
            self.config.max_concurrent_executions = 1;
        }

        tracing::info!(
            "Optimized NNAPI configuration for {} (API {}) with {} devices",
            self.device_info.device_model,
            self.device_info.android_api_level,
            self.device_info.available_devices.len()
        );

        Ok(())
    }

    // Private implementation methods

    /// Real Android device/NNAPI enumeration requires either JNI calls into
    /// `android.os.Build`/`ConnectivityManager` (needing a live `JavaVM`,
    /// which is not available yet -- `NNAPIEngine::new` calls this before
    /// `init_with_jvm` ever runs) or the NDK C API's
    /// `ANeuralNetworks_getDeviceCount`/`ANeuralNetworks_getDevice` family
    /// (see `crate::android::nnapi`, which binds those for real, and
    /// `crate::android::engine::AndroidInferenceEngine::detect_nnapi_devices`,
    /// which calls them). Neither is reachable from a bare constructor with
    /// no JVM/context, so this reports an honestly-empty/unknown profile
    /// rather than the fabricated "Google Pixel, API 30, Vulkan, OpenGL ES
    /// 3.2, one CPU device" reading a previous revision returned
    /// unconditionally, for every device, real or not.
    fn detect_device_info() -> Result<AndroidDeviceInfo> {
        Ok(AndroidDeviceInfo {
            android_api_level: 0,
            manufacturer: "Unknown".to_string(),
            device_model: "Unknown".to_string(),
            available_devices: Vec::new(),
            total_memory_mb: 0,
            available_memory_mb: 0,
            has_vulkan: false,
            opengl_es_version: "Unknown".to_string(),
        })
    }

    fn init_nnapi_context(&self) -> Result<()> {
        // Nothing to honestly fail on yet: no NNAPI object has been created
        // through this engine at this point (that happens in
        // `create_nnapi_model`, called from `load_model`, which is the
        // method that now reports the real gap -- see its doc comment).
        Ok(())
    }

    /// Building a real `ANeuralNetworksModel` operand graph from
    /// `_model_data` (parsing the checkpoint and emitting NNAPI operands
    /// and operations for each layer) is not implemented. This is the same
    /// gap documented on
    /// `crate::android::engine::AndroidInferenceEngine::nnapi_inference`,
    /// this crate's other NNAPI entry point: real execution needs a
    /// tensor-op-to-NNAPI-operand compiler that does not exist yet, and a
    /// per-element placeholder formula standing in for it would be
    /// indistinguishable from real accelerator output while computing
    /// nothing accelerator-specific at all. Reporting the gap honestly here
    /// means `load_model` fails at the point it actually cannot proceed,
    /// instead of returning a fabricated handle that lets `execute` later
    /// "succeed" with an empty result map.
    fn create_nnapi_model(&self, _model_data: &[u8]) -> Result<usize> {
        Err(TrustformersError::not_implemented(
            "NNAPI operand-graph model construction from checkpoint data".to_string(),
        )
        .into())
    }

    fn compile_model(&self, _model_handle: usize) -> Result<usize> {
        Err(TrustformersError::not_implemented("NNAPI model compilation".to_string()).into())
    }

    fn create_execution(&self) -> Result<usize> {
        Err(
            TrustformersError::not_implemented("NNAPI execution instance creation".to_string())
                .into(),
        )
    }

    fn set_input_tensors(
        &self,
        _execution_handle: usize,
        _input: &HashMap<String, Tensor>,
    ) -> Result<()> {
        Err(TrustformersError::not_implemented("NNAPI input tensor binding".to_string()).into())
    }

    fn prepare_output_tensors(&self) -> Result<Vec<String>> {
        Err(
            TrustformersError::not_implemented("NNAPI output tensor preparation".to_string())
                .into(),
        )
    }

    fn execute_inference(&self, _execution_handle: usize) -> Result<()> {
        Err(TrustformersError::not_implemented("NNAPI inference execution".to_string()).into())
    }

    fn get_output_tensors(
        &self,
        _execution_handle: usize,
        _output_names: Vec<String>,
    ) -> Result<HashMap<String, Tensor>> {
        Err(TrustformersError::not_implemented("NNAPI output tensor retrieval".to_string()).into())
    }

    fn cleanup_execution(&self, _execution_handle: usize) -> Result<()> {
        // `create_execution` always fails first (see above), so no
        // execution instance is ever actually created through this engine
        // today; nothing to honestly fail on cleaning up.
        Ok(())
    }

    fn select_optimal_devices(&self) -> Vec<NNAPIDeviceType> {
        let mut devices = Vec::new();

        // Prefer specialized hardware first
        for device in &self.device_info.available_devices {
            match device.device_type {
                NNAPIDeviceType::NPU => devices.push(NNAPIDeviceType::NPU),
                NNAPIDeviceType::DSP => devices.push(NNAPIDeviceType::DSP),
                NNAPIDeviceType::Accelerator => devices.push(NNAPIDeviceType::Accelerator),
                _ => {},
            }
        }

        // Add GPU if available and Vulkan is supported
        if self.device_info.has_vulkan {
            devices.push(NNAPIDeviceType::GPU);
        }

        // Always include CPU as fallback
        devices.push(NNAPIDeviceType::CPU);

        if devices.is_empty() {
            devices.push(NNAPIDeviceType::Any);
        }

        devices
    }

    fn select_execution_preference(&self) -> NNAPIExecutionPreference {
        // Select execution preference based on device characteristics
        if self.device_info.available_memory_mb < 1024 {
            NNAPIExecutionPreference::LowPower
        } else if self.device_info.available_devices.len() > 2 {
            NNAPIExecutionPreference::SustainedSpeed
        } else {
            NNAPIExecutionPreference::FastSingleAnswer
        }
    }
}

#[cfg(all(target_os = "android", feature = "nnapi"))]
impl Default for NNAPIConfig {
    fn default() -> Self {
        Self {
            preferred_devices: vec![NNAPIDeviceType::Any],
            allow_relaxed_computation: true,
            enable_compilation_caching: true,
            execution_preference: NNAPIExecutionPreference::FastSingleAnswer,
            max_concurrent_executions: 1,
            use_memory_mapping: true,
            operation_timeout_ms: 5000,
        }
    }
}

#[cfg(all(target_os = "android", feature = "nnapi"))]
impl NNAPIConfig {
    /// Validate NNAPI configuration
    pub fn validate(&self) -> Result<()> {
        if self.preferred_devices.is_empty() {
            return Err(TrustformersError::config_error {
                message: "Must specify at least one preferred device".to_string(),
                context: trustformers_core::error::ErrorContext::new(
                    trustformers_core::error::ErrorCode::E4001,
                    "validate".to_string(),
                ),
            });
        }

        if self.max_concurrent_executions == 0 {
            return Err(TrustformersError::config_error {
                message: "Concurrent executions must be > 0".to_string(),
                context: trustformers_core::error::ErrorContext::new(
                    trustformers_core::error::ErrorCode::E4001,
                    "validate".to_string(),
                ),
            });
        }

        if self.max_concurrent_executions > 8 {
            return Err(TrustformersError::config_error {
                message: "Too many concurrent executions for Android".to_string(),
                context: trustformers_core::error::ErrorContext::new(
                    trustformers_core::error::ErrorCode::E4001,
                    "validate".to_string(),
                ),
            });
        }

        if self.operation_timeout_ms < 100 {
            return Err(TrustformersError::config_error {
                message: "Operation timeout too short".to_string(),
                context: trustformers_core::error::ErrorContext::new(
                    trustformers_core::error::ErrorCode::E4001,
                    "validate".to_string(),
                ),
            });
        }

        Ok(())
    }

    /// Create power-optimized configuration
    pub fn power_optimized() -> Self {
        Self {
            preferred_devices: vec![
                NNAPIDeviceType::NPU,
                NNAPIDeviceType::DSP,
                NNAPIDeviceType::CPU,
            ],
            allow_relaxed_computation: true,
            enable_compilation_caching: true,
            execution_preference: NNAPIExecutionPreference::LowPower,
            max_concurrent_executions: 1,
            use_memory_mapping: false,
            operation_timeout_ms: 10000,
        }
    }

    /// Create performance-optimized configuration
    pub fn performance_optimized() -> Self {
        Self {
            preferred_devices: vec![
                NNAPIDeviceType::GPU,
                NNAPIDeviceType::NPU,
                NNAPIDeviceType::CPU,
            ],
            allow_relaxed_computation: true,
            enable_compilation_caching: true,
            execution_preference: NNAPIExecutionPreference::SustainedSpeed,
            max_concurrent_executions: 4,
            use_memory_mapping: true,
            operation_timeout_ms: 2000,
        }
    }
}

#[cfg(all(target_os = "android", feature = "nnapi"))]
impl NNAPIStats {
    fn new() -> Self {
        Self {
            total_executions: 0,
            avg_execution_time_ms: 0.0,
            compilation_time_ms: 0.0,
            memory_usage_mb: 0,
            device_utilization: HashMap::new(),
            estimated_power_mw: 0.0,
            compilation_cache_hit_rate: 0.0,
        }
    }

    fn update_execution(&mut self, execution_time_ms: f32) {
        self.total_executions += 1;

        // Update running average
        let alpha = 0.1;
        if self.total_executions == 1 {
            self.avg_execution_time_ms = execution_time_ms;
        } else {
            self.avg_execution_time_ms =
                alpha * execution_time_ms + (1.0 - alpha) * self.avg_execution_time_ms;
        }
    }
}

/// Convert mobile config to NNAPI config
#[cfg(all(target_os = "android", feature = "nnapi"))]
pub fn mobile_config_to_nnapi(mobile_config: &MobileConfig) -> NNAPIConfig {
    let mut nnapi_config = NNAPIConfig::default();

    // Map memory optimization to NNAPI settings
    match mobile_config.memory_optimization {
        MemoryOptimization::Maximum => {
            nnapi_config = NNAPIConfig::power_optimized();
            nnapi_config.max_concurrent_executions = 1;
            nnapi_config.use_memory_mapping = false;
        },
        MemoryOptimization::Balanced => {
            nnapi_config.execution_preference = NNAPIExecutionPreference::FastSingleAnswer;
            nnapi_config.max_concurrent_executions = 2;
            nnapi_config.use_memory_mapping = true;
        },
        MemoryOptimization::Minimal => {
            nnapi_config = NNAPIConfig::performance_optimized();
            nnapi_config.max_concurrent_executions = mobile_config.num_threads.max(1);
        },
    }

    // Enable relaxed computation for FP16
    nnapi_config.allow_relaxed_computation = mobile_config.use_fp16;

    nnapi_config
}

/// JNI exports for Android integration
#[cfg(all(target_os = "android", feature = "nnapi"))]
#[no_mangle]
pub extern "system" fn Java_com_trustformers_NNAPIEngine_createEngine(
    env: JNIEnv,
    _class: JClass,
    config_json: JString,
) -> jlong {
    let config_str: String = match env.get_string(config_json) {
        Ok(s) => s.into(),
        Err(_) => return 0,
    };

    match serde_json::from_str::<NNAPIConfig>(&config_str) {
        Ok(config) => match NNAPIEngine::new(config) {
            Ok(engine) => Box::into_raw(Box::new(engine)) as jlong,
            Err(_) => 0,
        },
        Err(_) => 0,
    }
}

#[cfg(all(target_os = "android", feature = "nnapi"))]
#[no_mangle]
pub extern "system" fn Java_com_trustformers_NNAPIEngine_loadModel(
    env: JNIEnv,
    _class: JClass,
    engine_ptr: jlong,
    model_data: jbyteArray,
) -> jlong {
    if engine_ptr == 0 {
        return 0;
    }

    let engine = unsafe { &mut *(engine_ptr as *mut NNAPIEngine) };

    // Real conversion of the Java byte array to the actual model bytes --
    // matching `crate::android::jni`'s established pattern for this JNI
    // binding version. The previous implementation ignored `model_data`
    // entirely and passed `vec![0u8; 1024]` of zeros to `load_model`
    // regardless of what the Java caller actually sent.
    let model_bytes = match env.convert_byte_array(model_data) {
        Ok(bytes) => bytes,
        Err(_) => return 0,
    };

    match engine.load_model(&model_bytes) {
        Ok(_) => 1,
        Err(_) => 0,
    }
}

#[cfg(all(target_os = "android", feature = "nnapi"))]
#[no_mangle]
pub extern "system" fn Java_com_trustformers_NNAPIEngine_execute(
    _env: JNIEnv,
    _class: JClass,
    engine_ptr: jlong,
    input_data: jobject,
) -> jobject {
    if engine_ptr == 0 {
        return std::ptr::null_mut();
    }

    let engine = unsafe { &mut *(engine_ptr as *mut NNAPIEngine) };

    // Convert Java input to HashMap<String, Tensor>
    // This is a simplified implementation
    let input = HashMap::new();

    match engine.execute(&input) {
        Ok(_output) => {
            // Convert output to Java object
            std::ptr::null_mut() // Placeholder
        },
        Err(_) => std::ptr::null_mut(),
    }
}

// Stub implementations for non-Android platforms

#[cfg(not(all(target_os = "android", feature = "nnapi")))]
pub struct NNAPIEngine;

#[cfg(not(all(target_os = "android", feature = "nnapi")))]
impl NNAPIEngine {
    pub fn new(_config: ()) -> Result<Self> {
        Err(TrustformersError::runtime_error("NNAPI only available on Android".into()).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(target_os = "android", feature = "nnapi"))]
    #[test]
    fn test_nnapi_config_validation() {
        let config = NNAPIConfig::default();
        assert!(config.validate().is_ok());

        let mut invalid_config = config.clone();
        invalid_config.preferred_devices.clear();
        assert!(invalid_config.validate().is_err());

        invalid_config.preferred_devices.push(NNAPIDeviceType::CPU);
        invalid_config.max_concurrent_executions = 0;
        assert!(invalid_config.validate().is_err());
    }

    #[cfg(all(target_os = "android", feature = "nnapi"))]
    #[test]
    fn test_optimized_configs() {
        let power_config = NNAPIConfig::power_optimized();
        assert_eq!(
            power_config.execution_preference,
            NNAPIExecutionPreference::LowPower
        );
        assert_eq!(power_config.max_concurrent_executions, 1);
        assert!(!power_config.use_memory_mapping);

        let perf_config = NNAPIConfig::performance_optimized();
        assert_eq!(
            perf_config.execution_preference,
            NNAPIExecutionPreference::SustainedSpeed
        );
        assert_eq!(perf_config.max_concurrent_executions, 4);
        assert!(perf_config.use_memory_mapping);
    }

    #[cfg(all(target_os = "android", feature = "nnapi"))]
    #[test]
    fn test_mobile_to_nnapi_config_conversion() {
        let mobile_config = crate::MobileConfig {
            memory_optimization: MemoryOptimization::Maximum,
            num_threads: 1,
            use_fp16: true,
            ..Default::default()
        };

        let nnapi_config = mobile_config_to_nnapi(&mobile_config);
        assert_eq!(
            nnapi_config.execution_preference,
            NNAPIExecutionPreference::LowPower
        );
        assert_eq!(nnapi_config.max_concurrent_executions, 1);
        assert!(nnapi_config.allow_relaxed_computation);
        assert!(!nnapi_config.use_memory_mapping);
    }

    /// Regression test for the P0 finding: `NNAPIEngine::execute` used to
    /// chain `create_nnapi_model`/`compile_model`/.../`get_output_tensors`
    /// placeholders that all unconditionally returned `Ok`, so
    /// `load_model` (and, downstream, `execute`) "succeeded" on every call
    /// without ever building or running a real NNAPI graph. Real
    /// operand-graph compilation is not implemented, so this must now fail
    /// honestly instead.
    #[cfg(all(target_os = "android", feature = "nnapi"))]
    #[test]
    fn test_load_model_reports_honest_failure_rather_than_fake_success() {
        let config = NNAPIConfig::default();
        let mut engine = NNAPIEngine::new(config).expect("engine construction itself is real");

        let result = engine.load_model(b"irrelevant bytes -- no compiler exists to read them yet");
        assert!(
            result.is_err(),
            "load_model must not report success when no operand-graph compiler exists to have \
             actually compiled anything"
        );
    }

    /// Regression test for the P0 finding: `detect_device_info` fabricated
    /// a "Google Pixel, API 30" reading unconditionally, for any device.
    #[cfg(all(target_os = "android", feature = "nnapi"))]
    #[test]
    fn test_detect_device_info_does_not_fabricate_a_specific_device() {
        let info = NNAPIEngine::detect_device_info().expect("detect_device_info");
        assert_eq!(info.manufacturer, "Unknown");
        assert_eq!(info.device_model, "Unknown");
        assert!(
            info.available_devices.is_empty(),
            "must not fabricate a CPU/GPU/NPU device entry"
        );
    }
}
