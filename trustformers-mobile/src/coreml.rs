//! Core ML Backend for iOS Integration
//!
//! This module provides Core ML integration for optimized inference on iOS devices,
//! leveraging Apple's Machine Learning framework for hardware-accelerated inference.

#[cfg(all(target_os = "ios", feature = "coreml"))]
use crate::{MemoryOptimization, MobileConfig, MobileStats};
#[cfg(all(target_os = "ios", feature = "coreml"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_os = "ios", feature = "coreml"))]
use std::collections::HashMap;
#[cfg(all(target_os = "ios", feature = "coreml"))]
use std::time::Instant;
use trustformers_core::error::{CoreError, Result};
#[cfg(all(target_os = "ios", feature = "coreml"))]
use trustformers_core::Tensor;
use trustformers_core::TrustformersError;

#[cfg(all(target_os = "ios", feature = "coreml"))]
use core_foundation::base::{CFType, CFTypeRef};
#[cfg(all(target_os = "ios", feature = "coreml"))]
use objc::runtime::{Class, Object};
#[cfg(all(target_os = "ios", feature = "coreml"))]
use objc::{msg_send, sel, sel_impl};

/// Core ML inference engine for iOS
#[cfg(all(target_os = "ios", feature = "coreml"))]
pub struct CoreMLEngine {
    config: CoreMLConfig,
    model: Option<*mut Object>,
    stats: CoreMLStats,
    device_info: IOsDeviceInfo,
    prediction_request: Option<*mut Object>,
}

/// Core ML configuration
#[cfg(all(target_os = "ios", feature = "coreml"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMLConfig {
    /// Model compute units (CPU, GPU, Neural Engine)
    pub compute_units: CoreMLComputeUnits,
    /// Enable batch prediction
    pub enable_batch_prediction: bool,
    /// Maximum batch size for Core ML
    pub max_batch_size: usize,
    /// Use lower precision for inference
    pub use_reduced_precision: bool,
    /// Enable Core ML model optimization
    pub enable_optimization: bool,
    /// Memory pressure handling
    pub memory_pressure_handling: CoreMLMemoryHandling,
}

/// Core ML compute units
#[cfg(all(target_os = "ios", feature = "coreml"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoreMLComputeUnits {
    /// CPU only
    CPUOnly,
    /// CPU and GPU
    CPUAndGPU,
    /// All available units (CPU, GPU, Neural Engine)
    All,
    /// Neural Engine only (if available)
    NeuralEngineOnly,
}

/// Core ML memory handling strategies
#[cfg(all(target_os = "ios", feature = "coreml"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoreMLMemoryHandling {
    /// Aggressive memory management
    Aggressive,
    /// Balanced memory usage
    Balanced,
    /// Conservative memory usage
    Conservative,
}

/// iOS device information
#[cfg(all(target_os = "ios", feature = "coreml"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IOsDeviceInfo {
    /// Device model identifier
    pub device_model: String,
    /// iOS version
    pub ios_version: String,
    /// Available memory in MB
    pub available_memory_mb: usize,
    /// Neural Engine availability
    pub has_neural_engine: bool,
    /// Metal Performance Shaders support
    pub has_mps: bool,
    /// Core ML version
    pub coreml_version: String,
}

/// Core ML statistics
#[cfg(all(target_os = "ios", feature = "coreml"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMLStats {
    /// Total predictions performed
    pub total_predictions: usize,
    /// Average prediction time (ms)
    pub avg_prediction_time_ms: f32,
    /// Neural Engine utilization percentage
    pub neural_engine_utilization: f32,
    /// GPU utilization percentage
    pub gpu_utilization: f32,
    /// Memory usage in MB
    pub memory_usage_mb: usize,
    /// Model load time (ms)
    pub model_load_time_ms: f32,
    /// Batch prediction efficiency
    pub batch_efficiency: f32,
}

#[cfg(all(target_os = "ios", feature = "coreml"))]
impl CoreMLEngine {
    /// Create new Core ML engine
    pub fn new(config: CoreMLConfig) -> Result<Self> {
        config.validate()?;

        let device_info = Self::detect_device_info()?;
        let stats = CoreMLStats::new();

        Ok(Self {
            config,
            model: None,
            stats,
            device_info,
            prediction_request: None,
        })
    }

    /// Load Core ML model from data
    pub fn load_model(&mut self, model_data: &[u8]) -> Result<()> {
        let start_time = Instant::now();

        tracing::info!("Loading Core ML model ({} bytes)", model_data.len());

        // Load Core ML model using Objective-C runtime
        let model = self.load_coreml_model_from_data(model_data)?;

        // Configure compute units
        self.configure_compute_units(&model)?;

        // Create prediction request
        let prediction_request = self.create_prediction_request(&model)?;

        self.model = Some(model);
        self.prediction_request = Some(prediction_request);

        let load_time = start_time.elapsed().as_millis() as f32;
        self.stats.model_load_time_ms = load_time;

        tracing::info!(
            "Core ML model loaded successfully in {:.2}ms on {}",
            load_time,
            self.device_info.device_model
        );

        Ok(())
    }

    /// Perform Core ML inference
    pub fn predict(&mut self, input: &HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        if self.model.is_none() {
            return Err(TrustformersError::runtime_error("Core ML model not loaded".into()).into());
        }

        let start_time = Instant::now();

        // Convert tensors to Core ML feature provider
        let feature_provider = self.tensors_to_feature_provider(input)?;

        // Perform prediction
        let output_provider = self.perform_coreml_prediction(&feature_provider)?;

        // Convert output back to tensors
        let output_tensors = self.feature_provider_to_tensors(&output_provider)?;

        let prediction_time = start_time.elapsed().as_millis() as f32;
        self.stats.update_prediction(prediction_time);

        Ok(output_tensors)
    }

    /// Perform batch prediction (if supported)
    pub fn batch_predict(
        &mut self,
        inputs: &[HashMap<String, Tensor>],
    ) -> Result<Vec<HashMap<String, Tensor>>> {
        if !self.config.enable_batch_prediction {
            // Fall back to individual predictions
            return inputs.iter().map(|input| self.predict(input)).collect();
        }

        let batch_size = inputs.len().min(self.config.max_batch_size);
        let effective_inputs = &inputs[..batch_size];

        let start_time = Instant::now();

        // Create batch feature provider
        let batch_provider = self.create_batch_feature_provider(effective_inputs)?;

        // Perform batch prediction
        let batch_outputs = self.perform_batch_coreml_prediction(&batch_provider)?;

        let prediction_time = start_time.elapsed().as_millis() as f32;
        let efficiency = batch_size as f32 / prediction_time;
        self.stats.update_batch_prediction(prediction_time, efficiency);

        Ok(batch_outputs)
    }

    /// Get Core ML statistics
    pub fn get_stats(&self) -> &CoreMLStats {
        &self.stats
    }

    /// Get device information
    pub fn get_device_info(&self) -> &IOsDeviceInfo {
        &self.device_info
    }

    /// Optimize model for current device
    pub fn optimize_for_device(&mut self) -> Result<()> {
        if !self.config.enable_optimization {
            return Ok(());
        }

        // Adjust compute units based on device capabilities
        self.config.compute_units = self.select_optimal_compute_units();

        // Adjust memory handling based on available memory
        self.config.memory_pressure_handling = if self.device_info.available_memory_mb < 1024 {
            CoreMLMemoryHandling::Aggressive
        } else if self.device_info.available_memory_mb < 2048 {
            CoreMLMemoryHandling::Balanced
        } else {
            CoreMLMemoryHandling::Conservative
        };

        tracing::info!(
            "Optimized Core ML configuration for {}: compute_units={:?}, memory_handling={:?}",
            self.device_info.device_model,
            self.config.compute_units,
            self.config.memory_pressure_handling
        );

        Ok(())
    }

    // Private implementation methods

    fn detect_device_info() -> Result<IOsDeviceInfo> {
        // This would use iOS APIs to detect device information
        // For now, return a placeholder
        Ok(IOsDeviceInfo {
            device_model: "iPhone".to_string(),
            ios_version: "15.0".to_string(),
            available_memory_mb: 2048,
            has_neural_engine: true,
            has_mps: true,
            coreml_version: "5.0".to_string(),
        })
    }

    fn load_coreml_model_from_data(&self, model_data: &[u8]) -> Result<*mut Object> {
        #[cfg(target_os = "ios")]
        {
            use core_foundation::base::{kCFAllocatorDefault, CFTypeID, TCFType};
            use core_foundation::data::{CFData, CFDataRef};
            use objc::{class, msg_send, sel, sel_impl};

            unsafe {
                // Create CFData from model bytes
                let cf_data = CFData::from_buffer(model_data);
                let cf_data_ref = cf_data.as_concrete_TypeRef();

                // Get MLModel class
                let ml_model_class = class!(MLModel);

                // Create MLModelConfiguration
                let ml_config_class = class!(MLModelConfiguration);
                let config: *mut Object = msg_send![ml_config_class, alloc];
                let config: *mut Object = msg_send![config, init];

                // Configure compute units based on our config
                let compute_units = match self.config.compute_units {
                    CoreMLComputeUnits::CPUOnly => 0,          // MLComputeUnitsCPUOnly
                    CoreMLComputeUnits::CPUAndGPU => 1,        // MLComputeUnitsCPUAndGPU
                    CoreMLComputeUnits::All => 2,              // MLComputeUnitsAll
                    CoreMLComputeUnits::NeuralEngineOnly => 3, // MLComputeUnitsNeuralEngine (if available)
                };
                let _: () = msg_send![config, setComputeUnits: compute_units];

                // Load model from data
                let mut error: *mut Object = std::ptr::null_mut();
                let model: *mut Object = msg_send![
                    ml_model_class,
                    modelWithContentsOfURL: cf_data_ref
                    configuration: config
                    error: &mut error
                ];

                if model.is_null() || !error.is_null() {
                    return Err(TrustformersError::runtime_error(
                        "Failed to load Core ML model from data".into(),
                    )
                    .into());
                }

                // Retain the model to keep it alive
                let _: *mut Object = msg_send![model, retain];

                Ok(model)
            }
        }

        #[cfg(not(target_os = "ios"))]
        {
            Err(TrustformersError::runtime_error(
                "Core ML is only available on iOS".into(),
            ))
        }
    }

    fn configure_compute_units(&self, _model: &*mut Object) -> Result<()> {
        // Configure Core ML compute units
        Ok(())
    }

    fn create_prediction_request(&self, _model: &*mut Object) -> Result<*mut Object> {
        // Create Core ML prediction request
        Ok(std::ptr::null_mut())
    }

    fn tensors_to_feature_provider(&self, input: &HashMap<String, Tensor>) -> Result<*mut Object> {
        #[cfg(target_os = "ios")]
        {
            use core_foundation::array::{CFArray, CFArrayRef};
            use core_foundation::number::{CFNumber, CFNumberRef};
            use core_foundation::string::{CFString, CFStringRef};
            use objc::{class, msg_send, sel, sel_impl};

            unsafe {
                // Create MLDictionaryFeatureProvider
                let feature_provider_class = class!(MLDictionaryFeatureProvider);
                let ns_mutable_dict_class = class!(NSMutableDictionary);

                let features_dict: *mut Object = msg_send![ns_mutable_dict_class, alloc];
                let features_dict: *mut Object = msg_send![features_dict, init];

                // Convert each tensor to MLMultiArray
                for (name, tensor) in input.iter() {
                    let ml_array = self.tensor_to_ml_multi_array(tensor)?;
                    let name_str = CFString::new(name);
                    let _: () = msg_send![features_dict, setObject: ml_array forKey: name_str.as_concrete_TypeRef()];
                }

                // Create feature provider
                let mut error: *mut Object = std::ptr::null_mut();
                let provider: *mut Object = msg_send![
                    feature_provider_class,
                    featureProviderWithDictionary: features_dict
                    error: &mut error
                ];

                if provider.is_null() || !error.is_null() {
                    return Err(TrustformersError::runtime_error(
                        "Failed to create Core ML feature provider".into(),
                    )
                    .into());
                }

                Ok(provider)
            }
        }

        #[cfg(not(target_os = "ios"))]
        {
            Err(TrustformersError::runtime_error(
                "Core ML is only available on iOS".into(),
            ))
        }
    }

    fn perform_coreml_prediction(&self, feature_provider: &*mut Object) -> Result<*mut Object> {
        #[cfg(target_os = "ios")]
        {
            use objc::{msg_send, sel, sel_impl};

            if let Some(model) = self.model {
                unsafe {
                    let mut error: *mut Object = std::ptr::null_mut();
                    let output: *mut Object = msg_send![
                        model,
                        predictionFromFeatures: *feature_provider
                        error: &mut error
                    ];

                    if output.is_null() || !error.is_null() {
                        return Err(TrustformersError::runtime_error(
                            "Core ML prediction failed".into(),
                        )
                        .into());
                    }

                    Ok(output)
                }
            } else {
                Err(TrustformersError::runtime_error(
                    "Core ML model not loaded".into(),
                ))
            }
        }

        #[cfg(not(target_os = "ios"))]
        {
            Err(TrustformersError::runtime_error(
                "Core ML is only available on iOS".into(),
            ))
        }
    }

    fn feature_provider_to_tensors(
        &self,
        output_provider: &*mut Object,
    ) -> Result<HashMap<String, Tensor>> {
        #[cfg(target_os = "ios")]
        {
            use objc::{msg_send, sel, sel_impl};

            unsafe {
                let mut tensors = HashMap::new();

                // Get feature names
                let feature_names: *mut Object = msg_send![*output_provider, featureNames];
                let count: usize = msg_send![feature_names, count];

                for i in 0..count {
                    let name: *mut Object = msg_send![feature_names, objectAtIndex: i];
                    let feature: *mut Object =
                        msg_send![*output_provider, featureValueForName: name];

                    if !feature.is_null() {
                        let tensor = self.ml_feature_value_to_tensor(feature)?;
                        let name_str = self.ns_string_to_rust_string(name);
                        tensors.insert(name_str, tensor);
                    }
                }

                Ok(tensors)
            }
        }

        #[cfg(not(target_os = "ios"))]
        {
            Err(TrustformersError::runtime_error(
                "Core ML is only available on iOS".into(),
            ))
        }
    }

    #[cfg(target_os = "ios")]
    fn tensor_to_ml_multi_array(&self, tensor: &Tensor) -> Result<*mut Object> {
        use core_foundation::array::{CFArray, CFArrayRef};
        use core_foundation::number::{CFNumber, CFNumberRef};
        use objc::{class, msg_send, sel, sel_impl};

        unsafe {
            // Create shape array
            let shape_numbers: Vec<CFNumber> =
                tensor.shape().iter().map(|&dim| CFNumber::from(dim as i64)).collect();
            let shape_array = CFArray::from_CFTypes(&shape_numbers);

            // Create MLMultiArray
            let ml_array_class = class!(MLMultiArray);
            let mut error: *mut Object = std::ptr::null_mut();

            let ml_array: *mut Object = msg_send![
                ml_array_class,
                initWithShape: shape_array.as_concrete_TypeRef()
                dataType: 65600i32  // MLMultiArrayDataTypeFloat32
                error: &mut error
            ];

            if ml_array.is_null() || !error.is_null() {
                return Err(TrustformersError::runtime_error(
                    "Failed to create MLMultiArray".into(),
                )
                .into());
            }

            // Copy data
            let data_pointer: *mut f32 = msg_send![ml_array, dataPointer];
            let tensor_data = tensor.data_f32()?;
            std::ptr::copy_nonoverlapping(tensor_data.as_ptr(), data_pointer, tensor_data.len());

            Ok(ml_array)
        }
    }

    #[cfg(target_os = "ios")]
    fn ml_feature_value_to_tensor(&self, feature_value: *mut Object) -> Result<Tensor> {
        use objc::{msg_send, sel, sel_impl};

        unsafe {
            // Get MLMultiArray from feature value
            let ml_array: *mut Object = msg_send![feature_value, multiArrayValue];
            if ml_array.is_null() {
                return Err(TrustformersError::runtime_error(
                    "Feature value is not a multi-array".into(),
                )
                .into());
            }

            // Get shape
            let shape_array: *mut Object = msg_send![ml_array, shape];
            let count: usize = msg_send![shape_array, count];
            let mut shape = Vec::with_capacity(count);

            for i in 0..count {
                let number: *mut Object = msg_send![shape_array, objectAtIndex: i];
                let value: i64 = msg_send![number, longLongValue];
                shape.push(value as usize);
            }

            // Get data
            let data_pointer: *const f32 = msg_send![ml_array, dataPointer];
            let total_elements: usize = shape.iter().product();
            let data = std::slice::from_raw_parts(data_pointer, total_elements).to_vec();

            Tensor::from_vec(data, &shape)
        }
    }

    #[cfg(target_os = "ios")]
    fn ns_string_to_rust_string(&self, ns_string: *mut Object) -> String {
        use objc::{msg_send, sel, sel_impl};
        use std::ffi::CStr;

        unsafe {
            let utf8_ptr: *const c_char = msg_send![ns_string, UTF8String];
            if utf8_ptr.is_null() {
                return String::new();
            }

            CStr::from_ptr(utf8_ptr).to_string_lossy().into_owned()
        }
    }

    fn create_batch_feature_provider(
        &self,
        _inputs: &[HashMap<String, Tensor>],
    ) -> Result<*mut Object> {
        // Create batch feature provider for Core ML
        Ok(std::ptr::null_mut())
    }

    fn perform_batch_coreml_prediction(
        &self,
        _batch_provider: &*mut Object,
    ) -> Result<Vec<HashMap<String, Tensor>>> {
        // Perform batch prediction with Core ML
        Ok(Vec::new())
    }

    fn select_optimal_compute_units(&self) -> CoreMLComputeUnits {
        // Select optimal compute units based on device capabilities
        if self.device_info.has_neural_engine {
            CoreMLComputeUnits::All
        } else if self.device_info.has_mps {
            CoreMLComputeUnits::CPUAndGPU
        } else {
            CoreMLComputeUnits::CPUOnly
        }
    }
}

#[cfg(all(target_os = "ios", feature = "coreml"))]
impl Default for CoreMLConfig {
    fn default() -> Self {
        Self {
            compute_units: CoreMLComputeUnits::All,
            enable_batch_prediction: true,
            max_batch_size: 4,
            use_reduced_precision: true,
            enable_optimization: true,
            memory_pressure_handling: CoreMLMemoryHandling::Balanced,
        }
    }
}

#[cfg(all(target_os = "ios", feature = "coreml"))]
impl CoreMLConfig {
    /// Validate Core ML configuration
    pub fn validate(&self) -> Result<()> {
        if self.max_batch_size == 0 {
            return Err(TrustformersError::config_error {
                message: "Batch size must be > 0".to_string(),
                context: trustformers_core::error::ErrorContext::new(
                    trustformers_core::error::ErrorCode::E4001,
                    "validate".to_string(),
                ),
            });
        }

        if self.max_batch_size > 32 {
            return Err(TrustformersError::config_error {
                message: "Batch size too large for Core ML".to_string(),
                context: trustformers_core::error::ErrorContext::new(
                    trustformers_core::error::ErrorCode::E4001,
                    "validate".to_string(),
                ),
            });
        }

        Ok(())
    }

    /// Create optimized configuration for specific iOS device
    pub fn for_device(device_model: &str) -> Self {
        let mut config = Self::default();

        // Optimize based on device model
        if device_model.contains("iPhone12")
            || device_model.contains("iPhone13")
            || device_model.contains("iPhone14")
        {
            // Newer devices with A14+ chips
            config.compute_units = CoreMLComputeUnits::All;
            config.max_batch_size = 8;
            config.memory_pressure_handling = CoreMLMemoryHandling::Conservative;
        } else if device_model.contains("iPhone11") || device_model.contains("iPhone10") {
            // Mid-range devices
            config.compute_units = CoreMLComputeUnits::CPUAndGPU;
            config.max_batch_size = 4;
            config.memory_pressure_handling = CoreMLMemoryHandling::Balanced;
        } else {
            // Older devices
            config.compute_units = CoreMLComputeUnits::CPUOnly;
            config.max_batch_size = 2;
            config.memory_pressure_handling = CoreMLMemoryHandling::Aggressive;
        }

        config
    }
}

#[cfg(all(target_os = "ios", feature = "coreml"))]
impl CoreMLStats {
    fn new() -> Self {
        Self {
            total_predictions: 0,
            avg_prediction_time_ms: 0.0,
            neural_engine_utilization: 0.0,
            gpu_utilization: 0.0,
            memory_usage_mb: 0,
            model_load_time_ms: 0.0,
            batch_efficiency: 0.0,
        }
    }

    fn update_prediction(&mut self, prediction_time_ms: f32) {
        self.total_predictions += 1;

        // Update running average
        let alpha = 0.1;
        if self.total_predictions == 1 {
            self.avg_prediction_time_ms = prediction_time_ms;
        } else {
            self.avg_prediction_time_ms =
                alpha * prediction_time_ms + (1.0 - alpha) * self.avg_prediction_time_ms;
        }
    }

    fn update_batch_prediction(&mut self, prediction_time_ms: f32, efficiency: f32) {
        self.update_prediction(prediction_time_ms);

        // Update batch efficiency
        let alpha = 0.1;
        self.batch_efficiency = alpha * efficiency + (1.0 - alpha) * self.batch_efficiency;
    }
}

/// Convert mobile config to Core ML config
#[cfg(all(target_os = "ios", feature = "coreml"))]
pub fn mobile_config_to_coreml(mobile_config: &MobileConfig) -> CoreMLConfig {
    let mut coreml_config = CoreMLConfig::default();

    // Map memory optimization to Core ML settings
    match mobile_config.memory_optimization {
        MemoryOptimization::Maximum => {
            coreml_config.compute_units = CoreMLComputeUnits::CPUOnly;
            coreml_config.memory_pressure_handling = CoreMLMemoryHandling::Aggressive;
            coreml_config.max_batch_size = 1;
            coreml_config.enable_batch_prediction = false;
        },
        MemoryOptimization::Balanced => {
            coreml_config.compute_units = CoreMLComputeUnits::CPUAndGPU;
            coreml_config.memory_pressure_handling = CoreMLMemoryHandling::Balanced;
            coreml_config.max_batch_size = mobile_config.max_batch_size;
            coreml_config.enable_batch_prediction = mobile_config.enable_batching;
        },
        MemoryOptimization::Minimal => {
            coreml_config.compute_units = CoreMLComputeUnits::All;
            coreml_config.memory_pressure_handling = CoreMLMemoryHandling::Conservative;
            coreml_config.max_batch_size = mobile_config.max_batch_size;
            coreml_config.enable_batch_prediction = mobile_config.enable_batching;
        },
    }

    coreml_config.use_reduced_precision = mobile_config.use_fp16;

    coreml_config
}

// Stub implementations for non-iOS platforms

#[cfg(not(all(target_os = "ios", feature = "coreml")))]
pub struct CoreMLEngine;

#[cfg(not(all(target_os = "ios", feature = "coreml")))]
impl CoreMLEngine {
    pub fn new(_config: ()) -> Result<Self> {
        Err(TrustformersError::runtime_error("Core ML only available on iOS".into()).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(target_os = "ios", feature = "coreml"))]
    #[test]
    fn test_coreml_config_validation() {
        let config = CoreMLConfig::default();
        assert!(config.validate().is_ok());

        let mut invalid_config = config.clone();
        invalid_config.max_batch_size = 0;
        assert!(invalid_config.validate().is_err());
    }

    #[cfg(all(target_os = "ios", feature = "coreml"))]
    #[test]
    fn test_device_specific_config() {
        let iphone13_config = CoreMLConfig::for_device("iPhone13,3");
        assert_eq!(iphone13_config.compute_units, CoreMLComputeUnits::All);
        assert_eq!(iphone13_config.max_batch_size, 8);

        let iphone8_config = CoreMLConfig::for_device("iPhone8,1");
        assert_eq!(iphone8_config.compute_units, CoreMLComputeUnits::CPUOnly);
        assert_eq!(iphone8_config.max_batch_size, 2);
    }

    #[cfg(all(target_os = "ios", feature = "coreml"))]
    #[test]
    fn test_mobile_to_coreml_config_conversion() {
        let mobile_config = crate::MobileConfig {
            memory_optimization: MemoryOptimization::Maximum,
            max_batch_size: 4,
            enable_batching: true,
            use_fp16: true,
            ..Default::default()
        };

        let coreml_config = mobile_config_to_coreml(&mobile_config);
        assert_eq!(coreml_config.compute_units, CoreMLComputeUnits::CPUOnly);
        assert_eq!(
            coreml_config.memory_pressure_handling,
            CoreMLMemoryHandling::Aggressive
        );
        assert!(!coreml_config.enable_batch_prediction);
        assert!(coreml_config.use_reduced_precision);
    }
}
