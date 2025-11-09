//! NNAPI Execution and Inference Management
//!
//! This module handles NNAPI model compilation, execution, and device management
//! for efficient neural network inference on Android devices.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr;
use trustformers_core::{Tensor};
use trustformers_core::error::{CoreError, Result};

use super::bindings::*;
use super::types::*;

/// NNAPI execution engine for running compiled models
pub struct NNAPIExecutor {
    model: *mut ANeuralNetworksModel,
    compilation: *mut ANeuralNetworksCompilation,
    input_count: usize,
    output_count: usize,
    input_operands: Vec<u32>,
    output_operands: Vec<u32>,
}

impl NNAPIExecutor {
    /// Create a new NNAPI executor from a model
    #[cfg(target_os = "android")]
    pub fn new(
        model: *mut ANeuralNetworksModel,
        input_count: usize,
        output_count: usize,
        input_operands: Vec<u32>,
        output_operands: Vec<u32>,
    ) -> Result<Self> {
        if model.is_null() {
            return Err(TrustformersError::runtime_error("Model pointer is null".into()).into());
        }

        let mut executor = Self {
            model,
            compilation: ptr::null_mut(),
            input_count,
            output_count,
            input_operands,
            output_operands,
        };

        // Create compilation
        executor.create_compilation()?;

        Ok(executor)
    }

    #[cfg(not(target_os = "android"))]
    pub fn new(
        _model: *mut ANeuralNetworksModel,
        _input_count: usize,
        _output_count: usize,
        _input_operands: Vec<u32>,
        _output_operands: Vec<u32>,
    ) -> Result<Self> {
        Err(TrustformersError::runtime_error(
            "NNAPI is only available on Android".into(),
        ))
    }

    /// Create compilation for the model
    #[cfg(target_os = "android")]
    fn create_compilation(&mut self) -> Result<()> {
        let result = unsafe { ANeuralNetworksCompilation_create(self.model, &mut self.compilation) };
        if !nnapi_is_success(result) {
            return Err(TrustformersError::runtime_error(format!(
                "Failed to create NNAPI compilation: {}",
                nnapi_result_to_string(result)
            )).into());
        }

        // Set execution preference for sustained performance
        let result = unsafe {
            ANeuralNetworksCompilation_setPreference(
                self.compilation,
                ANEURALNETWORKS_PREFER_SUSTAINED_SPEED,
            )
        };
        if !nnapi_is_success(result) {
            tracing::warn!("Failed to set NNAPI preference: {}", nnapi_result_to_string(result));
        }

        // Finish compilation
        let result = unsafe { ANeuralNetworksCompilation_finish(self.compilation) };
        if !nnapi_is_success(result) {
            unsafe { ANeuralNetworksCompilation_free(self.compilation) };
            self.compilation = ptr::null_mut();
            return Err(TrustformersError::runtime_error(format!(
                "Failed to finish NNAPI compilation: {}",
                nnapi_result_to_string(result)
            )).into());
        }

        tracing::info!("NNAPI compilation created successfully");
        Ok(())
    }

    /// Execute inference with the given input tensor
    #[cfg(target_os = "android")]
    pub fn execute(&self, input: &Tensor) -> Result<Tensor> {
        if self.compilation.is_null() {
            return Err(TrustformersError::runtime_error("Model not compiled".into()).into());
        }

        // Create execution
        let mut execution: *mut ANeuralNetworksExecution = ptr::null_mut();
        let result = unsafe { ANeuralNetworksExecution_create(self.compilation, &mut execution) };
        if !nnapi_is_success(result) {
            return Err(TrustformersError::runtime_error(format!(
                "Failed to create NNAPI execution: {}",
                nnapi_result_to_string(result)
            )).into());
        }

        // Convert input tensor to NNAPI format
        let input_data = input.as_slice::<f32>()?;
        let input_dims = input.shape();

        // Set input
        let input_type = create_tensor_operand_type(
            ANEURALNETWORKS_TENSOR_FLOAT32,
            &input_dims.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            0.0,
            0,
        );

        let input_size = input_data.len() * std::mem::size_of::<f32>();
        let result = unsafe {
            ANeuralNetworksExecution_setInput(
                execution,
                0, // Input index
                &input_type,
                input_data.as_ptr() as *const c_void,
                input_size,
            )
        };

        if !nnapi_is_success(result) {
            unsafe { ANeuralNetworksExecution_free(execution) };
            return Err(TrustformersError::runtime_error(format!(
                "Failed to set NNAPI input: {}",
                nnapi_result_to_string(result)
            )).into());
        }

        // Allocate output buffer (simplified - assuming known output shape)
        let output_size = 1 * 222 * 222 * 32; // From example Conv2D model
        let mut output_buffer = vec![0f32; output_size];

        // Set output
        let output_dims = [1u32, 222, 222, 32];
        let output_type = create_tensor_operand_type(
            ANEURALNETWORKS_TENSOR_FLOAT32,
            &output_dims,
            0.0,
            0,
        );

        let output_size_bytes = output_size * std::mem::size_of::<f32>();
        let result = unsafe {
            ANeuralNetworksExecution_setOutput(
                execution,
                0, // Output index
                &output_type,
                output_buffer.as_mut_ptr() as *mut c_void,
                output_size_bytes,
            )
        };

        if !nnapi_is_success(result) {
            unsafe { ANeuralNetworksExecution_free(execution) };
            return Err(TrustformersError::runtime_error(format!(
                "Failed to set NNAPI output: {}",
                nnapi_result_to_string(result)
            )).into());
        }

        // Execute computation
        let mut event: *mut ANeuralNetworksEvent = ptr::null_mut();
        let result = unsafe { ANeuralNetworksExecution_startCompute(execution, &mut event) };
        if !nnapi_is_success(result) {
            unsafe { ANeuralNetworksExecution_free(execution) };
            return Err(TrustformersError::runtime_error(format!(
                "Failed to start NNAPI computation: {}",
                nnapi_result_to_string(result)
            )).into());
        }

        // Wait for completion
        let result = unsafe { ANeuralNetworksEvent_wait(event) };
        if !nnapi_is_success(result) {
            unsafe {
                ANeuralNetworksEvent_free(event);
                ANeuralNetworksExecution_free(execution);
            }
            return Err(TrustformersError::runtime_error(format!(
                "NNAPI computation failed: {}",
                nnapi_result_to_string(result)
            )).into());
        }

        // Clean up
        unsafe {
            ANeuralNetworksEvent_free(event);
            ANeuralNetworksExecution_free(execution);
        }

        // Create output tensor
        let output_shape = [1, 222, 222, 32];
        Tensor::from_vec(output_buffer, &output_shape)
    }

    #[cfg(not(target_os = "android"))]
    pub fn execute(&self, _input: &Tensor) -> Result<Tensor> {
        Err(TrustformersError::runtime_error(
            "NNAPI execution is only available on Android".into(),
        ))
    }

    /// Get input count
    pub fn get_input_count(&self) -> usize {
        self.input_count
    }

    /// Get output count
    pub fn get_output_count(&self) -> usize {
        self.output_count
    }

    /// Check if executor is ready
    pub fn is_ready(&self) -> bool {
        !self.compilation.is_null()
    }
}

impl Drop for NNAPIExecutor {
    fn drop(&mut self) {
        #[cfg(target_os = "android")]
        if !self.compilation.is_null() {
            unsafe {
                ANeuralNetworksCompilation_free(self.compilation);
            }
        }
    }
}

/// NNAPI device detection and management
pub struct NNAPIDeviceManager;

impl NNAPIDeviceManager {
    /// Detect all available NNAPI devices
    #[cfg(target_os = "android")]
    pub fn detect_devices() -> Vec<NNAPIDeviceInfo> {
        let mut devices = Vec::new();
        let mut device_count: u32 = 0;

        // Get number of available NNAPI devices
        let result = unsafe { ANeuralNetworks_getDeviceCount(&mut device_count) };
        if !nnapi_is_success(result) {
            tracing::warn!("Failed to get NNAPI device count: {}", nnapi_result_to_string(result).into());
            return devices;
        }

        tracing::info!("Found {} NNAPI devices", device_count);

        // Enumerate each device
        for i in 0..device_count {
            if let Ok(device_info) = Self::get_device_info(i) {
                devices.push(device_info);
            }
        }

        devices
    }

    #[cfg(not(target_os = "android"))]
    pub fn detect_devices() -> Vec<NNAPIDeviceInfo> {
        Vec::new()
    }

    /// Get information for a specific device
    #[cfg(target_os = "android")]
    fn get_device_info(device_index: u32) -> Result<NNAPIDeviceInfo> {
        let mut device_ptr: *mut c_void = ptr::null_mut();

        // Get device handle
        let result = unsafe { ANeuralNetworks_getDevice(device_index, &mut device_ptr) };
        if !nnapi_is_success(result) {
            return Err(TrustformersError::runtime_error(format!(
                "Failed to get NNAPI device {}: {}",
                device_index,
                nnapi_result_to_string(result)
            )).into());
        }

        // Get device name
        let mut name_ptr: *const c_char = ptr::null();
        let name = unsafe {
            let result = ANeuralNetworks_getDeviceName(device_ptr, &mut name_ptr);
            if nnapi_is_success(result) && !name_ptr.is_null() {
                CStr::from_ptr(name_ptr).to_string_lossy().into_owned()
            } else {
                format!("Unknown Device {}", device_index)
            }
        };

        // Get device type
        let mut device_type: i32 = ANEURALNETWORKS_DEVICE_UNKNOWN;
        let result = unsafe { ANeuralNetworks_getDeviceType(device_ptr, &mut device_type) };
        if !nnapi_is_success(result) {
            tracing::warn!("Failed to get device type for {}: {}", name, nnapi_result_to_string(result));
        }

        // Get feature level
        let mut feature_level: i32 = 27; // Default to API level 27
        let result =
            unsafe { ANeuralNetworks_getDeviceFeatureLevel(device_ptr, &mut feature_level) };
        if !nnapi_is_success(result) {
            tracing::warn!("Failed to get feature level for {}: {}", name, nnapi_result_to_string(result));
        }

        // Get performance info
        let mut performance_info = ANeuralNetworksPerformanceInfo {
            exec_time: 0.0,
            power_usage: 0.0,
        };
        let result =
            unsafe { ANeuralNetworks_getDevicePerformanceInfo(device_ptr, &mut performance_info) };
        if !nnapi_is_success(result) {
            tracing::warn!("Failed to get performance info for {}: {}", name, nnapi_result_to_string(result));
        }

        // Check for vendor extensions
        let vendor_extensions = Self::get_vendor_extensions(device_ptr);

        let device_info = NNAPIDeviceInfo {
            index: device_index,
            device_ptr,
            name: name.clone(),
            device_type,
            feature_level,
            performance_info,
            vendor_extensions,
        };

        tracing::info!(
            "NNAPI Device {}: {} (Type: {}, Feature Level: {})",
            device_index,
            name,
            device_type_to_string(device_type),
            feature_level
        );

        Ok(device_info)
    }

    /// Get vendor extensions for a device
    #[cfg(target_os = "android")]
    fn get_vendor_extensions(device_ptr: *mut c_void) -> Vec<String> {
        let mut extensions = Vec::new();

        // Check for common vendor extensions
        let extension_names = [
            "com.qualcomm.qti.nnapi.extension",
            "com.mediatek.nnapi.extension",
            "com.samsung.android.npu.extension",
            "com.arm.compute.nnapi.extension",
            "com.google.android.gni.extension",
        ];

        for &extension_name in &extension_names {
            if let Ok(extension_cstr) = CString::new(extension_name) {
                let mut is_supported = false;

                let result = unsafe {
                    ANeuralNetworks_getDeviceExtensionSupport(
                        device_ptr,
                        extension_cstr.as_ptr(),
                        &mut is_supported,
                    )
                };

                if nnapi_is_success(result) && is_supported {
                    extensions.push(extension_name.to_string());
                    tracing::info!("Vendor extension supported: {}", extension_name);
                }
            }
        }

        extensions
    }

    /// Get the best NNAPI device for inference
    pub fn get_best_device() -> Option<NNAPIDeviceInfo> {
        let devices = Self::detect_devices();

        if devices.is_empty() {
            return None;
        }

        // Priority order: NPU/Accelerator > GPU > CPU > Other
        let mut best_device = None;
        let mut best_priority = -1;

        for device in devices {
            let priority = match device.device_type {
                ANEURALNETWORKS_DEVICE_ACCELERATOR => 3,
                ANEURALNETWORKS_DEVICE_GPU => 2,
                ANEURALNETWORKS_DEVICE_CPU => 1,
                _ => 0,
            };

            if priority > best_priority {
                best_priority = priority;
                best_device = Some(device);
            }
        }

        best_device
    }

    /// Check if hardware acceleration is available
    pub fn has_hardware_acceleration() -> bool {
        let devices = Self::detect_devices();
        devices.iter().any(|d| {
            d.device_type == ANEURALNETWORKS_DEVICE_GPU
                || d.device_type == ANEURALNETWORKS_DEVICE_ACCELERATOR
        })
    }
}

/// Convert NNAPI device type to human-readable string
pub fn device_type_to_string(device_type: i32) -> &'static str {
    match device_type {
        ANEURALNETWORKS_DEVICE_CPU => "CPU",
        ANEURALNETWORKS_DEVICE_GPU => "GPU",
        ANEURALNETWORKS_DEVICE_ACCELERATOR => "NPU/Accelerator",
        ANEURALNETWORKS_DEVICE_OTHER => "Other",
        _ => "Unknown",
    }
}

/// Create tensor operand type (helper function)
fn create_tensor_operand_type(
    data_type: i32,
    dimensions: &[u32],
    scale: f32,
    zero_point: i32,
) -> ANeuralNetworksOperandType {
    ANeuralNetworksOperandType {
        type_: data_type,
        dimensionCount: dimensions.len() as u32,
        dimensions: dimensions.as_ptr(),
        scale,
        zeroPoint: zero_point,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_type_string() {
        assert_eq!(device_type_to_string(ANEURALNETWORKS_DEVICE_CPU), "CPU");
        assert_eq!(device_type_to_string(ANEURALNETWORKS_DEVICE_GPU), "GPU");
        assert_eq!(device_type_to_string(ANEURALNETWORKS_DEVICE_ACCELERATOR), "NPU/Accelerator");
        assert_eq!(device_type_to_string(999), "Unknown");
    }

    #[test]
    fn test_device_detection() {
        let devices = NNAPIDeviceManager::detect_devices();
        // This will return empty on non-Android platforms
        tracing::info!("Detected {} NNAPI devices", devices.len());
    }

    #[test]
    fn test_hardware_acceleration() {
        let has_hw_accel = NNAPIDeviceManager::has_hardware_acceleration();
        tracing::info!("Hardware acceleration available: {}", has_hw_accel);
    }

    #[cfg(target_os = "android")]
    #[test]
    fn test_executor_creation() {
        use super::super::model::ExampleConv2DModel;

        let model = ExampleConv2DModel::new();
        if model.is_err() {
            // NNAPI might not be available in test environment
            return;
        }

        let model = model.unwrap();
        let executor = NNAPIExecutor::new(
            model.get_model_ptr(),
            1, // input count
            1, // output count
            vec![model.get_input_index()],
            vec![model.get_output_index()],
        );

        if executor.is_err() {
            // NNAPI might not be available or configured
            return;
        }

        let executor = executor.unwrap();
        assert!(executor.is_ready());
        assert_eq!(executor.get_input_count(), 1);
        assert_eq!(executor.get_output_count(), 1);
    }
}
