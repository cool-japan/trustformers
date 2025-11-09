//! Android Neural Networks API (NNAPI) Support
//!
//! This module provides low-level bindings and utilities for Android's NNAPI,
//! enabling hardware-accelerated neural network inference on Android devices.

use std::os::raw::{c_char, c_float, c_int, c_uint, c_void};
use trustformers_core::error::{CoreError, Result};

#[cfg(target_os = "android")]
pub struct NNAPIModel {
    pub model: *mut ANeuralNetworksModel,
    pub compilation: *mut ANeuralNetworksCompilation,
    pub execution: *mut ANeuralNetworksExecution,
    pub input_count: usize,
    pub output_count: usize,
    pub input_operands: Vec<u32>,
    pub output_operands: Vec<u32>,
}

#[cfg(target_os = "android")]
#[repr(C)]
pub struct ANeuralNetworksModel;

#[cfg(target_os = "android")]
#[repr(C)]
pub struct ANeuralNetworksCompilation;

#[cfg(target_os = "android")]
#[repr(C)]
pub struct ANeuralNetworksExecution;

#[cfg(target_os = "android")]
#[repr(C)]
pub struct ANeuralNetworksOperandType {
    pub type_: i32,
    pub dimensionCount: u32,
    pub dimensions: *const u32,
    pub scale: f32,
    pub zeroPoint: i32,
}

#[cfg(target_os = "android")]
#[repr(C)]
pub struct ANeuralNetworksEvent;

// NNAPI Constants
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_NO_ERROR: i32 = 0;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_TENSOR_FLOAT32: i32 = 0;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_TENSOR_INT32: i32 = 1;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_TENSOR_QUANT8_ASYMM: i32 = 2;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_FLOAT32: i32 = 3;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_INT32: i32 = 4;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_UINT32: i32 = 5;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_TENSOR_FLOAT16: i32 = 6;

// NNAPI Operation Types
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_ADD: i32 = 0;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_AVERAGE_POOL_2D: i32 = 1;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_CONCATENATION: i32 = 2;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_CONV_2D: i32 = 3;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_DEPTHWISE_CONV_2D: i32 = 4;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_DEPTH_TO_SPACE: i32 = 5;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_DEQUANTIZE: i32 = 6;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_EMBEDDING_LOOKUP: i32 = 7;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_FLOOR: i32 = 8;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_FULLY_CONNECTED: i32 = 9;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_HASHTABLE_LOOKUP: i32 = 10;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_L2_NORMALIZATION: i32 = 11;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_L2_POOL_2D: i32 = 12;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION: i32 = 13;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_LOGISTIC: i32 = 14;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_LSH_PROJECTION: i32 = 15;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_LSTM: i32 = 16;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_MAX_POOL_2D: i32 = 17;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_MUL: i32 = 18;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_RELU: i32 = 19;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_RELU1: i32 = 20;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_RELU6: i32 = 21;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_RESHAPE: i32 = 22;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_RESIZE_BILINEAR: i32 = 23;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_RNN: i32 = 24;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_SOFTMAX: i32 = 25;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_SPACE_TO_DEPTH: i32 = 26;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_SVDF: i32 = 27;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_TANH: i32 = 28;

// NNAPI Execution Preferences
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_PREFER_LOW_POWER: i32 = 0;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER: i32 = 1;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_PREFER_SUSTAINED_SPEED: i32 = 2;

// NNAPI Device Types
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_DEVICE_UNKNOWN: i32 = 0;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_DEVICE_OTHER: i32 = 1;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_DEVICE_CPU: i32 = 2;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_DEVICE_GPU: i32 = 3;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_DEVICE_ACCELERATOR: i32 = 4;

// NNAPI Performance Info Structure
#[cfg(target_os = "android")]
#[repr(C)]
pub struct ANeuralNetworksPerformanceInfo {
    pub exec_time: f32,
    pub power_usage: f32,
}

// NNAPI Device Information Structure
#[cfg(target_os = "android")]
#[derive(Debug, Clone)]
pub struct NNAPIDeviceInfo {
    pub index: u32,
    pub device_ptr: *mut c_void,
    pub name: String,
    pub device_type: i32,
    pub feature_level: i32,
    pub performance_info: ANeuralNetworksPerformanceInfo,
    pub vendor_extensions: Vec<String>,
}

// NNAPI Function Bindings
#[cfg(target_os = "android")]
extern "C" {
    pub fn ANeuralNetworksModel_create(model: *mut *mut ANeuralNetworksModel) -> i32;
    pub fn ANeuralNetworksModel_free(model: *mut ANeuralNetworksModel);
    pub fn ANeuralNetworksModel_finish(model: *mut ANeuralNetworksModel) -> i32;
    pub fn ANeuralNetworksModel_addOperand(
        model: *mut ANeuralNetworksModel,
        type_: *const ANeuralNetworksOperandType,
    ) -> i32;
    pub fn ANeuralNetworksModel_setOperandValue(
        model: *mut ANeuralNetworksModel,
        index: i32,
        buffer: *const c_void,
        length: usize,
    ) -> i32;
    pub fn ANeuralNetworksModel_addOperation(
        model: *mut ANeuralNetworksModel,
        type_: i32,
        input_count: u32,
        inputs: *const u32,
        output_count: u32,
        outputs: *const u32,
    ) -> i32;
    pub fn ANeuralNetworksModel_identifyInputsAndOutputs(
        model: *mut ANeuralNetworksModel,
        input_count: u32,
        inputs: *const u32,
        output_count: u32,
        outputs: *const u32,
    ) -> i32;
    pub fn ANeuralNetworksCompilation_create(
        model: *mut ANeuralNetworksModel,
        compilation: *mut *mut ANeuralNetworksCompilation,
    ) -> i32;
    pub fn ANeuralNetworksCompilation_free(compilation: *mut ANeuralNetworksCompilation);
    pub fn ANeuralNetworksCompilation_setPreference(
        compilation: *mut ANeuralNetworksCompilation,
        preference: i32,
    ) -> i32;
    pub fn ANeuralNetworksCompilation_finish(compilation: *mut ANeuralNetworksCompilation) -> i32;
    pub fn ANeuralNetworksExecution_create(
        compilation: *mut ANeuralNetworksCompilation,
        execution: *mut *mut ANeuralNetworksExecution,
    ) -> i32;
    pub fn ANeuralNetworksExecution_free(execution: *mut ANeuralNetworksExecution);
    pub fn ANeuralNetworksExecution_setInput(
        execution: *mut ANeuralNetworksExecution,
        index: i32,
        type_: *const ANeuralNetworksOperandType,
        buffer: *const c_void,
        length: usize,
    ) -> i32;
    pub fn ANeuralNetworksExecution_setOutput(
        execution: *mut ANeuralNetworksExecution,
        index: i32,
        type_: *const ANeuralNetworksOperandType,
        buffer: *mut c_void,
        length: usize,
    ) -> i32;
    pub fn ANeuralNetworksExecution_startCompute(
        execution: *mut ANeuralNetworksExecution,
        event: *mut *mut ANeuralNetworksEvent,
    ) -> i32;
    pub fn ANeuralNetworksEvent_wait(event: *mut ANeuralNetworksEvent) -> i32;
    pub fn ANeuralNetworksEvent_free(event: *mut ANeuralNetworksEvent);

    pub fn ANeuralNetworks_getDeviceCount(numDevices: *mut u32) -> i32;
    pub fn ANeuralNetworks_getDevice(devIndex: u32, device: *mut *mut c_void) -> i32;
    pub fn ANeuralNetworks_getDeviceName(device: *mut c_void, name: *mut *const c_char) -> i32;
    pub fn ANeuralNetworks_getDeviceType(device: *mut c_void, device_type: *mut i32) -> i32;
    pub fn ANeuralNetworks_getDeviceFeatureLevel(
        device: *mut c_void,
        feature_level: *mut i32,
    ) -> i32;
    pub fn ANeuralNetworks_getDeviceExtensionSupport(
        device: *mut c_void,
        extension_name: *const c_char,
        is_supported: *mut bool,
    ) -> i32;
    pub fn ANeuralNetworks_getDevicePerformanceInfo(
        device: *mut c_void,
        performance_info: *mut ANeuralNetworksPerformanceInfo,
    ) -> i32;
}

#[cfg(target_os = "android")]
impl NNAPIModel {
    pub fn new() -> Result<Self> {
        Ok(Self {
            model: std::ptr::null_mut(),
            compilation: std::ptr::null_mut(),
            execution: std::ptr::null_mut(),
            input_count: 0,
            output_count: 0,
            input_operands: Vec::new(),
            output_operands: Vec::new(),
        })
    }

    pub fn is_loaded(&self) -> bool {
        !self.model.is_null()
    }

    pub fn get_input_count(&self) -> usize {
        self.input_count
    }

    pub fn get_output_count(&self) -> usize {
        self.output_count
    }
}

#[cfg(target_os = "android")]
impl Drop for NNAPIModel {
    fn drop(&mut self) {
        unsafe {
            if !self.execution.is_null() {
                ANeuralNetworksExecution_free(self.execution);
            }
            if !self.compilation.is_null() {
                ANeuralNetworksCompilation_free(self.compilation);
            }
            if !self.model.is_null() {
                ANeuralNetworksModel_free(self.model);
            }
        }
    }
}

// Non-Android stub implementations
#[cfg(not(target_os = "android"))]
pub struct NNAPIModel;

#[cfg(not(target_os = "android"))]
impl NNAPIModel {
    pub fn new() -> Result<Self> {
        Ok(Self)
    }

    pub fn is_loaded(&self) -> bool {
        false
    }

    pub fn get_input_count(&self) -> usize {
        0
    }

    pub fn get_output_count(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nnapi_model_creation() {
        let model = NNAPIModel::new();
        assert!(model.is_ok());
    }

    #[test]
    fn test_nnapi_model_default_state() {
        let model = NNAPIModel::new().unwrap();
        #[cfg(target_os = "android")]
        {
            assert!(!model.is_loaded());
            assert_eq!(model.get_input_count(), 0);
            assert_eq!(model.get_output_count(), 0);
        }
        #[cfg(not(target_os = "android"))]
        {
            assert!(!model.is_loaded());
            assert_eq!(model.get_input_count(), 0);
            assert_eq!(model.get_output_count(), 0);
        }
    }

    #[cfg(target_os = "android")]
    #[test]
    fn test_nnapi_constants() {
        assert_eq!(ANEURALNETWORKS_NO_ERROR, 0);
        assert_eq!(ANEURALNETWORKS_TENSOR_FLOAT32, 0);
        assert_eq!(ANEURALNETWORKS_ADD, 0);
        assert_eq!(ANEURALNETWORKS_PREFER_LOW_POWER, 0);
        assert_eq!(ANEURALNETWORKS_DEVICE_CPU, 2);
    }
}
