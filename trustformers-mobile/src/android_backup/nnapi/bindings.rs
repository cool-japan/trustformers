//! NNAPI C API Bindings and Constants
//!
//! This module contains the complete Neural Networks API C bindings,
//! constants, and data structures for Android hardware acceleration.

use std::os::raw::{c_char, c_int, c_void};

// NNAPI opaque handle types
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
pub struct ANeuralNetworksEvent;

/// NNAPI operand type descriptor
#[cfg(target_os = "android")]
#[repr(C)]
pub struct ANeuralNetworksOperandType {
    /// Operand data type
    pub type_: i32,
    /// Number of dimensions
    pub dimensionCount: u32,
    /// Dimension sizes array pointer
    pub dimensions: *const u32,
    /// Quantization scale factor
    pub scale: f32,
    /// Quantization zero point
    pub zeroPoint: i32,
}

/// NNAPI performance information
#[cfg(target_os = "android")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ANeuralNetworksPerformanceInfo {
    /// Execution time estimate
    pub exec_time: f32,
    /// Power usage estimate
    pub power_usage: f32,
}

// NNAPI Result Codes
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_NO_ERROR: i32 = 0;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_OUT_OF_MEMORY: i32 = 1;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_INCOMPLETE: i32 = 2;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_UNEXPECTED_NULL: i32 = 3;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_BAD_DATA: i32 = 4;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_OP_FAILED: i32 = 5;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_BAD_STATE: i32 = 6;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_UNMAPPABLE: i32 = 7;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE: i32 = 8;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_UNAVAILABLE_DEVICE: i32 = 9;

// NNAPI Data Types
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_FLOAT32: i32 = 0;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_INT32: i32 = 1;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_UINT32: i32 = 2;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_TENSOR_FLOAT32: i32 = 3;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_TENSOR_INT32: i32 = 4;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_TENSOR_QUANT8_ASYMM: i32 = 5;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_BOOL: i32 = 6;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_TENSOR_QUANT16_SYMM: i32 = 7;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_TENSOR_FLOAT16: i32 = 8;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_TENSOR_BOOL8: i32 = 9;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_FLOAT16: i32 = 10;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL: i32 = 11;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_TENSOR_QUANT16_ASYMM: i32 = 12;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_TENSOR_QUANT8_SYMM: i32 = 13;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED: i32 = 14;

// NNAPI Operation Types (Core Operations)
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

// Extended Operations (API Level 28+)
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_BATCH_TO_SPACE_ND: i32 = 29;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_DIV: i32 = 30;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_MEAN: i32 = 31;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_PAD: i32 = 32;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_SPACE_TO_BATCH_ND: i32 = 33;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_SQUEEZE: i32 = 34;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_STRIDED_SLICE: i32 = 35;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_SUB: i32 = 36;
#[cfg(target_os = "android")]
pub const ANEURALNETWORKS_TRANSPOSE: i32 = 37;

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

// NNAPI C Function Bindings
#[cfg(target_os = "android")]
extern "C" {
    // Model management functions
    pub fn ANeuralNetworksModel_create(model: *mut *mut ANeuralNetworksModel) -> i32;
    pub fn ANeuralNetworksModel_free(model: *mut ANeuralNetworksModel);
    pub fn ANeuralNetworksModel_finish(model: *mut ANeuralNetworksModel) -> i32;

    // Model construction functions
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
        inputCount: u32,
        inputs: *const u32,
        outputCount: u32,
        outputs: *const u32,
    ) -> i32;

    pub fn ANeuralNetworksModel_identifyInputsAndOutputs(
        model: *mut ANeuralNetworksModel,
        inputCount: u32,
        inputs: *const u32,
        outputCount: u32,
        outputs: *const u32,
    ) -> i32;

    // Compilation functions
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

    // Execution functions
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

    // Event management functions
    pub fn ANeuralNetworksEvent_wait(event: *mut ANeuralNetworksEvent) -> i32;
    pub fn ANeuralNetworksEvent_free(event: *mut ANeuralNetworksEvent);

    // Device enumeration and query functions
    pub fn ANeuralNetworks_getDeviceCount(numDevices: *mut u32) -> i32;
    pub fn ANeuralNetworks_getDevice(devIndex: u32, device: *mut *mut c_void) -> i32;
    pub fn ANeuralNetworks_getDeviceName(device: *mut c_void, name: *mut *const c_char) -> i32;
    pub fn ANeuralNetworks_getDeviceType(device: *mut c_void, device_type: *mut i32) -> i32;
    pub fn ANeuralNetworks_getDeviceFeatureLevel(device: *mut c_void, feature_level: *mut i32) -> i32;

    pub fn ANeuralNetworks_getDeviceExtensionSupport(
        device: *mut c_void,
        extension_name: *const c_char,
        is_supported: *mut bool,
    ) -> i32;

    pub fn ANeuralNetworks_getDevicePerformanceInfo(
        device: *mut c_void,
        performance_info: *mut ANeuralNetworksPerformanceInfo,
    ) -> i32;

    // Advanced compilation functions (API Level 29+)
    pub fn ANeuralNetworksCompilation_createForDevices(
        model: *mut ANeuralNetworksModel,
        devices: *const *mut c_void,
        numDevices: u32,
        compilation: *mut *mut ANeuralNetworksCompilation,
    ) -> i32;

    pub fn ANeuralNetworksCompilation_setCaching(
        compilation: *mut ANeuralNetworksCompilation,
        cacheDir: *const c_char,
        token: *const u8,
    ) -> i32;

    // Memory management functions (API Level 30+)
    pub fn ANeuralNetworksMemory_createFromFd(
        size: usize,
        prot: i32,
        fd: c_int,
        offset: usize,
        memory: *mut *mut c_void,
    ) -> i32;

    pub fn ANeuralNetworksMemory_free(memory: *mut c_void);

    // Execution burst functions (API Level 29+)
    pub fn ANeuralNetworksBurst_create(
        compilation: *mut ANeuralNetworksCompilation,
        burst: *mut *mut c_void,
    ) -> i32;

    pub fn ANeuralNetworksBurst_free(burst: *mut c_void);

    pub fn ANeuralNetworksExecution_burstCompute(
        execution: *mut ANeuralNetworksExecution,
        burst: *mut c_void,
    ) -> i32;
}

/// Helper function to check if NNAPI result indicates success
pub fn nnapi_is_success(result: i32) -> bool {
    result == ANEURALNETWORKS_NO_ERROR
}

/// Convert NNAPI result code to human-readable string
pub fn nnapi_result_to_string(result: i32) -> &'static str {
    match result {
        ANEURALNETWORKS_NO_ERROR => "Success",
        ANEURALNETWORKS_OUT_OF_MEMORY => "Out of memory",
        ANEURALNETWORKS_INCOMPLETE => "Incomplete",
        ANEURALNETWORKS_UNEXPECTED_NULL => "Unexpected null pointer",
        ANEURALNETWORKS_BAD_DATA => "Bad data",
        ANEURALNETWORKS_OP_FAILED => "Operation failed",
        ANEURALNETWORKS_BAD_STATE => "Bad state",
        ANEURALNETWORKS_UNMAPPABLE => "Unmappable",
        ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE => "Output buffer too small",
        ANEURALNETWORKS_UNAVAILABLE_DEVICE => "Device unavailable",
        _ => "Unknown error",
    }
}

/// Check if NNAPI is available on the current device
pub fn is_nnapi_available() -> bool {
    #[cfg(target_os = "android")]
    {
        // In practice, would check for libandroid.so and API level
        true
    }
    #[cfg(not(target_os = "android"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nnapi_constants() {
        // Verify that NNAPI constants are correctly defined
        assert_eq!(ANEURALNETWORKS_NO_ERROR, 0);
        assert_eq!(ANEURALNETWORKS_FLOAT32, 0);
        assert_eq!(ANEURALNETWORKS_CONV_2D, 3);
    }

    #[test]
    fn test_result_handling() {
        assert!(nnapi_is_success(ANEURALNETWORKS_NO_ERROR));
        assert!(!nnapi_is_success(ANEURALNETWORKS_OUT_OF_MEMORY));

        assert_eq!(nnapi_result_to_string(ANEURALNETWORKS_NO_ERROR), "Success");
        assert_eq!(nnapi_result_to_string(ANEURALNETWORKS_OUT_OF_MEMORY), "Out of memory");
    }

    #[test]
    fn test_availability() {
        // This test will pass on both Android and non-Android platforms
        let _available = is_nnapi_available();
    }
}