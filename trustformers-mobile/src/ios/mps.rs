//! Metal Performance Shaders (MPS) Bindings for iOS
//!
//! This module provides Metal Performance Shaders API bindings for high-performance
//! neural network computation on iOS devices using Apple's MPS framework.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_uint, c_void};
use std::ptr;

// Import Metal types from the metal module
use super::metal::{MTLCommandQueue, MTLDevice};

// Metal Performance Shaders (MPS) API types
#[cfg(target_os = "ios")]
#[repr(C)]
pub struct MPSGraph;

#[cfg(target_os = "ios")]
#[repr(C)]
pub struct MPSGraphExecutable;

#[cfg(target_os = "ios")]
#[repr(C)]
pub struct MPSGraphTensor;

#[cfg(target_os = "ios")]
#[repr(C)]
pub struct MPSGraphTensorData;

#[cfg(target_os = "ios")]
#[repr(C)]
pub struct MPSGraphConvolution2DOpDescriptor;

#[cfg(target_os = "ios")]
#[repr(C)]
pub struct MPSGraphDepthwiseConvolution2DOpDescriptor;

#[cfg(target_os = "ios")]
#[repr(C)]
pub struct MPSGraphMatrixMultiplicationDescriptor;

#[cfg(target_os = "ios")]
#[repr(C)]
pub struct MPSGraphPooling2DOpDescriptor;

#[cfg(target_os = "ios")]
#[repr(C)]
pub struct MPSGraphDevice;

#[cfg(target_os = "ios")]
#[repr(C)]
pub struct MPSGraphExecutionDescriptor;

#[cfg(target_os = "ios")]
#[repr(C)]
pub struct MPSShape([usize; 4]);

#[cfg(target_os = "ios")]
#[repr(C)]
pub struct MPSDataType(u32);

// MPS data type constants
#[cfg(target_os = "ios")]
pub const MPS_DATA_TYPE_FLOAT32: u32 = 0x10000000;
#[cfg(target_os = "ios")]
pub const MPS_DATA_TYPE_FLOAT16: u32 = 0x10000001;
#[cfg(target_os = "ios")]
pub const MPS_DATA_TYPE_INT32: u32 = 0x10000002;
#[cfg(target_os = "ios")]
pub const MPS_DATA_TYPE_INT8: u32 = 0x10000003;
#[cfg(target_os = "ios")]
pub const MPS_DATA_TYPE_UINT8: u32 = 0x10000004;
#[cfg(target_os = "ios")]
pub const MPS_DATA_TYPE_BOOL: u32 = 0x10000005;

// Metal Performance Shaders C API bindings
#[cfg(target_os = "ios")]
extern "C" {
    // Graph creation and management
    pub fn MPSGraph_create() -> *mut MPSGraph;
    pub fn MPSGraph_release(graph: *mut MPSGraph);

    // Tensor data creation
    pub fn MPSGraphTensorData_create(
        device: *mut MTLDevice,
        data: *const c_void,
        shape: *const usize,
        shape_len: usize,
        data_type: u32,
    ) -> *mut MPSGraphTensorData;
    pub fn MPSGraphTensorData_release(tensor_data: *mut MPSGraphTensorData);
    pub fn MPSGraphTensorData_mpsndarray(tensor_data: *mut MPSGraphTensorData) -> *mut c_void;

    // Tensor operations
    pub fn MPSGraph_placeholderWithShape(
        graph: *mut MPSGraph,
        shape: *const usize,
        shape_len: usize,
        data_type: u32,
        name: *const c_char,
    ) -> *mut MPSGraphTensor;

    pub fn MPSGraph_constantWithScalar(
        graph: *mut MPSGraph,
        scalar: f32,
        data_type: u32,
    ) -> *mut MPSGraphTensor;

    pub fn MPSGraph_constantWithData(
        graph: *mut MPSGraph,
        data: *mut MPSGraphTensorData,
        shape: *const usize,
        shape_len: usize,
        data_type: u32,
    ) -> *mut MPSGraphTensor;

    pub fn MPSGraph_variableWithData(
        graph: *mut MPSGraph,
        data: *mut MPSGraphTensorData,
        shape: *const usize,
        shape_len: usize,
        data_type: u32,
        name: *const c_char,
    ) -> *mut MPSGraphTensor;

    // Neural network operations
    pub fn MPSGraph_convolution2DWithSourceTensor(
        graph: *mut MPSGraph,
        source: *mut MPSGraphTensor,
        weights: *mut MPSGraphTensor,
        descriptor: *mut MPSGraphConvolution2DOpDescriptor,
        name: *const c_char,
    ) -> *mut MPSGraphTensor;

    pub fn MPSGraph_depthwiseConvolution2DWithSourceTensor(
        graph: *mut MPSGraph,
        source: *mut MPSGraphTensor,
        weights: *mut MPSGraphTensor,
        descriptor: *mut MPSGraphDepthwiseConvolution2DOpDescriptor,
        name: *const c_char,
    ) -> *mut MPSGraphTensor;

    pub fn MPSGraph_matrixMultiplicationWithPrimaryTensor(
        graph: *mut MPSGraph,
        primary: *mut MPSGraphTensor,
        secondary: *mut MPSGraphTensor,
        name: *const c_char,
    ) -> *mut MPSGraphTensor;

    pub fn MPSGraph_batchMatMulWithPrimaryTensor(
        graph: *mut MPSGraph,
        primary: *mut MPSGraphTensor,
        secondary: *mut MPSGraphTensor,
        name: *const c_char,
    ) -> *mut MPSGraphTensor;

    // Activation functions
    pub fn MPSGraph_reLUWithTensor(
        graph: *mut MPSGraph,
        tensor: *mut MPSGraphTensor,
        name: *const c_char,
    ) -> *mut MPSGraphTensor;

    pub fn MPSGraph_sigmoidWithTensor(
        graph: *mut MPSGraph,
        tensor: *mut MPSGraphTensor,
        name: *const c_char,
    ) -> *mut MPSGraphTensor;

    pub fn MPSGraph_tanhWithTensor(
        graph: *mut MPSGraph,
        tensor: *mut MPSGraphTensor,
        name: *const c_char,
    ) -> *mut MPSGraphTensor;

    pub fn MPSGraph_softMaxWithTensor(
        graph: *mut MPSGraph,
        tensor: *mut MPSGraphTensor,
        axis: c_int,
        name: *const c_char,
    ) -> *mut MPSGraphTensor;

    pub fn MPSGraph_leakyReLUWithTensor(
        graph: *mut MPSGraph,
        tensor: *mut MPSGraphTensor,
        alpha: f32,
        name: *const c_char,
    ) -> *mut MPSGraphTensor;

    // Element-wise operations
    pub fn MPSGraph_additionWithPrimaryTensor(
        graph: *mut MPSGraph,
        primary: *mut MPSGraphTensor,
        secondary: *mut MPSGraphTensor,
        name: *const c_char,
    ) -> *mut MPSGraphTensor;

    pub fn MPSGraph_subtractionWithPrimaryTensor(
        graph: *mut MPSGraph,
        primary: *mut MPSGraphTensor,
        secondary: *mut MPSGraphTensor,
        name: *const c_char,
    ) -> *mut MPSGraphTensor;

    pub fn MPSGraph_multiplicationWithPrimaryTensor(
        graph: *mut MPSGraph,
        primary: *mut MPSGraphTensor,
        secondary: *mut MPSGraphTensor,
        name: *const c_char,
    ) -> *mut MPSGraphTensor;

    pub fn MPSGraph_divisionWithPrimaryTensor(
        graph: *mut MPSGraph,
        primary: *mut MPSGraphTensor,
        secondary: *mut MPSGraphTensor,
        name: *const c_char,
    ) -> *mut MPSGraphTensor;

    // Pooling operations
    pub fn MPSGraph_maxPooling2DWithSourceTensor(
        graph: *mut MPSGraph,
        source: *mut MPSGraphTensor,
        descriptor: *mut MPSGraphPooling2DOpDescriptor,
        name: *const c_char,
    ) -> *mut MPSGraphTensor;

    pub fn MPSGraph_averagePooling2DWithSourceTensor(
        graph: *mut MPSGraph,
        source: *mut MPSGraphTensor,
        descriptor: *mut MPSGraphPooling2DOpDescriptor,
        name: *const c_char,
    ) -> *mut MPSGraphTensor;

    pub fn MPSGraph_globalAveragePooling2DWithSourceTensor(
        graph: *mut MPSGraph,
        source: *mut MPSGraphTensor,
        name: *const c_char,
    ) -> *mut MPSGraphTensor;

    // Normalization operations
    pub fn MPSGraph_batchNormalizationWithSourceTensor(
        graph: *mut MPSGraph,
        source: *mut MPSGraphTensor,
        mean: *mut MPSGraphTensor,
        variance: *mut MPSGraphTensor,
        gamma: *mut MPSGraphTensor,
        beta: *mut MPSGraphTensor,
        epsilon: f32,
        name: *const c_char,
    ) -> *mut MPSGraphTensor;

    pub fn MPSGraph_layerNormalizationWithSourceTensor(
        graph: *mut MPSGraph,
        source: *mut MPSGraphTensor,
        normalizedShape: *const usize,
        normalizedShape_len: usize,
        gamma: *mut MPSGraphTensor,
        beta: *mut MPSGraphTensor,
        epsilon: f32,
        name: *const c_char,
    ) -> *mut MPSGraphTensor;

    // Reduction operations
    pub fn MPSGraph_reductionSumWithTensor(
        graph: *mut MPSGraph,
        tensor: *mut MPSGraphTensor,
        axes: *const c_int,
        axes_count: usize,
        name: *const c_char,
    ) -> *mut MPSGraphTensor;

    pub fn MPSGraph_reductionMeanWithTensor(
        graph: *mut MPSGraph,
        tensor: *mut MPSGraphTensor,
        axes: *const c_int,
        axes_count: usize,
        name: *const c_char,
    ) -> *mut MPSGraphTensor;

    pub fn MPSGraph_reductionMaxWithTensor(
        graph: *mut MPSGraph,
        tensor: *mut MPSGraphTensor,
        axes: *const c_int,
        axes_count: usize,
        name: *const c_char,
    ) -> *mut MPSGraphTensor;

    // Tensor manipulation operations
    pub fn MPSGraph_reshapeTensor(
        graph: *mut MPSGraph,
        tensor: *mut MPSGraphTensor,
        shape: *const usize,
        shape_len: usize,
        name: *const c_char,
    ) -> *mut MPSGraphTensor;

    pub fn MPSGraph_transposeTensor(
        graph: *mut MPSGraph,
        tensor: *mut MPSGraphTensor,
        dimension: usize,
        dimensionB: usize,
        name: *const c_char,
    ) -> *mut MPSGraphTensor;

    pub fn MPSGraph_concatTensors(
        graph: *mut MPSGraph,
        tensors: *mut *mut MPSGraphTensor,
        tensors_count: usize,
        dimension: c_int,
        name: *const c_char,
    ) -> *mut MPSGraphTensor;

    pub fn MPSGraph_sliceTensor(
        graph: *mut MPSGraph,
        tensor: *mut MPSGraphTensor,
        starts: *const c_int,
        ends: *const c_int,
        strides: *const c_int,
        dimension_count: usize,
        name: *const c_char,
    ) -> *mut MPSGraphTensor;

    // Graph compilation and execution
    pub fn MPSGraph_compileWithDevice(
        graph: *mut MPSGraph,
        device: *mut MTLDevice,
        feeds: *mut *mut MPSGraphTensor,
        feeds_count: usize,
        targetTensors: *mut *mut MPSGraphTensor,
        targetTensors_count: usize,
        targetOperations: *mut c_void,
        compilationDescriptor: *mut c_void,
    ) -> *mut MPSGraphExecutable;

    pub fn MPSGraphExecutable_runWithMTLCommandQueue(
        executable: *mut MPSGraphExecutable,
        command_queue: *mut MTLCommandQueue,
        feeds: *mut *mut MPSGraphTensorData,
        feeds_count: usize,
        targetTensors: *mut *mut MPSGraphTensorData,
        targetTensors_count: usize,
        executionDescriptor: *mut MPSGraphExecutionDescriptor,
    ) -> *mut c_void;

    pub fn MPSGraphExecutable_runAsyncWithMTLCommandQueue(
        executable: *mut MPSGraphExecutable,
        command_queue: *mut MTLCommandQueue,
        feeds: *mut *mut MPSGraphTensorData,
        feeds_count: usize,
        targetTensors: *mut *mut MPSGraphTensorData,
        targetTensors_count: usize,
        executionDescriptor: *mut MPSGraphExecutionDescriptor,
        resultsDictionary: *mut c_void,
    );

    pub fn MPSGraphExecutable_release(executable: *mut MPSGraphExecutable);

    // Descriptor creation and configuration
    pub fn MPSGraphConvolution2DOpDescriptor_create() -> *mut MPSGraphConvolution2DOpDescriptor;
    pub fn MPSGraphConvolution2DOpDescriptor_release(
        descriptor: *mut MPSGraphConvolution2DOpDescriptor,
    );
    pub fn MPSGraphConvolution2DOpDescriptor_setStrideInX(
        descriptor: *mut MPSGraphConvolution2DOpDescriptor,
        stride: usize,
    );
    pub fn MPSGraphConvolution2DOpDescriptor_setStrideInY(
        descriptor: *mut MPSGraphConvolution2DOpDescriptor,
        stride: usize,
    );
    pub fn MPSGraphConvolution2DOpDescriptor_setPaddingLeft(
        descriptor: *mut MPSGraphConvolution2DOpDescriptor,
        padding: usize,
    );
    pub fn MPSGraphConvolution2DOpDescriptor_setPaddingRight(
        descriptor: *mut MPSGraphConvolution2DOpDescriptor,
        padding: usize,
    );
    pub fn MPSGraphConvolution2DOpDescriptor_setPaddingTop(
        descriptor: *mut MPSGraphConvolution2DOpDescriptor,
        padding: usize,
    );
    pub fn MPSGraphConvolution2DOpDescriptor_setPaddingBottom(
        descriptor: *mut MPSGraphConvolution2DOpDescriptor,
        padding: usize,
    );
    pub fn MPSGraphConvolution2DOpDescriptor_setDilationRateInX(
        descriptor: *mut MPSGraphConvolution2DOpDescriptor,
        dilation: usize,
    );
    pub fn MPSGraphConvolution2DOpDescriptor_setDilationRateInY(
        descriptor: *mut MPSGraphConvolution2DOpDescriptor,
        dilation: usize,
    );
    pub fn MPSGraphConvolution2DOpDescriptor_setGroups(
        descriptor: *mut MPSGraphConvolution2DOpDescriptor,
        groups: usize,
    );

    pub fn MPSGraphPooling2DOpDescriptor_create() -> *mut MPSGraphPooling2DOpDescriptor;
    pub fn MPSGraphPooling2DOpDescriptor_release(descriptor: *mut MPSGraphPooling2DOpDescriptor);
    pub fn MPSGraphPooling2DOpDescriptor_setKernelWidth(
        descriptor: *mut MPSGraphPooling2DOpDescriptor,
        width: usize,
    );
    pub fn MPSGraphPooling2DOpDescriptor_setKernelHeight(
        descriptor: *mut MPSGraphPooling2DOpDescriptor,
        height: usize,
    );
    pub fn MPSGraphPooling2DOpDescriptor_setStrideInX(
        descriptor: *mut MPSGraphPooling2DOpDescriptor,
        stride: usize,
    );
    pub fn MPSGraphPooling2DOpDescriptor_setStrideInY(
        descriptor: *mut MPSGraphPooling2DOpDescriptor,
        stride: usize,
    );
    pub fn MPSGraphPooling2DOpDescriptor_setPaddingLeft(
        descriptor: *mut MPSGraphPooling2DOpDescriptor,
        padding: usize,
    );
    pub fn MPSGraphPooling2DOpDescriptor_setPaddingRight(
        descriptor: *mut MPSGraphPooling2DOpDescriptor,
        padding: usize,
    );
    pub fn MPSGraphPooling2DOpDescriptor_setPaddingTop(
        descriptor: *mut MPSGraphPooling2DOpDescriptor,
        padding: usize,
    );
    pub fn MPSGraphPooling2DOpDescriptor_setPaddingBottom(
        descriptor: *mut MPSGraphPooling2DOpDescriptor,
        padding: usize,
    );

    pub fn MPSGraphExecutionDescriptor_create() -> *mut MPSGraphExecutionDescriptor;
    pub fn MPSGraphExecutionDescriptor_release(descriptor: *mut MPSGraphExecutionDescriptor);
    pub fn MPSGraphExecutionDescriptor_setWaitUntilCompleted(
        descriptor: *mut MPSGraphExecutionDescriptor,
        wait: bool,
    );
}

// High-level MPS wrapper types
#[cfg(target_os = "ios")]
pub struct MPSComputeGraph {
    graph: *mut MPSGraph,
    tensors: std::collections::HashMap<String, *mut MPSGraphTensor>,
}

#[cfg(target_os = "ios")]
pub struct MPSTensorData {
    tensor_data: *mut MPSGraphTensorData,
    shape: Vec<usize>,
    data_type: MPSDataType,
}

#[cfg(target_os = "ios")]
pub struct MPSExecutable {
    executable: *mut MPSGraphExecutable,
    input_tensors: Vec<*mut MPSGraphTensor>,
    output_tensors: Vec<*mut MPSGraphTensor>,
}

#[cfg(target_os = "ios")]
pub struct MPSConvolution2DDescriptor {
    descriptor: *mut MPSGraphConvolution2DOpDescriptor,
}

#[cfg(target_os = "ios")]
pub struct MPSPooling2DDescriptor {
    descriptor: *mut MPSGraphPooling2DOpDescriptor,
}

#[cfg(target_os = "ios")]
impl MPSComputeGraph {
    /// Create new MPS compute graph
    pub fn new() -> Result<Self, String> {
        unsafe {
            let graph = MPSGraph_create();
            if graph.is_null() {
                return Err("Failed to create MPS graph".to_string());
            }

            Ok(Self {
                graph,
                tensors: std::collections::HashMap::new(),
            })
        }
    }

    /// Create placeholder tensor
    pub fn placeholder(
        &mut self,
        shape: &[usize],
        data_type: MPSDataType,
        name: &str,
    ) -> Result<String, String> {
        unsafe {
            let name_cstr = CString::new(name).map_err(|e| format!("Invalid name: {}", e))?;
            let tensor = MPSGraph_placeholderWithShape(
                self.graph,
                shape.as_ptr(),
                shape.len(),
                data_type.0,
                name_cstr.as_ptr(),
            );

            if tensor.is_null() {
                return Err("Failed to create placeholder tensor".to_string());
            }

            let tensor_name = format!("{}_{}", name, self.tensors.len());
            self.tensors.insert(tensor_name.clone(), tensor);
            Ok(tensor_name)
        }
    }

    /// Create constant tensor from scalar
    pub fn constant_scalar(
        &mut self,
        value: f32,
        data_type: MPSDataType,
        name: &str,
    ) -> Result<String, String> {
        unsafe {
            let tensor = MPSGraph_constantWithScalar(self.graph, value, data_type.0);

            if tensor.is_null() {
                return Err("Failed to create constant tensor".to_string());
            }

            let tensor_name = format!("{}_{}", name, self.tensors.len());
            self.tensors.insert(tensor_name.clone(), tensor);
            Ok(tensor_name)
        }
    }

    /// Add convolution 2D operation
    pub fn convolution_2d(
        &mut self,
        source: &str,
        weights: &str,
        descriptor: &MPSConvolution2DDescriptor,
        name: &str,
    ) -> Result<String, String> {
        unsafe {
            let source_tensor = self
                .tensors
                .get(source)
                .ok_or_else(|| format!("Source tensor '{}' not found", source))?;
            let weights_tensor = self
                .tensors
                .get(weights)
                .ok_or_else(|| format!("Weights tensor '{}' not found", weights))?;

            let name_cstr = CString::new(name).map_err(|e| format!("Invalid name: {}", e))?;
            let result_tensor = MPSGraph_convolution2DWithSourceTensor(
                self.graph,
                *source_tensor,
                *weights_tensor,
                descriptor.descriptor,
                name_cstr.as_ptr(),
            );

            if result_tensor.is_null() {
                return Err("Failed to create convolution operation".to_string());
            }

            let tensor_name = format!("{}_{}", name, self.tensors.len());
            self.tensors.insert(tensor_name.clone(), result_tensor);
            Ok(tensor_name)
        }
    }

    /// Add matrix multiplication operation
    pub fn matrix_multiplication(
        &mut self,
        primary: &str,
        secondary: &str,
        name: &str,
    ) -> Result<String, String> {
        unsafe {
            let primary_tensor = self
                .tensors
                .get(primary)
                .ok_or_else(|| format!("Primary tensor '{}' not found", primary))?;
            let secondary_tensor = self
                .tensors
                .get(secondary)
                .ok_or_else(|| format!("Secondary tensor '{}' not found", secondary))?;

            let name_cstr = CString::new(name).map_err(|e| format!("Invalid name: {}", e))?;
            let result_tensor = MPSGraph_matrixMultiplicationWithPrimaryTensor(
                self.graph,
                *primary_tensor,
                *secondary_tensor,
                name_cstr.as_ptr(),
            );

            if result_tensor.is_null() {
                return Err("Failed to create matrix multiplication operation".to_string());
            }

            let tensor_name = format!("{}_{}", name, self.tensors.len());
            self.tensors.insert(tensor_name.clone(), result_tensor);
            Ok(tensor_name)
        }
    }

    /// Add ReLU activation
    pub fn relu(&mut self, input: &str, name: &str) -> Result<String, String> {
        unsafe {
            let input_tensor = self
                .tensors
                .get(input)
                .ok_or_else(|| format!("Input tensor '{}' not found", input))?;

            let name_cstr = CString::new(name).map_err(|e| format!("Invalid name: {}", e))?;
            let result_tensor =
                MPSGraph_reLUWithTensor(self.graph, *input_tensor, name_cstr.as_ptr());

            if result_tensor.is_null() {
                return Err("Failed to create ReLU operation".to_string());
            }

            let tensor_name = format!("{}_{}", name, self.tensors.len());
            self.tensors.insert(tensor_name.clone(), result_tensor);
            Ok(tensor_name)
        }
    }

    /// Add sigmoid activation
    pub fn sigmoid(&mut self, input: &str, name: &str) -> Result<String, String> {
        unsafe {
            let input_tensor = self
                .tensors
                .get(input)
                .ok_or_else(|| format!("Input tensor '{}' not found", input))?;

            let name_cstr = CString::new(name).map_err(|e| format!("Invalid name: {}", e))?;
            let result_tensor =
                MPSGraph_sigmoidWithTensor(self.graph, *input_tensor, name_cstr.as_ptr());

            if result_tensor.is_null() {
                return Err("Failed to create sigmoid operation".to_string());
            }

            let tensor_name = format!("{}_{}", name, self.tensors.len());
            self.tensors.insert(tensor_name.clone(), result_tensor);
            Ok(tensor_name)
        }
    }

    /// Add element-wise addition
    pub fn addition(
        &mut self,
        primary: &str,
        secondary: &str,
        name: &str,
    ) -> Result<String, String> {
        unsafe {
            let primary_tensor = self
                .tensors
                .get(primary)
                .ok_or_else(|| format!("Primary tensor '{}' not found", primary))?;
            let secondary_tensor = self
                .tensors
                .get(secondary)
                .ok_or_else(|| format!("Secondary tensor '{}' not found", secondary))?;

            let name_cstr = CString::new(name).map_err(|e| format!("Invalid name: {}", e))?;
            let result_tensor = MPSGraph_additionWithPrimaryTensor(
                self.graph,
                *primary_tensor,
                *secondary_tensor,
                name_cstr.as_ptr(),
            );

            if result_tensor.is_null() {
                return Err("Failed to create addition operation".to_string());
            }

            let tensor_name = format!("{}_{}", name, self.tensors.len());
            self.tensors.insert(tensor_name.clone(), result_tensor);
            Ok(tensor_name)
        }
    }

    /// Add max pooling 2D operation
    pub fn max_pooling_2d(
        &mut self,
        source: &str,
        descriptor: &MPSPooling2DDescriptor,
        name: &str,
    ) -> Result<String, String> {
        unsafe {
            let source_tensor = self
                .tensors
                .get(source)
                .ok_or_else(|| format!("Source tensor '{}' not found", source))?;

            let name_cstr = CString::new(name).map_err(|e| format!("Invalid name: {}", e))?;
            let result_tensor = MPSGraph_maxPooling2DWithSourceTensor(
                self.graph,
                *source_tensor,
                descriptor.descriptor,
                name_cstr.as_ptr(),
            );

            if result_tensor.is_null() {
                return Err("Failed to create max pooling operation".to_string());
            }

            let tensor_name = format!("{}_{}", name, self.tensors.len());
            self.tensors.insert(tensor_name.clone(), result_tensor);
            Ok(tensor_name)
        }
    }

    /// Compile graph for execution
    pub fn compile(
        &self,
        device: *mut MTLDevice,
        input_names: &[String],
        output_names: &[String],
    ) -> Result<MPSExecutable, String> {
        unsafe {
            let mut input_tensors: Vec<*mut MPSGraphTensor> = Vec::new();
            let mut output_tensors: Vec<*mut MPSGraphTensor> = Vec::new();

            for name in input_names {
                let tensor = self
                    .tensors
                    .get(name)
                    .ok_or_else(|| format!("Input tensor '{}' not found", name))?;
                input_tensors.push(*tensor);
            }

            for name in output_names {
                let tensor = self
                    .tensors
                    .get(name)
                    .ok_or_else(|| format!("Output tensor '{}' not found", name))?;
                output_tensors.push(*tensor);
            }

            let executable = MPSGraph_compileWithDevice(
                self.graph,
                device,
                input_tensors.as_mut_ptr(),
                input_tensors.len(),
                output_tensors.as_mut_ptr(),
                output_tensors.len(),
                ptr::null_mut(),
                ptr::null_mut(),
            );

            if executable.is_null() {
                return Err("Failed to compile MPS graph".to_string());
            }

            Ok(MPSExecutable {
                executable,
                input_tensors,
                output_tensors,
            })
        }
    }
}

#[cfg(target_os = "ios")]
impl Drop for MPSComputeGraph {
    fn drop(&mut self) {
        unsafe {
            if !self.graph.is_null() {
                MPSGraph_release(self.graph);
            }
        }
    }
}

#[cfg(target_os = "ios")]
impl MPSTensorData {
    /// Create tensor data from raw data
    pub fn create(
        device: *mut MTLDevice,
        data: &[u8],
        shape: &[usize],
        data_type: MPSDataType,
    ) -> Result<Self, String> {
        unsafe {
            let tensor_data = MPSGraphTensorData_create(
                device,
                data.as_ptr() as *const c_void,
                shape.as_ptr(),
                shape.len(),
                data_type.0,
            );

            if tensor_data.is_null() {
                return Err("Failed to create MPS tensor data".to_string());
            }

            Ok(Self {
                tensor_data,
                shape: shape.to_vec(),
                data_type,
            })
        }
    }

    /// Get shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get data type
    pub fn data_type(&self) -> MPSDataType {
        self.data_type
    }
}

#[cfg(target_os = "ios")]
impl Drop for MPSTensorData {
    fn drop(&mut self) {
        unsafe {
            if !self.tensor_data.is_null() {
                MPSGraphTensorData_release(self.tensor_data);
            }
        }
    }
}

#[cfg(target_os = "ios")]
impl MPSExecutable {
    /// Execute graph with command queue
    pub fn run_with_command_queue(
        &self,
        command_queue: *mut MTLCommandQueue,
        inputs: &[&MPSTensorData],
        outputs: &mut [&mut MPSTensorData],
    ) -> Result<(), String> {
        unsafe {
            let mut input_data: Vec<*mut MPSGraphTensorData> =
                inputs.iter().map(|td| td.tensor_data).collect();
            let mut output_data: Vec<*mut MPSGraphTensorData> =
                outputs.iter().map(|td| td.tensor_data).collect();

            let execution_desc = MPSGraphExecutionDescriptor_create();
            if execution_desc.is_null() {
                return Err("Failed to create execution descriptor".to_string());
            }

            MPSGraphExecutionDescriptor_setWaitUntilCompleted(execution_desc, true);

            let _result = MPSGraphExecutable_runWithMTLCommandQueue(
                self.executable,
                command_queue,
                input_data.as_mut_ptr(),
                input_data.len(),
                output_data.as_mut_ptr(),
                output_data.len(),
                execution_desc,
            );

            MPSGraphExecutionDescriptor_release(execution_desc);
            Ok(())
        }
    }
}

#[cfg(target_os = "ios")]
impl Drop for MPSExecutable {
    fn drop(&mut self) {
        unsafe {
            if !self.executable.is_null() {
                MPSGraphExecutable_release(self.executable);
            }
        }
    }
}

#[cfg(target_os = "ios")]
impl MPSConvolution2DDescriptor {
    /// Create new convolution descriptor
    pub fn new() -> Result<Self, String> {
        unsafe {
            let descriptor = MPSGraphConvolution2DOpDescriptor_create();
            if descriptor.is_null() {
                return Err("Failed to create convolution descriptor".to_string());
            }

            Ok(Self { descriptor })
        }
    }

    /// Set stride
    pub fn set_stride(&self, stride_x: usize, stride_y: usize) {
        unsafe {
            MPSGraphConvolution2DOpDescriptor_setStrideInX(self.descriptor, stride_x);
            MPSGraphConvolution2DOpDescriptor_setStrideInY(self.descriptor, stride_y);
        }
    }

    /// Set padding
    pub fn set_padding(&self, left: usize, right: usize, top: usize, bottom: usize) {
        unsafe {
            MPSGraphConvolution2DOpDescriptor_setPaddingLeft(self.descriptor, left);
            MPSGraphConvolution2DOpDescriptor_setPaddingRight(self.descriptor, right);
            MPSGraphConvolution2DOpDescriptor_setPaddingTop(self.descriptor, top);
            MPSGraphConvolution2DOpDescriptor_setPaddingBottom(self.descriptor, bottom);
        }
    }

    /// Set dilation
    pub fn set_dilation(&self, dilation_x: usize, dilation_y: usize) {
        unsafe {
            MPSGraphConvolution2DOpDescriptor_setDilationRateInX(self.descriptor, dilation_x);
            MPSGraphConvolution2DOpDescriptor_setDilationRateInY(self.descriptor, dilation_y);
        }
    }

    /// Set groups
    pub fn set_groups(&self, groups: usize) {
        unsafe {
            MPSGraphConvolution2DOpDescriptor_setGroups(self.descriptor, groups);
        }
    }
}

#[cfg(target_os = "ios")]
impl Default for MPSConvolution2DDescriptor {
    fn default() -> Self {
        Self::new().expect("Failed to create default convolution descriptor")
    }
}

#[cfg(target_os = "ios")]
impl Drop for MPSConvolution2DDescriptor {
    fn drop(&mut self) {
        unsafe {
            if !self.descriptor.is_null() {
                MPSGraphConvolution2DOpDescriptor_release(self.descriptor);
            }
        }
    }
}

#[cfg(target_os = "ios")]
impl MPSPooling2DDescriptor {
    /// Create new pooling descriptor
    pub fn new() -> Result<Self, String> {
        unsafe {
            let descriptor = MPSGraphPooling2DOpDescriptor_create();
            if descriptor.is_null() {
                return Err("Failed to create pooling descriptor".to_string());
            }

            Ok(Self { descriptor })
        }
    }

    /// Set kernel size
    pub fn set_kernel_size(&self, width: usize, height: usize) {
        unsafe {
            MPSGraphPooling2DOpDescriptor_setKernelWidth(self.descriptor, width);
            MPSGraphPooling2DOpDescriptor_setKernelHeight(self.descriptor, height);
        }
    }

    /// Set stride
    pub fn set_stride(&self, stride_x: usize, stride_y: usize) {
        unsafe {
            MPSGraphPooling2DOpDescriptor_setStrideInX(self.descriptor, stride_x);
            MPSGraphPooling2DOpDescriptor_setStrideInY(self.descriptor, stride_y);
        }
    }

    /// Set padding
    pub fn set_padding(&self, left: usize, right: usize, top: usize, bottom: usize) {
        unsafe {
            MPSGraphPooling2DOpDescriptor_setPaddingLeft(self.descriptor, left);
            MPSGraphPooling2DOpDescriptor_setPaddingRight(self.descriptor, right);
            MPSGraphPooling2DOpDescriptor_setPaddingTop(self.descriptor, top);
            MPSGraphPooling2DOpDescriptor_setPaddingBottom(self.descriptor, bottom);
        }
    }
}

#[cfg(target_os = "ios")]
impl Default for MPSPooling2DDescriptor {
    fn default() -> Self {
        Self::new().expect("Failed to create default pooling descriptor")
    }
}

#[cfg(target_os = "ios")]
impl Drop for MPSPooling2DDescriptor {
    fn drop(&mut self) {
        unsafe {
            if !self.descriptor.is_null() {
                MPSGraphPooling2DOpDescriptor_release(self.descriptor);
            }
        }
    }
}

// Data type utilities
impl MPSDataType {
    /// Create float32 data type
    pub fn float32() -> Self {
        Self(MPS_DATA_TYPE_FLOAT32)
    }

    /// Create float16 data type
    pub fn float16() -> Self {
        Self(MPS_DATA_TYPE_FLOAT16)
    }

    /// Create int32 data type
    pub fn int32() -> Self {
        Self(MPS_DATA_TYPE_INT32)
    }

    /// Create int8 data type
    pub fn int8() -> Self {
        Self(MPS_DATA_TYPE_INT8)
    }

    /// Create uint8 data type
    pub fn uint8() -> Self {
        Self(MPS_DATA_TYPE_UINT8)
    }

    /// Create bool data type
    pub fn bool() -> Self {
        Self(MPS_DATA_TYPE_BOOL)
    }

    /// Get element size in bytes
    pub fn element_size(&self) -> usize {
        match self.0 {
            MPS_DATA_TYPE_FLOAT32 => 4,
            MPS_DATA_TYPE_FLOAT16 => 2,
            MPS_DATA_TYPE_INT32 => 4,
            MPS_DATA_TYPE_INT8 => 1,
            MPS_DATA_TYPE_UINT8 => 1,
            MPS_DATA_TYPE_BOOL => 1,
            _ => 4, // Default to 4 bytes
        }
    }
}

// Non-iOS stub implementations
#[cfg(not(target_os = "ios"))]
pub struct MPSComputeGraph;

#[cfg(not(target_os = "ios"))]
pub struct MPSTensorData;

#[cfg(not(target_os = "ios"))]
pub struct MPSExecutable;

#[cfg(not(target_os = "ios"))]
pub struct MPSConvolution2DDescriptor;

#[cfg(not(target_os = "ios"))]
pub struct MPSPooling2DDescriptor;

#[cfg(not(target_os = "ios"))]
#[derive(Clone, Copy)]
pub struct MPSDataType(u32);

#[cfg(not(target_os = "ios"))]
impl MPSComputeGraph {
    pub fn new() -> Result<Self, String> {
        Err("MPS not available on this platform".to_string())
    }
}

#[cfg(not(target_os = "ios"))]
impl MPSDataType {
    pub fn float32() -> Self {
        Self(0)
    }
    pub fn float16() -> Self {
        Self(0)
    }
    pub fn int32() -> Self {
        Self(0)
    }
    pub fn int8() -> Self {
        Self(0)
    }
    pub fn uint8() -> Self {
        Self(0)
    }
    pub fn bool() -> Self {
        Self(0)
    }
    pub fn element_size(&self) -> usize {
        0
    }
}
