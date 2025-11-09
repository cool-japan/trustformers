//! NNAPI Model Management and Construction
//!
//! This module handles NNAPI model creation, building, and management
//! for neural network acceleration on Android devices.

use std::os::raw::c_void;
use std::ptr;
use trustformers_core::error::{CoreError, Result};

use super::bindings::*;

/// NNAPI model builder for constructing neural network models
pub struct NNAPIModelBuilder {
    model: *mut ANeuralNetworksModel,
    operand_count: u32,
    operation_count: u32,
    finalized: bool,
}

impl NNAPIModelBuilder {
    /// Create a new NNAPI model builder
    #[cfg(target_os = "android")]
    pub fn new() -> Result<Self> {
        let mut model: *mut ANeuralNetworksModel = ptr::null_mut();

        let result = unsafe { ANeuralNetworksModel_create(&mut model) };
        if !nnapi_is_success(result) {
            return Err(TrustformersError::runtime_error(format!(
                "Failed to create NNAPI model: {}",
                nnapi_result_to_string(result)
            )).into());
        }

        Ok(Self {
            model,
            operand_count: 0,
            operation_count: 0,
            finalized: false,
        })
    }

    #[cfg(not(target_os = "android"))]
    pub fn new() -> Result<Self> {
        Err(TrustformersError::runtime_error(
            "NNAPI is only available on Android".into(),
        ))
    }

    /// Add an operand to the model
    #[cfg(target_os = "android")]
    pub fn add_operand(&mut self, operand_type: &ANeuralNetworksOperandType) -> Result<u32> {
        if self.finalized {
            return Err(TrustformersError::runtime_error("Model is already finalized".into()).into());
        }

        let result = unsafe { ANeuralNetworksModel_addOperand(self.model, operand_type) };
        if !nnapi_is_success(result) {
            return Err(TrustformersError::runtime_error(format!(
                "Failed to add operand: {}",
                nnapi_result_to_string(result)
            )).into());
        }

        let operand_index = self.operand_count;
        self.operand_count += 1;
        Ok(operand_index)
    }

    #[cfg(not(target_os = "android"))]
    pub fn add_operand(&mut self, _operand_type: &ANeuralNetworksOperandType) -> Result<u32> {
        Err(TrustformersError::runtime_error(
            "NNAPI is only available on Android".into(),
        ))
    }

    /// Set operand value (for constant operands)
    #[cfg(target_os = "android")]
    pub fn set_operand_value(&mut self, index: u32, buffer: &[u8]) -> Result<()> {
        if self.finalized {
            return Err(TrustformersError::runtime_error("Model is already finalized".into()).into());
        }

        let result = unsafe {
            ANeuralNetworksModel_setOperandValue(
                self.model,
                index as i32,
                buffer.as_ptr() as *const c_void,
                buffer.len(),
            )
        };

        if !nnapi_is_success(result) {
            return Err(TrustformersError::runtime_error(format!(
                "Failed to set operand value: {}",
                nnapi_result_to_string(result)
            )).into());
        }

        Ok(())
    }

    #[cfg(not(target_os = "android"))]
    pub fn set_operand_value(&mut self, _index: u32, _buffer: &[u8]) -> Result<()> {
        Err(TrustformersError::runtime_error(
            "NNAPI is only available on Android".into(),
        ))
    }

    /// Add an operation to the model
    #[cfg(target_os = "android")]
    pub fn add_operation(
        &mut self,
        operation_type: i32,
        inputs: &[u32],
        outputs: &[u32],
    ) -> Result<()> {
        if self.finalized {
            return Err(TrustformersError::runtime_error("Model is already finalized".into()).into());
        }

        let result = unsafe {
            ANeuralNetworksModel_addOperation(
                self.model,
                operation_type,
                inputs.len() as u32,
                inputs.as_ptr(),
                outputs.len() as u32,
                outputs.as_ptr(),
            )
        };

        if !nnapi_is_success(result) {
            return Err(TrustformersError::runtime_error(format!(
                "Failed to add operation: {}",
                nnapi_result_to_string(result)
            )).into());
        }

        self.operation_count += 1;
        Ok(())
    }

    #[cfg(not(target_os = "android"))]
    pub fn add_operation(
        &mut self,
        _operation_type: i32,
        _inputs: &[u32],
        _outputs: &[u32],
    ) -> Result<()> {
        Err(TrustformersError::runtime_error(
            "NNAPI is only available on Android".into(),
        ))
    }

    /// Identify model inputs and outputs
    #[cfg(target_os = "android")]
    pub fn identify_inputs_and_outputs(&mut self, inputs: &[u32], outputs: &[u32]) -> Result<()> {
        if self.finalized {
            return Err(TrustformersError::runtime_error("Model is already finalized".into()).into());
        }

        let result = unsafe {
            ANeuralNetworksModel_identifyInputsAndOutputs(
                self.model,
                inputs.len() as u32,
                inputs.as_ptr(),
                outputs.len() as u32,
                outputs.as_ptr(),
            )
        };

        if !nnapi_is_success(result) {
            return Err(TrustformersError::runtime_error(format!(
                "Failed to identify inputs and outputs: {}",
                nnapi_result_to_string(result)
            )).into());
        }

        Ok(())
    }

    #[cfg(not(target_os = "android"))]
    pub fn identify_inputs_and_outputs(&mut self, _inputs: &[u32], _outputs: &[u32]) -> Result<()> {
        Err(TrustformersError::runtime_error(
            "NNAPI is only available on Android".into(),
        ))
    }

    /// Finalize model construction
    #[cfg(target_os = "android")]
    pub fn finalize(&mut self) -> Result<()> {
        if self.finalized {
            return Ok(().into());
        }

        let result = unsafe { ANeuralNetworksModel_finish(self.model) };
        if !nnapi_is_success(result) {
            return Err(TrustformersError::runtime_error(format!(
                "Failed to finalize model: {}",
                nnapi_result_to_string(result)
            )).into());
        }

        self.finalized = true;
        tracing::info!(
            "NNAPI model finalized with {} operands and {} operations",
            self.operand_count,
            self.operation_count
        );
        Ok(())
    }

    #[cfg(not(target_os = "android"))]
    pub fn finalize(&mut self) -> Result<()> {
        Err(TrustformersError::runtime_error(
            "NNAPI is only available on Android".into(),
        ))
    }

    /// Get the raw model pointer (for internal use)
    pub fn get_model_ptr(&self) -> *mut ANeuralNetworksModel {
        self.model
    }

    /// Check if model is finalized
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }

    /// Get number of operands in the model
    pub fn get_operand_count(&self) -> u32 {
        self.operand_count
    }

    /// Get number of operations in the model
    pub fn get_operation_count(&self) -> u32 {
        self.operation_count
    }
}

impl Drop for NNAPIModelBuilder {
    fn drop(&mut self) {
        #[cfg(target_os = "android")]
        if !self.model.is_null() {
            unsafe {
                ANeuralNetworksModel_free(self.model);
            }
        }
    }
}

/// Create operand type for tensor
pub fn create_tensor_operand_type(
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

/// Create operand type for scalar
pub fn create_scalar_operand_type(data_type: i32) -> ANeuralNetworksOperandType {
    ANeuralNetworksOperandType {
        type_: data_type,
        dimensionCount: 0,
        dimensions: ptr::null(),
        scale: 0.0,
        zeroPoint: 0,
    }
}

/// Example model builder: Simple Conv2D + ReLU network
pub struct ExampleConv2DModel {
    builder: NNAPIModelBuilder,
    input_index: u32,
    output_index: u32,
}

impl ExampleConv2DModel {
    /// Create a new example Conv2D model
    pub fn new() -> Result<Self> {
        let mut builder = NNAPIModelBuilder::new()?;
        let (input_index, output_index) = Self::build_model(&mut builder)?;

        Ok(Self {
            builder,
            input_index,
            output_index,
        })
    }

    /// Build the example model structure
    fn build_model(builder: &mut NNAPIModelBuilder) -> Result<(u32, u32)> {
        // Define input tensor (1x224x224x3 NHWC)
        let input_dims = [1u32, 224, 224, 3];
        let input_type = create_tensor_operand_type(
            ANEURALNETWORKS_TENSOR_FLOAT32,
            &input_dims,
            0.0,
            0,
        );
        let input_index = builder.add_operand(&input_type)?;

        // Define weight tensor (32x3x3x3 - 32 filters, 3x3 kernel, 3 input channels)
        let weight_dims = [32u32, 3, 3, 3];
        let weight_type = create_tensor_operand_type(
            ANEURALNETWORKS_TENSOR_FLOAT32,
            &weight_dims,
            0.0,
            0,
        );
        let weight_index = builder.add_operand(&weight_type)?;

        // Initialize weights (simplified - all 0.1)
        let weight_count = 32 * 3 * 3 * 3;
        let weights: Vec<f32> = vec![0.1; weight_count];
        let weight_bytes = unsafe {
            std::slice::from_raw_parts(
                weights.as_ptr() as *const u8,
                weights.len() * std::mem::size_of::<f32>(),
            )
        };
        builder.set_operand_value(weight_index, weight_bytes)?;

        // Define bias tensor (32 values)
        let bias_dims = [32u32];
        let bias_type = create_tensor_operand_type(
            ANEURALNETWORKS_TENSOR_FLOAT32,
            &bias_dims,
            0.0,
            0,
        );
        let bias_index = builder.add_operand(&bias_type)?;

        // Initialize bias (all 0.0)
        let biases: Vec<f32> = vec![0.0; 32];
        let bias_bytes = unsafe {
            std::slice::from_raw_parts(
                biases.as_ptr() as *const u8,
                biases.len() * std::mem::size_of::<f32>(),
            )
        };
        builder.set_operand_value(bias_index, bias_bytes)?;

        // Add Conv2D parameters as scalar operands
        let padding_left = 1i32;
        let padding_right = 1i32;
        let padding_top = 1i32;
        let padding_bottom = 1i32;
        let stride_width = 1i32;
        let stride_height = 1i32;
        let activation = ANEURALNETWORKS_RELU;

        let int32_type = create_scalar_operand_type(ANEURALNETWORKS_INT32);

        // Add parameter operands
        let pad_left_index = builder.add_operand(&int32_type)?;
        builder.set_operand_value(
            pad_left_index,
            unsafe { std::slice::from_raw_parts(&padding_left as *const i32 as *const u8, 4) },
        )?;

        let pad_right_index = builder.add_operand(&int32_type)?;
        builder.set_operand_value(
            pad_right_index,
            unsafe { std::slice::from_raw_parts(&padding_right as *const i32 as *const u8, 4) },
        )?;

        let pad_top_index = builder.add_operand(&int32_type)?;
        builder.set_operand_value(
            pad_top_index,
            unsafe { std::slice::from_raw_parts(&padding_top as *const i32 as *const u8, 4) },
        )?;

        let pad_bottom_index = builder.add_operand(&int32_type)?;
        builder.set_operand_value(
            pad_bottom_index,
            unsafe { std::slice::from_raw_parts(&padding_bottom as *const i32 as *const u8, 4) },
        )?;

        let stride_w_index = builder.add_operand(&int32_type)?;
        builder.set_operand_value(
            stride_w_index,
            unsafe { std::slice::from_raw_parts(&stride_width as *const i32 as *const u8, 4) },
        )?;

        let stride_h_index = builder.add_operand(&int32_type)?;
        builder.set_operand_value(
            stride_h_index,
            unsafe { std::slice::from_raw_parts(&stride_height as *const i32 as *const u8, 4) },
        )?;

        let activation_index = builder.add_operand(&int32_type)?;
        builder.set_operand_value(
            activation_index,
            unsafe { std::slice::from_raw_parts(&activation as *const i32 as *const u8, 4) },
        )?;

        // Define output tensor (1x222x222x32 NHWC after conv with padding=1, stride=1)
        let output_dims = [1u32, 222, 222, 32];
        let output_type = create_tensor_operand_type(
            ANEURALNETWORKS_TENSOR_FLOAT32,
            &output_dims,
            0.0,
            0,
        );
        let output_index = builder.add_operand(&output_type)?;

        // Add Conv2D operation
        let conv_inputs = vec![
            input_index,
            weight_index,
            bias_index,
            pad_left_index,
            pad_right_index,
            pad_top_index,
            pad_bottom_index,
            stride_w_index,
            stride_h_index,
            activation_index,
        ];
        let conv_outputs = vec![output_index];

        builder.add_operation(ANEURALNETWORKS_CONV_2D, &conv_inputs, &conv_outputs)?;

        // Identify model inputs and outputs
        builder.identify_inputs_and_outputs(&[input_index], &[output_index])?;

        // Finalize the model
        builder.finalize()?;

        tracing::info!("Example Conv2D model built successfully");
        Ok((input_index, output_index))
    }

    /// Get input operand index
    pub fn get_input_index(&self) -> u32 {
        self.input_index
    }

    /// Get output operand index
    pub fn get_output_index(&self) -> u32 {
        self.output_index
    }

    /// Get the model pointer
    pub fn get_model_ptr(&self) -> *mut ANeuralNetworksModel {
        self.builder.get_model_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operand_type_creation() {
        let dims = [1, 224, 224, 3];
        let tensor_type = create_tensor_operand_type(
            ANEURALNETWORKS_TENSOR_FLOAT32,
            &dims,
            0.0,
            0,
        );

        assert_eq!(tensor_type.type_, ANEURALNETWORKS_TENSOR_FLOAT32);
        assert_eq!(tensor_type.dimensionCount, 4);
        assert_eq!(tensor_type.scale, 0.0);
        assert_eq!(tensor_type.zeroPoint, 0);

        let scalar_type = create_scalar_operand_type(ANEURALNETWORKS_INT32);
        assert_eq!(scalar_type.type_, ANEURALNETWORKS_INT32);
        assert_eq!(scalar_type.dimensionCount, 0);
    }

    #[cfg(target_os = "android")]
    #[test]
    fn test_model_builder() {
        let builder = NNAPIModelBuilder::new();
        if builder.is_err() {
            // NNAPI might not be available in test environment
            return;
        }

        let mut builder = builder.unwrap();
        assert_eq!(builder.get_operand_count(), 0);
        assert_eq!(builder.get_operation_count(), 0);
        assert!(!builder.is_finalized().into());
    }

    #[cfg(target_os = "android")]
    #[test]
    fn test_example_model() {
        let model = ExampleConv2DModel::new();
        if model.is_err() {
            // NNAPI might not be available in test environment
            return;
        }

        let model = model.unwrap();
        assert!(!model.get_model_ptr().is_null());
        assert_eq!(model.get_input_index(), 0);
    }
}