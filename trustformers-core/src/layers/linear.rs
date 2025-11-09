//! Linear (fully connected) layer implementation.
//!
//! This module provides the `Linear` layer, which performs affine transformations
//! of the form: `y = xW^T + b`, where `W` is the weight matrix and `b` is the
//! optional bias vector.

use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use crate::traits::Layer;

/// A linear transformation layer (fully connected layer).
///
/// The `Linear` layer applies a linear transformation to the incoming data:
/// `y = xW^T + b`. This is one of the most fundamental building blocks in
/// neural networks.
///
/// # Parameters
///
/// - `weight`: Learnable weight matrix of shape `[out_features, in_features]`
/// - `bias`: Optional learnable bias vector of shape `[out_features]`
///
/// # Input/Output Shapes
///
/// - Input: `[..., in_features]` - Can be 2D or 3D
/// - Output: `[..., out_features]` - Same number of dimensions as input
///
/// # Example
///
/// ```no_run
/// use trustformers_core::layers::Linear;
/// use trustformers_core::tensor::Tensor;
/// use trustformers_core::traits::Layer;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create a linear layer: 768 → 3072
/// let linear = Linear::new(768, 3072, true)?;
///
/// // Apply to 2D input: [seq_len, in_features]
/// let input_2d = Tensor::randn(&[128, 768])?;
/// let output_2d = linear.forward(input_2d)?;  // Shape: [128, 3072]
///
/// // Apply to 3D input: [batch, seq_len, in_features]
/// let input_3d = Tensor::randn(&[4, 128, 768])?;
/// let output_3d = linear.forward(input_3d)?;  // Shape: [4, 128, 3072]
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    /// Creates a new linear layer.
    ///
    /// # Arguments
    ///
    /// * `in_features` - Size of each input sample
    /// * `out_features` - Size of each output sample
    /// * `bias` - Whether to include a learnable bias
    ///
    /// # Returns
    ///
    /// A new `Linear` layer with randomly initialized weights using a normal
    /// distribution, and bias initialized to zeros if enabled.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use trustformers_core::layers::Linear;
    ///
    /// // Linear layer without bias
    /// let linear1 = Linear::new(512, 1024, false);
    ///
    /// // Linear layer with bias
    /// let linear2 = Linear::new(512, 1024, true);
    /// ```
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let weight = Tensor::randn(&[out_features, in_features]).unwrap();
        let bias = if bias { Some(Tensor::zeros(&[out_features]).unwrap()) } else { None };

        Self { weight, bias }
    }

    /// Sets the weight matrix for this layer.
    ///
    /// # Arguments
    ///
    /// * `weight` - The new weight tensor, must have shape `[out_features, in_features]`
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful.
    ///
    /// # Note
    ///
    /// This method is typically used when loading pretrained weights.
    pub fn set_weight(&mut self, weight: Tensor) -> Result<()> {
        self.weight = weight;
        Ok(())
    }

    /// Sets the bias vector for this layer.
    ///
    /// # Arguments
    ///
    /// * `bias` - The new bias tensor, must have shape `[out_features]`
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful.
    ///
    /// # Note
    ///
    /// This will enable bias even if the layer was created without bias.
    pub fn set_bias(&mut self, bias: Tensor) -> Result<()> {
        self.bias = Some(bias);
        Ok(())
    }

    /// Returns a reference to the weight matrix.
    ///
    /// # Returns
    ///
    /// A reference to the weight tensor of shape `[out_features, in_features]`.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Returns a reference to the bias vector if present.
    ///
    /// # Returns
    ///
    /// `Some(&bias)` if bias is enabled, `None` otherwise.
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    /// Returns the total number of learnable parameters in this layer.
    ///
    /// # Returns
    ///
    /// The total parameter count including weights and bias (if present).
    pub fn parameter_count(&self) -> usize {
        let weight_count = self.weight.len();
        let bias_count = self.bias.as_ref().map_or(0, |b| b.len());
        weight_count + bias_count
    }
}

impl Layer for Linear {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Handle different input shapes for matmul
        let input_shape = input.shape();
        let weight_t = self.weight.transpose(0, 1)?;

        let output = if input_shape.len() == 2 {
            // Standard 2D input: [seq_len, hidden_size] x [hidden_size, out_features]
            input.matmul(&weight_t)?
        } else if input_shape.len() == 3 {
            // Batched 3D input: [batch, seq_len, hidden_size] x [hidden_size, out_features]
            // Handle manually since tensor.matmul doesn't support 3D x 2D
            match (&input, &weight_t) {
                (Tensor::F32(inp), Tensor::F32(w)) => {
                    let batch = input_shape[0];
                    let seq_len = input_shape[1];
                    let hidden = input_shape[2];
                    let out_features = w.shape()[1];

                    // Ensure contiguous layout before reshaping input to 2D for dot product
                    let inp_contiguous = inp.to_owned();
                    let inp_2d = inp_contiguous
                        .into_shape_with_order([batch * seq_len, hidden])
                        .map_err(|e| {
                            TrustformersError::shape_error(format!(
                                "Failed to reshape input: {}",
                                e
                            ))
                        })?;

                    // Ensure contiguous layout for weight and convert to 2D for dot product
                    let w_contiguous = w.to_owned();
                    let w_2d = w_contiguous.into_dimensionality::<ndarray::Ix2>().map_err(|e| {
                        TrustformersError::shape_error(format!(
                            "Failed to convert weight to 2D: {}",
                            e
                        ))
                    })?;

                    // Perform dot product directly with ndarray
                    let out_2d = inp_2d.dot(&w_2d);

                    // Reshape back to 3D
                    let out_3d = out_2d
                        .into_shape_with_order(ndarray::IxDyn(&[batch, seq_len, out_features]))
                        .map_err(|e| {
                            TrustformersError::shape_error(format!(
                                "Failed to reshape output: {}",
                                e
                            ))
                        })?;

                    Tensor::F32(out_3d)
                },
                (Tensor::F64(inp), Tensor::F64(w)) => {
                    let batch = input_shape[0];
                    let seq_len = input_shape[1];
                    let hidden = input_shape[2];
                    let out_features = w.shape()[1];

                    // Ensure contiguous layout before reshaping
                    let inp_contiguous = inp.to_owned();
                    let inp_2d = inp_contiguous
                        .into_shape_with_order([batch * seq_len, hidden])
                        .map_err(|e| {
                            TrustformersError::shape_error(format!(
                                "Failed to reshape input: {}",
                                e
                            ))
                        })?;

                    // Ensure contiguous layout for weight and convert to 2D
                    let w_contiguous = w.to_owned();
                    let w_2d = w_contiguous.into_dimensionality::<ndarray::Ix2>().map_err(|e| {
                        TrustformersError::shape_error(format!(
                            "Failed to convert weight to 2D: {}",
                            e
                        ))
                    })?;

                    let out_2d = inp_2d.dot(&w_2d);

                    let out_3d = out_2d
                        .into_shape_with_order(ndarray::IxDyn(&[batch, seq_len, out_features]))
                        .map_err(|e| {
                            TrustformersError::shape_error(format!(
                                "Failed to reshape output: {}",
                                e
                            ))
                        })?;

                    Tensor::F64(out_3d)
                },
                _ => {
                    return Err(TrustformersError::tensor_op_error(
                        "Unsupported tensor types for 3D linear layer",
                        "Linear::forward",
                    ))
                },
            }
        } else {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "Linear layer doesn't support input with {} dimensions",
                    input_shape.len()
                ),
                "Linear::forward",
            ));
        };

        if let Some(ref bias) = self.bias {
            // Handle broadcasting for bias addition
            match (&output, bias) {
                (Tensor::F32(out_arr), Tensor::F32(bias_arr)) => {
                    // Broadcast bias to match output shape
                    let result = out_arr + bias_arr;
                    Ok(Tensor::F32(result))
                },
                _ => output.add(bias),
            }
        } else {
            Ok(output)
        }
    }
}
