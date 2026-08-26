//! Operations for automatic differentiation.
//!
//! This module provides implementations of differentiable operations
//! that can be used in the computation graph.

#![allow(unused_variables)] // Autodiff operations with reserved parameters

use super::graph::{GradientFunction, OperationType};
use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use std::sync::Arc;

/// Automatic differentiation operation
pub struct AutodiffOp {
    pub op_type: OperationType,
    pub grad_fn: Arc<dyn GradientFunction>,
}

/// Operation type for high-level operations
#[derive(Debug, Clone)]
pub enum OpType {
    /// Basic arithmetic
    Add,
    Subtract,
    Multiply,
    Divide,
    MatrixMultiply,

    /// Unary operations
    Negate,
    Square,
    Sqrt,
    Log,
    Exp,
    Reciprocal,

    /// Activation functions
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU(f32),
    Softmax,
    LogSoftmax,
    GELU,
    Swish,

    /// Tensor operations
    Reshape(Vec<usize>),
    Transpose(Vec<usize>),
    Slice(Vec<std::ops::Range<usize>>),
    Concat(usize),
    Split(Vec<usize>),
    Pad(Vec<(usize, usize)>),

    /// Reduction operations
    Sum(Option<Vec<usize>>),
    Mean(Option<Vec<usize>>),
    Max(Option<Vec<usize>>),
    Min(Option<Vec<usize>>),
    Var(Option<Vec<usize>>),
    Std(Option<Vec<usize>>),

    /// Normalization operations
    LayerNorm(f32),
    BatchNorm(f32),
    GroupNorm(usize, f32),
    InstanceNorm(f32),

    /// Loss functions
    MSELoss,
    CrossEntropyLoss,
    NLLLoss,
    BCELoss,

    /// Regularization
    Dropout(f32),

    /// Convolution operations
    Conv2D {
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    },
    MaxPool2D {
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    },
    AvgPool2D {
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    },

    /// Custom operation
    Custom(String),
}

/// Gradient function implementations
pub mod grad_fn {
    use super::*;

    /// Addition gradient function
    pub struct AddGradFn;

    impl GradientFunction for AddGradFn {
        fn backward(&self, grad_output: &Tensor, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
            if inputs.len() != 2 {
                return Err(TrustformersError::tensor_op_error(
                    "Add requires exactly 2 inputs",
                    "AddGradFn::backward",
                ));
            }

            // Gradient of addition is the same for both inputs
            let grad_a = grad_output.clone();
            let grad_b = grad_output.clone();

            Ok(vec![grad_a, grad_b])
        }

        fn operation_type(&self) -> OperationType {
            OperationType::Add
        }
    }

    /// Subtraction gradient function
    pub struct SubtractGradFn;

    impl GradientFunction for SubtractGradFn {
        fn backward(&self, grad_output: &Tensor, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
            if inputs.len() != 2 {
                return Err(TrustformersError::tensor_op_error(
                    "Subtract requires exactly 2 inputs",
                    "SubtractGradFn::backward",
                ));
            }

            // Gradient of subtraction: da = dout, db = -dout
            let grad_a = grad_output.clone();
            let grad_b = grad_output.neg()?;

            Ok(vec![grad_a, grad_b])
        }

        fn operation_type(&self) -> OperationType {
            OperationType::Subtract
        }
    }

    /// Multiplication gradient function
    pub struct MultiplyGradFn;

    impl GradientFunction for MultiplyGradFn {
        fn backward(&self, grad_output: &Tensor, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
            if inputs.len() != 2 {
                return Err(TrustformersError::tensor_op_error(
                    "Multiply requires exactly 2 inputs",
                    "MultiplyGradFn::backward",
                ));
            }

            let a = inputs[0];
            let b = inputs[1];

            // Gradient of multiplication: da = dout * b, db = dout * a
            let grad_a = grad_output.mul(b)?;
            let grad_b = grad_output.mul(a)?;

            Ok(vec![grad_a, grad_b])
        }

        fn operation_type(&self) -> OperationType {
            OperationType::Multiply
        }
    }

    /// Division gradient function
    pub struct DivideGradFn;

    impl GradientFunction for DivideGradFn {
        fn backward(&self, grad_output: &Tensor, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
            if inputs.len() != 2 {
                return Err(TrustformersError::tensor_op_error(
                    "Divide requires exactly 2 inputs",
                    "DivideGradFn::backward",
                ));
            }

            let a = inputs[0];
            let b = inputs[1];

            // Gradient of division: da = dout / b, db = -dout * a / (b * b)
            let grad_a = grad_output.div(b)?;
            let b_squared = b.mul(b)?;
            let grad_b = grad_output.mul(a)?.neg()?.div(&b_squared)?;

            Ok(vec![grad_a, grad_b])
        }

        fn operation_type(&self) -> OperationType {
            OperationType::Divide
        }
    }

    /// Matrix multiplication gradient function
    pub struct MatMulGradFn;

    impl GradientFunction for MatMulGradFn {
        fn backward(&self, grad_output: &Tensor, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
            if inputs.len() != 2 {
                return Err(TrustformersError::tensor_op_error(
                    "MatMul requires exactly 2 inputs",
                    "MatMulGradFn::backward",
                ));
            }

            let a = inputs[0];
            let b = inputs[1];

            // Gradient of matrix multiplication: da = dout @ b^T, db = a^T @ dout
            let b_transposed = b.transpose(1, 0)?;
            let grad_a = grad_output.matmul(&b_transposed)?;

            let a_transposed = a.transpose(1, 0)?;
            let grad_b = a_transposed.matmul(grad_output)?;

            Ok(vec![grad_a, grad_b])
        }

        fn operation_type(&self) -> OperationType {
            OperationType::MatrixMultiply
        }
    }

    /// Sigmoid gradient function
    pub struct SigmoidGradFn;

    impl GradientFunction for SigmoidGradFn {
        fn backward(&self, grad_output: &Tensor, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
            if inputs.len() != 1 {
                return Err(TrustformersError::tensor_op_error(
                    "Sigmoid requires exactly 1 input",
                    "SigmoidGradFn::backward",
                ));
            }

            let x = inputs[0];

            // Gradient of sigmoid: dout * sigmoid(x) * (1 - sigmoid(x))
            let sigmoid_x = x.sigmoid()?;
            let one = Tensor::ones(&sigmoid_x.shape())?;
            let one_minus_sigmoid = one.sub(&sigmoid_x)?;
            let grad_input = grad_output.mul(&sigmoid_x)?.mul(&one_minus_sigmoid)?;

            Ok(vec![grad_input])
        }

        fn operation_type(&self) -> OperationType {
            OperationType::Sigmoid
        }
    }

    /// Tanh gradient function
    pub struct TanhGradFn;

    impl GradientFunction for TanhGradFn {
        fn backward(&self, grad_output: &Tensor, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
            if inputs.len() != 1 {
                return Err(TrustformersError::tensor_op_error(
                    "Tanh requires exactly 1 input",
                    "TanhGradFn::backward",
                ));
            }

            let x = inputs[0];

            // Gradient of tanh: dout * (1 - tanh(x)^2)
            let tanh_x = x.tanh()?;
            let tanh_squared = tanh_x.mul(&tanh_x)?;
            let one = Tensor::ones(&tanh_squared.shape())?;
            let grad_input = grad_output.mul(&one.sub(&tanh_squared)?)?;

            Ok(vec![grad_input])
        }

        fn operation_type(&self) -> OperationType {
            OperationType::Tanh
        }
    }

    /// ReLU gradient function
    pub struct ReLUGradFn;

    impl GradientFunction for ReLUGradFn {
        fn backward(&self, grad_output: &Tensor, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
            if inputs.len() != 1 {
                return Err(TrustformersError::tensor_op_error(
                    "ReLU requires exactly 1 input",
                    "ReLUGradFn::backward",
                ));
            }

            let x = inputs[0];

            // Gradient of ReLU: dout * (x > 0)
            let zero = Tensor::zeros(&x.shape())?;
            let mask = x.greater(&zero)?;
            let grad_input = grad_output.mul(&mask)?;

            Ok(vec![grad_input])
        }

        fn operation_type(&self) -> OperationType {
            OperationType::ReLU
        }
    }

    /// GELU gradient function
    pub struct GELUGradFn;

    impl GradientFunction for GELUGradFn {
        fn backward(&self, grad_output: &Tensor, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
            if inputs.len() != 1 {
                return Err(TrustformersError::tensor_op_error(
                    "GELU requires exactly 1 input",
                    "GELUGradFn::backward",
                ));
            }

            let x = inputs[0];

            // GELU gradient: dout * (0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3))) +
            //                        x * 0.5 * (1 - tanh^2(sqrt(2/π) * (x + 0.044715 * x^3))) *
            //                        sqrt(2/π) * (1 + 3 * 0.044715 * x^2))

            // Compute GELU gradient using tensor operations
            let x_cubed = x.pow(3.0)?;
            let tanh_arg = x.add(&x_cubed.scalar_mul(0.044715)?)?;
            let tanh_arg_scaled = tanh_arg.scalar_mul(0.7978845608)?; // sqrt(2/π)
            let tanh_val = tanh_arg_scaled.tanh()?;
            let one = Tensor::ones(&x.shape())?;
            let tanh_plus_one = tanh_val.add(&one)?;
            let first_term = tanh_plus_one.scalar_mul(0.5)?;

            // Second term: x * 0.5 * (1 - tanh^2) * sqrt(2/π) * (1 + 3 * 0.044715 * x^2)
            let tanh_squared = tanh_val.pow(2.0)?;
            let one_minus_tanh_sq = one.sub(&tanh_squared)?;
            let x_squared = x.pow(2.0)?;
            let x_sq_term = x_squared.scalar_mul(3.0 * 0.044715)?;
            let x_sq_term_plus_one = x_sq_term.add(&one)?;
            let second_term = x
                .mul(&one_minus_tanh_sq)?
                .scalar_mul(0.5)?
                .scalar_mul(0.7978845608)?
                .mul(&x_sq_term_plus_one)?;

            let gelu_grad = first_term.add(&second_term)?;
            let result = grad_output.mul(&gelu_grad)?;

            Ok(vec![result])
        }

        fn operation_type(&self) -> OperationType {
            OperationType::Custom("GELU".to_string())
        }
    }

    /// Softmax gradient function
    pub struct SoftmaxGradFn {
        axis: i32,
    }

    impl SoftmaxGradFn {
        pub fn new(axis: i32) -> Self {
            Self { axis }
        }
    }

    impl GradientFunction for SoftmaxGradFn {
        fn backward(&self, grad_output: &Tensor, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
            if inputs.len() != 1 {
                return Err(TrustformersError::tensor_op_error(
                    "Softmax requires exactly 1 input",
                    "SoftmaxGradFn::backward",
                ));
            }

            let x = inputs[0];

            // Gradient of softmax: softmax(x) * (grad_output - sum(grad_output * softmax(x)))
            let softmax_x = x.softmax(self.axis)?;
            let grad_softmax = grad_output.mul(&softmax_x)?;

            // Convert negative axis to positive
            let axis = if self.axis < 0 {
                (x.shape().len() as i32 + self.axis) as usize
            } else {
                self.axis as usize
            };

            // Sum along the specified axis
            let sum_grad = grad_softmax.sum_axes(&[axis])?;

            // Subtract the sum from grad_output
            let diff = grad_output.sub(&sum_grad)?;

            // Multiply by softmax
            let result = softmax_x.mul(&diff)?;

            Ok(vec![result])
        }

        fn operation_type(&self) -> OperationType {
            OperationType::Softmax
        }
    }

    /// Sum gradient function
    pub struct SumGradFn {
        axes: Option<Vec<usize>>,
        original_shape: Vec<usize>,
    }

    impl SumGradFn {
        pub fn new(axes: Option<Vec<usize>>, original_shape: Vec<usize>) -> Self {
            Self {
                axes,
                original_shape,
            }
        }
    }

    impl GradientFunction for SumGradFn {
        fn backward(&self, grad_output: &Tensor, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
            if inputs.len() != 1 {
                return Err(TrustformersError::tensor_op_error(
                    "Sum requires exactly 1 input",
                    "SumGradFn::backward",
                ));
            }

            // Gradient of sum: broadcast the gradient back to original shape
            let grad_input = self.broadcast_gradient(grad_output, &self.original_shape)?;

            Ok(vec![grad_input])
        }

        fn operation_type(&self) -> OperationType {
            OperationType::Sum(self.axes.clone())
        }
    }

    impl SumGradFn {
        fn broadcast_gradient(
            &self,
            grad_output: &Tensor,
            original_shape: &[usize],
        ) -> Result<Tensor> {
            if let Some(axes) = &self.axes {
                // Sum was performed along specific axes
                let mut result = grad_output.clone();
                for &axis in axes {
                    result = result.unsqueeze(axis)?;
                }
                result.broadcast_to(original_shape)
            } else {
                // Sum was performed along all axes
                grad_output.broadcast_to(original_shape)
            }
        }
    }

    /// Mean gradient function
    pub struct MeanGradFn {
        axes: Option<Vec<usize>>,
        original_shape: Vec<usize>,
    }

    impl MeanGradFn {
        pub fn new(axes: Option<Vec<usize>>, original_shape: Vec<usize>) -> Self {
            Self {
                axes,
                original_shape,
            }
        }
    }

    impl GradientFunction for MeanGradFn {
        fn backward(&self, grad_output: &Tensor, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
            if inputs.len() != 1 {
                return Err(TrustformersError::tensor_op_error(
                    "Mean requires exactly 1 input",
                    "MeanGradFn::backward",
                ));
            }

            // Gradient of mean: broadcast the gradient back and divide by number of elements
            let grad_broadcasted = self.broadcast_gradient(grad_output, &self.original_shape)?;

            // Compute the number of elements that were averaged
            let num_elements = if let Some(axes) = &self.axes {
                axes.iter().map(|&axis| self.original_shape[axis]).product::<usize>()
            } else {
                self.original_shape.iter().product::<usize>()
            };

            let grad_input = grad_broadcasted.scalar_div(num_elements as f32)?;

            Ok(vec![grad_input])
        }

        fn operation_type(&self) -> OperationType {
            OperationType::Mean(self.axes.clone())
        }
    }

    impl MeanGradFn {
        fn broadcast_gradient(
            &self,
            grad_output: &Tensor,
            original_shape: &[usize],
        ) -> Result<Tensor> {
            if let Some(axes) = &self.axes {
                // Mean was performed along specific axes
                let mut result = grad_output.clone();
                for &axis in axes {
                    result = result.unsqueeze(axis)?;
                }
                result.broadcast_to(original_shape)
            } else {
                // Mean was performed along all axes
                grad_output.broadcast_to(original_shape)
            }
        }
    }

    /// Reshape gradient function
    pub struct ReshapeGradFn {
        original_shape: Vec<usize>,
    }

    impl ReshapeGradFn {
        pub fn new(original_shape: Vec<usize>) -> Self {
            Self { original_shape }
        }
    }

    impl GradientFunction for ReshapeGradFn {
        fn backward(&self, grad_output: &Tensor, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
            if inputs.len() != 1 {
                return Err(TrustformersError::tensor_op_error(
                    "Reshape requires exactly 1 input",
                    "ReshapeGradFn::backward",
                ));
            }

            // Gradient of reshape: reshape gradient back to original shape
            let grad_input = grad_output.reshape(&self.original_shape)?;

            Ok(vec![grad_input])
        }

        fn operation_type(&self) -> OperationType {
            OperationType::Reshape(self.original_shape.clone())
        }
    }

    /// Transpose (axis permutation) gradient function.
    ///
    /// The forward operation reorders the axes of its input by `permutation`
    /// (an empty permutation means "reverse every axis", matching
    /// `OperationType::Transpose` in [`crate::autodiff::variable`]). The
    /// gradient therefore applies the *inverse* permutation to `grad_output`.
    pub struct TransposeGradFn {
        permutation: Vec<usize>,
    }

    impl TransposeGradFn {
        pub fn new(permutation: Vec<usize>) -> Self {
            Self { permutation }
        }
    }

    impl GradientFunction for TransposeGradFn {
        fn backward(&self, grad_output: &Tensor, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
            if inputs.len() != 1 {
                return Err(TrustformersError::tensor_op_error(
                    "Transpose requires exactly 1 input",
                    "TransposeGradFn::backward",
                ));
            }

            // Gradient of an axis permutation is the inverse permutation applied
            // to the incoming gradient. This is a general N-d permutation: the
            // previous code applied only `inverse[0]`/`inverse[1]` as a single
            // axis swap, which produced a silently wrong gradient layout for any
            // rank > 2 permutation that was not a plain 2-axis swap.
            let ndim = grad_output.shape().len();
            let inverse_permutation = if self.permutation.is_empty() {
                // Empty permutation == reverse every axis; it is its own inverse.
                (0..ndim).rev().collect::<Vec<usize>>()
            } else {
                if self.permutation.len() != ndim {
                    return Err(TrustformersError::shape_error(format!(
                        "TransposeGradFn: permutation of length {} does not match the {}-D gradient",
                        self.permutation.len(),
                        ndim
                    )));
                }
                self.compute_inverse_permutation()?
            };

            Ok(vec![grad_output.permute(&inverse_permutation)?])
        }

        fn operation_type(&self) -> OperationType {
            OperationType::Transpose(self.permutation.clone())
        }
    }

    impl TransposeGradFn {
        fn compute_inverse_permutation(&self) -> Result<Vec<usize>> {
            let ndim = self.permutation.len();
            let mut inverse = vec![usize::MAX; ndim];
            for (i, &p) in self.permutation.iter().enumerate() {
                if p >= ndim {
                    return Err(TrustformersError::tensor_op_error(
                        &format!("Invalid permutation index: {}", p),
                        "TransposeGradFn::compute_inverse_permutation",
                    ));
                }
                // A permutation must be a bijection; a repeated index would
                // silently overwrite a slot and leave `usize::MAX` elsewhere.
                if inverse[p] != usize::MAX {
                    return Err(TrustformersError::tensor_op_error(
                        &format!("Duplicate permutation index: {}", p),
                        "TransposeGradFn::compute_inverse_permutation",
                    ));
                }
                inverse[p] = i;
            }
            Ok(inverse)
        }
    }

    /// Layer normalization gradient function.
    ///
    /// Forward pass (per normalized group of `n` elements):
    ///
    /// ```text
    /// mu     = mean(x)
    /// sigma2 = mean((x - mu)^2)
    /// rstd   = 1 / sqrt(sigma2 + eps)
    /// xhat   = (x - mu) * rstd
    /// y      = gamma * xhat + beta
    /// ```
    ///
    /// Backward pass, with `g = dL/dy * gamma`:
    ///
    /// ```text
    /// dL/dx     = rstd * (g - mean(g) - xhat * mean(g * xhat))
    /// dL/dgamma = sum_over_batch(dL/dy * xhat)
    /// dL/dbeta  = sum_over_batch(dL/dy)
    /// ```
    ///
    /// The normalized axes are the trailing axes of `input` matched by the shape
    /// of `weight`, which is the convention used by `LayerNorm` layers: `weight`
    /// and `bias` are broadcast over every leading (batch) axis.
    pub struct LayerNormGradFn {
        epsilon: f32,
    }

    impl LayerNormGradFn {
        pub fn new(epsilon: f32) -> Self {
            Self { epsilon }
        }
    }

    impl GradientFunction for LayerNormGradFn {
        fn backward(&self, grad_output: &Tensor, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
            if inputs.len() != 3 {
                return Err(TrustformersError::tensor_op_error(
                    "LayerNorm requires exactly 3 inputs (input, weight, bias)",
                    "LayerNormGradFn::backward",
                ));
            }

            layer_norm_backward(grad_output, inputs[0], inputs[1], inputs[2], self.epsilon)
        }

        fn operation_type(&self) -> OperationType {
            OperationType::LayerNorm(self.epsilon)
        }
    }

    /// Exact LayerNorm backward pass over the trailing `weight.ndim()` axes.
    ///
    /// Returns `[dL/dinput, dL/dweight, dL/dbias]`.
    pub(crate) fn layer_norm_backward(
        grad_output: &Tensor,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        epsilon: f32,
    ) -> Result<Vec<Tensor>> {
        let input_shape = input.shape();
        let weight_shape = weight.shape();
        let bias_shape = bias.shape();
        let grad_shape = grad_output.shape();

        if grad_shape != input_shape {
            return Err(TrustformersError::shape_error(format!(
                "LayerNorm backward: gradient shape {:?} does not match input shape {:?}",
                grad_shape, input_shape
            )));
        }
        if weight_shape != bias_shape {
            return Err(TrustformersError::shape_error(format!(
                "LayerNorm backward: weight shape {:?} does not match bias shape {:?}",
                weight_shape, bias_shape
            )));
        }
        if weight_shape.is_empty() || weight_shape.len() > input_shape.len() {
            return Err(TrustformersError::shape_error(format!(
                "LayerNorm backward: weight shape {:?} is not a suffix of input shape {:?}",
                weight_shape, input_shape
            )));
        }
        let split = input_shape.len() - weight_shape.len();
        if input_shape[split..] != weight_shape[..] {
            return Err(TrustformersError::shape_error(format!(
                "LayerNorm backward: weight shape {:?} is not a suffix of input shape {:?}",
                weight_shape, input_shape
            )));
        }

        let normalized_len: usize = weight_shape.iter().product();
        if normalized_len == 0 {
            return Err(TrustformersError::shape_error(
                "LayerNorm backward: normalized axes must not be empty".to_string(),
            ));
        }

        let x = input.to_vec_f32()?;
        let g_out = grad_output.to_vec_f32()?;
        let gamma = weight.to_vec_f32()?;
        if gamma.len() != normalized_len {
            return Err(TrustformersError::shape_error(format!(
                "LayerNorm backward: weight holds {} values but the normalized axes cover {}",
                gamma.len(),
                normalized_len
            )));
        }

        let rows = x.len() / normalized_len;
        let n = normalized_len as f32;

        let mut grad_input = vec![0.0f32; x.len()];
        let mut grad_weight = vec![0.0f32; normalized_len];
        let mut grad_bias = vec![0.0f32; normalized_len];

        for row in 0..rows {
            let offset = row * normalized_len;
            let x_row = &x[offset..offset + normalized_len];
            let g_row = &g_out[offset..offset + normalized_len];

            // Forward statistics (recomputed; the GradientFunction trait carries
            // no forward cache, and one extra streaming pass is cheap next to a
            // wrong gradient).
            let mean = x_row.iter().sum::<f32>() / n;
            let variance = x_row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
            let rstd = 1.0 / (variance + epsilon).sqrt();

            // Accumulate the two reduction terms in one pass.
            let mut sum_g = 0.0f32;
            let mut sum_g_xhat = 0.0f32;
            for i in 0..normalized_len {
                let x_hat = (x_row[i] - mean) * rstd;
                let g = g_row[i] * gamma[i];
                sum_g += g;
                sum_g_xhat += g * x_hat;

                grad_weight[i] += g_row[i] * x_hat;
                grad_bias[i] += g_row[i];
            }
            let mean_g = sum_g / n;
            let mean_g_xhat = sum_g_xhat / n;

            for i in 0..normalized_len {
                let x_hat = (x_row[i] - mean) * rstd;
                let g = g_row[i] * gamma[i];
                grad_input[offset + i] = rstd * (g - mean_g - x_hat * mean_g_xhat);
            }
        }

        Ok(vec![
            Tensor::from_vec(grad_input, input_shape.as_slice())?,
            Tensor::from_vec(grad_weight, weight_shape.as_slice())?,
            Tensor::from_vec(grad_bias, bias_shape.as_slice())?,
        ])
    }

    /// MSE Loss gradient function
    pub struct MSELossGradFn;

    impl GradientFunction for MSELossGradFn {
        fn backward(&self, grad_output: &Tensor, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
            if inputs.len() != 2 {
                return Err(TrustformersError::tensor_op_error(
                    "MSELoss requires exactly 2 inputs (prediction, target)",
                    "MSELossGradFn::backward",
                ));
            }

            let prediction = inputs[0];
            let target = inputs[1];

            // Gradient of MSE loss: 2 * (prediction - target) / N
            let diff = prediction.sub(target)?;
            let grad_prediction = diff.scalar_mul(2.0)?;
            let grad_target = grad_prediction.neg()?;

            Ok(vec![grad_prediction, grad_target])
        }

        fn operation_type(&self) -> OperationType {
            OperationType::Custom("MSELoss".to_string())
        }
    }

    /// Cross entropy loss gradient function
    pub struct CrossEntropyLossGradFn;

    impl GradientFunction for CrossEntropyLossGradFn {
        fn backward(&self, grad_output: &Tensor, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
            if inputs.len() != 2 {
                return Err(TrustformersError::tensor_op_error(
                    "CrossEntropyLoss requires exactly 2 inputs (logits, labels)",
                    "CrossEntropyLossGradFn::backward",
                ));
            }

            let logits = inputs[0];
            let labels = inputs[1];

            // Gradient of cross entropy loss: softmax(logits) - labels
            let softmax_logits = logits.softmax(-1)?;
            let grad_logits = softmax_logits.sub(labels)?;
            let grad_labels = grad_logits.neg()?;

            Ok(vec![grad_logits, grad_labels])
        }

        fn operation_type(&self) -> OperationType {
            OperationType::Custom("CrossEntropyLoss".to_string())
        }
    }
}

/// Helper functions for creating gradient functions
impl AutodiffOp {
    /// Create an addition operation
    pub fn add() -> Self {
        Self {
            op_type: OperationType::Add,
            grad_fn: Arc::new(grad_fn::AddGradFn),
        }
    }

    /// Create a subtraction operation
    pub fn subtract() -> Self {
        Self {
            op_type: OperationType::Subtract,
            grad_fn: Arc::new(grad_fn::SubtractGradFn),
        }
    }

    /// Create a multiplication operation
    pub fn multiply() -> Self {
        Self {
            op_type: OperationType::Multiply,
            grad_fn: Arc::new(grad_fn::MultiplyGradFn),
        }
    }

    /// Create a division operation
    pub fn divide() -> Self {
        Self {
            op_type: OperationType::Divide,
            grad_fn: Arc::new(grad_fn::DivideGradFn),
        }
    }

    /// Create a matrix multiplication operation
    pub fn matmul() -> Self {
        Self {
            op_type: OperationType::MatrixMultiply,
            grad_fn: Arc::new(grad_fn::MatMulGradFn),
        }
    }

    /// Create a sigmoid operation
    pub fn sigmoid() -> Self {
        Self {
            op_type: OperationType::Sigmoid,
            grad_fn: Arc::new(grad_fn::SigmoidGradFn),
        }
    }

    /// Create a tanh operation
    pub fn tanh() -> Self {
        Self {
            op_type: OperationType::Tanh,
            grad_fn: Arc::new(grad_fn::TanhGradFn),
        }
    }

    /// Create a ReLU operation
    pub fn relu() -> Self {
        Self {
            op_type: OperationType::ReLU,
            grad_fn: Arc::new(grad_fn::ReLUGradFn),
        }
    }

    /// Create a softmax operation
    pub fn softmax(axis: i32) -> Self {
        Self {
            op_type: OperationType::Softmax,
            grad_fn: Arc::new(grad_fn::SoftmaxGradFn::new(axis)),
        }
    }

    /// Create a sum operation
    pub fn sum(axes: Option<Vec<usize>>, original_shape: Vec<usize>) -> Self {
        Self {
            op_type: OperationType::Sum(axes.clone()),
            grad_fn: Arc::new(grad_fn::SumGradFn::new(axes, original_shape)),
        }
    }

    /// Create a mean operation
    pub fn mean(axes: Option<Vec<usize>>, original_shape: Vec<usize>) -> Self {
        Self {
            op_type: OperationType::Mean(axes.clone()),
            grad_fn: Arc::new(grad_fn::MeanGradFn::new(axes, original_shape)),
        }
    }

    /// Create a reshape operation
    pub fn reshape(original_shape: Vec<usize>, target_shape: Vec<usize>) -> Self {
        Self {
            op_type: OperationType::Reshape(target_shape),
            grad_fn: Arc::new(grad_fn::ReshapeGradFn::new(original_shape)),
        }
    }

    /// Create a transpose operation
    pub fn transpose(permutation: Vec<usize>) -> Self {
        Self {
            op_type: OperationType::Transpose(permutation.clone()),
            grad_fn: Arc::new(grad_fn::TransposeGradFn::new(permutation)),
        }
    }

    /// Create a layer normalization operation
    pub fn layer_norm(epsilon: f32) -> Self {
        Self {
            op_type: OperationType::LayerNorm(epsilon),
            grad_fn: Arc::new(grad_fn::LayerNormGradFn::new(epsilon)),
        }
    }

    /// Create an MSE loss operation
    pub fn mse_loss() -> Self {
        Self {
            op_type: OperationType::Custom("MSELoss".to_string()),
            grad_fn: Arc::new(grad_fn::MSELossGradFn),
        }
    }

    /// Create a cross entropy loss operation
    pub fn cross_entropy_loss() -> Self {
        Self {
            op_type: OperationType::Custom("CrossEntropyLoss".to_string()),
            grad_fn: Arc::new(grad_fn::CrossEntropyLossGradFn),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_add_gradient() {
        let grad_fn = grad_fn::AddGradFn;
        let grad_output = Tensor::ones(&[2, 2]).expect("Failed to create ones tensor");
        let input1 = Tensor::ones(&[2, 2]).expect("Failed to create ones tensor");
        let input2 = Tensor::ones(&[2, 2]).expect("Failed to create ones tensor");

        let gradients = grad_fn
            .backward(&grad_output, &[&input1, &input2])
            .expect("operation failed in test");

        assert_eq!(gradients.len(), 2);
        assert_eq!(gradients[0].shape(), vec![2, 2]);
        assert_eq!(gradients[1].shape(), vec![2, 2]);
    }

    #[test]
    fn test_multiply_gradient() {
        let grad_fn = grad_fn::MultiplyGradFn;
        let grad_output = Tensor::ones(&[2, 2]).expect("Failed to create ones tensor");
        let input1 = Tensor::scalar(2.0)
            .expect("tensor operation failed")
            .broadcast_to(&[2, 2])
            .expect("operation failed in test");
        let input2 = Tensor::scalar(3.0)
            .expect("tensor operation failed")
            .broadcast_to(&[2, 2])
            .expect("operation failed in test");

        let gradients = grad_fn
            .backward(&grad_output, &[&input1, &input2])
            .expect("operation failed in test");

        assert_eq!(gradients.len(), 2);
        // Gradient w.r.t. input1 should be input2 (3.0)
        assert_eq!(
            gradients[0].to_vec_f32().expect("operation failed in test")[0],
            3.0
        );
        // Gradient w.r.t. input2 should be input1 (2.0)
        assert_eq!(
            gradients[1].to_vec_f32().expect("operation failed in test")[0],
            2.0
        );
    }

    #[test]
    fn test_sigmoid_gradient() {
        let grad_fn = grad_fn::SigmoidGradFn;
        let grad_output = Tensor::ones(&[2, 2]).expect("Failed to create ones tensor");
        let input = Tensor::zeros(&[2, 2]).expect("Failed to create zero tensor");

        let gradients =
            grad_fn.backward(&grad_output, &[&input]).expect("operation failed in test");

        assert_eq!(gradients.len(), 1);
        // Gradient of sigmoid(0) = 0.5 * (1 - 0.5) = 0.25
        assert!(
            (gradients[0].to_vec_f32().expect("operation failed in test")[0] - 0.25).abs() < 1e-6
        );
    }

    #[test]
    fn test_relu_gradient() {
        let grad_fn = grad_fn::ReLUGradFn;
        let grad_output = Tensor::ones(&[2]).expect("Failed to create ones tensor");
        let input = Tensor::from_vec(vec![1.0, -1.0], &[2]).expect("Tensor from_vec failed");

        let gradients =
            grad_fn.backward(&grad_output, &[&input]).expect("operation failed in test");

        assert_eq!(gradients.len(), 1);
        let grad_values = gradients[0].to_vec_f32().expect("operation failed in test");
        assert_eq!(grad_values[0], 1.0); // Positive input
        assert_eq!(grad_values[1], 0.0); // Negative input
    }

    #[test]
    fn test_sum_gradient() {
        let original_shape = vec![2, 3];
        let grad_fn = grad_fn::SumGradFn::new(None, original_shape.clone());
        let grad_output = Tensor::scalar(1.0).expect("tensor operation failed");
        let input = Tensor::ones(&original_shape).expect("tensor operation failed");

        let gradients =
            grad_fn.backward(&grad_output, &[&input]).expect("operation failed in test");

        assert_eq!(gradients.len(), 1);
        assert_eq!(gradients[0].shape(), original_shape);
    }

    #[test]
    fn test_mean_gradient() {
        let original_shape = vec![2, 3];
        let grad_fn = grad_fn::MeanGradFn::new(None, original_shape.clone());
        let grad_output = Tensor::scalar(1.0).expect("tensor operation failed");
        let input = Tensor::ones(&original_shape).expect("tensor operation failed");

        let gradients =
            grad_fn.backward(&grad_output, &[&input]).expect("operation failed in test");

        assert_eq!(gradients.len(), 1);
        assert_eq!(gradients[0].shape(), original_shape);
        // Gradient should be 1/N where N is the number of elements
        let expected_grad = 1.0 / (2.0 * 3.0);
        assert!(
            (gradients[0].to_vec_f32().expect("operation failed in test")[0] - expected_grad).abs()
                < 1e-6
        );
    }

    #[test]
    fn test_reshape_gradient() {
        let original_shape = vec![2, 3];
        let grad_fn = grad_fn::ReshapeGradFn::new(original_shape.clone());
        let grad_output = Tensor::ones(&[6]).expect("Failed to create ones tensor");
        let input = Tensor::ones(&original_shape).expect("tensor operation failed");

        let gradients =
            grad_fn.backward(&grad_output, &[&input]).expect("operation failed in test");

        assert_eq!(gradients.len(), 1);
        assert_eq!(gradients[0].shape(), original_shape);
    }

    #[test]
    fn test_transpose_gradient() {
        let permutation = vec![1, 0];
        let grad_fn = grad_fn::TransposeGradFn::new(permutation);
        let grad_output = Tensor::ones(&[3, 2]).expect("Failed to create ones tensor");
        let input = Tensor::ones(&[2, 3]).expect("Failed to create ones tensor");

        let gradients =
            grad_fn.backward(&grad_output, &[&input]).expect("operation failed in test");

        assert_eq!(gradients.len(), 1);
        assert_eq!(gradients[0].shape(), vec![2, 3]);
    }

    /// Regression test for the rank > 2 transpose gradient.
    ///
    /// The previous implementation applied only `inverse[0]`/`inverse[1]` as a
    /// single axis swap, so a 3-cycle permutation produced a gradient with the
    /// wrong *shape* (here `[2,4,3]` instead of `[2,3,4]`) and, whenever the
    /// shapes happened to coincide, silently wrong values.
    #[test]
    fn transpose_gradient_handles_a_general_nd_permutation() {
        // Forward permutation [1, 2, 0]: input [2,3,4] -> output [3,4,2].
        let grad_fn = grad_fn::TransposeGradFn::new(vec![1, 2, 0]);
        // Give every element a distinct value so a wrong permutation is visible.
        let values: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let grad_output = Tensor::from_vec(values.clone(), &[3, 4, 2]).expect("build grad_output");
        let input = Tensor::zeros(&[2, 3, 4]).expect("build input");

        let gradients = grad_fn.backward(&grad_output, &[&input]).expect("transpose backward");
        assert_eq!(gradients.len(), 1);
        assert_eq!(gradients[0].shape(), vec![2, 3, 4]);

        // Inverse of [1,2,0] is [2,0,1]: grad_input[i,j,k] == grad_output[j,k,i].
        let actual = gradients[0].to_vec_f32().expect("grad values");
        let mut expected = vec![0.0f32; 24];
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    // grad_output is [3,4,2] row-major.
                    expected[i * 12 + j * 4 + k] = values[j * 8 + k * 2 + i];
                }
            }
        }
        assert_eq!(actual, expected);
    }

    /// An empty permutation means "reverse every axis" and is its own inverse.
    #[test]
    fn transpose_gradient_reverses_all_axes_for_an_empty_permutation() {
        let grad_fn = grad_fn::TransposeGradFn::new(Vec::new());
        let values: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let grad_output = Tensor::from_vec(values.clone(), &[4, 3, 2]).expect("grad_output");
        let input = Tensor::zeros(&[2, 3, 4]).expect("input");

        let gradients = grad_fn.backward(&grad_output, &[&input]).expect("backward");
        assert_eq!(gradients[0].shape(), vec![2, 3, 4]);

        let actual = gradients[0].to_vec_f32().expect("grad values");
        let mut expected = vec![0.0f32; 24];
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    // grad_output is [4,3,2] row-major, reversed indexing.
                    expected[i * 12 + j * 4 + k] = values[k * 6 + j * 2 + i];
                }
            }
        }
        assert_eq!(actual, expected);
    }

    /// A permutation whose length disagrees with the gradient rank is a caller
    /// error, not something to silently apply to the first two axes.
    #[test]
    fn transpose_gradient_rejects_a_mismatched_permutation_rank() {
        let grad_fn = grad_fn::TransposeGradFn::new(vec![1, 0]);
        let grad_output = Tensor::zeros(&[3, 4, 2]).expect("grad_output");
        let input = Tensor::zeros(&[2, 3, 4]).expect("input");
        assert!(grad_fn.backward(&grad_output, &[&input]).is_err());
    }

    /// A repeated index is not a permutation; it used to leave an uninitialised
    /// slot in the inverse and silently produce a wrong axis order.
    #[test]
    fn transpose_gradient_rejects_a_duplicated_permutation_index() {
        let grad_fn = grad_fn::TransposeGradFn::new(vec![1, 1, 0]);
        let grad_output = Tensor::zeros(&[3, 4, 2]).expect("grad_output");
        let input = Tensor::zeros(&[2, 3, 4]).expect("input");
        assert!(grad_fn.backward(&grad_output, &[&input]).is_err());
    }

    /// Reference LayerNorm forward used by the finite-difference check.
    ///
    /// `x` is `[rows, n]` flattened row-major; `gamma`/`beta` have `n` entries.
    fn reference_layer_norm(x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
        let n = gamma.len();
        let rows = x.len() / n;
        let mut out = vec![0.0f32; x.len()];
        for row in 0..rows {
            let offset = row * n;
            let slice = &x[offset..offset + n];
            let mean = slice.iter().sum::<f32>() / n as f32;
            let var = slice.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
            let rstd = 1.0 / (var + eps).sqrt();
            for i in 0..n {
                out[offset + i] = gamma[i] * (slice[i] - mean) * rstd + beta[i];
            }
        }
        out
    }

    /// Scalar loss used by the finite-difference check: `sum(w_i * y_i)`.
    ///
    /// Its gradient with respect to `y` is exactly `w`, which is what we feed in
    /// as `grad_output`, so the analytic and numerical gradients are comparable.
    fn weighted_loss(y: &[f32], weights: &[f32]) -> f64 {
        y.iter().zip(weights.iter()).map(|(&a, &b)| a as f64 * b as f64).sum()
    }

    /// Finite-difference gradient check for the LayerNorm backward pass.
    ///
    /// The previous implementation returned `grad_output * weight` for the input
    /// gradient (no 1/sigma factor, no mean-subtraction terms), `grad_output *
    /// input` for the weight gradient and an unreduced `grad_output` for the bias
    /// gradient. All three disagree with the numerical gradient below.
    #[test]
    fn test_layer_norm_gradient_matches_finite_differences() {
        let eps = 1e-5f32;
        let shape = [2usize, 3usize];
        let n = shape[1];
        let x = vec![0.5f32, -1.25, 2.0, 0.75, 0.1, -0.6];
        let gamma = vec![1.3f32, 0.7, -0.4];
        let beta = vec![0.05f32, -0.2, 0.4];
        // Upstream gradient (also the weights of the scalar loss).
        let g_out = vec![0.9f32, -0.3, 0.6, 0.2, 1.1, -0.8];

        let input = Tensor::from_vec(x.clone(), &shape).expect("input tensor");
        let weight = Tensor::from_vec(gamma.clone(), &[n]).expect("weight tensor");
        let bias = Tensor::from_vec(beta.clone(), &[n]).expect("bias tensor");
        let grad_output = Tensor::from_vec(g_out.clone(), &shape).expect("grad tensor");

        let grads = grad_fn::LayerNormGradFn::new(eps)
            .backward(&grad_output, &[&input, &weight, &bias])
            .expect("layer norm backward");
        assert_eq!(grads.len(), 3);
        assert_eq!(grads[0].shape(), shape.to_vec());
        assert_eq!(grads[1].shape(), vec![n]);
        assert_eq!(grads[2].shape(), vec![n]);

        let grad_input = grads[0].to_vec_f32().expect("grad input values");
        let grad_weight = grads[1].to_vec_f32().expect("grad weight values");
        let grad_bias = grads[2].to_vec_f32().expect("grad bias values");

        // Central differences in f64 accumulation; h = 1e-3 keeps the f32 forward
        // pass well above its rounding floor while staying inside the quadratic
        // truncation regime.
        let h = 1e-3f32;
        let tol = 2e-2f64;

        for i in 0..x.len() {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[i] += h;
            x_minus[i] -= h;
            let loss_plus =
                weighted_loss(&reference_layer_norm(&x_plus, &gamma, &beta, eps), &g_out);
            let loss_minus =
                weighted_loss(&reference_layer_norm(&x_minus, &gamma, &beta, eps), &g_out);
            let numeric = (loss_plus - loss_minus) / (2.0 * h as f64);
            let analytic = grad_input[i] as f64;
            assert!(
                (numeric - analytic).abs() <= tol * (1.0 + numeric.abs()),
                "grad_input[{i}]: analytic {analytic}, numeric {numeric}"
            );
        }

        for i in 0..n {
            let mut gamma_plus = gamma.clone();
            let mut gamma_minus = gamma.clone();
            gamma_plus[i] += h;
            gamma_minus[i] -= h;
            let numeric =
                (weighted_loss(&reference_layer_norm(&x, &gamma_plus, &beta, eps), &g_out)
                    - weighted_loss(&reference_layer_norm(&x, &gamma_minus, &beta, eps), &g_out))
                    / (2.0 * h as f64);
            assert!(
                (numeric - grad_weight[i] as f64).abs() <= tol * (1.0 + numeric.abs()),
                "grad_weight[{i}]: analytic {}, numeric {numeric}",
                grad_weight[i]
            );

            let mut beta_plus = beta.clone();
            let mut beta_minus = beta.clone();
            beta_plus[i] += h;
            beta_minus[i] -= h;
            let numeric_bias =
                (weighted_loss(&reference_layer_norm(&x, &gamma, &beta_plus, eps), &g_out)
                    - weighted_loss(&reference_layer_norm(&x, &gamma, &beta_minus, eps), &g_out))
                    / (2.0 * h as f64);
            assert!(
                (numeric_bias - grad_bias[i] as f64).abs() <= tol * (1.0 + numeric_bias.abs()),
                "grad_bias[{i}]: analytic {}, numeric {numeric_bias}",
                grad_bias[i]
            );
        }
    }

    /// The input gradient of LayerNorm is orthogonal to both `1` and `xhat`
    /// within each normalized group -- a property the old implementation
    /// (`grad_output * weight`) violated for any non-zero upstream gradient.
    #[test]
    fn test_layer_norm_grad_input_is_mean_free() {
        let eps = 1e-5f32;
        let input = Tensor::from_vec(vec![0.5f32, -1.25, 2.0, 0.75, 0.1, -0.6], &[2, 3])
            .expect("input tensor");
        let weight = Tensor::from_vec(vec![1.3f32, 0.7, -0.4], &[3]).expect("weight tensor");
        let bias = Tensor::from_vec(vec![0.05f32, -0.2, 0.4], &[3]).expect("bias tensor");
        let grad_output = Tensor::from_vec(vec![0.9f32, -0.3, 0.6, 0.2, 1.1, -0.8], &[2, 3])
            .expect("grad tensor");

        let grads = grad_fn::LayerNormGradFn::new(eps)
            .backward(&grad_output, &[&input, &weight, &bias])
            .expect("layer norm backward");
        let grad_input = grads[0].to_vec_f32().expect("grad input values");

        for row in grad_input.chunks(3) {
            let sum: f32 = row.iter().sum();
            assert!(sum.abs() < 1e-5, "row gradient must sum to ~0, got {sum}");
        }
    }

    /// Shape mismatches must be reported, not silently broadcast.
    #[test]
    fn test_layer_norm_backward_rejects_shape_mismatch() {
        let input = Tensor::ones(&[2, 3]).expect("input tensor");
        let weight = Tensor::ones(&[4]).expect("weight tensor");
        let bias = Tensor::ones(&[4]).expect("bias tensor");
        let grad_output = Tensor::ones(&[2, 3]).expect("grad tensor");

        let result =
            grad_fn::LayerNormGradFn::new(1e-5).backward(&grad_output, &[&input, &weight, &bias]);
        assert!(
            result.is_err(),
            "weight shape must be a suffix of the input shape"
        );
    }
}
