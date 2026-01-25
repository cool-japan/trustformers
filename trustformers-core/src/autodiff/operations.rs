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

    /// Transpose gradient function
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

            // Gradient of transpose: apply inverse permutation
            let inverse_permutation = self.compute_inverse_permutation()?;
            // For now, handle simple 2D transpose case
            let grad_input = if inverse_permutation.len() >= 2 {
                grad_output.transpose(inverse_permutation[0], inverse_permutation[1])?
            } else {
                grad_output.transpose(0, 1)?
            };

            Ok(vec![grad_input])
        }

        fn operation_type(&self) -> OperationType {
            OperationType::Transpose(self.permutation.clone())
        }
    }

    impl TransposeGradFn {
        fn compute_inverse_permutation(&self) -> Result<Vec<usize>> {
            let mut inverse = vec![0; self.permutation.len()];
            for (i, &p) in self.permutation.iter().enumerate() {
                if p >= self.permutation.len() {
                    return Err(TrustformersError::tensor_op_error(
                        &format!("Invalid permutation index: {}", p),
                        "TransposeGradFn::compute_inverse_permutation",
                    ));
                }
                inverse[p] = i;
            }
            Ok(inverse)
        }
    }

    /// Layer normalization gradient function
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

            let input = inputs[0];
            let weight = inputs[1];
            let bias = inputs[2];

            // This is a simplified implementation
            // In practice, you would compute the exact gradients for layer normalization
            let grad_input = grad_output.mul(weight)?;
            let grad_weight = grad_output.mul(input)?;
            let grad_bias = grad_output.clone();

            Ok(vec![grad_input, grad_weight, grad_bias])
        }

        fn operation_type(&self) -> OperationType {
            OperationType::LayerNorm(self.epsilon)
        }
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

        let gradients = grad_fn.backward(&grad_output, &[&input1, &input2]).unwrap();

        assert_eq!(gradients.len(), 2);
        assert_eq!(gradients[0].shape(), vec![2, 2]);
        assert_eq!(gradients[1].shape(), vec![2, 2]);
    }

    #[test]
    fn test_multiply_gradient() {
        let grad_fn = grad_fn::MultiplyGradFn;
        let grad_output = Tensor::ones(&[2, 2]).expect("Failed to create ones tensor");
        let input1 = Tensor::scalar(2.0).unwrap().broadcast_to(&[2, 2]).unwrap();
        let input2 = Tensor::scalar(3.0).unwrap().broadcast_to(&[2, 2]).unwrap();

        let gradients = grad_fn.backward(&grad_output, &[&input1, &input2]).unwrap();

        assert_eq!(gradients.len(), 2);
        // Gradient w.r.t. input1 should be input2 (3.0)
        assert_eq!(gradients[0].to_vec_f32().unwrap()[0], 3.0);
        // Gradient w.r.t. input2 should be input1 (2.0)
        assert_eq!(gradients[1].to_vec_f32().unwrap()[0], 2.0);
    }

    #[test]
    fn test_sigmoid_gradient() {
        let grad_fn = grad_fn::SigmoidGradFn;
        let grad_output = Tensor::ones(&[2, 2]).expect("Failed to create ones tensor");
        let input = Tensor::zeros(&[2, 2]).expect("Failed to create zero tensor");

        let gradients = grad_fn.backward(&grad_output, &[&input]).unwrap();

        assert_eq!(gradients.len(), 1);
        // Gradient of sigmoid(0) = 0.5 * (1 - 0.5) = 0.25
        assert!((gradients[0].to_vec_f32().unwrap()[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_relu_gradient() {
        let grad_fn = grad_fn::ReLUGradFn;
        let grad_output = Tensor::ones(&[2]).expect("Failed to create ones tensor");
        let input = Tensor::from_vec(vec![1.0, -1.0], &[2]).expect("Tensor from_vec failed");

        let gradients = grad_fn.backward(&grad_output, &[&input]).unwrap();

        assert_eq!(gradients.len(), 1);
        let grad_values = gradients[0].to_vec_f32().unwrap();
        assert_eq!(grad_values[0], 1.0); // Positive input
        assert_eq!(grad_values[1], 0.0); // Negative input
    }

    #[test]
    fn test_sum_gradient() {
        let original_shape = vec![2, 3];
        let grad_fn = grad_fn::SumGradFn::new(None, original_shape.clone());
        let grad_output = Tensor::scalar(1.0).unwrap();
        let input = Tensor::ones(&original_shape).unwrap();

        let gradients = grad_fn.backward(&grad_output, &[&input]).unwrap();

        assert_eq!(gradients.len(), 1);
        assert_eq!(gradients[0].shape(), original_shape);
    }

    #[test]
    fn test_mean_gradient() {
        let original_shape = vec![2, 3];
        let grad_fn = grad_fn::MeanGradFn::new(None, original_shape.clone());
        let grad_output = Tensor::scalar(1.0).unwrap();
        let input = Tensor::ones(&original_shape).unwrap();

        let gradients = grad_fn.backward(&grad_output, &[&input]).unwrap();

        assert_eq!(gradients.len(), 1);
        assert_eq!(gradients[0].shape(), original_shape);
        // Gradient should be 1/N where N is the number of elements
        let expected_grad = 1.0 / (2.0 * 3.0);
        assert!((gradients[0].to_vec_f32().unwrap()[0] - expected_grad).abs() < 1e-6);
    }

    #[test]
    fn test_reshape_gradient() {
        let original_shape = vec![2, 3];
        let grad_fn = grad_fn::ReshapeGradFn::new(original_shape.clone());
        let grad_output = Tensor::ones(&[6]).expect("Failed to create ones tensor");
        let input = Tensor::ones(&original_shape).unwrap();

        let gradients = grad_fn.backward(&grad_output, &[&input]).unwrap();

        assert_eq!(gradients.len(), 1);
        assert_eq!(gradients[0].shape(), original_shape);
    }

    #[test]
    fn test_transpose_gradient() {
        let permutation = vec![1, 0];
        let grad_fn = grad_fn::TransposeGradFn::new(permutation);
        let grad_output = Tensor::ones(&[3, 2]).expect("Failed to create ones tensor");
        let input = Tensor::ones(&[2, 3]).expect("Failed to create ones tensor");

        let gradients = grad_fn.backward(&grad_output, &[&input]).unwrap();

        assert_eq!(gradients.len(), 1);
        assert_eq!(gradients[0].shape(), vec![2, 3]);
    }
}
