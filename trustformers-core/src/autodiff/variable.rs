//! Variable implementation for automatic differentiation.
//!
//! This module provides the Variable type, which wraps tensors and enables
//! automatic gradient computation through the computational graph.

use super::graph::{ComputationGraph, NodeId, OperationType};
use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use std::sync::{Arc, Mutex};

/// Reference to a shared computation graph
pub type GraphRef = Arc<Mutex<ComputationGraph>>;

/// Variable that participates in automatic differentiation
#[derive(Debug, Clone)]
pub struct Variable {
    /// Reference to the computation graph
    graph: GraphRef,
    /// Node ID in the computation graph
    node_id: NodeId,
    /// Whether this variable requires gradients
    requires_grad: bool,
}

/// Shared reference to a variable
pub type VariableRef = Arc<Variable>;

impl Variable {
    /// Create a new variable from a tensor
    pub fn new(tensor: Tensor, requires_grad: bool) -> Self {
        let graph = Arc::new(Mutex::new(ComputationGraph::new()));
        let node_id = {
            let mut graph_guard = graph.lock().unwrap();
            graph_guard.add_node(tensor, requires_grad, None)
        };

        Self {
            graph,
            node_id,
            requires_grad,
        }
    }

    /// Create a new variable with a name
    pub fn new_with_name(tensor: Tensor, requires_grad: bool, name: String) -> Self {
        let graph = Arc::new(Mutex::new(ComputationGraph::new()));
        let node_id = {
            let mut graph_guard = graph.lock().unwrap();
            graph_guard.add_node(tensor, requires_grad, Some(name))
        };

        Self {
            graph,
            node_id,
            requires_grad,
        }
    }

    /// Create a new variable from an existing graph
    pub fn from_graph(graph: GraphRef, node_id: NodeId, requires_grad: bool) -> Self {
        Self {
            graph,
            node_id,
            requires_grad,
        }
    }

    /// Get the tensor data
    pub fn data(&self) -> Result<Tensor> {
        let graph = self.graph.lock().unwrap();
        graph.get_value(self.node_id).cloned().ok_or_else(|| {
            TrustformersError::tensor_op_error(
                &format!("Node {} not found in graph", self.node_id),
                "Variable::data",
            )
        })
    }

    /// Get the gradient
    pub fn grad(&self) -> Result<Option<Tensor>> {
        let graph = self.graph.lock().unwrap();
        Ok(graph.get_gradient(self.node_id).cloned())
    }

    /// Get the node ID
    pub fn node_id(&self) -> NodeId {
        self.node_id
    }

    /// Check if this variable requires gradients
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Get the graph reference
    pub fn graph(&self) -> GraphRef {
        self.graph.clone()
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> Result<Vec<usize>> {
        let graph = self.graph.lock().unwrap();
        graph.get_value(self.node_id).map(|tensor| tensor.shape()).ok_or_else(|| {
            TrustformersError::tensor_op_error(
                &format!("Node {} not found in graph", self.node_id),
                "Variable::shape",
            )
        })
    }

    /// Convert to a scalar value
    pub fn item(&self) -> Result<f32> {
        let tensor = self.data()?;
        tensor.to_scalar()
    }

    /// Compute backward pass for this variable
    pub fn backward(&self) -> Result<()> {
        let mut graph = self.graph.lock().unwrap();
        graph.backward(self.node_id, None)
    }

    /// Compute backward pass with custom gradient
    pub fn backward_with_grad(&self, grad: Tensor) -> Result<()> {
        let mut graph = self.graph.lock().unwrap();
        graph.backward(self.node_id, Some(grad))
    }

    /// Zero the gradients
    pub fn zero_grad(&self) {
        let mut graph = self.graph.lock().unwrap();
        graph.zero_grad();
    }

    /// Detach this variable from the computation graph
    pub fn detach(&self) -> Result<Variable> {
        let tensor = self.data()?;
        Ok(Variable::new(tensor, false))
    }

    /// Create a copy of this variable that requires gradients
    pub fn requires_grad_(&self) -> Result<Variable> {
        let tensor = self.data()?;
        Ok(Variable::new(tensor, true))
    }

    /// Update the value of this variable
    pub fn set_data(&self, tensor: Tensor) -> Result<()> {
        let mut graph = self.graph.lock().unwrap();
        graph.update_value(self.node_id, tensor)
    }

    // Arithmetic operations

    /// Add another variable
    pub fn add(&self, other: &Variable) -> Result<Variable> {
        self.binary_op(other, OperationType::Add)
    }

    /// Subtract another variable
    pub fn sub(&self, other: &Variable) -> Result<Variable> {
        self.binary_op(other, OperationType::Subtract)
    }

    /// Multiply by another variable
    pub fn mul(&self, other: &Variable) -> Result<Variable> {
        self.binary_op(other, OperationType::Multiply)
    }

    /// Divide by another variable
    pub fn div(&self, other: &Variable) -> Result<Variable> {
        self.binary_op(other, OperationType::Divide)
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &Variable) -> Result<Variable> {
        self.binary_op(other, OperationType::MatrixMultiply)
    }

    /// Negation
    pub fn neg(&self) -> Result<Variable> {
        self.unary_op(OperationType::Negate)
    }

    /// Square
    pub fn square(&self) -> Result<Variable> {
        self.unary_op(OperationType::Square)
    }

    /// Square root
    pub fn sqrt(&self) -> Result<Variable> {
        self.unary_op(OperationType::Sqrt)
    }

    /// Natural logarithm
    pub fn log(&self) -> Result<Variable> {
        self.unary_op(OperationType::Log)
    }

    /// Exponential
    pub fn exp(&self) -> Result<Variable> {
        self.unary_op(OperationType::Exp)
    }

    // Activation functions

    /// Sigmoid activation
    pub fn sigmoid(&self) -> Result<Variable> {
        self.unary_op(OperationType::Sigmoid)
    }

    /// Tanh activation
    pub fn tanh(&self) -> Result<Variable> {
        self.unary_op(OperationType::Tanh)
    }

    /// ReLU activation
    pub fn relu(&self) -> Result<Variable> {
        self.unary_op(OperationType::ReLU)
    }

    /// Leaky ReLU activation
    pub fn leaky_relu(&self, alpha: f32) -> Result<Variable> {
        self.unary_op(OperationType::LeakyReLU(alpha))
    }

    /// Softmax activation
    pub fn softmax(&self) -> Result<Variable> {
        self.unary_op(OperationType::Softmax)
    }

    // Tensor operations

    /// Reshape the tensor
    pub fn reshape(&self, shape: Vec<usize>) -> Result<Variable> {
        self.unary_op(OperationType::Reshape(shape))
    }

    /// Transpose the tensor
    pub fn transpose(&self, permutation: Vec<usize>) -> Result<Variable> {
        self.unary_op(OperationType::Transpose(permutation))
    }

    /// Sum along specified axes
    pub fn sum(&self, axes: Option<Vec<usize>>) -> Result<Variable> {
        self.unary_op(OperationType::Sum(axes))
    }

    /// Mean along specified axes
    pub fn mean(&self, axes: Option<Vec<usize>>) -> Result<Variable> {
        self.unary_op(OperationType::Mean(axes))
    }

    /// Max along specified axes
    pub fn max(&self, axes: Option<Vec<usize>>) -> Result<Variable> {
        self.unary_op(OperationType::Max(axes))
    }

    /// Min along specified axes
    pub fn min(&self, axes: Option<Vec<usize>>) -> Result<Variable> {
        self.unary_op(OperationType::Min(axes))
    }

    // Scalar operations

    /// Add a scalar
    pub fn add_scalar(&self, scalar: f32) -> Result<Variable> {
        let scalar_tensor = Tensor::scalar(scalar)?;
        let scalar_var = Variable::new(scalar_tensor, false);
        self.add(&scalar_var)
    }

    /// Subtract a scalar
    pub fn sub_scalar(&self, scalar: f32) -> Result<Variable> {
        let scalar_tensor = Tensor::scalar(scalar)?;
        let scalar_var = Variable::new(scalar_tensor, false);
        self.sub(&scalar_var)
    }

    /// Multiply by a scalar
    pub fn mul_scalar(&self, scalar: f32) -> Result<Variable> {
        let scalar_tensor = Tensor::scalar(scalar)?;
        let scalar_var = Variable::new(scalar_tensor, false);
        self.mul(&scalar_var)
    }

    /// Divide by a scalar
    pub fn div_scalar(&self, scalar: f32) -> Result<Variable> {
        let scalar_tensor = Tensor::scalar(scalar)?;
        let scalar_var = Variable::new(scalar_tensor, false);
        self.div(&scalar_var)
    }

    // Helper methods for operations

    /// Binary operation helper
    fn binary_op(&self, other: &Variable, op: OperationType) -> Result<Variable> {
        // Check if both variables are from the same graph
        if !Arc::ptr_eq(&self.graph, &other.graph) {
            return Err(TrustformersError::tensor_op_error(
                "Variables must be from the same computation graph",
                "Variable::binary_op",
            ));
        }

        // Compute the operation on the tensor data
        let result_tensor = self.compute_binary_tensor_op(&other.data()?, &op)?;

        // Add operation node to the graph
        let requires_grad = self.requires_grad || other.requires_grad;
        let node_id = {
            let mut graph = self.graph.lock().unwrap();
            graph.add_operation_node(
                result_tensor,
                op,
                vec![self.node_id, other.node_id],
                requires_grad,
                None,
            )?
        };

        Ok(Variable::from_graph(
            self.graph.clone(),
            node_id,
            requires_grad,
        ))
    }

    /// Unary operation helper
    fn unary_op(&self, op: OperationType) -> Result<Variable> {
        // Compute the operation on the tensor data
        let result_tensor = self.compute_unary_tensor_op(&op)?;

        // Add operation node to the graph
        let node_id = {
            let mut graph = self.graph.lock().unwrap();
            graph.add_operation_node(
                result_tensor,
                op,
                vec![self.node_id],
                self.requires_grad,
                None,
            )?
        };

        Ok(Variable::from_graph(
            self.graph.clone(),
            node_id,
            self.requires_grad,
        ))
    }

    /// Compute binary tensor operation
    fn compute_binary_tensor_op(&self, other: &Tensor, op: &OperationType) -> Result<Tensor> {
        let self_tensor = self.data()?;

        match op {
            OperationType::Add => Tensor::add(&self_tensor, other),
            OperationType::Subtract => Tensor::sub(&self_tensor, other),
            OperationType::Multiply => self_tensor.mul(other),
            OperationType::Divide => Tensor::div(&self_tensor, other),
            OperationType::MatrixMultiply => self_tensor.matmul(other),
            _ => Err(TrustformersError::tensor_op_error(
                &format!("Unsupported binary operation: {:?}", op),
                "Variable::compute_binary_tensor_op",
            )),
        }
    }

    /// Compute unary tensor operation
    fn compute_unary_tensor_op(&self, op: &OperationType) -> Result<Tensor> {
        let self_tensor = self.data()?;

        match op {
            OperationType::Negate => self_tensor.neg(),
            OperationType::Square => self_tensor.clone().mul(&self_tensor),
            OperationType::Sqrt => self_tensor.sqrt(),
            OperationType::Log => self_tensor.log(),
            OperationType::Exp => self_tensor.exp(),
            OperationType::Sigmoid => self_tensor.sigmoid(),
            OperationType::Tanh => self_tensor.tanh(),
            OperationType::ReLU => self_tensor.relu(),
            OperationType::LeakyReLU(alpha) => self_tensor.leaky_relu(*alpha),
            OperationType::Softmax => self_tensor.softmax(-1),
            OperationType::Reshape(shape) => self_tensor.reshape(shape),
            OperationType::Transpose(permutation) => {
                // For now, handle simple 2D transpose case
                if permutation.len() >= 2 {
                    self_tensor.transpose(permutation[0], permutation[1])
                } else {
                    // Default transpose for 2D case
                    self_tensor.transpose(0, 1)
                }
            },
            OperationType::Sum(axes) => {
                match axes {
                    Some(axes_vec) => self_tensor.sum_axes(axes_vec),
                    None => {
                        // Sum all elements (global sum)
                        let shape = self_tensor.shape();
                        let all_axes: Vec<usize> = (0..shape.len()).collect();
                        self_tensor.sum_axes(&all_axes)
                    },
                }
            },
            OperationType::Mean(_axes) => {
                // For now, just compute global mean
                self_tensor.mean()
            },
            _ => Err(TrustformersError::tensor_op_error(
                &format!("Unsupported unary operation: {:?}", op),
                "Variable::compute_unary_tensor_op",
            )),
        }
    }

    /// Set whether this variable requires gradients
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
        // Also update the node in the graph
        if let Ok(mut graph) = self.graph.lock() {
            if let Some(node) = graph.get_node_mut(self.node_id) {
                node.requires_grad = requires_grad;
            }
        }
    }

    /// Create a variable from a tensor (with requires_grad = false by default)
    pub fn from_tensor(tensor: Tensor) -> Self {
        Variable::new(tensor, false)
    }
}

/// Convenience functions for creating variables
impl Variable {
    /// Create a variable from a scalar
    pub fn scalar(value: f32, requires_grad: bool) -> Result<Self> {
        let tensor = Tensor::scalar(value)?;
        Ok(Variable::new(tensor, requires_grad))
    }

    /// Create a variable with zeros
    pub fn zeros(shape: &[usize], requires_grad: bool) -> Result<Self> {
        let tensor = Tensor::zeros(shape)?;
        Ok(Variable::new(tensor, requires_grad))
    }

    /// Create a variable with ones
    pub fn ones(shape: &[usize], requires_grad: bool) -> Result<Self> {
        let tensor = Tensor::ones(shape)?;
        Ok(Variable::new(tensor, requires_grad))
    }

    /// Create a variable with random normal distribution
    pub fn randn(shape: &[usize], requires_grad: bool) -> Result<Self> {
        let tensor = Tensor::randn(shape)?;
        Ok(Variable::new(tensor, requires_grad))
    }

    /// Create a variable with random uniform distribution
    pub fn rand(shape: &[usize], requires_grad: bool) -> Result<Self> {
        let tensor = Tensor::randn(shape)?;
        Ok(Variable::new(tensor, requires_grad))
    }
}

/// Operator overloading for Variables
use std::ops::{Add, Div, Mul, Neg, Sub};

impl Add for &Variable {
    type Output = Result<Variable>;

    fn add(self, rhs: Self) -> Self::Output {
        self.add(rhs)
    }
}

impl Sub for &Variable {
    type Output = Result<Variable>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(rhs)
    }
}

impl Mul for &Variable {
    type Output = Result<Variable>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(rhs)
    }
}

impl Div for &Variable {
    type Output = Result<Variable>;

    fn div(self, rhs: Self) -> Self::Output {
        self.div(rhs)
    }
}

impl Neg for &Variable {
    type Output = Result<Variable>;

    fn neg(self) -> Self::Output {
        self.neg()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_variable_creation() {
        let tensor = Tensor::ones(&[2, 3]).unwrap();
        let var = Variable::new(tensor, true);

        assert!(var.requires_grad());
        assert_eq!(var.shape().unwrap(), vec![2, 3]);
    }

    #[test]
    fn test_variable_operations() {
        use super::super::AutodiffEngine;
        use std::sync::Arc;

        let engine = Arc::new(AutodiffEngine::default());
        let a = engine.variable(Tensor::scalar(2.0).unwrap(), true);
        let b = engine.variable(Tensor::scalar(3.0).unwrap(), true);

        let c = a.add(&b).unwrap();
        assert_eq!(c.item().unwrap(), 5.0);

        let d = a.mul(&b).unwrap();
        assert_eq!(d.item().unwrap(), 6.0);
    }

    #[test]
    fn test_gradient_computation() {
        use super::super::AutodiffEngine;
        use std::sync::Arc;

        let engine = Arc::new(AutodiffEngine::default());
        let a = engine.variable(Tensor::scalar(2.0).unwrap(), true);
        let b = engine.variable(Tensor::scalar(3.0).unwrap(), true);

        let c = a.mul(&b).unwrap();
        engine.backward(&c, None).unwrap();

        let grad_a = engine.get_grad(&a).unwrap().unwrap();
        let grad_b = engine.get_grad(&b).unwrap().unwrap();

        assert_eq!(grad_a.to_scalar().unwrap(), 3.0);
        assert_eq!(grad_b.to_scalar().unwrap(), 2.0);
    }

    #[test]
    fn test_activation_functions() {
        let x = Variable::scalar(0.0, true).unwrap();

        let sigmoid_x = x.sigmoid().unwrap();
        assert_eq!(sigmoid_x.item().unwrap(), 0.5);

        let tanh_x = x.tanh().unwrap();
        assert_eq!(tanh_x.item().unwrap(), 0.0);
    }

    #[test]
    fn test_tensor_operations() {
        let x = Variable::ones(&[2, 3], true).unwrap();

        let sum_x = x.sum(None).unwrap();
        assert_eq!(sum_x.item().unwrap(), 6.0);

        let mean_x = x.mean(None).unwrap();
        assert_eq!(mean_x.item().unwrap(), 1.0);
    }

    #[test]
    fn test_reshape_operation() {
        let x = Variable::ones(&[2, 3], true).unwrap();
        let reshaped = x.reshape(vec![3, 2]).unwrap();

        assert_eq!(reshaped.shape().unwrap(), vec![3, 2]);
    }

    #[test]
    fn test_detach_operation() {
        let x = Variable::scalar(2.0, true).unwrap();
        let y = x.detach().unwrap();

        assert!(x.requires_grad());
        assert!(!y.requires_grad());
    }
}
