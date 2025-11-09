//! Computational graph for automatic differentiation.
//!
//! This module provides the computational graph infrastructure for tracking
//! operations and computing gradients through reverse-mode automatic differentiation.

#![allow(unused_variables)] // Autodiff graph

use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Unique identifier for nodes in the computation graph
pub type NodeId = usize;

/// Computational graph for tracking operations and gradients
#[derive(Debug)]
pub struct ComputationGraph {
    /// Nodes in the computation graph
    nodes: HashMap<NodeId, GraphNode>,
    /// Next available node ID
    next_id: NodeId,
    /// Topological ordering of nodes
    topological_order: Vec<NodeId>,
    /// Whether the graph is dirty and needs recomputation
    dirty: bool,
    /// Root nodes (variables with no parents)
    root_nodes: Vec<NodeId>,
    /// Leaf nodes (outputs that gradients flow back from)
    leaf_nodes: Vec<NodeId>,
}

/// Node in the computation graph
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique identifier for this node
    pub id: NodeId,
    /// The tensor value at this node
    pub value: Tensor,
    /// Gradient accumulated at this node
    pub gradient: Option<Tensor>,
    /// Operation that produced this node
    pub operation: Option<OperationType>,
    /// Parent nodes (inputs to the operation)
    pub parents: Vec<NodeId>,
    /// Child nodes (nodes that use this as input)
    pub children: Vec<NodeId>,
    /// Whether this node requires gradients
    pub requires_grad: bool,
    /// Whether this node is a leaf (variable)
    pub is_leaf: bool,
    /// Name for debugging
    pub name: Option<String>,
    /// Shape information for optimization
    pub shape: Vec<usize>,
}

/// Types of operations in the computation graph
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OperationType {
    /// Binary operations
    Add,
    Subtract,
    Multiply,
    Divide,
    MatrixMultiply,

    /// Unary operations
    Negate,
    Reciprocal,
    Square,
    Sqrt,
    Log,
    Exp,

    /// Activation functions
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU(f32),
    Softmax,
    LogSoftmax,

    /// Tensor operations
    Reshape(Vec<usize>),
    Transpose(Vec<usize>),
    Slice(Vec<std::ops::Range<usize>>),
    Concat(usize),     // axis
    Split(Vec<usize>), // split sizes

    /// Reduction operations
    Sum(Option<Vec<usize>>), // axes
    Mean(Option<Vec<usize>>), // axes
    Max(Option<Vec<usize>>),  // axes
    Min(Option<Vec<usize>>),  // axes

    /// Specialized operations
    LayerNorm(f32), // epsilon
    Dropout(f32),   // probability
    BatchNorm(f32), // epsilon

    /// Custom operation
    Custom(String),
}

/// Gradient function trait for operations
pub trait GradientFunction: Send + Sync {
    /// Compute gradients for the inputs given the gradient of the output
    fn backward(&self, grad_output: &Tensor, inputs: &[&Tensor]) -> Result<Vec<Tensor>>;

    /// Get the operation type
    fn operation_type(&self) -> OperationType;
}

impl ComputationGraph {
    /// Create a new empty computation graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
            topological_order: Vec::new(),
            dirty: false,
            root_nodes: Vec::new(),
            leaf_nodes: Vec::new(),
        }
    }

    /// Add a new node to the graph
    pub fn add_node(&mut self, value: Tensor, requires_grad: bool, name: Option<String>) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;

        let shape = value.shape();
        let node = GraphNode {
            id,
            value,
            gradient: None,
            operation: None,
            parents: Vec::new(),
            children: Vec::new(),
            requires_grad,
            is_leaf: true,
            name,
            shape,
        };

        self.nodes.insert(id, node);
        if requires_grad {
            self.root_nodes.push(id);
        }
        self.dirty = true;

        id
    }

    /// Add an operation node to the graph
    pub fn add_operation_node(
        &mut self,
        value: Tensor,
        operation: OperationType,
        parents: Vec<NodeId>,
        requires_grad: bool,
        name: Option<String>,
    ) -> Result<NodeId> {
        let id = self.next_id;
        self.next_id += 1;

        // Update parent nodes to include this as a child
        for parent_id in &parents {
            if let Some(parent) = self.nodes.get_mut(parent_id) {
                parent.children.push(id);
            } else {
                return Err(TrustformersError::tensor_op_error(
                    &format!("Parent node {} not found", parent_id),
                    "ComputationGraph::add_operation_node",
                ));
            }
        }

        let shape = value.shape();
        let node = GraphNode {
            id,
            value,
            gradient: None,
            operation: Some(operation),
            parents,
            children: Vec::new(),
            requires_grad,
            is_leaf: false,
            name,
            shape,
        };

        self.nodes.insert(id, node);
        self.dirty = true;

        Ok(id)
    }

    /// Get a node by ID
    pub fn get_node(&self, id: NodeId) -> Option<&GraphNode> {
        self.nodes.get(&id)
    }

    /// Get a mutable reference to a node by ID
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut GraphNode> {
        self.nodes.get_mut(&id)
    }

    /// Compute topological ordering of nodes
    pub fn compute_topological_order(&mut self) -> Result<()> {
        if !self.dirty {
            return Ok(());
        }

        let mut in_degree = HashMap::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        // Initialize in-degree counts
        for (id, node) in &self.nodes {
            in_degree.insert(*id, node.parents.len());
            if node.parents.is_empty() {
                queue.push_back(*id);
            }
        }

        // Process nodes in topological order
        while let Some(node_id) = queue.pop_front() {
            result.push(node_id);

            let Some(node) = self.nodes.get(&node_id) else {
                continue;
            };

            for child_id in &node.children {
                let Some(degree) = in_degree.get_mut(child_id) else {
                    continue;
                };

                *degree -= 1;
                if *degree == 0 {
                    queue.push_back(*child_id);
                }
            }
        }

        if result.len() != self.nodes.len() {
            return Err(TrustformersError::tensor_op_error(
                "Cycle detected in computation graph",
                "ComputationGraph::compute_topological_order",
            ));
        }

        self.topological_order = result;
        self.dirty = false;

        Ok(())
    }

    /// Perform backward pass to compute gradients
    pub fn backward(&mut self, output_id: NodeId, grad_output: Option<Tensor>) -> Result<()> {
        // Ensure topological order is computed
        self.compute_topological_order()?;

        // Initialize gradient for output node
        if let Some(output_node) = self.nodes.get_mut(&output_id) {
            output_node.gradient = Some(grad_output.unwrap_or_else(|| {
                Tensor::ones(&output_node.shape).expect("Failed to create ones tensor")
            }));
        } else {
            return Err(TrustformersError::tensor_op_error(
                &format!("Output node {} not found", output_id),
                "ComputationGraph::backward",
            ));
        }

        // Process nodes in reverse topological order
        for &node_id in self.topological_order.iter().rev() {
            let Some(node) = self.nodes.get(&node_id).cloned() else {
                continue;
            };

            let Some(ref grad) = node.gradient else {
                continue;
            };

            let Some(ref operation) = node.operation else {
                continue;
            };

            // Compute gradients for parent nodes
            let parent_gradients =
                self.compute_operation_gradients(operation, grad, &node.parents)?;

            // Accumulate gradients in parent nodes
            for (parent_id, parent_grad) in node.parents.iter().zip(parent_gradients.iter()) {
                let Some(parent_node) = self.nodes.get_mut(parent_id) else {
                    continue;
                };

                if !parent_node.requires_grad {
                    continue;
                }

                if let Some(ref mut existing_grad) = parent_node.gradient {
                    *existing_grad = existing_grad.add(parent_grad)?;
                } else {
                    parent_node.gradient = Some(parent_grad.clone());
                }
            }
        }

        Ok(())
    }

    /// Compute gradients for an operation
    fn compute_operation_gradients(
        &self,
        operation: &OperationType,
        grad_output: &Tensor,
        parent_ids: &[NodeId],
    ) -> Result<Vec<Tensor>> {
        let parent_values: Vec<&Tensor> =
            parent_ids.iter().map(|id| &self.nodes[id].value).collect();

        match operation {
            OperationType::Add => {
                // Gradient of addition is just the incoming gradient for both inputs
                Ok(vec![grad_output.clone(), grad_output.clone()])
            },
            OperationType::Subtract => {
                // Gradient of subtraction: da = dout, db = -dout
                Ok(vec![grad_output.clone(), grad_output.neg()?])
            },
            OperationType::Multiply => {
                // Gradient of multiplication: da = dout * b, db = dout * a
                if parent_values.len() != 2 {
                    return Err(TrustformersError::tensor_op_error(
                        "Multiply operation requires exactly 2 inputs",
                        "ComputationGraph::compute_operation_gradients",
                    ));
                }
                Ok(vec![
                    grad_output.mul(parent_values[1])?,
                    grad_output.mul(parent_values[0])?,
                ])
            },
            OperationType::Divide => {
                // Gradient of division: da = dout / b, db = -dout * a / (b * b)
                if parent_values.len() != 2 {
                    return Err(TrustformersError::tensor_op_error(
                        "Divide operation requires exactly 2 inputs",
                        "ComputationGraph::compute_operation_gradients",
                    ));
                }
                let a = parent_values[0];
                let b = parent_values[1];
                Ok(vec![
                    grad_output.div(b)?,
                    grad_output.mul(a)?.neg()?.div(&b.mul(b)?)?,
                ])
            },
            OperationType::MatrixMultiply => {
                // Gradient of matrix multiplication: da = dout @ b^T, db = a^T @ dout
                if parent_values.len() != 2 {
                    return Err(TrustformersError::tensor_op_error(
                        "MatrixMultiply operation requires exactly 2 inputs",
                        "ComputationGraph::compute_operation_gradients",
                    ));
                }
                let a = parent_values[0];
                let b = parent_values[1];

                // Compute gradients
                let a_shape = a.shape();
                let b_shape = b.shape();

                let grad_a = if a_shape.len() == 2 && b_shape.len() == 2 {
                    // Simple 2D matrix multiplication
                    grad_output.matmul(&b.transpose(1, 0)?)?
                } else {
                    // Handle batch matrix multiplication
                    let b_transposed = b.transpose(2, 1)?;
                    grad_output.matmul(&b_transposed)?
                };

                let grad_b = if a_shape.len() == 2 && b_shape.len() == 2 {
                    // Simple 2D matrix multiplication
                    a.transpose(1, 0)?.matmul(grad_output)?
                } else {
                    // Handle batch matrix multiplication
                    let a_transposed = a.permute(&[0, 2, 1])?;
                    a_transposed.matmul(grad_output)?
                };

                Ok(vec![grad_a, grad_b])
            },
            OperationType::Sigmoid => {
                // Gradient of sigmoid: dout * sigmoid(x) * (1 - sigmoid(x))
                if parent_values.len() != 1 {
                    return Err(TrustformersError::tensor_op_error(
                        "Sigmoid operation requires exactly 1 input",
                        "ComputationGraph::compute_operation_gradients",
                    ));
                }
                let sigmoid_out = parent_values[0].sigmoid()?;
                let one = Tensor::ones(&sigmoid_out.shape())?;
                let grad_input = grad_output.mul(&sigmoid_out)?.mul(&one.sub(&sigmoid_out)?)?;
                Ok(vec![grad_input])
            },
            OperationType::Tanh => {
                // Gradient of tanh: dout * (1 - tanh(x)^2)
                if parent_values.len() != 1 {
                    return Err(TrustformersError::tensor_op_error(
                        "Tanh operation requires exactly 1 input",
                        "ComputationGraph::compute_operation_gradients",
                    ));
                }
                let tanh_out = parent_values[0].tanh()?;
                let one = Tensor::ones(&tanh_out.shape())?;
                let grad_input = grad_output.mul(&one.sub(&tanh_out.mul(&tanh_out)?)?)?;
                Ok(vec![grad_input])
            },
            OperationType::ReLU => {
                // Gradient of ReLU: dout * (x > 0)
                if parent_values.len() != 1 {
                    return Err(TrustformersError::tensor_op_error(
                        "ReLU operation requires exactly 1 input",
                        "ComputationGraph::compute_operation_gradients",
                    ));
                }
                let input = parent_values[0];
                let zero = Tensor::zeros(&input.shape())?;
                let mask = input.greater(&zero)?;
                let grad_input = grad_output.mul(&mask)?;
                Ok(vec![grad_input])
            },
            OperationType::LeakyReLU(alpha) => {
                // Gradient of LeakyReLU: dout * (x > 0 ? 1 : alpha)
                if parent_values.len() != 1 {
                    return Err(TrustformersError::tensor_op_error(
                        "LeakyReLU operation requires exactly 1 input",
                        "ComputationGraph::compute_operation_gradients",
                    ));
                }
                let input = parent_values[0];
                let zero = Tensor::zeros(&input.shape())?;
                let alpha_tensor = Tensor::scalar(*alpha)?;
                let one = Tensor::ones(&input.shape())?;

                let positive_mask = input.greater(&zero)?;
                let negative_mask = one.sub(&positive_mask)?;

                let grad_input =
                    grad_output.mul(&positive_mask.add(&negative_mask.mul(&alpha_tensor)?)?)?;
                Ok(vec![grad_input])
            },
            OperationType::Sum(axes) => {
                // Gradient of sum: broadcast the gradient back to original shape
                if parent_values.len() != 1 {
                    return Err(TrustformersError::tensor_op_error(
                        "Sum operation requires exactly 1 input",
                        "ComputationGraph::compute_operation_gradients",
                    ));
                }
                let input_shape = parent_values[0].shape();
                let grad_input =
                    self.broadcast_gradient(grad_output, &input_shape, axes.as_ref())?;
                Ok(vec![grad_input])
            },
            OperationType::Mean(axes) => {
                // Gradient of mean: broadcast the gradient back and divide by the number of elements
                if parent_values.len() != 1 {
                    return Err(TrustformersError::tensor_op_error(
                        "Mean operation requires exactly 1 input",
                        "ComputationGraph::compute_operation_gradients",
                    ));
                }
                let input_shape = parent_values[0].shape();
                let grad_broadcasted =
                    self.broadcast_gradient(grad_output, &input_shape, axes.as_ref())?;

                // Compute the number of elements that were averaged
                let num_elements = if let Some(axes) = axes {
                    axes.iter().map(|&axis| input_shape[axis]).product::<usize>()
                } else {
                    input_shape.iter().product::<usize>()
                };

                let grad_input = grad_broadcasted.scalar_div(num_elements as f32)?;
                Ok(vec![grad_input])
            },
            OperationType::Reshape(target_shape) => {
                // Gradient of reshape: reshape gradient back to original shape
                if parent_values.len() != 1 {
                    return Err(TrustformersError::tensor_op_error(
                        "Reshape operation requires exactly 1 input",
                        "ComputationGraph::compute_operation_gradients",
                    ));
                }
                let original_shape = parent_values[0].shape();
                let grad_input = grad_output.reshape(&original_shape)?;
                Ok(vec![grad_input])
            },
            OperationType::Transpose(permutation) => {
                // Gradient of transpose: apply inverse permutation
                if parent_values.len() != 1 {
                    return Err(TrustformersError::tensor_op_error(
                        "Transpose operation requires exactly 1 input",
                        "ComputationGraph::compute_operation_gradients",
                    ));
                }
                let inverse_permutation = self.compute_inverse_permutation(permutation)?;
                let grad_input = grad_output.permute(&inverse_permutation)?;
                Ok(vec![grad_input])
            },
            _ => {
                // For unimplemented operations, return zero gradients
                let zero_grads = parent_values
                    .iter()
                    .map(|input| {
                        Tensor::zeros(&input.shape()).expect("Failed to create zeros tensor")
                    })
                    .collect();
                Ok(zero_grads)
            },
        }
    }

    /// Broadcast gradient back to original shape
    fn broadcast_gradient(
        &self,
        grad_output: &Tensor,
        original_shape: &[usize],
        axes: Option<&Vec<usize>>,
    ) -> Result<Tensor> {
        if let Some(axes) = axes {
            // Sum was performed along specific axes
            let mut result = grad_output.clone();
            for &axis in axes {
                result = result.unsqueeze(axis)?;
            }
            result.broadcast_to(original_shape)
        } else {
            // Sum was performed along all axes
            let grad_scalar = grad_output.clone();
            grad_scalar.broadcast_to(original_shape)
        }
    }

    /// Compute inverse permutation for transpose
    fn compute_inverse_permutation(&self, permutation: &[usize]) -> Result<Vec<usize>> {
        let mut inverse = vec![0; permutation.len()];
        for (i, &p) in permutation.iter().enumerate() {
            if p >= permutation.len() {
                return Err(TrustformersError::tensor_op_error(
                    &format!("Invalid permutation index: {}", p),
                    "ComputationGraph::compute_inverse_permutation",
                ));
            }
            inverse[p] = i;
        }
        Ok(inverse)
    }

    /// Clear all gradients in the graph
    pub fn zero_grad(&mut self) {
        for node in self.nodes.values_mut() {
            node.gradient = None;
        }
    }

    /// Get gradient for a specific node
    pub fn get_gradient(&self, node_id: NodeId) -> Option<&Tensor> {
        self.nodes.get(&node_id)?.gradient.as_ref()
    }

    /// Get value for a specific node
    pub fn get_value(&self, node_id: NodeId) -> Option<&Tensor> {
        self.nodes.get(&node_id).map(|node| &node.value)
    }

    /// Update the value of a node
    pub fn update_value(&mut self, node_id: NodeId, value: Tensor) -> Result<()> {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.value = value;
            node.shape = node.value.shape();
            Ok(())
        } else {
            Err(TrustformersError::tensor_op_error(
                &format!("Node {} not found", node_id),
                "ComputationGraph::update_value",
            ))
        }
    }

    /// Get all root nodes (variables)
    pub fn get_root_nodes(&self) -> &[NodeId] {
        &self.root_nodes
    }

    /// Get all leaf nodes
    pub fn get_leaf_nodes(&self) -> &[NodeId] {
        &self.leaf_nodes
    }

    /// Set a node as a leaf node
    pub fn set_leaf_node(&mut self, node_id: NodeId) {
        if !self.leaf_nodes.contains(&node_id) {
            self.leaf_nodes.push(node_id);
        }
    }

    /// Get the number of nodes in the graph
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get the topological order of nodes
    pub fn get_topological_order(&self) -> &[NodeId] {
        &self.topological_order
    }

    /// Export the graph structure for visualization
    pub fn export_graph(&self) -> GraphExport {
        let nodes: Vec<_> = self.nodes.values().cloned().collect();
        GraphExport {
            nodes,
            topological_order: self.topological_order.clone(),
        }
    }
}

/// Exported graph structure for visualization
#[derive(Debug, Clone)]
pub struct GraphExport {
    pub nodes: Vec<GraphNode>,
    pub topological_order: Vec<NodeId>,
}

impl Default for ComputationGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_graph_creation() {
        let mut graph = ComputationGraph::new();
        assert_eq!(graph.num_nodes(), 0);

        let tensor = Tensor::ones(&[2, 3]).unwrap();
        let node_id = graph.add_node(tensor, true, Some("test".to_string()));
        assert_eq!(graph.num_nodes(), 1);
        assert_eq!(node_id, 0);
    }

    #[test]
    fn test_topological_order() {
        let mut graph = ComputationGraph::new();

        // Create a simple computation: c = a + b
        let a = Tensor::ones(&[2, 2]).unwrap();
        let b = Tensor::ones(&[2, 2]).unwrap();
        let c = a.add(&b).unwrap();

        let node_a = graph.add_node(a, true, Some("a".to_string()));
        let node_b = graph.add_node(b, true, Some("b".to_string()));
        let node_c = graph
            .add_operation_node(
                c,
                OperationType::Add,
                vec![node_a, node_b],
                true,
                Some("c".to_string()),
            )
            .unwrap();

        graph.compute_topological_order().unwrap();
        let order = graph.get_topological_order();
        assert_eq!(order.len(), 3);

        // Verify that parents come before children
        let a_pos = order.iter().position(|&id| id == node_a).unwrap();
        let b_pos = order.iter().position(|&id| id == node_b).unwrap();
        let c_pos = order.iter().position(|&id| id == node_c).unwrap();

        assert!(a_pos < c_pos);
        assert!(b_pos < c_pos);
    }

    #[test]
    fn test_backward_pass() {
        let mut graph = ComputationGraph::new();

        // Create computation: c = a * b
        let a = Tensor::scalar(2.0).unwrap();
        let b = Tensor::scalar(3.0).unwrap();
        let c = a.mul(&b).unwrap();

        let node_a = graph.add_node(a.clone(), true, Some("a".to_string()));
        let node_b = graph.add_node(b.clone(), true, Some("b".to_string()));
        let node_c = graph
            .add_operation_node(
                c,
                OperationType::Multiply,
                vec![node_a, node_b],
                true,
                Some("c".to_string()),
            )
            .unwrap();

        // Backward pass
        graph.backward(node_c, None).unwrap();

        // Check gradients
        let grad_a = graph.get_gradient(node_a).unwrap();
        let grad_b = graph.get_gradient(node_b).unwrap();

        // Gradient of a should be b (3.0)
        // Gradient of b should be a (2.0)
        assert_eq!(grad_a.to_vec_f32().unwrap()[0], 3.0);
        assert_eq!(grad_b.to_vec_f32().unwrap()[0], 2.0);
    }

    #[test]
    fn test_gradient_accumulation() {
        let mut graph = ComputationGraph::new();

        // Create computation: d = a + a (gradient should accumulate)
        let a = Tensor::scalar(2.0).unwrap();
        let d = a.add(&a).unwrap();

        let node_a = graph.add_node(a.clone(), true, Some("a".to_string()));
        let node_d = graph
            .add_operation_node(
                d,
                OperationType::Add,
                vec![node_a, node_a],
                true,
                Some("d".to_string()),
            )
            .unwrap();

        // Backward pass
        graph.backward(node_d, None).unwrap();

        // Check gradient accumulation
        let grad_a = graph.get_gradient(node_a).unwrap();

        // Gradient should be 2.0 (1.0 + 1.0 from both uses)
        assert_eq!(grad_a.to_vec_f32().unwrap()[0], 2.0);
    }
}
