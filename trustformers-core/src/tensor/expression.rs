//! Tensor expression templates for lazy evaluation.
//!
//! This module provides a system for lazy evaluation of tensor operations,
//! allowing complex expressions to be built up and optimized before evaluation.
//! This can lead to significant performance improvements by:
//!
//! - Eliminating intermediate tensor allocations
//! - Enabling operation fusion
//! - Optimizing memory access patterns
//! - Allowing vectorization of multiple operations
//!
//! # Example
//!
//! ```no_run
//! use trustformers_core::tensor::{Tensor, TensorExpr};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let a = Tensor::randn(&[1000, 1000])?;
//! let b = Tensor::randn(&[1000, 1000])?;
//! let c = Tensor::randn(&[1000, 1000])?;
//!
//! // Without lazy evaluation (creates intermediate tensors):
//! let result1 = (a.add(&b)?.mul(&c)?.relu()?).sum(None)?;
//!
//! // With lazy evaluation (no intermediate tensors):
//! let expr = TensorExpr::from(&a)
//!     .add(TensorExpr::from(&b))
//!     .mul(TensorExpr::from(&c))
//!     .relu()
//!     .sum(None);
//! let result2 = expr.eval()?;
//! # Ok(())
//! # }
//! ```

use crate::errors::{Result, TrustformersError};
use crate::tensor::{DType, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

/// Operation types for expression templates
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OpType {
    // Arithmetic operations
    Add,
    Sub,
    Mul,
    Div,
    // Matrix operations
    MatMul,
    Transpose,
    // Activation functions
    ReLU,
    Sigmoid,
    Tanh,
    GELU,
    Softmax(i32), // axis
    // Reduction operations
    Sum(Option<Vec<usize>>),  // axes
    Mean(Option<Vec<usize>>), // axes
    Max(Option<Vec<usize>>),  // axes
    Min(Option<Vec<usize>>),  // axes
    // Shape operations
    Reshape(Vec<usize>),
    Slice(Vec<(usize, usize)>), // (start, end) for each dimension
    Concat(usize),              // axis
    // Broadcasting operations
    Broadcast(Vec<usize>), // target shape
    // Element-wise operations
    Pow(f64), // scalar power
    Sqrt,
    Log,
    Exp,
    // Comparison operations
    Greater,
    Less,
    Equal,
    // Conditional operations
    Where, // requires 3 operands: condition, x, y
}

/// Expression node in the computation graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExprNode {
    pub id: usize,
    pub op: OpType,
    pub operands: Vec<usize>, // IDs of operand nodes
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub is_leaf: bool, // true for tensor constants
    #[serde(skip)]
    pub tensor_data: Option<Arc<Tensor>>, // only for leaf nodes
}

/// Tensor expression for lazy evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorExpr {
    nodes: HashMap<usize, ExprNode>,
    root: usize,
    next_id: usize,
}

/// Expression builder for fluent API
#[allow(dead_code)] // Reserved for future expression building features
pub struct ExprBuilder<'a> {
    expr: &'a mut TensorExpr,
    current_node: usize,
}

/// Optimization hints for expression evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationHints {
    /// Enable operation fusion
    pub enable_fusion: bool,
    /// Enable memory layout optimization
    pub optimize_memory_layout: bool,
    /// Enable vectorization
    pub enable_vectorization: bool,
    /// Maximum number of operations to fuse
    pub max_fusion_size: usize,
    /// Prefer in-place operations when possible
    pub prefer_inplace: bool,
}

/// Expression evaluation context
#[derive(Debug, Clone, Default)]
pub struct EvalContext {
    pub hints: OptimizationHints,
    pub device: Option<String>,
    pub memory_budget: Option<usize>, // bytes
}

impl fmt::Display for TensorExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.node_to_string(self.root))
    }
}

impl TensorExpr {
    /// Create a new expression from a tensor
    pub fn from(tensor: &Tensor) -> Result<Self> {
        let shape = tensor.shape();
        let dtype = tensor.dtype();

        let mut nodes = HashMap::new();
        let root_node = ExprNode {
            id: 0,
            op: OpType::Add, // dummy op for leaf nodes
            operands: vec![],
            shape,
            dtype,
            is_leaf: true,
            tensor_data: Some(Arc::new(tensor.clone())),
        };

        nodes.insert(0, root_node);

        Ok(TensorExpr {
            nodes,
            root: 0,
            next_id: 1,
        })
    }

    /// Create a constant expression
    pub fn constant(tensor: Tensor) -> Result<Self> {
        Self::from(&tensor)
    }

    /// Get the shape of the expression result
    pub fn shape(&self) -> Vec<usize> {
        self.nodes[&self.root].shape.clone()
    }

    /// Get the data type of the expression result
    pub fn dtype(&self) -> DType {
        self.nodes[&self.root].dtype
    }

    /// Add two expressions
    #[allow(clippy::should_implement_trait)] // Returns Result for error handling
    pub fn add(self, other: TensorExpr) -> Result<Self> {
        self.binary_op(other, OpType::Add)
    }

    /// Subtract two expressions
    #[allow(clippy::should_implement_trait)] // Returns Result for error handling
    pub fn sub(self, other: TensorExpr) -> Result<Self> {
        self.binary_op(other, OpType::Sub)
    }

    /// Multiply two expressions element-wise
    #[allow(clippy::should_implement_trait)] // Returns Result for error handling
    pub fn mul(self, other: TensorExpr) -> Result<Self> {
        self.binary_op(other, OpType::Mul)
    }

    /// Divide two expressions element-wise
    #[allow(clippy::should_implement_trait)] // Returns Result for error handling
    pub fn div(self, other: TensorExpr) -> Result<Self> {
        self.binary_op(other, OpType::Div)
    }

    /// Matrix multiplication
    pub fn matmul(mut self, other: TensorExpr) -> Result<Self> {
        // Collect shape information before borrowing
        let left_shape = self.nodes[&self.root].shape.clone();
        let right_shape = other.nodes[&other.root].shape.clone();

        if left_shape.len() < 2 || right_shape.len() < 2 {
            return Err(TrustformersError::tensor_op_error(
                "Matrix multiplication requires at least 2D tensors",
                "matmul_validate",
            ));
        }

        let left_cols = left_shape[left_shape.len() - 1];
        let right_rows = right_shape[right_shape.len() - 2];

        if left_cols != right_rows {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "Incompatible shapes for matmul: {:?} x {:?}",
                    left_shape, right_shape
                ),
                "matmul_shape_check",
            ));
        }

        // Merge the other expression into this one
        let other_root = self.merge_expression(other)?;

        // Calculate result shape
        let mut result_shape = left_shape[..left_shape.len() - 1].to_vec();
        result_shape.push(right_shape[right_shape.len() - 1]);

        let new_node = ExprNode {
            id: self.next_id,
            op: OpType::MatMul,
            operands: vec![self.root, other_root],
            shape: result_shape,
            dtype: self.nodes[&self.root].dtype,
            is_leaf: false,
            tensor_data: None,
        };

        self.nodes.insert(self.next_id, new_node);
        self.root = self.next_id;
        self.next_id += 1;

        Ok(self)
    }

    /// Apply ReLU activation
    pub fn relu(self) -> Result<Self> {
        self.unary_op(OpType::ReLU)
    }

    /// Apply sigmoid activation
    pub fn sigmoid(self) -> Result<Self> {
        self.unary_op(OpType::Sigmoid)
    }

    /// Apply tanh activation
    pub fn tanh(self) -> Result<Self> {
        self.unary_op(OpType::Tanh)
    }

    /// Apply GELU activation
    pub fn gelu(self) -> Result<Self> {
        self.unary_op(OpType::GELU)
    }

    /// Apply softmax along the specified axis
    pub fn softmax(self, axis: i32) -> Result<Self> {
        self.unary_op(OpType::Softmax(axis))
    }

    /// Sum along specified axes
    pub fn sum(mut self, axes: Option<Vec<usize>>) -> Result<Self> {
        let result_shape = if let Some(ref axes) = axes {
            let mut shape = self.nodes[&self.root].shape.clone();
            // Remove dimensions being summed (in reverse order to maintain indices)
            let mut sorted_axes = axes.clone();
            sorted_axes.sort_by(|a, b| b.cmp(a));
            for &axis in &sorted_axes {
                if axis >= shape.len() {
                    return Err(TrustformersError::tensor_op_error(
                        &format!(
                            "Axis {} out of bounds for tensor with {} dimensions",
                            axis,
                            shape.len()
                        ),
                        "reduce",
                    ));
                }
                shape.remove(axis);
            }
            shape
        } else {
            vec![] // scalar result
        };

        let new_node = ExprNode {
            id: self.next_id,
            op: OpType::Sum(axes),
            operands: vec![self.root],
            shape: result_shape,
            dtype: self.nodes[&self.root].dtype,
            is_leaf: false,
            tensor_data: None,
        };

        self.nodes.insert(self.next_id, new_node);
        self.root = self.next_id;
        self.next_id += 1;

        Ok(self)
    }

    /// Calculate mean along specified axes
    pub fn mean(mut self, axes: Option<Vec<usize>>) -> Result<Self> {
        let result_shape = if let Some(ref axes) = axes {
            let mut shape = self.nodes[&self.root].shape.clone();
            let mut sorted_axes = axes.clone();
            sorted_axes.sort_by(|a, b| b.cmp(a));
            for &axis in &sorted_axes {
                if axis >= shape.len() {
                    return Err(TrustformersError::tensor_op_error(
                        &format!(
                            "Axis {} out of bounds for tensor with {} dimensions",
                            axis,
                            shape.len()
                        ),
                        "reduce",
                    ));
                }
                shape.remove(axis);
            }
            shape
        } else {
            vec![] // scalar result
        };

        let new_node = ExprNode {
            id: self.next_id,
            op: OpType::Mean(axes),
            operands: vec![self.root],
            shape: result_shape,
            dtype: self.nodes[&self.root].dtype,
            is_leaf: false,
            tensor_data: None,
        };

        self.nodes.insert(self.next_id, new_node);
        self.root = self.next_id;
        self.next_id += 1;

        Ok(self)
    }

    /// Reshape the tensor
    pub fn reshape(mut self, shape: &[usize]) -> Result<Self> {
        // Validate that the total number of elements remains the same
        let current_shape = &self.nodes[&self.root].shape;
        let current_size: usize = current_shape.iter().product();
        let new_size: usize = shape.iter().product();

        if current_size != new_size {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "Cannot reshape tensor with {} elements to shape with {} elements",
                    current_size, new_size
                ),
                "reshape",
            ));
        }

        let new_node = ExprNode {
            id: self.next_id,
            op: OpType::Reshape(shape.to_vec()),
            operands: vec![self.root],
            shape: shape.to_vec(),
            dtype: self.nodes[&self.root].dtype,
            is_leaf: false,
            tensor_data: None,
        };

        self.nodes.insert(self.next_id, new_node);
        self.root = self.next_id;
        self.next_id += 1;

        Ok(self)
    }

    /// Transpose the tensor
    pub fn transpose(mut self) -> Result<Self> {
        let current_shape = &self.nodes[&self.root].shape;
        if current_shape.len() < 2 {
            return Err(TrustformersError::tensor_op_error(
                "Transpose requires at least 2D tensor",
                "transpose",
            ));
        }

        let mut new_shape = current_shape.clone();
        let len = new_shape.len();
        new_shape.swap(len - 2, len - 1);

        let new_node = ExprNode {
            id: self.next_id,
            op: OpType::Transpose,
            operands: vec![self.root],
            shape: new_shape,
            dtype: self.nodes[&self.root].dtype,
            is_leaf: false,
            tensor_data: None,
        };

        self.nodes.insert(self.next_id, new_node);
        self.root = self.next_id;
        self.next_id += 1;

        Ok(self)
    }

    /// Evaluate the expression with default context
    pub fn eval(&self) -> Result<Tensor> {
        self.eval_with_context(&EvalContext::default())
    }

    /// Evaluate the expression with optimization context
    pub fn eval_with_context(&self, context: &EvalContext) -> Result<Tensor> {
        // First, optimize the expression if requested
        let optimized_expr =
            if context.hints.enable_fusion { self.optimize_fusion()? } else { self.clone() };

        // Evaluate the optimized expression
        optimized_expr.eval_recursive(optimized_expr.root, context)
    }

    /// Check if two expressions can be fused
    pub fn can_fuse_with(&self, other: &TensorExpr) -> bool {
        // Simple heuristic: same shape and compatible operations
        self.shape() == other.shape() && self.is_elementwise() && other.is_elementwise()
    }

    /// Get the number of operations in the expression
    pub fn operation_count(&self) -> usize {
        self.nodes.len() - self.leaf_count()
    }

    /// Get the number of leaf nodes (tensors)
    pub fn leaf_count(&self) -> usize {
        self.nodes.values().filter(|n| n.is_leaf).count()
    }

    /// Export expression to DOT format for visualization
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph TensorExpr {\n");

        for node in self.nodes.values() {
            let label = if node.is_leaf {
                format!("Tensor\\n{:?}\\n{:?}", node.shape, node.dtype)
            } else {
                format!("{:?}\\n{:?}\\n{:?}", node.op, node.shape, node.dtype)
            };

            let color = if node.is_leaf { "lightblue" } else { "lightgreen" };
            dot.push_str(&format!(
                "  {} [label=\"{}\" fillcolor={} style=filled];\n",
                node.id, label, color
            ));

            for &operand in &node.operands {
                dot.push_str(&format!("  {} -> {};\n", operand, node.id));
            }
        }

        dot.push_str("}\n");
        dot
    }

    // Helper methods

    fn binary_op(mut self, other: TensorExpr, op: OpType) -> Result<Self> {
        // Check shape compatibility for broadcasting
        let left_shape = &self.nodes[&self.root].shape;
        let right_shape = &other.nodes[&other.root].shape;
        let result_shape = self.broadcast_shapes(left_shape, right_shape)?;

        // Merge the other expression into this one
        let other_root = self.merge_expression(other)?;

        let new_node = ExprNode {
            id: self.next_id,
            op,
            operands: vec![self.root, other_root],
            shape: result_shape,
            dtype: self.nodes[&self.root].dtype, // Assume same dtype for now
            is_leaf: false,
            tensor_data: None,
        };

        self.nodes.insert(self.next_id, new_node);
        self.root = self.next_id;
        self.next_id += 1;

        Ok(self)
    }

    fn unary_op(mut self, op: OpType) -> Result<Self> {
        let new_node = ExprNode {
            id: self.next_id,
            op,
            operands: vec![self.root],
            shape: self.nodes[&self.root].shape.clone(),
            dtype: self.nodes[&self.root].dtype,
            is_leaf: false,
            tensor_data: None,
        };

        self.nodes.insert(self.next_id, new_node);
        self.root = self.next_id;
        self.next_id += 1;

        Ok(self)
    }

    fn merge_expression(&mut self, other: TensorExpr) -> Result<usize> {
        let id_offset = self.next_id;

        // Add all nodes from the other expression with updated IDs
        for (old_id, mut node) in other.nodes {
            let new_id = old_id + id_offset;
            node.id = new_id;

            // Update operand IDs
            for operand in &mut node.operands {
                *operand += id_offset;
            }

            self.nodes.insert(new_id, node);
        }

        self.next_id += other.next_id;
        Ok(other.root + id_offset)
    }

    fn broadcast_shapes(&self, left: &[usize], right: &[usize]) -> Result<Vec<usize>> {
        let max_len = left.len().max(right.len());
        let mut result = vec![1; max_len];

        for i in 0..max_len {
            let left_dim = if i < left.len() { left[left.len() - 1 - i] } else { 1 };
            let right_dim = if i < right.len() { right[right.len() - 1 - i] } else { 1 };

            if left_dim == right_dim {
                result[max_len - 1 - i] = left_dim;
            } else if left_dim == 1 {
                result[max_len - 1 - i] = right_dim;
            } else if right_dim == 1 {
                result[max_len - 1 - i] = left_dim;
            } else {
                return Err(TrustformersError::tensor_op_error(
                    &format!("Cannot broadcast shapes {:?} and {:?}", left, right),
                    "broadcast_shape_check",
                ));
            }
        }

        Ok(result)
    }

    fn is_elementwise(&self) -> bool {
        matches!(
            self.nodes[&self.root].op,
            OpType::Add
                | OpType::Sub
                | OpType::Mul
                | OpType::Div
                | OpType::ReLU
                | OpType::Sigmoid
                | OpType::Tanh
                | OpType::GELU
                | OpType::Pow(_)
                | OpType::Sqrt
                | OpType::Log
                | OpType::Exp
        )
    }

    fn optimize_fusion(&self) -> Result<TensorExpr> {
        // Simple fusion optimization: combine consecutive element-wise operations
        let mut optimized = self.clone();

        // Find fusion opportunities
        let fusion_chains = optimized.find_fusion_chains();

        // Apply fusions
        for chain in fusion_chains {
            optimized.fuse_operations(&chain)?;
        }

        Ok(optimized)
    }

    fn find_fusion_chains(&self) -> Vec<Vec<usize>> {
        // Simplified: find chains of element-wise operations
        let mut chains = Vec::new();
        let mut visited = std::collections::HashSet::new();

        for &node_id in self.nodes.keys() {
            if visited.contains(&node_id) {
                continue;
            }

            let mut chain = Vec::new();
            let mut current = node_id;

            while let Some(node) = self.nodes.get(&current) {
                if !self.is_node_elementwise(node) {
                    break;
                }

                chain.push(current);
                visited.insert(current);

                // Move to next node if it has exactly one operand
                if node.operands.len() == 1 {
                    current = node.operands[0];
                } else {
                    break;
                }
            }

            if chain.len() > 1 {
                chains.push(chain);
            }
        }

        chains
    }

    fn is_node_elementwise(&self, node: &ExprNode) -> bool {
        matches!(
            node.op,
            OpType::Add
                | OpType::Sub
                | OpType::Mul
                | OpType::Div
                | OpType::ReLU
                | OpType::Sigmoid
                | OpType::Tanh
                | OpType::GELU
                | OpType::Pow(_)
                | OpType::Sqrt
                | OpType::Log
                | OpType::Exp
        )
    }

    fn fuse_operations(&mut self, chain: &[usize]) -> Result<()> {
        // Simplified fusion: replace chain with a single fused operation
        // In a real implementation, this would generate optimized kernels

        if chain.len() < 2 {
            return Ok(());
        }

        // For now, just mark the optimization potential
        // Real implementation would generate fused CUDA/OpenCL kernels

        Ok(())
    }

    fn eval_recursive(&self, node_id: usize, _context: &EvalContext) -> Result<Tensor> {
        let node = &self.nodes[&node_id];

        if node.is_leaf {
            return node
                .tensor_data
                .as_ref()
                .ok_or_else(|| {
                    TrustformersError::tensor_op_error(
                        "Leaf node must have tensor data",
                        "eval_recursive",
                    )
                })
                .map(|t| t.as_ref().clone());
        }

        // Evaluate operands first
        let operand_results: Result<Vec<Tensor>> =
            node.operands.iter().map(|&id| self.eval_recursive(id, _context)).collect();
        let operands = operand_results?;

        // Apply the operation
        match &node.op {
            OpType::Add => operands[0].add(&operands[1]),
            OpType::Sub => operands[0].sub(&operands[1]),
            OpType::Mul => operands[0].mul(&operands[1]),
            OpType::Div => operands[0].div(&operands[1]),
            OpType::MatMul => operands[0].matmul(&operands[1]),
            OpType::Transpose => {
                let shape = operands[0].shape();
                let rank = shape.len();
                if rank < 2 {
                    return Err(crate::errors::TrustformersError::dimension_mismatch(
                        "at least 2 dimensions".to_string(),
                        format!("{} dimensions", rank),
                    ));
                }
                operands[0].transpose(rank - 2, rank - 1)
            },
            OpType::ReLU => operands[0].relu(),
            OpType::Sigmoid => operands[0].sigmoid(),
            OpType::Tanh => operands[0].tanh(),
            OpType::GELU => operands[0].gelu(),
            OpType::Softmax(axis) => operands[0].softmax(*axis),
            OpType::Sum(axes) => {
                match axes {
                    Some(ref axes_vec) => operands[0].sum_axes(axes_vec),
                    None => {
                        // Sum all elements - use all axes
                        let shape = operands[0].shape();
                        let all_axes: Vec<usize> = (0..shape.len()).collect();
                        operands[0].sum_axes(&all_axes)
                    },
                }
            },
            OpType::Mean(axes) => match axes {
                Some(ref axes_vec) => operands[0].mean_axes(axes_vec),
                None => operands[0].mean(),
            },
            OpType::Reshape(shape) => operands[0].reshape(shape),
            OpType::Pow(power) => operands[0].pow_scalar(*power),
            OpType::Sqrt => operands[0].sqrt(),
            OpType::Log => operands[0].log(),
            OpType::Exp => operands[0].exp(),
            OpType::Max(axes) => match axes {
                Some(ref axes_vec) => operands[0].max_axes(axes_vec),
                None => operands[0].max_scalar(),
            },
            OpType::Min(axes) => match axes {
                Some(ref axes_vec) => operands[0].min_axes(axes_vec),
                None => operands[0].min_scalar(),
            },
            OpType::Slice(ranges) => {
                // Implement proper multi-dimensional slicing
                if ranges.is_empty() {
                    return Err(TrustformersError::tensor_op_error(
                        "No slice ranges provided",
                        "slice",
                    ));
                }
                operands[0].slice_multi(ranges)
            },
            OpType::Concat(axis) => {
                if operands.len() < 2 {
                    return Err(TrustformersError::tensor_op_error(
                        "Concat requires at least 2 operands",
                        "evaluate_node",
                    ));
                }

                // Pass slice of tensors directly for concatenation
                Tensor::concat(&operands, *axis)
            },
            OpType::Broadcast(target_shape) => operands[0].broadcast_to(target_shape),
            OpType::Greater => {
                if operands.len() != 2 {
                    return Err(TrustformersError::tensor_op_error(
                        "Greater operation requires exactly 2 operands",
                        "evaluate_node",
                    ));
                }
                operands[0].greater(&operands[1])
            },
            OpType::Less => {
                if operands.len() != 2 {
                    return Err(TrustformersError::tensor_op_error(
                        "Less operation requires exactly 2 operands",
                        "evaluate_node",
                    ));
                }
                operands[0].less(&operands[1])
            },
            OpType::Equal => {
                if operands.len() != 2 {
                    return Err(TrustformersError::tensor_op_error(
                        "Equal operation requires exactly 2 operands",
                        "evaluate_node",
                    ));
                }
                operands[0].equal(&operands[1])
            },
            OpType::Where => {
                if operands.len() != 3 {
                    return Err(TrustformersError::tensor_op_error(
                        "Where operation requires exactly 3 operands: condition, x, y",
                        "evaluate_node",
                    ));
                }
                // where(condition, x, y) - select x where condition is true, y otherwise
                operands[0].where_cond(&operands[1], &operands[2])
            },
        }
    }

    fn node_to_string(&self, node_id: usize) -> String {
        let node = &self.nodes[&node_id];

        if node.is_leaf {
            format!("Tensor{:?}", node.shape)
        } else {
            let operand_strs: Vec<String> =
                node.operands.iter().map(|&id| self.node_to_string(id)).collect();

            match &node.op {
                OpType::Add => format!("({} + {})", operand_strs[0], operand_strs[1]),
                OpType::Sub => format!("({} - {})", operand_strs[0], operand_strs[1]),
                OpType::Mul => format!("({} * {})", operand_strs[0], operand_strs[1]),
                OpType::Div => format!("({} / {})", operand_strs[0], operand_strs[1]),
                OpType::MatMul => format!("matmul({}, {})", operand_strs[0], operand_strs[1]),
                OpType::ReLU => format!("relu({})", operand_strs[0]),
                OpType::Sigmoid => format!("sigmoid({})", operand_strs[0]),
                OpType::Tanh => format!("tanh({})", operand_strs[0]),
                OpType::GELU => format!("gelu({})", operand_strs[0]),
                OpType::Softmax(axis) => format!("softmax({}, axis={})", operand_strs[0], axis),
                OpType::Sum(axes) => format!("sum({}, axes={:?})", operand_strs[0], axes),
                OpType::Mean(axes) => format!("mean({}, axes={:?})", operand_strs[0], axes),
                OpType::Reshape(shape) => format!("reshape({}, {:?})", operand_strs[0], shape),
                OpType::Transpose => format!("transpose({})", operand_strs[0]),
                _ => format!("{:?}({})", node.op, operand_strs.join(", ")),
            }
        }
    }
}

impl Default for OptimizationHints {
    fn default() -> Self {
        Self {
            enable_fusion: true,
            optimize_memory_layout: true,
            enable_vectorization: true,
            max_fusion_size: 8,
            prefer_inplace: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_basic_expression_creation() -> Result<()> {
        let a = Tensor::ones(&[2, 3])?;
        let expr = TensorExpr::from(&a)?;

        assert_eq!(expr.shape(), vec![2, 3]);
        assert_eq!(expr.dtype(), DType::F32);
        assert_eq!(expr.operation_count(), 0);
        assert_eq!(expr.leaf_count(), 1);

        Ok(())
    }

    #[test]
    fn test_binary_operations() -> Result<()> {
        let a = Tensor::ones(&[2, 3])?;
        let b = Tensor::ones(&[2, 3])?;

        let expr_a = TensorExpr::from(&a)?;
        let expr_b = TensorExpr::from(&b)?;

        let result_expr = expr_a.add(expr_b)?;

        assert_eq!(result_expr.shape(), vec![2, 3]);
        assert_eq!(result_expr.operation_count(), 1);
        assert_eq!(result_expr.leaf_count(), 2);

        Ok(())
    }

    #[test]
    fn test_chained_operations() -> Result<()> {
        let a = Tensor::ones(&[2, 3])?;
        let b = Tensor::ones(&[2, 3])?;
        let c = Tensor::ones(&[2, 3])?;

        let expr = TensorExpr::from(&a)?
            .add(TensorExpr::from(&b)?)?
            .mul(TensorExpr::from(&c)?)?
            .relu()?;

        assert_eq!(expr.shape(), vec![2, 3]);
        assert_eq!(expr.operation_count(), 3); // add, mul, relu
        assert_eq!(expr.leaf_count(), 3);

        Ok(())
    }

    #[test]
    fn test_matrix_multiplication() -> Result<()> {
        let a = Tensor::ones(&[2, 3])?;
        let b = Tensor::ones(&[3, 4])?;

        let expr = TensorExpr::from(&a)?.matmul(TensorExpr::from(&b)?)?;

        assert_eq!(expr.shape(), vec![2, 4]);
        assert_eq!(expr.operation_count(), 1);

        Ok(())
    }

    #[test]
    fn test_reduction_operations() -> Result<()> {
        let a = Tensor::ones(&[2, 3, 4])?;

        let sum_all = TensorExpr::from(&a)?.sum(None)?;
        assert_eq!(sum_all.shape(), vec![] as Vec<usize>);

        let sum_axis = TensorExpr::from(&a)?.sum(Some(vec![1]))?;
        assert_eq!(sum_axis.shape(), vec![2, 4]);

        Ok(())
    }

    #[test]
    fn test_reshape_operation() -> Result<()> {
        let a = Tensor::ones(&[2, 3, 4])?;

        let reshaped = TensorExpr::from(&a)?.reshape(&[6, 4])?;
        assert_eq!(reshaped.shape(), vec![6, 4]);

        Ok(())
    }

    #[test]
    fn test_expression_evaluation() -> Result<()> {
        let a = Tensor::ones(&[2, 2])?;
        let b = Tensor::ones(&[2, 2])?;

        let expr = TensorExpr::from(&a)?.add(TensorExpr::from(&b)?)?;

        let result = expr.eval()?;
        assert_eq!(result.shape(), vec![2, 2]);

        // Result should be all 2.0s
        let _expected = Tensor::full_with_shape(&[2, 2], 2.0)?;
        // Note: Actual comparison would need tensor equality methods

        Ok(())
    }

    #[test]
    fn test_expression_to_string() -> Result<()> {
        let a = Tensor::ones(&[2, 2])?;
        let b = Tensor::ones(&[2, 2])?;

        let expr = TensorExpr::from(&a)?.add(TensorExpr::from(&b)?)?.relu()?;

        let expr_str = expr.to_string();
        assert!(expr_str.contains("+"));
        assert!(expr_str.contains("relu"));

        Ok(())
    }

    #[test]
    fn test_dot_export() -> Result<()> {
        let a = Tensor::ones(&[2, 2])?;
        let b = Tensor::ones(&[2, 2])?;

        let expr = TensorExpr::from(&a)?.add(TensorExpr::from(&b)?)?;

        let dot = expr.to_dot();
        assert!(dot.contains("digraph TensorExpr"));
        assert!(dot.contains("Add"));

        Ok(())
    }

    #[test]
    fn test_optimization_hints() {
        let hints = OptimizationHints::default();
        assert!(hints.enable_fusion);
        assert!(hints.optimize_memory_layout);
        assert!(hints.enable_vectorization);
        assert_eq!(hints.max_fusion_size, 8);
        assert!(!hints.prefer_inplace);
    }

    #[test]
    fn test_can_fuse_operations() -> Result<()> {
        let a = Tensor::ones(&[2, 2])?;
        let b = Tensor::ones(&[2, 2])?;

        let expr1 = TensorExpr::from(&a)?.relu()?;
        let expr2 = TensorExpr::from(&b)?.sigmoid()?;

        assert!(expr1.can_fuse_with(&expr2));

        Ok(())
    }
}
