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
//! let result1 = (a.add(&b)?.mul(&c)?.relu()?).sum(None, false)?;
//!
//! // With lazy evaluation (no intermediate tensors):
//! let expr = TensorExpr::from(&a)?
//!     .add(TensorExpr::from(&b)?)?
//!     .mul(TensorExpr::from(&c)?)?
//!     .relu()?
//!     .sum(None)?;
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
    /// A chain of *unary* elementwise operations collapsed into a single node.
    ///
    /// The ops are stored in application order: `FusedElementwise(vec![Exp,
    /// Sqrt])` computes `sqrt(exp(x))`. Evaluating it performs one traversal of
    /// the data instead of one full-size allocation and traversal per link, so
    /// an `n`-op chain over an `N`-element tensor drops from `n` intermediate
    /// buffers to zero.
    ///
    /// Produced by [`TensorExpr::optimize_fusion`]; never built directly by the
    /// public expression API.
    FusedElementwise(Vec<OpType>),
}

impl OpType {
    /// Whether this op maps one input element to one output element with no
    /// dependence on any other element (so a chain of them can be fused into a
    /// single pass).
    ///
    /// `Softmax` is deliberately excluded: it reduces over an axis. Binary ops
    /// are elementwise but take two operands, so they are not part of a *unary*
    /// fusion chain.
    fn is_unary_elementwise(&self) -> bool {
        matches!(
            self,
            OpType::ReLU
                | OpType::Sigmoid
                | OpType::Tanh
                | OpType::GELU
                | OpType::Pow(_)
                | OpType::Sqrt
                | OpType::Log
                | OpType::Exp
        )
    }

    /// Apply a single unary elementwise op to one `f32` value.
    ///
    /// The formulas mirror the corresponding `Tensor` methods exactly (notably
    /// the tanh GELU approximation), so fusing a chain cannot change results.
    fn apply_scalar_f32(&self, x: f32) -> Result<f32> {
        Ok(match self {
            OpType::ReLU => x.max(0.0),
            OpType::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            OpType::Tanh => x.tanh(),
            OpType::GELU => 0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x.powi(3))).tanh()),
            OpType::Pow(power) => x.powf(*power as f32),
            OpType::Sqrt => x.sqrt(),
            OpType::Log => x.ln(),
            OpType::Exp => x.exp(),
            other => {
                return Err(TrustformersError::tensor_op_error(
                    &format!("{:?} is not a unary elementwise operation", other),
                    "OpType::apply_scalar_f32",
                ));
            },
        })
    }

    /// `f64` counterpart of [`OpType::apply_scalar_f32`].
    fn apply_scalar_f64(&self, x: f64) -> Result<f64> {
        Ok(match self {
            OpType::ReLU => x.max(0.0),
            OpType::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            OpType::Tanh => x.tanh(),
            OpType::GELU => 0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x.powi(3))).tanh()),
            OpType::Pow(power) => x.powf(*power),
            OpType::Sqrt => x.sqrt(),
            OpType::Log => x.ln(),
            OpType::Exp => x.exp(),
            other => {
                return Err(TrustformersError::tensor_op_error(
                    &format!("{:?} is not a unary elementwise operation", other),
                    "OpType::apply_scalar_f64",
                ));
            },
        })
    }
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

    /// Elementwise square root.
    ///
    /// `OpType::Sqrt` was already evaluated by `eval_recursive` but had no
    /// builder, so the operation was unreachable through the public API.
    pub fn sqrt(self) -> Result<Self> {
        self.unary_op(OpType::Sqrt)
    }

    /// Elementwise natural logarithm.
    pub fn log(self) -> Result<Self> {
        self.unary_op(OpType::Log)
    }

    /// Elementwise exponential.
    pub fn exp(self) -> Result<Self> {
        self.unary_op(OpType::Exp)
    }

    /// Elementwise power with a scalar exponent.
    pub fn pow(self, exponent: f64) -> Result<Self> {
        self.unary_op(OpType::Pow(exponent))
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
        // First, optimize the expression if requested. `max_fusion_size` caps how
        // many links one fused node may absorb.
        let optimized_expr = if context.hints.enable_fusion {
            self.optimize_fusion_with_limit(context.hints.max_fusion_size)?
        } else {
            self.clone()
        };

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

    /// Rewrite chains of unary elementwise operations into single
    /// [`OpType::FusedElementwise`] nodes.
    ///
    /// This is a real rewrite, not a report: the returned expression has fewer
    /// nodes, and evaluating it performs one traversal per fused chain instead
    /// of one full-size allocation and traversal per link. The previous version
    /// discovered chains and then discarded them (`fuse_operations` had an empty
    /// body), so `optimize_fusion` returned an unmodified clone.
    ///
    /// Only chains whose intermediate nodes have exactly one consumer are fused
    /// -- a shared intermediate is needed by another branch, so collapsing it
    /// would force it to be recomputed.
    pub fn optimize_fusion(&self) -> Result<TensorExpr> {
        self.optimize_fusion_with_limit(usize::MAX)
    }

    /// [`TensorExpr::optimize_fusion`] with an explicit cap on how many
    /// operations one fused node may absorb (`OptimizationHints::max_fusion_size`).
    fn optimize_fusion_with_limit(&self, max_chain: usize) -> Result<TensorExpr> {
        if max_chain < 2 {
            return Ok(self.clone());
        }

        let mut optimized = self.clone();
        for chain in self.find_fusion_chains(max_chain) {
            optimized.fuse_operations(&chain)?;
        }
        optimized.prune_unreachable_nodes();
        Ok(optimized)
    }

    /// How many nodes consume each node's output.
    fn consumer_counts(&self) -> HashMap<usize, usize> {
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for node in self.nodes.values() {
            for &operand in &node.operands {
                *counts.entry(operand).or_insert(0) += 1;
            }
        }
        counts
    }

    /// Find maximal chains of unary elementwise nodes.
    ///
    /// A chain is returned outermost-first (`[outer, ..., inner]`); the node
    /// *below* the innermost link stays untouched and becomes the fused node's
    /// single operand.
    fn find_fusion_chains(&self, max_len: usize) -> Vec<Vec<usize>> {
        if max_len < 2 {
            return Vec::new();
        }

        let consumers = self.consumer_counts();

        // Visit consumers before their operands, so a chain always starts at the
        // outermost link that is still free. Without that order, capping a chain
        // (`max_len`) would strand the released interior nodes: they would still
        // look like the middle of a longer chain and be skipped forever.
        let node_ids = self.nodes_outermost_first();

        let mut claimed = std::collections::HashSet::new();
        let mut chains = Vec::new();

        for node_id in node_ids {
            if claimed.contains(&node_id) {
                continue;
            }
            if !self.is_fusable_link(node_id) {
                continue;
            }
            // Skip nodes that are still the interior of a chain rooted higher up.
            // Once that root has been processed the consumer is claimed, and this
            // node becomes a legal chain start on a later iteration.
            if self.has_unclaimed_fusable_sole_consumer(node_id, &consumers, &claimed) {
                continue;
            }

            let mut chain = vec![node_id];
            let mut current = node_id;
            while chain.len() < max_len {
                let Some(node) = self.nodes.get(&current) else {
                    break;
                };
                let Some(&operand) = node.operands.first() else {
                    break;
                };
                if !self.is_fusable_link(operand) {
                    break;
                }
                // Only absorb an intermediate that nothing else consumes.
                if consumers.get(&operand).copied().unwrap_or(0) != 1 {
                    break;
                }
                chain.push(operand);
                current = operand;
            }

            if chain.len() >= 2 {
                for &id in &chain {
                    claimed.insert(id);
                }
                chains.push(chain);
            }
        }

        chains
    }

    /// Whether `node_id` is a non-leaf, single-operand, unary elementwise node.
    fn is_fusable_link(&self, node_id: usize) -> bool {
        match self.nodes.get(&node_id) {
            Some(node) => {
                !node.is_leaf && node.operands.len() == 1 && node.op.is_unary_elementwise()
            },
            None => false,
        }
    }

    fn has_unclaimed_fusable_sole_consumer(
        &self,
        node_id: usize,
        consumers: &HashMap<usize, usize>,
        claimed: &std::collections::HashSet<usize>,
    ) -> bool {
        if consumers.get(&node_id).copied().unwrap_or(0) != 1 {
            return false;
        }
        self.nodes.values().any(|candidate| {
            candidate.operands.first() == Some(&node_id)
                && self.is_fusable_link(candidate.id)
                && !claimed.contains(&candidate.id)
        })
    }

    /// Node ids ordered so that every node precedes its operands (root first).
    ///
    /// Nodes unreachable from the root are appended in id order so the traversal
    /// is total and deterministic.
    fn nodes_outermost_first(&self) -> Vec<usize> {
        let mut order = Vec::with_capacity(self.nodes.len());
        let mut seen = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(self.root);

        while let Some(id) = queue.pop_front() {
            if !seen.insert(id) {
                continue;
            }
            order.push(id);
            if let Some(node) = self.nodes.get(&id) {
                for &operand in &node.operands {
                    queue.push_back(operand);
                }
            }
        }

        let mut orphans: Vec<usize> =
            self.nodes.keys().copied().filter(|id| !seen.contains(id)).collect();
        orphans.sort_unstable();
        order.extend(orphans);
        order
    }

    /// Replace `chain` (outermost first) with a single `FusedElementwise` node.
    ///
    /// The outermost node keeps its id, so every consumer -- including `root` --
    /// keeps pointing at it; only its `op` and `operands` change.
    fn fuse_operations(&mut self, chain: &[usize]) -> Result<()> {
        if chain.len() < 2 {
            return Ok(());
        }

        // Collect the ops innermost-first, which is application order.
        let mut ops = Vec::with_capacity(chain.len());
        for &node_id in chain.iter().rev() {
            let node = self.nodes.get(&node_id).ok_or_else(|| {
                TrustformersError::tensor_op_error(
                    &format!("fusion chain references unknown node {}", node_id),
                    "TensorExpr::fuse_operations",
                )
            })?;
            // Flatten a previously fused node so repeated passes stay linear.
            match &node.op {
                OpType::FusedElementwise(inner) => ops.extend(inner.iter().cloned()),
                other => ops.push(other.clone()),
            }
        }

        let innermost = *chain.last().unwrap_or(&chain[0]);
        let source = self
            .nodes
            .get(&innermost)
            .and_then(|node| node.operands.first().copied())
            .ok_or_else(|| {
                TrustformersError::tensor_op_error(
                    "innermost fusion link has no operand",
                    "TensorExpr::fuse_operations",
                )
            })?;

        let outermost = chain[0];
        let fused = self.nodes.get_mut(&outermost).ok_or_else(|| {
            TrustformersError::tensor_op_error(
                &format!("fusion chain references unknown node {}", outermost),
                "TensorExpr::fuse_operations",
            )
        })?;
        fused.op = OpType::FusedElementwise(ops);
        fused.operands = vec![source];

        // The interior nodes are now unreachable; `prune_unreachable_nodes`
        // removes them once every chain has been rewritten.
        Ok(())
    }

    /// Evaluate a fused chain of unary elementwise ops in a single traversal.
    ///
    /// One allocation for the output; every intermediate stays in a register.
    /// The per-op formulas are the same ones the unfused `Tensor` methods use,
    /// so results are bit-identical to evaluating the chain link by link
    /// (modulo the intermediate rounding that no longer happens through memory).
    fn eval_fused_elementwise(input: &Tensor, ops: &[OpType]) -> Result<Tensor> {
        if ops.is_empty() {
            return Ok(input.clone());
        }
        for op in ops {
            if !op.is_unary_elementwise() {
                return Err(TrustformersError::tensor_op_error(
                    &format!("{:?} cannot appear in a fused elementwise chain", op),
                    "eval_fused_elementwise",
                ));
            }
        }

        match input {
            Tensor::F32(a) => {
                let mut result = a.as_standard_layout().into_owned();
                for value in result.iter_mut() {
                    let mut acc = *value;
                    for op in ops {
                        acc = op.apply_scalar_f32(acc)?;
                    }
                    *value = acc;
                }
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let mut result = a.as_standard_layout().into_owned();
                for value in result.iter_mut() {
                    let mut acc = *value;
                    for op in ops {
                        acc = op.apply_scalar_f64(acc)?;
                    }
                    *value = acc;
                }
                Ok(Tensor::F64(result))
            },
            // Integer / complex / device tensors have no fused kernel: fall back
            // to applying the chain one operation at a time, which is exactly
            // what an unfused expression would have done.
            other => {
                let mut current = other.clone();
                for op in ops {
                    current = match op {
                        OpType::ReLU => current.relu()?,
                        OpType::Sigmoid => current.sigmoid()?,
                        OpType::Tanh => current.tanh()?,
                        OpType::GELU => current.gelu()?,
                        OpType::Pow(power) => current.pow_scalar(*power)?,
                        OpType::Sqrt => current.sqrt()?,
                        OpType::Log => current.log()?,
                        OpType::Exp => current.exp()?,
                        unsupported => {
                            return Err(TrustformersError::tensor_op_error(
                                &format!(
                                    "{:?} cannot appear in a fused elementwise chain",
                                    unsupported
                                ),
                                "eval_fused_elementwise",
                            ));
                        },
                    };
                }
                Ok(current)
            },
        }
    }

    /// Drop nodes no longer reachable from the root.
    fn prune_unreachable_nodes(&mut self) {
        let mut reachable = std::collections::HashSet::new();
        let mut stack = vec![self.root];
        while let Some(id) = stack.pop() {
            if !reachable.insert(id) {
                continue;
            }
            if let Some(node) = self.nodes.get(&id) {
                stack.extend(node.operands.iter().copied());
            }
        }
        self.nodes.retain(|id, _| reachable.contains(id));
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
            OpType::FusedElementwise(ops) => Self::eval_fused_elementwise(&operands[0], ops),
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
                OpType::Sqrt => format!("sqrt({})", operand_strs[0]),
                OpType::Log => format!("log({})", operand_strs[0]),
                OpType::Exp => format!("exp({})", operand_strs[0]),
                OpType::Pow(exponent) => format!("pow({}, {})", operand_strs[0], exponent),
                // Render a fused chain as the nested calls it replaced, so the
                // printed expression stays readable after `optimize_fusion`.
                OpType::FusedElementwise(ops) => {
                    let mut rendered =
                        operand_strs.first().cloned().unwrap_or_else(|| "<missing>".to_string());
                    for op in ops {
                        rendered = match op {
                            OpType::ReLU => format!("relu({})", rendered),
                            OpType::Sigmoid => format!("sigmoid({})", rendered),
                            OpType::Tanh => format!("tanh({})", rendered),
                            OpType::GELU => format!("gelu({})", rendered),
                            OpType::Sqrt => format!("sqrt({})", rendered),
                            OpType::Log => format!("log({})", rendered),
                            OpType::Exp => format!("exp({})", rendered),
                            OpType::Pow(exponent) => format!("pow({}, {})", rendered, exponent),
                            other => format!("{:?}({})", other, rendered),
                        };
                    }
                    format!("fused[{}]", rendered)
                },
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

    // ------------------------------------------------------------------
    // Elementwise fusion
    // ------------------------------------------------------------------

    /// Regression test: `optimize_fusion` used to return an unmodified clone
    /// because `fuse_operations` had an empty body. The rewrite must actually
    /// shrink the graph.
    #[test]
    fn optimize_fusion_collapses_a_unary_chain() -> Result<()> {
        let a = Tensor::from_vec(vec![0.25f32, 1.0, 4.0, 9.0], &[4])?;
        // sqrt(exp(relu(x))) -- three fusable unary links over one leaf.
        let expr = TensorExpr::from(&a)?.relu()?.exp()?.sqrt()?;
        assert_eq!(expr.operation_count(), 3);

        let fused = expr.optimize_fusion()?;
        assert_eq!(
            fused.operation_count(),
            1,
            "three unary links must collapse into one fused node"
        );
        assert_eq!(fused.leaf_count(), 1);
        Ok(())
    }

    /// The fused node must carry the ops in application order.
    #[test]
    fn fused_node_records_the_chain_in_application_order() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0f32, 2.0], &[2])?;
        let expr = TensorExpr::from(&a)?.relu()?.exp()?.sqrt()?;
        let fused = expr.optimize_fusion()?;

        let root_op = fused
            .nodes
            .get(&fused.root)
            .map(|node| node.op.clone())
            .expect("root node exists");
        match root_op {
            OpType::FusedElementwise(ops) => {
                assert_eq!(ops, vec![OpType::ReLU, OpType::Exp, OpType::Sqrt]);
            },
            other => panic!("expected a fused node, got {:?}", other),
        }
        Ok(())
    }

    /// Fusion must not change the numbers. Compared against a hand-computed
    /// reference: sqrt(exp(relu(x))) == exp(relu(x)/2).
    #[test]
    fn fused_evaluation_matches_the_unfused_result_and_a_hand_reference() -> Result<()> {
        let inputs = vec![-2.0f32, 0.0, 0.5, 2.0];
        let a = Tensor::from_vec(inputs.clone(), &[4])?;
        let expr = TensorExpr::from(&a)?.relu()?.exp()?.sqrt()?;

        let unfused_ctx = EvalContext {
            hints: OptimizationHints {
                enable_fusion: false,
                ..OptimizationHints::default()
            },
            ..EvalContext::default()
        };
        let unfused = expr.eval_with_context(&unfused_ctx)?.to_vec_f32()?;
        let fused = expr.eval()?.to_vec_f32()?;

        for (index, input) in inputs.iter().enumerate() {
            let expected = (input.max(0.0) / 2.0).exp();
            assert!(
                (fused[index] - expected).abs() < 1e-5,
                "fused[{index}] = {} but sqrt(exp(relu({input}))) = {expected}",
                fused[index]
            );
            assert!(
                (fused[index] - unfused[index]).abs() < 1e-5,
                "fusion changed the result at {index}: {} vs {}",
                fused[index],
                unfused[index]
            );
        }
        Ok(())
    }

    /// `max_fusion_size` must be honoured, and a limit below 2 disables fusion.
    #[test]
    fn max_fusion_size_caps_the_chain() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4])?;
        let expr = TensorExpr::from(&a)?.relu()?.exp()?.sqrt()?.tanh()?;
        assert_eq!(expr.operation_count(), 4);

        let capped = expr.optimize_fusion_with_limit(2)?;
        assert_eq!(
            capped.operation_count(),
            2,
            "a cap of 2 must leave two fused pairs"
        );

        let disabled = expr.optimize_fusion_with_limit(1)?;
        assert_eq!(
            disabled.operation_count(),
            4,
            "a cap below 2 disables fusion"
        );

        // The numbers survive either way.
        let reference = expr.eval()?.to_vec_f32()?;
        for value in capped.eval()?.to_vec_f32()?.iter().zip(reference.iter()) {
            assert!((value.0 - value.1).abs() < 1e-5);
        }
        Ok(())
    }

    /// A single unary op is not a chain and must be left alone.
    #[test]
    fn a_lone_unary_op_is_not_fused() -> Result<()> {
        let a = Tensor::from_vec(vec![-1.0f32, 1.0], &[2])?;
        let expr = TensorExpr::from(&a)?.relu()?;
        let fused = expr.optimize_fusion()?;
        assert_eq!(fused.operation_count(), 1);
        let root_op =
            fused.nodes.get(&fused.root).map(|node| node.op.clone()).expect("root exists");
        assert_eq!(root_op, OpType::ReLU);
        Ok(())
    }

    /// A binary op breaks the chain: `relu(x) * relu(x)` has no 2-link unary run
    /// rooted at the multiply, so nothing may be fused into it.
    #[test]
    fn a_binary_op_breaks_the_fusion_chain() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0f32, -1.0], &[2])?;
        let b = Tensor::from_vec(vec![2.0f32, 3.0], &[2])?;
        let expr = TensorExpr::from(&a)?.relu()?.mul(TensorExpr::from(&b)?.exp()?)?;
        let before = expr.operation_count();
        let fused = expr.optimize_fusion()?;
        assert_eq!(
            fused.operation_count(),
            before,
            "no unary chain of length >= 2 exists here"
        );
        // And the result must still be right: relu([1,-1]) * exp([2,3]).
        let values = fused.eval()?.to_vec_f32()?;
        assert!((values[0] - 2.0f32.exp()).abs() < 1e-4);
        assert!(values[1].abs() < 1e-6);
        Ok(())
    }

    /// Fusion must also leave `f64` results untouched.
    #[test]
    fn fusion_preserves_f64_results() -> Result<()> {
        let a = Tensor::from_vec_with_dtype(vec![0.25f64, 1.0, 4.0], &[3], DType::F64)?;
        let expr = TensorExpr::from(&a)?.sqrt()?.log()?;

        let unfused_ctx = EvalContext {
            hints: OptimizationHints {
                enable_fusion: false,
                ..OptimizationHints::default()
            },
            ..EvalContext::default()
        };
        let unfused = expr.eval_with_context(&unfused_ctx)?;
        let fused = expr.eval()?;

        match (unfused, fused) {
            (Tensor::F64(u), Tensor::F64(f)) => {
                for (a, b) in u.iter().zip(f.iter()) {
                    assert!((a - b).abs() < 1e-12, "{a} vs {b}");
                }
                // ln(sqrt(4)) == ln(2)
                assert!((f[[2]] - std::f64::consts::LN_2).abs() < 1e-12);
            },
            _ => panic!("F64 expected"),
        }
        Ok(())
    }
}
