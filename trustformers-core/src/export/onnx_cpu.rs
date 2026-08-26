//! A real CPU interpreter for a subset of ONNX.
//!
//! [`OnnxGraphExecutor`] executes a parsed [`ONNXModel`] node by node on the CPU in
//! pure Rust. There is no simulation: an operator either has an implementation
//! here, or the graph fails to run with a structured
//! [`ErrorKind::UnsupportedOperation`](crate::errors::ErrorKind::UnsupportedOperation)
//! naming the operator.
//!
//! # Supported operators
//!
//! `Abs`, `Add`, `Cast`, `Concat`, `Constant`, `Div`, `DequantizeLinear`, `Erf`,
//! `Exp`, `Gather`, `Gelu`, `Gemm`, `Identity`, `LayerNormalization`, `Log`,
//! `MatMul`, `Mul`, `Neg`, `Pow`, `QuantizeLinear`, `ReduceMean`, `Relu`,
//! `Reshape`, `Shape`, `Sigmoid`, `Softmax`, `Sqrt`, `Sub`, `Tanh`, `Transpose`,
//! `Unsqueeze`.
//!
//! Binary operators follow NumPy broadcasting. `MatMul` follows the ONNX rule:
//! the last two axes are the matrix, leading axes broadcast.

use super::onnx::{ONNXAttribute, ONNXDataType, ONNXGraph, ONNXModel, ONNXNode, ONNXTensor};
use crate::errors::unsupported_operation;
use crate::tensor::Tensor;
use anyhow::{anyhow, Result};
use scirs2_core::ndarray::{ArrayD, IxDyn};
use std::collections::{HashMap, HashSet};

/// A value flowing through the interpreter.
#[derive(Debug, Clone, PartialEq)]
pub enum CpuTensor {
    /// 32-bit float payload.
    F32 { data: Vec<f32>, shape: Vec<usize> },
    /// 64-bit integer payload (indices, shapes, token ids).
    I64 { data: Vec<i64>, shape: Vec<usize> },
}

impl CpuTensor {
    /// Build a float tensor, checking that the data matches the shape.
    pub fn f32(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(anyhow!(
                "tensor data of {} elements does not match shape {:?}",
                data.len(),
                shape
            ));
        }
        Ok(CpuTensor::F32 { data, shape })
    }

    /// Build an integer tensor, checking that the data matches the shape.
    pub fn i64(data: Vec<i64>, shape: Vec<usize>) -> Result<Self> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(anyhow!(
                "tensor data of {} elements does not match shape {:?}",
                data.len(),
                shape
            ));
        }
        Ok(CpuTensor::I64 { data, shape })
    }

    /// The tensor's shape.
    pub fn shape(&self) -> &[usize] {
        match self {
            CpuTensor::F32 { shape, .. } | CpuTensor::I64 { shape, .. } => shape,
        }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            CpuTensor::F32 { data, .. } => data.len(),
            CpuTensor::I64 { data, .. } => data.len(),
        }
    }

    /// Whether the tensor has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow the float payload, or fail if this is an integer tensor.
    pub fn f32_slice(&self) -> Result<&[f32]> {
        match self {
            CpuTensor::F32 { data, .. } => Ok(data),
            CpuTensor::I64 { .. } => Err(anyhow!("expected a float tensor, found int64")),
        }
    }

    /// Borrow the integer payload, or fail if this is a float tensor.
    pub fn i64_slice(&self) -> Result<&[i64]> {
        match self {
            CpuTensor::I64 { data, .. } => Ok(data),
            CpuTensor::F32 { .. } => Err(anyhow!("expected an int64 tensor, found float")),
        }
    }

    /// Copy the payload as floats, converting integers if needed.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            CpuTensor::F32 { data, .. } => data.clone(),
            CpuTensor::I64 { data, .. } => data.iter().map(|&v| v as f32).collect(),
        }
    }

    /// Copy the payload as integers, truncating floats if needed.
    pub fn to_i64_vec(&self) -> Vec<i64> {
        match self {
            CpuTensor::I64 { data, .. } => data.clone(),
            CpuTensor::F32 { data, .. } => data.iter().map(|&v| v as i64).collect(),
        }
    }

    /// Convert from the crate's public tensor type.
    pub fn from_tensor(tensor: &Tensor) -> Result<Self> {
        let shape = tensor.shape();
        match tensor {
            Tensor::I64(array) => CpuTensor::i64(array.iter().copied().collect(), shape),
            other => CpuTensor::f32(
                other
                    .to_vec_f32()
                    .map_err(|e| anyhow!("input tensor cannot be read as f32: {e}"))?,
                shape,
            ),
        }
    }

    /// Convert into the crate's public tensor type.
    pub fn into_tensor(self) -> Result<Tensor> {
        match self {
            CpuTensor::F32 { data, shape } => {
                let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
                    .map_err(|e| anyhow!("output tensor shape mismatch: {e}"))?;
                Ok(Tensor::F32(array))
            },
            CpuTensor::I64 { data, shape } => {
                let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
                    .map_err(|e| anyhow!("output tensor shape mismatch: {e}"))?;
                Ok(Tensor::I64(array))
            },
        }
    }
}

/// Build a [`CpuTensor`] from an ONNX initializer.
pub fn tensor_from_initializer(initializer: &ONNXTensor) -> Result<CpuTensor> {
    let shape: Vec<usize> = initializer
        .dims
        .iter()
        .map(|&d| {
            if d < 0 {
                Err(anyhow!(
                    "initializer '{}' has a negative dimension {d}",
                    initializer.name
                ))
            } else {
                Ok(d as usize)
            }
        })
        .collect::<Result<_>>()?;
    let count: usize = shape.iter().product();
    let raw = &initializer.raw_data;

    match initializer.data_type {
        ONNXDataType::Float => {
            expect_bytes(initializer, raw.len(), count * 4)?;
            CpuTensor::f32(
                raw.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                shape,
            )
        },
        ONNXDataType::Double => {
            expect_bytes(initializer, raw.len(), count * 8)?;
            CpuTensor::f32(
                raw.chunks_exact(8)
                    .map(|c| {
                        f64::from_le_bytes([
                            c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                        ]) as f32
                    })
                    .collect(),
                shape,
            )
        },
        ONNXDataType::Float16 => {
            expect_bytes(initializer, raw.len(), count * 2)?;
            CpuTensor::f32(
                raw.chunks_exact(2)
                    .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect(),
                shape,
            )
        },
        ONNXDataType::Int64 => {
            expect_bytes(initializer, raw.len(), count * 8)?;
            CpuTensor::i64(
                raw.chunks_exact(8)
                    .map(|c| {
                        i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                    })
                    .collect(),
                shape,
            )
        },
        ONNXDataType::Int32 => {
            expect_bytes(initializer, raw.len(), count * 4)?;
            CpuTensor::i64(
                raw.chunks_exact(4)
                    .map(|c| i64::from(i32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                    .collect(),
                shape,
            )
        },
        ONNXDataType::Int8 => {
            expect_bytes(initializer, raw.len(), count)?;
            CpuTensor::i64(raw.iter().map(|&b| i64::from(b as i8)).collect(), shape)
        },
        ONNXDataType::UInt8 => {
            expect_bytes(initializer, raw.len(), count)?;
            CpuTensor::i64(raw.iter().map(|&b| i64::from(b)).collect(), shape)
        },
        other => Err(unsupported_operation(
            format!("ONNX initializer of type {other:?}"),
            "TrustformeRS decodes FLOAT, FLOAT16, DOUBLE, INT8, UINT8, INT32 and INT64 initializers",
        )
        .into()),
    }
}

fn expect_bytes(initializer: &ONNXTensor, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(anyhow!(
            "initializer '{}' declares {:?} with dims {:?} ({expected} bytes) but carries {actual}",
            initializer.name,
            initializer.data_type,
            initializer.dims
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Executor
// ---------------------------------------------------------------------------

/// Executes a parsed ONNX graph on the CPU.
#[derive(Debug, Clone)]
pub struct OnnxGraphExecutor {
    model: ONNXModel,
    initializers: HashMap<String, CpuTensor>,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl OnnxGraphExecutor {
    /// Prepare a model for execution, decoding all its initializers.
    pub fn new(model: ONNXModel) -> Result<Self> {
        let mut initializers = HashMap::with_capacity(model.graph.initializers.len());
        for initializer in &model.graph.initializers {
            initializers.insert(
                initializer.name.clone(),
                tensor_from_initializer(initializer)?,
            );
        }

        let initializer_names: HashSet<&str> =
            model.graph.initializers.iter().map(|t| t.name.as_str()).collect();

        // A graph input that is also an initializer is a default value, not a
        // value the caller has to supply.
        let input_names: Vec<String> = model
            .graph
            .inputs
            .iter()
            .filter(|value| !initializer_names.contains(value.name.as_str()))
            .map(|value| value.name.clone())
            .collect();
        let output_names: Vec<String> =
            model.graph.outputs.iter().map(|value| value.name.clone()).collect();

        Ok(Self {
            model,
            initializers,
            input_names,
            output_names,
        })
    }

    /// The graph's parsed representation.
    pub fn model(&self) -> &ONNXModel {
        &self.model
    }

    /// The graph, for convenience.
    pub fn graph(&self) -> &ONNXGraph {
        &self.model.graph
    }

    /// Names the caller must supply to [`Self::run`].
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Names [`Self::run`] returns.
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// Report which operators in this graph have no implementation here.
    pub fn unsupported_operators(&self) -> Vec<String> {
        let mut missing: Vec<String> = self
            .model
            .graph
            .nodes
            .iter()
            .filter(|node| !is_supported_op(&node.op_type))
            .map(|node| node.op_type.clone())
            .collect();
        missing.sort();
        missing.dedup();
        missing
    }

    /// Execute the graph.
    ///
    /// Nodes are executed in the order they appear, which valid ONNX guarantees is
    /// topological.
    pub fn run(&self, inputs: HashMap<String, CpuTensor>) -> Result<HashMap<String, CpuTensor>> {
        for name in &self.input_names {
            if !inputs.contains_key(name) {
                return Err(anyhow!("missing required input '{name}'"));
            }
        }
        for name in inputs.keys() {
            if !self.input_names.contains(name) {
                return Err(anyhow!("unknown input '{name}'"));
            }
        }

        let mut env: HashMap<String, CpuTensor> = self.initializers.clone();
        env.extend(inputs);

        for node in &self.model.graph.nodes {
            let results = eval_node(node, &env)?;
            if results.len() != node.outputs.len() {
                return Err(anyhow!(
                    "node '{}' ({}) produced {} outputs but declares {}",
                    node.name,
                    node.op_type,
                    results.len(),
                    node.outputs.len()
                ));
            }
            for (name, value) in node.outputs.iter().zip(results) {
                env.insert(name.clone(), value);
            }
        }

        let mut outputs = HashMap::with_capacity(self.output_names.len());
        for name in &self.output_names {
            let value = env
                .remove(name)
                .ok_or_else(|| anyhow!("graph output '{name}' was never produced by any node"))?;
            outputs.insert(name.clone(), value);
        }
        Ok(outputs)
    }
}

/// Evaluate a single node against an environment of already-computed values.
///
/// Exposed so that graph transformations (constant folding, for example) can reuse
/// the very same kernels the interpreter runs, instead of a second, divergent
/// implementation.
pub fn eval_node(node: &ONNXNode, env: &HashMap<String, CpuTensor>) -> Result<Vec<CpuTensor>> {
    {
        let input = |index: usize| -> Result<&CpuTensor> {
            let name = node.inputs.get(index).ok_or_else(|| {
                anyhow!(
                    "node '{}' ({}) needs input #{index} but has {}",
                    node.name,
                    node.op_type,
                    node.inputs.len()
                )
            })?;
            env.get(name).ok_or_else(|| {
                anyhow!(
                    "node '{}' ({}) consumes '{name}', which is not defined",
                    node.name,
                    node.op_type
                )
            })
        };
        let optional = |index: usize| -> Option<&CpuTensor> {
            node.inputs
                .get(index)
                .filter(|name| !name.is_empty())
                .and_then(|name| env.get(name))
        };

        let result = match node.op_type.as_str() {
            "Add" => vec![binary_op(input(0)?, input(1)?, |a, b| a + b)?],
            "Sub" => vec![binary_op(input(0)?, input(1)?, |a, b| a - b)?],
            "Mul" => vec![binary_op(input(0)?, input(1)?, |a, b| a * b)?],
            "Div" => vec![binary_op(input(0)?, input(1)?, |a, b| a / b)?],
            "Pow" => vec![binary_op(input(0)?, input(1)?, |a, b| a.powf(b))?],
            "Relu" => vec![unary_op(input(0)?, |v| v.max(0.0))?],
            "Sigmoid" => vec![unary_op(input(0)?, |v| 1.0 / (1.0 + (-v).exp()))?],
            "Tanh" => vec![unary_op(input(0)?, f32::tanh)?],
            "Exp" => vec![unary_op(input(0)?, f32::exp)?],
            "Log" => vec![unary_op(input(0)?, f32::ln)?],
            "Sqrt" => vec![unary_op(input(0)?, f32::sqrt)?],
            "Neg" => vec![unary_op(input(0)?, |v| -v)?],
            "Abs" => vec![unary_op(input(0)?, f32::abs)?],
            "Erf" => vec![unary_op(input(0)?, erf)?],
            "Gelu" => {
                let approximate = string_attr(node, "approximate").unwrap_or("none");
                match approximate {
                    "tanh" => vec![unary_op(input(0)?, gelu_tanh)?],
                    "none" => vec![unary_op(input(0)?, gelu_exact)?],
                    other => {
                        return Err(anyhow!(
                            "Gelu attribute approximate='{other}' is not one of 'none'/'tanh'"
                        ))
                    },
                }
            },
            "Identity" => vec![input(0)?.clone()],
            "MatMul" => vec![matmul(input(0)?, input(1)?)?],
            "Gemm" => {
                let alpha = float_attr(node, "alpha").unwrap_or(1.0);
                let beta = float_attr(node, "beta").unwrap_or(1.0);
                let trans_a = int_attr(node, "transA").unwrap_or(0) != 0;
                let trans_b = int_attr(node, "transB").unwrap_or(0) != 0;
                vec![gemm(
                    input(0)?,
                    input(1)?,
                    optional(2),
                    alpha,
                    beta,
                    trans_a,
                    trans_b,
                )?]
            },
            "Softmax" => {
                let axis = int_attr(node, "axis").unwrap_or(-1);
                vec![softmax(input(0)?, axis)?]
            },
            "LayerNormalization" => {
                let axis = int_attr(node, "axis").unwrap_or(-1);
                let epsilon = float_attr(node, "epsilon").unwrap_or(1e-5);
                vec![layer_norm(
                    input(0)?,
                    input(1)?,
                    optional(2),
                    axis,
                    epsilon,
                )?]
            },
            "Reshape" => vec![reshape(input(0)?, input(1)?)?],
            "Transpose" => {
                let perm = ints_attr(node, "perm");
                vec![transpose(input(0)?, perm)?]
            },
            "Concat" => {
                let axis = int_attr(node, "axis").ok_or_else(|| {
                    anyhow!(
                        "Concat node '{}' has no required 'axis' attribute",
                        node.name
                    )
                })?;
                let values: Vec<&CpuTensor> =
                    (0..node.inputs.len()).map(input).collect::<Result<Vec<_>>>()?;
                vec![concat(&values, axis)?]
            },
            "Gather" => {
                let axis = int_attr(node, "axis").unwrap_or(0);
                vec![gather(input(0)?, input(1)?, axis)?]
            },
            "ReduceMean" => {
                let axes = ints_attr(node, "axes").or_else(|| optional(1).map(|t| t.to_i64_vec()));
                let keepdims = int_attr(node, "keepdims").unwrap_or(1) != 0;
                vec![reduce_mean(input(0)?, axes, keepdims)?]
            },
            "Shape" => {
                let shape = input(0)?.shape();
                vec![CpuTensor::i64(
                    shape.iter().map(|&d| d as i64).collect(),
                    vec![shape.len()],
                )?]
            },
            "Unsqueeze" => {
                let axes = ints_attr(node, "axes")
                    .or_else(|| optional(1).map(|t| t.to_i64_vec()))
                    .ok_or_else(|| anyhow!("Unsqueeze node '{}' has no axes", node.name))?;
                vec![unsqueeze(input(0)?, &axes)?]
            },
            "Cast" => {
                let to = int_attr(node, "to")
                    .ok_or_else(|| anyhow!("Cast node '{}' has no 'to' attribute", node.name))?;
                vec![cast(input(0)?, ONNXDataType::from_i32(to as i32)?)?]
            },
            "Constant" => vec![constant(node)?],
            "QuantizeLinear" => {
                vec![quantize_linear(input(0)?, input(1)?, optional(2))?]
            },
            "DequantizeLinear" => {
                let axis = int_attr(node, "axis").unwrap_or(1);
                vec![dequantize_linear(input(0)?, input(1)?, optional(2), axis)?]
            },
            other => {
                return Err(unsupported_operation(
                    format!("ONNX operator '{other}'"),
                    format!(
                        "the TrustformeRS CPU interpreter (node '{}'); supported operators are: {}",
                        node.name,
                        SUPPORTED_OPS.join(", ")
                    ),
                )
                .into())
            },
        };

        Ok(result)
    }
}

/// Build an ONNX initializer from an interpreter value.
pub fn initializer_from_tensor(name: &str, value: &CpuTensor) -> ONNXTensor {
    let dims: Vec<i64> = value.shape().iter().map(|&d| d as i64).collect();
    match value {
        CpuTensor::F32 { data, .. } => ONNXTensor {
            name: name.to_string(),
            data_type: ONNXDataType::Float,
            dims,
            raw_data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        },
        CpuTensor::I64 { data, .. } => ONNXTensor {
            name: name.to_string(),
            data_type: ONNXDataType::Int64,
            dims,
            raw_data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        },
    }
}

/// Operators the interpreter implements.
pub const SUPPORTED_OPS: &[&str] = &[
    "Abs",
    "Add",
    "Cast",
    "Concat",
    "Constant",
    "DequantizeLinear",
    "Div",
    "Erf",
    "Exp",
    "Gather",
    "Gelu",
    "Gemm",
    "Identity",
    "LayerNormalization",
    "Log",
    "MatMul",
    "Mul",
    "Neg",
    "Pow",
    "QuantizeLinear",
    "ReduceMean",
    "Relu",
    "Reshape",
    "Shape",
    "Sigmoid",
    "Softmax",
    "Sqrt",
    "Sub",
    "Tanh",
    "Transpose",
    "Unsqueeze",
];

/// Whether the interpreter implements `op_type`.
pub fn is_supported_op(op_type: &str) -> bool {
    SUPPORTED_OPS.contains(&op_type)
}

// ---------------------------------------------------------------------------
// Attribute helpers
// ---------------------------------------------------------------------------

fn int_attr(node: &ONNXNode, name: &str) -> Option<i64> {
    match node.attributes.get(name) {
        Some(ONNXAttribute::Int(value)) => Some(*value),
        _ => None,
    }
}

fn float_attr(node: &ONNXNode, name: &str) -> Option<f32> {
    match node.attributes.get(name) {
        Some(ONNXAttribute::Float(value)) => Some(*value),
        Some(ONNXAttribute::Int(value)) => Some(*value as f32),
        _ => None,
    }
}

fn ints_attr(node: &ONNXNode, name: &str) -> Option<Vec<i64>> {
    match node.attributes.get(name) {
        Some(ONNXAttribute::Ints(values)) => Some(values.clone()),
        Some(ONNXAttribute::Int(value)) => Some(vec![*value]),
        _ => None,
    }
}

fn string_attr<'a>(node: &'a ONNXNode, name: &str) -> Option<&'a str> {
    match node.attributes.get(name) {
        Some(ONNXAttribute::String(value)) => Some(value.as_str()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Numeric kernels
// ---------------------------------------------------------------------------

/// Row-major strides for `shape`.
fn strides_for(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for index in (0..shape.len().saturating_sub(1)).rev() {
        strides[index] = strides[index + 1] * shape[index + 1];
    }
    strides
}

/// NumPy broadcast of two shapes.
fn broadcast_shapes(left: &[usize], right: &[usize]) -> Result<Vec<usize>> {
    let rank = left.len().max(right.len());
    let mut result = vec![0usize; rank];
    for index in 0..rank {
        let l = if index < rank - left.len() { 1 } else { left[index - (rank - left.len())] };
        let r = if index < rank - right.len() { 1 } else { right[index - (rank - right.len())] };
        result[index] = match (l, r) {
            (a, b) if a == b => a,
            (1, b) => b,
            (a, 1) => a,
            _ => {
                return Err(anyhow!(
                    "shapes {left:?} and {right:?} are not broadcast compatible"
                ))
            },
        };
    }
    Ok(result)
}

/// Index into `data` shaped `shape` using a broadcast coordinate of `out_shape`.
fn broadcast_index(coord: &[usize], shape: &[usize], strides: &[usize]) -> usize {
    let offset = coord.len() - shape.len();
    let mut index = 0;
    for (axis, &dim) in shape.iter().enumerate() {
        let position = if dim == 1 { 0 } else { coord[axis + offset] };
        index += position * strides[axis];
    }
    index
}

fn next_coord(coord: &mut [usize], shape: &[usize]) -> bool {
    for axis in (0..shape.len()).rev() {
        coord[axis] += 1;
        if coord[axis] < shape[axis] {
            return true;
        }
        coord[axis] = 0;
    }
    false
}

fn binary_op<F: Fn(f32, f32) -> f32>(
    left: &CpuTensor,
    right: &CpuTensor,
    op: F,
) -> Result<CpuTensor> {
    let out_shape = broadcast_shapes(left.shape(), right.shape())?;
    let count: usize = out_shape.iter().product();
    let left_data = left.to_f32_vec();
    let right_data = right.to_f32_vec();
    let left_strides = strides_for(left.shape());
    let right_strides = strides_for(right.shape());

    let mut out = Vec::with_capacity(count);
    if out_shape.is_empty() {
        out.push(op(left_data[0], right_data[0]));
    } else {
        let mut coord = vec![0usize; out_shape.len()];
        loop {
            let l = left_data[broadcast_index(&coord, left.shape(), &left_strides)];
            let r = right_data[broadcast_index(&coord, right.shape(), &right_strides)];
            out.push(op(l, r));
            if !next_coord(&mut coord, &out_shape) {
                break;
            }
        }
    }

    CpuTensor::f32(out, out_shape)
}

fn unary_op<F: Fn(f32) -> f32>(value: &CpuTensor, op: F) -> Result<CpuTensor> {
    let data = value.to_f32_vec().into_iter().map(op).collect();
    CpuTensor::f32(data, value.shape().to_vec())
}

/// Abramowitz & Stegun 7.1.26 approximation of `erf`, accurate to ~1.5e-7.
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

fn gelu_exact(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_56;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x)).tanh())
}

/// ONNX `MatMul`: the last two axes form the matrix, leading axes broadcast.
fn matmul(left: &CpuTensor, right: &CpuTensor) -> Result<CpuTensor> {
    let mut left_shape = left.shape().to_vec();
    let mut right_shape = right.shape().to_vec();
    let left_was_vector = left_shape.len() == 1;
    let right_was_vector = right_shape.len() == 1;

    if left_was_vector {
        left_shape.insert(0, 1);
    }
    if right_was_vector {
        right_shape.push(1);
    }
    if left_shape.len() < 2 || right_shape.len() < 2 {
        return Err(anyhow!("MatMul needs at least 1-D operands"));
    }

    let m = left_shape[left_shape.len() - 2];
    let k = left_shape[left_shape.len() - 1];
    let k_right = right_shape[right_shape.len() - 2];
    let n = right_shape[right_shape.len() - 1];
    if k != k_right {
        return Err(anyhow!(
            "MatMul inner dimensions disagree: {:?} x {:?}",
            left.shape(),
            right.shape()
        ));
    }

    let batch_shape = broadcast_shapes(
        &left_shape[..left_shape.len() - 2],
        &right_shape[..right_shape.len() - 2],
    )?;
    let batch_count: usize = batch_shape.iter().product::<usize>().max(1);

    let left_data = left.to_f32_vec();
    let right_data = right.to_f32_vec();
    let left_batch_shape = &left_shape[..left_shape.len() - 2];
    let right_batch_shape = &right_shape[..right_shape.len() - 2];
    let left_batch_strides = strides_for(left_batch_shape);
    let right_batch_strides = strides_for(right_batch_shape);

    let mut out = vec![0.0f32; batch_count * m * n];
    let mut coord = vec![0usize; batch_shape.len()];
    for batch in 0..batch_count {
        let left_batch = if left_batch_shape.is_empty() {
            0
        } else {
            broadcast_index(&coord, left_batch_shape, &left_batch_strides)
        };
        let right_batch = if right_batch_shape.is_empty() {
            0
        } else {
            broadcast_index(&coord, right_batch_shape, &right_batch_strides)
        };

        let left_base = left_batch * m * k;
        let right_base = right_batch * k * n;
        let out_base = batch * m * n;

        for row in 0..m {
            for inner in 0..k {
                let a = left_data[left_base + row * k + inner];
                if a == 0.0 {
                    continue;
                }
                for column in 0..n {
                    out[out_base + row * n + column] +=
                        a * right_data[right_base + inner * n + column];
                }
            }
        }

        if !batch_shape.is_empty() && !next_coord(&mut coord, &batch_shape) {
            break;
        }
    }

    let mut out_shape = batch_shape;
    out_shape.push(m);
    out_shape.push(n);
    if right_was_vector {
        out_shape.pop();
    }
    if left_was_vector {
        let index = out_shape.len().saturating_sub(if right_was_vector { 1 } else { 2 });
        out_shape.remove(index);
    }

    CpuTensor::f32(out, out_shape)
}

#[allow(clippy::too_many_arguments)]
fn gemm(
    a: &CpuTensor,
    b: &CpuTensor,
    c: Option<&CpuTensor>,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
) -> Result<CpuTensor> {
    let a = if trans_a { transpose(a, None)? } else { a.clone() };
    let b = if trans_b { transpose(b, None)? } else { b.clone() };
    if a.shape().len() != 2 || b.shape().len() != 2 {
        return Err(anyhow!("Gemm operands must be 2-D"));
    }

    let product = matmul(&a, &b)?;
    let scaled = unary_op(&product, |v| v * alpha)?;
    match c {
        Some(bias) => {
            let scaled_bias = unary_op(bias, |v| v * beta)?;
            binary_op(&scaled, &scaled_bias, |x, y| x + y)
        },
        None => Ok(scaled),
    }
}

/// Resolve a possibly negative axis against `rank`.
fn resolve_axis(axis: i64, rank: usize) -> Result<usize> {
    let rank_i64 = rank as i64;
    let resolved = if axis < 0 { axis + rank_i64 } else { axis };
    if resolved < 0 || resolved >= rank_i64 {
        return Err(anyhow!("axis {axis} is out of range for rank {rank}"));
    }
    Ok(resolved as usize)
}

fn softmax(value: &CpuTensor, axis: i64) -> Result<CpuTensor> {
    let shape = value.shape().to_vec();
    let axis = resolve_axis(axis, shape.len())?;
    let mut data = value.to_f32_vec();

    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();

    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let base = outer_index * axis_len * inner + inner_index;
            let mut max = f32::NEG_INFINITY;
            for step in 0..axis_len {
                max = max.max(data[base + step * inner]);
            }
            let mut sum = 0.0f32;
            for step in 0..axis_len {
                let exponent = (data[base + step * inner] - max).exp();
                data[base + step * inner] = exponent;
                sum += exponent;
            }
            if sum != 0.0 {
                for step in 0..axis_len {
                    data[base + step * inner] /= sum;
                }
            }
        }
    }

    CpuTensor::f32(data, shape)
}

fn layer_norm(
    value: &CpuTensor,
    scale: &CpuTensor,
    bias: Option<&CpuTensor>,
    axis: i64,
    epsilon: f32,
) -> Result<CpuTensor> {
    let shape = value.shape().to_vec();
    let axis = resolve_axis(axis, shape.len())?;
    let normalized: usize = shape[axis..].iter().product();
    let outer: usize = shape[..axis].iter().product();

    let scale_data = scale.to_f32_vec();
    if scale_data.len() != normalized {
        return Err(anyhow!(
            "LayerNormalization scale has {} elements but {normalized} are normalized",
            scale_data.len()
        ));
    }
    let bias_data = match bias {
        Some(bias) => {
            let data = bias.to_f32_vec();
            if data.len() != normalized {
                return Err(anyhow!(
                    "LayerNormalization bias has {} elements but {normalized} are normalized",
                    data.len()
                ));
            }
            Some(data)
        },
        None => None,
    };

    let mut data = value.to_f32_vec();
    for group in 0..outer {
        let base = group * normalized;
        let slice = &mut data[base..base + normalized];
        let mean = slice.iter().sum::<f32>() / normalized as f32;
        let variance =
            slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / normalized as f32;
        let inv_std = 1.0 / (variance + epsilon).sqrt();
        for (index, element) in slice.iter_mut().enumerate() {
            let normalized_value = (*element - mean) * inv_std;
            *element =
                normalized_value * scale_data[index] + bias_data.as_ref().map_or(0.0, |b| b[index]);
        }
    }

    CpuTensor::f32(data, shape)
}

fn reshape(value: &CpuTensor, shape: &CpuTensor) -> Result<CpuTensor> {
    let requested = shape.to_i64_vec();
    let original = value.shape();
    let total: usize = original.iter().product();

    let mut resolved: Vec<usize> = Vec::with_capacity(requested.len());
    let mut inferred_axis: Option<usize> = None;
    let mut known: usize = 1;
    for (axis, &dim) in requested.iter().enumerate() {
        match dim {
            -1 => {
                if inferred_axis.is_some() {
                    return Err(anyhow!("Reshape allows at most one -1 dimension"));
                }
                inferred_axis = Some(axis);
                resolved.push(1);
            },
            0 => {
                let copied = *original
                    .get(axis)
                    .ok_or_else(|| anyhow!("Reshape 0 at axis {axis} has no source dimension"))?;
                known *= copied;
                resolved.push(copied);
            },
            d if d > 0 => {
                known *= d as usize;
                resolved.push(d as usize);
            },
            d => return Err(anyhow!("Reshape dimension {d} is invalid")),
        }
    }

    if let Some(axis) = inferred_axis {
        if known == 0 || !total.is_multiple_of(known) {
            return Err(anyhow!(
                "Reshape cannot infer a dimension: {total} elements into {requested:?}"
            ));
        }
        resolved[axis] = total / known;
    } else if known != total {
        return Err(anyhow!(
            "Reshape target {requested:?} holds {known} elements but the input has {total}"
        ));
    }

    match value {
        CpuTensor::F32 { data, .. } => CpuTensor::f32(data.clone(), resolved),
        CpuTensor::I64 { data, .. } => CpuTensor::i64(data.clone(), resolved),
    }
}

fn transpose(value: &CpuTensor, perm: Option<Vec<i64>>) -> Result<CpuTensor> {
    let shape = value.shape().to_vec();
    let rank = shape.len();
    let perm: Vec<usize> = match perm {
        Some(perm) => {
            if perm.len() != rank {
                return Err(anyhow!(
                    "Transpose perm has {} entries for a rank-{rank} tensor",
                    perm.len()
                ));
            }
            perm.into_iter().map(|axis| resolve_axis(axis, rank)).collect::<Result<_>>()?
        },
        None => (0..rank).rev().collect(),
    };

    let mut seen = vec![false; rank];
    for &axis in &perm {
        if seen[axis] {
            return Err(anyhow!("Transpose perm repeats axis {axis}"));
        }
        seen[axis] = true;
    }

    let out_shape: Vec<usize> = perm.iter().map(|&axis| shape[axis]).collect();
    let in_strides = strides_for(&shape);
    let count: usize = shape.iter().product();

    let permute = |source: &[f32]| -> Vec<f32> {
        let mut out = Vec::with_capacity(count);
        let mut coord = vec![0usize; rank];
        if rank == 0 {
            return source.to_vec();
        }
        loop {
            let mut index = 0;
            for (axis, &position) in coord.iter().enumerate() {
                index += position * in_strides[perm[axis]];
            }
            out.push(source[index]);
            if !next_coord(&mut coord, &out_shape) {
                break;
            }
        }
        out
    };

    match value {
        CpuTensor::F32 { data, .. } => CpuTensor::f32(permute(data), out_shape),
        CpuTensor::I64 { data, .. } => {
            let as_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
            let permuted = permute(&as_f32);
            CpuTensor::i64(permuted.into_iter().map(|v| v as i64).collect(), out_shape)
        },
    }
}

fn concat(values: &[&CpuTensor], axis: i64) -> Result<CpuTensor> {
    let first = values.first().ok_or_else(|| anyhow!("Concat needs at least one input"))?;
    let rank = first.shape().len();
    let axis = resolve_axis(axis, rank)?;

    let mut out_shape = first.shape().to_vec();
    out_shape[axis] = 0;
    for value in values {
        if value.shape().len() != rank {
            return Err(anyhow!("Concat inputs must all have rank {rank}"));
        }
        for (index, (&expected, &actual)) in out_shape.iter().zip(value.shape().iter()).enumerate()
        {
            if index != axis && expected != actual {
                return Err(anyhow!(
                    "Concat inputs disagree on axis {index}: {expected} vs {actual}"
                ));
            }
        }
        out_shape[axis] += value.shape()[axis];
    }

    let outer: usize = out_shape[..axis].iter().product();
    let inner: usize = out_shape[axis + 1..].iter().product();

    let mut out = Vec::with_capacity(out_shape.iter().product());
    for outer_index in 0..outer {
        for value in values {
            let block = value.shape()[axis] * inner;
            let data = value.to_f32_vec();
            out.extend_from_slice(&data[outer_index * block..(outer_index + 1) * block]);
        }
    }

    CpuTensor::f32(out, out_shape)
}

fn gather(value: &CpuTensor, indices: &CpuTensor, axis: i64) -> Result<CpuTensor> {
    let shape = value.shape().to_vec();
    let axis = resolve_axis(axis, shape.len())?;
    let axis_len = shape[axis];
    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product();

    let index_values = indices.to_i64_vec();
    let mut resolved = Vec::with_capacity(index_values.len());
    for &index in &index_values {
        let normalized = if index < 0 { index + axis_len as i64 } else { index };
        if normalized < 0 || normalized >= axis_len as i64 {
            return Err(anyhow!(
                "Gather index {index} is out of range for axis {axis} of length {axis_len}"
            ));
        }
        resolved.push(normalized as usize);
    }

    let data = value.to_f32_vec();
    let mut out = Vec::with_capacity(outer * resolved.len() * inner);
    for outer_index in 0..outer {
        for &index in &resolved {
            let base = (outer_index * axis_len + index) * inner;
            out.extend_from_slice(&data[base..base + inner]);
        }
    }

    let mut out_shape = shape[..axis].to_vec();
    out_shape.extend_from_slice(indices.shape());
    out_shape.extend_from_slice(&shape[axis + 1..]);

    CpuTensor::f32(out, out_shape)
}

fn reduce_mean(value: &CpuTensor, axes: Option<Vec<i64>>, keepdims: bool) -> Result<CpuTensor> {
    let shape = value.shape().to_vec();
    let rank = shape.len();
    let axes: Vec<usize> = match axes {
        Some(axes) => {
            axes.into_iter().map(|axis| resolve_axis(axis, rank)).collect::<Result<_>>()?
        },
        None => (0..rank).collect(),
    };
    let reduced: HashSet<usize> = axes.iter().copied().collect();

    let out_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(
            |(axis, &dim)| {
                if reduced.contains(&axis) {
                    keepdims.then_some(1)
                } else {
                    Some(dim)
                }
            },
        )
        .collect();

    let data = value.to_f32_vec();
    let out_count: usize = out_shape.iter().product::<usize>().max(1);
    let mut sums = vec![0.0f32; out_count];
    let mut counts = vec![0usize; out_count];

    let kept_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter(|(axis, _)| !reduced.contains(axis))
        .map(|(_, &dim)| dim)
        .collect();
    let kept_strides = strides_for(&kept_shape);

    let mut coord = vec![0usize; rank];
    for element in data.iter() {
        let mut out_index = 0usize;
        let mut kept_axis = 0usize;
        for (axis, &position) in coord.iter().enumerate() {
            if !reduced.contains(&axis) {
                out_index += position * kept_strides[kept_axis];
                kept_axis += 1;
            }
        }
        sums[out_index] += element;
        counts[out_index] += 1;
        if rank == 0 || !next_coord(&mut coord, &shape) {
            break;
        }
    }

    let out: Vec<f32> = sums
        .iter()
        .zip(counts.iter())
        .map(|(sum, count)| if *count == 0 { 0.0 } else { sum / *count as f32 })
        .collect();

    CpuTensor::f32(out, out_shape)
}

fn unsqueeze(value: &CpuTensor, axes: &[i64]) -> Result<CpuTensor> {
    let mut shape = value.shape().to_vec();
    let mut sorted: Vec<i64> = axes.to_vec();
    sorted.sort_unstable();
    for &axis in &sorted {
        let rank = shape.len() + 1;
        let resolved = resolve_axis(axis, rank)?;
        shape.insert(resolved, 1);
    }
    match value {
        CpuTensor::F32 { data, .. } => CpuTensor::f32(data.clone(), shape),
        CpuTensor::I64 { data, .. } => CpuTensor::i64(data.clone(), shape),
    }
}

fn cast(value: &CpuTensor, target: ONNXDataType) -> Result<CpuTensor> {
    match target {
        ONNXDataType::Float | ONNXDataType::Double => {
            CpuTensor::f32(value.to_f32_vec(), value.shape().to_vec())
        },
        ONNXDataType::Float16 => CpuTensor::f32(
            value
                .to_f32_vec()
                .into_iter()
                .map(|v| half::f16::from_f32(v).to_f32())
                .collect(),
            value.shape().to_vec(),
        ),
        ONNXDataType::Int64 | ONNXDataType::Int32 | ONNXDataType::Int8 => {
            CpuTensor::i64(value.to_i64_vec(), value.shape().to_vec())
        },
        other => Err(unsupported_operation(
            format!("ONNX Cast to {other:?}"),
            "the TrustformeRS CPU interpreter casts between FLOAT, FLOAT16, DOUBLE, INT8, INT32 \
             and INT64",
        )
        .into()),
    }
}

fn constant(node: &ONNXNode) -> Result<CpuTensor> {
    match node.attributes.get("value") {
        Some(ONNXAttribute::Tensor(tensor)) => tensor_from_initializer(tensor),
        Some(ONNXAttribute::Float(value)) => CpuTensor::f32(vec![*value], vec![]),
        Some(ONNXAttribute::Int(value)) => CpuTensor::i64(vec![*value], vec![]),
        Some(ONNXAttribute::Floats(values)) => CpuTensor::f32(values.clone(), vec![values.len()]),
        Some(ONNXAttribute::Ints(values)) => CpuTensor::i64(values.clone(), vec![values.len()]),
        _ => Err(anyhow!(
            "Constant node '{}' has no 'value' attribute this build understands",
            node.name
        )),
    }
}

/// ONNX `QuantizeLinear` with a scalar scale: `saturate(round(x / scale) + zero_point)`.
fn quantize_linear(
    value: &CpuTensor,
    scale: &CpuTensor,
    zero_point: Option<&CpuTensor>,
) -> Result<CpuTensor> {
    if scale.len() != 1 {
        return Err(unsupported_operation(
            "ONNX QuantizeLinear with a per-axis scale",
            "the TrustformeRS CPU interpreter implements per-tensor (scalar) scales only",
        )
        .into());
    }
    let scale_value = scale.to_f32_vec()[0];
    if scale_value == 0.0 {
        return Err(anyhow!("QuantizeLinear scale must not be zero"));
    }
    let zero = zero_point.map_or(0, |z| z.to_i64_vec().first().copied().unwrap_or(0));

    let data = value
        .to_f32_vec()
        .into_iter()
        .map(|v| ((v / scale_value).round() as i64 + zero).clamp(-128, 127))
        .collect();
    CpuTensor::i64(data, value.shape().to_vec())
}

/// ONNX `DequantizeLinear`: `(x - zero_point) * scale`, per-tensor or per-axis.
fn dequantize_linear(
    value: &CpuTensor,
    scale: &CpuTensor,
    zero_point: Option<&CpuTensor>,
    axis: i64,
) -> Result<CpuTensor> {
    let quantized = value.to_i64_vec();
    let scales = scale.to_f32_vec();
    let zeros = zero_point.map(|z| z.to_i64_vec()).unwrap_or_default();

    if scales.len() == 1 {
        let zero = zeros.first().copied().unwrap_or(0);
        let data = quantized.into_iter().map(|q| (q - zero) as f32 * scales[0]).collect();
        return CpuTensor::f32(data, value.shape().to_vec());
    }

    let shape = value.shape().to_vec();
    let axis = resolve_axis(axis, shape.len())?;
    if scales.len() != shape[axis] {
        return Err(anyhow!(
            "DequantizeLinear has {} scales for axis {axis} of length {}",
            scales.len(),
            shape[axis]
        ));
    }
    let inner: usize = shape[axis + 1..].iter().product();
    let axis_len = shape[axis];

    let data = quantized
        .into_iter()
        .enumerate()
        .map(|(index, q)| {
            let axis_position = (index / inner.max(1)) % axis_len;
            let zero = zeros.get(axis_position).copied().unwrap_or(0);
            (q - zero) as f32 * scales[axis_position]
        })
        .collect();
    CpuTensor::f32(data, shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::onnx::{
        ONNXDimension, ONNXGraph, ONNXModel, ONNXOpsetImport, ONNXTensorShape, ONNXTensorType,
        ONNXTypeInfo, ONNXValueInfo,
    };

    fn value_info(name: &str, dims: &[i64]) -> ONNXValueInfo {
        ONNXValueInfo {
            name: name.to_string(),
            type_info: ONNXTypeInfo {
                tensor_type: ONNXTensorType {
                    elem_type: ONNXDataType::Float,
                    shape: ONNXTensorShape {
                        dims: dims.iter().map(|&d| ONNXDimension::Value(d)).collect(),
                    },
                },
            },
        }
    }

    fn initializer(name: &str, dims: &[i64], values: &[f32]) -> ONNXTensor {
        ONNXTensor {
            name: name.to_string(),
            data_type: ONNXDataType::Float,
            dims: dims.to_vec(),
            raw_data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    fn node(op: &str, name: &str, inputs: &[&str], outputs: &[&str]) -> ONNXNode {
        ONNXNode {
            op_type: op.to_string(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: outputs.iter().map(|s| s.to_string()).collect(),
            attributes: HashMap::new(),
            name: name.to_string(),
        }
    }

    fn wrap(graph: ONNXGraph) -> ONNXModel {
        ONNXModel {
            graph,
            ir_version: 8,
            opset_imports: vec![ONNXOpsetImport {
                domain: String::new(),
                version: 17,
            }],
            producer_name: "test".to_string(),
            producer_version: "0".to_string(),
            model_version: 1,
        }
    }

    #[test]
    fn matmul_matches_a_hand_computed_product() {
        // [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = CpuTensor::f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("a");
        let b = CpuTensor::f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).expect("b");
        let product = matmul(&a, &b).expect("matmul");
        assert_eq!(product.shape(), &[2, 2]);
        assert_eq!(product.f32_slice().expect("f32"), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_broadcasts_leading_axes() {
        let a = CpuTensor::f32((1..=12).map(|v| v as f32).collect(), vec![2, 2, 3]).expect("a");
        let b = CpuTensor::f32((1..=3).map(|v| v as f32).collect(), vec![3, 1]).expect("b");
        let product = matmul(&a, &b).expect("matmul");
        assert_eq!(product.shape(), &[2, 2, 1]);
        // rows: [1,2,3]·[1,2,3] = 14; [4,5,6]·[1,2,3] = 32; etc.
        assert_eq!(product.f32_slice().expect("f32"), &[14.0, 32.0, 50.0, 68.0]);
    }

    #[test]
    fn softmax_sums_to_one_along_the_axis() {
        let x = CpuTensor::f32(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], vec![2, 3]).expect("x");
        let y = softmax(&x, -1).expect("softmax");
        let data = y.f32_slice().expect("f32");
        for row in data.chunks_exact(3) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
        // exp(1),exp(2),exp(3) normalized
        let denominator = 1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp();
        assert!((data[0] - 1.0f32.exp() / denominator).abs() < 1e-6);
    }

    #[test]
    fn layer_norm_matches_the_reference_formula() {
        let x = CpuTensor::f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).expect("x");
        let scale = CpuTensor::f32(vec![1.0; 4], vec![4]).expect("scale");
        let y = layer_norm(&x, &scale, None, -1, 1e-5).expect("layer norm");

        let mean = 2.5f32;
        let variance = ((1.0 - mean).powi(2)
            + (2.0 - mean).powi(2)
            + (3.0 - mean).powi(2)
            + (4.0 - mean).powi(2))
            / 4.0;
        let inv_std = 1.0 / (variance + 1e-5).sqrt();
        let expected: Vec<f32> =
            [1.0f32, 2.0, 3.0, 4.0].iter().map(|v| (v - mean) * inv_std).collect();

        for (actual, expected) in y.f32_slice().expect("f32").iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-5, "{actual} vs {expected}");
        }
    }

    #[test]
    fn erf_and_gelu_match_known_values() {
        assert!((erf(0.0)).abs() < 1e-6);
        assert!((erf(1.0) - 0.842_700_8).abs() < 1e-5);
        assert!((gelu_exact(0.0)).abs() < 1e-6);
        // GELU(1) = 0.5 * (1 + erf(1/sqrt2)) ~= 0.8413447
        assert!((gelu_exact(1.0) - 0.841_344_7).abs() < 1e-4);
    }

    #[test]
    fn transpose_and_reshape_agree_with_numpy_semantics() {
        let x = CpuTensor::f32((0..6).map(|v| v as f32).collect(), vec![2, 3]).expect("x");
        let t = transpose(&x, None).expect("transpose");
        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(t.f32_slice().expect("f32"), &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);

        let shape = CpuTensor::i64(vec![3, -1], vec![2]).expect("shape");
        let r = reshape(&x, &shape).expect("reshape");
        assert_eq!(r.shape(), &[3, 2]);
    }

    #[test]
    fn gather_selects_embedding_rows() {
        let table = CpuTensor::f32((0..12).map(|v| v as f32).collect(), vec![4, 3]).expect("table");
        let indices = CpuTensor::i64(vec![2, 0], vec![2]).expect("indices");
        let rows = gather(&table, &indices, 0).expect("gather");
        assert_eq!(rows.shape(), &[2, 3]);
        assert_eq!(
            rows.f32_slice().expect("f32"),
            &[6.0, 7.0, 8.0, 0.0, 1.0, 2.0]
        );
    }

    #[test]
    fn reduce_mean_over_the_last_axis() {
        let x = CpuTensor::f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("x");
        let y = reduce_mean(&x, Some(vec![-1]), true).expect("reduce");
        assert_eq!(y.shape(), &[2, 1]);
        assert_eq!(y.f32_slice().expect("f32"), &[1.5, 3.5]);
    }

    #[test]
    fn quantize_dequantize_round_trip() {
        let x = CpuTensor::f32(vec![-1.0, 0.0, 0.5, 1.0], vec![4]).expect("x");
        let scale = CpuTensor::f32(vec![0.01], vec![]).expect("scale");
        let q = quantize_linear(&x, &scale, None).expect("quantize");
        let d = dequantize_linear(&q, &scale, None, 1).expect("dequantize");
        for (original, recovered) in
            x.f32_slice().expect("f32").iter().zip(d.f32_slice().expect("f32"))
        {
            assert!(
                (original - recovered).abs() < 0.01,
                "{original} -> {recovered}"
            );
        }
    }

    #[test]
    fn executor_runs_a_two_node_graph() {
        // y = Relu(x @ w) + b
        let graph = ONNXGraph {
            nodes: vec![
                node("MatMul", "mm", &["x", "w"], &["h"]),
                node("Relu", "relu", &["h"], &["r"]),
                node("Add", "add", &["r", "b"], &["y"]),
            ],
            inputs: vec![value_info("x", &[1, 2])],
            outputs: vec![value_info("y", &[1, 2])],
            initializers: vec![
                initializer("w", &[2, 2], &[1.0, -1.0, 0.0, 2.0]),
                initializer("b", &[2], &[0.5, -0.5]),
            ],
            name: "g".to_string(),
        };
        let executor = OnnxGraphExecutor::new(wrap(graph)).expect("executor");
        assert_eq!(executor.input_names(), &["x".to_string()]);
        assert_eq!(executor.output_names(), &["y".to_string()]);

        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            CpuTensor::f32(vec![1.0, 1.0], vec![1, 2]).expect("x"),
        );
        let outputs = executor.run(inputs).expect("run");

        // x @ w = [1*1 + 1*0, 1*-1 + 1*2] = [1, 1]; relu -> [1, 1]; + b -> [1.5, 0.5]
        assert_eq!(
            outputs["y"].f32_slice().expect("f32"),
            &[1.5, 0.5],
            "graph must compute the real result"
        );
    }

    /// The whole point: the output has to depend on the input.
    #[test]
    fn executor_output_varies_with_the_input() {
        let graph = ONNXGraph {
            nodes: vec![node("MatMul", "mm", &["x", "w"], &["y"])],
            inputs: vec![value_info("x", &[1, 2])],
            outputs: vec![value_info("y", &[1, 2])],
            initializers: vec![initializer("w", &[2, 2], &[1.0, 2.0, 3.0, 4.0])],
            name: "g".to_string(),
        };
        let executor = OnnxGraphExecutor::new(wrap(graph)).expect("executor");

        let run = |a: f32, b: f32| {
            let mut inputs = HashMap::new();
            inputs.insert(
                "x".to_string(),
                CpuTensor::f32(vec![a, b], vec![1, 2]).expect("x"),
            );
            executor.run(inputs).expect("run")["y"].f32_slice().expect("f32").to_vec()
        };

        assert_eq!(run(1.0, 0.0), vec![1.0, 2.0]);
        assert_eq!(run(0.0, 1.0), vec![3.0, 4.0]);
        assert_ne!(run(1.0, 0.0), run(0.0, 1.0));
    }

    /// Running the same graph twice with the same input must give the same answer,
    /// which random "simulated inference" never could.
    #[test]
    fn executor_is_deterministic() {
        let graph = ONNXGraph {
            nodes: vec![node("Softmax", "sm", &["x"], &["y"])],
            inputs: vec![value_info("x", &[1, 3])],
            outputs: vec![value_info("y", &[1, 3])],
            initializers: Vec::new(),
            name: "g".to_string(),
        };
        let executor = OnnxGraphExecutor::new(wrap(graph)).expect("executor");

        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            CpuTensor::f32(vec![0.1, 0.2, 0.7], vec![1, 3]).expect("x"),
        );
        let first = executor.run(inputs.clone()).expect("run");
        let second = executor.run(inputs).expect("run");
        assert_eq!(first["y"], second["y"]);
    }

    #[test]
    fn unknown_operators_are_reported_not_faked() {
        let graph = ONNXGraph {
            nodes: vec![node("MultiHeadAttention", "attn", &["x"], &["y"])],
            inputs: vec![value_info("x", &[1, 2])],
            outputs: vec![value_info("y", &[1, 2])],
            initializers: Vec::new(),
            name: "g".to_string(),
        };
        let executor = OnnxGraphExecutor::new(wrap(graph)).expect("executor");
        assert_eq!(
            executor.unsupported_operators(),
            vec!["MultiHeadAttention".to_string()]
        );

        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            CpuTensor::f32(vec![1.0, 2.0], vec![1, 2]).expect("x"),
        );
        let err = executor.run(inputs).expect_err("must not fabricate an answer");
        assert!(err.to_string().contains("MultiHeadAttention"), "{err}");
        assert!(err.to_string().contains("Unsupported operation"), "{err}");
    }

    #[test]
    fn missing_inputs_are_rejected() {
        let graph = ONNXGraph {
            nodes: vec![node("Relu", "relu", &["x"], &["y"])],
            inputs: vec![value_info("x", &[2])],
            outputs: vec![value_info("y", &[2])],
            initializers: Vec::new(),
            name: "g".to_string(),
        };
        let executor = OnnxGraphExecutor::new(wrap(graph)).expect("executor");
        let err = executor.run(HashMap::new()).expect_err("missing input");
        assert!(err.to_string().contains("missing required input"), "{err}");
    }

    #[test]
    fn tensor_conversion_round_trips() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("tensor");
        let cpu = CpuTensor::from_tensor(&tensor).expect("convert");
        assert_eq!(cpu.shape(), &[2, 2]);
        let back = cpu.into_tensor().expect("convert back");
        assert_eq!(back.to_vec_f32().expect("f32"), vec![1.0, 2.0, 3.0, 4.0]);
    }
}
