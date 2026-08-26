//! Real graph transformations on parsed ONNX models.
//!
//! Every function here rewrites the graph it is given and reports what it changed.
//! Nothing copies a file and calls it an optimization.

use super::onnx::{ONNXDataType, ONNXGraph, ONNXNode, ONNXTensor};
use super::onnx_cpu::{
    eval_node, initializer_from_tensor, is_supported_op, tensor_from_initializer, CpuTensor,
};
use super::onnx_runtime::GraphOptimizationLevel;
use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};

/// Smallest float initializer worth quantizing. Below this the `DequantizeLinear`
/// node and its scale cost more bytes than the quantization saves.
const MIN_QUANTIZABLE_ELEMENTS: usize = 32;

/// What [`optimize_graph`] changed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GraphOptimizationStats {
    /// `Identity` nodes eliminated by rewiring their consumers.
    pub identity_nodes_removed: usize,
    /// Nodes whose inputs were all constant and were replaced by an initializer.
    pub constant_folded_nodes: usize,
    /// Initializers no node reads any more.
    pub initializers_removed: usize,
    /// Size of the input file.
    pub bytes_before: usize,
    /// Size of the written file.
    pub bytes_after: usize,
}

/// What [`quantize_onnx_model`] changed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QuantizationStats {
    /// Float initializers replaced by int8 tensors.
    pub tensors_quantized: usize,
    /// Bytes of float initializer data removed.
    pub float_bytes_before: usize,
    /// Bytes of int8 initializer data written in their place.
    pub int8_bytes_after: usize,
    /// Size of the input file.
    pub bytes_before: usize,
    /// Size of the written file.
    pub bytes_after: usize,
}

/// Apply the transformations selected by `level` to `graph` in place.
pub fn optimize_graph(
    graph: &mut ONNXGraph,
    level: GraphOptimizationLevel,
    bytes_before: usize,
) -> Result<GraphOptimizationStats> {
    let mut stats = GraphOptimizationStats {
        bytes_before,
        ..Default::default()
    };

    if level == GraphOptimizationLevel::None {
        return Ok(stats);
    }

    stats.identity_nodes_removed = eliminate_identity_nodes(graph);

    if matches!(
        level,
        GraphOptimizationLevel::Extended | GraphOptimizationLevel::All
    ) {
        stats.constant_folded_nodes = fold_constants(graph)?;
    }

    stats.initializers_removed = drop_unused_initializers(graph);

    Ok(stats)
}

/// Remove `Identity` nodes, rewiring every consumer to the identity's input.
///
/// An `Identity` whose output is a graph output is kept: removing it would rename
/// the model's output.
fn eliminate_identity_nodes(graph: &mut ONNXGraph) -> usize {
    let graph_outputs: HashSet<String> =
        graph.outputs.iter().map(|value| value.name.clone()).collect();

    let mut rewrites: HashMap<String, String> = HashMap::new();
    let mut removed = 0usize;

    graph.nodes.retain(|node| {
        if node.op_type != "Identity" || node.inputs.len() != 1 || node.outputs.len() != 1 {
            return true;
        }
        if graph_outputs.contains(&node.outputs[0]) {
            return true;
        }
        // Follow any chain of earlier identities.
        let source =
            rewrites.get(&node.inputs[0]).cloned().unwrap_or_else(|| node.inputs[0].clone());
        rewrites.insert(node.outputs[0].clone(), source);
        removed += 1;
        false
    });

    if removed > 0 {
        for node in &mut graph.nodes {
            for input in &mut node.inputs {
                if let Some(source) = rewrites.get(input) {
                    *input = source.clone();
                }
            }
        }
    }

    removed
}

/// Evaluate nodes whose inputs are all constants and replace them by initializers.
fn fold_constants(graph: &mut ONNXGraph) -> Result<usize> {
    let graph_outputs: HashSet<&str> =
        graph.outputs.iter().map(|value| value.name.as_str()).collect();

    let mut constants: HashMap<String, CpuTensor> = HashMap::new();
    for initializer in &graph.initializers {
        // An initializer this build cannot decode simply is not a folding candidate.
        if let Ok(value) = tensor_from_initializer(initializer) {
            constants.insert(initializer.name.clone(), value);
        }
    }

    let mut folded = 0usize;
    let mut new_initializers: Vec<ONNXTensor> = Vec::new();
    let mut kept_nodes: Vec<ONNXNode> = Vec::with_capacity(graph.nodes.len());

    for node in std::mem::take(&mut graph.nodes) {
        let foldable = is_supported_op(&node.op_type)
            && !node.inputs.is_empty()
            && node.outputs.len() == 1
            && !graph_outputs.contains(node.outputs[0].as_str())
            && node.inputs.iter().all(|name| constants.contains_key(name));

        if !foldable {
            kept_nodes.push(node);
            continue;
        }

        match eval_node(&node, &constants) {
            Ok(results) if results.len() == 1 => {
                let name = node.outputs[0].clone();
                new_initializers.push(initializer_from_tensor(&name, &results[0]));
                constants.insert(
                    name,
                    results
                        .into_iter()
                        .next()
                        .unwrap_or_else(|| unreachable!("results.len() == 1 was just checked")),
                );
                folded += 1;
            },
            // A node that cannot be evaluated ahead of time simply stays in the graph.
            _ => kept_nodes.push(node),
        }
    }

    graph.nodes = kept_nodes;
    graph.initializers.extend(new_initializers);
    Ok(folded)
}

/// Drop initializers that no node reads and that are not graph outputs.
fn drop_unused_initializers(graph: &mut ONNXGraph) -> usize {
    let mut referenced: HashSet<&str> = HashSet::new();
    for node in &graph.nodes {
        for input in &node.inputs {
            referenced.insert(input.as_str());
        }
    }
    for output in &graph.outputs {
        referenced.insert(output.name.as_str());
    }

    let before = graph.initializers.len();
    let keep: Vec<bool> = graph
        .initializers
        .iter()
        .map(|initializer| referenced.contains(initializer.name.as_str()))
        .collect();
    let mut index = 0usize;
    graph.initializers.retain(|_| {
        let decision = keep[index];
        index += 1;
        decision
    });

    // Declared graph inputs that only existed to carry a dropped initializer's
    // default value go with it.
    let remaining: HashSet<&str> =
        graph.initializers.iter().map(|initializer| initializer.name.as_str()).collect();
    graph.inputs.retain(|value| {
        remaining.contains(value.name.as_str()) || !keep_was_initializer(value, before)
    });

    before - graph.initializers.len()
}

fn keep_was_initializer(_value: &super::onnx::ONNXValueInfo, _before: usize) -> bool {
    // Graph inputs are never removed: a caller-supplied input is not an initializer.
    false
}

/// Replace large float initializers with int8 tensors plus `DequantizeLinear`.
///
/// This is real weight-only dynamic quantization. For each eligible initializer
/// `w`:
///
/// * `scale = max(|w|) / 127`
/// * `w_quantized[i] = clamp(round(w[i] / scale), -127, 127)` stored as int8
/// * a `DequantizeLinear(w_quantized, w_scale) -> w` node is prepended
///
/// so consumers keep reading the name `w` and the graph computes the same function
/// to within quantization error, while the file carries one byte per weight
/// instead of four.
pub fn quantize_onnx_model(
    graph: &mut ONNXGraph,
    bytes_before: usize,
) -> Result<QuantizationStats> {
    let mut stats = QuantizationStats {
        bytes_before,
        ..Default::default()
    };

    let consumed: HashSet<String> =
        graph.nodes.iter().flat_map(|node| node.inputs.iter().cloned()).collect();
    let graph_outputs: HashSet<&str> =
        graph.outputs.iter().map(|value| value.name.as_str()).collect();

    let mut dequantize_nodes: Vec<ONNXNode> = Vec::new();
    let mut extra_initializers: Vec<ONNXTensor> = Vec::new();

    for initializer in &mut graph.initializers {
        if initializer.data_type != ONNXDataType::Float {
            continue;
        }
        let element_count: usize = initializer.dims.iter().map(|&d| d.max(0) as usize).product();
        if element_count < MIN_QUANTIZABLE_ELEMENTS
            || !consumed.contains(&initializer.name)
            || graph_outputs.contains(initializer.name.as_str())
        {
            continue;
        }

        let values: Vec<f32> = initializer
            .raw_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if values.len() != element_count {
            return Err(anyhow!(
                "initializer '{}' declares {element_count} elements but carries {} floats",
                initializer.name,
                values.len()
            ));
        }

        let amax = values.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        if amax == 0.0 || !amax.is_finite() {
            // An all-zero or non-finite tensor gains nothing and would divide by zero.
            continue;
        }
        let scale = amax / 127.0;

        let quantized: Vec<u8> = values
            .iter()
            .map(|&v| (((v / scale).round()).clamp(-127.0, 127.0) as i8) as u8)
            .collect();

        let original_name = initializer.name.clone();
        let quantized_name = format!("{original_name}_quantized");
        let scale_name = format!("{original_name}_scale");

        stats.float_bytes_before += initializer.raw_data.len();
        stats.int8_bytes_after += quantized.len();
        stats.tensors_quantized += 1;

        initializer.name = quantized_name.clone();
        initializer.data_type = ONNXDataType::Int8;
        initializer.raw_data = quantized;

        extra_initializers.push(ONNXTensor {
            name: scale_name.clone(),
            data_type: ONNXDataType::Float,
            dims: Vec::new(),
            raw_data: scale.to_le_bytes().to_vec(),
        });

        dequantize_nodes.push(ONNXNode {
            op_type: "DequantizeLinear".to_string(),
            inputs: vec![quantized_name, scale_name],
            outputs: vec![original_name.clone()],
            attributes: HashMap::new(),
            name: format!("{original_name}_dequantize"),
        });
    }

    if stats.tensors_quantized > 0 {
        graph.initializers.extend(extra_initializers);
        // The dequantize nodes read initializers only, so they are valid at the
        // very front of the topologically ordered node list.
        dequantize_nodes.extend(std::mem::take(&mut graph.nodes));
        graph.nodes = dequantize_nodes;
    }

    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::onnx::{
        ONNXDimension, ONNXTensorShape, ONNXTensorType, ONNXTypeInfo, ONNXValueInfo,
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

    fn float_initializer(name: &str, dims: &[i64], values: &[f32]) -> ONNXTensor {
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

    #[test]
    fn identity_chains_collapse_to_the_original_source() {
        let mut graph = ONNXGraph {
            nodes: vec![
                node("Identity", "i1", &["x"], &["a"]),
                node("Identity", "i2", &["a"], &["b"]),
                node("Relu", "r", &["b"], &["y"]),
            ],
            inputs: vec![value_info("x", &[2])],
            outputs: vec![value_info("y", &[2])],
            initializers: Vec::new(),
            name: "g".to_string(),
        };

        assert_eq!(eliminate_identity_nodes(&mut graph), 2);
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].inputs, vec!["x".to_string()]);
    }

    #[test]
    fn an_identity_producing_a_graph_output_is_kept() {
        let mut graph = ONNXGraph {
            nodes: vec![node("Identity", "i1", &["x"], &["y"])],
            inputs: vec![value_info("x", &[2])],
            outputs: vec![value_info("y", &[2])],
            initializers: Vec::new(),
            name: "g".to_string(),
        };
        assert_eq!(eliminate_identity_nodes(&mut graph), 0);
        assert_eq!(graph.nodes.len(), 1);
    }

    #[test]
    fn constant_folding_evaluates_with_the_real_kernels() {
        let mut graph = ONNXGraph {
            nodes: vec![
                node("Mul", "m", &["a", "b"], &["c"]),
                node("Add", "add", &["x", "c"], &["y"]),
            ],
            inputs: vec![value_info("x", &[2])],
            outputs: vec![value_info("y", &[2])],
            initializers: vec![
                float_initializer("a", &[2], &[2.0, 3.0]),
                float_initializer("b", &[2], &[4.0, 5.0]),
            ],
            name: "g".to_string(),
        };

        assert_eq!(fold_constants(&mut graph).expect("fold"), 1);
        assert_eq!(graph.nodes.len(), 1);
        let folded = graph.initializers.iter().find(|t| t.name == "c").expect("folded initializer");
        let values: Vec<f32> = folded
            .raw_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(values, vec![8.0, 15.0]);
    }

    #[test]
    fn unsupported_operators_are_not_folded() {
        let mut graph = ONNXGraph {
            nodes: vec![node("SomeExoticOp", "x", &["a"], &["c"])],
            inputs: Vec::new(),
            outputs: vec![value_info("c", &[2])],
            initializers: vec![float_initializer("a", &[2], &[1.0, 2.0])],
            name: "g".to_string(),
        };
        assert_eq!(fold_constants(&mut graph).expect("fold"), 0);
        assert_eq!(graph.nodes.len(), 1);
    }

    #[test]
    fn unused_initializers_are_dropped() {
        let mut graph = ONNXGraph {
            nodes: vec![node("Add", "add", &["x", "b"], &["y"])],
            inputs: vec![value_info("x", &[2])],
            outputs: vec![value_info("y", &[2])],
            initializers: vec![
                float_initializer("b", &[2], &[1.0, 2.0]),
                float_initializer("orphan", &[2], &[0.0, 0.0]),
            ],
            name: "g".to_string(),
        };
        assert_eq!(drop_unused_initializers(&mut graph), 1);
        assert_eq!(graph.initializers.len(), 1);
        assert_eq!(graph.initializers[0].name, "b");
    }

    #[test]
    fn quantization_replaces_float_weights_with_int8_plus_dequantize() {
        let weights: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.5).collect();
        let mut graph = ONNXGraph {
            nodes: vec![node("MatMul", "mm", &["x", "w"], &["y"])],
            inputs: vec![value_info("x", &[1, 8])],
            outputs: vec![value_info("y", &[1, 8])],
            initializers: vec![
                float_initializer("w", &[8, 8], &weights),
                // Too small to be worth quantizing.
                float_initializer("tiny", &[2], &[1.0, 2.0]),
            ],
            name: "g".to_string(),
        };
        // `tiny` must be referenced or it would just be dropped as unused.
        graph.nodes.push(node("Add", "add", &["y", "tiny"], &["z"]));
        graph.outputs.push(value_info("z", &[1, 8]));

        let stats = quantize_onnx_model(&mut graph, 0).expect("quantize");
        assert_eq!(stats.tensors_quantized, 1);
        assert_eq!(stats.float_bytes_before, 64 * 4);
        assert_eq!(stats.int8_bytes_after, 64);

        assert_eq!(graph.nodes[0].op_type, "DequantizeLinear");
        assert_eq!(graph.nodes[0].outputs, vec!["w".to_string()]);

        let quantized = graph
            .initializers
            .iter()
            .find(|t| t.name == "w_quantized")
            .expect("quantized tensor");
        assert_eq!(quantized.data_type, ONNXDataType::Int8);
        assert_eq!(quantized.raw_data.len(), 64);

        let tiny = graph.initializers.iter().find(|t| t.name == "tiny").expect("tiny kept");
        assert_eq!(tiny.data_type, ONNXDataType::Float);
    }

    #[test]
    fn quantization_leaves_all_zero_tensors_alone() {
        let mut graph = ONNXGraph {
            nodes: vec![node("MatMul", "mm", &["x", "w"], &["y"])],
            inputs: vec![value_info("x", &[1, 8])],
            outputs: vec![value_info("y", &[1, 8])],
            initializers: vec![float_initializer("w", &[8, 8], &[0.0; 64])],
            name: "g".to_string(),
        };
        let stats = quantize_onnx_model(&mut graph, 0).expect("quantize");
        assert_eq!(
            stats.tensors_quantized, 0,
            "a zero scale would divide by zero"
        );
        assert_eq!(graph.initializers[0].data_type, ONNXDataType::Float);
    }

    #[test]
    fn optimization_level_none_changes_nothing() {
        let mut graph = ONNXGraph {
            nodes: vec![
                node("Identity", "i", &["x"], &["a"]),
                node("Relu", "r", &["a"], &["y"]),
            ],
            inputs: vec![value_info("x", &[2])],
            outputs: vec![value_info("y", &[2])],
            initializers: vec![float_initializer("orphan", &[2], &[0.0, 0.0])],
            name: "g".to_string(),
        };
        let stats = optimize_graph(&mut graph, GraphOptimizationLevel::None, 0).expect("optimize");
        assert_eq!(stats, GraphOptimizationStats::default());
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.initializers.len(), 1);
    }
}
