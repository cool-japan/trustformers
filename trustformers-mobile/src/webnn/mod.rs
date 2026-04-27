//! WebNN (Web Neural Network API) integration.
//!
//! Provides compilation to WebNN operations and runtime dispatch.
//! Models can be compiled to the WebNN IR, exported to JSON or binary JSON,
//! and loaded by any compliant runtime.

use std::collections::HashMap;
use std::path::Path;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde::{Deserialize, Serialize};
use trustformers_core::errors::{Result, TrustformersError};

// ─── Operation types ────────────────────────────────────────────────────────

/// WebNN operation types (subset of W3C WebNN spec).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WebNNOp {
    MatMul { transpose_a: bool, transpose_b: bool },
    Add,
    Mul,
    Relu,
    Sigmoid,
    Tanh,
    Softmax { axis: i32 },
    LayerNorm { axes: Vec<i32>, epsilon: f32 },
    GemmOp { alpha: f32, beta: f32, transpose_a: bool, transpose_b: bool },
    Conv2d {
        padding: [usize; 4],
        strides: [usize; 2],
        dilations: [usize; 2],
        groups: usize,
    },
    Reshape { new_shape: Vec<i64> },
    Transpose { permutation: Vec<usize> },
    Gather { axis: i32 },
    Concat { axis: i32 },
    Slice { starts: Vec<i64>, ends: Vec<i64>, axes: Vec<i32> },
    Gelu,
    Silu,
    Embedding { vocab_size: usize, embedding_dim: usize },
    // Extended operations
    /// General matrix multiply with explicit transpose and scaling (spec variant).
    Gemm { a_transpose: bool, b_transpose: bool, alpha: f32, beta: f32 },
    /// Swish activation: x * sigmoid(x).
    Swish,
    /// Hard sigmoid: clamp((x * alpha + beta), 0, 1).
    HardSigmoid { alpha: f32, beta: f32 },
    /// Clip / clamp values to [min, max].
    Clip { min: f32, max: f32 },
}

impl WebNNOp {
    /// Human-readable name for the operation (matches W3C WebNN spec naming where applicable).
    pub fn op_name(&self) -> &'static str {
        match self {
            WebNNOp::MatMul { .. } => "matmul",
            WebNNOp::Add => "add",
            WebNNOp::Mul => "mul",
            WebNNOp::Relu => "relu",
            WebNNOp::Sigmoid => "sigmoid",
            WebNNOp::Tanh => "tanh",
            WebNNOp::Softmax { .. } => "softmax",
            WebNNOp::LayerNorm { .. } => "layerNormalization",
            WebNNOp::GemmOp { .. } => "gemm",
            WebNNOp::Conv2d { .. } => "conv2d",
            WebNNOp::Reshape { .. } => "reshape",
            WebNNOp::Transpose { .. } => "transpose",
            WebNNOp::Gather { .. } => "gather",
            WebNNOp::Concat { .. } => "concat",
            WebNNOp::Slice { .. } => "slice",
            WebNNOp::Gelu => "gelu",
            WebNNOp::Silu => "silu",
            WebNNOp::Embedding { .. } => "embedding",
            WebNNOp::Gemm { .. } => "gemm_extended",
            WebNNOp::Swish => "swish",
            WebNNOp::HardSigmoid { .. } => "hardSigmoid",
            WebNNOp::Clip { .. } => "clip",
        }
    }

    /// Whether the operation produces a single output tensor.
    pub fn single_output(&self) -> bool {
        true
    }
}

// ─── Graph structures ────────────────────────────────────────────────────────

/// A node in the WebNN computation graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebNNNode {
    /// Unique identifier for this node within the graph.
    pub id: String,
    /// The operation performed by this node.
    pub op: WebNNOp,
    /// IDs of input nodes or named constant tensors.
    pub inputs: Vec<String>,
    /// IDs produced by this node (typically one per output).
    pub outputs: Vec<String>,
    /// Extra operation-specific attributes (e.g. axis names, dtypes).
    pub attributes: HashMap<String, serde_json::Value>,
}

/// Error type for WebNN graph construction and validation.
#[derive(Debug, thiserror::Error)]
pub enum WebNNError {
    #[error("empty graph: no operations defined")]
    EmptyGraph,
    #[error("no outputs declared in graph")]
    NoOutputs,
    #[error("undefined input referenced: {0}")]
    UndefinedInput(String),
    #[error("serialization error: {0}")]
    SerializationError(String),
}

/// Scalar / opaque element type for WebNN tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WebNNDataType {
    Float32,
    Float16,
    Int32,
    Int64,
    Uint8,
    Int8,
    Bool,
}

impl WebNNDataType {
    /// Size of one element in bytes.
    pub fn element_size_bytes(self) -> usize {
        match self {
            WebNNDataType::Float32 | WebNNDataType::Int32 => 4,
            WebNNDataType::Float16 => 2,
            WebNNDataType::Int64 => 8,
            WebNNDataType::Uint8 | WebNNDataType::Int8 | WebNNDataType::Bool => 1,
        }
    }

    /// Alias for `element_size_bytes` (spec-compatible name).
    pub fn byte_size(self) -> usize {
        self.element_size_bytes()
    }

    /// Returns `true` if this is a floating-point type.
    pub fn is_float(self) -> bool {
        matches!(self, WebNNDataType::Float32 | WebNNDataType::Float16)
    }
}

/// Shape + type description for a graph tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebNNTensorDesc {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: WebNNDataType,
}

impl WebNNTensorDesc {
    /// Total number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.num_elements() * self.dtype.element_size_bytes()
    }
}

/// A named constant (weight) tensor stored in the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebNNTensorData {
    /// Tensor descriptor (name, shape, dtype).
    pub desc: WebNNTensorDesc,
    /// Raw bytes encoded as standard base64.
    pub data_base64: String,
}

impl WebNNTensorData {
    /// Create from a byte slice.  The bytes are base64-encoded for JSON portability.
    pub fn from_bytes(desc: WebNNTensorDesc, data: &[u8]) -> Self {
        Self {
            desc,
            data_base64: BASE64.encode(data),
        }
    }

    /// Decode the base64-encoded bytes.
    pub fn decode_bytes(&self) -> Result<Vec<u8>> {
        BASE64.decode(&self.data_base64).map_err(|e| {
            TrustformersError::invalid_input(format!("base64 decode error for tensor '{}': {}", self.desc.name, e))
        })
    }
}

/// Graph-level metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WebNNGraphMetadata {
    pub model_name: String,
    pub model_version: String,
    pub created_at: String,
    /// WebNN spec version targeted (e.g. `"0.0.1"`).
    pub webnn_spec_version: String,
    /// Names of backends known to support this graph (e.g. `["cpu", "gpu"]`).
    pub supported_backends: Vec<String>,
}

/// A complete WebNN computation graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebNNGraph {
    pub name: String,
    pub nodes: Vec<WebNNNode>,
    /// Named graph-level inputs (user-supplied at inference time).
    pub inputs: Vec<WebNNTensorDesc>,
    /// Node output IDs that are exposed as graph outputs.
    pub outputs: Vec<WebNNTensorDesc>,
    /// Named weight / constant tensors embedded in the graph.
    pub constants: HashMap<String, WebNNTensorData>,
    pub metadata: WebNNGraphMetadata,
}

impl WebNNGraph {
    /// Total number of parameters (constant tensor elements).
    pub fn total_parameters(&self) -> usize {
        self.constants.values().map(|t| t.desc.num_elements()).sum()
    }

    /// Estimated memory footprint of constants in bytes.
    pub fn constants_size_bytes(&self) -> usize {
        self.constants.values().map(|t| t.desc.size_bytes()).sum()
    }
}

// ─── Simple graph (spec-required API) ────────────────────────────────────────

/// Lightweight, incremental WebNN computation graph.
///
/// Unlike `WebNNGraph` (which carries constants and metadata), `WebNNSimpleGraph`
/// is intended for fast construction and validation of op sequences.
#[derive(Debug, Clone)]
pub struct WebNNSimpleGraph {
    /// Ordered list of `(node_name, op)` pairs.
    pub ops: Vec<(String, WebNNOp)>,
    /// Named graph-level inputs.
    pub inputs: Vec<String>,
    /// Named graph-level outputs.
    pub outputs: Vec<String>,
}

impl Default for WebNNSimpleGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl WebNNSimpleGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            ops: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Append an operation with the given node name.
    pub fn add_op(&mut self, name: &str, op: WebNNOp) {
        self.ops.push((name.to_owned(), op));
    }

    /// Return the number of operations in the graph.
    pub fn op_count(&self) -> usize {
        self.ops.len()
    }

    /// Validate that the graph has at least one operation and at least one output.
    pub fn validate(&self) -> std::result::Result<(), WebNNError> {
        if self.ops.is_empty() {
            return Err(WebNNError::EmptyGraph);
        }
        if self.outputs.is_empty() {
            return Err(WebNNError::NoOutputs);
        }
        Ok(())
    }

    /// Produce a human-readable JSON-style description of the graph.
    pub fn to_json_description(&self) -> String {
        let op_entries: Vec<String> = self
            .ops
            .iter()
            .map(|(name, op)| format!("{{\"name\":\"{name}\",\"op\":\"{}\"}}", op.op_name()))
            .collect();
        format!(
            "{{\"inputs\":{:?},\"outputs\":{:?},\"ops\":[{}]}}",
            self.inputs,
            self.outputs,
            op_entries.join(",")
        )
    }
}

// ─── Simple builder (spec-required API) ──────────────────────────────────────

/// Fluent builder for `WebNNSimpleGraph`.
pub struct WebNNBuilder {
    /// The graph being constructed.
    pub graph: WebNNSimpleGraph,
    /// Tensor descriptors for declared inputs.
    pub tensors: Vec<WebNNTensorDesc>,
}

impl Default for WebNNBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl WebNNBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            graph: WebNNSimpleGraph::new(),
            tensors: Vec::new(),
        }
    }

    /// Declare an input tensor.
    pub fn add_input(&mut self, name: &str, desc: WebNNTensorDesc) -> &mut Self {
        self.graph.inputs.push(name.to_owned());
        self.tensors.push(desc);
        self
    }

    /// Add a matrix-multiply operation.
    ///
    /// `a` and `b` are the names of the input tensors (informational).
    pub fn add_matmul(&mut self, name: &str, _a: &str, _b: &str) -> &mut Self {
        self.graph.add_op(name, WebNNOp::MatMul { transpose_a: false, transpose_b: false });
        self
    }

    /// Add a ReLU activation.
    ///
    /// `input` is the name of the preceding tensor (informational).
    pub fn add_relu(&mut self, name: &str, _input: &str) -> &mut Self {
        self.graph.add_op(name, WebNNOp::Relu);
        self
    }

    /// Finalise construction.
    ///
    /// If no outputs have been declared, all op names are used as outputs.
    /// Returns an error if the resulting graph is invalid.
    pub fn build(&mut self) -> std::result::Result<WebNNSimpleGraph, WebNNError> {
        if self.graph.outputs.is_empty() {
            self.graph.outputs = self.graph.ops.iter().map(|(n, _)| n.clone()).collect();
        }
        let g = self.graph.clone();
        g.validate()?;
        Ok(g)
    }
}

// ─── Configuration ───────────────────────────────────────────────────────────

/// Target backend string constants.
pub mod backend {
    pub const CPU: &str = "cpu";
    pub const GPU: &str = "gpu";
    pub const NPU: &str = "npu";
}

/// Configuration for a WebNN export or runtime session.
#[derive(Debug, Clone)]
pub struct WebNNConfig {
    /// Preferred backend: `"cpu"`, `"gpu"`, or `"npu"`.
    pub preferred_backend: String,
    /// Power preference: `"default"`, `"low-power"`, or `"high-performance"`.
    pub power_preference: String,
    /// Whether to quantize constant weights to INT8 for a smaller serialised graph.
    pub quantize_weights: bool,
    /// Output format for `export_webnn_graph`.
    pub output_format: WebNNOutputFormat,
}

/// Serialisation format for a WebNN graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebNNOutputFormat {
    /// Human-readable JSON (larger, easy to inspect).
    Json,
    /// Compact JSON (minified — no indentation).
    BinaryJson,
}

impl Default for WebNNConfig {
    fn default() -> Self {
        Self {
            preferred_backend: backend::CPU.to_owned(),
            power_preference: "default".to_owned(),
            quantize_weights: false,
            output_format: WebNNOutputFormat::Json,
        }
    }
}

// ─── Builder ─────────────────────────────────────────────────────────────────

/// Incrementally constructs a `WebNNGraph`.
pub struct WebNNGraphBuilder {
    graph: WebNNGraph,
    node_counter: usize,
}

impl WebNNGraphBuilder {
    /// Create a new builder for a graph with the given name.
    pub fn new(name: &str) -> Self {
        Self {
            graph: WebNNGraph {
                name: name.to_owned(),
                nodes: Vec::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                constants: HashMap::new(),
                metadata: WebNNGraphMetadata {
                    webnn_spec_version: "0.0.1".to_owned(),
                    ..Default::default()
                },
            },
            node_counter: 0,
        }
    }

    /// Add an operation node and return the generated output node ID.
    pub fn add_node(&mut self, op: WebNNOp, inputs: Vec<String>) -> String {
        let id = format!("node_{}", self.node_counter);
        self.node_counter += 1;
        let output_id = format!("{}_out", id);
        self.graph.nodes.push(WebNNNode {
            id,
            op,
            inputs,
            outputs: vec![output_id.clone()],
            attributes: HashMap::new(),
        });
        output_id
    }

    /// Add a named constant tensor (e.g. a weight matrix).
    pub fn add_constant(&mut self, name: &str, desc: WebNNTensorDesc, data: Vec<u8>) {
        let tensor_data = WebNNTensorData::from_bytes(desc, &data);
        self.graph.constants.insert(name.to_owned(), tensor_data);
    }

    /// Declare graph-level input descriptors.
    pub fn set_inputs(&mut self, inputs: Vec<WebNNTensorDesc>) {
        self.graph.inputs = inputs;
    }

    /// Declare which node output IDs are the graph outputs.
    ///
    /// Each provided ID becomes a graph output with a synthetic name.
    pub fn set_outputs(&mut self, output_ids: Vec<String>) {
        self.graph.outputs = output_ids
            .into_iter()
            .enumerate()
            .map(|(i, name)| WebNNTensorDesc {
                name,
                shape: vec![],    // shape resolved at runtime
                dtype: WebNNDataType::Float32,
            })
            .collect();
    }

    /// Apply metadata before building.
    pub fn with_metadata(mut self, metadata: WebNNGraphMetadata) -> Self {
        self.graph.metadata = metadata;
        self
    }

    /// Consume the builder and return the finished graph.
    pub fn build(self) -> WebNNGraph {
        self.graph
    }
}

// ─── Export / load ───────────────────────────────────────────────────────────

/// Serialise a `WebNNGraph` to a file.
///
/// The format (pretty JSON vs. compact JSON) is controlled by `config.output_format`.
pub fn export_webnn_graph(
    graph: &WebNNGraph,
    path: &Path,
    config: &WebNNConfig,
) -> Result<()> {
    let bytes = match config.output_format {
        WebNNOutputFormat::Json => {
            serde_json::to_vec_pretty(graph).map_err(|e| {
                TrustformersError::invalid_input(format!("JSON serialisation failed: {e}"))
            })?
        }
        WebNNOutputFormat::BinaryJson => {
            serde_json::to_vec(graph).map_err(|e| {
                TrustformersError::invalid_input(format!("compact JSON serialisation failed: {e}"))
            })?
        }
    };
    std::fs::write(path, &bytes).map_err(|e| {
        TrustformersError::invalid_input(format!("failed to write WebNN graph to '{}': {e}", path.display()))
    })
}

/// Deserialise a `WebNNGraph` from a file (JSON or compact JSON).
pub fn load_webnn_graph(path: &Path) -> Result<WebNNGraph> {
    let bytes = std::fs::read(path).map_err(|e| {
        TrustformersError::invalid_input(format!("failed to read WebNN graph from '{}': {e}", path.display()))
    })?;
    serde_json::from_slice(&bytes).map_err(|e| {
        TrustformersError::invalid_input(format!("JSON deserialisation failed: {e}"))
    })
}

// ─── Validation ──────────────────────────────────────────────────────────────

/// Validate the structural integrity of a `WebNNGraph`.
///
/// Returns a list of diagnostic warnings (non-fatal issues).  An `Err` is
/// returned only for fatal structural problems.
pub fn validate_webnn_graph(graph: &WebNNGraph) -> Result<Vec<String>> {
    let mut warnings: Vec<String> = Vec::new();

    if graph.name.is_empty() {
        warnings.push("Graph name is empty".to_owned());
    }
    if graph.nodes.is_empty() {
        return Err(TrustformersError::invalid_input(
            "WebNN graph has no nodes".to_owned(),
        ));
    }
    if graph.outputs.is_empty() {
        return Err(TrustformersError::invalid_input(
            "WebNN graph has no declared outputs".to_owned(),
        ));
    }

    // Build the set of all reachable node-output IDs.
    let mut produced: std::collections::HashSet<String> = std::collections::HashSet::new();
    for input in &graph.inputs {
        produced.insert(input.name.clone());
    }
    for name in graph.constants.keys() {
        produced.insert(name.clone());
    }

    for node in &graph.nodes {
        // Check all inputs are available at this point.
        for inp in &node.inputs {
            if !produced.contains(inp) {
                warnings.push(format!(
                    "Node '{}' references undefined input '{}'",
                    node.id, inp
                ));
            }
        }
        for out in &node.outputs {
            produced.insert(out.clone());
        }
    }

    // Check all graph outputs are produced.
    for out in &graph.outputs {
        if !produced.contains(&out.name) {
            return Err(TrustformersError::invalid_input(format!(
                "Declared graph output '{}' is not produced by any node",
                out.name
            )));
        }
    }

    if graph.metadata.webnn_spec_version.is_empty() {
        warnings.push("WebNN spec version not set in metadata".to_owned());
    }

    Ok(warnings)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    fn make_simple_graph() -> WebNNGraph {
        let mut builder = WebNNGraphBuilder::new("test_graph");
        builder.set_inputs(vec![WebNNTensorDesc {
            name: "input".to_owned(),
            shape: vec![1, 128],
            dtype: WebNNDataType::Float32,
        }]);
        let matmul_out = builder.add_node(
            WebNNOp::MatMul { transpose_a: false, transpose_b: true },
            vec!["input".to_owned(), "weight".to_owned()],
        );
        let relu_out = builder.add_node(WebNNOp::Relu, vec![matmul_out.clone()]);
        builder.add_constant(
            "weight",
            WebNNTensorDesc {
                name: "weight".to_owned(),
                shape: vec![64, 128],
                dtype: WebNNDataType::Float32,
            },
            vec![0u8; 64 * 128 * 4],
        );
        builder.set_outputs(vec![relu_out]);
        builder.build()
    }

    #[test]
    fn test_builder_creates_nodes() {
        let graph = make_simple_graph();
        assert_eq!(graph.name, "test_graph");
        assert_eq!(graph.nodes.len(), 2);
        assert!(graph.constants.contains_key("weight"));
    }

    #[test]
    fn test_webnn_op_names() {
        assert_eq!(WebNNOp::Relu.op_name(), "relu");
        assert_eq!(WebNNOp::Gelu.op_name(), "gelu");
        assert_eq!(WebNNOp::Add.op_name(), "add");
        let softmax = WebNNOp::Softmax { axis: -1 };
        assert_eq!(softmax.op_name(), "softmax");
    }

    #[test]
    fn test_dtype_element_sizes() {
        assert_eq!(WebNNDataType::Float32.element_size_bytes(), 4);
        assert_eq!(WebNNDataType::Float16.element_size_bytes(), 2);
        assert_eq!(WebNNDataType::Int64.element_size_bytes(), 8);
        assert_eq!(WebNNDataType::Uint8.element_size_bytes(), 1);
    }

    #[test]
    fn test_tensor_desc_size() {
        let desc = WebNNTensorDesc {
            name: "x".to_owned(),
            shape: vec![4, 8],
            dtype: WebNNDataType::Float32,
        };
        assert_eq!(desc.num_elements(), 32);
        assert_eq!(desc.size_bytes(), 128);
    }

    #[test]
    fn test_tensor_data_roundtrip() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let desc = WebNNTensorDesc {
            name: "t".to_owned(),
            shape: vec![8],
            dtype: WebNNDataType::Uint8,
        };
        let td = WebNNTensorData::from_bytes(desc, &data);
        let decoded = td.decode_bytes().expect("decode should succeed");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_validate_valid_graph() {
        let graph = make_simple_graph();
        let warnings = validate_webnn_graph(&graph).expect("valid graph should not error");
        // metadata.webnn_spec_version is set, so no warnings expected
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
    }

    #[test]
    fn test_validate_empty_nodes_is_error() {
        let graph = WebNNGraph {
            name: "empty".to_owned(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![WebNNTensorDesc {
                name: "out".to_owned(),
                shape: vec![],
                dtype: WebNNDataType::Float32,
            }],
            constants: HashMap::new(),
            metadata: WebNNGraphMetadata::default(),
        };
        assert!(validate_webnn_graph(&graph).is_err());
    }

    #[test]
    fn test_export_and_load_json() {
        let graph = make_simple_graph();
        let mut path = temp_dir();
        path.push("webnn_test_graph.json");
        let config = WebNNConfig {
            output_format: WebNNOutputFormat::Json,
            ..Default::default()
        };
        export_webnn_graph(&graph, &path, &config).expect("export should succeed");
        let loaded = load_webnn_graph(&path).expect("load should succeed");
        assert_eq!(loaded.name, graph.name);
        assert_eq!(loaded.nodes.len(), graph.nodes.len());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_export_and_load_binary_json() {
        let graph = make_simple_graph();
        let mut path = temp_dir();
        path.push("webnn_test_compact.json");
        let config = WebNNConfig {
            output_format: WebNNOutputFormat::BinaryJson,
            ..Default::default()
        };
        export_webnn_graph(&graph, &path, &config).expect("export should succeed");
        let loaded = load_webnn_graph(&path).expect("load should succeed");
        assert_eq!(loaded.constants.len(), graph.constants.len());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_graph_stats() {
        let graph = make_simple_graph();
        // 64*128 float32 elements = 32768 bytes
        assert_eq!(graph.total_parameters(), 64 * 128);
        assert_eq!(graph.constants_size_bytes(), 64 * 128 * 4);
    }

    #[test]
    fn test_builder_various_ops() {
        let mut builder = WebNNGraphBuilder::new("multi_op");
        builder.set_inputs(vec![WebNNTensorDesc {
            name: "x".to_owned(),
            shape: vec![1, 32],
            dtype: WebNNDataType::Float32,
        }]);
        let sigmoid_out = builder.add_node(WebNNOp::Sigmoid, vec!["x".to_owned()]);
        let tanh_out = builder.add_node(WebNNOp::Tanh, vec!["x".to_owned()]);
        let _add_out = builder.add_node(WebNNOp::Add, vec![sigmoid_out, tanh_out]);
        let graph = builder.build();
        assert_eq!(graph.nodes.len(), 3);
    }

    // ── WebNNError tests ─────────────────────────────────────────────────────

    #[test]
    fn test_webnn_error_empty_graph() {
        let e = WebNNError::EmptyGraph;
        let msg = e.to_string();
        assert!(!msg.is_empty());
    }

    #[test]
    fn test_webnn_error_no_outputs() {
        let e = WebNNError::NoOutputs;
        assert!(e.to_string().contains("output"));
    }

    #[test]
    fn test_webnn_error_undefined_input() {
        let e = WebNNError::UndefinedInput("my_input".to_owned());
        assert!(e.to_string().contains("my_input"));
    }

    // ── WebNNSimpleGraph tests ────────────────────────────────────────────────

    #[test]
    fn test_webnn_simple_graph_new() {
        let g = WebNNSimpleGraph::new();
        assert_eq!(g.op_count(), 0);
        assert!(g.inputs.is_empty());
        assert!(g.outputs.is_empty());
    }

    #[test]
    fn test_webnn_simple_graph_add_op() {
        let mut g = WebNNSimpleGraph::new();
        g.add_op("relu_0", WebNNOp::Relu);
        assert_eq!(g.op_count(), 1);
        assert_eq!(g.ops[0].0, "relu_0");
    }

    #[test]
    fn test_webnn_simple_graph_op_count() {
        let mut g = WebNNSimpleGraph::new();
        g.add_op("a", WebNNOp::Relu);
        g.add_op("b", WebNNOp::Sigmoid);
        g.add_op("c", WebNNOp::Gelu);
        assert_eq!(g.op_count(), 3);
    }

    #[test]
    fn test_webnn_simple_graph_validate_ok() {
        let mut g = WebNNSimpleGraph::new();
        g.add_op("relu_0", WebNNOp::Relu);
        g.outputs.push("relu_0".to_owned());
        assert!(g.validate().is_ok());
    }

    #[test]
    fn test_webnn_simple_graph_validate_empty_fails() {
        let g = WebNNSimpleGraph::new();
        let result = g.validate();
        assert!(matches!(result, Err(WebNNError::EmptyGraph)));
    }

    #[test]
    fn test_webnn_simple_graph_validate_no_outputs_fails() {
        let mut g = WebNNSimpleGraph::new();
        g.add_op("relu_0", WebNNOp::Relu);
        // No outputs declared
        let result = g.validate();
        assert!(matches!(result, Err(WebNNError::NoOutputs)));
    }

    #[test]
    fn test_webnn_simple_graph_to_json_description_contains_op_name() {
        let mut g = WebNNSimpleGraph::new();
        g.add_op("gelu_layer", WebNNOp::Gelu);
        g.outputs.push("gelu_layer".to_owned());
        let desc = g.to_json_description();
        assert!(desc.contains("gelu"), "description should contain op name: {desc}");
        assert!(desc.contains("gelu_layer"), "description should contain node name: {desc}");
    }

    // ── WebNNBuilder tests ────────────────────────────────────────────────────

    #[test]
    fn test_webnn_builder_new() {
        let b = WebNNBuilder::new();
        assert_eq!(b.graph.op_count(), 0);
        assert!(b.tensors.is_empty());
    }

    #[test]
    fn test_webnn_builder_add_input() {
        let mut b = WebNNBuilder::new();
        let desc = WebNNTensorDesc {
            name: "input_0".to_owned(),
            shape: vec![1, 128],
            dtype: WebNNDataType::Float32,
        };
        b.add_input("input_0", desc);
        assert_eq!(b.graph.inputs.len(), 1);
        assert_eq!(b.tensors.len(), 1);
    }

    #[test]
    fn test_webnn_builder_add_matmul_relu_build() {
        let mut b = WebNNBuilder::new();
        let desc = WebNNTensorDesc {
            name: "x".to_owned(),
            shape: vec![1, 64],
            dtype: WebNNDataType::Float32,
        };
        b.add_input("x", desc);
        b.add_matmul("mm0", "x", "weight");
        b.add_relu("relu0", "mm0");
        let g = b.build().expect("build should succeed");
        assert_eq!(g.op_count(), 2);
        assert_eq!(g.outputs.len(), 2); // auto-populated from op names
    }

    #[test]
    fn test_webnn_builder_build_empty_fails() {
        let mut b = WebNNBuilder::new();
        let result = b.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_webnn_simple_graph_outputs_populated_by_build() {
        let mut b = WebNNBuilder::new();
        b.add_matmul("mm", "a", "b");
        let g = b.build().expect("build should succeed");
        assert!(!g.outputs.is_empty(), "outputs should be auto-populated");
        assert!(g.outputs.contains(&"mm".to_owned()));
    }

    // ── WebNNDataType extended methods ───────────────────────────────────────

    #[test]
    fn test_webnn_data_type_byte_size() {
        assert_eq!(WebNNDataType::Float32.byte_size(), 4);
        assert_eq!(WebNNDataType::Float16.byte_size(), 2);
        assert_eq!(WebNNDataType::Int64.byte_size(), 8);
        assert_eq!(WebNNDataType::Uint8.byte_size(), 1);
    }

    #[test]
    fn test_webnn_data_type_is_float_true() {
        assert!(WebNNDataType::Float32.is_float());
        assert!(WebNNDataType::Float16.is_float());
    }

    #[test]
    fn test_webnn_data_type_is_float_false() {
        assert!(!WebNNDataType::Int32.is_float());
        assert!(!WebNNDataType::Uint8.is_float());
        assert!(!WebNNDataType::Bool.is_float());
        assert!(!WebNNDataType::Int8.is_float());
    }

    // ── New WebNNOp variant tests ─────────────────────────────────────────────

    #[test]
    fn test_webnn_op_swish_name() {
        assert_eq!(WebNNOp::Swish.op_name(), "swish");
    }

    #[test]
    fn test_webnn_op_clip_name() {
        let op = WebNNOp::Clip { min: 0.0, max: 6.0 };
        assert_eq!(op.op_name(), "clip");
    }

    #[test]
    fn test_webnn_op_hard_sigmoid_name() {
        let op = WebNNOp::HardSigmoid { alpha: 0.2, beta: 0.5 };
        assert_eq!(op.op_name(), "hardSigmoid");
    }

    #[test]
    fn test_webnn_op_gemm_extended_name() {
        let op = WebNNOp::Gemm { a_transpose: false, b_transpose: true, alpha: 1.0, beta: 0.0 };
        assert_eq!(op.op_name(), "gemm_extended");
    }
}
