//! ONNX export.
//!
//! This module owns the in-memory `onnx.*` message types and the exporter. The
//! binary protobuf codec lives in [`super::onnx_proto`] and the CPU interpreter in
//! [`super::onnx_cpu`].
//!
//! # What can and cannot be exported
//!
//! [`ONNXExporter::export_graph`] writes a **real binary `onnx.ModelProto`** — the
//! same bytes `onnxruntime` and Netron read — from an [`ONNXModel`] the caller has
//! built, initializers and all.
//!
//! [`ONNXExporter::export`] (the [`ModelExporter`] entry point) cannot do that,
//! because an ONNX graph is a *topology* and the [`Model`] trait exposes only
//! parameters through [`Model::named_tensors`]. It therefore returns a structured
//! [`ErrorKind::UnsupportedOperation`](crate::errors::ErrorKind::UnsupportedOperation)
//! rather than guessing. A previous revision guessed: it emitted twelve
//! transformer blocks for every model and wrote the result as a human-readable text
//! dump under the `.onnx` extension, which no ONNX tool can open.

use super::{collect_model_tensors, ExportConfig, ExportFormat, ExportPrecision, ModelExporter};
use crate::errors::unsupported_operation;
use crate::traits::Model;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::Path;

/// Explanation attached to every refusal to derive an ONNX graph from a `Model`.
pub const ONNX_TOPOLOGY_UNSUPPORTED_REASON: &str =
    "an ONNX graph describes the model's operations and how they are wired \
     together; the `Model` trait exposes parameters only (`named_tensors`), so the \
     graph cannot be derived. Build an `ONNXModel` explicitly and call \
     `ONNXExporter::export_graph`, or export the weights to GGUF.";

/// ONNX model representation
#[derive(Debug, Clone, PartialEq)]
pub struct ONNXModel {
    pub graph: ONNXGraph,
    pub ir_version: i64,
    pub opset_imports: Vec<ONNXOpsetImport>,
    pub producer_name: String,
    pub producer_version: String,
    pub model_version: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ONNXGraph {
    pub nodes: Vec<ONNXNode>,
    pub inputs: Vec<ONNXValueInfo>,
    pub outputs: Vec<ONNXValueInfo>,
    pub initializers: Vec<ONNXTensor>,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ONNXNode {
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, ONNXAttribute>,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ONNXValueInfo {
    pub name: String,
    pub type_info: ONNXTypeInfo,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ONNXTypeInfo {
    pub tensor_type: ONNXTensorType,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ONNXTensorType {
    pub elem_type: ONNXDataType,
    pub shape: ONNXTensorShape,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ONNXTensorShape {
    pub dims: Vec<ONNXDimension>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ONNXDimension {
    Value(i64),
    Parameter(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ONNXDataType {
    Float = 1,
    UInt8 = 2,
    Int8 = 3,
    UInt16 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    String = 8,
    Bool = 9,
    Float16 = 10,
    Double = 11,
    UInt32 = 12,
    UInt64 = 13,
    Complex64 = 14,
    Complex128 = 15,
    BFloat16 = 16,
}

impl ONNXDataType {
    /// Decode a `TensorProto.DataType` wire value.
    pub fn from_i32(value: i32) -> Result<Self> {
        Ok(match value {
            1 => ONNXDataType::Float,
            2 => ONNXDataType::UInt8,
            3 => ONNXDataType::Int8,
            4 => ONNXDataType::UInt16,
            5 => ONNXDataType::Int16,
            6 => ONNXDataType::Int32,
            7 => ONNXDataType::Int64,
            8 => ONNXDataType::String,
            9 => ONNXDataType::Bool,
            10 => ONNXDataType::Float16,
            11 => ONNXDataType::Double,
            12 => ONNXDataType::UInt32,
            13 => ONNXDataType::UInt64,
            14 => ONNXDataType::Complex64,
            15 => ONNXDataType::Complex128,
            16 => ONNXDataType::BFloat16,
            other => return Err(anyhow!("unknown ONNX TensorProto data type {other}")),
        })
    }

    /// Size in bytes of one element, for the fixed-width types.
    ///
    /// Returns `None` for `String`, whose elements are variable length.
    pub fn element_size(&self) -> Option<usize> {
        Some(match self {
            ONNXDataType::Float | ONNXDataType::Int32 | ONNXDataType::UInt32 => 4,
            ONNXDataType::UInt8 | ONNXDataType::Int8 | ONNXDataType::Bool => 1,
            ONNXDataType::UInt16
            | ONNXDataType::Int16
            | ONNXDataType::Float16
            | ONNXDataType::BFloat16 => 2,
            ONNXDataType::Int64 | ONNXDataType::UInt64 | ONNXDataType::Double => 8,
            ONNXDataType::Complex64 => 8,
            ONNXDataType::Complex128 => 16,
            ONNXDataType::String => return None,
        })
    }

    /// The ONNX element type matching an export precision.
    pub fn from_precision(precision: ExportPrecision) -> Self {
        match precision {
            ExportPrecision::FP32 => ONNXDataType::Float,
            ExportPrecision::FP16 => ONNXDataType::Float16,
            // ONNX has no 4-bit tensor element type in the opsets this crate targets.
            ExportPrecision::INT8 | ExportPrecision::INT4 => ONNXDataType::Int8,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ONNXTensor {
    pub name: String,
    pub data_type: ONNXDataType,
    pub dims: Vec<i64>,
    pub raw_data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ONNXOpsetImport {
    pub domain: String,
    pub version: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ONNXAttribute {
    Int(i64),
    Float(f32),
    String(String),
    Tensor(ONNXTensor),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
    Strings(Vec<String>),
}

/// ONNX operator registry for managing supported operators
#[derive(Debug)]
pub struct ONNXOperatorRegistry {
    pub operators: HashMap<String, ONNXOperatorInfo>,
}

/// Information about an ONNX operator
#[derive(Debug, Clone)]
pub struct ONNXOperatorInfo {
    pub name: String,
    pub opset_version: i64,
    pub inputs: Vec<ONNXOperatorInput>,
    pub outputs: Vec<ONNXOperatorOutput>,
    pub attributes: Vec<ONNXOperatorAttribute>,
    pub description: String,
}

/// Input specification for an ONNX operator
#[derive(Debug, Clone)]
pub struct ONNXOperatorInput {
    pub name: String,
    pub types: Vec<ONNXDataType>,
    pub optional: bool,
    pub description: String,
}

/// Output specification for an ONNX operator
#[derive(Debug, Clone)]
pub struct ONNXOperatorOutput {
    pub name: String,
    pub types: Vec<ONNXDataType>,
    pub description: String,
}

/// Attribute specification for an ONNX operator
#[derive(Debug, Clone)]
pub struct ONNXOperatorAttribute {
    pub name: String,
    pub required: bool,
    pub attribute_type: String,
    pub description: String,
}

/// ONNX exporter implementation
#[derive(Clone)]
pub struct ONNXExporter {
    opset_version: i64,
}

impl Default for ONNXExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl ONNXExporter {
    pub fn new() -> Self {
        Self { opset_version: 14 }
    }

    /// Override the opset version declared in exported models.
    pub fn with_opset_version(mut self, version: i64) -> Self {
        self.opset_version = version;
        self
    }

    /// The opset version this exporter declares by default.
    pub fn opset_version(&self) -> i64 {
        self.opset_version
    }

    /// Serialise an explicitly built [`ONNXModel`] as a binary `onnx.ModelProto`.
    ///
    /// This is the real thing: the bytes written here are what `onnxruntime`,
    /// Netron and this crate's own [`super::onnx_cpu`] interpreter read. The
    /// caller supplies the graph, because the [`Model`] trait cannot express one.
    ///
    /// `output_path` is used verbatim — add the `.onnx` extension yourself.
    pub fn export_graph<P: AsRef<Path>>(&self, model: &ONNXModel, output_path: P) -> Result<()> {
        validate_graph(&model.graph)?;
        let bytes = super::onnx_proto::encode_model(model);
        std::fs::write(output_path.as_ref(), bytes)?;
        Ok(())
    }

    /// Serialise an explicitly built [`ONNXModel`] to bytes.
    pub fn encode_graph(&self, model: &ONNXModel) -> Result<Vec<u8>> {
        validate_graph(&model.graph)?;
        Ok(super::onnx_proto::encode_model(model))
    }

    /// Build an [`ONNXModel`] shell — opset, producer and IR version — with the
    /// caller's graph. No nodes or initializers are invented.
    pub fn wrap_graph(&self, graph: ONNXGraph, config: &ExportConfig) -> ONNXModel {
        ONNXModel {
            graph,
            ir_version: 8,
            opset_imports: vec![ONNXOpsetImport {
                domain: String::new(),
                version: config.opset_version.unwrap_or(self.opset_version),
            }],
            producer_name: "TrustformeRS".to_string(),
            producer_version: env!("CARGO_PKG_VERSION").to_string(),
            model_version: 1,
        }
    }
}

/// Reject graphs that would serialise into a file no runtime can execute.
fn validate_graph(graph: &ONNXGraph) -> Result<()> {
    if graph.nodes.is_empty() {
        return Err(anyhow!(
            "ONNX graph '{}' has no nodes; an initializer-only file is not an executable model",
            graph.name
        ));
    }
    if graph.outputs.is_empty() {
        return Err(anyhow!("ONNX graph '{}' declares no outputs", graph.name));
    }

    let mut produced: std::collections::HashSet<&str> =
        graph.inputs.iter().map(|value| value.name.as_str()).collect();
    for initializer in &graph.initializers {
        produced.insert(initializer.name.as_str());

        let element_count: i64 = initializer.dims.iter().product();
        if let Some(element_size) = initializer.data_type.element_size() {
            let expected = element_count.max(0) as usize * element_size;
            if initializer.raw_data.len() != expected {
                return Err(anyhow!(
                    "ONNX initializer '{}' declares dims {:?} of {:?} ({expected} bytes) but \
                     carries {} bytes",
                    initializer.name,
                    initializer.dims,
                    initializer.data_type,
                    initializer.raw_data.len()
                ));
            }
        }
    }

    for node in &graph.nodes {
        for input in &node.inputs {
            // The empty name is ONNX's way of skipping an optional input.
            if !input.is_empty() && !produced.contains(input.as_str()) {
                return Err(anyhow!(
                    "ONNX node '{}' ({}) consumes '{}', which no input, initializer or earlier \
                     node produces",
                    node.name,
                    node.op_type,
                    input
                ));
            }
        }
        for output in &node.outputs {
            produced.insert(output.as_str());
        }
    }

    for output in &graph.outputs {
        if !produced.contains(output.name.as_str()) {
            return Err(anyhow!(
                "ONNX graph output '{}' is never produced by any node",
                output.name
            ));
        }
    }

    Ok(())
}

impl Default for ONNXOperatorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ONNXOperatorRegistry {
    pub fn new() -> Self {
        Self {
            operators: HashMap::new(),
        }
    }

    pub fn register_all_operators(&mut self) {
        self.register_core_operators();
        self.register_math_operators();
        self.register_neural_network_operators();
        self.register_tensor_operators();
        self.register_control_flow_operators();
        self.register_quantization_operators();
    }

    fn register_core_operators(&mut self) {
        // Core mathematical operations
        self.register_operator(ONNXOperatorInfo {
            name: "Add".to_string(),
            opset_version: 7,
            inputs: vec![
                ONNXOperatorInput {
                    name: "A".to_string(),
                    types: vec![
                        ONNXDataType::Float,
                        ONNXDataType::Double,
                        ONNXDataType::Int32,
                        ONNXDataType::Int64,
                    ],
                    optional: false,
                    description: "First operand".to_string(),
                },
                ONNXOperatorInput {
                    name: "B".to_string(),
                    types: vec![
                        ONNXDataType::Float,
                        ONNXDataType::Double,
                        ONNXDataType::Int32,
                        ONNXDataType::Int64,
                    ],
                    optional: false,
                    description: "Second operand".to_string(),
                },
            ],
            outputs: vec![ONNXOperatorOutput {
                name: "C".to_string(),
                types: vec![
                    ONNXDataType::Float,
                    ONNXDataType::Double,
                    ONNXDataType::Int32,
                    ONNXDataType::Int64,
                ],
                description: "Result of addition".to_string(),
            }],
            attributes: vec![],
            description: "Element-wise addition".to_string(),
        });

        self.register_operator(ONNXOperatorInfo {
            name: "MatMul".to_string(),
            opset_version: 1,
            inputs: vec![
                ONNXOperatorInput {
                    name: "A".to_string(),
                    types: vec![
                        ONNXDataType::Float,
                        ONNXDataType::Double,
                        ONNXDataType::Float16,
                        ONNXDataType::BFloat16,
                    ],
                    optional: false,
                    description: "N-dimensional matrix A".to_string(),
                },
                ONNXOperatorInput {
                    name: "B".to_string(),
                    types: vec![
                        ONNXDataType::Float,
                        ONNXDataType::Double,
                        ONNXDataType::Float16,
                        ONNXDataType::BFloat16,
                    ],
                    optional: false,
                    description: "N-dimensional matrix B".to_string(),
                },
            ],
            outputs: vec![ONNXOperatorOutput {
                name: "Y".to_string(),
                types: vec![
                    ONNXDataType::Float,
                    ONNXDataType::Double,
                    ONNXDataType::Float16,
                    ONNXDataType::BFloat16,
                ],
                description: "Matrix multiplication result".to_string(),
            }],
            attributes: vec![],
            description: "Matrix multiplication".to_string(),
        });
    }

    fn register_math_operators(&mut self) {
        // Activation functions
        let activations = vec![
            ("Relu", "Rectified Linear Unit activation"),
            ("Sigmoid", "Sigmoid activation function"),
            ("Tanh", "Hyperbolic tangent activation"),
            ("Gelu", "Gaussian Error Linear Unit activation"),
            ("LeakyRelu", "Leaky ReLU activation"),
            ("Elu", "Exponential Linear Unit activation"),
            ("Selu", "Scaled Exponential Linear Unit activation"),
            ("Swish", "Swish activation function"),
        ];

        for (name, desc) in activations {
            self.register_operator(ONNXOperatorInfo {
                name: name.to_string(),
                opset_version: 6,
                inputs: vec![ONNXOperatorInput {
                    name: "X".to_string(),
                    types: vec![
                        ONNXDataType::Float,
                        ONNXDataType::Double,
                        ONNXDataType::Float16,
                    ],
                    optional: false,
                    description: "Input tensor".to_string(),
                }],
                outputs: vec![ONNXOperatorOutput {
                    name: "Y".to_string(),
                    types: vec![
                        ONNXDataType::Float,
                        ONNXDataType::Double,
                        ONNXDataType::Float16,
                    ],
                    description: "Output tensor".to_string(),
                }],
                attributes: if name == "LeakyRelu" {
                    vec![ONNXOperatorAttribute {
                        name: "alpha".to_string(),
                        attribute_type: "float".to_string(),
                        required: false,
                        description: "Coefficient of leakage".to_string(),
                    }]
                } else {
                    vec![]
                },
                description: desc.to_string(),
            });
        }

        // Mathematical functions
        let math_ops = vec![
            "Abs",
            "Acos",
            "Asin",
            "Atan",
            "Ceil",
            "Cos",
            "Cosh",
            "Exp",
            "Floor",
            "Log",
            "Neg",
            "Reciprocal",
            "Round",
            "Sign",
            "Sin",
            "Sinh",
            "Sqrt",
            "Tan",
            "Erf",
        ];

        for op in math_ops {
            self.register_operator(ONNXOperatorInfo {
                name: op.to_string(),
                opset_version: 6,
                inputs: vec![ONNXOperatorInput {
                    name: "input".to_string(),
                    types: vec![
                        ONNXDataType::Float,
                        ONNXDataType::Double,
                        ONNXDataType::Float16,
                    ],
                    optional: false,
                    description: "Input tensor".to_string(),
                }],
                outputs: vec![ONNXOperatorOutput {
                    name: "output".to_string(),
                    types: vec![
                        ONNXDataType::Float,
                        ONNXDataType::Double,
                        ONNXDataType::Float16,
                    ],
                    description: "Output tensor".to_string(),
                }],
                attributes: vec![],
                description: format!("{} mathematical function", op),
            });
        }
    }

    fn register_neural_network_operators(&mut self) {
        // Convolution
        self.register_operator(ONNXOperatorInfo {
            name: "Conv".to_string(),
            opset_version: 11,
            inputs: vec![
                ONNXOperatorInput {
                    name: "X".to_string(),
                    types: vec![
                        ONNXDataType::Float,
                        ONNXDataType::Double,
                        ONNXDataType::Float16,
                    ],
                    optional: false,
                    description: "Input data tensor".to_string(),
                },
                ONNXOperatorInput {
                    name: "W".to_string(),
                    types: vec![
                        ONNXDataType::Float,
                        ONNXDataType::Double,
                        ONNXDataType::Float16,
                    ],
                    optional: false,
                    description: "Weight tensor".to_string(),
                },
                ONNXOperatorInput {
                    name: "B".to_string(),
                    types: vec![
                        ONNXDataType::Float,
                        ONNXDataType::Double,
                        ONNXDataType::Float16,
                    ],
                    optional: true,
                    description: "Optional bias tensor".to_string(),
                },
            ],
            outputs: vec![ONNXOperatorOutput {
                name: "Y".to_string(),
                types: vec![
                    ONNXDataType::Float,
                    ONNXDataType::Double,
                    ONNXDataType::Float16,
                ],
                description: "Output data tensor".to_string(),
            }],
            attributes: vec![
                ONNXOperatorAttribute {
                    name: "strides".to_string(),
                    attribute_type: "ints".to_string(),
                    required: false,
                    description: "Stride along each spatial axis".to_string(),
                },
                ONNXOperatorAttribute {
                    name: "pads".to_string(),
                    attribute_type: "ints".to_string(),
                    required: false,
                    description: "Padding for the beginning and ending".to_string(),
                },
                ONNXOperatorAttribute {
                    name: "dilations".to_string(),
                    attribute_type: "ints".to_string(),
                    required: false,
                    description: "Dilation value along each spatial axis".to_string(),
                },
                ONNXOperatorAttribute {
                    name: "group".to_string(),
                    attribute_type: "int".to_string(),
                    required: false,
                    description:
                        "Number of groups input channels and output channels are divided into"
                            .to_string(),
                },
            ],
            description: "Convolution operator".to_string(),
        });

        // Batch Normalization
        self.register_operator(ONNXOperatorInfo {
            name: "BatchNormalization".to_string(),
            opset_version: 15,
            inputs: vec![
                ONNXOperatorInput {
                    name: "X".to_string(),
                    types: vec![
                        ONNXDataType::Float,
                        ONNXDataType::Double,
                        ONNXDataType::Float16,
                    ],
                    optional: false,
                    description: "Input tensor".to_string(),
                },
                ONNXOperatorInput {
                    name: "scale".to_string(),
                    types: vec![
                        ONNXDataType::Float,
                        ONNXDataType::Double,
                        ONNXDataType::Float16,
                    ],
                    optional: false,
                    description: "Scale tensor".to_string(),
                },
                ONNXOperatorInput {
                    name: "B".to_string(),
                    types: vec![
                        ONNXDataType::Float,
                        ONNXDataType::Double,
                        ONNXDataType::Float16,
                    ],
                    optional: false,
                    description: "Bias tensor".to_string(),
                },
                ONNXOperatorInput {
                    name: "input_mean".to_string(),
                    types: vec![
                        ONNXDataType::Float,
                        ONNXDataType::Double,
                        ONNXDataType::Float16,
                    ],
                    optional: false,
                    description: "Running mean".to_string(),
                },
                ONNXOperatorInput {
                    name: "input_var".to_string(),
                    types: vec![
                        ONNXDataType::Float,
                        ONNXDataType::Double,
                        ONNXDataType::Float16,
                    ],
                    optional: false,
                    description: "Running variance".to_string(),
                },
            ],
            outputs: vec![ONNXOperatorOutput {
                name: "Y".to_string(),
                types: vec![
                    ONNXDataType::Float,
                    ONNXDataType::Double,
                    ONNXDataType::Float16,
                ],
                description: "Output tensor".to_string(),
            }],
            attributes: vec![ONNXOperatorAttribute {
                name: "epsilon".to_string(),
                attribute_type: "float".to_string(),
                required: false,
                description: "Small constant to avoid division by zero".to_string(),
            }],
            description: "Batch normalization operator".to_string(),
        });
    }

    fn register_tensor_operators(&mut self) {
        // Tensor shape manipulation
        let shape_ops = vec![
            ("Reshape", "Reshape the input tensor"),
            ("Transpose", "Transpose the input tensor"),
            ("Squeeze", "Remove single-dimensional entries"),
            ("Unsqueeze", "Insert single-dimensional entries"),
            ("Flatten", "Flatten the input tensor"),
        ];

        for (name, desc) in shape_ops {
            self.register_operator(ONNXOperatorInfo {
                name: name.to_string(),
                opset_version: 13,
                inputs: if name == "Reshape" {
                    vec![
                        ONNXOperatorInput {
                            name: "data".to_string(),
                            types: vec![
                                ONNXDataType::Float,
                                ONNXDataType::Double,
                                ONNXDataType::Int32,
                                ONNXDataType::Int64,
                            ],
                            optional: false,
                            description: "Input tensor".to_string(),
                        },
                        ONNXOperatorInput {
                            name: "shape".to_string(),
                            types: vec![ONNXDataType::Int64],
                            optional: false,
                            description: "New shape".to_string(),
                        },
                    ]
                } else {
                    vec![ONNXOperatorInput {
                        name: "data".to_string(),
                        types: vec![
                            ONNXDataType::Float,
                            ONNXDataType::Double,
                            ONNXDataType::Int32,
                            ONNXDataType::Int64,
                        ],
                        optional: false,
                        description: "Input tensor".to_string(),
                    }]
                },
                outputs: vec![ONNXOperatorOutput {
                    name: "reshaped".to_string(),
                    types: vec![
                        ONNXDataType::Float,
                        ONNXDataType::Double,
                        ONNXDataType::Int32,
                        ONNXDataType::Int64,
                    ],
                    description: "Reshaped tensor".to_string(),
                }],
                attributes: match name {
                    "Transpose" => vec![ONNXOperatorAttribute {
                        name: "perm".to_string(),
                        attribute_type: "ints".to_string(),
                        required: false,
                        description: "A list of integers. By default, reverse the dimensions"
                            .to_string(),
                    }],
                    "Flatten" => vec![ONNXOperatorAttribute {
                        name: "axis".to_string(),
                        attribute_type: "int".to_string(),
                        required: false,
                        description: "Indicate which axis to flatten".to_string(),
                    }],
                    _ => vec![],
                },
                description: desc.to_string(),
            });
        }

        // Reduction operations
        let reduce_ops = vec![
            "ReduceSum",
            "ReduceMean",
            "ReduceMax",
            "ReduceMin",
            "ReduceProd",
            "ReduceL1",
            "ReduceL2",
            "ReduceLogSum",
            "ReduceLogSumExp",
            "ReduceSumSquare",
        ];

        for op in reduce_ops {
            self.register_operator(ONNXOperatorInfo {
                name: op.to_string(),
                opset_version: 13,
                inputs: vec![ONNXOperatorInput {
                    name: "data".to_string(),
                    types: vec![
                        ONNXDataType::Float,
                        ONNXDataType::Double,
                        ONNXDataType::Int32,
                        ONNXDataType::Int64,
                    ],
                    optional: false,
                    description: "Input tensor".to_string(),
                }],
                outputs: vec![ONNXOperatorOutput {
                    name: "reduced".to_string(),
                    types: vec![
                        ONNXDataType::Float,
                        ONNXDataType::Double,
                        ONNXDataType::Int32,
                        ONNXDataType::Int64,
                    ],
                    description: "Reduced tensor".to_string(),
                }],
                attributes: vec![
                    ONNXOperatorAttribute {
                        name: "axes".to_string(),
                        attribute_type: "ints".to_string(),
                        required: false,
                        description: "A list of integers, along which to reduce".to_string(),
                    },
                    ONNXOperatorAttribute {
                        name: "keepdims".to_string(),
                        attribute_type: "int".to_string(),
                        required: false,
                        description: "Keep the reduced dimension or not".to_string(),
                    },
                ],
                description: format!("{} reduction operation", op),
            });
        }
    }

    fn register_control_flow_operators(&mut self) {
        // Control flow operations
        self.register_operator(ONNXOperatorInfo {
            name: "If".to_string(),
            opset_version: 11,
            inputs: vec![ONNXOperatorInput {
                name: "cond".to_string(),
                types: vec![ONNXDataType::Bool],
                optional: false,
                description: "Condition tensor".to_string(),
            }],
            outputs: vec![ONNXOperatorOutput {
                name: "outputs".to_string(),
                types: vec![
                    ONNXDataType::Float,
                    ONNXDataType::Double,
                    ONNXDataType::Int32,
                    ONNXDataType::Int64,
                ],
                description: "Output values".to_string(),
            }],
            attributes: vec![
                ONNXOperatorAttribute {
                    name: "then_branch".to_string(),
                    attribute_type: "string".to_string(),
                    required: true,
                    description: "Graph to run if condition is true".to_string(),
                },
                ONNXOperatorAttribute {
                    name: "else_branch".to_string(),
                    attribute_type: "string".to_string(),
                    required: true,
                    description: "Graph to run if condition is false".to_string(),
                },
            ],
            description: "Conditional execution".to_string(),
        });

        self.register_operator(ONNXOperatorInfo {
            name: "Loop".to_string(),
            opset_version: 13,
            inputs: vec![
                ONNXOperatorInput {
                    name: "M".to_string(),
                    types: vec![ONNXDataType::Int64],
                    optional: true,
                    description: "Maximum trip count".to_string(),
                },
                ONNXOperatorInput {
                    name: "cond".to_string(),
                    types: vec![ONNXDataType::Bool],
                    optional: true,
                    description: "Loop termination condition".to_string(),
                },
            ],
            outputs: vec![ONNXOperatorOutput {
                name: "v_final".to_string(),
                types: vec![
                    ONNXDataType::Float,
                    ONNXDataType::Double,
                    ONNXDataType::Int32,
                    ONNXDataType::Int64,
                ],
                description: "Final loop carried values".to_string(),
            }],
            attributes: vec![ONNXOperatorAttribute {
                name: "body".to_string(),
                attribute_type: "string".to_string(),
                required: true,
                description: "Graph to execute in the loop".to_string(),
            }],
            description: "Loop execution".to_string(),
        });
    }

    fn register_quantization_operators(&mut self) {
        // Quantization operations
        self.register_operator(ONNXOperatorInfo {
            name: "QuantizeLinear".to_string(),
            opset_version: 13,
            inputs: vec![
                ONNXOperatorInput {
                    name: "x".to_string(),
                    types: vec![ONNXDataType::Float, ONNXDataType::Int32],
                    optional: false,
                    description: "Input tensor".to_string(),
                },
                ONNXOperatorInput {
                    name: "y_scale".to_string(),
                    types: vec![ONNXDataType::Float],
                    optional: false,
                    description: "Scale for doing quantization".to_string(),
                },
                ONNXOperatorInput {
                    name: "y_zero_point".to_string(),
                    types: vec![ONNXDataType::UInt8, ONNXDataType::Int8],
                    optional: true,
                    description: "Zero point for quantization".to_string(),
                },
            ],
            outputs: vec![ONNXOperatorOutput {
                name: "y".to_string(),
                types: vec![ONNXDataType::UInt8, ONNXDataType::Int8],
                description: "Quantized output tensor".to_string(),
            }],
            attributes: vec![],
            description: "Linear quantization operator".to_string(),
        });

        self.register_operator(ONNXOperatorInfo {
            name: "DequantizeLinear".to_string(),
            opset_version: 13,
            inputs: vec![
                ONNXOperatorInput {
                    name: "x".to_string(),
                    types: vec![ONNXDataType::UInt8, ONNXDataType::Int8],
                    optional: false,
                    description: "Quantized input tensor".to_string(),
                },
                ONNXOperatorInput {
                    name: "x_scale".to_string(),
                    types: vec![ONNXDataType::Float],
                    optional: false,
                    description: "Scale for doing dequantization".to_string(),
                },
                ONNXOperatorInput {
                    name: "x_zero_point".to_string(),
                    types: vec![ONNXDataType::UInt8, ONNXDataType::Int8],
                    optional: true,
                    description: "Zero point for dequantization".to_string(),
                },
            ],
            outputs: vec![ONNXOperatorOutput {
                name: "y".to_string(),
                types: vec![ONNXDataType::Float],
                description: "Dequantized output tensor".to_string(),
            }],
            attributes: vec![],
            description: "Linear dequantization operator".to_string(),
        });
    }

    fn register_operator(&mut self, operator_info: ONNXOperatorInfo) {
        self.operators.insert(operator_info.name.clone(), operator_info);
    }

    pub fn get_operator_names(&self) -> Vec<String> {
        self.operators.keys().cloned().collect()
    }

    pub fn has_operator(&self, name: &str) -> bool {
        self.operators.contains_key(name)
    }

    pub fn get_operator(&self, name: &str) -> Option<&ONNXOperatorInfo> {
        self.operators.get(name)
    }

    pub fn get_operators_by_opset(&self, opset_version: i64) -> Vec<&ONNXOperatorInfo> {
        self.operators.values().filter(|op| op.opset_version <= opset_version).collect()
    }

    pub fn validate_operator_usage(
        &self,
        op_name: &str,
        inputs: &[String],
        attributes: &HashMap<String, ONNXAttribute>,
    ) -> Result<()> {
        let op_info = self
            .get_operator(op_name)
            .ok_or_else(|| anyhow!("Unsupported operator: {}", op_name))?;

        // Check required inputs
        let required_inputs = op_info.inputs.iter().filter(|input| !input.optional).count();

        if inputs.len() < required_inputs {
            return Err(anyhow!(
                "Operator {} requires at least {} inputs, got {}",
                op_name,
                required_inputs,
                inputs.len()
            ));
        }

        // Check required attributes
        for attr in &op_info.attributes {
            if attr.required && !attributes.contains_key(&attr.name) {
                return Err(anyhow!(
                    "Operator {} requires attribute '{}'",
                    op_name,
                    attr.name
                ));
            }
        }

        Ok(())
    }
}

impl ModelExporter for ONNXExporter {
    /// Always fails with a structured `UnsupportedOperation` error.
    ///
    /// See the [module documentation](self); use [`ONNXExporter::export_graph`]
    /// with an explicitly built [`ONNXModel`] to write a real `.onnx` file.
    fn export<M: Model>(&self, model: &M, config: &ExportConfig) -> Result<()> {
        if config.format != ExportFormat::ONNX {
            return Err(anyhow!("ONNXExporter only supports ONNX format"));
        }

        // Surface the "no weights at all" problem first: it is the caller's bug,
        // whereas the missing topology is a limitation of the `Model` trait.
        let _tensors = collect_model_tensors(model)?;
        Err(unsupported_operation("ONNX graph export", ONNX_TOPOLOGY_UNSUPPORTED_REASON).into())
    }

    fn supported_formats(&self) -> Vec<ExportFormat> {
        vec![ExportFormat::ONNX]
    }

    fn validate_model<M: Model>(&self, model: &M, format: ExportFormat) -> Result<()> {
        if format != ExportFormat::ONNX {
            return Err(anyhow!("ONNXExporter only supports ONNX format"));
        }
        collect_model_tensors(model)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::test_support::TestModel;

    fn float_tensor(name: &str, dims: Vec<i64>, values: &[f32]) -> ONNXTensor {
        ONNXTensor {
            name: name.to_string(),
            data_type: ONNXDataType::Float,
            dims,
            raw_data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    fn float_value_info(name: &str, dims: Vec<i64>) -> ONNXValueInfo {
        ONNXValueInfo {
            name: name.to_string(),
            type_info: ONNXTypeInfo {
                tensor_type: ONNXTensorType {
                    elem_type: ONNXDataType::Float,
                    shape: ONNXTensorShape {
                        dims: dims.into_iter().map(ONNXDimension::Value).collect(),
                    },
                },
            },
        }
    }

    fn tiny_graph() -> ONNXGraph {
        ONNXGraph {
            nodes: vec![ONNXNode {
                op_type: "Add".to_string(),
                inputs: vec!["x".to_string(), "b".to_string()],
                outputs: vec!["y".to_string()],
                attributes: HashMap::new(),
                name: "add0".to_string(),
            }],
            inputs: vec![float_value_info("x", vec![2])],
            outputs: vec![float_value_info("y", vec![2])],
            initializers: vec![float_tensor("b", vec![2], &[1.0, 2.0])],
            name: "tiny".to_string(),
        }
    }

    /// Regression test for the exporter that wrote `"IR Version: 8\nProducer: ..."`
    /// into a file named `*.onnx`.
    #[test]
    fn export_graph_writes_binary_protobuf_not_text() {
        let dir = std::env::temp_dir().join("trustformers_onnx_export_graph");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("tiny.onnx");

        // `ExportConfig::opset_version` takes precedence over the exporter default.
        let exporter = ONNXExporter::new().with_opset_version(14);
        let config = ExportConfig {
            opset_version: Some(17),
            ..Default::default()
        };
        let model = exporter.wrap_graph(tiny_graph(), &config);
        exporter.export_graph(&model, &path).expect("write");

        let bytes = std::fs::read(&path).expect("read back");
        assert!(!bytes.starts_with(b"IR Version"), "must not be a text dump");
        // ModelProto field 1 (ir_version) is a varint: key byte 0x08.
        assert_eq!(bytes[0], 0x08);

        let decoded = crate::export::onnx_proto::decode_model(&bytes).expect("parse");
        assert_eq!(decoded.graph.name, "tiny");
        assert_eq!(decoded.graph.nodes[0].op_type, "Add");
        assert_eq!(decoded.graph.initializers[0].raw_data.len(), 8);
        assert_eq!(decoded.opset_imports[0].version, 17);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn export_graph_rejects_dangling_inputs() {
        let mut graph = tiny_graph();
        graph.initializers.clear();
        let exporter = ONNXExporter::new();
        let model = exporter.wrap_graph(graph, &ExportConfig::default());
        let err = exporter.encode_graph(&model).expect_err("dangling input");
        assert!(err.to_string().contains("consumes 'b'"), "{err}");
    }

    #[test]
    fn export_graph_rejects_initializers_whose_bytes_do_not_match_their_dims() {
        let mut graph = tiny_graph();
        graph.initializers[0].dims = vec![7];
        let exporter = ONNXExporter::new();
        let model = exporter.wrap_graph(graph, &ExportConfig::default());
        let err = exporter.encode_graph(&model).expect_err("bad initializer");
        assert!(err.to_string().contains("carries 8 bytes"), "{err}");
    }

    #[test]
    fn export_graph_rejects_node_free_graphs() {
        let mut graph = tiny_graph();
        graph.nodes.clear();
        let exporter = ONNXExporter::new();
        let model = exporter.wrap_graph(graph, &ExportConfig::default());
        let err = exporter.encode_graph(&model).expect_err("weights-only is not a model");
        assert!(err.to_string().contains("no nodes"), "{err}");
    }

    /// The `ModelExporter` entry point must refuse rather than invent a topology.
    #[test]
    fn model_exporter_entry_point_refuses_to_invent_a_graph() {
        let dir = std::env::temp_dir().join("trustformers_onnx_export_refuse");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let output = dir.join("model");

        let config = ExportConfig {
            format: ExportFormat::ONNX,
            output_path: output.to_string_lossy().to_string(),
            ..Default::default()
        };
        let err = ONNXExporter::new()
            .export(&TestModel::with_seed(1.0), &config)
            .expect_err("no topology, no export");
        assert!(err.to_string().contains("Unsupported operation"), "{err}");
        assert!(!output.with_extension("onnx").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_exporter_reports_missing_weights_first() {
        let config = ExportConfig {
            format: ExportFormat::ONNX,
            ..Default::default()
        };
        let err = ONNXExporter::new()
            .export(&TestModel::empty(), &config)
            .expect_err("no weights, no export");
        assert!(err.to_string().contains("named_tensors"), "{err}");
    }

    #[test]
    fn onnx_data_type_round_trips_through_its_wire_value() {
        for data_type in [
            ONNXDataType::Float,
            ONNXDataType::Int64,
            ONNXDataType::Float16,
            ONNXDataType::Bool,
        ] {
            assert_eq!(
                ONNXDataType::from_i32(data_type as i32).expect("known type"),
                data_type
            );
        }
        assert!(ONNXDataType::from_i32(999).is_err());
    }

    #[test]
    fn test_onnx_exporter_creation() {
        let exporter = ONNXExporter::new();
        assert_eq!(exporter.opset_version, 14);

        let exporter_v13 = exporter.with_opset_version(13);
        assert_eq!(exporter_v13.opset_version, 13);
    }

    #[test]
    fn test_onnx_data_types() {
        assert_eq!(ONNXDataType::Float as i64, 1);
        assert_eq!(ONNXDataType::Float16 as i64, 10);
        assert_eq!(ONNXDataType::Int64 as i64, 7);
    }

    #[test]
    fn test_supported_formats() {
        let exporter = ONNXExporter::new();
        let formats = exporter.supported_formats();
        assert_eq!(formats.len(), 1);
        assert_eq!(formats[0], ExportFormat::ONNX);
    }

    #[test]
    fn test_onnx_dimension_types() {
        let dim_value = ONNXDimension::Value(512);
        let dim_param = ONNXDimension::Parameter("batch_size".to_string());

        match dim_value {
            ONNXDimension::Value(v) => assert_eq!(v, 512),
            _ => panic!("Expected Value dimension"),
        }

        match dim_param {
            ONNXDimension::Parameter(p) => assert_eq!(p, "batch_size"),
            _ => panic!("Expected Parameter dimension"),
        }
    }

    #[test]
    fn test_onnx_attribute_types() {
        let int_attr = ONNXAttribute::Int(42);
        let float_attr = ONNXAttribute::Float(std::f32::consts::PI);
        let string_attr = ONNXAttribute::String("test".to_string());

        match int_attr {
            ONNXAttribute::Int(v) => assert_eq!(v, 42),
            _ => panic!("Expected Int attribute"),
        }

        match float_attr {
            ONNXAttribute::Float(v) => assert!((v - std::f32::consts::PI).abs() < 1e-6),
            _ => panic!("Expected Float attribute"),
        }

        match string_attr {
            ONNXAttribute::String(s) => assert_eq!(s, "test"),
            _ => panic!("Expected String attribute"),
        }
    }
}
