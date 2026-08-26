//! Binary ONNX (protobuf) encoder and decoder, written by hand in pure Rust.
//!
//! ONNX models are serialised `onnx.ModelProto` messages. This module implements
//! the protobuf wire format directly — varints, length-delimited fields and fixed
//! 32-bit fields — for the subset of `onnx.proto` that TrustformeRS emits and
//! executes. No `protoc`, no generated code, no C dependency.
//!
//! # Covered messages and field numbers
//!
//! Field numbers are taken from the canonical `onnx.proto`:
//!
//! | Message | Fields |
//! |---|---|
//! | `ModelProto` | `ir_version=1`, `producer_name=2`, `producer_version=3`, `domain=4`, `model_version=5`, `doc_string=6`, `graph=7`, `opset_import=8` |
//! | `OperatorSetIdProto` | `domain=1`, `version=2` |
//! | `GraphProto` | `node=1`, `name=2`, `initializer=5`, `doc_string=10`, `input=11`, `output=12`, `value_info=13` |
//! | `NodeProto` | `input=1`, `output=2`, `name=3`, `op_type=4`, `attribute=5`, `doc_string=6`, `domain=7` |
//! | `ValueInfoProto` | `name=1`, `type=2`, `doc_string=3` |
//! | `TypeProto` | `tensor_type=1` |
//! | `TypeProto.Tensor` | `elem_type=1`, `shape=2` |
//! | `TensorShapeProto` | `dim=1` |
//! | `TensorShapeProto.Dimension` | `dim_value=1`, `dim_param=2` |
//! | `TensorProto` | `dims=1`, `data_type=2`, `float_data=4`, `int32_data=5`, `int64_data=7`, `name=8`, `raw_data=9`, `doc_string=12` |
//! | `AttributeProto` | `name=1`, `f=2`, `i=3`, `s=4`, `t=5`, `floats=7`, `ints=8`, `strings=9`, `doc_string=13`, `type=20` |
//!
//! Unknown fields encountered while decoding are skipped according to their wire
//! type, so models produced by other tools load as long as they only use the
//! messages above; anything this crate cannot represent is dropped rather than
//! guessed at, and the caller is told when that affects execution.

use super::onnx::{
    ONNXAttribute, ONNXDataType, ONNXDimension, ONNXGraph, ONNXModel, ONNXNode, ONNXOpsetImport,
    ONNXTensor, ONNXTensorShape, ONNXTensorType, ONNXTypeInfo, ONNXValueInfo,
};
use anyhow::{anyhow, Result};
use std::collections::HashMap;

/// Protobuf wire types.
mod wire_type {
    pub const VARINT: u32 = 0;
    pub const SIXTY_FOUR_BIT: u32 = 1;
    pub const LENGTH_DELIMITED: u32 = 2;
    pub const START_GROUP: u32 = 3;
    pub const END_GROUP: u32 = 4;
    pub const THIRTY_TWO_BIT: u32 = 5;
}

/// `AttributeProto.AttributeType` values.
mod attribute_type {
    pub const FLOAT: i64 = 1;
    pub const INT: i64 = 2;
    pub const STRING: i64 = 3;
    pub const TENSOR: i64 = 4;
    pub const FLOATS: i64 = 6;
    pub const INTS: i64 = 7;
    pub const STRINGS: i64 = 8;
}

// ---------------------------------------------------------------------------
// Wire primitives
// ---------------------------------------------------------------------------

/// Append a base-128 varint.
fn put_varint(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Append a field key (`field_number << 3 | wire_type`).
fn put_key(out: &mut Vec<u8>, field: u32, wire: u32) {
    put_varint(out, ((field as u64) << 3) | wire as u64);
}

fn put_varint_field(out: &mut Vec<u8>, field: u32, value: i64) {
    put_key(out, field, wire_type::VARINT);
    put_varint(out, value as u64);
}

fn put_bytes_field(out: &mut Vec<u8>, field: u32, value: &[u8]) {
    put_key(out, field, wire_type::LENGTH_DELIMITED);
    put_varint(out, value.len() as u64);
    out.extend_from_slice(value);
}

fn put_string_field(out: &mut Vec<u8>, field: u32, value: &str) {
    put_bytes_field(out, field, value.as_bytes());
}

fn put_float_field(out: &mut Vec<u8>, field: u32, value: f32) {
    put_key(out, field, wire_type::THIRTY_TWO_BIT);
    out.extend_from_slice(&value.to_le_bytes());
}

/// Append a nested message, encoded by `encode` into a scratch buffer.
fn put_message_field<F: FnOnce(&mut Vec<u8>)>(out: &mut Vec<u8>, field: u32, encode: F) {
    let mut nested = Vec::new();
    encode(&mut nested);
    put_bytes_field(out, field, &nested);
}

/// A cursor over protobuf bytes.
struct Reader<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    fn is_empty(&self) -> bool {
        self.position >= self.data.len()
    }

    fn read_varint(&mut self) -> Result<u64> {
        let mut result: u64 = 0;
        let mut shift = 0u32;
        loop {
            if self.position >= self.data.len() {
                return Err(anyhow!("protobuf: truncated varint"));
            }
            let byte = self.data[self.position];
            self.position += 1;
            if shift >= 64 {
                return Err(anyhow!("protobuf: varint longer than 64 bits"));
            }
            result |= u64::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
        }
    }

    fn read_key(&mut self) -> Result<(u32, u32)> {
        let key = self.read_varint()?;
        let field = (key >> 3) as u32;
        let wire = (key & 0x7) as u32;
        if field == 0 {
            return Err(anyhow!("protobuf: field number 0 is invalid"));
        }
        Ok((field, wire))
    }

    fn read_bytes(&mut self) -> Result<&'a [u8]> {
        let len = self.read_varint()? as usize;
        let end = self
            .position
            .checked_add(len)
            .ok_or_else(|| anyhow!("protobuf: length overflow"))?;
        if end > self.data.len() {
            return Err(anyhow!(
                "protobuf: length-delimited field of {len} bytes exceeds the message"
            ));
        }
        let slice = &self.data[self.position..end];
        self.position = end;
        Ok(slice)
    }

    fn read_string(&mut self) -> Result<String> {
        let bytes = self.read_bytes()?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| anyhow!("protobuf: field is not valid UTF-8: {e}"))
    }

    fn read_fixed32(&mut self) -> Result<[u8; 4]> {
        if self.position + 4 > self.data.len() {
            return Err(anyhow!("protobuf: truncated 32-bit field"));
        }
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&self.data[self.position..self.position + 4]);
        self.position += 4;
        Ok(buf)
    }

    fn read_fixed64(&mut self) -> Result<[u8; 8]> {
        if self.position + 8 > self.data.len() {
            return Err(anyhow!("protobuf: truncated 64-bit field"));
        }
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&self.data[self.position..self.position + 8]);
        self.position += 8;
        Ok(buf)
    }

    /// Skip a field of unknown meaning but known wire type.
    fn skip(&mut self, wire: u32) -> Result<()> {
        match wire {
            wire_type::VARINT => {
                self.read_varint()?;
            },
            wire_type::SIXTY_FOUR_BIT => {
                self.read_fixed64()?;
            },
            wire_type::LENGTH_DELIMITED => {
                self.read_bytes()?;
            },
            wire_type::THIRTY_TWO_BIT => {
                self.read_fixed32()?;
            },
            wire_type::START_GROUP | wire_type::END_GROUP => {
                return Err(anyhow!("protobuf: groups are not supported in ONNX"));
            },
            other => return Err(anyhow!("protobuf: unknown wire type {other}")),
        }
        Ok(())
    }
}

/// Read a `repeated` numeric field that may be packed or unpacked.
fn read_repeated_varint(reader: &mut Reader<'_>, wire: u32, out: &mut Vec<i64>) -> Result<()> {
    match wire {
        wire_type::VARINT => out.push(reader.read_varint()? as i64),
        wire_type::LENGTH_DELIMITED => {
            let bytes = reader.read_bytes()?;
            let mut packed = Reader::new(bytes);
            while !packed.is_empty() {
                out.push(packed.read_varint()? as i64);
            }
        },
        other => {
            return Err(anyhow!(
                "protobuf: repeated int field has wire type {other}"
            ))
        },
    }
    Ok(())
}

/// Read a `repeated float` field that may be packed or unpacked.
fn read_repeated_float(reader: &mut Reader<'_>, wire: u32, out: &mut Vec<f32>) -> Result<()> {
    match wire {
        wire_type::THIRTY_TWO_BIT => out.push(f32::from_le_bytes(reader.read_fixed32()?)),
        wire_type::LENGTH_DELIMITED => {
            let bytes = reader.read_bytes()?;
            if bytes.len() % 4 != 0 {
                return Err(anyhow!(
                    "protobuf: packed float field is not a multiple of 4"
                ));
            }
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        },
        other => {
            return Err(anyhow!(
                "protobuf: repeated float field has wire type {other}"
            ))
        },
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

/// Serialise a model as a binary `onnx.ModelProto`.
pub fn encode_model(model: &ONNXModel) -> Vec<u8> {
    let mut out = Vec::new();

    put_varint_field(&mut out, 1, model.ir_version);
    put_string_field(&mut out, 2, &model.producer_name);
    put_string_field(&mut out, 3, &model.producer_version);
    put_varint_field(&mut out, 5, model.model_version);
    put_message_field(&mut out, 7, |nested| encode_graph(&model.graph, nested));
    for opset in &model.opset_imports {
        put_message_field(&mut out, 8, |nested| {
            if !opset.domain.is_empty() {
                put_string_field(nested, 1, &opset.domain);
            }
            put_varint_field(nested, 2, opset.version);
        });
    }

    out
}

fn encode_graph(graph: &ONNXGraph, out: &mut Vec<u8>) {
    for node in &graph.nodes {
        put_message_field(out, 1, |nested| encode_node(node, nested));
    }
    put_string_field(out, 2, &graph.name);
    for initializer in &graph.initializers {
        put_message_field(out, 5, |nested| encode_tensor(initializer, nested));
    }
    for input in &graph.inputs {
        put_message_field(out, 11, |nested| encode_value_info(input, nested));
    }
    for output in &graph.outputs {
        put_message_field(out, 12, |nested| encode_value_info(output, nested));
    }
}

fn encode_node(node: &ONNXNode, out: &mut Vec<u8>) {
    for input in &node.inputs {
        put_string_field(out, 1, input);
    }
    for output in &node.outputs {
        put_string_field(out, 2, output);
    }
    put_string_field(out, 3, &node.name);
    put_string_field(out, 4, &node.op_type);

    // Attribute order is not significant on the wire, but a deterministic order
    // keeps exported files byte-reproducible.
    let mut names: Vec<&String> = node.attributes.keys().collect();
    names.sort();
    for name in names {
        if let Some(attribute) = node.attributes.get(name) {
            put_message_field(out, 5, |nested| encode_attribute(name, attribute, nested));
        }
    }
}

fn encode_attribute(name: &str, attribute: &ONNXAttribute, out: &mut Vec<u8>) {
    put_string_field(out, 1, name);
    match attribute {
        ONNXAttribute::Float(value) => {
            put_float_field(out, 2, *value);
            put_varint_field(out, 20, attribute_type::FLOAT);
        },
        ONNXAttribute::Int(value) => {
            put_varint_field(out, 3, *value);
            put_varint_field(out, 20, attribute_type::INT);
        },
        ONNXAttribute::String(value) => {
            put_bytes_field(out, 4, value.as_bytes());
            put_varint_field(out, 20, attribute_type::STRING);
        },
        ONNXAttribute::Tensor(tensor) => {
            put_message_field(out, 5, |nested| encode_tensor(tensor, nested));
            put_varint_field(out, 20, attribute_type::TENSOR);
        },
        ONNXAttribute::Floats(values) => {
            for value in values {
                put_float_field(out, 7, *value);
            }
            put_varint_field(out, 20, attribute_type::FLOATS);
        },
        ONNXAttribute::Ints(values) => {
            for value in values {
                put_varint_field(out, 8, *value);
            }
            put_varint_field(out, 20, attribute_type::INTS);
        },
        ONNXAttribute::Strings(values) => {
            for value in values {
                put_bytes_field(out, 9, value.as_bytes());
            }
            put_varint_field(out, 20, attribute_type::STRINGS);
        },
    }
}

fn encode_tensor(tensor: &ONNXTensor, out: &mut Vec<u8>) {
    for dim in &tensor.dims {
        put_varint_field(out, 1, *dim);
    }
    put_varint_field(out, 2, tensor.data_type as i64);
    put_string_field(out, 8, &tensor.name);
    put_bytes_field(out, 9, &tensor.raw_data);
}

fn encode_value_info(value_info: &ONNXValueInfo, out: &mut Vec<u8>) {
    put_string_field(out, 1, &value_info.name);
    put_message_field(out, 2, |type_proto| {
        put_message_field(type_proto, 1, |tensor_type| {
            put_varint_field(
                tensor_type,
                1,
                value_info.type_info.tensor_type.elem_type as i64,
            );
            put_message_field(tensor_type, 2, |shape| {
                for dim in &value_info.type_info.tensor_type.shape.dims {
                    put_message_field(shape, 1, |dimension| match dim {
                        ONNXDimension::Value(value) => put_varint_field(dimension, 1, *value),
                        ONNXDimension::Parameter(name) => put_string_field(dimension, 2, name),
                    });
                }
            });
        });
    });
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

/// Parse a binary `onnx.ModelProto`.
pub fn decode_model(bytes: &[u8]) -> Result<ONNXModel> {
    let mut reader = Reader::new(bytes);
    let mut model = ONNXModel {
        graph: ONNXGraph {
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            initializers: Vec::new(),
            name: String::new(),
        },
        ir_version: 0,
        opset_imports: Vec::new(),
        producer_name: String::new(),
        producer_version: String::new(),
        model_version: 0,
    };
    let mut saw_graph = false;

    while !reader.is_empty() {
        let (field, wire) = reader.read_key()?;
        match (field, wire) {
            (1, wire_type::VARINT) => model.ir_version = reader.read_varint()? as i64,
            (2, wire_type::LENGTH_DELIMITED) => model.producer_name = reader.read_string()?,
            (3, wire_type::LENGTH_DELIMITED) => model.producer_version = reader.read_string()?,
            (5, wire_type::VARINT) => model.model_version = reader.read_varint()? as i64,
            (7, wire_type::LENGTH_DELIMITED) => {
                model.graph = decode_graph(reader.read_bytes()?)?;
                saw_graph = true;
            },
            (8, wire_type::LENGTH_DELIMITED) => {
                model.opset_imports.push(decode_opset(reader.read_bytes()?)?);
            },
            _ => reader.skip(wire)?,
        }
    }

    if !saw_graph {
        return Err(anyhow!(
            "ONNX ModelProto contains no GraphProto (field 7); this is not a usable model"
        ));
    }

    Ok(model)
}

fn decode_opset(bytes: &[u8]) -> Result<ONNXOpsetImport> {
    let mut reader = Reader::new(bytes);
    let mut opset = ONNXOpsetImport {
        domain: String::new(),
        version: 0,
    };
    while !reader.is_empty() {
        let (field, wire) = reader.read_key()?;
        match (field, wire) {
            (1, wire_type::LENGTH_DELIMITED) => opset.domain = reader.read_string()?,
            (2, wire_type::VARINT) => opset.version = reader.read_varint()? as i64,
            _ => reader.skip(wire)?,
        }
    }
    Ok(opset)
}

fn decode_graph(bytes: &[u8]) -> Result<ONNXGraph> {
    let mut reader = Reader::new(bytes);
    let mut graph = ONNXGraph {
        nodes: Vec::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        initializers: Vec::new(),
        name: String::new(),
    };

    while !reader.is_empty() {
        let (field, wire) = reader.read_key()?;
        match (field, wire) {
            (1, wire_type::LENGTH_DELIMITED) => {
                graph.nodes.push(decode_node(reader.read_bytes()?)?);
            },
            (2, wire_type::LENGTH_DELIMITED) => graph.name = reader.read_string()?,
            (5, wire_type::LENGTH_DELIMITED) => {
                graph.initializers.push(decode_tensor(reader.read_bytes()?)?);
            },
            (11, wire_type::LENGTH_DELIMITED) => {
                graph.inputs.push(decode_value_info(reader.read_bytes()?)?);
            },
            (12, wire_type::LENGTH_DELIMITED) => {
                graph.outputs.push(decode_value_info(reader.read_bytes()?)?);
            },
            _ => reader.skip(wire)?,
        }
    }

    Ok(graph)
}

fn decode_node(bytes: &[u8]) -> Result<ONNXNode> {
    let mut reader = Reader::new(bytes);
    let mut node = ONNXNode {
        op_type: String::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        attributes: HashMap::new(),
        name: String::new(),
    };

    while !reader.is_empty() {
        let (field, wire) = reader.read_key()?;
        match (field, wire) {
            (1, wire_type::LENGTH_DELIMITED) => node.inputs.push(reader.read_string()?),
            (2, wire_type::LENGTH_DELIMITED) => node.outputs.push(reader.read_string()?),
            (3, wire_type::LENGTH_DELIMITED) => node.name = reader.read_string()?,
            (4, wire_type::LENGTH_DELIMITED) => node.op_type = reader.read_string()?,
            (5, wire_type::LENGTH_DELIMITED) => {
                let (name, attribute) = decode_attribute(reader.read_bytes()?)?;
                node.attributes.insert(name, attribute);
            },
            _ => reader.skip(wire)?,
        }
    }

    Ok(node)
}

fn decode_attribute(bytes: &[u8]) -> Result<(String, ONNXAttribute)> {
    let mut reader = Reader::new(bytes);
    let mut name = String::new();
    let mut declared_type: Option<i64> = None;
    let mut float_value: Option<f32> = None;
    let mut int_value: Option<i64> = None;
    let mut string_value: Option<String> = None;
    let mut tensor_value: Option<ONNXTensor> = None;
    let mut floats: Vec<f32> = Vec::new();
    let mut ints: Vec<i64> = Vec::new();
    let mut strings: Vec<String> = Vec::new();

    while !reader.is_empty() {
        let (field, wire) = reader.read_key()?;
        match field {
            1 if wire == wire_type::LENGTH_DELIMITED => name = reader.read_string()?,
            2 if wire == wire_type::THIRTY_TWO_BIT => {
                float_value = Some(f32::from_le_bytes(reader.read_fixed32()?));
            },
            3 if wire == wire_type::VARINT => int_value = Some(reader.read_varint()? as i64),
            4 if wire == wire_type::LENGTH_DELIMITED => {
                string_value = Some(reader.read_string()?);
            },
            5 if wire == wire_type::LENGTH_DELIMITED => {
                tensor_value = Some(decode_tensor(reader.read_bytes()?)?);
            },
            7 => read_repeated_float(&mut reader, wire, &mut floats)?,
            8 => read_repeated_varint(&mut reader, wire, &mut ints)?,
            9 if wire == wire_type::LENGTH_DELIMITED => strings.push(reader.read_string()?),
            20 if wire == wire_type::VARINT => {
                declared_type = Some(reader.read_varint()? as i64);
            },
            _ => reader.skip(wire)?,
        }
    }

    // Prefer the declared type; fall back to whichever payload is present, which
    // is how attributes written by proto2 producers without `type` are handled.
    let attribute = match declared_type {
        Some(attribute_type::FLOAT) => ONNXAttribute::Float(float_value.unwrap_or(0.0)),
        Some(attribute_type::INT) => ONNXAttribute::Int(int_value.unwrap_or(0)),
        Some(attribute_type::STRING) => ONNXAttribute::String(string_value.unwrap_or_default()),
        Some(attribute_type::TENSOR) => ONNXAttribute::Tensor(tensor_value.ok_or_else(|| {
            anyhow!("ONNX attribute '{name}' declares TENSOR but carries no tensor")
        })?),
        Some(attribute_type::FLOATS) => ONNXAttribute::Floats(floats),
        Some(attribute_type::INTS) => ONNXAttribute::Ints(ints),
        Some(attribute_type::STRINGS) => ONNXAttribute::Strings(strings),
        _ => {
            if let Some(tensor) = tensor_value {
                ONNXAttribute::Tensor(tensor)
            } else if !floats.is_empty() {
                ONNXAttribute::Floats(floats)
            } else if !ints.is_empty() {
                ONNXAttribute::Ints(ints)
            } else if !strings.is_empty() {
                ONNXAttribute::Strings(strings)
            } else if let Some(value) = float_value {
                ONNXAttribute::Float(value)
            } else if let Some(value) = int_value {
                ONNXAttribute::Int(value)
            } else if let Some(value) = string_value {
                ONNXAttribute::String(value)
            } else {
                return Err(anyhow!(
                    "ONNX attribute '{name}' carries no value this build understands"
                ));
            }
        },
    };

    Ok((name, attribute))
}

fn decode_tensor(bytes: &[u8]) -> Result<ONNXTensor> {
    let mut reader = Reader::new(bytes);
    let mut dims: Vec<i64> = Vec::new();
    let mut data_type = ONNXDataType::Float;
    let mut name = String::new();
    let mut raw_data: Vec<u8> = Vec::new();
    let mut float_data: Vec<f32> = Vec::new();
    let mut int32_data: Vec<i64> = Vec::new();
    let mut int64_data: Vec<i64> = Vec::new();

    while !reader.is_empty() {
        let (field, wire) = reader.read_key()?;
        match field {
            1 => read_repeated_varint(&mut reader, wire, &mut dims)?,
            2 if wire == wire_type::VARINT => {
                data_type = ONNXDataType::from_i32(reader.read_varint()? as i32)?;
            },
            4 => read_repeated_float(&mut reader, wire, &mut float_data)?,
            5 => read_repeated_varint(&mut reader, wire, &mut int32_data)?,
            7 => read_repeated_varint(&mut reader, wire, &mut int64_data)?,
            8 if wire == wire_type::LENGTH_DELIMITED => name = reader.read_string()?,
            9 if wire == wire_type::LENGTH_DELIMITED => {
                raw_data = reader.read_bytes()?.to_vec();
            },
            _ => reader.skip(wire)?,
        }
    }

    // ONNX allows the payload in either `raw_data` or a typed `*_data` list.
    // Normalise onto `raw_data` so downstream code has one representation.
    if raw_data.is_empty() {
        if !float_data.is_empty() {
            raw_data = float_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        } else if !int64_data.is_empty() {
            raw_data = int64_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        } else if !int32_data.is_empty() {
            raw_data = int32_data.iter().flat_map(|v| (*v as i32).to_le_bytes()).collect();
        }
    }

    Ok(ONNXTensor {
        name,
        data_type,
        dims,
        raw_data,
    })
}

fn decode_value_info(bytes: &[u8]) -> Result<ONNXValueInfo> {
    let mut reader = Reader::new(bytes);
    let mut name = String::new();
    let mut elem_type = ONNXDataType::Float;
    let mut dims: Vec<ONNXDimension> = Vec::new();

    while !reader.is_empty() {
        let (field, wire) = reader.read_key()?;
        match (field, wire) {
            (1, wire_type::LENGTH_DELIMITED) => name = reader.read_string()?,
            (2, wire_type::LENGTH_DELIMITED) => {
                let (parsed_elem_type, parsed_dims) = decode_type_proto(reader.read_bytes()?)?;
                elem_type = parsed_elem_type;
                dims = parsed_dims;
            },
            _ => reader.skip(wire)?,
        }
    }

    Ok(ONNXValueInfo {
        name,
        type_info: ONNXTypeInfo {
            tensor_type: ONNXTensorType {
                elem_type,
                shape: ONNXTensorShape { dims },
            },
        },
    })
}

fn decode_type_proto(bytes: &[u8]) -> Result<(ONNXDataType, Vec<ONNXDimension>)> {
    let mut reader = Reader::new(bytes);
    let mut elem_type = ONNXDataType::Float;
    let mut dims = Vec::new();

    while !reader.is_empty() {
        let (field, wire) = reader.read_key()?;
        match (field, wire) {
            (1, wire_type::LENGTH_DELIMITED) => {
                let (parsed_elem_type, parsed_dims) = decode_tensor_type(reader.read_bytes()?)?;
                elem_type = parsed_elem_type;
                dims = parsed_dims;
            },
            _ => reader.skip(wire)?,
        }
    }

    Ok((elem_type, dims))
}

fn decode_tensor_type(bytes: &[u8]) -> Result<(ONNXDataType, Vec<ONNXDimension>)> {
    let mut reader = Reader::new(bytes);
    let mut elem_type = ONNXDataType::Float;
    let mut dims = Vec::new();

    while !reader.is_empty() {
        let (field, wire) = reader.read_key()?;
        match (field, wire) {
            (1, wire_type::VARINT) => {
                elem_type = ONNXDataType::from_i32(reader.read_varint()? as i32)?;
            },
            (2, wire_type::LENGTH_DELIMITED) => {
                dims = decode_tensor_shape(reader.read_bytes()?)?;
            },
            _ => reader.skip(wire)?,
        }
    }

    Ok((elem_type, dims))
}

fn decode_tensor_shape(bytes: &[u8]) -> Result<Vec<ONNXDimension>> {
    let mut reader = Reader::new(bytes);
    let mut dims = Vec::new();

    while !reader.is_empty() {
        let (field, wire) = reader.read_key()?;
        match (field, wire) {
            (1, wire_type::LENGTH_DELIMITED) => {
                dims.push(decode_dimension(reader.read_bytes()?)?);
            },
            _ => reader.skip(wire)?,
        }
    }

    Ok(dims)
}

fn decode_dimension(bytes: &[u8]) -> Result<ONNXDimension> {
    let mut reader = Reader::new(bytes);
    let mut dimension = ONNXDimension::Parameter(String::new());

    while !reader.is_empty() {
        let (field, wire) = reader.read_key()?;
        match (field, wire) {
            (1, wire_type::VARINT) => {
                dimension = ONNXDimension::Value(reader.read_varint()? as i64);
            },
            (2, wire_type::LENGTH_DELIMITED) => {
                dimension = ONNXDimension::Parameter(reader.read_string()?);
            },
            _ => reader.skip(wire)?,
        }
    }

    Ok(dimension)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_tensor(name: &str, values: &[f32]) -> ONNXTensor {
        ONNXTensor {
            name: name.to_string(),
            data_type: ONNXDataType::Float,
            dims: vec![values.len() as i64],
            raw_data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    fn one_node_model() -> ONNXModel {
        let mut attributes = HashMap::new();
        attributes.insert("alpha".to_string(), ONNXAttribute::Float(0.5));
        attributes.insert("axes".to_string(), ONNXAttribute::Ints(vec![-1]));

        ONNXModel {
            graph: ONNXGraph {
                nodes: vec![ONNXNode {
                    op_type: "Add".to_string(),
                    inputs: vec!["x".to_string(), "w".to_string()],
                    outputs: vec!["y".to_string()],
                    attributes,
                    name: "add0".to_string(),
                }],
                inputs: vec![ONNXValueInfo {
                    name: "x".to_string(),
                    type_info: ONNXTypeInfo {
                        tensor_type: ONNXTensorType {
                            elem_type: ONNXDataType::Float,
                            shape: ONNXTensorShape {
                                dims: vec![
                                    ONNXDimension::Parameter("batch".to_string()),
                                    ONNXDimension::Value(3),
                                ],
                            },
                        },
                    },
                }],
                outputs: vec![ONNXValueInfo {
                    name: "y".to_string(),
                    type_info: ONNXTypeInfo {
                        tensor_type: ONNXTensorType {
                            elem_type: ONNXDataType::Float,
                            shape: ONNXTensorShape {
                                dims: vec![
                                    ONNXDimension::Parameter("batch".to_string()),
                                    ONNXDimension::Value(3),
                                ],
                            },
                        },
                    },
                }],
                initializers: vec![scalar_tensor("w", &[1.0, 2.0, 3.0])],
                name: "g".to_string(),
            },
            ir_version: 8,
            opset_imports: vec![ONNXOpsetImport {
                domain: String::new(),
                version: 17,
            }],
            producer_name: "TrustformeRS".to_string(),
            producer_version: "0.2.1".to_string(),
            model_version: 1,
        }
    }

    #[test]
    fn varints_match_the_protobuf_specification() {
        let mut out = Vec::new();
        put_varint(&mut out, 0);
        assert_eq!(out, vec![0x00]);

        out.clear();
        put_varint(&mut out, 1);
        assert_eq!(out, vec![0x01]);

        out.clear();
        put_varint(&mut out, 300);
        assert_eq!(out, vec![0xAC, 0x02], "300 encodes as AC 02");

        out.clear();
        put_varint(&mut out, u64::MAX);
        assert_eq!(out.len(), 10);

        let mut reader = Reader::new(&[0xAC, 0x02]);
        assert_eq!(reader.read_varint().expect("varint"), 300);
    }

    /// Hand-computed wire bytes: only a self-consistent round trip could hide a
    /// wrong field number, so this pins the actual encoding down.
    #[test]
    fn tensor_proto_field_numbers_match_onnx_proto() {
        let tensor = ONNXTensor {
            name: "w".to_string(),
            data_type: ONNXDataType::Float,
            dims: vec![2],
            raw_data: vec![0xDE, 0xAD],
        };
        let mut encoded = Vec::new();
        encode_tensor(&tensor, &mut encoded);

        // dims (field 1, varint) -> key 0x08, value 2
        // data_type (field 2, varint) -> key 0x10, value 1 (FLOAT)
        // name (field 8, len) -> key 0x42, len 1, 'w'
        // raw_data (field 9, len) -> key 0x4A, len 2, DE AD
        assert_eq!(
            encoded,
            vec![0x08, 0x02, 0x10, 0x01, 0x42, 0x01, b'w', 0x4A, 0x02, 0xDE, 0xAD]
        );
    }

    #[test]
    fn model_proto_starts_with_ir_version_field_one() {
        let bytes = encode_model(&one_node_model());
        // field 1, wire type 0 -> key byte 0x08; ir_version 8 -> 0x08
        assert_eq!(&bytes[0..2], &[0x08, 0x08]);
    }

    #[test]
    fn model_round_trips_through_the_wire_format() {
        let original = one_node_model();
        let bytes = encode_model(&original);
        let decoded = decode_model(&bytes).expect("decode");

        assert_eq!(decoded.ir_version, 8);
        assert_eq!(decoded.producer_name, "TrustformeRS");
        assert_eq!(decoded.producer_version, "0.2.1");
        assert_eq!(decoded.model_version, 1);
        assert_eq!(decoded.opset_imports.len(), 1);
        assert_eq!(decoded.opset_imports[0].version, 17);
        assert_eq!(decoded.graph.name, "g");

        assert_eq!(decoded.graph.nodes.len(), 1);
        let node = &decoded.graph.nodes[0];
        assert_eq!(node.op_type, "Add");
        assert_eq!(node.name, "add0");
        assert_eq!(node.inputs, vec!["x".to_string(), "w".to_string()]);
        assert_eq!(node.outputs, vec!["y".to_string()]);
        assert!(
            matches!(node.attributes.get("alpha"), Some(ONNXAttribute::Float(v)) if (*v - 0.5).abs() < 1e-6)
        );
        assert!(
            matches!(node.attributes.get("axes"), Some(ONNXAttribute::Ints(v)) if v == &vec![-1])
        );

        assert_eq!(decoded.graph.initializers.len(), 1);
        assert_eq!(decoded.graph.initializers[0].name, "w");
        assert_eq!(decoded.graph.initializers[0].dims, vec![3]);
        assert_eq!(decoded.graph.initializers[0].raw_data.len(), 12);

        assert_eq!(decoded.graph.inputs.len(), 1);
        assert_eq!(decoded.graph.inputs[0].name, "x");
        assert!(matches!(
            &decoded.graph.inputs[0].type_info.tensor_type.shape.dims[0],
            ONNXDimension::Parameter(p) if p == "batch"
        ));
        assert!(matches!(
            decoded.graph.inputs[0].type_info.tensor_type.shape.dims[1],
            ONNXDimension::Value(3)
        ));
    }

    #[test]
    fn decoder_accepts_packed_repeated_fields() {
        // A TensorProto whose int64_data is packed rather than repeated.
        let mut bytes = Vec::new();
        put_varint_field(&mut bytes, 1, 3); // dims = 3
        put_varint_field(&mut bytes, 2, ONNXDataType::Int64 as i64);
        put_string_field(&mut bytes, 8, "ids");
        let mut packed = Vec::new();
        for value in [7i64, 8, 9] {
            put_varint(&mut packed, value as u64);
        }
        put_bytes_field(&mut bytes, 7, &packed);

        let tensor = decode_tensor(&bytes).expect("decode");
        assert_eq!(tensor.dims, vec![3]);
        assert_eq!(tensor.raw_data.len(), 24, "normalised into raw_data");
        let values: Vec<i64> = tensor
            .raw_data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect();
        assert_eq!(values, vec![7, 8, 9]);
    }

    #[test]
    fn decoder_skips_unknown_fields() {
        let mut bytes = encode_model(&one_node_model());
        // Append an unknown varint field (field 1000) and an unknown string field.
        put_varint_field(&mut bytes, 1000, 42);
        put_string_field(&mut bytes, 1001, "ignored");
        let decoded = decode_model(&bytes).expect("unknown fields must be skipped");
        assert_eq!(decoded.graph.nodes.len(), 1);
    }

    #[test]
    fn decoder_rejects_text_files() {
        let err = decode_model(b"IR Version: 8\nProducer: TrustformeRS\nGraph:\n")
            .expect_err("a text dump is not an ONNX model");
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn decoder_rejects_a_model_without_a_graph() {
        let mut bytes = Vec::new();
        put_varint_field(&mut bytes, 1, 8);
        put_string_field(&mut bytes, 2, "someone");
        let err = decode_model(&bytes).expect_err("a model needs a graph");
        assert!(err.to_string().contains("no GraphProto"), "{err}");
    }

    #[test]
    fn decoder_rejects_truncated_length_prefixes() {
        let mut bytes = Vec::new();
        put_key(&mut bytes, 7, wire_type::LENGTH_DELIMITED);
        put_varint(&mut bytes, 1000); // claims 1000 bytes that are not there
        let err = decode_model(&bytes).expect_err("truncated message must fail");
        assert!(err.to_string().contains("exceeds the message"), "{err}");
    }
}
