//! Low-level tensor/weight-format conversion helpers: natural (numeric-aware) sorting of tensor names, little-endian byte decoding, and SafeTensors/ONNX -> `Tensor` conversion.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{ArrayD, IxDyn};
use std::cmp::Ordering;
use trustformers_core::errors::{invalid_format, Result};
use trustformers_core::Tensor;

/// One run of a "natural sort" key: either a digit run (compared as a
/// number, so `"2"` sorts before `"10"`) or a non-digit run (compared as
/// text).
#[derive(Debug, PartialEq, Eq)]
pub(super) enum NaturalPart {
    Num(u64),
    Text(String),
}
/// Split `s` into alternating digit/non-digit runs for [`natural_cmp`].
fn natural_key(s: &str) -> Vec<NaturalPart> {
    let mut parts = Vec::new();
    let mut chars = s.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_ascii_digit() {
            let mut num_str = String::new();
            while let Some(&d) = chars.peek() {
                if d.is_ascii_digit() {
                    num_str.push(d);
                    chars.next();
                } else {
                    break;
                }
            }
            // A digit run longer than u64::MAX's digit count cannot occur in
            // any realistic tensor name; saturate rather than panic if it
            // somehow does.
            parts.push(NaturalPart::Num(num_str.parse().unwrap_or(u64::MAX)));
        } else {
            let mut text = String::new();
            while let Some(&d) = chars.peek() {
                if !d.is_ascii_digit() {
                    text.push(d);
                    chars.next();
                } else {
                    break;
                }
            }
            parts.push(NaturalPart::Text(text));
        }
    }
    parts
}

/// Compare two tensor names so that embedded integers order numerically
/// (`"layer.2.weight"` < `"layer.10.weight"`) while everything else orders
/// lexically. Falls back to a plain string comparison when the natural keys
/// tie (e.g. one name is a strict prefix of the other), which keeps the
/// order total and deterministic.
///
/// `pub(crate)` so other checkpoint-driven, architecture-free consumers of a
/// flat named-tensor bag (e.g. `coreml_converter`'s layer derivation) sort
/// checkpoint tensor names the same, single, correct way this crate has --
/// rather than each reimplementing (and potentially disagreeing on) numeric-
/// aware name ordering.
pub(crate) fn natural_cmp(a: &str, b: &str) -> Ordering {
    let ka = natural_key(a);
    let kb = natural_key(b);
    for (pa, pb) in ka.iter().zip(kb.iter()) {
        let ord = match (pa, pb) {
            (NaturalPart::Num(x), NaturalPart::Num(y)) => x.cmp(y),
            (NaturalPart::Text(x), NaturalPart::Text(y)) => x.cmp(y),
            // A digit run and a text run at the same position: digits sort
            // first, matching the common "prefix.N" < "prefix.suffix" case.
            (NaturalPart::Num(_), NaturalPart::Text(_)) => Ordering::Less,
            (NaturalPart::Text(_), NaturalPart::Num(_)) => Ordering::Greater,
        };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    ka.len().cmp(&kb.len()).then_with(|| a.cmp(b))
}

/// Decode a little-endian `f32` buffer. Errors (rather than silently
/// truncating) when the byte count is not a multiple of the element width.
pub(super) fn le_f32_vec(name: &str, data: &[u8]) -> Result<Vec<f32>> {
    if !data.len().is_multiple_of(4) {
        return Err(invalid_format(
            "a byte length that is a multiple of 4 for an F32 tensor",
            format!("tensor '{name}' has {} bytes", data.len()),
        ));
    }
    Ok(data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Decode a little-endian `f64` buffer.
pub(super) fn le_f64_vec(name: &str, data: &[u8]) -> Result<Vec<f64>> {
    if !data.len().is_multiple_of(8) {
        return Err(invalid_format(
            "a byte length that is a multiple of 8 for an F64 tensor",
            format!("tensor '{name}' has {} bytes", data.len()),
        ));
    }
    Ok(data
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect())
}

/// Decode a little-endian `i64` buffer.
pub(super) fn le_i64_vec(name: &str, data: &[u8]) -> Result<Vec<i64>> {
    if !data.len().is_multiple_of(8) {
        return Err(invalid_format(
            "a byte length that is a multiple of 8 for an I64 tensor",
            format!("tensor '{name}' has {} bytes", data.len()),
        ));
    }
    Ok(data
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect())
}

/// Decode a little-endian IEEE-754 half-precision buffer.
pub(super) fn le_f16_vec(name: &str, data: &[u8]) -> Result<Vec<half::f16>> {
    if !data.len().is_multiple_of(2) {
        return Err(invalid_format(
            "a byte length that is a multiple of 2 for an F16 tensor",
            format!("tensor '{name}' has {} bytes", data.len()),
        ));
    }
    Ok(data.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]])).collect())
}

/// Decode a little-endian bfloat16 buffer.
pub(super) fn le_bf16_vec(name: &str, data: &[u8]) -> Result<Vec<half::bf16>> {
    if !data.len().is_multiple_of(2) {
        return Err(invalid_format(
            "a byte length that is a multiple of 2 for a BF16 tensor",
            format!("tensor '{name}' has {} bytes", data.len()),
        ));
    }
    Ok(data.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]])).collect())
}

/// Build a [`Tensor`] from a shape and a `Vec` of already-decoded elements,
/// reporting a shape mismatch (never panicking) if the element count is
/// wrong for the declared shape.
pub(super) fn array_tensor<T, F>(
    name: &str,
    shape: &[usize],
    data: Vec<T>,
    wrap: F,
) -> Result<Tensor>
where
    F: FnOnce(ArrayD<T>) -> Tensor,
{
    let expected: usize = shape.iter().product();
    if data.len() != expected {
        return Err(invalid_format(
            format!("{expected} elements for shape {shape:?}"),
            format!("tensor '{name}' decoded to {} elements", data.len()),
        ));
    }
    let array = ArrayD::from_shape_vec(IxDyn(shape), data).map_err(|e| {
        invalid_format("a shape-compatible buffer", format!("tensor '{name}': {e}"))
    })?;
    Ok(wrap(array))
}

/// Convert one safetensors [`View`](safetensors::tensor::View) into a real
/// [`Tensor`] holding that view's own bytes. Returns `Ok(None)` for a dtype
/// this engine does not (yet) have a `Tensor` variant for, so the caller can
/// skip it without fabricating data in its place.
pub(super) fn safetensors_view_to_tensor(
    name: &str,
    dtype: safetensors::Dtype,
    shape: &[usize],
    data: &[u8],
) -> Result<Option<Tensor>> {
    use safetensors::Dtype;
    Ok(Some(match dtype {
        Dtype::F32 => array_tensor(name, shape, le_f32_vec(name, data)?, Tensor::F32)?,
        Dtype::F64 => array_tensor(name, shape, le_f64_vec(name, data)?, Tensor::F64)?,
        Dtype::I64 => array_tensor(name, shape, le_i64_vec(name, data)?, Tensor::I64)?,
        Dtype::F16 => array_tensor(name, shape, le_f16_vec(name, data)?, Tensor::F16)?,
        Dtype::BF16 => array_tensor(name, shape, le_bf16_vec(name, data)?, Tensor::BF16)?,
        _ => return Ok(None),
    }))
}

/// Convert one ONNX `TensorProto` initializer into a real [`Tensor`] holding
/// that initializer's own `raw_data`. Returns `Ok(None)` for a dtype this
/// engine does not (yet) have a `Tensor` variant for.
pub(super) fn onnx_tensor_to_tensor(
    name: &str,
    dtype: trustformers_core::export::ONNXDataType,
    shape: &[usize],
    raw: &[u8],
) -> Result<Option<Tensor>> {
    use trustformers_core::export::ONNXDataType;
    Ok(Some(match dtype {
        ONNXDataType::Float => array_tensor(name, shape, le_f32_vec(name, raw)?, Tensor::F32)?,
        ONNXDataType::Double => array_tensor(name, shape, le_f64_vec(name, raw)?, Tensor::F64)?,
        ONNXDataType::Int64 => array_tensor(name, shape, le_i64_vec(name, raw)?, Tensor::I64)?,
        ONNXDataType::Float16 => array_tensor(name, shape, le_f16_vec(name, raw)?, Tensor::F16)?,
        ONNXDataType::BFloat16 => array_tensor(name, shape, le_bf16_vec(name, raw)?, Tensor::BF16)?,
        _ => return Ok(None),
    }))
}
