//! A real, pure-Rust reader for the SafeTensors file format.
//!
//! Layout (see <https://huggingface.co/docs/safetensors>):
//! ```text
//! [0..8)   little-endian u64: N, the byte length of the JSON header
//! [8..8+N) UTF-8 JSON: { "tensor_name": {"dtype": ..., "shape": [...], "data_offsets": [start,end]}, ...,
//!                        "__metadata__": { ... } (optional, skipped) }
//! [8+N..)  raw tensor bytes; a given tensor's bytes are
//!          buffer[8+N+start .. 8+N+end)
//! ```
//!
//! Only float dtypes usable for CPU inference in this crate are supported
//! (F32/F16/BF16/F64, converted to `f32`); anything else is a structured
//! error, never a silent reinterpretation.

use crate::core::model::formats::ModelFormat;
use serde_json::Value;
use std::collections::HashMap;
use std::format;
use std::string::{String, ToString};
use std::vec::Vec;

use super::ModelFormatParser;
use crate::core::tensor::WasmTensor;

/// One tensor recovered from a SafeTensors buffer.
#[derive(Debug, Clone)]
pub struct ParsedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Cheap, allocation-free structural probe: does `data` look like a
/// SafeTensors file? Used both for format auto-detection and as a guard
/// before the full parse.
///
/// A real SafeTensors file's first 8 bytes are a little-endian `u64` header
/// length `n` such that `0 < n <= data.len() - 8`, and `data[8]` is `{`
/// (the header is always a JSON object). This intentionally does *not*
/// match on `data.starts_with(b"{\"")`, which tests byte 0 rather than byte
/// 8 and can never be true for a real SafeTensors file.
pub fn looks_like_safetensors(data: &[u8]) -> bool {
    if data.len() < 9 {
        return false;
    }
    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&data[0..8]);
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len == 0 || header_len > (data.len() - 8) as u64 {
        return false;
    }
    data[8] == b'{'
}

fn dtype_element_size(dtype: &str) -> Result<usize, String> {
    match dtype {
        "F64" | "I64" | "U64" => Ok(8),
        "F32" | "I32" | "U32" => Ok(4),
        "F16" | "BF16" | "I16" | "U16" => Ok(2),
        "I8" | "U8" | "BOOL" => Ok(1),
        other => Err(format!("unrecognized safetensors dtype '{other}'")),
    }
}

fn decode_to_f32(dtype: &str, bytes: &[u8]) -> Result<Vec<f32>, String> {
    match dtype {
        "F32" => Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        "F64" => Ok(bytes
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect()),
        "F16" => Ok(bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()),
        "BF16" => Ok(bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()),
        other => Err(format!(
            "safetensors dtype '{other}' is not a float type this WASM build can run inference \
             on (supported: F32, F16, BF16, F64)"
        )),
    }
}

/// One tensor's real byte range from [`safetensors_component_layout`]:
/// `(tensor_name, absolute_start, absolute_end)`.
pub type SafetensorsEntry = (String, usize, usize);

/// Return type of [`safetensors_component_layout`]: `(header_end,
/// entries)` — see that function's doc comment.
pub type SafetensorsLayout = (usize, Vec<SafetensorsEntry>);

/// Real per-tensor byte layout of a SafeTensors buffer, for callers that
/// need genuine component boundaries (e.g. chunked-loading splitters)
/// without paying for — or requiring — a full weight decode.
///
/// Returns `(header_end, entries)`: `header_end` is the absolute offset
/// where the JSON header ends and tensor bytes begin (`8 + header_len`),
/// and `entries` is `(tensor_name, absolute_start, absolute_end)` for
/// every tensor, sorted by `absolute_start`. Unlike [`parse_safetensors`]
/// this does not decode any tensor bytes to `f32` and does not restrict
/// `dtype` to the float types this crate can run inference on — every
/// dtype SafeTensors supports declares an exact byte range in
/// `data_offsets`, which is all a byte-range layout needs. It still
/// validates that each declared `dtype` is at least *recognized* (a
/// symptom of header corruption if not) and that every offset is
/// in-bounds and non-decreasing, so a malformed header is a descriptive
/// `Err`, never a best-effort partial layout.
pub fn safetensors_component_layout(data: &[u8]) -> Result<SafetensorsLayout, String> {
    if !looks_like_safetensors(data) {
        return Err(
            "not a valid safetensors buffer: expected an 8-byte little-endian header length \
             followed by a '{' at byte 8"
                .to_string(),
        );
    }

    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&data[0..8]);
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    let header_start = 8usize;
    let header_end = header_start + header_len;
    let body_start = header_end;

    let header_json = std::str::from_utf8(&data[header_start..header_end])
        .map_err(|e| format!("safetensors header is not valid UTF-8: {e}"))?;
    let header: HashMap<String, Value> = serde_json::from_str(header_json)
        .map_err(|e| format!("failed to parse safetensors JSON header: {e}"))?;

    let body_len = data.len() - body_start;
    let mut entries = Vec::with_capacity(header.len());

    // Deterministic order, matching `parse_safetensors`.
    let mut header_entries: Vec<(&String, &Value)> = header.iter().collect();
    header_entries.sort_by(|a, b| a.0.cmp(b.0));

    for (name, info) in header_entries {
        if name == "__metadata__" {
            continue; // free-form string metadata, not a tensor
        }

        let obj = info
            .as_object()
            .ok_or_else(|| format!("safetensors entry '{name}' is not a JSON object"))?;

        let dtype = obj
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("safetensors entry '{name}' is missing a string 'dtype'"))?;
        // Recognized-dtype check only (no decode): catches a corrupt
        // header without requiring `dtype` to be one of the float types
        // this crate can run inference on.
        dtype_element_size(dtype)?;

        let offsets = obj
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("safetensors entry '{name}' is missing 'data_offsets'"))?;
        if offsets.len() != 2 {
            return Err(format!(
                "safetensors entry '{name}' has {} data_offsets, expected exactly 2",
                offsets.len()
            ));
        }
        let start = offsets[0].as_u64().ok_or_else(|| {
            format!("safetensors entry '{name}' has a non-integer data_offsets[0]")
        })? as usize;
        let end = offsets[1].as_u64().ok_or_else(|| {
            format!("safetensors entry '{name}' has a non-integer data_offsets[1]")
        })? as usize;

        if end < start {
            return Err(format!(
                "safetensors entry '{name}' has end offset {end} before start offset {start}"
            ));
        }
        if end > body_len {
            return Err(format!(
                "safetensors entry '{name}' data_offsets [{start}, {end}) exceed the buffer's \
                 body length ({body_len} bytes available after the header)"
            ));
        }

        entries.push((name.clone(), body_start + start, body_start + end));
    }

    entries.sort_by_key(|(_, start, _)| *start);
    Ok((body_start, entries))
}

/// Parse a full SafeTensors buffer into named, shaped `f32` tensors.
///
/// Every offset and shape is validated against the buffer bounds and
/// against the declared dtype's element size; a malformed or truncated
/// file produces a descriptive `Err`, never a best-effort partial result.
pub fn parse_safetensors(data: &[u8]) -> Result<Vec<ParsedTensor>, String> {
    if !looks_like_safetensors(data) {
        return Err(
            "not a valid safetensors buffer: expected an 8-byte little-endian header length \
             followed by a '{' at byte 8"
                .to_string(),
        );
    }

    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&data[0..8]);
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    let header_start = 8usize;
    let header_end = header_start + header_len;
    let body_start = header_end;

    let header_json = std::str::from_utf8(&data[header_start..header_end])
        .map_err(|e| format!("safetensors header is not valid UTF-8: {e}"))?;
    let header: HashMap<String, Value> = serde_json::from_str(header_json)
        .map_err(|e| format!("failed to parse safetensors JSON header: {e}"))?;

    let body_len = data.len() - body_start;
    let mut tensors = Vec::with_capacity(header.len());

    // Sort by name for deterministic iteration order (HashMap order is not
    // stable across runs, which would make error messages/order-dependent
    // tests flaky).
    let mut entries: Vec<(&String, &Value)> = header.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));

    for (name, info) in entries {
        if name == "__metadata__" {
            continue; // free-form string metadata, not a tensor
        }

        let obj = info
            .as_object()
            .ok_or_else(|| format!("safetensors entry '{name}' is not a JSON object"))?;

        let dtype = obj
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("safetensors entry '{name}' is missing a string 'dtype'"))?;

        let shape: Vec<usize> = obj
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("safetensors entry '{name}' is missing an array 'shape'"))?
            .iter()
            .map(|v| {
                v.as_u64().map(|n| n as usize).ok_or_else(|| {
                    format!("safetensors entry '{name}' has a non-integer shape dimension")
                })
            })
            .collect::<Result<Vec<usize>, String>>()?;

        let offsets = obj
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("safetensors entry '{name}' is missing 'data_offsets'"))?;
        if offsets.len() != 2 {
            return Err(format!(
                "safetensors entry '{name}' has {} data_offsets, expected exactly 2",
                offsets.len()
            ));
        }
        let start = offsets[0].as_u64().ok_or_else(|| {
            format!("safetensors entry '{name}' has a non-integer data_offsets[0]")
        })? as usize;
        let end = offsets[1].as_u64().ok_or_else(|| {
            format!("safetensors entry '{name}' has a non-integer data_offsets[1]")
        })? as usize;

        if end < start {
            return Err(format!(
                "safetensors entry '{name}' has end offset {end} before start offset {start}"
            ));
        }
        if end > body_len {
            return Err(format!(
                "safetensors entry '{name}' data_offsets [{start}, {end}) exceed the buffer's \
                 body length ({body_len} bytes available after the header)"
            ));
        }

        let element_size = dtype_element_size(dtype)?;
        let byte_len = end - start;
        let expected_elements: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
        let expected_bytes = expected_elements * element_size;
        if byte_len != expected_bytes {
            return Err(format!(
                "safetensors entry '{name}': shape {shape:?} implies {expected_elements} elements \
                 ({expected_bytes} bytes at {element_size} bytes/element) but data_offsets span \
                 {byte_len} bytes"
            ));
        }

        let abs_start = body_start + start;
        let abs_end = body_start + end;
        let raw = &data[abs_start..abs_end];
        let values = decode_to_f32(dtype, raw)?;

        tensors.push(ParsedTensor {
            name: name.clone(),
            shape: if shape.is_empty() { vec![1] } else { shape },
            data: values,
        });
    }

    Ok(tensors)
}

/// [`ModelFormatParser`] adapter around [`parse_safetensors`].
pub struct SafeTensorsParser;

impl ModelFormatParser for SafeTensorsParser {
    fn can_parse(&self, data: &[u8]) -> bool {
        looks_like_safetensors(data)
    }

    fn parse_metadata(&self, data: &[u8]) -> Result<HashMap<String, String>, String> {
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "SafeTensors".to_string());
        metadata.insert("size_bytes".to_string(), data.len().to_string());

        if data.len() >= 8 {
            let mut len_bytes = [0u8; 8];
            len_bytes.copy_from_slice(&data[0..8]);
            let header_len = u64::from_le_bytes(len_bytes) as usize;
            if 8 + header_len <= data.len() {
                if let Ok(header_json) = std::str::from_utf8(&data[8..8 + header_len]) {
                    if let Ok(header) = serde_json::from_str::<HashMap<String, Value>>(header_json)
                    {
                        let tensor_count =
                            header.keys().filter(|k| k.as_str() != "__metadata__").count();
                        metadata.insert("tensor_count".to_string(), tensor_count.to_string());
                        if let Some(Value::Object(meta)) = header.get("__metadata__") {
                            for (k, v) in meta {
                                if let Some(s) = v.as_str() {
                                    metadata.insert(format!("meta.{k}"), s.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(metadata)
    }

    fn load_weights(&self, data: &[u8]) -> Result<Vec<(String, WasmTensor)>, String> {
        let parsed = parse_safetensors(data)?;
        let mut out = Vec::with_capacity(parsed.len());
        for t in parsed {
            // `WasmTensor::new`'s own error type is `JsValue`, which cannot
            // be inspected/formatted on non-wasm32 targets (constructing
            // one panics there); every shape/finiteness precondition it
            // checks is already guaranteed by `parse_safetensors` (shape
            // matches decoded element count, dtype decoding cannot itself
            // produce NaN/Inf from finite input), so this should be
            // unreachable — a generic message is used rather than
            // inspecting the underlying `JsValue`.
            let tensor = WasmTensor::new(t.data, t.shape)
                .map_err(|_| format!("tensor '{}': failed to construct output tensor", t.name))?;
            out.push((t.name, tensor));
        }
        Ok(out)
    }

    fn get_format(&self) -> ModelFormat {
        ModelFormat::SafeTensors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal, valid single-tensor safetensors buffer by hand,
    /// independent of any writer implementation — this both exercises the
    /// parser and documents the exact byte layout it expects.
    fn build_safetensors(entries: &[(&str, &str, Vec<usize>, Vec<u8>)]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut body = Vec::new();
        let mut offset = 0usize;
        for (name, dtype, shape, bytes) in entries {
            let start = offset;
            let end = offset + bytes.len();
            offset = end;
            body.extend_from_slice(bytes);
            let mut obj = serde_json::Map::new();
            obj.insert("dtype".to_string(), Value::String((*dtype).to_string()));
            obj.insert(
                "shape".to_string(),
                Value::Array(shape.iter().map(|d| Value::from(*d)).collect()),
            );
            obj.insert(
                "data_offsets".to_string(),
                Value::Array(vec![Value::from(start), Value::from(end)]),
            );
            header.insert((*name).to_string(), Value::Object(obj));
        }
        let header_json = serde_json::to_vec(&Value::Object(header)).expect("valid json");
        let mut out = Vec::new();
        out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_json);
        out.extend_from_slice(&body);
        out
    }

    #[test]
    fn test_looks_like_safetensors_true_for_valid_header() {
        let data = build_safetensors(&[(
            "w",
            "F32",
            vec![2],
            4.0f32
                .to_le_bytes()
                .iter()
                .chain(5.0f32.to_le_bytes().iter())
                .copied()
                .collect(),
        )]);
        assert!(looks_like_safetensors(&data));
    }

    #[test]
    fn test_looks_like_safetensors_false_for_json_looking_start() {
        // The OLD (buggy) detection heuristic checked `data.starts_with(b"{\"")`
        // — verify we correctly reject such a buffer as not-safetensors,
        // since a real file never starts with '{' at byte 0.
        let data = b"{\"not\": \"safetensors\"}".to_vec();
        assert!(!looks_like_safetensors(&data));
    }

    #[test]
    fn test_looks_like_safetensors_false_for_too_short() {
        assert!(!looks_like_safetensors(&[0u8; 4]));
    }

    #[test]
    fn test_looks_like_safetensors_false_for_huge_header_len() {
        let mut data = vec![0u8; 16];
        data[0..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(!looks_like_safetensors(&data));
    }

    #[test]
    fn test_parse_single_f32_tensor_roundtrip() {
        let values = [1.0f32, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let data = build_safetensors(&[("layers.0.attn.q_proj.weight", "F32", vec![2, 2], bytes)]);

        let parsed = parse_safetensors(&data).expect("valid safetensors buffer");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "layers.0.attn.q_proj.weight");
        assert_eq!(parsed[0].shape, vec![2, 2]);
        assert_eq!(parsed[0].data, values);
    }

    #[test]
    fn test_parse_multiple_tensors_and_metadata_skip() {
        let a: Vec<u8> = 1.0f32.to_le_bytes().to_vec();
        let b: Vec<u8> = [2.0f32, 3.0f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut data = build_safetensors(&[("a", "F32", vec![1], a), ("b", "F32", vec![2], b)]);

        // Inject a __metadata__ key by rebuilding the header manually since
        // build_safetensors() doesn't support it directly.
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&data[0..8]);
        let header_len = u64::from_le_bytes(len_bytes) as usize;
        let mut header: serde_json::Map<String, Value> =
            serde_json::from_slice(&data[8..8 + header_len]).unwrap();
        let mut meta = serde_json::Map::new();
        meta.insert("format".to_string(), Value::String("pt".to_string()));
        header.insert("__metadata__".to_string(), Value::Object(meta));
        let new_header_json = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut rebuilt = Vec::new();
        rebuilt.extend_from_slice(&(new_header_json.len() as u64).to_le_bytes());
        rebuilt.extend_from_slice(&new_header_json);
        rebuilt.extend_from_slice(&data[8 + header_len..]);
        data = rebuilt;

        let parsed = parse_safetensors(&data).expect("valid buffer with metadata key");
        assert_eq!(
            parsed.len(),
            2,
            "the __metadata__ entry must not be treated as a tensor"
        );
        let names: Vec<&str> = parsed.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"a"));
        assert!(names.contains(&"b"));
    }

    #[test]
    fn test_parse_f16_and_bf16_tensors() {
        let f16_val = half::f16::from_f32(1.5);
        let bf16_val = half::bf16::from_f32(-2.5);
        let f16_bytes = f16_val.to_le_bytes().to_vec();
        let bf16_bytes = bf16_val.to_le_bytes().to_vec();
        let data = build_safetensors(&[
            ("f16.weight", "F16", vec![1], f16_bytes),
            ("bf16.weight", "BF16", vec![1], bf16_bytes),
        ]);
        let parsed = parse_safetensors(&data).expect("valid buffer");
        let f16_tensor = parsed.iter().find(|t| t.name == "f16.weight").unwrap();
        let bf16_tensor = parsed.iter().find(|t| t.name == "bf16.weight").unwrap();
        assert!((f16_tensor.data[0] - 1.5).abs() < 1e-3);
        assert!((bf16_tensor.data[0] - (-2.5)).abs() < 1e-2);
    }

    #[test]
    fn test_parse_rejects_truncated_data_offsets() {
        let bytes = vec![0u8; 2]; // claim shape [2] F32 (needs 8 bytes) but only give 2
        let data = build_safetensors(&[("w", "F32", vec![2], bytes)]);
        let err = parse_safetensors(&data).expect_err("mismatched byte length must error");
        assert!(err.contains("w"));
    }

    #[test]
    fn test_parse_rejects_unsupported_dtype() {
        let data = build_safetensors(&[("w", "I64", vec![1], vec![0u8; 8])]);
        let err =
            parse_safetensors(&data).expect_err("integer dtype must error, not be reinterpreted");
        assert!(err.contains("I64") || err.contains("dtype"));
    }

    #[test]
    fn test_parse_rejects_offsets_past_buffer_end() {
        // Hand-craft a header claiming a huge end offset with no matching body.
        let header_json = br#"{"w":{"dtype":"F32","shape":[1000000],"data_offsets":[0,4000000]}}"#;
        let mut data = Vec::new();
        data.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
        data.extend_from_slice(header_json);
        // no body at all
        let err = parse_safetensors(&data).expect_err("out-of-bounds offsets must error");
        assert!(err.contains("exceed"));
    }

    #[test]
    fn test_parse_not_safetensors_data_errors_cleanly() {
        let err = parse_safetensors(b"just some random bytes, not safetensors at all!!")
            .expect_err("garbage input must not parse");
        assert!(err.contains("safetensors"));
    }

    #[test]
    fn test_parser_trait_load_weights_roundtrip() {
        let values = [1.0f32, -1.0, 0.5, 0.25];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let data = build_safetensors(&[("token_embeddings.weight", "F32", vec![2, 2], bytes)]);

        let parser = SafeTensorsParser;
        assert!(parser.can_parse(&data));
        let weights = parser.load_weights(&data).expect("valid safetensors");
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0].0, "token_embeddings.weight");
        assert_eq!(weights[0].1.data(), values);
    }

    // -----------------------------------------------------------------
    // `safetensors_component_layout`: real byte-range boundaries for
    // chunked-loading splitters, independent of whether the tensor's
    // dtype can be decoded to f32.
    // -----------------------------------------------------------------

    #[test]
    fn test_safetensors_component_layout_matches_declared_offsets() {
        let a: Vec<u8> = 1.0f32.to_le_bytes().to_vec();
        let b: Vec<u8> = [2.0f32, 3.0f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let data = build_safetensors(&[
            ("layer.a.weight", "F32", vec![1], a.clone()),
            ("layer.b.weight", "F32", vec![2], b.clone()),
        ]);

        let (header_end, entries) =
            safetensors_component_layout(&data).expect("valid safetensors buffer");

        // header_end must be exactly where the tensor body starts: the
        // buffer's own 8-byte length prefix plus the header it declares.
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&data[0..8]);
        let declared_header_len = u64::from_le_bytes(len_bytes) as usize;
        assert_eq!(header_end, 8 + declared_header_len);

        assert_eq!(entries.len(), 2);
        let (name_a, start_a, end_a) = &entries[0];
        assert_eq!(name_a, "layer.a.weight");
        assert_eq!(end_a - start_a, a.len());
        assert_eq!(*start_a, header_end); // first tensor starts right after the header
        let (name_b, start_b, end_b) = &entries[1];
        assert_eq!(name_b, "layer.b.weight");
        assert_eq!(end_b - start_b, b.len());
        assert_eq!(*start_b, *end_a); // contiguous, no gap

        // The returned ranges must reproduce the exact original bytes.
        assert_eq!(&data[*start_a..*end_a], &a[..]);
        assert_eq!(&data[*start_b..*end_b], &b[..]);
    }

    #[test]
    fn test_safetensors_component_layout_accepts_integer_dtype_that_parse_safetensors_rejects() {
        // `parse_safetensors` must decode to f32 and so rejects non-float
        // dtypes (see `test_parse_rejects_unsupported_dtype`); a pure byte
        // *layout* has no such requirement — every dtype SafeTensors
        // supports still declares an exact, real `data_offsets` range.
        let data = build_safetensors(&[("position_ids", "I64", vec![4], vec![0u8; 32])]);

        assert!(parse_safetensors(&data).is_err());

        let (_, entries) =
            safetensors_component_layout(&data).expect("layout must not require a float dtype");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, "position_ids");
        assert_eq!(entries[0].2 - entries[0].1, 32);
    }

    #[test]
    fn test_safetensors_component_layout_sorted_by_byte_position_not_name() {
        // Insert "zeta" first (occupies the earlier byte range) and
        // "alpha" second (the later range) — alphabetically "alpha" <
        // "zeta", but the returned order must follow physical layout.
        let data = build_safetensors(&[
            ("zeta", "F32", vec![1], 1.0f32.to_le_bytes().to_vec()),
            ("alpha", "F32", vec![1], 2.0f32.to_le_bytes().to_vec()),
        ]);

        let (_, entries) = safetensors_component_layout(&data).expect("valid buffer");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, "zeta", "earlier byte offset must sort first");
        assert_eq!(entries[1].0, "alpha");
        assert!(entries[0].1 < entries[1].1);
    }

    #[test]
    fn test_safetensors_component_layout_skips_metadata_key() {
        let data_with_meta = {
            let mut data =
                build_safetensors(&[("w", "F32", vec![1], 1.0f32.to_le_bytes().to_vec())]);
            let mut len_bytes = [0u8; 8];
            len_bytes.copy_from_slice(&data[0..8]);
            let header_len = u64::from_le_bytes(len_bytes) as usize;
            let mut header: serde_json::Map<String, Value> =
                serde_json::from_slice(&data[8..8 + header_len]).unwrap();
            let mut meta = serde_json::Map::new();
            meta.insert("format".to_string(), Value::String("pt".to_string()));
            header.insert("__metadata__".to_string(), Value::Object(meta));
            let new_header_json = serde_json::to_vec(&Value::Object(header)).unwrap();
            let mut rebuilt = Vec::new();
            rebuilt.extend_from_slice(&(new_header_json.len() as u64).to_le_bytes());
            rebuilt.extend_from_slice(&new_header_json);
            rebuilt.extend_from_slice(&data[8 + header_len..]);
            std::mem::swap(&mut data, &mut rebuilt);
            data
        };

        let (_, entries) =
            safetensors_component_layout(&data_with_meta).expect("valid buffer with metadata");
        assert_eq!(
            entries.len(),
            1,
            "the __metadata__ entry must not be treated as a tensor component"
        );
        assert_eq!(entries[0].0, "w");
    }

    #[test]
    fn test_safetensors_component_layout_rejects_non_safetensors_data() {
        let err = safetensors_component_layout(b"just some random bytes, not safetensors at all!!")
            .expect_err("garbage input must not produce a fabricated layout");
        assert!(err.contains("safetensors"));
    }

    #[test]
    fn test_safetensors_component_layout_rejects_offsets_past_buffer_end() {
        let header_json = br#"{"w":{"dtype":"F32","shape":[1000000],"data_offsets":[0,4000000]}}"#;
        let mut data = Vec::new();
        data.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
        data.extend_from_slice(header_json);
        let err = safetensors_component_layout(&data)
            .expect_err("out-of-bounds offsets must error, not a truncated fabricated range");
        assert!(err.contains("exceed"));
    }
}
