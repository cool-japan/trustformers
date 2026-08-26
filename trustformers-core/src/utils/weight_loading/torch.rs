//! A real reader for PyTorch checkpoints (`.bin`, `.pt`, `.pth`).
//!
//! Since PyTorch 1.6 `torch.save` writes a ZIP archive laid out like
//!
//! ```text
//! <archive>/data.pkl      the pickled state dict
//! <archive>/data/0        raw bytes of storage 0
//! <archive>/data/1        raw bytes of storage 1
//! <archive>/version       the serialization version
//! ```
//!
//! where `<archive>` is a prefix chosen when the file was written, so members are
//! located by suffix.
//!
//! `data.pkl` builds each tensor by "calling"
//! `torch._utils._rebuild_tensor_v2(storage, storage_offset, size, stride,
//! requires_grad, backward_hooks)`, where `storage` is a pickle *persistent id*
//! `('storage', torch.FloatStorage, key, location, numel)` pointing at
//! `<archive>/data/<key>`. This module reads that structure with
//! [`super::pickle`], which records those calls symbolically instead of executing
//! them, and reassembles `name -> tensor` with the real dtype, shape and bytes.
//!
//! # Documented limitations
//!
//! * Only row-major contiguous tensors are reconstructed. A view with a
//!   non-contiguous stride is reported as an error rather than silently read as if
//!   it were contiguous. State dicts saved by `torch.save` are contiguous.
//! * Deflate-compressed archive members are refused (see [`super::zip_archive`]).
//! * The legacy pre-1.6 non-ZIP pickle format is refused with a clear message.

use super::pickle::{self, PickleValue};
use super::zip_archive::{looks_like_zip, ZipArchive};
use crate::tensor::Tensor;
use anyhow::{anyhow, Result};
use scirs2_core::ndarray::{ArrayD, IxDyn};
use std::collections::BTreeMap;

/// Element type of a PyTorch storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TorchDType {
    F32,
    F64,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl TorchDType {
    /// Map a `torch.*Storage` class name onto an element type.
    pub fn from_storage_name(name: &str) -> Result<Self> {
        Ok(match name {
            "FloatStorage" => TorchDType::F32,
            "DoubleStorage" => TorchDType::F64,
            "HalfStorage" => TorchDType::F16,
            "BFloat16Storage" => TorchDType::BF16,
            "LongStorage" => TorchDType::I64,
            "IntStorage" => TorchDType::I32,
            "ShortStorage" => TorchDType::I16,
            "CharStorage" => TorchDType::I8,
            "ByteStorage" => TorchDType::U8,
            "BoolStorage" => TorchDType::Bool,
            other => {
                return Err(anyhow!(
                    "torch storage type '{other}' is not one this reader decodes (Float, Double, \
                     Half, BFloat16, Long, Int, Short, Char, Byte, Bool)"
                ))
            },
        })
    }

    /// Bytes per element.
    pub fn size_in_bytes(&self) -> usize {
        match self {
            TorchDType::F64 | TorchDType::I64 => 8,
            TorchDType::F32 | TorchDType::I32 => 4,
            TorchDType::F16 | TorchDType::BF16 | TorchDType::I16 => 2,
            TorchDType::I8 | TorchDType::U8 | TorchDType::Bool => 1,
        }
    }

    /// The name used in reports and error messages.
    pub fn name(&self) -> &'static str {
        match self {
            TorchDType::F32 => "f32",
            TorchDType::F64 => "f64",
            TorchDType::F16 => "f16",
            TorchDType::BF16 => "bf16",
            TorchDType::I64 => "i64",
            TorchDType::I32 => "i32",
            TorchDType::I16 => "i16",
            TorchDType::I8 => "i8",
            TorchDType::U8 => "u8",
            TorchDType::Bool => "bool",
        }
    }

    /// Whether values of this type are floating point.
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            TorchDType::F32 | TorchDType::F64 | TorchDType::F16 | TorchDType::BF16
        )
    }
}

/// One reconstructed tensor.
#[derive(Debug, Clone)]
pub struct TorchTensor {
    /// Element type as stored in the checkpoint.
    pub dtype: TorchDType,
    /// Row-major shape.
    pub shape: Vec<usize>,
    /// The decoded tensor: `F32` for floating types, `I64` for integer and bool.
    pub tensor: Tensor,
}

/// A checkpoint's tensors, keyed by their state-dict names.
#[derive(Debug, Clone, Default)]
pub struct TorchStateDict {
    tensors: BTreeMap<String, TorchTensor>,
}

impl TorchStateDict {
    /// Names in sorted order.
    pub fn names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    /// Look up a tensor by name.
    pub fn get(&self, name: &str) -> Option<&TorchTensor> {
        self.tensors.get(name)
    }

    /// Number of tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the checkpoint held no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// Parse a PyTorch checkpoint held in memory.
pub fn read_torch_bytes(bytes: &[u8]) -> Result<TorchStateDict> {
    if !looks_like_zip(bytes) {
        if bytes.starts_with(&[0x80]) {
            return Err(anyhow!(
                "this looks like a legacy (pre-PyTorch-1.6) checkpoint: a bare pickle stream \
                 rather than a ZIP archive. That format interleaves storages with the pickle and \
                 is not read by this crate. Re-save it with a modern PyTorch \
                 (`torch.save(obj, path)` defaults to the ZIP format) or convert it to \
                 safetensors, which this crate reads natively."
            ));
        }
        return Err(anyhow!(
            "not a PyTorch checkpoint: the file is neither a ZIP archive nor a pickle stream"
        ));
    }

    let archive = ZipArchive::parse(bytes)?;
    if !archive.has_suffix("data.pkl") {
        return Err(anyhow!(
            "the archive has no 'data.pkl' member, so it is not a PyTorch checkpoint"
        ));
    }

    let pickle_entry = archive
        .entries()
        .iter()
        .find(|entry| entry.name.ends_with("data.pkl"))
        .ok_or_else(|| anyhow!("the archive has no 'data.pkl' member"))?;
    let storage_prefix = pickle_entry.name.strip_suffix("data.pkl").unwrap_or_default().to_string();

    let root = pickle::load(archive.read(&pickle_entry.name)?)?;

    let mut tensors = BTreeMap::new();
    collect_tensors(&root, "", &archive, &storage_prefix, &mut tensors)?;

    if tensors.is_empty() {
        return Err(anyhow!(
            "the checkpoint parsed cleanly but contains no tensors built by \
             torch._utils._rebuild_tensor_v2/_v3; there is nothing to load"
        ));
    }

    Ok(TorchStateDict { tensors })
}

/// Parse a PyTorch checkpoint from disk.
pub fn read_torch_file(path: &std::path::Path) -> Result<TorchStateDict> {
    let bytes = std::fs::read(path)
        .map_err(|e| anyhow!("failed to read PyTorch file {}: {e}", path.display()))?;
    read_torch_bytes(&bytes).map_err(|e| anyhow!("{}: {e}", path.display()))
}

/// Keys whose sub-dictionary is the state dict itself rather than a namespace.
const STATE_DICT_WRAPPER_KEYS: &[&str] = &["state_dict", "model_state_dict", "model", "module"];

fn collect_tensors(
    value: &PickleValue,
    prefix: &str,
    archive: &ZipArchive<'_>,
    storage_prefix: &str,
    out: &mut BTreeMap<String, TorchTensor>,
) -> Result<()> {
    match value {
        PickleValue::Dict(entries) => {
            for (key, item) in entries {
                let Some(key) = key.as_str() else {
                    // Non-string keys never name a parameter.
                    continue;
                };

                // `torch.save({"state_dict": ...})` is a common wrapper; its keys
                // are the parameter names, not a namespace to prefix.
                let nested_prefix = if prefix.is_empty() && STATE_DICT_WRAPPER_KEYS.contains(&key) {
                    String::new()
                } else if prefix.is_empty() {
                    key.to_string()
                } else {
                    format!("{prefix}.{key}")
                };

                if let Some(tensor) = try_build_tensor(item, archive, storage_prefix)? {
                    out.insert(nested_prefix, tensor);
                } else if matches!(item, PickleValue::Dict(_)) {
                    collect_tensors(item, &nested_prefix, archive, storage_prefix, out)?;
                }
            }
            Ok(())
        },
        other => {
            if let Some(tensor) = try_build_tensor(other, archive, storage_prefix)? {
                out.insert(prefix.to_string(), tensor);
            }
            Ok(())
        },
    }
}

/// Recognise a tensor-building reduce and reconstruct it.
///
/// Returns `Ok(None)` for values that are not tensors, so that ordinary metadata
/// in a checkpoint is skipped rather than treated as an error.
fn try_build_tensor(
    value: &PickleValue,
    archive: &ZipArchive<'_>,
    storage_prefix: &str,
) -> Result<Option<TorchTensor>> {
    let (callable, args) = match value {
        PickleValue::Reduce { callable, args } => (callable.as_ref(), args.as_ref()),
        PickleValue::Object { callable, args, .. } => (callable.as_ref(), args.as_ref()),
        _ => return Ok(None),
    };

    let PickleValue::Global { module, name } = callable else {
        return Ok(None);
    };
    let args = args.as_sequence().unwrap_or(&[]);

    match (module.as_str(), name.as_str()) {
        ("torch._utils", "_rebuild_tensor_v2" | "_rebuild_tensor_v3" | "_rebuild_tensor") => {
            rebuild_tensor(args, archive, storage_prefix).map(Some)
        },
        ("torch._utils", "_rebuild_parameter" | "_rebuild_parameter_with_state") => {
            // The first argument is the tensor the parameter wraps.
            match args.first() {
                Some(inner) => try_build_tensor(inner, archive, storage_prefix),
                None => Err(anyhow!("_rebuild_parameter was called with no arguments")),
            }
        },
        _ => Ok(None),
    }
}

/// `_rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, hooks)`.
fn rebuild_tensor(
    args: &[PickleValue],
    archive: &ZipArchive<'_>,
    storage_prefix: &str,
) -> Result<TorchTensor> {
    if args.len() < 4 {
        return Err(anyhow!(
            "_rebuild_tensor_v2 expects at least 4 arguments, got {}",
            args.len()
        ));
    }

    let (dtype, storage_key, storage_elements) = parse_storage_persistent_id(&args[0])?;
    let storage_offset = args[1]
        .as_int()
        .ok_or_else(|| anyhow!("_rebuild_tensor_v2 storage_offset is not an integer"))?;
    if storage_offset < 0 {
        return Err(anyhow!("_rebuild_tensor_v2 storage_offset is negative"));
    }
    let storage_offset = storage_offset as usize;

    let shape = parse_dimension_list(&args[2], "size")?;
    let stride = parse_dimension_list(&args[3], "stride")?;

    let element_count: usize = shape.iter().product();
    let expected_stride = contiguous_strides(&shape);
    // A length-1 axis can carry any stride without changing the layout.
    let contiguous = stride.len() == expected_stride.len()
        && stride
            .iter()
            .zip(expected_stride.iter())
            .zip(shape.iter())
            .all(|((actual, expected), &dim)| dim <= 1 || actual == expected);
    if !contiguous {
        return Err(anyhow!(
            "tensor with shape {shape:?} has stride {stride:?}, which is not row-major \
             contiguous; this reader does not materialise strided views"
        ));
    }

    let record_name = format!("{storage_prefix}data/{storage_key}");
    let raw = archive.read(&record_name).map_err(|e| {
        anyhow!("storage record '{record_name}' referenced by the state dict is missing: {e}")
    })?;

    let element_size = dtype.size_in_bytes();
    if raw.len() % element_size != 0 {
        return Err(anyhow!(
            "storage record '{record_name}' holds {} bytes, not a multiple of the {} element \
             size for {}",
            raw.len(),
            element_size,
            dtype.name()
        ));
    }
    let available = raw.len() / element_size;
    if storage_elements != 0 && available < storage_elements {
        return Err(anyhow!(
            "storage record '{record_name}' holds {available} elements but the state dict \
             declares {storage_elements}"
        ));
    }
    if storage_offset + element_count > available {
        return Err(anyhow!(
            "tensor of {element_count} elements at offset {storage_offset} does not fit in \
             storage '{record_name}' of {available} elements"
        ));
    }

    let start = storage_offset * element_size;
    let end = start + element_count * element_size;
    let payload = &raw[start..end];

    let tensor = decode_payload(payload, dtype, &shape)?;

    Ok(TorchTensor {
        dtype,
        shape,
        tensor,
    })
}

/// `('storage', torch.FloatStorage, key, location, numel)`.
fn parse_storage_persistent_id(value: &PickleValue) -> Result<(TorchDType, String, usize)> {
    let PickleValue::PersistentId(inner) = value else {
        return Err(anyhow!(
            "_rebuild_tensor_v2 storage argument is not a pickle persistent id"
        ));
    };
    let fields = inner
        .as_sequence()
        .ok_or_else(|| anyhow!("storage persistent id is not a tuple"))?;
    if fields.len() < 5 {
        return Err(anyhow!(
            "storage persistent id has {} fields, expected 5",
            fields.len()
        ));
    }
    if fields[0].as_str() != Some("storage") {
        return Err(anyhow!(
            "persistent id tag is {:?}, expected \"storage\"",
            fields[0]
        ));
    }

    let dtype = match &fields[1] {
        PickleValue::Global { module, name } => {
            if module != "torch" {
                return Err(anyhow!(
                    "storage type comes from module '{module}', expected 'torch'"
                ));
            }
            TorchDType::from_storage_name(name)?
        },
        other => {
            return Err(anyhow!(
                "storage type is {other:?}, expected a torch global"
            ))
        },
    };

    let key = match &fields[2] {
        PickleValue::String(key) => key.clone(),
        PickleValue::Int(key) => key.to_string(),
        other => return Err(anyhow!("storage key is {other:?}, expected a string")),
    };

    let numel = fields[4].as_int().unwrap_or(0).max(0) as usize;

    Ok((dtype, key, numel))
}

fn parse_dimension_list(value: &PickleValue, what: &str) -> Result<Vec<usize>> {
    let items = value
        .as_sequence()
        .ok_or_else(|| anyhow!("_rebuild_tensor_v2 {what} is not a tuple"))?;
    items
        .iter()
        .map(|item| {
            let raw = item
                .as_int()
                .ok_or_else(|| anyhow!("_rebuild_tensor_v2 {what} contains a non-integer"))?;
            if raw < 0 {
                return Err(anyhow!("_rebuild_tensor_v2 {what} contains {raw}"));
            }
            Ok(raw as usize)
        })
        .collect()
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for index in (0..shape.len().saturating_sub(1)).rev() {
        strides[index] = strides[index + 1] * shape[index + 1];
    }
    strides
}

fn decode_payload(payload: &[u8], dtype: TorchDType, shape: &[usize]) -> Result<Tensor> {
    let shape_dyn = IxDyn(shape);

    let tensor = match dtype {
        TorchDType::F32 => {
            let values: Vec<f32> = payload
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Tensor::F32(array_from(shape_dyn, values)?)
        },
        TorchDType::F64 => {
            let values: Vec<f32> = payload
                .chunks_exact(8)
                .map(|c| {
                    f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect();
            Tensor::F32(array_from(shape_dyn, values)?)
        },
        TorchDType::F16 => {
            let values: Vec<f32> = payload
                .chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect();
            Tensor::F32(array_from(shape_dyn, values)?)
        },
        TorchDType::BF16 => {
            let values: Vec<f32> = payload
                .chunks_exact(2)
                .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect();
            Tensor::F32(array_from(shape_dyn, values)?)
        },
        TorchDType::I64 => {
            let values: Vec<i64> = payload
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            Tensor::I64(array_from(shape_dyn, values)?)
        },
        TorchDType::I32 => {
            let values: Vec<i64> = payload
                .chunks_exact(4)
                .map(|c| i64::from(i32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                .collect();
            Tensor::I64(array_from(shape_dyn, values)?)
        },
        TorchDType::I16 => {
            let values: Vec<i64> = payload
                .chunks_exact(2)
                .map(|c| i64::from(i16::from_le_bytes([c[0], c[1]])))
                .collect();
            Tensor::I64(array_from(shape_dyn, values)?)
        },
        TorchDType::I8 => {
            let values: Vec<i64> = payload.iter().map(|&b| i64::from(b as i8)).collect();
            Tensor::I64(array_from(shape_dyn, values)?)
        },
        TorchDType::U8 | TorchDType::Bool => {
            let values: Vec<i64> = payload.iter().map(|&b| i64::from(b)).collect();
            Tensor::I64(array_from(shape_dyn, values)?)
        },
    };

    Ok(tensor)
}

fn array_from<T>(shape: IxDyn, values: Vec<T>) -> Result<ArrayD<T>> {
    ArrayD::from_shape_vec(shape, values)
        .map_err(|e| anyhow!("tensor data does not match its declared shape: {e}"))
}

#[cfg(test)]
pub(crate) mod fixture {
    //! Builders for byte-exact `torch.save`-shaped fixtures.
    //!
    //! These write the pickle stream opcode by opcode, so a test exercises the real
    //! parser against the real layout rather than against the parser's own
    //! assumptions.

    use super::super::zip_archive::StoredZipBuilder;

    /// One tensor to place in a fixture checkpoint.
    pub struct FixtureTensor {
        pub name: String,
        pub storage_class: &'static str,
        pub storage_key: String,
        pub bytes: Vec<u8>,
        pub element_count: usize,
        pub storage_offset: usize,
        pub shape: Vec<usize>,
        pub stride: Vec<usize>,
    }

    fn binunicode(out: &mut Vec<u8>, text: &str) {
        out.push(b'X');
        out.extend_from_slice(&(text.len() as u32).to_le_bytes());
        out.extend_from_slice(text.as_bytes());
    }

    fn global(out: &mut Vec<u8>, module: &str, name: &str) {
        out.push(b'c');
        out.extend_from_slice(module.as_bytes());
        out.push(b'\n');
        out.extend_from_slice(name.as_bytes());
        out.push(b'\n');
    }

    fn integer(out: &mut Vec<u8>, value: i64) {
        if (0..=255).contains(&value) {
            out.push(b'K');
            out.push(value as u8);
        } else {
            out.push(b'J');
            out.extend_from_slice(&(value as i32).to_le_bytes());
        }
    }

    fn tuple_of_integers(out: &mut Vec<u8>, values: &[usize]) {
        out.push(b'(');
        for &value in values {
            integer(out, value as i64);
        }
        out.push(b't');
    }

    /// Build the `data.pkl` stream for a state dict of tensors.
    pub fn build_data_pkl(tensors: &[FixtureTensor]) -> Vec<u8> {
        let mut out = vec![0x80, 2];
        out.push(b'}'); // EMPTY_DICT
        out.push(b'('); // MARK

        for tensor in tensors {
            binunicode(&mut out, &tensor.name);

            global(&mut out, "torch._utils", "_rebuild_tensor_v2");
            out.push(b'('); // MARK for the argument tuple

            // ('storage', torch.<Class>, key, 'cpu', numel)
            out.push(b'(');
            binunicode(&mut out, "storage");
            global(&mut out, "torch", tensor.storage_class);
            binunicode(&mut out, &tensor.storage_key);
            binunicode(&mut out, "cpu");
            integer(&mut out, tensor.element_count as i64);
            out.push(b't');
            out.push(b'Q'); // BINPERSID

            integer(&mut out, tensor.storage_offset as i64);
            tuple_of_integers(&mut out, &tensor.shape);
            tuple_of_integers(&mut out, &tensor.stride);
            out.push(0x89); // NEWFALSE (requires_grad)
            global(&mut out, "collections", "OrderedDict");
            out.push(b')');
            out.push(b'R'); // backward hooks: an empty OrderedDict

            out.push(b't'); // TUPLE of the six arguments
            out.push(b'R'); // REDUCE
        }

        out.push(b'u'); // SETITEMS
        out.push(b'.'); // STOP
        out
    }

    /// Build a complete `torch.save`-shaped archive.
    pub fn build_checkpoint(archive_name: &str, tensors: &[FixtureTensor]) -> Vec<u8> {
        let mut builder = StoredZipBuilder::new()
            .add(&format!("{archive_name}/data.pkl"), build_data_pkl(tensors))
            .add(&format!("{archive_name}/version"), b"3\n".to_vec());

        let mut written: Vec<&str> = Vec::new();
        for tensor in tensors {
            if written.contains(&tensor.storage_key.as_str()) {
                continue;
            }
            written.push(&tensor.storage_key);
            builder = builder.add(
                &format!("{archive_name}/data/{}", tensor.storage_key),
                tensor.bytes.clone(),
            );
        }

        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use super::fixture::{build_checkpoint, FixtureTensor};
    use super::*;

    fn float_tensor(name: &str, key: &str, values: &[f32], shape: Vec<usize>) -> FixtureTensor {
        FixtureTensor {
            name: name.to_string(),
            storage_class: "FloatStorage",
            storage_key: key.to_string(),
            bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            element_count: values.len(),
            storage_offset: 0,
            stride: contiguous_strides(&shape),
            shape,
        }
    }

    #[test]
    fn reads_a_checkpoint_built_byte_by_byte() {
        let weights: Vec<f32> = (0..6).map(|i| i as f32 * 1.5).collect();
        let bias = vec![-1.0f32, 0.5];
        let bytes = build_checkpoint(
            "archive",
            &[
                float_tensor("linear.weight", "0", &weights, vec![2, 3]),
                float_tensor("linear.bias", "1", &bias, vec![2]),
            ],
        );

        let state = read_torch_bytes(&bytes).expect("read checkpoint");
        assert_eq!(state.len(), 2);
        assert_eq!(
            state.names(),
            vec!["linear.bias".to_string(), "linear.weight".to_string()]
        );

        let weight = state.get("linear.weight").expect("weight");
        assert_eq!(weight.dtype, TorchDType::F32);
        assert_eq!(weight.shape, vec![2, 3]);
        assert_eq!(weight.tensor.shape(), vec![2, 3]);
        assert_eq!(weight.tensor.to_vec_f32().expect("f32"), weights);

        let bias_tensor = state.get("linear.bias").expect("bias");
        assert_eq!(bias_tensor.tensor.to_vec_f32().expect("f32"), bias);
    }

    /// Regression test: the old reader returned all-zero tensors with guessed
    /// BERT shapes regardless of what the file actually held.
    #[test]
    fn values_come_from_the_file_not_from_a_template() {
        let make = |seed: f32| {
            let values: Vec<f32> = (0..4).map(|i| seed + i as f32).collect();
            build_checkpoint("archive", &[float_tensor("w", "0", &values, vec![4])])
        };

        let first = read_torch_bytes(&make(0.0)).expect("read");
        let second = read_torch_bytes(&make(10.0)).expect("read");

        assert_eq!(
            first.get("w").expect("w").tensor.to_vec_f32().expect("f32"),
            vec![0.0, 1.0, 2.0, 3.0]
        );
        assert_eq!(
            second.get("w").expect("w").tensor.to_vec_f32().expect("f32"),
            vec![10.0, 11.0, 12.0, 13.0]
        );
        assert_ne!(
            first.get("w").expect("w").tensor.to_vec_f32().expect("f32"),
            second.get("w").expect("w").tensor.to_vec_f32().expect("f32")
        );
    }

    #[test]
    fn shared_storage_with_an_offset_is_sliced_correctly() {
        let storage: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let raw: Vec<u8> = storage.iter().flat_map(|v| v.to_le_bytes()).collect();

        let bytes = build_checkpoint(
            "archive",
            &[
                FixtureTensor {
                    name: "head".to_string(),
                    storage_class: "FloatStorage",
                    storage_key: "0".to_string(),
                    bytes: raw.clone(),
                    element_count: 8,
                    storage_offset: 0,
                    shape: vec![4],
                    stride: vec![1],
                },
                FixtureTensor {
                    name: "tail".to_string(),
                    storage_class: "FloatStorage",
                    storage_key: "0".to_string(),
                    bytes: raw,
                    element_count: 8,
                    storage_offset: 4,
                    shape: vec![4],
                    stride: vec![1],
                },
            ],
        );

        let state = read_torch_bytes(&bytes).expect("read");
        assert_eq!(
            state.get("head").expect("head").tensor.to_vec_f32().expect("f32"),
            vec![0.0, 1.0, 2.0, 3.0]
        );
        assert_eq!(
            state.get("tail").expect("tail").tensor.to_vec_f32().expect("f32"),
            vec![4.0, 5.0, 6.0, 7.0]
        );
    }

    #[test]
    fn half_and_integer_storages_decode_with_the_right_dtype() {
        let half_bytes: Vec<u8> = [1.0f32, -2.5, 0.25, 4.0]
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        let long_bytes: Vec<u8> = [7i64, -3, 0].iter().flat_map(|v| v.to_le_bytes()).collect();

        let bytes = build_checkpoint(
            "archive",
            &[
                FixtureTensor {
                    name: "half".to_string(),
                    storage_class: "HalfStorage",
                    storage_key: "0".to_string(),
                    bytes: half_bytes,
                    element_count: 4,
                    storage_offset: 0,
                    shape: vec![2, 2],
                    stride: vec![2, 1],
                },
                FixtureTensor {
                    name: "ids".to_string(),
                    storage_class: "LongStorage",
                    storage_key: "1".to_string(),
                    bytes: long_bytes,
                    element_count: 3,
                    storage_offset: 0,
                    shape: vec![3],
                    stride: vec![1],
                },
            ],
        );

        let state = read_torch_bytes(&bytes).expect("read");
        let half = state.get("half").expect("half");
        assert_eq!(half.dtype, TorchDType::F16);
        assert_eq!(
            half.tensor.to_vec_f32().expect("f32"),
            vec![1.0, -2.5, 0.25, 4.0]
        );

        let ids = state.get("ids").expect("ids");
        assert_eq!(ids.dtype, TorchDType::I64);
        assert!(matches!(ids.tensor, Tensor::I64(_)));
    }

    #[test]
    fn a_non_contiguous_stride_is_reported_not_ignored() {
        let values: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let bytes = build_checkpoint(
            "archive",
            &[FixtureTensor {
                name: "view".to_string(),
                storage_class: "FloatStorage",
                storage_key: "0".to_string(),
                bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                element_count: 6,
                storage_offset: 0,
                shape: vec![2, 3],
                // The transpose of a [3, 2] tensor.
                stride: vec![1, 2],
            }],
        );

        let err = read_torch_bytes(&bytes).expect_err("strided views must be reported");
        assert!(
            err.to_string().contains("not row-major contiguous"),
            "{err}"
        );
    }

    #[test]
    fn a_missing_storage_record_is_reported() {
        let values = vec![1.0f32; 4];
        let tensors = vec![float_tensor("w", "0", &values, vec![4])];
        // The state dict references `archive/data/0`, which this archive omits.
        let bytes = super::super::zip_archive::StoredZipBuilder::new()
            .add("archive/data.pkl", super::fixture::build_data_pkl(&tensors))
            .add("archive/version", b"3\n".to_vec())
            .build();

        let err = read_torch_bytes(&bytes).expect_err("missing storage");
        assert!(err.to_string().contains("is missing"), "{err}");
    }

    #[test]
    fn a_legacy_non_zip_checkpoint_is_refused_clearly() {
        let legacy = vec![0x80u8, 0x02, b'}', b'.'];
        let err = read_torch_bytes(&legacy).expect_err("legacy format");
        let message = err.to_string();
        assert!(message.contains("legacy"), "{message}");
        assert!(message.contains("safetensors"), "{message}");
    }

    #[test]
    fn a_deflated_checkpoint_is_refused_clearly() {
        let values = vec![1.0f32; 4];
        let tensors = vec![float_tensor("w", "0", &values, vec![4])];
        let bytes = super::super::zip_archive::StoredZipBuilder::new()
            .add("archive/data.pkl", super::fixture::build_data_pkl(&tensors))
            .add(
                "archive/data/0",
                values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            )
            .build_claiming_deflate("archive/data/0");

        let err = read_torch_bytes(&bytes).expect_err("deflate");
        assert!(err.to_string().contains("compression method 8"), "{err}");
    }

    #[test]
    fn an_archive_without_data_pkl_is_refused() {
        let bytes = super::super::zip_archive::StoredZipBuilder::new()
            .add("something/else.txt", b"hello".to_vec())
            .build();
        let err = read_torch_bytes(&bytes).expect_err("not a checkpoint");
        assert!(err.to_string().contains("data.pkl"), "{err}");
    }

    #[test]
    fn a_checkpoint_with_no_tensors_is_refused() {
        let bytes = super::super::zip_archive::StoredZipBuilder::new()
            .add("archive/data.pkl", vec![0x80, 2, b'}', b'.'])
            .build();
        let err = read_torch_bytes(&bytes).expect_err("no tensors");
        assert!(err.to_string().contains("contains no tensors"), "{err}");
    }

    #[test]
    fn a_nested_state_dict_wrapper_is_unwrapped() {
        // {"state_dict": {"w": tensor}} — build it by wrapping the inner pickle.
        let values = vec![3.0f32, 4.0];
        let inner = [float_tensor("w", "0", &values, vec![2])];

        // Hand-build {"state_dict": <inner dict>}.
        let mut pickle_bytes = vec![0x80u8, 2, b'}', b'('];
        pickle_bytes.push(b'X');
        pickle_bytes.extend_from_slice(&(10u32).to_le_bytes());
        pickle_bytes.extend_from_slice(b"state_dict");
        // Splice the inner dict's opcodes, minus its PROTO header and STOP.
        let inner_pickle = super::fixture::build_data_pkl(&inner);
        pickle_bytes.extend_from_slice(&inner_pickle[2..inner_pickle.len() - 1]);
        pickle_bytes.push(b'u');
        pickle_bytes.push(b'.');

        let bytes = super::super::zip_archive::StoredZipBuilder::new()
            .add("archive/data.pkl", pickle_bytes)
            .add(
                "archive/data/0",
                values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            )
            .build();

        let state = read_torch_bytes(&bytes).expect("read");
        assert_eq!(state.names(), vec!["w".to_string()]);
        assert_eq!(
            state.get("w").expect("w").tensor.to_vec_f32().expect("f32"),
            values
        );
    }

    #[test]
    fn the_archive_prefix_may_be_anything() {
        let values = vec![1.0f32, 2.0];
        let bytes = build_checkpoint("pytorch_model", &[float_tensor("w", "0", &values, vec![2])]);
        let state = read_torch_bytes(&bytes).expect("read");
        assert_eq!(
            state.get("w").expect("w").tensor.to_vec_f32().expect("f32"),
            values
        );
    }
}
