//! On-disk checkpoint format for [`super::SmartCheckpointManager`].
//!
//! # Why this exists
//!
//! The previous serializer wrote tensor payloads through `Tensor::to_vec_u8()`,
//! which is an `f32 as u8` *saturating value cast*: every weight in `[-1, 1]`
//! became `0u8`, so the checkpoint destroyed exactly the data it claimed to
//! save. This module writes IEEE-754 `f32` little-endian bytes and ships a
//! reader, so a save→load round trip is bit-identical.
//!
//! # Format
//!
//! All integers are little-endian. Parameters are written in sorted-name order
//! so that byte-identical states produce byte-identical files.
//!
//! ```text
//! full        := "TFRS_CKPT_FULL" u32:version u32:param_count param*
//! param       := u32:name_len name u32:ndim u32:dim* u32:count f32:count
//!
//! differential:= "TFRS_CKPT_DIFF" u32:version u32:base_step u32:param_count delta*
//! delta       := u32:name_len name u32:ndim u32:dim* u32:changed (u32:index f32:value)*
//! ```
//!
//! A differential entry records only the elements whose absolute change from
//! the base state exceeds a threshold; parameters absent from the base are
//! written with every index.
//!
//! Compression is zero-run-length encoding over the serialized bytes — real,
//! lossless and dependency-free. Checkpoint payloads (especially differential
//! ones) are dominated by zero bytes, which is exactly what this codec targets.

use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;

/// Magic prefix of a full checkpoint.
pub const MAGIC_FULL: &[u8; 14] = b"TFRS_CKPT_FULL";
/// Magic prefix of a differential checkpoint.
pub const MAGIC_DIFF: &[u8; 14] = b"TFRS_CKPT_DIFF";
/// Magic prefix of a zero-RLE compressed container.
pub const MAGIC_RLE: &[u8; 14] = b"TFRS_CKPT_RLE0";
/// Magic prefix of a container that was left uncompressed because encoding it
/// would have made it larger.
pub const MAGIC_RAW: &[u8; 14] = b"TFRS_CKPT_RAW0";

/// Format version written into every checkpoint.
pub const FORMAT_VERSION: u32 = 1;

fn corrupt(reason: impl Into<String>) -> TrustformersError {
    TrustformersError::invalid_input(format!("corrupt checkpoint: {}", reason.into()))
}

struct Cursor<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self.position.checked_add(count).ok_or_else(|| corrupt("length overflow"))?;
        if end > self.data.len() {
            return Err(corrupt(format!(
                "wanted {count} bytes at offset {} but only {} remain",
                self.position,
                self.data.len().saturating_sub(self.position)
            )));
        }
        let slice = &self.data[self.position..end];
        self.position = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32> {
        let bytes = self.take(4)?;
        let mut word = [0u8; 4];
        word.copy_from_slice(bytes);
        Ok(u32::from_le_bytes(word))
    }

    fn f32(&mut self) -> Result<f32> {
        let bytes = self.take(4)?;
        let mut word = [0u8; 4];
        word.copy_from_slice(bytes);
        Ok(f32::from_le_bytes(word))
    }

    fn string(&mut self, length: usize) -> Result<String> {
        let bytes = self.take(length)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|err| corrupt(format!("parameter name is not valid UTF-8: {err}")))
    }

    fn is_exhausted(&self) -> bool {
        self.position >= self.data.len()
    }
}

fn ordered_names(state: &HashMap<String, Tensor>) -> Vec<String> {
    let mut names: Vec<String> = state.keys().cloned().collect();
    names.sort();
    names
}

fn write_header(buffer: &mut Vec<u8>, name: &str, shape: &[usize]) {
    buffer.extend_from_slice(&(name.len() as u32).to_le_bytes());
    buffer.extend_from_slice(name.as_bytes());
    buffer.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    for dimension in shape {
        buffer.extend_from_slice(&(*dimension as u32).to_le_bytes());
    }
}

fn read_header(cursor: &mut Cursor<'_>) -> Result<(String, Vec<usize>)> {
    let name_length = cursor.u32()? as usize;
    let name = cursor.string(name_length)?;
    let ndim = cursor.u32()? as usize;
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        shape.push(cursor.u32()? as usize);
    }
    Ok((name, shape))
}

/// Serialize `state` as a full checkpoint.
///
/// Tensors are stored as `f32`; other dtypes are converted on the way out.
pub fn encode_full(state: &HashMap<String, Tensor>) -> Result<Vec<u8>> {
    let mut buffer = Vec::new();
    buffer.extend_from_slice(MAGIC_FULL);
    buffer.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    buffer.extend_from_slice(&(state.len() as u32).to_le_bytes());

    for name in ordered_names(state) {
        let tensor = state.get(&name).ok_or_else(|| corrupt("parameter vanished"))?;
        let shape = tensor.shape();
        let values = tensor.to_vec_f32()?;

        write_header(&mut buffer, &name, &shape);
        buffer.extend_from_slice(&(values.len() as u32).to_le_bytes());
        for value in &values {
            buffer.extend_from_slice(&value.to_le_bytes());
        }
    }

    Ok(buffer)
}

/// Parse a full checkpoint produced by [`encode_full`].
pub fn decode_full(data: &[u8]) -> Result<HashMap<String, Tensor>> {
    let mut cursor = Cursor::new(data);
    let magic = cursor.take(MAGIC_FULL.len())?;
    if magic != MAGIC_FULL {
        return Err(corrupt("not a full checkpoint (bad magic)"));
    }
    let version = cursor.u32()?;
    if version != FORMAT_VERSION {
        return Err(corrupt(format!("unsupported format version {version}")));
    }

    let param_count = cursor.u32()? as usize;
    let mut state = HashMap::with_capacity(param_count);

    for _ in 0..param_count {
        let (name, shape) = read_header(&mut cursor)?;
        let count = cursor.u32()? as usize;
        let expected: usize = shape.iter().product();
        if count != expected {
            return Err(corrupt(format!(
                "parameter `{name}` declares shape {shape:?} ({expected} elements) but stores \
                 {count}"
            )));
        }
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(cursor.f32()?);
        }
        state.insert(name, Tensor::from_slice(&values, &shape)?);
    }

    if !cursor.is_exhausted() {
        return Err(corrupt("trailing bytes after the last parameter"));
    }

    Ok(state)
}

/// Serialize the difference between `state` and `base`.
///
/// Only elements whose absolute change exceeds `threshold` are recorded.
/// Parameters missing from `base`, or whose shape changed, are recorded in
/// full.
pub fn encode_differential(
    state: &HashMap<String, Tensor>,
    base: &HashMap<String, Tensor>,
    base_step: usize,
    threshold: f32,
) -> Result<Vec<u8>> {
    let mut buffer = Vec::new();
    buffer.extend_from_slice(MAGIC_DIFF);
    buffer.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    buffer.extend_from_slice(&(base_step as u32).to_le_bytes());
    buffer.extend_from_slice(&(state.len() as u32).to_le_bytes());

    for name in ordered_names(state) {
        let tensor = state.get(&name).ok_or_else(|| corrupt("parameter vanished"))?;
        let shape = tensor.shape();
        let values = tensor.to_vec_f32()?;

        let base_values = match base.get(&name) {
            Some(base_tensor) if base_tensor.shape() == shape => Some(base_tensor.to_vec_f32()?),
            _ => None,
        };

        let mut deltas: Vec<(u32, f32)> = Vec::new();
        match base_values {
            Some(base_values) => {
                for (index, (value, base_value)) in values.iter().zip(&base_values).enumerate() {
                    if (value - base_value).abs() > threshold {
                        deltas.push((index as u32, *value));
                    }
                }
            },
            None => {
                deltas.extend(values.iter().enumerate().map(|(i, v)| (i as u32, *v)));
            },
        }

        write_header(&mut buffer, &name, &shape);
        buffer.extend_from_slice(&(deltas.len() as u32).to_le_bytes());
        for (index, value) in deltas {
            buffer.extend_from_slice(&index.to_le_bytes());
            buffer.extend_from_slice(&value.to_le_bytes());
        }
    }

    Ok(buffer)
}

/// Apply a differential checkpoint on top of `base`, returning the base step it
/// was recorded against and the reconstructed state.
pub fn decode_differential(
    data: &[u8],
    base: &HashMap<String, Tensor>,
) -> Result<(usize, HashMap<String, Tensor>)> {
    let mut cursor = Cursor::new(data);
    let magic = cursor.take(MAGIC_DIFF.len())?;
    if magic != MAGIC_DIFF {
        return Err(corrupt("not a differential checkpoint (bad magic)"));
    }
    let version = cursor.u32()?;
    if version != FORMAT_VERSION {
        return Err(corrupt(format!("unsupported format version {version}")));
    }

    let base_step = cursor.u32()? as usize;
    let param_count = cursor.u32()? as usize;
    let mut state = HashMap::with_capacity(param_count);

    for _ in 0..param_count {
        let (name, shape) = read_header(&mut cursor)?;
        let total: usize = shape.iter().product();

        let mut values = match base.get(&name) {
            Some(base_tensor) if base_tensor.shape() == shape => base_tensor.to_vec_f32()?,
            _ => vec![0.0f32; total],
        };

        let changed = cursor.u32()? as usize;
        for _ in 0..changed {
            let index = cursor.u32()? as usize;
            let value = cursor.f32()?;
            if index >= values.len() {
                return Err(corrupt(format!(
                    "delta index {index} is out of range for parameter `{name}` ({total} \
                     elements)"
                )));
            }
            values[index] = value;
        }

        state.insert(name, Tensor::from_slice(&values, &shape)?);
    }

    if !cursor.is_exhausted() {
        return Err(corrupt("trailing bytes after the last delta"));
    }

    Ok((base_step, state))
}

/// Whether `data` is a differential checkpoint payload.
pub fn is_differential(data: &[u8]) -> bool {
    data.starts_with(MAGIC_DIFF)
}

/// Losslessly compress `data` with zero-run-length encoding.
///
/// A zero byte is emitted as `0x00 run_length`, where `run_length` counts
/// consecutive zeros (1..=255). Non-zero bytes are copied verbatim. When the
/// encoding does not shrink the payload the original is stored under
/// [`MAGIC_RAW`], so the container never grows by more than its header.
pub fn compress(data: &[u8]) -> Vec<u8> {
    let mut encoded = Vec::with_capacity(data.len());
    let mut index = 0usize;
    while index < data.len() {
        let byte = data[index];
        if byte == 0 {
            let mut run = 1usize;
            while index + run < data.len() && data[index + run] == 0 && run < 255 {
                run += 1;
            }
            encoded.push(0u8);
            encoded.push(run as u8);
            index += run;
        } else {
            encoded.push(byte);
            index += 1;
        }
    }

    let mut container = Vec::with_capacity(encoded.len() + MAGIC_RLE.len() + 8);
    if encoded.len() < data.len() {
        container.extend_from_slice(MAGIC_RLE);
        container.extend_from_slice(&(data.len() as u64).to_le_bytes());
        container.extend_from_slice(&encoded);
    } else {
        container.extend_from_slice(MAGIC_RAW);
        container.extend_from_slice(&(data.len() as u64).to_le_bytes());
        container.extend_from_slice(data);
    }
    container
}

/// Reverse [`compress`]. Payloads without a container header are returned
/// unchanged, so uncompressed checkpoints load through the same path.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>> {
    if !data.starts_with(MAGIC_RLE) && !data.starts_with(MAGIC_RAW) {
        return Ok(data.to_vec());
    }

    let mut cursor = Cursor::new(data);
    let magic = cursor.take(MAGIC_RLE.len())?.to_vec();
    let length_bytes = cursor.take(8)?;
    let mut word = [0u8; 8];
    word.copy_from_slice(length_bytes);
    let original_length = u64::from_le_bytes(word) as usize;

    if magic == MAGIC_RAW {
        let payload = cursor.take(original_length)?.to_vec();
        return Ok(payload);
    }

    let mut decoded = Vec::with_capacity(original_length);
    while !cursor.is_exhausted() {
        let byte = cursor.take(1)?[0];
        if byte == 0 {
            let run = cursor.take(1)?[0] as usize;
            if run == 0 {
                return Err(corrupt("zero-length run in compressed stream"));
            }
            decoded.extend(std::iter::repeat_n(0u8, run));
        } else {
            decoded.push(byte);
        }
    }

    if decoded.len() != original_length {
        return Err(corrupt(format!(
            "decompressed {} bytes but the header declared {original_length}",
            decoded.len()
        )));
    }

    Ok(decoded)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> HashMap<String, Tensor> {
        let mut state = HashMap::new();
        state.insert(
            "layer.0.weight".to_string(),
            Tensor::from_slice(&[0.1, -0.25, 0.75, -0.999, 0.0, 0.5], &[2, 3])
                .expect("tensor must build in test"),
        );
        state.insert(
            "layer.0.bias".to_string(),
            Tensor::from_slice(&[-0.01, 0.02], &[2]).expect("tensor must build in test"),
        );
        state
    }

    #[test]
    fn full_checkpoint_round_trip_is_bit_identical() {
        let original = state();
        let encoded = encode_full(&original).expect("encode in test");
        let decoded = decode_full(&encoded).expect("decode in test");

        assert_eq!(decoded.len(), original.len());
        for (name, tensor) in &original {
            let restored = decoded.get(name).expect("parameter must survive the round trip");
            assert_eq!(restored.shape(), tensor.shape(), "{name} shape");
            assert_eq!(
                restored.to_vec_f32().expect("read"),
                tensor.to_vec_f32().expect("read"),
                "{name} values must be bit-identical"
            );
        }
    }

    #[test]
    fn full_checkpoint_does_not_zero_sub_unit_weights() {
        // Direct regression against `to_vec_u8()`: every value below 1.0 used
        // to serialize as 0u8.
        let mut original = HashMap::new();
        original.insert(
            "w".to_string(),
            Tensor::from_slice(&[0.001, -0.5, 0.9999], &[3]).expect("tensor must build in test"),
        );

        let decoded =
            decode_full(&encode_full(&original).expect("encode in test")).expect("decode in test");
        let restored =
            decoded.get("w").expect("parameter must survive").to_vec_f32().expect("read");

        assert_eq!(restored, vec![0.001f32, -0.5, 0.9999]);
        assert!(restored.iter().all(|value| *value != 0.0));
    }

    #[test]
    fn differential_checkpoint_stores_only_changed_elements() {
        let base = state();
        let mut updated = base.clone();
        updated.insert(
            "layer.0.weight".to_string(),
            Tensor::from_slice(&[0.1, -0.25, 0.75, -0.999, 0.0, 1.5], &[2, 3])
                .expect("tensor must build in test"),
        );

        let full = encode_full(&updated).expect("encode in test");
        let diff = encode_differential(&updated, &base, 7, 0.0).expect("encode in test");
        assert!(
            diff.len() < full.len(),
            "a one-element change must produce a smaller differential ({} vs {})",
            diff.len(),
            full.len()
        );

        let (base_step, restored) = decode_differential(&diff, &base).expect("decode in test");
        assert_eq!(base_step, 7);
        for (name, tensor) in &updated {
            assert_eq!(
                restored.get(name).expect("parameter must survive").to_vec_f32().expect("read"),
                tensor.to_vec_f32().expect("read"),
                "{name}"
            );
        }
    }

    #[test]
    fn differential_checkpoint_records_new_parameters_in_full() {
        let base = HashMap::new();
        let updated = state();
        let diff = encode_differential(&updated, &base, 0, 0.0).expect("encode in test");
        let (_, restored) = decode_differential(&diff, &base).expect("decode in test");

        for (name, tensor) in &updated {
            assert_eq!(
                restored.get(name).expect("parameter must survive").to_vec_f32().expect("read"),
                tensor.to_vec_f32().expect("read"),
                "{name}"
            );
        }
    }

    #[test]
    fn compression_round_trip_is_lossless_and_shrinks_sparse_payloads() {
        let mut sparse = HashMap::new();
        sparse.insert(
            "w".to_string(),
            Tensor::from_slice(&[0.0f32; 512], &[512]).expect("tensor must build in test"),
        );
        let encoded = encode_full(&sparse).expect("encode in test");
        let compressed = compress(&encoded);

        assert!(
            compressed.len() < encoded.len(),
            "an all-zero payload must actually compress ({} vs {})",
            compressed.len(),
            encoded.len()
        );
        assert_eq!(
            decompress(&compressed).expect("decompress in test"),
            encoded
        );
    }

    #[test]
    fn compression_never_grows_incompressible_payloads_beyond_the_header() {
        let incompressible: Vec<u8> = (0..=255u8).cycle().take(1024).filter(|b| *b != 0).collect();
        let compressed = compress(&incompressible);
        assert!(compressed.starts_with(MAGIC_RAW));
        assert_eq!(
            decompress(&compressed).expect("decompress in test"),
            incompressible
        );
    }

    #[test]
    fn decode_rejects_truncated_and_mislabelled_payloads() {
        let encoded = encode_full(&state()).expect("encode in test");
        assert!(decode_full(&encoded[..encoded.len() - 3]).is_err());
        assert!(decode_full(b"NOT_A_CHECKPOINT").is_err());
        assert!(decode_full(
            &encode_differential(&state(), &state(), 0, 0.0).expect("encode in test")
        )
        .is_err());
    }
}
