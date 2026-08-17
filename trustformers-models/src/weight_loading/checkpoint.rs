//! In-memory checkpoints for [`Model::load_pretrained`](trustformers_core::traits::Model::load_pretrained).
//!
//! `Model::load_pretrained` receives a `&mut dyn Read`, not a path, so the model
//! implementations in this crate need a parser that works on a byte buffer. This
//! module provides it:
//!
//! * [`Checkpoint::from_reader`] / [`Checkpoint::from_bytes`] detect the container
//!   format (safetensors or a PyTorch ZIP/pickle archive), parse it for real and
//!   return every tensor it contains, converted to `f32`.
//! * [`WeightBinder`] copies those tensors into a model's layers while recording
//!   exactly which checkpoint entries were consumed. [`WeightBinder::finish`] then
//!   fails loudly, listing every parameter the checkpoint did not supply and every
//!   checkpoint tensor the architecture did not recognise.
//!
//! The point of the binder is that a load either fills **every** parameter it was
//! asked for or returns an error naming the gaps. Synthesising a tensor — random,
//! zeroed or otherwise — when the checkpoint does not contain one is never an
//! option: a caller that gets `Ok(())` must be able to rely on the model actually
//! holding the checkpoint's weights.
//!
//! PyTorch `.bin` / `.pt` archives are handed to
//! [`trustformers_core::utils::weight_loading::PyTorchReader`], which walks the ZIP
//! central directory and interprets `data.pkl`. No pickle parsing happens here.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::io::Read;

use safetensors::{Dtype, SafeTensors};
use scirs2_core::ndarray::{ArrayD, IxDyn};
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;
use trustformers_core::utils::weight_loading::PyTorchReader;

/// Container format of a checkpoint buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointFormat {
    /// `safetensors`: 8-byte little-endian header length, JSON header, tensor data.
    SafeTensors,
    /// A `torch.save` ZIP archive (PyTorch >= 1.6).
    PyTorchZip,
    /// A bare pickle stream (PyTorch < 1.6 legacy layout).
    PyTorchLegacyPickle,
}

/// Every tensor of a checkpoint, converted to `f32` and keyed by its name.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    format: CheckpointFormat,
    tensors: BTreeMap<String, Tensor>,
}

impl Checkpoint {
    /// Read a whole checkpoint from a byte stream.
    ///
    /// # Errors
    ///
    /// Fails when the stream cannot be read, when the container format is not
    /// recognised, or when the container itself is malformed.
    pub fn from_reader(reader: &mut dyn Read) -> Result<Self> {
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer).map_err(|e| {
            TrustformersError::weight_load_error(format!("failed to read checkpoint bytes: {e}"))
        })?;
        Self::from_bytes(&buffer)
    }

    /// Parse a checkpoint held in memory.
    ///
    /// # Errors
    ///
    /// Fails when the container format cannot be identified or the container is
    /// malformed. It never falls back to guessing tensor shapes or values.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        match detect_format(bytes)? {
            CheckpointFormat::SafeTensors => Self::from_safetensors_bytes(bytes),
            format @ (CheckpointFormat::PyTorchZip | CheckpointFormat::PyTorchLegacyPickle) => {
                let reader = PyTorchReader::from_bytes(bytes)?;
                let mut tensors = BTreeMap::new();
                let state_dict = reader.state_dict();
                for name in state_dict.names() {
                    let record = state_dict.get(&name).ok_or_else(|| {
                        TrustformersError::weight_load_error(format!(
                            "PyTorch checkpoint listed tensor {name} but does not hold it"
                        ))
                    })?;
                    tensors.insert(name, to_f32_tensor(&record.tensor)?);
                }
                Ok(Self { format, tensors })
            },
        }
    }

    /// Parse a safetensors buffer.
    fn from_safetensors_bytes(bytes: &[u8]) -> Result<Self> {
        let parsed = SafeTensors::deserialize(bytes)
            .map_err(|e| TrustformersError::safe_tensors_error(e.to_string()))?;

        let mut tensors = BTreeMap::new();
        for (name, view) in parsed.tensors() {
            let shape = view.shape().to_vec();
            let values = decode_safetensors_values(view.dtype(), view.data(), &name)?;
            let expected: usize = shape.iter().product();
            if values.len() != expected {
                return Err(TrustformersError::shape_error(format!(
                    "safetensors tensor {name} declares shape {shape:?} ({expected} elements) but \
                     carries {} elements",
                    values.len()
                )));
            }
            let array = ArrayD::from_shape_vec(IxDyn(&shape), values)
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
            tensors.insert(name, Tensor::F32(array));
        }

        Ok(Self {
            format: CheckpointFormat::SafeTensors,
            tensors,
        })
    }

    /// The container format this checkpoint was parsed from.
    pub fn format(&self) -> CheckpointFormat {
        self.format
    }

    /// Number of tensors in the checkpoint.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the checkpoint holds no tensors at all.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Every tensor name, in sorted order.
    pub fn names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    /// Whether a tensor with this exact name exists.
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Borrow a tensor by its exact name.
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Pick the first candidate prefix under which `probe` resolves.
    ///
    /// HuggingFace checkpoints for the same architecture differ only by a task
    /// wrapper prefix (`bert.embeddings...` for `BertForSequenceClassification`
    /// versus `embeddings...` for `BertModel`), so the caller supplies the
    /// candidates in preference order together with a parameter that every
    /// variant is guaranteed to have.
    ///
    /// # Errors
    ///
    /// Fails when no candidate resolves, listing the candidates it tried and a
    /// sample of the names the checkpoint actually holds.
    pub fn detect_prefix(&self, candidates: &[&str], probe: &str) -> Result<String> {
        for candidate in candidates {
            if self.contains(&format!("{candidate}{probe}")) {
                return Ok((*candidate).to_string());
            }
        }
        Err(TrustformersError::weight_load_error(format!(
            "checkpoint does not look like the expected architecture: none of the prefixes {:?} \
             resolve the probe parameter {:?}. Checkpoint holds {} tensors, for example: {}",
            candidates,
            probe,
            self.tensors.len(),
            sample_names(&self.names(), 8)
        )))
    }

    /// Fetch a tensor by exact name, checking its shape.
    ///
    /// Returns `None` when the checkpoint does not hold it — which is a
    /// legitimate outcome for an optional task head, and never a licence to
    /// substitute one.
    ///
    /// # Errors
    ///
    /// Fails when the tensor exists but does not have `expected_shape`.
    pub fn take_shaped(&self, name: &str, expected_shape: &[usize]) -> Result<Option<Tensor>> {
        let Some(tensor) = self.get(name) else {
            return Ok(None);
        };
        let actual = tensor.shape();
        if actual != expected_shape {
            return Err(TrustformersError::shape_error(format!(
                "checkpoint tensor {name} has shape {actual:?} but this model expects \
                 {expected_shape:?}"
            )));
        }
        Ok(Some(tensor.clone()))
    }

    /// Start binding this checkpoint's tensors into a model.
    pub fn binder<'a>(&'a self, prefix: &str) -> WeightBinder<'a> {
        WeightBinder {
            checkpoint: self,
            prefix: prefix.to_string(),
            consumed: HashSet::new(),
            missing: Vec::new(),
            loaded: Vec::new(),
        }
    }
}

/// Adapter exposing a parsed [`Checkpoint`] as a
/// [`WeightReader`](trustformers_core::traits::WeightReader).
///
/// Models whose loaders were written against `WeightReader` — GPT-2's
/// `load_weights_from_reader`, for instance — can then be driven from a byte
/// stream without a second implementation of the name mapping.
#[derive(Debug, Clone)]
pub struct CheckpointReader {
    checkpoint: Checkpoint,
}

impl CheckpointReader {
    /// Wrap a parsed checkpoint.
    pub fn new(checkpoint: Checkpoint) -> Self {
        Self { checkpoint }
    }

    /// Parse a checkpoint from a byte stream and wrap it.
    ///
    /// # Errors
    ///
    /// See [`Checkpoint::from_reader`].
    pub fn from_reader(reader: &mut dyn Read) -> Result<Self> {
        Ok(Self::new(Checkpoint::from_reader(reader)?))
    }

    /// The wrapped checkpoint.
    pub fn checkpoint(&self) -> &Checkpoint {
        &self.checkpoint
    }
}

impl trustformers_core::traits::WeightReader for CheckpointReader {
    fn read_tensor(&mut self, name: &str) -> Result<Tensor> {
        self.checkpoint.get(name).cloned().ok_or_else(|| {
            TrustformersError::weight_load_error(format!(
                "checkpoint has no tensor named {name}; it holds {} tensor(s), for example: {}",
                self.checkpoint.len(),
                sample_names(&self.checkpoint.names(), 8)
            ))
        })
    }

    fn list_tensors(&self) -> Vec<String> {
        self.checkpoint.names()
    }
}

/// Which unconsumed checkpoint entries an architecture tolerates.
///
/// Two shapes of legitimate leftover exist, and a prefix list alone cannot
/// express the second:
///
/// * whole **namespaces** — the pretraining and task heads HuggingFace ships in
///   the same file as the encoder (`cls.`, `classifier.`, `vocab_projector.`);
/// * per-layer **non-parameter buffers** — `…self_attn.rotary_emb.inv_freq`,
///   `…attn.masked_bias`, `…embeddings.position_ids`. These repeat under every
///   layer prefix, so they are matched by suffix.
///
/// Everything outside both lists fails the load: strictness on real weights is
/// the property worth keeping, strictness on a cached frequency table is not.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnusedTensors<'a> {
    /// Namespaces (matched with `starts_with`) that need not be consumed.
    pub prefixes: &'a [&'a str],
    /// Buffer names (matched with `ends_with`) that need not be consumed.
    pub suffixes: &'a [&'a str],
}

impl<'a> UnusedTensors<'a> {
    /// Build a policy from a namespace list and a buffer-suffix list.
    pub const fn new(prefixes: &'a [&'a str], suffixes: &'a [&'a str]) -> Self {
        Self { prefixes, suffixes }
    }

    /// Whether this leftover checkpoint entry is acceptable.
    pub fn tolerates(&self, name: &str) -> bool {
        self.prefixes.iter().any(|prefix| name.starts_with(prefix))
            || self.suffixes.iter().any(|suffix| name.ends_with(suffix))
    }
}

impl<'a> From<&'a [&'a str]> for UnusedTensors<'a> {
    fn from(prefixes: &'a [&'a str]) -> Self {
        Self::new(prefixes, &[])
    }
}

impl<'a, const N: usize> From<&'a [&'a str; N]> for UnusedTensors<'a> {
    fn from(prefixes: &'a [&'a str; N]) -> Self {
        Self::new(prefixes, &[])
    }
}

impl<'a> From<&'a UnusedTensors<'a>> for UnusedTensors<'a> {
    fn from(policy: &'a UnusedTensors<'a>) -> Self {
        *policy
    }
}

/// What a completed load actually did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoadReport {
    /// Checkpoint tensor names that were copied into model parameters.
    pub loaded: Vec<String>,
    /// Parameters the architecture asked for that the checkpoint did not hold.
    pub missing: Vec<String>,
    /// Checkpoint tensors that the architecture did not consume and that were
    /// not covered by an explicitly allowed auxiliary namespace.
    pub unexpected: Vec<String>,
    /// Checkpoint tensors that were left unconsumed but belong to an explicitly
    /// allowed auxiliary namespace (task heads, pretraining heads, ...).
    pub ignored: Vec<String>,
}

impl LoadReport {
    /// Whether every requested parameter was filled from the checkpoint.
    pub fn is_complete(&self) -> bool {
        self.missing.is_empty()
    }

    /// Record that a tensor previously reported as unconsumed was in fact bound.
    ///
    /// Task wrappers load the encoder first and then their own head from the
    /// same checkpoint; without this the head would keep showing up under
    /// [`LoadReport::ignored`] even though it reached the model.
    pub fn mark_loaded(&mut self, name: &str) {
        self.ignored.retain(|entry| entry != name);
        self.unexpected.retain(|entry| entry != name);
        if !self.loaded.iter().any(|entry| entry == name) {
            self.loaded.push(name.to_string());
            self.loaded.sort();
        }
    }

    /// Record a head parameter the checkpoint did not carry.
    ///
    /// Unlike a missing *encoder* parameter this is not fatal — a pretrained
    /// backbone legitimately ships without a fine-tuned head — but it must be
    /// visible, because the corresponding layer keeps its random initialisation.
    pub fn note_absent(&mut self, name: &str) {
        if !self.missing.iter().any(|entry| entry == name) {
            self.missing.push(name.to_string());
            self.missing.sort();
        }
    }
}

/// Copies checkpoint tensors into model parameters, tracking what was used.
///
/// Look-ups go through [`WeightBinder::take`], which records both the parameters
/// that could not be resolved and the checkpoint entries that were consumed, so
/// that [`WeightBinder::finish`] can report the complete picture in one error
/// instead of failing on the first gap.
pub struct WeightBinder<'a> {
    checkpoint: &'a Checkpoint,
    prefix: String,
    consumed: HashSet<String>,
    missing: Vec<String>,
    loaded: Vec<String>,
}

impl<'a> WeightBinder<'a> {
    /// The prefix every logical name is resolved under.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// The checkpoint being bound.
    pub fn checkpoint(&self) -> &'a Checkpoint {
        self.checkpoint
    }

    /// Fully-qualified checkpoint name for a logical parameter name.
    pub fn qualified(&self, name: &str) -> String {
        format!("{}{}", self.prefix, name)
    }

    /// Whether the checkpoint holds this logical parameter.
    pub fn has(&self, name: &str) -> bool {
        self.checkpoint.contains(&self.qualified(name))
    }

    /// Take a required parameter.
    ///
    /// Returns `None` when the checkpoint has no such tensor, recording the name
    /// so that [`WeightBinder::finish`] can report it. Callers therefore skip the
    /// assignment rather than substituting an invented tensor.
    pub fn take(&mut self, name: &str) -> Option<Tensor> {
        let qualified = self.qualified(name);
        match self.checkpoint.get(&qualified) {
            Some(tensor) => {
                if self.consumed.insert(qualified.clone()) {
                    self.loaded.push(qualified);
                }
                Some(tensor.clone())
            },
            None => {
                self.missing.push(qualified);
                None
            },
        }
    }

    /// Take a required parameter and check its shape before handing it over.
    ///
    /// A shape mismatch is a hard error: it means the checkpoint belongs to a
    /// differently-configured model and silently reshaping it would corrupt the
    /// weights.
    ///
    /// # Errors
    ///
    /// Fails when the tensor exists but does not have `expected_shape`.
    pub fn take_shaped(&mut self, name: &str, expected_shape: &[usize]) -> Result<Option<Tensor>> {
        let qualified = self.qualified(name);
        let Some(tensor) = self.take(name) else {
            return Ok(None);
        };
        let actual = tensor.shape();
        if actual != expected_shape {
            return Err(TrustformersError::shape_error(format!(
                "checkpoint tensor {qualified} has shape {actual:?} but this model expects \
                 {expected_shape:?}"
            )));
        }
        Ok(Some(tensor))
    }

    /// Take an optional parameter: absent is not an error and is not recorded.
    pub fn take_optional(&mut self, name: &str) -> Option<Tensor> {
        if self.has(name) {
            self.take(name)
        } else {
            None
        }
    }

    /// Mark a fully-qualified checkpoint name as consumed without reading it.
    ///
    /// Used when one checkpoint tensor feeds several model parameters (a fused
    /// `qkv_proj`, for instance) and the caller reads it through
    /// [`Checkpoint::get`] directly.
    pub fn mark_consumed(&mut self, qualified: &str) {
        if self.checkpoint.contains(qualified) && self.consumed.insert(qualified.to_string()) {
            self.loaded.push(qualified.to_string());
        }
    }

    /// Finish the load and report what happened.
    ///
    /// `allowed` describes the checkpoint entries this architecture legitimately
    /// does not consume. Anything else left over is treated as evidence that the
    /// checkpoint does not match the architecture.
    ///
    /// # Errors
    ///
    /// Fails when any requested parameter was missing, or when the checkpoint
    /// holds unconsumed tensors that `allowed` does not cover. The message lists
    /// both sets.
    pub fn finish<'p, A: Into<UnusedTensors<'p>>>(self, allowed: A) -> Result<LoadReport> {
        let allowed: UnusedTensors<'p> = allowed.into();
        let mut unexpected = Vec::new();
        let mut ignored = Vec::new();
        for name in self.checkpoint.names() {
            if self.consumed.contains(&name) {
                continue;
            }
            if allowed.tolerates(&name) {
                ignored.push(name);
            } else {
                unexpected.push(name);
            }
        }

        let missing: Vec<String> = dedup_sorted(self.missing);
        if !missing.is_empty() || !unexpected.is_empty() {
            let mut message = String::from("checkpoint does not match this model");
            if !missing.is_empty() {
                message.push_str(&format!(
                    "; {} parameter(s) missing from the checkpoint: {}",
                    missing.len(),
                    sample_names(&missing, 12)
                ));
            }
            if !unexpected.is_empty() {
                message.push_str(&format!(
                    "; {} checkpoint tensor(s) not recognised by this architecture: {}",
                    unexpected.len(),
                    sample_names(&unexpected, 12)
                ));
            }
            return Err(TrustformersError::weight_load_error(message));
        }

        Ok(LoadReport {
            loaded: dedup_sorted(self.loaded),
            missing,
            unexpected,
            ignored,
        })
    }
}

/// Identify the container format of a checkpoint buffer.
///
/// # Errors
///
/// Fails when the buffer is neither safetensors nor a PyTorch archive.
pub fn detect_format(bytes: &[u8]) -> Result<CheckpointFormat> {
    if bytes.len() < 8 {
        return Err(TrustformersError::weight_load_error(format!(
            "checkpoint buffer is {} byte(s) long; too small to hold any supported format",
            bytes.len()
        )));
    }

    if bytes.starts_with(b"PK\x03\x04") {
        return Ok(CheckpointFormat::PyTorchZip);
    }

    if bytes.starts_with(b"GGUF") {
        return Err(TrustformersError::weight_load_error(
            "this is a GGUF file; load it with weight_loading::GGUFLoader, which needs a file path \
             rather than a byte stream"
                .to_string(),
        ));
    }

    // safetensors: an 8-byte little-endian header length followed by that many
    // bytes of JSON object.
    let mut header_len_bytes = [0u8; 8];
    header_len_bytes.copy_from_slice(&bytes[..8]);
    let header_len = u64::from_le_bytes(header_len_bytes) as usize;
    if header_len > 0 && header_len.checked_add(8).is_some_and(|end| end <= bytes.len()) {
        let header = &bytes[8..8 + header_len];
        if serde_json::from_slice::<serde_json::Map<String, serde_json::Value>>(header).is_ok() {
            return Ok(CheckpointFormat::SafeTensors);
        }
    }

    // Bare pickle protocol opcode: `PROTO <version>`.
    if bytes[0] == 0x80 && (2..=5).contains(&bytes[1]) {
        return Ok(CheckpointFormat::PyTorchLegacyPickle);
    }

    Err(TrustformersError::weight_load_error(
        "unrecognised checkpoint container: expected safetensors (8-byte header length + JSON \
         header) or a PyTorch archive (ZIP 'PK\\x03\\x04' or a pickle PROTO opcode)"
            .to_string(),
    ))
}

/// Decode a safetensors tensor payload into `f32` values.
fn decode_safetensors_values(dtype: Dtype, data: &[u8], name: &str) -> Result<Vec<f32>> {
    let expect_stride = |stride: usize| -> Result<()> {
        if data.len().is_multiple_of(stride) {
            Ok(())
        } else {
            Err(TrustformersError::weight_load_error(format!(
                "safetensors tensor {name} has {} payload bytes, which is not a multiple of the \
                 {stride}-byte element size of {dtype:?}",
                data.len()
            )))
        }
    };

    match dtype {
        Dtype::F32 => {
            expect_stride(4)?;
            Ok(data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        },
        Dtype::F16 => {
            expect_stride(2)?;
            Ok(data
                .chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect())
        },
        Dtype::BF16 => {
            expect_stride(2)?;
            Ok(data
                .chunks_exact(2)
                .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect())
        },
        Dtype::F64 => {
            expect_stride(8)?;
            Ok(data
                .chunks_exact(8)
                .map(|c| {
                    f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect())
        },
        Dtype::I8 => Ok(data.iter().map(|&b| b as i8 as f32).collect()),
        Dtype::U8 => Ok(data.iter().map(|&b| b as f32).collect()),
        Dtype::BOOL => Ok(data.iter().map(|&b| if b == 0 { 0.0 } else { 1.0 }).collect()),
        Dtype::I16 => {
            expect_stride(2)?;
            Ok(data.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32).collect())
        },
        Dtype::U16 => {
            expect_stride(2)?;
            Ok(data.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as f32).collect())
        },
        Dtype::I32 => {
            expect_stride(4)?;
            Ok(data
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                .collect())
        },
        Dtype::U32 => {
            expect_stride(4)?;
            Ok(data
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                .collect())
        },
        Dtype::I64 => {
            expect_stride(8)?;
            Ok(data
                .chunks_exact(8)
                .map(|c| {
                    i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect())
        },
        Dtype::U64 => {
            expect_stride(8)?;
            Ok(data
                .chunks_exact(8)
                .map(|c| {
                    u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect())
        },
        other => Err(TrustformersError::weight_load_error(format!(
            "safetensors tensor {name} uses dtype {other:?}, which this loader cannot decode \
             without losing information"
        ))),
    }
}

/// Convert any dense CPU tensor to an `f32` tensor, preserving shape.
pub fn to_f32_tensor(tensor: &Tensor) -> Result<Tensor> {
    match tensor {
        Tensor::F32(_) => Ok(tensor.clone()),
        Tensor::F64(arr) => Ok(Tensor::F32(arr.mapv(|v| v as f32))),
        Tensor::F16(arr) => Ok(Tensor::F32(arr.mapv(|v| v.to_f32()))),
        Tensor::BF16(arr) => Ok(Tensor::F32(arr.mapv(|v| v.to_f32()))),
        Tensor::I64(arr) => Ok(Tensor::F32(arr.mapv(|v| v as f32))),
        other => Err(TrustformersError::weight_load_error(format!(
            "checkpoint tensor has element type {:?}, which is not a real-valued weight type",
            std::mem::discriminant(other)
        ))),
    }
}

/// Render at most `limit` names for an error message.
fn sample_names(names: &[String], limit: usize) -> String {
    if names.len() <= limit {
        return names.join(", ");
    }
    format!(
        "{}, ... ({} more)",
        names[..limit].join(", "),
        names.len() - limit
    )
}

/// Sort and de-duplicate a name list for stable reporting.
fn dedup_sorted(names: Vec<String>) -> Vec<String> {
    names.into_iter().collect::<BTreeSet<_>>().into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_loading::test_support::{build_safetensors, F32Tensor};

    #[test]
    fn detect_format_recognises_safetensors() {
        let bytes = build_safetensors(&[F32Tensor::new(
            "a.weight",
            &[2, 2],
            vec![1.0, 2.0, 3.0, 4.0],
        )]);
        assert_eq!(
            detect_format(&bytes).expect("format detection must succeed"),
            CheckpointFormat::SafeTensors
        );
    }

    #[test]
    fn detect_format_recognises_pytorch_zip() {
        let bytes = b"PK\x03\x04rest-of-archive".to_vec();
        assert_eq!(
            detect_format(&bytes).expect("format detection must succeed"),
            CheckpointFormat::PyTorchZip
        );
    }

    #[test]
    fn detect_format_rejects_garbage() {
        let bytes = vec![0xAB; 512];
        let err = detect_format(&bytes).expect_err("garbage must not be accepted");
        assert!(
            err.to_string().contains("unrecognised checkpoint container"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn detect_format_rejects_gguf_with_actionable_message() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&[0u8; 32]);
        let err = detect_format(&bytes).expect_err("GGUF must not parse as a checkpoint");
        assert!(
            err.to_string().contains("GGUFLoader"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn checkpoint_round_trips_exact_f32_values() {
        let bytes = build_safetensors(&[
            F32Tensor::new(
                "embeddings.weight",
                &[2, 3],
                vec![-1.5, 0.25, 7.0, 3.5, -0.125, 2.0],
            ),
            F32Tensor::new("norm.bias", &[3], vec![0.5, -0.5, 1.25]),
        ]);
        let checkpoint = Checkpoint::from_bytes(&bytes).expect("checkpoint must parse");
        assert_eq!(checkpoint.len(), 2);
        let embeddings =
            checkpoint.get("embeddings.weight").expect("embeddings tensor must be present");
        assert_eq!(embeddings.shape(), vec![2, 3]);
        match embeddings {
            Tensor::F32(arr) => {
                let values: Vec<f32> = arr.iter().copied().collect();
                assert_eq!(values, vec![-1.5, 0.25, 7.0, 3.5, -0.125, 2.0]);
            },
            other => panic!("expected an F32 tensor, got {other:?}"),
        }
    }

    #[test]
    fn checkpoint_decodes_bf16_payloads() {
        // BF16 is the dtype real Phi-3 / Llama checkpoints ship in.
        let values: Vec<f32> = vec![1.0, -2.0, 0.5, 256.0];
        let mut payload = Vec::new();
        for value in &values {
            payload.extend_from_slice(&half::bf16::from_f32(*value).to_bits().to_le_bytes());
        }
        let header = serde_json::json!({
            "w": {"dtype": "BF16", "shape": [4], "data_offsets": [0, payload.len()]}
        });
        let header_bytes = serde_json::to_vec(&header).expect("header must serialise");
        let mut bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header_bytes);
        bytes.extend_from_slice(&payload);

        let checkpoint = Checkpoint::from_bytes(&bytes).expect("BF16 checkpoint must parse");
        match checkpoint.get("w").expect("tensor w must be present") {
            Tensor::F32(arr) => {
                let decoded: Vec<f32> = arr.iter().copied().collect();
                assert_eq!(decoded, values);
            },
            other => panic!("expected an F32 tensor, got {other:?}"),
        }
    }

    #[test]
    fn binder_reports_every_missing_parameter_at_once() {
        let bytes = build_safetensors(&[F32Tensor::new("a", &[1], vec![1.0])]);
        let checkpoint = Checkpoint::from_bytes(&bytes).expect("checkpoint must parse");
        let mut binder = checkpoint.binder("");
        assert!(binder.take("a").is_some());
        assert!(binder.take("b").is_none());
        assert!(binder.take("c").is_none());
        let err = binder
            .finish(UnusedTensors::default())
            .expect_err("missing parameters must fail the load");
        let message = err.to_string();
        assert!(
            message.contains("2 parameter(s) missing"),
            "unexpected: {message}"
        );
        assert!(
            message.contains('b') && message.contains('c'),
            "unexpected: {message}"
        );
    }

    #[test]
    fn binder_rejects_unconsumed_tensors_outside_allowed_namespaces() {
        let bytes = build_safetensors(&[
            F32Tensor::new("a", &[1], vec![1.0]),
            F32Tensor::new("mystery.weight", &[1], vec![2.0]),
        ]);
        let checkpoint = Checkpoint::from_bytes(&bytes).expect("checkpoint must parse");
        let mut binder = checkpoint.binder("");
        assert!(binder.take("a").is_some());
        let err = binder
            .finish(UnusedTensors::default())
            .expect_err("unexpected tensors must fail the load");
        assert!(
            err.to_string().contains("mystery.weight"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn binder_tolerates_allowed_auxiliary_namespaces() {
        let bytes = build_safetensors(&[
            F32Tensor::new("a", &[1], vec![1.0]),
            F32Tensor::new("cls.predictions.bias", &[1], vec![2.0]),
        ]);
        let checkpoint = Checkpoint::from_bytes(&bytes).expect("checkpoint must parse");
        let mut binder = checkpoint.binder("");
        assert!(binder.take("a").is_some());
        let report = binder
            .finish(UnusedTensors::new(&["cls."], &[]))
            .expect("allowed extras must not fail the load");
        assert_eq!(report.loaded, vec!["a".to_string()]);
        assert_eq!(report.ignored, vec!["cls.predictions.bias".to_string()]);
        assert!(report.is_complete());
    }

    #[test]
    fn binder_shape_check_rejects_mismatched_tensor() {
        let bytes = build_safetensors(&[F32Tensor::new("w", &[2, 2], vec![1.0, 2.0, 3.0, 4.0])]);
        let checkpoint = Checkpoint::from_bytes(&bytes).expect("checkpoint must parse");
        let mut binder = checkpoint.binder("");
        let err = binder
            .take_shaped("w", &[4, 1])
            .expect_err("a shape mismatch must be an error, not a reshape");
        assert!(err.to_string().contains("[2, 2]"), "unexpected: {err}");
    }

    #[test]
    fn detect_prefix_finds_the_task_wrapper_prefix() {
        let bytes = build_safetensors(&[F32Tensor::new(
            "bert.embeddings.word_embeddings.weight",
            &[1, 1],
            vec![1.0],
        )]);
        let checkpoint = Checkpoint::from_bytes(&bytes).expect("checkpoint must parse");
        let prefix = checkpoint
            .detect_prefix(&["", "bert."], "embeddings.word_embeddings.weight")
            .expect("prefix detection must succeed");
        assert_eq!(prefix, "bert.");
    }

    #[test]
    fn detect_prefix_errors_when_no_candidate_matches() {
        let bytes = build_safetensors(&[F32Tensor::new("totally.other", &[1], vec![1.0])]);
        let checkpoint = Checkpoint::from_bytes(&bytes).expect("checkpoint must parse");
        let err = checkpoint
            .detect_prefix(&["", "bert."], "embeddings.word_embeddings.weight")
            .expect_err("a foreign checkpoint must be rejected");
        assert!(
            err.to_string().contains("totally.other"),
            "unexpected: {err}"
        );
    }
}
