//! Fixture builders for weight-loading tests.
//!
//! These helpers build **real** safetensors and GGUF byte streams in memory, so
//! the loader tests exercise the same parsing path a downloaded checkpoint would
//! take instead of a mocked shortcut. They are compiled only for tests.

use std::collections::BTreeMap;

/// A little-endian `f32` tensor destined for a safetensors fixture.
#[derive(Debug, Clone)]
pub struct F32Tensor {
    /// Fully-qualified checkpoint name.
    pub name: String,
    /// Row-major shape.
    pub shape: Vec<usize>,
    /// Row-major values; `shape.iter().product()` of them.
    pub values: Vec<f32>,
}

impl F32Tensor {
    /// Build a tensor entry, checking the value count against the shape.
    pub fn new(name: &str, shape: &[usize], values: Vec<f32>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            values.len(),
            expected,
            "fixture tensor {name}: shape {shape:?} needs {expected} values, got {}",
            values.len()
        );
        Self {
            name: name.to_string(),
            shape: shape.to_vec(),
            values,
        }
    }

    /// Build a tensor filled with a deterministic ramp, distinct per name.
    ///
    /// Every element is unique within the tensor and the sequence depends on the
    /// name, so a test that asserts "parameter X holds tensor Y" cannot pass by
    /// accident when the binder wires two parameters the wrong way round.
    pub fn ramp(name: &str, shape: &[usize], seed: f32) -> Self {
        let count: usize = shape.iter().product();
        let values = (0..count).map(|i| seed + i as f32 * 0.5).collect();
        Self::new(name, shape, values)
    }
}

/// Serialise tensors into a valid safetensors byte stream.
pub fn build_safetensors(tensors: &[F32Tensor]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut payload: Vec<u8> = Vec::new();

    // safetensors requires the header's data_offsets to be sorted and contiguous,
    // and `SafeTensors::deserialize` validates that, so emit in a stable order.
    let ordered: BTreeMap<&str, &F32Tensor> =
        tensors.iter().map(|t| (t.name.as_str(), t)).collect();

    for (name, tensor) in ordered {
        let start = payload.len();
        for value in &tensor.values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        let end = payload.len();
        header.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": "F32",
                "shape": tensor.shape,
                "data_offsets": [start, end],
            }),
        );
    }

    let header_bytes =
        serde_json::to_vec(&serde_json::Value::Object(header)).expect("fixture header serialises");
    let mut bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header_bytes);
    bytes.extend_from_slice(&payload);
    bytes
}

/// A metadata value for a GGUF fixture.
#[derive(Debug, Clone)]
pub enum GgufMetaValue {
    /// `GGUF_TYPE_UINT32`
    U32(u32),
    /// `GGUF_TYPE_STRING`
    Str(String),
    /// `GGUF_TYPE_ARRAY` of strings.
    StrArray(Vec<String>),
    /// `GGUF_TYPE_ARRAY` of `f32`.
    F32Array(Vec<f32>),
}

/// A tensor entry for a GGUF fixture: name, ggml type id, dims and raw bytes.
#[derive(Debug, Clone)]
pub struct GgufTensor {
    /// Tensor name.
    pub name: String,
    /// ggml type id (0 = F32, 12 = Q4_K, ...).
    pub ggml_type: u32,
    /// ggml dimension order (`ne[0]` fastest-varying first).
    pub dimensions: Vec<u64>,
    /// Already-quantised raw payload.
    pub data: Vec<u8>,
}

fn push_gguf_string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn push_gguf_value(out: &mut Vec<u8>, value: &GgufMetaValue) {
    match value {
        GgufMetaValue::U32(v) => {
            out.extend_from_slice(&4u32.to_le_bytes());
            out.extend_from_slice(&v.to_le_bytes());
        },
        GgufMetaValue::Str(v) => {
            out.extend_from_slice(&8u32.to_le_bytes());
            push_gguf_string(out, v);
        },
        GgufMetaValue::StrArray(items) => {
            out.extend_from_slice(&9u32.to_le_bytes());
            out.extend_from_slice(&8u32.to_le_bytes()); // element type: string
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                push_gguf_string(out, item);
            }
        },
        GgufMetaValue::F32Array(items) => {
            out.extend_from_slice(&9u32.to_le_bytes());
            out.extend_from_slice(&6u32.to_le_bytes()); // element type: f32
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                out.extend_from_slice(&item.to_le_bytes());
            }
        },
    }
}

/// Serialise a GGUF v3 file with the given metadata and tensors.
///
/// Tensor payloads are laid out back to back, each aligned to `alignment`
/// (written into the file as `general.alignment` when it differs from 32).
pub fn build_gguf(
    metadata: &[(String, GgufMetaValue)],
    tensors: &[GgufTensor],
    alignment: u64,
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());

    let mut kv: Vec<(String, GgufMetaValue)> = metadata.to_vec();
    if alignment != 32 {
        kv.push((
            "general.alignment".to_string(),
            GgufMetaValue::U32(alignment as u32),
        ));
    }
    out.extend_from_slice(&(kv.len() as u64).to_le_bytes());
    for (key, value) in &kv {
        push_gguf_string(&mut out, key);
        push_gguf_value(&mut out, value);
    }

    // Tensor offsets are relative to the aligned start of the data section.
    let mut relative_offset: u64 = 0;
    let mut offsets = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        offsets.push(relative_offset);
        let padded = tensor.data.len() as u64;
        relative_offset += padded.div_ceil(alignment) * alignment;
    }

    for (tensor, offset) in tensors.iter().zip(offsets.iter()) {
        push_gguf_string(&mut out, &tensor.name);
        out.extend_from_slice(&(tensor.dimensions.len() as u32).to_le_bytes());
        for dim in &tensor.dimensions {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(&tensor.ggml_type.to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
    }

    // Pad up to the aligned start of the tensor data section.
    let data_start = (out.len() as u64).div_ceil(alignment) * alignment;
    out.resize(data_start as usize, 0);

    for (tensor, offset) in tensors.iter().zip(offsets.iter()) {
        let absolute = (data_start + offset) as usize;
        if out.len() < absolute {
            out.resize(absolute, 0);
        }
        out.extend_from_slice(&tensor.data);
    }

    out
}

/// Write bytes to a uniquely named file under the system temp directory.
///
/// Returns the path; the caller removes it.
pub fn write_temp_file(stem: &str, extension: &str, bytes: &[u8]) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "trustformers_{stem}_{}_{unique}.{extension}",
        std::process::id()
    ));
    std::fs::write(&path, bytes).expect("fixture file must be writable");
    path
}

/// Shape of a BERT-family checkpoint fixture.
///
/// The tensor names come from [`crate::bert::layers::BertLayerNames`], the same
/// table the loader binds against, so a test proves the plumbing (a value written
/// under name X reaches parameter Y) rather than re-asserting a spelling.
#[cfg(feature = "bert")]
#[derive(Debug, Clone)]
pub struct BertFixtureSpec {
    /// Task-wrapper prefix, e.g. `""` or `"bert."`.
    pub prefix: String,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden width.
    pub hidden_size: usize,
    /// Number of encoder layers.
    pub num_layers: usize,
    /// Feed-forward width.
    pub intermediate_size: usize,
    /// Position table length.
    pub max_position_embeddings: usize,
    /// Segment table length, or `None` for architectures without one.
    pub type_vocab_size: Option<usize>,
    /// Whether to emit `pooler.dense.*`.
    pub include_pooler: bool,
    /// Parameter spelling for the encoder stack.
    pub names: crate::bert::layers::BertLayerNames,
}

#[cfg(feature = "bert")]
impl BertFixtureSpec {
    /// Every tensor a BERT-family encoder of this shape expects, with distinct
    /// deterministic values per tensor.
    pub fn tensors(&self) -> Vec<F32Tensor> {
        let prefix = &self.prefix;
        let hidden = self.hidden_size;
        let intermediate = self.intermediate_size;
        let mut seed = 0.0f32;
        let mut next_seed = || {
            seed += 1.0;
            seed
        };
        let mut tensors = vec![
            F32Tensor::ramp(
                &format!("{prefix}embeddings.word_embeddings.weight"),
                &[self.vocab_size, hidden],
                next_seed(),
            ),
            F32Tensor::ramp(
                &format!("{prefix}embeddings.position_embeddings.weight"),
                &[self.max_position_embeddings, hidden],
                next_seed(),
            ),
            F32Tensor::ramp(
                &format!("{prefix}embeddings.LayerNorm.weight"),
                &[hidden],
                next_seed(),
            ),
            F32Tensor::ramp(
                &format!("{prefix}embeddings.LayerNorm.bias"),
                &[hidden],
                next_seed(),
            ),
        ];
        if let Some(type_vocab_size) = self.type_vocab_size {
            tensors.push(F32Tensor::ramp(
                &format!("{prefix}embeddings.token_type_embeddings.weight"),
                &[type_vocab_size, hidden],
                next_seed(),
            ));
        }

        for layer in 0..self.num_layers {
            let layer_prefix = format!("{prefix}{}{layer}.", self.names.layer_stack_prefix);
            for projection in [
                self.names.query,
                self.names.key,
                self.names.value,
                self.names.attention_output,
            ] {
                tensors.push(F32Tensor::ramp(
                    &format!("{layer_prefix}{projection}.weight"),
                    &[hidden, hidden],
                    next_seed(),
                ));
                tensors.push(F32Tensor::ramp(
                    &format!("{layer_prefix}{projection}.bias"),
                    &[hidden],
                    next_seed(),
                ));
            }
            for norm in [self.names.attention_norm, self.names.output_norm] {
                tensors.push(F32Tensor::ramp(
                    &format!("{layer_prefix}{norm}.weight"),
                    &[hidden],
                    next_seed(),
                ));
                tensors.push(F32Tensor::ramp(
                    &format!("{layer_prefix}{norm}.bias"),
                    &[hidden],
                    next_seed(),
                ));
            }
            tensors.push(F32Tensor::ramp(
                &format!("{layer_prefix}{}.weight", self.names.intermediate),
                &[intermediate, hidden],
                next_seed(),
            ));
            tensors.push(F32Tensor::ramp(
                &format!("{layer_prefix}{}.bias", self.names.intermediate),
                &[intermediate],
                next_seed(),
            ));
            tensors.push(F32Tensor::ramp(
                &format!("{layer_prefix}{}.weight", self.names.feed_forward_output),
                &[hidden, intermediate],
                next_seed(),
            ));
            tensors.push(F32Tensor::ramp(
                &format!("{layer_prefix}{}.bias", self.names.feed_forward_output),
                &[hidden],
                next_seed(),
            ));
        }

        if self.include_pooler {
            tensors.push(F32Tensor::ramp(
                &format!("{prefix}pooler.dense.weight"),
                &[hidden, hidden],
                next_seed(),
            ));
            tensors.push(F32Tensor::ramp(
                &format!("{prefix}pooler.dense.bias"),
                &[hidden],
                next_seed(),
            ));
        }

        tensors
    }

    /// The fixture serialised as a safetensors byte stream.
    pub fn safetensors(&self) -> Vec<u8> {
        build_safetensors(&self.tensors())
    }
}

/// Fixture builder for a LLaMA-family decoder-only checkpoint.
///
/// Yi, Mistral, StarCoder2, Phi-4, Nemotron and the LLaMA line all ship the same
/// HuggingFace tensor layout — `model.layers.{i}.self_attn.{q,k,v,o}_proj`,
/// `model.layers.{i}.mlp.{gate,up,down}_proj`, two RMS norms per layer and a
/// final `model.norm` — differing only in widths, in whether the projections
/// carry biases, and in whether the MLP is gated. One spec therefore covers all
/// of them, which is what makes a per-model round-trip test cheap enough to be
/// worth having for every architecture.
#[derive(Debug, Clone)]
pub struct DecoderFixtureSpec {
    /// Prefix the checkpoint nests the backbone under, e.g. `"model."` or `""`.
    pub prefix: String,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Residual-stream width.
    pub hidden_size: usize,
    /// Feed-forward inner width.
    pub intermediate_size: usize,
    /// Number of decoder layers.
    pub num_layers: usize,
    /// Output width of `q_proj` (`num_attention_heads * head_dim`).
    pub q_width: usize,
    /// Output width of `k_proj` / `v_proj` (`num_key_value_heads * head_dim`).
    pub kv_width: usize,
    /// Whether the attention projections carry biases.
    pub attention_bias: bool,
    /// Whether the MLP projections carry biases.
    pub mlp_bias: bool,
    /// Whether the MLP is gated (`gate_proj` + `up_proj` + `down_proj`) rather
    /// than a plain two-layer FFN (`fc1` + `fc2`).
    pub gated_mlp: bool,
    /// Whether each layer carries a `post_attention_layernorm` (a parallel
    /// architecture such as Phi-2 has only `input_layernorm`).
    pub post_attention_norm: bool,
    /// Whether the norms carry biases (LayerNorm) rather than being RMS norms.
    pub norm_bias: bool,
    /// Whether to emit a standalone `lm_head.weight`.
    pub include_lm_head: bool,
    /// Spelling of the attention output projection (`o_proj` or `dense`).
    pub attention_output_name: String,
}

impl DecoderFixtureSpec {
    /// A LLaMA-style (SwiGLU, RMS norm, no biases) spec.
    pub fn llama_style(
        prefix: &str,
        vocab_size: usize,
        hidden_size: usize,
        intermediate_size: usize,
        num_layers: usize,
        q_width: usize,
        kv_width: usize,
    ) -> Self {
        Self {
            prefix: prefix.to_string(),
            vocab_size,
            hidden_size,
            intermediate_size,
            num_layers,
            q_width,
            kv_width,
            attention_bias: false,
            mlp_bias: false,
            gated_mlp: true,
            post_attention_norm: true,
            norm_bias: false,
            include_lm_head: false,
            attention_output_name: "o_proj".to_string(),
        }
    }

    /// Every tensor a decoder of this shape expects, each with a distinct
    /// deterministic ramp so a test can tell two parameters apart.
    pub fn tensors(&self) -> Vec<F32Tensor> {
        let prefix = &self.prefix;
        let hidden = self.hidden_size;
        let intermediate = self.intermediate_size;
        let mut seed = 0.0f32;
        let mut next_seed = || {
            seed += 1.0;
            seed
        };

        let mut tensors = vec![F32Tensor::ramp(
            &format!("{prefix}embed_tokens.weight"),
            &[self.vocab_size, hidden],
            next_seed(),
        )];

        let push_linear = |tensors: &mut Vec<F32Tensor>,
                           name: String,
                           out: usize,
                           inp: usize,
                           bias: bool,
                           s: f32| {
            tensors.push(F32Tensor::ramp(&format!("{name}.weight"), &[out, inp], s));
            if bias {
                tensors.push(F32Tensor::ramp(&format!("{name}.bias"), &[out], s + 0.25));
            }
        };

        for layer in 0..self.num_layers {
            let attn = format!("{prefix}layers.{layer}.self_attn");
            let seeds = [
                next_seed(),
                next_seed(),
                next_seed(),
                next_seed(),
                next_seed(),
                next_seed(),
                next_seed(),
            ];
            push_linear(
                &mut tensors,
                format!("{attn}.q_proj"),
                self.q_width,
                hidden,
                self.attention_bias,
                seeds[0],
            );
            push_linear(
                &mut tensors,
                format!("{attn}.k_proj"),
                self.kv_width,
                hidden,
                self.attention_bias,
                seeds[1],
            );
            push_linear(
                &mut tensors,
                format!("{attn}.v_proj"),
                self.kv_width,
                hidden,
                self.attention_bias,
                seeds[2],
            );
            push_linear(
                &mut tensors,
                format!("{attn}.{}", self.attention_output_name),
                hidden,
                self.q_width,
                self.attention_bias,
                seeds[3],
            );

            let mlp = format!("{prefix}layers.{layer}.mlp");
            if self.gated_mlp {
                push_linear(
                    &mut tensors,
                    format!("{mlp}.gate_proj"),
                    intermediate,
                    hidden,
                    self.mlp_bias,
                    seeds[4],
                );
                push_linear(
                    &mut tensors,
                    format!("{mlp}.up_proj"),
                    intermediate,
                    hidden,
                    self.mlp_bias,
                    seeds[5],
                );
                push_linear(
                    &mut tensors,
                    format!("{mlp}.down_proj"),
                    hidden,
                    intermediate,
                    self.mlp_bias,
                    seeds[6],
                );
            } else {
                push_linear(
                    &mut tensors,
                    format!("{mlp}.fc1"),
                    intermediate,
                    hidden,
                    self.mlp_bias,
                    seeds[4],
                );
                push_linear(
                    &mut tensors,
                    format!("{mlp}.fc2"),
                    hidden,
                    intermediate,
                    self.mlp_bias,
                    seeds[5],
                );
            }

            let input_norm = format!("{prefix}layers.{layer}.input_layernorm");
            tensors.push(F32Tensor::ramp(
                &format!("{input_norm}.weight"),
                &[hidden],
                next_seed(),
            ));
            if self.norm_bias {
                tensors.push(F32Tensor::ramp(
                    &format!("{input_norm}.bias"),
                    &[hidden],
                    next_seed(),
                ));
            }
            if self.post_attention_norm {
                let post_norm = format!("{prefix}layers.{layer}.post_attention_layernorm");
                tensors.push(F32Tensor::ramp(
                    &format!("{post_norm}.weight"),
                    &[hidden],
                    next_seed(),
                ));
                if self.norm_bias {
                    tensors.push(F32Tensor::ramp(
                        &format!("{post_norm}.bias"),
                        &[hidden],
                        next_seed(),
                    ));
                }
            }
        }

        let final_norm = format!("{prefix}norm");
        tensors.push(F32Tensor::ramp(
            &format!("{final_norm}.weight"),
            &[hidden],
            next_seed(),
        ));
        if self.norm_bias {
            tensors.push(F32Tensor::ramp(
                &format!("{final_norm}.bias"),
                &[hidden],
                next_seed(),
            ));
        }

        if self.include_lm_head {
            tensors.push(F32Tensor::ramp(
                "lm_head.weight",
                &[self.vocab_size, hidden],
                next_seed(),
            ));
        }

        tensors
    }

    /// The fixture serialised as a safetensors byte stream.
    pub fn safetensors(&self) -> Vec<u8> {
        build_safetensors(&self.tensors())
    }
}
