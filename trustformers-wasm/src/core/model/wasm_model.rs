//! [`WasmModel`]: real weight loading and real forward passes.
//!
//! ## What changed from the earlier "demo" implementation
//!
//! Previously every architecture's forward pass ignored `self.weights`
//! entirely and returned `WasmTensor::randn(...)` — pure noise dressed up
//! as inference output. `load_weights` also silently substituted freshly
//! randomized weights (`create_dummy_weights`) whenever real parsing
//! failed, logging a "✅ Successfully loaded" message regardless. Both are
//! gone: a forward pass now performs real matrix multiplications, softmax
//! attention, and normalization over tensors read out of `self.weights`
//! (see `forward.rs` for the math and `weights.rs` for the named lookup),
//! and any load failure — unknown format, corrupt file, a missing required
//! tensor — is returned as a structured `Err`, never masked.
//!
//! ## Checkpoint layout
//!
//! See `forward.rs`'s module docs: this crate defines its own SafeTensors
//! tensor-naming convention rather than mirroring any specific upstream
//! checkpoint's names byte-for-byte. `layers.<i>.attn.{q,k,v,o}_proj.{weight,bias}`,
//! `layers.<i>.norm{1,2}.{weight,bias}`, `layers.<i>.ffn.{fc1,fc2}.{weight,bias}`
//! (GELU-FFN architectures) or `layers.<i>.ffn.{gate,up,down}_proj.weight`
//! (SwiGLU architectures), `token_embeddings.weight`,
//! `position_embeddings.weight` (absolute-position architectures only),
//! `final_norm.{weight,bias}`, and an optional `lm_head.weight` (falls back
//! to tied embeddings when absent).

use super::config::{ModelArchitecture, ModelConfig, ModelFormat};
use super::formats::ModelFormatManager;
use super::forward::{
    embedding_lookup, layer_norm, linear, rms_norm, transformer_block, transpose, FfnWeights,
    LayerWeights, NormKind, TransformerConfig,
};
use super::weights::{layer_prefix, NamedWeights};
use crate::core::tensor::WasmTensor;
use std::collections::HashMap;
use std::format;
use std::string::{String, ToString};
use std::vec::Vec;
use wasm_bindgen::prelude::*;

/// WebAssembly-compatible model for inference.
#[wasm_bindgen]
pub struct WasmModel {
    config: ModelConfig,
    weights: NamedWeights,
    initialized: bool,
    format_manager: ModelFormatManager,
    model_format: Option<ModelFormat>,
    model_metadata: HashMap<String, String>,
}

#[wasm_bindgen]
impl WasmModel {
    /// Create a new, unloaded model with the given configuration.
    #[wasm_bindgen(constructor)]
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            weights: NamedWeights::new(),
            initialized: false,
            format_manager: ModelFormatManager::new(),
            model_format: None,
            model_metadata: HashMap::new(),
        }
    }

    /// Load model weights from binary data with automatic format detection.
    ///
    /// On any failure (unrecognized format, corrupt data, a parser that
    /// cannot recover real weights for this format) this returns `Err` —
    /// it never substitutes randomized placeholder weights.
    pub async fn load_weights(&mut self, weights_data: &[u8]) -> Result<(), JsValue> {
        self.load_weights_inner(weights_data, None).map_err(|e| JsValue::from_str(&e))
    }

    /// Load model weights with an explicit format, bypassing detection.
    pub async fn load_weights_with_format(
        &mut self,
        weights_data: &[u8],
        format: ModelFormat,
    ) -> Result<(), JsValue> {
        self.load_weights_inner(weights_data, Some(format))
            .map_err(|e| JsValue::from_str(&e))
    }

    /// Fetch model weights from a URL and load them.
    ///
    /// This performs a real `fetch()` (via `web_sys`) and requires a
    /// browser `window` — it is not meaningful outside a browser main
    /// thread and cannot be exercised by native unit tests, which is why
    /// this file's `#[cfg(test)]` module tests the parsing/forward-pass
    /// logic directly instead.
    pub async fn load_from_url(&mut self, url: &str) -> Result<(), JsValue> {
        use wasm_bindgen::JsCast;

        let window = web_sys::window().ok_or_else(|| {
            JsValue::from_str("load_from_url: no `window` object available in this context")
        })?;

        let response_value =
            wasm_bindgen_futures::JsFuture::from(window.fetch_with_str(url)).await.map_err(
                |e| JsValue::from_str(&format!("load_from_url: fetch('{url}') failed: {e:?}")),
            )?;
        let response: web_sys::Response = response_value
            .dyn_into()
            .map_err(|_| JsValue::from_str("load_from_url: fetch did not resolve to a Response"))?;

        if !response.ok() {
            return Err(JsValue::from_str(&format!(
                "load_from_url: {url} returned HTTP {}",
                response.status()
            )));
        }

        let buffer_value = wasm_bindgen_futures::JsFuture::from(
            response.array_buffer().map_err(|e| JsValue::from_str(&format!("{e:?}")))?,
        )
        .await
        .map_err(|e| {
            JsValue::from_str(&format!(
                "load_from_url: reading response body failed: {e:?}"
            ))
        })?;
        let array_buffer: js_sys::ArrayBuffer = buffer_value.dyn_into().map_err(|_| {
            JsValue::from_str("load_from_url: response body was not an ArrayBuffer")
        })?;

        let view = js_sys::Uint8Array::new(&array_buffer);
        let mut bytes = std::vec![0u8; view.length() as usize];
        view.copy_to(&mut bytes);

        self.load_weights(&bytes).await
    }

    /// Run a real forward pass over the loaded weights.
    pub fn forward(&self, input_ids: &WasmTensor) -> Result<WasmTensor, JsValue> {
        let (data, shape) = self.forward_inner(input_ids).map_err(|e| JsValue::from_str(&e))?;
        WasmTensor::new(data, shape)
    }

    /// Get model configuration.
    #[wasm_bindgen(getter)]
    pub fn config(&self) -> ModelConfig {
        self.config.clone()
    }

    /// Check if model weights are loaded.
    #[wasm_bindgen(getter)]
    pub fn initialized(&self) -> bool {
        self.initialized
    }

    /// Get memory usage in MB, computed from the real loaded weight sizes.
    pub fn memory_usage_mb(&self) -> f32 {
        (self.weights.total_parameters() * 4) as f32 / 1_048_576.0
    }

    /// Get detected model format.
    pub fn get_model_format(&self) -> Option<ModelFormat> {
        self.model_format
    }

    /// Get model metadata as a JavaScript object.
    pub fn get_model_metadata(&self) -> js_sys::Object {
        let metadata_obj = js_sys::Object::new();
        for (key, value) in &self.model_metadata {
            let _ = js_sys::Reflect::set(&metadata_obj, &key.into(), &value.into());
        }
        if let Some(format) = self.model_format {
            let _ = js_sys::Reflect::set(
                &metadata_obj,
                &"format".into(),
                &format!("{format:?}").into(),
            );
        }
        let _ = js_sys::Reflect::set(
            &metadata_obj,
            &"weight_count".into(),
            &self.weights.len().into(),
        );
        let _ = js_sys::Reflect::set(
            &metadata_obj,
            &"memory_usage_mb".into(),
            &self.memory_usage_mb().into(),
        );
        let _ = js_sys::Reflect::set(
            &metadata_obj,
            &"architecture".into(),
            &format!("{arch:?}", arch = self.config.architecture).into(),
        );
        metadata_obj
    }

    /// Whether this crate's inference backends can hardware-accelerate the
    /// detected format (informational only — has no bearing on whether
    /// weight loading itself is supported; see [`ModelFormat`] docs).
    pub fn supports_hardware_acceleration(&self) -> bool {
        matches!(
            self.model_format,
            Some(ModelFormat::TensorRT)
                | Some(ModelFormat::CoreML)
                | Some(ModelFormat::TensorFlowLite)
                | Some(ModelFormat::Onnx)
        )
    }

    /// Get supported model *file* formats (detection); note only
    /// SafeTensors can currently have its weights loaded.
    #[wasm_bindgen(js_name = getSupportedFormats)]
    pub fn get_supported_formats() -> js_sys::Array {
        let formats = js_sys::Array::new();
        for f in [
            "ONNX",
            "GGUF",
            "SafeTensors",
            "TensorRT",
            "CoreML",
            "TensorFlowLite",
            "TorchScript",
            "CustomBinary",
        ] {
            formats.push(&f.into());
        }
        formats
    }

    /// Detect format from binary data without loading the model.
    #[wasm_bindgen(js_name = detectFormat)]
    pub fn detect_format_static(data: &[u8]) -> js_sys::Object {
        let manager = ModelFormatManager::new();
        let result_obj = js_sys::Object::new();

        if let Some(detection) = manager.detect_format(data) {
            let _ = js_sys::Reflect::set(
                &result_obj,
                &"format".into(),
                &format!("{format:?}", format = detection.format).into(),
            );
            let _ = js_sys::Reflect::set(
                &result_obj,
                &"confidence".into(),
                &detection.confidence.into(),
            );

            let metadata_obj = js_sys::Object::new();
            for (key, value) in detection.metadata {
                let _ = js_sys::Reflect::set(&metadata_obj, &key.into(), &value.into());
            }
            let _ = js_sys::Reflect::set(&result_obj, &"metadata".into(), &metadata_obj.into());
            let _ = js_sys::Reflect::set(&result_obj, &"supported".into(), &true.into());
        } else {
            let _ = js_sys::Reflect::set(&result_obj, &"format".into(), &"Unknown".into());
            let _ = js_sys::Reflect::set(&result_obj, &"confidence".into(), &0.0.into());
            let _ = js_sys::Reflect::set(&result_obj, &"supported".into(), &false.into());
        }

        result_obj
    }
}

/// Architecture-specific forward-pass shape, independent of any particular
/// [`ModelConfig`] instance's sizes.
struct ArchSpec {
    causal: bool,
    norm_kind: NormKind,
    use_rope: bool,
    use_position_embeddings: bool,
    use_swiglu: bool,
    project_to_vocab: bool,
}

impl ArchSpec {
    const BERT: ArchSpec = ArchSpec {
        causal: false,
        norm_kind: NormKind::LayerNorm,
        use_rope: false,
        use_position_embeddings: true,
        use_swiglu: false,
        project_to_vocab: false,
    };
    const GPT2: ArchSpec = ArchSpec {
        causal: true,
        norm_kind: NormKind::LayerNorm,
        use_rope: false,
        use_position_embeddings: true,
        use_swiglu: false,
        project_to_vocab: true,
    };
    /// Simplified encoder-style stack using T5's actual normalization
    /// convention (RMSNorm, no absolute position embeddings). Does *not*
    /// implement T5's relative-position attention bias or the true
    /// encoder/decoder split with cross-attention — see module docs.
    const T5: ArchSpec = ArchSpec {
        causal: false,
        norm_kind: NormKind::RmsNorm,
        use_rope: false,
        use_position_embeddings: false,
        use_swiglu: false,
        project_to_vocab: true,
    };
    const LLAMA: ArchSpec = ArchSpec {
        causal: true,
        norm_kind: NormKind::RmsNorm,
        use_rope: true,
        use_position_embeddings: false,
        use_swiglu: true,
        project_to_vocab: true,
    };
    /// Structurally identical to LLaMA here; sliding-window attention is
    /// not implemented (full causal attention is used instead).
    const MISTRAL: ArchSpec = ArchSpec {
        causal: true,
        norm_kind: NormKind::RmsNorm,
        use_rope: true,
        use_position_embeddings: false,
        use_swiglu: true,
        project_to_vocab: true,
    };
}

impl WasmModel {
    /// Pure (`JsValue`-free) core shared by [`WasmModel::load_weights`] and
    /// [`WasmModel::load_weights_with_format`]. `format` of `None` means
    /// "auto-detect"; `Some(f)` bypasses detection entirely. On any
    /// failure the model's existing state is left untouched (no partial
    /// update, no placeholder weights).
    fn load_weights_inner(
        &mut self,
        weights_data: &[u8],
        format: Option<ModelFormat>,
    ) -> Result<(), String> {
        let (target_format, metadata) = match format {
            Some(f) => (f, None),
            None => {
                let detection =
                    self.format_manager.detect_format(weights_data).ok_or_else(|| {
                        "load_weights: could not detect a supported model format (expected \
                     SafeTensors; TensorRT/CoreML/TensorFlowLite are detected but their weights \
                     cannot be loaded in this build)"
                            .to_string()
                    })?;
                (detection.format, Some(detection.metadata))
            },
        };

        let loaded = self.format_manager.load_model(weights_data, Some(target_format))?;

        self.model_format = Some(target_format);
        if let Some(metadata) = metadata {
            self.model_metadata = metadata;
        }
        self.weights = NamedWeights::from_pairs(loaded);
        self.initialized = true;
        Ok(())
    }

    /// Pure (`JsValue`-free) core of [`WasmModel::forward`]: validates
    /// state/shape and runs the real transformer math, returning a plain
    /// `String` error on any failure. Kept separate from the
    /// `#[wasm_bindgen]`-exported `forward` so error paths — "not
    /// initialized", "missing weight tensor", "sequence too long" — are
    /// exercised by native tests. Constructing a `JsValue` (even a plain
    /// `JsValue::from_str`) unconditionally panics on non-wasm32 targets,
    /// so any `Result<_, JsValue>`-returning function's error arm is
    /// fundamentally untestable natively; routing every error through
    /// `String` first and converting to `JsValue` only in `forward` keeps
    /// the real logic testable.
    fn forward_inner(&self, input_ids: &WasmTensor) -> Result<(Vec<f32>, Vec<usize>), String> {
        if !self.initialized {
            return Err(
                "forward: model has no weights loaded (call load_weights first)".to_string(),
            );
        }

        let spec = match self.config.architecture {
            ModelArchitecture::Bert => ArchSpec::BERT,
            ModelArchitecture::GPT2 => ArchSpec::GPT2,
            ModelArchitecture::T5 => ArchSpec::T5,
            ModelArchitecture::Llama => ArchSpec::LLAMA,
            ModelArchitecture::Mistral => ArchSpec::MISTRAL,
        };

        let shape = input_ids.shape();
        if shape.len() != 2 || shape[0] != 1 {
            return Err(format!(
                "forward: expected input_ids shape [1, seq_len], got {shape:?} (batch_size > 1 \
                 is not yet supported)"
            ));
        }
        let ids: Vec<u32> = input_ids.data().iter().map(|&v| v.round().max(0.0) as u32).collect();

        run_transformer(&ids, &self.weights, &self.config, &spec)
            .map_err(|e| format!("forward: {e}"))
    }
}

/// Real transformer forward pass: embeddings -> N transformer blocks ->
/// final norm -> optional vocabulary projection. Pure function over
/// `&NamedWeights`/`&ModelConfig` — no `wasm_bindgen`/`JsValue` involved,
/// so it is exercised directly by the native tests below.
fn run_transformer(
    input_ids: &[u32],
    weights: &NamedWeights,
    config: &ModelConfig,
    spec: &ArchSpec,
) -> Result<(Vec<f32>, Vec<usize>), String> {
    let seq_len = input_ids.len();
    if seq_len == 0 {
        return Err("input_ids must not be empty".to_string());
    }
    if seq_len > config.max_position_embeddings {
        return Err(format!(
            "sequence length {seq_len} exceeds this model's max_position_embeddings ({})",
            config.max_position_embeddings
        ));
    }

    let hidden = config.hidden_size;
    let token_table = weights.required("token_embeddings.weight")?;
    let mut hidden_states = embedding_lookup(input_ids, token_table, config.vocab_size, hidden)?;

    if spec.use_position_embeddings {
        let pos_table = weights.required("position_embeddings.weight")?;
        let pos_rows = pos_table.len() / hidden.max(1);
        if pos_rows < seq_len {
            return Err(format!(
                "position_embeddings.weight only covers {pos_rows} positions, need {seq_len}"
            ));
        }
        for pos in 0..seq_len {
            for d in 0..hidden {
                hidden_states[pos * hidden + d] += pos_table[pos * hidden + d];
            }
        }
    }

    for layer_idx in 0..config.num_layers {
        let prefix = layer_prefix(layer_idx);
        let q_w = weights.required(&format!("{prefix}attn.q_proj.weight"))?;
        let k_w = weights.required(&format!("{prefix}attn.k_proj.weight"))?;
        let v_w = weights.required(&format!("{prefix}attn.v_proj.weight"))?;
        let o_w = weights.required(&format!("{prefix}attn.o_proj.weight"))?;
        let q_b = weights.optional(&format!("{prefix}attn.q_proj.bias"));
        let k_b = weights.optional(&format!("{prefix}attn.k_proj.bias"));
        let v_b = weights.optional(&format!("{prefix}attn.v_proj.bias"));
        let o_b = weights.optional(&format!("{prefix}attn.o_proj.bias"));
        let norm1_w = weights.required(&format!("{prefix}norm1.weight"))?;
        let norm2_w = weights.required(&format!("{prefix}norm2.weight"))?;
        let norm1_b = weights.optional(&format!("{prefix}norm1.bias"));
        let norm2_b = weights.optional(&format!("{prefix}norm2.bias"));

        let ffn = if spec.use_swiglu {
            FfnWeights::SwiGlu {
                gate_w: weights.required(&format!("{prefix}ffn.gate_proj.weight"))?,
                up_w: weights.required(&format!("{prefix}ffn.up_proj.weight"))?,
                down_w: weights.required(&format!("{prefix}ffn.down_proj.weight"))?,
            }
        } else {
            FfnWeights::Gelu {
                fc1_w: weights.required(&format!("{prefix}ffn.fc1.weight"))?,
                fc1_b: weights.optional(&format!("{prefix}ffn.fc1.bias")),
                fc2_w: weights.required(&format!("{prefix}ffn.fc2.weight"))?,
                fc2_b: weights.optional(&format!("{prefix}ffn.fc2.bias")),
            }
        };

        let layer_weights = LayerWeights {
            q_w,
            q_b,
            k_w,
            k_b,
            v_w,
            v_b,
            o_w,
            o_b,
            norm1_w,
            norm1_b,
            norm2_w,
            norm2_b,
            ffn,
        };
        let block_cfg = TransformerConfig {
            hidden_size: hidden,
            num_heads: config.num_heads,
            causal: spec.causal,
            norm_kind: spec.norm_kind,
            use_rope: spec.use_rope,
            eps: 1e-5,
            rope_base: 10000.0,
        };

        hidden_states = transformer_block(&hidden_states, seq_len, &layer_weights, &block_cfg)
            .map_err(|e| format!("layer {layer_idx}: {e}"))?;
    }

    let final_norm_w = weights.required("final_norm.weight")?;
    let final_norm_b = weights.optional("final_norm.bias");
    hidden_states = match spec.norm_kind {
        NormKind::LayerNorm => layer_norm(
            &hidden_states,
            seq_len,
            hidden,
            final_norm_w,
            final_norm_b,
            1e-5,
        )?,
        NormKind::RmsNorm => rms_norm(&hidden_states, seq_len, hidden, final_norm_w, 1e-5)?,
    };

    if !spec.project_to_vocab {
        return Ok((hidden_states, std::vec![1, seq_len, hidden]));
    }

    let vocab = config.vocab_size;
    let logits = if let Ok(lm_head) = weights.required("lm_head.weight") {
        linear(&hidden_states, seq_len, hidden, lm_head, vocab, None)?
    } else {
        // Tied input/output embeddings (as GPT-2 and many decoder-only
        // models actually do): logits = hidden @ token_embeddings^T.
        let tied = transpose(token_table, vocab, hidden);
        linear(&hidden_states, seq_len, hidden, &tied, vocab, None)?
    };
    Ok((logits, std::vec![1, seq_len, vocab]))
}

/// Quantized model for efficient inference.
#[wasm_bindgen]
pub struct QuantizedModel {
    base_model: WasmModel,
    quantization_type: QuantizationType,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    Int8,
    Int4,
    Dynamic,
}

#[wasm_bindgen]
impl QuantizedModel {
    /// Create a quantized model from a base model.
    pub fn from_model(model: WasmModel, quantization_type: QuantizationType) -> Self {
        Self {
            base_model: model,
            quantization_type,
        }
    }

    /// Run inference through the (currently un-quantized) base model.
    ///
    /// Actual weight quantization is not implemented; this intentionally
    /// does not claim otherwise via a fabricated "quantize" no-op — the
    /// forward pass always runs at full precision over the real weights.
    pub fn forward(&self, input_ids: &WasmTensor) -> Result<WasmTensor, JsValue> {
        self.base_model.forward(input_ids)
    }

    /// Get theoretical memory savings for the configured quantization type
    /// (informational only — no quantization is actually applied yet).
    pub fn memory_savings_percent(&self) -> f32 {
        match self.quantization_type {
            QuantizationType::Int8 => 75.0,
            QuantizationType::Int4 => 87.5,
            QuantizationType::Dynamic => 50.0,
        }
    }
}

#[cfg(test)]
impl WasmModel {
    /// Test-only constructor that installs weights directly, bypassing
    /// format detection/parsing entirely.
    pub(crate) fn with_weights_for_test(config: ModelConfig, weights: NamedWeights) -> Self {
        Self {
            config,
            weights,
            initialized: true,
            format_manager: ModelFormatManager::new(),
            model_format: None,
            model_metadata: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::model::config::ModelArchitecture;

    fn tiny_config(architecture: ModelArchitecture) -> ModelConfig {
        // Deliberately tiny: real bert_base()/llama_7b() configs allocate
        // hundreds of MB of weights, which is both slow and pointless for
        // a unit test that only needs to prove the math is real.
        ModelConfig {
            architecture,
            vocab_size: 12,
            hidden_size: 8,
            num_layers: 2,
            num_heads: 2,
            max_position_embeddings: 16,
            intermediate_size: 10,
            hidden_dropout_prob: 0.0,
            attention_dropout_prob: 0.0,
        }
    }

    fn tensor(data: Vec<f32>, shape: Vec<usize>) -> WasmTensor {
        WasmTensor::new(data, shape).expect("valid tensor")
    }

    /// Deterministic pseudo-random f32 generator (no external RNG dep
    /// needed — just enough spread to make matmuls non-degenerate).
    fn fill(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                ((s >> 8) as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn build_weights(
        config: &ModelConfig,
        use_swiglu: bool,
        use_pos_emb: bool,
        use_bias: bool,
    ) -> NamedWeights {
        let mut w = NamedWeights::new();
        let h = config.hidden_size;
        let inter = config.intermediate_size;
        let mut seed = 7u32;
        let mut next = |n: usize| {
            seed = seed.wrapping_add(101);
            fill(n, seed)
        };

        w.insert(
            "token_embeddings.weight",
            tensor(next(config.vocab_size * h), std::vec![config.vocab_size, h]),
        );
        if use_pos_emb {
            w.insert(
                "position_embeddings.weight",
                tensor(
                    next(config.max_position_embeddings * h),
                    std::vec![config.max_position_embeddings, h],
                ),
            );
        }

        for i in 0..config.num_layers {
            let p = layer_prefix(i);
            for name in ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj"] {
                w.insert(
                    format!("{p}{name}.weight"),
                    tensor(next(h * h), std::vec![h, h]),
                );
                if use_bias {
                    w.insert(format!("{p}{name}.bias"), tensor(next(h), std::vec![h]));
                }
            }
            w.insert(
                format!("{p}norm1.weight"),
                tensor(std::vec![1.0; h], std::vec![h]),
            );
            w.insert(
                format!("{p}norm2.weight"),
                tensor(std::vec![1.0; h], std::vec![h]),
            );
            if use_bias {
                w.insert(
                    format!("{p}norm1.bias"),
                    tensor(std::vec![0.0; h], std::vec![h]),
                );
                w.insert(
                    format!("{p}norm2.bias"),
                    tensor(std::vec![0.0; h], std::vec![h]),
                );
            }

            if use_swiglu {
                w.insert(
                    format!("{p}ffn.gate_proj.weight"),
                    tensor(next(h * inter), std::vec![h, inter]),
                );
                w.insert(
                    format!("{p}ffn.up_proj.weight"),
                    tensor(next(h * inter), std::vec![h, inter]),
                );
                w.insert(
                    format!("{p}ffn.down_proj.weight"),
                    tensor(next(inter * h), std::vec![inter, h]),
                );
            } else {
                w.insert(
                    format!("{p}ffn.fc1.weight"),
                    tensor(next(h * inter), std::vec![h, inter]),
                );
                w.insert(
                    format!("{p}ffn.fc2.weight"),
                    tensor(next(inter * h), std::vec![inter, h]),
                );
                if use_bias {
                    w.insert(
                        format!("{p}ffn.fc1.bias"),
                        tensor(next(inter), std::vec![inter]),
                    );
                    w.insert(format!("{p}ffn.fc2.bias"), tensor(next(h), std::vec![h]));
                }
            }
        }

        w.insert("final_norm.weight", tensor(std::vec![1.0; h], std::vec![h]));
        w
    }

    #[test]
    fn test_forward_without_weights_errors() {
        let config = tiny_config(ModelArchitecture::Bert);
        let model = WasmModel::new(config);
        assert!(!model.initialized());
        let input = tensor(std::vec![1.0, 2.0, 3.0], std::vec![1, 3]);
        // `forward_inner` (not `forward`) here: constructing a `JsValue`
        // (even a bare `JsValue::from_str`) unconditionally panics on
        // non-wasm32 targets, so an error path can only be asserted via
        // the `String`-returning inner function — see its doc comment.
        let err = model
            .forward_inner(&input)
            .expect_err("unloaded model must error, not return noise");
        assert!(err.contains("no weights"));
    }

    #[test]
    fn test_bert_forward_returns_hidden_states_shape() {
        let config = tiny_config(ModelArchitecture::Bert);
        let weights = build_weights(&config, false, true, true);
        let model = WasmModel::with_weights_for_test(config.clone(), weights);

        let input = tensor(std::vec![1.0, 2.0, 3.0, 4.0], std::vec![1, 4]);
        let out = model.forward(&input).expect("real weights must produce real output");
        assert_eq!(out.shape(), std::vec![1, 4, config.hidden_size]);
        assert!(out.data().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_gpt2_forward_returns_vocab_logits_shape() {
        let config = tiny_config(ModelArchitecture::GPT2);
        let weights = build_weights(&config, false, true, true);
        let model = WasmModel::with_weights_for_test(config.clone(), weights);

        let input = tensor(std::vec![0.0, 1.0, 2.0], std::vec![1, 3]);
        let out = model.forward(&input).expect("real weights must produce real output");
        assert_eq!(out.shape(), std::vec![1, 3, config.vocab_size]);
    }

    #[test]
    fn test_llama_forward_swiglu_rope_finite() {
        let config = tiny_config(ModelArchitecture::Llama);
        let weights = build_weights(&config, true, false, false);
        let model = WasmModel::with_weights_for_test(config.clone(), weights);

        let input = tensor(std::vec![1.0, 3.0, 5.0], std::vec![1, 3]);
        let out = model.forward(&input).expect("real weights must produce real output");
        assert_eq!(out.shape(), std::vec![1, 3, config.vocab_size]);
        assert!(out.data().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_mistral_forward_shape() {
        let config = tiny_config(ModelArchitecture::Mistral);
        let weights = build_weights(&config, true, false, false);
        let model = WasmModel::with_weights_for_test(config.clone(), weights);
        let input = tensor(std::vec![2.0, 4.0], std::vec![1, 2]);
        let out = model.forward(&input).expect("real weights must produce real output");
        assert_eq!(out.shape(), std::vec![1, 2, config.vocab_size]);
    }

    #[test]
    fn test_t5_forward_shape() {
        let config = tiny_config(ModelArchitecture::T5);
        let weights = build_weights(&config, false, false, true);
        let model = WasmModel::with_weights_for_test(config.clone(), weights);
        let input = tensor(std::vec![1.0, 1.0, 2.0], std::vec![1, 3]);
        let out = model.forward(&input).expect("real weights must produce real output");
        assert_eq!(out.shape(), std::vec![1, 3, config.vocab_size]);
    }

    #[test]
    fn test_forward_missing_required_tensor_errors_with_name() {
        let config = tiny_config(ModelArchitecture::Bert);
        // Deliberately empty: no token_embeddings.weight present.
        let model = WasmModel::with_weights_for_test(config, NamedWeights::new());
        let input = tensor(std::vec![1.0], std::vec![1, 1]);
        let err = model.forward_inner(&input).expect_err("missing weights must error");
        assert!(
            err.contains("token_embeddings.weight"),
            "error should name the missing tensor: {err}"
        );
    }

    #[test]
    fn test_forward_output_changes_with_different_input_tokens() {
        // Guards against a regression back to "ignores input, returns
        // constant/random output": two different prompts of equal length
        // must produce different logits.
        let config = tiny_config(ModelArchitecture::GPT2);
        let weights = build_weights(&config, false, true, true);
        let model = WasmModel::with_weights_for_test(config, weights);

        let out_a = model.forward(&tensor(std::vec![1.0, 2.0, 3.0], std::vec![1, 3])).unwrap();
        let out_b = model.forward(&tensor(std::vec![5.0, 6.0, 7.0], std::vec![1, 3])).unwrap();
        assert_ne!(out_a.data(), out_b.data());
    }

    #[test]
    fn test_forward_is_deterministic() {
        let config = tiny_config(ModelArchitecture::Bert);
        let weights = build_weights(&config, false, true, true);
        let model = WasmModel::with_weights_for_test(config, weights);
        let input = tensor(std::vec![1.0, 2.0, 3.0], std::vec![1, 3]);
        let out1 = model.forward(&input).unwrap();
        let out2 = model.forward(&input).unwrap();
        assert_eq!(
            out1.data(),
            out2.data(),
            "forward must be deterministic for identical input/weights"
        );
    }

    #[test]
    fn test_forward_rejects_batch_size_greater_than_one() {
        let config = tiny_config(ModelArchitecture::Bert);
        let weights = build_weights(&config, false, true, true);
        let model = WasmModel::with_weights_for_test(config, weights);
        let input = tensor(std::vec![1.0, 2.0, 3.0, 4.0], std::vec![2, 2]);
        assert!(model.forward_inner(&input).is_err());
    }

    #[test]
    fn test_forward_rejects_sequence_longer_than_max_position() {
        let config = tiny_config(ModelArchitecture::Bert);
        let weights = build_weights(&config, false, true, true);
        let model = WasmModel::with_weights_for_test(config.clone(), weights);
        let too_long: Vec<f32> = (0..(config.max_position_embeddings + 1))
            .map(|i| (i % config.vocab_size) as f32)
            .collect();
        let len = too_long.len();
        let input = tensor(too_long, std::vec![1, len]);
        let err = model.forward_inner(&input).expect_err("oversized sequence must error");
        assert!(err.contains("max_position_embeddings"));
    }

    #[test]
    fn test_memory_usage_reflects_real_weight_count() {
        let config = tiny_config(ModelArchitecture::Bert);
        let weights = build_weights(&config, false, true, true);
        let expected_params = weights.total_parameters();
        let model = WasmModel::with_weights_for_test(config, weights);
        let expected_mb = (expected_params * 4) as f32 / 1_048_576.0;
        assert!((model.memory_usage_mb() - expected_mb).abs() < 1e-6);
    }

    #[test]
    fn test_quantized_model_forward_delegates_to_real_base_model() {
        let config = tiny_config(ModelArchitecture::Bert);
        let weights = build_weights(&config, false, true, true);
        let base = WasmModel::with_weights_for_test(config.clone(), weights);
        let quantized = QuantizedModel::from_model(base, QuantizationType::Int8);

        let input = tensor(std::vec![1.0, 2.0], std::vec![1, 2]);
        let out = quantized
            .forward(&input)
            .expect("delegated forward must succeed with real weights");
        assert_eq!(out.shape(), std::vec![1, 2, config.hidden_size]);
    }
}
