//! End-to-end export proof: a real model's real weights survive a real file.
//!
//! Until `Model::named_tensors` was implemented, every exporter in
//! [`trustformers_core::export`] refused to write anything at all — by design,
//! because inventing weights is worse than failing. These tests close that gap
//! from the other side: they build a tiny GPT-2 with *known* weights, push it
//! through the real exporters, read the artifacts back with the real readers, and
//! assert that names, shapes and values match the live model.
//!
//! Nothing here is mocked. The GGUF bytes are parsed by
//! [`trustformers_core::export::gguf_format::read_gguf_file`], and the ONNX bytes
//! by [`trustformers_core::export::onnx_proto::decode_model`] plus the CPU
//! interpreter — the same code paths a downloaded artifact would take.
//!
//! # Regression value
//!
//! Against the pre-`named_tensors` code, every test in this file fails at the
//! export call with "model … exposes no named tensors". Against an implementation
//! that returns *copies* rather than live parameters, the mutation tests fail.

#![cfg(all(
    feature = "gpt2",
    feature = "bert",
    feature = "llama",
    feature = "mistral",
    feature = "t5"
))]

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use trustformers_core::export::gguf_format::read_gguf_file;
use trustformers_core::export::onnx_cpu::{CpuTensor, OnnxGraphExecutor};
use trustformers_core::export::onnx_proto::decode_model;
use trustformers_core::export::{
    ExportConfig, ExportFormat, ExportPrecision, GGUFExporter, ModelExporter, ONNXAttribute,
    ONNXDataType, ONNXDimension, ONNXExporter, ONNXGraph, ONNXNode, ONNXTensor, ONNXTensorShape,
    ONNXTensorType, ONNXTypeInfo, ONNXValueInfo,
};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Model;
use trustformers_models::bert::{BertConfig, BertModel};
use trustformers_models::gpt2::{Gpt2Config, Gpt2LMHeadModel, Gpt2Model};
use trustformers_models::llama::{LlamaConfig, LlamaForCausalLM};
use trustformers_models::mistral::{MistralConfig, MistralForCausalLM};
use trustformers_models::t5::{T5Config, T5ForConditionalGeneration, T5Model};

/// A scratch directory under the OS temp dir, wiped on both ends of the test.
struct ScratchDir {
    path: PathBuf,
}

impl ScratchDir {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("trustformers_export_round_trip_{name}"));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("scratch directory must be creatable");
        Self { path }
    }

    fn join(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// A GPT-2 config small enough to export in milliseconds but structurally
/// complete: two blocks, real attention widths, a non-default inner size.
fn tiny_gpt2_config() -> Gpt2Config {
    Gpt2Config {
        vocab_size: 11,
        n_positions: 8,
        n_embd: 4,
        n_layer: 2,
        n_head: 2,
        n_inner: Some(8),
        activation_function: "gelu".to_string(),
        resid_pdrop: 0.0,
        embd_pdrop: 0.0,
        attn_pdrop: 0.0,
        ..Gpt2Config::default()
    }
}

/// Overwrite every parameter with a value derived from its name and index, so
/// each tensor is unique and any mix-up between two same-shaped tensors shows up
/// as a value mismatch rather than passing silently.
fn install_known_weights<M: Model>(model: &mut M, seed: f32) {
    for (index, (name, parameter)) in model.named_tensors_mut().into_iter().enumerate() {
        let element_count = parameter.len();
        // A per-tensor base drawn from the name keeps same-shaped tensors apart.
        let base = seed + (index as f32) * 17.0 + (name.len() as f32) * 0.5;
        let values: Vec<f32> =
            (0..element_count).map(|offset| base + (offset as f32) * 0.0625).collect();
        let shape = parameter.shape();
        *parameter =
            Tensor::from_vec(values, &shape).expect("regenerated tensor keeps the live shape");
    }
}

/// Read a model's parameters into a plain map for comparison against a file.
fn snapshot<M: Model>(model: &M) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
    model
        .named_tensors()
        .into_iter()
        .map(|(name, tensor)| {
            (
                name,
                (
                    tensor.shape(),
                    tensor.to_vec_f32().expect("test fixtures are all F32"),
                ),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Named-tensor contract
// ---------------------------------------------------------------------------

/// Names must be unique — `collect_model_tensors` rejects duplicates outright,
/// so a clash would make the model unexportable.
#[test]
fn gpt2_named_tensors_are_unique_and_non_empty() {
    let model = Gpt2Model::new(tiny_gpt2_config()).expect("tiny GPT-2 must build");
    let named = model.named_tensors();
    assert!(
        !named.is_empty(),
        "GPT-2 must expose parameters or no exporter can write it"
    );

    let mut seen = HashSet::new();
    for (name, _) in &named {
        assert!(seen.insert(name.clone()), "duplicate tensor name '{name}'");
    }
}

/// Every name GPT-2 publishes is one its own loader binds, and the full
/// HuggingFace name set is present.
#[test]
fn gpt2_named_tensors_use_huggingface_names() {
    let config = tiny_gpt2_config();
    let model = Gpt2Model::new(config.clone()).expect("tiny GPT-2 must build");
    let names: HashSet<String> = model.named_tensors().into_iter().map(|(name, _)| name).collect();

    let mut expected: Vec<String> = vec!["wte.weight".into(), "wpe.weight".into()];
    for index in 0..config.n_layer {
        for leaf in [
            "ln_1.weight",
            "ln_1.bias",
            "attn.c_attn.weight",
            "attn.c_attn.bias",
            "attn.c_proj.weight",
            "attn.c_proj.bias",
            "ln_2.weight",
            "ln_2.bias",
            "mlp.c_fc.weight",
            "mlp.c_fc.bias",
            "mlp.c_proj.weight",
            "mlp.c_proj.bias",
        ] {
            expected.push(format!("h.{index}.{leaf}"));
        }
    }
    expected.push("ln_f.weight".into());
    expected.push("ln_f.bias".into());

    for name in &expected {
        assert!(names.contains(name), "missing HuggingFace name '{name}'");
    }
    assert_eq!(
        names.len(),
        expected.len(),
        "GPT-2 published unexpected extra names: {:?}",
        names.difference(&expected.iter().cloned().collect()).collect::<Vec<_>>()
    );
}

/// The LM-head model re-publishes the backbone under `transformer.` — the prefix
/// HuggingFace's `GPT2LMHeadModel` checkpoints use — plus its own head.
#[test]
fn gpt2_lm_head_names_carry_the_transformer_prefix() {
    let model = Gpt2LMHeadModel::new(tiny_gpt2_config()).expect("tiny GPT-2 LM head must build");
    let names: HashSet<String> = model.named_tensors().into_iter().map(|(name, _)| name).collect();

    assert!(names.contains("transformer.wte.weight"));
    assert!(names.contains("transformer.h.0.attn.c_attn.weight"));
    assert!(names.contains("transformer.ln_f.bias"));
    assert!(names.contains("lm_head.weight"));
    assert!(
        !names.contains("wte.weight"),
        "the backbone must not also appear unprefixed"
    );
}

/// `named_tensors_mut` must hand out the *live* parameters: a write through it
/// has to change what the model computes with, not a discarded copy.
#[test]
fn named_tensors_mut_writes_reach_the_live_model() {
    let mut model = Gpt2Model::new(tiny_gpt2_config()).expect("tiny GPT-2 must build");
    install_known_weights(&mut model, 0.25);

    let written = snapshot(&model);
    let observed = snapshot(&model);
    assert_eq!(
        written, observed,
        "values written through named_tensors_mut must be readable through named_tensors"
    );

    // And a targeted write lands on exactly one tensor.
    for (name, parameter) in model.named_tensors_mut() {
        if name == "ln_f.weight" {
            *parameter = Tensor::from_vec(vec![9.0; parameter.len()], &parameter.shape())
                .expect("shape preserved");
        }
    }
    let after = snapshot(&model);
    assert_eq!(
        after.get("ln_f.weight").map(|(_, values)| values.clone()),
        Some(vec![9.0; 4]),
        "the targeted write must land"
    );
    assert_eq!(
        after.get("wte.weight"),
        written.get("wte.weight"),
        "an unrelated tensor must not change"
    );
}

// ---------------------------------------------------------------------------
// The published names really are the loader's names
// ---------------------------------------------------------------------------

/// Build a real safetensors byte stream from `(name, shape, values)` triples.
///
/// This is a genuine container — `Checkpoint::from_reader` parses it through the
/// same path a downloaded `model.safetensors` takes — not a mocked shortcut.
fn build_safetensors(tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
    use std::collections::BTreeMap;

    // safetensors requires sorted, contiguous data offsets, and the parser
    // validates them, so emit in a stable order.
    let ordered: BTreeMap<&str, &(String, Vec<usize>, Vec<f32>)> =
        tensors.iter().map(|entry| (entry.0.as_str(), entry)).collect();

    let mut header = serde_json::Map::new();
    let mut payload: Vec<u8> = Vec::new();
    for (name, (_, shape, values)) in ordered {
        let start = payload.len();
        for value in values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, payload.len()],
            }),
        );
    }

    let header_bytes =
        serde_json::to_vec(&serde_json::Value::Object(header)).expect("header must serialise");
    let mut bytes = Vec::with_capacity(8 + header_bytes.len() + payload.len());
    bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&header_bytes);
    bytes.extend_from_slice(&payload);
    bytes
}

/// The strongest form of the naming claim: take the names `named_tensors`
/// publishes, write a **real safetensors checkpoint** keyed by exactly those
/// names, load it back through `Model::load_pretrained`, and check every value
/// arrives in the right parameter.
///
/// If `named_tensors` invented a name the loader does not bind, or spelled one
/// differently, `load_pretrained` fails on the missing tensor and this test goes
/// red — which is precisely the failure mode a hand-written name table invites.
///
/// # The Conv1D transpose
///
/// GPT-2's four `Conv1D`-derived projections are stored `[in, out]` on disk and
/// transposed to `[out, in]` on load, so the checkpoint is written with those
/// four names transposed. The rest pass through unchanged.
#[test]
fn gpt2_published_names_load_back_through_load_pretrained() {
    let config = tiny_gpt2_config();
    let mut source = Gpt2Model::new(config.clone()).expect("tiny GPT-2 must build");
    install_known_weights(&mut source, 7.25);
    let expected = snapshot(&source);

    let is_conv1d = |name: &str| {
        name.ends_with("attn.c_attn.weight")
            || name.ends_with("attn.c_proj.weight")
            || name.ends_with("mlp.c_fc.weight")
            || name.ends_with("mlp.c_proj.weight")
    };

    let checkpoint: Vec<(String, Vec<usize>, Vec<f32>)> = expected
        .iter()
        .map(|(name, (shape, values))| {
            if is_conv1d(name) {
                // Transpose [out, in] -> [in, out] to match HuggingFace's layout.
                let (rows, columns) = (shape[0], shape[1]);
                let mut transposed = vec![0.0f32; values.len()];
                for row in 0..rows {
                    for column in 0..columns {
                        transposed[column * rows + row] = values[row * columns + column];
                    }
                }
                (name.clone(), vec![columns, rows], transposed)
            } else {
                (name.clone(), shape.clone(), values.clone())
            }
        })
        .collect();

    let bytes = build_safetensors(&checkpoint);

    // A *fresh* model with different random weights, so a no-op load would fail.
    let mut loaded = Gpt2Model::new(config).expect("tiny GPT-2 must build");
    assert_ne!(
        snapshot(&loaded),
        expected,
        "the fresh model must start out different, or the test proves nothing"
    );

    let mut cursor = std::io::Cursor::new(bytes);
    loaded
        .load_pretrained(&mut cursor)
        .expect("a checkpoint keyed by named_tensors' own names must load");

    assert_eq!(
        snapshot(&loaded),
        expected,
        "every published name must round-trip through the real loader"
    );
}

/// Write a real `model.safetensors` into `directory` and return the directory.
///
/// The HuggingFace loader (`auto_create_loader`) discovers a bare
/// `model.safetensors` in a directory with no index file, which is exactly the
/// layout a single-shard HuggingFace repository has — so LLaMA's and Mistral's
/// real `load_from_path` runs unmodified against it.
fn write_checkpoint_directory(
    directory: &std::path::Path,
    tensors: &[(String, Vec<usize>, Vec<f32>)],
) {
    std::fs::create_dir_all(directory).expect("checkpoint directory must be creatable");
    std::fs::write(
        directory.join("model.safetensors"),
        build_safetensors(tensors),
    )
    .expect("checkpoint must be writable");
}

/// The `(name, shape, values)` triples of a model's published parameters, in the
/// form [`build_safetensors`] and [`write_checkpoint_directory`] want.
fn checkpoint_entries(
    snapshot: &HashMap<String, (Vec<usize>, Vec<f32>)>,
) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    snapshot
        .iter()
        .map(|(name, (shape, values))| (name.clone(), shape.clone(), values.clone()))
        .collect()
}

/// The same claim as the GPT-2 test, for BERT: a checkpoint keyed by exactly the
/// names `named_tensors` publishes must load through the real
/// `Model::load_pretrained`, which goes through `Checkpoint`/`WeightBinder` and
/// the `BertLayerNames::bert()` table.
///
/// # Why this is stronger than the GGUF round trip
///
/// `bert_and_llama_round_trip_through_gguf` compares `named_tensors` → file →
/// `named_tensors`: both sides come from the same name table, so a table that
/// disagreed with the *loader* would still pass. Here the file is consumed by the
/// loader's own name map, so any divergence — a renamed sub-module, a missing
/// `LayerNorm`, a `gamma`/`weight` spelling drift — fails the load or leaves a
/// value behind.
#[test]
fn bert_published_names_load_back_through_load_pretrained() {
    let config = BertConfig {
        vocab_size: 12,
        hidden_size: 8,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        intermediate_size: 16,
        max_position_embeddings: 8,
        type_vocab_size: 2,
        hidden_dropout_prob: 0.0,
        attention_probs_dropout_prob: 0.0,
        ..BertConfig::default()
    };

    let mut source = BertModel::new(config.clone()).expect("tiny BERT must build");
    install_known_weights(&mut source, 2.5);
    let expected = snapshot(&source);

    let bytes = build_safetensors(&checkpoint_entries(&expected));

    let mut loaded = BertModel::new(config).expect("tiny BERT must build");
    assert_ne!(
        snapshot(&loaded),
        expected,
        "the fresh model must start out different, or the test proves nothing"
    );

    let mut cursor = std::io::Cursor::new(bytes);
    loaded
        .load_pretrained(&mut cursor)
        .expect("a checkpoint keyed by BERT's own published names must load");

    assert_eq!(
        snapshot(&loaded),
        expected,
        "every published BERT name must round-trip through the real loader"
    );
}

/// LLaMA's loader is path-based (`load_from_path`), so the checkpoint is written
/// to a real directory and read back by the real HuggingFace loader.
///
/// # Why the assertion is on values, not on `Ok(())`
///
/// `LlamaModel::load_from_path_with_config` fetches each tensor with
/// `if let Ok(..) = loader.load_tensor(name)`, so a name the file does not carry
/// is skipped silently and the load still returns `Ok(())`. Comparing the whole
/// snapshot afterwards is therefore the only assertion that actually detects a
/// divergence between the published names and the names the loader looks up: a
/// skipped tensor keeps its random initialisation and the comparison fails.
#[test]
fn llama_published_names_load_back_through_the_real_loader() {
    let scratch = ScratchDir::new("loader_llama");
    let directory = scratch.join("checkpoint");

    let config = LlamaConfig {
        vocab_size: 12,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: None,
        max_position_embeddings: 8,
        ..LlamaConfig::default()
    };

    let mut source = LlamaForCausalLM::new(config.clone()).expect("tiny LLaMA must build");
    install_known_weights(&mut source, 5.5);
    let expected = snapshot(&source);

    write_checkpoint_directory(&directory, &checkpoint_entries(&expected));

    let mut loaded = LlamaForCausalLM::new(config).expect("tiny LLaMA must build");
    assert_ne!(
        snapshot(&loaded),
        expected,
        "the fresh model must start out different, or the test proves nothing"
    );

    loaded
        .load_from_path(&directory)
        .expect("LLaMA must load its own published names");

    assert_eq!(
        snapshot(&loaded),
        expected,
        "every published LLaMA name must be a name `load_from_path` looks up"
    );
}

/// Pin the LLaMA/Mistral loader's **silent-skip** behaviour, so the tests above
/// cannot be misread as proving something they do not.
///
/// `LlamaModel::load_from_path_with_config` (and Mistral's copy of it) wraps
/// every fetch in `if let Ok(..) = loader.load_tensor(name)`. An empty
/// checkpoint therefore loads successfully and changes nothing: no error, no
/// report, no signal that a caller pointed at the wrong directory.
///
/// This is not an assertion that the behaviour is *good* — it is the opposite of
/// what BERT's strict `WeightBinder`-based loader does, and a caller that wants
/// to know whether weights arrived cannot find out from the return value. It is
/// asserted here so that (a) the value-equality assertions in the tests above
/// are understood to be load-bearing rather than decorative, and (b) tightening
/// the loader later shows up as a deliberate change to this test rather than as
/// a surprise.
#[test]
fn llama_loader_silently_ignores_a_checkpoint_with_no_matching_names() {
    let scratch = ScratchDir::new("loader_llama_silent_skip");
    let directory = scratch.join("checkpoint");

    let config = LlamaConfig {
        vocab_size: 12,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: None,
        max_position_embeddings: 8,
        ..LlamaConfig::default()
    };

    let mut model = LlamaForCausalLM::new(config).expect("tiny LLaMA must build");
    install_known_weights(&mut model, 9.0);
    let before = snapshot(&model);

    // A real, parseable safetensors file whose single tensor matches no name the
    // loader ever asks for.
    write_checkpoint_directory(
        &directory,
        &[(
            "not.a.llama.parameter".to_string(),
            vec![2, 2],
            vec![1.0, 2.0, 3.0, 4.0],
        )],
    );

    model
        .load_from_path(&directory)
        .expect("the loader tolerates a checkpoint with no recognised names");

    assert_eq!(
        snapshot(&model),
        before,
        "nothing matched, so nothing may have changed — the load is a silent no-op"
    );
}

/// Mistral repeats the LLaMA claim under grouped-query attention, where `k_proj`
/// and `v_proj` are genuinely narrower than `q_proj` — a shape a loader that
/// guessed from `hidden_size` would get wrong, and which `set_weight` would then
/// reject.
#[test]
fn mistral_published_names_load_back_through_the_real_loader() {
    let scratch = ScratchDir::new("loader_mistral");
    let directory = scratch.join("checkpoint");

    let config = MistralConfig {
        vocab_size: 12,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        max_position_embeddings: 16,
        sliding_window: None,
        attention_dropout: 0.0,
        ..MistralConfig::default()
    };

    let mut source = MistralForCausalLM::new(config.clone()).expect("tiny Mistral must build");
    install_known_weights(&mut source, 13.5);
    let expected = snapshot(&source);

    write_checkpoint_directory(&directory, &checkpoint_entries(&expected));

    let mut loaded = MistralForCausalLM::new(config).expect("tiny Mistral must build");
    assert_ne!(
        snapshot(&loaded),
        expected,
        "the fresh model must start out different, or the test proves nothing"
    );

    loaded
        .load_from_path(&directory)
        .expect("Mistral must load its own published names");

    assert_eq!(
        snapshot(&loaded),
        expected,
        "every published Mistral name must be a name `load_from_path` looks up"
    );
}

/// T5's loader takes a [`trustformers_core::traits::WeightReader`], so the
/// checkpoint bytes are parsed by the real `Checkpoint` and handed over through
/// `CheckpointReader` — the same adapter a downloaded file goes through.
///
/// T5's reader is strict: `read_tensor` returns an error for an unknown name, so
/// a single divergence between `named_tensors` and `T5Stack::load_weights` fails
/// the call outright rather than being papered over.
#[test]
fn t5_published_names_load_back_through_the_real_reader() {
    use trustformers_models::weight_loading::CheckpointReader;

    let config = T5Config {
        vocab_size: 10,
        d_model: 4,
        d_kv: 2,
        d_ff: 8,
        num_layers: 2,
        num_decoder_layers: Some(2),
        num_heads: 2,
        relative_attention_num_buckets: 4,
        dropout_rate: 0.0,
        ..T5Config::default()
    };

    let mut source = T5Model::new(config.clone()).expect("tiny T5 must build");
    install_known_weights(&mut source, 11.5);
    let expected = snapshot(&source);

    let bytes = build_safetensors(&checkpoint_entries(&expected));

    let mut loaded = T5Model::new(config).expect("tiny T5 must build");
    assert_ne!(
        snapshot(&loaded),
        expected,
        "the fresh model must start out different, or the test proves nothing"
    );

    let mut cursor = std::io::Cursor::new(bytes);
    let mut reader = CheckpointReader::from_reader(&mut cursor).expect("checkpoint must parse");
    loaded
        .load_weights_from_reader(&mut reader)
        .expect("a checkpoint keyed by T5's own published names must load");

    assert_eq!(
        snapshot(&loaded),
        expected,
        "every published T5 name must round-trip through the real reader"
    );
}

/// The one deliberate asymmetry in the five families, pinned so it cannot drift
/// into a silent bug: `T5ForConditionalGeneration` *publishes* `lm_head.weight`
/// (it owns a distinct tensor, and the trait forbids listing `shared.weight`
/// twice), but `load_weights_from_reader` never reads an `lm_head.weight` key —
/// it re-ties the head by copying `shared.weight`, which is what HuggingFace's
/// `tie_word_embeddings` means for T5.
///
/// So a checkpoint written from `named_tensors` round-trips everywhere *except*
/// the head, which comes back tied. Asserting the tie (rather than equality with
/// the source head) states the real behaviour instead of hiding it.
#[test]
fn t5_conditional_generation_reader_ties_the_lm_head_to_shared() {
    use trustformers_models::weight_loading::CheckpointReader;

    let config = T5Config {
        vocab_size: 10,
        d_model: 4,
        d_kv: 2,
        d_ff: 8,
        num_layers: 1,
        num_decoder_layers: Some(1),
        num_heads: 2,
        relative_attention_num_buckets: 4,
        dropout_rate: 0.0,
        ..T5Config::default()
    };

    let mut source =
        T5ForConditionalGeneration::new(config.clone()).expect("tiny T5 LM must build");
    install_known_weights(&mut source, 17.5);
    let expected = snapshot(&source);

    let shared = expected.get("shared.weight").cloned().expect("shared table is published");
    let source_head = expected.get("lm_head.weight").cloned().expect("head is published");
    assert_ne!(
        source_head, shared,
        "the fixture must start with an untied head, or the tie proves nothing"
    );

    let bytes = build_safetensors(&checkpoint_entries(&expected));

    let mut loaded = T5ForConditionalGeneration::new(config).expect("tiny T5 LM must build");
    let mut cursor = std::io::Cursor::new(bytes);
    let mut reader = CheckpointReader::from_reader(&mut cursor).expect("checkpoint must parse");
    loaded
        .load_weights_from_reader(&mut reader)
        .expect("T5 LM must load its published names");

    let after = snapshot(&loaded);
    for (name, value) in &expected {
        if name == "lm_head.weight" {
            continue;
        }
        assert_eq!(after.get(name), Some(value), "'{name}' must round-trip");
    }
    assert_eq!(
        after.get("lm_head.weight"),
        Some(&shared),
        "the reader ties the head to `shared.weight` rather than reading `lm_head.weight`"
    );
}

// ---------------------------------------------------------------------------
// GGUF round trip
// ---------------------------------------------------------------------------

/// The headline proof: export a tiny GPT-2 to a real `.gguf` file, re-read it
/// with the real reader, and check every name, shape and value.
#[test]
fn gpt2_round_trips_through_the_real_gguf_exporter() {
    let scratch = ScratchDir::new("gguf_gpt2");
    let output = scratch.join("tiny_gpt2");

    let mut model = Gpt2Model::new(tiny_gpt2_config()).expect("tiny GPT-2 must build");
    install_known_weights(&mut model, 1.5);
    let expected = snapshot(&model);

    let config = ExportConfig {
        format: ExportFormat::GGUF,
        output_path: output.to_string_lossy().to_string(),
        precision: ExportPrecision::FP32,
        sequence_length: Some(8),
        vocab_size: Some(11),
        ..Default::default()
    };
    GGUFExporter::new().export(&model, &config).expect("GGUF export must succeed");

    let path = output.with_extension("gguf");
    assert!(path.exists(), "the exporter must have written a file");
    let parsed = read_gguf_file(&path).expect("the real GGUF reader must parse our own output");

    assert_eq!(
        parsed.tensors.len(),
        expected.len(),
        "every parameter must reach the file"
    );

    for (name, (shape, values)) in &expected {
        let (info, _) = parsed.tensor(name).unwrap_or_else(|| panic!("'{name}' missing from file"));

        // GGUF stores dimensions fastest-varying first, i.e. reversed.
        let file_shape: Vec<usize> =
            info.dimensions.iter().rev().map(|&dimension| dimension as usize).collect();
        assert_eq!(
            &file_shape, shape,
            "shape of '{name}' changed on the way out"
        );

        let file_values = parsed.tensor_f32(name).expect("tensor data must decode");
        assert_eq!(&file_values, values, "values of '{name}' changed");
    }

    // Metadata must describe this model, not a hard-coded skeleton. The
    // architecture string is read from the model's own config rather than
    // spelled out here, so the assertion cannot drift from the source of truth.
    assert_eq!(
        parsed
            .metadata
            .get("general.architecture")
            .and_then(trustformers_core::export::GGUFValue::as_str),
        Some(<Gpt2Config as trustformers_core::traits::Config>::architecture(model.get_config()))
    );
    assert_eq!(
        parsed
            .metadata
            .get("general.parameter_count")
            .and_then(trustformers_core::export::GGUFValue::as_u64),
        Some(model.num_parameters() as u64)
    );
}

/// Two GPT-2 models with identical shapes but different weights must produce
/// different files. This is the property the pre-`named_tensors` exporters could
/// not have: they wrote the same bytes whatever the model held.
#[test]
fn gguf_export_output_tracks_the_models_weights() {
    let scratch = ScratchDir::new("gguf_varies");

    let write = |seed: f32, name: &str| -> Vec<u8> {
        let output = scratch.join(name);
        let mut model = Gpt2Model::new(tiny_gpt2_config()).expect("tiny GPT-2 must build");
        install_known_weights(&mut model, seed);
        let config = ExportConfig {
            format: ExportFormat::GGUF,
            output_path: output.to_string_lossy().to_string(),
            ..Default::default()
        };
        GGUFExporter::new().export(&model, &config).expect("export");
        std::fs::read(output.with_extension("gguf")).expect("read back")
    };

    let first = write(0.0, "a");
    let second = write(100.0, "b");
    assert_eq!(
        first.len(),
        second.len(),
        "identical shapes must give identical file sizes"
    );
    assert_ne!(first, second, "different weights must give different files");
}

/// BERT and LLaMA go through the same exporter with their own name conventions.
#[test]
fn bert_and_llama_round_trip_through_gguf() {
    let scratch = ScratchDir::new("gguf_other_families");

    let mut bert = BertModel::new(BertConfig {
        vocab_size: 12,
        hidden_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        intermediate_size: 16,
        max_position_embeddings: 8,
        type_vocab_size: 2,
        hidden_dropout_prob: 0.0,
        attention_probs_dropout_prob: 0.0,
        ..BertConfig::default()
    })
    .expect("tiny BERT must build");
    install_known_weights(&mut bert, 3.0);

    let mut llama = LlamaForCausalLM::new(LlamaConfig {
        vocab_size: 12,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: None,
        max_position_embeddings: 8,
        ..LlamaConfig::default()
    })
    .expect("tiny LLaMA must build");
    install_known_weights(&mut llama, 5.0);

    // BERT's HuggingFace spelling.
    let bert_names: HashSet<String> =
        bert.named_tensors().into_iter().map(|(name, _)| name).collect();
    assert!(bert_names.contains("embeddings.word_embeddings.weight"));
    assert!(bert_names.contains("encoder.layer.0.attention.self.query.weight"));
    assert!(bert_names.contains("encoder.layer.0.output.LayerNorm.bias"));

    // LLaMA's HuggingFace spelling, including the `model.` namespace.
    let llama_names: HashSet<String> =
        llama.named_tensors().into_iter().map(|(name, _)| name).collect();
    assert!(llama_names.contains("model.embed_tokens.weight"));
    assert!(llama_names.contains("model.layers.0.self_attn.q_proj.weight"));
    assert!(llama_names.contains("model.layers.0.mlp.gate_proj.weight"));
    assert!(llama_names.contains("model.norm.weight"));
    assert!(llama_names.contains("lm_head.weight"));
    assert!(
        !llama_names.iter().any(|name| name.contains("rotary_emb")),
        "the rotary embedding holds no learnable parameters and must not be published"
    );

    for (label, expected, exported) in [
        ("bert", snapshot(&bert), scratch.join("bert")),
        ("llama", snapshot(&llama), scratch.join("llama")),
    ] {
        let config = ExportConfig {
            format: ExportFormat::GGUF,
            output_path: exported.to_string_lossy().to_string(),
            ..Default::default()
        };
        if label == "bert" {
            GGUFExporter::new().export(&bert, &config).expect("BERT export");
        } else {
            GGUFExporter::new().export(&llama, &config).expect("LLaMA export");
        }

        let parsed = read_gguf_file(exported.with_extension("gguf")).expect("re-read");
        assert_eq!(parsed.tensors.len(), expected.len(), "{label} tensor count");
        for (name, (shape, values)) in &expected {
            let (info, _) = parsed
                .tensor(name)
                .unwrap_or_else(|| panic!("{label}: '{name}' missing from file"));
            let file_shape: Vec<usize> =
                info.dimensions.iter().rev().map(|&dimension| dimension as usize).collect();
            assert_eq!(&file_shape, shape, "{label}: shape of '{name}'");
            assert_eq!(
                &parsed.tensor_f32(name).expect("decode"),
                values,
                "{label}: values of '{name}'"
            );
        }
    }
}

/// T5's names, which are the hardest of the five families: an encoder-decoder
/// pair, positionally numbered sub-layers whose indices differ between the
/// stacks, and one shared embedding table that must appear exactly once.
#[test]
fn t5_round_trips_through_gguf_with_encoder_decoder_names() {
    let scratch = ScratchDir::new("gguf_t5");
    let output = scratch.join("tiny_t5");

    let mut model = T5Model::new(T5Config {
        vocab_size: 10,
        d_model: 4,
        d_kv: 2,
        d_ff: 8,
        num_layers: 1,
        num_decoder_layers: Some(1),
        num_heads: 2,
        relative_attention_num_buckets: 4,
        dropout_rate: 0.0,
        ..T5Config::default()
    })
    .expect("tiny T5 must build");
    install_known_weights(&mut model, 11.0);
    let expected = snapshot(&model);

    let names: HashSet<&String> = expected.keys().collect();
    // Shared embedding: exactly once, at the root.
    assert!(names.contains(&"shared.weight".to_string()));
    // Encoder block 0: layer.0 = self-attention, layer.1 = feed-forward.
    assert!(names.contains(&"encoder.block.0.layer.0.SelfAttention.q.weight".to_string()));
    assert!(names.contains(&"encoder.block.0.layer.1.DenseReluDense.wi.weight".to_string()));
    assert!(names.contains(&"encoder.final_layer_norm.weight".to_string()));
    // Decoder block 0: layer.1 is cross-attention, so feed-forward shifts to layer.2.
    assert!(names.contains(&"decoder.block.0.layer.1.EncDecAttention.k.weight".to_string()));
    assert!(names.contains(&"decoder.block.0.layer.2.DenseReluDense.wo.weight".to_string()));
    assert!(names.contains(&"decoder.final_layer_norm.weight".to_string()));
    // Only self-attention carries a relative attention bias.
    assert!(names.contains(
        &"encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight".to_string()
    ));
    assert!(
        !names.contains(
            &"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight".to_string()
        ),
        "cross-attention has no relative attention bias"
    );
    // T5's norms are RMS norms: weight only, never a bias.
    assert!(
        !names.iter().any(|name| name.ends_with("layer_norm.bias")),
        "T5 layer norms have no bias term"
    );

    let config = ExportConfig {
        format: ExportFormat::GGUF,
        output_path: output.to_string_lossy().to_string(),
        ..Default::default()
    };
    GGUFExporter::new().export(&model, &config).expect("T5 GGUF export");

    let parsed = read_gguf_file(output.with_extension("gguf")).expect("re-read");
    assert_eq!(parsed.tensors.len(), expected.len());
    for (name, (shape, values)) in &expected {
        let (info, _) = parsed.tensor(name).unwrap_or_else(|| panic!("'{name}' missing"));
        let file_shape: Vec<usize> =
            info.dimensions.iter().rev().map(|&dimension| dimension as usize).collect();
        assert_eq!(&file_shape, shape, "shape of '{name}'");
        assert_eq!(
            &parsed.tensor_f32(name).expect("decode"),
            values,
            "values of '{name}'"
        );
    }
}

/// T5 ties its LM head to the shared embedding table by copying, so the two are
/// separate tensors here and both must be published under distinct names.
#[test]
fn t5_conditional_generation_publishes_a_distinct_lm_head() {
    let model = T5ForConditionalGeneration::new(T5Config {
        vocab_size: 10,
        d_model: 4,
        d_kv: 2,
        d_ff: 8,
        num_layers: 1,
        num_decoder_layers: Some(1),
        num_heads: 2,
        relative_attention_num_buckets: 4,
        dropout_rate: 0.0,
        ..T5Config::default()
    })
    .expect("tiny T5 LM must build");

    let mut seen = HashSet::new();
    for (name, _) in model.named_tensors() {
        assert!(seen.insert(name.clone()), "duplicate name '{name}'");
    }
    assert!(seen.contains("shared.weight"));
    assert!(seen.contains("lm_head.weight"));
}

/// Mistral shares LLaMA's checkpoint layout but has genuinely narrower K/V
/// projections under grouped-query attention; the published shapes must show
/// that asymmetry rather than a uniform guess.
#[test]
fn mistral_round_trips_through_gguf_with_grouped_query_shapes() {
    let scratch = ScratchDir::new("gguf_mistral");
    let output = scratch.join("tiny_mistral");

    let config = MistralConfig {
        vocab_size: 12,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        num_key_value_heads: 2, // GQA: half as many KV heads as query heads.
        max_position_embeddings: 16,
        sliding_window: None,
        attention_dropout: 0.0,
        ..MistralConfig::default()
    };
    let mut model = MistralForCausalLM::new(config.clone()).expect("tiny Mistral must build");
    install_known_weights(&mut model, 13.0);
    let expected = snapshot(&model);

    let head_dim = config.hidden_size / config.num_attention_heads;
    assert_eq!(
        expected
            .get("model.layers.0.self_attn.q_proj.weight")
            .map(|(shape, _)| shape.clone()),
        Some(vec![
            config.num_attention_heads * head_dim,
            config.hidden_size
        ]),
    );
    assert_eq!(
        expected
            .get("model.layers.0.self_attn.k_proj.weight")
            .map(|(shape, _)| shape.clone()),
        Some(vec![
            config.num_key_value_heads * head_dim,
            config.hidden_size
        ]),
        "grouped-query attention makes k_proj narrower than q_proj"
    );
    assert!(expected.contains_key("model.layers.0.mlp.down_proj.weight"));
    assert!(expected.contains_key("model.norm.weight"));
    assert!(expected.contains_key("lm_head.weight"));
    // Mistral uses no bias anywhere in its projections.
    assert!(
        !expected.keys().any(|name| name.contains("proj.bias")),
        "Mistral projections are bias-free"
    );

    let export_config = ExportConfig {
        format: ExportFormat::GGUF,
        output_path: output.to_string_lossy().to_string(),
        ..Default::default()
    };
    GGUFExporter::new().export(&model, &export_config).expect("Mistral GGUF export");

    let parsed = read_gguf_file(output.with_extension("gguf")).expect("re-read");
    assert_eq!(parsed.tensors.len(), expected.len());
    for (name, (shape, values)) in &expected {
        let (info, _) = parsed.tensor(name).unwrap_or_else(|| panic!("'{name}' missing"));
        let file_shape: Vec<usize> =
            info.dimensions.iter().rev().map(|&dimension| dimension as usize).collect();
        assert_eq!(&file_shape, shape, "shape of '{name}'");
        assert_eq!(
            &parsed.tensor_f32(name).expect("decode"),
            values,
            "values of '{name}'"
        );
    }
}

// ---------------------------------------------------------------------------
// ONNX round trip
// ---------------------------------------------------------------------------

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

/// Turn one of the model's live parameters into an ONNX initializer.
///
/// The bytes come from the model, never from a literal in this file — that is
/// the whole point of the round trip.
fn initializer_from_parameter(name: &str, tensor: &Tensor) -> ONNXTensor {
    let values = tensor.to_vec_f32().expect("test fixtures are F32");
    ONNXTensor {
        name: name.to_string(),
        data_type: ONNXDataType::Float,
        dims: tensor.shape().into_iter().map(|dimension| dimension as i64).collect(),
        raw_data: values.iter().flat_map(|value| value.to_le_bytes()).collect(),
    }
}

/// Export a real GPT-2 parameter inside a real binary `onnx.ModelProto`, then
/// decode the file and *execute* it on the ONNX CPU interpreter.
///
/// `ONNXExporter::export` deliberately refuses to synthesise a graph from a
/// `Model` (a `Model` exposes parameters, not topology). The supported path is
/// `export_graph` with an explicitly built graph — so this test builds the graph
/// and fills its initializers from `named_tensors`, which is exactly the
/// integration the missing `named_tensors` used to block.
#[test]
fn gpt2_parameters_round_trip_through_the_real_onnx_exporter() {
    let scratch = ScratchDir::new("onnx_gpt2");
    let path = scratch.join("tiny_gpt2.onnx");

    let mut model = Gpt2Model::new(tiny_gpt2_config()).expect("tiny GPT-2 must build");
    install_known_weights(&mut model, 2.5);

    // `h.0.mlp.c_fc` is `[inner, n_embd]` = [8, 4] in this crate's layout, so
    // `x @ W^T + b` maps a [1, 4] row to a [1, 8] row.
    let named: HashMap<String, &Tensor> = model.named_tensors().into_iter().collect();
    let weight = named.get("h.0.mlp.c_fc.weight").expect("c_fc weight must be published");
    let bias = named.get("h.0.mlp.c_fc.bias").expect("c_fc bias must be published");
    let weight_values = weight.to_vec_f32().expect("F32");
    let bias_values = bias.to_vec_f32().expect("F32");
    assert_eq!(weight.shape(), vec![8, 4]);
    assert_eq!(bias.shape(), vec![8]);

    let graph = ONNXGraph {
        name: "gpt2_c_fc".to_string(),
        nodes: vec![ONNXNode {
            op_type: "Gemm".to_string(),
            inputs: vec![
                "hidden".to_string(),
                "c_fc.weight".to_string(),
                "c_fc.bias".to_string(),
            ],
            outputs: vec!["intermediate".to_string()],
            // transB=1: the weight is [out, in], the ONNX convention for Gemm's
            // B operand is [in, out].
            attributes: HashMap::from([("transB".to_string(), ONNXAttribute::Int(1))]),
            name: "c_fc".to_string(),
        }],
        inputs: vec![float_value_info("hidden", vec![1, 4])],
        outputs: vec![float_value_info("intermediate", vec![1, 8])],
        initializers: vec![
            initializer_from_parameter("c_fc.weight", weight),
            initializer_from_parameter("c_fc.bias", bias),
        ],
    };

    let exporter = ONNXExporter::new();
    let onnx_model = exporter.wrap_graph(graph, &ExportConfig::default());
    exporter.export_graph(&onnx_model, &path).expect("ONNX export must succeed");

    // 1. The file is real binary protobuf, not a text dump.
    let bytes = std::fs::read(&path).expect("read back");
    assert!(!bytes.starts_with(b"IR Version"), "must not be a text dump");
    assert_eq!(bytes[0], 0x08, "ModelProto field 1 is a varint");

    // 2. The initializers decode back to exactly the model's parameters.
    let decoded = decode_model(&bytes).expect("the real decoder must parse our own output");
    assert_eq!(decoded.graph.name, "gpt2_c_fc");
    for (name, expected) in [("c_fc.weight", &weight_values), ("c_fc.bias", &bias_values)] {
        let initializer = decoded
            .graph
            .initializers
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap_or_else(|| panic!("initializer '{name}' missing"));
        let round_tripped: Vec<f32> = initializer
            .raw_data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        assert_eq!(&round_tripped, expected, "'{name}' changed on the way out");
    }
    assert_eq!(
        decoded
            .graph
            .initializers
            .iter()
            .find(|t| t.name == "c_fc.weight")
            .map(|t| &t.dims),
        Some(&vec![8i64, 4])
    );

    // 3. The decoded graph actually runs, and its output matches the same
    //    computation performed directly against the live parameters.
    let executor = OnnxGraphExecutor::new(decoded).expect("graph must be executable");
    let hidden = [0.5f32, -1.25, 2.0, 0.75];
    let outputs = executor
        .run(HashMap::from([(
            "hidden".to_string(),
            CpuTensor::f32(hidden.to_vec(), vec![1, 4]).expect("input tensor"),
        )]))
        .expect("interpreter run must succeed");
    let produced = outputs.get("intermediate").expect("graph output must be present").to_f32_vec();

    let mut reference = vec![0.0f32; 8];
    for (row, slot) in reference.iter_mut().enumerate() {
        let mut sum = bias_values[row];
        for (column, value) in hidden.iter().enumerate() {
            sum += weight_values[row * 4 + column] * value;
        }
        *slot = sum;
    }

    assert_eq!(produced.len(), reference.len());
    for (index, (actual, expected)) in produced.iter().zip(&reference).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-4 * expected.abs().max(1.0),
            "output[{index}] = {actual}, expected {expected}"
        );
    }
}

/// The breadth companion to the test above: **every** parameter GPT-2 publishes
/// survives the real ONNX encoder and decoder, not just the one the Gemm node
/// happens to consume.
///
/// The previous test proves the pipeline works end to end for a single tensor
/// (it also *executes* the graph); it cannot catch a per-tensor encoding bug —
/// an odd rank mishandled in the `dims` varint packing, a name with a `.` in it
/// truncated, a length-prefix that only happens to be right for a [8, 4] tensor.
/// Feeding all 30 of the tiny model's parameters through as initializers turns
/// each of those into a failing assertion.
///
/// Unreferenced initializers are valid ONNX (they are just constants no node
/// reads), so the graph stays the same one-node graph; only the decode side is
/// checked here, since executing it would exercise nothing new.
#[test]
fn every_gpt2_parameter_survives_the_onnx_encoder() {
    let scratch = ScratchDir::new("onnx_all_parameters");
    let path = scratch.join("tiny_gpt2_all.onnx");

    let mut model = Gpt2Model::new(tiny_gpt2_config()).expect("tiny GPT-2 must build");
    install_known_weights(&mut model, 4.75);
    let expected = snapshot(&model);
    assert!(
        expected.len() > 20,
        "the fixture must carry enough tensors to be a breadth check, got {}",
        expected.len()
    );

    let named = model.named_tensors();
    let initializers: Vec<ONNXTensor> = named
        .iter()
        .map(|(name, tensor)| initializer_from_parameter(name, tensor))
        .collect();

    // One trivial node so the graph is structurally well-formed; the parameters
    // ride along as constants.
    let graph = ONNXGraph {
        name: "gpt2_all_parameters".to_string(),
        nodes: vec![ONNXNode {
            op_type: "Identity".to_string(),
            inputs: vec!["hidden".to_string()],
            outputs: vec!["passthrough".to_string()],
            attributes: HashMap::new(),
            name: "identity".to_string(),
        }],
        inputs: vec![float_value_info("hidden", vec![1, 4])],
        outputs: vec![float_value_info("passthrough", vec![1, 4])],
        initializers,
    };

    let exporter = ONNXExporter::new();
    let onnx_model = exporter.wrap_graph(graph, &ExportConfig::default());
    exporter.export_graph(&onnx_model, &path).expect("ONNX export must succeed");

    let bytes = std::fs::read(&path).expect("read back");
    let decoded = decode_model(&bytes).expect("the real decoder must parse our own output");
    assert_eq!(
        decoded.graph.initializers.len(),
        expected.len(),
        "every published parameter must appear in the file exactly once"
    );

    for (name, (shape, values)) in &expected {
        let initializer = decoded
            .graph
            .initializers
            .iter()
            .find(|tensor| &tensor.name == name)
            .unwrap_or_else(|| panic!("initializer '{name}' missing from the decoded file"));

        let decoded_shape: Vec<usize> =
            initializer.dims.iter().map(|&dimension| dimension as usize).collect();
        assert_eq!(&decoded_shape, shape, "shape of '{name}'");

        let round_tripped: Vec<f32> = initializer
            .raw_data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        assert_eq!(&round_tripped, values, "values of '{name}'");
    }
}

/// `ONNXExporter::export` still refuses to guess a topology — that refusal is
/// correct behaviour and must not regress into a fabricated graph now that
/// models publish weights.
#[test]
fn onnx_model_export_still_refuses_to_invent_a_topology() {
    let scratch = ScratchDir::new("onnx_refusal");
    let model = Gpt2Model::new(tiny_gpt2_config()).expect("tiny GPT-2 must build");
    let config = ExportConfig {
        format: ExportFormat::ONNX,
        output_path: scratch.join("model").to_string_lossy().to_string(),
        ..Default::default()
    };

    let error = ONNXExporter::new()
        .export(&model, &config)
        .expect_err("a Model carries no graph, so this must fail");
    let message = error.to_string();
    assert!(
        message.contains("graph")
            || message.contains("topology")
            || message.contains("Unsupported"),
        "the refusal must explain itself: {message}"
    );
    assert!(
        !scratch.join("model.onnx").exists(),
        "no file may be written for a refused export"
    );

    // But validate_model now passes, because the weights genuinely exist.
    ONNXExporter::new()
        .validate_model(&model, ExportFormat::ONNX)
        .expect("a model with named tensors must validate");
}
