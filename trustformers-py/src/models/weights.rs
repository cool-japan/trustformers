//! Weight loading and saving helpers shared by every `models.rs` pyclass.
//!
//! Split out of `models.rs` to keep that file under the workspace's 2000-line
//! policy, and because these are the pure-Rust core of `from_pretrained` /
//! `save_pretrained` -- unit-tested directly in `weights_tests.rs` without
//! needing a linked `libpython` (see the module-level note on
//! [`save_pretrained_impl`] for why that split matters).

use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use trustformers_core::errors::{runtime_error, TrustformersError};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Model;

// Wrapper type to implement error conversion
#[derive(Debug)]
pub struct TrustformersErrorWrapper(pub TrustformersError);

impl From<TrustformersError> for TrustformersErrorWrapper {
    fn from(err: TrustformersError) -> Self {
        TrustformersErrorWrapper(err)
    }
}

impl From<TrustformersErrorWrapper> for PyErr {
    fn from(err: TrustformersErrorWrapper) -> PyErr {
        PyValueError::new_err(format!("TrustformersError: {:?}", err.0))
    }
}

// Helper function to convert TrustformersError to PyErr
pub(crate) fn trustformers_error_to_py_err(err: TrustformersError) -> PyErr {
    PyValueError::new_err(format!("TrustformersError: {:?}", err))
}

/// Load a model config as a JSON string.
///
/// This crate has no Hugging Face Hub downloader, so `model_name_or_path` must
/// resolve to a local `config.json`: either the path itself, a directory
/// containing one, or (for a direct weights-file path) its sibling
/// `config.json`.
///
/// This used to return the same hardcoded BERT-shaped JSON
/// (`{"model_type": "bert", "hidden_size": 768, ...}`) for every architecture
/// and every path, silently discarding whatever config a real checkpoint
/// carried -- which meant `from_pretrained` for GPT-2/T5/LLaMA/RWKV/Mamba
/// always fell back to `Config::default()`, since none of those
/// architectures' field names (`n_embd`, `d_model`, ...) exist in a
/// BERT-shaped document. A structured error when no local config exists is
/// the honest replacement.
pub(crate) fn load_config_from_hub(
    model_name_or_path: &str,
    _options: Option<()>,
) -> Result<String, Box<dyn std::error::Error>> {
    let path = std::path::Path::new(model_name_or_path);

    let config_path = if path.is_dir() {
        path.join("config.json")
    } else if path.is_file() {
        if path.file_name().and_then(|n| n.to_str()) == Some("config.json") {
            path.to_path_buf()
        } else {
            path.with_file_name("config.json")
        }
    } else {
        return Err(format!(
            "'{model_name_or_path}' is not a local file or directory; downloading configs \
             from the Hugging Face Hub is not implemented, so a local config.json is required"
        )
        .into());
    };

    if !config_path.is_file() {
        return Err(format!("no config.json found at {}", config_path.display()).into());
    }

    Ok(std::fs::read_to_string(&config_path)?)
}

/// Locate a local checkpoint file for `model_name_or_path`, if one exists.
///
/// Returns `Ok(None)` when nothing is found locally, so the caller can fall
/// back to random initialisation exactly as a "config only" HuggingFace flow
/// does (this crate has no Hugging Face Hub downloader). Returns `Err` only
/// when `model_name_or_path` names a directory that exists but holds neither
/// a `model.safetensors` nor a `pytorch_model.bin`, since that is a much
/// stronger signal of a wrong path than "no pretrained weights are available
/// yet".
pub(crate) fn find_checkpoint_path(
    model_name_or_path: &str,
) -> Result<Option<std::path::PathBuf>, TrustformersError> {
    let path = std::path::Path::new(model_name_or_path);
    if path.is_file() {
        return Ok(Some(path.to_path_buf()));
    }
    if path.is_dir() {
        for candidate in ["model.safetensors", "pytorch_model.bin"] {
            let candidate_path = path.join(candidate);
            if candidate_path.is_file() {
                return Ok(Some(candidate_path));
            }
        }
        return Err(runtime_error(format!(
            "directory '{}' exists but contains neither model.safetensors nor pytorch_model.bin",
            model_name_or_path
        )));
    }
    Ok(None)
}

/// Load pretrained weights for `model` from `model_name_or_path`, if a local
/// checkpoint is present.
///
/// Returns `Ok(false)` when no checkpoint was found locally -- the model
/// keeps its random initialisation, matching a "config only" HuggingFace
/// flow. Returns `Ok(true)` once weights are bound for real via
/// [`trustformers_core::traits::Model::load_pretrained`] (safetensors/PyTorch
/// parsing through `Checkpoint`, tensors bound by name through
/// `WeightBinder`).
///
/// A checkpoint that *is* found but fails to bind -- missing tensor, shape
/// mismatch, or an architecture whose checkpoint loader is not implemented
/// yet (LLaMA/T5/Mamba/RWKV's `Model::load_pretrained` all correctly return
/// `Err(not_implemented(..))` today rather than lying about success) -- is
/// always a hard `Err`. Silently falling back to random weights here would
/// be exactly the "no fabricated results" violation this function replaces:
/// the previous implementation called `weight_loader.list_tensors()`,
/// printed the count, and returned `Ok(())` without copying a single tensor
/// into the model.
pub(crate) fn load_pretrained_weights<M: Model>(
    model: &mut M,
    model_name_or_path: &str,
) -> Result<bool, TrustformersError> {
    let Some(checkpoint_path) = find_checkpoint_path(model_name_or_path)? else {
        return Ok(false);
    };
    let mut file = std::fs::File::open(&checkpoint_path).map_err(|e| {
        runtime_error(format!(
            "failed to open checkpoint {}: {}",
            checkpoint_path.display(),
            e
        ))
    })?;
    model.load_pretrained(&mut file)?;
    Ok(true)
}

/// Pure-Rust core of [`report_weight_loading`]: log the outcome (as
/// [`report_weight_loading`] does) and decide whether it is a hard failure,
/// returning the formatted message for the `Err` case instead of a `PyErr`
/// directly. Split out for the same reason as [`save_pretrained_impl`] --
/// unit-testable without linking `libpython`.
fn report_weight_loading_impl(
    result: Result<bool, TrustformersError>,
    model_name_or_path: &str,
) -> Result<(), String> {
    match result {
        Ok(true) => {
            tracing::info!(
                model = model_name_or_path,
                "successfully loaded pretrained weights"
            );
            Ok(())
        },
        Ok(false) => {
            tracing::info!(
                model = model_name_or_path,
                "no local weights found, using random initialization"
            );
            Ok(())
        },
        Err(e) => Err(format!(
            "Failed to load weights for {}: {}",
            model_name_or_path, e
        )),
    }
}

/// Report the outcome of [`load_pretrained_weights`] the same way across
/// every `from_pretrained` implementation: log success/absence, and turn a
/// real loading failure into a hard `PyErr` instead of a swallowed warning.
pub(crate) fn report_weight_loading(
    result: Result<bool, TrustformersError>,
    model_name_or_path: &str,
) -> Result<(), PyErr> {
    report_weight_loading_impl(result, model_name_or_path).map_err(PyValueError::new_err)
}

/// Serialise `named` tensors to a safetensors file at `path`.
///
/// Converts every tensor to `F32` regardless of its original dtype: every
/// `trustformers-models` architecture currently stores its parameters as
/// `Tensor::F32` internally (there is no mixed-precision model yet), so
/// dtype-preserving export was not worth the added complexity until one
/// exists.
pub(crate) fn write_safetensors(
    named: &[(String, &Tensor)],
    path: &std::path::Path,
) -> Result<usize, TrustformersError> {
    let mut buffers: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::with_capacity(named.len());
    for (name, tensor) in named {
        let shape = tensor.shape();
        let values = tensor.to_vec_f32()?;
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        buffers.push((name.clone(), shape, bytes));
    }

    let views = buffers
        .iter()
        .map(|(name, shape, bytes)| {
            let view =
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .map_err(|e| {
                        TrustformersError::tensor_op_error(&e.to_string(), "safetensors_export")
                    })?;
            Ok::<_, TrustformersError>((name.clone(), view))
        })
        .collect::<Result<Vec<_>, _>>()?;

    safetensors::serialize_to_file(views, None, path).map_err(|e| {
        TrustformersError::io_error(format!("failed to write safetensors file: {e}"))
    })?;

    Ok(buffers.len())
}

/// Save `model`'s config and parameters to `save_directory`, HuggingFace-style
/// (`config.json` + `model.safetensors`).
///
/// [`trustformers_core::traits::Model::named_tensors`] defaults to an empty
/// list, and as of this writing BERT/GPT-2/GPT-2-LM-head/T5/LLaMA all override
/// it for real, but RWKV and Mamba do not yet. Rather than writing an empty
/// (and therefore useless) `model.safetensors` for those two -- or silently
/// writing only `config.json` and calling that success -- this refuses
/// outright and writes nothing at all when there are no tensors to export,
/// matching the same rule already enforced by every exporter in
/// `trustformers_core::export` (see e.g. `GGUFExporter`, `CoreMLExporter`):
/// "refuses to write a file for a model that exposes no named tensors,
/// rather than inventing weights". Once RWKV/Mamba grow a real
/// `named_tensors()` override, this function starts exporting them for real
/// with no further changes needed here.
///
/// Pure-Rust core of [`save_pretrained_for_model`], kept separate so it is
/// unit-testable without a linked `libpython` (constructing a `PyErr`/
/// `PyValueError` requires the Python C API to be resolvable, which the
/// plain `cargo test` executable -- unlike the `cdylib` extension module --
/// does not link against; see `write_safetensors` above for the same split).
///
/// Returns the number of tensors written on success.
pub(crate) fn save_pretrained_impl<M>(
    model: &M,
    save_directory: &str,
    architecture: &str,
) -> Result<usize, TrustformersError>
where
    M: Model,
    M::Config: serde::Serialize,
{
    let named = model.named_tensors();
    if named.is_empty() {
        return Err(runtime_error(format!(
            "{architecture} does not override Model::named_tensors() in trustformers-models, \
             so its weights cannot be enumerated for export; nothing was written to \
             {save_directory}"
        )));
    }

    let save_path = std::path::Path::new(save_directory);
    std::fs::create_dir_all(save_path)
        .map_err(|e| TrustformersError::io_error(format!("failed to create {save_directory}: {e}")))?;

    let config_path = save_path.join("config.json");
    let config_json = serde_json::to_string_pretty(model.get_config()).map_err(|e| {
        TrustformersError::runtime_error(format!("failed to serialize config: {e}"))
    })?;
    std::fs::write(&config_path, &config_json).map_err(|e| {
        TrustformersError::io_error(format!("failed to write {}: {e}", config_path.display()))
    })?;

    let weights_path = save_path.join("model.safetensors");
    let tensor_count = write_safetensors(&named, &weights_path)?;

    tracing::info!(
        architecture,
        tensors = tensor_count,
        config = %config_path.display(),
        weights = %weights_path.display(),
        "saved pretrained model"
    );
    Ok(tensor_count)
}

/// Save `model`'s config and parameters to `save_directory` (PyO3 boundary:
/// maps [`save_pretrained_impl`]'s `TrustformersError` to a `PyValueError`).
pub(crate) fn save_pretrained_for_model<M>(
    model: &M,
    save_directory: &str,
    architecture: &str,
) -> Result<(), PyErr>
where
    M: Model,
    M::Config: serde::Serialize,
{
    save_pretrained_impl(model, save_directory, architecture)
        .map(|_tensor_count| ())
        .map_err(trustformers_error_to_py_err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use trustformers_models::bert::{BertConfig, BertModel};

    /// Build a tiny BERT config (1 layer, small dims) so fixture checkpoints
    /// stay small and tests run fast.
    fn tiny_bert_config() -> BertConfig {
        BertConfig {
            vocab_size: 11,
            hidden_size: 4,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 8,
            max_position_embeddings: 6,
            type_vocab_size: 2,
            ..BertConfig::default()
        }
    }

    /// A ramp of distinct `f32` values, so a test that asserts "checkpoint
    /// tensor X reached parameter Y" cannot pass by coincidentally matching
    /// randomly-initialised values.
    fn ramp(count: usize, seed: f32) -> Vec<f32> {
        (0..count).map(|i| seed + i as f32 * 0.5).collect()
    }

    fn safetensors_f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    /// Build a real safetensors byte stream (same on-disk format a downloaded
    /// checkpoint uses) holding every tensor a bare `BertModel` (no pooler)
    /// of `config`'s shape needs, with distinct deterministic values so the
    /// binder's name -> parameter wiring is actually exercised.
    fn build_bert_checkpoint_bytes(config: &BertConfig) -> Vec<u8> {
        let hidden = config.hidden_size;
        let mut seed = 0.0f32;
        let mut next = |count: usize| {
            let v = ramp(count, seed);
            seed += 100.0;
            v
        };

        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = vec![
            (
                "embeddings.word_embeddings.weight".to_string(),
                vec![config.vocab_size, hidden],
                next(config.vocab_size * hidden),
            ),
            (
                "embeddings.position_embeddings.weight".to_string(),
                vec![config.max_position_embeddings, hidden],
                next(config.max_position_embeddings * hidden),
            ),
            (
                "embeddings.token_type_embeddings.weight".to_string(),
                vec![config.type_vocab_size, hidden],
                next(config.type_vocab_size * hidden),
            ),
            (
                "embeddings.LayerNorm.weight".to_string(),
                vec![hidden],
                next(hidden),
            ),
            (
                "embeddings.LayerNorm.bias".to_string(),
                vec![hidden],
                next(hidden),
            ),
        ];

        for layer in 0..config.num_hidden_layers {
            let p = format!("encoder.layer.{layer}.");
            for proj in ["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense"] {
                tensors.push((format!("{p}{proj}.weight"), vec![hidden, hidden], next(hidden * hidden)));
                tensors.push((format!("{p}{proj}.bias"), vec![hidden], next(hidden)));
            }
            for norm in ["attention.output.LayerNorm", "output.LayerNorm"] {
                tensors.push((format!("{p}{norm}.weight"), vec![hidden], next(hidden)));
                tensors.push((format!("{p}{norm}.bias"), vec![hidden], next(hidden)));
            }
            tensors.push((
                format!("{p}intermediate.dense.weight"),
                vec![config.intermediate_size, hidden],
                next(config.intermediate_size * hidden),
            ));
            tensors.push((
                format!("{p}intermediate.dense.bias"),
                vec![config.intermediate_size],
                next(config.intermediate_size),
            ));
            tensors.push((
                format!("{p}output.dense.weight"),
                vec![hidden, config.intermediate_size],
                next(hidden * config.intermediate_size),
            ));
            tensors.push((format!("{p}output.dense.bias"), vec![hidden], next(hidden)));
        }

        let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensors
            .iter()
            .map(|(name, shape, values)| {
                let bytes = safetensors_f32_bytes(values);
                // Leak intentionally: the fixture only needs to outlive `serialize`,
                // called immediately below, inside this same function.
                let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());
                let view =
                    safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                        .expect("fixture tensor view is well-formed");
                (name.clone(), view)
            })
            .collect();

        safetensors::serialize(views, None).expect("fixture safetensors serialises")
    }

    /// Write `bytes` to a fresh file under the system temp directory, per
    /// COOLJAPAN policy (`std::env::temp_dir()`, never a hardcoded path).
    fn write_temp_file(stem: &str, bytes: &[u8]) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "trustformers_py_{stem}_{}_{unique}",
            std::process::id()
        ));
        let mut file = std::fs::File::create(&path).expect("temp file is writable");
        file.write_all(bytes).expect("temp file write succeeds");
        path
    }

    // ---- find_checkpoint_path ----

    #[test]
    fn find_checkpoint_path_returns_none_for_a_path_that_does_not_exist() {
        // Regression test for the old `load_model_weights`, which never
        // consulted the filesystem at all: it always called
        // `weight_loader.list_tensors()` on whatever `auto_create_loader`
        // handed back and reported "success" even when nothing existed.
        let result = find_checkpoint_path("/nonexistent/path/that/should/not/exist/12345");
        assert!(result.is_ok(), "a missing path is not an error, just 'nothing found'");
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn find_checkpoint_path_returns_the_file_itself_when_given_a_file_path() {
        let bytes = b"not a real checkpoint, just a byte probe";
        let path = write_temp_file("direct_file", bytes);

        let result = find_checkpoint_path(path.to_str().unwrap());

        assert_eq!(result.unwrap(), Some(path.clone()));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn find_checkpoint_path_finds_model_safetensors_inside_a_directory() {
        let dir = std::env::temp_dir().join(format!(
            "trustformers_py_ckpt_dir_{}_{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir creation succeeds");
        let weights_path = dir.join("model.safetensors");
        std::fs::write(&weights_path, b"probe").expect("write succeeds");

        let result = find_checkpoint_path(dir.to_str().unwrap());

        assert_eq!(result.unwrap(), Some(weights_path));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn find_checkpoint_path_rejects_a_directory_with_neither_known_filename() {
        let dir = std::env::temp_dir().join(format!(
            "trustformers_py_empty_ckpt_dir_{}_{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir creation succeeds");
        // A directory that exists, but holds neither model.safetensors nor
        // pytorch_model.bin -- almost certainly a wrong path, not "no weights yet".
        std::fs::write(dir.join("README.md"), b"not a checkpoint").ok();

        let result = find_checkpoint_path(dir.to_str().unwrap());

        assert!(result.is_err(), "an unrelated directory must be a hard error, not Ok(None)");
        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- load_config_from_hub ----

    #[test]
    fn load_config_from_hub_errors_when_no_local_config_exists() {
        // Regression test: the previous implementation ignored its
        // `model_name_or_path` argument entirely and returned the same
        // hardcoded BERT JSON for every input, including nonexistent paths.
        let result = load_config_from_hub("/nonexistent/path/xyz123", None);
        assert!(result.is_err());
    }

    #[test]
    fn load_config_from_hub_reads_a_real_config_json_verbatim() {
        let dir = std::env::temp_dir().join(format!(
            "trustformers_py_cfg_dir_{}_{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir creation succeeds");
        let config_body = r#"{"model_type": "gpt2", "n_embd": 999, "n_layer": 3}"#;
        std::fs::write(dir.join("config.json"), config_body).expect("write succeeds");

        let result = load_config_from_hub(dir.to_str().unwrap(), None).expect("config.json exists");

        // The old stub always emitted a BERT-shaped document; a real GPT-2
        // config with an unusual field value proves this reads real content
        // rather than the hardcoded fallback.
        assert!(result.contains("999"), "expected the real n_embd value, got: {result}");
        assert!(!result.contains("\"bert\""), "must not silently substitute the old BERT stub");
        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- load_pretrained_weights: the real regression test ----

    #[test]
    fn load_pretrained_weights_returns_false_when_nothing_is_found_locally() {
        let mut model = BertModel::new(tiny_bert_config()).expect("model construction succeeds");
        let loaded = load_pretrained_weights(&mut model, "/nonexistent/model/path/abc")
            .expect("a missing checkpoint is not an error");
        assert!(!loaded, "no checkpoint present must report false, not fabricate success");
    }

    #[test]
    fn load_pretrained_weights_actually_copies_checkpoint_tensors_into_the_model() {
        // This is the direct regression test for the P0 finding: the old
        // `load_model_weights` called `weight_loader.list_tensors()`, printed
        // the count, and returned `Ok(())` -- never copying a single tensor
        // into the model. Against the *old* code this assertion would fail,
        // because the embedding weight would still hold its random init.
        let config = tiny_bert_config();
        let checkpoint_bytes = build_bert_checkpoint_bytes(&config);
        let checkpoint_path = write_temp_file("bert_ckpt.safetensors", &checkpoint_bytes);

        let mut model = BertModel::new(config.clone()).expect("model construction succeeds");
        let loaded = load_pretrained_weights(&mut model, checkpoint_path.to_str().unwrap())
            .expect("a well-formed checkpoint loads without error");
        assert!(loaded, "an existing checkpoint must report true");

        // Confirm real tensor data actually reached the model: the exported
        // `named_tensors()` must contain the exact deterministic ramp values
        // the fixture wrote for the word-embedding table, not a random init.
        let named = trustformers_core::traits::Model::named_tensors(&model);
        let (_, embedding_tensor) = named
            .iter()
            .find(|(name, _)| name == "embeddings.word_embeddings.weight")
            .expect("word embedding is always exported");
        let values = embedding_tensor.to_vec_f32().expect("tensor reads back as f32");
        let expected = ramp(config.vocab_size * config.hidden_size, 0.0);
        assert_eq!(
            values, expected,
            "the checkpoint's exact values must have been copied into the live model parameter"
        );

        std::fs::remove_file(&checkpoint_path).ok();
    }

    #[test]
    fn load_pretrained_weights_fails_hard_on_a_checkpoint_missing_required_tensors() {
        // An incomplete/corrupt checkpoint must be a hard error, never a
        // silent fall-back to random weights while reporting success.
        let config = tiny_bert_config();
        // Serialise only a single, irrelevant tensor -- everything BertModel
        // actually needs is absent.
        let bytes = ramp(4, 0.0);
        let byte_data = safetensors_f32_bytes(&bytes);
        let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![4], &byte_data)
            .expect("fixture view is well-formed");
        let incomplete = safetensors::serialize([("irrelevant.tensor".to_string(), view)], None)
            .expect("fixture serialises");
        let checkpoint_path = write_temp_file("incomplete_ckpt.safetensors", &incomplete);

        let mut model = BertModel::new(config).expect("model construction succeeds");
        let result = load_pretrained_weights(&mut model, checkpoint_path.to_str().unwrap());

        assert!(result.is_err(), "a checkpoint missing required tensors must be a hard error");
        std::fs::remove_file(&checkpoint_path).ok();
    }

    // ---- report_weight_loading_impl (the pure logic behind report_weight_loading) ----

    #[test]
    fn report_weight_loading_turns_a_loading_failure_into_an_error() {
        // Regression test: the old code only ever `eprintln!`'d a warning on
        // failure and continued as if nothing had happened.
        let err = runtime_error("simulated checkpoint failure".to_string());
        let result = report_weight_loading_impl(Err(err), "some/model/path");
        assert!(result.is_err(), "a real loading failure must surface as an error to Python");
        assert!(result.unwrap_err().contains("simulated checkpoint failure"));
    }

    #[test]
    fn report_weight_loading_is_ok_for_both_found_and_not_found_cases() {
        assert!(report_weight_loading_impl(Ok(true), "path").is_ok());
        assert!(report_weight_loading_impl(Ok(false), "path").is_ok());
    }

    // ---- write_safetensors / save_pretrained_impl ----

    #[test]
    fn write_safetensors_round_trips_through_the_real_safetensors_reader() {
        let hidden = 4usize;
        let tensor = Tensor::F32(
            scirs2_core::ndarray::ArrayD::from_shape_vec(
                scirs2_core::ndarray::IxDyn(&[hidden]),
                ramp(hidden, 1.0),
            )
            .unwrap(),
        );
        let named: Vec<(String, &Tensor)> = vec![("probe.weight".to_string(), &tensor)];
        let out_path = std::env::temp_dir().join(format!(
            "trustformers_py_write_safetensors_{}_{}.safetensors",
            std::process::id(),
            line!()
        ));

        let count = write_safetensors(&named, &out_path).expect("write succeeds");
        assert_eq!(count, 1);

        let file_bytes = std::fs::read(&out_path).expect("file exists");
        let parsed = safetensors::SafeTensors::deserialize(&file_bytes).expect("real safetensors reader parses it");
        let view = parsed.tensor("probe.weight").expect("tensor is present under its name");
        assert_eq!(view.shape(), &[hidden]);

        std::fs::remove_file(&out_path).ok();
    }

    #[test]
    fn save_pretrained_impl_refuses_to_write_anything_for_a_model_with_no_named_tensors() {
        // RWKV/Mamba (and any future architecture) that has not yet overridden
        // `Model::named_tensors()` must not produce a `model.safetensors` file
        // containing zero tensors, and must not leave a `config.json` behind
        // implying the save otherwise succeeded.
        use trustformers_models::rwkv::{RwkvConfig, RwkvModel};

        let config = RwkvConfig {
            n_embd: 4,
            n_layer: 1,
            vocab_size: 8,
            n_head: 2,
            head_size: 2,
            ctx_len: 16,
            ..RwkvConfig::default()
        };
        let model = RwkvModel::new(config).expect("model construction succeeds");
        let dir = std::env::temp_dir().join(format!(
            "trustformers_py_empty_named_tensors_{}_{}",
            std::process::id(),
            line!()
        ));

        let result = save_pretrained_impl(&model, dir.to_str().unwrap(), "RwkvModel");

        assert!(result.is_err(), "a model with no exportable tensors must refuse to save");
        assert!(
            !dir.exists(),
            "nothing should be written to disk when there is nothing real to export"
        );
    }

    #[test]
    fn save_pretrained_impl_writes_a_loadable_config_and_weights_file() {
        // Regression test for the old `save_pretrained`, which wrote a
        // `pytorch_model.bin.info` *text file* containing the literal string
        // "Model weights would be saved here..." instead of any real weights.
        let config = tiny_bert_config();
        let model = BertModel::new(config.clone()).expect("model construction succeeds");
        let dir = std::env::temp_dir().join(format!(
            "trustformers_py_save_pretrained_{}_{}",
            std::process::id(),
            line!()
        ));

        let tensor_count =
            save_pretrained_impl(&model, dir.to_str().unwrap(), "BertModel").expect("save succeeds");
        assert!(tensor_count > 0);

        let config_path = dir.join("config.json");
        let weights_path = dir.join("model.safetensors");
        assert!(config_path.is_file(), "config.json must exist");
        assert!(weights_path.is_file(), "model.safetensors must exist");

        // The saved config.json must be genuinely loadable and hold the real
        // configured values, not a placeholder.
        let saved_config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
        assert_eq!(saved_config["vocab_size"], config.vocab_size);

        // The saved weights file must be real, parseable safetensors with the
        // expected tensor count -- not a text note.
        let weights_bytes = std::fs::read(&weights_path).unwrap();
        let parsed = safetensors::SafeTensors::deserialize(&weights_bytes)
            .expect("model.safetensors must be genuine, parseable safetensors");
        assert_eq!(parsed.tensors().len(), tensor_count);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_load_round_trip_preserves_exact_tensor_values() {
        // End-to-end: save a model with known weights, load a fresh model
        // from what was written, and confirm the values match exactly. This
        // is the strongest possible proof that `save_pretrained` writes real,
        // usable weights rather than a placeholder.
        let config = tiny_bert_config();
        let checkpoint_bytes = build_bert_checkpoint_bytes(&config);
        let checkpoint_path = write_temp_file("round_trip_source.safetensors", &checkpoint_bytes);

        let mut source_model = BertModel::new(config.clone()).expect("model construction succeeds");
        load_pretrained_weights(&mut source_model, checkpoint_path.to_str().unwrap())
            .expect("loading the fixture checkpoint succeeds");

        let dir = std::env::temp_dir().join(format!(
            "trustformers_py_round_trip_{}_{}",
            std::process::id(),
            line!()
        ));
        save_pretrained_impl(&source_model, dir.to_str().unwrap(), "BertModel")
            .expect("saving the loaded model succeeds");

        let mut reloaded_model = BertModel::new(config.clone()).expect("model construction succeeds");
        let reloaded = load_pretrained_weights(
            &mut reloaded_model,
            dir.join("model.safetensors").to_str().unwrap(),
        )
        .expect("loading the just-saved checkpoint succeeds");
        assert!(reloaded);

        let source_named = trustformers_core::traits::Model::named_tensors(&source_model);
        let reloaded_named = trustformers_core::traits::Model::named_tensors(&reloaded_model);
        assert_eq!(source_named.len(), reloaded_named.len());
        for (name, source_tensor) in &source_named {
            let (_, reloaded_tensor) = reloaded_named
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("{name} must round-trip through save+load"));
            assert_eq!(
                source_tensor.to_vec_f32().unwrap(),
                reloaded_tensor.to_vec_f32().unwrap(),
                "tensor {name} must be byte-for-byte identical after a save/load round trip"
            );
        }

        std::fs::remove_file(&checkpoint_path).ok();
        std::fs::remove_dir_all(&dir).ok();
    }
}
