//! In-memory registry of the checkpoints this server has loaded.
//!
//! Every field reported out of `ModelInfo` is measured, not estimated:
//! `num_parameters` comes from the loaded head's own [`Model::num_parameters`],
//! and there is no memory-usage figure at all — this server does not measure
//! process or device memory, and a parameter-count-derived estimate would be
//! fabricated telemetry presented as a measurement.
//!
//! Weight loading is checked before it is attempted: `trustformers`'s
//! task-specific `AutoModelFor*::from_pretrained` loaders silently keep a
//! freshly-initialised (untrained, random-weight) head when no checkpoint file
//! is present at the expected path, with no error and no warning. Serving
//! predictions from such a model would look exactly like a real answer while
//! being meaningless, so [`ModelManager::load_model`] refuses to call them
//! unless a real weight file already exists on disk.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tracing::info;

use trustformers::core::traits::{Model, Tokenizer};
use trustformers::{
    AutoConfig, AutoModelForCausalLM, AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer,
};

use crate::error::{AppError, AppResult};

/// The one weight-file name `AutoModelForSequenceClassification`,
/// `AutoModelForTokenClassification`, `AutoModelForQuestionAnswering` and
/// `AutoModelForCausalLM::from_pretrained` all look for.
///
/// This is deliberately narrower than `trustformers::automodel::WEIGHT_FILE_CANDIDATES`
/// (which also accepts `pytorch_model.bin` / `model.bin` for the plain,
/// non-task `AutoModel` loader): the task-specific loaders in
/// `trustformers::automodel_tasks` have no such fallback, so checking a
/// broader list here could let this server report a "successful" load for a
/// directory whose weight file the actual loader never looks at — it would
/// silently keep its random initialisation while this check waved it through.
const TASK_WEIGHT_FILE: &str = "model.safetensors";

fn require_weight_file(model_path: &str) -> AppResult<()> {
    let candidate = Path::new(model_path).join(TASK_WEIGHT_FILE);
    if candidate.exists() {
        Ok(())
    } else {
        Err(AppError::LoadError(format!(
            "no `{TASK_WEIGHT_FILE}` found under `{model_path}`; refusing to serve a \
             randomly-initialised model. Point `model_name` at a local directory containing a \
             real checkpoint (`config.json`, `{TASK_WEIGHT_FILE}`, and the tokenizer files for \
             that checkpoint)"
        )))
    }
}

/// The task a loaded checkpoint was assembled for.
///
/// Chosen by the caller of `POST /models` rather than guessed from the model
/// name: architecture alone does not determine a task (a BERT checkpoint could
/// back classification, NER, or QA), and guessing would risk silently loading
/// the wrong head.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TaskKind {
    TextClassification,
    TokenClassification,
    QuestionAnswering,
    TextGeneration,
}

impl TaskKind {
    pub fn as_str(self) -> &'static str {
        match self {
            TaskKind::TextClassification => "text-classification",
            TaskKind::TokenClassification => "token-classification",
            TaskKind::QuestionAnswering => "question-answering",
            TaskKind::TextGeneration => "text-generation",
        }
    }
}

impl std::fmt::Display for TaskKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The task-specific head actually driving a loaded model, plus whatever that
/// task needs that the others don't (label names, the mutex generation needs).
pub enum LoadedTask {
    Classification {
        model: AutoModelForSequenceClassification,
        labels: Vec<String>,
    },
    TokenClassification {
        model: AutoModelForTokenClassification,
        labels: Vec<String>,
    },
    QuestionAnswering {
        model: AutoModelForQuestionAnswering,
    },
    /// `AutoModelForCausalLM::generate` takes `&mut self` (the underlying
    /// architectures mutate internal decode state); a plain `std::sync::Mutex`
    /// is enough because every call happens inside `spawn_blocking`, never
    /// held across an `.await`.
    Generation {
        model: Mutex<AutoModelForCausalLM>,
    },
}

pub struct LoadedModel {
    pub id: String,
    pub model_name: String,
    pub task: TaskKind,
    pub task_data: LoadedTask,
    pub tokenizer: Arc<dyn Tokenizer + Send + Sync>,
    pub num_parameters: usize,
    pub loaded_at: chrono::DateTime<chrono::Utc>,
    pub load_duration: Duration,
}

impl LoadedModel {
    pub fn info(&self) -> ModelInfo {
        let labels = match &self.task_data {
            LoadedTask::Classification { labels, .. } => Some(labels.clone()),
            LoadedTask::TokenClassification { labels, .. } => Some(labels.clone()),
            LoadedTask::QuestionAnswering { .. } | LoadedTask::Generation { .. } => None,
        };
        ModelInfo {
            id: self.id.clone(),
            model_name: self.model_name.clone(),
            task: self.task.as_str().to_string(),
            num_parameters: self.num_parameters,
            labels,
            loaded_at: self.loaded_at.to_rfc3339(),
            load_duration_ms: self.load_duration.as_secs_f64() * 1000.0,
        }
    }

    /// Fails with [`AppError::TaskMismatch`] naming both the task this model
    /// was actually loaded for and the one the caller's endpoint needs,
    /// instead of a generic "wrong type" error.
    pub fn require_task(&self, expected: TaskKind) -> AppResult<()> {
        if self.task == expected {
            Ok(())
        } else {
            Err(AppError::TaskMismatch {
                model_id: self.id.clone(),
                loaded_task: self.task.as_str(),
                expected_task: expected.as_str(),
            })
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub model_name: String,
    pub task: String,
    /// Real parameter count from the loaded head's own [`Model::num_parameters`].
    pub num_parameters: usize,
    /// Present only for `text-classification` / `token-classification`, whose
    /// labels are either supplied by the caller of `POST /models` or default
    /// to `LABEL_0`, `LABEL_1`, ... — this server has no source of real
    /// semantic label names (no `id2label` is read from the checkpoint
    /// config), so a hardcoded guess like `["NEGATIVE", "POSITIVE"]` would be
    /// exactly the kind of fabricated-looking-real value this server must not
    /// produce.
    pub labels: Option<Vec<String>>,
    pub loaded_at: String,
    pub load_duration_ms: f64,
}

/// Recovers the label count [`ModelManager::load_model`]'s earlier
/// `num_labels` match already validated as `Some` for `task`.
///
/// That earlier match runs first (see the "Validate the task-specific
/// request shape" comment above `load_model`) precisely so a missing/zero
/// `num_labels` fails fast as a 400 before any filesystem or network access;
/// by the time this is called `num_labels` is always `Some` for
/// `TextClassification`/`TokenClassification`. A graceful `AppError` here —
/// rather than `.expect()` — means that if a future change to the earlier
/// match ever broke that invariant, the request fails with a diagnosable 500
/// instead of panicking the request task.
fn validated_num_labels(num_labels: Option<usize>, task: TaskKind) -> AppResult<usize> {
    num_labels.ok_or_else(|| {
        // `LoadError`, not `InferenceError`: this only ever fires (if ever)
        // from inside `load_model`, before any inference has happened.
        AppError::LoadError(format!(
            "internal error: `num_labels` was not validated before dispatching task `{}`",
            task.as_str()
        ))
    })
}

fn resolve_labels(labels: Option<Vec<String>>, num_labels: usize) -> Vec<String> {
    match labels {
        Some(labels) if labels.len() == num_labels => labels,
        _ => (0..num_labels).map(|i| format!("LABEL_{i}")).collect(),
    }
}

fn sequence_classifier_num_parameters(model: &AutoModelForSequenceClassification) -> usize {
    match model {
        AutoModelForSequenceClassification::Bert(m) => m.num_parameters(),
        AutoModelForSequenceClassification::Roberta(m) => m.num_parameters(),
        AutoModelForSequenceClassification::Albert(m) => m.num_parameters(),
    }
}

fn token_classifier_num_parameters(model: &AutoModelForTokenClassification) -> usize {
    match model {
        AutoModelForTokenClassification::Bert(m) => m.num_parameters(),
        AutoModelForTokenClassification::Roberta(m) => m.num_parameters(),
        AutoModelForTokenClassification::Albert(m) => m.num_parameters(),
    }
}

fn question_answering_num_parameters(model: &AutoModelForQuestionAnswering) -> usize {
    match model {
        AutoModelForQuestionAnswering::Bert(m) => m.num_parameters(),
        AutoModelForQuestionAnswering::Roberta(m) => m.num_parameters(),
        AutoModelForQuestionAnswering::Albert(m) => m.num_parameters(),
    }
}

fn causal_lm_num_parameters(model: &AutoModelForCausalLM) -> usize {
    match model {
        AutoModelForCausalLM::Gpt2(m) => m.num_parameters(),
        AutoModelForCausalLM::GptNeo(m) => m.num_parameters(),
        AutoModelForCausalLM::GptJ(m) => m.num_parameters(),
    }
}

/// Registry of loaded checkpoints, keyed by a server-generated id.
pub struct ModelManager {
    models: DashMap<String, Arc<LoadedModel>>,
    max_models: usize,
    /// Base directory a relative `model_name` is resolved against. `None`
    /// resolves relative names against the process's current directory,
    /// exactly like `AutoConfig`/`AutoModelFor*::from_pretrained` would if
    /// called with that same relative path directly.
    model_root: Option<PathBuf>,
}

impl ModelManager {
    pub fn new(max_models: usize) -> Self {
        Self { models: DashMap::new(), max_models, model_root: None }
    }

    pub fn with_model_root(max_models: usize, model_root: PathBuf) -> Self {
        Self { models: DashMap::new(), max_models, model_root: Some(model_root) }
    }

    /// Resolves the checkpoint directory `model_name` actually names.
    ///
    /// An absolute `model_name` is used as-is. A relative one is joined onto
    /// `model_root` when this manager was built `with_model_root` — this is
    /// what makes the `MODEL_CACHE_DIR`/`--model-root` deployment knob real:
    /// without it, `model_name` could only ever be an absolute path or a name
    /// relative to whatever directory the server process happened to be
    /// started from.
    fn resolve_model_path(&self, model_name: &str) -> AppResult<String> {
        let path = Path::new(model_name);
        let resolved = match (&self.model_root, path.is_absolute()) {
            (_, true) => path.to_path_buf(),
            (Some(root), false) => root.join(path),
            (None, false) => path.to_path_buf(),
        };
        resolved.to_str().map(str::to_string).ok_or_else(|| {
            AppError::BadRequest(format!(
                "resolved model path for `{model_name}` is not valid UTF-8"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_model(
        &self,
        model_name: &str,
        task: TaskKind,
        num_labels: Option<usize>,
        labels: Option<Vec<String>>,
    ) -> AppResult<Arc<LoadedModel>> {
        if self.models.len() >= self.max_models {
            return Err(AppError::ResourceExhausted(format!(
                "maximum number of concurrently loaded models ({}) reached",
                self.max_models
            )));
        }

        // Validate the task-specific request shape before touching the
        // filesystem or the network: a missing/invalid `num_labels` should
        // fail fast as a 400, not after an expensive weight-file check and a
        // config resolution that can itself reach out to the model hub.
        let num_labels = match task {
            TaskKind::TextClassification | TaskKind::TokenClassification => {
                let n = num_labels.ok_or_else(|| {
                    AppError::BadRequest(format!(
                        "`num_labels` is required when `task` is `{}`",
                        task.as_str()
                    ))
                })?;
                if n == 0 {
                    return Err(AppError::BadRequest("`num_labels` must be at least 1".to_string()));
                }
                Some(n)
            },
            TaskKind::QuestionAnswering | TaskKind::TextGeneration => None,
        };

        // A relative `model_name` resolves against `model_root` (the real
        // behaviour behind the `MODEL_CACHE_DIR` deployment knob); an
        // absolute one is used as-is regardless of `model_root`.
        let resolved_path = self.resolve_model_path(model_name)?;
        let resolved_path = resolved_path.as_str();

        require_weight_file(resolved_path)?;

        let start = Instant::now();
        // `AutoConfig::from_pretrained` is invoked indirectly by every
        // `AutoModelFor*::from_pretrained` call below; resolving it here first
        // means a config problem is reported as a config problem rather than
        // surfacing from inside whichever task loader happened to run.
        AutoConfig::from_pretrained(resolved_path)
            .map_err(|e| AppError::LoadError(format!("config for `{resolved_path}`: {e}")))?;

        let (task_data, num_parameters) = match task {
            TaskKind::TextClassification => {
                let n = validated_num_labels(num_labels, task)?;
                let resolved_labels = resolve_labels(labels, n);
                let model = AutoModelForSequenceClassification::from_pretrained(resolved_path, n)
                    .map_err(|e| AppError::LoadError(format!("weights for `{resolved_path}`: {e}")))?;
                let params = sequence_classifier_num_parameters(&model);
                (LoadedTask::Classification { model, labels: resolved_labels }, params)
            },
            TaskKind::TokenClassification => {
                let n = validated_num_labels(num_labels, task)?;
                let resolved_labels = resolve_labels(labels, n);
                let model = AutoModelForTokenClassification::from_pretrained(resolved_path, n)
                    .map_err(|e| AppError::LoadError(format!("weights for `{resolved_path}`: {e}")))?;
                let params = token_classifier_num_parameters(&model);
                (LoadedTask::TokenClassification { model, labels: resolved_labels }, params)
            },
            TaskKind::QuestionAnswering => {
                let model = AutoModelForQuestionAnswering::from_pretrained(resolved_path)
                    .map_err(|e| AppError::LoadError(format!("weights for `{resolved_path}`: {e}")))?;
                let params = question_answering_num_parameters(&model);
                (LoadedTask::QuestionAnswering { model }, params)
            },
            TaskKind::TextGeneration => {
                let model = AutoModelForCausalLM::from_pretrained(resolved_path)
                    .map_err(|e| AppError::LoadError(format!("weights for `{resolved_path}`: {e}")))?;
                let params = causal_lm_num_parameters(&model);
                (LoadedTask::Generation { model: Mutex::new(model) }, params)
            },
        };

        let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(
            AutoTokenizer::from_pretrained(resolved_path)
                .map_err(|e| AppError::LoadError(format!("tokenizer for `{resolved_path}`: {e}")))?,
        );

        let id = uuid::Uuid::new_v4().to_string();
        let loaded = Arc::new(LoadedModel {
            id: id.clone(),
            model_name: model_name.to_string(),
            task,
            task_data,
            tokenizer,
            num_parameters,
            loaded_at: chrono::Utc::now(),
            load_duration: start.elapsed(),
        });

        self.models.insert(id.clone(), Arc::clone(&loaded));
        info!(
            model_id = %id,
            model_name,
            task = task.as_str(),
            num_parameters,
            elapsed_ms = loaded.load_duration.as_secs_f64() * 1000.0,
            "model loaded"
        );

        Ok(loaded)
    }

    pub fn unload_model(&self, model_id: &str) -> AppResult<()> {
        self.models.remove(model_id).ok_or_else(|| AppError::ModelNotFound(model_id.to_string()))?;
        Ok(())
    }

    pub fn get(&self, model_id: &str) -> AppResult<Arc<LoadedModel>> {
        self.models
            .get(model_id)
            .map(|entry| Arc::clone(entry.value()))
            .ok_or_else(|| AppError::ModelNotFound(model_id.to_string()))
    }

    pub fn list(&self) -> Vec<ModelInfo> {
        self.models.iter().map(|entry| entry.value().info()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_labels_uses_caller_labels_when_the_count_matches() {
        let labels = resolve_labels(Some(vec!["neg".to_string(), "pos".to_string()]), 2);
        assert_eq!(labels, vec!["neg".to_string(), "pos".to_string()]);
    }

    #[test]
    fn resolve_labels_falls_back_to_placeholders_on_a_count_mismatch() {
        // Silently truncating or padding a caller's label list would attach
        // the wrong name to a real score; refusing to guess and falling back
        // to `LABEL_i` is the honest behaviour.
        let labels = resolve_labels(Some(vec!["only-one".to_string()]), 3);
        assert_eq!(labels, vec!["LABEL_0", "LABEL_1", "LABEL_2"]);
    }

    #[test]
    fn resolve_labels_falls_back_to_placeholders_when_none_supplied() {
        let labels = resolve_labels(None, 2);
        assert_eq!(labels, vec!["LABEL_0", "LABEL_1"]);
    }

    #[test]
    fn validated_num_labels_returns_the_value_when_present() {
        assert_eq!(validated_num_labels(Some(3), TaskKind::TextClassification).expect("must succeed"), 3);
    }

    #[test]
    fn validated_num_labels_names_the_task_in_a_clear_internal_error_instead_of_panicking() {
        // `load_model`'s own earlier match always supplies `Some` for this
        // task before this is ever called; this test exercises the
        // otherwise-unreachable `None` arm directly so it stays a graceful
        // `AppError` (and keeps naming the task) rather than silently
        // regressing back to a panic.
        let err = validated_num_labels(None, TaskKind::TokenClassification).expect_err("None must be rejected");
        assert!(matches!(err, AppError::LoadError(_)), "got {err:?}");
        assert!(err.to_string().contains("token-classification"), "message was: {err}");
    }

    #[test]
    fn a_relative_model_name_resolves_against_the_configured_model_root() {
        let manager = ModelManager::with_model_root(4, PathBuf::from("/srv/checkpoints"));
        let resolved = manager.resolve_model_path("bert-base-uncased").expect("resolves");
        assert_eq!(resolved, "/srv/checkpoints/bert-base-uncased");
    }

    #[test]
    fn an_absolute_model_name_ignores_the_configured_model_root() {
        let manager = ModelManager::with_model_root(4, PathBuf::from("/srv/checkpoints"));
        let resolved = manager.resolve_model_path("/elsewhere/my-model").expect("resolves");
        assert_eq!(resolved, "/elsewhere/my-model");
    }

    #[test]
    fn a_relative_model_name_is_left_relative_with_no_model_root_configured() {
        let manager = ModelManager::new(4);
        let resolved = manager.resolve_model_path("bert-base-uncased").expect("resolves");
        assert_eq!(resolved, "bert-base-uncased");
    }

    /// `Arc<LoadedModel>` (the `Ok` side of `load_model`/`get`) is not `Debug`
    /// — `LoadedModel` embeds trait objects and library model structs that
    /// don't derive it — so these tests unwrap the `Err` side by hand instead
    /// of using `Result::expect_err`, which requires `T: Debug`.
    fn expect_load_error(result: AppResult<Arc<LoadedModel>>, context: &str) -> AppError {
        match result {
            Err(err) => err,
            Ok(_) => panic!("{context}"),
        }
    }

    #[test]
    fn a_directory_with_no_weight_file_is_refused_before_any_model_is_built() {
        let dir = std::env::temp_dir().join(format!("trustformers-server-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("create temp dir");

        let manager = ModelManager::new(4);
        let result = manager.load_model(dir.to_str().expect("utf8 path"), TaskKind::TextClassification, Some(2), None);
        let err = expect_load_error(result, "a directory with no model.safetensors must be refused");
        assert!(matches!(err, AppError::LoadError(_)), "got {err:?}");
        let message = err.to_string();
        assert!(message.contains(TASK_WEIGHT_FILE), "message was: {message}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn text_classification_without_num_labels_is_a_bad_request_not_a_guess() {
        let dir = std::env::temp_dir().join(format!("trustformers-server-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        // A present-but-empty weight file is enough to get past the
        // weight-file check; the missing `num_labels` must still be rejected
        // before any config or weight parsing is attempted.
        std::fs::write(dir.join(TASK_WEIGHT_FILE), b"").expect("write stub weight file");

        let manager = ModelManager::new(4);
        let result = manager.load_model(dir.to_str().expect("utf8 path"), TaskKind::TextClassification, None, None);
        let err = expect_load_error(result, "missing num_labels must be rejected");
        assert!(matches!(err, AppError::BadRequest(_)), "got {err:?}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn zero_num_labels_is_rejected() {
        let dir = std::env::temp_dir().join(format!("trustformers-server-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        std::fs::write(dir.join(TASK_WEIGHT_FILE), b"").expect("write stub weight file");

        let manager = ModelManager::new(4);
        let result = manager.load_model(dir.to_str().expect("utf8 path"), TaskKind::TokenClassification, Some(0), None);
        let err = expect_load_error(result, "num_labels of 0 must be rejected");
        assert!(matches!(err, AppError::BadRequest(_)), "got {err:?}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_unknown_model_id_is_not_found() {
        let manager = ModelManager::new(4);
        let err = expect_load_error(manager.get("does-not-exist"), "must not be found");
        assert!(matches!(err, AppError::ModelNotFound(_)), "got {err:?}");
    }

    #[test]
    fn unloading_an_unknown_model_id_is_not_found() {
        let manager = ModelManager::new(4);
        let err = manager.unload_model("does-not-exist").expect_err("must not be found");
        assert!(matches!(err, AppError::ModelNotFound(_)), "got {err:?}");
    }

    #[test]
    fn task_kind_round_trips_through_its_wire_string() {
        for task in [
            TaskKind::TextClassification,
            TaskKind::TokenClassification,
            TaskKind::QuestionAnswering,
            TaskKind::TextGeneration,
        ] {
            let json = serde_json::to_string(&task).expect("serialize");
            let back: TaskKind = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, task);
        }
        assert_eq!(TaskKind::TextClassification.as_str(), "text-classification");
        assert_eq!(TaskKind::QuestionAnswering.as_str(), "question-answering");
    }
}
