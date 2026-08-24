//! Python bindings for training functionality

use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::path::PathBuf;
use trustformers_training::{EvaluationStrategy, SaveStrategy, TrainingArguments};

/// Owned Python reference alias (pyo3 0.28 removed the `PyObject` type alias from
/// the crate root; it is equivalent to `Py<PyAny>`).
type PyObject = Py<PyAny>;

/// Python wrapper for TrainingArguments
#[pyclass(name = "TrainingArguments", from_py_object)]
#[derive(Clone)]
pub struct PyTrainingArguments {
    inner: TrainingArguments,
}

#[pymethods]
impl PyTrainingArguments {
    #[new]
    #[pyo3(signature = (
        output_dir,
        num_epochs = 3,
        batch_size = 8,
        learning_rate = 5e-5,
        warmup_steps = 0,
        weight_decay = 0.01,
        max_grad_norm = 1.0,
        gradient_accumulation_steps = 1,
        eval_steps = 500,
        save_steps = 500,
        logging_steps = 100,
        eval_strategy = "steps",
        save_strategy = "steps",
        save_total_limit = 3,
        load_best_model_at_end = false,
        metric_for_best_model = "loss",
        greater_is_better = false,
        fp16 = false,
        bf16 = false,
        dataloader_num_workers = 0,
        seed = 42,
    ))]
    pub fn new(
        output_dir: String,
        num_epochs: usize,
        batch_size: usize,
        learning_rate: f32,
        warmup_steps: usize,
        weight_decay: f32,
        max_grad_norm: f32,
        gradient_accumulation_steps: usize,
        eval_steps: usize,
        save_steps: usize,
        logging_steps: usize,
        eval_strategy: &str,
        save_strategy: &str,
        save_total_limit: Option<usize>,
        load_best_model_at_end: bool,
        metric_for_best_model: &str,
        greater_is_better: bool,
        fp16: bool,
        bf16: bool,
        dataloader_num_workers: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let eval_strategy = match eval_strategy {
            "no" => EvaluationStrategy::No,
            "steps" => EvaluationStrategy::Steps,
            "epoch" => EvaluationStrategy::Epoch,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid evaluation strategy: {}",
                    eval_strategy
                )))
            },
        };

        let save_strategy = match save_strategy {
            "no" => SaveStrategy::No,
            "steps" => SaveStrategy::Steps,
            "epoch" => SaveStrategy::Epoch,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid save strategy: {}",
                    save_strategy
                )))
            },
        };

        let inner = TrainingArguments {
            output_dir: PathBuf::from(output_dir),
            num_train_epochs: num_epochs as f32,
            per_device_train_batch_size: batch_size,
            per_device_eval_batch_size: batch_size,
            learning_rate,
            warmup_steps,
            weight_decay,
            max_grad_norm,
            gradient_accumulation_steps,
            eval_steps,
            save_steps,
            logging_steps,
            evaluation_strategy: eval_strategy,
            save_strategy,
            save_total_limit,
            load_best_model_at_end,
            metric_for_best_model: Some(metric_for_best_model.to_string()),
            greater_is_better: Some(greater_is_better),
            fp16,
            bf16,
            dataloader_num_workers,
            seed,
            ..Default::default()
        };

        Ok(PyTrainingArguments { inner })
    }

    #[getter]
    fn output_dir(&self) -> String {
        self.inner.output_dir.to_string_lossy().to_string()
    }

    #[getter]
    fn num_epochs(&self) -> f32 {
        self.inner.num_train_epochs
    }

    #[getter]
    fn batch_size(&self) -> usize {
        self.inner.per_device_train_batch_size
    }

    #[getter]
    fn learning_rate(&self) -> f32 {
        self.inner.learning_rate
    }

    #[getter]
    fn warmup_steps(&self) -> usize {
        self.inner.warmup_steps
    }

    #[getter]
    fn weight_decay(&self) -> f32 {
        self.inner.weight_decay
    }

    #[getter]
    fn gradient_accumulation_steps(&self) -> usize {
        self.inner.gradient_accumulation_steps
    }

    #[getter]
    fn fp16(&self) -> bool {
        self.inner.fp16
    }

    #[getter]
    fn bf16(&self) -> bool {
        self.inner.bf16
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    fn __repr__(&self) -> String {
        format!(
            "TrainingArguments(output_dir='{}', num_epochs={}, batch_size={}, learning_rate={})",
            self.inner.output_dir.to_string_lossy(),
            self.inner.num_train_epochs,
            self.inner.per_device_train_batch_size,
            self.inner.learning_rate
        )
    }

    /// Convert to dictionary
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item(
            "output_dir",
            self.inner.output_dir.to_string_lossy().as_ref(),
        )?;
        dict.set_item("num_epochs", self.inner.num_train_epochs)?;
        dict.set_item("batch_size", self.inner.per_device_train_batch_size)?;
        dict.set_item("learning_rate", self.inner.learning_rate)?;
        dict.set_item("warmup_steps", self.inner.warmup_steps)?;
        dict.set_item("weight_decay", self.inner.weight_decay)?;
        dict.set_item(
            "gradient_accumulation_steps",
            self.inner.gradient_accumulation_steps,
        )?;
        dict.set_item("fp16", self.inner.fp16)?;
        dict.set_item("bf16", self.inner.bf16)?;
        dict.set_item("seed", self.inner.seed)?;
        Ok(dict)
    }
}

/// Why `PyTrainer::train()` cannot run a real training loop.
///
/// A real loop needs, at minimum: a forward pass (real, exists), a loss
/// function (real, exists -- `models::losses`), and a way to turn that loss
/// into a parameter update (backpropagation through the model, then an
/// optimizer step). The third piece does not exist for any model this crate
/// wraps:
///
/// * `trustformers_core`'s automatic-differentiation system
///   (`autodiff::variable::Variable` / `ComputationGraph`) is real, but it is
///   never constructed by `BertModel::forward`, `Gpt2LMHeadModel::forward`, or
///   any other model's `forward()` in `trustformers-models` -- those methods
///   operate on plain `Tensor`s and never build a graph node, so there is
///   nothing for `Variable::backward()` to walk back through even if a
///   caller wrapped the *input* in one.
/// * `trustformers_training::trainer::Trainer` (the training crate's own,
///   more complete trainer, which this binding does not currently drive) has
///   an explicit `ParameterAccess` trait for models that expose gradient-
///   writable parameters, but grep across the whole workspace
///   (`trustformers-core`, `trustformers-models`, `trustformers-training`)
///   finds zero `impl ParameterAccess for ...` -- no model implements it. Its
///   own `Trainer::apply_gradients_to_model` already documents this
///   limitation and is itself a no-op today, logging "Gradients computed but
///   not applied - model needs ParameterAccess trait" and returning `Ok(())`
///   without touching any weight.
///
/// So: forward and loss are real; backpropagation and the optimizer step are
/// not. Rather than run the first two and silently skip the third (which
/// would report a train_loss that never changes and imply progress that
/// never happened), `train()` refuses outright. This replaces a fabrication
/// that returned `train_loss: 0.5`, `total_steps: 1000` unconditionally, for
/// every model, dataset, and configuration.
fn no_training_path_available() -> PyErr {
    PyNotImplementedError::new_err(
        "Trainer.train() cannot run: this crate has a real forward pass and real loss functions \
         (see BertForSequenceClassification.forward(..., labels=...) etc.), but no backward/\
         gradient path from a loss to a model's own parameters exists anywhere in \
         trustformers-core, trustformers-models, or trustformers-training. Verified: \
         trustformers_core::autodiff::variable::Variable's ComputationGraph is never constructed \
         by any model's forward() (they operate on plain Tensors only), and \
         trustformers_training::trainer::ParameterAccess -- the trait a model would need to \
         implement for trustformers-training's own Trainer to apply computed gradients -- has \
         zero implementors in the workspace. There is no autograd path to run real gradient \
         descent through, so this refuses instead of returning a fabricated loss/step count.",
    )
}

/// Python wrapper for Trainer
#[pyclass(name = "Trainer")]
pub struct PyTrainer {
    /// The wrapped model object, kept so `save_model` can delegate to its real
    /// `save_pretrained` -- every concrete model class this crate registers
    /// (`BertModel`, `GPT2LMHeadModel`, `BertForSequenceClassification`, ...)
    /// has one.
    model: PyObject,
    model_name: String,
    args: PyTrainingArguments,
}

#[pymethods]
impl PyTrainer {
    #[new]
    #[pyo3(signature = (
        model,
        args,
        train_dataset = None,
        eval_dataset = None,
        tokenizer = None,
        data_collator = None,
        compute_metrics = None,
        callbacks = None,
        optimizers = None,
    ))]
    pub fn new(
        model: &Bound<'_, PyAny>,
        args: PyTrainingArguments,
        train_dataset: Option<&Bound<'_, PyAny>>,
        eval_dataset: Option<&Bound<'_, PyAny>>,
        tokenizer: Option<&Bound<'_, PyAny>>,
        data_collator: Option<&Bound<'_, PyAny>>,
        compute_metrics: Option<&Bound<'_, PyAny>>,
        callbacks: Option<&Bound<'_, PyAny>>,
        optimizers: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        // Accepted for HF Trainer API parity. `train_dataset`/`eval_dataset`
        // are not stored: there is no real training loop to run them through
        // (see `no_training_path_available`), and this binding defines no
        // dataset-to-model-input protocol for `evaluate`/`predict` to consume
        // one by either. `tokenizer` is likewise unused by any real path here.
        let _ = (
            train_dataset,
            eval_dataset,
            tokenizer,
            data_collator,
            compute_metrics,
            callbacks,
            optimizers,
        );

        let model_name = model.getattr("__class__")?.getattr("__name__")?.extract::<String>()?;

        Ok(PyTrainer {
            model: model.clone().unbind(),
            model_name,
            args,
        })
    }

    /// Train the model.
    ///
    /// Always refuses; see [`no_training_path_available`]. Replaces a
    /// fabrication that unconditionally returned `train_loss: 0.5`,
    /// `total_steps: 1000`.
    fn train(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let _ = py;
        Err(no_training_path_available())
    }

    /// Evaluate the model.
    ///
    /// Always refuses, for the same reason `predict` does: this binding
    /// stores no dataset-to-model-input protocol, so `eval_dataset` cannot be
    /// turned into real forward passes here (constructing `TokenizedInput`s
    /// and calling the model directly, as `pipelines::scoring::classify_with_bert`
    /// does, works today outside `Trainer`). Replaces a fabrication that
    /// unconditionally returned `eval_loss: 0.45`, `eval_accuracy: 0.92`,
    /// `eval_samples: 100`, discarding `eval_dataset` entirely.
    fn evaluate(
        &self,
        py: Python<'_>,
        eval_dataset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let _ = (py, eval_dataset);
        Err(PyNotImplementedError::new_err(
            "Trainer.evaluate() is not implemented: this binding defines no dataset-to-model-input \
             protocol (HuggingFace's Trainer expects a datasets.Dataset yielding per-example \
             tensors matching the model's forward signature; this crate integrates none), so \
             eval_dataset cannot be turned into real forward passes here. Build TokenizedInputs \
             from your data and call the model directly instead -- e.g. \
             BertForSequenceClassification.forward(input_ids, attention_mask, labels=...) returns a \
             real loss.",
        ))
    }

    /// Make predictions.
    ///
    /// Always refuses; see [`PyTrainer::evaluate`]. Replaces a fabrication
    /// that unconditionally returned the fixed vectors
    /// `predictions=[0.1, 0.9, 0.3, 0.7]`, `label_ids=[0, 1, 0, 1]`,
    /// discarding `test_dataset` entirely.
    fn predict(&self, py: Python<'_>, test_dataset: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let _ = (py, test_dataset);
        Err(PyNotImplementedError::new_err(
            "Trainer.predict() is not implemented: this binding defines no dataset-to-model-input \
             protocol (see Trainer.evaluate()'s error for the same gap), so test_dataset cannot be \
             turned into real forward passes here. Call the wrapped model's forward()/__call__() \
             directly on your own TokenizedInputs instead.",
        ))
    }

    /// Save the model: delegates to the wrapped model object's own real
    /// `save_pretrained(save_directory)`, which every concrete model class
    /// this crate registers implements for real (exports `config.json` and
    /// `model.safetensors` from the model's actual tensors).
    ///
    /// Replaces an implementation whose body was `Ok(())` after computing (and
    /// discarding) an unused save-directory string -- it reported success
    /// while writing nothing.
    fn save_model(&self, py: Python<'_>, output_dir: Option<String>) -> PyResult<()> {
        let save_dir =
            output_dir.unwrap_or_else(|| self.args.inner.output_dir.to_string_lossy().to_string());
        self.model.bind(py).call_method1("save_pretrained", (save_dir,))?;
        Ok(())
    }

    /// Push model to hub.
    ///
    /// Always refuses: this crate has no Hugging Face Hub client anywhere
    /// (`trustformers-py/src/tokenizers.rs` and `models/weights.rs` both
    /// document the same absence for `from_pretrained`). Replaces a
    /// fabrication that returned a `https://huggingface.co/{repo_name}` URL
    /// unconditionally, for any `repo_name`, without ever making a network
    /// call or uploading anything -- callers had no way to tell a successful
    /// push from this placeholder.
    fn push_to_hub(
        &self,
        repo_name: String,
        commit_message: Option<String>,
        private: Option<bool>,
    ) -> PyResult<String> {
        let _ = (commit_message, private);
        Err(PyNotImplementedError::new_err(format!(
            "Trainer.push_to_hub('{repo_name}') is not implemented: this crate has no Hugging Face \
             Hub client (no network call was made, nothing was uploaded). The previous \
             implementation returned 'https://huggingface.co/{repo_name}' unconditionally, which \
             looked like a successful push but was not one."
        )))
    }

    fn __repr__(&self) -> String {
        format!(
            "Trainer(model={}, args={})",
            self.model_name,
            self.args.__repr__()
        )
    }
}

/// Loss function types
#[pyclass(from_py_object)]
#[derive(Clone, Copy)]
pub enum PyLossFunction {
    CrossEntropy,
    MSE,
    MAE,
    Huber,
    CosineEmbedding,
    TripletMargin,
}

#[pymethods]
impl PyLossFunction {
    #[new]
    fn new(loss_type: &str) -> PyResult<Self> {
        match loss_type.to_lowercase().as_str() {
            "crossentropy" | "cross_entropy" => Ok(PyLossFunction::CrossEntropy),
            "mse" | "mean_squared_error" => Ok(PyLossFunction::MSE),
            "mae" | "mean_absolute_error" => Ok(PyLossFunction::MAE),
            "huber" => Ok(PyLossFunction::Huber),
            "cosine" | "cosine_embedding" => Ok(PyLossFunction::CosineEmbedding),
            "triplet" | "triplet_margin" => Ok(PyLossFunction::TripletMargin),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown loss function: {}",
                loss_type
            ))),
        }
    }

    fn __repr__(&self) -> String {
        match self {
            PyLossFunction::CrossEntropy => "LossFunction.CrossEntropy",
            PyLossFunction::MSE => "LossFunction.MSE",
            PyLossFunction::MAE => "LossFunction.MAE",
            PyLossFunction::Huber => "LossFunction.Huber",
            PyLossFunction::CosineEmbedding => "LossFunction.CosineEmbedding",
            PyLossFunction::TripletMargin => "LossFunction.TripletMargin",
        }
        .to_string()
    }
}

/// Learning rate scheduler
#[pyclass(name = "LRScheduler")]
pub struct PyLRScheduler {
    scheduler_type: String,
    warmup_steps: usize,
    num_training_steps: usize,
}

#[pymethods]
impl PyLRScheduler {
    #[new]
    #[pyo3(signature = (
        scheduler_type = "linear",
        warmup_steps = 0,
        num_training_steps = 1000,
    ))]
    pub fn new(scheduler_type: &str, warmup_steps: usize, num_training_steps: usize) -> Self {
        PyLRScheduler {
            scheduler_type: scheduler_type.to_string(),
            warmup_steps,
            num_training_steps,
        }
    }

    /// Get learning rate for current step
    fn get_lr(&self, current_step: usize) -> f32 {
        // Simple linear schedule with warmup
        if current_step < self.warmup_steps {
            current_step as f32 / self.warmup_steps as f32
        } else {
            let remaining_steps = self.num_training_steps - current_step;
            let remaining_ratio =
                remaining_steps as f32 / (self.num_training_steps - self.warmup_steps) as f32;
            remaining_ratio.max(0.0)
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LRScheduler(type='{}', warmup_steps={}, num_training_steps={})",
            self.scheduler_type, self.warmup_steps, self.num_training_steps
        )
    }
}

/// Training metrics tracker
#[pyclass(name = "TrainingMetrics")]
pub struct PyTrainingMetrics {
    metrics: HashMap<String, Vec<f32>>,
}

impl Default for PyTrainingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyTrainingMetrics {
    #[new]
    pub fn new() -> Self {
        PyTrainingMetrics {
            metrics: HashMap::new(),
        }
    }

    /// Add a metric value
    fn add(&mut self, name: String, value: f32) {
        self.metrics.entry(name).or_default().push(value);
    }

    /// Get metric values
    fn get(&self, name: &str) -> Option<Vec<f32>> {
        self.metrics.get(name).cloned()
    }

    /// Get average of metric
    fn get_average(&self, name: &str) -> Option<f32> {
        self.metrics
            .get(name)
            .map(|values| values.iter().sum::<f32>() / values.len() as f32)
    }

    /// Get all metrics as dictionary
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (name, values) in &self.metrics {
            dict.set_item(name, values)?;
        }
        Ok(dict)
    }

    /// Clear all metrics
    fn clear(&mut self) {
        self.metrics.clear();
    }

    fn __repr__(&self) -> String {
        format!("TrainingMetrics(metrics={})", self.metrics.len())
    }
}

/// Early stopping callback
#[pyclass(name = "EarlyStopping")]
pub struct PyEarlyStopping {
    patience: usize,
    min_delta: f32,
    mode: String,
    counter: usize,
    best_score: Option<f32>,
}

#[pymethods]
impl PyEarlyStopping {
    #[new]
    #[pyo3(signature = (patience = 3, min_delta = 0.0, mode = "min"))]
    pub fn new(patience: usize, min_delta: f32, mode: &str) -> Self {
        PyEarlyStopping {
            patience,
            min_delta,
            mode: mode.to_string(),
            counter: 0,
            best_score: None,
        }
    }

    /// Check if should stop
    fn should_stop(&mut self, current_score: f32) -> bool {
        let improved = match self.best_score {
            None => {
                self.best_score = Some(current_score);
                true
            },
            Some(best) => {
                let delta =
                    if self.mode == "min" { best - current_score } else { current_score - best };

                if delta > self.min_delta {
                    self.best_score = Some(current_score);
                    self.counter = 0;
                    true
                } else {
                    self.counter += 1;
                    false
                }
            },
        };

        !improved && self.counter >= self.patience
    }

    /// Reset the callback
    fn reset(&mut self) {
        self.counter = 0;
        self.best_score = None;
    }

    fn __repr__(&self) -> String {
        format!(
            "EarlyStopping(patience={}, min_delta={}, mode='{}', counter={})",
            self.patience, self.min_delta, self.mode, self.counter
        )
    }
}
