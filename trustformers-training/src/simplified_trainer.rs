use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use trustformers_core::tensor::Tensor;

use crate::losses::Loss;
use crate::metrics::{Metric, MetricCollection};

/// A dataset that `SimpleTrainer` can iterate.
///
/// Implementors expose their length and can materialise a contiguous slice of samples as an
/// `(inputs, targets)` tensor pair. The trainer derives the number of optimizer steps per
/// epoch from [`TrainingDataset::num_samples`], so the training loop is genuinely bound to
/// the data rather than to a hardcoded step count.
pub trait TrainingDataset {
    /// Total number of samples available.
    fn num_samples(&self) -> usize;

    /// Materialise samples `[start, start + len)` as `(inputs, targets)`.
    ///
    /// `len` is already clamped by the trainer so that `start + len <= num_samples()`.
    fn batch(&self, start: usize, len: usize) -> Result<(Tensor, Tensor)>;
}

/// A model `SimpleTrainer` can run forward and update.
///
/// The trainer computes `d(loss)/d(outputs)` with the configured [`Loss`] and hands it back
/// through [`TrainableModel::apply_output_gradient`], which is where the model backpropagates
/// into its own parameters and applies the optimizer step. This keeps `SimpleTrainer`
/// independent of any particular parameter representation while still performing a real
/// forward/backward/update cycle.
pub trait TrainableModel: Send + Sync {
    /// Forward pass.
    fn forward(&self, inputs: &Tensor) -> Result<Tensor>;

    /// Backpropagate `output_grad` (`d(loss)/d(outputs)`) and apply one optimizer step.
    fn apply_output_gradient(
        &mut self,
        inputs: &Tensor,
        output_grad: &Tensor,
        learning_rate: f64,
    ) -> Result<()>;
}

/// Scale `tensor` down so that its L2 norm does not exceed `max_norm`.
///
/// Returns the tensor unchanged when it is already within the budget, which keeps the
/// no-clipping path allocation-free apart from the clone the caller already owns.
fn clip_by_norm(tensor: &Tensor, max_norm: f64) -> Result<Tensor> {
    let norm = tensor.norm()? as f64;
    if !norm.is_finite() {
        return Err(anyhow::anyhow!(
            "gradient norm is not finite ({norm}); refusing to take an optimizer step"
        ));
    }
    if norm <= max_norm || norm == 0.0 {
        return Ok(tensor.clone());
    }
    Ok(tensor.scale((max_norm / norm) as f32)?)
}

/// An in-memory dataset backed by two row-aligned tensors.
///
/// `inputs` and `targets` must share their leading (sample) dimension; batching slices that
/// dimension, so no data is copied beyond the requested rows.
#[derive(Debug, Clone)]
pub struct TensorDataset {
    inputs: Tensor,
    targets: Tensor,
}

impl TensorDataset {
    /// Build a dataset from row-aligned input and target tensors.
    ///
    /// # Errors
    ///
    /// Fails when either tensor is scalar or when their leading dimensions differ.
    pub fn new(inputs: Tensor, targets: Tensor) -> Result<Self> {
        let input_shape = inputs.shape();
        let target_shape = targets.shape();
        if input_shape.is_empty() || target_shape.is_empty() {
            return Err(anyhow::anyhow!(
                "TensorDataset needs tensors with at least one (sample) dimension, \
                 got {input_shape:?} and {target_shape:?}"
            ));
        }
        if input_shape[0] != target_shape[0] {
            return Err(anyhow::anyhow!(
                "TensorDataset inputs and targets must agree on the sample dimension, \
                 got {} and {}",
                input_shape[0],
                target_shape[0]
            ));
        }
        Ok(Self { inputs, targets })
    }

    /// Borrow the full input tensor.
    pub fn inputs(&self) -> &Tensor {
        &self.inputs
    }

    /// Borrow the full target tensor.
    pub fn targets(&self) -> &Tensor {
        &self.targets
    }
}

impl TrainingDataset for TensorDataset {
    fn num_samples(&self) -> usize {
        self.inputs.shape()[0]
    }

    fn batch(&self, start: usize, len: usize) -> Result<(Tensor, Tensor)> {
        let end = start + len;
        if end > self.num_samples() {
            return Err(anyhow::anyhow!(
                "batch [{start}, {end}) exceeds dataset length {}",
                self.num_samples()
            ));
        }
        let inputs = self.inputs.slice(0, start, end)?;
        let targets = self.targets.slice(0, start, end)?;
        Ok((inputs, targets))
    }
}

/// Simplified trainer interface for easy model training
pub struct SimpleTrainer<M, D, L> {
    model: Arc<RwLock<M>>,
    train_dataset: D,
    eval_dataset: Option<D>,
    loss_fn: L,
    config: SimpleTrainingConfig,
    callbacks: Vec<Box<dyn SimpleCallback>>,
    metrics: MetricCollection,
    state: TrainingState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleTrainingConfig {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub num_epochs: u32,
    pub eval_steps: Option<u32>,
    pub save_steps: Option<u32>,
    pub logging_steps: u32,
    pub warmup_steps: u32,
    pub max_grad_norm: Option<f64>,
    pub seed: Option<u64>,
    pub output_dir: String,
    pub early_stopping_patience: Option<u32>,
    pub early_stopping_threshold: Option<f64>,
}

impl Default for SimpleTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            batch_size: 32,
            num_epochs: 3,
            eval_steps: Some(500),
            save_steps: Some(1000),
            logging_steps: 100,
            warmup_steps: 500,
            max_grad_norm: Some(1.0),
            seed: Some(42),
            output_dir: "./output".to_string(),
            early_stopping_patience: None,
            early_stopping_threshold: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainingState {
    pub epoch: u32,
    pub global_step: u32,
    pub train_loss: f64,
    pub eval_loss: Option<f64>,
    pub learning_rate: f64,
    pub is_training: bool,
    pub best_metric: Option<f64>,
    pub patience_counter: u32,
    pub should_stop: bool,
    pub start_time: Option<Instant>,
    pub metrics: HashMap<String, f64>,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            epoch: 0,
            global_step: 0,
            train_loss: 0.0,
            eval_loss: None,
            learning_rate: 0.0,
            is_training: false,
            best_metric: None,
            patience_counter: 0,
            should_stop: false,
            start_time: None,
            metrics: HashMap::new(),
        }
    }
}

/// Simplified callback interface
pub trait SimpleCallback: Send + Sync {
    fn on_train_begin(
        &mut self,
        _state: &TrainingState,
        _config: &SimpleTrainingConfig,
    ) -> Result<()> {
        Ok(())
    }

    fn on_train_end(&mut self, _state: &TrainingState) -> Result<()> {
        Ok(())
    }

    fn on_epoch_begin(&mut self, _epoch: u32, _state: &TrainingState) -> Result<()> {
        Ok(())
    }

    fn on_epoch_end(&mut self, _epoch: u32, _state: &TrainingState) -> Result<()> {
        Ok(())
    }

    fn on_step_begin(&mut self, _step: u32, _state: &TrainingState) -> Result<()> {
        Ok(())
    }

    fn on_step_end(&mut self, _step: u32, _state: &TrainingState) -> Result<()> {
        Ok(())
    }

    fn on_evaluate_begin(&mut self, _state: &TrainingState) -> Result<()> {
        Ok(())
    }

    fn on_evaluate_end(&mut self, _state: &TrainingState) -> Result<()> {
        Ok(())
    }

    fn on_save(&mut self, _state: &TrainingState) -> Result<()> {
        Ok(())
    }

    fn on_log(&mut self, _logs: &HashMap<String, f64>, _state: &TrainingState) -> Result<()> {
        Ok(())
    }
}

/// Built-in logging callback
pub struct LoggingCallback {
    log_level: LogLevel,
}

#[derive(Debug, Clone)]
pub enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
}

impl LoggingCallback {
    pub fn new(log_level: LogLevel) -> Self {
        Self { log_level }
    }
}

impl SimpleCallback for LoggingCallback {
    fn on_train_begin(
        &mut self,
        _state: &TrainingState,
        config: &SimpleTrainingConfig,
    ) -> Result<()> {
        tracing::info!(
            "🚀 Starting training with config: learning_rate={}, batch_size={}, epochs={}",
            config.learning_rate,
            config.batch_size,
            config.num_epochs
        );
        Ok(())
    }

    fn on_epoch_begin(&mut self, epoch: u32, _state: &TrainingState) -> Result<()> {
        tracing::info!("📚 Starting epoch {}", epoch);
        Ok(())
    }

    fn on_epoch_end(&mut self, epoch: u32, state: &TrainingState) -> Result<()> {
        let eval_info = if let Some(eval_loss) = state.eval_loss {
            format!(", eval_loss: {:.4}", eval_loss)
        } else {
            String::new()
        };

        tracing::info!(
            "✅ Epoch {} completed - train_loss: {:.4}{}",
            epoch,
            state.train_loss,
            eval_info
        );
        Ok(())
    }

    fn on_log(&mut self, logs: &HashMap<String, f64>, state: &TrainingState) -> Result<()> {
        if matches!(self.log_level, LogLevel::Debug) {
            tracing::debug!("📊 Step {} - {:?}", state.global_step, logs);
        }
        Ok(())
    }

    fn on_train_end(&mut self, state: &TrainingState) -> Result<()> {
        if let Some(start_time) = state.start_time {
            let duration = start_time.elapsed();
            tracing::info!("🎉 Training completed in {:.2}s", duration.as_secs_f64());
        }
        Ok(())
    }
}

/// Progress bar callback.
///
/// `update_progress` below writes directly to stdout with `\r`-redrawn
/// carriage returns, not through `tracing`: a line-based logger would emit
/// one log line per training step instead of redrawing a single bar, which
/// defeats the purpose of a progress indicator. This is an intentional,
/// caller-opted-in stdout writer (a caller must explicitly attach this
/// `SimpleCallback` to see it), not incidental print debugging.
pub struct ProgressCallback {
    total_steps: u32,
    current_step: u32,
    bar_width: usize,
}

impl ProgressCallback {
    pub fn new(total_steps: u32) -> Self {
        Self {
            total_steps,
            current_step: 0,
            bar_width: 50,
        }
    }

    fn update_progress(&mut self, step: u32) {
        self.current_step = step;
        let progress = (step as f64 / self.total_steps as f64).min(1.0);
        let filled = (progress * self.bar_width as f64) as usize;
        let empty = self.bar_width - filled;

        let bar = format!("[{}{}]", "█".repeat(filled), "░".repeat(empty));

        print!(
            "\r{} {:.1}% ({}/{})",
            bar,
            progress * 100.0,
            step,
            self.total_steps
        );
        if step >= self.total_steps {
            println!();
        }
    }
}

impl SimpleCallback for ProgressCallback {
    fn on_step_end(&mut self, step: u32, _state: &TrainingState) -> Result<()> {
        self.update_progress(step);
        Ok(())
    }
}

/// Early stopping callback
pub struct EarlyStoppingCallback {
    monitor: String,
    patience: u32,
    threshold: f64,
    mode: EarlyStoppingMode,
    best_value: Option<f64>,
    patience_counter: u32,
}

#[derive(Debug, Clone)]
pub enum EarlyStoppingMode {
    Min,
    Max,
}

impl EarlyStoppingCallback {
    pub fn new(monitor: String, patience: u32, threshold: f64, mode: EarlyStoppingMode) -> Self {
        Self {
            monitor,
            patience,
            threshold,
            mode,
            best_value: None,
            patience_counter: 0,
        }
    }
}

impl SimpleCallback for EarlyStoppingCallback {
    fn on_evaluate_end(&mut self, state: &TrainingState) -> Result<()> {
        if let Some(current_value) = state.metrics.get(&self.monitor) {
            let improved = match self.best_value {
                None => true,
                Some(best) => match self.mode {
                    EarlyStoppingMode::Min => *current_value < best - self.threshold,
                    EarlyStoppingMode::Max => *current_value > best + self.threshold,
                },
            };

            if improved {
                self.best_value = Some(*current_value);
                self.patience_counter = 0;
                tracing::info!("🎯 New best {}: {:.4}", self.monitor, current_value);
            } else {
                self.patience_counter += 1;
                if self.patience_counter >= self.patience {
                    tracing::info!(
                        "⏹️  Early stopping triggered. No improvement in {} for {} epochs",
                        self.monitor,
                        self.patience
                    );
                    // In a real implementation, we would set a flag to stop training
                }
            }
        }
        Ok(())
    }
}

/// Model checkpoint callback
pub struct CheckpointCallback {
    save_dir: String,
    save_best_only: bool,
    monitor: Option<String>,
    mode: EarlyStoppingMode,
    best_value: Option<f64>,
}

impl CheckpointCallback {
    pub fn new(save_dir: String, save_best_only: bool, monitor: Option<String>) -> Self {
        Self {
            save_dir,
            save_best_only,
            monitor,
            mode: EarlyStoppingMode::Min,
            best_value: None,
        }
    }
}

impl SimpleCallback for CheckpointCallback {
    fn on_save(&mut self, state: &TrainingState) -> Result<()> {
        let should_save = if self.save_best_only {
            if let (Some(_monitor), Some(current_value)) = (
                &self.monitor,
                self.monitor.as_ref().and_then(|m| state.metrics.get(m.as_str())),
            ) {
                let is_best = match self.best_value {
                    None => true,
                    Some(best) => match self.mode {
                        EarlyStoppingMode::Min => *current_value < best,
                        EarlyStoppingMode::Max => *current_value > best,
                    },
                };

                if is_best {
                    self.best_value = Some(*current_value);
                }
                is_best
            } else {
                true // Save if no monitor specified
            }
        } else {
            true // Always save if not save_best_only
        };

        if should_save {
            let checkpoint_path = format!("{}/checkpoint-{}", self.save_dir, state.global_step);
            tracing::info!("💾 Saving checkpoint to {}", checkpoint_path);
            // In a real implementation, would save model state here
        }

        Ok(())
    }
}

/// Metrics tracking callback
pub struct MetricsCallback {
    tracked_metrics: Vec<String>,
    history: HashMap<String, Vec<f64>>,
}

impl MetricsCallback {
    pub fn new(tracked_metrics: Vec<String>) -> Self {
        Self {
            tracked_metrics,
            history: HashMap::new(),
        }
    }

    pub fn get_history(&self, metric: &str) -> Option<&Vec<f64>> {
        self.history.get(metric)
    }

    pub fn get_all_history(&self) -> &HashMap<String, Vec<f64>> {
        &self.history
    }
}

impl SimpleCallback for MetricsCallback {
    fn on_log(&mut self, logs: &HashMap<String, f64>, _state: &TrainingState) -> Result<()> {
        for metric in &self.tracked_metrics {
            if let Some(value) = logs.get(metric) {
                self.history.entry(metric.clone()).or_default().push(*value);
            }
        }
        Ok(())
    }
}

impl<M, D, L> SimpleTrainer<M, D, L>
where
    M: TrainableModel,
    D: TrainingDataset + Clone,
    L: Loss + Send + Sync,
{
    pub fn new(model: M, train_dataset: D, loss_fn: L, config: SimpleTrainingConfig) -> Self {
        Self {
            model: Arc::new(RwLock::new(model)),
            train_dataset,
            eval_dataset: None,
            loss_fn,
            config,
            callbacks: Vec::new(),
            metrics: MetricCollection::new(),
            state: TrainingState::default(),
        }
    }

    pub fn with_eval_dataset(mut self, eval_dataset: D) -> Self {
        self.eval_dataset = Some(eval_dataset);
        self
    }

    pub fn add_callback(mut self, callback: Box<dyn SimpleCallback>) -> Self {
        self.callbacks.push(callback);
        self
    }

    pub fn add_metric(&mut self, metric: Box<dyn Metric>) -> &mut Self {
        self.metrics.add_metric_mut(metric);
        self
    }

    /// Start training with the configured parameters
    pub fn train(&mut self) -> Result<TrainingResults> {
        self.state.start_time = Some(Instant::now());
        self.state.learning_rate = self.config.learning_rate;
        self.state.is_training = true;

        // Call train begin callbacks
        for callback in &mut self.callbacks {
            callback.on_train_begin(&self.state, &self.config)?;
        }

        let mut training_history = Vec::new();

        for epoch in 1..=self.config.num_epochs {
            self.state.epoch = epoch;

            // Call epoch begin callbacks
            for callback in &mut self.callbacks {
                callback.on_epoch_begin(epoch, &self.state)?;
            }

            // Train epoch
            let epoch_result = self.train_epoch()?;
            training_history.push(epoch_result.clone());

            // Update state
            self.state.train_loss = epoch_result.train_loss;
            self.state.eval_loss = epoch_result.eval_loss;

            // Update metrics in state
            for (key, value) in &epoch_result.metrics {
                self.state.metrics.insert(key.clone(), *value);
            }

            // Call epoch end callbacks
            for callback in &mut self.callbacks {
                callback.on_epoch_end(epoch, &self.state)?;
            }

            // Check for early stopping
            if self.should_stop_early()? {
                tracing::info!("Training stopped early at epoch {}", epoch);
                break;
            }
        }

        self.state.is_training = false;

        // Call train end callbacks
        for callback in &mut self.callbacks {
            callback.on_train_end(&self.state)?;
        }

        Ok(TrainingResults {
            final_train_loss: self.state.train_loss,
            final_eval_loss: self.state.eval_loss,
            best_metric: self.state.best_metric,
            total_epochs: self.state.epoch,
            total_steps: self.state.global_step,
            training_time: self
                .state
                .start_time
                .context("start_time was not set before training started")?
                .elapsed(),
            history: training_history,
        })
    }

    /// Number of optimizer steps in one pass over the training set.
    fn steps_per_epoch(&self) -> usize {
        self.train_dataset.num_samples().div_ceil(self.config.batch_size.max(1))
    }

    fn train_epoch(&mut self) -> Result<EpochResult> {
        let batch_size = self.config.batch_size.max(1);
        let num_samples = self.train_dataset.num_samples();
        if num_samples == 0 {
            return Err(anyhow::anyhow!(
                "training dataset is empty; nothing to train on"
            ));
        }

        let mut total_loss = 0.0;
        let mut step_count = 0usize;

        // The loop length is derived from the dataset, not from a hardcoded constant.
        let steps_per_epoch = self.steps_per_epoch();

        for step in 1..=steps_per_epoch {
            self.state.global_step += 1;

            // Call step begin callbacks
            for callback in &mut self.callbacks {
                callback.on_step_begin(step as u32, &self.state)?;
            }

            let start = (step - 1) * batch_size;
            let len = batch_size.min(num_samples - start);
            let step_loss = self.train_step(start, len)?;
            total_loss += step_loss;
            step_count += 1;

            // Logging
            if self.state.global_step.is_multiple_of(self.config.logging_steps.max(1)) {
                let logs = {
                    let mut logs = HashMap::new();
                    logs.insert("train_loss".to_string(), step_loss);
                    logs.insert("learning_rate".to_string(), self.state.learning_rate);
                    logs
                };

                for callback in &mut self.callbacks {
                    callback.on_log(&logs, &self.state)?;
                }
            }

            // Evaluation
            if let Some(eval_steps) = self.config.eval_steps {
                if eval_steps > 0 && self.state.global_step.is_multiple_of(eval_steps) {
                    self.evaluate()?;
                }
            }

            // Saving
            if let Some(save_steps) = self.config.save_steps {
                if save_steps > 0 && self.state.global_step.is_multiple_of(save_steps) {
                    for callback in &mut self.callbacks {
                        callback.on_save(&self.state)?;
                    }
                }
            }

            // Call step end callbacks
            for callback in &mut self.callbacks {
                callback.on_step_end(step as u32, &self.state)?;
            }
        }

        let avg_train_loss = total_loss / step_count as f64;

        // Run evaluation at end of epoch if we have an eval dataset
        let eval_loss = self.evaluate()?;

        Ok(EpochResult {
            epoch: self.state.epoch,
            train_loss: avg_train_loss,
            eval_loss,
            metrics: self.state.metrics.clone(),
        })
    }

    /// One real optimizer step over the samples `[start, start + len)`.
    ///
    /// Forward through the model, compute the loss **and** its gradient with the configured
    /// [`Loss`], optionally clip the gradient to `max_grad_norm`, then let the model
    /// backpropagate and update its parameters.
    fn train_step(&mut self, start: usize, len: usize) -> Result<f64> {
        let (inputs, targets) = self.train_dataset.batch(start, len)?;

        let (loss, output_grad) = {
            let model = self
                .model
                .read()
                .map_err(|_| anyhow::anyhow!("model lock poisoned during forward pass"))?;
            let predictions = model.forward(&inputs)?;
            self.loss_fn.compute_with_gradients(&predictions, &targets)?
        };

        let output_grad = match self.config.max_grad_norm {
            Some(max_norm) if max_norm > 0.0 => clip_by_norm(&output_grad, max_norm)?,
            _ => output_grad,
        };

        {
            let mut model = self
                .model
                .write()
                .map_err(|_| anyhow::anyhow!("model lock poisoned during backward pass"))?;
            model.apply_output_gradient(&inputs, &output_grad, self.state.learning_rate)?;
        }

        Ok(loss as f64)
    }

    /// Evaluate on the eval dataset.
    ///
    /// Returns `None` — never a fabricated `0.0` — when no eval dataset was configured.
    fn evaluate(&mut self) -> Result<Option<f64>> {
        let Some(eval_dataset) = self.eval_dataset.as_ref() else {
            return Ok(None);
        };
        let num_samples = eval_dataset.num_samples();
        if num_samples == 0 {
            return Ok(None);
        }

        for callback in &mut self.callbacks {
            callback.on_evaluate_begin(&self.state)?;
        }

        let batch_size = self.config.batch_size.max(1);
        let mut total_loss = 0.0f64;
        let mut weight = 0usize;

        {
            let model = self
                .model
                .read()
                .map_err(|_| anyhow::anyhow!("model lock poisoned during evaluation"))?;
            let mut start = 0usize;
            while start < num_samples {
                let len = batch_size.min(num_samples - start);
                let (inputs, targets) = eval_dataset.batch(start, len)?;
                let predictions = model.forward(&inputs)?;
                let loss = self.loss_fn.compute(&predictions, &targets)?;
                total_loss += loss as f64 * len as f64;
                weight += len;
                start += len;
            }
        }

        let eval_loss = total_loss / weight as f64;
        self.state.eval_loss = Some(eval_loss);
        if self.state.best_metric.is_none_or(|best| eval_loss < best) {
            self.state.best_metric = Some(eval_loss);
        }

        for callback in &mut self.callbacks {
            callback.on_evaluate_end(&self.state)?;
        }

        Ok(Some(eval_loss))
    }

    fn should_stop_early(&self) -> Result<bool> {
        // Check if any callback has requested early stopping
        if let (Some(patience), Some(threshold)) = (
            self.config.early_stopping_patience,
            self.config.early_stopping_threshold,
        ) {
            if let Some(current_loss) = self.state.eval_loss {
                if let Some(best_metric) = self.state.best_metric {
                    if current_loss > best_metric + threshold {
                        return Ok(self.state.patience_counter >= patience);
                    }
                }
            }
        }

        Ok(self.state.should_stop)
    }

    /// Get current training state
    pub fn get_state(&self) -> &TrainingState {
        &self.state
    }

    /// Get model reference
    pub fn get_model(&self) -> Arc<RwLock<M>> {
        Arc::clone(&self.model)
    }
}

#[derive(Debug, Clone)]
pub struct TrainingResults {
    pub final_train_loss: f64,
    pub final_eval_loss: Option<f64>,
    pub best_metric: Option<f64>,
    pub total_epochs: u32,
    pub total_steps: u32,
    pub training_time: Duration,
    pub history: Vec<EpochResult>,
}

#[derive(Debug, Clone)]
pub struct EpochResult {
    pub epoch: u32,
    pub train_loss: f64,
    pub eval_loss: Option<f64>,
    pub metrics: HashMap<String, f64>,
}

/// Builder pattern for easier trainer configuration
pub struct SimpleTrainerBuilder<M, D, L> {
    model: Option<M>,
    train_dataset: Option<D>,
    eval_dataset: Option<D>,
    loss_fn: Option<L>,
    config: SimpleTrainingConfig,
    callbacks: Vec<Box<dyn SimpleCallback>>,
    metrics: Vec<Box<dyn Metric>>,
}

impl<M, D, L> Default for SimpleTrainerBuilder<M, D, L>
where
    M: TrainableModel,
    D: TrainingDataset + Clone,
    L: Loss + Send + Sync,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<M, D, L> SimpleTrainerBuilder<M, D, L>
where
    M: TrainableModel,
    D: TrainingDataset + Clone,
    L: Loss + Send + Sync,
{
    pub fn new() -> Self {
        Self {
            model: None,
            train_dataset: None,
            eval_dataset: None,
            loss_fn: None,
            config: SimpleTrainingConfig::default(),
            callbacks: Vec::new(),
            metrics: Vec::new(),
        }
    }

    pub fn model(mut self, model: M) -> Self {
        self.model = Some(model);
        self
    }

    pub fn train_dataset(mut self, dataset: D) -> Self {
        self.train_dataset = Some(dataset);
        self
    }

    pub fn eval_dataset(mut self, dataset: D) -> Self {
        self.eval_dataset = Some(dataset);
        self
    }

    pub fn loss_function(mut self, loss_fn: L) -> Self {
        self.loss_fn = Some(loss_fn);
        self
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.config.learning_rate = lr;
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    pub fn num_epochs(mut self, epochs: u32) -> Self {
        self.config.num_epochs = epochs;
        self
    }

    pub fn output_dir(mut self, dir: String) -> Self {
        self.config.output_dir = dir;
        self
    }

    pub fn with_logging(mut self) -> Self {
        self.callbacks.push(Box::new(LoggingCallback::new(LogLevel::Info)));
        self
    }

    pub fn with_progress_bar(self) -> Self {
        // Would need total steps calculation here
        self
    }

    pub fn with_early_stopping(mut self, monitor: String, patience: u32, threshold: f64) -> Self {
        self.callbacks.push(Box::new(EarlyStoppingCallback::new(
            monitor,
            patience,
            threshold,
            EarlyStoppingMode::Min,
        )));
        self
    }

    pub fn with_checkpoints(mut self, save_dir: String, save_best_only: bool) -> Self {
        self.callbacks.push(Box::new(CheckpointCallback::new(
            save_dir,
            save_best_only,
            Some("eval_loss".to_string()),
        )));
        self
    }

    pub fn build(self) -> Result<SimpleTrainer<M, D, L>> {
        let model = self.model.context("Model is required")?;
        let train_dataset = self.train_dataset.context("Training dataset is required")?;
        let loss_fn = self.loss_fn.context("Loss function is required")?;

        let mut trainer = SimpleTrainer::new(model, train_dataset, loss_fn, self.config);

        if let Some(eval_dataset) = self.eval_dataset {
            trainer = trainer.with_eval_dataset(eval_dataset);
        }

        for callback in self.callbacks {
            trainer = trainer.add_callback(callback);
        }

        for metric in self.metrics {
            trainer.add_metric(metric);
        }

        Ok(trainer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::losses::MSELoss;

    /// A real single-output linear model `y = x·w + b` with analytic gradients.
    ///
    /// Small enough to reason about by hand, but a genuine parametric model: the trainer
    /// drives its weights through `apply_output_gradient`, so a loss curve produced with it
    /// is the outcome of actual optimization.
    struct LinearModel {
        weights: Vec<f32>,
        bias: f32,
    }

    impl LinearModel {
        fn new(num_features: usize) -> Self {
            Self {
                weights: vec![0.0; num_features],
                bias: 0.0,
            }
        }
    }

    impl TrainableModel for LinearModel {
        fn forward(&self, inputs: &Tensor) -> Result<Tensor> {
            let shape = inputs.shape();
            let (rows, cols) = (shape[0], shape[1]);
            let data = inputs.data()?;
            let mut out = Vec::with_capacity(rows);
            for r in 0..rows {
                let mut acc = self.bias;
                for c in 0..cols {
                    acc += data[r * cols + c] * self.weights[c];
                }
                out.push(acc);
            }
            Ok(Tensor::from_vec(out, &[rows, 1])?)
        }

        fn apply_output_gradient(
            &mut self,
            inputs: &Tensor,
            output_grad: &Tensor,
            learning_rate: f64,
        ) -> Result<()> {
            let shape = inputs.shape();
            let (rows, cols) = (shape[0], shape[1]);
            let x = inputs.data()?;
            let g = output_grad.data()?;
            let lr = learning_rate as f32;
            for r in 0..rows {
                let dy = g[r];
                for c in 0..cols {
                    self.weights[c] -= lr * dy * x[r * cols + c];
                }
                self.bias -= lr * dy;
            }
            Ok(())
        }
    }

    /// y = 3*x0 - 2*x1 + 1 over a deterministic grid.
    fn regression_dataset(num_rows: usize) -> TensorDataset {
        let mut inputs = Vec::with_capacity(num_rows * 2);
        let mut targets = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            let x0 = (i % 7) as f32 / 7.0;
            let x1 = ((i * 3) % 11) as f32 / 11.0;
            inputs.push(x0);
            inputs.push(x1);
            targets.push(3.0 * x0 - 2.0 * x1 + 1.0);
        }
        TensorDataset::new(
            Tensor::from_vec(inputs, &[num_rows, 2]).expect("inputs"),
            Tensor::from_vec(targets, &[num_rows, 1]).expect("targets"),
        )
        .expect("dataset")
    }

    #[test]
    fn test_simple_trainer_creation() {
        let trainer = SimpleTrainer::new(
            LinearModel::new(2),
            regression_dataset(8),
            MSELoss::new(),
            SimpleTrainingConfig::default(),
        );
        assert_eq!(trainer.state.epoch, 0);
        assert!(!trainer.state.is_training);
    }

    #[test]
    fn test_simple_trainer_builder() {
        let result = SimpleTrainerBuilder::new()
            .model(LinearModel::new(2))
            .train_dataset(regression_dataset(8))
            .loss_function(MSELoss::new())
            .learning_rate(0.001)
            .batch_size(16)
            .num_epochs(5)
            .with_logging()
            .build();

        assert!(result.is_ok());
        let trainer = result.expect("operation failed in test");
        assert_eq!(trainer.config.learning_rate, 0.001);
        assert_eq!(trainer.config.batch_size, 16);
        assert_eq!(trainer.config.num_epochs, 5);
    }

    // ── Real training loop ────────────────────────────────────────────────────

    #[test]
    fn test_training_converges_on_a_tiny_regression_problem() {
        // Regression: `train_step` used to return `1 / (1 + step * 0.001)` regardless of the
        // model, the data and the loss. A synthetic curve like that is monotone by
        // construction; this test instead requires that the *model* actually learns, which
        // it can only do if forward/backward/update are real.
        let dataset = regression_dataset(64);
        let config = SimpleTrainingConfig {
            learning_rate: 0.2,
            batch_size: 8,
            num_epochs: 40,
            eval_steps: None,
            save_steps: None,
            logging_steps: 1_000_000,
            max_grad_norm: None,
            ..SimpleTrainingConfig::default()
        };
        let mut trainer =
            SimpleTrainer::new(LinearModel::new(2), dataset.clone(), MSELoss::new(), config);

        let results = trainer.train().expect("training failed");
        assert_eq!(results.history.len(), 40);

        let first = results.history.first().expect("first epoch").train_loss;
        let last = results.history.last().expect("last epoch").train_loss;
        assert!(
            last < first * 0.5,
            "loss must fall substantially: {first} -> {last}"
        );
        assert!(last < 0.05, "final loss should approach zero, got {last}");

        // The learned parameters must approach the generating coefficients.
        let model = trainer.get_model();
        let guard = model.read().expect("model lock");
        assert!(
            (guard.weights[0] - 3.0).abs() < 0.5,
            "w0 should approach 3.0, got {}",
            guard.weights[0]
        );
        assert!(
            (guard.weights[1] + 2.0).abs() < 0.5,
            "w1 should approach -2.0, got {}",
            guard.weights[1]
        );
    }

    #[test]
    fn test_steps_per_epoch_follows_the_dataset() {
        // Regression: the loop length was hardcoded to 100 regardless of the data.
        let config = SimpleTrainingConfig {
            batch_size: 4,
            num_epochs: 1,
            eval_steps: None,
            save_steps: None,
            logging_steps: 1_000_000,
            ..SimpleTrainingConfig::default()
        };
        let mut trainer = SimpleTrainer::new(
            LinearModel::new(2),
            regression_dataset(10),
            MSELoss::new(),
            config,
        );
        let results = trainer.train().expect("training failed");
        // 10 samples / batch 4 => 3 steps (4 + 4 + 2), not 100.
        assert_eq!(
            results.total_steps, 3,
            "step count must come from the dataset"
        );
    }

    #[test]
    fn test_no_eval_dataset_yields_no_eval_loss() {
        // Regression: `evaluate()` returned `Ok(0.0)` and the step loop wrote that into
        // `state.eval_loss`, so `TrainingResults.final_eval_loss` reported a fabricated 0.0.
        let config = SimpleTrainingConfig {
            batch_size: 4,
            num_epochs: 1,
            eval_steps: Some(1),
            save_steps: None,
            logging_steps: 1_000_000,
            ..SimpleTrainingConfig::default()
        };
        let mut trainer = SimpleTrainer::new(
            LinearModel::new(2),
            regression_dataset(8),
            MSELoss::new(),
            config,
        );
        let results = trainer.train().expect("training failed");
        assert!(
            results.final_eval_loss.is_none(),
            "without an eval dataset there is no eval loss, got {:?}",
            results.final_eval_loss
        );
    }

    #[test]
    fn test_eval_loss_is_computed_from_the_eval_dataset() {
        let config = SimpleTrainingConfig {
            learning_rate: 0.2,
            batch_size: 8,
            num_epochs: 20,
            eval_steps: None,
            save_steps: None,
            logging_steps: 1_000_000,
            max_grad_norm: None,
            ..SimpleTrainingConfig::default()
        };
        let mut trainer = SimpleTrainer::new(
            LinearModel::new(2),
            regression_dataset(32),
            MSELoss::new(),
            config,
        )
        .with_eval_dataset(regression_dataset(16));

        let results = trainer.train().expect("training failed");
        let eval_loss = results.final_eval_loss.expect("eval loss must be reported");
        assert!(eval_loss.is_finite() && eval_loss >= 0.0);
        assert!(
            eval_loss < 0.1,
            "eval loss should be small after training: {eval_loss}"
        );
    }

    #[test]
    fn test_empty_training_dataset_is_an_error() {
        let empty = TensorDataset::new(
            Tensor::from_vec(Vec::<f32>::new(), &[0, 2]).expect("inputs"),
            Tensor::from_vec(Vec::<f32>::new(), &[0, 1]).expect("targets"),
        )
        .expect("dataset");
        let mut trainer = SimpleTrainer::new(
            LinearModel::new(2),
            empty,
            MSELoss::new(),
            SimpleTrainingConfig::default(),
        );
        assert!(
            trainer.train().is_err(),
            "training on an empty dataset must error, not report a synthetic loss"
        );
    }

    #[test]
    fn test_tensor_dataset_rejects_misaligned_tensors() {
        let inputs = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).expect("inputs");
        let targets = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3, 1]).expect("targets");
        assert!(TensorDataset::new(inputs, targets).is_err());
    }

    #[test]
    fn test_logging_callback() {
        let mut callback = LoggingCallback::new(LogLevel::Info);
        let state = TrainingState::default();
        let config = SimpleTrainingConfig::default();

        // Test that callbacks don't panic
        assert!(callback.on_train_begin(&state, &config).is_ok());
        assert!(callback.on_epoch_begin(1, &state).is_ok());
        assert!(callback.on_epoch_end(1, &state).is_ok());
        assert!(callback.on_train_end(&state).is_ok());
    }

    #[test]
    fn test_early_stopping_callback() {
        let mut callback =
            EarlyStoppingCallback::new("eval_loss".to_string(), 3, 0.01, EarlyStoppingMode::Min);

        let mut state = TrainingState::default();
        state.metrics.insert("eval_loss".to_string(), 0.5);

        // First evaluation - should set best value
        assert!(callback.on_evaluate_end(&state).is_ok());
        assert_eq!(callback.best_value, Some(0.5));
        assert_eq!(callback.patience_counter, 0);

        // No improvement
        state.metrics.insert("eval_loss".to_string(), 0.6);
        assert!(callback.on_evaluate_end(&state).is_ok());
        assert_eq!(callback.patience_counter, 1);
    }

    #[test]
    fn test_metrics_callback() {
        let mut callback = MetricsCallback::new(vec!["loss".to_string(), "accuracy".to_string()]);

        let mut logs = HashMap::new();
        logs.insert("loss".to_string(), 0.5);
        logs.insert("accuracy".to_string(), 0.9);
        logs.insert("other_metric".to_string(), 0.1); // Should be ignored

        let state = TrainingState::default();
        assert!(callback.on_log(&logs, &state).is_ok());

        assert_eq!(callback.get_history("loss"), Some(&vec![0.5]));
        assert_eq!(callback.get_history("accuracy"), Some(&vec![0.9]));
        assert_eq!(callback.get_history("other_metric"), None);
    }

    #[test]
    fn test_config_defaults() {
        let config = SimpleTrainingConfig::default();
        assert_eq!(config.learning_rate, 3e-4);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.num_epochs, 3);
        assert_eq!(config.logging_steps, 100);
        assert_eq!(config.warmup_steps, 500);
        assert_eq!(config.seed, Some(42));
    }
}
