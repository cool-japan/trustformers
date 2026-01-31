use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use crate::losses::Loss;
use crate::metrics::{Metric, MetricCollection};

/// Simplified trainer interface for easy model training
#[allow(dead_code)]
pub struct SimpleTrainer<M, D, L> {
    model: Arc<RwLock<M>>,
    #[allow(dead_code)]
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
        println!(
            "🚀 Starting training with config: learning_rate={}, batch_size={}, epochs={}",
            config.learning_rate, config.batch_size, config.num_epochs
        );
        Ok(())
    }

    fn on_epoch_begin(&mut self, epoch: u32, _state: &TrainingState) -> Result<()> {
        println!("📚 Starting epoch {}", epoch);
        Ok(())
    }

    fn on_epoch_end(&mut self, epoch: u32, state: &TrainingState) -> Result<()> {
        let eval_info = if let Some(eval_loss) = state.eval_loss {
            format!(", eval_loss: {:.4}", eval_loss)
        } else {
            String::new()
        };

        println!(
            "✅ Epoch {} completed - train_loss: {:.4}{}",
            epoch, state.train_loss, eval_info
        );
        Ok(())
    }

    fn on_log(&mut self, logs: &HashMap<String, f64>, state: &TrainingState) -> Result<()> {
        if matches!(self.log_level, LogLevel::Debug) {
            println!("📊 Step {} - {:?}", state.global_step, logs);
        }
        Ok(())
    }

    fn on_train_end(&mut self, state: &TrainingState) -> Result<()> {
        if let Some(start_time) = state.start_time {
            let duration = start_time.elapsed();
            println!("🎉 Training completed in {:.2}s", duration.as_secs_f64());
        }
        Ok(())
    }
}

/// Progress bar callback
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
                println!("🎯 New best {}: {:.4}", self.monitor, current_value);
            } else {
                self.patience_counter += 1;
                if self.patience_counter >= self.patience {
                    println!(
                        "⏹️  Early stopping triggered. No improvement in {} for {} epochs",
                        self.monitor, self.patience
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
            println!("💾 Saving checkpoint to {}", checkpoint_path);
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
    M: Send + Sync,
    D: Clone,
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
                println!("Training stopped early at epoch {}", epoch);
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
            training_time: self.state.start_time.unwrap().elapsed(),
            history: training_history,
        })
    }

    fn train_epoch(&mut self) -> Result<EpochResult> {
        let mut total_loss = 0.0;
        let mut step_count = 0;

        // Simplified training loop (in practice would iterate over actual batches)
        let steps_per_epoch = 100; // Placeholder

        for step in 1..=steps_per_epoch {
            self.state.global_step += 1;

            // Call step begin callbacks
            for callback in &mut self.callbacks {
                callback.on_step_begin(step, &self.state)?;
            }

            // Simulate training step
            let step_loss = self.train_step()?;
            total_loss += step_loss;
            step_count += 1;

            // Logging
            if self.state.global_step % self.config.logging_steps == 0 {
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
                if self.state.global_step % eval_steps == 0 {
                    self.evaluate()?;
                }
            }

            // Saving
            if let Some(save_steps) = self.config.save_steps {
                if self.state.global_step % save_steps == 0 {
                    for callback in &mut self.callbacks {
                        callback.on_save(&self.state)?;
                    }
                }
            }

            // Call step end callbacks
            for callback in &mut self.callbacks {
                callback.on_step_end(step, &self.state)?;
            }
        }

        let avg_train_loss = total_loss / step_count as f64;

        // Run evaluation at end of epoch if we have eval dataset
        let eval_loss = if self.eval_dataset.is_some() { Some(self.evaluate()?) } else { None };

        Ok(EpochResult {
            epoch: self.state.epoch,
            train_loss: avg_train_loss,
            eval_loss,
            metrics: self.state.metrics.clone(),
        })
    }

    fn train_step(&mut self) -> Result<f64> {
        // Simplified training step - in practice would:
        // 1. Get batch from dataset
        // 2. Forward pass
        // 3. Compute loss
        // 4. Backward pass
        // 5. Update weights

        // Simulate decreasing loss
        let loss = 1.0 / (1.0 + self.state.global_step as f64 * 0.001);
        Ok(loss)
    }

    fn evaluate(&mut self) -> Result<f64> {
        if self.eval_dataset.is_none() {
            return Ok(0.0);
        }

        // Call evaluate begin callbacks
        for callback in &mut self.callbacks {
            callback.on_evaluate_begin(&self.state)?;
        }

        // Simplified evaluation - in practice would:
        // 1. Set model to eval mode
        // 2. Iterate over eval dataset
        // 3. Compute metrics
        // 4. Set model back to train mode

        let eval_loss = 0.5 / (1.0 + self.state.epoch as f64 * 0.1);

        // Update state
        self.state.eval_loss = Some(eval_loss);

        // Call evaluate end callbacks
        for callback in &mut self.callbacks {
            callback.on_evaluate_end(&self.state)?;
        }

        Ok(eval_loss)
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
    M: Send + Sync,
    D: Clone,
    L: Loss + Send + Sync,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<M, D, L> SimpleTrainerBuilder<M, D, L>
where
    M: Send + Sync,
    D: Clone,
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

    #[derive(Clone)]
    struct DummyDataset;

    struct DummyModel;

    #[test]
    fn test_simple_trainer_creation() {
        let model = DummyModel;
        let dataset = DummyDataset;
        let loss_fn = MSELoss::new();
        let config = SimpleTrainingConfig::default();

        let trainer = SimpleTrainer::new(model, dataset, loss_fn, config);
        assert_eq!(trainer.state.epoch, 0);
        assert!(!trainer.state.is_training);
    }

    #[test]
    fn test_simple_trainer_builder() {
        let result = SimpleTrainerBuilder::new()
            .model(DummyModel)
            .train_dataset(DummyDataset)
            .loss_function(MSELoss::new())
            .learning_rate(0.001)
            .batch_size(16)
            .num_epochs(5)
            .with_logging()
            .build();

        assert!(result.is_ok());
        let trainer = result.unwrap();
        assert_eq!(trainer.config.learning_rate, 0.001);
        assert_eq!(trainer.config.batch_size, 16);
        assert_eq!(trainer.config.num_epochs, 5);
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
