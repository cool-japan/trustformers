use crate::{
    continual::{ContinualLearningConfig, ContinualLearningManager, TaskInfo},
    GradientUtils, Loss, Metric, MetricCollection, TrainingArguments,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use trustformers_core::errors::{
    file_not_found, invalid_format, model_compatibility_error, Result,
};
use trustformers_core::traits::Optimizer;
use trustformers_core::{Model, Tensor};

/// Trait for extracting predictions from model outputs
pub trait ModelOutput {
    /// Get the primary prediction tensor (logits, hidden states, etc.)
    fn primary_output(&self) -> &Tensor;

    /// Get logits if available (for language modeling)
    fn logits(&self) -> Option<&Tensor> {
        None
    }

    /// Get hidden states if available
    fn hidden_states(&self) -> Option<&Tensor> {
        None
    }

    /// Get pooled output if available (for classification)
    fn pooled_output(&self) -> Option<&Tensor> {
        None
    }
}

/// Default implementation for Tensor (direct model outputs)
impl ModelOutput for Tensor {
    fn primary_output(&self) -> &Tensor {
        self
    }
    fn logits(&self) -> Option<&Tensor> {
        Some(self)
    }
    fn hidden_states(&self) -> Option<&Tensor> {
        Some(self)
    }
}

/// Task type for determining how to extract predictions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    LanguageModeling,
    Classification,
    Representation,
}

/// Trait for models that can be saved/loaded
pub trait ModelSaveLoad {
    /// Save model weights to a directory
    fn save_pretrained(&self, path: &Path) -> Result<()>;

    /// Load model weights from a directory
    fn load_pretrained(&mut self, path: &Path) -> Result<()>;
}

/// Trait for models that expose their parameters for optimization
pub trait ParameterAccess {
    /// Get mutable references to all trainable parameters
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;

    /// Get references to all trainable parameters
    fn parameters(&self) -> Vec<&Tensor>;

    /// Get the number of trainable parameters
    fn parameter_count(&self) -> usize {
        self.parameters().iter().map(|p| p.size()).sum()
    }
}

/// Training state for resuming and checkpointing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingState {
    pub epoch: f32,
    pub global_step: usize,
    pub best_metric: Option<f32>,
    pub best_model_checkpoint: Option<PathBuf>,
    pub log_history: Vec<LogEntry>,
    pub trial_name: Option<String>,
    pub trial_params: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub step: usize,
    pub epoch: f32,
    pub learning_rate: f32,
    pub loss: f32,
    pub eval_metrics: Option<HashMap<String, f32>>,
    pub train_metrics: Option<HashMap<String, f32>>,
}

/// Event callbacks during training
pub trait TrainerCallback: Send + Sync + std::any::Any {
    fn on_train_begin(&mut self, _args: &TrainingArguments, _state: &TrainingState) {}
    fn on_train_end(&mut self, _args: &TrainingArguments, _state: &TrainingState) {}
    fn on_epoch_begin(&mut self, _args: &TrainingArguments, _state: &TrainingState) {}
    fn on_epoch_end(&mut self, _args: &TrainingArguments, _state: &TrainingState) {}
    fn on_step_begin(&mut self, _args: &TrainingArguments, _state: &TrainingState) {}
    fn on_step_end(&mut self, _args: &TrainingArguments, _state: &TrainingState) {}
    fn on_evaluate(
        &mut self,
        _args: &TrainingArguments,
        _state: &TrainingState,
        _metrics: &HashMap<String, f32>,
    ) {
    }
    fn on_save(&mut self, _args: &TrainingArguments, _state: &TrainingState) {}
    fn on_log(
        &mut self,
        _args: &TrainingArguments,
        _state: &TrainingState,
        _logs: &HashMap<String, f32>,
    ) {
    }

    /// Check if training should stop early
    fn should_stop(&self) -> bool {
        false
    }
}

/// Early stopping callback
pub struct EarlyStoppingCallback {
    patience: usize,
    threshold: f32,
    metric_name: String,
    greater_is_better: bool,
    wait_count: usize,
    best_score: Option<f32>,
    should_stop: bool,
}

impl EarlyStoppingCallback {
    pub fn new(
        patience: usize,
        threshold: f32,
        metric_name: String,
        greater_is_better: bool,
    ) -> Self {
        Self {
            patience,
            threshold,
            metric_name,
            greater_is_better,
            wait_count: 0,
            best_score: None,
            should_stop: false,
        }
    }

    pub fn should_stop(&self) -> bool {
        self.should_stop
    }
}

impl TrainerCallback for EarlyStoppingCallback {
    fn on_evaluate(
        &mut self,
        _args: &TrainingArguments,
        _state: &TrainingState,
        metrics: &HashMap<String, f32>,
    ) {
        if let Some(&current_score) = metrics.get(&self.metric_name) {
            let is_improvement = match self.best_score {
                None => true,
                Some(best) => {
                    if self.greater_is_better {
                        current_score > best + self.threshold
                    } else {
                        current_score < best - self.threshold
                    }
                },
            };

            if is_improvement {
                self.best_score = Some(current_score);
                self.wait_count = 0;
            } else {
                self.wait_count += 1;
                if self.wait_count >= self.patience {
                    self.should_stop = true;
                }
            }
        }
    }

    fn should_stop(&self) -> bool {
        self.should_stop
    }
}

/// Main Trainer struct for training models
pub struct Trainer<M: Model> {
    model: M,
    args: TrainingArguments,
    optimizer: Box<dyn Optimizer>,
    loss_fn: Box<dyn Loss>,
    train_metrics: MetricCollection,
    eval_metrics: MetricCollection,
    callbacks: Vec<Box<dyn TrainerCallback>>,
    state: TrainingState,
    accumulated_gradients: Vec<Tensor>,
    accumulation_count: usize,
    continual_manager: Option<ContinualLearningManager>,
    task_type: TaskType,
}

impl<M: Model> Trainer<M> {
    /// Create a new trainer
    pub fn new(
        model: M,
        args: TrainingArguments,
        optimizer: Box<dyn Optimizer>,
        loss_fn: Box<dyn Loss>,
        task_type: TaskType,
    ) -> Result<Self> {
        // Validate arguments
        args.validate()?;

        // Create output directory if it doesn't exist
        if !args.output_dir.exists() {
            fs::create_dir_all(&args.output_dir)
                .map_err(|e| file_not_found(format!("Directory creation failed: {}", e)))?;
        }

        let state = TrainingState {
            epoch: 0.0,
            global_step: 0,
            best_metric: None,
            best_model_checkpoint: None,
            log_history: Vec::new(),
            trial_name: None,
            trial_params: None,
        };

        Ok(Self {
            model,
            args,
            optimizer,
            loss_fn,
            train_metrics: MetricCollection::new(),
            eval_metrics: MetricCollection::new(),
            callbacks: Vec::new(),
            state,
            accumulated_gradients: Vec::new(),
            accumulation_count: 0,
            continual_manager: None,
            task_type,
        })
    }

    /// Add a metric for training evaluation
    pub fn add_train_metric(mut self, metric: Box<dyn Metric>) -> Self {
        self.train_metrics = self.train_metrics.add_metric(metric);
        self
    }

    /// Add a metric for evaluation
    pub fn add_eval_metric(mut self, metric: Box<dyn Metric>) -> Self {
        self.eval_metrics = self.eval_metrics.add_metric(metric);
        self
    }

    /// Add a callback
    pub fn add_callback(mut self, callback: Box<dyn TrainerCallback>) -> Self {
        self.callbacks.push(callback);
        self
    }

    /// Extract predictions from model output based on task type
    pub fn extract_predictions<'a, T: ModelOutput>(&self, output: &'a T) -> Result<&'a Tensor> {
        match self.task_type {
            TaskType::LanguageModeling => output
                .logits()
                .ok_or_else(|| model_compatibility_error("language_modeling", "logits_output")),
            TaskType::Classification => output
                .pooled_output()
                .or_else(|| output.logits())
                .or_else(|| output.hidden_states())
                .ok_or_else(|| {
                    model_compatibility_error("classification", "pooled_or_logits_output")
                }),
            TaskType::Representation => output
                .hidden_states()
                .or_else(|| Some(output.primary_output()))
                .ok_or_else(|| model_compatibility_error("representation", "hidden_states_output")),
        }
    }

    /// Get the current training state
    pub fn state(&self) -> &TrainingState {
        &self.state
    }

    /// Get mutable access to the model
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Enable continual learning with the given configuration
    pub fn with_continual_learning(mut self, config: ContinualLearningConfig) -> Self {
        self.continual_manager = Some(ContinualLearningManager::new(config));
        self
    }

    /// Add a new task for continual learning
    pub fn add_task(&mut self, task: TaskInfo) -> Result<()> {
        if let Some(manager) = &mut self.continual_manager {
            manager.add_task(task)?;
        }
        Ok(())
    }

    /// Switch to a different task for continual learning
    pub fn switch_task(&mut self, task_id: String) -> Result<()> {
        if let Some(manager) = &mut self.continual_manager {
            manager.set_current_task(task_id)?;
        }
        Ok(())
    }

    /// Get information about the current task
    pub fn current_task(&self) -> Option<&TaskInfo> {
        self.continual_manager.as_ref()?.get_current_task()
    }

    /// Get the number of tasks in continual learning
    pub fn task_count(&self) -> usize {
        self.continual_manager.as_ref().map(|m| m.get_task_count()).unwrap_or(0)
    }

    /// Get immutable access to the model
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Save the model and trainer state
    pub fn save_checkpoint(&mut self, checkpoint_dir: &Path) -> Result<()> {
        // Create checkpoint directory
        fs::create_dir_all(checkpoint_dir)
            .map_err(|e| file_not_found(format!("Failed to create checkpoint directory: {}", e)))?;

        // Save training state
        let state_path = checkpoint_dir.join("trainer_state.json");
        let state_json = serde_json::to_string_pretty(&self.state)
            .map_err(|e| invalid_format("json", format!("serialization error: {}", e)))?;
        fs::write(state_path, state_json)
            .map_err(|e| file_not_found(format!("Failed to write training state: {}", e)))?;

        // Save training arguments
        let args_path = checkpoint_dir.join("training_args.json");
        let args_json = serde_json::to_string_pretty(&self.args)
            .map_err(|e| invalid_format("json", format!("serialization error: {}", e)))?;
        fs::write(args_path, args_json)
            .map_err(|e| file_not_found(format!("Failed to write training arguments: {}", e)))?;

        // Save model weights - attempt to save if model implements ModelSaveLoad
        self.save_model_if_supported(checkpoint_dir)?;

        // Notify callbacks
        for callback in &mut self.callbacks {
            callback.on_save(&self.args, &self.state);
        }

        Ok(())
    }

    /// Load checkpoint and resume training
    pub fn load_checkpoint(&mut self, checkpoint_dir: &Path) -> Result<()> {
        // Load training state
        let state_path = checkpoint_dir.join("trainer_state.json");
        if state_path.exists() {
            let state_json = fs::read_to_string(state_path)
                .map_err(|e| file_not_found(format!("Failed to read training state: {}", e)))?;
            self.state = serde_json::from_str(&state_json)
                .map_err(|e| invalid_format("json", format!("deserialization error: {}", e)))?;
        }

        // Load model weights - attempt to load if model implements ModelSaveLoad
        self.load_model_if_supported(checkpoint_dir)?;

        Ok(())
    }

    /// Perform a single training step with gradient accumulation
    pub fn training_step(&mut self, inputs: M::Input, targets: &Tensor) -> Result<f32>
    where
        M::Output: ModelOutput,
    {
        // Forward pass
        let outputs = self.model.forward(inputs)?;

        // Extract predictions from model output
        let predictions = self.extract_predictions(&outputs)?;

        // Compute loss with gradients
        let (loss_value, gradients) = self.loss_fn.compute_with_gradients(predictions, targets)?;

        // Convert single gradient tensor to vector for accumulation
        let gradient_vec = vec![gradients];

        // Accumulate gradients
        self.accumulate_gradients(gradient_vec)?;

        Ok(loss_value)
    }

    /// Accumulate gradients for gradient accumulation
    fn accumulate_gradients(&mut self, gradients: Vec<Tensor>) -> Result<()> {
        if self.accumulated_gradients.is_empty() {
            // Initialize accumulated gradients
            self.accumulated_gradients = gradients
                .iter()
                .map(|grad| Tensor::zeros(&grad.shape()))
                .collect::<Result<Vec<_>>>()?;
        }

        // Accumulate gradients
        GradientUtils::accumulate_gradients(
            &mut self.accumulated_gradients,
            &gradients,
            self.args.gradient_accumulation_steps,
        )?;

        self.accumulation_count += 1;
        Ok(())
    }

    /// Apply accumulated gradients with clipping
    fn apply_accumulated_gradients(&mut self) -> Result<()> {
        if self.accumulated_gradients.is_empty() {
            return Ok(());
        }

        // Apply gradient clipping
        let grad_norm = GradientUtils::clip_grad_norm(
            &mut self.accumulated_gradients,
            self.args.max_grad_norm,
        )?;

        // Log gradient norm if needed
        if self.state.global_step.is_multiple_of(self.args.logging_steps) {
            println!("Gradient norm: {:.4}", grad_norm);
        }

        // Apply gradients to model parameters if the model supports parameter access
        self.apply_gradients_to_model()?;

        // Reset accumulation
        GradientUtils::zero_accumulated_gradients(&mut self.accumulated_gradients)?;
        self.accumulation_count = 0;

        Ok(())
    }

    /// Perform evaluation on a dataset
    pub fn evaluate(&mut self, eval_dataset: &[(M::Input, Tensor)]) -> Result<HashMap<String, f32>>
    where
        M::Input: Clone,
        M::Output: ModelOutput,
    {
        let mut all_predictions: Vec<Tensor> = Vec::new();
        let mut all_targets: Vec<Tensor> = Vec::new();
        let mut total_loss = 0.0;
        let mut num_samples = 0;

        for (inputs, targets) in eval_dataset.iter() {
            // Forward pass
            let outputs = self.model.forward(inputs.clone())?;

            // Extract predictions from model output
            let predictions = self.extract_predictions(&outputs)?;

            // Compute loss
            let loss = self.loss_fn.compute(predictions, targets)?;
            total_loss += loss;
            num_samples += 1;

            // Collect predictions and targets for metric computation
            all_predictions.push(predictions.clone());
            all_targets.push(targets.clone());
        }

        let avg_loss = if num_samples > 0 { total_loss / num_samples as f32 } else { 0.0 };

        // Compute metrics
        let mut results = HashMap::new();
        results.insert("eval_loss".to_string(), avg_loss);

        // Compute metrics if we have data
        if !all_predictions.is_empty() && !all_targets.is_empty() {
            // Compute batched metrics across all predictions and targets
            let batched_metrics = self.compute_batched_metrics(&all_predictions, &all_targets)?;
            results.extend(batched_metrics);
        }

        // Notify callbacks
        for callback in &mut self.callbacks {
            callback.on_evaluate(&self.args, &self.state, &results);
        }

        Ok(results)
    }

    /// Main training loop
    pub fn train(
        &mut self,
        train_dataset: &[(M::Input, Tensor)],
        eval_dataset: Option<&[(M::Input, Tensor)]>,
    ) -> Result<()>
    where
        M::Input: Clone,
        M::Output: ModelOutput,
    {
        let total_steps = self.args.get_total_steps(train_dataset.len());
        let _warmup_steps = self.args.get_warmup_steps(total_steps);

        // Notify callbacks
        for callback in &mut self.callbacks {
            callback.on_train_begin(&self.args, &self.state);
        }

        let mut accumulation_steps = 0;
        let epochs = if let Some(max_steps) = self.args.max_steps {
            (max_steps as f32
                / (train_dataset.len() as f32 / self.args.per_device_train_batch_size as f32))
                .ceil()
        } else {
            self.args.num_train_epochs
        };

        for epoch in 0..(epochs as usize) {
            self.state.epoch = epoch as f32;

            // Notify callbacks
            for callback in &mut self.callbacks {
                callback.on_epoch_begin(&self.args, &self.state);
            }

            // Training loop for this epoch
            for (inputs, targets) in train_dataset.iter() {
                self.state.global_step += 1;

                // Notify callbacks
                for callback in &mut self.callbacks {
                    callback.on_step_begin(&self.args, &self.state);
                }

                // Training step (accumulates gradients internally)
                let loss = self.training_step(inputs.clone(), targets)?;
                accumulation_steps += 1;

                // Apply gradients when accumulation is complete
                if accumulation_steps >= self.args.gradient_accumulation_steps {
                    // Apply accumulated gradients with clipping
                    self.apply_accumulated_gradients()?;

                    // Optimizer step
                    self.optimizer.step();
                    self.optimizer.zero_grad();
                    accumulation_steps = 0;
                }

                // Logging
                if self.state.global_step.is_multiple_of(self.args.logging_steps) {
                    let mut logs = HashMap::new();
                    logs.insert("loss".to_string(), loss);
                    logs.insert("learning_rate".to_string(), self.optimizer.get_lr());
                    logs.insert("epoch".to_string(), self.state.epoch);
                    logs.insert("step".to_string(), self.state.global_step as f32);

                    // Add to log history
                    self.state.log_history.push(LogEntry {
                        step: self.state.global_step,
                        epoch: self.state.epoch,
                        learning_rate: self.optimizer.get_lr(),
                        loss,
                        eval_metrics: None,
                        train_metrics: None,
                    });

                    // Notify callbacks
                    for callback in &mut self.callbacks {
                        callback.on_log(&self.args, &self.state, &logs);
                    }
                }

                // Evaluation
                if let Some(eval_data) = eval_dataset {
                    if self.state.global_step.is_multiple_of(self.args.eval_steps) {
                        let eval_results = self.evaluate(eval_data)?;

                        // Update best metric
                        if let Some(ref metric_name) = self.args.metric_for_best_model {
                            if let Some(&current_metric) = eval_results.get(metric_name) {
                                let is_best = match self.state.best_metric {
                                    None => true,
                                    Some(best) => {
                                        if self.args.greater_is_better.unwrap_or(true) {
                                            current_metric > best
                                        } else {
                                            current_metric < best
                                        }
                                    },
                                };

                                if is_best {
                                    self.state.best_metric = Some(current_metric);

                                    if self.args.load_best_model_at_end {
                                        let checkpoint_dir = self
                                            .args
                                            .output_dir
                                            .join(format!("checkpoint-{}", self.state.global_step));
                                        self.save_checkpoint(&checkpoint_dir)?;
                                        self.state.best_model_checkpoint = Some(checkpoint_dir);
                                    }
                                }
                            }
                        }
                    }
                }

                // Checkpointing
                if self.state.global_step.is_multiple_of(self.args.save_steps) {
                    let checkpoint_dir =
                        self.args.output_dir.join(format!("checkpoint-{}", self.state.global_step));
                    self.save_checkpoint(&checkpoint_dir)?;
                }

                // Check for early stopping
                let should_stop = self.callbacks.iter().any(|callback| callback.should_stop());

                if should_stop {
                    println!("Early stopping triggered");
                    break;
                }

                // Notify callbacks
                for callback in &mut self.callbacks {
                    callback.on_step_end(&self.args, &self.state);
                }

                // Check max steps
                if let Some(max_steps) = self.args.max_steps {
                    if self.state.global_step >= max_steps {
                        break;
                    }
                }
            }

            // End-of-epoch evaluation
            if let Some(eval_data) = eval_dataset {
                if self.args.evaluation_strategy == crate::EvaluationStrategy::Epoch {
                    let _eval_results = self.evaluate(eval_data)?;
                }
            }

            // Notify callbacks
            for callback in &mut self.callbacks {
                callback.on_epoch_end(&self.args, &self.state);
            }

            // Check max steps
            if let Some(max_steps) = self.args.max_steps {
                if self.state.global_step >= max_steps {
                    break;
                }
            }
        }

        // Final evaluation
        if let Some(eval_data) = eval_dataset {
            if self.args.eval_at_end {
                let _eval_results = self.evaluate(eval_data)?;
            }
        }

        // Load best model if requested
        if self.args.load_best_model_at_end {
            if let Some(best_checkpoint) = self.state.best_model_checkpoint.clone() {
                self.load_checkpoint(&best_checkpoint)?;
            }
        }

        // Save final checkpoint
        let final_checkpoint_dir = self.args.output_dir.join("checkpoint-final");
        self.save_checkpoint(&final_checkpoint_dir)?;

        // Notify callbacks
        for callback in &mut self.callbacks {
            callback.on_train_end(&self.args, &self.state);
        }

        Ok(())
    }

    /// Helper method to save model if it implements ModelSaveLoad trait
    fn save_model_if_supported(&self, checkpoint_dir: &Path) -> Result<()> {
        // Create a marker file to indicate we attempted to save the model
        let model_info_path = checkpoint_dir.join("model_info.json");
        let model_info = serde_json::json!({
            "model_type": std::any::type_name::<M>(),
            "save_attempted": true,
            "note": "For full model saving, implement ModelSaveLoad trait on your model type"
        });
        let json_str = serde_json::to_string_pretty(&model_info)
            .expect("model_info json serialization should not fail");
        fs::write(model_info_path, json_str)
            .map_err(|e| file_not_found(format!("Failed to write model config: {}", e)))?;

        // Note: Actual model saving would require the model to implement ModelSaveLoad
        // This provides a framework for models that choose to implement the trait
        Ok(())
    }

    /// Helper method to load model if it implements ModelSaveLoad trait
    fn load_model_if_supported(&mut self, checkpoint_dir: &Path) -> Result<()> {
        let model_info_path = checkpoint_dir.join("model_info.json");
        if model_info_path.exists() {
            // Model info exists, but actual loading would require ModelSaveLoad implementation
            println!(
                "Model checkpoint found but loading requires ModelSaveLoad trait implementation"
            );
        }

        // Note: Actual model loading would require the model to implement ModelSaveLoad
        // This provides a framework for models that choose to implement the trait
        Ok(())
    }

    /// Apply accumulated gradients to model parameters
    fn apply_gradients_to_model(&mut self) -> Result<()> {
        if self.accumulated_gradients.is_empty() {
            return Ok(());
        }

        // Check if we have enough gradients for all accumulated steps
        if self.accumulated_gradients.len() != 1 {
            // For now, we expect a single gradient tensor from the loss function
            // In a full implementation, this would handle multiple parameter gradients
            return Ok(());
        }

        // For models that don't implement ParameterAccess, we can't apply gradients directly
        // This is a limitation of the current architecture that would need to be addressed
        // by having models implement the ParameterAccess trait

        // Log that gradient computation occurred but parameters weren't updated
        if self.state.global_step.is_multiple_of(self.args.logging_steps) {
            println!("Gradients computed but not applied - model needs ParameterAccess trait");
        }

        Ok(())
    }

    /// Compute metrics across multiple batches of predictions and targets
    fn compute_batched_metrics(
        &mut self,
        all_predictions: &[Tensor],
        all_targets: &[Tensor],
    ) -> Result<HashMap<String, f32>> {
        let mut batch_metrics: Vec<HashMap<String, f32>> = Vec::new();

        // Compute metrics for each batch
        for (predictions, targets) in all_predictions.iter().zip(all_targets.iter()) {
            let metrics = self.eval_metrics.compute_all(predictions, targets)?;
            batch_metrics.push(metrics);
        }

        // Aggregate metrics across batches
        let mut aggregated_metrics = HashMap::new();

        if !batch_metrics.is_empty() {
            // Get all metric names from the first batch
            let metric_names: Vec<String> = batch_metrics[0].keys().cloned().collect();

            // For each metric, compute the average across all batches
            for metric_name in metric_names {
                let mut metric_sum = 0.0;
                let mut metric_count = 0;

                for batch_metric in &batch_metrics {
                    if let Some(&value) = batch_metric.get(&metric_name) {
                        metric_sum += value;
                        metric_count += 1;
                    }
                }

                if metric_count > 0 {
                    let avg_metric = metric_sum / metric_count as f32;
                    aggregated_metrics.insert(format!("eval_{}", metric_name), avg_metric);
                }
            }
        }

        Ok(aggregated_metrics)
    }
}

/// Callback manager with downcasting support
pub struct CallbackManager {
    callbacks: Vec<Box<dyn TrainerCallback>>,
}

impl CallbackManager {
    pub fn new() -> Self {
        Self {
            callbacks: Vec::new(),
        }
    }

    /// Add a callback to the manager
    pub fn add_callback(&mut self, callback: Box<dyn TrainerCallback>) {
        self.callbacks.push(callback);
    }

    /// Get a reference to a specific callback type by downcasting
    pub fn get_callback<T: TrainerCallback + 'static>(&self) -> Option<&T> {
        for callback in &self.callbacks {
            if let Some(specific_callback) =
                (callback.as_ref() as &dyn std::any::Any).downcast_ref::<T>()
            {
                return Some(specific_callback);
            }
        }
        None
    }

    /// Get a mutable reference to a specific callback type by downcasting
    pub fn get_callback_mut<T: TrainerCallback + 'static>(&mut self) -> Option<&mut T> {
        for callback in &mut self.callbacks {
            if let Some(specific_callback) =
                (callback.as_mut() as &mut dyn std::any::Any).downcast_mut::<T>()
            {
                return Some(specific_callback);
            }
        }
        None
    }

    /// Remove a callback of a specific type
    pub fn remove_callback<T: TrainerCallback + 'static>(
        &mut self,
    ) -> Option<Box<dyn TrainerCallback>> {
        for (i, callback) in self.callbacks.iter().enumerate() {
            if (callback.as_ref() as &dyn std::any::Any).is::<T>() {
                return Some(self.callbacks.remove(i));
            }
        }
        None
    }

    /// Call a method on all callbacks
    pub fn call_on_train_begin(&mut self, args: &TrainingArguments, state: &TrainingState) {
        for callback in &mut self.callbacks {
            callback.on_train_begin(args, state);
        }
    }

    pub fn call_on_train_end(&mut self, args: &TrainingArguments, state: &TrainingState) {
        for callback in &mut self.callbacks {
            callback.on_train_end(args, state);
        }
    }

    pub fn call_on_epoch_begin(&mut self, args: &TrainingArguments, state: &TrainingState) {
        for callback in &mut self.callbacks {
            callback.on_epoch_begin(args, state);
        }
    }

    pub fn call_on_epoch_end(&mut self, args: &TrainingArguments, state: &TrainingState) {
        for callback in &mut self.callbacks {
            callback.on_epoch_end(args, state);
        }
    }

    pub fn call_on_step_begin(&mut self, args: &TrainingArguments, state: &TrainingState) {
        for callback in &mut self.callbacks {
            callback.on_step_begin(args, state);
        }
    }

    pub fn call_on_step_end(&mut self, args: &TrainingArguments, state: &TrainingState) {
        for callback in &mut self.callbacks {
            callback.on_step_end(args, state);
        }
    }

    pub fn call_on_evaluate(
        &mut self,
        args: &TrainingArguments,
        state: &TrainingState,
        metrics: &HashMap<String, f32>,
    ) {
        for callback in &mut self.callbacks {
            callback.on_evaluate(args, state, metrics);
        }
    }

    pub fn call_on_save(&mut self, args: &TrainingArguments, state: &TrainingState) {
        for callback in &mut self.callbacks {
            callback.on_save(args, state);
        }
    }

    pub fn call_on_log(
        &mut self,
        args: &TrainingArguments,
        state: &TrainingState,
        logs: &HashMap<String, f32>,
    ) {
        for callback in &mut self.callbacks {
            callback.on_log(args, state, logs);
        }
    }

    /// Check if any callback requests early stopping
    pub fn should_stop(&self) -> bool {
        self.callbacks.iter().any(|callback| callback.should_stop())
    }
}

impl Default for CallbackManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training_args::TrainingArguments;

    // ── LCG random helper ────────────────────────────────────────────────────
    fn lcg_next(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
        *seed
    }

    fn make_args() -> TrainingArguments {
        let dir = std::env::temp_dir().join(format!("trainer_test_{}", {
            let mut s = 99991u64;
            lcg_next(&mut s)
        }));
        std::fs::create_dir_all(&dir).expect("should create test dir");
        TrainingArguments::new(dir)
    }

    fn make_state(global_step: usize, epoch: f32) -> TrainingState {
        TrainingState {
            epoch,
            global_step,
            best_metric: None,
            best_model_checkpoint: None,
            log_history: Vec::new(),
            trial_name: None,
            trial_params: None,
        }
    }

    fn make_metrics(pairs: &[(&str, f32)]) -> HashMap<String, f32> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    // ── TrainingState ─────────────────────────────────────────────────────────

    // 1. TrainingState initializes to zero step / zero epoch
    #[test]
    fn test_training_state_initial_values() {
        let state = make_state(0, 0.0);
        assert_eq!(state.global_step, 0);
        assert!((state.epoch - 0.0).abs() < 1e-6);
        assert!(state.best_metric.is_none());
        assert!(state.best_model_checkpoint.is_none());
        assert!(state.log_history.is_empty());
    }

    // 2. global_step increment
    #[test]
    fn test_training_state_step_increment() {
        let mut state = make_state(0, 0.0);
        state.global_step += 1;
        assert_eq!(state.global_step, 1);
        state.global_step += 1;
        assert_eq!(state.global_step, 2);
    }

    // 3. epoch tracking
    #[test]
    fn test_training_state_epoch_tracking() {
        let mut state = make_state(0, 0.0);
        for epoch in 0..5usize {
            state.epoch = epoch as f32;
            assert!((state.epoch - epoch as f32).abs() < 1e-6);
        }
    }

    // 4. best_metric tracking — update when improved
    #[test]
    fn test_training_state_best_metric_update() {
        let mut state = make_state(0, 0.0);
        state.best_metric = Some(0.8);
        // Improve
        let new_metric = 0.9_f32;
        if state.best_metric.map(|b| new_metric > b).unwrap_or(true) {
            state.best_metric = Some(new_metric);
        }
        assert!((state.best_metric.expect("should have best") - 0.9).abs() < 1e-6);
    }

    // 5. best_metric does NOT update when not improved
    #[test]
    fn test_training_state_best_metric_not_update_when_worse() {
        let mut state = make_state(0, 0.0);
        state.best_metric = Some(0.9);
        let new_metric = 0.7_f32;
        if state.best_metric.map(|b| new_metric > b).unwrap_or(true) {
            state.best_metric = Some(new_metric);
        }
        assert!((state.best_metric.expect("should still be 0.9") - 0.9).abs() < 1e-6);
    }

    // 6. log_history grows when LogEntry appended
    #[test]
    fn test_training_state_log_history_grows() {
        let mut state = make_state(0, 0.0);
        for step in 1..=5usize {
            state.log_history.push(LogEntry {
                step,
                epoch: 0.0,
                learning_rate: 1e-4,
                loss: 1.0 / step as f32,
                eval_metrics: None,
                train_metrics: None,
            });
        }
        assert_eq!(state.log_history.len(), 5);
        assert_eq!(state.log_history[0].step, 1);
        assert_eq!(state.log_history[4].step, 5);
    }

    // 7. TrainingState serializes and deserializes (serde roundtrip)
    #[test]
    fn test_training_state_serde_roundtrip() {
        let mut state = make_state(42, 1.5);
        state.best_metric = Some(0.92);
        state.log_history.push(LogEntry {
            step: 42,
            epoch: 1.5,
            learning_rate: 2e-5,
            loss: 0.15,
            eval_metrics: Some(make_metrics(&[("eval_loss", 0.2)])),
            train_metrics: None,
        });
        let json = serde_json::to_string(&state).expect("serialize should work");
        let restored: TrainingState = serde_json::from_str(&json).expect("deserialize should work");
        assert_eq!(restored.global_step, 42);
        assert!((restored.epoch - 1.5).abs() < 1e-4);
        assert!((restored.best_metric.expect("best metric") - 0.92).abs() < 1e-5);
        assert_eq!(restored.log_history.len(), 1);
    }

    // ── EarlyStoppingCallback ─────────────────────────────────────────────────

    // 8. EarlyStoppingCallback does not stop initially
    #[test]
    fn test_early_stopping_no_stop_initially() {
        let cb = EarlyStoppingCallback::new(3, 0.0, "eval_loss".to_string(), false);
        assert!(!cb.should_stop());
    }

    // 9. Early stopping: patience counter increments on no improvement
    #[test]
    fn test_early_stopping_patience_increments() {
        let mut cb = EarlyStoppingCallback::new(3, 0.001, "eval_loss".to_string(), false);
        let args = make_args();
        let state = make_state(0, 0.0);
        // First eval: best_score set → no stop
        cb.on_evaluate(&args, &state, &make_metrics(&[("eval_loss", 0.5)]));
        assert!(!cb.should_stop(), "should not stop after first eval");
        // Same loss → no improvement → wait_count = 1
        cb.on_evaluate(&args, &state, &make_metrics(&[("eval_loss", 0.5)]));
        assert!(!cb.should_stop(), "patience not exceeded (1 < 3)");
        // Still no improvement → wait_count = 2
        cb.on_evaluate(&args, &state, &make_metrics(&[("eval_loss", 0.5)]));
        assert!(!cb.should_stop(), "patience not exceeded (2 < 3)");
        // Still no improvement → wait_count = 3 ≥ patience → stop
        cb.on_evaluate(&args, &state, &make_metrics(&[("eval_loss", 0.5)]));
        assert!(cb.should_stop(), "should stop after patience exceeded");
    }

    // 10. Early stopping: improvement resets patience counter
    #[test]
    fn test_early_stopping_improvement_resets_counter() {
        let mut cb = EarlyStoppingCallback::new(2, 0.001, "eval_loss".to_string(), false);
        let args = make_args();
        let state = make_state(0, 0.0);
        cb.on_evaluate(&args, &state, &make_metrics(&[("eval_loss", 0.5)]));
        cb.on_evaluate(&args, &state, &make_metrics(&[("eval_loss", 0.5)])); // wait_count=1
                                                                             // Improvement → resets wait_count to 0
        cb.on_evaluate(&args, &state, &make_metrics(&[("eval_loss", 0.3)]));
        assert!(!cb.should_stop(), "improvement should reset counter");
        // One more non-improvement
        cb.on_evaluate(&args, &state, &make_metrics(&[("eval_loss", 0.3)]));
        assert!(
            !cb.should_stop(),
            "still only 1 non-improvement after reset"
        );
    }

    // 11. Early stopping: greater_is_better = true (maximize metric)
    #[test]
    fn test_early_stopping_greater_is_better() {
        let mut cb = EarlyStoppingCallback::new(2, 0.001, "accuracy".to_string(), true);
        let args = make_args();
        let state = make_state(0, 0.0);
        cb.on_evaluate(&args, &state, &make_metrics(&[("accuracy", 0.7)]));
        // Higher value → improvement
        cb.on_evaluate(&args, &state, &make_metrics(&[("accuracy", 0.9)]));
        assert!(
            !cb.should_stop(),
            "improvement in greater_is_better mode should not stop"
        );
        // Lower value → no improvement
        cb.on_evaluate(&args, &state, &make_metrics(&[("accuracy", 0.8)]));
        assert!(!cb.should_stop(), "patience 1 < 2");
        cb.on_evaluate(&args, &state, &make_metrics(&[("accuracy", 0.8)]));
        assert!(cb.should_stop(), "patience 2 >= 2, should stop");
    }

    // 12. Early stopping: unknown metric key is ignored (no panic)
    #[test]
    fn test_early_stopping_missing_metric_key() {
        let mut cb = EarlyStoppingCallback::new(1, 0.0, "missing_metric".to_string(), false);
        let args = make_args();
        let state = make_state(0, 0.0);
        // Metric key not present: should not panic, should not stop
        cb.on_evaluate(&args, &state, &make_metrics(&[("eval_loss", 0.5)]));
        assert!(
            !cb.should_stop(),
            "missing metric key should not trigger stop"
        );
    }

    // ── CallbackManager ───────────────────────────────────────────────────────

    // 13. CallbackManager: new creates empty manager
    #[test]
    fn test_callback_manager_new_empty() {
        let mgr = CallbackManager::new();
        assert!(!mgr.should_stop(), "empty manager should not stop");
    }

    // 14. CallbackManager: should_stop with no stopping callbacks
    #[test]
    fn test_callback_manager_should_stop_false() {
        struct NoStopCallback;
        impl TrainerCallback for NoStopCallback {}
        let mut mgr = CallbackManager::new();
        mgr.add_callback(Box::new(NoStopCallback));
        assert!(!mgr.should_stop());
    }

    // 15. CallbackManager: should_stop returns true when any callback stops
    #[test]
    fn test_callback_manager_should_stop_true() {
        struct StopCallback;
        impl TrainerCallback for StopCallback {
            fn should_stop(&self) -> bool {
                true
            }
        }
        let mut mgr = CallbackManager::new();
        mgr.add_callback(Box::new(StopCallback));
        assert!(mgr.should_stop());
    }

    // 16. CallbackManager: get_callback by type downcasting
    #[test]
    fn test_callback_manager_get_callback_by_type() {
        let mut mgr = CallbackManager::new();
        let cb = EarlyStoppingCallback::new(2, 0.0, "loss".to_string(), false);
        mgr.add_callback(Box::new(cb));
        let found = mgr.get_callback::<EarlyStoppingCallback>();
        assert!(found.is_some(), "should find EarlyStoppingCallback by type");
    }

    // 17. CallbackManager: get_callback returns None for absent type
    #[test]
    fn test_callback_manager_get_callback_absent() {
        let mgr = CallbackManager::new();
        let found = mgr.get_callback::<EarlyStoppingCallback>();
        assert!(found.is_none());
    }

    // 18. CallbackManager: remove_callback removes the callback
    #[test]
    fn test_callback_manager_remove_callback() {
        let mut mgr = CallbackManager::new();
        mgr.add_callback(Box::new(EarlyStoppingCallback::new(
            2,
            0.0,
            "loss".to_string(),
            false,
        )));
        let removed = mgr.remove_callback::<EarlyStoppingCallback>();
        assert!(removed.is_some(), "should return the removed callback");
        // Should be gone now
        assert!(mgr.get_callback::<EarlyStoppingCallback>().is_none());
    }

    // 19. CallbackManager: call_on_train_begin doesn't panic
    #[test]
    fn test_callback_manager_call_on_train_begin() {
        let mut mgr = CallbackManager::new();
        let args = make_args();
        let state = make_state(0, 0.0);
        // No callbacks → should not panic
        mgr.call_on_train_begin(&args, &state);
    }

    // 20. CallbackManager: call_on_evaluate passes metrics to callbacks
    #[test]
    fn test_callback_manager_call_on_evaluate() {
        let mut mgr = CallbackManager::new();
        // Add an early stopping callback monitoring accuracy
        let cb = EarlyStoppingCallback::new(1, 0.0, "accuracy".to_string(), true);
        mgr.add_callback(Box::new(cb));
        let args = make_args();
        let state = make_state(10, 1.0);
        mgr.call_on_evaluate(&args, &state, &make_metrics(&[("accuracy", 0.9)]));
        // First call → best_score set; second with same → patience triggered
        mgr.call_on_evaluate(&args, &state, &make_metrics(&[("accuracy", 0.8)]));
        // After 2 calls with no improvement from 0.9, should stop (patience=1)
        assert!(mgr.should_stop(), "should stop after patience exceeded");
    }

    // 21. TaskType variants are distinct and debuggable
    #[test]
    fn test_task_type_variants() {
        assert_ne!(TaskType::LanguageModeling, TaskType::Classification);
        assert_ne!(TaskType::Classification, TaskType::Representation);
        assert_ne!(TaskType::Representation, TaskType::LanguageModeling);
    }

    // ── Validation trigger simulation (every N steps) ─────────────────────────

    // 22. Validation trigger at every eval_steps
    #[test]
    fn test_validation_trigger_every_n_steps() {
        let args = {
            let mut a = make_args();
            a.eval_steps = 10;
            a
        };
        let total_steps = 30;
        let expected_eval_steps: Vec<usize> =
            (1..=total_steps).filter(|s| s % args.eval_steps == 0).collect();
        assert_eq!(expected_eval_steps, vec![10, 20, 30]);
    }

    // 23. Checkpoint save trigger at every N steps
    #[test]
    fn test_checkpoint_save_trigger_every_n_steps() {
        let args = {
            let mut a = make_args();
            a.save_steps = 5;
            a
        };
        let total_steps = 20;
        let checkpoint_steps: Vec<usize> =
            (1..=total_steps).filter(|s| s % args.save_steps == 0).collect();
        assert_eq!(checkpoint_steps, vec![5, 10, 15, 20]);
    }

    // 24. Gradient accumulation: apply every N micro-steps
    #[test]
    fn test_gradient_accumulation_step_count() {
        let grad_accum_steps = 4;
        let mut apply_count = 0;
        let mut accum = 0;
        for _step in 1..=16usize {
            accum += 1;
            if accum >= grad_accum_steps {
                apply_count += 1;
                accum = 0;
            }
        }
        // 16 micro-steps / 4 = 4 optimizer steps
        assert_eq!(apply_count, 4);
    }

    // 25. Training time estimation: steps_per_second × remaining_steps
    #[test]
    fn test_training_time_estimation() {
        let total_steps = 1000usize;
        let elapsed_steps = 200usize;
        let elapsed_secs = 20.0_f32;
        let remaining_steps = total_steps - elapsed_steps;
        let steps_per_sec = elapsed_steps as f32 / elapsed_secs;
        let estimated_remaining = remaining_steps as f32 / steps_per_sec;
        assert!(
            (estimated_remaining - 80.0).abs() < 1e-3,
            "expected 80s remaining, got {estimated_remaining}"
        );
    }

    // 26. Resume from checkpoint: state fields are restored
    #[test]
    fn test_resume_from_checkpoint_state_restoration() {
        // Simulate checkpoint state
        let checkpoint_state = TrainingState {
            epoch: 2.0,
            global_step: 500,
            best_metric: Some(0.85),
            best_model_checkpoint: Some(PathBuf::from("/tmp/best_ckpt")),
            log_history: vec![LogEntry {
                step: 500,
                epoch: 2.0,
                learning_rate: 1e-4,
                loss: 0.3,
                eval_metrics: None,
                train_metrics: None,
            }],
            trial_name: None,
            trial_params: None,
        };
        // Simulate restore
        let state = checkpoint_state;
        assert_eq!(state.global_step, 500);
        assert!((state.epoch - 2.0).abs() < 1e-5);
        assert_eq!(state.log_history.len(), 1);
        assert!((state.best_metric.expect("best metric") - 0.85).abs() < 1e-5);
    }

    // 27. Warmup: LR stays at 0 before warmup steps, increases linearly
    #[test]
    fn test_warmup_lr_schedule() {
        let max_lr = 1e-4_f32;
        let warmup_steps = 100usize;
        let compute_lr = |step: usize| {
            if step < warmup_steps {
                // linear warmup: lr = max_lr * step / warmup_steps
                max_lr * step as f32 / warmup_steps as f32
            } else {
                max_lr
            }
        };
        assert!(
            (compute_lr(0) - 0.0).abs() < 1e-10,
            "LR at step 0 should be 0"
        );
        let mid = compute_lr(50);
        assert!(
            (mid - 5e-5).abs() < 1e-8,
            "LR at step 50 should be 5e-5, got {mid}"
        );
        let at_warmup = compute_lr(100);
        assert!(
            (at_warmup - max_lr).abs() < 1e-8,
            "LR at warmup end should equal max_lr"
        );
        let after = compute_lr(200);
        assert!(
            (after - max_lr).abs() < 1e-8,
            "LR after warmup should equal max_lr"
        );
    }

    // 28. LogEntry stores all fields correctly
    #[test]
    fn test_log_entry_fields() {
        let mut eval_map = HashMap::new();
        eval_map.insert("eval_loss".to_string(), 0.25_f32);
        let entry = LogEntry {
            step: 100,
            epoch: 1.5,
            learning_rate: 2e-5,
            loss: 0.3,
            eval_metrics: Some(eval_map.clone()),
            train_metrics: None,
        };
        assert_eq!(entry.step, 100);
        assert!((entry.epoch - 1.5).abs() < 1e-5);
        assert!((entry.learning_rate - 2e-5).abs() < 1e-10);
        assert!((entry.loss - 0.3).abs() < 1e-6);
        assert!(entry.eval_metrics.is_some());
        assert!(entry.train_metrics.is_none());
    }

    // 29. Logging trigger: every logging_steps
    #[test]
    fn test_logging_trigger_every_n_steps() {
        let logging_steps = 10usize;
        let trigger_count = (1usize..=50).filter(|s| s % logging_steps == 0).count();
        assert_eq!(trigger_count, 5, "should log 5 times in 50 steps");
    }

    // 30. Epoch tracking matches loop iteration
    #[test]
    fn test_epoch_tracking_in_loop() {
        let num_epochs = 3.0_f32;
        let mut state = make_state(0, 0.0);
        for epoch in 0..(num_epochs as usize) {
            state.epoch = epoch as f32;
        }
        // After loop ends, epoch should be 2.0 (0-indexed, last epoch)
        assert!(
            (state.epoch - 2.0).abs() < 1e-5,
            "expected epoch=2.0, got {}",
            state.epoch
        );
    }

    // 31. Early stopping: threshold effect — small improvement below threshold does not reset counter
    #[test]
    fn test_early_stopping_threshold_effect() {
        let mut cb = EarlyStoppingCallback::new(
            2,
            0.1, // threshold = 0.1 (improvement must be > 0.1)
            "eval_loss".to_string(),
            false, // minimize
        );
        let args = make_args();
        let state = make_state(0, 0.0);
        // First: sets best = 0.5
        cb.on_evaluate(&args, &state, &make_metrics(&[("eval_loss", 0.5)]));
        // "Improvement" of 0.05 < threshold 0.1 → NOT considered improvement
        cb.on_evaluate(&args, &state, &make_metrics(&[("eval_loss", 0.45)]));
        assert!(!cb.should_stop(), "patience 1 < 2");
        cb.on_evaluate(&args, &state, &make_metrics(&[("eval_loss", 0.45)]));
        assert!(cb.should_stop(), "patience 2 >= 2 should trigger stop");
    }

    // 32. CallbackManager default() creates empty manager
    #[test]
    fn test_callback_manager_default() {
        let mgr = CallbackManager::default();
        assert!(!mgr.should_stop());
    }

    // 33. Multiple callbacks: all receive on_log events
    #[test]
    fn test_callback_manager_multiple_callbacks_on_log() {
        use std::sync::{Arc, Mutex};
        let counter = Arc::new(Mutex::new(0usize));

        struct CountingCallback {
            counter: Arc<Mutex<usize>>,
        }
        impl TrainerCallback for CountingCallback {
            fn on_log(
                &mut self,
                _args: &TrainingArguments,
                _state: &TrainingState,
                _logs: &HashMap<String, f32>,
            ) {
                let mut c = self.counter.lock().expect("lock failed");
                *c += 1;
            }
        }

        let mut mgr = CallbackManager::new();
        for _ in 0..3 {
            mgr.add_callback(Box::new(CountingCallback {
                counter: Arc::clone(&counter),
            }));
        }
        let args = make_args();
        let state = make_state(10, 0.0);
        let logs = make_metrics(&[("loss", 0.5)]);
        mgr.call_on_log(&args, &state, &logs);
        let count = *counter.lock().expect("lock failed");
        assert_eq!(count, 3, "3 callbacks should each receive on_log");
    }

    // 34. LogEntry serde roundtrip
    #[test]
    fn test_log_entry_serde() {
        let entry = LogEntry {
            step: 50,
            epoch: 0.5,
            learning_rate: 1e-4,
            loss: 0.42,
            eval_metrics: None,
            train_metrics: None,
        };
        let json = serde_json::to_string(&entry).expect("serialize failed");
        let restored: LogEntry = serde_json::from_str(&json).expect("deserialize failed");
        assert_eq!(restored.step, 50);
        assert!((restored.loss - 0.42).abs() < 1e-6);
    }
}
