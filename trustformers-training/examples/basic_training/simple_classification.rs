//! Simple Classification Training Example
//!
//! This example walks through the end-to-end shape of TrustformeRS's training
//! infrastructure on a tiny synthetic classification task:
//!
//! - Implementing `trustformers_core::traits::{Config, Model}` for a small
//!   two-layer feedforward classifier (`SimpleClassifier`).
//! - Generating synthetic classification data and batching it into the
//!   `Vec<(Tensor, Tensor)>` shape that `Trainer::train`/`Trainer::evaluate`
//!   expect.
//! - Reusing the crate's own tested `CrossEntropyLoss`
//!   (`trustformers_training::losses`) instead of hand-rolling a loss.
//! - Implementing `TrainerCallback` for lightweight progress reporting.
//! - Building a `Trainer`, training it for a few epochs, and printing the
//!   final evaluation metrics.
//!
//! # Note on parameter updates
//!
//! `Trainer::train` computes the loss gradient with respect to the model's
//! output (logits) on every step, but it only *applies* that gradient to a
//! model's parameters for models that implement the optimizer-facing
//! `ParameterAccess` trait (see `trustformers_training::trainer`). This
//! example's `SimpleClassifier` does not implement `ParameterAccess`, so its
//! weights are not actually updated; the point of this example is to
//! exercise the public `Model` / `Loss` / `TrainerCallback` / `Trainer` API
//! surface end-to-end, not to demonstrate convergence. Because of this, the
//! printed loss is not expected to decrease across epochs.
//!
//! Run it with:
//! ```bash
//! cargo run --example simple_classification -p trustformers-training
//! ```
//!
//! Run its unit tests with:
//! ```bash
//! cargo test --example simple_classification -p trustformers-training
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Config, Model};
use trustformers_core::TrustformersError;
use trustformers_optim::Adam;
use trustformers_training::trainer::{TaskType, TrainerCallback, TrainingState};
use trustformers_training::{
    CrossEntropyLoss, EvaluationStrategy, SaveStrategy, Trainer, TrainingArguments,
};

/// Configuration for [`SimpleClassifier`], following the `Config` pattern
/// documented on `trustformers_core::traits::Config`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SimpleClassifierConfig {
    input_size: usize,
    hidden_size: usize,
    num_classes: usize,
}

impl Config for SimpleClassifierConfig {
    fn architecture(&self) -> &'static str {
        "simple_classifier"
    }
}

/// A minimal two-layer feedforward classifier: `Linear -> ReLU -> Linear`.
#[derive(Debug)]
struct SimpleClassifier {
    weights_1: Tensor,
    bias_1: Tensor,
    weights_2: Tensor,
    bias_2: Tensor,
    config: SimpleClassifierConfig,
}

impl SimpleClassifier {
    /// Creates a new classifier with Xavier/Glorot-initialized weights.
    fn new(input_size: usize, hidden_size: usize, num_classes: usize) -> Result<Self> {
        // Xavier/Glorot initialization scale for each linear layer.
        let scale_1 = (2.0 / (input_size + hidden_size) as f32).sqrt();
        let scale_2 = (2.0 / (hidden_size + num_classes) as f32).sqrt();

        let weights_1 = Tensor::randn(&[input_size, hidden_size])?.scalar_mul(scale_1)?;
        let bias_1 = Tensor::zeros(&[hidden_size])?;
        let weights_2 = Tensor::randn(&[hidden_size, num_classes])?.scalar_mul(scale_2)?;
        let bias_2 = Tensor::zeros(&[num_classes])?;

        Ok(Self {
            weights_1,
            bias_1,
            weights_2,
            bias_2,
            config: SimpleClassifierConfig {
                input_size,
                hidden_size,
                num_classes,
            },
        })
    }
}

impl Model for SimpleClassifier {
    type Config = SimpleClassifierConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output, TrustformersError> {
        let hidden = input.matmul(&self.weights_1)?.add(&self.bias_1)?.relu()?;
        let logits = hidden.matmul(&self.weights_2)?.add(&self.bias_2)?;
        Ok(logits)
    }

    fn load_pretrained(
        &mut self,
        _reader: &mut dyn std::io::Read,
    ) -> Result<(), TrustformersError> {
        // This example only ever trains from scratch; loading pretrained
        // weights is out of scope, but `Model` still requires a real body.
        Ok(())
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        self.weights_1.size() + self.bias_1.size() + self.weights_2.size() + self.bias_2.size()
    }
}

/// Prints lightweight progress information as training proceeds.
///
/// `print_frequency` throttles the per-step heartbeat printed from
/// `on_step_end`. It is intentionally independent of
/// `TrainingArguments::logging_steps`, which separately controls how often
/// the `Trainer` calls `on_log` with the current loss - this callback
/// demonstrates both mechanisms.
#[derive(Debug)]
struct ProgressCallback {
    print_frequency: usize,
}

impl ProgressCallback {
    fn new(print_frequency: usize) -> Self {
        Self {
            print_frequency: print_frequency.max(1),
        }
    }
}

impl TrainerCallback for ProgressCallback {
    fn on_train_begin(&mut self, _args: &TrainingArguments, _state: &TrainingState) {
        println!("Training started.");
    }

    fn on_epoch_begin(&mut self, _args: &TrainingArguments, state: &TrainingState) {
        let epoch_number = state.epoch as usize + 1;
        println!("Epoch {epoch_number} starting...");
    }

    fn on_step_end(&mut self, _args: &TrainingArguments, state: &TrainingState) {
        let step = state.global_step;
        if step.is_multiple_of(self.print_frequency) {
            println!("  ...step {step}");
        }
    }

    fn on_log(
        &mut self,
        _args: &TrainingArguments,
        state: &TrainingState,
        logs: &HashMap<String, f32>,
    ) {
        let step = state.global_step;
        if let Some(loss) = logs.get("loss") {
            println!("  step {step}: loss = {loss:.4}");
        }
    }

    fn on_epoch_end(&mut self, _args: &TrainingArguments, state: &TrainingState) {
        let epoch_number = state.epoch as usize + 1;
        let step = state.global_step;
        println!("Epoch {epoch_number} finished (global step {step}).");
    }

    fn on_evaluate(
        &mut self,
        _args: &TrainingArguments,
        _state: &TrainingState,
        metrics: &HashMap<String, f32>,
    ) {
        if let Some(loss) = metrics.get("eval_loss") {
            println!("  evaluation: eval_loss = {loss:.4}");
        }
    }

    fn on_train_end(&mut self, _args: &TrainingArguments, _state: &TrainingState) {
        println!("Training finished.");
    }
}

/// Generates a synthetic classification dataset.
///
/// Each sample's features are centered on a class-dependent pattern (`1.0`
/// at positions congruent to the sample's class modulo `num_classes`,
/// `-0.5` elsewhere) plus uniform noise, which keeps the task learnable
/// while still giving `CrossEntropyLoss` something non-trivial to compute
/// over.
fn generate_data(
    num_samples: usize,
    input_size: usize,
    num_classes: usize,
    noise_level: f32,
) -> Result<(Tensor, Tensor)> {
    let mut features = Vec::with_capacity(num_samples * input_size);
    let mut labels = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let class = fastrand::usize(0..num_classes);

        for i in 0..input_size {
            let base_value = if i % num_classes == class { 1.0 } else { -0.5 };
            let noise = (fastrand::f32() - 0.5) * noise_level * 2.0;
            features.push(base_value + noise);
        }
        labels.push(class as i64);
    }

    let features_tensor = Tensor::from_vec(features, &[num_samples, input_size])?;
    let labels_tensor = Tensor::from_vec_i64(labels, &[num_samples])?;

    Ok((features_tensor, labels_tensor))
}

/// Splits `features`/`labels` (both sized `num_samples` along axis 0) into
/// `Vec<(Tensor, Tensor)>` batches of at most `batch_size` samples each -
/// the shape `Trainer::train`/`Trainer::evaluate` expect.
fn create_batches(
    features: &Tensor,
    labels: &Tensor,
    batch_size: usize,
) -> Result<Vec<(Tensor, Tensor)>> {
    let num_samples = features.shape()[0];
    let mut batches = Vec::new();

    for start in (0..num_samples).step_by(batch_size) {
        let end = (start + batch_size).min(num_samples);
        let batch_features = features.slice(0, start, end)?;
        let batch_labels = labels.slice(0, start, end)?;
        batches.push((batch_features, batch_labels));
    }

    Ok(batches)
}

fn main() -> Result<()> {
    println!("TrustformeRS Simple Classification Training Example");
    println!("=====================================================");
    println!();

    // ---- configuration ----
    let input_size: usize = 10;
    let hidden_size: usize = 64;
    let num_classes: usize = 3;
    let batch_size: usize = 32;
    let num_train_samples: usize = 1_000;
    let num_eval_samples: usize = 200;
    let noise_level: f32 = 0.1;
    let num_epochs: f32 = 5.0;
    let logging_steps: usize = 10;
    let print_frequency: usize = 20;

    println!("Configuration:");
    println!("  input_size:        {input_size}");
    println!("  hidden_size:       {hidden_size}");
    println!("  num_classes:       {num_classes}");
    println!("  batch_size:        {batch_size}");
    println!("  num_epochs:        {num_epochs:.1}");
    println!("  num_train_samples: {num_train_samples}");
    println!("  num_eval_samples:  {num_eval_samples}");
    println!();

    // ---- synthetic data ----
    println!("Generating synthetic classification data...");
    let (train_features, train_labels) =
        generate_data(num_train_samples, input_size, num_classes, noise_level)?;
    let (eval_features, eval_labels) =
        generate_data(num_eval_samples, input_size, num_classes, noise_level)?;

    let train_batches = create_batches(&train_features, &train_labels, batch_size)?;
    let eval_batches = create_batches(&eval_features, &eval_labels, batch_size)?;

    println!("  training batches:   {}", train_batches.len());
    println!("  evaluation batches: {}", eval_batches.len());
    println!();

    // ---- model ----
    println!("Creating model...");
    let model = SimpleClassifier::new(input_size, hidden_size, num_classes)?;
    let model_config = model.get_config();
    println!(
        "  architecture: {}, input_size: {}, hidden_size: {}, num_classes: {}",
        model_config.architecture(),
        model_config.input_size,
        model_config.hidden_size,
        model_config.num_classes
    );
    println!("  model parameters: {}", model.num_parameters());
    println!();

    // ---- trainer ----
    let output_dir = std::env::temp_dir().join("trustformers_simple_classification_example");
    let training_args = TrainingArguments {
        output_dir,
        per_device_train_batch_size: batch_size,
        per_device_eval_batch_size: batch_size,
        num_train_epochs: num_epochs,
        learning_rate: 1e-4,
        logging_steps,
        evaluation_strategy: EvaluationStrategy::Epoch,
        save_strategy: SaveStrategy::Epoch,
        ..TrainingArguments::default()
    };

    let optimizer = Box::new(Adam::new(1e-4, (0.9, 0.999), 1e-8, 0.0));
    let loss_fn = Box::new(CrossEntropyLoss::new());

    let mut trainer = Trainer::new(
        model,
        training_args,
        optimizer,
        loss_fn,
        TaskType::Classification,
    )?
    .add_callback(Box::new(ProgressCallback::new(print_frequency)));

    println!("Starting training for {num_epochs:.1} epochs...");
    println!();
    trainer.train(&train_batches, Some(&eval_batches))?;
    println!();
    println!("Training completed.");
    println!();

    // ---- final evaluation ----
    println!("Final evaluation:");
    let final_metrics = trainer.evaluate(&eval_batches)?;
    let mut entries: Vec<(&String, &f32)> = final_metrics.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    for (name, value) in entries {
        println!("  {name}: {value:.4}");
    }
    println!();
    println!("Example completed successfully.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_training::Loss;

    #[test]
    fn test_model_creation() {
        let model = SimpleClassifier::new(10, 64, 3).expect("failed to create test model");
        assert_eq!(model.get_config().num_classes, 3);
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_model_forward_shape() {
        let model = SimpleClassifier::new(10, 64, 3).expect("failed to create test model");
        let input = Tensor::randn(&[2, 10]).expect("failed to create test input tensor");
        let output = model.forward(input).expect("forward pass failed");
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_data_generation() {
        let (features, labels) =
            generate_data(100, 10, 3, 0.1).expect("failed to generate test data");
        assert_eq!(features.shape(), &[100, 10]);
        assert_eq!(labels.shape(), &[100]);
    }

    #[test]
    fn test_create_batches() {
        let (features, labels) =
            generate_data(100, 10, 3, 0.1).expect("failed to generate test data");
        let batches = create_batches(&features, &labels, 32).expect("failed to create batches");

        assert_eq!(batches.len(), 4);
        assert_eq!(batches[0].0.shape(), &[32, 10]);
        assert_eq!(
            batches.last().expect("batches should be non-empty").0.shape(),
            &[4, 10]
        );
    }

    #[test]
    fn test_cross_entropy_loss_computation() {
        let loss_fn = CrossEntropyLoss::new();
        let predictions = Tensor::from_vec(vec![2.0, 1.0, 0.0, 0.0, 0.5, 2.0], &[2, 3])
            .expect("failed to create predictions tensor");
        let targets =
            Tensor::from_vec_i64(vec![0, 2], &[2]).expect("failed to create targets tensor");

        let loss = loss_fn
            .compute(&predictions, &targets)
            .expect("failed to compute cross entropy loss");

        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }
}
