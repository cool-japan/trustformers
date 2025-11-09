//! Simple Classification Training Example
#![allow(unused_variables)]
//!
//! This example demonstrates basic supervised learning using the TrustformeRS training infrastructure.
//! It shows how to:
//! - Set up a simple classification model
//! - Configure training parameters
//! - Use callbacks for monitoring and early stopping
//! - Save and load checkpoints
//! - Evaluate model performance

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_training::{
    trainer::{Trainer, TrainerConfig, TrainerCallback},
    training_args::TrainingArgs,
    metrics::{Metric, MetricCollection, MetricResult},
    losses::Loss,
};
use trustformers_core::{
    tensor::Tensor,
    Model, ModelOutput,
    error::TrustformersError,
};

/// Simple feedforward neural network for classification
#[derive(Debug, Clone)]
struct SimpleClassifier {
    weights_1: Tensor,
    bias_1: Tensor,
    weights_2: Tensor,
    bias_2: Tensor,
    num_classes: usize,
}

impl SimpleClassifier {
    pub fn new(input_size: usize, hidden_size: usize, num_classes: usize) -> Result<Self> {
        // Xavier initialization
        let scale_1 = (2.0 / (input_size + hidden_size) as f32).sqrt();
        let scale_2 = (2.0 / (hidden_size + num_classes) as f32).sqrt();

        Ok(Self {
            weights_1: Tensor::randn(&[input_size, hidden_size])? * scale_1,
            bias_1: Tensor::zeros(&[hidden_size])?,
            weights_2: Tensor::randn(&[hidden_size, num_classes])? * scale_2,
            bias_2: Tensor::zeros(&[num_classes])?,
            num_classes,
        })
    }

    fn forward_impl(&self, input: &Tensor) -> Result<Tensor> {
        // First layer: input -> hidden
        let hidden = input.matmul(&self.weights_1)? + &self.bias_1;
        let hidden_activated = hidden.relu()?;

        // Second layer: hidden -> output
        let output = hidden_activated.matmul(&self.weights_2)? + &self.bias_2;

        Ok(output)
    }
}

impl Model for SimpleClassifier {
    type Output = Tensor;

    fn forward(&self, input: &Tensor) -> Result<Self::Output, TrustformersError> {
        self.forward_impl(input)
            .map_err(|e| TrustformersError::model_error(format!("Forward pass failed: {}", e)))
    }

    fn num_parameters(&self) -> usize {
        self.weights_1.numel() + self.bias_1.numel() +
        self.weights_2.numel() + self.bias_2.numel()
    }
}

impl ModelOutput for Tensor {
    fn extract_predictions(&self) -> Tensor {
        // For classification, return raw logits
        self.clone()
    }
}

/// Accuracy metric for classification
#[derive(Debug, Clone)]
struct AccuracyMetric;

impl Metric for AccuracyMetric {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> MetricResult {
        // Convert logits to class predictions
        let predicted_classes = predictions.argmax(-1)?;

        // Count correct predictions
        let correct = predicted_classes.eq(targets)?.sum()?;
        let total = targets.numel() as f64;
        let accuracy = correct.to_scalar::<f64>()? / total;

        MetricResult::Single(accuracy)
    }

    fn name(&self) -> &str {
        "accuracy"
    }
}

/// Cross-entropy loss for classification
#[derive(Debug, Clone)]
struct CrossEntropyLoss;

impl Loss for CrossEntropyLoss {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor, TrustformersError> {
        // Apply log softmax to predictions
        let log_probs = predictions.log_softmax(-1)?;

        // Compute negative log likelihood
        let nll = log_probs.gather(-1, targets)?.neg()?;

        // Return mean loss
        Ok(nll.mean()?)
    }

    fn name(&self) -> &str {
        "cross_entropy"
    }
}

/// Simple progress callback that prints training progress
#[derive(Debug)]
struct ProgressCallback {
    print_frequency: usize,
}

impl ProgressCallback {
    fn new(print_frequency: usize) -> Self {
        Self { print_frequency }
    }
}

impl TrainerCallback for ProgressCallback {
    fn on_epoch_begin(&mut self, epoch: usize, _logs: &HashMap<String, f64>) {
        println!("Starting epoch {}", epoch + 1);
    }

    fn on_batch_end(&mut self, batch: usize, logs: &HashMap<String, f64>) {
        if batch % self.print_frequency == 0 {
            let loss = logs.get("loss").unwrap_or(&0.0);
            println!("  Batch {}: loss = {:.4}", batch, loss);
        }
    }

    fn on_epoch_end(&mut self, epoch: usize, logs: &HashMap<String, f64>) {
        let train_loss = logs.get("loss").unwrap_or(&0.0);
        let eval_loss = logs.get("eval_loss").unwrap_or(&0.0);
        let accuracy = logs.get("eval_accuracy").unwrap_or(&0.0);

        println!("Epoch {} completed:", epoch + 1);
        println!("  Train loss: {:.4}", train_loss);
        println!("  Eval loss: {:.4}", eval_loss);
        println!("  Accuracy: {:.4}", accuracy);
        println!();
    }
}

/// Configuration for the training example
#[derive(Debug, Serialize, Deserialize)]
struct ExampleConfig {
    /// Model architecture parameters
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_classes: usize,

    /// Training parameters
    pub learning_rate: f64,
    pub batch_size: usize,
    pub num_epochs: usize,

    /// Data generation parameters
    pub num_train_samples: usize,
    pub num_eval_samples: usize,
    pub noise_level: f32,

    /// Training configuration
    pub print_frequency: usize,
    pub save_checkpoints: bool,
    pub checkpoint_dir: String,
}

impl Default for ExampleConfig {
    fn default() -> Self {
        Self {
            input_size: 10,
            hidden_size: 64,
            num_classes: 3,
            learning_rate: 0.001,
            batch_size: 32,
            num_epochs: 10,
            num_train_samples: 1000,
            num_eval_samples: 200,
            noise_level: 0.1,
            print_frequency: 10,
            save_checkpoints: true,
            checkpoint_dir: "./checkpoints".to_string(),
        }
    }
}

/// Generate synthetic classification data
fn generate_data(num_samples: usize, input_size: usize, num_classes: usize, noise_level: f32) -> Result<(Tensor, Tensor)> {
    let mut features = Vec::new();
    let mut labels = Vec::new();

    for _ in 0..num_samples {
        // Generate random class
        let class = rand::random::<usize>() % num_classes;

        // Generate features based on class (with some pattern)
        let mut feature_vec = vec![0.0f32; input_size];
        for i in 0..input_size {
            // Create class-dependent patterns
            let base_value = if i % num_classes == class {
                1.0
            } else {
                -0.5
            };

            // Add noise
            let noise = (rand::random::<f32>() - 0.5) * noise_level * 2.0;
            feature_vec[i] = base_value + noise;
        }

        features.extend(feature_vec);
        labels.push(class as i64);
    }

    let features_tensor = Tensor::from_vec(features, &[num_samples, input_size])?;
    let labels_tensor = Tensor::from_vec(labels, &[num_samples])?;

    Ok((features_tensor, labels_tensor))
}

/// Create data loaders for training and evaluation
fn create_data_loaders(
    train_features: Tensor,
    train_labels: Tensor,
    eval_features: Tensor,
    eval_labels: Tensor,
    batch_size: usize,
) -> Result<(Vec<(Tensor, Tensor)>, Vec<(Tensor, Tensor)>)> {
    // Simple batching - in practice you'd use more sophisticated data loaders
    let mut train_batches = Vec::new();
    let mut eval_batches = Vec::new();

    // Create training batches
    let num_train_samples = train_features.shape()[0];
    for start in (0..num_train_samples).step_by(batch_size) {
        let end = (start + batch_size).min(num_train_samples);
        let batch_features = train_features.slice(0, start, end)?;
        let batch_labels = train_labels.slice(0, start, end)?;
        train_batches.push((batch_features, batch_labels));
    }

    // Create evaluation batches
    let num_eval_samples = eval_features.shape()[0];
    for start in (0..num_eval_samples).step_by(batch_size) {
        let end = (start + batch_size).min(num_eval_samples);
        let batch_features = eval_features.slice(0, start, end)?;
        let batch_labels = eval_labels.slice(0, start, end)?;
        eval_batches.push((batch_features, batch_labels));
    }

    Ok((train_batches, eval_batches))
}

fn main() -> Result<()> {
    println!("🚀 TrustformeRS Simple Classification Training Example");
    println!("=================================================");

    // Load configuration
    let config = ExampleConfig::default();
    println!("Configuration:");
    println!("  Input size: {}", config.input_size);
    println!("  Hidden size: {}", config.hidden_size);
    println!("  Number of classes: {}", config.num_classes);
    println!("  Learning rate: {}", config.learning_rate);
    println!("  Batch size: {}", config.batch_size);
    println!("  Number of epochs: {}", config.num_epochs);
    println!();

    // Generate synthetic data
    println!("📊 Generating synthetic data...");
    let (train_features, train_labels) = generate_data(
        config.num_train_samples,
        config.input_size,
        config.num_classes,
        config.noise_level,
    )?;
    let (eval_features, eval_labels) = generate_data(
        config.num_eval_samples,
        config.input_size,
        config.num_classes,
        config.noise_level,
    )?;

    println!("  Training samples: {}", config.num_train_samples);
    println!("  Evaluation samples: {}", config.num_eval_samples);
    println!();

    // Create data loaders
    let (train_batches, eval_batches) = create_data_loaders(
        train_features,
        train_labels,
        eval_features,
        eval_labels,
        config.batch_size,
    )?;

    println!("  Training batches: {}", train_batches.len());
    println!("  Evaluation batches: {}", eval_batches.len());
    println!();

    // Create model
    println!("🧠 Creating model...");
    let model = SimpleClassifier::new(
        config.input_size,
        config.hidden_size,
        config.num_classes,
    )?;
    println!("  Model parameters: {}", model.num_parameters());
    println!();

    // Create loss function
    let loss_fn = CrossEntropyLoss;

    // Create metrics
    let mut metrics = MetricCollection::new();
    metrics.add_metric("accuracy", Box::new(AccuracyMetric));

    // Create trainer configuration
    let training_args = TrainingArgs {
        learning_rate: config.learning_rate,
        num_epochs: config.num_epochs,
        batch_size: config.batch_size,
        weight_decay: 0.0001,
        warmup_steps: 100,
        evaluation_strategy: "epoch".to_string(),
        save_strategy: "epoch".to_string(),
        logging_steps: config.print_frequency,
        save_total_limit: Some(3),
        load_best_model_at_end: true,
        ..Default::default()
    };

    let trainer_config = TrainerConfig {
        output_dir: config.checkpoint_dir.clone(),
        ..Default::default()
    };

    // Create trainer
    println!("🎯 Initializing trainer...");
    let mut trainer = Trainer::new(
        model,
        Box::new(loss_fn),
        Some(Box::new(metrics)),
        training_args,
        trainer_config,
    )?;

    // Add progress callback
    trainer.add_callback(Box::new(ProgressCallback::new(config.print_frequency)));

    // Create checkpoint directory if it doesn't exist
    if config.save_checkpoints {
        std::fs::create_dir_all(&config.checkpoint_dir)?;
        println!("  Checkpoints will be saved to: {}", config.checkpoint_dir);
    }
    println!();

    // Start training
    println!("🔥 Starting training...");
    println!("Training for {} epochs", config.num_epochs);
    println!();

    trainer.train(train_batches, Some(eval_batches))?;

    println!("✅ Training completed successfully!");
    println!();

    // Final evaluation
    println!("📈 Final model evaluation:");
    let final_metrics = trainer.evaluate(eval_batches)?;
    for (name, value) in final_metrics {
        println!("  {}: {:.4}", name, value);
    }

    println!();
    println!("🎉 Example completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let model = SimpleClassifier::new(10, 64, 3).unwrap();
        assert_eq!(model.num_classes, 3);
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_model_forward() {
        let model = SimpleClassifier::new(10, 64, 3).unwrap();
        let input = Tensor::randn(&[2, 10]).unwrap();
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_data_generation() {
        let (features, labels) = generate_data(100, 10, 3, 0.1).unwrap();
        assert_eq!(features.shape(), &[100, 10]);
        assert_eq!(labels.shape(), &[100]);
    }

    #[test]
    fn test_accuracy_metric() {
        let metric = AccuracyMetric;
        let predictions = Tensor::from_vec(vec![2.0, 1.0, 0.0, 0.0, 2.0, 1.0], &[2, 3]).unwrap();
        let targets = Tensor::from_vec(vec![0i64, 2i64], &[2]).unwrap();

        let result = metric.compute(&predictions, &targets).unwrap();
        if let MetricResult::Single(accuracy) = result {
            assert!(accuracy >= 0.0 && accuracy <= 1.0);
        }
    }

    #[test]
    fn test_cross_entropy_loss() {
        let loss_fn = CrossEntropyLoss;
        let predictions = Tensor::randn(&[2, 3]).unwrap();
        let targets = Tensor::from_vec(vec![0i64, 2i64], &[2]).unwrap();

        let loss = loss_fn.compute(&predictions, &targets).unwrap();
        assert_eq!(loss.shape(), &[]);  // Scalar loss
    }
}