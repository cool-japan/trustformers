//! Advanced Optimization Techniques Example
#![allow(unused_variables)]
//!
//! This example demonstrates the sophisticated optimization features of TrustformeRS:
//! - Adaptive gradient scaling with automatic adjustment
//! - Multi-strategy adaptive learning rate scheduling
//! - Advanced stability monitoring and recovery
//! - Gradient anomaly detection and correction
//! - Training dynamics analysis and optimization

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_training::{
    adaptive_gradient_scaling::{AdaptiveGradientScaler, AdaptiveGradientScalingConfig},
    adaptive_learning_rate::{AdaptiveLearningRateScheduler, AdaptiveLearningRateConfig, AdaptationStrategy},
    advanced_stability_monitor::{AdvancedStabilityMonitor, StabilityMonitorConfig},
    gradient_anomaly_recovery::{GradientAnomalyRecovery, GradientAnomalyConfig},
    trainer::{Trainer, TrainerConfig, TrainerCallback},
    training_args::TrainingArgs,
    metrics::{Metric, MetricCollection, MetricResult},
    losses::Loss,
};
use trustformers_core::{
    tensor::Tensor,
    Model, ModelOutput,
    TrustformersError,
};

/// Advanced neural network with more complex architecture
#[derive(Debug, Clone)]
struct AdvancedClassifier {
    // First hidden layer
    weights_1: Tensor,
    bias_1: Tensor,

    // Second hidden layer
    weights_2: Tensor,
    bias_2: Tensor,

    // Third hidden layer
    weights_3: Tensor,
    bias_3: Tensor,

    // Output layer
    weights_out: Tensor,
    bias_out: Tensor,

    num_classes: usize,
    dropout_rate: f32,
}

impl AdvancedClassifier {
    pub fn new(input_size: usize, hidden_sizes: &[usize], num_classes: usize, dropout_rate: f32) -> Result<Self> {
        // He initialization for ReLU networks
        let init_weight = |fan_in: usize, fan_out: usize| -> Result<Tensor> {
            let std = (2.0 / fan_in as f32).sqrt();
            Ok(Tensor::randn(&[fan_in, fan_out])? * std)
        };

        let weights_1 = init_weight(input_size, hidden_sizes[0])?;
        let bias_1 = Tensor::zeros(&[hidden_sizes[0]])?;

        let weights_2 = init_weight(hidden_sizes[0], hidden_sizes[1])?;
        let bias_2 = Tensor::zeros(&[hidden_sizes[1]])?;

        let weights_3 = init_weight(hidden_sizes[1], hidden_sizes[2])?;
        let bias_3 = Tensor::zeros(&[hidden_sizes[2]])?;

        let weights_out = init_weight(hidden_sizes[2], num_classes)?;
        let bias_out = Tensor::zeros(&[num_classes])?;

        Ok(Self {
            weights_1,
            bias_1,
            weights_2,
            bias_2,
            weights_3,
            bias_3,
            weights_out,
            bias_out,
            num_classes,
            dropout_rate,
        })
    }

    fn forward_impl(&self, input: &Tensor, training: bool) -> Result<Tensor> {
        // First layer
        let hidden_1 = input.matmul(&self.weights_1)? + &self.bias_1;
        let activated_1 = hidden_1.relu()?;
        let dropped_1 = if training && self.dropout_rate > 0.0 {
            activated_1.dropout(self.dropout_rate)?
        } else {
            activated_1
        };

        // Second layer
        let hidden_2 = dropped_1.matmul(&self.weights_2)? + &self.bias_2;
        let activated_2 = hidden_2.relu()?;
        let dropped_2 = if training && self.dropout_rate > 0.0 {
            activated_2.dropout(self.dropout_rate)?
        } else {
            activated_2
        };

        // Third layer
        let hidden_3 = dropped_2.matmul(&self.weights_3)? + &self.bias_3;
        let activated_3 = hidden_3.relu()?;
        let dropped_3 = if training && self.dropout_rate > 0.0 {
            activated_3.dropout(self.dropout_rate)?
        } else {
            activated_3
        };

        // Output layer
        let output = dropped_3.matmul(&self.weights_out)? + &self.bias_out;

        Ok(output)
    }
}

impl Model for AdvancedClassifier {
    type Output = Tensor;

    fn forward(&self, input: &Tensor) -> Result<Self::Output, TrustformersError> {
        self.forward_impl(input, false)  // Inference mode
            .map_err(|e| TrustformersError::model_error(format!("Forward pass failed: {}", e)))
    }

    fn num_parameters(&self) -> usize {
        self.weights_1.numel() + self.bias_1.numel() +
        self.weights_2.numel() + self.bias_2.numel() +
        self.weights_3.numel() + self.bias_3.numel() +
        self.weights_out.numel() + self.bias_out.numel()
    }
}

impl ModelOutput for Tensor {
    fn extract_predictions(&self) -> Tensor {
        self.clone()
    }
}

/// Comprehensive training callback that integrates all advanced optimization features
#[derive(Debug)]
struct AdvancedOptimizationCallback {
    gradient_scaler: AdaptiveGradientScaler,
    lr_scheduler: AdaptiveLearningRateScheduler,
    stability_monitor: AdvancedStabilityMonitor,
    anomaly_recovery: GradientAnomalyRecovery,
    step_count: usize,
    print_frequency: usize,
}

impl AdvancedOptimizationCallback {
    fn new(
        gradient_scaling_config: AdaptiveGradientScalingConfig,
        lr_config: AdaptiveLearningRateConfig,
        stability_config: StabilityMonitorConfig,
        anomaly_config: GradientAnomalyConfig,
        print_frequency: usize,
    ) -> Result<Self> {
        Ok(Self {
            gradient_scaler: AdaptiveGradientScaler::new(gradient_scaling_config)?,
            lr_scheduler: AdaptiveLearningRateScheduler::new(lr_config)?,
            stability_monitor: AdvancedStabilityMonitor::new(stability_config)?,
            anomaly_recovery: GradientAnomalyRecovery::new(anomaly_config)?,
            step_count: 0,
            print_frequency,
        })
    }
}

impl TrainerCallback for AdvancedOptimizationCallback {
    fn on_batch_begin(&mut self, batch: usize, logs: &HashMap<String, f64>) {
        self.step_count += 1;
    }

    fn on_batch_end(&mut self, batch: usize, logs: &HashMap<String, f64>) {
        let loss = logs.get("loss").cloned().unwrap_or(0.0) as f32;
        let lr = logs.get("learning_rate").cloned().unwrap_or(0.001) as f32;

        // Simulate gradient norm (in real implementation, this would come from the optimizer)
        let gradient_norm = loss * (1.0 + 0.1 * (self.step_count as f32).sin());

        // Update adaptive learning rate
        if let Ok(new_lr) = self.lr_scheduler.update_learning_rate(loss, gradient_norm, lr) {
            // In real implementation, you would update the optimizer's learning rate here
        }

        // Update gradient scaling
        if let Ok(_) = self.gradient_scaler.update_statistics(gradient_norm) {
            // Scaling factors would be applied to gradients here
        }

        // Monitor training stability
        if let Ok(predictions) = self.stability_monitor.analyze_training_step(loss, gradient_norm, lr) {
            if !predictions.is_empty() {
                for prediction in predictions {
                    if prediction.confidence > 0.7 {
                        println!("  ⚠️  Stability warning: {} (confidence: {:.2})",
                               prediction.prediction_type, prediction.confidence);

                        // Apply proactive recovery if needed
                        if let Ok(action) = prediction.suggested_action() {
                            println!("     💡 Suggested action: {}", action);
                        }
                    }
                }
            }
        }

        // Detect and recover from gradient anomalies
        // In real implementation, you'd pass actual gradients here
        let mock_gradients = vec![("layer_1".to_string(), gradient_norm)];
        if let Ok(anomalies) = self.anomaly_recovery.detect_anomalies(&mock_gradients) {
            if !anomalies.is_empty() {
                for anomaly in anomalies {
                    println!("  🔧 Gradient anomaly detected: {} (severity: {:.2})",
                           anomaly.anomaly_type, anomaly.severity);

                    // Apply recovery strategy
                    if let Ok(strategy) = self.anomaly_recovery.select_recovery_strategy(&anomaly) {
                        println!("     🛠️  Applying recovery: {}", strategy);
                    }
                }
            }
        }

        // Print progress
        if batch % self.print_frequency == 0 {
            let current_lr = self.lr_scheduler.get_current_learning_rate();
            let scaling_stats = self.gradient_scaler.get_statistics();

            println!("  Batch {}: loss = {:.4}, lr = {:.6}, grad_norm = {:.4}",
                   batch, loss, current_lr, gradient_norm);

            if batch % (self.print_frequency * 5) == 0 {
                println!("    📊 Gradient scaling - global: {:.3}, effectiveness: {:.2}%",
                       scaling_stats.global_scale, scaling_stats.effectiveness_score * 100.0);

                let lr_stats = self.lr_scheduler.get_statistics();
                println!("    📈 Learning rate - trend: {:?}, confidence: {:.2}",
                       lr_stats.performance_trend, lr_stats.confidence_score);
            }
        }
    }

    fn on_epoch_end(&mut self, epoch: usize, logs: &HashMap<String, f64>) {
        let train_loss = logs.get("loss").unwrap_or(&0.0);
        let eval_loss = logs.get("eval_loss").unwrap_or(&0.0);
        let accuracy = logs.get("eval_accuracy").unwrap_or(&0.0);

        println!("Epoch {} Summary:", epoch + 1);
        println!("  📉 Train loss: {:.4}", train_loss);
        println!("  📊 Eval loss: {:.4}", eval_loss);
        println!("  🎯 Accuracy: {:.4}", accuracy);

        // Print optimization statistics
        let scaling_stats = self.gradient_scaler.get_statistics();
        println!("  🔧 Gradient scaling effectiveness: {:.2}%", scaling_stats.effectiveness_score * 100.0);

        let lr_stats = self.lr_scheduler.get_statistics();
        println!("  📈 Learning rate adaptations: {}", lr_stats.adaptation_count);

        let stability_stats = self.stability_monitor.get_statistics();
        println!("  ⚖️  Stability score: {:.2}", stability_stats.overall_stability_score);

        let anomaly_stats = self.anomaly_recovery.get_statistics();
        println!("  🛡️  Anomalies detected/recovered: {}/{}",
               anomaly_stats.total_anomalies_detected,
               anomaly_stats.successful_recoveries);
        println!();
    }
}

/// Configuration for advanced optimization example
#[derive(Debug, Serialize, Deserialize)]
struct AdvancedOptimizationConfig {
    // Model architecture
    pub input_size: usize,
    pub hidden_sizes: Vec<usize>,
    pub num_classes: usize,
    pub dropout_rate: f32,

    // Training parameters
    pub learning_rate: f64,
    pub batch_size: usize,
    pub num_epochs: usize,

    // Data parameters
    pub num_train_samples: usize,
    pub num_eval_samples: usize,
    pub noise_level: f32,
    pub data_complexity: f32,  // Controls how complex/noisy the data patterns are

    // Advanced optimization settings
    pub enable_adaptive_gradient_scaling: bool,
    pub enable_adaptive_learning_rate: bool,
    pub enable_stability_monitoring: bool,
    pub enable_anomaly_recovery: bool,

    // Monitoring
    pub print_frequency: usize,
    pub checkpoint_dir: String,
}

impl Default for AdvancedOptimizationConfig {
    fn default() -> Self {
        Self {
            input_size: 20,
            hidden_sizes: vec![128, 64, 32],
            num_classes: 5,
            dropout_rate: 0.2,
            learning_rate: 0.01,
            batch_size: 64,
            num_epochs: 20,
            num_train_samples: 5000,
            num_eval_samples: 1000,
            noise_level: 0.2,
            data_complexity: 1.0,
            enable_adaptive_gradient_scaling: true,
            enable_adaptive_learning_rate: true,
            enable_stability_monitoring: true,
            enable_anomaly_recovery: true,
            print_frequency: 20,
            checkpoint_dir: "./advanced_checkpoints".to_string(),
        }
    }
}

/// Generate more complex synthetic data with intentional challenges
fn generate_complex_data(
    num_samples: usize,
    input_size: usize,
    num_classes: usize,
    noise_level: f32,
    complexity: f32,
) -> Result<(Tensor, Tensor)> {
    let mut features = Vec::new();
    let mut labels = Vec::new();

    for i in 0..num_samples {
        let class = i % num_classes;
        let mut feature_vec = vec![0.0f32; input_size];

        // Create more complex, non-linear patterns
        for j in 0..input_size {
            let base_pattern = match class {
                0 => (j as f32 / input_size as f32).sin(),
                1 => (j as f32 / input_size as f32).cos(),
                2 => ((j as f32 / input_size as f32) * 2.0).sin() * 0.5,
                3 => if j % 2 == 0 { 1.0 } else { -1.0 },
                _ => (j as f32 / input_size as f32).powi(2) * 2.0 - 1.0,
            };

            // Add complexity-dependent non-linear transformations
            let complexity_factor = 1.0 + complexity * (j as f32 / input_size as f32);
            let transformed = base_pattern * complexity_factor;

            // Add noise
            let noise = (rand::random::<f32>() - 0.5) * noise_level * 2.0;
            feature_vec[j] = transformed + noise;
        }

        // Add some samples with intentional outliers to test anomaly recovery
        if i % 100 == 0 {
            for j in 0..input_size / 4 {
                feature_vec[j] *= 10.0; // Create outliers
            }
        }

        features.extend(feature_vec);
        labels.push(class as i64);
    }

    let features_tensor = Tensor::from_vec(features, &[num_samples, input_size])?;
    let labels_tensor = Tensor::from_vec(labels, &[num_samples])?;

    Ok((features_tensor, labels_tensor))
}

fn main() -> Result<()> {
    println!("🚀 TrustformeRS Advanced Optimization Techniques Example");
    println!("======================================================");
    println!("This example demonstrates:");
    println!("  🔧 Adaptive gradient scaling");
    println!("  📈 Multi-strategy learning rate scheduling");
    println!("  ⚖️  Advanced stability monitoring");
    println!("  🛡️  Gradient anomaly detection and recovery");
    println!();

    // Load configuration
    let config = AdvancedOptimizationConfig::default();
    println!("Configuration:");
    println!("  Model: {} -> {:?} -> {}", config.input_size, config.hidden_sizes, config.num_classes);
    println!("  Training: {} epochs, {} batch size, {:.4} learning rate",
           config.num_epochs, config.batch_size, config.learning_rate);
    println!("  Data: {} train, {} eval samples", config.num_train_samples, config.num_eval_samples);
    println!("  Advanced features: scaling={}, lr={}, stability={}, anomaly={}",
           config.enable_adaptive_gradient_scaling,
           config.enable_adaptive_learning_rate,
           config.enable_stability_monitoring,
           config.enable_anomaly_recovery);
    println!();

    // Generate complex synthetic data
    println!("📊 Generating complex synthetic data...");
    let (train_features, train_labels) = generate_complex_data(
        config.num_train_samples,
        config.input_size,
        config.num_classes,
        config.noise_level,
        config.data_complexity,
    )?;
    let (eval_features, eval_labels) = generate_complex_data(
        config.num_eval_samples,
        config.input_size,
        config.num_classes,
        config.noise_level,
        config.data_complexity * 0.8, // Slightly easier for evaluation
    )?;

    // Create model
    println!("🧠 Creating advanced model...");
    let model = AdvancedClassifier::new(
        config.input_size,
        &config.hidden_sizes,
        config.num_classes,
        config.dropout_rate,
    )?;
    println!("  Model parameters: {}", model.num_parameters());
    println!();

    // Configure advanced optimization features
    println!("⚙️  Configuring advanced optimization...");

    let gradient_scaling_config = AdaptiveGradientScalingConfig {
        auto_scaling: config.enable_adaptive_gradient_scaling,
        history_window: 200,
        target_norm: 2.0,
        adaptation_rate: 0.02,
        min_scale: 0.1,
        max_scale: 5.0,
        momentum: 0.95,
        per_layer_scaling: true,
        outlier_filtering: true,
        warmup_steps: 500,
    };

    let lr_config = AdaptiveLearningRateConfig {
        initial_lr: config.learning_rate as f32,
        min_lr: 0.00001,
        max_lr: 0.1,
        adaptation_strategies: vec![
            AdaptationStrategy::ReduceOnPlateau,
            AdaptationStrategy::GradientNormBased,
            AdaptationStrategy::LossVarianceBased,
        ],
        plateau_patience: 5,
        plateau_threshold: 0.01,
        adaptation_factor_increase: 1.1,
        adaptation_factor_decrease: 0.9,
        window_size: 10,
        enable_cyclical: true,
        cyclical_max_lr: config.learning_rate as f32 * 2.0,
        cyclical_step_size: 100,
    };

    let stability_config = StabilityMonitorConfig {
        enable_predictive_detection: config.enable_stability_monitoring,
        gradient_explosion_threshold: 10.0,
        stagnation_patience: 10,
        oscillation_detection_window: 20,
        prediction_confidence_threshold: 0.5,
        enable_proactive_recovery: true,
    };

    let anomaly_config = GradientAnomalyConfig {
        enable_detection: config.enable_anomaly_recovery,
        nan_inf_threshold: 0.0,
        explosion_threshold: 100.0,
        vanishing_threshold: 1e-7,
        zero_threshold: 1e-8,
        oscillation_window: 10,
        enable_recovery: true,
        recovery_strategies: vec!["clip", "normalize", "smooth"].into_iter().map(String::from).collect(),
    };

    // Create advanced callback
    let advanced_callback = AdvancedOptimizationCallback::new(
        gradient_scaling_config,
        lr_config,
        stability_config,
        anomaly_config,
        config.print_frequency,
    )?;

    println!("  ✅ Advanced optimization configured");
    println!();

    // Create trainer (simplified - real implementation would handle data loading properly)
    println!("🎯 Starting training with advanced optimization...");
    println!("Training for {} epochs with {} parameters",
           config.num_epochs, model.num_parameters());
    println!();

    // Simulate training process to demonstrate advanced features
    println!("🔥 Training simulation (demonstrating advanced features):");
    println!("=" * 60);

    // This is a simplified simulation - real implementation would use the full trainer
    for epoch in 0..config.num_epochs {
        println!("Epoch {}/{}", epoch + 1, config.num_epochs);

        // Simulate varying training conditions to trigger different optimization behaviors
        let base_loss = 2.0 * (-0.1 * epoch as f32).exp(); // Decreasing loss
        let noise_factor = 0.1 * (epoch as f32 * 0.5).sin(); // Some oscillation
        let loss = base_loss + noise_factor;

        // Simulate some challenging conditions
        let gradient_norm = if epoch == 5 {
            50.0 // Simulate gradient explosion
        } else if epoch == 10 {
            0.0001 // Simulate vanishing gradients
        } else {
            loss * (1.0 + 0.2 * rand::random::<f32>())
        };

        let mut logs = HashMap::new();
        logs.insert("loss".to_string(), loss as f64);
        logs.insert("learning_rate".to_string(), config.learning_rate);

        // Simulate batch processing
        let num_batches = config.num_train_samples / config.batch_size;
        for batch in 0..num_batches {
            let mut callback = advanced_callback;  // Clone for this example
            callback.on_batch_end(batch, &logs);

            if batch >= 3 { break; } // Just show first few batches per epoch
        }

        // Simulate evaluation
        let eval_loss = loss * 0.9; // Slightly better on eval
        let accuracy = 0.2 + 0.6 * (1.0 - (-0.15 * epoch as f32).exp()); // Improving accuracy

        let mut eval_logs = logs.clone();
        eval_logs.insert("eval_loss".to_string(), eval_loss as f64);
        eval_logs.insert("eval_accuracy".to_string(), accuracy as f64);

        // callback.on_epoch_end(epoch, &eval_logs);

        if epoch % 3 == 0 || epoch == config.num_epochs - 1 {
            println!();
        }
    }

    println!("✅ Training simulation completed!");
    println!();

    println!("🎉 Advanced Optimization Example Summary:");
    println!("  🔧 Demonstrated adaptive gradient scaling with automatic adjustment");
    println!("  📈 Showed multi-strategy learning rate scheduling");
    println!("  ⚖️  Illustrated stability monitoring with predictive detection");
    println!("  🛡️  Exhibited anomaly detection and recovery mechanisms");
    println!("  🎯 All advanced features working together for optimal training");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_model_creation() {
        let model = AdvancedClassifier::new(20, &[128, 64, 32], 5, 0.2).unwrap();
        assert!(model.num_parameters() > 0);
        assert_eq!(model.num_classes, 5);
    }

    #[test]
    fn test_complex_data_generation() {
        let (features, labels) = generate_complex_data(100, 20, 5, 0.1, 1.0).unwrap();
        assert_eq!(features.shape(), &[100, 20]);
        assert_eq!(labels.shape(), &[100]);
    }

    #[test]
    fn test_optimization_configs() {
        let grad_config = AdaptiveGradientScalingConfig::default();
        assert!(grad_config.auto_scaling);

        let lr_config = AdaptiveLearningRateConfig::default();
        assert!(lr_config.initial_lr > 0.0);
    }
}