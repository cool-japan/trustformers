//! On-Device Fine-Tuning for Mobile Deployment
//!
//! This module provides infrastructure for performing fine-tuning directly
//! on mobile devices with memory and compute constraints.

use crate::{MemoryOptimization, MobileConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::error::{CoreError, Result};
use trustformers_core::Tensor;
use trustformers_core::TrustformersError;

/// On-device training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnDeviceTrainingConfig {
    /// Learning rate for fine-tuning
    pub learning_rate: f32,
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size (usually 1 for mobile)
    pub batch_size: usize,
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Use gradient checkpointing to save memory
    pub gradient_checkpointing: bool,
    /// Fine-tuning method
    pub method: FineTuningMethod,
    /// Memory optimization for training
    pub memory_optimization: MemoryOptimization,
    /// Maximum memory for training (MB)
    pub max_training_memory_mb: usize,
}

/// Fine-tuning methods suitable for mobile devices
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FineTuningMethod {
    /// Low-Rank Adaptation (LoRA) - memory efficient
    LoRA { rank: usize, alpha: f32 },
    /// Adapter layers - lightweight fine-tuning
    Adapter { bottleneck_size: usize },
    /// Prefix tuning - only tune prefix tokens
    PrefixTuning { prefix_length: usize },
    /// Full fine-tuning (not recommended for mobile)
    Full,
}

impl Default for OnDeviceTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            epochs: 3,
            batch_size: 1,                  // Mobile-friendly
            gradient_accumulation_steps: 8, // Simulate larger batches
            max_sequence_length: 128,       // Conservative for mobile
            gradient_checkpointing: true,
            method: FineTuningMethod::LoRA {
                rank: 8,
                alpha: 16.0,
            },
            memory_optimization: MemoryOptimization::Maximum,
            max_training_memory_mb: 512, // 512MB limit (mobile-friendly)
        }
    }
}

/// On-device trainer for mobile fine-tuning
pub struct OnDeviceTrainer {
    config: OnDeviceTrainingConfig,
    mobile_config: MobileConfig,
    model_params: Option<HashMap<String, Tensor>>,
    trainable_params: HashMap<String, Tensor>,
    optimizer_state: OptimizerState,
    training_stats: OnDeviceTrainingStats,
}

impl OnDeviceTrainer {
    /// Create new on-device trainer
    pub fn new(config: OnDeviceTrainingConfig, mobile_config: MobileConfig) -> Result<Self> {
        // Validate training configuration for mobile constraints
        Self::validate_training_config(&config, &mobile_config)?;

        Ok(Self {
            config,
            mobile_config,
            model_params: None,
            trainable_params: HashMap::new(),
            optimizer_state: OptimizerState::new(),
            training_stats: OnDeviceTrainingStats::new(),
        })
    }

    /// Initialize training with base model parameters
    pub fn initialize_training(&mut self, base_params: HashMap<String, Tensor>) -> Result<()> {
        // Initialize trainable parameters based on fine-tuning method
        match self.config.method {
            FineTuningMethod::LoRA { rank, alpha } => {
                self.initialize_lora_params(&base_params, rank, alpha)?;
            },
            FineTuningMethod::Adapter { bottleneck_size } => {
                self.initialize_adapter_params(&base_params, bottleneck_size)?;
            },
            FineTuningMethod::PrefixTuning { prefix_length } => {
                self.initialize_prefix_params(&base_params, prefix_length)?;
            },
            FineTuningMethod::Full => {
                // Not recommended for mobile - clone all parameters
                self.trainable_params = base_params.clone();
            },
        }

        self.model_params = Some(base_params);

        // Estimate training memory requirements
        let memory_estimate = self.estimate_training_memory()?;
        if memory_estimate > self.config.max_training_memory_mb {
            return Err(TrustformersError::runtime_error(format!(
                "Training requires {}MB but limit is {}MB",
                memory_estimate, self.config.max_training_memory_mb
            ))
            .into());
        }

        tracing::info!(
            "On-device training initialized with {} trainable parameters",
            self.trainable_params.len()
        );
        tracing::info!("Estimated training memory: {}MB", memory_estimate);

        Ok(())
    }

    /// Perform one training step
    pub fn training_step(&mut self, input: &Tensor, target: &Tensor) -> Result<f32> {
        // Forward pass with gradient computation
        let (output, loss) = self.forward_with_loss(input, target)?;

        // Backward pass (compute gradients)
        let gradients = self.backward_pass(&output, &loss)?;

        // Update trainable parameters
        self.update_parameters(&gradients)?;

        // Update training statistics
        self.training_stats.update_step(loss);

        Ok(loss)
    }

    /// Train on a dataset for specified epochs
    pub fn train(&mut self, dataset: &[(Tensor, Tensor)]) -> Result<OnDeviceTrainingStats> {
        tracing::info!(
            "Starting on-device training for {} epochs",
            self.config.epochs
        );

        for epoch in 0..self.config.epochs {
            let mut epoch_loss = 0.0;
            let mut step_count = 0;

            // Process dataset in mini-batches
            for batch in dataset.chunks(self.config.batch_size) {
                let mut batch_loss = 0.0;

                // Gradient accumulation
                for step in 0..self.config.gradient_accumulation_steps.min(batch.len()) {
                    if step < batch.len() {
                        let (input, target) = &batch[step];
                        let step_loss = self.training_step(input, target)?;
                        batch_loss += step_loss;
                    }
                }

                epoch_loss += batch_loss;
                step_count += 1;

                // Log progress
                if step_count % 10 == 0 {
                    tracing::debug!(
                        "Epoch {}, Step {}, Loss: {:.4}",
                        epoch,
                        step_count,
                        batch_loss / self.config.gradient_accumulation_steps as f32
                    );
                }

                // Check memory usage
                if self.should_trigger_gc() {
                    self.mobile_gc()?;
                }
            }

            let avg_epoch_loss = epoch_loss / step_count as f32;
            self.training_stats.update_epoch(epoch, avg_epoch_loss);

            tracing::info!(
                "Epoch {} completed. Average loss: {:.4}",
                epoch,
                avg_epoch_loss
            );
        }

        tracing::info!("On-device training completed successfully");
        Ok(self.training_stats.clone())
    }

    /// Get trained parameters (only the fine-tuned ones)
    pub fn get_trained_parameters(&self) -> &HashMap<String, Tensor> {
        &self.trainable_params
    }

    /// Get training statistics
    pub fn get_training_stats(&self) -> &OnDeviceTrainingStats {
        &self.training_stats
    }

    /// Save training checkpoint for resuming
    pub fn save_checkpoint(&self) -> Result<OnDeviceCheckpoint> {
        Ok(OnDeviceCheckpoint {
            trainable_params: self.trainable_params.clone(),
            optimizer_state: self.optimizer_state.clone(),
            training_stats: self.training_stats.clone(),
            config: self.config.clone(),
        })
    }

    /// Load training checkpoint
    pub fn load_checkpoint(&mut self, checkpoint: OnDeviceCheckpoint) -> Result<()> {
        self.trainable_params = checkpoint.trainable_params;
        self.optimizer_state = checkpoint.optimizer_state;
        self.training_stats = checkpoint.training_stats;
        self.config = checkpoint.config;

        tracing::info!("Training checkpoint loaded successfully");
        Ok(())
    }

    // Private implementation methods

    fn validate_training_config(
        config: &OnDeviceTrainingConfig,
        mobile_config: &MobileConfig,
    ) -> Result<()> {
        // Check memory constraints
        if config.max_training_memory_mb > mobile_config.max_memory_mb {
            return Err(TrustformersError::config_error(
                "Training memory limit exceeds mobile memory limit",
                "mobile training validation",
            )
            .into());
        }

        // Check batch size is mobile-friendly
        if config.batch_size > 4 {
            return Err(TrustformersError::config_error(
                "Batch size too large for mobile training",
                "mobile training validation",
            )
            .into());
        }

        // Check sequence length is reasonable
        if config.max_sequence_length > 512 {
            return Err(TrustformersError::config_error(
                "Sequence length too long for mobile training",
                "mobile training validation",
            )
            .into());
        }

        Ok(())
    }

    fn initialize_lora_params(
        &mut self,
        base_params: &HashMap<String, Tensor>,
        rank: usize,
        alpha: f32,
    ) -> Result<()> {
        // Initialize LoRA parameters (A and B matrices)
        for (name, param) in base_params {
            if self.should_apply_lora(name) {
                let shape = param.shape();
                if shape.len() == 2 {
                    // For linear layers, create A and B matrices
                    let lora_a = Tensor::randn(&[shape[0], rank])?;
                    let lora_b = Tensor::zeros(&[rank, shape[1]])?; // Initialize B to zero

                    self.trainable_params.insert(format!("{}.lora_A", name), lora_a);
                    self.trainable_params.insert(format!("{}.lora_B", name), lora_b);
                }
            }
        }

        tracing::info!(
            "LoRA parameters initialized with rank {} and alpha {}",
            rank,
            alpha
        );
        Ok(())
    }

    fn initialize_adapter_params(
        &mut self,
        base_params: &HashMap<String, Tensor>,
        bottleneck_size: usize,
    ) -> Result<()> {
        // Initialize adapter layer parameters
        for (name, param) in base_params {
            if self.should_apply_adapter(name) {
                let shape = param.shape();
                if shape.len() == 2 {
                    // Create bottleneck adapter layers
                    let down_proj = Tensor::randn(&[shape[1], bottleneck_size])?;
                    let up_proj = Tensor::randn(&[bottleneck_size, shape[1]])?;

                    self.trainable_params.insert(format!("{}.adapter_down", name), down_proj);
                    self.trainable_params.insert(format!("{}.adapter_up", name), up_proj);
                }
            }
        }

        tracing::info!(
            "Adapter parameters initialized with bottleneck size {}",
            bottleneck_size
        );
        Ok(())
    }

    fn initialize_prefix_params(
        &mut self,
        base_params: &HashMap<String, Tensor>,
        prefix_length: usize,
    ) -> Result<()> {
        // Initialize prefix tuning parameters
        for (name, param) in base_params {
            if name.contains("embed") {
                let shape = param.shape();
                if shape.len() == 2 {
                    // Create prefix embeddings
                    let prefix_embed = Tensor::randn(&[prefix_length, shape[1]])?;
                    self.trainable_params.insert(format!("{}.prefix", name), prefix_embed);
                }
            }
        }

        tracing::info!(
            "Prefix tuning parameters initialized with prefix length {}",
            prefix_length
        );
        Ok(())
    }

    fn should_apply_lora(&self, param_name: &str) -> bool {
        // Apply LoRA to attention and MLP layers
        param_name.contains("attention")
            || param_name.contains("mlp")
            || param_name.contains("linear")
    }

    fn should_apply_adapter(&self, param_name: &str) -> bool {
        // Apply adapters to transformer layers
        param_name.contains("layer") && param_name.contains("linear")
    }

    fn estimate_training_memory(&self) -> Result<usize> {
        let mut total_memory = 0;

        // Model parameters memory
        if let Some(ref params) = self.model_params {
            for param in params.values() {
                total_memory += param.memory_usage();
            }
        }

        // Trainable parameters memory
        for param in self.trainable_params.values() {
            total_memory += param.memory_usage();
        }

        // Gradient memory (same as parameters)
        for param in self.trainable_params.values() {
            total_memory += param.memory_usage();
        }

        // Optimizer state memory (momentum, etc.)
        total_memory += total_memory / 2; // Estimate 50% overhead

        // Convert to MB
        Ok(total_memory / (1024 * 1024))
    }

    fn forward_with_loss(&self, input: &Tensor, target: &Tensor) -> Result<(Tensor, f32)> {
        // Simplified forward pass with loss computation
        // In practice, this would call the actual model forward pass
        let output = input.clone(); // Placeholder
        let loss = 0.5; // Placeholder loss
        Ok((output, loss))
    }

    fn backward_pass(&self, _output: &Tensor, _loss: &f32) -> Result<HashMap<String, Tensor>> {
        // Simplified backward pass to compute gradients
        // In practice, this would compute actual gradients
        let mut gradients = HashMap::new();

        for (name, param) in &self.trainable_params {
            let grad = Tensor::randn(&param.shape())?; // Placeholder gradient
            gradients.insert(name.clone(), grad);
        }

        Ok(gradients)
    }

    fn update_parameters(&mut self, gradients: &HashMap<String, Tensor>) -> Result<()> {
        // Simple SGD update (in practice, would use Adam or other optimizers)
        for (name, grad) in gradients {
            if let Some(param) = self.trainable_params.get_mut(name) {
                // param = param - learning_rate * grad
                let scaled_grad = grad.scalar_mul(self.config.learning_rate)?;
                *param = param.sub(&scaled_grad)?;
            }
        }

        Ok(())
    }

    fn should_trigger_gc(&self) -> bool {
        // Trigger GC based on memory pressure (simplified)
        self.training_stats.current_step % 50 == 0
    }

    fn mobile_gc(&self) -> Result<()> {
        // Trigger mobile-specific garbage collection
        tracing::debug!("Triggering mobile garbage collection");
        Ok(())
    }
}

/// Optimizer state for on-device training
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OptimizerState {
    #[serde(skip)]
    momentum: HashMap<String, Tensor>,
    step_count: usize,
}

impl OptimizerState {
    fn new() -> Self {
        Self {
            momentum: HashMap::new(),
            step_count: 0,
        }
    }
}

/// Training statistics for on-device fine-tuning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnDeviceTrainingStats {
    /// Current training step
    pub current_step: usize,
    /// Current epoch
    pub current_epoch: usize,
    /// Average loss over all steps
    pub avg_loss: f32,
    /// Loss history per epoch
    pub epoch_losses: Vec<f32>,
    /// Total training time in seconds
    pub total_training_time_seconds: f32,
    /// Memory usage during training (MB)
    pub peak_memory_usage_mb: usize,
}

impl OnDeviceTrainingStats {
    fn new() -> Self {
        Self {
            current_step: 0,
            current_epoch: 0,
            avg_loss: 0.0,
            epoch_losses: Vec::new(),
            total_training_time_seconds: 0.0,
            peak_memory_usage_mb: 0,
        }
    }

    fn update_step(&mut self, loss: f32) {
        self.current_step += 1;

        // Update running average loss
        let alpha = 0.1;
        if self.current_step == 1 {
            self.avg_loss = loss;
        } else {
            self.avg_loss = alpha * loss + (1.0 - alpha) * self.avg_loss;
        }
    }

    fn update_epoch(&mut self, epoch: usize, epoch_loss: f32) {
        self.current_epoch = epoch;
        self.epoch_losses.push(epoch_loss);
    }
}

/// Training checkpoint for saving/loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnDeviceCheckpoint {
    #[serde(skip)]
    pub trainable_params: HashMap<String, Tensor>,
    pub optimizer_state: OptimizerState,
    pub training_stats: OnDeviceTrainingStats,
    pub config: OnDeviceTrainingConfig,
}

/// Mobile-specific training utilities
pub struct MobileTrainingUtils;

impl MobileTrainingUtils {
    /// Create optimized training config for mobile device
    pub fn create_mobile_training_config(
        available_memory_mb: usize,
        device_performance: MobilePerformanceLevel,
    ) -> OnDeviceTrainingConfig {
        match device_performance {
            MobilePerformanceLevel::Low => Self::low_end_config(available_memory_mb),
            MobilePerformanceLevel::Medium => Self::mid_range_config(available_memory_mb),
            MobilePerformanceLevel::High => Self::high_end_config(available_memory_mb),
        }
    }

    fn low_end_config(memory_mb: usize) -> OnDeviceTrainingConfig {
        OnDeviceTrainingConfig {
            learning_rate: 5e-5,
            epochs: 1, // Single epoch for speed
            batch_size: 1,
            gradient_accumulation_steps: 16, // Simulate larger batches
            max_sequence_length: 64,         // Very short sequences
            gradient_checkpointing: true,
            method: FineTuningMethod::LoRA {
                rank: 4,
                alpha: 8.0,
            }, // Small rank
            memory_optimization: MemoryOptimization::Maximum,
            max_training_memory_mb: (memory_mb / 4).max(128), // Very conservative
        }
    }

    fn mid_range_config(memory_mb: usize) -> OnDeviceTrainingConfig {
        OnDeviceTrainingConfig {
            learning_rate: 1e-4,
            epochs: 2,
            batch_size: 1,
            gradient_accumulation_steps: 8,
            max_sequence_length: 128,
            gradient_checkpointing: true,
            method: FineTuningMethod::LoRA {
                rank: 8,
                alpha: 16.0,
            },
            memory_optimization: MemoryOptimization::Balanced,
            max_training_memory_mb: (memory_mb / 2).max(256),
        }
    }

    fn high_end_config(memory_mb: usize) -> OnDeviceTrainingConfig {
        OnDeviceTrainingConfig {
            learning_rate: 2e-4,
            epochs: 3,
            batch_size: 2,
            gradient_accumulation_steps: 4,
            max_sequence_length: 256,
            gradient_checkpointing: false, // Can afford more memory
            method: FineTuningMethod::LoRA {
                rank: 16,
                alpha: 32.0,
            },
            memory_optimization: MemoryOptimization::Balanced,
            max_training_memory_mb: (memory_mb * 3 / 4).max(512),
        }
    }
}

/// Mobile device performance levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MobilePerformanceLevel {
    /// Low-end devices (< 2GB RAM)
    Low,
    /// Mid-range devices (2-6GB RAM)
    Medium,
    /// High-end devices (> 6GB RAM)
    High,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_on_device_training_config() {
        let config = OnDeviceTrainingConfig::default();
        assert_eq!(config.batch_size, 1);
        assert!(config.gradient_checkpointing);
        assert!(matches!(config.method, FineTuningMethod::LoRA { .. }));
    }

    #[test]
    fn test_on_device_trainer_creation() {
        let training_config = OnDeviceTrainingConfig::default();
        let mobile_config = crate::MobileConfig::default();

        let trainer = OnDeviceTrainer::new(training_config, mobile_config);
        assert!(trainer.is_ok());
    }

    #[test]
    fn test_training_initialization() {
        let training_config = OnDeviceTrainingConfig::default();
        let mobile_config = crate::MobileConfig::default();
        let mut trainer = OnDeviceTrainer::new(training_config, mobile_config).unwrap();

        let mut base_params = HashMap::new();
        base_params.insert(
            "attention.linear".to_string(),
            Tensor::randn(&[128, 128]).unwrap(),
        );

        let result = trainer.initialize_training(base_params);
        assert!(result.is_ok());
        assert!(!trainer.trainable_params.is_empty());
    }

    #[test]
    fn test_mobile_training_utils() {
        let config = MobileTrainingUtils::create_mobile_training_config(
            2048,
            MobilePerformanceLevel::Medium,
        );
        assert_eq!(config.batch_size, 1);
        assert!(config.max_training_memory_mb <= 1024);

        let low_config =
            MobileTrainingUtils::create_mobile_training_config(1024, MobilePerformanceLevel::Low);
        assert_eq!(low_config.epochs, 1);
        assert_eq!(low_config.max_sequence_length, 64);
    }

    #[test]
    fn test_fine_tuning_methods() {
        let lora = FineTuningMethod::LoRA {
            rank: 8,
            alpha: 16.0,
        };
        let adapter = FineTuningMethod::Adapter {
            bottleneck_size: 64,
        };
        let prefix = FineTuningMethod::PrefixTuning { prefix_length: 16 };

        assert!(matches!(lora, FineTuningMethod::LoRA { .. }));
        assert!(matches!(adapter, FineTuningMethod::Adapter { .. }));
        assert!(matches!(prefix, FineTuningMethod::PrefixTuning { .. }));
    }

    #[test]
    fn test_training_stats() {
        let mut stats = OnDeviceTrainingStats::new();
        assert_eq!(stats.current_step, 0);
        assert_eq!(stats.avg_loss, 0.0);

        stats.update_step(1.0);
        assert_eq!(stats.current_step, 1);
        assert_eq!(stats.avg_loss, 1.0);

        stats.update_epoch(0, 0.8);
        assert_eq!(stats.epoch_losses.len(), 1);
        assert_eq!(stats.epoch_losses[0], 0.8);
    }
}
