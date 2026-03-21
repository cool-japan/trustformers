/*!
# Progressive Training Module

This module provides progressive training capabilities for transformer models, enabling
models to grow in capacity during training for improved efficiency and performance.

## Features

- **Layer Progressive Training**: Gradually add layers during training
- **Width Progressive Training**: Progressively increase hidden dimensions
- **Head Progressive Training**: Add attention heads incrementally
- **Multiple Growth Strategies**: Linear, exponential, adaptive growth schedules
- **Smooth Transitions**: Gradual parameter initialization and adaptation
- **Curriculum Integration**: Compatible with curriculum learning frameworks

## Usage

```rust
use trustformers_models::progressive_training::{
    ProgressiveTrainer, ProgressiveConfig, GrowthStrategy, GrowthDimension
};

let config = ProgressiveConfig {
    growth_dimension: GrowthDimension::Layers,
    growth_strategy: GrowthStrategy::Linear,
    initial_size: 6,
    final_size: 12,
    growth_epochs: vec![10, 20, 30],
    warmup_steps: 1000,
};

let mut trainer = ProgressiveTrainer::new(config)?;
```
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::errors::TrustformersError;

/// Configuration for progressive training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveConfig {
    /// Which dimension to grow progressively
    pub growth_dimension: GrowthDimension,
    /// Growth strategy to use
    pub growth_strategy: GrowthStrategy,
    /// Initial model size (layers, hidden dim, heads, etc.)
    pub initial_size: usize,
    /// Final model size
    pub final_size: usize,
    /// Epochs at which to trigger growth
    pub growth_epochs: Vec<usize>,
    /// Steps to warm up after each growth
    pub warmup_steps: usize,
    /// Whether to initialize new parameters with zeros
    pub zero_init_new_params: bool,
    /// Learning rate scaling factor after growth
    pub lr_scaling_factor: f64,
    /// Whether to use gradual weight initialization
    pub gradual_initialization: bool,
    /// Smoothing factor for parameter transitions
    pub transition_smoothing: f64,
    /// Whether to freeze old parameters during warmup
    pub freeze_old_params_during_warmup: bool,
}

impl Default for ProgressiveConfig {
    fn default() -> Self {
        Self {
            growth_dimension: GrowthDimension::Layers,
            growth_strategy: GrowthStrategy::Linear,
            initial_size: 6,
            final_size: 12,
            growth_epochs: vec![10, 20, 30, 40],
            warmup_steps: 1000,
            zero_init_new_params: true,
            lr_scaling_factor: 0.5,
            gradual_initialization: true,
            transition_smoothing: 0.1,
            freeze_old_params_during_warmup: false,
        }
    }
}

/// Dimensions along which models can grow
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GrowthDimension {
    /// Add transformer layers
    Layers,
    /// Increase hidden dimension
    HiddenDim,
    /// Add attention heads
    AttentionHeads,
    /// Increase intermediate (FFN) dimension
    IntermediateDim,
    /// Grow vocabulary size
    VocabSize,
    /// Multi-dimensional growth (combined)
    MultiDimensional,
}

/// Growth strategies for progressive training
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GrowthStrategy {
    /// Linear growth at fixed intervals
    Linear,
    /// Exponential growth (larger jumps later)
    Exponential,
    /// Logarithmic growth (larger jumps earlier)
    Logarithmic,
    /// Adaptive growth based on learning progress
    Adaptive,
    /// Custom growth schedule
    Custom,
    /// Staged growth (fixed size increases)
    Staged,
}

/// Growth schedule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrowthSchedule {
    /// Epoch -> new size mapping
    pub growth_points: HashMap<usize, usize>,
    /// Whether the schedule is adaptive
    pub adaptive: bool,
    /// Minimum epochs between growth steps
    pub min_growth_interval: usize,
    /// Maximum growth per step
    pub max_growth_per_step: usize,
}

/// Progressive trainer for growing models during training
pub struct ProgressiveTrainer {
    config: ProgressiveConfig,
    current_size: usize,
    current_epoch: usize,
    current_step: usize,
    growth_schedule: GrowthSchedule,
    growth_history: Vec<GrowthEvent>,
    warmup_remaining: usize,
    frozen_parameters: HashSet<String>,
    learning_progress: LearningProgress,
}

use std::collections::HashSet;

impl ProgressiveTrainer {
    /// Create a new progressive trainer
    pub fn new(config: ProgressiveConfig) -> Result<Self, TrustformersError> {
        let growth_schedule = Self::create_growth_schedule(&config)?;

        Ok(Self {
            current_size: config.initial_size,
            current_epoch: 0,
            current_step: 0,
            growth_schedule,
            growth_history: Vec::new(),
            warmup_remaining: 0,
            frozen_parameters: HashSet::new(),
            learning_progress: LearningProgress::new(),
            config,
        })
    }

    /// Create growth schedule based on configuration
    fn create_growth_schedule(
        config: &ProgressiveConfig,
    ) -> Result<GrowthSchedule, TrustformersError> {
        let mut growth_points = HashMap::new();

        match config.growth_strategy {
            GrowthStrategy::Linear => {
                let total_growth = config.final_size - config.initial_size;
                let num_steps = config.growth_epochs.len();
                let growth_per_step = total_growth / num_steps.max(1);

                for (i, &epoch) in config.growth_epochs.iter().enumerate() {
                    let new_size = config.initial_size + (i + 1) * growth_per_step;
                    growth_points.insert(epoch, new_size.min(config.final_size));
                }
            },
            GrowthStrategy::Exponential => {
                for (i, &epoch) in config.growth_epochs.iter().enumerate() {
                    let progress = (i + 1) as f64 / config.growth_epochs.len() as f64;
                    let exp_progress = progress.powf(2.0);
                    let new_size = config.initial_size
                        + ((config.final_size - config.initial_size) as f64 * exp_progress)
                            as usize;
                    growth_points.insert(epoch, new_size.min(config.final_size));
                }
            },
            GrowthStrategy::Logarithmic => {
                for (i, &epoch) in config.growth_epochs.iter().enumerate() {
                    let progress = (i + 1) as f64 / config.growth_epochs.len() as f64;
                    let log_progress = (1.0 + progress).ln() / (2.0_f64).ln();
                    let new_size = config.initial_size
                        + ((config.final_size - config.initial_size) as f64 * log_progress)
                            as usize;
                    growth_points.insert(epoch, new_size.min(config.final_size));
                }
            },
            GrowthStrategy::Adaptive => {
                // Initial schedule, will be updated based on learning progress
                for (i, &epoch) in config.growth_epochs.iter().enumerate() {
                    let progress = (i + 1) as f64 / config.growth_epochs.len() as f64;
                    let new_size = config.initial_size
                        + ((config.final_size - config.initial_size) as f64 * progress) as usize;
                    growth_points.insert(epoch, new_size.min(config.final_size));
                }
            },
            GrowthStrategy::Staged => {
                let stage_size =
                    (config.final_size - config.initial_size) / config.growth_epochs.len().max(1);
                for (i, &epoch) in config.growth_epochs.iter().enumerate() {
                    let new_size = config.initial_size + (i + 1) * stage_size;
                    growth_points.insert(epoch, new_size.min(config.final_size));
                }
            },
            GrowthStrategy::Custom => {
                // Custom schedule should be provided externally
            },
        }

        Ok(GrowthSchedule {
            growth_points,
            adaptive: matches!(config.growth_strategy, GrowthStrategy::Adaptive),
            min_growth_interval: 5,
            max_growth_per_step: (config.final_size - config.initial_size) / 2,
        })
    }

    /// Check if model should grow at current epoch
    pub fn should_grow(&self, epoch: usize) -> bool {
        if self.warmup_remaining > 0 {
            return false;
        }

        if let Some(&target_size) = self.growth_schedule.growth_points.get(&epoch) {
            return target_size > self.current_size;
        }

        // Adaptive growth based on learning plateau
        if self.growth_schedule.adaptive {
            return self.learning_progress.should_trigger_growth(epoch);
        }

        false
    }

    /// Grow the model to the target size
    pub fn grow_model(
        &mut self,
        model: &mut dyn ProgressiveModel,
        epoch: usize,
    ) -> Result<GrowthResult, TrustformersError> {
        let target_size = self
            .growth_schedule
            .growth_points
            .get(&epoch)
            .copied()
            .unwrap_or_else(|| self.determine_adaptive_growth_size(epoch));

        if target_size <= self.current_size {
            return Ok(GrowthResult::NoGrowthNeeded);
        }

        let growth_amount = target_size - self.current_size;
        let start_time = std::time::Instant::now();

        // Perform the actual growth based on dimension
        let growth_info = match self.config.growth_dimension {
            GrowthDimension::Layers => self.grow_layers(model, growth_amount)?,
            GrowthDimension::HiddenDim => self.grow_hidden_dimension(model, target_size)?,
            GrowthDimension::AttentionHeads => self.grow_attention_heads(model, target_size)?,
            GrowthDimension::IntermediateDim => {
                self.grow_intermediate_dimension(model, target_size)?
            },
            GrowthDimension::VocabSize => self.grow_vocabulary(model, target_size)?,
            GrowthDimension::MultiDimensional => self.grow_multi_dimensional(model, target_size)?,
        };

        // Record growth event
        let growth_event = GrowthEvent {
            epoch,
            old_size: self.current_size,
            new_size: target_size,
            growth_dimension: self.config.growth_dimension,
            growth_time: start_time.elapsed(),
            growth_info: growth_info.clone(),
        };

        self.growth_history.push(growth_event);
        self.current_size = target_size;
        self.warmup_remaining = self.config.warmup_steps;

        // Optionally freeze old parameters during warmup
        if self.config.freeze_old_params_during_warmup {
            self.freeze_old_parameters(model)?;
        }

        Ok(GrowthResult::Grown {
            old_size: self.current_size,
            new_size: target_size,
            growth_info,
        })
    }

    /// Grow model by adding layers
    fn grow_layers(
        &mut self,
        model: &mut dyn ProgressiveModel,
        num_layers: usize,
    ) -> Result<GrowthInfo, TrustformersError> {
        let mut added_parameters = 0;
        let mut initialization_method = String::new();

        for i in 0..num_layers {
            let layer_params = model.add_layer(self.current_size + i)?;
            added_parameters += layer_params;

            if self.config.gradual_initialization {
                // Initialize with small weights that gradually increase
                let scale = self.config.transition_smoothing * (i + 1) as f64 / num_layers as f64;
                model.scale_layer_parameters(self.current_size + i, scale)?;
                initialization_method = format!("Gradual scaling (factor: {})", scale);
            } else if self.config.zero_init_new_params {
                model.zero_initialize_layer(self.current_size + i)?;
                initialization_method = "Zero initialization".to_string();
            }
        }

        Ok(GrowthInfo {
            added_parameters,
            initialization_method,
            growth_type: "Layer addition".to_string(),
        })
    }

    /// Grow model by increasing hidden dimension
    fn grow_hidden_dimension(
        &mut self,
        model: &mut dyn ProgressiveModel,
        target_dim: usize,
    ) -> Result<GrowthInfo, TrustformersError> {
        let old_dim = model.get_hidden_dimension()?;
        let _growth = target_dim - old_dim;

        let added_parameters = model.expand_hidden_dimension(target_dim)?;

        // Initialize new dimensions
        if self.config.gradual_initialization {
            model.initialize_expanded_dimensions(
                old_dim,
                target_dim,
                self.config.transition_smoothing,
            )?;
        }

        Ok(GrowthInfo {
            added_parameters,
            initialization_method: "Hidden dimension expansion".to_string(),
            growth_type: format!("Hidden dim: {} -> {}", old_dim, target_dim),
        })
    }

    /// Grow model by adding attention heads
    fn grow_attention_heads(
        &mut self,
        model: &mut dyn ProgressiveModel,
        target_heads: usize,
    ) -> Result<GrowthInfo, TrustformersError> {
        let old_heads = model.get_num_attention_heads()?;
        let added_parameters = model.expand_attention_heads(target_heads)?;

        Ok(GrowthInfo {
            added_parameters,
            initialization_method: "Attention head expansion".to_string(),
            growth_type: format!("Attention heads: {} -> {}", old_heads, target_heads),
        })
    }

    /// Grow intermediate (FFN) dimension
    fn grow_intermediate_dimension(
        &mut self,
        model: &mut dyn ProgressiveModel,
        target_dim: usize,
    ) -> Result<GrowthInfo, TrustformersError> {
        let old_dim = model.get_intermediate_dimension()?;
        let added_parameters = model.expand_intermediate_dimension(target_dim)?;

        Ok(GrowthInfo {
            added_parameters,
            initialization_method: "Intermediate dimension expansion".to_string(),
            growth_type: format!("Intermediate dim: {} -> {}", old_dim, target_dim),
        })
    }

    /// Grow vocabulary size
    fn grow_vocabulary(
        &mut self,
        model: &mut dyn ProgressiveModel,
        target_vocab: usize,
    ) -> Result<GrowthInfo, TrustformersError> {
        let old_vocab = model.get_vocab_size()?;
        let added_parameters = model.expand_vocabulary(target_vocab)?;

        Ok(GrowthInfo {
            added_parameters,
            initialization_method: "Vocabulary expansion".to_string(),
            growth_type: format!("Vocab size: {} -> {}", old_vocab, target_vocab),
        })
    }

    /// Multi-dimensional growth
    fn grow_multi_dimensional(
        &mut self,
        model: &mut dyn ProgressiveModel,
        _target_size: usize,
    ) -> Result<GrowthInfo, TrustformersError> {
        // Implement coordinated growth across multiple dimensions
        let mut total_added_parameters = 0;

        // Grow layers first
        if self.current_size < self.config.final_size / 2 {
            let layer_growth = self.grow_layers(model, 1)?;
            total_added_parameters += layer_growth.added_parameters;
        }

        // Then grow width
        let current_hidden = model.get_hidden_dimension()?;
        if current_hidden < 1024 {
            // Example threshold
            let width_growth = self.grow_hidden_dimension(model, current_hidden + 64)?;
            total_added_parameters += width_growth.added_parameters;
        }

        Ok(GrowthInfo {
            added_parameters: total_added_parameters,
            initialization_method: "Multi-dimensional growth".to_string(),
            growth_type: "Combined layer and width growth".to_string(),
        })
    }

    /// Determine adaptive growth size based on learning progress
    fn determine_adaptive_growth_size(&self, _epoch: usize) -> usize {
        // Adaptive logic based on learning plateau detection
        if self.learning_progress.is_plateau() {
            (self.current_size as f64 * 1.2) as usize // 20% increase
        } else {
            self.current_size + 1 // Conservative growth
        }
    }

    /// Freeze old parameters during warmup
    fn freeze_old_parameters(
        &mut self,
        model: &mut dyn ProgressiveModel,
    ) -> Result<(), TrustformersError> {
        let old_param_names = model.get_parameter_names()?;
        for name in old_param_names {
            self.frozen_parameters.insert(name);
        }
        model.freeze_parameters(&self.frozen_parameters)?;
        Ok(())
    }

    /// Unfreeze parameters after warmup
    fn unfreeze_parameters(
        &mut self,
        model: &mut dyn ProgressiveModel,
    ) -> Result<(), TrustformersError> {
        model.unfreeze_parameters(&self.frozen_parameters)?;
        self.frozen_parameters.clear();
        Ok(())
    }

    /// Update training state
    pub fn step(
        &mut self,
        model: &mut dyn ProgressiveModel,
        loss: f64,
    ) -> Result<(), TrustformersError> {
        self.current_step += 1;

        // Update learning progress
        self.learning_progress.update(loss);

        // Handle warmup
        if self.warmup_remaining > 0 {
            self.warmup_remaining -= 1;
            if self.warmup_remaining == 0 && !self.frozen_parameters.is_empty() {
                self.unfreeze_parameters(model)?;
            }
        }

        Ok(())
    }

    /// Set current epoch
    pub fn set_epoch(&mut self, epoch: usize) {
        self.current_epoch = epoch;
        self.learning_progress.new_epoch();
    }

    /// Get current model size
    pub fn current_size(&self) -> usize {
        self.current_size
    }

    /// Get growth history
    pub fn growth_history(&self) -> &[GrowthEvent] {
        &self.growth_history
    }

    /// Get warmup status
    pub fn is_in_warmup(&self) -> bool {
        self.warmup_remaining > 0
    }

    /// Get learning progress
    pub fn learning_progress(&self) -> &LearningProgress {
        &self.learning_progress
    }

    /// Update growth schedule (for adaptive training)
    pub fn update_growth_schedule(&mut self, new_points: HashMap<usize, usize>) {
        self.growth_schedule.growth_points.extend(new_points);
    }
}

/// Information about a growth operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrowthInfo {
    /// Number of parameters added
    pub added_parameters: usize,
    /// Method used to initialize new parameters
    pub initialization_method: String,
    /// Type of growth performed
    pub growth_type: String,
}

/// Result of a growth operation
#[derive(Debug)]
pub enum GrowthResult {
    /// Model was grown successfully
    Grown {
        old_size: usize,
        new_size: usize,
        growth_info: GrowthInfo,
    },
    /// No growth was needed
    NoGrowthNeeded,
}

/// Record of a growth event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrowthEvent {
    /// Epoch when growth occurred
    pub epoch: usize,
    /// Size before growth
    pub old_size: usize,
    /// Size after growth
    pub new_size: usize,
    /// Dimension that was grown
    pub growth_dimension: GrowthDimension,
    /// Time taken for growth operation
    pub growth_time: std::time::Duration,
    /// Additional growth information
    pub growth_info: GrowthInfo,
}

/// Tracks learning progress for adaptive growth decisions
#[derive(Debug)]
pub struct LearningProgress {
    loss_history: Vec<f64>,
    recent_losses: std::collections::VecDeque<f64>,
    plateau_threshold: f64,
    plateau_patience: usize,
    #[allow(dead_code)]
    improvement_threshold: f64,
    current_epoch: usize,
}

impl Default for LearningProgress {
    fn default() -> Self {
        Self::new()
    }
}

impl LearningProgress {
    pub fn new() -> Self {
        Self {
            loss_history: Vec::new(),
            recent_losses: std::collections::VecDeque::with_capacity(10),
            plateau_threshold: 0.001,
            plateau_patience: 5,
            improvement_threshold: 0.01,
            current_epoch: 0,
        }
    }

    pub fn update(&mut self, loss: f64) {
        self.loss_history.push(loss);
        self.recent_losses.push_back(loss);
        if self.recent_losses.len() > 10 {
            self.recent_losses.pop_front();
        }
    }

    pub fn is_plateau(&self) -> bool {
        if self.recent_losses.len() < self.plateau_patience {
            return false;
        }

        let recent_avg = self.recent_losses.iter().sum::<f64>() / self.recent_losses.len() as f64;
        let older_losses = &self.loss_history[self.loss_history.len().saturating_sub(20)
            ..self.loss_history.len().saturating_sub(10)];

        if older_losses.is_empty() {
            return false;
        }

        let older_avg = older_losses.iter().sum::<f64>() / older_losses.len() as f64;
        let improvement = older_avg - recent_avg;

        improvement < self.plateau_threshold
    }

    pub fn should_trigger_growth(&self, _epoch: usize) -> bool {
        self.is_plateau() && self.loss_history.len() > 100
    }

    pub fn new_epoch(&mut self) {
        self.current_epoch += 1;
    }
}

/// Trait that models must implement to support progressive training
pub trait ProgressiveModel {
    /// Add a new layer to the model
    fn add_layer(&mut self, layer_index: usize) -> Result<usize, TrustformersError>;

    /// Expand hidden dimension to target size
    fn expand_hidden_dimension(&mut self, target_dim: usize) -> Result<usize, TrustformersError>;

    /// Expand number of attention heads
    fn expand_attention_heads(&mut self, target_heads: usize) -> Result<usize, TrustformersError>;

    /// Expand intermediate (FFN) dimension
    fn expand_intermediate_dimension(
        &mut self,
        target_dim: usize,
    ) -> Result<usize, TrustformersError>;

    /// Expand vocabulary size
    fn expand_vocabulary(&mut self, target_vocab: usize) -> Result<usize, TrustformersError>;

    /// Get current hidden dimension
    fn get_hidden_dimension(&self) -> Result<usize, TrustformersError>;

    /// Get current number of attention heads
    fn get_num_attention_heads(&self) -> Result<usize, TrustformersError>;

    /// Get current intermediate dimension
    fn get_intermediate_dimension(&self) -> Result<usize, TrustformersError>;

    /// Get current vocabulary size
    fn get_vocab_size(&self) -> Result<usize, TrustformersError>;

    /// Initialize a layer with zeros
    fn zero_initialize_layer(&mut self, layer_index: usize) -> Result<(), TrustformersError>;

    /// Scale layer parameters by a factor
    fn scale_layer_parameters(
        &mut self,
        layer_index: usize,
        scale: f64,
    ) -> Result<(), TrustformersError>;

    /// Initialize expanded dimensions with gradual scaling
    fn initialize_expanded_dimensions(
        &mut self,
        old_dim: usize,
        new_dim: usize,
        smoothing: f64,
    ) -> Result<(), TrustformersError>;

    /// Get names of all parameters
    fn get_parameter_names(&self) -> Result<Vec<String>, TrustformersError>;

    /// Freeze specified parameters
    fn freeze_parameters(&mut self, param_names: &HashSet<String>)
        -> Result<(), TrustformersError>;

    /// Unfreeze specified parameters
    fn unfreeze_parameters(
        &mut self,
        param_names: &HashSet<String>,
    ) -> Result<(), TrustformersError>;
}

/// Progressive training utilities
pub mod utils {

    /// Create a linear growth schedule
    pub fn create_linear_schedule(
        initial_size: usize,
        final_size: usize,
        num_steps: usize,
        start_epoch: usize,
        epoch_interval: usize,
    ) -> Vec<usize> {
        let _growth_per_step = (final_size - initial_size) / num_steps.max(1);
        (0..num_steps).map(|i| start_epoch + i * epoch_interval).collect()
    }

    /// Create an exponential growth schedule
    pub fn create_exponential_schedule(
        _initial_size: usize,
        _final_size: usize,
        num_steps: usize,
        start_epoch: usize,
        epoch_interval: usize,
    ) -> Vec<usize> {
        (0..num_steps)
            .map(|i| start_epoch + (epoch_interval as f64 * (1.5_f64.powi(i as i32))) as usize)
            .collect()
    }

    /// Estimate parameter count for a transformer model
    pub fn estimate_parameter_count(
        vocab_size: usize,
        hidden_dim: usize,
        num_layers: usize,
        _num_heads: usize,
        intermediate_dim: usize,
    ) -> usize {
        // Embedding layer
        let embedding_params = vocab_size * hidden_dim;

        // Per-layer parameters
        let attention_params = 4 * hidden_dim * hidden_dim; // Q, K, V, O projections
        let ffn_params = 2 * hidden_dim * intermediate_dim; // Up and down projections
        let norm_params = 2 * hidden_dim; // Layer norm
        let layer_params = attention_params + ffn_params + norm_params;

        // Total parameters
        embedding_params + num_layers * layer_params + hidden_dim // Final layer norm
    }

    /// Calculate optimal growth schedule based on target training time
    pub fn calculate_optimal_schedule(
        initial_size: usize,
        final_size: usize,
        total_epochs: usize,
        _computational_budget: f64,
    ) -> Vec<usize> {
        // Simple heuristic: grow more aggressively early when computation is cheaper
        let mut schedule = Vec::new();
        let num_growth_steps = ((final_size - initial_size) as f64).sqrt() as usize;

        for i in 0..num_growth_steps {
            let progress = i as f64 / num_growth_steps as f64;
            let epoch = (total_epochs as f64 * progress.sqrt()) as usize;
            schedule.push(epoch);
        }

        schedule
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progressive_config_default() {
        let config = ProgressiveConfig::default();
        assert_eq!(config.initial_size, 6);
        assert_eq!(config.final_size, 12);
        assert!(config.zero_init_new_params);
    }

    #[test]
    fn test_growth_schedule_creation() {
        let config = ProgressiveConfig {
            growth_strategy: GrowthStrategy::Linear,
            initial_size: 4,
            final_size: 12,
            growth_epochs: vec![10, 20, 30, 40],
            ..Default::default()
        };

        let schedule =
            ProgressiveTrainer::create_growth_schedule(&config).expect("operation failed");
        assert!(!schedule.growth_points.is_empty());
        assert_eq!(schedule.growth_points.len(), 4);
    }

    #[test]
    fn test_progressive_trainer_creation() {
        let config = ProgressiveConfig::default();
        let trainer = ProgressiveTrainer::new(config);
        assert!(trainer.is_ok());

        let trainer = trainer.expect("operation failed");
        assert_eq!(trainer.current_size(), 6);
        assert!(!trainer.is_in_warmup());
    }

    #[test]
    fn test_learning_progress() {
        let mut progress = LearningProgress::new();

        // Add some losses
        for i in 0..20 {
            progress.update(1.0 - i as f64 * 0.01); // Decreasing loss
        }

        assert!(!progress.is_plateau());

        // Add plateau losses - need enough constant losses so both
        // "older" (indices len-20 to len-10) and "recent" (last 10)
        // are all constant, making improvement near 0
        for _ in 0..25 {
            progress.update(0.8); // Constant loss
        }

        assert!(progress.is_plateau());
    }

    #[test]
    fn test_growth_dimensions() {
        assert_eq!(GrowthDimension::Layers as u8, 0);
        assert_ne!(GrowthDimension::Layers, GrowthDimension::HiddenDim);
    }

    #[test]
    fn test_growth_strategies() {
        assert_eq!(GrowthStrategy::Linear as u8, 0);
        assert_ne!(GrowthStrategy::Linear, GrowthStrategy::Exponential);
    }

    #[test]
    fn test_utils_parameter_estimation() {
        let params = utils::estimate_parameter_count(30000, 768, 12, 12, 3072);
        assert!(params > 100_000_000); // Should be > 100M for BERT-base size
    }

    #[test]
    fn test_utils_linear_schedule() {
        let schedule = utils::create_linear_schedule(6, 12, 3, 10, 5);
        assert_eq!(schedule.len(), 3);
        assert_eq!(schedule[0], 10);
        assert_eq!(schedule[1], 15);
        assert_eq!(schedule[2], 20);
    }
}
