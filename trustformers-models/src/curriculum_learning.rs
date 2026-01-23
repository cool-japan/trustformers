//! # Curriculum Learning Framework
//!
//! This module provides a comprehensive framework for curriculum learning,
//! enabling models to learn from training data in a structured, progressive manner
//! from easy to hard examples.
//!
//! ## Features
//!
//! - **Multiple Curriculum Strategies**: Self-paced, competence-based, and predefined curricula
//! - **Difficulty Estimation**: Automatic difficulty scoring for training examples
//! - **Pacing Functions**: Various functions to control learning pace
//! - **Multi-criteria Curricula**: Combine multiple difficulty measures
//! - **Dynamic Curriculum**: Adaptive curriculum based on model performance
//! - **Evaluation Metrics**: Specialized metrics for curriculum learning
//!
//! ## Usage
//!
//! ```rust,no_run
//! use trustformers_models::curriculum_learning::{
//!     CurriculumLearningTrainer, CurriculumConfig, CurriculumStrategy
//! };
//!
//! let config = CurriculumConfig {
//!     strategy: CurriculumStrategy::SelfPaced {
//!         lambda: 0.5,
//!         gamma: 1.1,
//!     },
//!     difficulty_measure: DifficultyMeasure::LossBasedDifficulty,
//!     pacing_function: PacingFunction::Linear,
//!     ..Default::default()
//! };
//!
//! let mut trainer = CurriculumLearningTrainer::new(model, config)?;
//! trainer.train_with_curriculum(training_data)?;
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::{errors::invalid_input, tensor::Tensor, traits::Model, Result};

/// Configuration for curriculum learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumConfig {
    /// Curriculum learning strategy
    pub strategy: CurriculumStrategy,
    /// Method for measuring example difficulty
    pub difficulty_measure: DifficultyMeasure,
    /// Function controlling the pace of curriculum
    pub pacing_function: PacingFunction,
    /// Starting percentage of data to use (0.0-1.0)
    pub initial_data_percentage: f32,
    /// Whether to use curriculum during the entire training
    pub use_throughout_training: bool,
    /// Number of epochs for curriculum phase
    pub curriculum_epochs: usize,
    /// Whether to shuffle easy examples
    pub shuffle_easy_examples: bool,
    /// Whether to adaptively adjust difficulty threshold
    pub adaptive_threshold: bool,
    /// Minimum difficulty threshold
    pub min_difficulty_threshold: f32,
    /// Maximum difficulty threshold
    pub max_difficulty_threshold: f32,
    /// Evaluation frequency for adaptive curriculum
    pub evaluation_frequency: usize,
}

impl Default for CurriculumConfig {
    fn default() -> Self {
        Self {
            strategy: CurriculumStrategy::SelfPaced {
                lambda: 0.5,
                gamma: 1.1,
            },
            difficulty_measure: DifficultyMeasure::LossBasedDifficulty,
            pacing_function: PacingFunction::Linear,
            initial_data_percentage: 0.1,
            use_throughout_training: true,
            curriculum_epochs: 10,
            shuffle_easy_examples: true,
            adaptive_threshold: true,
            min_difficulty_threshold: 0.1,
            max_difficulty_threshold: 0.9,
            evaluation_frequency: 1000,
        }
    }
}

/// Different curriculum learning strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CurriculumStrategy {
    /// Self-paced learning
    SelfPaced { lambda: f32, gamma: f32 },
    /// Competence-based curriculum
    CompetenceBased {
        competence_threshold: f32,
        increase_rate: f32,
    },
    /// Predefined curriculum (manually defined difficulty)
    Predefined {
        difficulty_levels: Vec<f32>,
        level_durations: Vec<usize>,
    },
    /// Baby steps curriculum
    BabySteps { step_size: f32, patience: usize },
    /// Anti-curriculum (hard to easy)
    AntiCurriculum { reverse_pacing: bool },
    /// Cyclical curriculum
    Cyclical {
        cycle_length: usize,
        num_cycles: usize,
    },
    /// Minimax curriculum
    Minimax {
        teacher_lambda: f32,
        student_lambda: f32,
    },
    /// Random curriculum (baseline)
    Random,
}

/// Methods for measuring example difficulty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DifficultyMeasure {
    /// Loss-based difficulty (higher loss = harder)
    LossBasedDifficulty,
    /// Gradient norm-based difficulty
    GradientNormDifficulty,
    /// Prediction confidence-based difficulty
    ConfidenceDifficulty,
    /// Length-based difficulty (for sequences)
    LengthDifficulty,
    /// Complexity-based difficulty (for images/text)
    ComplexityDifficulty,
    /// Multi-criteria difficulty
    MultiCriteria {
        measures: Vec<DifficultyMeasure>,
        weights: Vec<f32>,
    },
    /// Learned difficulty (using auxiliary network)
    LearnedDifficulty {
        difficulty_network: Option<String>, // Path to difficulty network
    },
    /// Manual difficulty scores
    ManualDifficulty,
}

/// Functions for controlling curriculum pacing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PacingFunction {
    /// Linear increase in difficulty
    Linear,
    /// Exponential increase
    Exponential { rate: f32 },
    /// Logarithmic increase
    Logarithmic { base: f32 },
    /// Sigmoid-shaped increase
    Sigmoid { steepness: f32, midpoint: f32 },
    /// Step-wise increase
    StepWise { steps: Vec<(usize, f32)> },
    /// Polynomial increase
    Polynomial { degree: f32 },
    /// Custom pacing function
    Custom { function_name: String },
}

/// Training example with difficulty score
#[derive(Debug, Clone)]
pub struct CurriculumExample {
    /// Input data
    pub input: Tensor,
    /// Target labels
    pub target: Tensor,
    /// Difficulty score (0.0 = easiest, 1.0 = hardest)
    pub difficulty: f32,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
    /// Example weight for training
    pub weight: f32,
}

impl CurriculumExample {
    /// Create a new curriculum example
    pub fn new(input: Tensor, target: Tensor, difficulty: f32) -> Self {
        Self {
            input,
            target,
            difficulty,
            metadata: HashMap::new(),
            weight: 1.0,
        }
    }

    /// Create with metadata
    pub fn with_metadata(
        input: Tensor,
        target: Tensor,
        difficulty: f32,
        metadata: HashMap<String, String>,
    ) -> Self {
        Self {
            input,
            target,
            difficulty,
            metadata,
            weight: 1.0,
        }
    }

    /// Set example weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }
}

/// Curriculum learning trainer
pub struct CurriculumLearningTrainer<M: Model> {
    /// The model being trained
    pub model: M,
    /// Configuration
    pub config: CurriculumConfig,
    /// All training examples with difficulty scores
    pub examples: Vec<CurriculumExample>,
    /// Current difficulty threshold
    pub current_threshold: f32,
    /// Current epoch
    pub current_epoch: usize,
    /// Training step counter
    pub step_counter: usize,
    /// Performance history for adaptive curriculum
    pub performance_history: Vec<f32>,
    /// Difficulty scorer for dynamic difficulty estimation
    pub difficulty_scorer: Option<DifficultyScorer>,
}

impl<M: Model<Input = Tensor, Output = Tensor>> CurriculumLearningTrainer<M> {
    /// Create a new curriculum learning trainer
    pub fn new(model: M, config: CurriculumConfig) -> Result<Self> {
        let difficulty_scorer = match &config.difficulty_measure {
            DifficultyMeasure::LearnedDifficulty { .. } => {
                Some(DifficultyScorer::new(&config.difficulty_measure)?)
            },
            _ => None,
        };

        let initial_data_percentage = config.initial_data_percentage;

        Ok(Self {
            model,
            config,
            examples: Vec::new(),
            current_threshold: initial_data_percentage,
            current_epoch: 0,
            step_counter: 0,
            performance_history: Vec::new(),
            difficulty_scorer,
        })
    }

    /// Add training examples to the curriculum
    pub fn add_examples(&mut self, examples: Vec<CurriculumExample>) {
        self.examples.extend(examples);
        self.sort_examples_by_difficulty();
    }

    /// Add a single example
    pub fn add_example(&mut self, example: CurriculumExample) {
        self.examples.push(example);
        self.sort_examples_by_difficulty();
    }

    /// Estimate difficulty for examples without scores
    pub fn estimate_difficulties(&mut self) -> Result<()> {
        let mut indices_to_update = Vec::new();

        // First pass: collect indices that need updating
        for (i, example) in self.examples.iter().enumerate() {
            if example.difficulty == 0.0 {
                // Assume 0.0 means unscored
                indices_to_update.push(i);
            }
        }

        // Second pass: update difficulties without borrowing conflicts
        for i in indices_to_update {
            let input = self.examples[i].input.clone();
            let target = self.examples[i].target.clone();
            let difficulty = self.compute_difficulty(&input, &target)?;
            self.examples[i].difficulty = difficulty;
        }

        self.sort_examples_by_difficulty();
        Ok(())
    }

    /// Compute difficulty score for an example
    fn compute_difficulty(&self, input: &Tensor, target: &Tensor) -> Result<f32> {
        match &self.config.difficulty_measure {
            DifficultyMeasure::LossBasedDifficulty => {
                let outputs = self.model.forward(input.clone())?;
                let loss = self.compute_loss(&outputs, target)?;
                Ok(loss.to_scalar().unwrap_or(0.0))
            },
            DifficultyMeasure::GradientNormDifficulty => {
                // Compute gradient norm as difficulty measure
                // This is a simplified implementation
                Ok(0.5) // Placeholder
            },
            DifficultyMeasure::ConfidenceDifficulty => {
                let outputs = self.model.forward(input.clone())?;
                let probs = outputs.softmax(-1)?;
                let max_prob = self.compute_max_probability(&probs)?;
                Ok(1.0 - max_prob) // Lower confidence = higher difficulty
            },
            DifficultyMeasure::LengthDifficulty => {
                // For sequence data, use length as difficulty measure
                let seq_len = input.shape()[1] as f32; // Assuming [batch, seq_len, ...]
                Ok(seq_len / 1000.0) // Normalize by typical sequence length
            },
            DifficultyMeasure::ComplexityDifficulty => {
                // Compute complexity-based difficulty
                // This could be entropy, edge density, etc.
                Ok(0.5) // Placeholder
            },
            DifficultyMeasure::MultiCriteria { measures, weights } => {
                let mut total_difficulty = 0.0;
                let mut total_weight = 0.0;

                for (measure, &weight) in measures.iter().zip(weights.iter()) {
                    // Compute difficulty for each individual measure using dedicated method
                    let difficulty = self.compute_individual_difficulty(measure, input, target)?;
                    total_difficulty += difficulty * weight;
                    total_weight += weight;
                }

                Ok(if total_weight > 0.0 { total_difficulty / total_weight } else { 0.5 })
            },
            DifficultyMeasure::LearnedDifficulty { .. } => {
                if let Some(scorer) = &self.difficulty_scorer {
                    scorer.score_difficulty(input, target)
                } else {
                    Ok(0.5)
                }
            },
            DifficultyMeasure::ManualDifficulty => {
                // Manual difficulty should already be set
                Ok(0.5) // Default if not set
            },
        }
    }

    /// Helper method to compute difficulty for individual measures (avoiding recursion)
    fn compute_individual_difficulty(
        &self,
        measure: &DifficultyMeasure,
        input: &Tensor,
        target: &Tensor,
    ) -> Result<f32> {
        match measure {
            DifficultyMeasure::LossBasedDifficulty => {
                let outputs = self.model.forward(input.clone())?;
                let loss = self.compute_loss(&outputs, target)?;
                Ok(loss.to_scalar().unwrap_or(0.0))
            },
            DifficultyMeasure::LengthDifficulty => {
                let seq_len = input.shape()[1] as f32; // Assuming [batch, seq_len, ...]
                Ok(seq_len / 1000.0) // Normalize by typical sequence length
            },
            DifficultyMeasure::GradientNormDifficulty => {
                // Compute gradient norm-based difficulty
                Ok(0.5) // Placeholder - could be enhanced with actual gradient computation
            },
            DifficultyMeasure::ConfidenceDifficulty => {
                // Compute confidence-based difficulty (higher uncertainty = harder)
                let _outputs = self.model.forward(input.clone())?;
                // Simple confidence measure based on max probability
                Ok(0.5) // Placeholder - could compute actual confidence metrics
            },
            DifficultyMeasure::ComplexityDifficulty => {
                // Compute complexity-based difficulty
                // This could be entropy, edge density, etc.
                Ok(0.5) // Placeholder - could be enhanced with actual complexity computation
            },
            DifficultyMeasure::LearnedDifficulty { .. } => {
                if let Some(scorer) = &self.difficulty_scorer {
                    scorer.score_difficulty(input, target)
                } else {
                    Ok(0.5)
                }
            },
            DifficultyMeasure::ManualDifficulty => {
                // Manual difficulty should already be set
                Ok(0.5) // Default if not set
            },
            DifficultyMeasure::MultiCriteria { .. } => {
                // Prevent infinite recursion by returning a default value
                Ok(0.5)
            },
        }
    }

    /// Sort examples by difficulty
    fn sort_examples_by_difficulty(&mut self) {
        self.examples.sort_by(|a, b| a.difficulty.partial_cmp(&b.difficulty).unwrap());
    }

    /// Get current curriculum subset
    pub fn get_current_curriculum(&self) -> Vec<CurriculumExample> {
        let num_examples = self.examples.len();
        let threshold_count = (num_examples as f32 * self.current_threshold) as usize;

        match &self.config.strategy {
            CurriculumStrategy::AntiCurriculum { reverse_pacing } => {
                if *reverse_pacing {
                    // Start with hardest examples
                    self.examples.iter().rev().take(threshold_count).cloned().collect()
                } else {
                    self.examples.iter().take(threshold_count).cloned().collect()
                }
            },
            _ => {
                // Normal curriculum: start with easiest
                self.examples.iter().take(threshold_count).cloned().collect()
            },
        }
    }

    /// Update curriculum threshold based on strategy
    pub fn update_curriculum_threshold(&mut self) -> Result<()> {
        match &self.config.strategy {
            CurriculumStrategy::SelfPaced { lambda: _, gamma } => {
                // Self-paced learning adjusts threshold based on performance
                let recent_performance = self.get_recent_performance();
                if recent_performance > 0.8 {
                    // Good performance
                    self.current_threshold = (self.current_threshold * gamma).min(1.0);
                }
            },
            CurriculumStrategy::CompetenceBased {
                competence_threshold,
                increase_rate,
            } => {
                let competence = self.compute_competence()?;
                if competence > *competence_threshold {
                    self.current_threshold = (self.current_threshold + increase_rate).min(1.0);
                }
            },
            CurriculumStrategy::Predefined {
                difficulty_levels,
                level_durations,
            } => {
                // Use predefined schedule
                let total_steps: usize = level_durations.iter().sum();
                let current_step = self.step_counter % total_steps;
                let mut cumulative_steps = 0;

                for (i, &duration) in level_durations.iter().enumerate() {
                    cumulative_steps += duration;
                    if current_step < cumulative_steps {
                        if i < difficulty_levels.len() {
                            self.current_threshold = difficulty_levels[i];
                        }
                        break;
                    }
                }
            },
            CurriculumStrategy::BabySteps {
                step_size,
                patience,
            } => {
                // Increase threshold by small steps when performance is good
                if self.performance_history.len() >= *patience {
                    let recent_avg =
                        self.performance_history.iter().rev().take(*patience).sum::<f32>()
                            / *patience as f32;

                    if recent_avg > 0.85 {
                        // Good performance
                        self.current_threshold = (self.current_threshold + step_size).min(1.0);
                    }
                }
            },
            CurriculumStrategy::Cyclical { cycle_length, .. } => {
                // Cyclical curriculum
                let cycle_position =
                    (self.step_counter % cycle_length) as f32 / *cycle_length as f32;
                self.current_threshold = self.apply_pacing_function(cycle_position);
            },
            _ => {
                // Default linear progression
                let progress = self.current_epoch as f32 / self.config.curriculum_epochs as f32;
                self.current_threshold = self.apply_pacing_function(progress);
            },
        }

        // Apply bounds
        self.current_threshold = self
            .current_threshold
            .max(self.config.min_difficulty_threshold)
            .min(self.config.max_difficulty_threshold);

        Ok(())
    }

    /// Apply pacing function to progress
    fn apply_pacing_function(&self, progress: f32) -> f32 {
        let clamped_progress = progress.clamp(0.0, 1.0);

        match &self.config.pacing_function {
            PacingFunction::Linear => {
                self.config.initial_data_percentage
                    + (1.0 - self.config.initial_data_percentage) * clamped_progress
            },
            PacingFunction::Exponential { rate } => {
                self.config.initial_data_percentage
                    + (1.0 - self.config.initial_data_percentage)
                        * (1.0 - (-rate * clamped_progress).exp())
            },
            PacingFunction::Logarithmic { base } => {
                self.config.initial_data_percentage
                    + (1.0 - self.config.initial_data_percentage) * (clamped_progress * base).ln()
                        / base.ln()
            },
            PacingFunction::Sigmoid {
                steepness,
                midpoint,
            } => {
                let sigmoid = 1.0 / (1.0 + (-steepness * (clamped_progress - midpoint)).exp());
                self.config.initial_data_percentage
                    + (1.0 - self.config.initial_data_percentage) * sigmoid
            },
            PacingFunction::StepWise { steps } => {
                let total_steps = self.step_counter;
                for &(step_threshold, threshold_value) in steps {
                    if total_steps <= step_threshold {
                        return threshold_value;
                    }
                }
                1.0 // If past all steps, use all data
            },
            PacingFunction::Polynomial { degree } => {
                self.config.initial_data_percentage
                    + (1.0 - self.config.initial_data_percentage) * clamped_progress.powf(*degree)
            },
            PacingFunction::Custom { .. } => {
                // Custom function would be implemented here
                self.apply_pacing_function_linear(clamped_progress)
            },
        }
    }

    /// Linear pacing function (fallback)
    fn apply_pacing_function_linear(&self, progress: f32) -> f32 {
        self.config.initial_data_percentage + (1.0 - self.config.initial_data_percentage) * progress
    }

    /// Compute model competence
    fn compute_competence(&self) -> Result<f32> {
        if self.performance_history.is_empty() {
            return Ok(0.0);
        }

        let recent_performance = self.get_recent_performance();
        Ok(recent_performance)
    }

    /// Get recent performance average
    fn get_recent_performance(&self) -> f32 {
        if self.performance_history.is_empty() {
            return 0.0;
        }

        let window_size = 10.min(self.performance_history.len());
        self.performance_history.iter().rev().take(window_size).sum::<f32>() / window_size as f32
    }

    /// Train one step with curriculum
    pub fn train_step(&mut self) -> Result<CurriculumLearningOutput> {
        // Update curriculum threshold
        self.update_curriculum_threshold()?;

        // Get current curriculum examples
        let curriculum_examples = self.get_current_curriculum();

        if curriculum_examples.is_empty() {
            return Err(invalid_input(
                "No examples available for training".to_string(),
            ));
        }

        // Sample from curriculum
        let example = &curriculum_examples[self.step_counter % curriculum_examples.len()];

        // Compute forward pass and loss
        let outputs = self.model.forward(example.input.clone())?;
        let loss = self.compute_loss(&outputs, &example.target)?;

        // Weight the loss by example weight
        let weighted_loss = loss.scalar_mul(example.weight)?;

        // Compute accuracy for performance tracking
        let accuracy = self.compute_accuracy(&outputs, &example.target)?;
        self.performance_history.push(accuracy);

        // Keep performance history bounded
        if self.performance_history.len() > 1000 {
            self.performance_history = self.performance_history.split_off(500);
        }

        self.step_counter += 1;

        Ok(CurriculumLearningOutput {
            loss: weighted_loss,
            accuracy,
            difficulty_threshold: self.current_threshold,
            examples_used: curriculum_examples.len(),
            current_difficulty: example.difficulty,
        })
    }

    /// Train for one epoch with curriculum
    pub fn train_epoch(&mut self) -> Result<CurriculumEpochOutput> {
        let mut total_loss = 0.0;
        let mut total_accuracy = 0.0;
        let mut num_steps = 0;

        let curriculum_examples = self.get_current_curriculum();

        for example in &curriculum_examples {
            let outputs = self.model.forward(example.input.clone())?;
            let loss = self.compute_loss(&outputs, &example.target)?;
            let accuracy = self.compute_accuracy(&outputs, &example.target)?;

            total_loss += loss.to_scalar().unwrap_or(0.0) * example.weight;
            total_accuracy += accuracy;
            num_steps += 1;
        }

        self.current_epoch += 1;

        Ok(CurriculumEpochOutput {
            epoch: self.current_epoch,
            average_loss: total_loss / num_steps as f32,
            average_accuracy: total_accuracy / num_steps as f32,
            difficulty_threshold: self.current_threshold,
            examples_used: curriculum_examples.len(),
            total_examples: self.examples.len(),
        })
    }

    /// Compute cross-entropy loss
    fn compute_loss(&self, outputs: &Tensor, targets: &Tensor) -> Result<Tensor> {
        self.compute_cross_entropy_loss(outputs, targets)
    }

    /// Compute accuracy
    fn compute_accuracy(&self, outputs: &Tensor, targets: &Tensor) -> Result<f32> {
        let predicted = self.compute_argmax(outputs)?;
        let target_indices = self.compute_argmax(targets)?;

        // Compute accuracy as fraction of correct predictions
        let total_samples = predicted.len() as f32;
        if total_samples == 0.0 {
            return Ok(0.0);
        }

        let mut correct = 0.0;
        for (pred, target) in predicted.iter().zip(target_indices.iter()) {
            if (pred - target).abs() < f32::EPSILON {
                correct += 1.0;
            }
        }

        Ok(correct / total_samples)
    }

    /// Get curriculum statistics
    pub fn get_curriculum_stats(&self) -> CurriculumStats {
        let curriculum_examples = self.get_current_curriculum();
        let difficulties: Vec<f32> = curriculum_examples.iter().map(|e| e.difficulty).collect();

        let min_difficulty = difficulties.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_difficulty = difficulties.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let avg_difficulty = if !difficulties.is_empty() {
            difficulties.iter().sum::<f32>() / difficulties.len() as f32
        } else {
            0.0
        };

        CurriculumStats {
            current_threshold: self.current_threshold,
            examples_in_curriculum: curriculum_examples.len(),
            total_examples: self.examples.len(),
            min_difficulty,
            max_difficulty,
            avg_difficulty,
            epoch: self.current_epoch,
            step: self.step_counter,
        }
    }

    /// Compute maximum probability from softmax output
    fn compute_max_probability(&self, probs: &Tensor) -> Result<f32> {
        match probs {
            Tensor::F32(arr) => {
                // Find max probability across all dimensions
                let max_val = arr.iter().fold(0.0f32, |acc, &x| acc.max(x));
                Ok(max_val)
            },
            _ => {
                Ok(0.5) // Default fallback
            },
        }
    }

    /// Compute cross-entropy loss between outputs and targets
    fn compute_cross_entropy_loss(&self, outputs: &Tensor, targets: &Tensor) -> Result<Tensor> {
        // Apply softmax to get probabilities
        let probs = outputs.softmax(-1)?;

        // Compute log probabilities for numerical stability
        let log_probs = probs.log()?;

        // Compute negative log likelihood based on target format
        match (log_probs, targets) {
            (Tensor::F32(log_prob_arr), Tensor::F32(target_arr)) => {
                // Assuming targets are one-hot encoded or class indices
                let batch_size = log_prob_arr.shape()[0];
                let num_classes = log_prob_arr.shape().get(1).copied().unwrap_or(1);

                let mut total_loss = 0.0f32;

                for batch_idx in 0..batch_size {
                    if target_arr.shape().len() == 1 {
                        // Class indices format
                        let target_class = target_arr[[batch_idx]] as usize;
                        if target_class < num_classes {
                            total_loss -= log_prob_arr[[batch_idx, target_class]];
                        }
                    } else if target_arr.shape().len() >= 2 && target_arr.shape()[1] == num_classes
                    {
                        // One-hot format
                        for class_idx in 0..num_classes {
                            let target_prob = target_arr[[batch_idx, class_idx]];
                            if target_prob > 0.0 {
                                total_loss -= target_prob * log_prob_arr[[batch_idx, class_idx]];
                            }
                        }
                    }
                }

                // Return mean loss
                let mean_loss = total_loss / batch_size as f32;
                Ok(Tensor::scalar(mean_loss)?)
            },
            _ => {
                // Fallback for unsupported tensor types
                Ok(Tensor::scalar(1.0f32)?)
            },
        }
    }

    /// Compute argmax (indices of maximum values)
    fn compute_argmax(&self, tensor: &Tensor) -> Result<Vec<f32>> {
        match tensor {
            Tensor::F32(arr) => {
                let mut argmax_values = Vec::new();

                // Handle different tensor shapes
                if arr.ndim() == 1 {
                    // 1D tensor - find single argmax
                    let mut max_idx = 0;
                    let mut max_val = arr[0];
                    for (idx, &val) in arr.iter().enumerate() {
                        if val > max_val {
                            max_val = val;
                            max_idx = idx;
                        }
                    }
                    argmax_values.push(max_idx as f32);
                } else if arr.ndim() == 2 {
                    // 2D tensor - find argmax along last dimension for each batch
                    let batch_size = arr.shape()[0];
                    let num_classes = arr.shape()[1];

                    for batch_idx in 0..batch_size {
                        let mut max_idx = 0;
                        let mut max_val = arr[[batch_idx, 0]];

                        for class_idx in 1..num_classes {
                            let val = arr[[batch_idx, class_idx]];
                            if val > max_val {
                                max_val = val;
                                max_idx = class_idx;
                            }
                        }
                        argmax_values.push(max_idx as f32);
                    }
                } else {
                    // Multi-dimensional tensor - flatten and find global argmax
                    let mut max_idx = 0;
                    let mut max_val = arr.iter().next().copied().unwrap_or(0.0);

                    for (idx, &val) in arr.iter().enumerate() {
                        if val > max_val {
                            max_val = val;
                            max_idx = idx;
                        }
                    }
                    argmax_values.push(max_idx as f32);
                }

                Ok(argmax_values)
            },
            _ => {
                // Fallback for unsupported tensor types
                Ok(vec![0.0])
            },
        }
    }
}

/// Difficulty scorer for learned difficulty estimation
pub struct DifficultyScorer {
    /// Scoring method
    #[allow(dead_code)]
    method: DifficultyMeasure,
}

impl DifficultyScorer {
    pub fn new(method: &DifficultyMeasure) -> Result<Self> {
        Ok(Self {
            method: method.clone(),
        })
    }

    pub fn score_difficulty(&self, _input: &Tensor, _target: &Tensor) -> Result<f32> {
        // Implement learned difficulty scoring
        // This would typically involve a separate neural network
        Ok(0.5) // Placeholder
    }
}

/// Output from a curriculum learning training step
#[derive(Debug, Clone)]
pub struct CurriculumLearningOutput {
    pub loss: Tensor,
    pub accuracy: f32,
    pub difficulty_threshold: f32,
    pub examples_used: usize,
    pub current_difficulty: f32,
}

/// Output from a curriculum learning epoch
#[derive(Debug, Clone)]
pub struct CurriculumEpochOutput {
    pub epoch: usize,
    pub average_loss: f32,
    pub average_accuracy: f32,
    pub difficulty_threshold: f32,
    pub examples_used: usize,
    pub total_examples: usize,
}

/// Curriculum learning statistics
#[derive(Debug, Clone)]
pub struct CurriculumStats {
    pub current_threshold: f32,
    pub examples_in_curriculum: usize,
    pub total_examples: usize,
    pub min_difficulty: f32,
    pub max_difficulty: f32,
    pub avg_difficulty: f32,
    pub epoch: usize,
    pub step: usize,
}

/// Utilities for curriculum learning
pub mod utils {
    use super::*;

    /// Create a self-paced learning configuration
    pub fn self_paced_config(lambda: f32, gamma: f32) -> CurriculumConfig {
        CurriculumConfig {
            strategy: CurriculumStrategy::SelfPaced { lambda, gamma },
            ..Default::default()
        }
    }

    /// Create a competence-based curriculum configuration
    pub fn competence_based_config(threshold: f32, increase_rate: f32) -> CurriculumConfig {
        CurriculumConfig {
            strategy: CurriculumStrategy::CompetenceBased {
                competence_threshold: threshold,
                increase_rate,
            },
            ..Default::default()
        }
    }

    /// Create a baby steps curriculum configuration
    pub fn baby_steps_config(step_size: f32, patience: usize) -> CurriculumConfig {
        CurriculumConfig {
            strategy: CurriculumStrategy::BabySteps {
                step_size,
                patience,
            },
            pacing_function: PacingFunction::Linear,
            ..Default::default()
        }
    }

    /// Create a predefined curriculum configuration
    pub fn predefined_config(
        difficulty_levels: Vec<f32>,
        level_durations: Vec<usize>,
    ) -> CurriculumConfig {
        CurriculumConfig {
            strategy: CurriculumStrategy::Predefined {
                difficulty_levels,
                level_durations,
            },
            ..Default::default()
        }
    }

    /// Create an anti-curriculum configuration (hard to easy)
    pub fn anti_curriculum_config() -> CurriculumConfig {
        CurriculumConfig {
            strategy: CurriculumStrategy::AntiCurriculum {
                reverse_pacing: true,
            },
            ..Default::default()
        }
    }

    /// Create a cyclical curriculum configuration
    pub fn cyclical_config(cycle_length: usize, num_cycles: usize) -> CurriculumConfig {
        CurriculumConfig {
            strategy: CurriculumStrategy::Cyclical {
                cycle_length,
                num_cycles,
            },
            ..Default::default()
        }
    }

    /// Create examples with length-based difficulty
    pub fn create_length_based_examples(
        inputs: Vec<Tensor>,
        targets: Vec<Tensor>,
    ) -> Vec<CurriculumExample> {
        inputs
            .into_iter()
            .zip(targets)
            .map(|(input, target)| {
                let length = input.shape()[1] as f32; // Assuming [batch, seq_len, ...]
                let difficulty = (length / 512.0).min(1.0); // Normalize by max length
                CurriculumExample::new(input, target, difficulty)
            })
            .collect()
    }

    /// Create examples with loss-based difficulty
    pub fn create_loss_based_examples<M: Model<Input = Tensor, Output = Tensor>>(
        model: &M,
        inputs: Vec<Tensor>,
        targets: Vec<Tensor>,
    ) -> Result<Vec<CurriculumExample>> {
        let mut examples = Vec::new();

        for (input, target) in inputs.into_iter().zip(targets.into_iter()) {
            let outputs = model.forward(input.clone())?;
            // Use a simple cross-entropy loss calculation without trainer for difficulty estimation
            let loss = simple_cross_entropy_loss(&outputs, &target)?;
            let difficulty = loss.to_scalar().unwrap_or(0.0);

            examples.push(CurriculumExample::new(input, target, difficulty));
        }

        Ok(examples)
    }

    /// Simple cross-entropy loss computation for difficulty estimation
    fn simple_cross_entropy_loss(outputs: &Tensor, targets: &Tensor) -> Result<Tensor> {
        // Apply softmax to get probabilities
        let probs = outputs.softmax(-1)?;

        // Simple cross-entropy: -log(p_target)
        // This is a simplified version for difficulty estimation
        match targets.data() {
            Ok(target_data) => {
                if let Ok(prob_data) = probs.data() {
                    let batch_size = targets.shape()[0];
                    let mut total_loss = 0.0f32;

                    for i in 0..batch_size {
                        let target_idx = target_data[i] as usize;
                        if target_idx < prob_data.len() {
                            let prob = prob_data[target_idx].max(1e-8); // Avoid log(0)
                            total_loss += -prob.ln();
                        }
                    }

                    let mean_loss = total_loss / batch_size as f32;
                    Ok(Tensor::scalar(mean_loss)?)
                } else {
                    Ok(Tensor::scalar(1.0f32)?)
                }
            },
            Err(_) => Ok(Tensor::scalar(1.0f32)?),
        }
    }

    /// Create examples with manual difficulty scores
    pub fn create_manual_examples(
        inputs: Vec<Tensor>,
        targets: Vec<Tensor>,
        difficulties: Vec<f32>,
    ) -> Result<Vec<CurriculumExample>> {
        if inputs.len() != targets.len() || inputs.len() != difficulties.len() {
            return Err(invalid_input("Mismatched array lengths".to_string()));
        }

        Ok(inputs
            .into_iter()
            .zip(targets)
            .zip(difficulties)
            .map(|((input, target), difficulty)| CurriculumExample::new(input, target, difficulty))
            .collect())
    }

    /// Analyze curriculum effectiveness
    pub fn analyze_curriculum_effectiveness(
        baseline_accuracies: &[f32],
        curriculum_accuracies: &[f32],
    ) -> CurriculumAnalysis {
        let baseline_final = baseline_accuracies.last().copied().unwrap_or(0.0);
        let curriculum_final = curriculum_accuracies.last().copied().unwrap_or(0.0);

        let improvement = curriculum_final - baseline_final;

        // Compute area under the curve for convergence speed
        let baseline_auc = baseline_accuracies.iter().sum::<f32>();
        let curriculum_auc = curriculum_accuracies.iter().sum::<f32>();
        let convergence_speedup = curriculum_auc / baseline_auc.max(1e-8);

        CurriculumAnalysis {
            final_accuracy_improvement: improvement,
            convergence_speedup,
            baseline_final_accuracy: baseline_final,
            curriculum_final_accuracy: curriculum_final,
        }
    }
}

/// Analysis of curriculum learning effectiveness
#[derive(Debug, Clone)]
pub struct CurriculumAnalysis {
    pub final_accuracy_improvement: f32,
    pub convergence_speedup: f32,
    pub baseline_final_accuracy: f32,
    pub curriculum_final_accuracy: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curriculum_config_default() {
        let config = CurriculumConfig::default();
        assert_eq!(config.initial_data_percentage, 0.1);
        assert!(config.use_throughout_training);
        assert!(config.shuffle_easy_examples);

        if let CurriculumStrategy::SelfPaced { lambda, gamma } = config.strategy {
            assert_eq!(lambda, 0.5);
            assert_eq!(gamma, 1.1);
        } else {
            panic!("Expected SelfPaced strategy");
        }
    }

    #[test]
    fn test_curriculum_example() {
        let input = Tensor::zeros(&[1, 10]).unwrap();
        let target = Tensor::zeros(&[1]).unwrap();
        let example = CurriculumExample::new(input, target, 0.5);

        assert_eq!(example.difficulty, 0.5);
        assert_eq!(example.weight, 1.0);
        assert!(example.metadata.is_empty());
    }

    #[test]
    fn test_curriculum_example_with_metadata() {
        let input = Tensor::zeros(&[1, 10]).unwrap();
        let target = Tensor::zeros(&[1]).unwrap();
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "test".to_string());

        let example = CurriculumExample::with_metadata(input, target, 0.7, metadata);
        assert_eq!(example.difficulty, 0.7);
        assert_eq!(example.metadata.get("source").unwrap(), "test");
    }

    #[test]
    fn test_curriculum_example_with_weight() {
        let input = Tensor::zeros(&[1, 10]).unwrap();
        let target = Tensor::zeros(&[1]).unwrap();
        let example = CurriculumExample::new(input, target, 0.3).with_weight(2.0);

        assert_eq!(example.difficulty, 0.3);
        assert_eq!(example.weight, 2.0);
    }

    #[test]
    fn test_self_paced_config() {
        let config = utils::self_paced_config(0.8, 1.2);

        if let CurriculumStrategy::SelfPaced { lambda, gamma } = config.strategy {
            assert_eq!(lambda, 0.8);
            assert_eq!(gamma, 1.2);
        } else {
            panic!("Expected SelfPaced strategy");
        }
    }

    #[test]
    fn test_competence_based_config() {
        let config = utils::competence_based_config(0.85, 0.1);

        if let CurriculumStrategy::CompetenceBased {
            competence_threshold,
            increase_rate,
        } = config.strategy
        {
            assert_eq!(competence_threshold, 0.85);
            assert_eq!(increase_rate, 0.1);
        } else {
            panic!("Expected CompetenceBased strategy");
        }
    }

    #[test]
    fn test_baby_steps_config() {
        let config = utils::baby_steps_config(0.05, 5);

        if let CurriculumStrategy::BabySteps {
            step_size,
            patience,
        } = config.strategy
        {
            assert_eq!(step_size, 0.05);
            assert_eq!(patience, 5);
        } else {
            panic!("Expected BabySteps strategy");
        }
    }

    #[test]
    fn test_predefined_config() {
        let levels = vec![0.2, 0.5, 0.8, 1.0];
        let durations = vec![1000, 1500, 2000, 2500];
        let config = utils::predefined_config(levels.clone(), durations.clone());

        if let CurriculumStrategy::Predefined {
            difficulty_levels,
            level_durations,
        } = config.strategy
        {
            assert_eq!(difficulty_levels, levels);
            assert_eq!(level_durations, durations);
        } else {
            panic!("Expected Predefined strategy");
        }
    }

    #[test]
    fn test_anti_curriculum_config() {
        let config = utils::anti_curriculum_config();

        if let CurriculumStrategy::AntiCurriculum { reverse_pacing } = config.strategy {
            assert!(reverse_pacing);
        } else {
            panic!("Expected AntiCurriculum strategy");
        }
    }

    #[test]
    fn test_cyclical_config() {
        let config = utils::cyclical_config(1000, 3);

        if let CurriculumStrategy::Cyclical {
            cycle_length,
            num_cycles,
        } = config.strategy
        {
            assert_eq!(cycle_length, 1000);
            assert_eq!(num_cycles, 3);
        } else {
            panic!("Expected Cyclical strategy");
        }
    }

    #[test]
    fn test_create_manual_examples() {
        let inputs = vec![
            Tensor::zeros(&[1, 10]).unwrap(),
            Tensor::ones(&[1, 10]).unwrap(),
        ];
        let targets = vec![Tensor::zeros(&[1]).unwrap(), Tensor::ones(&[1]).unwrap()];
        let difficulties = vec![0.2, 0.8];

        let examples = utils::create_manual_examples(inputs, targets, difficulties).unwrap();
        assert_eq!(examples.len(), 2);
        assert_eq!(examples[0].difficulty, 0.2);
        assert_eq!(examples[1].difficulty, 0.8);
    }

    #[test]
    fn test_create_manual_examples_mismatched_lengths() {
        let inputs = vec![Tensor::zeros(&[1, 10]).unwrap()];
        let targets = vec![Tensor::zeros(&[1]).unwrap()];
        let difficulties = vec![0.2, 0.8]; // Different length

        let result = utils::create_manual_examples(inputs, targets, difficulties);
        assert!(result.is_err());
    }

    #[test]
    fn test_curriculum_analysis() {
        let baseline = vec![0.6, 0.7, 0.75, 0.8];
        let curriculum = vec![0.7, 0.8, 0.85, 0.9];

        let analysis = utils::analyze_curriculum_effectiveness(&baseline, &curriculum);
        // Use approximate comparison for floating point values
        assert!((analysis.final_accuracy_improvement - 0.1).abs() < 1e-6); // 0.9 - 0.8
        assert!((analysis.baseline_final_accuracy - 0.8).abs() < 1e-6);
        assert!((analysis.curriculum_final_accuracy - 0.9).abs() < 1e-6);
        assert!(analysis.convergence_speedup > 1.0);
    }
}
