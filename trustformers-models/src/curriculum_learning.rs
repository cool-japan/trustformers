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
//!     CurriculumLearningTrainer, CurriculumConfig, CurriculumStrategy,
//!     DifficultyMeasure, PacingFunction, CurriculumExample,
//! };
//! use trustformers_core::{traits::{Config, Model}, tensor::Tensor, Result};
//! use serde::{Deserialize, Serialize};
//!
//! # #[derive(Debug, Clone, Serialize, Deserialize)]
//! # struct DocConfig;
//! # impl Config for DocConfig {
//! #     fn architecture(&self) -> &'static str { "doc" }
//! # }
//! # struct DocModel;
//! # impl Model for DocModel {
//! #     type Config = DocConfig;
//! #     type Input = Tensor;
//! #     type Output = Tensor;
//! #     fn forward(&self, input: Tensor) -> Result<Tensor> { Ok(input) }
//! #     fn load_pretrained(&mut self, _r: &mut dyn std::io::Read) -> Result<()> { Ok(()) }
//! #     fn get_config(&self) -> &DocConfig { &DocConfig }
//! #     fn num_parameters(&self) -> usize { 0 }
//! # }
//!
//! # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
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
//! # let model = DocModel;
//! let mut trainer = CurriculumLearningTrainer::new(model, config)?;
//! # let training_data = vec![CurriculumExample::new(Tensor::zeros(&[1, 4])?, Tensor::zeros(&[1, 4])?, 0.1)];
//! trainer.add_examples(training_data);
//! trainer.train_step()?;
//! # Ok(())
//! # }
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
                loss.to_scalar().map_err(|e| {
                    invalid_input(format!("Failed to convert loss tensor to scalar: {}", e))
                })
            },
            DifficultyMeasure::GradientNormDifficulty => {
                let outputs = self.model.forward(input.clone())?;
                gradient_norm_difficulty(&outputs, target)
            },
            DifficultyMeasure::ConfidenceDifficulty => {
                let outputs = self.model.forward(input.clone())?;
                let probs = outputs.softmax(-1)?;
                let max_prob = self.compute_max_probability(&probs)?;
                Ok(1.0 - max_prob) // Lower confidence = higher difficulty
            },
            DifficultyMeasure::LengthDifficulty => sequence_length_difficulty(input),
            DifficultyMeasure::ComplexityDifficulty => input_complexity_difficulty(input),
            DifficultyMeasure::MultiCriteria { measures, weights } => {
                if measures.len() != weights.len() {
                    return Err(invalid_input(format!(
                        "MultiCriteria difficulty has {} measure(s) but {} weight(s)",
                        measures.len(),
                        weights.len()
                    )));
                }
                let mut total_difficulty = 0.0;
                let mut total_weight = 0.0;

                for (measure, &weight) in measures.iter().zip(weights.iter()) {
                    // Compute difficulty for each individual measure using dedicated method
                    let difficulty = self.compute_individual_difficulty(measure, input, target)?;
                    total_difficulty += difficulty * weight;
                    total_weight += weight;
                }

                if total_weight <= 0.0 {
                    return Err(invalid_input(
                        "MultiCriteria difficulty weights sum to zero, so no combined score can \
                         be formed"
                            .to_string(),
                    ));
                }
                Ok(total_difficulty / total_weight)
            },
            DifficultyMeasure::LearnedDifficulty { .. } => {
                let scorer = self.difficulty_scorer.as_ref().ok_or_else(|| {
                    invalid_input(
                        "LearnedDifficulty was configured but no difficulty scorer was \
                         constructed for this trainer"
                            .to_string(),
                    )
                })?;
                scorer.score_difficulty(input, target)
            },
            DifficultyMeasure::ManualDifficulty => Err(invalid_input(
                "ManualDifficulty requires every example to carry a caller-supplied difficulty \
                 score, but this example has none (its difficulty is still 0.0). Set the score \
                 with `CurriculumExample::new(input, target, difficulty)` or pick an automatic \
                 measure."
                    .to_string(),
            )),
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
                loss.to_scalar().map_err(|e| {
                    invalid_input(format!("Failed to convert loss tensor to scalar: {}", e))
                })
            },
            DifficultyMeasure::LengthDifficulty => sequence_length_difficulty(input),
            DifficultyMeasure::GradientNormDifficulty => {
                let outputs = self.model.forward(input.clone())?;
                gradient_norm_difficulty(&outputs, target)
            },
            DifficultyMeasure::ConfidenceDifficulty => {
                let outputs = self.model.forward(input.clone())?;
                let probs = outputs.softmax(-1)?;
                let max_prob = self.compute_max_probability(&probs)?;
                Ok(1.0 - max_prob)
            },
            DifficultyMeasure::ComplexityDifficulty => input_complexity_difficulty(input),
            DifficultyMeasure::LearnedDifficulty { .. } => {
                let scorer = self.difficulty_scorer.as_ref().ok_or_else(|| {
                    invalid_input(
                        "LearnedDifficulty was configured but no difficulty scorer was \
                         constructed for this trainer"
                            .to_string(),
                    )
                })?;
                scorer.score_difficulty(input, target)
            },
            DifficultyMeasure::ManualDifficulty => Err(invalid_input(
                "ManualDifficulty requires a caller-supplied difficulty score; it cannot be \
                 derived from the example"
                    .to_string(),
            )),
            DifficultyMeasure::MultiCriteria { .. } => Err(invalid_input(
                "MultiCriteria difficulty measures cannot be nested inside another MultiCriteria \
                 measure; flatten the list instead"
                    .to_string(),
            )),
        }
    }

    /// Sort examples by difficulty
    fn sort_examples_by_difficulty(&mut self) {
        self.examples.sort_by(|a, b| {
            a.difficulty.partial_cmp(&b.difficulty).unwrap_or(std::cmp::Ordering::Equal)
        });
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

            let loss_scalar = loss.to_scalar().map_err(|e| {
                invalid_input(format!("Failed to convert loss tensor to scalar: {}", e))
            })?;
            total_loss += loss_scalar * example.weight;
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
                let num_classes = log_prob_arr.shape().get(1).copied().ok_or_else(|| {
                    invalid_input(format!(
                        "Invalid tensor shape: expected at least 2 dimensions, got {}",
                        log_prob_arr.shape().len()
                    ))
                })?;

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
                    let mut max_val = arr.iter().next().copied().ok_or_else(|| {
                        invalid_input("Cannot compute argmax on empty tensor".to_string())
                    })?;

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

/// Difficulty of an example measured by the norm of the loss gradient with
/// respect to the model's output logits.
///
/// For softmax cross-entropy the gradient of the loss w.r.t. the logits has the
/// closed form `p - q`, where `p = softmax(logits)` and `q` is the target
/// distribution (a one-hot vector for a class index). This is the *real*
/// gradient, not an approximation of it: the analytic form is exact, so no
/// autodiff pass is needed to obtain it. Its L2 norm is large when the model
/// puts its mass on the wrong class and small when it already predicts the
/// target confidently — precisely the "hard example" signal the curriculum
/// wants.
///
/// The norm is averaged over the batch and squashed into `[0, 1)` with
/// `n / (1 + n)` so that it composes with the other measures, which are also
/// unit-scaled.
///
/// A previous revision returned the constant `0.5` for this measure, which made
/// the resulting curriculum ordering meaningless: every example tied.
///
/// # Errors
///
/// Fails when the tensors cannot be read as `f32`, when the output has no class
/// axis, or when the targets match neither the class-index nor the one-hot
/// layout.
pub fn gradient_norm_difficulty(outputs: &Tensor, target: &Tensor) -> Result<f32> {
    let probs = outputs.softmax(-1)?;
    let prob_shape = probs.shape();
    let num_classes = *prob_shape.last().ok_or_else(|| {
        invalid_input("model outputs are a scalar; no class axis to differentiate".to_string())
    })?;
    if num_classes == 0 {
        return Err(invalid_input(format!(
            "model outputs with shape {prob_shape:?} carry no class scores"
        )));
    }
    let batch_size: usize = prob_shape[..prob_shape.len() - 1].iter().product();
    if batch_size == 0 {
        return Err(invalid_input(format!(
            "model outputs with shape {prob_shape:?} carry no examples"
        )));
    }

    let prob_data = probs
        .data()
        .map_err(|e| invalid_input(format!("failed to read the model's probabilities: {e}")))?;
    let target_data = target
        .data()
        .map_err(|e| invalid_input(format!("failed to read targets: {e}")))?;

    let mut total_norm = 0.0f32;
    for b in 0..batch_size {
        let row = &prob_data[b * num_classes..(b + 1) * num_classes];
        let mut squared = 0.0f32;
        if target_data.len() == batch_size {
            let raw = target_data[b];
            if raw < 0.0 || raw.fract() != 0.0 {
                return Err(invalid_input(format!(
                    "target {raw} at batch position {b} is not a non-negative integer class index"
                )));
            }
            let class = raw as usize;
            if class >= num_classes {
                return Err(invalid_input(format!(
                    "target class {class} at batch position {b} is out of range for \
                     {num_classes} classes"
                )));
            }
            for (c, &p) in row.iter().enumerate() {
                let g = if c == class { p - 1.0 } else { p };
                squared += g * g;
            }
        } else if target_data.len() == batch_size * num_classes {
            let target_row = &target_data[b * num_classes..(b + 1) * num_classes];
            for (&p, &q) in row.iter().zip(target_row.iter()) {
                let g = p - q;
                squared += g * g;
            }
        } else {
            return Err(invalid_input(format!(
                "targets with {} element(s) match neither class indices ([{batch_size}]) nor \
                 one-hot labels ([{batch_size}, {num_classes}]) for outputs with shape \
                 {prob_shape:?}",
                target_data.len()
            )));
        }
        total_norm += squared.sqrt();
    }

    let mean_norm = total_norm / batch_size as f32;
    Ok(mean_norm / (1.0 + mean_norm))
}

/// Difficulty of an example measured by the length of its sequence axis.
///
/// Longer sequences are harder, and the score is squashed into `[0, 1)` with
/// `len / (1 + len)` after normalising by a 512-token reference length so that
/// the measure never saturates at exactly 1 and stays comparable with the other
/// unit-scaled measures.
///
/// # Errors
///
/// Fails when the input has no sequence axis (fewer than two dimensions).
pub fn sequence_length_difficulty(input: &Tensor) -> Result<f32> {
    let shape = input.shape();
    let seq_len = match shape.len() {
        0 => {
            return Err(invalid_input(
                "LengthDifficulty needs a sequence axis, but the input is a scalar".to_string(),
            ))
        },
        1 => shape[0],
        // [batch, seq_len, ...]
        _ => shape[1],
    };
    let normalised = seq_len as f32 / 512.0;
    Ok(normalised / (1.0 + normalised))
}

/// Difficulty of an example measured by the Shannon entropy of its own value
/// distribution.
///
/// A featureless input (a constant image patch, a padded sequence) concentrates
/// all of its mass in one histogram bin and scores near 0; an input whose values
/// spread evenly across the observed range scores near 1. The histogram is built
/// over the input's *own* min/max range with `sqrt(n)` bins (capped at 64), so
/// the measure is scale-invariant, and the entropy is divided by `log2(bins)` to
/// land in `[0, 1]`.
///
/// A previous revision returned the constant `0.5`, so an all-zero tensor and a
/// richly-structured one were declared equally complex.
///
/// # Errors
///
/// Fails when the input cannot be read as `f32` or holds no elements.
pub fn input_complexity_difficulty(input: &Tensor) -> Result<f32> {
    let values = input
        .data()
        .map_err(|e| invalid_input(format!("failed to read the example's values: {e}")))?;
    if values.is_empty() {
        return Err(invalid_input(
            "ComplexityDifficulty cannot score an empty input tensor".to_string(),
        ));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid_input(
            "ComplexityDifficulty cannot score an input holding NaN or infinite values".to_string(),
        ));
    }

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in &values {
        min = min.min(v);
        max = max.max(v);
    }
    let range = max - min;
    if range <= f32::EPSILON {
        // Every value identical: zero entropy, the least complex input there is.
        return Ok(0.0);
    }

    let bins = ((values.len() as f32).sqrt().round() as usize).clamp(2, 64);
    let mut histogram = vec![0usize; bins];
    for &v in &values {
        let position = ((v - min) / range * bins as f32) as usize;
        histogram[position.min(bins - 1)] += 1;
    }

    let total = values.len() as f32;
    let mut entropy = 0.0f32;
    for &count in &histogram {
        if count > 0 {
            let p = count as f32 / total;
            entropy -= p * p.log2();
        }
    }
    Ok((entropy / (bins as f32).log2()).clamp(0.0, 1.0))
}

/// Difficulty scorer for learned difficulty estimation.
///
/// A learned scorer is an *auxiliary network* trained to predict per-example
/// difficulty. This crate ships no such network and no format for one, so a
/// scorer configured with
/// [`DifficultyMeasure::LearnedDifficulty`] reports that honestly rather than
/// returning an invented score — a previous revision returned the constant
/// `0.5`, which silently degraded a "learned" curriculum to no curriculum at
/// all.
pub struct DifficultyScorer {
    /// Scoring method
    method: DifficultyMeasure,
}

impl DifficultyScorer {
    /// Build a scorer for the given measure.
    ///
    /// # Errors
    ///
    /// Never fails; the `Result` is kept for API compatibility.
    pub fn new(method: &DifficultyMeasure) -> Result<Self> {
        Ok(Self {
            method: method.clone(),
        })
    }

    /// The measure this scorer was built for.
    pub fn method(&self) -> &DifficultyMeasure {
        &self.method
    }

    /// Score an example.
    ///
    /// # Errors
    ///
    /// Returns an error for [`DifficultyMeasure::LearnedDifficulty`] when no
    /// auxiliary difficulty network is available — which is always, since no
    /// loader for one exists. Other measures are delegated to the corresponding
    /// analytic scorer.
    pub fn score_difficulty(&self, input: &Tensor, _target: &Tensor) -> Result<f32> {
        match &self.method {
            DifficultyMeasure::LearnedDifficulty { difficulty_network } => {
                Err(invalid_input(format!(
                    "LearnedDifficulty requires a trained auxiliary difficulty network, and none \
                     can be loaded: no loader for such a network exists in this crate (configured \
                     path: {}). Use LossBasedDifficulty, ConfidenceDifficulty, \
                     GradientNormDifficulty, LengthDifficulty or ComplexityDifficulty instead.",
                    difficulty_network.as_deref().unwrap_or("<none configured>")
                )))
            },
            DifficultyMeasure::LengthDifficulty => sequence_length_difficulty(input),
            DifficultyMeasure::ComplexityDifficulty => input_complexity_difficulty(input),
            other => Err(invalid_input(format!(
                "DifficultyScorer cannot score {other:?} on its own; it needs the trainer's model"
            ))),
        }
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

        for (input, target) in inputs.into_iter().zip(targets) {
            let outputs = model.forward(input.clone())?;
            // Use a simple cross-entropy loss calculation without trainer for difficulty estimation
            let loss = simple_cross_entropy_loss(&outputs, &target)?;
            let difficulty = loss.to_scalar().map_err(|e| {
                invalid_input(format!(
                    "Failed to convert loss tensor to scalar for difficulty estimation: {}",
                    e
                ))
            })?;

            examples.push(CurriculumExample::new(input, target, difficulty));
        }

        Ok(examples)
    }

    /// Cross-entropy loss used to score example difficulty.
    ///
    /// `outputs` has shape `[batch, num_classes]`; `targets` is either class
    /// indices (`[batch]`) or one-hot / soft labels (`[batch, num_classes]`).
    ///
    /// The previous implementation indexed the *flat* probability buffer with the
    /// bare class index — `prob_data[target_idx]` — with no `batch * num_classes`
    /// stride. Every sample after the first therefore read a probability
    /// belonging to row 0, so the returned difficulty was wrong for the whole
    /// batch tail and identical for any two batches that shared a first row. It
    /// also swallowed every error into a hardcoded `1.0`, which is exactly the
    /// kind of fabricated metric a curriculum then sorts on.
    ///
    /// # Errors
    ///
    /// Fails when the tensors cannot be read as `f32`, when the shapes do not
    /// describe a `[batch, num_classes]` scoring problem, or when a class index
    /// is out of range.
    pub fn simple_cross_entropy_loss(outputs: &Tensor, targets: &Tensor) -> Result<Tensor> {
        // Apply softmax to get probabilities
        let probs = outputs.softmax(-1)?;
        let prob_shape = probs.shape();
        let num_classes = *prob_shape.last().ok_or_else(|| {
            invalid_input(
                "model outputs are a scalar; cross-entropy needs a class axis".to_string(),
            )
        })?;
        if num_classes == 0 {
            return Err(invalid_input(format!(
                "model outputs with shape {prob_shape:?} carry no class scores"
            )));
        }
        let batch_size: usize = prob_shape[..prob_shape.len() - 1].iter().product();
        if batch_size == 0 {
            return Err(invalid_input(format!(
                "model outputs with shape {prob_shape:?} carry no examples"
            )));
        }

        let prob_data = probs
            .data()
            .map_err(|e| invalid_input(format!("failed to read the model's probabilities: {e}")))?;
        let target_data = targets
            .data()
            .map_err(|e| invalid_input(format!("failed to read targets: {e}")))?;

        let mut total_loss = 0.0f32;
        if target_data.len() == batch_size {
            // Class indices: -log p[b, target[b]], with the batch stride applied.
            for b in 0..batch_size {
                let raw = target_data[b];
                if raw < 0.0 || raw.fract() != 0.0 {
                    return Err(invalid_input(format!(
                        "target {raw} at batch position {b} is not a non-negative integer class \
                         index"
                    )));
                }
                let class = raw as usize;
                if class >= num_classes {
                    return Err(invalid_input(format!(
                        "target class {class} at batch position {b} is out of range for \
                         {num_classes} classes"
                    )));
                }
                let prob = prob_data[b * num_classes + class].max(1e-8);
                total_loss -= prob.ln();
            }
        } else if target_data.len() == batch_size * num_classes {
            // One-hot / soft labels: -sum_c q[b, c] * log p[b, c].
            for b in 0..batch_size {
                for c in 0..num_classes {
                    let weight = target_data[b * num_classes + c];
                    if weight != 0.0 {
                        total_loss -= weight * prob_data[b * num_classes + c].max(1e-8).ln();
                    }
                }
            }
        } else {
            return Err(invalid_input(format!(
                "targets with {} element(s) match neither class indices ([{batch_size}]) nor \
                 one-hot labels ([{batch_size}, {num_classes}]) for outputs with shape \
                 {prob_shape:?}",
                target_data.len()
            )));
        }

        Tensor::scalar(total_loss / batch_size as f32)
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
        // Use 0.0 as default for empty accuracy arrays (no training = no accuracy).
        // Libraries must not write to stdout/stderr, so the degenerate case is
        // reported through the `tracing` facade instead of `eprintln!`.
        let baseline_final = baseline_accuracies.last().copied().unwrap_or_else(|| {
            tracing::warn!("empty baseline accuracy history; reporting 0.0 final accuracy");
            0.0
        });
        let curriculum_final = curriculum_accuracies.last().copied().unwrap_or_else(|| {
            tracing::warn!("empty curriculum accuracy history; reporting 0.0 final accuracy");
            0.0
        });

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

    /// Regression: `simple_cross_entropy_loss` indexed the flat probability
    /// buffer with the bare class index and no batch stride, so every sample
    /// after the first read row 0's probabilities.
    ///
    /// Row 0 is confident about class 0 and row 1 about class 2. With the stride
    /// bug, the loss for row 1's label (class 2) was read from row 0 — where
    /// class 2 is unlikely — so the reported difficulty was large. The correct
    /// value is small, because row 1 does predict class 2.
    #[test]
    fn cross_entropy_applies_the_batch_stride() {
        let outputs = Tensor::from_vec(vec![5.0, 0.0, 0.0, 0.0, 0.0, 5.0], &[2, 3])
            .expect("outputs must build");
        let targets = Tensor::from_vec(vec![0.0, 2.0], &[2]).expect("targets must build");

        let loss = utils::simple_cross_entropy_loss(&outputs, &targets)
            .expect("cross-entropy must succeed")
            .to_scalar()
            .expect("scalar");
        // Both rows predict their own label confidently, so the mean loss is the
        // single-row value log(1 + 2 e^-5) ~= 0.01342.
        assert!(
            (loss - 0.013_42).abs() < 1e-3,
            "both rows predict their own label; loss must be small, got {loss}"
        );

        // Swapping the labels must cost much more — impossible if row 1 is read
        // from row 0's slice.
        let swapped = Tensor::from_vec(vec![2.0, 0.0], &[2]).expect("targets must build");
        let swapped_loss = utils::simple_cross_entropy_loss(&outputs, &swapped)
            .expect("cross-entropy must succeed")
            .to_scalar()
            .expect("scalar");
        assert!(
            swapped_loss > loss + 1.0,
            "swapped labels must be far more costly: {swapped_loss} vs {loss}"
        );
    }

    /// Regression: the gradient-norm measure returned a constant `0.5`, so a
    /// confidently-correct example and a confidently-wrong one tied.
    #[test]
    fn gradient_norm_difficulty_separates_easy_from_hard_examples() {
        let outputs = Tensor::from_vec(vec![5.0, 0.0, 0.0], &[1, 3]).expect("outputs must build");
        let easy_target = Tensor::from_vec(vec![0.0], &[1]).expect("target must build");
        let hard_target = Tensor::from_vec(vec![2.0], &[1]).expect("target must build");

        let easy = gradient_norm_difficulty(&outputs, &easy_target).expect("scoring must succeed");
        let hard = gradient_norm_difficulty(&outputs, &hard_target).expect("scoring must succeed");

        assert!(
            hard > easy,
            "a wrong confident prediction must score harder: easy={easy}, hard={hard}"
        );
        assert!(
            (0.0..=1.0).contains(&easy) && (0.0..=1.0).contains(&hard),
            "difficulties must be unit-scaled: {easy}, {hard}"
        );
        assert_ne!(easy, 0.5, "the measure must not be the old constant");
    }

    /// Regression: the complexity measure returned a constant `0.5`, so a
    /// featureless input and a varied one were declared equally complex.
    #[test]
    fn complexity_difficulty_separates_flat_from_varied_inputs() {
        let flat = Tensor::zeros(&[64]).expect("flat tensor must build");
        let varied = Tensor::from_vec((0..64).map(|i| i as f32).collect(), &[64])
            .expect("varied tensor must build");

        let flat_score = input_complexity_difficulty(&flat).expect("scoring must succeed");
        let varied_score = input_complexity_difficulty(&varied).expect("scoring must succeed");

        assert!(
            flat_score < 1e-6,
            "a constant input has no entropy, got {flat_score}"
        );
        assert!(
            varied_score > 0.9,
            "a uniformly-spread input is near-maximally complex, got {varied_score}"
        );
    }

    /// Regression: the length measure divided by a hardcoded 1000 and could
    /// exceed 1; and every unscored measure fell through to 0.5.
    #[test]
    fn length_difficulty_is_monotonic_and_bounded() {
        let short = Tensor::zeros(&[1, 8, 4]).expect("tensor must build");
        let long = Tensor::zeros(&[1, 4096, 4]).expect("tensor must build");
        let short_score = sequence_length_difficulty(&short).expect("scoring must succeed");
        let long_score = sequence_length_difficulty(&long).expect("scoring must succeed");
        assert!(
            long_score > short_score,
            "longer sequences must be harder: {short_score} vs {long_score}"
        );
        assert!(
            long_score < 1.0,
            "the measure must stay below 1, got {long_score}"
        );
    }

    /// Regression: a `LearnedDifficulty` scorer returned the constant `0.5`
    /// while claiming to be a learned score.
    #[test]
    fn learned_difficulty_reports_the_absent_network_instead_of_inventing_a_score() {
        let scorer = DifficultyScorer::new(&DifficultyMeasure::LearnedDifficulty {
            difficulty_network: Some("scorer.safetensors".to_string()),
        })
        .expect("scorer must build");
        let input = Tensor::zeros(&[1, 4]).expect("tensor must build");
        let target = Tensor::zeros(&[1]).expect("tensor must build");
        let err = scorer
            .score_difficulty(&input, &target)
            .expect_err("no learned network exists, so no score may be produced");
        assert!(
            err.to_string().contains("auxiliary difficulty network"),
            "unexpected error: {err}"
        );
    }

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
        let input = Tensor::zeros(&[1, 10]).expect("operation failed");
        let target = Tensor::zeros(&[1]).expect("operation failed");
        let example = CurriculumExample::new(input, target, 0.5);

        assert_eq!(example.difficulty, 0.5);
        assert_eq!(example.weight, 1.0);
        assert!(example.metadata.is_empty());
    }

    #[test]
    fn test_curriculum_example_with_metadata() {
        let input = Tensor::zeros(&[1, 10]).expect("operation failed");
        let target = Tensor::zeros(&[1]).expect("operation failed");
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "test".to_string());

        let example = CurriculumExample::with_metadata(input, target, 0.7, metadata);
        assert_eq!(example.difficulty, 0.7);
        assert_eq!(
            example.metadata.get("source").expect("operation failed"),
            "test"
        );
    }

    #[test]
    fn test_curriculum_example_with_weight() {
        let input = Tensor::zeros(&[1, 10]).expect("operation failed");
        let target = Tensor::zeros(&[1]).expect("operation failed");
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
            Tensor::zeros(&[1, 10]).expect("operation failed"),
            Tensor::ones(&[1, 10]).expect("operation failed"),
        ];
        let targets = vec![
            Tensor::zeros(&[1]).expect("operation failed"),
            Tensor::ones(&[1]).expect("operation failed"),
        ];
        let difficulties = vec![0.2, 0.8];

        let examples =
            utils::create_manual_examples(inputs, targets, difficulties).expect("operation failed");
        assert_eq!(examples.len(), 2);
        assert_eq!(examples[0].difficulty, 0.2);
        assert_eq!(examples[1].difficulty, 0.8);
    }

    #[test]
    fn test_create_manual_examples_mismatched_lengths() {
        let inputs = vec![Tensor::zeros(&[1, 10]).expect("operation failed")];
        let targets = vec![Tensor::zeros(&[1]).expect("operation failed")];
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
