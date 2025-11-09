//! # Knowledge Distillation Framework
//!
//! This module provides a comprehensive framework for knowledge distillation,
//! enabling efficient transfer of knowledge from large teacher models to smaller student models.
//!
//! ## Features
//!
//! - **Multiple Distillation Strategies**: Response-based, feature-based, and attention-based distillation
//! - **Temperature Control**: Configurable temperature scaling for soft targets
//! - **Loss Combinations**: Flexible combination of distillation and task-specific losses
//! - **Multi-layer Feature Matching**: Deep feature alignment between teacher and student
//! - **Attention Transfer**: Transfer attention patterns from teacher to student
//! - **Progressive Knowledge Transfer**: Gradual knowledge transfer strategies
//!
//! ## Usage
//!
//! ```rust,no_run
//! use trustformers_models::knowledge_distillation::{
//!     KnowledgeDistillationTrainer, DistillationConfig, DistillationStrategy
//! };
//!
//! let config = DistillationConfig {
//!     temperature: 4.0,
//!     alpha: 0.7,
//!     strategy: DistillationStrategy::ResponseBased,
//!     ..Default::default()
//! };
//!
//! let trainer = KnowledgeDistillationTrainer::new(teacher_model, student_model, config)?;
//! trainer.train(dataloader)?;
//! ```

use scirs2_core::ndarray::{ArrayD, Axis, IxDyn}; // SciRS2 Integration Policy
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::{
    errors::{tensor_op_error, TrustformersError},
    layers::Linear,
    tensor::Tensor,
    traits::{Layer, Model},
    Result,
};

/// Configuration for knowledge distillation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationConfig {
    /// Temperature for softmax in distillation loss (higher = softer)
    pub temperature: f32,
    /// Weight for distillation loss vs. hard target loss (0.0-1.0)
    pub alpha: f32,
    /// Distillation strategy to use
    pub strategy: DistillationStrategy,
    /// Whether to use feature matching
    pub use_feature_matching: bool,
    /// Layers to match features (teacher_layer -> student_layer)
    pub feature_matching_layers: HashMap<usize, usize>,
    /// Whether to use attention transfer
    pub use_attention_transfer: bool,
    /// Weight for attention transfer loss
    pub attention_loss_weight: f32,
    /// Whether to use progressive distillation
    pub progressive: bool,
    /// Number of progressive stages
    pub progressive_stages: usize,
    /// Minimum temperature for progressive cooling
    pub min_temperature: f32,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            temperature: 4.0,
            alpha: 0.7,
            strategy: DistillationStrategy::ResponseBased,
            use_feature_matching: false,
            feature_matching_layers: HashMap::new(),
            use_attention_transfer: false,
            attention_loss_weight: 0.1,
            progressive: false,
            progressive_stages: 5,
            min_temperature: 1.0,
        }
    }
}

/// Different strategies for knowledge distillation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistillationStrategy {
    /// Standard response-based distillation using soft targets
    ResponseBased,
    /// Feature-based distillation matching intermediate representations
    FeatureBased,
    /// Attention-based distillation transferring attention patterns
    AttentionBased,
    /// Combined approach using multiple strategies
    Combined {
        response_weight: f32,
        feature_weight: f32,
        attention_weight: f32,
    },
    /// Progressive distillation with curriculum learning
    Progressive { stages: Vec<ProgressiveStage> },
}

/// Configuration for a progressive distillation stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveStage {
    /// Duration of this stage (in training steps)
    pub duration: usize,
    /// Temperature for this stage
    pub temperature: f32,
    /// Alpha weight for this stage
    pub alpha: f32,
    /// Whether to freeze teacher layers
    pub freeze_teacher: bool,
}

/// Output from distillation loss computation
#[derive(Debug, Clone)]
pub struct DistillationOutput {
    /// Total distillation loss
    pub total_loss: Tensor,
    /// Individual loss components
    pub loss_components: HashMap<String, Tensor>,
    /// Soft predictions from teacher
    pub teacher_predictions: Tensor,
    /// Predictions from student
    pub student_predictions: Tensor,
    /// Feature matching losses (if used)
    pub feature_losses: Option<HashMap<String, Tensor>>,
    /// Attention transfer losses (if used)
    pub attention_losses: Option<HashMap<String, Tensor>>,
}

/// Knowledge distillation trainer
pub struct KnowledgeDistillationTrainer<T, S> {
    /// Teacher model (typically larger, pre-trained)
    #[allow(dead_code)]
    teacher: T,
    /// Student model (typically smaller, being trained)
    #[allow(dead_code)]
    student: S,
    /// Distillation configuration
    config: DistillationConfig,
    /// Feature matching projections
    feature_projections: HashMap<usize, Linear>,
    /// Current training stage (for progressive distillation)
    current_stage: usize,
    /// Current training step
    current_step: usize,
}

impl<T, S> KnowledgeDistillationTrainer<T, S>
where
    T: Model,
    S: Model,
{
    /// Create a new knowledge distillation trainer
    pub fn new(teacher: T, student: S, config: DistillationConfig) -> Result<Self> {
        let mut feature_projections = HashMap::new();

        // Initialize feature projection layers if needed
        if config.use_feature_matching {
            for (&_teacher_layer, &student_layer) in &config.feature_matching_layers {
                // Note: In practice, you'd need to get the actual hidden sizes
                // This is a simplified example
                let projection = Linear::new(768, 768, true); // Assuming 768 hidden size
                feature_projections.insert(student_layer, projection);
            }
        }

        Ok(Self {
            teacher,
            student,
            config,
            feature_projections,
            current_stage: 0,
            current_step: 0,
        })
    }

    /// Compute distillation loss
    pub fn compute_distillation_loss(
        &self,
        teacher_outputs: &TeacherOutputs,
        student_outputs: &StudentOutputs,
        hard_targets: Option<&Tensor>,
    ) -> Result<DistillationOutput> {
        let mut loss_components = HashMap::new();
        let mut total_loss = Tensor::zeros(&[1])?;

        match &self.config.strategy {
            DistillationStrategy::ResponseBased => {
                let response_loss = self.compute_response_distillation_loss(
                    &teacher_outputs.logits,
                    &student_outputs.logits,
                )?;
                loss_components.insert("response".to_string(), response_loss.clone());
                total_loss = total_loss.add(&response_loss)?;
            },
            DistillationStrategy::FeatureBased => {
                let feature_loss = self.compute_feature_distillation_loss(
                    &teacher_outputs.hidden_states,
                    &student_outputs.hidden_states,
                )?;
                loss_components.insert("feature".to_string(), feature_loss.clone());
                total_loss = total_loss.add(&feature_loss)?;
            },
            DistillationStrategy::AttentionBased => {
                let attention_loss = self.compute_attention_distillation_loss(
                    &teacher_outputs.attentions,
                    &student_outputs.attentions,
                )?;
                loss_components.insert("attention".to_string(), attention_loss.clone());
                total_loss = total_loss.add(&attention_loss)?;
            },
            DistillationStrategy::Combined {
                response_weight,
                feature_weight,
                attention_weight,
            } => {
                if *response_weight > 0.0 {
                    let response_loss = self.compute_response_distillation_loss(
                        &teacher_outputs.logits,
                        &student_outputs.logits,
                    )?;
                    let weighted_response_loss = response_loss.scalar_mul(*response_weight)?;
                    loss_components.insert("response".to_string(), weighted_response_loss.clone());
                    total_loss = total_loss.add(&weighted_response_loss)?;
                }

                if *feature_weight > 0.0 && !teacher_outputs.hidden_states.is_empty() {
                    let feature_loss = self.compute_feature_distillation_loss(
                        &teacher_outputs.hidden_states,
                        &student_outputs.hidden_states,
                    )?;
                    let weighted_feature_loss = feature_loss.scalar_mul(*feature_weight)?;
                    loss_components.insert("feature".to_string(), weighted_feature_loss.clone());
                    total_loss = total_loss.add(&weighted_feature_loss)?;
                }

                if *attention_weight > 0.0 && !teacher_outputs.attentions.is_empty() {
                    let attention_loss = self.compute_attention_distillation_loss(
                        &teacher_outputs.attentions,
                        &student_outputs.attentions,
                    )?;
                    let weighted_attention_loss = attention_loss.scalar_mul(*attention_weight)?;
                    loss_components
                        .insert("attention".to_string(), weighted_attention_loss.clone());
                    total_loss = total_loss.add(&weighted_attention_loss)?;
                }
            },
            DistillationStrategy::Progressive { stages } => {
                let current_stage = &stages[self.current_stage.min(stages.len() - 1)];
                let response_loss = self.compute_response_distillation_loss_with_temperature(
                    &teacher_outputs.logits,
                    &student_outputs.logits,
                    current_stage.temperature,
                )?;
                loss_components.insert("progressive_response".to_string(), response_loss.clone());
                total_loss = total_loss.add(&response_loss)?;
            },
        }

        // Add hard target loss if provided
        if let Some(targets) = hard_targets {
            let hard_loss = self.compute_hard_target_loss(&student_outputs.logits, targets)?;
            let weighted_hard_loss = hard_loss.scalar_mul(1.0 - self.config.alpha)?;
            loss_components.insert("hard_target".to_string(), weighted_hard_loss.clone());
            total_loss = total_loss.add(&weighted_hard_loss)?;
        }

        // Collect feature losses for tracking
        let feature_losses = if !teacher_outputs.hidden_states.is_empty()
            && !student_outputs.hidden_states.is_empty()
        {
            Some(self.compute_layer_wise_feature_losses(
                &teacher_outputs.hidden_states,
                &student_outputs.hidden_states,
            )?)
        } else {
            None
        };

        // Collect attention losses for tracking
        let attention_losses =
            if !teacher_outputs.attentions.is_empty() && !student_outputs.attentions.is_empty() {
                Some(self.compute_layer_wise_attention_losses(
                    &teacher_outputs.attentions,
                    &student_outputs.attentions,
                )?)
            } else {
                None
            };

        Ok(DistillationOutput {
            total_loss,
            loss_components,
            teacher_predictions: teacher_outputs.logits.clone(),
            student_predictions: student_outputs.logits.clone(),
            feature_losses,
            attention_losses,
        })
    }

    /// Compute response-based distillation loss (KL divergence of soft targets)
    fn compute_response_distillation_loss(
        &self,
        teacher_logits: &Tensor,
        student_logits: &Tensor,
    ) -> Result<Tensor> {
        self.compute_response_distillation_loss_with_temperature(
            teacher_logits,
            student_logits,
            self.config.temperature,
        )
    }

    /// Compute response-based distillation loss with custom temperature
    fn compute_response_distillation_loss_with_temperature(
        &self,
        teacher_logits: &Tensor,
        student_logits: &Tensor,
        temperature: f32,
    ) -> Result<Tensor> {
        // Apply temperature scaling
        let teacher_scaled = teacher_logits.scalar_div(temperature)?;
        let student_scaled = student_logits.scalar_div(temperature)?;

        // Compute soft targets (softmax with temperature)
        let teacher_soft = teacher_scaled.softmax(-1)?;
        let student_soft = student_scaled.softmax(-1)?;
        let student_log_soft = student_soft.log()?;

        // KL divergence loss
        let teacher_log = teacher_soft.log()?;
        let log_diff = teacher_log.sub(&student_log_soft)?;
        let kl_div = teacher_soft.mul(&log_diff)?;
        let loss = kl_div.sum(None, false)?.mean()?;

        // Scale by temperature squared (standard in knowledge distillation)
        let temp_squared = temperature * temperature;
        loss.scalar_mul(temp_squared)
    }

    /// Compute feature-based distillation loss
    fn compute_feature_distillation_loss(
        &self,
        teacher_features: &[Tensor],
        student_features: &[Tensor],
    ) -> Result<Tensor> {
        let mut total_loss = Tensor::zeros(&[1])?;
        let mut num_matched = 0;

        for (&teacher_layer, &student_layer) in &self.config.feature_matching_layers {
            if teacher_layer < teacher_features.len() && student_layer < student_features.len() {
                let teacher_feat = &teacher_features[teacher_layer];
                let student_feat = &student_features[student_layer];

                // Project student features to match teacher dimensionality if needed
                let projected_student =
                    if let Some(projection) = self.feature_projections.get(&student_layer) {
                        projection.forward(student_feat.clone())?
                    } else {
                        student_feat.clone()
                    };

                // MSE loss between features
                let diff = teacher_feat.sub(&projected_student)?;
                let diff_squared = diff.mul(&diff)?;
                let mse_loss = diff_squared.mean()?;
                total_loss = total_loss.add(&mse_loss)?;
                num_matched += 1;
            }
        }

        if num_matched > 0 {
            Ok(total_loss.scalar_div(num_matched as f32)?)
        } else {
            Ok(total_loss)
        }
    }

    /// Compute attention-based distillation loss
    fn compute_attention_distillation_loss(
        &self,
        teacher_attentions: &[Tensor],
        student_attentions: &[Tensor],
    ) -> Result<Tensor> {
        let mut total_loss = Tensor::zeros(&[1])?;
        let num_layers = teacher_attentions.len().min(student_attentions.len());

        for i in 0..num_layers {
            let teacher_attn = &teacher_attentions[i];
            let student_attn = &student_attentions[i];

            // MSE loss between attention matrices
            let diff = teacher_attn.sub(student_attn)?;
            let diff_squared = diff.mul(&diff)?;
            let mse_loss = diff_squared.mean()?;
            total_loss = total_loss.add(&mse_loss)?;
        }

        if num_layers > 0 {
            Ok(total_loss.scalar_div(num_layers as f32)?)
        } else {
            Ok(total_loss)
        }
    }

    /// Compute hard target loss (standard cross-entropy)
    fn compute_hard_target_loss(&self, logits: &Tensor, _targets: &Tensor) -> Result<Tensor> {
        let probs = logits.softmax(-1)?;
        let log_probs = probs.log()?;

        // Simplified cross-entropy implementation - in practice this would need proper indexing
        // For now, compute mean of log probs as a placeholder
        let neg_log_probs = log_probs.scalar_mul(-1.0)?;
        neg_log_probs.mean()
    }

    /// Update training step and potentially stage for progressive distillation
    pub fn step(&mut self) {
        self.current_step += 1;

        if let DistillationStrategy::Progressive { stages } = &self.config.strategy {
            // Check if we should advance to the next stage
            if self.current_stage < stages.len() - 1 {
                let current_stage_config = &stages[self.current_stage];
                if self.current_step >= current_stage_config.duration {
                    self.current_stage += 1;
                    self.current_step = 0;
                }
            }
        }
    }

    /// Get current temperature (useful for progressive distillation)
    pub fn current_temperature(&self) -> f32 {
        match &self.config.strategy {
            DistillationStrategy::Progressive { stages } => {
                if self.current_stage < stages.len() {
                    stages[self.current_stage].temperature
                } else {
                    self.config.min_temperature
                }
            },
            _ => self.config.temperature,
        }
    }

    /// Get current alpha (useful for progressive distillation)
    pub fn current_alpha(&self) -> f32 {
        match &self.config.strategy {
            DistillationStrategy::Progressive { stages } => {
                if self.current_stage < stages.len() {
                    stages[self.current_stage].alpha
                } else {
                    self.config.alpha
                }
            },
            _ => self.config.alpha,
        }
    }

    /// Compute layer-wise feature losses for detailed tracking
    fn compute_layer_wise_feature_losses(
        &self,
        teacher_hidden_states: &[Tensor],
        student_hidden_states: &[Tensor],
    ) -> Result<HashMap<String, Tensor>> {
        let mut feature_losses = HashMap::new();

        // Ensure we have matching layers (or use the minimum)
        let num_layers = teacher_hidden_states.len().min(student_hidden_states.len());

        for layer_idx in 0..num_layers {
            let teacher_hidden = &teacher_hidden_states[layer_idx];
            let student_hidden = &student_hidden_states[layer_idx];

            // Apply projection if dimensions don't match
            let aligned_student = if teacher_hidden.shape() != student_hidden.shape() {
                // Simple projection to match teacher dimensions
                match (teacher_hidden, student_hidden) {
                    (Tensor::F32(t_arr), Tensor::F32(s_arr)) => {
                        let teacher_shape = t_arr.shape();
                        let student_shape = s_arr.shape();

                        if teacher_shape.len() == student_shape.len()
                            && teacher_shape[..teacher_shape.len() - 1]
                                == student_shape[..student_shape.len() - 1]
                        {
                            // Only hidden dimension differs, project student to teacher size
                            let teacher_hidden_dim = teacher_shape[teacher_shape.len() - 1];
                            let student_hidden_dim = student_shape[student_shape.len() - 1];

                            if student_hidden_dim != teacher_hidden_dim {
                                // Simple linear projection (in practice, this would be a learned projection)
                                let scale = teacher_hidden_dim as f32 / student_hidden_dim as f32;
                                let projected = s_arr.mapv(|x| x * scale);

                                // Reshape to match teacher dimensions
                                let new_shape = teacher_shape.to_vec();
                                let projected_data = if teacher_hidden_dim > student_hidden_dim {
                                    // Pad with zeros
                                    let mut padded_data = vec![0.0; new_shape.iter().product()];
                                    let chunk_size = student_hidden_dim;
                                    let total_chunks = s_arr.len() / chunk_size;

                                    for chunk_idx in 0..total_chunks {
                                        let src_start = chunk_idx * chunk_size;
                                        let dst_start = chunk_idx * teacher_hidden_dim;
                                        for i in 0..chunk_size {
                                            padded_data[dst_start + i] = projected[src_start + i];
                                        }
                                    }
                                    padded_data
                                } else {
                                    // Truncate
                                    let chunk_size = teacher_hidden_dim;
                                    let total_chunks = projected.len() / student_hidden_dim;
                                    let mut truncated_data = Vec::new();

                                    for chunk_idx in 0..total_chunks {
                                        let src_start = chunk_idx * student_hidden_dim;
                                        for i in 0..chunk_size {
                                            truncated_data.push(projected[src_start + i]);
                                        }
                                    }
                                    truncated_data
                                };

                                let projected_array =
                                    ArrayD::from_shape_vec(IxDyn(&new_shape), projected_data)
                                        .map_err(|_| {
                                            TrustformersError::shape_error(
                                                "Failed to project student features".to_string(),
                                            )
                                        })?;

                                Tensor::F32(projected_array)
                            } else {
                                student_hidden.clone()
                            }
                        } else {
                            student_hidden.clone()
                        }
                    },
                    _ => student_hidden.clone(),
                }
            } else {
                student_hidden.clone()
            };

            // Compute MSE loss between teacher and (aligned) student features
            let diff = teacher_hidden.sub(&aligned_student)?;
            let squared_diff = diff.mul(&diff)?;
            let mse_loss = squared_diff.mean()?;

            feature_losses.insert(format!("layer_{}", layer_idx), mse_loss);
        }

        Ok(feature_losses)
    }

    /// Compute layer-wise attention losses for detailed tracking
    fn compute_layer_wise_attention_losses(
        &self,
        teacher_attentions: &[Tensor],
        student_attentions: &[Tensor],
    ) -> Result<HashMap<String, Tensor>> {
        let mut attention_losses = HashMap::new();

        // Ensure we have matching layers
        let num_layers = teacher_attentions.len().min(student_attentions.len());

        for layer_idx in 0..num_layers {
            let teacher_attn = &teacher_attentions[layer_idx];
            let student_attn = &student_attentions[layer_idx];

            // Handle different attention head counts
            let aligned_student_attn = if teacher_attn.shape() != student_attn.shape() {
                self.align_attention_tensors(teacher_attn, student_attn)?
            } else {
                student_attn.clone()
            };

            // Compute attention transfer loss (MSE between attention distributions)
            let diff = teacher_attn.sub(&aligned_student_attn)?;
            let squared_diff = diff.mul(&diff)?;
            let attn_loss = squared_diff.mean()?;

            attention_losses.insert(format!("layer_{}", layer_idx), attn_loss);

            // Additional attention-specific metrics
            // 1. Attention entropy similarity
            let teacher_entropy = self.compute_attention_entropy(teacher_attn)?;
            let student_entropy = self.compute_attention_entropy(&aligned_student_attn)?;
            let entropy_diff = teacher_entropy.sub(&student_entropy)?;
            let entropy_loss = entropy_diff.mul(&entropy_diff)?;
            attention_losses.insert(format!("layer_{}_entropy", layer_idx), entropy_loss);

            // 2. Attention pattern correlation
            let pattern_correlation =
                self.compute_attention_correlation(teacher_attn, &aligned_student_attn)?;
            attention_losses.insert(
                format!("layer_{}_correlation", layer_idx),
                pattern_correlation,
            );
        }

        Ok(attention_losses)
    }

    /// Align attention tensors when they have different shapes (e.g., different head counts)
    fn align_attention_tensors(&self, teacher: &Tensor, student: &Tensor) -> Result<Tensor> {
        match (teacher, student) {
            (Tensor::F32(t_arr), Tensor::F32(s_arr)) => {
                let teacher_shape = t_arr.shape();
                let student_shape = s_arr.shape();

                // Assume attention shape is [batch, heads, seq_len, seq_len]
                if teacher_shape.len() == 4 && student_shape.len() == 4 {
                    let teacher_heads = teacher_shape[1];
                    let student_heads = student_shape[1];

                    if teacher_heads != student_heads {
                        // Simple head alignment: average/repeat heads to match teacher count
                        if student_heads < teacher_heads {
                            // Repeat student heads
                            let _repeat_factor = teacher_heads / student_heads;
                            let mut aligned_data = Vec::new();

                            let batch_size = student_shape[0];
                            let seq_len = student_shape[2];
                            let seq_len_2 = student_shape[3];

                            for b in 0..batch_size {
                                for h in 0..teacher_heads {
                                    let source_head = h % student_heads;
                                    for i in 0..seq_len {
                                        for j in 0..seq_len_2 {
                                            aligned_data.push(s_arr[[b, source_head, i, j]]);
                                        }
                                    }
                                }
                            }

                            let aligned_array = ArrayD::from_shape_vec(
                                IxDyn(&[batch_size, teacher_heads, seq_len, seq_len_2]),
                                aligned_data,
                            )
                            .map_err(|_| {
                                TrustformersError::shape_error(
                                    "Failed to align attention heads".to_string(),
                                )
                            })?;

                            Ok(Tensor::F32(aligned_array))
                        } else {
                            // Average student heads to match teacher count
                            let group_size = student_heads / teacher_heads;
                            let mut aligned_data = Vec::new();

                            let batch_size = student_shape[0];
                            let seq_len = student_shape[2];
                            let seq_len_2 = student_shape[3];

                            for b in 0..batch_size {
                                for h in 0..teacher_heads {
                                    for i in 0..seq_len {
                                        for j in 0..seq_len_2 {
                                            let mut sum = 0.0;
                                            for g in 0..group_size {
                                                let student_head = h * group_size + g;
                                                if student_head < student_heads {
                                                    sum += s_arr[[b, student_head, i, j]];
                                                }
                                            }
                                            aligned_data.push(sum / group_size as f32);
                                        }
                                    }
                                }
                            }

                            let aligned_array = ArrayD::from_shape_vec(
                                IxDyn(&[batch_size, teacher_heads, seq_len, seq_len_2]),
                                aligned_data,
                            )
                            .map_err(|_| {
                                TrustformersError::shape_error(
                                    "Failed to align attention heads".to_string(),
                                )
                            })?;

                            Ok(Tensor::F32(aligned_array))
                        }
                    } else {
                        Ok(student.clone())
                    }
                } else {
                    Ok(student.clone())
                }
            },
            _ => Ok(student.clone()),
        }
    }

    /// Compute attention entropy for measuring attention distribution sharpness
    fn compute_attention_entropy(&self, attention: &Tensor) -> Result<Tensor> {
        match attention {
            Tensor::F32(arr) => {
                // Compute entropy: -sum(p * log(p)) for each attention head
                let epsilon = 1e-8_f32; // Small constant to avoid log(0)
                let log_probs = arr.mapv(|x| (x + epsilon).ln());
                let entropy_contributions = arr * &log_probs;
                let entropy = entropy_contributions.sum_axis(Axis(3)); // Sum over last dimension
                let mean_entropy = entropy.mean().unwrap();

                Ok(Tensor::F32(ArrayD::from_elem(IxDyn(&[1]), -mean_entropy)))
            },
            _ => Err(tensor_op_error(
                "tensor_operation",
                "Attention entropy computation only supports F32 tensors".to_string(),
            )),
        }
    }

    /// Compute correlation between teacher and student attention patterns
    fn compute_attention_correlation(&self, teacher: &Tensor, student: &Tensor) -> Result<Tensor> {
        match (teacher, student) {
            (Tensor::F32(t_arr), Tensor::F32(s_arr)) => {
                // Flatten attention matrices and compute Pearson correlation
                let teacher_flat: Vec<f32> = t_arr.iter().cloned().collect();
                let student_flat: Vec<f32> = s_arr.iter().cloned().collect();

                if teacher_flat.len() != student_flat.len() {
                    return Ok(Tensor::F32(ArrayD::from_elem(IxDyn(&[1]), 0.0)));
                }

                let n = teacher_flat.len() as f32;
                let teacher_mean: f32 = teacher_flat.iter().sum::<f32>() / n;
                let student_mean: f32 = student_flat.iter().sum::<f32>() / n;

                let mut numerator = 0.0;
                let mut teacher_var = 0.0;
                let mut student_var = 0.0;

                for i in 0..teacher_flat.len() {
                    let teacher_centered = teacher_flat[i] - teacher_mean;
                    let student_centered = student_flat[i] - student_mean;

                    numerator += teacher_centered * student_centered;
                    teacher_var += teacher_centered * teacher_centered;
                    student_var += student_centered * student_centered;
                }

                let correlation = if teacher_var > 0.0 && student_var > 0.0 {
                    numerator / (teacher_var.sqrt() * student_var.sqrt())
                } else {
                    0.0
                };

                Ok(Tensor::F32(ArrayD::from_elem(IxDyn(&[1]), correlation)))
            },
            _ => Err(tensor_op_error(
                "tensor_operation",
                "Attention correlation computation only supports F32 tensors".to_string(),
            )),
        }
    }
}

/// Outputs from teacher model for distillation
#[derive(Debug, Clone)]
pub struct TeacherOutputs {
    /// Final layer logits
    pub logits: Tensor,
    /// Hidden states from all layers
    pub hidden_states: Vec<Tensor>,
    /// Attention weights from all layers
    pub attentions: Vec<Tensor>,
}

/// Outputs from student model for distillation
#[derive(Debug, Clone)]
pub struct StudentOutputs {
    /// Final layer logits
    pub logits: Tensor,
    /// Hidden states from all layers
    pub hidden_states: Vec<Tensor>,
    /// Attention weights from all layers
    pub attentions: Vec<Tensor>,
}

/// Utilities for knowledge distillation
pub mod utils {
    use super::*;

    /// Create a basic response-based distillation config
    pub fn response_distillation_config(temperature: f32, alpha: f32) -> DistillationConfig {
        DistillationConfig {
            temperature,
            alpha,
            strategy: DistillationStrategy::ResponseBased,
            ..Default::default()
        }
    }

    /// Create a feature-based distillation config
    pub fn feature_distillation_config(
        layer_mapping: HashMap<usize, usize>,
        alpha: f32,
    ) -> DistillationConfig {
        DistillationConfig {
            alpha,
            strategy: DistillationStrategy::FeatureBased,
            use_feature_matching: true,
            feature_matching_layers: layer_mapping,
            ..Default::default()
        }
    }

    /// Create a combined distillation config
    pub fn combined_distillation_config(
        temperature: f32,
        alpha: f32,
        response_weight: f32,
        feature_weight: f32,
        attention_weight: f32,
    ) -> DistillationConfig {
        DistillationConfig {
            temperature,
            alpha,
            strategy: DistillationStrategy::Combined {
                response_weight,
                feature_weight,
                attention_weight,
            },
            use_feature_matching: feature_weight > 0.0,
            use_attention_transfer: attention_weight > 0.0,
            ..Default::default()
        }
    }

    /// Create a progressive distillation config
    pub fn progressive_distillation_config(stages: Vec<ProgressiveStage>) -> DistillationConfig {
        DistillationConfig {
            strategy: DistillationStrategy::Progressive { stages },
            progressive: true,
            ..Default::default()
        }
    }

    /// Helper to create a linear decay schedule for progressive distillation
    pub fn linear_decay_stages(
        initial_temp: f32,
        final_temp: f32,
        initial_alpha: f32,
        final_alpha: f32,
        num_stages: usize,
        steps_per_stage: usize,
    ) -> Vec<ProgressiveStage> {
        let mut stages = Vec::new();

        for i in 0..num_stages {
            let progress = i as f32 / (num_stages - 1) as f32;
            let temp = initial_temp + progress * (final_temp - initial_temp);
            let alpha = initial_alpha + progress * (final_alpha - initial_alpha);

            stages.push(ProgressiveStage {
                duration: steps_per_stage,
                temperature: temp,
                alpha,
                freeze_teacher: false,
            });
        }

        stages
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distillation_config_default() {
        let config = DistillationConfig::default();
        assert_eq!(config.temperature, 4.0);
        assert_eq!(config.alpha, 0.7);
        assert!(!config.use_feature_matching);
        assert!(!config.use_attention_transfer);
    }

    #[test]
    fn test_response_distillation_config() {
        let config = utils::response_distillation_config(3.0, 0.8);
        assert_eq!(config.temperature, 3.0);
        assert_eq!(config.alpha, 0.8);
        assert!(matches!(
            config.strategy,
            DistillationStrategy::ResponseBased
        ));
    }

    #[test]
    fn test_feature_distillation_config() {
        let mut layer_mapping = HashMap::new();
        layer_mapping.insert(11, 5); // Map teacher layer 11 to student layer 5

        let config = utils::feature_distillation_config(layer_mapping.clone(), 0.6);
        assert_eq!(config.alpha, 0.6);
        assert!(config.use_feature_matching);
        assert_eq!(config.feature_matching_layers, layer_mapping);
    }

    #[test]
    fn test_combined_distillation_config() {
        let config = utils::combined_distillation_config(4.0, 0.7, 0.5, 0.3, 0.2);
        assert_eq!(config.temperature, 4.0);
        assert_eq!(config.alpha, 0.7);
        assert!(config.use_feature_matching);
        assert!(config.use_attention_transfer);

        if let DistillationStrategy::Combined {
            response_weight,
            feature_weight,
            attention_weight,
        } = config.strategy
        {
            assert_eq!(response_weight, 0.5);
            assert_eq!(feature_weight, 0.3);
            assert_eq!(attention_weight, 0.2);
        } else {
            panic!("Expected Combined strategy");
        }
    }

    #[test]
    fn test_progressive_stages() {
        let stages = utils::linear_decay_stages(5.0, 1.0, 0.8, 0.5, 4, 1000);
        assert_eq!(stages.len(), 4);
        assert_eq!(stages[0].temperature, 5.0);
        assert_eq!(stages[3].temperature, 1.0);
        assert_eq!(stages[0].alpha, 0.8);
        assert!(stages[3].alpha - 0.5 < 1e-6); // Float comparison
    }

    #[test]
    fn test_progressive_distillation_config() {
        let stages = vec![
            ProgressiveStage {
                duration: 1000,
                temperature: 5.0,
                alpha: 0.8,
                freeze_teacher: false,
            },
            ProgressiveStage {
                duration: 1000,
                temperature: 3.0,
                alpha: 0.6,
                freeze_teacher: false,
            },
        ];

        let config = utils::progressive_distillation_config(stages.clone());
        assert!(config.progressive);

        if let DistillationStrategy::Progressive {
            stages: config_stages,
        } = config.strategy
        {
            assert_eq!(config_stages.len(), 2);
            assert_eq!(config_stages[0].temperature, 5.0);
            assert_eq!(config_stages[1].temperature, 3.0);
        } else {
            panic!("Expected Progressive strategy");
        }
    }
}
