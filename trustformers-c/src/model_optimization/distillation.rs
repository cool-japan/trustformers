//! Knowledge distillation functionality for TrustformeRS C API
//!
//! This module provides comprehensive knowledge distillation capabilities including:
//! - Teacher-student training frameworks
//! - Various distillation methods and strategies
//! - Feature matching and attention transfer

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Knowledge distillation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationConfig {
    /// Distillation method
    pub method: DistillationMethod,
    /// Temperature for softmax distillation
    pub temperature: f64,
    /// Loss weighting factors
    pub loss_weights: DistillationLossWeights,
    /// Teacher model configuration
    pub teacher_config: TeacherConfig,
    /// Student model configuration
    pub student_config: StudentConfig,
    /// Training configuration
    pub training_config: DistillationTrainingConfig,
    /// Feature matching configuration
    pub feature_matching: FeatureMatchingConfig,
}

/// Distillation methods
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub enum DistillationMethod {
    /// Response-based distillation (output logits)
    Response = 0,
    /// Feature-based distillation (intermediate layers)
    Feature = 1,
    /// Attention transfer distillation
    AttentionTransfer = 2,
    /// Relation-based distillation
    Relation = 3,
    /// Progressive knowledge distillation
    Progressive = 4,
    /// Online distillation
    Online = 5,
    /// Self-distillation
    SelfDistillation = 6,
}

/// Distillation loss weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationLossWeights {
    /// Weight for task loss (ground truth)
    pub task_loss_weight: f64,
    /// Weight for distillation loss (teacher knowledge)
    pub distillation_loss_weight: f64,
    /// Weight for feature matching loss
    pub feature_loss_weight: f64,
    /// Weight for attention loss
    pub attention_loss_weight: f64,
}

/// Teacher model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeacherConfig {
    /// Path to teacher model
    pub model_path: String,
    /// Teacher model type
    pub model_type: String,
    /// Use ensemble of teachers
    pub use_ensemble: bool,
    /// Ensemble teacher paths
    pub ensemble_paths: Vec<String>,
    /// Teacher inference batch size
    pub inference_batch_size: u32,
}

/// Student model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudentConfig {
    /// Path to student model (if pre-existing)
    pub model_path: Option<String>,
    /// Student model architecture
    pub architecture: String,
    /// Model compression ratio
    pub compression_ratio: f64,
    /// Hidden dimension reduction factor
    pub hidden_dim_factor: f64,
    /// Number of layers reduction
    pub layer_reduction: u32,
    /// Attention head reduction factor
    pub attention_head_factor: f64,
}

/// Distillation training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationTrainingConfig {
    /// Number of training epochs
    pub epochs: u32,
    /// Learning rate schedule
    pub learning_rate_schedule: LearningRateSchedule,
    /// Batch size
    pub batch_size: u32,
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: u32,
    /// Warmup steps
    pub warmup_steps: u32,
    /// Evaluation frequency
    pub eval_frequency: u32,
    /// Early stopping patience
    pub early_stopping_patience: u32,
}

/// Learning rate schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningRateSchedule {
    /// Initial learning rate
    pub initial_lr: f64,
    /// Schedule type
    pub schedule_type: ScheduleType,
    /// Decay factor for step/exponential schedules
    pub decay_factor: f64,
    /// Step size for step schedule
    pub step_size: u32,
}

/// Learning rate schedule types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub enum ScheduleType {
    Constant = 0,
    Linear = 1,
    Cosine = 2,
    Exponential = 3,
    Step = 4,
}

/// Feature matching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureMatchingConfig {
    /// Enable feature matching
    pub enable: bool,
    /// Feature matching loss type
    pub loss_type: FeatureMatchingLoss,
    /// Layer pairs for feature matching (teacher -> student)
    pub layer_pairs: Vec<(String, String)>,
    /// Feature alignment method
    pub alignment_method: FeatureAlignment,
}

/// Feature matching loss types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub enum FeatureMatchingLoss {
    L2 = 0,
    L1 = 1,
    Cosine = 2,
    KL = 3,
}

/// Feature alignment methods
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub enum FeatureAlignment {
    Linear = 0,
    Convolution = 1,
    Attention = 2,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            method: DistillationMethod::Response,
            temperature: 4.0,
            loss_weights: DistillationLossWeights {
                task_loss_weight: 0.7,
                distillation_loss_weight: 0.3,
                feature_loss_weight: 0.1,
                attention_loss_weight: 0.1,
            },
            teacher_config: TeacherConfig {
                model_path: "".to_string(),
                model_type: "transformer".to_string(),
                use_ensemble: false,
                ensemble_paths: vec![],
                inference_batch_size: 32,
            },
            student_config: StudentConfig {
                model_path: None,
                architecture: "transformer".to_string(),
                compression_ratio: 0.5,
                hidden_dim_factor: 0.5,
                layer_reduction: 6,
                attention_head_factor: 0.5,
            },
            training_config: DistillationTrainingConfig {
                epochs: 10,
                learning_rate_schedule: LearningRateSchedule {
                    initial_lr: 5e-5,
                    schedule_type: ScheduleType::Linear,
                    decay_factor: 0.1,
                    step_size: 1000,
                },
                batch_size: 32,
                gradient_accumulation_steps: 1,
                warmup_steps: 1000,
                eval_frequency: 1000,
                early_stopping_patience: 5,
            },
            feature_matching: FeatureMatchingConfig {
                enable: false,
                loss_type: FeatureMatchingLoss::L2,
                layer_pairs: vec![],
                alignment_method: FeatureAlignment::Linear,
            },
        }
    }
}
