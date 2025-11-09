//! Pruning functionality for TrustformeRS C API
//!
//! This module provides comprehensive neural network pruning capabilities including:
//! - Automatic neural network pruning (structured and unstructured)
//! - Various pruning methods and strategies
//! - Sensitivity analysis and recovery training

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Pruning method types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub enum PruningMethod {
    /// Magnitude-based pruning (remove smallest weights)
    Magnitude = 0,
    /// Gradual magnitude pruning with schedule
    GradualMagnitude = 1,
    /// Structured pruning (remove entire neurons/channels)
    Structured = 2,
    /// SNIP (Single-shot Network Pruning)
    SNIP = 3,
    /// GraSP (Gradient Signal Preservation)
    GraSP = 4,
    /// Lottery ticket hypothesis pruning
    LotteryTicket = 5,
    /// Fisher information based pruning
    Fisher = 6,
    /// Movement pruning
    Movement = 7,
    /// Attention-based pruning for transformers
    AttentionBased = 8,
}

/// Pruning granularity
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub enum PruningGranularity {
    /// Remove individual weights (fine-grained)
    Unstructured = 0,
    /// Remove entire neurons/filters
    Neuron = 1,
    /// Remove entire channels
    Channel = 2,
    /// Remove attention heads
    AttentionHead = 3,
    /// Remove entire layers
    Layer = 4,
    /// Remove transformer blocks
    Block = 5,
}

/// Pruning schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningSchedule {
    /// Initial sparsity ratio (0.0 to 1.0)
    pub initial_sparsity: f64,
    /// Final sparsity ratio (0.0 to 1.0)
    pub final_sparsity: f64,
    /// Number of pruning steps
    pub pruning_steps: u32,
    /// Frequency of pruning (every N steps)
    pub pruning_frequency: u32,
    /// Recovery period after pruning (steps)
    pub recovery_period: u32,
}

/// Pruning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningConfig {
    /// Pruning method
    pub method: PruningMethod,
    /// Pruning granularity
    pub granularity: PruningGranularity,
    /// Target sparsity ratio (0.0 to 1.0)
    pub target_sparsity: f64,
    /// Pruning schedule
    pub schedule: PruningSchedule,
    /// Layer-specific configurations
    pub layer_configs: HashMap<String, LayerPruningConfig>,
    /// Preserve important layers (embedding, output)
    pub preserve_layers: Vec<String>,
    /// Recovery training configuration
    pub recovery_config: RecoveryConfig,
    /// Sensitivity analysis settings
    pub sensitivity_config: SensitivityConfig,
}

/// Per-layer pruning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerPruningConfig {
    /// Layer name pattern
    pub layer_pattern: String,
    /// Layer-specific sparsity target
    pub sparsity_override: Option<f64>,
    /// Skip pruning for this layer
    pub skip_pruning: bool,
    /// Use different method for this layer
    pub method_override: Option<PruningMethod>,
}

/// Recovery training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryConfig {
    /// Number of recovery training epochs
    pub recovery_epochs: u32,
    /// Learning rate for recovery training
    pub learning_rate: f64,
    /// Batch size for recovery training
    pub batch_size: u32,
    /// Use knowledge distillation during recovery
    pub use_distillation: bool,
    /// Recovery dataset path
    pub recovery_dataset: Option<String>,
}

/// Sensitivity analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityConfig {
    /// Perform layer-wise sensitivity analysis
    pub analyze_layers: bool,
    /// Perform per-head sensitivity for attention
    pub analyze_attention_heads: bool,
    /// Number of samples for sensitivity analysis
    pub analysis_samples: u32,
    /// Sensitivity metric threshold
    pub sensitivity_threshold: f64,
}

/// Sensitivity analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityResults {
    /// Overall model sensitivity
    pub overall_sensitivity: f64,
    /// Per-layer sensitivity scores
    pub layer_sensitivity: HashMap<String, f64>,
    /// Per-head sensitivity (for attention models)
    pub head_sensitivity: HashMap<String, f64>,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            method: PruningMethod::Magnitude,
            granularity: PruningGranularity::Unstructured,
            target_sparsity: 0.9,
            schedule: PruningSchedule {
                initial_sparsity: 0.0,
                final_sparsity: 0.9,
                pruning_steps: 100,
                pruning_frequency: 1000,
                recovery_period: 100,
            },
            layer_configs: HashMap::new(),
            preserve_layers: vec![
                "embedding".to_string(),
                "output".to_string(),
                "classifier".to_string(),
            ],
            recovery_config: RecoveryConfig {
                recovery_epochs: 10,
                learning_rate: 1e-4,
                batch_size: 32,
                use_distillation: false,
                recovery_dataset: None,
            },
            sensitivity_config: SensitivityConfig {
                analyze_layers: true,
                analyze_attention_heads: false,
                analysis_samples: 1000,
                sensitivity_threshold: 0.01,
            },
        }
    }
}
