//! # TrainingDynamicsConfig - Trait Implementations
//!
//! This module contains trait implementations for `TrainingDynamicsConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types_4::TrainingDynamicsConfig;

impl Default for TrainingDynamicsConfig {
    fn default() -> Self {
        Self {
            enable_loss_curve_analysis: true,
            enable_learning_rate_analysis: true,
            enable_batch_size_analysis: true,
            enable_convergence_detection: true,
            enable_plateau_identification: true,
            moving_average_window: 10,
            convergence_tolerance: 1e-6,
            plateau_threshold: 1e-4,
            min_epochs_for_convergence: 20,
            max_history_length: 10000,
        }
    }
}
