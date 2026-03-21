//! Model Optimization for TrustformeRS C API
//!
//! This module provides comprehensive model optimization capabilities including:
//! - Automatic neural network pruning (structured and unstructured)
//! - Knowledge distillation framework with teacher-student training
//! - Architecture search and optimization
//! - Performance analysis and optimization recommendations

pub mod distillation;
pub mod nas;
pub mod optimizer;
pub mod pruning;

// Re-export main types and functions from submodules
pub use distillation::{
    DistillationConfig, DistillationLossWeights, DistillationMethod, DistillationTrainingConfig,
    FeatureAlignment, FeatureMatchingConfig, FeatureMatchingLoss, LearningRateSchedule,
    ScheduleType, StudentConfig, TeacherConfig,
};

pub use nas::{
    trustformers_nas_config_create, trustformers_nas_config_free, trustformers_nas_manager_create,
    trustformers_nas_manager_free, trustformers_nas_result_free, trustformers_nas_run_search,
    trustformers_nas_set_algorithm, trustformers_nas_set_hardware_constraints,
    trustformers_nas_set_objectives, trustformers_nas_set_search_params,
    trustformers_nas_start_search, NASManager, TrustformersNASConfig, TrustformersNASResult,
};

pub use optimizer::{
    LayerOptimizationState, LayerOptimizationStats, ModelOptimizer, OptimizationPhase,
    OptimizationState, OptimizationStats, SensitivityResults,
};

pub use pruning::{
    LayerPruningConfig, PruningConfig, PruningGranularity, PruningMethod, PruningSchedule,
    RecoveryConfig, SensitivityConfig,
};
