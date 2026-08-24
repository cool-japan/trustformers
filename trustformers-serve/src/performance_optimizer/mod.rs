//! Performance Optimization for Test Parallelization
//!
//! This module provides advanced performance optimization features including
//! adaptive parallelism, load balancing, resource optimization, and intelligent
//! scaling for the TrustformeRS test parallelization framework.
//!
//! ## Architecture
//!
//! The performance optimization system is organized into several focused modules:
//! - [`types`] - Core types, enums, and data structures for performance optimization
//! - [`adaptive_parallelism`] - Adaptive parallelism control and estimation
//! - [`system_models`] - Hardware resource modeling (CPU, Memory, IO, Network, GPU)
//! - [`resource_modeling`] - Advanced system resource modeling and performance profiling
//! - [`performance_modeling`] - Machine learning models for performance prediction and optimization
//! - [`feedback_systems`] - Performance feedback and machine learning models
//! - [`optimization_history`] - Performance tracking, trending, and history management
//! - [`recommendations`] - Optimization recommendation generation and scoring
//! - [`real_time_metrics`] - Real-time metrics collection, monitoring, and alerting system (Phase 39 modular refactoring)
//! - [`manager`] - Main PerformanceOptimizer orchestrating all components
//!
//! ## Removed in 0.2.1: the orphan `core/` subtree and refactoring debris
//!
//! `core/` (~2,000 lines across `mod.rs`, `types.rs`, `functions/`, and
//! `simplelinearregression_traits.rs`) was deleted rather than mounted. It was
//! never declared, so it had never been compiled, and it could not have
//! compiled as written: `simplelinearregression_traits.rs` resolved
//! `LearningAlgorithm`, `TrainingDataset` and `ModelState` through
//! `use super::types::*`, and its `super::types` defines none of them. It also
//! carried a `SimpleLinearRegression` whose `train` ignored the training set and
//! returned `ModelState::default()`, leaving `predict` on the hard-coded
//! `slope = 1.0, intercept = 0.0` it was constructed with — a "learning
//! algorithm" that reports success without learning. The live prediction path
//! is [`performance_modeling`]; the live `OptimizationResult` lives in
//! [`test_characterization`].
//!
//! Two stale refactoring artefacts were removed with it —
//! `test_characterization/types.rs.backup_v2` and
//! `test_characterization/manager.rs.backup_refactored` — plus a
//! `mod_imports_fix.txt` scratch note. They were not Rust modules but they sat
//! under `src/`, so `cargo package` shipped them to crates.io as if they were
//! part of the crate.
//!
//! ## Removed in 0.2.1: the module-wide `#![allow(dead_code)]`
//!
//! This module carried a blanket `#![allow(dead_code)]` justified as
//! "infrastructure that is being incrementally implemented". It was hiding 184
//! warnings, almost all of them private struct fields that were assigned in a
//! constructor and never read again: caches nothing looked in, `config` handles
//! nothing consulted, `*_history` deques that were pushed to and never
//! examined, sub-analysers that were built and never called.
//!
//! Those fields have been removed, along with the constructor arguments and
//! `let` bindings that fed them and the handful of private methods that nothing
//! called. Nothing that any code path reads was touched, and the crate's test
//! count did not fall. Where a removal made a whole type or capability vanish,
//! the site carries its own note explaining what went and why — see
//! `real_time_metrics::optimization::advanced_algorithms` for the write-only
//! feedback histories and `manager` for the component managers that held no
//! state.
//!
//! The lint is on now, so a field that stops being read will be caught the next
//! time somebody builds this crate rather than accumulating behind an allow.

pub mod adaptive_parallelism;
pub mod feedback_systems;
pub mod manager;
pub mod optimization_history;
pub mod performance_modeling;
pub mod real_time_metrics;
pub mod recommendations;
pub mod resource_modeling;
pub mod system_models;
pub mod test_characterization;
pub mod types;
pub mod types_ml;

// Re-export main types for backward compatibility
pub use manager::PerformanceOptimizer;
pub use types::*;
// types_ml is re-exported via types.rs

// Re-export component types for easy access
pub use adaptive_parallelism::PerformanceSnapshot;
pub use adaptive_parallelism::{AdaptiveLearningModel, PerformanceFeedbackSystem, TrainingDataset};
pub use adaptive_parallelism::{
    AdaptiveParallelismController, OptimalParallelismEstimator, ParallelismEstimate,
    PerformanceModel,
};
pub use feedback_systems::PerformanceFeedback;
pub use optimization_history::{
    OptimizationEvent, OptimizationHistory, OptimizationStatistics, PerformanceTrend,
};
pub use performance_modeling::{ModelAccuracyMetrics, PerformancePrediction, ValidationResult};
pub use real_time_metrics::aggregator::RealTimeDataAggregator;
pub use real_time_metrics::monitor::ParallelPerformanceMonitor;
pub use real_time_metrics::{
    LiveOptimizationEngine, MetricsCollectionConfig, NotificationManager, OptimizationEngineConfig,
    RealTimeMetricsCollector, ThresholdConfig,
};
// The two commented-out entries that stood in the list above were stale:
// `AlertManager` was described as "only stub implementation in monitor module"
// -- the stub is deleted, and the real one is re-exported here -- and
// `ThresholdMonitor` as "part of threshold module which is not implemented",
// which the 1,200-line `real_time_metrics::threshold::monitor` disproves.
pub use real_time_metrics::threshold::{AlertManager, ThresholdMonitor};
pub use recommendations::{
    BatchingRecommendation, OptimizationRecommendations, ResourceOptimizationRecommendation,
};
pub use resource_modeling::{
    HardwareDetector, PerformanceProfiler, ResourceModelingManager, ResourceUtilizationTracker,
    TemperatureMonitor, TopologyAnalyzer,
};
pub use system_models::{
    CpuModel, GpuModel, IoModel, MemoryModel, NetworkModel, SystemResourceModel,
};

#[cfg(test)]
mod types_ml_tests;

#[cfg(test)]
mod types_tests;
