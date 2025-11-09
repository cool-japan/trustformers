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

// Allow dead code for this module as it contains extensive infrastructure that is being
// incrementally implemented and integrated
#![allow(dead_code)]

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

// Re-export main types for backward compatibility
pub use manager::PerformanceOptimizer;
pub use types::*;

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
    LiveOptimizationEngine,
    MetricsCollectionConfig,
    NotificationManager,
    OptimizationEngineConfig,
    RealTimeMetricsCollector,
    ThresholdConfig,
    // AlertManager,  // Disabled: Only stub implementation in monitor module
    // ThresholdMonitor,  // Disabled: Part of threshold module which is not implemented
};
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
