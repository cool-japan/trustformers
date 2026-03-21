//! Comprehensive Types Module for Real-Time Metrics System
//!
//! This module contains all 110+ types organized into logical categories for optimal
//! maintainability and comprehension. Each type includes comprehensive documentation
//! and appropriate traits and implementations.
//!
//! ## Module Organization
//!
//! - **common**: Shared utility types (AtomicF32, etc.)
//! - **enums**: Classification and state enumerations
//! - **config**: System-wide configuration structures
//! - **data_structures**: Core data containers and buffers
//! - **metrics**: Performance measurement and analysis structures
//! - **monitoring**: Real-time monitoring and thread management
//! - **alerts**: Alert generation and threshold management
//! - **errors**: Comprehensive error handling
//! - **traits**: Key interfaces and abstractions
//! - **support**: Utility and helper structures
//! - **statistics**: Statistical analysis types
//! - **helpers**: Additional helper types
//! - **streaming**: Streaming aggregation types
//! - **aggregators**: Aggregator and configuration types

// Module declarations
pub mod aggregators;
pub mod alerts;
pub mod common;
pub mod config;
pub mod data_structures;
pub mod enums;
pub mod errors;
pub mod helpers;
pub mod metrics;
pub mod monitoring;
pub mod statistics;
pub mod streaming;
pub mod support;
pub mod traits;

// Re-export common types
pub use common::*;

// Re-export all enums
pub use enums::{
    CleanupPriority, EnforcementLevel, FeedbackType, ImpactArea, InsightType, MemoryType,
    MonitoringEventType, MonitoringScope, ObjectiveType, OptimizationDirection, QualityIssueType,
    RiskType, ThresholdDirection, TrendDirection,
};

// Re-export configuration types
pub use config::{
    AggregationConfig, MetricsCollectionConfig, MonitorConfiguration, OptimizationEngineConfig,
    ThresholdConfig,
};

// Re-export data structures
pub use data_structures::{
    AggregationWindow, BufferStatistics, CircularBuffer, DataPoint, EfficiencyMetrics,
    LatencyStatistics, ThroughputStatistics, TimestampedMetrics, UtilizationStatistics,
    VariabilityMeasures, WindowStatistics,
};

// Re-export metrics types
pub use metrics::{
    AggregationResult, ConfidenceIntervals, ImpactAssessment, PerformanceBaseline,
    PerformanceInsight, QualityMetrics, VariabilityBounds,
};

// Re-export monitoring types
pub use monitoring::{MonitorThread, MonitoringEvent, ThreadStatistics};

// Re-export alert types
pub use alerts::{AlertEvent, EvaluationStatistics, SuppressionInfo, ThresholdMonitoringState};

// Re-export error types
pub use errors::{ErrorHandlingPolicy, ProcessingError, RealTimeMetricsError};

// Re-export trait definitions
pub use traits::{
    LiveOptimizationAlgorithm, PipelineStage, QualityChecker, SampleRateAlgorithm,
    StatisticalProcessor, ThresholdEvaluator,
};

// Re-export support types
pub use support::{
    AlgorithmStatistics, CheckerStatistics, PipelineConfig, PipelineInput, PipelineOutput,
    PipelineStatistics, ProcessingContext, ProcessingResults, ProcessorStatistics,
    QualityCheckResult, QualityControlConfig, QualityIssue, QualityRequirements, QualityStandards,
    QualityViolation,
};

// Re-export statistical types
pub use statistics::{
    DistributionAnalysis, OptimizationContext, OptimizationObjective, OptimizationRecommendation,
    RiskFactor, StatisticalResult, ThresholdEvaluation, TrendAnalysis,
};

// Re-export helper types
pub use helpers::{
    ImpactAlert, ImpactAnalysis, ImpactMonitorConfig, OverheadMeasurement, RateAdjustment,
    RateControllerStats, SampleRateConfig,
};

// Re-export streaming types
pub use streaming::{
    AnomalyTracker, BackpressureController, FlowController, PublishingStatistics, ResultFormatter,
    StreamStatistics,
};

// Re-export aggregator types
pub use aggregators::{
    AdvancedStatistics, AggregationMetadata, AggregatorPerformanceMetrics, BasicStatistics,
    CompressedData, Compression, CompressionConfig, CompressionStatistics, ConfidenceMethod,
    CoordinationConfig, FormattedResult, HistogramBin, OutlierParameters, OutlierResult,
    PipelineStageStats, QualityCriteria, RecommendationType, StageMetrics, StreamingWorker,
    ValidationRule,
};

// Import and re-export types from parent modules for convenience
pub use super::super::types::{
    ActionType, AdjustmentReason, EstimationAlgorithm, FeedbackProcessor, FeedbackSource,
    OptimizationEventType, PerformanceDataPoint, PerformanceFeedback, PerformanceMeasurement,
    PerformanceTrend, RealTimeMetrics, RecommendedAction, SystemState, TestCharacteristics,
};

// Re-export SeverityLevel
pub use crate::performance_optimizer::test_characterization::pattern_engine::SeverityLevel;
