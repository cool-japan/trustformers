//! Trait Definitions

use anyhow::Result;
use std::collections::HashMap;

// Import common types

// Import types from sibling modules
use super::aggregators::PipelineStageStats;
use super::config::ThresholdConfig;
use super::data_structures::TimestampedMetrics;
use super::errors::RealTimeMetricsError;
use super::statistics::{
    OptimizationContext, OptimizationRecommendation, StatisticalResult, ThresholdEvaluation,
};
use super::support::{
    AlgorithmStatistics, CheckerStatistics, PipelineInput, PipelineOutput, ProcessorStatistics,
    QualityCheckResult, QualityStandards,
};

// Import types from parent modules
pub use super::super::types::{
    ActionType, AdjustmentReason, EstimationAlgorithm, FeedbackProcessor, FeedbackSource,
    OptimizationEventType, PerformanceDataPoint, PerformanceFeedback, PerformanceMeasurement,
    PerformanceTrend, RealTimeMetrics, RecommendedAction, SystemState, TestCharacteristics,
};

// Import SeverityLevel from pattern engine
pub use crate::performance_optimizer::test_characterization::pattern_engine::SeverityLevel;

// TRAIT DEFINITIONS
// =============================================================================

/// Statistical processor trait for data analysis
///
/// Interface for statistical processors that analyze streaming metrics data
/// and generate statistical insights and analysis results.
pub trait StatisticalProcessor: Send + Sync {
    /// Process metrics data and generate statistics
    fn process(
        &self,
        data: &[TimestampedMetrics],
    ) -> Result<StatisticalResult, RealTimeMetricsError>;

    /// Get processor name
    fn name(&self) -> &str;

    /// Check if processor can handle data
    fn can_process(&self, data_type: &str) -> bool;

    /// Get processor configuration
    fn config(&self) -> &dyn std::any::Any;

    /// Reset processor state
    fn reset(&mut self);

    /// Get processor statistics
    fn statistics(&self) -> ProcessorStatistics;
}

/// Threshold evaluator trait for threshold monitoring
///
/// Interface for threshold evaluators that assess metric values against
/// configured thresholds and generate alerts when violations occur.
pub trait ThresholdEvaluator: Send + Sync {
    /// Evaluate threshold against current value
    fn evaluate(
        &self,
        config: &ThresholdConfig,
        value: f64,
    ) -> Result<ThresholdEvaluation, RealTimeMetricsError>;

    /// Get evaluator name
    fn name(&self) -> &str;

    /// Check if evaluator supports threshold type
    fn supports_threshold(&self, threshold_type: &str) -> bool;

    /// Update evaluator with historical data
    fn update_history(&mut self, data: &[TimestampedMetrics]);

    /// Get evaluator configuration
    fn configuration(&self) -> &dyn std::any::Any;
}

/// Live optimization algorithm trait
///
/// Interface for live optimization algorithms that analyze real-time
/// performance data and generate optimization recommendations.
pub trait LiveOptimizationAlgorithm: Send + Sync {
    /// Generate optimization recommendations
    fn optimize(
        &self,
        metrics: &RealTimeMetrics,
        history: &[TimestampedMetrics],
        context: &OptimizationContext,
    ) -> Result<Vec<OptimizationRecommendation>, RealTimeMetricsError>;

    /// Get algorithm name
    fn name(&self) -> &str;

    /// Get algorithm confidence for current data
    fn confidence(&self, data_quality: f32) -> f32;

    /// Check if algorithm is applicable
    fn is_applicable(&self, context: &OptimizationContext) -> bool;

    /// Update algorithm with feedback
    fn update_with_feedback(
        &mut self,
        feedback: &PerformanceFeedback,
    ) -> Result<(), RealTimeMetricsError>;

    /// Get algorithm statistics
    fn statistics(&self) -> AlgorithmStatistics;
}

/// Sample rate adjustment algorithm trait
///
/// Interface for algorithms that control adaptive sample rate adjustment
/// based on system conditions and performance requirements.
pub trait SampleRateAlgorithm: Send + Sync {
    /// Calculate optimal sample rate
    fn calculate_rate(
        &self,
        current_load: f32,
        target_accuracy: f32,
        resource_availability: f32,
    ) -> Result<f32, RealTimeMetricsError>;

    /// Get algorithm name
    fn name(&self) -> &str;

    /// Get algorithm parameters
    fn parameters(&self) -> HashMap<String, f64>;

    /// Update algorithm state
    fn update_state(&mut self, metrics: &RealTimeMetrics);

    /// Reset algorithm to initial state
    fn reset(&mut self);
}

/// Quality checker trait for data validation
///
/// Interface for quality checkers that validate data quality throughout
/// the processing pipeline.
pub trait QualityChecker: Send + Sync {
    /// Check data quality
    fn check(
        &self,
        data: &[TimestampedMetrics],
    ) -> Result<QualityCheckResult, RealTimeMetricsError>;

    /// Get checker name
    fn name(&self) -> &str;

    /// Get quality standards
    fn standards(&self) -> &QualityStandards;

    /// Update checker configuration
    fn update_standards(&mut self, standards: QualityStandards);

    /// Get checker statistics
    fn statistics(&self) -> CheckerStatistics;
}

/// Pipeline stage trait for processing pipeline
///
/// Interface for pipeline stages that process data in the metrics processing pipeline.
pub trait PipelineStage: Send + Sync {
    /// Process pipeline input and generate output
    fn process(&self, input: PipelineInput) -> Result<PipelineOutput, RealTimeMetricsError>;

    /// Get stage name
    fn name(&self) -> &str;

    /// Get stage configuration
    fn configuration(&self) -> &dyn std::any::Any;

    /// Get stage statistics
    fn statistics(&self) -> PipelineStageStats;

    /// Check if stage can process input type
    fn can_process(&self, input_type: &str) -> bool;
}

// =============================================================================
