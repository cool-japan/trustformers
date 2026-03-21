//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use anyhow::Result;
// Import types from real_time_metrics::types not re-exported through monitor::types
use super::super::types::TimestampedMetrics;

use super::types::{
    AnomalyAlgorithmStats, AnomalyDetectionConfig, AnomalyEvent, BaselineConfig,
    BaselineValidationResult, ForecastResult, PerformanceBaseline, ScalingDecision,
    ThreadPoolConfig, ThreadPoolMetrics, TrendAnalysisConfig, TrendAnalysisResult,
};

/// Trait for anomaly detection algorithms
pub trait AnomalyDetectionAlgorithm: std::fmt::Debug + Send + Sync {
    /// Detect anomalies in metrics data
    fn detect_anomaly(
        &self,
        metrics: &TimestampedMetrics,
        baseline: &PerformanceBaseline,
    ) -> Result<Option<AnomalyEvent>>;
    /// Get algorithm name
    fn name(&self) -> &str;
    /// Get detection confidence level
    fn confidence(&self) -> f32;
    /// Update algorithm parameters
    fn update_parameters(&mut self, config: &AnomalyDetectionConfig) -> Result<()>;
    /// Get algorithm statistics
    fn get_statistics(&self) -> AnomalyAlgorithmStats;
}
/// Trait for trend detection algorithms
pub trait TrendDetectionAlgorithm: std::fmt::Debug + Send + Sync {
    /// Analyze trends in historical data
    fn analyze_trend(&self, data: &[TimestampedMetrics]) -> Result<TrendAnalysisResult>;
    /// Get algorithm name
    fn name(&self) -> &str;
    /// Forecast future values
    fn forecast(&self, data: &[TimestampedMetrics], horizon: Duration) -> Result<ForecastResult>;
    /// Update algorithm parameters
    fn update_parameters(&mut self, config: &TrendAnalysisConfig) -> Result<()>;
}
/// Trait for thread scaling algorithms
pub trait ThreadScalingAlgorithm: std::fmt::Debug + Send + Sync {
    /// Determine if scaling is needed
    fn should_scale(
        &self,
        metrics: &ThreadPoolMetrics,
        config: &ThreadPoolConfig,
    ) -> ScalingDecision;
    /// Calculate optimal thread count
    fn calculate_optimal_threads(&self, metrics: &ThreadPoolMetrics) -> usize;
    /// Get algorithm name
    fn name(&self) -> &str;
    /// Update algorithm parameters
    fn update_parameters(&mut self, config: &ThreadPoolConfig) -> Result<()>;
}
/// Trait for baseline adaptation algorithms
pub trait BaselineAdaptationAlgorithm: std::fmt::Debug + Send + Sync {
    /// Calculate baseline adaptation
    fn adapt_baseline(
        &self,
        current: &PerformanceBaseline,
        new_data: &[TimestampedMetrics],
    ) -> Result<PerformanceBaseline>;
    /// Get algorithm name
    fn name(&self) -> &str;
    /// Validate baseline quality
    fn validate_baseline(&self, baseline: &PerformanceBaseline) -> BaselineValidationResult;
    /// Update algorithm parameters
    fn update_parameters(&mut self, config: &BaselineConfig) -> Result<()>;
}
