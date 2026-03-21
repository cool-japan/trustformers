//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::collector::*;
use super::super::types::*;
use crate::performance_optimizer::performance_modeling::ValidationResult;
use anyhow::Result;

use super::types::ProcessorConfig;

/// Statistical processor trait for advanced analysis
pub trait StatisticalProcessor: std::fmt::Debug {
    /// Process metrics and generate statistics
    fn process(&self, metrics: &[TimestampedMetrics]) -> Result<StatisticalResult>;
    /// Get processor name
    fn name(&self) -> &str;
    /// Get processor configuration
    fn config(&self) -> ProcessorConfig;
    /// Validate input data
    fn validate_input(&self, metrics: &[TimestampedMetrics]) -> Result<()>;
}
/// Quality scorer trait for data quality assessment
pub trait QualityScorer: std::fmt::Debug {
    /// Calculate quality score for metrics
    fn score(&self, metrics: &[TimestampedMetrics]) -> Result<f32>;
    /// Get scorer name
    fn name(&self) -> &str;
    /// Get quality criteria
    fn criteria(&self) -> QualityCriteria;
}
/// Data validator trait for input validation
pub trait DataValidator: std::fmt::Debug {
    /// Validate metrics data
    fn validate(&self, metrics: &TimestampedMetrics) -> Result<ValidationResult>;
    /// Get validator name
    fn name(&self) -> &str;
    /// Get validation rules
    fn rules(&self) -> Vec<ValidationRule>;
}
/// Outlier detector trait for anomaly detection
pub trait OutlierDetector: std::fmt::Debug {
    /// Detect outliers in metrics data
    fn detect(&self, metrics: &[TimestampedMetrics]) -> Result<Vec<OutlierResult>>;
    /// Get detector name
    fn name(&self) -> &str;
    /// Get detection parameters
    fn parameters(&self) -> OutlierParameters;
}
/// Aggregation result publisher trait
pub trait AggregationResultPublisher: std::fmt::Debug {
    /// Publish aggregation result
    fn publish(&self, result: &AggregationResult) -> Result<()>;
    /// Get publisher name
    fn name(&self) -> &str;
    /// Get delivery configuration
    fn delivery_config(&self) -> DeliveryConfig;
}
/// Compression algorithm trait
pub trait CompressionAlgorithm: std::fmt::Debug {
    /// Compress aggregation data
    fn compress(&self, data: &[AggregationResult]) -> Result<CompressedData>;
    /// Decompress aggregation data
    fn decompress(&self, data: &CompressedData) -> Result<Vec<AggregationResult>>;
    /// Get algorithm name
    fn name(&self) -> &str;
    /// Get compression ratio
    fn compression_ratio(&self) -> f32;
}
