//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::performance_optimizer::types::PerformanceDataPoint;
use anyhow::Result;

use super::types::{ExtractedFeatures, SelectionResult, TransformationResult};

/// Trait for feature extractors
pub trait FeatureExtractor: std::fmt::Debug + Send + Sync {
    /// Extract features from performance data points
    fn extract_features(&self, data_points: &[PerformanceDataPoint]) -> Result<ExtractedFeatures>;
    /// Get extractor name
    fn name(&self) -> &str;
    /// Get feature descriptions
    fn feature_descriptions(&self) -> Vec<String>;
}
/// Trait for feature transformers
pub trait FeatureTransformer: std::fmt::Debug + Send + Sync {
    /// Transform feature matrix
    fn transform(&self, features: &[Vec<f64>]) -> Result<TransformationResult>;
    /// Fit transformer on data and transform
    fn fit_transform(&self, features: &[Vec<f64>]) -> Result<TransformationResult>;
    /// Get transformer name
    fn name(&self) -> &str;
}
/// Trait for feature selectors
pub trait FeatureSelector: std::fmt::Debug + Send + Sync {
    /// Select features from feature matrix
    fn select_features(
        &self,
        features: &[Vec<f64>],
        feature_names: &[String],
    ) -> Result<SelectionResult>;
    /// Get selector name
    fn name(&self) -> &str;
}
