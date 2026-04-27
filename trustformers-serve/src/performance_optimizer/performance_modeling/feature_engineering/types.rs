//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
// use super::types::*; // Circular import removed
use crate::performance_optimizer::types::{PerformanceDataPoint, SystemState, TestCharacteristics};

use super::functions::{FeatureExtractor, FeatureSelector, FeatureTransformer};

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::performance_optimizer::performance_modeling::types::{
    FeatureEngineeringConfig, FeatureScalingMethod,
};

/// Transformed features after preprocessing
#[derive(Debug, Clone)]
pub struct TransformedFeatures {
    /// Transformed feature matrix
    pub feature_matrix: Vec<Vec<f64>>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Updated feature metadata
    pub feature_metadata: HashMap<String, FeatureMetadata>,
}
/// Feature type classification
#[derive(Debug, Clone)]
pub enum FeatureType {
    /// Raw numerical feature
    Numerical,
    /// Categorical feature (encoded)
    Categorical,
    /// Transformed feature
    Transformed,
    /// Scaled feature
    Scaled,
    /// Polynomial feature
    Polynomial,
    /// Interaction feature
    Interaction,
    /// Temporal feature
    Temporal,
}
/// Basic feature extractor for fundamental metrics
#[derive(Debug)]
pub struct BasicFeatureExtractor;
/// Summary statistics for a feature
#[derive(Debug, Clone)]
pub struct FeatureSummary {
    /// Feature mean
    pub mean: f32,
    /// Feature variance
    pub variance: f32,
    /// Minimum value
    pub min_value: f32,
    /// Maximum value
    pub max_value: f32,
}
/// Polynomial feature extractor
#[derive(Debug)]
pub struct PolynomialFeatureExtractor {
    pub(super) degree: usize,
}
impl PolynomialFeatureExtractor {
    pub fn new(degree: usize) -> Self {
        Self { degree }
    }
}
/// Importance-based selector
#[derive(Debug)]
pub struct ImportanceBasedSelector {
    pub(crate) threshold: f32,
}
impl ImportanceBasedSelector {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}
/// Raw features before transformation
#[derive(Debug, Clone)]
pub struct RawFeatures {
    /// Feature matrix
    pub feature_matrix: Vec<Vec<f64>>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Feature metadata
    pub feature_metadata: HashMap<String, FeatureMetadata>,
}
/// Feature statistics tracker
#[derive(Debug)]
pub struct FeatureStatisticsTracker {
    /// Per-feature statistics
    feature_stats: HashMap<String, FeatureStats>,
    /// Processing count
    processing_count: u64,
    /// Last update time
    last_updated: DateTime<Utc>,
}
impl FeatureStatisticsTracker {
    pub fn new() -> Self {
        Self {
            feature_stats: HashMap::new(),
            processing_count: 0,
            last_updated: Utc::now(),
        }
    }
    pub fn update_feature_statistics(&mut self, feature_name: &str, values: &[f64]) {
        if values.is_empty() {
            return;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let min_value = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_value = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let stats = FeatureStats {
            mean,
            variance,
            min_value,
            max_value,
            missing_count: 0,
            total_count: values.len(),
        };
        self.feature_stats.insert(feature_name.to_string(), stats);
        self.last_updated = Utc::now();
    }
    pub fn increment_processing_count(&mut self) {
        self.processing_count += 1;
    }
    pub fn get_current_statistics(&self) -> FeatureStatistics {
        FeatureStatistics {
            total_features: self.feature_stats.len(),
            processing_count: self.processing_count,
            last_updated: self.last_updated,
            feature_summary: self
                .feature_stats
                .iter()
                .map(|(name, stats)| {
                    (
                        name.clone(),
                        FeatureSummary {
                            mean: stats.mean as f32,
                            variance: stats.variance as f32,
                            min_value: stats.min_value as f32,
                            max_value: stats.max_value as f32,
                        },
                    )
                })
                .collect(),
        }
    }
}
/// Variance threshold selector
#[derive(Debug)]
pub struct VarianceThresholdSelector {
    pub(crate) threshold: f32,
}
impl VarianceThresholdSelector {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
    pub(crate) fn calculate_variance(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance
    }
}
/// Comprehensive feature engineering orchestrator
#[derive(Debug)]
pub struct FeatureEngineeringOrchestrator {
    /// Feature engineering configuration
    config: Arc<RwLock<FeatureEngineeringConfig>>,
    /// Feature extractors registry
    extractors: HashMap<String, Box<dyn FeatureExtractor>>,
    /// Feature transformers registry
    transformers: HashMap<String, Box<dyn FeatureTransformer>>,
    /// Feature selectors registry
    selectors: HashMap<String, Box<dyn FeatureSelector>>,
    /// Feature statistics tracking
    statistics: Arc<RwLock<FeatureStatisticsTracker>>,
    /// Feature importance tracker
    pub(super) importance_tracker: Arc<RwLock<FeatureImportanceTracker>>,
}
impl FeatureEngineeringOrchestrator {
    /// Create new feature engineering orchestrator
    pub fn new(config: FeatureEngineeringConfig) -> Self {
        let mut orchestrator = Self {
            config: Arc::new(RwLock::new(config.clone())),
            extractors: HashMap::new(),
            transformers: HashMap::new(),
            selectors: HashMap::new(),
            statistics: Arc::new(RwLock::new(FeatureStatisticsTracker::new())),
            importance_tracker: Arc::new(RwLock::new(FeatureImportanceTracker::new())),
        };
        orchestrator.register_extractor("basic", Box::new(BasicFeatureExtractor));
        orchestrator.register_extractor("system", Box::new(SystemStateExtractor));
        orchestrator.register_extractor("test", Box::new(TestCharacteristicsExtractor));
        orchestrator.register_extractor("temporal", Box::new(TemporalFeatureExtractor));
        orchestrator.register_extractor("interaction", Box::new(InteractionFeatureExtractor));
        orchestrator.register_extractor(
            "polynomial",
            Box::new(PolynomialFeatureExtractor::new(config.polynomial_degree)),
        );
        orchestrator.register_transformer("scaler", Box::new(StandardScaler::new()));
        orchestrator.register_transformer("normalizer", Box::new(MinMaxNormalizer::new()));
        orchestrator.register_transformer("log", Box::new(LogTransformer));
        orchestrator.register_transformer("robust", Box::new(RobustScaler::new()));
        orchestrator.register_selector(
            "variance",
            Box::new(VarianceThresholdSelector::new(config.selection_threshold)),
        );
        orchestrator.register_selector("correlation", Box::new(CorrelationSelector::new(0.9)));
        orchestrator.register_selector(
            "importance",
            Box::new(ImportanceBasedSelector::new(config.selection_threshold)),
        );
        orchestrator
    }
    /// Register feature extractor
    pub fn register_extractor(&mut self, name: &str, extractor: Box<dyn FeatureExtractor>) {
        self.extractors.insert(name.to_string(), extractor);
    }
    /// Register feature transformer
    pub fn register_transformer(&mut self, name: &str, transformer: Box<dyn FeatureTransformer>) {
        self.transformers.insert(name.to_string(), transformer);
    }
    /// Register feature selector
    pub fn register_selector(&mut self, name: &str, selector: Box<dyn FeatureSelector>) {
        self.selectors.insert(name.to_string(), selector);
    }
    /// Extract and engineer features from data points
    pub async fn engineer_features(
        &self,
        data_points: &[PerformanceDataPoint],
    ) -> Result<EngineeredFeatures> {
        if data_points.is_empty() {
            return Err(anyhow!("No data points provided for feature engineering"));
        }
        let start_time = std::time::Instant::now();
        let raw_features = self.extract_features(data_points).await?;
        tracing::debug!(
            "Extracted {} raw features",
            raw_features.feature_names.len()
        );
        let transformed_features = self.transform_features(raw_features).await?;
        tracing::debug!(
            "Transformed features to {} dimensions",
            transformed_features.feature_matrix[0].len()
        );
        let selected_features = self.select_features(transformed_features).await?;
        tracing::debug!(
            "Selected {} features after selection",
            selected_features.feature_names.len()
        );
        self.validate_and_update_statistics(&selected_features).await?;
        let engineering_time = start_time.elapsed();
        tracing::info!("Feature engineering completed in {:?}", engineering_time);
        Ok(EngineeredFeatures {
            feature_matrix: selected_features.feature_matrix,
            feature_names: selected_features.feature_names,
            feature_metadata: selected_features.feature_metadata,
            engineering_time,
            statistics: self.statistics.read().get_current_statistics(),
        })
    }
    /// Extract features from data points
    async fn extract_features(&self, data_points: &[PerformanceDataPoint]) -> Result<RawFeatures> {
        let mut all_features: Vec<Vec<f64>> = Vec::new();
        let mut feature_names = Vec::new();
        let mut feature_metadata = HashMap::new();
        for (extractor_name, extractor) in &self.extractors {
            match extractor.extract_features(data_points) {
                Ok(extracted) => {
                    let _start_idx = feature_names.len();
                    for (i, data_point_features) in extracted.features.iter().enumerate() {
                        if i < all_features.len() {
                            all_features[i].extend(data_point_features.clone());
                        } else {
                            all_features.push(data_point_features.clone());
                        }
                    }
                    let extracted_count = extracted.names.len();
                    for name in extracted.names {
                        let full_name = format!("{}_{}", extractor_name, name);
                        feature_names.push(full_name.clone());
                        feature_metadata.insert(
                            full_name,
                            FeatureMetadata {
                                extractor: extractor_name.clone(),
                                feature_type: FeatureType::Numerical,
                                importance_score: 0.0,
                                correlation_with_target: 0.0,
                                variance: 0.0,
                                missing_rate: 0.0,
                            },
                        );
                    }
                    tracing::debug!(
                        "Extractor '{}' produced {} features",
                        extractor_name,
                        extracted_count
                    );
                },
                Err(e) => {
                    tracing::warn!("Feature extractor '{}' failed: {}", extractor_name, e);
                },
            }
        }
        if all_features.is_empty() || feature_names.is_empty() {
            return Err(anyhow!("No features were successfully extracted"));
        }
        Ok(RawFeatures {
            feature_matrix: all_features,
            feature_names,
            feature_metadata,
        })
    }
    /// Transform features using registered transformers
    async fn transform_features(&self, raw_features: RawFeatures) -> Result<TransformedFeatures> {
        let config = self.config.read();
        let mut transformed_matrix = raw_features.feature_matrix;
        let mut feature_names = raw_features.feature_names;
        let mut feature_metadata = raw_features.feature_metadata;
        if let Some(scaler) = self.get_scaler_for_method(&config.scaling_method) {
            let scaling_result = scaler.fit_transform(&transformed_matrix)?;
            transformed_matrix = scaling_result.transformed_data;
            for (_name, metadata) in feature_metadata.iter_mut() {
                metadata.feature_type = FeatureType::Scaled;
            }
        }
        if config.enable_log_transforms {
            if let Some(log_transformer) = self.transformers.get("log") {
                match log_transformer.transform(&transformed_matrix) {
                    Ok(log_result) => {
                        for (i, log_features) in log_result.transformed_data.iter().enumerate() {
                            if i < transformed_matrix.len() {
                                transformed_matrix[i].extend(log_features.clone());
                            }
                        }
                        for name in &feature_names.clone() {
                            let log_name = format!("log_{}", name);
                            feature_names.push(log_name.clone());
                            feature_metadata.insert(
                                log_name,
                                FeatureMetadata {
                                    extractor: "log_transformer".to_string(),
                                    feature_type: FeatureType::Transformed,
                                    importance_score: 0.0,
                                    correlation_with_target: 0.0,
                                    variance: 0.0,
                                    missing_rate: 0.0,
                                },
                            );
                        }
                    },
                    Err(e) => {
                        tracing::warn!("Log transformation failed: {}", e);
                    },
                }
            }
        }
        if config.enable_polynomial_features {
            if let Some(poly_extractor) = self.extractors.get("polynomial") {
                let dummy_points: Vec<PerformanceDataPoint> = (0..transformed_matrix.len())
                    .map(|i| self.create_dummy_point_from_features(&transformed_matrix[i]))
                    .collect();
                match poly_extractor.extract_features(&dummy_points) {
                    Ok(poly_result) => {
                        for (i, poly_features) in poly_result.features.iter().enumerate() {
                            if i < transformed_matrix.len() {
                                transformed_matrix[i].extend(poly_features.clone());
                            }
                        }
                        for name in poly_result.names {
                            feature_names.push(name.clone());
                            feature_metadata.insert(
                                name,
                                FeatureMetadata {
                                    extractor: "polynomial".to_string(),
                                    feature_type: FeatureType::Polynomial,
                                    importance_score: 0.0,
                                    correlation_with_target: 0.0,
                                    variance: 0.0,
                                    missing_rate: 0.0,
                                },
                            );
                        }
                    },
                    Err(e) => {
                        tracing::warn!("Polynomial feature generation failed: {}", e);
                    },
                }
            }
        }
        Ok(TransformedFeatures {
            feature_matrix: transformed_matrix,
            feature_names,
            feature_metadata,
        })
    }
    /// Select features using registered selectors
    async fn select_features(
        &self,
        transformed_features: TransformedFeatures,
    ) -> Result<SelectedFeatures> {
        let config = self.config.read();
        let mut feature_matrix = transformed_features.feature_matrix;
        let mut feature_names = transformed_features.feature_names;
        let mut feature_metadata = transformed_features.feature_metadata;
        if let Some(variance_selector) = self.selectors.get("variance") {
            match variance_selector.select_features(&feature_matrix, &feature_names) {
                Ok(selection_result) => {
                    let selected_indices = selection_result.selected_indices;
                    feature_matrix = feature_matrix
                        .iter()
                        .map(|row| selected_indices.iter().map(|&i| row[i]).collect())
                        .collect();
                    let new_feature_names: Vec<String> =
                        selected_indices.iter().map(|&i| feature_names[i].clone()).collect();
                    let new_metadata: HashMap<String, FeatureMetadata> = new_feature_names
                        .iter()
                        .filter_map(|name| {
                            feature_metadata.get(name).map(|meta| (name.clone(), meta.clone()))
                        })
                        .collect();
                    feature_names = new_feature_names;
                    feature_metadata = new_metadata;
                    tracing::debug!("Variance selection kept {} features", feature_names.len());
                },
                Err(e) => {
                    tracing::warn!("Variance threshold selection failed: {}", e);
                },
            }
        }
        if let Some(correlation_selector) = self.selectors.get("correlation") {
            match correlation_selector.select_features(&feature_matrix, &feature_names) {
                Ok(selection_result) => {
                    let selected_indices = selection_result.selected_indices;
                    feature_matrix = feature_matrix
                        .iter()
                        .map(|row| selected_indices.iter().map(|&i| row[i]).collect())
                        .collect();
                    let new_feature_names: Vec<String> =
                        selected_indices.iter().map(|&i| feature_names[i].clone()).collect();
                    let new_metadata: HashMap<String, FeatureMetadata> = new_feature_names
                        .iter()
                        .filter_map(|name| {
                            feature_metadata.get(name).map(|meta| (name.clone(), meta.clone()))
                        })
                        .collect();
                    feature_names = new_feature_names;
                    feature_metadata = new_metadata;
                    tracing::debug!(
                        "Correlation selection kept {} features",
                        feature_names.len()
                    );
                },
                Err(e) => {
                    tracing::warn!("Correlation-based selection failed: {}", e);
                },
            }
        }
        if let Some(max_features) = config.max_features {
            if feature_names.len() > max_features {
                let keep_indices: Vec<usize> = (0..max_features).collect();
                feature_matrix = feature_matrix
                    .iter()
                    .map(|row| keep_indices.iter().map(|&i| row[i]).collect())
                    .collect();
                let new_feature_names: Vec<String> =
                    keep_indices.iter().map(|&i| feature_names[i].clone()).collect();
                let new_metadata: HashMap<String, FeatureMetadata> = new_feature_names
                    .iter()
                    .filter_map(|name| {
                        feature_metadata.get(name).map(|meta| (name.clone(), meta.clone()))
                    })
                    .collect();
                feature_names = new_feature_names;
                feature_metadata = new_metadata;
                tracing::debug!(
                    "Feature limit applied, kept {} features",
                    feature_names.len()
                );
            }
        }
        Ok(SelectedFeatures {
            feature_matrix,
            feature_names,
            feature_metadata,
        })
    }
    /// Validate features and update statistics
    async fn validate_and_update_statistics(&self, features: &SelectedFeatures) -> Result<()> {
        let mut statistics = self.statistics.write();
        for (feature_idx, feature_name) in features.feature_names.iter().enumerate() {
            let feature_values: Vec<f64> =
                features.feature_matrix.iter().map(|row| row[feature_idx]).collect();
            statistics.update_feature_statistics(feature_name, &feature_values);
        }
        statistics.increment_processing_count();
        Ok(())
    }
    /// Get scaler for scaling method
    fn get_scaler_for_method(
        &self,
        method: &FeatureScalingMethod,
    ) -> Option<&Box<dyn FeatureTransformer>> {
        match method {
            FeatureScalingMethod::None => None,
            FeatureScalingMethod::StandardScaling => self.transformers.get("scaler"),
            FeatureScalingMethod::MinMaxScaling => self.transformers.get("normalizer"),
            FeatureScalingMethod::RobustScaling => self.transformers.get("robust"),
        }
    }
    /// Create dummy data point from feature vector (for polynomial feature generation)
    fn create_dummy_point_from_features(&self, features: &[f64]) -> PerformanceDataPoint {
        PerformanceDataPoint {
            parallelism: features.first().copied().unwrap_or(1.0) as usize,
            throughput: features.get(1).copied().unwrap_or(100.0),
            latency: Duration::from_millis(features.get(2).copied().unwrap_or(10.0) as u64),
            cpu_utilization: features.get(3).copied().unwrap_or(0.5) as f32,
            memory_utilization: features.get(4).copied().unwrap_or(0.5) as f32,
            resource_efficiency: features.get(5).copied().unwrap_or(0.8) as f32,
            timestamp: Utc::now(),
            test_characteristics: TestCharacteristics::default(),
            system_state: SystemState::default(),
        }
    }
    /// Get feature engineering statistics
    pub fn get_statistics(&self) -> FeatureStatistics {
        self.statistics.read().get_current_statistics()
    }
    /// Update feature importance from model feedback
    pub fn update_feature_importance(&self, importance_scores: HashMap<String, f32>) -> Result<()> {
        let mut tracker = self.importance_tracker.write();
        tracker.update_importance(importance_scores);
        Ok(())
    }
}
/// Temporal feature extractor
#[derive(Debug)]
pub struct TemporalFeatureExtractor;
/// Feature importance tracker
#[derive(Debug)]
pub struct FeatureImportanceTracker {
    /// Current importance scores
    importance_scores: HashMap<String, f32>,
    /// Importance history
    importance_history: Vec<(DateTime<Utc>, HashMap<String, f32>)>,
}
impl FeatureImportanceTracker {
    pub fn new() -> Self {
        Self {
            importance_scores: HashMap::new(),
            importance_history: Vec::new(),
        }
    }
    pub fn update_importance(&mut self, new_scores: HashMap<String, f32>) {
        self.importance_history.push((Utc::now(), new_scores.clone()));
        self.importance_scores = new_scores;
        if self.importance_history.len() > 100 {
            self.importance_history.remove(0);
        }
    }
    pub fn get_current_importance(&self) -> HashMap<String, f32> {
        self.importance_scores.clone()
    }
}
/// Correlation-based selector
#[derive(Debug)]
pub struct CorrelationSelector {
    pub(crate) threshold: f64,
}
impl CorrelationSelector {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
    pub(crate) fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }
        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;
        let numerator: f64 =
            x.iter().zip(y.iter()).map(|(xi, yi)| (xi - mean_x) * (yi - mean_y)).sum();
        let sum_sq_x: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum();
        let sum_sq_y: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();
        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        if denominator > 1e-12 {
            numerator / denominator
        } else {
            0.0
        }
    }
}
/// Transformation result
#[derive(Debug, Clone)]
pub struct TransformationResult {
    /// Transformed feature data
    pub transformed_data: Vec<Vec<f64>>,
    /// Information about transformation
    pub transformation_info: String,
}
/// Interaction feature extractor
#[derive(Debug)]
pub struct InteractionFeatureExtractor;
/// Final engineered features
#[derive(Debug, Clone)]
pub struct EngineeredFeatures {
    /// Final feature matrix
    pub feature_matrix: Vec<Vec<f64>>,
    /// Final feature names
    pub feature_names: Vec<String>,
    /// Feature metadata
    pub feature_metadata: HashMap<String, FeatureMetadata>,
    /// Total engineering time
    pub engineering_time: Duration,
    /// Feature statistics
    pub statistics: FeatureStatistics,
}
/// System state feature extractor
#[derive(Debug)]
pub struct SystemStateExtractor;
/// Standard scaler (z-score normalization)
#[derive(Debug)]
pub struct StandardScaler {
    pub(super) means: Option<Vec<f64>>,
    pub(super) stds: Option<Vec<f64>>,
}
impl StandardScaler {
    pub fn new() -> Self {
        Self {
            means: None,
            stds: None,
        }
    }
    pub(crate) fn calculate_statistics(&mut self, features: &[Vec<f64>]) -> Result<()> {
        if features.is_empty() {
            return Err(anyhow!("No features provided for scaling"));
        }
        let n_samples = features.len() as f64;
        let n_features = features[0].len();
        let mut means = vec![0.0; n_features];
        let mut stds = vec![0.0; n_features];
        for sample in features {
            for (j, &value) in sample.iter().enumerate() {
                means[j] += value;
            }
        }
        for mean in &mut means {
            *mean /= n_samples;
        }
        for sample in features {
            for (j, &value) in sample.iter().enumerate() {
                stds[j] += (value - means[j]).powi(2);
            }
        }
        for std in &mut stds {
            *std = (*std / n_samples).sqrt();
            if *std == 0.0 {
                *std = 1.0;
            }
        }
        self.means = Some(means);
        self.stds = Some(stds);
        Ok(())
    }
}
/// Logarithmic transformer
#[derive(Debug)]
pub struct LogTransformer;
/// Robust scaler using median and IQR
#[derive(Debug)]
pub struct RobustScaler {
    pub(super) medians: Option<Vec<f64>>,
    pub(super) iqrs: Option<Vec<f64>>,
}
impl RobustScaler {
    pub fn new() -> Self {
        Self {
            medians: None,
            iqrs: None,
        }
    }
    pub(crate) fn calculate_robust_statistics(&mut self, features: &[Vec<f64>]) -> Result<()> {
        if features.is_empty() {
            return Err(anyhow!("No features provided for robust scaling"));
        }
        let n_features = features[0].len();
        let mut medians = Vec::new();
        let mut iqrs = Vec::new();
        for j in 0..n_features {
            let mut values: Vec<f64> = features.iter().map(|sample| sample[j]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = values.len();
            let median = if n.is_multiple_of(2) {
                (values[n / 2 - 1] + values[n / 2]) / 2.0
            } else {
                values[n / 2]
            };
            let q1_idx = n / 4;
            let q3_idx = 3 * n / 4;
            let q1 = values[q1_idx];
            let q3 = values[q3_idx];
            let iqr = q3 - q1;
            medians.push(median);
            iqrs.push(if iqr > 1e-12 { iqr } else { 1.0 });
        }
        self.medians = Some(medians);
        self.iqrs = Some(iqrs);
        Ok(())
    }
}
/// Min-Max normalizer
#[derive(Debug)]
pub struct MinMaxNormalizer {
    pub(super) mins: Option<Vec<f64>>,
    pub(super) maxs: Option<Vec<f64>>,
}
impl MinMaxNormalizer {
    pub fn new() -> Self {
        Self {
            mins: None,
            maxs: None,
        }
    }
    pub(crate) fn calculate_bounds(&mut self, features: &[Vec<f64>]) -> Result<()> {
        if features.is_empty() {
            return Err(anyhow!("No features provided for normalization"));
        }
        let n_features = features[0].len();
        let mut mins = vec![f64::INFINITY; n_features];
        let mut maxs = vec![f64::NEG_INFINITY; n_features];
        for sample in features {
            for (j, &value) in sample.iter().enumerate() {
                mins[j] = mins[j].min(value);
                maxs[j] = maxs[j].max(value);
            }
        }
        for (min, max) in mins.iter_mut().zip(maxs.iter_mut()) {
            if (*max - *min).abs() < 1e-12 {
                *max = *min + 1.0;
            }
        }
        self.mins = Some(mins);
        self.maxs = Some(maxs);
        Ok(())
    }
}
/// Test characteristics feature extractor
#[derive(Debug)]
pub struct TestCharacteristicsExtractor;
/// Extracted features from extractors
#[derive(Debug, Clone)]
pub struct ExtractedFeatures {
    /// Feature matrix (samples x features)
    pub features: Vec<Vec<f64>>,
    /// Feature names
    pub names: Vec<String>,
}
/// Feature selection result
#[derive(Debug, Clone)]
pub struct SelectionResult {
    /// Indices of selected features
    pub selected_indices: Vec<usize>,
    /// Names of selected features
    pub selected_names: Vec<String>,
    /// Information about selection process
    pub selection_info: String,
}
/// Feature metadata
#[derive(Debug, Clone)]
pub struct FeatureMetadata {
    /// Extractor that created this feature
    pub extractor: String,
    /// Feature type
    pub feature_type: FeatureType,
    /// Importance score
    pub importance_score: f32,
    /// Correlation with target
    pub correlation_with_target: f32,
    /// Feature variance
    pub variance: f32,
    /// Missing value rate
    pub missing_rate: f32,
}
/// Selected features after selection
#[derive(Debug, Clone)]
pub struct SelectedFeatures {
    /// Selected feature matrix
    pub feature_matrix: Vec<Vec<f64>>,
    /// Selected feature names
    pub feature_names: Vec<String>,
    /// Selected feature metadata
    pub feature_metadata: HashMap<String, FeatureMetadata>,
}
#[derive(Debug, Clone)]
struct FeatureStats {
    mean: f64,
    variance: f64,
    min_value: f64,
    max_value: f64,
    missing_count: usize,
    total_count: usize,
}
/// Feature statistics summary
#[derive(Debug, Clone)]
pub struct FeatureStatistics {
    /// Total number of features
    pub total_features: usize,
    /// Number of processing runs
    pub processing_count: u64,
    /// Last update time
    pub last_updated: DateTime<Utc>,
    /// Per-feature summary
    pub feature_summary: HashMap<String, FeatureSummary>,
}
