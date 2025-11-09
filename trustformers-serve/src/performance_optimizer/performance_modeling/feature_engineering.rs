//! Feature Engineering System for Performance Models
//!
//! This module provides comprehensive feature engineering capabilities including
//! feature extraction, preprocessing, transformation, selection, and advanced
//! feature engineering techniques for performance prediction models.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Datelike, Timelike, Utc};
use parking_lot::RwLock;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};

use super::types::*;
use crate::performance_optimizer::types::{PerformanceDataPoint, SystemState, TestCharacteristics};

// =============================================================================
// FEATURE ENGINEERING ORCHESTRATOR
// =============================================================================

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
    importance_tracker: Arc<RwLock<FeatureImportanceTracker>>,
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

        // Register default feature extractors
        orchestrator.register_extractor("basic", Box::new(BasicFeatureExtractor));
        orchestrator.register_extractor("system", Box::new(SystemStateExtractor));
        orchestrator.register_extractor("test", Box::new(TestCharacteristicsExtractor));
        orchestrator.register_extractor("temporal", Box::new(TemporalFeatureExtractor));
        orchestrator.register_extractor("interaction", Box::new(InteractionFeatureExtractor));
        orchestrator.register_extractor(
            "polynomial",
            Box::new(PolynomialFeatureExtractor::new(config.polynomial_degree)),
        );

        // Register default feature transformers
        orchestrator.register_transformer("scaler", Box::new(StandardScaler::new()));
        orchestrator.register_transformer("normalizer", Box::new(MinMaxNormalizer::new()));
        orchestrator.register_transformer("log", Box::new(LogTransformer));
        orchestrator.register_transformer("robust", Box::new(RobustScaler::new()));

        // Register default feature selectors
        orchestrator.register_selector(
            "variance",
            Box::new(VarianceThresholdSelector::new(config.selection_threshold)),
        );
        orchestrator.register_selector("correlation", Box::new(CorrelationSelector::new(0.9))); // Remove highly correlated features
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

        // Phase 1: Feature extraction
        let raw_features = self.extract_features(data_points).await?;
        tracing::debug!(
            "Extracted {} raw features",
            raw_features.feature_names.len()
        );

        // Phase 2: Feature transformation
        let transformed_features = self.transform_features(raw_features).await?;
        tracing::debug!(
            "Transformed features to {} dimensions",
            transformed_features.feature_matrix[0].len()
        );

        // Phase 3: Feature selection
        let selected_features = self.select_features(transformed_features).await?;
        tracing::debug!(
            "Selected {} features after selection",
            selected_features.feature_names.len()
        );

        // Phase 4: Feature validation and statistics
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

        // Extract features using each registered extractor
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

                    // Add feature names with extractor prefix
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

        // Apply scaling transformation
        if let Some(scaler) = self.get_scaler_for_method(&config.scaling_method) {
            let scaling_result = scaler.fit_transform(&transformed_matrix)?;
            transformed_matrix = scaling_result.transformed_data;

            // Update metadata with scaling info
            for (_name, metadata) in feature_metadata.iter_mut() {
                metadata.feature_type = FeatureType::Scaled;
            }
        }

        // Apply logarithmic transformation if enabled
        if config.enable_log_transforms {
            if let Some(log_transformer) = self.transformers.get("log") {
                match log_transformer.transform(&transformed_matrix) {
                    Ok(log_result) => {
                        // Add log features
                        for (i, log_features) in log_result.transformed_data.iter().enumerate() {
                            if i < transformed_matrix.len() {
                                transformed_matrix[i].extend(log_features.clone());
                            }
                        }

                        // Add log feature names
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

        // Apply polynomial features if enabled
        if config.enable_polynomial_features {
            if let Some(poly_extractor) = self.extractors.get("polynomial") {
                // Create dummy data points for polynomial extraction
                let dummy_points: Vec<PerformanceDataPoint> = (0..transformed_matrix.len())
                    .map(|i| self.create_dummy_point_from_features(&transformed_matrix[i]))
                    .collect();

                match poly_extractor.extract_features(&dummy_points) {
                    Ok(poly_result) => {
                        // Add polynomial features
                        for (i, poly_features) in poly_result.features.iter().enumerate() {
                            if i < transformed_matrix.len() {
                                transformed_matrix[i].extend(poly_features.clone());
                            }
                        }

                        // Add polynomial feature names
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

        // Apply variance threshold selection
        if let Some(variance_selector) = self.selectors.get("variance") {
            match variance_selector.select_features(&feature_matrix, &feature_names) {
                Ok(selection_result) => {
                    let selected_indices = selection_result.selected_indices;

                    // Filter features based on selection
                    feature_matrix = feature_matrix
                        .iter()
                        .map(|row| selected_indices.iter().map(|&i| row[i]).collect())
                        .collect();

                    let new_feature_names: Vec<String> =
                        selected_indices.iter().map(|&i| feature_names[i].clone()).collect();

                    // Update metadata
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

        // Apply correlation-based selection
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

        // Limit features if max_features is set
        if let Some(max_features) = config.max_features {
            if feature_names.len() > max_features {
                // Keep top features based on importance (simplified)
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

        // Update feature statistics
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
        // This is a simplified dummy implementation
        // In practice, you'd need proper mapping from features back to data point structure
        PerformanceDataPoint {
            parallelism: features.get(0).copied().unwrap_or(1.0) as usize,
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

impl FeatureEngineer for FeatureEngineeringOrchestrator {
    fn transform_features(&self, raw_features: &[f64]) -> Result<Vec<f64>> {
        // This is a simplified implementation for the trait
        // In practice, this would use the full transformation pipeline
        Ok(raw_features.to_vec())
    }

    fn feature_names(&self) -> Vec<String> {
        // Return cached feature names or generate them
        vec!["feature_1".to_string(), "feature_2".to_string()] // Simplified
    }

    fn feature_importance(&self) -> HashMap<String, f32> {
        self.importance_tracker.read().get_current_importance()
    }

    fn update_from_data(&mut self, _data: &[PerformanceDataPoint]) -> Result<()> {
        // This would trigger a full re-engineering of features
        // For now, just update statistics
        Ok(())
    }
}

// =============================================================================
// FEATURE EXTRACTORS
// =============================================================================

/// Trait for feature extractors
pub trait FeatureExtractor: std::fmt::Debug + Send + Sync {
    /// Extract features from performance data points
    fn extract_features(&self, data_points: &[PerformanceDataPoint]) -> Result<ExtractedFeatures>;

    /// Get extractor name
    fn name(&self) -> &str;

    /// Get feature descriptions
    fn feature_descriptions(&self) -> Vec<String>;
}

/// Basic feature extractor for fundamental metrics
#[derive(Debug)]
pub struct BasicFeatureExtractor;

impl FeatureExtractor for BasicFeatureExtractor {
    fn extract_features(&self, data_points: &[PerformanceDataPoint]) -> Result<ExtractedFeatures> {
        let mut features = Vec::new();
        let names = vec![
            "parallelism".to_string(),
            "parallelism_sqrt".to_string(),
            "parallelism_log".to_string(),
            "parallelism_squared".to_string(),
        ];

        for data_point in data_points {
            let parallelism = data_point.parallelism as f64;
            let point_features = vec![
                parallelism,
                parallelism.sqrt(),
                parallelism.ln(),
                parallelism * parallelism,
            ];
            features.push(point_features);
        }

        Ok(ExtractedFeatures { features, names })
    }

    fn name(&self) -> &str {
        "BasicFeatureExtractor"
    }

    fn feature_descriptions(&self) -> Vec<String> {
        vec![
            "Raw parallelism level".to_string(),
            "Square root of parallelism".to_string(),
            "Natural logarithm of parallelism".to_string(),
            "Parallelism squared".to_string(),
        ]
    }
}

/// System state feature extractor
#[derive(Debug)]
pub struct SystemStateExtractor;

impl FeatureExtractor for SystemStateExtractor {
    fn extract_features(&self, data_points: &[PerformanceDataPoint]) -> Result<ExtractedFeatures> {
        let mut features = Vec::new();
        let names = vec![
            "available_cores".to_string(),
            "available_memory_mb".to_string(),
            "load_average".to_string(),
            "active_processes".to_string(),
            "io_wait_percent".to_string(),
            "network_utilization".to_string(),
            "memory_pressure".to_string(),
            "cpu_pressure".to_string(),
        ];

        for data_point in data_points {
            let system = &data_point.system_state;
            let point_features = vec![
                system.available_cores as f64,
                system.available_memory_mb as f64,
                system.load_average as f64,
                system.active_processes as f64,
                system.io_wait_percent as f64,
                system.network_utilization as f64,
                // Derived features
                (system.available_memory_mb as f64) / (system.available_cores as f64).max(1.0), // Memory per core
                system.load_average as f64 / (system.available_cores as f64).max(1.0), // Load per core
            ];
            features.push(point_features);
        }

        Ok(ExtractedFeatures { features, names })
    }

    fn name(&self) -> &str {
        "SystemStateExtractor"
    }

    fn feature_descriptions(&self) -> Vec<String> {
        vec![
            "Available CPU cores".to_string(),
            "Available memory in MB".to_string(),
            "System load average".to_string(),
            "Number of active processes".to_string(),
            "I/O wait percentage".to_string(),
            "Network utilization".to_string(),
            "Memory pressure (memory per core)".to_string(),
            "CPU pressure (load per core)".to_string(),
        ]
    }
}

/// Test characteristics feature extractor
#[derive(Debug)]
pub struct TestCharacteristicsExtractor;

impl FeatureExtractor for TestCharacteristicsExtractor {
    fn extract_features(&self, data_points: &[PerformanceDataPoint]) -> Result<ExtractedFeatures> {
        let mut features = Vec::new();
        let names = vec![
            "avg_duration_secs".to_string(),
            "cpu_intensity".to_string(),
            "memory_intensity".to_string(),
            "io_intensity".to_string(),
            "dependency_complexity".to_string(),
            "parallel_capable".to_string(),
            "max_safe_concurrency".to_string(),
            "resource_intensity_score".to_string(),
        ];

        for data_point in data_points {
            let test_chars = &data_point.test_characteristics;
            let point_features = vec![
                test_chars.average_duration.as_secs_f64(),
                test_chars.resource_intensity.cpu_intensity as f64,
                test_chars.resource_intensity.memory_intensity as f64,
                test_chars.resource_intensity.io_intensity as f64,
                test_chars.dependency_complexity as f64,
                if test_chars.concurrency_requirements.parallel_capable { 1.0 } else { 0.0 },
                test_chars.concurrency_requirements.max_safe_concurrency as f64,
                // Composite resource intensity score
                (test_chars.resource_intensity.cpu_intensity
                    + test_chars.resource_intensity.memory_intensity
                    + test_chars.resource_intensity.io_intensity) as f64
                    / 3.0,
            ];
            features.push(point_features);
        }

        Ok(ExtractedFeatures { features, names })
    }

    fn name(&self) -> &str {
        "TestCharacteristicsExtractor"
    }

    fn feature_descriptions(&self) -> Vec<String> {
        vec![
            "Average test duration in seconds".to_string(),
            "CPU intensity score".to_string(),
            "Memory intensity score".to_string(),
            "I/O intensity score".to_string(),
            "Test dependency complexity".to_string(),
            "Whether test is parallel-capable".to_string(),
            "Maximum safe concurrency level".to_string(),
            "Overall resource intensity score".to_string(),
        ]
    }
}

/// Temporal feature extractor
#[derive(Debug)]
pub struct TemporalFeatureExtractor;

impl FeatureExtractor for TemporalFeatureExtractor {
    fn extract_features(&self, data_points: &[PerformanceDataPoint]) -> Result<ExtractedFeatures> {
        let mut features = Vec::new();
        let names = vec![
            "hour_of_day".to_string(),
            "day_of_week".to_string(),
            "day_of_month".to_string(),
            "month".to_string(),
            "hour_sin".to_string(),
            "hour_cos".to_string(),
            "day_sin".to_string(),
            "day_cos".to_string(),
        ];

        for data_point in data_points {
            let dt = data_point.timestamp;
            let hour = dt.hour() as f64;
            let day_of_week = dt.weekday().num_days_from_monday() as f64;
            let day_of_month = dt.day() as f64;
            let month = dt.month() as f64;

            // Cyclical encoding for temporal features
            let hour_radians = 2.0 * std::f64::consts::PI * hour / 24.0;
            let day_radians = 2.0 * std::f64::consts::PI * day_of_week / 7.0;

            let point_features = vec![
                hour,
                day_of_week,
                day_of_month,
                month,
                hour_radians.sin(),
                hour_radians.cos(),
                day_radians.sin(),
                day_radians.cos(),
            ];
            features.push(point_features);
        }

        Ok(ExtractedFeatures { features, names })
    }

    fn name(&self) -> &str {
        "TemporalFeatureExtractor"
    }

    fn feature_descriptions(&self) -> Vec<String> {
        vec![
            "Hour of day (0-23)".to_string(),
            "Day of week (0-6)".to_string(),
            "Day of month (1-31)".to_string(),
            "Month (1-12)".to_string(),
            "Sine of hour (cyclical)".to_string(),
            "Cosine of hour (cyclical)".to_string(),
            "Sine of day (cyclical)".to_string(),
            "Cosine of day (cyclical)".to_string(),
        ]
    }
}

/// Interaction feature extractor
#[derive(Debug)]
pub struct InteractionFeatureExtractor;

impl FeatureExtractor for InteractionFeatureExtractor {
    fn extract_features(&self, data_points: &[PerformanceDataPoint]) -> Result<ExtractedFeatures> {
        let mut features = Vec::new();
        let names = vec![
            "parallelism_x_cores".to_string(),
            "parallelism_x_load".to_string(),
            "cores_x_memory".to_string(),
            "cpu_intensity_x_parallelism".to_string(),
            "memory_intensity_x_cores".to_string(),
            "load_x_io_wait".to_string(),
            "duration_x_complexity".to_string(),
        ];

        for data_point in data_points {
            let parallelism = data_point.parallelism as f64;
            let system = &data_point.system_state;
            let test_chars = &data_point.test_characteristics;

            let point_features = vec![
                parallelism * (system.available_cores as f64),
                parallelism * (system.load_average as f64),
                (system.available_cores as f64) * (system.available_memory_mb as f64),
                (test_chars.resource_intensity.cpu_intensity as f64) * parallelism,
                (test_chars.resource_intensity.memory_intensity as f64)
                    * (system.available_cores as f64),
                (system.load_average as f64) * (system.io_wait_percent as f64),
                test_chars.average_duration.as_secs_f64()
                    * (test_chars.dependency_complexity as f64),
            ];
            features.push(point_features);
        }

        Ok(ExtractedFeatures { features, names })
    }

    fn name(&self) -> &str {
        "InteractionFeatureExtractor"
    }

    fn feature_descriptions(&self) -> Vec<String> {
        vec![
            "Parallelism × Available cores".to_string(),
            "Parallelism × System load".to_string(),
            "Available cores × Memory".to_string(),
            "CPU intensity × Parallelism".to_string(),
            "Memory intensity × Cores".to_string(),
            "System load × I/O wait".to_string(),
            "Duration × Dependency complexity".to_string(),
        ]
    }
}

/// Polynomial feature extractor
#[derive(Debug)]
pub struct PolynomialFeatureExtractor {
    degree: usize,
}

impl PolynomialFeatureExtractor {
    pub fn new(degree: usize) -> Self {
        Self { degree }
    }
}

impl FeatureExtractor for PolynomialFeatureExtractor {
    fn extract_features(&self, data_points: &[PerformanceDataPoint]) -> Result<ExtractedFeatures> {
        if self.degree < 2 {
            return Ok(ExtractedFeatures {
                features: Vec::new(),
                names: Vec::new(),
            });
        }

        // First extract base features
        let base_features = [
            data_points.iter().map(|d| d.parallelism as f64).collect::<Vec<_>>(),
            data_points
                .iter()
                .map(|d| d.system_state.load_average as f64)
                .collect::<Vec<_>>(),
            data_points
                .iter()
                .map(|d| d.test_characteristics.resource_intensity.cpu_intensity as f64)
                .collect::<Vec<_>>(),
        ];

        let mut features = Vec::new();
        let mut names = Vec::new();

        // Generate polynomial features
        for degree in 2..=self.degree {
            for base_idx in 0..base_features.len() {
                let feature_name = match base_idx {
                    0 => format!("parallelism^{}", degree),
                    1 => format!("load_avg^{}", degree),
                    2 => format!("cpu_intensity^{}", degree),
                    _ => format!("feature_{}^{}", base_idx, degree),
                };
                names.push(feature_name);
            }
        }

        // Generate polynomial values for each data point
        for i in 0..data_points.len() {
            let mut point_features = Vec::new();

            for degree in 2..=self.degree {
                for base_idx in 0..base_features.len() {
                    let base_value = base_features[base_idx][i];
                    let poly_value = base_value.powi(degree as i32);
                    point_features.push(poly_value);
                }
            }

            features.push(point_features);
        }

        Ok(ExtractedFeatures { features, names })
    }

    fn name(&self) -> &str {
        "PolynomialFeatureExtractor"
    }

    fn feature_descriptions(&self) -> Vec<String> {
        let mut descriptions = Vec::new();
        for degree in 2..=self.degree {
            descriptions.push(format!("Polynomial features of degree {}", degree));
        }
        descriptions
    }
}

// =============================================================================
// FEATURE TRANSFORMERS
// =============================================================================

/// Trait for feature transformers
pub trait FeatureTransformer: std::fmt::Debug + Send + Sync {
    /// Transform feature matrix
    fn transform(&self, features: &[Vec<f64>]) -> Result<TransformationResult>;

    /// Fit transformer on data and transform
    fn fit_transform(&self, features: &[Vec<f64>]) -> Result<TransformationResult>;

    /// Get transformer name
    fn name(&self) -> &str;
}

/// Standard scaler (z-score normalization)
#[derive(Debug)]
pub struct StandardScaler {
    means: Option<Vec<f64>>,
    stds: Option<Vec<f64>>,
}

impl StandardScaler {
    pub fn new() -> Self {
        Self {
            means: None,
            stds: None,
        }
    }

    fn calculate_statistics(&mut self, features: &[Vec<f64>]) -> Result<()> {
        if features.is_empty() {
            return Err(anyhow!("No features provided for scaling"));
        }

        let n_samples = features.len() as f64;
        let n_features = features[0].len();

        let mut means = vec![0.0; n_features];
        let mut stds = vec![0.0; n_features];

        // Calculate means
        for sample in features {
            for (j, &value) in sample.iter().enumerate() {
                means[j] += value;
            }
        }
        for mean in &mut means {
            *mean /= n_samples;
        }

        // Calculate standard deviations
        for sample in features {
            for (j, &value) in sample.iter().enumerate() {
                stds[j] += (value - means[j]).powi(2);
            }
        }
        for std in &mut stds {
            *std = (*std / n_samples).sqrt();
            if *std == 0.0 {
                *std = 1.0; // Prevent division by zero
            }
        }

        self.means = Some(means);
        self.stds = Some(stds);
        Ok(())
    }
}

impl FeatureTransformer for StandardScaler {
    fn transform(&self, features: &[Vec<f64>]) -> Result<TransformationResult> {
        let means = self.means.as_ref().ok_or_else(|| anyhow!("StandardScaler not fitted"))?;
        let stds = self.stds.as_ref().ok_or_else(|| anyhow!("StandardScaler not fitted"))?;

        let transformed_data = features
            .iter()
            .map(|sample| {
                sample
                    .iter()
                    .enumerate()
                    .map(|(j, &value)| (value - means[j]) / stds[j])
                    .collect()
            })
            .collect();

        Ok(TransformationResult {
            transformed_data,
            transformation_info: format!("StandardScaler: {} features scaled", means.len()),
        })
    }

    fn fit_transform(&self, features: &[Vec<f64>]) -> Result<TransformationResult> {
        let mut scaler = Self::new();
        scaler.calculate_statistics(features)?;
        scaler.transform(features)
    }

    fn name(&self) -> &str {
        "StandardScaler"
    }
}

/// Min-Max normalizer
#[derive(Debug)]
pub struct MinMaxNormalizer {
    mins: Option<Vec<f64>>,
    maxs: Option<Vec<f64>>,
}

impl MinMaxNormalizer {
    pub fn new() -> Self {
        Self {
            mins: None,
            maxs: None,
        }
    }

    fn calculate_bounds(&mut self, features: &[Vec<f64>]) -> Result<()> {
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

        // Ensure non-zero range
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

impl FeatureTransformer for MinMaxNormalizer {
    fn transform(&self, features: &[Vec<f64>]) -> Result<TransformationResult> {
        let mins = self.mins.as_ref().ok_or_else(|| anyhow!("MinMaxNormalizer not fitted"))?;
        let maxs = self.maxs.as_ref().ok_or_else(|| anyhow!("MinMaxNormalizer not fitted"))?;

        let transformed_data = features
            .iter()
            .map(|sample| {
                sample
                    .iter()
                    .enumerate()
                    .map(|(j, &value)| (value - mins[j]) / (maxs[j] - mins[j]))
                    .collect()
            })
            .collect();

        Ok(TransformationResult {
            transformed_data,
            transformation_info: format!("MinMaxNormalizer: {} features normalized", mins.len()),
        })
    }

    fn fit_transform(&self, features: &[Vec<f64>]) -> Result<TransformationResult> {
        let mut normalizer = Self::new();
        normalizer.calculate_bounds(features)?;
        normalizer.transform(features)
    }

    fn name(&self) -> &str {
        "MinMaxNormalizer"
    }
}

/// Logarithmic transformer
#[derive(Debug)]
pub struct LogTransformer;

impl FeatureTransformer for LogTransformer {
    fn transform(&self, features: &[Vec<f64>]) -> Result<TransformationResult> {
        let transformed_data = features
            .iter()
            .map(|sample| {
                sample
                    .iter()
                    .map(|&value| if value > 0.0 { (value + 1.0).ln() } else { 0.0 })
                    .collect()
            })
            .collect();

        Ok(TransformationResult {
            transformed_data,
            transformation_info: "LogTransformer: Applied log(x+1) transformation".to_string(),
        })
    }

    fn fit_transform(&self, features: &[Vec<f64>]) -> Result<TransformationResult> {
        self.transform(features)
    }

    fn name(&self) -> &str {
        "LogTransformer"
    }
}

/// Robust scaler using median and IQR
#[derive(Debug)]
pub struct RobustScaler {
    medians: Option<Vec<f64>>,
    iqrs: Option<Vec<f64>>,
}

impl RobustScaler {
    pub fn new() -> Self {
        Self {
            medians: None,
            iqrs: None,
        }
    }

    fn calculate_robust_statistics(&mut self, features: &[Vec<f64>]) -> Result<()> {
        if features.is_empty() {
            return Err(anyhow!("No features provided for robust scaling"));
        }

        let n_features = features[0].len();
        let mut medians = Vec::new();
        let mut iqrs = Vec::new();

        for j in 0..n_features {
            let mut values: Vec<f64> = features.iter().map(|sample| sample[j]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let n = values.len();
            let median = if n % 2 == 0 {
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

impl FeatureTransformer for RobustScaler {
    fn transform(&self, features: &[Vec<f64>]) -> Result<TransformationResult> {
        let medians = self.medians.as_ref().ok_or_else(|| anyhow!("RobustScaler not fitted"))?;
        let iqrs = self.iqrs.as_ref().ok_or_else(|| anyhow!("RobustScaler not fitted"))?;

        let transformed_data = features
            .iter()
            .map(|sample| {
                sample
                    .iter()
                    .enumerate()
                    .map(|(j, &value)| (value - medians[j]) / iqrs[j])
                    .collect()
            })
            .collect();

        Ok(TransformationResult {
            transformed_data,
            transformation_info: format!(
                "RobustScaler: {} features scaled using median and IQR",
                medians.len()
            ),
        })
    }

    fn fit_transform(&self, features: &[Vec<f64>]) -> Result<TransformationResult> {
        let mut scaler = Self::new();
        scaler.calculate_robust_statistics(features)?;
        scaler.transform(features)
    }

    fn name(&self) -> &str {
        "RobustScaler"
    }
}

// =============================================================================
// FEATURE SELECTORS
// =============================================================================

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

/// Variance threshold selector
#[derive(Debug)]
pub struct VarianceThresholdSelector {
    threshold: f32,
}

impl VarianceThresholdSelector {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    fn calculate_variance(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance
    }
}

impl FeatureSelector for VarianceThresholdSelector {
    fn select_features(
        &self,
        features: &[Vec<f64>],
        feature_names: &[String],
    ) -> Result<SelectionResult> {
        if features.is_empty() {
            return Err(anyhow!("No features provided for variance selection"));
        }

        let n_features = features[0].len();
        let mut selected_indices = Vec::new();

        for j in 0..n_features {
            let values: Vec<f64> = features.iter().map(|sample| sample[j]).collect();
            let variance = self.calculate_variance(&values);

            if variance > self.threshold as f64 {
                selected_indices.push(j);
            }
        }

        let selected_names = selected_indices.iter().map(|&i| feature_names[i].clone()).collect();
        let selected_count = selected_indices.len();

        Ok(SelectionResult {
            selected_indices,
            selected_names,
            selection_info: format!(
                "VarianceThreshold: Selected {}/{} features with variance > {}",
                selected_count, n_features, self.threshold
            ),
        })
    }

    fn name(&self) -> &str {
        "VarianceThresholdSelector"
    }
}

/// Correlation-based selector
#[derive(Debug)]
pub struct CorrelationSelector {
    threshold: f64,
}

impl CorrelationSelector {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
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

impl FeatureSelector for CorrelationSelector {
    fn select_features(
        &self,
        features: &[Vec<f64>],
        feature_names: &[String],
    ) -> Result<SelectionResult> {
        if features.is_empty() {
            return Err(anyhow!("No features provided for correlation selection"));
        }

        let n_features = features[0].len();
        let mut selected_indices = vec![0]; // Always keep the first feature
        let mut removed_features = HashSet::new();

        // Calculate correlation matrix and remove highly correlated features
        for i in 1..n_features {
            if removed_features.contains(&i) {
                continue;
            }

            let values_i: Vec<f64> = features.iter().map(|sample| sample[i]).collect();
            let mut should_keep = true;

            for &j in &selected_indices {
                let values_j: Vec<f64> = features.iter().map(|sample| sample[j]).collect();
                let correlation = self.calculate_correlation(&values_i, &values_j);

                if correlation.abs() > self.threshold {
                    should_keep = false;
                    break;
                }
            }

            if should_keep {
                selected_indices.push(i);
            } else {
                removed_features.insert(i);
            }
        }

        let selected_names = selected_indices.iter().map(|&i| feature_names[i].clone()).collect();
        let selected_count = selected_indices.len();

        Ok(SelectionResult {
            selected_indices,
            selected_names,
            selection_info: format!(
                "CorrelationSelector: Selected {}/{} features, removed {} highly correlated",
                selected_count,
                n_features,
                removed_features.len()
            ),
        })
    }

    fn name(&self) -> &str {
        "CorrelationSelector"
    }
}

/// Importance-based selector
#[derive(Debug)]
pub struct ImportanceBasedSelector {
    threshold: f32,
}

impl ImportanceBasedSelector {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl FeatureSelector for ImportanceBasedSelector {
    fn select_features(
        &self,
        features: &[Vec<f64>],
        feature_names: &[String],
    ) -> Result<SelectionResult> {
        // For now, we'll use a simplified importance calculation based on variance
        // In practice, this would use actual feature importance from trained models

        let n_features = features[0].len();
        let mut feature_importance = Vec::new();

        for j in 0..n_features {
            let values: Vec<f64> = features.iter().map(|sample| sample[j]).collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance =
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

            // Use normalized variance as importance proxy
            feature_importance.push((j, variance));
        }

        // Sort by importance (descending)
        feature_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Select features above threshold
        let max_importance = feature_importance[0].1;
        let selected_indices: Vec<usize> = feature_importance
            .iter()
            .filter_map(|(idx, importance)| {
                if importance / max_importance.max(1e-12) > self.threshold as f64 {
                    Some(*idx)
                } else {
                    None
                }
            })
            .collect();

        let selected_names = selected_indices.iter().map(|&i| feature_names[i].clone()).collect();
        let selected_count = selected_indices.len();

        Ok(SelectionResult {
            selected_indices,
            selected_names,
            selection_info: format!(
                "ImportanceBasedSelector: Selected {}/{} features above threshold {}",
                selected_count, n_features, self.threshold
            ),
        })
    }

    fn name(&self) -> &str {
        "ImportanceBasedSelector"
    }
}

// =============================================================================
// SUPPORTING TYPES AND DATA STRUCTURES
// =============================================================================

/// Extracted features from extractors
#[derive(Debug, Clone)]
pub struct ExtractedFeatures {
    /// Feature matrix (samples x features)
    pub features: Vec<Vec<f64>>,
    /// Feature names
    pub names: Vec<String>,
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

/// Transformation result
#[derive(Debug, Clone)]
pub struct TransformationResult {
    /// Transformed feature data
    pub transformed_data: Vec<Vec<f64>>,
    /// Information about transformation
    pub transformation_info: String,
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

#[derive(Debug, Clone)]
struct FeatureStats {
    mean: f64,
    variance: f64,
    min_value: f64,
    max_value: f64,
    missing_count: usize,
    total_count: usize,
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
            missing_count: 0, // Simplified
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

        // Keep only recent history
        if self.importance_history.len() > 100 {
            self.importance_history.remove(0);
        }
    }

    pub fn get_current_importance(&self) -> HashMap<String, f32> {
        self.importance_scores.clone()
    }
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
