//! Unified Quantization Calibration Toolkit
//!
//! This module provides a comprehensive calibration toolkit that unifies all quantization
//! calibration methods across TrustformeRS. It offers:
//! - Unified calibration dataset management
//! - Unified API for different calibration methods
//! - Quality assessment and validation tools
//! - Cross-quantization method calibration comparison
//! - Comprehensive calibration workflow management

#![allow(unused_variables)] // Calibration toolkit

use crate::errors::{file_not_found, invalid_input, runtime_error, TrustformersError};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Unified calibration dataset manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationDataset {
    /// Dataset name
    pub name: String,
    /// Input samples for calibration
    #[serde(skip)]
    pub samples: Vec<Tensor>,
    /// Optional target outputs for supervised calibration
    #[serde(skip)]
    pub targets: Option<Vec<Tensor>>,
    /// Dataset metadata
    pub metadata: CalibrationMetadata,
    /// Statistical properties of the dataset
    pub statistics: DatasetStatistics,
}

/// Metadata for calibration datasets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationMetadata {
    /// Dataset description
    pub description: String,
    /// Dataset source (file path, URL, etc.)
    pub source: String,
    /// Dataset version
    pub version: String,
    /// Creation timestamp
    pub created_at: u64,
    /// Dataset tags for organization
    pub tags: Vec<String>,
    /// Expected model architecture
    pub model_type: String,
    /// Recommended calibration methods
    pub recommended_methods: Vec<CalibrationMethod>,
}

/// Statistical properties of calibration dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStatistics {
    /// Number of samples
    pub sample_count: usize,
    /// Input tensor shapes
    pub input_shapes: Vec<Vec<usize>>,
    /// Statistical moments per dimension
    pub statistics: TensorStatistics,
    /// Dynamic range information
    pub dynamic_range: DynamicRange,
    /// Data distribution characteristics
    pub distribution: DistributionAnalysis,
}

/// Statistical moments for tensor data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorStatistics {
    /// Mean values per dimension
    pub mean: Vec<f32>,
    /// Standard deviation per dimension
    pub std: Vec<f32>,
    /// Minimum values per dimension
    pub min: Vec<f32>,
    /// Maximum values per dimension
    pub max: Vec<f32>,
    /// Percentile values (5th, 25th, 50th, 75th, 95th)
    pub percentiles: Vec<Vec<f32>>,
    /// Skewness measure
    pub skewness: Vec<f32>,
    /// Kurtosis measure
    pub kurtosis: Vec<f32>,
}

/// Dynamic range analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicRange {
    /// Overall dynamic range
    pub overall_range: f32,
    /// Per-channel dynamic ranges
    pub channel_ranges: Vec<f32>,
    /// Outlier detection (values beyond 3 sigma)
    pub outlier_ratio: f32,
    /// Suggested clipping thresholds
    pub suggested_clip_min: f32,
    pub suggested_clip_max: f32,
}

/// Distribution analysis for optimal quantization method selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionAnalysis {
    /// Distribution type (normal, uniform, exponential, multimodal)
    pub distribution_type: DistributionType,
    /// Normality test p-value
    pub normality_p_value: f32,
    /// Distribution entropy
    pub entropy: f32,
    /// Concentration measure (how concentrated the distribution is)
    pub concentration: f32,
    /// Multimodality indicator
    pub is_multimodal: bool,
    /// Number of detected modes for multimodal distributions
    pub mode_count: Option<usize>,
}

/// Distribution types detected in calibration data
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum DistributionType {
    Normal,
    Uniform,
    Exponential,
    Laplace,
    Gamma,
    Beta,
    Multimodal,
    Unknown,
}

/// Available calibration methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Hash, Eq)]
pub enum CalibrationMethod {
    /// Entropy-based calibration (KL divergence)
    Entropy,
    /// Percentile-based calibration
    Percentile,
    /// Mean-squared error minimization
    MSE,
    /// Signal-to-quantization-noise ratio
    SQNR,
    /// Cross-entropy based calibration
    CrossEntropy,
    /// Hessian-based importance (for GPTQ)
    Hessian,
    /// Activation-aware calibration (for AWQ)
    ActivationAware,
    /// Smooth calibration (for SmoothQuant)
    Smooth,
    /// Mixed-bit sensitivity analysis
    SensitivityBased,
    /// Learned quantization optimization
    Learned,
}

/// Calibration configuration for unified API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// Primary calibration method
    pub method: CalibrationMethod,
    /// Fallback methods if primary fails
    pub fallback_methods: Vec<CalibrationMethod>,
    /// Method-specific parameters
    pub parameters: HashMap<String, CalibrationParameter>,
    /// Quality thresholds for validation
    pub quality_thresholds: QualityThresholds,
    /// Cross-validation settings
    pub cross_validation: CrossValidationConfig,
}

/// Parameter values for calibration methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CalibrationParameter {
    Float(f32),
    Int(i32),
    Bool(bool),
    String(String),
    FloatArray(Vec<f32>),
    IntArray(Vec<i32>),
}

/// Quality thresholds for calibration validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Minimum acceptable accuracy retention (0.0-1.0)
    pub min_accuracy_retention: f32,
    /// Maximum acceptable SQNR degradation (dB)
    pub max_sqnr_degradation: f32,
    /// Maximum acceptable KL divergence
    pub max_kl_divergence: f32,
    /// Maximum acceptable inference latency increase
    pub max_latency_increase: f32,
    /// Minimum compression ratio
    pub min_compression_ratio: f32,
}

/// Cross-validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationConfig {
    /// Enable cross-validation
    pub enabled: bool,
    /// Number of folds for cross-validation
    pub folds: usize,
    /// Validation split ratio (0.0-1.0)
    pub validation_split: f32,
    /// Random seed for reproducibility
    pub random_seed: u64,
    /// Stratified sampling for balanced validation
    pub stratified: bool,
}

/// Calibration results from unified API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    /// Used calibration method
    pub method: CalibrationMethod,
    /// Whether primary method succeeded
    pub primary_success: bool,
    /// Calibration parameters found
    pub parameters: CalibrationParameters,
    /// Quality metrics achieved
    pub quality_metrics: QualityMetrics,
    /// Cross-validation results
    pub cross_validation: Option<CrossValidationResults>,
    /// Method comparison results
    pub method_comparison: Option<MethodComparison>,
    /// Recommendations for improvement
    pub recommendations: Vec<CalibrationRecommendation>,
}

/// Calibration parameters for a specific quantization scheme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationParameters {
    /// Quantization scales per layer/channel
    pub scales: HashMap<String, Vec<f32>>,
    /// Zero points per layer/channel
    pub zero_points: HashMap<String, Vec<i32>>,
    /// Clipping ranges per layer/channel
    pub clip_ranges: HashMap<String, (f32, f32)>,
    /// Bit allocations for mixed-bit quantization
    pub bit_allocations: HashMap<String, Vec<u8>>,
    /// Method-specific extra parameters
    pub extra_params: HashMap<String, CalibrationParameter>,
}

/// Quality metrics for calibration assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Accuracy retention (0.0-1.0)
    pub accuracy_retention: f32,
    /// Signal-to-quantization-noise ratio (dB)
    pub sqnr_db: f32,
    /// KL divergence from original distribution
    pub kl_divergence: f32,
    /// Compression ratio achieved
    pub compression_ratio: f32,
    /// Inference speedup factor
    pub speedup_factor: f32,
    /// Memory usage reduction (0.0-1.0)
    pub memory_reduction: f32,
    /// Per-layer quality breakdown
    pub layer_metrics: HashMap<String, LayerQualityMetrics>,
}

/// Quality metrics per layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerQualityMetrics {
    /// Layer name
    pub layer_name: String,
    /// Layer type
    pub layer_type: String,
    /// Quantization error (MSE)
    pub quantization_error: f32,
    /// Output distribution similarity
    pub distribution_similarity: f32,
    /// Gradient flow preservation
    pub gradient_preservation: f32,
    /// Activation pattern preservation
    pub activation_preservation: f32,
}

/// Cross-validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationResults {
    /// Mean quality metrics across folds
    pub mean_metrics: QualityMetrics,
    /// Standard deviation of metrics across folds
    pub std_metrics: QualityMetrics,
    /// Per-fold results
    pub fold_results: Vec<QualityMetrics>,
    /// Cross-validation score (0.0-1.0)
    pub cv_score: f32,
    /// Stability indicator (lower is more stable)
    pub stability_score: f32,
}

/// Comparison between different calibration methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodComparison {
    /// Results per method
    pub method_results: HashMap<CalibrationMethod, QualityMetrics>,
    /// Ranking of methods by overall quality
    pub method_ranking: Vec<(CalibrationMethod, f32)>,
    /// Best method recommendation
    pub recommended_method: CalibrationMethod,
    /// Trade-off analysis
    pub trade_offs: Vec<TradeOffAnalysis>,
}

/// Trade-off analysis between different aspects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeOffAnalysis {
    /// Method being analyzed
    pub method: CalibrationMethod,
    /// Accuracy vs compression trade-off
    pub accuracy_compression: f32,
    /// Speed vs quality trade-off
    pub speed_quality: f32,
    /// Memory vs accuracy trade-off
    pub memory_accuracy: f32,
    /// Overall balance score
    pub balance_score: f32,
}

/// Calibration recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Recommendation description
    pub description: String,
    /// Expected improvement
    pub expected_improvement: f32,
    /// Implementation difficulty (1-5)
    pub difficulty: u8,
    /// Priority level (1-5)
    pub priority: u8,
    /// Actionable steps
    pub action_steps: Vec<String>,
}

/// Types of calibration recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Increase calibration dataset size
    IncreaseDataset,
    /// Try different calibration method
    TryDifferentMethod,
    /// Adjust quantization bit width
    AdjustBitWidth,
    /// Use mixed-bit quantization
    UseMixedBit,
    /// Apply outlier clipping
    ApplyClipping,
    /// Use different quantization granularity
    ChangeGranularity,
    /// Apply calibration dataset preprocessing
    PreprocessDataset,
    /// Use ensemble calibration
    UseEnsemble,
    /// Apply post-quantization fine-tuning
    PostQuantTuning,
    /// Optimize for specific hardware
    OptimizeHardware,
}

/// Main unified calibration toolkit
pub struct CalibrationToolkit {
    /// Registered datasets
    datasets: HashMap<String, CalibrationDataset>,
    /// Calibration configurations
    #[allow(dead_code)]
    configs: HashMap<String, CalibrationConfig>,
    /// Calibration history
    history: Vec<CalibrationResult>,
    /// Performance cache for repeated calibrations
    cache: HashMap<String, CalibrationResult>,
}

impl CalibrationToolkit {
    /// Create a new calibration toolkit
    pub fn new() -> Self {
        Self {
            datasets: HashMap::new(),
            configs: HashMap::new(),
            history: Vec::new(),
            cache: HashMap::new(),
        }
    }

    /// Register a calibration dataset
    pub fn register_dataset(
        &mut self,
        dataset: CalibrationDataset,
    ) -> Result<(), TrustformersError> {
        // Validate dataset
        self.validate_dataset(&dataset)?;

        // Calculate statistics if not provided
        let mut dataset = dataset;
        if dataset.statistics.sample_count == 0 {
            dataset.statistics = self.calculate_dataset_statistics(&dataset.samples)?;
        }

        self.datasets.insert(dataset.name.clone(), dataset);
        Ok(())
    }

    /// Create calibration dataset from tensor samples
    pub fn create_dataset(
        &self,
        name: String,
        samples: Vec<Tensor>,
        metadata: CalibrationMetadata,
    ) -> Result<CalibrationDataset, TrustformersError> {
        let statistics = self.calculate_dataset_statistics(&samples)?;

        Ok(CalibrationDataset {
            name,
            samples,
            targets: None,
            metadata,
            statistics,
        })
    }

    /// Load calibration dataset from file
    pub fn load_dataset<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> Result<CalibrationDataset, TrustformersError> {
        use std::fs;

        let path = path.as_ref();
        let contents = fs::read_to_string(path).map_err(|e| runtime_error(e.to_string()))?;

        // Parse JSON format
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            let dataset: CalibrationDataset = serde_json::from_str(&contents)
                .map_err(|e| runtime_error(format!("Failed to parse JSON dataset: {}", e)))?;

            // Add to managed datasets
            self.datasets.insert(dataset.name.clone(), dataset.clone());

            Ok(dataset)
        } else {
            Err(invalid_input(format!(
                "Unsupported dataset format: {:?}. Only JSON (.json) is currently supported.",
                path.extension()
            )))
        }
    }

    /// Save calibration dataset to file
    pub fn save_dataset<P: AsRef<Path>>(
        &self,
        dataset: &CalibrationDataset,
        path: P,
    ) -> Result<(), TrustformersError> {
        use std::fs;

        let path = path.as_ref();

        // Support JSON format
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            let json_content = serde_json::to_string_pretty(dataset).map_err(|e| {
                runtime_error(format!("Failed to serialize dataset to JSON: {}", e))
            })?;

            fs::write(path, json_content).map_err(|e| runtime_error(e.to_string()))?;

            Ok(())
        } else {
            Err(invalid_input(format!(
                "Unsupported dataset format: {:?}. Only JSON (.json) is currently supported.",
                path.extension()
            )))
        }
    }

    /// Run unified calibration with automatic method selection
    pub fn calibrate(
        &mut self,
        dataset_name: &str,
        config: CalibrationConfig,
    ) -> Result<CalibrationResult, TrustformersError> {
        let dataset = self
            .datasets
            .get(dataset_name)
            .ok_or_else(|| file_not_found(format!("Dataset '{}' not found", dataset_name)))?;

        // Check cache first
        let cache_key = self.generate_cache_key(dataset_name, &config);
        if let Some(cached_result) = self.cache.get(&cache_key) {
            return Ok(cached_result.clone());
        }

        // Run calibration
        let result = self.run_calibration(dataset, &config)?;

        // Cache result
        self.cache.insert(cache_key, result.clone());

        // Add to history
        self.history.push(result.clone());

        Ok(result)
    }

    /// Compare multiple calibration methods
    pub fn compare_methods(
        &mut self,
        dataset_name: &str,
        methods: Vec<CalibrationMethod>,
    ) -> Result<MethodComparison, TrustformersError> {
        let mut method_results = HashMap::new();

        for method in &methods {
            let config = CalibrationConfig {
                method: *method,
                fallback_methods: Vec::new(),
                parameters: self.get_default_parameters(*method),
                quality_thresholds: QualityThresholds::default(),
                cross_validation: CrossValidationConfig::default(),
            };

            let result = self.calibrate(dataset_name, config)?;
            method_results.insert(*method, result.quality_metrics);
        }

        // Rank methods by overall quality score
        let mut method_ranking: Vec<_> = method_results
            .iter()
            .map(|(method, metrics)| (*method, self.calculate_overall_score(metrics)))
            .collect();
        method_ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("Partial comparison failed"));

        let recommended_method = method_ranking[0].0;

        // Generate trade-off analysis
        let trade_offs = methods
            .iter()
            .map(|method| self.analyze_trade_offs(*method, &method_results[method]))
            .collect();

        Ok(MethodComparison {
            method_results,
            method_ranking,
            recommended_method,
            trade_offs,
        })
    }

    /// Validate calibration quality and provide recommendations
    pub fn validate_calibration(
        &self,
        result: &CalibrationResult,
        thresholds: &QualityThresholds,
    ) -> Vec<CalibrationRecommendation> {
        let mut recommendations = Vec::new();

        // Check accuracy retention
        if result.quality_metrics.accuracy_retention < thresholds.min_accuracy_retention {
            recommendations.push(CalibrationRecommendation {
                recommendation_type: RecommendationType::TryDifferentMethod,
                description: format!(
                    "Accuracy retention {:.3} is below threshold {:.3}. Consider using a different calibration method or increasing bit width.",
                    result.quality_metrics.accuracy_retention,
                    thresholds.min_accuracy_retention
                ),
                expected_improvement: 0.1,
                difficulty: 2,
                priority: 5,
                action_steps: vec![
                    "Try entropy-based calibration".to_string(),
                    "Increase quantization bit width".to_string(),
                    "Use mixed-bit quantization for critical layers".to_string(),
                ],
            });
        }

        // Check compression ratio
        if result.quality_metrics.compression_ratio < thresholds.min_compression_ratio {
            recommendations.push(CalibrationRecommendation {
                recommendation_type: RecommendationType::UseMixedBit,
                description: format!(
                    "Compression ratio {:.2}x is below target {:.2}x. Consider using more aggressive quantization.",
                    result.quality_metrics.compression_ratio,
                    thresholds.min_compression_ratio
                ),
                expected_improvement: 0.2,
                difficulty: 3,
                priority: 3,
                action_steps: vec![
                    "Enable mixed-bit quantization".to_string(),
                    "Reduce bit width for less critical layers".to_string(),
                    "Apply weight pruning before quantization".to_string(),
                ],
            });
        }

        // Check SQNR degradation
        if result.quality_metrics.sqnr_db < -thresholds.max_sqnr_degradation {
            recommendations.push(CalibrationRecommendation {
                recommendation_type: RecommendationType::ApplyClipping,
                description: format!(
                    "SQNR degradation {:.2} dB exceeds threshold {:.2} dB. Apply outlier clipping or increase calibration data.",
                    result.quality_metrics.sqnr_db,
                    thresholds.max_sqnr_degradation
                ),
                expected_improvement: 0.15,
                difficulty: 2,
                priority: 4,
                action_steps: vec![
                    "Apply percentile-based outlier clipping".to_string(),
                    "Increase calibration dataset size".to_string(),
                    "Use more representative calibration data".to_string(),
                ],
            });
        }

        recommendations
    }

    /// Generate comprehensive calibration report
    pub fn generate_report(
        &self,
        result: &CalibrationResult,
        dataset_name: &str,
    ) -> CalibrationReport {
        let dataset = self.datasets.get(dataset_name);

        CalibrationReport {
            dataset_name: dataset_name.to_string(),
            dataset_info: dataset.map(|d| d.metadata.clone()),
            calibration_result: result.clone(),
            recommendations: self.validate_calibration(result, &QualityThresholds::default()),
            generated_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs(),
        }
    }

    // Private helper methods
    fn validate_dataset(&self, dataset: &CalibrationDataset) -> Result<(), TrustformersError> {
        if dataset.samples.is_empty() {
            return Err(invalid_input("Dataset cannot be empty".to_string()));
        }

        // Check tensor shape consistency
        let first_shape = dataset.samples[0].shape();
        for (i, sample) in dataset.samples.iter().enumerate() {
            if sample.shape() != first_shape {
                return Err(invalid_input(format!(
                    "Sample {} has inconsistent shape",
                    i
                )));
            }
        }

        Ok(())
    }

    fn calculate_dataset_statistics(
        &self,
        samples: &[Tensor],
    ) -> Result<DatasetStatistics, TrustformersError> {
        if samples.is_empty() {
            return Err(invalid_input(
                "Cannot calculate statistics for empty dataset".to_string(),
            ));
        }

        let sample_count = samples.len();
        let input_shapes = vec![samples[0].shape().to_vec()];

        // Calculate basic statistics (placeholder implementation)
        let dim_count = samples[0].len();
        let statistics = TensorStatistics {
            mean: vec![0.0; dim_count],
            std: vec![1.0; dim_count],
            min: vec![-1.0; dim_count],
            max: vec![1.0; dim_count],
            percentiles: vec![vec![0.0; 5]; dim_count],
            skewness: vec![0.0; dim_count],
            kurtosis: vec![3.0; dim_count],
        };

        let dynamic_range = DynamicRange {
            overall_range: 2.0,
            channel_ranges: vec![2.0; dim_count],
            outlier_ratio: 0.05,
            suggested_clip_min: -1.0,
            suggested_clip_max: 1.0,
        };

        let distribution = DistributionAnalysis {
            distribution_type: DistributionType::Normal,
            normality_p_value: 0.5,
            entropy: 3.0,
            concentration: 0.5,
            is_multimodal: false,
            mode_count: Some(1),
        };

        Ok(DatasetStatistics {
            sample_count,
            input_shapes,
            statistics,
            dynamic_range,
            distribution,
        })
    }

    fn generate_cache_key(&self, dataset_name: &str, config: &CalibrationConfig) -> String {
        // Generate a hash-based cache key from dataset name and config
        format!("{}_{:?}", dataset_name, config.method)
    }

    fn run_calibration(
        &self,
        dataset: &CalibrationDataset,
        config: &CalibrationConfig,
    ) -> Result<CalibrationResult, TrustformersError> {
        // Placeholder implementation - would integrate with specific calibration methods
        let parameters = CalibrationParameters {
            scales: HashMap::new(),
            zero_points: HashMap::new(),
            clip_ranges: HashMap::new(),
            bit_allocations: HashMap::new(),
            extra_params: HashMap::new(),
        };

        let quality_metrics = QualityMetrics {
            accuracy_retention: 0.95,
            sqnr_db: 40.0,
            kl_divergence: 0.01,
            compression_ratio: 4.0,
            speedup_factor: 2.0,
            memory_reduction: 0.75,
            layer_metrics: HashMap::new(),
        };

        Ok(CalibrationResult {
            method: config.method,
            primary_success: true,
            parameters,
            quality_metrics,
            cross_validation: None,
            method_comparison: None,
            recommendations: Vec::new(),
        })
    }

    fn get_default_parameters(
        &self,
        method: CalibrationMethod,
    ) -> HashMap<String, CalibrationParameter> {
        let mut params = HashMap::new();

        match method {
            CalibrationMethod::Entropy => {
                params.insert("num_bins".to_string(), CalibrationParameter::Int(2048));
                params.insert(
                    "divergence_threshold".to_string(),
                    CalibrationParameter::Float(0.01),
                );
            },
            CalibrationMethod::Percentile => {
                params.insert("percentile".to_string(), CalibrationParameter::Float(99.99));
                params.insert("symmetric".to_string(), CalibrationParameter::Bool(true));
            },
            CalibrationMethod::MSE => {
                params.insert(
                    "learning_rate".to_string(),
                    CalibrationParameter::Float(0.001),
                );
                params.insert(
                    "max_iterations".to_string(),
                    CalibrationParameter::Int(1000),
                );
            },
            _ => {
                // Default parameters for other methods
                params.insert("tolerance".to_string(), CalibrationParameter::Float(1e-6));
            },
        }

        params
    }

    fn calculate_overall_score(&self, metrics: &QualityMetrics) -> f32 {
        // Weighted combination of different metrics
        0.4 * metrics.accuracy_retention
            + 0.2 * (metrics.sqnr_db / 50.0).min(1.0)
            + 0.2 * (metrics.compression_ratio / 8.0).min(1.0)
            + 0.1 * metrics.speedup_factor / 4.0
            + 0.1 * metrics.memory_reduction
    }

    fn analyze_trade_offs(
        &self,
        method: CalibrationMethod,
        metrics: &QualityMetrics,
    ) -> TradeOffAnalysis {
        TradeOffAnalysis {
            method,
            accuracy_compression: metrics.accuracy_retention / (metrics.compression_ratio / 4.0),
            speed_quality: metrics.speedup_factor / 4.0 * metrics.accuracy_retention,
            memory_accuracy: metrics.memory_reduction * metrics.accuracy_retention,
            balance_score: self.calculate_overall_score(metrics),
        }
    }
}

/// Comprehensive calibration report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationReport {
    /// Dataset name used for calibration
    pub dataset_name: String,
    /// Dataset metadata
    pub dataset_info: Option<CalibrationMetadata>,
    /// Calibration results
    pub calibration_result: CalibrationResult,
    /// Recommendations for improvement
    pub recommendations: Vec<CalibrationRecommendation>,
    /// Report generation timestamp
    pub generated_at: u64,
}

// Default implementations
impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_accuracy_retention: 0.95,
            max_sqnr_degradation: 5.0,
            max_kl_divergence: 0.1,
            max_latency_increase: 0.1,
            min_compression_ratio: 2.0,
        }
    }
}

impl Default for CrossValidationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            folds: 5,
            validation_split: 0.2,
            random_seed: 42,
            stratified: false,
        }
    }
}

impl Default for CalibrationToolkit {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_toolkit_creation() {
        let toolkit = CalibrationToolkit::new();
        assert!(toolkit.datasets.is_empty());
        assert!(toolkit.configs.is_empty());
        assert!(toolkit.history.is_empty());
    }

    #[test]
    fn test_dataset_validation() {
        let toolkit = CalibrationToolkit::new();

        // Test empty dataset validation
        let empty_dataset = CalibrationDataset {
            name: "empty".to_string(),
            samples: Vec::new(),
            targets: None,
            metadata: CalibrationMetadata {
                description: "Empty dataset".to_string(),
                source: "test".to_string(),
                version: "1.0".to_string(),
                created_at: 0,
                tags: Vec::new(),
                model_type: "test".to_string(),
                recommended_methods: Vec::new(),
            },
            statistics: DatasetStatistics {
                sample_count: 0,
                input_shapes: Vec::new(),
                statistics: TensorStatistics {
                    mean: Vec::new(),
                    std: Vec::new(),
                    min: Vec::new(),
                    max: Vec::new(),
                    percentiles: Vec::new(),
                    skewness: Vec::new(),
                    kurtosis: Vec::new(),
                },
                dynamic_range: DynamicRange {
                    overall_range: 0.0,
                    channel_ranges: Vec::new(),
                    outlier_ratio: 0.0,
                    suggested_clip_min: 0.0,
                    suggested_clip_max: 0.0,
                },
                distribution: DistributionAnalysis {
                    distribution_type: DistributionType::Unknown,
                    normality_p_value: 0.0,
                    entropy: 0.0,
                    concentration: 0.0,
                    is_multimodal: false,
                    mode_count: None,
                },
            },
        };

        assert!(toolkit.validate_dataset(&empty_dataset).is_err());
    }

    #[test]
    fn test_quality_thresholds_default() {
        let thresholds = QualityThresholds::default();
        assert_eq!(thresholds.min_accuracy_retention, 0.95);
        assert_eq!(thresholds.max_sqnr_degradation, 5.0);
        assert_eq!(thresholds.min_compression_ratio, 2.0);
    }

    #[test]
    fn test_calibration_method_enum() {
        let method = CalibrationMethod::Entropy;
        assert_eq!(method, CalibrationMethod::Entropy);

        let serialized = serde_json::to_string(&method).expect("JSON serialization failed");
        let deserialized: CalibrationMethod =
            serde_json::from_str(&serialized).expect("JSON deserialization failed");
        assert_eq!(method, deserialized);
    }
}
