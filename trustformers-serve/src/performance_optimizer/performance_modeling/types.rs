//! Core type definitions for performance modeling
//!
//! This module provides comprehensive type definitions for machine learning-based
//! performance modeling, including model configurations, predictions, validation
//! results, and all supporting data structures used throughout the system.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

use crate::performance_optimizer::types::{PerformanceDataPoint, SystemState, TestCharacteristics};

// =============================================================================
// CORE MODEL TYPES
// =============================================================================

/// Performance prediction result with confidence metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePrediction {
    /// Predicted throughput value
    pub throughput: f64,
    /// Predicted latency
    pub latency: Duration,
    /// Prediction confidence (0.0 to 1.0)
    pub confidence: f32,
    /// Uncertainty bounds
    pub uncertainty_bounds: (f64, f64),
    /// Model used for prediction
    pub model_name: String,
    /// Feature contributions
    pub feature_importance: HashMap<String, f32>,
    /// Prediction timestamp
    pub predicted_at: DateTime<Utc>,
}

/// Model accuracy metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAccuracyMetrics {
    /// Overall model accuracy
    pub overall_accuracy: f32,
    /// R-squared coefficient
    pub r_squared: f32,
    /// Mean absolute error
    pub mean_absolute_error: f32,
    /// Root mean squared error
    pub root_mean_squared_error: f32,
    /// Cross-validation scores
    pub cross_validation_scores: Vec<f32>,
    /// Confidence interval
    pub confidence_interval: (f32, f32),
    /// Prediction stability metric
    pub prediction_stability: f32,
    /// Last validation timestamp
    pub last_validated: DateTime<Utc>,
}

/// Training configuration for performance models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTrainingConfig {
    /// Learning rate for adaptive models
    pub learning_rate: f32,
    /// Maximum number of training epochs
    pub max_epochs: usize,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Validation split ratio
    pub validation_split: f32,
    /// Batch size for neural networks
    pub batch_size: usize,
    /// Regularization strength
    pub regularization: f32,
    /// Enable feature normalization
    pub normalize_features: bool,
    /// Enable cross-validation
    pub enable_cross_validation: bool,
    /// Number of cross-validation folds
    pub cv_folds: usize,
}

impl Default for ModelTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            max_epochs: 1000,
            early_stopping_patience: 10,
            validation_split: 0.2,
            batch_size: 32,
            regularization: 0.001,
            normalize_features: true,
            enable_cross_validation: true,
            cv_folds: 5,
        }
    }
}

/// Feature engineering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEngineeringConfig {
    /// Enable polynomial features
    pub enable_polynomial_features: bool,
    /// Polynomial degree
    pub polynomial_degree: usize,
    /// Enable interaction features
    pub enable_interactions: bool,
    /// Enable logarithmic transformations
    pub enable_log_transforms: bool,
    /// Enable temporal features
    pub enable_temporal_features: bool,
    /// Feature scaling method
    pub scaling_method: FeatureScalingMethod,
    /// Feature selection threshold
    pub selection_threshold: f32,
    /// Maximum number of features
    pub max_features: Option<usize>,
}

impl Default for FeatureEngineeringConfig {
    fn default() -> Self {
        Self {
            enable_polynomial_features: true,
            polynomial_degree: 2,
            enable_interactions: true,
            enable_log_transforms: true,
            enable_temporal_features: true,
            scaling_method: FeatureScalingMethod::StandardScaling,
            selection_threshold: 0.01,
            max_features: Some(100),
        }
    }
}

/// Feature scaling methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureScalingMethod {
    /// No scaling
    None,
    /// Min-max scaling to [0, 1]
    MinMaxScaling,
    /// Standard scaling (z-score)
    StandardScaling,
    /// Robust scaling using median and IQR
    RobustScaling,
}

/// Validation strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Validation strategy type
    pub strategy: ValidationStrategyType,
    /// Test set size ratio
    pub test_size: f32,
    /// Number of cross-validation folds
    pub cv_folds: usize,
    /// Enable time series validation
    pub time_series_validation: bool,
    /// Validation metrics to compute
    pub metrics: Vec<ValidationMetric>,
    /// Minimum samples for validation
    pub min_validation_samples: usize,
    pub minimum_accuracy: f64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            strategy: ValidationStrategyType::CrossValidation,
            test_size: 0.2,
            cv_folds: 5,
            time_series_validation: true,
            metrics: vec![
                ValidationMetric::MeanAbsoluteError,
                ValidationMetric::RootMeanSquaredError,
                ValidationMetric::RSquared,
                ValidationMetric::MeanAbsolutePercentageError,
            ],
            min_validation_samples: 10,
            minimum_accuracy: 0.8,
        }
    }
}

/// Validation strategy types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValidationStrategyType {
    /// Hold-out validation
    HoldOut,
    /// K-fold cross-validation
    CrossValidation,
    /// Time series validation
    TimeSeries,
    /// Bootstrap validation
    Bootstrap,
    /// Leave-one-out validation
    LeaveOneOut,
}

/// Validation metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValidationMetric {
    /// Mean absolute error
    MeanAbsoluteError,
    /// Root mean squared error
    RootMeanSquaredError,
    /// R-squared coefficient
    RSquared,
    /// Mean absolute percentage error
    MeanAbsolutePercentageError,
    /// Mean squared logarithmic error
    MeanSquaredLogarithmicError,
    /// Explained variance score
    ExplainedVarianceScore,
}

// =============================================================================
// ADAPTIVE LEARNING TYPES
// =============================================================================

/// Adaptive learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLearningConfig {
    /// Enable online learning
    pub enable_online_learning: bool,
    /// Learning rate decay
    pub learning_rate_decay: f32,
    /// Adaptation window size
    pub adaptation_window: usize,
    /// Concept drift detection threshold
    pub drift_threshold: f32,
    /// Model update frequency
    pub update_frequency: Duration,
    /// Minimum samples for adaptation
    pub min_adaptation_samples: usize,
    /// Enable active learning
    pub enable_active_learning: bool,
    /// Uncertainty sampling threshold
    pub uncertainty_threshold: f32,
}

impl Default for AdaptiveLearningConfig {
    fn default() -> Self {
        Self {
            enable_online_learning: true,
            learning_rate_decay: 0.995,
            adaptation_window: 100,
            drift_threshold: 0.1,
            update_frequency: Duration::from_secs(300), // 5 minutes
            min_adaptation_samples: 10,
            enable_active_learning: true,
            uncertainty_threshold: 0.3,
        }
    }
}

/// Learning update information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningUpdate {
    /// Update type
    pub update_type: LearningUpdateType,
    /// Performance impact
    pub performance_impact: f32,
    /// Confidence change
    pub confidence_delta: f32,
    /// Learning metrics
    pub learning_metrics: LearningMetrics,
    /// Update timestamp
    pub updated_at: DateTime<Utc>,
}

/// Types of learning updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningUpdateType {
    /// Incremental parameter update
    Incremental,
    /// Full model retrain
    FullRetrain,
    /// Ensemble model update
    EnsembleUpdate,
    /// Concept drift adaptation
    ConceptDriftAdaptation,
    /// Active learning incorporation
    ActiveLearning,
}

/// Learning performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningMetrics {
    /// Learning rate used
    pub learning_rate: f32,
    /// Gradient norm
    pub gradient_norm: f32,
    /// Loss reduction
    pub loss_reduction: f32,
    /// Convergence indicator
    pub convergence_score: f32,
    /// Training time
    pub training_time: Duration,
    /// Memory usage
    pub memory_usage_mb: f32,
}

/// Model state tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStateSnapshot {
    /// Model parameters checksum
    pub parameters_checksum: String,
    /// Training data size
    pub training_data_size: usize,
    /// Model accuracy
    pub current_accuracy: f32,
    /// Last training time
    pub last_trained: DateTime<Utc>,
    /// Number of updates
    pub update_count: u64,
    /// Model version
    pub version: String,
}

// =============================================================================
// PREDICTION ENGINE TYPES
// =============================================================================

/// Prediction request configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRequest {
    /// Target parallelism levels
    pub parallelism_levels: Vec<usize>,
    /// Test characteristics
    pub test_characteristics: TestCharacteristics,
    /// Current system state
    pub system_state: SystemState,
    /// Prediction horizon
    pub prediction_horizon: Option<Duration>,
    /// Required confidence level
    pub confidence_level: f32,
    /// Include uncertainty analysis
    pub include_uncertainty: bool,
}

/// Batch prediction results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchPredictionResult {
    /// Individual predictions
    pub predictions: Vec<PerformancePrediction>,
    /// Batch statistics
    pub batch_statistics: BatchStatistics,
    /// Processing time
    pub processing_time: Duration,
    /// Model ensemble info
    pub ensemble_info: Option<EnsembleInfo>,
}

/// Batch prediction statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStatistics {
    /// Average prediction confidence
    pub average_confidence: f32,
    /// Prediction variance
    pub prediction_variance: f32,
    /// Optimal parallelism estimate
    pub optimal_parallelism: usize,
    /// Performance gain estimate
    pub estimated_performance_gain: f32,
}

/// Ensemble model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleInfo {
    /// Individual model contributions
    pub model_weights: HashMap<String, f32>,
    /// Ensemble method used
    pub ensemble_method: EnsembleMethod,
    /// Diversity score
    pub diversity_score: f32,
    /// Consensus level
    pub consensus_level: f32,
}

/// Ensemble methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnsembleMethod {
    /// Simple averaging
    SimpleAverage,
    /// Weighted averaging
    WeightedAverage,
    /// Stacking
    Stacking,
    /// Boosting
    Boosting,
    /// Bagging
    Bagging,
}

// =============================================================================
// MODEL FACTORY TYPES
// =============================================================================

/// Model factory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFactoryConfig {
    /// Available model types
    pub available_models: Vec<ModelTypeConfig>,
    /// Default model selection strategy
    pub selection_strategy: ModelSelectionStrategy,
    /// Auto-tuning configuration
    pub auto_tuning: AutoTuningConfig,
    /// Model caching settings
    pub caching: ModelCachingConfig,
}

/// Model type configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTypeConfig {
    /// Model type identifier
    pub model_type: String,
    /// Model-specific parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Performance characteristics
    pub performance_profile: ModelPerformanceProfile,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
}

impl ModelTypeConfig {
    /// Create a linear regression model configuration
    pub fn linear_regression() -> Self {
        Self {
            model_type: "linear_regression".to_string(),
            parameters: HashMap::new(),
            performance_profile: ModelPerformanceProfile {
                training_complexity: ComplexityClass::Linear,
                prediction_complexity: ComplexityClass::Constant,
                memory_complexity: ComplexityClass::Linear,
                accuracy_range: (0.7, 0.9),
                convergence_profile: ConvergenceProfile {
                    typical_convergence_epochs: 100,
                    stability_score: 0.9,
                    early_stopping_effectiveness: 0.8,
                },
            },
            resource_requirements: ResourceRequirements {
                min_memory_mb: 64,
                cpu_utilization: 0.5,
                gpu_requirement: GpuRequirement::None,
                disk_space_mb: 10,
            },
        }
    }

    /// Create a polynomial regression model configuration
    pub fn polynomial_regression() -> Self {
        Self {
            model_type: "polynomial_regression".to_string(),
            parameters: HashMap::new(),
            performance_profile: ModelPerformanceProfile {
                training_complexity: ComplexityClass::Quadratic,
                prediction_complexity: ComplexityClass::Constant,
                memory_complexity: ComplexityClass::Quadratic,
                accuracy_range: (0.75, 0.95),
                convergence_profile: ConvergenceProfile {
                    typical_convergence_epochs: 150,
                    stability_score: 0.85,
                    early_stopping_effectiveness: 0.75,
                },
            },
            resource_requirements: ResourceRequirements {
                min_memory_mb: 128,
                cpu_utilization: 0.7,
                gpu_requirement: GpuRequirement::Optional,
                disk_space_mb: 20,
            },
        }
    }
}

/// Model selection strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelSelectionStrategy {
    /// Select best performing model
    BestPerformance,
    /// Select fastest model
    FastestTraining,
    /// Balance performance and speed
    Balanced,
    /// Use ensemble of top models
    Ensemble,
    /// Custom selection logic
    Custom(String),
}

/// Auto-tuning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoTuningConfig {
    /// Enable hyperparameter tuning
    pub enable_hyperparameter_tuning: bool,
    /// Tuning algorithm
    pub tuning_algorithm: TuningAlgorithm,
    /// Maximum tuning iterations
    pub max_iterations: usize,
    /// Tuning timeout
    pub timeout: Duration,
    /// Parallel tuning jobs
    pub parallel_jobs: usize,
}

/// Hyperparameter tuning algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TuningAlgorithm {
    /// Grid search
    GridSearch,
    /// Random search
    RandomSearch,
    /// Bayesian optimization
    BayesianOptimization,
    /// Genetic algorithm
    GeneticAlgorithm,
    /// Tree-structured Parzen Estimator
    TreeStructuredParzenEstimator,
}

/// Model caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCachingConfig {
    /// Enable model caching
    pub enable_caching: bool,
    /// Cache size limit
    pub cache_size_mb: usize,
    /// Cache TTL
    pub cache_ttl: Duration,
    /// Eviction policy
    pub eviction_policy: CacheEvictionPolicy,
}

/// Cache eviction policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheEvictionPolicy {
    /// Least recently used
    LeastRecentlyUsed,
    /// Least frequently used
    LeastFrequentlyUsed,
    /// Time to live
    TimeToLive,
    /// Random eviction
    Random,
}

/// Model performance profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceProfile {
    /// Training time complexity
    pub training_complexity: ComplexityClass,
    /// Prediction time complexity
    pub prediction_complexity: ComplexityClass,
    /// Memory complexity
    pub memory_complexity: ComplexityClass,
    /// Typical accuracy range
    pub accuracy_range: (f32, f32),
    /// Convergence characteristics
    pub convergence_profile: ConvergenceProfile,
}

/// Computational complexity classes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityClass {
    /// Constant time O(1)
    Constant,
    /// Linear time O(n)
    Linear,
    /// Log-linear time O(n log n)
    LogLinear,
    /// Quadratic time O(n²)
    Quadratic,
    /// Polynomial time O(n^k)
    Polynomial(usize),
    /// Exponential time O(2^n)
    Exponential,
}

/// Convergence profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceProfile {
    /// Typical convergence time
    pub typical_convergence_epochs: usize,
    /// Convergence stability
    pub stability_score: f32,
    /// Early stopping effectiveness
    pub early_stopping_effectiveness: f32,
}

/// Resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Minimum RAM requirement (MB)
    pub min_memory_mb: usize,
    /// CPU core utilization
    pub cpu_utilization: f32,
    /// GPU requirement
    pub gpu_requirement: GpuRequirement,
    /// Disk space requirement (MB)
    pub disk_space_mb: usize,
}

/// GPU requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuRequirement {
    /// No GPU required
    None,
    /// Optional GPU acceleration
    Optional,
    /// GPU required
    Required,
    /// Specific GPU memory requirement (MB)
    SpecificMemory(usize),
}

// =============================================================================
// TRAIT DEFINITIONS
// =============================================================================

/// Trait for performance prediction models
pub trait PerformancePredictor: std::fmt::Debug + Send + Sync {
    /// Make performance prediction
    fn predict(&self, request: &PredictionRequest) -> Result<PerformancePrediction>;

    /// Get model accuracy metrics
    fn get_accuracy(&self) -> ModelAccuracyMetrics;

    /// Get model name
    fn name(&self) -> &str;

    /// Check if model supports online learning
    fn supports_online_learning(&self) -> bool;
}

/// Trait for adaptive learning models
pub trait AdaptiveLearner: std::fmt::Debug + Send + Sync {
    /// Update model with new data point
    fn update_with_data(&mut self, data_point: &PerformanceDataPoint) -> Result<LearningUpdate>;

    /// Detect concept drift
    fn detect_concept_drift(&self, recent_data: &[PerformanceDataPoint]) -> Result<bool>;

    /// Adapt to concept drift
    fn adapt_to_drift(&mut self, adaptation_data: &[PerformanceDataPoint]) -> Result<()>;

    /// Get learning configuration
    fn get_learning_config(&self) -> &AdaptiveLearningConfig;
}

/// Trait for model validation
pub trait ModelValidator: std::fmt::Debug + Send + Sync {
    /// Validate model performance
    fn validate(
        &self,
        model: &dyn PerformancePredictor,
        test_data: &[PerformanceDataPoint],
    ) -> Result<ValidationResult>;

    /// Get validation strategy
    fn strategy(&self) -> ValidationStrategyType;

    /// Get validation configuration
    fn config(&self) -> &ValidationConfig;
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Validation metrics
    pub metrics: HashMap<ValidationMetric, f32>,
    /// Cross-validation scores
    pub cv_scores: Vec<f32>,
    /// Validation confidence
    pub confidence: f32,
    /// Validation details
    pub details: ValidationDetails,
    /// Validation timestamp
    pub validated_at: DateTime<Utc>,
}

/// Detailed validation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationDetails {
    /// Number of test samples
    pub test_samples: usize,
    /// Test data statistics
    pub test_statistics: TestDataStatistics,
    /// Prediction errors
    pub prediction_errors: Vec<f32>,
    /// Residual analysis
    pub residual_analysis: ResidualAnalysis,
}

/// Test data statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestDataStatistics {
    /// Mean target value
    pub mean_target: f32,
    /// Target standard deviation
    pub target_std: f32,
    /// Feature correlations
    pub feature_correlations: HashMap<String, f32>,
    /// Data distribution info
    pub distribution_info: DistributionInfo,
}

/// Distribution information
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DistributionInfo {
    /// Distribution type
    pub distribution_type: DistributionType,
    /// Distribution parameters
    pub parameters: HashMap<String, f32>,
    /// Normality test p-value
    pub normality_p_value: f32,
}

/// Distribution types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionType {
    /// Normal distribution
    Normal,
    /// Log-normal distribution
    LogNormal,
    /// Exponential distribution
    Exponential,
    /// Uniform distribution
    Uniform,
    /// Custom distribution
    Custom(String),
}

impl Default for DistributionType {
    fn default() -> Self {
        DistributionType::Normal
    }
}

/// Residual analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualAnalysis {
    /// Residual autocorrelation
    pub autocorrelation: f32,
    /// Heteroscedasticity test p-value
    pub heteroscedasticity_p_value: f32,
    /// Residual normality test p-value
    pub normality_p_value: f32,
    /// Outlier detection results
    pub outliers: Vec<usize>,
}

/// Trait for feature engineering
pub trait FeatureEngineer: std::fmt::Debug + Send + Sync {
    /// Transform raw features
    fn transform_features(&self, raw_features: &[f64]) -> Result<Vec<f64>>;

    /// Get feature names
    fn feature_names(&self) -> Vec<String>;

    /// Get feature importance
    fn feature_importance(&self) -> HashMap<String, f32>;

    /// Update feature engineering based on data
    fn update_from_data(&mut self, data: &[PerformanceDataPoint]) -> Result<()>;
}

/// Trait for model factories
pub trait ModelFactory: std::fmt::Debug + Send + Sync {
    /// Create model based on configuration
    fn create_model(&self, config: &ModelTypeConfig) -> Result<Box<dyn PerformancePredictor>>;

    /// Auto-select best model for data
    fn auto_select_model(
        &self,
        training_data: &[PerformanceDataPoint],
    ) -> Result<Box<dyn PerformancePredictor>>;

    /// Get available model types
    fn available_models(&self) -> Vec<String>;

    /// Get factory configuration
    fn config(&self) -> &ModelFactoryConfig;
}

// =============================================================================
// MAIN MODEL CONFIGURATION
// =============================================================================

/// Prediction engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionConfig {
    /// Enable caching
    pub enable_caching: bool,
    /// Cache size
    pub cache_size: usize,
    /// Confidence threshold
    pub confidence_threshold: f32,
}

impl Default for PredictionConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            cache_size: 1000,
            confidence_threshold: 0.8,
        }
    }
}

/// Main model configuration aggregating all sub-configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Training configuration
    pub training: ModelTrainingConfig,
    /// Prediction configuration
    pub prediction: PredictionConfig,
    /// Feature engineering configuration
    pub feature_engineering: FeatureEngineeringConfig,
    /// Validation configuration
    pub validation: ValidationConfig,
    /// Adaptive learning configuration
    pub adaptive_learning: AdaptiveLearningConfig,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            training: ModelTrainingConfig::default(),
            prediction: PredictionConfig::default(),
            feature_engineering: FeatureEngineeringConfig::default(),
            validation: ValidationConfig::default(),
            adaptive_learning: AdaptiveLearningConfig::default(),
        }
    }
}

// =============================================================================
// STATISTICAL ANALYSIS TYPES
// =============================================================================

/// Normality test methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalityTest {
    /// Shapiro-Wilk test
    ShapiroWilk,
    /// Kolmogorov-Smirnov test
    KolmogorovSmirnov,
    /// Anderson-Darling test
    AndersonDarling,
    /// Jarque-Bera test
    JarqueBera,
}

/// Density estimation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DensityEstimation {
    /// Kernel density estimation
    KernelDensity { bandwidth: f64 },
    /// Histogram-based
    Histogram { bins: usize },
    /// Gaussian mixture model
    GaussianMixture { components: usize },
}

/// Goodness of fit test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoodnessOfFit {
    /// Test statistic value
    pub test_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Test conclusion
    pub conclusion: String,
    /// Degrees of freedom
    pub degrees_of_freedom: Option<usize>,
}

// ============================================================================
// Additional Prediction Types
// ============================================================================

/// Batch prediction request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRequestBatch {
    pub batch_id: String,
    pub requests: Vec<PredictionRequest>,
    pub priority: i32,
}

// Engineered features for ML models - defined in feature_engineering.rs

impl Default for PredictionRequestBatch {
    fn default() -> Self {
        Self {
            batch_id: String::new(),
            requests: Vec::new(),
            priority: 0,
        }
    }
}
