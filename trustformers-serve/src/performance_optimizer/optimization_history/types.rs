//! Core Types for Optimization History Management
//!
//! This module provides comprehensive type definitions for optimization history tracking,
//! trend analysis, pattern recognition, anomaly detection, effectiveness analysis, and
//! predictive analytics. These types form the foundation for all optimization history
//! operations and enable sophisticated performance analysis and optimization insights.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

use crate::performance_optimizer::types::{
    OptimizationEvent as OptEvent, OptimizationEventType, PerformanceDataPoint,
    PerformanceMeasurement, SystemState, TestCharacteristics,
};
use crate::test_performance_monitoring::TrendDirection;

// =============================================================================
// CONFIGURATION TYPES
// =============================================================================

/// Configuration for historical data retention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryRetentionConfig {
    /// Maximum number of events to retain
    pub max_events: usize,
    /// Maximum age of events to retain
    pub max_age: Duration,
    /// Enable automatic cleanup
    pub auto_cleanup: bool,
    /// Cleanup interval
    pub cleanup_interval: Duration,
    /// Compression threshold for old data
    pub compression_threshold: Duration,
}

impl Default for HistoryRetentionConfig {
    fn default() -> Self {
        Self {
            max_events: 10000,
            max_age: Duration::from_secs(30 * 24 * 60 * 60), // 30 days
            auto_cleanup: true,
            cleanup_interval: Duration::from_secs(24 * 60 * 60), // 24 hours
            compression_threshold: Duration::from_secs(7 * 24 * 60 * 60), // 7 days
        }
    }
}

/// Configuration for enhanced trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysisConfig {
    /// Minimum data points for trend analysis
    pub min_data_points: usize,
    /// Analysis window duration
    pub analysis_window: Duration,
    /// Confidence threshold for trend detection
    pub confidence_threshold: f32,
    /// Enable predictive analysis
    pub enable_prediction: bool,
    /// Prediction horizon
    pub prediction_horizon: Duration,
    /// Cache expiry duration
    pub cache_expiry: Duration,
    /// Enable machine learning models
    pub enable_ml_models: bool,
}

impl Default for TrendAnalysisConfig {
    fn default() -> Self {
        Self {
            min_data_points: 5,
            analysis_window: Duration::from_secs(60 * 60), // 1 hour
            confidence_threshold: 0.7,
            enable_prediction: true,
            prediction_horizon: Duration::from_secs(30 * 60), // 30 minutes
            cache_expiry: Duration::from_secs(5 * 60),        // 5 minutes
            enable_ml_models: true,
        }
    }
}

/// Configuration for pattern recognition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternRecognitionConfig {
    /// Minimum pattern length
    pub min_pattern_length: usize,
    /// Pattern similarity threshold
    pub similarity_threshold: f32,
    /// Enable machine learning
    pub enable_ml_learning: bool,
    /// Learning rate for pattern adaptation
    pub learning_rate: f32,
    /// Pattern cache size
    pub cache_size: usize,
}

impl Default for PatternRecognitionConfig {
    fn default() -> Self {
        Self {
            min_pattern_length: 3,
            similarity_threshold: 0.8,
            enable_ml_learning: true,
            learning_rate: 0.1,
            cache_size: 1000,
        }
    }
}

/// Configuration for anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionConfig {
    /// Enable anomaly detection
    pub enable_detection: bool,
    /// Detection sensitivity (0.0 to 1.0)
    pub sensitivity: f32,
    /// Minimum severity threshold
    pub min_severity_threshold: f32,
    /// Enable machine learning models
    pub enable_ml_detection: bool,
    /// Learning rate for adaptive detection
    pub learning_rate: f32,
}

impl Default for AnomalyDetectionConfig {
    fn default() -> Self {
        Self {
            enable_detection: true,
            sensitivity: 0.7,
            min_severity_threshold: 0.5,
            enable_ml_detection: true,
            learning_rate: 0.1,
        }
    }
}

/// Configuration for effectiveness analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectivenessAnalysisConfig {
    /// Enable ROI calculation
    pub enable_roi_calculation: bool,
    /// Cost calculation method
    pub cost_calculation_method: CostCalculationMethod,
    /// Minimum effectiveness threshold
    pub min_effectiveness_threshold: f32,
    /// Enable statistical significance testing
    pub enable_significance_testing: bool,
    /// Statistical confidence level
    pub confidence_level: f32,
}

impl Default for EffectivenessAnalysisConfig {
    fn default() -> Self {
        Self {
            enable_roi_calculation: true,
            cost_calculation_method: CostCalculationMethod::ResourceBased,
            min_effectiveness_threshold: 0.1,
            enable_significance_testing: true,
            confidence_level: 0.95,
        }
    }
}

/// Configuration for advanced statistics computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticsConfig {
    /// Enable advanced statistical metrics
    pub enable_advanced_metrics: bool,
    /// Statistical window size
    pub window_size: usize,
    /// Enable correlation analysis
    pub enable_correlation_analysis: bool,
    /// Enable distribution analysis
    pub enable_distribution_analysis: bool,
    /// Statistical significance threshold
    pub significance_threshold: f32,
}

impl Default for StatisticsConfig {
    fn default() -> Self {
        Self {
            enable_advanced_metrics: true,
            window_size: 100,
            enable_correlation_analysis: true,
            enable_distribution_analysis: true,
            significance_threshold: 0.05,
        }
    }
}

/// Configuration for predictive analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveAnalyticsConfig {
    /// Enable predictive analytics
    pub enable_prediction: bool,
    /// Prediction models to use
    pub prediction_models: Vec<PredictionModelType>,
    /// Prediction horizon
    pub prediction_horizon: Duration,
    /// Model update frequency
    pub model_update_frequency: Duration,
    /// Minimum data points for prediction
    pub min_data_points: usize,
}

impl Default for PredictiveAnalyticsConfig {
    fn default() -> Self {
        Self {
            enable_prediction: true,
            prediction_models: vec![
                PredictionModelType::LinearRegression,
                PredictionModelType::MovingAverage,
                PredictionModelType::ExponentialSmoothing,
            ],
            prediction_horizon: Duration::from_secs(30 * 60), // 30 minutes
            model_update_frequency: Duration::from_secs(10 * 60), // 10 minutes
            min_data_points: 10,
        }
    }
}

// =============================================================================
// TREND ANALYSIS TYPES
// =============================================================================

/// Cached trend analysis result
#[derive(Debug, Clone)]
pub struct CachedTrendAnalysis {
    /// Trend analysis result
    pub trend: PerformanceTrend,
    /// Cache timestamp
    pub cached_at: DateTime<Utc>,
    /// Analysis duration
    pub analysis_duration: Duration,
    /// Confidence score
    pub confidence: f32,
}

/// Result of advanced trend analysis
#[derive(Debug, Clone)]
pub struct TrendAnalysisResult {
    /// Detected trend direction
    pub direction: TrendDirection,
    /// Trend strength (0.0 to 1.0)
    pub strength: f32,
    /// Analysis confidence (0.0 to 1.0)
    pub confidence: f32,
    /// Trend duration
    pub duration: Duration,
    /// Statistical significance
    pub significance: f32,
    /// Supporting data points
    pub data_points: Vec<PerformanceDataPoint>,
    /// Analysis method used
    pub method: String,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

/// Trend prediction result
#[derive(Debug, Clone)]
pub struct TrendPrediction {
    /// Predicted direction
    pub predicted_direction: TrendDirection,
    /// Prediction confidence
    pub confidence: f32,
    /// Prediction horizon
    pub horizon: Duration,
    /// Expected performance values
    pub expected_values: Vec<PredictedPerformancePoint>,
    /// Prediction model used
    pub model: String,
    /// Prediction uncertainty
    pub uncertainty: f32,
}

/// Predicted performance point
#[derive(Debug, Clone)]
pub struct PredictedPerformancePoint {
    /// Timestamp of prediction
    pub timestamp: DateTime<Utc>,
    /// Predicted throughput
    pub predicted_throughput: f64,
    /// Predicted latency
    pub predicted_latency: Duration,
    /// Prediction confidence
    pub confidence: f32,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
}

// =============================================================================
// PATTERN RECOGNITION TYPES
// =============================================================================

/// Recognized optimization pattern
#[derive(Debug, Clone)]
pub struct RecognizedPattern {
    /// Pattern identifier
    pub id: String,
    /// Pattern type
    pub pattern_type: PatternType,
    /// Pattern description
    pub description: String,
    /// Pattern frequency
    pub frequency: f32,
    /// Pattern confidence
    pub confidence: f32,
    /// Associated events
    pub events: Vec<OptEvent>,
    /// Pattern effectiveness
    pub effectiveness: f32,
    /// First observed timestamp
    pub first_observed: DateTime<Utc>,
    /// Last observed timestamp
    pub last_observed: DateTime<Utc>,
    /// Pattern characteristics
    pub characteristics: HashMap<String, f64>,
}

/// Types of optimization patterns
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternType {
    /// Cyclical performance pattern
    Cyclical,
    /// Degradation pattern
    Degradation,
    /// Improvement pattern
    Improvement,
    /// Oscillation pattern
    Oscillation,
    /// Threshold pattern
    Threshold,
    /// Custom pattern
    Custom(String),
}

/// Pattern context for prediction
#[derive(Debug, Clone)]
pub struct PatternContext {
    /// Current system state
    pub system_state: SystemState,
    /// Recent performance data
    pub recent_performance: Vec<PerformanceDataPoint>,
    /// Current test characteristics
    pub test_characteristics: TestCharacteristics,
    /// Time context
    pub timestamp: DateTime<Utc>,
}

/// Pattern prediction result
#[derive(Debug, Clone)]
pub struct PatternPrediction {
    /// Predicted pattern type
    pub predicted_pattern: PatternType,
    /// Prediction confidence
    pub confidence: f32,
    /// Expected occurrence time
    pub expected_occurrence: DateTime<Utc>,
    /// Prediction horizon
    pub horizon: Duration,
    /// Recommended actions
    pub recommended_actions: Vec<String>,
}

// =============================================================================
// ANOMALY DETECTION TYPES
// =============================================================================

/// Detected performance anomaly
#[derive(Debug, Clone)]
pub struct DetectedAnomaly {
    /// Anomaly identifier
    pub id: String,
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Anomaly description
    pub description: String,
    /// Anomaly severity (0.0 to 1.0)
    pub severity: f32,
    /// Detection confidence
    pub confidence: f32,
    /// Anomalous data point
    pub data_point: PerformanceDataPoint,
    /// Detection timestamp
    pub detected_at: DateTime<Utc>,
    /// Expected value range
    pub expected_range: (f64, f64),
    /// Actual value deviation
    pub deviation: f64,
    /// Detection method
    pub detection_method: String,
    /// Anomaly metadata
    pub metadata: HashMap<String, String>,
}

/// Types of performance anomalies
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Performance spike
    Spike,
    /// Performance drop
    Drop,
    /// Unusual pattern
    UnusualPattern,
    /// System instability
    SystemInstability,
    /// Resource exhaustion
    ResourceExhaustion,
    /// Custom anomaly
    Custom(String),
}

/// Anomaly context for prediction
#[derive(Debug, Clone)]
pub struct AnomalyContext {
    /// Current performance data
    pub current_performance: Vec<PerformanceDataPoint>,
    /// Historical baseline
    pub baseline: PerformanceBaseline,
    /// System conditions
    pub system_state: SystemState,
    /// Time context
    pub timestamp: DateTime<Utc>,
}

/// Anomaly prediction result
#[derive(Debug, Clone)]
pub struct AnomalyPrediction {
    /// Predicted anomaly type
    pub predicted_anomaly: AnomalyType,
    /// Likelihood of occurrence
    pub likelihood: f32,
    /// Expected occurrence time
    pub expected_time: DateTime<Utc>,
    /// Prediction confidence
    pub confidence: f32,
    /// Preventive measures
    pub preventive_measures: Vec<String>,
}

/// Performance baseline for anomaly detection
#[derive(Debug, Clone)]
pub struct PerformanceBaseline {
    /// Baseline throughput
    pub baseline_throughput: f64,
    /// Baseline latency
    pub baseline_latency: Duration,
    /// Throughput variance
    pub throughput_variance: f64,
    /// Latency variance
    pub latency_variance: f64,
    /// Baseline timestamp
    pub baseline_timestamp: DateTime<Utc>,
    /// Sample size
    pub sample_size: usize,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
}

// =============================================================================
// EFFECTIVENESS ANALYSIS TYPES
// =============================================================================

/// Optimization effectiveness analysis result
#[derive(Debug, Clone)]
pub struct EffectivenessAnalysisResult {
    /// Effectiveness score (0.0 to 1.0)
    pub effectiveness_score: f32,
    /// Return on investment
    pub roi: f32,
    /// Cost-benefit analysis
    pub cost_benefit: CostBenefitAnalysis,
    /// Performance improvement
    pub performance_improvement: PerformanceImprovement,
    /// Statistical significance
    pub statistical_significance: StatisticalSignificance,
    /// Analysis timestamp
    pub analyzed_at: DateTime<Utc>,
}

/// Cost-benefit analysis result
#[derive(Debug, Clone)]
pub struct CostBenefitAnalysis {
    /// Implementation cost
    pub implementation_cost: f64,
    /// Operational cost
    pub operational_cost: f64,
    /// Total cost
    pub total_cost: f64,
    /// Performance benefit
    pub performance_benefit: f64,
    /// Resource savings
    pub resource_savings: f64,
    /// Total benefit
    pub total_benefit: f64,
    /// Net benefit
    pub net_benefit: f64,
    /// Payback period
    pub payback_period: Duration,
}

/// Performance improvement metrics
#[derive(Debug, Clone)]
pub struct PerformanceImprovement {
    /// Throughput improvement percentage
    pub throughput_improvement: f32,
    /// Latency improvement percentage
    pub latency_improvement: f32,
    /// Resource utilization improvement
    pub resource_improvement: f32,
    /// Overall performance improvement
    pub overall_improvement: f32,
    /// Improvement duration
    pub improvement_duration: Duration,
}

/// Statistical significance test result
#[derive(Debug, Clone)]
pub struct StatisticalSignificance {
    /// Test statistic
    pub test_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Confidence level
    pub confidence_level: f32,
    /// Is statistically significant
    pub is_significant: bool,
    /// Test method used
    pub test_method: String,
}

/// Cost calculation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CostCalculationMethod {
    /// Resource-based cost calculation
    ResourceBased,
    /// Time-based cost calculation
    TimeBased,
    /// Hybrid cost calculation
    Hybrid,
    /// Custom cost calculation
    Custom(String),
}

// =============================================================================
// STATISTICS TYPES
// =============================================================================

/// Comprehensive optimization statistics
#[derive(Debug, Clone)]
pub struct ComprehensiveOptimizationStatistics {
    /// Basic statistics
    pub basic_stats: BasicStatistics,
    /// Distribution analysis
    pub distribution_analysis: DistributionAnalysis,
    /// Correlation analysis
    pub correlation_analysis: CorrelationAnalysis,
    /// Time series analysis
    pub time_series_analysis: TimeSeriesAnalysis,
    /// Statistical tests
    pub statistical_tests: Vec<StatisticalTest>,
    /// Analysis timestamp
    pub analyzed_at: DateTime<Utc>,
}

/// Basic statistical metrics
#[derive(Debug, Clone)]
pub struct BasicStatistics {
    /// Mean performance
    pub mean: f64,
    /// Median performance
    pub median: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Variance
    pub variance: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Range
    pub range: f64,
    /// Skewness
    pub skewness: f64,
    /// Kurtosis
    pub kurtosis: f64,
}

/// Distribution analysis result
#[derive(Debug, Clone)]
pub struct DistributionAnalysis {
    /// Distribution type
    pub distribution_type: DistributionType,
    /// Distribution parameters
    pub parameters: HashMap<String, f64>,
    /// Goodness of fit
    pub goodness_of_fit: f64,
    /// Confidence level
    pub confidence_level: f32,
}

/// Correlation analysis result
#[derive(Debug, Clone)]
pub struct CorrelationAnalysis {
    /// Correlation coefficients
    pub correlations: HashMap<String, f64>,
    /// Statistical significance
    pub significance: HashMap<String, f64>,
    /// Correlation matrix
    pub correlation_matrix: Vec<Vec<f64>>,
}

/// Time series analysis result
#[derive(Debug, Clone)]
pub struct TimeSeriesAnalysis {
    /// Trend component
    pub trend: Vec<f64>,
    /// Seasonal component
    pub seasonal: Vec<f64>,
    /// Residual component
    pub residual: Vec<f64>,
    /// Autocorrelation
    pub autocorrelation: Vec<f64>,
    /// Stationarity test
    pub stationarity_test: StationarityTest,
}

/// Statistical test result
#[derive(Debug, Clone)]
pub struct StatisticalTest {
    /// Test name
    pub test_name: String,
    /// Test statistic
    pub test_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Critical value
    pub critical_value: f64,
    /// Is significant
    pub is_significant: bool,
}

/// Distribution types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionType {
    /// Normal distribution
    Normal,
    /// Exponential distribution
    Exponential,
    /// Uniform distribution
    Uniform,
    /// Gamma distribution
    Gamma,
    /// Custom distribution
    Custom(String),
}

impl Default for DistributionType {
    fn default() -> Self {
        DistributionType::Normal
    }
}

/// Stationarity test result
#[derive(Debug, Clone)]
pub struct StationarityTest {
    /// Test name
    pub test_name: String,
    /// Is stationary
    pub is_stationary: bool,
    /// Test statistic
    pub test_statistic: f64,
    /// P-value
    pub p_value: f64,
}

// =============================================================================
// PREDICTIVE ANALYTICS TYPES
// =============================================================================

/// Prediction model types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PredictionModelType {
    /// Linear regression
    LinearRegression,
    /// Moving average
    MovingAverage,
    /// Exponential smoothing
    ExponentialSmoothing,
    /// ARIMA model
    ARIMA,
    /// Neural network
    NeuralNetwork,
    /// Custom model
    Custom(String),
}

/// Performance prediction result
#[derive(Debug, Clone)]
pub struct PerformancePrediction {
    /// Prediction identifier
    pub id: String,
    /// Predicted values
    pub predicted_values: Vec<PredictedPerformancePoint>,
    /// Prediction model used
    pub model: PredictionModelType,
    /// Overall confidence
    pub confidence: f32,
    /// Prediction horizon
    pub horizon: Duration,
    /// Prediction uncertainty
    pub uncertainty: f32,
    /// Prediction timestamp
    pub predicted_at: DateTime<Utc>,
}

/// Predictive model performance metrics
#[derive(Debug, Clone)]
pub struct ModelPerformanceMetrics {
    /// Mean absolute error
    pub mae: f64,
    /// Mean squared error
    pub mse: f64,
    /// Root mean squared error
    pub rmse: f64,
    /// R-squared
    pub r_squared: f64,
    /// Mean absolute percentage error
    pub mape: f64,
}

// =============================================================================
// LEGACY TYPES FOR BACKWARD COMPATIBILITY
// =============================================================================

/// Legacy optimization history for backward compatibility
#[derive(Debug, Clone, Default)]
pub struct LegacyOptimizationHistory {
    /// Optimization events
    pub events: Vec<LegacyOptimizationEvent>,
    /// Performance trends
    pub trends: HashMap<String, LegacyPerformanceTrend>,
    /// Optimization effectiveness
    pub effectiveness: LegacyOptimizationEffectiveness,
    /// History statistics
    pub statistics: LegacyOptimizationStatistics,
}

/// Legacy optimization event for backward compatibility
#[derive(Debug, Clone)]
pub struct LegacyOptimizationEvent {
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Event type
    pub event_type: OptimizationEventType,
    /// Event description
    pub description: String,
    /// Performance before
    pub performance_before: Option<PerformanceMeasurement>,
    /// Performance after
    pub performance_after: Option<PerformanceMeasurement>,
    /// Optimization parameters
    pub parameters: HashMap<String, String>,
    /// Event metadata
    pub metadata: HashMap<String, String>,
}

/// Legacy performance trend for backward compatibility
#[derive(Debug, Clone)]
pub struct LegacyPerformanceTrend {
    /// Trend direction
    pub direction: TrendDirection,
    /// Trend strength
    pub strength: f32,
    /// Trend confidence
    pub confidence: f32,
    /// Trend period
    pub period: Duration,
    /// Trend data points
    pub data_points: Vec<PerformanceDataPoint>,
}

/// Legacy optimization effectiveness for backward compatibility
#[derive(Debug, Clone, Default)]
pub struct LegacyOptimizationEffectiveness {
    /// Overall effectiveness score
    pub overall_score: f32,
    /// Effectiveness by optimization type
    pub by_type: HashMap<OptimizationEventType, f32>,
    /// ROI metrics
    pub roi: f32,
    /// Success rate
    pub success_rate: f32,
}

/// Legacy optimization statistics for backward compatibility
#[derive(Debug, Clone, Default)]
pub struct LegacyOptimizationStatistics {
    /// Total optimizations
    pub total_optimizations: u64,
    /// Successful optimizations
    pub successful_optimizations: u64,
    /// Average improvement
    pub average_improvement: f32,
    /// Best improvement
    pub best_improvement: f32,
    /// Total ROI
    pub total_roi: f32,
}

/// Legacy optimization record for backward compatibility
#[derive(Debug, Clone)]
pub struct LegacyOptimizationRecord {
    /// Record identifier
    pub id: String,
    /// Optimization event
    pub event: LegacyOptimizationEvent,
    /// Effectiveness score
    pub effectiveness: f32,
    /// ROI calculation
    pub roi: f32,
}

// Type aliases for backward compatibility
pub type OptimizationHistory = LegacyOptimizationHistory;
pub type OptimizationEvent = LegacyOptimizationEvent;
pub type PerformanceTrend = LegacyPerformanceTrend;
pub type OptimizationStatistics = LegacyOptimizationStatistics;
pub type OptimizationRecord = LegacyOptimizationRecord;
pub type OptimizationEffectiveness = LegacyOptimizationEffectiveness;

// =============================================================================
// TRAIT DEFINITIONS
// =============================================================================

/// Trait for advanced trend detection algorithms
pub trait TrendDetectionAlgorithm {
    /// Detect trend in performance data
    fn detect_trend(&self, data_points: &[PerformanceDataPoint]) -> Result<TrendAnalysisResult>;
    /// Get algorithm name
    fn name(&self) -> &str;
    /// Get algorithm confidence for given data
    fn confidence(&self, data_points: &[PerformanceDataPoint]) -> f32;
    /// Check if algorithm is applicable
    fn is_applicable(&self, data_points: &[PerformanceDataPoint]) -> bool;
}

/// Trait for trend prediction models
pub trait TrendPredictionModel {
    /// Predict future trend
    fn predict_trend(
        &self,
        current_trend: &PerformanceTrend,
        horizon: Duration,
    ) -> Result<TrendPrediction>;
    /// Get model name
    fn name(&self) -> &str;
    /// Get prediction confidence
    fn prediction_confidence(&self, trend: &PerformanceTrend) -> f32;
}

/// Trait for pattern detection algorithms
pub trait PatternDetector {
    /// Detect patterns in optimization history
    fn detect_patterns(&self, events: &[OptEvent]) -> Result<Vec<RecognizedPattern>>;
    /// Get detector name
    fn name(&self) -> &str;
    /// Check if detector is applicable
    fn is_applicable(&self, events: &[OptEvent]) -> bool;
}

/// Trait for pattern learning models
pub trait PatternLearningModel {
    /// Learn from recognized patterns
    fn learn_from_patterns(&mut self, patterns: &[RecognizedPattern]) -> Result<()>;
    /// Predict pattern occurrence
    fn predict_pattern(&self, context: &PatternContext) -> Result<PatternPrediction>;
    /// Get model name
    fn name(&self) -> &str;
}

/// Trait for anomaly detection algorithms
pub trait AnomalyDetector {
    /// Detect anomalies in performance data
    fn detect_anomalies(
        &self,
        data_points: &[PerformanceDataPoint],
    ) -> Result<Vec<DetectedAnomaly>>;
    /// Get detector name
    fn name(&self) -> &str;
    /// Get detection confidence
    fn confidence(&self, data_points: &[PerformanceDataPoint]) -> f32;
}

/// Trait for anomaly learning models
pub trait AnomalyLearningModel {
    /// Learn from detected anomalies
    fn learn_from_anomalies(&mut self, anomalies: &[DetectedAnomaly]) -> Result<()>;
    /// Predict anomaly likelihood
    fn predict_anomaly(&self, context: &AnomalyContext) -> Result<AnomalyPrediction>;
    /// Get model name
    fn name(&self) -> &str;
}

/// Trait for effectiveness calculators
pub trait EffectivenessCalculator {
    /// Calculate optimization effectiveness
    fn calculate_effectiveness(
        &self,
        before: &PerformanceMeasurement,
        after: &PerformanceMeasurement,
    ) -> Result<EffectivenessAnalysisResult>;
    /// Get calculator name
    fn name(&self) -> &str;
}

/// Trait for cost calculators
pub trait CostCalculator {
    /// Calculate optimization cost
    fn calculate_cost(&self, optimization_params: &HashMap<String, String>) -> Result<f64>;
    /// Get calculator name
    fn name(&self) -> &str;
}

/// Trait for predictive models
pub trait PredictiveModel {
    /// Train the model with historical data
    fn train(&mut self, data: &[PerformanceDataPoint]) -> Result<()>;
    /// Make predictions
    fn predict(&self, horizon: Duration) -> Result<PerformancePrediction>;
    /// Get model type
    fn model_type(&self) -> PredictionModelType;
    /// Get model performance metrics
    fn performance_metrics(&self) -> ModelPerformanceMetrics;
}
