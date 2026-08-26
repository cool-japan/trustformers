//! Analysis-related types for test characterization

use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use super::super::{
    analysis::{TrendAnalysis, TrendAnalysisAlgorithm, TrendDirection},
    quality::{QualityAssessment, ValidationResult},
};
use super::enums::{TestCharacterizationError, TestCharacterizationResult};

#[derive(Debug, Clone)]
pub struct AccuracyRecord {
    /// Pattern identifier
    pub pattern_id: String,
    /// Overall accuracy score
    pub accuracy_score: f64,
    /// Precision metrics
    pub precision: f64,
    /// Recall metrics
    pub recall: f64,
    /// F1 score
    pub f1_score: f64,
    /// Validation history
    pub validation_history: Vec<ValidationResult>,
    /// False positive rate
    pub false_positive_rate: f64,
    /// False negative rate
    pub false_negative_rate: f64,
    /// Confidence intervals
    pub confidence_intervals: HashMap<String, (f64, f64)>,
    /// Statistical significance
    pub statistical_significance: f64,
    /// Reliability score
    pub reliability_score: f64,
}

#[derive(Debug, Clone)]
pub struct AlgorithmPerformance {
    /// Algorithm identifier
    pub algorithm_id: String,
    /// Average execution time
    pub average_execution_time: Duration,
    /// Accuracy score
    pub accuracy_score: f64,
    /// Resource overhead
    pub resource_overhead: f64,
    /// Reliability score
    pub reliability_score: f64,
    /// Usage frequency
    pub usage_frequency: f64,
    /// Error rate
    pub error_rate: f64,
    /// Performance trend
    pub trend: TrendDirection,
    /// Last updated timestamp
    pub last_updated: Instant,
    /// Quality assessments
    pub quality_assessments: Vec<QualityAssessment>,
    /// Total runs
    pub total_runs: usize,
    /// Successful runs
    pub successful_runs: usize,
    /// Total duration
    pub total_duration: Duration,
    /// Success rate
    pub success_rate: f64,
    /// Average duration
    pub avg_duration: Duration,
}

impl Default for AlgorithmPerformance {
    fn default() -> Self {
        Self {
            algorithm_id: String::new(),
            average_execution_time: Duration::ZERO,
            accuracy_score: 0.0,
            resource_overhead: 0.0,
            reliability_score: 0.0,
            usage_frequency: 0.0,
            error_rate: 0.0,
            trend: TrendDirection::Stable,
            last_updated: Instant::now(),
            quality_assessments: Vec::new(),
            total_runs: 0,
            successful_runs: 0,
            total_duration: Duration::ZERO,
            success_rate: 0.0,
            avg_duration: Duration::ZERO,
        }
    }
}

pub struct AlgorithmSelection {
    pub selected_algorithm: String,
    pub selection_reason: String,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct ArimaTrendAnalyzer {
    pub ar_order: usize,
    pub diff_order: usize,
    pub ma_order: usize,
}

impl ArimaTrendAnalyzer {
    /// Create a new ArimaTrendAnalyzer with default settings
    pub fn new() -> Self {
        Self {
            ar_order: 1,
            diff_order: 0,
            ma_order: 1,
        }
    }
}

impl Default for ArimaTrendAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl TrendAnalysisAlgorithm for ArimaTrendAnalyzer {
    /// Difference the series `diff_order` times, then read the direction off
    /// the mean of the differenced series against its own standard error.
    ///
    /// This is the integrated (`I`) part of ARIMA computed exactly; the `AR`
    /// and `MA` parts are not fitted, which is why `name()` says so. Before
    /// 0.2.1 this ignored `data` entirely and returned `Stable` with a
    /// hardcoded `confidence: 0.70`.
    fn analyze_trend(&self, data: &[(Instant, f64)]) -> TestCharacterizationResult<TrendAnalysis> {
        let mut values: Vec<f64> = data.iter().map(|(_, value)| *value).collect();
        for _ in 0..self.diff_order {
            values = values
                .windows(2)
                .filter_map(|pair| {
                    let (Some(previous), Some(current)) = (pair.first(), pair.get(1)) else {
                        return None;
                    };
                    Some(current - previous)
                })
                .collect();
        }
        // Direction comes from consecutive first differences of whatever series
        // survived the requested differencing.
        let deltas: Vec<f64> = values
            .windows(2)
            .filter_map(|pair| {
                let (Some(previous), Some(current)) = (pair.first(), pair.get(1)) else {
                    return None;
                };
                Some(current - previous)
            })
            .collect();
        if deltas.len() < 2 {
            return Err(TestCharacterizationError::InvalidInput {
                message: format!(
                    "differencing of order {} left {} usable step(s); at least two are needed",
                    self.diff_order,
                    deltas.len()
                ),
                field: "data".to_string(),
                value: data.len().to_string(),
            });
        }
        let n = deltas.len() as f64;
        let mean_delta = deltas.iter().sum::<f64>() / n;
        let variance =
            deltas.iter().map(|delta| (delta - mean_delta).powi(2)).sum::<f64>() / (n - 1.0);
        let standard_error = (variance / n).sqrt();
        // t-ratio of the mean step against zero; a mean smaller than its own
        // standard error is not evidence of a direction.
        let t_ratio = if standard_error > 0.0 { mean_delta / standard_error } else { 0.0 };
        let overall_direction = if t_ratio > 2.0 {
            TrendDirection::Increasing
        } else if t_ratio < -2.0 {
            TrendDirection::Decreasing
        } else if variance > 0.0 && mean_delta.abs() < variance.sqrt() {
            TrendDirection::Fluctuating
        } else {
            TrendDirection::Stable
        };
        Ok(TrendAnalysis {
            // Segmenting the window into individual trends is not attempted.
            detected_trends: Vec::new(),
            overall_direction,
            // Confidence is the t-ratio scaled so |t| >= 4 reads as certain.
            confidence: (t_ratio.abs() / 4.0).clamp(0.0, 1.0),
            // `analyze_trend` characterises; `predict` forecasts.
            forecast: Vec::new(),
        })
    }

    fn name(&self) -> &str {
        // Only the integrated (differencing) component is computed; the AR and
        // MA coefficients are never fitted, so the name says what it does.
        "DifferencedMeanTrendAnalyzer"
    }

    fn confidence(&self, data: &[(Instant, f64)]) -> f64 {
        // Confidence based on model order and data length
        let min_required = (self.ar_order + self.diff_order + self.ma_order) * 10;
        if data.len() >= min_required {
            0.85
        } else {
            0.60
        }
    }

    /// Extrapolate the mean first difference from the last observation.
    ///
    /// Before 0.2.1 this repeated the last value `steps` times, which is a
    /// random-walk forecast dressed up as "AR-based".
    fn predict(
        &self,
        data: &[(Instant, f64)],
        steps: usize,
    ) -> TestCharacterizationResult<Vec<f64>> {
        let values: Vec<f64> = data.iter().map(|(_, value)| *value).collect();
        let Some(last) = values.last().copied() else {
            return Err(TestCharacterizationError::InvalidInput {
                message: "a forecast needs at least one observation to start from".to_string(),
                field: "data".to_string(),
                value: "0".to_string(),
            });
        };
        let deltas: Vec<f64> = values
            .windows(2)
            .filter_map(|pair| {
                let (Some(previous), Some(current)) = (pair.first(), pair.get(1)) else {
                    return None;
                };
                Some(current - previous)
            })
            .collect();
        // With a single observation there is no drift to extrapolate; holding
        // the level is then the honest forecast, not a stand-in for one.
        let drift = if deltas.is_empty() {
            0.0
        } else {
            deltas.iter().sum::<f64>() / deltas.len() as f64
        };
        Ok((1..=steps).map(|step| last + drift * step as f64).collect())
    }
}

#[derive(Debug, Clone)]
pub struct CriticalPathAnalyzer {
    pub analysis_depth: usize,
    pub path_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct DetectedAnomaly {
    pub anomaly_type: String,
    pub severity: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

pub struct DetectedImprovement {
    pub improvement_type: String,
    pub improvement_magnitude: f64,
    pub confidence: f64,
}

pub struct DriftDetection {
    pub drift_detected: bool,
    pub drift_magnitude: f64,
    pub detection_timestamp: chrono::DateTime<chrono::Utc>,
}

pub struct EngineeredFeatures {
    pub features: HashMap<String, f64>,
    pub feature_names: Vec<String>,
}

pub struct EstimatedEffort {
    pub effort_hours: f64,
    pub confidence: f64,
}

pub struct EstimationCalibrationPoint {
    pub actual_value: f64,
    pub estimated_value: f64,
    pub error: f64,
}

#[derive(Debug, Clone)]
pub struct ExponentialTrendAnalyzer {
    pub base: f64,
    pub growth_rate: f64,
    pub confidence: f64,
}

impl ExponentialTrendAnalyzer {
    /// Create a new ExponentialTrendAnalyzer with default settings
    pub fn new() -> Self {
        Self {
            base: 1.0,
            growth_rate: 0.0,
            confidence: 0.0,
        }
    }
}

impl Default for ExponentialTrendAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl TrendAnalysisAlgorithm for ExponentialTrendAnalyzer {
    fn analyze_trend(&self, _data: &[(Instant, f64)]) -> TestCharacterizationResult<TrendAnalysis> {
        // Determine direction based on growth rate
        let direction = if self.growth_rate > 0.01 {
            TrendDirection::Increasing
        } else if self.growth_rate < -0.01 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        Ok(TrendAnalysis {
            detected_trends: Vec::new(),
            overall_direction: direction,
            confidence: self.confidence,
            forecast: Vec::new(),
        })
    }

    fn name(&self) -> &str {
        "ExponentialTrendAnalyzer"
    }

    fn confidence(&self, _data: &[(Instant, f64)]) -> f64 {
        self.confidence
    }

    fn predict(
        &self,
        _data: &[(Instant, f64)],
        steps: usize,
    ) -> TestCharacterizationResult<Vec<f64>> {
        // Exponential growth: y = base * e^(growth_rate * t)
        let forecast: Vec<f64> =
            (0..steps).map(|i| self.base * (self.growth_rate * i as f64).exp()).collect();
        Ok(forecast)
    }
}

pub struct FalsePositiveAssessment {
    pub false_positive_rate: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct ForecastingResults {
    pub forecasted_values: Vec<f64>,
    pub confidence_intervals: Vec<(f64, f64)>,
}

#[derive(Debug, Clone)]
pub struct LinearTrendAnalyzer {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
}

impl LinearTrendAnalyzer {
    /// Create a new LinearTrendAnalyzer with default settings
    pub fn new() -> Self {
        Self {
            slope: 0.0,
            intercept: 0.0,
            r_squared: 0.0,
        }
    }
}

impl Default for LinearTrendAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl TrendAnalysisAlgorithm for LinearTrendAnalyzer {
    fn analyze_trend(&self, _data: &[(Instant, f64)]) -> TestCharacterizationResult<TrendAnalysis> {
        // Determine direction based on slope
        let direction = if self.slope > 0.01 {
            TrendDirection::Increasing
        } else if self.slope < -0.01 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        Ok(TrendAnalysis {
            detected_trends: Vec::new(),
            overall_direction: direction,
            confidence: self.r_squared,
            forecast: Vec::new(),
        })
    }

    fn name(&self) -> &str {
        "LinearTrendAnalyzer"
    }

    fn confidence(&self, _data: &[(Instant, f64)]) -> f64 {
        // Use R² as confidence measure
        self.r_squared
    }

    fn predict(
        &self,
        data: &[(Instant, f64)],
        steps: usize,
    ) -> TestCharacterizationResult<Vec<f64>> {
        // Linear extrapolation: y = mx + b
        let start_x = data.len() as f64;
        let forecast: Vec<f64> =
            (0..steps).map(|i| self.slope * (start_x + i as f64) + self.intercept).collect();
        Ok(forecast)
    }
}

pub struct NormalityTest {
    pub test_type: String,
    pub p_value: f64,
    pub is_normal: bool,
}

pub struct RecoveryCharacteristics {
    pub recovery_time: Duration,
    pub success_rate: f64,
    pub failure_modes: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ScalabilityAnalysis {
    pub score: f64,
    pub bottlenecks: Vec<String>,
    pub recommended_threads: usize,
}

impl ScalabilityAnalysis {
    pub fn new() -> Self {
        Self {
            score: 0.0,
            bottlenecks: Vec::new(),
            recommended_threads: 1,
        }
    }
}

impl Default for ScalabilityAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl super::super::patterns::ThreadAnalysisAlgorithm for ScalabilityAnalysis {
    fn analyze(&self) -> String {
        format!(
            "Scalability score: {:.2}, efficiency: {:.2}%",
            self.score,
            self.score * 100.0
        )
    }

    fn name(&self) -> &str {
        "ScalabilityAnalysis"
    }
}

#[derive(Debug, Clone)]
pub struct ScalabilityPattern {
    pub pattern_type: String,
    pub efficiency_curve: Vec<f64>,
}

pub struct ScalabilityRating {
    pub rating: String,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalDecomposition {
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub residual: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalPattern {
    pub period: usize,
    pub amplitude: f64,
    pub phase: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalTrendAnalyzer {
    pub period: usize,
    pub amplitude: f64,
    pub phase_shift: f64,
}

impl SeasonalTrendAnalyzer {
    /// Create a new SeasonalTrendAnalyzer with default settings
    pub fn new() -> Self {
        Self {
            period: 24,
            amplitude: 1.0,
            phase_shift: 0.0,
        }
    }
}

impl Default for SeasonalTrendAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl TrendAnalysisAlgorithm for SeasonalTrendAnalyzer {
    /// Measure how strongly the series repeats at `period` samples, via the
    /// autocorrelation at that lag.
    ///
    /// Before 0.2.1 this ignored `data` and asserted `Cyclical` with a
    /// hardcoded `confidence: 0.75` for any input, including a flat line.
    fn analyze_trend(&self, data: &[(Instant, f64)]) -> TestCharacterizationResult<TrendAnalysis> {
        if self.period == 0 {
            return Err(TestCharacterizationError::InvalidInput {
                message: "seasonal period must be positive".to_string(),
                field: "period".to_string(),
                value: self.period.to_string(),
            });
        }
        let values: Vec<f64> = data.iter().map(|(_, value)| *value).collect();
        if values.len() <= self.period + 2 {
            return Err(TestCharacterizationError::InvalidInput {
                message: format!(
                    "a period of {} samples needs more than {} readings to autocorrelate; got {}",
                    self.period,
                    self.period + 2,
                    values.len()
                ),
                field: "data".to_string(),
                value: values.len().to_string(),
            });
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let denominator: f64 = values.iter().map(|value| (value - mean).powi(2)).sum();
        if denominator <= 0.0 {
            // A constant series repeats trivially and seasonally means nothing.
            return Ok(TrendAnalysis {
                detected_trends: Vec::new(),
                overall_direction: TrendDirection::Stable,
                confidence: 0.0,
                forecast: Vec::new(),
            });
        }
        let mut numerator = 0.0;
        for index in self.period..values.len() {
            let (Some(current), Some(lagged)) =
                (values.get(index), values.get(index - self.period))
            else {
                continue;
            };
            numerator += (current - mean) * (lagged - mean);
        }
        let autocorrelation = (numerator / denominator).clamp(-1.0, 1.0);
        let overall_direction = if autocorrelation >= 0.5 {
            TrendDirection::Seasonal
        } else if autocorrelation >= 0.2 {
            TrendDirection::Cyclical
        } else {
            TrendDirection::Random
        };
        Ok(TrendAnalysis {
            detected_trends: Vec::new(),
            overall_direction,
            // Confidence is the measured autocorrelation at the configured lag.
            confidence: autocorrelation.max(0.0),
            forecast: Vec::new(),
        })
    }

    fn name(&self) -> &str {
        "SeasonalTrendAnalyzer"
    }

    fn confidence(&self, data: &[(Instant, f64)]) -> f64 {
        // Confidence based on data length and periodicity
        if data.len() >= self.period * 2 {
            0.80
        } else {
            0.50
        }
    }

    /// A sinusoid of the configured period and amplitude, centred on the mean
    /// of the observed series.
    ///
    /// Before 0.2.1 the observed data was ignored (`_data`), so the forecast
    /// oscillated about zero regardless of where the series actually sat.
    fn predict(
        &self,
        data: &[(Instant, f64)],
        steps: usize,
    ) -> TestCharacterizationResult<Vec<f64>> {
        if self.period == 0 {
            return Err(TestCharacterizationError::InvalidInput {
                message: "seasonal period must be positive".to_string(),
                field: "period".to_string(),
                value: self.period.to_string(),
            });
        }
        if data.is_empty() {
            return Err(TestCharacterizationError::InvalidInput {
                message: "a seasonal forecast needs at least one observation to centre on"
                    .to_string(),
                field: "data".to_string(),
                value: "0".to_string(),
            });
        }
        let level = data.iter().map(|(_, value)| *value).sum::<f64>() / data.len() as f64;
        // The forecast continues the cycle from where the observed series ends.
        let offset = data.len();
        let forecast: Vec<f64> = (0..steps)
            .map(|i| {
                let phase = 2.0 * std::f64::consts::PI * ((offset + i) as f64)
                    / (self.period as f64)
                    + self.phase_shift;
                level + self.amplitude * phase.sin()
            })
            .collect();
        Ok(forecast)
    }
}

pub struct SensitivityLevel {
    pub level: String,
    pub sensitivity_score: f64,
    pub threshold: f64,
}

pub struct ComprehensiveCacheAnalysis {
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub hit_rate: f64,
}

pub struct ComputeUtilizationAnalysis {
    pub utilization_percentage: f64,
    pub idle_time: Duration,
}

#[derive(Debug, Clone)]
pub struct HoldTimeAnalysis {
    pub avg_hold_time_us: u64,
    pub max_hold_time_us: u64,
}

impl HoldTimeAnalysis {
    pub fn new() -> Self {
        Self {
            avg_hold_time_us: 0,
            max_hold_time_us: 0,
        }
    }
}

impl Default for HoldTimeAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl super::super::patterns::ThreadAnalysisAlgorithm for HoldTimeAnalysis {
    fn analyze(&self) -> String {
        format!("Average hold time: {} μs", self.avg_hold_time_us)
    }

    fn name(&self) -> &str {
        "HoldTimeAnalysis"
    }
}

impl super::super::locking::LockAnalysisAlgorithm for HoldTimeAnalysis {
    fn analyze(&self) -> String {
        // Convert microseconds to a normalized score
        let hold_time_ms = self.avg_hold_time_us as f64 / 1000.0;
        let score = (1.0 / (1.0 + hold_time_ms / 100.0)).min(1.0);
        format!("Lock hold time: {:.2}ms, score: {:.2}", hold_time_ms, score)
    }

    fn name(&self) -> &str {
        "HoldTimeAnalysis"
    }

    fn analyze_locks(&self) -> String {
        self.analyze()
    }
}

/// Analysis metadata for test characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    /// Analysis timestamp
    #[serde(skip_deserializing, default = "std::time::SystemTime::now")]
    pub timestamp: std::time::SystemTime,
    /// Analysis version
    pub version: String,
    /// Confidence score of the analysis
    pub confidence_score: f64,
    /// Additional notes
    pub notes: Vec<String>,
}

impl Default for AnalysisMetadata {
    fn default() -> Self {
        Self {
            timestamp: std::time::SystemTime::now(),
            version: "1.0.0".to_string(),
            confidence_score: 0.0,
            notes: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CommunicationPatternAnalysis {
    pub overhead: f64,
    pub pattern_type: String,
}

impl CommunicationPatternAnalysis {
    pub fn new() -> Self {
        Self {
            overhead: 0.0,
            pattern_type: String::new(),
        }
    }
}

impl Default for CommunicationPatternAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl super::super::patterns::ThreadAnalysisAlgorithm for CommunicationPatternAnalysis {
    fn analyze(&self) -> String {
        let score = 1.0 - self.overhead.min(1.0); // Higher overhead = lower score
        format!(
            "Communication pattern overhead: {:.2}%, score: {:.2}",
            self.overhead * 100.0,
            score
        )
    }

    fn name(&self) -> &str {
        "CommunicationPatternAnalysis"
    }
}
