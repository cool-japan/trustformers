//! # LinearRegressionTrendDetector - Trait Implementations
//!
//! This module contains trait implementations for `LinearRegressionTrendDetector`.
//!
//! ## Implemented Traits
//!
//! - `TrendDetectionAlgorithm`
//! - `Clone`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::types::{TimestampedMetrics, TrendDirection};
use anyhow::Result;
use std::time::Duration;

use super::functions::TrendDetectionAlgorithm;
use super::types::{
    ForecastResult, LinearRegressionTrendDetector, TrendAnalysisConfig, TrendAnalysisResult,
    TrendInfo, TrendStrength,
};

impl TrendDetectionAlgorithm for LinearRegressionTrendDetector {
    fn analyze_trend(&self, data: &[TimestampedMetrics]) -> Result<TrendAnalysisResult> {
        let mut detector = self.clone();
        for metrics in data {
            let timestamp = metrics.timestamp.timestamp() as f64;
            let throughput = metrics.metrics.current_throughput;
            detector.add_data_point(timestamp, throughput);
        }

        // Inline the trend analysis logic to avoid method name collision
        let trend_info =
            if let Some((slope, _intercept, r_squared)) = detector.calculate_regression() {
                let direction = if slope > 0.0 {
                    TrendDirection::Increasing
                } else if slope < 0.0 {
                    TrendDirection::Decreasing
                } else {
                    TrendDirection::Stable
                };
                let strength = match r_squared {
                    r if r >= 0.8 => TrendStrength::Strong,
                    r if r >= 0.5 => TrendStrength::Moderate,
                    r if r >= 0.3 => TrendStrength::Weak,
                    _ => TrendStrength::None,
                };
                TrendInfo {
                    direction,
                    strength,
                    slope,
                    r_squared,
                    confidence: r_squared,
                    significance: r_squared > 0.5,
                }
            } else {
                TrendInfo {
                    direction: TrendDirection::Stable,
                    strength: TrendStrength::None,
                    slope: 0.0,
                    r_squared: 0.0,
                    confidence: 0.0,
                    significance: false,
                }
            };

        let throughput_trend = trend_info.clone();
        let latency_trend = trend_info.clone();
        let cpu_trend = trend_info.clone();
        let memory_trend = trend_info.clone();
        let overall_trend = trend_info;
        let analysis_confidence = overall_trend.confidence;
        let recommendation = self.generate_recommendation(&overall_trend);
        Ok(TrendAnalysisResult {
            throughput_trend,
            latency_trend,
            cpu_trend,
            memory_trend,
            overall_trend,
            analysis_confidence,
            recommendation,
        })
    }
    fn name(&self) -> &str {
        "linear_regression"
    }
    fn forecast(&self, data: &[TimestampedMetrics], horizon: Duration) -> Result<ForecastResult> {
        let (slope, intercept, r_squared) = self
            .calculate_regression()
            .ok_or_else(|| anyhow::anyhow!("Insufficient data for forecasting"))?;
        let last_timestamp = data.last().map(|m| m.timestamp.timestamp() as f64).unwrap_or(0.0);
        let forecast_timestamp = last_timestamp + horizon.as_secs_f64();
        let forecast_value = slope * forecast_timestamp + intercept;
        let std_error = (1.0 - r_squared).sqrt() * forecast_value.abs() * 0.1;
        Ok(ForecastResult {
            forecast_value,
            confidence_interval: (forecast_value - std_error, forecast_value + std_error),
            confidence_level: r_squared,
            horizon,
            algorithm: "linear_regression".to_string(),
        })
    }
    fn update_parameters(&mut self, config: &TrendAnalysisConfig) -> Result<()> {
        self.min_data_points = config.min_data_points;
        self.significance_threshold = (1.0 - config.sensitivity as f64) * 0.1;
        Ok(())
    }
}

impl Clone for LinearRegressionTrendDetector {
    fn clone(&self) -> Self {
        Self {
            min_data_points: self.min_data_points,
            significance_threshold: self.significance_threshold,
            data_points: self.data_points.clone(),
            max_data_points: self.max_data_points,
        }
    }
}
