//! # MovingAverageTrendDetector - Trait Implementations
//!
//! This module contains trait implementations for `MovingAverageTrendDetector`.
//!
//! ## Implemented Traits
//!
//! - `TrendDetectionAlgorithm`
//! - `Clone`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::types::TimestampedMetrics;
use super::types::*;
use anyhow::Result;

use super::functions::TrendDetectionAlgorithm;
use super::types::{
    ForecastResult, MovingAverageTrendDetector, TrendAnalysisConfig, TrendAnalysisResult,
};

impl TrendDetectionAlgorithm for MovingAverageTrendDetector {
    fn analyze_trend(&self, data: &[TimestampedMetrics]) -> Result<TrendAnalysisResult> {
        let mut detector = self.clone();
        let mut moving_averages = Vec::new();
        for metrics in data {
            if let Some(avg) = detector.add_value(metrics.metrics.current_throughput) {
                moving_averages.push(avg);
            }
        }
        let trend_info = detector.calculate_trend(&moving_averages);
        Ok(TrendAnalysisResult {
            throughput_trend: trend_info.clone(),
            latency_trend: trend_info.clone(),
            cpu_trend: trend_info.clone(),
            memory_trend: trend_info.clone(),
            overall_trend: trend_info.clone(),
            analysis_confidence: trend_info.confidence,
            recommendation: format!(
                "Moving average trend: {:?} with {:?} strength",
                trend_info.direction, trend_info.strength
            ),
        })
    }
    fn name(&self) -> &str {
        "moving_average"
    }
    fn forecast(&self, _data: &[TimestampedMetrics], horizon: Duration) -> Result<ForecastResult> {
        let current_avg = if self.values.len() == self.window_size {
            self.values.iter().sum::<f64>() / self.window_size as f64
        } else {
            0.0
        };
        Ok(ForecastResult {
            forecast_value: current_avg,
            confidence_interval: (current_avg * 0.9, current_avg * 1.1),
            confidence_level: 0.6,
            horizon,
            algorithm: "moving_average".to_string(),
        })
    }
    fn update_parameters(&mut self, config: &TrendAnalysisConfig) -> Result<()> {
        self.change_threshold = (1.0 - config.sensitivity as f64) * 0.1;
        Ok(())
    }
}

impl Clone for MovingAverageTrendDetector {
    fn clone(&self) -> Self {
        Self {
            window_size: self.window_size,
            values: self.values.clone(),
            change_threshold: self.change_threshold,
        }
    }
}
