//! # TrendAnalysisAlgorithm - Trait Implementations
//!
//! This module contains trait implementations for `TrendAnalysisAlgorithm`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `ThresholdAdaptationAlgorithm`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::super::types::{AlertEvent, TimestampedMetrics};
use super::super::adaptive_controller::ThresholdAdaptationAlgorithm;
use super::super::error::Result;
use chrono::Utc;

use super::types::{TrendAnalysisAlgorithm, TrendDirection};

impl Default for TrendAnalysisAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl ThresholdAdaptationAlgorithm for TrendAnalysisAlgorithm {
    fn adapt_threshold(
        &self,
        current_threshold: f64,
        metrics_history: &[TimestampedMetrics],
        _alert_history: &[AlertEvent],
    ) -> Result<f64> {
        if metrics_history.len() < self.config.trend_window {
            return Ok(current_threshold);
        }
        let values: Vec<f64> = metrics_history
            .iter()
            .filter_map(|m| m.metrics.values().first().copied())
            .collect();
        if values.is_empty() {
            return Ok(current_threshold);
        }
        let (trend_direction, trend_strength) = self.analyze_trend(&values);
        let seasonal_patterns = if self.config.seasonal_detection {
            self.detect_seasonal_patterns(&values)
        } else {
            Vec::new()
        };
        {
            let mut state = self.trend_state.lock().expect("Trend state lock poisoned");
            state.current_trend = trend_direction;
            state.trend_strength = trend_strength;
            state.seasonal_patterns = seasonal_patterns;
            state.last_analysis = Utc::now();
        }
        let adaptation_factor =
            match self.trend_state.lock().expect("Trend state lock poisoned").current_trend {
                TrendDirection::Increasing => 1.0 + trend_strength as f64 * 0.2,
                TrendDirection::Decreasing => 1.0 - trend_strength as f64 * 0.1,
                TrendDirection::Stable => 1.0,
                TrendDirection::Cyclical => 1.0 + trend_strength as f64 * 0.1,
            };
        let adapted_threshold = current_threshold * adaptation_factor;
        Ok(adapted_threshold.max(0.0))
    }
    fn name(&self) -> &str {
        "trend_analysis_adaptation"
    }
    fn confidence(&self, data_quality: f32) -> f32 {
        let state = self.trend_state.lock().expect("Trend state lock poisoned");
        let trend_confidence = state.trend_strength.min(1.0);
        let seasonal_confidence =
            state.seasonal_patterns.iter().map(|p| p.confidence).fold(0.0f32, f32::max);
        (trend_confidence + seasonal_confidence * 0.5).min(1.0) * data_quality
    }
}
