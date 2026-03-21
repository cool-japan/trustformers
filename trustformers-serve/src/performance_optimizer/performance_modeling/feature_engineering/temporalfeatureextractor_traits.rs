//! # TemporalFeatureExtractor - Trait Implementations
//!
//! This module contains trait implementations for `TemporalFeatureExtractor`.
//!
//! ## Implemented Traits
//!
//! - `FeatureExtractor`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::performance_optimizer::types::PerformanceDataPoint;
use anyhow::Result;
use chrono::{Datelike, Timelike};

use super::functions::FeatureExtractor;
use super::types::{ExtractedFeatures, TemporalFeatureExtractor};

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
