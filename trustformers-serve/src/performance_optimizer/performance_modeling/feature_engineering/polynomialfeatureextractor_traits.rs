//! # PolynomialFeatureExtractor - Trait Implementations
//!
//! This module contains trait implementations for `PolynomialFeatureExtractor`.
//!
//! ## Implemented Traits
//!
//! - `FeatureExtractor`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::performance_optimizer::types::PerformanceDataPoint;
use anyhow::Result;

use super::functions::FeatureExtractor;
use super::types::{ExtractedFeatures, PolynomialFeatureExtractor};

impl FeatureExtractor for PolynomialFeatureExtractor {
    fn extract_features(&self, data_points: &[PerformanceDataPoint]) -> Result<ExtractedFeatures> {
        if self.degree < 2 {
            return Ok(ExtractedFeatures {
                features: Vec::new(),
                names: Vec::new(),
            });
        }
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
