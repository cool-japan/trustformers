//! # VarianceThresholdSelector - Trait Implementations
//!
//! This module contains trait implementations for `VarianceThresholdSelector`.
//!
//! ## Implemented Traits
//!
//! - `FeatureSelector`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{anyhow, Result};

use super::functions::FeatureSelector;
use super::types::{SelectionResult, VarianceThresholdSelector};

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
