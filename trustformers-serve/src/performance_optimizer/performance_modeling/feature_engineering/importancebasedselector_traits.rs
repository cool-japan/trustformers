//! # ImportanceBasedSelector - Trait Implementations
//!
//! This module contains trait implementations for `ImportanceBasedSelector`.
//!
//! ## Implemented Traits
//!
//! - `FeatureSelector`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;

use super::functions::FeatureSelector;
use super::types::{ImportanceBasedSelector, SelectionResult};

impl FeatureSelector for ImportanceBasedSelector {
    fn select_features(
        &self,
        features: &[Vec<f64>],
        feature_names: &[String],
    ) -> Result<SelectionResult> {
        let n_features = features[0].len();
        let mut feature_importance = Vec::new();
        for j in 0..n_features {
            let values: Vec<f64> = features.iter().map(|sample| sample[j]).collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance =
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
            feature_importance.push((j, variance));
        }
        feature_importance
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let max_importance = feature_importance[0].1;
        let selected_indices: Vec<usize> = feature_importance
            .iter()
            .filter_map(|(idx, importance)| {
                if importance / max_importance.max(1e-12) > self.threshold as f64 {
                    Some(*idx)
                } else {
                    None
                }
            })
            .collect();
        let selected_names = selected_indices.iter().map(|&i| feature_names[i].clone()).collect();
        let selected_count = selected_indices.len();
        Ok(SelectionResult {
            selected_indices,
            selected_names,
            selection_info: format!(
                "ImportanceBasedSelector: Selected {}/{} features above threshold {}",
                selected_count, n_features, self.threshold
            ),
        })
    }
    fn name(&self) -> &str {
        "ImportanceBasedSelector"
    }
}
