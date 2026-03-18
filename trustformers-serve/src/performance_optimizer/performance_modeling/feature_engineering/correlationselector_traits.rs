//! # CorrelationSelector - Trait Implementations
//!
//! This module contains trait implementations for `CorrelationSelector`.
//!
//! ## Implemented Traits
//!
//! - `FeatureSelector`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{anyhow, Result};
use std::collections::HashSet;

use super::functions::FeatureSelector;
use super::types::{CorrelationSelector, SelectionResult};

impl FeatureSelector for CorrelationSelector {
    fn select_features(
        &self,
        features: &[Vec<f64>],
        feature_names: &[String],
    ) -> Result<SelectionResult> {
        if features.is_empty() {
            return Err(anyhow!("No features provided for correlation selection"));
        }
        let n_features = features[0].len();
        let mut selected_indices = vec![0];
        let mut removed_features = HashSet::new();
        for i in 1..n_features {
            if removed_features.contains(&i) {
                continue;
            }
            let values_i: Vec<f64> = features.iter().map(|sample| sample[i]).collect();
            let mut should_keep = true;
            for &j in &selected_indices {
                let values_j: Vec<f64> = features.iter().map(|sample| sample[j]).collect();
                let correlation = self.calculate_correlation(&values_i, &values_j);
                if correlation.abs() > self.threshold {
                    should_keep = false;
                    break;
                }
            }
            if should_keep {
                selected_indices.push(i);
            } else {
                removed_features.insert(i);
            }
        }
        let selected_names = selected_indices.iter().map(|&i| feature_names[i].clone()).collect();
        let selected_count = selected_indices.len();
        Ok(SelectionResult {
            selected_indices,
            selected_names,
            selection_info: format!(
                "CorrelationSelector: Selected {}/{} features, removed {} highly correlated",
                selected_count,
                n_features,
                removed_features.len()
            ),
        })
    }
    fn name(&self) -> &str {
        "CorrelationSelector"
    }
}
