//! # StandardScaler - Trait Implementations
//!
//! This module contains trait implementations for `StandardScaler`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `FeatureTransformer`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{anyhow, Result};

use super::functions::FeatureTransformer;
use super::types::{StandardScaler, TransformationResult};

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureTransformer for StandardScaler {
    fn transform(&self, features: &[Vec<f64>]) -> Result<TransformationResult> {
        let means = self.means.as_ref().ok_or_else(|| anyhow!("StandardScaler not fitted"))?;
        let stds = self.stds.as_ref().ok_or_else(|| anyhow!("StandardScaler not fitted"))?;
        let transformed_data = features
            .iter()
            .map(|sample| {
                sample
                    .iter()
                    .enumerate()
                    .map(|(j, &value)| (value - means[j]) / stds[j])
                    .collect()
            })
            .collect();
        Ok(TransformationResult {
            transformed_data,
            transformation_info: format!("StandardScaler: {} features scaled", means.len()),
        })
    }
    fn fit_transform(&self, features: &[Vec<f64>]) -> Result<TransformationResult> {
        let mut scaler = Self::new();
        scaler.calculate_statistics(features)?;
        scaler.transform(features)
    }
    fn name(&self) -> &str {
        "StandardScaler"
    }
}
