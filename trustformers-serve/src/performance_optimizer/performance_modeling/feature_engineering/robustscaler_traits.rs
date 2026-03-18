//! # RobustScaler - Trait Implementations
//!
//! This module contains trait implementations for `RobustScaler`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `FeatureTransformer`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{anyhow, Result};

use super::functions::FeatureTransformer;
use super::types::{RobustScaler, TransformationResult};

impl Default for RobustScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureTransformer for RobustScaler {
    fn transform(&self, features: &[Vec<f64>]) -> Result<TransformationResult> {
        let medians = self.medians.as_ref().ok_or_else(|| anyhow!("RobustScaler not fitted"))?;
        let iqrs = self.iqrs.as_ref().ok_or_else(|| anyhow!("RobustScaler not fitted"))?;
        let transformed_data = features
            .iter()
            .map(|sample| {
                sample
                    .iter()
                    .enumerate()
                    .map(|(j, &value)| (value - medians[j]) / iqrs[j])
                    .collect()
            })
            .collect();
        Ok(TransformationResult {
            transformed_data,
            transformation_info: format!(
                "RobustScaler: {} features scaled using median and IQR",
                medians.len()
            ),
        })
    }
    fn fit_transform(&self, features: &[Vec<f64>]) -> Result<TransformationResult> {
        let mut scaler = Self::new();
        scaler.calculate_robust_statistics(features)?;
        scaler.transform(features)
    }
    fn name(&self) -> &str {
        "RobustScaler"
    }
}
