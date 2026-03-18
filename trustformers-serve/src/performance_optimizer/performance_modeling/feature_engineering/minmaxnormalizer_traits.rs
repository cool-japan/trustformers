//! # MinMaxNormalizer - Trait Implementations
//!
//! This module contains trait implementations for `MinMaxNormalizer`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `FeatureTransformer`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{anyhow, Result};

use super::functions::FeatureTransformer;
use super::types::{MinMaxNormalizer, TransformationResult};

impl Default for MinMaxNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureTransformer for MinMaxNormalizer {
    fn transform(&self, features: &[Vec<f64>]) -> Result<TransformationResult> {
        let mins = self.mins.as_ref().ok_or_else(|| anyhow!("MinMaxNormalizer not fitted"))?;
        let maxs = self.maxs.as_ref().ok_or_else(|| anyhow!("MinMaxNormalizer not fitted"))?;
        let transformed_data = features
            .iter()
            .map(|sample| {
                sample
                    .iter()
                    .enumerate()
                    .map(|(j, &value)| (value - mins[j]) / (maxs[j] - mins[j]))
                    .collect()
            })
            .collect();
        Ok(TransformationResult {
            transformed_data,
            transformation_info: format!("MinMaxNormalizer: {} features normalized", mins.len()),
        })
    }
    fn fit_transform(&self, features: &[Vec<f64>]) -> Result<TransformationResult> {
        let mut normalizer = Self::new();
        normalizer.calculate_bounds(features)?;
        normalizer.transform(features)
    }
    fn name(&self) -> &str {
        "MinMaxNormalizer"
    }
}
