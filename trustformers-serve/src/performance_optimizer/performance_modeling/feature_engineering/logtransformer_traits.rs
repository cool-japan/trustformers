//! # LogTransformer - Trait Implementations
//!
//! This module contains trait implementations for `LogTransformer`.
//!
//! ## Implemented Traits
//!
//! - `FeatureTransformer`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;

use super::functions::FeatureTransformer;
use super::types::{LogTransformer, TransformationResult};

impl FeatureTransformer for LogTransformer {
    fn transform(&self, features: &[Vec<f64>]) -> Result<TransformationResult> {
        let transformed_data = features
            .iter()
            .map(|sample| {
                sample
                    .iter()
                    .map(|&value| if value > 0.0 { (value + 1.0).ln() } else { 0.0 })
                    .collect()
            })
            .collect();
        Ok(TransformationResult {
            transformed_data,
            transformation_info: "LogTransformer: Applied log(x+1) transformation".to_string(),
        })
    }
    fn fit_transform(&self, features: &[Vec<f64>]) -> Result<TransformationResult> {
        self.transform(features)
    }
    fn name(&self) -> &str {
        "LogTransformer"
    }
}
