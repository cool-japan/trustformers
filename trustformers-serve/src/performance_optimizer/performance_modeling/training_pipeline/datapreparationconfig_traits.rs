//! # DataPreparationConfig - Trait Implementations
//!
//! This module contains trait implementations for `DataPreparationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    DataPreparationConfig, DataQualityConfig, MissingValueStrategy, OutlierHandlingStrategy,
};

impl Default for DataPreparationConfig {
    fn default() -> Self {
        Self {
            train_test_split: 0.8,
            enable_data_augmentation: false,
            quality_checks: DataQualityConfig::default(),
            outlier_handling: OutlierHandlingStrategy::IQRClipping,
            missing_value_strategy: MissingValueStrategy::Mean,
        }
    }
}
