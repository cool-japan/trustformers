//! # TrendAnalysisConfig - Trait Implementations
//!
//! This module contains trait implementations for `TrendAnalysisConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

use super::types::TrendAnalysisConfig;

impl Default for TrendAnalysisConfig {
    fn default() -> Self {
        Self {
            analysis_window: Duration::from_secs(7200),
            sensitivity: 0.7,
            min_data_points: 50,
            forecasting_enabled: true,
            forecast_horizon: Duration::from_secs(30 * 60),
            regression_enabled: true,
        }
    }
}
