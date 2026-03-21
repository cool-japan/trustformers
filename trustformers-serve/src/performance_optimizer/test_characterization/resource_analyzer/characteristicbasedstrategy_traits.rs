//! # CharacteristicBasedStrategy - Trait Implementations
//!
//! This module contains trait implementations for `CharacteristicBasedStrategy`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `SelectionStrategy`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::types::DataCharacteristics;
use anyhow::Result;
use std::collections::{HashMap, VecDeque};

use super::functions::SelectionStrategy;
use super::types::{AlgorithmPerformanceRecord, CharacteristicBasedStrategy};

impl Default for CharacteristicBasedStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl SelectionStrategy for CharacteristicBasedStrategy {
    fn select_algorithm(
        &self,
        characteristics: &DataCharacteristics,
        _performance_history: &HashMap<String, VecDeque<AlgorithmPerformanceRecord>>,
    ) -> Result<String> {
        if characteristics.variance < 0.2 && characteristics.trend_strength < 0.3 {
            Ok("mean".to_string())
        } else if characteristics.trend_strength > 0.5 {
            Ok("weighted".to_string())
        } else if characteristics.variance > 0.6 || characteristics.outlier_percentage > 0.1 {
            Ok("exponential".to_string())
        } else if characteristics.outlier_percentage > 0.05 {
            Ok("peak".to_string())
        } else {
            Ok("adaptive".to_string())
        }
    }
    fn name(&self) -> &str {
        "characteristic_based"
    }
    fn description(&self) -> &str {
        "Selects algorithms based on data characteristics analysis"
    }
}
