//! # PerformanceBasedStrategy - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceBasedStrategy`.
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
use super::types::{AlgorithmPerformanceRecord, PerformanceBasedStrategy};

impl Default for PerformanceBasedStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl SelectionStrategy for PerformanceBasedStrategy {
    fn select_algorithm(
        &self,
        _characteristics: &DataCharacteristics,
        performance_history: &HashMap<String, VecDeque<AlgorithmPerformanceRecord>>,
    ) -> Result<String> {
        let mut best_algorithm = "adaptive".to_string();
        let mut best_score = 0.0;
        for (algorithm_id, records) in performance_history {
            if records.is_empty() {
                continue;
            }
            let avg_duration =
                records.iter().map(|r| r.execution_duration.as_secs_f64()).sum::<f64>()
                    / records.len() as f64;
            let avg_quality =
                records.iter().map(|r| r.quality_score).sum::<f64>() / records.len() as f64;
            let score = avg_quality * 0.7 + (1.0 / (1.0 + avg_duration)) * 0.3;
            if score > best_score {
                best_score = score;
                best_algorithm = algorithm_id.clone();
            }
        }
        Ok(best_algorithm)
    }
    fn name(&self) -> &str {
        "performance_based"
    }
    fn description(&self) -> &str {
        "Selects algorithms based on historical performance metrics"
    }
}
