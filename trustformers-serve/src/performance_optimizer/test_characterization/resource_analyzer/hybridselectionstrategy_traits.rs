//! # HybridSelectionStrategy - Trait Implementations
//!
//! This module contains trait implementations for `HybridSelectionStrategy`.
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
use super::types::{AlgorithmPerformanceRecord, HybridSelectionStrategy};

impl Default for HybridSelectionStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl SelectionStrategy for HybridSelectionStrategy {
    fn select_algorithm(
        &self,
        characteristics: &DataCharacteristics,
        performance_history: &HashMap<String, VecDeque<AlgorithmPerformanceRecord>>,
    ) -> Result<String> {
        let characteristic_choice = self
            .characteristic_strategy
            .select_algorithm(characteristics, performance_history)?;
        if performance_history.values().any(|records| records.len() >= 10) {
            let performance_choice = self
                .performance_strategy
                .select_algorithm(characteristics, performance_history)?;
            if let Some(records) = performance_history.get(&performance_choice) {
                if let Some(latest) = records.back() {
                    if latest.quality_score > 0.7 {
                        return Ok(performance_choice);
                    }
                }
            }
        }
        Ok(characteristic_choice)
    }
    fn name(&self) -> &str {
        "hybrid"
    }
    fn description(&self) -> &str {
        "Combines characteristic-based and performance-based selection strategies"
    }
}
