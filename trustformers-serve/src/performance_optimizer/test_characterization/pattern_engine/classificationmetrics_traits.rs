//! # ClassificationMetrics - Trait Implementations
//!
//! This module contains trait implementations for `ClassificationMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::types::ClassificationMetrics;

impl Default for ClassificationMetrics {
    fn default() -> Self {
        Self {
            total_classifications: 0,
            successful_classifications: 0,
            failed_classifications: 0,
            avg_classification_time: Duration::default(),
            category_distribution: HashMap::new(),
            accuracy_by_category: HashMap::new(),
            last_updated: Instant::now(),
        }
    }
}
