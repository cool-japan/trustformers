//! # AggregationFunction - Trait Implementations
//!
//! This module contains trait implementations for `AggregationFunction`.
//!
//! ## Implemented Traits
//!
//! - `Hash`
//! - `Eq`
//! - `PartialEq`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use std::mem;

impl std::hash::Hash for AggregationFunction {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        mem::discriminant(self).hash(state);
        match self {
            AggregationFunction::Percentile(p) => {
                p.to_bits().hash(state);
            },
            AggregationFunction::Custom(s) => s.hash(state),
            _ => {},
        }
    }
}

impl Eq for AggregationFunction {}

impl PartialEq for AggregationFunction {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (AggregationFunction::Percentile(a), AggregationFunction::Percentile(b)) => {
                a.to_bits() == b.to_bits()
            },
            (AggregationFunction::Custom(a), AggregationFunction::Custom(b)) => a == b,
            _ => mem::discriminant(self) == mem::discriminant(other),
        }
    }
}
