//! # DependencyTracker - Trait Implementations
//!
//! This module contains trait implementations for `DependencyTracker`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

use super::types::{DependencyGraph, DependencyMetrics, DependencyTracker};

impl Default for DependencyTracker {
    fn default() -> Self {
        Self {
            _dependency_graph: Arc::new(RwLock::new(DependencyGraph::default())),
            _resolution_cache: Arc::new(Mutex::new(HashMap::new())),
            _blocked_tests: Arc::new(Mutex::new(HashMap::new())),
            _metrics: Arc::new(Mutex::new(DependencyMetrics::default())),
        }
    }
}
