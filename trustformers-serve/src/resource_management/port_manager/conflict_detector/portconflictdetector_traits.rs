//! # PortConflictDetector - Trait Implementations
//!
//! This module contains trait implementations for `PortConflictDetector`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::sync::{Arc, atomic::AtomicBool};

use parking_lot::{Mutex, RwLock};
use super::types::*;
use crate::resource_management::types::*;

use super::types::{ConflictDetectorConfig, ConflictPriorityThresholds, ConflictStatistics, PortConflictDetector};

impl Default for PortConflictDetector {
    fn default() -> Self {
        Self {
            conflict_rules: Arc::new(RwLock::new(Self::create_default_rules())),
            enabled: AtomicBool::new(true),
            conflict_history: Arc::new(Mutex::new(Vec::new())),
            basic_conflict_history: Arc::new(Mutex::new(Vec::new())),
            config: Arc::new(RwLock::new(ConflictDetectorConfig::default())),
            priority_thresholds: Arc::new(
                RwLock::new(ConflictPriorityThresholds::default()),
            ),
            statistics: Arc::new(Mutex::new(ConflictStatistics::default())),
        }
    }
}

