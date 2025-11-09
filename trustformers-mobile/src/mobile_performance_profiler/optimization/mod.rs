//! Optimization Engine Module
//!
//! This module provides intelligent optimization suggestion generation for mobile
//! ML inference workloads.

pub mod engine;

pub use engine::{
    BatteryContext, CPUUsagePattern, CacheType, ImpactEstimator, MemoryUsagePattern, NetworkType,
    OptimizationCondition, OptimizationEngine, OptimizationEngineStats, OptimizationEvent,
    OptimizationRule, SuggestionRanker, WorkloadType,
};
