//! Advanced Mobile Performance Profiler
//!
//! This module provides comprehensive performance profiling capabilities for mobile ML
//! workloads, integrating with platform-specific tools and providing detailed performance
//! analysis, bottleneck detection, and optimization recommendations.
//!
//! Split into cohesive submodules: [`config_types`], [`metrics_types`] (raw
//! collected metrics), [`analysis_types`] (bottleneck/regression/trend/alert
//! results), [`analysis_engines`] (the engines producing those results),
//! [`collector`] (metrics collection and sessions), [`platform_profilers`]
//! (iOS/Android/generic backends), [`engine`] (the `MobilePerformanceProfiler`
//! orchestrator) and [`utils`].

pub mod analysis_engines;
pub mod analysis_types;
pub mod collector;
pub mod config_types;
pub mod engine;
pub mod metrics_types;
pub mod platform_profilers;
#[cfg(test)]
mod tests;
pub mod utils;

// Re-export all types
pub use analysis_engines::*;
pub use analysis_types::*;
pub use collector::*;
pub use config_types::*;
pub use engine::*;
pub use metrics_types::*;
pub use platform_profilers::*;
pub use utils::*;
