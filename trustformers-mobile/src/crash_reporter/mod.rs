//! Mobile Crash Reporting and Analysis System
//!
//! This module provides comprehensive crash reporting capabilities for mobile ML applications,
//! including automatic crash detection, system state capture, crash analysis, and recovery
//! recommendations with privacy-aware data collection.
//!
//! Split into cohesive submodules: [`config`] (reporter/privacy/storage/platform
//! configuration), [`report_types`] (crash report data), [`analysis_types`]
//! (analysis/pattern/risk types), [`reporter`] (the `MobileCrashReporter`
//! engine) and [`engines`] (its analysis/storage/recovery helpers).

// For signal handling
extern crate libc;

pub mod analysis_types;
pub mod config;
pub mod engines;
pub mod report_types;
pub mod reporter;
#[cfg(test)]
mod tests;

// Re-export all types
pub use analysis_types::*;
pub use config::*;
pub use engines::*;
pub use report_types::*;
pub use reporter::*;
