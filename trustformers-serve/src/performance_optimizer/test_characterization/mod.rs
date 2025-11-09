//! Test Characterization Module
//!
//! This module provides comprehensive test characterization, profiling, and analysis
//! capabilities for performance optimization.

pub mod concurrency_detector;
pub mod manager;
pub mod pattern_engine;
pub mod profiling_pipeline;
pub mod real_time_profiler;
pub mod resource_analyzer;
pub mod synchronization_analyzer;
pub mod types;

// Re-export commonly used types
pub use manager::TestCharacterizationEngine;
pub use pattern_engine::{SeverityLevel, TestPatternRecognitionEngine};
pub use real_time_profiler::ProfileDataPoint;
pub use synchronization_analyzer::SynchronizationAnalyzer;
pub use types::*;
