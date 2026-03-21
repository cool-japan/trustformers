//! Bottleneck Detection Module
//!
//! This module provides comprehensive performance bottleneck detection capabilities
//! for mobile ML inference workloads.

pub mod detector;

pub use detector::{
    BottleneckCondition, BottleneckDetectionEvent, BottleneckDetectionStats, BottleneckDetector,
    BottleneckRule, HistoricalAnalyzer, SeverityCalculator,
};