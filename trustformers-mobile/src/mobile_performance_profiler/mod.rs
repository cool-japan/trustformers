//! Mobile Performance Profiler for Debugging
//!
//! This module provides comprehensive performance profiling and debugging
//! capabilities specifically designed for mobile ML inference, including
//! real-time monitoring, bottleneck detection, and optimization suggestions.
//!
//! ## Architecture
//!
//! The mobile performance profiler is organized into several focused modules:
//! - [`types`] - Core types, metrics structures, and shared data models
//! - [`config`] - Configuration management for profiling parameters
//! - [`metrics`] - Metrics collection, snapshots, and aggregation
//! - [`bottleneck_detection`] - Performance bottleneck detection and analysis
//! - [`optimization`] - Optimization engine and suggestion generation
//! - [`monitoring`] - Real-time monitoring, alerting, and event handling
//! - [`export`] - Data export, visualization, and reporting
//! - [`session`] - Session management and profiling lifecycle
//! - [`analysis`] - Performance analysis, trending, and health assessment
//! - [`profiler`] - Main profiler coordinator and public API

pub mod analysis;
pub mod bottleneck_detection;
pub mod collector;
pub mod config;
pub mod export;
pub mod metrics;
pub mod monitoring;
pub mod optimization;
pub mod profiler;
pub mod session;
pub mod simd_analytics;
pub mod types;

// Re-export main types for backward compatibility
pub use profiler::MobilePerformanceProfiler;
pub use types::*;

// Re-export component types for easy access
pub use analysis::{HealthStatus, PerformanceAnalyzer, SystemHealth, TrendingMetrics};
pub use bottleneck_detection::{BottleneckDetector, BottleneckType, PerformanceBottleneck};
pub use collector::{CollectionStatistics, MobileMetricsCollector};
pub use config::ExportConfig;
pub use config::{CpuProfilingConfig, MemoryProfilingConfig, MobileProfilerConfig, SamplingConfig};
pub use export::{ExportFormat, ProfilerExportManager, VisualizationEngine};
pub use monitoring::{AlertManager, AlertType, PerformanceAlert, RealTimeMonitor};
pub use optimization::OptimizationEngine;
pub use session::{ProfilingSession, SessionInfo, SessionMetadata};
pub use simd_analytics::{
    AnomalyScore, CrossMetricRelationship, SimdAnalyticsConfig, SimdPerformanceAnalytics,
    SimdPerformanceStats,
};
pub use types::MobileMetricsSnapshot;
pub use types::{InferenceMetrics, MemoryMetrics};
pub use types::{OptimizationSuggestion, SuggestionType};
