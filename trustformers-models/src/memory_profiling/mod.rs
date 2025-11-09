//! Memory Profiling Module for TrustformeRS Models
//!
//! This module provides comprehensive memory profiling and analysis capabilities for
//! transformer models. It includes real-time monitoring, alerts, and detailed
//! memory analysis tools.
//!
//! # Features
//!
//! - Real-time memory usage tracking
//! - Memory leak detection with advanced heuristics
//! - Allocation pattern analysis
//! - Peak memory analysis
//! - Memory fragmentation monitoring
//! - Historical data tracking with adaptive thresholds
//! - Automated alerts and recommendations
//! - Predictive memory usage analysis
//! - HTML and JSON report generation
//!
//! # Example
//!
//! ```rust
//! use trustformers_models::memory_profiling::{MemoryProfiler, ProfilerConfig};
//!
//! # tokio_test::block_on(async {
//! // Create and start the profiler
//! let config = ProfilerConfig::default();
//! let mut profiler = MemoryProfiler::new(config)?;
//!
//! // Start monitoring
//! profiler.start_monitoring().await?;
//!
//! // ... run your model training/inference ...
//!
//! // Generate comprehensive report
//! let report = profiler.generate_report().await?;
//! profiler.save_report(&report).await?;
//!
//! // Get analytics summary
//! let analytics = profiler.get_analytics_summary().await?;
//! println!("Memory efficiency: {:.1}%",
//!     report.summary.memory_efficiency_score * 100.0);
//! # Ok::<(), anyhow::Error>(())
//! # });
//! ```
//!
//! # Module Organization
//!
//! - [`types`] - Core data structures and configurations
//! - [`analytics`] - Advanced analytics components (adaptive thresholds, leak detection, prediction)
//! - [`profiler`] - Main memory profiler implementation
//! - [`system`] - System-level memory collection utilities
//! - [`reporting`] - Report generation and HTML formatting
//! - [`tests`] - Comprehensive test suite

// Public modules
pub mod analytics;
pub mod profiler;
pub mod reporting;
pub mod system;
pub mod types;

// Internal modules
#[cfg(test)]
mod tests;

// Re-export main types for convenience
pub use types::{
    AlertSeverity, AllocationInfo, MemoryAlert, MemoryAlertType, MemoryDashboardReport,
    MemoryMetrics, MemoryPattern, MemoryUsageSummary, PatternTrend, PatternType, ProfilerConfig,
    SystemInfo,
};

pub use analytics::{
    AdaptiveThresholds, AlertRecommendations, AnomalySeverity, AnomalyType, LeakAlert,
    LeakAlertType, MemoryAnomaly, MemoryPrediction, MemoryStatistics, StatisticalAnalyzer,
};

pub use profiler::{AnalyticsSummary, LeakDetectionConfig, MemoryProfiler, MonitoringStats};

pub use reporting::MemorySummary;
