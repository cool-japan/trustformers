//! Auto-generated module structure

pub mod analyticsconfig_traits;
pub mod analyticsstats_traits;
pub mod analyzers;
pub mod functions;
pub mod performancethresholds_traits;
pub mod qualitythresholds_traits;
pub mod statisticalanalyzerconfig_traits;
pub mod statisticalanalyzerstats_traits;
pub mod types;
pub mod types_analysis;

// Re-export all types (types_analysis is re-exported via types.rs)
pub use analyzers::{
    AnomalyDetector, CorrelationAnalyzer, DistributionAnalyzer, ForecastingEngine, PatternAnalyzer,
    PerformanceAnalyzer, QualityAnalyzer, TrendAnalyzer,
};
pub use functions::*;
pub use types::*;

#[cfg(test)]
mod types_tests;
