//! Free functions shared across the analytics module.
//!
//! ## Removed in 0.2.1: `create_analyzer_placeholder!`
//!
//! This file used to define a `create_analyzer_placeholder!` macro and invoke
//! it six times to generate `DistributionAnalyzer`, `CorrelationAnalyzer`,
//! `ForecastingEngine`, `QualityAnalyzer`, `PatternAnalyzer` and
//! `PerformanceAnalyzer`. Every generated `analyze` had the body
//! `pub async fn analyze(&self, _data: &[TimestampedMetrics]) -> Result<T> { Ok($default_result) }`
//! — the sampled data was discarded and a hand-written literal was returned:
//! Shapiro-Wilk `statistic: 0.95`, latency `p99: 100.0`, `current_availability:
//! 0.999`, `mttr: 300s`, `compliance_rate: 0.95`, `mape: 0.05`, and so on.
//! `AnalyticsEngine::analyze` then combined those constants into an
//! `AnalyticsResult` with a `confidence` score, which is exactly what a caller
//! would read as a measurement.
//!
//! The macro and all six literals are deleted. The analyzers now live in
//! `analytics::analyzers` and compute their results from the samples they are
//! given. See that module's documentation for what each one measures and which
//! result fields are `None` because no metric stream can support them.

pub use super::analyzers::series::erf;
