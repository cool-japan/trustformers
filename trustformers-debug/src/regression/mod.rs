//! Performance regression detection for TrustformeRS debug.
//!
//! # Modules
//!
//! - [`detector`] — Baseline-comparison regression detector operating on full
//!   [`PerfMeasurement`] profiling records.
//! - [`statistical`] — Streaming statistical detectors: z-score / relative-change
//!   detector ([`StatRegressionDetector`]) and CUSUM change-point algorithm
//!   ([`CusumDetector`]).

pub mod detector;
pub mod statistical;

pub use detector::{
    BaselineStats, PerfBaseline, PerfMeasurement, RegressionAlert, RegressionConfig,
    RegressionDetector, RegressionMetric, RegressionSeverity,
};
pub use statistical::{
    ChangeDirection,
    CusumAlert,
    CusumDetector,
    StatBaselineStats,
    StatRegressionConfig,
    StatRegressionDetector,
    StatRegressionDirection,
    StatRegressionEvent,
    StatRegressionSeverity,
};
