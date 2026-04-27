//! Dynamic inference precision selection for TrustformeRS Serve.
//!
//! This module provides automatic selection between FP32, FP16, BF16, INT8,
//! and INT4 inference based on hardware capabilities, latency budgets, and
//! quality targets.

pub mod dynamic_precision;

pub use dynamic_precision::{
    DynamicPrecisionSelector, HardwareMetrics, InferencePrecision, PrecisionPolicy,
    PrecisionRecommendation,
};
