//! # PerformanceBottleneckDetector - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceBottleneckDetector`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `AntiPatternDetectionAlgorithm`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::performance_optimizer::test_characterization::types::{
    DetectedPattern, TestExecutionData,
};
use anyhow::Result;
use std::future::Future;
use std::pin::Pin;

use super::functions::AntiPatternDetectionAlgorithm;
use super::types::{DetectedAntiPattern, PerformanceBottleneckDetector};

impl Default for PerformanceBottleneckDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl AntiPatternDetectionAlgorithm for PerformanceBottleneckDetector {
    fn detect_anti_patterns(
        &self,
        _test_data: &TestExecutionData,
        _patterns: &[DetectedPattern],
    ) -> Pin<Box<dyn Future<Output = Result<Vec<DetectedAntiPattern>>> + Send + '_>> {
        Box::pin(async move { Ok(Vec::new()) })
    }
    fn name(&self) -> &str {
        "Performance Bottleneck Detector"
    }
    fn supported_types(&self) -> Vec<String> {
        vec!["performance_bottleneck".to_string()]
    }
}
