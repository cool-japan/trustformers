//! Utilities module for common debugging operations and helper functions
//!
//! This module provides a comprehensive set of debugging utilities organized into focused submodules:
//!
//! - `health`: Health checking and diagnostic utilities
//! - `tensor_analysis`: Tensor analysis and statistical functions
//! - `performance`: Performance monitoring and profiling tools
//! - More modules to be added as the utilities.rs file is further refactored

pub mod health;
pub mod performance;
pub mod tensor_analysis;
pub mod weight_analysis;

// Re-export main types and functions for backward compatibility
pub use health::*;
pub use performance::*;
pub use tensor_analysis::*;
pub use weight_analysis::*;

use anyhow::Result;
use scirs2_core::ndarray; // SciRS2 Integration Policy

/// Common debugging utilities and helper functions
pub struct DebugUtils;

impl DebugUtils {
    /// Quick model health check with automatic issue detection
    pub async fn quick_health_check<T>(model: &T) -> Result<HealthCheckResult> {
        HealthChecker::quick_health_check(model).await
    }

    /// Convert health score to status string
    pub fn score_to_status(score: f64) -> String {
        HealthChecker::score_to_status(score)
    }

    /// Generate debug report summary
    pub fn generate_debug_summary(
        config: &crate::DebugConfig,
        results: &[crate::SimplifiedDebugResult],
    ) -> DebugSummary {
        HealthChecker::generate_debug_summary(config, results)
    }

    /// Export debug data in various formats
    pub async fn export_debug_data(
        session: &crate::DebugSession,
        format: ExportFormat,
        output_path: &str,
    ) -> Result<String> {
        HealthChecker::export_debug_data(session, format, output_path).await
    }

    /// Create a debug session template for common use cases
    pub fn create_debug_template(template_type: DebugTemplate) -> crate::DebugConfig {
        HealthChecker::create_debug_template(template_type)
    }

    /// Batch tensor analysis with statistical insights
    pub fn analyze_tensors_batch(tensors: &[ndarray::ArrayD<f32>]) -> Result<BatchTensorAnalysis> {
        TensorAnalyzer::analyze_tensors_batch(tensors)
    }

    /// Compute comprehensive statistics for a tensor
    pub fn compute_tensor_statistics(tensor: &ndarray::ArrayD<f32>) -> Result<TensorStatistics> {
        TensorAnalyzer::compute_tensor_statistics(tensor)
    }

    /// Detect anomalies in tensor statistics
    pub fn detect_tensor_anomalies(stats: &TensorStatistics) -> Vec<TensorAnomaly> {
        TensorAnalyzer::detect_tensor_anomalies(stats)
    }

    /// Compare tensors for drift detection
    pub fn compare_tensors(
        baseline: &ndarray::ArrayD<f32>,
        current: &ndarray::ArrayD<f32>,
    ) -> Result<TensorComparisonResult> {
        TensorAnalyzer::compare_tensors(baseline, current)
    }

    /// Hash configuration for tracking
    pub fn hash_config(config: &crate::DebugConfig) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        format!("{:?}", config).hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

/// Debugging convenience macros
#[macro_export]
macro_rules! debug_tensor {
    ($session:expr, $tensor:expr, $name:expr) => {
        $session.debug_tensor($tensor, $name)
    };
    ($tensor:expr) => {
        trustformers_debug::utilities::TensorAnalyzer::compute_tensor_statistics($tensor)
    };
}

#[macro_export]
macro_rules! debug_gradient {
    ($session:expr, $layer_name:expr, $gradients:expr) => {
        $session.debug_gradients($layer_name, $gradients)
    };
}

#[macro_export]
macro_rules! quick_debug {
    ($model:expr, $level:expr) => {
        trustformers_debug::quick_debug($model, $level).await
    };
}

#[macro_export]
macro_rules! perf_checkpoint {
    ($monitor:expr, $name:expr) => {
        $monitor.checkpoint($name)
    };
}

#[macro_export]
macro_rules! perf_end_checkpoint {
    ($monitor:expr, $name:expr) => {
        $monitor.end_checkpoint($name)
    };
}
