//! # ExponentialSmoothingAdaptation - Trait Implementations
//!
//! This module contains trait implementations for `ExponentialSmoothingAdaptation`.
//!
//! ## Implemented Traits
//!
//! - `BaselineAdaptationAlgorithm`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::types::TimestampedMetrics;
use super::types::*;
use anyhow::Result;
use chrono::Utc;

use super::functions::BaselineAdaptationAlgorithm;
use super::types::{
    BaselineConfig, BaselineValidationResult, ExponentialSmoothingAdaptation, PerformanceBaseline,
};

impl BaselineAdaptationAlgorithm for ExponentialSmoothingAdaptation {
    fn adapt_baseline(
        &self,
        current: &PerformanceBaseline,
        new_data: &[TimestampedMetrics],
    ) -> Result<PerformanceBaseline> {
        if new_data.is_empty() {
            return Ok(current.clone());
        }
        let new_throughput = new_data.iter().map(|m| m.metrics.current_throughput).sum::<f64>()
            / new_data.len() as f64;
        let new_latency_ms = new_data
            .iter()
            .map(|m| m.metrics.current_latency.as_secs_f64() * 1000.0)
            .sum::<f64>()
            / new_data.len() as f64;
        let new_cpu =
            new_data.iter().map(|m| m.metrics.current_cpu_utilization as f64).sum::<f64>()
                / new_data.len() as f64;
        let new_memory = new_data
            .iter()
            .map(|m| m.metrics.current_memory_utilization as f64)
            .sum::<f64>()
            / new_data.len() as f64;
        let smoothed_throughput =
            self.alpha * new_throughput + (1.0 - self.alpha) * current.baseline_throughput;
        let smoothed_latency = Duration::from_secs_f64(
            (self.alpha * new_latency_ms
                + (1.0 - self.alpha) * current.baseline_latency.as_secs_f64() * 1000.0)
                / 1000.0,
        );
        let smoothed_cpu =
            (self.alpha * new_cpu + (1.0 - self.alpha) * current.baseline_cpu as f64) as f32;
        let smoothed_memory =
            (self.alpha * new_memory + (1.0 - self.alpha) * current.baseline_memory as f64) as f32;
        let mut updated_baseline = current.clone();
        updated_baseline.baseline_throughput = smoothed_throughput;
        updated_baseline.baseline_latency = smoothed_latency;
        updated_baseline.baseline_cpu = smoothed_cpu;
        updated_baseline.baseline_memory = smoothed_memory;
        updated_baseline.timestamp = Utc::now();
        updated_baseline.version += 1;
        updated_baseline.sample_size += new_data.len();
        let adaptation_magnitude = ((smoothed_throughput - current.baseline_throughput).abs()
            / current.baseline_throughput.max(1.0)
            + (smoothed_latency.as_secs_f64() - current.baseline_latency.as_secs_f64()).abs()
                / current.baseline_latency.as_secs_f64().max(0.001)
            + (smoothed_cpu - current.baseline_cpu).abs() as f64
                / current.baseline_cpu.max(0.01) as f64
            + (smoothed_memory - current.baseline_memory).abs() as f64
                / current.baseline_memory.max(0.01) as f64)
            / 4.0;
        updated_baseline.quality_score = (current.quality_score * 0.9
            + (1.0 - adaptation_magnitude.min(1.0)) as f32 * 0.1)
            .clamp(0.0, 1.0);
        Ok(updated_baseline)
    }
    fn name(&self) -> &str {
        "exponential_smoothing"
    }
    fn validate_baseline(&self, baseline: &PerformanceBaseline) -> BaselineValidationResult {
        if baseline.quality_score >= 0.8 && baseline.stability_score >= 0.7 {
            BaselineValidationResult::Valid
        } else if baseline.quality_score >= 0.6 {
            BaselineValidationResult::NeedsRefresh
        } else {
            BaselineValidationResult::Invalid
        }
    }
    fn update_parameters(&mut self, config: &BaselineConfig) -> Result<()> {
        self.alpha = config.adaptation_rate.clamp(0.0, 1.0) as f64;
        self.beta = (config.adaptation_rate * 0.5).clamp(0.0, 1.0) as f64;
        Ok(())
    }
}
