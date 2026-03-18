//! # MeanIntensityAlgorithm - Trait Implementations
//!
//! This module contains trait implementations for `MeanIntensityAlgorithm`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `IntensityCalculationAlgorithm`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::types::{
    DataCharacteristics, IntensityCalculationMethod, ResourceIntensity, ResourceUsageDataPoint,
};
use anyhow::Result;

use super::functions::IntensityCalculationAlgorithm;
use super::types::MeanIntensityAlgorithm;

impl Default for MeanIntensityAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl IntensityCalculationAlgorithm for MeanIntensityAlgorithm {
    fn calculate_intensity(
        &self,
        usage_data: &[ResourceUsageDataPoint],
    ) -> Result<ResourceIntensity> {
        if usage_data.is_empty() {
            return Ok(ResourceIntensity::default());
        }
        let cpu_sum: f64 = usage_data.iter().map(|p| p.snapshot.cpu_usage).sum();
        let memory_sum: f64 = usage_data.iter().map(|p| p.snapshot.memory_usage as f64).sum();
        let io_sum: f64 = usage_data
            .iter()
            .map(|p| p.snapshot.io_read_rate + p.snapshot.io_write_rate)
            .sum();
        let network_sum: f64 = usage_data
            .iter()
            .map(|p| p.snapshot.network_rx_rate + p.snapshot.network_tx_rate)
            .sum();
        let count = usage_data.len() as f64;
        let cpu_intensity = (cpu_sum / count).clamp(0.0, 1.0);
        let memory_intensity = (memory_sum / count / 1_000_000_000.0).clamp(0.0, 1.0);
        let io_intensity = (io_sum / count / 10_000_000.0).clamp(0.0, 1.0);
        let network_intensity = (network_sum / count / 1_000_000.0).clamp(0.0, 1.0);
        let overall_intensity =
            (cpu_intensity + memory_intensity + io_intensity + network_intensity) / 4.0;
        let cpu_values: Vec<f64> = usage_data.iter().map(|p| p.snapshot.cpu_usage).collect();
        let variance = if cpu_values.len() > 1 {
            let mean = cpu_sum / count;
            let var_sum: f64 = cpu_values.iter().map(|&v| (v - mean).powi(2)).sum();
            var_sum / count
        } else {
            0.0
        };
        Ok(ResourceIntensity {
            cpu_intensity,
            memory_intensity,
            io_intensity,
            network_intensity,
            gpu_intensity: 0.0,
            overall_intensity,
            peak_periods: Vec::new(),
            usage_variance: variance,
            baseline_comparison: 1.0,
            calculation_method: IntensityCalculationMethod::MovingAverage,
        })
    }
    fn name(&self) -> &str {
        "mean"
    }
    fn description(&self) -> &str {
        "Arithmetic mean-based intensity calculation for stable workloads"
    }
    fn is_suitable_for(&self, characteristics: &DataCharacteristics) -> bool {
        characteristics.variance < 0.3 && characteristics.trend_strength < 0.5
    }
}
