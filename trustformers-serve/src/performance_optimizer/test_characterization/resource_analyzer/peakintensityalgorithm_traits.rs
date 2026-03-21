//! # PeakIntensityAlgorithm - Trait Implementations
//!
//! This module contains trait implementations for `PeakIntensityAlgorithm`.
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
use std::time::{Duration, Instant};

use super::functions::IntensityCalculationAlgorithm;
use super::types::PeakIntensityAlgorithm;

impl Default for PeakIntensityAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl IntensityCalculationAlgorithm for PeakIntensityAlgorithm {
    fn calculate_intensity(
        &self,
        usage_data: &[ResourceUsageDataPoint],
    ) -> Result<ResourceIntensity> {
        if usage_data.is_empty() {
            return Ok(ResourceIntensity::default());
        }
        let cpu_peak = usage_data.iter().map(|p| p.snapshot.cpu_usage).fold(0.0f64, f64::max);
        let memory_peak =
            usage_data.iter().map(|p| p.snapshot.memory_usage as f64).fold(0.0f64, f64::max);
        let io_peak = usage_data
            .iter()
            .map(|p| p.snapshot.io_read_rate + p.snapshot.io_write_rate)
            .fold(0.0f64, f64::max);
        let network_peak = usage_data
            .iter()
            .map(|p| p.snapshot.network_rx_rate + p.snapshot.network_tx_rate)
            .fold(0.0f64, f64::max);
        let cpu_intensity = cpu_peak.clamp(0.0, 1.0);
        let memory_intensity = (memory_peak / 1_000_000_000.0).clamp(0.0, 1.0);
        let io_intensity = (io_peak / 10_000_000.0).clamp(0.0, 1.0);
        let network_intensity = (network_peak / 1_000_000.0).clamp(0.0, 1.0);
        let overall_intensity =
            (cpu_intensity + memory_intensity + io_intensity + network_intensity) / 4.0;
        let cpu_values: Vec<f64> = usage_data.iter().map(|p| p.snapshot.cpu_usage).collect();
        let mean_cpu = cpu_values.iter().sum::<f64>() / cpu_values.len() as f64;
        let variance = if cpu_values.len() > 1 {
            let var_sum: f64 = cpu_values.iter().map(|&v| (v - mean_cpu).powi(2)).sum();
            var_sum / cpu_values.len() as f64
        } else {
            0.0
        };
        let mut sorted_cpu = cpu_values.clone();
        sorted_cpu.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let percentile_80_idx = (sorted_cpu.len() as f64 * 0.8) as usize;
        let threshold = sorted_cpu.get(percentile_80_idx).copied().unwrap_or(0.0);
        let peak_periods: Vec<(Instant, Duration)> = usage_data
            .iter()
            .filter(|p| p.snapshot.cpu_usage >= threshold)
            .map(|p| (p.timestamp, Duration::from_millis(100)))
            .collect();
        Ok(ResourceIntensity {
            cpu_intensity,
            memory_intensity,
            io_intensity,
            network_intensity,
            gpu_intensity: 0.0,
            overall_intensity,
            peak_periods,
            usage_variance: variance,
            baseline_comparison: 1.0,
            calculation_method: IntensityCalculationMethod::Peak,
        })
    }
    fn name(&self) -> &str {
        "peak"
    }
    fn description(&self) -> &str {
        "Peak-based intensity for capacity planning and worst-case analysis"
    }
    fn is_suitable_for(&self, characteristics: &DataCharacteristics) -> bool {
        characteristics.outlier_percentage > 0.05
    }
}
