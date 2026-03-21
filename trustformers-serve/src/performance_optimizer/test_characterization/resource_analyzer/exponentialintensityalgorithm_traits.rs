//! # ExponentialIntensityAlgorithm - Trait Implementations
//!
//! This module contains trait implementations for `ExponentialIntensityAlgorithm`.
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
use super::types::ExponentialIntensityAlgorithm;

impl Default for ExponentialIntensityAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl IntensityCalculationAlgorithm for ExponentialIntensityAlgorithm {
    fn calculate_intensity(
        &self,
        usage_data: &[ResourceUsageDataPoint],
    ) -> Result<ResourceIntensity> {
        if usage_data.is_empty() {
            return Ok(ResourceIntensity::default());
        }
        let mut cpu_ema = usage_data[0].snapshot.cpu_usage;
        let mut memory_ema = usage_data[0].snapshot.memory_usage as f64;
        let mut io_ema = usage_data[0].snapshot.io_read_rate + usage_data[0].snapshot.io_write_rate;
        let mut network_ema =
            usage_data[0].snapshot.network_rx_rate + usage_data[0].snapshot.network_tx_rate;
        for point in usage_data.iter().skip(1) {
            cpu_ema =
                self.decay_factor * point.snapshot.cpu_usage + (1.0 - self.decay_factor) * cpu_ema;
            memory_ema = self.decay_factor * point.snapshot.memory_usage as f64
                + (1.0 - self.decay_factor) * memory_ema;
            io_ema = self.decay_factor
                * (point.snapshot.io_read_rate + point.snapshot.io_write_rate)
                + (1.0 - self.decay_factor) * io_ema;
            network_ema = self.decay_factor
                * (point.snapshot.network_rx_rate + point.snapshot.network_tx_rate)
                + (1.0 - self.decay_factor) * network_ema;
        }
        let cpu_intensity = cpu_ema.clamp(0.0, 1.0);
        let memory_intensity = (memory_ema / 1_000_000_000.0).clamp(0.0, 1.0);
        let io_intensity = (io_ema / 10_000_000.0).clamp(0.0, 1.0);
        let network_intensity = (network_ema / 1_000_000.0).clamp(0.0, 1.0);
        let overall_intensity =
            (cpu_intensity + memory_intensity + io_intensity + network_intensity) / 4.0;
        let cpu_values: Vec<f64> = usage_data.iter().map(|p| p.snapshot.cpu_usage).collect();
        let variance = if cpu_values.len() > 1 {
            let var_sum: f64 = cpu_values.iter().map(|&v| (v - cpu_ema).powi(2)).sum();
            var_sum / cpu_values.len() as f64
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
            calculation_method: IntensityCalculationMethod::ExponentialWeighted,
        })
    }
    fn name(&self) -> &str {
        "exponential"
    }
    fn description(&self) -> &str {
        "Exponential decay weighting for highly dynamic workloads"
    }
    fn is_suitable_for(&self, characteristics: &DataCharacteristics) -> bool {
        characteristics.variance > 0.5 || characteristics.outlier_percentage > 0.1
    }
}
