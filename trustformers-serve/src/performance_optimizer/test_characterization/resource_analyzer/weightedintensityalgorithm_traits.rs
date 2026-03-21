//! # WeightedIntensityAlgorithm - Trait Implementations
//!
//! This module contains trait implementations for `WeightedIntensityAlgorithm`.
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
use super::types::WeightedIntensityAlgorithm;

impl Default for WeightedIntensityAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl IntensityCalculationAlgorithm for WeightedIntensityAlgorithm {
    fn calculate_intensity(
        &self,
        usage_data: &[ResourceUsageDataPoint],
    ) -> Result<ResourceIntensity> {
        if usage_data.is_empty() {
            return Ok(ResourceIntensity::default());
        }
        let weights = self.calculate_weights(usage_data.len());
        let total_weight: f64 = weights.iter().sum();
        let cpu_weighted: f64 =
            usage_data.iter().zip(&weights).map(|(p, &w)| p.snapshot.cpu_usage * w).sum();
        let memory_weighted: f64 = usage_data
            .iter()
            .zip(&weights)
            .map(|(p, &w)| p.snapshot.memory_usage as f64 * w)
            .sum();
        let io_weighted: f64 = usage_data
            .iter()
            .zip(&weights)
            .map(|(p, &w)| (p.snapshot.io_read_rate + p.snapshot.io_write_rate) * w)
            .sum();
        let network_weighted: f64 = usage_data
            .iter()
            .zip(&weights)
            .map(|(p, &w)| (p.snapshot.network_rx_rate + p.snapshot.network_tx_rate) * w)
            .sum();
        let cpu_intensity = (cpu_weighted / total_weight).clamp(0.0, 1.0);
        let memory_intensity = (memory_weighted / total_weight / 1_000_000_000.0).clamp(0.0, 1.0);
        let io_intensity = (io_weighted / total_weight / 10_000_000.0).clamp(0.0, 1.0);
        let network_intensity = (network_weighted / total_weight / 1_000_000.0).clamp(0.0, 1.0);
        let overall_intensity =
            (cpu_intensity + memory_intensity + io_intensity + network_intensity) / 4.0;
        let cpu_values: Vec<f64> = usage_data.iter().map(|p| p.snapshot.cpu_usage).collect();
        let mean_cpu = cpu_weighted / total_weight;
        let variance = if cpu_values.len() > 1 {
            let var_sum: f64 =
                cpu_values.iter().zip(&weights).map(|(&v, &w)| w * (v - mean_cpu).powi(2)).sum();
            var_sum / total_weight
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
        "weighted"
    }
    fn description(&self) -> &str {
        "Time-weighted intensity calculation emphasizing recent data"
    }
    fn is_suitable_for(&self, characteristics: &DataCharacteristics) -> bool {
        characteristics.trend_strength > 0.3
    }
}
