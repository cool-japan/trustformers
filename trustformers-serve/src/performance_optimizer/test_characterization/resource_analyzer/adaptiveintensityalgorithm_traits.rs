//! # AdaptiveIntensityAlgorithm - Trait Implementations
//!
//! This module contains trait implementations for `AdaptiveIntensityAlgorithm`.
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
use std::time::Duration;

use super::functions::IntensityCalculationAlgorithm;
use super::types::AdaptiveIntensityAlgorithm;

impl Default for AdaptiveIntensityAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl IntensityCalculationAlgorithm for AdaptiveIntensityAlgorithm {
    fn calculate_intensity(
        &self,
        usage_data: &[ResourceUsageDataPoint],
    ) -> Result<ResourceIntensity> {
        if usage_data.is_empty() {
            return Ok(ResourceIntensity::default());
        }
        let mean_result = self.mean_algorithm.calculate_intensity(usage_data)?;
        let weighted_result = self.weighted_algorithm.calculate_intensity(usage_data)?;
        let exponential_result = self.exponential_algorithm.calculate_intensity(usage_data)?;
        let peak_result = self.peak_algorithm.calculate_intensity(usage_data)?;
        let variance = self.calculate_variance(usage_data);
        let trend_strength = self.calculate_trend_strength(usage_data);
        let spike_frequency = self.calculate_spike_frequency(usage_data);
        let mean_weight = if variance < 0.3 { 0.4 } else { 0.2 };
        let weighted_weight = if trend_strength > 0.3 { 0.3 } else { 0.2 };
        let exponential_weight = if variance > 0.5 { 0.3 } else { 0.1 };
        let peak_weight = if spike_frequency > 0.05 { 0.3 } else { 0.1 };
        let total_weight = mean_weight + weighted_weight + exponential_weight + peak_weight;
        let cpu_intensity = (mean_result.cpu_intensity * mean_weight
            + weighted_result.cpu_intensity * weighted_weight
            + exponential_result.cpu_intensity * exponential_weight
            + peak_result.cpu_intensity * peak_weight)
            / total_weight;
        let memory_intensity = (mean_result.memory_intensity * mean_weight
            + weighted_result.memory_intensity * weighted_weight
            + exponential_result.memory_intensity * exponential_weight
            + peak_result.memory_intensity * peak_weight)
            / total_weight;
        let io_intensity = (mean_result.io_intensity * mean_weight
            + weighted_result.io_intensity * weighted_weight
            + exponential_result.io_intensity * exponential_weight
            + peak_result.io_intensity * peak_weight)
            / total_weight;
        let network_intensity = (mean_result.network_intensity * mean_weight
            + weighted_result.network_intensity * weighted_weight
            + exponential_result.network_intensity * exponential_weight
            + peak_result.network_intensity * peak_weight)
            / total_weight;
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
        let peak_periods = if spike_frequency > 0.05 {
            let mut sorted_cpu = cpu_values.clone();
            sorted_cpu.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let percentile_80_idx = (sorted_cpu.len() as f64 * 0.8) as usize;
            let threshold = sorted_cpu.get(percentile_80_idx).copied().unwrap_or(0.0);
            usage_data
                .iter()
                .filter(|p| p.snapshot.cpu_usage >= threshold)
                .map(|p| (p.timestamp, Duration::from_millis(100)))
                .collect()
        } else {
            Vec::new()
        };
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
            calculation_method: IntensityCalculationMethod::Statistical,
        })
    }
    fn name(&self) -> &str {
        "adaptive"
    }
    fn description(&self) -> &str {
        "Adaptive algorithm combining multiple approaches based on data characteristics"
    }
    fn is_suitable_for(&self, _characteristics: &DataCharacteristics) -> bool {
        true
    }
}
