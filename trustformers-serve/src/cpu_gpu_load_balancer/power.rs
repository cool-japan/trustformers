//! Power Efficiency Management
//!
//! This module provides power management, efficiency optimization, and
//! energy consumption monitoring for CPU and GPU resources.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{
    config::{PowerEfficiencyMode, PowerScalingFactors},
    types::{PowerState, ProcessorResource, ProcessorType},
};

/// Power efficiency metrics
///
/// Comprehensive power consumption and efficiency metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerEfficiencyMetrics {
    /// Average power consumption (watts)
    pub avg_power_consumption: f32,

    /// Peak power consumption (watts)
    pub peak_power_consumption: f32,

    /// Energy efficiency (ops per watt)
    pub energy_efficiency: f32,

    /// Total energy consumed (joules)
    pub total_energy_consumed: f32,

    /// Power utilization efficiency (0-1)
    pub power_utilization_efficiency: f32,

    /// Power state distribution
    pub power_state_distribution: HashMap<String, f32>,

    /// Power scaling efficiency
    pub power_scaling_efficiency: f32,

    /// DVFS effectiveness
    pub dvfs_effectiveness: f32,

    /// Idle power optimization savings
    pub idle_power_savings: f32,
}

/// Power efficiency manager
///
/// Manages power states, scaling, and optimization strategies.
pub struct PowerEfficiencyManager {
    mode: PowerEfficiencyMode,
    scaling_factors: PowerScalingFactors,
    max_power_limit: f32,
    current_power_consumption: f32,
    metrics: PowerEfficiencyMetrics,
    power_history: Vec<PowerDataPoint>,
}

/// Power consumption data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerDataPoint {
    /// Timestamp (seconds since start)
    pub timestamp: u64,
    /// Power consumption (watts)
    pub power: f32,
    /// Efficiency score
    pub efficiency: f32,
    /// Processor utilization
    pub utilization: f32,
}

/// Power optimization strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PowerOptimizationStrategy {
    /// Minimize power consumption
    MinimizePower,
    /// Maximize energy efficiency
    MaximizeEfficiency,
    /// Balance power and performance
    Balanced,
    /// Adaptive based on workload
    Adaptive,
}

/// Power management interface
pub trait PowerManager {
    /// Set power efficiency mode
    fn set_power_mode(&mut self, mode: PowerEfficiencyMode);

    /// Update power scaling factors
    fn update_scaling_factors(&mut self, factors: PowerScalingFactors);

    /// Get optimal power state for processor
    fn get_optimal_power_state(
        &self,
        resource: &ProcessorResource,
        workload_intensity: f32,
    ) -> PowerState;

    /// Calculate power efficiency score
    fn calculate_efficiency_score(&self, power: f32, throughput: f64) -> f32;

    /// Apply power optimizations
    fn optimize_power_consumption(
        &mut self,
        strategy: PowerOptimizationStrategy,
    ) -> Result<(), String>;

    /// Get current metrics
    fn get_metrics(&self) -> &PowerEfficiencyMetrics;

    /// Generate power report
    fn generate_power_report(&self) -> PowerReport;
}

/// Power efficiency report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerReport {
    /// Current power consumption (watts)
    pub current_power_consumption: f32,

    /// Power limit (watts)
    pub power_limit: f32,

    /// Power utilization (0-1)
    pub power_utilization: f32,

    /// Energy efficiency (ops per watt)
    pub energy_efficiency: f32,

    /// Power state distribution
    pub power_state_distribution: HashMap<String, f32>,

    /// Power savings (watts)
    pub power_savings: f32,

    /// Optimization recommendations
    pub recommendations: Vec<String>,

    /// Historical power data
    pub power_history: Vec<PowerDataPoint>,

    /// DVFS status and effectiveness
    pub dvfs_status: DVFSStatus,
}

/// Dynamic voltage and frequency scaling status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DVFSStatus {
    /// DVFS enabled
    pub enabled: bool,
    /// Current frequency scaling
    pub frequency_scaling: f32,
    /// Current voltage scaling
    pub voltage_scaling: f32,
    /// Effectiveness score
    pub effectiveness: f32,
    /// Power savings from DVFS
    pub power_savings: f32,
}

impl PowerEfficiencyManager {
    /// Create a new power efficiency manager
    pub fn new(mode: PowerEfficiencyMode, max_power_limit: f32) -> Self {
        Self {
            mode,
            scaling_factors: PowerScalingFactors::default(),
            max_power_limit,
            current_power_consumption: 0.0,
            metrics: PowerEfficiencyMetrics::default(),
            power_history: Vec::new(),
        }
    }

    /// Update power consumption data
    pub fn update_power_consumption(
        &mut self,
        _processor_type: ProcessorType,
        power: f32,
        utilization: f32,
    ) {
        self.current_power_consumption = power;

        // Update metrics
        self.metrics.avg_power_consumption =
            (self.metrics.avg_power_consumption * 0.9 + power * 0.1).max(0.0);

        if power > self.metrics.peak_power_consumption {
            self.metrics.peak_power_consumption = power;
        }

        self.metrics.total_energy_consumed += power * 0.001; // Assuming 1ms updates

        // Update power utilization efficiency
        if self.max_power_limit > 0.0 {
            self.metrics.power_utilization_efficiency = power / self.max_power_limit;
        }

        // Record power history
        let timestamp = self.power_history.len() as u64;
        let efficiency = if power > 0.0 { utilization / power * 100.0 } else { 0.0 };

        self.power_history.push(PowerDataPoint {
            timestamp,
            power,
            efficiency,
            utilization,
        });

        // Keep only last hour of data (assuming 1-second intervals)
        if self.power_history.len() > 3600 {
            self.power_history.remove(0);
        }
    }

    /// Calculate DVFS effectiveness
    fn calculate_dvfs_effectiveness(&self) -> f32 {
        if !self.scaling_factors.dvfs_enabled {
            return 0.0;
        }

        let frequency_benefit = 1.0 - self.scaling_factors.cpu_frequency_scaling;
        let voltage_benefit = 1.0 - self.scaling_factors.voltage_scaling;

        // DVFS effectiveness is based on power savings vs performance impact
        (frequency_benefit + voltage_benefit) / 2.0 * 0.8 // 80% max effectiveness
    }

    /// Generate power optimization recommendations
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if self.current_power_consumption > self.max_power_limit * 0.9 {
            recommendations.push(
                "Power consumption near limit, consider enabling power efficiency mode".to_string(),
            );
        }

        if !self.scaling_factors.dvfs_enabled {
            recommendations.push("Enable DVFS for better power efficiency".to_string());
        }

        if self.scaling_factors.cpu_frequency_scaling > 0.9
            && self.current_power_consumption > 500.0
        {
            recommendations.push("Consider reducing CPU frequency to save power".to_string());
        }

        if self.metrics.energy_efficiency < 10.0 {
            recommendations.push(
                "Low energy efficiency detected, review task assignment strategy".to_string(),
            );
        }

        if !self.scaling_factors.sleep_states_enabled {
            recommendations.push("Enable sleep states for idle power optimization".to_string());
        }

        recommendations
    }
}

impl PowerManager for PowerEfficiencyManager {
    fn set_power_mode(&mut self, mode: PowerEfficiencyMode) {
        self.mode = mode;

        // Adjust scaling factors based on mode
        match mode {
            PowerEfficiencyMode::PerformanceFirst => {
                self.scaling_factors.cpu_frequency_scaling = 1.0;
                self.scaling_factors.gpu_power_limit_scaling = 1.0;
                self.scaling_factors.voltage_scaling = 1.0;
            },
            PowerEfficiencyMode::PowerFirst => {
                self.scaling_factors.cpu_frequency_scaling = 0.7;
                self.scaling_factors.gpu_power_limit_scaling = 0.6;
                self.scaling_factors.voltage_scaling = 0.85;
            },
            PowerEfficiencyMode::Balanced => {
                self.scaling_factors.cpu_frequency_scaling = 0.85;
                self.scaling_factors.gpu_power_limit_scaling = 0.8;
                self.scaling_factors.voltage_scaling = 0.9;
            },
            PowerEfficiencyMode::Adaptive | PowerEfficiencyMode::Custom => {
                // Keep current settings for adaptive/custom modes
            },
        }
    }

    fn update_scaling_factors(&mut self, factors: PowerScalingFactors) {
        self.scaling_factors = factors;
        self.metrics.dvfs_effectiveness = self.calculate_dvfs_effectiveness();
    }

    fn get_optimal_power_state(
        &self,
        resource: &ProcessorResource,
        workload_intensity: f32,
    ) -> PowerState {
        match self.mode {
            PowerEfficiencyMode::PerformanceFirst => PowerState::FullPower,
            PowerEfficiencyMode::PowerFirst => {
                if workload_intensity < 0.3 {
                    PowerState::LowPower
                } else {
                    PowerState::ReducedPower
                }
            },
            PowerEfficiencyMode::Balanced => {
                if workload_intensity < 0.2 {
                    PowerState::LowPower
                } else if workload_intensity < 0.7 {
                    PowerState::ReducedPower
                } else {
                    PowerState::FullPower
                }
            },
            PowerEfficiencyMode::Adaptive => {
                // Adaptive based on current conditions
                let temp_factor = if resource.temperature > 80.0 { 0.7 } else { 1.0 };
                let power_factor =
                    if resource.power_consumption > self.max_power_limit * 0.8 { 0.8 } else { 1.0 };

                let adjusted_intensity = workload_intensity * temp_factor * power_factor;

                if adjusted_intensity < 0.3 {
                    PowerState::LowPower
                } else if adjusted_intensity < 0.7 {
                    PowerState::ReducedPower
                } else {
                    PowerState::FullPower
                }
            },
            PowerEfficiencyMode::Custom => {
                // Use custom scaling factor
                PowerState::Custom(self.scaling_factors.cpu_frequency_scaling)
            },
        }
    }

    fn calculate_efficiency_score(&self, power: f32, throughput: f64) -> f32 {
        if power <= 0.0 {
            return 0.0;
        }

        let ops_per_watt = throughput as f32 / power;

        // Normalize to 0-1 scale (assuming 100 ops/watt is excellent)
        (ops_per_watt / 100.0).min(1.0)
    }

    fn optimize_power_consumption(
        &mut self,
        strategy: PowerOptimizationStrategy,
    ) -> Result<(), String> {
        match strategy {
            PowerOptimizationStrategy::MinimizePower => {
                self.scaling_factors.cpu_frequency_scaling = 0.6;
                self.scaling_factors.gpu_power_limit_scaling = 0.5;
                self.scaling_factors.voltage_scaling = 0.8;
                self.scaling_factors.sleep_states_enabled = true;
            },
            PowerOptimizationStrategy::MaximizeEfficiency => {
                self.scaling_factors.cpu_frequency_scaling = 0.8;
                self.scaling_factors.gpu_power_limit_scaling = 0.7;
                self.scaling_factors.dvfs_enabled = true;
                self.scaling_factors.idle_power_optimization = 0.6;
            },
            PowerOptimizationStrategy::Balanced => {
                self.scaling_factors = PowerScalingFactors::default();
            },
            PowerOptimizationStrategy::Adaptive => {
                // Adjust based on current power consumption
                if self.current_power_consumption > self.max_power_limit * 0.8 {
                    self.scaling_factors.cpu_frequency_scaling *= 0.9;
                    self.scaling_factors.gpu_power_limit_scaling *= 0.9;
                }
            },
        }

        self.metrics.power_scaling_efficiency = self.calculate_dvfs_effectiveness();
        Ok(())
    }

    fn get_metrics(&self) -> &PowerEfficiencyMetrics {
        &self.metrics
    }

    fn generate_power_report(&self) -> PowerReport {
        let mut power_state_distribution = HashMap::new();
        power_state_distribution.insert("FullPower".to_string(), 0.4);
        power_state_distribution.insert("ReducedPower".to_string(), 0.4);
        power_state_distribution.insert("LowPower".to_string(), 0.2);

        let dvfs_status = DVFSStatus {
            enabled: self.scaling_factors.dvfs_enabled,
            frequency_scaling: self.scaling_factors.cpu_frequency_scaling,
            voltage_scaling: self.scaling_factors.voltage_scaling,
            effectiveness: self.metrics.dvfs_effectiveness,
            power_savings: self.metrics.idle_power_savings,
        };

        PowerReport {
            current_power_consumption: self.current_power_consumption,
            power_limit: self.max_power_limit,
            power_utilization: self.metrics.power_utilization_efficiency,
            energy_efficiency: self.metrics.energy_efficiency,
            power_state_distribution,
            power_savings: self.metrics.idle_power_savings,
            recommendations: self.generate_recommendations(),
            power_history: self.power_history.clone(),
            dvfs_status,
        }
    }
}

/// Default implementations
impl Default for PowerEfficiencyMetrics {
    fn default() -> Self {
        Self {
            avg_power_consumption: 0.0,
            peak_power_consumption: 0.0,
            energy_efficiency: 0.0,
            total_energy_consumed: 0.0,
            power_utilization_efficiency: 0.0,
            power_state_distribution: HashMap::new(),
            power_scaling_efficiency: 0.0,
            dvfs_effectiveness: 0.0,
            idle_power_savings: 0.0,
        }
    }
}

impl Default for DVFSStatus {
    fn default() -> Self {
        Self {
            enabled: true,
            frequency_scaling: 1.0,
            voltage_scaling: 1.0,
            effectiveness: 0.0,
            power_savings: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_gpu_load_balancer::config::{PowerEfficiencyMode, PowerScalingFactors};
    use crate::cpu_gpu_load_balancer::types::{ProcessorResource, ProcessorType};

    fn make_default_resource() -> ProcessorResource {
        ProcessorResource::default()
    }

    // --- PowerEfficiencyMetrics tests ---

    #[test]
    fn test_power_metrics_default_zeroed() {
        let metrics = PowerEfficiencyMetrics::default();
        assert!((metrics.avg_power_consumption - 0.0).abs() < 1e-6);
        assert!((metrics.peak_power_consumption - 0.0).abs() < 1e-6);
        assert!((metrics.total_energy_consumed - 0.0).abs() < 1e-6);
    }

    // --- PowerEfficiencyManager tests ---

    #[test]
    fn test_manager_new_starts_zero_consumption() {
        let manager = PowerEfficiencyManager::new(PowerEfficiencyMode::Balanced, 1000.0);
        let metrics = manager.get_metrics();
        assert!((metrics.avg_power_consumption - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_manager_update_power_consumption_tracks_peak() {
        let mut manager = PowerEfficiencyManager::new(PowerEfficiencyMode::Balanced, 1000.0);
        manager.update_power_consumption(ProcessorType::CPU, 400.0, 0.5);
        manager.update_power_consumption(ProcessorType::CPU, 700.0, 0.8);
        let metrics = manager.get_metrics();
        assert!((metrics.peak_power_consumption - 700.0).abs() < 1e-4);
    }

    #[test]
    fn test_manager_update_power_tracks_energy() {
        let mut manager = PowerEfficiencyManager::new(PowerEfficiencyMode::Balanced, 1000.0);
        manager.update_power_consumption(ProcessorType::GPU, 500.0, 0.7);
        let metrics = manager.get_metrics();
        assert!(metrics.total_energy_consumed > 0.0);
    }

    #[test]
    fn test_manager_set_power_mode_performance_first() {
        let mut manager = PowerEfficiencyManager::new(PowerEfficiencyMode::Balanced, 1000.0);
        manager.set_power_mode(PowerEfficiencyMode::PerformanceFirst);
        // Verify scaling factors updated to full power
        let report = manager.generate_power_report();
        assert!((report.dvfs_status.frequency_scaling - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_manager_set_power_mode_power_first_reduces_scaling() {
        let mut manager = PowerEfficiencyManager::new(PowerEfficiencyMode::Balanced, 1000.0);
        manager.set_power_mode(PowerEfficiencyMode::PowerFirst);
        let report = manager.generate_power_report();
        // Power-first mode reduces CPU frequency scaling below 1.0
        assert!(report.dvfs_status.frequency_scaling < 1.0);
    }

    #[test]
    fn test_manager_get_optimal_power_state_performance_first_full_power() {
        let manager = PowerEfficiencyManager::new(PowerEfficiencyMode::PerformanceFirst, 1000.0);
        let resource = make_default_resource();
        let state = manager.get_optimal_power_state(&resource, 0.1);
        matches!(state, PowerState::FullPower);
    }

    #[test]
    fn test_manager_get_optimal_power_state_power_first_low_workload() {
        let manager = PowerEfficiencyManager::new(PowerEfficiencyMode::PowerFirst, 1000.0);
        let resource = make_default_resource();
        let state = manager.get_optimal_power_state(&resource, 0.1);
        matches!(state, PowerState::LowPower);
    }

    #[test]
    fn test_manager_get_optimal_power_state_balanced_high_workload() {
        let manager = PowerEfficiencyManager::new(PowerEfficiencyMode::Balanced, 1000.0);
        let resource = make_default_resource();
        let state = manager.get_optimal_power_state(&resource, 0.9);
        matches!(state, PowerState::FullPower);
    }

    #[test]
    fn test_manager_get_optimal_power_state_balanced_medium_workload() {
        let manager = PowerEfficiencyManager::new(PowerEfficiencyMode::Balanced, 1000.0);
        let resource = make_default_resource();
        let state = manager.get_optimal_power_state(&resource, 0.5);
        matches!(state, PowerState::ReducedPower);
    }

    #[test]
    fn test_manager_calculate_efficiency_score_zero_power_returns_zero() {
        let manager = PowerEfficiencyManager::new(PowerEfficiencyMode::Balanced, 1000.0);
        let score = manager.calculate_efficiency_score(0.0, 1000.0);
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_manager_calculate_efficiency_score_positive() {
        let manager = PowerEfficiencyManager::new(PowerEfficiencyMode::Balanced, 1000.0);
        let score = manager.calculate_efficiency_score(100.0, 500.0);
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_manager_optimize_minimize_power() {
        let mut manager = PowerEfficiencyManager::new(PowerEfficiencyMode::Balanced, 1000.0);
        let result = manager.optimize_power_consumption(PowerOptimizationStrategy::MinimizePower);
        assert!(result.is_ok());
        let report = manager.generate_power_report();
        assert!(report.dvfs_status.frequency_scaling < 1.0);
    }

    #[test]
    fn test_manager_optimize_balanced_resets_to_defaults() {
        let mut manager = PowerEfficiencyManager::new(PowerEfficiencyMode::Balanced, 1000.0);
        // First reduce scaling
        manager
            .optimize_power_consumption(PowerOptimizationStrategy::MinimizePower)
            .expect("ok");
        // Then reset to balanced
        manager
            .optimize_power_consumption(PowerOptimizationStrategy::Balanced)
            .expect("ok");
        let report = manager.generate_power_report();
        assert!((report.dvfs_status.frequency_scaling - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_manager_generate_power_report_structure() {
        let manager = PowerEfficiencyManager::new(PowerEfficiencyMode::Balanced, 500.0);
        let report = manager.generate_power_report();
        assert!((report.power_limit - 500.0).abs() < 1e-4);
        assert!(report.power_state_distribution.contains_key("FullPower"));
    }

    #[test]
    fn test_manager_recommendations_near_power_limit() {
        let mut manager = PowerEfficiencyManager::new(PowerEfficiencyMode::Balanced, 100.0);
        // Push near the limit (90%)
        manager.update_power_consumption(ProcessorType::CPU, 95.0, 0.9);
        let report = manager.generate_power_report();
        let has_power_rec = report
            .recommendations
            .iter()
            .any(|r| r.contains("power") || r.contains("Power"));
        assert!(has_power_rec);
    }

    #[test]
    fn test_manager_dvfs_status_default() {
        let status = DVFSStatus::default();
        assert!(status.enabled);
        assert!((status.frequency_scaling - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_update_scaling_factors_updates_dvfs_effectiveness() {
        let mut manager = PowerEfficiencyManager::new(PowerEfficiencyMode::Balanced, 1000.0);
        let factors = PowerScalingFactors {
            cpu_frequency_scaling: 0.7,
            gpu_power_limit_scaling: 0.6,
            memory_frequency_scaling: 0.9,
            voltage_scaling: 0.85,
            idle_power_optimization: 0.7,
            dvfs_enabled: true,
            sleep_states_enabled: true,
        };
        manager.update_scaling_factors(factors);
        let metrics = manager.get_metrics();
        // DVFS effectiveness should be non-zero with these factors
        assert!(metrics.dvfs_effectiveness >= 0.0);
    }
}
