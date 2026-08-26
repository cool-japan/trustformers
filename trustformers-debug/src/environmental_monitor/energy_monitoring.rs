//! Energy consumption monitoring and analysis
// reason: debug/profiling scaffolding — structs are constructed and their fields/methods
// are retained for the data model, serialization completeness, and future consumers that
// do not yet read every member. Consolidated from many item-level #[allow(dead_code)].
#![allow(dead_code)]

use crate::environmental_monitor::types::*;
use anyhow::Result;
use std::collections::HashMap;
use tracing::{debug, info};

/// Energy consumption monitoring system
#[derive(Debug)]
pub struct EnergyConsumptionMonitor {
    device_monitors: HashMap<String, DeviceEnergyMonitor>,
    consumption_history: Vec<EnergyMeasurement>,
    power_profiles: HashMap<String, PowerProfile>,
    efficiency_metrics: EnergyEfficiencyMetrics,
    /// Real operation count reported by the caller via
    /// [`Self::record_operations`]; `None` until one is.
    reported_operations: Option<u64>,
    /// Real sustained FLOP rate reported by the caller via
    /// [`Self::record_flops_per_second`]; `None` until one is.
    reported_flops_per_second: Option<f64>,
}

/// Device-specific energy monitor
#[derive(Debug)]
struct DeviceEnergyMonitor {
    device_id: String,
    device_type: DeviceType,
    power_measurement_method: PowerMeasurementMethod,
    baseline_power: f64,  // Watts
    current_power: f64,   // Watts
    energy_consumed: f64, // kWh
    last_update: std::time::SystemTime,
}

/// Power profile for different device types
#[derive(Debug, Clone)]
struct PowerProfile {
    device_type: DeviceType,
    idle_power: f64,
    max_power: f64,
    utilization_curve: Vec<(f64, f64)>, // (utilization, power_ratio)
}

impl EnergyConsumptionMonitor {
    /// Create a new energy consumption monitor
    pub fn new() -> Self {
        Self {
            device_monitors: HashMap::new(),
            consumption_history: Vec::new(),
            power_profiles: Self::create_default_power_profiles(),
            reported_operations: None,
            reported_flops_per_second: None,
            efficiency_metrics: EnergyEfficiencyMetrics {
                operations_per_kwh: None,
                flops_per_watt: None,
                model_energy_efficiency: 0.0,
                training_energy_efficiency: 0.0,
                inference_energy_efficiency: 0.0,
                // No reference baselines are available; see
                // `ComparativeEfficiency`.
                comparative_efficiency: ComparativeEfficiency {
                    vs_cpu_only: None,
                    vs_previous_generation: None,
                    vs_cloud_baseline: None,
                    efficiency_percentile: None,
                },
            },
        }
    }

    /// Add a device for energy monitoring
    pub fn add_device(
        &mut self,
        device_id: String,
        device_type: DeviceType,
        measurement_method: PowerMeasurementMethod,
    ) -> Result<()> {
        let power_profile = self.get_power_profile(&device_type);

        let monitor = DeviceEnergyMonitor {
            device_id: device_id.clone(),
            device_type: device_type.clone(),
            power_measurement_method: measurement_method,
            baseline_power: power_profile.idle_power,
            current_power: power_profile.idle_power,
            energy_consumed: 0.0,
            last_update: std::time::SystemTime::now(),
        };

        self.device_monitors.insert(device_id.clone(), monitor);
        info!("Added device {} for energy monitoring", device_id);
        Ok(())
    }

    /// Record energy measurement for a device
    pub fn record_measurement(
        &mut self,
        device_id: &str,
        power_watts: f64,
        utilization: f64,
        temperature: Option<f64>,
    ) -> Result<EnergyMeasurement> {
        let now = std::time::SystemTime::now();

        // Get device type and update energy consumption in a separate scope
        let (device_type, updated_energy_kwh) = {
            let device = self
                .device_monitors
                .get_mut(device_id)
                .ok_or_else(|| anyhow::anyhow!("Device {} not found", device_id))?;

            let duration_hours = now.duration_since(device.last_update)?.as_secs_f64() / 3600.0;

            // Update energy consumption
            device.energy_consumed += power_watts * duration_hours / 1000.0; // Convert to kWh
            device.current_power = power_watts;
            device.last_update = now;

            (device.device_type.clone(), device.energy_consumed)
        };

        // Calculate efficiency ratio (now self is no longer mutably borrowed)
        let power_profile = self.get_power_profile(&device_type);
        let efficiency_ratio =
            self.calculate_efficiency_ratio(power_profile, power_watts, utilization);

        let measurement = EnergyMeasurement {
            timestamp: now,
            device_id: device_id.to_string(),
            power_watts,
            energy_kwh: updated_energy_kwh,
            utilization: Some(utilization),
            temperature,
            efficiency_ratio: Some(efficiency_ratio),
        };

        self.consumption_history.push(measurement.clone());
        self.update_efficiency_metrics();

        Ok(measurement)
    }

    /// Get power profile for device type
    fn get_power_profile(&self, device_type: &DeviceType) -> &PowerProfile {
        self.power_profiles
            .get(&self.device_type_key(device_type))
            .unwrap_or(&self.power_profiles["default"])
    }

    /// Convert device type to key for power profiles
    fn device_type_key(&self, device_type: &DeviceType) -> String {
        match device_type {
            DeviceType::GPU => "gpu".to_string(),
            DeviceType::CPU => "cpu".to_string(),
            DeviceType::Memory => "memory".to_string(),
            DeviceType::Storage => "storage".to_string(),
            DeviceType::Network => "network".to_string(),
            DeviceType::Cooling => "cooling".to_string(),
            DeviceType::Other(name) => name.clone(),
        }
    }

    /// Calculate efficiency ratio based on power and utilization
    fn calculate_efficiency_ratio(
        &self,
        profile: &PowerProfile,
        power: f64,
        utilization: f64,
    ) -> f64 {
        let expected_power =
            profile.idle_power + (profile.max_power - profile.idle_power) * utilization;

        if expected_power > 0.0 {
            expected_power / power.max(1.0) // Avoid division by zero
        } else {
            1.0
        }
    }

    /// Update overall efficiency metrics
    fn update_efficiency_metrics(&mut self) {
        if self.consumption_history.is_empty() {
            return;
        }

        let recent_measurements: Vec<_> = self.consumption_history
            .iter()
            .rev()
            .take(100) // Last 100 measurements
            .collect();

        // Work-per-energy needs a real work count, which only the caller can
        // supply (see `record_operations` / `record_flops`). The monitor
        // samples power and utilization, never work done -- so nothing is
        // derived here.
        let total_energy: f64 = recent_measurements.iter().map(|m| m.energy_kwh).sum();
        if total_energy > 0.0 {
            if let Some(operations) = self.reported_operations {
                self.efficiency_metrics.operations_per_kwh = Some(operations as f64 / total_energy);
            }
        }

        let avg_power: f64 = recent_measurements.iter().map(|m| m.power_watts).sum::<f64>()
            / recent_measurements.len() as f64;
        if avg_power > 0.0 {
            if let Some(flops_per_second) = self.reported_flops_per_second {
                self.efficiency_metrics.flops_per_watt = Some(flops_per_second / avg_power);
            }
        }
    }

    /// Report the real number of model operations completed since monitoring
    /// began, so [`EnergyEfficiencyMetrics::operations_per_kwh`] can be
    /// computed from a measured work count instead of an assumed one.
    pub fn record_operations(&mut self, operations: u64) {
        self.reported_operations = Some(operations);
    }

    /// Report the real sustained FLOP rate of the workload, so
    /// [`EnergyEfficiencyMetrics::flops_per_watt`] can be computed against a
    /// measured rate instead of an assumed device peak.
    pub fn record_flops_per_second(&mut self, flops_per_second: f64) {
        self.reported_flops_per_second = Some(flops_per_second);
    }

    /// Get current energy consumption for all devices
    pub fn get_current_consumption(&self) -> f64 {
        self.device_monitors.values().map(|d| d.current_power).sum::<f64>() / 1000.0
        // Convert to kW
    }

    /// Get total energy consumed
    pub fn get_total_energy_consumed(&self) -> f64 {
        self.device_monitors.values().map(|d| d.energy_consumed).sum()
    }

    /// Get efficiency metrics
    pub fn get_efficiency_metrics(&self) -> &EnergyEfficiencyMetrics {
        &self.efficiency_metrics
    }

    /// Get consumption history
    pub fn get_consumption_history(&self) -> &[EnergyMeasurement] {
        &self.consumption_history
    }

    /// Get measurements for a specific device
    pub fn get_device_measurements(&self, device_id: &str) -> Vec<&EnergyMeasurement> {
        self.consumption_history.iter().filter(|m| m.device_id == device_id).collect()
    }

    /// Detect energy waste patterns
    pub fn detect_energy_waste(&self) -> Vec<WasteMeasurement> {
        let mut waste_measurements = Vec::new();

        for device in self.device_monitors.values() {
            // Check for idle waste
            let power_profile = self.get_power_profile(&device.device_type);
            let idle_threshold = power_profile.idle_power * 1.5; // 50% above idle

            if device.current_power < idle_threshold && device.current_power > 0.0 {
                let wasted_power = device.current_power - power_profile.idle_power;
                let waste_measurement = WasteMeasurement {
                    timestamp: std::time::SystemTime::now(),
                    waste_type: WasteType::IdleResources,
                    wasted_energy_kwh: wasted_power / 1000.0, // Per hour
                    wasted_cost_usd: (wasted_power / 1000.0) * 0.12, // Assuming $0.12/kWh
                    efficiency_lost_percentage: (wasted_power / device.current_power) * 100.0,
                    description: format!("Device {} running above idle power", device.device_id),
                };
                waste_measurements.push(waste_measurement);
            }
        }

        waste_measurements
    }

    /// Predict energy consumption for next N hours
    pub fn predict_energy_consumption(&self, hours: u32) -> f64 {
        if self.consumption_history.len() < 10 {
            return self.get_current_consumption() * hours as f64;
        }

        // Simple trend analysis on recent consumption
        let recent_power: Vec<f64> = self.consumption_history
            .iter()
            .rev()
            .take(24) // Last 24 measurements
            .map(|m| m.power_watts)
            .collect();

        let avg_power = recent_power.iter().sum::<f64>() / recent_power.len() as f64;
        (avg_power / 1000.0) * hours as f64 // Convert to kWh
    }

    /// Create default power profiles for different device types
    fn create_default_power_profiles() -> HashMap<String, PowerProfile> {
        let mut profiles = HashMap::new();

        // GPU profile
        profiles.insert(
            "gpu".to_string(),
            PowerProfile {
                device_type: DeviceType::GPU,
                idle_power: 50.0, // 50W idle
                max_power: 350.0, // 350W max
                utilization_curve: vec![
                    (0.0, 0.15), // 15% of max at 0% utilization
                    (0.2, 0.3),  // 30% of max at 20% utilization
                    (0.5, 0.6),  // 60% of max at 50% utilization
                    (0.8, 0.85), // 85% of max at 80% utilization
                    (1.0, 1.0),  // 100% of max at 100% utilization
                ],
            },
        );

        // CPU profile
        profiles.insert(
            "cpu".to_string(),
            PowerProfile {
                device_type: DeviceType::CPU,
                idle_power: 15.0, // 15W idle
                max_power: 125.0, // 125W max
                utilization_curve: vec![
                    (0.0, 0.12),
                    (0.2, 0.25),
                    (0.5, 0.55),
                    (0.8, 0.8),
                    (1.0, 1.0),
                ],
            },
        );

        // Memory profile
        profiles.insert(
            "memory".to_string(),
            PowerProfile {
                device_type: DeviceType::Memory,
                idle_power: 5.0, // 5W idle
                max_power: 20.0, // 20W max
                utilization_curve: vec![(0.0, 0.25), (0.5, 0.6), (1.0, 1.0)],
            },
        );

        // Default profile
        profiles.insert(
            "default".to_string(),
            PowerProfile {
                device_type: DeviceType::Other("default".to_string()),
                idle_power: 10.0,
                max_power: 50.0,
                utilization_curve: vec![(0.0, 0.2), (1.0, 1.0)],
            },
        );

        profiles
    }

    /// Export energy data to CSV
    pub fn export_to_csv(&self) -> String {
        let mut csv = String::from(
            "timestamp,device_id,power_watts,energy_kwh,utilization,temperature,efficiency_ratio\n",
        );

        for measurement in &self.consumption_history {
            let timestamp = measurement
                .timestamp
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            csv.push_str(&format!(
                "{},{},{:.2},{:.6},{},{},{}\n",
                timestamp,
                measurement.device_id,
                measurement.power_watts,
                measurement.energy_kwh,
                measurement.utilization.map_or(String::new(), |u| format!("{u:.4}")),
                measurement.temperature.map_or("".to_string(), |t| format!("{:.1}", t)),
                measurement.efficiency_ratio.map_or(String::new(), |e| format!("{e:.4}"))
            ));
        }

        csv
    }

    /// Clear measurement history
    pub fn clear_history(&mut self) {
        self.consumption_history.clear();
        debug!("Cleared energy measurement history");
    }

    /// Reset device energy counters
    pub fn reset_device_counters(&mut self) {
        for device in self.device_monitors.values_mut() {
            device.energy_consumed = 0.0;
            device.last_update = std::time::SystemTime::now();
        }
        debug!("Reset all device energy counters");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_monitor_creation() {
        let monitor = EnergyConsumptionMonitor::new();
        assert!(monitor.device_monitors.is_empty());
        assert!(monitor.consumption_history.is_empty());
        assert_eq!(monitor.get_total_energy_consumed(), 0.0);
    }

    #[test]
    fn test_add_device() {
        let mut monitor = EnergyConsumptionMonitor::new();

        let result = monitor.add_device(
            "gpu-0".to_string(),
            DeviceType::GPU,
            PowerMeasurementMethod::NVML,
        );

        assert!(result.is_ok());
        assert_eq!(monitor.device_monitors.len(), 1);
        assert!(monitor.device_monitors.contains_key("gpu-0"));
    }

    #[test]
    fn test_record_measurement() {
        let mut monitor = EnergyConsumptionMonitor::new();

        let _ = monitor.add_device(
            "gpu-0".to_string(),
            DeviceType::GPU,
            PowerMeasurementMethod::NVML,
        );

        let measurement = monitor
            .record_measurement(
                "gpu-0",
                200.0,      // 200W power
                0.8,        // 80% utilization
                Some(65.0), // 65°C temperature
            )
            .expect("operation failed in test");

        assert_eq!(measurement.device_id, "gpu-0");
        assert_eq!(measurement.power_watts, 200.0);
        assert_eq!(measurement.utilization, Some(0.8));
        assert_eq!(measurement.temperature, Some(65.0));
        assert_eq!(monitor.consumption_history.len(), 1);
    }

    #[test]
    fn test_energy_consumption_tracking() {
        let mut monitor = EnergyConsumptionMonitor::new();

        let _ = monitor.add_device(
            "gpu-0".to_string(),
            DeviceType::GPU,
            PowerMeasurementMethod::NVML,
        );

        // Simulate measurements over time
        std::thread::sleep(std::time::Duration::from_millis(100));
        let _ = monitor.record_measurement("gpu-0", 200.0, 0.8, Some(65.0));

        std::thread::sleep(std::time::Duration::from_millis(100));
        let _ = monitor.record_measurement("gpu-0", 250.0, 0.9, Some(70.0));

        assert!(monitor.get_total_energy_consumed() > 0.0);
        assert_eq!(monitor.get_consumption_history().len(), 2);
    }

    #[test]
    fn test_waste_detection() {
        let mut monitor = EnergyConsumptionMonitor::new();

        let _ = monitor.add_device(
            "gpu-0".to_string(),
            DeviceType::GPU,
            PowerMeasurementMethod::NVML,
        );

        // Record low power consumption (potential waste)
        let _ = monitor.record_measurement("gpu-0", 70.0, 0.1, Some(40.0));

        let waste = monitor.detect_energy_waste();
        assert!(!waste.is_empty());
        assert!(matches!(waste[0].waste_type, WasteType::IdleResources));
    }

    #[test]
    fn test_power_profiles() {
        let monitor = EnergyConsumptionMonitor::new();

        let gpu_profile = monitor.get_power_profile(&DeviceType::GPU);
        assert_eq!(gpu_profile.idle_power, 50.0);
        assert_eq!(gpu_profile.max_power, 350.0);

        let cpu_profile = monitor.get_power_profile(&DeviceType::CPU);
        assert_eq!(cpu_profile.idle_power, 15.0);
        assert_eq!(cpu_profile.max_power, 125.0);
    }

    #[test]
    fn test_efficiency_calculation() {
        let monitor = EnergyConsumptionMonitor::new();
        let profile = monitor.get_power_profile(&DeviceType::GPU);

        let efficiency = monitor.calculate_efficiency_ratio(profile, 200.0, 0.5);
        assert!(efficiency > 0.0);
        assert!(efficiency <= 2.0); // Should be reasonable
    }
}
