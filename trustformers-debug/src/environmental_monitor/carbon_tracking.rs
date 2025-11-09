//! Carbon footprint tracking and analysis

use crate::environmental_monitor::{config::EnvironmentalConfig, types::*};
use anyhow::Result;
use std::collections::HashMap;
use tracing::{debug, info};

/// Carbon footprint tracking system
#[derive(Debug)]
pub struct CarbonFootprintTracker {
    carbon_intensity_db: CarbonIntensityDatabase,
    measurement_history: Vec<CarbonMeasurement>,
    cumulative_emissions: CarbonEmissions,
    emission_factors: EmissionFactors,
}

/// Regional carbon intensity database
#[derive(Debug)]
struct CarbonIntensityDatabase {
    regional_intensities: HashMap<String, f64>, // gCO2/kWh
    #[allow(dead_code)]
    time_based_intensities: HashMap<String, Vec<TimeBasedIntensity>>,
    renewable_percentages: HashMap<String, f64>,
}

/// Time-based carbon intensity data
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct TimeBasedIntensity {
    #[allow(dead_code)]
    hour: u32,
    carbon_intensity: f64,
    renewable_percentage: f64,
}

/// Emission factors for different activities
#[derive(Debug)]
#[allow(dead_code)]
struct EmissionFactors {
    #[allow(dead_code)]
    gpu_manufacturing_kg_co2: f64,
    cpu_manufacturing_kg_co2: f64,
    infrastructure_kg_co2_per_hour: f64,
    cooling_efficiency_multiplier: f64,
    pue_factor: f64, // Power Usage Effectiveness
}

impl Default for EmissionFactors {
    fn default() -> Self {
        Self {
            gpu_manufacturing_kg_co2: 150.0, // Typical GPU manufacturing footprint
            cpu_manufacturing_kg_co2: 50.0,  // Typical CPU manufacturing footprint
            infrastructure_kg_co2_per_hour: 0.1, // Data center infrastructure
            cooling_efficiency_multiplier: 1.4, // Cooling overhead
            pue_factor: 1.2,                 // Power Usage Effectiveness
        }
    }
}

impl CarbonFootprintTracker {
    /// Create a new carbon footprint tracker
    pub fn new(_config: &EnvironmentalConfig) -> Self {
        let mut regional_intensities = HashMap::new();
        regional_intensities.insert("US-West".to_string(), 350.0); // gCO2/kWh
        regional_intensities.insert("US-East".to_string(), 450.0);
        regional_intensities.insert("EU-North".to_string(), 200.0);
        regional_intensities.insert("EU-Central".to_string(), 400.0);
        regional_intensities.insert("Asia-Pacific".to_string(), 600.0);

        let mut renewable_percentages = HashMap::new();
        renewable_percentages.insert("US-West".to_string(), 45.0);
        renewable_percentages.insert("US-East".to_string(), 25.0);
        renewable_percentages.insert("EU-North".to_string(), 70.0);
        renewable_percentages.insert("EU-Central".to_string(), 35.0);
        renewable_percentages.insert("Asia-Pacific".to_string(), 20.0);

        Self {
            carbon_intensity_db: CarbonIntensityDatabase {
                regional_intensities,
                time_based_intensities: HashMap::new(),
                renewable_percentages,
            },
            measurement_history: Vec::new(),
            cumulative_emissions: CarbonEmissions {
                total_co2_kg: 0.0,
                scope1_emissions_kg: 0.0,
                scope2_emissions_kg: 0.0,
                scope3_emissions_kg: 0.0,
                training_emissions_kg: 0.0,
                inference_emissions_kg: 0.0,
                equivalent_metrics: EquivalentMetrics {
                    car_miles_equivalent: 0.0,
                    tree_months_to_offset: 0.0,
                    coal_pounds_equivalent: 0.0,
                    households_daily_energy: 0.0,
                },
            },
            emission_factors: EmissionFactors::default(),
        }
    }

    /// Record carbon emissions for an energy measurement
    pub fn record_emissions(
        &mut self,
        energy_kwh: f64,
        region: &str,
        measurement_type: MeasurementType,
    ) -> Result<CarbonMeasurement> {
        let carbon_intensity = self.get_carbon_intensity(region);
        let co2_emissions_kg = (energy_kwh * carbon_intensity) / 1000.0; // Convert gCO2 to kg

        let scope2_emissions = co2_emissions_kg * self.emission_factors.pue_factor;
        let scope3_emissions_kg = if measurement_type == MeasurementType::Training {
            Some(self.calculate_scope3_emissions(energy_kwh))
        } else {
            None
        };

        let measurement = CarbonMeasurement {
            timestamp: std::time::SystemTime::now(),
            energy_consumed_kwh: energy_kwh,
            carbon_intensity_gco2_kwh: carbon_intensity,
            co2_emissions_kg,
            scope2_emissions_kg: scope2_emissions,
            scope3_emissions_kg,
            region: region.to_string(),
            measurement_type,
        };

        self.update_cumulative_emissions(&measurement);
        self.measurement_history.push(measurement.clone());

        info!(
            "Recorded {} kg CO2 emissions for {:.2} kWh in region {}",
            co2_emissions_kg, energy_kwh, region
        );

        Ok(measurement)
    }

    /// Get carbon intensity for a region
    pub fn get_carbon_intensity(&self, region: &str) -> f64 {
        self.carbon_intensity_db
            .regional_intensities
            .get(region)
            .cloned()
            .unwrap_or(500.0) // Global average fallback
    }

    /// Get renewable energy percentage for a region
    pub fn get_renewable_percentage(&self, region: &str) -> f64 {
        self.carbon_intensity_db
            .renewable_percentages
            .get(region)
            .cloned()
            .unwrap_or(30.0) // Global average fallback
    }

    /// Calculate scope 3 emissions (infrastructure, manufacturing)
    fn calculate_scope3_emissions(&self, energy_kwh: f64) -> f64 {
        // Simplified scope 3 calculation based on energy usage
        let infrastructure_emissions =
            self.emission_factors.infrastructure_kg_co2_per_hour * energy_kwh;

        // Add manufacturing amortization (very simplified)
        let manufacturing_amortization = 0.001; // kg CO2 per kWh

        infrastructure_emissions + (manufacturing_amortization * energy_kwh)
    }

    /// Update cumulative emissions
    fn update_cumulative_emissions(&mut self, measurement: &CarbonMeasurement) {
        self.cumulative_emissions.total_co2_kg += measurement.co2_emissions_kg;
        self.cumulative_emissions.scope2_emissions_kg += measurement.scope2_emissions_kg;

        if let Some(scope3) = measurement.scope3_emissions_kg {
            self.cumulative_emissions.scope3_emissions_kg += scope3;
        }

        match measurement.measurement_type {
            MeasurementType::Training => {
                self.cumulative_emissions.training_emissions_kg += measurement.co2_emissions_kg;
            },
            MeasurementType::Inference => {
                self.cumulative_emissions.inference_emissions_kg += measurement.co2_emissions_kg;
            },
            _ => {},
        }

        // Update equivalent metrics
        self.update_equivalent_metrics();
    }

    /// Update equivalent metrics for carbon emissions
    fn update_equivalent_metrics(&mut self) {
        let total_co2 = self.cumulative_emissions.total_co2_kg;

        // Car miles equivalent (average car emits ~0.4 kg CO2 per mile)
        self.cumulative_emissions.equivalent_metrics.car_miles_equivalent = total_co2 / 0.4;

        // Tree months to offset (average tree absorbs ~21 kg CO2 per year)
        self.cumulative_emissions.equivalent_metrics.tree_months_to_offset =
            (total_co2 / 21.0) * 12.0;

        // Coal pounds equivalent (~1 kg CO2 per 0.4 kg coal)
        self.cumulative_emissions.equivalent_metrics.coal_pounds_equivalent =
            (total_co2 / 0.4) * 2.20462; // Convert kg to pounds

        // Household daily energy (~30 kWh per day, ~15 kg CO2)
        self.cumulative_emissions.equivalent_metrics.households_daily_energy = total_co2 / 15.0;
    }

    /// Get cumulative emissions
    pub fn get_cumulative_emissions(&self) -> &CarbonEmissions {
        &self.cumulative_emissions
    }

    /// Get measurement history
    pub fn get_measurement_history(&self) -> &[CarbonMeasurement] {
        &self.measurement_history
    }

    /// Get emissions for a specific time period
    pub fn get_period_emissions(
        &self,
        start: std::time::SystemTime,
        end: std::time::SystemTime,
    ) -> Vec<CarbonMeasurement> {
        self.measurement_history
            .iter()
            .filter(|m| m.timestamp >= start && m.timestamp <= end)
            .cloned()
            .collect()
    }

    /// Calculate emissions forecast based on historical data
    pub fn forecast_emissions(&self, hours_ahead: u32) -> Result<f64> {
        if self.measurement_history.len() < 2 {
            return Ok(0.0);
        }

        // Simple linear regression on recent emissions
        let recent_measurements: Vec<_> = self.measurement_history
            .iter()
            .rev()
            .take(24) // Last 24 measurements
            .collect();

        if recent_measurements.is_empty() {
            return Ok(0.0);
        }

        let avg_emissions_per_hour =
            recent_measurements.iter().map(|m| m.co2_emissions_kg).sum::<f64>()
                / recent_measurements.len() as f64;

        Ok(avg_emissions_per_hour * hours_ahead as f64)
    }

    /// Optimize scheduling for lower carbon intensity
    pub fn find_low_carbon_window(&self, region: &str, _duration_hours: u32) -> Option<u32> {
        // Simplified low-carbon window finding
        // In a real implementation, this would use time-based carbon intensity data
        let _base_intensity = self.get_carbon_intensity(region);

        // Find the hour with lowest expected carbon intensity (simplified)
        let renewable_pct = self.get_renewable_percentage(region);

        if renewable_pct > 50.0 {
            Some(14) // Afternoon when solar is strong
        } else {
            Some(2) // Early morning when demand is low
        }
    }

    /// Clear measurement history (useful for testing or privacy)
    pub fn clear_history(&mut self) {
        self.measurement_history.clear();
        debug!("Cleared carbon measurement history");
    }

    /// Export measurements to CSV format
    pub fn export_to_csv(&self) -> String {
        let mut csv = String::from("timestamp,energy_kwh,carbon_intensity,co2_kg,region,type\n");

        for measurement in &self.measurement_history {
            let timestamp = measurement
                .timestamp
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            csv.push_str(&format!(
                "{},{:.4},{:.2},{:.6},{},{:?}\n",
                timestamp,
                measurement.energy_consumed_kwh,
                measurement.carbon_intensity_gco2_kwh,
                measurement.co2_emissions_kg,
                measurement.region,
                measurement.measurement_type
            ));
        }

        csv
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_carbon_tracker_creation() {
        let config = EnvironmentalConfig::default();
        let tracker = CarbonFootprintTracker::new(&config);

        assert_eq!(tracker.cumulative_emissions.total_co2_kg, 0.0);
        assert!(tracker.measurement_history.is_empty());
    }

    #[test]
    fn test_carbon_intensity_lookup() {
        let config = EnvironmentalConfig::default();
        let tracker = CarbonFootprintTracker::new(&config);

        assert_eq!(tracker.get_carbon_intensity("US-West"), 350.0);
        assert_eq!(tracker.get_carbon_intensity("EU-North"), 200.0);
        assert_eq!(tracker.get_carbon_intensity("Unknown"), 500.0); // Fallback
    }

    #[test]
    fn test_emissions_recording() {
        let config = EnvironmentalConfig::default();
        let mut tracker = CarbonFootprintTracker::new(&config);

        let measurement = tracker
            .record_emissions(
                10.0, // 10 kWh
                "US-West",
                MeasurementType::Training,
            )
            .unwrap();

        assert_eq!(measurement.energy_consumed_kwh, 10.0);
        assert_eq!(measurement.carbon_intensity_gco2_kwh, 350.0);
        assert_eq!(measurement.co2_emissions_kg, 3.5); // 10 * 350 / 1000
        assert_eq!(tracker.measurement_history.len(), 1);
    }

    #[test]
    fn test_cumulative_emissions() {
        let config = EnvironmentalConfig::default();
        let mut tracker = CarbonFootprintTracker::new(&config);

        // Record multiple measurements
        let _ = tracker.record_emissions(5.0, "US-West", MeasurementType::Training);
        let _ = tracker.record_emissions(3.0, "US-East", MeasurementType::Inference);

        let cumulative = tracker.get_cumulative_emissions();
        assert!(cumulative.total_co2_kg > 0.0);
        assert!(cumulative.training_emissions_kg > 0.0);
        assert!(cumulative.inference_emissions_kg > 0.0);
    }

    #[test]
    fn test_equivalent_metrics() {
        let config = EnvironmentalConfig::default();
        let mut tracker = CarbonFootprintTracker::new(&config);

        let _ = tracker.record_emissions(10.0, "US-West", MeasurementType::Training);

        let metrics = &tracker.get_cumulative_emissions().equivalent_metrics;
        assert!(metrics.car_miles_equivalent > 0.0);
        assert!(metrics.tree_months_to_offset > 0.0);
        assert!(metrics.coal_pounds_equivalent > 0.0);
        assert!(metrics.households_daily_energy > 0.0);
    }
}
