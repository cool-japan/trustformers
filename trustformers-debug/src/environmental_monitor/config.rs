//! Configuration for environmental monitoring

use serde::{Deserialize, Serialize};

/// Configuration for environmental monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalConfig {
    /// Enable carbon footprint tracking
    pub enable_carbon_tracking: bool,
    /// Enable detailed energy monitoring
    pub enable_energy_monitoring: bool,
    /// Enable efficiency optimization recommendations
    pub enable_efficiency_analysis: bool,
    /// Enable sustainability reporting
    pub enable_sustainability_reporting: bool,
    /// Enable real-time environmental alerts
    pub enable_environmental_alerts: bool,
    /// Geographic region for carbon intensity calculations
    pub region: String,
    /// Carbon intensity override (gCO2/kWh) - use regional average if None
    pub carbon_intensity_override: Option<f64>,
    /// Energy price per kWh (USD)
    pub energy_price_per_kwh: f64,
    /// Monitoring interval (seconds)
    pub monitoring_interval_secs: u64,
    /// Carbon footprint alert threshold (kg CO2)
    pub carbon_alert_threshold: f64,
    /// Energy consumption alert threshold (kWh)
    pub energy_alert_threshold: f64,
    /// Include scope 2 emissions (electricity)
    pub include_scope2_emissions: bool,
    /// Include scope 3 emissions (infrastructure, manufacturing)
    pub include_scope3_emissions: bool,
}

impl Default for EnvironmentalConfig {
    fn default() -> Self {
        Self {
            enable_carbon_tracking: true,
            enable_energy_monitoring: true,
            enable_efficiency_analysis: true,
            enable_sustainability_reporting: true,
            enable_environmental_alerts: true,
            region: "US-West".to_string(),
            carbon_intensity_override: None,
            energy_price_per_kwh: 0.12, // US average
            monitoring_interval_secs: 60,
            carbon_alert_threshold: 10.0,  // kg CO2
            energy_alert_threshold: 100.0, // kWh
            include_scope2_emissions: true,
            include_scope3_emissions: false,
        }
    }
}

/// Regional configuration presets
impl EnvironmentalConfig {
    /// US West Coast configuration (low carbon intensity)
    pub fn us_west() -> Self {
        Self {
            region: "US-West".to_string(),
            carbon_intensity_override: Some(350.0), // gCO2/kWh
            energy_price_per_kwh: 0.15,
            ..Default::default()
        }
    }

    /// US East Coast configuration (medium carbon intensity)
    pub fn us_east() -> Self {
        Self {
            region: "US-East".to_string(),
            carbon_intensity_override: Some(450.0), // gCO2/kWh
            energy_price_per_kwh: 0.13,
            ..Default::default()
        }
    }

    /// European configuration (low carbon intensity)
    pub fn europe() -> Self {
        Self {
            region: "EU".to_string(),
            carbon_intensity_override: Some(300.0), // gCO2/kWh
            energy_price_per_kwh: 0.20,
            ..Default::default()
        }
    }

    /// Asia-Pacific configuration (high carbon intensity)
    pub fn asia_pacific() -> Self {
        Self {
            region: "APAC".to_string(),
            carbon_intensity_override: Some(600.0), // gCO2/kWh
            energy_price_per_kwh: 0.10,
            ..Default::default()
        }
    }

    /// High-precision monitoring configuration
    pub fn high_precision() -> Self {
        Self {
            monitoring_interval_secs: 10,
            enable_environmental_alerts: true,
            carbon_alert_threshold: 1.0,
            energy_alert_threshold: 10.0,
            include_scope3_emissions: true,
            ..Default::default()
        }
    }

    /// Low-overhead monitoring configuration
    pub fn low_overhead() -> Self {
        Self {
            monitoring_interval_secs: 300, // 5 minutes
            enable_environmental_alerts: false,
            enable_efficiency_analysis: false,
            include_scope3_emissions: false,
            ..Default::default()
        }
    }

    /// Compliance monitoring configuration (detailed tracking)
    pub fn compliance_focused() -> Self {
        Self {
            enable_carbon_tracking: true,
            enable_energy_monitoring: true,
            enable_efficiency_analysis: true,
            enable_sustainability_reporting: true,
            enable_environmental_alerts: true,
            include_scope2_emissions: true,
            include_scope3_emissions: true,
            monitoring_interval_secs: 30,
            ..Default::default()
        }
    }

    /// Development environment configuration (minimal monitoring)
    pub fn development() -> Self {
        Self {
            enable_carbon_tracking: false,
            enable_energy_monitoring: true,
            enable_efficiency_analysis: false,
            enable_sustainability_reporting: false,
            enable_environmental_alerts: false,
            monitoring_interval_secs: 120,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EnvironmentalConfig::default();
        assert!(config.enable_carbon_tracking);
        assert!(config.enable_energy_monitoring);
        assert_eq!(config.region, "US-West");
        assert_eq!(config.monitoring_interval_secs, 60);
    }

    #[test]
    fn test_regional_configs() {
        let us_west = EnvironmentalConfig::us_west();
        assert_eq!(us_west.region, "US-West");
        assert_eq!(us_west.carbon_intensity_override, Some(350.0));

        let europe = EnvironmentalConfig::europe();
        assert_eq!(europe.region, "EU");
        assert_eq!(europe.carbon_intensity_override, Some(300.0));
    }

    #[test]
    fn test_specialized_configs() {
        let high_precision = EnvironmentalConfig::high_precision();
        assert_eq!(high_precision.monitoring_interval_secs, 10);
        assert!(high_precision.include_scope3_emissions);

        let low_overhead = EnvironmentalConfig::low_overhead();
        assert_eq!(low_overhead.monitoring_interval_secs, 300);
        assert!(!low_overhead.enable_environmental_alerts);

        let development = EnvironmentalConfig::development();
        assert!(!development.enable_carbon_tracking);
        assert!(!development.enable_sustainability_reporting);
    }
}
