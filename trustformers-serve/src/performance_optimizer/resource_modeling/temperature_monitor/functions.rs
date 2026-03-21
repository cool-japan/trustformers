//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;

use super::types::{
    AlertEvent, AnalysisResult, GeneratedReport, HeatFlowResult, ReportData, SensorReadings,
    TrendAnalysis,
};

pub trait ThermalAnalysisAlgorithm {
    fn analyze(&self, readings: &SensorReadings) -> Result<AnalysisResult>;
}
pub trait AlertChannel {
    fn send_alert(&self, alert: &AlertEvent) -> Result<()>;
}
pub trait ReferenceTemperatureSource {
    fn get_reference_temperature(&self) -> Result<f32>;
}
pub trait HeatFlowModel {
    fn calculate_heat_flow(&self, readings: &SensorReadings) -> Result<HeatFlowResult>;
}
pub trait ReportGenerator {
    fn generate(&self, data: &ReportData) -> Result<GeneratedReport>;
}
pub trait TrendAnalyzer {
    fn analyze_trends(&self, history: &[SensorReadings]) -> Result<TrendAnalysis>;
}
#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::collections::HashMap;

    use crate::performance_optimizer::resource_modeling::temperature_monitor::{
        CoolingConfig, PredictorConfig, SensorManagerConfig, StateManagerConfig,
        TemperatureThresholds, ThermalSensorType, ThermalState,
    };
    use crate::performance_optimizer::resource_modeling::{
        CoolingController, TemperatureMonitor, TemperatureMonitorConfig, ThermalSensorManager,
        ThermalStateManager, ThrottlingPredictor,
    };
    use tokio::test;
    #[test]
    async fn test_temperature_monitor_creation() {
        let config = TemperatureMonitorConfig::default();
        let monitor = TemperatureMonitor::new(config).await.expect("Failed to create monitor");
        let current_temp = monitor
            .get_current_temperature()
            .await
            .expect("Failed to get current temperature");
        assert!(current_temp.cpu_temperature > 0.0);
    }
    #[test]
    async fn test_thermal_sensor_manager() {
        let config = SensorManagerConfig::default();
        let manager = ThermalSensorManager::new(config)
            .await
            .expect("Failed to create thermal sensor manager");
        let readings = manager.read_sensors().await.expect("Failed to read sensors");
        assert!(readings.cpu_temperature > 0.0);
        assert!(readings.system_temperature > 0.0);
    }
    #[test]
    async fn test_thermal_state_transitions() {
        let config = StateManagerConfig::default();
        let manager = ThermalStateManager::new(config)
            .await
            .expect("Failed to create thermal state manager");
        let thresholds = TemperatureThresholds::default();
        let readings = SensorReadings {
            cpu_temperature: 80.0,
            gpu_temperature: Some(75.0),
            system_temperature: 78.0,
            ambient_temperature: Some(25.0),
            motherboard_temperature: Some(70.0),
            memory_temperature: Some(65.0),
            storage_temperature: Some(40.0),
            timestamp: Utc::now(),
            thermal_throttling: false,
            sensor_details: HashMap::new(),
        };
        let state = manager
            .update_state(&readings, &thresholds)
            .await
            .expect("Failed to update state");
        assert_eq!(state, ThermalState::Warning);
    }
    #[test]
    async fn test_throttling_prediction() {
        let config = PredictorConfig::default();
        let predictor = ThrottlingPredictor::new(config)
            .await
            .expect("Failed to create throttling predictor");
        let readings = SensorReadings {
            cpu_temperature: 85.0,
            gpu_temperature: Some(80.0),
            system_temperature: 83.0,
            ambient_temperature: Some(30.0),
            motherboard_temperature: Some(75.0),
            memory_temperature: Some(70.0),
            storage_temperature: Some(45.0),
            timestamp: Utc::now(),
            thermal_throttling: false,
            sensor_details: HashMap::new(),
        };
        let predictions = predictor
            .predict_throttling(&readings)
            .await
            .expect("Failed to predict throttling");
        assert!(!predictions.is_empty());
        assert!(predictions[0].probability >= 0.0 && predictions[0].probability <= 1.0);
    }
    #[test]
    async fn test_cooling_controller() {
        let config = CoolingConfig::default();
        let controller = CoolingController::new(config)
            .await
            .expect("Failed to create cooling controller");
        let status = controller.get_status().await.expect("Failed to get status");
        assert!(status.cooling_effectiveness > 0.0);
        assert!(status.cooling_effectiveness <= 1.0);
    }
    #[test]
    async fn test_temperature_thresholds() {
        let thresholds = TemperatureThresholds::default();
        assert!(thresholds.warning_temperature > thresholds.normal_temperature);
        assert!(thresholds.critical_temperature > thresholds.warning_temperature);
        assert!(thresholds.emergency_temperature > thresholds.critical_temperature);
    }
    #[test]
    async fn test_thermal_sensor_type_determination() {
        assert!(matches!(
            ThermalSensorManager::determine_sensor_type("CPU Core 0"),
            ThermalSensorType::Cpu
        ));
        assert!(matches!(
            ThermalSensorManager::determine_sensor_type("GPU Temperature"),
            ThermalSensorType::Gpu
        ));
        assert!(matches!(
            ThermalSensorManager::determine_sensor_type("System Board"),
            ThermalSensorType::System
        ));
    }
}
