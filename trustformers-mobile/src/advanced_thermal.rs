//! Advanced Thermal Management with Predictive Algorithms
//!
//! This module provides advanced thermal management capabilities including:
//! - Predictive thermal modeling
//! - ML-based throttle prediction
//! - Adaptive cooling strategies
//! - Multi-sensor fusion
//! - Workload-aware thermal planning

use crate::device_info::{MobileDeviceInfo, ThermalState};
use crate::thermal_power::{ThermalPowerConfig, ThermalThresholds};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use trustformers_core::errors::{Result, TrustformersError};

/// Advanced thermal prediction model
#[derive(Debug, Clone)]
pub struct ThermalPredictionModel {
    /// Historical temperature samples (°C)
    temperature_history: VecDeque<f32>,

    /// Historical workload intensity (0.0-1.0)
    workload_history: VecDeque<f32>,

    /// Prediction horizon (seconds)
    prediction_horizon_secs: u32,

    /// Model coefficients (learned from data)
    coefficients: ThermalCoefficients,

    /// Ambient temperature estimate (°C)
    ambient_temperature: f32,
}

/// Thermal model coefficients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalCoefficients {
    /// Thermal resistance (°C/W)
    pub thermal_resistance: f32,

    /// Thermal capacitance (J/°C)
    pub thermal_capacitance: f32,

    /// Workload power coefficient (W per unit workload)
    pub workload_power_coeff: f32,

    /// Cooling coefficient
    pub cooling_coeff: f32,

    /// Time constant (seconds)
    pub time_constant: f32,
}

impl Default for ThermalCoefficients {
    fn default() -> Self {
        Self {
            thermal_resistance: 10.0,  // °C/W
            thermal_capacitance: 5.0,  // J/°C
            workload_power_coeff: 2.0, // W per unit
            cooling_coeff: 0.1,        // cooling rate
            time_constant: 30.0,       // 30 seconds
        }
    }
}

impl ThermalPredictionModel {
    /// Create new thermal prediction model
    pub fn new(prediction_horizon_secs: u32) -> Self {
        Self {
            temperature_history: VecDeque::with_capacity(60),
            workload_history: VecDeque::with_capacity(60),
            prediction_horizon_secs,
            coefficients: ThermalCoefficients::default(),
            ambient_temperature: 25.0, // Assume 25°C ambient
        }
    }

    /// Update model with new measurement
    pub fn update(&mut self, temperature: f32, workload_intensity: f32) {
        self.temperature_history.push_back(temperature);
        self.workload_history.push_back(workload_intensity);

        // Keep only recent history
        if self.temperature_history.len() > 60 {
            self.temperature_history.pop_front();
        }
        if self.workload_history.len() > 60 {
            self.workload_history.pop_front();
        }

        // Update ambient temperature estimate (running minimum)
        if let Some(&min_temp) = self
            .temperature_history
            .iter()
            .min_by(|a, b| a.partial_cmp(b).expect("Operation failed"))
        {
            self.ambient_temperature = self.ambient_temperature * 0.99 + min_temp * 0.01;
        }
    }

    /// Predict temperature after given workload
    pub fn predict_temperature(&self, future_workload: f32, horizon_secs: u32) -> f32 {
        let current_temp =
            self.temperature_history.back().copied().unwrap_or(self.ambient_temperature);

        // Simple exponential thermal model
        // T(t) = T_ambient + (T_current - T_ambient) * exp(-t/τ) + P*R*(1 - exp(-t/τ))
        let tau = self.coefficients.time_constant;
        let power = future_workload * self.coefficients.workload_power_coeff;
        let thermal_rise = power * self.coefficients.thermal_resistance;

        let t = horizon_secs as f32;
        let decay = (-t / tau).exp();

        self.ambient_temperature
            + (current_temp - self.ambient_temperature) * decay
            + thermal_rise * (1.0 - decay)
    }

    /// Predict if thermal throttling will be needed
    pub fn predict_throttle_needed(&self, future_workload: f32, threshold: f32) -> bool {
        let predicted_temp =
            self.predict_temperature(future_workload, self.prediction_horizon_secs);
        predicted_temp > threshold
    }

    /// Get temperature trend (°C/s)
    pub fn get_temperature_trend(&self) -> f32 {
        if self.temperature_history.len() < 2 {
            return 0.0;
        }

        let recent = self.temperature_history.iter().rev().take(10).copied().collect::<Vec<_>>();
        if recent.len() < 2 {
            return 0.0;
        }

        // Linear regression on recent samples
        let n = recent.len() as f32;
        let sum_x: f32 = (0..recent.len()).map(|i| i as f32).sum();
        let sum_y: f32 = recent.iter().sum();
        let sum_xy: f32 = recent.iter().enumerate().map(|(i, &y)| i as f32 * y).sum();
        let sum_x2: f32 = (0..recent.len()).map(|i| (i as f32).powi(2)).sum();

        (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2))
    }

    /// Calibrate model from historical data
    pub fn calibrate(&mut self) -> Result<()> {
        if self.temperature_history.len() < 10 || self.workload_history.len() < 10 {
            return Err(TrustformersError::runtime_error(
                "Insufficient data for calibration".to_string(),
            ));
        }

        // Simple parameter estimation
        let temp_range: f32 = self
            .temperature_history
            .iter()
            .max_by(|a, b| a.partial_cmp(b).expect("Operation failed"))
            .expect("Operation failed")
            - self
                .temperature_history
                .iter()
                .min_by(|a, b| a.partial_cmp(b).expect("Operation failed"))
                .expect("Operation failed");

        let workload_max = self
            .workload_history
            .iter()
            .max_by(|a, b| a.partial_cmp(b).expect("Operation failed"))
            .expect("Operation failed");

        if *workload_max > 0.0 {
            // Estimate thermal resistance from temperature range and workload
            self.coefficients.thermal_resistance =
                temp_range / (workload_max * self.coefficients.workload_power_coeff);
        }

        Ok(())
    }
}

/// Adaptive cooling strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveCoolingStrategy {
    /// Current cooling mode
    pub mode: CoolingMode,

    /// Cooldown rate (°C/s)
    pub cooldown_rate: f32,

    /// Active cooling available
    pub active_cooling: bool,

    /// Passive cooling efficiency
    pub passive_efficiency: f32,
}

/// Cooling modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoolingMode {
    /// No active cooling
    Passive,
    /// Low-power active cooling
    ActiveLow,
    /// Medium active cooling
    ActiveMedium,
    /// High-power active cooling
    ActiveHigh,
    /// Emergency cooling
    Emergency,
}

impl AdaptiveCoolingStrategy {
    /// Create new cooling strategy
    pub fn new(active_cooling: bool) -> Self {
        Self {
            mode: CoolingMode::Passive,
            cooldown_rate: 0.5, // 0.5°C/s passive cooling
            active_cooling,
            passive_efficiency: 1.0,
        }
    }

    /// Determine optimal cooling mode
    pub fn determine_mode(&mut self, temperature: f32, threshold: f32) -> CoolingMode {
        let temp_margin = temperature - threshold;

        if !self.active_cooling {
            return CoolingMode::Passive;
        }

        let mode = if temp_margin > 15.0 {
            CoolingMode::Emergency
        } else if temp_margin > 10.0 {
            CoolingMode::ActiveHigh
        } else if temp_margin > 5.0 {
            CoolingMode::ActiveMedium
        } else if temp_margin > 0.0 {
            CoolingMode::ActiveLow
        } else {
            CoolingMode::Passive
        };

        self.mode = mode;
        mode
    }

    /// Get cooldown rate for current mode
    pub fn get_cooldown_rate(&self) -> f32 {
        match self.mode {
            CoolingMode::Passive => self.cooldown_rate * self.passive_efficiency,
            CoolingMode::ActiveLow => self.cooldown_rate * 1.5,
            CoolingMode::ActiveMedium => self.cooldown_rate * 2.0,
            CoolingMode::ActiveHigh => self.cooldown_rate * 3.0,
            CoolingMode::Emergency => self.cooldown_rate * 5.0,
        }
    }
}

/// Multi-sensor thermal fusion
#[derive(Debug, Clone)]
pub struct MultiSensorThermalFusion {
    /// CPU temperature sensors
    cpu_sensors: Vec<f32>,

    /// GPU temperature sensors
    gpu_sensors: Vec<f32>,

    /// Battery temperature
    battery_temp: Option<f32>,

    /// Ambient sensor
    ambient_temp: Option<f32>,

    /// Sensor weights for fusion
    sensor_weights: SensorWeights,
}

/// Sensor weights for fusion algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorWeights {
    pub cpu_weight: f32,
    pub gpu_weight: f32,
    pub battery_weight: f32,
    pub ambient_weight: f32,
}

impl Default for SensorWeights {
    fn default() -> Self {
        Self {
            cpu_weight: 0.4,
            gpu_weight: 0.3,
            battery_weight: 0.2,
            ambient_weight: 0.1,
        }
    }
}

impl Default for MultiSensorThermalFusion {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiSensorThermalFusion {
    /// Create new sensor fusion system
    pub fn new() -> Self {
        Self {
            cpu_sensors: Vec::new(),
            gpu_sensors: Vec::new(),
            battery_temp: None,
            ambient_temp: None,
            sensor_weights: SensorWeights::default(),
        }
    }

    /// Update sensor readings
    pub fn update_sensors(
        &mut self,
        cpu: Vec<f32>,
        gpu: Vec<f32>,
        battery: Option<f32>,
        ambient: Option<f32>,
    ) {
        self.cpu_sensors = cpu;
        self.gpu_sensors = gpu;
        self.battery_temp = battery;
        self.ambient_temp = ambient;
    }

    /// Get fused temperature estimate
    pub fn get_fused_temperature(&self) -> f32 {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        // CPU sensors
        if !self.cpu_sensors.is_empty() {
            let cpu_max = self
                .cpu_sensors
                .iter()
                .max_by(|a, b| a.partial_cmp(b).expect("Operation failed"))
                .copied()
                .unwrap_or(0.0);
            weighted_sum += cpu_max * self.sensor_weights.cpu_weight;
            total_weight += self.sensor_weights.cpu_weight;
        }

        // GPU sensors
        if !self.gpu_sensors.is_empty() {
            let gpu_max = self
                .gpu_sensors
                .iter()
                .max_by(|a, b| a.partial_cmp(b).expect("Operation failed"))
                .copied()
                .unwrap_or(0.0);
            weighted_sum += gpu_max * self.sensor_weights.gpu_weight;
            total_weight += self.sensor_weights.gpu_weight;
        }

        // Battery
        if let Some(temp) = self.battery_temp {
            weighted_sum += temp * self.sensor_weights.battery_weight;
            total_weight += self.sensor_weights.battery_weight;
        }

        // Ambient
        if let Some(temp) = self.ambient_temp {
            weighted_sum += temp * self.sensor_weights.ambient_weight;
            total_weight += self.sensor_weights.ambient_weight;
        }

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            25.0 // Default room temperature
        }
    }

    /// Get thermal hotspot location
    pub fn get_hotspot(&self) -> ThermalHotspot {
        let cpu_max = self
            .cpu_sensors
            .iter()
            .max_by(|a, b| a.partial_cmp(b).expect("Operation failed"))
            .copied()
            .unwrap_or(0.0);
        let gpu_max = self
            .gpu_sensors
            .iter()
            .max_by(|a, b| a.partial_cmp(b).expect("Operation failed"))
            .copied()
            .unwrap_or(0.0);
        let battery = self.battery_temp.unwrap_or(0.0);

        if cpu_max >= gpu_max && cpu_max >= battery {
            ThermalHotspot::CPU
        } else if gpu_max >= battery {
            ThermalHotspot::GPU
        } else {
            ThermalHotspot::Battery
        }
    }
}

/// Thermal hotspot location
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThermalHotspot {
    CPU,
    GPU,
    Battery,
    Memory,
    NPU,
    Unknown,
}

/// Workload-aware thermal planner
#[derive(Debug, Clone)]
pub struct WorkloadThermalPlanner {
    /// Prediction model
    predictor: ThermalPredictionModel,

    /// Thermal budget (°C above ambient)
    thermal_budget: f32,

    /// Planning horizon (seconds)
    planning_horizon: u32,

    /// Workload queue
    workload_queue: VecDeque<PlannedWorkload>,
}

/// Planned workload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedWorkload {
    pub id: String,
    pub intensity: f32,
    pub duration_secs: u32,
    pub priority: WorkloadPriority,
    pub thermal_tolerance: f32,
}

/// Workload priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum WorkloadPriority {
    Low,
    Normal,
    High,
    Critical,
}

impl WorkloadThermalPlanner {
    /// Create new thermal planner
    pub fn new(thermal_budget: f32, planning_horizon: u32) -> Self {
        Self {
            predictor: ThermalPredictionModel::new(planning_horizon),
            thermal_budget,
            planning_horizon,
            workload_queue: VecDeque::new(),
        }
    }

    /// Add workload to queue
    pub fn add_workload(&mut self, workload: PlannedWorkload) {
        self.workload_queue.push_back(workload);
    }

    /// Plan workload execution considering thermal constraints
    pub fn plan_execution(&self, current_temp: f32, ambient_temp: f32) -> WorkloadExecutionPlan {
        let available_thermal_budget = self.thermal_budget - (current_temp - ambient_temp);

        let mut plan = WorkloadExecutionPlan {
            immediate_workloads: Vec::new(),
            delayed_workloads: Vec::new(),
            throttled_workloads: Vec::new(),
        };

        for workload in &self.workload_queue {
            let predicted_temp =
                self.predictor.predict_temperature(workload.intensity, workload.duration_secs);
            let temp_rise = predicted_temp - current_temp;

            if temp_rise <= available_thermal_budget * 0.5 {
                // Safe to execute immediately
                plan.immediate_workloads.push(workload.clone());
            } else if temp_rise <= available_thermal_budget {
                // Execute with throttling
                plan.throttled_workloads.push(workload.clone());
            } else {
                // Delay execution
                plan.delayed_workloads.push(workload.clone());
            }
        }

        plan
    }

    /// Update with new measurements
    pub fn update(&mut self, temperature: f32, workload_intensity: f32) {
        self.predictor.update(temperature, workload_intensity);
    }
}

/// Workload execution plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadExecutionPlan {
    pub immediate_workloads: Vec<PlannedWorkload>,
    pub throttled_workloads: Vec<PlannedWorkload>,
    pub delayed_workloads: Vec<PlannedWorkload>,
}

/// Advanced thermal manager combining all algorithms
pub struct AdvancedThermalManager {
    prediction_model: ThermalPredictionModel,
    cooling_strategy: AdaptiveCoolingStrategy,
    sensor_fusion: MultiSensorThermalFusion,
    workload_planner: WorkloadThermalPlanner,
    config: ThermalPowerConfig,
}

impl AdvancedThermalManager {
    /// Create new advanced thermal manager
    pub fn new(config: ThermalPowerConfig, device_info: &MobileDeviceInfo) -> Result<Self> {
        // Determine if device has active cooling (rare on mobile)
        let active_cooling = false; // Most mobile devices use passive cooling

        Ok(Self {
            prediction_model: ThermalPredictionModel::new(30),
            cooling_strategy: AdaptiveCoolingStrategy::new(active_cooling),
            sensor_fusion: MultiSensorThermalFusion::new(),
            workload_planner: WorkloadThermalPlanner::new(20.0, 60),
            config,
        })
    }

    /// Update with new sensor readings
    pub fn update_sensors(
        &mut self,
        cpu_temps: Vec<f32>,
        gpu_temps: Vec<f32>,
        battery_temp: Option<f32>,
        workload_intensity: f32,
    ) {
        // Update sensor fusion
        self.sensor_fusion.update_sensors(cpu_temps, gpu_temps, battery_temp, None);

        // Get fused temperature
        let fused_temp = self.sensor_fusion.get_fused_temperature();

        // Update prediction model
        self.prediction_model.update(fused_temp, workload_intensity);

        // Update cooling strategy
        let threshold = self.config.thermal_thresholds.moderate_throttle_celsius;
        self.cooling_strategy.determine_mode(fused_temp, threshold);
    }

    /// Predict if workload will cause throttling
    pub fn predict_throttle(&self, workload_intensity: f32) -> bool {
        let threshold = self.config.thermal_thresholds.light_throttle_celsius;
        self.prediction_model.predict_throttle_needed(workload_intensity, threshold)
    }

    /// Get recommended workload intensity
    pub fn recommend_workload_intensity(&self) -> f32 {
        let current_temp = self.sensor_fusion.get_fused_temperature();
        let threshold = self.config.thermal_thresholds.moderate_throttle_celsius;
        let margin = threshold - current_temp;

        if margin > 10.0 {
            1.0 // Full intensity
        } else if margin > 5.0 {
            0.75 // 75% intensity
        } else if margin > 0.0 {
            0.5 // 50% intensity
        } else {
            0.25 // 25% intensity
        }
    }

    /// Get current thermal hotspot
    pub fn get_hotspot(&self) -> ThermalHotspot {
        self.sensor_fusion.get_hotspot()
    }

    /// Get temperature trend
    pub fn get_temperature_trend(&self) -> f32 {
        self.prediction_model.get_temperature_trend()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_prediction_model() {
        let mut model = ThermalPredictionModel::new(30);

        // Add some measurements
        model.update(30.0, 0.5);
        model.update(32.0, 0.7);
        model.update(35.0, 0.9);

        // Predict future temperature
        let predicted = model.predict_temperature(1.0, 30);
        assert!(predicted > 35.0); // Should be higher with increased workload
    }

    #[test]
    fn test_cooling_strategy() {
        let mut strategy = AdaptiveCoolingStrategy::new(false);

        let mode = strategy.determine_mode(60.0, 45.0);
        assert_eq!(mode, CoolingMode::Passive); // No active cooling available

        let mut strategy_active = AdaptiveCoolingStrategy::new(true);
        let mode = strategy_active.determine_mode(60.0, 45.0);
        assert_eq!(mode, CoolingMode::ActiveHigh); // 15°C over threshold
    }

    #[test]
    fn test_sensor_fusion() {
        let mut fusion = MultiSensorThermalFusion::new();

        fusion.update_sensors(
            vec![40.0, 42.0, 38.0],
            vec![45.0, 47.0],
            Some(35.0),
            Some(25.0),
        );

        let fused = fusion.get_fused_temperature();
        assert!(fused > 35.0 && fused < 50.0);

        let hotspot = fusion.get_hotspot();
        assert_eq!(hotspot, ThermalHotspot::GPU); // GPU has highest temp
    }

    #[test]
    fn test_workload_planner() {
        let mut planner = WorkloadThermalPlanner::new(20.0, 60);

        let workload = PlannedWorkload {
            id: "test".to_string(),
            intensity: 0.8,
            duration_secs: 10,
            priority: WorkloadPriority::Normal,
            thermal_tolerance: 5.0,
        };

        planner.add_workload(workload);

        let plan = planner.plan_execution(30.0, 25.0);
        assert!(
            plan.immediate_workloads.len()
                + plan.throttled_workloads.len()
                + plan.delayed_workloads.len()
                > 0
        );
    }
}
