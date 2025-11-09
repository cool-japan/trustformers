//! Thermal Throttling and Power Management for Mobile Inference
//!
//! This module provides comprehensive thermal and power management capabilities
//! for mobile ML inference, including dynamic throttling, power-aware scheduling,
//! and adaptive performance scaling.

use crate::{
    device_info::{ChargingStatus, MobileDeviceInfo, ThermalState},
    MobileConfig,
};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use trustformers_core::error::{CoreError, Result};
use trustformers_core::TrustformersError;

/// Thermal and power management system for mobile inference
pub struct ThermalPowerManager {
    config: ThermalPowerConfig,
    thermal_monitor: ThermalMonitor,
    power_monitor: PowerMonitor,
    throttling_controller: ThrottlingController,
    inference_scheduler: PowerAwareScheduler,
    stats: ThermalPowerStats,
}

/// Configuration for thermal and power management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalPowerConfig {
    /// Enable thermal monitoring
    pub enable_thermal_monitoring: bool,
    /// Enable power monitoring
    pub enable_power_monitoring: bool,
    /// Thermal monitoring interval (ms)
    pub thermal_check_interval_ms: u64,
    /// Power monitoring interval (ms)
    pub power_check_interval_ms: u64,
    /// Temperature thresholds for throttling
    pub thermal_thresholds: ThermalThresholds,
    /// Power thresholds for optimization
    pub power_thresholds: PowerThresholds,
    /// Throttling strategy
    pub throttling_strategy: ThrottlingStrategy,
    /// Power optimization strategy
    pub power_strategy: PowerOptimizationStrategy,
    /// Maximum thermal history size
    pub max_thermal_history: usize,
    /// Maximum power history size
    pub max_power_history: usize,
}

/// Temperature thresholds for different actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalThresholds {
    /// Start light throttling (°C)
    pub light_throttle_celsius: f32,
    /// Start moderate throttling (°C)
    pub moderate_throttle_celsius: f32,
    /// Start aggressive throttling (°C)
    pub aggressive_throttle_celsius: f32,
    /// Emergency shutdown threshold (°C)
    pub emergency_celsius: f32,
    /// Cool-down threshold to reduce throttling (°C)
    pub cooldown_celsius: f32,
}

/// Power thresholds for different optimization levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerThresholds {
    /// Low battery threshold (%)
    pub low_battery_percent: u8,
    /// Critical battery threshold (%)
    pub critical_battery_percent: u8,
    /// Power save mode activation (%)
    pub power_save_percent: u8,
    /// Maximum power consumption (mW)
    pub max_power_mw: Option<f32>,
}

/// Throttling strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThrottlingStrategy {
    /// Conservative throttling (prioritize device safety)
    Conservative,
    /// Balanced throttling (balance performance and safety)
    Balanced,
    /// Aggressive throttling (prioritize performance)
    Aggressive,
    /// Custom throttling with user-defined parameters
    Custom,
}

/// Power optimization strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PowerOptimizationStrategy {
    /// Maximize battery life
    MaxBatteryLife,
    /// Balance performance and battery
    Balanced,
    /// Maximize performance (ignore battery)
    MaxPerformance,
    /// Adaptive based on charging state
    Adaptive,
}

/// Thermal monitoring system
struct ThermalMonitor {
    current_state: ThermalState,
    temperature_history: VecDeque<TemperatureReading>,
    last_check: Instant,
    check_interval: Duration,
}

/// Temperature reading with timestamp
#[derive(Debug, Clone)]
pub struct TemperatureReading {
    pub timestamp: Instant,
    pub temperature_celsius: f32,
    pub thermal_state: ThermalState,
    pub sensor_name: String,
}

/// Power monitoring system
struct PowerMonitor {
    battery_level: Option<u8>,
    charging_status: ChargingStatus,
    power_consumption_mw: Option<f32>,
    power_history: VecDeque<PowerReading>,
    last_check: Instant,
    check_interval: Duration,
}

/// Power reading with timestamp
#[derive(Debug, Clone)]
pub struct PowerReading {
    timestamp: Instant,
    battery_level: Option<u8>,
    charging_status: ChargingStatus,
    power_consumption_mw: Option<f32>,
    estimated_time_remaining_minutes: Option<u32>,
}

/// Throttling controller for dynamic performance adjustment
struct ThrottlingController {
    current_throttle_level: ThrottleLevel,
    base_config: MobileConfig,
    throttled_config: MobileConfig,
    throttle_history: VecDeque<ThrottleEvent>,
}

/// Throttling levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ThrottleLevel {
    None,
    Light,
    Moderate,
    Aggressive,
    Emergency,
}

/// Throttling event
#[derive(Debug, Clone)]
struct ThrottleEvent {
    timestamp: Instant,
    level: ThrottleLevel,
    reason: ThrottleReason,
    config_changes: Vec<String>,
}

/// Reason for throttling
#[derive(Debug, Clone, PartialEq, Eq)]
enum ThrottleReason {
    ThermalPressure,
    PowerConstraint,
    BatteryLow,
    CombinedFactors,
}

/// Power-aware inference scheduler
struct PowerAwareScheduler {
    inference_queue: VecDeque<InferenceRequest>,
    scheduled_inferences: VecDeque<ScheduledInference>,
    scheduling_strategy: SchedulingStrategy,
}

/// Inference request
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    id: String,
    priority: InferencePriority,
    estimated_duration_ms: u64,
    power_budget_mw: Option<f32>,
    deadline: Option<Instant>,
}

/// Scheduled inference with timing
#[derive(Debug, Clone)]
pub struct ScheduledInference {
    request: InferenceRequest,
    scheduled_time: Instant,
    expected_completion: Instant,
    config: MobileConfig,
}

/// Inference priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum InferencePriority {
    Background,
    Normal,
    High,
    Critical,
}

/// Scheduling strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SchedulingStrategy {
    FIFO,         // First in, first out
    Priority,     // Priority-based
    PowerAware,   // Consider power constraints
    ThermalAware, // Consider thermal constraints
    Adaptive,     // Adapt based on conditions
}

/// Thermal and power management statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalPowerStats {
    /// Total monitoring time (seconds)
    pub total_monitoring_time_seconds: f64,
    /// Average temperature (°C)
    pub avg_temperature_celsius: f32,
    /// Peak temperature (°C)
    pub peak_temperature_celsius: f32,
    /// Time spent in each thermal state (seconds)
    pub thermal_state_durations: std::collections::HashMap<String, f64>,
    /// Total throttling events
    pub total_throttle_events: usize,
    /// Time spent throttled (seconds)
    pub total_throttle_time_seconds: f64,
    /// Average battery level (%)
    pub avg_battery_level: Option<f32>,
    /// Average power consumption (mW)
    pub avg_power_consumption_mw: Option<f32>,
    /// Power saved through optimization (mWh)
    pub power_saved_mwh: f32,
    /// Inference throughput degradation (%)
    pub throughput_degradation_percent: f32,
}

impl Default for ThermalPowerConfig {
    fn default() -> Self {
        Self {
            enable_thermal_monitoring: true,
            enable_power_monitoring: true,
            thermal_check_interval_ms: 1000, // Check every second
            power_check_interval_ms: 5000,   // Check every 5 seconds
            thermal_thresholds: ThermalThresholds::default(),
            power_thresholds: PowerThresholds::default(),
            throttling_strategy: ThrottlingStrategy::Balanced,
            power_strategy: PowerOptimizationStrategy::Adaptive,
            max_thermal_history: 1000,
            max_power_history: 500,
        }
    }
}

impl ThermalPowerConfig {
    /// Create a power-saving optimized configuration
    /// This configuration prioritizes battery life and thermal management
    pub fn power_saving() -> Self {
        Self {
            enable_thermal_monitoring: true,
            enable_power_monitoring: true,
            thermal_check_interval_ms: 500, // More frequent checks for better safety
            power_check_interval_ms: 2000,  // More frequent power monitoring
            thermal_thresholds: ThermalThresholds {
                light_throttle_celsius: 55.0, // Very conservative thermal limits
                moderate_throttle_celsius: 60.0,
                aggressive_throttle_celsius: 65.0,
                emergency_celsius: 70.0, // Lower emergency threshold
                cooldown_celsius: 50.0,
            },
            power_thresholds: PowerThresholds {
                low_battery_percent: 30, // Higher threshold for power saving
                critical_battery_percent: 15,
                power_save_percent: 50,     // Enter power save mode earlier
                max_power_mw: Some(2000.0), // Very conservative power limit (2W)
            },
            throttling_strategy: ThrottlingStrategy::Conservative,
            power_strategy: PowerOptimizationStrategy::MaxBatteryLife,
            max_thermal_history: 1500, // More history for better analysis
            max_power_history: 1000,
        }
    }

    /// Create a balanced performance configuration
    /// This configuration balances performance and battery life
    pub fn balanced() -> Self {
        Self {
            enable_thermal_monitoring: true,
            enable_power_monitoring: true,
            thermal_check_interval_ms: 1000, // Standard check interval
            power_check_interval_ms: 5000,
            thermal_thresholds: ThermalThresholds::default(), // Use default balanced thresholds
            power_thresholds: PowerThresholds::default(),
            throttling_strategy: ThrottlingStrategy::Balanced,
            power_strategy: PowerOptimizationStrategy::Balanced,
            max_thermal_history: 1000,
            max_power_history: 500,
        }
    }

    /// Create a high-performance configuration
    /// This configuration prioritizes maximum performance
    pub fn high_performance() -> Self {
        Self {
            enable_thermal_monitoring: true, // Still monitor for safety
            enable_power_monitoring: false,  // Don't limit for power concerns
            thermal_check_interval_ms: 2000, // Less frequent checks for better performance
            power_check_interval_ms: 10000,  // Less frequent power monitoring
            thermal_thresholds: ThermalThresholds {
                light_throttle_celsius: 75.0, // Allow higher temperatures
                moderate_throttle_celsius: 80.0,
                aggressive_throttle_celsius: 85.0,
                emergency_celsius: 90.0, // Higher emergency threshold
                cooldown_celsius: 70.0,
            },
            power_thresholds: PowerThresholds {
                low_battery_percent: 10, // Lower threshold, prioritize performance
                critical_battery_percent: 5,
                power_save_percent: 15, // Only enter power save at very low battery
                max_power_mw: Some(10000.0), // Allow higher power consumption (10W)
            },
            throttling_strategy: ThrottlingStrategy::Aggressive, // Aggressive = prioritize performance
            power_strategy: PowerOptimizationStrategy::MaxPerformance,
            max_thermal_history: 500, // Less history for better performance
            max_power_history: 250,
        }
    }
}

impl Default for ThermalThresholds {
    fn default() -> Self {
        Self {
            light_throttle_celsius: 65.0,      // Start light throttling at 65°C
            moderate_throttle_celsius: 70.0,   // Moderate throttling at 70°C
            aggressive_throttle_celsius: 75.0, // Aggressive throttling at 75°C
            emergency_celsius: 80.0,           // Emergency at 80°C
            cooldown_celsius: 60.0,            // Cool down to 60°C before reducing throttling
        }
    }
}

impl Default for PowerThresholds {
    fn default() -> Self {
        Self {
            low_battery_percent: 20,      // Low battery at 20%
            critical_battery_percent: 10, // Critical at 10%
            power_save_percent: 30,       // Power save mode at 30%
            max_power_mw: Some(5000.0),   // Max 5W power consumption
        }
    }
}

impl ThermalPowerManager {
    /// Create new thermal and power manager
    pub fn new(config: ThermalPowerConfig, device_info: &MobileDeviceInfo) -> Result<Self> {
        let thermal_monitor = ThermalMonitor::new(
            Duration::from_millis(config.thermal_check_interval_ms),
            config.max_thermal_history,
        );

        let power_monitor = PowerMonitor::new(
            Duration::from_millis(config.power_check_interval_ms),
            config.max_power_history,
        );

        let throttling_controller = ThrottlingController::new();
        let inference_scheduler = PowerAwareScheduler::new();
        let stats = ThermalPowerStats::new();

        Ok(Self {
            config,
            thermal_monitor,
            power_monitor,
            throttling_controller,
            inference_scheduler,
            stats,
        })
    }

    /// Start monitoring thermal and power conditions
    pub fn start_monitoring(&mut self) -> Result<()> {
        if self.config.enable_thermal_monitoring {
            self.thermal_monitor.start()?;
            tracing::info!("Thermal monitoring started");
        }

        if self.config.enable_power_monitoring {
            self.power_monitor.start()?;
            tracing::info!("Power monitoring started");
        }

        Ok(())
    }

    /// Stop monitoring
    pub fn stop_monitoring(&mut self) {
        self.thermal_monitor.stop();
        self.power_monitor.stop();
        tracing::info!("Thermal and power monitoring stopped");
    }

    /// Update monitoring data and adjust configuration if needed
    pub fn update(&mut self, mobile_config: &mut MobileConfig) -> Result<bool> {
        let mut config_changed = false;

        // Update thermal monitoring
        if self.config.enable_thermal_monitoring {
            self.thermal_monitor.update()?;

            // Check if thermal throttling is needed
            if let Some(new_throttle_level) = self.evaluate_thermal_throttling()? {
                if new_throttle_level != self.throttling_controller.current_throttle_level {
                    self.apply_thermal_throttling(mobile_config, new_throttle_level)?;
                    config_changed = true;
                }
            }
        }

        // Update power monitoring
        if self.config.enable_power_monitoring {
            self.power_monitor.update()?;

            // Check if power optimization is needed
            if self.evaluate_power_optimization()? {
                self.apply_power_optimization(mobile_config)?;
                config_changed = true;
            }
        }

        // Update statistics
        self.update_stats();

        Ok(config_changed)
    }

    /// Schedule inference request with power/thermal awareness
    pub fn schedule_inference(
        &mut self,
        request: InferenceRequest,
        current_config: &MobileConfig,
    ) -> Result<Option<ScheduledInference>> {
        // Check if we can run inference immediately
        if self.can_run_inference_now(&request, current_config)? {
            let scheduled = ScheduledInference {
                scheduled_time: Instant::now(),
                expected_completion: Instant::now()
                    + Duration::from_millis(request.estimated_duration_ms),
                config: current_config.clone(),
                request,
            };
            Ok(Some(scheduled))
        } else {
            // Queue for later execution
            self.inference_scheduler.queue_request(request);
            Ok(None)
        }
    }

    /// Get next scheduled inference if conditions allow
    pub fn get_next_inference(
        &mut self,
        current_config: &MobileConfig,
    ) -> Option<ScheduledInference> {
        self.inference_scheduler.get_next_ready_inference(current_config)
    }

    /// Get current thermal and power statistics
    pub fn get_stats(&self) -> &ThermalPowerStats {
        &self.stats
    }

    /// Get current thermal reading
    pub fn get_current_reading(&self) -> Result<TemperatureReading> {
        self.thermal_monitor
            .temperature_history
            .back()
            .cloned()
            .ok_or_else(|| TrustformersError::runtime_error("No thermal reading available".into()))
            .map_err(|e| e.into())
    }

    /// Get current power consumption in mW
    /// This method addresses TODO in mobile testing framework
    pub fn get_current_power(&self) -> Option<f32> {
        self.power_monitor
            .power_history
            .back()
            .and_then(|reading| reading.power_consumption_mw)
    }

    /// Get thermal state history
    pub fn get_thermal_history(&self) -> Vec<TemperatureReading> {
        self.thermal_monitor.temperature_history.iter().cloned().collect()
    }

    /// Get power history
    pub fn get_power_history(&self) -> Vec<PowerReading> {
        self.power_monitor.power_history.iter().cloned().collect()
    }

    /// Create optimized configuration for current thermal/power state
    pub fn create_optimized_config(&self, base_config: &MobileConfig) -> MobileConfig {
        let mut optimized = base_config.clone();

        // Apply thermal optimizations
        self.apply_thermal_optimizations(&mut optimized);

        // Apply power optimizations
        self.apply_power_optimizations(&mut optimized);

        optimized
    }

    // Private implementation methods

    fn evaluate_thermal_throttling(&self) -> Result<Option<ThrottleLevel>> {
        let current_temp = self.thermal_monitor.get_current_temperature()?;
        let thermal_state = self.thermal_monitor.current_state;

        let new_level = match thermal_state {
            ThermalState::Critical | ThermalState::Emergency => ThrottleLevel::Emergency,
            ThermalState::Serious => {
                if current_temp >= self.config.thermal_thresholds.aggressive_throttle_celsius {
                    ThrottleLevel::Aggressive
                } else {
                    ThrottleLevel::Moderate
                }
            },
            ThermalState::Fair => {
                if current_temp >= self.config.thermal_thresholds.moderate_throttle_celsius {
                    ThrottleLevel::Moderate
                } else if current_temp >= self.config.thermal_thresholds.light_throttle_celsius {
                    ThrottleLevel::Light
                } else {
                    ThrottleLevel::None
                }
            },
            ThermalState::Nominal => {
                if current_temp <= self.config.thermal_thresholds.cooldown_celsius {
                    ThrottleLevel::None
                } else {
                    self.throttling_controller.current_throttle_level // Maintain current level
                }
            },
            _ => ThrottleLevel::None,
        };

        Ok(Some(new_level))
    }

    fn apply_thermal_throttling(
        &mut self,
        config: &mut MobileConfig,
        level: ThrottleLevel,
    ) -> Result<()> {
        let changes = match level {
            ThrottleLevel::None => {
                // Restore base configuration
                *config = self.throttling_controller.base_config.clone();
                vec!["Restored base configuration".to_string()]
            },
            ThrottleLevel::Light => {
                // Light throttling: reduce threads by 25%
                config.num_threads = (config.num_threads * 3 / 4).max(1);
                config.max_batch_size = (config.max_batch_size * 3 / 4).max(1);
                vec!["Reduced threads and batch size by 25%".to_string()]
            },
            ThrottleLevel::Moderate => {
                // Moderate throttling: reduce threads by 50%, enable memory optimization
                config.num_threads = (config.num_threads / 2).max(1);
                config.max_batch_size = 1;
                config.memory_optimization = crate::MemoryOptimization::Balanced;
                config.enable_batching = false;
                vec!["Reduced threads by 50%, disabled batching".to_string()]
            },
            ThrottleLevel::Aggressive => {
                // Aggressive throttling: single thread, maximum memory optimization
                config.num_threads = 1;
                config.max_batch_size = 1;
                config.memory_optimization = crate::MemoryOptimization::Maximum;
                config.enable_batching = false;
                config.backend = crate::MobileBackend::CPU; // Prefer CPU over GPU/NPU
                vec!["Single thread, CPU only, maximum memory optimization".to_string()]
            },
            ThrottleLevel::Emergency => {
                // Emergency: halt inference temporarily
                config.num_threads = 0; // Special case to halt inference
                vec!["Emergency throttling: inference halted".to_string()]
            },
        };

        self.throttling_controller.current_throttle_level = level;
        self.throttling_controller.throttle_history.push_back(ThrottleEvent {
            timestamp: Instant::now(),
            level,
            reason: ThrottleReason::ThermalPressure,
            config_changes: changes,
        });

        // Limit history size
        while self.throttling_controller.throttle_history.len() > 100 {
            self.throttling_controller.throttle_history.pop_front();
        }

        tracing::info!("Applied thermal throttling level: {:?}", level);
        Ok(())
    }

    fn evaluate_power_optimization(&self) -> Result<bool> {
        let battery_level = self.power_monitor.battery_level.unwrap_or(100);
        let charging = matches!(self.power_monitor.charging_status, ChargingStatus::Charging);
        let power_consumption = self.power_monitor.power_consumption_mw.unwrap_or(0.0);

        // Check if power optimization is needed
        let needs_optimization = match self.config.power_strategy {
            PowerOptimizationStrategy::MaxBatteryLife => !charging,
            PowerOptimizationStrategy::Balanced => {
                battery_level < self.config.power_thresholds.low_battery_percent && !charging
            },
            PowerOptimizationStrategy::MaxPerformance => false,
            PowerOptimizationStrategy::Adaptive => {
                (!charging && battery_level < self.config.power_thresholds.power_save_percent)
                    || (self
                        .config
                        .power_thresholds
                        .max_power_mw
                        .is_some_and(|max| power_consumption > max))
            },
        };

        Ok(needs_optimization)
    }

    fn apply_power_optimization(&mut self, config: &mut MobileConfig) -> Result<()> {
        let battery_level = self.power_monitor.battery_level.unwrap_or(100);

        if battery_level < self.config.power_thresholds.critical_battery_percent {
            // Critical battery: aggressive power saving
            config.memory_optimization = crate::MemoryOptimization::Maximum;
            config.num_threads = 1;
            config.enable_batching = false;
            config.backend = crate::MobileBackend::CPU;

            tracing::warn!("Critical battery level: applying aggressive power optimization");
        } else if battery_level < self.config.power_thresholds.low_battery_percent {
            // Low battery: moderate power saving
            config.num_threads = (config.num_threads / 2).max(1);
            config.memory_optimization = crate::MemoryOptimization::Balanced;

            tracing::info!("Low battery level: applying moderate power optimization");
        }

        Ok(())
    }

    fn can_run_inference_now(
        &self,
        request: &InferenceRequest,
        config: &MobileConfig,
    ) -> Result<bool> {
        // Check thermal constraints
        if self.thermal_monitor.current_state == ThermalState::Emergency {
            return Ok(false);
        }

        // Check if threads are available (emergency throttling sets threads to 0)
        if config.num_threads == 0 {
            return Ok(false);
        }

        // Check power constraints
        if let Some(power_budget) = request.power_budget_mw {
            if let Some(current_power) = self.power_monitor.power_consumption_mw {
                if current_power + power_budget
                    > self.config.power_thresholds.max_power_mw.unwrap_or(f32::MAX)
                {
                    return Ok(false);
                }
            }
        }

        // Check battery level for non-critical requests
        if matches!(
            request.priority,
            InferencePriority::Background | InferencePriority::Normal
        ) {
            if let Some(battery) = self.power_monitor.battery_level {
                if battery < self.config.power_thresholds.critical_battery_percent {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    fn apply_thermal_optimizations(&self, config: &mut MobileConfig) {
        match self.thermal_monitor.current_state {
            ThermalState::Serious | ThermalState::Critical => {
                config.memory_optimization = crate::MemoryOptimization::Maximum;
                config.num_threads = (config.num_threads / 2).max(1);
                config.enable_batching = false;
            },
            ThermalState::Fair => {
                config.num_threads = (config.num_threads * 3 / 4).max(1);
            },
            _ => {},
        }
    }

    fn apply_power_optimizations(&self, config: &mut MobileConfig) {
        if let Some(battery) = self.power_monitor.battery_level {
            if battery < self.config.power_thresholds.low_battery_percent {
                config.memory_optimization = crate::MemoryOptimization::Maximum;
                config.backend = crate::MobileBackend::CPU; // CPU is usually more power efficient

                if battery < self.config.power_thresholds.critical_battery_percent {
                    config.num_threads = 1;
                    config.enable_batching = false;
                }
            }
        }
    }

    fn update_stats(&mut self) {
        // Update thermal statistics
        if let Ok(current_temp) = self.thermal_monitor.get_current_temperature() {
            self.stats.peak_temperature_celsius =
                self.stats.peak_temperature_celsius.max(current_temp);

            // Update average temperature
            let history_len = self.thermal_monitor.temperature_history.len() as f32;
            if history_len > 0.0 {
                let sum: f32 = self
                    .thermal_monitor
                    .temperature_history
                    .iter()
                    .map(|r| r.temperature_celsius)
                    .sum();
                self.stats.avg_temperature_celsius = sum / history_len;
            }
        }

        // Update power statistics
        if let Some(battery) = self.power_monitor.battery_level {
            if let Some(ref mut avg_battery) = self.stats.avg_battery_level {
                *avg_battery = (*avg_battery + battery as f32) / 2.0;
            } else {
                self.stats.avg_battery_level = Some(battery as f32);
            }
        }

        if let Some(power) = self.power_monitor.power_consumption_mw {
            if let Some(ref mut avg_power) = self.stats.avg_power_consumption_mw {
                *avg_power = (*avg_power + power) / 2.0;
            } else {
                self.stats.avg_power_consumption_mw = Some(power);
            }
        }

        // Update throttling statistics
        self.stats.total_throttle_events = self.throttling_controller.throttle_history.len();
    }
}

// Implementation for component types

impl ThermalMonitor {
    fn new(check_interval: Duration, max_history: usize) -> Self {
        Self {
            current_state: ThermalState::Nominal,
            temperature_history: VecDeque::with_capacity(max_history),
            last_check: Instant::now(),
            check_interval,
        }
    }

    fn start(&mut self) -> Result<()> {
        self.last_check = Instant::now();
        Ok(())
    }

    fn stop(&mut self) {
        // Nothing to do for stop
    }

    fn update(&mut self) -> Result<()> {
        if self.last_check.elapsed() >= self.check_interval {
            let temperature = self.read_temperature()?;
            let thermal_state = Self::temperature_to_state(temperature);

            let reading = TemperatureReading {
                timestamp: Instant::now(),
                temperature_celsius: temperature,
                thermal_state,
                sensor_name: "CPU".to_string(), // Simplified
            };

            self.temperature_history.push_back(reading);

            // Limit history size
            while self.temperature_history.len() > self.temperature_history.capacity() {
                self.temperature_history.pop_front();
            }

            self.current_state = thermal_state;
            self.last_check = Instant::now();
        }

        Ok(())
    }

    fn get_current_temperature(&self) -> Result<f32> {
        self.temperature_history
            .back()
            .map(|r| r.temperature_celsius)
            .ok_or_else(|| {
                TrustformersError::runtime_error("No temperature readings available".into())
            })
            .map_err(|e| e.into())
    }

    fn read_temperature(&self) -> Result<f32> {
        // Platform-specific temperature reading
        #[cfg(target_os = "android")]
        {
            self.read_android_temperature()
        }

        #[cfg(target_os = "ios")]
        {
            self.read_ios_temperature()
        }

        #[cfg(not(any(target_os = "android", target_os = "ios")))]
        {
            // Simulate temperature for testing
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            self.last_check.elapsed().as_millis().hash(&mut hasher);
            let hash_val = hasher.finish();
            let variation = (hash_val % 20) as f32; // 0-19°C variation
            Ok(45.0 + variation) // 45-64°C
        }
    }

    #[cfg(target_os = "android")]
    fn read_android_temperature(&self) -> Result<f32> {
        // Read from /sys/class/thermal/thermal_zone*/temp
        // This is a simplified implementation
        Ok(50.0) // Placeholder
    }

    #[cfg(target_os = "ios")]
    fn read_ios_temperature(&self) -> Result<f32> {
        // Use iOS thermal state APIs
        // This is a simplified implementation
        Ok(48.0) // Placeholder
    }

    fn temperature_to_state(temperature: f32) -> ThermalState {
        match temperature {
            t if t < 55.0 => ThermalState::Nominal,
            t if t < 65.0 => ThermalState::Fair,
            t if t < 75.0 => ThermalState::Serious,
            t if t < 85.0 => ThermalState::Critical,
            _ => ThermalState::Emergency,
        }
    }
}

impl PowerMonitor {
    fn new(check_interval: Duration, max_history: usize) -> Self {
        Self {
            battery_level: None,
            charging_status: ChargingStatus::Unknown,
            power_consumption_mw: None,
            power_history: VecDeque::with_capacity(max_history),
            last_check: Instant::now(),
            check_interval,
        }
    }

    fn start(&mut self) -> Result<()> {
        self.last_check = Instant::now();
        Ok(())
    }

    fn stop(&mut self) {
        // Nothing to do for stop
    }

    fn update(&mut self) -> Result<()> {
        if self.last_check.elapsed() >= self.check_interval {
            let (battery_level, charging_status, power_consumption) = self.read_power_info()?;

            let reading = PowerReading {
                timestamp: Instant::now(),
                battery_level,
                charging_status,
                power_consumption_mw: power_consumption,
                estimated_time_remaining_minutes: self
                    .estimate_time_remaining(battery_level, power_consumption),
            };

            self.power_history.push_back(reading);

            // Limit history size
            while self.power_history.len() > self.power_history.capacity() {
                self.power_history.pop_front();
            }

            self.battery_level = battery_level;
            self.charging_status = charging_status;
            self.power_consumption_mw = power_consumption;
            self.last_check = Instant::now();
        }

        Ok(())
    }

    fn read_power_info(&self) -> Result<(Option<u8>, ChargingStatus, Option<f32>)> {
        // Platform-specific power reading
        #[cfg(target_os = "android")]
        {
            self.read_android_power_info()
        }

        #[cfg(target_os = "ios")]
        {
            self.read_ios_power_info()
        }

        #[cfg(not(any(target_os = "android", target_os = "ios")))]
        {
            // Simulate power info for testing
            let battery = Some(75u8); // 75% battery
            let charging = ChargingStatus::Discharging;
            let power = Some(2500.0); // 2.5W consumption
            Ok((battery, charging, power))
        }
    }

    #[cfg(target_os = "android")]
    fn read_android_power_info(&self) -> Result<(Option<u8>, ChargingStatus, Option<f32>)> {
        // Use Android BatteryManager APIs
        Ok((Some(80), ChargingStatus::Discharging, Some(2000.0)))
    }

    #[cfg(target_os = "ios")]
    fn read_ios_power_info(&self) -> Result<(Option<u8>, ChargingStatus, Option<f32>)> {
        // Use iOS UIDevice battery APIs
        Ok((Some(85), ChargingStatus::Discharging, Some(1800.0)))
    }

    fn estimate_time_remaining(
        &self,
        battery_level: Option<u8>,
        power_consumption: Option<f32>,
    ) -> Option<u32> {
        if let (Some(battery), Some(power)) = (battery_level, power_consumption) {
            if power > 0.0 {
                // Rough estimation: assume 3000mAh battery at 3.7V = ~11Wh
                let battery_energy_wh = 11.0 * (battery as f32 / 100.0);
                let power_w = power / 1000.0;
                let hours_remaining = battery_energy_wh / power_w;
                Some((hours_remaining * 60.0) as u32) // Convert to minutes
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl ThrottlingController {
    fn new() -> Self {
        Self {
            current_throttle_level: ThrottleLevel::None,
            base_config: MobileConfig::default(),
            throttled_config: MobileConfig::default(),
            throttle_history: VecDeque::new(),
        }
    }
}

impl PowerAwareScheduler {
    fn new() -> Self {
        Self {
            inference_queue: VecDeque::new(),
            scheduled_inferences: VecDeque::new(),
            scheduling_strategy: SchedulingStrategy::Adaptive,
        }
    }

    fn queue_request(&mut self, request: InferenceRequest) {
        // Insert in priority order
        let insert_pos = self
            .inference_queue
            .iter()
            .position(|r| r.priority < request.priority)
            .unwrap_or(self.inference_queue.len());

        self.inference_queue.insert(insert_pos, request);
    }

    fn get_next_ready_inference(&mut self, _config: &MobileConfig) -> Option<ScheduledInference> {
        // For now, just return the highest priority request
        self.inference_queue.pop_front().map(|request| {
            ScheduledInference {
                scheduled_time: Instant::now(),
                expected_completion: Instant::now()
                    + Duration::from_millis(request.estimated_duration_ms),
                config: MobileConfig::default(), // Would be optimized config
                request,
            }
        })
    }
}

impl ThermalPowerStats {
    fn new() -> Self {
        Self {
            total_monitoring_time_seconds: 0.0,
            avg_temperature_celsius: 0.0,
            peak_temperature_celsius: 0.0,
            thermal_state_durations: std::collections::HashMap::new(),
            total_throttle_events: 0,
            total_throttle_time_seconds: 0.0,
            avg_battery_level: None,
            avg_power_consumption_mw: None,
            power_saved_mwh: 0.0,
            throughput_degradation_percent: 0.0,
        }
    }
}

/// Utility functions for thermal and power management
pub struct ThermalPowerUtils;

impl ThermalPowerUtils {
    /// Create optimized thermal/power config for device
    pub fn create_optimized_config(device_info: &MobileDeviceInfo) -> ThermalPowerConfig {
        let mut config = ThermalPowerConfig::default();

        // Adjust based on device performance tier
        match device_info.performance_scores.overall_tier {
            crate::device_info::PerformanceTier::Budget => {
                config.throttling_strategy = ThrottlingStrategy::Conservative;
                config.power_strategy = PowerOptimizationStrategy::MaxBatteryLife;
                config.thermal_thresholds.light_throttle_celsius = 60.0; // More aggressive
            },
            crate::device_info::PerformanceTier::Flagship => {
                config.throttling_strategy = ThrottlingStrategy::Aggressive;
                config.power_strategy = PowerOptimizationStrategy::Balanced;
                config.thermal_thresholds.light_throttle_celsius = 70.0; // Less aggressive
            },
            _ => {
                // Use defaults for Mid and High tiers
            },
        }

        // Adjust for thermal capabilities
        if !device_info.thermal_info.throttling_supported {
            config.enable_thermal_monitoring = false;
        }

        config
    }

    /// Estimate power consumption for inference configuration
    pub fn estimate_power_consumption(
        config: &MobileConfig,
        device_info: &MobileDeviceInfo,
    ) -> f32 {
        let base_power = match config.backend {
            crate::MobileBackend::CPU => 2000.0,    // 2W for CPU
            crate::MobileBackend::GPU => 3500.0,    // 3.5W for GPU
            crate::MobileBackend::CoreML => 1500.0, // 1.5W for Neural Engine
            crate::MobileBackend::NNAPI => 2500.0,  // 2.5W for various NPUs
            crate::MobileBackend::Metal => 3200.0,  // 3.2W for Metal acceleration
            crate::MobileBackend::Vulkan => 3000.0, // 3W for Vulkan acceleration
            crate::MobileBackend::OpenCL => 3100.0, // 3.1W for OpenCL acceleration
            crate::MobileBackend::Custom => 2000.0, // Default estimate
        };

        // Scale by thread count (ensure at least some power usage)
        let thread_factor = if config.num_threads == 0 {
            0.5 // Minimum power even when auto-detect
        } else {
            (config.num_threads as f32 / device_info.cpu_info.total_cores as f32).max(0.25)
        };

        // Scale by batch size
        let batch_factor = if config.enable_batching {
            1.0 + (config.max_batch_size as f32 * 0.1)
        } else {
            1.0
        };

        base_power * thread_factor * batch_factor
    }

    /// Calculate thermal safety margin
    pub fn calculate_thermal_margin(current_temp: f32, thresholds: &ThermalThresholds) -> f32 {
        if current_temp >= thresholds.emergency_celsius {
            0.0 // No margin
        } else if current_temp >= thresholds.aggressive_throttle_celsius {
            (thresholds.emergency_celsius - current_temp)
                / (thresholds.emergency_celsius - thresholds.aggressive_throttle_celsius)
        } else {
            1.0 // Full margin
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_power_config() {
        let config = ThermalPowerConfig::default();
        assert!(config.enable_thermal_monitoring);
        assert!(config.enable_power_monitoring);
        assert_eq!(config.throttling_strategy, ThrottlingStrategy::Balanced);
    }

    #[test]
    fn test_thermal_thresholds() {
        let thresholds = ThermalThresholds::default();
        assert!(thresholds.light_throttle_celsius < thresholds.moderate_throttle_celsius);
        assert!(thresholds.moderate_throttle_celsius < thresholds.aggressive_throttle_celsius);
        assert!(thresholds.aggressive_throttle_celsius < thresholds.emergency_celsius);
    }

    #[test]
    fn test_power_thresholds() {
        let thresholds = PowerThresholds::default();
        assert!(thresholds.critical_battery_percent < thresholds.low_battery_percent);
        assert!(thresholds.low_battery_percent < thresholds.power_save_percent);
    }

    #[test]
    fn test_throttle_levels() {
        assert!(ThrottleLevel::None < ThrottleLevel::Light);
        assert!(ThrottleLevel::Light < ThrottleLevel::Moderate);
        assert!(ThrottleLevel::Moderate < ThrottleLevel::Aggressive);
        assert!(ThrottleLevel::Aggressive < ThrottleLevel::Emergency);
    }

    #[test]
    fn test_inference_priorities() {
        assert!(InferencePriority::Background < InferencePriority::Normal);
        assert!(InferencePriority::Normal < InferencePriority::High);
        assert!(InferencePriority::High < InferencePriority::Critical);
    }

    #[test]
    fn test_thermal_monitor() {
        let monitor = ThermalMonitor::new(Duration::from_secs(1), 100);
        assert_eq!(monitor.current_state, ThermalState::Nominal);
        assert!(monitor.temperature_history.is_empty());
    }

    #[test]
    fn test_power_consumption_estimation() {
        let config = MobileConfig::default();
        let device_info = crate::device_info::MobileDeviceDetector::detect().unwrap();

        let estimated_power = ThermalPowerUtils::estimate_power_consumption(&config, &device_info);
        assert!(estimated_power > 0.0);
        assert!(estimated_power < 10000.0); // Reasonable upper bound
    }

    #[test]
    fn test_thermal_margin_calculation() {
        let thresholds = ThermalThresholds::default();

        let margin_normal = ThermalPowerUtils::calculate_thermal_margin(50.0, &thresholds);
        assert_eq!(margin_normal, 1.0);

        let margin_emergency =
            ThermalPowerUtils::calculate_thermal_margin(thresholds.emergency_celsius, &thresholds);
        assert_eq!(margin_emergency, 0.0);
    }

    #[test]
    fn test_temperature_to_thermal_state() {
        assert_eq!(
            ThermalMonitor::temperature_to_state(45.0),
            ThermalState::Nominal
        );
        assert_eq!(
            ThermalMonitor::temperature_to_state(60.0),
            ThermalState::Fair
        );
        assert_eq!(
            ThermalMonitor::temperature_to_state(70.0),
            ThermalState::Serious
        );
        assert_eq!(
            ThermalMonitor::temperature_to_state(80.0),
            ThermalState::Critical
        );
        assert_eq!(
            ThermalMonitor::temperature_to_state(90.0),
            ThermalState::Emergency
        );
    }
}
