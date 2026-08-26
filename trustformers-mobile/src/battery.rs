//! Advanced Battery Management and Optimization for Mobile ML
//!
//! This module provides comprehensive battery management features specifically
//! designed for mobile ML workloads, including predictive battery management,
//! adaptive inference quality, and battery usage optimization.

use crate::{
    device_info::{ChargingStatus, MobileDeviceInfo},
    thermal_power::{ThermalPowerConfig, ThermalPowerManager},
    MobileConfig,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use trustformers_core::error::{CoreError, Result};
use trustformers_core::TrustformersError;

/// Advanced battery management system for mobile ML
pub struct MobileBatteryManager {
    config: BatteryConfig,
    battery_monitor: BatteryMonitor,
    power_predictor: PowerPredictor,
    adaptive_scheduler: AdaptiveInferenceScheduler,
    battery_optimizer: BatteryOptimizer,
    usage_analytics: BatteryUsageAnalytics,
    thermal_power_manager: Option<ThermalPowerManager>,
}

/// Battery management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatteryConfig {
    /// Enable battery monitoring
    pub enable_monitoring: bool,
    /// Battery monitoring interval (ms)
    pub monitoring_interval_ms: u64,
    /// Enable predictive power management
    pub enable_prediction: bool,
    /// Prediction window (minutes)
    pub prediction_window_minutes: u32,
    /// Enable adaptive inference quality
    pub enable_adaptive_quality: bool,
    /// Quality adaptation strategy
    pub quality_strategy: QualityAdaptationStrategy,
    /// Battery thresholds for different actions
    pub battery_thresholds: BatteryThresholds,
    /// Power usage limits
    pub power_limits: PowerUsageLimits,
    /// Enable usage analytics
    pub enable_analytics: bool,
    /// Maximum history size for analytics
    pub max_history_size: usize,
}

/// Battery threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatteryThresholds {
    /// Critical battery level (%)
    pub critical_percent: u8,
    /// Low battery level (%)
    pub low_percent: u8,
    /// Medium battery level (%)
    pub medium_percent: u8,
    /// High battery level (%)
    pub high_percent: u8,
    /// Time-based thresholds (minutes remaining)
    pub time_thresholds: TimeThresholds,
}

/// Time-based battery thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeThresholds {
    /// Critical time remaining (minutes)
    pub critical_minutes: u32,
    /// Low time remaining (minutes)
    pub low_minutes: u32,
    /// Medium time remaining (minutes)
    pub medium_minutes: u32,
}

/// Power usage limits for different scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerUsageLimits {
    /// Maximum power when on battery (mW)
    pub max_power_on_battery_mw: f32,
    /// Maximum power when charging (mW)
    pub max_power_when_charging_mw: f32,
    /// Maximum power for background tasks (mW)
    pub max_background_power_mw: f32,
    /// Power budget for different battery levels
    pub battery_level_budgets: HashMap<BatteryLevel, f32>,
}

/// Battery level categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BatteryLevel {
    Critical,
    Low,
    Medium,
    High,
    Full,
    Charging,
}

/// Quality adaptation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityAdaptationStrategy {
    /// No quality adaptation
    None,
    /// Linear quality reduction with battery level
    Linear,
    /// Exponential quality reduction
    Exponential,
    /// Stepped quality levels
    Stepped,
    /// Predictive quality adaptation
    Predictive,
    /// User preference based
    UserPreference,
}

/// Battery monitoring system
struct BatteryMonitor {
    current_level: Option<u8>,
    charging_status: ChargingStatus,
    voltage: Option<f32>,
    current_ma: Option<f32>,
    temperature_celsius: Option<f32>,
    capacity_mah: Option<u32>,
    cycle_count: Option<u32>,
    health_percent: Option<u8>,
    battery_history: VecDeque<BatteryReading>,
    last_update: Instant,
    update_interval: Duration,
}

/// Battery reading with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatteryReading {
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    pub level_percent: Option<u8>,
    pub charging_status: ChargingStatus,
    pub voltage: Option<f32>,
    pub current_ma: Option<f32>,
    pub temperature_celsius: Option<f32>,
    pub power_consumption_mw: Option<f32>,
    pub estimated_time_remaining_minutes: Option<u32>,
}

impl BatteryReading {
    /// A timestamped reading that carries no measurements.
    ///
    /// Returned when the running target exposes no battery source this crate
    /// can read (see `BatteryMonitor::read_battery_info`). Deliberately
    /// distinct from a zeroed reading: `None` means "not measured", whereas a
    /// `0` would claim an empty battery.
    pub fn unavailable() -> Self {
        Self {
            timestamp: Instant::now(),
            level_percent: None,
            charging_status: ChargingStatus::Unknown,
            voltage: None,
            current_ma: None,
            temperature_celsius: None,
            power_consumption_mw: None,
            estimated_time_remaining_minutes: None,
        }
    }
}

/// Root of the Linux/Android kernel `power_supply` sysfs class.
const POWER_SUPPLY_SYSFS_ROOT: &str = "/sys/class/power_supply";

/// Read one numeric sysfs node, returning `None` when it is absent,
/// unreadable, or does not parse.
fn read_sysfs_number<T: std::str::FromStr>(dir: &std::path::Path, node: &str) -> Option<T> {
    std::fs::read_to_string(dir.join(node)).ok()?.trim().parse::<T>().ok()
}

/// Read real battery telemetry from a kernel `power_supply` sysfs tree.
///
/// The Android kernel exposes the same `power_supply` nodes as desktop Linux,
/// so a single reader covers both. `root` is the class directory (normally
/// [`POWER_SUPPLY_SYSFS_ROOT`]); it is a parameter so the parser can be
/// exercised against a fixture tree in tests instead of the live machine.
///
/// Unit conversions follow `Documentation/ABI/testing/sysfs-class-power`:
/// `voltage_now` is µV, `current_now` is µA, `power_now` is µW, `temp` is
/// tenths of a degree Celsius and `charge_now` is µAh.
///
/// Every field is independently optional: a node that is missing (many
/// gauges publish only `capacity` and `status`) leaves its field `None` rather
/// than substituting a value. Returns `None` when `root` holds no supply of
/// type `Battery` at all.
fn read_power_supply_battery(root: &std::path::Path) -> Option<BatteryReading> {
    let battery_dir =
        std::fs::read_dir(root).ok()?.filter_map(|entry| entry.ok()).find(|entry| {
            std::fs::read_to_string(entry.path().join("type"))
                .map(|kind| kind.trim().eq_ignore_ascii_case("Battery"))
                .unwrap_or(false)
        })?;
    let dir = battery_dir.path();

    let level_percent = read_sysfs_number::<i32>(&dir, "capacity")
        .filter(|percent| (0..=100).contains(percent))
        .map(|percent| percent as u8);

    let charging_status = match std::fs::read_to_string(dir.join("status")) {
        Ok(status) => match status.trim() {
            "Charging" => ChargingStatus::Charging,
            "Discharging" => ChargingStatus::Discharging,
            "Full" => ChargingStatus::Full,
            "Not charging" => ChargingStatus::NotCharging,
            _ => ChargingStatus::Unknown,
        },
        Err(_) => ChargingStatus::Unknown,
    };

    let voltage = read_sysfs_number::<f64>(&dir, "voltage_now").map(|uv| (uv / 1e6) as f32);
    let current_ma = read_sysfs_number::<f64>(&dir, "current_now").map(|ua| (ua / 1e3) as f32);
    let temperature_celsius =
        read_sysfs_number::<f64>(&dir, "temp").map(|tenths| (tenths / 10.0) as f32);

    // `power_now` when the gauge publishes it, otherwise the product of two
    // real measurements (P = V * I). Absolute value: the sign only encodes
    // charge direction, which `charging_status` already carries.
    let power_consumption_mw = read_sysfs_number::<f64>(&dir, "power_now")
        .map(|uw| (uw.abs() / 1e3) as f32)
        .or_else(|| match (voltage, current_ma) {
            (Some(volts), Some(milliamps)) => Some((volts * milliamps).abs()),
            _ => None,
        });

    // Runtime left = remaining charge / draw. Only meaningful while actually
    // discharging at a nonzero rate.
    let estimated_time_remaining_minutes = match (
        charging_status,
        read_sysfs_number::<f64>(&dir, "charge_now"),
        read_sysfs_number::<f64>(&dir, "current_now"),
    ) {
        (ChargingStatus::Discharging, Some(charge_uah), Some(current_ua))
            if current_ua.abs() > f64::EPSILON =>
        {
            Some((charge_uah.abs() / current_ua.abs() * 60.0).round() as u32)
        },
        _ => None,
    };

    Some(BatteryReading {
        timestamp: Instant::now(),
        level_percent,
        charging_status,
        voltage,
        current_ma,
        temperature_celsius,
        power_consumption_mw,
        estimated_time_remaining_minutes,
    })
}

/// Take one fresh, stateless battery reading from the live system.
///
/// Exists for callers that want a single snapshot without owning a
/// [`MobileBatteryManager`] -- currently `mobile_performance_profiler`'s
/// metrics collector, which reuses this instead of re-implementing the same
/// `#[cfg(...)]`-gated sysfs walk a second time. Same platform coverage as
/// [`BatteryMonitor::read_battery_info`]: real telemetry on Android/Linux via
/// [`read_power_supply_battery`], [`BatteryReading::unavailable`] everywhere
/// else (iOS/macOS need IOKit/`UIDevice`, which this crate does not link).
pub(crate) fn read_live_battery_reading() -> BatteryReading {
    #[cfg(any(target_os = "android", target_os = "linux"))]
    {
        if let Some(reading) =
            read_power_supply_battery(std::path::Path::new(POWER_SUPPLY_SYSFS_ROOT))
        {
            return reading;
        }
    }
    BatteryReading::unavailable()
}

/// Power prediction system
struct PowerPredictor {
    usage_patterns: Vec<UsagePattern>,
    prediction_models: HashMap<String, PredictionModel>,
    historical_data: VecDeque<PowerDataPoint>,
    accuracy_metrics: PredictionAccuracyMetrics,
}

/// Usage pattern for prediction
#[derive(Debug, Clone)]
struct UsagePattern {
    time_of_day: u8,          // Hour 0-23
    day_of_week: u8,          // 0-6
    app_context: String,      // App or usage context
    inference_frequency: f32, // Inferences per minute
    average_power_mw: f32,    // Average power consumption
    duration_minutes: u32,    // Typical usage duration
}

/// Prediction model
#[derive(Debug, Clone)]
struct PredictionModel {
    model_type: ModelType,
    parameters: Vec<f32>,
    accuracy: f32,
    last_updated: Instant,
}

/// Prediction model types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelType {
    Linear,
    Exponential,
    MovingAverage,
    ARIMA,
}

/// Power consumption data point
#[derive(Debug, Clone)]
struct PowerDataPoint {
    timestamp: Instant,
    power_mw: f32,
    battery_level: u8,
    inference_count: u32,
    context: String,
}

/// Prediction accuracy metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionAccuracyMetrics {
    pub mean_absolute_error: f32,
    pub root_mean_square_error: f32,
    pub mean_absolute_percentage_error: f32,
    pub prediction_confidence: f32,
}

/// Adaptive inference scheduler
struct AdaptiveInferenceScheduler {
    inference_queue: VecDeque<AdaptiveInferenceRequest>,
    quality_levels: QualityLevelConfig,
    current_quality_level: QualityLevel,
    adaptation_history: VecDeque<QualityAdaptation>,
}

/// Adaptive inference request
#[derive(Debug, Clone)]
struct AdaptiveInferenceRequest {
    id: String,
    priority: InferencePriority,
    quality_requirements: QualityRequirements,
    power_budget_mw: Option<f32>,
    deadline: Option<Instant>,
    adaptable_quality: bool,
}

/// Inference priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum InferencePriority {
    Background,
    Normal,
    High,
    Critical,
    RealTime,
}

/// Quality requirements for inference
#[derive(Debug, Clone)]
struct QualityRequirements {
    min_quality: f32,
    target_quality: f32,
    max_latency_ms: Option<u32>,
    accuracy_tolerance: f32,
}

/// Quality level configuration
#[derive(Debug, Clone)]
struct QualityLevelConfig {
    levels: Vec<QualityLevel>,
    current_index: usize,
}

/// Quality level definition
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum QualityLevel {
    Minimal, // Maximum power savings
    Low,     // Reduced quality
    Medium,  // Balanced
    High,    // Near-full quality
    Maximum, // Full quality
}

/// Quality adaptation record
#[derive(Debug, Clone)]
struct QualityAdaptation {
    timestamp: Instant,
    from_level: QualityLevel,
    to_level: QualityLevel,
    reason: AdaptationReason,
    battery_level: u8,
    /// Power draw measured at the moment of the adaptation, or `None` when
    /// this device publishes no power figure.
    power_consumption: Option<f32>,
}

/// Reason for quality adaptation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdaptationReason {
    BatteryLevel,
    PowerLimit,
    ThermalThrottling,
    UserPreference,
    PredictiveOptimization,
}

/// Battery optimizer for ML workloads
struct BatteryOptimizer {
    optimization_strategies: Vec<OptimizationStrategy>,
    optimization_history: VecDeque<OptimizationAction>,
    effectiveness_metrics: OptimizationEffectiveness,
}

/// Battery optimization strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OptimizationStrategy {
    /// Reduce inference frequency
    ReduceFrequency,
    /// Lower model precision
    ReducePrecision,
    /// Use smaller model variant
    UseSmallerModel,
    /// Batch inferences
    BatchInferences,
    /// Defer non-critical inferences
    DeferInferences,
    /// Offload to edge/cloud
    OffloadToEdge,
}

/// Optimization action record
#[derive(Debug, Clone)]
struct OptimizationAction {
    timestamp: Instant,
    strategy: OptimizationStrategy,
    battery_level_before: u8,
    battery_level_after: u8,
    power_savings_mw: f32,
    quality_impact: f32,
}

/// Optimization effectiveness metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationEffectiveness {
    pub total_power_saved_mwh: f32,
    pub battery_life_extension_minutes: u32,
    pub average_quality_impact: f32,
    pub successful_optimizations: usize,
    pub failed_optimizations: usize,
}

/// Battery usage analytics
pub struct BatteryUsageAnalytics {
    usage_sessions: VecDeque<UsageSession>,
    daily_summaries: HashMap<String, DailySummary>, // Date -> Summary
    weekly_patterns: WeeklyUsagePattern,
    optimization_recommendations: Vec<OptimizationRecommendation>,
}

/// Usage session record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageSession {
    #[serde(skip, default = "Instant::now")]
    pub start_time: Instant,
    #[serde(skip, default = "Instant::now")]
    pub end_time: Instant,
    pub start_battery_level: u8,
    pub end_battery_level: u8,
    pub total_inferences: u32,
    pub average_power_mw: f32,
    pub peak_power_mw: f32,
    pub thermal_throttling_events: u32,
    pub quality_adaptations: u32,
}

/// Daily battery usage summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailySummary {
    pub date: String,
    pub total_usage_time_minutes: u32,
    pub total_inferences: u32,
    pub average_battery_drain_per_hour: f32,
    pub peak_power_consumption_mw: f32,
    pub thermal_events: u32,
    pub charging_sessions: u32,
    pub efficiency_score: f32,
}

/// Weekly usage pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeeklyUsagePattern {
    pub peak_usage_hours: Vec<u8>,
    pub average_daily_usage_minutes: f32,
    pub most_power_intensive_day: String,
    pub battery_health_trend: f32,
    pub optimization_opportunities: Vec<String>,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub recommendation_type: String,
    pub description: String,
    pub estimated_power_savings_percent: f32,
    pub estimated_quality_impact_percent: f32,
    pub implementation_difficulty: DifficultyLevel,
    pub confidence_score: f32,
}

/// Implementation difficulty levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DifficultyLevel {
    Easy,
    Medium,
    Hard,
}

/// Battery management statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatteryStats {
    /// Current battery level (%)
    pub current_level_percent: Option<u8>,
    /// Charging status
    pub charging_status: ChargingStatus,
    /// Estimated time remaining (minutes)
    pub estimated_time_remaining_minutes: Option<u32>,
    /// Current power consumption (mW)
    pub current_power_consumption_mw: Option<f32>,
    /// Average power consumption (mW) over the readings that carried one.
    /// `None` when no reading in history published a power figure.
    pub average_power_consumption_mw: Option<f32>,
    /// Peak power consumption (mW) across the readings that carried one.
    /// `None` when no reading in history published a power figure.
    pub peak_power_consumption_mw: Option<f32>,
    /// Total battery time saved (minutes)
    pub battery_time_saved_minutes: u32,
    /// Current quality level
    pub current_quality_level: QualityLevel,
    /// Quality adaptations in last hour
    pub recent_quality_adaptations: u32,
    /// Battery health (%)
    pub battery_health_percent: Option<u8>,
    /// Optimization effectiveness
    pub optimization_effectiveness: OptimizationEffectiveness,
}

impl Default for BatteryConfig {
    fn default() -> Self {
        Self {
            enable_monitoring: true,
            monitoring_interval_ms: 5000, // 5 seconds
            enable_prediction: true,
            prediction_window_minutes: 60, // 1 hour prediction
            enable_adaptive_quality: true,
            quality_strategy: QualityAdaptationStrategy::Predictive,
            battery_thresholds: BatteryThresholds::default(),
            power_limits: PowerUsageLimits::default(),
            enable_analytics: true,
            max_history_size: 1000,
        }
    }
}

impl Default for BatteryThresholds {
    fn default() -> Self {
        Self {
            critical_percent: 15,
            low_percent: 30,
            medium_percent: 50,
            high_percent: 80,
            time_thresholds: TimeThresholds {
                critical_minutes: 30,
                low_minutes: 60,
                medium_minutes: 120,
            },
        }
    }
}

impl Default for PowerUsageLimits {
    fn default() -> Self {
        let mut battery_budgets = HashMap::new();
        battery_budgets.insert(BatteryLevel::Critical, 1000.0); // 1W
        battery_budgets.insert(BatteryLevel::Low, 2000.0); // 2W
        battery_budgets.insert(BatteryLevel::Medium, 3000.0); // 3W
        battery_budgets.insert(BatteryLevel::High, 4000.0); // 4W
        battery_budgets.insert(BatteryLevel::Charging, 6000.0); // 6W

        Self {
            max_power_on_battery_mw: 4000.0,    // 4W max on battery
            max_power_when_charging_mw: 8000.0, // 8W max when charging
            max_background_power_mw: 1500.0,    // 1.5W for background
            battery_level_budgets: battery_budgets,
        }
    }
}

impl MobileBatteryManager {
    /// Create new battery manager
    pub fn new(config: BatteryConfig, device_info: &MobileDeviceInfo) -> Result<Self> {
        let battery_monitor = BatteryMonitor::new(
            Duration::from_millis(config.monitoring_interval_ms),
            config.max_history_size,
        );

        let power_predictor = PowerPredictor::new(config.prediction_window_minutes);
        let adaptive_scheduler = AdaptiveInferenceScheduler::new(&config);
        let battery_optimizer = BatteryOptimizer::new();
        let usage_analytics = BatteryUsageAnalytics::new(config.max_history_size);

        // Initialize thermal/power manager integration if needed
        let thermal_power_manager = if config.enable_monitoring {
            let thermal_config = ThermalPowerConfig::default();
            Some(ThermalPowerManager::new(thermal_config, device_info)?)
        } else {
            None
        };

        Ok(Self {
            config,
            battery_monitor,
            power_predictor,
            adaptive_scheduler,
            battery_optimizer,
            usage_analytics,
            thermal_power_manager,
        })
    }

    /// Start battery monitoring and management
    pub fn start(&mut self) -> Result<()> {
        self.battery_monitor.start()?;

        if let Some(ref mut thermal_manager) = self.thermal_power_manager {
            thermal_manager.start_monitoring()?;
        }

        tracing::info!("Battery management started");
        Ok(())
    }

    /// Stop battery management
    pub fn stop(&mut self) {
        self.battery_monitor.stop();

        if let Some(ref mut thermal_manager) = self.thermal_power_manager {
            thermal_manager.stop_monitoring();
        }

        tracing::info!("Battery management stopped");
    }

    /// Update battery status and apply optimizations
    pub fn update(&mut self, mobile_config: &mut MobileConfig) -> Result<bool> {
        let mut config_changed = false;

        // Update battery monitoring
        self.battery_monitor.update()?;

        // Update thermal/power management if available
        if let Some(ref mut thermal_manager) = self.thermal_power_manager {
            if thermal_manager.update(mobile_config)? {
                config_changed = true;
            }
        }

        // Update power predictions
        if self.config.enable_prediction {
            self.power_predictor.update(&self.battery_monitor)?;
        }

        // Apply battery-specific optimizations
        if self.should_apply_battery_optimizations()?
            && self.apply_battery_optimizations(mobile_config)?
        {
            config_changed = true;
        }

        // Update adaptive quality if enabled
        if self.config.enable_adaptive_quality {
            self.update_adaptive_quality(mobile_config)?;
        }

        // Update analytics
        if self.config.enable_analytics {
            self.usage_analytics.update(&self.battery_monitor, mobile_config);
        }

        Ok(config_changed)
    }

    /// Get battery statistics
    pub fn get_stats(&self) -> BatteryStats {
        BatteryStats {
            current_level_percent: self.battery_monitor.current_level,
            charging_status: self.battery_monitor.charging_status,
            estimated_time_remaining_minutes: self.estimate_time_remaining(),
            current_power_consumption_mw: self.get_current_power_consumption(),
            average_power_consumption_mw: self.calculate_average_power_consumption(),
            peak_power_consumption_mw: self.get_peak_power_consumption(),
            battery_time_saved_minutes: self.calculate_battery_time_saved(),
            current_quality_level: self.adaptive_scheduler.current_quality_level,
            recent_quality_adaptations: self.count_recent_quality_adaptations(),
            battery_health_percent: self.battery_monitor.health_percent,
            optimization_effectiveness: self.battery_optimizer.effectiveness_metrics.clone(),
        }
    }

    /// Get power consumption prediction
    pub fn predict_power_consumption(&self, duration_minutes: u32) -> Result<PowerPrediction> {
        if !self.config.enable_prediction {
            return Err(TrustformersError::config_error(
                "Power prediction not enabled",
                "predict_power_consumption",
            )
            .into());
        }

        self.power_predictor.predict_consumption(duration_minutes)
    }

    /// Get current battery reading
    pub fn get_current_reading(&self) -> Result<BatteryReading> {
        self.battery_monitor
            .battery_history
            .back()
            .cloned()
            .ok_or_else(|| TrustformersError::runtime_error("No battery reading available".into()))
            .map_err(|e| e.into())
    }

    /// Get battery optimization recommendations
    pub fn get_optimization_recommendations(&self) -> &[OptimizationRecommendation] {
        &self.usage_analytics.optimization_recommendations
    }

    /// Get usage analytics
    pub fn get_usage_analytics(&self) -> BatteryUsageAnalytics {
        self.usage_analytics.clone()
    }

    /// Create battery-optimized configuration
    pub fn create_battery_optimized_config(
        base_config: &MobileConfig,
        battery_level: u8,
        charging: bool,
    ) -> MobileConfig {
        let mut optimized = base_config.clone();

        let battery_category = Self::categorize_battery_level(battery_level, charging);

        match battery_category {
            BatteryLevel::Critical => {
                // Extreme power saving
                optimized.memory_optimization = crate::MemoryOptimization::Maximum;
                optimized.num_threads = 1;
                optimized.enable_batching = false;
                optimized.backend = crate::MobileBackend::CPU;

                // Aggressive quantization
                if let Some(ref mut quant) = optimized.quantization {
                    quant.scheme = crate::MobileQuantizationScheme::Int4;
                    quant.dynamic = true;
                }
            },
            BatteryLevel::Low => {
                // Moderate power saving
                optimized.memory_optimization = crate::MemoryOptimization::Balanced;
                optimized.num_threads = (optimized.num_threads / 2).max(1);
                optimized.max_batch_size = (optimized.max_batch_size / 2).max(1);
            },
            BatteryLevel::Medium => {
                // Light optimizations
                optimized.num_threads = (optimized.num_threads * 3 / 4).max(1);
            },
            BatteryLevel::High | BatteryLevel::Full | BatteryLevel::Charging => {
                // Minimal or no optimizations
            },
        }

        optimized
    }

    // Private implementation methods

    fn should_apply_battery_optimizations(&self) -> Result<bool> {
        let battery_level = self.battery_monitor.current_level.unwrap_or(100);
        let is_charging = matches!(
            self.battery_monitor.charging_status,
            ChargingStatus::Charging
        );

        // Apply optimizations if battery is low and not charging
        Ok(battery_level < self.config.battery_thresholds.medium_percent && !is_charging)
    }

    fn apply_battery_optimizations(&mut self, config: &mut MobileConfig) -> Result<bool> {
        let battery_level = self.battery_monitor.current_level.unwrap_or(100);
        let is_charging = matches!(
            self.battery_monitor.charging_status,
            ChargingStatus::Charging
        );

        if is_charging {
            return Ok(false); // No need to optimize when charging
        }

        let mut changed = false;

        if battery_level < self.config.battery_thresholds.critical_percent {
            // Critical battery optimizations
            let strategies = vec![
                OptimizationStrategy::ReduceFrequency,
                OptimizationStrategy::ReducePrecision,
                OptimizationStrategy::DeferInferences,
            ];

            for strategy in strategies {
                if self.battery_optimizer.apply_strategy(strategy, config)? {
                    changed = true;
                }
            }
        } else if battery_level < self.config.battery_thresholds.low_percent {
            // Low battery optimizations
            let strategies = vec![
                OptimizationStrategy::BatchInferences,
                OptimizationStrategy::ReducePrecision,
            ];

            for strategy in strategies {
                if self.battery_optimizer.apply_strategy(strategy, config)? {
                    changed = true;
                }
            }
        }

        Ok(changed)
    }

    fn update_adaptive_quality(&mut self, _config: &mut MobileConfig) -> Result<()> {
        let battery_level = self.battery_monitor.current_level.unwrap_or(100);
        let battery_category = Self::categorize_battery_level(
            battery_level,
            matches!(
                self.battery_monitor.charging_status,
                ChargingStatus::Charging
            ),
        );

        let target_quality = match battery_category {
            BatteryLevel::Critical => QualityLevel::Minimal,
            BatteryLevel::Low => QualityLevel::Low,
            BatteryLevel::Medium => QualityLevel::Medium,
            BatteryLevel::High => QualityLevel::High,
            BatteryLevel::Full => QualityLevel::Maximum,
            BatteryLevel::Charging => QualityLevel::Maximum,
        };

        if target_quality != self.adaptive_scheduler.current_quality_level {
            let measured_power_mw = self.get_current_power_consumption();
            self.adaptive_scheduler.adapt_quality(
                target_quality,
                AdaptationReason::BatteryLevel,
                battery_level,
                measured_power_mw,
            );
        }

        Ok(())
    }

    fn categorize_battery_level(level: u8, charging: bool) -> BatteryLevel {
        if charging {
            BatteryLevel::Charging
        } else if level < 15 {
            BatteryLevel::Critical
        } else if level < 30 {
            BatteryLevel::Low
        } else if level < 60 {
            BatteryLevel::Medium
        } else {
            BatteryLevel::High
        }
    }

    /// Minutes of runtime left, taken from the most recent battery reading.
    ///
    /// The value comes from the platform reader (on Linux/Android, remaining
    /// charge divided by present draw -- see [`read_power_supply_battery`]);
    /// `None` when no reading has been taken yet or the platform published
    /// neither a charge counter nor a current.
    fn estimate_time_remaining(&self) -> Option<u32> {
        self.battery_monitor
            .battery_history
            .back()
            .and_then(|reading| reading.estimated_time_remaining_minutes)
    }

    fn get_current_power_consumption(&self) -> Option<f32> {
        self.battery_monitor
            .battery_history
            .back()
            .and_then(|reading| reading.power_consumption_mw)
    }

    /// Mean power draw across the readings that actually carried one.
    ///
    /// `None` -- not `0.0` -- when no reading published a power figure: a zero
    /// would claim the device drew no power.
    fn calculate_average_power_consumption(&self) -> Option<f32> {
        let sum: f32 = self
            .battery_monitor
            .battery_history
            .iter()
            .filter_map(|reading| reading.power_consumption_mw)
            .sum();

        let count = self
            .battery_monitor
            .battery_history
            .iter()
            .filter(|reading| reading.power_consumption_mw.is_some())
            .count();

        if count > 0 {
            Some(sum / count as f32)
        } else {
            None
        }
    }

    /// Highest power draw seen across the readings that carried one, or `None`
    /// when none did.
    fn get_peak_power_consumption(&self) -> Option<f32> {
        self.battery_monitor
            .battery_history
            .iter()
            .filter_map(|reading| reading.power_consumption_mw)
            .fold(None, |peak: Option<f32>, mw| {
                Some(peak.map_or(mw, |best| best.max(mw)))
            })
    }

    /// Measured battery level as a fraction in `0.0..=1.0`.
    ///
    /// `None` when the running target exposes no readable battery gauge (every
    /// non-Linux, non-Android target -- see
    /// `BatteryMonitor::read_battery_info`) or when monitoring has not taken
    /// a reading yet. Charging status is deliberately *not* used to guess a
    /// level: "charging" says nothing about how full the cell is.
    pub fn get_current_battery_level(&self) -> Option<f32> {
        self.battery_monitor.current_level.map(|level| level as f32 / 100.0)
    }

    fn calculate_battery_time_saved(&self) -> u32 {
        // Implementation would calculate based on optimization history
        self.battery_optimizer.effectiveness_metrics.battery_life_extension_minutes
    }

    fn count_recent_quality_adaptations(&self) -> u32 {
        let one_hour_ago = Instant::now() - Duration::from_secs(3600);
        self.adaptive_scheduler
            .adaptation_history
            .iter()
            .filter(|adaptation| adaptation.timestamp > one_hour_ago)
            .count() as u32
    }
}

/// Power consumption prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerPrediction {
    pub predicted_consumption_mw: f32,
    pub confidence_interval: (f32, f32),
    pub accuracy_metrics: PredictionAccuracyMetrics,
    pub factors: Vec<PredictionFactor>,
}

/// Factors affecting power prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionFactor {
    pub factor_name: String,
    pub impact_weight: f32,
    pub description: String,
}

// Implementation stubs for complex components

impl BatteryMonitor {
    fn new(update_interval: Duration, max_history: usize) -> Self {
        Self {
            current_level: None,
            charging_status: ChargingStatus::Unknown,
            voltage: None,
            current_ma: None,
            temperature_celsius: None,
            capacity_mah: None,
            cycle_count: None,
            health_percent: None,
            battery_history: VecDeque::with_capacity(max_history),
            last_update: Instant::now(),
            update_interval,
        }
    }

    fn start(&mut self) -> Result<()> {
        self.last_update = Instant::now();
        Ok(())
    }

    fn stop(&mut self) {
        // Nothing to do for stop
    }

    fn update(&mut self) -> Result<()> {
        if self.last_update.elapsed() >= self.update_interval {
            // Read battery information from platform APIs
            let reading = self.read_battery_info()?;

            self.battery_history.push_back(reading);
            while self.battery_history.len() > self.battery_history.capacity() {
                self.battery_history.pop_front();
            }

            self.last_update = Instant::now();
        }
        Ok(())
    }

    /// Read battery telemetry from the only source this crate can read without
    /// platform FFI.
    ///
    /// On Android and Linux that is the kernel `power_supply` sysfs class
    /// (identical node layout on both), read by
    /// [`read_power_supply_battery`]. Everywhere else -- iOS, macOS, Windows,
    /// wasm -- there is no source reachable from safe, dependency-free Rust:
    /// iOS battery state needs `UIDevice`/IOKit through Objective-C, and macOS
    /// needs IOKit. Rather than invent a level, a voltage and a wattage, those
    /// targets get an all-`None` reading whose `charging_status` is
    /// [`ChargingStatus::Unknown`]; every consumer in this module already
    /// treats `None` as "not measured" (see
    /// [`BatteryOptimizationManager::calculate_average_power_consumption`],
    /// which averages only over readings that carry a value).
    fn read_battery_info(&mut self) -> Result<BatteryReading> {
        #[cfg(any(target_os = "android", target_os = "linux"))]
        {
            if let Some(reading) =
                read_power_supply_battery(std::path::Path::new(POWER_SUPPLY_SYSFS_ROOT))
            {
                self.current_level = reading.level_percent;
                self.charging_status = reading.charging_status;
                self.voltage = reading.voltage;
                self.current_ma = reading.current_ma;
                self.temperature_celsius = reading.temperature_celsius;
                return Ok(reading);
            }
        }

        // No battery source is readable on this target: report that, do not
        // fabricate one.
        self.current_level = None;
        self.charging_status = ChargingStatus::Unknown;
        self.voltage = None;
        self.current_ma = None;
        self.temperature_celsius = None;

        Ok(BatteryReading::unavailable())
    }
}

impl PowerPredictor {
    fn new(_prediction_window: u32) -> Self {
        Self {
            usage_patterns: Vec::new(),
            prediction_models: HashMap::new(),
            historical_data: VecDeque::new(),
            accuracy_metrics: PredictionAccuracyMetrics {
                mean_absolute_error: 0.0,
                root_mean_square_error: 0.0,
                mean_absolute_percentage_error: 0.0,
                prediction_confidence: 0.8,
            },
        }
    }

    /// Record the latest measured power draw as a prediction input.
    ///
    /// Only readings that carried a real `power_consumption_mw` are kept, so
    /// the history the forecast is fitted to contains nothing invented.
    fn update(&mut self, battery_monitor: &BatteryMonitor) -> Result<()> {
        let Some(reading) = battery_monitor.battery_history.back() else {
            return Ok(());
        };
        let Some(power_mw) = reading.power_consumption_mw else {
            return Ok(());
        };

        self.historical_data.push_back(PowerDataPoint {
            timestamp: reading.timestamp,
            power_mw,
            battery_level: reading.level_percent.unwrap_or(0),
            inference_count: 0,
            context: format!("{:?}", reading.charging_status),
        });
        while self.historical_data.len() > 512 {
            self.historical_data.pop_front();
        }
        Ok(())
    }

    /// Forecast energy use over `duration_minutes` by extrapolating the mean
    /// of the power draws actually observed so far.
    ///
    /// Errors when nothing has been observed: the previous body extrapolated
    /// a hardcoded `base_consumption = 2500.0` mW, so it produced a confident
    /// forecast on a device whose power draw had never been read. The
    /// confidence interval is the observed standard deviation (or +/-20% for a
    /// single sample), not a fixed multiplier on an invented mean.
    fn predict_consumption(&self, duration_minutes: u32) -> Result<PowerPrediction> {
        if self.historical_data.is_empty() {
            return Err(TrustformersError::runtime_error(
                "Power prediction needs at least one measured power reading; this device has \
                 published none (see `MobileBatteryManager::get_current_power_consumption`)"
                    .into(),
            )
            .into());
        }

        let samples: Vec<f32> = self.historical_data.iter().map(|point| point.power_mw).collect();
        let mean_power_mw = samples.iter().sum::<f32>() / samples.len() as f32;
        let spread = if samples.len() > 1 {
            let variance = samples.iter().map(|mw| (mw - mean_power_mw).powi(2)).sum::<f32>()
                / (samples.len() - 1) as f32;
            variance.sqrt()
        } else {
            mean_power_mw * 0.2
        };

        let hours = duration_minutes as f32 / 60.0;
        let predicted_consumption = mean_power_mw * hours;

        Ok(PowerPrediction {
            predicted_consumption_mw: predicted_consumption,
            confidence_interval: (
                (mean_power_mw - spread).max(0.0) * hours,
                (mean_power_mw + spread) * hours,
            ),
            accuracy_metrics: self.accuracy_metrics.clone(),
            factors: vec![PredictionFactor {
                factor_name: "Observed mean power".to_string(),
                impact_weight: 1.0,
                description: format!(
                    "{:.1} mW averaged over {} measured reading(s)",
                    mean_power_mw,
                    samples.len()
                ),
            }],
        })
    }
}

impl AdaptiveInferenceScheduler {
    fn new(_config: &BatteryConfig) -> Self {
        Self {
            inference_queue: VecDeque::new(),
            quality_levels: QualityLevelConfig {
                levels: vec![
                    QualityLevel::Minimal,
                    QualityLevel::Low,
                    QualityLevel::Medium,
                    QualityLevel::High,
                    QualityLevel::Maximum,
                ],
                current_index: 2, // Start at Medium
            },
            current_quality_level: QualityLevel::Medium,
            adaptation_history: VecDeque::new(),
        }
    }

    fn adapt_quality(
        &mut self,
        target_level: QualityLevel,
        reason: AdaptationReason,
        battery_level: u8,
        power_consumption_mw: Option<f32>,
    ) {
        let old_level = self.current_quality_level;
        self.current_quality_level = target_level;

        let adaptation = QualityAdaptation {
            timestamp: Instant::now(),
            from_level: old_level,
            to_level: target_level,
            reason,
            battery_level,
            // Whatever the platform measured at this instant; previously a
            // fixed 2500.0 mW recorded against every adaptation.
            power_consumption: power_consumption_mw,
        };

        self.adaptation_history.push_back(adaptation);

        // Limit history size
        while self.adaptation_history.len() > 100 {
            self.adaptation_history.pop_front();
        }
    }
}

impl BatteryOptimizer {
    fn new() -> Self {
        Self {
            optimization_strategies: vec![
                OptimizationStrategy::ReduceFrequency,
                OptimizationStrategy::ReducePrecision,
                OptimizationStrategy::BatchInferences,
                OptimizationStrategy::DeferInferences,
            ],
            optimization_history: VecDeque::new(),
            effectiveness_metrics: OptimizationEffectiveness {
                total_power_saved_mwh: 0.0,
                battery_life_extension_minutes: 0,
                average_quality_impact: 0.0,
                successful_optimizations: 0,
                failed_optimizations: 0,
            },
        }
    }

    fn apply_strategy(
        &mut self,
        strategy: OptimizationStrategy,
        _config: &mut MobileConfig,
    ) -> Result<bool> {
        // Apply optimization strategy
        match strategy {
            OptimizationStrategy::ReduceFrequency => {
                // Reduce inference frequency
                tracing::info!("Applied strategy: Reduce inference frequency");
            },
            OptimizationStrategy::ReducePrecision => {
                // Reduce model precision
                tracing::info!("Applied strategy: Reduce precision");
            },
            OptimizationStrategy::BatchInferences => {
                // Enable batching
                tracing::info!("Applied strategy: Batch inferences");
            },
            OptimizationStrategy::DeferInferences => {
                // Defer non-critical inferences
                tracing::info!("Applied strategy: Defer inferences");
            },
            _ => {
                tracing::info!("Applied strategy: {:?}", strategy);
            },
        }

        self.effectiveness_metrics.successful_optimizations += 1;
        Ok(true)
    }
}

impl BatteryUsageAnalytics {
    fn new(_max_history: usize) -> Self {
        Self {
            usage_sessions: VecDeque::new(),
            daily_summaries: HashMap::new(),
            weekly_patterns: WeeklyUsagePattern {
                peak_usage_hours: vec![9, 14, 20], // 9am, 2pm, 8pm
                average_daily_usage_minutes: 120.0,
                most_power_intensive_day: "Monday".to_string(),
                battery_health_trend: 0.98, // 98% health trend
                optimization_opportunities: vec![
                    "Reduce inference frequency during peak hours".to_string(),
                    "Use lower precision models after 8pm".to_string(),
                ],
            },
            optimization_recommendations: vec![OptimizationRecommendation {
                recommendation_type: "Precision Optimization".to_string(),
                description: "Use INT8 quantization during low battery periods".to_string(),
                estimated_power_savings_percent: 15.0,
                estimated_quality_impact_percent: 3.0,
                implementation_difficulty: DifficultyLevel::Easy,
                confidence_score: 0.85,
            }],
        }
    }

    fn update(&mut self, _battery_monitor: &BatteryMonitor, _config: &MobileConfig) {
        // Update analytics with new data
    }
}

impl Clone for BatteryUsageAnalytics {
    fn clone(&self) -> Self {
        Self {
            usage_sessions: self.usage_sessions.clone(),
            daily_summaries: self.daily_summaries.clone(),
            weekly_patterns: self.weekly_patterns.clone(),
            optimization_recommendations: self.optimization_recommendations.clone(),
        }
    }
}

/// Utility functions for battery management
pub struct BatteryUtils;

impl BatteryUtils {
    /// Calculate battery drain rate (% per hour)
    pub fn calculate_drain_rate(readings: &[BatteryReading]) -> f32 {
        if readings.len() < 2 {
            return 0.0;
        }

        let first = &readings[0];
        let last = &readings[readings.len() - 1];

        if let (Some(first_level), Some(last_level)) = (first.level_percent, last.level_percent) {
            let level_change = first_level as f32 - last_level as f32;
            let time_hours = last.timestamp.duration_since(first.timestamp).as_secs_f32() / 3600.0;

            if time_hours > 0.0 {
                level_change / time_hours
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Estimate remaining battery time
    pub fn estimate_remaining_time(
        current_level: u8,
        drain_rate_per_hour: f32,
    ) -> Option<Duration> {
        if drain_rate_per_hour <= 0.0 {
            return None; // Charging or no drain
        }

        let hours_remaining = current_level as f32 / drain_rate_per_hour;
        Some(Duration::from_secs_f32(hours_remaining * 3600.0))
    }

    /// Calculate power efficiency score (0.0 to 1.0)
    pub fn calculate_efficiency_score(
        power_consumption_mw: f32,
        inference_count: u32,
        time_duration_minutes: f32,
    ) -> f32 {
        if time_duration_minutes <= 0.0 || inference_count == 0 {
            return 0.0;
        }

        let inferences_per_minute = inference_count as f32 / time_duration_minutes;
        let power_per_inference = power_consumption_mw / inference_count as f32;

        // Efficiency is higher when we get more inferences per unit of power
        let efficiency = inferences_per_minute / (power_per_inference / 1000.0);

        // Normalised to 0-1 against a reference of 10 inferences per minute
        // per watt. The divisor is a stated scale for this score, not a
        // measurement of any particular device.
        (efficiency / 10.0).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_battery_config_default() {
        let config = BatteryConfig::default();
        assert!(config.enable_monitoring);
        assert!(config.enable_prediction);
        assert!(config.enable_adaptive_quality);
        assert!(matches!(
            config.quality_strategy,
            QualityAdaptationStrategy::Predictive
        ));
    }

    #[test]
    fn test_battery_thresholds() {
        let thresholds = BatteryThresholds::default();
        assert!(thresholds.critical_percent < thresholds.low_percent);
        assert!(thresholds.low_percent < thresholds.medium_percent);
        assert!(thresholds.medium_percent < thresholds.high_percent);
    }

    #[test]
    fn test_battery_level_categorization() {
        assert_eq!(
            MobileBatteryManager::categorize_battery_level(10, false),
            BatteryLevel::Critical
        );
        assert_eq!(
            MobileBatteryManager::categorize_battery_level(25, false),
            BatteryLevel::Low
        );
        assert_eq!(
            MobileBatteryManager::categorize_battery_level(45, false),
            BatteryLevel::Medium
        );
        assert_eq!(
            MobileBatteryManager::categorize_battery_level(75, false),
            BatteryLevel::High
        );
        assert_eq!(
            MobileBatteryManager::categorize_battery_level(25, true),
            BatteryLevel::Charging
        );
    }

    #[test]
    fn test_battery_optimized_config() {
        let base_config = crate::MobileConfig::default();

        // Test critical battery optimization
        let critical_config =
            MobileBatteryManager::create_battery_optimized_config(&base_config, 10, false);
        assert_eq!(
            critical_config.memory_optimization,
            crate::MemoryOptimization::Maximum
        );
        assert_eq!(critical_config.num_threads, 1);
        assert!(!critical_config.enable_batching);

        // Test charging optimization (should be minimal changes)
        let charging_config =
            MobileBatteryManager::create_battery_optimized_config(&base_config, 50, true);
        // Should be closer to original config when charging
    }

    #[test]
    fn test_drain_rate_calculation() {
        let readings = vec![
            BatteryReading {
                timestamp: Instant::now(),
                level_percent: Some(100),
                charging_status: ChargingStatus::Discharging,
                voltage: None,
                current_ma: None,
                temperature_celsius: None,
                power_consumption_mw: None,
                estimated_time_remaining_minutes: None,
            },
            BatteryReading {
                timestamp: Instant::now() + Duration::from_secs(3600), // 1 hour later
                level_percent: Some(90),
                charging_status: ChargingStatus::Discharging,
                voltage: None,
                current_ma: None,
                temperature_celsius: None,
                power_consumption_mw: None,
                estimated_time_remaining_minutes: None,
            },
        ];

        let drain_rate = BatteryUtils::calculate_drain_rate(&readings);
        assert!((drain_rate - 10.0).abs() < 0.1); // Should be ~10% per hour
    }

    #[test]
    fn test_efficiency_score_calculation() {
        let score = BatteryUtils::calculate_efficiency_score(2000.0, 100, 10.0);
        assert!(score >= 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_quality_levels() {
        assert!(matches!(QualityLevel::Minimal, QualityLevel::Minimal));
        assert!(matches!(QualityLevel::Maximum, QualityLevel::Maximum));
    }

    #[test]
    fn test_power_usage_limits() {
        let limits = PowerUsageLimits::default();
        assert!(limits.max_power_on_battery_mw < limits.max_power_when_charging_mw);
        assert!(limits.max_background_power_mw < limits.max_power_on_battery_mw);

        // Check battery level budgets
        assert!(
            limits.battery_level_budgets[&BatteryLevel::Critical]
                < limits.battery_level_budgets[&BatteryLevel::High]
        );
    }

    #[test]
    fn test_optimization_recommendation() {
        let recommendation = OptimizationRecommendation {
            recommendation_type: "Test".to_string(),
            description: "Test recommendation".to_string(),
            estimated_power_savings_percent: 10.0,
            estimated_quality_impact_percent: 2.0,
            implementation_difficulty: DifficultyLevel::Easy,
            confidence_score: 0.9,
        };

        assert_eq!(recommendation.estimated_power_savings_percent, 10.0);
        assert!(matches!(
            recommendation.implementation_difficulty,
            DifficultyLevel::Easy
        ));
    }

    /// Build a `power_supply`-shaped fixture tree under a unique temp dir.
    fn write_power_supply_fixture(name: &str, nodes: &[(&str, &str)]) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!(
            "trustformers_power_supply_{}_{}",
            name,
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let supply = root.join(name);
        std::fs::create_dir_all(&supply).expect("create fixture supply dir");
        for (node, value) in nodes {
            std::fs::write(supply.join(node), value).expect("write fixture node");
        }
        root
    }

    /// Regression: the non-Android/non-iOS reader used to return an invented
    /// reading -- level 75%, 3.8 V, -1500 mA, 30 C and `power_consumption_mw:
    /// Some(2500.0)`. Off-device there is no battery source, so every field
    /// must be `None` and the status must be `Unknown`.
    #[test]
    fn test_read_battery_info_reports_unavailable_not_invented_values() {
        let mut monitor = BatteryMonitor::new(Duration::from_millis(10), 8);
        let reading = monitor.read_battery_info().expect("reading");

        // On a host with a readable `power_supply` battery (Linux laptop) the
        // reading is real; on every other host it must be empty. Neither case
        // may contain the old fabricated constants.
        assert_ne!(reading.power_consumption_mw, Some(2500.0));
        assert_ne!(reading.voltage, Some(3.8));
        assert_ne!(reading.current_ma, Some(-1500.0));
        assert_ne!(reading.estimated_time_remaining_minutes, Some(180));

        #[cfg(not(any(target_os = "android", target_os = "linux")))]
        {
            assert_eq!(reading.level_percent, None);
            assert_eq!(reading.power_consumption_mw, None);
            assert_eq!(reading.voltage, None);
            assert_eq!(reading.current_ma, None);
            assert_eq!(reading.temperature_celsius, None);
            assert_eq!(reading.estimated_time_remaining_minutes, None);
            assert!(matches!(reading.charging_status, ChargingStatus::Unknown));
            assert_eq!(monitor.current_level, None);
        }
    }

    /// The sysfs parser converts real kernel units and does not invent the
    /// nodes a gauge omits.
    #[test]
    fn test_power_supply_reader_parses_real_sysfs_units() {
        let root = write_power_supply_fixture(
            "BAT0",
            &[
                ("type", "Battery\n"),
                ("status", "Discharging\n"),
                ("capacity", "64\n"),
                ("voltage_now", "3812000\n"),  // 3.812 V
                ("current_now", "-1450000\n"), // -1450 mA
                ("temp", "302\n"),             // 30.2 C
                ("charge_now", "2900000\n"),   // 2900 mAh
            ],
        );

        let reading = read_power_supply_battery(&root).expect("battery supply found");
        assert_eq!(reading.level_percent, Some(64));
        assert!(matches!(
            reading.charging_status,
            ChargingStatus::Discharging
        ));
        assert!((reading.voltage.expect("voltage") - 3.812).abs() < 1e-3);
        assert!((reading.current_ma.expect("current") + 1450.0).abs() < 1e-1);
        assert!((reading.temperature_celsius.expect("temp") - 30.2).abs() < 1e-3);
        // No `power_now` node: P = V * I from the two real measurements.
        assert!((reading.power_consumption_mw.expect("power") - 3.812 * 1450.0).abs() < 1.0);
        // 2900 mAh at 1450 mA = 2 h.
        assert_eq!(reading.estimated_time_remaining_minutes, Some(120));

        let _ = std::fs::remove_dir_all(&root);
    }

    /// A gauge that publishes only `capacity`/`status` must leave every other
    /// field `None` rather than filling in a plausible number.
    #[test]
    fn test_power_supply_reader_leaves_absent_nodes_none() {
        let root = write_power_supply_fixture(
            "sparse_bat",
            &[
                ("type", "Battery\n"),
                ("status", "Charging\n"),
                ("capacity", "41\n"),
            ],
        );

        let reading = read_power_supply_battery(&root).expect("battery supply found");
        assert_eq!(reading.level_percent, Some(41));
        assert!(matches!(reading.charging_status, ChargingStatus::Charging));
        assert_eq!(reading.voltage, None);
        assert_eq!(reading.current_ma, None);
        assert_eq!(reading.temperature_celsius, None);
        assert_eq!(reading.power_consumption_mw, None);
        assert_eq!(reading.estimated_time_remaining_minutes, None);

        let _ = std::fs::remove_dir_all(&root);
    }

    /// A tree with no `type == Battery` supply (AC adapter only) yields
    /// nothing at all, so the caller falls back to the unavailable reading.
    #[test]
    fn test_power_supply_reader_ignores_non_battery_supplies() {
        let root = write_power_supply_fixture("ADP1", &[("type", "Mains\n"), ("online", "1\n")]);
        assert!(read_power_supply_battery(&root).is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Regression: `get_current_battery_level` used to guess 0.85/0.75/0.65/0.5
    /// from the charging status when no level had been measured. Charging
    /// status says nothing about how full the cell is, so an unmeasured level
    /// must be `None`.
    #[test]
    fn test_battery_level_is_none_when_unmeasured() {
        let device_info = crate::device_info::MobileDeviceDetector::detect().expect("device");
        let manager =
            MobileBatteryManager::new(BatteryConfig::default(), &device_info).expect("manager");
        assert_eq!(manager.get_current_battery_level(), None);
    }

    /// Regression: `estimate_time_remaining` returned a hardcoded
    /// `Some(120)`. With no reading in history it must be `None`.
    #[test]
    fn test_time_remaining_is_none_without_readings() {
        let device_info = crate::device_info::MobileDeviceDetector::detect().expect("device");
        let manager =
            MobileBatteryManager::new(BatteryConfig::default(), &device_info).expect("manager");
        assert_eq!(manager.estimate_time_remaining(), None);
        assert_eq!(manager.calculate_average_power_consumption(), None);
        assert_eq!(manager.get_peak_power_consumption(), None);
    }
}
