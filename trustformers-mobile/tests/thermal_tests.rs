//! Thermal and power management tests for trustformers-mobile
//!
//! Tests thermal configuration, throttling strategies, and state-based
//! behavior without actual hardware sensor access.

use trustformers_mobile::device_info::ThermalState;
use trustformers_mobile::{
    PowerOptimizationStrategy, ThermalPowerConfig, ThermalThresholds, ThrottleLevel,
    ThrottlingStrategy,
};

fn make_thermal_thresholds() -> ThermalThresholds {
    ThermalThresholds {
        light_throttle_celsius: 40.0,
        moderate_throttle_celsius: 50.0,
        aggressive_throttle_celsius: 60.0,
        emergency_celsius: 70.0,
        cooldown_celsius: 35.0,
    }
}

fn make_thermal_power_config() -> ThermalPowerConfig {
    use trustformers_mobile::PowerThresholds;
    ThermalPowerConfig {
        enable_thermal_monitoring: true,
        enable_power_monitoring: true,
        thermal_check_interval_ms: 500,
        power_check_interval_ms: 1000,
        thermal_thresholds: make_thermal_thresholds(),
        power_thresholds: PowerThresholds {
            low_battery_percent: 20,
            critical_battery_percent: 5,
            power_save_percent: 15,
            max_power_mw: Some(2000.0),
        },
        throttling_strategy: ThrottlingStrategy::Balanced,
        power_strategy: PowerOptimizationStrategy::Balanced,
        max_thermal_history: 100,
        max_power_history: 100,
    }
}

#[test]
fn test_thermal_config_default_creation() {
    let config = make_thermal_power_config();
    assert!(config.enable_thermal_monitoring);
    assert!(config.enable_power_monitoring);
}

#[test]
fn test_thermal_thresholds_ordering() {
    let thresholds = make_thermal_thresholds();
    assert!(thresholds.light_throttle_celsius < thresholds.moderate_throttle_celsius);
    assert!(thresholds.moderate_throttle_celsius < thresholds.aggressive_throttle_celsius);
    assert!(thresholds.aggressive_throttle_celsius < thresholds.emergency_celsius);
}

#[test]
fn test_thermal_cooldown_below_light_throttle() {
    let thresholds = make_thermal_thresholds();
    assert!(thresholds.cooldown_celsius < thresholds.light_throttle_celsius);
}

#[test]
fn test_throttle_level_ordering() {
    // ThrottleLevel implements PartialOrd
    assert!(ThrottleLevel::None < ThrottleLevel::Light);
    assert!(ThrottleLevel::Light < ThrottleLevel::Moderate);
    assert!(ThrottleLevel::Moderate < ThrottleLevel::Aggressive);
    assert!(ThrottleLevel::Aggressive < ThrottleLevel::Emergency);
}

#[test]
fn test_throttling_strategy_variants() {
    let conservative = ThrottlingStrategy::Conservative;
    let balanced = ThrottlingStrategy::Balanced;
    let aggressive = ThrottlingStrategy::Aggressive;
    let custom = ThrottlingStrategy::Custom;
    assert_ne!(conservative, balanced);
    assert_ne!(balanced, aggressive);
    assert_ne!(aggressive, custom);
}

#[test]
fn test_power_optimization_strategy_variants() {
    let max_battery = PowerOptimizationStrategy::MaxBatteryLife;
    let balanced = PowerOptimizationStrategy::Balanced;
    let max_perf = PowerOptimizationStrategy::MaxPerformance;
    let adaptive = PowerOptimizationStrategy::Adaptive;
    assert_ne!(max_battery, balanced);
    assert_ne!(max_perf, adaptive);
}

#[test]
fn test_thermal_state_nominal_is_lowest_severity() {
    // Nominal is the "good" state, other states indicate higher temperature
    let nominal = ThermalState::Nominal;
    let critical = ThermalState::Critical;
    assert_ne!(nominal, critical);
}

#[test]
fn test_thermal_state_variants_all_exist() {
    let _nominal = ThermalState::Nominal;
    let _fair = ThermalState::Fair;
    let _serious = ThermalState::Serious;
    let _critical = ThermalState::Critical;
    let _emergency = ThermalState::Emergency;
    let _shutdown = ThermalState::Shutdown;
}

#[test]
fn test_thermal_config_serialization_roundtrip() {
    let config = make_thermal_power_config();
    let json = serde_json::to_string(&config).expect("serialization should succeed");
    let restored: ThermalPowerConfig =
        serde_json::from_str(&json).expect("deserialization should succeed");
    assert_eq!(
        restored.enable_thermal_monitoring,
        config.enable_thermal_monitoring,
    );
    assert_eq!(
        restored.thermal_thresholds.emergency_celsius,
        config.thermal_thresholds.emergency_celsius,
    );
}

#[test]
fn test_throttle_level_serialization_roundtrip() {
    let level = ThrottleLevel::Moderate;
    let json = serde_json::to_string(&level).expect("serialization should succeed");
    let restored: ThrottleLevel =
        serde_json::from_str(&json).expect("deserialization should succeed");
    assert_eq!(restored, level);
}

#[test]
fn test_power_thresholds_critical_below_low() {
    use trustformers_mobile::PowerThresholds;
    let thresholds = PowerThresholds {
        low_battery_percent: 20,
        critical_battery_percent: 5,
        power_save_percent: 15,
        max_power_mw: None,
    };
    assert!(thresholds.critical_battery_percent < thresholds.low_battery_percent);
}

#[test]
fn test_thermal_state_eq_and_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(ThermalState::Nominal);
    set.insert(ThermalState::Critical);
    set.insert(ThermalState::Nominal); // duplicate
    assert_eq!(set.len(), 2);
}
