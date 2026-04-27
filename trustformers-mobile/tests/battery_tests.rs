//! Battery management tests for trustformers-mobile
//!
//! Tests battery configuration, threshold logic, and adaptive behavior
//! without relying on actual hardware.

use trustformers_mobile::{BatteryConfig, BatteryLevel, BatteryThresholds, PowerUsageLimits};

fn make_battery_thresholds() -> BatteryThresholds {
    BatteryThresholds {
        critical_percent: 5,
        low_percent: 20,
        medium_percent: 50,
        high_percent: 80,
        time_thresholds: trustformers_mobile::battery::TimeThresholds {
            critical_minutes: 15,
            low_minutes: 60,
            medium_minutes: 120,
        },
    }
}

fn make_power_limits() -> PowerUsageLimits {
    use std::collections::HashMap;
    let mut budgets = HashMap::new();
    budgets.insert(BatteryLevel::Critical, 100.0_f32);
    budgets.insert(BatteryLevel::Low, 300.0_f32);
    budgets.insert(BatteryLevel::Medium, 600.0_f32);
    budgets.insert(BatteryLevel::High, 1000.0_f32);
    budgets.insert(BatteryLevel::Full, 1500.0_f32);
    budgets.insert(BatteryLevel::Charging, 2000.0_f32);
    PowerUsageLimits {
        max_power_on_battery_mw: 1000.0,
        max_power_when_charging_mw: 2000.0,
        max_background_power_mw: 200.0,
        battery_level_budgets: budgets,
    }
}

fn make_battery_config() -> BatteryConfig {
    use trustformers_mobile::battery::QualityAdaptationStrategy;
    BatteryConfig {
        enable_monitoring: true,
        monitoring_interval_ms: 1000,
        enable_prediction: true,
        prediction_window_minutes: 30,
        enable_adaptive_quality: true,
        quality_strategy: QualityAdaptationStrategy::Stepped,
        battery_thresholds: make_battery_thresholds(),
        power_limits: make_power_limits(),
        enable_analytics: true,
        max_history_size: 100,
    }
}

#[test]
fn test_battery_config_default_fields() {
    let config = make_battery_config();
    assert!(config.enable_monitoring);
    assert!(config.enable_prediction);
    assert!(config.enable_adaptive_quality);
    assert!(config.enable_analytics);
    assert_eq!(config.monitoring_interval_ms, 1000);
}

#[test]
fn test_battery_level_threshold_ordering() {
    let thresholds = make_battery_thresholds();
    // Critical < Low < Medium < High
    assert!(thresholds.critical_percent < thresholds.low_percent);
    assert!(thresholds.low_percent < thresholds.medium_percent);
    assert!(thresholds.medium_percent < thresholds.high_percent);
}

#[test]
fn test_battery_level_critical_threshold_is_5_percent() {
    let thresholds = make_battery_thresholds();
    assert_eq!(thresholds.critical_percent, 5);
}

#[test]
fn test_battery_level_low_threshold_is_20_percent() {
    let thresholds = make_battery_thresholds();
    assert_eq!(thresholds.low_percent, 20);
}

#[test]
fn test_battery_level_enum_variants_exist() {
    let _critical = BatteryLevel::Critical;
    let _low = BatteryLevel::Low;
    let _medium = BatteryLevel::Medium;
    let _high = BatteryLevel::High;
    let _full = BatteryLevel::Full;
    let _charging = BatteryLevel::Charging;
}

#[test]
fn test_power_limits_charging_higher_than_battery() {
    let limits = make_power_limits();
    assert!(limits.max_power_when_charging_mw > limits.max_power_on_battery_mw);
}

#[test]
fn test_power_limits_background_lower_than_battery() {
    let limits = make_power_limits();
    assert!(limits.max_background_power_mw < limits.max_power_on_battery_mw);
}

#[test]
fn test_power_budget_by_level_critical_lowest() {
    let limits = make_power_limits();
    let critical_budget = limits
        .battery_level_budgets
        .get(&BatteryLevel::Critical)
        .copied()
        .unwrap_or(0.0);
    let high_budget = limits.battery_level_budgets.get(&BatteryLevel::High).copied().unwrap_or(0.0);
    assert!(critical_budget < high_budget);
}

#[test]
fn test_power_budget_by_level_charging_highest() {
    let limits = make_power_limits();
    let charging_budget = limits
        .battery_level_budgets
        .get(&BatteryLevel::Charging)
        .copied()
        .unwrap_or(0.0);
    let low_budget = limits.battery_level_budgets.get(&BatteryLevel::Low).copied().unwrap_or(0.0);
    assert!(charging_budget > low_budget);
}

#[test]
fn test_battery_config_serialization_roundtrip() {
    let config = make_battery_config();
    let json = serde_json::to_string(&config).expect("serialization should succeed");
    let restored: BatteryConfig =
        serde_json::from_str(&json).expect("deserialization should succeed");
    assert_eq!(restored.enable_monitoring, config.enable_monitoring);
    assert_eq!(
        restored.monitoring_interval_ms,
        config.monitoring_interval_ms
    );
    assert_eq!(
        restored.battery_thresholds.critical_percent,
        config.battery_thresholds.critical_percent,
    );
}

#[test]
fn test_battery_level_eq() {
    assert_eq!(BatteryLevel::Critical, BatteryLevel::Critical);
    assert_ne!(BatteryLevel::Critical, BatteryLevel::High);
}

#[test]
fn test_time_thresholds_ordering() {
    let thresholds = make_battery_thresholds();
    let t = &thresholds.time_thresholds;
    assert!(t.critical_minutes < t.low_minutes);
    assert!(t.low_minutes < t.medium_minutes);
}

#[test]
fn test_quality_adaptation_strategy_serialization() {
    use trustformers_mobile::battery::QualityAdaptationStrategy;
    let strategy = QualityAdaptationStrategy::Stepped;
    let json = serde_json::to_string(&strategy).expect("serialization should succeed");
    let restored: QualityAdaptationStrategy =
        serde_json::from_str(&json).expect("deserialization should succeed");
    assert_eq!(restored, strategy);
}
