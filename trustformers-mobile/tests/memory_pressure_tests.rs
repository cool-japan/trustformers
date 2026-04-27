//! Memory pressure and lifecycle tests for trustformers-mobile
//!
//! Tests memory pressure levels, cleanup strategies, and lifecycle state
//! transitions without relying on actual hardware.

use std::collections::HashMap;
use trustformers_mobile::lifecycle::config::{
    CleanupPriority, CleanupStrategy, MemoryPressureThresholds, MemoryWarningResponse,
    ModelEvictionPolicy,
};
use trustformers_mobile::{AppState, LifecycleConfig, MemoryPressureLevel, StateTransition};

fn make_memory_pressure_thresholds() -> MemoryPressureThresholds {
    MemoryPressureThresholds {
        warning_percent: 70,
        critical_percent: 85,
        emergency_percent: 95,
    }
}

fn make_cleanup_strategy(level: MemoryPressureLevel) -> CleanupStrategy {
    match level {
        MemoryPressureLevel::Warning => CleanupStrategy {
            clear_model_cache: true,
            clear_intermediate_tensors: true,
            reduce_batch_sizes: true,
            compress_models: false,
            offload_to_disk: false,
            cleanup_priority: CleanupPriority::Low,
        },
        MemoryPressureLevel::Critical => CleanupStrategy {
            clear_model_cache: true,
            clear_intermediate_tensors: true,
            reduce_batch_sizes: true,
            compress_models: true,
            offload_to_disk: false,
            cleanup_priority: CleanupPriority::High,
        },
        MemoryPressureLevel::Emergency => CleanupStrategy {
            clear_model_cache: true,
            clear_intermediate_tensors: true,
            reduce_batch_sizes: true,
            compress_models: true,
            offload_to_disk: true,
            cleanup_priority: CleanupPriority::Critical,
        },
        MemoryPressureLevel::Normal => CleanupStrategy {
            clear_model_cache: false,
            clear_intermediate_tensors: false,
            reduce_batch_sizes: false,
            compress_models: false,
            offload_to_disk: false,
            cleanup_priority: CleanupPriority::Low,
        },
    }
}

#[test]
fn test_memory_pressure_level_ordering() {
    assert!(MemoryPressureLevel::Normal < MemoryPressureLevel::Warning);
    assert!(MemoryPressureLevel::Warning < MemoryPressureLevel::Critical);
    assert!(MemoryPressureLevel::Critical < MemoryPressureLevel::Emergency);
}

#[test]
fn test_memory_pressure_thresholds_ordering() {
    let thresholds = make_memory_pressure_thresholds();
    assert!(thresholds.warning_percent < thresholds.critical_percent);
    assert!(thresholds.critical_percent < thresholds.emergency_percent);
}

#[test]
fn test_cleanup_strategy_normal_does_nothing() {
    let strategy = make_cleanup_strategy(MemoryPressureLevel::Normal);
    assert!(!strategy.clear_model_cache);
    assert!(!strategy.clear_intermediate_tensors);
    assert!(!strategy.offload_to_disk);
}

#[test]
fn test_cleanup_strategy_warning_clears_cache() {
    let strategy = make_cleanup_strategy(MemoryPressureLevel::Warning);
    assert!(strategy.clear_model_cache);
    assert!(strategy.clear_intermediate_tensors);
}

#[test]
fn test_cleanup_strategy_critical_compresses_models() {
    let strategy = make_cleanup_strategy(MemoryPressureLevel::Critical);
    assert!(strategy.compress_models);
    // Critical does not yet offload
    assert!(!strategy.offload_to_disk);
}

#[test]
fn test_cleanup_strategy_emergency_offloads_to_disk() {
    let strategy = make_cleanup_strategy(MemoryPressureLevel::Emergency);
    assert!(strategy.offload_to_disk);
    assert!(strategy.compress_models);
}

#[test]
fn test_cleanup_priority_ordering() {
    assert!(CleanupPriority::Low < CleanupPriority::Medium);
    assert!(CleanupPriority::Medium < CleanupPriority::High);
    assert!(CleanupPriority::High < CleanupPriority::Critical);
}

#[test]
fn test_app_state_variants_all_exist() {
    let _launching = AppState::Launching;
    let _active = AppState::Active;
    let _background = AppState::Background;
    let _inactive = AppState::Inactive;
    let _suspended = AppState::Suspended;
    let _terminating = AppState::Terminating;
    let _unknown = AppState::Unknown;
}

#[test]
fn test_state_transition_variants_exist() {
    let _t1 = StateTransition::LaunchToActive;
    let _t2 = StateTransition::ActiveToBackground;
    let _t3 = StateTransition::BackgroundToActive;
    let _t4 = StateTransition::AnyToTerminating;
}

#[test]
fn test_lifecycle_config_default_creation() {
    let config = LifecycleConfig::default();
    assert!(config.enable_background_execution);
    assert!(config.enable_state_persistence);
    assert_eq!(config.background_execution_limit_seconds, 30);
}

#[test]
fn test_lifecycle_config_serialization_roundtrip() {
    let config = LifecycleConfig::default();
    let json = serde_json::to_string(&config).expect("serialization should succeed");
    let restored: LifecycleConfig =
        serde_json::from_str(&json).expect("deserialization should succeed");
    assert_eq!(
        restored.enable_background_execution,
        config.enable_background_execution,
    );
    assert_eq!(
        restored.background_execution_limit_seconds,
        config.background_execution_limit_seconds,
    );
}

#[test]
fn test_memory_pressure_level_hash_usable_as_map_key() {
    let mut map: HashMap<MemoryPressureLevel, &str> = HashMap::new();
    map.insert(MemoryPressureLevel::Normal, "normal");
    map.insert(MemoryPressureLevel::Warning, "warning");
    map.insert(MemoryPressureLevel::Critical, "critical");
    map.insert(MemoryPressureLevel::Emergency, "emergency");
    assert_eq!(map.len(), 4);
    assert_eq!(map[&MemoryPressureLevel::Warning], "warning");
}

#[test]
fn test_model_eviction_policy_variants() {
    let _lru = ModelEvictionPolicy::LeastRecentlyUsed;
    let _lfu = ModelEvictionPolicy::LeastFrequentlyUsed;
    let _size = ModelEvictionPolicy::SizeBasedEviction;
    let _priority = ModelEvictionPolicy::PriorityBasedEviction;
    let _adaptive = ModelEvictionPolicy::AdaptiveEviction;
}

#[test]
fn test_memory_warning_response_variants() {
    let _clear = MemoryWarningResponse::ClearCaches;
    let _reduce = MemoryWarningResponse::ReduceBatchSizes;
    let _offload = MemoryWarningResponse::OffloadModels;
    let _pause = MemoryWarningResponse::PauseBackgroundTasks;
    let _notify = MemoryWarningResponse::NotifyUser;
    let _gc = MemoryWarningResponse::ForceGarbageCollection;
}
