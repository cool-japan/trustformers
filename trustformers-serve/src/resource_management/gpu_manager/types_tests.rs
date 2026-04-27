//! Tests for GPU manager types

use super::types::*;

#[test]
fn test_gpu_pool_config_default() {
    let config = GpuPoolConfig::default();
    assert!(config.max_devices > 0);
}

#[test]
fn test_gpu_monitoring_config_default() {
    let config = GpuMonitoringConfig::default();
    let _ = format!("{:?}", config);
}

#[test]
fn test_gpu_alert_config_default() {
    let config = GpuAlertConfig::default();
    let _ = format!("{:?}", config);
}

#[test]
fn test_gpu_alert_thresholds_default() {
    let thresholds = GpuAlertThresholds::default();
    assert!(thresholds.temperature_warning > 0.0);
    assert!(thresholds.temperature_critical > thresholds.temperature_warning);
}

#[test]
fn test_gpu_performance_analysis_default() {
    let analysis = GpuPerformanceAnalysis::default();
    let _ = format!("{:?}", analysis);
}

#[test]
fn test_load_balancing_strategy_default() {
    let strategy = LoadBalancingStrategy::default();
    let _ = format!("{}", strategy);
}

#[test]
fn test_alert_severity_priority() {
    assert_eq!(AlertSeverity::Info.priority(), 0);
    assert_eq!(AlertSeverity::Warning.priority(), 1);
    assert_eq!(AlertSeverity::Error.priority(), 2);
    assert_eq!(AlertSeverity::Critical.priority(), 3);
}

#[test]
fn test_alert_severity_immediate_attention() {
    assert!(!AlertSeverity::Info.requires_immediate_attention());
    assert!(!AlertSeverity::Warning.requires_immediate_attention());
    assert!(AlertSeverity::Error.requires_immediate_attention());
    assert!(AlertSeverity::Critical.requires_immediate_attention());
}

#[test]
fn test_gpu_capability_cuda() {
    let cap = GpuCapability::Cuda("12.0".to_string());
    assert!(cap.supports_framework("cuda"));
    assert!(!cap.supports_framework("opencl"));
}

#[test]
fn test_gpu_capability_opencl() {
    let cap = GpuCapability::OpenCl("3.0".to_string());
    assert!(cap.supports_framework("opencl"));
    assert!(!cap.supports_framework("vulkan"));
}

#[test]
fn test_gpu_capability_vulkan() {
    let cap = GpuCapability::Vulkan("1.3".to_string());
    assert!(cap.supports_framework("vulkan"));
}

#[test]
fn test_gpu_capability_ml() {
    let cap = GpuCapability::MachineLearning(vec!["pytorch".to_string()]);
    assert!(cap.supports_framework("pytorch"));
    assert!(!cap.supports_framework("jax"));
}

#[test]
fn test_gpu_constraint_type_is_resource() {
    assert!(GpuConstraintType::MaxMemoryUsage.is_resource_constraint());
    assert!(GpuConstraintType::MaxUtilization.is_resource_constraint());
    assert!(GpuConstraintType::PowerLimit.is_resource_constraint());
}

#[test]
fn test_gpu_constraint_type_is_performance() {
    assert!(GpuConstraintType::MinPerformance.is_performance_constraint());
    assert!(GpuConstraintType::TemperatureLimit.is_performance_constraint());
}

#[test]
fn test_trend_direction_problems() {
    assert!(TrendDirection::Degrading.indicates_problems());
    assert!(!TrendDirection::Improving.indicates_problems());
}

#[test]
fn test_trend_direction_positive() {
    assert!(TrendDirection::Improving.is_positive());
    assert!(TrendDirection::Stable.is_positive());
    assert!(!TrendDirection::Degrading.is_positive());
}

#[test]
fn test_gpu_device_status_display() {
    assert_eq!(format!("{}", GpuDeviceStatus::Available), "Available");
    assert_eq!(format!("{}", GpuDeviceStatus::Busy), "Busy");
}

#[test]
fn test_alert_severity_display() {
    assert_eq!(format!("{}", AlertSeverity::Critical), "Critical");
}

#[test]
fn test_gpu_device_status_available() {
    let s = GpuDeviceStatus::Available;
    let _ = format!("{:?}", s);
}

#[test]
fn test_gpu_device_status_busy() {
    let s = GpuDeviceStatus::Busy;
    let _ = format!("{:?}", s);
}

#[test]
fn test_gpu_device_status_error() {
    let s = GpuDeviceStatus::Error("test error".to_string());
    let _ = format!("{:?}", s);
}

#[test]
fn test_gpu_manager_error_device_not_found() {
    let err = GpuManagerError::DeviceNotFound { device_id: 0 };
    let _ = format!("{}", err);
}

#[test]
fn test_gpu_manager_error_insufficient_memory() {
    let err = GpuManagerError::InsufficientMemory { required_mb: 8192, available_mb: 4096 };
    let msg = format!("{}", err);
    assert!(msg.contains("8192"));
}

#[test]
fn test_gpu_result_ok() {
    let result: GpuResult<i32> = Ok(42);
    assert!(result.is_ok());
}

#[test]
fn test_gpu_result_err() {
    let result: GpuResult<i32> = Err(GpuManagerError::DeviceNotFound { device_id: 0 });
    assert!(result.is_err());
}

#[test]
fn test_gpu_benchmark_type_compute() {
    let t = GpuBenchmarkType::Compute;
    let _ = format!("{:?}", t);
}

#[test]
fn test_gpu_benchmark_type_memory_bandwidth() {
    let t = GpuBenchmarkType::MemoryBandwidth;
    let _ = format!("{:?}", t);
}

#[test]
fn test_regression_severity_minor() {
    let s = RegressionSeverity::Minor;
    let _ = format!("{:?}", s);
}

#[test]
fn test_regression_severity_moderate() {
    let s = RegressionSeverity::Moderate;
    let _ = format!("{:?}", s);
}

#[test]
fn test_recommendation_difficulty_easy() {
    let d = RecommendationDifficulty::Easy;
    let _ = format!("{:?}", d);
}

#[test]
fn test_recommendation_difficulty_hard() {
    let d = RecommendationDifficulty::Hard;
    let _ = format!("{:?}", d);
}

#[test]
fn test_recommendation_difficulty_very_hard() {
    let d = RecommendationDifficulty::VeryHard;
    let _ = format!("{:?}", d);
}
