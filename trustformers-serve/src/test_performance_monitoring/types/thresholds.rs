//! Thresholds Type Definitions

use serde::{Deserialize, Serialize};
use std::time::Duration;

// Import types from sibling modules
use super::enums::{PressureLevel, ThresholdAction};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub cpu_usage_threshold: f64,
    pub memory_usage_threshold: f64,
    pub execution_time_threshold: Duration,
    pub failure_rate_threshold: f64,
    pub throughput_threshold: f64,
    pub resource_pressure_threshold: PressureLevel,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_usage_threshold: 0.8,                           // 80%
            memory_usage_threshold: 0.85,                       // 85%
            execution_time_threshold: Duration::from_secs(300), // 5 minutes
            failure_rate_threshold: 0.1,                        // 10%
            throughput_threshold: 1.0,                          // 1 test per second minimum
            resource_pressure_threshold: PressureLevel::High,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerformanceThreshold {
    /// Name of the metric to monitor
    pub metric_name: String,
    /// Threshold value
    pub threshold_value: f64,
    /// Comparison operator (greater_than, less_than, equals)
    pub comparison_operator: String,
    /// Action to take when threshold is exceeded
    pub action: ThresholdAction,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ResourceThresholds {
    pub cpu_threshold: f64,
    pub memory_threshold: f64,
    pub disk_threshold: f64,
    pub network_threshold: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThresholdCache {
    pub enabled: bool,
    pub ttl: Duration,
    pub max_entries: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    pub cpu_threshold: f64,
    pub memory_threshold: f64,
    pub latency_threshold_ms: f64,
    pub max_execution_time: Duration,
    pub max_memory_usage: u64,
    pub max_cpu_usage: f64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            cpu_threshold: 0.8,
            memory_threshold: 0.85,
            latency_threshold_ms: 1000.0,
            max_execution_time: Duration::from_secs(300),
            max_memory_usage: 1024 * 1024 * 1024,
            max_cpu_usage: 85.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SystemPressureThresholds {
    pub cpu_pressure_threshold: f64,
    pub memory_pressure_threshold: f64,
    pub disk_pressure_threshold: f64,
    pub file_descriptor_threshold: u32,
}

impl Default for SystemPressureThresholds {
    fn default() -> Self {
        Self {
            cpu_pressure_threshold: 80.0,
            memory_pressure_threshold: 0.85,
            disk_pressure_threshold: 80.0,
            file_descriptor_threshold: 10_000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg { state: seed }
        }
        fn next(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            self.state
        }
        fn next_f64(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn test_alert_thresholds_default() {
        let at = AlertThresholds::default();
        assert!((at.cpu_usage_threshold - 0.8).abs() < f64::EPSILON);
        assert!((at.memory_usage_threshold - 0.85).abs() < f64::EPSILON);
        assert_eq!(at.execution_time_threshold, Duration::from_secs(300));
        assert!((at.failure_rate_threshold - 0.1).abs() < f64::EPSILON);
        assert!((at.throughput_threshold - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_alert_thresholds_cpu_range() {
        let at = AlertThresholds::default();
        assert!(at.cpu_usage_threshold > 0.0 && at.cpu_usage_threshold < 1.0);
    }

    #[test]
    fn test_alert_thresholds_memory_range() {
        let at = AlertThresholds::default();
        assert!(at.memory_usage_threshold > 0.0 && at.memory_usage_threshold < 1.0);
    }

    #[test]
    fn test_alert_thresholds_custom_values() {
        let at = AlertThresholds {
            cpu_usage_threshold: 0.95,
            memory_usage_threshold: 0.90,
            execution_time_threshold: Duration::from_secs(600),
            failure_rate_threshold: 0.05,
            throughput_threshold: 5.0,
            resource_pressure_threshold: super::super::enums::PressureLevel::Critical,
        };
        assert!((at.cpu_usage_threshold - 0.95).abs() < f64::EPSILON);
        assert!((at.memory_usage_threshold - 0.90).abs() < f64::EPSILON);
        assert_eq!(at.execution_time_threshold, Duration::from_secs(600));
    }

    #[test]
    fn test_performance_threshold_log_action() {
        let pt = PerformanceThreshold {
            metric_name: "cpu_usage".to_string(),
            threshold_value: 80.0,
            comparison_operator: "greater_than".to_string(),
            action: super::super::enums::ThresholdAction::Log,
        };
        assert_eq!(pt.metric_name, "cpu_usage");
        assert!((pt.threshold_value - 80.0).abs() < f64::EPSILON);
        assert_eq!(pt.comparison_operator, "greater_than");
    }

    #[test]
    fn test_performance_threshold_alert_action() {
        let pt = PerformanceThreshold {
            metric_name: "error_rate".to_string(),
            threshold_value: 0.05,
            comparison_operator: "greater_than".to_string(),
            action: super::super::enums::ThresholdAction::Alert,
        };
        assert_eq!(pt.metric_name, "error_rate");
        assert!(pt.threshold_value > 0.0);
    }

    #[test]
    fn test_resource_thresholds_default() {
        let rt = ResourceThresholds::default();
        assert_eq!(rt.cpu_threshold, 0.0);
        assert_eq!(rt.memory_threshold, 0.0);
        assert_eq!(rt.disk_threshold, 0.0);
        assert_eq!(rt.network_threshold, 0.0);
    }

    #[test]
    fn test_resource_thresholds_custom() {
        let rt = ResourceThresholds {
            cpu_threshold: 0.8,
            memory_threshold: 0.75,
            disk_threshold: 0.90,
            network_threshold: 0.60,
        };
        assert!(rt.cpu_threshold > 0.0);
        assert!(rt.memory_threshold > 0.0);
        assert!(rt.disk_threshold > 0.0);
        assert!(rt.network_threshold > 0.0);
    }

    #[test]
    fn test_threshold_cache_construction() {
        let tc = ThresholdCache {
            enabled: true,
            ttl: Duration::from_secs(60),
            max_entries: 1000,
        };
        assert!(tc.enabled);
        assert_eq!(tc.ttl, Duration::from_secs(60));
        assert_eq!(tc.max_entries, 1000);
    }

    #[test]
    fn test_threshold_cache_disabled() {
        let tc = ThresholdCache {
            enabled: false,
            ttl: Duration::from_secs(0),
            max_entries: 0,
        };
        assert!(!tc.enabled);
        assert_eq!(tc.max_entries, 0);
    }

    #[test]
    fn test_performance_thresholds_default() {
        let pt = PerformanceThresholds::default();
        assert!((pt.cpu_threshold - 0.8).abs() < f64::EPSILON);
        assert!((pt.memory_threshold - 0.85).abs() < f64::EPSILON);
        assert!((pt.latency_threshold_ms - 1000.0).abs() < f64::EPSILON);
        assert_eq!(pt.max_execution_time, Duration::from_secs(300));
        assert_eq!(pt.max_memory_usage, 1024 * 1024 * 1024);
        assert!((pt.max_cpu_usage - 85.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_performance_thresholds_latency_positive() {
        let pt = PerformanceThresholds::default();
        assert!(pt.latency_threshold_ms > 0.0);
    }

    #[test]
    fn test_performance_thresholds_memory_limit() {
        let pt = PerformanceThresholds::default();
        assert!(pt.max_memory_usage > 0);
        assert!(pt.max_memory_usage == 1024 * 1024 * 1024); // 1 GB
    }

    #[test]
    fn test_system_pressure_thresholds_default() {
        let spt = SystemPressureThresholds::default();
        assert!((spt.cpu_pressure_threshold - 80.0).abs() < f64::EPSILON);
        assert!((spt.memory_pressure_threshold - 0.85).abs() < f64::EPSILON);
        assert!((spt.disk_pressure_threshold - 80.0).abs() < f64::EPSILON);
        assert_eq!(spt.file_descriptor_threshold, 10_000);
    }

    #[test]
    fn test_system_pressure_thresholds_fd_positive() {
        let spt = SystemPressureThresholds::default();
        assert!(spt.file_descriptor_threshold > 0);
    }

    #[test]
    fn test_system_pressure_thresholds_custom() {
        let spt = SystemPressureThresholds {
            cpu_pressure_threshold: 90.0,
            memory_pressure_threshold: 0.95,
            disk_pressure_threshold: 85.0,
            file_descriptor_threshold: 50_000,
        };
        assert!(spt.cpu_pressure_threshold > 80.0);
        assert!(spt.memory_pressure_threshold > 0.85);
        assert!(spt.file_descriptor_threshold > 10_000);
    }

    #[test]
    fn test_resource_thresholds_random_values() {
        let mut lcg = Lcg::new(314159);
        for _ in 0..5 {
            let cpu = lcg.next_f64();
            let mem = lcg.next_f64();
            let rt = ResourceThresholds {
                cpu_threshold: cpu,
                memory_threshold: mem,
                disk_threshold: lcg.next_f64(),
                network_threshold: lcg.next_f64(),
            };
            assert!(rt.cpu_threshold >= 0.0 && rt.cpu_threshold <= 1.0);
            assert!(rt.memory_threshold >= 0.0 && rt.memory_threshold <= 1.0);
        }
    }

    #[test]
    fn test_alert_thresholds_pressure_level_display() {
        let at = AlertThresholds::default();
        let display = format!("{}", at.resource_pressure_threshold);
        assert!(!display.is_empty());
    }

    #[test]
    fn test_performance_threshold_custom_operator() {
        let pt = PerformanceThreshold {
            metric_name: "p99_latency".to_string(),
            threshold_value: 500.0,
            comparison_operator: "less_than".to_string(),
            action: super::super::enums::ThresholdAction::Stop,
        };
        assert_eq!(pt.comparison_operator, "less_than");
        assert!((pt.threshold_value - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_threshold_cache_ttl_nonzero() {
        let tc = ThresholdCache {
            enabled: true,
            ttl: Duration::from_secs(300),
            max_entries: 5000,
        };
        assert!(tc.ttl > Duration::from_secs(0));
        assert!(tc.max_entries > 0);
    }
}
