#[cfg(test)]
mod tests {
    use crate::lifecycle::stats::*;
    use std::collections::HashMap;

    fn make_usage_stats(current: f32, avg: f32) -> UsageStats {
        UsageStats {
            current,
            average: avg,
            minimum: current * 0.5,
            maximum: current * 1.5,
            p95: current * 1.2,
            std_deviation: current * 0.1,
            sample_count: 100,
        }
    }

    fn make_battery_usage_stats() -> BatteryUsageStats {
        BatteryUsageStats {
            current_level_percent: 80,
            drain_rate_percent_per_hour: 5.0,
            avg_battery_level_percent: 75.0,
            time_since_last_charge_hours: 2.0,
            charging_cycles: 1,
            low_battery_events: 0,
            critical_battery_events: 0,
        }
    }

    fn make_storage_stats() -> StorageStats {
        StorageStats {
            read_operations: 1000,
            write_operations: 500,
            bytes_read: 1024 * 1024 * 100,
            bytes_written: 1024 * 1024 * 50,
            avg_read_speed_mbps: 200.0,
            avg_write_speed_mbps: 150.0,
            storage_usage_mb: 512,
            available_storage_mb: 4096,
        }
    }

    fn make_thermal_stats() -> ThermalStats {
        ThermalStats {
            current_temperature_celsius: 35.0,
            avg_temperature_celsius: 33.0,
            max_temperature_celsius: 42.0,
            thermal_events: 2,
            throttling_events: 1,
            time_in_thermal_warning_seconds: 30,
            temperature_trend: TemperatureTrend::Stable,
        }
    }

    // --- UsageStats Tests ---

    #[test]
    fn test_usage_stats_creation() {
        let stats = make_usage_stats(50.0, 45.0);
        assert_eq!(stats.current, 50.0);
        assert_eq!(stats.average, 45.0);
        assert_eq!(stats.sample_count, 100);
    }

    #[test]
    fn test_usage_stats_clone() {
        let stats = make_usage_stats(60.0, 55.0);
        let cloned = stats.clone();
        assert_eq!(cloned.current, 60.0);
        assert_eq!(cloned.sample_count, stats.sample_count);
    }

    #[test]
    fn test_usage_stats_serialization() {
        let stats = make_usage_stats(70.0, 65.0);
        let json = serde_json::to_string(&stats).expect("Failed to serialize");
        let deserialized: UsageStats = serde_json::from_str(&json).expect("Failed to deserialize");
        assert!((deserialized.current - 70.0).abs() < 1e-5);
    }

    // --- BatteryUsageStats Tests ---

    #[test]
    fn test_battery_usage_stats_creation() {
        let stats = make_battery_usage_stats();
        assert_eq!(stats.current_level_percent, 80);
        assert!((stats.drain_rate_percent_per_hour - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_battery_usage_stats_low_events() {
        let stats = BatteryUsageStats {
            low_battery_events: 3,
            critical_battery_events: 1,
            ..make_battery_usage_stats()
        };
        assert_eq!(stats.low_battery_events, 3);
        assert_eq!(stats.critical_battery_events, 1);
    }

    #[test]
    fn test_battery_usage_stats_serialization() {
        let stats = make_battery_usage_stats();
        let json = serde_json::to_string(&stats).expect("Failed to serialize");
        let deserialized: BatteryUsageStats =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized.current_level_percent, 80);
    }

    // --- StorageStats Tests ---

    #[test]
    fn test_storage_stats_creation() {
        let stats = make_storage_stats();
        assert_eq!(stats.read_operations, 1000);
        assert_eq!(stats.write_operations, 500);
    }

    #[test]
    fn test_storage_stats_byte_counts() {
        let stats = make_storage_stats();
        assert!(stats.bytes_read > stats.bytes_written);
    }

    #[test]
    fn test_storage_stats_serialization() {
        let stats = make_storage_stats();
        let json = serde_json::to_string(&stats).expect("Failed to serialize");
        let deserialized: StorageStats =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized.storage_usage_mb, 512);
    }

    // --- ThermalStats Tests ---

    #[test]
    fn test_thermal_stats_creation() {
        let stats = make_thermal_stats();
        assert!((stats.current_temperature_celsius - 35.0).abs() < 1e-5);
        assert_eq!(stats.temperature_trend, TemperatureTrend::Stable);
    }

    #[test]
    fn test_thermal_stats_with_events() {
        let stats = ThermalStats {
            thermal_events: 10,
            throttling_events: 5,
            ..make_thermal_stats()
        };
        assert_eq!(stats.thermal_events, 10);
        assert_eq!(stats.throttling_events, 5);
    }

    #[test]
    fn test_thermal_stats_serialization() {
        let stats = make_thermal_stats();
        let json = serde_json::to_string(&stats).expect("Failed to serialize");
        let deserialized: ThermalStats =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert!((deserialized.max_temperature_celsius - 42.0).abs() < 1e-5);
    }

    // --- TemperatureTrend Tests ---

    #[test]
    fn test_temperature_trend_variants() {
        assert_eq!(TemperatureTrend::Stable, TemperatureTrend::Stable);
        assert_ne!(TemperatureTrend::Rising, TemperatureTrend::Falling);
        let _ = TemperatureTrend::Oscillating;
    }

    #[test]
    fn test_temperature_trend_serialization() {
        let trend = TemperatureTrend::Rising;
        let json = serde_json::to_string(&trend).expect("Failed to serialize");
        let deserialized: TemperatureTrend =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized, TemperatureTrend::Rising);
    }

    // --- AccuracyTrend Tests ---

    #[test]
    fn test_accuracy_trend_variants() {
        assert_eq!(AccuracyTrend::Stable, AccuracyTrend::Stable);
        assert_ne!(AccuracyTrend::Improving, AccuracyTrend::Degrading);
        let _ = AccuracyTrend::Fluctuating;
    }

    // --- QueueWaitStats Tests ---

    #[test]
    fn test_queue_wait_stats_creation() {
        let stats = QueueWaitStats {
            avg_wait_time_seconds: 1.5,
            min_wait_time_seconds: 0.1,
            max_wait_time_seconds: 10.0,
            p95_wait_time_seconds: 5.0,
        };
        assert!(stats.avg_wait_time_seconds < stats.max_wait_time_seconds);
        assert!(stats.min_wait_time_seconds < stats.p95_wait_time_seconds);
    }

    // --- AvgResourceConsumption Tests ---

    #[test]
    fn test_avg_resource_consumption_creation() {
        let consumption = AvgResourceConsumption {
            avg_cpu_percent: 25.0,
            avg_memory_mb: 128.0,
            avg_network_mb: 10.0,
            avg_battery_mah: 50.0,
            avg_execution_time_seconds: 2.5,
        };
        assert!((consumption.avg_cpu_percent - 25.0).abs() < 1e-5);
    }

    #[test]
    fn test_avg_resource_consumption_serialization() {
        let consumption = AvgResourceConsumption {
            avg_cpu_percent: 30.0,
            avg_memory_mb: 256.0,
            avg_network_mb: 20.0,
            avg_battery_mah: 75.0,
            avg_execution_time_seconds: 3.0,
        };
        let json = serde_json::to_string(&consumption).expect("Failed to serialize");
        let deserialized: AvgResourceConsumption =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert!((deserialized.avg_memory_mb - 256.0).abs() < 1e-5);
    }

    // --- ModelLoadingStats Tests ---

    #[test]
    fn test_model_loading_stats_creation() {
        let stats = ModelLoadingStats {
            total_loads: 10,
            avg_loading_time_seconds: 1.2,
            cache_hit_rate_percent: 80.0,
            failed_loads: 1,
            loaded_models_memory_mb: 512,
        };
        assert_eq!(stats.total_loads, 10);
        assert!((stats.cache_hit_rate_percent - 80.0).abs() < 1e-5);
    }

    // --- QueueBacklogStats Tests ---

    #[test]
    fn test_queue_backlog_stats_creation() {
        let stats = QueueBacklogStats {
            current_queue_size: 5,
            avg_queue_size: 3.2,
            max_queue_size: 20,
            queue_overflow_events: 0,
            avg_processing_time_ms: 50.0,
        };
        assert_eq!(stats.current_queue_size, 5);
        assert_eq!(stats.queue_overflow_events, 0);
    }

    // --- AccuracyStats Tests ---

    #[test]
    fn test_accuracy_stats_creation() {
        let stats = AccuracyStats {
            avg_accuracy_score: 92.5,
            accuracy_trend: AccuracyTrend::Improving,
            model_drift_events: 0,
            accuracy_degradation_events: 1,
        };
        assert!((stats.avg_accuracy_score - 92.5).abs() < 1e-5);
        assert_eq!(stats.accuracy_trend, AccuracyTrend::Improving);
    }

    // --- ResourceUsageStats Tests ---

    #[test]
    fn test_resource_usage_stats_creation() {
        let stats = ResourceUsageStats {
            cpu_stats: make_usage_stats(40.0, 35.0),
            memory_stats: make_usage_stats(60.0, 55.0),
            network_stats: make_usage_stats(10.0, 8.0),
            battery_stats: make_battery_usage_stats(),
            gpu_stats: None,
            storage_stats: make_storage_stats(),
            thermal_stats: make_thermal_stats(),
        };
        assert!((stats.cpu_stats.current - 40.0).abs() < 1e-5);
        assert!(stats.gpu_stats.is_none());
    }

    #[test]
    fn test_resource_usage_stats_with_gpu() {
        let stats = ResourceUsageStats {
            cpu_stats: make_usage_stats(40.0, 35.0),
            memory_stats: make_usage_stats(60.0, 55.0),
            network_stats: make_usage_stats(10.0, 8.0),
            battery_stats: make_battery_usage_stats(),
            gpu_stats: Some(make_usage_stats(80.0, 70.0)),
            storage_stats: make_storage_stats(),
            thermal_stats: make_thermal_stats(),
        };
        assert!(stats.gpu_stats.is_some());
        let gpu = stats.gpu_stats.as_ref().expect("expected gpu stats");
        assert!((gpu.current - 80.0).abs() < 1e-5);
    }
}
