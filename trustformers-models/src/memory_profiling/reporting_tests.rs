#[cfg(test)]
mod tests {
    use crate::memory_profiling::reporting::*;
    use crate::memory_profiling::types::*;
    use std::collections::HashMap;

    // --- MemorySummary tests ---

    #[test]
    fn test_memory_summary_creation() {
        let summary = MemorySummary {
            current_memory_mb: 256.0,
            peak_memory_mb: 512.0,
            average_memory_mb: 300.0,
            total_allocations: 10000,
            potential_leaks: 5,
            active_alerts: 2,
        };
        assert!((summary.current_memory_mb - 256.0).abs() < f64::EPSILON);
        assert!((summary.peak_memory_mb - 512.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_memory_summary_zero_values() {
        let summary = MemorySummary {
            current_memory_mb: 0.0,
            peak_memory_mb: 0.0,
            average_memory_mb: 0.0,
            total_allocations: 0,
            potential_leaks: 0,
            active_alerts: 0,
        };
        assert_eq!(summary.total_allocations, 0);
        assert_eq!(summary.potential_leaks, 0);
    }

    #[test]
    fn test_memory_summary_peak_greater_than_current() {
        let summary = MemorySummary {
            current_memory_mb: 200.0,
            peak_memory_mb: 500.0,
            average_memory_mb: 300.0,
            total_allocations: 5000,
            potential_leaks: 0,
            active_alerts: 0,
        };
        assert!(summary.peak_memory_mb >= summary.current_memory_mb);
    }

    #[test]
    fn test_memory_summary_clone() {
        let summary = MemorySummary {
            current_memory_mb: 128.0,
            peak_memory_mb: 256.0,
            average_memory_mb: 180.0,
            total_allocations: 1000,
            potential_leaks: 1,
            active_alerts: 3,
        };
        let cloned = summary.clone();
        assert!((cloned.current_memory_mb - 128.0).abs() < f64::EPSILON);
        assert_eq!(cloned.total_allocations, 1000);
    }

    #[test]
    fn test_memory_summary_with_alerts() {
        let summary = MemorySummary {
            current_memory_mb: 900.0,
            peak_memory_mb: 1000.0,
            average_memory_mb: 800.0,
            total_allocations: 50000,
            potential_leaks: 10,
            active_alerts: 5,
        };
        assert!(summary.active_alerts > 0);
        assert!(summary.potential_leaks > 0);
    }

    // --- MemoryUsageSummary (from types) tests ---

    #[test]
    fn test_memory_usage_summary_empty() {
        let summary = MemoryUsageSummary {
            total_runtime_seconds: 0.0,
            peak_memory_mb: 0.0,
            average_memory_mb: 0.0,
            memory_efficiency_score: 0.0,
            total_allocations: 0,
            total_deallocations: 0,
            leaked_allocations: 0,
            fragmentation_events: 0,
            gc_pressure_events: 0,
            alert_count_by_severity: HashMap::new(),
        };
        assert!((summary.total_runtime_seconds - 0.0).abs() < f64::EPSILON);
        assert_eq!(summary.total_allocations, 0);
    }

    #[test]
    fn test_memory_usage_summary_with_data() {
        let mut alert_counts = HashMap::new();
        alert_counts.insert(AlertSeverity::Warning, 3);
        alert_counts.insert(AlertSeverity::Error, 1);

        let summary = MemoryUsageSummary {
            total_runtime_seconds: 300.0,
            peak_memory_mb: 2048.0,
            average_memory_mb: 1024.0,
            memory_efficiency_score: 0.85,
            total_allocations: 100000,
            total_deallocations: 99000,
            leaked_allocations: 1000,
            fragmentation_events: 50,
            gc_pressure_events: 10,
            alert_count_by_severity: alert_counts,
        };
        assert!((summary.memory_efficiency_score - 0.85).abs() < f64::EPSILON);
        assert_eq!(summary.leaked_allocations, 1000);
        assert_eq!(summary.alert_count_by_severity.len(), 2);
    }

    #[test]
    fn test_memory_usage_summary_efficiency_score() {
        let summary = MemoryUsageSummary {
            total_runtime_seconds: 60.0,
            peak_memory_mb: 512.0,
            average_memory_mb: 400.0,
            memory_efficiency_score: 0.92,
            total_allocations: 10000,
            total_deallocations: 10000,
            leaked_allocations: 0,
            fragmentation_events: 0,
            gc_pressure_events: 0,
            alert_count_by_severity: HashMap::new(),
        };
        assert!(summary.memory_efficiency_score > 0.0);
        assert!(summary.memory_efficiency_score <= 1.0);
    }

    #[test]
    fn test_memory_usage_summary_no_leaks() {
        let summary = MemoryUsageSummary {
            total_runtime_seconds: 120.0,
            peak_memory_mb: 256.0,
            average_memory_mb: 200.0,
            memory_efficiency_score: 0.95,
            total_allocations: 5000,
            total_deallocations: 5000,
            leaked_allocations: 0,
            fragmentation_events: 2,
            gc_pressure_events: 0,
            alert_count_by_severity: HashMap::new(),
        };
        assert_eq!(summary.leaked_allocations, 0);
        assert_eq!(summary.total_allocations, summary.total_deallocations);
    }

    // --- ProfilerConfig tests ---

    #[test]
    fn test_profiler_config_default() {
        let config = ProfilerConfig::default();
        assert_eq!(config.max_data_points, 10000);
        assert_eq!(config.collection_interval_ms, 1000);
        assert!(config.enable_leak_detection);
        assert!(config.enable_pattern_analysis);
    }

    #[test]
    fn test_profiler_config_thresholds() {
        let config = ProfilerConfig::default();
        assert!((config.memory_alert_threshold_mb - 1024.0).abs() < f64::EPSILON);
        assert!(config.enable_gc_suggestions);
    }

    #[test]
    fn test_profiler_config_output_dir() {
        let config = ProfilerConfig::default();
        assert_eq!(config.output_dir, "./memory_reports");
    }

    #[test]
    fn test_profiler_config_custom() {
        let config = ProfilerConfig {
            max_data_points: 5000,
            collection_interval_ms: 500,
            enable_leak_detection: false,
            enable_pattern_analysis: false,
            memory_alert_threshold_mb: 512.0,
            enable_gc_suggestions: false,
            output_dir: "/tmp/reports".to_string(),
        };
        assert_eq!(config.max_data_points, 5000);
        assert!(!config.enable_leak_detection);
        assert_eq!(config.output_dir, "/tmp/reports");
    }

    // --- MemoryAlert tests ---

    #[test]
    fn test_memory_alert_high_usage() {
        let alert = MemoryAlert {
            id: uuid::Uuid::new_v4(),
            timestamp: std::time::SystemTime::now(),
            alert_type: MemoryAlertType::HighMemoryUsage,
            severity: AlertSeverity::Warning,
            message: "Memory usage is high".to_string(),
            details: HashMap::new(),
            recommendations: vec!["Reduce batch size".to_string()],
        };
        assert!(matches!(alert.alert_type, MemoryAlertType::HighMemoryUsage));
        assert!(matches!(alert.severity, AlertSeverity::Warning));
    }

    #[test]
    fn test_memory_alert_leak_detected() {
        let alert = MemoryAlert {
            id: uuid::Uuid::new_v4(),
            timestamp: std::time::SystemTime::now(),
            alert_type: MemoryAlertType::MemoryLeak,
            severity: AlertSeverity::Error,
            message: "Potential memory leak detected".to_string(),
            details: HashMap::new(),
            recommendations: vec!["Check allocations".to_string()],
        };
        assert!(matches!(alert.alert_type, MemoryAlertType::MemoryLeak));
        assert!(matches!(alert.severity, AlertSeverity::Error));
    }

    #[test]
    fn test_memory_alert_types() {
        let types = vec![
            MemoryAlertType::HighMemoryUsage,
            MemoryAlertType::MemoryLeak,
            MemoryAlertType::RapidGrowth,
            MemoryAlertType::FragmentationHigh,
            MemoryAlertType::GcPressure,
            MemoryAlertType::OutOfMemory,
        ];
        assert_eq!(types.len(), 6);
    }

    #[test]
    fn test_alert_severity_levels() {
        let severities = vec![
            AlertSeverity::Info,
            AlertSeverity::Warning,
            AlertSeverity::Error,
            AlertSeverity::Critical,
        ];
        assert_eq!(severities.len(), 4);
    }
}
