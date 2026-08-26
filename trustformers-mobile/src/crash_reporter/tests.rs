//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use std::time::Duration;

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;

    #[test]
    fn test_crash_reporter_creation() {
        let config = CrashReporterConfig::default();
        let reporter = MobileCrashReporter::new(config);
        assert!(reporter.is_ok());
    }

    #[test]
    fn test_crash_severity_assessment() {
        let config = CrashReporterConfig::default();
        let reporter = MobileCrashReporter::new(config).expect("Operation failed");

        let crash_info = CrashInfo {
            crash_type: CrashType::SegmentationFault,
            stack_trace: None,
            memory_dump: None,
            context: CrashContext {
                current_operation: None,
                user_actions: Vec::new(),
                system_events: Vec::new(),
                model_anomalies: Vec::new(),
                performance_bottlenecks: Vec::new(),
                error_logs: Vec::new(),
            },
        };

        let severity = reporter.assess_crash_severity(&crash_info);
        assert_eq!(severity, CrashSeverity::Critical);
    }

    #[test]
    fn test_privacy_compliance() {
        let mut config = CrashReporterConfig::default();
        config.privacy_config.include_user_data = false;
        config.privacy_config.anonymize_data = true;

        let reporter = MobileCrashReporter::new(config).expect("Operation failed");

        let crash_info = CrashInfo {
            crash_type: CrashType::OutOfMemory,
            stack_trace: None,
            memory_dump: None,
            context: CrashContext {
                current_operation: None,
                user_actions: Vec::new(),
                system_events: Vec::new(),
                model_anomalies: Vec::new(),
                performance_bottlenecks: Vec::new(),
                error_logs: Vec::new(),
            },
        };

        assert!(reporter.is_privacy_compliant(&crash_info));
    }

    #[test]
    fn test_recovery_suggestions() {
        let config = CrashReporterConfig::default();
        let reporter = MobileCrashReporter::new(config).expect("Operation failed");

        let crash_report = CrashReport {
            report_id: "test".to_string(),
            timestamp: 0,
            crash_type: CrashType::OutOfMemory,
            severity: CrashSeverity::High,
            system_info: SystemCrashInfo {
                device_info: crate::device_info::MobileDeviceDetector::detect()
                    .expect("Operation failed"),
                performance_metrics: None,
                memory_usage: MemoryUsageInfo {
                    total_mb: 1024.0,
                    used_mb: 1000.0,
                    available_mb: 24.0,
                    heap_mb: 800.0,
                    stack_mb: 50.0,
                    gpu_mb: None,
                },
                cpu_info: CpuCrashInfo {
                    usage_percent: 90.0,
                    frequency_mhz: 2000,
                    temperature_c: Some(80.0),
                    throttling: true,
                    active_cores: 4,
                },
                gpu_info: None,
                thermal_state: None,
                battery_info: None,
                network_info: None,
            },
            app_info: AppCrashInfo {
                app_version: "1.0.0".to_string(),
                build_number: "1".to_string(),
                framework_version: "1.0.0".to_string(),
                app_state: AppState::Active,
                foreground_status: ForegroundStatus::Foreground,
                session_duration: Duration::from_secs(300),
                model_info: None,
                recent_operations: Vec::new(),
            },
            stack_trace: None,
            memory_dump: None,
            context: CrashContext {
                current_operation: None,
                user_actions: Vec::new(),
                system_events: Vec::new(),
                model_anomalies: Vec::new(),
                performance_bottlenecks: Vec::new(),
                error_logs: Vec::new(),
            },
            analysis: None,
            recovery_suggestions: Vec::new(),
            is_privacy_compliant: true,
        };

        let suggestions =
            reporter.generate_recovery_suggestions(&crash_report).expect("Operation failed");
        assert!(!suggestions.is_empty());
        assert_eq!(suggestions[0].suggestion_type, RecoveryStrategy::ClearCache);
    }

    /// Regression test for the previous `collect_memory_usage`, which
    /// returned every field as a hardcoded `0.0` regardless of actual
    /// memory pressure. A process that is definitely resident (this test
    /// process) must report a nonzero total and nonzero heap (RSS) figure.
    #[test]
    fn test_collect_memory_usage_reports_real_nonzero_figures() {
        let config = CrashReporterConfig::default();
        let reporter = MobileCrashReporter::new(config).expect("reporter creation failed");

        let memory = reporter.collect_memory_usage().expect("memory collection failed");
        assert!(
            memory.total_mb > 0.0,
            "total_mb must be a real measured figure"
        );
        assert!(
            memory.heap_mb > 0.0,
            "heap_mb (this process's RSS) must be nonzero"
        );
    }

    /// Regression test for the previous `collect_cpu_info`, which returned
    /// the hardcoded constants `usage_percent: 0.0`, `frequency_mhz: 0`, and
    /// `active_cores: 1` on every device regardless of actual hardware.
    #[test]
    fn test_collect_cpu_info_reports_real_core_count() {
        let config = CrashReporterConfig::default();
        let reporter = MobileCrashReporter::new(config).expect("reporter creation failed");

        let cpu = reporter.collect_cpu_info().expect("cpu collection failed");
        // `usage_percent` legitimately can be 0.0 on an idle core, so it is
        // not a safe inequality check; `active_cores` reflecting the same
        // real detected core count `sysinfo` itself reports (rather than
        // the old hardcoded `1`) is the property this test can assert
        // without flakiness on any host.
        let mut system = sysinfo::System::new();
        system.refresh_cpu_usage();
        let expected_cores = system.cpus().len().max(1);
        assert_eq!(cpu.active_cores, expected_cores);
        assert!(cpu.usage_percent >= 0.0 && cpu.usage_percent.is_finite());
    }
}
