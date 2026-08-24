//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use crate::device_info::{MobileDeviceInfo, PerformanceTier};
use serde_json::json;
use std::time::Instant;

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    use crate::device_info::{BasicDeviceInfo, CpuInfo, MemoryInfo, PerformanceScores};

    fn create_test_device_info() -> MobileDeviceInfo {
        MobileDeviceInfo {
            platform: crate::MobilePlatform::Generic,
            basic_info: BasicDeviceInfo {
                platform: crate::MobilePlatform::Generic,
                manufacturer: "Test".to_string(),
                model: "TestDevice".to_string(),
                os_version: "1.0".to_string(),
                hardware_id: "test123".to_string(),
                device_generation: Some(2023),
            },
            cpu_info: CpuInfo {
                architecture: "arm64".to_string(),
                total_cores: 8,
                core_count: 8,
                performance_cores: 4,
                efficiency_cores: 4,
                max_frequency_mhz: Some(3000),
                l1_cache_kb: Some(64),
                l2_cache_kb: Some(512),
                l3_cache_kb: Some(8192),
                features: vec!["NEON".to_string()],
                simd_support: crate::device_info::SimdSupport::Advanced,
            },
            memory_info: MemoryInfo {
                total_mb: 4096,
                available_mb: 2048,
                total_memory: 4096,
                available_memory: 2048,
                bandwidth_mbps: Some(25600),
                memory_type: "LPDDR5".to_string(),
                frequency_mhz: Some(6400),
                is_low_memory_device: false,
            },
            gpu_info: None,
            npu_info: None,
            thermal_info: crate::device_info::ThermalInfo {
                current_state: crate::device_info::ThermalState::Nominal,
                state: crate::device_info::ThermalState::Nominal,
                throttling_supported: true,
                temperature_sensors: vec![],
                thermal_zones: vec![],
            },
            power_info: crate::device_info::PowerInfo {
                battery_capacity_mah: Some(3000),
                battery_level_percent: Some(75),
                battery_level: Some(75),
                battery_health_percent: Some(95),
                charging_status: crate::device_info::ChargingStatus::NotCharging,
                is_charging: false,
                power_save_mode: false,
                low_power_mode_available: true,
            },
            available_backends: vec![crate::MobileBackend::CPU],
            performance_scores: PerformanceScores {
                cpu_single_core: Some(1200),
                cpu_multi_core: Some(8500),
                gpu_score: None,
                memory_score: Some(9200),
                overall_tier: PerformanceTier::High,
                tier: PerformanceTier::High,
            },
        }
    }

    #[test]
    fn test_profiler_creation() {
        let device_info = create_test_device_info();
        let config = ProfilerConfig::default();

        let profiler = MobilePerformanceProfiler::new(config, &device_info);
        assert!(profiler.is_ok());
    }

    #[test]
    fn test_profiler_config_defaults() {
        let config = ProfilerConfig::default();
        assert!(config.enable_realtime_profiling);
        assert_eq!(config.profiling_interval_ms, 1000);
        assert!(config.enable_platform_integration);
        assert_eq!(config.max_history_size, 1000);
    }

    #[test]
    fn test_metrics_config_defaults() {
        let config = MetricsConfig::default();
        assert!(config.collect_cpu);
        assert!(config.collect_gpu);
        assert!(config.collect_memory);
        assert!(config.collect_network);
        assert!(config.collect_inference);
        assert_eq!(config.sampling_rate_hz, 10);
    }

    #[test]
    fn test_bottleneck_config_defaults() {
        let config = BottleneckConfig::default();
        assert!(config.detect_cpu_bottlenecks);
        assert!(config.detect_memory_bottlenecks);
        assert!(config.detect_io_bottlenecks);
        assert!(config.detect_thermal_bottlenecks);
        assert_eq!(config.cpu_threshold_percent, 80.0);
        assert_eq!(config.memory_threshold_percent, 85.0);
    }

    #[test]
    fn test_alert_thresholds_defaults() {
        let thresholds = AlertThresholds::default();
        assert_eq!(thresholds.cpu_threshold_percent, 90.0);
        assert_eq!(thresholds.memory_threshold_percent, 90.0);
        assert_eq!(thresholds.latency_threshold_ms, 500.0);
        assert_eq!(thresholds.temperature_threshold_celsius, 85.0);
        assert_eq!(thresholds.battery_threshold_percent, 20);
        assert_eq!(thresholds.power_threshold_mw, 5000.0);
    }

    #[test]
    fn test_performance_score_calculation() {
        let device_info = create_test_device_info();
        let config = ProfilerConfig::default();
        let profiler =
            MobilePerformanceProfiler::new(config, &device_info).expect("Operation failed");

        let metrics = MetricsSnapshot {
            timestamp: Instant::now(),
            platform_metrics: PlatformMetrics::default(),
            inference_metrics: InferenceMetrics::default(),
            thermal_metrics: None,
            battery_metrics: None,
        };

        let bottlenecks = vec![];
        let score = profiler.calculate_performance_score(&metrics, &bottlenecks);
        assert!((0.0..=100.0).contains(&score));
    }

    #[test]
    fn test_optimized_config_generation() {
        let device_info = create_test_device_info();
        let config = MobileProfilerUtils::create_optimized_config(&device_info);

        // Should be optimized for high-performance device
        assert_eq!(config.profiling_interval_ms, 1000);
        assert_eq!(config.metrics_config.sampling_rate_hz, 10);
        assert_eq!(config.max_history_size, 1000);
        assert!(config.metrics_config.detailed_collection);
    }

    #[test]
    fn test_efficiency_score_calculation() {
        let metrics = MetricsSnapshot {
            timestamp: Instant::now(),
            platform_metrics: PlatformMetrics {
                cpu_metrics: CpuMetrics {
                    utilization_percent: 50.0,
                    ..Default::default()
                },
                memory_metrics: MemoryMetrics {
                    pressure_level: MemoryPressureLevel::Low,
                    ..Default::default()
                },
                ..Default::default()
            },
            inference_metrics: InferenceMetrics {
                latency_ms: 100.0,
                ..Default::default()
            },
            thermal_metrics: None,
            battery_metrics: None,
        };

        let score = MobileProfilerUtils::calculate_efficiency_score(&metrics);
        assert!((0.0..=100.0).contains(&score));
    }

    #[test]
    fn test_platform_profiler_capabilities() {
        let ios_profiler = IOSProfiler::new().expect("Operation failed");
        let capabilities = ios_profiler.get_capabilities();
        assert!(capabilities.contains(&ProfilerCapability::CpuProfiling));
        assert!(capabilities.contains(&ProfilerCapability::InstrumentsIntegration));

        let android_profiler = AndroidProfiler::new().expect("Operation failed");
        let capabilities = android_profiler.get_capabilities();
        assert!(capabilities.contains(&ProfilerCapability::CpuProfiling));
        assert!(capabilities.contains(&ProfilerCapability::SystraceIntegration));
    }

    #[test]
    fn test_bottleneck_severity_calculation() {
        let config = BottleneckConfig::default();
        let detector = BottleneckDetector::new(config);

        let severity = detector.calculate_bottleneck_severity(120.0, 80.0);
        assert_eq!(severity, BottleneckSeverity::High);

        let severity = detector.calculate_bottleneck_severity(160.0, 80.0);
        assert_eq!(severity, BottleneckSeverity::Critical);
    }

    #[test]
    fn test_memory_pressure_levels() {
        assert!(MemoryPressureLevel::Critical > MemoryPressureLevel::High);
        assert!(MemoryPressureLevel::High > MemoryPressureLevel::Medium);
        assert!(MemoryPressureLevel::Medium > MemoryPressureLevel::Low);
    }

    #[test]
    fn test_export_format_serialization() {
        let format = ExportFormat::JSON;
        let serialized = serde_json::to_string(&format).expect("Operation failed");
        let deserialized: ExportFormat =
            serde_json::from_str(&serialized).expect("Operation failed");
        assert_eq!(format, deserialized);
    }

    /// Regression test for the previous `collect_metrics`, which was
    /// `Ok(PlatformMetrics::default())` for every platform profiler --
    /// indistinguishable from "profiling collected literally nothing"
    /// (`per_core_utilization: vec![]`, `frequency_mhz: vec![]`) regardless
    /// of whether profiling had even started. Real `sysinfo`-backed
    /// collection must report at least one CPU core on any host this
    /// workspace actually builds and tests on.
    #[test]
    fn test_collect_metrics_reports_real_per_core_data_not_empty_default() {
        for metrics in [
            IOSProfiler::new()
                .expect("IOSProfiler::new")
                .collect_metrics()
                .expect("collect"),
            AndroidProfiler::new()
                .expect("AndroidProfiler::new")
                .collect_metrics()
                .expect("collect"),
            GenericProfiler::new()
                .expect("GenericProfiler::new")
                .collect_metrics()
                .expect("collect"),
        ] {
            assert!(
                !metrics.cpu_metrics.per_core_utilization.is_empty(),
                "must report real per-core data, not the empty `PlatformMetrics::default()` \
                 vector every profiler used to return unconditionally"
            );
            assert!(!metrics.cpu_metrics.frequency_mhz.is_empty());
            assert_eq!(
                metrics.cpu_metrics.per_core_utilization.len(),
                metrics.cpu_metrics.frequency_mhz.len()
            );
        }
    }

    /// Regression test for the previous `check_instruments_availability` /
    /// `check_systrace_availability` / `check_perfetto_availability`, which
    /// each returned a hardcoded `true` under their respective
    /// `#[cfg(target_os = ...)]` -- a fabricated "yes, this vendor tool is
    /// attached" signal with no actual check behind it. None of this crate
    /// has a real hook to verify vendor-tool attachment, so all three must
    /// now honestly report `false`.
    #[test]
    fn test_vendor_tool_availability_checks_are_honest_not_fabricated() {
        let ios = IOSProfiler::new().expect("IOSProfiler::new");
        assert!(
            !ios.instruments_integration,
            "no real Instruments attachment check exists"
        );

        let android = AndroidProfiler::new().expect("AndroidProfiler::new");
        assert!(
            !android.systrace_integration,
            "no real systrace attachment check exists"
        );
        assert!(
            !android.perfetto_integration,
            "no real Perfetto attachment check exists"
        );
    }

    /// Regression test for `export_data`, which previously returned
    /// `Ok(vec![])` -- indistinguishable from "successfully exported an
    /// empty trace" -- for `Instruments`/`Trace`/`Perfetto`/`CSV`, formats
    /// this crate cannot actually produce (or, for `CSV`, simply never
    /// populated). `JSON` must now be a real serialization of real
    /// collected metrics, and the unimplemented binary vendor formats must
    /// error rather than silently succeed with nothing.
    #[test]
    fn test_export_data_is_real_json_or_an_honest_error_never_fake_empty_success() {
        let ios = IOSProfiler::new().expect("IOSProfiler::new");
        let json = ios.export_data(ExportFormat::JSON).expect("JSON export should succeed");
        assert!(!json.is_empty());
        let parsed: PlatformMetrics =
            serde_json::from_slice(&json).expect("exported JSON must round-trip");
        assert!(!parsed.cpu_metrics.per_core_utilization.is_empty());
        assert!(
            ios.export_data(ExportFormat::Instruments).is_err(),
            "must not fake-succeed at a format this crate cannot actually write"
        );

        let android = AndroidProfiler::new().expect("AndroidProfiler::new");
        assert!(android.export_data(ExportFormat::JSON).is_ok());
        assert!(android.export_data(ExportFormat::Trace).is_err());
        assert!(android.export_data(ExportFormat::Perfetto).is_err());

        let generic = GenericProfiler::new().expect("GenericProfiler::new");
        assert!(generic.export_data(ExportFormat::JSON).is_ok());
        let csv = generic.export_data(ExportFormat::CSV).expect("CSV export should succeed");
        let csv_text = String::from_utf8(csv).expect("CSV must be valid UTF-8");
        assert!(csv_text.starts_with("cpu_utilization_percent,"));
        assert_eq!(
            csv_text.lines().count(),
            2,
            "a header row plus exactly one data row"
        );
    }
}
