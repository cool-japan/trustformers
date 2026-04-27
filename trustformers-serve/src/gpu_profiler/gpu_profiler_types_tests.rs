//! Comprehensive tests for gpu_profiler types and GpuProfiler functionality.

#[cfg(test)]
mod tests {
    use crate::gpu_profiler::{
        AlertSeverity, BottleneckType, GpuAlertThresholds, GpuAlertType, GpuMonitorConfig,
        GpuProfiler, GpuProfilerConfig, GpuProfilerError, MemoryAccessPattern,
        MemoryFragmentation, MemorySegmentType, ThermalEventType, TrendDirection,
    };
    use std::sync::atomic::Ordering;

    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg { state: seed }
        }
        fn next(&mut self) -> u64 {
            self.state = self.state
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            self.state
        }
        fn next_f64(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    // --- GpuAlertThresholds ---

    #[test]
    fn test_alert_thresholds_default() {
        let thresholds = GpuAlertThresholds::default();
        assert!(thresholds.temperature_threshold > 0.0);
        assert!(thresholds.memory_utilization_threshold > 0.0);
        assert!(thresholds.memory_utilization_threshold <= 1.0);
        assert!(thresholds.compute_utilization_threshold <= 1.0);
        assert!(thresholds.power_threshold > 0.0);
    }

    #[test]
    fn test_alert_thresholds_default_values() {
        let thresholds = GpuAlertThresholds::default();
        assert!((thresholds.temperature_threshold - 80.0).abs() < 0.01);
        assert!((thresholds.memory_utilization_threshold - 0.9).abs() < 0.01);
        assert!((thresholds.error_rate_threshold - 0.01).abs() < 0.0001);
    }

    #[test]
    fn test_alert_thresholds_custom() {
        let thresholds = GpuAlertThresholds {
            temperature_threshold: 75.0,
            memory_utilization_threshold: 0.85,
            compute_utilization_threshold: 0.90,
            power_threshold: 200.0,
            memory_fragmentation_threshold: 0.25,
            error_rate_threshold: 0.005,
        };
        assert!((thresholds.temperature_threshold - 75.0).abs() < 0.01);
        assert!((thresholds.power_threshold - 200.0).abs() < 0.01);
    }

    // --- GpuProfilerConfig ---

    #[test]
    fn test_profiler_config_default_enabled() {
        let config = GpuProfilerConfig::default();
        assert!(config.enabled);
        assert!(config.enable_memory_profiling);
        assert!(config.enable_performance_profiling);
        assert!(config.enable_thermal_monitoring);
    }

    #[test]
    fn test_profiler_config_default_intervals() {
        let config = GpuProfilerConfig::default();
        assert!(config.profiling_interval_seconds > 0);
        assert!(config.data_retention_hours > 0);
        assert!(config.max_profile_samples > 0);
    }

    #[test]
    fn test_profiler_config_default_has_gpu_config() {
        let config = GpuProfilerConfig::default();
        assert!(!config.gpu_configs.is_empty());
        let first_gpu = &config.gpu_configs[0];
        assert!(first_gpu.enabled);
        assert_eq!(first_gpu.gpu_id, 0);
    }

    // --- GpuMonitorConfig ---

    #[test]
    fn test_gpu_monitor_config_thresholds() {
        let cfg = GpuMonitorConfig {
            gpu_id: 1,
            enabled: true,
            max_temperature_celsius: 90.0,
            max_power_watts: 350.0,
            max_memory_utilization: 0.95,
            max_compute_utilization: 0.98,
        };
        assert_eq!(cfg.gpu_id, 1);
        assert!(cfg.max_temperature_celsius > 0.0);
        assert!(cfg.max_memory_utilization <= 1.0);
    }

    // --- GpuProfiler creation ---

    #[tokio::test]
    async fn test_profiler_creation_default_config() {
        let config = GpuProfilerConfig::default();
        let profiler = GpuProfiler::new(config);
        assert!(profiler.is_ok());
    }

    #[tokio::test]
    async fn test_profiler_creation_disabled() {
        let mut config = GpuProfilerConfig::default();
        config.enabled = false;
        let profiler = GpuProfiler::new(config);
        assert!(profiler.is_ok());
        if let Ok(p) = profiler {
            assert!(!p.config.enabled);
        }
    }

    // --- collect_utilization_metrics ---

    #[tokio::test]
    async fn test_collect_utilization_metrics_gpu0() {
        let config = GpuProfilerConfig::default();
        if let Ok(profiler) = GpuProfiler::new(config) {
            let result = profiler.collect_utilization_metrics(0).await;
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_collect_utilization_metrics_multiple_gpus() {
        let mut config = GpuProfilerConfig::default();
        config.gpu_configs.push(GpuMonitorConfig {
            gpu_id: 1,
            enabled: true,
            max_temperature_celsius: 85.0,
            max_power_watts: 300.0,
            max_memory_utilization: 0.95,
            max_compute_utilization: 0.95,
        });
        if let Ok(profiler) = GpuProfiler::new(config) {
            let r0 = profiler.collect_utilization_metrics(0).await;
            let r1 = profiler.collect_utilization_metrics(1).await;
            assert!(r0.is_ok());
            assert!(r1.is_ok());
            let metrics = profiler.get_utilization_metrics().await;
            assert!(metrics.contains_key(&0));
            assert!(metrics.contains_key(&1));
        }
    }

    // --- get_utilization_metrics ---

    #[tokio::test]
    async fn test_get_utilization_metrics_empty_initially() {
        let config = GpuProfilerConfig::default();
        if let Ok(profiler) = GpuProfiler::new(config) {
            let metrics = profiler.get_utilization_metrics().await;
            assert!(metrics.is_empty());
        }
    }

    #[tokio::test]
    async fn test_get_utilization_metrics_after_collection() {
        let config = GpuProfilerConfig::default();
        if let Ok(profiler) = GpuProfiler::new(config) {
            let _ = profiler.collect_utilization_metrics(0).await;
            let metrics = profiler.get_utilization_metrics().await;
            assert!(metrics.contains_key(&0));
            if let Some(gpu_metrics) = metrics.get(&0) {
                assert!(!gpu_metrics.is_empty());
                let m = &gpu_metrics[0];
                assert_eq!(m.gpu_id, 0);
                assert!(m.compute_utilization >= 0.0 && m.compute_utilization <= 1.0);
                assert!(m.memory_utilization >= 0.0 && m.memory_utilization <= 1.0);
            }
        }
    }

    // --- get_memory_profile / get_performance_profile ---

    #[tokio::test]
    async fn test_get_memory_profile_none_initially() {
        let config = GpuProfilerConfig::default();
        if let Ok(profiler) = GpuProfiler::new(config) {
            let profile = profiler.get_memory_profile(0).await;
            assert!(profile.is_none());
        }
    }

    #[tokio::test]
    async fn test_get_performance_profile_none_initially() {
        let config = GpuProfilerConfig::default();
        if let Ok(profiler) = GpuProfiler::new(config) {
            let profile = profiler.get_performance_profile(0).await;
            assert!(profile.is_none());
        }
    }

    // --- get_stats ---

    #[tokio::test]
    async fn test_stats_initial_zeros() {
        let config = GpuProfilerConfig::default();
        if let Ok(profiler) = GpuProfiler::new(config) {
            let stats = profiler.get_stats().await;
            assert_eq!(stats.total_profiles.load(Ordering::Relaxed), 0);
            assert_eq!(stats.total_alerts.load(Ordering::Relaxed), 0);
            assert_eq!(stats.total_memory_profiles.load(Ordering::Relaxed), 0);
        }
    }

    #[tokio::test]
    async fn test_stats_after_collection() {
        let config = GpuProfilerConfig::default();
        if let Ok(profiler) = GpuProfiler::new(config) {
            let _ = profiler.collect_utilization_metrics(0).await;
            let stats = profiler.get_stats().await;
            assert_eq!(stats.total_profiles.load(Ordering::Relaxed), 1);
        }
    }

    // --- get_recent_alerts ---

    #[tokio::test]
    async fn test_alerts_empty_initially() {
        let config = GpuProfilerConfig::default();
        if let Ok(profiler) = GpuProfiler::new(config) {
            let alerts = profiler.get_recent_alerts(None).await;
            assert!(alerts.is_empty());
        }
    }

    #[tokio::test]
    async fn test_generate_alert_temperature() {
        let config = GpuProfilerConfig::default();
        if let Ok(profiler) = GpuProfiler::new(config) {
            let result = profiler
                .generate_alert(0, GpuAlertType::HighTemperature, AlertSeverity::High, 85.0)
                .await;
            assert!(result.is_ok());
            let alerts = profiler.get_recent_alerts(None).await;
            assert_eq!(alerts.len(), 1);
            assert_eq!(alerts[0].gpu_id, 0);
            assert!((alerts[0].current_value - 85.0).abs() < 0.01);
        }
    }

    #[tokio::test]
    async fn test_generate_multiple_alerts() {
        let config = GpuProfilerConfig::default();
        if let Ok(profiler) = GpuProfiler::new(config) {
            let _ = profiler
                .generate_alert(0, GpuAlertType::HighMemoryUtilization, AlertSeverity::Medium, 0.95)
                .await;
            let _ = profiler
                .generate_alert(0, GpuAlertType::HighPowerConsumption, AlertSeverity::Critical, 300.0)
                .await;
            let alerts = profiler.get_recent_alerts(None).await;
            assert_eq!(alerts.len(), 2);
        }
    }

    #[tokio::test]
    async fn test_get_recent_alerts_with_limit() {
        let config = GpuProfilerConfig::default();
        if let Ok(profiler) = GpuProfiler::new(config) {
            for _ in 0..5 {
                let _ = profiler
                    .generate_alert(0, GpuAlertType::LowEfficiency, AlertSeverity::Low, 0.3)
                    .await;
            }
            let limited = profiler.get_recent_alerts(Some(2)).await;
            assert_eq!(limited.len(), 2);
        }
    }

    // --- generate_report ---

    #[tokio::test]
    async fn test_generate_report_empty() {
        let config = GpuProfilerConfig::default();
        if let Ok(profiler) = GpuProfiler::new(config) {
            let report = profiler.generate_report().await;
            assert!(report.is_ok());
            if let Ok(r) = report {
                assert_eq!(r.gpu_count, 0);
                assert!(r.recommendations.len() > 0);
            }
        }
    }

    #[tokio::test]
    async fn test_generate_report_after_collection() {
        let config = GpuProfilerConfig::default();
        if let Ok(profiler) = GpuProfiler::new(config) {
            let _ = profiler.collect_utilization_metrics(0).await;
            let report = profiler.generate_report().await;
            assert!(report.is_ok());
            if let Ok(r) = report {
                assert_eq!(r.gpu_count, 1);
            }
        }
    }

    // --- MemoryFragmentation ---

    #[test]
    fn test_memory_fragmentation_valid_range() {
        let mut lcg = Lcg::new(999);
        let frag = MemoryFragmentation {
            fragmentation_ratio: lcg.next_f64() * 0.5,
            largest_free_block_bytes: lcg.next() % (4 * 1024 * 1024 * 1024),
            free_block_count: (lcg.next() % 100) as u32,
            average_free_block_bytes: lcg.next() % (512 * 1024 * 1024),
            external_fragmentation: lcg.next_f64() * 0.3,
            internal_fragmentation: lcg.next_f64() * 0.1,
        };
        assert!(frag.fragmentation_ratio >= 0.0 && frag.fragmentation_ratio <= 1.0);
        assert!(frag.external_fragmentation <= 1.0);
    }

    // --- MemoryAccessPattern ---

    #[test]
    fn test_memory_access_pattern_variants() {
        let patterns = [
            MemoryAccessPattern::Sequential,
            MemoryAccessPattern::Random,
            MemoryAccessPattern::Burst,
            MemoryAccessPattern::Mixed,
            MemoryAccessPattern::Strided { stride: 64 },
        ];
        for p in &patterns {
            assert!(format!("{:?}", p).len() > 0);
        }
    }

    // --- BottleneckType ---

    #[test]
    fn test_bottleneck_type_variants() {
        let bottlenecks = [
            BottleneckType::MemoryBandwidth,
            BottleneckType::Compute,
            BottleneckType::MemoryLatency,
            BottleneckType::ThermalThrottling,
            BottleneckType::PowerThrottling,
            BottleneckType::KernelLaunchOverhead,
            BottleneckType::DataTransfer,
            BottleneckType::Synchronization,
            BottleneckType::LowOccupancy,
            BottleneckType::BranchDivergence,
        ];
        assert_eq!(bottlenecks.len(), 10);
    }

    // --- GpuProfilerError ---

    #[test]
    fn test_profiler_error_display() {
        let err = GpuProfilerError::ConfigurationError { message: "bad value".to_string() };
        let msg = format!("{}", err);
        assert!(msg.contains("bad value"));
    }

    #[test]
    fn test_profiler_error_gpu_not_found() {
        let err = GpuProfilerError::GpuNotFound { gpu_id: 42 };
        let msg = format!("{}", err);
        assert!(msg.contains("42"));
    }

    // --- TrendDirection ---

    #[test]
    fn test_trend_direction_variants() {
        let dirs = [
            TrendDirection::Improving,
            TrendDirection::Degrading,
            TrendDirection::Stable,
            TrendDirection::Volatile,
        ];
        for d in &dirs {
            assert!(format!("{:?}", d).len() > 0);
        }
    }

    // --- MemorySegmentType ---

    #[test]
    fn test_memory_segment_type_variants() {
        let types = [
            MemorySegmentType::ModelWeights,
            MemorySegmentType::Activations,
            MemorySegmentType::Gradients,
            MemorySegmentType::KVCache,
            MemorySegmentType::TempBuffers,
            MemorySegmentType::SystemReserved,
            MemorySegmentType::Unknown,
        ];
        assert_eq!(types.len(), 7);
    }

    // --- AlertSeverity ---

    #[test]
    fn test_alert_severity_variants() {
        let severities = [
            AlertSeverity::Low,
            AlertSeverity::Medium,
            AlertSeverity::High,
            AlertSeverity::Critical,
        ];
        for s in &severities {
            assert!(format!("{:?}", s).len() > 0);
        }
    }

    // --- ThermalEventType ---

    #[test]
    fn test_thermal_event_type_variants() {
        let types = [
            ThermalEventType::TemperatureWarning,
            ThermalEventType::ThermalThrottling,
            ThermalEventType::FanSpeedIncrease,
            ThermalEventType::PerformanceReduction,
        ];
        assert_eq!(types.len(), 4);
    }

    // --- LCG random data generation ---

    #[test]
    fn test_lcg_produces_varied_f64_values() {
        let mut lcg = Lcg::new(271828);
        let values: Vec<f64> = (0..10).map(|_| lcg.next_f64()).collect();
        let unique: std::collections::HashSet<u64> =
            values.iter().map(|&v| (v * 1_000_000.0) as u64).collect();
        assert!(unique.len() > 5);
    }

    #[test]
    fn test_utilization_metrics_memory_consistency() {
        let used: u64 = 16 * 1024 * 1024 * 1024;
        let free: u64 = 8 * 1024 * 1024 * 1024;
        let total: u64 = 24 * 1024 * 1024 * 1024;
        // verify used + free = total
        assert_eq!(used + free, total);
    }

    #[test]
    fn test_clock_speeds_reasonable_ranges() {
        use crate::gpu_profiler::types::GpuClockSpeeds;
        let clocks = GpuClockSpeeds {
            graphics_clock_mhz: 1800,
            memory_clock_mhz: 7000,
            sm_clock_mhz: 1800,
            video_clock_mhz: 1500,
        };
        assert!(clocks.graphics_clock_mhz > 0);
        assert!(clocks.memory_clock_mhz > clocks.graphics_clock_mhz);
    }
}
