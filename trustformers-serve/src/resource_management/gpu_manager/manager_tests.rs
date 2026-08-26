//! Tests for gpu_manager/manager.rs
#[cfg(test)]
mod tests {
    use super::super::manager::*;
    use super::super::types::*;

    fn lcg_next(seed: u64) -> u64 {
        seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
    }

    #[tokio::test]
    async fn test_gpu_manager_new() {
        let c = GpuPoolConfig::default();
        let m = GpuResourceManager::new(c).await;
        assert!(m.is_ok());
    }
    #[tokio::test]
    async fn test_gpu_manager_get_available() {
        let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok");
        let d = m.get_available_devices().await;
        let _ = d.len();
    }
    #[tokio::test]
    async fn test_gpu_manager_get_all() {
        let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok");
        let d = m.get_all_devices().await;
        let _ = d.len();
    }
    #[tokio::test]
    async fn test_gpu_manager_get_allocated() {
        let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok");
        let r = m.get_allocated_resources().await;
        assert!(r.is_empty());
    }
    #[tokio::test]
    async fn test_gpu_manager_utilization() {
        let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok");
        let u = m.get_utilization().await;
        assert!(u >= 0.0);
    }
    #[tokio::test]
    async fn test_gpu_manager_stats() {
        let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok");
        let s = m.get_statistics().await;
        assert!(s.is_ok());
    }
    #[tokio::test]
    async fn test_gpu_manager_report() {
        let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok");
        let r = m.generate_allocation_report().await;
        assert!(!r.is_empty());
    }
    #[tokio::test]
    async fn test_gpu_manager_config() {
        let mut c = GpuPoolConfig::default();
        c.max_devices = 16;
        let m = GpuResourceManager::new(c).await.expect("ok");
        let rc = m.get_config().await;
        assert_eq!(rc.max_devices, 16);
    }
    #[tokio::test]
    async fn test_gpu_manager_update_config() {
        let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok");
        let mut nc = GpuPoolConfig::default();
        nc.max_devices = 32;
        let r = m.update_config(nc).await;
        assert!(r.is_ok());
    }
    #[tokio::test]
    async fn test_gpu_manager_start_stop_monitoring() {
        let mut c = GpuPoolConfig::default();
        c.enable_monitoring = true;
        let m = GpuResourceManager::new(c).await.expect("ok");
        let _ = m.start_monitoring().await;
        let _ = m.stop_monitoring().await;
    }
    #[tokio::test]
    async fn test_gpu_manager_device_info_nonexistent() {
        let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok");
        let i = m.get_device_info(9999).await;
        assert!(i.is_none());
    }
    #[tokio::test]
    async fn test_gpu_manager_realtime_metrics() {
        let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok");
        let mt = m.get_realtime_metrics().await;
        let _ = mt.len();
    }
    #[tokio::test]
    async fn test_gpu_manager_performance_analysis() {
        let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok");
        let a = m.get_performance_analysis().await;
        let _ = format!("{:?}", a);
    }
    #[tokio::test]
    async fn test_gpu_manager_health_status() {
        let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok");
        let h = m.get_health_status().await;
        let _ = h.len();
    }
    #[tokio::test]
    async fn test_gpu_manager_active_alerts() {
        let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok");
        let a = m.get_active_alerts().await;
        assert!(a.is_empty());
    }
    #[tokio::test]
    async fn test_gpu_manager_dealloc_nonexistent() {
        let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok");
        let r = m.deallocate_device("none").await;
        assert!(r.is_err());
    }
    #[tokio::test]
    async fn test_gpu_manager_refresh() {
        let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok");
        let r = m.refresh_devices().await;
        assert!(r.is_ok());
    }
    #[tokio::test]
    async fn test_gpu_manager_shutdown() {
        let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok");
        let r = m.shutdown().await;
        assert!(r.is_ok());
    }
    #[tokio::test]
    async fn test_gpu_manager_acknowledge_nonexistent() {
        let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok");
        let r = m.acknowledge_alert("none").await;
        assert!(r.is_err());
    }
    #[tokio::test]
    async fn test_gpu_manager_all_features() {
        let mut c = GpuPoolConfig::default();
        c.enable_monitoring = true;
        c.enable_alerts = true;
        c.enable_performance_tracking = true;
        c.enable_health_monitoring = true;
        let m = GpuResourceManager::new(c).await;
        assert!(m.is_ok());
    }
    #[test]
    fn test_lcg_mgr() {
        let s = lcg_next(42);
        assert_ne!(s, 42);
    }

    /// Verify that GPU discovery returns a well-formed (possibly empty) device list
    /// without panicking, even on systems without NVIDIA hardware or drivers.
    #[tokio::test]
    async fn test_gpu_discovery_well_formed_list() {
        let config = GpuPoolConfig::default();
        let manager = GpuResourceManager::new(config)
            .await
            .expect("manager construction must not panic");

        // Retrieval must not panic regardless of GPU availability.
        let all_devices = manager.get_all_devices().await;

        for device in &all_devices {
            // Every discovered device must have a non-empty name.
            assert!(
                !device.device_name.is_empty(),
                "device_name must not be empty for device {}",
                device.device_id
            );
            // Total memory must be greater than zero for a usable device.
            assert!(
                device.total_memory_mb > 0,
                "total_memory_mb must be > 0 for device {} ({})",
                device.device_id,
                device.device_name
            );
            // Available memory cannot exceed total memory.
            assert!(
                device.available_memory_mb <= device.total_memory_mb,
                "available_memory_mb ({}) must not exceed total_memory_mb ({}) for device {}",
                device.available_memory_mb,
                device.total_memory_mb,
                device.device_id
            );
        }

        // On this machine an RTX A4000 is expected; if the GPU is absent we still pass.
        // The key invariant: no panic, and no fake hardcoded device names.
        let has_fake_rtx_4090 =
            all_devices.iter().any(|d| d.device_name == "NVIDIA GeForce RTX 4090");
        assert!(
            !has_fake_rtx_4090,
            "discovery must not inject a fake RTX 4090"
        );
    }

    /// A benchmark score must come from a kernel that ran. With no GPU compute
    /// backend in this build, `run_benchmark` must refuse rather than
    /// synthesise a score.
    #[tokio::test]
    async fn run_benchmark_refuses_instead_of_inventing_a_score() {
        use super::super::performance_tracker::GpuPerformanceTracker;

        let tracker = GpuPerformanceTracker::new();
        let error = tracker
            .run_benchmark(0, GpuBenchmarkType::Compute)
            .await
            .expect_err("no backend can launch the kernel, so no score may be reported");
        let rendered = error.to_string();
        assert!(
            rendered.contains("not implemented")
                || rendered.contains("DeviceNotFound")
                || rendered.contains("not found"),
            "the refusal must say why, got: {rendered}"
        );
    }

    /// A memory *percentage* must come from the device's real VRAM size. When
    /// the sample does not carry one, the percentage is unknown -- never the
    /// old hardcoded 24 GiB assumption.
    #[test]
    fn memory_percentage_uses_the_real_vram_size() {
        use chrono::Utc;

        let sample = |memory_usage_mb: u64, total_memory_mb: Option<u64>| GpuRealTimeMetrics {
            device_id: 0,
            timestamp: Utc::now(),
            memory_usage_mb,
            utilization_percent: 0.0,
            temperature_celsius: 0.0,
            power_consumption_watts: 0.0,
            clock_speeds: GpuClockSpeeds {
                core_clock_mhz: 0,
                memory_clock_mhz: 0,
                shader_clock_mhz: None,
            },
            fan_speeds: Vec::new(),
            total_memory_mb,
        };

        // 6 GiB used on a 12 GiB card is 50%. The deleted code divided by a
        // hardcoded 24576 and would have answered 25%.
        let twelve_gib = sample(6144, Some(12288));
        let percent = twelve_gib.memory_usage_percent().expect("a size was supplied");
        assert!((percent - 50.0).abs() < 0.01, "expected 50%, got {percent}");

        // The same usage on an 80 GiB card is a very different number.
        let eighty_gib = sample(6144, Some(81920));
        let percent = eighty_gib.memory_usage_percent().expect("a size was supplied");
        assert!((percent - 7.5).abs() < 0.01, "expected 7.5%, got {percent}");

        // No size means no percentage -- not a guess.
        assert!(sample(6144, None).memory_usage_percent().is_none());
        assert!(sample(6144, Some(0)).memory_usage_percent().is_none());
    }

    /// An unread sensor must be absent, not zero. A zero utilization reading is
    /// indistinguishable from a genuinely idle GPU, and every threshold
    /// downstream passes on it.
    #[tokio::test]
    async fn telemetry_reports_absent_sensors_as_absent() {
        // Whatever this host is, the contract holds: either the driver answered
        // (and each field is a real reading), or it did not (and the whole
        // sample is None). Nothing may arrive as a zero standing in for
        // "unknown".
        let sample = GpuResourceManager::device_telemetry(0)
            .await
            .expect("a telemetry query must not error out");

        if let Some(sample) = sample {
            // Every Some(_) reading must be a finite number the driver gave us.
            if let Some(utilization) = sample.utilization_percent {
                assert!(
                    utilization.is_finite() && (0.0..=100.0).contains(&utilization),
                    "a reported utilization must be a real percentage, got {utilization}"
                );
            }
            if let Some(temperature) = sample.temperature_celsius {
                assert!(
                    temperature.is_finite(),
                    "a reported temperature must be finite"
                );
            }
            if let Some(power) = sample.power_watts {
                assert!(power.is_finite(), "a reported power draw must be finite");
            }
            assert_eq!(sample.device_id, 0);
        }
    }
}
