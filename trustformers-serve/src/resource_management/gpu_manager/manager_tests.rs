//! Tests for gpu_manager/manager.rs
#[cfg(test)]
mod tests {
    use super::super::manager::*;
    use super::super::types::*;

    fn lcg_next(seed: u64) -> u64 { seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) }

    #[tokio::test] async fn test_gpu_manager_new() { let c = GpuPoolConfig::default(); let m = GpuResourceManager::new(c).await; assert!(m.is_ok()); }
    #[tokio::test] async fn test_gpu_manager_get_available() { let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok"); let d = m.get_available_devices().await; let _ = d.len(); }
    #[tokio::test] async fn test_gpu_manager_get_all() { let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok"); let d = m.get_all_devices().await; let _ = d.len(); }
    #[tokio::test] async fn test_gpu_manager_get_allocated() { let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok"); let r = m.get_allocated_resources().await; assert!(r.is_empty()); }
    #[tokio::test] async fn test_gpu_manager_utilization() { let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok"); let u = m.get_utilization().await; assert!(u >= 0.0); }
    #[tokio::test] async fn test_gpu_manager_stats() { let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok"); let s = m.get_statistics().await; assert!(s.is_ok()); }
    #[tokio::test] async fn test_gpu_manager_report() { let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok"); let r = m.generate_allocation_report().await; assert!(!r.is_empty()); }
    #[tokio::test] async fn test_gpu_manager_config() { let mut c = GpuPoolConfig::default(); c.max_devices = 16; let m = GpuResourceManager::new(c).await.expect("ok"); let rc = m.get_config().await; assert_eq!(rc.max_devices, 16); }
    #[tokio::test] async fn test_gpu_manager_update_config() { let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok"); let mut nc = GpuPoolConfig::default(); nc.max_devices = 32; let r = m.update_config(nc).await; assert!(r.is_ok()); }
    #[tokio::test] async fn test_gpu_manager_start_stop_monitoring() { let mut c = GpuPoolConfig::default(); c.enable_monitoring = true; let m = GpuResourceManager::new(c).await.expect("ok"); let _ = m.start_monitoring().await; let _ = m.stop_monitoring().await; }
    #[tokio::test] async fn test_gpu_manager_device_info_nonexistent() { let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok"); let i = m.get_device_info(9999).await; assert!(i.is_none()); }
    #[tokio::test] async fn test_gpu_manager_realtime_metrics() { let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok"); let mt = m.get_realtime_metrics().await; let _ = mt.len(); }
    #[tokio::test] async fn test_gpu_manager_performance_analysis() { let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok"); let a = m.get_performance_analysis().await; let _ = format!("{:?}", a); }
    #[tokio::test] async fn test_gpu_manager_health_status() { let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok"); let h = m.get_health_status().await; let _ = h.len(); }
    #[tokio::test] async fn test_gpu_manager_active_alerts() { let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok"); let a = m.get_active_alerts().await; assert!(a.is_empty()); }
    #[tokio::test] async fn test_gpu_manager_dealloc_nonexistent() { let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok"); let r = m.deallocate_device("none").await; assert!(r.is_err()); }
    #[tokio::test] async fn test_gpu_manager_refresh() { let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok"); let r = m.refresh_devices().await; assert!(r.is_ok()); }
    #[tokio::test] async fn test_gpu_manager_shutdown() { let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok"); let r = m.shutdown().await; assert!(r.is_ok()); }
    #[tokio::test] async fn test_gpu_manager_acknowledge_nonexistent() { let m = GpuResourceManager::new(GpuPoolConfig::default()).await.expect("ok"); let r = m.acknowledge_alert("none").await; assert!(r.is_err()); }
    #[tokio::test] async fn test_gpu_manager_all_features() { let mut c = GpuPoolConfig::default(); c.enable_monitoring = true; c.enable_alerts = true; c.enable_performance_tracking = true; c.enable_health_monitoring = true; let m = GpuResourceManager::new(c).await; assert!(m.is_ok()); }
    #[test] fn test_lcg_mgr() { let s = lcg_next(42); assert_ne!(s, 42); }
}
