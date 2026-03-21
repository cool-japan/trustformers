//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;
use serde::Serializer;
use std::sync::atomic::{AtomicU64, Ordering};

pub fn serialize_atomic_u64<S>(value: &AtomicU64, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_u64(value.load(Ordering::Relaxed))
}
#[cfg(test)]
mod tests {

    use crate::gpu_profiler::AlertSeverity;
    use crate::{GpuAlertType, GpuProfiler, GpuProfilerConfig};
    #[tokio::test]
    async fn test_gpu_profiler_creation() {
        let config = GpuProfilerConfig::default();
        let profiler = GpuProfiler::new(config).expect("test operation should succeed");
        assert!(profiler.config.enabled);
    }
    #[tokio::test]
    async fn test_utilization_collection() {
        let config = GpuProfilerConfig::default();
        let profiler = GpuProfiler::new(config).expect("test operation should succeed");
        let result = profiler.collect_utilization_metrics(0).await;
        assert!(result.is_ok());
        let metrics = profiler.get_utilization_metrics().await;
        assert!(metrics.contains_key(&0));
    }
    #[tokio::test]
    async fn test_alert_generation() {
        let config = GpuProfilerConfig::default();
        let profiler = GpuProfiler::new(config).expect("test operation should succeed");
        let result = profiler
            .generate_alert(0, GpuAlertType::HighTemperature, AlertSeverity::High, 85.0)
            .await;
        assert!(result.is_ok());
        let alerts = profiler.get_recent_alerts(None).await;
        assert_eq!(alerts.len(), 1);
    }
    #[tokio::test]
    async fn test_report_generation() {
        let config = GpuProfilerConfig::default();
        let profiler = GpuProfiler::new(config).expect("test operation should succeed");
        profiler
            .collect_utilization_metrics(0)
            .await
            .expect("async operation should succeed in test");
        let report = profiler.generate_report().await;
        assert!(report.is_ok());
    }
}
