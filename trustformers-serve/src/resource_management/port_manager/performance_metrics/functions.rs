//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

use super::types::{PerformanceConfig, PerformanceGrade, PortPerformanceMetrics};

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::test;
    #[test]
    async fn test_performance_metrics_initialization() {
        let metrics = PortPerformanceMetrics::new().await.unwrap();
        let snapshot = metrics.get_current_snapshot().await;
        assert_eq!(snapshot.total_allocations, 0);
        assert_eq!(snapshot.total_deallocations, 0);
        assert_eq!(snapshot.avg_allocation_time_ms, 0.0);
        assert_eq!(snapshot.success_rate_percent, 100.0);
    }
    #[test]
    async fn test_allocation_success_recording() {
        let metrics = PortPerformanceMetrics::new().await.unwrap();
        metrics.record_allocation_success(Duration::from_millis(50)).await;
        metrics.record_allocation_success(Duration::from_millis(75)).await;
        let snapshot = metrics.get_current_snapshot().await;
        assert_eq!(snapshot.total_allocations, 2);
        assert_eq!(snapshot.avg_allocation_time_ms, 62.5);
        assert_eq!(snapshot.success_rate_percent, 100.0);
    }
    #[test]
    async fn test_allocation_failure_recording() {
        let metrics = PortPerformanceMetrics::new().await.unwrap();
        metrics.record_allocation_success(Duration::from_millis(50)).await;
        metrics.record_allocation_failure().await;
        let snapshot = metrics.get_current_snapshot().await;
        assert_eq!(snapshot.total_allocations, 1);
        assert_eq!(snapshot.success_rate_percent, 50.0);
    }
    #[test]
    async fn test_conflict_recording() {
        let metrics = PortPerformanceMetrics::new().await.unwrap();
        metrics.record_allocation_success(Duration::from_millis(50)).await;
        metrics.record_conflict().await;
        let snapshot = metrics.get_current_snapshot().await;
        assert_eq!(snapshot.total_conflicts, 1);
        assert_eq!(snapshot.conflict_rate_percent, 100.0);
    }
    #[test]
    async fn test_snapshot_creation_and_history() {
        let metrics = PortPerformanceMetrics::new().await.unwrap();
        metrics.record_allocation_success(Duration::from_millis(50)).await;
        let snapshot1 = metrics.create_snapshot().await;
        metrics.record_allocation_success(Duration::from_millis(100)).await;
        let snapshot2 = metrics.create_snapshot().await;
        let history = metrics.get_performance_history().await;
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].total_allocations, 1);
        assert_eq!(history[1].total_allocations, 2);
    }
    #[test]
    async fn test_performance_trends_calculation() {
        let metrics = PortPerformanceMetrics::new().await.unwrap();
        for i in 1..=5 {
            metrics.record_allocation_success(Duration::from_millis(i * 10)).await;
            metrics.create_snapshot().await;
        }
        let trends = metrics.get_performance_trends().await;
        assert!(trends.allocation_time_trend > 0.0);
    }
    #[test]
    async fn test_percentile_calculations() {
        let metrics = PortPerformanceMetrics::new().await.unwrap();
        let times = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        for time in times {
            metrics.record_allocation_success(Duration::from_millis(time)).await;
        }
        let snapshot = metrics.get_current_snapshot().await;
        assert!(snapshot.median_allocation_time_ms > 0.0);
        assert!(snapshot.p95_allocation_time_ms >= snapshot.median_allocation_time_ms);
        assert!(snapshot.p99_allocation_time_ms >= snapshot.p95_allocation_time_ms);
    }
    #[test]
    async fn test_performance_report_generation() {
        let metrics = PortPerformanceMetrics::new().await.unwrap();
        metrics.record_allocation_success(Duration::from_millis(50)).await;
        metrics.record_deallocation_success(Duration::from_millis(25)).await;
        metrics.record_conflict().await;
        let report = metrics.generate_performance_report().await;
        assert!(report.contains("Performance Report"));
        assert!(report.contains("Average Allocation Time"));
        assert!(report.contains("Success Rate"));
        assert!(report.contains("Conflict Rate"));
    }
    #[test]
    async fn test_performance_analysis() {
        let metrics = PortPerformanceMetrics::new().await.unwrap();
        for i in 1..=10 {
            metrics.record_allocation_success(Duration::from_millis(i * 20)).await;
            metrics.create_snapshot().await;
        }
        let analysis = metrics.analyze_performance().await;
        assert!(! matches!(analysis.performance_grade, PerformanceGrade::Critical));
    }
    #[test]
    async fn test_configuration_update() {
        let metrics = PortPerformanceMetrics::new().await.unwrap();
        let mut new_config = PerformanceConfig::default();
        new_config.history_size = 50;
        new_config.enable_detailed_timing = false;
        metrics.update_config(new_config.clone()).await;
        let updated_config = metrics.get_config().await;
        assert_eq!(updated_config.history_size, 50);
        assert!(! updated_config.enable_detailed_timing);
    }
    #[test]
    async fn test_metrics_reset() {
        let metrics = PortPerformanceMetrics::new().await.unwrap();
        metrics.record_allocation_success(Duration::from_millis(50)).await;
        metrics.record_conflict().await;
        metrics.create_snapshot().await;
        metrics.reset_metrics().await;
        let snapshot = metrics.get_current_snapshot().await;
        assert_eq!(snapshot.total_allocations, 0);
        assert_eq!(snapshot.total_conflicts, 0);
        let history = metrics.get_performance_history().await;
        assert!(history.is_empty());
    }
    #[test]
    async fn test_data_cleanup() {
        let metrics = PortPerformanceMetrics::new().await.unwrap();
        for _ in 0..1000 {
            metrics.record_allocation_success(Duration::from_millis(50)).await;
        }
        let cleaned = metrics.cleanup_old_data().await;
        assert!(cleaned > 0);
    }
    #[test]
    async fn test_data_export() {
        let metrics = PortPerformanceMetrics::new().await.unwrap();
        metrics.record_allocation_success(Duration::from_millis(50)).await;
        metrics.create_snapshot().await;
        let export_data = metrics.export_performance_data().await.unwrap();
        assert!(export_data.contains("current_snapshot"));
        assert!(export_data.contains("trends"));
        assert!(export_data.contains("history"));
    }
    #[test]
    async fn test_concurrent_metrics_recording() {
        use std::sync::Arc;
        use tokio::task;
        let metrics = Arc::new(PortPerformanceMetrics::new().await.unwrap());
        let mut handles = vec![];
        for i in 0..10 {
            let metrics_clone = Arc::clone(&metrics);
            let handle = task::spawn(async move {
                for _ in 0..10 {
                    metrics_clone
                        .record_allocation_success(Duration::from_millis(i * 10 + 50))
                        .await;
                }
            });
            handles.push(handle);
        }
        for handle in handles {
            handle.await.unwrap();
        }
        let snapshot = metrics.get_current_snapshot().await;
        assert_eq!(snapshot.total_allocations, 100);
    }
}
