//! Additional tests for the realtime_dashboard module.

use super::*;

// ── DashboardConfig ───────────────────────────────────────────────────────────

#[test]
fn test_dashboard_config_default() {
    let cfg = DashboardConfig::default();
    assert_eq!(cfg.websocket_port, 8080);
    assert_eq!(cfg.update_frequency_ms, 100);
    assert_eq!(cfg.max_data_points, 1000);
    assert!(cfg.enable_gpu_monitoring);
    assert!(cfg.enable_memory_profiling);
    assert!(!cfg.enable_network_monitoring);
    assert!(cfg.enable_performance_alerts);
}

#[test]
fn test_dashboard_config_clone() {
    let cfg = DashboardConfig::default();
    let cloned = cfg.clone();
    assert_eq!(cfg.websocket_port, cloned.websocket_port);
    assert_eq!(cfg.max_data_points, cloned.max_data_points);
}

// ── AlertThresholds ───────────────────────────────────────────────────────────

#[test]
fn test_alert_thresholds_default() {
    let thresholds = AlertThresholds::default();
    assert!((thresholds.memory_threshold - 90.0).abs() < 1e-10);
    assert!((thresholds.gpu_utilization_threshold - 95.0).abs() < 1e-10);
    assert!((thresholds.temperature_threshold - 80.0).abs() < 1e-10);
    assert!((thresholds.loss_spike_threshold - 2.0).abs() < 1e-10);
    assert!((thresholds.gradient_norm_threshold - 10.0).abs() < 1e-10);
}

#[test]
fn test_alert_thresholds_custom() {
    let thresholds = AlertThresholds {
        memory_threshold: 80.0,
        gpu_utilization_threshold: 90.0,
        temperature_threshold: 75.0,
        loss_spike_threshold: 3.0,
        gradient_norm_threshold: 5.0,
    };
    assert!((thresholds.memory_threshold - 80.0).abs() < 1e-10);
    assert!((thresholds.gradient_norm_threshold - 5.0).abs() < 1e-10);
}

// ── MetricCategory ────────────────────────────────────────────────────────────

#[test]
fn test_metric_category_variants_eq() {
    assert_eq!(MetricCategory::Training, MetricCategory::Training);
    assert_ne!(MetricCategory::Training, MetricCategory::GPU);
    assert_eq!(
        MetricCategory::Custom("my_metric".to_string()),
        MetricCategory::Custom("my_metric".to_string()),
    );
    assert_ne!(
        MetricCategory::Custom("a".to_string()),
        MetricCategory::Custom("b".to_string()),
    );
}

#[test]
fn test_metric_category_all_variants() {
    let variants = [
        MetricCategory::Training,
        MetricCategory::Memory,
        MetricCategory::GPU,
        MetricCategory::Network,
        MetricCategory::Performance,
        MetricCategory::Custom("custom".to_string()),
    ];
    for v in &variants {
        let _cloned = v.clone();
    }
}

// ── AlertSeverity ─────────────────────────────────────────────────────────────

#[test]
fn test_alert_severity_variants() {
    let variants = [
        AlertSeverity::Info,
        AlertSeverity::Warning,
        AlertSeverity::Error,
        AlertSeverity::Critical,
    ];
    for v in &variants {
        let _cloned = v.clone();
    }
}

// ── AnomalyType ───────────────────────────────────────────────────────────────

#[test]
fn test_anomaly_type_variants() {
    let variants = [
        AnomalyType::Spike,
        AnomalyType::Drop,
        AnomalyType::GradualIncrease,
        AnomalyType::GradualDecrease,
        AnomalyType::Outlier,
    ];
    for v in &variants {
        let _cloned = v.clone();
        let _debug = format!("{:?}", v);
    }
}

// ── TrendDirection ────────────────────────────────────────────────────────────

#[test]
fn test_trend_direction_variants() {
    let variants = [
        TrendDirection::Increasing,
        TrendDirection::Decreasing,
        TrendDirection::Stable,
    ];
    for v in &variants {
        let _cloned = v.clone();
    }
}

// ── ExportFormat ──────────────────────────────────────────────────────────────

#[test]
fn test_export_format_variants() {
    let variants = [
        ExportFormat::JSON,
        ExportFormat::CSV,
        ExportFormat::MessagePack,
    ];
    for v in &variants {
        let _cloned = v.clone();
    }
}

// ── MetricDataPoint ───────────────────────────────────────────────────────────

#[test]
fn test_metric_data_point_construction() {
    let pt = MetricDataPoint {
        timestamp: 1234567890,
        value: 3.14,
        label: "loss".to_string(),
        category: MetricCategory::Training,
    };
    assert_eq!(pt.label, "loss");
    assert!((pt.value - 3.14).abs() < 1e-10);
    assert_eq!(pt.category, MetricCategory::Training);
}

// ── DashboardAlert ────────────────────────────────────────────────────────────

#[test]
fn test_dashboard_alert_construction() {
    let alert = DashboardAlert {
        id: "alert-001".to_string(),
        timestamp: 9999999,
        severity: AlertSeverity::Warning,
        category: MetricCategory::Memory,
        title: "High Memory".to_string(),
        message: "Memory usage at 92%".to_string(),
        value: Some(92.0),
        threshold: Some(90.0),
    };
    assert_eq!(alert.title, "High Memory");
    assert!(alert.value.is_some());
    assert!(matches!(alert.severity, AlertSeverity::Warning));
}

// ── DashboardBuilder ──────────────────────────────────────────────────────────

#[test]
fn test_builder_default_config() {
    let dashboard = DashboardBuilder::new().build();
    let cfg = dashboard.get_config();
    assert_eq!(cfg.websocket_port, 8080);
    assert_eq!(cfg.max_data_points, 1000);
}

#[test]
fn test_builder_custom_port() {
    let dashboard = DashboardBuilder::new().port(9090).build();
    assert_eq!(dashboard.get_config().websocket_port, 9090);
}

#[test]
fn test_builder_disable_gpu_monitoring() {
    let dashboard = DashboardBuilder::new().gpu_monitoring(false).build();
    assert!(!dashboard.get_config().enable_gpu_monitoring);
}

#[test]
fn test_builder_disable_memory_profiling() {
    let dashboard = DashboardBuilder::new().memory_profiling(false).build();
    assert!(!dashboard.get_config().enable_memory_profiling);
}

#[test]
fn test_builder_alert_thresholds() {
    let thresholds = AlertThresholds {
        memory_threshold: 80.0,
        gpu_utilization_threshold: 88.0,
        temperature_threshold: 70.0,
        loss_spike_threshold: 1.5,
        gradient_norm_threshold: 8.0,
    };
    let dashboard = DashboardBuilder::new().alert_thresholds(thresholds.clone()).build();
    let cfg = dashboard.get_config();
    assert!((cfg.alert_thresholds.memory_threshold - 80.0).abs() < 1e-10);
    assert!((cfg.alert_thresholds.gradient_norm_threshold - 8.0).abs() < 1e-10);
}

// ── RealtimeDashboard ─────────────────────────────────────────────────────────

#[test]
fn test_dashboard_new() {
    let dashboard = RealtimeDashboard::new(DashboardConfig::default());
    let stats = dashboard.get_system_stats();
    assert_eq!(stats.data_points_collected, 0);
    assert_eq!(stats.total_alerts, 0);
}

#[test]
fn test_dashboard_add_single_metric() {
    let dashboard = RealtimeDashboard::new(DashboardConfig::default());
    let result = dashboard.add_metric(MetricCategory::Training, "loss".to_string(), 1.23);
    assert!(result.is_ok());
    let history = dashboard.get_historical_data(&MetricCategory::Training);
    assert_eq!(history.len(), 1);
    assert!((history[0].value - 1.23).abs() < 1e-10);
}

#[test]
fn test_dashboard_add_multiple_categories() {
    let dashboard = RealtimeDashboard::new(DashboardConfig::default());
    let _ = dashboard.add_metric(MetricCategory::Training, "loss".to_string(), 0.5);
    let _ = dashboard.add_metric(MetricCategory::GPU, "utilization".to_string(), 80.0);
    let _ = dashboard.add_metric(MetricCategory::Memory, "usage".to_string(), 4096.0);

    let training = dashboard.get_historical_data(&MetricCategory::Training);
    let gpu = dashboard.get_historical_data(&MetricCategory::GPU);
    let memory = dashboard.get_historical_data(&MetricCategory::Memory);

    assert_eq!(training.len(), 1);
    assert_eq!(gpu.len(), 1);
    assert_eq!(memory.len(), 1);
}

#[test]
fn test_dashboard_stop() {
    let dashboard = RealtimeDashboard::new(DashboardConfig::default());
    dashboard.stop(); // Should not panic even when not started
}

#[test]
fn test_dashboard_update_config() {
    let dashboard = RealtimeDashboard::new(DashboardConfig::default());
    let mut new_cfg = DashboardConfig::default();
    new_cfg.websocket_port = 9999;
    let result = dashboard.update_config(new_cfg);
    assert!(result.is_ok());
    assert_eq!(dashboard.get_config().websocket_port, 9999);
}

// ── SystemStats ───────────────────────────────────────────────────────────────

#[test]
fn test_system_stats_construction() {
    let stats = SystemStats {
        uptime: 3600,
        total_alerts: 5,
        active_connections: 2,
        data_points_collected: 1000,
        memory_usage_mb: 128.0,
        cpu_usage_percent: 45.0,
    };
    assert_eq!(stats.uptime, 3600);
    assert_eq!(stats.total_alerts, 5);
    assert!((stats.cpu_usage_percent - 45.0).abs() < 1e-10);
}

// ── AnomalyDetection ──────────────────────────────────────────────────────────

#[test]
fn test_anomaly_detection_construction() {
    let anomaly = AnomalyDetection {
        timestamp: 12345,
        value: 15.0,
        expected_range: (0.0, 10.0),
        anomaly_type: AnomalyType::Spike,
        confidence_score: 0.95,
        category: MetricCategory::Training,
        description: "Unexpected spike in training loss".to_string(),
    };
    assert!(anomaly.confidence_score > 0.0 && anomaly.confidence_score <= 1.0);
    assert!(matches!(anomaly.anomaly_type, AnomalyType::Spike));
    assert_eq!(anomaly.expected_range, (0.0, 10.0));
}
