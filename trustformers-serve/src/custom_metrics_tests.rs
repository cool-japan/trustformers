//! Tests for the custom metrics collector.
//!
//! Split out of `custom_metrics.rs` in 0.2.1: the honesty work on the
//! collection, trend-fitting and export paths took that file past the
//! 2000-line limit, and the tests are the part of it that stands on its own.

use super::*;

#[tokio::test]
async fn test_custom_metrics_collector_creation() {
    let config = CustomMetricsConfig::default();
    let collector = CustomMetricsCollector::new(config).expect("collection should succeed in test");
    assert!(collector.config.enabled);
}

#[tokio::test]
async fn test_metric_collection() {
    let config = CustomMetricsConfig::default();
    let collector = CustomMetricsCollector::new(config).expect("collection should succeed in test");

    let result = collector.collect_business_metric("revenue", 1000.0, HashMap::new()).await;

    assert!(result.is_ok());

    let summary = collector.get_metrics_summary().await;
    assert_eq!(summary.total_metrics, 1);
}

#[tokio::test]
async fn test_analytics() {
    let config = CustomMetricsConfig::default();
    let collector = CustomMetricsCollector::new(config).expect("collection should succeed in test");

    // Collect some test metrics
    for i in 0..10 {
        collector
            .collect_performance_metric("latency", i as f64 * 10.0, None, HashMap::new())
            .await
            .expect("test operation should succeed");
    }

    let analytics = collector
        .analyze_metrics()
        .await
        .expect("async operation should succeed in test");
    assert!(!analytics.averages.is_empty());
}

#[test]
fn test_default_config() {
    let config = CustomMetricsConfig::default();
    assert!(config.enabled);
    assert_eq!(config.collection_interval_seconds, 10);
    assert!(config.enable_real_time_analytics);
    assert!(config.enable_business_metrics);
    assert!(config.enable_performance_profiling);
    assert_eq!(config.retention_period_hours, 24);
    assert_eq!(config.max_metric_series, 10000);
}

#[test]
fn test_default_alert_thresholds() {
    let thresholds = AlertThresholds::default();
    assert!((thresholds.cpu_usage_threshold - 0.8).abs() < f64::EPSILON);
    assert!((thresholds.memory_usage_threshold - 0.9).abs() < f64::EPSILON);
    assert!((thresholds.error_rate_threshold - 0.05).abs() < f64::EPSILON);
    assert_eq!(thresholds.queue_depth_threshold, 100);
}

#[test]
fn test_default_export_config() {
    let config = MetricsExportConfig::default();
    assert!(config.prometheus_enabled);
    assert!(!config.influxdb_enabled);
    assert!(config.custom_endpoints.is_empty());
    assert_eq!(config.export_interval_seconds, 60);
}

#[tokio::test]
async fn test_collect_disabled() {
    let mut config = CustomMetricsConfig::default();
    config.enabled = false;
    let collector = CustomMetricsCollector::new(config).expect("creation ok");
    let result = collector.collect_business_metric("rev", 100.0, HashMap::new()).await;
    assert!(result.is_ok());
    let summary = collector.get_metrics_summary().await;
    assert_eq!(summary.total_metrics, 0);
}

#[tokio::test]
async fn test_collect_business_disabled() {
    let mut config = CustomMetricsConfig::default();
    config.enable_business_metrics = false;
    let collector = CustomMetricsCollector::new(config).expect("creation ok");
    let result = collector.collect_business_metric("rev", 100.0, HashMap::new()).await;
    assert!(result.is_ok());
    let summary = collector.get_metrics_summary().await;
    assert_eq!(summary.total_metrics, 0);
}

#[tokio::test]
async fn test_collect_performance_disabled() {
    let mut config = CustomMetricsConfig::default();
    config.enable_performance_profiling = false;
    let collector = CustomMetricsCollector::new(config).expect("creation ok");
    let result = collector.collect_performance_metric("lat", 50.0, None, HashMap::new()).await;
    assert!(result.is_ok());
    let summary = collector.get_metrics_summary().await;
    assert_eq!(summary.total_metrics, 0);
}

#[tokio::test]
async fn test_collect_system_metric() {
    let config = CustomMetricsConfig::default();
    let collector = CustomMetricsCollector::new(config).expect("creation ok");
    let result = collector
        .collect_system_metric("cpu", 0.5, SystemMetricType::CpuUsage, HashMap::new())
        .await;
    assert!(result.is_ok());
    let summary = collector.get_metrics_summary().await;
    assert_eq!(summary.total_metrics, 1);
}

#[tokio::test]
async fn test_collect_multiple_metrics() {
    let config = CustomMetricsConfig::default();
    let collector = CustomMetricsCollector::new(config).expect("creation ok");

    // Use LCG for deterministic values
    let mut lcg: u64 = 42;
    for i in 0..10 {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = (lcg % 1000) as f64 / 10.0;
        collector
            .collect_business_metric(&format!("metric_{}", i), val, HashMap::new())
            .await
            .expect("collect ok");
    }
    let summary = collector.get_metrics_summary().await;
    assert_eq!(summary.total_metrics, 10);
}

#[tokio::test]
async fn test_metric_name_business() {
    let config = CustomMetricsConfig::default();
    let collector = CustomMetricsCollector::new(config).expect("creation ok");
    let metric = CustomMetric::Business {
        name: "revenue".to_string(),
        value: 100.0,
        labels: HashMap::new(),
        timestamp: SystemTime::now(),
    };
    let name = collector.get_metric_name(&metric);
    assert_eq!(name, "business_revenue");
}

#[tokio::test]
async fn test_metric_name_performance() {
    let config = CustomMetricsConfig::default();
    let collector = CustomMetricsCollector::new(config).expect("creation ok");
    let metric = CustomMetric::Performance {
        name: "latency".to_string(),
        value: 50.0,
        percentile: Some(95.0),
        labels: HashMap::new(),
        timestamp: SystemTime::now(),
    };
    let name = collector.get_metric_name(&metric);
    assert_eq!(name, "performance_latency");
}

#[tokio::test]
async fn test_metric_name_system() {
    let config = CustomMetricsConfig::default();
    let collector = CustomMetricsCollector::new(config).expect("creation ok");
    let metric = CustomMetric::System {
        name: "cpu_usage".to_string(),
        value: 0.75,
        metric_type: SystemMetricType::CpuUsage,
        labels: HashMap::new(),
        timestamp: SystemTime::now(),
    };
    let name = collector.get_metric_name(&metric);
    assert_eq!(name, "system_cpu_usage");
}

#[tokio::test]
async fn test_metric_name_application() {
    let config = CustomMetricsConfig::default();
    let collector = CustomMetricsCollector::new(config).expect("creation ok");
    let metric = CustomMetric::Application {
        name: "accuracy".to_string(),
        value: 0.95,
        metric_type: ApplicationMetricType::ModelAccuracy,
        labels: HashMap::new(),
        timestamp: SystemTime::now(),
    };
    let name = collector.get_metric_name(&metric);
    assert_eq!(name, "application_accuracy");
}

#[tokio::test]
async fn test_metric_name_custom() {
    let config = CustomMetricsConfig::default();
    let collector = CustomMetricsCollector::new(config).expect("creation ok");
    let metric = CustomMetric::Custom {
        name: "my_metric".to_string(),
        value: 42.0,
        metric_type: "gauge".to_string(),
        labels: HashMap::new(),
        timestamp: SystemTime::now(),
    };
    let name = collector.get_metric_name(&metric);
    assert_eq!(name, "custom_my_metric");
}

#[tokio::test]
async fn test_get_metric_value_variants() {
    let config = CustomMetricsConfig::default();
    let collector = CustomMetricsCollector::new(config).expect("creation ok");

    let business = CustomMetric::Business {
        name: "a".to_string(),
        value: 1.0,
        labels: HashMap::new(),
        timestamp: SystemTime::now(),
    };
    assert!((collector.get_metric_value(&business) - 1.0).abs() < f64::EPSILON);

    let perf = CustomMetric::Performance {
        name: "b".to_string(),
        value: 2.0,
        percentile: None,
        labels: HashMap::new(),
        timestamp: SystemTime::now(),
    };
    assert!((collector.get_metric_value(&perf) - 2.0).abs() < f64::EPSILON);

    let sys = CustomMetric::System {
        name: "c".to_string(),
        value: 3.0,
        metric_type: SystemMetricType::CpuUsage,
        labels: HashMap::new(),
        timestamp: SystemTime::now(),
    };
    assert!((collector.get_metric_value(&sys) - 3.0).abs() < f64::EPSILON);

    let app = CustomMetric::Application {
        name: "d".to_string(),
        value: 4.0,
        metric_type: ApplicationMetricType::CacheHitRate,
        labels: HashMap::new(),
        timestamp: SystemTime::now(),
    };
    assert!((collector.get_metric_value(&app) - 4.0).abs() < f64::EPSILON);

    let custom = CustomMetric::Custom {
        name: "e".to_string(),
        value: 5.0,
        metric_type: "counter".to_string(),
        labels: HashMap::new(),
        timestamp: SystemTime::now(),
    };
    assert!((collector.get_metric_value(&custom) - 5.0).abs() < f64::EPSILON);
}

#[tokio::test]
async fn test_analyze_empty_metrics() {
    let config = CustomMetricsConfig::default();
    let collector = CustomMetricsCollector::new(config).expect("creation ok");
    let result = collector.analyze_metrics().await;
    assert!(result.is_ok());
    if let Ok(analytics) = result {
        assert!(analytics.averages.is_empty());
    }
}

#[tokio::test]
async fn test_metrics_summary_initial() {
    let config = CustomMetricsConfig::default();
    let collector = CustomMetricsCollector::new(config).expect("creation ok");
    let summary = collector.get_metrics_summary().await;
    assert_eq!(summary.total_metrics, 0);
    assert_eq!(summary.active_metric_series, 0);
    assert_eq!(summary.alert_count, 0);
}

#[tokio::test]
async fn test_export_metrics_no_error() {
    let config = CustomMetricsConfig::default();
    let collector = CustomMetricsCollector::new(config).expect("creation ok");
    let result = collector.export_metrics().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_collect_with_labels() {
    let config = CustomMetricsConfig::default();
    let collector = CustomMetricsCollector::new(config).expect("creation ok");

    let mut labels = HashMap::new();
    labels.insert("region".to_string(), "us-east".to_string());
    labels.insert("env".to_string(), "prod".to_string());

    let result = collector.collect_business_metric("requests", 500.0, labels).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_active_metrics_tracking() {
    let config = CustomMetricsConfig::default();
    let collector = CustomMetricsCollector::new(config).expect("creation ok");

    collector.collect_business_metric("a", 1.0, HashMap::new()).await.expect("ok");
    collector.collect_business_metric("b", 2.0, HashMap::new()).await.expect("ok");

    let active = collector.active_metrics.read().await;
    assert_eq!(active.len(), 2);
    assert!(active.contains("business_a"));
    assert!(active.contains("business_b"));
}

#[test]
fn test_custom_metrics_error_display() {
    let err = CustomMetricsError::ConfigurationError {
        message: "bad config".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("bad config"));
}

#[test]
fn test_collection_error_display() {
    let err = CustomMetricsError::CollectionError {
        message: "collection failed".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("collection failed"));
}

#[test]
fn test_trend_direction_debug() {
    let direction = TrendDirection::Increasing;
    let debug_str = format!("{:?}", direction);
    assert_eq!(debug_str, "Increasing");
}

#[test]
fn test_anomaly_severity_debug() {
    let sev = AnomalySeverity::Critical;
    let debug_str = format!("{:?}", sev);
    assert_eq!(debug_str, "Critical");
}

#[test]
fn test_bottleneck_type_debug() {
    let bt = BottleneckType::GpuBound;
    let debug_str = format!("{:?}", bt);
    assert_eq!(debug_str, "GpuBound");
}

#[tokio::test]
async fn test_prometheus_metrics_are_really_recorded() {
    // 0.2.1 regression guard: update_prometheus_metrics used to be a
    // no-op, so the three registries stayed empty no matter what was
    // recorded.
    let collector = CustomMetricsCollector::new(CustomMetricsConfig::default())
        .unwrap_or_else(|e| panic!("collector construction failed: {e}"));

    collector
        .collect_metric(CustomMetric::System {
            name: "cpu".to_string(),
            value: 42.5,
            metric_type: SystemMetricType::CpuUsage,
            labels: HashMap::new(),
            timestamp: SystemTime::now(),
        })
        .await
        .unwrap_or_else(|e| panic!("collect_metric failed: {e}"));
    assert_eq!(
        collector.prometheus_gauge_value("system_cpu").await,
        Some(42.5)
    );

    collector
        .collect_metric(CustomMetric::Performance {
            name: "latency".to_string(),
            value: 12.0,
            percentile: None,
            labels: HashMap::new(),
            timestamp: SystemTime::now(),
        })
        .await
        .unwrap_or_else(|e| panic!("collect_metric failed: {e}"));
    assert_eq!(
        collector.prometheus_histogram_count("performance_latency").await,
        Some(1)
    );

    collector
        .collect_metric(CustomMetric::Business {
            name: "orders".to_string(),
            value: 3.0,
            labels: HashMap::new(),
            timestamp: SystemTime::now(),
        })
        .await
        .unwrap_or_else(|e| panic!("collect_metric failed: {e}"));
    assert_eq!(
        collector.prometheus_counter_value("business_orders").await,
        Some(3)
    );
}

#[tokio::test]
async fn test_negative_business_metric_is_rejected_not_silently_dropped() {
    let collector = CustomMetricsCollector::new(CustomMetricsConfig::default())
        .unwrap_or_else(|e| panic!("collector construction failed: {e}"));
    let result = collector
        .collect_metric(CustomMetric::Business {
            name: "refunds".to_string(),
            value: -1.0,
            labels: HashMap::new(),
            timestamp: SystemTime::now(),
        })
        .await;
    assert!(
        result.is_err(),
        "a counter cannot decrease; this must be reported"
    );
}

// The honesty regressions below were a separate `mod honesty_tests` inside
// `custom_metrics.rs`; the split flattened both modules into this file.

fn snapshot(cpu_percent: f64, used: u64, total: u64) -> HostSnapshot {
    HostSnapshot {
        cpu_percent,
        used_memory_bytes: used,
        total_memory_bytes: total,
        memory_percent: if total == 0 { 0.0 } else { used as f64 / total as f64 * 100.0 },
        process_memory_bytes: 0,
        disk_percent: None,
    }
}

fn system_sample(name: &str, value: f64, timestamp: SystemTime) -> CustomMetric {
    CustomMetric::System {
        name: name.to_string(),
        value,
        metric_type: SystemMetricType::CpuUsage,
        labels: HashMap::new(),
        timestamp,
    }
}

/// Regression: `get_cpu_usage` / `get_memory_usage` returned `Ok(0.5)` and
/// `Ok(0.7)` regardless of the host. The readings now track the snapshot
/// they are derived from, and the zeroed measurement-unavailable snapshot
/// yields `None` rather than an idle-looking zero.
#[test]
fn host_readings_track_the_snapshot() {
    let idle = snapshot(3.0, 1_000, 10_000);
    let busy = snapshot(97.0, 9_000, 10_000);

    let idle_cpu = CustomMetricsCollector::cpu_usage_fraction(&idle)
        .expect("a snapshot with memory is a real measurement");
    let busy_cpu = CustomMetricsCollector::cpu_usage_fraction(&busy)
        .expect("a snapshot with memory is a real measurement");
    assert!((idle_cpu - 0.03).abs() < 1e-9, "{idle_cpu}");
    assert!((busy_cpu - 0.97).abs() < 1e-9, "{busy_cpu}");
    assert!(idle_cpu < busy_cpu, "the reading must follow the host");

    let idle_memory = CustomMetricsCollector::memory_usage_fraction(&idle)
        .expect("a snapshot with memory is a real measurement");
    let busy_memory = CustomMetricsCollector::memory_usage_fraction(&busy)
        .expect("a snapshot with memory is a real measurement");
    assert!((idle_memory - 0.1).abs() < 1e-9, "{idle_memory}");
    assert!((busy_memory - 0.9).abs() < 1e-9, "{busy_memory}");

    let unavailable = snapshot(0.0, 0, 0);
    assert!(
        CustomMetricsCollector::cpu_usage_fraction(&unavailable).is_none(),
        "the zeroed fallback is an absence, not an idle CPU"
    );
    assert!(
        CustomMetricsCollector::memory_usage_fraction(&unavailable).is_none(),
        "the zeroed fallback is an absence, not empty memory"
    );
}

/// The published readings are fractions of one, matching the contract
/// `AlertThresholds` documents and `detect_anomalies` compares against.
/// Publishing `sysinfo`'s 0-100 percent would make every sample exceed the
/// 0.8 CPU threshold.
#[tokio::test]
async fn collected_system_metrics_are_fractions_and_do_not_all_alarm() {
    let collector = CustomMetricsCollector::new(CustomMetricsConfig::default())
        .unwrap_or_else(|e| panic!("collector construction failed: {e}"));
    collector
        .collect_system_metrics()
        .await
        .unwrap_or_else(|e| panic!("collection failed: {e}"));

    let storage = collector.metrics_storage.read().await;
    for (name, metrics) in storage.iter() {
        for metric in metrics {
            let value = collector.get_metric_value(metric);
            assert!(
                (0.0..=1.0).contains(&value),
                "{name} published {value}, which is not a fraction of one"
            );
        }
    }
    assert!(
        storage.contains_key("system_cpu_usage"),
        "a CPU reading must be published on any host sysinfo can measure"
    );
}

/// Regression: every metric was handed `strength: 0.5`, `confidence: 0.8`
/// and `TrendDirection::Stable`, so a doubling series and a flat one
/// produced identical trends.
#[test]
fn trends_are_fitted_from_the_samples() {
    let rising: Vec<(f64, f64)> = (0..20).map(|i| (i as f64, 10.0 + 2.0 * i as f64)).collect();
    let trend = CustomMetricsCollector::fit_trend(&rising).expect("20 samples support a fit");
    assert!(
        matches!(trend.direction, TrendDirection::Increasing),
        "{trend:?}"
    );
    assert!(
        trend.strength > 0.99,
        "a straight line correlates: {trend:?}"
    );
    assert!(
        (trend.strength - 0.5).abs() > 1e-9 && (trend.confidence - 0.8).abs() > 1e-9,
        "no longer the hardcoded pair: {trend:?}"
    );
    assert!(
        (trend.duration.as_secs_f64() - 19.0).abs() < 1e-9,
        "the duration is the observed span, not a fixed five minutes: {trend:?}"
    );

    let falling: Vec<(f64, f64)> = (0..20).map(|i| (i as f64, 100.0 - 3.0 * i as f64)).collect();
    let trend = CustomMetricsCollector::fit_trend(&falling).expect("20 samples support a fit");
    assert!(
        matches!(trend.direction, TrendDirection::Decreasing),
        "{trend:?}"
    );

    let flat: Vec<(f64, f64)> = (0..20).map(|i| (i as f64, 42.0)).collect();
    let trend = CustomMetricsCollector::fit_trend(&flat).expect("20 samples support a fit");
    assert!(
        matches!(trend.direction, TrendDirection::Stable),
        "{trend:?}"
    );
    assert!(trend.strength.abs() < 1e-12, "{trend:?}");

    assert!(
        CustomMetricsCollector::fit_trend(&[(0.0, 1.0), (1.0, 2.0)]).is_none(),
        "two samples cannot support a trend"
    );
    assert!(
        CustomMetricsCollector::fit_trend(&[(0.0, 1.0), (0.0, 2.0), (0.0, 3.0)]).is_none(),
        "a zero-length observation span cannot support a trend"
    );
}

/// The window drives the trend map end to end, and a metric that leaves the
/// window loses its trend rather than keeping a stale one.
#[tokio::test]
async fn analyze_trends_reads_the_window() {
    let collector = CustomMetricsCollector::new(CustomMetricsConfig::default())
        .unwrap_or_else(|e| panic!("collector construction failed: {e}"));
    let mut analytics = RealTimeAnalytics {
        metrics_window: VecDeque::new(),
        window_size: Duration::from_secs(300),
        current_averages: HashMap::new(),
        trends: HashMap::new(),
        anomalies: Vec::new(),
    };
    let base = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
    for i in 0..12u64 {
        analytics.metrics_window.push_back(system_sample(
            "climber",
            i as f64,
            base + Duration::from_secs(i * 10),
        ));
    }
    collector
        .analyze_trends(&mut analytics)
        .await
        .unwrap_or_else(|e| panic!("trend analysis failed: {e}"));
    let trend = analytics
        .trends
        .get("system_climber")
        .unwrap_or_else(|| panic!("the fitted series must appear: {:?}", analytics.trends));
    assert!(
        matches!(trend.direction, TrendDirection::Increasing),
        "{trend:?}"
    );

    analytics.metrics_window.clear();
    collector
        .analyze_trends(&mut analytics)
        .await
        .unwrap_or_else(|e| panic!("trend analysis failed: {e}"));
    assert!(
        analytics.trends.is_empty(),
        "an empty window supports no trend at all"
    );
}

/// Regression: `generate_insights` returned the same three sentences on
/// every call, including on an empty window.
#[tokio::test]
async fn insights_are_derived_from_the_analytics() {
    let collector = CustomMetricsCollector::new(CustomMetricsConfig::default())
        .unwrap_or_else(|e| panic!("collector construction failed: {e}"));
    let empty = RealTimeAnalytics {
        metrics_window: VecDeque::new(),
        window_size: Duration::from_secs(300),
        current_averages: HashMap::new(),
        trends: HashMap::new(),
        anomalies: Vec::new(),
    };
    let insights = collector
        .generate_insights(&empty)
        .await
        .unwrap_or_else(|e| panic!("insight generation failed: {e}"));
    assert!(
        insights.is_empty(),
        "nothing was observed, so nothing can be reported: {insights:?}"
    );

    let mut loaded = empty.clone();
    loaded.anomalies.push(Anomaly {
        metric_name: "system_cpu_usage".to_string(),
        value: 0.95,
        expected_value: 0.8,
        deviation: 0.15,
        severity: AnomalySeverity::High,
        timestamp: SystemTime::now(),
        anomaly_type: AnomalyType::Spike,
    });
    let insights = collector
        .generate_insights(&loaded)
        .await
        .unwrap_or_else(|e| panic!("insight generation failed: {e}"));
    assert_eq!(insights.len(), 1, "{insights:?}");
    assert!(
        insights[0].contains("system_cpu_usage") && insights[0].contains("0.95"),
        "the insight must quote what was measured: {insights:?}"
    );
}

/// Regression: the three export sinks all returned `Ok(())` without doing
/// anything. InfluxDB now names the configuration it lacks.
#[tokio::test]
async fn influxdb_export_names_the_missing_endpoint() {
    let collector = CustomMetricsCollector::new(CustomMetricsConfig::default())
        .unwrap_or_else(|e| panic!("collector construction failed: {e}"));
    let error = collector
        .export_to_influxdb()
        .await
        .expect_err("there is no InfluxDB destination to write to");
    let message = error.to_string();
    assert!(
        message.contains("URL") && message.contains("custom_endpoints"),
        "the error must name what is missing: {message}"
    );
}

/// An unencodable format is refused by name rather than shipped as some
/// other encoding.
#[test]
fn export_encoding_is_refused_for_formats_without_an_encoder() {
    let mut values = HashMap::new();
    values.insert("system_cpu_usage".to_string(), 0.25);
    let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);

    let (body, content_type) =
        CustomMetricsCollector::encode_metrics(&values, &MetricsFormat::Prometheus, now)
            .unwrap_or_else(|e| panic!("prometheus encoding failed: {e}"));
    assert!(body.contains("system_cpu_usage 0.25"), "{body}");
    assert!(content_type.starts_with("text/plain"));

    let (body, _) = CustomMetricsCollector::encode_metrics(&values, &MetricsFormat::InfluxDB, now)
        .unwrap_or_else(|e| panic!("influx encoding failed: {e}"));
    assert!(
        body.contains("system_cpu_usage value=0.25 1700000000000000000"),
        "{body}"
    );

    let error = CustomMetricsCollector::encode_metrics(&values, &MetricsFormat::OpenTelemetry, now)
        .expect_err("no OpenTelemetry encoder exists here");
    assert!(
        error.to_string().contains("OpenTelemetry"),
        "{}",
        error.to_string()
    );
}
