//! Tests for port health monitor types

use super::types::*;
use chrono::Utc;
use std::time::Duration;

/// Simple LCG for deterministic pseudo-random values
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() % 10000) as f32 / 10000.0
    }
}

#[test]
fn test_efficiency_metrics_creation() {
    let metrics = EfficiencyMetrics {
        allocation_efficiency: 0.92,
        utilization_efficiency: 0.85,
        recycling_rate: 0.78,
        waste_ratio: 0.05,
        throughput_efficiency: 0.9,
    };
    assert!(metrics.allocation_efficiency > 0.0);
    assert!(metrics.waste_ratio < metrics.utilization_efficiency);
}

#[test]
fn test_predictive_indicators_creation() {
    let indicators = PredictiveIndicators {
        predicted_load: 0.75,
        confidence: 0.9,
        trend_direction: TrendDirection::Improving,
        predicted_exhaustion_time: Some(Duration::from_secs(3600)),
        risk_score: 0.3,
    };
    assert!(indicators.predicted_load > 0.0);
    assert!(indicators.confidence > 0.0);
}

#[test]
fn test_health_status_variants() {
    let statuses = [
        HealthStatus::Healthy,
        HealthStatus::Degraded,
        HealthStatus::Unhealthy,
        HealthStatus::Critical,
    ];
    assert_eq!(statuses.len(), 4);
}

#[test]
fn test_alert_severity_variants() {
    let severities = [
        AlertSeverity::Info,
        AlertSeverity::Warning,
        AlertSeverity::Error,
        AlertSeverity::Critical,
    ];
    assert_eq!(severities.len(), 4);
}

#[test]
fn test_event_severity_variants() {
    let severities = [
        EventSeverity::Low,
        EventSeverity::Medium,
        EventSeverity::High,
        EventSeverity::Critical,
    ];
    assert_eq!(severities.len(), 4);
}

#[test]
fn test_trend_direction_variants() {
    let dirs = [
        TrendDirection::Improving,
        TrendDirection::Stable,
        TrendDirection::Degrading,
        TrendDirection::Unknown,
    ];
    assert_eq!(dirs.len(), 4);
}

#[test]
fn test_alert_status_variants() {
    let statuses = [
        AlertStatus::Active,
        AlertStatus::Acknowledged,
        AlertStatus::Resolved,
    ];
    assert_eq!(statuses.len(), 3);
}

#[test]
fn test_health_event_type_variants() {
    let types = [
        HealthEventType::HealthCheckPassed,
        HealthEventType::HealthCheckFailed,
        HealthEventType::ThresholdExceeded,
        HealthEventType::RecoveryDetected,
        HealthEventType::AlertTriggered,
        HealthEventType::AlertResolved,
        HealthEventType::StatusChanged,
        HealthEventType::MetricAnomaly,
    ];
    assert_eq!(types.len(), 8);
}

#[test]
fn test_alert_type_variants_count() {
    let types = [
        AlertType::HighLatency,
        AlertType::HighErrorRate,
        AlertType::PortExhaustion,
        AlertType::ConnectionLimit,
        AlertType::ResourceContention,
        AlertType::HealthDegraded,
        AlertType::PerformanceAnomaly,
        AlertType::ConfigurationIssue,
    ];
    assert_eq!(types.len(), 8);
}

#[test]
fn test_port_health_thresholds_creation() {
    let thresholds = PortHealthThresholds {
        max_latency: Duration::from_millis(100),
        max_error_rate: 0.05,
        max_connection_count: 1000,
        max_retry_rate: 0.1,
        min_availability: 0.99,
        max_response_time: Duration::from_millis(500),
        warning_utilization: 0.7,
        critical_utilization: 0.9,
        max_consecutive_failures: 3,
        health_check_interval: Duration::from_secs(10),
    };
    assert!(thresholds.min_availability > 0.0);
    assert!(thresholds.warning_utilization < thresholds.critical_utilization);
}

#[test]
fn test_port_health_config_creation() {
    let config = PortHealthConfig {
        check_interval: Duration::from_secs(5),
        history_retention: Duration::from_secs(3600),
        max_history_entries: 1000,
        enable_predictive: true,
        alert_cooldown: Duration::from_secs(60),
        auto_recovery: true,
        thresholds: PortHealthThresholds {
            max_latency: Duration::from_millis(100),
            max_error_rate: 0.05,
            max_connection_count: 1000,
            max_retry_rate: 0.1,
            min_availability: 0.99,
            max_response_time: Duration::from_millis(500),
            warning_utilization: 0.7,
            critical_utilization: 0.9,
            max_consecutive_failures: 3,
            health_check_interval: Duration::from_secs(10),
        },
    };
    assert!(config.enable_predictive);
    assert!(config.auto_recovery);
}

#[test]
fn test_performance_baseline_creation() {
    let baseline = PerformanceBaseline {
        avg_latency: Duration::from_millis(50),
        p95_latency: Duration::from_millis(100),
        p99_latency: Duration::from_millis(200),
        avg_throughput: 1000.0,
        error_rate: 0.01,
        sample_count: 1000,
        established_at: Utc::now(),
    };
    assert!(baseline.avg_latency < baseline.p95_latency);
    assert!(baseline.p95_latency < baseline.p99_latency);
}

#[test]
fn test_health_trend_analysis_creation() {
    let analysis = HealthTrendAnalysis {
        direction: TrendDirection::Stable,
        magnitude: 0.1,
        confidence: 0.85,
        data_points: 100,
        period: Duration::from_secs(3600),
    };
    assert!(analysis.confidence > 0.0 && analysis.confidence <= 1.0);
    assert!(analysis.data_points > 0);
}

#[test]
fn test_port_health_alert_creation() {
    let alert = PortHealthAlert {
        alert_type: AlertType::HighLatency,
        severity: AlertSeverity::Warning,
        message: "Latency above threshold".to_string(),
        port: 8080,
        timestamp: Utc::now(),
        status: AlertStatus::Active,
        threshold_value: 100.0,
        actual_value: 150.0,
        recommended_action: Some("Investigate network".to_string()),
    };
    assert_eq!(alert.port, 8080);
    assert!(alert.actual_value > alert.threshold_value);
}

#[test]
fn test_health_statistics_creation() {
    let stats = HealthStatistics {
        total_checks: 1000,
        passed_checks: 980,
        failed_checks: 20,
        average_check_duration: Duration::from_millis(5),
        uptime_percentage: 0.98,
    };
    assert_eq!(stats.total_checks, stats.passed_checks + stats.failed_checks);
    assert!(stats.uptime_percentage > 0.0);
}

#[test]
fn test_port_health_event_creation() {
    let event = PortHealthEvent {
        event_type: HealthEventType::HealthCheckPassed,
        severity: EventSeverity::Low,
        port: 8443,
        timestamp: Utc::now(),
        message: "Health check passed".to_string(),
        details: std::collections::HashMap::new(),
    };
    assert_eq!(event.port, 8443);
}

#[test]
fn test_port_health_status_creation() {
    let status = PortHealthStatus {
        port: 8080,
        health: HealthStatus::Healthy,
        last_check: Utc::now(),
        consecutive_failures: 0,
        current_latency: Duration::from_millis(20),
        current_error_rate: 0.001,
        current_connections: 50,
        availability: 0.999,
        uptime: Duration::from_secs(86400),
        efficiency: EfficiencyMetrics {
            allocation_efficiency: 0.9,
            utilization_efficiency: 0.85,
            recycling_rate: 0.75,
            waste_ratio: 0.05,
            throughput_efficiency: 0.88,
        },
        predictive: PredictiveIndicators {
            predicted_load: 0.6,
            confidence: 0.85,
            trend_direction: TrendDirection::Stable,
            predicted_exhaustion_time: None,
            risk_score: 0.15,
        },
    };
    assert_eq!(status.port, 8080);
    assert_eq!(status.consecutive_failures, 0);
}

#[test]
fn test_health_trend_analysis_with_lcg_data() {
    let mut lcg = Lcg::new(42);
    let analyses: Vec<HealthTrendAnalysis> = (0..5)
        .map(|_| HealthTrendAnalysis {
            direction: TrendDirection::Stable,
            magnitude: lcg.next_f32(),
            confidence: lcg.next_f32().clamp(0.0, 1.0),
            data_points: (lcg.next_u64() % 1000) as usize,
            period: Duration::from_secs(lcg.next_u64() % 7200),
        })
        .collect();
    assert_eq!(analyses.len(), 5);
    for a in &analyses {
        assert!(a.confidence >= 0.0 && a.confidence <= 1.0);
    }
}

#[test]
fn test_efficiency_metrics_bounds() {
    let mut lcg = Lcg::new(123);
    for _ in 0..10 {
        let m = EfficiencyMetrics {
            allocation_efficiency: lcg.next_f32().clamp(0.0, 1.0),
            utilization_efficiency: lcg.next_f32().clamp(0.0, 1.0),
            recycling_rate: lcg.next_f32().clamp(0.0, 1.0),
            waste_ratio: lcg.next_f32().clamp(0.0, 1.0),
            throughput_efficiency: lcg.next_f32().clamp(0.0, 1.0),
        };
        assert!(m.allocation_efficiency >= 0.0 && m.allocation_efficiency <= 1.0);
        assert!(m.waste_ratio >= 0.0 && m.waste_ratio <= 1.0);
    }
}

#[test]
fn test_port_health_event_with_details() {
    let mut details = std::collections::HashMap::new();
    details.insert("reason".to_string(), "timeout".to_string());
    details.insert("duration_ms".to_string(), "5000".to_string());
    let event = PortHealthEvent {
        event_type: HealthEventType::HealthCheckFailed,
        severity: EventSeverity::High,
        port: 9090,
        timestamp: Utc::now(),
        message: "Health check timed out".to_string(),
        details,
    };
    assert_eq!(event.details.len(), 2);
}
