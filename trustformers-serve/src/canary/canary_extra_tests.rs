#![cfg(test)]
/// Extended tests for the canary deployment module.
use super::*;

fn simple_config() -> CanaryConfig {
    CanaryConfig::new("canary-v2", "baseline-v1")
}

// ── 32. CanaryConfig::new — correct model IDs ─────────────────────────────
#[test]
fn test_canary_config_model_ids() {
    let cfg = simple_config();
    assert_eq!(cfg.canary_model_id, "canary-v2");
    assert_eq!(cfg.baseline_model_id, "baseline-v1");
}

// ── 33. CanaryConfig::new — auto_rollback is true by default ─────────────
#[test]
fn test_canary_config_auto_rollback_true() {
    let cfg = simple_config();
    assert!(cfg.auto_rollback);
}

// ── 34. CanaryConfig::new — auto_promote is false by default ─────────────
#[test]
fn test_canary_config_auto_promote_false() {
    let cfg = simple_config();
    assert!(!cfg.auto_promote);
}

// ── 35. CanaryConfig::new — initial_traffic_percent is 5.0 ───────────────
#[test]
fn test_canary_config_initial_traffic() {
    let cfg = simple_config();
    assert!((cfg.initial_traffic_percent - 5.0).abs() < f32::EPSILON);
}

// ── 36. CanaryMetrics::canary_error_rate — 0.0 when no requests ──────────
#[test]
fn test_canary_metrics_error_rate_zero_when_empty() {
    let m = CanaryMetrics::default();
    assert_eq!(m.canary_error_rate(), 0.0);
}

// ── 37. CanaryMetrics::canary_error_rate — correct fraction ──────────────
#[test]
fn test_canary_metrics_error_rate_correct() {
    let m = CanaryMetrics {
        canary_requests: 100,
        canary_errors: 5,
        ..CanaryMetrics::default()
    };
    let rate = m.canary_error_rate();
    assert!(
        (rate - 0.05).abs() < 1e-6,
        "error rate should be 0.05, got {rate}"
    );
}

// ── 38. CanaryMetrics::baseline_error_rate — 0.0 when no requests ─────────
#[test]
fn test_baseline_error_rate_zero_when_empty() {
    let m = CanaryMetrics::default();
    assert_eq!(m.baseline_error_rate(), 0.0);
}

// ── 39. CanaryMetrics::latency_ratio — 1.0 when no baseline data ─────────
#[test]
fn test_latency_ratio_one_when_no_baseline() {
    let m = CanaryMetrics {
        canary_requests: 10,
        canary_total_latency_ms: 100.0,
        ..CanaryMetrics::default()
    };
    assert!((m.latency_ratio() - 1.0).abs() < 1e-5);
}

// ── 40. CanaryMetrics::latency_ratio — correct ratio ─────────────────────
#[test]
fn test_latency_ratio_correct() {
    let m = CanaryMetrics {
        canary_requests: 10,
        canary_total_latency_ms: 200.0,
        baseline_requests: 10,
        baseline_total_latency_ms: 100.0,
        ..CanaryMetrics::default()
    };
    // canary mean = 20, baseline mean = 10, ratio = 2.0
    let ratio = m.latency_ratio();
    assert!(
        (ratio - 2.0).abs() < 1e-5,
        "ratio should be 2.0, got {ratio}"
    );
}

// ── 41. RollbackThreshold::default — sensible values ─────────────────────
#[test]
fn test_rollback_threshold_default() {
    let t = RollbackThreshold::default();
    assert!(t.max_error_rate > 0.0 && t.max_error_rate < 1.0);
    assert!(t.max_latency_p99_ratio > 1.0);
    assert!(t.min_success_rate > 0.0 && t.min_success_rate < 1.0);
}

// ── 42. PromotionCriteria::default — min_requests = 1000 ─────────────────
#[test]
fn test_promotion_criteria_default_min_requests() {
    let p = PromotionCriteria::default();
    assert_eq!(p.min_requests, 1000);
}

// ── 43. CanaryMetricsHistory::push_canary_metrics — length grows ──────────
#[test]
fn test_canary_metrics_history_push_grows() {
    let mut h = CanaryMetricsHistory::default();
    assert_eq!(h.snapshots.len(), 0);
    h.push_canary_metrics(0, 0.01, 100.0);
    assert_eq!(h.snapshots.len(), 1);
    h.push_canary_metrics(1, 0.02, 110.0);
    assert_eq!(h.snapshots.len(), 2);
}

// ── 44. CanaryMetricsHistory — snapshot values stored correctly ───────────
#[test]
fn test_canary_metrics_history_snapshot_values() {
    let mut h = CanaryMetricsHistory::default();
    h.push_canary_metrics(5, 0.03, 120.0);
    let snap = &h.snapshots[0];
    assert_eq!(snap.step, 5);
    assert!((snap.error_rate - 0.03).abs() < 1e-6);
    assert!((snap.p99_latency_ms - 120.0).abs() < 1e-6);
}

// ── 45. CanaryPhase::Idle display name ────────────────────────────────────
#[test]
fn test_canary_phase_idle_display_name() {
    let phase = CanaryPhase::Idle;
    assert_eq!(phase.display_name(), "Idle");
}

// ── 46. CanaryPhase::FullyPromoted display name ───────────────────────────
#[test]
fn test_canary_phase_fully_promoted_display_name() {
    let phase = CanaryPhase::FullyPromoted;
    assert_eq!(phase.display_name(), "FullyPromoted");
}

// ── 47. CanaryPhase::RolledBack display name ──────────────────────────────
#[test]
fn test_canary_phase_rolled_back_display_name() {
    let phase = CanaryPhase::RolledBack {
        reason: "high error".to_string(),
    };
    assert_eq!(phase.display_name(), "RolledBack");
}

// ── 48. CanaryPhase::Running display name ─────────────────────────────────
#[test]
fn test_canary_phase_running_display_name() {
    let phase = CanaryPhase::Running {
        traffic_percent: 10.0,
        started_at: std::time::Instant::now(),
    };
    assert_eq!(phase.display_name(), "Running");
}

// ── 49. CanaryError::NotActive display is non-empty ───────────────────────
#[test]
fn test_canary_error_display_non_empty() {
    let e = CanaryError::NotActive {
        phase: "Idle".to_string(),
    };
    assert!(!e.to_string().is_empty());
}

// ── 50. CanaryError::InvalidConfig display contains message ───────────────
#[test]
fn test_canary_error_invalid_config_message() {
    let e = CanaryError::InvalidConfig("bad step".to_string());
    assert!(
        e.to_string().contains("bad step"),
        "error message must contain the config reason"
    );
}

// ── 51. CanaryMetrics::canary_mean_latency_ms — 0.0 when empty ───────────
#[test]
fn test_canary_mean_latency_ms_zero_when_empty() {
    let m = CanaryMetrics::default();
    assert_eq!(m.canary_mean_latency_ms(), 0.0);
}

// ── 52. CanaryMetrics::baseline_mean_latency_ms — correct value ───────────
#[test]
fn test_baseline_mean_latency_correct() {
    let m = CanaryMetrics {
        baseline_requests: 5,
        baseline_total_latency_ms: 250.0,
        ..CanaryMetrics::default()
    };
    assert!((m.baseline_mean_latency_ms() - 50.0).abs() < 1e-9);
}
