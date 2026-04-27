/// Extended tests for the A/B testing module.
#[cfg(test)]
mod ab_testing_extra_tests {
    use super::super::*;

    fn even_split() -> TrafficSplit {
        TrafficSplit::new(vec![
            ("control".to_string(), 0.5),
            ("treatment".to_string(), 0.5),
        ])
    }

    // ── 34. TrafficSplit::validate — empty variants returns EmptyTrafficSplit ──
    #[test]
    fn test_traffic_split_empty_returns_error() {
        let split = TrafficSplit::new(vec![]);
        let err = split.validate().unwrap_err();
        assert!(matches!(err, AbTestError::EmptyTrafficSplit));
    }

    // ── 35. TrafficSplit::validate — fractions not summing to 1.0 returns error
    #[test]
    fn test_traffic_split_invalid_sum_returns_error() {
        let split = TrafficSplit::new(vec![
            ("a".to_string(), 0.3),
            ("b".to_string(), 0.4), // sum = 0.7, not 1.0
        ]);
        let err = split.validate().unwrap_err();
        assert!(matches!(err, AbTestError::InvalidTrafficSplit { .. }));
    }

    // ── 36. TrafficSplit::validate — sum exactly 1.0 is ok ────────────────────
    #[test]
    fn test_traffic_split_valid_sum_ok() {
        let split = even_split();
        assert!(split.validate().is_ok());
    }

    // ── 37. TrafficSplit::select_variant — same user hash gives same variant ──
    #[test]
    fn test_select_variant_deterministic() {
        let split = even_split();
        let v1 = split.select_variant(12345);
        let v2 = split.select_variant(12345);
        assert_eq!(v1, v2);
    }

    // ── 38. TrafficSplit::select_variant — returns one of the known variants ──
    #[test]
    fn test_select_variant_returns_known_variant() {
        let split = even_split();
        let known = ["control", "treatment"];
        for hash in 0u64..20 {
            let v = split.select_variant(hash);
            assert!(known.contains(&v), "variant '{v}' must be one of the known variants");
        }
    }

    // ── 39. TrafficSplit::select_variant — approximate distribution is even ───
    #[test]
    fn test_select_variant_approximate_distribution() {
        let split = even_split();
        let mut control_count = 0usize;
        let total = 10_000u64;
        for hash in 0..total {
            if split.select_variant(hash) == "control" {
                control_count += 1;
            }
        }
        let fraction = control_count as f64 / total as f64;
        assert!(
            (fraction - 0.5).abs() < 0.05,
            "expected ~50% control, got {fraction:.3}"
        );
    }

    // ── 40. ExperimentVariantStats::new — initial fields are zero ─────────────
    #[test]
    fn test_variant_stats_initial_fields() {
        let stats = ExperimentVariantStats::new("control");
        assert_eq!(stats.num_requests, 0);
        assert_eq!(stats.error_count, 0);
        assert!((stats.total_latency_ms - 0.0).abs() < f64::EPSILON);
    }

    // ── 41. ExperimentVariantStats::mean_latency_ms — 0.0 when empty ─────────
    #[test]
    fn test_variant_stats_mean_latency_zero_when_empty() {
        let stats = ExperimentVariantStats::new("v");
        assert_eq!(stats.mean_latency_ms(), 0.0);
    }

    // ── 42. ExperimentVariantStats::mean_latency_ms — correct value ───────────
    #[test]
    fn test_variant_stats_mean_latency_correct() {
        let mut stats = ExperimentVariantStats::new("v");
        stats.record_request(100.0, false);
        stats.record_request(200.0, false);
        assert!((stats.mean_latency_ms() - 150.0).abs() < 1e-9);
    }

    // ── 43. ExperimentVariantStats::error_rate — 0.0 when empty ─────────────
    #[test]
    fn test_variant_stats_error_rate_zero_when_empty() {
        let stats = ExperimentVariantStats::new("v");
        assert_eq!(stats.error_rate(), 0.0);
    }

    // ── 44. ExperimentVariantStats::error_rate — correct fraction ─────────────
    #[test]
    fn test_variant_stats_error_rate_correct() {
        let mut stats = ExperimentVariantStats::new("v");
        stats.record_request(50.0, true);
        stats.record_request(50.0, false);
        stats.record_request(50.0, false);
        stats.record_request(50.0, false);
        let rate = stats.error_rate();
        assert!((rate - 0.25).abs() < 1e-6, "error rate should be 0.25, got {rate}");
    }

    // ── 45. ExperimentVariantStats::record_metric — accumulates values ────────
    #[test]
    fn test_variant_stats_record_metric_accumulates() {
        let mut stats = ExperimentVariantStats::new("v");
        stats.record_metric("throughput", 10.0);
        stats.record_metric("throughput", 15.0);
        let val = stats.custom_metrics.get("throughput").copied().unwrap_or(0.0);
        assert!((val - 25.0).abs() < 1e-9, "accumulated metric should be 25.0, got {val}");
    }

    // ── 46. ExperimentConfig::new — status is Draft ──────────────────────────
    #[test]
    fn test_experiment_config_new_status_draft() {
        let cfg = ExperimentConfig::new("exp-1", "Test Exp", even_split());
        assert_eq!(cfg.status, ExperimentStatus::Draft);
    }

    // ── 47. ExperimentConfig::new — primary_metric is "latency_p99" ──────────
    #[test]
    fn test_experiment_config_new_primary_metric() {
        let cfg = ExperimentConfig::new("exp-1", "Test Exp", even_split());
        assert_eq!(cfg.primary_metric, "latency_p99");
    }

    // ── 48. ExperimentConfig::new — min_sample_size is 100 ───────────────────
    #[test]
    fn test_experiment_config_new_min_sample_size() {
        let cfg = ExperimentConfig::new("exp-1", "Test Exp", even_split());
        assert_eq!(cfg.min_sample_size, 100);
    }

    // ── 49. ExperimentStatus variants all differ ──────────────────────────────
    #[test]
    fn test_experiment_status_variants_differ() {
        assert_ne!(ExperimentStatus::Draft, ExperimentStatus::Running);
        assert_ne!(ExperimentStatus::Paused, ExperimentStatus::Completed);
    }

    // ── 50. StatisticalTest::two_sample_z_test — returns NAN for zero n ───────
    #[test]
    fn test_z_test_nan_for_zero_n() {
        let z = StatisticalTest::two_sample_z_test(0, 1.0, 1.0, 10, 1.0, 1.0);
        assert!(z.is_nan());
    }

    // ── 51. StatisticalTest::two_sample_z_test — positive z when mean1 > mean2
    #[test]
    fn test_z_test_positive_when_mean1_greater() {
        let z = StatisticalTest::two_sample_z_test(100, 20.0, 5.0, 100, 15.0, 5.0);
        assert!(z > 0.0, "z-score must be positive when mean1 > mean2, got {z}");
    }

    // ── 52. StatisticalTest::is_significant — high z-score is significant ─────
    #[test]
    fn test_z_test_high_z_significant() {
        assert!(
            StatisticalTest::is_significant(5.0, 0.05),
            "z=5.0 must be significant at alpha=0.05"
        );
    }

    // ── 53. StatisticalTest::is_significant — low z-score not significant ─────
    #[test]
    fn test_z_test_low_z_not_significant() {
        assert!(
            !StatisticalTest::is_significant(0.5, 0.05),
            "z=0.5 must not be significant at alpha=0.05"
        );
    }

    // ── 54. StatisticalTest::compute_z_critical — returns 1.96 for alpha=0.05 ─
    #[test]
    fn test_z_critical_alpha_005() {
        let z = StatisticalTest::compute_z_critical(0.05);
        assert!((z - 1.96).abs() < 1e-9);
    }

    // ── 55. StatisticalTest::compute_z_critical — returns 2.576 for alpha=0.01
    #[test]
    fn test_z_critical_alpha_001() {
        let z = StatisticalTest::compute_z_critical(0.01);
        assert!((z - 2.576).abs() < 1e-9);
    }

    // ── 56. StatisticalTest::approximate_p_value — in [0.0, 1.0] ─────────────
    #[test]
    fn test_p_value_in_unit_interval() {
        for z in [0.0, 1.0, 1.96, 3.0, 5.0, -2.0_f64] {
            let p = StatisticalTest::approximate_p_value(z);
            assert!(
                (0.0_f32..=1.0_f32).contains(&p),
                "p_value({z}) = {p} must be in [0, 1]"
            );
        }
    }
}
