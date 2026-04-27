#![cfg(test)]
/// Extended tests for the load shedding module.
use super::*;

fn zero_load() -> SystemLoad {
    SystemLoad {
        cpu_utilization: 0.0,
        memory_utilization: 0.0,
        active_requests: 0,
        queue_depth: 0,
        estimated_queue_latency_ms: 0.0,
    }
}

fn high_cpu_load() -> SystemLoad {
    SystemLoad {
        cpu_utilization: 0.95,
        memory_utilization: 0.1,
        active_requests: 10,
        queue_depth: 5,
        estimated_queue_latency_ms: 100.0,
    }
}

fn high_mem_load() -> SystemLoad {
    SystemLoad {
        cpu_utilization: 0.1,
        memory_utilization: 0.95,
        active_requests: 5,
        queue_depth: 3,
        estimated_queue_latency_ms: 50.0,
    }
}

// ── 29. LoadSheddingConfig::default — policy is None ─────────────────────
#[test]
fn test_config_default_policy_is_none() {
    let cfg = LoadSheddingConfig::default();
    assert_eq!(cfg.policy, LoadSheddingPolicy::None);
}

// ── 30. LoadSheddingConfig::default — 3 priority levels ──────────────────
#[test]
fn test_config_default_priority_levels() {
    let cfg = LoadSheddingConfig::default();
    assert_eq!(cfg.priority_levels, 3);
    assert_eq!(cfg.priority_thresholds.len(), 3);
}

// ── 31. LoadSheddingConfig::default — backoff_factor > 1 ─────────────────
#[test]
fn test_config_default_backoff_factor_gt_one() {
    let cfg = LoadSheddingConfig::default();
    assert!(cfg.backoff_factor > 1.0);
}

// ── 32. LoadShedder::new — initial drop_probability from config ───────────
#[test]
fn test_load_shedder_new_initial_prob() {
    let cfg = LoadSheddingConfig {
        policy: LoadSheddingPolicy::ProbabilisticDrop {
            drop_probability: 0.3,
        },
        ..LoadSheddingConfig::default()
    };
    let shedder = LoadShedder::new(cfg);
    assert!(
        (shedder.current_drop_probability - 0.3).abs() < f32::EPSILON,
        "initial prob should equal config, got {}",
        shedder.current_drop_probability
    );
}

// ── 33. LoadShedder::new — non-probabilistic policy starts at 0.0 ─────────
#[test]
fn test_load_shedder_new_non_probabilistic_starts_zero() {
    let shedder = LoadShedder::new(LoadSheddingConfig::default());
    assert_eq!(shedder.current_drop_probability, 0.0);
}

// ── 34. should_shed — None policy always accepts ──────────────────────────
#[test]
fn test_none_policy_always_accepts() {
    let mut shedder = LoadShedder::new(LoadSheddingConfig::default());
    for _ in 0..10 {
        let d = shedder.should_shed(0, &high_cpu_load());
        assert_eq!(
            d,
            SheddingDecision::Accept,
            "None policy must always accept"
        );
    }
}

// ── 35. should_shed — increments total_requests counter ──────────────────
#[test]
fn test_should_shed_increments_total_requests() {
    let mut shedder = LoadShedder::new(LoadSheddingConfig::default());
    shedder.should_shed(0, &zero_load());
    shedder.should_shed(0, &zero_load());
    assert_eq!(shedder.shedding_stats.total_requests, 2);
}

// ── 36. ResourceBased — cpu above threshold triggers Drop ────────────────
#[test]
fn test_resource_based_high_cpu_triggers_drop() {
    let cfg = LoadSheddingConfig {
        policy: LoadSheddingPolicy::ResourceBased {
            cpu_threshold: 0.8,
            memory_threshold: 0.9,
        },
        cooldown_ms: 0,
        ..LoadSheddingConfig::default()
    };
    let mut shedder = LoadShedder::new(cfg);
    let d = shedder.should_shed(2, &high_cpu_load());
    match d {
        SheddingDecision::Drop {
            reason: SheddingReason::HighCpuLoad { .. },
        } => {},
        other => panic!("expected HighCpuLoad drop, got {other:?}"),
    }
}

// ── 37. ResourceBased — memory above threshold triggers Drop ─────────────
#[test]
fn test_resource_based_high_memory_triggers_drop() {
    let cfg = LoadSheddingConfig {
        policy: LoadSheddingPolicy::ResourceBased {
            cpu_threshold: 0.9,
            memory_threshold: 0.8,
        },
        cooldown_ms: 0,
        ..LoadSheddingConfig::default()
    };
    let mut shedder = LoadShedder::new(cfg);
    let d = shedder.should_shed(2, &high_mem_load());
    match d {
        SheddingDecision::Drop {
            reason: SheddingReason::HighMemoryLoad { .. },
        } => {},
        other => panic!("expected HighMemoryLoad drop, got {other:?}"),
    }
}

// ── 38. LatencyBased — below threshold accepts ────────────────────────────
#[test]
fn test_latency_based_below_threshold_accepts() {
    let cfg = LoadSheddingConfig {
        policy: LoadSheddingPolicy::LatencyBased {
            max_queue_latency_ms: 200.0,
        },
        cooldown_ms: 0,
        ..LoadSheddingConfig::default()
    };
    let mut shedder = LoadShedder::new(cfg);
    let load = SystemLoad {
        estimated_queue_latency_ms: 100.0,
        ..zero_load()
    };
    let d = shedder.should_shed(2, &load);
    assert_eq!(d, SheddingDecision::Accept);
}

// ── 39. LatencyBased — above threshold drops ─────────────────────────────
#[test]
fn test_latency_based_above_threshold_drops() {
    let cfg = LoadSheddingConfig {
        policy: LoadSheddingPolicy::LatencyBased {
            max_queue_latency_ms: 100.0,
        },
        cooldown_ms: 0,
        ..LoadSheddingConfig::default()
    };
    let mut shedder = LoadShedder::new(cfg);
    let load = SystemLoad {
        estimated_queue_latency_ms: 500.0,
        ..zero_load()
    };
    let d = shedder.should_shed(2, &load);
    assert!(matches!(d, SheddingDecision::Drop { .. }));
}

// ── 40. SheddingStats::shed_rate — zero when no shedding ─────────────────
#[test]
fn test_shedding_stats_shed_rate_zero() {
    let stats = SheddingStats::default();
    assert_eq!(stats.shed_rate(), 0.0);
}

// ── 41. SheddingStats::shed_rate — correct fraction ──────────────────────
#[test]
fn test_shedding_stats_shed_rate_correct() {
    let stats = SheddingStats {
        total_requests: 10,
        total_shed: 4,
        total_degraded: 0,
        total_deferred: 0,
    };
    let rate = stats.shed_rate();
    assert!(
        (rate - 0.4).abs() < 1e-6,
        "shed_rate should be 0.4, got {rate}"
    );
}

// ── 42. SystemLoad::is_overloaded — false when below both thresholds ──────
#[test]
fn test_system_load_is_overloaded_false_below() {
    let load = SystemLoad {
        cpu_utilization: 0.5,
        memory_utilization: 0.5,
        ..zero_load()
    };
    assert!(!load.is_overloaded(0.8, 0.8));
}

// ── 43. SystemLoad::is_overloaded — true when cpu exceeds threshold ───────
#[test]
fn test_system_load_is_overloaded_true_cpu() {
    let load = SystemLoad {
        cpu_utilization: 0.9,
        memory_utilization: 0.1,
        ..zero_load()
    };
    assert!(load.is_overloaded(0.8, 0.9));
}

// ── 44. SystemLoad::is_overloaded — true when memory exceeds threshold ────
#[test]
fn test_system_load_is_overloaded_true_memory() {
    let load = SystemLoad {
        cpu_utilization: 0.1,
        memory_utilization: 0.95,
        ..zero_load()
    };
    assert!(load.is_overloaded(0.9, 0.9));
}

// ── 45. SystemLoad::overall_load_score — in [0.0, 1.0] ───────────────────
#[test]
fn test_overall_load_score_in_unit_interval() {
    let loads = [
        zero_load(),
        high_cpu_load(),
        high_mem_load(),
        SystemLoad {
            cpu_utilization: 1.0,
            memory_utilization: 1.0,
            queue_depth: 1000,
            ..zero_load()
        },
    ];
    for load in &loads {
        let score = load.overall_load_score();
        assert!(
            (0.0..=1.0).contains(&score),
            "overall_load_score {score} must be in [0, 1]"
        );
    }
}

// ── 46. SystemLoad::overall_load_score — zero load gives 0.0 ─────────────
#[test]
fn test_overall_load_score_zero_load() {
    let score = zero_load().overall_load_score();
    assert_eq!(score, 0.0);
}

// ── 47. SheddingDecision::Accept == Accept ────────────────────────────────
#[test]
fn test_shedding_decision_accept_equals() {
    assert_eq!(SheddingDecision::Accept, SheddingDecision::Accept);
}

// ── 48. SheddingDecision::Drop with different reasons differ ─────────────
#[test]
fn test_shedding_decision_drop_differs_from_accept() {
    let d = SheddingDecision::Drop {
        reason: SheddingReason::ProbabilisticDrop,
    };
    assert_ne!(d, SheddingDecision::Accept);
}

// ── 49. feedback — shed correct increases drop probability ────────────────
#[test]
fn test_feedback_correct_shed_increases_probability() {
    let mut shedder = LoadShedder::new(LoadSheddingConfig {
        policy: LoadSheddingPolicy::ProbabilisticDrop {
            drop_probability: 0.1,
        },
        ..LoadSheddingConfig::default()
    });
    shedder.current_drop_probability = 0.2;
    shedder.feedback(true, None); // assume correct, latency unknown → increase
    assert!(
        shedder.current_drop_probability >= 0.2,
        "correct shed should not decrease probability"
    );
}

// ── 50. SheddingStats — total_shed increments on Drop decision ────────────
#[test]
fn test_total_shed_increments_on_drop() {
    let cfg = LoadSheddingConfig {
        policy: LoadSheddingPolicy::ResourceBased {
            cpu_threshold: 0.5,
            memory_threshold: 0.9,
        },
        cooldown_ms: 0,
        ..LoadSheddingConfig::default()
    };
    let mut shedder = LoadShedder::new(cfg);
    let load = SystemLoad {
        cpu_utilization: 0.8,
        ..zero_load()
    };
    shedder.should_shed(2, &load);
    assert!(
        shedder.shedding_stats.total_shed >= 1,
        "total_shed must be >= 1 after a drop decision"
    );
}

// ── 51. GradualDegradation — low load accepts ─────────────────────────────
#[test]
fn test_gradual_degradation_low_load_accepts() {
    let cfg = LoadSheddingConfig {
        policy: LoadSheddingPolicy::GradualDegradation { min_quality: 0.1 },
        cooldown_ms: 0,
        ..LoadSheddingConfig::default()
    };
    let mut shedder = LoadShedder::new(cfg);
    let d = shedder.should_shed(2, &zero_load());
    assert_eq!(d, SheddingDecision::Accept);
}

// ── 52. LoadSheddingPolicy variants are not all equal ────────────────────
#[test]
fn test_load_shedding_policy_variants_differ() {
    let a = LoadSheddingPolicy::None;
    let b = LoadSheddingPolicy::ProbabilisticDrop {
        drop_probability: 0.5,
    };
    assert_ne!(a, b);
}
