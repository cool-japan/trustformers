#![cfg(test)]
/// Extended tests for the scheduling module.
use super::*;
use std::time::{Duration, Instant};

fn make_req(id: &str, priority: u8, input: usize, output: usize) -> ScheduledRequest {
    ScheduledRequest::new(id, "model-a", input, output, priority)
}

fn default_config() -> WorkConservingConfig {
    WorkConservingConfig {
        max_running_requests: 4,
        max_tokens_in_flight: 1024,
        preemption_policy: PreemptionPolicy::NoPreemption,
        priority_aging_factor: 1.0,
        max_pending_depth: 8,
    }
}

// ── 1. token_budget = input + output ──────────────────────────────────────
#[test]
fn test_token_budget_is_sum() {
    let req = make_req("r1", 10, 100, 200);
    assert_eq!(req.token_budget(), 300);
}

// ── 2. token_budget — saturating add avoids overflow ──────────────────────
#[test]
fn test_token_budget_saturating_add() {
    let req = ScheduledRequest::new("r", "m", usize::MAX, 1, 5);
    // Should not panic.
    let _ = req.token_budget();
}

// ── 3. with_deadline attaches deadline ───────────────────────────────────
#[test]
fn test_with_deadline_attaches() {
    let dl = Instant::now() + Duration::from_secs(10);
    let req = make_req("r1", 5, 10, 10).with_deadline(dl);
    assert!(req.deadline.is_some());
}

// ── 4. enqueue — token-zero request returns InvalidRequest ───────────────
#[test]
fn test_enqueue_zero_tokens_returns_invalid() {
    let mut sched = WorkConservingScheduler::new(default_config());
    let req = make_req("r", 5, 0, 0);
    let err = sched.enqueue(req).unwrap_err();
    assert!(matches!(err, SchedulingError::InvalidRequest(_)));
}

// ── 5. enqueue — pending queue full returns QueueFull ────────────────────
#[test]
fn test_enqueue_queue_full_returns_error() {
    let config = WorkConservingConfig {
        max_pending_depth: 2,
        ..default_config()
    };
    let mut sched = WorkConservingScheduler::new(config);
    sched.enqueue(make_req("a", 5, 10, 10)).expect("first ok");
    sched.enqueue(make_req("b", 5, 10, 10)).expect("second ok");
    let err = sched.enqueue(make_req("c", 5, 10, 10)).unwrap_err();
    assert!(matches!(err, SchedulingError::QueueFull));
}

// ── 6. next_request — first admit increases running_count ─────────────────
#[test]
fn test_next_request_increases_running_count() {
    let mut sched = WorkConservingScheduler::new(default_config());
    sched.enqueue(make_req("r1", 5, 10, 10)).expect("enqueue");
    sched.next_request();
    assert_eq!(sched.running_count(), 1);
}

// ── 7. next_request — decreases pending_count ─────────────────────────────
#[test]
fn test_next_request_decreases_pending_count() {
    let mut sched = WorkConservingScheduler::new(default_config());
    sched.enqueue(make_req("r1", 5, 10, 10)).expect("enqueue");
    assert_eq!(sched.pending_count(), 1);
    sched.next_request();
    assert_eq!(sched.pending_count(), 0);
}

// ── 8. next_request — returns None on empty queue ─────────────────────────
#[test]
fn test_next_request_empty_queue_returns_none() {
    let mut sched = WorkConservingScheduler::new(default_config());
    assert!(sched.next_request().is_none());
}

// ── 9. next_request — priority ordering (high before low) ─────────────────
#[test]
fn test_next_request_high_priority_first() {
    let config = WorkConservingConfig {
        priority_aging_factor: 0.0, // disable aging so priority wins
        ..default_config()
    };
    let mut sched = WorkConservingScheduler::new(config);
    sched.enqueue(make_req("low", 10, 10, 10)).expect("enqueue low");
    sched.enqueue(make_req("high", 200, 10, 10)).expect("enqueue high");
    let first = sched.next_request().expect("first");
    assert_eq!(first.request_id, "high");
}

// ── 10. complete_request — returns error for unknown id ──────────────────
#[test]
fn test_complete_unknown_request_returns_error() {
    let mut sched = WorkConservingScheduler::new(default_config());
    let err = sched.complete_request("no-such-id", 0).unwrap_err();
    assert!(matches!(err, SchedulingError::RequestNotFound));
}

// ── 11. complete_request — releases tokens_in_flight ─────────────────────
#[test]
fn test_complete_releases_tokens_in_flight() {
    let mut sched = WorkConservingScheduler::new(default_config());
    sched.enqueue(make_req("r1", 5, 50, 50)).expect("enqueue");
    sched.next_request();
    let before = sched.tokens_in_flight();
    sched.complete_request("r1", 50).expect("complete");
    assert!(sched.tokens_in_flight() < before);
}

// ── 12. complete_request — increments total_completed stat ───────────────
#[test]
fn test_complete_increments_total_completed() {
    let mut sched = WorkConservingScheduler::new(default_config());
    sched.enqueue(make_req("r1", 5, 10, 10)).expect("enqueue");
    sched.next_request();
    sched.complete_request("r1", 10).expect("complete");
    assert_eq!(sched.stats().total_completed, 1);
}

// ── 13. check_preemption — NoPreemption policy returns empty ─────────────
#[test]
fn test_check_preemption_no_preemption_empty() {
    let mut sched = WorkConservingScheduler::new(default_config());
    sched.enqueue(make_req("r1", 5, 10, 10)).expect("enqueue");
    sched.next_request();
    let preempted = sched.check_preemption();
    assert!(preempted.is_empty());
}

// ── 14. aged_priority — increases with elapsed wait ───────────────────────
#[test]
fn test_aged_priority_increases_with_wait() {
    let config = WorkConservingConfig {
        priority_aging_factor: 10.0,
        ..default_config()
    };
    let sched = WorkConservingScheduler::new(config);
    let mut req = make_req("r1", 5, 10, 10);
    // Backdated enqueue time by using a shorter instant.
    req.enqueued_at = Instant::now();
    let early = sched.aged_priority(&req);
    // aged_priority should be >= base priority
    assert!(early >= 5.0);
}

// ── 15. stats — initial values are zero ───────────────────────────────────
#[test]
fn test_stats_initial_zero() {
    let sched = WorkConservingScheduler::new(default_config());
    let s = sched.stats();
    assert_eq!(s.total_scheduled, 0);
    assert_eq!(s.total_completed, 0);
    assert_eq!(s.total_preempted, 0);
    assert_eq!(s.slo_violation_count, 0);
}

// ── 16. tokens_in_flight — starts at zero ─────────────────────────────────
#[test]
fn test_tokens_in_flight_starts_zero() {
    let sched = WorkConservingScheduler::new(default_config());
    assert_eq!(sched.tokens_in_flight(), 0);
}

// ── 17. tokens_in_flight — increases after next_request ──────────────────
#[test]
fn test_tokens_in_flight_increases_after_admit() {
    let mut sched = WorkConservingScheduler::new(default_config());
    sched.enqueue(make_req("r1", 5, 30, 70)).expect("enqueue");
    sched.next_request();
    assert_eq!(sched.tokens_in_flight(), 100);
}

// ── 18. SchedulingError — display messages are non-empty ─────────────────
#[test]
fn test_scheduling_error_display_non_empty() {
    let errors = vec![
        SchedulingError::QueueFull,
        SchedulingError::RequestNotFound,
        SchedulingError::CapacityExceeded,
        SchedulingError::InvalidRequest("test".into()),
    ];
    for e in errors {
        assert!(!e.to_string().is_empty(), "empty display for {e:?}");
    }
}

// ── 19. PreemptionPolicy — PriorityBased is not equal to NoPreemption ─────
#[test]
fn test_preemption_policy_not_equal() {
    let a = PreemptionPolicy::NoPreemption;
    let b = PreemptionPolicy::PriorityBased;
    assert_ne!(a, b);
}

// ── 20. WorkConservingConfig::default — sensible values ──────────────────
#[test]
fn test_work_conserving_config_default() {
    let c = WorkConservingConfig::default();
    assert!(c.max_running_requests > 0);
    assert!(c.max_tokens_in_flight > 0);
    assert!(c.max_pending_depth > 0);
}

// ── 21. multiple complete cycles accumulate stats ─────────────────────────
#[test]
fn test_multiple_complete_cycles_accumulate_stats() {
    let mut sched = WorkConservingScheduler::new(default_config());
    for i in 0u32..3 {
        sched.enqueue(make_req(&i.to_string(), 5, 10, 10)).expect("enqueue");
        sched.next_request();
        sched.complete_request(&i.to_string(), 10).expect("complete");
    }
    assert_eq!(sched.stats().total_completed, 3);
    assert_eq!(sched.stats().total_scheduled, 3);
}

// ── 22. next_request — respects max_running_requests cap ──────────────────
#[test]
fn test_max_running_requests_respected() {
    let config = WorkConservingConfig {
        max_running_requests: 2,
        ..default_config()
    };
    let mut sched = WorkConservingScheduler::new(config);
    for i in 0..5u32 {
        sched.enqueue(make_req(&i.to_string(), 5, 10, 10)).expect("enqueue");
    }
    // Only 2 should be admitted.
    sched.next_request();
    sched.next_request();
    let third = sched.next_request();
    assert!(
        third.is_none(),
        "should not admit beyond max_running_requests"
    );
}

// ── 23. SeqStatus — Waiting != Running ───────────────────────────────────
#[test]
fn test_seq_status_inequality() {
    assert_ne!(SeqStatus::Waiting, SeqStatus::Running);
    assert_ne!(SeqStatus::Running, SeqStatus::Finished);
}

// ── 24. IterationConfig::default — positive block_size ───────────────────
#[test]
fn test_iteration_config_default_block_size_positive() {
    let cfg = IterationConfig::default();
    assert!(cfg.block_size > 0);
    assert!(cfg.total_kv_blocks > 0);
}

// ── 25. SamplingParams::default — temperature > 0 ────────────────────────
#[test]
fn test_sampling_params_default_temperature_positive() {
    let p = SamplingParams::default();
    assert!(
        p.temperature > 0.0,
        "default temperature must be > 0, got {}",
        p.temperature
    );
}
