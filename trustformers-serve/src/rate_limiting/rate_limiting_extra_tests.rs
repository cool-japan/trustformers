#![cfg(test)]
/// Extended tests for the rate limiting module.
use super::*;

fn tiny_limiter(global_burst: f64, user_burst: f64, model_burst: f64) -> RateLimiter {
    RateLimiter::new(
        RateLimitConfig {
            requests_per_second: 0.0,
            burst_size: global_burst,
            tokens_per_request: 1.0,
        },
        RateLimitConfig {
            requests_per_second: 0.0,
            burst_size: user_burst,
            tokens_per_request: 1.0,
        },
        RateLimitConfig {
            requests_per_second: 0.0,
            burst_size: model_burst,
            tokens_per_request: 1.0,
        },
    )
}

// ── 26. TokenBucket — starts full ─────────────────────────────────────────
#[test]
fn test_token_bucket_starts_full() {
    let bucket = TokenBucket::new(100.0, 10.0);
    let avail = bucket.available();
    assert!(
        (avail - 100.0).abs() < 1e-6,
        "bucket should start full at 100.0, got {avail}"
    );
}

// ── 27. TokenBucket::try_consume — returns true when tokens available ─────
#[test]
fn test_token_bucket_try_consume_success() {
    let mut bucket = TokenBucket::new(10.0, 1.0);
    assert!(
        bucket.try_consume(5.0),
        "should succeed with tokens available"
    );
}

// ── 28. TokenBucket::try_consume — returns false when depleted ────────────
#[test]
fn test_token_bucket_try_consume_depleted() {
    let mut bucket = TokenBucket::new(1.0, 0.0); // no refill
    bucket.try_consume(1.0); // use all
    assert!(!bucket.try_consume(0.01), "should fail when empty");
}

// ── 29. TokenBucket::try_consume — does not consume on failure ────────────
#[test]
fn test_token_bucket_no_consume_on_failure() {
    let mut bucket = TokenBucket::new(1.0, 0.0);
    let before = bucket.available();
    let _ = bucket.try_consume(10.0); // should fail; tokens unchanged
    let after = bucket.available();
    assert!(
        (after - before).abs() < 1e-6,
        "tokens must not change on failure: before={before} after={after}"
    );
}

// ── 30. TokenBucket::available — never exceeds capacity ───────────────────
#[test]
fn test_token_bucket_available_does_not_exceed_capacity() {
    let bucket = TokenBucket::new(50.0, 1000.0);
    let avail = bucket.available();
    assert!(
        avail <= 50.0,
        "available must not exceed capacity 50.0, got {avail}"
    );
}

// ── 31. TokenBucket::time_until_available — None when already full ────────
#[test]
fn test_time_until_available_none_when_full() {
    let bucket = TokenBucket::new(100.0, 10.0);
    assert!(
        bucket.time_until_available(1.0).is_none(),
        "should return None when bucket is full"
    );
}

// ── 32. RateLimitConfig::strict — correct values ───────────────────────────
#[test]
fn test_rate_limit_config_strict() {
    let c = RateLimitConfig::strict();
    assert!((c.requests_per_second - 1.0).abs() < 1e-9);
    assert!((c.burst_size - 5.0).abs() < 1e-9);
}

// ── 33. RateLimitConfig::generous — higher than default ───────────────────
#[test]
fn test_rate_limit_config_generous_higher_than_strict() {
    let g = RateLimitConfig::generous();
    let s = RateLimitConfig::strict();
    assert!(g.requests_per_second > s.requests_per_second);
    assert!(g.burst_size > s.burst_size);
}

// ── 34. RateLimitDecision::Allow::is_allowed returns true ─────────────────
#[test]
fn test_allow_decision_is_allowed_true() {
    let d = RateLimitDecision::Allow;
    assert!(d.is_allowed());
}

// ── 35. RateLimitDecision::Deny::is_allowed returns false ────────────────
#[test]
fn test_deny_decision_is_allowed_false() {
    let d = RateLimitDecision::Deny {
        wait_ms: 100,
        reason: "test".to_string(),
    };
    assert!(!d.is_allowed());
}

// ── 36. check_request — zero cost returns InvalidCost ────────────────────
#[test]
fn test_check_request_zero_cost_returns_error() {
    let limiter = RateLimiter::new(
        RateLimitConfig::default(),
        RateLimitConfig::default(),
        RateLimitConfig::default(),
    );
    let err = limiter.check_request("u", "m", 0.0).unwrap_err();
    assert!(matches!(err, RateLimitError::InvalidCost(_)));
}

// ── 37. check_request — negative cost returns InvalidCost ────────────────
#[test]
fn test_check_request_negative_cost_returns_error() {
    let limiter = RateLimiter::new(
        RateLimitConfig::default(),
        RateLimitConfig::default(),
        RateLimitConfig::default(),
    );
    let err = limiter.check_request("u", "m", -1.0).unwrap_err();
    assert!(matches!(err, RateLimitError::InvalidCost(_)));
}

// ── 38. check_request — global bucket empty → Deny ───────────────────────
#[test]
fn test_check_request_global_bucket_exhausted() {
    let limiter = tiny_limiter(1.0, 1000.0, 1000.0);
    limiter.check_request("u", "m", 1.0).expect("first ok");
    let d = limiter.check_request("u", "m", 1.0).expect("second ok");
    assert!(!d.is_allowed(), "should deny after global bucket is empty");
}

// ── 39. check_request — user bucket exhaustion ───────────────────────────
#[test]
fn test_check_request_user_bucket_exhausted() {
    let limiter = tiny_limiter(1000.0, 1.0, 1000.0);
    limiter.check_request("alice", "m", 1.0).expect("first ok");
    let d = limiter.check_request("alice", "m", 1.0).expect("second ok");
    assert!(!d.is_allowed(), "should deny after user bucket is empty");
}

// ── 40. check_request — model bucket exhaustion ──────────────────────────
#[test]
fn test_check_request_model_bucket_exhausted() {
    let limiter = tiny_limiter(1000.0, 1000.0, 1.0);
    limiter.check_request("u", "slow-model", 1.0).expect("first ok");
    let d = limiter.check_request("u", "slow-model", 1.0).expect("second ok");
    assert!(!d.is_allowed(), "should deny after model bucket is empty");
}

// ── 41. check_request — large cost against full bucket succeeds ──────────
#[test]
fn test_check_request_large_cost_when_bucket_full() {
    let limiter = RateLimiter::new(
        RateLimitConfig {
            requests_per_second: 0.0,
            burst_size: 100.0,
            tokens_per_request: 1.0,
        },
        RateLimitConfig {
            requests_per_second: 0.0,
            burst_size: 100.0,
            tokens_per_request: 1.0,
        },
        RateLimitConfig {
            requests_per_second: 0.0,
            burst_size: 100.0,
            tokens_per_request: 1.0,
        },
    );
    let d = limiter.check_request("u", "m", 50.0).expect("ok");
    assert!(
        d.is_allowed(),
        "50 tokens from a bucket of 100 should be allowed"
    );
}

// ── 42. stats — active_user_buckets starts at 0 ──────────────────────────
#[test]
fn test_stats_initial_user_buckets_zero() {
    let limiter = RateLimiter::new(
        RateLimitConfig::default(),
        RateLimitConfig::default(),
        RateLimitConfig::default(),
    );
    let s = limiter.stats();
    assert_eq!(s.active_user_buckets, 0);
}

// ── 43. stats — active_model_buckets starts at 0 ─────────────────────────
#[test]
fn test_stats_initial_model_buckets_zero() {
    let limiter = RateLimiter::new(
        RateLimitConfig::default(),
        RateLimitConfig::default(),
        RateLimitConfig::default(),
    );
    assert_eq!(limiter.stats().active_model_buckets, 0);
}

// ── 44. RateLimitError::InvalidCost displays token value ─────────────────
#[test]
fn test_rate_limit_error_invalid_cost_display() {
    let e = RateLimitError::InvalidCost(-5.0);
    assert!(
        e.to_string().contains("-5"),
        "error message must mention the invalid cost"
    );
}

// ── 45. RateLimiterStats — global_available > 0 when just created ─────────
#[test]
fn test_rate_limiter_stats_global_available_positive() {
    let limiter = RateLimiter::new(
        RateLimitConfig {
            requests_per_second: 0.0,
            burst_size: 50.0,
            tokens_per_request: 1.0,
        },
        RateLimitConfig::default(),
        RateLimitConfig::default(),
    );
    let s = limiter.stats();
    assert!(
        s.global_available > 0.0,
        "global_available must be > 0 for a fresh limiter"
    );
}

// ── 46. RateLimitConfig::default — non-zero values ────────────────────────
#[test]
fn test_rate_limit_config_default_non_zero() {
    let c = RateLimitConfig::default();
    assert!(c.requests_per_second > 0.0);
    assert!(c.burst_size > 0.0);
    assert!(c.tokens_per_request > 0.0);
}

// ── 47. multiple successful requests drain global bucket ──────────────────
#[test]
fn test_multiple_requests_drain_global_bucket() {
    let limiter = RateLimiter::new(
        RateLimitConfig {
            requests_per_second: 0.0,
            burst_size: 5.0,
            tokens_per_request: 1.0,
        },
        RateLimitConfig {
            requests_per_second: 0.0,
            burst_size: 1000.0,
            tokens_per_request: 1.0,
        },
        RateLimitConfig {
            requests_per_second: 0.0,
            burst_size: 1000.0,
            tokens_per_request: 1.0,
        },
    );
    for i in 0u32..5 {
        let d = limiter.check_request(&format!("u{i}"), "m", 1.0).expect("ok");
        assert!(d.is_allowed(), "request {i} should be allowed");
    }
    // 6th must be denied.
    let d = limiter.check_request("u6", "m", 1.0).expect("ok");
    assert!(!d.is_allowed(), "6th request should be denied");
}
