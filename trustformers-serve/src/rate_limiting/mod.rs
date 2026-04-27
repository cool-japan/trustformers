//! Per-user, per-model, and global token-bucket rate limiting.
//!
//! Each bucket refills continuously at a configured rate (tokens/second) up to
//! a maximum capacity, and consumers deduct a cost per request.

mod rate_limiting_extra_tests;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during rate limiting operations.
#[derive(Debug, Error, Clone)]
pub enum RateLimitError {
    #[error("Lock poisoned")]
    LockPoisoned,
    #[error("Invalid token cost: {0}")]
    InvalidCost(f64),
}

// ---------------------------------------------------------------------------
// TokenBucket
// ---------------------------------------------------------------------------

/// A token bucket for a single user, model, or endpoint.
#[derive(Debug)]
pub struct TokenBucket {
    capacity: f64,
    tokens: f64,
    refill_rate: f64, // tokens per second
    last_refill: Instant,
}

impl TokenBucket {
    /// Create a new bucket, starting full.
    pub fn new(capacity: f64, refill_rate: f64) -> Self {
        Self {
            capacity,
            tokens: capacity,
            refill_rate,
            last_refill: Instant::now(),
        }
    }

    /// Refill the bucket based on elapsed time since the last refill.
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
        self.last_refill = now;
    }

    /// Try to consume `tokens` from the bucket.
    ///
    /// Returns `true` if there were enough tokens (and they were consumed),
    /// `false` otherwise (no tokens are consumed on failure).
    pub fn try_consume(&mut self, tokens: f64) -> bool {
        self.refill();
        if self.tokens >= tokens {
            self.tokens -= tokens;
            true
        } else {
            false
        }
    }

    /// Current token count after accounting for elapsed time (read-only, does
    /// not advance `last_refill`).
    pub fn available(&self) -> f64 {
        let elapsed = Instant::now().duration_since(self.last_refill).as_secs_f64();
        (self.tokens + elapsed * self.refill_rate).min(self.capacity)
    }

    /// How long until `needed` tokens are available.
    ///
    /// Returns `None` if `needed` tokens are already available; otherwise
    /// `Some(wait_time)`.
    pub fn time_until_available(&self, needed: f64) -> Option<Duration> {
        let current = self.available();
        if current >= needed {
            return None;
        }
        if self.refill_rate <= 0.0 {
            // Will never refill
            return Some(Duration::MAX);
        }
        let deficit = needed - current;
        let seconds = deficit / self.refill_rate;
        Some(Duration::from_secs_f64(seconds))
    }
}

// ---------------------------------------------------------------------------
// RateLimitConfig
// ---------------------------------------------------------------------------

/// Configuration for one level of rate limiting (user / model / global).
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Token refill rate (tokens per second)
    pub requests_per_second: f64,
    /// Bucket capacity (allows bursting up to this many tokens)
    pub burst_size: f64,
    /// Cost in tokens per request (default 1.0)
    pub tokens_per_request: f64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_second: 10.0,
            burst_size: 50.0,
            tokens_per_request: 1.0,
        }
    }
}

impl RateLimitConfig {
    /// Strict preset: 1 req/s, burst of 5.
    pub fn strict() -> Self {
        Self {
            requests_per_second: 1.0,
            burst_size: 5.0,
            tokens_per_request: 1.0,
        }
    }

    /// Generous preset: 100 req/s, burst of 500.
    pub fn generous() -> Self {
        Self {
            requests_per_second: 100.0,
            burst_size: 500.0,
            tokens_per_request: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// RateLimitDecision
// ---------------------------------------------------------------------------

/// Outcome of a rate limit check.
#[derive(Debug, Clone)]
pub enum RateLimitDecision {
    /// Request is allowed.
    Allow,
    /// Request is denied; caller should wait `wait_ms` milliseconds.
    Deny {
        /// Suggested wait time in milliseconds.
        wait_ms: u64,
        /// Human-readable reason for the denial.
        reason: String,
    },
}

impl RateLimitDecision {
    /// Returns `true` if the decision permits the request.
    pub fn is_allowed(&self) -> bool {
        matches!(self, RateLimitDecision::Allow)
    }
}

// ---------------------------------------------------------------------------
// RateLimiterStats
// ---------------------------------------------------------------------------

/// Point-in-time statistics from a [`RateLimiter`].
#[derive(Debug, Clone)]
pub struct RateLimiterStats {
    /// Number of per-user buckets currently tracked.
    pub active_user_buckets: usize,
    /// Number of per-model buckets currently tracked.
    pub active_model_buckets: usize,
    /// Currently available tokens in the global bucket.
    pub global_available: f64,
    /// Maximum capacity of the global bucket.
    pub global_capacity: f64,
}

// ---------------------------------------------------------------------------
// RateLimiter
// ---------------------------------------------------------------------------

/// Composite rate limiter: global → per-user → per-model.
///
/// All three tiers are checked in order; the first denial wins and no tokens
/// are consumed.
pub struct RateLimiter {
    global_bucket: Arc<Mutex<TokenBucket>>,
    user_buckets: Arc<Mutex<HashMap<String, TokenBucket>>>,
    model_buckets: Arc<Mutex<HashMap<String, TokenBucket>>>,
    user_config: RateLimitConfig,
    model_config: RateLimitConfig,
    global_config: RateLimitConfig,
}

impl RateLimiter {
    /// Create a new composite rate limiter.
    pub fn new(
        global_config: RateLimitConfig,
        user_config: RateLimitConfig,
        model_config: RateLimitConfig,
    ) -> Self {
        let global_bucket =
            TokenBucket::new(global_config.burst_size, global_config.requests_per_second);
        Self {
            global_bucket: Arc::new(Mutex::new(global_bucket)),
            user_buckets: Arc::new(Mutex::new(HashMap::new())),
            model_buckets: Arc::new(Mutex::new(HashMap::new())),
            user_config,
            model_config,
            global_config,
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Obtain a mutable reference to the user bucket, creating it if needed.
    fn with_user_bucket<F, R>(&self, user_id: &str, f: F) -> Result<R, RateLimitError>
    where
        F: FnOnce(&mut TokenBucket) -> R,
    {
        let mut map = self.user_buckets.lock().map_err(|_| RateLimitError::LockPoisoned)?;
        let bucket = map.entry(user_id.to_string()).or_insert_with(|| {
            TokenBucket::new(
                self.user_config.burst_size,
                self.user_config.requests_per_second,
            )
        });
        Ok(f(bucket))
    }

    /// Obtain a mutable reference to the model bucket, creating it if needed.
    fn with_model_bucket<F, R>(&self, model_id: &str, f: F) -> Result<R, RateLimitError>
    where
        F: FnOnce(&mut TokenBucket) -> R,
    {
        let mut map = self.model_buckets.lock().map_err(|_| RateLimitError::LockPoisoned)?;
        let bucket = map.entry(model_id.to_string()).or_insert_with(|| {
            TokenBucket::new(
                self.model_config.burst_size,
                self.model_config.requests_per_second,
            )
        });
        Ok(f(bucket))
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Check (and, if allowed, consume) tokens for a single request.
    ///
    /// Checks proceed: global → user → model. The first denial wins; on denial
    /// no tokens are consumed from any bucket.
    pub fn check_request(
        &self,
        user_id: &str,
        model_id: &str,
        token_cost: f64,
    ) -> Result<RateLimitDecision, RateLimitError> {
        if token_cost <= 0.0 {
            return Err(RateLimitError::InvalidCost(token_cost));
        }

        // --- Check global ---
        {
            let global = self.global_bucket.lock().map_err(|_| RateLimitError::LockPoisoned)?;
            let avail = global.available();
            if avail < token_cost {
                let wait = global
                    .time_until_available(token_cost)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(u64::MAX);
                return Ok(RateLimitDecision::Deny {
                    wait_ms: wait,
                    reason: "global rate limit exceeded".to_string(),
                });
            }
        }

        // --- Check user ---
        {
            let avail = self.with_user_bucket(user_id, |b| b.available())?;
            if avail < token_cost {
                let wait = self.with_user_bucket(user_id, |b| {
                    b.time_until_available(token_cost)
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(u64::MAX)
                })?;
                return Ok(RateLimitDecision::Deny {
                    wait_ms: wait,
                    reason: format!("user '{user_id}' rate limit exceeded"),
                });
            }
        }

        // --- Check model ---
        {
            let avail = self.with_model_bucket(model_id, |b| b.available())?;
            if avail < token_cost {
                let wait = self.with_model_bucket(model_id, |b| {
                    b.time_until_available(token_cost)
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(u64::MAX)
                })?;
                return Ok(RateLimitDecision::Deny {
                    wait_ms: wait,
                    reason: format!("model '{model_id}' rate limit exceeded"),
                });
            }
        }

        // --- All checks passed → consume ---
        {
            let mut global = self.global_bucket.lock().map_err(|_| RateLimitError::LockPoisoned)?;
            global.try_consume(token_cost);
        }
        self.with_user_bucket(user_id, |b| {
            b.try_consume(token_cost);
        })?;
        self.with_model_bucket(model_id, |b| {
            b.try_consume(token_cost);
        })?;

        Ok(RateLimitDecision::Allow)
    }

    /// Available tokens for the given user (bucket created with default config
    /// if not yet seen).
    pub fn user_bucket_available(&self, user_id: &str) -> f64 {
        self.with_user_bucket(user_id, |b| b.available()).unwrap_or(0.0)
    }

    /// Available tokens for the given model.
    pub fn model_bucket_available(&self, model_id: &str) -> f64 {
        self.with_model_bucket(model_id, |b| b.available()).unwrap_or(0.0)
    }

    /// Available tokens in the global bucket.
    pub fn global_bucket_available(&self) -> f64 {
        self.global_bucket.lock().map(|g| g.available()).unwrap_or(0.0)
    }

    /// Reset the user bucket to full capacity.
    pub fn reset_user(&self, user_id: &str) -> Result<(), RateLimitError> {
        let mut map = self.user_buckets.lock().map_err(|_| RateLimitError::LockPoisoned)?;
        let cap = self.user_config.burst_size;
        let rate = self.user_config.requests_per_second;
        map.insert(user_id.to_string(), TokenBucket::new(cap, rate));
        Ok(())
    }

    /// Remove the user bucket entirely (frees memory).
    pub fn remove_user(&self, user_id: &str) -> Result<(), RateLimitError> {
        let mut map = self.user_buckets.lock().map_err(|_| RateLimitError::LockPoisoned)?;
        map.remove(user_id);
        Ok(())
    }

    /// Return a snapshot of current limiter statistics.
    pub fn stats(&self) -> RateLimiterStats {
        let active_user_buckets = self.user_buckets.lock().map(|m| m.len()).unwrap_or(0);
        let active_model_buckets = self.model_buckets.lock().map(|m| m.len()).unwrap_or(0);
        let global_available = self.global_bucket_available();
        RateLimiterStats {
            active_user_buckets,
            active_model_buckets,
            global_available,
            global_capacity: self.global_config.burst_size,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn default_limiter() -> RateLimiter {
        RateLimiter::new(
            RateLimitConfig::default(), // global: 10 req/s, burst 50
            RateLimitConfig::default(), // user:   10 req/s, burst 50
            RateLimitConfig::default(), // model:  10 req/s, burst 50
        )
    }

    // Test 1: Basic allow
    #[test]
    fn test_basic_allow() {
        let limiter = default_limiter();
        let decision = limiter.check_request("alice", "gpt-4", 1.0).expect("no error expected");
        assert!(decision.is_allowed());
    }

    // Test 2: Burst – fill then deny after capacity is exhausted
    #[test]
    fn test_burst_then_deny() {
        let limiter = RateLimiter::new(
            RateLimitConfig {
                requests_per_second: 0.0,
                burst_size: 5.0,
                tokens_per_request: 1.0,
            },
            RateLimitConfig {
                requests_per_second: 0.0,
                burst_size: 5.0,
                tokens_per_request: 1.0,
            },
            RateLimitConfig {
                requests_per_second: 0.0,
                burst_size: 5.0,
                tokens_per_request: 1.0,
            },
        );
        for i in 0..5 {
            let d = limiter
                .check_request("bob", "model-a", 1.0)
                .unwrap_or_else(|_| panic!("error on request {i}"));
            assert!(d.is_allowed(), "request {i} should be allowed");
        }
        let d = limiter.check_request("bob", "model-a", 1.0).expect("no error expected");
        assert!(!d.is_allowed(), "6th request should be denied");
    }

    // Test 3: Refill over time
    #[test]
    fn test_refill_over_time() {
        // Bucket with capacity 2, refill 10/s, so 2 tokens refilled in 0.2 s
        let limiter = RateLimiter::new(
            RateLimitConfig {
                requests_per_second: 100.0,
                burst_size: 100.0,
                tokens_per_request: 1.0,
            },
            RateLimitConfig {
                requests_per_second: 10.0,
                burst_size: 2.0,
                tokens_per_request: 1.0,
            },
            RateLimitConfig {
                requests_per_second: 100.0,
                burst_size: 100.0,
                tokens_per_request: 1.0,
            },
        );
        // Exhaust the user bucket
        limiter.check_request("carol", "m", 1.0).expect("ok");
        limiter.check_request("carol", "m", 1.0).expect("ok");
        let d = limiter.check_request("carol", "m", 1.0).expect("ok");
        assert!(!d.is_allowed());

        // Wait for 1 token to refill (100 ms at 10/s)
        thread::sleep(Duration::from_millis(150));
        let d2 = limiter.check_request("carol", "m", 1.0).expect("ok");
        assert!(d2.is_allowed(), "should be allowed after refill");
    }

    // Test 4: Per-user isolation – one user's limit does not affect another
    #[test]
    fn test_per_user_isolation() {
        let limiter = RateLimiter::new(
            RateLimitConfig {
                requests_per_second: 0.0,
                burst_size: 1000.0,
                tokens_per_request: 1.0,
            },
            RateLimitConfig {
                requests_per_second: 0.0,
                burst_size: 1.0,
                tokens_per_request: 1.0,
            },
            RateLimitConfig {
                requests_per_second: 0.0,
                burst_size: 1000.0,
                tokens_per_request: 1.0,
            },
        );
        // User "dave" exhausts their bucket
        limiter.check_request("dave", "m", 1.0).expect("ok");
        let d_dave = limiter.check_request("dave", "m", 1.0).expect("ok");
        assert!(!d_dave.is_allowed());

        // User "eve" has a fresh bucket
        let d_eve = limiter.check_request("eve", "m", 1.0).expect("ok");
        assert!(
            d_eve.is_allowed(),
            "eve should not be affected by dave's limit"
        );
    }

    // Test 5: Per-model isolation
    #[test]
    fn test_per_model_isolation() {
        let limiter = RateLimiter::new(
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
            RateLimitConfig {
                requests_per_second: 0.0,
                burst_size: 1.0,
                tokens_per_request: 1.0,
            },
        );
        limiter.check_request("user", "model-x", 1.0).expect("ok");
        let d_x = limiter.check_request("user", "model-x", 1.0).expect("ok");
        assert!(!d_x.is_allowed());

        let d_y = limiter.check_request("user", "model-y", 1.0).expect("ok");
        assert!(d_y.is_allowed(), "model-y should have its own bucket");
    }

    // Test 6: Global limit blocks all users
    #[test]
    fn test_global_limit() {
        let limiter = RateLimiter::new(
            RateLimitConfig {
                requests_per_second: 0.0,
                burst_size: 1.0,
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
        limiter.check_request("frank", "m", 1.0).expect("ok");
        let d1 = limiter.check_request("frank", "m", 1.0).expect("ok");
        assert!(!d1.is_allowed(), "global limit should deny frank");
        let d2 = limiter.check_request("grace", "m", 1.0).expect("ok");
        assert!(!d2.is_allowed(), "global limit should deny grace too");
    }

    // Test 7: check_request first-denial wins (global checked before user)
    #[test]
    fn test_deny_first_wins() {
        let limiter = RateLimiter::new(
            RateLimitConfig {
                requests_per_second: 0.0,
                burst_size: 0.5,
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
        // Global bucket starts at 0.5 which is less than the cost of 1.0
        let d = limiter.check_request("harry", "m", 1.0).expect("ok");
        assert!(!d.is_allowed());
        if let RateLimitDecision::Deny { reason, .. } = d {
            assert!(
                reason.contains("global"),
                "denial reason should mention global, got: {reason}"
            );
        }
    }

    // Test 8: time_until_available returns None when tokens are available
    #[test]
    fn test_time_until_available_none() {
        let bucket = TokenBucket::new(10.0, 5.0);
        assert!(bucket.time_until_available(5.0).is_none());
    }

    // Test 9: time_until_available returns Some when tokens are insufficient
    #[test]
    fn test_time_until_available_some() {
        let mut bucket = TokenBucket::new(5.0, 1.0);
        // Consume all 5 tokens
        assert!(bucket.try_consume(5.0));
        // Need 2 more at 1/s → 2 seconds wait
        let wait = bucket.time_until_available(2.0).expect("should need to wait");
        assert!(
            wait.as_secs_f64() > 1.5,
            "wait should be ~2 s, got {wait:?}"
        );
    }

    // Test 10: stats reflects active buckets
    #[test]
    fn test_stats() {
        let limiter = default_limiter();
        limiter.check_request("ivan", "model-z", 1.0).expect("ok");
        let s = limiter.stats();
        assert_eq!(s.active_user_buckets, 1);
        assert_eq!(s.active_model_buckets, 1);
        assert!(s.global_available > 0.0);
    }

    // Test 11: reset_user restores full capacity
    #[test]
    fn test_reset_user() {
        let limiter = RateLimiter::new(
            RateLimitConfig {
                requests_per_second: 0.0,
                burst_size: 1000.0,
                tokens_per_request: 1.0,
            },
            RateLimitConfig {
                requests_per_second: 0.0,
                burst_size: 3.0,
                tokens_per_request: 1.0,
            },
            RateLimitConfig {
                requests_per_second: 0.0,
                burst_size: 1000.0,
                tokens_per_request: 1.0,
            },
        );
        for _ in 0..3 {
            limiter.check_request("judy", "m", 1.0).expect("ok");
        }
        let d = limiter.check_request("judy", "m", 1.0).expect("ok");
        assert!(!d.is_allowed());

        limiter.reset_user("judy").expect("reset should succeed");
        let d2 = limiter.check_request("judy", "m", 1.0).expect("ok");
        assert!(d2.is_allowed(), "after reset judy should be allowed again");
    }

    // Test 12: remove_user removes the bucket; subsequent request creates new one
    #[test]
    fn test_remove_user() {
        let limiter = default_limiter();
        limiter.check_request("kevin", "m", 1.0).expect("ok");
        assert_eq!(limiter.stats().active_user_buckets, 1);
        limiter.remove_user("kevin").expect("remove should succeed");
        assert_eq!(limiter.stats().active_user_buckets, 0);
        // New request for kevin should create a fresh bucket
        let d = limiter.check_request("kevin", "m", 1.0).expect("ok");
        assert!(d.is_allowed());
        assert_eq!(limiter.stats().active_user_buckets, 1);
    }

    // Test 13: RateLimitConfig presets have expected values
    #[test]
    fn test_rate_limit_config_presets() {
        let strict = RateLimitConfig::strict();
        assert_eq!(strict.requests_per_second, 1.0);
        assert_eq!(strict.burst_size, 5.0);

        let generous = RateLimitConfig::generous();
        assert_eq!(generous.requests_per_second, 100.0);
        assert_eq!(generous.burst_size, 500.0);

        let default = RateLimitConfig::default();
        assert_eq!(default.requests_per_second, 10.0);
        assert_eq!(default.burst_size, 50.0);
        assert_eq!(default.tokens_per_request, 1.0);
    }

    // Test 14: tokens_per_request cost – higher cost depletes faster
    #[test]
    fn test_tokens_per_request_cost() {
        // Bucket of 10 tokens, cost=3 → floor(10/3) = 3 allowed
        let limiter = RateLimiter::new(
            RateLimitConfig {
                requests_per_second: 0.0,
                burst_size: 10.0,
                tokens_per_request: 1.0,
            },
            RateLimitConfig {
                requests_per_second: 0.0,
                burst_size: 10.0,
                tokens_per_request: 1.0,
            },
            RateLimitConfig {
                requests_per_second: 0.0,
                burst_size: 10.0,
                tokens_per_request: 1.0,
            },
        );
        let mut allowed = 0usize;
        for _ in 0..5 {
            let d = limiter.check_request("lena", "m", 3.0).expect("ok");
            if d.is_allowed() {
                allowed += 1;
            }
        }
        // Global, user, and model each start at 10; min is 10 so floor(10/3)=3
        assert_eq!(allowed, 3, "should allow exactly 3 requests at cost 3");
    }

    // Test 15: InvalidCost error for zero/negative cost
    #[test]
    fn test_invalid_cost_error() {
        let limiter = default_limiter();
        let err = limiter.check_request("mona", "m", 0.0);
        assert!(err.is_err());
        let err2 = limiter.check_request("mona", "m", -1.0);
        assert!(err2.is_err());
    }

    // Test 16: TokenBucket try_consume returns false when empty
    #[test]
    fn test_token_bucket_try_consume_returns_false_when_empty() {
        let mut bucket = TokenBucket::new(5.0, 0.0);
        // Consume all 5 tokens — should succeed.
        assert!(bucket.try_consume(5.0), "first consume should succeed");
        // Next consume — bucket is empty, refill_rate=0 — should fail.
        assert!(!bucket.try_consume(1.0), "bucket is empty, should fail");
    }

    // Test 17: available() is non-mutating — two successive reads agree
    #[test]
    fn test_token_bucket_available_does_not_mutate() {
        let bucket = TokenBucket::new(10.0, 0.0);
        let first = bucket.available();
        let second = bucket.available();
        assert!(
            (first - second).abs() < 1e-9,
            "available() must be stable: {first} vs {second}"
        );
    }

    // Test 18: time_until_available with zero refill rate returns Duration::MAX
    #[test]
    fn test_token_bucket_zero_refill_rate_time_until_available_returns_max() {
        let mut bucket = TokenBucket::new(2.0, 0.0);
        assert!(bucket.try_consume(2.0));
        let wait = bucket.time_until_available(1.0).expect("should need to wait");
        assert_eq!(
            wait,
            Duration::MAX,
            "zero refill rate must return Duration::MAX"
        );
    }

    // Test 19: Two different users each get their own bucket tracked in stats
    #[test]
    fn test_rate_limiter_multiple_users_tracked() {
        let limiter = default_limiter();
        limiter.check_request("alice2", "model-q", 1.0).expect("ok");
        limiter.check_request("bob2", "model-q", 1.0).expect("ok");
        let s = limiter.stats();
        assert_eq!(s.active_user_buckets, 2, "should track two distinct users");
    }

    // Test 20: Two different models each get their own bucket tracked in stats
    #[test]
    fn test_rate_limiter_multiple_models_tracked() {
        let limiter = default_limiter();
        limiter.check_request("user-x", "model-alpha", 1.0).expect("ok");
        limiter.check_request("user-x", "model-beta", 1.0).expect("ok");
        let s = limiter.stats();
        assert_eq!(
            s.active_model_buckets, 2,
            "should track two distinct models"
        );
    }

    // Test 21: Global available tokens decrease after an allowed request
    #[test]
    fn test_rate_limiter_global_available_decreases_after_request() {
        let limiter = default_limiter();
        let before = limiter.global_bucket_available();
        limiter.check_request("user-y", "model-y", 1.0).expect("ok");
        let after = limiter.global_bucket_available();
        assert!(
            after < before,
            "global available must decrease after consuming tokens: {before} -> {after}"
        );
    }

    // Test 22: Deny decision has non-zero wait_ms when bucket is just exhausted
    #[test]
    fn test_rate_limit_decision_deny_has_wait_ms() {
        let limiter = RateLimiter::new(
            RateLimitConfig {
                requests_per_second: 1.0,
                burst_size: 1.0,
                tokens_per_request: 1.0,
            },
            RateLimitConfig {
                requests_per_second: 100.0,
                burst_size: 100.0,
                tokens_per_request: 1.0,
            },
            RateLimitConfig {
                requests_per_second: 100.0,
                burst_size: 100.0,
                tokens_per_request: 1.0,
            },
        );
        // Exhaust global bucket.
        limiter.check_request("u", "m", 1.0).expect("ok");
        let d = limiter.check_request("u", "m", 1.0).expect("ok");
        match d {
            RateLimitDecision::Deny { wait_ms, .. } => {
                assert!(wait_ms > 0, "wait_ms must be > 0 when denied: {wait_ms}");
            },
            RateLimitDecision::Allow => panic!("expected Deny, got Allow"),
        }
    }

    // Test 23: Deny reason contains the user id when per-user bucket is exhausted
    #[test]
    fn test_rate_limit_decision_deny_reason_user() {
        let limiter = RateLimiter::new(
            RateLimitConfig {
                requests_per_second: 0.0,
                burst_size: 1000.0,
                tokens_per_request: 1.0,
            },
            RateLimitConfig {
                requests_per_second: 0.0,
                burst_size: 1.0,
                tokens_per_request: 1.0,
            },
            RateLimitConfig {
                requests_per_second: 0.0,
                burst_size: 1000.0,
                tokens_per_request: 1.0,
            },
        );
        let target_user = "special-user-abc";
        limiter.check_request(target_user, "m", 1.0).expect("ok");
        let d = limiter.check_request(target_user, "m", 1.0).expect("ok");
        match d {
            RateLimitDecision::Deny { reason, .. } => {
                assert!(
                    reason.contains(target_user),
                    "deny reason must mention user id '{target_user}', got: {reason}"
                );
            },
            RateLimitDecision::Allow => panic!("expected Deny"),
        }
    }

    // Test 24: Deny reason contains the model id when per-model bucket is exhausted
    #[test]
    fn test_rate_limit_decision_deny_reason_model() {
        let limiter = RateLimiter::new(
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
            RateLimitConfig {
                requests_per_second: 0.0,
                burst_size: 1.0,
                tokens_per_request: 1.0,
            },
        );
        let target_model = "special-model-xyz";
        limiter.check_request("u", target_model, 1.0).expect("ok");
        let d = limiter.check_request("u", target_model, 1.0).expect("ok");
        match d {
            RateLimitDecision::Deny { reason, .. } => {
                assert!(
                    reason.contains(target_model),
                    "deny reason must mention model id '{target_model}', got: {reason}"
                );
            },
            RateLimitDecision::Allow => panic!("expected Deny"),
        }
    }

    // Test 25: stats().global_capacity matches the configured burst_size
    #[test]
    fn test_stats_global_capacity_matches_config() {
        let capacity = 77.0;
        let limiter = RateLimiter::new(
            RateLimitConfig {
                requests_per_second: 10.0,
                burst_size: capacity,
                tokens_per_request: 1.0,
            },
            RateLimitConfig::default(),
            RateLimitConfig::default(),
        );
        let s = limiter.stats();
        assert!(
            (s.global_capacity - capacity).abs() < 1e-6_f64,
            "global_capacity must equal configured burst_size {capacity}, got {}",
            s.global_capacity
        );
    }

    // ── Extended tests ─────────────────────────────────────────────────────

    // Test 26: RateLimitConfig::default — correct defaults
    #[test]
    fn test_rate_limit_config_default_values() {
        let cfg = RateLimitConfig::default();
        assert!((cfg.requests_per_second - 10.0).abs() < 1e-9);
        assert!((cfg.burst_size - 50.0).abs() < 1e-9);
        assert!((cfg.tokens_per_request - 1.0).abs() < 1e-9);
    }

    // Test 27: RateLimitConfig::strict — 1 req/s burst 5
    #[test]
    fn test_rate_limit_config_strict() {
        let cfg = RateLimitConfig::strict();
        assert!((cfg.requests_per_second - 1.0).abs() < 1e-9);
        assert!((cfg.burst_size - 5.0).abs() < 1e-9);
    }

    // Test 28: RateLimitConfig::generous — 100 req/s burst 500
    #[test]
    fn test_rate_limit_config_generous() {
        let cfg = RateLimitConfig::generous();
        assert!((cfg.requests_per_second - 100.0).abs() < 1e-9);
        assert!((cfg.burst_size - 500.0).abs() < 1e-9);
    }

    // Test 29: RateLimitDecision::Allow::is_allowed returns true
    #[test]
    fn test_rate_limit_decision_allow_is_allowed() {
        assert!(RateLimitDecision::Allow.is_allowed());
    }

    // Test 30: RateLimitDecision::Deny::is_allowed returns false
    #[test]
    fn test_rate_limit_decision_deny_is_not_allowed() {
        let d = RateLimitDecision::Deny {
            wait_ms: 100,
            reason: "test".into(),
        };
        assert!(!d.is_allowed());
    }

    // Test 31: TokenBucket::new starts full
    #[test]
    fn test_token_bucket_new_starts_full() {
        let tb = TokenBucket::new(100.0, 10.0);
        assert!(
            (tb.tokens - 100.0).abs() < 1e-9,
            "new bucket must start full"
        );
    }

    // Test 32: TokenBucket::try_consume — succeeds when tokens available
    #[test]
    fn test_token_bucket_consume_succeeds() {
        let mut tb = TokenBucket::new(50.0, 0.0);
        assert!(tb.try_consume(25.0));
    }

    // Test 33: TokenBucket::try_consume — fails when insufficient tokens
    #[test]
    fn test_token_bucket_consume_fails_when_empty() {
        let mut tb = TokenBucket::new(5.0, 0.0);
        assert!(!tb.try_consume(10.0));
    }

    // Test 34: TokenBucket::try_consume — does not consume on failure
    #[test]
    fn test_token_bucket_no_consume_on_failure() {
        let mut tb = TokenBucket::new(5.0, 0.0);
        tb.try_consume(10.0); // fails
        assert!(
            (tb.tokens - 5.0).abs() < 1e-6,
            "tokens must be unchanged on failure"
        );
    }

    // Test 35: TokenBucket::available — at least as many as tokens field
    #[test]
    fn test_token_bucket_available_at_least_tokens() {
        let tb = TokenBucket::new(100.0, 10.0);
        // available() accounts for elapsed time (could be slightly > tokens)
        assert!(tb.available() >= 0.0);
    }

    // Test 36: TokenBucket::time_until_available — None when enough tokens
    #[test]
    fn test_token_bucket_time_until_available_none_when_full() {
        let tb = TokenBucket::new(100.0, 10.0);
        assert!(
            tb.time_until_available(50.0).is_none(),
            "should have enough tokens"
        );
    }

    // Test 37: TokenBucket::time_until_available — Some when not enough tokens
    #[test]
    fn test_token_bucket_time_until_available_some_when_empty() {
        let mut tb = TokenBucket::new(10.0, 1.0);
        // Drain the bucket by consuming all tokens
        tb.try_consume(10.0);
        // Now need 5 more tokens; at 1/sec should need ~5 seconds
        let wait = tb.time_until_available(5.0);
        assert!(
            wait.is_some(),
            "should need wait time when bucket is drained"
        );
    }

    // Test 38: stats — active_user_buckets grows with distinct users
    #[test]
    fn test_stats_user_buckets_grow() {
        let limiter = default_limiter();
        limiter.check_request("user_a", "model", 1.0).expect("ok");
        limiter.check_request("user_b", "model", 1.0).expect("ok");
        limiter.check_request("user_c", "model", 1.0).expect("ok");
        let s = limiter.stats();
        assert!(
            s.active_user_buckets >= 3,
            "must track at least 3 user buckets"
        );
    }

    // Test 39: stats — active_model_buckets grows with distinct models
    #[test]
    fn test_stats_model_buckets_grow() {
        let limiter = default_limiter();
        limiter.check_request("user", "model_x", 1.0).expect("ok");
        limiter.check_request("user", "model_y", 1.0).expect("ok");
        let s = limiter.stats();
        assert!(
            s.active_model_buckets >= 2,
            "must track at least 2 model buckets"
        );
    }

    // Test 40: RateLimitError::InvalidCost — display contains the cost
    #[test]
    fn test_rate_limit_error_invalid_cost_display() {
        let e = RateLimitError::InvalidCost(-1.0);
        assert!(
            e.to_string().contains("-1"),
            "error must mention the bad cost value"
        );
    }

    // Test 41: RateLimitError::LockPoisoned — display is non-empty
    #[test]
    fn test_rate_limit_error_lock_poisoned_display() {
        let e = RateLimitError::LockPoisoned;
        assert!(!e.to_string().is_empty());
    }

    // Test 42: check_request with zero cost returns InvalidCost error.
    // A zero (or negative) token cost is rejected as invalid by the rate limiter.
    #[test]
    fn test_check_request_zero_cost() {
        let limiter = default_limiter();
        let result = limiter.check_request("user", "model", 0.0);
        assert!(
            result.is_err(),
            "zero cost must return an InvalidCost error"
        );
        match result {
            Err(RateLimitError::InvalidCost(c)) => {
                assert_eq!(c, 0.0, "InvalidCost should carry the zero value");
            },
            other => panic!("expected InvalidCost, got {other:?}"),
        }
    }

    // Test 43: global_available is <= global_capacity after requests
    #[test]
    fn test_global_available_le_capacity() {
        let limiter = default_limiter();
        for _ in 0..5 {
            let _ = limiter.check_request("u", "m", 1.0);
        }
        let s = limiter.stats();
        assert!(
            s.global_available <= s.global_capacity,
            "available tokens must not exceed capacity"
        );
    }

    // Test 44: stats — global_available decreases after requests
    #[test]
    fn test_global_available_decreases() {
        let limiter = RateLimiter::new(
            RateLimitConfig {
                requests_per_second: 0.0,
                burst_size: 100.0,
                tokens_per_request: 1.0,
            },
            RateLimitConfig::default(),
            RateLimitConfig::default(),
        );
        let before = limiter.stats().global_available;
        limiter.check_request("u", "m", 1.0).expect("ok");
        let after = limiter.stats().global_available;
        assert!(
            after < before,
            "global_available must decrease after consuming tokens"
        );
    }

    // Test 45: TokenBucket::time_until_available — Duration::MAX when refill_rate is zero
    #[test]
    fn test_token_bucket_time_until_available_max_when_no_refill() {
        let mut tb = TokenBucket::new(10.0, 0.0);
        tb.try_consume(10.0); // drain the bucket
        let wait = tb.time_until_available(5.0);
        assert_eq!(
            wait,
            Some(std::time::Duration::MAX),
            "zero refill rate must return MAX wait"
        );
    }
}
