//! Rate Limiting Module
//!
//! Provides rate limiting functionality to prevent abuse and ensure fair resource usage.
//! Supports multiple algorithms including token bucket, sliding window, and fixed window.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use thiserror::Error;
use tokio::sync::RwLock;

/// Rate limiting algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RateLimitAlgorithm {
    /// Token bucket algorithm
    TokenBucket,
    /// Sliding window algorithm
    SlidingWindow,
    /// Fixed window algorithm
    FixedWindow,
}

/// Rate limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Algorithm to use
    pub algorithm: RateLimitAlgorithm,
    /// Maximum requests per window
    pub max_requests: usize,
    /// Time window duration in seconds
    pub window_seconds: u64,
    /// Maximum burst size (for token bucket)
    pub max_burst: Option<usize>,
    /// Refill rate per second (for token bucket)
    pub refill_rate: Option<f64>,
    /// Enable per-user rate limiting
    pub per_user_limits: bool,
    /// Enable per-IP rate limiting
    pub per_ip_limits: bool,
    /// Global rate limit (applies to all requests)
    pub global_limit: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            algorithm: RateLimitAlgorithm::TokenBucket,
            max_requests: 100,
            window_seconds: 60,
            max_burst: Some(10),
            refill_rate: Some(1.0),
            per_user_limits: true,
            per_ip_limits: true,
            global_limit: false,
        }
    }
}

/// Rate limit errors
#[derive(Debug, Error)]
pub enum RateLimitError {
    #[error("Rate limit exceeded for key: {key}, retry after {retry_after_seconds} seconds")]
    RateLimitExceeded {
        key: String,
        retry_after_seconds: u64,
    },

    #[error("Global rate limit exceeded, retry after {retry_after_seconds} seconds")]
    GlobalRateLimitExceeded { retry_after_seconds: u64 },

    #[error("Invalid rate limit configuration: {0}")]
    InvalidConfig(String),
}

/// Rate limit bucket for token bucket algorithm
#[derive(Debug, Clone)]
struct TokenBucket {
    tokens: f64,
    last_refill: Instant,
    max_tokens: f64,
    refill_rate: f64,
}

impl TokenBucket {
    fn new(max_tokens: f64, refill_rate: f64) -> Self {
        Self {
            tokens: max_tokens,
            last_refill: Instant::now(),
            max_tokens,
            refill_rate,
        }
    }

    fn try_consume(&mut self, tokens: f64) -> bool {
        self.refill();

        if self.tokens >= tokens {
            self.tokens -= tokens;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        let tokens_to_add = elapsed * self.refill_rate;

        self.tokens = (self.tokens + tokens_to_add).min(self.max_tokens);
        self.last_refill = now;
    }

    fn retry_after(&self) -> Duration {
        let tokens_needed = 1.0 - self.tokens;
        if tokens_needed <= 0.0 {
            Duration::from_secs(0)
        } else {
            Duration::from_secs_f64(tokens_needed / self.refill_rate)
        }
    }
}

/// Sliding window entry
#[derive(Debug, Clone)]
struct SlidingWindowEntry {
    requests: Vec<Instant>,
    window_duration: Duration,
    max_requests: usize,
}

impl SlidingWindowEntry {
    fn new(window_duration: Duration, max_requests: usize) -> Self {
        Self {
            requests: Vec::new(),
            window_duration,
            max_requests,
        }
    }

    fn try_add_request(&mut self) -> bool {
        let now = Instant::now();
        let cutoff = now - self.window_duration;

        // Remove old requests
        self.requests.retain(|&request_time| request_time > cutoff);

        if self.requests.len() < self.max_requests {
            self.requests.push(now);
            true
        } else {
            false
        }
    }

    fn retry_after(&self) -> Duration {
        if self.requests.is_empty() {
            return Duration::from_secs(0);
        }

        let oldest_request = *self.requests.first().unwrap();
        let retry_time = oldest_request + self.window_duration;
        let now = Instant::now();

        if retry_time > now {
            retry_time - now
        } else {
            Duration::from_secs(0)
        }
    }
}

/// Fixed window entry
#[derive(Debug, Clone)]
struct FixedWindowEntry {
    count: usize,
    window_start: Instant,
    window_duration: Duration,
    max_requests: usize,
}

impl FixedWindowEntry {
    fn new(window_duration: Duration, max_requests: usize) -> Self {
        Self {
            count: 0,
            window_start: Instant::now(),
            window_duration,
            max_requests,
        }
    }

    fn try_add_request(&mut self) -> bool {
        let now = Instant::now();

        // Check if we need to reset the window
        if now.duration_since(self.window_start) >= self.window_duration {
            self.count = 0;
            self.window_start = now;
        }

        if self.count < self.max_requests {
            self.count += 1;
            true
        } else {
            false
        }
    }

    fn retry_after(&self) -> Duration {
        let window_end = self.window_start + self.window_duration;
        let now = Instant::now();

        if window_end > now {
            window_end - now
        } else {
            Duration::from_secs(0)
        }
    }
}

/// Rate limiter state for different algorithms
#[derive(Debug)]
enum RateLimiterState {
    TokenBucket(TokenBucket),
    SlidingWindow(SlidingWindowEntry),
    FixedWindow(FixedWindowEntry),
}

impl RateLimiterState {
    fn new(config: &RateLimitConfig) -> Result<Self> {
        match config.algorithm {
            RateLimitAlgorithm::TokenBucket => {
                let max_tokens = config.max_burst.unwrap_or(config.max_requests) as f64;
                let refill_rate = config.refill_rate.unwrap_or(1.0);
                Ok(Self::TokenBucket(TokenBucket::new(max_tokens, refill_rate)))
            },
            RateLimitAlgorithm::SlidingWindow => {
                let window_duration = Duration::from_secs(config.window_seconds);
                Ok(Self::SlidingWindow(SlidingWindowEntry::new(
                    window_duration,
                    config.max_requests,
                )))
            },
            RateLimitAlgorithm::FixedWindow => {
                let window_duration = Duration::from_secs(config.window_seconds);
                Ok(Self::FixedWindow(FixedWindowEntry::new(
                    window_duration,
                    config.max_requests,
                )))
            },
        }
    }

    fn try_consume(&mut self) -> bool {
        match self {
            Self::TokenBucket(bucket) => bucket.try_consume(1.0),
            Self::SlidingWindow(window) => window.try_add_request(),
            Self::FixedWindow(window) => window.try_add_request(),
        }
    }

    fn retry_after(&self) -> Duration {
        match self {
            Self::TokenBucket(bucket) => bucket.retry_after(),
            Self::SlidingWindow(window) => window.retry_after(),
            Self::FixedWindow(window) => window.retry_after(),
        }
    }
}

/// Rate limiting service
#[derive(Debug)]
pub struct RateLimitService {
    config: RateLimitConfig,
    limiters: Arc<RwLock<HashMap<String, RateLimiterState>>>,
    global_limiter: Arc<RwLock<Option<RateLimiterState>>>,
}

impl RateLimitService {
    /// Create new rate limiting service
    pub fn new(config: RateLimitConfig) -> Result<Self> {
        let global_limiter =
            if config.global_limit { Some(RateLimiterState::new(&config)?) } else { None };

        Ok(Self {
            config,
            limiters: Arc::new(RwLock::new(HashMap::new())),
            global_limiter: Arc::new(RwLock::new(global_limiter)),
        })
    }

    /// Check if request is allowed for given key
    pub async fn check_rate_limit(&self, key: &str) -> Result<(), RateLimitError> {
        // Check global rate limit first
        if let Some(ref mut global) = *self.global_limiter.write().await {
            if !global.try_consume() {
                let retry_after = global.retry_after().as_secs();
                return Err(RateLimitError::GlobalRateLimitExceeded {
                    retry_after_seconds: retry_after,
                });
            }
        }

        // Check per-key rate limit
        let mut limiters = self.limiters.write().await;
        let limiter = limiters
            .entry(key.to_string())
            .or_insert_with(|| RateLimiterState::new(&self.config).unwrap());

        if !limiter.try_consume() {
            let retry_after = limiter.retry_after().as_secs();
            return Err(RateLimitError::RateLimitExceeded {
                key: key.to_string(),
                retry_after_seconds: retry_after,
            });
        }

        Ok(())
    }

    /// Generate rate limit key for request
    pub fn generate_key(&self, user_id: Option<&str>, ip_address: Option<&str>) -> String {
        let mut parts = Vec::new();

        if self.config.per_user_limits {
            if let Some(user) = user_id {
                parts.push(format!("user:{}", user));
            }
        }

        if self.config.per_ip_limits {
            if let Some(ip) = ip_address {
                parts.push(format!("ip:{}", ip));
            }
        }

        if parts.is_empty() {
            "global".to_string()
        } else {
            parts.join(":")
        }
    }

    /// Clean up expired limiters (should be called periodically)
    pub async fn cleanup_expired(&self) {
        let mut limiters = self.limiters.write().await;
        let cutoff = Instant::now() - Duration::from_secs(self.config.window_seconds * 2);

        limiters.retain(|_, limiter| match limiter {
            RateLimiterState::TokenBucket(bucket) => bucket.last_refill > cutoff,
            RateLimiterState::SlidingWindow(window) => window.requests.iter().any(|&t| t > cutoff),
            RateLimiterState::FixedWindow(window) => window.window_start > cutoff,
        });
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> RateLimitStats {
        let limiters = self.limiters.read().await;
        RateLimitStats {
            active_limiters: limiters.len(),
            algorithm: self.config.algorithm.clone(),
            max_requests: self.config.max_requests,
            window_seconds: self.config.window_seconds,
        }
    }
}

/// Rate limit statistics
#[derive(Debug, Clone, Serialize)]
pub struct RateLimitStats {
    pub active_limiters: usize,
    pub algorithm: RateLimitAlgorithm,
    pub max_requests: usize,
    pub window_seconds: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_token_bucket_rate_limit() {
        let config = RateLimitConfig {
            algorithm: RateLimitAlgorithm::TokenBucket,
            max_requests: 5,
            window_seconds: 10,
            max_burst: Some(5),
            refill_rate: Some(1.0),
            per_user_limits: true,
            per_ip_limits: false,
            global_limit: false,
        };

        let service = RateLimitService::new(config).unwrap();

        // Should allow up to max_burst requests
        for _ in 0..5 {
            assert!(service.check_rate_limit("user:test").await.is_ok());
        }

        // Next request should be rate limited
        assert!(service.check_rate_limit("user:test").await.is_err());
    }

    #[tokio::test]
    async fn test_sliding_window_rate_limit() {
        let config = RateLimitConfig {
            algorithm: RateLimitAlgorithm::SlidingWindow,
            max_requests: 3,
            window_seconds: 2,
            max_burst: None,
            refill_rate: None,
            per_user_limits: true,
            per_ip_limits: false,
            global_limit: false,
        };

        let service = RateLimitService::new(config).unwrap();

        // Should allow max_requests
        for _ in 0..3 {
            assert!(service.check_rate_limit("user:test").await.is_ok());
        }

        // Next request should be rate limited
        assert!(service.check_rate_limit("user:test").await.is_err());

        // After window expires, should allow requests again
        sleep(Duration::from_secs(3)).await;
        assert!(service.check_rate_limit("user:test").await.is_ok());
    }

    #[test]
    fn test_key_generation() {
        let config = RateLimitConfig {
            per_user_limits: true,
            per_ip_limits: true,
            ..Default::default()
        };
        let service = RateLimitService::new(config).unwrap();

        assert_eq!(
            service.generate_key(Some("user123"), Some("192.168.1.1")),
            "user:user123:ip:192.168.1.1"
        );

        assert_eq!(service.generate_key(Some("user123"), None), "user:user123");

        assert_eq!(
            service.generate_key(None, Some("192.168.1.1")),
            "ip:192.168.1.1"
        );
    }
}
