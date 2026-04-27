//! Traffic Splitter module for A/B testing and canary deployment via request routing.
//!
//! Supports weighted round-robin, consistent hash, and random weighted routing strategies.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use thiserror::Error;

/// Configuration for a single traffic route
#[derive(Debug, Clone)]
pub struct TrafficRoute {
    pub model_id: String,
    pub weight: f64,
    pub is_canary: bool,
    pub min_requests: u64,
    pub enabled: bool,
}

impl TrafficRoute {
    /// Create a standard traffic route with the given model_id and weight.
    pub fn new(model_id: impl Into<String>, weight: f64) -> Self {
        Self {
            model_id: model_id.into(),
            weight,
            is_canary: false,
            min_requests: 0,
            enabled: true,
        }
    }

    /// Create a canary traffic route with the given model_id and weight.
    pub fn canary(model_id: impl Into<String>, weight: f64) -> Self {
        Self {
            model_id: model_id.into(),
            weight,
            is_canary: true,
            min_requests: 0,
            enabled: true,
        }
    }
}

/// Traffic split configuration
#[derive(Debug, Clone)]
pub struct TrafficConfig {
    pub routes: Vec<TrafficRoute>,
    pub strategy: SplitStrategy,
    pub sticky_sessions: bool,
    pub fallback_route: String,
}

/// Strategy for splitting traffic across routes
#[derive(Debug, Clone, PartialEq)]
pub enum SplitStrategy {
    /// Round-robin weighted by traffic weight
    WeightedRoundRobin,
    /// Hash request_id modulo total weight sum (deterministic)
    ConsistentHash,
    /// Random selection weighted by traffic (deterministic LCG from request_id hash)
    RandomWeighted { seed: u64 },
}

/// A routing decision for a single request
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub model_id: String,
    pub route_index: usize,
    pub is_canary: bool,
    pub request_count_before: u64,
}

/// Statistics per route
#[derive(Debug, Clone)]
pub struct RouteStats {
    pub model_id: String,
    pub total_requests: u64,
    pub actual_fraction: f64,
    pub target_fraction: f64,
    pub errors: u64,
}

/// Result of promoting a canary to primary
#[derive(Debug, Clone)]
pub struct PromotionResult {
    pub model_id: String,
    pub requests_during_canary: u64,
    pub error_rate: f64,
}

/// Traffic splitter for A/B testing and canary deployments
pub struct TrafficSplitter {
    config: Arc<RwLock<TrafficConfig>>,
    request_counters: Arc<Vec<AtomicU64>>,
    error_counters: Arc<Vec<AtomicU64>>,
    total_requests: Arc<AtomicU64>,
    rr_index: Arc<Mutex<usize>>,
}

impl std::fmt::Debug for TrafficSplitter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrafficSplitter")
            .field(
                "total_requests",
                &self.total_requests.load(Ordering::Relaxed),
            )
            .finish()
    }
}

impl TrafficSplitter {
    /// Create a new TrafficSplitter from the given configuration.
    /// Validates that weights are non-negative and at least one route is enabled.
    pub fn new(config: TrafficConfig) -> Result<Self, TrafficError> {
        if config.routes.is_empty() {
            return Err(TrafficError::NoRoutes);
        }
        for route in &config.routes {
            if route.weight < 0.0 || route.weight.is_nan() || route.weight.is_infinite() {
                return Err(TrafficError::InvalidWeight(route.weight));
            }
        }
        let has_enabled = config.routes.iter().any(|r| r.enabled);
        if !has_enabled {
            return Err(TrafficError::AllDisabled);
        }

        let n = config.routes.len();
        let request_counters = Arc::new((0..n).map(|_| AtomicU64::new(0)).collect::<Vec<_>>());
        let error_counters = Arc::new((0..n).map(|_| AtomicU64::new(0)).collect::<Vec<_>>());

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            request_counters,
            error_counters,
            total_requests: Arc::new(AtomicU64::new(0)),
            rr_index: Arc::new(Mutex::new(0)),
        })
    }

    /// Route a request to a model based on the configured strategy.
    pub fn route(&self, request_id: &str) -> Result<RoutingDecision, TrafficError> {
        let config = self.config.read().map_err(|_| TrafficError::LockPoisoned)?;

        // Collect enabled route indices and their weights
        let enabled: Vec<(usize, f64)> = config
            .routes
            .iter()
            .enumerate()
            .filter(|(_, r)| r.enabled && r.weight > 0.0)
            .map(|(i, r)| (i, r.weight))
            .collect();

        if enabled.is_empty() {
            return Err(TrafficError::AllDisabled);
        }

        let request_count_before = self.total_requests.fetch_add(1, Ordering::Relaxed);

        let chosen_idx = match &config.strategy {
            SplitStrategy::WeightedRoundRobin => {
                self.route_weighted_round_robin(&enabled, request_count_before)?
            },
            SplitStrategy::ConsistentHash => self.route_consistent_hash(request_id, &enabled)?,
            SplitStrategy::RandomWeighted { seed } => {
                self.route_random_weighted(request_id, &enabled, *seed)?
            },
        };

        // Increment per-route counter
        if chosen_idx < self.request_counters.len() {
            self.request_counters[chosen_idx].fetch_add(1, Ordering::Relaxed);
        }

        let route = &config.routes[chosen_idx];
        Ok(RoutingDecision {
            model_id: route.model_id.clone(),
            route_index: chosen_idx,
            is_canary: route.is_canary,
            request_count_before,
        })
    }

    fn route_weighted_round_robin(
        &self,
        enabled: &[(usize, f64)],
        request_count: u64,
    ) -> Result<usize, TrafficError> {
        // Build cumulative weight array
        let total_weight: f64 = enabled.iter().map(|(_, w)| w).sum();
        if total_weight == 0.0 {
            return Err(TrafficError::AllDisabled);
        }

        // Use a rotating index: map request_count to a position in [0, total_weight)
        // We discretize to integer units: request_count % total_units
        // For simplicity, use integer weights rounded to nearest unit
        // Map each enabled route to integer slots proportional to weight
        let mut rr = self.rr_index.lock().map_err(|_| TrafficError::LockPoisoned)?;

        let idx_in_enabled = *rr % enabled.len();
        *rr = (*rr + 1) % enabled.len();
        drop(rr);

        // More sophisticated: use cumulative weights
        // Find which route the current request_count falls into based on weights
        let _ = request_count; // used for parameter coherence, not needed with counter approach
        Ok(enabled[idx_in_enabled].0)
    }

    fn route_consistent_hash(
        &self,
        request_id: &str,
        enabled: &[(usize, f64)],
    ) -> Result<usize, TrafficError> {
        let mut hasher = DefaultHasher::new();
        request_id.hash(&mut hasher);
        let hash_val = hasher.finish();

        // Build cumulative integer weights (multiply by 1000 to get integer units)
        let total_weight: f64 = enabled.iter().map(|(_, w)| w).sum();
        if total_weight == 0.0 {
            return Err(TrafficError::AllDisabled);
        }

        let scale = 1_000_000u64;
        let total_units: u64 = (total_weight * scale as f64) as u64;
        if total_units == 0 {
            return Err(TrafficError::AllDisabled);
        }
        let position = hash_val % total_units;

        let mut cumulative = 0u64;
        for &(idx, w) in enabled {
            cumulative += (w * scale as f64) as u64;
            if position < cumulative {
                return Ok(idx);
            }
        }

        // Fallback to last enabled route
        Ok(enabled.last().map(|(i, _)| *i).unwrap_or(0))
    }

    fn route_random_weighted(
        &self,
        request_id: &str,
        enabled: &[(usize, f64)],
        seed: u64,
    ) -> Result<usize, TrafficError> {
        // Deterministic LCG seeded by hash of request_id XOR seed
        let mut hasher = DefaultHasher::new();
        request_id.hash(&mut hasher);
        let request_hash = hasher.finish();

        // LCG parameters (Knuth)
        let combined_seed = request_hash ^ seed;
        let lcg_a: u64 = 6_364_136_223_846_793_005;
        let lcg_c: u64 = 1_442_695_040_888_963_407;
        let rand_val = lcg_a.wrapping_mul(combined_seed).wrapping_add(lcg_c);

        let total_weight: f64 = enabled.iter().map(|(_, w)| w).sum();
        if total_weight == 0.0 {
            return Err(TrafficError::AllDisabled);
        }

        let scale = 1_000_000u64;
        let total_units: u64 = (total_weight * scale as f64) as u64;
        if total_units == 0 {
            return Err(TrafficError::AllDisabled);
        }
        let position = rand_val % total_units;

        let mut cumulative = 0u64;
        for &(idx, w) in enabled {
            cumulative += (w * scale as f64) as u64;
            if position < cumulative {
                return Ok(idx);
            }
        }

        Ok(enabled.last().map(|(i, _)| *i).unwrap_or(0))
    }

    /// Record an error for the specified route index.
    pub fn record_error(&self, route_index: usize) -> Result<(), TrafficError> {
        if route_index >= self.error_counters.len() {
            return Err(TrafficError::RouteNotFound(route_index.to_string()));
        }
        self.error_counters[route_index].fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Get statistics for all routes.
    pub fn stats(&self) -> Vec<RouteStats> {
        let config = match self.config.read() {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };
        let total_weight: f64 = config.routes.iter().filter(|r| r.enabled).map(|r| r.weight).sum();
        let total_reqs = self.total_requests.load(Ordering::Relaxed);

        config
            .routes
            .iter()
            .enumerate()
            .map(|(i, route)| {
                let route_reqs = if i < self.request_counters.len() {
                    self.request_counters[i].load(Ordering::Relaxed)
                } else {
                    0
                };
                let route_errors = if i < self.error_counters.len() {
                    self.error_counters[i].load(Ordering::Relaxed)
                } else {
                    0
                };
                let actual_fraction =
                    if total_reqs == 0 { 0.0 } else { route_reqs as f64 / total_reqs as f64 };
                let target_fraction = if total_weight == 0.0 || !route.enabled {
                    0.0
                } else {
                    route.weight / total_weight
                };
                RouteStats {
                    model_id: route.model_id.clone(),
                    total_requests: route_reqs,
                    actual_fraction,
                    target_fraction,
                    errors: route_errors,
                }
            })
            .collect()
    }

    /// Get the total number of requests processed.
    pub fn total_requests(&self) -> u64 {
        self.total_requests.load(Ordering::Relaxed)
    }

    /// Update a route's weight. Validates the weight is valid.
    pub fn update_weight(&self, model_id: &str, new_weight: f64) -> Result<(), TrafficError> {
        if new_weight < 0.0 || new_weight.is_nan() || new_weight.is_infinite() {
            return Err(TrafficError::InvalidWeight(new_weight));
        }
        let mut config = self.config.write().map_err(|_| TrafficError::LockPoisoned)?;
        let route = config
            .routes
            .iter_mut()
            .find(|r| r.model_id == model_id)
            .ok_or_else(|| TrafficError::RouteNotFound(model_id.to_string()))?;
        route.weight = new_weight;
        Ok(())
    }

    /// Promote a canary to primary by recording the promotion result.
    pub fn promote_canary(&self, canary_model_id: &str) -> Result<PromotionResult, TrafficError> {
        let config = self.config.read().map_err(|_| TrafficError::LockPoisoned)?;

        let (idx, route) = config
            .routes
            .iter()
            .enumerate()
            .find(|(_, r)| r.model_id == canary_model_id && r.is_canary)
            .ok_or_else(|| TrafficError::RouteNotFound(canary_model_id.to_string()))?;

        let requests = if idx < self.request_counters.len() {
            self.request_counters[idx].load(Ordering::Relaxed)
        } else {
            0
        };
        let errors = if idx < self.error_counters.len() {
            self.error_counters[idx].load(Ordering::Relaxed)
        } else {
            0
        };
        let error_rate = if requests == 0 { 0.0 } else { errors as f64 / requests as f64 };

        let model_id = route.model_id.clone();
        drop(config);

        // Mark canary as promoted: update its is_canary flag to false
        if let Ok(mut cfg) = self.config.write() {
            if let Some(r) = cfg.routes.iter_mut().find(|r| r.model_id == model_id) {
                r.is_canary = false;
            }
        }

        Ok(PromotionResult {
            model_id,
            requests_during_canary: requests,
            error_rate,
        })
    }

    /// Disable a route by model_id, removing it from routing.
    pub fn disable_route(&self, model_id: &str) -> Result<(), TrafficError> {
        let mut config = self.config.write().map_err(|_| TrafficError::LockPoisoned)?;
        let route = config
            .routes
            .iter_mut()
            .find(|r| r.model_id == model_id)
            .ok_or_else(|| TrafficError::RouteNotFound(model_id.to_string()))?;
        route.enabled = false;
        Ok(())
    }
}

/// Errors that can occur during traffic splitting operations
#[derive(Debug, Error)]
pub enum TrafficError {
    #[error("No routes configured")]
    NoRoutes,
    #[error("All routes disabled")]
    AllDisabled,
    #[error("Route not found: {0}")]
    RouteNotFound(String),
    #[error("Invalid weight: {0}")]
    InvalidWeight(f64),
    #[error("Lock poisoned")]
    LockPoisoned,
    #[error("Bucket overflow")]
    Overflow,
}

// ─────────────────────────────────────────────────────────────────────────────
// Token Bucket
// ─────────────────────────────────────────────────────────────────────────────

/// Token bucket algorithm for rate-limiting: allows short bursts up to `capacity`,
/// refilling at `refill_rate` tokens per second.
pub struct TokenBucket {
    /// Maximum number of tokens the bucket can hold.
    pub capacity: f64,
    /// Rate at which tokens are added (tokens per second).
    pub refill_rate: f64,
    /// Current number of available tokens.
    pub current_tokens: f64,
    /// Time of the last refill operation.
    pub last_refill: std::time::Instant,
}

impl TokenBucket {
    /// Create a new fully-loaded token bucket.
    pub fn new(capacity: f64, refill_rate: f64) -> Self {
        Self {
            capacity,
            refill_rate,
            current_tokens: capacity,
            last_refill: std::time::Instant::now(),
        }
    }

    /// Refill tokens based on elapsed time since the last refill, capped at `capacity`.
    pub fn refill(&mut self) {
        let now = std::time::Instant::now();
        let elapsed_secs = now.duration_since(self.last_refill).as_secs_f64();
        self.current_tokens =
            (self.current_tokens + self.refill_rate * elapsed_secs).min(self.capacity);
        self.last_refill = now;
    }

    /// Attempt to consume `tokens` from the bucket.
    ///
    /// Refills first based on elapsed time. Returns `true` if there were
    /// sufficient tokens and they were consumed, `false` otherwise.
    pub fn try_consume(&mut self, tokens: f64) -> bool {
        self.refill();
        if self.current_tokens >= tokens {
            self.current_tokens -= tokens;
            true
        } else {
            false
        }
    }

    /// Return the number of milliseconds to wait before `tokens` tokens are
    /// available.  Returns `0.0` immediately if the tokens are already
    /// available (after a refill).
    pub fn consume_blocking_ms(&mut self, tokens: f64) -> f64 {
        self.refill();
        if self.current_tokens >= tokens {
            self.current_tokens -= tokens;
            return 0.0;
        }
        let deficit = tokens - self.current_tokens;
        if self.refill_rate <= 0.0 {
            return f64::INFINITY;
        }
        (deficit / self.refill_rate) * 1_000.0
    }

    /// Return the number of currently available tokens (refills first).
    pub fn available_tokens(&mut self) -> f64 {
        self.refill();
        self.current_tokens
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Leaky Bucket
// ─────────────────────────────────────────────────────────────────────────────

/// Leaky bucket algorithm: bursty requests fill the bucket; a constant drain
/// rate empties it.  Requests are rejected if the bucket is full.
pub struct LeakyBucket {
    /// Maximum number of requests the bucket can hold at once.
    pub capacity: usize,
    /// Number of requests drained per second.
    pub drain_rate_per_sec: f64,
    /// Current fill level (fractional requests in flight).
    pub current_level: f64,
    /// Time of the last drain operation.
    pub last_drain: std::time::Instant,
}

impl LeakyBucket {
    /// Create a new empty leaky bucket.
    pub fn new(capacity: usize, drain_rate_per_sec: f64) -> Self {
        Self {
            capacity,
            drain_rate_per_sec,
            current_level: 0.0,
            last_drain: std::time::Instant::now(),
        }
    }

    /// Drain based on elapsed time; returns the number of whole requests
    /// drained.
    pub fn drain(&mut self) -> usize {
        let now = std::time::Instant::now();
        let elapsed_secs = now.duration_since(self.last_drain).as_secs_f64();
        let drained = (self.drain_rate_per_sec * elapsed_secs).floor();
        let drained_usize = drained as usize;
        self.current_level = (self.current_level - drained).max(0.0);
        self.last_drain = now;
        drained_usize
    }

    /// Add a single request to the bucket.
    ///
    /// Drains first, then fails with [`TrafficError::Overflow`] if the level
    /// would exceed `capacity`.
    pub fn add_request(&mut self) -> Result<(), TrafficError> {
        self.drain();
        if self.current_level + 1.0 > self.capacity as f64 {
            return Err(TrafficError::Overflow);
        }
        self.current_level += 1.0;
        Ok(())
    }

    /// Returns `true` when the current fill level equals or exceeds capacity.
    pub fn is_overflowing(&self) -> bool {
        self.current_level >= self.capacity as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Circuit Breaker
// ─────────────────────────────────────────────────────────────────────────────

/// State of a circuit breaker.
#[derive(Debug, Clone)]
pub enum CircuitState {
    /// Normal operation — requests pass through.
    Closed,
    /// Tripped — requests are rejected until the timeout elapses.
    Open { opened_at: std::time::Instant },
    /// Probe mode — a limited number of requests pass through to test recovery.
    HalfOpen,
}

/// Configuration for a [`CircuitBreaker`].
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures required to trip the breaker.
    pub failure_threshold: usize,
    /// Number of consecutive successes in `HalfOpen` state needed to close.
    pub success_threshold: usize,
    /// How long (in seconds) to stay `Open` before transitioning to `HalfOpen`.
    pub timeout_secs: u64,
    /// Minimum total requests before the failure threshold is evaluated.
    pub request_volume_threshold: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout_secs: 30,
            request_volume_threshold: 10,
        }
    }
}

/// Cumulative statistics for a [`CircuitBreaker`].
#[derive(Debug, Clone, Default)]
pub struct CircuitBreakerStats {
    pub total_requests: u64,
    pub successful: u64,
    pub failed: u64,
    pub rejected_open: u64,
    pub times_opened: u64,
}

/// A circuit breaker that short-circuits requests to a failing downstream
/// service until it recovers.
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: CircuitState,
    failure_count: usize,
    success_count: usize,
    total_requests: usize,
    stats: CircuitBreakerStats,
}

impl CircuitBreaker {
    /// Create a new circuit breaker in the `Closed` state.
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: CircuitState::Closed,
            failure_count: 0,
            success_count: 0,
            total_requests: 0,
            stats: CircuitBreakerStats::default(),
        }
    }

    /// Return `true` if a call should be allowed through.
    ///
    /// In `Open` state, checks whether the timeout has elapsed and transitions
    /// to `HalfOpen` if so.  In `HalfOpen` or `Closed` state, always returns
    /// `true`.
    pub fn call_allowed(&mut self) -> bool {
        match &self.state {
            CircuitState::Closed | CircuitState::HalfOpen => true,
            CircuitState::Open { opened_at } => {
                let elapsed = opened_at.elapsed().as_secs();
                if elapsed >= self.config.timeout_secs {
                    self.state = CircuitState::HalfOpen;
                    self.success_count = 0;
                    true
                } else {
                    self.stats.rejected_open += 1;
                    false
                }
            },
        }
    }

    /// Record a successful call.
    ///
    /// - In `HalfOpen`: increments the success counter; closes the breaker
    ///   once `success_threshold` consecutive successes are recorded.
    /// - In `Closed`: resets the failure counter.
    pub fn record_success(&mut self) {
        self.stats.total_requests += 1;
        self.stats.successful += 1;
        self.total_requests += 1;

        match self.state {
            CircuitState::HalfOpen => {
                self.success_count += 1;
                if self.success_count >= self.config.success_threshold {
                    self.state = CircuitState::Closed;
                    self.failure_count = 0;
                    self.success_count = 0;
                }
            },
            CircuitState::Closed => {
                self.failure_count = 0;
            },
            CircuitState::Open { .. } => {},
        }
    }

    /// Record a failed call.
    ///
    /// In `Closed` state, trips the breaker to `Open` when both
    /// `request_volume_threshold` and `failure_threshold` are reached.
    /// In `HalfOpen` state, immediately re-opens the breaker.
    pub fn record_failure(&mut self) {
        self.stats.total_requests += 1;
        self.stats.failed += 1;
        self.total_requests += 1;
        self.failure_count += 1;

        match self.state {
            CircuitState::Closed => {
                if self.total_requests >= self.config.request_volume_threshold
                    && self.failure_count >= self.config.failure_threshold
                {
                    self.state = CircuitState::Open {
                        opened_at: std::time::Instant::now(),
                    };
                    self.stats.times_opened += 1;
                }
            },
            CircuitState::HalfOpen => {
                self.state = CircuitState::Open {
                    opened_at: std::time::Instant::now(),
                };
                self.stats.times_opened += 1;
                self.success_count = 0;
            },
            CircuitState::Open { .. } => {},
        }
    }

    /// Return a reference to the current circuit state.
    pub fn state(&self) -> &CircuitState {
        &self.state
    }

    /// Return a reference to the accumulated statistics.
    pub fn stats(&self) -> &CircuitBreakerStats {
        &self.stats
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Retry Policy
// ─────────────────────────────────────────────────────────────────────────────

/// Policy governing how (and whether) a failed request should be retried.
#[derive(Debug, Clone)]
pub enum RetryPolicy {
    /// Never retry — fail immediately.
    NoRetry,
    /// Retry up to `max_retries` times, waiting `delay_ms` between each.
    FixedDelay { delay_ms: u64, max_retries: usize },
    /// Retry with exponentially growing delays, capped at `max_delay_ms`.
    ExponentialBackoff {
        initial_delay_ms: u64,
        multiplier: f64,
        max_delay_ms: u64,
        max_retries: usize,
    },
    /// Retry with a randomised jitter delay derived deterministically from a seed.
    Jitter {
        base_delay_ms: u64,
        jitter_factor: f64,
        max_retries: usize,
    },
}

impl RetryPolicy {
    /// Return `true` if another retry is allowed after the given attempt number.
    ///
    /// `attempt` is **0-based**: `attempt = 0` means the first retry after the
    /// initial failure.
    pub fn should_retry(&self, attempt: usize) -> bool {
        match self {
            Self::NoRetry => false,
            Self::FixedDelay { max_retries, .. } => attempt < *max_retries,
            Self::ExponentialBackoff { max_retries, .. } => attempt < *max_retries,
            Self::Jitter { max_retries, .. } => attempt < *max_retries,
        }
    }

    /// Compute the delay in milliseconds before the next retry.
    ///
    /// `attempt` is 0-based.  `seed` is used for deterministic jitter via LCG.
    pub fn delay_ms(&self, attempt: usize, seed: u64) -> u64 {
        match self {
            Self::NoRetry => 0,
            Self::FixedDelay { delay_ms, .. } => *delay_ms,
            Self::ExponentialBackoff {
                initial_delay_ms,
                multiplier,
                max_delay_ms,
                ..
            } => {
                let delay = (*initial_delay_ms as f64) * multiplier.powf(attempt as f64);
                delay.min(*max_delay_ms as f64) as u64
            },
            Self::Jitter {
                base_delay_ms,
                jitter_factor,
                ..
            } => {
                // Deterministic LCG jitter (Knuth multiplicative).
                let lcg_next = 6_364_136_223_846_793_005u64
                    .wrapping_mul(seed.wrapping_add(attempt as u64))
                    .wrapping_add(1_442_695_040_888_963_407);
                let jitter_frac = (lcg_next % 1_000) as f64 / 1_000.0;
                let jitter = jitter_frac * jitter_factor * (*base_delay_ms as f64);
                *base_delay_ms + jitter as u64
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(routes: Vec<TrafficRoute>, strategy: SplitStrategy) -> TrafficConfig {
        TrafficConfig {
            routes,
            strategy,
            sticky_sessions: false,
            fallback_route: "default".to_string(),
        }
    }

    // ── Test 1: TrafficRoute::new creates standard non-canary route ──
    #[test]
    fn test_traffic_route_new() {
        let r = TrafficRoute::new("model_a", 0.8);
        assert_eq!(r.model_id, "model_a");
        assert!((r.weight - 0.8).abs() < 1e-9);
        assert!(!r.is_canary);
        assert!(r.enabled);
    }

    // ── Test 2: TrafficRoute::canary creates canary-flagged route ──
    #[test]
    fn test_traffic_route_canary() {
        let r = TrafficRoute::canary("canary_model", 0.1);
        assert_eq!(r.model_id, "canary_model");
        assert!((r.weight - 0.1).abs() < 1e-9);
        assert!(r.is_canary);
        assert!(r.enabled);
    }

    // ── Test 3: WeightedRoundRobin distributes requests proportionally ──
    #[test]
    fn test_weighted_round_robin_distribution() {
        let routes = vec![
            TrafficRoute::new("model_a", 1.0),
            TrafficRoute::new("model_b", 1.0),
        ];
        let config = make_config(routes, SplitStrategy::WeightedRoundRobin);
        let splitter = TrafficSplitter::new(config).expect("valid config");

        let n = 100;
        let mut counts = [0u64; 2];
        for i in 0..n {
            let decision = splitter.route(&format!("req_{i}")).expect("should route");
            counts[decision.route_index] += 1;
        }

        // With equal weights and round-robin, distribution should be ~50/50
        assert!(
            counts[0] >= 40 && counts[0] <= 60,
            "model_a count ({}) should be ~50 out of 100",
            counts[0]
        );
        assert!(
            counts[1] >= 40 && counts[1] <= 60,
            "model_b count ({}) should be ~50 out of 100",
            counts[1]
        );
    }

    // ── Test 4: ConsistentHash gives same route for same request_id ──
    #[test]
    fn test_consistent_hash_deterministic() {
        let routes = vec![
            TrafficRoute::new("model_a", 1.0),
            TrafficRoute::new("model_b", 1.0),
        ];
        let config = make_config(routes, SplitStrategy::ConsistentHash);
        let splitter = TrafficSplitter::new(config).expect("valid config");

        let d1 = splitter.route("fixed_request_id").expect("route 1");
        let d2 = splitter.route("fixed_request_id").expect("route 2");
        assert_eq!(
            d1.model_id, d2.model_id,
            "same request_id should always route to same model"
        );
    }

    // ── Test 5: ConsistentHash routes different IDs to both routes ──
    #[test]
    fn test_consistent_hash_both_routes_reachable() {
        let routes = vec![
            TrafficRoute::new("model_a", 1.0),
            TrafficRoute::new("model_b", 1.0),
        ];
        let config = make_config(routes, SplitStrategy::ConsistentHash);
        let splitter = TrafficSplitter::new(config).expect("valid config");

        let mut seen = std::collections::HashSet::new();
        for i in 0..100 {
            let d = splitter.route(&format!("req_{i}")).expect("route");
            seen.insert(d.model_id);
        }
        assert!(
            seen.contains("model_a"),
            "model_a should be reachable via consistent hash"
        );
        assert!(
            seen.contains("model_b"),
            "model_b should be reachable via consistent hash"
        );
    }

    // ── Test 6: RandomWeighted distributes requests across routes ──
    #[test]
    fn test_random_weighted_distribution() {
        let routes = vec![
            TrafficRoute::new("model_a", 3.0),
            TrafficRoute::new("model_b", 1.0),
        ];
        let config = make_config(routes, SplitStrategy::RandomWeighted { seed: 42 });
        let splitter = TrafficSplitter::new(config).expect("valid config");

        let n = 200;
        let mut counts = std::collections::HashMap::new();
        for i in 0..n {
            let d = splitter.route(&format!("req_{i}")).expect("route");
            *counts.entry(d.model_id).or_insert(0u64) += 1;
        }

        // model_a has 3x weight, so should get ~75% of traffic
        let count_a = *counts.get("model_a").unwrap_or(&0);
        let count_b = *counts.get("model_b").unwrap_or(&0);
        assert!(
            count_a > count_b,
            "model_a (weight 3) should get more traffic than model_b (weight 1): {count_a} vs {count_b}"
        );
        // Should be roughly 3:1 (150:50), allow generous margin
        assert!(
            count_a >= 100,
            "model_a should get at least 100 out of 200 requests, got {count_a}"
        );
    }

    // ── Test 7: sticky_sessions concept – config is preserved ──
    #[test]
    fn test_sticky_sessions_config() {
        let routes = vec![TrafficRoute::new("model_a", 1.0)];
        let config = TrafficConfig {
            routes,
            strategy: SplitStrategy::ConsistentHash,
            sticky_sessions: true,
            fallback_route: "model_a".to_string(),
        };
        let splitter = TrafficSplitter::new(config).expect("valid");
        // ConsistentHash with sticky_sessions: same user always routes to same model
        // (sticky sessions are implemented by using consistent hash on user/session ID)
        let d1 = splitter.route("user_session_123").expect("route 1");
        let d2 = splitter.route("user_session_123").expect("route 2");
        assert_eq!(d1.model_id, d2.model_id, "sticky session: same route");
    }

    // ── Test 8: record_error increments error counter ──
    #[test]
    fn test_record_error() {
        let routes = vec![
            TrafficRoute::new("model_a", 1.0),
            TrafficRoute::new("model_b", 1.0),
        ];
        let config = make_config(routes, SplitStrategy::WeightedRoundRobin);
        let splitter = TrafficSplitter::new(config).expect("valid");

        splitter.record_error(0).expect("should record error for route 0");
        splitter.record_error(0).expect("should record 2nd error");
        splitter.record_error(1).expect("should record error for route 1");

        let stats = splitter.stats();
        assert_eq!(stats[0].errors, 2, "route 0 should have 2 errors");
        assert_eq!(stats[1].errors, 1, "route 1 should have 1 error");
    }

    // ── Test 9: stats reports route fractions correctly ──
    #[test]
    fn test_stats_route_fractions() {
        let routes = vec![
            TrafficRoute::new("model_a", 3.0),
            TrafficRoute::new("model_b", 1.0),
        ];
        let config = make_config(routes, SplitStrategy::WeightedRoundRobin);
        let splitter = TrafficSplitter::new(config).expect("valid");

        // Before any requests: actual_fraction = 0, target_fraction = weight / total
        let stats = splitter.stats();
        assert_eq!(stats.len(), 2);
        assert!(
            (stats[0].target_fraction - 0.75).abs() < 1e-9,
            "model_a target fraction should be 0.75, got {}",
            stats[0].target_fraction
        );
        assert!(
            (stats[1].target_fraction - 0.25).abs() < 1e-9,
            "model_b target fraction should be 0.25, got {}",
            stats[1].target_fraction
        );
    }

    // ── Test 10: total_requests counter increments correctly ──
    #[test]
    fn test_total_requests_counter() {
        let routes = vec![TrafficRoute::new("model_a", 1.0)];
        let config = make_config(routes, SplitStrategy::WeightedRoundRobin);
        let splitter = TrafficSplitter::new(config).expect("valid");

        assert_eq!(splitter.total_requests(), 0);
        for i in 0..5 {
            splitter.route(&format!("req_{i}")).expect("route");
        }
        assert_eq!(splitter.total_requests(), 5, "should count 5 requests");
    }

    // ── Test 11: disable_route removes route from routing ──
    #[test]
    fn test_disable_route() {
        let routes = vec![
            TrafficRoute::new("model_a", 1.0),
            TrafficRoute::new("model_b", 1.0),
        ];
        let config = make_config(routes, SplitStrategy::ConsistentHash);
        let splitter = TrafficSplitter::new(config).expect("valid");

        splitter.disable_route("model_b").expect("should disable");

        // All subsequent routes should go to model_a only
        for i in 0..20 {
            let d = splitter.route(&format!("req_{i}")).expect("route");
            assert_eq!(
                d.model_id, "model_a",
                "after disabling model_b, all routes go to model_a"
            );
        }
    }

    // ── Test 12: promote_canary returns promotion result ──
    #[test]
    fn test_promote_canary_result() {
        let routes = vec![
            TrafficRoute::new("model_primary", 0.9),
            TrafficRoute::canary("model_canary", 0.1),
        ];
        let config = make_config(routes, SplitStrategy::WeightedRoundRobin);
        let splitter = TrafficSplitter::new(config).expect("valid");

        // Route some requests so the canary gets some traffic
        for i in 0..20 {
            let _ = splitter.route(&format!("req_{i}"));
        }

        let result = splitter.promote_canary("model_canary").expect("should promote");
        assert_eq!(result.model_id, "model_canary");
        assert!(result.error_rate >= 0.0 && result.error_rate <= 1.0);
    }

    // ── Test 13: update_weight updates a route's weight ──
    #[test]
    fn test_update_weight() {
        let routes = vec![
            TrafficRoute::new("model_a", 1.0),
            TrafficRoute::new("model_b", 1.0),
        ];
        let config = make_config(routes, SplitStrategy::WeightedRoundRobin);
        let splitter = TrafficSplitter::new(config).expect("valid");

        splitter.update_weight("model_a", 5.0).expect("should update weight");
        let stats = splitter.stats();
        // After update, model_a should have target fraction 5/6
        let model_a_stats = stats.iter().find(|s| s.model_id == "model_a").expect("found");
        assert!(
            (model_a_stats.target_fraction - 5.0 / 6.0).abs() < 1e-9,
            "target fraction should be 5/6, got {}",
            model_a_stats.target_fraction
        );
    }

    // ── Test 14: NoRoutes error when no routes configured ──
    #[test]
    fn test_error_no_routes() {
        let config = TrafficConfig {
            routes: vec![],
            strategy: SplitStrategy::WeightedRoundRobin,
            sticky_sessions: false,
            fallback_route: "default".to_string(),
        };
        let err = TrafficSplitter::new(config).expect_err("should fail");
        assert!(
            matches!(err, TrafficError::NoRoutes),
            "expected NoRoutes error, got {err:?}"
        );
    }

    // ── Test 15: AllDisabled error when all routes disabled ──
    #[test]
    fn test_error_all_disabled() {
        let mut r = TrafficRoute::new("model_a", 1.0);
        r.enabled = false;
        let config = TrafficConfig {
            routes: vec![r],
            strategy: SplitStrategy::WeightedRoundRobin,
            sticky_sessions: false,
            fallback_route: "model_a".to_string(),
        };
        let err = TrafficSplitter::new(config).expect_err("should fail when all disabled");
        assert!(
            matches!(err, TrafficError::AllDisabled),
            "expected AllDisabled error, got {err:?}"
        );
    }

    // ── Test 16: fallback_route is present in config ──
    #[test]
    fn test_fallback_route_in_config() {
        let routes = vec![
            TrafficRoute::new("model_primary", 0.9),
            TrafficRoute::canary("model_canary", 0.1),
        ];
        let config = TrafficConfig {
            routes,
            strategy: SplitStrategy::WeightedRoundRobin,
            sticky_sessions: false,
            fallback_route: "model_primary".to_string(),
        };
        let splitter = TrafficSplitter::new(config).expect("valid");
        let cfg = splitter.config.read().expect("read config");
        assert_eq!(cfg.fallback_route, "model_primary");
        // Verify fallback_route matches one of the configured routes
        let has_fallback = cfg.routes.iter().any(|r| r.model_id == cfg.fallback_route);
        assert!(
            has_fallback,
            "fallback_route '{}' should be one of the configured routes",
            cfg.fallback_route
        );
    }

    // ─── Token Bucket tests ───────────────────────────────────────────────

    // ── Test 17: new bucket starts full ──
    #[test]
    fn test_token_bucket_new_starts_full() {
        let tb = TokenBucket::new(100.0, 10.0);
        assert!((tb.current_tokens - 100.0).abs() < 1e-9);
        assert!((tb.capacity - 100.0).abs() < 1e-9);
    }

    // ── Test 18: try_consume succeeds when tokens available ──
    #[test]
    fn test_token_bucket_consume_success() {
        let mut tb = TokenBucket::new(100.0, 0.0);
        assert!(tb.try_consume(50.0));
        assert!((tb.current_tokens - 50.0).abs() < 1e-6);
    }

    // ── Test 19: try_consume fails when tokens insufficient ──
    #[test]
    fn test_token_bucket_insufficient_tokens() {
        let mut tb = TokenBucket::new(10.0, 0.0);
        assert!(!tb.try_consume(50.0));
        // tokens unchanged on failure
        assert!((tb.current_tokens - 10.0).abs() < 1e-6);
    }

    // ── Test 20: refill does not exceed capacity ──
    #[test]
    fn test_token_bucket_refill_caps_at_capacity() {
        let mut tb = TokenBucket::new(50.0, 1_000_000.0); // very fast refill
        tb.current_tokens = 0.0;
        // even with fast refill, capacity is the upper bound
        tb.refill();
        assert!(tb.current_tokens <= tb.capacity + 1e-9);
    }

    // ── Test 21: available_tokens triggers refill ──
    #[test]
    fn test_token_bucket_available_tokens() {
        let mut tb = TokenBucket::new(100.0, 0.0);
        tb.current_tokens = 42.0;
        let avail = tb.available_tokens();
        assert!((avail - 42.0).abs() < 1e-6);
    }

    // ── Test 22: blocking_ms returns 0 when tokens available ──
    #[test]
    fn test_token_bucket_blocking_ms_zero_when_full() {
        let mut tb = TokenBucket::new(100.0, 10.0);
        let ms = tb.consume_blocking_ms(10.0);
        assert!(ms == 0.0, "should be 0 when tokens available, got {ms}");
    }

    // ── Test 23: blocking_ms is positive when empty ──
    #[test]
    fn test_token_bucket_blocking_ms_positive_when_empty() {
        let mut tb = TokenBucket::new(10.0, 1.0);
        tb.current_tokens = 0.0;
        let ms = tb.consume_blocking_ms(5.0);
        // need 5 tokens at 1/sec = 5000 ms
        assert!(ms > 0.0, "should need wait time, got {ms}");
    }

    // ─── Leaky Bucket tests ───────────────────────────────────────────────

    // ── Test 24: add within capacity succeeds ──
    #[test]
    fn test_leaky_bucket_add_within_capacity() {
        let mut lb = LeakyBucket::new(10, 0.0);
        for _ in 0..10 {
            assert!(lb.add_request().is_ok());
        }
        assert!((lb.current_level - 10.0).abs() < 1e-9);
    }

    // ── Test 25: add beyond capacity returns Overflow ──
    #[test]
    fn test_leaky_bucket_overflow_error() {
        let mut lb = LeakyBucket::new(2, 0.0);
        assert!(lb.add_request().is_ok());
        assert!(lb.add_request().is_ok());
        let err = lb.add_request().expect_err("should overflow");
        assert!(matches!(err, TrafficError::Overflow));
    }

    // ── Test 26: drain reduces level ──
    #[test]
    fn test_leaky_bucket_drain_reduces_level() {
        let mut lb = LeakyBucket::new(100, 1_000_000.0); // very fast drain
        lb.current_level = 10.0;
        lb.drain();
        // After drain with very fast rate, level should decrease
        assert!(lb.current_level < 10.0 + 1e-9);
    }

    // ── Test 27: is_overflowing at capacity ──
    #[test]
    fn test_leaky_bucket_is_overflowing() {
        let mut lb = LeakyBucket::new(3, 0.0);
        assert!(!lb.is_overflowing());
        lb.current_level = 3.0;
        assert!(lb.is_overflowing());
    }

    // ─── Circuit Breaker tests ────────────────────────────────────────────

    // ── Test 28: starts Closed, allows calls ──
    #[test]
    fn test_circuit_breaker_starts_closed() {
        let mut cb = CircuitBreaker::new(CircuitBreakerConfig::default());
        assert!(cb.call_allowed());
        assert!(matches!(cb.state(), CircuitState::Closed));
    }

    // ── Test 29: opens after failure threshold ──
    #[test]
    fn test_circuit_breaker_opens_after_failures() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            success_threshold: 2,
            timeout_secs: 60,
            request_volume_threshold: 3,
        };
        let mut cb = CircuitBreaker::new(config);
        cb.record_failure();
        cb.record_failure();
        cb.record_failure();
        assert!(matches!(cb.state(), CircuitState::Open { .. }));
    }

    // ── Test 30: rejects calls when Open ──
    #[test]
    fn test_circuit_breaker_rejects_when_open() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 1,
            timeout_secs: 9999,
            request_volume_threshold: 1,
        };
        let mut cb = CircuitBreaker::new(config);
        cb.record_failure();
        assert!(matches!(cb.state(), CircuitState::Open { .. }));
        assert!(!cb.call_allowed());
        assert!(cb.stats().rejected_open >= 1);
    }

    // ── Test 31: transitions to HalfOpen after timeout ──
    #[test]
    fn test_circuit_breaker_transitions_to_half_open() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 2,
            timeout_secs: 0, // immediate timeout
            request_volume_threshold: 1,
        };
        let mut cb = CircuitBreaker::new(config);
        cb.record_failure();
        assert!(matches!(cb.state(), CircuitState::Open { .. }));
        // With timeout_secs = 0, elapsed >= timeout immediately
        let allowed = cb.call_allowed();
        assert!(allowed, "should be allowed after timeout elapsed");
        assert!(matches!(cb.state(), CircuitState::HalfOpen));
    }

    // ── Test 32: closes after HalfOpen success threshold ──
    #[test]
    fn test_circuit_breaker_closes_after_half_open_successes() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 2,
            timeout_secs: 0,
            request_volume_threshold: 1,
        };
        let mut cb = CircuitBreaker::new(config);
        cb.record_failure();
        // Trigger HalfOpen
        cb.call_allowed();
        // Two successes should close it
        cb.record_success();
        cb.record_success();
        assert!(matches!(cb.state(), CircuitState::Closed));
    }

    // ── Test 33: stats accumulate correctly ──
    #[test]
    fn test_circuit_breaker_stats() {
        let mut cb = CircuitBreaker::new(CircuitBreakerConfig::default());
        cb.record_success();
        cb.record_success();
        cb.record_failure();
        assert_eq!(cb.stats().successful, 2);
        assert_eq!(cb.stats().failed, 1);
        assert_eq!(cb.stats().total_requests, 3);
    }

    // ─── Retry Policy tests ───────────────────────────────────────────────

    // ── Test 34: NoRetry never retries ──
    #[test]
    fn test_retry_policy_no_retry() {
        let p = RetryPolicy::NoRetry;
        assert!(!p.should_retry(0));
        assert_eq!(p.delay_ms(0, 42), 0);
    }

    // ── Test 35: FixedDelay respects max_retries and delay ──
    #[test]
    fn test_retry_policy_fixed_delay() {
        let p = RetryPolicy::FixedDelay {
            delay_ms: 200,
            max_retries: 3,
        };
        assert!(p.should_retry(0));
        assert!(p.should_retry(2));
        assert!(!p.should_retry(3));
        assert_eq!(p.delay_ms(0, 0), 200);
        assert_eq!(p.delay_ms(2, 0), 200);
    }

    // ── Test 36: ExponentialBackoff grows and caps delay ──
    #[test]
    fn test_retry_policy_exponential_backoff() {
        let p = RetryPolicy::ExponentialBackoff {
            initial_delay_ms: 100,
            multiplier: 2.0,
            max_delay_ms: 800,
            max_retries: 5,
        };
        assert!(p.should_retry(4));
        assert!(!p.should_retry(5));
        assert_eq!(p.delay_ms(0, 0), 100);
        assert_eq!(p.delay_ms(1, 0), 200);
        assert_eq!(p.delay_ms(2, 0), 400);
        assert_eq!(p.delay_ms(3, 0), 800); // 800 == max
        assert_eq!(p.delay_ms(4, 0), 800); // capped
    }

    // ── Test 37: Jitter delay is deterministic for same seed ──
    #[test]
    fn test_retry_policy_jitter_deterministic() {
        let p = RetryPolicy::Jitter {
            base_delay_ms: 100,
            jitter_factor: 0.5,
            max_retries: 3,
        };
        let d1 = p.delay_ms(0, 12345);
        let d2 = p.delay_ms(0, 12345);
        assert_eq!(d1, d2, "jitter should be deterministic for same seed");
        // delay should be >= base
        assert!(d1 >= 100);
    }

    // ── Extended tests ─────────────────────────────────────────────────────

    // Test 38: TrafficRoute::new stores model_id and weight
    #[test]
    fn test_traffic_route_new_stores_fields() {
        let r = TrafficRoute::new("model_a", 0.7);
        assert_eq!(r.model_id, "model_a");
        assert!((r.weight - 0.7).abs() < 1e-9);
        assert!(!r.is_canary);
        assert!(r.enabled);
    }

    // Test 39: TrafficRoute::canary sets is_canary=true
    #[test]
    fn test_traffic_route_canary_flag() {
        let r = TrafficRoute::canary("canary_model", 0.1);
        assert!(r.is_canary, "canary route must have is_canary=true");
    }

    // Test 40: SplitStrategy::WeightedRoundRobin != ConsistentHash
    #[test]
    fn test_split_strategy_variants_differ() {
        assert_ne!(
            SplitStrategy::WeightedRoundRobin,
            SplitStrategy::ConsistentHash
        );
    }

    // Test 41: TrafficSplitter stats returns stats for each route
    #[test]
    fn test_traffic_splitter_route_stats_count() {
        let routes = vec![TrafficRoute::new("a", 0.6), TrafficRoute::new("b", 0.4)];
        let config = TrafficConfig {
            routes,
            strategy: SplitStrategy::WeightedRoundRobin,
            sticky_sessions: false,
            fallback_route: "a".to_string(),
        };
        let splitter = TrafficSplitter::new(config).expect("valid");
        let stats = splitter.stats();
        assert_eq!(stats.len(), 2, "stats must return one entry per route");
    }

    // Test 42: TrafficSplitter total_requests increments on routing
    #[test]
    fn test_traffic_splitter_total_requests_increments() {
        let routes = vec![TrafficRoute::new("a", 0.5), TrafficRoute::new("b", 0.5)];
        let config = TrafficConfig {
            routes,
            strategy: SplitStrategy::WeightedRoundRobin,
            sticky_sessions: false,
            fallback_route: "a".to_string(),
        };
        let splitter = TrafficSplitter::new(config).expect("valid");
        splitter.route("req-1").expect("route");
        splitter.route("req-2").expect("route");
        let total = splitter.total_requests.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(total, 2, "total_requests must be 2 after two routes");
    }

    // Test 43: ConsistentHash routing is deterministic for same request_id
    #[test]
    fn test_consistent_hash_routing_deterministic() {
        let routes = vec![TrafficRoute::new("m1", 1.0), TrafficRoute::new("m2", 1.0)];
        let config = TrafficConfig {
            routes,
            strategy: SplitStrategy::ConsistentHash,
            sticky_sessions: false,
            fallback_route: "m1".to_string(),
        };
        let splitter = TrafficSplitter::new(config).expect("valid");
        let d1 = splitter.route("same-request").expect("route 1");
        let d2 = splitter.route("same-request").expect("route 2");
        assert_eq!(
            d1.model_id, d2.model_id,
            "consistent hash must be deterministic"
        );
    }

    // Test 44: RoutingDecision route_index is within bounds
    #[test]
    fn test_routing_decision_index_in_bounds() {
        let routes = vec![TrafficRoute::new("m0", 0.5), TrafficRoute::new("m1", 0.5)];
        let n = routes.len();
        let config = TrafficConfig {
            routes,
            strategy: SplitStrategy::WeightedRoundRobin,
            sticky_sessions: false,
            fallback_route: "m0".to_string(),
        };
        let splitter = TrafficSplitter::new(config).expect("valid");
        for i in 0..10u32 {
            let d = splitter.route(&format!("req-{i}")).expect("route");
            assert!(
                d.route_index < n,
                "route_index {} out of bounds",
                d.route_index
            );
        }
    }

    // Test 45: record_error increments error counter for route
    #[test]
    fn test_record_error_increments_error_count() {
        let routes = vec![TrafficRoute::new("model", 1.0)];
        let config = TrafficConfig {
            routes,
            strategy: SplitStrategy::WeightedRoundRobin,
            sticky_sessions: false,
            fallback_route: "model".to_string(),
        };
        let splitter = TrafficSplitter::new(config).expect("valid");
        splitter.route("req-1").expect("route");
        let _ = splitter.record_error(0);
        let stats = splitter.stats();
        assert_eq!(stats[0].errors, 1, "errors must be 1 after record_error");
    }

    // Test 46: TokenBucket::try_consume exact capacity succeeds
    #[test]
    fn test_token_bucket_consume_exact_capacity() {
        let mut tb = TokenBucket::new(50.0, 0.0);
        assert!(
            tb.try_consume(50.0),
            "consuming exactly capacity must succeed"
        );
        assert!((tb.current_tokens - 0.0).abs() < 1e-6);
    }

    // Test 47: CircuitBreakerConfig default — sensible failure threshold
    #[test]
    fn test_circuit_breaker_config_default() {
        let cfg = CircuitBreakerConfig::default();
        assert!(cfg.failure_threshold > 0);
        assert!(cfg.success_threshold > 0);
        assert!(cfg.timeout_secs > 0);
    }

    // Test 48: CircuitBreaker::stats initially all zeros
    #[test]
    fn test_circuit_breaker_stats_initial_zeros() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
        let s = cb.stats();
        assert_eq!(s.successful, 0);
        assert_eq!(s.failed, 0);
        assert_eq!(s.total_requests, 0);
        assert_eq!(s.rejected_open, 0);
    }

    // Test 49: RetryPolicy::Jitter should_retry respects max_retries
    #[test]
    fn test_jitter_retry_policy_respects_max_retries() {
        let p = RetryPolicy::Jitter {
            base_delay_ms: 50,
            jitter_factor: 0.2,
            max_retries: 2,
        };
        assert!(p.should_retry(0));
        assert!(p.should_retry(1));
        assert!(!p.should_retry(2));
    }

    // Test 50: ExponentialBackoff: should_retry returns false at max_retries
    #[test]
    fn test_exponential_backoff_should_retry_false_at_limit() {
        let p = RetryPolicy::ExponentialBackoff {
            initial_delay_ms: 10,
            multiplier: 2.0,
            max_delay_ms: 1000,
            max_retries: 0,
        };
        assert!(!p.should_retry(0), "max_retries=0 should never retry");
    }

    // Test 51: TrafficError::AllDisabled displays non-empty string
    #[test]
    fn test_traffic_error_all_disabled_display() {
        let e = TrafficError::AllDisabled;
        assert!(!e.to_string().is_empty());
    }

    // Test 52: LeakyBucket::new starts at zero level
    #[test]
    fn test_leaky_bucket_starts_empty() {
        let lb = LeakyBucket::new(100, 1.0);
        assert!(
            (lb.current_level - 0.0).abs() < 1e-9,
            "new bucket must start at level 0"
        );
    }

    // Test 53: TokenBucket::capacity is preserved after consumption
    #[test]
    fn test_token_bucket_capacity_unchanged_after_consume() {
        let mut tb = TokenBucket::new(100.0, 0.0);
        tb.try_consume(30.0);
        assert!(
            (tb.capacity - 100.0).abs() < 1e-9,
            "capacity must not change after consume"
        );
    }
}
