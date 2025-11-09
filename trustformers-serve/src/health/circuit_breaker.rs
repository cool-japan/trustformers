//! Circuit Breaker Implementation
//!
//! Provides circuit breaker pattern for preventing cascading failures
//! and protecting against unavailable dependencies or overloaded services.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};
use tokio::sync::RwLock;

/// Circuit breaker implementation
#[derive(Clone)]
pub struct CircuitBreaker {
    name: String,
    config: CircuitBreakerConfig,
    state: Arc<RwLock<CircuitBreakerState>>,
    stats: Arc<RwLock<CircuitBreakerStats>>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(name: String, config: CircuitBreakerConfig) -> Self {
        Self {
            name,
            config,
            state: Arc::new(RwLock::new(CircuitBreakerState::Closed)),
            stats: Arc::new(RwLock::new(CircuitBreakerStats::new())),
        }
    }

    /// Check if the circuit breaker allows execution
    pub async fn can_execute(&self) -> bool {
        let state = self.state.read().await;

        match *state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open { opened_at } => {
                // Check if we should transition to half-open
                let current_time = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                let elapsed_duration = Duration::from_secs(current_time.saturating_sub(opened_at));
                if elapsed_duration >= self.config.timeout {
                    drop(state);
                    self.transition_to_half_open().await;
                    true
                } else {
                    false
                }
            },
            CircuitBreakerState::HalfOpen => true,
        }
    }

    /// Record a successful operation
    pub async fn record_success(&self) {
        let mut stats = self.stats.write().await;
        stats.record_success();

        let current_state = self.state.read().await.clone();

        match current_state {
            CircuitBreakerState::HalfOpen => {
                // Transition back to closed if we have enough successes
                if stats.consecutive_successes >= self.config.success_threshold {
                    drop(stats);
                    self.transition_to_closed().await;
                }
            },
            CircuitBreakerState::Closed => {
                // Check if we should evaluate failure rate
                if stats.total_requests % self.config.evaluation_window == 0 {
                    let failure_rate = stats.failure_rate();
                    if self.should_open(failure_rate, &stats).await {
                        drop(stats);
                        self.transition_to_open().await;
                    }
                }
            },
            _ => {},
        }
    }

    /// Record a failed operation
    pub async fn record_failure(&self) {
        let mut stats = self.stats.write().await;
        stats.record_failure();

        let current_state = self.state.read().await.clone();

        match current_state {
            CircuitBreakerState::HalfOpen => {
                // Transition back to open on any failure in half-open state
                drop(stats);
                self.transition_to_open().await;
            },
            CircuitBreakerState::Closed => {
                // Check if we should open based on failure threshold
                let failure_rate = stats.failure_rate();
                if self.should_open(failure_rate, &stats).await {
                    drop(stats);
                    self.transition_to_open().await;
                }
            },
            _ => {},
        }
    }

    /// Get current circuit breaker state
    pub async fn get_state(&self) -> CircuitBreakerState {
        self.state.read().await.clone()
    }

    /// Get circuit breaker statistics
    pub async fn get_stats(&self) -> CircuitBreakerStats {
        self.stats.read().await.clone()
    }

    /// Check circuit breaker health
    pub async fn check_health(&self) -> Result<()> {
        let state = self.state.read().await;
        let stats = self.stats.read().await;

        match *state {
            CircuitBreakerState::Open { .. } => {
                Err(anyhow!("Circuit breaker {} is open", self.name))
            },
            CircuitBreakerState::HalfOpen => {
                if stats.consecutive_failures > 0 {
                    Err(anyhow!("Circuit breaker {} is unstable", self.name))
                } else {
                    Ok(())
                }
            },
            CircuitBreakerState::Closed => {
                let failure_rate = stats.failure_rate();
                if failure_rate > self.config.failure_threshold.rate * 0.8 {
                    Err(anyhow!(
                        "Circuit breaker {} has high failure rate: {:.2}%",
                        self.name,
                        failure_rate * 100.0
                    ))
                } else {
                    Ok(())
                }
            },
        }
    }

    /// Reset circuit breaker statistics
    pub async fn reset(&self) {
        *self.stats.write().await = CircuitBreakerStats::new();
        *self.state.write().await = CircuitBreakerState::Closed;
    }

    /// Force circuit breaker to open
    pub async fn force_open(&self) {
        self.transition_to_open().await;
    }

    /// Force circuit breaker to close
    pub async fn force_close(&self) {
        self.transition_to_closed().await;
    }

    /// Check if circuit breaker should open
    async fn should_open(&self, failure_rate: f64, stats: &CircuitBreakerStats) -> bool {
        // Must have minimum number of requests
        if stats.total_requests < self.config.failure_threshold.min_requests {
            return false;
        }

        // Check failure rate threshold
        if failure_rate >= self.config.failure_threshold.rate {
            return true;
        }

        // Check consecutive failures
        if let Some(max_consecutive) = self.config.failure_threshold.max_consecutive_failures {
            if stats.consecutive_failures >= max_consecutive {
                return true;
            }
        }

        false
    }

    /// Transition to closed state
    async fn transition_to_closed(&self) {
        *self.state.write().await = CircuitBreakerState::Closed;
        self.stats.write().await.reset_consecutive_counters();
        tracing::info!("Circuit breaker {} transitioned to CLOSED", self.name);
    }

    /// Transition to open state
    async fn transition_to_open(&self) {
        *self.state.write().await = CircuitBreakerState::Open {
            opened_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        self.stats.write().await.record_state_change();
        tracing::warn!("Circuit breaker {} transitioned to OPEN", self.name);
    }

    /// Transition to half-open state
    async fn transition_to_half_open(&self) {
        *self.state.write().await = CircuitBreakerState::HalfOpen;
        self.stats.write().await.reset_consecutive_counters();
        tracing::info!("Circuit breaker {} transitioned to HALF-OPEN", self.name);
    }
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Failure threshold for opening the circuit
    pub failure_threshold: FailureThreshold,

    /// Number of successful requests needed to close from half-open
    pub success_threshold: u32,

    /// Timeout before attempting to close an open circuit
    pub timeout: Duration,

    /// Window size for evaluating failure rate
    pub evaluation_window: u32,

    /// Maximum number of concurrent requests in half-open state
    pub max_concurrent_half_open: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: FailureThreshold::default(),
            success_threshold: 3,
            timeout: Duration::from_secs(60),
            evaluation_window: 100,
            max_concurrent_half_open: 3,
        }
    }
}

/// Failure threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureThreshold {
    /// Failure rate threshold (0.0 to 1.0)
    pub rate: f64,

    /// Minimum number of requests before evaluating
    pub min_requests: u32,

    /// Maximum consecutive failures before opening
    pub max_consecutive_failures: Option<u32>,
}

impl Default for FailureThreshold {
    fn default() -> Self {
        Self {
            rate: 0.5, // 50% failure rate
            min_requests: 10,
            max_consecutive_failures: Some(5),
        }
    }
}

/// Circuit breaker state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CircuitBreakerState {
    /// Circuit is closed, requests are allowed
    Closed,
    /// Circuit is open, requests are blocked
    Open { opened_at: u64 }, // Unix timestamp in seconds
    /// Circuit is half-open, limited requests are allowed
    HalfOpen,
}

/// Circuit breaker statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerStats {
    /// Total number of requests
    pub total_requests: u32,
    /// Total number of successful requests
    pub successful_requests: u32,
    /// Total number of failed requests
    pub failed_requests: u32,
    /// Consecutive successful requests
    pub consecutive_successes: u32,
    /// Consecutive failed requests
    pub consecutive_failures: u32,
    /// Number of state changes
    pub state_changes: u32,
    /// Last state change time
    pub last_state_change: Option<u64>, // Unix timestamp in seconds
    /// Creation time
    pub created_at: u64, // Unix timestamp in seconds
}

impl CircuitBreakerStats {
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            consecutive_successes: 0,
            consecutive_failures: 0,
            state_changes: 0,
            last_state_change: None,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    pub fn record_success(&mut self) {
        self.total_requests += 1;
        self.successful_requests += 1;
        self.consecutive_successes += 1;
        self.consecutive_failures = 0;
    }

    pub fn record_failure(&mut self) {
        self.total_requests += 1;
        self.failed_requests += 1;
        self.consecutive_failures += 1;
        self.consecutive_successes = 0;
    }

    pub fn record_state_change(&mut self) {
        self.state_changes += 1;
        self.last_state_change = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        );
    }

    pub fn reset_consecutive_counters(&mut self) {
        self.consecutive_successes = 0;
        self.consecutive_failures = 0;
    }

    pub fn failure_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.failed_requests as f64 / self.total_requests as f64
        }
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.successful_requests as f64 / self.total_requests as f64
        }
    }

    pub fn uptime(&self) -> Duration {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Duration::from_secs(now.saturating_sub(self.created_at))
    }
}

/// Circuit breaker error types
#[derive(Debug, thiserror::Error)]
pub enum CircuitBreakerError {
    #[error("Circuit breaker is open")]
    CircuitOpen,

    #[error("Circuit breaker is unstable")]
    CircuitUnstable,

    #[error("Maximum concurrent requests exceeded")]
    MaxConcurrentExceeded,

    #[error("Operation timeout")]
    Timeout,
}

/// Circuit breaker wrapper for protecting operations
pub struct ProtectedOperation<F> {
    circuit_breaker: CircuitBreaker,
    operation: F,
}

impl<F, T, E> ProtectedOperation<F>
where
    F: std::future::Future<Output = std::result::Result<T, E>>,
    E: std::error::Error + Send + Sync + 'static,
{
    pub fn new(circuit_breaker: CircuitBreaker, operation: F) -> Self {
        Self {
            circuit_breaker,
            operation,
        }
    }

    pub async fn execute(self) -> Result<T> {
        // Check if circuit breaker allows execution
        if !self.circuit_breaker.can_execute().await {
            return Err(anyhow!("Circuit breaker open"));
        }

        // Execute the operation
        match self.operation.await {
            Ok(result) => {
                self.circuit_breaker.record_success().await;
                Ok(result)
            },
            Err(error) => {
                self.circuit_breaker.record_failure().await;
                Err(anyhow!("Operation failed: {}", error))
            },
        }
    }
}

/// Helper macro for protecting operations with circuit breaker
#[macro_export]
macro_rules! protect_with_circuit_breaker {
    ($circuit_breaker:expr, $operation:expr) => {{
        use $crate::health::circuit_breaker::ProtectedOperation;
        ProtectedOperation::new($circuit_breaker, $operation).execute().await
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[tokio::test]
    async fn test_circuit_breaker_closed_state() {
        let config = CircuitBreakerConfig::default();
        let cb = CircuitBreaker::new("test".to_string(), config);

        assert!(cb.can_execute().await);

        // Record some successes
        for _ in 0..5 {
            cb.record_success().await;
        }

        let stats = cb.get_stats().await;
        assert_eq!(stats.successful_requests, 5);
        assert!(matches!(cb.get_state().await, CircuitBreakerState::Closed));
    }

    #[tokio::test]
    async fn test_circuit_breaker_opens_on_failures() {
        let mut config = CircuitBreakerConfig::default();
        config.failure_threshold.min_requests = 3;
        config.failure_threshold.rate = 0.5;

        let cb = CircuitBreaker::new("test".to_string(), config);

        // Record enough failures to open the circuit
        for _ in 0..5 {
            cb.record_failure().await;
        }

        let state = cb.get_state().await;
        assert!(matches!(state, CircuitBreakerState::Open { .. }));
    }

    #[tokio::test]
    async fn test_circuit_breaker_half_open_transition() {
        let mut config = CircuitBreakerConfig::default();
        config.timeout = Duration::from_secs(1);

        let cb = CircuitBreaker::new("test".to_string(), config);

        // Force open
        cb.force_open().await;

        // Wait for timeout (a bit longer to account for timing precision)
        tokio::time::sleep(Duration::from_millis(1100)).await;

        // Should allow execution (transition to half-open)
        assert!(cb.can_execute().await);

        let state = cb.get_state().await;
        assert!(matches!(state, CircuitBreakerState::HalfOpen));
    }

    #[tokio::test]
    async fn test_protected_operation() {
        let config = CircuitBreakerConfig::default();
        let cb = CircuitBreaker::new("test".to_string(), config);

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let operation = async move {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            Ok::<_, std::io::Error>("success")
        };

        let result = ProtectedOperation::new(cb, operation).execute().await;
        assert!(result.is_ok());
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}
