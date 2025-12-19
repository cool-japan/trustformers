//! Retry Policy Implementation
//!
//! Provides configurable retry policies with different backoff strategies
//! for handling transient failures and improving system resilience.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::RwLock;

/// Retry policy implementation
#[derive(Clone)]
pub struct RetryPolicy {
    config: RetryConfig,
    strategy: Arc<dyn RetryStrategy + Send + Sync>,
    stats: Arc<RwLock<RetryStats>>,
}

impl RetryPolicy {
    /// Create a new retry policy
    pub fn new(config: RetryConfig) -> Self {
        let strategy: Arc<dyn RetryStrategy + Send + Sync> = match config.strategy {
            RetryStrategyType::ExponentialBackoff => Arc::new(ExponentialBackoff::new(
                config.initial_delay,
                config.max_delay,
                config.multiplier,
            )),
            RetryStrategyType::LinearBackoff => Arc::new(LinearBackoff::new(
                config.initial_delay,
                config.max_delay,
                config.increment,
            )),
            RetryStrategyType::FixedDelay => Arc::new(FixedDelay::new(config.initial_delay)),
        };

        Self {
            config,
            strategy,
            stats: Arc::new(RwLock::new(RetryStats::new())),
        }
    }

    /// Execute operation with retry policy
    pub async fn execute<F, T, E>(&self, mut operation: F) -> Result<T>
    where
        F: FnMut() -> std::pin::Pin<
                Box<dyn std::future::Future<Output = std::result::Result<T, E>> + Send>,
            > + Send,
        E: std::error::Error + Send + Sync + 'static,
    {
        let mut attempt = 0;
        let start_time = Instant::now();
        let mut last_error;

        loop {
            attempt += 1;

            // Record attempt
            self.stats.write().await.record_attempt();

            // Execute operation
            match operation().await {
                Ok(result) => {
                    // Record success
                    let duration = start_time.elapsed();
                    self.stats.write().await.record_success(attempt, duration);
                    return Ok(result);
                },
                Err(error) => {
                    last_error = Some(error);

                    // Check if we should retry
                    if attempt >= self.config.max_attempts {
                        break;
                    }

                    // Check if error is retryable
                    if !self.is_retryable_error(&last_error.as_ref().unwrap()).await {
                        break;
                    }

                    // Calculate delay for next attempt
                    let delay = self.strategy.next_delay(attempt);

                    // Check total timeout
                    if start_time.elapsed() + delay > self.config.total_timeout {
                        break;
                    }

                    // Record retry
                    self.stats.write().await.record_retry(delay);

                    // Wait before next attempt
                    tokio::time::sleep(delay).await;
                },
            }
        }

        // Record final failure
        let duration = start_time.elapsed();
        self.stats.write().await.record_failure(attempt, duration);

        if let Some(error) = last_error {
            Err(anyhow!(
                "Operation failed after {} attempts: {}",
                attempt,
                error
            ))
        } else {
            Err(anyhow!("Operation failed after {} attempts", attempt))
        }
    }

    /// Check if error is retryable
    async fn is_retryable_error<E: std::error::Error>(&self, _error: &E) -> bool {
        // In practice, you'd check specific error types
        // For now, assume all errors are retryable
        true
    }

    /// Get retry statistics
    pub async fn get_stats(&self) -> RetryStats {
        self.stats.read().await.clone()
    }

    /// Reset statistics
    pub async fn reset_stats(&self) {
        *self.stats.write().await = RetryStats::new();
    }
}

/// Retry strategy trait
#[async_trait::async_trait]
pub trait RetryStrategy {
    /// Calculate next delay based on attempt number
    fn next_delay(&self, attempt: u32) -> Duration;

    /// Reset strategy state
    fn reset(&self) {}
}

/// Exponential backoff strategy
#[derive(Debug)]
pub struct ExponentialBackoff {
    initial_delay: Duration,
    max_delay: Duration,
    multiplier: f64,
}

impl ExponentialBackoff {
    pub fn new(initial_delay: Duration, max_delay: Duration, multiplier: f64) -> Self {
        Self {
            initial_delay,
            max_delay,
            multiplier,
        }
    }
}

#[async_trait::async_trait]
impl RetryStrategy for ExponentialBackoff {
    fn next_delay(&self, attempt: u32) -> Duration {
        let delay_ms =
            self.initial_delay.as_millis() as f64 * self.multiplier.powi(attempt as i32 - 1);
        let delay = Duration::from_millis(delay_ms as u64);
        delay.min(self.max_delay)
    }
}

/// Linear backoff strategy
#[derive(Debug)]
pub struct LinearBackoff {
    initial_delay: Duration,
    max_delay: Duration,
    increment: Duration,
}

impl LinearBackoff {
    pub fn new(initial_delay: Duration, max_delay: Duration, increment: Duration) -> Self {
        Self {
            initial_delay,
            max_delay,
            increment,
        }
    }
}

#[async_trait::async_trait]
impl RetryStrategy for LinearBackoff {
    fn next_delay(&self, attempt: u32) -> Duration {
        let delay = self.initial_delay + self.increment * (attempt - 1);
        delay.min(self.max_delay)
    }
}

/// Fixed delay strategy
#[derive(Debug)]
pub struct FixedDelay {
    delay: Duration,
}

impl FixedDelay {
    pub fn new(delay: Duration) -> Self {
        Self { delay }
    }
}

#[async_trait::async_trait]
impl RetryStrategy for FixedDelay {
    fn next_delay(&self, _attempt: u32) -> Duration {
        self.delay
    }
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_attempts: u32,

    /// Initial delay between attempts
    pub initial_delay: Duration,

    /// Maximum delay between attempts
    pub max_delay: Duration,

    /// Total timeout for all attempts
    pub total_timeout: Duration,

    /// Retry strategy type
    pub strategy: RetryStrategyType,

    /// Multiplier for exponential backoff
    pub multiplier: f64,

    /// Increment for linear backoff
    pub increment: Duration,

    /// Enable jitter to avoid thundering herd
    pub enable_jitter: bool,

    /// Jitter factor (0.0 to 1.0)
    pub jitter_factor: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            total_timeout: Duration::from_secs(60),
            strategy: RetryStrategyType::ExponentialBackoff,
            multiplier: 2.0,
            increment: Duration::from_millis(100),
            enable_jitter: true,
            jitter_factor: 0.1,
        }
    }
}

/// Retry strategy types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetryStrategyType {
    /// Exponential backoff with multiplier
    ExponentialBackoff,
    /// Linear backoff with fixed increment
    LinearBackoff,
    /// Fixed delay between attempts
    FixedDelay,
}

/// Retry statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryStats {
    /// Total number of operations attempted
    pub total_operations: u64,

    /// Total number of successful operations
    pub successful_operations: u64,

    /// Total number of failed operations
    pub failed_operations: u64,

    /// Total number of retry attempts
    pub total_retries: u64,

    /// Average number of attempts per operation
    pub avg_attempts: f64,

    /// Average operation duration
    pub avg_duration_ms: f64,

    /// Maximum operation duration
    pub max_duration_ms: f64,

    /// Success rate
    pub success_rate: f64,

    /// Creation time
    #[serde(skip)]
    pub created_at: u64, // Unix timestamp in seconds
}

impl Default for RetryStats {
    fn default() -> Self {
        Self::new()
    }
}

impl RetryStats {
    pub fn new() -> Self {
        Self {
            total_operations: 0,
            successful_operations: 0,
            failed_operations: 0,
            total_retries: 0,
            avg_attempts: 0.0,
            avg_duration_ms: 0.0,
            max_duration_ms: 0.0,
            success_rate: 0.0,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    pub fn record_attempt(&mut self) {
        // This is called for each individual attempt, not each operation
    }

    pub fn record_success(&mut self, attempts: u32, duration: Duration) {
        self.total_operations += 1;
        self.successful_operations += 1;
        self.total_retries += (attempts - 1) as u64; // Retries = attempts - 1

        let duration_ms = duration.as_millis() as f64;
        self.avg_duration_ms = (self.avg_duration_ms * (self.total_operations - 1) as f64
            + duration_ms)
            / self.total_operations as f64;
        self.max_duration_ms = self.max_duration_ms.max(duration_ms);

        self.update_derived_stats();
    }

    pub fn record_failure(&mut self, attempts: u32, duration: Duration) {
        self.total_operations += 1;
        self.failed_operations += 1;
        self.total_retries += (attempts - 1) as u64; // Retries = attempts - 1

        let duration_ms = duration.as_millis() as f64;
        self.avg_duration_ms = (self.avg_duration_ms * (self.total_operations - 1) as f64
            + duration_ms)
            / self.total_operations as f64;
        self.max_duration_ms = self.max_duration_ms.max(duration_ms);

        self.update_derived_stats();
    }

    pub fn record_retry(&mut self, _delay: Duration) {
        // Individual retry recorded
    }

    fn update_derived_stats(&mut self) {
        if self.total_operations > 0 {
            self.success_rate = self.successful_operations as f64 / self.total_operations as f64;
            self.avg_attempts =
                (self.total_retries + self.total_operations) as f64 / self.total_operations as f64;
        }
    }
}

/// Retry builder for convenient configuration
pub struct RetryBuilder {
    config: RetryConfig,
}

impl RetryBuilder {
    pub fn new() -> Self {
        Self {
            config: RetryConfig::default(),
        }
    }

    pub fn max_attempts(mut self, attempts: u32) -> Self {
        self.config.max_attempts = attempts;
        self
    }

    pub fn initial_delay(mut self, delay: Duration) -> Self {
        self.config.initial_delay = delay;
        self
    }

    pub fn max_delay(mut self, delay: Duration) -> Self {
        self.config.max_delay = delay;
        self
    }

    pub fn total_timeout(mut self, timeout: Duration) -> Self {
        self.config.total_timeout = timeout;
        self
    }

    pub fn strategy(mut self, strategy: RetryStrategyType) -> Self {
        self.config.strategy = strategy;
        self
    }

    pub fn exponential_backoff(mut self, multiplier: f64) -> Self {
        self.config.strategy = RetryStrategyType::ExponentialBackoff;
        self.config.multiplier = multiplier;
        self
    }

    pub fn linear_backoff(mut self, increment: Duration) -> Self {
        self.config.strategy = RetryStrategyType::LinearBackoff;
        self.config.increment = increment;
        self
    }

    pub fn fixed_delay(mut self, delay: Duration) -> Self {
        self.config.strategy = RetryStrategyType::FixedDelay;
        self.config.initial_delay = delay;
        self
    }

    pub fn with_jitter(mut self, factor: f64) -> Self {
        self.config.enable_jitter = true;
        self.config.jitter_factor = factor;
        self
    }

    pub fn build(self) -> RetryPolicy {
        RetryPolicy::new(self.config)
    }
}

impl Default for RetryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to create exponential backoff retry policy
pub fn exponential_backoff(
    max_attempts: u32,
    initial_delay: Duration,
    multiplier: f64,
) -> RetryPolicy {
    RetryBuilder::new()
        .max_attempts(max_attempts)
        .exponential_backoff(multiplier)
        .initial_delay(initial_delay)
        .build()
}

/// Helper function to create linear backoff retry policy
pub fn linear_backoff(
    max_attempts: u32,
    initial_delay: Duration,
    increment: Duration,
) -> RetryPolicy {
    RetryBuilder::new()
        .max_attempts(max_attempts)
        .linear_backoff(increment)
        .initial_delay(initial_delay)
        .build()
}

/// Helper function to create fixed delay retry policy
pub fn fixed_delay(max_attempts: u32, delay: Duration) -> RetryPolicy {
    RetryBuilder::new().max_attempts(max_attempts).fixed_delay(delay).build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[tokio::test]
    async fn test_exponential_backoff() {
        let backoff =
            ExponentialBackoff::new(Duration::from_millis(100), Duration::from_secs(1), 2.0);

        assert_eq!(backoff.next_delay(1), Duration::from_millis(100));
        assert_eq!(backoff.next_delay(2), Duration::from_millis(200));
        assert_eq!(backoff.next_delay(3), Duration::from_millis(400));
    }

    #[tokio::test]
    async fn test_linear_backoff() {
        let backoff = LinearBackoff::new(
            Duration::from_millis(100),
            Duration::from_secs(1),
            Duration::from_millis(50),
        );

        assert_eq!(backoff.next_delay(1), Duration::from_millis(100));
        assert_eq!(backoff.next_delay(2), Duration::from_millis(150));
        assert_eq!(backoff.next_delay(3), Duration::from_millis(200));
    }

    #[tokio::test]
    async fn test_retry_policy_success() {
        let policy = RetryBuilder::new()
            .max_attempts(3)
            .fixed_delay(Duration::from_millis(10))
            .build();

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let result = policy
            .execute(move || {
                let counter = counter_clone.clone();
                Box::pin(async move {
                    let count = counter.fetch_add(1, Ordering::SeqCst);
                    if count < 2 {
                        Err(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            "temporary failure",
                        ))
                    } else {
                        Ok("success")
                    }
                })
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(counter.load(Ordering::SeqCst), 3);

        let stats = policy.get_stats().await;
        assert_eq!(stats.total_operations, 1);
        assert_eq!(stats.successful_operations, 1);
        assert_eq!(stats.total_retries, 2);
    }

    #[tokio::test]
    async fn test_retry_policy_failure() {
        let policy = RetryBuilder::new()
            .max_attempts(2)
            .fixed_delay(Duration::from_millis(10))
            .build();

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let result = policy
            .execute(move || {
                let counter = counter_clone.clone();
                Box::pin(async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Err::<&str, _>(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "permanent failure",
                    ))
                })
            })
            .await;

        assert!(result.is_err());
        assert_eq!(counter.load(Ordering::SeqCst), 2);

        let stats = policy.get_stats().await;
        assert_eq!(stats.total_operations, 1);
        assert_eq!(stats.failed_operations, 1);
        assert_eq!(stats.total_retries, 1);
    }
}
