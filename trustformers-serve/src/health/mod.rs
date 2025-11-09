// Allow dead code for infrastructure under development
#![allow(dead_code)]

//! Health Check and High Availability Module
//!
//! Provides health monitoring, circuit breaker patterns, and high availability
//! features for production deployment of the inference server.

pub mod circuit_breaker;
pub mod failover;
pub mod health_check;
pub mod retry;

pub use health_check::{
    ComponentHealth, HealthCheck, HealthCheckResult, HealthCheckService, HealthEndpoint,
    HealthStatus, SystemHealth,
};

pub use circuit_breaker::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError, CircuitBreakerState,
    CircuitBreakerStats, FailureThreshold,
};

pub use retry::{
    ExponentialBackoff, FixedDelay, LinearBackoff, RetryConfig, RetryPolicy, RetryStats,
    RetryStrategy,
};

pub use failover::{
    FailoverConfig, FailoverManager, FailoverStats, FailoverStrategy, LoadBalancer, NodeHealth,
};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};
use tokio::sync::RwLock;

/// High availability service orchestrator
#[derive(Clone)]
pub struct HighAvailabilityService {
    config: HAConfig,
    health_service: Arc<HealthCheckService>,
    circuit_breakers: Arc<RwLock<std::collections::HashMap<String, CircuitBreaker>>>,
    failover_manager: Arc<FailoverManager>,
    retry_policies: Arc<RwLock<std::collections::HashMap<String, RetryPolicy>>>,
    metrics: Arc<HAMetrics>,
}

impl HighAvailabilityService {
    /// Create a new high availability service
    pub fn new(config: HAConfig) -> Self {
        Self {
            config: config.clone(),
            health_service: Arc::new(HealthCheckService::new(config.health_config)),
            circuit_breakers: Arc::new(RwLock::new(std::collections::HashMap::new())),
            failover_manager: Arc::new(FailoverManager::new(config.failover_config)),
            retry_policies: Arc::new(RwLock::new(std::collections::HashMap::new())),
            metrics: Arc::new(HAMetrics::new()),
        }
    }

    /// Start the HA service
    pub async fn start(&self) -> Result<()> {
        // Start health monitoring
        self.health_service.start_monitoring().await?;

        // Start circuit breaker monitoring
        self.start_circuit_breaker_monitoring().await?;

        // Start failover monitoring
        self.failover_manager.start_monitoring().await?;

        Ok(())
    }

    /// Get or create circuit breaker for service
    pub async fn get_circuit_breaker(&self, service_name: String) -> CircuitBreaker {
        let mut breakers = self.circuit_breakers.write().await;

        breakers
            .entry(service_name.clone())
            .or_insert_with(|| {
                CircuitBreaker::new(service_name, self.config.circuit_breaker_config.clone())
            })
            .clone()
    }

    /// Execute operation with circuit breaker protection
    pub async fn execute_protected<F, T, E>(&self, service_name: String, operation: F) -> Result<T>
    where
        F: FnMut() -> std::pin::Pin<
                Box<dyn std::future::Future<Output = std::result::Result<T, E>> + Send>,
            > + Send,
        E: std::error::Error + Send + Sync + 'static,
    {
        let circuit_breaker = self.get_circuit_breaker(service_name.clone()).await;

        // Check circuit breaker state
        if !circuit_breaker.can_execute().await {
            return Err(anyhow::anyhow!(
                "Circuit breaker open for service: {}",
                service_name
            ));
        }

        // Execute with retry policy
        let retry_policy = self.get_retry_policy(service_name.clone()).await;
        let result = retry_policy.execute(operation).await;

        // Update circuit breaker based on result
        match &result {
            Ok(_) => circuit_breaker.record_success().await,
            Err(_) => circuit_breaker.record_failure().await,
        }

        result.map_err(|e| anyhow::anyhow!("{}", e))
    }

    /// Get or create retry policy for service
    async fn get_retry_policy(&self, service_name: String) -> RetryPolicy {
        let mut policies = self.retry_policies.write().await;

        policies
            .entry(service_name)
            .or_insert_with(|| RetryPolicy::new(self.config.retry_config.clone()))
            .clone()
    }

    /// Get overall system health
    pub async fn get_system_health(&self) -> SystemHealth {
        self.health_service.get_system_health().await
    }

    /// Get HA statistics
    pub async fn get_stats(&self) -> HAStats {
        HAStats {
            health_checks: self.health_service.get_stats().await,
            circuit_breakers: self.get_circuit_breaker_stats().await,
            failover: self.failover_manager.get_stats().await,
            retry_stats: self.get_retry_stats().await,
        }
    }

    /// Start circuit breaker monitoring
    async fn start_circuit_breaker_monitoring(&self) -> Result<()> {
        let breakers = self.circuit_breakers.clone();
        let interval = self.config.monitoring_interval;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(interval);

            loop {
                interval.tick().await;

                // Check circuit breaker states
                let breakers_read = breakers.read().await;
                for (name, breaker) in breakers_read.iter() {
                    if let Err(e) = breaker.check_health().await {
                        tracing::warn!("Circuit breaker {} health check failed: {}", name, e);
                    }
                }
            }
        });

        Ok(())
    }

    /// Get circuit breaker statistics
    async fn get_circuit_breaker_stats(
        &self,
    ) -> std::collections::HashMap<String, CircuitBreakerStats> {
        let mut stats = std::collections::HashMap::new();
        let breakers = self.circuit_breakers.read().await;

        for (name, breaker) in breakers.iter() {
            stats.insert(name.clone(), breaker.get_stats().await);
        }

        stats
    }

    /// Get retry statistics
    async fn get_retry_stats(&self) -> std::collections::HashMap<String, RetryStats> {
        let mut stats = std::collections::HashMap::new();
        let policies = self.retry_policies.read().await;

        for (name, policy) in policies.iter() {
            stats.insert(name.clone(), policy.get_stats().await);
        }

        stats
    }
}

/// High availability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HAConfig {
    /// Health check configuration
    pub health_config: health_check::HealthConfig,

    /// Circuit breaker configuration
    pub circuit_breaker_config: CircuitBreakerConfig,

    /// Retry configuration
    pub retry_config: RetryConfig,

    /// Failover configuration
    pub failover_config: failover::FailoverConfig,

    /// Monitoring interval
    pub monitoring_interval: Duration,

    /// Enable graceful shutdown
    pub enable_graceful_shutdown: bool,

    /// Shutdown timeout
    pub shutdown_timeout: Duration,
}

impl Default for HAConfig {
    fn default() -> Self {
        Self {
            health_config: health_check::HealthConfig::default(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
            retry_config: RetryConfig::default(),
            failover_config: failover::FailoverConfig::default(),
            monitoring_interval: Duration::from_secs(30),
            enable_graceful_shutdown: true,
            shutdown_timeout: Duration::from_secs(30),
        }
    }
}

/// High availability statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HAStats {
    pub health_checks: health_check::HealthStats,
    pub circuit_breakers: std::collections::HashMap<String, CircuitBreakerStats>,
    pub failover: failover::FailoverStats,
    pub retry_stats: std::collections::HashMap<String, RetryStats>,
}

/// HA metrics collector
#[derive(Debug)]
pub struct HAMetrics {
    requests_protected: Arc<RwLock<u64>>,
    failures_prevented: Arc<RwLock<u64>>,
    successful_failovers: Arc<RwLock<u64>>,
    retry_attempts: Arc<RwLock<u64>>,
}

impl HAMetrics {
    pub fn new() -> Self {
        Self {
            requests_protected: Arc::new(RwLock::new(0)),
            failures_prevented: Arc::new(RwLock::new(0)),
            successful_failovers: Arc::new(RwLock::new(0)),
            retry_attempts: Arc::new(RwLock::new(0)),
        }
    }

    pub async fn record_request_protected(&self) {
        *self.requests_protected.write().await += 1;
    }

    pub async fn record_failure_prevented(&self) {
        *self.failures_prevented.write().await += 1;
    }

    pub async fn record_successful_failover(&self) {
        *self.successful_failovers.write().await += 1;
    }

    pub async fn record_retry_attempt(&self) {
        *self.retry_attempts.write().await += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ha_service() {
        let config = HAConfig::default();
        let service = HighAvailabilityService::new(config);

        // Test circuit breaker creation
        let cb = service.get_circuit_breaker("test_service".to_string()).await;
        assert!(cb.can_execute().await);

        // Test system health
        let health = service.get_system_health().await;
        assert!(matches!(
            health.status,
            HealthStatus::Healthy | HealthStatus::Degraded
        ));
    }

    #[tokio::test]
    async fn test_protected_execution() {
        let config = HAConfig::default();
        let service = HighAvailabilityService::new(config);

        // Test successful operation
        let result = service
            .execute_protected("test".to_string(), || {
                Box::pin(async { Ok::<_, std::io::Error>("success") })
            })
            .await;

        assert!(result.is_ok());
    }
}
