//! Health Check Service Implementation
//!
//! Provides comprehensive health monitoring for all system components
//! including model availability, resource usage, and service dependencies.

use anyhow::Result;
use axum::{extract::State, http::StatusCode, response::Json, routing::get, Router};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::RwLock;

/// Health check service
#[derive(Clone)]
pub struct HealthCheckService {
    config: HealthConfig,
    checks: Arc<RwLock<HashMap<String, Box<dyn HealthCheck + Send + Sync>>>>,
    results: Arc<RwLock<HashMap<String, HealthCheckResult>>>,
    stats: Arc<RwLock<HealthStats>>,
}

impl HealthCheckService {
    /// Create a new health check service
    pub fn new(config: HealthConfig) -> Self {
        Self {
            config,
            checks: Arc::new(RwLock::new(HashMap::new())),
            results: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(HealthStats::new())),
        }
    }

    /// Register a health check
    pub async fn register_check(&self, name: String, check: Box<dyn HealthCheck + Send + Sync>) {
        self.checks.write().await.insert(name, check);
    }

    /// Start health monitoring
    pub async fn start_monitoring(&self) -> Result<()> {
        let service = self.clone();
        let interval = self.config.check_interval;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(interval);

            loop {
                interval.tick().await;

                if let Err(e) = service.run_all_checks().await {
                    tracing::error!("Health check execution failed: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Run all registered health checks
    pub async fn run_all_checks(&self) -> Result<()> {
        let checks = self.checks.read().await;
        let mut results = HashMap::new();
        let start_time = Instant::now();

        for (name, check) in checks.iter() {
            let check_start = Instant::now();
            let result = check.check().await;
            let duration = check_start.elapsed();

            let check_result = HealthCheckResult {
                name: name.clone(),
                status: result.status,
                message: result.message,
                details: result.details,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                duration: duration.as_millis() as u64,
            };

            results.insert(name.clone(), check_result);
        }

        // Update results
        *self.results.write().await = results;

        // Update stats
        let total_duration = start_time.elapsed();
        self.stats.write().await.record_check_run(total_duration);

        Ok(())
    }

    /// Get individual health check result
    pub async fn get_check_result(&self, name: &str) -> Option<HealthCheckResult> {
        self.results.read().await.get(name).cloned()
    }

    /// Get all health check results
    pub async fn get_all_results(&self) -> HashMap<String, HealthCheckResult> {
        self.results.read().await.clone()
    }

    /// Get overall system health
    pub async fn get_system_health(&self) -> SystemHealth {
        let results = self.results.read().await;
        let mut component_healths = HashMap::new();
        let mut overall_status = HealthStatus::Healthy;

        for (name, result) in results.iter() {
            let component = ComponentHealth {
                name: name.clone(),
                status: result.status.clone(),
                message: result.message.clone(),
                last_check: result.timestamp,
                response_time_ms: result.duration,
            };

            component_healths.insert(name.clone(), component);

            // Determine overall status
            match result.status {
                HealthStatus::Unhealthy => overall_status = HealthStatus::Unhealthy,
                HealthStatus::Degraded if overall_status == HealthStatus::Healthy => {
                    overall_status = HealthStatus::Degraded;
                },
                _ => {},
            }
        }

        SystemHealth {
            status: overall_status,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            components: component_healths,
            uptime_seconds: self.get_uptime().await.as_secs(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Get health statistics
    pub async fn get_stats(&self) -> HealthStats {
        self.stats.read().await.clone()
    }

    /// Get service uptime
    async fn get_uptime(&self) -> Duration {
        let stats = self.stats.read().await;
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Duration::from_secs(current_time.saturating_sub(stats.start_time))
    }

    /// Create router for health endpoints
    pub fn create_router(&self) -> Router {
        Router::new()
            .route("/health", get(health_endpoint))
            .route("/health/ready", get(readiness_endpoint))
            .route("/health/live", get(liveness_endpoint))
            .route("/health/detailed", get(detailed_health_endpoint))
            .with_state(self.clone())
    }
}

/// Health check trait
#[async_trait::async_trait]
pub trait HealthCheck {
    /// Perform the health check
    async fn check(&self) -> HealthCheckResult;

    /// Get the name of this health check
    fn name(&self) -> &str;

    /// Get the timeout for this check
    fn timeout(&self) -> Duration {
        Duration::from_secs(30)
    }
}

/// Model health check
pub struct ModelHealthCheck {
    name: String,
    model_path: String,
}

impl ModelHealthCheck {
    pub fn new(name: String, model_path: String) -> Self {
        Self { name, model_path }
    }
}

#[async_trait::async_trait]
impl HealthCheck for ModelHealthCheck {
    async fn check(&self) -> HealthCheckResult {
        // Check if model is accessible
        let status = if std::path::Path::new(&self.model_path).exists() {
            HealthStatus::Healthy
        } else {
            HealthStatus::Unhealthy
        };

        let message = match status {
            HealthStatus::Healthy => "Model is accessible".to_string(),
            _ => format!("Model not found at path: {}", self.model_path),
        };

        HealthCheckResult {
            name: self.name.clone(),
            status,
            message,
            details: Some(serde_json::json!({
                "model_path": self.model_path,
                "exists": std::path::Path::new(&self.model_path).exists()
            })),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            duration: 1,
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Database health check
pub struct DatabaseHealthCheck {
    name: String,
    connection_string: String,
}

impl DatabaseHealthCheck {
    pub fn new(name: String, connection_string: String) -> Self {
        Self {
            name,
            connection_string,
        }
    }
}

#[async_trait::async_trait]
impl HealthCheck for DatabaseHealthCheck {
    async fn check(&self) -> HealthCheckResult {
        // Simulate database connection check
        let status = if !self.connection_string.is_empty() {
            HealthStatus::Healthy
        } else {
            HealthStatus::Unhealthy
        };

        HealthCheckResult {
            name: self.name.clone(),
            status,
            message: "Database connection check".to_string(),
            details: Some(serde_json::json!({
                "connection_string": self.connection_string
            })),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            duration: 5,
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Memory usage health check
pub struct MemoryHealthCheck {
    name: String,
    threshold_percent: f64,
}

impl MemoryHealthCheck {
    pub fn new(name: String, threshold_percent: f64) -> Self {
        Self {
            name,
            threshold_percent,
        }
    }

    fn get_memory_usage(&self) -> f64 {
        // Simplified memory usage calculation
        // In practice, you'd use system metrics
        0.5 // 50% usage
    }
}

#[async_trait::async_trait]
impl HealthCheck for MemoryHealthCheck {
    async fn check(&self) -> HealthCheckResult {
        let usage = self.get_memory_usage();

        let status = if usage < self.threshold_percent {
            HealthStatus::Healthy
        } else if usage < self.threshold_percent * 1.2 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Unhealthy
        };

        let message = format!("Memory usage: {:.1}%", usage * 100.0);

        HealthCheckResult {
            name: self.name.clone(),
            status,
            message,
            details: Some(serde_json::json!({
                "usage_percent": usage * 100.0,
                "threshold_percent": self.threshold_percent * 100.0
            })),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            duration: 2,
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthConfig {
    /// Interval between health checks
    pub check_interval: Duration,

    /// Timeout for individual checks
    pub check_timeout: Duration,

    /// Enable detailed health information
    pub enable_detailed: bool,

    /// Memory usage threshold
    pub memory_threshold: f64,

    /// Disk usage threshold
    pub disk_threshold: f64,

    /// CPU usage threshold
    pub cpu_threshold: f64,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(30),
            check_timeout: Duration::from_secs(10),
            enable_detailed: true,
            memory_threshold: 0.8, // 80%
            disk_threshold: 0.9,   // 90%
            cpu_threshold: 0.8,    // 80%
        }
    }
}

/// Health status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthStatus {
    /// Service is healthy
    Healthy,
    /// Service is degraded but functional
    Degraded,
    /// Service is unhealthy
    Unhealthy,
}

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    pub name: String,
    pub status: HealthStatus,
    pub message: String,
    pub details: Option<serde_json::Value>,
    pub timestamp: u64, // Unix timestamp in seconds
    pub duration: u64,  // Duration in milliseconds
}

/// Component health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub status: HealthStatus,
    pub message: String,
    pub last_check: u64, // Unix timestamp in seconds
    pub response_time_ms: u64,
}

/// Overall system health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub status: HealthStatus,
    pub timestamp: u64, // Unix timestamp in seconds
    pub components: HashMap<String, ComponentHealth>,
    pub uptime_seconds: u64,
    pub version: String,
}

/// Health statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStats {
    pub start_time: u64, // Unix timestamp in seconds
    pub total_checks: u64,
    pub successful_checks: u64,
    pub failed_checks: u64,
    pub avg_check_duration_ms: f64,
    pub last_check_time: Option<u64>, // Unix timestamp in seconds
}

impl HealthStats {
    pub fn new() -> Self {
        Self {
            start_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            total_checks: 0,
            successful_checks: 0,
            failed_checks: 0,
            avg_check_duration_ms: 0.0,
            last_check_time: None,
        }
    }

    pub fn record_check_run(&mut self, duration: Duration) {
        self.total_checks += 1;
        self.last_check_time = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        );

        let duration_ms = duration.as_millis() as f64;
        self.avg_check_duration_ms = (self.avg_check_duration_ms * (self.total_checks - 1) as f64
            + duration_ms)
            / self.total_checks as f64;
    }
}

/// Health endpoint handlers
pub async fn health_endpoint(
    State(service): State<HealthCheckService>,
) -> Result<Json<SystemHealth>, StatusCode> {
    let health = service.get_system_health().await;

    match health.status {
        HealthStatus::Healthy => Ok(Json(health)),
        HealthStatus::Degraded => {
            // Return 200 but with degraded status
            Ok(Json(health))
        },
        HealthStatus::Unhealthy => Err(StatusCode::SERVICE_UNAVAILABLE),
    }
}

pub async fn readiness_endpoint(
    State(service): State<HealthCheckService>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let health = service.get_system_health().await;

    let ready = matches!(
        health.status,
        HealthStatus::Healthy | HealthStatus::Degraded
    );

    let response = serde_json::json!({
        "ready": ready,
        "status": health.status,
        "timestamp": chrono::Utc::now().timestamp_millis()
    });

    if ready {
        Ok(Json(response))
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

pub async fn liveness_endpoint(
    State(service): State<HealthCheckService>,
) -> Json<serde_json::Value> {
    let stats = service.get_stats().await;

    let current_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let uptime_seconds = current_time.saturating_sub(stats.start_time);

    Json(serde_json::json!({
        "alive": true,
        "uptime_seconds": uptime_seconds,
        "total_checks": stats.total_checks,
        "timestamp": chrono::Utc::now().timestamp_millis()
    }))
}

pub async fn detailed_health_endpoint(
    State(service): State<HealthCheckService>,
) -> Json<serde_json::Value> {
    let health = service.get_system_health().await;
    let stats = service.get_stats().await;
    let results = service.get_all_results().await;

    Json(serde_json::json!({
        "system": health,
        "statistics": stats,
        "detailed_results": results
    }))
}

/// Health endpoint struct for grouping
pub struct HealthEndpoint;

impl HealthEndpoint {
    /// Create health check router
    pub fn router(service: HealthCheckService) -> Router {
        service.create_router()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_check_service() {
        let config = HealthConfig::default();
        let service = HealthCheckService::new(config);

        // Register a test health check
        let model_check =
            ModelHealthCheck::new("test_model".to_string(), "/tmp/test_model".to_string());

        service.register_check("test_model".to_string(), Box::new(model_check)).await;

        // Run checks
        service.run_all_checks().await.unwrap();

        // Get system health
        let health = service.get_system_health().await;
        assert!(!health.components.is_empty());
    }

    #[tokio::test]
    async fn test_memory_health_check() {
        let check = MemoryHealthCheck::new("memory".to_string(), 0.8);
        let result = check.check().await;

        assert_eq!(result.name, "memory");
        assert!(matches!(result.status, HealthStatus::Healthy));
    }
}
