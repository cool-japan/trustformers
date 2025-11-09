// Allow dead code for infrastructure under development
#![allow(dead_code)]

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tracing::info;

/// Advanced load balancer for horizontal scaling and cluster coordination
#[derive(Debug, Clone)]
pub struct LoadBalancer {
    config: LoadBalancerConfig,
    state: Arc<RwLock<LoadBalancerState>>,
    metrics: Arc<Mutex<LoadBalancerMetrics>>,
}

/// Load balancer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancerConfig {
    /// Load balancing algorithm
    pub algorithm: LoadBalancingAlgorithm,
    /// Health checking configuration
    pub health_check: HealthCheckSettings,
    /// Auto-scaling configuration
    pub auto_scaling: AutoScalingConfig,
    /// Session affinity settings
    pub session_affinity: SessionAffinityConfig,
    /// Circuit breaker configuration
    pub circuit_breaker: CircuitBreakerSettings,
    /// Retry policy
    pub retry_policy: RetryPolicy,
    /// Connection pooling
    pub connection_pool: ConnectionPoolConfig,
}

/// Load balancing algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingAlgorithm {
    /// Round-robin distribution
    RoundRobin,
    /// Weighted round-robin
    WeightedRoundRobin,
    /// Least connections
    LeastConnections,
    /// Weighted least connections
    WeightedLeastConnections,
    /// IP hash-based routing
    IPHash,
    /// Least response time
    LeastResponseTime,
    /// Resource-based routing
    ResourceBased { cpu_weight: f64, memory_weight: f64 },
    /// Geographic routing
    Geographic,
    /// Consistent hashing
    ConsistentHashing { virtual_nodes: u32 },
    /// Custom algorithm
    Custom {
        name: String,
        parameters: HashMap<String, String>,
    },
}

/// Health check settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckSettings {
    /// Health check interval
    pub interval: Duration,
    /// Request timeout
    pub timeout: Duration,
    /// Failure threshold for marking unhealthy
    pub failure_threshold: u32,
    /// Success threshold for marking healthy
    pub success_threshold: u32,
    /// Health check endpoint
    pub endpoint: String,
    /// Expected response codes
    pub expected_codes: Vec<u16>,
    /// Enable passive health checks
    pub passive_checks: bool,
}

/// Auto-scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScalingConfig {
    /// Enable auto-scaling
    pub enabled: bool,
    /// Minimum number of instances
    pub min_instances: u32,
    /// Maximum number of instances
    pub max_instances: u32,
    /// Target CPU utilization percentage
    pub target_cpu_utilization: f64,
    /// Target memory utilization percentage
    pub target_memory_utilization: f64,
    /// Target request rate per instance
    pub target_request_rate: f64,
    /// Scale-up cooldown period
    pub scale_up_cooldown: Duration,
    /// Scale-down cooldown period
    pub scale_down_cooldown: Duration,
    /// Scaling policies
    pub scaling_policies: Vec<ScalingPolicy>,
}

/// Scaling policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingPolicy {
    /// Policy name
    pub name: String,
    /// Metric to monitor
    pub metric: ScalingMetric,
    /// Threshold value
    pub threshold: f64,
    /// Comparison operator
    pub operator: ComparisonOperator,
    /// Number of instances to scale
    pub scale_amount: i32,
    /// Cooldown period
    pub cooldown: Duration,
}

/// Scaling metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingMetric {
    CpuUtilization,
    MemoryUtilization,
    RequestRate,
    ResponseTime,
    ErrorRate,
    ActiveConnections,
    QueueLength,
    Custom(String),
}

/// Comparison operators for scaling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equal,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

/// Session affinity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionAffinityConfig {
    /// Enable session affinity
    pub enabled: bool,
    /// Affinity type
    pub affinity_type: AffinityType,
    /// Session timeout
    pub session_timeout: Duration,
    /// Fallback behavior when preferred instance is unavailable
    pub fallback_behavior: FallbackBehavior,
}

/// Session affinity types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AffinityType {
    /// Cookie-based affinity
    Cookie { name: String, secure: bool },
    /// IP-based affinity
    IPAddress,
    /// Header-based affinity
    Header { name: String },
    /// Custom affinity
    Custom { identifier: String },
}

/// Fallback behavior for session affinity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FallbackBehavior {
    /// Route to any available instance
    AnyAvailable,
    /// Route to least loaded instance
    LeastLoaded,
    /// Return error
    Error,
}

/// Circuit breaker settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerSettings {
    /// Enable circuit breaker
    pub enabled: bool,
    /// Failure threshold
    pub failure_threshold: u32,
    /// Timeout for opening circuit
    pub timeout: Duration,
    /// Half-open state test requests
    pub half_open_max_requests: u32,
    /// Recovery time
    pub recovery_time: Duration,
}

/// Retry policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum number of retries
    pub max_retries: u32,
    /// Base retry delay
    pub base_delay: Duration,
    /// Maximum retry delay
    pub max_delay: Duration,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    /// Jitter enable
    pub jitter: bool,
    /// Retryable conditions
    pub retryable_conditions: Vec<RetryCondition>,
}

/// Retry conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetryCondition {
    HttpStatus(u16),
    Timeout,
    ConnectionError,
    Custom(String),
}

/// Connection pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPoolConfig {
    /// Maximum connections per instance
    pub max_connections_per_instance: u32,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Idle timeout
    pub idle_timeout: Duration,
    /// Keep-alive duration
    pub keep_alive: Duration,
    /// Enable connection pooling
    pub enabled: bool,
}

/// Load balancer internal state
#[derive(Debug)]
struct LoadBalancerState {
    /// Backend instances
    instances: Vec<BackendInstance>,
    /// Current round-robin index
    round_robin_index: usize,
    /// Session mapping
    sessions: HashMap<String, String>,
    /// Circuit breaker states
    circuit_breakers: HashMap<String, CircuitBreakerState>,
    /// Instance weights (for weighted algorithms)
    instance_weights: HashMap<String, f64>,
    /// Consistent hashing ring
    hash_ring: Vec<HashRingNode>,
}

/// Backend instance information
#[derive(Debug, Clone)]
pub struct BackendInstance {
    /// Instance ID
    pub id: String,
    /// Instance address
    pub address: String,
    /// Instance port
    pub port: u16,
    /// Instance weight
    pub weight: f64,
    /// Current health status
    pub health_status: InstanceHealth,
    /// Current connections
    pub active_connections: u32,
    /// Response time statistics
    pub response_time_stats: ResponseTimeStats,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
    /// Last health check time
    pub last_health_check: Option<Instant>,
    /// Instance metadata
    pub metadata: HashMap<String, String>,
}

/// Instance health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InstanceHealth {
    Healthy,
    Unhealthy,
    Warning,
    Maintenance,
    Draining,
}

/// Response time statistics
#[derive(Debug, Clone)]
pub struct ResponseTimeStats {
    pub avg_response_time: Duration,
    pub min_response_time: Duration,
    pub max_response_time: Duration,
    pub p95_response_time: Duration,
    pub p99_response_time: Duration,
    pub sample_count: u32,
}

/// Resource utilization metrics
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub network_in: f64,
    pub network_out: f64,
    pub disk_io: f64,
}

/// Circuit breaker state
#[derive(Debug, Clone)]
struct CircuitBreakerState {
    state: CircuitState,
    failure_count: u32,
    last_failure_time: Option<Instant>,
    half_open_requests: u32,
}

/// Circuit breaker states
#[derive(Debug, Clone, PartialEq)]
enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Hash ring node for consistent hashing
#[derive(Debug, Clone)]
struct HashRingNode {
    hash: u64,
    instance_id: String,
    virtual_node_id: u32,
}

/// Load balancer metrics
#[derive(Debug, Default)]
pub struct LoadBalancerMetrics {
    /// Total requests processed
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Retried requests
    pub retried_requests: u64,
    /// Circuit breaker trips
    pub circuit_breaker_trips: u64,
    /// Average response time
    pub avg_response_time: Duration,
    /// Request distribution per instance
    pub instance_request_counts: HashMap<String, u64>,
    /// Current instance count
    pub instance_count: u32,
    /// Healthy instance count
    pub healthy_instance_count: u32,
    /// Auto-scaling events
    pub auto_scaling_events: u64,
}

impl LoadBalancer {
    /// Create a new load balancer
    pub fn new(config: LoadBalancerConfig) -> Self {
        let state = LoadBalancerState {
            instances: Vec::new(),
            round_robin_index: 0,
            sessions: HashMap::new(),
            circuit_breakers: HashMap::new(),
            instance_weights: HashMap::new(),
            hash_ring: Vec::new(),
        };

        Self {
            config,
            state: Arc::new(RwLock::new(state)),
            metrics: Arc::new(Mutex::new(LoadBalancerMetrics::default())),
        }
    }

    /// Add a backend instance
    pub async fn add_instance(&self, instance: BackendInstance) -> Result<()> {
        let mut state = self.state.write().await;
        let mut metrics = self.metrics.lock().await;

        // Check if instance already exists
        if state.instances.iter().any(|i| i.id == instance.id) {
            return Err(anyhow!("Instance {} already exists", instance.id));
        }

        // Add to instances
        state.instances.push(instance.clone());

        // Update weights
        state.instance_weights.insert(instance.id.clone(), instance.weight);

        // Initialize circuit breaker
        state.circuit_breakers.insert(
            instance.id.clone(),
            CircuitBreakerState {
                state: CircuitState::Closed,
                failure_count: 0,
                last_failure_time: None,
                half_open_requests: 0,
            },
        );

        // Update hash ring for consistent hashing
        if matches!(
            self.config.algorithm,
            LoadBalancingAlgorithm::ConsistentHashing { .. }
        ) {
            self.rebuild_hash_ring(&mut state).await;
        }

        // Update metrics
        metrics.instance_count += 1;
        if matches!(instance.health_status, InstanceHealth::Healthy) {
            metrics.healthy_instance_count += 1;
        }

        info!(
            "Added instance {} at {}:{}",
            instance.id, instance.address, instance.port
        );
        Ok(())
    }

    /// Remove a backend instance
    pub async fn remove_instance(&self, instance_id: &str) -> Result<()> {
        let mut state = self.state.write().await;
        let mut metrics = self.metrics.lock().await;

        // Find and remove instance
        let removed = state.instances.iter().position(|i| i.id == instance_id);
        if let Some(index) = removed {
            let instance = state.instances.remove(index);

            // Remove from weights and circuit breakers
            state.instance_weights.remove(&instance.id);
            state.circuit_breakers.remove(&instance.id);

            // Remove sessions pointing to this instance
            state.sessions.retain(|_, v| v != &instance.id);

            // Update hash ring
            if matches!(
                self.config.algorithm,
                LoadBalancingAlgorithm::ConsistentHashing { .. }
            ) {
                self.rebuild_hash_ring(&mut state).await;
            }

            // Update metrics
            metrics.instance_count -= 1;
            if matches!(instance.health_status, InstanceHealth::Healthy) {
                metrics.healthy_instance_count -= 1;
            }

            info!("Removed instance {}", instance_id);
            Ok(())
        } else {
            Err(anyhow!("Instance {} not found", instance_id))
        }
    }

    /// Select an instance for load balancing
    pub async fn select_instance(
        &self,
        request_context: &RequestContext,
    ) -> Result<Option<BackendInstance>> {
        let mut state = self.state.write().await;

        // Filter healthy instances (clone to avoid borrowing conflicts)
        let healthy_instances: Vec<BackendInstance> = state
            .instances
            .iter()
            .filter(|i| matches!(i.health_status, InstanceHealth::Healthy))
            .cloned()
            .collect();

        if healthy_instances.is_empty() {
            return Ok(None);
        }

        // Check session affinity first
        if self.config.session_affinity.enabled {
            if let Some(instance_id) = self.get_session_instance(request_context, &state).await {
                if let Some(instance) = healthy_instances.iter().find(|i| i.id == instance_id) {
                    return Ok(Some(instance.clone()));
                }
            }
        }

        // Select based on algorithm
        let instance_refs: Vec<&BackendInstance> = healthy_instances.iter().collect();
        let selected = match &self.config.algorithm {
            LoadBalancingAlgorithm::RoundRobin => {
                self.select_round_robin(&instance_refs, &mut state).await
            },
            LoadBalancingAlgorithm::WeightedRoundRobin => {
                self.select_weighted_round_robin(&instance_refs, &state).await
            },
            LoadBalancingAlgorithm::LeastConnections => {
                self.select_least_connections(&instance_refs).await
            },
            LoadBalancingAlgorithm::WeightedLeastConnections => {
                self.select_weighted_least_connections(&instance_refs, &state).await
            },
            LoadBalancingAlgorithm::IPHash => {
                self.select_ip_hash(&instance_refs, request_context).await
            },
            LoadBalancingAlgorithm::LeastResponseTime => {
                self.select_least_response_time(&instance_refs).await
            },
            LoadBalancingAlgorithm::ResourceBased {
                cpu_weight,
                memory_weight,
            } => self.select_resource_based(&instance_refs, *cpu_weight, *memory_weight).await,
            LoadBalancingAlgorithm::Geographic => {
                self.select_geographic(&instance_refs, request_context).await
            },
            LoadBalancingAlgorithm::ConsistentHashing { .. } => {
                self.select_consistent_hash(&instance_refs, request_context, &state).await
            },
            LoadBalancingAlgorithm::Custom { name, parameters } => {
                self.select_custom(&instance_refs, name, parameters, request_context).await
            },
        };

        // Update session affinity
        if let Some(ref instance) = selected {
            if self.config.session_affinity.enabled {
                self.update_session_affinity(request_context, &instance.id, &mut state).await;
            }
        }

        Ok(selected)
    }

    /// Record request result for metrics and circuit breaker
    pub async fn record_request_result(
        &self,
        instance_id: &str,
        success: bool,
        response_time: Duration,
    ) -> Result<()> {
        let mut state = self.state.write().await;
        let mut metrics = self.metrics.lock().await;

        // Update instance statistics
        if let Some(instance) = state.instances.iter_mut().find(|i| i.id == instance_id) {
            self.update_response_time_stats(&mut instance.response_time_stats, response_time)
                .await;
        }

        // Update circuit breaker
        if let Some(cb_state) = state.circuit_breakers.get_mut(instance_id) {
            if success {
                cb_state.failure_count = 0;
                if cb_state.state == CircuitState::HalfOpen {
                    cb_state.half_open_requests += 1;
                    if cb_state.half_open_requests
                        >= self.config.circuit_breaker.half_open_max_requests
                    {
                        cb_state.state = CircuitState::Closed;
                        cb_state.half_open_requests = 0;
                    }
                }
            } else {
                cb_state.failure_count += 1;
                cb_state.last_failure_time = Some(Instant::now());

                if cb_state.failure_count >= self.config.circuit_breaker.failure_threshold {
                    cb_state.state = CircuitState::Open;
                    metrics.circuit_breaker_trips += 1;
                }
            }
        }

        // Update global metrics
        metrics.total_requests += 1;
        if success {
            metrics.successful_requests += 1;
        } else {
            metrics.failed_requests += 1;
        }

        // Update instance request counts
        *metrics.instance_request_counts.entry(instance_id.to_string()).or_insert(0) += 1;

        // Update average response time
        let total = metrics.successful_requests + metrics.failed_requests;
        metrics.avg_response_time = Duration::from_nanos(
            (metrics.avg_response_time.as_nanos() as u64 * (total - 1)
                + response_time.as_nanos() as u64)
                / total,
        );

        Ok(())
    }

    /// Get load balancer metrics
    pub async fn get_metrics(&self) -> LoadBalancerMetrics {
        let metrics = self.metrics.lock().await;
        LoadBalancerMetrics {
            total_requests: metrics.total_requests,
            successful_requests: metrics.successful_requests,
            failed_requests: metrics.failed_requests,
            retried_requests: metrics.retried_requests,
            circuit_breaker_trips: metrics.circuit_breaker_trips,
            avg_response_time: metrics.avg_response_time,
            instance_request_counts: metrics.instance_request_counts.clone(),
            instance_count: metrics.instance_count,
            healthy_instance_count: metrics.healthy_instance_count,
            auto_scaling_events: metrics.auto_scaling_events,
        }
    }

    /// Get current instances
    pub async fn get_instances(&self) -> Vec<BackendInstance> {
        let state = self.state.read().await;
        state.instances.clone()
    }

    // Private selection algorithms
    async fn select_round_robin(
        &self,
        instances: &[&BackendInstance],
        state: &mut LoadBalancerState,
    ) -> Option<BackendInstance> {
        if instances.is_empty() {
            return None;
        }

        let index = state.round_robin_index % instances.len();
        state.round_robin_index += 1;
        Some(instances[index].clone())
    }

    async fn select_weighted_round_robin(
        &self,
        instances: &[&BackendInstance],
        _state: &LoadBalancerState,
    ) -> Option<BackendInstance> {
        if instances.is_empty() {
            return None;
        }

        // Simple weighted selection based on cumulative weights
        let total_weight: f64 = instances.iter().map(|i| i.weight).sum();
        let mut rand_val = fastrand::f64() * total_weight;

        for instance in instances {
            rand_val -= instance.weight;
            if rand_val <= 0.0 {
                return Some((*instance).clone());
            }
        }

        // Fallback to first instance
        Some(instances[0].clone())
    }

    async fn select_least_connections(
        &self,
        instances: &[&BackendInstance],
    ) -> Option<BackendInstance> {
        instances.iter().min_by_key(|i| i.active_connections).map(|&i| i.clone())
    }

    async fn select_weighted_least_connections(
        &self,
        instances: &[&BackendInstance],
        _state: &LoadBalancerState,
    ) -> Option<BackendInstance> {
        instances
            .iter()
            .min_by(|a, b| {
                let a_ratio = a.active_connections as f64 / a.weight;
                let b_ratio = b.active_connections as f64 / b.weight;
                a_ratio.partial_cmp(&b_ratio).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|&i| i.clone())
    }

    async fn select_ip_hash(
        &self,
        instances: &[&BackendInstance],
        context: &RequestContext,
    ) -> Option<BackendInstance> {
        if instances.is_empty() {
            return None;
        }

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        context.client_ip.hash(&mut hasher);
        let hash = hasher.finish();
        let index = (hash as usize) % instances.len();
        Some(instances[index].clone())
    }

    async fn select_least_response_time(
        &self,
        instances: &[&BackendInstance],
    ) -> Option<BackendInstance> {
        instances
            .iter()
            .min_by_key(|i| i.response_time_stats.avg_response_time)
            .map(|&i| i.clone())
    }

    async fn select_resource_based(
        &self,
        instances: &[&BackendInstance],
        cpu_weight: f64,
        memory_weight: f64,
    ) -> Option<BackendInstance> {
        instances
            .iter()
            .min_by(|a, b| {
                let a_score = a.resource_utilization.cpu_usage * cpu_weight
                    + a.resource_utilization.memory_usage * memory_weight;
                let b_score = b.resource_utilization.cpu_usage * cpu_weight
                    + b.resource_utilization.memory_usage * memory_weight;
                a_score.partial_cmp(&b_score).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|&i| i.clone())
    }

    async fn select_geographic(
        &self,
        instances: &[&BackendInstance],
        _context: &RequestContext,
    ) -> Option<BackendInstance> {
        // Placeholder for geographic selection
        // Would use geolocation services to find nearest instance
        instances.first().map(|&i| i.clone())
    }

    async fn select_consistent_hash(
        &self,
        instances: &[&BackendInstance],
        context: &RequestContext,
        state: &LoadBalancerState,
    ) -> Option<BackendInstance> {
        if state.hash_ring.is_empty() {
            return instances.first().map(|&i| i.clone());
        }

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        context.client_ip.hash(&mut hasher);
        let hash = hasher.finish();

        // Find the next node in the ring
        let node = state
            .hash_ring
            .iter()
            .find(|node| node.hash >= hash)
            .or_else(|| state.hash_ring.first())?;

        instances.iter().find(|i| i.id == node.instance_id).map(|&i| i.clone())
    }

    async fn select_custom(
        &self,
        instances: &[&BackendInstance],
        _name: &str,
        _parameters: &HashMap<String, String>,
        _context: &RequestContext,
    ) -> Option<BackendInstance> {
        // Placeholder for custom selection logic
        instances.first().map(|&i| i.clone())
    }

    // Helper methods
    async fn rebuild_hash_ring(&self, state: &mut LoadBalancerState) {
        if let LoadBalancingAlgorithm::ConsistentHashing { virtual_nodes } = &self.config.algorithm
        {
            state.hash_ring.clear();

            for instance in &state.instances {
                for i in 0..*virtual_nodes {
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    format!("{}:{}", instance.id, i).hash(&mut hasher);
                    let hash = hasher.finish();

                    state.hash_ring.push(HashRingNode {
                        hash,
                        instance_id: instance.id.clone(),
                        virtual_node_id: i,
                    });
                }
            }

            state.hash_ring.sort_by_key(|node| node.hash);
        }
    }

    async fn get_session_instance(
        &self,
        context: &RequestContext,
        state: &LoadBalancerState,
    ) -> Option<String> {
        match &self.config.session_affinity.affinity_type {
            AffinityType::Cookie { name, .. } => {
                context.cookies.get(name).and_then(|cookie| state.sessions.get(cookie)).cloned()
            },
            AffinityType::IPAddress => state.sessions.get(&context.client_ip).cloned(),
            AffinityType::Header { name } => {
                context.headers.get(name).and_then(|header| state.sessions.get(header)).cloned()
            },
            AffinityType::Custom { identifier } => state.sessions.get(identifier).cloned(),
        }
    }

    async fn update_session_affinity(
        &self,
        context: &RequestContext,
        instance_id: &str,
        state: &mut LoadBalancerState,
    ) {
        let session_key = match &self.config.session_affinity.affinity_type {
            AffinityType::Cookie { name, .. } => context
                .cookies
                .get(name)
                .cloned()
                .unwrap_or_else(|| format!("session_{}", fastrand::u64(..))),
            AffinityType::IPAddress => context.client_ip.clone(),
            AffinityType::Header { name } => context.headers.get(name).cloned().unwrap_or_default(),
            AffinityType::Custom { identifier } => identifier.clone(),
        };

        state.sessions.insert(session_key, instance_id.to_string());
    }

    async fn update_response_time_stats(
        &self,
        stats: &mut ResponseTimeStats,
        response_time: Duration,
    ) {
        stats.sample_count += 1;

        // Update min/max
        if response_time < stats.min_response_time || stats.sample_count == 1 {
            stats.min_response_time = response_time;
        }
        if response_time > stats.max_response_time || stats.sample_count == 1 {
            stats.max_response_time = response_time;
        }

        // Update average (simple moving average)
        stats.avg_response_time = Duration::from_nanos(
            (stats.avg_response_time.as_nanos() as u64 * (stats.sample_count - 1) as u64
                + response_time.as_nanos() as u64)
                / stats.sample_count as u64,
        );

        // For simplicity, approximate percentiles (would use proper quantile estimation in production)
        stats.p95_response_time =
            Duration::from_nanos((stats.avg_response_time.as_nanos() as f64 * 1.5) as u64);
        stats.p99_response_time =
            Duration::from_nanos((stats.avg_response_time.as_nanos() as f64 * 2.0) as u64);
    }
}

/// Request context for load balancing decisions
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// Client IP address
    pub client_ip: String,
    /// Request headers
    pub headers: HashMap<String, String>,
    /// Request cookies
    pub cookies: HashMap<String, String>,
    /// Request path
    pub path: String,
    /// Request method
    pub method: String,
    /// User agent
    pub user_agent: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Default for LoadBalancerConfig {
    fn default() -> Self {
        Self {
            algorithm: LoadBalancingAlgorithm::RoundRobin,
            health_check: HealthCheckSettings {
                interval: Duration::from_secs(30),
                timeout: Duration::from_secs(5),
                failure_threshold: 3,
                success_threshold: 2,
                endpoint: "/health".to_string(),
                expected_codes: vec![200],
                passive_checks: true,
            },
            auto_scaling: AutoScalingConfig {
                enabled: false,
                min_instances: 1,
                max_instances: 10,
                target_cpu_utilization: 70.0,
                target_memory_utilization: 80.0,
                target_request_rate: 100.0,
                scale_up_cooldown: Duration::from_secs(300),
                scale_down_cooldown: Duration::from_secs(600),
                scaling_policies: Vec::new(),
            },
            session_affinity: SessionAffinityConfig {
                enabled: false,
                affinity_type: AffinityType::Cookie {
                    name: "lb_session".to_string(),
                    secure: true,
                },
                session_timeout: Duration::from_secs(3600),
                fallback_behavior: FallbackBehavior::LeastLoaded,
            },
            circuit_breaker: CircuitBreakerSettings {
                enabled: true,
                failure_threshold: 5,
                timeout: Duration::from_secs(60),
                half_open_max_requests: 3,
                recovery_time: Duration::from_secs(30),
            },
            retry_policy: RetryPolicy {
                max_retries: 3,
                base_delay: Duration::from_millis(100),
                max_delay: Duration::from_secs(10),
                backoff_multiplier: 2.0,
                jitter: true,
                retryable_conditions: vec![
                    RetryCondition::HttpStatus(502),
                    RetryCondition::HttpStatus(503),
                    RetryCondition::HttpStatus(504),
                    RetryCondition::Timeout,
                    RetryCondition::ConnectionError,
                ],
            },
            connection_pool: ConnectionPoolConfig {
                max_connections_per_instance: 100,
                connection_timeout: Duration::from_secs(10),
                idle_timeout: Duration::from_secs(300),
                keep_alive: Duration::from_secs(60),
                enabled: true,
            },
        }
    }
}

impl Default for ResponseTimeStats {
    fn default() -> Self {
        Self {
            avg_response_time: Duration::from_millis(0),
            min_response_time: Duration::from_millis(u64::MAX),
            max_response_time: Duration::from_millis(0),
            p95_response_time: Duration::from_millis(0),
            p99_response_time: Duration::from_millis(0),
            sample_count: 0,
        }
    }
}

impl Default for ResourceUtilization {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0.0,
            network_in: 0.0,
            network_out: 0.0,
            disk_io: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_load_balancer_creation() {
        let config = LoadBalancerConfig::default();
        let lb = LoadBalancer::new(config);

        let instances = lb.get_instances().await;
        assert!(instances.is_empty());
    }

    #[tokio::test]
    async fn test_add_instance() {
        let config = LoadBalancerConfig::default();
        let lb = LoadBalancer::new(config);

        let instance = BackendInstance {
            id: "test-1".to_string(),
            address: "127.0.0.1".to_string(),
            port: 8080,
            weight: 1.0,
            health_status: InstanceHealth::Healthy,
            active_connections: 0,
            response_time_stats: ResponseTimeStats::default(),
            resource_utilization: ResourceUtilization::default(),
            last_health_check: None,
            metadata: HashMap::new(),
        };

        let result = lb.add_instance(instance).await;
        assert!(result.is_ok());

        let instances = lb.get_instances().await;
        assert_eq!(instances.len(), 1);
    }

    #[tokio::test]
    async fn test_round_robin_selection() {
        let config = LoadBalancerConfig {
            algorithm: LoadBalancingAlgorithm::RoundRobin,
            ..Default::default()
        };
        let lb = LoadBalancer::new(config);

        // Add multiple instances
        for i in 0..3 {
            let instance = BackendInstance {
                id: format!("test-{}", i),
                address: "127.0.0.1".to_string(),
                port: 8080 + i as u16,
                weight: 1.0,
                health_status: InstanceHealth::Healthy,
                active_connections: 0,
                response_time_stats: ResponseTimeStats::default(),
                resource_utilization: ResourceUtilization::default(),
                last_health_check: None,
                metadata: HashMap::new(),
            };
            lb.add_instance(instance).await.unwrap();
        }

        let context = RequestContext {
            client_ip: "192.168.1.1".to_string(),
            headers: HashMap::new(),
            cookies: HashMap::new(),
            path: "/test".to_string(),
            method: "GET".to_string(),
            user_agent: "test".to_string(),
            metadata: HashMap::new(),
        };

        // Test round-robin behavior
        let selected1 = lb.select_instance(&context).await.unwrap().unwrap();
        let selected2 = lb.select_instance(&context).await.unwrap().unwrap();
        let selected3 = lb.select_instance(&context).await.unwrap().unwrap();
        let selected4 = lb.select_instance(&context).await.unwrap().unwrap();

        assert_eq!(selected1.id, "test-0");
        assert_eq!(selected2.id, "test-1");
        assert_eq!(selected3.id, "test-2");
        assert_eq!(selected4.id, "test-0"); // Should wrap around
    }

    #[tokio::test]
    async fn test_metrics_recording() {
        let config = LoadBalancerConfig::default();
        let lb = LoadBalancer::new(config);

        let instance = BackendInstance {
            id: "test-1".to_string(),
            address: "127.0.0.1".to_string(),
            port: 8080,
            weight: 1.0,
            health_status: InstanceHealth::Healthy,
            active_connections: 0,
            response_time_stats: ResponseTimeStats::default(),
            resource_utilization: ResourceUtilization::default(),
            last_health_check: None,
            metadata: HashMap::new(),
        };

        lb.add_instance(instance).await.unwrap();

        // Record some requests
        lb.record_request_result("test-1", true, Duration::from_millis(100))
            .await
            .unwrap();
        lb.record_request_result("test-1", false, Duration::from_millis(200))
            .await
            .unwrap();

        let metrics = lb.get_metrics().await;
        assert_eq!(metrics.total_requests, 2);
        assert_eq!(metrics.successful_requests, 1);
        assert_eq!(metrics.failed_requests, 1);
    }
}
