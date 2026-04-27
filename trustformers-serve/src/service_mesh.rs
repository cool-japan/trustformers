use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tracing::info;

/// Service mesh integration for horizontal scaling and traffic management
#[derive(Debug, Clone)]
pub struct ServiceMeshManager {
    config: ServiceMeshConfig,
    state: Arc<RwLock<ServiceMeshState>>,
    metrics: Arc<Mutex<ServiceMeshMetrics>>,
}

/// Configuration for service mesh integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMeshConfig {
    /// Type of service mesh to integrate with
    pub mesh_type: ServiceMeshType,
    /// Service name for registration
    pub service_name: String,
    /// Service version
    pub service_version: String,
    /// Namespace for service deployment
    pub namespace: String,
    /// Port for service communication
    pub port: u16,
    /// Health check configuration
    pub health_check: HealthCheckConfig,
    /// Traffic management settings
    pub traffic_management: TrafficManagementConfig,
    /// Security settings
    pub security: SecurityConfig,
    /// Retry and timeout settings
    pub reliability: ReliabilityConfig,
}

/// Supported service mesh types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceMeshType {
    /// Istio service mesh
    Istio,
    /// Linkerd service mesh
    Linkerd,
    /// Consul Connect
    ConsulConnect,
    /// AWS App Mesh
    AppMesh,
    /// Envoy proxy
    Envoy,
    /// Custom service mesh
    Custom { name: String },
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Health check endpoint path
    pub path: String,
    /// Check interval
    pub interval: Duration,
    /// Timeout for health checks
    pub timeout: Duration,
    /// Number of consecutive failures before marking unhealthy
    pub failure_threshold: u32,
    /// Number of consecutive successes before marking healthy
    pub success_threshold: u32,
}

/// Traffic management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficManagementConfig {
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Circuit breaker settings
    pub circuit_breaker: CircuitBreakerConfig,
    /// Retry policy
    pub retry_policy: RetryPolicyConfig,
    /// Rate limiting
    pub rate_limiting: RateLimitingConfig,
    /// Traffic splitting for canary deployments
    pub traffic_splitting: TrafficSplittingConfig,
}

/// Load balancing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastConnections,
    WeightedRoundRobin,
    IPHash,
    Random,
    LeastResponseTime,
    Custom { algorithm: String },
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    pub enabled: bool,
    pub failure_threshold: u32,
    pub timeout: Duration,
    pub success_threshold: u32,
}

/// Retry policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicyConfig {
    pub max_retries: u32,
    pub initial_interval: Duration,
    pub max_interval: Duration,
    pub multiplier: f64,
    pub retryable_status_codes: Vec<u16>,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitingConfig {
    pub enabled: bool,
    pub requests_per_second: u32,
    pub burst_size: u32,
    pub quota_window: Duration,
}

/// Traffic splitting configuration for canary deployments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficSplittingConfig {
    pub enabled: bool,
    pub canary_percentage: u8,
    pub header_based_routing: Vec<HeaderRoutingRule>,
    pub geo_based_routing: Option<GeoRoutingConfig>,
}

/// Header-based routing rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeaderRoutingRule {
    pub header_name: String,
    pub header_value: String,
    pub target_version: String,
}

/// Geo-based routing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoRoutingConfig {
    pub enabled: bool,
    pub region_mappings: HashMap<String, String>,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable mTLS
    pub mtls_enabled: bool,
    /// Certificate path
    pub cert_path: Option<String>,
    /// Private key path
    pub key_path: Option<String>,
    /// CA certificate path
    pub ca_cert_path: Option<String>,
    /// JWT authentication
    pub jwt_auth: JwtAuthConfig,
    /// RBAC settings
    pub rbac: RbacConfig,
}

/// JWT authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtAuthConfig {
    pub enabled: bool,
    pub issuer: Option<String>,
    pub audience: Option<String>,
    pub jwks_uri: Option<String>,
}

/// RBAC configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RbacConfig {
    pub enabled: bool,
    pub policies: Vec<RbacPolicy>,
}

/// RBAC policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RbacPolicy {
    pub name: String,
    pub subjects: Vec<String>,
    pub actions: Vec<String>,
    pub resources: Vec<String>,
}

/// Reliability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityConfig {
    pub timeout: Duration,
    pub keep_alive: Duration,
    pub connection_pool_size: u32,
    pub idle_timeout: Duration,
}

/// Internal state of service mesh manager
#[derive(Debug)]
struct ServiceMeshState {
    /// Whether the service is registered
    registered: bool,
    /// Last registration time
    last_registration: Option<Instant>,
    /// Service endpoints
    endpoints: Vec<ServiceEndpoint>,
    /// Current health status
    health_status: HealthStatus,
    /// Traffic management rules
    traffic_rules: Vec<TrafficRule>,
}

/// Service endpoint information
#[derive(Debug, Clone)]
pub struct ServiceEndpoint {
    pub id: String,
    pub address: String,
    pub port: u16,
    pub weight: u32,
    pub health_status: HealthStatus,
    pub metadata: HashMap<String, String>,
}

/// Health status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Unhealthy,
    Warning,
    Unknown,
}

/// Traffic management rule
#[derive(Debug, Clone)]
pub struct TrafficRule {
    pub id: String,
    pub rule_type: TrafficRuleType,
    pub conditions: Vec<TrafficCondition>,
    pub actions: Vec<TrafficAction>,
    pub priority: u32,
}

/// Traffic rule type
#[derive(Debug, Clone)]
pub enum TrafficRuleType {
    Routing,
    RateLimit,
    CircuitBreaker,
    Retry,
    Timeout,
}

/// Traffic condition
#[derive(Debug, Clone)]
pub struct TrafficCondition {
    pub condition_type: ConditionType,
    pub value: String,
    pub operator: ConditionOperator,
}

/// Condition type
#[derive(Debug, Clone)]
pub enum ConditionType {
    Header,
    Path,
    Method,
    SourceIP,
    UserAgent,
}

/// Condition operator
#[derive(Debug, Clone)]
pub enum ConditionOperator {
    Equals,
    NotEquals,
    Contains,
    StartsWith,
    EndsWith,
    Regex,
}

/// Traffic action
#[derive(Debug, Clone)]
pub struct TrafficAction {
    pub action_type: ActionType,
    pub parameters: HashMap<String, String>,
}

/// Action type
#[derive(Debug, Clone)]
pub enum ActionType {
    Route,
    Deny,
    RateLimit,
    AddHeader,
    RemoveHeader,
    Rewrite,
}

/// Service mesh metrics
#[derive(Debug, Default)]
pub struct ServiceMeshMetrics {
    /// Total requests processed
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Average response time
    pub avg_response_time: Duration,
    /// Circuit breaker trips
    pub circuit_breaker_trips: u64,
    /// Rate limit violations
    pub rate_limit_violations: u64,
    /// Current active connections
    pub active_connections: u32,
    /// Service registration count
    pub registration_count: u64,
    /// Health check failures
    pub health_check_failures: u64,
}

impl ServiceMeshManager {
    /// Create a new service mesh manager
    pub fn new(config: ServiceMeshConfig) -> Self {
        let state = ServiceMeshState {
            registered: false,
            last_registration: None,
            endpoints: Vec::new(),
            health_status: HealthStatus::Unknown,
            traffic_rules: Vec::new(),
        };

        Self {
            config,
            state: Arc::new(RwLock::new(state)),
            metrics: Arc::new(Mutex::new(ServiceMeshMetrics::default())),
        }
    }

    /// Register service with the mesh
    pub async fn register_service(&self) -> Result<()> {
        let mut state = self.state.write().await;
        let mut metrics = self.metrics.lock().await;

        match self.config.mesh_type {
            ServiceMeshType::Istio => self.register_with_istio().await?,
            ServiceMeshType::Linkerd => self.register_with_linkerd().await?,
            ServiceMeshType::ConsulConnect => self.register_with_consul().await?,
            ServiceMeshType::AppMesh => self.register_with_app_mesh().await?,
            ServiceMeshType::Envoy => self.register_with_envoy().await?,
            ServiceMeshType::Custom { ref name } => self.register_with_custom_mesh(name).await?,
        }

        state.registered = true;
        state.last_registration = Some(Instant::now());
        metrics.registration_count += 1;

        info!(
            "Service '{}' registered with {:?} mesh",
            self.config.service_name, self.config.mesh_type
        );

        Ok(())
    }

    /// Deregister service from the mesh
    pub async fn deregister_service(&self) -> Result<()> {
        let mut state = self.state.write().await;

        match self.config.mesh_type {
            ServiceMeshType::Istio => self.deregister_from_istio().await?,
            ServiceMeshType::Linkerd => self.deregister_from_linkerd().await?,
            ServiceMeshType::ConsulConnect => self.deregister_from_consul().await?,
            ServiceMeshType::AppMesh => self.deregister_from_app_mesh().await?,
            ServiceMeshType::Envoy => self.deregister_from_envoy().await?,
            ServiceMeshType::Custom { ref name } => self.deregister_from_custom_mesh(name).await?,
        }

        state.registered = false;
        state.last_registration = None;

        info!(
            "Service '{}' deregistered from {:?} mesh",
            self.config.service_name, self.config.mesh_type
        );

        Ok(())
    }

    /// Update service health status
    pub async fn update_health_status(&self, status: HealthStatus) -> Result<()> {
        let mut state = self.state.write().await;
        state.health_status = status;

        match self.config.mesh_type {
            ServiceMeshType::Istio => self.update_istio_health().await?,
            ServiceMeshType::Linkerd => self.update_linkerd_health().await?,
            ServiceMeshType::ConsulConnect => self.update_consul_health().await?,
            ServiceMeshType::AppMesh => self.update_app_mesh_health().await?,
            ServiceMeshType::Envoy => self.update_envoy_health().await?,
            ServiceMeshType::Custom { ref name } => self.update_custom_mesh_health(name).await?,
        }

        Ok(())
    }

    /// Configure traffic management rules
    pub async fn configure_traffic_rules(&self, rules: Vec<TrafficRule>) -> Result<()> {
        let mut state = self.state.write().await;
        state.traffic_rules = rules;

        match self.config.mesh_type {
            ServiceMeshType::Istio => self.configure_istio_traffic().await?,
            ServiceMeshType::Linkerd => self.configure_linkerd_traffic().await?,
            ServiceMeshType::ConsulConnect => self.configure_consul_traffic().await?,
            ServiceMeshType::AppMesh => self.configure_app_mesh_traffic().await?,
            ServiceMeshType::Envoy => self.configure_envoy_traffic().await?,
            ServiceMeshType::Custom { ref name } => {
                self.configure_custom_mesh_traffic(name).await?
            },
        }

        info!("Configured {} traffic rules", state.traffic_rules.len());
        Ok(())
    }

    /// Get service endpoints
    pub async fn get_endpoints(&self) -> Vec<ServiceEndpoint> {
        let state = self.state.read().await;
        state.endpoints.clone()
    }

    /// Get service mesh metrics
    pub async fn get_metrics(&self) -> ServiceMeshMetrics {
        let metrics = self.metrics.lock().await;
        ServiceMeshMetrics {
            total_requests: metrics.total_requests,
            successful_requests: metrics.successful_requests,
            failed_requests: metrics.failed_requests,
            avg_response_time: metrics.avg_response_time,
            circuit_breaker_trips: metrics.circuit_breaker_trips,
            rate_limit_violations: metrics.rate_limit_violations,
            active_connections: metrics.active_connections,
            registration_count: metrics.registration_count,
            health_check_failures: metrics.health_check_failures,
        }
    }

    /// Record request metrics
    pub async fn record_request(&self, success: bool, response_time: Duration) {
        let mut metrics = self.metrics.lock().await;
        metrics.total_requests += 1;

        if success {
            metrics.successful_requests += 1;
        } else {
            metrics.failed_requests += 1;
        }

        // Update average response time (simple moving average)
        let total = metrics.successful_requests + metrics.failed_requests;
        metrics.avg_response_time = Duration::from_nanos(
            (metrics.avg_response_time.as_nanos() as u64 * (total - 1)
                + response_time.as_nanos() as u64)
                / total,
        );
    }

    // Private implementation methods for different service meshes
    async fn register_with_istio(&self) -> Result<()> {
        // Implementation would integrate with Istio control plane
        // This would typically involve creating ServiceEntry, DestinationRule, and VirtualService resources
        info!("Registering with Istio service mesh");
        // Placeholder for actual Istio integration
        Ok(())
    }

    async fn register_with_linkerd(&self) -> Result<()> {
        // Implementation would integrate with Linkerd control plane
        info!("Registering with Linkerd service mesh");
        // Placeholder for actual Linkerd integration
        Ok(())
    }

    async fn register_with_consul(&self) -> Result<()> {
        // Implementation would integrate with Consul Connect
        info!("Registering with Consul Connect");
        // Placeholder for actual Consul integration
        Ok(())
    }

    async fn register_with_app_mesh(&self) -> Result<()> {
        // Implementation would integrate with AWS App Mesh
        info!("Registering with AWS App Mesh");
        // Placeholder for actual App Mesh integration
        Ok(())
    }

    async fn register_with_envoy(&self) -> Result<()> {
        // Implementation would configure Envoy proxy
        info!("Registering with Envoy proxy");
        // Placeholder for actual Envoy integration
        Ok(())
    }

    async fn register_with_custom_mesh(&self, name: &str) -> Result<()> {
        info!("Registering with custom mesh: {}", name);
        // Placeholder for custom mesh integration
        Ok(())
    }

    // Similar deregistration methods
    async fn deregister_from_istio(&self) -> Result<()> {
        info!("Deregistering from Istio service mesh");
        Ok(())
    }

    async fn deregister_from_linkerd(&self) -> Result<()> {
        info!("Deregistering from Linkerd service mesh");
        Ok(())
    }

    async fn deregister_from_consul(&self) -> Result<()> {
        info!("Deregistering from Consul Connect");
        Ok(())
    }

    async fn deregister_from_app_mesh(&self) -> Result<()> {
        info!("Deregistering from AWS App Mesh");
        Ok(())
    }

    async fn deregister_from_envoy(&self) -> Result<()> {
        info!("Deregistering from Envoy proxy");
        Ok(())
    }

    async fn deregister_from_custom_mesh(&self, name: &str) -> Result<()> {
        info!("Deregistering from custom mesh: {}", name);
        Ok(())
    }

    // Health update methods
    async fn update_istio_health(&self) -> Result<()> {
        // Update health status in Istio
        Ok(())
    }

    async fn update_linkerd_health(&self) -> Result<()> {
        // Update health status in Linkerd
        Ok(())
    }

    async fn update_consul_health(&self) -> Result<()> {
        // Update health status in Consul
        Ok(())
    }

    async fn update_app_mesh_health(&self) -> Result<()> {
        // Update health status in App Mesh
        Ok(())
    }

    async fn update_envoy_health(&self) -> Result<()> {
        // Update health status in Envoy
        Ok(())
    }

    async fn update_custom_mesh_health(&self, _name: &str) -> Result<()> {
        // Update health status in custom mesh
        Ok(())
    }

    // Traffic configuration methods
    async fn configure_istio_traffic(&self) -> Result<()> {
        // Configure Istio traffic management
        Ok(())
    }

    async fn configure_linkerd_traffic(&self) -> Result<()> {
        // Configure Linkerd traffic management
        Ok(())
    }

    async fn configure_consul_traffic(&self) -> Result<()> {
        // Configure Consul traffic management
        Ok(())
    }

    async fn configure_app_mesh_traffic(&self) -> Result<()> {
        // Configure App Mesh traffic management
        Ok(())
    }

    async fn configure_envoy_traffic(&self) -> Result<()> {
        // Configure Envoy traffic management
        Ok(())
    }

    async fn configure_custom_mesh_traffic(&self, _name: &str) -> Result<()> {
        // Configure custom mesh traffic management
        Ok(())
    }
}

impl Default for ServiceMeshConfig {
    fn default() -> Self {
        Self {
            mesh_type: ServiceMeshType::Istio,
            service_name: "trustformers-serve".to_string(),
            service_version: "v1".to_string(),
            namespace: "default".to_string(),
            port: 8080,
            health_check: HealthCheckConfig {
                path: "/health".to_string(),
                interval: Duration::from_secs(30),
                timeout: Duration::from_secs(5),
                failure_threshold: 3,
                success_threshold: 2,
            },
            traffic_management: TrafficManagementConfig {
                load_balancing: LoadBalancingStrategy::RoundRobin,
                circuit_breaker: CircuitBreakerConfig {
                    enabled: true,
                    failure_threshold: 5,
                    timeout: Duration::from_secs(60),
                    success_threshold: 3,
                },
                retry_policy: RetryPolicyConfig {
                    max_retries: 3,
                    initial_interval: Duration::from_millis(100),
                    max_interval: Duration::from_secs(10),
                    multiplier: 2.0,
                    retryable_status_codes: vec![500, 502, 503, 504],
                },
                rate_limiting: RateLimitingConfig {
                    enabled: true,
                    requests_per_second: 1000,
                    burst_size: 100,
                    quota_window: Duration::from_secs(60),
                },
                traffic_splitting: TrafficSplittingConfig {
                    enabled: false,
                    canary_percentage: 10,
                    header_based_routing: Vec::new(),
                    geo_based_routing: None,
                },
            },
            security: SecurityConfig {
                mtls_enabled: true,
                cert_path: None,
                key_path: None,
                ca_cert_path: None,
                jwt_auth: JwtAuthConfig {
                    enabled: false,
                    issuer: None,
                    audience: None,
                    jwks_uri: None,
                },
                rbac: RbacConfig {
                    enabled: false,
                    policies: Vec::new(),
                },
            },
            reliability: ReliabilityConfig {
                timeout: Duration::from_secs(30),
                keep_alive: Duration::from_secs(60),
                connection_pool_size: 100,
                idle_timeout: Duration::from_secs(300),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_service_mesh_manager_creation() {
        let config = ServiceMeshConfig::default();
        let manager = ServiceMeshManager::new(config);

        let state = manager.state.read().await;
        assert!(!state.registered);
        assert!(state.endpoints.is_empty());
    }

    #[tokio::test]
    async fn test_service_registration() {
        let config = ServiceMeshConfig::default();
        let manager = ServiceMeshManager::new(config);

        let result = manager.register_service().await;
        assert!(result.is_ok());

        let state = manager.state.read().await;
        assert!(state.registered);
        assert!(state.last_registration.is_some());
    }

    #[tokio::test]
    async fn test_health_status_update() {
        let config = ServiceMeshConfig::default();
        let manager = ServiceMeshManager::new(config);

        let result = manager.update_health_status(HealthStatus::Healthy).await;
        assert!(result.is_ok());

        let state = manager.state.read().await;
        matches!(state.health_status, HealthStatus::Healthy);
    }

    #[tokio::test]
    async fn test_metrics_recording() {
        let config = ServiceMeshConfig::default();
        let manager = ServiceMeshManager::new(config);

        manager.record_request(true, Duration::from_millis(100)).await;
        manager.record_request(false, Duration::from_millis(200)).await;

        let metrics = manager.get_metrics().await;
        assert_eq!(metrics.total_requests, 2);
        assert_eq!(metrics.successful_requests, 1);
        assert_eq!(metrics.failed_requests, 1);
    }

    #[tokio::test]
    async fn test_traffic_rule_configuration() {
        let config = ServiceMeshConfig::default();
        let manager = ServiceMeshManager::new(config);

        let rules = vec![TrafficRule {
            id: "test-rule".to_string(),
            rule_type: TrafficRuleType::Routing,
            conditions: vec![],
            actions: vec![],
            priority: 1,
        }];

        let result = manager.configure_traffic_rules(rules).await;
        assert!(result.is_ok());

        let state = manager.state.read().await;
        assert_eq!(state.traffic_rules.len(), 1);
    }

    #[test]
    fn test_default_config() {
        let config = ServiceMeshConfig::default();
        assert!(matches!(config.mesh_type, ServiceMeshType::Istio));
        assert_eq!(config.service_name, "trustformers-serve");
        assert_eq!(config.namespace, "default");
        assert_eq!(config.port, 8080);
        assert!(config.security.mtls_enabled);
    }

    #[tokio::test]
    async fn test_unhealthy_status() {
        let manager = ServiceMeshManager::new(ServiceMeshConfig::default());
        let result = manager.update_health_status(HealthStatus::Unhealthy).await;
        assert!(result.is_ok());
        let state = manager.state.read().await;
        assert!(matches!(state.health_status, HealthStatus::Unhealthy));
    }

    #[tokio::test]
    async fn test_degraded_status() {
        let manager = ServiceMeshManager::new(ServiceMeshConfig::default());
        let result = manager.update_health_status(HealthStatus::Warning).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_record_many_requests() {
        let manager = ServiceMeshManager::new(ServiceMeshConfig::default());

        let mut lcg: u64 = 42;
        for _ in 0..20 {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
            let success = !lcg.is_multiple_of(3); // ~66% success rate
            let latency_ms = lcg % 500;
            manager.record_request(success, Duration::from_millis(latency_ms)).await;
        }

        let metrics = manager.get_metrics().await;
        assert_eq!(metrics.total_requests, 20);
        assert!(metrics.successful_requests > 0);
        assert!(metrics.failed_requests > 0);
    }

    #[tokio::test]
    async fn test_multiple_traffic_rules() {
        let manager = ServiceMeshManager::new(ServiceMeshConfig::default());

        let rules = vec![
            TrafficRule {
                id: "route-1".to_string(),
                rule_type: TrafficRuleType::Routing,
                conditions: vec![],
                actions: vec![],
                priority: 1,
            },
            TrafficRule {
                id: "rate-limit-1".to_string(),
                rule_type: TrafficRuleType::RateLimit,
                conditions: vec![],
                actions: vec![],
                priority: 2,
            },
        ];

        let result = manager.configure_traffic_rules(rules).await;
        assert!(result.is_ok());

        let state = manager.state.read().await;
        assert_eq!(state.traffic_rules.len(), 2);
    }

    #[tokio::test]
    async fn test_metrics_initial_state() {
        let manager = ServiceMeshManager::new(ServiceMeshConfig::default());
        let metrics = manager.get_metrics().await;
        assert_eq!(metrics.total_requests, 0);
        assert_eq!(metrics.successful_requests, 0);
        assert_eq!(metrics.failed_requests, 0);
    }

    #[test]
    fn test_service_mesh_type_debug() {
        let types = vec![
            ServiceMeshType::Istio,
            ServiceMeshType::Linkerd,
            ServiceMeshType::ConsulConnect,
            ServiceMeshType::AppMesh,
            ServiceMeshType::Envoy,
            ServiceMeshType::Custom {
                name: "custom".to_string(),
            },
        ];
        for t in types {
            assert!(!format!("{:?}", t).is_empty());
        }
    }

    #[test]
    fn test_load_balancing_strategy_debug() {
        let strategies = vec![
            LoadBalancingStrategy::RoundRobin,
            LoadBalancingStrategy::LeastConnections,
            LoadBalancingStrategy::WeightedRoundRobin,
            LoadBalancingStrategy::IPHash,
            LoadBalancingStrategy::Random,
            LoadBalancingStrategy::LeastResponseTime,
            LoadBalancingStrategy::Custom {
                algorithm: "test".to_string(),
            },
        ];
        for s in strategies {
            assert!(!format!("{:?}", s).is_empty());
        }
    }

    #[tokio::test]
    async fn test_register_and_metrics() {
        let manager = ServiceMeshManager::new(ServiceMeshConfig::default());
        manager.register_service().await.expect("register ok");

        // After registration, do some requests
        manager.record_request(true, Duration::from_millis(50)).await;
        manager.record_request(true, Duration::from_millis(100)).await;

        let metrics = manager.get_metrics().await;
        assert_eq!(metrics.total_requests, 2);
        assert_eq!(metrics.successful_requests, 2);
    }

    #[tokio::test]
    async fn test_empty_traffic_rules() {
        let manager = ServiceMeshManager::new(ServiceMeshConfig::default());
        let result = manager.configure_traffic_rules(vec![]).await;
        assert!(result.is_ok());

        let state = manager.state.read().await;
        assert!(state.traffic_rules.is_empty());
    }

    #[tokio::test]
    async fn test_health_check_config() {
        let config = ServiceMeshConfig::default();
        assert_eq!(config.health_check.path, "/health");
        assert_eq!(config.health_check.failure_threshold, 3);
        assert!(config.health_check.success_threshold > 0);
    }

    #[tokio::test]
    async fn test_traffic_rule_types() {
        let types = vec![
            TrafficRuleType::Routing,
            TrafficRuleType::RateLimit,
            TrafficRuleType::CircuitBreaker,
            TrafficRuleType::Retry,
            TrafficRuleType::Timeout,
        ];
        for t in types {
            assert!(!format!("{:?}", t).is_empty());
        }
    }

    #[tokio::test]
    async fn test_manager_clone_shares_state() {
        let manager = ServiceMeshManager::new(ServiceMeshConfig::default());
        let cloned = manager.clone();

        manager.record_request(true, Duration::from_millis(10)).await;

        let metrics = cloned.get_metrics().await;
        assert_eq!(metrics.total_requests, 1);
    }

    #[tokio::test]
    async fn test_all_success_requests() {
        let manager = ServiceMeshManager::new(ServiceMeshConfig::default());
        for _ in 0..10 {
            manager.record_request(true, Duration::from_millis(50)).await;
        }
        let metrics = manager.get_metrics().await;
        assert_eq!(metrics.total_requests, 10);
        assert_eq!(metrics.successful_requests, 10);
        assert_eq!(metrics.failed_requests, 0);
    }

    #[tokio::test]
    async fn test_all_failure_requests() {
        let manager = ServiceMeshManager::new(ServiceMeshConfig::default());
        for _ in 0..10 {
            manager.record_request(false, Duration::from_millis(200)).await;
        }
        let metrics = manager.get_metrics().await;
        assert_eq!(metrics.total_requests, 10);
        assert_eq!(metrics.successful_requests, 0);
        assert_eq!(metrics.failed_requests, 10);
    }

    #[test]
    fn test_traffic_rule_priority() {
        let rule_a = TrafficRule {
            id: "a".to_string(),
            rule_type: TrafficRuleType::Routing,
            conditions: vec![],
            actions: vec![],
            priority: 1,
        };
        let rule_b = TrafficRule {
            id: "b".to_string(),
            rule_type: TrafficRuleType::RateLimit,
            conditions: vec![],
            actions: vec![],
            priority: 10,
        };
        assert!(rule_a.priority < rule_b.priority);
    }

    #[tokio::test]
    async fn test_register_twice() {
        let manager = ServiceMeshManager::new(ServiceMeshConfig::default());
        manager.register_service().await.expect("first register ok");
        manager.register_service().await.expect("second register ok");
        let state = manager.state.read().await;
        assert!(state.registered);
    }

    #[tokio::test]
    async fn test_health_transitions() {
        let manager = ServiceMeshManager::new(ServiceMeshConfig::default());
        manager.update_health_status(HealthStatus::Healthy).await.expect("ok");
        manager.update_health_status(HealthStatus::Warning).await.expect("ok");
        manager.update_health_status(HealthStatus::Unhealthy).await.expect("ok");
        manager.update_health_status(HealthStatus::Unknown).await.expect("ok");
        manager.update_health_status(HealthStatus::Healthy).await.expect("ok");
    }

    #[test]
    fn test_health_status_debug() {
        let statuses = vec![
            HealthStatus::Healthy,
            HealthStatus::Unhealthy,
            HealthStatus::Warning,
            HealthStatus::Unknown,
        ];
        for s in statuses {
            assert!(!format!("{:?}", s).is_empty());
        }
    }

    #[tokio::test]
    async fn test_overwrite_traffic_rules() {
        let manager = ServiceMeshManager::new(ServiceMeshConfig::default());

        let rules1 = vec![TrafficRule {
            id: "r1".to_string(),
            rule_type: TrafficRuleType::Routing,
            conditions: vec![],
            actions: vec![],
            priority: 1,
        }];
        manager.configure_traffic_rules(rules1).await.expect("ok");

        let rules2 = vec![
            TrafficRule {
                id: "r2".to_string(),
                rule_type: TrafficRuleType::Timeout,
                conditions: vec![],
                actions: vec![],
                priority: 1,
            },
            TrafficRule {
                id: "r3".to_string(),
                rule_type: TrafficRuleType::CircuitBreaker,
                conditions: vec![],
                actions: vec![],
                priority: 2,
            },
        ];
        manager.configure_traffic_rules(rules2).await.expect("ok");

        let state = manager.state.read().await;
        assert_eq!(state.traffic_rules.len(), 2);
    }

    #[test]
    fn test_config_serialization() {
        let config = ServiceMeshConfig::default();
        let json = serde_json::to_string(&config);
        assert!(json.is_ok());
    }

    #[tokio::test]
    async fn test_concurrent_metric_recording() {
        let manager = ServiceMeshManager::new(ServiceMeshConfig::default());
        let m1 = manager.clone();
        let m2 = manager.clone();

        let h1 = tokio::spawn(async move {
            for _ in 0..10 {
                m1.record_request(true, Duration::from_millis(10)).await;
            }
        });
        let h2 = tokio::spawn(async move {
            for _ in 0..10 {
                m2.record_request(false, Duration::from_millis(10)).await;
            }
        });

        h1.await.expect("join ok");
        h2.await.expect("join ok");

        let metrics = manager.get_metrics().await;
        assert_eq!(metrics.total_requests, 20);
    }
}
