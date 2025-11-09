//! Container type definitions for TrustformeRS C API
//!
//! This module contains all the type definitions for container orchestration,
//! including configuration structures, enums, and supporting types.

use crate::error::{TrustformersError, TrustformersResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Container platform types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContainerPlatform {
    /// Docker containerization
    Docker,
    /// Kubernetes orchestration
    Kubernetes,
    /// Docker Swarm
    DockerSwarm,
    /// Amazon ECS
    AmazonECS,
    /// Google Cloud Run
    GoogleCloudRun,
    /// Azure Container Instances
    AzureContainerInstances,
    /// Red Hat OpenShift
    OpenShift,
}

/// Container deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerDeploymentConfig {
    /// Target platform
    pub platform: ContainerPlatform,
    /// Application name
    pub app_name: String,
    /// Environment (dev, staging, prod)
    pub environment: String,
    /// Container image configuration
    pub image_config: DockerImageConfig,
    /// Orchestration configuration
    pub orchestration: OrchestrationConfig,
    /// Serverless optimization
    pub serverless: Option<ServerlessConfig>,
    /// Monitoring configuration
    pub monitoring: MonitoringConfig,
    /// Auto-scaling configuration
    pub auto_scaling: Option<AutoScalingConfig>,
    /// Security configuration
    pub security: SecurityConfig,
    /// Network configuration
    pub network: NetworkConfig,
    /// Storage configuration
    pub storage: StorageConfig,
}

/// Orchestration configuration for different platforms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrchestrationConfig {
    /// Docker Swarm configuration
    DockerSwarm(DockerSwarmConfig),
    /// Amazon ECS configuration
    ECS(ECSConfig),
    /// Google Cloud Run configuration
    CloudRun(CloudRunConfig),
    /// Azure Container Instances
    ACI(ACIConfig),
    /// OpenShift configuration
    OpenShift(OpenShiftConfig),
}

/// Docker Swarm orchestration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerSwarmConfig {
    /// Number of replicas
    pub replicas: u32,
    /// Update configuration
    pub update_config: SwarmUpdateConfig,
    /// Resource limits
    pub resources: SwarmResourceLimits,
    /// Node constraints
    pub constraints: Vec<String>,
    /// Service labels
    pub labels: HashMap<String, String>,
    /// Networks to attach to
    pub networks: Vec<String>,
}

/// Swarm service update configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmUpdateConfig {
    /// Number of containers to update at once
    pub parallelism: u32,
    /// Delay between updates
    pub delay: String,
    /// Action on failure
    pub failure_action: SwarmFailureAction,
    /// Update order
    pub order: SwarmUpdateOrder,
}

/// Actions to take when update fails
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmFailureAction {
    Pause,
    Continue,
    Rollback,
}

/// Order for updating services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmUpdateOrder {
    StopFirst,
    StartFirst,
}

/// Resource limits for Swarm services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmResourceLimits {
    /// CPU limit (in cores)
    pub cpu_limit: f64,
    /// Memory limit (in MB)
    pub memory_limit: u64,
    /// CPU reservation
    pub cpu_reservation: Option<f64>,
    /// Memory reservation
    pub memory_reservation: Option<u64>,
}

/// Amazon ECS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ECSConfig {
    /// ECS cluster name
    pub cluster_name: String,
    /// Service name
    pub service_name: String,
    /// Task definition family
    pub task_definition_family: String,
    /// Desired count
    pub desired_count: u32,
    /// Launch type
    pub launch_type: ECSLaunchType,
    /// Network mode
    pub network_mode: ECSNetworkMode,
    /// Load balancer configuration
    pub load_balancer: Option<ECSLoadBalancer>,
    /// Auto scaling configuration
    pub auto_scaling: Option<ECSAutoScalingConfig>,
}

/// ECS network modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ECSNetworkMode {
    Bridge,
    Host,
    AwsVpc,
    None,
}

/// ECS launch types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ECSLaunchType {
    EC2,
    Fargate,
    External,
}

/// ECS load balancer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ECSLoadBalancer {
    /// Target group ARN
    pub target_group_arn: String,
    /// Container name
    pub container_name: String,
    /// Container port
    pub container_port: u16,
}

/// Google Cloud Run configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudRunConfig {
    /// Service name
    pub service_name: String,
    /// Region
    pub region: String,
    /// Memory allocation (in GB)
    pub memory: f64,
    /// CPU allocation
    pub cpu: f64,
    /// Maximum instances
    pub max_instances: u32,
    /// Minimum instances
    pub min_instances: u32,
    /// Request timeout (in seconds)
    pub timeout: u32,
    /// Concurrency per instance
    pub concurrency: u32,
}

/// Serverless optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerlessConfig {
    /// Cold start optimization
    pub cold_start: ColdStartConfig,
    /// Resource optimization
    pub resource_optimization: ResourceOptimizationConfig,
    /// Memory optimization
    pub memory_optimization: MemoryOptimization,
    /// Event-driven scaling
    pub event_scaling: EventDrivenScalingConfig,
}

/// Cold start optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdStartConfig {
    /// Enable pre-warming
    pub enable_prewarming: bool,
    /// Pre-warm instance count
    pub prewarm_instances: u32,
    /// Keep-alive duration (in seconds)
    pub keep_alive_duration: u32,
    /// Lazy loading strategy
    pub lazy_loading: bool,
    /// Initialization optimization
    pub init_optimization: InitOptimizationConfig,
}

/// Resource optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceOptimizationConfig {
    /// Enable automatic resource adjustment
    pub auto_adjust_resources: bool,
    /// CPU utilization threshold
    pub cpu_threshold: f64,
    /// Memory utilization threshold
    pub memory_threshold: f64,
    /// Scaling policies
    pub scaling_policies: Vec<ScalingPolicy>,
}

/// Memory optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimization {
    /// Enable garbage collection tuning
    pub gc_tuning: bool,
    /// Memory pool optimization
    pub memory_pool_optimization: bool,
    /// Buffer management strategy
    pub buffer_management: BufferManagementStrategy,
    /// Cache optimization
    pub cache_optimization: CacheOptimizationConfig,
}

/// Event-driven scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventDrivenScalingConfig {
    /// Enable event-driven scaling
    pub enabled: bool,
    /// Queue depth thresholds
    pub queue_thresholds: HashMap<String, u32>,
    /// Response time targets
    pub response_time_targets: HashMap<String, f64>,
    /// Custom metrics
    pub custom_metrics: Vec<CustomMetricConfig>,
}

/// Scaling policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingPolicy {
    /// Metric name
    pub metric_name: String,
    /// Target value
    pub target_value: f64,
    /// Scale-out cooldown (seconds)
    pub scale_out_cooldown: u32,
    /// Scale-in cooldown (seconds)
    pub scale_in_cooldown: u32,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable container monitoring
    pub enabled: bool,
    /// Metrics collection interval (seconds)
    pub metrics_interval: u32,
    /// Log aggregation
    pub log_aggregation: LogAggregationConfig,
    /// Health check configuration
    pub health_checks: HealthCheckConfig,
    /// Alerting configuration
    pub alerting: AlertingConfig,
}

/// Auto-scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScalingConfig {
    /// Enable auto-scaling
    pub enabled: bool,
    /// Minimum instances
    pub min_instances: u32,
    /// Maximum instances
    pub max_instances: u32,
    /// Target CPU utilization
    pub target_cpu_utilization: f64,
    /// Target memory utilization
    pub target_memory_utilization: f64,
    /// Scaling policies
    pub scaling_policies: Vec<ScalingPolicy>,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable security scanning
    pub security_scanning: bool,
    /// Image scanning
    pub image_scanning: ImageScanningConfig,
    /// Runtime security
    pub runtime_security: RuntimeSecurityConfig,
    /// Network policies
    pub network_policies: Vec<NetworkPolicy>,
    /// Secret management
    pub secret_management: SecretManagementConfig,
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Network mode
    pub network_mode: String,
    /// Port mappings
    pub port_mappings: Vec<PortMapping>,
    /// Load balancer configuration
    pub load_balancer: Option<LoadBalancerConfig>,
    /// Service mesh integration
    pub service_mesh: Option<ServiceMeshConfig>,
}

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Persistent volumes
    pub persistent_volumes: Vec<PersistentVolumeConfig>,
    /// Temporary storage
    pub temp_storage: TempStorageConfig,
    /// Backup configuration
    pub backup: Option<BackupConfig>,
}

// Additional supporting types

/// Docker image configuration (imported from docker module)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerImageConfig {
    /// Base image
    pub base_image: String,
    /// Image tag
    pub tag: String,
    /// Build context
    pub build_context: String,
    /// Dockerfile path
    pub dockerfile_path: String,
    /// Build arguments
    pub build_args: HashMap<String, String>,
    /// Environment variables
    pub env_vars: HashMap<String, String>,
    /// Exposed ports
    pub exposed_ports: Vec<u16>,
    /// Volume mounts
    pub volumes: Vec<VolumeMount>,
}

/// Azure Container Instances configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ACIConfig {
    /// Resource group
    pub resource_group: String,
    /// Container group name
    pub container_group_name: String,
    /// Location
    pub location: String,
    /// CPU cores
    pub cpu: f64,
    /// Memory (in GB)
    pub memory: f64,
    /// Restart policy
    pub restart_policy: String,
}

/// OpenShift configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenShiftConfig {
    /// Project name
    pub project: String,
    /// Application name
    pub application: String,
    /// Build configuration
    pub build_config: BuildConfig,
    /// Deployment configuration
    pub deployment_config: DeploymentConfig,
}

/// ECS Auto Scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ECSAutoScalingConfig {
    /// Minimum capacity
    pub min_capacity: u32,
    /// Maximum capacity
    pub max_capacity: u32,
    /// Target CPU utilization
    pub target_cpu: f64,
    /// Target memory utilization
    pub target_memory: f64,
}

/// Initialization optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitOptimizationConfig {
    /// Parallel initialization
    pub parallel_init: bool,
    /// Dependency caching
    pub dependency_caching: bool,
    /// Resource pre-allocation
    pub resource_preallocation: bool,
}

/// Buffer management strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BufferManagementStrategy {
    FixedSize,
    Dynamic,
    Pooled,
    RingBuffer,
}

/// Cache optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheOptimizationConfig {
    /// Enable L1 cache optimization
    pub l1_optimization: bool,
    /// Enable L2 cache optimization
    pub l2_optimization: bool,
    /// Cache warming strategy
    pub cache_warming: CacheWarmingStrategy,
}

/// Custom metric configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomMetricConfig {
    /// Metric name
    pub name: String,
    /// Metric type
    pub metric_type: String,
    /// Collection endpoint
    pub endpoint: String,
    /// Collection interval
    pub interval: u32,
}

/// Log aggregation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogAggregationConfig {
    /// Enable log aggregation
    pub enabled: bool,
    /// Log level
    pub log_level: String,
    /// Log format
    pub format: String,
    /// Destination
    pub destination: String,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Enable health checks
    pub enabled: bool,
    /// Health check endpoint
    pub endpoint: String,
    /// Check interval (seconds)
    pub interval: u32,
    /// Timeout (seconds)
    pub timeout: u32,
    /// Failure threshold
    pub failure_threshold: u32,
}

/// Alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfig {
    /// Enable alerting
    pub enabled: bool,
    /// Alert rules
    pub rules: Vec<AlertRule>,
    /// Notification channels
    pub notification_channels: Vec<NotificationChannel>,
}

/// Image scanning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageScanningConfig {
    /// Enable scanning
    pub enabled: bool,
    /// Scan on push
    pub scan_on_push: bool,
    /// Vulnerability thresholds
    pub vulnerability_thresholds: VulnerabilityThresholds,
}

/// Runtime security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeSecurityConfig {
    /// Enable runtime monitoring
    pub enabled: bool,
    /// Security policies
    pub policies: Vec<SecurityPolicy>,
    /// Compliance checks
    pub compliance_checks: Vec<ComplianceCheck>,
}

/// Network policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPolicy {
    /// Policy name
    pub name: String,
    /// Ingress rules
    pub ingress: Vec<IngressRule>,
    /// Egress rules
    pub egress: Vec<EgressRule>,
}

/// Secret management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretManagementConfig {
    /// Secret store type
    pub store_type: String,
    /// Encryption settings
    pub encryption: EncryptionConfig,
    /// Rotation policy
    pub rotation_policy: RotationPolicy,
}

/// Port mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortMapping {
    /// Host port
    pub host_port: u16,
    /// Container port
    pub container_port: u16,
    /// Protocol
    pub protocol: String,
}

/// Load balancer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancerConfig {
    /// Load balancer type
    pub lb_type: String,
    /// Health check
    pub health_check: HealthCheckConfig,
    /// SSL configuration
    pub ssl: Option<SSLConfig>,
}

/// Service mesh configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMeshConfig {
    /// Mesh type (istio, linkerd, consul)
    pub mesh_type: String,
    /// Configuration
    pub config: HashMap<String, String>,
}

/// Persistent volume configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentVolumeConfig {
    /// Volume name
    pub name: String,
    /// Size
    pub size: String,
    /// Storage class
    pub storage_class: String,
    /// Mount path
    pub mount_path: String,
}

/// Temporary storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TempStorageConfig {
    /// Size limit
    pub size_limit: String,
    /// Clean up policy
    pub cleanup_policy: String,
}

/// Backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    /// Backup schedule
    pub schedule: String,
    /// Retention policy
    pub retention: String,
    /// Backup destination
    pub destination: String,
}

/// Volume mount
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeMount {
    /// Source path
    pub source: String,
    /// Target path
    pub target: String,
    /// Read-only flag
    pub read_only: bool,
}

/// Build configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildConfig {
    /// Source strategy
    pub strategy: String,
    /// Source repository
    pub source: String,
    /// Output image
    pub output_image: String,
}

/// Deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentConfig {
    /// Replicas
    pub replicas: u32,
    /// Update strategy
    pub strategy: String,
    /// Resource limits
    pub resources: ResourceLimits,
}

/// Cache warming strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheWarmingStrategy {
    None,
    Aggressive,
    Conservative,
    Predictive,
}

/// Alert rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Rule name
    pub name: String,
    /// Condition
    pub condition: String,
    /// Threshold
    pub threshold: f64,
    /// Duration
    pub duration: String,
}

/// Notification channel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationChannel {
    /// Channel type
    pub channel_type: String,
    /// Configuration
    pub config: HashMap<String, String>,
}

/// Vulnerability thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnerabilityThresholds {
    /// Critical threshold
    pub critical: u32,
    /// High threshold
    pub high: u32,
    /// Medium threshold
    pub medium: u32,
}

/// Security policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityPolicy {
    /// Policy name
    pub name: String,
    /// Policy type
    pub policy_type: String,
    /// Rules
    pub rules: Vec<SecurityRule>,
}

/// Compliance check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceCheck {
    /// Check name
    pub name: String,
    /// Standard
    pub standard: String,
    /// Requirements
    pub requirements: Vec<String>,
}

/// Ingress rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngressRule {
    /// Allowed sources
    pub from: Vec<String>,
    /// Allowed ports
    pub ports: Vec<u16>,
}

/// Egress rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EgressRule {
    /// Allowed destinations
    pub to: Vec<String>,
    /// Allowed ports
    pub ports: Vec<u16>,
}

/// Encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Encryption algorithm
    pub algorithm: String,
    /// Key management
    pub key_management: String,
}

/// Rotation policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotationPolicy {
    /// Rotation interval
    pub interval: String,
    /// Automatic rotation
    pub automatic: bool,
}

/// SSL configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSLConfig {
    /// Certificate path
    pub cert_path: String,
    /// Key path
    pub key_path: String,
    /// CA certificate path
    pub ca_cert_path: Option<String>,
}

/// Resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// CPU limit
    pub cpu: String,
    /// Memory limit
    pub memory: String,
}

/// Security rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityRule {
    /// Rule name
    pub name: String,
    /// Action
    pub action: String,
    /// Conditions
    pub conditions: Vec<String>,
}
