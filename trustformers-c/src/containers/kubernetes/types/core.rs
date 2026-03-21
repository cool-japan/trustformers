//! Core Kubernetes Configuration Types
//!
//! This module contains the fundamental Kubernetes configuration structures
//! including deployment, pod, container specifications, and related core types.

use crate::error::{TrustformersError, TrustformersResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Re-export other type modules
pub use super::{autoscaling::*, monitoring::*, networking::*, security::*, storage::*};

/// Kubernetes deployment configuration
///
/// This is the main configuration structure that orchestrates all aspects
/// of a Kubernetes deployment including services, storage, autoscaling, and monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesConfig {
    /// Deployment metadata
    pub metadata: KubernetesMetadata,
    /// Deployment specification
    pub deployment: DeploymentConfig,
    /// Service configuration
    pub service: ServiceConfig,
    /// Ingress configuration
    pub ingress: Option<IngressConfig>,
    /// ConfigMap configuration
    pub config_maps: Vec<ConfigMapConfig>,
    /// Secret configuration
    pub secrets: Vec<SecretConfig>,
    /// Persistent Volume configuration
    pub persistent_volumes: Vec<PVConfig>,
    /// Horizontal Pod Autoscaler configuration
    pub hpa: Option<HPAConfig>,
    /// Vertical Pod Autoscaler configuration
    pub vpa: Option<VPAConfig>,
    /// Pod Disruption Budget
    pub pdb: Option<PDBConfig>,
    /// Network Policy
    pub network_policy: Option<NetworkPolicyConfig>,
    /// Service Monitor (Prometheus)
    pub service_monitor: Option<ServiceMonitorConfig>,
}

/// Kubernetes metadata
///
/// Standard metadata applied to all Kubernetes resources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesMetadata {
    /// Application name
    pub name: String,
    /// Namespace
    pub namespace: String,
    /// Labels
    pub labels: HashMap<String, String>,
    /// Annotations
    pub annotations: HashMap<String, String>,
}

/// Deployment configuration
///
/// Configures the Kubernetes Deployment resource including replica count,
/// update strategy, and pod template specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentConfig {
    /// Number of replicas
    pub replicas: u32,
    /// Deployment strategy
    pub strategy: DeploymentStrategy,
    /// Pod template specification
    pub pod_spec: PodSpec,
    /// Revision history limit
    pub revision_history_limit: Option<u32>,
}

/// Deployment strategies
///
/// Defines how updates to the deployment should be applied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentStrategy {
    /// Rolling update strategy
    ///
    /// Gradually replaces old pods with new ones to maintain availability.
    RollingUpdate {
        /// Maximum unavailable pods (absolute number or percentage)
        max_unavailable: String,
        /// Maximum surge (absolute number or percentage)
        max_surge: String,
    },
    /// Recreate strategy
    ///
    /// Terminates all existing pods before creating new ones.
    Recreate,
}

/// Pod specification
///
/// Template specification for pods created by the deployment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodSpec {
    /// Container specifications
    pub containers: Vec<ContainerSpec>,
    /// Init containers
    pub init_containers: Vec<ContainerSpec>,
    /// Service account
    pub service_account: Option<String>,
    /// Security context
    pub security_context: SecurityContext,
    /// DNS policy
    pub dns_policy: DNSPolicy,
    /// Restart policy
    pub restart_policy: RestartPolicy,
    /// Node selector
    pub node_selector: HashMap<String, String>,
    /// Tolerations
    pub tolerations: Vec<Toleration>,
    /// Affinity
    pub affinity: Option<AffinityConfig>,
    /// Priority class
    pub priority_class: Option<String>,
    /// Termination grace period
    pub termination_grace_period: Option<u64>,
}

/// Container specification
///
/// Comprehensive container configuration including image, resources,
/// health checks, security settings, and environment configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerSpec {
    /// Container name
    pub name: String,
    /// Container image
    pub image: String,
    /// Image pull policy
    pub image_pull_policy: ImagePullPolicy,
    /// Container ports
    pub ports: Vec<ContainerPort>,
    /// Environment variables
    pub env: Vec<EnvVar>,
    /// Volume mounts
    pub volume_mounts: Vec<VolumeMount>,
    /// Resource requirements
    pub resources: ResourceRequirements,
    /// Liveness probe
    pub liveness_probe: Option<ProbeConfig>,
    /// Readiness probe
    pub readiness_probe: Option<ProbeConfig>,
    /// Startup probe
    pub startup_probe: Option<ProbeConfig>,
    /// Security context
    pub security_context: Option<ContainerSecurityContext>,
    /// Command and args
    pub command: Vec<String>,
    /// Arguments
    pub args: Vec<String>,
}

/// Image pull policies
///
/// Defines when container images should be pulled from the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImagePullPolicy {
    /// Always pull the image, even if it exists locally
    Always,
    /// Only pull if the image is not present locally
    IfNotPresent,
    /// Never pull the image
    Never,
}

/// Container port
///
/// Specifies a port exposed by a container.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerPort {
    /// Port name
    pub name: String,
    /// Port number
    pub container_port: u16,
    /// Protocol
    pub protocol: Protocol,
}

/// Network protocols
///
/// Supported network protocols for container ports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Protocol {
    /// Transmission Control Protocol
    TCP,
    /// User Datagram Protocol
    UDP,
    /// Stream Control Transmission Protocol
    SCTP,
}

/// Environment variable
///
/// Specifies an environment variable for a container.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvVar {
    /// Variable name
    pub name: String,
    /// Variable value (if specified directly)
    pub value: Option<String>,
    /// Value from source (ConfigMap, Secret, etc.)
    pub value_from: Option<EnvVarSource>,
}

/// Environment variable sources
///
/// Different sources from which environment variable values can be derived.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnvVarSource {
    /// ConfigMap key reference
    ConfigMapKeyRef {
        /// ConfigMap name
        name: String,
        /// Key in ConfigMap
        key: String,
        /// Optional flag
        optional: bool,
    },
    /// Secret key reference
    SecretKeyRef {
        /// Secret name
        name: String,
        /// Key in Secret
        key: String,
        /// Optional flag
        optional: bool,
    },
    /// Field reference
    FieldRef {
        /// Field path
        field_path: String,
    },
    /// Resource field reference
    ResourceFieldRef {
        /// Resource name
        resource: String,
        /// Divisor
        divisor: Option<String>,
    },
}

/// Volume mount
///
/// Specifies a volume mount for a container.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeMount {
    /// Mount name
    pub name: String,
    /// Mount path
    pub mount_path: String,
    /// Sub path
    pub sub_path: Option<String>,
    /// Read-only flag
    pub read_only: bool,
}

/// Resource requirements
///
/// Specifies compute resource requirements and limits for a container.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Resource limits (maximum allowed)
    pub limits: HashMap<String, String>,
    /// Resource requests (minimum required)
    pub requests: HashMap<String, String>,
}

/// Probe configuration
///
/// Health check configuration for containers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeConfig {
    /// Probe action
    pub probe_action: ProbeAction,
    /// Initial delay before starting probes
    pub initial_delay_seconds: u32,
    /// Probe period
    pub period_seconds: u32,
    /// Probe timeout
    pub timeout_seconds: u32,
    /// Success threshold
    pub success_threshold: u32,
    /// Failure threshold
    pub failure_threshold: u32,
}

/// Probe actions
///
/// Different types of health check probes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProbeAction {
    /// HTTP GET probe
    HttpGet {
        /// Request path
        path: String,
        /// Port number
        port: u16,
        /// HTTP headers
        http_headers: Vec<HttpHeader>,
        /// URL scheme
        scheme: HttpScheme,
    },
    /// TCP socket probe
    TcpSocket {
        /// Port number
        port: u16,
    },
    /// Command execution probe
    Exec {
        /// Command to execute
        command: Vec<String>,
    },
}

/// HTTP header
///
/// HTTP header for HTTP GET probes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpHeader {
    /// Header name
    pub name: String,
    /// Header value
    pub value: String,
}

/// HTTP schemes
///
/// URL schemes for HTTP probes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HttpScheme {
    /// HTTP protocol
    HTTP,
    /// HTTPS protocol
    HTTPS,
}

/// DNS policies
///
/// Determines how DNS resolution is handled in pods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DNSPolicy {
    /// Use cluster DNS with fallback to upstream DNS for host network
    ClusterFirst,
    /// Use cluster DNS with host network
    ClusterFirstWithHostNet,
    /// Use host's DNS resolution
    Default,
    /// No DNS configuration
    None,
}

/// Restart policies
///
/// Determines when containers should be restarted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RestartPolicy {
    /// Always restart containers
    Always,
    /// Only restart on failure
    OnFailure,
    /// Never restart containers
    Never,
}

/// Default implementations for core types
impl Default for KubernetesConfig {
    fn default() -> Self {
        Self {
            metadata: KubernetesMetadata::default(),
            deployment: DeploymentConfig::default(),
            service: ServiceConfig::default(),
            ingress: None,
            config_maps: Vec::new(),
            secrets: Vec::new(),
            persistent_volumes: Vec::new(),
            hpa: None,
            vpa: None,
            pdb: None,
            network_policy: None,
            service_monitor: None,
        }
    }
}

impl Default for KubernetesMetadata {
    fn default() -> Self {
        let mut labels = HashMap::new();
        labels.insert("app".to_string(), "trustformers".to_string());
        labels.insert("component".to_string(), "inference".to_string());

        Self {
            name: "trustformers-deployment".to_string(),
            namespace: "default".to_string(),
            labels,
            annotations: HashMap::new(),
        }
    }
}

impl Default for DeploymentConfig {
    fn default() -> Self {
        Self {
            replicas: 1,
            strategy: DeploymentStrategy::RollingUpdate {
                max_unavailable: "25%".to_string(),
                max_surge: "25%".to_string(),
            },
            pod_spec: PodSpec::default(),
            revision_history_limit: Some(10),
        }
    }
}

impl Default for PodSpec {
    fn default() -> Self {
        Self {
            containers: vec![ContainerSpec::default()],
            init_containers: Vec::new(),
            service_account: None,
            security_context: SecurityContext::default(),
            dns_policy: DNSPolicy::ClusterFirst,
            restart_policy: RestartPolicy::Always,
            node_selector: HashMap::new(),
            tolerations: Vec::new(),
            affinity: None,
            priority_class: None,
            termination_grace_period: Some(30),
        }
    }
}

impl Default for ContainerSpec {
    fn default() -> Self {
        let mut resources = HashMap::new();
        resources.insert("cpu".to_string(), "500m".to_string());
        resources.insert("memory".to_string(), "1Gi".to_string());

        let mut limits = HashMap::new();
        limits.insert("cpu".to_string(), "1000m".to_string());
        limits.insert("memory".to_string(), "2Gi".to_string());

        Self {
            name: "trustformers".to_string(),
            image: "trustformers:latest".to_string(),
            image_pull_policy: ImagePullPolicy::IfNotPresent,
            ports: vec![ContainerPort {
                name: "http".to_string(),
                container_port: 8080,
                protocol: Protocol::TCP,
            }],
            env: Vec::new(),
            volume_mounts: Vec::new(),
            resources: ResourceRequirements {
                requests: resources,
                limits,
            },
            liveness_probe: None,
            readiness_probe: None,
            startup_probe: None,
            security_context: None,
            command: Vec::new(),
            args: Vec::new(),
        }
    }
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        let mut requests = HashMap::new();
        requests.insert("cpu".to_string(), "100m".to_string());
        requests.insert("memory".to_string(), "128Mi".to_string());

        let mut limits = HashMap::new();
        limits.insert("cpu".to_string(), "500m".to_string());
        limits.insert("memory".to_string(), "512Mi".to_string());

        Self { requests, limits }
    }
}
