//! Networking Configuration Types
//!
//! This module contains networking-related Kubernetes configuration structures
//! including services, ingress, network policies, and related networking types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::security::LabelSelector;

/// Service configuration
///
/// Defines a Kubernetes Service for exposing applications.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    /// Service type
    pub service_type: ServiceType,
    /// Service ports
    pub ports: Vec<ServicePort>,
    /// Selector
    pub selector: HashMap<String, String>,
    /// External name (for ExternalName type)
    pub external_name: Option<String>,
    /// Load balancer source ranges
    pub load_balancer_source_ranges: Vec<String>,
    /// External traffic policy
    pub external_traffic_policy: Option<ExternalTrafficPolicy>,
}

/// Service types
///
/// Different types of Kubernetes services.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceType {
    /// ClusterIP service (internal only)
    ClusterIP,
    /// NodePort service (accessible via node ports)
    NodePort,
    /// LoadBalancer service (cloud load balancer)
    LoadBalancer,
    /// ExternalName service (DNS alias)
    ExternalName,
}

/// Service port
///
/// Port configuration for a service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServicePort {
    /// Port name
    pub name: String,
    /// Port number
    pub port: u16,
    /// Target port
    pub target_port: u16,
    /// Node port (for NodePort/LoadBalancer)
    pub node_port: Option<u16>,
    /// Protocol
    pub protocol: Protocol,
}

/// Network protocols
///
/// Supported network protocols.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Protocol {
    /// Transmission Control Protocol
    TCP,
    /// User Datagram Protocol
    UDP,
    /// Stream Control Transmission Protocol
    SCTP,
}

/// External traffic policies
///
/// Policies for handling external traffic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExternalTrafficPolicy {
    /// Local policy (preserve source IP)
    Local,
    /// Cluster policy (distribute across all nodes)
    Cluster,
}

/// Ingress configuration
///
/// Defines HTTP/HTTPS load balancing rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngressConfig {
    /// Ingress class
    pub ingress_class: Option<String>,
    /// TLS configuration
    pub tls: Vec<IngressTLS>,
    /// Rules
    pub rules: Vec<IngressRule>,
}

/// Ingress TLS
///
/// TLS configuration for ingress.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngressTLS {
    /// Hosts covered by this TLS certificate
    pub hosts: Vec<String>,
    /// Secret name containing the TLS certificate
    pub secret_name: String,
}

/// Ingress rule
///
/// HTTP routing rule for ingress.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngressRule {
    /// Host name (optional, * for default)
    pub host: String,
    /// HTTP paths
    pub http: IngressHTTP,
}

/// Ingress HTTP
///
/// HTTP-specific ingress configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngressHTTP {
    /// Paths
    pub paths: Vec<IngressPath>,
}

/// Ingress path
///
/// A single path rule in an ingress.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngressPath {
    /// Path pattern
    pub path: String,
    /// Path type
    pub path_type: PathType,
    /// Backend service
    pub backend: IngressBackend,
}

/// Path types
///
/// Different types of path matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PathType {
    /// Exact path match
    Exact,
    /// Prefix path match
    Prefix,
    /// Implementation-specific matching
    ImplementationSpecific,
}

/// Ingress backend
///
/// Backend service for an ingress path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngressBackend {
    /// Service
    pub service: IngressServiceBackend,
}

/// Ingress service backend
///
/// Service backend configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngressServiceBackend {
    /// Service name
    pub name: String,
    /// Service port
    pub port: ServiceBackendPort,
}

/// Service backend port
///
/// Port specification for service backends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceBackendPort {
    /// Port number
    Number(u16),
    /// Port name
    Name(String),
}

/// Network Policy configuration
///
/// Defines network access rules for pods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPolicyConfig {
    /// Pod selector
    pub pod_selector: LabelSelector,
    /// Policy types
    pub policy_types: Vec<PolicyType>,
    /// Ingress rules
    pub ingress: Vec<NetworkPolicyIngressRule>,
    /// Egress rules
    pub egress: Vec<NetworkPolicyEgressRule>,
}

/// Policy types
///
/// Types of network policies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyType {
    /// Ingress policy
    Ingress,
    /// Egress policy
    Egress,
}

/// Network policy ingress rule
///
/// Rule for incoming network traffic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPolicyIngressRule {
    /// Allowed sources
    pub from: Vec<NetworkPolicyPeer>,
    /// Allowed ports
    pub ports: Vec<NetworkPolicyPort>,
}

/// Network policy egress rule
///
/// Rule for outgoing network traffic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPolicyEgressRule {
    /// Allowed destinations
    pub to: Vec<NetworkPolicyPeer>,
    /// Allowed ports
    pub ports: Vec<NetworkPolicyPort>,
}

/// Network policy peer
///
/// Source or destination for network policy rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkPolicyPeer {
    /// Pod selector
    PodSelector(LabelSelector),
    /// Namespace selector
    NamespaceSelector(LabelSelector),
    /// IP block
    IPBlock {
        /// CIDR block
        cidr: String,
        /// Excluded CIDRs
        except: Vec<String>,
    },
}

/// Network policy port
///
/// Port specification for network policies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPolicyPort {
    /// Protocol
    pub protocol: Option<Protocol>,
    /// Port
    pub port: Option<String>,
    /// End port (for port ranges)
    pub end_port: Option<u16>,
}

/// Default implementations for networking types
impl Default for ServiceConfig {
    fn default() -> Self {
        let mut selector = HashMap::new();
        selector.insert("app".to_string(), "trustformers".to_string());

        Self {
            service_type: ServiceType::ClusterIP,
            ports: vec![ServicePort {
                name: "http".to_string(),
                port: 80,
                target_port: 8080,
                node_port: None,
                protocol: Protocol::TCP,
            }],
            selector,
            external_name: None,
            load_balancer_source_ranges: Vec::new(),
            external_traffic_policy: None,
        }
    }
}

impl Default for IngressConfig {
    fn default() -> Self {
        Self {
            ingress_class: Some("nginx".to_string()),
            tls: Vec::new(),
            rules: vec![IngressRule {
                host: "trustformers.example.com".to_string(),
                http: IngressHTTP {
                    paths: vec![IngressPath {
                        path: "/".to_string(),
                        path_type: PathType::Prefix,
                        backend: IngressBackend {
                            service: IngressServiceBackend {
                                name: "trustformers-service".to_string(),
                                port: ServiceBackendPort::Number(80),
                            },
                        },
                    }],
                },
            }],
        }
    }
}

impl Default for NetworkPolicyConfig {
    fn default() -> Self {
        Self {
            pod_selector: LabelSelector::default(),
            policy_types: vec![PolicyType::Ingress],
            ingress: vec![NetworkPolicyIngressRule {
                from: Vec::new(),
                ports: vec![NetworkPolicyPort {
                    protocol: Some(Protocol::TCP),
                    port: Some("8080".to_string()),
                    end_port: None,
                }],
            }],
            egress: Vec::new(),
        }
    }
}
