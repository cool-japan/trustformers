//! Security Configuration Types
//!
//! This module contains security-related Kubernetes configuration structures
//! including security contexts, tolerations, affinity rules, and label selectors.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Security context for pods
///
/// Defines security settings for the entire pod.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityContext {
    /// Run as non-root
    pub run_as_non_root: Option<bool>,
    /// Run as user
    pub run_as_user: Option<u64>,
    /// Run as group
    pub run_as_group: Option<u64>,
    /// FS group
    pub fs_group: Option<u64>,
    /// SELinux options
    pub se_linux_options: Option<SELinuxOptions>,
    /// Windows options
    pub windows_options: Option<WindowsOptions>,
    /// Supplemental groups
    pub supplemental_groups: Vec<u64>,
}

/// Container security context
///
/// Defines security settings for individual containers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerSecurityContext {
    /// Allow privilege escalation
    pub allow_privilege_escalation: Option<bool>,
    /// Capabilities
    pub capabilities: Option<Capabilities>,
    /// Privileged
    pub privileged: Option<bool>,
    /// Proc mount
    pub proc_mount: Option<String>,
    /// Read-only root filesystem
    pub read_only_root_filesystem: Option<bool>,
    /// Run as group
    pub run_as_group: Option<u64>,
    /// Run as non-root
    pub run_as_non_root: Option<bool>,
    /// Run as user
    pub run_as_user: Option<u64>,
    /// SELinux options
    pub se_linux_options: Option<SELinuxOptions>,
}

/// Capabilities configuration
///
/// Specifies Linux capabilities to add or drop for containers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capabilities {
    /// Capabilities to add
    pub add: Vec<String>,
    /// Capabilities to drop
    pub drop: Vec<String>,
}

/// SELinux options
///
/// SELinux security context configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SELinuxOptions {
    /// Level
    pub level: Option<String>,
    /// Role
    pub role: Option<String>,
    /// Type
    pub type_: Option<String>,
    /// User
    pub user: Option<String>,
}

/// Windows options
///
/// Windows-specific security configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowsOptions {
    /// GMSA credential name
    pub gmsa_credential_name: Option<String>,
    /// GMSA credential spec
    pub gmsa_credential_spec: Option<String>,
    /// Host process
    pub host_process: Option<bool>,
    /// Run as username
    pub run_as_username: Option<String>,
}

/// Toleration
///
/// Allows pods to be scheduled on nodes with matching taints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Toleration {
    /// Effect
    pub effect: Option<TaintEffect>,
    /// Key
    pub key: Option<String>,
    /// Operator
    pub operator: Option<TolerationOperator>,
    /// Toleration seconds
    pub toleration_seconds: Option<u64>,
    /// Value
    pub value: Option<String>,
}

/// Taint effects
///
/// Effects that taints can have on pod scheduling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaintEffect {
    /// Pod will not be scheduled on the node
    NoSchedule,
    /// Pod will preferably not be scheduled on the node
    PreferNoSchedule,
    /// Pod will be evicted from the node
    NoExecute,
}

/// Toleration operators
///
/// Operators for matching taints in tolerations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TolerationOperator {
    /// Equal comparison
    Equal,
    /// Existence check
    Exists,
}

/// Affinity configuration
///
/// Pod and node affinity rules for scheduling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffinityConfig {
    /// Node affinity
    pub node_affinity: Option<NodeAffinity>,
    /// Pod affinity
    pub pod_affinity: Option<PodAffinity>,
    /// Pod anti-affinity
    pub pod_anti_affinity: Option<PodAntiAffinity>,
}

/// Node affinity
///
/// Rules for scheduling pods on nodes based on node labels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeAffinity {
    /// Required during scheduling
    pub required_during_scheduling_ignored_during_execution: Option<NodeSelector>,
    /// Preferred during scheduling
    pub preferred_during_scheduling_ignored_during_execution: Vec<PreferredSchedulingTerm>,
}

/// Node selector
///
/// Selects nodes based on labels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSelector {
    /// Node selector terms
    pub node_selector_terms: Vec<NodeSelectorTerm>,
}

/// Node selector term
///
/// A set of node selector requirements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSelectorTerm {
    /// Match expressions
    pub match_expressions: Vec<NodeSelectorRequirement>,
    /// Match fields
    pub match_fields: Vec<NodeSelectorRequirement>,
}

/// Node selector requirement
///
/// A single node selector requirement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSelectorRequirement {
    /// Key
    pub key: String,
    /// Operator
    pub operator: NodeSelectorOperator,
    /// Values
    pub values: Vec<String>,
}

/// Node selector operators
///
/// Operators for node selector requirements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeSelectorOperator {
    /// In operator
    In,
    /// Not in operator
    NotIn,
    /// Exists operator
    Exists,
    /// Does not exist operator
    DoesNotExist,
    /// Greater than operator
    Gt,
    /// Less than operator
    Lt,
}

/// Preferred scheduling term
///
/// A preferred node selector term with weight.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferredSchedulingTerm {
    /// Preference
    pub preference: NodeSelectorTerm,
    /// Weight (1-100)
    pub weight: u32,
}

/// Pod affinity
///
/// Rules for scheduling pods based on other pods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodAffinity {
    /// Required during scheduling
    pub required_during_scheduling_ignored_during_execution: Vec<PodAffinityTerm>,
    /// Preferred during scheduling
    pub preferred_during_scheduling_ignored_during_execution: Vec<WeightedPodAffinityTerm>,
}

/// Pod anti-affinity
///
/// Rules for avoiding co-location with other pods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodAntiAffinity {
    /// Required during scheduling
    pub required_during_scheduling_ignored_during_execution: Vec<PodAffinityTerm>,
    /// Preferred during scheduling
    pub preferred_during_scheduling_ignored_during_execution: Vec<WeightedPodAffinityTerm>,
}

/// Pod affinity term
///
/// A single pod affinity rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodAffinityTerm {
    /// Label selector
    pub label_selector: Option<LabelSelector>,
    /// Namespace selector
    pub namespace_selector: Option<LabelSelector>,
    /// Namespaces
    pub namespaces: Vec<String>,
    /// Topology key
    pub topology_key: String,
}

/// Weighted pod affinity term
///
/// A pod affinity term with weight.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedPodAffinityTerm {
    /// Pod affinity term
    pub pod_affinity_term: PodAffinityTerm,
    /// Weight (1-100)
    pub weight: u32,
}

/// Label selector
///
/// Selects objects based on labels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelSelector {
    /// Match labels (equality-based)
    pub match_labels: HashMap<String, String>,
    /// Match expressions (set-based)
    pub match_expressions: Vec<LabelSelectorRequirement>,
}

/// Label selector requirement
///
/// A single label selector requirement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelSelectorRequirement {
    /// Key
    pub key: String,
    /// Operator
    pub operator: LabelSelectorOperator,
    /// Values
    pub values: Vec<String>,
}

/// Label selector operators
///
/// Operators for label selector requirements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LabelSelectorOperator {
    /// In operator
    In,
    /// Not in operator
    NotIn,
    /// Exists operator
    Exists,
    /// Does not exist operator
    DoesNotExist,
}

/// Default implementations for security types
impl Default for SecurityContext {
    fn default() -> Self {
        Self {
            run_as_non_root: Some(true),
            run_as_user: Some(65534),  // nobody user
            run_as_group: Some(65534), // nobody group
            fs_group: Some(65534),
            se_linux_options: None,
            windows_options: None,
            supplemental_groups: Vec::new(),
        }
    }
}

impl Default for ContainerSecurityContext {
    fn default() -> Self {
        Self {
            allow_privilege_escalation: Some(false),
            capabilities: Some(Capabilities {
                add: Vec::new(),
                drop: vec!["ALL".to_string()],
            }),
            privileged: Some(false),
            proc_mount: None,
            read_only_root_filesystem: Some(true),
            run_as_group: Some(65534),
            run_as_non_root: Some(true),
            run_as_user: Some(65534),
            se_linux_options: None,
        }
    }
}

impl Default for LabelSelector {
    fn default() -> Self {
        Self {
            match_labels: HashMap::new(),
            match_expressions: Vec::new(),
        }
    }
}
