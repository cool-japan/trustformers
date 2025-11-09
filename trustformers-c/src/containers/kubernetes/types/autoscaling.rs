//! Autoscaling Configuration Types
//!
//! This module contains autoscaling-related Kubernetes configuration structures
//! including Horizontal Pod Autoscaler (HPA), Vertical Pod Autoscaler (VPA),
//! and Pod Disruption Budget (PDB) configurations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::security::LabelSelector;

/// Horizontal Pod Autoscaler configuration
///
/// Automatically scales the number of pods based on observed metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HPAConfig {
    /// Minimum number of replicas
    pub min_replicas: u32,
    /// Maximum number of replicas
    pub max_replicas: u32,
    /// Target CPU utilization percentage (deprecated, use metrics instead)
    pub target_cpu_utilization_percentage: Option<u32>,
    /// Metrics specifications
    pub metrics: Vec<MetricSpec>,
    /// Scaling behavior configuration
    pub behavior: Option<HPABehavior>,
}

/// HPA metric specification
///
/// Defines a metric for autoscaling decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSpec {
    /// Type of metric
    pub metric_type: MetricType,
    /// Detailed metric specification
    pub spec: MetricSpecDetails,
}

/// Metric types
///
/// Different types of metrics that can be used for autoscaling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    /// Resource metrics (CPU, memory)
    Resource,
    /// Pod-level custom metrics
    Pods,
    /// Object-level custom metrics
    Object,
    /// External metrics from outside the cluster
    External,
}

/// Metric specification details
///
/// Detailed specification for different metric types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricSpecDetails {
    /// Resource metric specification
    Resource {
        /// Resource name (cpu, memory)
        name: String,
        /// Target specification
        target: ResourceMetricTarget,
    },
    /// Pods metric specification
    Pods {
        /// Metric identifier
        metric: MetricIdentifier,
        /// Target specification
        target: MetricTarget,
    },
    /// Object metric specification
    Object {
        /// Metric identifier
        metric: MetricIdentifier,
        /// Object being described
        described_object: CrossVersionObjectReference,
        /// Target specification
        target: MetricTarget,
    },
    /// External metric specification
    External {
        /// Metric identifier
        metric: MetricIdentifier,
        /// Target specification
        target: MetricTarget,
    },
}

/// Resource metric target
///
/// Target specification for resource metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetricTarget {
    /// Type of target
    pub target_type: MetricTargetType,
    /// Average utilization percentage
    pub average_utilization: Option<u32>,
    /// Average value
    pub average_value: Option<String>,
    /// Absolute value
    pub value: Option<String>,
}

/// Metric target
///
/// Target specification for custom metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricTarget {
    /// Type of target
    pub target_type: MetricTargetType,
    /// Average value
    pub average_value: Option<String>,
    /// Absolute value
    pub value: Option<String>,
}

/// Metric target types
///
/// Different ways to specify metric targets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricTargetType {
    /// Utilization percentage
    Utilization,
    /// Absolute value
    Value,
    /// Average value across pods
    AverageValue,
}

/// Metric identifier
///
/// Identifies a specific metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricIdentifier {
    /// Metric name
    pub name: String,
    /// Label selector for the metric
    pub selector: Option<LabelSelector>,
}

/// Cross-version object reference
///
/// Reference to a Kubernetes object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossVersionObjectReference {
    /// API version
    pub api_version: String,
    /// Object kind
    pub kind: String,
    /// Object name
    pub name: String,
}

/// HPA behavior configuration
///
/// Defines scaling behavior policies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HPABehavior {
    /// Scale down behavior
    pub scale_down: Option<HPAScalingRules>,
    /// Scale up behavior
    pub scale_up: Option<HPAScalingRules>,
}

/// HPA scaling rules
///
/// Rules governing scaling behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HPAScalingRules {
    /// Stabilization window in seconds
    pub stabilization_window_seconds: Option<u32>,
    /// Policy selection strategy
    pub select_policy: Option<ScalingPolicySelect>,
    /// Scaling policies
    pub policies: Vec<HPAScalingPolicy>,
}

/// Scaling policy selection
///
/// Strategy for selecting among multiple policies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingPolicySelect {
    /// Use the policy that allows the maximum change
    Max,
    /// Use the policy that allows the minimum change
    Min,
    /// Disable scaling
    Disabled,
}

/// HPA scaling policy
///
/// Individual scaling policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HPAScalingPolicy {
    /// Type of policy
    pub policy_type: HPAScalingPolicyType,
    /// Policy value
    pub value: u32,
    /// Period in seconds
    pub period_seconds: u32,
}

/// HPA scaling policy types
///
/// Types of scaling policies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HPAScalingPolicyType {
    /// Scale by number of pods
    Pods,
    /// Scale by percentage
    Percent,
}

/// Vertical Pod Autoscaler configuration
///
/// Automatically adjusts pod resource requests and limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VPAConfig {
    /// Update policy
    pub update_policy: VPAUpdatePolicy,
    /// Resource policy
    pub resource_policy: VPAResourcePolicy,
}

/// VPA update policy
///
/// Defines how VPA updates are applied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VPAUpdatePolicy {
    /// Update mode
    pub update_mode: VPAUpdateMode,
}

/// VPA update modes
///
/// Different modes for applying VPA updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VPAUpdateMode {
    /// VPA is disabled
    Off,
    /// VPA only sets recommendations on pod creation
    Initial,
    /// VPA recreates pods to apply recommendations
    Recreation,
    /// VPA automatically applies recommendations
    Auto,
}

/// VPA resource policy
///
/// Defines resource policies for containers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VPAResourcePolicy {
    /// Container-specific policies
    pub container_policies: Vec<VPAContainerResourcePolicy>,
}

/// VPA container resource policy
///
/// Resource policy for a specific container.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VPAContainerResourcePolicy {
    /// Container name
    pub container_name: String,
    /// Minimum allowed resources
    pub min_allowed: HashMap<String, String>,
    /// Maximum allowed resources
    pub max_allowed: HashMap<String, String>,
    /// Resources under VPA control
    pub controlled_resources: Vec<String>,
    /// Values under VPA control
    pub controlled_values: Vec<String>,
}

/// Pod Disruption Budget configuration
///
/// Limits the number of pods that can be disrupted simultaneously.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PDBConfig {
    /// Minimum number/percentage of pods that must remain available
    pub min_available: Option<IntOrString>,
    /// Maximum number/percentage of pods that can be unavailable
    pub max_unavailable: Option<IntOrString>,
    /// Selector for pods covered by this PDB
    pub selector: LabelSelector,
}

/// Integer or string value
///
/// Used for fields that can accept either an absolute number or percentage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntOrString {
    /// Integer value
    Int(u32),
    /// String value (often a percentage)
    String(String),
}

/// Default implementations for autoscaling types
impl Default for HPAConfig {
    fn default() -> Self {
        Self {
            min_replicas: 1,
            max_replicas: 10,
            target_cpu_utilization_percentage: Some(80),
            metrics: vec![MetricSpec {
                metric_type: MetricType::Resource,
                spec: MetricSpecDetails::Resource {
                    name: "cpu".to_string(),
                    target: ResourceMetricTarget {
                        target_type: MetricTargetType::Utilization,
                        average_utilization: Some(80),
                        average_value: None,
                        value: None,
                    },
                },
            }],
            behavior: None,
        }
    }
}

impl Default for VPAConfig {
    fn default() -> Self {
        Self {
            update_policy: VPAUpdatePolicy {
                update_mode: VPAUpdateMode::Auto,
            },
            resource_policy: VPAResourcePolicy {
                container_policies: Vec::new(),
            },
        }
    }
}

impl Default for PDBConfig {
    fn default() -> Self {
        Self {
            min_available: Some(IntOrString::String("50%".to_string())),
            max_unavailable: None,
            selector: LabelSelector::default(),
        }
    }
}

impl ToString for IntOrString {
    fn to_string(&self) -> String {
        match self {
            IntOrString::Int(i) => i.to_string(),
            IntOrString::String(s) => s.clone(),
        }
    }
}
