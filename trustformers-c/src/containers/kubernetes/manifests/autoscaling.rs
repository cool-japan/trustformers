//! Kubernetes Autoscaling Manifest Generation
//!
//! This module provides functionality for generating Kubernetes HorizontalPodAutoscaler
//! manifests for automatic scaling of TrustformeRS deployments.

use super::ManifestGenerator;
use crate::error::TrustformersResult;
use std::collections::HashMap;

/// Kubernetes HorizontalPodAutoscaler configuration
#[derive(Debug, Clone)]
pub struct HorizontalPodAutoscalerManifest {
    pub name: String,
    pub namespace: String,
    pub target_deployment: String,
    pub min_replicas: i32,
    pub max_replicas: i32,
    pub target_cpu_percentage: i32,
    pub target_memory_percentage: Option<i32>,
    pub labels: HashMap<String, String>,
}

impl Default for HorizontalPodAutoscalerManifest {
    fn default() -> Self {
        Self {
            name: "trustformers-hpa".to_string(),
            namespace: "default".to_string(),
            target_deployment: "trustformers-deployment".to_string(),
            min_replicas: 1,
            max_replicas: 10,
            target_cpu_percentage: 70,
            target_memory_percentage: Some(80),
            labels: HashMap::new(),
        }
    }
}

impl ManifestGenerator for HorizontalPodAutoscalerManifest {
    fn generate_yaml(&self) -> TrustformersResult<String> {
        let mut yaml = format!(
            r#"apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {}
  namespace: {}"#,
            self.name, self.namespace
        );

        // Add labels if present
        if !self.labels.is_empty() {
            yaml.push_str("\n  labels:");
            for (key, value) in &self.labels {
                yaml.push_str(&format!("\n    {}: {}", key, value));
            }
        }

        // Add spec section
        yaml.push_str(&format!(
            r#"
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {}
  minReplicas: {}
  maxReplicas: {}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: {}"#,
            self.target_deployment,
            self.min_replicas,
            self.max_replicas,
            self.target_cpu_percentage
        ));

        // Add memory metric if specified
        if let Some(memory_percentage) = self.target_memory_percentage {
            yaml.push_str(&format!(
                r#"
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: {}"#,
                memory_percentage
            ));
        }

        yaml.push('\n');
        Ok(yaml)
    }
}
