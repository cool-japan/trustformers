//! Kubernetes Service Manifest Generation
//!
//! This module provides functionality for generating Kubernetes Service manifests
//! for TrustformeRS model serving.

use super::formatting::{format_annotations, format_labels, format_selector};
use super::ManifestGenerator;
use crate::error::TrustformersResult;
use std::collections::HashMap;

/// Kubernetes Service configuration
#[derive(Debug, Clone)]
pub struct ServiceManifest {
    pub name: String,
    pub namespace: String,
    pub port: i32,
    pub target_port: i32,
    pub labels: HashMap<String, String>,
    pub selector: HashMap<String, String>,
}

impl Default for ServiceManifest {
    fn default() -> Self {
        let mut selector = HashMap::new();
        selector.insert("app".to_string(), "trustformers".to_string());

        Self {
            name: "trustformers-service".to_string(),
            namespace: "default".to_string(),
            port: 80,
            target_port: 8080,
            labels: HashMap::new(),
            selector,
        }
    }
}

impl ManifestGenerator for ServiceManifest {
    fn generate_yaml(&self) -> TrustformersResult<String> {
        // Build merged label map — always include a name-based fallback.
        let mut merged_labels = self.labels.clone();
        merged_labels.entry("app".to_string()).or_insert_with(|| self.name.clone());

        // Build selector, falling back to the name-based label.
        let mut merged_selector = self.selector.clone();
        if merged_selector.is_empty() {
            merged_selector.insert("app".to_string(), self.name.clone());
        }

        let labels_yaml = format_labels(&merged_labels);
        let selector_yaml = format_selector(&merged_selector);

        let annotations_section = if self.labels.is_empty() {
            String::new()
        } else {
            format!("  annotations:\n{}\n", format_annotations(&self.labels))
        };

        let yaml = format!(
            r#"
apiVersion: v1
kind: Service
metadata:
  name: {}
  namespace: {}
  labels:
{}
{}spec:
  selector:
{}
  ports:
  - protocol: TCP
    port: {}
    targetPort: {}
  type: ClusterIP
"#,
            self.name,
            self.namespace,
            labels_yaml,
            annotations_section,
            selector_yaml,
            self.port,
            self.target_port,
        );

        Ok(yaml)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_service_manifest() {
        let manifest = ServiceManifest::default();
        let yaml = manifest.generate_yaml().expect("generate_yaml should succeed");
        assert!(yaml.contains("kind: Service"), "missing kind");
        assert!(yaml.contains("name: trustformers-service"), "missing name");
        assert!(yaml.contains("namespace: default"), "missing namespace");
        assert!(yaml.contains("port: 80"), "missing port");
        assert!(yaml.contains("targetPort: 8080"), "missing targetPort");
        assert!(yaml.contains("type: ClusterIP"), "missing type");
    }

    #[test]
    fn test_selector_is_emitted() {
        let manifest = ServiceManifest::default();
        let yaml = manifest.generate_yaml().expect("generate_yaml should succeed");
        // Default selector contains "app: trustformers"
        assert!(yaml.contains("app: trustformers"), "selector should appear");
    }

    #[test]
    fn test_custom_selector_is_used() {
        let mut manifest = ServiceManifest::default();
        manifest.selector.clear();
        manifest.selector.insert("component".to_string(), "inference".to_string());
        let yaml = manifest.generate_yaml().expect("generate_yaml should succeed");
        assert!(
            yaml.contains("component: inference"),
            "custom selector should appear"
        );
    }

    #[test]
    fn test_custom_ports() {
        let manifest = ServiceManifest {
            port: 443,
            target_port: 9090,
            ..ServiceManifest::default()
        };
        let yaml = manifest.generate_yaml().expect("generate_yaml should succeed");
        assert!(yaml.contains("port: 443"), "custom port");
        assert!(yaml.contains("targetPort: 9090"), "custom targetPort");
    }
}
