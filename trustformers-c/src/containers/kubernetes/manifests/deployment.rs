//! Kubernetes Deployment Manifest Generation
//!
//! This module provides functionality for generating Kubernetes Deployment manifests
//! for TrustformeRS model serving.

use super::formatting::{format_annotations, format_labels};
use super::ManifestGenerator;
use crate::error::TrustformersResult;
use std::collections::HashMap;

/// Kubernetes Deployment configuration
#[derive(Debug, Clone)]
pub struct DeploymentManifest {
    pub name: String,
    pub namespace: String,
    pub replicas: i32,
    pub image: String,
    pub labels: HashMap<String, String>,
    pub annotations: HashMap<String, String>,
}

impl Default for DeploymentManifest {
    fn default() -> Self {
        Self {
            name: "trustformers-deployment".to_string(),
            namespace: "default".to_string(),
            replicas: 1,
            image: "trustformers:latest".to_string(),
            labels: HashMap::new(),
            annotations: HashMap::new(),
        }
    }
}

impl ManifestGenerator for DeploymentManifest {
    fn generate_yaml(&self) -> TrustformersResult<String> {
        // Build merged label set: user-supplied labels plus the mandatory selector label.
        let mut merged_labels = self.labels.clone();
        merged_labels.entry("app".to_string()).or_insert_with(|| self.name.clone());

        let labels_yaml = format_labels(&merged_labels);

        // Annotations section — emit only when non-empty.
        let annotations_section = if self.annotations.is_empty() {
            String::new()
        } else {
            format!(
                "  annotations:\n{}\n",
                format_annotations(&self.annotations)
            )
        };

        let yaml = format!(
            r#"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {}
  namespace: {}
{}spec:
  replicas: {}
  selector:
    matchLabels:
{}
  template:
    metadata:
      labels:
{}
    spec:
      containers:
      - name: trustformers
        image: {}
        ports:
        - containerPort: 8080
"#,
            self.name,
            self.namespace,
            annotations_section,
            self.replicas,
            labels_yaml,
            labels_yaml,
            self.image,
        );

        Ok(yaml)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_manifest_generates_valid_yaml() {
        let manifest = DeploymentManifest::default();
        let yaml = manifest.generate_yaml().expect("generate_yaml should succeed");
        assert!(yaml.contains("kind: Deployment"), "missing kind");
        assert!(
            yaml.contains("name: trustformers-deployment"),
            "missing name"
        );
        assert!(yaml.contains("namespace: default"), "missing namespace");
        assert!(yaml.contains("replicas: 1"), "missing replicas");
        assert!(yaml.contains("image: trustformers:latest"), "missing image");
    }

    #[test]
    fn test_labels_are_included_in_yaml() {
        let mut manifest = DeploymentManifest::default();
        manifest.labels.insert("env".to_string(), "prod".to_string());
        let yaml = manifest.generate_yaml().expect("generate_yaml should succeed");
        assert!(
            yaml.contains("env: prod"),
            "user label should appear in YAML"
        );
        // Default 'app' label is added when not present
        assert!(yaml.contains("app:"), "app label should be present");
    }

    #[test]
    fn test_annotations_section_is_emitted_when_non_empty() {
        let mut manifest = DeploymentManifest::default();
        manifest
            .annotations
            .insert("prometheus.io/scrape".to_string(), "true".to_string());
        let yaml = manifest.generate_yaml().expect("generate_yaml should succeed");
        assert!(
            yaml.contains("annotations:"),
            "annotations section should be present"
        );
        assert!(
            yaml.contains("prometheus.io/scrape"),
            "annotation key should appear"
        );
    }

    #[test]
    fn test_no_annotations_section_when_empty() {
        let manifest = DeploymentManifest::default();
        let yaml = manifest.generate_yaml().expect("generate_yaml should succeed");
        assert!(
            !yaml.contains("annotations:"),
            "annotations section should be absent when empty"
        );
    }

    #[test]
    fn test_custom_replicas_and_image() {
        let manifest = DeploymentManifest {
            replicas: 3,
            image: "myimage:v2".to_string(),
            ..DeploymentManifest::default()
        };
        let yaml = manifest.generate_yaml().expect("generate_yaml should succeed");
        assert!(yaml.contains("replicas: 3"), "custom replica count");
        assert!(yaml.contains("image: myimage:v2"), "custom image");
    }
}
