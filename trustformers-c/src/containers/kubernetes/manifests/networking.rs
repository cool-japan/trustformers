//! Kubernetes Networking Manifest Generation
//!
//! This module provides functionality for generating Kubernetes networking manifests
//! including Ingress and NetworkPolicy resources.

use super::formatting::{format_annotations, format_labels, format_selector};
use super::ManifestGenerator;
use crate::error::TrustformersResult;
use std::collections::HashMap;

/// Kubernetes Ingress configuration
#[derive(Debug, Clone)]
pub struct IngressManifest {
    pub name: String,
    pub namespace: String,
    pub hostname: String,
    pub service_name: String,
    pub service_port: i32,
    pub tls_enabled: bool,
    pub labels: HashMap<String, String>,
    pub annotations: HashMap<String, String>,
}

impl Default for IngressManifest {
    fn default() -> Self {
        Self {
            name: "trustformers-ingress".to_string(),
            namespace: "default".to_string(),
            hostname: "trustformers.example.com".to_string(),
            service_name: "trustformers-service".to_string(),
            service_port: 80,
            tls_enabled: false,
            labels: HashMap::new(),
            annotations: HashMap::new(),
        }
    }
}

impl ManifestGenerator for IngressManifest {
    fn generate_yaml(&self) -> TrustformersResult<String> {
        let mut merged_labels = self.labels.clone();
        merged_labels.entry("app".to_string()).or_insert_with(|| self.name.clone());

        let labels_yaml = format_labels(&merged_labels);

        let annotations_section = if self.annotations.is_empty() {
            String::new()
        } else {
            format!(
                "  annotations:\n{}\n",
                format_annotations(&self.annotations)
            )
        };

        let tls_section = if self.tls_enabled {
            format!(
                "  tls:\n  - hosts:\n    - {}\n    secretName: {}-tls\n",
                self.hostname, self.name
            )
        } else {
            String::new()
        };

        let yaml = format!(
            r#"
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {}
  namespace: {}
  labels:
{}
{}spec:
{}  rules:
  - host: {}
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: {}
            port:
              number: {}
"#,
            self.name,
            self.namespace,
            labels_yaml,
            annotations_section,
            tls_section,
            self.hostname,
            self.service_name,
            self.service_port,
        );

        Ok(yaml)
    }
}

/// Kubernetes NetworkPolicy configuration
#[derive(Debug, Clone)]
pub struct NetworkPolicyManifest {
    pub name: String,
    pub namespace: String,
    pub pod_selector: HashMap<String, String>,
    pub ingress_rules: Vec<String>,
    pub egress_rules: Vec<String>,
}

impl Default for NetworkPolicyManifest {
    fn default() -> Self {
        let mut pod_selector = HashMap::new();
        pod_selector.insert("app".to_string(), "trustformers".to_string());

        Self {
            name: "trustformers-network-policy".to_string(),
            namespace: "default".to_string(),
            pod_selector,
            ingress_rules: vec![],
            egress_rules: vec![],
        }
    }
}

impl ManifestGenerator for NetworkPolicyManifest {
    fn generate_yaml(&self) -> TrustformersResult<String> {
        // Build pod selector — fall back to name-based label when empty.
        let mut effective_selector = self.pod_selector.clone();
        if effective_selector.is_empty() {
            effective_selector.insert("app".to_string(), self.name.clone());
        }
        let selector_yaml = format_selector(&effective_selector);

        // Render ingress rules list.
        let ingress_yaml = if self.ingress_rules.is_empty() {
            "  ingress: []\n".to_string()
        } else {
            let items: String = self.ingress_rules.iter().map(|r| format!("  - {}\n", r)).collect();
            format!("  ingress:\n{}", items)
        };

        // Render egress rules list.
        let egress_yaml = if self.egress_rules.is_empty() {
            "  egress: []\n".to_string()
        } else {
            let items: String = self.egress_rules.iter().map(|r| format!("  - {}\n", r)).collect();
            format!("  egress:\n{}", items)
        };

        let yaml = format!(
            r#"
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: {}
  namespace: {}
spec:
  podSelector:
    matchLabels:
{}
  policyTypes:
  - Ingress
  - Egress
{}{}
"#,
            self.name, self.namespace, selector_yaml, ingress_yaml, egress_yaml,
        );

        Ok(yaml)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── IngressManifest ──────────────────────────────────────────────────────

    #[test]
    fn test_ingress_default_manifest() {
        let manifest = IngressManifest::default();
        let yaml = manifest.generate_yaml().expect("generate_yaml should succeed");
        assert!(yaml.contains("kind: Ingress"), "missing kind");
        assert!(
            yaml.contains("trustformers.example.com"),
            "missing hostname"
        );
        assert!(
            yaml.contains("trustformers-service"),
            "missing service name"
        );
        assert!(yaml.contains("number: 80"), "missing service port");
    }

    #[test]
    fn test_ingress_tls_section_added_when_enabled() {
        let manifest = IngressManifest {
            tls_enabled: true,
            ..IngressManifest::default()
        };
        let yaml = manifest.generate_yaml().expect("generate_yaml should succeed");
        assert!(yaml.contains("tls:"), "TLS section should be present");
        assert!(yaml.contains("trustformers-ingress-tls"), "TLS secret name");
    }

    #[test]
    fn test_ingress_no_tls_section_when_disabled() {
        let manifest = IngressManifest::default();
        let yaml = manifest.generate_yaml().expect("generate_yaml should succeed");
        assert!(!yaml.contains("tls:"), "TLS section should be absent");
    }

    #[test]
    fn test_ingress_annotations_appear_in_yaml() {
        let mut manifest = IngressManifest::default();
        manifest.annotations.insert(
            "nginx.ingress.kubernetes.io/rewrite-target".to_string(),
            "/".to_string(),
        );
        let yaml = manifest.generate_yaml().expect("generate_yaml should succeed");
        assert!(
            yaml.contains("annotations:"),
            "annotations section expected"
        );
        assert!(
            yaml.contains("nginx.ingress.kubernetes.io/rewrite-target"),
            "annotation key expected"
        );
    }

    // ── NetworkPolicyManifest ────────────────────────────────────────────────

    #[test]
    fn test_network_policy_default_manifest() {
        let manifest = NetworkPolicyManifest::default();
        let yaml = manifest.generate_yaml().expect("generate_yaml should succeed");
        assert!(yaml.contains("kind: NetworkPolicy"), "missing kind");
        assert!(
            yaml.contains("trustformers-network-policy"),
            "missing policy name"
        );
        assert!(yaml.contains("ingress: []"), "empty ingress should be []");
        assert!(yaml.contains("egress: []"), "empty egress should be []");
    }

    #[test]
    fn test_network_policy_ingress_rules_appear() {
        let mut manifest = NetworkPolicyManifest::default();
        manifest
            .ingress_rules
            .push("from: [{podSelector: {matchLabels: {role: frontend}}}]".to_string());
        let yaml = manifest.generate_yaml().expect("generate_yaml should succeed");
        assert!(yaml.contains("ingress:"), "ingress section expected");
        assert!(yaml.contains("role: frontend"), "rule content expected");
    }

    #[test]
    fn test_network_policy_pod_selector_fallback() {
        let mut manifest = NetworkPolicyManifest::default();
        manifest.pod_selector.clear(); // empty → falls back to name-based label
        let yaml = manifest.generate_yaml().expect("generate_yaml should succeed");
        assert!(
            yaml.contains("trustformers-network-policy"),
            "fallback selector should use name"
        );
    }
}
