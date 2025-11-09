//! Kubernetes Storage Manifest Generation
//!
//! This module provides functionality for generating Kubernetes storage manifests
//! including PersistentVolumes and PersistentVolumeClaims.

use super::ManifestGenerator;
use crate::error::TrustformersResult;
use std::collections::HashMap;

/// Kubernetes PersistentVolumeClaim configuration
#[derive(Debug, Clone)]
pub struct PersistentVolumeClaimManifest {
    pub name: String,
    pub namespace: String,
    pub storage_size: String,
    pub access_modes: Vec<String>,
    pub storage_class: Option<String>,
    pub labels: HashMap<String, String>,
}

impl Default for PersistentVolumeClaimManifest {
    fn default() -> Self {
        Self {
            name: "trustformers-pvc".to_string(),
            namespace: "default".to_string(),
            storage_size: "10Gi".to_string(),
            access_modes: vec!["ReadWriteOnce".to_string()],
            storage_class: None,
            labels: HashMap::new(),
        }
    }
}

impl ManifestGenerator for PersistentVolumeClaimManifest {
    fn generate_yaml(&self) -> TrustformersResult<String> {
        let mut yaml = format!(
            r#"apiVersion: v1
kind: PersistentVolumeClaim
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
        yaml.push_str("\nspec:");

        // Add all access modes
        yaml.push_str("\n  accessModes:");
        for mode in &self.access_modes {
            yaml.push_str(&format!("\n  - {}", mode));
        }

        // Add storage class if specified
        if let Some(ref sc) = self.storage_class {
            yaml.push_str(&format!("\n  storageClassName: {}", sc));
        }

        // Add storage resources
        yaml.push_str(&format!(
            r#"
  resources:
    requests:
      storage: {}
"#,
            self.storage_size
        ));

        Ok(yaml)
    }
}
