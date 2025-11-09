//! Kubernetes Manifest Generation
//!
//! This module provides functionality for generating Kubernetes YAML manifests
//! from configuration structures.

use crate::error::TrustformersResult;

pub mod autoscaling;
pub mod deployment;
pub mod networking;
pub mod service;
pub mod storage;

// Re-export key functionality
pub use autoscaling::*;
pub use deployment::*;
pub use networking::*;
pub use service::*;
pub use storage::*;

/// Trait for generating Kubernetes manifests
pub trait ManifestGenerator {
    /// Generate YAML manifest as string
    fn generate_yaml(&self) -> TrustformersResult<String>;
}

/// Common formatting functions for YAML generation
pub mod formatting {
    use std::collections::HashMap;

    /// Format labels as YAML
    pub fn format_labels(labels: &HashMap<String, String>) -> String {
        if labels.is_empty() {
            return "    {}".to_string();
        }

        labels
            .iter()
            .map(|(k, v)| format!("    {}: {}", k, v))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Format annotations as YAML
    pub fn format_annotations(annotations: &HashMap<String, String>) -> String {
        if annotations.is_empty() {
            return "    {}".to_string();
        }

        annotations
            .iter()
            .map(|(k, v)| format!("    {}: {}", k, v))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Format selector as YAML
    pub fn format_selector(selector: &HashMap<String, String>) -> String {
        if selector.is_empty() {
            return "    {}".to_string();
        }

        selector
            .iter()
            .map(|(k, v)| format!("    {}: {}", k, v))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Format resource requirements as YAML
    pub fn format_resources(resources: &HashMap<String, String>) -> String {
        if resources.is_empty() {
            return "            {}".to_string();
        }

        resources
            .iter()
            .map(|(k, v)| format!("            {}: {}", k, v))
            .collect::<Vec<_>>()
            .join("\n")
    }
}
