//! Monitoring Configuration Types
//!
//! This module contains monitoring-related Kubernetes configuration structures
//! including Service Monitor configurations for Prometheus integration.

use serde::{Deserialize, Serialize};

use super::security::LabelSelector;

/// Service Monitor configuration
///
/// Defines how Prometheus should scrape metrics from services.
/// This is a custom resource definition (CRD) from the Prometheus Operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMonitorConfig {
    /// Endpoints to scrape
    pub endpoints: Vec<ServiceMonitorEndpoint>,
    /// Service selector
    pub selector: LabelSelector,
    /// Namespace selector (optional)
    pub namespace_selector: Option<NamespaceSelector>,
}

/// Service monitor endpoint
///
/// Configuration for a single endpoint to scrape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMonitorEndpoint {
    /// Port name or number to scrape
    pub port: String,
    /// HTTP path to scrape (default: /metrics)
    pub path: Option<String>,
    /// Scrape interval (default: 30s)
    pub interval: Option<String>,
    /// Scrape timeout (default: 10s)
    pub scrape_timeout: Option<String>,
    /// Honor labels from scraped data
    pub honor_labels: Option<bool>,
    /// Honor timestamps from scraped data
    pub honor_timestamps: Option<bool>,
}

/// Namespace selector
///
/// Selects namespaces for monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespaceSelector {
    /// Monitor all namespaces
    pub any: Option<bool>,
    /// Specific namespace names to monitor
    pub match_names: Vec<String>,
}

/// Default implementations for monitoring types
impl Default for ServiceMonitorConfig {
    fn default() -> Self {
        let mut selector = LabelSelector::default();
        selector.match_labels.insert("app".to_string(), "trustformers".to_string());

        Self {
            endpoints: vec![ServiceMonitorEndpoint::default()],
            selector,
            namespace_selector: None,
        }
    }
}

impl Default for ServiceMonitorEndpoint {
    fn default() -> Self {
        Self {
            port: "http".to_string(),
            path: Some("/metrics".to_string()),
            interval: Some("30s".to_string()),
            scrape_timeout: Some("10s".to_string()),
            honor_labels: Some(false),
            honor_timestamps: Some(true),
        }
    }
}

impl Default for NamespaceSelector {
    fn default() -> Self {
        Self {
            any: Some(false),
            match_names: vec!["default".to_string()],
        }
    }
}
