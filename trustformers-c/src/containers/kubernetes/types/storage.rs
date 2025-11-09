//! Storage Configuration Types
//!
//! This module contains storage-related Kubernetes configuration structures
//! including ConfigMaps, Secrets, Persistent Volumes, and Persistent Volume Claims.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::core::ResourceRequirements;

/// ConfigMap configuration
///
/// Defines a Kubernetes ConfigMap for storing non-confidential configuration data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigMapConfig {
    /// ConfigMap name
    pub name: String,
    /// Data (UTF-8 string data)
    pub data: HashMap<String, String>,
    /// Binary data (base64-encoded)
    pub binary_data: HashMap<String, String>,
}

/// Secret configuration
///
/// Defines a Kubernetes Secret for storing sensitive data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretConfig {
    /// Secret name
    pub name: String,
    /// Secret type
    pub secret_type: SecretType,
    /// Data (base64-encoded)
    pub data: HashMap<String, String>,
    /// String data (plain text, will be base64-encoded)
    pub string_data: HashMap<String, String>,
}

/// Secret types
///
/// Different types of Kubernetes secrets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecretType {
    /// Arbitrary user-defined data
    Opaque,
    /// Docker registry authentication
    DockerConfigJson,
    /// Legacy Docker registry authentication
    DockerCfg,
    /// Basic authentication credentials
    BasicAuth,
    /// SSH authentication
    SSHAuth,
    /// TLS certificates and keys
    TLS,
    /// Service account token
    ServiceAccountToken,
}

/// Persistent Volume configuration
///
/// Defines a persistent volume and its associated claim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PVConfig {
    /// PV name
    pub name: String,
    /// Storage class
    pub storage_class: String,
    /// Access modes
    pub access_modes: Vec<AccessMode>,
    /// Storage capacity
    pub capacity: String,
    /// Persistent Volume Claim
    pub pvc: PVCConfig,
}

/// Access modes for persistent volumes
///
/// Defines how a persistent volume can be accessed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessMode {
    /// Volume can be mounted as read-write by a single node
    ReadWriteOnce,
    /// Volume can be mounted as read-only by many nodes
    ReadOnlyMany,
    /// Volume can be mounted as read-write by many nodes
    ReadWriteMany,
    /// Volume can be mounted as read-write by a single pod
    ReadWriteOncePod,
}

/// Persistent Volume Claim configuration
///
/// Defines a request for persistent storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PVCConfig {
    /// PVC name
    pub name: String,
    /// Access modes requested
    pub access_modes: Vec<AccessMode>,
    /// Resource requirements (storage size)
    pub resources: ResourceRequirements,
    /// Storage class name
    pub storage_class: Option<String>,
    /// Specific volume name to bind to
    pub volume_name: Option<String>,
}

/// Default implementations for storage types
impl Default for ConfigMapConfig {
    fn default() -> Self {
        let mut data = HashMap::new();
        data.insert(
            "app.config".to_string(),
            "# Application configuration".to_string(),
        );

        Self {
            name: "trustformers-config".to_string(),
            data,
            binary_data: HashMap::new(),
        }
    }
}

impl Default for SecretConfig {
    fn default() -> Self {
        Self {
            name: "trustformers-secrets".to_string(),
            secret_type: SecretType::Opaque,
            data: HashMap::new(),
            string_data: HashMap::new(),
        }
    }
}

impl Default for PVConfig {
    fn default() -> Self {
        Self {
            name: "trustformers-storage".to_string(),
            storage_class: "standard".to_string(),
            access_modes: vec![AccessMode::ReadWriteOnce],
            capacity: "10Gi".to_string(),
            pvc: PVCConfig::default(),
        }
    }
}

impl Default for PVCConfig {
    fn default() -> Self {
        let mut requests = HashMap::new();
        requests.insert("storage".to_string(), "10Gi".to_string());

        Self {
            name: "trustformers-storage-claim".to_string(),
            access_modes: vec![AccessMode::ReadWriteOnce],
            resources: ResourceRequirements {
                requests,
                limits: HashMap::new(),
            },
            storage_class: Some("standard".to_string()),
            volume_name: None,
        }
    }
}

impl ToString for AccessMode {
    fn to_string(&self) -> String {
        match self {
            AccessMode::ReadWriteOnce => "ReadWriteOnce".to_string(),
            AccessMode::ReadOnlyMany => "ReadOnlyMany".to_string(),
            AccessMode::ReadWriteMany => "ReadWriteMany".to_string(),
            AccessMode::ReadWriteOncePod => "ReadWriteOncePod".to_string(),
        }
    }
}

impl ToString for SecretType {
    fn to_string(&self) -> String {
        match self {
            SecretType::Opaque => "Opaque".to_string(),
            SecretType::DockerConfigJson => "kubernetes.io/dockerconfigjson".to_string(),
            SecretType::DockerCfg => "kubernetes.io/dockercfg".to_string(),
            SecretType::BasicAuth => "kubernetes.io/basic-auth".to_string(),
            SecretType::SSHAuth => "kubernetes.io/ssh-auth".to_string(),
            SecretType::TLS => "kubernetes.io/tls".to_string(),
            SecretType::ServiceAccountToken => "kubernetes.io/service-account-token".to_string(),
        }
    }
}
