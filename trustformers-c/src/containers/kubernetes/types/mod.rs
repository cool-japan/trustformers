//! Kubernetes Configuration Types
//!
//! This module contains all Kubernetes configuration types organized by domain:
//! - Core types (deployments, pods, containers)
//! - Networking types (services, ingress, network policies)
//! - Storage types (ConfigMaps, Secrets, PVs/PVCs)
//! - Security types (security contexts, affinity, tolerations)
//! - Autoscaling types (HPA, VPA, PDB)
//! - Monitoring types (Service Monitors)

// Core Kubernetes types
pub mod autoscaling;
pub mod core;
pub mod monitoring;
pub mod networking;
pub mod security;
pub mod storage;

// Re-export all types for convenience
pub use autoscaling::*;
pub use core::*;
pub use monitoring::*;
pub use networking::*;
pub use security::*;
pub use storage::*;
