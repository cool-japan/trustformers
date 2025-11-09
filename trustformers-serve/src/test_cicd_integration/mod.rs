//! CI/CD Integration and Configuration Management
//!
//! This module provides comprehensive CI/CD pipeline integration and configuration
//! management capabilities for the TrustformeRS test parallelization framework,
//! including environment-specific configurations, automated optimization, and
//! reporting integration for various CI/CD systems.

pub mod compliance;
pub mod environment;
pub mod manager;
pub mod optimization;
pub mod pipeline;
pub mod reporting;
pub mod security;
pub mod types;

// Re-export main types
pub use manager::CicdIntegrationManager;
pub use types::*;
