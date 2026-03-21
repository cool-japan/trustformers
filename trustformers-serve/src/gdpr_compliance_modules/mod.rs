//! GDPR Compliance System Modules
//!
//! This module provides comprehensive GDPR compliance and data privacy management
//! capabilities organized into focused submodules for better maintainability.
//!
//! The compliance system implements:
//! - Data subject rights management
//! - Consent management and tracking
//! - Data processing oversight
//! - Privacy impact assessments
//! - Breach detection and notification
//! - Cross-border transfer compliance
//! - Privacy by design principles

pub mod breach_management;
pub mod compliance_monitoring;
pub mod consent_management;
pub mod cross_border;
pub mod data_processing;
pub mod data_retention;
pub mod data_subject_rights;
pub mod privacy_by_design;
pub mod privacy_impact;
pub mod service;
pub mod types;

// Re-export core types and functionality for convenience
pub use breach_management::*;
pub use compliance_monitoring::*;
pub use consent_management::*;
pub use cross_border::*;
pub use data_processing::*;
pub use data_retention::*;
pub use data_subject_rights::*;
pub use privacy_by_design::*;
pub use privacy_impact::*;
pub use service::*;
pub use types::*;
