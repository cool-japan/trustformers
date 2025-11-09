//! GDPR Compliance System
//!
//! Comprehensive GDPR compliance and data privacy management system implementing
//! data subject rights, privacy by design, consent management, and regulatory compliance.

pub mod consent;
pub mod data_processing;
pub mod manager;
pub mod monitoring;
pub mod rights;
pub mod types;

// Re-export main types
pub use types::*;
