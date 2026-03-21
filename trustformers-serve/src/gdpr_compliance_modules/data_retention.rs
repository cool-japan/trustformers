//! Data retention and deletion policies
//!
//! This module implements data retention policies and automated deletion
//! as required by GDPR Article 5(1)(e) - storage limitation principle.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Data retention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRetentionConfig {
    /// Enable data retention management
    pub enabled: bool,
    /// Default retention period
    pub default_retention_period: Duration,
    /// Retention policies
    pub retention_policies: Vec<RetentionPolicy>,
    /// Automatic deletion
    pub auto_deletion: bool,
    /// Legal hold support
    pub legal_hold: LegalHoldConfig,
    /// Retention review
    pub retention_review: RetentionReviewConfig,
}

impl Default for DataRetentionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_retention_period: Duration::from_secs(365 * 24 * 3600), // 1 year
            retention_policies: Vec::new(),
            auto_deletion: true,
            legal_hold: LegalHoldConfig::default(),
            retention_review: RetentionReviewConfig::default(),
        }
    }
}

/// Retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Policy identifier
    pub id: String,
    /// Policy name
    pub name: String,
    /// Data categories covered
    pub data_categories: Vec<String>,
    /// Retention period
    pub retention_period: Duration,
    /// Deletion method
    pub deletion_method: DeletionMethod,
    /// Exceptions
    pub exceptions: Vec<RetentionException>,
}

/// Deletion methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeletionMethod {
    /// Secure deletion
    SecureDelete,
    /// Anonymization
    Anonymize,
    /// Archival
    Archive,
}

/// Retention exceptions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetentionException {
    /// Legal requirement
    LegalRequirement,
    /// Ongoing litigation
    Litigation,
    /// Regulatory investigation
    Investigation,
}

/// Legal hold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegalHoldConfig {
    /// Enable legal holds
    pub enabled: bool,
    /// Automatic hold detection
    pub auto_detection: bool,
    /// Hold notification
    pub notifications: bool,
}

impl Default for LegalHoldConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_detection: false,
            notifications: true,
        }
    }
}

/// Retention review configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionReviewConfig {
    /// Enable periodic review
    pub enabled: bool,
    /// Review frequency
    pub review_frequency: Duration,
    /// Automatic approval
    pub auto_approval: bool,
}

impl Default for RetentionReviewConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            review_frequency: Duration::from_secs(90 * 24 * 3600), // 90 days
            auto_approval: false,
        }
    }
}
