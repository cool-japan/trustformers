//! Core GDPR types and enums
//!
//! This module contains the fundamental types and enumerations used throughout
//! the GDPR compliance system.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// GDPR compliance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GdprComplianceConfig {
    /// Enable GDPR compliance features
    pub enabled: bool,
    /// Data processing configuration
    pub data_processing: crate::gdpr_compliance_modules::data_processing::DataProcessingConfig,
    /// Consent management configuration
    pub consent_management:
        crate::gdpr_compliance_modules::consent_management::ConsentManagementConfig,
    /// Data subject rights configuration
    pub data_subject_rights:
        crate::gdpr_compliance_modules::data_subject_rights::DataSubjectRightsConfig,
    /// Data retention configuration
    pub data_retention: crate::gdpr_compliance_modules::data_retention::DataRetentionConfig,
    /// Privacy impact assessment configuration
    pub privacy_impact_assessment:
        crate::gdpr_compliance_modules::privacy_impact::PrivacyImpactAssessmentConfig,
    /// Data breach management configuration
    pub data_breach_management:
        crate::gdpr_compliance_modules::breach_management::DataBreachManagementConfig,
    /// Cross-border transfer configuration
    pub cross_border_transfers:
        crate::gdpr_compliance_modules::cross_border::CrossBorderTransferConfig,
    /// Compliance monitoring configuration
    pub compliance_monitoring:
        crate::gdpr_compliance_modules::compliance_monitoring::ComplianceMonitoringConfig,
    /// Privacy by design configuration
    pub privacy_by_design: crate::gdpr_compliance_modules::privacy_by_design::PrivacyByDesignConfig,
}

impl Default for GdprComplianceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            data_processing: Default::default(),
            consent_management: Default::default(),
            data_subject_rights: Default::default(),
            data_retention: Default::default(),
            privacy_impact_assessment: Default::default(),
            data_breach_management: Default::default(),
            cross_border_transfers: Default::default(),
            compliance_monitoring: Default::default(),
            privacy_by_design: Default::default(),
        }
    }
}

/// Legal bases for data processing under GDPR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LegalBasis {
    /// Article 6(1)(a) - Consent
    Consent,
    /// Article 6(1)(b) - Contract
    Contract,
    /// Article 6(1)(c) - Legal obligation
    LegalObligation,
    /// Article 6(1)(d) - Vital interests
    VitalInterests,
    /// Article 6(1)(e) - Public task
    PublicTask,
    /// Article 6(1)(f) - Legitimate interests
    LegitimateInterests,
}

/// Processing purpose
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingPurpose {
    /// Purpose identifier
    pub id: String,
    /// Purpose name
    pub name: String,
    /// Purpose description
    pub description: String,
    /// Legal basis for this purpose
    pub legal_basis: LegalBasis,
    /// Data retention period
    pub retention_period: Duration,
}

/// Data categories under GDPR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataCategory {
    /// Regular personal data
    PersonalData,
    /// Special category data (Article 9)
    SpecialCategoryData,
    /// Pseudonymized data
    PseudonymizedData,
    /// Anonymized data
    AnonymizedData,
    /// Data relating to criminal convictions
    CriminalData,
}

/// Data subject representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSubject {
    /// Unique identifier
    pub id: String,
    /// Email address
    pub email: Option<String>,
    /// Phone number
    pub phone: Option<String>,
    /// Physical address
    pub address: Option<String>,
    /// Registration timestamp
    pub registered_at: std::time::SystemTime,
    /// Last activity timestamp
    pub last_activity: std::time::SystemTime,
    /// Data subject preferences
    pub preferences: DataSubjectPreferences,
}

/// Data subject preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSubjectPreferences {
    /// Preferred language
    pub language: String,
    /// Communication preferences
    pub communication_channels: Vec<String>,
    /// Data processing preferences
    pub data_processing_opt_ins: Vec<String>,
    /// Marketing preferences
    pub marketing_consent: bool,
}

/// Consent status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsentStatus {
    /// Consent given
    Given,
    /// Consent withdrawn
    Withdrawn,
    /// Consent expired
    Expired,
    /// Consent pending
    Pending,
    /// Consent refused
    Refused,
}

/// Request type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequestType {
    /// Right of access (Article 15)
    Access,
    /// Right to rectification (Article 16)
    Rectification,
    /// Right to erasure (Article 17)
    Erasure,
    /// Right to restrict processing (Article 18)
    Restriction,
    /// Right to data portability (Article 20)
    Portability,
    /// Right to object (Article 21)
    Objection,
    /// Rights related to automated decision-making (Article 22)
    AutomatedDecision,
}

/// Request status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequestStatus {
    /// Request submitted
    Submitted,
    /// Request under review
    UnderReview,
    /// Request requires verification
    VerificationRequired,
    /// Request verified
    Verified,
    /// Request being processed
    Processing,
    /// Request completed
    Completed,
    /// Request rejected
    Rejected,
    /// Request cancelled
    Cancelled,
}

/// Verification status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationStatus {
    /// Not verified
    NotVerified,
    /// Pending verification
    Pending,
    /// Verified successfully
    Verified,
    /// Verification failed
    Failed,
    /// Verification expired
    Expired,
}

/// Contact details structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactDetails {
    /// Name
    pub name: String,
    /// Email address
    pub email: String,
    /// Phone number
    pub phone: Option<String>,
    /// Physical address
    pub address: Option<String>,
}

/// GDPR compliance error enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GdprComplianceError {
    /// Configuration error
    ConfigurationError(String),
    /// Consent management error
    ConsentError(String),
    /// Data subject request error
    RequestError(String),
    /// Data processing error
    ProcessingError(String),
    /// Verification error
    VerificationError(String),
    /// Storage error
    StorageError(String),
    /// Network error
    NetworkError(String),
    /// Permission error
    PermissionError(String),
    /// Internal error
    InternalError(String),
}
