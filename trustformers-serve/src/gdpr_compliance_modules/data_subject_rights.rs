//! Data subject rights implementation
//!
//! This module implements the rights of data subjects under GDPR Chapter III,
//! including access, rectification, erasure, portability, and objection rights.

use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

use super::types::{RequestStatus, RequestType, VerificationStatus};

/// Data subject rights configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSubjectRightsConfig {
    /// Enable data subject rights
    pub enabled: bool,
    /// Right of access configuration
    pub right_of_access: RightOfAccessConfig,
    /// Right to rectification configuration
    pub right_to_rectification: RightToRectificationConfig,
    /// Right to erasure configuration
    pub right_to_erasure: RightToErasureConfig,
    /// Right to restrict processing
    pub right_to_restrict: RightToRestrictConfig,
    /// Right to data portability
    pub right_to_portability: RightToPortabilityConfig,
    /// Right to object
    pub right_to_object: RightToObjectConfig,
    /// Request handling
    pub request_handling: RequestHandlingConfig,
}

impl Default for DataSubjectRightsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            right_of_access: RightOfAccessConfig::default(),
            right_to_rectification: RightToRectificationConfig::default(),
            right_to_erasure: RightToErasureConfig::default(),
            right_to_restrict: RightToRestrictConfig::default(),
            right_to_portability: RightToPortabilityConfig::default(),
            right_to_object: RightToObjectConfig::default(),
            request_handling: RequestHandlingConfig::default(),
        }
    }
}

/// Right of access configuration (Article 15)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RightOfAccessConfig {
    /// Enable right of access
    pub enabled: bool,
    /// Maximum requests per period
    pub max_requests_per_period: u32,
    /// Request period
    pub request_period: Duration,
    /// Export formats supported
    pub export_formats: Vec<ExportFormat>,
    /// Include metadata
    pub include_metadata: bool,
    /// Response timeframe
    pub response_timeframe: Duration,
}

impl Default for RightOfAccessConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_requests_per_period: 3,
            request_period: Duration::from_secs(365 * 24 * 3600), // 1 year
            export_formats: vec![ExportFormat::JSON, ExportFormat::PDF],
            include_metadata: true,
            response_timeframe: Duration::from_secs(30 * 24 * 3600), // 30 days
        }
    }
}

/// Export formats for data access requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    /// JSON format
    JSON,
    /// PDF format
    PDF,
    /// CSV format
    CSV,
    /// XML format
    XML,
}

/// Right to rectification configuration (Article 16)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RightToRectificationConfig {
    /// Enable right to rectification
    pub enabled: bool,
    /// Automatic verification
    pub auto_verification: bool,
    /// Verification requirements
    pub verification_required: bool,
    /// Third party notification
    pub notify_third_parties: bool,
    /// Response timeframe
    pub response_timeframe: Duration,
}

impl Default for RightToRectificationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_verification: false,
            verification_required: true,
            notify_third_parties: true,
            response_timeframe: Duration::from_secs(30 * 24 * 3600), // 30 days
        }
    }
}

/// Right to erasure configuration (Article 17)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RightToErasureConfig {
    /// Enable right to erasure
    pub enabled: bool,
    /// Erasure methods
    pub erasure_methods: Vec<ErasureMethod>,
    /// Verification required
    pub verification_required: bool,
    /// Legal exceptions check
    pub check_exceptions: bool,
    /// Third party notification
    pub notify_third_parties: bool,
    /// Response timeframe
    pub response_timeframe: Duration,
}

impl Default for RightToErasureConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            erasure_methods: vec![ErasureMethod::SecureDelete, ErasureMethod::Anonymization],
            verification_required: true,
            check_exceptions: true,
            notify_third_parties: true,
            response_timeframe: Duration::from_secs(30 * 24 * 3600), // 30 days
        }
    }
}

/// Erasure methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErasureMethod {
    /// Secure deletion
    SecureDelete,
    /// Data anonymization
    Anonymization,
    /// Data pseudonymization
    Pseudonymization,
    /// Archive with restricted access
    Archive,
}

/// Right to restrict processing configuration (Article 18)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RightToRestrictConfig {
    /// Enable right to restrict
    pub enabled: bool,
    /// Restriction methods
    pub restriction_methods: Vec<RestrictionMethod>,
    /// Temporary restriction
    pub allow_temporary: bool,
    /// Response timeframe
    pub response_timeframe: Duration,
}

impl Default for RightToRestrictConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            restriction_methods: vec![
                RestrictionMethod::ProcessingHalt,
                RestrictionMethod::AccessRestriction,
            ],
            allow_temporary: true,
            response_timeframe: Duration::from_secs(30 * 24 * 3600), // 30 days
        }
    }
}

/// Restriction methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RestrictionMethod {
    /// Halt all processing
    ProcessingHalt,
    /// Restrict access
    AccessRestriction,
    /// Move to separate storage
    Quarantine,
}

/// Right to data portability configuration (Article 20)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RightToPortabilityConfig {
    /// Enable right to portability
    pub enabled: bool,
    /// Portable formats
    pub portable_formats: Vec<PortableFormat>,
    /// Direct transfer support
    pub direct_transfer: bool,
    /// Response timeframe
    pub response_timeframe: Duration,
}

impl Default for RightToPortabilityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            portable_formats: vec![PortableFormat::JSON, PortableFormat::CSV],
            direct_transfer: false,
            response_timeframe: Duration::from_secs(30 * 24 * 3600), // 30 days
        }
    }
}

/// Portable formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PortableFormat {
    /// JSON format
    JSON,
    /// CSV format
    CSV,
    /// XML format
    XML,
}

/// Right to object configuration (Article 21)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RightToObjectConfig {
    /// Enable right to object
    pub enabled: bool,
    /// Objection grounds
    pub objection_grounds: Vec<ObjectionGround>,
    /// Override conditions
    pub override_conditions: Vec<OverrideCondition>,
    /// Response timeframe
    pub response_timeframe: Duration,
}

impl Default for RightToObjectConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            objection_grounds: vec![
                ObjectionGround::LegitimateInterests,
                ObjectionGround::DirectMarketing,
            ],
            override_conditions: vec![OverrideCondition::LegalRequirement],
            response_timeframe: Duration::from_secs(30 * 24 * 3600), // 30 days
        }
    }
}

/// Objection grounds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ObjectionGround {
    /// Legitimate interests processing
    LegitimateInterests,
    /// Direct marketing
    DirectMarketing,
    /// Public interest
    PublicInterest,
    /// Scientific research
    ScientificResearch,
}

/// Override conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OverrideCondition {
    /// Legal requirement
    LegalRequirement,
    /// Vital interests
    VitalInterests,
    /// Legitimate interests override
    LegitimateInterestsOverride,
}

/// Request handling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestHandlingConfig {
    /// Enable request handling
    pub enabled: bool,
    /// Verification requirements
    pub verification: RequestVerificationConfig,
    /// Response timeframes
    pub timeframes: RequestTimeframesConfig,
    /// Notification settings
    pub notifications: RequestNotificationConfig,
}

impl Default for RequestHandlingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            verification: RequestVerificationConfig::default(),
            timeframes: RequestTimeframesConfig::default(),
            notifications: RequestNotificationConfig::default(),
        }
    }
}

/// Request verification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestVerificationConfig {
    /// Enable verification
    pub enabled: bool,
    /// Verification methods
    pub methods: Vec<String>,
    /// Verification timeframe
    pub timeframe: Duration,
}

impl Default for RequestVerificationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            methods: vec!["email".to_string(), "identity_document".to_string()],
            timeframe: Duration::from_secs(7 * 24 * 3600), // 7 days
        }
    }
}

/// Request timeframes configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestTimeframesConfig {
    /// Default response timeframe
    pub default_response: Duration,
    /// Complex request timeframe
    pub complex_response: Duration,
    /// Extension notification timeframe
    pub extension_notification: Duration,
}

impl Default for RequestTimeframesConfig {
    fn default() -> Self {
        Self {
            default_response: Duration::from_secs(30 * 24 * 3600), // 30 days
            complex_response: Duration::from_secs(60 * 24 * 3600), // 60 days
            extension_notification: Duration::from_secs(25 * 24 * 3600), // 25 days
        }
    }
}

/// Request notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestNotificationConfig {
    /// Enable notifications
    pub enabled: bool,
    /// Notification channels
    pub channels: Vec<NotificationChannel>,
    /// Acknowledgment required
    pub acknowledgment_required: bool,
}

impl Default for RequestNotificationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            channels: vec![NotificationChannel::Email],
            acknowledgment_required: false,
        }
    }
}

/// Notification channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannel {
    /// Email notification
    Email,
    /// SMS notification
    SMS,
    /// Push notification
    Push,
    /// Postal mail
    Mail,
}

/// Data subject request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSubjectRequest {
    /// Request identifier
    pub id: String,
    /// Data subject identifier
    pub subject_id: String,
    /// Request type
    pub request_type: RequestType,
    /// Request status
    pub status: RequestStatus,
    /// Request details
    pub details: RequestDetails,
    /// Verification status
    pub verification_status: VerificationStatus,
    /// Submitted timestamp
    pub submitted_at: SystemTime,
    /// Completed timestamp
    pub completed_at: Option<SystemTime>,
}

/// Request details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestDetails {
    /// Request description
    pub description: String,
    /// Specific data categories
    pub data_categories: Vec<String>,
    /// Processing activities
    pub processing_activities: Vec<String>,
    /// Additional information
    pub additional_info: Option<String>,
}
