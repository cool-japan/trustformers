//! Consent management system
//!
//! This module implements comprehensive consent management capabilities as required
//! by GDPR Article 7, including consent collection, storage, verification, and withdrawal.

use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

use super::types::ConsentStatus;

/// Consent management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentManagementConfig {
    /// Enable consent management
    pub enabled: bool,
    /// Consent storage configuration
    pub storage: ConsentStorageConfig,
    /// Consent verification
    pub verification: ConsentVerificationConfig,
    /// Consent withdrawal
    pub withdrawal: ConsentWithdrawalConfig,
    /// Granular consent
    pub granular_consent: bool,
    /// Consent renewal
    pub renewal: ConsentRenewalConfig,
}

impl Default for ConsentManagementConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            storage: ConsentStorageConfig::default(),
            verification: ConsentVerificationConfig::default(),
            withdrawal: ConsentWithdrawalConfig::default(),
            granular_consent: true,
            renewal: ConsentRenewalConfig::default(),
        }
    }
}

/// Consent storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentStorageConfig {
    /// Storage backend
    pub backend: ConsentStorageBackend,
    /// Encryption enabled
    pub encryption: bool,
    /// Consent audit trail
    pub audit_trail: bool,
    /// Tamper protection
    pub tamper_protection: bool,
}

impl Default for ConsentStorageConfig {
    fn default() -> Self {
        Self {
            backend: ConsentStorageBackend::Database,
            encryption: true,
            audit_trail: true,
            tamper_protection: true,
        }
    }
}

/// Consent storage backends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsentStorageBackend {
    /// Database storage
    Database,
    /// File system storage
    FileSystem { path: String },
    /// External consent management platform
    External { endpoint: String, api_key: String },
    /// Blockchain storage
    Blockchain { network: String },
}

/// Consent verification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentVerificationConfig {
    /// Enable verification
    pub enabled: bool,
    /// Verification methods
    pub methods: Vec<VerificationMethod>,
    /// Multi-factor verification
    pub multi_factor: bool,
    /// Verification validity period
    pub validity_period: Duration,
}

impl Default for ConsentVerificationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            methods: vec![VerificationMethod::Email, VerificationMethod::SMS],
            multi_factor: false,
            validity_period: Duration::from_secs(365 * 24 * 3600), // 1 year
        }
    }
}

/// Verification methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationMethod {
    /// Email verification
    Email,
    /// SMS verification
    SMS,
    /// Digital signature
    DigitalSignature,
    /// Biometric verification
    Biometric,
    /// Two-factor authentication
    TwoFactor,
}

/// Consent withdrawal configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentWithdrawalConfig {
    /// Enable withdrawal
    pub enabled: bool,
    /// Withdrawal methods
    pub methods: Vec<WithdrawalMethod>,
    /// Immediate effect
    pub immediate_effect: bool,
    /// Grace period
    pub grace_period: Option<Duration>,
    /// Confirmation required
    pub confirmation_required: bool,
}

impl Default for ConsentWithdrawalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            methods: vec![
                WithdrawalMethod::WebPortal,
                WithdrawalMethod::Email,
                WithdrawalMethod::API,
            ],
            immediate_effect: true,
            grace_period: None,
            confirmation_required: true,
        }
    }
}

/// Consent withdrawal methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WithdrawalMethod {
    /// Web portal
    WebPortal,
    /// Email request
    Email,
    /// API call
    API,
    /// Phone call
    Phone,
    /// Postal mail
    Mail,
}

/// Consent renewal configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRenewalConfig {
    /// Enable automatic renewal requests
    pub enabled: bool,
    /// Renewal period
    pub renewal_period: Duration,
    /// Advance notice period
    pub notice_period: Duration,
    /// Automatic expiry
    pub auto_expiry: bool,
}

impl Default for ConsentRenewalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            renewal_period: Duration::from_secs(365 * 24 * 3600), // 1 year
            notice_period: Duration::from_secs(30 * 24 * 3600),   // 30 days
            auto_expiry: true,
        }
    }
}

/// Consent record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRecord {
    /// Unique consent identifier
    pub id: String,
    /// Data subject identifier
    pub subject_id: String,
    /// Processing purpose
    pub purpose: String,
    /// Consent status
    pub status: ConsentStatus,
    /// Consent mechanism used
    pub mechanism: ConsentMechanism,
    /// Timestamp when consent was given
    pub given_at: SystemTime,
    /// Timestamp when consent was withdrawn (if applicable)
    pub withdrawn_at: Option<SystemTime>,
    /// Consent expiry date
    pub expires_at: Option<SystemTime>,
    /// Evidence of consent
    pub evidence: ConsentEvidence,
    /// Consent version
    pub version: String,
}

/// Consent mechanism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsentMechanism {
    /// Web form
    WebForm,
    /// Email opt-in
    EmailOptIn,
    /// API call
    API,
    /// Mobile app
    MobileApp,
    /// Physical form
    PhysicalForm,
    /// Verbal consent
    Verbal,
    /// Implied consent
    Implied,
}

/// Consent evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentEvidence {
    /// IP address
    pub ip_address: Option<String>,
    /// User agent
    pub user_agent: Option<String>,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Digital signature
    pub digital_signature: Option<String>,
    /// Witness information
    pub witness: Option<String>,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}
