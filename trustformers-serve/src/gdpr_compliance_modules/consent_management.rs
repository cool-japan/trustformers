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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consent_management_config_default() {
        let cfg = ConsentManagementConfig::default();
        assert!(cfg.enabled);
        assert!(cfg.granular_consent);
        assert!(cfg.withdrawal.enabled);
        assert!(cfg.verification.enabled);
    }

    #[test]
    fn test_consent_storage_config_default() {
        let cfg = ConsentStorageConfig::default();
        assert!(cfg.encryption);
        assert!(cfg.audit_trail);
        assert!(cfg.tamper_protection);
        assert!(matches!(cfg.backend, ConsentStorageBackend::Database));
    }

    #[test]
    fn test_consent_storage_backend_filesystem() {
        let fs = ConsentStorageBackend::FileSystem {
            path: "/tmp/consent".to_string(),
        };
        match fs {
            ConsentStorageBackend::FileSystem { path } => assert_eq!(path, "/tmp/consent"),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_consent_storage_backend_external() {
        let ext = ConsentStorageBackend::External {
            endpoint: "https://consent.example.com".to_string(),
            api_key: "key123".to_string(),
        };
        match ext {
            ConsentStorageBackend::External { endpoint, api_key } => {
                assert!(endpoint.starts_with("https://"));
                assert!(!api_key.is_empty());
            },
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_consent_verification_config_default() {
        let cfg = ConsentVerificationConfig::default();
        assert!(cfg.enabled);
        assert!(!cfg.multi_factor);
        assert_eq!(cfg.methods.len(), 2);
    }

    #[test]
    fn test_verification_method_variants() {
        assert_eq!(format!("{:?}", VerificationMethod::Email), "Email");
        assert_eq!(format!("{:?}", VerificationMethod::SMS), "SMS");
        assert_eq!(
            format!("{:?}", VerificationMethod::DigitalSignature),
            "DigitalSignature"
        );
        assert_eq!(format!("{:?}", VerificationMethod::Biometric), "Biometric");
        assert_eq!(format!("{:?}", VerificationMethod::TwoFactor), "TwoFactor");
    }

    #[test]
    fn test_consent_withdrawal_config_default() {
        let cfg = ConsentWithdrawalConfig::default();
        assert!(cfg.enabled);
        assert!(cfg.immediate_effect);
        assert!(cfg.confirmation_required);
        assert!(cfg.grace_period.is_none());
        assert_eq!(cfg.methods.len(), 3);
    }

    #[test]
    fn test_withdrawal_method_variants() {
        assert_eq!(format!("{:?}", WithdrawalMethod::WebPortal), "WebPortal");
        assert_eq!(format!("{:?}", WithdrawalMethod::Email), "Email");
        assert_eq!(format!("{:?}", WithdrawalMethod::API), "API");
        assert_eq!(format!("{:?}", WithdrawalMethod::Phone), "Phone");
        assert_eq!(format!("{:?}", WithdrawalMethod::Mail), "Mail");
    }

    #[test]
    fn test_consent_renewal_config_default() {
        let cfg = ConsentRenewalConfig::default();
        assert!(cfg.enabled);
        assert!(cfg.auto_expiry);
        assert!(cfg.notice_period < cfg.renewal_period);
    }

    #[test]
    fn test_consent_mechanism_variants() {
        assert_eq!(format!("{:?}", ConsentMechanism::WebForm), "WebForm");
        assert_eq!(format!("{:?}", ConsentMechanism::EmailOptIn), "EmailOptIn");
        assert_eq!(format!("{:?}", ConsentMechanism::API), "API");
        assert_eq!(format!("{:?}", ConsentMechanism::MobileApp), "MobileApp");
    }

    #[test]
    fn test_consent_evidence_creation() {
        let evidence = ConsentEvidence {
            ip_address: Some("192.168.1.1".to_string()),
            user_agent: Some("Mozilla/5.0".to_string()),
            timestamp: std::time::SystemTime::now(),
            digital_signature: None,
            witness: None,
            metadata: std::collections::HashMap::new(),
        };
        assert!(evidence.ip_address.is_some());
        assert!(evidence.digital_signature.is_none());
    }

    #[test]
    fn test_consent_record_creation() {
        use crate::gdpr_compliance_modules::types::ConsentStatus;
        let record = ConsentRecord {
            id: "consent-001".to_string(),
            subject_id: "user-001".to_string(),
            purpose: "analytics".to_string(),
            status: ConsentStatus::Given,
            mechanism: ConsentMechanism::WebForm,
            given_at: std::time::SystemTime::now(),
            withdrawn_at: None,
            expires_at: None,
            evidence: ConsentEvidence {
                ip_address: Some("10.0.0.1".to_string()),
                user_agent: None,
                timestamp: std::time::SystemTime::now(),
                digital_signature: None,
                witness: None,
                metadata: std::collections::HashMap::new(),
            },
            version: "1.0".to_string(),
        };
        assert_eq!(record.id, "consent-001");
        assert!(record.withdrawn_at.is_none());
    }
}
