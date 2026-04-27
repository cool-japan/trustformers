//! Consent management types and functions

// Re-export from manager for now
pub use super::manager::{
    ConsentManagementConfig, ConsentRenewalConfig, ConsentStorageBackend, ConsentStorageConfig,
    ConsentVerificationConfig, ConsentWithdrawalConfig, VerificationMethod, WithdrawalMethod,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consent_management_config_default_enabled() {
        let config = ConsentManagementConfig::default();
        assert!(config.enabled);
        assert!(config.granular_consent);
    }

    #[test]
    fn test_consent_storage_config_default_audit_trail() {
        let config = ConsentStorageConfig::default();
        assert!(config.audit_trail);
        assert!(config.encryption);
        assert!(config.tamper_protection);
    }

    #[test]
    fn test_consent_storage_backend_database_variant() {
        let backend = ConsentStorageBackend::Database;
        let s = format!("{:?}", backend);
        assert!(s.contains("Database"));
    }

    #[test]
    fn test_consent_storage_backend_filesystem_variant() {
        let backend = ConsentStorageBackend::FileSystem {
            path: "/tmp/consent".to_string(),
        };
        if let ConsentStorageBackend::FileSystem { path } = backend {
            assert_eq!(path, "/tmp/consent");
        } else {
            panic!("expected FileSystem variant");
        }
    }

    #[test]
    fn test_consent_verification_config_default_methods() {
        let config = ConsentVerificationConfig::default();
        assert!(config.enabled);
        assert!(!config.methods.is_empty());
    }

    #[test]
    fn test_verification_method_variants() {
        let methods = [VerificationMethod::Email, VerificationMethod::SMS];
        for m in &methods {
            let s = format!("{:?}", m);
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn test_consent_withdrawal_config_default() {
        let config = ConsentWithdrawalConfig::default();
        assert!(config.enabled);
    }

    #[test]
    fn test_withdrawal_method_variants() {
        let s = format!("{:?}", WithdrawalMethod::Email);
        assert!(s.contains("Email"));
    }

    #[test]
    fn test_consent_renewal_config_default() {
        let _config = ConsentRenewalConfig::default(); // just verify it can be created
    }

    #[test]
    fn test_consent_management_config_storage_backend() {
        let config = ConsentManagementConfig::default();
        let s = format!("{:?}", config.storage.backend);
        assert!(!s.is_empty());
    }

    #[tokio::test]
    async fn test_consent_recording_via_service() {
        use super::super::manager::{
            ConsentMechanism, GdprComplianceConfig, GdprComplianceService,
        };
        let config = GdprComplianceConfig::default();
        let service = GdprComplianceService::new(config)
            .unwrap_or_else(|e| panic!("service creation failed: {}", e));
        let result = service.record_consent("user-1", "analytics", ConsentMechanism::WebForm).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_consent_recording_multiple_purposes() {
        use super::super::manager::{
            ConsentMechanism, GdprComplianceConfig, GdprComplianceService,
        };
        let config = GdprComplianceConfig::default();
        let service = GdprComplianceService::new(config)
            .unwrap_or_else(|e| panic!("service creation failed: {}", e));
        for purpose in &["analytics", "marketing", "personalization"] {
            let result = service.record_consent("user-2", purpose, ConsentMechanism::WebForm).await;
            assert!(result.is_ok(), "consent for {} should succeed", purpose);
        }
    }
}
