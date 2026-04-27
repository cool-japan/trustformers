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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gdpr_compliance_config_default_enabled() {
        let config = GdprComplianceConfig::default();
        assert!(config.enabled);
    }

    #[test]
    fn test_gdpr_config_all_sub_configs_enabled() {
        let config = GdprComplianceConfig::default();
        assert!(config.data_processing.enabled);
        assert!(config.consent_management.enabled);
        assert!(config.data_subject_rights.enabled);
    }

    #[test]
    fn test_legal_basis_variants_accessible() {
        let bases = [
            LegalBasis::Consent,
            LegalBasis::Contract,
            LegalBasis::LegalObligation,
            LegalBasis::VitalInterests,
            LegalBasis::PublicTask,
            LegalBasis::LegitimateInterests,
        ];
        let names: std::collections::HashSet<String> =
            bases.iter().map(|b| format!("{:?}", b)).collect();
        assert_eq!(names.len(), 6);
    }

    #[test]
    fn test_request_type_all_variants() {
        let types = [
            RequestType::Access,
            RequestType::Rectification,
            RequestType::Erasure,
            RequestType::Restriction,
            RequestType::Portability,
            RequestType::Objection,
            RequestType::AutomatedDecision,
        ];
        let names: std::collections::HashSet<String> =
            types.iter().map(|t| format!("{:?}", t)).collect();
        assert_eq!(names.len(), 7);
    }

    #[test]
    fn test_consent_mechanism_variants() {
        let mechanisms = [
            ConsentMechanism::WebForm,
            ConsentMechanism::EmailOptIn,
            ConsentMechanism::API,
            ConsentMechanism::MobileApp,
            ConsentMechanism::PhysicalForm,
            ConsentMechanism::Verbal,
            ConsentMechanism::Implied,
        ];
        for m in &mechanisms {
            let s = format!("{:?}", m);
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn test_consent_status_variants() {
        let statuses = [
            ConsentStatus::Given,
            ConsentStatus::Withdrawn,
            ConsentStatus::Expired,
            ConsentStatus::Pending,
            ConsentStatus::Refused,
        ];
        let names: std::collections::HashSet<String> =
            statuses.iter().map(|s| format!("{:?}", s)).collect();
        assert_eq!(names.len(), 5);
    }

    #[test]
    fn test_request_status_all_variants() {
        let statuses = [
            RequestStatus::Submitted,
            RequestStatus::UnderReview,
            RequestStatus::VerificationRequired,
            RequestStatus::Verified,
            RequestStatus::Processing,
            RequestStatus::Completed,
            RequestStatus::Rejected,
            RequestStatus::Cancelled,
        ];
        let names: std::collections::HashSet<String> =
            statuses.iter().map(|s| format!("{:?}", s)).collect();
        assert_eq!(names.len(), 8);
    }

    #[test]
    fn test_verification_status_variants() {
        let statuses = [
            VerificationStatus::NotVerified,
            VerificationStatus::Pending,
            VerificationStatus::Verified,
            VerificationStatus::Failed,
            VerificationStatus::Expired,
        ];
        let names: std::collections::HashSet<String> =
            statuses.iter().map(|s| format!("{:?}", s)).collect();
        assert_eq!(names.len(), 5);
    }

    #[test]
    fn test_data_category_variants() {
        let cats = [
            DataCategory::PersonalData,
            DataCategory::SpecialCategoryData,
            DataCategory::PseudonymizedData,
            DataCategory::AnonymizedData,
            DataCategory::CriminalData,
        ];
        for cat in &cats {
            let s = format!("{:?}", cat);
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn test_gdpr_compliance_error_variants() {
        let errors = [
            GdprComplianceError::ConfigurationError("cfg".to_string()),
            GdprComplianceError::ConsentError("consent".to_string()),
            GdprComplianceError::RequestError("request".to_string()),
            GdprComplianceError::ProcessingError("process".to_string()),
            GdprComplianceError::StorageError("storage".to_string()),
            GdprComplianceError::InternalError("internal".to_string()),
        ];
        for e in &errors {
            let s = format!("{:?}", e);
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn test_request_details_with_additional_info() {
        let details = RequestDetails {
            description: "Export my data".to_string(),
            data_categories: vec!["personal_data".to_string()],
            processing_activities: vec!["analytics".to_string()],
            additional_info: Some("Prefer JSON format".to_string()),
        };
        assert_eq!(details.data_categories.len(), 1);
        assert!(details.additional_info.is_some());
    }

    #[test]
    fn test_request_details_without_additional_info() {
        let details = RequestDetails {
            description: "Delete my account".to_string(),
            data_categories: vec!["all".to_string()],
            processing_activities: vec![],
            additional_info: None,
        };
        assert!(details.additional_info.is_none());
        assert!(details.processing_activities.is_empty());
    }

    #[tokio::test]
    async fn test_service_creation_succeeds() {
        let config = GdprComplianceConfig::default();
        let result = GdprComplianceService::new(config);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_record_consent_web_form() {
        let config = GdprComplianceConfig::default();
        let service = GdprComplianceService::new(config)
            .unwrap_or_else(|e| panic!("service creation failed: {}", e));
        let result = service
            .record_consent("user-mod-1", "marketing", ConsentMechanism::WebForm)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_record_consent_api_mechanism() {
        let config = GdprComplianceConfig::default();
        let service = GdprComplianceService::new(config)
            .unwrap_or_else(|e| panic!("service creation failed: {}", e));
        let result = service.record_consent("user-mod-2", "analytics", ConsentMechanism::API).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_submit_access_request() {
        let config = GdprComplianceConfig::default();
        let service = GdprComplianceService::new(config)
            .unwrap_or_else(|e| panic!("service creation failed: {}", e));
        let details = RequestDetails {
            description: "I want to see my data".to_string(),
            data_categories: vec!["personal_data".to_string()],
            processing_activities: vec!["user_profiling".to_string()],
            additional_info: None,
        };
        let result = service
            .submit_data_subject_request("user-mod-3", RequestType::Access, details)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_submit_erasure_request() {
        let config = GdprComplianceConfig::default();
        let service = GdprComplianceService::new(config)
            .unwrap_or_else(|e| panic!("service creation failed: {}", e));
        let details = RequestDetails {
            description: "Please delete all my data".to_string(),
            data_categories: vec!["all".to_string()],
            processing_activities: vec![],
            additional_info: Some("Account closure".to_string()),
        };
        let result = service
            .submit_data_subject_request("user-mod-4", RequestType::Erasure, details)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_submit_portability_request() {
        let config = GdprComplianceConfig::default();
        let service = GdprComplianceService::new(config)
            .unwrap_or_else(|e| panic!("service creation failed: {}", e));
        let details = RequestDetails {
            description: "Export my data in machine-readable format".to_string(),
            data_categories: vec!["personal_data".to_string()],
            processing_activities: vec!["profile_building".to_string()],
            additional_info: None,
        };
        let result = service
            .submit_data_subject_request("user-mod-5", RequestType::Portability, details)
            .await;
        assert!(result.is_ok());
    }
}
