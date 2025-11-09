//! GDPR Compliance System
//!
//! Comprehensive GDPR compliance and data privacy management system implementing
//! data subject rights, privacy by design, consent management, and regulatory compliance.
//!
//! ## Refactoring Summary
//!
//! Previously this was a single 2,773-line file containing all GDPR compliance functionality.
//! It has been split into focused modules:
//!
//! - `gdpr_compliance_modules/types.rs` - Core GDPR types and enums (162 lines)
//! - `gdpr_compliance_modules/data_processing.rs` - Data processing and activities (149 lines)
//! - `gdpr_compliance_modules/consent_management.rs` - Consent management system (208 lines)
//! - `gdpr_compliance_modules/data_subject_rights.rs` - Data subject rights implementation (439 lines)
//! - `gdpr_compliance_modules/data_retention.rs` - Data retention policies (88 lines)
//! - `gdpr_compliance_modules/privacy_impact.rs` - Privacy impact assessments (53 lines)
//! - `gdpr_compliance_modules/breach_management.rs` - Data breach management (284 lines)
//! - `gdpr_compliance_modules/cross_border.rs` - Cross-border transfer compliance (82 lines)
//! - `gdpr_compliance_modules/compliance_monitoring.rs` - Compliance monitoring and auditing (103 lines)
//! - `gdpr_compliance_modules/privacy_by_design.rs` - Privacy by design principles (98 lines)
//! - `gdpr_compliance_modules/service.rs` - Main service implementation (371 lines)
//!
//! This refactoring improves:
//! - Code maintainability and readability
//! - Module compilation times
//! - Test isolation
//! - Code reuse through focused modules
//! - Developer experience when working on specific GDPR requirements
//! - Compliance auditing and verification

// Re-export the entire GDPR compliance modules
pub use crate::gdpr_compliance_modules::*;

// Import the GDPR compliance modules
use crate::gdpr_compliance_modules;

// Convenience exports for backwards compatibility
pub use gdpr_compliance_modules::{
    AssessmentCriterion,
    AuditScope,

    BreachAlertThresholds,
    BreachDetectionConfig,
    BreachDocumentationConfig,
    BreachNotificationConfig,
    CommunicationPlan,
    ComplianceAuditConfig,
    ComplianceCheck,
    // Compliance monitoring
    ComplianceMonitoringConfig,
    ComplianceReportingConfig,
    ComplianceStatus,
    ConsentEvidence,

    // Consent management
    ConsentManagementConfig,
    ConsentMechanism,
    ConsentRecord,
    ConsentRenewalConfig,
    ConsentStatus,
    ConsentStorageBackend,
    ConsentStorageConfig,
    ConsentVerificationConfig,
    ConsentWithdrawalConfig,
    ContactDetails,
    ControllerInfo,
    // Cross-border transfers
    CrossBorderTransferConfig,
    // Data breach management
    DataBreachManagementConfig,
    DataCategory,
    DataMinimizationConfig,
    // Data processing
    DataProcessingConfig,
    // Data retention
    DataRetentionConfig,
    DataSubject,
    DataSubjectNotification,
    DataSubjectPreferences,
    DataSubjectRequest,
    // Data subject rights
    DataSubjectRightsConfig,
    DefaultSettingsConfig,

    DeletionMethod,
    DesignPrinciple,
    DetectionMethod,
    DocumentationRequirement,

    ErasureMethod,
    EscalationLevel,
    ExportFormat,
    // Core types and configuration
    GdprComplianceConfig,
    GdprComplianceError,

    // Main service
    GdprComplianceService,
    GdprComplianceStats,
    GdprPrometheusMetrics,
    HighRiskCriteria,
    IncidentResponseConfig,
    InternationalTransfer,
    LegalBasis,
    LegalHoldConfig,
    MinimizationStrategy,

    NotificationChannel,
    ObjectionGround,
    OverrideCondition,
    PIAFramework,

    PIATrigger,
    PortableFormat,
    // Privacy by design
    PrivacyByDesignConfig,
    PrivacyControl,
    PrivacyEngineeringConfig,
    // Privacy impact assessment
    PrivacyImpactAssessmentConfig,
    PrivacyPattern,
    ProcessingActivity,
    ProcessingPurpose,
    RecipientInfo,
    RecipientType,
    ReportType,
    RequestDetails,

    RequestHandlingConfig,
    RequestNotificationConfig,
    RequestProcessingResult,
    RequestStatus,
    RequestTimeframesConfig,
    RequestType,
    RequestVerificationConfig,
    RestrictionMethod,
    RetentionException,
    RetentionPolicy,
    RetentionReviewConfig,

    RightOfAccessConfig,
    RightToErasureConfig,
    RightToObjectConfig,
    RightToPortabilityConfig,
    RightToRectificationConfig,
    RightToRestrictConfig,
    SafeguardsConfig,
    SupervisoryAuthorityNotification,
    ThirdPartyNotification,
    ThirdPartyRecipient,
    TransferAssessmentConfig,
    TransferDocumentationConfig,

    TransferMechanism,
    VerificationMethod,
    VerificationStatus,
    WithdrawalMethod,
};

// Re-export tests for compatibility
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gdpr_compliance_service_creation() {
        let config = GdprComplianceConfig::default();
        let service = GdprComplianceService::new(config);
        assert!(service.is_ok());
    }

    #[tokio::test]
    async fn test_consent_recording() {
        let config = GdprComplianceConfig::default();
        let service = GdprComplianceService::new(config).unwrap();

        let result = service
            .record_consent("test_subject", "marketing", ConsentMechanism::WebForm)
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_data_subject_request_submission() {
        let config = GdprComplianceConfig::default();
        let service = GdprComplianceService::new(config).unwrap();

        let details = RequestDetails {
            description: "Access request".to_string(),
            data_categories: vec!["personal_data".to_string()],
            processing_activities: vec!["user_analytics".to_string()],
            additional_info: None,
        };

        let result = service
            .submit_data_subject_request("test_subject", RequestType::Access, details)
            .await;

        assert!(result.is_ok());
    }

    #[test]
    fn test_gdpr_config_defaults() {
        let config = GdprComplianceConfig::default();
        assert!(config.enabled);
        assert!(config.data_processing.enabled);
        assert!(config.consent_management.enabled);
        assert!(config.data_subject_rights.enabled);
    }
}
