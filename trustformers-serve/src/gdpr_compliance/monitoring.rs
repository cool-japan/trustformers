//! Compliance monitoring and auditing

// Re-export from manager for now
pub use super::manager::{
    ComplianceMonitoringConfig, CrossBorderTransferConfig, DataBreachManagementConfig,
    PrivacyByDesignConfig, PrivacyImpactAssessmentConfig,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compliance_monitoring_config_default() {
        let config = ComplianceMonitoringConfig::default();
        let s = format!("{:?}", config);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_data_breach_management_config_default() {
        let config = DataBreachManagementConfig::default();
        let s = format!("{:?}", config);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_cross_border_transfer_config_default() {
        let config = CrossBorderTransferConfig::default();
        let s = format!("{:?}", config);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_privacy_impact_assessment_config_default() {
        let config = PrivacyImpactAssessmentConfig::default();
        let s = format!("{:?}", config);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_privacy_by_design_config_default() {
        let config = PrivacyByDesignConfig::default();
        let s = format!("{:?}", config);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_compliance_monitoring_config_enabled() {
        let config = ComplianceMonitoringConfig::default();
        // Just ensure the enabled field is accessible
        let _ = format!("{}", config.enabled);
    }

    #[test]
    fn test_data_breach_management_notification_enabled() {
        let config = DataBreachManagementConfig::default();
        // Verify notification config is accessible
        let s = format!("{:?}", config.notification);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_cross_border_transfer_config_fields() {
        let config = CrossBorderTransferConfig::default();
        assert!(config.enabled);
        let s = format!("{:?}", config.assessment);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_privacy_impact_assessment_triggers() {
        let config = PrivacyImpactAssessmentConfig::default();
        // Verify triggers are accessible
        let s = format!("{:?}", config.triggers);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_privacy_by_design_principles() {
        let config = PrivacyByDesignConfig::default();
        let s = format!("{:?}", config.principles);
        assert!(!s.is_empty());
    }

    #[tokio::test]
    async fn test_gdpr_service_creation_with_default_config() {
        use super::super::manager::{GdprComplianceConfig, GdprComplianceService};
        let config = GdprComplianceConfig::default();
        let result = GdprComplianceService::new(config);
        assert!(result.is_ok());
    }
}
