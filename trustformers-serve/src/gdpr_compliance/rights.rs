//! Data subject rights management

// Re-export from manager for now
pub use super::manager::DataSubjectRightsConfig;

// Re-export additional rights types for testing convenience
pub use super::manager::{
    RequestHandlingConfig, RequestStatus, RequestType, RightOfAccessConfig, RightToErasureConfig,
    RightToObjectConfig, RightToPortabilityConfig, RightToRectificationConfig,
    RightToRestrictConfig,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_subject_rights_config_default_enabled() {
        let config = DataSubjectRightsConfig::default();
        assert!(config.enabled);
    }

    #[test]
    fn test_right_of_access_config_default() {
        let config = RightOfAccessConfig::default();
        assert!(config.enabled);
        assert!(config.max_requests_per_period > 0);
        assert!(!config.export_formats.is_empty());
    }

    #[test]
    fn test_right_to_erasure_config_default() {
        let _config = RightToErasureConfig::default(); // just verify it can be created
    }

    #[test]
    fn test_right_to_portability_config_default() {
        let _config = RightToPortabilityConfig::default(); // just verify it can be created
    }

    #[test]
    fn test_right_to_object_config_default() {
        let _config = RightToObjectConfig::default(); // just verify it can be created
    }

    #[test]
    fn test_request_type_access_variant() {
        let rt = RequestType::Access;
        let s = format!("{:?}", rt);
        assert!(s.contains("Access"));
    }

    #[test]
    fn test_request_type_erasure_variant() {
        let rt = RequestType::Erasure;
        let s = format!("{:?}", rt);
        assert!(s.contains("Erasure"));
    }

    #[test]
    fn test_request_status_submitted_variant() {
        let rs = RequestStatus::Submitted;
        let s = format!("{:?}", rs);
        assert!(s.contains("Submitted"));
    }

    #[test]
    fn test_request_handling_config_default() {
        let config = RequestHandlingConfig::default();
        let s = format!("{:?}", config);
        assert!(!s.is_empty());
    }

    #[tokio::test]
    async fn test_submit_access_request_via_service() {
        use super::super::manager::{GdprComplianceConfig, GdprComplianceService, RequestDetails};
        let config = GdprComplianceConfig::default();
        let service = GdprComplianceService::new(config)
            .unwrap_or_else(|e| panic!("service creation failed: {}", e));
        let details = RequestDetails {
            description: "I want access to my data".to_string(),
            data_categories: vec!["personal_data".to_string()],
            processing_activities: vec!["user_profiling".to_string()],
            additional_info: None,
        };
        let result = service
            .submit_data_subject_request("subject-1", RequestType::Access, details)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_submit_erasure_request_via_service() {
        use super::super::manager::{GdprComplianceConfig, GdprComplianceService, RequestDetails};
        let config = GdprComplianceConfig::default();
        let service = GdprComplianceService::new(config)
            .unwrap_or_else(|e| panic!("service creation failed: {}", e));
        let details = RequestDetails {
            description: "Please delete my account data".to_string(),
            data_categories: vec!["all".to_string()],
            processing_activities: vec![],
            additional_info: Some("Account closure".to_string()),
        };
        let result = service
            .submit_data_subject_request("subject-2", RequestType::Erasure, details)
            .await;
        assert!(result.is_ok());
    }
}
