//! Tests for encryption/compliance types

#[cfg(test)]
mod tests {
    use super::super::types::*;
    use std::collections::{HashMap, HashSet};
    use std::time::{Duration, SystemTime};

    // ===== AuditEventType tests =====

    #[test]
    fn test_audit_event_type_equality() {
        let a = AuditEventType::Authentication;
        let b = AuditEventType::Authentication;
        assert_eq!(a, b);
        let c = AuditEventType::Authorization;
        assert_ne!(a, c);
    }

    #[test]
    fn test_audit_event_type_all_variants() {
        let variants = vec![
            AuditEventType::Authentication,
            AuditEventType::Authorization,
            AuditEventType::DataAccess,
            AuditEventType::DataModification,
            AuditEventType::EncryptionOperation,
            AuditEventType::KeyManagement,
            AuditEventType::ConfigurationChange,
            AuditEventType::SystemEvent,
            AuditEventType::ComplianceCheck,
            AuditEventType::PolicyViolation,
        ];
        assert_eq!(variants.len(), 10);
        // All variants are distinct
        let set: std::collections::HashSet<String> =
            variants.iter().map(|v| format!("{:?}", v)).collect();
        assert_eq!(set.len(), 10, "All AuditEventType variants must be distinct");
    }

    // ===== AuditOutcome tests =====

    #[test]
    fn test_audit_outcome_success_ne_failure() {
        let success = AuditOutcome::Success;
        let failure = AuditOutcome::Failure;
        assert_ne!(success, failure);
    }

    #[test]
    fn test_audit_outcome_variants() {
        let outcomes = vec![
            AuditOutcome::Success,
            AuditOutcome::Failure,
            AuditOutcome::PartialSuccess,
            AuditOutcome::Denied,
            AuditOutcome::UnderReview,
        ];
        assert_eq!(outcomes.len(), 5);
    }

    // ===== AuditSeverity tests =====

    #[test]
    fn test_audit_severity_ordering() {
        // Critical > Error > Warning > Info
        assert!(AuditSeverity::Critical > AuditSeverity::Error);
        assert!(AuditSeverity::Error > AuditSeverity::Warning);
        assert!(AuditSeverity::Warning > AuditSeverity::Info);
    }

    #[test]
    fn test_audit_severity_equality() {
        assert_eq!(AuditSeverity::Info, AuditSeverity::Info);
        assert_ne!(AuditSeverity::Info, AuditSeverity::Critical);
    }

    // ===== RequirementStatus tests =====

    #[test]
    fn test_requirement_status_variants() {
        let statuses = vec![
            RequirementStatus::Compliant,
            RequirementStatus::PartiallyCompliant,
            RequirementStatus::NonCompliant,
            RequirementStatus::NotApplicable,
            RequirementStatus::UnderReview,
        ];
        assert_eq!(statuses.len(), 5);
    }

    #[test]
    fn test_requirement_status_compliant_ne_noncompliant() {
        assert_ne!(RequirementStatus::Compliant, RequirementStatus::NonCompliant);
    }

    // ===== RequirementCategory tests =====

    #[test]
    fn test_requirement_category_all_variants() {
        let categories = vec![
            RequirementCategory::DataProtection,
            RequirementCategory::AccessControl,
            RequirementCategory::Encryption,
            RequirementCategory::Audit,
            RequirementCategory::Retention,
            RequirementCategory::BreachNotification,
            RequirementCategory::DataSubjectRights,
        ];
        assert_eq!(categories.len(), 7);
    }

    // ===== RequirementSeverity tests =====

    #[test]
    fn test_requirement_severity_ordering() {
        // Critical > High > Medium > Low > Info
        assert!(RequirementSeverity::Critical > RequirementSeverity::High);
        assert!(RequirementSeverity::High > RequirementSeverity::Medium);
        assert!(RequirementSeverity::Medium > RequirementSeverity::Low);
        assert!(RequirementSeverity::Low > RequirementSeverity::Info);
    }

    // ===== RecommendationPriority tests =====

    #[test]
    fn test_recommendation_priority_ordering() {
        assert!(RecommendationPriority::Critical > RecommendationPriority::High);
        assert!(RecommendationPriority::High > RecommendationPriority::Medium);
        assert!(RecommendationPriority::Medium > RecommendationPriority::Low);
    }

    // ===== ImplementationEffort tests =====

    #[test]
    fn test_implementation_effort_variants() {
        let efforts = vec![
            ImplementationEffort::Low,
            ImplementationEffort::Medium,
            ImplementationEffort::High,
            ImplementationEffort::VeryHigh,
        ];
        assert_eq!(efforts.len(), 4);
    }

    #[test]
    fn test_implementation_effort_equality() {
        assert_eq!(ImplementationEffort::Low, ImplementationEffort::Low);
        assert_ne!(ImplementationEffort::Low, ImplementationEffort::VeryHigh);
    }

    // ===== ComparisonOperator tests =====

    #[test]
    fn test_comparison_operator_all_variants() {
        let ops = vec![
            ComparisonOperator::Equal,
            ComparisonOperator::GreaterThan,
            ComparisonOperator::LessThan,
            ComparisonOperator::GreaterThanOrEqual,
            ComparisonOperator::LessThanOrEqual,
        ];
        assert_eq!(ops.len(), 5);
    }

    #[test]
    fn test_comparison_operator_equal_distinct_from_greater_than() {
        assert_ne!(ComparisonOperator::Equal, ComparisonOperator::GreaterThan);
    }

    // ===== DataClassification tests =====

    #[test]
    fn test_data_classification_confidence_range() {
        let classification = DataClassification {
            data_id: "doc-1234".to_string(),
            level: "Confidential".to_string(),
            confidence: 0.92,
            classified_at: SystemTime::now(),
            policies_applied: vec!["policy-01".to_string()],
            manual_override: false,
        };
        assert!(
            classification.confidence >= 0.0 && classification.confidence <= 1.0,
            "Confidence must be in [0,1], got {}",
            classification.confidence
        );
        assert!(!classification.data_id.is_empty());
    }

    #[test]
    fn test_data_classification_manual_override_flag() {
        let auto = DataClassification {
            data_id: "doc-auto".to_string(),
            level: "Public".to_string(),
            confidence: 0.85,
            classified_at: SystemTime::now(),
            policies_applied: Vec::new(),
            manual_override: false,
        };
        let manual = DataClassification {
            data_id: "doc-manual".to_string(),
            level: "TopSecret".to_string(),
            confidence: 1.0,
            classified_at: SystemTime::now(),
            policies_applied: Vec::new(),
            manual_override: true,
        };
        assert!(!auto.manual_override);
        assert!(manual.manual_override);
    }

    // ===== ComplianceAssessment tests =====

    #[test]
    fn test_compliance_assessment_score_range() {
        use crate::encryption::types::ComplianceStandard;
        let assessment = ComplianceAssessment {
            id: "assessment-001".to_string(),
            standard: ComplianceStandard::GDPR,
            assessed_at: SystemTime::now(),
            compliance_score: 0.87,
            requirement_results: Vec::new(),
            recommendations: Vec::new(),
            next_assessment: SystemTime::now(),
        };
        assert!(
            assessment.compliance_score >= 0.0 && assessment.compliance_score <= 1.0,
            "Compliance score must be in [0,1]"
        );
    }

    // ===== RequirementResult tests =====

    #[test]
    fn test_requirement_result_score_range() {
        let result = RequirementResult {
            requirement_id: "req-001".to_string(),
            status: RequirementStatus::Compliant,
            score: 1.0,
            validation_results: Vec::new(),
            evidence: Vec::new(),
        };
        assert!(result.score >= 0.0 && result.score <= 1.0);
        assert_eq!(result.status, RequirementStatus::Compliant);
    }

    #[test]
    fn test_requirement_result_partial_compliance_score() {
        let result = RequirementResult {
            requirement_id: "req-002".to_string(),
            status: RequirementStatus::PartiallyCompliant,
            score: 0.6,
            validation_results: Vec::new(),
            evidence: Vec::new(),
        };
        assert!(result.score >= 0.0 && result.score <= 1.0);
        assert_eq!(result.status, RequirementStatus::PartiallyCompliant);
    }

    // ===== AuditFilter tests =====

    #[test]
    fn test_audit_filter_no_filters_matches_all() {
        let filter = AuditFilter {
            event_types: None,
            time_range: None,
            actor: None,
            min_severity: None,
        };
        let entry = AuditEntry {
            id: "entry-001".to_string(),
            timestamp: SystemTime::now(),
            event_type: AuditEventType::DataAccess,
            actor: "user-alice".to_string(),
            resource: "/data/sensitive.csv".to_string(),
            action: "read".to_string(),
            outcome: AuditOutcome::Success,
            context: HashMap::new(),
            severity: AuditSeverity::Info,
            compliance_tags: Vec::new(),
        };
        assert!(filter.matches(&entry), "Filter with no conditions should match all entries");
    }

    #[test]
    fn test_audit_filter_by_event_type() {
        let filter = AuditFilter {
            event_types: Some(vec![AuditEventType::Authentication]),
            time_range: None,
            actor: None,
            min_severity: None,
        };
        let matching_entry = AuditEntry {
            id: "entry-002".to_string(),
            timestamp: SystemTime::now(),
            event_type: AuditEventType::Authentication,
            actor: "user-bob".to_string(),
            resource: "/auth/login".to_string(),
            action: "login".to_string(),
            outcome: AuditOutcome::Success,
            context: HashMap::new(),
            severity: AuditSeverity::Info,
            compliance_tags: Vec::new(),
        };
        let non_matching_entry = AuditEntry {
            id: "entry-003".to_string(),
            timestamp: SystemTime::now(),
            event_type: AuditEventType::DataAccess,
            actor: "user-bob".to_string(),
            resource: "/data/file.txt".to_string(),
            action: "read".to_string(),
            outcome: AuditOutcome::Success,
            context: HashMap::new(),
            severity: AuditSeverity::Info,
            compliance_tags: Vec::new(),
        };
        assert!(filter.matches(&matching_entry));
        assert!(!filter.matches(&non_matching_entry));
    }

    #[test]
    fn test_audit_filter_by_actor() {
        let filter = AuditFilter {
            event_types: None,
            time_range: None,
            actor: Some("user-alice".to_string()),
            min_severity: None,
        };
        let alice_entry = AuditEntry {
            id: "entry-004".to_string(),
            timestamp: SystemTime::now(),
            event_type: AuditEventType::DataModification,
            actor: "user-alice".to_string(),
            resource: "/data/record.db".to_string(),
            action: "update".to_string(),
            outcome: AuditOutcome::Success,
            context: HashMap::new(),
            severity: AuditSeverity::Warning,
            compliance_tags: Vec::new(),
        };
        let bob_entry = AuditEntry {
            id: "entry-005".to_string(),
            timestamp: SystemTime::now(),
            event_type: AuditEventType::DataModification,
            actor: "user-bob".to_string(),
            resource: "/data/record.db".to_string(),
            action: "update".to_string(),
            outcome: AuditOutcome::Success,
            context: HashMap::new(),
            severity: AuditSeverity::Warning,
            compliance_tags: Vec::new(),
        };
        assert!(filter.matches(&alice_entry));
        assert!(!filter.matches(&bob_entry));
    }

    // ===== PolicyAction tests =====

    #[test]
    fn test_policy_action_require_encryption() {
        let action = PolicyAction::RequireEncryption {
            algorithm: "AES256GCM".to_string(),
        };
        if let PolicyAction::RequireEncryption { algorithm } = action {
            assert!(algorithm.contains("AES"));
        } else {
            panic!("Expected RequireEncryption variant");
        }
    }

    #[test]
    fn test_policy_action_all_variants_present() {
        let actions = vec![
            PolicyAction::LogAccess,
            PolicyAction::BlockAction,
            PolicyAction::RequireEncryption { algorithm: "AES256GCM".to_string() },
            PolicyAction::RestrictAccess { allowed_roles: vec!["admin".to_string()] },
            PolicyAction::NotifyAdmin { message: "high severity event".to_string() },
        ];
        assert_eq!(actions.len(), 5);
    }

    // ===== AuditTrailConfig tests =====

    #[test]
    fn test_audit_trail_config_creation() {
        let config = AuditTrailConfig {
            max_size: 10000,
            retention_period: Duration::from_secs(90 * 24 * 3600),
            encryption_enabled: true,
            tamper_protection: true,
            auto_archival: false,
        };
        assert!(config.max_size > 0);
        assert!(config.retention_period.as_secs() > 0);
        assert!(config.encryption_enabled);
        assert!(config.tamper_protection);
        assert!(!config.auto_archival);
    }
}
