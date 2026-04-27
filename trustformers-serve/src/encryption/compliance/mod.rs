//! Auto-generated module structure

pub mod audittrailconfig_traits;
pub mod types;
pub mod functions;

// Re-export all types
pub use audittrailconfig_traits::*;
pub use types::*;
pub use functions::*;

#[cfg(test)]
#[path = "types_tests.rs"]
mod types_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::{Duration, SystemTime};

    fn make_audit_entry(event_type: AuditEventType, actor: &str) -> AuditEntry {
        AuditEntry {
            id: format!("entry-{}", actor),
            timestamp: SystemTime::now(),
            event_type,
            actor: actor.to_string(),
            resource: "/resource".to_string(),
            action: "action".to_string(),
            outcome: AuditOutcome::Success,
            context: HashMap::new(),
            severity: AuditSeverity::Info,
            compliance_tags: vec![],
        }
    }

    #[test]
    fn test_audit_trail_config_default_values() {
        let config = AuditTrailConfig::default();
        assert!(config.max_size > 0);
        assert!(config.encryption_enabled);
        assert!(config.tamper_protection);
        assert!(config.auto_archival);
    }

    #[test]
    fn test_audit_trail_config_custom() {
        let config = AuditTrailConfig {
            max_size: 500_000,
            retention_period: Duration::from_secs(30 * 24 * 3600),
            encryption_enabled: false,
            tamper_protection: false,
            auto_archival: false,
        };
        assert_eq!(config.max_size, 500_000);
        assert!(!config.encryption_enabled);
    }

    #[test]
    fn test_audit_severity_ordering() {
        assert!(AuditSeverity::Critical > AuditSeverity::Error);
        assert!(AuditSeverity::Error > AuditSeverity::Warning);
        assert!(AuditSeverity::Warning > AuditSeverity::Info);
    }

    #[test]
    fn test_audit_severity_equality() {
        assert_eq!(AuditSeverity::Info, AuditSeverity::Info);
        assert_ne!(AuditSeverity::Info, AuditSeverity::Critical);
    }

    #[test]
    fn test_audit_event_type_equality_checks() {
        assert_eq!(AuditEventType::KeyManagement, AuditEventType::KeyManagement);
        assert_ne!(AuditEventType::KeyManagement, AuditEventType::DataAccess);
    }

    #[test]
    fn test_audit_event_type_all_variants_distinct() {
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
        let set: std::collections::HashSet<String> =
            variants.iter().map(|v| format!("{:?}", v)).collect();
        assert_eq!(set.len(), 10);
    }

    #[test]
    fn test_audit_outcome_variants() {
        let outcomes = [
            AuditOutcome::Success,
            AuditOutcome::Failure,
            AuditOutcome::PartialSuccess,
            AuditOutcome::Denied,
            AuditOutcome::UnderReview,
        ];
        for (i, o1) in outcomes.iter().enumerate() {
            for (j, o2) in outcomes.iter().enumerate() {
                if i == j {
                    assert_eq!(o1, o2);
                } else {
                    assert_ne!(o1, o2);
                }
            }
        }
    }

    #[test]
    fn test_audit_filter_no_filters_matches_all() {
        let filter = AuditFilter {
            event_types: None,
            time_range: None,
            actor: None,
            min_severity: None,
        };
        let entry = make_audit_entry(AuditEventType::DataAccess, "alice");
        assert!(filter.matches(&entry));
    }

    #[test]
    fn test_audit_filter_by_event_type_matches() {
        let filter = AuditFilter {
            event_types: Some(vec![AuditEventType::Authentication]),
            time_range: None,
            actor: None,
            min_severity: None,
        };
        let matching = make_audit_entry(AuditEventType::Authentication, "bob");
        let non_matching = make_audit_entry(AuditEventType::DataAccess, "bob");
        assert!(filter.matches(&matching));
        assert!(!filter.matches(&non_matching));
    }

    #[test]
    fn test_audit_filter_by_actor() {
        let filter = AuditFilter {
            event_types: None,
            time_range: None,
            actor: Some("alice".to_string()),
            min_severity: None,
        };
        let alice_entry = make_audit_entry(AuditEventType::DataModification, "alice");
        let bob_entry = make_audit_entry(AuditEventType::DataModification, "bob");
        assert!(filter.matches(&alice_entry));
        assert!(!filter.matches(&bob_entry));
    }

    #[test]
    fn test_audit_filter_by_severity_info_fails_warning_threshold() {
        let filter = AuditFilter {
            event_types: None,
            time_range: None,
            actor: None,
            min_severity: Some(AuditSeverity::Warning),
        };
        let info_entry = make_audit_entry(AuditEventType::SystemEvent, "sys");
        // Info < Warning
        assert!(!filter.matches(&info_entry));
    }

    #[test]
    fn test_audit_filter_by_severity_critical_passes_warning_threshold() {
        let filter = AuditFilter {
            event_types: None,
            time_range: None,
            actor: None,
            min_severity: Some(AuditSeverity::Warning),
        };
        let mut critical_entry = make_audit_entry(AuditEventType::PolicyViolation, "attacker");
        critical_entry.severity = AuditSeverity::Critical;
        assert!(filter.matches(&critical_entry));
    }

    #[test]
    fn test_requirement_status_compliant_ne_noncompliant() {
        assert_ne!(RequirementStatus::Compliant, RequirementStatus::NonCompliant);
    }

    #[test]
    fn test_requirement_result_score_in_range() {
        let r = RequirementResult {
            requirement_id: "req-test".to_string(),
            status: RequirementStatus::Compliant,
            score: 0.95,
            validation_results: vec![],
            evidence: vec!["evidence-1".to_string()],
        };
        assert!(r.score >= 0.0 && r.score <= 1.0);
        assert_eq!(r.status, RequirementStatus::Compliant);
    }

    #[test]
    fn test_compliance_recommendation_fields() {
        let rec = ComplianceRecommendation {
            id: "rec-001".to_string(),
            category: RecommendationCategory::Technical,
            description: "Upgrade to AES-256".to_string(),
            priority: RecommendationPriority::High,
            effort: ImplementationEffort::Medium,
            remediation_steps: vec!["Step 1".to_string(), "Step 2".to_string()],
        };
        assert_eq!(rec.id, "rec-001");
        assert_eq!(rec.priority, RecommendationPriority::High);
        assert_eq!(rec.remediation_steps.len(), 2);
    }

    #[test]
    fn test_recommendation_priority_ordering() {
        assert!(RecommendationPriority::Critical > RecommendationPriority::High);
        assert!(RecommendationPriority::High > RecommendationPriority::Medium);
        assert!(RecommendationPriority::Medium > RecommendationPriority::Low);
    }

    #[test]
    fn test_data_classification_confidence_valid() {
        let dc = DataClassification {
            data_id: "doc-001".to_string(),
            level: "Confidential".to_string(),
            confidence: 0.87,
            classified_at: SystemTime::now(),
            policies_applied: vec!["p1".to_string()],
            manual_override: false,
        };
        assert!(dc.confidence >= 0.0 && dc.confidence <= 1.0);
        assert!(!dc.manual_override);
    }

    #[test]
    fn test_requirement_category_all_variants() {
        let cats = vec![
            RequirementCategory::DataProtection,
            RequirementCategory::AccessControl,
            RequirementCategory::Encryption,
            RequirementCategory::Audit,
            RequirementCategory::Retention,
            RequirementCategory::BreachNotification,
            RequirementCategory::DataSubjectRights,
        ];
        assert_eq!(cats.len(), 7);
    }
}
