//! Compliance and audit types

// Re-export from security module for convenience
pub use super::security::{
    AccessRestriction, AccessRestrictionType, AlertEscalationConfig, AlertTemplate,
    AuditArchiveConfig, AuditConfig, AuditNotificationConfig, AuditNotificationRule,
    AuditRetentionConfig, AuditRetentionPolicy, AuditStorageBackend, AuditStorageConfig,
    AuditStorageFormat, AuditTarget, ComplianceAction, ComplianceAlertConfig, ComplianceCheck,
    ComplianceCheckType, ComplianceConfig, ComplianceDashboardConfig, ComplianceEvidence,
    ComplianceFramework, ComplianceMonitoringConfig, ComplianceMonitoringRule,
    ComplianceReportDistribution, ComplianceReportGeneration, ComplianceReportTemplate,
    ComplianceReportingConfig, ComplianceRequirement, ComplianceStatus, ComplianceWidget,
    ComplianceWidgetType, DashboardAccessControl, EscalationLevel, EvidenceType, WidgetPosition,
};

// Additional compliance-specific types can be added here as needed
