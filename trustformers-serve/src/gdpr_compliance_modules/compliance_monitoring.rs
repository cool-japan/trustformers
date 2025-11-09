//! Compliance monitoring and auditing
//!
//! This module implements compliance monitoring, auditing, and reporting
//! capabilities for ongoing GDPR compliance verification.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Compliance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceMonitoringConfig {
    /// Enable monitoring
    pub enabled: bool,
    /// Monitoring frequency
    pub monitoring_frequency: Duration,
    /// Compliance checks
    pub compliance_checks: Vec<ComplianceCheck>,
    /// Reporting
    pub reporting: ComplianceReportingConfig,
    /// Auditing
    pub auditing: ComplianceAuditConfig,
}

impl Default for ComplianceMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            monitoring_frequency: Duration::from_secs(24 * 3600), // Daily
            compliance_checks: vec![
                ComplianceCheck::ConsentValidity,
                ComplianceCheck::DataRetention,
                ComplianceCheck::SecurityMeasures,
            ],
            reporting: ComplianceReportingConfig::default(),
            auditing: ComplianceAuditConfig::default(),
        }
    }
}

/// Compliance checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceCheck {
    /// Consent validity
    ConsentValidity,
    /// Data retention compliance
    DataRetention,
    /// Security measures
    SecurityMeasures,
    /// Data subject rights handling
    DataSubjectRights,
    /// Breach response
    BreachResponse,
}

/// Compliance reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReportingConfig {
    /// Enable reporting
    pub enabled: bool,
    /// Report types
    pub report_types: Vec<ReportType>,
    /// Reporting frequency
    pub frequency: Duration,
}

impl Default for ComplianceReportingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            report_types: vec![ReportType::Summary, ReportType::Detailed],
            frequency: Duration::from_secs(30 * 24 * 3600), // Monthly
        }
    }
}

/// Report types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportType {
    /// Summary report
    Summary,
    /// Detailed report
    Detailed,
    /// Executive report
    Executive,
    /// Technical report
    Technical,
}

/// Compliance audit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceAuditConfig {
    /// Enable auditing
    pub enabled: bool,
    /// Audit frequency
    pub frequency: Duration,
    /// Audit scope
    pub scope: Vec<AuditScope>,
}

impl Default for ComplianceAuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            frequency: Duration::from_secs(90 * 24 * 3600), // Quarterly
            scope: vec![
                AuditScope::DataProcessing,
                AuditScope::ConsentManagement,
                AuditScope::SecurityControls,
            ],
        }
    }
}

/// Audit scope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditScope {
    /// Data processing activities
    DataProcessing,
    /// Consent management
    ConsentManagement,
    /// Security controls
    SecurityControls,
    /// Data subject rights
    DataSubjectRights,
    /// Cross-border transfers
    CrossBorderTransfers,
}
