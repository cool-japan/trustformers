//! Data breach management and notification
//!
//! This module implements data breach detection, notification, and management
//! capabilities as required by GDPR Articles 33 and 34.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Data breach management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataBreachManagementConfig {
    /// Enable breach management
    pub enabled: bool,
    /// Breach detection
    pub detection: BreachDetectionConfig,
    /// Incident response
    pub incident_response: IncidentResponseConfig,
    /// Breach notification
    pub notification: BreachNotificationConfig,
    /// Documentation requirements
    pub documentation: BreachDocumentationConfig,
}

impl Default for DataBreachManagementConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            detection: BreachDetectionConfig::default(),
            incident_response: IncidentResponseConfig::default(),
            notification: BreachNotificationConfig::default(),
            documentation: BreachDocumentationConfig::default(),
        }
    }
}

/// Breach detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreachDetectionConfig {
    /// Enable automated detection
    pub enabled: bool,
    /// Detection methods
    pub methods: Vec<DetectionMethod>,
    /// Alert thresholds
    pub thresholds: BreachAlertThresholds,
}

impl Default for BreachDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            methods: vec![
                DetectionMethod::AnomalyDetection,
                DetectionMethod::AccessPatternAnalysis,
            ],
            thresholds: BreachAlertThresholds::default(),
        }
    }
}

/// Detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DetectionMethod {
    /// Anomaly detection
    AnomalyDetection,
    /// Access pattern analysis
    AccessPatternAnalysis,
    /// Data integrity monitoring
    IntegrityMonitoring,
    /// External threat detection
    ThreatDetection,
}

/// Breach alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreachAlertThresholds {
    /// Maximum failed login attempts
    pub max_failed_logins: u32,
    /// Unusual access volume threshold
    pub unusual_access_threshold: u32,
    /// Data export size threshold
    pub export_size_threshold: u64,
}

impl Default for BreachAlertThresholds {
    fn default() -> Self {
        Self {
            max_failed_logins: 5,
            unusual_access_threshold: 100,
            export_size_threshold: 1_000_000, // 1MB
        }
    }
}

/// Incident response configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncidentResponseConfig {
    /// Enable incident response
    pub enabled: bool,
    /// Response team contacts
    pub response_team: Vec<String>,
    /// Escalation levels
    pub escalation_levels: Vec<EscalationLevel>,
    /// Communication plan
    pub communication_plan: CommunicationPlan,
}

impl Default for IncidentResponseConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            response_team: Vec::new(),
            escalation_levels: Vec::new(),
            communication_plan: CommunicationPlan::default(),
        }
    }
}

/// Escalation level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationLevel {
    /// Level identifier
    pub level: u32,
    /// Level description
    pub description: String,
    /// Contact information
    pub contacts: Vec<String>,
}

/// Communication plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationPlan {
    /// Internal communication channels
    pub internal_channels: Vec<String>,
    /// External communication channels
    pub external_channels: Vec<String>,
    /// Communication templates
    pub templates: Vec<String>,
}

impl Default for CommunicationPlan {
    fn default() -> Self {
        Self {
            internal_channels: vec!["email".to_string(), "slack".to_string()],
            external_channels: vec!["email".to_string(), "website".to_string()],
            templates: Vec::new(),
        }
    }
}

/// Breach notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreachNotificationConfig {
    /// Supervisory authority notification
    pub supervisory_authority: SupervisoryAuthorityNotification,
    /// Data subject notification
    pub data_subject: DataSubjectNotification,
    /// Third party notification
    pub third_party: ThirdPartyNotification,
}

impl Default for BreachNotificationConfig {
    fn default() -> Self {
        Self {
            supervisory_authority: SupervisoryAuthorityNotification::default(),
            data_subject: DataSubjectNotification::default(),
            third_party: ThirdPartyNotification::default(),
        }
    }
}

/// Supervisory authority notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupervisoryAuthorityNotification {
    /// Enable notification
    pub enabled: bool,
    /// Notification timeframe
    pub timeframe: Duration,
    /// Authority contact
    pub contact: String,
}

impl Default for SupervisoryAuthorityNotification {
    fn default() -> Self {
        Self {
            enabled: true,
            timeframe: Duration::from_secs(72 * 3600), // 72 hours
            contact: "dpa@authority.eu".to_string(),
        }
    }
}

/// Data subject notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSubjectNotification {
    /// Enable notification
    pub enabled: bool,
    /// High risk criteria
    pub high_risk_criteria: HighRiskCriteria,
    /// Notification methods
    pub methods: Vec<String>,
}

impl Default for DataSubjectNotification {
    fn default() -> Self {
        Self {
            enabled: true,
            high_risk_criteria: HighRiskCriteria::default(),
            methods: vec!["email".to_string()],
        }
    }
}

/// High risk criteria for data subject notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighRiskCriteria {
    /// Identity theft risk
    pub identity_theft_risk: bool,
    /// Financial loss risk
    pub financial_loss_risk: bool,
    /// Physical harm risk
    pub physical_harm_risk: bool,
    /// Discrimination risk
    pub discrimination_risk: bool,
}

impl Default for HighRiskCriteria {
    fn default() -> Self {
        Self {
            identity_theft_risk: true,
            financial_loss_risk: true,
            physical_harm_risk: true,
            discrimination_risk: true,
        }
    }
}

/// Third party notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThirdPartyNotification {
    /// Enable notification
    pub enabled: bool,
    /// Recipients
    pub recipients: Vec<ThirdPartyRecipient>,
}

impl Default for ThirdPartyNotification {
    fn default() -> Self {
        Self {
            enabled: false,
            recipients: Vec::new(),
        }
    }
}

/// Third party recipient
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThirdPartyRecipient {
    /// Recipient name
    pub name: String,
    /// Contact information
    pub contact: String,
    /// Notification conditions
    pub conditions: Vec<String>,
}

/// Breach documentation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreachDocumentationConfig {
    /// Enable documentation
    pub enabled: bool,
    /// Documentation requirements
    pub requirements: Vec<DocumentationRequirement>,
    /// Retention period
    pub retention_period: Duration,
}

impl Default for BreachDocumentationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            requirements: vec![
                DocumentationRequirement::IncidentDetails,
                DocumentationRequirement::ImpactAssessment,
                DocumentationRequirement::ResponseActions,
            ],
            retention_period: Duration::from_secs(5 * 365 * 24 * 3600), // 5 years
        }
    }
}

/// Documentation requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentationRequirement {
    /// Incident details
    IncidentDetails,
    /// Impact assessment
    ImpactAssessment,
    /// Response actions
    ResponseActions,
    /// Lessons learned
    LessonsLearned,
}
