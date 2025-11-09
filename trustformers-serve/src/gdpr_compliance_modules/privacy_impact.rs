//! Privacy Impact Assessment (PIA) implementation
//!
//! This module implements Privacy Impact Assessment capabilities as required
//! by GDPR Article 35 for high-risk processing activities.

use serde::{Deserialize, Serialize};

/// Privacy Impact Assessment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyImpactAssessmentConfig {
    /// Enable PIA
    pub enabled: bool,
    /// PIA triggers
    pub triggers: Vec<PIATrigger>,
    /// PIA framework
    pub framework: PIAFramework,
    /// Automatic assessment
    pub auto_assessment: bool,
}

impl Default for PrivacyImpactAssessmentConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            triggers: vec![
                PIATrigger::LargeScale,
                PIATrigger::SpecialCategory,
                PIATrigger::AutomatedDecision,
            ],
            framework: PIAFramework::Standard,
            auto_assessment: false,
        }
    }
}

/// PIA triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIATrigger {
    /// Large scale processing
    LargeScale,
    /// Special category data
    SpecialCategory,
    /// Automated decision making
    AutomatedDecision,
    /// Systematic monitoring
    SystematicMonitoring,
    /// Vulnerable data subjects
    VulnerableSubjects,
}

/// PIA frameworks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIAFramework {
    /// Standard PIA framework
    Standard,
    /// ISO 27001 based
    ISO27001,
    /// NIST based
    NIST,
    /// Custom framework
    Custom,
}
