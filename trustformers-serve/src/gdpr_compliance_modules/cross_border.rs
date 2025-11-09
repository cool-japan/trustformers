//! Cross-border transfer management
//!
//! This module implements cross-border data transfer compliance as required
//! by GDPR Chapter V (Articles 44-49).

use serde::{Deserialize, Serialize};

/// Cross-border transfer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossBorderTransferConfig {
    /// Enable transfer monitoring
    pub enabled: bool,
    /// Transfer assessment
    pub assessment: TransferAssessmentConfig,
    /// Safeguards
    pub safeguards: SafeguardsConfig,
    /// Documentation
    pub documentation: TransferDocumentationConfig,
}

impl Default for CrossBorderTransferConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            assessment: TransferAssessmentConfig::default(),
            safeguards: SafeguardsConfig::default(),
            documentation: TransferDocumentationConfig::default(),
        }
    }
}

/// Transfer assessment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferAssessmentConfig {
    /// Enable assessment
    pub enabled: bool,
    /// Assessment criteria
    pub criteria: Vec<AssessmentCriterion>,
    /// Automatic assessment
    pub auto_assessment: bool,
}

impl Default for TransferAssessmentConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            criteria: vec![
                AssessmentCriterion::AdequacyDecision,
                AssessmentCriterion::Safeguards,
                AssessmentCriterion::DataSubjectRights,
            ],
            auto_assessment: false,
        }
    }
}

/// Assessment criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssessmentCriterion {
    /// Adequacy decision
    AdequacyDecision,
    /// Appropriate safeguards
    Safeguards,
    /// Data subject rights
    DataSubjectRights,
    /// Legal protections
    LegalProtections,
}

/// Safeguards configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeguardsConfig {
    /// Standard contractual clauses
    pub standard_clauses: bool,
    /// Binding corporate rules
    pub binding_rules: bool,
    /// Certification schemes
    pub certifications: bool,
}

impl Default for SafeguardsConfig {
    fn default() -> Self {
        Self {
            standard_clauses: true,
            binding_rules: false,
            certifications: false,
        }
    }
}

/// Transfer documentation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferDocumentationConfig {
    /// Enable documentation
    pub enabled: bool,
    /// Documentation requirements
    pub requirements: Vec<String>,
}

impl Default for TransferDocumentationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            requirements: vec![
                "transfer_purpose".to_string(),
                "legal_basis".to_string(),
                "safeguards".to_string(),
            ],
        }
    }
}
