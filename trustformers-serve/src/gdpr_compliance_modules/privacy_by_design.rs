//! Privacy by Design implementation
//!
//! This module implements Privacy by Design principles as required by GDPR
//! Article 25 - Data protection by design and by default.

use serde::{Deserialize, Serialize};

/// Privacy by Design configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyByDesignConfig {
    /// Enable privacy by design
    pub enabled: bool,
    /// Design principles
    pub principles: Vec<DesignPrinciple>,
    /// Privacy engineering
    pub engineering: PrivacyEngineeringConfig,
    /// Default settings
    pub default_settings: DefaultSettingsConfig,
}

impl Default for PrivacyByDesignConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            principles: vec![
                DesignPrinciple::DataMinimization,
                DesignPrinciple::PurposeLimitation,
                DesignPrinciple::TransparencyControl,
            ],
            engineering: PrivacyEngineeringConfig::default(),
            default_settings: DefaultSettingsConfig::default(),
        }
    }
}

/// Design principles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DesignPrinciple {
    /// Data minimization
    DataMinimization,
    /// Purpose limitation
    PurposeLimitation,
    /// Transparency and control
    TransparencyControl,
    /// Security by design
    SecurityByDesign,
    /// Privacy-friendly defaults
    PrivacyFriendlyDefaults,
}

/// Privacy engineering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyEngineeringConfig {
    /// Enable privacy engineering
    pub enabled: bool,
    /// Privacy patterns
    pub patterns: Vec<PrivacyPattern>,
    /// Privacy controls
    pub controls: Vec<PrivacyControl>,
}

impl Default for PrivacyEngineeringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            patterns: vec![
                PrivacyPattern::DataMinimizer,
                PrivacyPattern::ConsentManager,
                PrivacyPattern::AccessController,
            ],
            controls: vec![
                PrivacyControl::DataMasking,
                PrivacyControl::Pseudonymization,
                PrivacyControl::AccessControl,
            ],
        }
    }
}

/// Privacy patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrivacyPattern {
    /// Data minimizer
    DataMinimizer,
    /// Consent manager
    ConsentManager,
    /// Access controller
    AccessController,
    /// Anonymizer
    Anonymizer,
    /// Audit trail
    AuditTrail,
}

/// Privacy controls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrivacyControl {
    /// Data masking
    DataMasking,
    /// Pseudonymization
    Pseudonymization,
    /// Access control
    AccessControl,
    /// Encryption
    Encryption,
    /// Anonymization
    Anonymization,
}

/// Default settings configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultSettingsConfig {
    /// Privacy-friendly defaults
    pub privacy_friendly: bool,
    /// Minimal data collection
    pub minimal_collection: bool,
    /// Opt-in by default
    pub opt_in_default: bool,
}

impl Default for DefaultSettingsConfig {
    fn default() -> Self {
        Self {
            privacy_friendly: true,
            minimal_collection: true,
            opt_in_default: true,
        }
    }
}
