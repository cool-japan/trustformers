//! Data processing configuration and tracking
//!
//! This module handles data processing activities, legal bases, data minimization,
//! and processing activity records as required by GDPR Articles 5, 6, and 30.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use super::types::{ContactDetails, DataCategory, LegalBasis, ProcessingPurpose};

/// Data processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataProcessingConfig {
    /// Enable data processing tracking
    pub enabled: bool,
    /// Legal bases for processing
    pub legal_bases: Vec<LegalBasis>,
    /// Processing purposes
    pub processing_purposes: Vec<ProcessingPurpose>,
    /// Data categories
    pub data_categories: Vec<DataCategory>,
    /// Processing activities register
    pub processing_activities: Vec<ProcessingActivity>,
    /// Data minimization configuration
    pub data_minimization: DataMinimizationConfig,
}

impl Default for DataProcessingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            legal_bases: vec![
                LegalBasis::Consent,
                LegalBasis::Contract,
                LegalBasis::LegalObligation,
                LegalBasis::LegitimateInterests,
            ],
            processing_purposes: vec![
                ProcessingPurpose {
                    id: "model_training".to_string(),
                    name: "Model Training".to_string(),
                    description: "Training AI models on user data".to_string(),
                    legal_basis: LegalBasis::Consent,
                    retention_period: Duration::from_secs(365 * 24 * 3600), // 1 year
                },
                ProcessingPurpose {
                    id: "service_provision".to_string(),
                    name: "Service Provision".to_string(),
                    description: "Providing AI inference services".to_string(),
                    legal_basis: LegalBasis::Contract,
                    retention_period: Duration::from_secs(30 * 24 * 3600), // 30 days
                },
            ],
            data_categories: vec![
                DataCategory::PersonalData,
                DataCategory::SpecialCategoryData,
                DataCategory::PseudonymizedData,
                DataCategory::AnonymizedData,
            ],
            processing_activities: Vec::new(),
            data_minimization: DataMinimizationConfig::default(),
        }
    }
}

/// Processing activity record (Article 30 GDPR)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingActivity {
    /// Activity identifier
    pub id: String,
    /// Activity name
    pub name: String,
    /// Controller information
    pub controller: ControllerInfo,
    /// Joint controllers (if any)
    pub joint_controllers: Vec<ControllerInfo>,
    /// Data protection officer contact
    pub dpo_contact: Option<String>,
    /// Processing purposes
    pub purposes: Vec<String>,
    /// Categories of data subjects
    pub data_subject_categories: Vec<String>,
    /// Categories of personal data
    pub personal_data_categories: Vec<DataCategory>,
    /// Recipients of data
    pub recipients: Vec<RecipientInfo>,
    /// International transfers
    pub international_transfers: Vec<InternationalTransfer>,
    /// Retention periods
    pub retention_periods: HashMap<String, Duration>,
    /// Security measures
    pub security_measures: Vec<String>,
}

/// Controller information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControllerInfo {
    /// Organization name
    pub name: String,
    /// Contact details
    pub contact: ContactDetails,
    /// Representative (if outside EU)
    pub representative: Option<ContactDetails>,
}

/// Recipient information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecipientInfo {
    /// Recipient name
    pub name: String,
    /// Recipient type
    pub recipient_type: RecipientType,
    /// Contact details
    pub contact: ContactDetails,
    /// Data shared
    pub data_shared: Vec<String>,
}

/// Recipient types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecipientType {
    /// Internal recipient
    Internal,
    /// Third party processor
    Processor,
    /// Third party controller
    Controller,
    /// Public authority
    PublicAuthority,
    /// Other
    Other { description: String },
}

/// International transfer information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternationalTransfer {
    /// Destination country
    pub country: String,
    /// Transfer mechanism
    pub mechanism: TransferMechanism,
    /// Adequacy decision
    pub adequacy_decision: bool,
    /// Safeguards in place
    pub safeguards: Vec<String>,
}

/// Transfer mechanisms for international transfers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransferMechanism {
    /// Adequacy decision
    AdequacyDecision,
    /// Standard contractual clauses
    StandardContractualClauses,
    /// Binding corporate rules
    BindingCorporateRules,
    /// Certification schemes
    CertificationScheme,
    /// Codes of conduct
    CodeOfConduct,
    /// Derogations
    Derogation { basis: String },
}

/// Data minimization configuration (Article 5(1)(c) GDPR)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataMinimizationConfig {
    /// Enable data minimization
    pub enabled: bool,
    /// Automatic data reduction
    pub auto_reduction: bool,
    /// Minimization strategies
    pub strategies: Vec<MinimizationStrategy>,
    /// Data necessity checks
    pub necessity_checks: bool,
}

impl Default for DataMinimizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_reduction: true,
            strategies: vec![
                MinimizationStrategy::RemoveUnnecessaryFields,
                MinimizationStrategy::AggregateData,
                MinimizationStrategy::PseudonymizeData,
            ],
            necessity_checks: true,
        }
    }
}

/// Data minimization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MinimizationStrategy {
    /// Remove unnecessary fields
    RemoveUnnecessaryFields,
    /// Aggregate data
    AggregateData,
    /// Pseudonymize data
    PseudonymizeData,
    /// Anonymize data
    AnonymizeData,
    /// Reduce data granularity
    ReduceGranularity,
    /// Limit collection scope
    LimitCollectionScope,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_processing_config_default() {
        let cfg = DataProcessingConfig::default();
        assert!(cfg.enabled);
        assert!(!cfg.legal_bases.is_empty());
        assert!(!cfg.processing_purposes.is_empty());
        assert!(!cfg.data_categories.is_empty());
        assert!(cfg.processing_activities.is_empty());
    }

    #[test]
    fn test_data_processing_config_default_legal_bases_count() {
        let cfg = DataProcessingConfig::default();
        assert_eq!(cfg.legal_bases.len(), 4);
    }

    #[test]
    fn test_data_processing_config_default_purposes_count() {
        let cfg = DataProcessingConfig::default();
        assert_eq!(cfg.processing_purposes.len(), 2);
    }

    #[test]
    fn test_data_minimization_config_default() {
        let cfg = DataMinimizationConfig::default();
        assert!(cfg.enabled);
        assert!(cfg.auto_reduction);
        assert!(cfg.necessity_checks);
        assert_eq!(cfg.strategies.len(), 3);
    }

    #[test]
    fn test_minimization_strategy_variants() {
        assert_eq!(
            format!("{:?}", MinimizationStrategy::RemoveUnnecessaryFields),
            "RemoveUnnecessaryFields"
        );
        assert_eq!(
            format!("{:?}", MinimizationStrategy::AggregateData),
            "AggregateData"
        );
        assert_eq!(
            format!("{:?}", MinimizationStrategy::PseudonymizeData),
            "PseudonymizeData"
        );
        assert_eq!(
            format!("{:?}", MinimizationStrategy::AnonymizeData),
            "AnonymizeData"
        );
        assert_eq!(
            format!("{:?}", MinimizationStrategy::ReduceGranularity),
            "ReduceGranularity"
        );
        assert_eq!(
            format!("{:?}", MinimizationStrategy::LimitCollectionScope),
            "LimitCollectionScope"
        );
    }

    #[test]
    fn test_recipient_type_variants() {
        assert_eq!(format!("{:?}", RecipientType::Internal), "Internal");
        assert_eq!(format!("{:?}", RecipientType::Processor), "Processor");
        assert_eq!(format!("{:?}", RecipientType::Controller), "Controller");
        assert_eq!(
            format!("{:?}", RecipientType::PublicAuthority),
            "PublicAuthority"
        );
        let other = RecipientType::Other {
            description: "research lab".to_string(),
        };
        match other {
            RecipientType::Other { description } => assert_eq!(description, "research lab"),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_transfer_mechanism_variants() {
        assert_eq!(
            format!("{:?}", TransferMechanism::AdequacyDecision),
            "AdequacyDecision"
        );
        assert_eq!(
            format!("{:?}", TransferMechanism::StandardContractualClauses),
            "StandardContractualClauses"
        );
        assert_eq!(
            format!("{:?}", TransferMechanism::BindingCorporateRules),
            "BindingCorporateRules"
        );
        assert_eq!(
            format!("{:?}", TransferMechanism::CertificationScheme),
            "CertificationScheme"
        );
        assert_eq!(
            format!("{:?}", TransferMechanism::CodeOfConduct),
            "CodeOfConduct"
        );
        let derogation = TransferMechanism::Derogation {
            basis: "vital interests".to_string(),
        };
        match derogation {
            TransferMechanism::Derogation { basis } => assert!(!basis.is_empty()),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_international_transfer_creation() {
        let transfer = InternationalTransfer {
            country: "US".to_string(),
            mechanism: TransferMechanism::StandardContractualClauses,
            adequacy_decision: false,
            safeguards: vec!["SCCs signed".to_string()],
        };
        assert_eq!(transfer.country, "US");
        assert!(!transfer.adequacy_decision);
        assert_eq!(transfer.safeguards.len(), 1);
    }

    #[test]
    fn test_controller_info_creation() {
        use crate::gdpr_compliance_modules::types::ContactDetails;
        let ctrl = ControllerInfo {
            name: "COOLJAPAN OU".to_string(),
            contact: ContactDetails {
                name: "DPO".to_string(),
                email: "dpo@example.com".to_string(),
                phone: None,
                address: None,
            },
            representative: None,
        };
        assert_eq!(ctrl.name, "COOLJAPAN OU");
        assert!(ctrl.representative.is_none());
    }

    #[test]
    fn test_recipient_info_creation() {
        use crate::gdpr_compliance_modules::types::ContactDetails;
        let recipient = RecipientInfo {
            name: "Analytics Co".to_string(),
            recipient_type: RecipientType::Processor,
            contact: ContactDetails {
                name: "Contact".to_string(),
                email: "contact@analytics.co".to_string(),
                phone: None,
                address: None,
            },
            data_shared: vec!["usage_data".to_string()],
        };
        assert_eq!(recipient.name, "Analytics Co");
        assert_eq!(recipient.data_shared.len(), 1);
    }
}
