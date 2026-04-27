//! Data processing and retention types

// Re-export from manager for now
pub use super::manager::{
    ContactDetails, ControllerInfo, DataCategory, DataMinimizationConfig, DataProcessingConfig,
    DataRetentionConfig, InternationalTransfer, LegalBasis, MinimizationStrategy,
    ProcessingActivity, ProcessingPurpose, RecipientInfo, RecipientType, TransferMechanism,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_data_processing_config_default_enabled() {
        let config = DataProcessingConfig::default();
        assert!(config.enabled);
    }

    #[test]
    fn test_data_processing_config_default_has_legal_bases() {
        let config = DataProcessingConfig::default();
        assert!(!config.legal_bases.is_empty());
    }

    #[test]
    fn test_data_processing_config_default_has_processing_purposes() {
        let config = DataProcessingConfig::default();
        assert!(!config.processing_purposes.is_empty());
    }

    #[test]
    fn test_data_processing_config_default_has_data_categories() {
        let config = DataProcessingConfig::default();
        assert!(!config.data_categories.is_empty());
    }

    #[test]
    fn test_legal_basis_consent_variant() {
        let lb = LegalBasis::Consent;
        let s = format!("{:?}", lb);
        assert!(s.contains("Consent"));
    }

    #[test]
    fn test_legal_basis_all_variants() {
        let bases = [
            LegalBasis::Consent,
            LegalBasis::Contract,
            LegalBasis::LegalObligation,
            LegalBasis::VitalInterests,
            LegalBasis::PublicTask,
            LegalBasis::LegitimateInterests,
        ];
        let set: std::collections::HashSet<String> =
            bases.iter().map(|b| format!("{:?}", b)).collect();
        assert_eq!(set.len(), 6);
    }

    #[test]
    fn test_data_category_variants() {
        let cats = [
            DataCategory::PersonalData,
            DataCategory::SpecialCategoryData,
            DataCategory::PseudonymizedData,
            DataCategory::AnonymizedData,
        ];
        for cat in &cats {
            let s = format!("{:?}", cat);
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn test_processing_purpose_creation() {
        let purpose = ProcessingPurpose {
            id: "analytics".to_string(),
            name: "Analytics".to_string(),
            description: "Track user behavior".to_string(),
            legal_basis: LegalBasis::Consent,
            retention_period: Duration::from_secs(365 * 24 * 3600),
        };
        assert_eq!(purpose.id, "analytics");
        assert_eq!(purpose.retention_period.as_secs(), 365 * 24 * 3600);
    }

    #[test]
    fn test_data_minimization_config_default() {
        let config = DataMinimizationConfig::default();
        let s = format!("{:?}", config);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_minimization_strategy_variants() {
        let strategies = [
            MinimizationStrategy::RemoveUnnecessaryFields,
            MinimizationStrategy::AggregateData,
            MinimizationStrategy::PseudonymizeData,
            MinimizationStrategy::AnonymizeData,
            MinimizationStrategy::ReduceGranularity,
            MinimizationStrategy::LimitCollectionScope,
        ];
        for s in &strategies {
            let debug = format!("{:?}", s);
            assert!(!debug.is_empty());
        }
    }

    #[test]
    fn test_data_retention_config_default() {
        let config = DataRetentionConfig::default();
        let s = format!("{:?}", config);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_transfer_mechanism_variants() {
        let mechs = [
            TransferMechanism::AdequacyDecision,
            TransferMechanism::StandardContractualClauses,
            TransferMechanism::BindingCorporateRules,
        ];
        for m in &mechs {
            let s = format!("{:?}", m);
            assert!(!s.is_empty());
        }
    }
}
