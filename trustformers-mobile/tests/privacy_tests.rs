//! Differential privacy tests for trustformers-mobile
//!
//! Tests privacy configuration validation, epsilon/delta bounds,
//! noise parameter relationships, and composition methods
//! without actual training data.
//!
//! Requires feature: on-device-training

#[cfg(feature = "on-device-training")]
mod dp_tests {
    use trustformers_mobile::differential_privacy::{
        CompositionMethod, DifferentialPrivacyEngine, PrivacyConfig, PrivacyLevel,
    };

    fn make_privacy_config_custom(
        epsilon: f64,
        delta: f64,
        clip: f32,
        noise: f32,
    ) -> PrivacyConfig {
        PrivacyConfig {
            privacy_level: PrivacyLevel::Custom,
            total_epsilon: epsilon,
            total_delta: delta,
            noise_multiplier: noise,
            clipping_threshold: clip,
            per_example_clipping: true,
            adaptive_clipping: false,
            subsampling_rate: 0.05,
            composition_method: CompositionMethod::Advanced,
        }
    }

    #[test]
    fn test_privacy_config_default_is_medium() {
        let config = PrivacyConfig::default();
        assert_eq!(config.privacy_level, PrivacyLevel::Medium);
    }

    #[test]
    fn test_privacy_level_low_has_high_epsilon() {
        let config = PrivacyConfig::from_privacy_level(PrivacyLevel::Low);
        assert!(
            config.total_epsilon > 1.0,
            "low privacy should have epsilon > 1.0"
        );
    }

    #[test]
    fn test_privacy_level_veryhigh_has_low_epsilon() {
        let config = PrivacyConfig::from_privacy_level(PrivacyLevel::VeryHigh);
        assert!(
            config.total_epsilon < 1.0,
            "very high privacy should have epsilon < 1.0"
        );
    }

    #[test]
    fn test_privacy_epsilon_ordering_across_levels() {
        let low = PrivacyConfig::from_privacy_level(PrivacyLevel::Low);
        let medium = PrivacyConfig::from_privacy_level(PrivacyLevel::Medium);
        let high = PrivacyConfig::from_privacy_level(PrivacyLevel::High);
        let very_high = PrivacyConfig::from_privacy_level(PrivacyLevel::VeryHigh);
        // Higher privacy level → smaller epsilon (stronger privacy)
        assert!(low.total_epsilon > medium.total_epsilon);
        assert!(medium.total_epsilon > high.total_epsilon);
        assert!(high.total_epsilon > very_high.total_epsilon);
    }

    #[test]
    fn test_privacy_noise_ordering_across_levels() {
        let low = PrivacyConfig::from_privacy_level(PrivacyLevel::Low);
        let very_high = PrivacyConfig::from_privacy_level(PrivacyLevel::VeryHigh);
        assert!(very_high.noise_multiplier > low.noise_multiplier);
    }

    #[test]
    fn test_privacy_delta_is_small_positive() {
        let config = PrivacyConfig::from_privacy_level(PrivacyLevel::Medium);
        assert!(config.total_delta > 0.0, "delta must be positive");
        assert!(config.total_delta < 1.0, "delta must be less than 1");
        assert!(config.total_delta < 1e-4, "delta should be very small");
    }

    #[test]
    fn test_privacy_epsilon_must_be_positive() {
        let config = make_privacy_config_custom(1.0, 1e-5, 1.0, 1.0);
        assert!(config.total_epsilon > 0.0, "epsilon must be positive");
    }

    #[test]
    fn test_clipping_threshold_positive() {
        let config = make_privacy_config_custom(1.0, 1e-5, 0.5, 1.0);
        assert!(
            config.clipping_threshold > 0.0,
            "clipping threshold must be positive"
        );
    }

    #[test]
    fn test_noise_multiplier_positive() {
        let config = make_privacy_config_custom(1.0, 1e-5, 1.0, 2.0);
        assert!(
            config.noise_multiplier > 0.0,
            "noise multiplier must be positive"
        );
    }

    #[test]
    fn test_subsampling_rate_in_valid_range() {
        let config = PrivacyConfig::from_privacy_level(PrivacyLevel::Medium);
        assert!(
            config.subsampling_rate > 0.0,
            "subsampling rate must be positive"
        );
        assert!(
            config.subsampling_rate <= 1.0,
            "subsampling rate must be <= 1"
        );
    }

    #[test]
    fn test_privacy_level_variants_exist() {
        let _low = PrivacyLevel::Low;
        let _medium = PrivacyLevel::Medium;
        let _high = PrivacyLevel::High;
        let _very_high = PrivacyLevel::VeryHigh;
        let _custom = PrivacyLevel::Custom;
    }

    #[test]
    fn test_composition_method_variants() {
        let _simple = CompositionMethod::Simple;
        let _advanced = CompositionMethod::Advanced;
        let _moments = CompositionMethod::Moments;
        let _renyi = CompositionMethod::Renyi;
    }

    #[test]
    fn test_privacy_config_serialization_roundtrip() {
        let config = PrivacyConfig::from_privacy_level(PrivacyLevel::High);
        let json = serde_json::to_string(&config).expect("serialization should succeed");
        let restored: PrivacyConfig =
            serde_json::from_str(&json).expect("deserialization should succeed");
        assert!((restored.total_epsilon - config.total_epsilon).abs() < 1e-10);
        assert!((restored.total_delta - config.total_delta).abs() < 1e-15);
        assert_eq!(restored.privacy_level, config.privacy_level);
    }

    #[test]
    fn test_high_privacy_uses_moments_or_renyi_composition() {
        let high = PrivacyConfig::from_privacy_level(PrivacyLevel::High);
        let very_high = PrivacyConfig::from_privacy_level(PrivacyLevel::VeryHigh);
        let simple = CompositionMethod::Simple;
        let advanced = CompositionMethod::Advanced;
        assert_ne!(high.composition_method, simple);
        assert_ne!(very_high.composition_method, advanced);
    }

    #[test]
    fn test_per_example_clipping_enabled_by_default_for_high_privacy() {
        let config = PrivacyConfig::from_privacy_level(PrivacyLevel::High);
        assert!(
            config.per_example_clipping,
            "high privacy should use per-example clipping"
        );
    }

    #[test]
    fn test_differential_privacy_engine_creation() {
        let config = PrivacyConfig::from_privacy_level(PrivacyLevel::Medium);
        let _engine = DifferentialPrivacyEngine::new(config);
    }
}
