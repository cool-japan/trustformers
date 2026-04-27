//! Federated learning aggregation tests for trustformers-mobile
//!
//! Tests federated learning configuration, aggregation strategies, and
//! differential privacy config integration without actual networking.
//!
//! Requires feature: on-device-training

#[cfg(feature = "on-device-training")]
mod federated_tests {
    use trustformers_mobile::{
        AggregationStrategy, ClientSelectionStrategy, DifferentialPrivacyConfig,
        FederatedLearningConfig, NoiseMechanism,
    };

    fn make_federated_config() -> FederatedLearningConfig {
        FederatedLearningConfig {
            server_endpoint: "https://fl.example.com".to_string(),
            client_id: "test-client-001".to_string(),
            local_epochs: 5,
            min_clients_for_aggregation: 10,
            enable_differential_privacy: true,
            dp_config: Some(DifferentialPrivacyConfig::default()),
            enable_secure_aggregation: true,
            communication_rounds: 50,
            enable_compression: true,
            compression_ratio: 0.1,
            client_selection: ClientSelectionStrategy::ResourceBased,
            aggregation_strategy: AggregationStrategy::FedAvg,
        }
    }

    #[test]
    fn test_federated_config_default_creation() {
        let config = FederatedLearningConfig::default();
        assert!(
            !config.server_endpoint.is_empty(),
            "server endpoint should not be empty"
        );
        assert!(
            !config.client_id.is_empty(),
            "client_id should not be empty"
        );
    }

    #[test]
    fn test_federated_config_local_epochs_positive() {
        let config = make_federated_config();
        assert!(config.local_epochs > 0, "local_epochs must be positive");
    }

    #[test]
    fn test_federated_config_min_clients_positive() {
        let config = make_federated_config();
        assert!(
            config.min_clients_for_aggregation > 0,
            "min_clients must be positive for security"
        );
    }

    #[test]
    fn test_federated_config_compression_ratio_valid() {
        let config = make_federated_config();
        assert!(
            config.compression_ratio > 0.0 && config.compression_ratio <= 1.0,
            "compression_ratio must be in (0, 1]"
        );
    }

    #[test]
    fn test_aggregation_strategy_fedavg_variant() {
        let strategy = AggregationStrategy::FedAvg;
        assert_eq!(strategy, AggregationStrategy::FedAvg);
    }

    #[test]
    fn test_aggregation_strategy_variants_exist() {
        let _fedavg = AggregationStrategy::FedAvg;
        let _weighted = AggregationStrategy::WeightedAvg;
        let _momentum = AggregationStrategy::FedMomentum;
        let _yogi = AggregationStrategy::FedYogi;
        let _personalized = AggregationStrategy::PersonalizedFed;
    }

    #[test]
    fn test_client_selection_strategy_variants_exist() {
        let _random = ClientSelectionStrategy::Random;
        let _resource = ClientSelectionStrategy::ResourceBased;
        let _quality = ClientSelectionStrategy::QualityBased;
        let _round_robin = ClientSelectionStrategy::RoundRobin;
        let _speed = ClientSelectionStrategy::SpeedOptimized;
    }

    #[test]
    fn test_noise_mechanism_variants_exist() {
        let _gaussian = NoiseMechanism::Gaussian;
        let _laplacian = NoiseMechanism::Laplacian;
        let _exponential = NoiseMechanism::Exponential;
    }

    #[test]
    fn test_dp_config_default_epsilon_positive() {
        let config = DifferentialPrivacyConfig::default();
        assert!(config.epsilon > 0.0, "epsilon must be positive");
    }

    #[test]
    fn test_dp_config_default_delta_small_positive() {
        let config = DifferentialPrivacyConfig::default();
        assert!(
            config.delta > 0.0 && config.delta < 1.0,
            "delta must be in (0, 1)"
        );
    }

    #[test]
    fn test_dp_config_default_clipping_norm_positive() {
        let config = DifferentialPrivacyConfig::default();
        assert!(config.clipping_norm > 0.0, "clipping norm must be positive");
    }

    #[test]
    fn test_federated_config_with_dp_enabled() {
        let config = make_federated_config();
        assert!(config.enable_differential_privacy);
        assert!(config.dp_config.is_some());
    }

    #[test]
    fn test_federated_config_secure_aggregation_enabled() {
        let config = make_federated_config();
        assert!(config.enable_secure_aggregation);
    }

    #[test]
    fn test_federated_config_serialization_roundtrip() {
        let config = make_federated_config();
        let json = serde_json::to_string(&config).expect("serialization should succeed");
        let restored: FederatedLearningConfig =
            serde_json::from_str(&json).expect("deserialization should succeed");
        assert_eq!(restored.server_endpoint, config.server_endpoint);
        assert_eq!(restored.local_epochs, config.local_epochs);
        assert_eq!(restored.aggregation_strategy, config.aggregation_strategy);
    }

    #[test]
    fn test_communication_rounds_positive() {
        let config = make_federated_config();
        assert!(
            config.communication_rounds > 0,
            "communication rounds must be positive"
        );
    }
}
