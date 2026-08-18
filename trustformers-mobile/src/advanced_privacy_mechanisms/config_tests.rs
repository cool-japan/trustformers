//! Configuration-default tests for the privacy module.
//!
//! Split out of `mod.rs` to keep each file under the 2000-line limit. The
//! behavioural regression tests live in `engine_tests.rs`.

use super::*;

use super::*;

#[tokio::test]
async fn test_advanced_privacy_mechanisms() {
    let config = AdvancedPrivacyConfig::default();
    let privacy_engine = AdvancedPrivacyMechanisms::new(config).expect("Operation failed");

    let model_update = Tensor::zeros(&[10, 10]).expect("Operation failed");

    let result = privacy_engine
        .execute_private_federated_round(&model_update, "client_001")
        .await;

    let private_result = match result {
        Ok(value) => value,
        Err(error) => panic!("private federated round failed: {error}"),
    };
    // `execution_time` is a real measurement, so it must be non-zero. It is
    // deliberately NOT bounded from above: the round now performs genuine
    // Paillier key generation and post-quantum crypto, so any wall-clock
    // ceiling would be a flaky assertion about the machine rather than about
    // the code. `engine_tests.rs` asserts the artifacts are real instead.
    assert!(private_result.execution_time > Duration::ZERO);
    assert!(private_result.privacy_guarantees.epsilon > 0.0);
}

#[test]
fn test_privacy_config_defaults() {
    let config = AdvancedPrivacyConfig::default();
    assert!(config.differential_privacy.renyi_dp.alpha == 2.0);
    assert!(config.secure_mpc.num_parties == 100);
    assert!(config.homomorphic_encryption.security_level == 128);
    assert!(config.post_quantum.enabled);
}

#[test]
fn test_budget_allocation() {
    let allocation = BudgetAllocation {
        differential_privacy: DifferentialPrivacyBudget {
            epsilon: 1.0,
            delta: 1e-5,
            renyi_alpha: 2.0,
        },
        computational_budget: 1000.0,
        communication_budget: 2000.0,
    };

    let total_cost = allocation.total_cost();
    assert!(total_cost > 0.0);
    assert!(total_cost < 10.0); // Reasonable bounds
}

#[test]
fn test_renyi_dp_config_defaults() {
    let config = RenyiDPConfig::default();
    assert_eq!(config.alpha, 2.0);
    assert_eq!(config.epsilon_alpha, 1.0);
    assert_eq!(config.target_delta, 1e-5);
    assert!(!config.orders.is_empty());
    assert!(config.orders.contains(&2.0));
}

#[test]
fn test_privacy_amplification_config_defaults() {
    let config = PrivacyAmplificationConfig::default();
    assert!(config.enable_subsampling);
    assert_eq!(config.sampling_probability, 0.01);
    assert!(config.enable_shuffling);
    assert_eq!(config.shuffle_buffer_size, 10000);
    assert!(config.enable_iteration_amplification);
}

#[test]
fn test_concentrated_dp_config_defaults() {
    let config = ConcentratedDPConfig::default();
    assert_eq!(config.mu, 0.5);
    assert_eq!(config.privacy_loss_bound, 10.0);
    assert_eq!(config.tail_bound, 1e-6);
}

#[test]
fn test_local_dp_config_defaults() {
    let config = LocalDPConfig::default();
    assert!(config.enabled);
    assert_eq!(config.epsilon_local, 1.0);
}

#[test]
fn test_randomized_response_config_defaults() {
    let config = RandomizedResponseConfig::default();
    assert_eq!(config.true_probability, 0.75);
    assert_eq!(config.false_probability, 0.25);
    assert!(config.use_optimal);
}

#[test]
fn test_local_hashing_config_defaults() {
    let config = LocalHashingConfig::default();
    assert_eq!(config.num_hash_functions, 2);
    assert_eq!(config.domain_size, 1024);
    assert!(config.consistent_hashing);
}

#[test]
fn test_secure_multiparty_config_defaults() {
    let config = SecureMultipartyConfig::default();
    assert_eq!(config.num_parties, 100);
    assert_eq!(config.threshold, 67);
    assert!(matches!(config.protocol, MPCProtocol::SecureAggregation));
    assert!(matches!(config.secret_sharing, SecretSharingScheme::Shamir));
}

#[test]
fn test_communication_optimization_defaults() {
    let config = CommunicationOptimization::default();
    assert!(config.enable_compression);
    assert!(config.batch_communications);
    assert!(config.max_batch_size > 0);
}

#[test]
fn test_bootstrapping_config_defaults() {
    let config = BootstrappingConfig::default();
    assert!(config.enabled);
    assert_eq!(config.threshold, 0.1);
    assert_eq!(config.precision, 20);
}

#[test]
fn test_zero_knowledge_config_defaults() {
    let config = ZeroKnowledgeConfig::default();
    assert!(matches!(config.proof_system, ZKProofSystem::SchnorrSigma));
    assert!(config.circuit_optimization.enable_minimization);
}

#[test]
fn test_circuit_optimization_defaults() {
    let config = CircuitOptimization::default();
    assert!(config.enable_minimization);
    assert_eq!(config.gate_optimization_level, 2);
    assert!(config.wire_optimization);
    assert!(config.parallel_generation);
}

#[test]
fn test_zk_verification_config_defaults() {
    let config = ZKVerificationConfig::default();
    assert!(config.batch_verification);
    assert!(config.parallel_verification);
    assert!(config.enable_proof_caching);
    assert_eq!(config.cache_size_limit, 1000);
}

#[test]
fn test_private_retrieval_config_defaults() {
    let config = PrivateRetrievalConfig::default();
    assert!(matches!(config.scheme, PIRScheme::SingleServer));
    assert!(config.preprocessing.enabled);
}

#[test]
fn test_pir_preprocessing_defaults() {
    let config = PIRPreprocessing::default();
    assert!(config.enabled);
    assert!(matches!(
        config.strategy,
        PreprocessingStrategy::Incremental
    ));
    assert_eq!(config.update_frequency, Duration::from_secs(3600));
}

#[test]
fn test_pir_communication_config_defaults() {
    let config = PIRCommunicationConfig::default();
    assert!(config.enabled);
    assert!(config.batch_queries);
    assert_eq!(config.max_batch_size, 100);
    assert!(config.enable_compression);
}

#[test]
fn test_federated_analytics_config_defaults() {
    let config = FederatedAnalyticsConfig::default();
    assert_eq!(config.enabled_analytics.len(), 3);
}

#[test]
fn test_private_statistics_config_defaults() {
    let config = PrivateStatisticsConfig::default();
    assert!(!config.statistics.is_empty());
    assert_eq!(config.privacy_budget_per_statistic, 0.1);
}

#[test]
fn test_post_quantum_config_defaults() {
    let config = PostQuantumConfig::default();
    assert!(config.enabled);
    assert!(matches!(config.kem, KEMAlgorithm::MlKem768));
    assert!(matches!(config.signature, SignatureAlgorithm::MlDsa65));
    assert!(config.hybrid_security);
}

#[test]
fn test_adaptive_budgeting_config_defaults() {
    let config = AdaptiveBudgetingConfig::default();
    assert!(config.enabled);
    assert_eq!(config.initial_budget, 10.0);
    assert!(matches!(
        config.allocation_strategy,
        BudgetAllocationStrategy::Proportional
    ));
}

#[test]
fn test_budget_renewal_config_defaults() {
    let config = BudgetRenewalConfig::default();
    assert!(matches!(config.strategy, RenewalStrategy::Periodic));
    assert_eq!(config.interval, Duration::from_secs(3600 * 24));
    assert_eq!(config.amount, 1.0);
}

#[test]
fn test_fairness_constraints_defaults() {
    let config = FairnessConstraints::default();
    assert!(config.enabled);
    assert_eq!(config.max_budget_per_client, 2.0);
    assert_eq!(config.min_participation_rate, 0.1);
    assert!(matches!(config.fairness_metric, FairnessMetric::MaxMin));
}

#[test]
fn test_privacy_state_new() {
    let state = PrivacyState::new();
    assert!(state.client_budgets.is_empty());
    assert_eq!(state.global_privacy_loss, 0.0);
    assert!(state.active_sessions.is_empty());
}

#[test]
fn test_privacy_state_update_client_budget() {
    let mut state = PrivacyState::new();
    let allocation = BudgetAllocation {
        differential_privacy: DifferentialPrivacyBudget {
            epsilon: 0.5,
            delta: 1e-5,
            renyi_alpha: 2.0,
        },
        computational_budget: 100.0,
        communication_budget: 200.0,
    };
    state.update_client_budget("client1", &allocation);
    let budget = state.client_budgets.get("client1");
    assert!(budget.is_some());
}

#[test]
fn test_budget_allocation_total_cost_calculation() {
    let allocation = BudgetAllocation {
        differential_privacy: DifferentialPrivacyBudget {
            epsilon: 2.0,
            delta: 1e-5,
            renyi_alpha: 3.0,
        },
        computational_budget: 0.0,
        communication_budget: 0.0,
    };
    // With zero computational and communication budgets, cost = epsilon
    assert_eq!(allocation.total_cost(), 2.0);
}

#[test]
fn test_budget_allocation_components() {
    let allocation = BudgetAllocation {
        differential_privacy: DifferentialPrivacyBudget {
            epsilon: 1.0,
            delta: 1e-5,
            renyi_alpha: 2.0,
        },
        computational_budget: 500.0,
        communication_budget: 1000.0,
    };
    let cost = allocation.total_cost();
    // cost = 1.0 + 500*0.001 + 1000*0.0001 = 1.0 + 0.5 + 0.1 = 1.6
    assert!((cost - 1.6).abs() < 1e-10);
}

#[test]
fn test_privacy_state_default() {
    let state = PrivacyState::default();
    assert!(state.client_budgets.is_empty());
}

#[test]
fn test_privacy_state_multiple_clients() {
    let mut state = PrivacyState::new();
    let allocation = BudgetAllocation {
        differential_privacy: DifferentialPrivacyBudget {
            epsilon: 0.1,
            delta: 1e-5,
            renyi_alpha: 2.0,
        },
        computational_budget: 10.0,
        communication_budget: 20.0,
    };
    state.update_client_budget("client_a", &allocation);
    state.update_client_budget("client_b", &allocation);
    assert_eq!(state.client_budgets.len(), 2);
}

#[test]
fn test_heavy_hitters_config_defaults() {
    let config = HeavyHittersConfig::default();
    assert!(config.threshold > 0.0);
    assert!(config.max_heavy_hitters > 0);
}

#[test]
fn test_private_histogram_config_defaults() {
    let config = PrivateHistogramConfig::default();
    assert!(config.num_bins > 0);
}

#[test]
fn test_clipping_bounds_defaults() {
    let bounds = ClippingBounds::default();
    assert!(bounds.lower_bound < bounds.upper_bound);
}
