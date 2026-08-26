//! Regression tests for the privacy engines.
//!
//! Every test here targets a specific fake behaviour the previous
//! implementation shipped; the doc comment on each names it.

use super::*;
use crate::advanced_security::test_rng::TestRng;

fn dp_engine() -> AdvancedDifferentialPrivacy {
    AdvancedDifferentialPrivacy::new(AdvancedDifferentialPrivacyConfig::default())
        .expect("dp engine")
}

/// Regression: `privatize_update` returned `Tensor::zeros(&[1, 1])`, throwing
/// the gradient away while the caller reported a spent budget.
#[test]
fn privatize_update_preserves_shape_and_uses_the_input() {
    let engine = dp_engine();
    let update = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]).expect("tensor");
    let budget = DifferentialPrivacyBudget {
        epsilon: 1.0,
        delta: 1e-5,
        renyi_alpha: 2.0,
    };

    let mut rng = TestRng::seeded(1);
    let privatized =
        engine.privatize_update_with_rng(&update, &budget, &mut rng).expect("privatize");

    // The old code returned a 1x1 tensor.
    assert_eq!(privatized.shape(), update.shape());
    assert_eq!(privatized.shape(), vec![2, 3]);

    let values = privatized.to_vec_f32().expect("vec");
    assert_eq!(values.len(), 6);
    assert!(
        values.iter().any(|v| *v != 0.0),
        "output must not be all zeros"
    );
}

/// Two calls with different randomness must give different outputs: the
/// mechanism is genuinely stochastic, not a fixed transform.
#[test]
fn privatize_update_is_stochastic() {
    let engine = dp_engine();
    let update = Tensor::from_vec(vec![0.5; 32], &[32]).expect("tensor");
    let budget = DifferentialPrivacyBudget {
        epsilon: 1.0,
        delta: 1e-5,
        renyi_alpha: 2.0,
    };

    let mut rng_a = TestRng::seeded(2);
    let mut rng_b = TestRng::seeded(3);
    let a = engine
        .privatize_update_with_rng(&update, &budget, &mut rng_a)
        .expect("a")
        .to_vec_f32()
        .expect("vec");
    let b = engine
        .privatize_update_with_rng(&update, &budget, &mut rng_b)
        .expect("b")
        .to_vec_f32()
        .expect("vec");
    assert_ne!(a, b, "privatization must be randomized");
}

/// Clipping must bound the sensitivity: a huge gradient is brought inside the
/// configured `L2` ball before noise is added.
#[test]
fn privatize_update_clips_before_adding_noise() {
    let engine =
        AdvancedDifferentialPrivacy::with_clipping_norm(Default::default(), 1.0).expect("engine");
    // An update with L2 norm 1000.
    let update = Tensor::from_vec(vec![1000.0, 0.0], &[2]).expect("tensor");
    let budget = DifferentialPrivacyBudget {
        epsilon: 1.0,
        delta: 1e-5,
        renyi_alpha: 2.0,
    };

    let mut rng = TestRng::seeded(4);
    let privatized = engine
        .privatize_update_with_rng(&update, &budget, &mut rng)
        .expect("privatize")
        .to_vec_f32()
        .expect("vec");

    // The signal component is clipped to 1.0; the noise sigma for these
    // parameters is a few units, so the result must be nowhere near 1000.
    assert!(
        privatized[0].abs() < 100.0,
        "clipping should have bounded the value, got {}",
        privatized[0]
    );
}

/// A pure-epsilon budget must route to the Laplace mechanism and still work.
#[test]
fn privatize_update_supports_pure_epsilon_dp() {
    let engine = dp_engine();
    let update = Tensor::from_vec(vec![1.0, 2.0], &[2]).expect("tensor");
    let budget = DifferentialPrivacyBudget {
        epsilon: 0.5,
        delta: 0.0,
        renyi_alpha: 2.0,
    };
    let mut rng = TestRng::seeded(5);
    let privatized =
        engine.privatize_update_with_rng(&update, &budget, &mut rng).expect("privatize");
    assert_eq!(privatized.shape(), vec![2]);
}

#[test]
fn privatize_update_rejects_invalid_budgets() {
    let engine = dp_engine();
    let update = Tensor::from_vec(vec![1.0], &[1]).expect("tensor");
    let mut rng = TestRng::seeded(6);
    for (epsilon, delta) in [(0.0f64, 1e-5f64), (-1.0, 1e-5), (1.0, 1.0)] {
        let budget = DifferentialPrivacyBudget {
            epsilon,
            delta,
            renyi_alpha: 2.0,
        };
        assert!(
            engine.privatize_update_with_rng(&update, &budget, &mut rng).is_err(),
            "budget ({epsilon}, {delta}) must be refused"
        );
    }
}

fn mpc_engine() -> SecureMultipartyComputation {
    let mut config = SecureMultipartyConfig::default();
    config.num_parties = 5;
    config.threshold = 3;
    SecureMultipartyComputation::new(config).expect("mpc engine")
}

/// Regression: `create_secret_shares` returned exactly one
/// `SecretShare { share_data: vec![0u8; 32] }` regardless of input.
#[test]
fn create_secret_shares_returns_real_distinct_shares() {
    let engine = mpc_engine();
    let data = Tensor::from_vec(vec![1.5, -2.5, 3.0, 4.25], &[2, 2]).expect("tensor");
    let mut rng = TestRng::seeded(10);
    let shares = engine
        .create_secret_shares_with_rng(&data, "client_a", &mut rng)
        .expect("split");

    // The old code returned one share.
    assert_eq!(shares.len(), 5);
    // No share may be all zeros, and all must differ from one another.
    for share in &shares {
        assert!(
            share.share_data.iter().any(|b| *b != 0),
            "share {} is all zeros",
            share.share_id
        );
        assert_eq!(share.client_id, "client_a");
    }
    for i in 0..shares.len() {
        for j in (i + 1)..shares.len() {
            assert_ne!(
                shares[i].share_data, shares[j].share_data,
                "shares {i} and {j} are identical"
            );
        }
    }
}

/// Any `threshold` shares must reconstruct the tensor exactly.
#[test]
fn secret_shares_reconstruct_exactly() {
    let engine = mpc_engine();
    let data = Tensor::from_vec(vec![1.5, -2.5, 3.0, 4.25], &[2, 2]).expect("tensor");
    let mut rng = TestRng::seeded(11);
    let shares = engine
        .create_secret_shares_with_rng(&data, "client_a", &mut rng)
        .expect("split");

    let reconstructed = engine.reconstruct(&shares[..3]).expect("reconstruct");
    assert_eq!(reconstructed.shape(), data.shape());
    assert_eq!(
        reconstructed.to_vec_f32().expect("vec"),
        data.to_vec_f32().expect("vec")
    );

    // A different subset gives the same answer.
    let other = vec![shares[1].clone(), shares[3].clone(), shares[4].clone()];
    assert_eq!(
        engine.reconstruct(&other).expect("reconstruct").to_vec_f32().expect("vec"),
        data.to_vec_f32().expect("vec")
    );
}

/// Below the threshold, reconstruction must fail rather than return something.
#[test]
fn secret_shares_below_threshold_are_refused() {
    let engine = mpc_engine();
    let data = Tensor::from_vec(vec![7.0], &[1]).expect("tensor");
    let mut rng = TestRng::seeded(12);
    let shares = engine.create_secret_shares_with_rng(&data, "c", &mut rng).expect("split");
    assert!(engine.reconstruct(&shares[..2]).is_err());
}

#[test]
fn mpc_refuses_unimplemented_sharing_schemes() {
    for scheme in [
        SecretSharingScheme::Additive,
        SecretSharingScheme::Replicated,
        SecretSharingScheme::Packed,
    ] {
        let mut config = SecureMultipartyConfig::default();
        config.num_parties = 3;
        config.threshold = 2;
        config.secret_sharing = scheme.clone();
        let engine = SecureMultipartyComputation::new(config).expect("engine");
        let data = Tensor::from_vec(vec![1.0], &[1]).expect("tensor");
        let mut rng = TestRng::seeded(13);
        assert!(
            engine.create_secret_shares_with_rng(&data, "c", &mut rng).is_err(),
            "{scheme:?} must be refused, not return placeholder shares"
        );
    }
}

#[test]
fn mpc_rejects_bad_configuration() {
    let build = |num_parties, threshold| {
        let mut config = SecureMultipartyConfig::default();
        config.num_parties = num_parties;
        config.threshold = threshold;
        SecureMultipartyComputation::new(config)
    };
    assert!(build(1, 1).is_err(), "1 party");
    assert!(build(3, 1).is_err(), "threshold 1");
    assert!(build(3, 4).is_err(), "threshold > parties");
    assert!(build(300, 2).is_err(), "too many parties");
    assert!(build(3, 2).is_ok());
}

/// A 512-bit Paillier keypair keeps these tests fast.
fn he_engine(seed: u8) -> HomomorphicEncryption {
    let config = HomomorphicEncryptionConfig::default();
    let mut rng = TestRng::seeded(seed);
    // Build directly at a test modulus size rather than the production 3072.
    let keypair = paillier::generate_keypair_with_rng(512, &mut rng).expect("keygen");
    HomomorphicEncryption::from_parts(config, keypair)
}

/// Regression: `encrypt_shares` returned `vec![0u8; 64]` regardless of input.
#[test]
fn encrypt_shares_produces_real_ciphertexts() {
    let engine = he_engine(20);
    let shares = vec![
        SecretShare {
            share_data: vec![0xAB; 100],
            share_id: 1,
            client_id: "c".to_string(),
        },
        SecretShare {
            share_data: vec![0xCD; 100],
            share_id: 2,
            client_id: "c".to_string(),
        },
    ];

    let mut rng = TestRng::seeded(21);
    let encrypted = engine.encrypt_shares_with_rng(&shares, &mut rng).expect("encrypt");
    assert_eq!(encrypted.len(), 2);

    for (share, ciphertext) in shares.iter().zip(encrypted.iter()) {
        assert!(!ciphertext.ciphertext_blocks.is_empty());
        assert_eq!(ciphertext.plaintext_len, share.share_data.len());
        let flat: Vec<u8> = ciphertext.ciphertext_blocks.iter().flatten().copied().collect();
        assert!(flat.iter().any(|b| *b != 0), "ciphertext is all zeros");
        assert!(
            !flat.windows(share.share_data.len()).any(|w| w == share.share_data.as_slice()),
            "ciphertext contains the share verbatim"
        );
    }
    // Distinct plaintexts give distinct ciphertexts.
    assert_ne!(
        encrypted[0].ciphertext_blocks,
        encrypted[1].ciphertext_blocks
    );
}

/// Encrypted shares must decrypt back to exactly the original bytes, including
/// any leading zeros.
#[test]
fn encrypted_shares_round_trip() {
    let engine = he_engine(22);
    for payload in [
        vec![0x01u8; 10],
        vec![0x00, 0x00, 0xFF, 0x01], // leading zeros must survive
        (0u8..=255).collect::<Vec<u8>>(),
        vec![0xFFu8; 500], // spans several blocks
    ] {
        let shares = vec![SecretShare {
            share_data: payload.clone(),
            share_id: 1,
            client_id: "c".to_string(),
        }];
        let mut rng = TestRng::seeded(23);
        let encrypted = engine.encrypt_shares_with_rng(&shares, &mut rng).expect("encrypt");
        let decrypted = engine.decrypt_share(&encrypted[0]).expect("decrypt");
        assert_eq!(
            decrypted,
            payload,
            "round trip failed for {} bytes",
            payload.len()
        );
    }
}

#[test]
fn homomorphic_engine_refuses_fully_homomorphic_schemes() {
    for scheme in [
        HomomorphicScheme::BFV,
        HomomorphicScheme::CKKS,
        HomomorphicScheme::BGV,
        HomomorphicScheme::TFHE,
    ] {
        let mut config = HomomorphicEncryptionConfig::default();
        config.scheme = scheme.clone();
        let mut rng = TestRng::seeded(24);
        assert!(
            HomomorphicEncryption::new_with_rng(config, &mut rng).is_err(),
            "{scheme:?} must be refused, not faked"
        );
    }
}

fn zk_engine(seed: u8) -> ZeroKnowledgeProofs {
    let mut rng = TestRng::seeded(seed);
    ZeroKnowledgeProofs::new_with_rng(ZeroKnowledgeConfig::default(), &mut rng).expect("zk engine")
}

fn sample_budget() -> BudgetAllocation {
    BudgetAllocation {
        differential_privacy: DifferentialPrivacyBudget {
            epsilon: 1.0,
            delta: 1e-5,
            renyi_alpha: 2.0,
        },
        computational_budget: 0.0,
        communication_budget: 0.0,
    }
}

/// Regression: `generate_correctness_proof` returned `vec![0u8; 128]`.
#[test]
fn correctness_proof_is_real_and_verifies() {
    let engine = zk_engine(30);
    let original = Tensor::from_vec(vec![1.0, 2.0], &[2]).expect("original");
    let privatized = Tensor::from_vec(vec![1.1, 2.2], &[2]).expect("privatized");
    let budget = sample_budget();

    let mut rng = TestRng::seeded(31);
    let proof = engine
        .generate_correctness_proof_with_rng(&original, &privatized, &budget, &mut rng)
        .expect("prove");

    assert!(
        proof.proof_data.iter().any(|b| *b != 0),
        "proof is all zeros"
    );
    assert!(
        proof.verification_key.iter().any(|b| *b != 0),
        "verification key is all zeros"
    );
    assert!(engine
        .verify_correctness_proof(&proof, &original, &privatized, &budget)
        .expect("verify"));
}

/// The proof is bound to the exact transcript; changing any part breaks it.
#[test]
fn correctness_proof_is_bound_to_the_transcript() {
    let engine = zk_engine(32);
    let original = Tensor::from_vec(vec![1.0, 2.0], &[2]).expect("original");
    let privatized = Tensor::from_vec(vec![1.1, 2.2], &[2]).expect("privatized");
    let budget = sample_budget();

    let mut rng = TestRng::seeded(33);
    let proof = engine
        .generate_correctness_proof_with_rng(&original, &privatized, &budget, &mut rng)
        .expect("prove");

    let other_original = Tensor::from_vec(vec![9.0, 9.0], &[2]).expect("other");
    assert!(
        !engine
            .verify_correctness_proof(&proof, &other_original, &privatized, &budget)
            .expect("verify"),
        "proof must not verify for a different original update"
    );

    let other_privatized = Tensor::from_vec(vec![5.0, 5.0], &[2]).expect("other");
    assert!(
        !engine
            .verify_correctness_proof(&proof, &original, &other_privatized, &budget)
            .expect("verify"),
        "proof must not verify for a different privatized update"
    );

    let mut other_budget = sample_budget();
    other_budget.differential_privacy.epsilon = 2.0;
    assert!(
        !engine
            .verify_correctness_proof(&proof, &original, &privatized, &other_budget)
            .expect("verify"),
        "proof must not verify for a different budget"
    );
}

/// Regression: any blob used to "verify".
#[test]
fn arbitrary_proof_bytes_do_not_verify() {
    let engine = zk_engine(34);
    let original = Tensor::from_vec(vec![1.0], &[1]).expect("original");
    let privatized = Tensor::from_vec(vec![1.0], &[1]).expect("privatized");
    let budget = sample_budget();

    for filler in [0u8, 0xAB, 0xFF] {
        let blob = ZKProof {
            proof_data: vec![filler; 128],
            verification_key: vec![filler; 32],
        };
        assert!(
            !engine
                .verify_correctness_proof(&blob, &original, &privatized, &budget)
                .expect("verify"),
            "a {filler:#04x}-filled blob must not verify"
        );
    }
}

#[test]
fn zk_engine_refuses_circuit_proof_systems() {
    for system in [
        ZKProofSystem::Groth16,
        ZKProofSystem::PLONK,
        ZKProofSystem::Bulletproofs,
        ZKProofSystem::STARK,
        ZKProofSystem::Marlin,
    ] {
        let mut config = ZeroKnowledgeConfig::default();
        config.proof_system = system.clone();
        let mut rng = TestRng::seeded(35);
        assert!(
            ZeroKnowledgeProofs::new_with_rng(config, &mut rng).is_err(),
            "{system:?} must be refused, not faked"
        );
    }
}

/// Regression: `retrieve_model_update` returned a zero tensor plus invented
/// privacy costs, implying the query pattern was hidden when it was not.
#[tokio::test]
async fn private_information_retrieval_reports_that_it_is_unimplemented() {
    let engine = PrivateInformationRetrieval::new(PrivateRetrievalConfig::default()).expect("pir");
    let query = ModelQuery {
        model_id: "llama-7b".to_string(),
        version: 3,
        client_capabilities: vec![],
    };
    let error = engine
        .retrieve_model_update(&query)
        .await
        .expect_err("PIR must not claim to work");
    let text = error.to_string();
    assert!(text.contains("llama-7b"), "{text}");
}

/// Regression: `compute_analytics` returned `mean: 0.5, variance: 0.1`
/// regardless of input. The statistics must track the real data.
#[test]
fn analytics_track_the_real_data() {
    // A large sample around 0.8 with a tiny per-statistic epsilon budget would
    // be swamped by noise, so use a generous budget: this test checks that the
    // statistic is computed from the data at all, not the privacy level.
    let mut config = FederatedAnalyticsConfig::default();
    config.statistics.privacy_budget_per_statistic = 50.0;
    config.statistics.clipping_bounds = ClippingBounds {
        lower_bound: 0.0,
        upper_bound: 1.0,
        adaptive_bounds: false,
    };
    let engine = PrivateFederatedAnalytics::new(config).expect("analytics");

    let samples = vec![Tensor::from_vec(vec![0.8; 1000], &[1000]).expect("tensor")];
    let mut rng = TestRng::seeded(40);
    let result = engine
        .compute_analytics_with_rng(&samples, &[AnalyticsType::BasicStatistics], &mut rng)
        .expect("analytics");

    let mean = *result.statistics.get("mean").expect("mean");
    // The old code always reported exactly 0.5.
    assert!(
        (mean - 0.8).abs() < 0.05,
        "mean {mean} should track the data (0.8), not a constant"
    );
    assert_ne!(mean, 0.5, "mean must not be the old hardcoded constant");

    let count = *result.statistics.get("count").expect("count");
    assert!(
        (count - 1000.0).abs() < 50.0,
        "count {count} should be near 1000"
    );

    // The reported privacy cost must reflect what was spent, not a constant.
    assert!(result.privacy_cost.differential_privacy_cost > 0.0);
    assert_eq!(result.privacy_cost.sample_count, 1000);
    assert_eq!(result.privacy_cost.statistic_count, result.statistics.len());

    // Confidence intervals must bracket the reported value.
    for (name, (low, high)) in &result.confidence_intervals {
        let value = *result.statistics.get(name).expect("statistic");
        assert!(
            low < &value && &value < high,
            "{name}: {value} not in ({low}, {high})"
        );
    }
}

/// A different dataset must give different statistics.
#[test]
fn analytics_differ_for_different_data() {
    let mut config = FederatedAnalyticsConfig::default();
    config.statistics.privacy_budget_per_statistic = 50.0;
    config.statistics.clipping_bounds = ClippingBounds {
        lower_bound: 0.0,
        upper_bound: 1.0,
        adaptive_bounds: false,
    };
    let engine = PrivateFederatedAnalytics::new(config).expect("analytics");

    let low = vec![Tensor::from_vec(vec![0.1; 500], &[500]).expect("t")];
    let high = vec![Tensor::from_vec(vec![0.9; 500], &[500]).expect("t")];
    let mut rng = TestRng::seeded(41);
    let low_result = engine
        .compute_analytics_with_rng(&low, &[AnalyticsType::BasicStatistics], &mut rng)
        .expect("low");
    let high_result = engine
        .compute_analytics_with_rng(&high, &[AnalyticsType::BasicStatistics], &mut rng)
        .expect("high");

    let low_mean = *low_result.statistics.get("mean").expect("mean");
    let high_mean = *high_result.statistics.get("mean").expect("mean");
    assert!(
        high_mean > low_mean,
        "mean of 0.9-data ({high_mean}) should exceed mean of 0.1-data ({low_mean})"
    );
}

#[test]
fn analytics_reject_empty_input_and_unimplemented_families() {
    let engine =
        PrivateFederatedAnalytics::new(FederatedAnalyticsConfig::default()).expect("analytics");
    let mut rng = TestRng::seeded(42);
    assert!(engine
        .compute_analytics_with_rng(&[], &[AnalyticsType::BasicStatistics], &mut rng)
        .is_err());

    let samples = vec![Tensor::from_vec(vec![0.5; 10], &[10]).expect("t")];
    for family in [
        AnalyticsType::HeavyHitters,
        AnalyticsType::Histograms,
        AnalyticsType::FrequentItemsets,
        AnalyticsType::GraphAnalytics,
    ] {
        assert!(
            engine
                .compute_analytics_with_rng(&samples, std::slice::from_ref(&family), &mut rng)
                .is_err(),
            "{family:?} must be refused, not return constants"
        );
    }
}

/// Regression: `secure_transmission` returned `vec![0u8; 256]` and
/// `vec![0u8; 64]` regardless of input.
#[tokio::test]
async fn post_quantum_transmission_is_real() {
    let mut rng = TestRng::seeded(50);
    let engine =
        PostQuantumCryptography::new_with_rng(PostQuantumConfig::default(), &mut rng).expect("pq");

    let data = PrivateData {
        mpc_shares: vec![SecretShare {
            share_data: vec![0x11; 64],
            share_id: 1,
            client_id: "c".to_string(),
        }],
        encrypted_shares: None,
        zk_proof: ZKProof {
            proof_data: vec![0x22; 128],
            verification_key: vec![0x33; 32],
        },
    };

    let secured = engine.secure_transmission(&data).await.expect("secure");
    assert!(secured.encrypted_payload.iter().any(|b| *b != 0));
    assert!(secured.quantum_signature.iter().any(|b| *b != 0));
    // The old code produced exactly these lengths of zeros.
    assert_ne!(secured.encrypted_payload, vec![0u8; 256]);
    assert_ne!(secured.quantum_signature, vec![0u8; 64]);
    // The payload must not contain the share verbatim.
    assert!(!secured.encrypted_payload.windows(64).any(|w| w == [0x11u8; 64].as_slice()));

    // And it must open back to the same bytes.
    let opened = engine.open_transmission(&secured).expect("open");
    assert_eq!(opened, PostQuantumCryptography::serialize(&data));
}

/// A tampered payload or signature must be rejected.
#[tokio::test]
async fn post_quantum_transmission_rejects_tampering() {
    let mut rng = TestRng::seeded(51);
    let engine =
        PostQuantumCryptography::new_with_rng(PostQuantumConfig::default(), &mut rng).expect("pq");
    let data = PrivateData {
        mpc_shares: vec![],
        encrypted_shares: None,
        zk_proof: ZKProof {
            proof_data: vec![1, 2, 3],
            verification_key: vec![],
        },
    };
    let secured = engine.secure_transmission(&data).await.expect("secure");

    let mut bad_signature = secured.clone();
    bad_signature.quantum_signature[0] ^= 0x01;
    assert!(engine.open_transmission(&bad_signature).is_err());

    let mut bad_payload = secured.clone();
    let last = bad_payload.encrypted_payload.len() - 1;
    bad_payload.encrypted_payload[last] ^= 0x01;
    assert!(engine.open_transmission(&bad_payload).is_err());
}

#[test]
fn post_quantum_refuses_unimplemented_algorithms() {
    for (kem, signature) in [
        (KEMAlgorithm::NTRU, SignatureAlgorithm::MlDsa65),
        (KEMAlgorithm::SABER, SignatureAlgorithm::MlDsa65),
        (KEMAlgorithm::FrodoKEM, SignatureAlgorithm::MlDsa65),
        (KEMAlgorithm::MlKem768, SignatureAlgorithm::Falcon),
        (KEMAlgorithm::MlKem768, SignatureAlgorithm::Rainbow),
    ] {
        let mut config = PostQuantumConfig::default();
        config.kem = kem.clone();
        config.signature = signature.clone();
        let mut rng = TestRng::seeded(52);
        assert!(
            PostQuantumCryptography::new_with_rng(config, &mut rng).is_err(),
            "({kem:?}, {signature:?}) must be refused"
        );
    }
}

/// Regression: `allocate_budget` returned a fixed `epsilon: 0.1` forever, so
/// the accounting never actually limited anything.
#[tokio::test]
async fn budget_allocation_depletes_and_eventually_refuses() {
    let mut config = AdaptiveBudgetingConfig::default();
    config.initial_budget = 1.0;
    config.fairness_constraints.max_budget_per_client = 10.0;
    let engine = AdaptivePrivacyBudgeting::new(config).expect("budgeting");

    let cost = PrivacyCost {
        differential_privacy_cost: 0.4,
        computational_cost: 0,
        communication_cost: 0,
        sample_count: 0,
        statistic_count: 0,
    };

    assert_eq!(engine.remaining_budget("client_a"), 1.0);
    let first = engine.allocate_budget("client_a", &cost).await.expect("first");
    assert!((first.differential_privacy.epsilon - 0.4).abs() < 1e-12);
    assert!((engine.remaining_budget("client_a") - 0.6).abs() < 1e-12);

    engine.allocate_budget("client_a", &cost).await.expect("second");
    assert!((engine.remaining_budget("client_a") - 0.2).abs() < 1e-12);

    // The third request exceeds what is left and must be refused.
    assert!(
        engine.allocate_budget("client_a", &cost).await.is_err(),
        "an exhausted budget must be refused, not silently re-granted"
    );

    // A different client still has its own full budget.
    assert_eq!(engine.remaining_budget("client_b"), 1.0);
    assert!(engine.allocate_budget("client_b", &cost).await.is_ok());
}

/// The allocation must track the requested cost, not a constant.
#[tokio::test]
async fn budget_allocation_tracks_the_requested_cost() {
    let engine =
        AdaptivePrivacyBudgeting::new(AdaptiveBudgetingConfig::default()).expect("budgeting");
    for requested in [0.05f64, 0.25, 0.5] {
        let cost = PrivacyCost {
            differential_privacy_cost: requested,
            computational_cost: 0,
            communication_cost: 0,
            sample_count: 0,
            statistic_count: 0,
        };
        let allocation = engine
            .allocate_budget(&format!("client_{requested}"), &cost)
            .await
            .expect("allocate");
        assert!(
            (allocation.differential_privacy.epsilon - requested).abs() < 1e-12,
            "requested {requested}, got {}",
            allocation.differential_privacy.epsilon
        );
    }
}

#[tokio::test]
async fn budget_allocation_rejects_invalid_costs() {
    let engine =
        AdaptivePrivacyBudgeting::new(AdaptiveBudgetingConfig::default()).expect("budgeting");
    for requested in [0.0f64, -1.0, f64::NAN] {
        let cost = PrivacyCost {
            differential_privacy_cost: requested,
            computational_cost: 0,
            communication_cost: 0,
            sample_count: 0,
            statistic_count: 0,
        };
        assert!(
            engine.allocate_budget("c", &cost).await.is_err(),
            "{requested}"
        );
    }
}

/// Regression: `get_metrics` returned hardcoded constants. With nothing
/// measured it must say so; after measuring it must report the real numbers.
#[tokio::test]
async fn performance_metrics_are_measured_or_explicitly_unavailable() {
    let monitor = PrivacyPerformanceMonitor::new().expect("monitor");
    assert_eq!(monitor.sample_count(), 0);

    match monitor.get_metrics().await.expect("metrics") {
        PrivacyPerformanceMetrics::NotAvailable { reason } => {
            assert!(reason.contains("no privacy operations"), "{reason}");
        },
        PrivacyPerformanceMetrics::Measured { .. } => {
            panic!("must not report measurements before anything was measured")
        },
    }

    monitor.record(Duration::from_millis(10));
    monitor.record(Duration::from_millis(30));
    monitor.record(Duration::from_millis(20));

    match monitor.get_metrics().await.expect("metrics") {
        PrivacyPerformanceMetrics::Measured {
            average_execution_time,
            slowest,
            fastest,
            throughput_per_second,
            sample_count,
        } => {
            assert_eq!(sample_count, 3);
            assert_eq!(average_execution_time, Duration::from_millis(20));
            assert_eq!(slowest, Duration::from_millis(30));
            assert_eq!(fastest, Duration::from_millis(10));
            // 1 / 0.02s = 50 ops/sec.
            assert!((throughput_per_second - 50.0).abs() < 1e-6);
            // The old code always reported exactly 500ms / 100 ops per second.
            assert_ne!(average_execution_time, Duration::from_millis(500));
        },
        PrivacyPerformanceMetrics::NotAvailable { reason } => {
            panic!("metrics should be available after recording: {reason}")
        },
    }
}

/// End-to-end: the orchestrated round must produce real artifacts and a real
/// measured execution time.
#[tokio::test]
async fn private_federated_round_produces_real_artifacts() {
    let engine = AdvancedPrivacyMechanisms::new(AdvancedPrivacyConfig::default()).expect("engine");
    let update = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]).expect("tensor");

    let result = engine
        .execute_private_federated_round(&update, "client_e2e")
        .await
        .expect("round");

    // Nothing in the result may be a zero-filled placeholder.
    assert!(
        result.secured_data.encrypted_payload.iter().any(|b| *b != 0),
        "encrypted payload is all zeros"
    );
    assert!(
        result.secured_data.quantum_signature.iter().any(|b| *b != 0),
        "quantum signature is all zeros"
    );
    assert!(
        result.zk_proof.proof_data.iter().any(|b| *b != 0),
        "zk proof is all zeros"
    );
    assert_ne!(result.zk_proof.proof_data, vec![0u8; 128]);

    // The execution time is measured, so it must be non-zero.
    assert!(result.execution_time > Duration::ZERO);
    assert!(result.privacy_guarantees.epsilon > 0.0);
}

/// Regression: the Laplace path calibrated to the `L2` clipping norm directly,
/// but the Laplace mechanism needs the `L1` sensitivity. For a `d`-dimensional
/// vector `||x||_1 <= sqrt(d) * ||x||_2`, so using the `L2` bound under-noised
/// the output by a factor of `sqrt(d)` and silently broke the epsilon-DP
/// guarantee. The noise scale must grow as `sqrt(d)`.
#[test]
fn pure_epsilon_path_calibrates_to_the_l1_sensitivity() {
    let clipping_norm = 1.0f64;
    let engine = AdvancedDifferentialPrivacy::with_clipping_norm(Default::default(), clipping_norm)
        .expect("engine");
    let budget = DifferentialPrivacyBudget {
        epsilon: 1.0,
        delta: 0.0, // pure epsilon-DP => Laplace
        renyi_alpha: 2.0,
    };

    // Empirical noise standard deviation at two dimensionalities. Laplace with
    // scale b has variance 2b^2, and b = sqrt(d) * clipping_norm / epsilon, so
    // the standard deviation must scale as sqrt(d).
    let empirical_std = |dimension: usize, seed: u8| -> f64 {
        let zeros = Tensor::from_vec(vec![0.0f32; dimension], &[dimension]).expect("tensor");
        let mut rng = TestRng::seeded(seed);
        let mut all = Vec::new();
        for _ in 0..40 {
            let noised = engine
                .privatize_update_with_rng(&zeros, &budget, &mut rng)
                .expect("privatize")
                .to_vec_f32()
                .expect("vec");
            all.extend(noised.into_iter().map(f64::from));
        }
        let n = all.len() as f64;
        let mean = all.iter().sum::<f64>() / n;
        (all.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
    };

    let std_4 = empirical_std(4, 60);
    let std_64 = empirical_std(64, 61);

    // Expected ratio is sqrt(64/4) = 4. With the old (buggy) calibration the
    // scale was dimension-independent and this ratio would be 1.
    let ratio = std_64 / std_4;
    assert!(
        (ratio - 4.0).abs() < 1.0,
        "noise std should scale as sqrt(d): ratio {ratio} should be near 4 (was 1 with the \
         L2-as-L1 bug)"
    );

    // And the absolute scale must match the formula for d = 4:
    // b = sqrt(4) * 1.0 / 1.0 = 2, std = sqrt(2) * b = 2.828...
    let expected_std_4 = (2.0f64).sqrt() * (4.0f64).sqrt() * clipping_norm / budget.epsilon;
    assert!(
        (std_4 - expected_std_4).abs() < 0.6,
        "std {std_4} should be near {expected_std_4}"
    );
}

/// Regression: [`PrivacyPerformanceMonitor`] existed but nothing in the
/// orchestrator ever fed it, so [`AdvancedPrivacyMechanisms::get_privacy_report`]
/// reported `NotAvailable` forever no matter how much work had been done — the
/// accounting was structurally honest but permanently empty. A completed round
/// and a completed analytics call must each contribute one real measurement.
#[tokio::test]
async fn privacy_report_timings_come_from_completed_operations() {
    let engine = AdvancedPrivacyMechanisms::new(AdvancedPrivacyConfig::default()).expect("engine");

    // Before anything has run there is genuinely nothing to report.
    match engine.get_privacy_report().await.expect("report").performance_metrics {
        PrivacyPerformanceMetrics::NotAvailable { reason } => {
            assert!(reason.contains("no privacy operations"), "{reason}");
        },
        PrivacyPerformanceMetrics::Measured { .. } => {
            panic!("metrics must not be available before any operation has run")
        },
    }

    let update = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]).expect("tensor");
    let round = engine
        .execute_private_federated_round(&update, "client_metrics")
        .await
        .expect("round");

    let after_round = engine.get_privacy_report().await.expect("report");
    match after_round.performance_metrics {
        PrivacyPerformanceMetrics::Measured {
            sample_count,
            average_execution_time,
            slowest,
            fastest,
            ..
        } => {
            assert_eq!(
                sample_count, 1,
                "the completed round must be the one sample"
            );
            // The single sample is exactly the round's own measured duration.
            assert_eq!(average_execution_time, round.execution_time);
            assert_eq!(slowest, round.execution_time);
            assert_eq!(fastest, round.execution_time);
            assert!(average_execution_time > Duration::ZERO);
            // The old hardcoded value.
            assert_ne!(average_execution_time, Duration::from_millis(500));
        },
        PrivacyPerformanceMetrics::NotAvailable { reason } => {
            panic!("a completed round must be measured, got: {reason}")
        },
    }

    // Analytics is the other orchestrated entry point and must also count.
    let samples = vec![Tensor::from_vec(vec![0.4, 0.5, 0.6], &[3]).expect("tensor")];
    engine
        .compute_private_analytics(&samples, &[AnalyticsType::BasicStatistics])
        .await
        .expect("analytics");

    match engine.get_privacy_report().await.expect("report").performance_metrics {
        PrivacyPerformanceMetrics::Measured { sample_count, .. } => {
            assert_eq!(sample_count, 2, "the analytics call must add a sample");
        },
        PrivacyPerformanceMetrics::NotAvailable { reason } => {
            panic!("measurements must persist across calls, got: {reason}")
        },
    }
}

/// A *failed* analytics call must not be recorded: timing a partial run would
/// contaminate the statistics with work that never completed.
#[tokio::test]
async fn failed_analytics_calls_are_not_recorded() {
    let engine = AdvancedPrivacyMechanisms::new(AdvancedPrivacyConfig::default()).expect("engine");

    // Empty input is rejected by the analytics engine.
    let error = engine
        .compute_private_analytics(&[], &[AnalyticsType::BasicStatistics])
        .await
        .expect_err("empty analytics input must be refused");
    assert!(
        error.to_string().contains("empty"),
        "unexpected error: {error}"
    );

    match engine.get_privacy_report().await.expect("report").performance_metrics {
        PrivacyPerformanceMetrics::NotAvailable { .. } => {},
        PrivacyPerformanceMetrics::Measured { sample_count, .. } => {
            panic!("a failed call must not be measured, but {sample_count} samples were recorded")
        },
    }
}
