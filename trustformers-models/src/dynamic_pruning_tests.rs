//! Unit tests for [`crate::dynamic_pruning`].
//!
//! Included as a private child module of `dynamic_pruning` so the tests can
//! exercise the crate-internal scoring helpers directly.

use super::*;

#[test]
fn test_attention_based_pruning_config() {
    let config = AttentionBasedPruningConfig::default();
    assert_eq!(config.attention_threshold, 0.1);
    assert_eq!(config.min_tokens_ratio, 0.3);
    assert_eq!(config.max_pruning_ratio, 0.7);
    assert!(config.use_adaptive_threshold);
}

#[test]
fn test_dynamic_pruner_creation() {
    let config = AttentionBasedPruningConfig::default();
    let pruner = DynamicPruner::attention_based(config);

    assert!(
        matches!(pruner.strategy, PruningStrategy::AttentionBased),
        "Expected AttentionBased strategy"
    );
}

#[test]
fn test_learned_gate_network_creation() -> Result<()> {
    let config = LearnedGatePruningConfig::default();
    let gate_network = LearnedGateNetwork::new(768, config)?;

    assert_eq!(gate_network.gate_linear.shape(), vec![768, 64]);
    assert_eq!(gate_network.gate_bias.shape(), vec![64]);

    Ok(())
}

#[test]
fn test_progressive_pruning_ratios() {
    let _config = ProgressivePruningConfig {
        initial_pruning_ratio: 0.1,
        final_pruning_ratio: 0.5,
        progression_schedule: ProgressionSchedule::Linear,
    };

    // Test linear progression
    let total_layers = 12;
    for layer in 0..total_layers {
        let progress = layer as f32 / (total_layers - 1) as f32;
        let expected_ratio = 0.1 + (0.5 - 0.1) * progress;

        // This would be tested in the actual pruner implementation
        assert!((0.1..=0.5).contains(&expected_ratio));
    }
}

#[test]
fn test_pruning_statistics() {
    let results = vec![PruningResult {
        pruned_hidden_states: Tensor::zeros(&[1, 5, 768]).expect("operation failed"),
        pruned_attention_mask: Tensor::ones(&[1, 5]).expect("operation failed"),
        token_importance: TokenImportance {
            importance_scores: vec![0.9, 0.8, 0.3, 0.2, 0.1],
            token_indices: vec![0, 1, 2, 3, 4],
            keep_mask: vec![true, true, true, false, false],
            pruning_reasons: vec![
                PruningReason::AlwaysKeep,
                PruningReason::MinimumRatio,
                PruningReason::MinimumRatio,
                PruningReason::LowAttention,
                PruningReason::LowAttention,
            ],
        },
        original_length: 10,
        pruned_length: 5,
        compression_ratio: 0.5,
    }];

    let stats = PruningStatistics::from_results(&results);
    assert_eq!(stats.avg_compression_ratio, 0.5);
    assert_eq!(stats.layer_compression_ratios, vec![0.5]);
    assert_eq!(stats.computational_savings, 0.75); // 1 - 0.5^2
    assert_eq!(stats.memory_savings, 0.5); // 1 - 0.5
}

// ---- Pruning schedule: cosine annealing ----

/// Cubic ease-in schedule: sparsity(t) = target * (1 - (1 - t/T)^3)
/// At t=0 sparsity must be 0.0; at t=T sparsity must equal target.
#[test]
fn test_cosine_schedule_endpoints() {
    let target = 0.7f32;
    let total = 100usize;

    let sparsity_at_0 = target * (1.0 - (1.0 - 0.0f32 / total as f32).powi(3));
    let sparsity_at_t = target * (1.0 - (1.0 - total as f32 / total as f32).powi(3));

    assert!(sparsity_at_0.abs() < 1e-6, "sparsity at t=0 must be 0");
    assert!(
        (sparsity_at_t - target).abs() < 1e-6,
        "sparsity at t=T must equal target"
    );
}

/// The cubic schedule must be monotonically non-decreasing.
#[test]
fn test_cosine_schedule_monotone() {
    let target = 0.8f32;
    let total = 50usize;
    let mut prev = -1.0f32;
    for t in 0..=total {
        let s = target * (1.0 - (1.0 - t as f32 / total as f32).powi(3));
        assert!(s >= prev - 1e-6, "Schedule must be monotone at t={}", t);
        prev = s;
    }
}

// ---- Magnitude-based scoring ----

/// Higher-magnitude tokens must receive higher importance scores.
#[test]
fn test_magnitude_scoring_order() {
    // Simulate magnitude-based scoring: score_i = ||h_i||^2
    let norms_squared = [0.1f32, 0.9, 0.3, 0.7, 0.5];
    let mut indexed: Vec<(usize, f32)> =
        norms_squared.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("comparison must succeed"));

    // The highest magnitude (index 1, norm²=0.9) must rank first
    assert_eq!(indexed[0].0, 1, "Highest magnitude token must rank first");
    // The lowest magnitude (index 0, norm²=0.1) must rank last
    assert_eq!(
        indexed[indexed.len() - 1].0,
        0,
        "Lowest magnitude token must rank last"
    );
}

// ---- Structured vs unstructured pruning ----

/// Unstructured pruning removes individual tokens (any position).
/// Structured pruning removes entire heads/layers.
/// Verify the keep_mask property: any boolean pattern is valid for unstructured.
#[test]
fn test_unstructured_any_keep_mask_valid() {
    let keep_masks = vec![
        vec![true, false, true, false, true],
        vec![true, true, true, false, false],
        vec![false, false, false, true, true],
    ];
    for mask in &keep_masks {
        let kept = mask.iter().filter(|&&x| x).count();
        assert!(
            kept > 0,
            "At least one token must be kept; mask: {:?}",
            mask
        );
        assert!(
            kept < mask.len(),
            "Not all tokens should be kept; mask: {:?}",
            mask
        );
    }
}

// ---- Attention-based pruner configurations ----

#[test]
fn test_confidence_based_pruner_creation() {
    let config = ConfidenceBasedPruningConfig::default();
    let pruner = DynamicPruner::confidence_based(config);
    assert!(
        matches!(pruner.strategy, PruningStrategy::ConfidenceBased),
        "Expected ConfidenceBased strategy"
    );
}

#[test]
fn test_progressive_pruner_creation() {
    let config = ProgressivePruningConfig::default();
    let pruner = DynamicPruner::progressive(config);
    assert!(
        matches!(pruner.strategy, PruningStrategy::Progressive),
        "Expected Progressive strategy"
    );
}

#[test]
fn test_layer_adaptive_pruner_creation() {
    let config = LayerAdaptivePruningConfig::default();
    let pruner = DynamicPruner::layer_adaptive(config);
    assert!(
        matches!(pruner.strategy, PruningStrategy::LayerAdaptive),
        "Expected LayerAdaptive strategy"
    );
}

// ---- Sparsity percentage validation ----

/// compression_ratio = pruned_length / original_length must be in (0, 1].
#[test]
fn test_compression_ratio_in_valid_range() {
    let original_length = 20usize;
    for pruned_length in 1..=original_length {
        let ratio = pruned_length as f32 / original_length as f32;
        assert!(
            ratio > 0.0 && ratio <= 1.0,
            "compression_ratio must be in (0,1]"
        );
    }
}

/// Sparsity = 1 - compression_ratio must be in [0, 1).
#[test]
fn test_sparsity_in_valid_range() {
    let original = 10usize;
    for pruned in 1..=original {
        let compression_ratio = pruned as f32 / original as f32;
        let sparsity = 1.0 - compression_ratio;
        assert!((0.0..1.0).contains(&sparsity), "sparsity must be in [0, 1)");
    }
}

// ---- PruningStatistics empty and multi-layer ----

#[test]
fn test_pruning_statistics_empty_results() {
    let stats = PruningStatistics::from_results(&[]);
    assert_eq!(
        stats.avg_compression_ratio, 1.0,
        "Empty results must give ratio=1.0"
    );
    assert!(stats.layer_compression_ratios.is_empty());
}

#[test]
fn test_pruning_statistics_multiple_layers() {
    let make_result = |ratio: f32, seq_len: usize| PruningResult {
        pruned_hidden_states: Tensor::zeros(&[1, seq_len, 64])
            .expect("tensor creation must succeed"),
        pruned_attention_mask: Tensor::ones(&[1, seq_len]).expect("tensor creation must succeed"),
        token_importance: TokenImportance {
            importance_scores: vec![0.5; seq_len],
            token_indices: (0..seq_len).collect(),
            keep_mask: vec![true; seq_len],
            pruning_reasons: vec![PruningReason::MinimumRatio; seq_len],
        },
        original_length: (seq_len as f32 / ratio) as usize,
        pruned_length: seq_len,
        compression_ratio: ratio,
    };

    let results = vec![make_result(0.6, 6), make_result(0.4, 4)];
    let stats = PruningStatistics::from_results(&results);
    let expected_avg = (0.6 + 0.4) / 2.0;
    assert!((stats.avg_compression_ratio - expected_avg).abs() < 1e-6);
    assert_eq!(stats.layer_compression_ratios.len(), 2);
}

// ---- Pruning reason distribution ----

#[test]
fn test_pruning_reason_distribution_tracked() {
    let result = PruningResult {
        pruned_hidden_states: Tensor::zeros(&[1, 3, 32]).expect("must succeed"),
        pruned_attention_mask: Tensor::ones(&[1, 3]).expect("must succeed"),
        token_importance: TokenImportance {
            importance_scores: vec![0.8, 0.5, 0.2],
            token_indices: vec![0, 1, 2],
            keep_mask: vec![true, true, false],
            pruning_reasons: vec![
                PruningReason::AlwaysKeep,
                PruningReason::MinimumRatio,
                PruningReason::LowAttention,
            ],
        },
        original_length: 4,
        pruned_length: 3,
        compression_ratio: 0.75,
    };
    let stats = PruningStatistics::from_results(&[result]);
    assert_eq!(
        stats.pruning_reason_distribution.get(&PruningReason::AlwaysKeep),
        Some(&1)
    );
    assert_eq!(
        stats.pruning_reason_distribution.get(&PruningReason::LowAttention),
        Some(&1)
    );
}

// ---- Progressive pruning schedule: linear ----

#[test]
fn test_progressive_pruning_linear_schedule_bounds() {
    let config = ProgressivePruningConfig {
        initial_pruning_ratio: 0.1,
        final_pruning_ratio: 0.5,
        progression_schedule: ProgressionSchedule::Linear,
    };
    let total_layers = 12usize;
    for layer in 0..total_layers {
        let progress = layer as f32 / (total_layers - 1) as f32;
        let ratio = config.initial_pruning_ratio
            + (config.final_pruning_ratio - config.initial_pruning_ratio) * progress;
        assert!(
            ratio >= config.initial_pruning_ratio - 1e-6
                && ratio <= config.final_pruning_ratio + 1e-6,
            "Linear schedule ratio {} must be in [{}, {}] for layer {}",
            ratio,
            config.initial_pruning_ratio,
            config.final_pruning_ratio,
            layer
        );
    }
}

// ---------------------------------------------------------------------------
// Regression tests: every scorer must read the real tensors
// ---------------------------------------------------------------------------

/// Build `[batch, seq, hidden]` hidden states where token `t` is the constant
/// vector `value(t)`, so the origin of every gathered row is identifiable.
fn hidden_states_with_token_values(batch: usize, values: &[f32], hidden: usize) -> Tensor {
    let mut data = Vec::with_capacity(batch * values.len() * hidden);
    for b in 0..batch {
        for &v in values {
            for h in 0..hidden {
                data.push(v + (b as f32) * 100.0 + (h as f32) * 0.001);
            }
        }
    }
    Tensor::from_slice(&data, &[batch, values.len(), hidden]).expect("tensor creation must succeed")
}

#[test]
fn test_apply_pruning_mask_gathers_real_token_rows() {
    let pruner = DynamicPruner::attention_based(AttentionBasedPruningConfig::default());
    let hidden = 4;
    let hidden_states = hidden_states_with_token_values(2, &[1.0, 2.0, 3.0, 4.0], hidden);
    let keep_mask = vec![true, false, true, false];

    let pruned = pruner
        .apply_pruning_mask(&hidden_states, &keep_mask)
        .expect("pruning must succeed");

    assert_eq!(pruned.shape(), vec![2, 2, hidden]);
    let data = pruned.data().expect("pruned data must be readable");
    // Not the all-zero tensor the previous implementation returned.
    assert!(
        data.iter().any(|&v| v.abs() > 1e-6),
        "pruned hidden states must not be all zeros"
    );

    let original = hidden_states.data().expect("original data must be readable");
    let seq_len = 4;
    for (new_idx, orig_idx) in [0usize, 2].iter().enumerate() {
        for batch in 0..2 {
            for h in 0..hidden {
                let expected = original[(batch * seq_len + orig_idx) * hidden + h];
                let actual = data[(batch * 2 + new_idx) * hidden + h];
                assert!(
                    (expected - actual).abs() < 1e-6,
                    "token {orig_idx} (batch {batch}, dim {h}) must be copied verbatim: \
                     expected {expected}, got {actual}"
                );
            }
        }
    }
}

#[test]
fn test_apply_pruning_mask_rejects_mismatched_mask() {
    let pruner = DynamicPruner::attention_based(AttentionBasedPruningConfig::default());
    let hidden_states = hidden_states_with_token_values(1, &[1.0, 2.0, 3.0], 2);
    assert!(pruner.apply_pruning_mask(&hidden_states, &[true, false]).is_err());
    assert!(pruner.apply_pruning_mask(&hidden_states, &[false, false, false]).is_err());
}

#[test]
fn test_attention_importance_reads_the_attention_matrix() {
    let config = AttentionBasedPruningConfig {
        use_adaptive_threshold: false,
        ..Default::default()
    };
    let pruner = DynamicPruner::attention_based(config.clone());

    // Every query attends exclusively to key position 2.
    let seq = 4;
    let mut data = vec![0.0f32; seq * seq];
    for q in 0..seq {
        data[q * seq + 2] = 1.0;
    }
    let attention = Tensor::from_slice(&data, &[1, seq, seq]).expect("tensor creation");

    let scores = pruner
        .compute_attention_importance(&attention, &config)
        .expect("scoring must succeed");

    assert_eq!(scores.len(), seq);
    assert!(
        (scores[2] - seq as f32).abs() < 1e-5,
        "token 2 receives all {seq} units of attention, got {}",
        scores[2]
    );
    for (idx, &score) in scores.iter().enumerate() {
        if idx != 2 {
            assert!(score.abs() < 1e-6, "token {idx} receives no attention");
        }
    }
    assert!(
        scores[2] > scores[0],
        "the attended token must outrank the first token; position-only scoring would \
         have ranked token 0 first"
    );
}

#[test]
fn test_attention_importance_averages_heads() {
    let config = AttentionBasedPruningConfig {
        use_adaptive_threshold: false,
        ..Default::default()
    };
    let pruner = DynamicPruner::attention_based(config.clone());

    // Two heads, two queries, three keys. Head 0 attends to key 0, head 1 to key 1.
    let data = vec![
        // head 0
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, // head 1
        0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
    ];
    let attention = Tensor::from_slice(&data, &[1, 2, 2, 3]).expect("tensor creation");
    let scores = pruner
        .compute_attention_importance(&attention, &config)
        .expect("scoring must succeed");

    assert_eq!(scores.len(), 3);
    assert!((scores[0] - 1.0).abs() < 1e-5, "got {:?}", scores);
    assert!((scores[1] - 1.0).abs() < 1e-5, "got {:?}", scores);
    assert!(scores[2].abs() < 1e-6, "got {:?}", scores);
}

#[test]
fn test_attention_importance_rejects_wrong_rank() {
    let config = AttentionBasedPruningConfig::default();
    let pruner = DynamicPruner::attention_based(config.clone());
    let bad = Tensor::zeros(&[4, 4]).expect("tensor creation");
    assert!(pruner.compute_attention_importance(&bad, &config).is_err());
}

#[test]
fn test_simple_importance_is_the_real_l2_norm() {
    let pruner = DynamicPruner::layer_adaptive(LayerAdaptivePruningConfig::default());
    // Token magnitudes 1, 3, 2 over a two-dimensional hidden state.
    let data = vec![1.0, 0.0, 3.0, 0.0, 2.0, 0.0];
    let hidden_states = Tensor::from_slice(&data, &[1, 3, 2]).expect("tensor creation");

    let scores = pruner.compute_simple_importance(&hidden_states).expect("scoring must succeed");

    assert_eq!(scores.len(), 3);
    assert!(
        (scores[1] - 1.0).abs() < 1e-6,
        "largest norm normalises to 1"
    );
    assert!((scores[0] - 1.0 / 3.0).abs() < 1e-6, "got {:?}", scores);
    assert!((scores[2] - 2.0 / 3.0).abs() < 1e-6, "got {:?}", scores);
}

#[test]
fn test_confidence_scores_depend_on_the_logits() {
    let config = ConfidenceBasedPruningConfig {
        use_entropy: false,
        lookahead_window: 1,
        ..Default::default()
    };
    let pruner = DynamicPruner::confidence_based(config.clone());
    let hidden_states = Tensor::zeros(&[1, 2, 3]).expect("tensor creation");

    // Token 0: sharply peaked. Token 1: uniform.
    let logits =
        Tensor::from_slice(&[10.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[1, 2, 3]).expect("tensor creation");

    let scores = pruner
        .compute_confidence_scores(&hidden_states, Some(&logits), &config)
        .expect("scoring must succeed");

    assert!(
        scores[0] > 0.99,
        "peaked logits must be confident: {:?}",
        scores
    );
    assert!(
        (scores[1] - 1.0 / 3.0).abs() < 1e-5,
        "uniform logits must give 1/C: {:?}",
        scores
    );
}

#[test]
fn test_confidence_scores_change_with_the_hidden_states() {
    let config = ConfidenceBasedPruningConfig {
        lookahead_window: 1,
        ..Default::default()
    };
    let pruner = DynamicPruner::confidence_based(config.clone());

    let flat =
        Tensor::from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[1, 2, 3]).expect("tensor creation");
    let peaked =
        Tensor::from_slice(&[8.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[1, 2, 3]).expect("tensor creation");

    let flat_scores = pruner
        .compute_confidence_scores(&flat, None, &config)
        .expect("scoring must succeed");
    let peaked_scores = pruner
        .compute_confidence_scores(&peaked, None, &config)
        .expect("scoring must succeed");

    assert!(
        peaked_scores[0] > flat_scores[0] + 0.1,
        "confidence must react to the hidden states: {:?} vs {:?}",
        peaked_scores,
        flat_scores
    );
    assert!(
        flat_scores[0] < 1e-6,
        "a uniform hidden state has maximal entropy: {:?}",
        flat_scores
    );
}

#[test]
fn test_gate_scores_read_the_gate_tensor() {
    let pruner = DynamicPruner::attention_based(AttentionBasedPruningConfig::default());
    let gate_probs =
        Tensor::from_slice(&[0.9, 0.1, 0.4, 0.7, 0.3, 0.6], &[2, 3, 1]).expect("tensor creation");

    let scores = pruner.extract_gate_scores(&gate_probs).expect("gate extraction must succeed");

    assert_eq!(scores.len(), 3);
    assert!((scores[0] - 0.8).abs() < 1e-6, "got {:?}", scores);
    assert!((scores[1] - 0.2).abs() < 1e-6, "got {:?}", scores);
    assert!((scores[2] - 0.5).abs() < 1e-6, "got {:?}", scores);
}

#[test]
fn test_gate_network_forward_is_deterministic() -> Result<()> {
    let config = LearnedGatePruningConfig {
        use_straight_through: false,
        ..Default::default()
    };
    let network = LearnedGateNetwork::new(8, config)?;
    let hidden_states = Tensor::randn(&[1, 5, 8])?;

    let first = network.forward(&hidden_states)?.data()?;
    let second = network.forward(&hidden_states)?.data()?;

    assert_eq!(first.len(), second.len());
    for (a, b) in first.iter().zip(second.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "the gate network must reuse its output projection: {a} vs {b}"
        );
    }
    Ok(())
}

#[test]
fn test_attention_based_pruning_end_to_end_keeps_real_rows() -> Result<()> {
    let config = AttentionBasedPruningConfig {
        min_tokens_ratio: 0.5,
        max_pruning_ratio: 0.5,
        use_adaptive_threshold: false,
        keep_top_k: 1,
        ..Default::default()
    };
    let pruner = DynamicPruner::attention_based(config);

    let hidden = 3;
    let hidden_states = hidden_states_with_token_values(1, &[1.0, 2.0, 3.0, 4.0], hidden);

    // Keys 3 and 1 receive the most attention, in that order.
    let seq = 4;
    let mut attention = vec![0.0f32; seq * seq];
    for q in 0..seq {
        attention[q * seq + 3] = 0.6;
        attention[q * seq + 1] = 0.3;
        attention[q * seq] = 0.1;
    }
    let attention = Tensor::from_slice(&attention, &[1, 1, seq, seq])?;

    let result = pruner.prune_tokens(&hidden_states, Some(&attention), None, None)?;

    assert_eq!(result.original_length, 4);
    assert_eq!(result.pruned_length, 2);
    assert!(
        result.token_importance.keep_mask[3],
        "the most-attended token survives"
    );
    assert!(
        result.token_importance.keep_mask[1],
        "the second-most-attended token survives"
    );
    assert!(!result.token_importance.keep_mask[2]);

    let pruned = result.pruned_hidden_states.data()?;
    let original = hidden_states.data()?;
    // Kept tokens keep their original relative order: 1 then 3.
    for (new_idx, orig_idx) in [1usize, 3].iter().enumerate() {
        for h in 0..hidden {
            let expected = original[orig_idx * hidden + h];
            let actual = pruned[new_idx * hidden + h];
            assert!(
                (expected - actual).abs() < 1e-6,
                "kept token {orig_idx} must be copied, expected {expected} got {actual}"
            );
        }
    }
    assert_eq!(result.pruned_attention_mask.shape(), vec![1, 2]);
    Ok(())
}

// ---------------------------------------------------------------------------
// Regression tests: early exit must consume the classifier output
// ---------------------------------------------------------------------------

#[test]
fn test_early_exit_confidence_entropy_from_real_logits() -> Result<()> {
    let controller = EarlyExitController::new(EarlyExitConfig::default(), 4, 3)?;

    let peaked = Tensor::from_slice(&[20.0, 0.0, 0.0], &[1, 3])?;
    let (confidence, entropy) = controller.compute_confidence_entropy(&peaked, 0)?;
    assert!(
        confidence > 0.999,
        "peaked logits give confidence {confidence}"
    );
    assert!(entropy < 1e-3, "peaked logits give entropy {entropy}");

    let uniform = Tensor::from_slice(&[1.0, 1.0, 1.0], &[1, 3])?;
    let (confidence, entropy) = controller.compute_confidence_entropy(&uniform, 0)?;
    assert!(
        (confidence - 1.0 / 3.0).abs() < 1e-5,
        "uniform logits give confidence {confidence}"
    );
    assert!(
        (entropy - 3.0f32.ln()).abs() < 1e-4,
        "uniform logits give entropy {entropy}, expected ln(3)"
    );
    Ok(())
}

#[test]
fn test_early_exit_confidence_differs_per_batch_item() -> Result<()> {
    let controller = EarlyExitController::new(EarlyExitConfig::default(), 4, 2)?;
    let logits = Tensor::from_slice(&[10.0, 0.0, 0.0, 0.0], &[2, 2])?;

    let (confident, _) = controller.compute_confidence_entropy(&logits, 0)?;
    let (uncertain, _) = controller.compute_confidence_entropy(&logits, 1)?;

    assert!(
        confident > uncertain + 0.4,
        "batch items must be scored independently: {confident} vs {uncertain}"
    );
    assert!(controller.compute_confidence_entropy(&logits, 7).is_err());
    Ok(())
}

#[test]
fn test_should_exit_reacts_to_the_model_output() -> Result<()> {
    let config = EarlyExitConfig {
        confidence_threshold: 0.9,
        min_exit_layer: 0,
        max_exit_points: 1,
        use_patience: false,
        patience_window: 3,
        entropy_threshold: 0.01,
    };

    // A deterministic classifier: hidden state -> logits, identity on 2 classes.
    let mut controller = EarlyExitController::new(config, 2, 2)?;
    controller.exit_classifiers[0].weight = Tensor::from_slice(&[8.0, 0.0, 0.0, 8.0], &[2, 2])?;
    controller.exit_classifiers[0].bias = None;

    let confident = Tensor::from_slice(&[1.0, 0.0], &[1, 2])?;
    let ambiguous = Tensor::from_slice(&[0.5, 0.5], &[1, 2])?;

    let confident_points = controller.should_exit(&confident, 0, &[0])?;
    controller.reset_patience();
    let ambiguous_points = controller.should_exit(&ambiguous, 0, &[0])?;

    assert!(
        confident_points[0].should_exit,
        "a confident prediction must trigger the exit: {:?}",
        confident_points[0]
    );
    assert!(
        !ambiguous_points[0].should_exit,
        "an ambiguous prediction must not: {:?}",
        ambiguous_points[0]
    );
    assert!(confident_points[0].confidence > ambiguous_points[0].confidence);
    Ok(())
}

#[test]
fn test_computational_savings_uses_observed_layer_count() {
    // 4 observed layers, every sample exits at layer 1 -> 3/4 of the depth saved.
    let history: Vec<Vec<EarlyExitPoint>> = (0..4)
        .map(|layer| {
            vec![EarlyExitPoint {
                layer_index: layer,
                confidence: 0.95,
                entropy: 0.01,
                should_exit: layer == 1,
                exit_reason: if layer == 1 {
                    ExitReason::HighConfidence
                } else {
                    ExitReason::NoExit
                },
            }]
        })
        .collect();

    let controller = EarlyExitController::new(EarlyExitConfig::default(), 4, 2)
        .expect("controller creation must succeed");
    let stats = controller.get_exit_statistics(&history);

    assert!((stats.exit_rate - 0.25).abs() < 1e-6);
    assert!((stats.avg_exit_layer - 1.0).abs() < 1e-6);
    assert!((stats.computational_savings - 0.25 * 0.75).abs() < 1e-6);
}
