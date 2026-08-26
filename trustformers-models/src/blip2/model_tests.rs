//! Unit tests for [`crate::blip2::model`].
//!
//! Split out of `model.rs` to keep that file under the 2000-line limit.

use super::*;
use scirs2_core::random::{SeedableRng, StdRng};

// ── Sampling ───────────────────────────────────────────────────────────────

/// The old `sample_token` ignored `top_p` and returned `argmax`, so every draw
/// from a flat distribution was the same token. Real sampling must visit more
/// than one token.
#[test]
fn test_sampling_is_not_disguised_argmax() {
    let logits = vec![0.0f32; 8];
    let mut rng = StdRng::seed_from_u64(7);
    let mut seen = std::collections::BTreeSet::new();
    for _ in 0..64 {
        seen.insert(sample_token_id(&logits, 1.0, 1.0, &mut rng).expect("sample"));
    }
    assert!(
        seen.len() > 1,
        "a uniform distribution must not always yield the same token, got {seen:?}"
    );
}

/// The nucleus must actually restrict the candidates.
///
/// softmax([2, 1, 0, 0]) ≈ [0.61, 0.22, 0.08, 0.08], so `top_p = 0.6` admits
/// token 0 alone while the tail still carries ~39% of the mass — enough that
/// sampling the full distribution visits it within a few draws. A `top_p` that
/// is accepted and then ignored therefore fails this test.
#[test]
fn test_top_p_restricts_the_candidate_set() {
    let logits = vec![2.0f32, 1.0, 0.0, 0.0];

    let mut nucleus_rng = StdRng::seed_from_u64(11);
    for _ in 0..32 {
        let id = sample_token_id(&logits, 1.0, 0.6, &mut nucleus_rng).expect("sample");
        assert_eq!(id, 0, "top-p 0.6 admits only the leading token");
    }

    // Control: the same logits without a nucleus must reach the tail.
    let mut full_rng = StdRng::seed_from_u64(11);
    let mut seen = std::collections::BTreeSet::new();
    for _ in 0..32 {
        seen.insert(sample_token_id(&logits, 1.0, 1.0, &mut full_rng).expect("sample"));
    }
    assert!(
        seen.len() > 1,
        "without top-p the tail must be reachable, got {seen:?}"
    );
}

/// Temperature must reach the distribution: a near-zero temperature collapses
/// onto the maximum, and `temperature <= 0` is explicit greedy decoding.
#[test]
fn test_temperature_is_applied() {
    let logits = vec![0.0f32, 1.0, 0.5];
    let mut rng = StdRng::seed_from_u64(13);
    for _ in 0..32 {
        assert_eq!(
            sample_token_id(&logits, 0.01, 1.0, &mut rng).expect("cold"),
            1
        );
    }
    assert_eq!(
        sample_token_id(&logits, 0.0, 1.0, &mut rng).expect("greedy"),
        1
    );
}

/// Only the last position of each batch row is decoded, and one id comes back
/// per row.
#[test]
fn test_sample_next_ids_decodes_the_last_position_per_row() {
    // [batch=2, seq=2, vocab=3]; the last position differs per row.
    let logits = Tensor::from_vec(
        vec![
            9.0, 0.0, 0.0, // row 0, position 0 (must be ignored)
            0.0, 9.0, 0.0, // row 0, position 1 → token 1
            0.0, 0.0, 9.0, // row 1, position 0 (must be ignored)
            9.0, 0.0, 0.0, // row 1, position 1 → token 0
        ],
        &[2, 2, 3],
    )
    .expect("logits");
    let mut rng = StdRng::seed_from_u64(17);
    let ids = sample_next_ids(&logits, 0.0, 1.0, &mut rng).expect("ids");
    assert_eq!(ids, vec![1, 0]);
}

#[test]
fn test_sample_next_ids_rejects_malformed_logits() {
    let mut rng = StdRng::seed_from_u64(19);
    let flat = Tensor::from_vec(vec![0.0f32; 4], &[4]).expect("flat");
    assert!(sample_next_ids(&flat, 1.0, 1.0, &mut rng).is_err());
    assert!(sample_token_id(&[], 1.0, 1.0, &mut rng).is_err());
}

/// Small Q-Former config so the forward pass stays cheap in tests.
fn tiny_qformer_config() -> Blip2QFormerConfig {
    Blip2QFormerConfig {
        vocab_size: 32,
        hidden_size: 16,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        intermediate_size: 32,
        hidden_act: "gelu".to_string(),
        hidden_dropout_prob: 0.0,
        attention_probs_dropout_prob: 0.0,
        max_position_embeddings: 16,
        type_vocab_size: 2,
        initializer_range: 0.02,
        layer_norm_eps: 1e-12,
        position_embedding_type: "absolute".to_string(),
        cross_attention_frequency: 2,
        encoder_width: 16,
    }
}

/// The LM head must be a persistent layer. The old code built a fresh
/// `Linear::new(...)` inside `forward`, so two identical calls produced
/// different logits and leaked a vocab-sized allocation per call.
#[test]
fn test_qformer_lm_head_is_persistent() {
    let config = tiny_qformer_config();
    let model = Blip2QFormerModel::new(config).expect("qformer");
    let input_ids = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).expect("input ids");

    let first = model.forward(&input_ids, None, None, None).expect("first forward");
    let second = model.forward(&input_ids, None, None, None).expect("second forward");

    let a = first.logits.to_vec_f32().expect("logits a");
    let b = second.logits.to_vec_f32().expect("logits b");
    assert_eq!(a.len(), b.len(), "logits shape must be stable");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < 1e-6,
            "logit {i} changed between calls ({x} vs {y}) — the head is being re-randomised"
        );
    }
}

#[test]
fn test_qformer_lm_head_weight_is_loadable() {
    let config = tiny_qformer_config();
    let (vocab, hidden) = (config.vocab_size, config.hidden_size);
    let mut model = Blip2QFormerModel::new(config).expect("qformer");

    let weight = Tensor::zeros(&[vocab, hidden]).expect("zero head");
    model.set_lm_head_weight(weight).expect("load head");

    let input_ids = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]).expect("input ids");
    let out = model.forward(&input_ids, None, None, None).expect("forward");
    for logit in out.logits.to_vec_f32().expect("logits") {
        assert!(
            logit.abs() < 1e-6,
            "a zeroed head must produce zero logits, got {logit}"
        );
    }

    // A wrongly shaped head must be rejected rather than silently accepted.
    let bad = Tensor::zeros(&[hidden, vocab]).expect("bad head");
    assert!(model.set_lm_head_weight(bad).is_err());
}

// ── Cross-entropy loss ─────────────────────────────────────────────────

#[test]
fn test_shifted_cross_entropy_matches_hand_computation() {
    // [batch=1, seq=2, vocab=3]
    let logits = Tensor::from_vec(vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0], &[1, 2, 3]).expect("logits");
    let labels = Tensor::from_vec(vec![0.0, 2.0], &[1, 2]).expect("labels");
    let loss = shifted_cross_entropy(&logits, &labels).expect("loss");
    let value = loss.to_vec_f32().expect("loss data")[0];

    let expected = ((0.0f32).exp() + (1.0f32).exp() + (2.0f32).exp()).ln() - 2.0;
    assert!(
        (value - expected).abs() < 1e-5,
        "loss {value} != hand-computed {expected}"
    );
}

/// A constant `Tensor::scalar(1.0)` placeholder would pass any smoke test;
/// the loss must actually depend on the labels.
#[test]
fn test_shifted_cross_entropy_depends_on_labels() {
    let logits = Tensor::from_vec(vec![0.0, 1.0, 5.0, 3.0, 0.0, 0.0], &[1, 2, 3]).expect("logits");
    let good = Tensor::from_vec(vec![0.0, 2.0], &[1, 2]).expect("good labels");
    let bad = Tensor::from_vec(vec![0.0, 0.0], &[1, 2]).expect("bad labels");
    let good_loss =
        shifted_cross_entropy(&logits, &good).expect("good").to_vec_f32().expect("data")[0];
    let bad_loss =
        shifted_cross_entropy(&logits, &bad).expect("bad").to_vec_f32().expect("data")[0];
    assert!(
        good_loss < bad_loss,
        "predicting the high-logit token must cost less ({good_loss} vs {bad_loss})"
    );
    assert!(
        (good_loss - 1.0).abs() > 1e-6 || (bad_loss - 1.0).abs() > 1e-6,
        "the loss must not be a constant 1.0"
    );
}

#[test]
fn test_shifted_cross_entropy_aligns_labels_to_sequence_tail() {
    // Visual prefix of one token: labels cover only the last two positions.
    let logits = Tensor::from_vec(
        vec![9.0, 9.0, 9.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0],
        &[1, 3, 3],
    )
    .expect("logits");
    let labels = Tensor::from_vec(vec![0.0, 2.0], &[1, 2]).expect("labels");
    let value = shifted_cross_entropy(&logits, &labels)
        .expect("loss")
        .to_vec_f32()
        .expect("data")[0];
    let expected = ((0.0f32).exp() + (1.0f32).exp() + (2.0f32).exp()).ln() - 2.0;
    assert!((value - expected).abs() < 1e-5, "{value} != {expected}");
}

#[test]
fn test_shifted_cross_entropy_rejects_impossible_inputs() {
    let logits = Tensor::from_vec(vec![0.0, 1.0, 2.0], &[1, 1, 3]).expect("logits");
    let labels = Tensor::from_vec(vec![1.0], &[1, 1]).expect("labels");
    assert!(
        shifted_cross_entropy(&logits, &labels).is_err(),
        "a single position cannot be shifted"
    );

    let logits = Tensor::from_vec(vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0], &[1, 2, 3]).expect("logits");
    let out_of_range = Tensor::from_vec(vec![0.0, 7.0], &[1, 2]).expect("labels");
    assert!(
        shifted_cross_entropy(&logits, &out_of_range).is_err(),
        "labels outside the vocabulary must be an error"
    );
}

#[test]
#[ignore] // Heavy test - large model creation (~17s), run with --ignored
fn test_blip2_model_creation() {
    let config = Blip2Config::default();
    let model = Blip2Model::new(config);
    assert!(model.is_ok());
}

#[test]
#[ignore] // Heavy test - vision model creation (~17s), run with --ignored
fn test_blip2_vision_model() {
    let config = Blip2VisionConfig::default();
    let model = Blip2VisionModel::new(config);
    assert!(model.is_ok());
}

#[test]
#[ignore] // Heavy test - QFormer model creation, run with --ignored
fn test_blip2_qformer_model() {
    let config = Blip2QFormerConfig::default();
    let model = Blip2QFormerModel::new(config);
    assert!(model.is_ok());
}

#[test]
fn test_blip2_patch_embedding() {
    let config = Blip2VisionConfig::default();
    let embedding = Blip2PatchEmbedding::new(&config);
    assert!(embedding.is_ok());
}

#[test]
fn test_blip2_mlp() {
    let mlp = Blip2MLP::new(768, 3072, "gelu");
    assert!(mlp.is_ok());
}

#[test]
#[ignore] // Very heavy test - OPT 2.7B language model (~45s), run with --ignored
fn test_blip2_opt_language_model() {
    let config = Blip2TextConfig::opt_2_7b();
    let model = Blip2OptLanguageModel::new(config);
    assert!(model.is_ok());
}

#[test]
#[ignore] // Very heavy test - T5 XL language model (~30s), run with --ignored
fn test_blip2_t5_language_model() {
    let config = Blip2TextConfig::flan_t5_xl();
    let model = Blip2T5LanguageModel::new(config);
    assert!(model.is_ok());
}
