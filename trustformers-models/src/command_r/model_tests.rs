//! Unit tests for the Command-R model components.
//!
//! Kept in a separate file so `model.rs` stays under the 2000-line limit; the
//! module is still a child of `command_r::model`, so it can reach the private
//! internals it needs to verify.

use super::*;
use scirs2_core::random::{SeedableRng, StdRng};

/// Deterministic pseudo-random test data (no RNG dependency in assertions).
fn lcg_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn positions_tensor(range: std::ops::Range<usize>) -> Tensor {
    let len = range.len();
    Tensor::from_vec(range.map(|p| p as f32).collect(), &[1, len]).expect("positions")
}

// ── KV cache ────────────────────────────────────────────────────────────

/// Incremental decoding through the cache must reproduce the full-sequence
/// forward pass. The old code dropped the freshly computed keys/values, so
/// step 2 attended to a stale one-token cache and this diverged.
#[test]
fn test_attention_cache_matches_full_sequence() {
    let config = CommandRConfig::tiny();
    let attention = CommandRAttention::new(&config).expect("attention");
    let seq_len = 4;
    let hidden_size = config.hidden_size;
    let data = lcg_vec(seq_len * hidden_size, 1234);
    let hidden = Tensor::from_vec(data, &[1, seq_len, hidden_size]).expect("hidden states");

    let (full_output, full_cache) = attention
        .forward(&hidden, None, &positions_tensor(0..seq_len), None)
        .expect("full forward");
    let (full_key, _full_value) = full_cache.expect("cache returned");
    assert_eq!(
        full_key.shape(),
        &[1, seq_len, config.num_key_value_heads, config.head_dim()],
        "cache must hold every processed token"
    );
    let full_data = full_output.data().expect("full data");

    let mut cache: Option<(Tensor, Tensor)> = None;
    let mut last_step = Vec::new();
    for t in 0..seq_len {
        let step = hidden.slice(1, t, t + 1).expect("token slice");
        let past = cache.as_ref().map(|(k, v)| (k, v));
        let (out, present) = attention
            .forward(&step, None, &positions_tensor(t..t + 1), past)
            .expect("incremental forward");
        cache = present;
        last_step = out.data().expect("step data");
    }

    let (cached_key, cached_value) = cache.expect("final cache");
    assert_eq!(
        cached_key.shape(),
        &[1, seq_len, config.num_key_value_heads, config.head_dim()],
        "the cache must grow by one token per step"
    );
    assert_eq!(cached_value.shape(), cached_key.shape());

    let offset = (seq_len - 1) * hidden_size;
    for i in 0..hidden_size {
        let expected = full_data[offset + i];
        let got = last_step[i];
        assert!(
            (expected - got).abs() < 1e-4,
            "cached decoding diverged at {i}: {got} vs {expected}"
        );
    }
}

#[test]
fn test_attention_cache_grows_by_one_per_step() {
    let config = CommandRConfig::tiny();
    let attention = CommandRAttention::new(&config).expect("attention");
    let hidden = Tensor::from_vec(lcg_vec(config.hidden_size, 7), &[1, 1, config.hidden_size])
        .expect("hidden");

    let (_, first) = attention
        .forward(&hidden, None, &positions_tensor(0..1), None)
        .expect("first step");
    let (first_key, first_value) = first.expect("first cache");
    assert_eq!(first_key.shape()[1], 1);

    let (_, second) = attention
        .forward(
            &hidden,
            None,
            &positions_tensor(1..2),
            Some((&first_key, &first_value)),
        )
        .expect("second step");
    let (second_key, _) = second.expect("second cache");
    assert_eq!(
        second_key.shape()[1],
        2,
        "the present cache must include the newly computed key"
    );
}

/// Batched decoding: every sequence in the batch must be attended independently
/// and the cache must grow for all of them.
#[test]
fn test_attention_batched_cache_matches_full_sequence() {
    let config = CommandRConfig::tiny();
    let attention = CommandRAttention::new(&config).expect("attention");
    let batch = 2;
    let seq_len = 3;
    let hidden_size = config.hidden_size;
    let data = lcg_vec(batch * seq_len * hidden_size, 4321);
    let hidden = Tensor::from_vec(data, &[batch, seq_len, hidden_size]).expect("hidden states");

    let (full_output, full_cache) = attention
        .forward(&hidden, None, &positions_tensor(0..seq_len), None)
        .expect("batched full forward");
    assert_eq!(full_output.shape(), &[batch, seq_len, hidden_size]);
    let (full_key, _) = full_cache.expect("cache");
    assert_eq!(full_key.shape()[0], batch, "cache keeps the batch axis");
    assert_eq!(full_key.shape()[1], seq_len);
    let full_data = full_output.data().expect("full data");

    let mut cache: Option<(Tensor, Tensor)> = None;
    let mut last_step = Vec::new();
    for t in 0..seq_len {
        let step = hidden.slice(1, t, t + 1).expect("token slice");
        let past = cache.as_ref().map(|(k, v)| (k, v));
        let (out, present) = attention
            .forward(&step, None, &positions_tensor(t..t + 1), past)
            .expect("incremental forward");
        cache = present;
        last_step = out.data().expect("step data");
    }

    // Compare the final position of every batch element.
    for b in 0..batch {
        let full_offset = (b * seq_len + (seq_len - 1)) * hidden_size;
        let step_offset = b * hidden_size;
        for i in 0..hidden_size {
            let expected = full_data[full_offset + i];
            let got = last_step[step_offset + i];
            assert!(
                (expected - got).abs() < 1e-4,
                "batch {b} element {i}: {got} vs {expected}"
            );
        }
    }
}

#[test]
fn test_attention_rejects_mismatched_cache() {
    let config = CommandRConfig::tiny();
    let attention = CommandRAttention::new(&config).expect("attention");
    let hidden = Tensor::from_vec(lcg_vec(config.hidden_size, 9), &[1, 1, config.hidden_size])
        .expect("hidden");
    let bogus = Tensor::zeros(&[1, 2, 3]).expect("bogus cache");
    let result = attention.forward(
        &hidden,
        None,
        &positions_tensor(0..1),
        Some((&bogus, &bogus)),
    );
    assert!(result.is_err(), "a malformed cache must be rejected");
}

/// Grouped-query attention: fewer KV heads than query heads must still work
/// and must produce an input-dependent output.
#[test]
fn test_attention_grouped_query_heads() {
    let mut config = CommandRConfig::tiny();
    config.num_key_value_heads = 2; // 4 query heads / 2 kv heads => 2 groups
    let attention = CommandRAttention::new(&config).expect("attention");
    let seq_len = 3;
    let hidden = Tensor::from_vec(
        lcg_vec(seq_len * config.hidden_size, 21),
        &[1, seq_len, config.hidden_size],
    )
    .expect("hidden");

    let (out, cache) = attention
        .forward(&hidden, None, &positions_tensor(0..seq_len), None)
        .expect("gqa forward");
    assert_eq!(out.shape(), &[1, seq_len, config.hidden_size]);
    let (key, _) = cache.expect("cache");
    assert_eq!(key.shape()[2], 2, "cache keeps num_key_value_heads heads");

    let other = Tensor::from_vec(
        lcg_vec(seq_len * config.hidden_size, 22),
        &[1, seq_len, config.hidden_size],
    )
    .expect("hidden");
    let (out2, _) = attention
        .forward(&other, None, &positions_tensor(0..seq_len), None)
        .expect("gqa forward 2");
    let a = out.data().expect("a");
    let b = out2.data().expect("b");
    let diff = a.iter().zip(b.iter()).fold(0.0f32, |acc, (x, y)| acc.max((x - y).abs()));
    assert!(diff > 1e-6, "attention output must depend on the input");
}

#[test]
fn test_attention_is_causal() {
    let config = CommandRConfig::tiny();
    let attention = CommandRAttention::new(&config).expect("attention");
    let seq_len = 3;
    let hidden_size = config.hidden_size;
    let mut data = lcg_vec(seq_len * hidden_size, 31);
    let baseline = Tensor::from_vec(data.clone(), &[1, seq_len, hidden_size]).expect("hidden");
    let (out_a, _) = attention
        .forward(&baseline, None, &positions_tensor(0..seq_len), None)
        .expect("forward a");

    // Change the last token only; earlier positions must not move.
    for value in data.iter_mut().skip((seq_len - 1) * hidden_size) {
        *value += 1.0;
    }
    let perturbed = Tensor::from_vec(data, &[1, seq_len, hidden_size]).expect("hidden");
    let (out_b, _) = attention
        .forward(&perturbed, None, &positions_tensor(0..seq_len), None)
        .expect("forward b");

    let a = out_a.data().expect("a");
    let b = out_b.data().expect("b");
    for i in 0..(seq_len - 1) * hidden_size {
        assert!(
            (a[i] - b[i]).abs() < 1e-5,
            "position {} must not see the future",
            i / hidden_size
        );
    }
}

// ── Sampling ────────────────────────────────────────────────────────────

#[test]
fn test_sample_next_token_greedy_for_zero_temperature() {
    let logits = [0.1f32, 5.0, -3.0, 2.0];
    let mut rng = StdRng::seed_from_u64(1);
    let token = CommandRForCausalLM::sample_next_token(&logits, 0.0, None, None, &mut rng)
        .expect("greedy sample");
    assert_eq!(token, 1, "temperature 0 must pick the argmax");
}

#[test]
fn test_sample_next_token_top_k_one_is_deterministic() {
    let logits = [1.0f32, 2.0, 3.0, 4.0];
    let mut rng = StdRng::seed_from_u64(2);
    for _ in 0..20 {
        let token = CommandRForCausalLM::sample_next_token(&logits, 1.0, Some(1), None, &mut rng)
            .expect("top-1 sample");
        assert_eq!(token, 3, "top-k=1 must always return the top token");
    }
}

#[test]
fn test_sample_next_token_top_k_restricts_support() {
    // Only the two largest logits (indices 3 and 2) may ever be drawn.
    let logits = [0.0f32, 0.5, 4.0, 4.1];
    let mut rng = StdRng::seed_from_u64(3);
    for _ in 0..200 {
        let token = CommandRForCausalLM::sample_next_token(&logits, 1.0, Some(2), None, &mut rng)
            .expect("top-k sample");
        assert!(token == 2 || token == 3, "top-k=2 drew token {token}");
    }
}

#[test]
fn test_sample_next_token_top_p_restricts_support() {
    let logits = [0.0f32, 0.1, 0.2, 8.0];
    let mut rng = StdRng::seed_from_u64(4);
    for _ in 0..200 {
        let token = CommandRForCausalLM::sample_next_token(&logits, 1.0, None, Some(0.5), &mut rng)
            .expect("top-p sample");
        assert_eq!(token, 3, "a 0.5 nucleus contains only the dominant token");
    }
}

/// The multinomial path must follow the softmax distribution, not the argmax.
/// The old `categorical_sample` returned the argmax every time, so the
/// observed frequency would have been 1.0 instead of ~0.75.
#[test]
fn test_sample_next_token_follows_distribution() {
    // softmax([0, ln 3]) = [0.25, 0.75]
    let logits = [0.0f32, 3.0f32.ln()];
    let mut rng = StdRng::seed_from_u64(12345);
    let draws = 4000;
    let mut ones = 0usize;
    for _ in 0..draws {
        let token = CommandRForCausalLM::sample_next_token(&logits, 1.0, None, None, &mut rng)
            .expect("sample");
        if token == 1 {
            ones += 1;
        }
    }
    let frequency = ones as f32 / draws as f32;
    assert!(
        (frequency - 0.75).abs() < 0.05,
        "expected ~75% draws of the 0.75-probability token, got {frequency}"
    );
}

#[test]
fn test_sample_next_token_rejects_empty_logits() {
    let mut rng = StdRng::seed_from_u64(5);
    assert!(
        CommandRForCausalLM::sample_next_token(&[], 1.0, None, None, &mut rng).is_err(),
        "empty logits must be an error, not a silent 0"
    );
}

// ── Generation ──────────────────────────────────────────────────────────

/// Config whose EOS id cannot be produced, so generation always runs to length.
fn tiny_config_without_eos() -> CommandRConfig {
    CommandRConfig {
        eos_token_id: None,
        ..CommandRConfig::tiny()
    }
}

#[test]
fn test_generate_greedy_is_deterministic_and_grows() {
    // EOS is disabled so the run cannot stop early: with a randomly initialised
    // head the argmax may legitimately land on the EOS id, which would make an
    // exact-length assertion flaky rather than meaningful.
    let config = tiny_config_without_eos();
    let mut model = CommandRForCausalLM::new(&config).expect("model");
    let prompt = Tensor::from_vec_i64(vec![1, 2, 3], &[1, 3]).expect("prompt");

    let mut rng = StdRng::seed_from_u64(7);
    let first = model
        .generate_with_rng(&prompt, 3, 0.0, None, None, &mut rng)
        .expect("generate");
    let second = model
        .generate_with_rng(&prompt, 3, 0.0, None, None, &mut rng)
        .expect("generate");

    assert_eq!(first.shape()[1], 6, "3 prompt + 3 generated tokens");
    assert_eq!(
        first.data().expect("first"),
        second.data().expect("second"),
        "greedy decoding must be deterministic"
    );
    assert!(
        matches!(first, Tensor::I64(_)),
        "generated ids must stay integral"
    );

    // The prompt is preserved at the front of the returned sequence.
    let ids = first.data().expect("ids");
    assert_eq!(&ids[..3], &[1.0, 2.0, 3.0]);
}

#[test]
fn test_generate_stops_at_eos() {
    // The tiny config uses EOS id 2, so a run that emits it must stop there.
    let config = CommandRConfig::tiny();
    let mut model = CommandRForCausalLM::new(&config).expect("model");
    let prompt = Tensor::from_vec_i64(vec![1], &[1, 1]).expect("prompt");
    let mut rng = StdRng::seed_from_u64(3);
    let out = model
        .generate_with_rng(&prompt, 4, 0.0, None, None, &mut rng)
        .expect("generate");

    let ids = out.data().expect("ids");
    assert!(ids.len() >= 2, "at least one token must be generated");
    assert!(ids.len() <= 5, "generation must respect max_length");
    if let Some(eos_position) = ids.iter().position(|&id| id as usize == 2) {
        assert_eq!(
            eos_position,
            ids.len() - 1,
            "generation must stop at the first EOS token"
        );
    }
}

#[test]
fn test_generate_respects_top_k_one() {
    let config = tiny_config_without_eos();
    let mut model = CommandRForCausalLM::new(&config).expect("model");
    let prompt = Tensor::from_vec_i64(vec![5, 6], &[1, 2]).expect("prompt");

    let mut greedy_rng = StdRng::seed_from_u64(11);
    let greedy = model
        .generate_with_rng(&prompt, 2, 0.0, None, None, &mut greedy_rng)
        .expect("greedy");
    let mut sampled_rng = StdRng::seed_from_u64(99);
    let top1 = model
        .generate_with_rng(&prompt, 2, 1.0, Some(1), None, &mut sampled_rng)
        .expect("top-1");
    assert_eq!(
        greedy.data().expect("greedy"),
        top1.data().expect("top1"),
        "top-k=1 sampling must agree with greedy decoding"
    );
}

#[test]
fn test_causal_lm_loss_matches_hand_computation() {
    // logits: [batch=1, seq=2, vocab=3]
    let logits = Tensor::from_vec(vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0], &[1, 2, 3]).expect("logits");
    // Position 0 predicts label at position 1 (= 2).
    let labels = Tensor::from_vec(vec![0.0, 2.0], &[1, 2]).expect("labels");
    let loss = CommandRForCausalLM::causal_lm_loss(&logits, &labels).expect("loss");
    let value = loss.data().expect("loss data")[0];

    // -log softmax([0,1,2])[2] = ln(e^0 + e^1 + e^2) - 2
    let expected = ((0.0f32).exp() + (1.0f32).exp() + (2.0f32).exp()).ln() - 2.0;
    assert!(
        (value - expected).abs() < 1e-5,
        "loss {value} != hand-computed {expected}"
    );
}

#[test]
fn test_causal_lm_loss_depends_on_labels() {
    let logits = Tensor::from_vec(vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0], &[1, 2, 3]).expect("logits");
    let labels_a = Tensor::from_vec(vec![0.0, 2.0], &[1, 2]).expect("labels a");
    let labels_b = Tensor::from_vec(vec![0.0, 0.0], &[1, 2]).expect("labels b");
    let a = CommandRForCausalLM::causal_lm_loss(&logits, &labels_a)
        .expect("a")
        .data()
        .expect("a data")[0];
    let b = CommandRForCausalLM::causal_lm_loss(&logits, &labels_b)
        .expect("b")
        .data()
        .expect("b data")[0];
    assert!(
        (a - b).abs() > 1e-4,
        "a constant loss would not distinguish label sets"
    );
}

#[test]
fn test_rope_rotation_is_position_dependent() {
    let rope = CommandRRoPE::new(8, 128, 10000.0).expect("rope");
    let mut at_zero = vec![0.5f32, -0.25, 0.75, 1.0, -0.5, 0.25, -0.75, 0.125];
    let original = at_zero.clone();
    rope.rotate_in_place(&mut at_zero, 1, 1, 8, &[0]).expect("rotate at 0");
    for (got, want) in at_zero.iter().zip(original.iter()) {
        assert!((got - want).abs() < 1e-6, "position 0 must be the identity");
    }

    let mut at_three = original.clone();
    rope.rotate_in_place(&mut at_three, 1, 1, 8, &[3]).expect("rotate at 3");
    let changed = at_three
        .iter()
        .zip(original.iter())
        .any(|(got, want)| (got - want).abs() > 1e-6);
    assert!(changed, "position 3 must actually rotate the vector");

    let norm_before: f32 = original.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_after: f32 = at_three.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (norm_before - norm_after).abs() < 1e-5,
        "RoPE preserves norm"
    );
}

#[test]
fn test_rope_rejects_positions_beyond_context() {
    let rope = CommandRRoPE::new(4, 8, 10000.0).expect("rope");
    let mut data = vec![0.0f32; 4];
    assert!(
        rope.rotate_in_place(&mut data, 1, 1, 4, &[8]).is_err(),
        "positions past max_sequence_length must be rejected"
    );
}

// Tests using tiny configuration for fast execution
#[test]
fn test_command_r_model_creation_tiny() {
    let config = CommandRConfig::tiny();
    let model = CommandRModel::new(&config);
    assert!(model.is_ok());
}

#[test]
fn test_command_r_causal_lm_creation_tiny() {
    let config = CommandRConfig::tiny();
    let model = CommandRForCausalLM::new(&config);
    assert!(model.is_ok());
}

#[test]
#[ignore = "Forward pass requires proper hidden state input - model's forward method is shadowed by Model trait"]
fn test_command_r_forward_pass_tiny() {
    let config = CommandRConfig::tiny();
    let model = CommandRModel::new(&config).expect("operation failed");

    // The Model trait's forward expects hidden states (F32 tensor), not input_ids
    // Create a proper hidden state tensor for testing
    let batch_size = 1;
    let seq_len = 4;
    let hidden_states =
        Tensor::zeros(&[batch_size, seq_len, config.hidden_size]).expect("operation failed");

    let result = <CommandRModel as Model>::forward(&model, hidden_states);
    assert!(result.is_ok(), "Forward pass failed: {:?}", result.err());
}

#[test]
fn test_command_r_attention_creation_tiny() {
    let config = CommandRConfig::tiny();
    let attention = CommandRAttention::new(&config);
    assert!(attention.is_ok());
}

#[test]
fn test_command_r_mlp_creation_tiny() {
    let config = CommandRConfig::tiny();
    let mlp = CommandRMLP::new(&config);
    assert!(mlp.is_ok());
}

#[test]
fn test_command_r_decoder_layer_creation_tiny() {
    let config = CommandRConfig::tiny();
    let layer = CommandRDecoderLayer::new(&config);
    assert!(layer.is_ok());
}

#[test]
fn test_rope_creation() {
    let rope = CommandRRoPE::new(128, 4096, 10000.0);
    assert!(rope.is_ok());
}

// Full model size tests - ignored by default due to memory/time requirements
#[test]
#[ignore = "Full model size test - requires significant memory and time"]
fn test_command_r_model_creation() {
    let config = CommandRConfig::command_r();
    let model = CommandRModel::new(&config);
    assert!(model.is_ok());
}

#[test]
#[ignore = "Full model size test - requires significant memory and time"]
fn test_command_r_plus_model_creation() {
    let config = CommandRConfig::command_r_plus();
    let model = CommandRModel::new(&config);
    assert!(model.is_ok());
}

#[test]
#[ignore = "Full model size test - requires significant memory and time"]
fn test_command_r_causal_lm_creation() {
    let config = CommandRConfig::command_r();
    let model = CommandRForCausalLM::new(&config);
    assert!(model.is_ok());
}

#[test]
#[ignore = "Full model size test - requires significant memory and time"]
fn test_command_r_forward_pass() {
    let config = CommandRConfig::command_r();
    let model = CommandRModel::new(&config).expect("operation failed");

    // Use I64 tensor for input_ids (token IDs should be integers)
    let input_ids = Tensor::from_vec_i64(vec![1, 2, 3, 4], &[1, 4]).expect("operation failed");

    let result = model.forward(&input_ids, None, None, None);
    assert!(result.is_ok());
}

#[test]
#[ignore = "Full model size test - requires significant memory and time"]
fn test_command_r_attention_creation() {
    let config = CommandRConfig::command_r();
    let attention = CommandRAttention::new(&config);
    assert!(attention.is_ok());
}

#[test]
#[ignore = "Full model size test - requires significant memory and time"]
fn test_command_r_mlp_creation() {
    let config = CommandRConfig::command_r();
    let mlp = CommandRMLP::new(&config);
    assert!(mlp.is_ok());
}

#[test]
#[ignore = "Full model size test - requires significant memory and time"]
fn test_command_r_decoder_layer_creation() {
    let config = CommandRConfig::command_r();
    let layer = CommandRDecoderLayer::new(&config);
    assert!(layer.is_ok());
}

// ── Weight loading ──────────────────────────────────────────────────────────

use crate::weight_loading::test_support::{build_safetensors, F32Tensor};
use std::io::Cursor;

/// Every tensor a `CohereForCausalLM` checkpoint of `config` would carry.
fn checkpoint_tensors(config: &CommandRConfig, with_lm_head: bool) -> Vec<F32Tensor> {
    let hidden = config.hidden_size;
    let head_dim = config.head_dim();
    let q_width = config.num_attention_heads * head_dim;
    let kv_width = config.num_key_value_heads * head_dim;
    let inter = config.intermediate_size;
    let mut tensors = vec![F32Tensor::ramp(
        "model.embed_tokens.weight",
        &[config.vocab_size, hidden],
        0.0,
    )];
    for i in 0..config.num_hidden_layers {
        let seed = (i as f32 + 1.0) * 100.0;
        tensors.push(F32Tensor::ramp(
            &format!("model.layers.{i}.self_attn.q_proj.weight"),
            &[q_width, hidden],
            seed,
        ));
        tensors.push(F32Tensor::ramp(
            &format!("model.layers.{i}.self_attn.k_proj.weight"),
            &[kv_width, hidden],
            seed + 1.0,
        ));
        tensors.push(F32Tensor::ramp(
            &format!("model.layers.{i}.self_attn.v_proj.weight"),
            &[kv_width, hidden],
            seed + 2.0,
        ));
        tensors.push(F32Tensor::ramp(
            &format!("model.layers.{i}.self_attn.o_proj.weight"),
            &[hidden, q_width],
            seed + 3.0,
        ));
        tensors.push(F32Tensor::ramp(
            &format!("model.layers.{i}.mlp.gate_proj.weight"),
            &[inter, hidden],
            seed + 4.0,
        ));
        tensors.push(F32Tensor::ramp(
            &format!("model.layers.{i}.mlp.up_proj.weight"),
            &[inter, hidden],
            seed + 5.0,
        ));
        tensors.push(F32Tensor::ramp(
            &format!("model.layers.{i}.mlp.down_proj.weight"),
            &[hidden, inter],
            seed + 6.0,
        ));
        tensors.push(F32Tensor::ramp(
            &format!("model.layers.{i}.input_layernorm.weight"),
            &[hidden],
            seed + 7.0,
        ));
    }
    tensors.push(F32Tensor::ramp("model.norm.weight", &[hidden], 999.0));
    if with_lm_head {
        tensors.push(F32Tensor::ramp(
            "lm_head.weight",
            &[config.vocab_size, hidden],
            7.0,
        ));
    }
    tensors
}

/// `load_pretrained` used to print progress and assign mock tensors while
/// returning `Ok`. It must now bind the checkpoint's actual values.
#[test]
fn test_load_pretrained_binds_real_weights() {
    let config = CommandRConfig::tiny();
    let bytes = build_safetensors(&checkpoint_tensors(&config, true));
    let mut model = CommandRForCausalLM::new(&config).expect("model");

    model.load_pretrained(&mut Cursor::new(bytes)).expect("checkpoint must load");

    let q = model.model.layers[0].self_attn.q_proj.weight().data().expect("q weights");
    assert!((q[0] - 100.0).abs() < 1e-6, "q_proj[0] = {}", q[0]);
    assert!((q[1] - 100.5).abs() < 1e-6, "q_proj[1] = {}", q[1]);

    let q1 = model.model.layers[1].self_attn.q_proj.weight().data().expect("layer 1");
    assert!(
        (q1[0] - 200.0).abs() < 1e-6,
        "layer 1 q_proj[0] = {}",
        q1[0]
    );

    let head = model.lm_head.weight().data().expect("lm head");
    assert!((head[0] - 7.0).abs() < 1e-6, "lm_head[0] = {}", head[0]);
}

#[test]
fn test_load_pretrained_ties_head_to_embeddings() {
    let config = CommandRConfig::tiny();
    let bytes = build_safetensors(&checkpoint_tensors(&config, false));
    let mut model = CommandRForCausalLM::new(&config).expect("model");
    model.load_pretrained(&mut Cursor::new(bytes)).expect("tied load");

    let head = model.lm_head.weight().data().expect("lm head");
    assert_eq!(head.len(), config.vocab_size * config.hidden_size);
    assert!(head[0].abs() < 1e-6, "tied head[0] = {}", head[0]);
    assert!((head[3] - 1.5).abs() < 1e-6, "tied head[3] = {}", head[3]);
}

#[test]
fn test_load_pretrained_rejects_wrong_shapes() {
    let config = CommandRConfig::tiny();
    let mut tensors = checkpoint_tensors(&config, true);
    tensors[0] = F32Tensor::ramp("model.embed_tokens.weight", &[config.vocab_size, 4], 0.0);
    let bytes = build_safetensors(&tensors);
    let mut model = CommandRForCausalLM::new(&config).expect("model");
    assert!(
        model.load_pretrained(&mut Cursor::new(bytes)).is_err(),
        "a mis-shaped tensor must fail the load"
    );
}

#[test]
fn test_load_pretrained_rejects_foreign_checkpoint() {
    let config = CommandRConfig::tiny();
    let bytes = build_safetensors(&[F32Tensor::ramp("gpt2.h.0.attn.weight", &[2, 2], 1.0)]);
    let mut model = CommandRForCausalLM::new(&config).expect("model");
    assert!(
        model.load_pretrained(&mut Cursor::new(bytes)).is_err(),
        "a checkpoint for another architecture must be rejected"
    );
}
