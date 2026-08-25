//! Unit tests for the GPT-2 building blocks.
//!
//! Declared from `model_blocks.rs` with `#[path]` so that it stays a child
//! module of the code under test — the assertions reach the blocks' private
//! `Linear` fields — while keeping both files under the 2000-line ceiling.

use super::super::model_ops::*;
use super::*;
use crate::gpt2::config::Gpt2Config;
use scirs2_core::ndarray::{ArrayD, IxDyn};
use trustformers_core::tensor::Tensor;

// LCG PRNG: a=6364136223846793005, c=1442695040888963407
fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state
}

fn lcg_f32_range(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let raw = (lcg_next(state) >> 11) as f32 / (1u64 << 53) as f32;
    lo + raw * (hi - lo)
}

fn make_array(shape: &[usize], seed: u64) -> ArrayD<f32> {
    let mut state = seed;
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| lcg_f32_range(&mut state, -1.0, 1.0)).collect();
    ArrayD::from_shape_vec(IxDyn(shape), data).expect("Failed to create array")
}

fn make_tensor(shape: &[usize], seed: u64) -> Tensor {
    Tensor::F32(make_array(shape, seed))
}

/// Flatten a tensor's values in row-major order for exact comparison.
fn values_of(tensor: &Tensor) -> Vec<f32> {
    match tensor {
        Tensor::F32(arr) => arr.iter().copied().collect(),
        other => panic!("expected an F32 tensor, got {other:?}"),
    }
}

/// Transpose a row-major `[rows, cols]` value list.
fn transposed(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(values.len());
    for col in 0..cols {
        for row in 0..rows {
            out.push(values[row * cols + col]);
        }
    }
    out
}

// ---- Conv1D weight orientation ----

#[test]
fn attention_load_weights_transposes_the_conv1d_projections() {
    // HuggingFace's GPT-2 uses `Conv1D`, which stores its weight as
    // `[in_features, out_features]` — the transpose of the `[out, in]`
    // layout `Linear` expects. The square `attn.c_proj` weight is the one
    // projection where a dropped transpose cannot be caught by a shape
    // check, so the values are compared element by element here.
    use crate::weight_loading::checkpoint::CheckpointReader;
    use crate::weight_loading::test_support::{build_safetensors, F32Tensor};

    let config = Gpt2Config {
        n_embd: 4,
        n_head: 2,
        n_layer: 1,
        ..Gpt2Config::default()
    };

    // Deliberately asymmetric so that A != A^T.
    let c_attn_values: Vec<f32> = (0..4 * 12).map(|i| i as f32 * 0.25 - 3.0).collect();
    let c_proj_values: Vec<f32> = (0..4 * 4).map(|i| (i * i) as f32 * 0.5 - 1.0).collect();

    let bytes = build_safetensors(&[
        F32Tensor::new("h.0.attn.c_attn.weight", &[4, 12], c_attn_values.clone()),
        F32Tensor::new("h.0.attn.c_attn.bias", &[12], vec![0.5; 12]),
        F32Tensor::new("h.0.attn.c_proj.weight", &[4, 4], c_proj_values.clone()),
        F32Tensor::new("h.0.attn.c_proj.bias", &[4], vec![-0.25; 4]),
    ]);
    let mut reader =
        CheckpointReader::from_reader(&mut bytes.as_slice()).expect("fixture must parse");

    let mut attention =
        Gpt2Attention::new_with_device(&config, Device::CPU).expect("attention must build");
    attention.load_weights(&mut reader, "h.0.attn").expect("weights must load");

    assert_eq!(attention.c_attn.weight().shape(), vec![12, 4]);
    assert_eq!(
        values_of(attention.c_attn.weight()),
        transposed(&c_attn_values, 4, 12),
        "the fused QKV projection must be stored transposed"
    );

    assert_eq!(attention.c_proj.weight().shape(), vec![4, 4]);
    let expected_proj = transposed(&c_proj_values, 4, 4);
    assert_ne!(
        expected_proj, c_proj_values,
        "the fixture must be asymmetric for this assertion to mean anything"
    );
    assert_eq!(
        values_of(attention.c_proj.weight()),
        expected_proj,
        "the square output projection must be stored transposed too"
    );

    // Biases are `[out_features]` in both conventions and must be copied verbatim.
    assert_eq!(
        values_of(attention.c_attn.bias().expect("the fused projection must have a bias")),
        vec![0.5; 12]
    );
}

#[test]
fn mlp_load_weights_transposes_both_conv1d_projections() {
    use crate::weight_loading::checkpoint::CheckpointReader;
    use crate::weight_loading::test_support::{build_safetensors, F32Tensor};

    let config = Gpt2Config {
        n_embd: 3,
        n_head: 1,
        n_layer: 1,
        ..Gpt2Config::default()
    };
    let inner = 4 * config.n_embd;

    let c_fc_values: Vec<f32> = (0..3 * inner).map(|i| i as f32 - 5.0).collect();
    let c_proj_values: Vec<f32> = (0..inner * 3).map(|i| 2.0 * i as f32 + 1.0).collect();

    let bytes = build_safetensors(&[
        F32Tensor::new("h.0.mlp.c_fc.weight", &[3, inner], c_fc_values.clone()),
        F32Tensor::new("h.0.mlp.c_fc.bias", &[inner], vec![0.0; inner]),
        F32Tensor::new("h.0.mlp.c_proj.weight", &[inner, 3], c_proj_values.clone()),
        F32Tensor::new("h.0.mlp.c_proj.bias", &[3], vec![0.0; 3]),
    ]);
    let mut reader =
        CheckpointReader::from_reader(&mut bytes.as_slice()).expect("fixture must parse");

    let mut mlp = Gpt2MLP::new_with_device(&config, Device::CPU).expect("mlp must build");
    mlp.load_weights(&mut reader, "h.0.mlp").expect("weights must load");

    assert_eq!(mlp.c_fc.weight().shape(), vec![inner, 3]);
    assert_eq!(
        values_of(mlp.c_fc.weight()),
        transposed(&c_fc_values, 3, inner)
    );

    assert_eq!(mlp.c_proj.weight().shape(), vec![3, inner]);
    assert_eq!(
        values_of(mlp.c_proj.weight()),
        transposed(&c_proj_values, inner, 3)
    );
}

#[test]
fn attention_load_weights_errors_when_a_projection_is_absent() {
    use crate::weight_loading::checkpoint::CheckpointReader;
    use crate::weight_loading::test_support::{build_safetensors, F32Tensor};

    let config = Gpt2Config {
        n_embd: 4,
        n_head: 2,
        n_layer: 1,
        ..Gpt2Config::default()
    };
    // `c_proj` is missing: the loader must say so rather than leaving the
    // random initialisation in place and reporting success.
    let bytes = build_safetensors(&[
        F32Tensor::new("h.0.attn.c_attn.weight", &[4, 12], vec![1.0; 48]),
        F32Tensor::new("h.0.attn.c_attn.bias", &[12], vec![0.0; 12]),
    ]);
    let mut reader =
        CheckpointReader::from_reader(&mut bytes.as_slice()).expect("fixture must parse");

    let mut attention =
        Gpt2Attention::new_with_device(&config, Device::CPU).expect("attention must build");
    let err = attention
        .load_weights(&mut reader, "h.0.attn")
        .expect_err("a missing projection must fail the load");
    assert!(
        err.to_string().contains("h.0.attn.c_proj.weight"),
        "unexpected error: {err}"
    );
}

// ---- create_causal_mask tests ----

#[test]
fn test_causal_mask_shape() {
    let seq_len = 5;
    let mask = create_causal_mask(seq_len).expect("create_causal_mask failed");
    let shape = mask.shape();
    assert_eq!(shape, &[1, 1, seq_len, seq_len]);
}

#[test]
fn test_causal_mask_diagonal_not_neg_inf() {
    let seq_len = 4;
    let mask = create_causal_mask(seq_len).expect("create_causal_mask failed");
    if let Tensor::F32(arr) = &mask {
        for i in 0..seq_len {
            let val = arr[[0, 0, i, i]];
            assert!(
                val.is_finite(),
                "Diagonal of causal mask must be finite at ({i},{i})"
            );
        }
    } else {
        panic!("Expected F32 tensor");
    }
}

#[test]
fn test_causal_mask_future_tokens_are_neg_inf() {
    let seq_len = 5;
    let mask = create_causal_mask(seq_len).expect("create_causal_mask failed");
    if let Tensor::F32(arr) = &mask {
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                let val = arr[[0, 0, i, j]];
                assert!(
                    val.is_infinite() && val < 0.0,
                    "Future token at ({i},{j}) must be -inf, got {val}"
                );
            }
        }
    } else {
        panic!("Expected F32 tensor");
    }
}

#[test]
fn test_causal_mask_past_tokens_are_zero() {
    let seq_len = 4;
    let mask = create_causal_mask(seq_len).expect("create_causal_mask failed");
    if let Tensor::F32(arr) = &mask {
        for i in 0..seq_len {
            for j in 0..=i {
                let val = arr[[0, 0, i, j]];
                assert!(
                    val == 0.0,
                    "Past/current token at ({i},{j}) must be 0, got {val}"
                );
            }
        }
    } else {
        panic!("Expected F32 tensor");
    }
}

#[test]
fn test_causal_mask_length_1() {
    let mask = create_causal_mask(1).expect("create_causal_mask(1) failed");
    if let Tensor::F32(arr) = &mask {
        assert_eq!(arr[[0, 0, 0, 0]], 0.0);
    }
}

// ---- softmax tests ----

#[test]
fn test_softmax_sums_to_one() {
    let mut state = 7u64;
    let n = 10;
    let data: Vec<f32> = (0..n).map(|_| lcg_f32_range(&mut state, -2.0, 2.0)).collect();
    let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).expect("array creation failed");
    let result = softmax(arr).expect("softmax failed");
    let sum: f32 = result.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax sum must be ~1.0, got {sum}"
    );
}

#[test]
fn test_softmax_all_positive() {
    let mut state = 13u64;
    let n = 8;
    let data: Vec<f32> = (0..n).map(|_| lcg_f32_range(&mut state, -3.0, 3.0)).collect();
    let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).expect("array creation failed");
    let result = softmax(arr).expect("softmax failed");
    for val in result.iter() {
        assert!(*val >= 0.0, "softmax output must be non-negative");
    }
}

// ---- log_softmax tests ----

#[test]
fn test_log_softmax_non_positive() {
    let mut state = 17u64;
    let n = 8;
    let data: Vec<f32> = (0..n).map(|_| lcg_f32_range(&mut state, -2.0, 2.0)).collect();
    let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).expect("array creation failed");
    let result = log_softmax(arr).expect("log_softmax failed");
    for val in result.iter() {
        assert!(
            *val <= 0.0 + 1e-6,
            "log_softmax output must be <= 0, got {val}"
        );
    }
}

#[test]
fn test_log_softmax_exp_sums_to_one() {
    let mut state = 31u64;
    let n = 6;
    let data: Vec<f32> = (0..n).map(|_| lcg_f32_range(&mut state, -1.0, 1.0)).collect();
    let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).expect("array creation failed");
    let result = log_softmax(arr).expect("log_softmax failed");
    let sum_exp: f32 = result.iter().map(|x| x.exp()).sum();
    assert!(
        (sum_exp - 1.0).abs() < 1e-5,
        "exp(log_softmax) must sum to 1, got {sum_exp}"
    );
}

// ---- apply_top_k_filtering tests ----

#[test]
fn test_top_k_keeps_k_finite_values() {
    let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let arr = ArrayD::from_shape_vec(IxDyn(&[10]), data).expect("array creation failed");
    let k = 3;
    let result = apply_top_k_filtering(arr, k).expect("top_k filter failed");
    let finite_count = result.iter().filter(|&&v| v.is_finite()).count();
    assert_eq!(finite_count, k, "top-k should keep exactly k finite values");
}

#[test]
fn test_top_k_largest_values_retained() {
    // data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let arr = ArrayD::from_shape_vec(IxDyn(&[10]), data).expect("array failed");
    let k = 3;
    let result = apply_top_k_filtering(arr, k).expect("top_k filter failed");
    // Top 3 values are 7, 8, 9 at indices 7, 8, 9
    assert!(result[7].is_finite());
    assert!(result[8].is_finite());
    assert!(result[9].is_finite());
    assert!(result[0].is_infinite());
}

// ---- apply_top_p_filtering tests ----

#[test]
fn test_top_p_at_least_one_finite() {
    let data: Vec<f32> = (0..10).map(|i| i as f32 + 1.0).collect();
    let arr = ArrayD::from_shape_vec(IxDyn(&[10]), data).expect("array failed");
    let result = apply_top_p_filtering(arr, 0.5).expect("top_p filter failed");
    let finite_count = result.iter().filter(|&&v| v.is_finite()).count();
    assert!(finite_count >= 1, "top-p must keep at least one token");
}

#[test]
fn test_top_p_full_probability_keeps_all() {
    let data: Vec<f32> = (0..5).map(|i| i as f32 + 1.0).collect();
    let arr = ArrayD::from_shape_vec(IxDyn(&[5]), data).expect("array failed");
    let result = apply_top_p_filtering(arr, 1.0).expect("top_p filter failed");
    let finite_count = result.iter().filter(|&&v| v.is_finite()).count();
    assert_eq!(finite_count, 5, "p=1.0 should keep all tokens");
}

// ---- stack_tensors tests ----

#[test]
fn test_stack_tensors_basic() {
    let t1 = make_tensor(&[3, 4], 11);
    let t2 = make_tensor(&[3, 4], 22);
    let stacked = stack_tensors(&[t1, t2]).expect("stack_tensors failed");
    let shape = stacked.shape();
    assert_eq!(shape[0], 2, "Batch dim must be 2");
    assert_eq!(shape[1], 3);
    assert_eq!(shape[2], 4);
}

#[test]
fn test_stack_tensors_empty_fails() {
    let result = stack_tensors(&[]);
    assert!(result.is_err(), "Stacking empty list must fail");
}

#[test]
fn test_stack_tensors_shape_mismatch_fails() {
    let t1 = make_tensor(&[3, 4], 11);
    let t2 = make_tensor(&[4, 4], 22); // different shape
    let result = stack_tensors(&[t1, t2]);
    assert!(
        result.is_err(),
        "Stacking tensors with different shapes must fail"
    );
}

// ---- Gpt2Block creation test ----

#[test]
fn test_gpt2_block_creates_ok() {
    let cfg = Gpt2Config::default();
    let block = Gpt2Block::new(&cfg);
    assert!(
        block.is_ok(),
        "Gpt2Block::new should succeed with default config"
    );
}

#[test]
fn test_gpt2_block_parameter_count_nonzero() {
    let cfg = Gpt2Config::default();
    let block = Gpt2Block::new(&cfg).expect("Block creation failed");
    assert!(block.parameter_count() > 0, "Block must have parameters");
}

// ---- MLP inner dim test ----

#[test]
fn test_gpt2_mlp_inner_dim_4x() {
    // When n_inner is None, inner dim = 4 * n_embd
    let cfg = Gpt2Config::default();
    assert!(cfg.n_inner.is_none(), "Default n_inner must be None");
    // The MLP created with this config should have inner_dim = 4 * 768 = 3072
    // We verify by checking the block can be created (it uses 4*n_embd internally)
    let block = Gpt2Block::new(&cfg).expect("Block creation failed");
    // The parameter count should reflect the 4x expansion
    let count = block.parameter_count();
    // rough lower bound: at least n_embd * 4 * n_embd for c_fc weight
    assert!(
        count > 768 * 3072,
        "MLP param count must reflect 4x expansion"
    );
}

// ---- ActivationType tests ----

#[test]
fn test_gelu_activation_on_zero() {
    let t = Tensor::from_vec(vec![0.0f32], &[1]).expect("tensor creation failed");
    let result = trustformers_core::ops::activations::gelu(&t).expect("gelu failed");
    if let Tensor::F32(arr) = result {
        assert!(arr[0].abs() < 1e-5, "gelu(0) must be ~0");
    }
}

#[test]
fn test_silu_activation_on_positive() {
    let t = Tensor::from_vec(vec![2.0f32], &[1]).expect("tensor creation failed");
    let result = trustformers_core::ops::activations::silu(&t).expect("silu failed");
    if let Tensor::F32(arr) = result {
        // SiLU(2) = 2 * sigmoid(2) ≈ 1.762
        assert!(
            arr[0] > 1.5 && arr[0] < 2.0,
            "SiLU(2) should be ~1.76, got {}",
            arr[0]
        );
    }
}

// ── KV-cache attention-mask widening ─────────────────────────────────────────

/// A mask that is already `kv_seq_len` wide is not this function's business, but the
/// narrow one it *is* meant for must land in the trailing columns with the cached
/// prefix left unmasked.
#[test]
fn widen_cached_attention_mask_places_the_block_at_the_end() {
    // create_causal_mask(2) == [[0, -inf], [0, 0]] shaped [1, 1, 2, 2].
    let mask = match create_causal_mask(2).expect("causal mask") {
        Tensor::F32(arr) => arr,
        other => panic!(
            "create_causal_mask must return F32, got {:?}",
            other.dtype()
        ),
    };
    let widened = widen_cached_attention_mask(&mask, 2, 5).expect("widening must succeed");
    assert_eq!(widened.shape(), &[1, 1, 2, 5]);

    // Row 0 sits at absolute position 3: keys 0..=3 visible, key 4 masked.
    for col in 0..4 {
        assert_eq!(
            widened[[0, 0, 0, col]],
            0.0,
            "row 0 col {col} must stay visible"
        );
    }
    assert!(widened[[0, 0, 0, 4]].is_infinite() && widened[[0, 0, 0, 4]] < 0.0);
    // Row 1 sits at absolute position 4: every key is visible.
    for col in 0..5 {
        assert_eq!(
            widened[[0, 0, 1, col]],
            0.0,
            "row 1 col {col} must stay visible"
        );
    }
}

/// Single-token decode: the widened mask is an all-visible row, which is why the
/// missing widening only ever surfaced on multi-token continuations.
#[test]
fn widen_cached_attention_mask_is_all_visible_for_one_query_row() {
    let mask = match create_causal_mask(1).expect("causal mask") {
        Tensor::F32(arr) => arr,
        other => panic!(
            "create_causal_mask must return F32, got {:?}",
            other.dtype()
        ),
    };
    let widened = widen_cached_attention_mask(&mask, 1, 6).expect("widening must succeed");
    assert_eq!(widened.shape(), &[1, 1, 1, 6]);
    assert!(widened.iter().all(|v| *v == 0.0));
}

/// Shapes the rule cannot interpret are refused with a structured error rather than
/// reaching `ndarray` and aborting the process with a broadcast panic.
#[test]
fn widen_cached_attention_mask_refuses_uninterpretable_shapes() {
    // Mask wider than the query block but narrower than the keys.
    let odd = ArrayD::<f32>::zeros(IxDyn(&[1, 1, 2, 3]));
    assert!(widen_cached_attention_mask(&odd, 2, 5).is_err());

    // A genuinely per-head mask carries information the widening would discard.
    let per_head = ArrayD::<f32>::zeros(IxDyn(&[1, 4, 2, 2]));
    let err =
        widen_cached_attention_mask(&per_head, 2, 5).expect_err("a per-head mask must be refused");
    assert!(
        format!("{err}").contains("per-batch or per-head"),
        "error should name the reason, got: {err}"
    );

    // kv shorter than the query block is nonsense.
    let mask = ArrayD::<f32>::zeros(IxDyn(&[1, 1, 5, 5]));
    assert!(widen_cached_attention_mask(&mask, 5, 2).is_err());
}
