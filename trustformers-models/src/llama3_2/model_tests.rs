//! Unit tests for the Llama-3.2 model components.
//!
//! Kept in a separate file so `model.rs` stays under the 2000-line limit; the
//! module is still a child of `llama3_2::model`, so it can reach the private
//! internals it needs to verify.

use super::*;
use crate::llama3_2::config::Llama32Config;
use trustformers_core::traits::Layer;

// LCG — no rand crate
fn lcg_next(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
    (*state >> 33) as f32 / (1u64 << 31) as f32
}

fn small_config() -> Llama32Config {
    Llama32Config::small_test()
}

// ── Config spec tests ─────────────────────────────────────────────────────

#[test]
fn test_llama32_1b_vocab_size() {
    // LLaMA-3.2 uses shared 128 256 Tiktoken vocabulary
    let cfg = Llama32Config::llama32_3b();
    assert_eq!(cfg.vocab_size, 128256, "vocab_size must be 128256");
}

#[test]
fn test_llama32_rope_theta() {
    let cfg = Llama32Config::llama32_3b();
    assert_eq!(cfg.rope_theta, 500000.0, "RoPE theta must be 500000");
}

#[test]
fn test_llama32_3b_num_attention_heads() {
    let cfg = Llama32Config::llama32_3b();
    assert_eq!(
        cfg.num_attention_heads, 24,
        "3B model must have 24 query heads"
    );
}

#[test]
fn test_llama32_3b_num_kv_heads() {
    let cfg = Llama32Config::llama32_3b();
    assert_eq!(cfg.num_key_value_heads, 8, "3B model must have 8 KV heads");
}

#[test]
fn test_gqa_group_size_3b() {
    let cfg = Llama32Config::llama32_3b();
    let group_size = cfg.num_attention_heads / cfg.num_key_value_heads;
    assert_eq!(group_size, 3, "3B GQA group size = 24/8 = 3");
}

#[test]
fn test_gqa_group_size_small_test() {
    let cfg = small_config();
    let group_size = cfg.num_attention_heads / cfg.num_key_value_heads;
    // small_test: heads=4, kv_heads=2 → group_size=2
    assert_eq!(group_size, 2, "small_test GQA group size must be 2");
}

#[test]
fn test_head_dim_divides_hidden_size() {
    let cfg = small_config();
    assert_eq!(
        cfg.hidden_size % cfg.num_attention_heads,
        0,
        "hidden_size must be divisible by num_attention_heads"
    );
    let expected_head_dim = cfg.hidden_size / cfg.num_attention_heads;
    assert_eq!(
        cfg.head_dim, expected_head_dim,
        "head_dim must equal hidden_size / num_heads"
    );
}

#[test]
fn test_llama32_11b_config() {
    let cfg = Llama32Config::llama32_11b();
    assert_eq!(cfg.vocab_size, 128256);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.num_key_value_heads, 8);
    let group_size = cfg.num_attention_heads / cfg.num_key_value_heads;
    assert_eq!(group_size, 4, "11B model GQA group_size = 32/8 = 4");
}

// ── RMSNorm ───────────────────────────────────────────────────────────────

#[test]
fn test_rms_norm_no_bias_construction() {
    // Llama32RmsNorm has only `weight`, no bias
    let norm = Llama32RmsNorm::new(32, 1e-5).expect("RMSNorm should construct");
    assert_eq!(
        norm.parameter_count(),
        32,
        "RMSNorm parameter count = hidden_size (weight only)"
    );
}

#[test]
fn test_rms_norm_normalizes_non_zero_input() {
    let norm = Llama32RmsNorm::new(4, 1e-5).expect("RMSNorm should construct");
    let input =
        Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[4]).expect("tensor should construct");
    let output = norm.forward(input).expect("RMSNorm forward should succeed");
    let out_vals: Vec<f32> = match &output {
        Tensor::F32(arr) => arr.iter().copied().collect(),
        _ => panic!("expected F32"),
    };
    // The mean square should be ~1 after normalisation (with unit weights)
    let mean_sq: f32 = out_vals.iter().map(|&x| x * x).sum::<f32>() / out_vals.len() as f32;
    assert!(
        (mean_sq - 1.0).abs() < 0.1,
        "RMSNorm: mean square of output should ≈ 1"
    );
}

#[test]
fn test_rms_norm_handles_uniform_input() {
    let norm = Llama32RmsNorm::new(8, 1e-5).expect("RMSNorm should construct");
    let data = vec![0.5_f32; 8];
    let input = Tensor::from_vec(data, &[8]).expect("tensor should construct");
    let output = norm.forward(input).expect("forward should succeed");
    // All values equal → all outputs should equal each other after normalisation
    match &output {
        Tensor::F32(arr) => {
            let first = arr[[0]];
            assert!(
                arr.iter().all(|&v| (v - first).abs() < 1e-5),
                "uniform input must produce uniform output after RMSNorm"
            );
        },
        _ => panic!("expected F32"),
    }
}

// ── Vision patch embedding ─────────────────────────────────────────────

#[test]
fn test_vision_patch_embedding_construction() {
    let cfg = small_config();
    let emb = VisionPatchEmbedding::new(&cfg).expect("VisionPatchEmbedding should construct");
    assert_eq!(emb.num_patches(), cfg.num_patches);
    assert_eq!(emb.vision_hidden_size(), cfg.vision_hidden_size);
}

#[test]
fn test_vision_patch_embedding_num_patches_formula() {
    let cfg = small_config();
    let expected = (cfg.image_size / cfg.patch_size).pow(2);
    assert_eq!(
        cfg.num_patches, expected,
        "num_patches must equal (image/patch)^2"
    );
}

#[test]
fn test_vision_patch_embedding_parameter_count_positive() {
    let cfg = small_config();
    let emb = VisionPatchEmbedding::new(&cfg).expect("VisionPatchEmbedding should construct");
    assert!(emb.parameter_count() > 0);
}

#[test]
fn test_vision_patch_embed_patches_output_shape() {
    let cfg = small_config();
    let emb = VisionPatchEmbedding::new(&cfg).expect("VisionPatchEmbedding should construct");
    let h = cfg.image_size;
    let w = cfg.image_size;
    let pixel_values: Vec<f32> = {
        let mut st = 42u64;
        (0..h * w * 3).map(|_| lcg_next(&mut st)).collect()
    };
    let out = emb.embed_patches(&pixel_values, h, w).expect("embed_patches should succeed");
    let shape = out.shape();
    let expected_patches = cfg.num_patches;
    assert_eq!(
        shape[0], expected_patches,
        "output[0] must equal num_patches"
    );
    assert_eq!(
        shape[1], cfg.vision_hidden_size,
        "output[1] must equal vision_hidden_size"
    );
}

// ── RoPE ─────────────────────────────────────────────────────────────────

#[test]
fn test_rotary_embedding_inv_freq_count() {
    let cfg = small_config();
    let rope = Llama32RotaryEmbedding::new(
        cfg.head_dim,
        cfg.max_position_embeddings,
        cfg.rope_theta,
        cfg.rope_scaling_factor,
        cfg.use_scaled_rope,
    );
    assert_eq!(
        rope.half_dim(),
        cfg.head_dim / 2,
        "half_dim must be head_dim/2"
    );
}

#[test]
fn test_rope_apply_returns_same_shape() {
    let cfg = small_config();
    let rope = Llama32RotaryEmbedding::new(
        cfg.head_dim,
        cfg.max_position_embeddings,
        cfg.rope_theta,
        cfg.rope_scaling_factor,
        cfg.use_scaled_rope,
    );
    let seq_len = 4usize;
    let dim = cfg.head_dim;
    let data: Vec<f32> = {
        let mut st = 55u64;
        (0..seq_len * dim).map(|_| lcg_next(&mut st)).collect()
    };
    let q = Tensor::from_vec(data.clone(), &[seq_len, dim]).expect("q tensor should construct");
    let k = Tensor::from_vec(data, &[seq_len, dim]).expect("k tensor should construct");
    let positions: Vec<usize> = (0..seq_len).collect();
    let (q_rot, k_rot) = rope
        .apply_rotary_emb(&q, &k, &positions)
        .expect("apply_rotary_emb should succeed");
    assert_eq!(q_rot.shape(), q.shape(), "RoPE must preserve q shape");
    assert_eq!(k_rot.shape(), k.shape(), "RoPE must preserve k shape");
}

// ── Self-attention & decoder layer ────────────────────────────────────────

#[test]
fn test_self_attention_construction() {
    let cfg = small_config();
    let attn = Llama32SelfAttention::new(&cfg).expect("Llama32SelfAttention should construct");
    assert!(attn.parameter_count() > 0);
    assert_eq!(
        attn.num_query_groups,
        cfg.num_attention_heads / cfg.num_key_value_heads
    );
}

#[test]
fn test_decoder_layer_without_cross_attention() {
    let cfg = small_config();
    let layer = Llama32DecoderLayer::new(&cfg, false).expect("decoder layer should construct");
    assert!(
        !layer.has_cross_attention(),
        "layer without cross-attn flag must not have it"
    );
}

#[test]
fn test_decoder_layer_with_cross_attention() {
    let cfg = small_config();
    let layer = Llama32DecoderLayer::new(&cfg, true)
        .expect("decoder layer with cross-attn should construct");
    assert!(
        layer.has_cross_attention(),
        "layer must have cross-attention when requested"
    );
}

// ── Vision model ─────────────────────────────────────────────────────────

#[test]
fn test_vision_model_construction() {
    let cfg = small_config();
    let model = Llama32VisionModel::new(cfg).expect("Llama32VisionModel should construct");
    assert!(model.parameter_count() > 0, "model must have parameters");
}

#[test]
fn test_vision_model_text_only_forward() {
    let cfg = small_config();
    let model = Llama32VisionModel::new(cfg.clone()).expect("Llama32VisionModel should construct");
    let input_ids = vec![0u32, 1, 2];
    let out = model
        .forward_text_only(input_ids.clone())
        .expect("text-only forward should succeed");
    let shape = out.shape();
    // Output must have hidden_size as last dimension
    assert_eq!(
        shape[shape.len() - 1],
        cfg.hidden_size,
        "output last dim must equal hidden_size"
    );
}

#[test]
fn test_cross_attention_decoder_construction() {
    let cfg = small_config();
    let decoder = Llama32CrossAttentionDecoder::new(cfg.clone()).expect("decoder should construct");
    assert!(decoder.parameter_count() > 0);
    assert_eq!(
        decoder.config().num_hidden_layers,
        cfg.num_hidden_layers,
        "decoder must have correct number of layers"
    );
}

#[test]
fn test_lcg_values_in_range() {
    let mut state = 11111u64;
    for _ in 0..20 {
        let v = lcg_next(&mut state);
        assert!((0.0..1.0).contains(&v), "LCG value must be in [0,1)");
    }
}

// ── Real attention: SDPA reference, vision self-attention, cross-attention ──

fn lcg_data(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n).map(|_| lcg_next(&mut state) - 0.5).collect()
}

fn identity_matrix(size: usize) -> Vec<f32> {
    (0..size * size).map(|i| if i / size == i % size { 1.0 } else { 0.0 }).collect()
}

/// Hand-computed single-head attention:
/// q = [1, 0], k = [[1,0],[0,1]], v = [[2,3],[4,5]], head_dim = 2.
#[test]
fn test_multi_head_sdpa_matches_hand_computation() {
    let queries = vec![1.0f32, 0.0];
    let keys = vec![1.0f32, 0.0, 0.0, 1.0];
    let values = vec![2.0f32, 3.0, 4.0, 5.0];
    let out = multi_head_sdpa(&queries, &keys, &values, 1, 2, 1, 2, false).expect("sdpa");

    let scale = 1.0f32 / 2.0f32.sqrt();
    let e0 = (1.0f32 * scale).exp();
    let e1 = (0.0f32).exp();
    let sum = e0 + e1;
    let expected = [
        (e0 / sum) * 2.0 + (e1 / sum) * 4.0,
        (e0 / sum) * 3.0 + (e1 / sum) * 5.0,
    ];
    for (got, want) in out.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-6, "got {got}, expected {want}");
    }
}

#[test]
fn test_multi_head_sdpa_causal_masks_future() {
    let queries = vec![1.0f32, 0.0, 0.0, 1.0];
    let keys = queries.clone();
    let values = vec![1.0f32, 0.0, 0.0, 1.0];
    let out = multi_head_sdpa(&queries, &keys, &values, 2, 2, 1, 2, true).expect("sdpa");
    // The first query only sees the first value.
    assert!((out[0] - 1.0).abs() < 1e-6);
    assert!(out[1].abs() < 1e-6);
}

#[test]
fn test_multi_head_sdpa_rejects_shape_mismatch() {
    assert!(multi_head_sdpa(&[1.0, 0.0], &[1.0], &[1.0], 1, 1, 1, 2, false).is_err());
}

/// The vision encoder must read its own values: zeroing `v_proj` collapses
/// every output row onto the (constant) output-projection bias. The old code
/// forwarded a scaled query instead, so the rows stayed distinct.
#[test]
fn test_vision_attention_uses_values() {
    let cfg = small_config();
    let mut attn = VisionAttention::new(&cfg).expect("vision attention");
    let vision_hidden = cfg.vision_hidden_size;
    attn.v_proj
        .set_weight(
            Tensor::from_vec(
                vec![0.0f32; vision_hidden * vision_hidden],
                &[vision_hidden, vision_hidden],
            )
            .expect("zero weight"),
        )
        .expect("set v weight");
    attn.v_proj
        .set_bias(Tensor::zeros(&[vision_hidden]).expect("zero bias"))
        .expect("set v bias");

    let seq_len = 4;
    let input = Tensor::from_vec(
        lcg_data(seq_len * vision_hidden, 4242),
        &[seq_len, vision_hidden],
    )
    .expect("input");
    let out = attn.forward(input).expect("forward").data().expect("data");

    for token in 1..seq_len {
        for d in 0..vision_hidden {
            let first = out[d];
            let current = out[token * vision_hidden + d];
            assert!(
                (first - current).abs() < 1e-6,
                "with zero values every row must equal the projection bias"
            );
        }
    }
}

#[test]
fn test_vision_attention_matches_naive_reference() {
    let cfg = small_config();
    let mut attn = VisionAttention::new(&cfg).expect("vision attention");
    let width = cfg.vision_hidden_size;
    let head_dim = width / cfg.vision_num_attention_heads;
    let identity = Tensor::from_vec(identity_matrix(width), &[width, width]).expect("identity");
    for proj in [
        &mut attn.q_proj,
        &mut attn.k_proj,
        &mut attn.v_proj,
        &mut attn.out_proj,
    ] {
        proj.set_weight(identity.clone()).expect("weight");
        proj.set_bias(Tensor::zeros(&[width]).expect("bias")).expect("bias");
    }

    let seq_len = 3;
    let x = lcg_data(seq_len * width, 777);
    let got = attn
        .forward(Tensor::from_vec(x.clone(), &[seq_len, width]).expect("input"))
        .expect("forward")
        .data()
        .expect("data");

    let expected = multi_head_sdpa(
        &x,
        &x,
        &x,
        seq_len,
        seq_len,
        cfg.vision_num_attention_heads,
        head_dim,
        false,
    )
    .expect("reference");
    for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "element {i}: {a} vs {b}");
    }
}

/// Cross-attention must actually read the vision features. The old code
/// dropped K and V, so changing the image had no effect whatsoever.
#[test]
fn test_cross_attention_depends_on_vision_features() {
    let cfg = small_config();
    let layer = CrossAttentionLayer::new(&cfg).expect("cross attention");
    let seq_len = 3;
    let num_patches = 4;
    let text = Tensor::from_vec(
        lcg_data(seq_len * cfg.hidden_size, 31337),
        &[seq_len, cfg.hidden_size],
    )
    .expect("text");

    let vision_a = Tensor::from_vec(
        lcg_data(num_patches * cfg.vision_hidden_size, 555),
        &[num_patches, cfg.vision_hidden_size],
    )
    .expect("vision a");
    let vision_b = Tensor::from_vec(
        lcg_data(num_patches * cfg.vision_hidden_size, 556),
        &[num_patches, cfg.vision_hidden_size],
    )
    .expect("vision b");

    let out_a = layer
        .cross_attend(text.clone(), &vision_a)
        .expect("cross attend a")
        .data()
        .expect("data a");
    let out_b = layer
        .cross_attend(text, &vision_b)
        .expect("cross attend b")
        .data()
        .expect("data b");

    assert_eq!(out_a.len(), seq_len * cfg.hidden_size);
    let diff = out_a
        .iter()
        .zip(out_b.iter())
        .fold(0.0f32, |acc, (a, b)| acc.max((a - b).abs()));
    assert!(
        diff > 1e-6,
        "cross-attention output must change with the vision features (diff {diff})"
    );
}

#[test]
fn test_cross_attention_uses_vision_values() {
    let cfg = small_config();
    let mut layer = CrossAttentionLayer::new(&cfg).expect("cross attention");
    let total_head_dim = cfg.num_attention_heads * cfg.head_dim;
    layer
        .v_proj
        .set_weight(
            Tensor::from_vec(
                vec![0.0f32; total_head_dim * cfg.vision_hidden_size],
                &[total_head_dim, cfg.vision_hidden_size],
            )
            .expect("zero weight"),
        )
        .expect("set v weight");

    let seq_len = 2;
    let num_patches = 4;
    let text = Tensor::from_vec(
        lcg_data(seq_len * cfg.hidden_size, 24680),
        &[seq_len, cfg.hidden_size],
    )
    .expect("text");
    let vision = Tensor::from_vec(
        lcg_data(num_patches * cfg.vision_hidden_size, 13579),
        &[num_patches, cfg.vision_hidden_size],
    )
    .expect("vision");

    let out = layer.cross_attend(text, &vision).expect("cross attend").data().expect("data");
    for value in &out {
        assert!(
            value.abs() < 1e-6,
            "zero values (and a bias-free o_proj) must give a zero context, got {value}"
        );
    }
}

/// Batched vision self-attention: each image in the batch is attended on its
/// own, so running one item alone must reproduce its slice of the batch.
#[test]
fn test_vision_attention_batched_matches_single() {
    let cfg = small_config();
    let attn = VisionAttention::new(&cfg).expect("vision attention");
    let width = cfg.vision_hidden_size;
    let batch = 2;
    let seq_len = 3;
    let data = lcg_data(batch * seq_len * width, 8080);

    let batched = attn
        .forward(Tensor::from_vec(data.clone(), &[batch, seq_len, width]).expect("input"))
        .expect("batched forward")
        .data()
        .expect("data");
    assert_eq!(batched.len(), batch * seq_len * width);

    let single = attn
        .forward(
            Tensor::from_vec(data[..seq_len * width].to_vec(), &[1, seq_len, width])
                .expect("single"),
        )
        .expect("single forward")
        .data()
        .expect("data");
    for (i, expected) in single.iter().enumerate() {
        assert!(
            (batched[i] - expected).abs() < 1e-5,
            "batch element 0 diverged at {i}"
        );
    }
}

/// Batched cross-attention: text batch `b` must read vision batch `b` only.
#[test]
fn test_cross_attention_batched_pairs_streams() {
    let cfg = small_config();
    let layer = CrossAttentionLayer::new(&cfg).expect("cross attention");
    let batch = 2;
    let seq_len = 2;
    let num_patches = 3;
    let text_data = lcg_data(batch * seq_len * cfg.hidden_size, 606);
    let vision_data = lcg_data(batch * num_patches * cfg.vision_hidden_size, 707);

    let text =
        Tensor::from_vec(text_data.clone(), &[batch, seq_len, cfg.hidden_size]).expect("text");
    let vision = Tensor::from_vec(
        vision_data.clone(),
        &[batch, num_patches, cfg.vision_hidden_size],
    )
    .expect("vision");
    let batched = layer
        .cross_attend(text, &vision)
        .expect("batched cross attend")
        .data()
        .expect("data");
    assert_eq!(batched.len(), batch * seq_len * cfg.hidden_size);

    // Item 0 computed alone must match the batched result.
    let text0 = Tensor::from_vec(
        text_data[..seq_len * cfg.hidden_size].to_vec(),
        &[seq_len, cfg.hidden_size],
    )
    .expect("text0");
    let vision0 = Tensor::from_vec(
        vision_data[..num_patches * cfg.vision_hidden_size].to_vec(),
        &[num_patches, cfg.vision_hidden_size],
    )
    .expect("vision0");
    let single = layer
        .cross_attend(text0, &vision0)
        .expect("single cross attend")
        .data()
        .expect("data");
    for (i, expected) in single.iter().enumerate() {
        assert!(
            (batched[i] - expected).abs() < 1e-5,
            "batch element 0 diverged at {i}"
        );
    }
}

#[test]
fn test_cross_attention_rejects_batch_mismatch() {
    let cfg = small_config();
    let layer = CrossAttentionLayer::new(&cfg).expect("cross attention");
    let text = Tensor::from_vec(
        lcg_data(2 * 2 * cfg.hidden_size, 99),
        &[2, 2, cfg.hidden_size],
    )
    .expect("text");
    let vision = Tensor::from_vec(
        lcg_data(3 * 4 * cfg.vision_hidden_size, 98),
        &[3, 4, cfg.vision_hidden_size],
    )
    .expect("vision");
    assert!(layer.cross_attend(text, &vision).is_err());
}

#[test]
fn test_rms_norm_normalises_each_row() {
    let norm = Llama32RmsNorm::new(4, 1e-6).expect("norm");
    let input =
        Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0, 20.0, 20.0, 20.0, 20.0], &[2, 4]).expect("input");
    let out = norm.forward(input).expect("forward").data().expect("data");
    for value in out {
        assert!(
            (value - 1.0).abs() < 1e-3,
            "each row must be normalised on its own, got {value}"
        );
    }
}

// ── Text self-attention: real RoPE and real SDPA ──────────────────────────

/// Tiny text config: hidden 4 = 2 query heads × head_dim 2, 2 KV heads.
fn tiny_text_config() -> Llama32Config {
    Llama32Config {
        hidden_size: 4,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        head_dim: 2,
        rope_theta: 10000.0,
        use_scaled_rope: false,
        ..Llama32Config::small_test()
    }
}

fn identity_weight(n: usize) -> Tensor {
    Tensor::from_vec(identity_matrix(n), &[n, n]).expect("identity weight")
}

/// RoPE must rotate *every* head block. The old body cloned the input and threw
/// the angle away, so no head moved at all.
#[test]
fn test_llama32_rope_rotates_every_head_and_matches_hand_computation() {
    // head_dim = 2 → half = 1 → inv_freq = [1.0]; scaling disabled.
    let rope = Llama32RotaryEmbedding::new(2, 32, 10000.0, 1.0, false);
    assert_eq!(rope.half_dim(), 1);
    assert_eq!(rope.head_dim(), 2);
    assert_eq!(rope.max_seq_len(), 32);

    let values = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
    let q = Tensor::from_vec(values.clone(), &[2, 4]).expect("q");
    let k = Tensor::from_vec(values, &[2, 4]).expect("k");
    let (q_out, _) = rope.apply_rotary_emb(&q, &k, &[0, 1]).expect("rope");
    let data = q_out.data().expect("data");

    // Position 0 is the identity.
    assert!((data[0] - 1.0).abs() < 1e-6);
    assert!(data[1].abs() < 1e-6);

    let (sin, cos) = (1.0f32.sin(), 1.0f32.cos());
    // Head 0 at position 1: (1, 0) → (cos, sin).
    assert!((data[4] - cos).abs() < 1e-6, "head 0 x: {}", data[4]);
    assert!((data[5] - sin).abs() < 1e-6, "head 0 y: {}", data[5]);
    // Head 1 at position 1: (0, 1) → (-sin, cos). The old code never touched it.
    assert!((data[6] + sin).abs() < 1e-6, "head 1 x: {}", data[6]);
    assert!((data[7] - cos).abs() < 1e-6, "head 1 y: {}", data[7]);
}

/// Scaled RoPE divides the position, so position 2 under factor 2 must land
/// exactly where position 1 lands unscaled.
#[test]
fn test_llama32_scaled_rope_interpolates_positions() {
    let scaled = Llama32RotaryEmbedding::new(2, 32, 10000.0, 2.0, true);
    let plain = Llama32RotaryEmbedding::new(2, 32, 10000.0, 1.0, false);
    let x = Tensor::from_vec(vec![0.3, -0.7], &[1, 2]).expect("x");

    let (a, _) = scaled.apply_rotary_emb(&x, &x, &[2]).expect("scaled");
    let (b, _) = plain.apply_rotary_emb(&x, &x, &[1]).expect("plain");
    let (a, b) = (a.data().expect("a"), b.data().expect("b"));
    for i in 0..2 {
        assert!(
            (a[i] - b[i]).abs() < 1e-6,
            "scaled position 2 must equal plain position 1: {} vs {}",
            a[i],
            b[i]
        );
    }

    // ... and it must not be the identity.
    let (c, _) = scaled.apply_rotary_emb(&x, &x, &[0]).expect("pos 0");
    let c = c.data().expect("c");
    assert!(
        (a[0] - c[0]).abs() > 1e-4 || (a[1] - c[1]).abs() > 1e-4,
        "a non-zero position must actually rotate"
    );
}

/// The text attention output must depend on V. With `v_proj = 0` a real SDPA
/// returns zeros; the old code returned `o_proj(scale * Q)`, which does not.
#[test]
fn test_llama32_self_attention_reads_values() {
    let cfg = tiny_text_config();
    let mut attn = Llama32SelfAttention::new(&cfg).expect("attention");
    attn.q_proj.set_weight(identity_weight(4)).expect("q");
    attn.k_proj.set_weight(identity_weight(4)).expect("k");
    attn.o_proj.set_weight(identity_weight(4)).expect("o");
    attn.v_proj
        .set_weight(Tensor::from_vec(vec![0.0; 16], &[4, 4]).expect("zero"))
        .expect("v");

    let input =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]).expect("input");
    let out = attn.forward(input).expect("forward").data().expect("data");
    for value in out {
        assert!(
            value.abs() < 1e-6,
            "with zero values the context must be zero, got {value}"
        );
    }
}

/// Full independent reference: identity projections, RoPE recomputed in the
/// test, then causal softmax attention per head.
#[test]
fn test_llama32_self_attention_matches_naive_reference() {
    let cfg = tiny_text_config();
    let mut attn = Llama32SelfAttention::new(&cfg).expect("attention");
    attn.q_proj.set_weight(identity_weight(4)).expect("q");
    attn.k_proj.set_weight(identity_weight(4)).expect("k");
    attn.v_proj.set_weight(identity_weight(4)).expect("v");
    attn.o_proj.set_weight(identity_weight(4)).expect("o");

    let rows = [[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let got = attn
        .forward(Tensor::from_vec(flat, &[2, 4]).expect("input"))
        .expect("forward")
        .data()
        .expect("data");

    let (head_dim, half, heads, seq) = (2usize, 1usize, 2usize, 2usize);
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / (10000.0f32).powf(2.0 * i as f32 / head_dim as f32))
        .collect();

    let mut q_ref: Vec<Vec<f32>> = rows.iter().map(|r| r.to_vec()).collect();
    let mut k_ref = q_ref.clone();
    let v_ref = q_ref.clone();
    for (pos, (q_row, k_row)) in q_ref.iter_mut().zip(k_ref.iter_mut()).enumerate() {
        for head in 0..heads {
            let base = head * head_dim;
            for (i, &freq) in inv_freq.iter().enumerate() {
                let angle = pos as f32 * freq;
                let (sin, cos) = (angle.sin(), angle.cos());
                for row in [&mut *q_row, &mut *k_row] {
                    let x = row[base + i];
                    let y = row[base + i + half];
                    row[base + i] = x * cos - y * sin;
                    row[base + i + half] = x * sin + y * cos;
                }
            }
        }
    }

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut expected = vec![0.0f32; seq * heads * head_dim];
    for head in 0..heads {
        let base = head * head_dim;
        for query_pos in 0..seq {
            let scores: Vec<f32> = (0..=query_pos)
                .map(|key_pos| {
                    (0..head_dim)
                        .map(|d| q_ref[query_pos][base + d] * k_ref[key_pos][base + d])
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for (key_pos, weight) in exps.iter().enumerate() {
                for d in 0..head_dim {
                    expected[query_pos * heads * head_dim + base + d] +=
                        weight / sum * v_ref[key_pos][base + d];
                }
            }
        }
    }

    for (i, (actual, want)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - want).abs() < 1e-5,
            "element {i}: got {actual}, reference {want}"
        );
    }
}

/// Causality: a later token must not influence an earlier output.
#[test]
fn test_llama32_self_attention_is_causal() {
    let cfg = tiny_text_config();
    let attn = Llama32SelfAttention::new(&cfg).expect("attention");

    let first =
        Tensor::from_vec(vec![0.4, -0.3, 0.7, 0.1, 0.2, 0.9, -0.5, 0.3], &[2, 4]).expect("first");
    let second =
        Tensor::from_vec(vec![0.4, -0.3, 0.7, 0.1, -9.0, 4.0, 6.0, -2.0], &[2, 4]).expect("second");
    let a = attn.forward(first).expect("a").data().expect("a data");
    let b = attn.forward(second).expect("b").data().expect("b data");

    for i in 0..4 {
        assert!(
            (a[i] - b[i]).abs() < 1e-6,
            "token 0 must not see token 1: {} vs {}",
            a[i],
            b[i]
        );
    }
    assert!(
        (4..8).any(|i| (a[i] - b[i]).abs() > 1e-4),
        "token 1 must react to its own change"
    );
}

/// GQA with a real KV expansion: the default small config has 4 query heads and
/// 2 KV heads, so the forward must succeed and stay input-sensitive.
#[test]
fn test_llama32_self_attention_grouped_query_is_input_sensitive() {
    let cfg = small_config();
    assert_eq!(cfg.num_attention_heads / cfg.num_key_value_heads, 2);
    let attn = Llama32SelfAttention::new(&cfg).expect("attention");
    assert_eq!(attn.num_heads(), cfg.num_attention_heads);
    assert_eq!(attn.num_kv_heads(), cfg.num_key_value_heads);
    assert_eq!(attn.head_dim(), cfg.head_dim);

    let a =
        Tensor::from_vec(lcg_data(3 * cfg.hidden_size, 4242), &[3, cfg.hidden_size]).expect("a");
    let b = Tensor::from_vec(lcg_data(3 * cfg.hidden_size, 777), &[3, cfg.hidden_size]).expect("b");
    let out_a = attn.forward(a).expect("forward a").data().expect("a data");
    let out_b = attn.forward(b).expect("forward b").data().expect("b data");
    assert_eq!(out_a.len(), 3 * cfg.hidden_size);
    assert!(
        out_a.iter().zip(out_b.iter()).any(|(x, y)| (x - y).abs() > 1e-5),
        "different inputs must give different attention outputs"
    );
    assert!(
        out_a.iter().any(|v| v.abs() > 1e-6),
        "output must not be all zeros"
    );
}

/// Batched text attention must match running each batch item on its own.
#[test]
fn test_llama32_self_attention_batched_matches_single() {
    let cfg = tiny_text_config();
    let attn = Llama32SelfAttention::new(&cfg).expect("attention");

    let row_a = vec![0.1f32, -0.2, 0.3, 0.4, 0.5, 0.6, -0.7, 0.8];
    let row_b = vec![-0.9f32, 0.2, 0.1, 0.0, 0.4, -0.4, 0.6, 0.2];
    let mut batched = row_a.clone();
    batched.extend_from_slice(&row_b);

    let out_batched = attn
        .forward(Tensor::from_vec(batched, &[2, 2, 4]).expect("batched"))
        .expect("batched forward")
        .data()
        .expect("data");
    let out_a = attn
        .forward(Tensor::from_vec(row_a, &[2, 4]).expect("a"))
        .expect("a")
        .data()
        .expect("a data");
    let out_b = attn
        .forward(Tensor::from_vec(row_b, &[2, 4]).expect("b"))
        .expect("b")
        .data()
        .expect("b data");

    for i in 0..8 {
        assert!((out_batched[i] - out_a[i]).abs() < 1e-6, "batch 0 item {i}");
        assert!(
            (out_batched[8 + i] - out_b[i]).abs() < 1e-6,
            "batch 1 item {i}"
        );
    }
}

/// LayerNorm is per token: pooling mean/variance over the whole tensor would let
/// one patch shift another.
#[test]
fn test_vision_layer_norm_normalises_each_row() {
    let norm = VisionLayerNorm::new(4, 1e-6).expect("norm");
    let input = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 101.0, 102.0, 103.0, 104.0],
        &[2, 4],
    )
    .expect("input");
    let out = norm.forward(input).expect("forward").data().expect("data");

    // Both rows are the same ramp shifted by 100, so per-row normalisation gives
    // both rows identical output.
    for i in 0..4 {
        assert!(
            (out[i] - out[4 + i]).abs() < 1e-4,
            "row {i}: {} vs {}",
            out[i],
            out[4 + i]
        );
    }
    // Each row has zero mean after normalisation.
    let row_mean: f32 = out[..4].iter().sum::<f32>() / 4.0;
    assert!(row_mean.abs() < 1e-5, "row mean must be 0, got {row_mean}");
}
