use crate::starcoder2::{
    config::StarCoder2Config,
    fim::{format_fim_prompt, parse_fim_output, FimTokens},
    model::{
        StarCoder2Attention, StarCoder2ForCausalLM, StarCoder2MLP, StarCoder2Model,
        StarCoder2RmsNorm, StarCoder2RotaryEmbedding,
    },
};
use trustformers_core::{tensor::Tensor, traits::Config, traits::Layer, traits::Model};

// ── Config preset tests ───────────────────────────────────────────────────────

#[test]
fn test_config_3b_preset() {
    let cfg = StarCoder2Config::starcoder2_3b();
    assert_eq!(cfg.vocab_size, 49152);
    assert_eq!(cfg.hidden_size, 3072);
    assert_eq!(cfg.intermediate_size, 12288);
    assert_eq!(cfg.num_hidden_layers, 30);
    assert_eq!(cfg.num_attention_heads, 24);
    assert_eq!(cfg.num_key_value_heads, 2);
    assert!(cfg.use_bias);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_7b_preset() {
    let cfg = StarCoder2Config::starcoder2_7b();
    assert_eq!(cfg.vocab_size, 49152);
    assert_eq!(cfg.hidden_size, 4608);
    assert_eq!(cfg.num_hidden_layers, 32);
    assert_eq!(cfg.num_key_value_heads, 2);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_15b_preset() {
    let cfg = StarCoder2Config::starcoder2_15b();
    assert_eq!(cfg.vocab_size, 49152);
    assert_eq!(cfg.hidden_size, 6144);
    assert_eq!(cfg.num_hidden_layers, 40);
    assert_eq!(cfg.num_key_value_heads, 2);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_num_kv_heads_is_two_for_all_sizes() {
    for cfg in [
        StarCoder2Config::starcoder2_3b(),
        StarCoder2Config::starcoder2_7b(),
        StarCoder2Config::starcoder2_15b(),
    ] {
        assert_eq!(
            cfg.num_key_value_heads, 2,
            "Expected 2 KV heads for all sizes"
        );
    }
}

#[test]
fn test_config_use_bias_true_for_all_presets() {
    for cfg in [
        StarCoder2Config::starcoder2_3b(),
        StarCoder2Config::starcoder2_7b(),
        StarCoder2Config::starcoder2_15b(),
    ] {
        assert!(cfg.use_bias, "StarCoder2 should always have bias=true");
    }
}

#[test]
fn test_config_head_dim_and_query_groups() {
    let cfg = StarCoder2Config::starcoder2_3b();
    // head_dim = 3072 / 24 = 128
    assert_eq!(cfg.head_dim(), 128);
    // query groups = 24 / 2 = 12
    assert_eq!(cfg.num_query_groups(), 12);
}

#[test]
fn test_config_small_test() {
    let cfg = StarCoder2Config::small_test();
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.num_key_value_heads, 2);
    assert!(cfg.use_bias);
}

// ── RMS norm test ─────────────────────────────────────────────────────────────

#[test]
fn test_rms_norm_forward() {
    let norm = StarCoder2RmsNorm::new(4, 1e-5).expect("build norm");
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]).expect("build tensor");
    let output = norm.forward(input).expect("forward");
    assert_eq!(output.shape(), &[4]);
}

// ── Rotary embedding test ─────────────────────────────────────────────────────

#[test]
fn test_rotary_embedding_half_dim() {
    let rope = StarCoder2RotaryEmbedding::new(128, 16384, 10000.0);
    assert_eq!(rope.half_dim(), 64);
    assert_eq!(rope.head_dim, 128);
}

#[test]
fn test_rotary_embedding_apply() {
    let rope = StarCoder2RotaryEmbedding::new(16, 64, 10000.0);
    let q = Tensor::from_vec(vec![0.1f32; 16], &[16]).expect("q");
    let k = Tensor::from_vec(vec![0.2f32; 16], &[16]).expect("k");
    let (q_r, k_r) = rope.apply_rotary_emb(&q, &k, &[0, 1]).expect("apply");
    assert_eq!(q_r.shape(), q.shape());
    assert_eq!(k_r.shape(), k.shape());
}

// ── Attention test ────────────────────────────────────────────────────────────

#[test]
fn test_attention_num_kv_heads() {
    let cfg = StarCoder2Config::small_test();
    let attn = StarCoder2Attention::new(&cfg).expect("build attn");
    assert_eq!(attn.num_kv_heads(), 2);
    assert_eq!(attn.num_heads(), cfg.num_attention_heads);
    assert_eq!(attn.head_dim(), cfg.head_dim());
}

// ── FIM tests ─────────────────────────────────────────────────────────────────

#[test]
fn test_fim_format_prompt() {
    let prompt = format_fim_prompt("prefix_code", "suffix_code");
    assert_eq!(
        prompt,
        "<fim_prefix>prefix_code<fim_suffix>suffix_code<fim_middle>"
    );
}

#[test]
fn test_fim_parse_output() {
    let out = "<fim_prefix>x<fim_suffix>z<fim_middle>middle_part<|endoftext|>";
    let parsed = parse_fim_output(out).expect("parse");
    assert_eq!(parsed, "middle_part");
}

#[test]
fn test_fim_parse_no_eot() {
    let out = "<fim_middle>generated code here";
    assert_eq!(
        parse_fim_output(out),
        Some("generated code here".to_string())
    );
}

#[test]
fn test_fim_tokens_default() {
    let tokens = FimTokens::default();
    assert_eq!(tokens.prefix_id, 1);
    assert_eq!(tokens.suffix_id, 3);
}

// ── Model forward test ────────────────────────────────────────────────────────

#[test]
fn test_model_forward_small() {
    let cfg = StarCoder2Config::small_test();
    let model = StarCoder2Model::new(cfg).expect("build model");
    let output = model.forward(vec![1u32, 2, 3]).expect("forward");
    // output shape: [1, seq_len, hidden_size]
    let shape = output.shape().to_vec();
    assert_eq!(shape[0], 1);
    assert_eq!(shape[1], 3);
}

#[test]
fn test_causal_lm_forward_small() {
    let cfg = StarCoder2Config::small_test();
    let lm = StarCoder2ForCausalLM::new(cfg.clone()).expect("build lm");
    let logits = lm.forward(vec![1u32, 2]).expect("forward");
    // shape: [1, seq_len, vocab_size]
    let shape = logits.shape().to_vec();
    assert_eq!(shape[0], 1);
    assert_eq!(shape[1], 2);
    assert_eq!(shape[2], cfg.vocab_size);
}

// ── MLP test ──────────────────────────────────────────────────────────────────

#[test]
fn test_mlp_parameter_count_includes_bias() {
    let cfg = StarCoder2Config::small_test();
    assert!(cfg.use_bias);
    let mlp = StarCoder2MLP::new(&cfg).expect("build mlp");
    // With bias: each Linear has (in*out + out) params
    // gate: 64*128 + 128, up: same, down: 128*64 + 64
    let expected_no_bias =
        cfg.hidden_size * cfg.intermediate_size * 2 + cfg.intermediate_size * cfg.hidden_size;
    // With bias the count is strictly larger
    assert!(mlp.parameter_count() > expected_no_bias);
}

// ── Additional FIM tests ──────────────────────────────────────────────────────

#[test]
fn test_fim_format_psm_ordering() {
    // Verify Prefix-Suffix-Middle ordering is exact.
    let prompt = format_fim_prompt("AAA", "BBB");
    let prefix_pos = prompt.find("<fim_prefix>").expect("no prefix marker");
    let suffix_pos = prompt.find("<fim_suffix>").expect("no suffix marker");
    let middle_pos = prompt.find("<fim_middle>").expect("no middle marker");
    assert!(prefix_pos < suffix_pos, "prefix must come before suffix");
    assert!(suffix_pos < middle_pos, "suffix must come before middle");
}

#[test]
fn test_fim_format_empty_prefix() {
    let prompt = format_fim_prompt("", "some_suffix");
    assert_eq!(prompt, "<fim_prefix><fim_suffix>some_suffix<fim_middle>");
}

#[test]
fn test_fim_parse_empty_middle() {
    // Middle section may be empty (e.g. model generated nothing).
    let raw = "<fim_prefix>x<fim_suffix>y<fim_middle><|endoftext|>";
    let middle = parse_fim_output(raw);
    assert_eq!(middle, Some(String::new()));
}

#[test]
fn test_fim_parse_middle_with_newlines() {
    let raw = "<fim_middle>line1\nline2\nline3<|endoftext|>";
    let middle = parse_fim_output(raw);
    assert_eq!(middle, Some("line1\nline2\nline3".to_string()));
}

#[test]
fn test_fim_tokens_all_fields() {
    let tokens = FimTokens::default();
    assert_eq!(tokens.prefix_id, 1);
    assert_eq!(tokens.middle_id, 2);
    assert_eq!(tokens.suffix_id, 3);
    assert_eq!(tokens.pad_id, 4);
    // All IDs should be distinct.
    let ids = [
        tokens.prefix_id,
        tokens.middle_id,
        tokens.suffix_id,
        tokens.pad_id,
    ];
    let unique: std::collections::HashSet<u32> = ids.iter().copied().collect();
    assert_eq!(unique.len(), 4, "all FIM token IDs must be distinct");
}

#[test]
fn test_fim_tokens_custom() {
    // Ensure the struct is freely constructible.
    let custom = FimTokens {
        prefix_id: 10,
        middle_id: 11,
        suffix_id: 12,
        pad_id: 13,
    };
    assert_eq!(custom.prefix_id, 10);
    assert_eq!(custom.middle_id, 11);
    assert_eq!(custom.suffix_id, 12);
    assert_eq!(custom.pad_id, 13);
}

// ── Additional config / architecture tests ────────────────────────────────────

#[test]
fn test_config_default_is_3b() {
    let default = StarCoder2Config::default();
    let three_b = StarCoder2Config::starcoder2_3b();
    assert_eq!(default.hidden_size, three_b.hidden_size);
    assert_eq!(default.num_hidden_layers, three_b.num_hidden_layers);
}

#[test]
fn test_config_rope_theta() {
    // All presets should use θ = 10 000.
    for cfg in [
        StarCoder2Config::starcoder2_3b(),
        StarCoder2Config::starcoder2_7b(),
        StarCoder2Config::starcoder2_15b(),
    ] {
        assert!(
            (cfg.rope_theta - 10000.0).abs() < 1.0,
            "rope_theta should be 10000"
        );
    }
}

#[test]
fn test_config_sliding_window_none_for_released() {
    for cfg in [
        StarCoder2Config::starcoder2_3b(),
        StarCoder2Config::starcoder2_7b(),
        StarCoder2Config::starcoder2_15b(),
    ] {
        assert!(
            cfg.sliding_window.is_none(),
            "no sliding window in released checkpoints"
        );
    }
}

#[test]
fn test_config_max_position_embeddings() {
    for cfg in [
        StarCoder2Config::starcoder2_3b(),
        StarCoder2Config::starcoder2_7b(),
        StarCoder2Config::starcoder2_15b(),
    ] {
        assert_eq!(
            cfg.max_position_embeddings, 16384,
            "all presets support 16K context"
        );
    }
}

#[test]
fn test_rms_norm_unit_rms() {
    let norm = StarCoder2RmsNorm::new(4, 1e-5).expect("build norm");
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]).expect("tensor");
    let output = norm.forward(input).expect("forward");
    // Check RMS of output ≈ 1.
    let out_vec: Vec<f32> = match &output {
        trustformers_core::tensor::Tensor::F32(arr) => arr.iter().copied().collect(),
        _ => panic!("expected F32"),
    };
    let rms: f32 = (out_vec.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
    assert!(
        (rms - 1.0).abs() < 0.01,
        "RMS of output should be ≈ 1, got {rms}"
    );
}

#[test]
fn test_model_single_token_forward() {
    let cfg = StarCoder2Config::small_test();
    let model = StarCoder2Model::new(cfg.clone()).expect("build model");
    let output = model.forward(vec![7u32]).expect("single token forward");
    let shape = output.shape().to_vec();
    assert_eq!(shape[0], 1, "batch");
    assert_eq!(shape[1], 1, "seq_len=1");
    assert_eq!(shape[2], cfg.hidden_size);
}
