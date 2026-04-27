use crate::gemma2::{
    apply_soft_cap_inplace, format_chat_prompt, geglu, gelu, soft_cap, Gemma2Config, Gemma2Error,
    Gemma2ForCausalLM,
};
use trustformers_core::{device::Device, tensor::Tensor, traits::Config};

// ── Tiny config for fast construction ────────────────────────────────────

fn tiny_config() -> Gemma2Config {
    let head_dim = 8usize;
    Gemma2Config {
        vocab_size: 64,
        hidden_size: 16,      // 2 heads × 8 head_dim
        num_hidden_layers: 4, // 2 local + 2 global
        num_attention_heads: 2,
        num_key_value_heads: 1, // GQA: 2x sharing
        intermediate_size: 32,
        head_dim,
        max_position_embeddings: 32,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-6,
        sliding_window: 16,
        attention_logit_softcapping: 50.0,
        final_logit_softcapping: 30.0,
        query_pre_attn_scalar: 1.0 / (head_dim as f64).sqrt(),
        model_type: "gemma2".to_string(),
    }
}

// ── Config presets ────────────────────────────────────────────────────────

#[test]
fn test_gemma2_2b_preset() {
    let cfg = Gemma2Config::gemma2_2b();
    assert_eq!(cfg.vocab_size, 256000, "2B vocab must be 256000");
    assert_eq!(cfg.num_hidden_layers, 26, "2B must have 26 layers");
    assert_eq!(cfg.head_dim, 256, "Gemma-2 head_dim is fixed at 256");
    assert_eq!(cfg.hidden_size, 2304);
    assert_eq!(cfg.num_attention_heads, 8);
    assert_eq!(cfg.num_key_value_heads, 4);
    assert!((cfg.attention_logit_softcapping - 50.0).abs() < 1e-9);
    assert!((cfg.final_logit_softcapping - 30.0).abs() < 1e-9);
    cfg.validate().expect("2B config must be valid");
}

#[test]
fn test_gemma2_9b_preset() {
    let cfg = Gemma2Config::gemma2_9b();
    assert_eq!(cfg.vocab_size, 256000, "9B vocab must be 256000");
    assert_eq!(cfg.num_hidden_layers, 42, "9B must have 42 layers");
    assert_eq!(cfg.head_dim, 256, "Gemma-2 head_dim is fixed at 256");
    assert_eq!(cfg.hidden_size, 3584);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.num_key_value_heads, 8);
    cfg.validate().expect("9B config must be valid");
}

// ── attn_logit_softcapping = 50.0 ────────────────────────────────────────

#[test]
fn test_gemma2_attn_logit_softcapping_value() {
    let cfg = Gemma2Config::gemma2_9b();
    assert!(
        (cfg.attention_logit_softcapping - 50.0).abs() < 1e-9,
        "attention_logit_softcapping must be 50.0"
    );
}

// ── final_logit_softcapping = 30.0 ───────────────────────────────────────

#[test]
fn test_gemma2_final_logit_softcapping_value() {
    let cfg = Gemma2Config::gemma2_9b();
    assert!(
        (cfg.final_logit_softcapping - 30.0).abs() < 1e-9,
        "final_logit_softcapping must be 30.0"
    );
}

// ── Alternating local/global attention ───────────────────────────────────

#[test]
fn test_alternating_attention_even_layers_local() {
    // Even layer indices are local (sliding window)
    assert!(Gemma2Config::is_local_layer(0), "layer 0 must be local");
    assert!(Gemma2Config::is_local_layer(2), "layer 2 must be local");
    assert!(Gemma2Config::is_local_layer(4), "layer 4 must be local");
    assert!(Gemma2Config::is_local_layer(40), "layer 40 must be local");
}

#[test]
fn test_alternating_attention_odd_layers_global() {
    // Odd layer indices are global (full causal attention)
    assert!(!Gemma2Config::is_local_layer(1), "layer 1 must be global");
    assert!(!Gemma2Config::is_local_layer(3), "layer 3 must be global");
    assert!(!Gemma2Config::is_local_layer(41), "layer 41 must be global");
}

#[test]
fn test_alternating_pattern_9b_all_layers() {
    let cfg = Gemma2Config::gemma2_9b();
    let mut local_count = 0usize;
    let mut global_count = 0usize;
    for i in 0..cfg.num_hidden_layers {
        if Gemma2Config::is_local_layer(i) {
            local_count += 1;
        } else {
            global_count += 1;
        }
    }
    // 42 layers: 21 local + 21 global
    assert_eq!(local_count, 21, "9B must have 21 local layers");
    assert_eq!(global_count, 21, "9B must have 21 global layers");
}

// ── GQA ──────────────────────────────────────────────────────────────────

#[test]
fn test_gemma2_gqa_kv_heads_less_than_q_heads() {
    let cfg_2b = Gemma2Config::gemma2_2b();
    assert!(
        cfg_2b.num_key_value_heads < cfg_2b.num_attention_heads,
        "2B: num_kv_heads must be less than num_heads (GQA)"
    );

    let cfg_9b = Gemma2Config::gemma2_9b();
    assert!(
        cfg_9b.num_key_value_heads < cfg_9b.num_attention_heads,
        "9B: num_kv_heads must be less than num_heads (GQA)"
    );
}

#[test]
fn test_gemma2_kv_group_size() {
    let cfg = Gemma2Config::gemma2_9b();
    assert_eq!(cfg.kv_group_size(), 2, "9B group size = 16 / 8 = 2");

    let cfg2b = Gemma2Config::gemma2_2b();
    assert_eq!(cfg2b.kv_group_size(), 2, "2B group size = 8 / 4 = 2");
}

// ── head_dim = 256 ────────────────────────────────────────────────────────

#[test]
fn test_gemma2_head_dim_fixed_at_256() {
    assert_eq!(Gemma2Config::gemma2_2b().head_dim, 256);
    assert_eq!(Gemma2Config::gemma2_9b().head_dim, 256);
}

// ── vocab_size = 256000 ───────────────────────────────────────────────────

#[test]
fn test_gemma2_vocab_size_256000() {
    assert_eq!(Gemma2Config::gemma2_2b().vocab_size, 256000);
    assert_eq!(Gemma2Config::gemma2_9b().vocab_size, 256000);
}

// ── architecture label ────────────────────────────────────────────────────

#[test]
fn test_gemma2_architecture_label() {
    let cfg = Gemma2Config::gemma2_9b();
    assert_eq!(cfg.architecture(), "Gemma-2");
}

// ── Config validation ─────────────────────────────────────────────────────

#[test]
fn test_gemma2_tiny_config_valid() {
    tiny_config().validate().expect("tiny config must pass validation");
}

#[test]
fn test_gemma2_config_invalid_kv_heads_not_divisor() {
    let cfg = Gemma2Config {
        num_attention_heads: 4,
        num_key_value_heads: 3, // 4 % 3 != 0
        ..tiny_config()
    };
    assert!(
        cfg.validate().is_err(),
        "num_heads not divisible by num_kv_heads must fail"
    );
}

#[test]
fn test_gemma2_config_invalid_zero_vocab() {
    let cfg = Gemma2Config {
        vocab_size: 0,
        ..tiny_config()
    };
    assert!(cfg.validate().is_err(), "vocab_size=0 must be invalid");
}

#[test]
fn test_gemma2_config_invalid_zero_sliding_window() {
    let cfg = Gemma2Config {
        sliding_window: 0,
        ..tiny_config()
    };
    assert!(cfg.validate().is_err(), "sliding_window=0 must be invalid");
}

#[test]
fn test_gemma2_config_invalid_zero_head_dim() {
    let cfg = Gemma2Config {
        head_dim: 0,
        ..tiny_config()
    };
    assert!(cfg.validate().is_err(), "head_dim=0 must be invalid");
}

// ── soft_cap scalar function ──────────────────────────────────────────────

#[test]
fn test_soft_cap_scalar_zero() {
    let v = soft_cap(0.0, 50.0);
    assert!(v.abs() < 1e-6, "soft_cap(0) must be 0, got {v}");
}

#[test]
fn test_soft_cap_scalar_large_positive_approaches_cap() {
    let v = soft_cap(1000.0, 50.0);
    assert!(
        (v - 50.0).abs() < 0.001,
        "large positive must approach cap=50, got {v}"
    );
}

#[test]
fn test_soft_cap_scalar_large_negative_approaches_neg_cap() {
    let v = soft_cap(-1000.0, 50.0);
    assert!(
        (v + 50.0).abs() < 0.001,
        "large negative must approach -cap=-50, got {v}"
    );
}

// ── apply_soft_cap_inplace ────────────────────────────────────────────────

#[test]
fn test_apply_soft_cap_inplace_bounds() {
    let cap = 30.0_f64;
    let mut data = vec![1000.0f32, -2000.0, 0.0, 15.0, -15.0];
    apply_soft_cap_inplace(&mut data, cap);
    for &v in &data {
        assert!(
            (-30.001..=30.001).contains(&v),
            "value {v} must be within [-30, 30]"
        );
    }
}

#[test]
fn test_apply_soft_cap_inplace_zero_unchanged() {
    let mut data = vec![0.0f32];
    apply_soft_cap_inplace(&mut data, 50.0);
    assert!(data[0].abs() < 1e-5, "soft_cap(0) must stay 0");
}

// ── GEGLU activation ──────────────────────────────────────────────────────

#[test]
fn test_geglu_length_preserved() {
    let gate = vec![1.0f32, 2.0, -1.0, 0.0];
    let up = vec![1.0f32, 1.0, 1.0, 1.0];
    let out = geglu(&gate, &up);
    assert_eq!(out.len(), 4, "geglu output length must match input");
}

#[test]
fn test_geglu_positive_gate_positive_output() {
    let gate = vec![1.0f32];
    let up = vec![1.0f32];
    let out = geglu(&gate, &up);
    assert!(
        out[0] > 0.0,
        "gelu(1.0)*1.0 must be positive, got {}",
        out[0]
    );
}

#[test]
fn test_geglu_zero_gate_zero_output() {
    let gate = vec![0.0f32, 0.0];
    let up = vec![5.0f32, 10.0];
    let out = geglu(&gate, &up);
    for &v in &out {
        assert!(v.abs() < 1e-5, "gelu(0)*up must be 0, got {v}");
    }
}

// ── gelu scalar ───────────────────────────────────────────────────────────

#[test]
fn test_gelu_zero() {
    assert!(gelu(0.0).abs() < 1e-5, "gelu(0) must be 0");
}

#[test]
fn test_gelu_positive_input() {
    assert!(gelu(1.0) > 0.0, "gelu(1.0) must be positive");
}

#[test]
fn test_gelu_large_approx_x() {
    let v = gelu(10.0);
    assert!((v - 10.0).abs() < 0.1, "gelu(10) ≈ 10, got {v}");
}

// ── Attention layer local/global flag ─────────────────────────────────────

#[test]
fn test_gemma2_attention_layer0_is_local() {
    use crate::gemma2::Gemma2Attention;
    let cfg = tiny_config();
    let attn = Gemma2Attention::new(&cfg, 0, Device::CPU).expect("attn creation");
    assert!(attn.is_local(), "layer 0 attention must be local");
}

#[test]
fn test_gemma2_attention_layer1_is_global() {
    use crate::gemma2::Gemma2Attention;
    let cfg = tiny_config();
    let attn = Gemma2Attention::new(&cfg, 1, Device::CPU).expect("attn creation");
    assert!(!attn.is_local(), "layer 1 attention must be global");
}

// ── Decoder layer forward ─────────────────────────────────────────────────

#[test]
fn test_gemma2_decoder_layer_forward_preserves_shape() {
    use crate::gemma2::Gemma2DecoderLayer;
    use trustformers_core::traits::Layer;

    let cfg = tiny_config();
    let layer = Gemma2DecoderLayer::new(&cfg, 0, Device::CPU).expect("layer creation");
    let input = Tensor::from_vec(vec![0.1f32; 16], &[1, 16]).expect("tensor");
    let out = layer.forward(input).expect("decoder layer forward");
    // Shape must be preserved
    let total: usize = out.shape().iter().product();
    assert_eq!(total, 16, "decoder layer must preserve total elements");
}

// ── CausalLM forward with logit softcapping ──────────────────────────────

#[test]
fn test_gemma2_causal_lm_forward_ids() {
    let cfg = tiny_config();
    let model = Gemma2ForCausalLM::new(cfg.clone()).expect("causal lm creation");
    let result = model.forward_ids(&[1u32, 2, 3]);
    assert!(result.is_ok(), "forward_ids failed: {:?}", result.err());
    let logits = result.expect("logits");
    let expected_len = 3 * cfg.vocab_size;
    assert_eq!(
        logits.len(),
        expected_len,
        "logits len must be seq_len * vocab_size"
    );
}

#[test]
fn test_gemma2_causal_lm_logits_bounded_by_softcap() {
    let cfg = tiny_config();
    let model = Gemma2ForCausalLM::new(cfg.clone()).expect("causal lm creation");
    let logits = model.forward_ids(&[1u32, 2]).expect("logits");
    let cap = cfg.final_logit_softcapping as f32;
    for &v in &logits {
        assert!(
            v >= -(cap + 0.001) && v <= cap + 0.001,
            "logit {v} must be within [-{cap}, {cap}]"
        );
    }
}

#[test]
fn test_gemma2_causal_lm_empty_input_error() {
    let cfg = tiny_config();
    let model = Gemma2ForCausalLM::new(cfg).expect("causal lm creation");
    let result = model.forward_ids(&[]);
    assert!(result.is_err(), "empty input must return an error");
    matches!(result.unwrap_err(), Gemma2Error::EmptyInput);
}

// ── Error display ─────────────────────────────────────────────────────────

#[test]
fn test_gemma2_error_display_invalid_config() {
    let s = format!("{}", Gemma2Error::InvalidConfig("bad_param".to_string()));
    assert!(
        s.contains("bad_param"),
        "InvalidConfig display must include the message"
    );
}

#[test]
fn test_gemma2_error_display_shape_mismatch() {
    let err = Gemma2Error::ShapeMismatch {
        expected: vec![2, 3],
        got: vec![4, 5],
    };
    let s = format!("{err}");
    assert!(
        s.contains("2") && s.contains("3"),
        "ShapeMismatch must show expected dims"
    );
    assert!(
        s.contains("4") && s.contains("5"),
        "ShapeMismatch must show got dims"
    );
}

#[test]
fn test_gemma2_error_display_sequence_too_long() {
    let err = Gemma2Error::SequenceTooLong {
        max: 4096,
        got: 8192,
    };
    let s = format!("{err}");
    assert!(s.contains("4096"), "SequenceTooLong must show max");
    assert!(s.contains("8192"), "SequenceTooLong must show got");
}

#[test]
fn test_gemma2_error_display_empty_input() {
    let s = format!("{}", Gemma2Error::EmptyInput);
    assert!(
        s.contains("empty") || s.contains("Empty"),
        "EmptyInput display must mention empty"
    );
}

// ── Chat formatting ───────────────────────────────────────────────────────

#[test]
fn test_gemma2_chat_format_structure() {
    let prompt = format_chat_prompt("Hello, Gemma!");
    assert!(
        prompt.contains("<start_of_turn>user"),
        "must contain user turn start"
    );
    assert!(
        prompt.contains("Hello, Gemma!"),
        "must contain user message"
    );
    assert!(prompt.contains("<end_of_turn>"), "must contain turn end");
    assert!(
        prompt.contains("<start_of_turn>model"),
        "must contain model turn start"
    );
}

// ── Greedy generation ─────────────────────────────────────────────────────

#[test]
fn test_gemma2_generate_returns_correct_length() {
    let cfg = tiny_config();
    let model = Gemma2ForCausalLM::new(cfg).expect("model");
    let result = model.generate(&[1u32, 2, 3], 3);
    assert!(result.is_ok(), "generate failed: {:?}", result.err());
    assert_eq!(
        result.expect("generated").len(),
        3,
        "must return exactly max_new_tokens tokens"
    );
}

#[test]
fn test_gemma2_generate_empty_input_error() {
    let cfg = tiny_config();
    let model = Gemma2ForCausalLM::new(cfg).expect("model");
    let result = model.generate(&[], 1);
    assert!(result.is_err(), "empty input must fail");
}

// ── query_pre_attn_scalar formula ─────────────────────────────────────────

#[test]
fn test_gemma2_query_pre_attn_scalar_formula() {
    let head_dim = 256usize;
    let expected = 1.0 / (head_dim as f64).sqrt();
    let cfg = Gemma2Config::gemma2_9b();
    assert!(
        (cfg.query_pre_attn_scalar - expected).abs() < 1e-9,
        "query_pre_attn_scalar must be 1/sqrt(head_dim)"
    );
}
