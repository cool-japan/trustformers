use super::config::Qwen2Config;
use trustformers_core::traits::Config;
use super::model::{
    Qwen2Attention, Qwen2ForCausalLM, Qwen2MLP, Qwen2Model, Qwen2RmsNorm,
    Qwen2RotaryEmbedding,
};
use super::tasks::format_qwen2_chat;
use trustformers_core::{layers::Linear, tensor::Tensor, traits::Layer};

// ─────────────────────────────────────────────────────────────────────────────
// Config preset tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_qwen2_config_0_5b_vocab_size() {
    let cfg = Qwen2Config::qwen2_0_5b();
    assert_eq!(cfg.vocab_size, 151936);
}

#[test]
fn test_qwen2_config_0_5b_kv_heads() {
    let cfg = Qwen2Config::qwen2_0_5b();
    assert_eq!(cfg.num_key_value_heads, 2);
    assert_eq!(cfg.num_attention_heads, 14);
}

#[test]
fn test_qwen2_config_7b_intermediate() {
    let cfg = Qwen2Config::qwen2_7b();
    assert_eq!(cfg.intermediate_size, 18944);
    assert_eq!(cfg.num_key_value_heads, 4);
}

#[test]
fn test_qwen2_config_72b_layers() {
    let cfg = Qwen2Config::qwen2_72b();
    assert_eq!(cfg.num_hidden_layers, 80);
    assert_eq!(cfg.num_key_value_heads, 8);
}

#[test]
fn test_qwen2_config_rope_theta() {
    let cfg = Qwen2Config::qwen2_7b();
    // rope_theta must be 1,000,000 (10× higher than LLaMA-3)
    assert!((cfg.rope_theta - 1_000_000.0_f64).abs() < 1e-6);
}

#[test]
fn test_qwen2_config_qkv_bias() {
    let cfg = Qwen2Config::small_test();
    assert!(cfg.qkv_bias);
}

#[test]
fn test_qwen2_config_validation_ok() {
    let cfg = Qwen2Config::small_test();
    assert!(cfg.validate().is_ok());
}

// ─────────────────────────────────────────────────────────────────────────────
// RMS norm
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_qwen2_rms_norm_unit_vector() {
    let norm = Qwen2RmsNorm::new(4, 1e-6).expect("RmsNorm::new");
    // Input is already a unit-ish vector; output should be close to it
    let input = Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0], &[4])
        .expect("tensor");
    let out = norm.forward(input).expect("forward");
    match out {
        Tensor::F32(arr) => {
            // After RMSNorm the first element should still dominate
            assert!(arr[[0]] > 0.5, "first element should dominate: {arr:?}");
        },
        _ => panic!("expected F32 tensor"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rotary embeddings
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_qwen2_rope_inv_freq_length() {
    let rope = Qwen2RotaryEmbedding::new(16, 64, 1_000_000.0);
    assert_eq!(rope.half_dim(), 8);
}

#[test]
fn test_qwen2_rope_theta_effect() {
    // Higher theta → lower inv_freq values (wider period)
    let rope_high = Qwen2RotaryEmbedding::new(16, 64, 1_000_000.0);
    let rope_low = Qwen2RotaryEmbedding::new(16, 64, 10_000.0);
    // First inv_freq component (i=0): 1/theta^0 = 1.0 for both; check i=1
    assert!(
        rope_high.inv_freq[1] < rope_low.inv_freq[1],
        "higher theta should produce smaller inv_freq"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// GQA repeat_kv
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_qwen2_repeat_kv_expansion() {
    let cfg = Qwen2Config::small_test(); // 4 Q heads, 2 KV heads → groups=2
    let attn = Qwen2Attention::new(&cfg).expect("Qwen2Attention::new");

    // KV tensor: [head_dim] = [16]
    let head_dim = cfg.head_dim();
    let kv = Tensor::from_vec(vec![1.0_f32; head_dim], &[head_dim])
        .expect("tensor");
    let expanded = attn.repeat_kv(&kv).expect("repeat_kv");
    // Should be doubled in last dim
    assert_eq!(
        expanded.shape().iter().product::<usize>(),
        head_dim * cfg.num_query_groups()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Attention with qkv_bias
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_qwen2_attention_bias_present() {
    let cfg = Qwen2Config::small_test();
    let attn = Qwen2Attention::new(&cfg).expect("Qwen2Attention::new");
    assert!(attn.q_bias.is_some(), "q_bias should be Some when qkv_bias=true");
    assert!(attn.k_bias.is_some(), "k_bias should be Some when qkv_bias=true");
}

#[test]
fn test_qwen2_attention_forward_shape() {
    let cfg = Qwen2Config::small_test();
    let attn = Qwen2Attention::new(&cfg).expect("Qwen2Attention::new");
    let seq = 3_usize;
    let input = Tensor::from_vec(
        vec![0.1_f32; seq * cfg.hidden_size],
        &[seq, cfg.hidden_size],
    )
    .expect("tensor");
    let out = attn.forward(input).expect("forward");
    // Output should have hidden_size columns
    let out_elems: usize = out.shape().iter().product();
    assert_eq!(out_elems, seq * cfg.hidden_size);
}

// ─────────────────────────────────────────────────────────────────────────────
// SwiGLU MLP
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_qwen2_mlp_forward_shape() {
    let cfg = Qwen2Config::small_test();
    let mlp = Qwen2MLP::new(&cfg).expect("Qwen2MLP::new");
    // Linear layers require at least 2D input
    let input = Tensor::from_vec(
        vec![0.5_f32; cfg.hidden_size],
        &[1, cfg.hidden_size],
    )
    .expect("tensor");
    let out = mlp.forward(input).expect("forward");
    assert_eq!(out.shape().iter().product::<usize>(), cfg.hidden_size);
}

// ─────────────────────────────────────────────────────────────────────────────
// Full model forward
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_qwen2_model_forward_small() {
    let cfg = Qwen2Config::small_test();
    let model = Qwen2Model::new(cfg.clone()).expect("Qwen2Model::new");
    let input_ids = vec![1_u32, 2, 3];
    let out = model.run(input_ids).expect("run");
    // Output shape: [seq_len, hidden_size]
    assert_eq!(out.shape().iter().product::<usize>(), 3 * cfg.hidden_size);
}

#[test]
fn test_qwen2_causal_lm_forward_small() {
    let cfg = Qwen2Config::small_test();
    let model = Qwen2ForCausalLM::new(cfg.clone()).expect("Qwen2ForCausalLM::new");
    let input_ids = vec![0_u32, 1, 2];
    let logits = model.forward(input_ids).expect("forward");
    // Output shape: [seq_len, vocab_size]
    assert_eq!(
        logits.shape().iter().product::<usize>(),
        3 * cfg.vocab_size
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Chat formatting
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_qwen2_chat_format_contains_im_start() {
    let messages = vec![
        ("user".to_string(), "Hello!".to_string()),
    ];
    let prompt = format_qwen2_chat("You are helpful.", &messages);
    assert!(
        prompt.contains("<|im_start|>"),
        "prompt should contain <|im_start|>"
    );
    assert!(
        prompt.contains("<|im_end|>"),
        "prompt should contain <|im_end|>"
    );
}

#[test]
fn test_qwen2_chat_format_system_block() {
    let system = "You are a helpful assistant.";
    let messages: Vec<(String, String)> = vec![];
    let prompt = format_qwen2_chat(system, &messages);
    assert!(prompt.contains("system"), "prompt should contain system role");
    assert!(prompt.contains(system), "prompt should contain system content");
}

#[test]
fn test_qwen2_chat_format_ends_with_assistant() {
    let messages = vec![("user".to_string(), "Hi".to_string())];
    let prompt = format_qwen2_chat("", &messages);
    assert!(
        prompt.ends_with("<|im_start|>assistant\n"),
        "prompt should end with open assistant turn"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear layer sanity (tests trustformers_core usage pattern)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_linear_forward_basic() {
    // Ensure the Linear layer runs without error (smoke test)
    let layer = Linear::new(4, 8, false);
    let input = Tensor::from_vec(vec![1.0_f32; 4], &[1, 4]).expect("tensor");
    let out = layer.forward(input).expect("forward");
    assert_eq!(out.shape().iter().product::<usize>(), 8);
}

// ─────────────────────────────────────────────────────────────────────────────
// Extended config preset tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_qwen2_0_5b_head_dim() {
    let cfg = Qwen2Config::qwen2_0_5b();
    // head_dim = hidden_size / num_attention_heads = 896 / 14 = 64
    assert_eq!(cfg.head_dim(), 64);
}

#[test]
fn test_qwen2_7b_head_dim() {
    let cfg = Qwen2Config::qwen2_7b();
    // head_dim = 3584 / 28 = 128
    assert_eq!(cfg.head_dim(), 128);
}

#[test]
fn test_qwen2_72b_head_dim() {
    let cfg = Qwen2Config::qwen2_72b();
    // head_dim = 8192 / 64 = 128
    assert_eq!(cfg.head_dim(), 128);
}

#[test]
fn test_qwen2_0_5b_no_sliding_window() {
    let cfg = Qwen2Config::qwen2_0_5b();
    assert!(!cfg.use_sliding_window, "0.5B should not use sliding window");
    assert!(cfg.sliding_window.is_none(), "0.5B sliding_window should be None");
}

#[test]
fn test_qwen2_7b_sliding_window_enabled() {
    let cfg = Qwen2Config::qwen2_7b();
    assert!(cfg.use_sliding_window, "7B should use sliding window");
    assert_eq!(cfg.sliding_window, Some(32768));
}

#[test]
fn test_qwen2_72b_sliding_window_enabled() {
    let cfg = Qwen2Config::qwen2_72b();
    assert!(cfg.use_sliding_window);
    assert_eq!(cfg.sliding_window, Some(32768));
}

#[test]
fn test_qwen2_gqa_0_5b() {
    let cfg = Qwen2Config::qwen2_0_5b();
    // 14 Q heads / 2 KV heads = group size 7
    assert!(cfg.uses_gqa());
    assert_eq!(cfg.num_query_groups(), 7);
}

#[test]
fn test_qwen2_gqa_72b() {
    let cfg = Qwen2Config::qwen2_72b();
    // 64 Q heads / 8 KV heads = 8
    assert!(cfg.uses_gqa());
    assert_eq!(cfg.num_query_groups(), 8);
}

#[test]
fn test_qwen2_all_presets_vocab_size() {
    for cfg in [
        Qwen2Config::qwen2_0_5b(),
        Qwen2Config::qwen2_7b(),
        Qwen2Config::qwen2_72b(),
    ] {
        assert_eq!(cfg.vocab_size, 151936, "all Qwen2 presets must share vocab_size=151936");
    }
}

#[test]
fn test_qwen2_all_presets_rope_theta() {
    for cfg in [
        Qwen2Config::qwen2_0_5b(),
        Qwen2Config::qwen2_7b(),
        Qwen2Config::qwen2_72b(),
    ] {
        assert!(
            (cfg.rope_theta - 1_000_000.0_f64).abs() < 1e-3,
            "all Qwen2 presets must have rope_theta=1_000_000"
        );
    }
}

#[test]
fn test_qwen2_all_presets_validate() {
    for cfg in [
        Qwen2Config::qwen2_0_5b(),
        Qwen2Config::qwen2_7b(),
        Qwen2Config::qwen2_72b(),
        Qwen2Config::small_test(),
    ] {
        assert!(cfg.validate().is_ok(), "preset config should validate cleanly");
    }
}

#[test]
fn test_qwen2_config_architecture_string() {
    let cfg = Qwen2Config::default();
    assert_eq!(cfg.architecture(), "Qwen2");
}

#[test]
fn test_qwen2_config_default_is_7b() {
    let default_cfg = Qwen2Config::default();
    let qwen2_7b = Qwen2Config::qwen2_7b();
    assert_eq!(default_cfg.hidden_size, qwen2_7b.hidden_size);
    assert_eq!(default_cfg.num_hidden_layers, qwen2_7b.num_hidden_layers);
}

#[test]
fn test_qwen2_config_clone() {
    let cfg = Qwen2Config::qwen2_7b();
    let cloned = cfg.clone();
    assert_eq!(cloned.hidden_size, cfg.hidden_size);
    assert_eq!(cloned.vocab_size, cfg.vocab_size);
    assert_eq!(cloned.num_attention_heads, cfg.num_attention_heads);
}

#[test]
fn test_qwen2_config_debug() {
    let cfg = Qwen2Config::small_test();
    let debug_str = format!("{:?}", cfg);
    assert!(debug_str.contains("Qwen2Config"));
    assert!(debug_str.contains("vocab_size"));
    assert!(debug_str.contains("hidden_size"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Config validation error cases
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_qwen2_validate_zero_vocab_size() {
    let mut cfg = Qwen2Config::small_test();
    cfg.vocab_size = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_qwen2_validate_zero_hidden_size() {
    let mut cfg = Qwen2Config::small_test();
    cfg.hidden_size = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_qwen2_validate_zero_num_hidden_layers() {
    let mut cfg = Qwen2Config::small_test();
    cfg.num_hidden_layers = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_qwen2_validate_zero_intermediate_size() {
    let mut cfg = Qwen2Config::small_test();
    cfg.intermediate_size = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_qwen2_validate_bad_kv_heads_divisibility() {
    let mut cfg = Qwen2Config::small_test();
    // 4 Q heads, 3 KV heads → not divisible
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 3;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_qwen2_validate_bad_hidden_not_divisible_by_heads() {
    let mut cfg = Qwen2Config::small_test();
    // hidden_size=64, num_attention_heads=5 → not divisible
    cfg.hidden_size = 64;
    cfg.num_attention_heads = 5;
    assert!(cfg.validate().is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// RMSNorm extended tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_qwen2_rms_norm_parameter_count() {
    let norm = Qwen2RmsNorm::new(128, 1e-6).expect("RmsNorm::new");
    assert_eq!(norm.parameter_count(), 128);
}

#[test]
fn test_qwen2_rms_norm_uniform_input() {
    // For a uniform input vector, RMSNorm output should also be uniform
    let norm = Qwen2RmsNorm::new(8, 1e-6).expect("RmsNorm::new");
    let input = Tensor::from_vec(vec![2.0_f32; 8], &[8]).expect("tensor");
    let out = norm.forward(input).expect("forward");
    match out {
        Tensor::F32(arr) => {
            let v0 = arr[[0]];
            // All elements should be equal
            for &v in arr.iter() {
                assert!((v - v0).abs() < 1e-5, "uniform input → uniform output");
            }
        },
        _ => panic!("expected F32"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RoPE extended tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_qwen2_rope_apply_preserves_shape() {
    let cfg = Qwen2Config::small_test();
    let rope = Qwen2RotaryEmbedding::new(cfg.head_dim(), cfg.max_position_embeddings, cfg.rope_theta);
    let q = Tensor::from_vec(vec![0.5_f32; cfg.head_dim()], &[cfg.head_dim()])
        .expect("tensor");
    let k = Tensor::from_vec(vec![0.5_f32; cfg.head_dim()], &[cfg.head_dim()])
        .expect("tensor");
    let positions: Vec<usize> = (0..1).collect();
    let (q_out, k_out) = rope.apply_rotary_emb(&q, &k, &positions).expect("apply_rotary_emb");
    assert_eq!(q_out.shape(), q.shape());
    assert_eq!(k_out.shape(), k.shape());
}

#[test]
fn test_qwen2_rope_first_inv_freq_is_one() {
    // At i=0, inv_freq = 1 / theta^(0/head_dim) = 1.0
    let rope = Qwen2RotaryEmbedding::new(16, 64, 1_000_000.0);
    assert!((rope.inv_freq[0] - 1.0_f64).abs() < 1e-9,
        "first inv_freq component must be 1.0, got {}", rope.inv_freq[0]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Attention extended tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_qwen2_attention_no_bias_when_qkv_bias_false() {
    let mut cfg = Qwen2Config::small_test();
    cfg.qkv_bias = false;
    let attn = Qwen2Attention::new(&cfg).expect("Qwen2Attention::new");
    assert!(attn.q_bias.is_none(), "q_bias should be None when qkv_bias=false");
    assert!(attn.k_bias.is_none(), "k_bias should be None when qkv_bias=false");
}

#[test]
fn test_qwen2_attention_q_bias_length() {
    let cfg = Qwen2Config::small_test();
    let attn = Qwen2Attention::new(&cfg).expect("Qwen2Attention::new");
    let expected_q_dim = cfg.num_attention_heads * cfg.head_dim();
    if let Some(ref bias) = attn.q_bias {
        assert_eq!(bias.len(), expected_q_dim);
    }
}

#[test]
fn test_qwen2_attention_k_bias_length() {
    let cfg = Qwen2Config::small_test();
    let attn = Qwen2Attention::new(&cfg).expect("Qwen2Attention::new");
    let expected_k_dim = cfg.num_key_value_heads * cfg.head_dim();
    if let Some(ref bias) = attn.k_bias {
        assert_eq!(bias.len(), expected_k_dim);
    }
}

#[test]
fn test_qwen2_attention_accessors() {
    let cfg = Qwen2Config::small_test();
    let attn = Qwen2Attention::new(&cfg).expect("Qwen2Attention::new");
    assert_eq!(attn.num_heads(), cfg.num_attention_heads);
    assert_eq!(attn.num_kv_heads(), cfg.num_key_value_heads);
    assert_eq!(attn.head_dim(), cfg.head_dim());
}

// ─────────────────────────────────────────────────────────────────────────────
// Model parameter count sanity
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_qwen2_model_parameter_count_positive() {
    let cfg = Qwen2Config::small_test();
    let model = super::model::Qwen2Model::new(cfg).expect("Qwen2Model::new");
    assert!(model.parameter_count() > 0, "model should have parameters");
}

#[test]
fn test_qwen2_causal_lm_parameter_count_exceeds_base() {
    let cfg = Qwen2Config::small_test();
    let base = super::model::Qwen2Model::new(cfg.clone()).expect("Qwen2Model::new");
    let causal = super::model::Qwen2ForCausalLM::new(cfg).expect("Qwen2ForCausalLM::new");
    assert!(
        causal.parameter_count() > base.parameter_count(),
        "CausalLM should add lm_head parameters on top of base model"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Chat model tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_qwen2_chat_model_forward() {
    let cfg = Qwen2Config::small_test();
    let model = super::tasks::Qwen2ChatModel::new(cfg.clone()).expect("Qwen2ChatModel::new");
    let logits = model.forward(vec![1u32, 2]).expect("forward");
    // 2 tokens × vocab_size
    assert_eq!(logits.shape().iter().product::<usize>(), 2 * cfg.vocab_size);
}

#[test]
fn test_qwen2_chat_model_build_prompt_no_system() {
    let cfg = Qwen2Config::small_test();
    let model = super::tasks::Qwen2ChatModel::new(cfg).expect("Qwen2ChatModel::new");
    let messages = vec![("user".to_string(), "What is Rust?".to_string())];
    let prompt = model.build_prompt("", &messages);
    // No system block when system is empty
    assert!(!prompt.contains("system\n"), "empty system should not emit system block");
    assert!(prompt.contains("What is Rust?"));
}

#[test]
fn test_qwen2_chat_model_greedy_next_token_returns_valid_index() {
    let cfg = Qwen2Config::small_test();
    let model = super::tasks::Qwen2ChatModel::new(cfg.clone()).expect("Qwen2ChatModel::new");
    let logits = Tensor::from_vec(
        vec![0.1_f32; cfg.vocab_size],
        &[cfg.vocab_size],
    ).expect("tensor");
    let token = model.greedy_next_token(&logits).expect("greedy_next_token");
    assert!((token as usize) < cfg.vocab_size, "greedy token must be within vocab");
}

#[test]
fn test_qwen2_chat_model_parameter_count() {
    let cfg = Qwen2Config::small_test();
    let model = super::tasks::Qwen2ChatModel::new(cfg).expect("Qwen2ChatModel::new");
    assert!(model.parameter_count() > 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Chat format detailed tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_qwen2_chat_format_multiple_turns() {
    let messages = vec![
        ("user".to_string(), "Hello".to_string()),
        ("assistant".to_string(), "Hi there!".to_string()),
        ("user".to_string(), "How are you?".to_string()),
    ];
    let prompt = format_qwen2_chat("", &messages);
    assert!(prompt.contains("Hello"));
    assert!(prompt.contains("Hi there!"));
    assert!(prompt.contains("How are you?"));
    // Each turn should have im_start
    assert_eq!(prompt.matches("<|im_start|>").count(), 4); // 3 messages + 1 trailing assistant
}

#[test]
fn test_qwen2_chat_format_empty_messages() {
    let prompt = format_qwen2_chat("system prompt", &[]);
    // Only system block + trailing assistant
    assert!(prompt.contains("system prompt"));
    assert!(prompt.ends_with("<|im_start|>assistant\n"));
}
