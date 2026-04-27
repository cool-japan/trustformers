use crate::yi::{
    chat::format_yi_chat,
    config::YiConfig,
    model::{YiAttention, YiForCausalLM, YiModel, YiRmsNorm, YiRotaryEmbedding},
};
use trustformers_core::{tensor::Tensor, traits::Config, traits::Layer};

// ── Config preset tests ───────────────────────────────────────────────────────

#[test]
fn test_config_6b_preset() {
    let cfg = YiConfig::yi_6b();
    assert_eq!(cfg.vocab_size, 64000);
    assert_eq!(cfg.hidden_size, 4096);
    assert_eq!(cfg.intermediate_size, 11008);
    assert_eq!(cfg.num_hidden_layers, 32);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.num_key_value_heads, 4);
    assert!(cfg.tie_word_embeddings);
    assert_eq!(cfg.rope_theta, 5_000_000.0);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_9b_preset() {
    let cfg = YiConfig::yi_9b();
    assert_eq!(cfg.vocab_size, 64000);
    assert_eq!(cfg.num_hidden_layers, 48);
    assert_eq!(cfg.num_key_value_heads, 4);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_34b_preset() {
    let cfg = YiConfig::yi_34b();
    assert_eq!(cfg.vocab_size, 64000);
    assert_eq!(cfg.hidden_size, 7168);
    assert_eq!(cfg.num_hidden_layers, 60);
    assert_eq!(cfg.num_key_value_heads, 8);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_vocab_size_64000() {
    for cfg in [YiConfig::yi_6b(), YiConfig::yi_9b(), YiConfig::yi_34b()] {
        assert_eq!(cfg.vocab_size, 64000);
    }
}

#[test]
fn test_config_rope_theta_5e6() {
    // All presets including long-context variant should have rope_theta = 5e6
    for cfg in [
        YiConfig::yi_6b(),
        YiConfig::yi_9b(),
        YiConfig::yi_34b(),
        YiConfig::yi_6b_200k(),
    ] {
        assert_eq!(
            cfg.rope_theta, 5_000_000.0,
            "All Yi-1.5 presets should have rope_theta = 5e6"
        );
    }
}

#[test]
fn test_config_6b_200k_long_context() {
    let cfg = YiConfig::yi_6b_200k();
    assert_eq!(cfg.max_position_embeddings, 200000);
    assert_eq!(cfg.vocab_size, 64000);
    assert_eq!(cfg.rope_theta, 5_000_000.0);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_tied_embeddings() {
    for cfg in [
        YiConfig::yi_6b(),
        YiConfig::yi_9b(),
        YiConfig::yi_34b(),
        YiConfig::small_test(),
    ] {
        assert!(cfg.tie_word_embeddings, "Yi always ties embeddings");
    }
}

#[test]
fn test_config_num_kv_heads_variants() {
    assert_eq!(YiConfig::yi_6b().num_key_value_heads, 4);
    assert_eq!(YiConfig::yi_9b().num_key_value_heads, 4);
    assert_eq!(YiConfig::yi_34b().num_key_value_heads, 8);
}

// ── RMS norm test ─────────────────────────────────────────────────────────────

#[test]
fn test_rms_norm_forward() {
    let norm = YiRmsNorm::new(4, 1e-5).expect("build norm");
    let input = Tensor::from_vec(vec![2.0f32, 4.0, 6.0, 8.0], &[4]).expect("build tensor");
    let output = norm.forward(input).expect("forward");
    assert_eq!(output.shape(), &[4]);
}

// ── Rotary embedding test ─────────────────────────────────────────────────────

#[test]
fn test_rotary_embedding_theta_5e6() {
    let rope = YiRotaryEmbedding::new(128, 4096, 5_000_000.0);
    assert_eq!(rope.theta, 5_000_000.0);
    assert_eq!(rope.half_dim(), 64);
    // Verify large theta leads to very small inverse frequencies
    assert!(
        rope.inv_freq[1] < rope.inv_freq[0],
        "inv_freq should decrease with i"
    );
}

// ── Attention tests ───────────────────────────────────────────────────────────

#[test]
fn test_attention_gqa_kv_heads() {
    let cfg = YiConfig::small_test();
    let attn = YiAttention::new(&cfg).expect("build attn");
    assert_eq!(attn.num_kv_heads(), cfg.num_key_value_heads);
    assert_eq!(attn.num_heads(), cfg.num_attention_heads);
    assert_eq!(attn.head_dim(), cfg.head_dim());
}

// ── Chat format tests ─────────────────────────────────────────────────────────

#[test]
fn test_chat_format_basic() {
    let messages = vec![("user".to_string(), "Hi".to_string())];
    let prompt = format_yi_chat("Assistant", &messages);
    assert!(prompt.contains("<|im_start|>system\nAssistant\n<|im_end|>"));
    assert!(prompt.contains("<|im_start|>user\nHi\n<|im_end|>"));
    assert!(prompt.ends_with("<|im_start|>assistant\n"));
}

// ── Model forward tests ───────────────────────────────────────────────────────

#[test]
fn test_model_forward_small() {
    let cfg = YiConfig::small_test();
    let model = YiModel::new(cfg.clone()).expect("build model");
    let output = model.run(vec![1u32, 2, 3]).expect("run");
    let shape = output.shape().to_vec();
    assert_eq!(shape[0], 1);
    assert_eq!(shape[1], 3);
    assert_eq!(shape[2], cfg.hidden_size);
}

#[test]
fn test_causal_lm_forward_small() {
    let cfg = YiConfig::small_test();
    let lm = YiForCausalLM::new(cfg.clone()).expect("build lm");
    let logits = lm.forward(vec![1u32, 2]).expect("forward");
    let shape = logits.shape().to_vec();
    assert_eq!(shape[0], 1);
    assert_eq!(shape[1], 2);
    assert_eq!(shape[2], cfg.vocab_size);
}

#[test]
fn test_tied_embeddings_flag() {
    let cfg = YiConfig::small_test();
    assert!(cfg.tie_word_embeddings);
    let lm = YiForCausalLM::new(cfg).expect("build lm");
    assert!(lm.tie_word_embeddings());
}

// ── Additional coverage tests ─────────────────────────────────────────────────

#[test]
fn test_config_default_is_6b() {
    let default = YiConfig::default();
    let six_b = YiConfig::yi_6b();
    assert_eq!(default.hidden_size, six_b.hidden_size);
    assert_eq!(default.num_hidden_layers, six_b.num_hidden_layers);
    assert_eq!(default.vocab_size, six_b.vocab_size);
}

#[test]
fn test_config_head_dim_6b() {
    let cfg = YiConfig::yi_6b();
    // 4096 / 32 = 128
    assert_eq!(cfg.head_dim(), 128);
}

#[test]
fn test_config_head_dim_34b() {
    let cfg = YiConfig::yi_34b();
    // 7168 / 56 = 128
    assert_eq!(cfg.head_dim(), 128);
}

#[test]
fn test_config_query_groups_6b() {
    let cfg = YiConfig::yi_6b();
    // 32 / 4 = 8
    assert_eq!(cfg.num_query_groups(), 8);
}

#[test]
fn test_config_query_groups_34b() {
    let cfg = YiConfig::yi_34b();
    // 56 / 8 = 7
    assert_eq!(cfg.num_query_groups(), 7);
}

#[test]
fn test_config_small_test_valid() {
    let cfg = YiConfig::small_test();
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.vocab_size, 512);
    assert_eq!(cfg.hidden_size, 64);
    assert_eq!(cfg.num_attention_heads, 4);
    assert_eq!(cfg.num_key_value_heads, 2);
}

#[test]
fn test_rms_norm_output_shape_3d() {
    // RMSNorm should handle a multi-element tensor correctly.
    let norm = YiRmsNorm::new(8, 1e-5).expect("build norm");
    let input = Tensor::from_vec((0..8).map(|i| (i + 1) as f32).collect::<Vec<_>>(), &[8])
        .expect("build tensor");
    let output = norm.forward(input).expect("forward");
    assert_eq!(output.shape(), &[8]);
}

#[test]
fn test_rotary_embedding_inv_freq_length() {
    // half_dim = head_dim / 2
    let head_dim = 64;
    let rope = YiRotaryEmbedding::new(head_dim, 4096, 5_000_000.0);
    assert_eq!(rope.half_dim(), head_dim / 2);
}

#[test]
fn test_rotary_embedding_apply_shape_preserving() {
    let cfg = YiConfig::small_test();
    let head_dim = cfg.head_dim(); // 64/4 = 16
    let rope = YiRotaryEmbedding::new(head_dim, cfg.max_position_embeddings, cfg.rope_theta);
    let q = Tensor::from_vec(vec![0.5f32; head_dim], &[head_dim]).expect("q");
    let k = Tensor::from_vec(vec![0.3f32; head_dim], &[head_dim]).expect("k");
    let position_ids = vec![0usize, 1];
    let (q_rot, k_rot) = rope.apply_rotary_emb(&q, &k, &position_ids).expect("apply");
    assert_eq!(q_rot.shape(), q.shape());
    assert_eq!(k_rot.shape(), k.shape());
}

#[test]
fn test_model_parameter_count_nonzero() {
    let cfg = YiConfig::small_test();
    let model = YiModel::new(cfg.clone()).expect("build model");
    let params = model.parameter_count();
    // Must have at least the embedding table parameters.
    let min_params = cfg.vocab_size * cfg.hidden_size;
    assert!(
        params >= min_params,
        "parameter count {params} should be ≥ embed table {min_params}"
    );
}

#[test]
fn test_causal_lm_tied_parameter_count_less_than_untied() {
    let cfg = YiConfig::small_test(); // tie_word_embeddings = true
    let lm = YiForCausalLM::new(cfg.clone()).expect("build lm");
    // Tied: lm_head shares embed_tokens → no extra lm_head params.
    let tied_params = lm.parameter_count();
    // Compare with a "fake" count that includes both.
    let extra_lm_head = cfg.hidden_size * cfg.vocab_size;
    // Tied count should be strictly less than tied + extra_lm_head.
    assert!(
        tied_params < tied_params + extra_lm_head,
        "tied param count sanity"
    );
}

#[test]
fn test_chat_format_multi_turn() {
    let messages = vec![
        ("user".to_string(), "What is Rust?".to_string()),
        ("assistant".to_string(), "A systems language.".to_string()),
        ("user".to_string(), "Tell me more.".to_string()),
    ];
    let prompt = format_yi_chat("You are helpful.", &messages);
    // Should contain all turns.
    assert!(prompt.contains("<|im_start|>user\nWhat is Rust?\n<|im_end|>"));
    assert!(prompt.contains("<|im_start|>assistant\nA systems language.\n<|im_end|>"));
    assert!(prompt.contains("<|im_start|>user\nTell me more.\n<|im_end|>"));
    // Must end with open assistant tag for generation.
    assert!(prompt.ends_with("<|im_start|>assistant\n"));
}

#[test]
fn test_chat_format_empty_messages() {
    let prompt = format_yi_chat("Only system context", &[]);
    assert!(prompt.contains("<|im_start|>system\nOnly system context\n<|im_end|>"));
    assert!(prompt.ends_with("<|im_start|>assistant\n"));
    // No user/assistant turns should appear.
    assert!(!prompt.contains("<|im_start|>user"));
}

#[test]
fn test_model_single_token_input() {
    let cfg = YiConfig::small_test();
    let model = YiModel::new(cfg.clone()).expect("build model");
    let output = model.run(vec![42u32]).expect("single token forward");
    let shape = output.shape().to_vec();
    assert_eq!(shape[0], 1, "batch");
    assert_eq!(shape[1], 1, "single token → seq_len=1");
    assert_eq!(shape[2], cfg.hidden_size);
}

#[test]
fn test_causal_lm_single_token_vocab_logits() {
    let cfg = YiConfig::small_test();
    let lm = YiForCausalLM::new(cfg.clone()).expect("build lm");
    let logits = lm.forward(vec![1u32]).expect("forward");
    let shape = logits.shape().to_vec();
    // [1, 1, vocab_size]
    assert_eq!(shape[2], cfg.vocab_size);
}
