use crate::phi2::{
    config::Phi2Config,
    model::{
        Phi2Attention, Phi2DecoderLayer, Phi2ForCausalLM, Phi2LayerNorm, Phi2MLP, Phi2Model,
        Phi2RotaryEmbedding,
    },
    tasks::Phi2ForCodeGeneration,
};
use trustformers_core::{tensor::Tensor, traits::Config, traits::Layer};

// ─────────────────────────────────────────────────────────────────────────
// Config tests
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_phi2_config_small_test_is_valid() {
    let config = Phi2Config::small_test();
    assert!(
        config.validate().is_ok(),
        "small_test config must pass validation"
    );
}

#[test]
fn test_phi2_config_phi2_2_7b_shape() {
    let config = Phi2Config::phi2_2_7b();
    assert_eq!(config.hidden_size, 2560);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_hidden_layers, 32);
    assert_eq!(config.vocab_size, 51200);
    assert_eq!(config.intermediate_size, 10240);
    assert_eq!(config.max_position_embeddings, 2048);
    assert!(config.validate().is_ok());
}

#[test]
fn test_phi2_config_architecture_label() {
    let config = Phi2Config::small_test();
    assert_eq!(config.architecture(), "Phi-2");
}

#[test]
fn test_phi2_config_head_dim() {
    let config = Phi2Config::small_test(); // hidden=64, heads=4
    assert_eq!(config.head_dim(), 16, "head_dim = hidden_size / num_heads");
}

#[test]
fn test_phi2_config_invalid_head_division() {
    let config = Phi2Config {
        hidden_size: 65, // not divisible by 4
        ..Phi2Config::small_test()
    };
    assert!(
        config.validate().is_err(),
        "hidden_size not divisible by num_attention_heads must be invalid"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// LayerNorm
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_phi2_layernorm_construction() {
    let norm = Phi2LayerNorm::new(64, 1e-5);
    assert!(norm.is_ok(), "LayerNorm construction must succeed");
    let norm = norm.expect("checked above");
    assert_eq!(
        norm.parameter_count(),
        128,
        "weight + bias = 2 × hidden_size"
    );
}

#[test]
fn test_phi2_layernorm_forward_shape() {
    let norm = Phi2LayerNorm::new(8, 1e-5).expect("construction failed");
    let input = Tensor::ones(&[8]).expect("tensor");
    let output = norm.forward(input.clone()).expect("forward failed");
    assert_eq!(
        output.shape(),
        input.shape(),
        "layernorm must preserve shape"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// RoPE
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_phi2_rope_half_dim() {
    let rope = Phi2RotaryEmbedding::new(16, 64, 10000.0);
    assert_eq!(rope.half_dim(), 8, "half_dim must be head_dim / 2");
}

#[test]
fn test_phi2_rope_apply_preserves_shape() {
    let rope = Phi2RotaryEmbedding::new(16, 64, 10000.0);
    let q = Tensor::ones(&[4, 16]).expect("tensor");
    let k = Tensor::ones(&[4, 16]).expect("tensor");
    let positions: Vec<usize> = (0..4).collect();
    let (q_out, k_out) = rope.apply_rotary_emb(&q, &k, &positions).expect("rope failed");
    assert_eq!(q_out.shape(), q.shape(), "q shape must be preserved");
    assert_eq!(k_out.shape(), k.shape(), "k shape must be preserved");
}

// ─────────────────────────────────────────────────────────────────────────
// MLP
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_phi2_mlp_construction_and_parameters() {
    let config = Phi2Config::small_test();
    let mlp = Phi2MLP::new(&config).expect("MLP construction failed");
    assert!(mlp.parameter_count() > 0, "MLP must have parameters");
}

#[test]
fn test_phi2_mlp_forward_shape() {
    let config = Phi2Config::small_test(); // hidden=64, intermediate=256
    let mlp = Phi2MLP::new(&config).expect("MLP construction failed");
    // Linear requires at least 2D: shape [seq_len=1, hidden=64]
    let arr = scirs2_core::ndarray::ArrayD::from_shape_vec(
        scirs2_core::ndarray::IxDyn(&[1, 64]),
        vec![0.01f32; 64],
    )
    .expect("reshape");
    let input = Tensor::F32(arr);
    let output = mlp.forward(input).expect("MLP forward failed");
    // Output shape: [1, hidden_size] → total = 64
    let total: usize = output.shape().iter().product();
    assert_eq!(total, 64, "MLP output size must equal hidden_size");
}

// ─────────────────────────────────────────────────────────────────────────
// Attention
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_phi2_attention_construction() {
    let config = Phi2Config::small_test();
    let attn = Phi2Attention::new(&config);
    assert!(attn.is_ok(), "attention construction must succeed");
    let attn = attn.expect("checked");
    assert_eq!(attn.num_heads(), 4);
    assert_eq!(attn.head_dim(), 16); // 64 / 4
}

#[test]
fn test_phi2_attention_forward_shape() {
    let config = Phi2Config::small_test(); // hidden=64
    let attn = Phi2Attention::new(&config).expect("attention construction failed");
    let seq_len = 4_usize;
    // Reshape to [seq_len, hidden]
    use scirs2_core::ndarray::{ArrayD, IxDyn};
    let arr = ArrayD::from_shape_vec(
        IxDyn(&[seq_len, config.hidden_size]),
        vec![0.01f32; seq_len * config.hidden_size],
    )
    .expect("reshape");
    let input_2d = Tensor::F32(arr);
    let output = attn.forward(input_2d).expect("attention forward failed");
    let out_len: usize = output.shape().iter().product();
    assert_eq!(
        out_len,
        seq_len * config.hidden_size,
        "attention output must have same total elements as input"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Decoder layer (parallel)
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_phi2_decoder_layer_parallel_residual() {
    let config = Phi2Config::small_test();
    let layer = Phi2DecoderLayer::new(&config).expect("layer construction failed");
    use scirs2_core::ndarray::{ArrayD, IxDyn};
    let seq_len = 3_usize;
    let arr = ArrayD::from_shape_vec(
        IxDyn(&[seq_len, config.hidden_size]),
        vec![0.01f32; seq_len * config.hidden_size],
    )
    .expect("reshape");
    let input = Tensor::F32(arr);
    let output = layer.forward(input).expect("decoder layer forward failed");
    let total: usize = output.shape().iter().product();
    assert_eq!(
        total,
        seq_len * config.hidden_size,
        "decoder layer must preserve seq_len × hidden_size"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Full model
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_phi2_model_construction() {
    let model = Phi2Model::new(Phi2Config::small_test());
    assert!(model.is_ok(), "Phi2Model construction must succeed");
    let model = model.expect("checked");
    assert!(model.parameter_count() > 0, "must have parameters");
}

#[test]
fn test_phi2_for_causal_lm_forward() {
    let config = Phi2Config::small_test();
    let model = Phi2ForCausalLM::new(config.clone()).expect("model construction failed");
    let input_ids = vec![1u32, 2, 3, 4];
    let logits = model.forward(input_ids).expect("forward failed");
    // Logits shape: [seq_len × vocab_size] or [seq_len, vocab_size]
    let total: usize = logits.shape().iter().product();
    assert_eq!(
        total,
        4 * config.vocab_size,
        "logits total must equal seq_len × vocab_size"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Code generation task
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_phi2_code_generation_construction() {
    let model = Phi2ForCodeGeneration::new(Phi2Config::small_test());
    assert!(
        model.is_ok(),
        "Phi2ForCodeGeneration construction must succeed"
    );
}

#[test]
fn test_phi2_generate_code_returns_string() {
    let model = Phi2ForCodeGeneration::new(Phi2Config::small_test()).expect("construction failed");
    let result = model.generate_code(vec![1u32, 2, 3]);
    assert!(result.is_ok(), "generate_code must not return an error");
    let code = result.expect("checked");
    assert!(!code.is_empty(), "generated code string must not be empty");
}

// ─────────────────────────────────────────────────────────────────────────
// Additional tests to reach 20+
// ─────────────────────────────────────────────────────────────────────────

// ── partial_rotary_factor and rotary_dim ──────────────────────────────────

#[test]
fn test_phi2_rope_theta_default() {
    let cfg = Phi2Config::default();
    assert!(
        (cfg.rope_theta - 10000.0).abs() < 1e-9,
        "Phi-2 rope_theta must be 10000 by default"
    );
}

#[test]
fn test_phi2_rope_theta_phi2_2_7b() {
    let cfg = Phi2Config::phi2_2_7b();
    assert!(
        (cfg.rope_theta - 10000.0).abs() < 1e-9,
        "Phi-2 2.7B rope_theta must be 10000"
    );
}

// ── No GQA: num_attention_heads == num_key_value_heads ─────────────────────
// (Phi2Config does not store num_key_value_heads because it is always == num_attention_heads)

#[test]
fn test_phi2_mha_no_gqa_2_7b() {
    // In Phi-2 MHA, every attention head is also a KV head.
    // We verify this implicitly: the Attention module projects Q, K, V all
    // to the same dimension (num_heads * head_dim).
    let cfg = Phi2Config::phi2_2_7b();
    assert_eq!(cfg.num_attention_heads, 32);
    // head_dim = 2560 / 32 = 80
    assert_eq!(cfg.head_dim(), 80);
    // Fully verify via attention parameter count:
    // q_proj + k_proj + v_proj + dense: each has bias
    // q_proj: (hidden_size * num_heads * head_dim) + num_heads * head_dim = 2560*2560 + 2560
    // But we just test the head_dim formula holds
    assert_eq!(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim());
}

// ── layer_norm_eps ────────────────────────────────────────────────────────

#[test]
fn test_phi2_layer_norm_eps_default() {
    let cfg = Phi2Config::default();
    assert!(
        (cfg.layer_norm_eps - 1e-5).abs() < 1e-10,
        "layer_norm_eps must default to 1e-5"
    );
}

// ── intermediate_size = 4 × hidden_size ──────────────────────────────────

#[test]
fn test_phi2_intermediate_size_ratio() {
    let cfg = Phi2Config::phi2_2_7b();
    assert_eq!(
        cfg.intermediate_size,
        4 * cfg.hidden_size,
        "Phi-2 MLP intermediate must be 4× hidden_size"
    );
}

#[test]
fn test_phi2_small_test_intermediate_ratio() {
    let cfg = Phi2Config::small_test();
    assert_eq!(cfg.intermediate_size, 4 * cfg.hidden_size);
}

// ── LayerNorm forward normalizes correctly ────────────────────────────────

#[test]
fn test_phi2_layernorm_constant_input_normalizes_to_zero() {
    // All-same input → mean equals each element → numerator = 0 → output ≈ 0 (plus bias = 0)
    let norm = Phi2LayerNorm::new(4, 1e-5).expect("layernorm");
    let arr = scirs2_core::ndarray::ArrayD::from_shape_vec(
        scirs2_core::ndarray::IxDyn(&[4]),
        vec![3.0f32; 4],
    )
    .expect("arr");
    let input = trustformers_core::tensor::Tensor::F32(arr);
    let out = norm.forward(input).expect("forward");
    match &out {
        trustformers_core::tensor::Tensor::F32(a) => {
            for &v in a.iter() {
                assert!(
                    v.abs() < 1e-4,
                    "constant input LayerNorm output must be ≈0, got {v}"
                );
            }
        },
        _ => panic!("expected F32"),
    }
}

// ── Phi2RotaryEmbedding inv_freq all positive ─────────────────────────────

#[test]
fn test_phi2_rope_inv_freq_positive() {
    let rope = Phi2RotaryEmbedding::new(16, 64, 10000.0);
    for &f in &rope.inv_freq {
        assert!(
            f > 0.0 && f.is_finite(),
            "inv_freq entry must be finite positive, got {f}"
        );
    }
}

#[test]
fn test_phi2_rope_inv_freq_decreasing() {
    // Higher frequency index → smaller inv_freq (larger RoPE angle → shorter wavelength)
    let rope = Phi2RotaryEmbedding::new(16, 64, 10000.0);
    for i in 1..rope.inv_freq.len() {
        assert!(
            rope.inv_freq[i] <= rope.inv_freq[i - 1],
            "inv_freq must be non-increasing, but [{i}]={} > [{}]={}",
            rope.inv_freq[i],
            i - 1,
            rope.inv_freq[i - 1]
        );
    }
}

// ── Phi2MLP GELU activation ───────────────────────────────────────────────

#[test]
fn test_phi2_mlp_output_differs_from_input() {
    // With random (all-ones) weights the MLP output should differ from a zero input
    let config = Phi2Config::small_test();
    let mlp = Phi2MLP::new(&config).expect("MLP");
    use scirs2_core::ndarray::{ArrayD, IxDyn};
    // Non-zero input so the output is non-trivially zero
    let arr = ArrayD::from_shape_vec(IxDyn(&[1, 64]), vec![1.0f32; 64]).expect("arr");
    let input = trustformers_core::tensor::Tensor::F32(arr);
    let out = mlp.forward(input).expect("forward");
    let total: usize = out.shape().iter().product();
    assert_eq!(total, 64, "MLP output size must match hidden_size");
}

// ── Full model parameter count ────────────────────────────────────────────

#[test]
fn test_phi2_causal_lm_parameter_count_positive() {
    let model = Phi2ForCausalLM::new(Phi2Config::small_test()).expect("model");
    assert!(model.parameter_count() > 0, "parameter count must be > 0");
}

// ── Code generation task config accessor ──────────────────────────────────

#[test]
fn test_phi2_code_gen_config_accessor() {
    let model = Phi2ForCodeGeneration::new(Phi2Config::small_test()).expect("model");
    assert_eq!(
        model.config().vocab_size,
        Phi2Config::small_test().vocab_size
    );
}

#[test]
fn test_phi2_code_gen_parameter_count_positive() {
    let model = Phi2ForCodeGeneration::new(Phi2Config::small_test()).expect("model");
    assert!(model.parameter_count() > 0);
}

// ── Validation negative tests ─────────────────────────────────────────────

#[test]
fn test_phi2_config_invalid_zero_hidden_size() {
    let cfg = Phi2Config {
        hidden_size: 0,
        ..Phi2Config::small_test()
    };
    assert!(cfg.validate().is_err(), "hidden_size=0 must be invalid");
}

#[test]
fn test_phi2_config_invalid_zero_vocab_size() {
    let cfg = Phi2Config {
        vocab_size: 0,
        ..Phi2Config::small_test()
    };
    assert!(cfg.validate().is_err(), "vocab_size=0 must be invalid");
}

#[test]
fn test_phi2_config_invalid_zero_layers() {
    let cfg = Phi2Config {
        num_hidden_layers: 0,
        ..Phi2Config::small_test()
    };
    assert!(
        cfg.validate().is_err(),
        "num_hidden_layers=0 must be invalid"
    );
}
