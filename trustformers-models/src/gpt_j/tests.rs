use super::config::GptJConfig;
use super::model::{GptJLMHeadModel, GptJModel, GptJRotaryEmbedding};
use trustformers_core::{
    tensor::Tensor,
    traits::{Config, Model, TokenizedInput},
};

// LCG random number generator (no rand dependency)
#[allow(dead_code)]
struct Lcg {
    state: u64,
}

#[allow(dead_code)]
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005u64)
            .wrapping_add(1442695040888963407u64);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next() >> 11) as f32 / (1u64 << 53) as f32
    }
}

// ─── Config Tests ─────────────────────────────────────────────────────────────

#[test]
fn test_gptj_config_default() {
    let config = GptJConfig::default();
    assert_eq!(config.vocab_size, 50400);
    assert_eq!(config.n_embd, 4096);
    assert_eq!(config.n_layer, 28);
    assert_eq!(config.n_head, 16);
    assert_eq!(config.n_positions, 2048);
    assert_eq!(config.rotary_dim, 64);
    assert_eq!(config.model_type, "gptj");
}

#[test]
fn test_gptj_config_validate_ok() {
    let config = GptJConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_gptj_config_head_not_divisible() {
    let config = GptJConfig {
        n_embd: 100,
        n_head: 7,
        ..GptJConfig::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_gptj_config_rotary_dim_too_large() {
    let config = GptJConfig {
        n_embd: 64,
        n_head: 8,      // head_dim = 8
        rotary_dim: 16, // 16 > 8 → invalid
        ..GptJConfig::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_gptj_config_6b_preset() {
    let config = GptJConfig::gpt_j_6b();
    assert_eq!(config.n_embd, 4096);
    assert_eq!(config.n_layer, 28);
    assert_eq!(config.n_head, 16);
    assert_eq!(config.rotary_dim, 64);
    assert!(config.validate().is_ok());
}

#[test]
fn test_gptj_config_from_pretrained_6b() {
    let config = GptJConfig::from_pretrained_name("EleutherAI/gpt-j-6b");
    assert_eq!(config.n_embd, 4096);
    assert!(config.validate().is_ok());
}

#[test]
fn test_gptj_config_from_pretrained_default() {
    let config = GptJConfig::from_pretrained_name("unknown-model");
    assert_eq!(config.n_embd, 4096); // Falls back to 6B
    assert!(config.validate().is_ok());
}

#[test]
fn test_gptj_config_head_dim() {
    let config = GptJConfig {
        n_embd: 64,
        n_head: 8,
        rotary_dim: 8,
        ..GptJConfig::default()
    };
    assert_eq!(config.head_dim(), 8);
}

#[test]
fn test_gptj_config_architecture_name() {
    let config = GptJConfig::default();
    assert_eq!(config.architecture(), "GPT-J");
}

// ─── RoPE Tests ───────────────────────────────────────────────────────────────

#[test]
fn test_gptj_rope_creation() {
    let rope = GptJRotaryEmbedding::new(64, 2048, 10000.0);
    assert_eq!(rope.dim, 64);
    assert_eq!(rope.max_seq_len, 2048);
    assert!((rope.base - 10000.0).abs() < 1e-5);
}

#[test]
fn test_gptj_rope_apply_empty_positions() {
    let rope = GptJRotaryEmbedding::new(8, 64, 10000.0);
    let q = Tensor::zeros(&[1, 1, 8]).expect("zeros failed");
    let k = Tensor::zeros(&[1, 1, 8]).expect("zeros failed");
    let positions: Vec<usize> = vec![];
    let result = rope.apply_rotary_emb(&q, &k, &positions);
    assert!(
        result.is_ok(),
        "Empty positions RoPE failed: {:?}",
        result.err()
    );
}

#[test]
fn test_gptj_rope_apply_single_position() {
    let rope = GptJRotaryEmbedding::new(8, 64, 10000.0);
    let q = Tensor::zeros(&[1, 1, 8]).expect("zeros failed");
    let k = Tensor::zeros(&[1, 1, 8]).expect("zeros failed");
    let positions = vec![0usize];
    let result = rope.apply_rotary_emb(&q, &k, &positions);
    assert!(
        result.is_ok(),
        "Single position RoPE failed: {:?}",
        result.err()
    );
}

#[test]
fn test_gptj_rope_apply_multiple_positions() {
    let rope = GptJRotaryEmbedding::new(16, 128, 10000.0);
    let q = Tensor::zeros(&[1, 4, 16]).expect("zeros failed");
    let k = Tensor::zeros(&[1, 4, 16]).expect("zeros failed");
    let positions = vec![0usize, 1, 2, 3];
    let result = rope.apply_rotary_emb(&q, &k, &positions);
    assert!(
        result.is_ok(),
        "Multi-position RoPE failed: {:?}",
        result.err()
    );
    if let Ok((q_out, k_out)) = result {
        let q_shape = q_out.shape();
        let k_shape = k_out.shape();
        assert_eq!(q_shape, k_shape);
    }
}

// ─── Model Construction Tests ─────────────────────────────────────────────────

fn tiny_gptj_config() -> GptJConfig {
    GptJConfig {
        vocab_size: 100,
        n_embd: 32,
        n_layer: 1,
        n_head: 4, // head_dim = 8
        n_positions: 64,
        rotary_dim: 8, // <= head_dim = 8
        activation_function: "gelu_new".to_string(),
        resid_pdrop: 0.0,
        embd_pdrop: 0.0,
        attn_pdrop: 0.0,
        layer_norm_epsilon: 1e-5,
        initializer_range: 0.02,
        use_cache: false,
        bos_token_id: 1,
        eos_token_id: 2,
        model_type: "gptj".to_string(),
    }
}

#[test]
fn test_gptj_model_construction() {
    let config = tiny_gptj_config();
    let result = GptJModel::new(config);
    assert!(
        result.is_ok(),
        "GptJModel construction failed: {:?}",
        result.err()
    );
}

#[test]
fn test_gptj_lm_head_construction() {
    let config = tiny_gptj_config();
    let result = GptJLMHeadModel::new(config);
    assert!(
        result.is_ok(),
        "GptJLMHeadModel construction failed: {:?}",
        result.err()
    );
}

#[test]
fn test_gptj_model_num_parameters() {
    let config = tiny_gptj_config();
    if let Ok(model) = GptJModel::new(config) {
        assert!(
            model.num_parameters() > 0,
            "Model should have positive parameter count"
        );
    }
}

#[test]
fn test_gptj_model_config_access() {
    let config = tiny_gptj_config();
    let vocab_size = config.vocab_size;
    if let Ok(model) = GptJModel::new(config) {
        assert_eq!(model.get_config().vocab_size, vocab_size);
    }
}

#[test]
fn test_gptj_model_forward_pass() {
    let config = tiny_gptj_config();
    if let Ok(model) = GptJModel::new(config) {
        let input = TokenizedInput {
            input_ids: vec![1u32, 2, 3, 4],
            attention_mask: vec![1u8; 4],
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        };
        let result = model.forward(input);
        assert!(
            result.is_ok(),
            "GptJModel forward pass failed: {:?}",
            result.err()
        );
    }
}

#[test]
fn test_gptj_lm_head_forward() {
    let config = tiny_gptj_config();
    if let Ok(model) = GptJLMHeadModel::new(config) {
        let input = TokenizedInput {
            input_ids: vec![1u32, 2, 3],
            attention_mask: vec![1u8; 3],
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        };
        let result = model.forward(input);
        assert!(
            result.is_ok(),
            "GptJLMHeadModel forward failed: {:?}",
            result.err()
        );
    }
}

#[test]
fn test_gptj_lm_head_output_logits_shape() {
    let config = tiny_gptj_config();
    let vocab_size = config.vocab_size;
    if let Ok(model) = GptJLMHeadModel::new(config) {
        let seq_len = 3;
        let input = TokenizedInput {
            input_ids: vec![1u32; seq_len],
            attention_mask: vec![1u8; seq_len],
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        };
        if let Ok(output) = model.forward(input) {
            let shape = output.logits.shape();
            // shape should be [batch, seq_len, vocab_size] or similar
            assert!(shape.len() >= 2, "Logits should have at least 2 dims");
            // Last dim should be vocab_size
            assert_eq!(shape[shape.len() - 1], vocab_size);
        }
    }
}

#[test]
fn test_gptj_model_single_token() {
    let config = tiny_gptj_config();
    if let Ok(model) = GptJModel::new(config) {
        let input = TokenizedInput {
            input_ids: vec![5u32],
            attention_mask: vec![1u8],
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        };
        let result = model.forward(input);
        assert!(
            result.is_ok(),
            "Single-token forward failed: {:?}",
            result.err()
        );
    }
}
