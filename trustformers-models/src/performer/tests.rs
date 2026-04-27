use super::config::PerformerConfig;
use super::model::{PerformerForMaskedLM, PerformerForSequenceClassification, PerformerModel};
use trustformers_core::traits::{Config, Model};

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
fn test_performer_config_default() {
    let config = PerformerConfig::default();
    assert_eq!(config.vocab_size, 30522);
    assert_eq!(config.hidden_size, 768);
    assert_eq!(config.num_attention_heads, 12);
    assert_eq!(config.num_random_features, 256);
    assert!(config.use_favor_plus);
    assert!(config.normalize_features);
    assert!(!config.causal_attention);
    assert_eq!(config.kernel_type, "relu");
    assert!(config.ortho_features);
}

#[test]
fn test_performer_config_validate_ok() {
    let config = PerformerConfig {
        num_random_features: 64, // <= 2 * head_dim = 2 * 64 = 128
        ..PerformerConfig::default()
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_performer_config_head_not_divisible() {
    let config = PerformerConfig {
        hidden_size: 100,
        num_attention_heads: 7,
        num_random_features: 10,
        ..PerformerConfig::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_performer_config_invalid_kernel_type() {
    let config = PerformerConfig {
        kernel_type: "quadratic".to_string(),
        num_random_features: 64,
        ..PerformerConfig::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_performer_config_too_many_random_features() {
    let config = PerformerConfig {
        hidden_size: 64,
        num_attention_heads: 8,
        // head_dim = 8, so 2 * head_dim = 16; num_random_features = 100 > 16
        num_random_features: 100,
        ..PerformerConfig::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_performer_config_valid_kernels() {
    for kernel in &["relu", "exp", "softmax+"] {
        let config = PerformerConfig {
            kernel_type: kernel.to_string(),
            num_random_features: 64,
            ..PerformerConfig::default()
        };
        assert!(
            config.validate().is_ok(),
            "Kernel {} should be valid",
            kernel
        );
    }
}

#[test]
fn test_performer_config_base_preset() {
    let config = PerformerConfig::performer_base();
    assert_eq!(config.hidden_size, 768);
    assert_eq!(config.num_random_features, 256);
}

#[test]
fn test_performer_config_large_preset() {
    let config = PerformerConfig::performer_large();
    assert_eq!(config.hidden_size, 1024);
    assert_eq!(config.num_attention_heads, 16);
    assert_eq!(config.num_random_features, 512);
}

#[test]
fn test_performer_config_causal_preset() {
    let config = PerformerConfig::performer_causal();
    assert!(config.causal_attention);
    assert_eq!(config.kernel_type, "relu");
}

#[test]
fn test_performer_config_long_preset() {
    let config = PerformerConfig::performer_long();
    assert_eq!(config.max_position_embeddings, 16384);
    assert_eq!(config.num_random_features, 512);
    assert!(config.redraw_features);
}

#[test]
fn test_performer_config_head_dim() {
    let config = PerformerConfig {
        hidden_size: 768,
        num_attention_heads: 12,
        num_random_features: 64,
        ..PerformerConfig::default()
    };
    assert_eq!(config.head_dim(), 64);
}

#[test]
fn test_performer_config_approximation_quality() {
    let config = PerformerConfig {
        hidden_size: 64,
        num_attention_heads: 8,  // head_dim = 8
        num_random_features: 16, // quality = 16/8 = 2.0
        ..PerformerConfig::default()
    };
    let quality = config.approximation_quality();
    assert!(
        (quality - 2.0).abs() < 1e-5,
        "Expected quality 2.0, got {}",
        quality
    );
}

#[test]
fn test_performer_config_is_efficient() {
    let config = PerformerConfig {
        max_position_embeddings: 512,
        num_random_features: 256, // 256 < 512 → efficient
        ..PerformerConfig::default()
    };
    assert!(config.is_efficient());

    let inefficient_config = PerformerConfig {
        max_position_embeddings: 100,
        num_random_features: 200, // 200 > 100 → not efficient
        ..PerformerConfig::default()
    };
    assert!(!inefficient_config.is_efficient());
}

#[test]
fn test_performer_config_architecture_name() {
    let config = PerformerConfig::default();
    assert_eq!(config.architecture(), "Performer");
}

// ─── Model Construction Tests ─────────────────────────────────────────────────

fn tiny_performer_config() -> PerformerConfig {
    PerformerConfig {
        vocab_size: 100,
        hidden_size: 32,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        intermediate_size: 64,
        max_position_embeddings: 64,
        num_random_features: 16,
        hidden_dropout_prob: 0.0,
        attention_probs_dropout_prob: 0.0,
        causal_attention: false,
        kernel_type: "relu".to_string(),
        redraw_features: false,
        use_favor_plus: true,
        normalize_features: true,
        ortho_features: false, // simpler for tests
        numerical_stabilizer: 1e-6,
        ..PerformerConfig::default()
    }
}

#[test]
fn test_performer_model_construction() {
    let config = tiny_performer_config();
    let result = PerformerModel::new(config);
    assert!(
        result.is_ok(),
        "PerformerModel construction failed: {:?}",
        result.err()
    );
}

#[test]
fn test_performer_model_num_parameters() {
    let config = tiny_performer_config();
    if let Ok(model) = PerformerModel::new(config) {
        assert!(model.num_parameters() > 0);
    }
}

#[test]
fn test_performer_classification_construction() {
    let config = tiny_performer_config();
    let result = PerformerForSequenceClassification::new(config, 4);
    assert!(
        result.is_ok(),
        "Classification model failed: {:?}",
        result.err()
    );
}

#[test]
fn test_performer_masked_lm_construction() {
    let config = tiny_performer_config();
    let result = PerformerForMaskedLM::new(config);
    assert!(result.is_ok(), "MaskedLM model failed: {:?}", result.err());
}

#[test]
fn test_performer_model_config_access() {
    let config = tiny_performer_config();
    let vocab_size = config.vocab_size;
    if let Ok(model) = PerformerModel::new(config) {
        assert_eq!(model.get_config().vocab_size, vocab_size);
    }
}

#[test]
fn test_performer_device_default() {
    let config = tiny_performer_config();
    if let Ok(model) = PerformerModel::new(config) {
        assert_eq!(model.device(), trustformers_core::device::Device::CPU);
    }
}

#[test]
fn test_performer_exp_kernel_construction() {
    let config = PerformerConfig {
        vocab_size: 100,
        hidden_size: 32,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        intermediate_size: 64,
        max_position_embeddings: 64,
        num_random_features: 16,
        hidden_dropout_prob: 0.0,
        attention_probs_dropout_prob: 0.0,
        kernel_type: "exp".to_string(),
        redraw_features: false,
        use_favor_plus: true,
        normalize_features: true,
        ortho_features: false,
        numerical_stabilizer: 1e-6,
        causal_attention: false,
        ..PerformerConfig::default()
    };
    let result = PerformerModel::new(config);
    assert!(
        result.is_ok(),
        "Exp kernel construction failed: {:?}",
        result.err()
    );
}
